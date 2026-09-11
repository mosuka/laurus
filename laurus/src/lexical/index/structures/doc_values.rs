//! DocValues implementation for efficient field access during sorting and aggregations.
//!
//! DocValues are column-oriented storage for field values, optimized for:
//! - Sorting search results by field values
//! - Faceting and aggregations
//! - Field-based scoring
//!
//! Unlike stored fields (row-oriented), DocValues store values in a columnar format
//! where accessing all values of a single field is very efficient.
//!
//! ## On-disk layout
//!
//! ```text
//! "DVFF"(4B) | version(2B) | num_fields(u32 LE) | {
//!     name_len(u32 LE) | name | num_values(u64 LE) | data_len(u64 LE) | rkyv payload
//! } x num_fields
//! ```
//!
//! [`DocValuesReader::load`] (Issue #1047 Phase 2) reads only the header and
//! each field's directory entry (name, offset, length); the rkyv payload
//! itself is skipped over via `seek` and deserialized lazily, on first
//! access, by [`DocValuesReader::materialize`]. A query that sorts or
//! facets on one field no longer pays to deserialize every other stored
//! field's DocValues column.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Arc, RwLock};

use crate::error::{LaurusError, Result};
use crate::lexical::core::field::FieldValue;
use crate::storage::Storage;
use crate::util::alloc_bounds::{checked_capacity, checked_len};

/// DocValues file extension
const DOC_VALUES_EXTENSION: &str = ".dv";

/// Minimum bytes one field's directory record can occupy on disk: a 4-byte
/// name length + a (possibly empty) name + an 8-byte value count + an
/// 8-byte payload length. Used to bound a header-declared `num_fields`
/// against the file's true size before it drives a `BTreeMap` build-out
/// (Issue #1047).
const MIN_FIELD_RECORD_SIZE: u64 = 4 + 8 + 8;

/// DocValues format for a single field, materialized in memory.
/// Stores a mapping from document ID to field value.
#[derive(Debug, Clone)]
pub struct FieldDocValues {
    /// Field name
    pub field_name: String,
    /// Mapping from doc_id to field value
    values: ahash::AHashMap<u64, FieldValue>,
}

impl FieldDocValues {
    /// Create a new FieldDocValues
    pub fn new(field_name: String) -> Self {
        FieldDocValues {
            field_name,
            values: ahash::AHashMap::new(),
        }
    }

    /// Set a value for a document
    pub fn set(&mut self, doc_id: u64, value: FieldValue) {
        self.values.insert(doc_id, value);
    }

    /// Get a value for a document
    pub fn get(&self, doc_id: u64) -> Option<&FieldValue> {
        self.values.get(&doc_id)
    }

    /// Get the number of values
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Writer for DocValues
pub struct DocValuesWriter {
    /// Storage for writing DocValues
    storage: Arc<dyn Storage>,
    /// Segment name
    segment_name: String,
    /// Field DocValues being built (field_name -> FieldDocValues).
    ///
    /// `BTreeMap`, not `HashMap`: [`Self::write_to_output`] iterates this
    /// in field-name order, so two writers fed the same documents in a
    /// different field-insertion order still produce byte-identical `.dv`
    /// files for identical content (Issue #1047).
    fields: BTreeMap<String, FieldDocValues>,
}

impl DocValuesWriter {
    /// Create a new DocValuesWriter
    pub fn new(storage: Arc<dyn Storage>, segment_name: String) -> Self {
        DocValuesWriter {
            storage,
            segment_name,
            fields: BTreeMap::new(),
        }
    }

    /// Add a field value for a document
    pub fn add_value(&mut self, doc_id: u64, field_name: &str, value: FieldValue) {
        self.fields
            .entry(field_name.to_string())
            .or_insert_with(|| FieldDocValues::new(field_name.to_string()))
            .set(doc_id, value);
    }

    /// Write DocValues to storage under this writer's configured segment name.
    pub fn write(&self) -> Result<()> {
        self.write_to(&self.segment_name)
    }

    /// Write DocValues to storage under an explicit `segment_name`.
    ///
    /// Used by the segment merge path (Issue #753), which accumulates values in
    /// a writer constructed with a placeholder name and then flushes them to the
    /// final merged segment's name. The normal flush path calls [`Self::write`],
    /// which delegates here with the writer's own `segment_name`.
    ///
    /// # Arguments
    ///
    /// * `segment_name` - Segment name the `.dv` file is written under.
    pub fn write_to(&self, segment_name: &str) -> Result<()> {
        let dv_filename = format!("{}{}", segment_name, DOC_VALUES_EXTENSION);
        let mut output = self.storage.create_output(&dv_filename)?;
        self.write_to_output(&mut output)?;
        output.flush()?;
        Ok(())
    }

    /// Write the DocValues payload to an already-open output (#554).
    ///
    /// The format is position-independent, so it serializes identically
    /// into a loose `.dv` file or a compound-container part. The output is
    /// neither flushed nor closed here — the caller owns its lifecycle.
    ///
    /// # Arguments
    ///
    /// * `output` - Destination for the DVFF payload.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or a write fails.
    pub fn write_to_output(&self, output: &mut dyn std::io::Write) -> Result<()> {
        // Write magic number and version
        output.write_all(b"DVFF")?; // DocValues File Format
        output.write_all(&[1u8, 0u8])?; // Version 1.0

        // Write number of fields
        let num_fields = self.fields.len() as u32;
        output.write_all(&num_fields.to_le_bytes())?;

        // Write each field's DocValues, in field-name order (`fields` is a
        // `BTreeMap`).
        for (field_name, field_dv) in &self.fields {
            // Write field name length and name
            let name_bytes = field_name.as_bytes();
            output.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            output.write_all(name_bytes)?;

            // Write number of values
            let num_values = field_dv.values.len() as u64;
            output.write_all(&num_values.to_le_bytes())?;

            // `values` is an `AHashMap`, whose iteration order is not
            // stable across insertion patterns -- sort by doc_id so
            // identical content always serializes to identical bytes
            // (Issue #1047), not just identical field order.
            let mut values_vec: Vec<(u64, FieldValue)> = field_dv
                .values
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .collect();
            values_vec.sort_by_key(|(doc_id, _)| *doc_id);

            let serialized = rkyv::to_bytes::<rkyv::rancor::Error>(&values_vec)
                .map_err(|e| LaurusError::Index(format!("Failed to serialize DocValues: {}", e)))?;

            output.write_all(&(serialized.len() as u64).to_le_bytes())?;
            output.write_all(&serialized)?;
        }

        Ok(())
    }
}

/// One field's on-disk location within a `.dv` file's directory, resolved
/// once at [`DocValuesReader::load`] time. The payload itself is read
/// lazily, on first access, via [`DocValuesReader::materialize`].
#[derive(Debug, Clone, Copy)]
struct FieldRecord {
    /// Byte offset of the field's rkyv payload from the start of the file.
    offset: u64,
    /// Byte length of the field's rkyv payload.
    len: u64,
}

/// Reader for DocValues.
///
/// [`Self::load`] reads only the file's directory (each field's name and
/// on-disk offset/length); no field's `doc_id -> value` payload is
/// deserialized until [`Self::get_value`] or [`Self::materialize`] first
/// asks for it, and the result is cached for later calls (Issue #1047
/// Phase 2). [`Self::has_field`] and [`Self::field_names`] never touch the
/// payload at all -- they answer straight from the directory.
///
/// Holds `storage` + `file_name` rather than an open file handle or mmap,
/// mirroring
/// [`BKDReader`](crate::lexical::index::structures::bkd_tree::BKDReader):
/// each access re-opens the file, so nothing here holds a handle across
/// the segment's lifetime that would block a concurrent merge's
/// `delete_segment_files` (notably on Windows).
#[derive(Debug)]
pub struct DocValuesReader {
    storage: Arc<dyn Storage>,
    file_name: String,
    /// field_name -> on-disk location, in file order.
    directory: BTreeMap<String, FieldRecord>,
    /// Fields materialized so far.
    cache: RwLock<HashMap<String, Arc<FieldDocValues>>>,
}

impl DocValuesReader {
    /// Load a `.dv` file's directory from storage.
    ///
    /// Only the header and each field's directory entry are read here;
    /// see [`Self::materialize`] for the lazy payload read. A missing
    /// `.dv` file is not an error -- it loads as an empty reader (no
    /// segment has ever needed DocValues), so a miss on this field stays
    /// O(1).
    pub fn load(storage: Arc<dyn Storage>, segment_name: &str) -> Result<Self> {
        let file_name = format!("{}{}", segment_name, DOC_VALUES_EXTENSION);

        // Try to open the DocValues file
        let mut input = match storage.open_input(&file_name) {
            Ok(input) => input,
            Err(_) => {
                // If DocValues file doesn't exist, return an empty reader.
                return Ok(DocValuesReader {
                    storage,
                    file_name,
                    directory: BTreeMap::new(),
                    cache: RwLock::new(HashMap::new()),
                });
            }
        };

        // Ground truth for the bounds checks below: a header-declared
        // count or length that this file cannot physically back is
        // corruption, not a value to allocate for (Issue #1047, same
        // technique as Issue #806's vector-segment bounds checks).
        let file_size = input.size()?;

        // Read and verify magic number
        let mut magic = [0u8; 4];
        input.read_exact(&mut magic)?;
        if &magic != b"DVFF" {
            return Err(LaurusError::Index(
                "Invalid DocValues file format".to_string(),
            ));
        }

        // Read version
        let mut version = [0u8; 2];
        input.read_exact(&mut version)?;
        if version[0] != 1 {
            return Err(LaurusError::Index(format!(
                "Unsupported DocValues version: {}.{}",
                version[0], version[1]
            )));
        }

        // Read number of fields
        let mut num_fields_bytes = [0u8; 4];
        input.read_exact(&mut num_fields_bytes)?;
        let num_fields = u32::from_le_bytes(num_fields_bytes);

        let available = file_size.saturating_sub(input.stream_position()?);
        let num_fields = checked_capacity(
            num_fields as usize,
            MIN_FIELD_RECORD_SIZE,
            available,
            "num_fields",
        )?;

        let mut directory = BTreeMap::new();

        // Read each field's directory entry
        for _ in 0..num_fields {
            // Read field name
            let mut name_len_bytes = [0u8; 4];
            input.read_exact(&mut name_len_bytes)?;
            let name_len = u32::from_le_bytes(name_len_bytes) as usize;

            let available = file_size.saturating_sub(input.stream_position()?);
            let name_len = checked_len(name_len, available, "field name length")?;

            let mut name_bytes = vec![0u8; name_len];
            input.read_exact(&mut name_bytes)?;
            let field_name = String::from_utf8(name_bytes)
                .map_err(|e| LaurusError::Index(format!("Invalid field name: {}", e)))?;

            // Number of values is informational only (kept for on-disk
            // format stability) -- the payload length below is what
            // actually bounds the read.
            let mut num_values_bytes = [0u8; 8];
            input.read_exact(&mut num_values_bytes)?;

            // Read serialized values' length
            let mut data_len_bytes = [0u8; 8];
            input.read_exact(&mut data_len_bytes)?;
            let data_len = u64::from_le_bytes(data_len_bytes);

            let available = file_size.saturating_sub(input.stream_position()?);
            let data_len = checked_len(data_len as usize, available, "field data length")? as u64;

            let offset = input.stream_position()?;
            directory.insert(
                field_name,
                FieldRecord {
                    offset,
                    len: data_len,
                },
            );

            // Skip the payload -- deserialized lazily on first access.
            let skip = i64::try_from(data_len).map_err(|_| {
                LaurusError::Index(
                    "field data length overflows a seek offset — segment is corrupted".to_string(),
                )
            })?;
            input.seek(SeekFrom::Current(skip))?;
        }

        Ok(DocValuesReader {
            storage,
            file_name,
            directory,
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// Materialize `field_name`'s `doc_id -> value` map, reading and
    /// deserializing its payload on first access and caching the result
    /// for later calls. `Ok(None)` means this segment's `.dv` file has no
    /// column for `field_name` at all; a `Err` means the column exists
    /// but its payload could not be read or deserialized.
    fn materialize(&self, field_name: &str) -> Result<Option<Arc<FieldDocValues>>> {
        let record = match self.directory.get(field_name) {
            Some(record) => *record,
            None => return Ok(None),
        };

        if let Some(fdv) = self.cache.read().unwrap().get(field_name) {
            return Ok(Some(fdv.clone()));
        }

        // Re-open per access rather than holding a handle for the
        // reader's lifetime -- mirrors `BKDReader` (see the struct doc
        // comment above).
        let mut input = self.storage.open_input(&self.file_name)?;
        input.seek(SeekFrom::Start(record.offset))?;
        let mut data = vec![0u8; record.len as usize];
        input.read_exact(&mut data)?;

        let values_vec: Vec<(u64, FieldValue)> =
            rkyv::from_bytes::<Vec<(u64, FieldValue)>, rkyv::rancor::Error>(&data).map_err(
                |e| LaurusError::Index(format!("Failed to deserialize DocValues: {}", e)),
            )?;
        let values = values_vec.into_iter().collect();
        let fdv = Arc::new(FieldDocValues {
            field_name: field_name.to_string(),
            values,
        });

        self.cache
            .write()
            .unwrap()
            .insert(field_name.to_string(), fdv.clone());
        Ok(Some(fdv))
    }

    /// Get a value for a document and field, materializing the field's
    /// column on first access.
    ///
    /// `Ok(None)` covers two distinct cases callers must not conflate:
    /// this segment has no DocValues column for `field_name` at all, or
    /// the column exists but has no value for this particular `doc_id`.
    /// Use [`Self::has_field`] first if the distinction matters (Issue
    /// #1047 — an index-wide "some segment has this column" does not
    /// guarantee this segment does).
    pub fn get_value(&self, field_name: &str, doc_id: u64) -> Result<Option<FieldValue>> {
        Ok(self
            .materialize(field_name)?
            .and_then(|fdv| fdv.get(doc_id).cloned()))
    }

    /// Check if a field has a DocValues column in this segment. A pure
    /// directory lookup -- never materializes the payload.
    pub fn has_field(&self, field_name: &str) -> bool {
        self.directory.contains_key(field_name)
    }

    /// Get all field names with a DocValues column in this segment. A
    /// pure directory lookup -- never materializes any payload.
    pub fn field_names(&self) -> Vec<String> {
        self.directory.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;

    #[test]
    fn test_field_doc_values() {
        let mut dv = FieldDocValues::new("test_field".to_string());

        // Set some values
        dv.set(0, crate::data::DataValue::Int64(100));
        dv.set(1, crate::data::DataValue::Text("hello".to_string()));
        dv.set(5, crate::data::DataValue::Float64(3.15));

        // Get values
        assert_eq!(dv.get(0), Some(&crate::data::DataValue::Int64(100)));
        assert_eq!(
            dv.get(1),
            Some(&crate::data::DataValue::Text("hello".to_string()))
        );
        assert_eq!(dv.get(2), None);
        assert_eq!(dv.get(5), Some(&crate::data::DataValue::Float64(3.15)));
    }

    #[test]
    fn test_doc_values_write_read() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let segment_name = "segment_0".to_string();

        // Write DocValues
        {
            let mut writer = DocValuesWriter::new(storage.clone(), segment_name.clone());
            writer.add_value(0, "year", crate::data::DataValue::Int64(2023));
            writer.add_value(1, "year", crate::data::DataValue::Int64(2024));
            writer.add_value(0, "rating", crate::data::DataValue::Float64(4.5));
            writer.add_value(1, "rating", crate::data::DataValue::Float64(5.0));
            writer.write().unwrap();
        }

        // Read DocValues
        {
            let reader = DocValuesReader::load(storage.clone(), &segment_name).unwrap();
            assert!(reader.has_field("year"));
            assert!(reader.has_field("rating"));
            assert!(!reader.has_field("unknown"));

            assert_eq!(
                reader.get_value("year", 0).unwrap(),
                Some(crate::data::DataValue::Int64(2023))
            );
            assert_eq!(
                reader.get_value("year", 1).unwrap(),
                Some(crate::data::DataValue::Int64(2024))
            );
            assert_eq!(
                reader.get_value("rating", 0).unwrap(),
                Some(crate::data::DataValue::Float64(4.5))
            );
            assert_eq!(
                reader.get_value("rating", 1).unwrap(),
                Some(crate::data::DataValue::Float64(5.0))
            );
        }
    }

    #[test]
    fn field_names_and_has_field_are_directory_only_and_need_no_materialization() {
        // A corrupted payload for a field that is never queried through
        // `get_value` must not matter to `has_field`/`field_names` -- both
        // stop at the directory.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let segment_name = "segment_dir_only".to_string();
        {
            let mut writer = DocValuesWriter::new(storage.clone(), segment_name.clone());
            writer.add_value(0, "a", crate::data::DataValue::Int64(1));
            writer.add_value(0, "z", crate::data::DataValue::Int64(2));
            writer.write().unwrap();
        }

        let reader = DocValuesReader::load(storage.clone(), &segment_name).unwrap();
        assert_eq!(reader.field_names(), vec!["a".to_string(), "z".to_string()]);
        assert!(reader.has_field("a"));
        assert!(reader.has_field("z"));
        assert!(!reader.has_field("missing"));
    }

    #[test]
    fn get_value_on_a_missing_field_is_ok_none_not_an_error() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let segment_name = "segment_missing_field".to_string();
        {
            let mut writer = DocValuesWriter::new(storage.clone(), segment_name.clone());
            writer.add_value(0, "present", crate::data::DataValue::Int64(1));
            writer.write().unwrap();
        }

        let reader = DocValuesReader::load(storage.clone(), &segment_name).unwrap();
        assert_eq!(reader.get_value("absent", 0).unwrap(), None);
    }

    #[test]
    fn written_bytes_are_deterministic_regardless_of_insertion_order() {
        // Same (doc_id, field, value) triples fed in two different orders
        // must produce byte-identical `.dv` files (Issue #1047): the
        // `BTreeMap` field ordering and the doc_id-sorted `values_vec`
        // together remove every source of nondeterminism `AHashMap`
        // iteration would otherwise introduce.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut forward = DocValuesWriter::new(storage.clone(), "fwd".to_string());
        forward.add_value(0, "alpha", crate::data::DataValue::Int64(1));
        forward.add_value(1, "alpha", crate::data::DataValue::Int64(2));
        forward.add_value(0, "beta", crate::data::DataValue::Int64(3));
        forward.add_value(1, "beta", crate::data::DataValue::Int64(4));

        let mut reverse = DocValuesWriter::new(storage.clone(), "rev".to_string());
        reverse.add_value(1, "beta", crate::data::DataValue::Int64(4));
        reverse.add_value(0, "beta", crate::data::DataValue::Int64(3));
        reverse.add_value(1, "alpha", crate::data::DataValue::Int64(2));
        reverse.add_value(0, "alpha", crate::data::DataValue::Int64(1));

        let mut forward_bytes = Vec::new();
        forward.write_to_output(&mut forward_bytes).unwrap();
        let mut reverse_bytes = Vec::new();
        reverse.write_to_output(&mut reverse_bytes).unwrap();

        assert_eq!(forward_bytes, reverse_bytes);
    }

    #[test]
    fn load_rejects_a_num_fields_the_file_cannot_back() {
        // A flipped `num_fields` byte must surface a clean "corrupted"
        // error before any per-field allocation, not read past EOF or
        // abort the process (Issue #1047, same technique as #806).
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let file_name = "corrupt.dv";
        {
            let mut output = storage.create_output(file_name).unwrap();
            output.write_all(b"DVFF").unwrap();
            output.write_all(&[1u8, 0u8]).unwrap();
            // Declare an impossible number of fields for a file this
            // short.
            output.write_all(&u32::MAX.to_le_bytes()).unwrap();
            output.flush().unwrap();
        }

        let err = DocValuesReader::load(storage, "corrupt").unwrap_err();
        match err {
            LaurusError::Index(msg) => assert!(
                msg.contains("corrupted"),
                "expected a corruption message, got: {msg}"
            ),
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_a_field_data_len_the_file_cannot_back() {
        // A valid, small `num_fields` (1) paired with a field whose
        // declared payload length exceeds what remains in the file must
        // also be rejected as corruption.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let file_name = "corrupt_len.dv";
        {
            let mut output = storage.create_output(file_name).unwrap();
            output.write_all(b"DVFF").unwrap();
            output.write_all(&[1u8, 0u8]).unwrap();
            output.write_all(&1u32.to_le_bytes()).unwrap(); // num_fields = 1
            let name = b"f";
            output
                .write_all(&(name.len() as u32).to_le_bytes())
                .unwrap();
            output.write_all(name).unwrap();
            output.write_all(&1u64.to_le_bytes()).unwrap(); // num_values (unused)
            // Declare a payload far larger than any bytes that follow.
            output.write_all(&(1u64 << 40).to_le_bytes()).unwrap();
            output.flush().unwrap();
        }

        let err = DocValuesReader::load(storage, "corrupt_len").unwrap_err();
        match err {
            LaurusError::Index(msg) => assert!(
                msg.contains("corrupted"),
                "expected a corruption message, got: {msg}"
            ),
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    /// Issue #1047 Phase 2: querying one field must not read another
    /// field's payload bytes off disk. Wraps storage in a byte-counting
    /// shim and proves that materializing a small field reads far fewer
    /// bytes than a much larger sibling field's payload occupies.
    #[test]
    fn materializing_one_field_does_not_read_another_fields_payload() {
        use std::sync::atomic::{AtomicU64, Ordering};

        #[derive(Debug)]
        struct CountingInput {
            inner: Box<dyn crate::storage::StorageInput>,
            bytes_read: Arc<AtomicU64>,
        }
        impl Read for CountingInput {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = self.inner.read(buf)?;
                self.bytes_read.fetch_add(n as u64, Ordering::Relaxed);
                Ok(n)
            }
        }
        impl Seek for CountingInput {
            fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
                self.inner.seek(pos)
            }
        }
        impl crate::storage::StorageInput for CountingInput {
            fn size(&self) -> Result<u64> {
                self.inner.size()
            }
            fn clone_input(&self) -> Result<Box<dyn crate::storage::StorageInput>> {
                self.inner.clone_input()
            }
            fn close(&mut self) -> Result<()> {
                self.inner.close()
            }
        }

        #[derive(Debug)]
        struct CountingStorage {
            inner: Arc<dyn Storage>,
            bytes_read: Arc<AtomicU64>,
        }
        impl Storage for CountingStorage {
            fn open_input(&self, name: &str) -> Result<Box<dyn crate::storage::StorageInput>> {
                Ok(Box::new(CountingInput {
                    inner: self.inner.open_input(name)?,
                    bytes_read: self.bytes_read.clone(),
                }))
            }
            fn create_output(&self, name: &str) -> Result<Box<dyn crate::storage::StorageOutput>> {
                self.inner.create_output(name)
            }
            fn create_output_append(
                &self,
                name: &str,
            ) -> Result<Box<dyn crate::storage::StorageOutput>> {
                self.inner.create_output_append(name)
            }
            fn delete_file(&self, name: &str) -> Result<()> {
                self.inner.delete_file(name)
            }
            fn file_exists(&self, name: &str) -> bool {
                self.inner.file_exists(name)
            }
            fn list_files(&self) -> Result<Vec<String>> {
                self.inner.list_files()
            }
            fn file_size(&self, name: &str) -> Result<u64> {
                self.inner.file_size(name)
            }
            fn rename_file(&self, from: &str, to: &str) -> Result<()> {
                self.inner.rename_file(from, to)
            }
            fn metadata(&self, name: &str) -> Result<crate::storage::FileMetadata> {
                self.inner.metadata(name)
            }
            fn create_temp_output(
                &self,
                prefix: &str,
            ) -> Result<(String, Box<dyn crate::storage::StorageOutput>)> {
                self.inner.create_temp_output(prefix)
            }
            fn sync(&self) -> Result<()> {
                self.inner.sync()
            }
            fn close(&mut self) -> Result<()> {
                Ok(())
            }
        }

        let bytes_read = Arc::new(AtomicU64::new(0));
        let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let storage: Arc<dyn Storage> = Arc::new(CountingStorage {
            inner: inner.clone(),
            bytes_read: bytes_read.clone(),
        });

        let segment_name = "segment_lazy".to_string();
        let huge_text: String = "x".repeat(1_000_000);
        {
            let mut writer = DocValuesWriter::new(inner.clone(), segment_name.clone());
            writer.add_value(0, "small", crate::data::DataValue::Int64(42));
            writer.add_value(0, "huge", crate::data::DataValue::Text(huge_text));
            writer.write().unwrap();
        }

        let reader = DocValuesReader::load(storage, &segment_name).unwrap();
        let bytes_for_load = bytes_read.load(Ordering::Relaxed);
        assert!(
            bytes_for_load < 1_000,
            "load() read {bytes_for_load} bytes -- it must read only the header \
             and directory, not the huge field's ~1MB payload"
        );

        // Reset so what follows isolates the cost of materializing "small".
        bytes_read.store(0, Ordering::Relaxed);
        let value = reader.get_value("small", 0).unwrap();
        assert_eq!(value, Some(crate::data::DataValue::Int64(42)));
        let bytes_for_small_field = bytes_read.load(Ordering::Relaxed);
        assert!(
            bytes_for_small_field < 1_000,
            "materializing the small field read {bytes_for_small_field} bytes -- \
             the huge field's ~1MB payload must not have been touched"
        );

        // The huge field's payload is read once it is actually requested
        // -- laziness defers the cost, it does not eliminate it.
        let huge_value = reader.get_value("huge", 0).unwrap();
        assert!(matches!(huge_value, Some(crate::data::DataValue::Text(_))));
        let bytes_for_huge_field = bytes_read.load(Ordering::Relaxed) - bytes_for_small_field;
        assert!(
            bytes_for_huge_field > 500_000,
            "materializing the huge field only read {bytes_for_huge_field} bytes -- \
             expected roughly its ~1MB payload"
        );
    }
}
