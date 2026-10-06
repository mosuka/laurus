//! LMV1: the on-disk format of one multi-vector segment (Issue #1177).
//!
//! All integers are little-endian.
//!
//! | Offset | Size | Content |
//! | --- | --- | --- |
//! | 0 | 64 | Header: magic `LMV1`, version `u16`, element kind `u8`, reserved `u8`, dimension `u32`, reserved `u32`, document count `u64`, vector count `u64`, zero padding |
//! | 64 | 24 × documents | Document table, ascending by doc id: doc id `u64`, first vector `u64`, vector count `u32`, reserved `u32` |
//! | | 4 × dimension × vectors | Payload: `f32` values, one document's vectors after another |
//! | end − 16 | 16 | Footer: CRC-32 of header + table `u32`, CRC-32 of payload `u32`, magic `LMVF`, reserved `u32` |
//!
//! The payload starts at an 8-byte-aligned offset, so a memory-mapped
//! segment is read as `&[f32]` without copying. Opening a segment checks the
//! header, the exact file size and the header + table checksum; the payload
//! checksum, which covers nearly the whole file, is checked only by
//! [`SegmentReader::verify_payload`] (merges and validation), so opening a
//! multi-gigabyte segment does not read it.
//!
//! Element kind `3` is reserved for a future 1-bit kind and rejected by
//! this version. See [`crate::vector::core::multi_vector`] for the f32 /
//! f16 / int8 row layouts and codecs.

use std::borrow::Cow;
use std::io::{Read, Seek, SeekFrom, Write};

use parking_lot::Mutex;

use crate::error::{LaurusError, Result};
use crate::storage::{Storage, StorageInput, StorageOutput};
use crate::vector::core::multi_vector::{MultiVectorRows, MultiVectorStorage};

const MAGIC: [u8; 4] = *b"LMV1";
const FOOTER_MAGIC: [u8; 4] = *b"LMVF";
const VERSION: u16 = 1;

const HEADER_LEN: usize = 64;
const DOC_ENTRY_LEN: usize = 24;
const FOOTER_LEN: usize = 16;

/// One document of a segment: its id and how many vectors it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DocEntry {
    /// Internal document id.
    pub doc_id: u64,
    /// Number of token vectors.
    pub vector_count: u32,
}

/// Counts and element kind declared by a segment header.
#[derive(Debug, Clone, Copy)]
struct Header {
    kind: MultiVectorStorage,
    dimension: usize,
    doc_count: u64,
    vector_count: u64,
}

impl Header {
    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut bytes = [0u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
        bytes[6] = self.kind.tag();
        bytes[8..12].copy_from_slice(&(self.dimension as u32).to_le_bytes());
        bytes[16..24].copy_from_slice(&self.doc_count.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.vector_count.to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes[0..4] != MAGIC {
            return Err(LaurusError::IncompatibleFormat(
                "not a multi-vector segment (bad magic)".to_string(),
            ));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(LaurusError::IncompatibleFormat(format!(
                "multi-vector segment version {version} is not supported (expected {VERSION})"
            )));
        }
        let kind = MultiVectorStorage::from_tag(bytes[6]).ok_or_else(|| {
            LaurusError::IncompatibleFormat(format!(
                "multi-vector element kind {} is not supported by this version",
                bytes[6]
            ))
        })?;
        let dimension = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize;
        if dimension == 0 {
            return Err(LaurusError::index(
                "multi-vector segment declares dimension 0",
            ));
        }
        Ok(Self {
            kind,
            dimension,
            doc_count: u64::from_le_bytes(bytes[16..24].try_into().expect("8 bytes")),
            vector_count: u64::from_le_bytes(bytes[24..32].try_into().expect("8 bytes")),
        })
    }

    /// Byte offset of the payload.
    fn payload_offset(&self) -> Option<u64> {
        self.doc_count
            .checked_mul(DOC_ENTRY_LEN as u64)?
            .checked_add(HEADER_LEN as u64)
    }

    /// Byte length of the payload.
    fn payload_len(&self) -> Option<u64> {
        self.vector_count
            .checked_mul(self.kind.row_bytes(self.dimension) as u64)
    }

    /// Exact byte length of a segment with these counts.
    fn file_len(&self) -> Option<u64> {
        self.payload_offset()?
            .checked_add(self.payload_len()?)?
            .checked_add(FOOTER_LEN as u64)
    }
}

/// Receives one document's vectors while a segment is written.
pub(crate) struct PayloadSink<'a> {
    output: &'a mut Box<dyn StorageOutput>,
    hasher: &'a mut crc32fast::Hasher,
    buffer: &'a mut Vec<u8>,
    kind: MultiVectorStorage,
    dimension: usize,
    /// Rows still expected for the current document.
    remaining: usize,
}

impl PayloadSink<'_> {
    /// Encode and append `values` (whole vectors, row-major) to the
    /// current document.
    ///
    /// # Errors
    ///
    /// Returns an error when `values` is not a whole number of rows, when
    /// it holds more rows than the document's table entry declared, or
    /// when writing fails.
    pub(crate) fn write(&mut self, values: &[f32]) -> Result<()> {
        if !values.len().is_multiple_of(self.dimension) {
            return Err(LaurusError::internal(
                "multi-vector payload is not a whole number of rows",
            ));
        }
        let rows = values.len() / self.dimension;
        if rows > self.remaining {
            return Err(LaurusError::internal(
                "multi-vector payload is longer than its table entry declares",
            ));
        }
        self.buffer.clear();
        for row in values.chunks_exact(self.dimension) {
            self.kind.encode_row(row, self.buffer);
        }
        self.hasher.update(self.buffer);
        self.output.write_all(self.buffer)?;
        self.remaining -= rows;
        Ok(())
    }

    /// Append already-encoded row bytes verbatim (the merge byte-copy
    /// fast path): `bytes.len()` must be a whole number of
    /// `self.kind.row_bytes(self.dimension)`-sized rows.
    ///
    /// # Errors
    ///
    /// Returns an error when `bytes` is not a whole number of rows, when
    /// it holds more rows than the document's table entry declared, or
    /// when writing fails.
    pub(crate) fn write_raw(&mut self, bytes: &[u8]) -> Result<()> {
        let row_bytes = self.kind.row_bytes(self.dimension);
        if !bytes.len().is_multiple_of(row_bytes) {
            return Err(LaurusError::internal(
                "multi-vector payload is not a whole number of rows",
            ));
        }
        let rows = bytes.len() / row_bytes;
        if rows > self.remaining {
            return Err(LaurusError::internal(
                "multi-vector payload is longer than its table entry declares",
            ));
        }
        self.hasher.update(bytes);
        self.output.write_all(bytes)?;
        self.remaining -= rows;
        Ok(())
    }
}

/// Write a segment atomically: to `{file_name}.tmp`, fsynced, then renamed.
///
/// `fill` is called once per entry of `docs`, in order, and must write
/// exactly that document's `vector_count × dimension` values.
///
/// # Arguments
///
/// * `storage` - Storage to write into.
/// * `file_name` - Final file name (e.g. `segment_000003.mv`).
/// * `kind` - On-disk element kind every vector is encoded as.
/// * `dimension` - Dimension of every vector.
/// * `docs` - The documents, strictly ascending by doc id, each with at
///   least one vector.
/// * `fill` - Writes the payload of the `i`-th document.
///
/// # Errors
///
/// Returns an error when `docs` is not strictly ascending, declares an
/// empty document, `fill` writes the wrong number of values, or I/O fails.
/// The temporary file is removed on failure.
pub(crate) fn write_segment(
    storage: &dyn Storage,
    file_name: &str,
    kind: MultiVectorStorage,
    dimension: usize,
    docs: &[DocEntry],
    fill: impl FnMut(usize, &mut PayloadSink<'_>) -> Result<()>,
) -> Result<()> {
    let tmp = format!("{file_name}.tmp");
    match write_segment_to(storage, &tmp, kind, dimension, docs, fill) {
        Ok(()) => storage.rename_file(&tmp, file_name),
        Err(e) => {
            if storage.file_exists(&tmp) {
                let _ = storage.delete_file(&tmp);
            }
            Err(e)
        }
    }
}

fn write_segment_to(
    storage: &dyn Storage,
    file_name: &str,
    kind: MultiVectorStorage,
    dimension: usize,
    docs: &[DocEntry],
    mut fill: impl FnMut(usize, &mut PayloadSink<'_>) -> Result<()>,
) -> Result<()> {
    if dimension == 0 {
        return Err(LaurusError::internal(
            "multi-vector segment dimension must be greater than 0",
        ));
    }
    if docs.windows(2).any(|w| w[0].doc_id >= w[1].doc_id) {
        return Err(LaurusError::internal(
            "multi-vector segment documents must be strictly ascending by doc id",
        ));
    }
    if docs.iter().any(|d| d.vector_count == 0) {
        return Err(LaurusError::internal(
            "multi-vector segment documents must hold at least one vector",
        ));
    }

    let header = Header {
        kind,
        dimension,
        doc_count: docs.len() as u64,
        vector_count: docs.iter().map(|d| u64::from(d.vector_count)).sum(),
    };
    let mut meta = Vec::with_capacity(HEADER_LEN + docs.len() * DOC_ENTRY_LEN);
    meta.extend_from_slice(&header.encode());
    let mut first_vector = 0u64;
    for doc in docs {
        meta.extend_from_slice(&doc.doc_id.to_le_bytes());
        meta.extend_from_slice(&first_vector.to_le_bytes());
        meta.extend_from_slice(&doc.vector_count.to_le_bytes());
        meta.extend_from_slice(&0u32.to_le_bytes());
        first_vector += u64::from(doc.vector_count);
    }
    let meta_crc = crc32fast::hash(&meta);

    let mut output = storage.create_output(file_name)?;
    output.write_all(&meta)?;

    let mut payload_hasher = crc32fast::Hasher::new();
    let mut buffer = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        let mut sink = PayloadSink {
            output: &mut output,
            hasher: &mut payload_hasher,
            buffer: &mut buffer,
            kind,
            dimension,
            remaining: doc.vector_count as usize,
        };
        fill(i, &mut sink)?;
        if sink.remaining != 0 {
            return Err(LaurusError::internal(format!(
                "multi-vector payload for doc {} is shorter than its table entry declares",
                doc.doc_id
            )));
        }
    }

    let mut footer = [0u8; FOOTER_LEN];
    footer[0..4].copy_from_slice(&meta_crc.to_le_bytes());
    footer[4..8].copy_from_slice(&payload_hasher.finalize().to_le_bytes());
    footer[8..12].copy_from_slice(&FOOTER_MAGIC);
    output.write_all(&footer)?;
    output.flush_and_sync()?;
    output.close()
}

/// Read access to one sealed multi-vector segment.
#[derive(Debug)]
pub(crate) struct SegmentReader {
    kind: MultiVectorStorage,
    dimension: usize,
    vector_count: u64,
    doc_ids: Vec<u64>,
    /// `(first vector, vector count)` per entry of `doc_ids`.
    spans: Vec<(u64, u32)>,
    payload_offset: u64,
    payload_crc: u32,
    /// Zero-copy view of the whole file. This input is never read or
    /// seeked, so `as_slice` keeps returning the file from offset 0.
    mapped: Option<Box<dyn StorageInput>>,
    /// Seekable handle used when the storage offers no zero-copy view.
    stream: Option<Mutex<Box<dyn StorageInput>>>,
}

impl SegmentReader {
    /// Open the segment `file_name`.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::IncompatibleFormat`] for a file that is not an
    /// LMV1 segment or uses an unsupported version / element kind, and an
    /// index error when the size, the document table or the header + table
    /// checksum is inconsistent.
    pub(crate) fn open(storage: &dyn Storage, file_name: &str) -> Result<Self> {
        let mut input = storage.open_input(file_name)?;
        let file_len = input.size()?;
        if file_len < (HEADER_LEN + FOOTER_LEN) as u64 {
            return Err(corrupt(file_name, "file is shorter than header + footer"));
        }

        if let Some(bytes) = input.as_slice() {
            let header = Header::decode(&bytes[..HEADER_LEN])?;
            check_len(file_name, &header, file_len)?;
            let table_end = header.payload_offset().expect("checked by check_len") as usize;
            let footer = &bytes[bytes.len() - FOOTER_LEN..];
            let table = parse_meta(file_name, &header, &bytes[..table_end], footer)?;
            return Ok(Self {
                kind: header.kind,
                dimension: header.dimension,
                vector_count: header.vector_count,
                doc_ids: table.doc_ids,
                spans: table.spans,
                payload_offset: table_end as u64,
                payload_crc: table.payload_crc,
                mapped: Some(input),
                stream: None,
            });
        }

        let mut header_bytes = [0u8; HEADER_LEN];
        input.read_exact(&mut header_bytes)?;
        let header = Header::decode(&header_bytes)?;
        check_len(file_name, &header, file_len)?;
        let table_end = header.payload_offset().expect("checked by check_len") as usize;
        let mut meta = vec![0u8; table_end];
        meta[..HEADER_LEN].copy_from_slice(&header_bytes);
        input.read_exact(&mut meta[HEADER_LEN..])?;
        let mut footer = [0u8; FOOTER_LEN];
        input.seek(SeekFrom::Start(file_len - FOOTER_LEN as u64))?;
        input.read_exact(&mut footer)?;
        let table = parse_meta(file_name, &header, &meta, &footer)?;
        Ok(Self {
            kind: header.kind,
            dimension: header.dimension,
            vector_count: header.vector_count,
            doc_ids: table.doc_ids,
            spans: table.spans,
            payload_offset: table_end as u64,
            payload_crc: table.payload_crc,
            mapped: None,
            stream: Some(Mutex::new(input)),
        })
    }

    /// Dimension of every vector.
    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    /// Document ids, ascending.
    pub(crate) fn doc_ids(&self) -> &[u64] {
        &self.doc_ids
    }

    /// The documents of this segment, ascending by doc id.
    pub(crate) fn entries(&self) -> impl Iterator<Item = DocEntry> + '_ {
        self.doc_ids
            .iter()
            .zip(&self.spans)
            .map(|(&doc_id, &(_, vector_count))| DocEntry {
                doc_id,
                vector_count,
            })
    }

    /// The still-encoded rows of `doc_id`, or `None` when this segment
    /// does not hold it.
    ///
    /// The returned bytes are borrowed without copying when the storage
    /// offers a zero-copy view; read into a new buffer otherwise. Used
    /// both for scoring (via [`MultiVectorRows::to_f32`]) and for a
    /// same-kind merge byte copy (via the raw bytes it wraps).
    ///
    /// # Errors
    ///
    /// Returns an error when reading from storage fails.
    pub(crate) fn rows(&self, doc_id: u64) -> Result<Option<MultiVectorRows<'_>>> {
        let Ok(index) = self.doc_ids.binary_search(&doc_id) else {
            return Ok(None);
        };
        let (first, count) = self.spans[index];
        let row = self.kind.row_bytes(self.dimension) as u64;
        let start = self.payload_offset + first * row;
        let len = (u64::from(count) * row) as usize;

        if let Some(input) = &self.mapped {
            let bytes = input
                .as_slice()
                .ok_or_else(|| LaurusError::internal("zero-copy view disappeared"))?;
            let bytes = &bytes[start as usize..start as usize + len];
            return Ok(Some(MultiVectorRows::new(
                self.kind,
                self.dimension,
                Cow::Borrowed(bytes),
            )));
        }

        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| LaurusError::internal("multi-vector segment has no input"))?;
        let mut input = stream.lock();
        input.seek(SeekFrom::Start(start))?;
        let mut bytes = vec![0u8; len];
        input.read_exact(&mut bytes)?;
        Ok(Some(MultiVectorRows::new(
            self.kind,
            self.dimension,
            Cow::Owned(bytes),
        )))
    }

    /// Check the payload checksum, which opening does not verify.
    ///
    /// # Errors
    ///
    /// Returns an index error when the payload does not match its checksum,
    /// or an I/O error when reading fails.
    pub(crate) fn verify_payload(&self) -> Result<()> {
        let payload_len = (self.vector_count * self.kind.row_bytes(self.dimension) as u64) as usize;
        let start = self.payload_offset as usize;
        let actual = if let Some(input) = &self.mapped {
            let bytes = input
                .as_slice()
                .ok_or_else(|| LaurusError::internal("zero-copy view disappeared"))?;
            crc32fast::hash(&bytes[start..start + payload_len])
        } else {
            let stream = self
                .stream
                .as_ref()
                .ok_or_else(|| LaurusError::internal("multi-vector segment has no input"))?;
            let mut input = stream.lock();
            input.seek(SeekFrom::Start(self.payload_offset))?;
            let mut hasher = crc32fast::Hasher::new();
            let mut remaining = payload_len;
            let mut chunk = vec![0u8; 1 << 16];
            while remaining > 0 {
                let n = remaining.min(chunk.len());
                input.read_exact(&mut chunk[..n])?;
                hasher.update(&chunk[..n]);
                remaining -= n;
            }
            hasher.finalize()
        };
        if actual != self.payload_crc {
            return Err(LaurusError::index(
                "multi-vector segment payload checksum mismatch",
            ));
        }
        Ok(())
    }
}

fn corrupt(file_name: &str, reason: &str) -> LaurusError {
    LaurusError::index(format!(
        "multi-vector segment '{file_name}' is corrupted: {reason}"
    ))
}

/// The file must be exactly as long as its header declares, which also
/// bounds every allocation sized from the header.
fn check_len(file_name: &str, header: &Header, file_len: u64) -> Result<()> {
    match header.file_len() {
        Some(expected) if expected == file_len => Ok(()),
        Some(expected) => Err(corrupt(
            file_name,
            &format!("expected {expected} bytes, found {file_len}"),
        )),
        None => Err(corrupt(file_name, "declared counts overflow")),
    }
}

/// The decoded document table of a segment, with its payload checksum.
struct DocTable {
    doc_ids: Vec<u64>,
    /// `(first vector, vector count)` per entry of `doc_ids`.
    spans: Vec<(u64, u32)>,
    payload_crc: u32,
}

/// Verify the footer and the header + table checksum, then decode the table.
fn parse_meta(file_name: &str, header: &Header, meta: &[u8], footer: &[u8]) -> Result<DocTable> {
    if footer[8..12] != FOOTER_MAGIC {
        return Err(corrupt(file_name, "bad footer magic"));
    }
    let meta_crc = u32::from_le_bytes(footer[0..4].try_into().expect("4 bytes"));
    let payload_crc = u32::from_le_bytes(footer[4..8].try_into().expect("4 bytes"));
    if crc32fast::hash(meta) != meta_crc {
        return Err(corrupt(
            file_name,
            "header / document table checksum mismatch",
        ));
    }

    let doc_count = header.doc_count as usize;
    let mut doc_ids = Vec::with_capacity(doc_count);
    let mut spans = Vec::with_capacity(doc_count);
    let mut next_vector = 0u64;
    for entry in meta[HEADER_LEN..].as_chunks::<DOC_ENTRY_LEN>().0 {
        let doc_id = u64::from_le_bytes(entry[0..8].try_into().expect("8 bytes"));
        let first = u64::from_le_bytes(entry[8..16].try_into().expect("8 bytes"));
        let count = u32::from_le_bytes(entry[16..20].try_into().expect("4 bytes"));
        if doc_ids.last().is_some_and(|&last| last >= doc_id) {
            return Err(corrupt(file_name, "document table is not ascending"));
        }
        if first != next_vector || count == 0 {
            return Err(corrupt(
                file_name,
                "document table spans are not contiguous",
            ));
        }
        next_vector += u64::from(count);
        doc_ids.push(doc_id);
        spans.push((first, count));
    }
    if next_vector != header.vector_count {
        return Err(corrupt(
            file_name,
            "document table does not cover the declared vector count",
        ));
    }
    Ok(DocTable {
        doc_ids,
        spans,
        payload_crc,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::storage::file::{FileStorage, FileStorageConfig};
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn memory() -> Arc<dyn Storage> {
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
    }

    fn file(dir: &tempfile::TempDir, use_mmap: bool) -> Arc<dyn Storage> {
        let mut config = FileStorageConfig::new(dir.path());
        config.use_mmap = use_mmap;
        Arc::new(FileStorage::new(dir.path(), config).unwrap())
    }

    /// Two documents, dimension 3: doc 4 holds two vectors, doc 9 one.
    fn sample() -> Vec<(u64, Vec<Vec<f32>>)> {
        vec![
            (4, vec![vec![1.0, 2.0, 3.0], vec![-1.0, 0.5, 0.25]]),
            (9, vec![vec![7.0, 8.0, 9.0]]),
        ]
    }

    fn write(
        storage: &dyn Storage,
        name: &str,
        kind: MultiVectorStorage,
        dim: usize,
        docs: &[(u64, Vec<Vec<f32>>)],
    ) {
        let entries: Vec<DocEntry> = docs
            .iter()
            .map(|(doc_id, vectors)| DocEntry {
                doc_id: *doc_id,
                vector_count: vectors.len() as u32,
            })
            .collect();
        write_segment(storage, name, kind, dim, &entries, |i, sink| {
            for v in &docs[i].1 {
                sink.write(v)?;
            }
            Ok(())
        })
        .unwrap();
    }

    fn read_all(storage: &dyn Storage, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        storage
            .open_input(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    fn overwrite(storage: &dyn Storage, name: &str, bytes: &[u8]) {
        let mut output = storage.create_output(name).unwrap();
        output.write_all(bytes).unwrap();
        output.close().unwrap();
    }

    /// Per-element tolerance for `vectors()`/`to_f32()`, exact for `F32`.
    fn tolerance(kind: MultiVectorStorage) -> f32 {
        match kind {
            MultiVectorStorage::F32 => 0.0,
            MultiVectorStorage::F16 => 2e-3,
            MultiVectorStorage::Int8 => 5e-2,
        }
    }

    /// Test-only convenience wrapping [`SegmentReader::rows`] +
    /// [`MultiVectorRows::to_f32`], mirroring the removed
    /// `SegmentReader::vectors` that only tests still needed.
    fn vectors(reader: &SegmentReader, doc_id: u64) -> Option<Cow<'_, [f32]>> {
        reader.rows(doc_id).unwrap().map(|rows| rows.to_f32())
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= tol,
                "{actual:?} vs {expected:?} (tol {tol})"
            );
        }
    }

    fn assert_round_trip(storage: &dyn Storage, kind: MultiVectorStorage) {
        write(storage, "seg.mv", kind, 3, &sample());
        let reader = SegmentReader::open(storage, "seg.mv").unwrap();
        assert_eq!(reader.rows(4).unwrap().unwrap().kind(), kind);
        assert_eq!(reader.dimension(), 3);
        assert_eq!(reader.doc_ids(), &[4, 9]);
        assert_eq!(
            reader.entries().collect::<Vec<_>>(),
            vec![
                DocEntry {
                    doc_id: 4,
                    vector_count: 2
                },
                DocEntry {
                    doc_id: 9,
                    vector_count: 1
                },
            ]
        );
        let tol = tolerance(kind);
        assert_close(
            &vectors(&reader, 4).unwrap(),
            &[1.0, 2.0, 3.0, -1.0, 0.5, 0.25],
            tol,
        );
        assert_close(&vectors(&reader, 9).unwrap(), &[7.0, 8.0, 9.0], tol);
        assert!(vectors(&reader, 5).is_none());
        reader.verify_payload().unwrap();
        assert!(!storage.file_exists("seg.mv.tmp"));
    }

    #[test]
    fn test_round_trip_in_memory_for_every_storage_kind() {
        for kind in [
            MultiVectorStorage::F32,
            MultiVectorStorage::F16,
            MultiVectorStorage::Int8,
        ] {
            assert_round_trip(memory().as_ref(), kind);
        }
    }

    #[test]
    fn test_round_trip_on_file_with_mmap_is_zero_copy_only_for_f32() {
        for kind in [
            MultiVectorStorage::F32,
            MultiVectorStorage::F16,
            MultiVectorStorage::Int8,
        ] {
            let dir = tempfile::TempDir::new().unwrap();
            let storage = file(&dir, true);
            assert_round_trip(storage.as_ref(), kind);
            // A memory map is page-aligned and the payload offset is a
            // multiple of 8, so an `F32` row set is borrowed, not copied.
            // `F16`/`Int8` rows can never be reinterpreted as `&[f32]`.
            let reader = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap();
            let rows = reader.rows(4).unwrap().unwrap();
            assert_eq!(rows.as_f32().is_some(), kind == MultiVectorStorage::F32);
        }
    }

    #[test]
    fn test_round_trip_on_file_without_mmap() {
        let dir = tempfile::TempDir::new().unwrap();
        assert_round_trip(file(&dir, false).as_ref(), MultiVectorStorage::F32);
    }

    #[test]
    fn test_unaligned_f32_bytes_are_decoded() {
        let values = [1.5f32, -2.0, 0.125];
        let mut bytes = vec![0u8];
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let unaligned = Cow::Borrowed(&bytes[1..]);
        let rows = MultiVectorRows::new(MultiVectorStorage::F32, 3, unaligned);
        // At most one of the two offsets can be 4-byte aligned, but
        // `to_f32` decodes correctly either way.
        assert_eq!(&*rows.to_f32(), &values);
    }

    #[test]
    fn test_empty_segment_round_trips() {
        let storage = memory();
        write(
            storage.as_ref(),
            "empty.mv",
            MultiVectorStorage::F32,
            4,
            &[],
        );
        let reader = SegmentReader::open(storage.as_ref(), "empty.mv").unwrap();
        assert!(reader.doc_ids().is_empty());
        assert!(vectors(&reader, 1).is_none());
        reader.verify_payload().unwrap();
    }

    #[test]
    fn test_table_corruption_is_detected_on_open() {
        let storage = memory();
        write(
            storage.as_ref(),
            "seg.mv",
            MultiVectorStorage::F32,
            3,
            &sample(),
        );
        let mut bytes = read_all(storage.as_ref(), "seg.mv");
        bytes[HEADER_LEN] ^= 0x01; // first doc id
        overwrite(storage.as_ref(), "seg.mv", &bytes);
        let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
    }

    /// Byte 6 holds the element kind; at dimension 2 an `F16` row set
    /// (`2 * dim` = 4 bytes/row) and an `Int8` one (`dim + 2` = 4
    /// bytes/row) are the same file length, so only this checksum tells
    /// them apart.
    #[test]
    fn test_element_kind_tamper_is_caught_by_the_table_checksum_even_at_matching_file_length() {
        let storage = memory();
        write(
            storage.as_ref(),
            "seg.mv",
            MultiVectorStorage::Int8,
            2,
            &[(1, vec![vec![1.0, 2.0]])],
        );
        let mut bytes = read_all(storage.as_ref(), "seg.mv");
        assert_eq!(bytes[6], MultiVectorStorage::Int8.tag());
        bytes[6] = MultiVectorStorage::F16.tag();
        overwrite(storage.as_ref(), "seg.mv", &bytes);
        let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
    }

    #[test]
    fn test_payload_corruption_is_detected_by_verify() {
        let storage = memory();
        write(
            storage.as_ref(),
            "seg.mv",
            MultiVectorStorage::F32,
            3,
            &sample(),
        );
        let mut bytes = read_all(storage.as_ref(), "seg.mv");
        let payload_start = HEADER_LEN + 2 * DOC_ENTRY_LEN;
        bytes[payload_start] ^= 0x01;
        overwrite(storage.as_ref(), "seg.mv", &bytes);
        let reader = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap();
        let err = reader.verify_payload().unwrap_err();
        assert!(err.to_string().contains("payload checksum"), "{err}");
    }

    #[test]
    fn test_truncated_file_is_rejected() {
        let storage = memory();
        write(
            storage.as_ref(),
            "seg.mv",
            MultiVectorStorage::F32,
            3,
            &sample(),
        );
        let bytes = read_all(storage.as_ref(), "seg.mv");
        overwrite(storage.as_ref(), "seg.mv", &bytes[..bytes.len() - 4]);
        let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
        assert!(err.to_string().contains("expected"), "{err}");
    }

    #[test]
    fn test_reserved_and_out_of_range_element_kinds_are_incompatible() {
        let storage = memory();
        write(
            storage.as_ref(),
            "seg.mv",
            MultiVectorStorage::F32,
            3,
            &sample(),
        );
        let original = read_all(storage.as_ref(), "seg.mv");

        for bad_kind in [3u8, 4, 255] {
            let mut bytes = original.clone();
            bytes[6] = bad_kind;
            overwrite(storage.as_ref(), "seg.mv", &bytes);
            let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
            assert!(matches!(err, LaurusError::IncompatibleFormat(_)), "{err}");
        }
    }

    #[test]
    fn test_bad_magic_is_incompatible() {
        let storage = memory();
        write(
            storage.as_ref(),
            "seg.mv",
            MultiVectorStorage::F32,
            3,
            &sample(),
        );
        let mut bytes = read_all(storage.as_ref(), "seg.mv");
        bytes[0] = b'X';
        overwrite(storage.as_ref(), "seg.mv", &bytes);
        let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
        assert!(matches!(err, LaurusError::IncompatibleFormat(_)), "{err}");
    }

    #[test]
    fn test_writer_rejects_bad_input_and_cleans_up() {
        let storage = memory();
        let unsorted = [
            DocEntry {
                doc_id: 2,
                vector_count: 1,
            },
            DocEntry {
                doc_id: 1,
                vector_count: 1,
            },
        ];
        assert!(
            write_segment(
                storage.as_ref(),
                "a.mv",
                MultiVectorStorage::F32,
                2,
                &unsorted,
                |_, _| Ok(())
            )
            .is_err()
        );

        let one = [DocEntry {
            doc_id: 1,
            vector_count: 2,
        }];
        let short = write_segment(
            storage.as_ref(),
            "b.mv",
            MultiVectorStorage::F32,
            2,
            &one,
            |_, sink| sink.write(&[1.0, 2.0]),
        );
        assert!(short.unwrap_err().to_string().contains("shorter"));
        let long = write_segment(
            storage.as_ref(),
            "c.mv",
            MultiVectorStorage::F32,
            2,
            &one,
            |_, sink| sink.write(&[1.0; 6]),
        );
        assert!(long.unwrap_err().to_string().contains("longer"));

        for name in ["a.mv", "b.mv", "c.mv"] {
            assert!(!storage.file_exists(name));
            assert!(!storage.file_exists(&format!("{name}.tmp")));
        }
    }

    #[test]
    fn test_writer_rejects_a_partial_row() {
        let storage = memory();
        let one = [DocEntry {
            doc_id: 1,
            vector_count: 2,
        }];
        // Dimension 2, but 3 values: not a whole number of rows.
        let err = write_segment(
            storage.as_ref(),
            "partial.mv",
            MultiVectorStorage::F32,
            2,
            &one,
            |_, sink| sink.write(&[1.0, 2.0, 3.0]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("whole number of rows"), "{err}");
    }
}
