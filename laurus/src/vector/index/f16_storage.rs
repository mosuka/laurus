//! Float16 vector pool shared by the HNSW / Flat / IVF readers (Issue
//! [#623](https://github.com/mosuka/laurus/issues/623)).
//!
//! A pool hands out each stored row as a byte slice for the kernels in
//! [`crate::vector::core::distance_f16`]. Where the rows live depends on
//! the storage's loading mode:
//!
//! - **Heap** (`Eager`, or a backend that cannot expose its bytes): the
//!   rows are copied into one contiguous buffer at load.
//! - **Mapped** (`Lazy`): the pool keeps a clone of the segment's input
//!   and slices rows out of [`StorageInput::as_slice`] in place, so a
//!   segment costs no heap beyond the lookup tables and the page cache
//!   decides what stays resident. The input is held until the pool is
//!   dropped, which keeps the file mapped; on Windows a segment deleted
//!   by a merge stays on disk until the reader is dropped (the next open
//!   removes leftover files).
//!
//! # On-disk record
//!
//! ```text
//! [ doc_id u64 LE ][ field_id u16 LE ][ dim × binary16 LE ]
//! ```
//!
//! The rows of one segment form runs of equally spaced records: HNSW and
//! Flat write a single run, IVF one run per non-empty inverted list (each
//! list is preceded by its 4-byte length). The pool stores one
//! `(first position, byte offset)` pair per run.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::storage::StorageInput;
use crate::vector::core::distance_f16::{F16_BYTES, decode_f16_row, encode_f16_row};
use crate::vector::core::vector::Vector;

/// Bytes of one stored Float16 row.
#[inline]
pub(crate) const fn f16_row_size(dim: usize) -> usize {
    dim * F16_BYTES
}

/// Write the row of one Float16 record. The caller writes the preceding
/// `doc_id` and `field_id`.
///
/// # Arguments
///
/// * `output` - Destination positioned after the record prefix.
/// * `dim` - The field dimension.
/// * `data` - The vector's values.
/// * `scratch` - Reused encode buffer.
///
/// # Errors
///
/// [`LaurusError::InvalidOperation`] if `data.len() != dim`;
/// [`LaurusError::Io`] on a write failure.
pub(crate) fn write_f16_row<W: Write>(
    output: &mut W,
    dim: usize,
    data: &[f32],
    scratch: &mut Vec<u8>,
) -> Result<()> {
    if data.len() != dim {
        return Err(LaurusError::InvalidOperation(format!(
            "Vector dimension mismatch: expected {dim}, got {}",
            data.len()
        )));
    }
    scratch.clear();
    encode_f16_row(data, scratch);
    output.write_all(scratch)?;
    Ok(())
}

/// Read one Float16 row and decode it to f32 (writer load paths, which
/// keep their buffers in f32).
pub(crate) fn read_f16_row<R: Read>(input: &mut R, dim: usize) -> Result<Vec<f32>> {
    let mut row = vec![0u8; f16_row_size(dim)];
    input.read_exact(&mut row)?;
    let mut out = Vec::with_capacity(dim);
    decode_f16_row(&row, &mut out);
    Ok(out)
}

/// Where a pool's rows live.
enum Rows {
    /// Rows copied into one buffer, `row_size` bytes each, in position
    /// order.
    Heap(Vec<u8>),
    /// Rows read in place from the segment's input.
    Mapped {
        /// A clone of the segment's input, positioned at 0 and never
        /// moved, so `as_slice()` spans the whole file.
        input: Box<dyn StorageInput>,
        /// First pool position of each run of equally spaced records.
        run_starts: Vec<u32>,
        /// Byte offset of the first row of each run.
        run_offsets: Vec<u64>,
        /// Bytes from one record's row to the next within a run.
        stride: u64,
    },
}

impl std::fmt::Debug for Rows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rows::Heap(data) => f.debug_tuple("Heap").field(&data.len()).finish(),
            Rows::Mapped {
                run_starts, stride, ..
            } => f
                .debug_struct("Mapped")
                .field("runs", &run_starts.len())
                .field("stride", stride)
                .finish(),
        }
    }
}

/// One segment's Float16 vectors.
///
/// Built once at reader load by [`F16PoolBuilder`] and shared across
/// search threads via `Arc`. Mirrors the lookup API of
/// [`crate::vector::index::quantized_storage::QuantizedVectorPool`].
#[derive(Debug)]
pub struct F16VectorPool {
    /// Vector dimension.
    dim: usize,
    /// Bytes per row (`2 × dim`).
    row_size: usize,
    rows: Rows,
    /// Per-field `doc_id -> position`.
    field_index: HashMap<String, Arc<HashMap<u64, u32>>>,
    /// Total number of rows.
    vector_count: usize,
}

impl F16VectorPool {
    /// The vector dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of stored rows.
    #[inline]
    pub fn vector_count(&self) -> usize {
        self.vector_count
    }

    /// Whether rows are read in place from the segment's input.
    #[inline]
    pub fn is_mapped(&self) -> bool {
        matches!(self.rows, Rows::Mapped { .. })
    }

    /// The stored row at position `pos` (`2 × dim` bytes).
    ///
    /// # Panics
    ///
    /// If `pos >= vector_count` (positions come from this pool's own
    /// lookup tables).
    #[inline]
    pub fn row_at(&self, pos: u32) -> &[u8] {
        match &self.rows {
            Rows::Heap(data) => {
                let start = pos as usize * self.row_size;
                &data[start..start + self.row_size]
            }
            Rows::Mapped {
                input,
                run_starts,
                run_offsets,
                stride,
            } => {
                let run = if run_starts.len() == 1 {
                    0
                } else {
                    run_starts.partition_point(|&s| s <= pos) - 1
                };
                let start = (run_offsets[run] + (pos - run_starts[run]) as u64 * stride) as usize;
                let bytes = input
                    .as_slice()
                    .expect("mapped pools are only built over inputs that expose a slice");
                &bytes[start..start + self.row_size]
            }
        }
    }

    /// The stored row for `(doc_id, field)`, if present.
    #[inline]
    pub fn get_row(&self, doc_id: u64, field: &str) -> Option<&[u8]> {
        let pos = self.field_index.get(field)?.get(&doc_id).copied()?;
        Some(self.row_at(pos))
    }

    /// The per-field `doc_id -> position` map, shared so the search loop
    /// can hold it without a per-candidate field-name lookup.
    #[inline]
    pub fn field_position_index(&self, field: &str) -> Option<Arc<HashMap<u64, u32>>> {
        self.field_index.get(field).cloned()
    }

    /// Whether the segment holds a vector for `(doc_id, field)`.
    #[inline]
    pub fn contains(&self, doc_id: u64, field: &str) -> bool {
        self.field_index
            .get(field)
            .is_some_and(|m| m.contains_key(&doc_id))
    }

    /// Every `(doc_id, field)` key, sorted by doc id.
    pub fn keys(&self) -> Vec<(u64, String)> {
        let mut keys: Vec<(u64, String)> = self
            .field_index
            .iter()
            .flat_map(|(field, map)| map.keys().map(move |id| (*id, field.clone())))
            .collect();
        keys.sort_by_key(|(id, _)| *id);
        keys
    }

    /// Decode the vector for `(doc_id, field)` to f32. Used by
    /// `get_vector`, `include_vectors`, the fan-out rescore and merges;
    /// the search kernels read rows directly.
    pub fn decode_to_vector(&self, doc_id: u64, field: &str) -> Option<Vector> {
        let row = self.get_row(doc_id, field)?;
        let mut data = Vec::with_capacity(self.dim);
        decode_f16_row(row, &mut data);
        Some(Vector::new(data))
    }

    /// Bytes this pool keeps on the heap: the rows when they were copied,
    /// plus the run tables. Excludes the lookup maps, like
    /// `QuantizedVectorPool::heap_size`.
    pub fn heap_size(&self) -> usize {
        match &self.rows {
            Rows::Heap(data) => data.len(),
            Rows::Mapped {
                run_starts,
                run_offsets,
                ..
            } => run_starts.len() * 4 + run_offsets.len() * 8,
        }
    }

    /// Base address and stride for direct-address software prefetch, when
    /// every row lies at a fixed stride (one heap buffer or one mapped
    /// run). The address is only used as a prefetch hint.
    pub fn prefetch_base_stride(&self) -> Option<(usize, usize)> {
        match &self.rows {
            Rows::Heap(data) => Some((data.as_ptr() as usize, self.row_size)),
            Rows::Mapped {
                input,
                run_offsets,
                stride,
                ..
            } if run_offsets.len() == 1 => {
                let base = input.as_slice()?.as_ptr() as usize + run_offsets[0] as usize;
                Some((base, *stride as usize))
            }
            Rows::Mapped { .. } => None,
        }
    }
}

/// Read one byte from every page of every row, so a mapped pool's pages
/// are faulted in before the first query (searcher warmup, Issue #677).
///
/// # Returns
///
/// A checksum of the touched bytes, for the caller to keep alive with
/// `std::hint::black_box`.
pub(crate) fn touch_rows(pool: &F16VectorPool) -> u64 {
    const PAGE: usize = 4096;
    let mut acc = 0u64;
    for pos in 0..pool.vector_count() as u32 {
        let row = pool.row_at(pos);
        for b in row.iter().step_by(PAGE) {
            acc = acc.wrapping_add(*b as u64);
        }
        if let Some(last) = row.last() {
            acc = acc.wrapping_add(*last as u64);
        }
    }
    acc
}

/// Builds an [`F16VectorPool`] while a reader parses a segment's records.
///
/// The reader reads each record's prefix itself and then calls
/// [`Self::push`] with the stream positioned at the row; the builder
/// either copies the row (heap) or records its offset and skips it
/// (mapped).
pub(crate) struct F16PoolBuilder {
    dim: usize,
    row_size: usize,
    mode: BuilderMode,
    /// Positions per field id, in field-dictionary order.
    by_field: Vec<HashMap<u64, u32>>,
    count: u32,
}

enum BuilderMode {
    Heap(Vec<u8>),
    Mapped {
        input: Box<dyn StorageInput>,
        file_len: u64,
        run_starts: Vec<u32>,
        run_offsets: Vec<u64>,
        stride: u64,
        /// Where the next row would start if it continued the current run.
        next: Option<u64>,
    },
}

impl F16PoolBuilder {
    /// A builder that copies rows into a heap buffer.
    ///
    /// # Arguments
    ///
    /// * `dim` - The field dimension.
    /// * `capacity` - Expected number of rows (already bounded by the
    ///   caller against the file size).
    pub(crate) fn heap(dim: usize, capacity: usize) -> Self {
        let row_size = f16_row_size(dim);
        Self {
            dim,
            row_size,
            mode: BuilderMode::Heap(Vec::with_capacity(capacity * row_size)),
            by_field: Vec::new(),
            count: 0,
        }
    }

    /// A builder that reads rows in place from `input`, or a heap builder
    /// when `input` cannot expose its bytes as one slice.
    ///
    /// # Arguments
    ///
    /// * `dim` - The field dimension.
    /// * `capacity` - Expected number of rows, for the heap fallback.
    /// * `input` - A fresh clone of the segment's input (position 0); it is
    ///   never moved, so its slice covers the whole file.
    /// * `record_stride` - Bytes from one record's row to the next
    ///   (record prefix + row).
    ///
    /// # Errors
    ///
    /// Forwards a failure to read the input's size.
    pub(crate) fn mapped(
        dim: usize,
        capacity: usize,
        input: Box<dyn StorageInput>,
        record_stride: u64,
    ) -> Result<Self> {
        let file_len = input.size()?;
        let whole_file = input
            .as_slice()
            .is_some_and(|bytes| bytes.len() as u64 == file_len);
        if !whole_file {
            return Ok(Self::heap(dim, capacity));
        }
        Ok(Self {
            dim,
            row_size: f16_row_size(dim),
            mode: BuilderMode::Mapped {
                input,
                file_len,
                run_starts: Vec::new(),
                run_offsets: Vec::new(),
                stride: record_stride,
                next: None,
            },
            by_field: Vec::new(),
            count: 0,
        })
    }

    /// Add the record whose row starts at the current position of `reader`,
    /// leaving `reader` positioned after the row.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The record's document id.
    /// * `field_id` - The record's field id in the segment dictionary.
    /// * `reader` - The parse stream, positioned at the row.
    ///
    /// # Errors
    ///
    /// [`LaurusError::Index`] if the row runs past the end of the mapped
    /// file or the segment holds more than `u32::MAX` rows; I/O errors from
    /// `reader`.
    pub(crate) fn push<R: Read + Seek>(
        &mut self,
        doc_id: u64,
        field_id: u16,
        reader: &mut R,
    ) -> Result<()> {
        if self.count == u32::MAX {
            return Err(LaurusError::index(
                "Float16 vector segment holds more than u32::MAX rows".to_string(),
            ));
        }
        match &mut self.mode {
            BuilderMode::Heap(data) => {
                let start = data.len();
                data.resize(start + self.row_size, 0);
                reader.read_exact(&mut data[start..])?;
            }
            BuilderMode::Mapped {
                file_len,
                run_starts,
                run_offsets,
                stride,
                next,
                ..
            } => {
                let offset = reader.stream_position().map_err(LaurusError::Io)?;
                if offset + self.row_size as u64 > *file_len {
                    return Err(LaurusError::index(format!(
                        "vector segment corrupt: Float16 row at byte {offset} runs past \
                         the end of the file ({file_len} bytes)"
                    )));
                }
                if *next != Some(offset) {
                    run_starts.push(self.count);
                    run_offsets.push(offset);
                }
                *next = Some(offset + *stride);
                reader
                    .seek(SeekFrom::Current(self.row_size as i64))
                    .map_err(LaurusError::Io)?;
            }
        }
        let fid = field_id as usize;
        if self.by_field.len() <= fid {
            self.by_field.resize_with(fid + 1, HashMap::new);
        }
        self.by_field[fid].insert(doc_id, self.count);
        self.count += 1;
        Ok(())
    }

    /// Finish the pool.
    ///
    /// # Arguments
    ///
    /// * `field_dict` - The segment's field dictionary, indexed by the
    ///   field ids passed to [`Self::push`].
    pub(crate) fn build(self, field_dict: &[Arc<str>]) -> F16VectorPool {
        let rows = match self.mode {
            BuilderMode::Heap(data) => Rows::Heap(data),
            BuilderMode::Mapped {
                input,
                run_starts,
                run_offsets,
                stride,
                ..
            } => {
                if run_starts.is_empty() {
                    // No rows: nothing to map.
                    Rows::Heap(Vec::new())
                } else {
                    Rows::Mapped {
                        input,
                        run_starts,
                        run_offsets,
                        stride,
                    }
                }
            }
        };
        let field_index = self
            .by_field
            .into_iter()
            .enumerate()
            .filter(|(_, map)| !map.is_empty())
            .map(|(fid, map)| (field_dict[fid].to_string(), Arc::new(map)))
            .collect();
        F16VectorPool {
            dim: self.dim,
            row_size: self.row_size,
            rows,
            field_index,
            vector_count: self.count as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use crate::storage::Storage;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    const DIM: usize = 3;

    fn record(out: &mut Vec<u8>, doc_id: u64, fid: u16, values: &[f32]) {
        out.extend_from_slice(&doc_id.to_le_bytes());
        out.extend_from_slice(&fid.to_le_bytes());
        let mut scratch = Vec::new();
        write_f16_row(out, DIM, values, &mut scratch).unwrap();
    }

    /// `(doc_id, field_id, values)` of one test record.
    type Record = (u64, u16, [f32; DIM]);

    /// A segment-like byte layout: an odd-length preamble, then two runs
    /// of records separated by a 4-byte list length (the IVF layout).
    fn segment() -> (Vec<u8>, Vec<Record>) {
        let records = vec![
            (10, 0, [1.0, 2.0, 3.0]),
            (20, 0, [4.0, 5.0, 6.0]),
            (10, 1, [-1.0, -2.0, -3.0]),
            (30, 0, [0.5, 0.25, 0.125]),
        ];
        let mut bytes = vec![7u8; 5];
        bytes.extend_from_slice(&2u32.to_le_bytes());
        for (doc, fid, v) in &records[..2] {
            record(&mut bytes, *doc, *fid, v);
        }
        bytes.extend_from_slice(&2u32.to_le_bytes());
        for (doc, fid, v) in &records[2..] {
            record(&mut bytes, *doc, *fid, v);
        }
        (bytes, records)
    }

    fn parse(mut builder: F16PoolBuilder, bytes: &[u8]) -> F16VectorPool {
        let mut cursor = Cursor::new(bytes);
        cursor.set_position(5);
        for _ in 0..2 {
            let mut len = [0u8; 4];
            cursor.read_exact(&mut len).unwrap();
            for _ in 0..u32::from_le_bytes(len) {
                let mut id = [0u8; 8];
                cursor.read_exact(&mut id).unwrap();
                let mut fid = [0u8; 2];
                cursor.read_exact(&mut fid).unwrap();
                builder
                    .push(u64::from_le_bytes(id), u16::from_le_bytes(fid), &mut cursor)
                    .unwrap();
            }
        }
        let dict: Vec<Arc<str>> = vec![Arc::from("a"), Arc::from("b")];
        builder.build(&dict)
    }

    fn mapped_input(bytes: &[u8]) -> Box<dyn StorageInput> {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut out = storage.create_output("seg").unwrap();
        out.write_all(bytes).unwrap();
        out.close().unwrap();
        storage.open_input("seg").unwrap()
    }

    fn check(pool: &F16VectorPool, records: &[Record]) {
        assert_eq!(pool.vector_count(), records.len());
        let names = ["a", "b"];
        for (doc, fid, values) in records {
            let v = pool.decode_to_vector(*doc, names[*fid as usize]).unwrap();
            assert_eq!(v.data.as_slice(), values, "doc {doc} field {fid}");
        }
        assert!(pool.get_row(99, "a").is_none());
        assert!(pool.get_row(10, "missing").is_none());
        let idx = pool.field_position_index("a").unwrap();
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn heap_pool_returns_every_row() {
        let (bytes, records) = segment();
        let pool = parse(F16PoolBuilder::heap(DIM, 4), &bytes);
        assert!(!pool.is_mapped());
        check(&pool, &records);
    }

    #[test]
    fn mapped_pool_returns_the_same_rows_across_runs() {
        let (bytes, records) = segment();
        let record_stride = 10 + f16_row_size(DIM) as u64;
        let builder = F16PoolBuilder::mapped(DIM, 4, mapped_input(&bytes), record_stride).unwrap();
        let pool = parse(builder, &bytes);
        assert!(pool.is_mapped(), "MemoryStorage inputs expose a slice");
        check(&pool, &records);
        let heap = parse(F16PoolBuilder::heap(DIM, 4), &bytes);
        for pos in 0..4 {
            assert_eq!(pool.row_at(pos), heap.row_at(pos), "position {pos}");
        }
        // Two runs: the second list's length breaks the stride.
        assert_eq!(pool.heap_size(), 2 * 4 + 2 * 8);
        assert!(pool.prefetch_base_stride().is_none());
    }

    #[test]
    fn mapped_builder_rejects_rows_past_the_end() {
        let (bytes, _) = segment();
        let truncated = &bytes[..bytes.len() - 1];
        let record_stride = 10 + f16_row_size(DIM) as u64;
        let mut builder =
            F16PoolBuilder::mapped(DIM, 4, mapped_input(truncated), record_stride).unwrap();
        let mut cursor = Cursor::new(truncated);
        cursor.set_position(truncated.len() as u64 - f16_row_size(DIM) as u64 + 1);
        let err = builder.push(1, 0, &mut cursor).unwrap_err();
        assert!(err.to_string().contains("runs past the end"), "{err}");
    }

    #[test]
    fn read_f16_row_decodes_what_write_wrote() {
        let mut buf = Vec::new();
        let mut scratch = Vec::new();
        write_f16_row(&mut buf, DIM, &[0.5, -1.5, 2.0], &mut scratch).unwrap();
        assert_eq!(buf.len(), f16_row_size(DIM));
        let back = read_f16_row(&mut Cursor::new(buf), DIM).unwrap();
        assert_eq!(back, vec![0.5, -1.5, 2.0]);
        assert!(write_f16_row(&mut Vec::new(), DIM, &[1.0], &mut scratch).is_err());
    }
}
