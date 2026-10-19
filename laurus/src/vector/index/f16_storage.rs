//! Float16 vector pool shared by the HNSW / Flat / IVF readers (Issue
//! [#623](https://github.com/mosuka/laurus/issues/623)).
//!
//! A pool hands out each stored row as a byte slice for the kernels in
//! [`crate::vector::core::distance_f16`]. Where the rows live depends on
//! the storage's loading mode:
//!
//! - **Heap** (`Eager`, or a backend that cannot expose its bytes): the
//!   rows are copied into one contiguous buffer at load.
//! - **Mapped** (`Lazy`): the rows are read in place from the segment's
//!   input through `MappedRows` (see the crate-internal `row_pool` module
//!   for the run table and the mapping's lifetime).
//!
//! # On-disk record
//!
//! ```text
//! [ doc_id u64 LE ][ field_id u16 LE ][ dim × binary16 LE ]
//! ```

use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::storage::StorageInput;
use crate::vector::core::distance_f16::{F16_BYTES, decode_f16_row, encode_f16_row};
use crate::vector::core::vector::Vector;
use crate::vector::index::row_pool::{
    FieldPositions, MappedRows, MappedRowsBuilder, MappedRowsView,
};

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
    Mapped(MappedRows),
}

impl std::fmt::Debug for Rows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rows::Heap(data) => f.debug_tuple("Heap").field(&data.len()).finish(),
            Rows::Mapped(rows) => f.debug_tuple("Mapped").field(rows).finish(),
        }
    }
}

/// The rows of a heap pool, borrowed for one search
/// ([`F16VectorPool::rows`]).
#[derive(Clone, Copy)]
pub(crate) struct F16HeapRows<'a> {
    data: &'a [u8],
    row_size: usize,
}

impl<'a> F16HeapRows<'a> {
    /// The row at `pos`.
    ///
    /// # Panics
    ///
    /// If `pos` is not a position of the pool.
    #[inline]
    pub(crate) fn row_at(&self, pos: u32) -> &'a [u8] {
        let start = pos as usize * self.row_size;
        &self.data[start..start + self.row_size]
    }

    /// Base address and stride of the rows, for direct-address software
    /// prefetch.
    pub(crate) fn base_stride(&self) -> (usize, usize) {
        (self.data.as_ptr() as usize, self.row_size)
    }
}

/// The rows of a mapped pool, borrowed for one search
/// ([`F16VectorPool::rows`]).
#[derive(Clone, Copy)]
pub(crate) struct F16MappedRows<'a> {
    rows: MappedRowsView<'a>,
    row_size: usize,
}

impl<'a> F16MappedRows<'a> {
    /// The row at `pos`.
    ///
    /// # Panics
    ///
    /// If `pos` is not a position of the pool.
    #[inline]
    pub(crate) fn row_at(&self, pos: u32) -> &'a [u8] {
        self.rows.row(pos, self.row_size)
    }

    /// Base address and stride for direct-address software prefetch, when
    /// the rows form one run.
    pub(crate) fn base_stride(&self) -> Option<(usize, usize)> {
        self.rows.base_stride()
    }
}

/// A pool's rows borrowed for one search: [`F16HeapRows`] or
/// [`F16MappedRows`].
#[derive(Clone, Copy)]
pub(crate) enum F16Rows<'a> {
    Heap(F16HeapRows<'a>),
    Mapped(F16MappedRows<'a>),
}

impl<'a> F16Rows<'a> {
    /// The row at `pos` (see [`F16VectorPool::row_at`]).
    #[inline]
    pub(crate) fn row_at(&self, pos: u32) -> &'a [u8] {
        match self {
            F16Rows::Heap(rows) => rows.row_at(pos),
            F16Rows::Mapped(rows) => rows.row_at(pos),
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
        self.rows().row_at(pos)
    }

    /// The rows borrowed for one search. A hot loop takes this once and
    /// reads every candidate through it: the storage kind is resolved up
    /// front, and a mapped pool resolves its file slice once instead of
    /// per candidate.
    #[inline]
    pub(crate) fn rows(&self) -> F16Rows<'_> {
        match &self.rows {
            Rows::Heap(data) => F16Rows::Heap(F16HeapRows {
                data,
                row_size: self.row_size,
            }),
            Rows::Mapped(rows) => F16Rows::Mapped(F16MappedRows {
                rows: rows.view(),
                row_size: self.row_size,
            }),
        }
    }

    /// The stored row for `(doc_id, field)`, if present.
    #[inline]
    pub fn get_row(&self, doc_id: u64, field: &str) -> Option<&[u8]> {
        Some(self.row_at(self.position(doc_id, field)?))
    }

    /// The position of `(doc_id, field)` in this pool, if present.
    #[inline]
    pub fn position(&self, doc_id: u64, field: &str) -> Option<u32> {
        self.field_index.get(field)?.get(&doc_id).copied()
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
            Rows::Mapped(rows) => rows.heap_size(),
        }
    }

    /// Base address and stride for direct-address software prefetch, when
    /// every row lies at a fixed stride (one heap buffer or one mapped
    /// run). The address is only used as a prefetch hint.
    pub fn prefetch_base_stride(&self) -> Option<(usize, usize)> {
        match self.rows() {
            F16Rows::Heap(rows) => Some(rows.base_stride()),
            F16Rows::Mapped(rows) => rows.base_stride(),
        }
    }

    /// Read one byte from every page of every mapped row, so the pages are
    /// faulted in before the first query (searcher warmup, Issue #677).
    ///
    /// # Returns
    ///
    /// A checksum of the touched bytes, for the caller to keep alive with
    /// `std::hint::black_box`; `None` for a heap pool, which needs no
    /// warming.
    pub(crate) fn touch_mapped_rows(&self) -> Option<u64> {
        match &self.rows {
            Rows::Heap(_) => None,
            Rows::Mapped(rows) => Some(rows.touch(self.vector_count as u32, self.row_size)),
        }
    }
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
    positions: FieldPositions,
    count: u32,
}

enum BuilderMode {
    Heap(Vec<u8>),
    Mapped(MappedRowsBuilder),
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
            positions: FieldPositions::default(),
            count: 0,
        }
    }

    /// A builder that reads rows in place from `input`, or a heap builder
    /// when there is no input or it cannot expose its bytes as one slice.
    ///
    /// # Arguments
    ///
    /// * `dim` - The field dimension.
    /// * `capacity` - Expected number of rows, for the heap fallback.
    /// * `input` - A fresh clone of the segment's input (position 0), or
    ///   `None` when the input could not be cloned.
    ///
    /// # Errors
    ///
    /// Forwards a failure to read the input's size.
    pub(crate) fn mapped(
        dim: usize,
        capacity: usize,
        input: Option<Box<dyn StorageInput>>,
    ) -> Result<Self> {
        let row_size = f16_row_size(dim);
        let Some(rows) = input
            .map(|input| MappedRowsBuilder::new(input, row_size))
            .transpose()?
            .flatten()
        else {
            return Ok(Self::heap(dim, capacity));
        };
        Ok(Self {
            dim,
            row_size,
            mode: BuilderMode::Mapped(rows),
            positions: FieldPositions::default(),
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
            BuilderMode::Mapped(rows) => rows.push(self.count, reader)?,
        }
        self.positions.insert(field_id, doc_id, self.count);
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
            // No rows: nothing to map.
            BuilderMode::Mapped(rows) => rows
                .build()
                .map_or_else(|| Rows::Heap(Vec::new()), Rows::Mapped),
        };
        F16VectorPool {
            dim: self.dim,
            row_size: self.row_size,
            rows,
            field_index: self.positions.into_index(field_dict),
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
        let builder = F16PoolBuilder::mapped(DIM, 4, Some(mapped_input(&bytes))).unwrap();
        let pool = parse(builder, &bytes);
        assert!(pool.is_mapped(), "MemoryStorage inputs expose a slice");
        check(&pool, &records);
        let heap = parse(F16PoolBuilder::heap(DIM, 4), &bytes);
        for pos in 0..4 {
            assert_eq!(pool.row_at(pos), heap.row_at(pos), "position {pos}");
        }
        // Two runs: the second list's length breaks the stride. Each run
        // keeps a start, an offset and a stride.
        assert_eq!(pool.heap_size(), 2 * (4 + 8 + 8));
        assert!(pool.prefetch_base_stride().is_none());
        assert!(pool.touch_mapped_rows().is_some());
        assert!(heap.touch_mapped_rows().is_none());
    }

    #[test]
    fn mapped_builder_without_an_input_copies_the_rows() {
        let (bytes, records) = segment();
        let pool = parse(F16PoolBuilder::mapped(DIM, 4, None).unwrap(), &bytes);
        assert!(!pool.is_mapped());
        check(&pool, &records);
    }

    #[test]
    fn mapped_builder_rejects_rows_past_the_end() {
        let (bytes, _) = segment();
        let truncated = &bytes[..bytes.len() - 1];
        let mut builder = F16PoolBuilder::mapped(DIM, 4, Some(mapped_input(truncated))).unwrap();
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
