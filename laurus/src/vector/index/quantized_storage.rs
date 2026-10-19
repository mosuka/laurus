//! int8 vector pool used by the HNSW / Flat / IVF readers for
//! `Scalar8Bit` segments (Issue #481 Stage 1, Step 6 onward).
//!
//! The search hot loop pulls `(int8 slice, QuantizedVectorMeta)` directly
//! out of this pool and feeds it to
//! [`crate::vector::core::distance_quantized::distance_quantized`]. Where
//! the rows live depends on the storage's loading mode:
//!
//! - **Heap** (`Eager`, or a backend that cannot expose its bytes): the
//!   rows are copied at load into a Structure-of-Arrays (SoA) layout
//!   (Issue #837): one contiguous int8 buffer with each row zero-padded to
//!   a 32-byte boundary, plus two parallel meta arrays (`sum_q`,
//!   `norm_q`).
//! - **Mapped** (`Lazy`, Issue #1463): the rows are read in place from
//!   the segment's input, as stored: `dim` int8 bytes followed by the
//!   8-byte meta. A segment costs no heap beyond the lookup tables (see
//!   the crate-internal `row_pool` module for the run table and the
//!   mapping's lifetime).
//!
//! # Why SoA + padding on the heap
//!
//! - **SoA** keeps `sum_q` / `norm_q` in their own `Vec<u32>` / `Vec<f32>`,
//!   so the per-candidate meta read is a plain indexed load instead of a
//!   decode of inline bytes.
//! - **Padding** each row to `pad_dim = next multiple of 32` means the SIMD
//!   kernels ([`crate::vector::core::distance_quantized::dot_u8_to_i32`]
//!   et al.) process a whole number of 32-byte blocks with **no scalar
//!   tail**. The padded lanes are zero on both the query and the
//!   candidate, so each padded pair contributes 0 to dot / squared-diff /
//!   abs-diff and the result is identical to the unpadded computation —
//!   which is why a mapped row of `dim` bytes, scored against the first
//!   `dim` query bytes, yields the same distance. `sum_q` is a sum over
//!   the same zero-padded bytes and is therefore unchanged, and the affine
//!   dot-product reconstruction uses the *true* `dim` for its `N·offset²`
//!   term (see [`crate::vector::core::distance_quantized::QuantizedQuery`]).
//!
//! Heap footprint: `pad_dim` bytes of int8 payload per vector plus 8 bytes
//! of meta, plus `O(field_count + vector_count)` for the lookup tables.

use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::storage::StorageInput;
use crate::vector::core::quantization::{QuantizedVectorMeta, ScalarQuantParams};
use crate::vector::core::vector::Vector;
use crate::vector::index::row_pool::{
    FieldPositions, MappedRows, MappedRowsBuilder, MappedRowsView,
};

/// Bytes of one stored record payload: `dim` int8 values plus the meta.
#[inline]
const fn stored_row_len(dim: usize) -> usize {
    dim + QuantizedVectorMeta::SERIALIZED_SIZE
}

/// Decode the 8 meta bytes that follow a stored int8 row.
#[inline]
fn decode_meta(bytes: &[u8; QuantizedVectorMeta::SERIALIZED_SIZE]) -> QuantizedVectorMeta {
    let (sum_q, norm_q) = bytes.split_at(4);
    QuantizedVectorMeta {
        sum_q: u32::from_le_bytes(sum_q.try_into().expect("4 bytes")),
        norm_q: f32::from_le_bytes(norm_q.try_into().expect("4 bytes")),
    }
}

/// Where a pool's rows live.
enum Rows {
    /// Rows copied into SoA buffers in position order.
    Heap {
        /// Per-row stride of `int8_data`: `dim` rounded up to the 32-byte
        /// SIMD block ([`QuantizedVectorPool::padded_dim`]). The bytes in
        /// `[dim, pad_dim)` of each row are zero.
        pad_dim: usize,
        /// Row `i` is `int8_data[i * pad_dim..(i + 1) * pad_dim]`.
        int8_data: Vec<u8>,
        /// Per-row `sum_q` (`Σ` of the quantized bytes).
        sum_q: Vec<u32>,
        /// Per-row `norm_q` (f32 norm of the dequantized vector).
        norm_q: Vec<f32>,
    },
    /// Records read in place from the segment's input; each row is `dim`
    /// int8 bytes followed by the 8-byte meta.
    Mapped(MappedRows),
}

impl std::fmt::Debug for Rows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rows::Heap {
                pad_dim, int8_data, ..
            } => f
                .debug_struct("Heap")
                .field("pad_dim", pad_dim)
                .field("bytes", &int8_data.len())
                .finish(),
            Rows::Mapped(rows) => f.debug_tuple("Mapped").field(rows).finish(),
        }
    }
}

/// The rows of a heap pool, borrowed for one search
/// ([`QuantizedVectorPool::records`]).
#[derive(Clone, Copy)]
pub(crate) struct HeapRecords<'a> {
    pad_dim: usize,
    int8_data: &'a [u8],
    sum_q: &'a [u32],
    norm_q: &'a [f32],
}

impl<'a> HeapRecords<'a> {
    /// The zero-padded int8 row (`pad_dim` bytes) and meta at `pos`.
    ///
    /// # Panics
    ///
    /// If `pos` is not a position of the pool.
    #[inline]
    pub(crate) fn record_at(&self, pos: u32) -> (&'a [u8], QuantizedVectorMeta) {
        let start = (pos as usize) * self.pad_dim;
        let int8 = &self.int8_data[start..start + self.pad_dim];
        let meta = QuantizedVectorMeta {
            sum_q: self.sum_q[pos as usize],
            norm_q: self.norm_q[pos as usize],
        };
        (int8, meta)
    }

    /// Base address and stride of the rows, for direct-address software
    /// prefetch.
    pub(crate) fn base_stride(&self) -> (usize, usize) {
        (self.int8_data.as_ptr() as usize, self.pad_dim)
    }
}

/// The rows of a mapped pool, borrowed for one search
/// ([`QuantizedVectorPool::records`]).
#[derive(Clone, Copy)]
pub(crate) struct MappedRecords<'a> {
    dim: usize,
    rows: MappedRowsView<'a>,
}

impl<'a> MappedRecords<'a> {
    /// The stored int8 row (`dim` bytes) at `pos`, and the meta decoded
    /// from the 8 bytes that follow it.
    ///
    /// # Panics
    ///
    /// If `pos` is not a position of the pool.
    #[inline]
    pub(crate) fn record_at(&self, pos: u32) -> (&'a [u8], QuantizedVectorMeta) {
        let record = self.rows.row(pos, stored_row_len(self.dim));
        let (int8, meta) = record.split_at(self.dim);
        let meta = meta.try_into().expect("the row ends with the 8-byte meta");
        (int8, decode_meta(meta))
    }

    /// Base address and stride for direct-address software prefetch, when
    /// the rows form one run.
    pub(crate) fn base_stride(&self) -> Option<(usize, usize)> {
        self.rows.base_stride()
    }
}

/// A pool's rows borrowed for one search: [`HeapRecords`] or
/// [`MappedRecords`].
#[derive(Clone, Copy)]
pub(crate) enum Records<'a> {
    Heap(HeapRecords<'a>),
    Mapped(MappedRecords<'a>),
}

impl<'a> Records<'a> {
    /// The int8 row and meta at `pos` (see [`QuantizedVectorPool::record_at`]).
    #[inline]
    pub(crate) fn record_at(&self, pos: u32) -> (&'a [u8], QuantizedVectorMeta) {
        match self {
            Records::Heap(rows) => rows.record_at(pos),
            Records::Mapped(rows) => rows.record_at(pos),
        }
    }
}

/// int8 representation of one segment's vectors.
///
/// Index-agnostic (HNSW, Flat and IVF). Built once at reader load time and
/// shared across search threads via `Arc<QuantizedVectorPool>`. All fields
/// are immutable after construction.
#[derive(Debug)]
pub struct QuantizedVectorPool {
    /// Per-segment quantization params (`offset`, `scale`).
    pub params: ScalarQuantParams,
    /// True vector dimension (number of meaningful int8 values).
    pub dim: usize,
    rows: Rows,
    /// Per-field doc_id -> vector position.
    ///
    /// `field_index[field][doc_id] = i` such that [`Self::record_at`]`(i)`
    /// is the row for `(doc_id, field)`. The outer `String` key lookup
    /// only happens once per search; the inner `u64` lookup is the
    /// per-candidate hot-path op.
    pub field_index: HashMap<String, Arc<HashMap<u64, u32>>>,
    /// Total vector count (one per stored record).
    pub vector_count: usize,
}

impl QuantizedVectorPool {
    /// Padded per-vector int8 stride for a given `dim`: `dim` rounded up
    /// to the SIMD block size (32). Exposed so callers that only know the
    /// dimension (e.g. the HNSW prefetch-address builder) can compute the
    /// stride without a pool instance. Delegates to the canonical
    /// [`crate::vector::core::distance_quantized::padded_dim`] so the pool
    /// and the per-query padding always agree.
    #[inline]
    pub const fn padded_dim(dim: usize) -> usize {
        crate::vector::core::distance_quantized::padded_dim(dim)
    }

    /// Build a heap pool from a sequence of
    /// `(doc_id, field_name, int8_data, meta)` records and the per-segment
    /// quantization params. Used by the rerank sidecar's int8 view; the
    /// segment readers use [`QuantizedPoolBuilder`].
    ///
    /// The records may be in any order; the field index is built from
    /// the iteration order, and each int8 payload is written at the
    /// position equal to the iteration index, zero-padded to
    /// [`Self::padded_dim`].
    pub fn build(
        params: ScalarQuantParams,
        dim: usize,
        records: impl IntoIterator<Item = (u64, String, Vec<u8>, QuantizedVectorMeta)>,
    ) -> Self {
        let pad_dim = Self::padded_dim(dim);
        let mut int8_data: Vec<u8> = Vec::new();
        let mut sum_q: Vec<u32> = Vec::new();
        let mut norm_q: Vec<f32> = Vec::new();
        let mut by_field: HashMap<String, HashMap<u64, u32>> = HashMap::new();

        for (doc_id, field, int8, meta) in records {
            debug_assert_eq!(int8.len(), dim, "int8 payload length must equal dim");
            let pos = sum_q.len() as u32;
            int8_data.extend_from_slice(&int8);
            // Zero-fill the padding tail up to the next pad_dim boundary.
            int8_data.resize((pos as usize + 1) * pad_dim, 0);
            sum_q.push(meta.sum_q);
            norm_q.push(meta.norm_q);
            by_field.entry(field).or_default().insert(doc_id, pos);
        }

        let vector_count = sum_q.len();
        let field_index: HashMap<String, Arc<HashMap<u64, u32>>> = by_field
            .into_iter()
            .map(|(field, map)| (field, Arc::new(map)))
            .collect();

        Self {
            params,
            dim,
            rows: Rows::Heap {
                pad_dim,
                int8_data,
                sum_q,
                norm_q,
            },
            field_index,
            vector_count,
        }
    }

    /// Whether rows are read in place from the segment's input.
    #[inline]
    pub fn is_mapped(&self) -> bool {
        matches!(self.rows, Rows::Mapped(_))
    }

    /// Length of the int8 slices [`Self::record_at`] returns:
    /// [`Self::padded_dim`]`(dim)` for a heap pool (zero-padded), `dim` for
    /// a mapped pool. [`crate::vector::core::distance_quantized::distance_quantized`]
    /// accepts both.
    #[inline]
    pub fn row_len(&self) -> usize {
        match &self.rows {
            Rows::Heap { pad_dim, .. } => *pad_dim,
            Rows::Mapped(_) => self.dim,
        }
    }

    /// Borrow the int8 payload + meta for `(doc_id, field)`.
    ///
    /// Returns `None` if the key is not present in this segment.
    /// The int8 slice has length [`Self::row_len`]; its first
    /// [`Self::dim`] bytes are the real quantized values and the rest (on
    /// a heap pool) are zero padding.
    #[inline]
    pub fn get_record(&self, doc_id: u64, field: &str) -> Option<(&[u8], QuantizedVectorMeta)> {
        Some(self.record_at(self.position(doc_id, field)?))
    }

    /// The position of `(doc_id, field)` in this pool, if present.
    #[inline]
    pub fn position(&self, doc_id: u64, field: &str) -> Option<u32> {
        self.field_index.get(field)?.get(&doc_id).copied()
    }

    /// Borrow the int8 payload + meta at vector position `pos`.
    ///
    /// Lower-level than [`Self::get_record`]; useful when the caller
    /// has already cached the per-field doc_id -> position map (the
    /// search hot loop does this once per search via
    /// [`Self::field_position_index`]). The returned int8 slice has
    /// length [`Self::row_len`]. A heap pool reads the meta from its
    /// parallel arrays; a mapped pool decodes the 8 bytes that follow the
    /// row.
    ///
    /// # Panics
    ///
    /// If `pos >= vector_count` (positions come from this pool's own
    /// lookup tables).
    #[inline]
    pub fn record_at(&self, pos: u32) -> (&[u8], QuantizedVectorMeta) {
        self.records().record_at(pos)
    }

    /// The rows borrowed for one search. A hot loop takes this once and
    /// reads every candidate through it: the storage kind is resolved up
    /// front, and a mapped pool resolves its file slice once instead of
    /// per candidate.
    #[inline]
    pub(crate) fn records(&self) -> Records<'_> {
        match &self.rows {
            Rows::Heap {
                pad_dim,
                int8_data,
                sum_q,
                norm_q,
            } => Records::Heap(HeapRecords {
                pad_dim: *pad_dim,
                int8_data,
                sum_q,
                norm_q,
            }),
            Rows::Mapped(rows) => Records::Mapped(MappedRecords {
                dim: self.dim,
                rows: rows.view(),
            }),
        }
    }

    /// Cheap O(1) lookup of the per-field doc_id -> position map,
    /// returning the inner `Arc<HashMap>` so the search hot loop can
    /// hold it without cloning the field name on each call.
    #[inline]
    pub fn field_position_index(&self, field: &str) -> Option<Arc<HashMap<u64, u32>>> {
        self.field_index.get(field).cloned()
    }

    /// Whether the segment contains the given key.
    #[inline]
    pub fn contains(&self, doc_id: u64, field: &str) -> bool {
        self.field_index
            .get(field)
            .is_some_and(|m| m.contains_key(&doc_id))
    }

    /// Iterate over `(doc_id, field_name)` pairs in this segment.
    pub fn keys(&self) -> Vec<(u64, String)> {
        let mut keys: Vec<(u64, String)> = self
            .field_index
            .iter()
            .flat_map(|(field, map)| map.keys().map(move |id| (*id, field.clone())))
            .collect();
        keys.sort_by_key(|(id, _)| *id);
        keys
    }

    /// Dequantize the vector for `(doc_id, field)` into a fresh
    /// `Vector` (f32). Used by the legacy
    /// [`crate::vector::reader::VectorIndexReader::get_vector`] API
    /// path; the search hot loop never calls this.
    ///
    /// Only the first [`Self::dim`] bytes are dequantized; the zero
    /// padding is dropped.
    pub fn dequantize_to_vector(&self, doc_id: u64, field: &str) -> Option<Vector> {
        let (int8, _meta) = self.get_record(doc_id, field)?;
        let data: Vec<f32> = int8[..self.dim]
            .iter()
            .map(|&b| self.params.dequantize_value(b))
            .collect();
        Some(Vector::new(data))
    }

    /// Number of fields with at least one vector.
    #[inline]
    pub fn field_count(&self) -> usize {
        self.field_index.len()
    }

    /// Sorted list of field names present in this segment.
    pub fn field_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.field_index.keys().cloned().collect();
        names.sort();
        names
    }

    /// Per-field doc-id list (sorted). Used to back
    /// [`crate::vector::reader::VectorIndexReader::doc_ids_for_field`].
    pub fn doc_ids_for_field(&self, field: &str) -> Arc<[u64]> {
        let Some(map) = self.field_index.get(field) else {
            return Arc::<[u64]>::from(Vec::<u64>::new());
        };
        let mut ids: Vec<u64> = map.keys().copied().collect();
        ids.sort_unstable();
        Arc::<[u64]>::from(ids)
    }

    /// Bytes this pool keeps on the heap: the int8 payload and meta arrays
    /// when the rows were copied, or the run tables of a mapped pool. Used
    /// for reader memory-footprint reporting; excludes the lookup tables.
    #[inline]
    pub fn heap_size(&self) -> usize {
        match &self.rows {
            Rows::Heap {
                int8_data,
                sum_q,
                norm_q,
                ..
            } => int8_data.len() + sum_q.len() * 4 + norm_q.len() * 4,
            Rows::Mapped(rows) => rows.heap_size(),
        }
    }

    /// Base address and stride for direct-address software prefetch, when
    /// every row lies at a fixed stride (the heap buffer, or a mapped pool
    /// whose rows form one run). The address is only used as a prefetch
    /// hint.
    pub fn prefetch_base_stride(&self) -> Option<(usize, usize)> {
        match self.records() {
            Records::Heap(rows) => Some(rows.base_stride()),
            Records::Mapped(rows) => rows.base_stride(),
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
            Rows::Heap { .. } => None,
            Rows::Mapped(rows) => {
                Some(rows.touch(self.vector_count as u32, stored_row_len(self.dim)))
            }
        }
    }
}

/// Builds a [`QuantizedVectorPool`] while a reader parses a `Scalar8Bit`
/// segment's records (Issue #1463).
///
/// The reader reads each record's prefix itself and then calls
/// [`Self::push`] with the stream positioned at the int8 row; the builder
/// either copies the row and its meta (heap) or records the row's offset
/// and skips it (mapped). The heap builder only reads forward, so the
/// HNSW `Eager` load can still fold its checksum into the parse.
pub(crate) struct QuantizedPoolBuilder {
    params: ScalarQuantParams,
    dim: usize,
    mode: BuilderMode,
    positions: FieldPositions,
    count: u32,
}

enum BuilderMode {
    Heap {
        pad_dim: usize,
        int8_data: Vec<u8>,
        sum_q: Vec<u32>,
        norm_q: Vec<f32>,
    },
    Mapped(MappedRowsBuilder),
}

impl QuantizedPoolBuilder {
    /// A builder that copies the rows into the heap SoA layout.
    ///
    /// # Arguments
    ///
    /// * `params` - The segment's quantization params.
    /// * `dim` - The field dimension.
    /// * `capacity` - Expected number of rows (already bounded by the
    ///   caller against the file size).
    pub(crate) fn heap(params: ScalarQuantParams, dim: usize, capacity: usize) -> Self {
        let pad_dim = QuantizedVectorPool::padded_dim(dim);
        Self {
            params,
            dim,
            mode: BuilderMode::Heap {
                pad_dim,
                int8_data: Vec::with_capacity(capacity * pad_dim),
                sum_q: Vec::with_capacity(capacity),
                norm_q: Vec::with_capacity(capacity),
            },
            positions: FieldPositions::default(),
            count: 0,
        }
    }

    /// A builder that reads rows in place from `input`, or a heap builder
    /// when there is no input or it cannot expose its bytes as one slice.
    ///
    /// # Arguments
    ///
    /// * `params` - The segment's quantization params.
    /// * `dim` - The field dimension.
    /// * `capacity` - Expected number of rows, for the heap fallback.
    /// * `input` - A fresh clone of the segment's input (position 0), or
    ///   `None` when the input could not be cloned.
    ///
    /// # Errors
    ///
    /// Forwards a failure to read the input's size.
    pub(crate) fn mapped(
        params: ScalarQuantParams,
        dim: usize,
        capacity: usize,
        input: Option<Box<dyn StorageInput>>,
    ) -> Result<Self> {
        let Some(rows) = input
            .map(|input| MappedRowsBuilder::new(input, stored_row_len(dim)))
            .transpose()?
            .flatten()
        else {
            return Ok(Self::heap(params, dim, capacity));
        };
        Ok(Self {
            params,
            dim,
            mode: BuilderMode::Mapped(rows),
            positions: FieldPositions::default(),
            count: 0,
        })
    }

    /// Add the record whose int8 row starts at the current position of
    /// `reader`, leaving `reader` positioned after the record's meta.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The record's document id.
    /// * `field_id` - The record's field id in the segment dictionary.
    /// * `reader` - The parse stream, positioned at the int8 row.
    ///
    /// # Errors
    ///
    /// [`LaurusError::Index`] if the row and its meta run past the end of
    /// the mapped file or the segment holds more than `u32::MAX` rows; I/O
    /// errors from `reader`.
    pub(crate) fn push<R: Read + Seek>(
        &mut self,
        doc_id: u64,
        field_id: u16,
        reader: &mut R,
    ) -> Result<()> {
        if self.count == u32::MAX {
            return Err(LaurusError::index(
                "Scalar8Bit vector segment holds more than u32::MAX rows".to_string(),
            ));
        }
        match &mut self.mode {
            BuilderMode::Heap {
                pad_dim,
                int8_data,
                sum_q,
                norm_q,
            } => {
                let start = int8_data.len();
                // Zero-filled, so the bytes past `dim` are the padding.
                int8_data.resize(start + *pad_dim, 0);
                reader.read_exact(&mut int8_data[start..start + self.dim])?;
                let mut meta = [0u8; QuantizedVectorMeta::SERIALIZED_SIZE];
                reader.read_exact(&mut meta)?;
                let meta = decode_meta(&meta);
                sum_q.push(meta.sum_q);
                norm_q.push(meta.norm_q);
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
    pub(crate) fn build(self, field_dict: &[Arc<str>]) -> QuantizedVectorPool {
        let empty_heap = |dim| Rows::Heap {
            pad_dim: QuantizedVectorPool::padded_dim(dim),
            int8_data: Vec::new(),
            sum_q: Vec::new(),
            norm_q: Vec::new(),
        };
        let rows = match self.mode {
            BuilderMode::Heap {
                pad_dim,
                int8_data,
                sum_q,
                norm_q,
            } => Rows::Heap {
                pad_dim,
                int8_data,
                sum_q,
                norm_q,
            },
            // No rows: nothing to map.
            BuilderMode::Mapped(rows) => rows
                .build()
                .map_or_else(|| empty_heap(self.dim), Rows::Mapped),
        };
        QuantizedVectorPool {
            params: self.params,
            dim: self.dim,
            rows,
            field_index: self.positions.into_index(field_dict),
            vector_count: self.count as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(sum_q: u32, norm_q: f32) -> QuantizedVectorMeta {
        QuantizedVectorMeta { sum_q, norm_q }
    }

    fn sample_params() -> ScalarQuantParams {
        ScalarQuantParams {
            offset: -1.0,
            scale: 2.0 / 255.0,
        }
    }

    #[test]
    fn build_packs_records_in_iteration_order() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            4,
            vec![
                (10, "embedding".to_string(), vec![0, 1, 2, 3], meta(6, 1.0)),
                (
                    20,
                    "embedding".to_string(),
                    vec![10, 20, 30, 40],
                    meta(100, 2.0),
                ),
            ],
        );
        assert_eq!(q.vector_count, 2);
        assert_eq!(q.dim, 4);
        assert!(!q.is_mapped());
        // dim 4 pads up to one 32-byte block.
        assert_eq!(q.row_len(), 32);
        assert_eq!(q.heap_size(), 2 * 32 + 2 * 8);

        let (int8_0, meta_0) = q.get_record(10, "embedding").unwrap();
        assert_eq!(int8_0.len(), 32);
        assert_eq!(&int8_0[..4], &[0u8, 1, 2, 3]);
        assert!(int8_0[4..].iter().all(|&b| b == 0), "padding is zero");
        assert_eq!(meta_0.sum_q, 6);
        assert_eq!(meta_0.norm_q, 1.0);

        let (int8_1, meta_1) = q.get_record(20, "embedding").unwrap();
        assert_eq!(&int8_1[..4], &[10u8, 20, 30, 40]);
        assert_eq!(meta_1.sum_q, 100);
        assert_eq!(meta_1.norm_q, 2.0);
    }

    #[test]
    fn build_pads_non_multiple_of_32_dim() {
        // dim 100 -> pad_dim 128 (next multiple of 32).
        let int8: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let q = QuantizedVectorPool::build(
            sample_params(),
            100,
            vec![(1, "f".to_string(), int8.clone(), meta(42, 1.0))],
        );
        assert_eq!(q.row_len(), 128);
        assert_eq!(q.heap_size(), 128 + 8);
        let (payload, m) = q.record_at(0);
        assert_eq!(payload.len(), 128);
        assert_eq!(&payload[..100], &int8[..]);
        assert!(payload[100..].iter().all(|&b| b == 0), "tail padding zero");
        assert_eq!(m.sum_q, 42);
    }

    #[test]
    fn get_record_returns_none_for_missing_keys() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            2,
            vec![(1, "f".to_string(), vec![5, 6], meta(11, 1.0))],
        );
        assert!(q.get_record(2, "f").is_none(), "missing doc_id");
        assert!(q.get_record(1, "other").is_none(), "missing field");
    }

    #[test]
    fn field_position_index_supports_hot_loop_lookup() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            2,
            vec![
                (1, "embedding".to_string(), vec![5, 6], meta(11, 1.0)),
                (2, "embedding".to_string(), vec![7, 8], meta(15, 1.5)),
                (1, "thumbnail".to_string(), vec![9, 10], meta(19, 0.5)),
            ],
        );
        let idx = q.field_position_index("embedding").unwrap();
        assert_eq!(idx.len(), 2);
        let pos = *idx.get(&2).unwrap();
        let (int8, meta_back) = q.record_at(pos);
        assert_eq!(&int8[..2], &[7u8, 8]);
        assert_eq!(meta_back.sum_q, 15);
    }

    #[test]
    fn dequantize_to_vector_inverts_quantize_value() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            3,
            vec![(1, "f".to_string(), vec![0, 128, 255], meta(383, 1.0))],
        );
        let v = q.dequantize_to_vector(1, "f").unwrap();
        // Padding must not leak into the dequantized vector.
        assert_eq!(v.data.len(), 3);
        // offset = -1.0, scale = 2/255
        // u8 = 0   -> -1.0
        // u8 = 128 -> -1.0 + 128 * 2/255 = ~0.0039
        // u8 = 255 -> -1.0 + 255 * 2/255 = 1.0
        assert!((v.data[0] - (-1.0)).abs() < 1e-6);
        assert!((v.data[1] - (-1.0 + 128.0 * 2.0 / 255.0)).abs() < 1e-6);
        assert!((v.data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn keys_sorted_by_doc_id() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            1,
            vec![
                (5, "a".to_string(), vec![10], meta(10, 0.0)),
                (1, "a".to_string(), vec![20], meta(20, 0.0)),
                (3, "b".to_string(), vec![30], meta(30, 0.0)),
            ],
        );
        let keys = q.keys();
        let ids: Vec<u64> = keys.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![1, 3, 5]);
    }

    #[test]
    fn doc_ids_for_field_returns_sorted_arc() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            1,
            vec![
                (5, "f".to_string(), vec![10], meta(10, 0.0)),
                (1, "f".to_string(), vec![20], meta(20, 0.0)),
                (3, "f".to_string(), vec![30], meta(30, 0.0)),
            ],
        );
        let ids = q.doc_ids_for_field("f");
        assert_eq!(ids.as_ref(), &[1u64, 3, 5][..]);
        assert!(q.doc_ids_for_field("missing").is_empty());
    }

    #[test]
    fn field_names_sorted_alphabetically() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            1,
            vec![
                (1, "z".to_string(), vec![0], meta(0, 0.0)),
                (1, "a".to_string(), vec![0], meta(0, 0.0)),
                (1, "m".to_string(), vec![0], meta(0, 0.0)),
            ],
        );
        assert_eq!(q.field_names(), vec!["a", "m", "z"]);
    }

    #[test]
    fn padded_dim_rounds_up_to_block() {
        assert_eq!(QuantizedVectorPool::padded_dim(0), 0);
        assert_eq!(QuantizedVectorPool::padded_dim(1), 32);
        assert_eq!(QuantizedVectorPool::padded_dim(32), 32);
        assert_eq!(QuantizedVectorPool::padded_dim(33), 64);
        assert_eq!(QuantizedVectorPool::padded_dim(100), 128);
        assert_eq!(QuantizedVectorPool::padded_dim(128), 128);
    }

    #[test]
    fn contains_reflects_field_and_doc_id() {
        let q = QuantizedVectorPool::build(
            sample_params(),
            1,
            vec![(1, "f".to_string(), vec![0], meta(0, 0.0))],
        );
        assert!(q.contains(1, "f"));
        assert!(!q.contains(2, "f"));
        assert!(!q.contains(1, "g"));
    }

    // ------------------------------------------------------------------
    // QuantizedPoolBuilder (Issue #1463)
    // ------------------------------------------------------------------

    use crate::storage::Storage;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use std::io::Cursor;

    const DIM: usize = 5;

    /// `(doc_id, field_id, int8 row, meta)` of one test record.
    type Record = (u64, u16, [u8; DIM], QuantizedVectorMeta);

    fn record(out: &mut Vec<u8>, (doc_id, fid, row, meta): &Record) {
        out.extend_from_slice(&doc_id.to_le_bytes());
        out.extend_from_slice(&fid.to_le_bytes());
        out.extend_from_slice(row);
        out.extend_from_slice(&meta.sum_q.to_le_bytes());
        out.extend_from_slice(&meta.norm_q.to_le_bytes());
    }

    /// A segment-like byte layout: an odd-length preamble, then two runs
    /// of records separated by a 4-byte list length (the IVF layout).
    fn segment() -> (Vec<u8>, Vec<Record>) {
        let records = vec![
            (10, 0, [1, 2, 3, 4, 5], meta(15, 1.5)),
            (20, 0, [6, 7, 8, 9, 10], meta(40, 2.5)),
            (10, 1, [255, 0, 255, 0, 255], meta(765, 3.5)),
            (30, 0, [11, 12, 13, 14, 15], meta(65, -4.5)),
        ];
        let mut bytes = vec![7u8; 5];
        bytes.extend_from_slice(&2u32.to_le_bytes());
        for r in &records[..2] {
            record(&mut bytes, r);
        }
        bytes.extend_from_slice(&2u32.to_le_bytes());
        for r in &records[2..] {
            record(&mut bytes, r);
        }
        (bytes, records)
    }

    fn parse(mut builder: QuantizedPoolBuilder, bytes: &[u8]) -> QuantizedVectorPool {
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
        assert_eq!(
            cursor.position(),
            bytes.len() as u64,
            "the parse consumed every byte"
        );
        let dict: Vec<Arc<str>> = vec![Arc::from("a"), Arc::from("b")];
        builder.build(&dict)
    }

    fn mapped_input(bytes: &[u8]) -> Box<dyn StorageInput> {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut out = storage.create_output("seg").unwrap();
        std::io::Write::write_all(&mut out, bytes).unwrap();
        out.close().unwrap();
        storage.open_input("seg").unwrap()
    }

    fn check(pool: &QuantizedVectorPool, records: &[Record]) {
        assert_eq!(pool.vector_count, records.len());
        let names = ["a", "b"];
        for (doc, fid, row, m) in records {
            let (int8, got) = pool.get_record(*doc, names[*fid as usize]).unwrap();
            assert_eq!(int8.len(), pool.row_len());
            assert_eq!(&int8[..DIM], row, "doc {doc} field {fid}");
            assert!(int8[DIM..].iter().all(|&b| b == 0), "padding is zero");
            assert_eq!(got, *m, "doc {doc} field {fid}");
            let v = pool
                .dequantize_to_vector(*doc, names[*fid as usize])
                .unwrap();
            let want: Vec<f32> = row
                .iter()
                .map(|&b| pool.params.dequantize_value(b))
                .collect();
            assert_eq!(v.data.as_slice(), want.as_slice());
        }
        assert!(pool.get_record(99, "a").is_none());
        assert_eq!(pool.field_position_index("a").unwrap().len(), 3);
    }

    #[test]
    fn heap_builder_matches_build() {
        let (bytes, records) = segment();
        let pool = parse(QuantizedPoolBuilder::heap(sample_params(), DIM, 4), &bytes);
        assert!(!pool.is_mapped());
        assert_eq!(pool.row_len(), 32);
        check(&pool, &records);
        let built = QuantizedVectorPool::build(
            sample_params(),
            DIM,
            records
                .iter()
                .map(|(d, f, r, m)| (*d, ["a", "b"][*f as usize].to_string(), r.to_vec(), *m)),
        );
        for pos in 0..4 {
            assert_eq!(pool.record_at(pos), built.record_at(pos), "position {pos}");
        }
        assert_eq!(pool.heap_size(), built.heap_size());
        assert!(pool.touch_mapped_rows().is_none());
    }

    #[test]
    fn mapped_builder_reads_the_same_records_in_place() {
        let (bytes, records) = segment();
        let builder =
            QuantizedPoolBuilder::mapped(sample_params(), DIM, 4, Some(mapped_input(&bytes)))
                .unwrap();
        let pool = parse(builder, &bytes);
        assert!(pool.is_mapped(), "MemoryStorage inputs expose a slice");
        assert_eq!(pool.row_len(), DIM, "mapped rows are not padded");
        check(&pool, &records);
        let heap = parse(QuantizedPoolBuilder::heap(sample_params(), DIM, 4), &bytes);
        for pos in 0..4 {
            let (m_row, m_meta) = pool.record_at(pos);
            let (h_row, h_meta) = heap.record_at(pos);
            assert_eq!(m_row, &h_row[..DIM], "position {pos}");
            assert_eq!(m_meta, h_meta, "position {pos}");
        }
        // Two runs; no row bytes on the heap.
        assert_eq!(pool.heap_size(), 2 * (4 + 8 + 8));
        assert!(pool.prefetch_base_stride().is_none());
        assert!(pool.touch_mapped_rows().is_some());
    }

    #[test]
    fn mapped_builder_without_an_input_copies_the_rows() {
        let (bytes, records) = segment();
        let pool = parse(
            QuantizedPoolBuilder::mapped(sample_params(), DIM, 4, None).unwrap(),
            &bytes,
        );
        assert!(!pool.is_mapped());
        check(&pool, &records);
    }

    #[test]
    fn mapped_builder_rejects_a_record_cut_inside_its_meta() {
        let (bytes, _) = segment();
        // Drop the last byte of the last record's `norm_q`.
        let truncated = &bytes[..bytes.len() - 1];
        let mut builder =
            QuantizedPoolBuilder::mapped(sample_params(), DIM, 4, Some(mapped_input(truncated)))
                .unwrap();
        let mut cursor = Cursor::new(truncated);
        cursor.set_position((truncated.len() - (DIM + 7)) as u64);
        let err = builder.push(30, 0, &mut cursor).unwrap_err();
        assert!(err.to_string().contains("runs past the end"), "{err}");
    }

    #[test]
    fn empty_mapped_builder_builds_an_empty_heap_pool() {
        let (bytes, _) = segment();
        let pool =
            QuantizedPoolBuilder::mapped(sample_params(), DIM, 0, Some(mapped_input(&bytes)))
                .unwrap()
                .build(&[]);
        assert!(!pool.is_mapped());
        assert_eq!(pool.vector_count, 0);
        assert_eq!(pool.row_len(), 32);
    }
}
