//! In-memory rerank storage pool used by the Stage 2 two-stage rerank
//! flow (Issue #481 Stage 2).
//!
//! Sibling type to [`super::quantized_storage::QuantizedVectorPool`].
//! While `QuantizedVectorPool` holds the int8 + meta payload that
//! drives the wide candidate fetch, this pool holds the matching
//! full-precision vectors that the rerank kernel rescores against.
//!
//! Built once at reader load time from the LRS1 sidecar (see
//! [`super::rerank_sidecar`]) and shared across search threads via
//! `Arc<RerankStoragePool>`. All fields are immutable after
//! construction.
//!
//! Memory footprint: `dim * bytes_per_element` per vector
//! (`bytes_per_element` = 4 for [`RerankStorageKind::F32`]) plus
//! `O(field_count + vector_count)` for the lookup tables.
//!
//! # Endianness
//!
//! The on-disk LRS1 payload is little-endian and the in-memory
//! [`RerankStoragePool::data`] buffer holds those bytes verbatim. The
//! [`RerankStoragePool::get_f32_slice`] hot-path accessor reinterprets the bytes
//! as `&[f32]` via [`std::slice::from_raw_parts`], which is sound
//! only on little-endian hosts. A `compile_error!` below catches
//! any attempt to build for a big-endian target.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "RerankStoragePool requires a little-endian host because the LRS1 sidecar's f32 payload \
     is stored little-endian and reinterpreted in place"
);

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::error::{LaurusError, Result};
use crate::vector::core::quantization::{QuantizedVectorMeta, ScalarQuantParams};
use crate::vector::core::rerank::RerankStorageKind;
use crate::vector::index::quantized_storage::QuantizedVectorPool;

/// In-memory rerank storage for one segment's full-precision vectors.
///
/// Index-agnostic (used by HNSW first; Flat / IVF could follow). The
/// payload layout matches the LRS1 sidecar: `vector_count * dim *
/// bytes_per_element` bytes laid out as a flat AoS buffer with each
/// vector stored back-to-back in the same order as the matching LVS1
/// segment.
#[derive(Debug)]
pub struct RerankStoragePool {
    /// On-disk encoding of each stored element.
    pub kind: RerankStorageKind,
    /// Vector dimension.
    pub dim: usize,
    /// Tightly-packed AoS payload: for vector index `i`,
    /// `data[i * record_size .. (i + 1) * record_size]` holds the
    /// per-vector bytes in the encoding selected by [`Self::kind`].
    pub data: Vec<u8>,
    /// Per-field doc_id -> vector position.
    ///
    /// Mirrors [`super::quantized_storage::QuantizedVectorPool::field_index`]
    /// so the search hot loop can reuse the same lookup pattern: hold
    /// the inner `Arc<HashMap<u64, u32>>` once per field, then do an
    /// O(1) `u64` lookup per candidate.
    pub field_index: HashMap<String, Arc<HashMap<u64, u32>>>,
    /// Total vector count (matches `data.len() / record_size`).
    pub vector_count: usize,
    /// Lazily-derived int8 (SQ) view of [`Self::data`], built on first
    /// access by [`Self::int8_view`]. See that method for why this is
    /// derived rather than a second on-disk sidecar (Issue #673).
    int8_view_cache: OnceLock<Option<Arc<QuantizedVectorPool>>>,
}

impl RerankStoragePool {
    /// Bytes occupied by one vector record under the given encoding.
    #[inline]
    pub const fn record_size(dim: usize, kind: RerankStorageKind) -> usize {
        dim * kind.bytes_per_element()
    }

    /// Build from a sequence of `(doc_id, field_name, vector)` records.
    ///
    /// The records may be in any order; the field index is built from
    /// the iteration order, and each vector's payload is written at
    /// the position equal to the iteration index. Callers that need
    /// the position to match the matching LVS1 segment must feed the
    /// records in the same order they fed the LVS1 writer.
    ///
    /// # Panics
    ///
    /// Panics if any input vector's length does not equal `dim`.
    pub fn build(
        kind: RerankStorageKind,
        dim: usize,
        records: impl IntoIterator<Item = (u64, String, Vec<f32>)>,
    ) -> Self {
        let record_size = Self::record_size(dim, kind);
        let mut data: Vec<u8> = Vec::new();
        let mut by_field: HashMap<String, HashMap<u64, u32>> = HashMap::new();

        for (doc_id, field, vector) in records {
            assert_eq!(
                vector.len(),
                dim,
                "rerank storage record dim mismatch: expected {dim}, got {}",
                vector.len()
            );
            let pos = (data.len() / record_size) as u32;
            match kind {
                RerankStorageKind::F32 => {
                    data.reserve(record_size);
                    for v in &vector {
                        data.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            by_field.entry(field).or_default().insert(doc_id, pos);
        }

        let vector_count = data.len() / record_size;
        let field_index: HashMap<String, Arc<HashMap<u64, u32>>> = by_field
            .into_iter()
            .map(|(field, map)| (field, Arc::new(map)))
            .collect();

        Self {
            kind,
            dim,
            data,
            field_index,
            vector_count,
            int8_view_cache: OnceLock::new(),
        }
    }

    /// Build directly from a parsed LRS1 sidecar payload.
    ///
    /// `payload` must be `vector_count * dim * bytes_per_element` long
    /// (which is what [`super::rerank_sidecar::read_sidecar`] returns).
    /// `field_assignment` provides the per-position `(doc_id,
    /// field_name)` pairing so the pool can reconstruct the
    /// `field_index` without re-encoding the bytes. The assignment
    /// must be `vector_count` long and ordered by position.
    ///
    /// # Arguments
    ///
    /// * `kind` - On-disk encoding of each stored element.
    /// * `dim` - Vector dimension.
    /// * `vector_count` - Number of vectors the payload is expected to hold.
    /// * `payload` - Raw LRS1 payload bytes (becomes [`Self::data`]).
    /// * `field_assignment` - Per-position `(doc_id, field_name)` pairs,
    ///   ordered by position.
    ///
    /// # Returns
    ///
    /// The constructed [`RerankStoragePool`].
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::Index`] if `payload.len()` does not equal
    /// `vector_count * dim * bytes_per_element`, or if
    /// `field_assignment.len()` does not equal `vector_count` — both
    /// indicate the sidecar payload and its declared shape are
    /// inconsistent (corruption). Validating here instead of panicking
    /// keeps a corrupt or hostile sidecar from aborting the process
    /// (Issue #805).
    pub fn from_sidecar_payload(
        kind: RerankStorageKind,
        dim: usize,
        vector_count: usize,
        payload: Vec<u8>,
        field_assignment: &[(u64, String)],
    ) -> Result<Self> {
        let record_size = Self::record_size(dim, kind);
        let expected_len = vector_count * record_size;
        if payload.len() != expected_len {
            return Err(LaurusError::index(format!(
                "rerank sidecar payload length mismatch: expected {expected_len} bytes \
                 (vector_count={vector_count} * dim={dim} * \
                 bytes_per_element={}), got {}",
                kind.bytes_per_element(),
                payload.len()
            )));
        }
        if field_assignment.len() != vector_count {
            return Err(LaurusError::index(format!(
                "rerank sidecar field assignment length mismatch: expected {vector_count}, got {}",
                field_assignment.len()
            )));
        }

        let mut by_field: HashMap<String, HashMap<u64, u32>> = HashMap::new();
        for (pos, (doc_id, field)) in field_assignment.iter().enumerate() {
            by_field
                .entry(field.clone())
                .or_default()
                .insert(*doc_id, pos as u32);
        }

        let field_index: HashMap<String, Arc<HashMap<u64, u32>>> = by_field
            .into_iter()
            .map(|(field, map)| (field, Arc::new(map)))
            .collect();

        Ok(Self {
            kind,
            dim,
            data: payload,
            field_index,
            vector_count,
            int8_view_cache: OnceLock::new(),
        })
    }

    /// Borrow the f32 slice for `(doc_id, field)`.
    ///
    /// Returns `None` if the key is not present in this segment.
    /// The returned slice has length [`Self::dim`].
    #[inline]
    pub fn get_f32_slice(&self, doc_id: u64, field: &str) -> Option<&[f32]> {
        let pos = self.field_index.get(field)?.get(&doc_id).copied()?;
        Some(self.f32_slice_at(pos))
    }

    /// Borrow the f32 slice at vector position `pos`.
    ///
    /// Lower-level than [`Self::get_f32_slice`]; useful when the
    /// caller has already cached the per-field doc_id -> position map
    /// (the search hot loop does this once per search via
    /// [`Self::field_position_index`]).
    ///
    /// # Panics
    ///
    /// Panics if `pos >= self.vector_count`.
    #[inline]
    pub fn f32_slice_at(&self, pos: u32) -> &[f32] {
        debug_assert_eq!(self.kind, RerankStorageKind::F32);
        let record_size = Self::record_size(self.dim, self.kind);
        let start = (pos as usize) * record_size;
        let end = start + record_size;
        let bytes = &self.data[start..end];
        // SAFETY:
        // - `bytes` is a `&[u8]` slice that originated from a
        //   `Vec<u8>` whose allocator returns memory aligned to at
        //   least the alignment of `f32` (the global allocator is at
        //   least 8-byte aligned).
        // - `start` is a multiple of `record_size`, which itself is
        //   `dim * 4`, so the slice start is 4-byte aligned.
        // - The byte length equals `self.dim * 4`, exactly enough for
        //   `self.dim` `f32` values.
        // - The underlying bytes were written via `f32::to_le_bytes`
        //   (see `Self::build`) or copied from a little-endian LRS1
        //   sidecar payload. The host is little-endian (enforced by
        //   the module-level `compile_error!`), so the bit pattern
        //   matches `f32`'s native representation.
        // - The pool is shared via `Arc<Self>`; `&self` lifetime ties
        //   the returned slice to the pool's borrow, so no aliasing
        //   with mutable accessors is possible.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), self.dim) }
    }

    /// Lazily-derived int8 scalar-quantized view of this pool's f32
    /// payload, used as the middle stage of the PQ → SQ → f32 rerank
    /// chain (Issue #673).
    ///
    /// Every 3-stage chain ends in an exact f32 stage — this pool always
    /// holds that f32 data already. Deriving an int8 view from it (rather
    /// than persisting a second on-disk sidecar) avoids storing the same
    /// information twice and keeps the derived positions trivially
    /// consistent with [`Self::f32_slice_at`] (both are indexed by the
    /// same `pos`).
    ///
    /// Built once on first call via a per-segment quantization trained
    /// from this pool's own f32 vectors, then cached for the pool's
    /// lifetime. Returns `None` when the pool is empty or training fails
    /// (e.g. non-finite values) — callers must treat that the same as
    /// "no SQ stage available" and fall back to the 2-stage chain, never
    /// error the query over this optimization.
    pub fn int8_view(&self) -> Option<&Arc<QuantizedVectorPool>> {
        self.int8_view_cache
            .get_or_init(|| self.build_int8_view())
            .as_ref()
    }

    /// Build the derived int8 view backing [`Self::int8_view`].
    fn build_int8_view(&self) -> Option<Arc<QuantizedVectorPool>> {
        if self.vector_count == 0 {
            return None;
        }

        // Invert `field_index` (field -> doc_id -> pos) into a
        // position-ordered `(doc_id, field)` assignment so records can be
        // fed to `QuantizedVectorPool::build` in the same position order
        // as this pool's own `f32_slice_at`.
        let mut assignment: Vec<Option<(u64, &str)>> = vec![None; self.vector_count];
        for (field, map) in &self.field_index {
            for (&doc_id, &pos) in map.iter() {
                *assignment.get_mut(pos as usize)? = Some((doc_id, field.as_str()));
            }
        }

        let params = ScalarQuantParams::train_from_slices(
            (0..self.vector_count as u32).map(|pos| self.f32_slice_at(pos)),
        )
        .ok()?;

        let mut records = Vec::with_capacity(self.vector_count);
        for (pos, slot) in assignment.into_iter().enumerate() {
            let (doc_id, field) = slot?;
            let f32_vec = self.f32_slice_at(pos as u32);
            let q = params.quantize_slice(f32_vec);
            let meta = QuantizedVectorMeta::from_quantized(&q, &params);
            records.push((doc_id, field.to_string(), q, meta));
        }

        Some(Arc::new(QuantizedVectorPool::build(
            params, self.dim, records,
        )))
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_size_is_dim_times_four_for_f32() {
        assert_eq!(RerankStoragePool::record_size(0, RerankStorageKind::F32), 0);
        assert_eq!(
            RerankStoragePool::record_size(128, RerankStorageKind::F32),
            512
        );
    }

    #[test]
    fn build_packs_records_in_iteration_order() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            3,
            vec![
                (10, "embedding".to_string(), vec![1.0, 2.0, 3.0]),
                (20, "embedding".to_string(), vec![-1.0, 0.0, 1.0]),
            ],
        );
        assert_eq!(pool.vector_count, 2);
        assert_eq!(pool.dim, 3);
        assert_eq!(pool.data.len(), 2 * 3 * 4);

        let v0 = pool.get_f32_slice(10, "embedding").unwrap();
        assert_eq!(v0, &[1.0, 2.0, 3.0]);

        let v1 = pool.get_f32_slice(20, "embedding").unwrap();
        assert_eq!(v1, &[-1.0, 0.0, 1.0]);
    }

    #[test]
    fn get_f32_slice_returns_none_for_missing_keys() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            2,
            vec![(1, "f".to_string(), vec![5.0, 6.0])],
        );
        assert!(pool.get_f32_slice(2, "f").is_none(), "missing doc_id");
        assert!(pool.get_f32_slice(1, "other").is_none(), "missing field");
    }

    #[test]
    fn field_position_index_supports_hot_loop_lookup() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            2,
            vec![
                (1, "embedding".to_string(), vec![5.0, 6.0]),
                (2, "embedding".to_string(), vec![7.0, 8.0]),
                (1, "thumbnail".to_string(), vec![9.0, 10.0]),
            ],
        );
        let idx = pool.field_position_index("embedding").unwrap();
        assert_eq!(idx.len(), 2);
        let pos = *idx.get(&2).unwrap();
        assert_eq!(pool.f32_slice_at(pos), &[7.0, 8.0]);
    }

    #[test]
    fn contains_reflects_field_and_doc_id() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            1,
            vec![(1, "f".to_string(), vec![1.5])],
        );
        assert!(pool.contains(1, "f"));
        assert!(!pool.contains(2, "f"));
        assert!(!pool.contains(1, "g"));
    }

    #[test]
    fn field_names_sorted_alphabetically() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            1,
            vec![
                (1, "z".to_string(), vec![0.0]),
                (1, "a".to_string(), vec![0.0]),
                (1, "m".to_string(), vec![0.0]),
            ],
        );
        assert_eq!(pool.field_names(), vec!["a", "m", "z"]);
    }

    #[test]
    fn field_count_matches_distinct_fields() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            1,
            vec![
                (1, "a".to_string(), vec![0.0]),
                (2, "a".to_string(), vec![0.0]),
                (1, "b".to_string(), vec![0.0]),
            ],
        );
        assert_eq!(pool.field_count(), 2);
    }

    #[test]
    #[should_panic(expected = "dim mismatch")]
    fn build_panics_on_dim_mismatch() {
        RerankStoragePool::build(
            RerankStorageKind::F32,
            3,
            vec![(1, "f".to_string(), vec![1.0, 2.0])],
        );
    }

    #[test]
    fn from_sidecar_payload_round_trips_full_precision() {
        let dim = 4;
        let vectors: Vec<f32> = vec![
            1.5, 2.5, 3.5, 4.5, // pos 0
            -1.0, 0.0, 1.0, 2.0, // pos 1
        ];
        let mut payload = Vec::new();
        for v in &vectors {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let assignment = vec![(7u64, "f".to_string()), (9u64, "f".to_string())];
        let pool = RerankStoragePool::from_sidecar_payload(
            RerankStorageKind::F32,
            dim,
            2,
            payload,
            &assignment,
        )
        .unwrap();
        assert_eq!(pool.vector_count, 2);
        assert_eq!(pool.get_f32_slice(7, "f").unwrap(), &vectors[0..4]);
        assert_eq!(pool.get_f32_slice(9, "f").unwrap(), &vectors[4..8]);
    }

    /// Assert `err` is a [`LaurusError::Index`] whose message contains
    /// `fragment`.
    fn assert_index_error(err: LaurusError, fragment: &str) {
        match err {
            LaurusError::Index(msg) => assert!(
                msg.contains(fragment),
                "message {msg:?} should contain {fragment:?}"
            ),
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn from_sidecar_payload_rejects_wrong_payload_size() {
        let err = RerankStoragePool::from_sidecar_payload(
            RerankStorageKind::F32,
            4,
            2,
            vec![0u8; 10], // expects 32 bytes
            &[(1u64, "f".to_string()), (2u64, "f".to_string())],
        )
        .unwrap_err();
        assert_index_error(err, "payload length mismatch");
    }

    #[test]
    fn from_sidecar_payload_rejects_wrong_assignment_size() {
        let err = RerankStoragePool::from_sidecar_payload(
            RerankStorageKind::F32,
            4,
            2,
            vec![0u8; 32],
            &[(1u64, "f".to_string())], // wrong length
        )
        .unwrap_err();
        assert_index_error(err, "field assignment length mismatch");
    }

    #[test]
    fn f32_slice_pointer_is_aligned_to_four() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            8,
            (0..16).map(|i| (i as u64, "f".to_string(), vec![i as f32; 8])),
        );
        for pos in 0..16u32 {
            let slice = pool.f32_slice_at(pos);
            let addr = slice.as_ptr() as usize;
            assert_eq!(addr % std::mem::align_of::<f32>(), 0, "pos {pos}");
            assert_eq!(slice.len(), 8);
        }
    }

    #[test]
    fn int8_view_positions_match_f32_pool_positions() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            3,
            vec![
                (10, "embedding".to_string(), vec![0.0, 1.0, 2.0]),
                (20, "embedding".to_string(), vec![-1.5, 3.0, 0.5]),
                (30, "thumbnail".to_string(), vec![5.0, -2.0, 1.0]),
            ],
        );
        let int8_pool = pool.int8_view().expect("training should succeed");
        assert_eq!(int8_pool.vector_count, 3);
        assert_eq!(int8_pool.dim, 3);

        for &(doc_id, field) in &[(10u64, "embedding"), (20, "embedding"), (30, "thumbnail")] {
            let f32_pos = pool.field_position_index(field).unwrap()[&doc_id];
            let int8_pos = int8_pool.field_position_index(field).unwrap()[&doc_id];
            assert_eq!(
                f32_pos, int8_pos,
                "derived int8 position must match the f32 pool position for ({doc_id}, {field})"
            );
        }
    }

    #[test]
    fn int8_view_dequantizes_within_quantization_error() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            4,
            vec![
                (1, "f".to_string(), vec![0.0, 10.0, -5.0, 2.5]),
                (2, "f".to_string(), vec![3.0, -8.0, 6.0, 0.0]),
            ],
        );
        let int8_pool = pool.int8_view().unwrap();
        for &doc_id in &[1u64, 2] {
            let original = pool.get_f32_slice(doc_id, "f").unwrap();
            let (codes, meta) = int8_pool.get_record(doc_id, "f").unwrap();
            let dequantized = int8_pool.params.dequantize(codes);
            for (o, d) in original.iter().zip(&dequantized) {
                // One quantization step of slack: scale = range / 255.
                assert!(
                    (o - d).abs() <= int8_pool.params.scale + 1e-4,
                    "dequantized {d} too far from original {o}"
                );
            }
            assert!(meta.norm_q >= 0.0);
        }
    }

    #[test]
    fn int8_view_is_built_only_once() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            2,
            vec![(1, "f".to_string(), vec![1.0, 2.0])],
        );
        let first = pool.int8_view().unwrap() as *const Arc<QuantizedVectorPool>;
        let second = pool.int8_view().unwrap() as *const Arc<QuantizedVectorPool>;
        assert_eq!(first, second, "int8_view must cache the derived pool");
    }

    #[test]
    fn int8_view_is_none_for_empty_pool() {
        let pool = RerankStoragePool::build(RerankStorageKind::F32, 3, Vec::new());
        assert!(pool.int8_view().is_none());
    }

    #[test]
    fn int8_view_is_none_when_training_fails_on_non_finite_data() {
        let pool = RerankStoragePool::build(
            RerankStorageKind::F32,
            2,
            vec![(1, "f".to_string(), vec![f32::NAN, 1.0])],
        );
        assert!(pool.int8_view().is_none());
    }
}
