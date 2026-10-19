use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::vector::Vector;
use crate::vector::index::f16_storage::F16VectorPool;
#[cfg(feature = "pq-fastscan")]
use crate::vector::index::pq_fastscan_storage::PqFastScanPool;
use crate::vector::index::pq_storage::PqVectorPool;
use crate::vector::index::quantized_storage::QuantizedVectorPool;

/// Storage for one segment's vectors.
///
/// Each variant holds an immutable pool behind an `Arc`, so a reader's
/// storage is freely shareable across search threads. The `Scalar8Bit`
/// and `Float16` pools keep their rows on the heap (`Eager` loading) or
/// read them in place from the mapped segment file (`Lazy` loading,
/// Issues #623 and #1463).
#[derive(Debug, Clone)]
pub enum VectorStorage {
    /// All vectors are loaded into memory as f32 (legacy path used by
    /// Flat / IVF until Step 7 of Issue #481 Stage 1).
    Owned(Arc<HashMap<(u64, String), Vector>>),
    /// int8 rows + per-vector meta (Issue #481 Stage 1, Step 6), on the
    /// heap or read in place from the mapped segment depending on the
    /// loading mode (Issue #1463). The search hot loop accesses the inner
    /// [`QuantizedVectorPool`] directly via [`Self::quantized_pool`]
    /// instead of going through [`Self::get`], which dequantizes for the
    /// [`crate::vector::reader::VectorIndexReader::get_vector`] API.
    OwnedQuantized(Arc<QuantizedVectorPool>),
    /// All vectors are loaded into memory as PQ codes plus the
    /// per-segment codebook (Issue #481 Stage 3, HNSW only). The
    /// search hot loop accesses the inner [`PqVectorPool`] directly
    /// via [`Self::pq_pool`] and feeds codes + the per-query LUT to
    /// [`crate::vector::core::distance_quantized::distance_pq_adc`].
    OwnedPq(Arc<PqVectorPool>),
    /// All vectors are loaded into memory as 4-bit packed FastScan
    /// codes plus the per-segment K=16 codebook (Issue #695 / part D
    /// of #651, HNSW only, experimental). The search hot loop walks
    /// the inner [`PqFastScanPool`] directly through
    /// [`crate::vector::index::pq_fastscan_avx2::distance_pq_fastscan_block`]
    /// which dispatches to AVX2 / NEON / scalar by CPU. Available only
    /// when the crate is built with the `pq-fastscan` cargo feature.
    #[cfg(feature = "pq-fastscan")]
    OwnedPqFastScan(Arc<PqFastScanPool>),
    /// Float16 rows (Issue #623), on the heap or read in place from the
    /// mapped segment depending on the loading mode. The search hot loop
    /// reads rows through [`Self::f16_pool`] and
    /// [`crate::vector::core::distance_f16::distance_f16`]; [`Self::get`]
    /// decodes to f32.
    Float16(Arc<F16VectorPool>),
}

impl VectorStorage {
    /// If this storage is the Scalar8Bit quantized variant, return the
    /// underlying [`QuantizedVectorPool`] so the search hot loop can pull
    /// `(int8 slice, meta)` directly without going through the
    /// dequantizing [`Self::get`] path.
    pub fn quantized_pool(&self) -> Option<&Arc<QuantizedVectorPool>> {
        match self {
            VectorStorage::OwnedQuantized(pool) => Some(pool),
            _ => None,
        }
    }

    /// If this storage is the in-memory PQ variant (Stage 3), return
    /// the underlying [`PqVectorPool`] so the search hot loop can pull
    /// `(codes, codebook)` directly and dispatch to the PQ ADC
    /// kernel.
    pub fn pq_pool(&self) -> Option<&Arc<PqVectorPool>> {
        match self {
            VectorStorage::OwnedPq(pool) => Some(pool),
            _ => None,
        }
    }

    /// If this storage is the in-memory PQ FastScan variant (Issue
    /// #695 / part D of #651), return the underlying
    /// [`PqFastScanPool`] so the search hot loop can walk the
    /// block-transposed 4-bit packed codes directly and dispatch to
    /// the AVX2 / NEON / scalar FastScan kernel via
    /// [`crate::vector::index::pq_fastscan_avx2::distance_pq_fastscan_block`].
    #[cfg(feature = "pq-fastscan")]
    pub fn pq_fastscan_pool(&self) -> Option<&Arc<PqFastScanPool>> {
        match self {
            VectorStorage::OwnedPqFastScan(pool) => Some(pool),
            _ => None,
        }
    }

    /// If this storage holds Float16 rows (Issue #623), return the
    /// underlying [`F16VectorPool`] so the search hot loop can score rows
    /// in place.
    pub fn f16_pool(&self) -> Option<&Arc<F16VectorPool>> {
        match self {
            VectorStorage::Float16(pool) => Some(pool),
            _ => None,
        }
    }

    /// The per-field `doc_id -> position` map of a pool, used by readers
    /// to build the HNSW ordinal tables. `None` for the legacy
    /// [`Self::Owned`].
    pub fn field_position_index(&self, field: &str) -> Option<Arc<HashMap<u64, u32>>> {
        match self {
            VectorStorage::OwnedQuantized(pool) => pool.field_position_index(field),
            VectorStorage::OwnedPq(pool) => pool.field_position_index(field),
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(pool) => pool.field_position_index(field),
            VectorStorage::Float16(pool) => pool.field_position_index(field),
            VectorStorage::Owned(_) => None,
        }
    }

    /// Returns all keys stored in this vector storage.
    pub fn keys(&self) -> Vec<(u64, String)> {
        match self {
            VectorStorage::Owned(map) => map.keys().cloned().collect(),
            VectorStorage::OwnedQuantized(pool) => pool.keys(),
            VectorStorage::OwnedPq(pool) => pool.keys(),
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(pool) => pool.keys(),
            VectorStorage::Float16(pool) => pool.keys(),
        }
    }

    /// Returns the number of vectors stored.
    pub fn len(&self) -> usize {
        match self {
            VectorStorage::Owned(map) => map.len(),
            VectorStorage::OwnedQuantized(pool) => pool.vector_count,
            VectorStorage::OwnedPq(pool) => pool.vector_count,
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(pool) => pool.vector_count(),
            VectorStorage::Float16(pool) => pool.vector_count(),
        }
    }

    /// Returns `true` if no vectors are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if a vector exists for `(doc_id, field_name)`.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document id.
    /// * `field_name` - The vector field name.
    pub fn contains(&self, doc_id: u64, field_name: &str) -> bool {
        match self {
            VectorStorage::Owned(map) => map.contains_key(&(doc_id, field_name.to_string())),
            VectorStorage::OwnedQuantized(pool) => pool.contains(doc_id, field_name),
            VectorStorage::OwnedPq(pool) => pool.contains(doc_id, field_name),
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(pool) => pool.contains(doc_id, field_name),
            VectorStorage::Float16(pool) => pool.contains(doc_id, field_name),
        }
    }

    /// Retrieves a vector by its key, decoded to f32.
    ///
    /// For the `Owned` variant the vector is cloned (O(1) due to `Arc`
    /// wrapping); the pool variants dequantize or decode the stored row
    /// into a new [`Vector`].
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document id.
    /// * `field_name` - The vector field name.
    /// * `_dimension` - The expected number of dimensions (the pools know
    ///   their own).
    ///
    /// # Returns
    ///
    /// `Ok(Some(vector))` if the key exists, `Ok(None)` otherwise.
    ///
    /// # Errors
    ///
    /// None of the current variants fail; the `Result` is kept for the
    /// callers' error plumbing.
    pub fn get(&self, doc_id: u64, field_name: &str, _dimension: usize) -> Result<Option<Vector>> {
        match self {
            // Match-only legacy variant (never constructed by current
            // readers); the owned-key probe is acceptable here.
            VectorStorage::Owned(map) => Ok(map.get(&(doc_id, field_name.to_string())).cloned()),
            VectorStorage::OwnedQuantized(pool) => {
                Ok(pool.dequantize_to_vector(doc_id, field_name))
            }
            VectorStorage::OwnedPq(pool) => Ok(pool.dequantize_to_vector(doc_id, field_name)),
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(pool) => {
                Ok(pool.dequantize_to_vector(doc_id, field_name))
            }
            VectorStorage::Float16(pool) => Ok(pool.decode_to_vector(doc_id, field_name)),
        }
    }
}
