use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, RwLock};

use crate::error::{LaurusError, Result};
use crate::storage::{Storage, StorageInput};
use crate::vector::core::quantization::{QuantizedVectorMeta, ScalarQuantParams};
use crate::vector::core::vector::Vector;
use crate::vector::index::f16_storage::F16VectorPool;
#[cfg(feature = "pq-fastscan")]
use crate::vector::index::pq_fastscan_storage::PqFastScanPool;
use crate::vector::index::pq_storage::PqVectorPool;
use crate::vector::index::quantized_storage::QuantizedVectorPool;

/// Storage for vectors (in-memory or on-demand from disk).
///
/// # Thread Safety
///
/// - The `Owned` variant holds an immutable `Arc<HashMap>` that is freely
///   shareable across threads.
/// - The `OnDemand` variant stores a reference to the underlying
///   [`Storage`] and the file name so that each call to [`get`](Self::get)
///   opens an independent file handle.  This eliminates the previous
///   `Mutex`-based serialization and allows fully concurrent reads.
#[derive(Debug, Clone)]
pub enum VectorStorage {
    /// All vectors are loaded into memory as f32 (legacy path used by
    /// Flat / IVF until Step 7 of Issue #481 Stage 1).
    Owned(Arc<HashMap<(u64, String), Vector>>),
    /// All vectors are loaded into memory as int8 + per-vector meta
    /// (Issue #481 Stage 1, Step 6). Used by HNSW Eager mode; the
    /// search hot loop accesses the inner [`QuantizedVectorPool`]
    /// directly via [`Self::quantized_pool`] instead of going through
    /// [`Self::get`], which dequantizes lazily for the legacy
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
    /// Vectors are read from disk on demand.
    ///
    /// Each [`get`](Self::get) call opens a fresh [`StorageInput`](crate::storage::StorageInput)
    /// via [`Storage::open_input`], performs a single seek + read, and closes
    /// the handle.  For mmap-backed storage this is essentially free; for
    /// file-backed storage the OS typically caches the file descriptor.
    ///
    /// `quant_params` controls how the per-vector data section is decoded:
    /// `Some(params)` -> Issue #481 Stage 1 quantized format
    /// (int8 + per-vector meta, dequantized on read);
    /// `None` -> legacy f32 format (still in use by Flat / IVF until
    /// Step 7 of #481 Stage 1 migrates them).
    OnDemand {
        /// Reference to the storage backend (e.g. file system, mmap).
        storage: Arc<dyn Storage>,
        /// Name of the vector index file within the storage.
        file_name: String,
        /// Pre-built mapping from `(doc_id, field_id)` to the byte
        /// offset of the record's payload (Issue #633 PR-B: interned
        /// u16 field ids instead of one heap `String` per record).
        offsets: Arc<HashMap<(u64, u16), u64>>,
        /// Per-segment field-name dictionary; `field_id` indexes into
        /// it. Resolution from a name is a linear scan (segments hold
        /// 1–3 fields in practice), which allocates nothing.
        field_dict: Arc<[Arc<str>]>,
        /// Per-segment quantization params, if the on-disk vector
        /// format is the Stage-1 quantized layout. `None` means the
        /// legacy f32 layout.
        quant_params: Option<ScalarQuantParams>,
        /// Lazily-opened input handle, shared across `get()` calls in
        /// the same search. Avoids paying the
        /// `Storage::open_input(file_name)` cost (`statx` syscall via
        /// the mmap-cache metadata check, mmap-cache lookup, `Arc`
        /// clone, and `Box` allocation) on every candidate-vector
        /// lookup. Subsequent gets call `clone_input()` to obtain a
        /// fresh seek cursor without re-opening the file. See #522.
        ///
        /// The lock is read-heavy: after the first `get()` populates
        /// it, every subsequent `get()` takes a read lock and never
        /// blocks another reader, so concurrent HNSW searches do not
        /// serialise on this field.
        cached_input: Arc<RwLock<Option<Box<dyn StorageInput>>>>,
    },
}

impl VectorStorage {
    /// If this storage is the in-memory Scalar8Bit quantized variant,
    /// return the underlying [`QuantizedVectorPool`] so the search hot
    /// loop can pull `(int8 slice, meta)` directly without going
    /// through the dequantizing [`Self::get`] path.
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

    /// The per-field `doc_id -> position` map of an in-memory pool, used by
    /// readers to build the HNSW ordinal tables. `None` for
    /// [`Self::OnDemand`] and the legacy [`Self::Owned`].
    pub fn field_position_index(&self, field: &str) -> Option<Arc<HashMap<u64, u32>>> {
        match self {
            VectorStorage::OwnedQuantized(pool) => pool.field_position_index(field),
            VectorStorage::OwnedPq(pool) => pool.field_position_index(field),
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(pool) => pool.field_position_index(field),
            VectorStorage::Float16(pool) => pool.field_position_index(field),
            VectorStorage::Owned(_) | VectorStorage::OnDemand { .. } => None,
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
            VectorStorage::OnDemand {
                offsets,
                field_dict,
                ..
            } => offsets
                .keys()
                .map(|&(doc_id, fid)| (doc_id, field_dict[fid as usize].to_string()))
                .collect(),
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
            VectorStorage::OnDemand { offsets, .. } => offsets.len(),
        }
    }

    /// Returns `true` if no vectors are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if a vector exists for `(doc_id, field_name)`.
    ///
    /// Allocation-free on every variant (Issue #633 PR-B): the
    /// `OnDemand` arm resolves the field name against the segment
    /// dictionary instead of materializing an owned key.
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
            VectorStorage::OnDemand {
                offsets,
                field_dict,
                ..
            } => crate::vector::index::format::resolve_field_id(field_dict, field_name)
                .is_some_and(|fid| offsets.contains_key(&(doc_id, fid))),
        }
    }

    /// Retrieves a vector by its key.
    ///
    /// For the `Owned` variant the vector is cloned (O(1) due to `Arc`
    /// wrapping).  For the `OnDemand` variant a fresh file handle is opened,
    /// the reader seeks to the recorded offset, and the vector data is read
    /// directly.
    ///
    /// Allocation-free key handling (Issue #633 PR-B): callers pass
    /// `(doc_id, &str)` and the `OnDemand` arm resolves the name against
    /// the segment dictionary — the former per-call
    /// `field_name.to_string()` key materialization is gone.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document id.
    /// * `field_name` - The vector field name.
    /// * `dimension` - The expected number of dimensions (used to size the read buffer).
    ///
    /// # Returns
    ///
    /// `Ok(Some(vector))` if the key exists, `Ok(None)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on I/O failure.
    pub fn get(&self, doc_id: u64, field_name: &str, dimension: usize) -> Result<Option<Vector>> {
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
            VectorStorage::OnDemand {
                storage,
                file_name,
                offsets,
                field_dict,
                quant_params,
                cached_input,
            } => {
                let Some(fid) =
                    crate::vector::index::format::resolve_field_id(field_dict, field_name)
                else {
                    return Ok(None);
                };
                let Some(&offset) = offsets.get(&(doc_id, fid)) else {
                    return Ok(None);
                };
                // Reuse the cached input handle if it has been opened by a
                // previous `get()` on this storage. The cache is read-heavy:
                // after the first opener wins the write lock, every subsequent
                // call takes the read lock and clones via `clone_input()` —
                // no `statx` syscall, no mmap-cache lookup, just an `Arc`
                // clone and a `Box` allocation for the fresh cursor.
                let mut input = {
                    let guard = cached_input
                        .read()
                        .map_err(|_| LaurusError::internal("cached_input RwLock poisoned"))?;
                    if let Some(cached) = guard.as_ref() {
                        cached.clone_input()?
                    } else {
                        // First call — drop the read lock and acquire the
                        // write lock so we can lazily open the file. The
                        // double-check after locking handles the rare case
                        // where another thread populated the cache between
                        // our read-lock release and write-lock acquisition.
                        drop(guard);
                        let mut wguard = cached_input
                            .write()
                            .map_err(|_| LaurusError::internal("cached_input RwLock poisoned"))?;
                        if wguard.is_none() {
                            *wguard = Some(storage.open_input(file_name).map_err(|e| {
                                LaurusError::internal(format!("Failed to open vector file: {e}"))
                            })?);
                        }
                        wguard
                            .as_ref()
                            .expect("cached_input populated above")
                            .clone_input()?
                    }
                };

                // Offsets point at the record's payload start (the readers
                // compute them past the doc_id + field-reference prefix at
                // load, Issue #633), so no per-access prefix re-parse is
                // needed — one seek lands directly on the vector data.
                input
                    .seek(SeekFrom::Start(offset))
                    .map_err(LaurusError::Io)?;

                // Read vector data — branch on the on-disk format.
                let values = match quant_params {
                    Some(params) => {
                        // Stage-1 quantized: int8 payload + per-vector meta.
                        // Dequantize back to f32 here so callers see the same
                        // Vector type as the legacy path. Step 6 of #481
                        // Stage 1 will offer an int8-native read path for the
                        // search hot loop.
                        let mut int8_buf = vec![0u8; dimension];
                        input.read_exact(&mut int8_buf)?;
                        let mut meta_buf = [0u8; QuantizedVectorMeta::SERIALIZED_SIZE];
                        input.read_exact(&mut meta_buf)?;
                        int8_buf
                            .iter()
                            .map(|&b| params.dequantize_value(b))
                            .collect()
                    }
                    None => {
                        // Legacy f32 layout (Flat / IVF until Step 7).
                        let mut values = vec![0.0f32; dimension];
                        for value in &mut values {
                            let mut value_buf = [0u8; 4];
                            input.read_exact(&mut value_buf)?;
                            *value = f32::from_le_bytes(value_buf);
                        }
                        values
                    }
                };
                Ok(Some(Vector::new(values)))
            }
        }
    }
}
