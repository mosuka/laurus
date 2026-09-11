//! Flat vector index reader implementation.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization::QuantizedVectorMeta;
use crate::vector::core::vector::Vector;
use crate::vector::index::format::{
    FieldInterner, QuantHeader, VectorSegmentHeader, record_prefix_size,
};
use crate::vector::index::quantized_io::quantized_record_payload_size;
use crate::vector::index::quantized_storage::QuantizedVectorPool;
use crate::vector::reader::{ValidationReport, VectorIndexMetadata, VectorStats};
use crate::vector::reader::{VectorIndexReader, VectorIterator};

use crate::maintenance::deletion::DeletionBitmap;
/// Storage for vectors (in-memory or on-demand).
use crate::vector::index::storage::VectorStorage;

/// Reader for flat (brute-force) vector indexes.
#[derive(Debug)]
pub struct FlatVectorIndexReader {
    vectors: VectorStorage,
    /// `(doc_id, field_id)` per record; ids index [`Self::field_dict`]
    /// (Issue #633 PR-B — interned, no per-record heap `String`).
    vector_ids: Vec<(u64, u16)>,
    /// Per-segment field-name dictionary (synthesized at load for
    /// v1/v2 segments, taken from the header for v3).
    field_dict: Arc<[Arc<str>]>,
    dimension: usize,
    distance_metric: DistanceMetric,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
    /// Pre-built per-field doc-id list (`field_name → Arc<[u64]>`). Built
    /// once at load so `doc_ids_for_field` returns a refcount-shared
    /// slice without re-cloning `vector_ids`. #405.
    vector_ids_by_field: std::collections::HashMap<String, Arc<[u64]>>,
    /// Stage 2 rerank sidecar pool (Issue #481, extended to Flat by
    /// #650 PR-2 / #932). `Some` only when the `.f32` sidecar exists and
    /// the loading mode is Eager; absence keeps Stage 1 behavior.
    rerank_storage: Option<Arc<crate::vector::index::rerank_storage::RerankStoragePool>>,
}

/// Group `vector_ids` by field name into refcount-shared slices.
fn build_vector_ids_by_field(
    vector_ids: &[(u64, u16)],
    field_dict: &[Arc<str>],
) -> std::collections::HashMap<String, Arc<[u64]>> {
    let mut by_field: Vec<Vec<u64>> = vec![Vec::new(); field_dict.len()];
    for &(doc_id, fid) in vector_ids {
        by_field[fid as usize].push(doc_id);
    }
    field_dict
        .iter()
        .zip(by_field)
        .map(|(field, ids)| (field.to_string(), Arc::<[u64]>::from(ids)))
        .collect()
}

impl FlatVectorIndexReader {
    /// Create a reader from serialized bytes.
    pub fn from_bytes(_data: &[u8]) -> Result<Self> {
        Err(LaurusError::InvalidOperation(
            "from_bytes is deprecated, use load() instead".to_string(),
        ))
    }

    /// Load a flat vector index from storage.
    ///
    /// # Arguments
    ///
    /// * `storage` - Shared storage backend (cloned into `OnDemand` for concurrent reads).
    /// * `path` - Base path/name for the index file (`.flat` extension is appended).
    /// * `distance_metric` - Distance metric used for similarity computations.
    ///
    /// # Returns
    ///
    /// A new `FlatIndexReader` instance.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on I/O or format errors.
    pub fn load(
        storage: Arc<dyn Storage>,
        path: &str,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        use crate::util::alloc_bounds::checked_capacity;
        use std::io::{Read, Seek};

        // Open the index file
        let file_name = format!("{}.flat", path);
        let mut input = storage.open_input(&file_name)?;

        // Ground truth for bounding allocations sized from the unverified
        // header counts below (Issue #806). The `.flat` reader has no
        // pre-parse checksum, so every header count reaches its allocation
        // unverified.
        let file_size = input.size()?;

        // Read metadata
        let mut num_vectors_buf = [0u8; 4];
        input.read_exact(&mut num_vectors_buf)?;
        let num_vectors = u32::from_le_bytes(num_vectors_buf) as usize;

        let mut dimension_buf = [0u8; 4];
        input.read_exact(&mut dimension_buf)?;
        let dimension = u32::from_le_bytes(dimension_buf) as usize;

        // Read the Issue #481 Stage 1 vector segment header (LVS1).
        // Pre-Stage-1 segments are rejected with IncompatibleFormat.
        // Matched by reference so `header` (version + field dictionary,
        // Issue #633) stays alive for the record parse below.
        // Issue #921: pass the bytes physically left in the file so the
        // header's PQ codebook allocation is bounded before it reserves.
        let header_available =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        let header = VectorSegmentHeader::read_from(&mut input, header_available)?;
        let params = match &header.quant {
            QuantHeader::Scalar8Bit(p) => *p,
            QuantHeader::ProductQuantization { .. } => {
                return Err(crate::error::LaurusError::NotImplemented(
                    "Product quantization (Issue #481 Stage 3) is HNSW-only; \
                     the Flat reader does not support PQ segments yet"
                        .to_string(),
                ));
            }
            #[cfg(feature = "pq-fastscan")]
            QuantHeader::ProductQuantizationFastScan { .. } => {
                return Err(crate::error::LaurusError::NotImplemented(
                    "PQ FastScan (#695) is HNSW-only; the Flat reader does not \
                     support PQ FastScan segments"
                        .to_string(),
                ));
            }
        };

        // Bytes left for the per-vector records section, captured once at its
        // start (Issue #806). Each record is at least the version-dependent
        // prefix (doc_id + field reference, Issue #633) + the fixed quantized
        // payload (dim int8 + 8 meta), so this stride also bounds the
        // per-record `dimension`-sized int8 read.
        let records_remaining =
            file_size.saturating_sub(input.stream_position().map_err(LaurusError::Io)?);
        let record_stride =
            record_prefix_size(header.version) + quantized_record_payload_size(dimension) as u64;

        // Interned field ids (Issue #633 PR-B): one shared dictionary per
        // segment instead of one heap `String` per record.
        let mut interner = FieldInterner::from_header(&header);

        let (vectors, vector_ids, field_dict) = match storage.loading_mode() {
            crate::storage::LoadingMode::Eager => {
                // Step 7 of #481 Stage 1: load vectors as int8 + meta
                // directly into a QuantizedVectorPool.
                checked_capacity(
                    num_vectors,
                    record_stride,
                    records_remaining,
                    "flat num_vectors",
                )?;
                let mut vector_ids = Vec::with_capacity(num_vectors);
                let mut records: Vec<(u64, String, Vec<u8>, QuantizedVectorMeta)> =
                    Vec::with_capacity(num_vectors);

                for _ in 0..num_vectors {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    // Field reference: interned id (v3 = dictionary id,
                    // v1/v2 = inline name interned on first appearance).
                    let fid = interner.read_record_field_id(
                        &header,
                        &mut input,
                        records_remaining,
                        "flat field_name_len",
                    )?;

                    // Read int8 + meta directly (no dequantize).
                    let mut int8 = vec![0u8; dimension];
                    input.read_exact(&mut int8)?;
                    let mut sum_q_buf = [0u8; 4];
                    let mut norm_q_buf = [0u8; 4];
                    input.read_exact(&mut sum_q_buf)?;
                    input.read_exact(&mut norm_q_buf)?;
                    let meta = QuantizedVectorMeta {
                        sum_q: u32::from_le_bytes(sum_q_buf),
                        norm_q: f32::from_le_bytes(norm_q_buf),
                    };

                    vector_ids.push((doc_id, fid));
                    // The pool's build input keeps the String shape; this
                    // clone is transient (the pool retains only per-field
                    // grouping keys), so nothing per-record is kept.
                    records.push((doc_id, interner.name(fid).to_string(), int8, meta));
                }
                let pool = QuantizedVectorPool::build(params, dimension, records);
                (
                    VectorStorage::OwnedQuantized(Arc::new(pool)),
                    vector_ids,
                    interner.into_dict(),
                )
            }
            crate::storage::LoadingMode::Lazy => {
                checked_capacity(
                    num_vectors,
                    record_stride,
                    records_remaining,
                    "flat num_vectors",
                )?;
                let mut offsets = HashMap::with_capacity(num_vectors);
                let mut vector_ids = Vec::with_capacity(num_vectors);

                // Seek to the start of the per-vector entries: Flat preamble
                // (count u32 + dim u32 = 8 bytes) + the parsed header's real
                // size (which includes the v3 field dictionary, Issue #633 —
                // reconstructing a fresh header here would omit it).
                let start_pos = 8u64 + header.serialized_size() as u64;
                input
                    .seek(std::io::SeekFrom::Start(start_pos))
                    .map_err(LaurusError::Io)?;

                let quant_payload_size = quantized_record_payload_size(dimension) as i64;

                for _ in 0..num_vectors {
                    let mut doc_id_buf = [0u8; 8];
                    input.read_exact(&mut doc_id_buf)?;
                    let doc_id = u64::from_le_bytes(doc_id_buf);

                    let fid = interner.read_record_field_id(
                        &header,
                        &mut input,
                        records_remaining,
                        "flat field_name_len",
                    )?;

                    // Offsets point at the payload start (right after the
                    // record prefix), so `VectorStorage::get` seeks straight
                    // to the int8 data without re-parsing the prefix.
                    let payload_offset = input.stream_position().map_err(LaurusError::Io)?;
                    offsets.insert((doc_id, fid), payload_offset);
                    vector_ids.push((doc_id, fid));

                    // Skip int8 payload + per-vector meta.
                    input
                        .seek(std::io::SeekFrom::Current(quant_payload_size))
                        .map_err(LaurusError::Io)?;
                }

                let field_dict = interner.into_dict();
                (
                    VectorStorage::OnDemand {
                        storage: storage.clone(),
                        file_name: file_name.clone(),
                        offsets: Arc::new(offsets),
                        field_dict: field_dict.clone(),
                        quant_params: Some(params),
                        cached_input: Arc::new(std::sync::RwLock::new(None)),
                    },
                    vector_ids,
                    field_dict,
                )
            }
        };

        let vector_ids_by_field = build_vector_ids_by_field(&vector_ids, &field_dict);

        // Stage 2 rerank sidecar (Issue #481, extended to Flat by #650
        // PR-2 / #932): loaded eagerly when present, mirroring HNSW. The
        // pool's positions pair with `vector_ids` (the record order the
        // writer also used for the sidecar payload) — an identity mapping.
        // Lazy mode skips the sidecar to honor its memory-savings promise.
        let rerank_storage = crate::vector::index::rerank_sidecar::load_rerank_sidecar(
            storage.as_ref(),
            &file_name,
            dimension,
            &vector_ids,
            &field_dict,
        )?;

        Ok(Self {
            vectors,
            vector_ids,
            field_dict,
            dimension,
            distance_metric,
            deletion_bitmap: None,
            vector_ids_by_field,
            rerank_storage,
        })
    }

    /// Borrow the optional Stage 2 rerank storage pool (#932).
    pub fn rerank_storage(
        &self,
    ) -> Option<&Arc<crate::vector::index::rerank_storage::RerankStoragePool>> {
        self.rerank_storage.as_ref()
    }

    pub fn set_deletion_bitmap(&mut self, bitmap: Arc<DeletionBitmap>) {
        self.deletion_bitmap = Some(bitmap);
    }

    /// Borrow the underlying [`VectorStorage`] so the Flat searcher
    /// can detect the [`VectorStorage::OwnedQuantized`] variant and
    /// switch to the int8 hot path (Issue #481 Stage 1 Step 7).
    pub fn vectors(&self) -> &VectorStorage {
        &self.vectors
    }

    /// Whether `doc_id` is logically deleted per the attached bitmap (if
    /// any). `pub(crate)` so the searcher's per-storage-mode fast paths
    /// (e.g. the quantized-pool scan in [`crate::vector::index::flat::searcher`],
    /// which reads straight from the quantized pool and bypasses
    /// [`VectorIndexReader::get_vector`]) can filter deleted docs without
    /// going through that slower, allocation-heavy accessor.
    pub(crate) fn is_deleted(&self, doc_id: u64) -> bool {
        if let Some(bitmap) = &self.deletion_bitmap {
            bitmap.is_deleted(doc_id)
        } else {
            false
        }
    }
}

impl VectorIndexReader for FlatVectorIndexReader {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        if self.is_deleted(doc_id) {
            return Ok(None);
        }
        self.vectors.get(doc_id, field_name, self.dimension)
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        let mut result = Vec::new();
        for &(id, fid) in &self.vector_ids {
            let field = &self.field_dict[fid as usize];
            if id == doc_id
                && !self.is_deleted(id)
                && let Some(vec) = self.vectors.get(id, field, self.dimension)?
            {
                result.push((field.to_string(), vec));
            }
        }
        Ok(result)
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        let mut result = Vec::with_capacity(doc_ids.len());
        for (id, field) in doc_ids {
            if self.is_deleted(*id) {
                result.push(None);
            } else {
                result.push(self.vectors.get(*id, field, self.dimension)?);
            }
        }
        Ok(result)
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
        // Rehydrated at the trait boundary (Issue #633 PR-B): internal
        // state is `(u64, u16)` + the shared dictionary.
        Ok(self
            .vector_ids
            .iter()
            .map(|&(id, fid)| (id, self.field_dict[fid as usize].to_string()))
            .collect())
    }

    fn doc_ids_for_field(&self, field_name: &str) -> Arc<[u64]> {
        // O(1) HashMap lookup + Arc clone (refcount bump). Default impl
        // would clone `vector_ids` (Vec<(u64, String)>) and filter
        // linearly per call. #405.
        self.vector_ids_by_field
            .get(field_name)
            .cloned()
            .unwrap_or_else(|| Vec::<u64>::new().into())
    }

    fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    fn stats(&self) -> VectorStats {
        let _memory_usage = match &self.vectors {
            VectorStorage::Owned(vectors) => vectors.len() * (8 + self.dimension * 4),
            VectorStorage::OwnedQuantized(pool) => pool.heap_size(),
            VectorStorage::OwnedPq(pool) => pool.data.len() + pool.codebook.len() * 4,
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(_) => {
                unreachable!("Flat reader rejects PQ FastScan at the segment header (HNSW-only)")
            }
            VectorStorage::OnDemand { offsets, .. } => {
                // Estimate memory for offsets map + ID list
                offsets.len() * (8 + 32 + 8) // Key + Valid + Offset roughly
            }
        };
        VectorStats {
            vector_count: self.vectors.len(),
            dimension: self.dimension,
            memory_usage: self.vectors.len() * (8 + self.dimension * 4),
            build_time_ms: 0,
        }
    }

    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool {
        self.vectors.contains(doc_id, field_name)
    }

    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>> {
        let mut result = Vec::new();
        for &(id, fid) in &self.vector_ids {
            let field = &self.field_dict[fid as usize];
            if id >= start_doc_id
                && id < end_doc_id
                && !self.is_deleted(id)
                && let Some(vec) = self.vectors.get(id, field, self.dimension)?
            {
                result.push((id, field.to_string(), vec));
            }
        }
        Ok(result)
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        // One dictionary resolve, then integer compares per record.
        let Some(target) =
            crate::vector::index::format::resolve_field_id(&self.field_dict, field_name)
        else {
            return Ok(Vec::new());
        };
        let mut result = Vec::new();
        for &(id, fid) in &self.vector_ids {
            if fid == target
                && !self.is_deleted(id)
                && let Some(vec) = self.vectors.get(id, field_name, self.dimension)?
            {
                result.push((id, vec));
            }
        }
        Ok(result)
    }

    fn field_names(&self) -> Result<Vec<String>> {
        Ok(self.field_dict.iter().map(|f| f.to_string()).collect())
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        Ok(Box::new(FlatVectorIterator {
            storage: self.vectors.clone(),
            keys: self.vector_ids.clone(),
            field_dict: self.field_dict.clone(),
            current: 0,
            dimension: self.dimension,
            deletion_bitmap: self.deletion_bitmap.clone(),
        }))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        Ok(VectorIndexMetadata {
            index_type: "flat".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            version: "1".to_string(),
            build_config: serde_json::json!({}),
            custom_metadata: std::collections::HashMap::new(),
        })
    }

    fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        if self.vector_ids.len() != self.vectors.len() {
            errors.push(format!(
                "Mismatch between vector_ids count ({}) and vectors count ({})",
                self.vector_ids.len(),
                self.vectors.len()
            ));
        }

        match &self.vectors {
            VectorStorage::Owned(map) => {
                for &(id, fid) in &self.vector_ids {
                    let field = &self.field_dict[fid as usize];
                    let (id, field) = (&id, &field.to_string());
                    if let Some(vector) = map.get(&(*id, field.clone())) {
                        if vector.dimension() != self.dimension {
                            errors.push(format!(
                                "Vector {}:{} has dimension {}, expected {}",
                                id,
                                field,
                                vector.dimension(),
                                self.dimension
                            ));
                        }
                        if !vector.is_valid() {
                            errors.push(format!(
                                "Vector {}:{} contains invalid values (NaN or infinity)",
                                id, field
                            ));
                        }
                    } else {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in storage",
                            id, field
                        ));
                    }
                }
            }
            VectorStorage::OwnedQuantized(pool) => {
                for &(id, fid) in &self.vector_ids {
                    let field = &self.field_dict[fid as usize];
                    let id = &id;
                    if !pool.contains(*id, field) {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in quantized pool",
                            id, field
                        ));
                    }
                }
                warnings.push(
                    "OwnedQuantized mode: dimension / NaN checks skipped (int8 storage \
                     guarantees finite values within [offset, offset + 255*scale])"
                        .to_string(),
                );
            }
            VectorStorage::OwnedPq(pool) => {
                for &(id, fid) in &self.vector_ids {
                    let field = &self.field_dict[fid as usize];
                    let id = &id;
                    if !pool.contains(*id, field) {
                        errors.push(format!(
                            "Vector {}:{} found in keys but missing in PQ pool",
                            id, field
                        ));
                    }
                }
                warnings.push(
                    "OwnedPq mode: dimension / NaN checks skipped (codes index into \
                     the trained codebook which is bounded by construction)"
                        .to_string(),
                );
            }
            #[cfg(feature = "pq-fastscan")]
            VectorStorage::OwnedPqFastScan(_) => {
                unreachable!("Flat reader rejects PQ FastScan at the segment header (HNSW-only)")
            }
            VectorStorage::OnDemand { offsets, .. } => {
                for &(id, fid) in &self.vector_ids {
                    let field = &self.field_dict[fid as usize];
                    let id = &id;
                    if !offsets.contains_key(&(*id, fid)) {
                        errors.push(format!(
                            "Vector {}:{} in ids but missing in storage",
                            id, field
                        ));
                    }
                }
                warnings.push("OnDemand mode: Deep vector validation skipped".to_string());
            }
        }

        Ok(ValidationReport {
            repair_suggestions: Vec::new(),
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }
}

/// Iterator for flat vector index.
struct FlatVectorIterator {
    storage: VectorStorage,
    keys: Vec<(u64, u16)>,
    field_dict: Arc<[Arc<str>]>,
    current: usize,
    dimension: usize,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
}

impl VectorIterator for FlatVectorIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        // Use a loop instead of recursion to avoid stack overflow when
        // many consecutive entries are deleted.
        while self.current < self.keys.len() {
            let (doc_id, fid) = self.keys[self.current];
            let field = &self.field_dict[fid as usize];

            // Skip deleted entries
            if let Some(bitmap) = &self.deletion_bitmap
                && bitmap.is_deleted(doc_id)
            {
                self.current += 1;
                continue;
            }

            if let Some(vec) = self.storage.get(doc_id, field, self.dimension)? {
                self.current += 1;
                return Ok(Some((doc_id, field.to_string(), vec)));
            } else {
                return Err(LaurusError::internal(format!(
                    "Vector {}:{} found in keys but missing in storage",
                    doc_id, field
                )));
            }
        }

        Ok(None)
    }

    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool> {
        while self.current < self.keys.len() {
            let (id, fid) = self.keys[self.current];
            let field = &self.field_dict[fid as usize];
            if id > doc_id || (id == doc_id && field.as_ref() as &str >= field_name) {
                return Ok(true);
            }
            self.current += 1;
        }
        Ok(false)
    }

    fn position(&self) -> (u64, String) {
        if self.current < self.keys.len() {
            let (id, fid) = self.keys[self.current];
            (id, self.field_dict[fid as usize].to_string())
        } else {
            (u64::MAX, String::new())
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.current = 0;
        Ok(())
    }
}

#[cfg(test)]
mod alloc_bound_tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::core::quantization::ScalarQuantParams;
    use std::io::Write;

    /// Build an in-memory storage holding `bytes` under `name`.
    fn storage_with(name: &str, bytes: Vec<u8>) -> Arc<dyn Storage> {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut out = storage.create_output(name).unwrap();
        out.write_all(&bytes).unwrap();
        out.flush_and_sync().unwrap();
        Arc::new(storage)
    }

    /// Serialized neutral LVS1 (Scalar8Bit) header bytes.
    fn neutral_header_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        VectorSegmentHeader::scalar_8bit(ScalarQuantParams {
            offset: 0.0,
            scale: 1.0,
        })
        .write_to(&mut buf)
        .unwrap();
        buf
    }

    #[test]
    fn load_rejects_oversized_num_vectors_without_aborting() {
        // A `.flat` segment whose `num_vectors` field is corrupted to a huge
        // value while the file holds no records must be rejected cleanly,
        // never drive a multi-GiB `Vec::with_capacity` that aborts the
        // process via `handle_alloc_error` (Issue #806).
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // num_vectors (corrupt)
        bytes.extend_from_slice(&4u32.to_le_bytes()); // dimension
        bytes.extend_from_slice(&neutral_header_bytes()); // LVS1 header, no records

        let storage = storage_with("corrupt.flat", bytes);
        let err = FlatVectorIndexReader::load(storage, "corrupt", DistanceMetric::Cosine)
            .expect_err("oversized num_vectors must be rejected as corruption");
        match err {
            LaurusError::Index(msg) => {
                assert!(msg.contains("num_vectors"), "got: {msg}");
                assert!(msg.contains("corrupted"), "got: {msg}");
            }
            other => panic!("expected Index error, got {other:?}"),
        }
    }
}
