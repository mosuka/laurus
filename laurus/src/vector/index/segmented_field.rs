use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::vector::core::field::FieldOption;
use crate::vector::core::vector::{StoredVector, Vector};
use crate::vector::index::VectorIndexWriter;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::field::{
    FieldHit, FieldSearchInput, FieldSearchResults, VectorField, VectorFieldReader,
    VectorFieldStats, VectorFieldWriter,
};
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::searcher::HnswSearcher;
use crate::vector::index::hnsw::segment::merge_engine::MergeEngine;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::index::segment::manager::{ManagedSegmentInfo, SegmentManager};
use crate::vector::index::segment::merge::MergeConfig;
use crate::vector::index::segment::reader_cache::SegmentedReaderCache;
use crate::vector::search::searcher::{
    VectorIndexQuery, VectorIndexQueryParams, VectorIndexSearcher,
};
use crate::vector::store::config::VectorFieldConfig;
use crate::vector::writer::VectorIndexWriterConfig;

/// A vector field implementation that partitions data into segments.
///
/// This implementation allows for efficient ingestion and background merging
/// of HNSW vector segments.
#[derive(Debug, Clone)]
pub struct SegmentedVectorField {
    /// Field name.
    pub name: String,

    /// Field configuration.
    pub config: VectorFieldConfig,

    /// Manager for segments.
    pub segment_manager: Arc<SegmentManager>,

    /// Storage backend.
    pub storage: Arc<dyn Storage>,

    /// Active segment for current writes.
    pub active_segment: Arc<RwLock<Option<(String, HnswIndexWriter)>>>,

    /// Deletion bitmap shared by this field's sealed-segment readers and
    /// merge engine.
    ///
    /// **Must be scoped to this ONE field, never shared across fields**
    /// (Issue #880): the bitmap is doc-granular while the upsert dance is
    /// per-field — `delete_document` marks the id doc-wide and
    /// `add_stored_vector` un-marks it, so a bitmap shared between fields
    /// would let field A's re-add permanently revive field B's sealed copy
    /// when an upsert drops field B (nothing newer ever shadows it). The
    /// #634 PR-3 adapter must allocate one bitmap per field (a doc-level
    /// delete then marks every field's bitmap).
    pub deletion_bitmap: Option<Arc<DeletionBitmap>>,

    /// Per-segment [`HnswIndexReader`] cache used by
    /// [`Self::search_managed_segments`]. Without this cache every query
    /// reloaded every managed segment from disk; with it, the second and
    /// subsequent queries against the same segment hit memory only.
    ///
    /// Invalidated by [`Self::perform_merge_with_policy`] when source
    /// segments are removed as part of a merge. Issue
    /// [#660](https://github.com/mosuka/laurus/issues/660).
    pub reader_cache: Arc<SegmentedReaderCache<HnswIndexReader>>,
}

impl SegmentedVectorField {
    pub fn create(
        name: impl Into<String>,
        config: VectorFieldConfig,
        segment_manager: Arc<SegmentManager>,
        storage: Arc<dyn Storage>,
        deletion_bitmap: Option<Arc<DeletionBitmap>>,
    ) -> Result<Self> {
        let name_str = name.into();

        // Validate config
        match &config.vector {
            Some(FieldOption::Hnsw(_)) => {}
            _ => {
                return Err(LaurusError::invalid_config(
                    "SegmentedVectorField requires HNSW configuration",
                ));
            }
        }

        let field = Self {
            name: name_str,
            config,
            segment_manager,
            storage,
            active_segment: Arc::new(RwLock::new(None)),
            deletion_bitmap,
            reader_cache: Arc::new(SegmentedReaderCache::new()),
        };

        Ok(field)
    }

    fn ensure_active_segment(&self) -> Result<()> {
        // ... same as before

        // Optimistic check
        if self.active_segment.read().is_some() {
            return Ok(());
        }

        let mut active_lock = self.active_segment.write();
        if active_lock.is_some() {
            return Ok(());
        }

        // Create new active segment
        let segment_id = self.segment_manager.generate_segment_id();

        // Get HNSW parameters from config
        let opt = match &self.config.vector {
            Some(FieldOption::Hnsw(opt)) => opt,
            _ => {
                return Err(LaurusError::invalid_config(
                    "SegmentedVectorField requires HNSW configuration".to_string(),
                ));
            }
        };

        // Option-derived fields (incl. rerank_storage, the quantizer, and
        // metric-conditional normalize_vectors) come from the shared
        // conversion helper (Issues #790 / #794).
        let hnsw_config = HnswIndexConfig::from_hnsw_option(opt);

        let writer_config = VectorIndexWriterConfig {
            ..Default::default()
        };

        let writer = HnswIndexWriter::with_storage(
            hnsw_config,
            writer_config,
            &segment_id,
            self.storage.clone(),
        )?;
        *active_lock = Some((segment_id, writer));

        Ok(())
    }

    /// Trigger a background merge of segments using various policies.
    pub fn perform_merge(&self) -> Result<()> {
        let policy = crate::vector::index::segment::merge_policy::SimpleMergePolicy::new();
        self.perform_merge_with_policy(&policy)
    }

    /// Trigger a merge with a specific policy.
    pub fn perform_merge_with_policy(
        &self,
        policy: &dyn crate::vector::index::segment::merge_policy::MergePolicy,
    ) -> Result<()> {
        if let Some(mut candidate) = self.segment_manager.check_merge(policy) {
            // Close generation gaps in the candidate (Issue #880): the merged
            // segment inherits max(source generations), which is only correct
            // when no NON-source segment's generation falls strictly inside
            // the candidate's generation range — otherwise a stale copy from
            // an old source could be laundered above that segment under
            // newest-generation-wins dedup. Policies are free to pick any
            // set; this expansion restores the contiguity invariant.
            {
                let min_gen = candidate.segments.iter().map(|s| s.generation).min();
                let max_gen = candidate.segments.iter().map(|s| s.generation).max();
                if let (Some(min_gen), Some(max_gen)) = (min_gen, max_gen) {
                    for info in self.segment_manager.list_segments() {
                        let inside = info.generation > min_gen && info.generation < max_gen;
                        let already = candidate
                            .segments
                            .iter()
                            .any(|s| s.segment_id == info.segment_id);
                        if inside && !already {
                            candidate.total_vectors += info.vector_count;
                            candidate.total_size += info.size_bytes;
                            candidate.segments.push(info);
                        }
                    }
                }
            }
            let opt = match &self.config.vector {
                Some(FieldOption::Hnsw(opt)) => opt,
                _ => {
                    return Err(LaurusError::invalid_config(
                        "SegmentedVectorField requires HNSW configuration".to_string(),
                    ));
                }
            };

            // Option-derived fields come from the shared conversion
            // helper (Issues #790 / #794). The helper maps
            // `distance_metric` and the metric-conditional
            // `normalize_vectors` (normalize only for Cosine), so a
            // non-Cosine segmented field is merged with its own metric
            // and without the magnitude-corrupting normalization the
            // merge config used to apply via the always-on default.
            let mut engine = MergeEngine::new(
                MergeConfig::default(),
                self.storage.clone(),
                HnswIndexConfig::from_hnsw_option(opt),
                VectorIndexWriterConfig {
                    ..Default::default()
                },
            );

            if let Some(bitmap) = &self.deletion_bitmap {
                engine.set_deletion_bitmap(bitmap.clone());
            }

            let new_segment_id = self.segment_manager.generate_segment_id();
            let result =
                engine.merge_segments(candidate.segments.clone(), new_segment_id.clone())?;

            // Register the engine's own segment info (Issue #880): it carries
            // the inherited max(source generations). The previous hand-built
            // info hard-coded generation 0, which sorted the merged segment
            // as the OLDEST source — inverting newest-wins dedup so untouched
            // older segments shadowed the merged (newest) copies, and a
            // subsequent merge could permanently drop them as "duplicates".
            let info = result.merged_segment.clone();

            // Capture source segment ids before `apply_merge` consumes
            // `candidate`, so we can invalidate their cache entries below.
            // Issue #660: source readers are no longer reachable through the
            // manager after `apply_merge`; clearing their cache entries
            // prevents stale orphan readers from lingering in memory.
            let source_ids: Vec<String> = candidate
                .segments
                .iter()
                .map(|s| s.segment_id.clone())
                .collect();

            self.segment_manager.apply_merge(candidate, info)?;

            for id in &source_ids {
                self.reader_cache.invalidate(id);
            }
        }
        Ok(())
    }
}

#[async_trait]
impl VectorField for SegmentedVectorField {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &VectorFieldConfig {
        &self.config
    }

    fn writer(&self) -> &dyn VectorFieldWriter {
        self
    }

    fn reader(&self) -> &dyn VectorFieldReader {
        self
    }

    fn writer_handle(&self) -> Arc<dyn VectorFieldWriter> {
        Arc::new(self.clone())
    }

    fn reader_handle(&self) -> Arc<dyn VectorFieldReader> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl VectorFieldWriter for SegmentedVectorField {
    async fn add_stored_vector(
        &self,
        doc_id: u64,
        vector: &StoredVector,
        _version: u64,
    ) -> Result<()> {
        let vec = vector.to_vector();

        self.ensure_active_segment()?;
        let mut active_opt = self.active_segment.write();
        if let Some((_, writer)) = active_opt.as_mut() {
            writer.add_vectors(vec![(doc_id, self.name.clone(), vec)])?;
        } else {
            return Err(LaurusError::internal(
                "No active segment available".to_string(),
            ));
        }

        // Same-id upsert completion (Issue #880): the delete-first step
        // marked this id in the shared bitmap so sealed-segment copies stop
        // matching; clear the mark now so the NEW copy is not shadowed by
        // its own delete once this segment flushes. The revived old copies
        // are masked by newest-source-wins containment at search and removed
        // physically at merge.
        //
        // The bitmap flip happens while the `active_segment` write lock is
        // still held (lock order: active_segment → bitmap, same as
        // `delete_document`): releasing the lock first would let a racing
        // pure delete interleave between the buffer insert and this
        // undelete, whose mark this undelete would then erase — losing the
        // delete and resurrecting the stale sealed copy.
        if let Some(bitmap) = &self.deletion_bitmap {
            let _ = bitmap.undelete_document(doc_id)?;
        }
        drop(active_opt);
        Ok(())
    }

    async fn has_storage(&self) -> bool {
        self.active_segment
            .read()
            .as_ref()
            .map(|(_, w)| w.has_storage())
            .unwrap_or(false)
    }

    async fn vectors(&self) -> Vec<(u64, String, Vector)> {
        if let Some((_, writer)) = self.active_segment.read().as_ref() {
            writer.vectors().to_vec()
        } else {
            Vec::new()
        }
    }

    async fn rebuild(&self, _vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        Ok(())
    }

    async fn delete_document(&self, doc_id: u64, _version: u64) -> Result<()> {
        // The doc may live in an already-sealed segment, which the active
        // writer cannot touch — previously this was a silent no-op (Issue
        // #880). Mark the shared deletion bitmap: every managed-segment
        // reader is constructed with the same bitmap
        // (`search_managed_segments`), so the deletion is search-visible
        // immediately, and the merge engine filters through it as well.
        //
        // Both steps run under the `active_segment` write lock (lock order:
        // active_segment → bitmap, same as `add_stored_vector`) so a racing
        // upsert cannot interleave between them, and the bitmap is marked
        // BEFORE the buffered copy is removed — sealed-segment searches do
        // not take this lock, so the reverse order would open a window where
        // the active copy is gone but the stale sealed copy is not yet
        // masked.
        let mut newly_deleted = false;
        {
            let mut active_opt = self.active_segment.write();
            if let Some(bitmap) = &self.deletion_bitmap {
                newly_deleted = bitmap.delete_document(doc_id)?;
            }
            // Remove from the active (unflushed) buffer.
            if let Some((_, writer)) = active_opt.as_mut() {
                let _ = writer.delete_document(doc_id);
            }
        }

        // Flag `has_deletions` so the merge policy prioritizes the affected
        // segments (outside the active lock — it persists the manifest).
        if newly_deleted && !self.segment_manager.list_segments().is_empty() {
            self.segment_manager.mark_all_has_deletions()?;
        }
        Ok(())
    }

    async fn flush(&self) -> Result<()> {
        let mut active_lock = self.active_segment.write();
        if let Some((segment_id, mut writer)) = active_lock.take() {
            writer.finalize()?;
            writer.write()?;

            let vector_count = writer.vectors().len() as u64;

            let info = ManagedSegmentInfo::new(
                segment_id,
                vector_count,
                0, // offset
                0, // generation
            );

            self.segment_manager.add_segment(info)?;
        }
        Ok(())
    }

    async fn optimize(&self) -> Result<()> {
        let policy = crate::vector::index::segment::merge_policy::ForceMergePolicy::new();
        self.perform_merge_with_policy(&policy)
    }
}

impl SegmentedVectorField {
    /// Brute-force top-`limit` search over the active (unflushed)
    /// segment's buffered vectors (Issue #640).
    ///
    /// Mirrors the Flat searcher's scan: the query is prepared once per
    /// call (`prepare_query` caches `‖query‖²`, so Cosine / Angular run
    /// two SIMD accumulator chains per candidate instead of three), the
    /// per-candidate distance runs through
    /// [`crate::vector::search::searcher::parallel_scan`] (rayon when the
    /// buffer holds ≥ 2048 vectors, serial below), and top-`limit`
    /// selection uses `select_nth_unstable_by` instead of sorting the
    /// whole buffer.
    ///
    /// # Arguments
    ///
    /// * `query` - The f32 query vector.
    /// * `limit` - Maximum number of hits to return.
    /// * `weight` - Query weight multiplied into each hit's score.
    ///
    /// # Returns
    ///
    /// Up to `limit` hits ranked by similarity descending; empty when
    /// there is no active segment (or `limit == 0`).
    /// The caller passes the active writer borrowed from an already-held
    /// `active_segment` guard, so this helper takes no lock of its own —
    /// `search` holds one read guard across the whole request, both for
    /// scanning and for the newest-source containment probes (Issue #880).
    fn search_active_segment(
        &self,
        writer: &HnswIndexWriter,
        query: &[f32],
        limit: usize,
        weight: f32,
    ) -> Result<Vec<FieldHit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        // Safe unwrap because verified in create()
        let distance_metric = match &self.config.vector {
            Some(FieldOption::Hnsw(opt)) => opt.distance,
            _ => return Ok(Vec::new()), // Should not happen
        };

        let vectors = writer.vectors();
        // Prepare once per query: caches the query norm for Cosine /
        // Angular; the other metrics pass through unchanged.
        let prepared = distance_metric.prepare_query(query);

        let mut candidates: Vec<(u64, f32, f32)> =
            crate::vector::search::searcher::parallel_scan(vectors, |(doc_id, _field, vector)| {
                let distance = distance_metric.distance_with_prepared(&prepared, &vector.data)?;
                let similarity = distance_metric.distance_to_similarity(distance);
                Ok(Some((*doc_id, similarity, distance)))
            })?;

        // Partial top-`limit` selection, then sort only the survivors
        // (same pattern as the store-level search).
        if candidates.len() > limit {
            candidates.select_nth_unstable_by(limit - 1, |a, b| b.1.total_cmp(&a.1));
            candidates.truncate(limit);
        }
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let hits = candidates
            .into_iter()
            .map(|(doc_id, similarity, distance)| FieldHit {
                doc_id,
                field: self.name.clone(),
                score: similarity * weight,
                distance,
            })
            .collect();

        Ok(hits)
    }

    /// Load the readers of every sealed segment, newest generation first
    /// (Issue #660 cache; entries invalidated on merge).
    fn sealed_readers_newest_first(&self) -> Result<Vec<Arc<HnswIndexReader>>> {
        let mut segments = self.segment_manager.list_segments();
        // Newest generation first: earlier entries are authoritative for
        // newest-source-wins masking.
        segments.sort_by_key(|s| std::cmp::Reverse(s.generation));

        // Safe unwrap because verified in create()
        let distance_metric = match &self.config.vector {
            Some(FieldOption::Hnsw(opt)) => opt.distance,
            _ => return Ok(Vec::new()),
        };

        segments
            .into_iter()
            .map(|info| {
                let storage = self.storage.clone();
                let deletion_bitmap = self.deletion_bitmap.clone();
                let segment_id = info.segment_id;
                self.reader_cache.get_or_load(&segment_id, || {
                    let mut r = HnswIndexReader::load(storage, &segment_id, distance_metric)?;
                    if let Some(bitmap) = deletion_bitmap {
                        r.set_deletion_bitmap(bitmap);
                    }
                    Ok(r)
                })
            })
            .collect()
    }

    /// Search every sealed segment, masking each hit against every NEWER
    /// source by **containment** (Issue #880).
    ///
    /// Same-id upserts replayed from the WAL leave stale copies in older
    /// segments. Masking must be containment-based — "does any newer source
    /// HOLD this doc?" — not based on which hits newer sources returned:
    /// after an upsert, an old-embedding query rarely ranks the NEW copy
    /// into its own segment's top-k, so a returned-hits mask would let the
    /// stale exact-match copy through (and pre-#880 it was even scored once
    /// per copy).
    ///
    /// `readers` must be ordered newest generation first (from
    /// [`Self::sealed_readers_newest_first`]); `active_writer` is the
    /// still-newer unflushed buffer, borrowed from the guard `search` holds.
    fn search_managed_segments(
        &self,
        readers: &[Arc<HnswIndexReader>],
        active_writer: Option<&HnswIndexWriter>,
        query: &[f32],
        limit: usize,
        weight: f32,
    ) -> Result<Vec<FieldHit>> {
        let mut all_hits = Vec::new();

        for (idx, reader) in readers.iter().enumerate() {
            // Issue #644: prefer the schema-level `default_ef_search` when
            // the user has configured one. Otherwise fall back to the legacy
            // segmented-field heuristic of `ef_construction.max(50) * 2`
            // (which kept Round-1 / Round-2 multi-segment recall stable).
            // Per-query `VectorIndexQueryParams.ef_search` overrides both.
            let default_ef = if let Some(FieldOption::Hnsw(opt)) = &self.config.vector {
                opt.default_ef_search
                    .unwrap_or_else(|| opt.ef_construction.max(50) * 2)
            } else {
                50
            };
            let searcher = HnswSearcher::with_default_ef_search(reader.clone(), Some(default_ef))?;

            // Over-fetch per segment (Issue #880): containment masking
            // discards shadowed hits AFTER the per-segment top-k selection,
            // so stale copies would otherwise consume result slots and push
            // live docs out. Doubling the fetch bounds the loss cheaply; a
            // fully adaptive refill is left to the #634 PR-3 adapter.
            let params = VectorIndexQueryParams {
                top_k: limit.saturating_mul(2),
                ..Default::default()
            };

            let request = VectorIndexQuery {
                query: Vector::new(query.to_vec()),
                params,
                field_name: Some(self.name.clone()),
                // SegmentedVectorField is not on the VectorStore::search path
                // and does not yet thread the filter through (Issue #645 is
                // scoped to the main HNSW path). Filtering for this path stays
                // a post-filter concern.
                filter: None,
            };

            let results = searcher.search(&request)?;
            for res in results.results {
                // Newest-source-wins: drop the hit when ANY newer source
                // holds a copy of this doc — the active buffer, or a sealed
                // segment with a higher generation.
                let shadowed = active_writer.is_some_and(|w| w.contains_doc(res.doc_id))
                    || readers[..idx]
                        .iter()
                        .any(|newer| newer.vectors().contains(res.doc_id, &self.name));
                if shadowed {
                    continue;
                }
                all_hits.push(FieldHit {
                    doc_id: res.doc_id,
                    field: self.name.clone(),
                    score: res.similarity * weight,
                    distance: res.distance,
                });
            }
        }

        Ok(all_hits)
    }
}

impl VectorFieldReader for SegmentedVectorField {
    fn search(&self, request: FieldSearchInput) -> Result<FieldSearchResults> {
        if request.field != self.name {
            return Err(LaurusError::invalid_argument(format!(
                "field mismatch: expected '{}', got '{}'",
                self.name, request.field
            )));
        }

        if request.query_vectors.is_empty() {
            return Ok(FieldSearchResults::default());
        }

        // Sealed readers are resolved once, newest generation first —
        // BEFORE taking the active guard: a cold load hits storage, and
        // holding the read guard across that I/O would stall every writer
        // (and, with parking_lot's writer priority, every later reader)
        // behind a slow disk (Issue #880).
        let readers = self.sealed_readers_newest_first()?;

        // One read guard across the rest of the request: the active buffer
        // is both scanned and used as the newest containment source, so it
        // must not change between the two (Issue #880).
        let active_guard = self.active_segment.read();

        let mut merged: HashMap<u64, FieldHit> = HashMap::new();

        for query in &request.query_vectors {
            // Deliberately just `query.weight`, NOT multiplied by the
            // field's `base_weight` (Issue #1084): that factor is applied
            // one layer up, in `VectorStore::search_impl`'s per-field
            // query-weight computation. Applying it here too — if this
            // type is ever wired back into the production search path —
            // would double-apply it.
            let effective_weight = query.weight;
            let query_vec = &query.vector.data;

            // Within ONE query, a doc id must be scored exactly once even
            // when copies of it live in several sources (the active buffer
            // plus K sealed segments — same-id WAL-replay upserts leave
            // stale older copies behind). The newest source wins by
            // CONTAINMENT (Issue #880): a sealed hit is dropped when any
            // newer source holds the doc at all — not merely when a newer
            // source happened to return it. Accumulation (`score +=`) is
            // reserved for the multi-QUERY case below, which is intentional
            // weighting.
            let mut per_query: Vec<FieldHit> = Vec::new();

            // 1. Search Active (the newest source — always wins)
            if let Some((_, writer)) = active_guard.as_ref() {
                per_query.extend(self.search_active_segment(
                    writer,
                    query_vec,
                    request.limit,
                    effective_weight,
                )?);
            }

            // 2. Search Managed (newest generation first, containment-masked
            //    against the active buffer and every newer segment)
            let managed_hits = self.search_managed_segments(
                &readers,
                active_guard.as_ref().map(|(_, w)| w),
                query_vec,
                request.limit,
                effective_weight,
            )?;
            per_query.extend(managed_hits);

            // 3. Fold this query's deduplicated hits into the multi-query
            //    accumulator.
            for hit in per_query {
                match merged.entry(hit.doc_id) {
                    Entry::Vacant(e) => {
                        e.insert(hit);
                    }
                    Entry::Occupied(mut e) => {
                        let entry = e.get_mut();
                        entry.score += hit.score;
                        entry.distance = entry.distance.min(hit.distance);
                    }
                }
            }
        }

        let mut hits: Vec<FieldHit> = merged.into_values().collect();
        hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        if hits.len() > request.limit {
            hits.truncate(request.limit);
        }

        Ok(FieldSearchResults { hits })
    }

    fn stats(&self) -> Result<VectorFieldStats> {
        let mut active_count = 0;
        if let Some((_, writer)) = self.active_segment.read().as_ref() {
            active_count = writer.vectors().len();
        }

        let manager_stats = self.segment_manager.stats();
        let managed_count = manager_stats.total_vectors;

        // Safe unwrap because verified in create()
        let dimension = match &self.config.vector {
            Some(FieldOption::Hnsw(opt)) => opt.dimension,
            _ => 0,
        };

        Ok(VectorFieldStats {
            vector_count: active_count + managed_count as usize,
            dimension,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::core::field::HnswOption;
    use crate::vector::index::segment::manager::SegmentManagerConfig;
    use crate::vector::store::request::QueryVector;

    fn field_with_bitmap() -> (
        SegmentedVectorField,
        Arc<DeletionBitmap>,
        Arc<SegmentManager>,
    ) {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = Arc::new(
            SegmentManager::new(
                SegmentManagerConfig {
                    max_segments: 100,
                    min_vectors_per_segment: 1,
                    ..Default::default()
                },
                storage.clone(),
                crate::vector::index::hnsw::segment::LAYOUT,
            )
            .unwrap(),
        );
        // Unbounded-range global bitmap, as the production wiring will use.
        let bitmap = Arc::new(DeletionBitmap::new("global".to_string(), 0, u64::MAX - 1));
        let config = VectorFieldConfig {
            vector: Some(FieldOption::Hnsw(HnswOption {
                dimension: 4,
                distance: crate::vector::core::distance::DistanceMetric::Euclidean,
                m: 16,
                ef_construction: 200,
                default_ef_search: None,
                base_weight: 1.0,
                quantizer: Default::default(),
                rerank_storage: None,
                embedder: None,
                pq_codebook_path: None,
            })),
            lexical: None,
        };
        let field = SegmentedVectorField::create(
            "embedding",
            config,
            manager.clone(),
            storage,
            Some(bitmap.clone()),
        )
        .unwrap();
        (field, bitmap, manager)
    }

    fn query(vector: Vec<f32>, limit: usize) -> FieldSearchInput {
        FieldSearchInput {
            field: "embedding".to_string(),
            query_vectors: vec![QueryVector {
                vector: Vector::new(vector),
                weight: 1.0,
                fields: None,
            }],
            limit,
            allowed_ids: None,
        }
    }

    /// #880: deleting a doc that lives in a SEALED segment must make it
    /// search-invisible (previously a silent no-op — only the active buffer
    /// was touched) and flag `has_deletions` for the merge policy.
    #[tokio::test]
    async fn sealed_segment_delete_is_search_invisible_and_flagged() {
        let (field, bitmap, manager) = field_with_bitmap();

        field
            .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
            .await
            .unwrap();
        field
            .add_stored_vector(2, &StoredVector::new(vec![0.0, 1.0, 0.0, 0.0]), 0)
            .await
            .unwrap();
        field.flush().await.unwrap();

        // Sanity: both docs are searchable from the sealed segment.
        let results = field.search(query(vec![1.0, 0.0, 0.0, 0.0], 2)).unwrap();
        assert_eq!(results.hits.len(), 2);

        // Delete doc 1 — it lives ONLY in the sealed segment.
        field.delete_document(1, 0).await.unwrap();

        assert!(bitmap.is_deleted(1), "the shared bitmap must be marked");
        let results = field.search(query(vec![1.0, 0.0, 0.0, 0.0], 2)).unwrap();
        assert!(
            results.hits.iter().all(|h| h.doc_id != 1),
            "a sealed-segment doc must be search-invisible after delete \
             (#880), got {:?}",
            results.hits
        );
        assert!(
            manager.list_segments().iter().all(|s| s.has_deletions),
            "has_deletions must be flagged for the merge policy (#880)"
        );

        // The deletion also survives a force merge (the merge engine filters
        // through the same bitmap).
        field
            .add_stored_vector(3, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
            .await
            .unwrap();
        field.flush().await.unwrap();
        VectorFieldWriter::optimize(&field).await.unwrap();
        let results = field.search(query(vec![1.0, 0.0, 0.0, 0.0], 3)).unwrap();
        assert!(
            results.hits.iter().all(|h| h.doc_id != 1),
            "the deletion must survive a merge (#880), got {:?}",
            results.hits
        );
    }

    /// #880: the same-id upsert dance — delete marks the bitmap (masking the
    /// sealed old copy), the re-add clears the mark so the NEW copy is not
    /// shadowed by its own delete once flushed. End-to-end: after flush, the
    /// doc is searchable via its newest copy only.
    #[tokio::test]
    async fn same_id_upsert_readd_clears_bitmap_and_stays_searchable() {
        let (field, bitmap, _manager) = field_with_bitmap();

        field
            .add_stored_vector(1, &StoredVector::new(vec![1.0, 0.0, 0.0, 0.0]), 0)
            .await
            .unwrap();
        field.flush().await.unwrap();

        // Upsert: delete-first masks the sealed copy...
        field.delete_document(1, 0).await.unwrap();
        assert!(bitmap.is_deleted(1));
        // ...and the re-add clears the mark for the new copy.
        field
            .add_stored_vector(1, &StoredVector::new(vec![0.0, 0.0, 1.0, 0.0]), 0)
            .await
            .unwrap();
        assert!(
            !bitmap.is_deleted(1),
            "the re-add must clear the delete mark so the new copy is not \
             shadowed after flush (#880)"
        );
        field.flush().await.unwrap();

        // The doc resolves via its NEWEST copy (newest-wins dedup masks the
        // revived old copy).
        let results = field.search(query(vec![0.0, 0.0, 1.0, 0.0], 1)).unwrap();
        assert_eq!(results.hits[0].doc_id, 1);
        assert!(results.hits[0].distance < 1e-3);
        let results = field.search(query(vec![1.0, 0.0, 0.0, 0.0], 1)).unwrap();
        assert!(
            results.hits[0].doc_id != 1 || results.hits[0].distance > 1.0,
            "the stale old copy must stay shadowed (#880), got {:?}",
            results.hits
        );
    }
}
