//! Shared multi-segment fan-out search layer for segment-per-commit vector
//! indexes (Issue #889; originated in HNSW's segment-per-commit design,
//! Issues #634/#880/#883).
//!
//! [`SegmentedReaderFacade`] and [`SegmentFanoutSearcher`] operate purely
//! through the [`VectorIndexReader`]/[`VectorIndexSearcher`] trait objects —
//! every per-reader call they make (`contains_vector`, `get_vector`,
//! `vector_ids`, ...) is already a trait method, so the fan-out logic itself
//! needs no index-type-specific code. Only the per-segment concrete searcher
//! construction differs between index types, which is why
//! [`SegmentFanoutSearcher`] takes it as a closure rather than hard-coding a
//! concrete searcher type.

use std::sync::Arc;

use crate::error::Result;
use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;
use crate::vector::reader::{
    ValidationReport, VectorIndexMetadata, VectorIndexReader, VectorIterator, VectorStats,
};
use crate::vector::search::searcher::{
    VectorIndexQuery, VectorIndexQueryResults, VectorIndexSearcher,
};

/// Read facade over a segment-per-commit index's sealed segments
/// (newest-wins, deletion-filtered).
///
/// Materializes the distinct live `(doc_id, field) -> segment` mapping once
/// at construction; per-vector data is fetched from the owning segment
/// reader on demand.
#[derive(Debug)]
pub struct SegmentedReaderFacade {
    /// Sealed readers, newest generation first.
    readers: Vec<Arc<dyn VectorIndexReader>>,
    /// Distinct live keys with the owning (newest) reader's index, in
    /// first-seen (newest-segment) order.
    entries: Vec<(u64, String, usize)>,
    dimension: usize,
    metric: DistanceMetric,
}

impl SegmentedReaderFacade {
    /// Build the facade over `readers` (already ordered newest-generation
    /// first by the caller), filtering out anything `bitmap` marks deleted.
    pub fn new(
        readers: Vec<Arc<dyn VectorIndexReader>>,
        bitmap: Option<Arc<DeletionBitmap>>,
        dimension: usize,
        metric: DistanceMetric,
    ) -> Self {
        let mut seen: std::collections::HashSet<(u64, String)> = std::collections::HashSet::new();
        let mut entries = Vec::new();
        for (idx, reader) in readers.iter().enumerate() {
            if let Ok(ids) = reader.vector_ids() {
                for (doc_id, field) in ids {
                    if let Some(b) = &bitmap
                        && b.is_deleted(doc_id)
                    {
                        continue;
                    }
                    if seen.insert((doc_id, field.clone())) {
                        entries.push((doc_id, field, idx));
                    }
                }
            }
        }
        Self {
            readers,
            entries,
            dimension,
            metric,
        }
    }

    fn owner_of(&self, doc_id: u64, field_name: &str) -> Option<&Arc<dyn VectorIndexReader>> {
        self.entries
            .iter()
            .find(|(d, f, _)| *d == doc_id && f == field_name)
            .map(|(_, _, idx)| &self.readers[*idx])
    }
}

impl VectorIndexReader for SegmentedReaderFacade {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_vector(&self, doc_id: u64, field_name: &str) -> Result<Option<Vector>> {
        match self.owner_of(doc_id, field_name) {
            Some(reader) => reader.get_vector(doc_id, field_name),
            None => Ok(None),
        }
    }

    fn get_vectors_for_doc(&self, doc_id: u64) -> Result<Vec<(String, Vector)>> {
        let mut out = Vec::new();
        for (d, field, idx) in &self.entries {
            if *d == doc_id
                && let Some(v) = self.readers[*idx].get_vector(doc_id, field)?
            {
                out.push((field.clone(), v));
            }
        }
        Ok(out)
    }

    fn get_vectors(&self, doc_ids: &[(u64, String)]) -> Result<Vec<Option<Vector>>> {
        doc_ids
            .iter()
            .map(|(d, f)| self.get_vector(*d, f))
            .collect()
    }

    fn vector_ids(&self) -> Result<Vec<(u64, String)>> {
        Ok(self
            .entries
            .iter()
            .map(|(d, f, _)| (*d, f.clone()))
            .collect())
    }

    fn vector_count(&self) -> usize {
        self.entries.len()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn distance_metric(&self) -> DistanceMetric {
        self.metric
    }

    fn stats(&self) -> VectorStats {
        VectorStats {
            vector_count: self.entries.len(),
            dimension: self.dimension,
            memory_usage: 0,
            build_time_ms: 0,
        }
    }

    fn contains_vector(&self, doc_id: u64, field_name: &str) -> bool {
        self.entries
            .iter()
            .any(|(d, f, _)| *d == doc_id && f == field_name)
    }

    fn get_vector_range(
        &self,
        start_doc_id: u64,
        end_doc_id: u64,
    ) -> Result<Vec<(u64, String, Vector)>> {
        let mut out = Vec::new();
        for (doc_id, field, idx) in &self.entries {
            if *doc_id >= start_doc_id
                && *doc_id < end_doc_id
                && let Some(v) = self.readers[*idx].get_vector(*doc_id, field)?
            {
                out.push((*doc_id, field.clone(), v));
            }
        }
        Ok(out)
    }

    fn get_vectors_by_field(&self, field_name: &str) -> Result<Vec<(u64, Vector)>> {
        let mut out = Vec::new();
        for (doc_id, field, idx) in &self.entries {
            if field == field_name
                && let Some(v) = self.readers[*idx].get_vector(*doc_id, field)?
            {
                out.push((*doc_id, v));
            }
        }
        Ok(out)
    }

    fn field_names(&self) -> Result<Vec<String>> {
        let mut names: Vec<String> = Vec::new();
        for (_, field, _) in &self.entries {
            if !names.iter().any(|n| n == field) {
                names.push(field.clone());
            }
        }
        Ok(names)
    }

    fn vector_iterator(&self) -> Result<Box<dyn VectorIterator>> {
        // Materialize through the newest-wins entries; segmented iteration is
        // facade-level only (merge uses per-segment readers directly).
        let mut items = Vec::with_capacity(self.entries.len());
        for (doc_id, field, idx) in &self.entries {
            if let Some(v) = self.readers[*idx].get_vector(*doc_id, field)? {
                items.push((*doc_id, field.clone(), v));
            }
        }
        Ok(Box::new(FacadeIterator { items, pos: 0 }))
    }

    fn metadata(&self) -> Result<VectorIndexMetadata> {
        Ok(VectorIndexMetadata {
            index_type: "segmented".to_string(),
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            version: "1".to_string(),
            build_config: serde_json::Value::Null,
            custom_metadata: std::collections::HashMap::new(),
        })
    }

    fn validate(&self) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        for reader in &self.readers {
            let report = reader.validate()?;
            errors.extend(report.errors);
        }
        Ok(ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            repair_suggestions: Vec::new(),
        })
    }
}

/// Iterator over the facade's materialized newest-wins entries.
#[derive(Debug)]
struct FacadeIterator {
    items: Vec<(u64, String, Vector)>,
    pos: usize,
}

impl VectorIterator for FacadeIterator {
    fn next(&mut self) -> Result<Option<(u64, String, Vector)>> {
        if self.pos >= self.items.len() {
            return Ok(None);
        }
        let item = self.items[self.pos].clone();
        self.pos += 1;
        Ok(Some(item))
    }

    fn skip_to(&mut self, doc_id: u64, field_name: &str) -> Result<bool> {
        while self.pos < self.items.len() {
            let (d, f, _) = &self.items[self.pos];
            if *d == doc_id && f == field_name {
                return Ok(true);
            }
            self.pos += 1;
        }
        Ok(false)
    }

    fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        Ok(())
    }

    fn position(&self) -> (u64, String) {
        if self.pos < self.items.len() {
            let (d, f, _) = &self.items[self.pos];
            (*d, f.clone())
        } else {
            (u64::MAX, String::new())
        }
    }
}

/// Multi-segment fan-out searcher (#880 newest-wins semantics, #883
/// expanding-refill), shared across index types.
///
/// Per-segment concrete searcher construction is supplied by the caller as
/// `make_searcher`, since that's the only index-type-specific piece (e.g.
/// HNSW's `ef_search`-carrying constructor vs. IVF's `n_probe`-carrying
/// one).
/// Builds the concrete per-segment searcher for one reader.
type MakeSearcher =
    Box<dyn Fn(Arc<dyn VectorIndexReader>) -> Result<Box<dyn VectorIndexSearcher>> + Send + Sync>;

pub struct SegmentFanoutSearcher {
    /// Sealed readers, newest generation first, deletion bitmap attached.
    readers: Vec<Arc<dyn VectorIndexReader>>,
    /// The shared deletion bitmap (also attached to the readers), used by
    /// [`Self::count`] to exclude soft-deleted docs.
    bitmap: Option<Arc<DeletionBitmap>>,
    /// Builds the concrete per-segment searcher for one reader.
    make_searcher: MakeSearcher,
}

impl std::fmt::Debug for SegmentFanoutSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentFanoutSearcher")
            .field("segments", &self.readers.len())
            .field("has_bitmap", &self.bitmap.is_some())
            .finish_non_exhaustive()
    }
}

impl SegmentFanoutSearcher {
    /// Build a fan-out searcher over `readers` (newest generation first).
    ///
    /// `make_searcher` is invoked once per segment per `search`/`warmup`
    /// call to build that segment's concrete searcher.
    pub fn new(
        readers: Vec<Arc<dyn VectorIndexReader>>,
        bitmap: Option<Arc<DeletionBitmap>>,
        make_searcher: impl Fn(Arc<dyn VectorIndexReader>) -> Result<Box<dyn VectorIndexSearcher>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            readers,
            bitmap,
            make_searcher: Box::new(make_searcher),
        }
    }

    /// Whether any reader NEWER than `idx` contains `(doc_id, field)` — the
    /// containment mask that makes the newest copy win (#880).
    fn shadowed(&self, idx: usize, doc_id: u64, field: &str) -> bool {
        self.readers[..idx]
            .iter()
            .any(|r| r.contains_vector(doc_id, field))
    }

    /// Probe one segment: run the expanding-refill loop (#883) and drop
    /// newest-wins-shadowed hits (#880).
    ///
    /// Over-fetch per segment: containment masking drops shadowed hits
    /// AFTER the per-segment top-k, so stale copies would otherwise
    /// consume result slots (#880). Start at 2x for the common case; if
    /// masking dropped this segment below the requested limit AND the pass
    /// was truncated at the budget (i.e. live hits may sit below the cut
    /// behind a band of masked stale copies), EXPAND the budget
    /// geometrically and re-query until enough live hits survive or the
    /// segment is exhausted (#883). A fixed multiplier cannot bound recall
    /// against an arbitrarily deep stale band — the true nearest neighbour
    /// could sit below any constant cut — so the budget doubles each round.
    /// Bounded: top_k reaches the segment size in O(log n) queries, and
    /// the common (no-pollution) case stops after one pass.
    ///
    /// After the refill loop the surviving hits are rescored on the
    /// shared dequantized-f32 basis (Issue #927, see
    /// [`Self::rescore_on_shared_basis`]) unless the segment's scores
    /// already sit on the exact f32 rerank basis, and `min_similarity`
    /// is re-applied to the final scores.
    ///
    /// Entirely self-contained per segment (the concrete searcher is built
    /// here, masking reads only immutable newer readers), so `search` can
    /// run one probe per worker thread (#926).
    ///
    /// # Arguments
    ///
    /// * `idx` - Index of the segment's reader (newest-first ordering).
    /// * `request` - The original query; its `top_k` is replaced by the
    ///   expanding budget internally.
    /// * `limit` - The caller-requested `top_k`.
    /// * `metric` - The index's distance metric.
    /// * `prepared_query` - The raw query prepared once per search
    ///   (shared read-only across worker threads).
    ///
    /// # Returns
    ///
    /// The segment's surviving (unmasked, comparably-scored) hits and
    /// the number of candidates its probes examined.
    fn probe_segment(
        &self,
        idx: usize,
        request: &VectorIndexQuery,
        limit: usize,
        metric: DistanceMetric,
        prepared_query: &crate::vector::core::distance::PreparedQuery<'_>,
    ) -> Result<(
        Vec<crate::vector::search::searcher::VectorIndexQueryResult>,
        usize,
    )> {
        let searcher = (self.make_searcher)(self.readers[idx].clone())?;

        let mut probe = request.clone();
        probe.params.top_k = limit.saturating_mul(2);
        let mut candidates_examined = 0usize;
        // Whether this segment's scores are already on the exact f32
        // rerank basis (Issue #481 Stage 2) — comparable across
        // segments and MORE precise than the dequantized rescore, so
        // the #927 rescore below must not overwrite them. Stamped by
        // the per-segment searcher; constant across refill rounds
        // (same segment, same sidecar).
        let mut exact_basis = false;
        let mut kept = loop {
            let results = searcher.search(&probe)?;
            candidates_examined += results.candidates_examined;
            exact_basis |= results
                .query_metadata
                .get(crate::vector::search::searcher::SCORE_BASIS_METADATA_KEY)
                .is_some_and(|v| v == crate::vector::search::searcher::SCORE_BASIS_F32_RERANK);
            let returned = results.results.len();
            let kept: Vec<_> = results
                .results
                .into_iter()
                .filter(|hit| !self.shadowed(idx, hit.doc_id, &hit.field_name))
                .collect();

            // Enough live hits survived masking, or the segment returned
            // fewer than requested (nothing deeper to fetch) — done.
            if kept.len() >= limit || returned < probe.params.top_k {
                break kept;
            }
            let next = probe.params.top_k.saturating_mul(2);
            if next == probe.params.top_k {
                break kept; // budget saturated (overflow guard)
            }
            probe.params.top_k = next;
        };

        // Issue #927: overwrite the segment-local scores with the
        // shared-basis ones, then re-apply `min_similarity` on the
        // final scores — the per-segment filter ran on local scores,
        // which clamping can only inflate (clamped distance ≤ true
        // distance), so it never dropped a hit the shared basis would
        // keep; the inverse (an inflated score sneaking past the
        // threshold) is corrected here. Skipped when the segment's
        // scores already sit on the exact f32 rerank basis — a MORE
        // precise shared basis the dequantized rescore would degrade.
        if !exact_basis {
            self.rescore_on_shared_basis(&mut kept, idx, metric, prepared_query)?;
            kept.retain(|hit| hit.similarity >= request.params.min_similarity);
        }
        Ok((kept, candidates_examined))
    }
    /// Rescore `hits` against the raw query in the shared
    /// dequantized-f32 space (Issue #927).
    ///
    /// Per-segment similarities are computed by the quantized kernels
    /// against **per-segment** affine params, with the query clamped into
    /// each segment's value range — so they are comparable only within a
    /// segment. A segment whose range excludes the query reports its
    /// boundary doc as an exact match (`similarity = 1.0`), outranking
    /// truly closer docs from other segments. Clamping shifts every
    /// distance in a segment by the same offset (`|doc − clamp(q)| =
    /// |doc − q| − |q − edge|` for out-of-range queries), so the
    /// segment-internal candidate ordering the graph search produced is
    /// correct — only the absolute scores must be recomputed on a shared
    /// basis before the cross-segment sort (the Stage-2 rerank
    /// precedent).
    ///
    /// # Arguments
    ///
    /// * `hits` - One segment's surviving hits; `distance`/`similarity`
    ///   are overwritten in place.
    /// * `idx` - The segment's reader index (fallback vector source when
    ///   a hit does not carry its dequantized vector).
    /// * `metric` - The index's distance metric.
    /// * `prepared_query` - The raw query prepared once per search.
    ///
    /// # Errors
    ///
    /// Forwards reader / distance-kernel errors.
    fn rescore_on_shared_basis(
        &self,
        hits: &mut [crate::vector::search::searcher::VectorIndexQueryResult],
        idx: usize,
        metric: DistanceMetric,
        prepared_query: &crate::vector::core::distance::PreparedQuery,
    ) -> Result<()> {
        for hit in hits.iter_mut() {
            let distance = if let Some(v) = &hit.vector {
                metric.distance_with_prepared(prepared_query, &v.data)?
            } else if let Some(v) = self.readers[idx].get_vector(hit.doc_id, &hit.field_name)? {
                metric.distance_with_prepared(prepared_query, &v.data)?
            } else {
                // The hit was just returned by this segment, so a missing
                // vector is unreachable in practice; keep the local score
                // rather than dropping the hit.
                continue;
            };
            hit.distance = distance;
            hit.similarity = metric.distance_to_similarity(distance);
        }
        Ok(())
    }
}

impl VectorIndexSearcher for SegmentFanoutSearcher {
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults> {
        let started = crate::util::time::Timer::now();
        let limit = request.params.top_k;
        let mut merged = VectorIndexQueryResults::new();
        if limit == 0 || self.readers.is_empty() {
            return Ok(merged);
        }

        // Issue #927: shared scoring basis for the cross-segment merge —
        // see `rescore_on_shared_basis`. Prepared once and shared
        // read-only across the per-segment probes.
        let metric = self.readers[0].distance_metric();
        let prepared_query = metric.prepare_query(&request.query.data);

        // One self-contained probe per segment (see `probe_segment`). On
        // native targets with more than one segment the probes run
        // concurrently (#926) — the indexed collect keeps per-segment
        // results in newest-first segment order, so the merged vector (and
        // therefore the sort below) sees byte-identical input to the serial
        // path and the results are bit-for-bit the same. wasm32 keeps the
        // serial loop (no threads), matching the crate-wide convention.
        let probe_all_serial = || -> Result<Vec<_>> {
            (0..self.readers.len())
                .map(|idx| self.probe_segment(idx, request, limit, metric, &prepared_query))
                .collect()
        };
        #[cfg(not(target_arch = "wasm32"))]
        let per_segment = if self.readers.len() > 1 {
            use rayon::prelude::*;
            (0..self.readers.len())
                .into_par_iter()
                .map(|idx| self.probe_segment(idx, request, limit, metric, &prepared_query))
                .collect::<Result<Vec<_>>>()?
        } else {
            probe_all_serial()?
        };
        #[cfg(target_arch = "wasm32")]
        let per_segment = probe_all_serial()?;

        for (mut kept, candidates_examined) in per_segment {
            merged.candidates_examined += candidates_examined;
            merged.results.append(&mut kept);
        }

        // Sort by ascending distance rather than descending similarity
        // (#927): every metric's `distance` is lower-is-more-similar and
        // stays precise at long range, while `distance_to_similarity`'s
        // `exp(-d)` (Euclidean/Manhattan) underflows to 0.0 far from the
        // query — collapsing distant hits into ties whose unstable-sort
        // order would leak segment order into the results. Doc id breaks
        // exact-tie ordering deterministically.
        merged.results.sort_unstable_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then(a.doc_id.cmp(&b.doc_id))
        });
        merged.results.truncate(limit);
        merged.search_time_ms = started.elapsed_ms() as f64;
        Ok(merged)
    }

    fn count(&self, request: VectorIndexQuery) -> Result<u64> {
        // Count distinct live `(doc_id, field)` keys across segments with
        // the same newest-wins masking as `search`, excluding soft-deleted
        // docs.
        //
        // Issue #672: iterate per field through `doc_ids_for_field` (an
        // O(1) `Arc` clone on every reader type, #405) instead of
        // `vector_ids()`, whose trait-boundary rehydration materializes a
        // fresh `String` per record on every call — a full-corpus
        // allocation per segment per count query. `field_names()` is
        // dictionary-backed and tiny on all three readers.
        let mut count = 0u64;
        for (idx, reader) in self.readers.iter().enumerate() {
            let fields: Vec<String> = match request.field_name {
                Some(ref field_name) => vec![field_name.clone()],
                None => reader.field_names()?,
            };
            for field in &fields {
                for &doc_id in reader.doc_ids_for_field(field).iter() {
                    if let Some(bitmap) = &self.bitmap
                        && bitmap.is_deleted(doc_id)
                    {
                        continue;
                    }
                    if !self.shadowed(idx, doc_id, field) {
                        count += 1;
                    }
                }
            }
        }
        Ok(count)
    }

    /// Warm every segment (Issue #1483): build each segment's concrete
    /// searcher and run its `warmup`.
    ///
    /// The per-segment searchers are not kept (`search` builds them per
    /// probe), so this builds them here. That is enough for the warming to
    /// last: `HnswSearcher::warmup` pre-faults the segment's mapped rows into
    /// the OS page cache, which outlives the searcher. Segments are warmed one
    /// after another — a one-time startup pass whose per-segment touch is
    /// sequential — and the first error is returned.
    fn warmup(&mut self) -> Result<()> {
        for reader in &self.readers {
            (self.make_searcher)(reader.clone())?.warmup()?;
        }
        Ok(())
    }
}
