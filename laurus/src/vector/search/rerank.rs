//! Composable rerank pipeline (Issue #650 PR-1 / #931).
//!
//! Stage-2 rerank (Issue #481) was an inline match arm in the HNSW
//! searcher, unusable by Flat/IVF and inexpressible as a multi-stage
//! chain. This module hosts the index-type-agnostic pieces: an
//! object-safe [`RerankStage`], the SoA [`RerankCandidates`] buffer the
//! stages narrow, and the [`RerankPipeline`] driver. A per-segment
//! searcher builds the pipeline once per query from whatever backing
//! data its reader exposes and runs it over the quantized candidate
//! set before result conversion.
//!
//! A stage's per-query preparation (prepared query, LUTs) coincides
//! with its single per-query invocation, so preparation happens inside
//! [`RerankStage::rescore`] rather than through an associated
//! `Prepared` type — keeping the trait object-safe (the deviation from
//! the #650 sketch is recorded on #931).

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::distance_quantized::{QuantizedQuery, distance_quantized};
use crate::vector::core::vector::Vector;
use crate::vector::index::quantized_storage::QuantizedVectorPool;
use crate::vector::index::rerank_storage::RerankStoragePool;

/// SoA candidate buffer flowing through the pipeline: parallel
/// `doc_ids` / `distances` columns (the #650 data-structure note — no
/// tuple-of-structs).
#[derive(Debug, Default)]
pub(crate) struct RerankCandidates {
    /// Candidate doc ids, parallel to [`Self::distances`].
    pub doc_ids: Vec<u64>,
    /// Current-basis distance per candidate (lower is more similar).
    pub distances: Vec<f32>,
}

impl RerankCandidates {
    /// Create an empty buffer sized for `capacity` candidates.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            doc_ids: Vec::with_capacity(capacity),
            distances: Vec::with_capacity(capacity),
        }
    }

    /// Append one candidate.
    #[inline]
    pub fn push(&mut self, doc_id: u64, distance: f32) {
        self.doc_ids.push(doc_id);
        self.distances.push(distance);
    }

    /// Number of candidates currently held.
    #[inline]
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Sort both columns ascending by distance with a doc-id tiebreak —
    /// the crate-wide ordering convention (#927/#933: similarity's
    /// `exp(-d)` underflows at long range; distance stays precise).
    pub fn sort_by_distance(&mut self) {
        let mut order: Vec<u32> = (0..self.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            self.distances[a as usize]
                .total_cmp(&self.distances[b as usize])
                .then(self.doc_ids[a as usize].cmp(&self.doc_ids[b as usize]))
        });
        self.doc_ids = order.iter().map(|&i| self.doc_ids[i as usize]).collect();
        self.distances = order.iter().map(|&i| self.distances[i as usize]).collect();
    }
}

/// One rerank stage: rescores (a prefix of) the incoming candidates
/// against a more precise representation and re-sorts them on the new
/// basis.
///
/// Implementations must be cheap to construct per query — any per-query
/// preparation (prepared query, LUT) happens inside
/// [`Self::rescore`], which is called exactly once per query.
pub(crate) trait RerankStage: Send + Sync {
    /// Rescore the top `take_n` incoming candidates (the buffer arrives
    /// sorted ascending on the previous basis), replace the buffer with
    /// the rescored survivors sorted ascending on the new basis, and
    /// report `true`. A candidate absent from this stage's backing data
    /// is dropped (it cannot be scored on the new basis). Return
    /// `Ok(false)` — leaving the buffer untouched — when the stage's
    /// backing data is unavailable.
    ///
    /// # Arguments
    ///
    /// * `query` - The raw query vector.
    /// * `candidates` - The SoA candidate buffer, sorted ascending by
    ///   the previous stage's distances.
    /// * `take_n` - How many leading candidates to rescore (the
    ///   caller-computed widening budget for this stage).
    fn rescore(
        &self,
        query: &Vector,
        candidates: &mut RerankCandidates,
        take_n: usize,
    ) -> Result<bool>;
}

/// Rescore against the exact f32 vectors in the LRS1 rerank sidecar
/// (Issue #481 Stage 2) — the final, exact stage of every pipeline.
pub(crate) struct F32SidecarStage {
    pool: Arc<RerankStoragePool>,
    /// `doc_id -> pool position` for the searched field, resolved once
    /// at stage construction.
    positions: Option<Arc<HashMap<u64, u32>>>,
    metric: DistanceMetric,
}

impl F32SidecarStage {
    /// Build the stage for one query against `field_name`.
    ///
    /// # Arguments
    ///
    /// * `pool` - The reader's loaded sidecar pool.
    /// * `field_name` - The searched field (position index key).
    /// * `metric` - The index's distance metric.
    pub fn new(pool: Arc<RerankStoragePool>, field_name: &str, metric: DistanceMetric) -> Self {
        let positions = pool.field_position_index(field_name);
        Self {
            pool,
            positions,
            metric,
        }
    }
}

impl RerankStage for F32SidecarStage {
    fn rescore(
        &self,
        query: &Vector,
        candidates: &mut RerankCandidates,
        take_n: usize,
    ) -> Result<bool> {
        // Issue #1390: no position index for this field at all means this
        // stage has nothing to rescore against -- honor the trait's
        // "backing data unavailable" contract (leave `candidates`
        // untouched, report `false`) instead of emptying the buffer while
        // reporting `true`.
        let Some(positions) = self.positions.as_ref() else {
            return Ok(false);
        };
        let prepared = self.metric.prepare_query(&query.data);
        let take_n = take_n.min(candidates.len());
        let mut rescored = RerankCandidates::with_capacity(take_n);
        for i in 0..take_n {
            let doc_id = candidates.doc_ids[i];
            // A doc without a sidecar record for this field cannot be
            // scored on the exact basis — drop it (pre-refactor
            // behavior: the inline arm `continue`d on a missing
            // position).
            let Some(pos) = positions.get(&doc_id).copied() else {
                continue;
            };
            let distance = self
                .metric
                .distance_with_prepared(&prepared, self.pool.f32_slice_at(pos))?;
            rescored.push(doc_id, distance);
        }
        rescored.sort_by_distance();
        *candidates = rescored;
        Ok(true)
    }
}

/// Rescore against a derived int8 (SQ) view of the rerank sidecar
/// (Issue #673) — a middle stage that narrows PQ/graph candidates with a
/// cheap int8 kernel before the final exact [`F32SidecarStage`] runs.
///
/// The int8 view is derived from the same f32 payload
/// [`F32SidecarStage`] reads (see [`RerankStoragePool::int8_view`]), so
/// this stage never requires its own on-disk sidecar.
pub(crate) struct SqRerankStage {
    pool: Arc<QuantizedVectorPool>,
    /// `doc_id -> pool position` for the searched field, resolved once
    /// at stage construction.
    positions: Option<Arc<HashMap<u64, u32>>>,
    metric: DistanceMetric,
}

impl SqRerankStage {
    /// Build the stage for one query against `field_name`.
    ///
    /// # Arguments
    ///
    /// * `pool` - The derived int8 view ([`RerankStoragePool::int8_view`]).
    /// * `field_name` - The searched field (position index key).
    /// * `metric` - The index's distance metric.
    pub fn new(pool: Arc<QuantizedVectorPool>, field_name: &str, metric: DistanceMetric) -> Self {
        let positions = pool.field_position_index(field_name);
        Self {
            pool,
            positions,
            metric,
        }
    }
}

impl RerankStage for SqRerankStage {
    fn rescore(
        &self,
        query: &Vector,
        candidates: &mut RerankCandidates,
        take_n: usize,
    ) -> Result<bool> {
        let prepared = QuantizedQuery::prepare(&query.data, &self.pool.params);
        let take_n = take_n.min(candidates.len());
        let mut rescored = RerankCandidates::with_capacity(take_n);
        for i in 0..take_n {
            let doc_id = candidates.doc_ids[i];
            // A doc without an int8 record for this field cannot be
            // scored on this stage's basis — drop it, mirroring
            // `F32SidecarStage`'s missing-position behavior.
            let Some(pos) = self
                .positions
                .as_ref()
                .and_then(|idx| idx.get(&doc_id).copied())
            else {
                continue;
            };
            let (cand, meta) = self.pool.record_at(pos);
            let distance = distance_quantized(self.metric, &prepared, cand, meta);
            rescored.push(doc_id, distance);
        }
        rescored.sort_by_distance();
        *candidates = rescored;
        Ok(true)
    }
}

/// Multi-stage rerank driver: stage `i` rescores the top
/// `top_k * factors[i]` candidates of the previous basis.
pub(crate) struct RerankPipeline {
    stages: Vec<Box<dyn RerankStage>>,
    /// Per-stage widening factors, parallel to `stages`.
    factors: Vec<usize>,
}

impl RerankPipeline {
    /// Build a pipeline from parallel `stages` / `factors`.
    ///
    /// # Panics
    ///
    /// Debug-asserts the two are the same length and non-empty.
    pub fn new(stages: Vec<Box<dyn RerankStage>>, factors: Vec<usize>) -> Self {
        debug_assert_eq!(stages.len(), factors.len());
        debug_assert!(!stages.is_empty());
        Self { stages, factors }
    }

    /// Run every stage over `candidates` (which must arrive sorted
    /// ascending on the generation basis).
    ///
    /// # Returns
    ///
    /// Whether the FINAL stage applied — i.e. whether the surviving
    /// scores sit on that stage's basis. For a pipeline ending in
    /// [`F32SidecarStage`] this is exactly the
    /// `score_basis = "f32-rerank"` condition the multi-segment fan-out
    /// keys on (#927).
    ///
    /// # Arguments
    ///
    /// * `query` - The raw query vector.
    /// * `candidates` - The generation-stage candidates, sorted
    ///   ascending by distance.
    /// * `top_k` - The caller-requested result count; stage `i`
    ///   consumes at most `top_k * factors[i]` candidates.
    pub fn run(
        &self,
        query: &Vector,
        candidates: &mut RerankCandidates,
        top_k: usize,
    ) -> Result<bool> {
        let mut last_applied = false;
        for (stage, &factor) in self.stages.iter().zip(&self.factors) {
            let take_n = top_k.saturating_mul(factor.max(1));
            last_applied = stage.rescore(query, candidates, take_n)?;
        }
        Ok(last_applied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test stage that halves each distance and reverses nothing —
    /// used to check narrowing and ordering mechanics without storage.
    struct HalveStage;
    impl RerankStage for HalveStage {
        fn rescore(
            &self,
            _query: &Vector,
            candidates: &mut RerankCandidates,
            take_n: usize,
        ) -> Result<bool> {
            let take_n = take_n.min(candidates.len());
            let mut out = RerankCandidates::with_capacity(take_n);
            for i in 0..take_n {
                out.push(candidates.doc_ids[i], candidates.distances[i] / 2.0);
            }
            out.sort_by_distance();
            *candidates = out;
            Ok(true)
        }
    }

    /// A stage whose backing data is absent: must leave the buffer
    /// untouched and report `false`.
    struct AbsentStage;
    impl RerankStage for AbsentStage {
        fn rescore(
            &self,
            _query: &Vector,
            _candidates: &mut RerankCandidates,
            _take_n: usize,
        ) -> Result<bool> {
            Ok(false)
        }
    }

    fn buffer(pairs: &[(u64, f32)]) -> RerankCandidates {
        let mut c = RerankCandidates::with_capacity(pairs.len());
        for &(id, d) in pairs {
            c.push(id, d);
        }
        c
    }

    #[test]
    fn sort_by_distance_breaks_ties_by_doc_id() {
        let mut c = buffer(&[(9, 1.0), (2, 0.5), (7, 0.5)]);
        c.sort_by_distance();
        assert_eq!(c.doc_ids, vec![2, 7, 9]);
        assert_eq!(c.distances, vec![0.5, 0.5, 1.0]);
    }

    #[test]
    fn pipeline_narrows_by_per_stage_factor_and_reports_last_stage() {
        let pipeline =
            RerankPipeline::new(vec![Box::new(HalveStage), Box::new(HalveStage)], vec![3, 1]);
        // top_k = 2: stage 1 takes 6 (all), stage 2 takes 2.
        let mut c = buffer(&[(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0), (6, 6.0)]);
        let applied = pipeline.run(&Vector::new(vec![0.0]), &mut c, 2).unwrap();
        assert!(applied);
        assert_eq!(c.doc_ids, vec![1, 2]);
        assert_eq!(c.distances, vec![0.25, 0.5]);
    }

    #[test]
    fn absent_final_stage_reports_not_applied_and_preserves_buffer() {
        let pipeline = RerankPipeline::new(vec![Box::new(AbsentStage)], vec![4]);
        let mut c = buffer(&[(1, 1.0), (2, 2.0)]);
        let applied = pipeline.run(&Vector::new(vec![0.0]), &mut c, 1).unwrap();
        assert!(!applied);
        assert_eq!(c.doc_ids, vec![1, 2]);
        assert_eq!(c.distances, vec![1.0, 2.0]);
    }

    use crate::vector::core::quantization::{QuantizedVectorMeta, ScalarQuantParams};

    /// Build a 2-doc int8 pool under field `"f"`: doc 1 at the origin,
    /// doc 2 far away, so int8 Euclidean distance unambiguously ranks
    /// doc 1 first for a query near the origin.
    fn int8_pool_near_and_far() -> Arc<QuantizedVectorPool> {
        let params =
            ScalarQuantParams::train(&[Vector::new(vec![0.0, 0.0]), Vector::new(vec![10.0, 10.0])])
                .unwrap();
        let mut records = Vec::new();
        for (doc_id, data) in [(1u64, vec![0.0_f32, 0.0]), (2, vec![10.0, 10.0])] {
            let q = params.quantize_slice(&data);
            let meta = QuantizedVectorMeta::from_quantized(&q, &params);
            records.push((doc_id, "f".to_string(), q, meta));
        }
        Arc::new(QuantizedVectorPool::build(params, 2, records))
    }

    #[test]
    fn sq_stage_rescores_and_reorders_by_int8_distance() {
        let stage = SqRerankStage::new(int8_pool_near_and_far(), "f", DistanceMetric::Euclidean);
        // Previous-stage distances are irrelevant here: the stage must
        // recompute from the int8 payload, not trust the incoming order.
        let mut c = buffer(&[(2, 0.0), (1, 100.0)]);
        let applied = stage
            .rescore(&Vector::new(vec![0.0, 0.0]), &mut c, 2)
            .unwrap();
        assert!(applied);
        assert_eq!(c.doc_ids, vec![1, 2], "doc 1 (near query) must rank first");
    }

    #[test]
    fn sq_stage_drops_candidates_absent_from_the_pool() {
        let stage = SqRerankStage::new(int8_pool_near_and_far(), "f", DistanceMetric::Euclidean);
        let mut c = buffer(&[(1, 0.0), (99, 0.0)]); // doc 99 has no int8 record
        let applied = stage
            .rescore(&Vector::new(vec![0.0, 0.0]), &mut c, 2)
            .unwrap();
        assert!(applied);
        assert_eq!(c.doc_ids, vec![1]);
    }

    /// Issue #1390: when the pool has records for other fields but none
    /// for the field this stage was built against, `positions` is `None`
    /// -- distinct from a single missing `doc_id` within an existing
    /// index (covered by `sq_stage_drops_candidates_absent_from_the_pool`
    /// for the sibling stage).
    #[test]
    fn f32_sidecar_stage_without_position_index_reports_false_and_leaves_candidates_untouched() {
        let pool = Arc::new(
            crate::vector::index::rerank_storage::RerankStoragePool::build(
                crate::vector::core::rerank::RerankStorageKind::F32,
                2,
                vec![(1, "other".to_string(), vec![1.0, 2.0])],
            ),
        );
        let stage = F32SidecarStage::new(pool, "embedding", DistanceMetric::Euclidean);

        let mut c = buffer(&[(1, 0.5), (2, 1.5)]);
        let applied = stage
            .rescore(&Vector::new(vec![0.0, 0.0]), &mut c, 2)
            .unwrap();

        assert!(!applied, "a stage with no backing data must report false");
        assert_eq!(c.doc_ids, vec![1, 2], "candidates must be left untouched");
        assert_eq!(
            c.distances,
            vec![0.5, 1.5],
            "candidates must be left untouched"
        );
    }
}
