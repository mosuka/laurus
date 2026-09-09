//! Vector searcher trait and query/response types.

use std::sync::Arc;

use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::core::vector::Vector;
use crate::vector::search::filter_set::FilterSet;

/// Candidate count at or above which a brute-force distance scan
/// (Flat / IVF) is dispatched to rayon instead of looping serially.
///
/// Below this the serial loop wins because rayon's per-job dispatch
/// (~1-2 µs) would dominate a scan of a few hundred candidates. This
/// counts *candidates within one query*, in contrast to
/// [`VectorIndexSearcher::parallel_threshold`] which counts *queries in a
/// batch* (#710). Both guards can be active at once: a batch parallelises
/// across queries, and each large-enough query further parallelises its
/// own scan on the same rayon global pool (work-stealing bounds total
/// parallelism to the pool size, so there is no OS-thread
/// oversubscription).
///
/// Issue [#662](https://github.com/mosuka/laurus/issues/662).
// `dead_code` only on `wasm32` (no `native` feature → no rayon → the
// const is never read by the serial-only `parallel_scan`).
#[cfg_attr(not(feature = "native"), allow(dead_code))]
pub(crate) const PARALLEL_SCAN_THRESHOLD: usize = 2048;

/// Map `compute` over every brute-force candidate, in parallel when the
/// candidate count reaches [`PARALLEL_SCAN_THRESHOLD`] and the `native`
/// feature (rayon) is enabled, serially otherwise.
///
/// Used by the Flat and IVF searchers to parallelise their per-candidate
/// distance computation (#662). The distance kernel has no side effects,
/// so the output order is irrelevant — callers sort the collected results
/// afterwards, keeping the search deterministic.
///
/// # Arguments
///
/// * `items` - The candidate identifiers to scan (e.g. `doc_id`s or
///   `(doc_id, field_name)` pairs).
/// * `compute` - Maps one candidate to `Ok(Some(result))`, `Ok(None)` to
///   skip it (e.g. its vector is missing), or `Err` to abort the scan.
///
/// # Returns
///
/// The collected non-skipped results in unspecified order, or the first
/// error `compute` produced.
pub(crate) fn parallel_scan<I, T, F>(items: &[I], compute: F) -> Result<Vec<T>>
where
    I: Sync,
    T: Send,
    F: Fn(&I) -> Result<Option<T>> + Sync + Send,
{
    #[cfg(feature = "native")]
    {
        if items.len() >= PARALLEL_SCAN_THRESHOLD {
            use rayon::prelude::*;
            return Ok(items
                .par_iter()
                .map(&compute)
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect());
        }
    }
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if let Some(t) = compute(item)? {
            out.push(t);
        }
    }
    Ok(out)
}

/// Low-level query for a single-vector search against a vector index.
///
/// This type represents a single nearest-neighbor query at the index level,
/// in contrast to the high-level [`VectorSearchRequest`] which can contain
/// multiple query vectors and aggregation settings.
///
/// Naming convention: low-level index operations use "Query" (e.g.,
/// `VectorIndexQuery`, `VectorIndexQueryParams`), while high-level
/// store/engine operations use "Request" (e.g., `VectorSearchRequest`).
#[derive(Debug, Clone)]
pub struct VectorIndexQuery {
    /// The query vector.
    pub query: Vector,
    /// Search configuration.
    pub params: VectorIndexQueryParams,
    /// Optional field name to filter search results.
    /// If None, searches across all fields.
    pub field_name: Option<String>,
    /// Filter-aware traversal allow-set (Issue #645).
    ///
    /// When `Some`, the HNSW searcher keeps only documents in this set in
    /// its result heap while still expanding the search frontier through
    /// non-matching neighbours, so it can reach matching documents that are
    /// surrounded by non-matching ones (which a post-filter alone cannot).
    /// `None` (the default) is the unchanged plain-ANN path.
    ///
    /// Flat / IVF searchers honour it inline too (Issues #645 / #740). The set
    /// is a typed [`FilterSet`] (Issue #739) wrapped in an [`Arc`] so the
    /// 1 → N field expansion in the store shares a single allocation.
    pub filter: Option<Arc<FilterSet>>,
}

impl VectorIndexQuery {
    /// Create a new vector search request.
    pub fn new(query: Vector) -> Self {
        VectorIndexQuery {
            query,
            params: VectorIndexQueryParams::default(),
            field_name: None,
            filter: None,
        }
    }

    /// Set the filter-aware traversal allow-set (Issue #645).
    ///
    /// # Arguments
    ///
    /// * `filter` - Shared [`FilterSet`] of allowed document IDs. Only these
    ///   IDs are eligible for the result heap; the frontier still expands
    ///   through the others to preserve graph connectivity.
    pub fn filter(mut self, filter: Arc<FilterSet>) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set the number of results to return.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.params.top_k = top_k;
        self
    }

    /// Set minimum similarity threshold.
    pub fn min_similarity(mut self, threshold: f32) -> Self {
        self.params.min_similarity = threshold;
        self
    }

    /// Set whether to include scores in results.
    pub fn include_scores(mut self, include: bool) -> Self {
        self.params.include_scores = include;
        self
    }

    /// Set whether to include vectors in results.
    pub fn include_vectors(mut self, include: bool) -> Self {
        self.params.include_vectors = include;
        self
    }

    /// Set search timeout in milliseconds.
    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        self.params.timeout_ms = Some(timeout);
        self
    }

    /// Set field name to filter search results.
    pub fn field_name(mut self, field_name: String) -> Self {
        self.field_name = Some(field_name);
        self
    }

    /// Set two-stage rerank multiplier (Issue #481 Stage 2 — pre-design).
    ///
    /// Currently the searcher returns
    /// [`crate::error::LaurusError::NotImplemented`] when this is set
    /// to `Some(_)`. The API surface is reserved here so existing
    /// callers can opt in once Stage 2 lands without further proto /
    /// binding revisions.
    pub fn rerank_factor(mut self, factor: usize) -> Self {
        self.params.rerank_factor = Some(factor);
        self
    }

    /// Override the HNSW `ef_search` candidate-list size for this query
    /// (Issue [#644](https://github.com/mosuka/laurus/issues/644)).
    ///
    /// `ef_search` controls the recall / latency trade-off on the HNSW
    /// graph search. Higher values explore more graph neighbours and
    /// give higher recall, at the cost of latency. When unset, the
    /// searcher uses the schema-level
    /// [`crate::vector::core::field::HnswOption::default_ef_search`]
    /// (or its internal fallback of `50` when neither is set).
    ///
    /// The effective `ef_search` is always lifted to at least
    /// `top_k * rerank_factor.unwrap_or(1)` so the candidate heap is
    /// never undersized for the requested `top_k`.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.params.ef_search = Some(ef);
        self
    }
}

/// Configuration for low-level vector index query operations.
///
/// Used with [`VectorIndexQuery`] to configure nearest-neighbor search
/// parameters at the index level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexQueryParams {
    /// Number of results to return.
    pub top_k: usize,
    /// Minimum similarity threshold.
    pub min_similarity: f32,
    /// Whether to return similarity scores.
    pub include_scores: bool,
    /// Whether to include vector data in results.
    pub include_vectors: bool,
    /// Search timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Two-stage rerank multiplier (Issue #481 Stage 2 — pre-design).
    ///
    /// When set to `Some(n)`, the index searcher will first fetch
    /// `top_k * n` candidates via the int8 quantized hot path and then
    /// re-score them against the full f32 vectors before returning
    /// the top `top_k`. This recovers the recall lost to scalar
    /// quantization at modest extra cost.
    ///
    /// **Stage 2 of Issue #481 — currently returns
    /// [`crate::error::LaurusError::NotImplemented`] when set to
    /// `Some(_)`.** `None` (the default) runs the Stage 1
    /// quantized-only search.
    #[serde(default)]
    pub rerank_factor: Option<usize>,
    /// Per-query override for the HNSW `ef_search` candidate-list size
    /// (Issue [#644](https://github.com/mosuka/laurus/issues/644)).
    ///
    /// When `None` (the default), the searcher uses the schema-level
    /// [`crate::vector::core::field::HnswOption::default_ef_search`]
    /// or its internal fallback (`50`). When `Some(ef)`, this value
    /// takes precedence over the schema default.
    ///
    /// The effective `ef_search` used by the searcher is always lifted
    /// to at least `max(top_k, top_k * rerank_factor.unwrap_or(1))` so
    /// the candidate heap is never undersized for the requested
    /// `top_k` (or for the candidate-widening implied by Stage-2
    /// rerank).
    ///
    /// This field is ignored by non-HNSW index types.
    #[serde(default)]
    pub ef_search: Option<usize>,
}

impl Default for VectorIndexQueryParams {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_similarity: 0.0,
            include_scores: true,
            include_vectors: false,
            timeout_ms: None,
            rerank_factor: None,
            ef_search: None,
        }
    }
}

/// A single result from a low-level vector index query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexQueryResult {
    /// Document ID.
    pub doc_id: u64,
    /// Field name of the matched vector.
    pub field_name: String,
    /// Similarity score (higher is more similar).
    pub similarity: f32,
    /// Distance score (lower is more similar).
    pub distance: f32,
    /// Optional vector data.
    pub vector: Option<Vector>,
}

/// `query_metadata` key a per-segment searcher sets when its scores were
/// computed against the exact f32 rerank sidecar (Issue #481 Stage 2) —
/// i.e. `distance(raw_query, true_f32_vector)`, a basis that is already
/// comparable across segments AND more precise than the dequantized
/// reconstruction. The multi-segment fan-out (Issue #927) skips its
/// shared-basis rescore for such results so the sidecar's precision is
/// not overwritten.
pub(crate) const SCORE_BASIS_METADATA_KEY: &str = "score_basis";

/// `query_metadata` value for [`SCORE_BASIS_METADATA_KEY`]: exact f32
/// sidecar rerank basis.
pub(crate) const SCORE_BASIS_F32_RERANK: &str = "f32-rerank";

/// Collection of results from a low-level vector index query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexQueryResults {
    /// Individual search results.
    pub results: Vec<VectorIndexQueryResult>,
    /// Total number of candidates examined.
    pub candidates_examined: usize,
    /// Search execution time in milliseconds.
    pub search_time_ms: f64,
    /// Query metadata.
    pub query_metadata: std::collections::HashMap<String, String>,
}

impl VectorIndexQueryResults {
    /// Create new empty search results.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            candidates_examined: 0,
            search_time_ms: 0.0,
            query_metadata: std::collections::HashMap::new(),
        }
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Sort results by similarity (descending).
    pub fn sort_by_similarity(&mut self) {
        self.results
            .sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
    }

    /// Sort results by distance (ascending).
    pub fn sort_by_distance(&mut self) {
        self.results
            .sort_by(|a, b| a.distance.total_cmp(&b.distance));
    }

    /// Take the top k results.
    pub fn take_top_k(&mut self, k: usize) {
        if self.results.len() > k {
            self.results.truncate(k);
        }
    }

    /// Filter results by minimum similarity.
    pub fn filter_by_similarity(&mut self, min_similarity: f32) {
        self.results
            .retain(|result| result.similarity >= min_similarity);
    }

    /// Get the best (highest similarity) result.
    pub fn best_result(&self) -> Option<&VectorIndexQueryResult> {
        self.results
            .iter()
            .max_by(|a, b| a.similarity.total_cmp(&b.similarity))
    }
}

impl Default for VectorIndexQueryResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for vector searchers.
pub trait VectorIndexSearcher: Send + Sync + std::fmt::Debug {
    /// Execute a vector similarity search.
    fn search(&self, request: &VectorIndexQuery) -> Result<VectorIndexQueryResults>;

    /// Count the number of vectors matching the query.
    fn count(&self, request: VectorIndexQuery) -> Result<u64>;

    /// Warm up the searcher (pre-load data, etc.).
    fn warmup(&mut self) -> Result<()> {
        // No-op by default. Implementations can override this method to perform
        // any necessary warm-up steps, such as loading index data into memory.
        Ok(())
    }

    /// Threshold above which [`Self::search_batch`] switches to rayon
    /// parallel iteration over the input queries.
    ///
    /// Below this threshold the serial loop wins because rayon's
    /// thread-pool dispatch overhead (~1-2 µs) would otherwise dominate
    /// a single 50-200 µs query. The default of `4` matches the value
    /// Phase 1 of [#648](https://github.com/mosuka/laurus/issues/648)
    /// settled on for the HNSW / Flat / IVF mix; concrete searchers
    /// MAY override to tune for their per-query cost.
    ///
    /// Issue [#712](https://github.com/mosuka/laurus/issues/712)
    /// Phase 2 of [#648](https://github.com/mosuka/laurus/issues/648).
    fn parallel_threshold(&self) -> usize {
        4
    }

    /// Execute `B` independent vector queries in one call.
    ///
    /// The default impl dispatches via [`Self::search_batch_with_threshold`]
    /// using [`Self::parallel_threshold`]: when `queries.len()` meets the
    /// threshold the per-query searches run on rayon's global thread pool,
    /// otherwise they run serially. Overriders MAY exploit shared state
    /// across the `B` queries (e.g., per-field prefetch index, codebook)
    /// to amortise setup costs.
    ///
    /// Returns one [`VectorIndexQueryResults`] per input query, in the
    /// same order as `queries`.
    ///
    /// Issue [#712](https://github.com/mosuka/laurus/issues/712)
    /// Phase 2 of [#648](https://github.com/mosuka/laurus/issues/648).
    fn search_batch(&self, queries: &[VectorIndexQuery]) -> Result<Vec<VectorIndexQueryResults>> {
        self.search_batch_with_threshold(queries, self.parallel_threshold())
    }

    /// Test-only variant of [`Self::search_batch`] that lets the caller
    /// pin the parallelisation threshold.
    ///
    /// `parallel_threshold == 0` forces parallel execution;
    /// `usize::MAX` forces serial. Production code should call
    /// [`Self::search_batch`] which uses [`Self::parallel_threshold`].
    #[doc(hidden)]
    fn search_batch_with_threshold(
        &self,
        queries: &[VectorIndexQuery],
        parallel_threshold: usize,
    ) -> Result<Vec<VectorIndexQueryResults>> {
        #[cfg(feature = "native")]
        {
            use rayon::prelude::*;
            if queries.len() >= parallel_threshold {
                return queries
                    .par_iter()
                    .map(|q| self.search(q))
                    .collect::<Result<Vec<_>>>();
            }
        }
        let _ = parallel_threshold;
        queries.iter().map(|q| self.search(q)).collect()
    }
}

// ── High-level search request types ──────────────────────────────────────────

/// How a vector search query is specified.
///
/// Mirrors [`LexicalSearchQuery`](crate::lexical::search::searcher::LexicalSearchQuery)
/// for symmetry:
///
/// | | Lexical | Vector |
/// |---|---|---|
/// | Deferred resolution | [`Dsl(String)`](crate::lexical::search::searcher::LexicalSearchQuery::Dsl) | [`Payloads`](Self::Payloads) |
/// | Pre-built | [`Obj(Box<dyn Query>)`](crate::lexical::search::searcher::LexicalSearchQuery::Obj) | [`Vectors`](Self::Vectors) |
#[derive(Debug, Clone)]
pub enum VectorSearchQuery {
    /// Raw payloads (text, bytes, etc.) to be embedded into vectors at
    /// search time by the engine's configured embedder.
    Payloads(Vec<crate::vector::store::request::QueryPayload>),

    /// Pre-embedded query vectors, ready for nearest-neighbor search.
    Vectors(Vec<crate::vector::store::request::QueryVector>),
}

fn default_query_limit() -> usize {
    10
}

/// Default overfetch factor (Issue #675).
///
/// `2.0` matches the historical hardcoded behaviour (`top_k = limit * 2`) that
/// [`VectorStore::search`](crate::vector::store::VectorStore::search) applied
/// before the factor was honoured, and the documented gRPC default. Keeping the
/// declared default at `2.0` means callers that do not set `overfetch`
/// (including the engine, which passes `2.0` explicitly) see no behaviour
/// change now that the factor drives `top_k`.
fn default_overfetch() -> f32 {
    2.0
}

/// Parameters for vector search operations.
///
/// Analogous to
/// [`LexicalSearchParams`](crate::lexical::search::searcher::LexicalSearchParams),
/// this struct groups all configuration knobs for a vector search independently
/// of the query specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchParams {
    /// Fields to search in.
    #[serde(default)]
    pub fields: Option<Vec<crate::vector::store::request::FieldSelector>>,
    /// Maximum number of results to return.
    #[serde(default = "default_query_limit")]
    pub limit: usize,
    /// How to combine scores from multiple query vectors.
    #[serde(default)]
    pub score_mode: crate::vector::store::request::VectorScoreMode,
    /// Overfetch factor for better result quality (Issue #675).
    ///
    /// Each per-field index query fetches `ceil(limit * overfetch)` candidates
    /// (see [`Self::overfetch_top_k`]) so the multi-vector score-mode merge has
    /// headroom before the final truncation to [`limit`](Self::limit). A factor
    /// `<= 1.0` (or non-finite) disables overfetch. Defaults to `2.0`
    /// ([`default_overfetch`]).
    #[serde(default = "default_overfetch")]
    pub overfetch: f32,
    /// Minimum score threshold. Results below this score are filtered out.
    ///
    /// Compared against the raw per-field similarity BEFORE `QueryVector`'s
    /// `weight` and the field's `base_weight` (Issue #1084) are applied —
    /// this is a similarity floor, not a final-score floor, matching the
    /// pre-existing contract for `QueryVector.weight`.
    #[serde(default)]
    pub min_score: f32,
    /// List of allowed document IDs (for internal use by Engine filtering).
    ///
    /// External callers set this; the store builds a [`FilterSet`] from it by
    /// shape. The Engine prefers [`allowed_filter`](Self::allowed_filter) to
    /// avoid re-materialising a set it already holds.
    #[serde(skip)]
    pub allowed_ids: Option<Vec<u64>>,
    /// Pre-built allow-set shared from the lexical filter cache (Issue #739).
    ///
    /// When `Some`, the store wraps it as a [`FilterSet::Bitmap`] directly
    /// (zero-copy `Arc` share) and ignores [`allowed_ids`](Self::allowed_ids),
    /// so the engine's filtered hybrid search materialises the set once
    /// (`InvertedIndexReader::matching_doc_ids` → here) instead of
    /// `RoaringTreemap → Vec<u64> → AHashSet`. Internal use by the Engine.
    #[serde(skip)]
    pub allowed_filter: Option<Arc<RoaringTreemap>>,
    /// Optional Stage 2 rerank factor (Issue #481).
    ///
    /// When `Some(factor)`, the HNSW searcher widens the int8 candidate
    /// fetch to `top_k * factor` and rescores against the LRS1 sidecar's
    /// f32 vectors. Honored only on HNSW fields with `rerank_storage`
    /// configured at index time; otherwise silently ignored.
    #[serde(default)]
    pub rerank_factor: Option<usize>,
    /// Per-query override for the HNSW `ef_search` candidate-list size
    /// (Issue [#644](https://github.com/mosuka/laurus/issues/644)).
    ///
    /// When `None` (the default), the searcher uses the schema-level
    /// [`crate::vector::core::field::HnswOption::default_ef_search`]
    /// or its internal fallback (`50`). Ignored by non-HNSW index
    /// types.
    #[serde(default)]
    pub ef_search: Option<usize>,
}

impl Default for VectorSearchParams {
    fn default() -> Self {
        Self {
            fields: None,
            limit: default_query_limit(),
            score_mode: crate::vector::store::request::VectorScoreMode::default(),
            overfetch: default_overfetch(),
            min_score: 0.0,
            allowed_ids: None,
            allowed_filter: None,
            rerank_factor: None,
            ef_search: None,
        }
    }
}

impl VectorSearchParams {
    /// Resolve the per-field index `top_k` (the overfetch candidate pool) from
    /// [`limit`](Self::limit) and [`overfetch`](Self::overfetch) (Issue #675).
    ///
    /// Overfetching pulls more candidates than `limit` so the multi-vector
    /// score-mode merge in
    /// [`VectorStore::search`](crate::vector::store::VectorStore::search) has
    /// headroom before the final truncation to `limit`. An `overfetch` factor
    /// `f` yields `ceil(limit * f)` candidates; a factor `<= 1.0` or non-finite
    /// disables overfetch (`top_k == limit`), and the result never drops below
    /// `limit`. Before this was honoured a hardcoded `2x` was always used.
    ///
    /// # Returns
    ///
    /// The number of candidates each per-field index query should request.
    pub(crate) fn overfetch_top_k(&self) -> usize {
        if !self.overfetch.is_finite() || self.overfetch <= 1.0 {
            return self.limit;
        }
        let scaled = (self.limit as f32 * self.overfetch).ceil();
        if scaled >= usize::MAX as f32 {
            usize::MAX
        } else {
            (scaled as usize).max(self.limit)
        }
    }
}

/// Request model for collection-level vector search.
///
/// Mirrors
/// [`LexicalSearchRequest`](crate::lexical::search::searcher::LexicalSearchRequest)
/// structure: a query enum paired with a params struct.
#[derive(Debug, Clone)]
pub struct VectorSearchRequest {
    /// The query to execute.
    pub query: VectorSearchQuery,
    /// Search configuration.
    pub params: VectorSearchParams,
}

impl Default for VectorSearchRequest {
    fn default() -> Self {
        Self {
            query: VectorSearchQuery::Vectors(Vec::new()),
            params: VectorSearchParams::default(),
        }
    }
}

// ── High-level searcher trait ────────────────────────────────────────────────

/// Trait for high-level vector search implementations.
///
/// This trait defines the interface for executing searches against vector indexes,
/// analogous to [`crate::lexical::search::searcher::LexicalSearcher`] for lexical search.
///
/// Unlike [`VectorIndexSearcher`] which operates at the low-level (single vector queries),
/// `VectorSearcher` handles high-level search requests with multiple query vectors,
/// field selection, filters, and score aggregation.
pub trait VectorSearcher: Send + Sync + std::fmt::Debug {
    /// Execute a search with the given request.
    ///
    /// This method processes a high-level search request that may contain
    /// multiple query vectors across different fields, applies filters,
    /// and aggregates scores according to the specified score mode.
    fn search(
        &self,
        request: &VectorSearchRequest,
    ) -> crate::error::Result<crate::vector::store::response::VectorSearchResults>;

    /// Count the number of matching documents for a request.
    ///
    /// Returns the number of documents that match the given search request,
    /// applying the min_score threshold if specified in the request.
    fn count(&self, request: &VectorSearchRequest) -> crate::error::Result<u64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both the serial and rayon paths of [`parallel_scan`] must return the
    /// same multiset of results. The rayon path's output order is
    /// unspecified, so results are sorted before comparison. Exercises a
    /// count below `PARALLEL_SCAN_THRESHOLD` (serial) and above it
    /// (parallel).
    #[test]
    fn parallel_scan_parallel_matches_serial() {
        for n in [100usize, PARALLEL_SCAN_THRESHOLD + 500] {
            let items: Vec<u64> = (0..n as u64).collect();
            let mut got = parallel_scan(&items[..], |&x| Ok(Some(x * 2))).unwrap();
            got.sort_unstable();
            let expected: Vec<u64> = (0..n as u64).map(|x| x * 2).collect();
            assert_eq!(got, expected, "n = {n}");
        }
    }

    /// `Ok(None)` candidates are skipped on both paths.
    #[test]
    fn parallel_scan_skips_none() {
        for n in [100usize, PARALLEL_SCAN_THRESHOLD + 500] {
            let items: Vec<u64> = (0..n as u64).collect();
            let mut got =
                parallel_scan(&items[..], |&x| Ok(if x % 2 == 0 { Some(x) } else { None }))
                    .unwrap();
            got.sort_unstable();
            let expected: Vec<u64> = (0..n as u64).filter(|x| x % 2 == 0).collect();
            assert_eq!(got, expected, "n = {n}");
        }
    }

    /// An `Err` from any candidate aborts the whole scan, on both paths.
    #[test]
    fn parallel_scan_propagates_error() {
        for n in [100usize, PARALLEL_SCAN_THRESHOLD + 500] {
            let items: Vec<u64> = (0..n as u64).collect();
            let result: Result<Vec<u64>> = parallel_scan(&items[..], |&x| {
                if x == (n as u64 / 2) {
                    Err(crate::error::LaurusError::internal("boom"))
                } else {
                    Ok(Some(x))
                }
            });
            assert!(result.is_err(), "n = {n}");
        }
    }

    fn params_with_overfetch(limit: usize, overfetch: f32) -> VectorSearchParams {
        VectorSearchParams {
            limit,
            overfetch,
            ..Default::default()
        }
    }

    /// `overfetch_top_k` scales `limit` by the factor with a ceiling, honouring
    /// the user-supplied value (Issue #675).
    #[test]
    fn overfetch_top_k_scales_limit() {
        assert_eq!(params_with_overfetch(10, 2.0).overfetch_top_k(), 20);
        assert_eq!(params_with_overfetch(10, 3.0).overfetch_top_k(), 30);
        // Non-integer factors round up so the pool never undershoots.
        assert_eq!(params_with_overfetch(10, 1.5).overfetch_top_k(), 15);
        assert_eq!(params_with_overfetch(3, 1.5).overfetch_top_k(), 5);
    }

    /// The default factor (`2.0`) reproduces the historical `limit * 2`
    /// candidate pool, so callers that never set `overfetch` are unaffected.
    #[test]
    fn overfetch_top_k_default_is_2x() {
        assert_eq!(VectorSearchParams::default().overfetch, 2.0);
        assert_eq!(
            params_with_overfetch(7, default_overfetch()).overfetch_top_k(),
            14
        );
    }

    /// Factors `<= 1.0` (and degenerate values) disable overfetch — `top_k`
    /// equals `limit` and never drops below it.
    #[test]
    fn overfetch_top_k_clamps_low_and_degenerate_factors() {
        assert_eq!(params_with_overfetch(10, 1.0).overfetch_top_k(), 10);
        assert_eq!(params_with_overfetch(10, 0.5).overfetch_top_k(), 10);
        assert_eq!(params_with_overfetch(10, 0.0).overfetch_top_k(), 10);
        assert_eq!(params_with_overfetch(10, -1.0).overfetch_top_k(), 10);
        assert_eq!(params_with_overfetch(10, f32::NAN).overfetch_top_k(), 10);
        assert_eq!(
            params_with_overfetch(10, f32::INFINITY).overfetch_top_k(),
            10
        );
        // limit == 0 stays 0 regardless of factor.
        assert_eq!(params_with_overfetch(0, 4.0).overfetch_top_k(), 0);
    }
}
