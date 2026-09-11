//! Collector implementations for gathering search results.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::Debug;

use crate::error::Result;
use crate::lexical::query::SearchHit;

/// Trait for collecting search results.
pub trait Collector: Send + Debug {
    /// Collect a document hit.
    fn collect(&mut self, doc_id: u64, score: f32) -> Result<()>;

    /// Get the final results.
    fn results(&self) -> Vec<SearchHit>;

    /// Get the total number of hits collected.
    fn total_hits(&self) -> u64;

    /// Check if this collector needs more results.
    fn needs_more(&self) -> bool;

    /// Get the minimum score threshold.
    fn min_score(&self) -> f32;

    /// Reset the collector for a new search.
    fn reset(&mut self);

    /// The current "competitive" score floor — a candidate whose
    /// upper-bound score is at or below this value cannot enter the
    /// final result set, so the searcher is free to skip it.
    ///
    /// Top-K collectors override this to expose the K-th heap score
    /// once the heap is full, enabling MaxScore / WAND-style early
    /// termination (#403). The default returns `f32::NEG_INFINITY`,
    /// which disables the optimisation for collectors that do not have
    /// a meaningful upper-bound concept (count-only, all-docs, etc.).
    ///
    /// # Returns
    ///
    /// The score that an incoming candidate's upper bound must exceed
    /// to be worth collecting.
    fn min_competitive(&self) -> f32 {
        f32::NEG_INFINITY
    }

    /// Whether this collector benefits from the Block-Max-WAND fast
    /// path (#475 PR-F) — i.e. whether [`Self::min_competitive`] will
    /// eventually return a meaningful (finite) value that the BMW
    /// pivot loop can use to skip non-competitive blocks.
    ///
    /// Top-K-style collectors override this to `true`. Collectors
    /// without a heap-based competitive floor (`CountCollector`,
    /// `AllDocsCollector`) keep the default `false` so the searcher
    /// stays on the existing matcher-driven path, where BMW would add
    /// overhead without delivering any pruning benefit.
    fn bmw_capable(&self) -> bool {
        false
    }

    /// Hint: the number of top hits the collector wants (#476 Phase 1).
    /// The per-segment fanout uses this to size each segment's local
    /// [`TopDocsCollector`] so cross-segment merge has enough headroom
    /// without wasting work. Returns `None` for collectors without a
    /// meaningful top-K (e.g. [`CountCollector`]).
    fn requested_top_k(&self) -> Option<usize> {
        None
    }
}

/// A collector that keeps the top N documents by score.
#[derive(Debug)]
pub struct TopDocsCollector {
    /// Maximum number of documents to collect.
    max_docs: usize,
    /// Minimum score threshold.
    min_score: f32,
    /// Collected hits (min-heap based on score).
    hits: BinaryHeap<ScoredDoc>,
    /// Total number of documents processed.
    total_hits: u64,
}

/// A scored document for use in the heap.
#[derive(Debug, Clone)]
struct ScoredDoc {
    doc_id: u64,
    score: f32,
}

/// A document with field value for field-based sorting.
#[derive(Debug, Clone)]
struct FieldScoredDoc {
    doc_id: u64,
    score: f32,
    field_value: crate::lexical::core::field::FieldValue,
    ascending: bool,
}

/// A collector that keeps the top N documents sorted by a field value.
/// This performs sorting during collection (Lucene-style) rather than after.
#[derive(Debug)]
pub struct TopFieldCollector<'a> {
    /// Maximum number of documents to collect.
    max_docs: usize,
    /// Minimum score threshold.
    min_score: f32,
    /// Field name to sort by.
    field_name: String,
    /// Sort order (true for ascending, false for descending).
    ascending: bool,
    /// Collected hits (min-heap for ascending, needs reverse comparison).
    hits: BinaryHeap<FieldScoredDoc>,
    /// Total number of documents processed.
    total_hits: u64,
    /// Reference to the index reader for accessing field values.
    reader: &'a dyn crate::lexical::reader::LexicalIndexReader,
    /// Whether ANY segment has a DocValues column for `field_name`,
    /// resolved once at construction (Issue #1053). `field_name` is fixed
    /// for the whole collector's lifetime, so caching this here -- rather
    /// than re-probing per document -- avoids paying a lock-guarded
    /// `has_doc_values` check on every hit for no benefit; mirrors the
    /// same per-field caching `FacetCollector::collect_doc` does for the
    /// same reason (#597).
    ///
    /// This is index-wide, not per-document: `InvertedIndexReader::
    /// has_doc_values` is `any(...)` across segments (Issue #1047), so
    /// `true` does not guarantee `get_doc_value` will find a value for
    /// every doc -- a segment lacking the column entirely (mixed old/new
    /// segments, or a field with `doc_values: false`) still needs the
    /// stored-document fallback per miss. See [`Self::get_field_value`].
    has_dv: bool,
}

impl<'a> TopFieldCollector<'a> {
    /// Create a new top field collector.
    pub fn new(
        max_docs: usize,
        field_name: String,
        ascending: bool,
        reader: &'a dyn crate::lexical::reader::LexicalIndexReader,
    ) -> Self {
        let has_dv = reader.has_doc_values(&field_name);
        TopFieldCollector {
            max_docs,
            min_score: 0.0,
            field_name,
            ascending,
            hits: BinaryHeap::new(),
            total_hits: 0,
            reader,
            has_dv,
        }
    }

    /// Create a new top field collector with minimum score threshold.
    pub fn with_min_score(
        max_docs: usize,
        min_score: f32,
        field_name: String,
        ascending: bool,
        reader: &'a dyn crate::lexical::reader::LexicalIndexReader,
    ) -> Self {
        let has_dv = reader.has_doc_values(&field_name);
        TopFieldCollector {
            max_docs,
            min_score,
            field_name,
            ascending,
            hits: BinaryHeap::new(),
            total_hits: 0,
            reader,
            has_dv,
        }
    }

    /// Get the field value for a document, preferring DocValues.
    ///
    /// When `field_name` has no DocValues column at all, or a `Some`
    /// column exists index-wide but this particular document's segment
    /// doesn't have it (Issue #1047: mixed segments, e.g. straddling a
    /// `doc_values: false` change or a #1052-era segment boundary), falls
    /// back to the stored document instead of yielding `Null` --
    /// otherwise every such value compares equal and sorting silently
    /// degrades to doc-id order for exactly the documents that lack the
    /// column. Mirrors `FacetCollector::collect_doc`'s stored-document
    /// fallback for the same absent-column case. Uses `document_fields`
    /// rather than `document` to avoid cloning every field of a
    /// wide-schema document just to read one.
    fn get_field_value(&self, doc_id: u64) -> crate::lexical::core::field::FieldValue {
        use crate::lexical::core::field::FieldValue;

        if self.has_dv
            && let Ok(Some(value)) = self.reader.get_doc_value(&self.field_name, doc_id)
        {
            return value;
        }

        match self.reader.document_fields(doc_id, &[&self.field_name]) {
            Ok(Some(mut fields)) => fields.remove(&self.field_name).unwrap_or(FieldValue::Null),
            _ => FieldValue::Null,
        }
    }

    /// Check if a new document should be collected based on field value.
    ///
    /// Compares against the current heap-worst (`Ord`-greatest, see
    /// [`FieldScoredDoc::cmp`]) — a new doc is worth collecting when it
    /// ranks better (`Ordering::Less`) than the worst.
    fn should_collect(&self, new_doc: &FieldScoredDoc) -> bool {
        if self.hits.len() < self.max_docs {
            return true;
        }

        match self.hits.peek() {
            Some(worst) => new_doc.cmp(worst) == Ordering::Less,
            None => true,
        }
    }
}

impl<'a> Collector for TopFieldCollector<'a> {
    fn collect(&mut self, doc_id: u64, score: f32) -> Result<()> {
        self.total_hits += 1;

        if self.max_docs == 0 {
            return Ok(());
        }

        // Check minimum score threshold
        if score < self.min_score {
            return Ok(());
        }

        // Get field value during collection (Lucene-style)
        let field_value = self.get_field_value(doc_id);

        let scored_doc = FieldScoredDoc {
            doc_id,
            score,
            field_value,
            ascending: self.ascending,
        };

        if self.hits.len() < self.max_docs {
            // We have space, just add it
            self.hits.push(scored_doc);
        } else {
            // Check if this document should replace the worst one
            if self.should_collect(&scored_doc) {
                self.hits.pop();
                self.hits.push(scored_doc);
            }
        }

        Ok(())
    }

    fn results(&self) -> Vec<SearchHit> {
        // `FieldScoredDoc::Ord` is defined so that `Less` always means
        // "ranks better" (for both sort directions — see the `cmp` impl
        // below), so a plain ascending sort by `Ord` yields best-first
        // output regardless of `self.ascending`.
        let mut sorted_docs: Vec<_> = self.hits.iter().cloned().collect();
        sorted_docs.sort_unstable();

        sorted_docs
            .into_iter()
            .map(|doc| SearchHit {
                doc_id: doc.doc_id,
                score: doc.score,
                document: None,
            })
            .collect()
    }

    fn total_hits(&self) -> u64 {
        self.total_hits
    }

    fn needs_more(&self) -> bool {
        // Field-sort values (dates, popularity, ...) have no algebraic
        // upper bound over doc_id-ordered iteration the way BM25 scores
        // do, so — unlike `TopDocsCollector` (#459) — this collector
        // cannot expose a meaningful `min_competitive()` to drive early
        // termination. The previous `self.hits.len() < self.max_docs`
        // short-circuit stopped the searcher loop the instant the heap
        // filled, so any later-iterated doc with a better field value
        // was silently dropped — returning the first K matches by
        // doc_id rather than the true top K by field value (#608).
        // `min_competitive()` / `bmw_capable()` intentionally stay at
        // their trait defaults (`NEG_INFINITY` / `false`): every
        // candidate must be visited for a field-sorted search to be
        // correct.
        true
    }

    fn min_score(&self) -> f32 {
        self.min_score
    }

    fn reset(&mut self) {
        self.hits.clear();
        self.total_hits = 0;
    }
}

impl PartialEq for FieldScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.field_value == other.field_value && self.doc_id == other.doc_id
    }
}

impl Eq for FieldScoredDoc {}

impl PartialOrd for FieldScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Natural ascending order over sort-field values, used as the single
/// source of truth for both [`FieldScoredDoc::cmp`] (heap order) and
/// [`TopFieldCollector::results`] (final output order) — see #608.
///
/// `Null` always sorts greatest (last in ascending order), independent
/// of sort direction; direction is applied by the caller via
/// [`Ordering::reverse`].
///
/// Mixed-type pairs (#945): `Int64` vs `Float64` compare numerically —
/// a dynamic-schema field can store `42` and `42.5` as different
/// variants, and a type-priority order would be numerically wrong for
/// them. Every other mixed pair orders by [`sort_type_rank`], a
/// deterministic type precedence. Same-rank pairs without a per-type
/// comparison (e.g. two arrays) compare `Equal`; the caller's doc-id
/// tie-break keeps the final order deterministic.
fn compare_sort_key(
    a: &crate::lexical::core::field::FieldValue,
    b: &crate::lexical::core::field::FieldValue,
) -> Ordering {
    use crate::lexical::core::field::FieldValue;

    match (a, b) {
        (FieldValue::Null, FieldValue::Null) => Ordering::Equal,
        (FieldValue::Null, _) => Ordering::Greater,
        (_, FieldValue::Null) => Ordering::Less,
        (FieldValue::Text(a), FieldValue::Text(b)) => a.cmp(b),
        (FieldValue::Int64(a), FieldValue::Int64(b)) => a.cmp(b),
        (FieldValue::Float64(a), FieldValue::Float64(b)) => a.total_cmp(b),
        (FieldValue::Int64(a), FieldValue::Float64(b)) => (*a as f64).total_cmp(b),
        (FieldValue::Float64(a), FieldValue::Int64(b)) => a.total_cmp(&(*b as f64)),
        (FieldValue::Bool(a), FieldValue::Bool(b)) => a.cmp(b),
        (FieldValue::DateTime(a), FieldValue::DateTime(b)) => a.cmp(b),
        (FieldValue::Geo(a), FieldValue::Geo(b)) => a
            .lat
            .total_cmp(&b.lat)
            .then_with(|| a.lon.total_cmp(&b.lon)),
        (FieldValue::Bytes(a, _), FieldValue::Bytes(b, _)) => a.cmp(b),
        _ => sort_type_rank(a).cmp(&sort_type_rank(b)),
    }
}

/// Deterministic precedence for mixed-type sort-key pairs (#945):
/// `Bool < numeric < DateTime < Text < Geo < GeoEcef < Bytes < Vector <
/// Int64Array < Float64Array`. `Null` never reaches this (handled first
/// in [`compare_sort_key`]); `Int64` and `Float64` share a rank because
/// they compare numerically instead. Exhaustive on purpose: adding a
/// `FieldValue` variant must force a rank decision here.
fn sort_type_rank(v: &crate::lexical::core::field::FieldValue) -> u8 {
    use crate::lexical::core::field::FieldValue;

    match v {
        FieldValue::Null => 0,
        FieldValue::Bool(_) => 1,
        FieldValue::Int64(_) | FieldValue::Float64(_) => 2,
        FieldValue::DateTime(_) => 3,
        FieldValue::Text(_) => 4,
        FieldValue::Geo(_) => 5,
        FieldValue::GeoEcef(_) => 6,
        FieldValue::Bytes(_, _) => 7,
        FieldValue::Vector(_) => 8,
        FieldValue::Int64Array(_) => 9,
        FieldValue::Float64Array(_) => 10,
    }
}

impl Ord for FieldScoredDoc {
    /// Heap order: `Ordering::Less` always means "ranks better", for
    /// both sort directions — `BinaryHeap` (a max-heap) then naturally
    /// pops the `Ord`-greatest element, which is always the current
    /// *worst* ranked document, i.e. the correct eviction target
    /// (#608; mirrors [`ScoredDoc::cmp`]'s min-heap-via-reversal
    /// pattern for `TopDocsCollector`).
    fn cmp(&self, other: &Self) -> Ordering {
        let key = compare_sort_key(&self.field_value, &other.field_value);
        let by_rank = if self.ascending { key } else { key.reverse() };
        by_rank.then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}

impl Eq for ScoredDoc {}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower scores come first
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.doc_id.cmp(&self.doc_id))
    }
}

impl TopDocsCollector {
    /// Create a new top docs collector.
    pub fn new(max_docs: usize) -> Self {
        TopDocsCollector {
            max_docs,
            min_score: 0.0,
            hits: BinaryHeap::new(),
            total_hits: 0,
        }
    }

    /// Create a new top docs collector with minimum score threshold.
    pub fn with_min_score(max_docs: usize, min_score: f32) -> Self {
        TopDocsCollector {
            max_docs,
            min_score,
            hits: BinaryHeap::new(),
            total_hits: 0,
        }
    }

    /// Get the maximum number of documents to collect.
    pub fn max_docs(&self) -> usize {
        self.max_docs
    }

    /// Get the current minimum score in the collection.
    pub fn current_min_score(&self) -> f32 {
        if self.hits.len() < self.max_docs {
            self.min_score
        } else {
            self.hits
                .peek()
                .map(|doc| doc.score)
                .unwrap_or(self.min_score)
        }
    }
}

impl Collector for TopDocsCollector {
    fn collect(&mut self, doc_id: u64, score: f32) -> Result<()> {
        self.total_hits += 1;

        // Check minimum score threshold
        if score < self.min_score {
            return Ok(());
        }

        let scored_doc = ScoredDoc { doc_id, score };

        if self.hits.len() < self.max_docs {
            // We have space, just add it
            self.hits.push(scored_doc);
        } else {
            // Check if this score is better than the worst score
            if let Some(worst) = self.hits.peek()
                && score > worst.score
            {
                // Replace the worst document
                self.hits.pop();
                self.hits.push(scored_doc);
            }
        }

        Ok(())
    }

    fn results(&self) -> Vec<SearchHit> {
        let mut results: Vec<_> = self
            .hits
            .iter()
            .map(|doc| SearchHit {
                doc_id: doc.doc_id,
                score: doc.score,
                document: None,
            })
            .collect();

        // Sort by score descending
        results.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

        results
    }

    fn total_hits(&self) -> u64 {
        self.total_hits
    }

    fn needs_more(&self) -> bool {
        // Top-K collectors must see every candidate: a doc later in
        // the iteration with a higher score still needs to displace
        // the current heap-min. The previous
        // `self.hits.len() < self.max_docs` short-circuit returned the
        // first K matches by `doc_id` rather than the highest-scoring
        // K (#459). Early termination is now driven by
        // [`Collector::min_competitive`] + [`Scorer::max_score`] in
        // the searcher loop, which only fires when no future doc can
        // beat the current K-th score.
        true
    }

    fn min_score(&self) -> f32 {
        self.current_min_score()
    }

    fn min_competitive(&self) -> f32 {
        // The MaxScore early-termination signal: once the heap is full
        // the heap-min equals the K-th best score, and any future doc
        // whose upper-bound score is ≤ this value cannot enter the
        // top-K. While the heap has spare slots, return
        // `NEG_INFINITY` so the searcher does not attempt to skip
        // anything before the heap fills.
        //
        // Currently dead code in the default search path because
        // `needs_more()` short-circuits the loop the moment the heap
        // fills — fixing that without per-posting block-max metadata
        // would walk every candidate and regress wall time on common
        // OR workloads (the global `BM25Scorer::max_score()` upper
        // bound is too loose at `k1 + 1`). The correctness fix +
        // MaxScore activation lands together with the block-max index
        // format in the next PR (#403 PR-B onward).
        if self.hits.len() < self.max_docs {
            f32::NEG_INFINITY
        } else {
            self.hits.peek().map(|d| d.score).unwrap_or(self.min_score)
        }
    }

    fn bmw_capable(&self) -> bool {
        true
    }

    fn requested_top_k(&self) -> Option<usize> {
        Some(self.max_docs)
    }

    fn reset(&mut self) {
        self.hits.clear();
        self.total_hits = 0;
    }
}

/// A collector that just counts the number of matching documents.
#[derive(Debug)]
pub struct CountCollector {
    /// Total number of documents that matched.
    count: u64,
    /// Minimum score threshold.
    min_score: f32,
}

impl CountCollector {
    /// Create a new count collector.
    pub fn new() -> Self {
        CountCollector {
            count: 0,
            min_score: 0.0,
        }
    }

    /// Create a new count collector with minimum score threshold.
    pub fn with_min_score(min_score: f32) -> Self {
        CountCollector {
            count: 0,
            min_score,
        }
    }

    /// Get the current count.
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for CountCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for CountCollector {
    fn collect(&mut self, _doc_id: u64, score: f32) -> Result<()> {
        if score >= self.min_score {
            self.count += 1;
        }
        Ok(())
    }

    fn results(&self) -> Vec<SearchHit> {
        // Count collector doesn't return actual documents
        Vec::new()
    }

    fn total_hits(&self) -> u64 {
        self.count
    }

    fn needs_more(&self) -> bool {
        // Count collector always needs more to get the full count
        true
    }

    fn min_score(&self) -> f32 {
        self.min_score
    }

    fn reset(&mut self) {
        self.count = 0;
    }
}

/// A collector that collects all matching documents.
#[derive(Debug)]
pub struct AllDocsCollector {
    /// All collected hits.
    hits: Vec<SearchHit>,
    /// Minimum score threshold.
    min_score: f32,
    /// Cached sorted results to avoid repeated clone+sort.
    sorted_cache: RefCell<Option<Vec<SearchHit>>>,
}

impl AllDocsCollector {
    /// Create a new all docs collector.
    pub fn new() -> Self {
        AllDocsCollector {
            hits: Vec::new(),
            min_score: 0.0,
            sorted_cache: RefCell::new(None),
        }
    }

    /// Create a new all docs collector with minimum score threshold.
    pub fn with_min_score(min_score: f32) -> Self {
        AllDocsCollector {
            hits: Vec::new(),
            min_score,
            sorted_cache: RefCell::new(None),
        }
    }
}

impl Default for AllDocsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for AllDocsCollector {
    fn collect(&mut self, doc_id: u64, score: f32) -> Result<()> {
        if score >= self.min_score {
            self.hits.push(SearchHit {
                doc_id,
                score,
                document: None,
            });
            // Invalidate cached sorted results.
            *self.sorted_cache.borrow_mut() = None;
        }
        Ok(())
    }

    fn results(&self) -> Vec<SearchHit> {
        let mut cache = self.sorted_cache.borrow_mut();
        if let Some(ref cached) = *cache {
            return cached.clone();
        }
        let mut results = self.hits.clone();
        results.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        *cache = Some(results.clone());
        results
    }

    fn total_hits(&self) -> u64 {
        self.hits.len() as u64
    }

    fn needs_more(&self) -> bool {
        // All docs collector always needs more
        true
    }

    fn min_score(&self) -> f32 {
        self.min_score
    }

    fn reset(&mut self) {
        self.hits.clear();
        *self.sorted_cache.borrow_mut() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_docs_collector() {
        let mut collector = TopDocsCollector::new(3);

        assert_eq!(collector.max_docs(), 3);
        assert_eq!(collector.total_hits(), 0);
        assert!(collector.needs_more());

        // Add some documents
        collector.collect(1, 0.5).unwrap();
        collector.collect(2, 0.8).unwrap();
        collector.collect(3, 0.3).unwrap();

        assert_eq!(collector.total_hits(), 3);
        // `needs_more` is `true` even when the heap is full, because a
        // higher-scoring doc later in the iteration can still displace
        // the current worst. Early termination is now driven by
        // `min_competitive` + `Scorer::max_score()` in the searcher
        // loop (see #459).
        assert!(collector.needs_more());

        // Add a better document - should replace the worst
        collector.collect(4, 0.9).unwrap();

        assert_eq!(collector.total_hits(), 4);

        let results = collector.results();
        assert_eq!(results.len(), 3);

        // Results should be sorted by score descending
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);

        // The best document should be first
        assert_eq!(results[0].doc_id, 4);
        assert_eq!(results[0].score, 0.9);
    }

    #[test]
    fn test_top_docs_collector_with_min_score() {
        let mut collector = TopDocsCollector::with_min_score(3, 0.5);

        assert_eq!(collector.min_score(), 0.5);

        // Add documents, some below threshold
        collector.collect(1, 0.3).unwrap(); // Below threshold
        collector.collect(2, 0.8).unwrap(); // Above threshold
        collector.collect(3, 0.6).unwrap(); // Above threshold

        assert_eq!(collector.total_hits(), 3);

        let results = collector.results();
        assert_eq!(results.len(), 2); // Only 2 above threshold

        // Check that low-score document was filtered out
        assert!(!results.iter().any(|hit| hit.score == 0.3));
    }

    #[test]
    fn test_top_docs_collector_replaces_lower_after_full() {
        // Verifies the correctness contract restored by the
        // `needs_more = true` change (#459): a higher-scoring doc that
        // arrives **after** the heap fills must displace the current
        // heap-min, regardless of doc-id order. The pre-fix collector
        // short-circuited the search loop the moment the heap filled
        // and returned the first K matches by `doc_id`, dropping any
        // higher-scoring doc that came later in the iteration.
        let mut collector = TopDocsCollector::new(3);

        collector.collect(1, 0.10).unwrap();
        collector.collect(2, 0.20).unwrap();
        collector.collect(3, 0.30).unwrap();
        assert!(collector.needs_more());

        collector.collect(4, 0.99).unwrap();

        let results = collector.results();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].doc_id, 4);
        assert!((results[0].score - 0.99).abs() < 1e-6);
        // Lowest-scoring doc (1) should have been evicted.
        assert!(results.iter().all(|hit| hit.doc_id != 1));
    }

    #[test]
    fn test_top_docs_collector_min_competitive() {
        // Validates the #403 PR-A `min_competitive` signal:
        //   - empty / partially full heap → `NEG_INFINITY` (no useful
        //     bound; the searcher must keep collecting).
        //   - full heap → heap-min, i.e. the K-th best score.
        //   - replaces the worst when a higher score arrives → bound
        //     advances upward.
        let mut collector = TopDocsCollector::new(3);

        assert_eq!(collector.min_competitive(), f32::NEG_INFINITY);

        collector.collect(1, 0.5).unwrap();
        assert_eq!(collector.min_competitive(), f32::NEG_INFINITY);

        collector.collect(2, 0.8).unwrap();
        assert_eq!(collector.min_competitive(), f32::NEG_INFINITY);

        collector.collect(3, 0.3).unwrap();
        // Heap full — bound is the K-th score (lowest of the three).
        assert!((collector.min_competitive() - 0.3).abs() < 1e-6);

        // A higher-scoring doc tightens the bound to the next-lowest.
        collector.collect(4, 0.9).unwrap();
        assert!((collector.min_competitive() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_count_collector() {
        let mut collector = CountCollector::new();

        assert_eq!(collector.count(), 0);
        assert_eq!(collector.total_hits(), 0);
        assert!(collector.needs_more());

        // Add some documents
        collector.collect(1, 0.5).unwrap();
        collector.collect(2, 0.8).unwrap();
        collector.collect(3, 0.3).unwrap();

        assert_eq!(collector.count(), 3);
        assert_eq!(collector.total_hits(), 3);

        // Results should be empty for count collector
        let results = collector.results();
        assert!(results.is_empty());
    }

    #[test]
    fn test_count_collector_with_min_score() {
        let mut collector = CountCollector::with_min_score(0.5);

        // Add documents, some below threshold
        collector.collect(1, 0.3).unwrap(); // Below threshold
        collector.collect(2, 0.8).unwrap(); // Above threshold
        collector.collect(3, 0.6).unwrap(); // Above threshold

        assert_eq!(collector.count(), 2); // Only 2 above threshold
        assert_eq!(collector.total_hits(), 2);
    }

    #[test]
    fn test_all_docs_collector() {
        let mut collector = AllDocsCollector::new();

        assert_eq!(collector.total_hits(), 0);
        assert!(collector.needs_more());

        // Add some documents
        collector.collect(1, 0.5).unwrap();
        collector.collect(2, 0.8).unwrap();
        collector.collect(3, 0.3).unwrap();

        assert_eq!(collector.total_hits(), 3);

        let results = collector.results();
        assert_eq!(results.len(), 3);

        // Results should be sorted by score descending
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);

        // Check specific order
        assert_eq!(results[0].doc_id, 2); // score 0.8
        assert_eq!(results[1].doc_id, 1); // score 0.5
        assert_eq!(results[2].doc_id, 3); // score 0.3
    }

    #[test]
    fn test_all_docs_collector_with_min_score() {
        let mut collector = AllDocsCollector::with_min_score(0.5);

        // Add documents, some below threshold
        collector.collect(1, 0.3).unwrap(); // Below threshold
        collector.collect(2, 0.8).unwrap(); // Above threshold
        collector.collect(3, 0.6).unwrap(); // Above threshold

        assert_eq!(collector.total_hits(), 2); // Only 2 above threshold

        let results = collector.results();
        assert_eq!(results.len(), 2);

        // Check that low-score document was filtered out
        assert!(!results.iter().any(|hit| hit.score == 0.3));
    }

    #[test]
    fn test_collector_reset() {
        let mut collector = TopDocsCollector::new(3);

        // Add some documents
        collector.collect(1, 0.5).unwrap();
        collector.collect(2, 0.8).unwrap();

        assert_eq!(collector.total_hits(), 2);
        assert_eq!(collector.results().len(), 2);

        // Reset collector
        collector.reset();

        assert_eq!(collector.total_hits(), 0);
        assert_eq!(collector.results().len(), 0);
        assert!(collector.needs_more());
    }

    #[test]
    fn test_scored_doc_ordering() {
        let doc1 = ScoredDoc {
            doc_id: 1,
            score: 0.5,
        };
        let doc2 = ScoredDoc {
            doc_id: 2,
            score: 0.8,
        };
        let doc3 = ScoredDoc {
            doc_id: 3,
            score: 0.8,
        };

        // Higher score should be "less" (for min-heap)
        assert!(doc2 < doc1);

        // Same score should compare by doc_id
        assert!(doc3 < doc2);

        // Test in heap
        let mut heap = BinaryHeap::new();
        heap.push(doc1);
        heap.push(doc2);
        heap.push(doc3);

        // Min-heap: lowest score should be at top
        assert_eq!(heap.peek().unwrap().score, 0.5);
    }

    /// Minimal reader that serves a fixed `doc_id -> FieldValue` map via
    /// DocValues, for [`TopFieldCollector`] tests. Mirrors the shape of
    /// `DvMockReader` in `lexical::search::features::facet`.
    #[derive(Debug)]
    struct FieldValueMockReader {
        values: std::collections::HashMap<u64, crate::lexical::core::field::FieldValue>,
    }

    impl FieldValueMockReader {
        fn new(values: &[(u64, crate::lexical::core::field::FieldValue)]) -> Self {
            Self {
                values: values.iter().cloned().collect(),
            }
        }
    }

    impl crate::lexical::reader::LexicalIndexReader for FieldValueMockReader {
        fn doc_count(&self) -> u64 {
            self.values.len() as u64
        }
        fn max_doc(&self) -> u64 {
            self.values.len() as u64
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(&self, _doc_id: u64) -> Result<Option<crate::Document>> {
            Ok(None)
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>> {
            Ok(None)
        }
        fn field_stats(&self, _field: &str) -> Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn get_doc_value(
            &self,
            _field: &str,
            doc_id: u64,
        ) -> Result<Option<crate::lexical::core::field::FieldValue>> {
            Ok(self.values.get(&doc_id).cloned())
        }
        fn has_doc_values(&self, _field: &str) -> bool {
            // This mock exists to exercise the DocValues path -- see
            // `DocFallbackMockReader` below for the stored-document
            // fallback path (Issue #1053).
            true
        }
    }

    /// Build a `TopFieldCollector` over an `Int64` field named `"value"`
    /// and feed it `docs` (`(doc_id, value)` pairs) in order.
    fn collect_field_sorted(docs: &[(u64, i64)], max_docs: usize, ascending: bool) -> Vec<u64> {
        use crate::lexical::core::field::FieldValue;

        let values: Vec<(u64, FieldValue)> = docs
            .iter()
            .map(|(id, v)| (*id, FieldValue::Int64(*v)))
            .collect();
        let reader = FieldValueMockReader::new(&values);
        let mut collector =
            TopFieldCollector::new(max_docs, "value".to_string(), ascending, &reader);
        for (doc_id, _) in docs {
            collector.collect(*doc_id, 0.0).unwrap();
        }
        collector.results().iter().map(|h| h.doc_id).collect()
    }

    // Regression tests for #608: `TopFieldCollector` had two independent
    // bugs. (1) `needs_more()` short-circuited once the heap filled,
    // mirroring the pre-#459 `TopDocsCollector` bug — the searcher loop
    // stopped scanning candidates entirely once `max_docs` hits were
    // collected. (2) Because of (1), `collect()`'s eviction branch never
    // ran in production, hiding the fact that `FieldScoredDoc::Ord` was
    // inverted for BOTH sort directions, so once eviction *did* run it
    // discarded the best candidate instead of the worst. These tests call
    // `collect()` directly (bypassing the searcher loop's `needs_more()`
    // gate entirely) so bug (2) is provable in isolation from bug (1).

    #[test]
    fn test_top_field_collector_desc_evicts_lowest_not_highest() {
        // Values arrive in ascending doc_id / value order; only the last
        // 3 (largest) values are the correct descending top-3.
        let docs: Vec<(u64, i64)> = (1..=5).map(|i| (i, i as i64)).collect();
        let ids = collect_field_sorted(&docs, 3, false);
        assert_eq!(ids, vec![5, 4, 3], "descending top-3 by value");
    }

    #[test]
    fn test_top_field_collector_asc_evicts_highest_not_lowest() {
        let docs: Vec<(u64, i64)> = (1..=5).map(|i| (i, i as i64)).collect();
        let ids = collect_field_sorted(&docs, 3, true);
        assert_eq!(ids, vec![1, 2, 3], "ascending top-3 by value");
    }

    #[test]
    fn test_top_field_collector_needs_more_stays_true_when_full() {
        // Verifies the correctness contract restored by the
        // `needs_more = true` change (mirrors #459's fix for
        // `TopDocsCollector`): a document later in the iteration with a
        // better field value must still be able to displace the current
        // heap-worst. Early termination for field sorts is intentionally
        // left to the searcher's default `min_competitive = NEG_INFINITY`
        // (i.e. no early termination — see `TopFieldCollector::needs_more`
        // doc comment for why enabling BMW here would be unsound).
        use crate::lexical::core::field::FieldValue;

        let values: Vec<(u64, FieldValue)> =
            vec![(1, FieldValue::Int64(10)), (2, FieldValue::Int64(20))];
        let reader = FieldValueMockReader::new(&values);
        let mut collector = TopFieldCollector::new(2, "value".to_string(), false, &reader);

        assert!(collector.needs_more());
        collector.collect(1, 0.0).unwrap();
        collector.collect(2, 0.0).unwrap();

        assert!(
            collector.needs_more(),
            "must stay true even when the heap is full, so later docs can still evict"
        );
    }

    #[test]
    fn test_top_field_collector_results_are_deterministic_on_ties() {
        // All docs share the same field value; the tie-break (doc_id)
        // must make the result order deterministic instead of depending
        // on `BinaryHeap`'s unspecified iteration order.
        let docs: Vec<(u64, i64)> = vec![(3, 1), (1, 1), (2, 1)];
        let ids = collect_field_sorted(&docs, 3, true);
        assert_eq!(ids, vec![1, 2, 3], "ties break by doc_id ascending");
    }

    #[test]
    fn test_top_field_collector_null_ordering() {
        use crate::lexical::core::field::FieldValue;

        let values: Vec<(u64, FieldValue)> = vec![
            (1, FieldValue::Int64(5)),
            (2, FieldValue::Null),
            (3, FieldValue::Int64(1)),
        ];
        let reader = FieldValueMockReader::new(&values);

        // Ascending: Null sorts last (greatest).
        let mut asc = TopFieldCollector::new(3, "value".to_string(), true, &reader);
        for (doc_id, _) in &values {
            asc.collect(*doc_id, 0.0).unwrap();
        }
        let asc_ids: Vec<u64> = asc.results().iter().map(|h| h.doc_id).collect();
        assert_eq!(asc_ids, vec![3, 1, 2]);

        // Descending reverses the whole rank (including the Null
        // placement), so Null sorts first here — this matches the
        // pre-#608 `results()` behavior, which is unchanged by this fix.
        let mut desc = TopFieldCollector::new(3, "value".to_string(), false, &reader);
        for (doc_id, _) in &values {
            desc.collect(*doc_id, 0.0).unwrap();
        }
        let desc_ids: Vec<u64> = desc.results().iter().map(|h| h.doc_id).collect();
        assert_eq!(desc_ids, vec![2, 1, 3]);
    }

    #[test]
    fn test_top_field_collector_zero_limit_returns_empty() {
        let docs: Vec<(u64, i64)> = vec![(1, 10)];
        let ids = collect_field_sorted(&docs, 0, true);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_top_field_collector_reset() {
        use crate::lexical::core::field::FieldValue;

        let values: Vec<(u64, FieldValue)> = vec![(1, FieldValue::Int64(1))];
        let reader = FieldValueMockReader::new(&values);
        let mut collector = TopFieldCollector::new(3, "value".to_string(), true, &reader);

        collector.collect(1, 0.0).unwrap();
        assert_eq!(collector.total_hits(), 1);
        assert_eq!(collector.results().len(), 1);

        collector.reset();

        assert_eq!(collector.total_hits(), 0);
        assert_eq!(collector.results().len(), 0);
        assert!(collector.needs_more());
    }

    /// #945: `Int64` and `Float64` for the same sort field (dynamic
    /// schema can store `42` and `42.5` as different variants) must
    /// compare numerically, not fall to the mixed-type fallback.
    #[test]
    fn sort_key_compares_int_and_float_numerically() {
        use crate::lexical::core::field::FieldValue;

        assert_eq!(
            compare_sort_key(&FieldValue::Int64(1), &FieldValue::Float64(2.5)),
            Ordering::Less
        );
        assert_eq!(
            compare_sort_key(&FieldValue::Float64(2.5), &FieldValue::Int64(3)),
            Ordering::Less
        );
        assert_eq!(
            compare_sort_key(&FieldValue::Float64(3.5), &FieldValue::Int64(3)),
            Ordering::Greater
        );
        assert_eq!(
            compare_sort_key(&FieldValue::Int64(2), &FieldValue::Float64(2.0)),
            Ordering::Equal
        );
    }

    /// #945: mixed non-numeric type pairs must order by a deterministic
    /// type rank instead of collapsing to `Equal`.
    #[test]
    fn sort_key_orders_mixed_types_by_rank() {
        use crate::lexical::core::field::FieldValue;

        // numeric < Text
        assert_eq!(
            compare_sort_key(&FieldValue::Int64(999), &FieldValue::Text("a".into())),
            Ordering::Less
        );
        assert_eq!(
            compare_sort_key(&FieldValue::Text("a".into()), &FieldValue::Float64(999.0)),
            Ordering::Greater
        );
        // Bool < numeric
        assert_eq!(
            compare_sort_key(&FieldValue::Bool(true), &FieldValue::Int64(0)),
            Ordering::Less
        );
        // Text < Bytes
        assert_eq!(
            compare_sort_key(
                &FieldValue::Text("z".into()),
                &FieldValue::Bytes(vec![0], None)
            ),
            Ordering::Less
        );
    }

    /// #945: the documented `Null`-sorts-greatest contract (#608) must
    /// hold against mixed types too.
    #[test]
    fn sort_key_null_stays_greatest() {
        use crate::lexical::core::field::FieldValue;

        for other in [
            FieldValue::Bool(true),
            FieldValue::Int64(i64::MAX),
            FieldValue::Float64(f64::INFINITY),
            FieldValue::Text("zzz".into()),
            FieldValue::Bytes(vec![255], None),
        ] {
            assert_eq!(
                compare_sort_key(&FieldValue::Null, &other),
                Ordering::Greater,
                "Null must sort greatest vs {other:?}"
            );
            assert_eq!(compare_sort_key(&other, &FieldValue::Null), Ordering::Less);
        }
    }

    /// Reader with no DocValues at all, serving field values from stored
    /// documents instead (Issue #1053). Mirrors the shape of `DvMockReader`
    /// in `lexical::search::features::facet`, minus the per-field DocValues
    /// toggle -- `TopFieldCollector` tests here only need the
    /// no-DocValues-at-all case, since `has_dv` is resolved once for the
    /// single sort field, not per document.
    #[derive(Debug)]
    struct DocFallbackMockReader {
        docs: Vec<crate::Document>,
    }

    impl DocFallbackMockReader {
        fn new(docs: Vec<crate::Document>) -> Self {
            Self { docs }
        }
    }

    impl crate::lexical::reader::LexicalIndexReader for DocFallbackMockReader {
        fn doc_count(&self) -> u64 {
            self.docs.len() as u64
        }
        fn max_doc(&self) -> u64 {
            self.docs.len() as u64
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(&self, doc_id: u64) -> Result<Option<crate::Document>> {
            Ok(self.docs.get(doc_id as usize).cloned())
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>> {
            Ok(None)
        }
        fn field_stats(&self, _field: &str) -> Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        // `get_doc_value`/`has_doc_values` deliberately left at the
        // trait's defaults (`Ok(None)` / `false`): this reader simulates a
        // field with no DocValues column at all.
    }

    /// Issue #1053 regression: sorting by a `Bytes` field with no
    /// DocValues column must order by content, not silently degrade to
    /// doc-id order. Three documents hold content `[2]`, `[3]`, `[1]` at
    /// doc ids `0`, `1`, `2` respectively, so content order (`[1] < [2] <
    /// [3]`, i.e. doc ids `2, 0, 1`) differs from doc-id order (`0, 1,
    /// 2`) -- this one test is red before the stored-document fallback
    /// (every value resolves to `Null`, yielding doc-id order `[0, 1,
    /// 2]`) and green after (content order `[2, 0, 1]`), directly
    /// distinguishing the two behaviors.
    #[test]
    fn test_top_field_collector_bytes_sort_falls_back_to_stored_document() {
        let docs = vec![
            crate::Document::builder()
                .add_bytes("blob", vec![2])
                .build(), // doc_id 0
            crate::Document::builder()
                .add_bytes("blob", vec![3])
                .build(), // doc_id 1
            crate::Document::builder()
                .add_bytes("blob", vec![1])
                .build(), // doc_id 2
        ];
        let reader = DocFallbackMockReader::new(docs);
        let mut collector = TopFieldCollector::new(3, "blob".to_string(), true, &reader);
        for doc_id in 0..3 {
            collector.collect(doc_id, 0.0).unwrap();
        }
        let ids: Vec<u64> = collector.results().iter().map(|h| h.doc_id).collect();
        assert_eq!(
            ids,
            vec![2, 0, 1],
            "must order by content ([1] < [2] < [3]), not doc id"
        );
    }

    /// Issue #1053 acceptance criterion: `Vector` fields genuinely have no
    /// ordering (`compare_sort_key` falls through to `sort_type_rank`,
    /// which gives every `Vector` the same rank), so the stored-document
    /// fallback must not change their sort order -- it stays doc-id order
    /// either way.
    #[test]
    fn test_top_field_collector_vector_sort_unaffected_by_stored_document_fallback() {
        let docs = vec![
            crate::Document::builder()
                .add_vector("embedding", vec![3.0, 0.0])
                .build(), // doc_id 0
            crate::Document::builder()
                .add_vector("embedding", vec![1.0, 0.0])
                .build(), // doc_id 1
            crate::Document::builder()
                .add_vector("embedding", vec![2.0, 0.0])
                .build(), // doc_id 2
        ];
        let reader = DocFallbackMockReader::new(docs);
        let mut collector = TopFieldCollector::new(3, "embedding".to_string(), true, &reader);
        for doc_id in 0..3 {
            collector.collect(doc_id, 0.0).unwrap();
        }
        let ids: Vec<u64> = collector.results().iter().map(|h| h.doc_id).collect();
        assert_eq!(
            ids,
            vec![0, 1, 2],
            "Vector values always tie on rank, so order stays doc-id order"
        );
    }

    /// Reader where `has_doc_values` reports `true` (simulating "some
    /// segment has this column") but `get_doc_value` always misses
    /// (simulating THIS document's segment lacking the column) — the
    /// mixed-segment case Issue #1047 identified as unreachable via
    /// `DocFallbackMockReader` (which reports `has_doc_values = false`
    /// index-wide, the "no column anywhere" case #1053 already covered).
    #[derive(Debug)]
    struct DocValuesMissMockReader {
        docs: Vec<crate::Document>,
    }

    impl DocValuesMissMockReader {
        fn new(docs: Vec<crate::Document>) -> Self {
            Self { docs }
        }
    }

    impl crate::lexical::reader::LexicalIndexReader for DocValuesMissMockReader {
        fn doc_count(&self) -> u64 {
            self.docs.len() as u64
        }
        fn max_doc(&self) -> u64 {
            self.docs.len() as u64
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(&self, doc_id: u64) -> Result<Option<crate::Document>> {
            Ok(self.docs.get(doc_id as usize).cloned())
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>> {
            Ok(None)
        }
        fn field_stats(&self, _field: &str) -> Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn has_doc_values(&self, _field: &str) -> bool {
            true
        }
        // `get_doc_value` deliberately left at the trait's default
        // (`Ok(None)`): every lookup misses, as if this doc's segment
        // never wrote the column despite another segment having it.
    }

    /// Issue #1047 regression: when `has_doc_values` is `true` index-wide
    /// but this document's segment misses (`get_doc_value` returns
    /// `Ok(None)`), `get_field_value` must fall back to the stored
    /// document instead of collapsing to `Null`. Before the fix every
    /// value resolved to `Null` (doc-id order `[0, 1, 2]`); after, content
    /// order (`[1] < [2] < [3]`, i.e. `[2, 0, 1]`).
    #[test]
    fn test_top_field_collector_falls_back_when_has_dv_is_true_but_the_value_is_missing() {
        let docs = vec![
            crate::Document::builder()
                .add_bytes("blob", vec![2])
                .build(), // doc_id 0
            crate::Document::builder()
                .add_bytes("blob", vec![3])
                .build(), // doc_id 1
            crate::Document::builder()
                .add_bytes("blob", vec![1])
                .build(), // doc_id 2
        ];
        let reader = DocValuesMissMockReader::new(docs);
        let mut collector = TopFieldCollector::new(3, "blob".to_string(), true, &reader);
        for doc_id in 0..3 {
            collector.collect(doc_id, 0.0).unwrap();
        }
        let ids: Vec<u64> = collector.results().iter().map(|h| h.doc_id).collect();
        assert_eq!(
            ids,
            vec![2, 0, 1],
            "a DocValues miss with has_dv=true must still fall back to the \
             stored document, ordering by content ([1] < [2] < [3]) not doc id"
        );
    }
}
