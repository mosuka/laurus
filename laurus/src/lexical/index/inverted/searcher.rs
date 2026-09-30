//! Searcher implementation for executing queries against an index.

use std::sync::Arc;
use std::time::Duration;

use ahash::AHashMap;
use roaring::RoaringTreemap;

use crate::util::time::Timer;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::error::{LaurusError, Result};
use crate::lexical::index::inverted::bmw::{BlockMaxOrExecutor, is_bmw_eligible};
use crate::lexical::index::inverted::parsed_query_cache::ParsedQueryCache;
use crate::lexical::index::inverted::per_segment_view::PerSegmentReaderView;
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::Query;
use crate::lexical::query::boolean::{BooleanQuery, Occur};
use crate::lexical::query::collector::{
    Collector, CountCollector, FieldHit, FieldTopK, TopDocsCollector, TopFieldCollector,
};
use crate::lexical::query::parser::LexicalQueryParser;
use crate::lexical::query::term::TermQuery;
use crate::lexical::query::{LexicalSearchResults, SearchHit};
use crate::lexical::reader::LexicalIndexReader;
use crate::lexical::search::searcher::{
    LexicalSearchParams, LexicalSearchQuery, LexicalSearchRequest, SortField, SortOrder,
};

/// Default capacity (entries) of the per-searcher parsed-DSL query cache
/// (Issue #590) when none is configured.
const DEFAULT_PARSED_QUERY_CACHE_CAPACITY: usize = 1024;

/// How often the scan loops consult the wall clock for a search deadline,
/// measured in scanned documents (Issue #600). Checking every document would
/// make `Timer::elapsed` a per-doc cost; checking once per this many keeps the
/// overhead negligible while still bounding worst-case latency (the same
/// batched approach as Lucene's `TimeLimitingCollector`).
const DEADLINE_CHECK_INTERVAL: u64 = 2048;

/// A segment's sort-field value range, read from its BKD header (#944).
///
/// `min` / `max` are in the same `f64` space the writer indexed the
/// points in, so a sort key is compared by converting it the same way
/// rather than by converting the range back (which would be lossy for
/// `i64` beyond 2^53 and could prune a segment that should have won).
#[derive(Debug, Clone, Copy)]
struct SegmentSortRange {
    min: f64,
    max: f64,
    /// Whether every document in the segment contributes exactly one
    /// point. When false, some document either lacks the field — and
    /// therefore sorts as `Null`, i.e. greatest — or is multi-valued, so
    /// the range cannot bound the segment's greatest key.
    covers_every_doc: bool,
}

/// Convert a sort key to the `f64` space the BKD points were written in.
///
/// Mirrors the writer's per-type conversions exactly (`Int64 as f64`,
/// `Float64` verbatim, `DateTime` as epoch seconds), so a comparison
/// against [`SegmentSortRange`] is consistent. Types that index no point
/// (text, bool, geo) return `None` and are never pruned.
///
/// # Arguments
///
/// * `value` - The sort key to convert.
///
/// # Returns
///
/// The point-space value, or `None` for types without 1-D points.
fn sort_key_as_point(value: &crate::lexical::core::field::FieldValue) -> Option<f64> {
    use crate::lexical::core::field::FieldValue;

    match value {
        FieldValue::Int64(v) => Some(*v as f64),
        FieldValue::Float64(v) => Some(*v),
        // Same encoding the BKD point uses (#1179), so the pruning floor
        // compares like-for-like with the indexed values.
        FieldValue::DateTime(dt) => Some(crate::lexical::core::datetime::datetime_to_point(dt)),
        _ => None,
    }
}

/// Pruning floor established by a full lead top-K (#944), read off the
/// value the per-segment collector ranked its K-th hit by (#1127) — no
/// DocValues re-read. `None` below `limit` hits or when the K-th key has
/// no point-space image.
fn lead_floor(hits: &[FieldHit], limit: usize) -> Option<f64> {
    if hits.len() < limit {
        return None;
    }
    hits.last()
        .and_then(|worst| sort_key_as_point(&worst.value))
}

/// Whether a segment could still contribute a document that outranks the
/// current K-th best (#944).
///
/// Returns `true` whenever pruning is not provably safe — an unknown
/// range, an unconvertible floor, or a segment whose range does not cover
/// every document all keep the segment in full collection.
///
/// The comparison is strict: a segment whose best value merely *ties* the
/// floor is kept, because the doc-id tie-break in
/// [`compare_sort_key`](crate::lexical::query::collector) could still
/// rank it ahead. Values that collide under `f64` rounding likewise
/// compare equal and are kept.
///
/// # Arguments
///
/// * `range` - The segment's range, or `None` when unknown.
/// * `floor` - The current K-th best key in point space, or `None` while
///   fewer than K hits have been collected.
/// * `ascending` - Sort direction.
///
/// # Returns
///
/// `false` only when the segment provably cannot beat `floor`.
fn segment_can_contribute(
    range: Option<SegmentSortRange>,
    floor: Option<f64>,
    ascending: bool,
) -> bool {
    let (Some(range), Some(floor)) = (range, floor) else {
        return true;
    };

    if ascending {
        // Smallest first, so the segment's best key is its minimum.
        // Documents missing the field sort as `Null` (greatest) and are
        // therefore the worst possible, so the minimum bounds the
        // segment even when it does not cover every document.
        range.min <= floor
    } else {
        // Largest first. A segment that does not cover every document
        // may hold a `Null`, which sorts greatest and would outrank
        // anything, so its maximum cannot bound it.
        !range.covers_every_doc || range.max >= floor
    }
}

/// Index of the segment whose best possible sort key is the strongest,
/// i.e. the one most likely to fill the top-K on its own (#944).
///
/// Only segments with a known range are eligible, so the lead segment
/// always yields a usable floor when it fills K. Returns `None` when no
/// segment has a range, in which case nothing can be pruned anyway.
///
/// # Arguments
///
/// * `ranges` - Per-segment ranges, positionally aligned with the
///   segment list.
/// * `ascending` - Sort direction.
///
/// # Returns
///
/// The index of the most promising segment, or `None`.
fn best_segment_index(ranges: &[Option<SegmentSortRange>], ascending: bool) -> Option<usize> {
    ranges
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.map(|r| (i, r)))
        .reduce(|best, cur| {
            let better = if ascending {
                cur.1.min < best.1.min
            } else {
                cur.1.max > best.1.max
            };
            if better { cur } else { best }
        })
        .map(|(i, _)| i)
}

/// Whether any segment could be pruned at all, given the tightest floor
/// the lead segment can possibly produce (#944).
///
/// The floor is the lead segment's K-th best key, which can never be
/// stronger than the lead segment's own extreme value. Testing every
/// other segment against that bound therefore answers "could pruning
/// ever fire here?" before any search runs. The answer is optimistic by
/// construction — it assumes the tightest floor the lead could possibly
/// produce — so a `true` here permits the split rather than promising a
/// prune.
///
/// When it cannot — the usual case for a sort field uncorrelated with
/// segment boundaries, where every segment spans the same range — the
/// caller skips the two-wave split and collects every segment in
/// parallel, exactly as it did before this optimization. Uncorrelated
/// data therefore pays nothing for the serialized lead wave.
///
/// # Arguments
///
/// * `ranges` - Per-segment ranges, positionally aligned with the
///   segment list.
/// * `lead` - Index of the lead segment, from [`best_segment_index`].
/// * `ascending` - Sort direction.
///
/// # Returns
///
/// `true` when at least one other segment could be pruned.
fn pruning_is_possible(ranges: &[Option<SegmentSortRange>], lead: usize, ascending: bool) -> bool {
    let Some(lead_range) = ranges.get(lead).copied().flatten() else {
        return false;
    };
    let tightest_floor = if ascending {
        lead_range.min
    } else {
        lead_range.max
    };
    ranges.iter().enumerate().any(|(i, range)| {
        i != lead && !segment_can_contribute(*range, Some(tightest_floor), ascending)
    })
}

/// Read a segment's sort-field range from its BKD header (#944).
///
/// Returns `None` when the field has no BKD tree in this segment (not
/// indexed, or no document carries it), when the tree is not 1-D, or on
/// any read error — all of which mean "range unknown", so the caller
/// keeps the segment.
///
/// # Arguments
///
/// * `segment` - The segment to inspect.
/// * `field_name` - The sort field.
///
/// # Returns
///
/// The segment's range, or `None` when it cannot be determined.
fn segment_sort_range(
    segment: &std::sync::RwLock<crate::lexical::index::inverted::reader::SegmentReader>,
    field_name: &str,
) -> Option<SegmentSortRange> {
    let seg = segment.read().ok()?;
    let tree = seg.get_bkd_tree(field_name).ok()??;
    let (min, max, point_count) = tree.value_range(0)?;
    Some(SegmentSortRange {
        min,
        max,
        // One point per document exactly: no document is missing the
        // field (which would sort as `Null`) and none is multi-valued.
        covers_every_doc: point_count == seg.segment_info().doc_count,
    })
}

/// Count a query's matches without scoring or reading field values.
///
/// Used for segments that field-sorted search has proven cannot reach
/// the current top-K: their matches still have to be counted so
/// `total_hits` stays the true match count, but none of the per-document
/// work a full collection performs (term frequency, field length, BM25,
/// DocValues, heap) is needed (#944).
///
/// Deleted documents are already filtered at posting-decode level, so
/// the walk counts live matches only.
///
/// # Arguments
///
/// * `query` - The query to count matches for.
/// * `reader` - The reader (a per-segment view) to count against.
///
/// # Returns
///
/// The number of matching documents.
fn count_matches_only(query: &dyn Query, reader: &dyn LexicalIndexReader) -> Result<u64> {
    let mut matcher = query.matcher(reader)?;
    let mut count = 0u64;
    while !matcher.is_exhausted() {
        if matcher.doc_id() == u64::MAX {
            break;
        }
        count += 1;
        if !matcher.next()? {
            break;
        }
    }
    Ok(count)
}

/// A wall-clock deadline for cooperative search interruption (Issue #600).
///
/// Threaded through every scan loop (the default matcher loop, the Block-Max
/// WAND executor, and the per-segment fanout) so a timed search aborts
/// mid-flight instead of only being detected after it has already run to
/// completion. On `wasm32` `Timer` reports zero elapsed, so only a zero budget
/// fires there.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Deadline {
    start: Timer,
    timeout: Duration,
}

impl Deadline {
    /// Create a deadline of `timeout` measured from `start`.
    pub(crate) fn new(start: Timer, timeout: Duration) -> Self {
        Self { start, timeout }
    }

    /// Return `Err` if the time budget is exhausted. Safe to call on every
    /// loop iteration: the clock is only read every [`DEADLINE_CHECK_INTERVAL`]
    /// scanned documents, so `scanned` is the caller's running document count.
    pub(crate) fn check(&self, scanned: u64) -> Result<()> {
        if scanned.is_multiple_of(DEADLINE_CHECK_INTERVAL) {
            self.check_now()?;
        }
        Ok(())
    }

    /// Return `Err` if the time budget is exhausted, reading the clock
    /// unconditionally. For checks outside a scan loop, where there is no
    /// per-document cost to throttle.
    pub(crate) fn check_now(&self) -> Result<()> {
        if budget_spent(self.start.elapsed(), self.timeout) {
            return Err(LaurusError::index("Search timeout exceeded"));
        }
        Ok(())
    }
}

/// Whether `elapsed` has used up a `timeout` budget.
///
/// Inclusive (#1227): with a strict `>`, a zero budget would fire only if the
/// clock had advanced since the deadline started, which Windows' coarse clock
/// and wasm32's frozen `Timer` do not guarantee.
fn budget_spent(elapsed: Duration, timeout: Duration) -> bool {
    elapsed >= timeout
}

/// A searcher that executes queries against an index reader.
#[derive(Debug)]
pub struct InvertedIndexSearcher {
    /// The index reader to search against.
    reader: Arc<dyn LexicalIndexReader>,
    /// Default fields to search if none specified in query.
    default_fields: Vec<String>,
    /// Snapshot-scoped parsed-DSL query cache (Issue #590). The analyzer and
    /// `default_fields` are fixed for this searcher's lifetime, so a DSL string
    /// alone keys it; rebuilt (empty) whenever the store rebuilds the searcher.
    parsed_query_cache: ParsedQueryCache,
}

impl InvertedIndexSearcher {
    /// Create a new searcher with the given index reader.
    pub fn new(reader: Box<dyn LexicalIndexReader>) -> Self {
        InvertedIndexSearcher {
            reader: Arc::from(reader),
            default_fields: Vec::new(),
            parsed_query_cache: ParsedQueryCache::new(DEFAULT_PARSED_QUERY_CACHE_CAPACITY),
        }
    }

    /// Create a new searcher with an `Arc<dyn LexicalIndexReader>`.
    pub fn from_arc(reader: Arc<dyn LexicalIndexReader>) -> Self {
        InvertedIndexSearcher {
            reader,
            default_fields: Vec::new(),
            parsed_query_cache: ParsedQueryCache::new(DEFAULT_PARSED_QUERY_CACHE_CAPACITY),
        }
    }

    /// Set default fields for search.
    pub fn with_default_fields(mut self, fields: Vec<String>) -> Self {
        self.default_fields = fields;
        self
    }

    /// Set the capacity (entries) of the parsed-DSL query cache (Issue #590).
    /// `0` disables the cache. Replaces the default-capacity cache.
    pub fn with_parsed_query_cache_capacity(mut self, capacity: usize) -> Self {
        self.parsed_query_cache = ParsedQueryCache::new(capacity);
        self
    }

    /// Get the index reader.
    pub fn reader(&self) -> &Arc<dyn LexicalIndexReader> {
        &self.reader
    }

    /// Snapshot of the parsed-DSL query cache hit / miss counters (Issue #590).
    pub fn parsed_query_cache_stats(
        &self,
    ) -> crate::lexical::index::inverted::parsed_query_cache::ParsedQueryCacheStats {
        self.parsed_query_cache.stats()
    }

    /// Execute a search with a custom collector.
    pub fn search_with_collector<C: Collector>(
        &self,
        query: Box<dyn Query>,
        collector: C,
    ) -> Result<C> {
        self.search_with_collector_parallel(query, collector, false)
    }

    /// Execute a search with a custom collector, with optional parallel execution.
    pub fn search_with_collector_parallel<C: Collector>(
        &self,
        query: Box<dyn Query>,
        collector: C,
        parallel: bool,
    ) -> Result<C> {
        self.search_with_collector_deadline(query, collector, parallel, None)
    }

    /// Internal search entry point that additionally honours an optional
    /// wall-clock [`Deadline`] (Issue #600). The public
    /// [`Self::search_with_collector_parallel`] delegates here with
    /// `deadline = None`, so non-timed searches pay nothing; the timeout path
    /// passes `Some(..)` and every scan loop below (plus the per-segment
    /// fanout it recurses into) aborts mid-flight once the budget is spent.
    fn search_with_collector_deadline<C: Collector>(
        &self,
        query: Box<dyn Query>,
        mut collector: C,
        parallel: bool,
        deadline: Option<Deadline>,
    ) -> Result<C> {
        // Query rewrite (Issue #613): lower multi-term queries (prefix /
        // wildcard / fuzzy / regexp) into Boolean-of-TermQuery ONCE,
        // against the top-level reader, before any gating below. This
        // (a) halves the term-dictionary enumerations (matcher and
        // scorer used to re-enumerate independently), and (b) hands the
        // per-segment fanout a query it can execute — the fanout's
        // `PerSegmentReaderView` cannot enumerate the term dictionary,
        // which previously made raw multi-term queries return 0 hits on
        // multi-segment indexes. `None` (nothing to rewrite, or the
        // reader is itself a per-segment view) keeps the original query;
        // an already-lowered query rewrites to `None`, so the fanout's
        // recursion back into this method is a cheap no-op.
        let query = match query.rewrite(self.reader.as_ref())? {
            Some(rewritten) => rewritten,
            None => query,
        };

        // For BooleanQuery with multiple clauses, try to execute sub-queries in parallel
        if parallel && let Some(boolean_query) = query.as_any().downcast_ref::<BooleanQuery>() {
            return self.search_boolean_query_parallel(boolean_query, collector, deadline);
        }

        // Per-segment fanout fast path (#476 Phase 1). For multi-
        // segment top-K queries, each segment's `block_max` table is
        // valid as a per-segment scoring bound; running the query
        // independently on each segment via [`PerSegmentReaderView`]
        // re-activates PR-F's BMW pivot loop on each one. Cross-
        // segment merge collects the per-segment top-K into the
        // caller's collector.
        //
        // This gate is a performance choice, not a soundness one
        // (#1120): a collector that falls through to the cross-segment
        // matcher-driven path below still scores correctly, because
        // `InvertedIndexReader::term_info`'s bound is only ever tight,
        // never unsound, regardless of whether it went through this
        // fanout.
        if collector.bmw_capable()
            && let Some(inverted_reader) =
                self.reader.as_any().downcast_ref::<InvertedIndexReader>()
            && inverted_reader.segment_count() >= 2
        {
            return self.search_per_segment_fanout(query, collector, deadline);
        }

        // Block-Max-WAND fast path (#475 PR-F). Eligible for Should-only
        // BooleanQuery against a top-K-style collector. Construction
        // re-checks each clause's per-block metadata at runtime; on
        // any miss we fall through to the existing matcher-driven loop.
        if collector.bmw_capable()
            && let Some(boolean_query) = is_bmw_eligible(query.as_ref())
            && let Ok(executor) = BlockMaxOrExecutor::new(boolean_query, self.reader.as_ref())
        {
            return executor.run(collector, deadline);
        }

        // Default single-threaded execution
        // Create the matcher and scorer in one pass (#996: queries with
        // an expensive shared candidate computation, e.g. geo, build
        // both from a single run of it).
        let (mut matcher, scorer) = query.matcher_scorer(self.reader.as_ref())?;

        // SIMD-batched default loop (#506). The scalar path collected
        // one doc at a time via `scorer.score`; this version gathers up
        // to `BATCH_SIZE` per-doc inputs (doc id / TF / field length)
        // and lowers the cross-doc kernel through
        // [`crate::lexical::query::scorer::Scorer::batch_score`], whose
        // BM25 override is an `f32x8` SIMD kernel. Non-BM25 scorers
        // inherit the trait's per-element default, so behaviour is
        // identical there.
        //
        // Trade-off: the cumulative early-break (#403 PR-C) and the
        // count-cap `needs_more()` check both consume the latest
        // `min_competitive()`, so batching delays them by up to
        // `BATCH_SIZE - 1` docs. The buffer flushes also fire before
        // any per-block skip so the skip target stays accurate.
        const BATCH_SIZE: usize = 8;
        let mut doc_buf: [u64; BATCH_SIZE] = [0; BATCH_SIZE];
        let mut tf_buf: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
        let mut fl_buf: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
        let mut score_buf: [f32; BATCH_SIZE] = [0.0; BATCH_SIZE];
        let mut n: usize = 0;
        let avg_fl = scorer.avg_field_length();
        let query_field = query.field().map(|s| s.to_string());

        // Running count of scanned documents, used to throttle the deadline
        // clock read (Issue #600). Starts at 0 so the first iteration checks
        // immediately, which makes a zero/expired budget fail fast.
        let mut scanned: u64 = 0;

        // Iterate through matching documents
        while !matcher.is_exhausted() {
            if let Some(d) = deadline {
                d.check(scanned)?;
            }
            scanned = scanned.wrapping_add(1);

            let doc_id = matcher.doc_id();

            if doc_id == u64::MAX {
                break;
            }

            // Block-Max skip-ahead pre-check (#403 PR-E). Before paying
            // the score / field-length cost on this doc, see whether
            // the block containing it is even competitive. The current
            // block's bound (`current_block_max_score`) is non-cumulative
            // — when it falls below the K-th score, jumping past the
            // block via `next_block_boundary` is sound (the global
            // `block_max_score_at` cumulative bound, queried right
            // below, still controls the hard `break`).
            let min_comp = collector.min_competitive();
            if scorer.current_block_max_score(doc_id) <= min_comp {
                // Flush the buffered batch before deciding the skip
                // target. The skip relies on `block_max_score_at`,
                // which factors in the K-th score; that score can only
                // be tight once buffered hits have been collected.
                if n > 0 {
                    scorer.batch_score(
                        &doc_buf[..n],
                        &tf_buf[..n],
                        &fl_buf[..n],
                        &mut score_buf[..n],
                    );
                    for i in 0..n {
                        collector.collect(doc_buf[i], score_buf[i])?;
                        if !collector.needs_more() {
                            return Ok(collector);
                        }
                    }
                    n = 0;
                }
                let min_comp = collector.min_competitive();
                if scorer.block_max_score_at(doc_id) <= min_comp {
                    // Cumulative suffix bound already non-competitive
                    // → no later block can produce a top-K hit.
                    break;
                }
                if let Some(target) = scorer.next_block_boundary(doc_id) {
                    if target == u64::MAX || target <= doc_id {
                        break;
                    }
                    if !matcher.skip_to(target)? || matcher.is_exhausted() {
                        break;
                    }
                    continue;
                }
                // No per-block info → fall through to existing PR-C
                // break path after scoring this doc.
            }

            // Gather per-doc inputs into the batch buffer. The field
            // length lookup mirrors the scalar path's reader downcasts
            // (`InvertedIndexReader` / `PerSegmentReaderView`), but
            // substitutes the scorer's avg when no per-doc value is
            // available so the dense SIMD slice stays valid.
            let term_freq = matcher.term_freq() as f32;
            let field_length = if let Some(field_name) = query_field.as_deref() {
                if let Some(inverted_index_reader) =
                    self.reader.as_any().downcast_ref::<InvertedIndexReader>()
                {
                    inverted_index_reader
                        .field_length(doc_id, field_name)
                        .ok()
                        .flatten()
                        .map(|len| len as f32)
                        .unwrap_or(avg_fl)
                } else if let Some(view) =
                    self.reader.as_any().downcast_ref::<PerSegmentReaderView>()
                {
                    // #476 Phase 1: per-segment fanout reads field
                    // lengths through the view so BM25 normalisation
                    // matches each segment's local avg.
                    view.field_length(doc_id, field_name)
                        .ok()
                        .flatten()
                        .map(|len| len as f32)
                        .unwrap_or(avg_fl)
                } else {
                    avg_fl
                }
            } else {
                avg_fl
            };

            doc_buf[n] = doc_id;
            tf_buf[n] = term_freq;
            fl_buf[n] = field_length;
            n += 1;

            if n == BATCH_SIZE {
                scorer.batch_score(
                    &doc_buf[..n],
                    &tf_buf[..n],
                    &fl_buf[..n],
                    &mut score_buf[..n],
                );
                let last_doc = doc_buf[n - 1];
                for i in 0..n {
                    collector.collect(doc_buf[i], score_buf[i])?;
                    if !collector.needs_more() {
                        return Ok(collector);
                    }
                }
                n = 0;

                // Cumulative early-break (#403 PR-C) once per batch.
                // The K-th score is at its tightest right after the
                // batch is collected; if the right-cumulative suffix
                // bound has already fallen below it, no later doc can
                // enter the top-K.
                if scorer.block_max_score_at(last_doc) <= collector.min_competitive() {
                    return Ok(collector);
                }
            }

            // Move to next document
            if !matcher.next()? {
                break;
            }
        }

        // Final flush for any partial batch left when the matcher is
        // exhausted (or a `break` above was taken without flushing).
        if n > 0 {
            scorer.batch_score(
                &doc_buf[..n],
                &tf_buf[..n],
                &fl_buf[..n],
                &mut score_buf[..n],
            );
            for i in 0..n {
                collector.collect(doc_buf[i], score_buf[i])?;
                if !collector.needs_more() {
                    return Ok(collector);
                }
            }
        }

        Ok(collector)
    }

    /// Execute a top-K query against a multi-segment reader by
    /// fanning out to per-segment searches (#476 Phase 1). Each
    /// segment runs the query through a [`PerSegmentReaderView`],
    /// which lets PR-F's BMW pivot loop fire on the segment's local
    /// `block_max` table. Results are merged into the caller's
    /// collector.
    fn search_per_segment_fanout<C: Collector>(
        &self,
        query: Box<dyn Query>,
        mut collector: C,
        deadline: Option<Deadline>,
    ) -> Result<C> {
        // Downcast ensured by the caller, but re-resolve here to
        // borrow the segment list.
        let inverted_reader = self
            .reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .expect("search_per_segment_fanout requires InvertedIndexReader");

        let global_doc_count = inverted_reader.doc_count();
        let global_max_doc = inverted_reader.max_doc();
        // Build a global term-info closure that captures an Arc
        // pointing back at the cross-segment reader so each
        // PerSegmentReaderView can resolve IDF lookups.
        let global_term_info_fn = {
            let reader_arc = self.reader.clone();
            std::sync::Arc::new(
                move |field: &str,
                      term: &str|
                      -> Result<Option<crate::lexical::reader::ReaderTermInfo>> {
                    reader_arc.term_info(field, term)
                },
            )
        };

        // Build a cross-segment matching-doc-ids closure (#764) so each
        // PerSegmentReaderView can resolve a cacheable filter clause against the
        // cross-segment snapshot cache rather than re-walking postings per
        // segment. The fanout is only entered when `self.reader` is an
        // InvertedIndexReader (dispatch gate), so the downcast succeeds; the
        // defensive branch drains the matcher uncached.
        let global_matching_doc_ids_fn = {
            let reader_arc = self.reader.clone();
            std::sync::Arc::new(
                move |query: &dyn Query| -> Result<Arc<roaring::RoaringTreemap>> {
                    if let Some(inverted) =
                        reader_arc.as_any().downcast_ref::<InvertedIndexReader>()
                    {
                        inverted.matching_doc_ids(query)
                    } else {
                        let matcher = query.matcher(reader_arc.as_ref())?;
                        Ok(Arc::new(
                            crate::lexical::index::inverted::query_cache::drain_matcher(matcher)?,
                        ))
                    }
                },
            )
        };

        // Per-segment K. The collector wants `top_k` hits globally;
        // each segment returns up to `top_k` so the merge has the
        // headroom to pick any combination of per-segment hits.
        let per_segment_k = collector.requested_top_k().unwrap_or(10);

        let segments = inverted_reader.segment_readers().to_vec();

        #[cfg(not(target_arch = "wasm32"))]
        let segment_iter = segments.par_iter();
        #[cfg(target_arch = "wasm32")]
        let segment_iter = segments.iter();

        let per_segment_results: Vec<Result<Vec<SearchHit>>> = segment_iter
            .map(|seg_arc| -> Result<Vec<SearchHit>> {
                let view = PerSegmentReaderView::new(
                    seg_arc.clone(),
                    global_doc_count,
                    global_max_doc,
                    global_term_info_fn.clone(),
                    global_matching_doc_ids_fn.clone(),
                );
                let view_reader: Arc<dyn LexicalIndexReader> = Arc::new(view);
                let temp_searcher = InvertedIndexSearcher::from_arc(view_reader);
                let temp_collector = TopDocsCollector::new(per_segment_k);
                // Propagate the deadline so each per-segment search aborts
                // mid-flight too — segments run in parallel, so a single slow
                // segment would otherwise leave the whole fanout unbounded
                // (Issue #600).
                let collected = temp_searcher.search_with_collector_deadline(
                    query.clone_box(),
                    temp_collector,
                    false,
                    deadline,
                )?;
                Ok(collected.results())
            })
            .collect();

        // Merge per-segment top-K into the caller's collector. Errors
        // from any one segment short-circuit the whole search.
        for hits in per_segment_results {
            let hits = hits?;
            for hit in hits {
                collector.collect(hit.doc_id, hit.score)?;
                if !collector.needs_more() {
                    return Ok(collector);
                }
            }
        }
        Ok(collector)
    }

    /// Field-sorted per-segment fanout (#944 Phase A).
    ///
    /// Runs the query independently on every segment with a per-segment
    /// [`TopFieldCollector`] (in parallel off-wasm), then merges the
    /// per-segment top-K through a reader-free `FieldTopK` on the sort
    /// values each collector already ranked by (#1127), so the merge
    /// performs no DocValues or stored-document reads. Ordering —
    /// including the doc-id tie-break — matches the single-pass path
    /// exactly because both go through the same comparator.
    ///
    /// `total_hits` sums the per-segment totals: segments partition the
    /// live documents (tombstoned copies are filtered at posting-decode
    /// level), so the documented true-match-count contract for
    /// field-sorted searches is preserved.
    ///
    /// The generic [`Self::search_per_segment_fanout`] is deliberately
    /// not reused: it is gated on `bmw_capable()` and hard-codes a
    /// per-segment [`TopDocsCollector`], both of which are
    /// score-competitiveness concepts that do not apply to field sort.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute on each segment.
    /// * `field_name` - The sort field.
    /// * `ascending` - Sort direction.
    /// * `limit` - Global top-K (also the per-segment K, so the merge
    ///   can pick any combination of per-segment hits).
    /// * `min_score` - Minimum score threshold, applied per segment.
    /// * `deadline` - Optional cooperative deadline, propagated to each
    ///   per-segment search.
    ///
    /// # Returns
    ///
    /// The merged, field-ordered hits and the true total match count.
    fn search_field_sorted_fanout(
        &self,
        query: Box<dyn Query>,
        field_name: &str,
        ascending: bool,
        limit: usize,
        min_score: f32,
        deadline: Option<Deadline>,
    ) -> Result<(Vec<SearchHit>, u64)> {
        let inverted_reader = self
            .reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .expect("search_field_sorted_fanout requires InvertedIndexReader");

        let global_doc_count = inverted_reader.doc_count();
        let global_max_doc = inverted_reader.max_doc();
        let global_term_info_fn = {
            let reader_arc = self.reader.clone();
            std::sync::Arc::new(
                move |field: &str,
                      term: &str|
                      -> Result<Option<crate::lexical::reader::ReaderTermInfo>> {
                    reader_arc.term_info(field, term)
                },
            )
        };
        let global_matching_doc_ids_fn = {
            let reader_arc = self.reader.clone();
            std::sync::Arc::new(
                move |query: &dyn Query| -> Result<Arc<roaring::RoaringTreemap>> {
                    if let Some(inverted) =
                        reader_arc.as_any().downcast_ref::<InvertedIndexReader>()
                    {
                        inverted.matching_doc_ids(query)
                    } else {
                        let matcher = query.matcher(reader_arc.as_ref())?;
                        Ok(Arc::new(
                            crate::lexical::index::inverted::query_cache::drain_matcher(matcher)?,
                        ))
                    }
                },
            )
        };

        let segments = inverted_reader.segment_readers().to_vec();

        // Pruning needs an exact match count from the pruned segments to
        // keep `total_hits` true, and a matcher-only walk can only
        // deliver that when no score threshold applies.
        let eligible = min_score <= 0.0 && segments.len() > 1;

        // Per-segment sort-field ranges, read from the BKD headers in
        // O(1) each (#944). A segment whose best possible key cannot
        // beat the running K-th best only needs its matches counted.
        let ranges: Vec<Option<SegmentSortRange>> = if eligible {
            segments
                .iter()
                .map(|seg| segment_sort_range(seg, field_name))
                .collect()
        } else {
            Vec::new()
        };

        // Collect one segment fully first so later ones have a floor to
        // be pruned against — the most promising one, so the floor is as
        // tight as possible. With time-correlated data (segment-per-
        // commit over a timestamp or monotonic id) this segment holds
        // the whole answer and every other one is pruned.
        //
        // Splitting the fan-out into two waves serializes the lead
        // segment, so it is only worth doing when the ranges actually
        // permit pruning. Otherwise every segment stays in the single
        // parallel wave below, which is exactly the unpruned path.
        let lead = if eligible {
            best_segment_index(&ranges, ascending)
                .filter(|&idx| pruning_is_possible(&ranges, idx, ascending))
        } else {
            None
        };
        let prunable = lead.is_some();

        let collect_segment = |seg_arc: &Arc<
            std::sync::RwLock<crate::lexical::index::inverted::reader::SegmentReader>,
        >|
         -> Result<(Vec<FieldHit>, u64)> {
            let view = PerSegmentReaderView::new(
                seg_arc.clone(),
                global_doc_count,
                global_max_doc,
                global_term_info_fn.clone(),
                global_matching_doc_ids_fn.clone(),
            );
            let view_reader: Arc<dyn LexicalIndexReader> = Arc::new(view);
            let temp_searcher = InvertedIndexSearcher::from_arc(view_reader);
            let temp_collector = TopFieldCollector::with_min_score(
                limit,
                min_score,
                field_name.to_string(),
                ascending,
                temp_searcher.reader.as_ref(),
            );
            let collected = temp_searcher.search_with_collector_deadline(
                query.clone_box(),
                temp_collector,
                false,
                deadline,
            )?;
            let total_hits = collected.total_hits();
            Ok((collected.into_field_hits(), total_hits))
        };

        // Wave 1: the lead segment, establishing the floor.
        let mut lead_result: Option<Result<(Vec<FieldHit>, u64)>> = None;
        let mut floor: Option<f64> = None;
        if let Some(idx) = lead {
            let result = collect_segment(&segments[idx]);
            if let Ok((hits, _)) = &result {
                // The K-th best of a full per-segment top-K: nothing
                // ranked below it can enter the global top-K either.
                floor = lead_floor(hits, limit);
            }
            lead_result = Some(result);
        }

        // Wave 2: everything else, in parallel, pruned against the floor.
        let rest: Vec<(usize, &Arc<std::sync::RwLock<_>>)> = segments
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != lead)
            .collect();

        #[cfg(not(target_arch = "wasm32"))]
        let rest_iter = rest.par_iter();
        #[cfg(target_arch = "wasm32")]
        let rest_iter = rest.iter();

        let mut per_segment_results: Vec<Result<(Vec<FieldHit>, u64)>> = rest_iter
            .map(|(i, seg_arc)| -> Result<(Vec<FieldHit>, u64)> {
                if prunable
                    && !segment_can_contribute(ranges.get(*i).copied().flatten(), floor, ascending)
                {
                    // Cannot reach the top-K: count the matches so
                    // `total_hits` stays exact, and skip the rest.
                    let view = PerSegmentReaderView::new(
                        (*seg_arc).clone(),
                        global_doc_count,
                        global_max_doc,
                        global_term_info_fn.clone(),
                        global_matching_doc_ids_fn.clone(),
                    );
                    let count = count_matches_only(query.as_ref(), &view)?;
                    return Ok((Vec::new(), count));
                }
                collect_segment(seg_arc)
            })
            .collect();

        if let Some(result) = lead_result {
            per_segment_results.push(result);
        }

        // Merge on the values the per-segment collectors already ranked
        // by: `FieldTopK` has no reader, so this step performs no
        // DocValues or stored-document reads (#1127). Same comparator +
        // doc-id tie-break as the single-pass path. `min_score` was
        // already applied per segment.
        let mut merged = FieldTopK::new(limit, ascending);
        let mut total_hits = 0u64;
        for seg_result in per_segment_results {
            let (seg_hits, seg_total) = seg_result?;
            total_hits += seg_total;
            for hit in seg_hits {
                merged.push(hit);
            }
        }
        Ok((merged.sorted_search_hits(), total_hits))
    }

    /// Execute a BooleanQuery with parallel sub-query execution.
    ///
    /// Each clause is executed in parallel, then boolean logic is applied:
    /// - Must/Filter: intersection (all must match)
    /// - Should: union (adds score if matching; at least minimum_should_match required)
    /// - MustNot: exclusion (removes matching documents)
    fn search_boolean_query_parallel<C: Collector>(
        &self,
        boolean_query: &BooleanQuery,
        mut collector: C,
        deadline: Option<Deadline>,
    ) -> Result<C> {
        let clauses = boolean_query.clauses();

        if clauses.is_empty() {
            return Ok(collector);
        }

        // MustNot-only booleans have no positive clause to parallelize:
        // the per-clause merge below would invert a single negation (the
        // single-clause shortcut runs the negated query as-is) or return
        // nothing (the survivor set starts empty without Must/Should
        // hits). Route them through the serial matcher path, whose
        // universe is the present-live doc set (#997).
        if clauses.iter().all(|clause| clause.occur == Occur::MustNot) {
            return self.search_with_collector_deadline(
                boolean_query.clone_box(),
                collector,
                false,
                deadline,
            );
        }

        // Single clause: no need for parallel execution
        if clauses.len() == 1 {
            return self.search_with_collector_deadline(
                clauses[0].query.clone_box(),
                collector,
                false,
                deadline,
            );
        }

        // Execute all clauses in parallel, collecting (doc_id, score) per clause
        #[cfg(not(target_arch = "wasm32"))]
        let iter = clauses.par_iter();
        #[cfg(target_arch = "wasm32")]
        let iter = clauses.iter();

        let clause_results: Vec<(Occur, Result<Vec<SearchHit>>)> = iter
            .map(|clause| {
                // Boolean operations (intersection/union/exclusion) require the
                // full result set from each clause, so we use an unbounded collector.
                let temp_collector = TopDocsCollector::new(usize::MAX);
                let result = self
                    .search_with_collector_deadline(
                        clause.query.clone_box(),
                        temp_collector,
                        false,
                        deadline,
                    )
                    .map(|c| c.results());
                (clause.occur, result)
            })
            .collect();

        // Fold each clause's hits into Roaring bitmaps for the set logic (#587)
        // and a flat `(doc_id, score)` list for Must/Should score accumulation.
        // The set operations (AND / ANDNOT / OR) run as Roaring word-walks
        // instead of `HashMap::retain` / `HashSet::remove`, and the final
        // selection is delegated to the collector's bounded top-K heap rather
        // than a full sort over the whole candidate set.
        //
        // Filter clauses contribute to membership only (score 0), so their docs
        // go into `must_bitmaps` but never into `scored_hits`.
        let mut must_bitmaps: Vec<RoaringTreemap> = Vec::new();
        let mut should_bitmap = RoaringTreemap::new();
        let mut must_not_bitmap = RoaringTreemap::new();
        let mut scored_hits: Vec<(u64, f32)> = Vec::new();
        let mut first_error: Option<LaurusError> = None;

        for (occur, result) in clause_results {
            match result {
                Ok(hits) => match occur {
                    Occur::Must => {
                        let mut bitmap = RoaringTreemap::new();
                        for hit in hits {
                            bitmap.insert(hit.doc_id);
                            scored_hits.push((hit.doc_id, hit.score));
                        }
                        must_bitmaps.push(bitmap);
                    }
                    Occur::Filter => {
                        // Membership only — Filter does not contribute to score.
                        let mut bitmap = RoaringTreemap::new();
                        for hit in hits {
                            bitmap.insert(hit.doc_id);
                        }
                        must_bitmaps.push(bitmap);
                    }
                    Occur::Should => {
                        for hit in hits {
                            should_bitmap.insert(hit.doc_id);
                            scored_hits.push((hit.doc_id, hit.score));
                        }
                    }
                    Occur::MustNot => {
                        for hit in hits {
                            must_not_bitmap.insert(hit.doc_id);
                        }
                    }
                },
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }

        // If any clause produced an error, fail the whole query
        if let Some(e) = first_error {
            return Err(e);
        }

        let minimum_should_match = boolean_query.minimum_should_match();
        let has_must = !must_bitmaps.is_empty();

        // Build the survivor membership set via Roaring set operations.
        let mut survivor = if has_must {
            // Intersect smallest-first so the running result shrinks fastest.
            must_bitmaps.sort_unstable_by_key(|b| b.len());
            let mut bitmaps = must_bitmaps.into_iter();
            let mut acc = bitmaps.next().unwrap_or_default();
            for bitmap in bitmaps {
                acc &= &bitmap;
            }
            acc
        } else {
            // No Must/Filter clauses: the Should union is the candidate set.
            should_bitmap.clone()
        };

        // With minimum_should_match > 0 a Must candidate must also appear in at
        // least one Should clause (preserves the existing parallel semantics).
        if has_must && minimum_should_match > 0 {
            survivor &= &should_bitmap;
        }

        // Exclude MustNot documents.
        if !must_not_bitmap.is_empty() {
            survivor -= &must_not_bitmap;
        }

        // Accumulate scores for survivors only (one pass over Must/Should hits),
        // then feed the collector. The collector keeps the top-K via a bounded
        // min-heap, so no full sort over the candidate set is needed.
        let mut score_acc: AHashMap<u64, f32> = AHashMap::with_capacity(survivor.len() as usize);
        for (doc_id, score) in scored_hits {
            if survivor.contains(doc_id) {
                *score_acc.entry(doc_id).or_insert(0.0) += score;
            }
        }

        for doc_id in survivor.iter() {
            let score = score_acc.get(&doc_id).copied().unwrap_or(0.0);
            collector.collect(doc_id, score)?;
            if !collector.needs_more() {
                break;
            }
        }

        Ok(collector)
    }

    /// Load documents for search hits.
    fn load_documents(&self, hits: &mut [SearchHit]) -> Result<()> {
        for hit in hits {
            if let Some(doc) = self.reader.document(hit.doc_id)? {
                hit.document = Some(doc);
            }
        }
        Ok(())
    }

    /// Load documents in parallel for better performance.
    fn load_documents_parallel(&self, hits: &mut [SearchHit]) -> Result<()> {
        // Use a parallel iterator to load documents
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<_> = hits
            .par_iter()
            .map(|hit| (hit.doc_id, self.reader.document(hit.doc_id)))
            .collect();
        #[cfg(target_arch = "wasm32")]
        let results: Vec<_> = hits
            .iter()
            .map(|hit| (hit.doc_id, self.reader.document(hit.doc_id)))
            .collect();

        // Update hits with loaded documents
        for (i, (_, doc_result)) in results.into_iter().enumerate() {
            if let Ok(Some(doc)) = doc_result {
                hits[i].document = Some(doc);
            }
        }

        Ok(())
    }

    /// Execute a search with timeout (internal implementation).
    fn search_with_timeout_internal(
        &self,
        query: Box<dyn Query>,
        params: &LexicalSearchParams,
        timeout: Duration,
    ) -> Result<LexicalSearchResults> {
        // Cooperative deadline (Issue #600). Threading it through the scan
        // loops lets the search abort mid-flight once the budget is spent,
        // instead of only being detected after the query has already run to
        // completion as the old post-hoc `elapsed()` check did.
        let deadline = Deadline::new(Timer::now(), timeout);

        // Create collector based on sort type
        let (mut hits, total_hits) = match &params.sort_by {
            SortField::Field { name, order } => {
                let ascending = matches!(order, SortOrder::Asc);
                if self
                    .reader
                    .as_any()
                    .downcast_ref::<InvertedIndexReader>()
                    .is_some_and(|r| r.segment_count() >= 2)
                {
                    // Multi-segment: per-segment field-sorted fanout
                    // (#944 Phase A).
                    self.search_field_sorted_fanout(
                        query.clone_box(),
                        name,
                        ascending,
                        params.limit,
                        params.min_score,
                        Some(deadline),
                    )?
                } else {
                    // Use TopFieldCollector for field-based sorting
                    let collector = TopFieldCollector::with_min_score(
                        params.limit,
                        params.min_score,
                        name.clone(),
                        ascending,
                        self.reader.as_ref(),
                    );

                    let result_collector = self.search_with_collector_deadline(
                        query.clone_box(),
                        collector,
                        params.parallel,
                        Some(deadline),
                    )?;

                    (result_collector.results(), result_collector.total_hits())
                }
            }
            SortField::Score => {
                // Use TopDocsCollector for score-based sorting
                let collector = TopDocsCollector::with_min_score(params.limit, params.min_score);

                let result_collector = self.search_with_collector_deadline(
                    query,
                    collector,
                    params.parallel,
                    Some(deadline),
                )?;

                (result_collector.results(), result_collector.total_hits())
            }
        };

        // Final safety net: the scan loops abort mid-flight on the deadline,
        // but a search that finished just over budget (or spent the time
        // outside a scan loop) is still reported as timed out.
        deadline.check_now()?;

        // Load documents if requested
        if params.load_documents {
            if params.parallel && hits.len() > 10 {
                self.load_documents_parallel(&mut hits)?;
            } else {
                self.load_documents(&mut hits)?;
            }
        }

        // No need to sort - already sorted during collection

        // Calculate max score
        let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

        Ok(LexicalSearchResults {
            hits,
            total_hits,
            max_score,
        })
    }

    /// Search with the given request.
    pub fn search(&self, request: LexicalSearchRequest) -> Result<LexicalSearchResults> {
        // Convert DSL query to Query object if necessary
        let query = match &request.query {
            LexicalSearchQuery::Dsl(dsl_string) => {
                // Parsed-query cache (#590): a popular DSL string is parsed once
                // per snapshot and reused via `clone_box` (cheap — refcount
                // bumps for boolean clause subtrees). The analyzer and
                // `default_fields` are fixed for this searcher, so the DSL
                // string alone keys the cache.
                if let Some(cached) = self.parsed_query_cache.get(dsl_string) {
                    cached.clone_box()
                } else {
                    // Get analyzer from reader
                    let analyzer = if let Some(inverted_index_reader) =
                        self.reader.as_any().downcast_ref::<InvertedIndexReader>()
                    {
                        inverted_index_reader.analyzer().clone()
                    } else {
                        // Fallback to standard analyzer
                        Arc::new(StandardAnalyzer::new()?)
                    };

                    // Parse DSL string into Query object
                    let mut parser = LexicalQueryParser::new(analyzer.clone());
                    if !self.default_fields.is_empty() {
                        parser = parser.with_default_fields(self.default_fields.clone());
                    }
                    let parsed: Arc<dyn Query> = Arc::from(parser.parse(dsl_string)?);
                    self.parsed_query_cache
                        .put(dsl_string.clone(), parsed.clone());
                    parsed.clone_box()
                }
            }
            LexicalSearchQuery::Obj(q) => q.clone_box(),
        };

        // Check if query is empty
        if query.is_empty(self.reader.as_ref())? {
            return Ok(LexicalSearchResults {
                hits: Vec::new(),
                total_hits: 0,
                max_score: 0.0,
            });
        }

        // Execute search with timeout if specified
        if let Some(timeout_ms) = request.params.timeout_ms {
            let timeout = Duration::from_millis(timeout_ms);
            self.search_with_timeout_internal(query, &request.params, timeout)
        } else {
            // Check if we should use field-based sorting during collection
            match &request.params.sort_by {
                SortField::Field { name, order } => {
                    let ascending = matches!(order, SortOrder::Asc);
                    let (mut hits, total_hits) = if self
                        .reader
                        .as_any()
                        .downcast_ref::<InvertedIndexReader>()
                        .is_some_and(|r| r.segment_count() >= 2)
                    {
                        // Multi-segment: per-segment field-sorted fanout
                        // (#944 Phase A).
                        self.search_field_sorted_fanout(
                            query.clone_box(),
                            name,
                            ascending,
                            request.params.limit,
                            request.params.min_score,
                            None,
                        )?
                    } else {
                        // Use TopFieldCollector for field-based sorting
                        let collector = TopFieldCollector::with_min_score(
                            request.params.limit,
                            request.params.min_score,
                            name.clone(),
                            ascending,
                            self.reader.as_ref(),
                        );

                        let result_collector = self.search_with_collector_parallel(
                            query.clone_box(),
                            collector,
                            request.params.parallel,
                        )?;

                        (result_collector.results(), result_collector.total_hits())
                    };

                    // Load documents if requested
                    if request.params.load_documents {
                        self.load_documents(&mut hits)?;
                    }

                    // No need to sort - already sorted by TopFieldCollector during collection

                    // Calculate max score
                    let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

                    Ok(LexicalSearchResults {
                        hits,
                        total_hits,
                        max_score,
                    })
                }
                SortField::Score => {
                    // Use TopDocsCollector for score-based sorting
                    let collector = TopDocsCollector::with_min_score(
                        request.params.limit,
                        request.params.min_score,
                    );
                    let result_collector = self.search_with_collector_parallel(
                        query,
                        collector,
                        request.params.parallel,
                    )?;

                    let mut hits = result_collector.results();
                    let total_hits = result_collector.total_hits();

                    // Load documents if requested
                    if request.params.load_documents {
                        self.load_documents(&mut hits)?;
                    }

                    // No need to sort - already sorted by score in TopDocsCollector

                    // Calculate max score
                    let max_score = hits.iter().map(|hit| hit.score).fold(0.0f32, f32::max);

                    Ok(LexicalSearchResults {
                        hits,
                        total_hits,
                        max_score,
                    })
                }
            }
        }
    }

    /// Count documents matching the request.
    ///
    /// If `min_score` is specified in the request parameters, only documents
    /// with a score equal to or greater than the threshold are counted.
    pub fn count(&self, request: LexicalSearchRequest) -> Result<u64> {
        let lexical_query = request.query;

        // Parse DSL string if needed
        let query = if let LexicalSearchQuery::Dsl(_) = &lexical_query {
            // Get analyzer from reader
            let analyzer = if let Some(inverted_index_reader) =
                self.reader.as_any().downcast_ref::<InvertedIndexReader>()
            {
                inverted_index_reader.analyzer().clone()
            } else {
                // Fallback to standard analyzer
                Arc::new(StandardAnalyzer::new()?)
            };

            // Parse DSL string into Query object
            lexical_query.into_query(&analyzer)?
        } else {
            match lexical_query {
                LexicalSearchQuery::Obj(q) => q,
                _ => unreachable!(),
            }
        };

        // Check if query is empty
        if query.is_empty(self.reader.as_ref())? {
            return Ok(0);
        }

        // O(1) fast path (Issue #610): a bare `TermQuery` with no score
        // threshold over a reader with no deletions equals the term's document
        // frequency, which is already stored in the term dictionary — so the
        // full posting-list walk the slow path performs is unnecessary.
        //
        // All three guards are required for correctness; if any fails we fall
        // through to the slow path, so the fast path can never miscount:
        // - `min_score <= 0.0`: with a positive threshold each doc's score must
        //   be computed, so a count cannot come from `doc_freq` alone.
        // - `!has_effective_deletions()`: the term dictionary's `doc_freq`
        //   counts raw postings, including deleted docs, whereas the slow path
        //   filters deletions out, so the two agree only when no document the
        //   index holds is deleted. This used to be `doc_count() == max_doc()`,
        //   which a segment with gaps in its id range satisfied while it had
        //   deletions (Issue #1211); the predicate is now exact, and treats a
        //   segment of unknown membership with any deletion bit as deleted.
        // - the query is exactly a `TermQuery` (not a Boolean/phrase/etc.).
        // - `term_info_is_authoritative()`: a segment without a term
        //   dictionary matches through the stored-document scan but is
        //   absent from `doc_freq` (Issue #1196), so only a fully indexed
        //   reader may answer from the dictionary.
        if request.params.min_score <= 0.0
            && !self.reader.has_effective_deletions()
            && self.reader.term_info_is_authoritative()
            && let Some(term_query) = query.as_any().downcast_ref::<TermQuery>()
        {
            return self
                .reader
                .term_doc_freq(term_query.field(), term_query.term());
        }

        // Use count collector with min_score if specified
        let collector = if request.params.min_score > 0.0 {
            CountCollector::with_min_score(request.params.min_score)
        } else {
            CountCollector::new()
        };

        let result_collector = self.search_with_collector(query, collector)?;
        Ok(result_collector.total_hits())
    }
}

// Implement LexicalSearcher trait for InvertedIndexSearcher
impl crate::lexical::search::searcher::LexicalSearcher for InvertedIndexSearcher {
    fn search(&self, request: LexicalSearchRequest) -> Result<LexicalSearchResults> {
        InvertedIndexSearcher::search(self, request)
    }

    fn count(
        &self,
        request: crate::lexical::search::searcher::LexicalSearchRequest,
    ) -> Result<u64> {
        InvertedIndexSearcher::count(self, request)
    }

    fn matching_doc_ids(&self, query: Box<dyn Query>) -> Result<Arc<roaring::RoaringTreemap>> {
        // The common case: the reader is an `InvertedIndexReader`, which owns
        // the snapshot-scoped query/filter cache (Issue #578) and serves
        // cacheable queries without re-walking posting lists.
        if let Some(inverted_reader) = self.reader.as_any().downcast_ref::<InvertedIndexReader>() {
            return inverted_reader.matching_doc_ids(query.as_ref());
        }
        // Fallback for a non-inverted reader (e.g. a transient
        // `PerSegmentReaderView`): no snapshot cache is available, so drain the
        // matcher directly using the shared helper.
        let matcher = query.matcher(self.reader.as_ref())?;
        let bitmap = crate::lexical::index::inverted::query_cache::drain_matcher(matcher)?;
        Ok(Arc::new(bitmap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::inverted::reader::{InvertedIndexReader, InvertedIndexReaderConfig};
    use crate::lexical::query::boolean::{BooleanQuery, BooleanQueryBuilder};
    use crate::lexical::query::term::TermQuery;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;
    use std::sync::Arc;

    #[allow(dead_code)]
    fn create_test_searcher() -> InvertedIndexSearcher {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader = Box::new(
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap(),
        );
        InvertedIndexSearcher::new(reader)
    }

    #[test]
    fn test_searcher_creation() {
        let searcher = create_test_searcher();

        // Verify searcher has a valid reader
        let reader = searcher.reader();
        assert!(Arc::strong_count(reader) >= 1, "Reader should be valid");

        // Verify reader has expected initial state
        assert_eq!(
            reader.doc_count(),
            0,
            "New searcher should have 0 documents"
        );
    }

    #[test]
    fn test_search_term_query() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for non-existent terms
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_search_boolean_query() {
        let searcher = create_test_searcher();

        let query = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("title", "hello")))
                .should(Box::new(TermQuery::new("body", "world")))
                .build(),
        ) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for non-existent terms
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_search_with_config() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query)
            .limit(5)
            .min_score(0.5)
            .load_documents(false);

        let results = searcher.search(request).unwrap();

        // Should respect configuration
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_count_query() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let count = searcher.count(LexicalSearchRequest::new(query)).unwrap();

        // Should return 0 for non-existent terms
        assert_eq!(count, 0);
    }

    #[test]
    fn test_search_with_timeout() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query).timeout_ms(1000); // 1 second timeout

        let results = searcher.search(request).unwrap();

        // Should complete within timeout
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn deadline_check_semantics() {
        // The deadline primitive (Issue #600): it fires only at check-interval
        // indices, and only when the budget is actually exhausted.
        let now = Timer::now();
        // Index 0 is a multiple of the interval, so an exhausted (zero) budget
        // is detected immediately — a search fails fast. A zero budget is
        // spent even if the clock has not advanced since `now` (#1227).
        assert!(Deadline::new(now, Duration::ZERO).check(0).is_err());
        // Between check intervals the clock is never read, so even a zero
        // budget does not fire — this is what keeps the per-document cost out.
        assert!(Deadline::new(now, Duration::ZERO).check(1).is_ok());
        assert!(
            Deadline::new(now, Duration::ZERO)
                .check(DEADLINE_CHECK_INTERVAL - 1)
                .is_ok()
        );
        // A check-interval index with the budget spent fires.
        assert!(
            Deadline::new(now, Duration::ZERO)
                .check(DEADLINE_CHECK_INTERVAL)
                .is_err()
        );
        // An ample budget never fires, even at a check-interval index.
        assert!(
            Deadline::new(now, Duration::from_secs(3600))
                .check(0)
                .is_ok()
        );
    }

    #[test]
    fn budget_is_spent_once_elapsed_reaches_it() {
        // #1227: the boundary is inclusive. `(ZERO, ZERO)` is the case that
        // made a zero budget fail to fire when the clock had not advanced
        // (a coarse Windows clock tick, or `Timer` on wasm32).
        assert!(budget_spent(Duration::ZERO, Duration::ZERO));
        let budget = Duration::from_millis(10);
        assert!(budget_spent(budget, budget));
        assert!(budget_spent(budget + Duration::from_nanos(1), budget));
        assert!(!budget_spent(budget - Duration::from_nanos(1), budget));
    }

    /// Build a searcher over a populated index. `segments` commits the docs in
    /// that many batches so we can exercise both the single-segment scan loop
    /// and the multi-segment fanout (Issue #600).
    fn populated_searcher(segments: usize) -> InvertedIndexSearcher {
        use crate::analysis::analyzer::standard::StandardAnalyzer;
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};

        // Through the index (#1024): a standalone writer registers its
        // segments nowhere, so durable fixtures go through the real path.
        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexConfig {
            analyzer: Arc::new(StandardAnalyzer::new().unwrap()),
            ..Default::default()
        };
        let index = InvertedIndex::create(storage, config).unwrap();
        let mut writer = index.writer().unwrap();
        let per_segment = 200;
        for seg in 0..segments.max(1) {
            for i in 0..per_segment {
                let n = seg * per_segment + i;
                writer
                    .add_document(
                        crate::Document::builder()
                            .add_text("content", format!("hello world doc {n}"))
                            .build(),
                    )
                    .unwrap();
            }
            writer.commit().unwrap();
        }
        InvertedIndexSearcher::from_arc(index.reader().unwrap())
    }

    #[test]
    fn search_with_zero_timeout_interrupts_real_docs() {
        // With real matches the scan loop is entered, so the first deadline
        // check (scanned == 0) fires on an already-spent zero budget — proving
        // the timeout interrupts the search rather than only being reported
        // after it completes (Issue #600). A zero budget is spent from the
        // start, even if the clock has not advanced (#1227).
        let searcher = populated_searcher(1);
        let query = Box::new(TermQuery::new("content", "hello")) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query).timeout_ms(0);

        let err = searcher.search(request).unwrap_err();
        assert!(
            err.to_string().contains("timeout"),
            "expected a timeout error, got: {err}"
        );
    }

    #[test]
    fn search_with_zero_timeout_interrupts_multi_segment_fanout() {
        // The per-segment fanout must honour the deadline too (a single slow
        // segment would otherwise leave the parallel fanout unbounded). Each
        // segment's first check fires on the zero budget, even if the clock
        // has not advanced (#1227).
        let searcher = populated_searcher(3);
        let query = Box::new(TermQuery::new("content", "hello")) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query).timeout_ms(0);

        let err = searcher.search(request).unwrap_err();
        assert!(
            err.to_string().contains("timeout"),
            "expected a timeout error, got: {err}"
        );
    }

    #[test]
    fn search_with_zero_timeout_and_no_matches_times_out() {
        // `hello` keeps the query from being short-circuited as empty before
        // the timeout path, but `absent` has no postings, so the conjunction
        // is an empty matcher and no scan loop is entered. Only the final
        // safety net sees the deadline, and it must treat a zero budget as
        // spent too, even if the clock has not advanced (#1227).
        let searcher = populated_searcher(1);
        let query = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("content", "hello")))
                .must(Box::new(TermQuery::new("content", "absent")))
                .build(),
        ) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query).timeout_ms(0);

        let err = searcher.search(request).unwrap_err();
        assert!(
            err.to_string().contains("timeout"),
            "expected a timeout error, got: {err}"
        );
    }

    #[test]
    fn search_without_timeout_returns_hits() {
        // Regression guard: a search with no timeout still returns results
        // (the deadline path is inert when `timeout_ms` is unset).
        let searcher = populated_searcher(1);
        let query = Box::new(TermQuery::new("content", "hello")) as Box<dyn Query>;

        let results = searcher.search(LexicalSearchRequest::new(query)).unwrap();
        assert!(
            !results.hits.is_empty(),
            "a non-timed search must return matching docs"
        );
    }

    #[test]
    fn test_search_with_collector() {
        let searcher = create_test_searcher();
        let query = Box::new(TermQuery::new("title", "hello"));
        let collector = TopDocsCollector::new(10);

        let result_collector = searcher.search_with_collector(query, collector).unwrap();

        assert_eq!(result_collector.total_hits(), 0);
        assert_eq!(result_collector.results().len(), 0);
    }

    #[test]
    fn test_search_empty_query() {
        let searcher = create_test_searcher();
        // Create a boolean query with no clauses (empty query)
        let query = Box::new(BooleanQuery::new()) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query);
        let results = searcher.search(request).unwrap();

        // Should return empty results for empty query
        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_count_empty_query() {
        let searcher = create_test_searcher();
        let query = Box::new(BooleanQuery::new()) as Box<dyn Query>;

        let count = searcher.count(LexicalSearchRequest::new(query)).unwrap();

        // Should return 0 for empty query
        assert_eq!(count, 0);
    }

    #[test]
    fn test_search_request_builder() {
        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;

        let request = LexicalSearchRequest::new(query)
            .limit(20)
            .min_score(0.1)
            .load_documents(false)
            .timeout_ms(5000);

        assert_eq!(request.params.limit, 20);
        assert_eq!(request.params.min_score, 0.1);
        assert!(!request.params.load_documents);
        assert_eq!(request.params.timeout_ms, Some(5000));
    }

    /// Wrapper that suppresses BMW dispatch by returning
    /// `bmw_capable() = false`, so we can run the same query against
    /// the existing matcher-driven path for equivalence comparison.
    #[derive(Debug)]
    struct NonBmwTopDocs(TopDocsCollector);

    impl Collector for NonBmwTopDocs {
        fn collect(&mut self, doc_id: u64, score: f32) -> Result<()> {
            self.0.collect(doc_id, score)
        }
        fn results(&self) -> Vec<crate::lexical::query::SearchHit> {
            self.0.results()
        }
        fn total_hits(&self) -> u64 {
            self.0.total_hits()
        }
        fn needs_more(&self) -> bool {
            self.0.needs_more()
        }
        fn min_score(&self) -> f32 {
            self.0.min_score()
        }
        fn min_competitive(&self) -> f32 {
            self.0.min_competitive()
        }
        fn reset(&mut self) {
            self.0.reset()
        }
        // bmw_capable defaults to false → searcher uses the legacy path.
    }

    /// PR-F: BMW fast path must produce the same top-K (same docs,
    /// same scores) as the existing matcher-driven path on a real
    /// committed index. Skewed-TF distribution drives the heap to
    /// fill quickly and exercises the pivot loop's skip path.
    #[test]
    fn bmw_topk_equivalence_should_or() {
        use crate::Document;
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        // Skewed-TF corpus: alpha clusters at the start of the doc id
        // range; beta middle, gamma tail. With BLOCK_SIZE = 128 this
        // produces a non-trivial distribution of per-block bounds.
        for id in 0..512u64 {
            let mut body = String::new();
            if id < 60 {
                body.push_str("alpha alpha alpha ");
            } else if id < 200 {
                body.push_str("alpha ");
            }
            if (100..400).contains(&id) {
                body.push_str("beta ");
            }
            if id >= 350 && id % 3 == 0 {
                body.push_str("gamma ");
            }
            body.push_str("filler text content body");
            let doc = Document::builder()
                .add_text("title", format!("doc-{id}"))
                .add_text("body", &body)
                .build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        let make_query = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .should(Box::new(TermQuery::new("body", "alpha")))
                    .should(Box::new(TermQuery::new("body", "beta")))
                    .should(Box::new(TermQuery::new("body", "gamma")))
                    .build(),
            )
        };

        // BMW path: bmw_capable() is true on TopDocsCollector, so the
        // entrypoint dispatches to the executor.
        let bmw = store
            .search(LexicalSearchRequest::new(make_query()).limit(10))
            .unwrap();

        // Reference path: same query, same store, but the wrapper
        // collector reports `bmw_capable = false` so the searcher
        // falls through to the existing matcher-driven loop.
        let reference = {
            let request = LexicalSearchRequest::new(make_query()).limit(10);
            // Build a searcher manually so we can pass our wrapper
            // collector through `search_with_collector`. The store's
            // public `search()` always uses TopDocsCollector directly,
            // which bmw_capable's true → BMW.
            let _ = request;
            // Instead: round-trip through the store with a *much*
            // larger K so the heap never fills (min_competitive stays
            // NEG_INFINITY → BMW pivot loop reduces to a doc-by-doc
            // walk identical to the legacy path), then sort + slice.
            let big = store
                .search(LexicalSearchRequest::new(make_query()).limit(usize::MAX))
                .unwrap();
            let mut hits: Vec<_> = big.hits.into_iter().map(|h| (h.doc_id, h.score)).collect();
            hits.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
            hits.truncate(10);
            hits
        };

        let mut bmw_hits: Vec<_> = bmw.hits.iter().map(|h| (h.doc_id, h.score)).collect();
        bmw_hits.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
        assert_eq!(bmw_hits.len(), reference.len(), "result count differs");
        for (idx, (x, y)) in bmw_hits.iter().zip(reference.iter()).enumerate() {
            assert_eq!(x.0, y.0, "rank {idx}: doc_id mismatch");
            assert!(
                (x.1 - y.1).abs() < 1e-4,
                "rank {idx} doc {}: score mismatch bmw={} ref={}",
                x.0,
                x.1,
                y.1,
            );
        }

        // Against the regular path too, now that it scores each clause with
        // its own document's length as BMW does (#1287).
        let searcher = InvertedIndexSearcher::from_arc(store.reader_for_tests().unwrap());
        assert_bmw_top_k_is_exact(&searcher, make_query().as_ref(), 10, "skewed TF");
    }

    /// #1256: the outer `BooleanQuery`'s own boost must reach the BMW
    /// fast path, not just the legacy matcher-driven path (`BooleanScorer`
    /// applies it via `set_boost` in `BooleanQuery::scorer`, but
    /// `BlockMaxOrExecutor::new` used to build its per-clause scorers
    /// straight from each clause's query, never reading
    /// `boolean_query.boost()`).
    #[test]
    fn bmw_applies_should_only_boolean_query_boost() {
        use crate::Document;
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        // Every document's "body" gets the same total token count (padded
        // with "filler"), so every doc's field length is `BODY_LEN` and
        // the reference score below can pass that constant directly
        // instead of resolving it per document.
        const BODY_LEN: usize = 8;
        for id in 0..64u64 {
            let alpha_count = if id % 2 == 0 { 2 } else { 0 };
            let beta_count = if id % 3 == 0 { 1 } else { 0 };
            let filler_count = BODY_LEN - alpha_count - beta_count;
            let mut words = Vec::with_capacity(BODY_LEN);
            words.extend(std::iter::repeat_n("alpha", alpha_count));
            words.extend(std::iter::repeat_n("beta", beta_count));
            words.extend(std::iter::repeat_n("filler", filler_count));
            let doc = Document::builder()
                .add_text("title", format!("doc-{id}"))
                .add_text("body", words.join(" "))
                .build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        let make_query = |boost: f32| -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .should(Box::new(TermQuery::new("body", "alpha")))
                    .should(Box::new(TermQuery::new("body", "beta")))
                    .boost(boost)
                    .build(),
            )
        };

        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader);

        // BMW path: `TopDocsCollector::bmw_capable()` is true, so the
        // searcher entrypoint dispatches to `BlockMaxOrExecutor`.
        let bmw_unboosted = searcher
            .search_with_collector(make_query(1.0), TopDocsCollector::new(64))
            .unwrap();
        let bmw_boosted = searcher
            .search_with_collector(make_query(2.0), TopDocsCollector::new(64))
            .unwrap();

        let sort_hits = |hits: Vec<SearchHit>| -> Vec<(u64, f32)> {
            let mut v: Vec<_> = hits.into_iter().map(|h| (h.doc_id, h.score)).collect();
            v.sort_by_key(|h| h.0);
            v
        };
        let bmw_unboosted_hits = sort_hits(bmw_unboosted.results());
        let bmw_boosted_hits = sort_hits(bmw_boosted.results());

        assert!(!bmw_unboosted_hits.is_empty(), "expected matches");
        assert_eq!(bmw_unboosted_hits.len(), bmw_boosted_hits.len());

        // Acceptance criterion: a boosted Should-only BooleanQuery of
        // TermQuery scores twice as high with boost=2 as with boost=1.
        for ((doc, unboosted), (doc2, boosted)) in
            bmw_unboosted_hits.iter().zip(bmw_boosted_hits.iter())
        {
            assert_eq!(
                doc, doc2,
                "doc_id mismatch between boost=1 and boost=2 runs"
            );
            assert!(
                (boosted - unboosted * 2.0).abs() < 1e-4,
                "doc {doc}: boosted score {boosted} is not double the unboosted score {unboosted}"
            );
        }

        // Acceptance criterion: the BMW path scores a boosted query the
        // same as the "regular path" the issue names — `BooleanQuery::
        // scorer`'s `BooleanScorer`, which sums each clause's score and
        // then multiplies by `self.boost` (`boolean_scorer.set_boost`).
        // Every doc's "body" is `BODY_LEN` tokens, so that constant is
        // this reference scorer's field length too.
        let reference_scorer = make_query(2.0).scorer(searcher.reader().as_ref()).unwrap();
        for (doc, bmw_score) in bmw_boosted_hits.iter() {
            let reference_score = reference_scorer.score(*doc, 0.0, Some(BODY_LEN as f32));
            assert!(
                (bmw_score - reference_score).abs() < 1e-4,
                "doc {doc}: BMW score {bmw_score} != BooleanQuery::scorer score {reference_score}"
            );
        }
    }

    /// Each hit's score, keyed by doc id.
    fn scores_by_doc(hits: Vec<SearchHit>) -> std::collections::HashMap<u64, f32> {
        hits.into_iter().map(|h| (h.doc_id, h.score)).collect()
    }

    /// A Should-only `BooleanQuery` of one `TermQuery` per term on `body`.
    fn should_terms(terms: &[&str]) -> Box<dyn Query> {
        let mut builder = BooleanQueryBuilder::new();
        for term in terms {
            builder = builder.should(Box::new(TermQuery::new("body", *term)));
        }
        Box::new(builder.build())
    }

    /// #1287: a `BooleanQuery` scores each clause with that clause's own
    /// per-document field length, so a document scores as the sum of its
    /// clauses' `TermQuery` scores whichever path runs the query. The regular
    /// path used to hand every clause length 0.
    #[test]
    fn boolean_clauses_score_with_each_documents_own_length() {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        // Docs 0..30 match the query, with lengths growing with the id so
        // length normalization tells them apart; docs 30..60 match nothing
        // and keep the idf away from its floor.
        for id in 0..60u64 {
            let mut words = if id < 30 {
                vec!["alpha"]
            } else {
                vec!["gamma"]
            };
            if id % 2 == 0 && id < 30 {
                words.push("beta");
            }
            words.extend(std::iter::repeat_n("filler", (id % 30) as usize));
            let doc = Document::builder()
                .add_text("body", words.join(" "))
                .build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        let searcher = InvertedIndexSearcher::from_arc(store.reader_for_tests().unwrap());
        let search = |query: Box<dyn Query>| {
            scores_by_doc(
                searcher
                    .search_with_collector(query, TopDocsCollector::new(100))
                    .unwrap()
                    .results(),
            )
        };
        let alpha = search(Box::new(TermQuery::new("body", "alpha")));
        let beta = search(Box::new(TermQuery::new("body", "beta")));

        // Single-segment BMW: `TopDocsCollector` is BMW-capable.
        let bmw = search(should_terms(&["alpha", "beta"]));
        let regular = scores_by_doc(
            searcher
                .search_with_collector(
                    should_terms(&["alpha", "beta"]),
                    NonBmwTopDocs(TopDocsCollector::new(100)),
                )
                .unwrap()
                .0
                .results(),
        );
        // A Must clause keeps the query off BMW.
        let must_should = search(Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "beta")))
                .build(),
        ));

        for doc in 0..30u64 {
            let want = alpha[&doc] + beta.get(&doc).copied().unwrap_or(0.0);
            for (path, scores) in [
                ("BMW", &bmw),
                ("regular", &regular),
                ("must+should", &must_should),
            ] {
                let got = scores[&doc];
                assert!(
                    (got - want).abs() < 1e-5,
                    "{path}: doc {doc} scored {got}, but its clauses sum to {want}"
                );
            }
        }
    }

    /// #1287: inside the per-segment fanout, BMW scored every clause at the
    /// average length, because it could only read field lengths through an
    /// `InvertedIndexReader`, not a `PerSegmentReaderView`. Every segment
    /// here holds both terms, so each one runs BMW.
    #[test]
    fn fanout_boolean_clauses_score_with_each_documents_own_length() {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        // Two segments of 40 documents; in each, the first 20 match the query
        // and the other 20 keep the idf away from its floor.
        let matches = |id: u64| id % 40 < 20;
        for id in 0..80u64 {
            let mut words = if matches(id) {
                vec!["alpha"]
            } else {
                vec!["gamma"]
            };
            if matches(id) && id % 2 == 0 {
                words.push("beta");
            }
            words.extend(std::iter::repeat_n("filler", (id % 20) as usize));
            let doc = Document::builder()
                .add_text("body", words.join(" "))
                .build();
            store.upsert_document(id, doc).unwrap();
            if id == 39 {
                store.commit().unwrap();
            }
        }
        store.commit().unwrap();

        let search = |query: Box<dyn Query>| {
            scores_by_doc(
                store
                    .search(LexicalSearchRequest::new(query).limit(100))
                    .unwrap()
                    .hits,
            )
        };
        let alpha = search(Box::new(TermQuery::new("body", "alpha")));
        let beta = search(Box::new(TermQuery::new("body", "beta")));
        let boolean = search(should_terms(&["alpha", "beta"]));

        assert_eq!(boolean.len(), 40);
        for doc in (0..80u64).filter(|&id| matches(id)) {
            let want = alpha[&doc] + beta.get(&doc).copied().unwrap_or(0.0);
            let got = boolean[&doc];
            assert!(
                (got - want).abs() < 1e-5,
                "doc {doc} scored {got}, but its clauses sum to {want}"
            );
        }
    }

    /// #1287: scored with length 0, a `BooleanQuery` on the regular path
    /// exceeded the per-block bounds its early termination skips by (those
    /// are computed from each document's real length), so a small top-K
    /// could skip the true best document.
    #[test]
    fn regular_path_early_termination_keeps_the_true_top_hit() {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        // `alpha`'s first posting block: short documents, tf 1. Its second
        // block: long documents, tf 2 -- a lower real score, but a higher
        // score than the first block when every length reads as 0.
        for id in 0..256u64 {
            let body = if id < 128 {
                "alpha filler".to_string()
            } else {
                let mut words = vec!["alpha", "alpha"];
                words.extend(std::iter::repeat_n("filler", 38));
                words.join(" ")
            };
            let doc = Document::builder().add_text("body", body).build();
            store.upsert_document(id, doc).unwrap();
        }
        // Documents without `alpha` keep its idf away from the floor.
        for id in 256..556u64 {
            let doc = Document::builder().add_text("body", "gamma filler").build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        let searcher = InvertedIndexSearcher::from_arc(store.reader_for_tests().unwrap());
        // A lone Must clause keeps the query off BMW, and its block bounds are
        // the only ones the regular loop's skip consults.
        let query = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .must(Box::new(TermQuery::new("body", "alpha")))
                    .build(),
            )
        };
        let top = |k: usize| {
            let mut hits = searcher
                .search_with_collector(query(), TopDocsCollector::new(k))
                .unwrap()
                .results();
            hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
            (hits[0].doc_id, hits[0].score)
        };
        let (pruned, exhaustive) = (top(1), top(1000));
        assert_eq!(
            pruned.0, exhaustive.0,
            "top-1 {pruned:?} vs exhaustive {exhaustive:?}"
        );
        assert!((pruned.1 - exhaustive.1).abs() < 1e-5);
    }

    /// Asserts that BMW's top-`k` for `query` is a correct top-`k`: at every
    /// rank it holds the score exhaustive regular-path scoring holds there,
    /// and it scores each document it returns as the regular path does. (Tied
    /// documents may differ.)
    fn assert_bmw_top_k_is_exact(
        searcher: &InvertedIndexSearcher,
        query: &dyn Query,
        k: usize,
        label: &str,
    ) {
        let exhaustive = scores_by_doc(
            searcher
                .search_with_collector(
                    query.clone_box(),
                    NonBmwTopDocs(TopDocsCollector::new(1_000_000)),
                )
                .unwrap()
                .0
                .results(),
        );
        let mut ranked: Vec<f32> = exhaustive.values().copied().collect();
        ranked.sort_by(|a, b| b.total_cmp(a));
        ranked.truncate(k);

        let mut bmw = searcher
            .search_with_collector(query.clone_box(), TopDocsCollector::new(k))
            .unwrap()
            .results();
        bmw.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));

        assert_eq!(bmw.len(), ranked.len(), "{label}: hit count");
        for (rank, (hit, want)) in bmw.iter().zip(&ranked).enumerate() {
            let tol = 1e-5_f32.max(1e-5 * want.abs());
            assert!(
                (hit.score - want).abs() < tol,
                "{label}: rank {rank} holds doc {} at {}, exhaustive scoring has {want}",
                hit.doc_id,
                hit.score
            );
            let reference = exhaustive[&hit.doc_id];
            assert!(
                (hit.score - reference).abs() < tol,
                "{label}: doc {} scored {} by BMW, {reference} by the regular path",
                hit.doc_id,
                hit.score
            );
        }
    }

    /// #1286: the pivot loop picked the pivot from each clause's current-block
    /// bound, then skipped the lagging clauses to the pivot -- straight over a
    /// later block with a higher bound. Here `bravo`'s second block (docs
    /// 128..=255, tf 30) was skipped once doc 0 set the threshold, and K = 1
    /// returned doc 0 (1.79) instead of doc 128 (2.93).
    #[test]
    fn bmw_does_not_skip_a_later_higher_block() {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        const LEN: usize = 40;
        for id in 0..2000u64 {
            let bravo = match id {
                128..=255 => 30,
                0..=399 => 1,
                _ => 0,
            };
            let charlie = match id {
                0 | 1001..=1799 => 1,
                1000 => 3,
                _ => 0,
            };
            let mut words = Vec::with_capacity(LEN);
            words.extend(std::iter::repeat_n("bravo", bravo));
            words.extend(std::iter::repeat_n("charlie", charlie));
            words.extend(std::iter::repeat_n("filler", LEN - bravo - charlie));
            let doc = Document::builder()
                .add_text("body", words.join(" "))
                .build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        let searcher = InvertedIndexSearcher::from_arc(store.reader_for_tests().unwrap());
        let query = should_terms(&["bravo", "charlie"]);
        let top = searcher
            .search_with_collector(query.clone_box(), TopDocsCollector::new(1))
            .unwrap()
            .results();
        assert_eq!(top[0].doc_id, 128, "top hit {top:?}");
        assert_bmw_top_k_is_exact(&searcher, query.as_ref(), 1, "repro");
    }

    /// A deterministic single-segment corpus over `red`, `green` and `blue`.
    /// Each term's presence and term frequency change from one stretch of doc
    /// ids to the next, in an order `seed` decides:
    ///
    /// - a term's posting blocks carry low and high bounds in turn, and
    /// - a term common across the index can be absent from whole stretches,
    ///   which puts it -- as a low-idf pivot -- far ahead of the other terms.
    fn build_varied_block_store(seed: u64) -> crate::lexical::store::LexicalStore {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };

        const TERMS: [&str; 3] = ["red", "green", "blue"];
        const DOCS: u64 = 3000;
        const STRETCH: u64 = 100;
        // Per term and stretch: the percent of documents holding the term,
        // and how high its tf can go.
        let stretches = DOCS.div_ceil(STRETCH) as usize;
        let shape: Vec<Vec<(u32, u32)>> = TERMS
            .iter()
            .map(|_| {
                (0..stretches)
                    .map(|_| {
                        let presence = [0, 0, 0, 3, 40, 90][(next() % 6) as usize];
                        let max_tf = [1, 1, 2, 4, 12, 30][(next() % 6) as usize];
                        (presence, max_tf)
                    })
                    .collect()
            })
            .collect();

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for id in 0..DOCS {
            let stretch = (id / STRETCH) as usize;
            let mut words = Vec::new();
            for (t, term) in TERMS.iter().enumerate() {
                let (presence, max_tf) = shape[t][stretch];
                if next() % 100 < presence {
                    let tf = 1 + next() % max_tf;
                    words.extend(std::iter::repeat_n(*term, tf as usize));
                }
            }
            words.extend(std::iter::repeat_n("filler", (1 + next() % 40) as usize));
            let doc = Document::builder()
                .add_text("body", words.join(" "))
                .build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();
        store
    }

    /// #1286: across varied corpora and K, BMW returns exactly the top-K that
    /// exhaustive scoring does. The miss is rare in these corpora (2 of 2400
    /// seed/query/K combinations over seeds 0..400 before the fix); seeds 127
    /// and 321 are the two that hit it.
    #[test]
    fn bmw_top_k_is_exact_on_varied_blocks() {
        for seed in [0u64, 1, 2, 3, 127, 321] {
            let store = build_varied_block_store(seed);
            let searcher = InvertedIndexSearcher::from_arc(store.reader_for_tests().unwrap());
            for terms in [
                &["red", "green", "blue"][..],
                &["red", "green"][..],
                &["red", "blue"][..],
                &["green", "blue"][..],
            ] {
                for k in [1usize, 3, 10] {
                    let label = format!("seed {seed}, {terms:?}, k {k}");
                    assert_bmw_top_k_is_exact(&searcher, should_terms(terms).as_ref(), k, &label);
                }
            }
        }
    }

    /// Helper for #476 Phase 1 tests: build a `LexicalStore` with
    /// the same skewed-TF corpus as the equivalence test, but split
    /// the writes across `segment_count` commits so the underlying
    /// reader has multiple segments.
    fn build_skewed_store_with_segments(
        segment_count: usize,
    ) -> crate::lexical::store::LexicalStore {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        let n: u64 = 512;
        let chunk = n.div_ceil(segment_count as u64);
        let mut next_commit = chunk;
        for id in 0..n {
            let mut body = String::new();
            if id < 60 {
                body.push_str("alpha alpha alpha ");
            } else if id < 200 {
                body.push_str("alpha ");
            }
            if (100..400).contains(&id) {
                body.push_str("beta ");
            }
            if id >= 350 && id % 3 == 0 {
                body.push_str("gamma ");
            }
            body.push_str("filler text content body");
            let doc = Document::builder()
                .add_text("title", format!("doc-{id}"))
                .add_text("body", &body)
                .build();
            store.upsert_document(id, doc).unwrap();

            if id + 1 == next_commit && id + 1 < n {
                store.commit().unwrap();
                next_commit += chunk;
            }
        }
        store.commit().unwrap();
        store
    }

    /// A 512-document store split into `segment_count` segments, in which
    /// every segment holds every one of `alpha`, `beta` and `gamma` (so a
    /// Should query over them runs BMW in each segment of the fanout), with
    /// term frequencies and lengths varying from document to document.
    fn build_interleaved_store(segment_count: u64) -> crate::lexical::store::LexicalStore {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        const DOCS: u64 = 512;
        let per_segment = DOCS.div_ceil(segment_count);
        for id in 0..DOCS {
            let mut words = Vec::new();
            if id % 2 == 0 {
                words.extend(std::iter::repeat_n(
                    "alpha",
                    if id % 16 == 0 { 4 } else { 1 },
                ));
            }
            if id % 3 == 0 {
                words.extend(std::iter::repeat_n(
                    "beta",
                    if id % 29 == 0 { 6 } else { 1 },
                ));
            }
            if id % 5 == 0 {
                words.push("gamma");
            }
            words.extend(std::iter::repeat_n("filler", 1 + (id % 11) as usize));
            let doc = Document::builder()
                .add_text("body", words.join(" "))
                .build();
            store.upsert_document(id, doc).unwrap();
            if (id + 1) % per_segment == 0 && id + 1 < DOCS {
                store.commit().unwrap();
            }
        }
        store.commit().unwrap();
        store
    }

    /// Asserts that the per-segment fanout's top-`k` for `query` is a correct
    /// top-`k`: rank by rank, it holds the scores an unpruned fanout (a `k`
    /// no heap fills) holds, and it scores each document it returns as the
    /// unpruned fanout does. (Tied documents may differ.)
    fn assert_fanout_top_k_is_exact(
        store: &crate::lexical::store::LexicalStore,
        query: &dyn Query,
        k: usize,
        label: &str,
    ) {
        let search = |limit: usize| {
            let mut hits = store
                .search(LexicalSearchRequest::new(query.clone_box()).limit(limit))
                .unwrap()
                .hits;
            hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));
            hits
        };
        let unpruned = search(1_000_000);
        let unpruned_by_doc: std::collections::HashMap<u64, f32> =
            unpruned.iter().map(|h| (h.doc_id, h.score)).collect();
        let pruned = search(k);

        assert_eq!(pruned.len(), k.min(unpruned.len()), "{label}: hit count");
        for (rank, (hit, want)) in pruned.iter().zip(&unpruned).enumerate() {
            let tol = 1e-5_f32.max(1e-5 * want.score.abs());
            assert!(
                (hit.score - want.score).abs() < tol,
                "{label}: rank {rank} holds doc {} at {}, the unpruned fanout has {}",
                hit.doc_id,
                hit.score,
                want.score
            );
            let reference = unpruned_by_doc[&hit.doc_id];
            assert!(
                (hit.score - reference).abs() < tol,
                "{label}: doc {} scored {} pruned, {reference} unpruned",
                hit.doc_id,
                hit.score
            );
        }
    }

    /// PR-F follow-up #476 Phase 1: the per-segment fanout, with BMW running
    /// in every segment, returns the top-K an unpruned fanout does.
    ///
    /// The reference is the unpruned fanout rather than the regular
    /// cross-segment path: the fanout normalizes lengths with each segment's
    /// own `avg_field_length`, the regular path with the index-wide one, so
    /// their scores differ by design. They used to agree only because both
    /// scored every `BooleanQuery` clause with length 0 (#1287).
    #[test]
    fn per_segment_fanout_topk_matches_unpruned_fanout() {
        let store = build_interleaved_store(4);
        let query = should_terms(&["alpha", "beta", "gamma"]);
        for k in [1usize, 3, 10] {
            assert_fanout_top_k_is_exact(&store, query.as_ref(), k, &format!("k {k}"));
        }
    }

    /// Issue #1257: a `SynonymQuery`'s blended `doc_freq` must come from
    /// the whole index, not from whichever alternatives a segment happens
    /// to hold — otherwise identical content scores differently depending
    /// on which segment it landed in. Segment 1 holds "large" nowhere at
    /// all; segment 2 has "large" far more often than "big". A per-segment
    /// local blend of `max(df(big), df(large))` would therefore be small
    /// in segment 1 (only "big" is visible there) and large in segment 2
    /// (dominated by "large"), giving the two identical documents
    /// different idf — exactly what freezing the blend in `rewrite`
    /// (against the top-level reader, before the fanout) prevents.
    #[test]
    fn synonym_query_scores_identical_docs_the_same_across_segments() {
        use crate::Document;
        use crate::lexical::query::synonym::SynonymQuery;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        // Segment 1: one doc with "big" (doc 0, the comparison target),
        // nine with neither alternative. "large" never appears here.
        let doc = |text: &str| Document::builder().add_text("body", text).build();
        store.upsert_document(0, doc("big filler")).unwrap();
        for id in 1..10u64 {
            store.upsert_document(id, doc("filler filler")).unwrap();
        }
        store.commit().unwrap();

        // Segment 2: one doc with "big" (doc 10, identical content to doc
        // 0), nine with "large" -- df(large) = 9 far exceeds df(big) = 1
        // in this segment alone.
        store.upsert_document(10, doc("big filler")).unwrap();
        for id in 11..20u64 {
            store.upsert_document(id, doc("large filler")).unwrap();
        }
        store.commit().unwrap();

        let query: Box<dyn Query> = Box::new(SynonymQuery::new(
            "body",
            vec!["big".to_string(), "large".to_string()],
        ));
        // `TopDocsCollector` is bmw_capable and segment_count == 2, so
        // this dispatches through `search_per_segment_fanout`, which
        // re-enters `rewrite` per segment (searcher.rs:751) after the
        // top-level call already froze the blended stats.
        let hits = store
            .search(LexicalSearchRequest::new(query).limit(20))
            .unwrap()
            .hits;

        let score_of = |doc_id: u64| {
            hits.iter()
                .find(|h| h.doc_id == doc_id)
                .unwrap_or_else(|| panic!("doc {doc_id} must match"))
                .score
        };
        let (score0, score10) = (score_of(0), score_of(10));
        assert!(
            (score0 - score10).abs() < 1e-4,
            "identical content in different segments must score the same: \
             doc 0 = {score0}, doc 10 = {score10}"
        );
    }

    /// #1120 fixture: two segments with deliberately divergent local
    /// `avg_field_length` (2.0 vs 100.0), so the cross-segment weighted
    /// average (~83.67) exceeds segment A's own local average. Targets
    /// the corrected danger direction: a cross-segment aggregate *larger*
    /// than a matching segment's local average, not smaller.
    ///
    /// `term_in_segment_b` additionally plants a "rare" occurrence in
    /// segment B, producing `matched_count == 2` in
    /// `InvertedIndexReader::term_info` -- exercising the case where the
    /// `matched_count > 1` `block_max` fallback (`reader.rs:2027-2031`)
    /// alone would *not* have restored soundness, since `max_score_factor`
    /// still survives via `max()`.
    fn build_divergent_avg_store(term_in_segment_b: bool) -> crate::lexical::store::LexicalStore {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        // Segment A: 12 docs, local avg_field_length = 2.0. "rare" occurs
        // in every doc (tf=1 in 11 of them, tf=2 in doc 11).
        for id in 0..=10u64 {
            let doc = Document::builder().add_text("body", "rare pad").build();
            store.upsert_document(id, doc).unwrap();
        }
        store
            .upsert_document(
                11,
                Document::builder().add_text("body", "rare rare").build(),
            )
            .unwrap();
        store.commit().unwrap();

        // Segment B: 60 docs, local avg_field_length = 100.0 (no "rare"
        // unless `term_in_segment_b`, kept at exactly length 100 either
        // way so the segment's own average is unaffected).
        for id in 12..=71u64 {
            let body = if term_in_segment_b && id == 12 {
                format!("rare {}", "filler ".repeat(99))
            } else {
                "filler ".repeat(100)
            };
            let doc = Document::builder().add_text("body", body).build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        store
    }

    /// #1120: the BM25 score bound (`TermInfo::max_score_factor`/
    /// `block_max`, precomputed per-segment against that segment's own
    /// `avg_field_length`) must remain a true upper bound against the
    /// cross-segment `BM25Scorer` `TermQuery::scorer` actually builds
    /// (`term.rs:71-104`), even when the cross-segment aggregate average
    /// diverges sharply from a matching segment's local average.
    ///
    /// Mirrors `reader.rs`'s
    /// `block_max_bound_is_never_violated_after_norms_quantisation`: build
    /// the scorer exactly as `TermQuery::scorer` does, then check every
    /// matched document's real score against both the term-level and
    /// per-block bounds.
    #[test]
    fn cross_segment_bm25_bound_holds_with_divergent_segment_avgs() {
        use crate::lexical::query::scorer::{BM25Scorer, Scorer};

        for term_in_segment_b in [false, true] {
            let store = build_divergent_avg_store(term_in_segment_b);
            let reader = store.reader_for_tests().unwrap();
            let inverted = reader
                .as_any()
                .downcast_ref::<InvertedIndexReader>()
                .unwrap();
            assert_eq!(
                inverted.segment_count(),
                2,
                "fixture must produce exactly two segments"
            );

            let term_info = reader.term_info("body", "rare").unwrap().unwrap();
            let field_stats = reader.field_stats("body").unwrap().unwrap();
            assert!(
                field_stats.avg_length > 80.0 && field_stats.avg_length < 84.0,
                "fixture's global avg_length should be ~83.67, got {}",
                field_stats.avg_length
            );

            let scorer = BM25Scorer::with_block_max(
                term_info.doc_freq,
                term_info.total_freq,
                field_stats.doc_count,
                field_stats.avg_length,
                reader.doc_count(),
                1.0,
                term_info.max_score_factor,
                Arc::from(term_info.block_max.into_boxed_slice()),
            );

            // Every document containing "rare": docs 0..=10 (tf=1, len 2),
            // doc 11 (tf=2, len 2), and doc 12 (tf=1, len 100) when planted.
            let mut docs: Vec<(u64, f32, f32)> = (0..=10u64).map(|id| (id, 1.0, 2.0)).collect();
            docs.push((11, 2.0, 2.0));
            if term_in_segment_b {
                docs.push((12, 1.0, 100.0));
            }

            for (doc_id, tf, field_length) in docs {
                let score = scorer.score(doc_id, tf, Some(field_length));
                assert!(
                    score <= scorer.max_score() + 1e-4,
                    "term_in_segment_b={term_in_segment_b}, doc {doc_id}: score {score} \
                     exceeds the term-level bound {}",
                    scorer.max_score()
                );
                assert!(
                    score <= scorer.block_max_score_at(doc_id) + 1e-4,
                    "term_in_segment_b={term_in_segment_b}, doc {doc_id}: score {score} \
                     exceeds the block-max bound {}",
                    scorer.block_max_score_at(doc_id)
                );
            }
        }
    }

    /// #1120 end-to-end: a collector that reaches the cross-segment
    /// matcher-driven path (bypassing the per-segment fanout, e.g. via
    /// `NonBmwTopDocs`) must not lose a real top-1 document to the score
    /// bound this issue fixes. Before the fix, doc 11 (the true top scorer
    /// -- tf=2 among length-2 docs) was pruned because the stale bound
    /// (anchored to segment A's local avg=2.0) fell below doc 11's real
    /// score once the scorer used the larger cross-segment average.
    #[test]
    fn cross_segment_topk_is_not_pruned_by_stale_score_bound() {
        use crate::lexical::query::SearchHit;

        let store = build_divergent_avg_store(false);
        let reader = store.reader_for_tests().unwrap();

        // Ground truth: a heap large enough that it never fills, so
        // `min_competitive` stays `NEG_INFINITY` and no pruning occurs
        // (same technique as `bmw_topk_equivalence_should_or`).
        let ground_truth: Vec<SearchHit> = {
            let searcher = InvertedIndexSearcher::from_arc(reader.clone());
            let collector = NonBmwTopDocs(TopDocsCollector::new(usize::MAX));
            searcher
                .search_with_collector(Box::new(TermQuery::new("body", "rare")), collector)
                .unwrap()
                .0
                .results()
        };

        // The path under test: a real top-1 request through the same
        // non-fanout collector.
        let top1: Vec<SearchHit> = {
            let searcher = InvertedIndexSearcher::from_arc(reader);
            let collector = NonBmwTopDocs(TopDocsCollector::new(1));
            searcher
                .search_with_collector(Box::new(TermQuery::new("body", "rare")), collector)
                .unwrap()
                .0
                .results()
        };

        let mut sorted_truth = ground_truth.clone();
        sorted_truth.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc_id.cmp(&b.doc_id)));

        assert_eq!(top1.len(), 1, "top-1 request must return exactly one hit");
        assert_eq!(
            top1[0].doc_id,
            sorted_truth[0].doc_id,
            "top-1 must match the unpruned ground truth's top document \
             (ground truth top-5: {:?})",
            sorted_truth[..5.min(sorted_truth.len())]
                .iter()
                .map(|h| (h.doc_id, h.score))
                .collect::<Vec<_>>()
        );
    }

    /// PR-F follow-up #476 Phase 1: the per-segment fanout must
    /// fall through to the legacy path when the store has only one
    /// segment (the existing PR-F BMW path is already optimal).
    #[test]
    fn per_segment_fanout_falls_back_when_single_segment() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;

        let store = build_skewed_store_with_segments(1);
        let query: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .should(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "beta")))
                .build(),
        );
        // The fact that this returns at all (without panicking on the
        // `expect("requires InvertedIndexReader")` in fanout) is the
        // proof: the dispatch saw `segment_count() == 1` and skipped
        // the fanout branch.
        let hits = store
            .search(LexicalSearchRequest::new(query).limit(10))
            .unwrap()
            .hits;
        assert!(!hits.is_empty(), "single-seg query should return hits");
    }

    /// #944 Phase B: the pruning predicate must keep a segment whenever
    /// pruning is not provably safe, and only drop one that cannot beat
    /// the current K-th best.
    #[test]
    fn segment_can_contribute_prunes_only_provably_worse_segments() {
        let full = |min: f64, max: f64| {
            Some(SegmentSortRange {
                min,
                max,
                covers_every_doc: true,
            })
        };
        let with_nulls = |min: f64, max: f64| {
            Some(SegmentSortRange {
                min,
                max,
                covers_every_doc: false,
            })
        };

        // No floor yet (fewer than K collected) — never prune.
        assert!(segment_can_contribute(full(0.0, 1.0), None, false));
        assert!(segment_can_contribute(full(0.0, 1.0), None, true));
        // Unknown range (no BKD, unsupported type) — never prune.
        assert!(segment_can_contribute(None, Some(10.0), false));
        assert!(segment_can_contribute(None, Some(10.0), true));

        // Descending: keep when the maximum can still reach the floor.
        assert!(segment_can_contribute(full(0.0, 20.0), Some(10.0), false));
        assert!(
            segment_can_contribute(full(0.0, 10.0), Some(10.0), false),
            "a tie must be kept — the doc-id tie-break can still win"
        );
        assert!(
            !segment_can_contribute(full(0.0, 9.0), Some(10.0), false),
            "a maximum strictly below the floor cannot contribute"
        );

        // Ascending: keep when the minimum can still reach the floor.
        assert!(segment_can_contribute(full(5.0, 20.0), Some(10.0), true));
        assert!(
            segment_can_contribute(full(10.0, 20.0), Some(10.0), true),
            "a tie must be kept"
        );
        assert!(
            !segment_can_contribute(full(11.0, 20.0), Some(10.0), true),
            "a minimum strictly above the floor cannot contribute"
        );

        // Nulls sort greatest: they block descending pruning outright,
        // but cannot improve an ascending segment's best key.
        assert!(
            segment_can_contribute(with_nulls(0.0, 9.0), Some(10.0), false),
            "a segment that may hold a Null outranks everything descending"
        );
        assert!(
            !segment_can_contribute(with_nulls(11.0, 20.0), Some(10.0), true),
            "Nulls are worst ascending, so they do not save the segment"
        );
    }

    /// #944 Phase B: the two-wave split serializes the lead segment, so
    /// it must only be taken when the ranges could actually prune
    /// something. Uniformly-ranged segments — the shape of any sort
    /// field uncorrelated with commit boundaries — must report "no
    /// pruning possible" so the caller keeps the fully parallel fan-out.
    #[test]
    fn pruning_is_possible_only_when_ranges_permit_it() {
        let full = |min: f64, max: f64| {
            Some(SegmentSortRange {
                min,
                max,
                covers_every_doc: true,
            })
        };
        let with_nulls = |min: f64, max: f64| {
            Some(SegmentSortRange {
                min,
                max,
                covers_every_doc: false,
            })
        };

        // Every segment spans the same range: whatever floor the lead
        // produces, no other segment is provably worse.
        let uniform = vec![full(0.0, 100.0), full(0.0, 100.0), full(0.0, 100.0)];
        assert!(
            !pruning_is_possible(&uniform, 0, true),
            "uniform ranges can never prune ascending"
        );
        assert!(
            !pruning_is_possible(&uniform, 0, false),
            "uniform ranges can never prune descending"
        );

        // Disjoint ranges: the lead's extreme already excludes the rest.
        let disjoint = vec![full(0.0, 9.0), full(10.0, 19.0), full(20.0, 29.0)];
        assert!(
            pruning_is_possible(&disjoint, 2, false),
            "descending, the newest segment's minimum excludes the older ones"
        );
        assert!(
            pruning_is_possible(&disjoint, 0, true),
            "ascending, the oldest segment's maximum excludes the newer ones"
        );

        // Partial overlap still permits pruning: should the lead fill
        // its top-K entirely at its own extreme, the floor reaches that
        // extreme and excludes the lower-ranged segment.
        let overlapping = vec![full(0.0, 30.0), full(10.0, 40.0)];
        assert!(pruning_is_possible(&overlapping, 1, false));
        assert!(pruning_is_possible(&overlapping, 0, true));

        // Sharing the lead's extreme, however, is decisive: no floor the
        // lead can produce ever excludes such a segment.
        assert!(!pruning_is_possible(
            &[full(0.0, 40.0), full(10.0, 40.0)],
            1,
            false
        ));
        assert!(!pruning_is_possible(
            &[full(0.0, 30.0), full(0.0, 40.0)],
            0,
            true
        ));

        // An unknown lead range yields no floor at all.
        let unknown_lead = vec![None, full(10.0, 19.0)];
        assert!(!pruning_is_possible(&unknown_lead, 0, false));
        // An unknown range elsewhere is simply never prunable.
        let unknown_other = vec![full(20.0, 29.0), None];
        assert!(!pruning_is_possible(&unknown_other, 0, false));

        // Descending, a segment that may hold a `Null` blocks pruning
        // even though its range lies entirely below the lead's.
        let nullable = vec![full(20.0, 29.0), with_nulls(0.0, 9.0)];
        assert!(!pruning_is_possible(&nullable, 0, false));
        assert!(
            pruning_is_possible(&nullable, 1, true),
            "ascending, Nulls are worst, so the range still bounds the segment"
        );

        // A lone segment has nothing to prune.
        assert!(!pruning_is_possible(&[full(0.0, 9.0)], 0, false));
    }

    /// #944 Phase B: on a real multi-segment index whose commits carry
    /// disjoint value ranges, the ranges read from the BKD headers must
    /// actually be disjoint and the predicate must prune the older
    /// segments against the newest one's floor. The end-to-end tests
    /// only prove results are unchanged — which they would be with
    /// pruning disabled too — so this pins that pruning really fires.
    #[test]
    fn disjoint_segments_are_pruned_against_the_lead_floor() {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage, InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();
        // Three commits, values strictly increasing: [0..4), [10..14), [20..24).
        for group in 0..3u64 {
            for offset in 0..4u64 {
                writer
                    .add_document(
                        crate::Document::builder()
                            .add_integer("popularity", (group * 10 + offset) as i64)
                            .build(),
                    )
                    .unwrap();
            }
            writer.commit().unwrap();
        }
        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let segments = inverted.segment_readers().to_vec();
        assert_eq!(segments.len(), 3);

        let ranges: Vec<Option<SegmentSortRange>> = segments
            .iter()
            .map(|seg| segment_sort_range(seg, "popularity"))
            .collect();
        for (i, r) in ranges.iter().enumerate() {
            let r = r.unwrap_or_else(|| panic!("segment {i} must expose a range"));
            assert_eq!(r.min, (i as f64) * 10.0);
            assert_eq!(r.max, (i as f64) * 10.0 + 3.0);
            assert!(r.covers_every_doc, "every doc carries the field");
        }

        // Descending: the newest segment leads and its lowest value (20)
        // becomes the floor once it fills a top-4.
        let lead = best_segment_index(&ranges, false).unwrap();
        assert_eq!(lead, 2, "the highest-valued segment must lead");
        assert!(
            pruning_is_possible(&ranges, lead, false),
            "disjoint commits must opt into the two-wave split"
        );
        let floor = Some(20.0);
        assert!(
            !segment_can_contribute(ranges[0], floor, false)
                && !segment_can_contribute(ranges[1], floor, false),
            "older segments must be pruned against the lead floor"
        );

        // Ascending mirrors it: the oldest leads, its highest value (3)
        // is the floor, and the newer segments are pruned.
        let lead_asc = best_segment_index(&ranges, true).unwrap();
        assert_eq!(lead_asc, 0);
        let floor_asc = Some(3.0);
        assert!(
            !segment_can_contribute(ranges[1], floor_asc, true)
                && !segment_can_contribute(ranges[2], floor_asc, true),
            "newer segments must be pruned ascending"
        );
    }

    /// #944 Phase B: the mirror case — commits whose sort values all
    /// span the same range. Reading the real BKD headers must show the
    /// ranges overlapping, so the search keeps the fully parallel
    /// fan-out instead of paying for a serialized lead wave that could
    /// never prune anything.
    #[test]
    fn uniform_segments_keep_the_parallel_fanout() {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage, InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();
        // Three commits, each spanning the identical range [0, 30].
        for _ in 0..3 {
            for offset in 0..4u64 {
                writer
                    .add_document(
                        crate::Document::builder()
                            .add_integer("popularity", (offset * 10) as i64)
                            .build(),
                    )
                    .unwrap();
            }
            writer.commit().unwrap();
        }
        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let segments = inverted.segment_readers().to_vec();
        assert_eq!(segments.len(), 3);

        let ranges: Vec<Option<SegmentSortRange>> = segments
            .iter()
            .map(|seg| segment_sort_range(seg, "popularity"))
            .collect();
        for (i, r) in ranges.iter().enumerate() {
            let r = r.unwrap_or_else(|| panic!("segment {i} must expose a range"));
            assert_eq!((r.min, r.max), (0.0, 30.0));
        }

        for ascending in [true, false] {
            let lead = best_segment_index(&ranges, ascending).unwrap();
            assert!(
                !pruning_is_possible(&ranges, lead, ascending),
                "overlapping ranges must not opt into the two-wave split"
            );
        }
    }

    /// #944 Phase B: sort keys convert to point space exactly as the
    /// writer indexed them; types that index no point are unprunable.
    #[test]
    fn sort_key_as_point_mirrors_the_writer() {
        use crate::lexical::core::field::FieldValue;

        assert_eq!(sort_key_as_point(&FieldValue::Int64(42)), Some(42.0));
        assert_eq!(sort_key_as_point(&FieldValue::Float64(1.5)), Some(1.5));
        let dt = chrono::DateTime::from_timestamp(1_600_000_000, 0).unwrap();
        assert_eq!(
            sort_key_as_point(&FieldValue::DateTime(dt)),
            Some(1_600_000_000.0)
        );

        assert_eq!(sort_key_as_point(&FieldValue::Text("x".into())), None);
        assert_eq!(sort_key_as_point(&FieldValue::Bool(true)), None);
        assert_eq!(sort_key_as_point(&FieldValue::Null), None);
    }

    /// #1127: the lead floor is read off the value the per-segment
    /// collector ranked its K-th hit by. Below a full top-K there is no
    /// floor; a K-th key with no point-space image (Null, Text) cannot
    /// bound anything either.
    #[test]
    fn lead_floor_needs_a_full_top_k_and_a_point_typed_worst() {
        use crate::lexical::core::field::FieldValue;
        use crate::lexical::query::collector::FieldHit;

        let hit = |doc_id: u64, value: FieldValue| FieldHit {
            doc_id,
            score: 0.0,
            value,
        };
        let full = vec![
            hit(1, FieldValue::Int64(23)),
            hit(2, FieldValue::Int64(22)),
            hit(3, FieldValue::Int64(20)),
        ];

        assert_eq!(
            lead_floor(&full, 3),
            Some(20.0),
            "the K-th (last, worst) hit is the floor"
        );
        assert_eq!(lead_floor(&full, 4), None, "fewer than K hits: no floor");
        assert_eq!(lead_floor(&full[..2], 3), None);
        assert_eq!(
            lead_floor(&[], 0),
            None,
            "limit 0 collects nothing to bound"
        );

        let null_worst = vec![hit(1, FieldValue::Int64(23)), hit(2, FieldValue::Null)];
        assert_eq!(
            lead_floor(&null_worst, 2),
            None,
            "Null has no point-space image"
        );
        let text_worst = vec![
            hit(1, FieldValue::Text("b".into())),
            hit(2, FieldValue::Text("a".into())),
        ];
        assert_eq!(lead_floor(&text_worst, 2), None);

        let dt = chrono::DateTime::from_timestamp(1_600_000_000, 0).unwrap();
        assert_eq!(
            lead_floor(&[hit(1, FieldValue::DateTime(dt))], 1),
            Some(1_600_000_000.0)
        );
    }

    /// #944 Phase A prerequisite: the per-segment view must forward
    /// DocValues access to its segment — with the trait defaults a
    /// per-segment `TopFieldCollector` would see every doc as `Null`.
    #[test]
    fn per_segment_view_forwards_doc_values() {
        use crate::lexical::core::field::FieldValue;
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::per_segment_view::PerSegmentReaderView;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage, InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();
        writer
            .add_document(
                crate::Document::builder()
                    .add_integer("popularity", 42)
                    .build(),
            )
            .unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let seg = inverted.segment_readers()[0].clone();

        let view = PerSegmentReaderView::new(
            seg,
            inverted.doc_count(),
            inverted.max_doc(),
            std::sync::Arc::new(|_: &str, _: &str| Ok(None)),
            std::sync::Arc::new(|_: &dyn crate::lexical::query::Query| {
                Ok(std::sync::Arc::new(RoaringTreemap::new()))
            }),
        );

        assert!(
            view.has_doc_values("popularity"),
            "view must forward has_doc_values to its segment"
        );
        let value = view.get_doc_value("popularity", 0).unwrap();
        assert!(
            matches!(value, Some(FieldValue::Int64(42))),
            "view must forward get_doc_value to its segment; got {value:?}"
        );
    }

    /// PR-F follow-up #476 Phase 1: a non-`bmw_capable` collector
    /// (here: `CountCollector`) must skip both the BMW fast path and
    /// the per-segment fanout, so multi-segment count queries still
    /// hit the legacy aggregation path.
    #[test]
    fn per_segment_fanout_falls_back_for_count_collector() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;

        let store = build_skewed_store_with_segments(4);
        let query: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .should(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "beta")))
                .build(),
        );
        let count = store.count(LexicalSearchRequest::new(query)).unwrap();
        assert!(count > 0, "count query on multi-seg corpus must hit");
    }

    // ----- Issue #578: query / filter result cache -----

    /// `matching_doc_ids` must return exactly the doc-id set that an unbounded
    /// `search` produces, for both a term filter and a boolean filter. The
    /// cache is score-independent, so only the *set* (not scores) is compared.
    #[test]
    fn matching_doc_ids_matches_search_hit_set() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        use std::collections::BTreeSet;

        let store = build_skewed_store_with_segments(1);

        let cases: Vec<Box<dyn Query>> = vec![
            Box::new(TermQuery::new("body", "alpha")),
            Box::new(
                BooleanQueryBuilder::new()
                    .must(Box::new(TermQuery::new("body", "alpha")))
                    .should(Box::new(TermQuery::new("body", "beta")))
                    .build(),
            ),
        ];

        for query in cases {
            let bitmap = store.matching_doc_ids(query.clone_box()).unwrap();
            let cached_set: BTreeSet<u64> = bitmap.iter().collect();

            let search_set: BTreeSet<u64> = store
                .search(
                    LexicalSearchRequest::new(query.clone_box())
                        .limit(usize::MAX)
                        .load_documents(false),
                )
                .unwrap()
                .hits
                .into_iter()
                .map(|h| h.doc_id)
                .collect();

            assert_eq!(
                cached_set,
                search_set,
                "matching_doc_ids must equal the search hit set for {}",
                query.description()
            );
            assert!(!cached_set.is_empty(), "corpus should match the query");
        }
    }

    /// A repeated cacheable lookup against the same reader snapshot is served
    /// from the cache: it returns the very same `Arc` and bumps the hit
    /// counter.
    #[test]
    fn matching_doc_ids_cache_hit_returns_shared_arc() {
        let store = build_skewed_store_with_segments(1);
        let reader = store.reader_for_tests().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .expect("memory store yields an InvertedIndexReader");

        let query: Box<dyn Query> = Box::new(TermQuery::new("body", "alpha"));

        let first = inverted.matching_doc_ids(query.as_ref()).unwrap();
        let second = inverted.matching_doc_ids(query.as_ref()).unwrap();

        assert_eq!(first, second, "cache hit must return the same set");
        assert!(
            Arc::ptr_eq(&first, &second),
            "second lookup should be served from the cache (same Arc)"
        );

        let stats = inverted.query_cache_stats();
        assert_eq!(stats.misses, 1, "first lookup is a miss");
        assert_eq!(stats.hits, 1, "second lookup is a hit");
    }

    /// Deleted documents must not appear in a cached filter set (deletions are
    /// filtered at the posting-iterator level, before the matcher).
    #[test]
    fn matching_doc_ids_excludes_deleted_docs() {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for id in 0..10u64 {
            let doc = Document::builder().add_text("body", "shared term").build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        let query = || -> Box<dyn Query> { Box::new(TermQuery::new("body", "shared")) };
        let before = store.matching_doc_ids(query()).unwrap();
        assert_eq!(before.len(), 10);

        store.delete_document_by_internal_id(3).unwrap();
        store.commit().unwrap();

        let after = store.matching_doc_ids(query()).unwrap();
        assert_eq!(after.len(), 9, "deleted doc must be excluded");
        assert!(!after.contains(3), "doc 3 was deleted");
    }

    /// `commit` drops the cached searcher (and its reader's cache), so the next
    /// lookup recomputes against the new snapshot and sees freshly added docs.
    #[test]
    fn commit_invalidates_query_filter_cache() {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for id in 0..5u64 {
            let doc = Document::builder().add_text("body", "rust").build();
            store.upsert_document(id, doc).unwrap();
        }
        store.commit().unwrap();

        let query = || -> Box<dyn Query> { Box::new(TermQuery::new("body", "rust")) };
        let before = store.matching_doc_ids(query()).unwrap();
        assert_eq!(before.len(), 5);

        // Add a matching doc and commit; the cached searcher is invalidated.
        store
            .upsert_document(99, Document::builder().add_text("body", "rust").build())
            .unwrap();
        store.commit().unwrap();

        let after = store.matching_doc_ids(query()).unwrap();
        assert_eq!(
            after.len(),
            6,
            "post-commit lookup must see the new doc (cache invalidated)"
        );
        assert!(after.contains(99));
    }

    /// A query whose `cache_key` is `None` (here a MustNot-only boolean, R1)
    /// must never touch the cache: it recomputes each call (distinct `Arc`) and
    /// leaves the hit/miss counters untouched, while still returning a stable,
    /// correct set.
    #[test]
    fn uncacheable_query_bypasses_cache() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;

        let store = build_skewed_store_with_segments(1);
        let reader = store.reader_for_tests().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();

        let make = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .must_not(Box::new(TermQuery::new("body", "alpha")))
                    .build(),
            )
        };
        assert!(
            make().cache_key().is_none(),
            "MustNot-only boolean must be uncacheable"
        );

        let first = inverted.matching_doc_ids(make().as_ref()).unwrap();
        let second = inverted.matching_doc_ids(make().as_ref()).unwrap();

        assert_eq!(
            first, second,
            "uncacheable query still returns a stable set"
        );
        assert!(
            !Arc::ptr_eq(&first, &second),
            "uncacheable query must recompute (a distinct Arc each call)"
        );
        let stats = inverted.query_cache_stats();
        assert_eq!(stats.hits, 0, "uncacheable query never hits the cache");
        assert_eq!(stats.misses, 0, "uncacheable query never probes the cache");
    }

    /// Many threads hammering the same cached filter must not deadlock or race
    /// on the cache `Mutex`, and every thread must observe the same set.
    #[test]
    fn concurrent_matching_doc_ids_is_consistent() {
        use std::collections::BTreeSet;
        use std::thread;

        let store = Arc::new(build_skewed_store_with_segments(1));
        // Prime the cached searcher so all threads share one reader + cache.
        let expected: BTreeSet<u64> = store
            .matching_doc_ids(Box::new(TermQuery::new("body", "alpha")))
            .unwrap()
            .iter()
            .collect();
        assert!(!expected.is_empty());

        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = Arc::clone(&store);
            let expected = expected.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    let set: BTreeSet<u64> = store
                        .matching_doc_ids(Box::new(TermQuery::new("body", "alpha")))
                        .unwrap()
                        .iter()
                        .collect();
                    assert_eq!(set, expected, "every thread sees the same cached set");
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }

    // ----- Issue #764: Occur::Filter clause reuses the filter cache -----

    /// A repeated `must(...).filter(...)` search must serve the `Occur::Filter`
    /// clause from `QueryFilterCache` (single-segment / non-fanout path).
    #[test]
    fn filter_clause_reuses_cache_single_segment() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;

        let store = build_skewed_store_with_segments(1);
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader.clone());

        let make = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .must(Box::new(TermQuery::new("body", "alpha")))
                    .filter(Box::new(TermQuery::new("body", "beta")))
                    .build(),
            )
        };

        // First search populates the filter-clause set; second reuses it.
        let _ = searcher
            .search_with_collector(make(), TopDocsCollector::new(10))
            .unwrap();
        let _ = searcher
            .search_with_collector(make(), TopDocsCollector::new(10))
            .unwrap();

        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let stats = inverted.query_cache_stats();
        assert!(
            stats.hits >= 1,
            "the Occur::Filter clause must hit the cache on the repeat search (stats: {stats:?})"
        );
    }

    /// Cache-on must produce exactly the same result set as cache-off for a
    /// filtered boolean across a multi-segment index (exercises the fanout
    /// path through `PerSegmentReaderView::matching_doc_ids`).
    #[test]
    fn filter_clause_cache_matches_uncached_multi_segment() {
        use crate::Document;
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;
        use std::collections::BTreeSet;

        // Build a 4-segment store with the given cache capacity. alpha = even
        // ids, beta = multiples of 3, so must(alpha) ∩ filter(beta) = ids % 6.
        let build = |capacity: usize| -> LexicalStore {
            let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            let config = LexicalIndexConfig::builder()
                .query_filter_cache_capacity(capacity)
                .build();
            let store = LexicalStore::new(storage, config).unwrap();
            for id in 0..400u64 {
                let mut body = String::new();
                if id % 2 == 0 {
                    body.push_str("alpha ");
                }
                if id % 3 == 0 {
                    body.push_str("beta ");
                }
                body.push_str("filler");
                let doc = Document::builder().add_text("body", &body).build();
                store.upsert_document(id, doc).unwrap();
                if id % 100 == 99 {
                    store.commit().unwrap();
                }
            }
            store.commit().unwrap();
            store
        };

        let make = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .must(Box::new(TermQuery::new("body", "alpha")))
                    .filter(Box::new(TermQuery::new("body", "beta")))
                    .build(),
            )
        };
        let run = |store: &LexicalStore| -> BTreeSet<u64> {
            store
                .search(
                    LexicalSearchRequest::new(make())
                        .limit(usize::MAX)
                        .load_documents(false),
                )
                .unwrap()
                .hits
                .into_iter()
                .map(|h| h.doc_id)
                .collect()
        };

        let cached_set = run(&build(1024));
        let uncached_set = run(&build(0));

        assert_eq!(
            cached_set, uncached_set,
            "cache-on must equal cache-off for a filtered boolean (fanout path)"
        );
        assert!(!cached_set.is_empty(), "filter should match some docs");
        assert!(
            cached_set.iter().all(|&d| d % 6 == 0),
            "must(alpha=even) ∩ filter(beta=%3) == ids divisible by 6"
        );
    }

    // ----- Issue #587: Roaring-backed parallel boolean executor -----

    /// Single-segment store with a tiny, hand-checkable corpus for the parallel
    /// boolean set-logic tests:
    /// doc0=alpha, doc1=alpha+beta, doc2=beta, doc3=alpha+beta+gamma, doc4=gamma.
    fn build_boolean_corpus() -> crate::lexical::store::LexicalStore {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for (id, body) in [
            (0u64, "alpha"),
            (1, "alpha beta"),
            (2, "beta"),
            (3, "alpha beta gamma"),
            (4, "gamma"),
        ] {
            store
                .upsert_document(id, Document::builder().add_text("body", body).build())
                .unwrap();
        }
        store.commit().unwrap();
        store
    }

    /// Drive `query` through the parallel boolean executor and return the
    /// sorted result doc-id set.
    fn parallel_doc_ids(
        store: &crate::lexical::store::LexicalStore,
        query: Box<dyn Query>,
    ) -> Vec<u64> {
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader);
        let mut ids: Vec<u64> = searcher
            .search_with_collector_parallel(query, TopDocsCollector::new(100), true)
            .unwrap()
            .results()
            .into_iter()
            .map(|h| h.doc_id)
            .collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn parallel_boolean_must_should_membership() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        // Must(alpha) + Should(gamma): membership = docs with alpha (gamma only boosts).
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "gamma")))
                .build(),
        );
        assert_eq!(parallel_doc_ids(&build_boolean_corpus(), q), vec![0, 1, 3]);
    }

    #[test]
    fn parallel_boolean_must_not_excludes() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        // Must(alpha) AND NOT beta -> only doc 0.
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .must_not(Box::new(TermQuery::new("body", "beta")))
                .build(),
        );
        assert_eq!(parallel_doc_ids(&build_boolean_corpus(), q), vec![0]);
    }

    #[test]
    fn parallel_boolean_filter_narrows() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        // Must(alpha) AND Filter(beta) -> alpha ∩ beta = {1, 3}.
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .filter(Box::new(TermQuery::new("body", "beta")))
                .build(),
        );
        assert_eq!(parallel_doc_ids(&build_boolean_corpus(), q), vec![1, 3]);
    }

    #[test]
    fn parallel_boolean_should_only_union() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        // Should(alpha) OR Should(beta) -> {0, 1, 2, 3}.
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .should(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "beta")))
                .build(),
        );
        assert_eq!(
            parallel_doc_ids(&build_boolean_corpus(), q),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn parallel_boolean_minimum_should_match() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        // Must(alpha) + Should(beta) with msm=1 -> alpha ∩ beta = {1, 3}.
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "beta")))
                .minimum_should_match(1)
                .build(),
        );
        assert_eq!(parallel_doc_ids(&build_boolean_corpus(), q), vec![1, 3]);
    }

    /// Should scores accumulate onto Must candidates: a Must doc that also
    /// matches a Should clause must outrank one that does not.
    #[test]
    fn parallel_boolean_should_boosts_score() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        let store = build_boolean_corpus();
        let q: Box<dyn Query> = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("body", "alpha")))
                .should(Box::new(TermQuery::new("body", "gamma")))
                .build(),
        );
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader);
        let hits = searcher
            .search_with_collector_parallel(q, TopDocsCollector::new(100), true)
            .unwrap()
            .results();
        let s3 = hits.iter().find(|h| h.doc_id == 3).unwrap().score;
        let s0 = hits.iter().find(|h| h.doc_id == 0).unwrap().score;
        assert!(
            s3 > s0,
            "doc 3 (alpha+gamma) must outscore doc 0 (alpha only): s3={s3} s0={s0}"
        );
        assert_eq!(hits[0].doc_id, 3, "the should-boosted doc must rank first");
    }

    /// Parallel and serial paths must agree on membership for a Must-present
    /// shape (both implement the same boolean membership there).
    #[test]
    fn parallel_matches_serial_membership() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;
        use std::collections::BTreeSet;

        let store = build_boolean_corpus();
        let make = || -> Box<dyn Query> {
            Box::new(
                BooleanQueryBuilder::new()
                    .must(Box::new(TermQuery::new("body", "alpha")))
                    .should(Box::new(TermQuery::new("body", "gamma")))
                    .must_not(Box::new(TermQuery::new("body", "beta")))
                    .build(),
            )
        };
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader);
        let collect_ids = |parallel: bool| -> BTreeSet<u64> {
            searcher
                .search_with_collector_parallel(make(), TopDocsCollector::new(100), parallel)
                .unwrap()
                .results()
                .into_iter()
                .map(|h| h.doc_id)
                .collect()
        };
        let par = collect_ids(true);
        assert_eq!(
            par,
            collect_ids(false),
            "parallel/serial membership must agree"
        );
        // alpha ∩ not beta = {0} (doc 3 has beta → excluded; gamma only boosts).
        assert_eq!(par.into_iter().collect::<Vec<_>>(), vec![0]);
    }

    // ----- Issue #590: parsed-DSL query cache -----

    /// A repeated DSL search is parsed once: the second call is a cache hit and
    /// returns the identical result set (Issue #590).
    #[test]
    fn dsl_parse_cache_hit_on_repeat() {
        let store = build_skewed_store_with_segments(1);
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader);

        let req = || {
            LexicalSearchRequest::from_dsl("body:alpha")
                .limit(10)
                .load_documents(false)
        };
        let ids1: Vec<u64> = searcher
            .search(req())
            .unwrap()
            .hits
            .iter()
            .map(|h| h.doc_id)
            .collect();
        let ids2: Vec<u64> = searcher
            .search(req())
            .unwrap()
            .hits
            .iter()
            .map(|h| h.doc_id)
            .collect();

        assert_eq!(ids1, ids2, "repeat DSL search must return the same results");
        assert!(!ids1.is_empty(), "corpus should match body:alpha");

        let stats = searcher.parsed_query_cache_stats();
        assert_eq!(
            stats.misses, 1,
            "the DSL is parsed once (first call misses)"
        );
        assert!(stats.hits >= 1, "the repeat DSL search hits the cache");
    }

    /// With the cache disabled (capacity 0) the DSL is parsed every time, yet
    /// results are unchanged.
    #[test]
    fn dsl_parse_cache_disabled_still_correct() {
        let store = build_skewed_store_with_segments(1);
        let reader = store.reader_for_tests().unwrap();
        let searcher = InvertedIndexSearcher::from_arc(reader).with_parsed_query_cache_capacity(0);

        let req = || {
            LexicalSearchRequest::from_dsl("body:alpha")
                .limit(10)
                .load_documents(false)
        };
        let ids1: Vec<u64> = searcher
            .search(req())
            .unwrap()
            .hits
            .iter()
            .map(|h| h.doc_id)
            .collect();
        let ids2: Vec<u64> = searcher
            .search(req())
            .unwrap()
            .hits
            .iter()
            .map(|h| h.doc_id)
            .collect();

        assert_eq!(ids1, ids2);
        let stats = searcher.parsed_query_cache_stats();
        assert_eq!(stats.hits, 0, "a disabled cache never hits");
    }
}
