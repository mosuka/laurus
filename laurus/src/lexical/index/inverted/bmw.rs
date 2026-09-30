//! Block-Max-WAND fast path for Should-only `BooleanQuery` (#475 PR-F).
//!
//! [`BlockMaxOrExecutor`] implements the per-clause Block-Max-WAND
//! pivot loop on top of the per-block bound metadata that landed in
//! [`crate::lexical::query::scorer`] (PR-C / PR-E). The executor
//! drives each clause's matcher independently and picks a *pivot*
//! clause whose prefix-sum of suffix bounds exceeds the collector's
//! current K-th score. It then checks the bounds of the blocks that
//! hold the pivot doc: it scores the pivot doc (if all prefix clauses
//! align there), skips lagging clauses to it, or, when those blocks
//! cannot compete, skips past the first of them to end (#1286).
//!
//! The standard whole-query searcher in
//! [`super::searcher::InvertedIndexSearcher::search_with_collector_parallel`]
//! checks BMW eligibility at the entrypoint and dispatches here for
//! Should-only Boolean queries against a [`TopDocsCollector`]; every
//! other query / collector keeps the existing matcher-driven path.

use crate::error::Result;
use crate::lexical::index::inverted::searcher::Deadline;
use crate::lexical::query::Query;
use crate::lexical::query::boolean::{BooleanQuery, Occur};
use crate::lexical::query::collector::Collector;
use crate::lexical::query::synonym::SynonymQuery;
use crate::lexical::query::term::TermQuery;
use crate::lexical::reader::LexicalIndexReader;

/// One clause of a Should-only Boolean OR, paired with its scorer
/// and matcher. The scorer is constructed once at executor-start
/// time so the pivot loop can pull per-block bounds without
/// re-resolving the term.
///
/// The `scorer` / `matcher` fields are concrete-type enums (#466)
/// so the per-doc inner loop dispatches to BM25 / PostingMatcher
/// arms via `match` instead of paying a vtable lookup.
struct BmwClause {
    /// The per-clause scorer (specialised for BM25 / Constant in the
    /// production hot path). One without a block-max table bounds every
    /// block with its `max_score()` (#1283).
    scorer: crate::lexical::query::scorer::LeafScorer,
    /// The per-clause matcher (specialised for PostingMatcher in the
    /// production hot path). Walks its posting list independently of
    /// the other clauses' matchers.
    matcher: crate::lexical::query::matcher::LeafMatcher,
}

/// Block-Max-WAND executor for a Should-only `BooleanQuery`.
pub(crate) struct BlockMaxOrExecutor {
    clauses: Vec<BmwClause>,
}

impl BlockMaxOrExecutor {
    /// Build an executor from a Should-only [`BooleanQuery`].
    ///
    /// A clause whose scorer has no block-max table joins the pivot loop
    /// anyway, with its `max_score()` as the bound of every block (#1283):
    /// its `block_max_score_at` and `current_block_max_score` return
    /// `max_score()`, and its `next_block_boundary` returns `None`, which
    /// `run` treats as a block without end. That covers a `SynonymQuery`
    /// (whose combined term frequency no single alternative's table
    /// bounds), and a `TermQuery` whose table is absent: a term missing
    /// from the index (`max_score()` 0, and its clause starts exhausted),
    /// a legacy v1/v2 dictionary, or an aggregated reader whose table the
    /// #1120 guard dropped. Each such `max_score()` is a sound bound
    /// (`k1 + 1` for BM25's TF component when no tighter factor applies).
    /// Such a query used to leave BMW for the regular path entirely.
    pub fn new(boolean_query: &BooleanQuery, reader: &dyn LexicalIndexReader) -> Result<Self> {
        // The regular path sums the clause scores and then multiplies
        // by the outer BooleanQuery's boost (`BooleanScorer::score`).
        // Since the pivot loop sums per-clause scores/bounds directly
        // (see `run`), folding the outer boost into each clause's own
        // scorer boost here gives the same result:
        // `outer * sum(c_i) == sum(outer * c_i)`.
        let outer_boost = boolean_query.boost();
        let mut clauses = Vec::with_capacity(boolean_query.clauses().len());
        for clause in boolean_query.clauses() {
            let mut scorer = clause.query.scorer(reader)?;
            if outer_boost != 1.0 {
                scorer.set_boost(scorer.boost() * outer_boost);
            }
            let matcher = clause.query.matcher(reader)?;
            clauses.push(BmwClause {
                scorer: crate::lexical::query::scorer::LeafScorer::from_box(scorer),
                matcher: crate::lexical::query::matcher::LeafMatcher::from_box(matcher),
            });
        }
        Ok(BlockMaxOrExecutor { clauses })
    }

    /// Drive the pivot loop and feed competitive documents into
    /// `collector`. Returns the same collector once exhausted or
    /// short-circuited via `needs_more()`.
    ///
    /// Every skip stays within the documents its bounds cover (#1286):
    ///
    /// 1. The pivot is the first clause, in document order, at which the
    ///    clauses' suffix bounds (`block_max_score_at`, valid for every
    ///    document from a clause's current one on) sum past the threshold.
    ///    A document before the pivot's can only match the clauses in front
    ///    of it, whose suffix bounds do not pass the threshold, so every
    ///    clause may skip to the pivot document.
    /// 2. The bounds of the blocks holding the pivot document
    ///    (`current_block_max_score`) refine that. When they do not pass the
    ///    threshold either, no document can until the first of those blocks
    ///    ends or the next clause starts, so the clauses skip there.
    ///
    /// `deadline` (Issue #600) is consulted once per
    /// [`crate::lexical::index::inverted::searcher::DEADLINE_CHECK_INTERVAL`]
    /// pivot iterations so a timed search aborts this loop mid-flight rather
    /// than running it to completion.
    pub fn run<C: Collector>(mut self, mut collector: C, deadline: Option<Deadline>) -> Result<C> {
        // Active clauses (still iterating). Indexed into `self.clauses`.
        let mut active: Vec<usize> = Vec::with_capacity(self.clauses.len());
        for (i, c) in self.clauses.iter().enumerate() {
            if !c.matcher.is_exhausted() && c.matcher.doc_id() != u64::MAX {
                active.push(i);
            }
        }

        let mut scanned: u64 = 0;
        loop {
            if let Some(d) = deadline {
                d.check(scanned)?;
            }
            scanned = scanned.wrapping_add(1);

            if active.is_empty() {
                break;
            }

            // Sort active clauses ascending by current matcher.doc_id.
            // The pivot prefix-sum scan below depends on this ordering.
            active.sort_by_key(|&i| self.clauses[i].matcher.doc_id());

            let min_comp = collector.min_competitive();

            // 1. The pivot: the first clause at which the suffix bounds sum
            //    past the threshold. Without one, no document from here on
            //    can make the top-K.
            let mut sum = 0.0_f32;
            let mut pivot: Option<usize> = None;
            for (j, &i) in active.iter().enumerate() {
                let doc_id = self.clauses[i].matcher.doc_id();
                sum += self.clauses[i].scorer.block_max_score_at(doc_id);
                if sum > min_comp {
                    pivot = Some(j);
                    break;
                }
            }
            let Some(mut last) = pivot else {
                break;
            };
            // The pivot side takes every clause sitting on the pivot document.
            let pivot_doc = self.clauses[active[last]].matcher.doc_id();
            while last + 1 < active.len()
                && self.clauses[active[last + 1]].matcher.doc_id() == pivot_doc
            {
                last += 1;
            }
            let pivot_side = &active[..=last];

            // 2. Refine with the blocks holding the pivot document.
            let block_sum: f32 = pivot_side
                .iter()
                .map(|&i| self.clauses[i].scorer.current_block_max_score(pivot_doc))
                .sum();
            if block_sum > min_comp {
                if self.clauses[active[0]].matcher.doc_id() == pivot_doc {
                    // Every pivot-side clause sits on the pivot document.
                    let mut total_score = 0.0_f32;
                    for &i in pivot_side {
                        let tf = self.clauses[i].matcher.term_freq() as f32;
                        // Each leaf looks up its own document's field
                        // length (#1287).
                        total_score += self.clauses[i].scorer.score(pivot_doc, tf, None);
                    }
                    collector.collect(pivot_doc, total_score)?;
                    if !collector.needs_more() {
                        break;
                    }
                    for &i in pivot_side {
                        self.clauses[i].matcher.next()?;
                    }
                } else {
                    // Bring the clauses lagging behind to the pivot document;
                    // they land on it or past it.
                    for &i in pivot_side {
                        if self.clauses[i].matcher.doc_id() < pivot_doc {
                            self.clauses[i].matcher.skip_to(pivot_doc)?;
                        }
                    }
                }
            } else {
                // No document can pass the threshold before the first of the
                // pivot-side blocks ends, or before the next clause starts. A
                // clause without per-block bounds (`None`) bounds every
                // document alike, so it does not end the range.
                let mut next = active
                    .get(last + 1)
                    .map_or(u64::MAX, |&i| self.clauses[i].matcher.doc_id());
                for &i in pivot_side {
                    if let Some(boundary) = self.clauses[i].scorer.next_block_boundary(pivot_doc) {
                        next = next.min(boundary);
                    }
                }
                debug_assert!(next > pivot_doc, "a skip must move past the pivot");
                for &i in pivot_side {
                    self.clauses[i].matcher.skip_to(next)?;
                }
            }

            // Drop any matcher that exhausted during this iteration.
            active.retain(|&i| {
                !self.clauses[i].matcher.is_exhausted()
                    && self.clauses[i].matcher.doc_id() != u64::MAX
            });
        }

        Ok(collector)
    }
}

/// Whether `query` is a leaf the executor supports: a [`TermQuery`] or a
/// [`SynonymQuery`], whose BM25 scorer bounds its own scores. Any other
/// clause type is not yet wired in (a `BlockMaxConjunction` follow-up could
/// extend this).
fn is_bmw_leaf(query: &dyn Query) -> bool {
    let any = query.as_any();
    any.is::<TermQuery>() || any.is::<SynonymQuery>()
}

/// Cheap eligibility check at the searcher entrypoint: BMW fast
/// path requires a Should-only [`BooleanQuery`] with at least two
/// clauses and `minimum_should_match == 0`, every clause a leaf
/// [`is_bmw_leaf`] knows.
pub(crate) fn is_bmw_eligible(query: &dyn Query) -> Option<&BooleanQuery> {
    let bq = query.as_any().downcast_ref::<BooleanQuery>()?;
    if bq.minimum_should_match() > 0 {
        return None;
    }
    if bq.clauses().len() < 2 {
        return None;
    }
    if bq
        .clauses()
        .iter()
        .any(|c| !matches!(c.occur, Occur::Should))
    {
        return None;
    }
    // Future work (PhraseQuery / NumericRange) extends `is_bmw_leaf`. A
    // leaf without a block-max table, such as a `SynonymQuery` (#1257),
    // still qualifies: it runs with a constant bound (see
    // `BlockMaxOrExecutor::new`).
    if !bq.clauses().iter().all(|c| is_bmw_leaf(c.query.as_ref())) {
        return None;
    }
    Some(bq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::query::boolean::BooleanQueryBuilder;
    use crate::lexical::query::term::TermQuery;

    /// Eligibility: must reject must / must_not, single-clause,
    /// and minimum_should_match > 0. The end-to-end top-K equivalence
    /// vs the existing matcher-driven path is covered by the
    /// integration test in [`super::super::searcher::tests`] using a
    /// real `InvertedIndexReader`.
    #[test]
    fn eligibility_rejects_non_should_only_or_thin_queries() {
        let single = BooleanQueryBuilder::new()
            .should(Box::new(TermQuery::new("text", "x")))
            .build();
        assert!(is_bmw_eligible(&single).is_none());

        let mixed = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("text", "x")))
            .should(Box::new(TermQuery::new("text", "y")))
            .build();
        assert!(is_bmw_eligible(&mixed).is_none());

        let msm = BooleanQueryBuilder::new()
            .should(Box::new(TermQuery::new("text", "x")))
            .should(Box::new(TermQuery::new("text", "y")))
            .minimum_should_match(2)
            .build();
        assert!(is_bmw_eligible(&msm).is_none());

        let ok = BooleanQueryBuilder::new()
            .should(Box::new(TermQuery::new("text", "x")))
            .should(Box::new(TermQuery::new("text", "y")))
            .build();
        assert!(is_bmw_eligible(&ok).is_some());
    }

    /// A `SynonymQuery` clause (#1257) is eligible, and although it has no
    /// block-max table by design, it joins the executor with a constant
    /// bound (#1283) rather than sending the query to the standard path.
    /// `searcher::tests::bmw_runs_clauses_without_block_max_metadata`
    /// checks the results on a real index.
    #[test]
    fn synonym_clause_joins_the_executor() {
        use crate::lexical::index::inverted::reader::InvertedIndexReader;
        use crate::lexical::query::synonym::SynonymQuery;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
        use std::sync::Arc;

        let query = BooleanQueryBuilder::new()
            .should(Box::new(TermQuery::new("text", "x")))
            .should(Box::new(SynonymQuery::new(
                "text",
                vec!["y".to_string(), "z".to_string()],
            )))
            .build();
        assert!(is_bmw_eligible(&query).is_some());

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader = InvertedIndexReader::new(
            vec![],
            storage,
            crate::lexical::index::inverted::reader::InvertedIndexReaderConfig::default(),
        )
        .unwrap();
        assert!(BlockMaxOrExecutor::new(&query, &reader).is_ok());
    }
}
