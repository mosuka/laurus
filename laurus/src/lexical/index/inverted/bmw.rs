//! Block-Max-WAND fast path for Should-only `BooleanQuery` (#475 PR-F).
//!
//! [`BlockMaxOrExecutor`] implements the per-clause Block-Max-WAND
//! pivot loop on top of the per-block bound metadata that landed in
//! [`crate::lexical::query::scorer`] (PR-C / PR-E). The executor
//! drives each clause's matcher independently, picks a *pivot* clause
//! whose prefix-sum of per-block bounds exceeds the collector's
//! current K-th score, and either scores the pivot doc (if all
//! prefix clauses align there) or skips clauses that lag behind
//! the pivot — bypassing every doc in non-competitive blocks.
//!
//! The standard whole-query searcher in
//! [`super::searcher::InvertedIndexSearcher::search_with_collector_parallel`]
//! checks BMW eligibility at the entrypoint and dispatches here for
//! Should-only Boolean queries against a [`TopDocsCollector`]; every
//! other query / collector keeps the existing matcher-driven path.

use crate::error::{LaurusError, Result};
use crate::lexical::index::inverted::reader::InvertedIndexReader;
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
    /// production hot path). Must expose a finite-block bound —
    /// checked at construction time.
    scorer: crate::lexical::query::scorer::LeafScorer,
    /// The per-clause matcher (specialised for PostingMatcher in the
    /// production hot path). Walks its posting list independently of
    /// the other clauses' matchers.
    matcher: crate::lexical::query::matcher::LeafMatcher,
    /// Field name extracted from the clause's underlying
    /// [`TermQuery`] or [`SynonymQuery`], used to look up per-doc
    /// field length at scoring time. `None` when the clause is
    /// neither (in that case the executor falls back to the
    /// scorer's avg field length).
    field_name: Option<String>,
}

/// Block-Max-WAND executor for a Should-only `BooleanQuery`.
pub(crate) struct BlockMaxOrExecutor<'r> {
    clauses: Vec<BmwClause>,
    inverted_reader: Option<&'r InvertedIndexReader>,
}

impl<'r> BlockMaxOrExecutor<'r> {
    /// Build an executor from a Should-only [`BooleanQuery`]. Returns
    /// an error if any clause's scorer does not carry per-block
    /// metadata — the caller should fall back to the standard search
    /// path in that case.
    pub fn new(boolean_query: &BooleanQuery, reader: &'r dyn LexicalIndexReader) -> Result<Self> {
        let mut clauses = Vec::with_capacity(boolean_query.clauses().len());
        for clause in boolean_query.clauses() {
            let scorer = clause.query.scorer(reader)?;
            // Runtime eligibility: every clause must expose per-block
            // metadata. `next_block_boundary(0).is_none()` is the
            // documented contract for "no per-block info".
            if scorer.next_block_boundary(0).is_none() {
                return Err(LaurusError::InvalidOperation(
                    "BMW fast path requires per-block scorer for every clause".into(),
                ));
            }
            let matcher = clause.query.matcher(reader)?;
            let field_name = field_name_of(clause.query.as_ref()).map(str::to_string);
            clauses.push(BmwClause {
                scorer: crate::lexical::query::scorer::LeafScorer::from_box(scorer),
                matcher: crate::lexical::query::matcher::LeafMatcher::from_box(matcher),
                field_name,
            });
        }
        let inverted_reader = reader.as_any().downcast_ref::<InvertedIndexReader>();
        Ok(BlockMaxOrExecutor {
            clauses,
            inverted_reader,
        })
    }

    /// Drive the pivot loop and feed competitive documents into
    /// `collector`. Returns the same collector once exhausted or
    /// short-circuited via `needs_more()`.
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

            // Find pivot k: smallest j such that the prefix sum of
            // current per-block bounds for active[0..=j] strictly
            // exceeds min_comp. If no such j exists the entire current
            // frontier is non-competitive in itself.
            let mut sum = 0.0_f32;
            let mut pivot_k: Option<usize> = None;
            for (j, &i) in active.iter().enumerate() {
                let doc_id = self.clauses[i].matcher.doc_id();
                sum += self.clauses[i].scorer.current_block_max_score(doc_id);
                if sum > min_comp {
                    pivot_k = Some(j);
                    break;
                }
            }

            match pivot_k {
                None => {
                    // No pivot found at the current frontier. Use the
                    // *cumulative* (right_max) bound to decide whether
                    // any later block could still produce a top-K
                    // candidate; if not, we are globally done.
                    let mut cum_sum = 0.0_f32;
                    for &i in &active {
                        let d = self.clauses[i].matcher.doc_id();
                        cum_sum += self.clauses[i].scorer.block_max_score_at(d);
                    }
                    if cum_sum <= min_comp {
                        break;
                    }
                    // Otherwise advance the lead clause past its
                    // current block — that's the smallest doc_id in
                    // the frontier, so doing so is guaranteed to make
                    // progress.
                    let lead = active[0];
                    let lead_doc = self.clauses[lead].matcher.doc_id();
                    self.advance_clause_past_block(lead, lead_doc)?;
                }
                Some(k) => {
                    let pivot_doc = self.clauses[active[k]].matcher.doc_id();
                    let head_doc = self.clauses[active[0]].matcher.doc_id();

                    if head_doc == pivot_doc {
                        // All clauses in active[0..=k] are aligned at
                        // pivot_doc. Score the union of every clause
                        // (active OR not in the prefix) currently
                        // sitting at pivot_doc.
                        let mut total_score = 0.0_f32;
                        for &i in &active {
                            if self.clauses[i].matcher.doc_id() == pivot_doc {
                                let tf = self.clauses[i].matcher.term_freq() as f32;
                                let fl = self.field_length_for(i, pivot_doc);
                                total_score += self.clauses[i].scorer.score(pivot_doc, tf, fl);
                            } else {
                                // active is sorted by doc_id; once we
                                // pass pivot_doc no later clause can
                                // share it.
                                break;
                            }
                        }
                        collector.collect(pivot_doc, total_score)?;
                        if !collector.needs_more() {
                            break;
                        }

                        // Advance every clause that contributed.
                        for &i in active.iter() {
                            if self.clauses[i].matcher.doc_id() == pivot_doc {
                                self.clauses[i].matcher.next()?;
                            } else {
                                break;
                            }
                        }
                    } else {
                        // Some clauses in [0..k] lag behind pivot_doc.
                        // Skip each one forward to pivot_doc — they
                        // either land on it (and contribute next
                        // iteration) or land past it.
                        for &i in &active[..k] {
                            self.clauses[i].matcher.skip_to(pivot_doc)?;
                        }
                    }
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

    /// Skip a single clause past the block containing `at_doc`.
    /// `at_doc` is the matcher's current position; the scorer
    /// resolves the block boundary and `matcher.skip_to` jumps
    /// forward. If the clause has no further blocks it is left
    /// in an exhausted state for the active-list filter to drop.
    fn advance_clause_past_block(&mut self, clause_idx: usize, at_doc: u64) -> Result<()> {
        let target = self.clauses[clause_idx].scorer.next_block_boundary(at_doc);
        match target {
            Some(t) if t == u64::MAX => {
                // Past last block. Force exhaustion.
                self.clauses[clause_idx].matcher.skip_to(u64::MAX)?;
            }
            Some(t) => {
                self.clauses[clause_idx].matcher.skip_to(t)?;
            }
            None => {
                // Construction enforces Some, but be defensive: fall
                // back to a single doc step rather than looping.
                self.clauses[clause_idx].matcher.next()?;
            }
        }
        Ok(())
    }

    /// Per-doc field length for the clause's matcher position.
    /// Returns `None` to let the scorer fall back to its average
    /// field length when the underlying reader cannot satisfy the
    /// query (e.g. non-`InvertedIndexReader` readers, missing field).
    fn field_length_for(&self, clause_idx: usize, doc_id: u64) -> Option<f32> {
        let field = self.clauses[clause_idx].field_name.as_deref()?;
        let reader = self.inverted_reader?;
        reader
            .field_length(doc_id, field)
            .ok()
            .flatten()
            .map(|n| n as f32)
    }
}

/// Inspect a clause's underlying [`Query`] for a [`TermQuery`] or
/// [`SynonymQuery`] so the executor can look up the field's per-doc
/// length at scoring time. Any other clause type is not yet wired in
/// (a `BlockMaxConjunction` follow-up could extend this).
fn field_name_of(query: &dyn Query) -> Option<&str> {
    if let Some(t) = query.as_any().downcast_ref::<TermQuery>() {
        return Some(t.field());
    }
    query
        .as_any()
        .downcast_ref::<SynonymQuery>()
        .map(|s| s.field())
}

/// Cheap eligibility check at the searcher entrypoint: BMW fast
/// path requires a Should-only [`BooleanQuery`] with at least two
/// clauses and `minimum_should_match == 0`. The runtime per-block
/// check happens in [`BlockMaxOrExecutor::new`].
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
    // Every clause must be a shape `field_name_of` knows, so one list of
    // clause types serves both this gate and the per-doc field-length
    // lookup; future work (PhraseQuery / NumericRange) extends that one
    // function. A `SynonymQuery` clause (Issue #1257) carries no block-max
    // table by design — its combined term frequency can exceed any one
    // alternative's per-block bound — so `BlockMaxOrExecutor::new`'s
    // runtime `next_block_boundary` check declines it, and with it the
    // whole `BooleanQuery`: a query mixing a `SynonymQuery` clause with
    // `TermQuery` siblings loses the fast path entirely, not just for that
    // clause. That is correctness-safe (the standard path scores the same)
    // but means a field with a synonym dictionary gives up BMW for its
    // multi-position bare-term queries until a clause can opt into a
    // metadata-free bound.
    if bq
        .clauses()
        .iter()
        .any(|c| field_name_of(c.query.as_ref()).is_none())
    {
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

    /// Issue #1257: a `SynonymQuery` clause is a structurally eligible
    /// shape (it carries a field, like `TermQuery`), but it has no
    /// block-max table by design, so the executor's runtime check must
    /// still decline it — falling back to the standard search path rather
    /// than using an unsound bound.
    #[test]
    fn synonym_clause_is_structurally_eligible_but_declines_at_runtime() {
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
        assert!(
            BlockMaxOrExecutor::new(&query, &reader).is_err(),
            "no per-block metadata for the synonym clause: must not build the executor"
        );
    }
}
