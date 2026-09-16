//! Boolean query implementation for combining multiple queries.

use std::sync::Arc;

use roaring::RoaringTreemap;

use crate::error::Result;
use crate::lexical::index::inverted::per_segment_view::PerSegmentReaderView;
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::matcher::{
    ConjunctionMatcher, ConjunctionNotMatcher, DisjunctionMatcher, EmptyMatcher, Matcher,
    PreComputedMatcher,
};
use crate::lexical::query::scorer::{BM25Scorer, Scorer};
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::{LexicalIndexReader, scan_doc_ids};

/// Occurrence requirements for boolean clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occur {
    /// The clause must match (equivalent to AND).
    Must,
    /// The clause should match (equivalent to OR).
    Should,
    /// The clause must not match (equivalent to NOT).
    MustNot,
    /// The clause must match but does not contribute to scoring.
    /// Used for filtering results without affecting relevance scores.
    Filter,
}

/// A clause in a boolean query.
///
/// `query` is stored as `Arc<dyn Query>` so cloning a clause — and by
/// extension cloning a [`BooleanQuery`] — is a refcount bump rather
/// than a deep `clone_box()` of the entire subtree (#413). External
/// constructors continue to accept `Box<dyn Query>` for backwards
/// compatibility; the box is converted into an `Arc` once at clause
/// construction.
#[derive(Debug, Clone)]
pub struct BooleanClause {
    /// The query for this clause.
    pub query: Arc<dyn Query>,
    /// The occurrence requirement.
    pub occur: Occur,
}

impl BooleanClause {
    /// Create a new boolean clause.
    pub fn new(query: Box<dyn Query>, occur: Occur) -> Self {
        BooleanClause {
            query: Arc::<dyn Query>::from(query),
            occur,
        }
    }

    /// Create a MUST clause.
    pub fn must(query: Box<dyn Query>) -> Self {
        BooleanClause::new(query, Occur::Must)
    }

    /// Create a SHOULD clause.
    pub fn should(query: Box<dyn Query>) -> Self {
        BooleanClause::new(query, Occur::Should)
    }

    /// Create a MUST_NOT clause.
    pub fn must_not(query: Box<dyn Query>) -> Self {
        BooleanClause::new(query, Occur::MustNot)
    }

    /// Create a FILTER clause (matches like Must but does not affect scoring).
    pub fn filter(query: Box<dyn Query>) -> Self {
        BooleanClause::new(query, Occur::Filter)
    }
}

/// A boolean query that combines multiple queries with boolean logic.
#[derive(Debug)]
pub struct BooleanQuery {
    /// The clauses in this boolean query.
    clauses: Vec<BooleanClause>,
    /// The boost factor for this query.
    boost: f32,
    /// Minimum number of should clauses that must match.
    minimum_should_match: usize,
}

impl BooleanQuery {
    /// Create a new empty boolean query.
    pub fn new() -> Self {
        BooleanQuery {
            clauses: Vec::new(),
            boost: 1.0,
            minimum_should_match: 0,
        }
    }

    /// Add a clause to this boolean query.
    pub fn add_clause(&mut self, clause: BooleanClause) {
        self.clauses.push(clause);
    }

    /// Add a MUST clause.
    pub fn add_must(&mut self, query: Box<dyn Query>) {
        self.add_clause(BooleanClause::must(query));
    }

    /// Add a SHOULD clause.
    pub fn add_should(&mut self, query: Box<dyn Query>) {
        self.add_clause(BooleanClause::should(query));
    }

    /// Add a MUST_NOT clause.
    pub fn add_must_not(&mut self, query: Box<dyn Query>) {
        self.add_clause(BooleanClause::must_not(query));
    }

    /// Add a FILTER clause (matches like Must but does not affect scoring).
    pub fn add_filter(&mut self, query: Box<dyn Query>) {
        self.add_clause(BooleanClause::filter(query));
    }

    /// Set the boost factor.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set the minimum number of should clauses that must match.
    pub fn with_minimum_should_match(mut self, minimum: usize) -> Self {
        self.minimum_should_match = minimum;
        self
    }

    /// Get the clauses.
    pub fn clauses(&self) -> &[BooleanClause] {
        &self.clauses
    }

    /// Get the minimum should match value.
    pub fn minimum_should_match(&self) -> usize {
        self.minimum_should_match
    }

    /// Check if this query is empty.
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Get clauses by occurrence type.
    pub fn clauses_by_occur(&self, occur: Occur) -> Vec<&BooleanClause> {
        self.clauses.iter().filter(|c| c.occur == occur).collect()
    }
}

impl Default for BooleanQuery {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BooleanQuery {
    fn clone(&self) -> Self {
        // Cloning is now a refcount bump per clause: `BooleanClause`
        // derives `Clone` and `clause.query` is `Arc<dyn Query>` so
        // there is no deep `clone_box()` walk through the subtree
        // anymore (#413).
        BooleanQuery {
            clauses: self.clauses.clone(),
            boost: self.boost,
            minimum_should_match: self.minimum_should_match,
        }
    }
}

/// Resolve the cached doc-id set for `query` when `reader` exposes a snapshot
/// query/filter cache (#578/#764).
///
/// Reachable on the cross-segment [`InvertedIndexReader`] and on the
/// per-segment fanout [`PerSegmentReaderView`] (which delegates to the
/// cross-segment cache). Doc ids are global in both, so the same cached set is
/// valid in either context. Returns `None` when no cache is reachable, so the
/// caller falls back to the query's normal matcher.
fn cached_filter_doc_ids(
    query: &dyn Query,
    reader: &dyn LexicalIndexReader,
) -> Result<Option<Arc<RoaringTreemap>>> {
    if let Some(inverted) = reader.as_any().downcast_ref::<InvertedIndexReader>() {
        return Ok(Some(inverted.matching_doc_ids(query)?));
    }
    if let Some(view) = reader.as_any().downcast_ref::<PerSegmentReaderView>() {
        return Ok(Some(view.matching_doc_ids(query)?));
    }
    Ok(None)
}

/// Build the matcher for a single boolean clause, reusing the snapshot-scoped
/// filter cache for cacheable `Occur::Filter` clauses (#764).
///
/// A Filter clause selects documents without contributing to scoring, so its
/// matched set can be served from the cache as a `RoaringTreemap` and iterated
/// via a membership-only [`PreComputedMatcher`] instead of re-walking posting
/// lists. Any non-Filter clause, an uncacheable query
/// ([`Query::cache_key`] is `None`), or a reader without a reachable cache
/// falls back to the query's normal matcher — so behaviour is unchanged there.
fn clause_matcher(
    clause: &BooleanClause,
    reader: &dyn LexicalIndexReader,
) -> Result<Box<dyn Matcher>> {
    if clause.occur == Occur::Filter
        && clause.query.cache_key().is_some()
        && let Some(bitmap) = cached_filter_doc_ids(clause.query.as_ref(), reader)?
    {
        let doc_ids: Vec<u64> = bitmap.iter().collect();
        return Ok(Box::new(PreComputedMatcher::new(doc_ids)));
    }
    clause.query.matcher(reader)
}

impl Query for BooleanQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        if self.clauses.is_empty() {
            return Ok(Box::new(EmptyMatcher::new()));
        }

        let must_clauses = self.clauses_by_occur(Occur::Must);
        let filter_clauses = self.clauses_by_occur(Occur::Filter);
        let should_clauses = self.clauses_by_occur(Occur::Should);
        let must_not_clauses = self.clauses_by_occur(Occur::MustNot);

        // Combine Must and Filter clauses for matching (Filter behaves like Must)
        let mut required_clauses: Vec<&BooleanClause> = Vec::new();
        required_clauses.extend(&must_clauses);
        required_clauses.extend(&filter_clauses);

        // Handle MUST/FILTER and MUST_NOT clauses
        if !required_clauses.is_empty()
            || (!must_not_clauses.is_empty() && should_clauses.is_empty())
        {
            // Create positive matcher from MUST and FILTER clauses
            let mut positive_matcher = if !required_clauses.is_empty() {
                if required_clauses.len() == 1 {
                    // Single required clause
                    clause_matcher(required_clauses[0], reader)?
                } else {
                    // Multiple required clauses - use ConjunctionMatcher
                    let mut matchers = Vec::new();
                    for clause in &required_clauses {
                        let matcher = clause_matcher(clause, reader)?;
                        if matcher.is_exhausted() {
                            return Ok(Box::new(EmptyMatcher::new()));
                        }
                        matchers.push(matcher);
                    }
                    Box::new(ConjunctionMatcher::new(matchers))
                }
            } else {
                // No required clauses, but we have MUST_NOT clauses and no
                // SHOULD clauses: match every present, live document and
                // exclude the MUST_NOT matches. The universe is the ids
                // actually present in the reader (segment-bounded under the
                // fanout, correct for sparse id spaces) minus soft-deleted
                // docs — not the dense `0..max_doc()` range, which yielded
                // cross-segment, phantom, and deleted ids (#997).
                let universe: Vec<u64> = scan_doc_ids(reader)?
                    .filter(|&doc_id| !reader.is_deleted(doc_id))
                    .collect();
                Box::new(PreComputedMatcher::new(universe))
            };

            // If minimum_should_match is set and we have SHOULD clauses, combine them with MUST
            if self.minimum_should_match > 0 && !should_clauses.is_empty() {
                let mut should_matchers = Vec::new();
                for clause in &should_clauses {
                    let matcher = clause.query.matcher(reader)?;
                    if !matcher.is_exhausted() {
                        should_matchers.push(matcher);
                    }
                }

                if !should_matchers.is_empty() {
                    let should_matcher = if should_matchers.len() == 1 {
                        should_matchers.into_iter().next().unwrap()
                    } else {
                        Box::new(DisjunctionMatcher::new(should_matchers))
                    };

                    // Combine MUST and SHOULD with ConjunctionMatcher
                    positive_matcher = Box::new(ConjunctionMatcher::new(vec![
                        positive_matcher,
                        should_matcher,
                    ]));
                } else if !required_clauses.is_empty() {
                    // SHOULD clauses are exhausted but minimum_should_match requires them
                    // This means no documents can match
                    return Ok(Box::new(EmptyMatcher::new()));
                }
            }

            // Handle MUST_NOT clauses
            if !must_not_clauses.is_empty() {
                let mut negative_matchers = Vec::new();
                for clause in &must_not_clauses {
                    let matcher = clause.query.matcher(reader)?;
                    if !matcher.is_exhausted() {
                        negative_matchers.push(matcher);
                    }
                }

                if !negative_matchers.is_empty() {
                    // The positive matcher is either the required-clause
                    // conjunction or the present-live universe (MustNot-only
                    // shape); either way `ConjunctionNotMatcher` applies the
                    // exclusions. #997 removed the dense `NotMatcher` path,
                    // which walked `0..max_doc()` and broke multi-segment
                    // MustNot-only results.
                    Ok(Box::new(ConjunctionNotMatcher::new(
                        positive_matcher,
                        negative_matchers,
                    )))
                } else {
                    // All negative matchers are exhausted, just return positive matcher
                    Ok(positive_matcher)
                }
            } else {
                // No MUST_NOT clauses, just return positive matcher
                Ok(positive_matcher)
            }
        } else if !should_clauses.is_empty() {
            // SHOULD clauses (possibly with MUST_NOT)
            let mut should_matchers = Vec::new();
            for clause in &should_clauses {
                let matcher = clause.query.matcher(reader)?;
                if !matcher.is_exhausted() {
                    should_matchers.push(matcher);
                }
            }

            if should_matchers.is_empty() {
                return Ok(Box::new(EmptyMatcher::new()));
            }

            let positive_matcher = if should_matchers.len() == 1 {
                should_matchers.into_iter().next().unwrap()
            } else {
                Box::new(DisjunctionMatcher::new(should_matchers))
            };

            // Handle MUST_NOT clauses with SHOULD
            if !must_not_clauses.is_empty() {
                let mut negative_matchers = Vec::new();
                for clause in &must_not_clauses {
                    let matcher = clause.query.matcher(reader)?;
                    if !matcher.is_exhausted() {
                        negative_matchers.push(matcher);
                    }
                }

                if !negative_matchers.is_empty() {
                    // Combine SHOULD (positive) with MUST_NOT (negative)
                    Ok(Box::new(ConjunctionNotMatcher::new(
                        positive_matcher,
                        negative_matchers,
                    )))
                } else {
                    Ok(positive_matcher)
                }
            } else {
                Ok(positive_matcher)
            }
        } else {
            Ok(Box::new(EmptyMatcher::new()))
        }
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        use crate::lexical::query::scorer::BooleanScorer;

        let mut sub_queries = Vec::new();

        // Collect queries from MUST and SHOULD clauses
        for clause in &self.clauses {
            if clause.occur == Occur::Must || clause.occur == Occur::Should {
                sub_queries.push(clause.query.clone_box());
            }
        }

        if sub_queries.is_empty() {
            // Fallback for empty boolean query
            let scorer = BM25Scorer::new(
                1,
                1,
                reader.doc_count(),
                10.0,
                reader.doc_count(),
                self.boost,
            );
            return Ok(Box::new(scorer));
        }

        let mut boolean_scorer = BooleanScorer::new(reader, sub_queries)?;
        boolean_scorer.set_boost(self.boost);
        Ok(Box::new(boolean_scorer))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        if self.clauses.is_empty() {
            return "()".to_string();
        }

        let mut parts = Vec::new();

        for clause in &self.clauses {
            let clause_desc = match clause.occur {
                Occur::Must => format!("+{}", clause.query.description()),
                Occur::Should => clause.query.description(),
                Occur::MustNot => format!("-{}", clause.query.description()),
                Occur::Filter => format!("#{}", clause.query.description()),
            };
            parts.push(clause_desc);
        }

        let result = format!("({})", parts.join(" "));

        if self.boost == 1.0 {
            result
        } else {
            format!("{}^{}", result, self.boost)
        }
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, reader: &dyn LexicalIndexReader) -> Result<bool> {
        if self.clauses.is_empty() {
            return Ok(true);
        }

        // MustNot clauses only restrict the result set; their own
        // emptiness says nothing about whether the query can match. An
        // exhausted negative excludes NOTHING — the query matches the
        // whole universe, not nothing (#997).
        let mut has_positive = false;
        for clause in &self.clauses {
            if clause.occur == Occur::MustNot {
                continue;
            }
            has_positive = true;
            if !clause.query.is_empty(reader)? {
                return Ok(false);
            }
        }

        if !has_positive {
            // MustNot-only: matches the present-live universe, which is
            // empty only when the reader holds no documents.
            return Ok(reader.doc_count() == 0);
        }

        Ok(true)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        let mut total_cost = 0;

        for clause in &self.clauses {
            total_cost += clause.query.cost(reader)?;
        }

        Ok(total_cost)
    }

    fn rewrite(&self, reader: &dyn LexicalIndexReader) -> Result<Option<Box<dyn Query>>> {
        // Lower multi-term subqueries once at the searcher level
        // (Issue #613). Only scoring clauses (Should / Must) are
        // rewritten: Filter and MustNot clauses are matched through the
        // cached doc-id path keyed by the ORIGINAL clause's
        // `cache_key`, which a rewrite would invalidate.
        let mut rewritten_any = false;
        let mut clauses = Vec::with_capacity(self.clauses.len());
        for clause in &self.clauses {
            let lowered = match clause.occur {
                Occur::Should | Occur::Must => clause.query.rewrite(reader)?,
                Occur::MustNot | Occur::Filter => None,
            };
            match lowered {
                Some(query) => {
                    rewritten_any = true;
                    clauses.push(BooleanClause::new(query, clause.occur));
                }
                None => clauses.push(clause.clone()),
            }
        }
        if !rewritten_any {
            return Ok(None);
        }
        let mut rewritten =
            BooleanQuery::new().with_minimum_should_match(self.minimum_should_match);
        rewritten.set_boost(self.boost);
        for clause in clauses {
            rewritten.add_clause(clause);
        }
        Ok(Some(Box::new(rewritten)))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn collect_field_refs(&self, out: &mut std::collections::HashSet<String>) {
        for clause in &self.clauses {
            clause.query.collect_field_refs(out);
        }
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        // A MustNot clause describes what a hit does not contain.
        for clause in &self.clauses {
            if clause.occur != Occur::MustNot {
                clause.query.collect_highlight_terms(field, out);
            }
        }
    }

    fn apply_field_boosts(&mut self, boosts: &std::collections::HashMap<String, f32>) {
        // Apply overall boost if targeted (BooleanQuery doesn't target a field usually, but we check anyway)
        if let Some(f) = self.field()
            && let Some(&b) = boosts.get(f)
        {
            self.set_boost(self.boost() * b);
        }

        // Recursively apply to all clauses. Sub-queries are stored as
        // `Arc<dyn Query>` (#413) so the tree is shared by default;
        // when a sub-query has other holders we deep-copy it via
        // `clone_box`, mutate the owned copy, and swap the `Arc`.
        // `apply_field_boosts` is invoked at request preparation
        // time, not in the per-doc hot path, so the conditional
        // clone is acceptable.
        for clause in &mut self.clauses {
            if let Some(q) = Arc::get_mut(&mut clause.query) {
                q.apply_field_boosts(boosts);
            } else {
                let mut owned = clause.query.clone_box();
                owned.apply_field_boosts(boosts);
                clause.query = Arc::<dyn Query>::from(owned);
            }
        }
    }

    fn cache_key(&self) -> Option<String> {
        // R1 safety: a boolean with no positive (Must/Should/Filter) clause is
        // realised as `AllMatcher` / `NotMatcher` over the entire
        // `[0, max_doc)` doc-id space (see `matcher` above), which includes
        // deleted and never-assigned ids. That set is not a canonical "live
        // document" set, so refuse to cache it.
        let has_positive = self
            .clauses
            .iter()
            .any(|c| matches!(c.occur, Occur::Must | Occur::Should | Occur::Filter));
        if !has_positive {
            return None;
        }

        // Compose from each clause's own `cache_key`. If any child is
        // uncacheable, the whole boolean is uncacheable — a non-canonical
        // child would make the composite key non-canonical (the `?` below
        // short-circuits to `None`). Clause order is preserved: the matched
        // set is order-independent, so order only affects cache fragmentation,
        // never correctness. `minimum_should_match` affects membership and is
        // included; boost is score-only and excluded. Each child key is
        // `{:?}`-escaped so the concatenation is unambiguous.
        let mut key = format!("bool|msm={}|", self.minimum_should_match);
        for clause in &self.clauses {
            let tag = match clause.occur {
                Occur::Must => 'M',
                Occur::Should => 'S',
                Occur::MustNot => 'N',
                Occur::Filter => 'F',
            };
            let child = clause.query.cache_key()?;
            key.push_str(&format!("{tag}{child:?};"));
        }
        Some(key)
    }
}

/// Builder for creating boolean queries.
#[derive(Debug)]
pub struct BooleanQueryBuilder {
    query: BooleanQuery,
}

impl BooleanQueryBuilder {
    /// Create a new boolean query builder.
    pub fn new() -> Self {
        BooleanQueryBuilder {
            query: BooleanQuery::new(),
        }
    }

    /// Add a MUST clause.
    pub fn must(mut self, query: Box<dyn Query>) -> Self {
        self.query.add_must(query);
        self
    }

    /// Add a SHOULD clause.
    pub fn should(mut self, query: Box<dyn Query>) -> Self {
        self.query.add_should(query);
        self
    }

    /// Add a MUST_NOT clause.
    pub fn must_not(mut self, query: Box<dyn Query>) -> Self {
        self.query.add_must_not(query);
        self
    }

    /// Add a FILTER clause (matches like Must but does not affect scoring).
    pub fn filter(mut self, query: Box<dyn Query>) -> Self {
        self.query.add_filter(query);
        self
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.query = self.query.with_boost(boost);
        self
    }

    /// Set the minimum should match.
    pub fn minimum_should_match(mut self, minimum: usize) -> Self {
        self.query = self.query.with_minimum_should_match(minimum);
        self
    }

    /// Build the boolean query.
    pub fn build(self) -> BooleanQuery {
        self.query
    }
}

impl Default for BooleanQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::inverted::reader::{InvertedIndexReader, InvertedIndexReaderConfig};
    use crate::lexical::query::term::TermQuery;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;
    use std::sync::Arc;

    #[allow(dead_code)]
    #[test]
    fn test_boolean_query_creation() {
        let query = BooleanQuery::new();

        assert!(query.is_empty());
        assert_eq!(query.clauses().len(), 0);
        assert_eq!(query.boost(), 1.0);
        assert_eq!(query.minimum_should_match(), 0);
    }

    #[test]
    fn test_boolean_query_clauses() {
        let mut query = BooleanQuery::new();

        query.add_must(Box::new(TermQuery::new("title", "hello")));
        query.add_should(Box::new(TermQuery::new("body", "world")));
        query.add_must_not(Box::new(TermQuery::new("title", "spam")));

        assert_eq!(query.clauses().len(), 3);
        assert!(!query.is_empty());

        let must_clauses = query.clauses_by_occur(Occur::Must);
        let should_clauses = query.clauses_by_occur(Occur::Should);
        let must_not_clauses = query.clauses_by_occur(Occur::MustNot);

        assert_eq!(must_clauses.len(), 1);
        assert_eq!(should_clauses.len(), 1);
        assert_eq!(must_not_clauses.len(), 1);
    }

    #[test]
    fn test_boolean_query_builder() {
        let query = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("title", "hello")))
            .should(Box::new(TermQuery::new("body", "world")))
            .must_not(Box::new(TermQuery::new("title", "spam")))
            .boost(2.0)
            .minimum_should_match(1)
            .build();

        assert_eq!(query.clauses().len(), 3);
        assert_eq!(query.boost(), 2.0);
        assert_eq!(query.minimum_should_match(), 1);
    }

    #[test]
    fn test_boolean_query_description() {
        let query = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("title", "hello")))
            .should(Box::new(TermQuery::new("body", "world")))
            .must_not(Box::new(TermQuery::new("title", "spam")))
            .build();

        let desc = query.description();
        assert!(desc.contains("+title:hello"));
        assert!(desc.contains("body:world"));
        assert!(desc.contains("-title:spam"));
    }

    #[test]
    fn test_boolean_query_matcher() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let query = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("title", "hello")))
            .build();

        let matcher = query.matcher(&reader).unwrap();
        // Should create a matcher without error
        assert!(matcher.is_exhausted() || matcher.doc_id() != u64::MAX);
    }

    #[test]
    fn test_boolean_query_scorer() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let query = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("title", "hello")))
            .build();

        let scorer = query.scorer(&reader).unwrap();
        // Should create a scorer without error
        assert!(scorer.score(0, 1.0, None) >= 0.0);
    }

    #[test]
    fn test_boolean_clause_creation() {
        let query = Box::new(TermQuery::new("title", "hello"));

        let must_clause = BooleanClause::must(query.clone_box());
        assert_eq!(must_clause.occur, Occur::Must);

        let should_clause = BooleanClause::should(query.clone_box());
        assert_eq!(should_clause.occur, Occur::Should);

        let must_not_clause = BooleanClause::must_not(query.clone_box());
        assert_eq!(must_not_clause.occur, Occur::MustNot);
    }

    #[test]
    fn test_empty_boolean_query() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader =
            InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
                .unwrap();

        let query = BooleanQuery::new();

        assert!(query.is_empty());
        assert_eq!(query.cost(&reader).unwrap(), 0);

        let matcher = query.matcher(&reader).unwrap();
        assert!(matcher.is_exhausted());
    }

    // ----- Issue #578: cache_key composition rules -----

    /// `cache_key` composes children and excludes boost (which scales scores,
    /// not membership); a different child term yields a different key.
    #[test]
    fn cache_key_composes_and_excludes_boost() {
        let base = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("f", "a")))
            .should(Box::new(TermQuery::new("g", "b")))
            .build();
        let key = base
            .cache_key()
            .expect("a boolean over cacheable children is cacheable");

        let mut boosted = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("f", "a")))
            .should(Box::new(TermQuery::new("g", "b")))
            .build();
        boosted.set_boost(3.5);
        assert_eq!(
            boosted.cache_key().unwrap(),
            key,
            "boost must not change the cache key"
        );

        let other = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("f", "z")))
            .should(Box::new(TermQuery::new("g", "b")))
            .build();
        assert_ne!(
            other.cache_key().unwrap(),
            key,
            "a different child term must change the key"
        );
    }

    /// A boolean with no positive (Must/Should/Filter) clause is realised over
    /// the whole doc-id space (R1) and must be uncacheable.
    #[test]
    fn cache_key_none_for_mustnot_only() {
        let q = BooleanQueryBuilder::new()
            .must_not(Box::new(TermQuery::new("f", "a")))
            .build();
        assert!(
            q.cache_key().is_none(),
            "a MustNot-only boolean has no positive clause and must be uncacheable"
        );
    }

    /// An uncacheable child (here a span query, whose key is `None`) must make
    /// the whole boolean uncacheable.
    #[test]
    fn cache_key_none_when_child_uncacheable() {
        use crate::lexical::query::span::{SpanQueryWrapper, SpanTermQuery};

        let span: Box<dyn Query> = Box::new(SpanQueryWrapper::new(Box::new(SpanTermQuery::new(
            "f", "a",
        ))));
        assert!(span.cache_key().is_none(), "span queries are not cacheable");

        let q = BooleanQueryBuilder::new()
            .must(span)
            .should(Box::new(TermQuery::new("g", "b")))
            .build();
        assert!(
            q.cache_key().is_none(),
            "an uncacheable child must make the whole boolean uncacheable"
        );
    }

    /// Reader with a sparse id set: ids exist beyond `max_doc()` (sparse
    /// post-merge id spaces, fanout views reporting the global
    /// `max_doc`) — see #997.
    #[derive(Debug)]
    struct SparseIdReader {
        ids: Vec<u64>,
        max_doc: u64,
    }

    impl crate::lexical::reader::LexicalIndexReader for SparseIdReader {
        fn doc_count(&self) -> u64 {
            self.max_doc
        }
        fn max_doc(&self) -> u64 {
            self.max_doc
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(
            &self,
            _doc_id: u64,
        ) -> crate::error::Result<Option<crate::lexical::core::document::Document>> {
            Ok(None)
        }
        fn doc_ids(&self) -> crate::error::Result<Vec<u64>> {
            Ok(self.ids.clone())
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>>
        {
            Ok(None)
        }
        fn field_stats(
            &self,
            _field: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> crate::error::Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// #997 regression: a MustNot-only query whose negatives all come up
    /// empty must match exactly the docs present in the reader — not the
    /// dense `0..max_doc()` range, which over-scans under the fanout and
    /// cannot reach ids above `max_doc()`.
    #[test]
    fn must_not_only_universe_is_present_docs() {
        let reader = SparseIdReader {
            ids: vec![5, 1500],
            max_doc: 10,
        };
        let query = BooleanQueryBuilder::new()
            .must_not(Box::new(TermQuery::new("body", "absent")))
            .build();

        let mut matcher = query.matcher(&reader).unwrap();
        let mut found = Vec::new();
        while !matcher.is_exhausted() {
            let doc_id = matcher.doc_id();
            if doc_id == u64::MAX {
                break;
            }
            found.push(doc_id);
            if !matcher.next().unwrap() {
                break;
            }
        }

        assert_eq!(
            found,
            vec![5, 1500],
            "universe must be the present ids, not 0..max_doc()"
        );
    }
}
