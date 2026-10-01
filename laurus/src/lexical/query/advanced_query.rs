//! Advanced query system with complex query composition and optimization.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::boolean::{BooleanClause, BooleanQuery, Occur};
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::scorer::Scorer;
use crate::lexical::query::{HighlightTerm, Query, QueryResult};
use crate::lexical::reader::LexicalIndexReader;

/// Configuration for [`AdvancedQuery::execute`].
///
/// A search does not read it: it takes its score threshold and time budget
/// from the request
/// ([`LexicalSearchRequest::min_score`](crate::lexical::search::searcher::LexicalSearchRequest::min_score),
/// [`LexicalSearchRequest::timeout_ms`](crate::lexical::search::searcher::LexicalSearchRequest::timeout_ms)).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedQueryConfig {
    /// Enable [`AdvancedQuery::optimize`]. Neither `execute` nor a search
    /// depends on it.
    pub enable_optimization: bool,

    /// Maximum number of clauses to allow in boolean queries.
    pub max_clause_count: usize,

    /// Enable query caching.
    pub enable_caching: bool,

    /// Query timeout in milliseconds.
    pub timeout_ms: u64,

    /// Enable early termination for expensive queries.
    pub enable_early_termination: bool,

    /// Minimum score threshold for results.
    pub min_score: f32,
}

impl Default for AdvancedQueryConfig {
    fn default() -> Self {
        AdvancedQueryConfig {
            enable_optimization: true,
            max_clause_count: 1024,
            enable_caching: true,
            timeout_ms: 30000, // 30 seconds
            enable_early_termination: true,
            min_score: 0.0,
        }
    }
}

/// Advanced query with complex composition capabilities.
///
/// It means the equivalent [`BooleanQuery`]: the core query as a `Must`
/// clause, the filters and post filters as `Filter` clauses and the negative
/// filters as `MustNot` clauses, scaled by the boost. With no filters and a
/// boost of 1.0 it is the core query itself. A search runs that query, and so
/// does [`AdvancedQuery::execute`].
///
/// The minimum score and the [`AdvancedQueryConfig`] apply only in
/// `execute`. A search takes its score threshold from
/// [`LexicalSearchRequest::min_score`](crate::lexical::search::searcher::LexicalSearchRequest::min_score)
/// instead.
#[derive(Debug)]
pub struct AdvancedQuery {
    /// The core query.
    core_query: Box<dyn Query>,

    /// Field boosts for scoring.
    field_boosts: HashMap<String, f32>,

    /// Query-level boost factor.
    boost: f32,

    /// Minimum score threshold.
    min_score: f32,

    /// Filters to apply (must match).
    filters: Vec<Box<dyn Query>>,

    /// Negative filters (must not match).
    negative_filters: Vec<Box<dyn Query>>,

    /// Post filters (applied after scoring).
    post_filters: Vec<Box<dyn Query>>,

    /// Query configuration.
    config: AdvancedQueryConfig,
}

impl AdvancedQuery {
    /// Create a new advanced query.
    pub fn new(core_query: Box<dyn Query>) -> Self {
        AdvancedQuery {
            core_query,
            field_boosts: HashMap::new(),
            boost: 1.0,
            min_score: 0.0,
            filters: Vec::new(),
            negative_filters: Vec::new(),
            post_filters: Vec::new(),
            config: AdvancedQueryConfig::default(),
        }
    }

    /// Set field boost for scoring.
    pub fn add_field_boost(mut self, field: String, boost: f32) -> Self {
        self.field_boosts.insert(field, boost);
        self
    }

    /// Set query-level boost.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set minimum score threshold.
    ///
    /// It applies only in [`AdvancedQuery::execute`]: no `BooleanQuery`
    /// expresses it, so a search filters by
    /// [`LexicalSearchRequest::min_score`](crate::lexical::search::searcher::LexicalSearchRequest::min_score)
    /// instead.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Add a filter (must match).
    pub fn with_filter(mut self, filter: Box<dyn Query>) -> Self {
        self.filters.push(filter);
        self
    }

    /// Add a negative filter (must not match).
    pub fn with_negative_filter(mut self, filter: Box<dyn Query>) -> Self {
        self.negative_filters.push(filter);
        self
    }

    /// Add a post filter.
    pub fn with_post_filter(mut self, filter: Box<dyn Query>) -> Self {
        self.post_filters.push(filter);
        self
    }

    /// Set configuration. It applies only in [`AdvancedQuery::execute`].
    pub fn with_config(mut self, config: AdvancedQueryConfig) -> Self {
        self.config = config;
        self
    }

    /// Optimize the query for better performance.
    ///
    /// Folds the filters and negative filters into the core query. Neither
    /// [`AdvancedQuery::execute`] nor a search needs this: both already run
    /// the equivalent `BooleanQuery`.
    pub fn optimize(&mut self) -> Result<()> {
        if !self.config.enable_optimization {
            return Ok(());
        }

        // Combine filters into boolean query for efficiency
        if !self.filters.is_empty() || !self.negative_filters.is_empty() {
            let mut boolean_builder = BooleanQueryBuilder::new();

            // Add core query as must clause
            boolean_builder = boolean_builder.add_clause(self.core_query.clone_box(), Occur::Must);

            // Add filters as filter clauses (match without affecting score)
            for filter in &self.filters {
                boolean_builder = boolean_builder.add_clause(filter.clone_box(), Occur::Filter);
            }

            // Add negative filters as must_not clauses
            for neg_filter in &self.negative_filters {
                boolean_builder =
                    boolean_builder.add_clause(neg_filter.clone_box(), Occur::MustNot);
            }

            // Replace core query with optimized boolean query
            self.core_query = Box::new(boolean_builder.build());
            self.filters.clear();
            self.negative_filters.clear();
        }

        Ok(())
    }

    /// Execute the advanced query.
    ///
    /// Runs the same query a search does, then applies what no
    /// `BooleanQuery` expresses: the minimum score and the
    /// [`AdvancedQueryConfig`]'s timeout and early termination. Results are
    /// sorted by descending score.
    pub fn execute(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<QueryResult>> {
        // Build the matcher and scorer in one pass (#999).
        let (mut matcher, scorer) = self.lowered().matcher_scorer(reader)?;
        let min_score = self.min_score.max(self.config.min_score);

        let mut results = Vec::new();
        let start_time = crate::util::time::Timer::now();

        // Drain the matcher. It is positioned on its first match at
        // construction, so read the current doc before advancing (the
        // previous `while matcher.next()` loop dropped the first hit),
        // and advance exactly once per iteration so score rejections
        // never skip the advance.
        while !matcher.is_exhausted() {
            let doc_id = matcher.doc_id();
            if doc_id == u64::MAX {
                break;
            }

            // Check timeout
            if self.config.timeout_ms > 0 && start_time.elapsed_ms() > self.config.timeout_ms {
                break;
            }

            let score = scorer.score(doc_id, matcher.term_freq() as f32, None);
            if score >= min_score {
                results.push(QueryResult { doc_id, score });

                // Early termination check
                if self.config.enable_early_termination && results.len() > 10000 {
                    break;
                }
            }

            if !matcher.next()? {
                break;
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.total_cmp(&a.score));

        Ok(results)
    }

    /// The query this one means (see [`AdvancedQuery`]).
    ///
    /// Post filters become `Filter` clauses: they differ from filters only
    /// in being checked after scoring, which changes no hit and no score.
    fn lowered(&self) -> Box<dyn Query> {
        if self.filters.is_empty()
            && self.negative_filters.is_empty()
            && self.post_filters.is_empty()
            && self.boost == 1.0
        {
            return self.core_query.clone_box();
        }
        let mut query = BooleanQuery::new().with_boost(self.boost);
        query.add_must(self.core_query.clone_box());
        for filter in self.filters.iter().chain(&self.post_filters) {
            query.add_filter(filter.clone_box());
        }
        for filter in &self.negative_filters {
            query.add_must_not(filter.clone_box());
        }
        Box::new(query)
    }
}

impl Query for AdvancedQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        self.lowered().matcher(reader)
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        self.lowered().scorer(reader)
    }

    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        self.lowered().matcher_scorer(reader)
    }

    fn rewrite(&self, reader: &dyn LexicalIndexReader) -> Result<Option<Box<dyn Query>>> {
        // Always replaced, so the per-segment fanout's second rewrite sees
        // the lowered query, never this wrapper again.
        let lowered = self.lowered();
        Ok(Some(lowered.rewrite(reader)?.unwrap_or(lowered)))
    }

    fn cache_key(&self) -> Option<String> {
        self.lowered().cache_key()
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        format!(
            "AdvancedQuery(core: {}, boost: {})",
            self.core_query.description(),
            self.boost
        )
    }

    fn is_empty(&self, reader: &dyn LexicalIndexReader) -> Result<bool> {
        self.core_query.is_empty(reader)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        let base_cost = self.core_query.cost(reader)?;
        // Saturating, as in `BooleanQuery::cost` (Issue #1224).
        let filter_cost = self
            .filters
            .iter()
            .map(|f| f.cost(reader))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(0u64, u64::saturating_add);
        Ok(base_cost.saturating_add(filter_cost))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn collect_field_refs(&self, out: &mut std::collections::HashSet<String>) {
        self.core_query.collect_field_refs(out);
        for filter in &self.filters {
            filter.collect_field_refs(out);
        }
        for filter in &self.negative_filters {
            filter.collect_field_refs(out);
        }
        for filter in &self.post_filters {
            filter.collect_field_refs(out);
        }
    }

    fn collect_positional_field_refs(&self, out: &mut std::collections::HashSet<String>) {
        self.core_query.collect_positional_field_refs(out);
        for filter in self
            .filters
            .iter()
            .chain(&self.negative_filters)
            .chain(&self.post_filters)
        {
            filter.collect_positional_field_refs(out);
        }
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        // Negative filters describe what a hit does not contain.
        self.core_query.collect_highlight_terms(field, out);
        for filter in self.filters.iter().chain(&self.post_filters) {
            filter.collect_highlight_terms(field, out);
        }
    }

    fn apply_field_boosts(&mut self, boosts: &HashMap<String, f32>) {
        // Apply field-level boosts from AdvanceQuery's own field_boosts first
        if !self.field_boosts.is_empty() {
            self.core_query.apply_field_boosts(&self.field_boosts);
        }

        // Then apply external boosts
        self.core_query.apply_field_boosts(boosts);

        for filter in &mut self.filters {
            filter.apply_field_boosts(boosts);
        }
        for filter in &mut self.negative_filters {
            filter.apply_field_boosts(boosts);
        }
        for filter in &mut self.post_filters {
            filter.apply_field_boosts(boosts);
        }
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }
}

impl Clone for AdvancedQuery {
    fn clone(&self) -> Self {
        AdvancedQuery {
            core_query: self.core_query.clone_box(),
            field_boosts: self.field_boosts.clone(),
            boost: self.boost,
            min_score: self.min_score,
            filters: self.filters.iter().map(|f| f.clone_box()).collect(),
            negative_filters: self
                .negative_filters
                .iter()
                .map(|f| f.clone_box())
                .collect(),
            post_filters: self.post_filters.iter().map(|f| f.clone_box()).collect(),
            config: self.config.clone(),
        }
    }
}

/// Builder for complex boolean queries with advanced features.
#[derive(Debug)]
pub struct BooleanQueryBuilder {
    /// Query clauses.
    clauses: Vec<(Box<dyn Query>, Occur)>,

    /// Minimum number of should clauses that must match.
    minimum_should_match: usize,

    /// Query boost.
    boost: f32,

    /// Configuration.
    config: AdvancedQueryConfig,
}

impl BooleanQueryBuilder {
    /// Create a new boolean query builder.
    pub fn new() -> Self {
        BooleanQueryBuilder {
            clauses: Vec::new(),
            minimum_should_match: 0,
            boost: 1.0,
            config: AdvancedQueryConfig::default(),
        }
    }

    /// Add a query clause.
    pub fn add_clause(mut self, query: Box<dyn Query>, occur: Occur) -> Self {
        self.clauses.push((query, occur));
        self
    }

    /// Set minimum should match.
    pub fn minimum_should_match(mut self, count: usize) -> Self {
        self.minimum_should_match = count;
        self
    }

    /// Set boost.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set configuration.
    pub fn config(mut self, config: AdvancedQueryConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the boolean query.
    pub fn build(self) -> BooleanQuery {
        let mut boolean_query = BooleanQuery::new();

        for (query, occur) in self.clauses {
            match occur {
                Occur::Must => boolean_query.add_must(query),
                Occur::Should => boolean_query.add_should(query),
                Occur::MustNot => boolean_query.add_must_not(query),
                Occur::Filter => boolean_query.add_filter(query),
            }
        }

        if self.minimum_should_match > 0 {
            boolean_query = boolean_query.with_minimum_should_match(self.minimum_should_match);
        }

        boolean_query.with_boost(self.boost)
    }
}

impl Default for BooleanQueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-field query that searches across multiple fields.
#[derive(Debug, Clone)]
pub struct MultiFieldQuery {
    /// Query text.
    query_text: String,

    /// Fields to search with their boosts.
    fields: HashMap<String, f32>,

    /// Query type for each field.
    query_type: MultiFieldQueryType,

    /// Weight given to every matching field but the best one, when
    /// `query_type` is [`MultiFieldQueryType::BestFields`]. Expected to be
    /// in `[0, 1]`; `0.0` (the default) keeps only the best field's score,
    /// `1.0` sums every matching field's score. Other query types ignore it.
    tie_breaker: f32,

    /// Query-level boost, applied to whichever scorer this query runs as
    /// (the `BooleanQuery` for `MostFields`, the `DisjunctionMaxScorer` for
    /// `BestFields`, or the `CombinedFieldsScorer` for `CombinedFields`).
    boost: f32,

    /// The blended document frequency of a
    /// [`MultiFieldQueryType::CombinedFields`] query, frozen by
    /// [`rewrite`](Query::rewrite) against the top-level reader so every
    /// segment of a multi-segment fanout scores against the same value
    /// instead of blending only the fields its own segment holds the term
    /// in.
    ///
    /// Frozen statistics belong to the reader snapshot they were taken
    /// from, and `rewrite` never refreshes them, so a rewritten query must
    /// not be cached and reused across a `commit` (as with `SynonymQuery`).
    doc_freq: Option<u64>,
}

/// Type of multi-field query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiFieldQueryType {
    /// Dis-max: the best matching field's score, plus [`MultiFieldQuery::tie_breaker`]
    /// times the sum of the other matching fields' scores. Matches like an
    /// OR across fields (Elasticsearch's `multi_match` `best_fields`).
    BestFields,

    /// Every field must match (AND across fields); the score is the sum of
    /// the matching fields' scores (Elasticsearch's `multi_match` `most_fields`).
    MostFields,

    /// BM25F: the fields are scored as one concatenated field, so a document
    /// scores the same whichever of them its occurrences of the term fall
    /// in (Issue #1319). Matches like an OR across fields (Elasticsearch's
    /// `combined_fields`, Lucene's `CombinedFieldQuery`).
    ///
    /// A document's term frequency and length are `Σ w·tf` and `Σ w·len`
    /// over the fields, with `w` the field's boost; the document frequency
    /// is the largest of the fields', and the average length
    /// `Σ w·avg_len·doc_count / max(doc_count)`. A boost is therefore a
    /// BM25F weight — `2.0` counts the field as if it were indexed twice —
    /// rather than a multiplier on the field's score. Every boost must be
    /// finite and greater than zero; searching with any other returns
    /// [`LaurusError::InvalidArgument`](crate::error::LaurusError::InvalidArgument).
    CombinedFields,
}

impl MultiFieldQuery {
    /// Create a new multi-field query.
    pub fn new(query_text: String) -> Self {
        MultiFieldQuery {
            query_text,
            fields: HashMap::new(),
            query_type: MultiFieldQueryType::BestFields,
            tie_breaker: 0.0,
            boost: 1.0,
            doc_freq: None,
        }
    }

    /// Add a field with boost (a BM25F weight for
    /// [`MultiFieldQueryType::CombinedFields`]).
    pub fn add_field(mut self, field: String, boost: f32) -> Self {
        self.fields.insert(field, boost);
        // The blended document frequency depends on the field set.
        self.doc_freq = None;
        self
    }

    /// Set query type.
    pub fn query_type(mut self, query_type: MultiFieldQueryType) -> Self {
        self.query_type = query_type;
        self
    }

    /// Set tie breaker for best fields queries.
    pub fn tie_breaker(mut self, tie_breaker: f32) -> Self {
        self.tie_breaker = tie_breaker;
        self
    }

    /// The configured fields and their boosts, ordered by field name so the
    /// combined sums do not depend on `HashMap` iteration order.
    fn sorted_fields(&self) -> Vec<(&str, f32)> {
        let mut fields: Vec<(&str, f32)> = self
            .fields
            .iter()
            .map(|(field, &boost)| (field.as_str(), boost))
            .collect();
        fields.sort_unstable_by(|a, b| a.0.cmp(b.0));
        fields
    }

    /// Reject a [`MultiFieldQueryType::CombinedFields`] boost that is not a
    /// finite number greater than zero: a BM25F weight scales lengths, so a
    /// zero or negative one could make the combined average length zero or
    /// negative.
    fn check_weights(&self) -> Result<()> {
        if let Some((field, boost)) = self
            .sorted_fields()
            .into_iter()
            .find(|&(_, boost)| !(boost.is_finite() && boost > 0.0))
        {
            return Err(LaurusError::invalid_argument(format!(
                "MultiFieldQuery CombinedFields boost for field {field:?} must be finite and greater than 0, got {boost}"
            )));
        }
        Ok(())
    }

    /// The document frequency [`MultiFieldQueryType::CombinedFields`]
    /// scores with: the largest of the fields', as Lucene's
    /// `CombinedFieldQuery` blends it (at least that many documents hold
    /// the term).
    fn combined_doc_freq(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        let mut doc_freq = 0;
        for field in self.fields.keys() {
            if let Some(info) = reader.term_info(field, &self.query_text)? {
                doc_freq = doc_freq.max(info.doc_freq);
            }
        }
        Ok(doc_freq)
    }

    /// One `TermQuery` per configured field, carrying that field's boost
    /// (Issue #1317).
    fn field_queries(&self) -> Vec<Box<dyn Query>> {
        self.fields
            .iter()
            .map(|(field, &boost)| {
                Box::new(
                    crate::lexical::query::term::TermQuery::new(
                        field.clone(),
                        self.query_text.clone(),
                    )
                    .with_boost(boost),
                ) as Box<dyn Query>
            })
            .collect()
    }

    /// The `BooleanQuery` this query runs as: one `TermQuery` per field,
    /// `Must` for [`MultiFieldQueryType::MostFields`] and `Should`
    /// otherwise, carrying this query's boost. Used for matching by every
    /// query type, and for scoring [`MultiFieldQueryType::MostFields`]
    /// (summed, like `BooleanScorer`); [`MultiFieldQueryType::BestFields`]
    /// and [`MultiFieldQueryType::CombinedFields`] score through their own
    /// scorers instead (see `scorer`).
    fn boolean_query(&self) -> BooleanQuery {
        let occur = match self.query_type {
            MultiFieldQueryType::MostFields => Occur::Must,
            MultiFieldQueryType::BestFields | MultiFieldQueryType::CombinedFields => Occur::Should,
        };
        let mut query = BooleanQuery::new().with_boost(self.boost);
        for field_query in self.field_queries() {
            query.add_clause(BooleanClause::new(field_query, occur));
        }
        query
    }
}

impl Query for MultiFieldQuery {
    /// Matching is independent of how the query types score: `Should`
    /// (union) or `Must` (intersection) over the same per-field
    /// `TermQuery`s `scorer` scores. A `CombinedFields` query with an
    /// invalid boost is rejected here too, so it fails as a filter clause
    /// as well as a scoring one.
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        if matches!(self.query_type, MultiFieldQueryType::CombinedFields) {
            self.check_weights()?;
        }
        self.boolean_query().matcher(reader)
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        match self.query_type {
            MultiFieldQueryType::BestFields => Ok(Box::new(
                crate::lexical::query::scorer::DisjunctionMaxScorer::new(
                    reader,
                    self.field_queries(),
                    self.tie_breaker,
                    self.boost,
                )?,
            )),
            MultiFieldQueryType::MostFields => self.boolean_query().scorer(reader),
            MultiFieldQueryType::CombinedFields => {
                self.check_weights()?;
                let doc_freq = match self.doc_freq {
                    Some(doc_freq) => doc_freq,
                    None => self.combined_doc_freq(reader)?,
                };
                Ok(Box::new(
                    crate::lexical::query::scorer::CombinedFieldsScorer::new(
                        reader,
                        &self.query_text,
                        &self.sorted_fields(),
                        doc_freq,
                        self.boost,
                    )?,
                ))
            }
        }
    }

    /// Freezes a `CombinedFields` query's blended document frequency against
    /// the top-level reader (see `doc_freq`); every other query type, and an
    /// already-frozen one, is left as is.
    ///
    /// Only ever rewrites into another `MultiFieldQuery`: unlike
    /// `AdvancedQuery` (#1316), replacing this query with its
    /// `boolean_query()` would route it into the parallel path's
    /// `downcast::<BooleanQuery>()` or the BMW path, both of which discard
    /// the `DisjunctionMaxScorer` / `CombinedFieldsScorer` and silently fall
    /// back to the summed `BooleanScorer` instead.
    fn rewrite(&self, reader: &dyn LexicalIndexReader) -> Result<Option<Box<dyn Query>>> {
        if !matches!(self.query_type, MultiFieldQueryType::CombinedFields)
            || self.doc_freq.is_some()
        {
            return Ok(None);
        }
        // A per-segment fanout view reports no `term_info` for a field its
        // own segment lacks the term in, so only the top-level reader can
        // blend the document frequency every segment should share.
        if reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .is_none()
        {
            return Ok(None);
        }
        let mut rewritten = self.clone();
        rewritten.doc_freq = Some(self.combined_doc_freq(reader)?);
        Ok(Some(Box::new(rewritten)))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn apply_field_boosts(&mut self, boosts: &HashMap<String, f32>) {
        for (f, &b) in boosts {
            if let Some(field_boost) = self.fields.get_mut(f) {
                *field_boost *= b;
            }
        }
    }

    fn description(&self) -> String {
        format!(
            "MultiFieldQuery(text: {}, type: {:?}, fields: {:?})",
            self.query_text,
            self.query_type,
            self.sorted_fields()
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.query_text.is_empty() || self.fields.is_empty())
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        // Estimate cost based on number of fields
        Ok(self.fields.len() as u64 * 100)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        // The matcher expands to one unanalysed `TermQuery` per configured
        // field, so the raw text is the exact term.
        if !self.query_text.is_empty() && field.is_none_or(|f| self.fields.contains_key(f)) {
            out.push(HighlightTerm::Exact(self.query_text.clone()));
        }
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use super::*;
    use crate::Document;
    use crate::lexical::index::inverted::reader::InvertedIndexReader;
    use crate::lexical::query::term::TermQuery;
    use crate::lexical::query::{
        FuzzyQuery, PrefixQuery, RegexpQuery, SynonymQuery, WildcardQuery,
    };
    use crate::lexical::search::searcher::LexicalSearchRequest;
    use crate::lexical::store::LexicalStore;
    use crate::lexical::store::config::LexicalIndexConfig;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    #[allow(dead_code)]
    #[test]
    fn test_advanced_query_creation() {
        let core_query = Box::new(TermQuery::new("title".to_string(), "test".to_string()));
        let advanced_query = AdvancedQuery::new(core_query)
            .with_boost(2.0)
            .with_min_score(0.5)
            .add_field_boost("title".to_string(), 1.5);

        assert_eq!(advanced_query.boost, 2.0);
        assert_eq!(advanced_query.min_score, 0.5);
        assert_eq!(advanced_query.field_boosts.get("title"), Some(&1.5));
    }

    #[test]
    fn test_boolean_query_builder() {
        let builder = BooleanQueryBuilder::new()
            .minimum_should_match(2)
            .boost(1.5);

        assert_eq!(builder.minimum_should_match, 2);
        assert_eq!(builder.boost, 1.5);
    }

    #[test]
    fn test_multi_field_query() {
        let query = MultiFieldQuery::new("test query".to_string())
            .add_field("title".to_string(), 2.0)
            .add_field("content".to_string(), 1.0)
            .query_type(MultiFieldQueryType::BestFields)
            .tie_breaker(0.3);

        assert_eq!(query.query_text, "test query");
        assert_eq!(query.fields.len(), 2);
        assert_eq!(query.tie_breaker, 0.3);
    }

    #[test]
    fn test_advanced_query_config() {
        let config = AdvancedQueryConfig {
            enable_optimization: false,
            max_clause_count: 500,
            timeout_ms: 10000,
            ..Default::default()
        };

        assert!(!config.enable_optimization);
        assert_eq!(config.max_clause_count, 500);
        assert_eq!(config.timeout_ms, 10000);
    }

    /// Minimal reader: the mock queries below ignore it entirely.
    #[derive(Debug)]
    struct TestReader;

    impl LexicalIndexReader for TestReader {
        fn doc_count(&self) -> u64 {
            0
        }
        fn max_doc(&self) -> u64 {
            0
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

    /// Constant scorer for the mock query.
    #[derive(Debug, Clone)]
    struct UnitScorer;

    impl Scorer for UnitScorer {
        fn score(&self, _doc_id: u64, _term_freq: f32, _field_length: Option<f32>) -> f32 {
            1.0
        }
        fn boost(&self) -> f32 {
            1.0
        }
        fn set_boost(&mut self, _boost: f32) {}
        fn max_score(&self) -> f32 {
            1.0
        }
        fn name(&self) -> &'static str {
            "UnitScorer"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Query yielding a fixed doc-id set, counting matcher constructions
    /// (#1001: post filters must build their matcher once per execute,
    /// not once per candidate document).
    #[derive(Debug)]
    struct FixedDocsQuery {
        docs: Vec<u64>,
        matcher_builds: std::sync::Arc<std::sync::atomic::AtomicU64>,
        boost: f32,
    }

    impl FixedDocsQuery {
        fn new(docs: Vec<u64>) -> Self {
            FixedDocsQuery {
                docs,
                matcher_builds: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                boost: 1.0,
            }
        }
    }

    impl Query for FixedDocsQuery {
        fn matcher(&self, _reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
            self.matcher_builds
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(Box::new(
                crate::lexical::query::matcher::PreComputedMatcher::new(self.docs.clone()),
            ))
        }
        fn scorer(&self, _reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
            Ok(Box::new(UnitScorer))
        }
        fn boost(&self) -> f32 {
            self.boost
        }
        fn set_boost(&mut self, boost: f32) {
            self.boost = boost;
        }
        fn description(&self) -> String {
            "FixedDocsQuery".to_string()
        }
        fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
            Ok(self.docs.is_empty())
        }
        fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
            Ok(self.docs.len() as u64)
        }
        fn clone_box(&self) -> Box<dyn Query> {
            Box::new(FixedDocsQuery {
                docs: self.docs.clone(),
                matcher_builds: self.matcher_builds.clone(),
                boost: self.boost,
            })
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// #1001 regression: each post filter's matcher must be built once
    /// per `execute()`, not once per candidate document.
    #[test]
    fn post_filters_build_one_matcher_per_execute() {
        let reader = TestReader;
        let filter = FixedDocsQuery::new(vec![2, 4]);
        let filter_builds = filter.matcher_builds.clone();

        let query = AdvancedQuery::new(Box::new(FixedDocsQuery::new(vec![1, 2, 3, 4, 5])))
            .with_post_filter(Box::new(filter));
        let results = query.execute(&reader).unwrap();

        let mut ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![2, 4]);
        assert_eq!(
            filter_builds.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "post-filter matcher must be built once per execute"
        );
    }

    /// #1001 regression: `execute()` must not drop the matcher's first
    /// hit (matchers are positioned on their first match at
    /// construction; the old loop advanced before reading it).
    #[test]
    fn execute_keeps_the_first_hit() {
        let reader = TestReader;
        let query = AdvancedQuery::new(Box::new(FixedDocsQuery::new(vec![7, 9])));

        let results = query.execute(&reader).unwrap();

        let mut ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![7, 9], "the first hit must not be dropped");
    }

    /// A reader whose every term claims `u64::MAX` documents, as a corrupt
    /// dictionary can, so a `TermQuery` costs `u64::MAX`.
    #[derive(Debug)]
    struct SaturatedStatsReader;

    impl LexicalIndexReader for SaturatedStatsReader {
        fn doc_count(&self) -> u64 {
            0
        }
        fn max_doc(&self) -> u64 {
            0
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
        fn term_info(
            &self,
            field: &str,
            term: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(Some(crate::lexical::reader::ReaderTermInfo {
                field: field.to_string(),
                term: term.to_string(),
                doc_freq: u64::MAX,
                total_freq: u64::MAX,
                posting_offset: 0,
                posting_size: 0,
                max_score_factor: 0.0,
                block_max: Vec::new(),
            }))
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

    /// Query costs summed from `u64::MAX` term costs saturate rather than
    /// overflow (Issue #1224): across a boolean query's clauses, across
    /// the filters, and filters plus the core query.
    #[test]
    fn cost_sums_saturate_at_u64_max() {
        use crate::lexical::query::boolean::BooleanQuery;

        let reader = SaturatedStatsReader;
        let mut core = BooleanQuery::new();
        core.add_should(Box::new(TermQuery::new("body", "a")));
        core.add_should(Box::new(TermQuery::new("body", "b")));
        assert_eq!(core.cost(&reader).unwrap(), u64::MAX);

        let query = AdvancedQuery::new(Box::new(core))
            .with_filter(Box::new(TermQuery::new("body", "c")))
            .with_filter(Box::new(TermQuery::new("body", "d")));
        assert_eq!(query.cost(&reader).unwrap(), u64::MAX);
    }

    // apple: 0 1 2 5 6 7 8 10 11 / red: 0 2 3 6 8 9 / green: 1 4 7 11
    // fresh: 0 4 7 10 / ripe: 1 2 3 8 10
    const FRUIT_A: &[(u64, &str)] = &[
        (0, "apple red fresh"),
        (1, "apple green ripe"),
        (2, "apple apple red ripe filler filler"),
        (3, "apricot red ripe"),
        (4, "banana green fresh"),
        (5, "apple cherry"),
    ];
    const FRUIT_B: &[(u64, &str)] = &[
        (6, "apple red"),
        (7, "apple green fresh filler"),
        (8, "apricot apple red ripe"),
        (9, "cherry red"),
        (10, "apple fresh fresh ripe"),
        (11, "banana apple green filler filler filler"),
    ];

    /// A store holding one committed segment per slice of `(id, body)`.
    fn store_with_segments(segments: &[&[(u64, &str)]]) -> LexicalStore {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for segment in segments {
            for &(id, body) in *segment {
                let doc = Document::builder().add_text("body", body).build();
                store.upsert_document(id, doc).unwrap();
            }
            store.commit().unwrap();
        }
        let reader = store.reader_for_tests().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        assert_eq!(inverted.segment_count(), segments.len());
        store
    }

    fn term(text: &str) -> Box<dyn Query> {
        Box::new(TermQuery::new("body", text))
    }

    /// The hits of `query` on `store`, as doc id → score.
    fn search_scores(
        store: &LexicalStore,
        query: Box<dyn Query>,
        parallel: bool,
    ) -> BTreeMap<u64, f32> {
        store
            .search(
                LexicalSearchRequest::new(query)
                    .limit(100)
                    .parallel(parallel),
            )
            .unwrap()
            .hits
            .into_iter()
            .map(|hit| (hit.doc_id, hit.score))
            .collect()
    }

    fn assert_scores_eq(label: &str, actual: &BTreeMap<u64, f32>, expected: &BTreeMap<u64, f32>) {
        assert_eq!(
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>(),
            "{label}: hits"
        );
        for (doc_id, want) in expected {
            let got = actual[doc_id];
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                "{label}: doc {doc_id} scores {got}, expected {want}"
            );
        }
    }

    /// #1305: on every search path — one segment (the matcher loop), two
    /// segments (the per-segment fanout), each also with `parallel` — an
    /// `AdvancedQuery` returns the hits `execute()` returns, each scored as
    /// the bare core scores it on that path, times the boost.
    #[test]
    fn search_matches_execute_on_every_path() {
        let one = store_with_segments(&[&[FRUIT_A, FRUIT_B].concat()]);
        let two = store_with_segments(&[FRUIT_A, FRUIT_B]);
        let apple = || AdvancedQuery::new(term("apple"));
        let cases: Vec<(&str, AdvancedQuery, &[u64])> = vec![
            (
                "boost",
                apple().with_boost(2.5),
                &[0, 1, 2, 5, 6, 7, 8, 10, 11],
            ),
            ("filter", apple().with_filter(term("red")), &[0, 2, 6, 8]),
            (
                "negative filter",
                apple().with_negative_filter(term("green")),
                &[0, 2, 5, 6, 8, 10],
            ),
            (
                "post filter",
                apple().with_post_filter(term("ripe")),
                &[1, 2, 8, 10],
            ),
            (
                "all",
                apple()
                    .with_boost(0.5)
                    .with_filter(term("red"))
                    .with_negative_filter(term("fresh"))
                    .with_post_filter(term("ripe")),
                &[2, 8],
            ),
        ];

        for (store_label, store) in [("one segment", &one), ("two segments", &two)] {
            let reader = store.reader_for_tests().unwrap();
            for parallel in [false, true] {
                let bare = search_scores(store, term("apple"), parallel);
                for (case, query, expected_ids) in &cases {
                    let label = format!("{store_label}, parallel={parallel}, {case}");
                    let expected_ids: BTreeSet<u64> = expected_ids.iter().copied().collect();

                    let executed: BTreeSet<u64> = query
                        .execute(reader.as_ref())
                        .unwrap()
                        .iter()
                        .map(|result| result.doc_id)
                        .collect();
                    assert_eq!(executed, expected_ids, "{label}: execute()");

                    let expected: BTreeMap<u64, f32> = expected_ids
                        .iter()
                        .map(|doc_id| (*doc_id, bare[doc_id] * query.boost()))
                        .collect();
                    let hits = search_scores(store, Box::new(query.clone()), parallel);
                    assert_scores_eq(&label, &hits, &expected);
                }
            }
        }
    }

    /// #1305: a wrapped multi-term core is lowered at the top-level
    /// rewrite, so on two segments it returns what the bare query returns.
    /// Unrewritten, the fanout's per-segment views cannot enumerate its
    /// terms.
    #[test]
    fn wrapped_multi_term_cores_match_the_bare_queries_on_two_segments() {
        let store = store_with_segments(&[FRUIT_A, FRUIT_B]);
        let cores: Vec<(&str, Box<dyn Query>)> = vec![
            ("prefix", Box::new(PrefixQuery::new("body", "ap"))),
            (
                "wildcard",
                Box::new(WildcardQuery::new("body", "ch*y").unwrap()),
            ),
            ("fuzzy", Box::new(FuzzyQuery::new("body", "aple"))),
            (
                "regexp",
                Box::new(RegexpQuery::new("body", "ban.*").unwrap()),
            ),
        ];
        for (name, core) in cores {
            let bare = search_scores(&store, core.clone_box(), false);
            assert!(!bare.is_empty(), "{name}: the bare query must match");
            let wrapped = search_scores(
                &store,
                Box::new(AdvancedQuery::new(core.clone_box())),
                false,
            );
            assert_scores_eq(name, &wrapped, &bare);

            // With a filter the wrapper lowers to a `BooleanQuery`, whose
            // own rewrite lowers the core.
            let mut boolean = BooleanQuery::new();
            boolean.add_must(core.clone_box());
            boolean.add_filter(term("red"));
            let bare = search_scores(&store, Box::new(boolean), false);
            let wrapped = search_scores(
                &store,
                Box::new(AdvancedQuery::new(core.clone_box()).with_filter(term("red"))),
                false,
            );
            assert_scores_eq(&format!("{name} with a filter"), &wrapped, &bare);
        }
    }

    /// #1305: a wrapped `SynonymQuery` is frozen with the whole index's
    /// statistics at the top-level rewrite, so on two segments holding
    /// different alternatives it scores each document as the bare query
    /// does (the #1257 fixture: "large" is absent from the first segment).
    #[test]
    fn wrapped_synonym_core_scores_as_the_bare_query_on_two_segments() {
        let first: Vec<(u64, &str)> = std::iter::once((0, "big filler"))
            .chain((1..10).map(|id| (id, "filler filler")))
            .collect();
        let second: Vec<(u64, &str)> = std::iter::once((10, "big filler"))
            .chain((11..20).map(|id| (id, "large filler")))
            .collect();
        let store = store_with_segments(&[&first, &second]);
        let synonym =
            || -> Box<dyn Query> { Box::new(SynonymQuery::new("body", vec!["big", "large"])) };

        let bare = search_scores(&store, synonym(), false);
        assert_eq!(bare.len(), 11, "docs 0, 10 and 11..20 hold an alternative");
        let wrapped = search_scores(&store, Box::new(AdvancedQuery::new(synonym())), false);
        assert_scores_eq("synonym", &wrapped, &bare);
    }

    /// #1305: an `AdvancedQuery` used as a `Filter` clause is served from
    /// the cross-segment filter cache through its lowered query's
    /// `cache_key`, so its multi-term core is enumerated against the whole
    /// index, not a per-segment view.
    #[test]
    fn advanced_query_as_a_filter_clause_matches_on_two_segments() {
        let store = store_with_segments(&[FRUIT_A, FRUIT_B]);
        let red_filtered_by = |filter: Box<dyn Query>| -> Box<dyn Query> {
            let mut query = BooleanQuery::new();
            query.add_must(term("red"));
            query.add_filter(filter);
            Box::new(query)
        };
        let prefix = || -> Box<dyn Query> { Box::new(PrefixQuery::new("body", "ap")) };

        let bare = search_scores(&store, red_filtered_by(prefix()), false);
        assert_eq!(
            bare.keys().copied().collect::<Vec<_>>(),
            vec![0, 2, 3, 6, 8]
        );
        let wrapped = search_scores(
            &store,
            red_filtered_by(Box::new(AdvancedQuery::new(prefix()))),
            false,
        );
        assert_scores_eq("filter clause", &wrapped, &bare);
    }

    /// #1305: `execute()` applies filters and negative filters whether or
    /// not `enable_optimization` is set; turning it off used to drop them.
    #[test]
    fn execute_applies_filters_without_optimization() {
        let store = store_with_segments(&[FRUIT_A, FRUIT_B]);
        let reader = store.reader_for_tests().unwrap();
        let query = AdvancedQuery::new(term("apple"))
            .with_filter(term("red"))
            .with_negative_filter(term("fresh"))
            .with_config(AdvancedQueryConfig {
                enable_optimization: false,
                ..Default::default()
            });

        let ids: BTreeSet<u64> = query
            .execute(reader.as_ref())
            .unwrap()
            .iter()
            .map(|result| result.doc_id)
            .collect();
        assert_eq!(ids, BTreeSet::from([2, 6, 8]));
    }

    // ----- #1317: MultiFieldQuery per-field boost, tie_breaker, BestFields -----

    // title: "apple" matches 0, 2 / body: "apple" matches 1, 2.
    // Different per-field term frequencies and lengths give field 0 and
    // field 1 distinguishable BM25 scores on doc 2, so a dis-max test can
    // tell "max of the fields" apart from "sum of the fields".
    const TITLE_BODY: &[(u64, &str, &str)] = &[
        (0, "apple", "filler"),
        (
            1,
            "filler filler filler filler filler filler filler filler",
            "apple apple apple",
        ),
        (2, "apple", "apple apple apple"),
    ];

    /// A store holding one committed segment per slice of `(id, title, body)`.
    fn store_with_two_field_segments(segments: &[&[(u64, &str, &str)]]) -> LexicalStore {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for segment in segments {
            for &(id, title, body) in *segment {
                let doc = Document::builder()
                    .add_text("title", title)
                    .add_text("body", body)
                    .build();
                store.upsert_document(id, doc).unwrap();
            }
            store.commit().unwrap();
        }
        store
    }

    /// #1317: a per-field boost passed to `add_field` scales that field's
    /// contribution to the score (acceptance criterion 1).
    #[test]
    fn per_field_boost_scales_the_fields_contribution() {
        let store = store_with_segments(&[&[FRUIT_A, FRUIT_B].concat()]);
        let unboosted = search_scores(
            &store,
            Box::new(MultiFieldQuery::new("apple".to_string()).add_field("body".to_string(), 1.0)),
            false,
        );
        let boosted = search_scores(
            &store,
            Box::new(MultiFieldQuery::new("apple".to_string()).add_field("body".to_string(), 3.0)),
            false,
        );
        assert!(!unboosted.is_empty(), "expected matches");
        for (doc_id, score) in &unboosted {
            assert!(
                (boosted[doc_id] - score * 3.0).abs() < 1e-4,
                "doc {doc_id}: boosted score {} is not 3x the unboosted score {score}",
                boosted[doc_id]
            );
        }
    }

    /// #1317: `BestFields` with the default `tie_breaker` (0.0) scores a
    /// document as the *best* matching field's score, not the sum of every
    /// matching field — distinguishing it from the old (and from
    /// `MostFields`'s still-correct) summing behavior.
    #[test]
    fn best_fields_scores_as_the_max_of_matching_fields() {
        let store = store_with_two_field_segments(&[TITLE_BODY]);
        let title_scores = search_scores(&store, Box::new(TermQuery::new("title", "apple")), false);
        let body_scores = search_scores(&store, Box::new(TermQuery::new("body", "apple")), false);
        let title_score = title_scores[&2];
        let body_score = body_scores[&2];
        assert_ne!(
            title_score, body_score,
            "the fixture must give the two fields distinguishable scores"
        );

        let hits = search_scores(
            &store,
            Box::new(
                MultiFieldQuery::new("apple".to_string())
                    .add_field("title".to_string(), 1.0)
                    .add_field("body".to_string(), 1.0)
                    .query_type(MultiFieldQueryType::BestFields),
            ),
            false,
        );

        let max = title_score.max(body_score);
        assert!((hits[&2] - max).abs() < 1e-5, "doc 2 must score as the max");
        assert!(
            (hits[&2] - (title_score + body_score)).abs() > 1e-4,
            "doc 2 must not score as the sum of both fields"
        );
    }

    /// #1317: `tie_breaker` is wired up — 1.0 recovers the old summed
    /// score, and 0.5 lands exactly between the max-only and summed scores.
    #[test]
    fn tie_breaker_blends_in_the_other_matching_fields() {
        let store = store_with_two_field_segments(&[TITLE_BODY]);
        let title_score =
            search_scores(&store, Box::new(TermQuery::new("title", "apple")), false)[&2];
        let body_score =
            search_scores(&store, Box::new(TermQuery::new("body", "apple")), false)[&2];
        let max = title_score.max(body_score);
        let rest = title_score + body_score - max;

        let query_with = |tie_breaker: f32| -> Box<dyn Query> {
            Box::new(
                MultiFieldQuery::new("apple".to_string())
                    .add_field("title".to_string(), 1.0)
                    .add_field("body".to_string(), 1.0)
                    .query_type(MultiFieldQueryType::BestFields)
                    .tie_breaker(tie_breaker),
            )
        };

        let summed = search_scores(&store, query_with(1.0), false);
        assert!((summed[&2] - (title_score + body_score)).abs() < 1e-5);

        let blended = search_scores(&store, query_with(0.5), false);
        assert!((blended[&2] - (max + 0.5 * rest)).abs() < 1e-5);
    }

    /// #1317: `MostFields` keeps requiring every field to match (the
    /// `Occur::Must` behavior) and keeps summing their scores.
    #[test]
    fn most_fields_still_requires_every_field_and_sums_scores() {
        let store = store_with_two_field_segments(&[TITLE_BODY]);
        let title_score =
            search_scores(&store, Box::new(TermQuery::new("title", "apple")), false)[&2];
        let body_score =
            search_scores(&store, Box::new(TermQuery::new("body", "apple")), false)[&2];

        let hits = search_scores(
            &store,
            Box::new(
                MultiFieldQuery::new("apple".to_string())
                    .add_field("title".to_string(), 1.0)
                    .add_field("body".to_string(), 1.0)
                    .query_type(MultiFieldQueryType::MostFields),
            ),
            false,
        );

        // Docs 0 and 1 each match only one field, so `Must` on both excludes them.
        assert_eq!(hits.keys().collect::<Vec<_>>(), vec![&2]);
        assert!((hits[&2] - (title_score + body_score)).abs() < 1e-5);
    }

    /// #1317: a boosted `BestFields` query with a non-trivial `tie_breaker`
    /// scores identically on every search path (one segment, two segments,
    /// each with and without `parallel`), each computed from that same
    /// path's own per-field `TermQuery` scores (BM25 stats such as average
    /// field length are per-segment, so no single absolute value holds
    /// across every path).
    #[test]
    fn best_fields_scores_agree_across_every_search_path() {
        let one = store_with_two_field_segments(&[TITLE_BODY]);
        let two = store_with_two_field_segments(&[&TITLE_BODY[..1], &TITLE_BODY[1..]]);

        for (label, store) in [("one segment", &one), ("two segments", &two)] {
            for parallel in [false, true] {
                let title_scores = search_scores(
                    store,
                    Box::new(TermQuery::new("title", "apple").with_boost(2.0)),
                    parallel,
                );
                let body_scores =
                    search_scores(store, Box::new(TermQuery::new("body", "apple")), parallel);

                let hits = search_scores(
                    store,
                    Box::new(
                        MultiFieldQuery::new("apple".to_string())
                            .add_field("title".to_string(), 2.0)
                            .add_field("body".to_string(), 1.0)
                            .query_type(MultiFieldQueryType::BestFields)
                            .tie_breaker(0.3),
                    ),
                    parallel,
                );

                assert!(
                    !hits.is_empty(),
                    "{label}, parallel={parallel}: expected matches"
                );
                for (doc_id, score) in &hits {
                    let t = title_scores.get(doc_id).copied().unwrap_or(0.0);
                    let b = body_scores.get(doc_id).copied().unwrap_or(0.0);
                    let max = t.max(b);
                    let expected = max + 0.3 * (t + b - max);
                    assert!(
                        (score - expected).abs() < 1e-4,
                        "{label}, parallel={parallel}, doc {doc_id}: got {score}, expected {expected}"
                    );
                }
            }
        }
    }

    // ----- #1319: MultiFieldQuery CombinedFields (BM25F) -----

    /// A document as `(id, [(field, text)])`, so a fixture can leave a field
    /// out of a document altogether.
    type FieldDoc<'a> = (u64, &'a [(&'a str, &'a str)]);

    /// A store holding one committed segment per batch of documents.
    fn store_with_document_segments(segments: Vec<Vec<(u64, Document)>>) -> LexicalStore {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        let segment_count = segments.len();
        for segment in segments {
            for (id, doc) in segment {
                store.upsert_document(id, doc).unwrap();
            }
            store.commit().unwrap();
        }
        let reader = store.reader_for_tests().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        assert_eq!(inverted.segment_count(), segment_count);
        store
    }

    /// A store holding one committed segment per slice of documents.
    fn store_with_field_docs(segments: &[&[FieldDoc]]) -> LexicalStore {
        store_with_document_segments(
            segments
                .iter()
                .map(|segment| {
                    segment
                        .iter()
                        .map(|&(id, fields)| {
                            let doc = fields
                                .iter()
                                .fold(Document::builder(), |doc, &(field, text)| {
                                    doc.add_text(field, text)
                                })
                                .build();
                            (id, doc)
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// A `CombinedFields` query for "apple" over `(field, weight)` pairs.
    fn combined(fields: &[(&str, f32)]) -> MultiFieldQuery {
        fields.iter().fold(
            MultiFieldQuery::new("apple".to_string())
                .query_type(MultiFieldQueryType::CombinedFields),
            |query, &(field, weight)| query.add_field(field.to_string(), weight),
        )
    }

    /// The same query as a `BestFields` one, every field weighted 1.
    fn best_fields(fields: &[&str]) -> Box<dyn Query> {
        Box::new(fields.iter().fold(
            MultiFieldQuery::new("apple".to_string()).query_type(MultiFieldQueryType::BestFields),
            |query, &field| query.add_field(field.to_string(), 1.0),
        ))
    }

    /// A `BooleanQuery` holding each `(query, occur)` clause.
    fn boolean_of(clauses: Vec<(Box<dyn Query>, Occur)>) -> Box<dyn Query> {
        let mut query = BooleanQuery::new();
        for (clause, occur) in clauses {
            query.add_clause(BooleanClause::new(clause, occur));
        }
        Box::new(query)
    }

    /// Documents without "apple", so its IDF is not clamped to the floor.
    const NO_APPLE: &[(&str, &str)] = &[("title", "fig plum"), ("body", "kiwi pear")];

    /// #1319 (acceptance criterion 2): with weight 1, a term split across
    /// two short fields scores the same as the same term concentrated in
    /// one field, given the same combined length — including when the
    /// document lacks the other field altogether, which adds no length.
    /// `BestFields` scores each field on its own, so it does not.
    #[test]
    fn combined_fields_scores_split_and_concentrated_terms_alike() {
        let store = store_with_field_docs(&[&[
            // tf 1 + 1, length 2 + 2.
            (0, &[("title", "apple fig"), ("body", "apple plum")]),
            // tf 2 + 0, length 3 + 1.
            (1, &[("title", "apple apple fig"), ("body", "plum")]),
            // No title: tf 2, length 4.
            (2, &[("body", "apple apple fig plum")]),
            (3, NO_APPLE),
            (4, NO_APPLE),
            (5, NO_APPLE),
            (6, NO_APPLE),
        ]]);

        let hits = search_scores(
            &store,
            Box::new(combined(&[("title", 1.0), ("body", 1.0)])),
            false,
        );
        assert_eq!(hits.keys().collect::<Vec<_>>(), vec![&0, &1, &2]);
        for doc_id in [1, 2] {
            assert!(
                (hits[&doc_id] - hits[&0]).abs() <= 1e-6,
                "doc {doc_id} scores {}, doc 0 scores {}",
                hits[&doc_id],
                hits[&0]
            );
        }

        let per_field = search_scores(&store, best_fields(&["title", "body"]), false);
        assert!(
            (per_field[&0] - per_field[&1]).abs() > 1e-4,
            "the fixture must tell independent per-field scoring apart"
        );
    }

    /// #1319 (acceptance criterion 1): one blended document frequency
    /// serves every field. "apple" is rare in title and common in body, so
    /// `BestFields` scores a title match above an equally long body match;
    /// `CombinedFields` scores the two alike.
    #[test]
    fn combined_fields_blends_the_document_frequency_across_fields() {
        let body_apple: &[(&str, &str)] = &[("title", "fig plum"), ("body", "apple kiwi")];
        let store = store_with_field_docs(&[&[
            (0, &[("title", "apple fig"), ("body", "plum kiwi")]),
            (1, body_apple),
            (2, body_apple),
            (3, body_apple),
            (4, body_apple),
            (5, NO_APPLE),
            (6, NO_APPLE),
            (7, NO_APPLE),
            (8, NO_APPLE),
            (9, NO_APPLE),
        ]]);

        let hits = search_scores(
            &store,
            Box::new(combined(&[("title", 1.0), ("body", 1.0)])),
            false,
        );
        assert!((hits[&0] - hits[&1]).abs() <= 1e-6, "{hits:?}");

        let per_field = search_scores(&store, best_fields(&["title", "body"]), false);
        assert!(
            per_field[&0] > per_field[&1] + 1e-4,
            "the rarer title match must win under per-field IDF: {per_field:?}"
        );
    }

    /// Every document holds title, body, and `all` = title + " " + body
    /// (`all2` = title twice, then body). The documents with "apple" in body
    /// include every one with it in title, so the blended (largest)
    /// document frequency equals the concatenated field's.
    const CONCATENATED: &[(u64, &str, &str)] = &[
        (0, "apple fig", "apple plum kiwi"),
        (1, "fig", "apple apple pear"),
        (2, "plum kiwi fig", "pear"),
        (3, "apple apple", "apple fig"),
        (4, "kiwi", "plum pear fig"),
        (5, "pear plum", "apple"),
    ];

    /// A store holding `CONCATENATED`, committed as consecutive segments of
    /// `segment_sizes` documents each.
    fn concatenated_store(segment_sizes: &[usize]) -> LexicalStore {
        let mut docs = CONCATENATED.iter().map(|&(id, title, body)| {
            let doc = Document::builder()
                .add_text("title", title)
                .add_text("body", body)
                .add_text("all", format!("{title} {body}"))
                .add_text("all2", format!("{title} {title} {body}"))
                .build();
            (id, doc)
        });
        store_with_document_segments(
            segment_sizes
                .iter()
                .map(|&size| docs.by_ref().take(size).collect())
                .collect(),
        )
    }

    /// A labelled query and the reference query it must score like.
    type ScoredLike = (&'static str, Box<dyn Query>, Box<dyn Query>);

    /// #1319: `CombinedFields` scores exactly like one field holding the
    /// fields' concatenation, on every search path: one and two segments,
    /// at the top level and nested in a `BooleanQuery` (which is what
    /// `parallel` changes the path for).
    #[test]
    fn combined_fields_scores_like_the_concatenated_field_on_every_path() {
        let stores = [
            ("one segment", concatenated_store(&[6])),
            ("two segments", concatenated_store(&[3, 3])),
        ];
        let never = || -> Box<dyn Query> { Box::new(TermQuery::new("body", "zzz")) };
        for (label, store) in &stores {
            for parallel in [false, true] {
                let cases: [ScoredLike; 2] = [
                    (
                        "top level",
                        Box::new(combined(&[("title", 1.0), ("body", 1.0)])),
                        Box::new(TermQuery::new("all", "apple")),
                    ),
                    (
                        "nested",
                        boolean_of(vec![
                            (
                                Box::new(combined(&[("title", 1.0), ("body", 1.0)])),
                                Occur::Should,
                            ),
                            (never(), Occur::Should),
                        ]),
                        boolean_of(vec![
                            (Box::new(TermQuery::new("all", "apple")), Occur::Should),
                            (never(), Occur::Should),
                        ]),
                    ),
                ];
                for (shape, query, reference) in cases {
                    let expected = search_scores(store, reference, parallel);
                    assert_eq!(expected.len(), 4, "{label}: the fixture's apple docs");
                    assert_scores_eq(
                        &format!("{label}, {shape}, parallel={parallel}"),
                        &search_scores(store, query, parallel),
                        &expected,
                    );
                }
            }
        }
    }

    /// #1319: a BM25F weight scales a field's term frequency and length
    /// alike, so weight 2 on title scores like a field holding title twice.
    #[test]
    fn combined_fields_weight_counts_the_field_that_many_times() {
        for (label, store) in [
            ("one segment", concatenated_store(&[6])),
            ("two segments", concatenated_store(&[3, 3])),
        ] {
            assert_scores_eq(
                label,
                &search_scores(
                    &store,
                    Box::new(combined(&[("title", 2.0), ("body", 1.0)])),
                    false,
                ),
                &search_scores(&store, Box::new(TermQuery::new("all2", "apple")), false),
            );
        }
    }

    /// #1319: the blended document frequency is frozen against the
    /// top-level reader. Title holds "apple" in three documents and body in
    /// two, but segment 2 has none in title: blending per segment would give
    /// segment 2 the body's frequency instead, and so score doc 4 apart from
    /// doc 0, which has the same content and the same segment statistics.
    #[test]
    fn combined_fields_scores_identical_documents_alike_across_segments() {
        let body_apple: &[(&str, &str)] = &[("title", "pear fig"), ("body", "apple plum")];
        let title_apple: &[(&str, &str)] = &[("title", "apple fig"), ("body", "kiwi plum")];
        let neither: &[(&str, &str)] = &[("title", "pear fig"), ("body", "kiwi plum")];
        let store = store_with_field_docs(&[
            &[
                (0, body_apple),
                (1, title_apple),
                (2, title_apple),
                (3, title_apple),
            ],
            &[(4, body_apple), (5, neither), (6, neither), (7, neither)],
        ]);

        let query = || combined(&[("title", 1.0), ("body", 1.0)]);
        for parallel in [false, true] {
            let cases: [(&str, Box<dyn Query>); 2] = [
                ("top level", Box::new(query())),
                (
                    "nested",
                    boolean_of(vec![
                        (Box::new(query()), Occur::Must),
                        (Box::new(TermQuery::new("body", "apple")), Occur::Filter),
                    ]),
                ),
            ];
            for (shape, query) in cases {
                let hits = search_scores(&store, query, parallel);
                assert!(
                    (hits[&0] - hits[&4]).abs() <= 1e-6,
                    "{shape}, parallel={parallel}: {hits:?}"
                );
            }
        }
    }

    /// #1319: a boost that is not finite and greater than zero is rejected
    /// when searching and counting; a fractional one is a valid weight.
    #[test]
    fn combined_fields_rejects_non_positive_or_non_finite_weights() {
        let store = store_with_field_docs(&[&[
            (0, &[("title", "apple fig"), ("body", "apple plum")]),
            (1, NO_APPLE),
        ]]);
        for weight in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY] {
            let request = || {
                LexicalSearchRequest::new(Box::new(combined(&[("title", weight), ("body", 1.0)])))
            };
            let error = store.search(request()).unwrap_err();
            assert!(
                matches!(error, LaurusError::InvalidArgument(_)),
                "weight {weight}: search returned {error:?}"
            );
            let error = store.count(request()).unwrap_err();
            assert!(
                matches!(error, LaurusError::InvalidArgument(_)),
                "weight {weight}: count returned {error:?}"
            );
        }

        let hits = search_scores(
            &store,
            Box::new(combined(&[("title", 0.5), ("body", 1.0)])),
            false,
        );
        assert_eq!(hits.keys().collect::<Vec<_>>(), vec![&0]);
        assert!(hits[&0].is_finite() && hits[&0] > 0.0, "{hits:?}");
        assert_eq!(
            store
                .count(LexicalSearchRequest::new(Box::new(combined(&[
                    ("title", 0.5),
                    ("body", 1.0)
                ]))))
                .unwrap(),
            1
        );
    }

    /// #1319: `rewrite` freezes a `CombinedFields` query's document
    /// frequency once against the top-level reader, and leaves the other
    /// query types alone.
    #[test]
    fn combined_fields_rewrite_freezes_the_document_frequency_once() {
        let store = concatenated_store(&[3, 3]);
        let reader = store.reader_for_tests().unwrap();

        let rewritten = combined(&[("title", 1.0), ("body", 1.0)])
            .rewrite(reader.as_ref())
            .unwrap()
            .expect("CombinedFields rewrites against the top-level reader");
        let frozen = rewritten
            .as_any()
            .downcast_ref::<MultiFieldQuery>()
            .unwrap();
        // The larger of title's (docs 0, 3) and body's (docs 0, 1, 3, 5).
        assert_eq!(frozen.doc_freq, Some(4));
        assert!(rewritten.rewrite(reader.as_ref()).unwrap().is_none());
        assert_eq!(
            frozen.clone().add_field("all".to_string(), 1.0).doc_freq,
            None,
            "adding a field invalidates the frozen value"
        );

        for query_type in [
            MultiFieldQueryType::BestFields,
            MultiFieldQueryType::MostFields,
        ] {
            let query = MultiFieldQuery::new("apple".to_string())
                .add_field("title".to_string(), 1.0)
                .query_type(query_type);
            assert!(query.rewrite(reader.as_ref()).unwrap().is_none());
        }
    }
}
