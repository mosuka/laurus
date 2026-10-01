//! Advanced query system with complex query composition and optimization.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::query::boolean::{BooleanQuery, Occur};
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

    /// Cross-field matching strategy.
    tie_breaker: f32,
}

/// Type of multi-field query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiFieldQueryType {
    /// Best matching field.
    BestFields,

    /// Most matching fields.
    MostFields,

    /// Cross-field matching.
    CrossFields,

    /// Boolean combination.
    Boolean,
}

impl MultiFieldQuery {
    /// Create a new multi-field query.
    pub fn new(query_text: String) -> Self {
        MultiFieldQuery {
            query_text,
            fields: HashMap::new(),
            query_type: MultiFieldQueryType::BestFields,
            tie_breaker: 0.0,
        }
    }

    /// Add a field with boost.
    pub fn add_field(mut self, field: String, boost: f32) -> Self {
        self.fields.insert(field, boost);
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
}

impl Query for MultiFieldQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        // Create boolean query based on type
        let mut boolean_builder = BooleanQueryBuilder::new();

        match self.query_type {
            MultiFieldQueryType::BestFields | MultiFieldQueryType::Boolean => {
                // Add each field as a should clause
                for field in self.fields.keys() {
                    let term_query = crate::lexical::query::term::TermQuery::new(
                        field.clone(),
                        self.query_text.clone(),
                    );
                    boolean_builder =
                        boolean_builder.add_clause(Box::new(term_query), Occur::Should);
                }
            }
            MultiFieldQueryType::MostFields => {
                // All fields should match
                for field in self.fields.keys() {
                    let term_query = crate::lexical::query::term::TermQuery::new(
                        field.clone(),
                        self.query_text.clone(),
                    );
                    boolean_builder = boolean_builder.add_clause(Box::new(term_query), Occur::Must);
                }
            }
            MultiFieldQueryType::CrossFields => {
                // Create phrase query across fields (simplified)
                let mut combined_query = BooleanQuery::new();
                for field in self.fields.keys() {
                    let term_query = crate::lexical::query::term::TermQuery::new(
                        field.clone(),
                        self.query_text.clone(),
                    );
                    combined_query.add_should(Box::new(term_query));
                }
                return combined_query.matcher(reader);
            }
        }

        boolean_builder.build().matcher(reader)
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        // Create boolean query and use its scorer
        let mut boolean_builder = BooleanQueryBuilder::new();

        match self.query_type {
            MultiFieldQueryType::BestFields | MultiFieldQueryType::Boolean => {
                for field in self.fields.keys() {
                    let term_query = crate::lexical::query::term::TermQuery::new(
                        field.clone(),
                        self.query_text.clone(),
                    );
                    boolean_builder =
                        boolean_builder.add_clause(Box::new(term_query), Occur::Should);
                }
            }
            MultiFieldQueryType::MostFields => {
                for field in self.fields.keys() {
                    let term_query = crate::lexical::query::term::TermQuery::new(
                        field.clone(),
                        self.query_text.clone(),
                    );
                    boolean_builder = boolean_builder.add_clause(Box::new(term_query), Occur::Must);
                }
            }
            MultiFieldQueryType::CrossFields => {
                let mut combined_query = BooleanQuery::new();
                for field in self.fields.keys() {
                    let term_query = crate::lexical::query::term::TermQuery::new(
                        field.clone(),
                        self.query_text.clone(),
                    );
                    combined_query.add_should(Box::new(term_query));
                }
                return combined_query.scorer(reader);
            }
        }

        boolean_builder.build().scorer(reader)
    }

    fn boost(&self) -> f32 {
        1.0 // Default boost for multi-field queries
    }

    fn set_boost(&mut self, _boost: f32) {
        // Multi-field queries manage boosts per field
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
            "MultiFieldQuery(text: {}, fields: {:?})",
            self.query_text,
            self.fields.keys().collect::<Vec<_>>()
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
}
