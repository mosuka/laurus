//! Advanced query system with complex query composition and optimization.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::query::boolean::{BooleanQuery, Occur};
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::scorer::Scorer;
use crate::lexical::query::{HighlightTerm, Query, QueryResult};
use crate::lexical::reader::LexicalIndexReader;

/// Configuration for advanced query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedQueryConfig {
    /// Enable query optimization.
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

    /// Set configuration.
    pub fn with_config(mut self, config: AdvancedQueryConfig) -> Self {
        self.config = config;
        self
    }

    /// Optimize the query for better performance.
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

    /// Execute the advanced query with optimization.
    pub fn execute(&mut self, reader: &dyn LexicalIndexReader) -> Result<Vec<QueryResult>> {
        // Optimize query first
        self.optimize()?;

        // Build the matcher and scorer in one pass (#999).
        let (mut matcher, scorer) = self.core_query.matcher_scorer(reader)?;

        // Build each post filter's matcher once: candidates arrive in
        // ascending doc-id order, so a single monotonic forward pass per
        // filter suffices (#1001 — the previous code rebuilt every
        // filter matcher per candidate document).
        let mut post_filter_matchers = self
            .post_filters
            .iter()
            .map(|filter| filter.matcher(reader))
            .collect::<Result<Vec<_>>>()?;

        let mut results = Vec::new();
        let start_time = crate::util::time::Timer::now();

        // Drain the core matcher. It is positioned on its first match at
        // construction, so read the current doc before advancing (the
        // previous `while matcher.next()` loop dropped the first hit),
        // and advance exactly once per iteration so score / filter
        // rejections never skip the advance.
        while !matcher.is_exhausted() {
            let doc_id = matcher.doc_id();
            if doc_id == u64::MAX {
                break;
            }

            // Check timeout
            if self.config.timeout_ms > 0 && start_time.elapsed_ms() > self.config.timeout_ms {
                break;
            }

            // Calculate score
            let mut score = scorer.score(doc_id, matcher.term_freq() as f32, None);
            score *= self.boost;

            // Minimum score threshold, then post filters
            let admitted = score >= self.min_score.max(self.config.min_score)
                && Self::post_filters_match(&mut post_filter_matchers, doc_id)?;
            if admitted {
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

    /// Check `doc_id` against every prebuilt post-filter matcher.
    ///
    /// The matchers advance monotonically: candidates arrive in
    /// ascending doc-id order, so `skip_to` never needs to move
    /// backwards. A matcher already past `doc_id` reports a mismatch —
    /// the same verdict a freshly built matcher would reach — and an
    /// exhausted matcher correctly rejects everything after its last
    /// match.
    fn post_filters_match(matchers: &mut [Box<dyn Matcher>], doc_id: u64) -> Result<bool> {
        for matcher in matchers.iter_mut() {
            if !matcher.skip_to(doc_id)? || matcher.doc_id() != doc_id {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Query for AdvancedQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        self.core_query.matcher(reader)
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        self.core_query.scorer(reader)
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
        let filter_cost = self
            .filters
            .iter()
            .map(|f| f.cost(reader))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum::<u64>();
        Ok(base_cost + filter_cost)
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
    use super::*;
    use crate::lexical::query::term::TermQuery;

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

        let mut query = AdvancedQuery::new(Box::new(FixedDocsQuery::new(vec![1, 2, 3, 4, 5])))
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
        let mut query = AdvancedQuery::new(Box::new(FixedDocsQuery::new(vec![7, 9])));

        let results = query.execute(&reader).unwrap();

        let mut ids: Vec<u64> = results.iter().map(|r| r.doc_id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![7, 9], "the first hit must not be dropped");
    }
}
