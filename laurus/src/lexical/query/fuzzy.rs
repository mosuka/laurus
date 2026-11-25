//! Fuzzy query implementation for approximate string matching.

// use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::index::inverted::core::automaton::{Automaton, LevenshteinAutomaton};
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::multi_term::{
    MultiTermQuery, RewriteMethod, multi_term_matcher, multi_term_scorer, rewrite_multi_term,
    validate_multi_term,
};
use crate::lexical::query::scorer::Scorer;
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::LexicalIndexReader;

/// A fuzzy query for approximate string matching.
///
/// It matches the terms within an edit distance of its own and scores the 50
/// closest of them (Issue #1740): the exact term first, then those with the
/// highest `1 - distance / length of the shorter term`, and among equally close
/// ones the most frequent. That value is the factor of a term's score, and
/// every term is scored with the frequency of the most frequent of them, so how
/// rare a term is plays no part.
///
/// The limit is on the terms that are scored. A filter and an exclusion
/// (`MustNot`) cover every matching term.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyQuery {
    /// Field to search in
    field: String,
    /// Term to search for
    term: String,
    /// Maximum edit distance (Levenshtein distance)
    max_edits: u32,
    /// Minimum prefix length that must match exactly
    prefix_length: u32,
    /// Whether transpositions count as single edits (Damerau-Levenshtein)
    transpositions: bool,
    /// Boost factor for the query
    boost: f32,
    /// Rewrite method for multi-term expansion, with the limit on the terms
    rewrite_method: RewriteMethod,
}

impl FuzzyQuery {
    /// The number of terms that are scored by default, as in Lucene.
    const DEFAULT_MAX_EXPANSIONS: usize = 50;

    /// Create a new fuzzy query with default settings.
    pub fn new<F: Into<String>, T: Into<String>>(field: F, term: T) -> Self {
        FuzzyQuery {
            field: field.into(),
            term: term.into(),
            max_edits: 2,
            prefix_length: 0,
            transpositions: true,
            boost: 1.0,
            rewrite_method: RewriteMethod::TopTermsBlended {
                max_expansions: Self::DEFAULT_MAX_EXPANSIONS,
            },
        }
    }

    /// Set the maximum edit distance.
    pub fn max_edits(mut self, max_edits: u32) -> Self {
        self.max_edits = max_edits;
        self
    }

    /// Set the minimum prefix length that must match exactly.
    pub fn prefix_length(mut self, prefix_length: u32) -> Self {
        self.prefix_length = prefix_length;
        self
    }

    /// Set whether transpositions should be considered single edits.
    pub fn transpositions(mut self, transpositions: bool) -> Self {
        self.transpositions = transpositions;
        self
    }

    /// Set the maximum number of terms that are scored, the closest ones.
    /// The default is 50, as in Lucene; a search rejects `0`.
    ///
    /// The limit is part of the rewrite method, so of this and
    /// [`with_rewrite_method`](Self::with_rewrite_method) the later call wins.
    /// It turns [`RewriteMethod::BooleanQuery`] into
    /// [`RewriteMethod::TopTermsScoring`], and has no effect on
    /// [`RewriteMethod::ConstantScore`], which scores no term.
    pub fn max_expansions(mut self, max_expansions: usize) -> Self {
        self.rewrite_method = match self.rewrite_method {
            RewriteMethod::TopTermsBlended { .. } => {
                RewriteMethod::TopTermsBlended { max_expansions }
            }
            RewriteMethod::TopTermsScoring { .. } | RewriteMethod::BooleanQuery => {
                RewriteMethod::TopTermsScoring { max_expansions }
            }
            RewriteMethod::ConstantScore => RewriteMethod::ConstantScore,
        };
        self
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the search term.
    pub fn term(&self) -> &str {
        &self.term
    }

    /// Get the maximum edit distance.
    pub fn get_max_edits(&self) -> u32 {
        self.max_edits
    }

    /// Get the prefix length.
    pub fn get_prefix_length(&self) -> u32 {
        self.prefix_length
    }

    /// Check if transpositions are enabled.
    pub fn get_transpositions(&self) -> bool {
        self.transpositions
    }

    /// Get the maximum number of terms that are scored, or `None` if the
    /// rewrite method has no limit.
    pub fn get_max_expansions(&self) -> Option<usize> {
        self.rewrite_method.max_expansions()
    }

    /// Set the rewrite method, and with it the limit on the terms.
    pub fn with_rewrite_method(mut self, rewrite_method: RewriteMethod) -> Self {
        self.rewrite_method = rewrite_method;
        self
    }

    /// Get the rewrite method.
    pub fn rewrite_method(&self) -> RewriteMethod {
        self.rewrite_method
    }

    /// The edit-distance automaton this query enumerates terms with — and
    /// what the highlighter tests tokens against (#594).
    fn levenshtein(&self) -> LevenshteinAutomaton {
        LevenshteinAutomaton::new(
            &self.term,
            self.max_edits,
            self.prefix_length as usize,
            self.transpositions,
        )
    }
}

impl MultiTermQuery for FuzzyQuery {
    fn field(&self) -> &str {
        &self.field
    }

    fn rewrite_method(&self) -> RewriteMethod {
        self.rewrite_method
    }

    fn automaton(&self) -> Result<Box<dyn Automaton>> {
        Ok(Box::new(self.levenshtein()))
    }

    /// `1 - distance / length of the shorter term`, in characters, and at
    /// least `0`, as in Lucene: `1` for the exact term.
    fn term_boost(&self, term: &str) -> f32 {
        if term == self.term {
            return 1.0;
        }
        let shorter = self.term.chars().count().min(term.chars().count());
        match self.levenshtein().distance(term) {
            Some(distance) if distance < shorter => 1.0 - distance as f32 / shorter as f32,
            _ => 0.0,
        }
    }
}

impl Query for FuzzyQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        multi_term_matcher(self, reader)
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        multi_term_scorer(self, reader)
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn description(&self) -> String {
        format!(
            "FuzzyQuery(field: {}, term: {}, max_edits: {}, prefix: {})",
            self.field, self.term, self.max_edits, self.prefix_length
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.term.is_empty())
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        // Rough estimate
        Ok(reader.doc_count())
    }

    fn rewrite(&self, reader: &dyn LexicalIndexReader) -> Result<Option<Box<dyn Query>>> {
        rewrite_multi_term(self, reader)
    }

    fn validate_boosts(&self) -> Result<()> {
        validate_multi_term(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        if !self.term.is_empty() && field.is_none_or(|f| f == self.field) {
            out.push(HighlightTerm::Fuzzy(self.levenshtein()));
        }
    }

    fn cache_key(&self) -> Option<String> {
        // Every field below affects which terms (and therefore which
        // documents) match: edit distance, the untouched prefix length and
        // transposition support. The boost and the rewrite method with its
        // limit are score-only and excluded.
        Some(format!(
            "fuzzy|{:?}|{:?}|{}|{}|{}",
            self.field, self.term, self.max_edits, self.prefix_length, self.transpositions
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_query_creation() {
        let query = FuzzyQuery::new("content", "hello")
            .max_edits(1)
            .prefix_length(2)
            .transpositions(false)
            .with_boost(1.5);

        assert_eq!(query.field(), "content");
        assert_eq!(query.term(), "hello");
        assert_eq!(query.get_max_edits(), 1);
        assert_eq!(query.get_prefix_length(), 2);
        assert!(!query.get_transpositions());
        assert_eq!(query.boost(), 1.5);
    }

    /// The limit is part of the rewrite method, so the later call of the two
    /// wins (Issue #1740).
    #[test]
    fn the_limit_is_part_of_the_rewrite_method() {
        let query = || FuzzyQuery::new("title", "test");
        let scoring = |max_expansions| RewriteMethod::TopTermsScoring { max_expansions };
        let blended = |max_expansions| RewriteMethod::TopTermsBlended { max_expansions };
        for (query, method, limit) in [
            (query(), blended(50), Some(50)),
            (query().max_expansions(7), blended(7), Some(7)),
            (query().with_rewrite_method(scoring(3)), scoring(3), Some(3)),
            (
                query().with_rewrite_method(scoring(3)).max_expansions(7),
                scoring(7),
                Some(7),
            ),
            (
                query().max_expansions(7).with_rewrite_method(scoring(3)),
                scoring(3),
                Some(3),
            ),
            (
                query().with_rewrite_method(RewriteMethod::BooleanQuery),
                RewriteMethod::BooleanQuery,
                None,
            ),
            (
                query()
                    .with_rewrite_method(RewriteMethod::BooleanQuery)
                    .max_expansions(7),
                scoring(7),
                Some(7),
            ),
            (
                query()
                    .with_rewrite_method(RewriteMethod::ConstantScore)
                    .max_expansions(7),
                RewriteMethod::ConstantScore,
                None,
            ),
        ] {
            assert_eq!(query.rewrite_method(), method);
            assert_eq!(query.get_max_expansions(), limit, "{method:?}");
        }
    }

    /// A term's factor is `1 - distance / length of the shorter term`, in
    /// characters, and at least zero (Issue #1740).
    #[test]
    fn a_closer_term_has_a_higher_factor() {
        for (own, term, expected) in [
            ("hello", "hello", 1.0),
            ("hello", "hallo", 0.8),
            ("hello", "helloo", 0.8),
            ("hello", "hell", 0.75),
            ("hello", "halli", 0.6),
            // Characters, not bytes.
            ("über", "uber", 0.75),
            ("über", "übel", 0.75),
            ("ab", "abc", 0.5),
            ("ab", "a", 0.0),
            ("a", "ab", 0.0),
            ("a", "b", 0.0),
            ("a", "abc", 0.0),
            ("", "", 1.0),
            ("", "a", 0.0),
            // Not within the distance.
            ("hello", "world", 0.0),
        ] {
            let query = FuzzyQuery::new("title", own);
            assert!(
                (query.term_boost(term) - expected).abs() < 1e-6,
                "{own}, {term}: {}",
                query.term_boost(term)
            );
        }
        // A transposition is one edit, or two.
        let query = FuzzyQuery::new("title", "hello");
        assert!((query.term_boost("ehllo") - 0.8).abs() < 1e-6);
        let query = query.transpositions(false);
        assert!((query.term_boost("ehllo") - 0.6).abs() < 1e-6);
    }

    /// The cache of filters is keyed by the matches, which the limit, the
    /// rewrite method and the boost do not change (Issue #1740).
    #[test]
    fn the_cache_key_is_that_of_the_matches() {
        let key = |query: FuzzyQuery| query.cache_key().unwrap();
        let query = || FuzzyQuery::new("title", "test");
        let own = key(query());
        assert_eq!(key(query().max_expansions(7)), own);
        assert_eq!(key(query().with_boost(2.0)), own);
        assert_eq!(
            key(query().with_rewrite_method(RewriteMethod::ConstantScore)),
            own
        );
        for other in [
            query().max_edits(1),
            query().prefix_length(1),
            query().transpositions(false),
            FuzzyQuery::new("title", "tests"),
            FuzzyQuery::new("body", "test"),
        ] {
            assert_ne!(key(other), own);
        }
    }

    #[test]
    fn test_fuzzy_query_description() {
        let query = FuzzyQuery::new("title", "test")
            .max_edits(2)
            .prefix_length(1);
        let description = query.description();
        assert!(description.contains("FuzzyQuery"));
        assert!(description.contains("title"));
        assert!(description.contains("test"));
        assert!(description.contains("max_edits: 2"));
        assert!(description.contains("prefix: 1"));
    }
}
