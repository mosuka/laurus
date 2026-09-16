//! Fuzzy query implementation for approximate string matching.

// use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::index::inverted::core::automaton::LevenshteinAutomaton;
use crate::lexical::index::inverted::core::terms::{TermDictionaryAccess, TermsEnum};
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::multi_term::{MultiTermQuery, RewriteMethod};
use crate::lexical::query::scorer::Scorer;
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::LexicalIndexReader;

/// A fuzzy query for approximate string matching.
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
    /// Maximum number of terms to expand to (default: 50, like Lucene)
    /// This prevents queries from matching too many terms and consuming excessive resources.
    max_expansions: usize,
    /// Boost factor for the query
    boost: f32,
    /// Rewrite method for multi-term expansion
    rewrite_method: RewriteMethod,
}

impl FuzzyQuery {
    /// Create a new fuzzy query with default settings.
    pub fn new<F: Into<String>, T: Into<String>>(field: F, term: T) -> Self {
        FuzzyQuery {
            field: field.into(),
            term: term.into(),
            max_edits: 2,
            prefix_length: 0,
            transpositions: true,
            max_expansions: 50, // Same default as Lucene
            boost: 1.0,
            rewrite_method: RewriteMethod::default(),
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

    /// Set the maximum number of terms to expand to.
    /// This prevents queries from matching too many terms and consuming excessive resources.
    /// Default is 50, same as Lucene.
    pub fn max_expansions(mut self, max_expansions: usize) -> Self {
        self.max_expansions = max_expansions;
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

    /// Get the maximum number of terms to expand to.
    pub fn get_max_expansions(&self) -> usize {
        self.max_expansions
    }

    /// Set the rewrite method.
    pub fn with_rewrite_method(mut self, rewrite_method: RewriteMethod) -> Self {
        self.rewrite_method = rewrite_method;
        self
    }

    /// Get the rewrite method.
    pub fn rewrite_method(&self) -> RewriteMethod {
        self.rewrite_method
    }

    /// Find matching terms using efficient term dictionary enumeration.
    ///
    /// This uses the Term Dictionary API and Levenshtein Automaton for efficient matching,
    /// similar to Lucene's FuzzyTermsEnum approach.
    ///
    /// Falls back to legacy document scanning if Term Dictionary API is not available.
    fn get_terms_enum(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<Option<Box<dyn TermsEnum>>> {
        if let Some(inverted_reader) = reader.as_any().downcast_ref::<InvertedIndexReader>()
            && let Some(terms) = inverted_reader.terms(&self.field)?
        {
            let terms_enum =
                crate::lexical::index::inverted::core::automaton::AutomatonTermsEnum::new(
                    terms.iterator()?,
                    self.automaton(),
                );
            return Ok(Some(Box::new(terms_enum)));
        }
        Ok(None)
    }

    /// The edit-distance automaton this query enumerates terms with — and
    /// what the highlighter tests tokens against (#594).
    fn automaton(&self) -> LevenshteinAutomaton {
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

    fn enumerate_terms(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<(String, u64, f32)>> {
        let mut results = Vec::new();
        if let Some(mut terms_enum) = self.get_terms_enum(reader)? {
            while let Some(term_stats) = terms_enum.next()? {
                // TODO: Calculate actual similarity score if needed
                results.push((term_stats.term.clone(), term_stats.doc_freq, 1.0));
            }
        }
        Ok(results)
    }
}

impl Query for FuzzyQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        let rewritten = MultiTermQuery::rewrite(self, reader)?;
        rewritten.matcher(reader)
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        let rewritten = MultiTermQuery::rewrite(self, reader)?;
        rewritten.scorer(reader)
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
        // Lower once at the searcher level (Issue #613). When the reader
        // cannot enumerate terms (e.g. the per-segment fanout view), keep
        // the original query so the matcher/scorer fallback stays in
        // charge; probing via the downcast costs nothing extra.
        if reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .is_none()
        {
            return Ok(None);
        }
        Ok(Some(MultiTermQuery::rewrite(self, reader)?))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        if !self.term.is_empty() && field.is_none_or(|f| f == self.field) {
            out.push(HighlightTerm::Fuzzy(self.automaton()));
        }
    }

    fn cache_key(&self) -> Option<String> {
        // Every field below affects which terms (and therefore which
        // documents) match: edit distance, the untouched prefix length,
        // transposition support, the expansion cap, and the rewrite method.
        // Boost is score-only and excluded.
        Some(format!(
            "fuzzy|{:?}|{:?}|{}|{}|{}|{}|{:?}",
            self.field,
            self.term,
            self.max_edits,
            self.prefix_length,
            self.transpositions,
            self.max_expansions,
            self.rewrite_method
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
