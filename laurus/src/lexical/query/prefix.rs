//! Prefix query implementation.
//!
//! This module provides support for finding terms that start with a specific prefix.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::index::inverted::core::terms::{TermDictionaryAccess, TermsEnum};
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::multi_term::{MultiTermQuery, RewriteMethod};
use crate::lexical::query::scorer::Scorer;
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::LexicalIndexReader;

/// A query that matches terms starting with a specific prefix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixQuery {
    /// Field to search in
    field: String,
    /// Prefix to search for
    prefix: String,
    /// Boost factor for the query
    boost: f32,
    /// Rewrite method for multi-term expansion
    rewrite_method: RewriteMethod,
}

impl PrefixQuery {
    /// Create a new prefix query.
    pub fn new<F: Into<String>, P: Into<String>>(field: F, prefix: P) -> Self {
        PrefixQuery {
            field: field.into(),
            prefix: prefix.into(),
            boost: 1.0,
            rewrite_method: RewriteMethod::default(),
        }
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set the rewrite method.
    pub fn with_rewrite_method(mut self, rewrite_method: RewriteMethod) -> Self {
        self.rewrite_method = rewrite_method;
        self
    }

    /// Get the prefix.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the rewrite method.
    pub fn rewrite_method(&self) -> RewriteMethod {
        self.rewrite_method
    }
}

impl MultiTermQuery for PrefixQuery {
    fn field(&self) -> &str {
        &self.field
    }

    fn rewrite_method(&self) -> RewriteMethod {
        self.rewrite_method
    }

    fn get_terms_enum(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<Option<Box<dyn TermsEnum>>> {
        if let Some(inverted_reader) = reader.as_any().downcast_ref::<InvertedIndexReader>()
            && let Some(terms) = inverted_reader.terms(&self.field)?
        {
            // Use Generic AutomatonTermsEnum with RegexAutomaton for prefix
            // Pattern: ^escaped_prefix.*
            let escaped_prefix = regex::escape(&self.prefix);
            let pattern = format!("^{}.*", escaped_prefix);

            let regex_automaton =
                crate::lexical::index::inverted::core::automaton::RegexAutomaton::new(&pattern)?;

            let terms_enum =
                crate::lexical::index::inverted::core::automaton::AutomatonTermsEnum::new(
                    terms.iterator()?,
                    regex_automaton,
                );

            return Ok(Some(Box::new(terms_enum)));
        }
        Ok(None)
    }

    fn enumerate_terms(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<(String, u64, f32)>> {
        if let Some(mut terms_enum) = self.get_terms_enum(reader)? {
            let mut results = Vec::new();
            let max = self.rewrite_method.max_expansions();
            while let Some(term_stats) = terms_enum.next()? {
                results.push((term_stats.term.clone(), term_stats.doc_freq, 1.0));
                if let Some(m) = max
                    && results.len() >= m
                {
                    break;
                }
            }
            return Ok(results);
        }
        Ok(Vec::new())
    }
}

impl Query for PrefixQuery {
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

    fn description(&self) -> String {
        format!(
            "PrefixQuery(field: {}, prefix: {})",
            self.field, self.prefix
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.prefix.is_empty())
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
        if !self.prefix.is_empty() && field.is_none_or(|f| f == self.field) {
            out.push(HighlightTerm::Prefix(self.prefix.clone()));
        }
    }

    fn cache_key(&self) -> Option<String> {
        // Field + prefix + rewrite method determine the matched set (the
        // rewrite method can cap term expansion, which affects membership);
        // boost is score-only and excluded.
        Some(format!(
            "prefix|{:?}|{:?}|{:?}",
            self.field, self.prefix, self.rewrite_method
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Removed unused imports:
    // use crate::lexical::index::inverted::reader::{InvertedIndexReader, InvertedIndexReaderConfig};
    // use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    // use std::sync::Arc;

    #[test]
    fn test_prefix_query_creation() {
        let query = PrefixQuery::new("field", "pre").with_boost(2.0);

        // Changed to call MultiTermQuery::field() which returns &str
        assert_eq!(MultiTermQuery::field(&query), "field");
        assert_eq!(query.prefix(), "pre");
        assert_eq!(query.boost(), 2.0);
    }

    // Note: Integration tests with actual index would be better,
    // but we can test basic structural properties here.
}
