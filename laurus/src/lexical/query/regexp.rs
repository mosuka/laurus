//! Regular expression query implementation.

use std::fmt::Debug;
use std::sync::Arc;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::lexical::index::inverted::core::automaton::{AutomatonTermsEnum, RegexAutomaton};
use crate::lexical::index::inverted::core::terms::{TermDictionaryAccess, TermsEnum};
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::multi_term::{MultiTermQuery, RewriteMethod};
use crate::lexical::query::scorer::Scorer;
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::LexicalIndexReader;

/// A query that matches documents containing terms that match a regular expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegexpQuery {
    /// The field to search in.
    field: String,
    /// The regular expression pattern.
    pattern: String,
    /// The compiled regex for matching.
    #[serde(skip)]
    regex: Option<Arc<Regex>>,
    /// The boost factor for this query.
    boost: f32,
    /// Rewrite method for multi-term expansion
    rewrite_method: RewriteMethod,
}

impl RegexpQuery {
    /// Create a new regexp query.
    pub fn new<S: Into<String>>(field: S, pattern: S) -> Result<Self> {
        let field = field.into();
        let pattern = pattern.into();
        let regex = Regex::new(&pattern)
            .map_err(|e| LaurusError::analysis(format!("Invalid regexp pattern: {e}")))?;

        Ok(RegexpQuery {
            field,
            pattern,
            regex: Some(Arc::new(regex)),
            boost: 1.0,
            rewrite_method: RewriteMethod::default(),
        })
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

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the pattern.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }
}

impl MultiTermQuery for RegexpQuery {
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
            let regex_automaton = if let Some(regex) = &self.regex {
                RegexAutomaton::from_regex(regex.as_ref().clone(), self.pattern.clone())
            } else {
                RegexAutomaton::new(&self.pattern)?
            };

            let terms_enum = AutomatonTermsEnum::new(terms.iterator()?, regex_automaton);
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

impl Query for RegexpQuery {
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
            "RegexpQuery(field: {}, pattern: {})",
            self.field, self.pattern
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.pattern.is_empty())
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
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

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        if self.pattern.is_empty() || field.is_some_and(|f| f != self.field) {
            return;
        }
        // `regex` is `None` after deserialization; recompile like
        // `get_terms_enum` does.
        let regex = match &self.regex {
            Some(regex) => Arc::clone(regex),
            None => match Regex::new(&self.pattern) {
                Ok(regex) => Arc::new(regex),
                Err(_) => return,
            },
        };
        out.push(HighlightTerm::Regex(regex));
    }

    fn cache_key(&self) -> Option<String> {
        // Field + pattern + rewrite method determine the matched set; boost is
        // score-only and excluded. The compiled automaton is derived from the
        // pattern, so it need not appear in the key.
        Some(format!(
            "regexp|{:?}|{:?}|{:?}",
            self.field, self.pattern, self.rewrite_method
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regexp_query_creation() {
        let query = RegexpQuery::new("field", "^abc.*").unwrap();
        assert_eq!(query.field(), "field");
        assert_eq!(query.pattern(), "^abc.*");
        assert_eq!(query.boost(), 1.0);
    }

    #[test]
    fn test_prefix_extraction() {
        use crate::lexical::index::inverted::core::automaton::{Automaton, RegexAutomaton};

        // Test via RegexAutomaton
        let automaton = RegexAutomaton::new("^abc.*").unwrap();
        assert_eq!(automaton.initial_seek_term().as_deref(), Some("abc"));

        let automaton = RegexAutomaton::new("abc.*").unwrap();
        assert_eq!(automaton.initial_seek_term(), None); // No anchor

        let automaton = RegexAutomaton::new("^abc\\.def").unwrap();
        assert_eq!(automaton.initial_seek_term().as_deref(), Some("abc.def"));

        let automaton = RegexAutomaton::new("^a(b|c)").unwrap();
        assert_eq!(automaton.initial_seek_term().as_deref(), Some("a"));
    }
}
