//! Regular expression query implementation.

use std::fmt::Debug;
use std::sync::Arc;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::lexical::index::inverted::core::automaton::{Automaton, RegexAutomaton};
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::multi_term::{
    MultiTermQuery, RewriteMethod, multi_term_matcher, multi_term_scorer, rewrite_multi_term,
    validate_multi_term,
};
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

    fn automaton(&self) -> Result<Box<dyn Automaton>> {
        Ok(Box::new(match &self.regex {
            Some(regex) => RegexAutomaton::from_regex(regex.as_ref().clone(), self.pattern.clone()),
            // A deserialized query has not been compiled.
            None => RegexAutomaton::new(&self.pattern)?,
        }))
    }
}

impl Query for RegexpQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

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
        rewrite_multi_term(self, reader)
    }

    fn validate_boosts(&self) -> Result<()> {
        validate_multi_term(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        if self.pattern.is_empty() || field.is_some_and(|f| f != self.field) {
            return;
        }
        // `regex` is `None` after deserialization; recompile like
        // `automaton` does.
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
        // Field + pattern determine the matched set; the boost and the rewrite
        // method are score-only and excluded. The compiled automaton is derived
        // from the pattern, so it need not appear in the key.
        Some(format!("regexp|{:?}|{:?}", self.field, self.pattern))
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
