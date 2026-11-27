//! Wildcard query implementation for pattern matching.

use std::fmt::Debug;
use std::sync::Arc;

use regex::Regex;

use crate::error::Result;
use crate::lexical::index::inverted::core::automaton::{Automaton, RegexAutomaton};
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::multi_term::{
    MultiTermQuery, RewriteMethod, multi_term_matcher, multi_term_scorer, rewrite_multi_term,
    validate_multi_term,
};
use crate::lexical::query::scorer::Scorer;
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::LexicalIndexReader;

/// A query that matches documents containing terms that match a wildcard pattern.
///
/// A pattern reads as in Lucene's `WildcardQuery`, whole terms against the
/// whole pattern:
/// - `*` matches zero or more characters, and `?` exactly one, line breaks
///   included
/// - a backslash takes the next character literally, whatever it is (`\*`,
///   `\?` and `\\` match `*`, `?` and `\`, and `\d` matches `d`); at the end
///   of a pattern it is a backslash itself
/// - every other character matches itself, a special character of a regexp
///   (`.`, `+`, `(`, `[`, `|` and others) too
///
/// No pattern is an error (Issue #1745), short of one so long that its
/// regexp exceeds the size limit of the `regex` crate (100,000 `?`, for
/// example; see [`WildcardQuery::new`]). The backslash used to be passed on
/// to the regexp with the next character, so `\d` matched a digit and `\p`
/// failed, and `*` and `?` did not match a line break.
#[derive(Debug, Clone)]
pub struct WildcardQuery {
    /// The field to search in.
    field: String,
    /// The wildcard pattern.
    pattern: String,
    /// The compiled regex for matching.
    regex: Arc<Regex>,
    /// The boost factor for this query.
    /// The boost factor for this query.
    boost: f32,
    /// Rewrite method for multi-term expansion
    rewrite_method: RewriteMethod,
}

impl WildcardQuery {
    /// Create a new wildcard query.
    ///
    /// # Errors
    ///
    /// Returns an analysis error only when the pattern is so long that its
    /// regexp exceeds the size limit of the `regex` crate.
    pub fn new<S: Into<String>>(field: S, pattern: S) -> Result<Self> {
        let field = field.into();
        let pattern = pattern.into();
        let regex_pattern = Self::compile_pattern(&pattern);
        let regex = Regex::new(&regex_pattern).map_err(|e| {
            crate::error::LaurusError::analysis(format!("Invalid wildcard pattern: {e}"))
        })?;

        Ok(WildcardQuery {
            field,
            pattern,
            regex: Arc::new(regex),
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

    /// Get the wildcard pattern.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Get the rewrite method.
    pub fn rewrite_method(&self) -> RewriteMethod {
        self.rewrite_method
    }

    /// The regexp of a wildcard pattern, read as in Lucene's
    /// `WildcardQuery.toAutomaton` (see [`WildcardQuery`]).
    ///
    /// It matches whole strings (`\A` and `\z`), and `.` matches a line break
    /// too (`s`). Every literal character goes in through `regex::escape`, so
    /// the regexp is never invalid.
    fn compile_pattern(pattern: &str) -> String {
        fn push_literal(regex: &mut String, c: char) {
            regex.push_str(&regex::escape(c.encode_utf8(&mut [0; 4])));
        }

        let mut regex = String::from(r"(?s)\A");
        let mut chars = pattern.chars();
        while let Some(c) = chars.next() {
            match c {
                '*' => regex.push_str(".*"),
                '?' => regex.push('.'),
                // The next character literally; a trailing backslash is one.
                '\\' => push_literal(&mut regex, chars.next().unwrap_or('\\')),
                c => push_literal(&mut regex, c),
            }
        }
        regex.push_str(r"\z");
        regex
    }

    /// Check if a term matches the wildcard pattern.
    pub fn matches(&self, term: &str) -> bool {
        self.regex.is_match(term)
    }
}

impl MultiTermQuery for WildcardQuery {
    fn field(&self) -> &str {
        &self.field
    }

    fn rewrite_method(&self) -> RewriteMethod {
        self.rewrite_method
    }

    fn automaton(&self) -> Result<Box<dyn Automaton>> {
        let pattern = Self::compile_pattern(&self.pattern);
        Ok(Box::new(RegexAutomaton::from_regex(
            self.regex.as_ref().clone(),
            pattern,
        )))
    }
}

impl Query for WildcardQuery {
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
            "WildcardQuery(field: {}, pattern: {})",
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
        // Wildcard queries can be expensive
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
        if !self.pattern.is_empty() && field.is_none_or(|f| f == self.field) {
            out.push(HighlightTerm::Regex(Arc::clone(&self.regex)));
        }
    }

    fn cache_key(&self) -> Option<String> {
        // Field + pattern determine the matched set; the boost and the rewrite
        // method are score-only and excluded. The compiled regex is derived
        // from the pattern, so it need not appear in the key.
        Some(format!("wildcard|{:?}|{:?}", self.field, self.pattern))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wildcard_query_creation() {
        let query = WildcardQuery::new("content", "hello*").unwrap();

        assert_eq!(query.field(), "content");
        assert_eq!(query.pattern(), "hello*");
        assert_eq!(query.boost(), 1.0);
    }

    #[test]
    fn test_wildcard_query_with_boost() {
        let query = WildcardQuery::new("content", "test?")
            .unwrap()
            .with_boost(2.5);

        assert_eq!(query.boost(), 2.5);
    }

    #[test]
    fn test_wildcard_pattern_compilation() {
        // Test simple wildcard
        let query = WildcardQuery::new("field", "hello*").unwrap();
        assert!(query.matches("hello"));
        assert!(query.matches("helloworld"));
        assert!(!query.matches("hell"));

        // Test question mark
        let query = WildcardQuery::new("field", "h?llo").unwrap();
        assert!(query.matches("hello"));
        assert!(query.matches("hallo"));
        assert!(query.matches("hxllo"));
        assert!(!query.matches("heello"));

        // Test combination
        let query = WildcardQuery::new("field", "h*l?o").unwrap();
        assert!(query.matches("hello"));
        assert!(query.matches("hallo"));
        assert!(query.matches("heeello"));
        assert!(query.matches("hllo")); // Actually matches because ? can be 'l'
    }

    #[test]
    fn test_escaped_wildcards() {
        let query = WildcardQuery::new("field", "hello\\*world").unwrap();
        assert!(query.matches("hello*world"));
        assert!(!query.matches("helloworld"));
        assert!(!query.matches("hello123world"));

        let query = WildcardQuery::new("field", "hello\\?world").unwrap();
        assert!(query.matches("hello?world"));
        assert!(!query.matches("helloxworld"));
    }

    #[test]
    fn test_special_regex_characters() {
        let query = WildcardQuery::new("field", "hello.world").unwrap();
        assert!(query.matches("hello.world"));
        assert!(!query.matches("helloxworld"));

        let query = WildcardQuery::new("field", "hello+world").unwrap();
        assert!(query.matches("hello+world"));
        assert!(!query.matches("helloworld"));
    }

    /// Issue #1745: a pattern reads as in Lucene's `WildcardQuery`. A
    /// backslash takes the next character literally, and at the end is a
    /// backslash itself; `*` and `?` match any characters, line breaks
    /// included; every other character is itself, a special character of a
    /// regexp too.
    #[test]
    fn a_pattern_reads_as_in_lucene() {
        let mut wrong = Vec::new();
        for (pattern, matching, not_matching) in [
            (r"te\dt", "tedt", "te1t"),
            (r"\a*", "abc", "\x07bc"),
            (r"a\nb", "anb", "a\nb"),
            (r"te\", "te\\", "te$"),
            (r"te\*", "te*", "tex"),
            (r"a\\b", "a\\b", "ab"),
            ("a*b", "a\nb", "a\nc"),
            ("a?b", "a\nb", "ab"),
            ("a?", "a\r", "a"),
            ("?", "é", "ée"),
            ("ü?", "üé", "ü"),
            ("te**", "te", "t"),
            ("a.b", "a.b", "axb"),
            ("a+b", "a+b", "aab"),
            ("(a)", "(a)", "a"),
            ("a|b", "a|b", "a"),
            ("#a", "#a", "a"),
            ("a&b", "a&b", "ab"),
            ("a~b", "a~b", "ab"),
            ("a-b", "a-b", "ab"),
        ] {
            let query = WildcardQuery::new("field", pattern).unwrap();
            if !query.matches(matching) || query.matches(not_matching) {
                wrong.push(format!(
                    "{pattern:?}: matches {matching:?} {}, matches {not_matching:?} {}",
                    query.matches(matching),
                    query.matches(not_matching)
                ));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1745: the term dictionary is still sought from the literal
    /// start of a pattern, escapes read as their characters.
    #[test]
    fn the_seek_term_is_the_literal_start_of_the_pattern() {
        let mut wrong = Vec::new();
        for (pattern, expected) in [
            ("te*", Some("te")),
            (r"te\*x*", Some("te*x")),
            (r"\a*", Some("a")),
            (r"te\", Some("te\\")),
            ("a.b*", Some("a.b")),
            ("über?", Some("über")),
            ("*a", None),
            ("?a", None),
        ] {
            let query = WildcardQuery::new("field", pattern).unwrap();
            let seek = query.automaton().unwrap().initial_seek_term();
            if seek.as_deref() != expected {
                wrong.push(format!("{pattern:?}: {seek:?}, expected {expected:?}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }
}
