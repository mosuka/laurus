//! Wildcard query implementation for pattern matching.

use std::fmt::Debug;
use std::sync::Arc;

use regex::Regex;

use crate::error::Result;
use crate::lexical::index::inverted::core::terms::{TermDictionaryAccess, TermsEnum};
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::multi_term::{MultiTermQuery, RewriteMethod};
use crate::lexical::query::scorer::Scorer;
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::LexicalIndexReader;

/// A query that matches documents containing terms that match a wildcard pattern.
///
/// Supports the following wildcards:
/// - `*` matches zero or more characters
/// - `?` matches exactly one character
/// - `\*` and `\?` match literal `*` and `?` characters
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
    pub fn new<S: Into<String>>(field: S, pattern: S) -> Result<Self> {
        let field = field.into();
        let pattern = pattern.into();
        let regex_pattern = Self::compile_pattern(&pattern)?;
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

    /// Compile a wildcard pattern into a regex string.
    fn compile_pattern(pattern: &str) -> Result<String> {
        let mut regex_pattern = String::new();
        regex_pattern.push('^'); // Match from the beginning

        let chars: Vec<char> = pattern.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            match chars[i] {
                '\\' => {
                    // Handle escape sequences
                    if i + 1 < chars.len() {
                        match chars[i + 1] {
                            '*' => {
                                regex_pattern.push_str("\\*");
                                i += 1; // Skip the escaped character
                            }
                            '?' => {
                                regex_pattern.push_str("\\?");
                                i += 1; // Skip the escaped character
                            }
                            c => {
                                // Other escaped characters
                                regex_pattern.push('\\');
                                regex_pattern.push(c);
                                i += 1; // Skip the escaped character
                            }
                        }
                    } else {
                        regex_pattern.push('\\');
                    }
                }
                '*' => {
                    regex_pattern.push_str(".*");
                }
                '?' => {
                    regex_pattern.push('.');
                }
                // Regex special characters that need escaping
                '^' | '$' | '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '|' => {
                    regex_pattern.push('\\');
                    regex_pattern.push(chars[i]);
                }
                c => {
                    regex_pattern.push(c);
                }
            }
            i += 1;
        }

        regex_pattern.push('$'); // Match to the end
        Ok(regex_pattern)
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

    fn get_terms_enum(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<Option<Box<dyn TermsEnum>>> {
        if let Some(inverted_reader) = reader.as_any().downcast_ref::<InvertedIndexReader>()
            && let Some(terms) = inverted_reader.terms(&self.field)?
        {
            let regex_pattern = Self::compile_pattern(&self.pattern)?;
            let regex_automaton =
                crate::lexical::index::inverted::core::automaton::RegexAutomaton::new(
                    &regex_pattern,
                )?;

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
        // Now mostly a fallback, or could reuse rewrite logic if we exposed it differently.
        // But the trait default implementation of rewrite() calls get_terms_enum().
        // So enumerate_terms is only called if get_terms_enum returns None,
        // OR if needed directly.
        // Let's keep existing implementation for compatibility or use get_terms_enum manually.
        if let Some(mut terms_enum) = self.get_terms_enum(reader)? {
            let mut results = Vec::new();
            let max = self.rewrite_method.max_expansions();

            // Simple scan limited by max_expansions (arbitrary selection if not scoring)
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

impl Query for WildcardQuery {
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
        if !self.pattern.is_empty() && field.is_none_or(|f| f == self.field) {
            out.push(HighlightTerm::Regex(Arc::clone(&self.regex)));
        }
    }

    fn cache_key(&self) -> Option<String> {
        // Field + pattern + rewrite method determine the matched set; boost is
        // score-only and excluded. The compiled regex is derived from the
        // pattern, so it need not appear in the key.
        Some(format!(
            "wildcard|{:?}|{:?}|{:?}",
            self.field, self.pattern, self.rewrite_method
        ))
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
}
