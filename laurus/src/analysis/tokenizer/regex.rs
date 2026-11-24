//! Regex-based tokenizer implementation.
//!
//! This module provides a flexible tokenizer that uses regular expressions to
//! identify tokens. It supports both matching mode (extract matches) and gaps
//! mode (extract text between matches).
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::tokenizer::Tokenizer;
//! use laurus::analysis::tokenizer::regex::RegexTokenizer;
//!
//! // Extract word characters
//! let tokenizer = RegexTokenizer::new().unwrap();
//! let tokens: Vec<_> = tokenizer.tokenize("hello-world").unwrap().collect();
//! assert_eq!(tokens.len(), 2);
//! assert_eq!(tokens[0].text, "hello");
//! assert_eq!(tokens[1].text, "world");
//!
//! // Extract gaps (text between whitespace)
//! let tokenizer = RegexTokenizer::with_gaps(r"\s+").unwrap();
//! let tokens: Vec<_> = tokenizer.tokenize("hello world").unwrap().collect();
//! assert_eq!(tokens[0].text, "hello");
//! assert_eq!(tokens[1].text, "world");
//! ```

use std::sync::Arc;

use regex::Regex;

use crate::analysis::token::{Token, TokenStream};
use crate::analysis::tokenizer::Tokenizer;
use crate::error::{LaurusError, Result};

/// A regex-based tokenizer that extracts tokens using regular expressions.
///
/// This is the default tokenizer and is equivalent to Whoosh's RegexTokenizer.
/// It supports two modes:
///
/// - **Matching mode** (default): Extracts text that matches the pattern
/// - **Gaps mode**: Extracts text between pattern matches
///
/// # Default Pattern
///
/// The default pattern is `r"\w+"`, which matches sequences of word characters
/// (letters, digits, and underscores).
///
/// # Examples
///
/// ## Basic Usage
///
/// ```
/// use laurus::analysis::tokenizer::Tokenizer;
/// use laurus::analysis::tokenizer::regex::RegexTokenizer;
///
/// let tokenizer = RegexTokenizer::new().unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("hello world").unwrap().collect();
/// assert_eq!(tokens.len(), 2);
/// ```
///
/// ## Custom Pattern
///
/// ```
/// use laurus::analysis::tokenizer::regex::RegexTokenizer;
/// use laurus::analysis::tokenizer::Tokenizer;
///
/// // Extract email-like tokens
/// let tokenizer = RegexTokenizer::with_pattern(r"\w+@\w+\.\w+").unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("contact user@example.com").unwrap().collect();
/// assert_eq!(tokens[0].text, "user@example.com");
/// ```
///
/// ## Gaps Mode
///
/// ```
/// use laurus::analysis::tokenizer::regex::RegexTokenizer;
/// use laurus::analysis::tokenizer::Tokenizer;
///
/// // Extract text between punctuation
/// let tokenizer = RegexTokenizer::with_gaps(r"[,.]").unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("one,two.three").unwrap().collect();
/// assert_eq!(tokens.len(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct RegexTokenizer {
    /// The regex pattern used to extract tokens
    pattern: Arc<Regex>,
    /// Whether to extract gaps (text between matches) instead of matches
    gaps: bool,
}

impl RegexTokenizer {
    /// Create a new regex tokenizer with the default pattern.
    ///
    /// The default pattern `r"\w+"` matches sequences of word characters.
    pub fn new() -> Result<Self> {
        Self::with_pattern(r"\w+")
    }

    /// Create a new regex tokenizer with a custom pattern.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] if `pattern` is not a valid
    /// regular expression.
    pub fn with_pattern(pattern: &str) -> Result<Self> {
        let regex = Regex::new(pattern)
            .map_err(|e| LaurusError::invalid_argument(format!("Invalid regex pattern: {e}")))?;

        Ok(RegexTokenizer {
            pattern: Arc::new(regex),
            gaps: false,
        })
    }

    /// Create a tokenizer that extracts gaps (text between matches) instead of matches.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] if `pattern` is not a valid
    /// regular expression.
    pub fn with_gaps(pattern: &str) -> Result<Self> {
        let regex = Regex::new(pattern)
            .map_err(|e| LaurusError::invalid_argument(format!("Invalid regex pattern: {e}")))?;

        Ok(RegexTokenizer {
            pattern: Arc::new(regex),
            gaps: true,
        })
    }

    /// Get the regex pattern used by this tokenizer.
    pub fn pattern(&self) -> &str {
        self.pattern.as_str()
    }

    /// Check if this tokenizer extracts gaps.
    pub fn gaps(&self) -> bool {
        self.gaps
    }
}

impl Default for RegexTokenizer {
    fn default() -> Self {
        Self::new().expect("Default regex pattern should be valid")
    }
}

impl Tokenizer for RegexTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenStream> {
        let pattern = Arc::clone(&self.pattern);
        let gaps = self.gaps;
        let text = text.to_owned();

        let tokens = if gaps {
            // Extract gaps between matches
            let mut tokens = Vec::new();
            let mut last_end = 0;
            let mut position = 0;

            for mat in pattern.find_iter(&text) {
                if mat.start() > last_end {
                    let gap_text = &text[last_end..mat.start()];
                    if !gap_text.is_empty() {
                        tokens.push(Token::with_offsets(
                            gap_text,
                            position,
                            last_end,
                            mat.start(),
                        ));
                        position += 1;
                    }
                }
                last_end = mat.end();
            }

            // Add final gap if any
            if last_end < text.len() {
                let gap_text = &text[last_end..];
                if !gap_text.is_empty() {
                    tokens.push(Token::with_offsets(
                        gap_text,
                        position,
                        last_end,
                        text.len(),
                    ));
                }
            }

            tokens
        } else {
            // Extract matches
            pattern
                .find_iter(&text)
                .enumerate()
                .map(|(position, mat)| {
                    Token::with_offsets(mat.as_str(), position, mat.start(), mat.end())
                })
                .collect()
        };

        Ok(Box::new(tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "regex"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_tokenizer() {
        let tokenizer = RegexTokenizer::new().unwrap();
        let tokens: Vec<Token> = tokenizer.tokenize("hello world").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[0].start_offset, 0);
        assert_eq!(tokens[0].end_offset, 5);

        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[1].position, 1);
        assert_eq!(tokens[1].start_offset, 6);
        assert_eq!(tokens[1].end_offset, 11);
    }

    #[test]
    fn test_regex_tokenizer_with_gaps() {
        let tokenizer = RegexTokenizer::with_gaps(r"\s+").unwrap();
        let tokens: Vec<Token> = tokenizer.tokenize("hello world").unwrap().collect();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_tokenizer_name() {
        assert_eq!(RegexTokenizer::new().unwrap().name(), "regex");
    }
}
