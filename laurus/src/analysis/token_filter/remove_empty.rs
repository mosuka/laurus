//! Remove empty filter implementation.
//!
//! This module provides a filter that removes empty tokens and stopped tokens
//! from the stream, cleaning up the token flow before indexing.
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::token_filter::Filter;
//! use laurus::analysis::token_filter::remove_empty::RemoveEmptyFilter;
//! use laurus::analysis::token::Token;
//!
//! let filter = RemoveEmptyFilter::new();
//! let tokens = vec![
//!     Token::new("hello", 0),
//!     Token::new("", 1),         // Will be removed
//!     Token::new("world", 2)
//! ];
//!
//! let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
//!     .unwrap()
//!     .collect();
//!
//! assert_eq!(result.len(), 2);
//! assert_eq!(result[0].text, "hello");
//! assert_eq!(result[1].text, "world");
//! ```

use crate::analysis::token::{TokenStream, remove_tokens};
use crate::analysis::token_filter::Filter;
use crate::error::Result;

/// A filter that removes empty tokens from the stream.
///
/// This filter removes two types of tokens:
/// - Tokens with empty text (`text.is_empty()`)
/// - Tokens marked as stopped
///
/// This is typically used near the end of an analysis pipeline to clean up
/// tokens that have been emptied or stopped by previous filters.
///
/// A removed token leaves no position, and the token graph stays well
/// formed, as with [`StopFilter`](crate::analysis::token_filter::stop::StopFilter).
///
/// # Examples
///
/// ```
/// use laurus::analysis::token_filter::Filter;
/// use laurus::analysis::token_filter::remove_empty::RemoveEmptyFilter;
/// use laurus::analysis::token::Token;
///
/// let filter = RemoveEmptyFilter::new();
/// let tokens = vec![
///     Token::new("valid", 0),
///     Token::new("", 1),              // Removed: empty
///     Token::new("stopped", 2).stop(), // Removed: stopped
///     Token::new("kept", 3)
/// ];
///
/// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
///     .unwrap()
///     .collect();
///
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0].text, "valid");
/// assert_eq!(result[1].text, "kept");
/// ```
#[derive(Clone, Debug, Default)]
pub struct RemoveEmptyFilter;

impl RemoveEmptyFilter {
    /// Create a new remove empty filter.
    pub fn new() -> Self {
        RemoveEmptyFilter
    }
}

impl Filter for RemoveEmptyFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let filtered_tokens = remove_tokens(tokens.collect(), |token| {
            !token.is_stopped() && !token.text.is_empty()
        });

        Ok(Box::new(filtered_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "remove_empty"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_remove_empty_filter() {
        let filter = RemoveEmptyFilter::new();
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("", 1),
            Token::new("world", 2),
            Token::new("test", 3).stop(),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[1].text, "world");
    }

    #[test]
    fn test_filter_name() {
        assert_eq!(RemoveEmptyFilter::new().name(), "remove_empty");
    }

    /// #1259: "a", stacked on the stopped "the", and "b", stacked on an
    /// empty token, take the removed tokens' positions instead of moving
    /// back onto the word before them.
    #[test]
    fn a_token_stacked_on_a_removed_token_keeps_its_position() {
        let tokens = vec![
            Token::new("big", 0),
            Token::new("the", 1).stop(),
            Token::new("a", 1).with_position_increment(0),
            Token::new("", 2),
            Token::new("b", 2).with_position_increment(0),
            Token::new("dog", 3),
        ];
        let result: Vec<Token> = RemoveEmptyFilter::new()
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect();
        let texts: Vec<&str> = result.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, ["big", "a", "b", "dog"]);
        assert_eq!(
            crate::analysis::token::token_positions(&result),
            vec![0, 1, 2, 3]
        );
    }
}
