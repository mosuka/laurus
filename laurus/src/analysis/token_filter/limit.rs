//! Limit filter implementation.
//!
//! This module provides a filter that limits the maximum number of tokens
//! in a stream. This is useful for truncating long documents or controlling
//! indexing costs.
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::token_filter::Filter;
//! use laurus::analysis::token_filter::limit::LimitFilter;
//! use laurus::analysis::token::Token;
//!
//! let filter = LimitFilter::new(3);
//! let tokens = vec![
//!     Token::new("one", 0),
//!     Token::new("two", 1),
//!     Token::new("three", 2),
//!     Token::new("four", 3),
//!     Token::new("five", 4),
//! ];
//!
//! let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
//!     .unwrap()
//!     .collect();
//!
//! // Only first 3 tokens are kept
//! assert_eq!(result.len(), 3);
//! ```

use crate::analysis::token::{Token, TokenStream, token_arcs};
use crate::analysis::token_filter::Filter;
use crate::error::Result;

/// A filter that limits the number of tokens in the stream.
///
/// This filter truncates the token stream after a specified number of tokens,
/// which is useful for:
///
/// - Controlling indexing costs for large documents
/// - Implementing "index first N tokens only" strategies
/// - Testing and development with truncated input
/// - Implementing document preview features
///
/// A cut inside a token graph, such as a multi-word synonym, cuts every
/// path there: the kept paths end together, one position after the last
/// kept token's.
///
/// # Examples
///
/// ```
/// use laurus::analysis::token_filter::Filter;
/// use laurus::analysis::token_filter::limit::LimitFilter;
/// use laurus::analysis::token::Token;
///
/// let filter = LimitFilter::new(2);
/// let tokens = vec![
///     Token::new("first", 0),
///     Token::new("second", 1),
///     Token::new("third", 2),
/// ];
///
/// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
///     .unwrap()
///     .collect();
///
/// assert_eq!(result.len(), 2);
/// assert_eq!(result[0].text, "first");
/// assert_eq!(result[1].text, "second");
/// ```
#[derive(Clone, Debug)]
pub struct LimitFilter {
    limit: usize,
}

impl LimitFilter {
    /// Create a new limit filter with the given limit.
    ///
    /// # Arguments
    ///
    /// * `limit` - The maximum number of tokens to pass through
    ///
    /// # Examples
    ///
    /// ```
    /// use laurus::analysis::token_filter::limit::LimitFilter;
    ///
    /// let filter = LimitFilter::new(100);
    /// assert_eq!(filter.limit(), 100);
    /// ```
    pub fn new(limit: usize) -> Self {
        LimitFilter { limit }
    }

    /// Get the limit.
    pub fn limit(&self) -> usize {
        self.limit
    }
}

impl Filter for LimitFilter {
    fn filter(&self, mut tokens: TokenStream) -> Result<TokenStream> {
        let mut limited_tokens: Vec<Token> = tokens.by_ref().take(self.limit).collect();
        if tokens.next().is_some() {
            // Arcs past the last kept node lead only to cut tokens, so they
            // end at the node after it.
            let arcs = token_arcs(&limited_tokens);
            if let Some(&(last, _)) = arcs.last() {
                let end = last + 1;
                for (token, (from, to)) in limited_tokens.iter_mut().zip(arcs) {
                    if to > end {
                        token.position_length = (end - from) as usize;
                    }
                }
            }
        }
        Ok(Box::new(limited_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "limit"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_limit_filter() {
        let filter = LimitFilter::new(2);
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("world", 1),
            Token::new("test", 2),
            Token::new("limit", 3),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[1].text, "world");
    }

    #[test]
    fn test_filter_name() {
        assert_eq!(LimitFilter::new(10).name(), "limit");
    }

    fn arc(text: &str, increment: usize, length: usize) -> Token {
        Token::new(text, 0)
            .with_position_increment(increment)
            .with_position_length(length)
    }

    /// Each token as an arc `(text, from, to)` of the token graph.
    fn arcs(tokens: &[Token]) -> Vec<(&str, u32, u32)> {
        tokens
            .iter()
            .zip(crate::analysis::token::token_positions(tokens))
            .map(|(t, from)| (t.text.as_str(), from, from + t.position_length as u32))
            .collect()
    }

    /// "ml is" with "machine learning".
    fn ml_is() -> Vec<Token> {
        vec![
            arc("ml", 1, 2),
            arc("machine", 0, 1),
            arc("learning", 1, 1),
            arc("is", 1, 1),
        ]
    }

    fn limit(limit: usize, tokens: Vec<Token>) -> Vec<Token> {
        LimitFilter::new(limit)
            .filter(Box::new(tokens.into_iter()))
            .unwrap()
            .collect()
    }

    /// #1259: cut inside a graph, every kept path ends at the node after
    /// the last kept one, so "machine" is still a path next to "ml".
    #[test]
    fn a_cut_inside_a_graph_ends_every_kept_path_at_one_node() {
        assert_eq!(
            arcs(&limit(2, ml_is())),
            vec![("ml", 0, 1), ("machine", 0, 1)]
        );
    }

    /// A cut after a whole graph, or no cut at all, leaves it as it is.
    #[test]
    fn a_cut_outside_a_graph_leaves_it_unchanged() {
        let whole = ml_is();
        assert_eq!(limit(3, ml_is()), whole[..3]);
        assert_eq!(limit(4, ml_is()), whole);
        assert_eq!(limit(10, ml_is()), whole);
    }
}
