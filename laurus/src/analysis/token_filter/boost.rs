//! Boost filter implementation.
//!
//! This module provides a filter that multiplies token boost factors,
//! allowing you to increase or decrease the scoring weight of tokens.
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::token_filter::Filter;
//! use laurus::analysis::token_filter::boost::BoostFilter;
//! use laurus::analysis::token::Token;
//!
//! let filter = BoostFilter::new(2.0);
//! let tokens = vec![Token::new("important", 0)];
//! let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
//!     .unwrap()
//!     .collect();
//!
//! // Token boost multiplied by 2.0
//! assert_eq!(result[0].boost, 2.0);
//! ```

use crate::analysis::token::TokenStream;
use crate::analysis::token_filter::Filter;
use crate::error::Result;

/// A filter that applies a boost multiplier to all tokens.
///
/// This filter multiplies each token's boost factor by a constant value,
/// allowing you to increase or decrease the scoring importance of tokens
/// from a particular field or analysis stage.
///
/// The boost takes effect when a query is analyzed: the query parser
/// multiplies a query term's boost by its tokens' boost (Issue #1701), as
/// Lucene's `QueryBuilder` applies a token's `BoostAttribute`. Indexing does
/// not use a token's boost. In a schema, the factor of a `boost` token
/// filter must be finite and at least 0
/// ([`validate_analyzer_definition`](crate::analysis::analyzer::registry::validate_analyzer_definition)).
///
/// # Use Cases
///
/// - Boosting title field tokens vs body field tokens
/// - Emphasizing or de-emphasizing certain token sources
/// - Implementing field-level relevance weighting
///
/// # Examples
///
/// ```
/// use laurus::analysis::token_filter::Filter;
/// use laurus::analysis::token_filter::boost::BoostFilter;
/// use laurus::analysis::token::Token;
///
/// // Double the weight of all tokens
/// let filter = BoostFilter::new(2.0);
/// let tokens = vec![
///     Token::new("word", 0),
///     Token::new("another", 1).with_boost(1.5)
/// ];
///
/// let result: Vec<_> = filter.filter(Box::new(tokens.into_iter()))
///     .unwrap()
///     .collect();
///
/// assert_eq!(result[0].boost, 2.0);    // 1.0 * 2.0
/// assert_eq!(result[1].boost, 3.0);    // 1.5 * 2.0
/// ```
#[derive(Clone, Debug)]
pub struct BoostFilter {
    boost: f32,
}

impl BoostFilter {
    /// Create a new boost filter with the given boost factor.
    ///
    /// # Arguments
    ///
    /// * `boost` - The multiplier to apply to token boosts
    ///
    /// # Examples
    ///
    /// ```
    /// use laurus::analysis::token_filter::boost::BoostFilter;
    ///
    /// let filter = BoostFilter::new(1.5);
    /// assert_eq!(filter.boost(), 1.5);
    /// ```
    pub fn new(boost: f32) -> Self {
        BoostFilter { boost }
    }

    /// Get the boost factor.
    pub fn boost(&self) -> f32 {
        self.boost
    }
}

impl Filter for BoostFilter {
    fn filter(&self, tokens: TokenStream) -> Result<TokenStream> {
        let filtered_tokens = tokens
            .map(|token| {
                if token.is_stopped() {
                    token
                } else {
                    let boost = token.boost;
                    token.with_boost(boost * self.boost)
                }
            })
            .collect::<Vec<_>>();

        Ok(Box::new(filtered_tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "boost"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::token::Token;

    #[test]
    fn test_boost_filter() {
        let filter = BoostFilter::new(2.0);
        let tokens = vec![
            Token::new("hello", 0),
            Token::new("world", 1).with_boost(1.5),
            Token::new("test", 2).stop(),
        ];
        let token_stream = Box::new(tokens.into_iter());

        let result: Vec<Token> = filter.filter(token_stream).unwrap().collect();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].boost, 2.0);
        assert_eq!(result[1].boost, 3.0);
        assert_eq!(result[2].boost, 1.0); // Stopped tokens are not processed
    }

    #[test]
    fn test_filter_name() {
        assert_eq!(BoostFilter::new(1.0).name(), "boost");
    }
}
