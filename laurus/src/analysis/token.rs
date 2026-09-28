//! Token types and utilities for text analysis.
//!
//! This module defines the core data structures for representing text tokens,
//! which are the fundamental units that flow through the analysis pipeline.
//!
//! # Core Types
//!
//! - [`Token`] - A single analyzed token with text, position, and metadata
//! - [`TokenType`] - Classification of token content (alphanumeric, CJK, etc.)
//! - [`TokenMetadata`] - Additional metadata attached to tokens
//! - [`TokenStream`] - Type alias for boxed iterator of tokens
//!
//! # Token Graphs
//!
//! Tokens support graph structures through `position_increment` and `position_length`
//! fields, enabling proper handling of synonyms and multi-word phrases:
//!
//! ```text
//! Input: "machine learning"
//! With synonym: "ml"
//!
//! Token Graph:
//!   Position 0: "machine" (pos_inc=1, pos_len=1)
//!   Position 0: "ml"      (pos_inc=0, pos_len=2)  ← same position, spans 2
//!   Position 1: "learning"(pos_inc=1, pos_len=1)
//! ```
//!
//! # Examples
//!
//! Creating a simple token:
//!
//! ```
//! use laurus::analysis::token::Token;
//!
//! let token = Token::new("hello", 0);
//! assert_eq!(token.text, "hello");
//! assert_eq!(token.position, 0);
//! assert_eq!(token.boost, 1.0);
//! ```
//!
//! Creating a token with offsets:
//!
//! ```
//! use laurus::analysis::token::Token;
//!
//! let token = Token::with_offsets("world", 1, 6, 11);
//! assert_eq!(token.text, "world");
//! assert_eq!(token.start_offset, 6);
//! assert_eq!(token.end_offset, 11);
//! ```
//!
//! Working with token metadata:
//!
//! ```
//! use laurus::analysis::token::{Token, TokenType};
//!
//! let token = Token::new("hello", 0)
//!     .with_token_type(TokenType::Alphanum)
//!     .with_boost(1.5);
//!
//! assert_eq!(token.boost, 1.5);
//! assert_eq!(
//!     token.metadata.as_ref().unwrap().token_type,
//!     Some(TokenType::Alphanum)
//! );
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

/// A token represents a single unit of text after tokenization.
///
/// This is the fundamental unit that flows through the analysis pipeline.
/// It contains the text content, position information, and metadata.
///
/// # Fields
///
/// - `text` - The token's text content
/// - `position` - Position in the token stream (0-based)
/// - `start_offset` / `end_offset` - Byte offsets in original text
/// - `boost` - Scoring weight multiplier (default: 1.0)
/// - `stopped` - Whether the token was marked for removal
/// - `position_increment` - Position relative to previous token (default: 1)
/// - `position_length` - Number of positions this token spans (default: 1)
/// - `metadata` - Optional additional metadata
///
/// # Examples
///
/// ```
/// use laurus::analysis::token::Token;
///
/// // Simple token
/// let mut token = Token::new("search", 0);
/// assert_eq!(token.text, "search");
/// assert_eq!(token.position, 0);
///
/// // Token with boost
/// token = token.with_boost(2.0);
/// assert_eq!(token.boost, 2.0);
///
/// // Mark token as stopped
/// token = token.stop();
/// assert!(token.is_stopped());
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Token {
    /// The text content of the token
    pub text: String,

    /// The position of the token in the original token stream (0-based)
    pub position: usize,

    /// The byte offset where this token starts in the original text
    pub start_offset: usize,

    /// The byte offset where this token ends in the original text
    pub end_offset: usize,

    /// Boost factor for this token (default: 1.0)
    pub boost: f32,

    /// Whether this token has been marked as stopped (removed) by a filter
    pub stopped: bool,

    /// Additional metadata that can be attached to tokens
    pub metadata: Option<TokenMetadata>,

    /// Position increment from the previous token (default: 1).
    ///
    /// This determines the position of this token relative to the previous token.
    /// - 1 (default): Normal increment, next position
    /// - 0: Same position as previous token (e.g., for synonyms)
    /// - >1: Skip positions (e.g., for removed stop words)
    ///
    /// Used for phrase queries and positional information in the token graph.
    pub position_increment: usize,

    /// How many positions this token spans (default: 1).
    ///
    /// For multi-word synonyms, this indicates how many token positions
    /// this token covers. For example, if "machine learning" is replaced
    /// by "ml", the "ml" token would have position_length=2.
    ///
    /// This is essential for correctly handling token graphs in synonym expansion.
    pub position_length: usize,
}

/// Token type classification for different kinds of tokens.
///
/// This enum is used to classify tokens by their content type, which helps
/// with language-specific processing and compound word detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenType {
    /// Alphanumeric text (English, Latin scripts)
    Alphanum,
    /// Numeric values
    Num,
    /// CJK (Chinese, Japanese, Korean) characters
    Cjk,
    /// Katakana characters (Japanese)
    Katakana,
    /// Hiragana characters (Japanese)
    Hiragana,
    /// Hangul characters (Korean)
    Hangul,
    /// Punctuation marks
    Punctuation,
    /// Whitespace
    Whitespace,
    /// Synonym token (generated by SynonymGraphFilter)
    Synonym,
    /// Email addresses
    Email,
    /// URLs
    Url,
    /// Other/unknown token types
    Other,
}

/// Additional metadata that can be attached to tokens
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenMetadata {
    /// The original text before filtering (useful for highlighting)
    pub original_text: Option<String>,

    /// Token type classification
    pub token_type: Option<TokenType>,

    /// Language hint for language-specific processing
    pub language: Option<String>,

    /// Additional custom attributes
    pub attributes: std::collections::HashMap<String, String>,
}

impl Token {
    /// Create a new token with the given text and position.
    pub fn new<S: Into<String>>(text: S, position: usize) -> Self {
        Token {
            text: text.into(),
            position,
            start_offset: 0,
            end_offset: 0,
            boost: 1.0,
            stopped: false,
            metadata: None,
            position_increment: 1,
            position_length: 1,
        }
    }

    /// Create a new token with text, position, and character offsets.
    pub fn with_offsets<S: Into<String>>(
        text: S,
        position: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Self {
        Token {
            text: text.into(),
            position,
            start_offset,
            end_offset,
            boost: 1.0,
            stopped: false,
            metadata: None,
            position_increment: 1,
            position_length: 1,
        }
    }

    /// Get the length of the token text.
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Check if the token is empty.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Set the boost factor for this token.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Mark this token as stopped.
    pub fn stop(mut self) -> Self {
        self.stopped = true;
        self
    }

    /// Check if this token is stopped.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Set metadata for this token.
    pub fn with_metadata(mut self, metadata: TokenMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get a reference to the metadata.
    pub fn metadata(&self) -> Option<&TokenMetadata> {
        self.metadata.as_ref()
    }

    /// Get a mutable reference to the metadata.
    pub fn metadata_mut(&mut self) -> Option<&mut TokenMetadata> {
        self.metadata.as_mut()
    }

    /// Set the original text in metadata.
    pub fn with_original_text<S: Into<String>>(mut self, original: S) -> Self {
        let metadata = self.metadata.get_or_insert_with(TokenMetadata::new);
        metadata.original_text = Some(original.into());
        self
    }

    /// Set the token type in metadata.
    pub fn with_token_type(mut self, token_type: TokenType) -> Self {
        let metadata = self.metadata.get_or_insert_with(TokenMetadata::new);
        metadata.token_type = Some(token_type);
        self
    }

    /// Clone this token with updated text.
    pub fn with_text<S: Into<String>>(&self, text: S) -> Self {
        let mut token = self.clone();
        token.text = text.into();
        token
    }

    /// Clone this token with updated position.
    pub fn with_position(&self, position: usize) -> Self {
        let mut token = self.clone();
        token.position = position;
        token
    }

    /// Set the position increment.
    pub fn with_position_increment(mut self, increment: usize) -> Self {
        self.position_increment = increment;
        self
    }

    /// Set the position length.
    pub fn with_position_length(mut self, length: usize) -> Self {
        self.position_length = length;
        self
    }
}

impl TokenMetadata {
    /// Create a new empty metadata object.
    pub fn new() -> Self {
        TokenMetadata {
            original_text: None,
            token_type: None,
            language: None,
            attributes: std::collections::HashMap::new(),
        }
    }

    /// Set a custom attribute.
    pub fn set_attribute<K, V>(&mut self, key: K, value: V)
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.attributes.insert(key.into(), value.into());
    }

    /// Get a custom attribute.
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }
}

impl Default for TokenMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

/// A token stream represents a sequence of tokens from the analysis pipeline.
pub type TokenStream = Box<dyn Iterator<Item = Token> + Send>;

/// Trait for types that can produce a token stream.
pub trait IntoTokenStream {
    /// Convert this type into a token stream.
    fn into_token_stream(self) -> TokenStream;
}

impl IntoTokenStream for Vec<Token> {
    fn into_token_stream(self) -> TokenStream {
        Box::new(self.into_iter())
    }
}

/// Numbers a token stream the way the index stores it.
///
/// A token with `position_increment == 0` shares the previous token's
/// position; any other token takes the next position, and the first token
/// is at 0 whatever its increment. An increment above 1 is not a gap: no
/// filter leaves one (a removed stop word does not), so a stream without
/// stacked tokens is numbered 0, 1, 2, … The indexer, the highlighter and
/// the query parser all number tokens through this type, so a phrase
/// compares the same positions on both sides.
#[derive(Debug, Default)]
pub(crate) struct TokenPositions {
    current: Option<u32>,
}

impl TokenPositions {
    /// Return the position of `token`, the next token of the stream.
    pub(crate) fn assign(&mut self, token: &Token) -> u32 {
        let position = match self.current {
            None => 0,
            Some(current) if token.position_increment == 0 => current,
            Some(current) => current.saturating_add(1),
        };
        self.current = Some(position);
        position
    }
}

/// The positions [`TokenPositions`] assigns to `tokens`, in order.
pub(crate) fn token_positions(tokens: &[Token]) -> Vec<u32> {
    let mut positions = TokenPositions::default();
    tokens.iter().map(|token| positions.assign(token)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(text: &str, increment: usize) -> Token {
        Token::new(text, 0).with_position_increment(increment)
    }

    #[test]
    fn stacked_tokens_share_a_position() {
        let tokens = [
            token("a", 1),
            token("big", 1),
            token("large", 0),
            token("dog", 1),
        ];
        assert_eq!(token_positions(&tokens), vec![0, 1, 1, 2]);
    }

    /// `FlattenGraphFilter` can give the first token increment 0.
    #[test]
    fn first_token_is_at_zero_whatever_its_increment() {
        assert_eq!(
            token_positions(&[token("a", 0), token("b", 0), token("c", 1)]),
            vec![0, 0, 1]
        );
        assert_eq!(token_positions(&[token("a", 5), token("b", 1)]), vec![0, 1]);
    }

    #[test]
    fn an_increment_above_one_is_not_a_gap() {
        assert_eq!(
            token_positions(&[token("a", 1), token("b", 3), token("c", 2)]),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn no_tokens_no_positions() {
        assert!(token_positions(&[]).is_empty());
    }

    #[test]
    fn test_token_creation() {
        let token = Token::new("hello", 0);
        assert_eq!(token.text, "hello");
        assert_eq!(token.position, 0);
        assert_eq!(token.start_offset, 0);
        assert_eq!(token.end_offset, 0);
        assert_eq!(token.boost, 1.0);
        assert!(!token.stopped);
        assert!(token.metadata.is_none());
    }

    #[test]
    fn test_token_with_offsets() {
        let token = Token::with_offsets("world", 1, 6, 11);
        assert_eq!(token.text, "world");
        assert_eq!(token.position, 1);
        assert_eq!(token.start_offset, 6);
        assert_eq!(token.end_offset, 11);
    }

    #[test]
    fn test_token_methods() {
        let token = Token::new("test", 0)
            .with_boost(2.0)
            .stop()
            .with_original_text("TEST")
            .with_token_type(TokenType::Alphanum);

        assert_eq!(token.boost, 2.0);
        assert!(token.is_stopped());
        assert!(token.metadata.is_some());

        let metadata = token.metadata.as_ref().unwrap();
        assert_eq!(metadata.original_text.as_deref(), Some("TEST"));
        assert_eq!(metadata.token_type, Some(TokenType::Alphanum));
    }

    #[test]
    fn test_token_metadata() {
        let mut metadata = TokenMetadata::new();
        metadata.set_attribute("custom", "value");

        assert_eq!(metadata.get_attribute("custom"), Some("value"));
        assert_eq!(metadata.get_attribute("missing"), None);
    }

    #[test]
    fn test_token_display() {
        let token = Token::new("hello", 0);
        assert_eq!(format!("{token}"), "hello");
    }

    #[test]
    fn test_token_stream() {
        let tokens = vec![Token::new("hello", 0), Token::new("world", 1)];

        let stream = tokens.into_token_stream();
        let collected: Vec<_> = stream.collect();

        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].text, "hello");
        assert_eq!(collected[1].text, "world");
    }
}
