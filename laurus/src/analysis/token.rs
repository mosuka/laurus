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

use std::collections::HashMap;
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

impl TokenType {
    const ALL: [TokenType; 12] = [
        TokenType::Alphanum,
        TokenType::Num,
        TokenType::Cjk,
        TokenType::Katakana,
        TokenType::Hiragana,
        TokenType::Hangul,
        TokenType::Punctuation,
        TokenType::Whitespace,
        TokenType::Synonym,
        TokenType::Email,
        TokenType::Url,
        TokenType::Other,
    ];

    /// Return the lowercase name of this token type, such as `"alphanum"` or
    /// `"synonym"`. [`FromStr`](std::str::FromStr) parses it back.
    pub fn as_str(&self) -> &'static str {
        match self {
            TokenType::Alphanum => "alphanum",
            TokenType::Num => "num",
            TokenType::Cjk => "cjk",
            TokenType::Katakana => "katakana",
            TokenType::Hiragana => "hiragana",
            TokenType::Hangul => "hangul",
            TokenType::Punctuation => "punctuation",
            TokenType::Whitespace => "whitespace",
            TokenType::Synonym => "synonym",
            TokenType::Email => "email",
            TokenType::Url => "url",
            TokenType::Other => "other",
        }
    }
}

impl std::str::FromStr for TokenType {
    type Err = crate::error::LaurusError;

    /// Parse a token type name (case-insensitive), as returned by
    /// [`TokenType::as_str`].
    ///
    /// This is the canonical parser used by all language bindings so the
    /// accepted spelling is identical across Python, Node.js, WASM, Ruby,
    /// and PHP.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LaurusError::invalid_argument`] for any
    /// unrecognised value.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let name = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|token_type| token_type.as_str() == name)
            .ok_or_else(|| {
                let expected: Vec<&str> = Self::ALL.iter().map(TokenType::as_str).collect();
                crate::error::LaurusError::invalid_argument(format!(
                    "unknown token type '{s}' (expected one of: {})",
                    expected.join(", ")
                ))
            })
    }
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

    /// Create a new token with text, position, and byte offsets.
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
/// compares the same positions on both sides. The indexer and the
/// highlighter number a stream after [`flatten_token_graph`], since the
/// index stores no `position_length`.
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

/// Lay a token graph out on the positions the index stores.
///
/// A token is an arc from its node ([`token_positions`]) to its node +
/// `position_length`. When alternatives have inner nodes of their own,
/// such as the members of a multi-word synonym group, every node moves to
/// the most arcs on a path to it, so the alternatives' words share
/// positions, as Lucene's `FlattenGraphFilter` lays them out. A node no
/// arc reaches follows the node before it. Tokens are then ordered by
/// position, keeping their order within one, and get increment 0 when
/// stacked and 1 otherwise (the first token keeps its own), and the
/// `position_length` between their new positions. Offsets are kept.
///
/// A stream whose tokens all span one position is already flat and comes
/// back unchanged.
pub(crate) fn flatten_token_graph(tokens: Vec<Token>) -> Vec<Token> {
    if tokens.iter().all(|token| token.position_length <= 1) {
        return tokens;
    }

    let nodes = token_positions(&tokens);
    // The new position of each node an arc reaches, so far. Arcs only go
    // forward, so a node's entry is final once the stream gets to it.
    let mut reached: HashMap<u32, u32> = HashMap::new();
    let mut previous: Option<(u32, u32)> = None;
    let mut arcs = Vec::with_capacity(tokens.len());
    for (token, &node) in tokens.iter().zip(&nodes) {
        let from = match previous {
            Some((previous_node, position)) if previous_node == node => position,
            Some((_, position)) => reached
                .get(&node)
                .copied()
                .unwrap_or(position.saturating_add(1)),
            None => 0,
        };
        previous = Some((node, from));
        let length = u32::try_from(token.position_length.max(1)).unwrap_or(u32::MAX);
        let to = node.saturating_add(length);
        let position = reached.entry(to).or_insert(0);
        *position = (*position).max(from.saturating_add(1));
        arcs.push((from, to));
    }

    let mut laid_out: Vec<(Token, u32, u32)> = tokens
        .into_iter()
        .zip(arcs)
        .map(|(token, (from, to))| (token, from, reached[&to]))
        .collect();
    laid_out.sort_by_key(|&(_, from, _)| from);

    let mut previous_position = None;
    laid_out
        .into_iter()
        .map(|(mut token, from, to)| {
            if let Some(previous_position) = previous_position {
                token.position_increment = usize::from(previous_position != from);
            }
            previous_position = Some(from);
            token.position_length = (to - from) as usize;
            token
        })
        .collect()
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

    fn arc(text: &str, increment: usize, length: usize, offsets: (usize, usize)) -> Token {
        Token::with_offsets(text, 0, offsets.0, offsets.1)
            .with_position_increment(increment)
            .with_position_length(length)
    }

    /// Text, increment, length and offsets of each token, in order.
    fn shape(tokens: &[Token]) -> Vec<(&str, usize, usize, (usize, usize))> {
        tokens
            .iter()
            .map(|t| {
                (
                    t.text.as_str(),
                    t.position_increment,
                    t.position_length,
                    (t.start_offset, t.end_offset),
                )
            })
            .collect()
    }

    #[test]
    fn a_stream_of_one_position_arcs_is_already_flat() {
        let tokens = vec![
            arc("a", 1, 1, (0, 1)),
            arc("big", 1, 1, (2, 5)),
            arc("large", 0, 1, (2, 5)),
            arc("dog", 1, 1, (6, 9)),
        ];
        assert_eq!(flatten_token_graph(tokens.clone()), tokens);
        assert!(flatten_token_graph(Vec::new()).is_empty());
    }

    /// Alternatives whose words share the inner nodes are already laid out
    /// on the longest path.
    #[test]
    fn a_graph_without_side_nodes_is_unchanged() {
        let tokens = vec![
            arc("ml", 1, 3, (0, 2)),
            arc("machine", 0, 1, (0, 2)),
            arc("statistical", 0, 1, (0, 2)),
            arc("learning", 1, 2, (0, 2)),
            arc("machine", 0, 1, (0, 2)),
            arc("learning", 1, 1, (0, 2)),
            arc("is", 1, 1, (3, 5)),
        ];
        assert_eq!(flatten_token_graph(tokens.clone()), tokens);
    }

    /// "statistical machine learning is" with "ml" and "machine learning":
    /// the kept words have inner nodes 1 and 2, the two-word synonym node 3.
    /// Node 3 moves back to position 1, so its token moves before the one
    /// at node 2, and every token keeps its offsets.
    #[test]
    fn side_nodes_move_onto_the_longest_path() {
        let graph = vec![
            arc("statistical", 1, 1, (0, 11)),
            arc("ml", 0, 4, (0, 28)),
            arc("machine", 0, 3, (0, 28)),
            arc("machine", 1, 1, (12, 19)),
            arc("learning", 1, 2, (20, 28)),
            arc("learning", 1, 1, (0, 28)),
            arc("is", 1, 1, (29, 31)),
        ];
        let flat = flatten_token_graph(graph);
        assert_eq!(
            shape(&flat),
            vec![
                ("statistical", 1, 1, (0, 11)),
                ("ml", 0, 3, (0, 28)),
                ("machine", 0, 1, (0, 28)),
                ("machine", 1, 1, (12, 19)),
                ("learning", 0, 2, (0, 28)),
                ("learning", 1, 1, (20, 28)),
                ("is", 1, 1, (29, 31)),
            ]
        );
        assert_eq!(token_positions(&flat), vec![0, 0, 0, 1, 1, 2, 3]);
    }

    /// A filter removed "machine" from "ml" → "machine learning", so no
    /// arc reaches the node of "learning"; it follows the node before it,
    /// as [`token_positions`] numbers it.
    #[test]
    fn a_node_no_arc_reaches_follows_the_one_before() {
        let tokens = vec![
            arc("ml", 1, 2, (0, 2)),
            arc("learning", 1, 1, (0, 2)),
            arc("is", 1, 1, (3, 5)),
        ];
        let flat = flatten_token_graph(tokens.clone());
        assert_eq!(flat, tokens);
        assert_eq!(token_positions(&flat), vec![0, 1, 2]);
    }

    #[test]
    fn a_huge_position_length_does_not_overflow() {
        let flat = flatten_token_graph(vec![
            arc("a", 1, usize::MAX, (0, 1)),
            arc("b", 1, 1, (2, 3)),
        ]);
        assert_eq!(token_positions(&flat), vec![0, 1]);
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

    #[test]
    fn token_type_names_parse_back() {
        for token_type in TokenType::ALL {
            assert_eq!(
                token_type.as_str().parse::<TokenType>().unwrap(),
                token_type
            );
        }
    }

    #[test]
    fn token_type_parse_ignores_case_and_surrounding_space() {
        assert_eq!(
            " AlphaNum ".parse::<TokenType>().unwrap(),
            TokenType::Alphanum
        );
    }

    #[test]
    fn unknown_token_type_is_an_invalid_argument() {
        let err = "bogus".parse::<TokenType>().unwrap_err();
        assert!(
            matches!(
                &err,
                crate::error::LaurusError::InvalidArgument(m)
                    if m.starts_with("unknown token type 'bogus' (expected one of: alphanum, num,")
            ),
            "{err}"
        );
    }
}
