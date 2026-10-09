//! PHP wrappers for the Laurus analysis pipeline.
//!
//! `SynonymGraphFilter::__construct`'s `keepOriginal` parameter is camelCase
//! (not Rust-idiomatic snake_case) because `#[php_impl]` exposes a
//! parameter's literal Rust identifier as its PHP named-argument name, and
//! the rest of this binding's public surface is camelCase (Issue #1384). The
//! lint is allowed for the whole module because the macro binds each
//! argument by that identifier in code it generates outside the `impl`
//! block, where a function-level `#[allow]` does not reach.
#![allow(non_snake_case)]

use std::cell::RefCell;

use ext_php_rs::prelude::*;
use laurus::analysis::synonym::dictionary::SynonymDictionary;
use laurus::analysis::token::{Token, TokenType};
use laurus::analysis::token_filter::Filter;
use laurus::analysis::token_filter::synonym_graph::SynonymGraphFilter;
use laurus::analysis::tokenizer::Tokenizer;
use laurus::analysis::tokenizer::whitespace::WhitespaceTokenizer;

use crate::errors::laurus_err;

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

/// A single token produced by the analysis pipeline (`Laurus\Token`).
///
/// Properties:
///   - `text` (string): The token text.
///   - `position` (int): Position in the token stream.
///   - `startOffset` (int): UTF-8 byte start offset in the original text.
///   - `endOffset` (int): UTF-8 byte end offset in the original text.
///   - `boost` (float): Score boost factor (1.0 = no adjustment).
///   - `stopped` (bool): Whether this token has been removed by a stop filter.
///   - `positionIncrement` (int): Difference from the previous token's position.
///   - `positionLength` (int): Number of positions spanned by this token.
///   - `tokenType` (?string): Token type, such as `"alphanum"` or `"synonym"`.
#[php_class]
#[php(name = "Laurus\\Token")]
pub struct PhpToken {
    text: String,
    position: usize,
    start_offset: usize,
    end_offset: usize,
    boost: f32,
    stopped: bool,
    position_increment: usize,
    position_length: usize,
    token_type: Option<TokenType>,
}

#[php_impl]
impl PhpToken {
    /// Return the token text.
    pub fn get_text(&self) -> String {
        self.text.clone()
    }

    /// Return the token position.
    pub fn get_position(&self) -> i64 {
        self.position as i64
    }

    /// Return the start offset.
    pub fn get_start_offset(&self) -> i64 {
        self.start_offset as i64
    }

    /// Return the end offset.
    pub fn get_end_offset(&self) -> i64 {
        self.end_offset as i64
    }

    /// Return the boost factor.
    pub fn get_boost(&self) -> f64 {
        self.boost as f64
    }

    /// Return whether this token is stopped.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Return the position increment.
    pub fn get_position_increment(&self) -> i64 {
        self.position_increment as i64
    }

    /// Return the position length.
    pub fn get_position_length(&self) -> i64 {
        self.position_length as i64
    }

    /// Return the token type, such as `"alphanum"` or `"synonym"`, or null.
    pub fn get_token_type(&self) -> Option<String> {
        self.token_type
            .map(|token_type| token_type.as_str().to_string())
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "Token(text='{}', position={}, boost={:.2}, pos_inc={}, pos_len={})",
            self.text, self.position, self.boost, self.position_increment, self.position_length
        )
    }
}

impl From<Token> for PhpToken {
    fn from(t: Token) -> Self {
        Self {
            text: t.text,
            position: t.position,
            start_offset: t.start_offset,
            end_offset: t.end_offset,
            boost: t.boost,
            stopped: t.stopped,
            position_increment: t.position_increment,
            position_length: t.position_length,
            token_type: t.metadata.and_then(|m| m.token_type),
        }
    }
}

/// Rebuild the core token from everything a `PhpToken` carries.
///
/// Token metadata other than the type (original text, language,
/// attributes) is not carried.
impl From<&PhpToken> for Token {
    fn from(t: &PhpToken) -> Self {
        let mut token =
            Token::with_offsets(t.text.clone(), t.position, t.start_offset, t.end_offset)
                .with_boost(t.boost)
                .with_position_increment(t.position_increment)
                .with_position_length(t.position_length);
        token.stopped = t.stopped;
        if let Some(token_type) = t.token_type {
            token = token.with_token_type(token_type);
        }
        token
    }
}

// ---------------------------------------------------------------------------
// SynonymDictionary
// ---------------------------------------------------------------------------

/// A dictionary of synonym groups used by `SynonymGraphFilter`
/// (`Laurus\SynonymDictionary`).
#[php_class]
#[php(name = "Laurus\\SynonymDictionary")]
pub struct PhpSynonymDictionary {
    pub inner: RefCell<SynonymDictionary>,
}

#[php_impl]
impl PhpSynonymDictionary {
    /// Create an empty synonym dictionary.
    pub fn __construct() -> PhpResult<Self> {
        SynonymDictionary::new(None)
            .map(|inner| Self {
                inner: RefCell::new(inner),
            })
            .map_err(laurus_err)
    }

    /// Add a bidirectional synonym group.
    ///
    /// All terms in the group are treated as synonyms of each other.
    ///
    /// # Arguments
    ///
    /// * `terms` - Array of synonym strings.
    pub fn add_synonym_group(&self, terms: Vec<String>) {
        self.inner.borrow_mut().add_synonym_group(terms);
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "SynonymDictionary(max_phrase_length={})",
            self.inner.borrow().max_phrase_length()
        )
    }
}

// ---------------------------------------------------------------------------
// WhitespaceTokenizer
// ---------------------------------------------------------------------------

/// Splits text on whitespace boundaries (`Laurus\WhitespaceTokenizer`).
#[php_class]
#[php(name = "Laurus\\WhitespaceTokenizer")]
pub struct PhpWhitespaceTokenizer {
    inner: WhitespaceTokenizer,
}

#[php_impl]
impl PhpWhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    pub fn __construct() -> Self {
        Self {
            inner: WhitespaceTokenizer,
        }
    }

    /// Tokenize a text string and return an array of `Token` objects.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to tokenize.
    ///
    /// # Returns
    ///
    /// Array of `Token` objects.
    pub fn tokenize(&self, text: String) -> PhpResult<Vec<PhpToken>> {
        let tokens: Vec<PhpToken> = self
            .inner
            .tokenize(&text)
            .map(|stream| stream.map(PhpToken::from).collect())
            .map_err(laurus_err)?;
        Ok(tokens)
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        "WhitespaceTokenizer()".to_string()
    }
}

// ---------------------------------------------------------------------------
// SynonymGraphFilter
// ---------------------------------------------------------------------------

/// Token filter that expands tokens with their synonyms from a
/// `SynonymDictionary` (`Laurus\SynonymGraphFilter`).
#[php_class]
#[php(name = "Laurus\\SynonymGraphFilter")]
pub struct PhpSynonymGraphFilter {
    inner: SynonymGraphFilter,
}

#[php_impl]
impl PhpSynonymGraphFilter {
    /// Create a new synonym graph filter.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The `SynonymDictionary` to use for expansion.
    /// * `keepOriginal` - Whether to retain the original token (default: true).
    /// * `boost` - Weight multiplier for synonym tokens (default: 1.0).
    #[php(defaults(keepOriginal = true, boost = 1.0))]
    pub fn __construct(dictionary: &PhpSynonymDictionary, keepOriginal: bool, boost: f64) -> Self {
        let boost = boost as f32;
        let mut filt = SynonymGraphFilter::new(dictionary.inner.borrow().clone(), keepOriginal);
        if (boost - 1.0f32).abs() > f32::EPSILON {
            filt = filt.with_boost(boost);
        }
        Self { inner: filt }
    }

    /// Apply the synonym filter to a list of tokens.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Array of `Token` objects.
    ///
    /// # Returns
    ///
    /// Array of expanded `Token` objects.
    pub fn apply(&self, tokens: Vec<&PhpToken>) -> PhpResult<Vec<PhpToken>> {
        let rust_tokens: Vec<Token> = tokens.into_iter().map(Token::from).collect();

        let stream: Box<dyn Iterator<Item = Token> + Send> = Box::new(rust_tokens.into_iter());

        let result: Vec<PhpToken> = self
            .inner
            .filter(stream)
            .map(|out| out.map(PhpToken::from).collect())
            .map_err(laurus_err)?;
        Ok(result)
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        "SynonymGraphFilter()".to_string()
    }
}
