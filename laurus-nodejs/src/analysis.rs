//! Node.js wrappers for the Laurus analysis pipeline.
//!
//! ## Example
//!
//! ```javascript
//! const { SynonymDictionary, WhitespaceTokenizer, SynonymGraphFilter } = require("laurus-nodejs");
//!
//! const synDict = new SynonymDictionary();
//! synDict.addSynonymGroup(["ml", "machine learning"]);
//!
//! const tokenizer = new WhitespaceTokenizer();
//! const tokens = tokenizer.tokenize("ml tutorial");
//!
//! const filter = new SynonymGraphFilter(synDict, true, 0.8);
//! const expanded = filter.apply(tokens);
//! for (const token of expanded) {
//!     console.log(token.text, token.position, token.boost);
//! }
//! ```

use crate::errors::laurus_err;
use laurus::analysis::synonym::dictionary::SynonymDictionary;
use laurus::analysis::token::{Token, TokenType};
use laurus::analysis::token_filter::Filter;
use laurus::analysis::token_filter::synonym_graph::SynonymGraphFilter;
use laurus::analysis::tokenizer::Tokenizer;
use laurus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

/// A single token produced by the analysis pipeline.
///
/// Properties:
///   - `text` (string): The token text.
///   - `position` (number): Position in the token stream.
///   - `startOffset` (number): UTF-8 byte start offset in the original text.
///   - `endOffset` (number): UTF-8 byte end offset in the original text.
///   - `boost` (number): Score boost factor (1.0 = no adjustment).
///   - `stopped` (boolean): Whether this token has been removed by a stop filter.
///   - `positionIncrement` (number): Difference from the previous token's position.
///   - `positionLength` (number): Number of positions spanned by this token.
///   - `tokenType` (string, optional): Token type, such as `"alphanum"` or `"synonym"`.
#[napi(object)]
pub struct JsToken {
    /// The token text.
    pub text: String,
    /// Position in the token stream.
    pub position: u32,
    /// UTF-8 byte start offset in the original text.
    pub start_offset: u32,
    /// UTF-8 byte end offset in the original text.
    pub end_offset: u32,
    /// Score boost factor (1.0 = no adjustment).
    pub boost: f64,
    /// Whether this token has been removed by a stop filter.
    pub stopped: bool,
    /// Difference from the previous token's position.
    pub position_increment: u32,
    /// Number of positions spanned by this token.
    pub position_length: u32,
    /// Token type, such as `"alphanum"` or `"synonym"`. A token built by
    /// hand may leave it out.
    pub token_type: Option<String>,
}

impl From<Token> for JsToken {
    fn from(t: Token) -> Self {
        Self {
            text: t.text,
            position: t.position as u32,
            start_offset: t.start_offset as u32,
            end_offset: t.end_offset as u32,
            boost: t.boost as f64,
            stopped: t.stopped,
            position_increment: t.position_increment as u32,
            position_length: t.position_length as u32,
            token_type: t
                .metadata
                .and_then(|m| m.token_type)
                .map(|token_type| token_type.as_str().to_string()),
        }
    }
}

/// Rebuild the core token from everything a `JsToken` carries.
///
/// Token metadata other than the type (original text, language,
/// attributes) is not carried.
impl TryFrom<JsToken> for Token {
    type Error = napi::Error;

    fn try_from(t: JsToken) -> Result<Self> {
        let mut token = Token::with_offsets(
            t.text,
            t.position as usize,
            t.start_offset as usize,
            t.end_offset as usize,
        )
        .with_boost(t.boost as f32)
        .with_position_increment(t.position_increment as usize)
        .with_position_length(t.position_length as usize);
        token.stopped = t.stopped;
        if let Some(name) = t.token_type {
            token = token.with_token_type(name.parse::<TokenType>().map_err(laurus_err)?);
        }
        Ok(token)
    }
}

// ---------------------------------------------------------------------------
// SynonymDictionary
// ---------------------------------------------------------------------------

/// A dictionary of synonym groups used by `SynonymGraphFilter`.
///
/// ## Example
///
/// ```javascript
/// const { SynonymDictionary } = require("laurus-nodejs");
///
/// const synDict = new SynonymDictionary();
/// synDict.addSynonymGroup(["ml", "machine learning"]);
/// synDict.addSynonymGroup(["ai", "artificial intelligence"]);
/// ```
#[napi(js_name = "SynonymDictionary")]
pub struct JsSynonymDictionary {
    pub(crate) inner: SynonymDictionary,
}

#[napi]
impl JsSynonymDictionary {
    /// Create an empty synonym dictionary.
    #[napi(constructor)]
    pub fn new() -> Result<Self> {
        SynonymDictionary::new(None)
            .map(|inner| Self { inner })
            .map_err(laurus_err)
    }

    /// Add a bidirectional synonym group.
    ///
    /// All terms in the group are treated as synonyms of each other.
    ///
    /// # Arguments
    ///
    /// * `terms` - List of synonym strings (e.g. `["ml", "machine learning"]`).
    #[napi]
    pub fn add_synonym_group(&mut self, terms: Vec<String>) {
        self.inner.add_synonym_group(terms);
    }
}

// ---------------------------------------------------------------------------
// WhitespaceTokenizer
// ---------------------------------------------------------------------------

/// Splits text on whitespace boundaries.
///
/// ## Example
///
/// ```javascript
/// const { WhitespaceTokenizer } = require("laurus-nodejs");
///
/// const tokenizer = new WhitespaceTokenizer();
/// const tokens = tokenizer.tokenize("ml tutorial");
/// ```
#[napi(js_name = "WhitespaceTokenizer")]
pub struct JsWhitespaceTokenizer {
    inner: WhitespaceTokenizer,
}

#[napi]
impl JsWhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: WhitespaceTokenizer,
        }
    }

    /// Tokenize a text string and return a list of Token objects.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to tokenize.
    ///
    /// # Returns
    ///
    /// An array of Token objects.
    #[napi]
    pub fn tokenize(&self, text: String) -> Result<Vec<JsToken>> {
        self.inner
            .tokenize(&text)
            .map(|stream| stream.map(JsToken::from).collect())
            .map_err(laurus_err)
    }
}

// ---------------------------------------------------------------------------
// SynonymGraphFilter
// ---------------------------------------------------------------------------

/// Token filter that expands tokens with their synonyms from a
/// `SynonymDictionary`.
///
/// ## Example
///
/// ```javascript
/// const { SynonymDictionary, WhitespaceTokenizer, SynonymGraphFilter } = require("laurus-nodejs");
///
/// const synDict = new SynonymDictionary();
/// synDict.addSynonymGroup(["ml", "machine learning"]);
///
/// const tokenizer = new WhitespaceTokenizer();
/// const tokens = tokenizer.tokenize("ml tutorial");
///
/// const filter = new SynonymGraphFilter(synDict);
/// const expanded = filter.apply(tokens);
/// ```
#[napi(js_name = "SynonymGraphFilter")]
pub struct JsSynonymGraphFilter {
    inner: SynonymGraphFilter,
}

#[napi]
impl JsSynonymGraphFilter {
    /// Create a new synonym graph filter.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The `SynonymDictionary` to use for expansion.
    /// * `keep_original` - Whether to retain the original token alongside synonyms (default `true`).
    /// * `boost` - Weight multiplier for synonym tokens (0.0–1.0, default 1.0).
    #[napi(constructor)]
    pub fn new(
        dictionary: &JsSynonymDictionary,
        keep_original: Option<bool>,
        boost: Option<f64>,
    ) -> Self {
        let mut filt =
            SynonymGraphFilter::new(dictionary.inner.clone(), keep_original.unwrap_or(true));
        let boost_val = boost.unwrap_or(1.0) as f32;
        if (boost_val - 1.0f32).abs() > f32::EPSILON {
            filt = filt.with_boost(boost_val);
        }
        Self { inner: filt }
    }

    /// Apply the synonym filter to a list of tokens.
    ///
    /// # Arguments
    ///
    /// * `tokens` - A list of Token objects (e.g. from `WhitespaceTokenizer.tokenize()`).
    ///
    /// # Returns
    ///
    /// A list of expanded Token objects.
    #[napi]
    pub fn apply(&self, tokens: Vec<JsToken>) -> Result<Vec<JsToken>> {
        let rust_tokens = tokens
            .into_iter()
            .map(Token::try_from)
            .collect::<Result<Vec<_>>>()?;

        let stream: Box<dyn Iterator<Item = Token> + Send> = Box::new(rust_tokens.into_iter());

        self.inner
            .filter(stream)
            .map(|out| out.map(JsToken::from).collect())
            .map_err(laurus_err)
    }
}
