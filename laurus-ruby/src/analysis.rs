//! Ruby wrappers for the Laurus analysis pipeline.

use std::cell::RefCell;

use crate::errors::laurus_err;
use laurus::analysis::synonym::dictionary::SynonymDictionary;
use laurus::analysis::token::{Token, TokenType};
use laurus::analysis::token_filter::Filter;
use laurus::analysis::token_filter::synonym_graph::SynonymGraphFilter;
use laurus::analysis::tokenizer::Tokenizer;
use laurus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use magnus::prelude::*;
use magnus::scan_args::{get_kwargs, scan_args};
use magnus::{Error, RArray, RHash, RModule, Ruby, Value};

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

/// A single token produced by the analysis pipeline (`Laurus::Token`).
///
/// Attributes:
///   - `text` (String): The token text.
///   - `position` (Integer): Position in the token stream.
///   - `start_offset` (Integer): UTF-8 byte start offset in the original text.
///   - `end_offset` (Integer): UTF-8 byte end offset in the original text.
///   - `boost` (Float): Score boost factor (1.0 = no adjustment).
///   - `stopped` (bool): Whether this token has been removed by a stop filter.
///   - `position_increment` (Integer): Difference from the previous token's position.
///   - `position_length` (Integer): Number of positions spanned by this token.
///   - `token_type` (String or nil): Token type, such as `"alphanum"` or `"synonym"`.
#[magnus::wrap(class = "Laurus::Token")]
pub struct RbToken {
    pub text: String,
    pub position: usize,
    pub start_offset: usize,
    pub end_offset: usize,
    pub boost: f32,
    pub stopped: bool,
    pub position_increment: usize,
    pub position_length: usize,
    pub token_type: Option<TokenType>,
}

impl RbToken {
    /// Return the token text.
    fn text(&self) -> String {
        self.text.clone()
    }
    /// Return the token position.
    fn position(&self) -> usize {
        self.position
    }
    /// Return the start offset.
    fn start_offset(&self) -> usize {
        self.start_offset
    }
    /// Return the end offset.
    fn end_offset(&self) -> usize {
        self.end_offset
    }
    /// Return the boost factor.
    fn boost(&self) -> f32 {
        self.boost
    }
    /// Return whether this token is stopped.
    fn stopped(&self) -> bool {
        self.stopped
    }
    /// Return the position increment.
    fn position_increment(&self) -> usize {
        self.position_increment
    }
    /// Return the position length.
    fn position_length(&self) -> usize {
        self.position_length
    }
    /// Return the token type, such as `"alphanum"` or `"synonym"`, or nil.
    fn token_type(&self) -> Option<&'static str> {
        self.token_type.map(|token_type| token_type.as_str())
    }
    fn inspect(&self) -> String {
        format!(
            "Token(text='{}', position={}, boost={:.2}, pos_inc={}, pos_len={})",
            self.text, self.position, self.boost, self.position_increment, self.position_length
        )
    }
}

impl From<Token> for RbToken {
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

/// Rebuild the core token from everything an `RbToken` carries.
///
/// Token metadata other than the type (original text, language,
/// attributes) is not carried.
impl From<&RbToken> for Token {
    fn from(t: &RbToken) -> Self {
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

/// Convert a vector of `RbToken` into a Ruby `RArray`.
fn tokens_to_rarray(ruby: &Ruby, tokens: Vec<RbToken>) -> Result<RArray, Error> {
    let arr = ruby.ary_new_capa(tokens.len());
    for token in tokens {
        arr.push(ruby.into_value(token))?;
    }
    Ok(arr)
}

// ---------------------------------------------------------------------------
// SynonymDictionary
// ---------------------------------------------------------------------------

/// A dictionary of synonym groups used by `SynonymGraphFilter`
/// (`Laurus::SynonymDictionary`).
#[magnus::wrap(class = "Laurus::SynonymDictionary")]
pub struct RbSynonymDictionary {
    pub inner: RefCell<SynonymDictionary>,
}

impl RbSynonymDictionary {
    /// Create an empty synonym dictionary.
    fn new() -> Result<Self, Error> {
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
    fn add_synonym_group(&self, terms: RArray) -> Result<(), Error> {
        let terms: Vec<String> = terms.to_vec()?;
        self.inner.borrow_mut().add_synonym_group(terms);
        Ok(())
    }

    fn inspect(&self) -> String {
        format!(
            "SynonymDictionary(max_phrase_length={})",
            self.inner.borrow().max_phrase_length()
        )
    }
}

// ---------------------------------------------------------------------------
// WhitespaceTokenizer
// ---------------------------------------------------------------------------

/// Splits text on whitespace boundaries (`Laurus::WhitespaceTokenizer`).
#[magnus::wrap(class = "Laurus::WhitespaceTokenizer")]
pub struct RbWhitespaceTokenizer {
    inner: WhitespaceTokenizer,
}

impl RbWhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    fn new() -> Self {
        Self {
            inner: WhitespaceTokenizer,
        }
    }

    /// Tokenize a text string and return an Array of `Token` objects.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to tokenize.
    ///
    /// # Returns
    ///
    /// Array of `Token` objects.
    fn tokenize(&self, text: String) -> Result<RArray, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let tokens: Vec<RbToken> = self
            .inner
            .tokenize(&text)
            .map(|stream| stream.map(RbToken::from).collect())
            .map_err(laurus_err)?;
        tokens_to_rarray(&ruby, tokens)
    }

    fn inspect(&self) -> String {
        "WhitespaceTokenizer()".to_string()
    }
}

// ---------------------------------------------------------------------------
// SynonymGraphFilter
// ---------------------------------------------------------------------------

/// Token filter that expands tokens with their synonyms from a
/// `SynonymDictionary` (`Laurus::SynonymGraphFilter`).
#[magnus::wrap(class = "Laurus::SynonymGraphFilter")]
pub struct RbSynonymGraphFilter {
    inner: SynonymGraphFilter,
}

impl RbSynonymGraphFilter {
    /// Create a new synonym graph filter.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `dictionary` (SynonymDictionary): The dictionary to use for expansion.
    ///   - `keep_original:` (bool, default true): Whether to retain the original token.
    ///   - `boost:` (f32, default 1.0): Weight multiplier for synonym tokens.
    fn new(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(&RbSynonymDictionary,), (), (), (), RHash, ()>(args)?;
        let (dictionary,) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<bool>, Option<f32>), ()>(
            args.keywords,
            &[],
            &["keep_original", "boost"],
        )?;
        let (keep_original, boost) = kwargs.optional;
        let keep_original = keep_original.unwrap_or(true);
        let boost = boost.unwrap_or(1.0);

        let mut filt = SynonymGraphFilter::new(dictionary.inner.borrow().clone(), keep_original);
        if (boost - 1.0f32).abs() > f32::EPSILON {
            filt = filt.with_boost(boost);
        }
        Ok(Self { inner: filt })
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
    fn apply(&self, tokens: RArray) -> Result<RArray, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        // Manually iterate the array to extract RbToken references
        let len = tokens.len();
        let mut rust_tokens = Vec::with_capacity(len);
        for i in 0..len {
            let val: Value = tokens.entry(i as isize)?;
            let pt: &RbToken = <&RbToken>::try_convert(val)?;
            rust_tokens.push(Token::from(pt));
        }

        let stream: Box<dyn Iterator<Item = Token> + Send> = Box::new(rust_tokens.into_iter());

        let result: Vec<RbToken> = self
            .inner
            .filter(stream)
            .map(|out| out.map(RbToken::from).collect())
            .map_err(laurus_err)?;
        tokens_to_rarray(&ruby, result)
    }

    fn inspect(&self) -> String {
        "SynonymGraphFilter()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Class registration
// ---------------------------------------------------------------------------

/// Register analysis-related classes under the `Laurus` module.
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `module` - The `Laurus` module.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    // Token
    let token = module.define_class("Token", ruby.class_object())?;
    token.define_method("text", magnus::method!(RbToken::text, 0))?;
    token.define_method("position", magnus::method!(RbToken::position, 0))?;
    token.define_method("start_offset", magnus::method!(RbToken::start_offset, 0))?;
    token.define_method("end_offset", magnus::method!(RbToken::end_offset, 0))?;
    token.define_method("boost", magnus::method!(RbToken::boost, 0))?;
    token.define_method("stopped", magnus::method!(RbToken::stopped, 0))?;
    token.define_method(
        "position_increment",
        magnus::method!(RbToken::position_increment, 0),
    )?;
    token.define_method(
        "position_length",
        magnus::method!(RbToken::position_length, 0),
    )?;
    token.define_method("token_type", magnus::method!(RbToken::token_type, 0))?;
    token.define_method("inspect", magnus::method!(RbToken::inspect, 0))?;
    token.define_method("to_s", magnus::method!(RbToken::inspect, 0))?;

    // SynonymDictionary
    let syn_dict = module.define_class("SynonymDictionary", ruby.class_object())?;
    syn_dict.define_singleton_method("new", magnus::function!(RbSynonymDictionary::new, 0))?;
    syn_dict.define_method(
        "add_synonym_group",
        magnus::method!(RbSynonymDictionary::add_synonym_group, 1),
    )?;
    syn_dict.define_method("inspect", magnus::method!(RbSynonymDictionary::inspect, 0))?;
    syn_dict.define_method("to_s", magnus::method!(RbSynonymDictionary::inspect, 0))?;

    // WhitespaceTokenizer
    let ws_tok = module.define_class("WhitespaceTokenizer", ruby.class_object())?;
    ws_tok.define_singleton_method("new", magnus::function!(RbWhitespaceTokenizer::new, 0))?;
    ws_tok.define_method(
        "tokenize",
        magnus::method!(RbWhitespaceTokenizer::tokenize, 1),
    )?;
    ws_tok.define_method(
        "inspect",
        magnus::method!(RbWhitespaceTokenizer::inspect, 0),
    )?;
    ws_tok.define_method("to_s", magnus::method!(RbWhitespaceTokenizer::inspect, 0))?;

    // SynonymGraphFilter
    let sgf = module.define_class("SynonymGraphFilter", ruby.class_object())?;
    sgf.define_singleton_method("new", magnus::function!(RbSynonymGraphFilter::new, -1))?;
    sgf.define_method("apply", magnus::method!(RbSynonymGraphFilter::apply, 1))?;
    sgf.define_method("inspect", magnus::method!(RbSynonymGraphFilter::inspect, 0))?;
    sgf.define_method("to_s", magnus::method!(RbSynonymGraphFilter::inspect, 0))?;

    Ok(())
}
