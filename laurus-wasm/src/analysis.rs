//! WASM wrappers for the Laurus analysis pipeline.

use std::sync::Arc;

use crate::errors::laurus_err;
use laurus::Analyzer;
use laurus::analysis::analyzer::language::japanese::JapaneseAnalyzer;
use laurus::analysis::synonym::dictionary::SynonymDictionary;
use laurus::analysis::token::{Token, TokenType};
use laurus::analysis::token_filter::Filter;
use laurus::analysis::token_filter::synonym_graph::SynonymGraphFilter;
use laurus::analysis::tokenizer::Tokenizer;
use laurus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

/// A single token produced by the analysis pipeline.
#[derive(Serialize, Deserialize)]
pub struct WasmToken {
    /// The token text.
    pub text: String,
    /// Position in the token stream.
    pub position: u32,
    /// UTF-8 byte start offset in the original text.
    #[serde(rename = "startOffset")]
    pub start_offset: u32,
    /// UTF-8 byte end offset in the original text.
    #[serde(rename = "endOffset")]
    pub end_offset: u32,
    /// Score boost factor (1.0 = no adjustment).
    pub boost: f64,
    /// Whether this token has been removed by a stop filter.
    pub stopped: bool,
    /// Difference from the previous token's position.
    #[serde(rename = "positionIncrement")]
    pub position_increment: u32,
    /// Number of positions spanned by this token.
    #[serde(rename = "positionLength")]
    pub position_length: u32,
    /// Token type, such as `"alphanum"` or `"synonym"`. A token built by
    /// hand may leave it out.
    #[serde(rename = "tokenType", default, skip_serializing_if = "Option::is_none")]
    pub token_type: Option<String>,
}

impl From<Token> for WasmToken {
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

/// Rebuild the core token from everything a `WasmToken` carries.
///
/// Token metadata other than the type (original text, language,
/// attributes) is not carried.
impl TryFrom<WasmToken> for Token {
    type Error = JsValue;

    fn try_from(t: WasmToken) -> Result<Self, JsValue> {
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
#[wasm_bindgen(js_name = "SynonymDictionary")]
pub struct WasmSynonymDictionary {
    pub(crate) inner: SynonymDictionary,
}

#[wasm_bindgen(js_class = "SynonymDictionary")]
impl WasmSynonymDictionary {
    /// Create an empty synonym dictionary.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmSynonymDictionary, JsValue> {
        SynonymDictionary::new(None)
            .map(|inner| Self { inner })
            .map_err(laurus_err)
    }

    /// Add a bidirectional synonym group.
    ///
    /// All terms in the group are treated as synonyms of each other.
    #[wasm_bindgen(js_name = "addSynonymGroup")]
    pub fn add_synonym_group(&mut self, terms: Vec<String>) {
        self.inner.add_synonym_group(terms);
    }
}

// ---------------------------------------------------------------------------
// WhitespaceTokenizer
// ---------------------------------------------------------------------------

/// Splits text on whitespace boundaries.
#[wasm_bindgen(js_name = "WhitespaceTokenizer")]
pub struct WasmWhitespaceTokenizer {
    inner: WhitespaceTokenizer,
}

#[wasm_bindgen(js_class = "WhitespaceTokenizer")]
impl WasmWhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: WhitespaceTokenizer,
        }
    }

    /// Tokenize a text string and return a list of Token objects.
    #[wasm_bindgen]
    pub fn tokenize(&self, text: String) -> Result<JsValue, JsValue> {
        let tokens: Vec<WasmToken> = self
            .inner
            .tokenize(&text)
            .map(|stream| stream.map(WasmToken::from).collect())
            .map_err(laurus_err)?;
        serde_wasm_bindgen::to_value(&tokens)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
    }
}

// ---------------------------------------------------------------------------
// SynonymGraphFilter
// ---------------------------------------------------------------------------

/// Token filter that expands tokens with their synonyms.
#[wasm_bindgen(js_name = "SynonymGraphFilter")]
pub struct WasmSynonymGraphFilter {
    inner: SynonymGraphFilter,
}

#[wasm_bindgen(js_class = "SynonymGraphFilter")]
impl WasmSynonymGraphFilter {
    /// Create a new synonym graph filter.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The `SynonymDictionary` to use for expansion.
    /// * `keep_original` - Whether to retain the original token alongside synonyms (default `true`).
    /// * `boost` - Weight multiplier for synonym tokens (0.0–1.0, default 1.0).
    #[wasm_bindgen(constructor)]
    pub fn new(
        dictionary: &WasmSynonymDictionary,
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
    /// Accepts and returns a JS array of Token objects (serialized via serde).
    #[wasm_bindgen]
    pub fn apply(&self, tokens: JsValue) -> Result<JsValue, JsValue> {
        let wasm_tokens: Vec<WasmToken> = serde_wasm_bindgen::from_value(tokens)
            .map_err(|e| JsValue::from_str(&format!("Invalid token array: {e}")))?;

        let rust_tokens = wasm_tokens
            .into_iter()
            .map(Token::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        let stream: Box<dyn Iterator<Item = Token> + Send> = Box::new(rust_tokens.into_iter());

        let result: Vec<WasmToken> = self
            .inner
            .filter(stream)
            .map(|out| out.map(WasmToken::from).collect())
            .map_err(laurus_err)?;

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))
    }
}

// ---------------------------------------------------------------------------
// JapaneseAnalyzer
// ---------------------------------------------------------------------------

/// Japanese morphological analyzer constructed from raw Lindera dictionary
/// bytes.
///
/// Browser WASM has no real filesystem, so the standard
/// `{ "language": "japanese", "dict": "/path/to/ipadic" }` analyzer
/// spec cannot be used. Instead, fetch the eight Lindera dictionary
/// files (typically extracted from a `lindera-ipadic-X.Y.Z.zip` and
/// stored in OPFS), pass them to `JapaneseAnalyzer.fromBytes()`, and
/// register the result on a `Schema` via `Schema.addAnalyzer(name, ...)`.
///
/// ```javascript
/// import { JapaneseAnalyzer, Schema, Index } from "laurus-wasm";
/// import { downloadDictionary, loadDictionaryFiles } from "laurus-wasm/opfs";
///
/// await downloadDictionary("./dict/lindera-ipadic.zip", "ipadic");
/// const f = await loadDictionaryFiles("ipadic");
/// const ja = JapaneseAnalyzer.fromBytes(
///   f.metadata, f.dictTrie, f.dictValsIdx, f.dictVals,
///   f.dictWordsIdx, f.dictWords, f.matrixMtx, f.charDef, f.unk,
///   "normal"
/// );
/// const schema = new Schema();
/// schema.addAnalyzer("ja-ipadic", ja);
/// schema.addTextField("body", undefined, undefined, undefined, undefined, "ja-ipadic");
/// const index = await Index.create(schema);
/// ```
#[wasm_bindgen(js_name = "JapaneseAnalyzer")]
pub struct WasmJapaneseAnalyzer {
    pub(crate) inner: Arc<dyn Analyzer>,
}

#[wasm_bindgen(js_class = "JapaneseAnalyzer")]
impl WasmJapaneseAnalyzer {
    /// Build a Japanese analyzer from raw dictionary byte arrays.
    ///
    /// The nine byte arrays must come from a built Lindera dictionary
    /// directory (e.g. `lindera-ipadic-X.Y.Z.zip` extracted to OPFS).
    /// A trailing `mode` argument (default `"normal"`) selects the
    /// Lindera segmentation mode.
    ///
    /// # Arguments
    ///
    /// * `metadata` - `metadata.json`
    /// * `dict_trie` - `dict.trie` (prefix trie)
    /// * `dict_vals_idx` - `dict.valsidx`
    /// * `dict_vals` - `dict.vals`
    /// * `dict_words_idx` - `dict.wordsidx`
    /// * `dict_words` - `dict.words`
    /// * `matrix_mtx` - `matrix.mtx`
    /// * `char_def` - `char_def.bin`
    /// * `unk` - `unk.bin`
    /// * `mode` - `"normal"` (default) or `"decompose"`
    ///
    /// # Errors
    ///
    /// Returns a JS error if any component fails to deserialize or the
    /// mode string is not recognized.
    #[wasm_bindgen(js_name = "fromBytes")]
    #[allow(clippy::too_many_arguments)]
    pub fn from_bytes(
        metadata: &[u8],
        dict_trie: &[u8],
        dict_vals_idx: &[u8],
        dict_vals: &[u8],
        dict_words_idx: &[u8],
        dict_words: &[u8],
        matrix_mtx: &[u8],
        char_def: &[u8],
        unk: &[u8],
        mode: Option<String>,
    ) -> Result<WasmJapaneseAnalyzer, JsValue> {
        let mode_str = mode.as_deref().unwrap_or("normal");
        let analyzer = JapaneseAnalyzer::from_bytes(
            mode_str,
            metadata,
            dict_trie,
            dict_vals_idx,
            dict_vals,
            dict_words_idx,
            dict_words,
            matrix_mtx,
            char_def,
            unk,
        )
        .map_err(laurus_err)?;
        Ok(Self {
            inner: Arc::new(analyzer),
        })
    }
}

impl WasmJapaneseAnalyzer {
    /// Internal accessor used by `WasmSchema.addAnalyzer` to clone the
    /// underlying `Arc<dyn Analyzer>` into the schema's runtime registry.
    pub(crate) fn analyzer(&self) -> Arc<dyn Analyzer> {
        self.inner.clone()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test;

    /// Bad metadata bytes must surface a JS error mentioning metadata.
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test]
    fn from_bytes_invalid_metadata_returns_js_error() {
        use super::WasmJapaneseAnalyzer;

        let empty: &[u8] = &[];
        let result = WasmJapaneseAnalyzer::from_bytes(
            b"not valid json",
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            None,
        );
        assert!(result.is_err());
        let msg = result
            .err()
            .unwrap()
            .as_string()
            .unwrap_or_default()
            .to_lowercase();
        assert!(
            msg.contains("metadata"),
            "expected metadata error, got: {msg}"
        );
    }

    /// An invalid mode string must short-circuit before any
    /// dictionary-component deserialization is attempted.
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test]
    fn from_bytes_invalid_mode_returns_js_error() {
        use super::WasmJapaneseAnalyzer;

        let empty: &[u8] = &[];
        let result = WasmJapaneseAnalyzer::from_bytes(
            b"{}",
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            Some("not-a-mode".into()),
        );
        assert!(result.is_err());
    }

    #[cfg(target_arch = "wasm32")]
    mod synonym_graph_filter {
        use wasm_bindgen::JsValue;
        use wasm_bindgen_test::wasm_bindgen_test;

        use super::super::{
            WasmSynonymDictionary, WasmSynonymGraphFilter, WasmWhitespaceTokenizer,
        };

        fn filter(group: &[&str]) -> WasmSynonymGraphFilter {
            let mut dict = WasmSynonymDictionary::new().unwrap();
            dict.add_synonym_group(group.iter().map(|s| s.to_string()).collect());
            WasmSynonymGraphFilter::new(&dict, None, None)
        }

        fn tokenize(text: &str) -> JsValue {
            WasmWhitespaceTokenizer::new()
                .tokenize(text.to_string())
                .unwrap()
        }

        fn to_vec(tokens: JsValue) -> Vec<JsValue> {
            js_sys::Array::from(&tokens).iter().collect()
        }

        fn get(token: &JsValue, key: &str) -> JsValue {
            js_sys::Reflect::get(token, &JsValue::from_str(key)).unwrap()
        }

        fn text(token: &JsValue) -> String {
            get(token, "text").as_string().unwrap()
        }

        fn offsets(token: &JsValue) -> (f64, f64) {
            (
                get(token, "startOffset").as_f64().unwrap(),
                get(token, "endOffset").as_f64().unwrap(),
            )
        }

        fn find(tokens: &[JsValue], wanted: &str) -> JsValue {
            tokens.iter().find(|t| text(t) == wanted).unwrap().clone()
        }

        /// A token of `ml` built by hand, with `extra` spliced into the object.
        fn hand_built_ml(extra: &str) -> JsValue {
            js_sys::JSON::parse(&format!(
                r#"[{{"text":"ml","position":0,"startOffset":0,"endOffset":2,"boost":1.0,"stopped":false,"positionIncrement":1,"positionLength":1{extra}}}]"#
            ))
            .unwrap()
        }

        #[wasm_bindgen_test]
        fn tokenizer_sets_offsets_and_token_type() {
            let tokens = to_vec(tokenize("hello"));
            assert_eq!(offsets(&tokens[0]), (0.0, 5.0));
            assert_eq!(
                get(&tokens[0], "tokenType").as_string().as_deref(),
                Some("alphanum")
            );
        }

        #[wasm_bindgen_test]
        fn apply_keeps_offsets_and_token_types() {
            let filter = filter(&["ml", "machine learning"]);
            let tokens = to_vec(filter.apply(tokenize("ml tutorial")).unwrap());

            // Each synonym token carries the offsets of the word it replaces.
            for wanted in ["ml", "machine", "learning"] {
                assert_eq!(offsets(&find(&tokens, wanted)), (0.0, 2.0), "{wanted}");
            }
            assert_eq!(offsets(&find(&tokens, "tutorial")), (3.0, 11.0));
            for (wanted, token_type) in [
                ("ml", "alphanum"),
                ("tutorial", "alphanum"),
                ("machine", "synonym"),
                ("learning", "synonym"),
            ] {
                let actual = get(&find(&tokens, wanted), "tokenType").as_string();
                assert_eq!(actual.as_deref(), Some(token_type), "{wanted}");
            }
        }

        #[wasm_bindgen_test]
        fn apply_does_not_match_across_a_gap() {
            let filter = filter(&["東京大学", "東大"]);
            // CJK words must touch to form one word, and a space separates these.
            let tokens = to_vec(filter.apply(tokenize("東京 大学")).unwrap());
            let texts: Vec<String> = tokens.iter().map(text).collect();
            assert_eq!(texts, ["東京", "大学"]);
        }

        #[wasm_bindgen_test]
        fn apply_matches_alphanumeric_words_across_a_space() {
            let filter = filter(&["ml", "machine learning"]);
            let tokens = to_vec(filter.apply(tokenize("machine learning")).unwrap());
            assert!(tokens.iter().any(|t| text(t) == "ml"));
        }

        #[wasm_bindgen_test]
        fn apply_accepts_a_token_built_by_hand_without_a_type() {
            let filter = filter(&["ml", "machine learning"]);
            let tokens = to_vec(filter.apply(hand_built_ml("")).unwrap());
            assert!(tokens.iter().any(|t| text(t) == "machine"));
            assert!(get(&find(&tokens, "ml"), "tokenType").is_undefined());
        }

        #[wasm_bindgen_test]
        fn apply_rejects_an_unknown_token_type() {
            let filter = filter(&["ml", "machine learning"]);
            let err = filter
                .apply(hand_built_ml(r#","tokenType":"bogus""#))
                .err()
                .unwrap()
                .as_string()
                .unwrap_or_default();
            assert!(err.contains("unknown token type 'bogus'"), "{err}");
        }
    }
}
