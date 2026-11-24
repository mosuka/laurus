//! Lindera-based morphological tokenizer.
//!
//! This module provides a tokenizer using the Lindera library for
//! morphological analysis of CJK (Chinese, Japanese, Korean) languages.
//! Lindera performs dictionary-based word segmentation, which is essential
//! for languages that don't use spaces to separate words.
//!
//! # Dictionary loading
//!
//! Pass a filesystem path to a Lindera dictionary directory as the
//! `dict_uri` argument:
//!
//! - Japanese: an IPADIC build (e.g. `/var/lib/lindera/ipadic`)
//! - Korean: a ko-dic build (e.g. `/var/lib/lindera/ko-dic`)
//! - Chinese: a cc-cedict build (e.g. `/var/lib/lindera/cc-cedict`)
//!
//! `laurus` no longer enables Lindera's `embed-*` features by default,
//! so `embedded://*` URIs are not resolvable at runtime. The
//! laurus test suite continues to use `embedded://*` URIs because the
//! features are activated for the test build via `[dev-dependencies]`.
//!
//! # Examples
//!
//! ```
//! use laurus::analysis::tokenizer::lindera::LinderaTokenizer;
//! use laurus::analysis::tokenizer::Tokenizer;
//!
//! // In tests, the embedded ipadic is available; in production, supply a path.
//! let tokenizer = LinderaTokenizer::new("normal", "embedded://ipadic", None).unwrap();
//! let tokens: Vec<_> = tokenizer.tokenize("日本語の解析").unwrap().collect();
//!
//! // Tokens: ["日本", "語", "の", "解析"]
//! assert!(tokens.len() > 0);
//! ```

use std::borrow::Cow;
use std::str::FromStr;

use lindera::dictionary::{load_dictionary, load_user_dictionary};
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;

use crate::analysis::token::{Token, TokenStream, TokenType};
use crate::analysis::tokenizer::Tokenizer;
use crate::error::{LaurusError, Result};

/// A tokenizer that uses Lindera for morphological analysis.
///
/// This tokenizer performs dictionary-based word segmentation for CJK languages,
/// breaking text into meaningful morphemes (words, particles, suffixes, etc.).
/// Pass a filesystem path to a Lindera dictionary directory at construction
/// time.
///
/// # Segmentation Modes
///
/// - `"normal"`: Standard segmentation
/// - `"decompose"`: Decomposes compound words (generates more tokens)
///
/// # Examples
///
/// ```
/// use laurus::analysis::tokenizer::lindera::LinderaTokenizer;
/// use laurus::analysis::tokenizer::Tokenizer;
///
/// // Japanese with IPADIC at runtime: pass a path like
/// // "/var/lib/lindera/ipadic". In tests we use the embedded form.
/// let tokenizer = LinderaTokenizer::new("normal", "embedded://ipadic", None).unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("形態素解析").unwrap().collect();
///
/// // Korean with ko-dic
/// let tokenizer = LinderaTokenizer::new("normal", "embedded://ko-dic", None).unwrap();
/// let tokens: Vec<_> = tokenizer.tokenize("한국어").unwrap().collect();
/// ```
pub struct LinderaTokenizer {
    // Add any necessary fields for the tokenizer
    inner: Segmenter,
}

impl LinderaTokenizer {
    /// Create a new Lindera tokenizer.
    ///
    /// # Arguments
    ///
    /// * `mode_str` - Segmentation mode: "normal" or "decompose"
    /// * `dict_uri` - Lindera dictionary URI. In production, supply a
    ///   filesystem path to a dictionary directory (e.g.,
    ///   `"/var/lib/lindera/ipadic"`). `embedded://*` URIs only resolve
    ///   when the matching `embed-*` Lindera feature is enabled, which
    ///   `laurus` does not enable by default.
    /// * `user_dict_uri` - Optional user dictionary path for custom words
    ///
    /// # Returns
    ///
    /// A new `LinderaTokenizer` instance
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The mode string is invalid
    /// - The dictionary cannot be loaded
    /// - The user dictionary cannot be loaded
    ///
    /// # Examples
    ///
    /// ```
    /// use laurus::analysis::tokenizer::lindera::LinderaTokenizer;
    ///
    /// // Japanese tokenizer (test-only embedded URI; in production use a path).
    /// let tokenizer = LinderaTokenizer::new(
    ///     "normal",
    ///     "embedded://ipadic",
    ///     None
    /// ).unwrap();
    ///
    /// // With user dictionary
    /// // let tokenizer = LinderaTokenizer::new(
    /// //     "normal",
    /// //     "/var/lib/lindera/ipadic",
    /// //     Some("/etc/laurus/user_dict.csv")
    /// // ).unwrap();
    /// ```
    pub fn new(mode_str: &str, dict_uri: &str, user_dict_uri: Option<&str>) -> Result<Self> {
        let mode = Mode::from_str(mode_str).map_err(|e| {
            LaurusError::invalid_argument(format!("Invalid mode '{}': {}", mode_str, e))
        })?;
        let dict = load_dictionary(dict_uri)
            .map_err(|e| LaurusError::analysis(format!("Failed to load dictionary: {}", e)))?;
        let metadata = &dict.metadata;
        let user_dict = match user_dict_uri {
            Some(uri) => Some(load_user_dictionary(uri, metadata).map_err(|e| {
                LaurusError::analysis(format!("Failed to load user dictionary: {}", e))
            })?),
            None => None,
        };
        let inner = Segmenter::new(mode, dict, user_dict);

        Ok(Self { inner })
    }

    /// Create a Lindera tokenizer from raw dictionary byte arrays.
    ///
    /// Builds the underlying [`lindera::dictionary::Dictionary`] in
    /// memory from each component file, bypassing filesystem access.
    /// Useful for environments without a real filesystem (browser
    /// WASM with OPFS-loaded dictionaries) and for embedding
    /// dictionaries shipped through alternate channels.
    ///
    /// # Arguments
    ///
    /// * `mode_str` - Segmentation mode: `"normal"` or `"decompose"`.
    /// * `metadata` - Contents of `metadata.json`.
    /// * `dict_trie` - Contents of `dict.trie` (prefix trie).
    /// * `dict_vals_idx` - Contents of `dict.valsidx` (prefix sum from
    ///   a surface ordinal to its run of `dict.vals` records).
    /// * `dict_vals` - Contents of `dict.vals` (word value data).
    /// * `dict_words_idx` - Contents of `dict.wordsidx` (word details
    ///   index).
    /// * `dict_words` - Contents of `dict.words` (word details).
    /// * `matrix_mtx` - Contents of `matrix.mtx` (connection cost
    ///   matrix).
    /// * `char_def` - Contents of `char_def.bin` (character
    ///   definitions).
    /// * `unk` - Contents of `unk.bin` (unknown word dictionary).
    ///
    /// # Returns
    ///
    /// A new `LinderaTokenizer` instance with no user dictionary.
    ///
    /// # Errors
    ///
    /// Returns an error if the mode is invalid or any component fails
    /// to deserialize from the supplied bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn from_bytes(
        mode_str: &str,
        metadata: &[u8],
        dict_trie: &[u8],
        dict_vals_idx: &[u8],
        dict_vals: &[u8],
        dict_words_idx: &[u8],
        dict_words: &[u8],
        matrix_mtx: &[u8],
        char_def: &[u8],
        unk: &[u8],
    ) -> Result<Self> {
        use std::sync::Arc;

        use lindera::dictionary::Dictionary;
        use lindera_dictionary::dictionary::character_definition::CharacterDefinition;
        use lindera_dictionary::dictionary::connection_cost_matrix::ConnectionCostMatrix;
        use lindera_dictionary::dictionary::metadata::Metadata;
        use lindera_dictionary::dictionary::prefix_dictionary::PrefixDictionary;
        use lindera_dictionary::dictionary::unknown_dictionary::UnknownDictionary;

        let mode = Mode::from_str(mode_str).map_err(|e| {
            LaurusError::invalid_argument(format!("Invalid mode '{}': {}", mode_str, e))
        })?;
        let meta = Metadata::load(metadata)
            .map_err(|e| LaurusError::analysis(format!("Failed to load metadata: {}", e)))?;
        // `PrefixDictionary::load` validates the trie header and the
        // `dict.valsidx` record alignment before the bytes are walked, so
        // caller-supplied dictionaries (a user dictionary read from disk,
        // the wasm OPFS cache) cannot make later lookups read out of bounds.
        let prefix_dictionary = PrefixDictionary::load(
            dict_trie.to_vec(),
            dict_vals_idx.to_vec(),
            dict_vals.to_vec(),
            dict_words_idx.to_vec(),
            dict_words.to_vec(),
        )
        .map_err(|e| LaurusError::analysis(format!("Failed to load prefix dictionary: {}", e)))?;
        let connection_cost_matrix = ConnectionCostMatrix::load(matrix_mtx.to_vec())
            .map_err(|e| LaurusError::analysis(format!("Failed to load cost matrix: {}", e)))?;
        let character_definition = CharacterDefinition::load(char_def).map_err(|e| {
            LaurusError::analysis(format!("Failed to load character definition: {}", e))
        })?;
        let unknown_dictionary = UnknownDictionary::load(unk).map_err(|e| {
            LaurusError::analysis(format!("Failed to load unknown dictionary: {}", e))
        })?;

        let dict = Dictionary {
            prefix_dictionary: Arc::new(prefix_dictionary),
            connection_cost_matrix: Arc::new(connection_cost_matrix),
            character_definition: Arc::new(character_definition),
            unknown_dictionary: Arc::new(unknown_dictionary),
            metadata: Arc::new(meta),
        };
        let inner = Segmenter::new(mode, dict, None);
        Ok(Self { inner })
    }

    /// Detect token type based on character content.
    ///
    /// Analyzes the token text to determine its type:
    /// - All numeric → Num
    /// - All Hiragana → Hiragana
    /// - All Katakana → Katakana
    /// - Contains Hangul → Hangul
    /// - Contains CJK → Cjk
    /// - ASCII alphanumeric → Alphanum
    /// - All punctuation → Punctuation
    /// - Otherwise → Other
    fn detect_token_type(text: &str) -> TokenType {
        if text.is_empty() {
            return TokenType::Other;
        }

        // Check if all characters are numeric
        if text.chars().all(|c| c.is_numeric()) {
            return TokenType::Num;
        }

        // Check if it's Hiragana
        if text.chars().all(|c| matches!(c, '\u{3040}'..='\u{309F}')) {
            return TokenType::Hiragana;
        }

        // Check if it's Katakana
        if text.chars().all(|c| matches!(c, '\u{30A0}'..='\u{30FF}')) {
            return TokenType::Katakana;
        }

        // Check if it's Hangul
        if text
            .chars()
            .any(|c| matches!(c, '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}'))
        {
            return TokenType::Hangul;
        }

        // Check if it contains CJK characters
        if text.chars().any(|c| {
            matches!(c,
                '\u{4E00}'..='\u{9FFF}' |  // CJK Unified Ideographs
                '\u{3400}'..='\u{4DBF}' |  // CJK Extension A
                '\u{20000}'..='\u{2A6DF}' | // CJK Extension B
                '\u{2A700}'..='\u{2B73F}' | // CJK Extension C
                '\u{2B740}'..='\u{2B81F}' | // CJK Extension D
                '\u{2B820}'..='\u{2CEAF}'   // CJK Extension E
            )
        }) {
            return TokenType::Cjk;
        }

        // Check if it's alphanumeric (ASCII)
        if text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return TokenType::Alphanum;
        }

        // Check if it's punctuation
        if text.chars().all(|c| c.is_ascii_punctuation()) {
            return TokenType::Punctuation;
        }

        TokenType::Other
    }
}

impl Tokenizer for LinderaTokenizer {
    fn tokenize(&self, text: &str) -> Result<TokenStream> {
        let mut tokens = Vec::new();

        for token in self
            .inner
            .segment(Cow::Borrowed(text))
            .map_err(|e| LaurusError::analysis(format!("Failed to segment text: {}", e)))?
        {
            let token_type = Self::detect_token_type(&token.surface);
            tokens.push(
                Token::with_offsets(
                    token.surface,
                    token.position,
                    token.byte_start,
                    token.byte_end,
                )
                .with_token_type(token_type),
            );
        }

        Ok(Box::new(tokens.into_iter()))
    }

    fn name(&self) -> &'static str {
        "lindera"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_japanese() {
        let tokenizer = LinderaTokenizer::new("normal", "embedded://ipadic", None).unwrap();

        let text = "日本語の形態素解析を行うことができます。";

        let tokens: Vec<Token> = tokenizer.tokenize(text).unwrap().collect();

        assert_eq!(tokens.len(), 11);
        assert_eq!(tokens[0].text, "日本語");
        assert_eq!(tokens[1].text, "の");
        assert_eq!(tokens[2].text, "形態素");
        assert_eq!(tokens[3].text, "解析");
        assert_eq!(tokens[4].text, "を");
        assert_eq!(tokens[5].text, "行う");
        assert_eq!(tokens[6].text, "こと");
        assert_eq!(tokens[7].text, "が");
        assert_eq!(tokens[8].text, "でき");
        assert_eq!(tokens[9].text, "ます");
        assert_eq!(tokens[10].text, "。");
    }

    #[test]
    fn test_tokenize_korean() {
        let tokenizer = LinderaTokenizer::new("normal", "embedded://ko-dic", None).unwrap();

        let text = "한국어의형태해석을실시할수있습니다.";

        let tokens: Vec<Token> = tokenizer.tokenize(text).unwrap().collect();

        assert_eq!(tokens.len(), 11);
        assert_eq!(tokens[0].text, "한국어");
        assert_eq!(tokens[1].text, "의");
        assert_eq!(tokens[2].text, "형태");
        assert_eq!(tokens[3].text, "해석");
        assert_eq!(tokens[4].text, "을");
        assert_eq!(tokens[5].text, "실시");
        assert_eq!(tokens[6].text, "할");
        assert_eq!(tokens[7].text, "수");
        assert_eq!(tokens[8].text, "있");
        assert_eq!(tokens[9].text, "습니다");
        assert_eq!(tokens[10].text, ".");
    }

    #[test]
    fn test_tokenize_chinese() {
        let tokenizer = LinderaTokenizer::new("normal", "embedded://cc-cedict", None).unwrap();

        let text = "能够进行汉语的形态素解析。";

        let tokens: Vec<Token> = tokenizer.tokenize(text).unwrap().collect();

        // Jieba tokenizes Chinese text differently from CC-CEDICT.
        assert!(!tokens.is_empty());
        // Verify that key Chinese words are present in the output.
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"能够"));
        assert!(texts.contains(&"进行"));
        assert!(texts.contains(&"汉语"));
        assert!(texts.contains(&"解析"));
    }

    #[test]
    fn test_tokenizer_name() {
        let tokenizer = LinderaTokenizer::new("normal", "embedded://ipadic", None).unwrap();

        assert_eq!(tokenizer.name(), "lindera");
    }

    /// Minimal dictionary source files, mirroring the fixture used by
    /// `lindera-dictionary`'s own builder tests: the mandatory DEFAULT
    /// character category plus KANJI, one unknown-word entry per category,
    /// a two-word lexicon and a 1x1 connection cost matrix.
    #[cfg(feature = "native")]
    const TEST_CHAR_DEF: &str = "DEFAULT 0 1 0\nKANJI 0 0 2\n0x4E00..0x9FFF KANJI\n";
    #[cfg(feature = "native")]
    const TEST_UNK_DEF: &str = "DEFAULT,0,0,0,補助記号,一般,*,*,*,*,*,*,*\n\
         KANJI,0,0,0,名詞,一般,*,*,*,*,*,*,*\n";
    #[cfg(feature = "native")]
    const TEST_LEX_CSV: &str = "日本,0,0,100,名詞,固有名詞,地域,国,*,*,日本,ニッポン,ニッポン\n\
         本,0,0,200,名詞,一般,*,*,*,*,本,ホン,ホン\n";
    #[cfg(feature = "native")]
    const TEST_MATRIX_DEF: &str = "1 1\n0 0 0\n";

    /// A dictionary built by Lindera's own builder must round-trip through
    /// [`LinderaTokenizer::from_bytes`] and segment text.
    ///
    /// The component files are positional, so this pins the mapping between
    /// each on-disk artifact and its `from_bytes` argument: passing them in
    /// the wrong order (for instance swapping `dict.trie` and `dict.valsidx`)
    /// fails here, where the error-path tests below cannot see it.
    #[cfg(feature = "native")]
    #[test]
    fn test_from_bytes_round_trips_a_built_dictionary() {
        use std::fs;

        use lindera_dictionary::builder::DictionaryBuilder;
        use lindera_dictionary::dictionary::metadata::Metadata;

        let input = tempfile::tempdir().unwrap();
        let output = tempfile::tempdir().unwrap();
        fs::write(input.path().join("char.def"), TEST_CHAR_DEF).unwrap();
        fs::write(input.path().join("unk.def"), TEST_UNK_DEF).unwrap();
        fs::write(input.path().join("lex.csv"), TEST_LEX_CSV).unwrap();
        fs::write(input.path().join("matrix.def"), TEST_MATRIX_DEF).unwrap();

        DictionaryBuilder::new(Metadata::default())
            .build_dictionary(input.path(), output.path())
            .expect("building the fixture dictionary should succeed");

        let read = |name: &str| fs::read(output.path().join(name)).unwrap();
        let tokenizer = LinderaTokenizer::from_bytes(
            "normal",
            &read("metadata.json"),
            &read("dict.trie"),
            &read("dict.valsidx"),
            &read("dict.vals"),
            &read("dict.wordsidx"),
            &read("dict.words"),
            &read("matrix.mtx"),
            &read("char_def.bin"),
            &read("unk.bin"),
        )
        .expect("a freshly built dictionary should load from bytes");

        let tokens: Vec<Token> = tokenizer.tokenize("日本").unwrap().collect();
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_ref()).collect();
        assert_eq!(texts, vec!["日本"]);
    }

    #[test]
    fn test_from_bytes_invalid_metadata_errors() {
        let empty: &[u8] = &[];
        let result = LinderaTokenizer::from_bytes(
            "normal",
            b"not valid json".as_slice(),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        );
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("metadata"),
            "expected metadata error, got: {msg}"
        );
    }

    #[test]
    fn test_from_bytes_invalid_mode_errors() {
        // An invalid mode string must short-circuit before any
        // deserialization is attempted.
        let empty: &[u8] = &[];
        let result = LinderaTokenizer::from_bytes(
            "not-a-mode",
            b"{}".as_slice(),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        );
        // The caller's mistake (Issue #1730).
        let result = result.map(|_| ());
        assert!(
            matches!(
                &result,
                Err(LaurusError::InvalidArgument(m)) if m.starts_with("Invalid mode 'not-a-mode': ")
            ),
            "expected mode error, got: {result:?}"
        );
    }
}
