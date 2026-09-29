//! Configuration types for custom analyzer definitions within a schema.
//!
//! These types allow users to declaratively define custom text analyzers
//! composed of a tokenizer, optional char filters, and optional token
//! filters. Definitions are stored in the schema's `analyzers` map and
//! referenced by name from [`TextOption::analyzer`].
//!
//! # JSON Format
//!
//! ```json
//! {
//!   "char_filters": [{"type": "unicode_normalization", "form": "nfkc"}],
//!   "tokenizer": {"type": "regex", "pattern": "\\w+"},
//!   "token_filters": [{"type": "lowercase"}, {"type": "stop"}]
//! }
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Reference to an analyzer for a text field.
///
/// Two shapes are accepted (decoded via serde's untagged representation):
///
/// 1. **A bare string** — the name of a built-in or user-defined analyzer.
///    Example: `"standard"`, `"english"`, `"keyword"`, `"simple"`,
///    `"noop"`, or any name registered in [`Schema::analyzers`].
/// 2. **A structured object** — a parameterized built-in analyzer.
///    Currently only the language-specific Japanese preset:
///    `{"language": "japanese", "mode": "normal", "dict": "/path/to/ipadic"}`.
///
/// # JSON Examples
///
/// ```json
/// // Built-in preset that needs no parameters.
/// "standard"
///
/// // Japanese preset that requires a dictionary path.
/// {"language": "japanese", "mode": "normal", "dict": "/var/lib/lindera/ipadic"}
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnalyzerSpec {
    /// Built-in or user-defined analyzer referenced by name.
    Named(String),
    /// Parameterized built-in analyzer.
    Builtin(BuiltinAnalyzerSpec),
}

/// Parameterized built-in analyzer presets.
///
/// Variants are tagged by the `"language"` discriminator in JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "language", rename_all = "snake_case")]
pub enum BuiltinAnalyzerSpec {
    /// Japanese analyzer (Lindera + Japanese filters).
    Japanese {
        /// Lindera segmentation mode: `"normal"` or `"decompose"`.
        /// Defaults to `"normal"` when omitted.
        #[serde(default = "default_lindera_mode")]
        mode: String,
        /// Filesystem path to the lindera dictionary directory
        /// (e.g. `/var/lib/lindera/ipadic`).
        dict: String,
        /// Optional user dictionary path.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user_dict: Option<String>,
    },
}

impl From<&str> for AnalyzerSpec {
    fn from(value: &str) -> Self {
        AnalyzerSpec::Named(value.to_string())
    }
}

impl From<String> for AnalyzerSpec {
    fn from(value: String) -> Self {
        AnalyzerSpec::Named(value)
    }
}

impl From<BuiltinAnalyzerSpec> for AnalyzerSpec {
    fn from(value: BuiltinAnalyzerSpec) -> Self {
        AnalyzerSpec::Builtin(value)
    }
}

fn default_lindera_mode() -> String {
    "normal".to_string()
}

/// A custom analyzer definition composed of a tokenizer and optional
/// char/token filter chains.
///
/// # Fields
///
/// * `char_filters` - Applied to raw text before tokenization.
/// * `tokenizer` - Splits text into tokens.
/// * `token_filters` - Applied sequentially to the token stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerDefinition {
    /// Char filters applied to raw text before tokenization.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub char_filters: Vec<CharFilterConfig>,

    /// The tokenizer that splits text into tokens.
    pub tokenizer: TokenizerConfig,

    /// Token filters applied to the token stream after tokenization.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub token_filters: Vec<TokenFilterConfig>,
}

/// Configuration for a tokenizer component.
///
/// Uses `{"type": "..."}` JSON format via serde's internally tagged
/// representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TokenizerConfig {
    /// Splits on whitespace boundaries.
    Whitespace,

    /// Splits on Unicode word boundaries.
    UnicodeWord,

    /// Splits using a regular expression pattern.
    Regex {
        /// The regex pattern (default: `\w+`).
        #[serde(default = "default_regex_pattern")]
        pattern: String,

        /// If `true`, the pattern matches gaps between tokens
        /// rather than the tokens themselves.
        #[serde(default)]
        gaps: bool,
    },

    /// Produces n-grams of the specified size range.
    Ngram {
        /// Minimum n-gram size.
        min_gram: usize,
        /// Maximum n-gram size.
        max_gram: usize,
    },

    /// Morphological tokenizer using Lindera.
    Lindera {
        /// Tokenization mode: `"normal"` or `"decompose"`.
        mode: String,
        /// Dictionary URI. In production builds, supply a filesystem path
        /// to a Lindera dictionary directory (e.g. `"/var/lib/lindera/ipadic"`).
        /// `embedded://*` URIs are only resolvable when the matching
        /// `embed-*` Lindera feature is enabled, which `laurus` does not do
        /// by default.
        dict: String,
        /// Optional user dictionary URI (filesystem path).
        #[serde(default)]
        user_dict: Option<String>,
    },

    /// Treats the entire input as a single token.
    Whole,
}

/// Configuration for a char filter component.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CharFilterConfig {
    /// Applies Unicode normalization (NFC, NFD, NFKC, or NFKD).
    UnicodeNormalization {
        /// Normalization form: `"nfc"`, `"nfd"`, `"nfkc"`, or `"nfkd"`.
        form: String,
    },

    /// Replaces text matching a regex pattern.
    PatternReplace {
        /// The regex pattern to match.
        pattern: String,
        /// The replacement string.
        replacement: String,
    },

    /// Replaces strings using a mapping dictionary.
    Mapping {
        /// Key-value pairs for replacement. A `BTreeMap` (not `HashMap`)
        /// so schema serialization stays deterministic (Issue #1060).
        mapping: BTreeMap<String, String>,
    },

    /// Expands Japanese iteration marks (踊り字).
    JapaneseIterationMark {
        /// Whether to normalize kanji iteration marks.
        #[serde(default = "default_true")]
        kanji: bool,
        /// Whether to normalize kana iteration marks.
        #[serde(default = "default_true")]
        kana: bool,
    },
}

/// Configuration for a token filter component.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TokenFilterConfig {
    /// Converts tokens to lowercase.
    Lowercase,

    /// Removes stop words from the token stream.
    Stop {
        /// Custom stop word list. If `None`, uses default English
        /// stop words.
        #[serde(default)]
        words: Option<Vec<String>>,
    },

    /// Applies stemming to tokens.
    Stem {
        /// Stemmer type: `"porter"` (default), `"simple"`, or
        /// `"identity"`.
        #[serde(default)]
        stem_type: Option<String>,
    },

    /// Multiplies token scores by a boost factor.
    Boost {
        /// The boost multiplier.
        boost: f32,
    },

    /// Limits the number of tokens in the stream.
    Limit {
        /// Maximum number of tokens to emit.
        limit: usize,
    },

    /// Strips leading and trailing whitespace from tokens.
    Strip,

    /// Removes empty tokens from the stream.
    RemoveEmpty,

    /// Flattens a synonym graph into a linear token stream. Indexing already
    /// does this, and since the analyzer also parses queries, the filter makes
    /// quoted multi-word synonyms inexact there.
    FlattenGraph,
}

fn default_regex_pattern() -> String {
    r"\w+".to_string()
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_definition_serde_roundtrip() {
        let def = AnalyzerDefinition {
            char_filters: vec![CharFilterConfig::UnicodeNormalization {
                form: "nfkc".into(),
            }],
            tokenizer: TokenizerConfig::Regex {
                pattern: r"\w+".into(),
                gaps: false,
            },
            token_filters: vec![
                TokenFilterConfig::Lowercase,
                TokenFilterConfig::Stop {
                    words: Some(vec!["the".into(), "a".into()]),
                },
                TokenFilterConfig::Stem { stem_type: None },
            ],
        };

        let json = serde_json::to_string(&def).unwrap();
        let deserialized: AnalyzerDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.token_filters.len(), 3);
        assert_eq!(deserialized.char_filters.len(), 1);
    }

    #[test]
    fn test_tokenizer_config_variants() {
        let configs = vec![
            r#"{"type": "whitespace"}"#,
            r#"{"type": "unicode_word"}"#,
            r#"{"type": "regex", "pattern": "\\w+", "gaps": false}"#,
            r#"{"type": "ngram", "min_gram": 2, "max_gram": 3}"#,
            r#"{"type": "whole"}"#,
        ];
        for json in configs {
            let config: TokenizerConfig = serde_json::from_str(json).unwrap();
            let serialized = serde_json::to_string(&config).unwrap();
            let _roundtrip: TokenizerConfig = serde_json::from_str(&serialized).unwrap();
        }
    }

    #[test]
    fn test_char_filter_config_variants() {
        let configs = vec![
            r#"{"type": "unicode_normalization", "form": "nfkc"}"#,
            r#"{"type": "pattern_replace", "pattern": "foo", "replacement": "bar"}"#,
            r#"{"type": "mapping", "mapping": {"a": "b"}}"#,
            r#"{"type": "japanese_iteration_mark"}"#,
        ];
        for json in configs {
            let config: CharFilterConfig = serde_json::from_str(json).unwrap();
            let serialized = serde_json::to_string(&config).unwrap();
            let _roundtrip: CharFilterConfig = serde_json::from_str(&serialized).unwrap();
        }
    }

    #[test]
    fn test_token_filter_config_variants() {
        let configs = vec![
            r#"{"type": "lowercase"}"#,
            r#"{"type": "stop"}"#,
            r#"{"type": "stop", "words": ["the", "a"]}"#,
            r#"{"type": "stem"}"#,
            r#"{"type": "stem", "stem_type": "porter"}"#,
            r#"{"type": "boost", "boost": 2.0}"#,
            r#"{"type": "limit", "limit": 100}"#,
            r#"{"type": "strip"}"#,
            r#"{"type": "remove_empty"}"#,
            r#"{"type": "flatten_graph"}"#,
        ];
        for json in configs {
            let config: TokenFilterConfig = serde_json::from_str(json).unwrap();
            let serialized = serde_json::to_string(&config).unwrap();
            let _roundtrip: TokenFilterConfig = serde_json::from_str(&serialized).unwrap();
        }
    }

    #[test]
    fn test_full_schema_with_analyzers_json() {
        let json = r#"{
            "char_filters": [{"type": "unicode_normalization", "form": "nfkc"}],
            "tokenizer": {"type": "lindera", "mode": "normal", "dict": "embedded://ipadic"},
            "token_filters": [{"type": "lowercase"}]
        }"#;
        let def: AnalyzerDefinition = serde_json::from_str(json).unwrap();
        assert!(matches!(def.tokenizer, TokenizerConfig::Lindera { .. }));
    }

    #[test]
    fn test_minimal_definition() {
        let json = r#"{"tokenizer": {"type": "whitespace"}}"#;
        let def: AnalyzerDefinition = serde_json::from_str(json).unwrap();
        assert!(def.char_filters.is_empty());
        assert!(def.token_filters.is_empty());
    }

    #[test]
    fn test_analyzer_spec_named_string_form() {
        let spec: AnalyzerSpec = serde_json::from_str(r#""standard""#).unwrap();
        assert!(matches!(spec, AnalyzerSpec::Named(ref s) if s == "standard"));

        let serialized = serde_json::to_string(&spec).unwrap();
        assert_eq!(serialized, r#""standard""#);
    }

    #[test]
    fn test_analyzer_spec_japanese_struct_form() {
        let json = r#"{"language": "japanese", "dict": "/var/lib/lindera/ipadic"}"#;
        let spec: AnalyzerSpec = serde_json::from_str(json).unwrap();
        match spec {
            AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
                mode,
                dict,
                user_dict,
            }) => {
                assert_eq!(mode, "normal"); // default
                assert_eq!(dict, "/var/lib/lindera/ipadic");
                assert!(user_dict.is_none());
            }
            other => panic!("expected Japanese variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_analyzer_spec_japanese_with_mode_and_user_dict() {
        // "decompose" (not "search" — Lindera 3.x's `Mode::from_str` only
        // accepts "normal"/"decompose"; this crate's doc comments used to
        // claim "search" was valid too, which was a documentation bug).
        let json = r#"{
            "language": "japanese",
            "mode": "decompose",
            "dict": "/var/lib/lindera/ipadic",
            "user_dict": "/etc/laurus/user.csv"
        }"#;
        let spec: AnalyzerSpec = serde_json::from_str(json).unwrap();
        match spec {
            AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
                mode,
                dict,
                user_dict,
            }) => {
                assert_eq!(mode, "decompose");
                assert_eq!(dict, "/var/lib/lindera/ipadic");
                assert_eq!(user_dict.as_deref(), Some("/etc/laurus/user.csv"));
            }
            other => panic!("expected Japanese variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_analyzer_spec_serialize_japanese() {
        let spec = AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
            mode: "normal".into(),
            dict: "/var/lib/lindera/ipadic".into(),
            user_dict: None,
        });
        let serialized = serde_json::to_string(&spec).unwrap();
        // Round-trip and inspect.
        let roundtrip: AnalyzerSpec = serde_json::from_str(&serialized).unwrap();
        assert_eq!(spec, roundtrip);
        // The serialized form must include the language discriminator.
        assert!(serialized.contains(r#""language":"japanese""#));
        // user_dict is None and skipped.
        assert!(!serialized.contains("user_dict"));
    }

    #[test]
    fn test_analyzer_spec_from_str_into() {
        let spec: AnalyzerSpec = "english".into();
        assert!(matches!(spec, AnalyzerSpec::Named(ref s) if s == "english"));
    }
}
