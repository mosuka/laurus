//! Analyzer registry for creating analyzers by name.
//!
//! Provides a lookup function that maps well-known analyzer names to
//! concrete [`Analyzer`] instances. This is used by the engine to
//! construct per-field analyzers from schema declarations.
//!
//! # Supported Analyzers
//!
//! | Name | Description |
//! |------|-------------|
//! | `standard` | Regex tokenizer + lowercase + English stop words |
//! | `keyword` | Treats the entire input as a single token |
//! | `english` | English-optimized (equivalent to `standard`) |
//! | `japanese` | Lindera/IPADIC tokenizer + Japanese stop words |
//! | `simple` | Regex tokenizer only, no filters |
//! | `noop` | Produces no tokens (for stored-only fields) |

use std::collections::HashSet;
use std::sync::Arc;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::keyword::KeywordAnalyzer;
use crate::analysis::analyzer::language::english::EnglishAnalyzer;
use crate::analysis::analyzer::language::japanese::JapaneseAnalyzer;
use crate::analysis::analyzer::noop::NoOpAnalyzer;
use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
use crate::analysis::analyzer::simple::SimpleAnalyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::analysis::char_filter::CharFilter;
use crate::analysis::char_filter::japanese_iteration_mark::JapaneseIterationMarkCharFilter;
use crate::analysis::char_filter::mapping::MappingCharFilter;
use crate::analysis::char_filter::pattern_replace::PatternReplaceCharFilter;
use crate::analysis::char_filter::unicode_normalize::{
    NormalizationForm, UnicodeNormalizationCharFilter,
};
use crate::analysis::token_filter::Filter;
use crate::analysis::token_filter::boost::BoostFilter;
use crate::analysis::token_filter::flatten_graph::FlattenGraphFilter;
use crate::analysis::token_filter::limit::LimitFilter;
use crate::analysis::token_filter::lowercase::LowercaseFilter;
use crate::analysis::token_filter::remove_empty::RemoveEmptyFilter;
use crate::analysis::token_filter::stem::{StemFilter, identity::IdentityStemmer};
use crate::analysis::token_filter::stop::StopFilter;
use crate::analysis::token_filter::strip::StripFilter;
use crate::analysis::tokenizer::Tokenizer;
use crate::analysis::tokenizer::lindera::LinderaTokenizer;
use crate::analysis::tokenizer::ngram::NgramTokenizer;
use crate::analysis::tokenizer::regex::RegexTokenizer;
use crate::analysis::tokenizer::unicode_word::UnicodeWordTokenizer;
use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use crate::analysis::tokenizer::whole::WholeTokenizer;
use crate::engine::schema::analyzer::{
    AnalyzerDefinition, AnalyzerSpec, BuiltinAnalyzerSpec, CharFilterConfig, TokenFilterConfig,
    TokenizerConfig,
};
use crate::error::{LaurusError, Result};

/// Names of the built-in analyzers that [`create_analyzer_by_name`]
/// resolves before `schema.analyzers` is consulted, so a custom definition
/// under one of them would never be used.
///
/// `"japanese"` is not listed: it is a built-in, but
/// [`create_analyzer_by_name`] rejects it, so `schema.analyzers["japanese"]`
/// is reachable.
pub const RESERVED_ANALYZER_NAMES: [&str; 5] = ["standard", "keyword", "english", "simple", "noop"];

/// Returns `true` if `name` is in [`RESERVED_ANALYZER_NAMES`].
///
/// # Arguments
///
/// * `name` - The analyzer name to check.
pub fn is_reserved_analyzer_name(name: &str) -> bool {
    RESERVED_ANALYZER_NAMES.contains(&name)
}

/// Validates that a custom analyzer definition does not use a name
/// reserved for a built-in analyzer.
///
/// # Arguments
///
/// * `name` - The name of the custom analyzer definition.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] if `name` is in
/// [`RESERVED_ANALYZER_NAMES`].
pub fn validate_analyzer_name(name: &str) -> Result<()> {
    if is_reserved_analyzer_name(name) {
        return Err(LaurusError::invalid_argument(format!(
            "Analyzer name '{name}' is reserved for a built-in analyzer; choose another name"
        )));
    }
    Ok(())
}

/// Create an analyzer instance by its well-known name.
///
/// Built-in analyzers that take no parameters are resolved by name.
/// `"japanese"` is **not** included here because it requires a Lindera
/// dictionary path; use [`create_analyzer_from_spec`] with
/// [`BuiltinAnalyzerSpec::Japanese`] instead.
///
/// The names this function resolves must match
/// [`RESERVED_ANALYZER_NAMES`].
///
/// # Arguments
///
/// * `name` - The analyzer name (e.g. `"standard"`).
///
/// # Returns
///
/// An `Arc<dyn Analyzer>` wrapping the requested analyzer.
///
/// # Errors
///
/// Returns an error if `name` is not a recognized analyzer name.
pub fn create_analyzer_by_name(name: &str) -> Result<Arc<dyn Analyzer>> {
    match name {
        "standard" => Ok(Arc::new(StandardAnalyzer::new()?)),
        "keyword" => Ok(Arc::new(KeywordAnalyzer::new())),
        "english" => Ok(Arc::new(EnglishAnalyzer::new()?)),
        "japanese" => Err(LaurusError::invalid_argument(
            "Analyzer 'japanese' requires a dictionary path. Specify it via \
             { \"language\": \"japanese\", \"dict\": \"/path/to/ipadic\" } in the schema.",
        )),
        "simple" => Ok(Arc::new(SimpleAnalyzer::new(Arc::new(
            RegexTokenizer::new()?,
        )))),
        "noop" => Ok(Arc::new(NoOpAnalyzer::new())),
        unknown => Err(LaurusError::invalid_argument(format!(
            "Unknown analyzer: {unknown}"
        ))),
    }
}

/// Resolve an [`AnalyzerSpec`] from a schema into a concrete analyzer.
///
/// Resolution order:
/// 1. If `spec` is [`AnalyzerSpec::Builtin`], construct the
///    parameterized built-in directly.
/// 2. If `spec` is [`AnalyzerSpec::Named`]:
///    1. Look up `runtime_analyzers` first (pre-constructed analyzers
///       registered on the engine, e.g. from byte arrays in a WASM
///       environment).
///    2. Try the name as a parameter-less built-in via
///       [`create_analyzer_by_name`].
///    3. Fall back to a user-defined [`AnalyzerDefinition`] in
///       `schema_analyzers`.
///
/// # Arguments
///
/// * `spec` - The analyzer reference from a text field option.
/// * `schema_analyzers` - Custom analyzer definitions registered on the
///   schema (serialized form).
/// * `runtime_analyzers` - Pre-constructed analyzers registered at
///   runtime via [`EngineBuilder::register_runtime_analyzer`]. These
///   take precedence over named built-ins, so a runtime registration
///   under `"keyword"` would shadow the built-in. Pass an empty map
///   when no runtime analyzers are needed.
///
/// # Errors
///
/// Returns an error if no resolution path applies (unknown name with
/// no runtime, built-in, or schema entry) or if dictionary loading
/// fails for a built-in preset.
///
/// [`EngineBuilder::register_runtime_analyzer`]: crate::engine::EngineBuilder::register_runtime_analyzer
pub fn create_analyzer_from_spec(
    spec: &AnalyzerSpec,
    schema_analyzers: &std::collections::BTreeMap<String, AnalyzerDefinition>,
    runtime_analyzers: &std::collections::HashMap<String, Arc<dyn Analyzer>>,
) -> Result<Arc<dyn Analyzer>> {
    match spec {
        AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
            mode,
            dict,
            user_dict,
        }) => Ok(Arc::new(JapaneseAnalyzer::new(
            mode,
            dict,
            user_dict.as_deref(),
        )?)),
        AnalyzerSpec::Named(name) => {
            if let Some(analyzer) = runtime_analyzers.get(name) {
                return Ok(analyzer.clone());
            }
            match create_analyzer_by_name(name) {
                Ok(analyzer) => Ok(analyzer),
                Err(_) => {
                    let def = schema_analyzers.get(name).ok_or_else(|| {
                        LaurusError::invalid_argument(format!(
                            "Unknown analyzer '{name}': not registered at runtime, \
                             not a built-in, and not defined in schema.analyzers"
                        ))
                    })?;
                    create_analyzer_from_definition(name, def)
                }
            }
        }
    }
}

/// Create an analyzer from a custom definition.
///
/// Builds a [`PipelineAnalyzer`] by constructing the tokenizer,
/// char filters, and token filters according to the given
/// [`AnalyzerDefinition`].
///
/// # Arguments
///
/// * `name` - The name to assign to the resulting analyzer.
/// * `definition` - The analyzer pipeline definition.
///
/// # Returns
///
/// An `Arc<dyn Analyzer>` wrapping the constructed pipeline.
///
/// # Errors
///
/// Returns an error if any component configuration is invalid
/// (e.g. bad regex pattern, unknown stemmer type).
pub fn create_analyzer_from_definition(
    name: &str,
    definition: &AnalyzerDefinition,
) -> Result<Arc<dyn Analyzer>> {
    // 1. Build tokenizer.
    let tokenizer: Arc<dyn Tokenizer> = match &definition.tokenizer {
        TokenizerConfig::Whitespace => Arc::new(WhitespaceTokenizer::new()),
        TokenizerConfig::UnicodeWord => Arc::new(UnicodeWordTokenizer::new()),
        TokenizerConfig::Regex { pattern, gaps } => {
            if *gaps {
                Arc::new(RegexTokenizer::with_gaps(pattern)?)
            } else {
                Arc::new(RegexTokenizer::with_pattern(pattern)?)
            }
        }
        TokenizerConfig::Ngram { min_gram, max_gram } => {
            Arc::new(NgramTokenizer::new(*min_gram, *max_gram)?)
        }
        TokenizerConfig::Lindera {
            mode,
            dict,
            user_dict,
        } => Arc::new(LinderaTokenizer::new(mode, dict, user_dict.as_deref())?),
        TokenizerConfig::Whole => Arc::new(WholeTokenizer::new()),
    };

    // 2. Build pipeline.
    let mut pipeline = PipelineAnalyzer::new(tokenizer).with_name(name.to_string());

    // 3. Add char filters.
    for cf_config in &definition.char_filters {
        let cf: Arc<dyn CharFilter> = match cf_config {
            CharFilterConfig::UnicodeNormalization { form } => {
                let nf = match form.to_lowercase().as_str() {
                    "nfc" => NormalizationForm::NFC,
                    "nfd" => NormalizationForm::NFD,
                    "nfkc" => NormalizationForm::NFKC,
                    "nfkd" => NormalizationForm::NFKD,
                    _ => {
                        return Err(LaurusError::invalid_argument(format!(
                            "Unknown normalization form: {form}"
                        )));
                    }
                };
                Arc::new(UnicodeNormalizationCharFilter::new(nf))
            }
            CharFilterConfig::PatternReplace {
                pattern,
                replacement,
            } => Arc::new(PatternReplaceCharFilter::new(pattern, replacement)?),
            CharFilterConfig::Mapping { mapping } => {
                Arc::new(MappingCharFilter::new(mapping.clone())?)
            }
            CharFilterConfig::JapaneseIterationMark { kanji, kana } => {
                Arc::new(JapaneseIterationMarkCharFilter::new(*kanji, *kana))
            }
        };
        pipeline = pipeline.add_char_filter(cf);
    }

    // 4. Add token filters.
    for tf_config in &definition.token_filters {
        let tf: Arc<dyn Filter> = match tf_config {
            TokenFilterConfig::Lowercase => Arc::new(LowercaseFilter::new()),
            TokenFilterConfig::Stop { words } => {
                if let Some(word_list) = words {
                    let set: HashSet<String> = word_list.iter().cloned().collect();
                    Arc::new(StopFilter::with_stop_words(set))
                } else {
                    Arc::new(StopFilter::new())
                }
            }
            TokenFilterConfig::Stem { stem_type } => {
                let stemmer_name = stem_type.as_deref().unwrap_or("porter");
                match stemmer_name {
                    "porter" => Arc::new(StemFilter::new()),
                    "simple" => Arc::new(StemFilter::simple()),
                    "identity" => {
                        Arc::new(StemFilter::with_stemmer(Box::new(IdentityStemmer::new())))
                    }
                    _ => {
                        return Err(LaurusError::invalid_argument(format!(
                            "Unknown stemmer type: {stemmer_name}"
                        )));
                    }
                }
            }
            TokenFilterConfig::Boost { boost } => Arc::new(BoostFilter::new(*boost)),
            TokenFilterConfig::Limit { limit } => Arc::new(LimitFilter::new(*limit)),
            TokenFilterConfig::Strip => Arc::new(StripFilter::new()),
            TokenFilterConfig::RemoveEmpty => Arc::new(RemoveEmptyFilter::new()),
            TokenFilterConfig::FlattenGraph => Arc::new(FlattenGraphFilter::new()),
        };
        pipeline = pipeline.add_filter(tf);
    }

    Ok(Arc::new(pipeline))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_standard() {
        let analyzer = create_analyzer_by_name("standard").unwrap();
        assert_eq!(analyzer.name(), "standard");
    }

    #[test]
    fn test_create_keyword() {
        let analyzer = create_analyzer_by_name("keyword").unwrap();
        assert_eq!(analyzer.name(), "keyword");
    }

    #[test]
    fn test_create_english() {
        let analyzer = create_analyzer_by_name("english").unwrap();
        assert_eq!(analyzer.name(), "english");
    }

    #[test]
    fn test_create_japanese_by_name_returns_error() {
        // Japanese requires a dictionary path and must be resolved through
        // create_analyzer_from_spec with a Japanese builtin spec.
        let result = create_analyzer_by_name("japanese");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("dictionary path"), "got: {msg}");
    }

    #[test]
    fn test_create_japanese_from_spec() {
        let spec = AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
            mode: "normal".into(),
            dict: "embedded://ipadic".into(),
            user_dict: None,
        });
        let analyzer =
            create_analyzer_from_spec(&spec, &Default::default(), &Default::default()).unwrap();
        assert_eq!(analyzer.name(), "japanese");
    }

    #[test]
    fn test_create_named_from_spec_falls_back_to_schema_analyzers() {
        let spec = AnalyzerSpec::Named("my_custom".into());
        let mut analyzers = std::collections::BTreeMap::new();
        analyzers.insert(
            "my_custom".to_string(),
            AnalyzerDefinition {
                char_filters: vec![],
                tokenizer: TokenizerConfig::Whitespace,
                token_filters: vec![],
            },
        );
        let analyzer = create_analyzer_from_spec(&spec, &analyzers, &Default::default()).unwrap();
        assert_eq!(analyzer.name(), "my_custom");
    }

    #[test]
    fn test_create_named_from_spec_resolves_builtin() {
        let spec = AnalyzerSpec::Named("standard".into());
        let analyzer =
            create_analyzer_from_spec(&spec, &Default::default(), &Default::default()).unwrap();
        assert_eq!(analyzer.name(), "standard");
    }

    #[test]
    fn test_runtime_analyzer_takes_precedence() {
        // Register a custom analyzer under a name that overlaps with a
        // built-in to confirm runtime resolution beats built-in lookup.
        let custom: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
        let mut runtime = std::collections::HashMap::new();
        runtime.insert("standard".to_string(), custom.clone());

        let spec = AnalyzerSpec::Named("standard".into());
        let resolved = create_analyzer_from_spec(&spec, &Default::default(), &runtime).unwrap();

        // The runtime registration (KeywordAnalyzer) wins over the
        // built-in StandardAnalyzer.
        assert_eq!(resolved.name(), "keyword");
        assert!(Arc::ptr_eq(&resolved, &custom));
    }

    #[test]
    fn test_runtime_analyzer_named_lookup() {
        // A name not present anywhere except the runtime registry must
        // resolve via the runtime registry.
        let custom: Arc<dyn Analyzer> = Arc::new(KeywordAnalyzer::new());
        let mut runtime = std::collections::HashMap::new();
        runtime.insert("ja-ipadic".to_string(), custom.clone());

        let spec = AnalyzerSpec::Named("ja-ipadic".into());
        let resolved = create_analyzer_from_spec(&spec, &Default::default(), &runtime).unwrap();
        assert!(Arc::ptr_eq(&resolved, &custom));
    }

    #[test]
    fn test_create_simple() {
        let analyzer = create_analyzer_by_name("simple").unwrap();
        assert_eq!(analyzer.name(), "simple");
    }

    #[test]
    fn test_create_noop() {
        let analyzer = create_analyzer_by_name("noop").unwrap();
        assert_eq!(analyzer.name(), "noop");
    }

    #[test]
    fn test_unknown_returns_error() {
        let result = create_analyzer_by_name("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_reserved_analyzer_names_all_resolve_by_name() {
        for name in RESERVED_ANALYZER_NAMES {
            assert!(
                create_analyzer_by_name(name).is_ok(),
                "'{name}' is reserved but create_analyzer_by_name does not resolve it"
            );
        }
    }

    #[test]
    fn test_validate_analyzer_name_rejects_reserved_names() {
        for name in RESERVED_ANALYZER_NAMES {
            let msg = validate_analyzer_name(name).unwrap_err().to_string();
            assert!(
                msg.contains(&format!(
                    "Analyzer name '{name}' is reserved for a built-in analyzer"
                )),
                "got: {msg}"
            );
        }
    }

    #[test]
    fn test_validate_analyzer_name_accepts_non_reserved_names() {
        // `japanese` is a built-in, but create_analyzer_by_name rejects it,
        // so schema.analyzers["japanese"] is reachable and must stay allowed.
        for name in ["japanese", "Standard", "my_custom"] {
            assert!(validate_analyzer_name(name).is_ok(), "'{name}' rejected");
        }
    }

    #[test]
    fn test_create_from_definition_whitespace_lowercase() {
        let def = AnalyzerDefinition {
            char_filters: vec![],
            tokenizer: TokenizerConfig::Whitespace,
            token_filters: vec![TokenFilterConfig::Lowercase],
        };
        let analyzer = create_analyzer_from_definition("my_ws", &def).unwrap();
        assert_eq!(analyzer.name(), "my_ws");
        let tokens: Vec<_> = analyzer.analyze("Hello World").unwrap().collect();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_create_from_definition_with_stop_words() {
        let def = AnalyzerDefinition {
            char_filters: vec![],
            tokenizer: TokenizerConfig::Regex {
                pattern: r"\w+".into(),
                gaps: false,
            },
            token_filters: vec![
                TokenFilterConfig::Lowercase,
                TokenFilterConfig::Stop {
                    words: Some(vec!["the".into(), "a".into()]),
                },
            ],
        };
        let analyzer = create_analyzer_from_definition("custom_stop", &def).unwrap();
        let tokens: Vec<_> = analyzer.analyze("The quick brown fox").unwrap().collect();
        // "The" is filtered out (lowercased to "the", then stopped).
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_create_from_definition_with_char_filter() {
        let def = AnalyzerDefinition {
            char_filters: vec![CharFilterConfig::UnicodeNormalization {
                form: "nfkc".into(),
            }],
            tokenizer: TokenizerConfig::Whitespace,
            token_filters: vec![],
        };
        let analyzer = create_analyzer_from_definition("nfkc_analyzer", &def).unwrap();
        // Fullwidth "Ａ" should be normalized to "A".
        let tokens: Vec<_> = analyzer.analyze("\u{ff21}").unwrap().collect();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "A");
    }

    #[test]
    fn test_create_from_definition_whole_tokenizer() {
        let def = AnalyzerDefinition {
            char_filters: vec![],
            tokenizer: TokenizerConfig::Whole,
            token_filters: vec![],
        };
        let analyzer = create_analyzer_from_definition("exact", &def).unwrap();
        let tokens: Vec<_> = analyzer.analyze("Hello World").unwrap().collect();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].text, "Hello World");
    }
}
