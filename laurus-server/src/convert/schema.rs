//! Conversion between [`laurus::Schema`] and the protobuf `Schema` message.
//!
//! Handles mapping of all field option variants (text, integer, float, boolean,
//! datetime, geo, bytes, HNSW, flat, IVF), distance metrics, and quantization
//! configuration.

use std::collections::{BTreeMap, HashMap};

use laurus::vector::core::rerank::RerankStorageKind;
use laurus::{
    AnalyzerDefinition, AnalyzerSpec, BooleanOption, BuiltinAnalyzerSpec, BytesOption,
    CharFilterConfig, DateTimeOption, DistanceMetric, DynamicFieldPolicy, EmbedderDefinition,
    FieldChangeKind, FieldOption, FlatOption, FloatOption, Geo3dOption, GeoOption, HnswOption,
    IntegerOption, IvfOption, QuantizationMethod, Schema, TextOption, TokenFilterConfig,
    TokenizerConfig,
};

use crate::proto::laurus::v1;

/// Convert a laurus Schema into a proto Schema.
pub fn to_proto(schema: &Schema) -> v1::Schema {
    let fields: HashMap<String, v1::FieldOption> = schema
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), field_option_to_proto(v)))
        .collect();
    let analyzers = schema
        .analyzers
        .iter()
        .map(|(k, v)| (k.clone(), analyzer_definition_to_proto(v)))
        .collect();
    let embedders = schema
        .embedders
        .iter()
        .map(|(k, v)| (k.clone(), embedder_definition_to_proto(v)))
        .collect();
    v1::Schema {
        fields,
        default_fields: schema.default_fields.clone(),
        analyzers,
        embedders,
        dynamic_field_policy: dynamic_field_policy_to_proto(&schema.dynamic_field_policy) as i32,
        pending_reindex: schema.pending_reindex.iter().cloned().collect(),
    }
}

/// Convert a proto Schema into a laurus Schema.
pub fn from_proto(proto: &v1::Schema) -> Result<Schema, String> {
    let mut fields = BTreeMap::new();
    for (name, fo) in &proto.fields {
        let option = field_option_from_proto(fo)
            .ok_or_else(|| format!("Field '{name}' has no option set"))?;
        fields.insert(name.clone(), option);
    }
    let mut analyzers = BTreeMap::new();
    for (name, ad) in &proto.analyzers {
        analyzers.insert(name.clone(), analyzer_definition_from_proto(ad)?);
    }
    let mut embedders = BTreeMap::new();
    for (name, ed) in &proto.embedders {
        embedders.insert(name.clone(), embedder_definition_from_proto(ed)?);
    }
    Ok(Schema {
        analyzers,
        embedders,
        fields,
        default_fields: proto.default_fields.clone(),
        dynamic_field_policy: dynamic_field_policy_from_proto(proto.dynamic_field_policy),
        pending_reindex: proto.pending_reindex.iter().cloned().collect(),
    })
}

/// Convert a laurus `DynamicFieldPolicy` into a proto enum value.
///
/// # Arguments
///
/// * `policy` - The laurus dynamic field policy.
fn dynamic_field_policy_to_proto(policy: &DynamicFieldPolicy) -> v1::DynamicFieldPolicy {
    match policy {
        DynamicFieldPolicy::Strict => v1::DynamicFieldPolicy::Strict,
        DynamicFieldPolicy::Dynamic => v1::DynamicFieldPolicy::Dynamic,
        DynamicFieldPolicy::Ignore => v1::DynamicFieldPolicy::Ignore,
    }
}

/// Convert a proto `DynamicFieldPolicy` value into a laurus `DynamicFieldPolicy`.
///
/// The proto value `UNSPECIFIED` (zero) and any unrecognised value are mapped to
/// [`DynamicFieldPolicy::Dynamic`], matching the laurus default.
///
/// # Arguments
///
/// * `value` - The proto enum value (i32).
fn dynamic_field_policy_from_proto(value: i32) -> DynamicFieldPolicy {
    match v1::DynamicFieldPolicy::try_from(value) {
        Ok(v1::DynamicFieldPolicy::Strict) => DynamicFieldPolicy::Strict,
        Ok(v1::DynamicFieldPolicy::Dynamic) => DynamicFieldPolicy::Dynamic,
        Ok(v1::DynamicFieldPolicy::Ignore) => DynamicFieldPolicy::Ignore,
        Ok(v1::DynamicFieldPolicy::Unspecified) | Err(_) => DynamicFieldPolicy::default(),
    }
}

/// Convert a laurus `FieldChangeKind` (an `Engine::update_field` outcome)
/// into a proto enum value.
///
/// # Arguments
///
/// * `kind` - The laurus field change classification.
pub fn field_change_kind_to_proto(kind: FieldChangeKind) -> v1::FieldChangeKind {
    match kind {
        FieldChangeKind::MetadataOnly => v1::FieldChangeKind::MetadataOnly,
        FieldChangeKind::Reindex => v1::FieldChangeKind::Reindex,
        FieldChangeKind::Destructive => v1::FieldChangeKind::Destructive,
    }
}

/// Convert a laurus `FieldOption` into a proto `FieldOption`.
///
/// # Arguments
///
/// * `fo` - The laurus field option to convert.
pub fn field_option_to_proto(fo: &FieldOption) -> v1::FieldOption {
    use v1::field_option::Option as Opt;
    let option = match fo {
        FieldOption::Text(o) => Some(Opt::Text(v1::TextOption {
            indexed: o.indexed,
            stored: o.stored,
            term_vectors: Some(o.term_vectors),
            analyzer: o.analyzer.as_ref().map(analyzer_spec_to_proto),
        })),
        FieldOption::Integer(o) => Some(Opt::Integer(v1::IntegerOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
        })),
        FieldOption::Float(o) => Some(Opt::Float(v1::FloatOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
        })),
        FieldOption::Boolean(o) => Some(Opt::Boolean(v1::BooleanOption {
            indexed: o.indexed,
            stored: o.stored,
        })),
        FieldOption::DateTime(o) => Some(Opt::DateTime(v1::DateTimeOption {
            indexed: o.indexed,
            stored: o.stored,
        })),
        FieldOption::Geo(o) => Some(Opt::Geo(v1::GeoOption {
            indexed: o.indexed,
            stored: o.stored,
        })),
        FieldOption::Geo3d(o) => Some(Opt::Geo3d(v1::Geo3dOption {
            indexed: o.indexed,
            stored: o.stored,
        })),
        FieldOption::Bytes(o) => Some(Opt::Bytes(v1::BytesOption { stored: o.stored })),
        FieldOption::Hnsw(o) => Some(Opt::Hnsw(v1::HnswOption {
            dimension: o.dimension as u32,
            distance: distance_to_proto(&o.distance) as i32,
            m: o.m as u32,
            ef_construction: o.ef_construction as u32,
            base_weight: Some(o.base_weight),
            quantizer: Some(quantization_to_proto(&o.quantizer)),
            embedder: o.embedder.clone().unwrap_or_default(),
            default_ef_search: o.default_ef_search.map(|v| v as u32),
            rerank_storage: o.rerank_storage.map(|k| rerank_storage_to_proto(k) as i32),
            pq_codebook_path: o.pq_codebook_path.clone(),
        })),
        FieldOption::Flat(o) => Some(Opt::Flat(v1::FlatOption {
            dimension: o.dimension as u32,
            distance: distance_to_proto(&o.distance) as i32,
            base_weight: Some(o.base_weight),
            quantizer: Some(quantization_to_proto(&o.quantizer)),
            embedder: o.embedder.clone().unwrap_or_default(),
            rerank_storage: o.rerank_storage.map(|k| rerank_storage_to_proto(k) as i32),
        })),
        FieldOption::Ivf(o) => Some(Opt::Ivf(v1::IvfOption {
            dimension: o.dimension as u32,
            distance: distance_to_proto(&o.distance) as i32,
            n_clusters: o.n_clusters as u32,
            n_probe: o.n_probe as u32,
            base_weight: Some(o.base_weight),
            quantizer: Some(quantization_to_proto(&o.quantizer)),
            embedder: o.embedder.clone().unwrap_or_default(),
            rerank_storage: o.rerank_storage.map(|k| rerank_storage_to_proto(k) as i32),
        })),
    };
    v1::FieldOption { option }
}

/// Convert a proto `FieldOption` into a laurus `FieldOption`.
///
/// Returns `None` if the proto message has no option variant set.
///
/// # Arguments
///
/// * `fo` - The proto field option to convert.
pub fn field_option_from_proto(fo: &v1::FieldOption) -> Option<FieldOption> {
    use v1::field_option::Option as Opt;
    match &fo.option {
        Some(Opt::Text(o)) => Some(FieldOption::Text(TextOption {
            indexed: o.indexed,
            stored: o.stored,
            // Unset means "use the engine's default", matching
            // `TextOption::default()` (#1083).
            term_vectors: o.term_vectors.unwrap_or(true),
            // The proto message has no `doc_values` field yet; pin to the
            // pre-#1047 implicit default until the protocol is wired up.
            doc_values: true,
            analyzer: o.analyzer.as_ref().and_then(analyzer_spec_from_proto),
        })),
        Some(Opt::Integer(o)) => Some(FieldOption::Integer(IntegerOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: true,
        })),
        Some(Opt::Float(o)) => Some(FieldOption::Float(FloatOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: true,
        })),
        Some(Opt::Boolean(o)) => Some(FieldOption::Boolean(BooleanOption {
            indexed: o.indexed,
            stored: o.stored,
            doc_values: true,
        })),
        Some(Opt::DateTime(o)) => Some(FieldOption::DateTime(DateTimeOption {
            indexed: o.indexed,
            stored: o.stored,
            doc_values: true,
        })),
        Some(Opt::Geo(o)) => Some(FieldOption::Geo(GeoOption {
            indexed: o.indexed,
            stored: o.stored,
            doc_values: true,
        })),
        Some(Opt::Geo3d(o)) => Some(FieldOption::Geo3d(Geo3dOption {
            indexed: o.indexed,
            stored: o.stored,
            doc_values: true,
        })),
        Some(Opt::Bytes(o)) => Some(FieldOption::Bytes(BytesOption { stored: o.stored })),
        Some(Opt::Hnsw(o)) => Some(FieldOption::Hnsw(HnswOption {
            dimension: o.dimension as usize,
            distance: distance_from_proto(o.distance),
            m: o.m as usize,
            ef_construction: o.ef_construction as usize,
            default_ef_search: o.default_ef_search.map(|v| v as usize),
            // Unset means "use the engine's default" (#1084).
            base_weight: o.base_weight.unwrap_or(1.0),
            quantizer: o
                .quantizer
                .as_ref()
                .map(quantization_from_proto)
                .unwrap_or_default(),
            rerank_storage: o.rerank_storage.and_then(rerank_storage_from_proto),
            embedder: if o.embedder.is_empty() {
                None
            } else {
                Some(o.embedder.clone())
            },
            // An explicitly empty path would configure a codebook that can
            // never exist; normalize to "unset" (same defensive convention
            // as `embedder` above).
            pq_codebook_path: o.pq_codebook_path.clone().filter(|p| !p.is_empty()),
        })),
        Some(Opt::Flat(o)) => Some(FieldOption::Flat(FlatOption {
            dimension: o.dimension as usize,
            distance: distance_from_proto(o.distance),
            // Unset means "use the engine's default" (#1084).
            base_weight: o.base_weight.unwrap_or(1.0),
            quantizer: o
                .quantizer
                .as_ref()
                .map(quantization_from_proto)
                .unwrap_or_default(),
            rerank_storage: o.rerank_storage.and_then(rerank_storage_from_proto),
            embedder: if o.embedder.is_empty() {
                None
            } else {
                Some(o.embedder.clone())
            },
        })),
        Some(Opt::Ivf(o)) => Some(FieldOption::Ivf(IvfOption {
            dimension: o.dimension as usize,
            distance: distance_from_proto(o.distance),
            n_clusters: o.n_clusters as usize,
            n_probe: o.n_probe as usize,
            // Unset means "use the engine's default" (#1084).
            base_weight: o.base_weight.unwrap_or(1.0),
            quantizer: o
                .quantizer
                .as_ref()
                .map(quantization_from_proto)
                .unwrap_or_default(),
            rerank_storage: o.rerank_storage.and_then(rerank_storage_from_proto),
            embedder: if o.embedder.is_empty() {
                None
            } else {
                Some(o.embedder.clone())
            },
        })),
        None => None,
    }
}

fn distance_to_proto(d: &DistanceMetric) -> v1::DistanceMetric {
    match d {
        DistanceMetric::Cosine => v1::DistanceMetric::Cosine,
        DistanceMetric::Euclidean => v1::DistanceMetric::Euclidean,
        DistanceMetric::Manhattan => v1::DistanceMetric::Manhattan,
        DistanceMetric::DotProduct => v1::DistanceMetric::DotProduct,
        DistanceMetric::Angular => v1::DistanceMetric::Angular,
    }
}

fn distance_from_proto(d: i32) -> DistanceMetric {
    match v1::DistanceMetric::try_from(d) {
        Ok(v1::DistanceMetric::Cosine) => DistanceMetric::Cosine,
        Ok(v1::DistanceMetric::Euclidean) => DistanceMetric::Euclidean,
        Ok(v1::DistanceMetric::Manhattan) => DistanceMetric::Manhattan,
        Ok(v1::DistanceMetric::DotProduct) => DistanceMetric::DotProduct,
        Ok(v1::DistanceMetric::Angular) => DistanceMetric::Angular,
        Err(_) => DistanceMetric::Cosine,
    }
}

fn quantization_to_proto(q: &QuantizationMethod) -> v1::QuantizationConfig {
    match q {
        QuantizationMethod::Scalar8Bit => v1::QuantizationConfig {
            method: v1::QuantizationMethod::Scalar8bit as i32,
            subvector_count: 0,
        },
        QuantizationMethod::ProductQuantization { subvector_count } => v1::QuantizationConfig {
            method: v1::QuantizationMethod::ProductQuantization as i32,
            subvector_count: *subvector_count as u32,
        },
        #[cfg(feature = "pq-fastscan")]
        QuantizationMethod::ProductQuantizationFastScan { subvector_count } => {
            v1::QuantizationConfig {
                method: v1::QuantizationMethod::ProductQuantizationFastscan as i32,
                subvector_count: *subvector_count as u32,
            }
        }
    }
}

fn quantization_from_proto(q: &v1::QuantizationConfig) -> QuantizationMethod {
    // Issue #481 Stage 1 removed the unquantized variant from the Rust enum.
    // The proto wire format still carries `None = 0` for backward compatibility,
    // but it is silently mapped to `Scalar8Bit` (the new default) to preserve
    // forward compatibility for older clients that omit the quantizer field.
    match v1::QuantizationMethod::try_from(q.method) {
        Ok(v1::QuantizationMethod::Scalar8bit) | Ok(v1::QuantizationMethod::None) | Err(_) => {
            QuantizationMethod::Scalar8Bit
        }
        Ok(v1::QuantizationMethod::ProductQuantization) => {
            QuantizationMethod::ProductQuantization {
                subvector_count: q.subvector_count as usize,
            }
        }
        #[cfg(feature = "pq-fastscan")]
        Ok(v1::QuantizationMethod::ProductQuantizationFastscan) => {
            QuantizationMethod::ProductQuantizationFastScan {
                subvector_count: q.subvector_count as usize,
            }
        }
        // Feature off: a client that sends FastScan over gRPC will be
        // silently downgraded to scalar quantization. Returning an
        // error here would require changing the function signature
        // across many call sites; we accept the lossy mapping for
        // Phase 1 and revisit when FastScan exits experimental.
        #[cfg(not(feature = "pq-fastscan"))]
        Ok(v1::QuantizationMethod::ProductQuantizationFastscan) => QuantizationMethod::Scalar8Bit,
    }
}

/// Convert a laurus [`RerankStorageKind`] into the proto enum (Issue #793).
fn rerank_storage_to_proto(kind: RerankStorageKind) -> v1::RerankStorageKind {
    match kind {
        RerankStorageKind::F32 => v1::RerankStorageKind::F32,
    }
}

/// Convert a proto `rerank_storage` enum value into an optional laurus
/// [`RerankStorageKind`] (Issue #793).
///
/// `UNSPECIFIED` and any unknown value map to `None` (Stage-1, no
/// sidecar), so an absent or zero field keeps the historical behavior.
fn rerank_storage_from_proto(value: i32) -> Option<RerankStorageKind> {
    match v1::RerankStorageKind::try_from(value) {
        Ok(v1::RerankStorageKind::F32) => Some(RerankStorageKind::F32),
        Ok(v1::RerankStorageKind::Unspecified) | Err(_) => None,
    }
}

// ---- Analyzer spec conversion ----

/// Convert a laurus [`AnalyzerSpec`] into the proto wire form.
fn analyzer_spec_to_proto(spec: &AnalyzerSpec) -> v1::AnalyzerSpec {
    let proto_spec = match spec {
        AnalyzerSpec::Named(name) => v1::analyzer_spec::Spec::Named(name.clone()),
        AnalyzerSpec::Builtin(builtin) => {
            v1::analyzer_spec::Spec::Builtin(builtin_analyzer_spec_to_proto(builtin))
        }
    };
    v1::AnalyzerSpec {
        spec: Some(proto_spec),
    }
}

/// Convert a proto [`v1::AnalyzerSpec`] into a laurus [`AnalyzerSpec`].
///
/// Returns `None` when the proto message has no spec set, which means the
/// engine default analyzer should be used.
fn analyzer_spec_from_proto(proto: &v1::AnalyzerSpec) -> Option<AnalyzerSpec> {
    match proto.spec.as_ref()? {
        v1::analyzer_spec::Spec::Named(name) if name.is_empty() => None,
        v1::analyzer_spec::Spec::Named(name) => Some(AnalyzerSpec::Named(name.clone())),
        v1::analyzer_spec::Spec::Builtin(builtin) => {
            builtin_analyzer_spec_from_proto(builtin).map(AnalyzerSpec::Builtin)
        }
    }
}

fn builtin_analyzer_spec_to_proto(spec: &BuiltinAnalyzerSpec) -> v1::BuiltinAnalyzerSpec {
    let preset = match spec {
        BuiltinAnalyzerSpec::Japanese {
            mode,
            dict,
            user_dict,
        } => v1::builtin_analyzer_spec::Preset::Japanese(v1::JapaneseAnalyzerSpec {
            mode: mode.clone(),
            dict: dict.clone(),
            user_dict: user_dict.clone().unwrap_or_default(),
        }),
    };
    v1::BuiltinAnalyzerSpec {
        preset: Some(preset),
    }
}

fn builtin_analyzer_spec_from_proto(
    proto: &v1::BuiltinAnalyzerSpec,
) -> Option<BuiltinAnalyzerSpec> {
    match proto.preset.as_ref()? {
        v1::builtin_analyzer_spec::Preset::Japanese(jp) => Some(BuiltinAnalyzerSpec::Japanese {
            mode: if jp.mode.is_empty() {
                "normal".to_string()
            } else {
                jp.mode.clone()
            },
            dict: jp.dict.clone(),
            user_dict: if jp.user_dict.is_empty() {
                None
            } else {
                Some(jp.user_dict.clone())
            },
        }),
    }
}

// ---- Analyzer definition conversion ----

fn analyzer_definition_to_proto(def: &AnalyzerDefinition) -> v1::AnalyzerDefinition {
    v1::AnalyzerDefinition {
        char_filters: def.char_filters.iter().map(char_filter_to_proto).collect(),
        tokenizer: Some(tokenizer_to_proto(&def.tokenizer)),
        token_filters: def
            .token_filters
            .iter()
            .map(token_filter_to_proto)
            .collect(),
    }
}

fn analyzer_definition_from_proto(
    proto: &v1::AnalyzerDefinition,
) -> Result<AnalyzerDefinition, String> {
    let tokenizer = tokenizer_from_proto(
        proto
            .tokenizer
            .as_ref()
            .ok_or("AnalyzerDefinition missing tokenizer")?,
    )?;
    let char_filters = proto
        .char_filters
        .iter()
        .map(char_filter_from_proto)
        .collect::<Result<Vec<_>, _>>()?;
    let token_filters = proto
        .token_filters
        .iter()
        .map(token_filter_from_proto)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AnalyzerDefinition {
        char_filters,
        tokenizer,
        token_filters,
    })
}

fn tokenizer_to_proto(config: &TokenizerConfig) -> v1::ComponentConfig {
    let (type_name, params) = match config {
        TokenizerConfig::Whitespace => ("whitespace", HashMap::new()),
        TokenizerConfig::UnicodeWord => ("unicode_word", HashMap::new()),
        TokenizerConfig::Regex { pattern, gaps } => {
            let mut p = HashMap::new();
            p.insert("pattern".into(), pattern.clone());
            if *gaps {
                p.insert("gaps".into(), "true".into());
            }
            ("regex", p)
        }
        TokenizerConfig::Ngram { min_gram, max_gram } => {
            let mut p = HashMap::new();
            p.insert("min_gram".into(), min_gram.to_string());
            p.insert("max_gram".into(), max_gram.to_string());
            ("ngram", p)
        }
        TokenizerConfig::Lindera {
            mode,
            dict,
            user_dict,
        } => {
            let mut p = HashMap::new();
            p.insert("mode".into(), mode.clone());
            p.insert("dict".into(), dict.clone());
            if let Some(ud) = user_dict {
                p.insert("user_dict".into(), ud.clone());
            }
            ("lindera", p)
        }
        TokenizerConfig::Whole => ("whole", HashMap::new()),
    };
    v1::ComponentConfig {
        r#type: type_name.into(),
        params,
    }
}

fn tokenizer_from_proto(proto: &v1::ComponentConfig) -> Result<TokenizerConfig, String> {
    match proto.r#type.as_str() {
        "whitespace" => Ok(TokenizerConfig::Whitespace),
        "unicode_word" => Ok(TokenizerConfig::UnicodeWord),
        "regex" => Ok(TokenizerConfig::Regex {
            pattern: proto
                .params
                .get("pattern")
                .cloned()
                .unwrap_or_else(|| r"\w+".into()),
            gaps: proto.params.get("gaps").is_some_and(|v| v == "true"),
        }),
        "ngram" => {
            let min_gram = proto
                .params
                .get("min_gram")
                .ok_or("ngram: missing min_gram")?
                .parse::<usize>()
                .map_err(|e| format!("ngram: invalid min_gram: {e}"))?;
            let max_gram = proto
                .params
                .get("max_gram")
                .ok_or("ngram: missing max_gram")?
                .parse::<usize>()
                .map_err(|e| format!("ngram: invalid max_gram: {e}"))?;
            Ok(TokenizerConfig::Ngram { min_gram, max_gram })
        }
        "lindera" => Ok(TokenizerConfig::Lindera {
            mode: proto
                .params
                .get("mode")
                .cloned()
                .unwrap_or_else(|| "normal".into()),
            dict: proto
                .params
                .get("dict")
                .cloned()
                .ok_or("lindera: missing dict")?,
            user_dict: proto.params.get("user_dict").cloned(),
        }),
        "whole" => Ok(TokenizerConfig::Whole),
        other => Err(format!("Unknown tokenizer type: {other}")),
    }
}

fn char_filter_to_proto(config: &CharFilterConfig) -> v1::ComponentConfig {
    let (type_name, params) = match config {
        CharFilterConfig::UnicodeNormalization { form } => {
            let mut p = HashMap::new();
            p.insert("form".into(), form.clone());
            ("unicode_normalization", p)
        }
        CharFilterConfig::PatternReplace {
            pattern,
            replacement,
        } => {
            let mut p = HashMap::new();
            p.insert("pattern".into(), pattern.clone());
            p.insert("replacement".into(), replacement.clone());
            ("pattern_replace", p)
        }
        CharFilterConfig::Mapping { mapping } => {
            // Encode mapping as key=value pairs in params. `mapping` is a
            // `BTreeMap` (Issue #1060); `params` is the proto's `HashMap`.
            let p: HashMap<String, String> = mapping
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            ("mapping", p)
        }
        CharFilterConfig::JapaneseIterationMark { kanji, kana } => {
            let mut p = HashMap::new();
            p.insert("kanji".into(), kanji.to_string());
            p.insert("kana".into(), kana.to_string());
            ("japanese_iteration_mark", p)
        }
    };
    v1::ComponentConfig {
        r#type: type_name.into(),
        params,
    }
}

fn char_filter_from_proto(proto: &v1::ComponentConfig) -> Result<CharFilterConfig, String> {
    match proto.r#type.as_str() {
        "unicode_normalization" => Ok(CharFilterConfig::UnicodeNormalization {
            form: proto
                .params
                .get("form")
                .cloned()
                .unwrap_or_else(|| "nfkc".into()),
        }),
        "pattern_replace" => Ok(CharFilterConfig::PatternReplace {
            pattern: proto
                .params
                .get("pattern")
                .cloned()
                .ok_or("pattern_replace: missing pattern")?,
            replacement: proto.params.get("replacement").cloned().unwrap_or_default(),
        }),
        "mapping" => Ok(CharFilterConfig::Mapping {
            mapping: proto
                .params
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }),
        "japanese_iteration_mark" => Ok(CharFilterConfig::JapaneseIterationMark {
            kanji: proto.params.get("kanji").is_none_or(|v| v != "false"),
            kana: proto.params.get("kana").is_none_or(|v| v != "false"),
        }),
        other => Err(format!("Unknown char filter type: {other}")),
    }
}

fn token_filter_to_proto(config: &TokenFilterConfig) -> v1::ComponentConfig {
    let (type_name, params) = match config {
        TokenFilterConfig::Lowercase => ("lowercase", HashMap::new()),
        TokenFilterConfig::Stop { words } => {
            let mut p = HashMap::new();
            if let Some(word_list) = words {
                p.insert("words".into(), word_list.join(","));
            }
            ("stop", p)
        }
        TokenFilterConfig::Stem { stem_type } => {
            let mut p = HashMap::new();
            if let Some(st) = stem_type {
                p.insert("stem_type".into(), st.clone());
            }
            ("stem", p)
        }
        TokenFilterConfig::Boost { boost } => {
            let mut p = HashMap::new();
            p.insert("boost".into(), boost.to_string());
            ("boost", p)
        }
        TokenFilterConfig::Limit { limit } => {
            let mut p = HashMap::new();
            p.insert("limit".into(), limit.to_string());
            ("limit", p)
        }
        TokenFilterConfig::Strip => ("strip", HashMap::new()),
        TokenFilterConfig::RemoveEmpty => ("remove_empty", HashMap::new()),
        TokenFilterConfig::FlattenGraph => ("flatten_graph", HashMap::new()),
    };
    v1::ComponentConfig {
        r#type: type_name.into(),
        params,
    }
}

fn token_filter_from_proto(proto: &v1::ComponentConfig) -> Result<TokenFilterConfig, String> {
    match proto.r#type.as_str() {
        "lowercase" => Ok(TokenFilterConfig::Lowercase),
        "stop" => Ok(TokenFilterConfig::Stop {
            words: proto.params.get("words").map(|w| {
                w.split(',')
                    .map(|s| s.trim().to_string())
                    .collect::<Vec<_>>()
            }),
        }),
        "stem" => Ok(TokenFilterConfig::Stem {
            stem_type: proto.params.get("stem_type").cloned(),
        }),
        "boost" => {
            let boost = proto
                .params
                .get("boost")
                .ok_or("boost: missing boost")?
                .parse::<f32>()
                .map_err(|e| format!("boost: invalid value: {e}"))?;
            Ok(TokenFilterConfig::Boost { boost })
        }
        "limit" => {
            let limit = proto
                .params
                .get("limit")
                .ok_or("limit: missing limit")?
                .parse::<usize>()
                .map_err(|e| format!("limit: invalid value: {e}"))?;
            Ok(TokenFilterConfig::Limit { limit })
        }
        "strip" => Ok(TokenFilterConfig::Strip),
        "remove_empty" => Ok(TokenFilterConfig::RemoveEmpty),
        "flatten_graph" => Ok(TokenFilterConfig::FlattenGraph),
        other => Err(format!("Unknown token filter type: {other}")),
    }
}

// ---- Embedder definition conversion ----

fn embedder_definition_to_proto(def: &EmbedderDefinition) -> v1::EmbedderConfig {
    let (type_name, params) = match def {
        EmbedderDefinition::Precomputed => ("precomputed", HashMap::new()),
        EmbedderDefinition::CandleBert { model } => {
            let mut p = HashMap::new();
            p.insert("model".into(), model.clone());
            ("candle_bert", p)
        }
        EmbedderDefinition::CandleClip { model } => {
            let mut p = HashMap::new();
            p.insert("model".into(), model.clone());
            ("candle_clip", p)
        }
        EmbedderDefinition::Openai { model } => {
            let mut p = HashMap::new();
            p.insert("model".into(), model.clone());
            ("openai", p)
        }
    };
    v1::EmbedderConfig {
        r#type: type_name.into(),
        params,
    }
}

fn embedder_definition_from_proto(
    proto: &v1::EmbedderConfig,
) -> Result<EmbedderDefinition, String> {
    match proto.r#type.as_str() {
        "precomputed" => Ok(EmbedderDefinition::Precomputed),
        "candle_bert" => Ok(EmbedderDefinition::CandleBert {
            model: proto
                .params
                .get("model")
                .cloned()
                .ok_or("candle_bert: missing model")?,
        }),
        "candle_clip" => Ok(EmbedderDefinition::CandleClip {
            model: proto
                .params
                .get("model")
                .cloned()
                .ok_or("candle_clip: missing model")?,
        }),
        "openai" => Ok(EmbedderDefinition::Openai {
            model: proto
                .params
                .get("model")
                .cloned()
                .ok_or("openai: missing model")?,
        }),
        other => Err(format!("Unknown embedder type: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use laurus::{BooleanOption, FlatOption, IntegerOption, TextOption};

    /// Every `FieldChangeKind` variant maps to its own distinct,
    /// non-`Unspecified` proto value -- `Unspecified` is reserved for the
    /// proto3 zero-default (a client that never set the field), which a
    /// server response never sends.
    #[test]
    fn field_change_kind_to_proto_covers_every_variant_distinctly() {
        let mapped: Vec<v1::FieldChangeKind> = [
            FieldChangeKind::MetadataOnly,
            FieldChangeKind::Reindex,
            FieldChangeKind::Destructive,
        ]
        .into_iter()
        .map(field_change_kind_to_proto)
        .collect();

        assert!(
            !mapped.contains(&v1::FieldChangeKind::Unspecified),
            "no laurus FieldChangeKind should map to Unspecified: {mapped:?}"
        );
        let unique: std::collections::HashSet<_> = mapped.iter().collect();
        assert_eq!(
            unique.len(),
            mapped.len(),
            "every FieldChangeKind variant must map to a distinct proto value: {mapped:?}"
        );
    }

    /// Round-trip a `DynamicFieldPolicy` through proto and back for every
    /// concrete variant, verifying the enum mapping is complete.
    #[test]
    fn dynamic_field_policy_round_trip_all_variants() {
        for policy in [
            DynamicFieldPolicy::Strict,
            DynamicFieldPolicy::Dynamic,
            DynamicFieldPolicy::Ignore,
        ] {
            let proto_value = dynamic_field_policy_to_proto(&policy) as i32;
            let back = dynamic_field_policy_from_proto(proto_value);
            assert_eq!(policy, back, "round-trip mismatch for {policy:?}");
        }
    }

    /// `UNSPECIFIED` (the proto default) maps to the laurus default
    /// (`Dynamic`).
    #[test]
    fn dynamic_field_policy_unspecified_maps_to_default() {
        let unspecified = v1::DynamicFieldPolicy::Unspecified as i32;
        assert_eq!(
            dynamic_field_policy_from_proto(unspecified),
            DynamicFieldPolicy::default()
        );
        assert_eq!(DynamicFieldPolicy::default(), DynamicFieldPolicy::Dynamic);
    }

    /// Unknown i32 values coming from a newer client fall back to the
    /// laurus default (`Dynamic`), matching the forward-compatibility
    /// contract for proto3 enums.
    #[test]
    fn dynamic_field_policy_unknown_value_maps_to_default() {
        assert_eq!(
            dynamic_field_policy_from_proto(9999),
            DynamicFieldPolicy::default()
        );
    }

    /// Round-trip a non-trivial `Schema` through proto and back, checking
    /// that `dynamic_field_policy`, fields, and default_fields survive.
    #[test]
    fn schema_round_trip_preserves_policy_and_fields() {
        let schema = Schema::builder()
            .dynamic_field_policy(DynamicFieldPolicy::Strict)
            .add_field("title", FieldOption::Text(TextOption::default()))
            .add_field("year", FieldOption::Integer(IntegerOption::default()))
            .add_field("published", FieldOption::Boolean(BooleanOption::default()))
            .add_field(
                "embedding",
                FieldOption::Flat(FlatOption {
                    dimension: 128,
                    ..Default::default()
                }),
            )
            .add_default_field("title")
            .build();

        let proto = to_proto(&schema);
        let back = from_proto(&proto).expect("from_proto must succeed");

        assert_eq!(back.dynamic_field_policy, DynamicFieldPolicy::Strict);
        assert_eq!(back.default_fields, vec!["title".to_string()]);
        assert_eq!(back.fields.len(), 4);
        assert!(matches!(
            back.fields.get("title"),
            Some(FieldOption::Text(_))
        ));
        assert!(matches!(
            back.fields.get("year"),
            Some(FieldOption::Integer(_))
        ));
        assert!(matches!(
            back.fields.get("published"),
            Some(FieldOption::Boolean(_))
        ));
        assert!(matches!(
            back.fields.get("embedding"),
            Some(FieldOption::Flat(_))
        ));
    }

    /// A schema constructed with the default policy round-trips to the
    /// same default, even though the proto representation is encoded via
    /// the `Dynamic` enum value.
    #[test]
    fn schema_round_trip_default_policy() {
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();

        let proto = to_proto(&schema);
        let back = from_proto(&proto).unwrap();
        assert_eq!(back.dynamic_field_policy, DynamicFieldPolicy::Dynamic);
    }

    /// Issue #793: an HNSW field's `rerank_storage` and `quantizer`
    /// survive a proto round-trip. Before #793 the proto `HnswOption`
    /// had no `rerank_storage` field, so `from_proto` hard-coded `None`
    /// and the Stage-2 sidecar could never be enabled over gRPC. The
    /// quantizer assertion is the "audit quantizer" part of the issue —
    /// it was already wired and must keep round-tripping.
    #[test]
    fn hnsw_rerank_storage_and_quantizer_round_trip_through_proto() {
        let schema = Schema::builder()
            .add_field(
                "embedding",
                FieldOption::Hnsw(HnswOption {
                    dimension: 8,
                    rerank_storage: Some(RerankStorageKind::F32),
                    quantizer: QuantizationMethod::ProductQuantization { subvector_count: 4 },
                    ..Default::default()
                }),
            )
            .build();

        // Laurus -> proto carries the field (it is absent in the proto
        // representation before this fix).
        let proto = to_proto(&schema);
        let proto_field = proto
            .fields
            .get("embedding")
            .and_then(|f| f.option.as_ref())
            .expect("embedding field option must be present");
        match proto_field {
            v1::field_option::Option::Hnsw(h) => {
                assert_eq!(
                    h.rerank_storage,
                    Some(v1::RerankStorageKind::F32 as i32),
                    "to_proto must serialize rerank_storage"
                );
            }
            other => panic!("expected proto Hnsw option, got {other:?}"),
        }

        // proto -> Laurus restores both fields (previously rerank_storage
        // was hard-coded to None).
        let back = from_proto(&proto).expect("from_proto must succeed");
        match back.fields.get("embedding") {
            Some(FieldOption::Hnsw(h)) => {
                assert_eq!(h.rerank_storage, Some(RerankStorageKind::F32));
                assert_eq!(
                    h.quantizer,
                    QuantizationMethod::ProductQuantization { subvector_count: 4 }
                );
            }
            other => panic!("expected FieldOption::Hnsw, got {other:?}"),
        }
    }

    /// Issue #631: an HNSW field's `pq_codebook_path` survives a proto
    /// round-trip (so a shared PQ codebook can be configured over gRPC),
    /// and an unset value stays `None`.
    #[test]
    fn hnsw_pq_codebook_path_round_trips_through_proto() {
        let schema = Schema::builder()
            .add_field(
                "embedding",
                FieldOption::Hnsw(HnswOption {
                    dimension: 8,
                    quantizer: QuantizationMethod::ProductQuantization { subvector_count: 4 },
                    pq_codebook_path: Some("embedding.pqcb".to_string()),
                    ..Default::default()
                }),
            )
            .add_field(
                "plain",
                FieldOption::Hnsw(HnswOption {
                    dimension: 8,
                    ..Default::default()
                }),
            )
            .build();

        let proto = to_proto(&schema);
        match proto
            .fields
            .get("embedding")
            .and_then(|f| f.option.as_ref())
        {
            Some(v1::field_option::Option::Hnsw(h)) => {
                assert_eq!(
                    h.pq_codebook_path.as_deref(),
                    Some("embedding.pqcb"),
                    "to_proto must serialize pq_codebook_path"
                );
            }
            other => panic!("expected proto Hnsw option, got {other:?}"),
        }

        let back = from_proto(&proto).expect("from_proto must succeed");
        match back.fields.get("embedding") {
            Some(FieldOption::Hnsw(h)) => {
                assert_eq!(h.pq_codebook_path.as_deref(), Some("embedding.pqcb"));
            }
            other => panic!("expected FieldOption::Hnsw, got {other:?}"),
        }
        match back.fields.get("plain") {
            Some(FieldOption::Hnsw(h)) => {
                assert_eq!(h.pq_codebook_path, None, "unset must stay None");
            }
            other => panic!("expected FieldOption::Hnsw, got {other:?}"),
        }

        // An explicitly empty proto string names a codebook that can never
        // exist — from_proto must normalize it to "unset".
        let mut degenerate = proto.clone();
        if let Some(v1::field_option::Option::Hnsw(h)) = degenerate
            .fields
            .get_mut("embedding")
            .and_then(|f| f.option.as_mut())
        {
            h.pq_codebook_path = Some(String::new());
        } else {
            panic!("embedding must be an Hnsw proto option");
        }
        let back = from_proto(&degenerate).expect("from_proto must succeed");
        match back.fields.get("embedding") {
            Some(FieldOption::Hnsw(h)) => {
                assert_eq!(h.pq_codebook_path, None, "empty must normalize to None");
            }
            other => panic!("expected FieldOption::Hnsw, got {other:?}"),
        }
    }

    /// Issue #1084: `base_weight` round-trips as `Some` through proto for
    /// Hnsw/Flat/Ivf, and a gRPC client that omits it (proto `None`, the
    /// zero-value before this fix) restores to the engine's default
    /// (`1.0`) rather than the silent `0.0` a plain `float` produced.
    #[test]
    fn base_weight_round_trips_through_proto_and_unset_defaults_to_one() {
        let schema = Schema::builder()
            .add_field(
                "hnsw_explicit",
                FieldOption::Hnsw(HnswOption {
                    dimension: 8,
                    base_weight: 2.5,
                    ..Default::default()
                }),
            )
            .add_field(
                "flat_explicit",
                FieldOption::Flat(FlatOption {
                    dimension: 8,
                    base_weight: 3.5,
                    ..Default::default()
                }),
            )
            .add_field(
                "ivf_explicit",
                FieldOption::Ivf(laurus::IvfOption {
                    dimension: 8,
                    base_weight: 4.5,
                    ..Default::default()
                }),
            )
            .build();

        let proto = to_proto(&schema);
        for (name, expected) in [
            ("hnsw_explicit", 2.5f32),
            ("flat_explicit", 3.5f32),
            ("ivf_explicit", 4.5f32),
        ] {
            let base_weight = match proto.fields.get(name).and_then(|f| f.option.as_ref()) {
                Some(v1::field_option::Option::Hnsw(h)) => h.base_weight,
                Some(v1::field_option::Option::Flat(f)) => f.base_weight,
                Some(v1::field_option::Option::Ivf(i)) => i.base_weight,
                other => panic!("unexpected proto option for {name}: {other:?}"),
            };
            assert_eq!(
                base_weight,
                Some(expected),
                "to_proto must carry an explicit base_weight as Some for {name}"
            );
        }

        let back = from_proto(&proto).expect("from_proto must succeed");
        for (name, expected) in [
            ("hnsw_explicit", 2.5f32),
            ("flat_explicit", 3.5f32),
            ("ivf_explicit", 4.5f32),
        ] {
            let base_weight = match back.fields.get(name) {
                Some(FieldOption::Hnsw(h)) => h.base_weight,
                Some(FieldOption::Flat(f)) => f.base_weight,
                Some(FieldOption::Ivf(i)) => i.base_weight,
                other => panic!("unexpected laurus option for {name}: {other:?}"),
            };
            assert_eq!(base_weight, expected, "round-trip mismatch for {name}");
        }

        // A gRPC client omitting base_weight entirely (proto `None`) must
        // restore to the engine default, not the proto3 zero-value.
        let mut unset = proto.clone();
        if let Some(v1::field_option::Option::Hnsw(h)) = unset
            .fields
            .get_mut("hnsw_explicit")
            .and_then(|f| f.option.as_mut())
        {
            h.base_weight = None;
        } else {
            panic!("hnsw_explicit must be an Hnsw proto option");
        }
        let back = from_proto(&unset).expect("from_proto must succeed");
        match back.fields.get("hnsw_explicit") {
            Some(FieldOption::Hnsw(h)) => {
                assert_eq!(
                    h.base_weight, 1.0,
                    "unset base_weight must default to 1.0, not the proto3 zero-value"
                );
            }
            other => panic!("expected FieldOption::Hnsw, got {other:?}"),
        }
    }

    /// Issue #793: `rerank_storage` round-trips for Flat and IVF fields
    /// too (carried for schema fidelity even though those indexes do not
    /// emit a sidecar yet), and an unset value stays `None`.
    #[test]
    fn flat_and_ivf_rerank_storage_round_trip_through_proto() {
        let schema = Schema::builder()
            .add_field(
                "flat_vec",
                FieldOption::Flat(FlatOption {
                    dimension: 8,
                    rerank_storage: Some(RerankStorageKind::F32),
                    ..Default::default()
                }),
            )
            .add_field(
                "ivf_vec",
                FieldOption::Ivf(IvfOption {
                    dimension: 8,
                    rerank_storage: None,
                    ..Default::default()
                }),
            )
            .build();

        let back = from_proto(&to_proto(&schema)).expect("from_proto must succeed");
        match back.fields.get("flat_vec") {
            Some(FieldOption::Flat(f)) => {
                assert_eq!(f.rerank_storage, Some(RerankStorageKind::F32));
            }
            other => panic!("expected FieldOption::Flat, got {other:?}"),
        }
        match back.fields.get("ivf_vec") {
            Some(FieldOption::Ivf(i)) => {
                assert_eq!(i.rerank_storage, None);
            }
            other => panic!("expected FieldOption::Ivf, got {other:?}"),
        }
    }

    /// `FieldOption::Geo3d` round-trips through the proto `Geo3dOption`
    /// variant added in #305 (it used to be silently dropped to `None`
    /// before the proto representation existed).
    #[test]
    fn schema_field_option_geo3d_round_trip() {
        let schema = Schema::builder()
            .add_geo3d_field(
                "position",
                Geo3dOption {
                    indexed: true,
                    stored: false,
                    doc_values: true,
                },
            )
            .build();

        let proto = to_proto(&schema);
        let back = from_proto(&proto).expect("from_proto must succeed");

        match back.fields.get("position") {
            Some(FieldOption::Geo3d(o)) => {
                assert!(o.indexed);
                assert!(!o.stored);
            }
            other => panic!("expected FieldOption::Geo3d, got {other:?}"),
        }
    }
}
