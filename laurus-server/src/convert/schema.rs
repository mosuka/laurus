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
    IntegerOption, IvfOption, MultiVectorOption, MultiVectorStorage, QuantizationMethod, Schema,
    TextOption, TokenFilterConfig, TokenizerConfig,
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
///
/// # Errors
///
/// Returns an error if a field option cannot be converted (see
/// [`field_option_from_proto`]; the message names the field), if an
/// analyzer or embedder definition is invalid, or if `dynamic_field_policy`
/// is a value this server does not know (Issue #1473).
pub fn from_proto(proto: &v1::Schema) -> Result<Schema, String> {
    let mut fields = BTreeMap::new();
    for (name, fo) in &proto.fields {
        let option = field_option_from_proto(fo).map_err(|e| format!("Field '{name}': {e}"))?;
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
        dynamic_field_policy: dynamic_field_policy_from_proto(proto.dynamic_field_policy)?,
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
/// `UNSPECIFIED` (zero, what a client that leaves the field unset sends)
/// maps to [`DynamicFieldPolicy::Dynamic`], the laurus default.
///
/// # Arguments
///
/// * `value` - The proto enum value (i32).
///
/// # Errors
///
/// Returns an error, rather than falling back to the default, for a value
/// this server does not know, such as one a newer client sends
/// (Issue #1473).
fn dynamic_field_policy_from_proto(value: i32) -> Result<DynamicFieldPolicy, String> {
    match v1::DynamicFieldPolicy::try_from(value) {
        Ok(v1::DynamicFieldPolicy::Strict) => Ok(DynamicFieldPolicy::Strict),
        Ok(v1::DynamicFieldPolicy::Dynamic) => Ok(DynamicFieldPolicy::Dynamic),
        Ok(v1::DynamicFieldPolicy::Ignore) => Ok(DynamicFieldPolicy::Ignore),
        Ok(v1::DynamicFieldPolicy::Unspecified) => Ok(DynamicFieldPolicy::default()),
        Err(_) => Err(format!("unknown dynamic field policy {value}")),
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
            doc_values: Some(o.doc_values),
            multi_valued: o.multi_valued,
            position_increment_gap: Some(o.position_increment_gap),
        })),
        FieldOption::Integer(o) => Some(Opt::Integer(v1::IntegerOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::Float(o) => Some(Opt::Float(v1::FloatOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::Boolean(o) => Some(Opt::Boolean(v1::BooleanOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::DateTime(o) => Some(Opt::DateTime(v1::DateTimeOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::Geo(o) => Some(Opt::Geo(v1::GeoOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::Geo3d(o) => Some(Opt::Geo3d(v1::Geo3dOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::Bytes(o) => Some(Opt::Bytes(v1::BytesOption {
            stored: o.stored,
            multi_valued: o.multi_valued,
        })),
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
        FieldOption::MultiVector(o) => Some(Opt::MultiVector(v1::MultiVectorOption {
            dimension: o.dimension as u32,
            distance: distance_to_proto(&o.distance) as i32,
            embedder: o.embedder.clone().unwrap_or_default(),
            storage: Some(multi_vector_storage_to_proto(o.storage) as i32),
            pool_factor: Some(o.pool_factor),
        })),
    };
    v1::FieldOption { option }
}

/// Convert a proto `FieldOption` into a laurus `FieldOption`.
///
/// # Arguments
///
/// * `fo` - The proto field option to convert.
///
/// # Errors
///
/// Returns an error if the proto message has no option variant set, if a
/// vector option's quantizer is unknown or unsupported by this build (see
/// [`quantization_from_proto`]), or if a vector option's distance, rerank
/// storage or multi-vector storage is a value this server does not know
/// (Issue #1473).
pub fn field_option_from_proto(fo: &v1::FieldOption) -> Result<FieldOption, String> {
    use v1::field_option::Option as Opt;
    let option = match &fo.option {
        Some(Opt::Text(o)) => FieldOption::Text(TextOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            // Same tri-state treatment as `term_vectors` below (#1175): an
            // omitted gap is the engine default, not 0.
            position_increment_gap: o
                .position_increment_gap
                .unwrap_or(laurus::lexical::core::field::DEFAULT_POSITION_INCREMENT_GAP),
            // Unset means "use the engine's default", matching
            // `TextOption::default()` (#1083).
            term_vectors: o.term_vectors.unwrap_or(true),
            // Same tri-state treatment for the same reason (#1047).
            doc_values: o.doc_values.unwrap_or(true),
            analyzer: o.analyzer.as_ref().and_then(analyzer_spec_from_proto),
        }),
        Some(Opt::Integer(o)) => FieldOption::Integer(IntegerOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: o.doc_values.unwrap_or(true),
        }),
        Some(Opt::Float(o)) => FieldOption::Float(FloatOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: o.doc_values.unwrap_or(true),
        }),
        Some(Opt::Boolean(o)) => FieldOption::Boolean(BooleanOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: o.doc_values.unwrap_or(true),
        }),
        Some(Opt::DateTime(o)) => FieldOption::DateTime(DateTimeOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: o.doc_values.unwrap_or(true),
        }),
        Some(Opt::Geo(o)) => FieldOption::Geo(GeoOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            doc_values: o.doc_values.unwrap_or(true),
        }),
        Some(Opt::Geo3d(o)) => FieldOption::Geo3d(Geo3dOption {
            indexed: o.indexed,
            stored: o.stored,
            multi_valued: o.multi_valued,
            // Same tri-state treatment as every other lexical option; this
            // arm used to hardcode `true` and drop an explicit `false`.
            doc_values: o.doc_values.unwrap_or(true),
        }),
        Some(Opt::Bytes(o)) => FieldOption::Bytes(BytesOption {
            stored: o.stored,
            multi_valued: o.multi_valued,
        }),
        Some(Opt::Hnsw(o)) => FieldOption::Hnsw(HnswOption {
            dimension: o.dimension as usize,
            distance: distance_from_proto(o.distance)?,
            m: o.m as usize,
            ef_construction: o.ef_construction as usize,
            default_ef_search: o.default_ef_search.map(|v| v as usize),
            // Unset means "use the engine's default" (#1084).
            base_weight: o.base_weight.unwrap_or(1.0),
            quantizer: o
                .quantizer
                .as_ref()
                .map(quantization_from_proto)
                .transpose()?
                .unwrap_or_default(),
            rerank_storage: rerank_storage_from_proto(o.rerank_storage)?,
            embedder: if o.embedder.is_empty() {
                None
            } else {
                Some(o.embedder.clone())
            },
            // An explicitly empty path would configure a codebook that can
            // never exist; normalize to "unset" (same defensive convention
            // as `embedder` above).
            pq_codebook_path: o.pq_codebook_path.clone().filter(|p| !p.is_empty()),
        }),
        Some(Opt::Flat(o)) => FieldOption::Flat(FlatOption {
            dimension: o.dimension as usize,
            distance: distance_from_proto(o.distance)?,
            // Unset means "use the engine's default" (#1084).
            base_weight: o.base_weight.unwrap_or(1.0),
            quantizer: o
                .quantizer
                .as_ref()
                .map(quantization_from_proto)
                .transpose()?
                .unwrap_or_default(),
            rerank_storage: rerank_storage_from_proto(o.rerank_storage)?,
            embedder: if o.embedder.is_empty() {
                None
            } else {
                Some(o.embedder.clone())
            },
        }),
        Some(Opt::Ivf(o)) => FieldOption::Ivf(IvfOption {
            dimension: o.dimension as usize,
            distance: distance_from_proto(o.distance)?,
            n_clusters: o.n_clusters as usize,
            n_probe: o.n_probe as usize,
            // Unset means "use the engine's default" (#1084).
            base_weight: o.base_weight.unwrap_or(1.0),
            quantizer: o
                .quantizer
                .as_ref()
                .map(quantization_from_proto)
                .transpose()?
                .unwrap_or_default(),
            rerank_storage: rerank_storage_from_proto(o.rerank_storage)?,
            embedder: if o.embedder.is_empty() {
                None
            } else {
                Some(o.embedder.clone())
            },
        }),
        Some(Opt::MultiVector(o)) => FieldOption::MultiVector(MultiVectorOption {
            dimension: o.dimension as usize,
            distance: distance_from_proto(o.distance)?,
            embedder: if o.embedder.is_empty() {
                None
            } else {
                Some(o.embedder.clone())
            },
            storage: multi_vector_storage_from_proto(o.storage)?,
            // Unset (an older client) means no pooling.
            pool_factor: o.pool_factor.unwrap_or(1),
        }),
        None => return Err("field_option has no option set".to_string()),
    };
    Ok(option)
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

/// Convert a proto `DistanceMetric` value into a laurus [`DistanceMetric`].
///
/// `COSINE` is the zero value, so a client that leaves `distance` unset
/// gets cosine, the default.
///
/// # Errors
///
/// Returns an error, rather than falling back to cosine, for a value this
/// server does not know, such as one a newer client sends (Issue #1473).
fn distance_from_proto(d: i32) -> Result<DistanceMetric, String> {
    match v1::DistanceMetric::try_from(d) {
        Ok(v1::DistanceMetric::Cosine) => Ok(DistanceMetric::Cosine),
        Ok(v1::DistanceMetric::Euclidean) => Ok(DistanceMetric::Euclidean),
        Ok(v1::DistanceMetric::Manhattan) => Ok(DistanceMetric::Manhattan),
        Ok(v1::DistanceMetric::DotProduct) => Ok(DistanceMetric::DotProduct),
        Ok(v1::DistanceMetric::Angular) => Ok(DistanceMetric::Angular),
        Err(_) => Err(format!("unknown distance metric {d}")),
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
        QuantizationMethod::Float16 => v1::QuantizationConfig {
            method: v1::QuantizationMethod::Float16 as i32,
            subvector_count: 0,
        },
    }
}

/// Convert a proto `QuantizationConfig` into a laurus [`QuantizationMethod`].
///
/// `NONE` (the value a client that leaves `method` unset sends) maps to
/// [`QuantizationMethod::Scalar8Bit`], the default. Issue #481 Stage 1
/// removed unquantized storage, so `NONE` cannot mean anything else.
///
/// # Errors
///
/// Returns an error, rather than falling back to `Scalar8Bit`, for a value
/// this server does not know (for example a method added by a newer
/// client) and for `PRODUCT_QUANTIZATION_FASTSCAN` on a server built
/// without the `pq-fastscan` feature (Issue #1459).
fn quantization_from_proto(q: &v1::QuantizationConfig) -> Result<QuantizationMethod, String> {
    match v1::QuantizationMethod::try_from(q.method) {
        Ok(v1::QuantizationMethod::Scalar8bit) | Ok(v1::QuantizationMethod::None) => {
            Ok(QuantizationMethod::Scalar8Bit)
        }
        Ok(v1::QuantizationMethod::ProductQuantization) => {
            Ok(QuantizationMethod::ProductQuantization {
                subvector_count: q.subvector_count as usize,
            })
        }
        #[cfg(feature = "pq-fastscan")]
        Ok(v1::QuantizationMethod::ProductQuantizationFastscan) => {
            Ok(QuantizationMethod::ProductQuantizationFastScan {
                subvector_count: q.subvector_count as usize,
            })
        }
        #[cfg(not(feature = "pq-fastscan"))]
        Ok(v1::QuantizationMethod::ProductQuantizationFastscan) => Err(
            "quantization method QUANTIZATION_METHOD_PRODUCT_QUANTIZATION_FASTSCAN \
             requires a server built with the `pq-fastscan` feature"
                .to_string(),
        ),
        Ok(v1::QuantizationMethod::Float16) => Ok(QuantizationMethod::Float16),
        Err(_) => Err(format!("unknown quantization method {}", q.method)),
    }
}

/// Convert a laurus [`RerankStorageKind`] into the proto enum (Issue #793).
fn rerank_storage_to_proto(kind: RerankStorageKind) -> v1::RerankStorageKind {
    match kind {
        RerankStorageKind::F32 => v1::RerankStorageKind::F32,
    }
}

/// Convert an optional proto `rerank_storage` enum value into an optional
/// laurus [`RerankStorageKind`] (Issue #793).
///
/// An absent field and `UNSPECIFIED` map to `None` (Stage-1, no sidecar),
/// so a client that leaves the field unset keeps the historical behavior.
///
/// # Errors
///
/// Returns an error, rather than `None`, for a value this server does not
/// know, such as one a newer client sends (Issue #1473).
fn rerank_storage_from_proto(value: Option<i32>) -> Result<Option<RerankStorageKind>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    match v1::RerankStorageKind::try_from(value) {
        Ok(v1::RerankStorageKind::F32) => Ok(Some(RerankStorageKind::F32)),
        Ok(v1::RerankStorageKind::Unspecified) => Ok(None),
        Err(_) => Err(format!("unknown rerank storage kind {value}")),
    }
}

/// Convert a laurus [`MultiVectorStorage`] into the proto wire form
/// (Issue #1346).
fn multi_vector_storage_to_proto(storage: MultiVectorStorage) -> v1::MultiVectorStorage {
    match storage {
        MultiVectorStorage::F32 => v1::MultiVectorStorage::F32,
        MultiVectorStorage::F16 => v1::MultiVectorStorage::F16,
        MultiVectorStorage::Int8 => v1::MultiVectorStorage::Int8,
        MultiVectorStorage::Binary => v1::MultiVectorStorage::Binary,
    }
}

/// Convert an optional proto `storage` enum value into a laurus
/// [`MultiVectorStorage`]. An absent field and `UNSPECIFIED` map to
/// [`MultiVectorStorage::F32`], the default.
///
/// # Errors
///
/// Returns an error, rather than falling back to `F32`, for a value this
/// server does not know, such as one a newer client sends (Issue #1473).
fn multi_vector_storage_from_proto(value: Option<i32>) -> Result<MultiVectorStorage, String> {
    let Some(value) = value else {
        return Ok(MultiVectorStorage::F32);
    };
    match v1::MultiVectorStorage::try_from(value) {
        Ok(v1::MultiVectorStorage::F16) => Ok(MultiVectorStorage::F16),
        Ok(v1::MultiVectorStorage::Int8) => Ok(MultiVectorStorage::Int8),
        Ok(v1::MultiVectorStorage::Binary) => Ok(MultiVectorStorage::Binary),
        Ok(v1::MultiVectorStorage::F32 | v1::MultiVectorStorage::Unspecified) => {
            Ok(MultiVectorStorage::F32)
        }
        Err(_) => Err(format!("unknown multi-vector storage {value}")),
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
        EmbedderDefinition::CandleColbert {
            model,
            revision,
            query_maxlen,
            doc_maxlen,
        } => {
            let mut p = HashMap::new();
            p.insert("model".into(), model.clone());
            if let Some(revision) = revision {
                p.insert("revision".into(), revision.clone());
            }
            if let Some(n) = query_maxlen {
                p.insert("query_maxlen".into(), n.to_string());
            }
            if let Some(n) = doc_maxlen {
                p.insert("doc_maxlen".into(), n.to_string());
            }
            ("candle_colbert", p)
        }
    };
    v1::EmbedderConfig {
        r#type: type_name.into(),
        params,
    }
}

/// Parse the optional count parameter `key` of a `kind` embedder.
fn optional_count(
    proto: &v1::EmbedderConfig,
    kind: &str,
    key: &str,
) -> Result<Option<usize>, String> {
    proto
        .params
        .get(key)
        .map(|value| {
            value
                .parse()
                .map_err(|_| format!("{kind}: {key} must be a non-negative integer, got '{value}'"))
        })
        .transpose()
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
        "candle_colbert" => Ok(EmbedderDefinition::CandleColbert {
            model: proto
                .params
                .get("model")
                .cloned()
                .ok_or("candle_colbert: missing model")?,
            revision: proto.params.get("revision").cloned(),
            query_maxlen: optional_count(proto, "candle_colbert", "query_maxlen")?,
            doc_maxlen: optional_count(proto, "candle_colbert", "doc_maxlen")?,
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
            assert_eq!(Ok(policy), back, "round-trip mismatch for {policy:?}");
        }
    }

    /// `UNSPECIFIED` (the proto default) maps to the laurus default
    /// (`Dynamic`).
    #[test]
    fn dynamic_field_policy_unspecified_maps_to_default() {
        let unspecified = v1::DynamicFieldPolicy::Unspecified as i32;
        assert_eq!(
            dynamic_field_policy_from_proto(unspecified),
            Ok(DynamicFieldPolicy::default())
        );
        assert_eq!(DynamicFieldPolicy::default(), DynamicFieldPolicy::Dynamic);
    }

    /// Issue #1473: a policy value this server does not know, such as one a
    /// newer client sends, is an error instead of the default `Dynamic`,
    /// and `from_proto` (CreateIndex) rejects the schema.
    #[test]
    fn dynamic_field_policy_unknown_value_is_rejected() {
        assert_eq!(
            dynamic_field_policy_from_proto(9999),
            Err("unknown dynamic field policy 9999".to_string())
        );
        let schema = v1::Schema {
            dynamic_field_policy: 9999,
            ..Default::default()
        };
        assert_eq!(
            from_proto(&schema).unwrap_err(),
            "unknown dynamic field policy 9999"
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

    /// Issue #623: `QuantizationMethod::Float16` maps to its own proto
    /// value (`QUANTIZATION_METHOD_FLOAT16`, not `SCALAR_8BIT`) and
    /// survives the round-trip, both through the
    /// quantizer conversion itself and through a schema that uses it on
    /// every index kind that supports it (HNSW, Flat, IVF).
    #[test]
    fn float16_quantizer_round_trips_through_proto() {
        let proto = quantization_to_proto(&QuantizationMethod::Float16);
        assert_eq!(proto.method, v1::QuantizationMethod::Float16 as i32);
        assert_eq!(proto.subvector_count, 0);
        assert_eq!(
            quantization_from_proto(&proto),
            Ok(QuantizationMethod::Float16)
        );

        let schema = Schema::builder()
            .add_field(
                "hnsw_vec",
                FieldOption::Hnsw(HnswOption {
                    dimension: 8,
                    quantizer: QuantizationMethod::Float16,
                    ..Default::default()
                }),
            )
            .add_field(
                "flat_vec",
                FieldOption::Flat(FlatOption {
                    dimension: 8,
                    quantizer: QuantizationMethod::Float16,
                    ..Default::default()
                }),
            )
            .add_field(
                "ivf_vec",
                FieldOption::Ivf(IvfOption {
                    dimension: 8,
                    quantizer: QuantizationMethod::Float16,
                    ..Default::default()
                }),
            )
            .build();

        let back = from_proto(&to_proto(&schema)).expect("from_proto must succeed");
        match back.fields.get("hnsw_vec") {
            Some(FieldOption::Hnsw(h)) => assert_eq!(h.quantizer, QuantizationMethod::Float16),
            other => panic!("expected FieldOption::Hnsw, got {other:?}"),
        }
        match back.fields.get("flat_vec") {
            Some(FieldOption::Flat(f)) => assert_eq!(f.quantizer, QuantizationMethod::Float16),
            other => panic!("expected FieldOption::Flat, got {other:?}"),
        }
        match back.fields.get("ivf_vec") {
            Some(FieldOption::Ivf(i)) => assert_eq!(i.quantizer, QuantizationMethod::Float16),
            other => panic!("expected FieldOption::Ivf, got {other:?}"),
        }
    }

    /// The proto options of an HNSW, a Flat and an IVF field whose quantizer
    /// carries the raw `method` value, which may be one this build does not
    /// know (Issue #1459).
    fn vector_options_with_quantizer_method(method: i32) -> [v1::FieldOption; 3] {
        let mut options = [
            field_option_to_proto(&FieldOption::Hnsw(HnswOption {
                dimension: 8,
                ..Default::default()
            })),
            field_option_to_proto(&FieldOption::Flat(FlatOption {
                dimension: 8,
                ..Default::default()
            })),
            field_option_to_proto(&FieldOption::Ivf(IvfOption {
                dimension: 8,
                ..Default::default()
            })),
        ];
        for option in &mut options {
            let quantizer = Some(v1::QuantizationConfig {
                method,
                subvector_count: 4,
            });
            match &mut option.option {
                Some(v1::field_option::Option::Hnsw(o)) => o.quantizer = quantizer,
                Some(v1::field_option::Option::Flat(o)) => o.quantizer = quantizer,
                Some(v1::field_option::Option::Ivf(o)) => o.quantizer = quantizer,
                other => unreachable!("not a vector option: {other:?}"),
            }
        }
        options
    }

    /// Issue #1459: `NONE` (what a client that leaves `method` unset sends)
    /// still means the default `Scalar8Bit`, but a value this server does
    /// not know, such as one a newer client sends, is an error instead of
    /// `Scalar8Bit` on every vector option that carries a quantizer.
    /// `from_proto` (CreateIndex) names the field.
    #[test]
    fn unknown_quantization_method_is_rejected() {
        for option in vector_options_with_quantizer_method(v1::QuantizationMethod::None as i32) {
            let quantizer = match field_option_from_proto(&option) {
                Ok(FieldOption::Hnsw(o)) => o.quantizer,
                Ok(FieldOption::Flat(o)) => o.quantizer,
                Ok(FieldOption::Ivf(o)) => o.quantizer,
                other => panic!("expected a vector option, got {other:?}"),
            };
            assert_eq!(quantizer, QuantizationMethod::Scalar8Bit);
        }

        for option in vector_options_with_quantizer_method(99) {
            assert_eq!(
                field_option_from_proto(&option).unwrap_err(),
                "unknown quantization method 99",
                "{option:?}"
            );
        }

        let [hnsw, _, _] = vector_options_with_quantizer_method(99);
        let schema = v1::Schema {
            fields: HashMap::from([("embedding".to_string(), hnsw)]),
            ..Default::default()
        };
        assert_eq!(
            from_proto(&schema).unwrap_err(),
            "Field 'embedding': unknown quantization method 99"
        );
    }

    /// Issue #1459: a server built without `pq-fastscan` rejects a FastScan
    /// quantizer, as the proto comment says, instead of storing the field
    /// as `Scalar8Bit`.
    #[cfg(not(feature = "pq-fastscan"))]
    #[test]
    fn fastscan_quantizer_is_rejected_without_the_feature() {
        let method = v1::QuantizationMethod::ProductQuantizationFastscan as i32;
        for option in vector_options_with_quantizer_method(method) {
            let err = field_option_from_proto(&option).unwrap_err();
            assert!(err.contains("`pq-fastscan` feature"), "{err}");
        }
    }

    /// The other half of the pair above: with `pq-fastscan`, the same
    /// request maps to a FastScan quantizer.
    #[cfg(feature = "pq-fastscan")]
    #[test]
    fn fastscan_quantizer_is_accepted_with_the_feature() {
        let method = v1::QuantizationMethod::ProductQuantizationFastscan as i32;
        let [hnsw, _, _] = vector_options_with_quantizer_method(method);
        match field_option_from_proto(&hnsw) {
            Ok(FieldOption::Hnsw(o)) => assert_eq!(
                o.quantizer,
                QuantizationMethod::ProductQuantizationFastScan { subvector_count: 4 }
            ),
            other => panic!("expected FieldOption::Hnsw, got {other:?}"),
        }
    }

    /// A field option with no variant set is an error, and `from_proto`
    /// (CreateIndex) names the field.
    #[test]
    fn field_option_without_a_variant_is_rejected() {
        let empty = v1::FieldOption { option: None };
        assert_eq!(
            field_option_from_proto(&empty).unwrap_err(),
            "field_option has no option set"
        );
        let schema = v1::Schema {
            fields: HashMap::from([("body".to_string(), empty)]),
            ..Default::default()
        };
        assert_eq!(
            from_proto(&schema).unwrap_err(),
            "Field 'body': field_option has no option set"
        );
    }

    /// The proto options of an HNSW, a Flat, an IVF and a multi-vector field
    /// whose `distance` is the raw value `distance`, which may be one this
    /// build does not know (Issue #1473).
    fn vector_options_with_distance(distance: i32) -> [v1::FieldOption; 4] {
        use v1::field_option::Option as Opt;
        [
            Opt::Hnsw(v1::HnswOption {
                dimension: 8,
                distance,
                ..Default::default()
            }),
            Opt::Flat(v1::FlatOption {
                dimension: 8,
                distance,
                ..Default::default()
            }),
            Opt::Ivf(v1::IvfOption {
                dimension: 8,
                distance,
                ..Default::default()
            }),
            Opt::MultiVector(v1::MultiVectorOption {
                dimension: 8,
                distance,
                ..Default::default()
            }),
        ]
        .map(|option| v1::FieldOption {
            option: Some(option),
        })
    }

    /// The distance of a converted vector field option.
    fn distance_of(option: &FieldOption) -> DistanceMetric {
        match option {
            FieldOption::Hnsw(o) => o.distance,
            FieldOption::Flat(o) => o.distance,
            FieldOption::Ivf(o) => o.distance,
            FieldOption::MultiVector(o) => o.distance,
            other => panic!("not a vector option: {other:?}"),
        }
    }

    /// Issue #1473: `COSINE` (the zero value, what a client that leaves
    /// `distance` unset sends) is cosine, but a distance this server does
    /// not know, such as one a newer client sends, is an error instead of
    /// cosine on every vector option. `from_proto` (CreateIndex) names the
    /// field.
    #[test]
    fn unknown_distance_metric_is_rejected() {
        for option in vector_options_with_distance(v1::DistanceMetric::Cosine as i32) {
            let converted = field_option_from_proto(&option).unwrap();
            assert_eq!(distance_of(&converted), DistanceMetric::Cosine);
        }

        for option in vector_options_with_distance(99) {
            assert_eq!(
                field_option_from_proto(&option).unwrap_err(),
                "unknown distance metric 99",
                "{option:?}"
            );
        }

        let [hnsw, _, _, _] = vector_options_with_distance(99);
        let schema = v1::Schema {
            fields: HashMap::from([("embedding".to_string(), hnsw)]),
            ..Default::default()
        };
        assert_eq!(
            from_proto(&schema).unwrap_err(),
            "Field 'embedding': unknown distance metric 99"
        );
    }

    /// The proto options of an HNSW, a Flat and an IVF field whose
    /// `rerank_storage` is the raw value `rerank_storage`, which may be one
    /// this build does not know (Issue #1473).
    fn vector_options_with_rerank_storage(rerank_storage: Option<i32>) -> [v1::FieldOption; 3] {
        use v1::field_option::Option as Opt;
        [
            Opt::Hnsw(v1::HnswOption {
                dimension: 8,
                rerank_storage,
                ..Default::default()
            }),
            Opt::Flat(v1::FlatOption {
                dimension: 8,
                rerank_storage,
                ..Default::default()
            }),
            Opt::Ivf(v1::IvfOption {
                dimension: 8,
                rerank_storage,
                ..Default::default()
            }),
        ]
        .map(|option| v1::FieldOption {
            option: Some(option),
        })
    }

    /// Issue #1473: an absent `rerank_storage` and `UNSPECIFIED` mean no
    /// sidecar, but a value this server does not know, such as one a newer
    /// client sends, is an error instead of no sidecar on every vector
    /// option that carries it. `from_proto` (CreateIndex) names the field.
    #[test]
    fn unknown_rerank_storage_kind_is_rejected() {
        for value in [None, Some(v1::RerankStorageKind::Unspecified as i32)] {
            for option in vector_options_with_rerank_storage(value) {
                let rerank_storage = match field_option_from_proto(&option) {
                    Ok(FieldOption::Hnsw(o)) => o.rerank_storage,
                    Ok(FieldOption::Flat(o)) => o.rerank_storage,
                    Ok(FieldOption::Ivf(o)) => o.rerank_storage,
                    other => panic!("expected a vector option, got {other:?}"),
                };
                assert_eq!(rerank_storage, None, "{value:?}");
            }
        }

        for option in vector_options_with_rerank_storage(Some(99)) {
            assert_eq!(
                field_option_from_proto(&option).unwrap_err(),
                "unknown rerank storage kind 99",
                "{option:?}"
            );
        }

        let [hnsw, _, _] = vector_options_with_rerank_storage(Some(99));
        let schema = v1::Schema {
            fields: HashMap::from([("embedding".to_string(), hnsw)]),
            ..Default::default()
        };
        assert_eq!(
            from_proto(&schema).unwrap_err(),
            "Field 'embedding': unknown rerank storage kind 99"
        );
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
                    multi_valued: false,
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
                assert!(!o.multi_valued);
            }
            other => panic!("expected FieldOption::Geo3d, got {other:?}"),
        }
    }

    /// #1180: `BooleanOption.multi_valued` (proto field 4) round-trips, and
    /// `doc_values = false` survives the trip.
    #[test]
    fn schema_field_option_boolean_multi_valued_round_trip() {
        let schema = Schema::builder()
            .add_field(
                "flags",
                FieldOption::Boolean(BooleanOption {
                    indexed: true,
                    stored: true,
                    multi_valued: true,
                    doc_values: false,
                }),
            )
            .build();
        let back = from_proto(&to_proto(&schema)).expect("from_proto must succeed");
        match back.fields.get("flags") {
            Some(FieldOption::Boolean(o)) => {
                assert!(o.multi_valued);
                assert!(!o.doc_values);
            }
            other => panic!("expected FieldOption::Boolean, got {other:?}"),
        }
    }

    /// #1176: `BytesOption.multi_valued` (proto field 2) round-trips. Unlike
    /// the other multi-valued options, `BytesOption` has no `indexed` /
    /// `doc_values` flags to check alongside it.
    #[test]
    fn schema_field_option_bytes_multi_valued_round_trip() {
        let schema = Schema::builder()
            .add_field(
                "attachments",
                FieldOption::Bytes(BytesOption {
                    stored: true,
                    multi_valued: true,
                }),
            )
            .build();
        let back = from_proto(&to_proto(&schema)).expect("from_proto must succeed");
        match back.fields.get("attachments") {
            Some(FieldOption::Bytes(o)) => {
                assert!(o.stored);
                assert!(o.multi_valued);
            }
            other => panic!("expected FieldOption::Bytes, got {other:?}"),
        }
    }

    /// #1177: `MultiVectorOption` (proto field 12) round-trips its
    /// dimension and distance.
    #[test]
    fn schema_field_option_multi_vector_round_trip() {
        // #1346: every storage kind round-trips through proto (Binary
        // requires Cosine).
        for (storage, distance) in [
            (MultiVectorStorage::F32, DistanceMetric::DotProduct),
            (MultiVectorStorage::F16, DistanceMetric::DotProduct),
            (MultiVectorStorage::Int8, DistanceMetric::DotProduct),
            (MultiVectorStorage::Binary, DistanceMetric::Cosine),
        ] {
            let schema = Schema::builder()
                .add_multi_vector_field(
                    "tokens",
                    MultiVectorOption::new(96)
                        .distance(distance)
                        .storage(storage),
                )
                .build();
            let back = from_proto(&to_proto(&schema)).expect("from_proto must succeed");
            match back.fields.get("tokens") {
                Some(FieldOption::MultiVector(o)) => {
                    assert_eq!(o.dimension, 96);
                    assert_eq!(o.distance, distance);
                    assert_eq!(o.embedder, None);
                    assert_eq!(o.storage, storage, "{storage:?}");
                }
                other => panic!("expected FieldOption::MultiVector, got {other:?}"),
            }
        }
    }

    /// #1346: `pool_factor` round-trips through proto, and an absent value
    /// (a client that predates the field) reads as 1, no pooling.
    #[test]
    fn schema_field_option_multi_vector_pool_factor_round_trips_and_defaults_to_1() {
        let schema = Schema::builder()
            .add_multi_vector_field("tokens", MultiVectorOption::new(96).pool_factor(3))
            .build();
        let proto = to_proto(&schema);
        let back = from_proto(&proto).expect("from_proto must succeed");
        assert!(matches!(
            back.fields.get("tokens"),
            Some(FieldOption::MultiVector(o)) if o.pool_factor == 3
        ));

        let mut field =
            field_option_to_proto(&FieldOption::MultiVector(MultiVectorOption::new(96)));
        if let Some(v1::field_option::Option::MultiVector(o)) = &mut field.option {
            o.pool_factor = None;
        }
        assert!(matches!(
            field_option_from_proto(&field),
            Ok(FieldOption::MultiVector(o)) if o.pool_factor == 1
        ));
    }

    /// #1346: an absent or `UNSPECIFIED` proto `storage` value (as every
    /// segment written before this field existed would decode to) maps to
    /// [`MultiVectorStorage::F32`], not a parse error.
    #[test]
    fn schema_field_option_multi_vector_storage_absent_or_unspecified_defaults_to_f32() {
        assert_eq!(
            multi_vector_storage_from_proto(None),
            Ok(MultiVectorStorage::F32)
        );
        assert_eq!(
            multi_vector_storage_from_proto(Some(v1::MultiVectorStorage::Unspecified as i32)),
            Ok(MultiVectorStorage::F32)
        );
    }

    /// Issue #1473: a multi-vector storage value this server does not know,
    /// such as one a newer client sends, is an error instead of `F32`, and
    /// `from_proto` (CreateIndex) names the field.
    #[test]
    fn unknown_multi_vector_storage_is_rejected() {
        let mut option =
            field_option_to_proto(&FieldOption::MultiVector(MultiVectorOption::new(96)));
        match &mut option.option {
            Some(v1::field_option::Option::MultiVector(o)) => o.storage = Some(99),
            other => unreachable!("not a multi-vector option: {other:?}"),
        }
        assert_eq!(
            field_option_from_proto(&option).unwrap_err(),
            "unknown multi-vector storage 99"
        );
        let schema = v1::Schema {
            fields: HashMap::from([("tokens".to_string(), option)]),
            ..Default::default()
        };
        assert_eq!(
            from_proto(&schema).unwrap_err(),
            "Field 'tokens': unknown multi-vector storage 99"
        );
    }

    /// #1349: a `candle_colbert` embedder definition and the multi-vector
    /// field naming it round-trip, with and without the optional params.
    #[test]
    fn candle_colbert_embedder_round_trips_through_proto() {
        let full = EmbedderDefinition::CandleColbert {
            model: "colbert-ir/colbertv2.0".to_string(),
            revision: Some("c1e84128".to_string()),
            query_maxlen: Some(32),
            doc_maxlen: Some(180),
        };
        let bare = EmbedderDefinition::CandleColbert {
            model: "answerdotai/answerai-colbert-small-v1".to_string(),
            revision: None,
            query_maxlen: None,
            doc_maxlen: None,
        };
        let schema = Schema::builder()
            .add_embedder("colbert", full)
            .add_embedder("small", bare)
            .add_multi_vector_field("tokens", MultiVectorOption::new(128).embedder("colbert"))
            .build();
        let proto = to_proto(&schema);
        assert_eq!(proto.embedders["colbert"].params["query_maxlen"], "32");
        assert!(!proto.embedders["small"].params.contains_key("revision"));

        let back = from_proto(&proto).expect("from_proto must succeed");
        assert!(matches!(
            &back.embedders["colbert"],
            EmbedderDefinition::CandleColbert {
                revision: Some(r),
                query_maxlen: Some(32),
                doc_maxlen: Some(180),
                ..
            } if r == "c1e84128"
        ));
        assert!(matches!(
            &back.embedders["small"],
            EmbedderDefinition::CandleColbert {
                revision: None,
                query_maxlen: None,
                doc_maxlen: None,
                ..
            }
        ));
        assert_eq!(back.fields["tokens"].embedder_name(), Some("colbert"));
    }

    /// #1349: a non-numeric `candle_colbert` length is rejected.
    #[test]
    fn candle_colbert_rejects_a_non_numeric_length() {
        let proto = v1::EmbedderConfig {
            r#type: "candle_colbert".to_string(),
            params: HashMap::from([
                ("model".to_string(), "colbert-ir/colbertv2.0".to_string()),
                ("doc_maxlen".to_string(), "long".to_string()),
            ]),
        };
        let err = embedder_definition_from_proto(&proto).unwrap_err();
        assert!(err.contains("doc_maxlen"), "{err}");
    }

    /// #1175: `TextOption.multi_valued` (proto field 6) and
    /// `position_increment_gap` (proto field 7) round-trip, and an omitted
    /// gap on the wire means the engine default rather than 0.
    #[test]
    fn schema_field_option_text_multi_valued_round_trip() {
        let schema = Schema::builder()
            .add_field(
                "notes",
                FieldOption::Text(TextOption {
                    multi_valued: true,
                    position_increment_gap: 5,
                    doc_values: false,
                    ..Default::default()
                }),
            )
            .build();
        let back = from_proto(&to_proto(&schema)).expect("from_proto must succeed");
        match back.fields.get("notes") {
            Some(FieldOption::Text(o)) => {
                assert!(o.multi_valued);
                assert_eq!(o.position_increment_gap, 5);
                assert!(!o.doc_values);
            }
            other => panic!("expected FieldOption::Text, got {other:?}"),
        }

        // A client that predates #1175 omits the gap entirely.
        let proto = v1::FieldOption {
            option: Some(v1::field_option::Option::Text(v1::TextOption {
                indexed: true,
                stored: true,
                term_vectors: None,
                analyzer: None,
                doc_values: None,
                multi_valued: true,
                position_increment_gap: None,
            })),
        };
        match field_option_from_proto(&proto) {
            Ok(FieldOption::Text(o)) => assert_eq!(
                o.position_increment_gap,
                laurus::lexical::core::field::DEFAULT_POSITION_INCREMENT_GAP
            ),
            other => panic!("expected FieldOption::Text, got {other:?}"),
        }
    }

    /// #1184: `DateTimeOption.multi_valued` (proto field 4) round-trips.
    #[test]
    fn schema_field_option_datetime_multi_valued_round_trip() {
        let schema = Schema::builder()
            .add_field(
                "times",
                FieldOption::DateTime(DateTimeOption {
                    indexed: true,
                    stored: true,
                    multi_valued: true,
                    doc_values: false,
                }),
            )
            .build();
        let back = from_proto(&to_proto(&schema)).expect("from_proto must succeed");
        match back.fields.get("times") {
            Some(FieldOption::DateTime(o)) => {
                assert!(o.multi_valued);
                assert!(!o.doc_values);
            }
            other => panic!("expected FieldOption::DateTime, got {other:?}"),
        }
    }

    /// #1174: `multi_valued` round-trips for Geo and Geo3d (proto field 4),
    /// and `Geo3dOption.doc_values = false` survives the trip — the reverse
    /// conversion used to hardcode `true`.
    #[test]
    fn schema_field_option_geo_multi_valued_round_trip() {
        let schema = Schema::builder()
            .add_field(
                "locations",
                FieldOption::Geo(GeoOption {
                    indexed: true,
                    stored: true,
                    multi_valued: true,
                    doc_values: false,
                }),
            )
            .add_field(
                "positions",
                FieldOption::Geo3d(Geo3dOption {
                    indexed: true,
                    stored: true,
                    multi_valued: true,
                    doc_values: false,
                }),
            )
            .build();

        let back = from_proto(&to_proto(&schema)).expect("from_proto must succeed");
        match back.fields.get("locations") {
            Some(FieldOption::Geo(o)) => {
                assert!(o.multi_valued);
                assert!(!o.doc_values);
            }
            other => panic!("expected FieldOption::Geo, got {other:?}"),
        }
        match back.fields.get("positions") {
            Some(FieldOption::Geo3d(o)) => {
                assert!(o.multi_valued);
                assert!(
                    !o.doc_values,
                    "explicit doc_values=false must not be coerced to true"
                );
            }
            other => panic!("expected FieldOption::Geo3d, got {other:?}"),
        }
    }

    /// #1047: `doc_values` follows the same tri-state contract as
    /// `term_vectors` (#1083) -- `to_proto` carries an explicit setting as
    /// `Some`, and `from_proto` restores an unset value to the engine's
    /// default (`true`), not the proto3 zero-value (`false`). Covers
    /// `Text` (which also carries `term_vectors`, to prove the two flags
    /// don't interfere) and `Integer` (a non-`Text` arm).
    #[test]
    fn doc_values_round_trips_through_proto_and_unset_defaults_to_true() {
        let schema = Schema::builder()
            .add_field(
                "text_explicit_false",
                FieldOption::Text(TextOption {
                    doc_values: false,
                    ..Default::default()
                }),
            )
            .add_field(
                "integer_explicit_false",
                FieldOption::Integer(IntegerOption {
                    doc_values: false,
                    ..Default::default()
                }),
            )
            .build();

        let proto = to_proto(&schema);
        for (name, opt_matcher) in [
            (
                "text_explicit_false",
                &(|o: &v1::FieldOption| match o.option.as_ref() {
                    Some(v1::field_option::Option::Text(t)) => t.doc_values,
                    other => panic!("unexpected proto option: {other:?}"),
                }) as &dyn Fn(&v1::FieldOption) -> Option<bool>,
            ),
            (
                "integer_explicit_false",
                &(|o: &v1::FieldOption| match o.option.as_ref() {
                    Some(v1::field_option::Option::Integer(i)) => i.doc_values,
                    other => panic!("unexpected proto option: {other:?}"),
                }),
            ),
        ] {
            let doc_values = opt_matcher(proto.fields.get(name).expect("field must exist"));
            assert_eq!(
                doc_values,
                Some(false),
                "to_proto must carry an explicit doc_values: false as Some for {name}"
            );
        }

        let back = from_proto(&proto).expect("from_proto must succeed");
        match back.fields.get("text_explicit_false") {
            Some(FieldOption::Text(o)) => assert!(!o.doc_values),
            other => panic!("expected FieldOption::Text, got {other:?}"),
        }
        match back.fields.get("integer_explicit_false") {
            Some(FieldOption::Integer(o)) => assert!(!o.doc_values),
            other => panic!("expected FieldOption::Integer, got {other:?}"),
        }

        // A client omitting doc_values entirely (proto `None`) must
        // restore to the engine default (`true`), not the proto3
        // zero-value (`false`).
        let mut unset = proto.clone();
        if let Some(v1::field_option::Option::Text(t)) = unset
            .fields
            .get_mut("text_explicit_false")
            .and_then(|f| f.option.as_mut())
        {
            t.doc_values = None;
        } else {
            panic!("text_explicit_false must be a Text proto option");
        }
        let back = from_proto(&unset).expect("from_proto must succeed");
        match back.fields.get("text_explicit_false") {
            Some(FieldOption::Text(o)) => assert!(
                o.doc_values,
                "unset doc_values must default to true, not the proto3 zero-value"
            ),
            other => panic!("expected FieldOption::Text, got {other:?}"),
        }
    }
}
