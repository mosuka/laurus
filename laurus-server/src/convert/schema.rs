//! Conversion between [`laurus::Schema`] and the protobuf `Schema` message.
//!
//! Handles mapping of all field option variants (text, integer, float, boolean,
//! datetime, geo, bytes, HNSW, flat, IVF), distance metrics, and quantization
//! configuration.

use std::collections::{BTreeMap, HashMap};

use laurus::vector::core::field::{
    DEFAULT_HNSW_EF_CONSTRUCTION, DEFAULT_HNSW_M, DEFAULT_IVF_N_CLUSTERS, DEFAULT_IVF_N_PROBE,
};
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::{
    AnalyzerDefinition, AnalyzerSpec, BooleanOption, BuiltinAnalyzerSpec, BytesOption,
    DateTimeOption, DistanceMetric, DynamicFieldPolicy, EmbedderDefinition, FieldChangeKind,
    FieldOption, FlatOption, FloatOption, Geo3dOption, GeoOption, HnswOption, IntegerOption,
    IvfOption, MultiVectorOption, MultiVectorStorage, QuantizationMethod, Schema, TextOption,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::convert::proto_struct;
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
/// analyzer definition is invalid (the message names the analyzer and the
/// component, such as `Analyzer 'a': token_filters[0]: ...`, Issue #1478),
/// if an embedder definition is invalid (the message names the embedder,
/// such as ``Embedder 'e': missing field `model` ``, Issue #1702), or if
/// `dynamic_field_policy` is a value this server does not know (Issue #1473).
pub fn from_proto(proto: &v1::Schema) -> Result<Schema, String> {
    let mut fields = BTreeMap::new();
    for (name, fo) in &proto.fields {
        let option = field_option_from_proto(fo).map_err(|e| format!("Field '{name}': {e}"))?;
        fields.insert(name.clone(), option);
    }
    let mut analyzers = BTreeMap::new();
    for (name, ad) in &proto.analyzers {
        let definition =
            analyzer_definition_from_proto(ad).map_err(|e| format!("Analyzer '{name}': {e}"))?;
        analyzers.insert(name.clone(), definition);
    }
    let mut embedders = BTreeMap::new();
    for (name, ed) in &proto.embedders {
        let definition =
            embedder_definition_from_proto(ed).map_err(|e| format!("Embedder '{name}': {e}"))?;
        embedders.insert(name.clone(), definition);
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
        // `indexed` and `stored` are always set, so an explicit `false`
        // reaches the wire (Issue #1709).
        FieldOption::Text(o) => Some(Opt::Text(v1::TextOption {
            indexed: Some(o.indexed),
            stored: Some(o.stored),
            term_vectors: Some(o.term_vectors),
            analyzer: o.analyzer.as_ref().map(analyzer_spec_to_proto),
            doc_values: Some(o.doc_values),
            multi_valued: o.multi_valued,
            position_increment_gap: Some(o.position_increment_gap),
        })),
        FieldOption::Integer(o) => Some(Opt::Integer(v1::IntegerOption {
            indexed: Some(o.indexed),
            stored: Some(o.stored),
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::Float(o) => Some(Opt::Float(v1::FloatOption {
            indexed: Some(o.indexed),
            stored: Some(o.stored),
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::Boolean(o) => Some(Opt::Boolean(v1::BooleanOption {
            indexed: Some(o.indexed),
            stored: Some(o.stored),
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::DateTime(o) => Some(Opt::DateTime(v1::DateTimeOption {
            indexed: Some(o.indexed),
            stored: Some(o.stored),
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::Geo(o) => Some(Opt::Geo(v1::GeoOption {
            indexed: Some(o.indexed),
            stored: Some(o.stored),
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::Geo3d(o) => Some(Opt::Geo3d(v1::Geo3dOption {
            indexed: Some(o.indexed),
            stored: Some(o.stored),
            multi_valued: o.multi_valued,
            doc_values: Some(o.doc_values),
        })),
        FieldOption::Bytes(o) => Some(Opt::Bytes(v1::BytesOption {
            stored: Some(o.stored),
            multi_valued: o.multi_valued,
        })),
        FieldOption::Hnsw(o) => Some(Opt::Hnsw(v1::HnswOption {
            dimension: o.dimension as u32,
            distance: distance_to_proto(&o.distance) as i32,
            m: Some(o.m as u32),
            ef_construction: Some(o.ef_construction as u32),
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
            n_clusters: Some(o.n_clusters as u32),
            n_probe: Some(o.n_probe as u32),
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
/// An unset optional value of a lexical option, such as `indexed` or
/// `stored` (Issue #1709), gets the value of the core option's `Default`,
/// so a key a client leaves out means what it means in `schema.toml`.
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
        // An unset optional value of a lexical option is the core option's
        // default, as an omitted key in `schema.toml` is: `indexed` and
        // `stored` (Issue #1709), `term_vectors` (#1083), `doc_values`
        // (#1047) and `position_increment_gap` (#1175). Each arm is a struct
        // literal, so a field added to an option has to be handled here.
        Some(Opt::Text(o)) => {
            let default = TextOption::default();
            FieldOption::Text(TextOption {
                indexed: o.indexed.unwrap_or(default.indexed),
                stored: o.stored.unwrap_or(default.stored),
                multi_valued: o.multi_valued,
                position_increment_gap: o
                    .position_increment_gap
                    .unwrap_or(default.position_increment_gap),
                term_vectors: o.term_vectors.unwrap_or(default.term_vectors),
                doc_values: o.doc_values.unwrap_or(default.doc_values),
                analyzer: o.analyzer.as_ref().and_then(analyzer_spec_from_proto),
            })
        }
        Some(Opt::Integer(o)) => {
            let default = IntegerOption::default();
            FieldOption::Integer(IntegerOption {
                indexed: o.indexed.unwrap_or(default.indexed),
                stored: o.stored.unwrap_or(default.stored),
                multi_valued: o.multi_valued,
                doc_values: o.doc_values.unwrap_or(default.doc_values),
            })
        }
        Some(Opt::Float(o)) => {
            let default = FloatOption::default();
            FieldOption::Float(FloatOption {
                indexed: o.indexed.unwrap_or(default.indexed),
                stored: o.stored.unwrap_or(default.stored),
                multi_valued: o.multi_valued,
                doc_values: o.doc_values.unwrap_or(default.doc_values),
            })
        }
        Some(Opt::Boolean(o)) => {
            let default = BooleanOption::default();
            FieldOption::Boolean(BooleanOption {
                indexed: o.indexed.unwrap_or(default.indexed),
                stored: o.stored.unwrap_or(default.stored),
                multi_valued: o.multi_valued,
                doc_values: o.doc_values.unwrap_or(default.doc_values),
            })
        }
        Some(Opt::DateTime(o)) => {
            let default = DateTimeOption::default();
            FieldOption::DateTime(DateTimeOption {
                indexed: o.indexed.unwrap_or(default.indexed),
                stored: o.stored.unwrap_or(default.stored),
                multi_valued: o.multi_valued,
                doc_values: o.doc_values.unwrap_or(default.doc_values),
            })
        }
        Some(Opt::Geo(o)) => {
            let default = GeoOption::default();
            FieldOption::Geo(GeoOption {
                indexed: o.indexed.unwrap_or(default.indexed),
                stored: o.stored.unwrap_or(default.stored),
                multi_valued: o.multi_valued,
                doc_values: o.doc_values.unwrap_or(default.doc_values),
            })
        }
        Some(Opt::Geo3d(o)) => {
            let default = Geo3dOption::default();
            FieldOption::Geo3d(Geo3dOption {
                indexed: o.indexed.unwrap_or(default.indexed),
                stored: o.stored.unwrap_or(default.stored),
                multi_valued: o.multi_valued,
                doc_values: o.doc_values.unwrap_or(default.doc_values),
            })
        }
        Some(Opt::Bytes(o)) => {
            let default = BytesOption::default();
            FieldOption::Bytes(BytesOption {
                stored: o.stored.unwrap_or(default.stored),
                multi_valued: o.multi_valued,
            })
        }
        Some(Opt::Hnsw(o)) => FieldOption::Hnsw(HnswOption {
            dimension: o.dimension as usize,
            distance: distance_from_proto(o.distance)?,
            // Unset means "use the engine's default" (#1479), as for a TOML
            // schema; an explicit 0 is kept so the engine rejects it.
            m: o.m.map_or(DEFAULT_HNSW_M, |m| m as usize),
            ef_construction: o
                .ef_construction
                .map_or(DEFAULT_HNSW_EF_CONSTRUCTION, |ef| ef as usize),
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
            // Same as `m` above (#1479).
            n_clusters: o.n_clusters.map_or(DEFAULT_IVF_N_CLUSTERS, |n| n as usize),
            n_probe: o.n_probe.map_or(DEFAULT_IVF_N_PROBE, |n| n as usize),
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
        char_filters: def.char_filters.iter().map(component_to_proto).collect(),
        tokenizer: Some(component_to_proto(&def.tokenizer)),
        token_filters: def.token_filters.iter().map(component_to_proto).collect(),
    }
}

/// Convert a proto analyzer definition, naming the component that fails:
/// `tokenizer: ...`, `char_filters[1]: ...` or `token_filters[0]: ...`.
fn analyzer_definition_from_proto(
    proto: &v1::AnalyzerDefinition,
) -> Result<AnalyzerDefinition, String> {
    let tokenizer = component_from_proto(
        proto
            .tokenizer
            .as_ref()
            .ok_or("AnalyzerDefinition missing tokenizer")?,
    )
    .map_err(|e| format!("tokenizer: {e}"))?;
    let char_filters = proto
        .char_filters
        .iter()
        .enumerate()
        .map(|(i, c)| component_from_proto(c).map_err(|e| format!("char_filters[{i}]: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    let token_filters = proto
        .token_filters
        .iter()
        .enumerate()
        .map(|(i, c)| component_from_proto(c).map_err(|e| format!("token_filters[{i}]: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AnalyzerDefinition {
        char_filters,
        tokenizer,
        token_filters,
    })
}

/// The error for a component in the form before Issue #1478, from a server
/// or a client on the other side of that change.
const LEGACY_COMPONENT_ERROR: &str = "the component sets `type` or `params`, the form \
     before Issue #1478, which is no longer read; send it as `config` (a server and a \
     client from before and after that change do not work together)";

/// Convert a core analyzer component ([`laurus::TokenizerConfig`],
/// [`laurus::CharFilterConfig`] or [`laurus::TokenFilterConfig`]) to its proto
/// form, whose `config` holds the component's serde form (see
/// [`serde_to_struct`], Issue #1478). The fields of the form before Issue
/// #1478 stay empty, so that a server from before then rejects the component
/// as an unknown type instead of building it with its defaults.
fn component_to_proto<T: Serialize>(component: &T) -> v1::ComponentConfig {
    v1::ComponentConfig {
        config: Some(serde_to_struct(component)),
        ..Default::default()
    }
}

/// Convert a proto analyzer component to a core one with serde (see
/// [`struct_to_serde`], Issue #1478).
///
/// # Errors
///
/// Returns an error for a component that sets a field of the form before
/// Issue #1478 (`type`, `params`) or has no `config`, or whose `config` does
/// not convert (see [`struct_to_serde`]).
fn component_from_proto<T: DeserializeOwned>(proto: &v1::ComponentConfig) -> Result<T, String> {
    #[allow(deprecated)]
    let legacy = !proto.r#type.is_empty() || !proto.params.is_empty();
    if legacy {
        return Err(LEGACY_COMPONENT_ERROR.to_string());
    }
    let config = proto
        .config
        .as_ref()
        .ok_or("the component has no `config`")?;
    struct_to_serde(config)
}

// ---- Embedder definition conversion ----

/// The error for an embedder in the form before Issue #1702, from a server or
/// a client on the other side of that change.
const LEGACY_EMBEDDER_ERROR: &str = "the embedder sets `type` or `params`, the form \
     before Issue #1702, which is no longer read; send it as `config` (a server and a \
     client from before and after that change do not work together)";

/// Convert a core embedder definition to its proto form, whose `config` holds
/// the definition's serde form (see [`serde_to_struct`], Issue #1702). The
/// fields of the form before Issue #1702 stay empty, so that a server from
/// before then rejects the embedder as an unknown type instead of reading
/// something else.
fn embedder_definition_to_proto(def: &EmbedderDefinition) -> v1::EmbedderConfig {
    v1::EmbedderConfig {
        config: Some(serde_to_struct(def)),
        ..Default::default()
    }
}

/// Convert a proto embedder definition to a core one with serde (see
/// [`struct_to_serde`], Issue #1702), so that a parameter is kept with its
/// type and a key the type does not take is rejected, as in `schema.toml`.
///
/// # Errors
///
/// Returns an error for an embedder that sets a field of the form before
/// Issue #1702 (`type`, `params`) or has no `config`, or whose `config` does
/// not convert (see [`struct_to_serde`]).
fn embedder_definition_from_proto(
    proto: &v1::EmbedderConfig,
) -> Result<EmbedderDefinition, String> {
    #[allow(deprecated)]
    let legacy = !proto.r#type.is_empty() || !proto.params.is_empty();
    if legacy {
        return Err(LEGACY_EMBEDDER_ERROR.to_string());
    }
    let config = proto
        .config
        .as_ref()
        .ok_or("the embedder has no `config`")?;
    struct_to_serde(config)
}

// ---- Serde forms in a `Struct` ----

/// Convert a value of a core schema type (an analyzer component or an
/// embedder definition) to a `Struct` holding its serde form: the shape
/// `schema.toml` stores and the HTTP API takes (Issues #1478, #1702).
///
/// The value goes through JSON text, so that an `f32` such as a boost of 0.1
/// is written as 0.1; `serde_json::to_value` would widen it to
/// 0.10000000149011612 first. An unset optional parameter, which serializes as
/// `null`, is left out, while an empty list or table (`words: []`,
/// `mapping: {}`) is kept, since it is not the same as an omitted key.
fn serde_to_struct<T: Serialize>(value: &T) -> prost_types::Struct {
    let text = serde_json::to_string(value).expect("a schema type serializes to JSON");
    let serde_json::Value::Object(mut obj) =
        serde_json::from_str(&text).expect("serialized JSON parses")
    else {
        unreachable!("a schema type serializes to a JSON object");
    };
    obj.retain(|_, value| !value.is_null());
    proto_struct::json_object_to_proto_struct(&obj)
}

/// Convert a `Struct` a client sent to a core schema type with serde, so that
/// it is read by the rules `schema.toml` is read by: the same types, required
/// keys and defaults (Issues #1478, #1702). A top-level `null` is the same as
/// omitting the key, as everywhere in the HTTP API.
///
/// # Errors
///
/// Returns an error for a `Struct` that holds a number that is not finite or
/// a value with no kind (see [`proto_struct::proto_struct_to_json`]), or that
/// does not deserialize: an unknown `type`, a missing required key, a key the
/// type does not take, or a value of the wrong type.
fn struct_to_serde<T: DeserializeOwned>(config: &prost_types::Struct) -> Result<T, String> {
    let mut obj = proto_struct::proto_struct_to_json(config)?;
    obj.retain(|_, value| !value.is_null());
    serde_json::from_value(serde_json::Value::Object(obj)).map_err(|e| e.to_string())
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

    /// Issue #1702: embedders travel in `EmbedderConfig.config` as their
    /// serde form and are read back with serde, as analyzer components are
    /// (Issue #1478).
    mod embedder_definitions {
        use prost::Message;
        use serde_json::{Value, json};

        use super::super::*;

        /// An embedder whose `config` is `json`, which must be an object.
        fn embedder(json: Value) -> v1::EmbedderConfig {
            v1::EmbedderConfig {
                config: Some(proto_struct::json_object_to_proto_struct(
                    json.as_object().expect("an embedder is an object"),
                )),
                ..Default::default()
            }
        }

        /// A schema whose only embedder is `proto`, named `e`.
        fn schema_with(proto: v1::EmbedderConfig) -> v1::Schema {
            v1::Schema {
                embedders: HashMap::from([("e".to_string(), proto)]),
                ..Default::default()
            }
        }

        /// A `candle_colbert` embedder with all of its optional parameters.
        fn full_colbert() -> EmbedderDefinition {
            EmbedderDefinition::CandleColbert {
                model: "colbert-ir/colbertv2.0".into(),
                revision: Some("c1e84128".into()),
                query_maxlen: Some(32),
                doc_maxlen: Some(180),
            }
        }

        /// Every embedder type, `candle_colbert` with all of its optional
        /// parameters and with none.
        fn every_embedder() -> Vec<EmbedderDefinition> {
            let all = vec![
                EmbedderDefinition::Precomputed,
                EmbedderDefinition::CandleBert {
                    model: "sentence-transformers/all-MiniLM-L6-v2".into(),
                },
                EmbedderDefinition::CandleClip {
                    model: "openai/clip-vit-base-patch32".into(),
                },
                EmbedderDefinition::Openai {
                    model: "text-embedding-3-small".into(),
                },
                full_colbert(),
                EmbedderDefinition::CandleColbert {
                    model: "answerdotai/answerai-colbert-small-v1".into(),
                    revision: None,
                    query_maxlen: None,
                    doc_maxlen: None,
                },
            ];
            // No wildcard: a new embedder type does not compile here until
            // it is added above.
            for definition in &all {
                match definition {
                    EmbedderDefinition::Precomputed
                    | EmbedderDefinition::CandleBert { .. }
                    | EmbedderDefinition::CandleClip { .. }
                    | EmbedderDefinition::Openai { .. }
                    | EmbedderDefinition::CandleColbert { .. } => {}
                }
            }
            all
        }

        /// Every type goes to proto with its serde form in `config` and
        /// nothing in the fields of the form before Issue #1702, through the
        /// protobuf encoding, and back unchanged. The core type does not
        /// implement `PartialEq`, so it is compared as serde JSON.
        #[test]
        fn every_embedder_type_round_trips() {
            for definition in every_embedder() {
                let serde_form = serde_json::to_value(&definition).unwrap();
                let proto = embedder_definition_to_proto(&definition);
                let config = proto_struct::proto_struct_to_json(
                    proto.config.as_ref().expect("config is set"),
                )
                .unwrap();
                assert_eq!(Value::Object(config), serde_form);
                #[allow(deprecated)]
                let legacy_is_empty = proto.r#type.is_empty() && proto.params.is_empty();
                assert!(legacy_is_empty, "{definition:?}");

                let decoded = v1::EmbedderConfig::decode(proto.encode_to_vec().as_slice())
                    .expect("the message decodes");
                let back = embedder_definition_from_proto(&decoded)
                    .unwrap_or_else(|e| panic!("{definition:?} does not come back: {e}"));
                assert_eq!(serde_json::to_value(&back).unwrap(), serde_form);
            }

            // And through the whole schema, beside the field that names it.
            let schema = Schema::builder()
                .add_embedder("colbert", full_colbert())
                .add_multi_vector_field("tokens", MultiVectorOption::new(128).embedder("colbert"))
                .build();
            let back = from_proto(&to_proto(&schema)).expect("from_proto must succeed");
            assert_eq!(
                serde_json::to_value(&back.embedders["colbert"]).unwrap(),
                serde_json::to_value(full_colbert()).unwrap()
            );
            assert_eq!(back.fields["tokens"].embedder_name(), Some("colbert"));
        }

        #[test]
        fn the_form_before_issue_1702_is_rejected() {
            let precomputed = embedder(json!({"type": "precomputed"}));
            #[allow(deprecated)]
            let cases = [
                // `type` set as well as `config`.
                v1::EmbedderConfig {
                    r#type: "precomputed".into(),
                    ..precomputed.clone()
                },
                // `params` set as well as `config`.
                v1::EmbedderConfig {
                    params: HashMap::from([("model".to_string(), "m".to_string())]),
                    ..embedder(json!({"type": "candle_bert", "model": "m"}))
                },
                // The old form of an embedder with no parameters.
                v1::EmbedderConfig {
                    r#type: "precomputed".into(),
                    ..Default::default()
                },
                // The old form with parameters.
                v1::EmbedderConfig {
                    r#type: "candle_bert".into(),
                    params: HashMap::from([("model".to_string(), "m".to_string())]),
                    config: None,
                },
            ];
            for proto in &cases {
                let err = embedder_definition_from_proto(proto).unwrap_err();
                assert_eq!(err, LEGACY_EMBEDDER_ERROR, "{proto:?}");
            }

            let err = embedder_definition_from_proto(&v1::EmbedderConfig::default()).unwrap_err();
            assert_eq!(err, "the embedder has no `config`");
            assert!(embedder_definition_from_proto(&precomputed).is_ok());
        }

        /// The embedders of the reproduction in Issue #1702, which the server
        /// used to store with another type or without a key, and others that
        /// `schema.toml` rejects too.
        #[test]
        fn keys_and_types_are_the_ones_schema_toml_takes() {
            let cases: [(Value, &str); 8] = [
                (
                    json!({"type": "candle_colbert", "model": "m", "query_maxlen": "32"}),
                    "invalid type: string \"32\", expected usize",
                ),
                (
                    json!({"type": "candle_colbert", "model": "m", "revision": 123}),
                    "invalid type: integer `123`, expected a string",
                ),
                (
                    json!({"type": "candle_bert", "model": true}),
                    "invalid type: boolean `true`, expected a string",
                ),
                (
                    json!({"type": "candle_colbert", "model": "m", "doc_max_len": 300}),
                    "unknown field `doc_max_len`",
                ),
                (
                    json!({"type": "precomputed", "model": "x"}),
                    "unknown field `model`",
                ),
                (json!({"type": "candle_bert"}), "missing field `model`"),
                (json!({"model": "m"}), "missing field `type`"),
                (json!({"type": "nope"}), "unknown variant `nope`"),
            ];
            for (json, expected) in cases {
                let err = embedder_definition_from_proto(&embedder(json.clone())).unwrap_err();
                assert!(err.contains(expected), "{json}: {err}");
                // schema.toml rejects the same embedder.
                let toml_value: toml::Value = serde_json::from_value(json.clone()).unwrap();
                let toml_text = toml::to_string(&toml_value).unwrap();
                assert!(
                    toml::from_str::<EmbedderDefinition>(&toml_text).is_err(),
                    "schema.toml takes {json}"
                );
            }
        }

        #[test]
        fn whole_doubles_fill_lengths_and_others_are_rejected() {
            let colbert = embedder_definition_from_proto(&embedder(json!({
                "type": "candle_colbert",
                "model": "m",
                "query_maxlen": 32.0,
                "doc_maxlen": 180.0
            })))
            .unwrap();
            assert!(matches!(
                colbert,
                EmbedderDefinition::CandleColbert {
                    query_maxlen: Some(32),
                    doc_maxlen: Some(180),
                    ..
                }
            ));
            for n in [32.5, -1.0, 1e20] {
                let result = embedder_definition_from_proto(&embedder(
                    json!({"type": "candle_colbert", "model": "m", "query_maxlen": n}),
                ));
                assert!(
                    result.is_err(),
                    "query_maxlen {n} must be rejected: {result:?}"
                );
            }
        }

        #[test]
        fn a_top_level_null_is_the_same_as_omitting_the_key() {
            let colbert = embedder_definition_from_proto(&embedder(json!({
                "type": "candle_colbert",
                "model": "m",
                "revision": null,
                "query_maxlen": null
            })))
            .unwrap();
            assert!(matches!(
                colbert,
                EmbedderDefinition::CandleColbert {
                    revision: None,
                    query_maxlen: None,
                    doc_maxlen: None,
                    ..
                }
            ));
            // So a key the type does not take is not reported when it is
            // `null`: it sets nothing.
            let precomputed = embedder_definition_from_proto(&embedder(
                json!({"type": "precomputed", "model": null}),
            ));
            assert!(
                matches!(precomputed, Ok(EmbedderDefinition::Precomputed)),
                "{precomputed:?}"
            );
        }

        #[test]
        fn errors_name_the_embedder() {
            #[allow(deprecated)]
            let legacy = v1::EmbedderConfig {
                r#type: "precomputed".into(),
                ..Default::default()
            };
            let cases = [
                (
                    embedder(json!({"type": "candle_bert"})),
                    "Embedder 'e': missing field `model`".to_string(),
                ),
                (
                    embedder(json!({"type": "precomputed", "model": "x"})),
                    "Embedder 'e': unknown field `model`".to_string(),
                ),
                (legacy, format!("Embedder 'e': {LEGACY_EMBEDDER_ERROR}")),
                (
                    v1::EmbedderConfig::default(),
                    "Embedder 'e': the embedder has no `config`".to_string(),
                ),
            ];
            for (proto, prefix) in cases {
                let err = from_proto(&schema_with(proto)).unwrap_err();
                assert!(err.starts_with(&prefix), "{err}");
            }
        }
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
                indexed: Some(true),
                stored: Some(true),
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

    /// Issue #1478: analyzer components travel in `ComponentConfig.config`
    /// as their serde form and are read back with serde.
    mod analyzer_components {
        use std::collections::BTreeMap;
        use std::fmt::Debug;

        use laurus::{CharFilterConfig, TokenFilterConfig, TokenizerConfig};
        use prost::Message;
        use prost_types::value::Kind;
        use serde_json::{Value, json};

        use super::super::*;

        /// A component whose `config` is `json`, which must be an object.
        fn component(json: Value) -> v1::ComponentConfig {
            v1::ComponentConfig {
                config: Some(proto_struct::json_object_to_proto_struct(
                    json.as_object().expect("a component is an object"),
                )),
                ..Default::default()
            }
        }

        /// The JSON object a component's `config` holds.
        fn config_json(proto: &v1::ComponentConfig) -> Value {
            Value::Object(
                proto_struct::proto_struct_to_json(proto.config.as_ref().expect("config is set"))
                    .expect("config converts to JSON"),
            )
        }

        /// Every tokenizer type, with parameters other than the defaults.
        fn every_tokenizer() -> Vec<TokenizerConfig> {
            let all = vec![
                TokenizerConfig::Whitespace,
                TokenizerConfig::UnicodeWord,
                TokenizerConfig::Regex {
                    pattern: "[a-z]+".into(),
                    gaps: true,
                },
                TokenizerConfig::Ngram {
                    min_gram: 2,
                    max_gram: 3,
                },
                TokenizerConfig::Lindera {
                    mode: "decompose".into(),
                    dict: "/var/lib/lindera/ipadic".into(),
                    user_dict: Some("/var/lib/lindera/user.csv".into()),
                },
                TokenizerConfig::Whole,
            ];
            // No wildcard: a new tokenizer type does not compile here until
            // it is added above.
            for tokenizer in &all {
                match tokenizer {
                    TokenizerConfig::Whitespace
                    | TokenizerConfig::UnicodeWord
                    | TokenizerConfig::Regex { .. }
                    | TokenizerConfig::Ngram { .. }
                    | TokenizerConfig::Lindera { .. }
                    | TokenizerConfig::Whole => {}
                }
            }
            all
        }

        /// Every char filter type, with a mapping that has a key `type` and
        /// an empty one.
        fn every_char_filter() -> Vec<CharFilterConfig> {
            let all = vec![
                CharFilterConfig::UnicodeNormalization { form: "nfd".into() },
                CharFilterConfig::PatternReplace {
                    pattern: "a+".into(),
                    replacement: "b".into(),
                },
                CharFilterConfig::Mapping {
                    mapping: BTreeMap::from([
                        ("type".to_string(), "t".to_string()),
                        ("ph".to_string(), "f".to_string()),
                    ]),
                },
                CharFilterConfig::Mapping {
                    mapping: BTreeMap::new(),
                },
                CharFilterConfig::JapaneseIterationMark {
                    kanji: false,
                    kana: true,
                },
            ];
            for char_filter in &all {
                match char_filter {
                    CharFilterConfig::UnicodeNormalization { .. }
                    | CharFilterConfig::PatternReplace { .. }
                    | CharFilterConfig::Mapping { .. }
                    | CharFilterConfig::JapaneseIterationMark { .. } => {}
                }
            }
            all
        }

        /// Every token filter type, with stop words that have commas and
        /// spaces, an empty stop list and no stop list, and a boost of 0.1.
        fn every_token_filter() -> Vec<TokenFilterConfig> {
            let all = vec![
                TokenFilterConfig::Lowercase,
                TokenFilterConfig::Stop {
                    words: Some(vec!["new,york".into(), " padded ".into(), "a,b".into()]),
                },
                TokenFilterConfig::Stop {
                    words: Some(vec![]),
                },
                TokenFilterConfig::Stop { words: None },
                TokenFilterConfig::Stem {
                    stem_type: Some("simple".into()),
                },
                TokenFilterConfig::Stem { stem_type: None },
                TokenFilterConfig::Boost { boost: 0.1 },
                TokenFilterConfig::Limit { limit: 5 },
                TokenFilterConfig::Strip,
                TokenFilterConfig::RemoveEmpty,
                TokenFilterConfig::FlattenGraph,
            ];
            for token_filter in &all {
                match token_filter {
                    TokenFilterConfig::Lowercase
                    | TokenFilterConfig::Stop { .. }
                    | TokenFilterConfig::Stem { .. }
                    | TokenFilterConfig::Boost { .. }
                    | TokenFilterConfig::Limit { .. }
                    | TokenFilterConfig::Strip
                    | TokenFilterConfig::RemoveEmpty
                    | TokenFilterConfig::FlattenGraph => {}
                }
            }
            all
        }

        /// `component` goes to proto, through the protobuf encoding, and
        /// back unchanged. The core types do not implement `PartialEq`, so
        /// they are compared as serde JSON.
        fn assert_round_trips<T: Serialize + DeserializeOwned + Debug>(component: &T) {
            let proto = component_to_proto(component);
            let decoded = v1::ComponentConfig::decode(proto.encode_to_vec().as_slice())
                .expect("the message decodes");
            let back: T = component_from_proto(&decoded)
                .unwrap_or_else(|e| panic!("{component:?} does not come back: {e}"));
            assert_eq!(
                serde_json::to_value(&back).unwrap(),
                serde_json::to_value(component).unwrap(),
                "{component:?}"
            );
        }

        #[test]
        fn every_component_type_round_trips() {
            every_tokenizer().iter().for_each(assert_round_trips);
            every_char_filter().iter().for_each(assert_round_trips);
            every_token_filter().iter().for_each(assert_round_trips);
        }

        #[test]
        fn to_proto_writes_the_serde_form_in_config() {
            let cases = [
                (
                    component_to_proto(&TokenFilterConfig::Stop {
                        words: Some(vec!["a,b".into(), " c".into()]),
                    }),
                    json!({"type": "stop", "words": ["a,b", " c"]}),
                ),
                (
                    component_to_proto(&TokenFilterConfig::Stop {
                        words: Some(vec![]),
                    }),
                    json!({"type": "stop", "words": []}),
                ),
                (
                    component_to_proto(&TokenFilterConfig::Stop { words: None }),
                    json!({"type": "stop"}),
                ),
                (
                    component_to_proto(&CharFilterConfig::Mapping {
                        mapping: BTreeMap::new(),
                    }),
                    json!({"type": "mapping", "mapping": {}}),
                ),
                (
                    component_to_proto(&TokenizerConfig::Lindera {
                        mode: "normal".into(),
                        dict: "/d".into(),
                        user_dict: None,
                    }),
                    json!({"type": "lindera", "mode": "normal", "dict": "/d"}),
                ),
                (
                    component_to_proto(&TokenizerConfig::Ngram {
                        min_gram: 2,
                        max_gram: 3,
                    }),
                    json!({"type": "ngram", "min_gram": 2, "max_gram": 3}),
                ),
            ];
            for (proto, expected) in cases {
                assert_eq!(config_json(&proto), expected);
                #[allow(deprecated)]
                let legacy_is_empty = proto.r#type.is_empty() && proto.params.is_empty();
                assert!(legacy_is_empty, "{expected}");
            }

            // 0.1 as the double 0.1, not the widened f32 0.10000000149011612.
            let proto = component_to_proto(&TokenFilterConfig::Boost { boost: 0.1 });
            let boost = &proto.config.as_ref().unwrap().fields["boost"];
            assert_eq!(boost.kind, Some(Kind::NumberValue(0.1)));
        }

        #[test]
        fn the_form_before_issue_1478_is_rejected() {
            let whitespace = component(json!({"type": "whitespace"}));
            #[allow(deprecated)]
            let cases = [
                // `type` set as well as `config`.
                v1::ComponentConfig {
                    r#type: "whitespace".into(),
                    ..whitespace.clone()
                },
                // `params` that would change nothing if ignored.
                v1::ComponentConfig {
                    params: HashMap::from([("form".to_string(), "nfc".to_string())]),
                    ..component(json!({"type": "unicode_normalization", "form": "nfc"}))
                },
                // The old form alone.
                v1::ComponentConfig {
                    r#type: "lowercase".into(),
                    ..Default::default()
                },
            ];
            for proto in &cases {
                let err = component_from_proto::<TokenizerConfig>(proto).unwrap_err();
                assert_eq!(err, LEGACY_COMPONENT_ERROR, "{proto:?}");
            }

            let err = component_from_proto::<TokenizerConfig>(&v1::ComponentConfig::default())
                .unwrap_err();
            assert_eq!(err, "the component has no `config`");
            assert!(component_from_proto::<TokenizerConfig>(&whitespace).is_ok());
        }

        #[test]
        fn whole_doubles_fill_integer_parameters_and_others_are_rejected() {
            let ngram: TokenizerConfig = component_from_proto(&component(
                json!({"type": "ngram", "min_gram": 2.0, "max_gram": 3.0}),
            ))
            .unwrap();
            assert!(matches!(
                ngram,
                TokenizerConfig::Ngram {
                    min_gram: 2,
                    max_gram: 3
                }
            ));
            for limit in [2.5, -1.0, 1e20, 18_446_744_073_709_551_616.0] {
                let result = component_from_proto::<TokenFilterConfig>(&component(
                    json!({"type": "limit", "limit": limit}),
                ));
                assert!(
                    result.is_err(),
                    "limit {limit} must be rejected: {result:?}"
                );
            }
        }

        #[test]
        fn a_number_that_is_not_finite_or_a_value_with_no_kind_is_rejected() {
            let with_limit = |kind: Option<Kind>| {
                let mut proto = component(json!({"type": "limit", "limit": 1}));
                proto
                    .config
                    .as_mut()
                    .unwrap()
                    .fields
                    .insert("limit".to_string(), prost_types::Value { kind });
                proto
            };
            for n in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let err = component_from_proto::<TokenFilterConfig>(&with_limit(Some(
                    Kind::NumberValue(n),
                )))
                .unwrap_err();
                assert!(
                    err.starts_with("limit: ") && err.contains("finite"),
                    "{err}"
                );
            }
            let err = component_from_proto::<TokenFilterConfig>(&with_limit(None)).unwrap_err();
            assert_eq!(err, "limit: the value has no kind");
        }

        #[test]
        fn required_keys_and_types_are_the_ones_schema_toml_has() {
            let cases: [(Value, &str); 7] = [
                (
                    json!({"type": "unicode_normalization"}),
                    "missing field `form`",
                ),
                (
                    json!({"type": "pattern_replace", "pattern": "a"}),
                    "missing field `replacement`",
                ),
                (
                    json!({"type": "lindera", "dict": "/d"}),
                    "missing field `mode`",
                ),
                (
                    json!({"type": "ngram", "max_gram": 3}),
                    "missing field `min_gram`",
                ),
                (
                    json!({"type": "japanese_iteration_mark", "kanji": "False"}),
                    "invalid type: string \"False\", expected a boolean",
                ),
                (
                    json!({"type": "ngram", "min_gram": "2", "max_gram": 3}),
                    "invalid type: string \"2\", expected usize",
                ),
                (json!({"type": "nope"}), "unknown variant `nope`"),
            ];
            for (json, expected) in cases {
                let proto = component(json.clone());
                let err = match json["type"].as_str() {
                    Some(
                        "unicode_normalization" | "pattern_replace" | "japanese_iteration_mark",
                    ) => component_from_proto::<CharFilterConfig>(&proto).map(|_| ()),
                    _ => component_from_proto::<TokenizerConfig>(&proto).map(|_| ()),
                }
                .unwrap_err();
                assert!(err.contains(expected), "{json}: {err}");
                // schema.toml rejects the same component.
                let toml_value: toml::Value = serde_json::from_value(json.clone()).unwrap();
                let toml_text = toml::to_string(&toml_value).unwrap();
                let from_toml = match json["type"].as_str() {
                    Some(
                        "unicode_normalization" | "pattern_replace" | "japanese_iteration_mark",
                    ) => toml::from_str::<CharFilterConfig>(&toml_text).map(|_| ()),
                    _ => toml::from_str::<TokenizerConfig>(&toml_text).map(|_| ()),
                };
                assert!(from_toml.is_err(), "schema.toml takes {json}");
            }
        }

        #[test]
        fn a_top_level_null_is_the_same_as_omitting_the_key() {
            let regex: TokenizerConfig = component_from_proto(&component(
                json!({"type": "regex", "pattern": null, "gaps": null}),
            ))
            .unwrap();
            assert!(
                matches!(&regex, TokenizerConfig::Regex { pattern, gaps: false } if pattern == r"\w+"),
                "{regex:?}"
            );
            let marks: CharFilterConfig = component_from_proto(&component(
                json!({"type": "japanese_iteration_mark", "kanji": null}),
            ))
            .unwrap();
            assert!(matches!(
                marks,
                CharFilterConfig::JapaneseIterationMark {
                    kanji: true,
                    kana: true
                }
            ));
            let stop: TokenFilterConfig =
                component_from_proto(&component(json!({"type": "stop", "words": null}))).unwrap();
            assert!(matches!(stop, TokenFilterConfig::Stop { words: None }));
        }

        #[test]
        fn errors_name_the_analyzer_and_the_component() {
            let analyzer = |char_filters, tokenizer, token_filters| v1::Schema {
                analyzers: HashMap::from([(
                    "a".to_string(),
                    v1::AnalyzerDefinition {
                        char_filters,
                        tokenizer: Some(tokenizer),
                        token_filters,
                    },
                )]),
                ..Default::default()
            };
            let ok = || component(json!({"type": "lowercase"}));
            let bad = || component(json!({"type": "limit"}));
            let whitespace = || component(json!({"type": "whitespace"}));
            let cases = [
                (analyzer(vec![], bad(), vec![]), "Analyzer 'a': tokenizer: "),
                (
                    analyzer(vec![bad()], whitespace(), vec![]),
                    "Analyzer 'a': char_filters[0]: ",
                ),
                (
                    analyzer(vec![], whitespace(), vec![ok(), bad()]),
                    "Analyzer 'a': token_filters[1]: ",
                ),
            ];
            for (schema, prefix) in cases {
                let err = from_proto(&schema).unwrap_err();
                assert!(err.starts_with(prefix), "{err}");
            }
        }
    }

    /// `indexed` (`None` for `BytesOption`, which has none) and `stored` of
    /// a lexical option.
    fn indexed_and_stored(option: &FieldOption) -> (Option<bool>, bool) {
        match option {
            FieldOption::Text(o) => (Some(o.indexed), o.stored),
            FieldOption::Integer(o) => (Some(o.indexed), o.stored),
            FieldOption::Float(o) => (Some(o.indexed), o.stored),
            FieldOption::Boolean(o) => (Some(o.indexed), o.stored),
            FieldOption::DateTime(o) => (Some(o.indexed), o.stored),
            FieldOption::Geo(o) => (Some(o.indexed), o.stored),
            FieldOption::Geo3d(o) => (Some(o.indexed), o.stored),
            FieldOption::Bytes(o) => (None, o.stored),
            other => panic!("not a lexical option: {other:?}"),
        }
    }

    /// Issue #1709: a lexical option whose `indexed` and `stored` a gRPC
    /// client leaves out gets the core option's defaults (`true`), as an
    /// omitted key in `schema.toml` does. Both used to be `false`, so the
    /// field was neither searchable nor returned.
    #[test]
    fn unset_indexed_and_stored_get_the_core_defaults() {
        use v1::field_option::Option as Opt;

        let cases = [
            (
                "text",
                Opt::Text(v1::TextOption::default()),
                FieldOption::Text(TextOption::default()),
            ),
            (
                "integer",
                Opt::Integer(v1::IntegerOption::default()),
                FieldOption::Integer(IntegerOption::default()),
            ),
            (
                "float",
                Opt::Float(v1::FloatOption::default()),
                FieldOption::Float(FloatOption::default()),
            ),
            (
                "boolean",
                Opt::Boolean(v1::BooleanOption::default()),
                FieldOption::Boolean(BooleanOption::default()),
            ),
            (
                "date_time",
                Opt::DateTime(v1::DateTimeOption::default()),
                FieldOption::DateTime(DateTimeOption::default()),
            ),
            (
                "geo",
                Opt::Geo(v1::GeoOption::default()),
                FieldOption::Geo(GeoOption::default()),
            ),
            (
                "geo3d",
                Opt::Geo3d(v1::Geo3dOption::default()),
                FieldOption::Geo3d(Geo3dOption::default()),
            ),
            (
                "bytes",
                Opt::Bytes(v1::BytesOption::default()),
                FieldOption::Bytes(BytesOption::default()),
            ),
        ];
        let mut wrong = Vec::new();
        for (kind, proto, default) in cases {
            let (want_indexed, want_stored) = indexed_and_stored(&default);
            // The test can only tell the defaults from `false` while they
            // are `true`.
            assert!(want_stored && want_indexed != Some(false), "{kind}");
            let converted = field_option_from_proto(&v1::FieldOption {
                option: Some(proto),
            })
            .unwrap();
            let (indexed, stored) = indexed_and_stored(&converted);
            if indexed != want_indexed {
                wrong.push(format!("{kind}.indexed: {indexed:?}"));
            }
            if stored != want_stored {
                wrong.push(format!("{kind}.stored: {stored}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1709: an explicit `indexed` and `stored`, `false` included,
    /// survive the path MCP takes: a core option converted to proto, sent
    /// over the wire, and converted back. A plain proto `bool` put nothing on
    /// the wire for `false`; an unset `optional bool` would now read as
    /// `true`.
    #[test]
    fn explicit_indexed_and_stored_survive_the_wire() {
        use prost::Message as _;

        let options = |indexed: bool, stored: bool| {
            [
                (
                    "text",
                    FieldOption::Text(TextOption {
                        indexed,
                        stored,
                        ..Default::default()
                    }),
                ),
                (
                    "integer",
                    FieldOption::Integer(IntegerOption {
                        indexed,
                        stored,
                        ..Default::default()
                    }),
                ),
                (
                    "float",
                    FieldOption::Float(FloatOption {
                        indexed,
                        stored,
                        ..Default::default()
                    }),
                ),
                (
                    "boolean",
                    FieldOption::Boolean(BooleanOption {
                        indexed,
                        stored,
                        ..Default::default()
                    }),
                ),
                (
                    "date_time",
                    FieldOption::DateTime(DateTimeOption {
                        indexed,
                        stored,
                        ..Default::default()
                    }),
                ),
                (
                    "geo",
                    FieldOption::Geo(GeoOption {
                        indexed,
                        stored,
                        ..Default::default()
                    }),
                ),
                (
                    "geo3d",
                    FieldOption::Geo3d(Geo3dOption {
                        indexed,
                        stored,
                        ..Default::default()
                    }),
                ),
                (
                    "bytes",
                    FieldOption::Bytes(BytesOption {
                        stored,
                        ..Default::default()
                    }),
                ),
            ]
        };
        let mut wrong = Vec::new();
        for indexed in [true, false] {
            for stored in [true, false] {
                for (kind, option) in options(indexed, stored) {
                    let bytes = field_option_to_proto(&option).encode_to_vec();
                    let decoded = v1::FieldOption::decode(bytes.as_slice()).unwrap();
                    let back = field_option_from_proto(&decoded).unwrap();
                    let got = indexed_and_stored(&back);
                    if got != indexed_and_stored(&option) {
                        wrong.push(format!("{kind} indexed={indexed} stored={stored}: {got:?}"));
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }
}
