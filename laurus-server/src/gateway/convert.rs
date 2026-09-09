//! Conversion helpers between JSON (`serde_json::Value`) and proto messages.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::convert::document::to_proto;
use crate::proto::laurus::v1;

// ---------------------------------------------------------------------------
// Value conversion
// ---------------------------------------------------------------------------

/// Converts a proto `Value` to a JSON value.
pub fn proto_value_to_json(val: &v1::Value) -> Value {
    use v1::value::Kind;
    match &val.kind {
        Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(*b),
        Some(Kind::Int64Value(i)) => json!(*i),
        Some(Kind::Float64Value(f)) => json!(*f),
        Some(Kind::TextValue(s)) => Value::String(s.clone()),
        Some(Kind::BytesValue(b)) => {
            use std::io::Write;
            let mut buf = Vec::new();
            // Base64 encode
            let engine = base64_engine();
            write!(buf, "{}", base64_encode(&engine, b)).ok();
            Value::String(String::from_utf8_lossy(&buf).into_owned())
        }
        Some(Kind::VectorValue(v)) => Value::Array(v.values.iter().map(|f| json!(*f)).collect()),
        Some(Kind::DatetimeValue(us)) => {
            let secs = us / 1_000_000;
            let nanos = ((us % 1_000_000) * 1_000) as u32;
            if let Some(dt) = chrono::DateTime::from_timestamp(secs, nanos) {
                Value::String(dt.to_rfc3339())
            } else {
                json!(*us)
            }
        }
        Some(Kind::GeoValue(g)) => {
            json!({"latitude": g.latitude, "longitude": g.longitude})
        }
        Some(Kind::Geo3dValue(p)) => {
            json!({"x": p.x, "y": p.y, "z": p.z})
        }
        Some(Kind::Int64ArrayValue(arr)) => {
            Value::Array(arr.values.iter().map(|v| json!(*v)).collect())
        }
        Some(Kind::Float64ArrayValue(arr)) => {
            Value::Array(arr.values.iter().map(|v| json!(*v)).collect())
        }
        None => Value::Null,
    }
}

// Base64 helper (standard encoding without external crates)
fn base64_engine() -> Base64Engine {
    Base64Engine
}

struct Base64Engine;

fn base64_encode(_engine: &Base64Engine, data: &[u8]) -> String {
    // Standard base64 encoding implementation
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Document conversion
// ---------------------------------------------------------------------------

/// Convert a JSON object to a proto `Document`.
///
/// Input format: `{"fields": {"field_name": value, ...}}`. Delegates to
/// [`laurus::json_to_document`] — the same canonical converter used by
/// laurus-cli and laurus-mcp, so all three transports accept identical
/// document JSON — and lowers the resulting [`laurus::Document`] to the
/// proto wire form via [`crate::convert::document::to_proto`].
///
/// # Arguments
///
/// * `json` - The JSON document to convert.
///
/// # Errors
///
/// Returns a `String` error message when the input is structurally invalid
/// (missing `fields` key, non-object `fields`) or when any field value
/// cannot be inferred. The error includes the offending field name.
pub fn json_to_proto_document(json: &Value) -> Result<v1::Document, String> {
    let doc = laurus::json_to_document(json).map_err(|e| e.to_string())?;
    Ok(to_proto(&doc))
}

/// Converts a proto `Document`'s fields into a JSON fields map: `{field:
/// value, ...}`, with no `"fields"` wrapper and no `id`.
///
/// Shared by [`proto_document_to_json`] (wraps the result under `"fields"`)
/// and [`proto_search_result_to_json`] (inserts it directly under the
/// search result's own `"fields"` key), so both surfaces agree on the same
/// `{"fields": {...}}` shape [`json_to_proto_document`] accepts back.
///
/// The internal `_id` system field is excluded from the output since it
/// is already available as the top-level `id` in search results and
/// document responses.
fn proto_document_fields_to_json(doc: &v1::Document) -> Map<String, Value> {
    doc.fields
        .iter()
        .filter(|(k, _)| k.as_str() != "_id")
        .map(|(k, v)| (k.clone(), proto_value_to_json(v)))
        .collect()
}

/// Converts a proto `Document` to a JSON value: `{"fields": {...}}`.
pub fn proto_document_to_json(doc: &v1::Document) -> Value {
    json!({ "fields": proto_document_fields_to_json(doc) })
}

// ---------------------------------------------------------------------------
// Schema conversion
// ---------------------------------------------------------------------------

/// Converts a JSON object to a proto `Schema`.
pub fn json_to_proto_schema(json: &Value) -> Result<v1::Schema, String> {
    let fields_val = json
        .get("fields")
        .ok_or_else(|| "missing \"fields\" key in schema".to_string())?;
    let fields_obj = fields_val
        .as_object()
        .ok_or_else(|| "\"fields\" must be an object".to_string())?;

    let mut fields = HashMap::new();
    for (name, opt_json) in fields_obj {
        let field_option =
            json_to_proto_field_option(opt_json).map_err(|e| format!("field \"{name}\": {e}"))?;
        fields.insert(name.clone(), field_option);
    }

    let default_fields = json
        .get("default_fields")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let analyzers = json
        .get("analyzers")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(name, def_json)| {
                    json_to_proto_analyzer_definition(def_json)
                        .map(|def| (name.clone(), def))
                        .map_err(|e| format!("analyzer \"{name}\": {e}"))
                })
                .collect::<Result<HashMap<_, _>, String>>()
        })
        .transpose()?
        .unwrap_or_default();

    let embedders = json
        .get("embedders")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .map(|(name, def_json)| {
                    json_to_proto_embedder_definition(def_json)
                        .map(|def| (name.clone(), def))
                        .map_err(|e| format!("embedder \"{name}\": {e}"))
                })
                .collect::<Result<HashMap<_, _>, String>>()
        })
        .transpose()?
        .unwrap_or_default();

    let dynamic_field_policy = json
        .get("dynamic_field_policy")
        .and_then(|v| v.as_str())
        .map(json_to_proto_dynamic_field_policy)
        .transpose()?
        .unwrap_or(v1::DynamicFieldPolicy::Unspecified) as i32;

    Ok(v1::Schema {
        fields,
        default_fields,
        analyzers,
        embedders,
        dynamic_field_policy,
        // Not user-supplied at schema-creation time: only `UpdateField`
        // populates this (Issue #1079).
        pending_reindex: Vec::new(),
    })
}

/// Converts a JSON string to a proto `DynamicFieldPolicy`.
///
/// Accepts the lowercase names `"strict"`, `"dynamic"`, and `"ignore"`.
///
/// # Arguments
///
/// * `s` - The policy name.
///
/// # Errors
///
/// Returns an error if the string does not match a known policy name.
fn json_to_proto_dynamic_field_policy(s: &str) -> Result<v1::DynamicFieldPolicy, String> {
    match s {
        "strict" => Ok(v1::DynamicFieldPolicy::Strict),
        "dynamic" => Ok(v1::DynamicFieldPolicy::Dynamic),
        "ignore" => Ok(v1::DynamicFieldPolicy::Ignore),
        other => Err(format!(
            "unknown dynamic_field_policy \"{other}\" (expected \"strict\", \"dynamic\", or \"ignore\")"
        )),
    }
}

/// Converts a proto `DynamicFieldPolicy` enum value to its JSON string name.
///
/// The proto value `UNSPECIFIED` maps to `"dynamic"` (the default). Returns
/// `None` if the value is unrecognized.
///
/// # Arguments
///
/// * `value` - The proto enum value (i32).
fn proto_dynamic_field_policy_to_json(value: i32) -> Option<&'static str> {
    match v1::DynamicFieldPolicy::try_from(value).ok()? {
        v1::DynamicFieldPolicy::Strict => Some("strict"),
        v1::DynamicFieldPolicy::Dynamic | v1::DynamicFieldPolicy::Unspecified => Some("dynamic"),
        v1::DynamicFieldPolicy::Ignore => Some("ignore"),
    }
}

/// Converts a proto `FieldChangeKind` enum value (an `UpdateField` outcome)
/// to its JSON string name. Returns `None` if the value is unrecognized.
///
/// # Arguments
///
/// * `value` - The proto enum value (i32).
pub fn proto_field_change_kind_to_json(value: i32) -> Option<&'static str> {
    match v1::FieldChangeKind::try_from(value).ok()? {
        v1::FieldChangeKind::MetadataOnly => Some("metadata_only"),
        v1::FieldChangeKind::Reindex => Some("reindex"),
        v1::FieldChangeKind::Destructive => Some("destructive"),
        v1::FieldChangeKind::Unspecified => None,
    }
}

/// Converts a proto `Schema` to a JSON value.
pub fn proto_schema_to_json(schema: &v1::Schema) -> Value {
    let fields: Map<String, Value> = schema
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), proto_field_option_to_json(v)))
        .collect();
    let mut result = json!({
        "fields": fields,
        "default_fields": schema.default_fields,
    });
    if !schema.analyzers.is_empty() {
        let analyzers: Map<String, Value> = schema
            .analyzers
            .iter()
            .map(|(k, v)| (k.clone(), proto_analyzer_definition_to_json(v)))
            .collect();
        result["analyzers"] = Value::Object(analyzers);
    }
    if !schema.embedders.is_empty() {
        let embedders: Map<String, Value> = schema
            .embedders
            .iter()
            .map(|(k, v)| (k.clone(), proto_embedder_definition_to_json(v)))
            .collect();
        result["embedders"] = Value::Object(embedders);
    }
    if let Some(name) = proto_dynamic_field_policy_to_json(schema.dynamic_field_policy) {
        result["dynamic_field_policy"] = Value::String(name.to_string());
    }
    if !schema.pending_reindex.is_empty() {
        result["pending_reindex"] = Value::Array(
            schema
                .pending_reindex
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        );
    }
    result
}

// ---------------------------------------------------------------------------
// FieldOption conversion
// ---------------------------------------------------------------------------

/// Converts a JSON object to a proto `FieldOption`.
///
/// The JSON object must contain exactly one key identifying the field type
/// (e.g. `"text"`, `"integer"`, `"hnsw"`) with its configuration as the value.
///
/// # Arguments
///
/// * `json` - The JSON value representing a single field option.
///
/// # Errors
///
/// Returns an error string if the JSON is not an object or contains an
/// unknown field option type.
/// Convert a JSON analyzer reference into the proto wire form.
///
/// Accepts either a bare string (`"standard"`) or a structured object
/// (`{"language": "japanese", "dict": "..."}`). The structured form
/// currently supports only the Japanese preset.
fn json_to_analyzer_spec(value: &Value) -> Result<v1::AnalyzerSpec, String> {
    use v1::analyzer_spec::Spec;
    let inner = if let Some(name) = value.as_str() {
        Spec::Named(name.to_string())
    } else if let Some(obj) = value.as_object() {
        let language = obj
            .get("language")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "analyzer object must include a \"language\" field".to_string())?;
        match language {
            "japanese" => {
                let dict = obj
                    .get("dict")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "japanese analyzer requires a \"dict\" path".to_string())?;
                let mode = obj
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("normal")
                    .to_string();
                let user_dict = obj
                    .get("user_dict")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Spec::Builtin(v1::BuiltinAnalyzerSpec {
                    preset: Some(v1::builtin_analyzer_spec::Preset::Japanese(
                        v1::JapaneseAnalyzerSpec {
                            mode,
                            dict: dict.to_string(),
                            user_dict,
                        },
                    )),
                })
            }
            other => return Err(format!("unsupported analyzer language: {other}")),
        }
    } else {
        return Err("analyzer must be a string or object".to_string());
    };
    Ok(v1::AnalyzerSpec { spec: Some(inner) })
}

/// Convert a proto [`v1::AnalyzerSpec`] into its JSON wire form.
///
/// Returns `None` when the proto message has no spec set or the named
/// variant carries an empty string (which means "use the default").
fn analyzer_spec_to_json(spec: &v1::AnalyzerSpec) -> Option<Value> {
    use v1::analyzer_spec::Spec;
    match spec.spec.as_ref()? {
        Spec::Named(name) if name.is_empty() => None,
        Spec::Named(name) => Some(Value::String(name.clone())),
        Spec::Builtin(builtin) => match builtin.preset.as_ref()? {
            v1::builtin_analyzer_spec::Preset::Japanese(jp) => {
                let mut obj = Map::new();
                obj.insert(
                    "language".to_string(),
                    Value::String("japanese".to_string()),
                );
                let mode = if jp.mode.is_empty() {
                    "normal".to_string()
                } else {
                    jp.mode.clone()
                };
                obj.insert("mode".to_string(), Value::String(mode));
                obj.insert("dict".to_string(), Value::String(jp.dict.clone()));
                if !jp.user_dict.is_empty() {
                    obj.insert("user_dict".to_string(), Value::String(jp.user_dict.clone()));
                }
                Some(Value::Object(obj))
            }
        },
    }
}

pub fn json_to_proto_field_option(json: &Value) -> Result<v1::FieldOption, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "field option must be an object".to_string())?;

    use v1::field_option::Option as Opt;

    let option = if let Some(v) = obj.get("text") {
        Opt::Text(v1::TextOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
            // Unset (omitted by the client) is distinct from an explicit
            // `false`: it means "use the engine's default" (#1083), unlike
            // `indexed`/`stored` above, which have no such tri-state and
            // are left as their existing `unwrap_or(false)` behavior.
            term_vectors: v.get("term_vectors").and_then(|v| v.as_bool()),
            analyzer: v.get("analyzer").map(json_to_analyzer_spec).transpose()?,
        })
    } else if let Some(v) = obj.get("integer") {
        Opt::Integer(v1::IntegerOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
            multi_valued: v
                .get("multi_valued")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    } else if let Some(v) = obj.get("float") {
        Opt::Float(v1::FloatOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
            multi_valued: v
                .get("multi_valued")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    } else if let Some(v) = obj.get("boolean") {
        Opt::Boolean(v1::BooleanOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    } else if let Some(v) = obj.get("date_time") {
        Opt::DateTime(v1::DateTimeOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    } else if let Some(v) = obj.get("geo") {
        Opt::Geo(v1::GeoOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    } else if let Some(v) = obj.get("geo3d") {
        Opt::Geo3d(v1::Geo3dOption {
            indexed: v.get("indexed").and_then(|v| v.as_bool()).unwrap_or(false),
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    } else if let Some(v) = obj.get("bytes") {
        Opt::Bytes(v1::BytesOption {
            stored: v.get("stored").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    } else if let Some(v) = obj.get("hnsw") {
        Opt::Hnsw(json_to_hnsw_option(v)?)
    } else if let Some(v) = obj.get("flat") {
        Opt::Flat(json_to_flat_option(v)?)
    } else if let Some(v) = obj.get("ivf") {
        Opt::Ivf(json_to_ivf_option(v)?)
    } else {
        return Err("unknown field option type".to_string());
    };

    Ok(v1::FieldOption {
        option: Some(option),
    })
}

fn proto_field_option_to_json(opt: &v1::FieldOption) -> Value {
    use v1::field_option::Option as Opt;
    match &opt.option {
        Some(Opt::Text(v)) => {
            let mut text_obj = json!({
                "indexed": v.indexed,
                "stored": v.stored,
            });
            // Surfaced only when explicitly set (#1083): an absent
            // `term_vectors` means "use the engine's default", same as
            // `analyzer` below, rather than a `null` standing in for `false`.
            if let Some(term_vectors) = v.term_vectors {
                text_obj["term_vectors"] = json!(term_vectors);
            }
            if let Some(spec) = v.analyzer.as_ref().and_then(analyzer_spec_to_json) {
                text_obj["analyzer"] = spec;
            }
            json!({ "text": text_obj })
        }
        Some(Opt::Integer(v)) => json!({
            "integer": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::Float(v)) => json!({
            "float": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::Boolean(v)) => json!({
            "boolean": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::DateTime(v)) => json!({
            "date_time": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::Geo(v)) => json!({
            "geo": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::Geo3d(v)) => json!({
            "geo3d": { "indexed": v.indexed, "stored": v.stored }
        }),
        Some(Opt::Bytes(v)) => json!({
            "bytes": { "stored": v.stored }
        }),
        Some(Opt::Hnsw(v)) => json!({ "hnsw": hnsw_option_to_json(v) }),
        Some(Opt::Flat(v)) => json!({ "flat": flat_option_to_json(v) }),
        Some(Opt::Ivf(v)) => json!({ "ivf": ivf_option_to_json(v) }),
        None => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Vector field option conversion
// ---------------------------------------------------------------------------

fn parse_distance_metric(s: &str) -> i32 {
    match s.to_lowercase().as_str() {
        "cosine" => v1::DistanceMetric::Cosine as i32,
        "euclidean" => v1::DistanceMetric::Euclidean as i32,
        "manhattan" => v1::DistanceMetric::Manhattan as i32,
        "dot_product" => v1::DistanceMetric::DotProduct as i32,
        "angular" => v1::DistanceMetric::Angular as i32,
        _ => v1::DistanceMetric::Cosine as i32,
    }
}

fn distance_metric_to_string(val: i32) -> &'static str {
    match v1::DistanceMetric::try_from(val) {
        Ok(v1::DistanceMetric::Cosine) => "cosine",
        Ok(v1::DistanceMetric::Euclidean) => "euclidean",
        Ok(v1::DistanceMetric::Manhattan) => "manhattan",
        Ok(v1::DistanceMetric::DotProduct) => "dot_product",
        Ok(v1::DistanceMetric::Angular) => "angular",
        Err(_) => "cosine",
    }
}

fn json_to_quantizer(json: &Value) -> Option<v1::QuantizationConfig> {
    let obj = json.as_object()?;
    let method_str = obj.get("method")?.as_str()?;
    let method = match method_str.to_lowercase().as_str() {
        "scalar_8bit" => v1::QuantizationMethod::Scalar8bit as i32,
        "product_quantization" => v1::QuantizationMethod::ProductQuantization as i32,
        "product_quantization_fastscan" => {
            v1::QuantizationMethod::ProductQuantizationFastscan as i32
        }
        _ => v1::QuantizationMethod::None as i32,
    };
    let subvector_count = obj
        .get("subvector_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    Some(v1::QuantizationConfig {
        method,
        subvector_count,
    })
}

fn quantizer_to_json(q: &v1::QuantizationConfig) -> Value {
    let method = match v1::QuantizationMethod::try_from(q.method) {
        Ok(v1::QuantizationMethod::None) => "none",
        Ok(v1::QuantizationMethod::Scalar8bit) => "scalar_8bit",
        Ok(v1::QuantizationMethod::ProductQuantization) => "product_quantization",
        Ok(v1::QuantizationMethod::ProductQuantizationFastscan) => "product_quantization_fastscan",
        Err(_) => "none",
    };
    json!({
        "method": method,
        "subvector_count": q.subvector_count,
    })
}

/// Parse the JSON `rerank_storage` field (string, e.g. `"F32"`) into the
/// proto enum value (Issue #793). Returns `None` (no sidecar) when the
/// field is absent or unrecognized.
fn json_to_rerank_storage(json: &Value) -> Option<i32> {
    match json.get("rerank_storage").and_then(|v| v.as_str()) {
        Some(s) if s.eq_ignore_ascii_case("f32") => Some(v1::RerankStorageKind::F32 as i32),
        _ => None,
    }
}

/// Render a proto `rerank_storage` enum value as a JSON string (Issue
/// #793). Returns `None` for `UNSPECIFIED`/unknown so the field is
/// omitted from the JSON object.
fn rerank_storage_to_json(value: i32) -> Option<&'static str> {
    match v1::RerankStorageKind::try_from(value) {
        Ok(v1::RerankStorageKind::F32) => Some("F32"),
        _ => None,
    }
}

fn json_to_hnsw_option(json: &Value) -> Result<v1::HnswOption, String> {
    Ok(v1::HnswOption {
        dimension: json.get("dimension").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        distance: json
            .get("distance")
            .and_then(|v| v.as_str())
            .map(parse_distance_metric)
            .unwrap_or(v1::DistanceMetric::Cosine as i32),
        m: json.get("m").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        ef_construction: json
            .get("ef_construction")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        // Unset (omitted by the client) means "use the engine's default"
        // (#1084), unlike `dimension`/`m`/`ef_construction` above, which
        // have no such tri-state and are left as their existing
        // `unwrap_or(0)` behavior.
        base_weight: json
            .get("base_weight")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        quantizer: json.get("quantizer").and_then(json_to_quantizer),
        embedder: json
            .get("embedder")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        // Issue #644: optional schema-level `ef_search` default.
        default_ef_search: json
            .get("default_ef_search")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
        // Issue #793: optional Stage-2 rerank sidecar storage.
        rerank_storage: json_to_rerank_storage(json),
        // Issue #631: optional shared PQ codebook file name. An explicitly
        // empty string would configure a codebook that can never exist;
        // treat it as unset.
        pq_codebook_path: json
            .get("pq_codebook_path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string),
    })
}

fn json_to_flat_option(json: &Value) -> Result<v1::FlatOption, String> {
    Ok(v1::FlatOption {
        dimension: json.get("dimension").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        distance: json
            .get("distance")
            .and_then(|v| v.as_str())
            .map(parse_distance_metric)
            .unwrap_or(v1::DistanceMetric::Cosine as i32),
        // Unset means "use the engine's default" (#1084).
        base_weight: json
            .get("base_weight")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        quantizer: json.get("quantizer").and_then(json_to_quantizer),
        embedder: json
            .get("embedder")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        // Issue #793: optional Stage-2 rerank sidecar storage.
        rerank_storage: json_to_rerank_storage(json),
    })
}

fn json_to_ivf_option(json: &Value) -> Result<v1::IvfOption, String> {
    Ok(v1::IvfOption {
        dimension: json.get("dimension").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        distance: json
            .get("distance")
            .and_then(|v| v.as_str())
            .map(parse_distance_metric)
            .unwrap_or(v1::DistanceMetric::Cosine as i32),
        n_clusters: json.get("n_clusters").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        n_probe: json.get("n_probe").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        // Unset means "use the engine's default" (#1084).
        base_weight: json
            .get("base_weight")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        quantizer: json.get("quantizer").and_then(json_to_quantizer),
        embedder: json
            .get("embedder")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        // Issue #793: optional Stage-2 rerank sidecar storage.
        rerank_storage: json_to_rerank_storage(json),
    })
}

fn hnsw_option_to_json(opt: &v1::HnswOption) -> Value {
    let mut obj = json!({
        "dimension": opt.dimension,
        "distance": distance_metric_to_string(opt.distance),
        "m": opt.m,
        "ef_construction": opt.ef_construction,
    });
    // Surfaced only when explicitly set (#1084): an absent `base_weight`
    // means "use the engine's default", same as `analyzer`/`term_vectors`
    // elsewhere, rather than a `null` standing in for `0.0`.
    if let Some(base_weight) = opt.base_weight {
        obj["base_weight"] = json!(base_weight);
    }
    if let Some(q) = &opt.quantizer {
        obj["quantizer"] = quantizer_to_json(q);
    }
    if !opt.embedder.is_empty() {
        obj["embedder"] = json!(opt.embedder);
    }
    if let Some(s) = opt.rerank_storage.and_then(rerank_storage_to_json) {
        obj["rerank_storage"] = json!(s);
    }
    if let Some(path) = &opt.pq_codebook_path {
        obj["pq_codebook_path"] = json!(path);
    }
    obj
}

fn flat_option_to_json(opt: &v1::FlatOption) -> Value {
    let mut obj = json!({
        "dimension": opt.dimension,
        "distance": distance_metric_to_string(opt.distance),
    });
    // Unset means "use the engine's default" (#1084).
    if let Some(base_weight) = opt.base_weight {
        obj["base_weight"] = json!(base_weight);
    }
    if let Some(q) = &opt.quantizer {
        obj["quantizer"] = quantizer_to_json(q);
    }
    if !opt.embedder.is_empty() {
        obj["embedder"] = json!(opt.embedder);
    }
    if let Some(s) = opt.rerank_storage.and_then(rerank_storage_to_json) {
        obj["rerank_storage"] = json!(s);
    }
    obj
}

fn ivf_option_to_json(opt: &v1::IvfOption) -> Value {
    let mut obj = json!({
        "dimension": opt.dimension,
        "distance": distance_metric_to_string(opt.distance),
        "n_clusters": opt.n_clusters,
        "n_probe": opt.n_probe,
    });
    // Unset means "use the engine's default" (#1084).
    if let Some(base_weight) = opt.base_weight {
        obj["base_weight"] = json!(base_weight);
    }
    if let Some(q) = &opt.quantizer {
        obj["quantizer"] = quantizer_to_json(q);
    }
    if !opt.embedder.is_empty() {
        obj["embedder"] = json!(opt.embedder);
    }
    if let Some(s) = opt.rerank_storage.and_then(rerank_storage_to_json) {
        obj["rerank_storage"] = json!(s);
    }
    obj
}

// ---------------------------------------------------------------------------
// SearchRequest conversion
// ---------------------------------------------------------------------------

/// Converts a JSON object to a proto `SearchRequest`.
pub fn json_to_proto_search_request(json: &Value) -> Result<v1::SearchRequest, String> {
    let query = json
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let query_vectors = json
        .get("query_vectors")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(json_to_query_vector).collect())
        .unwrap_or_default();

    let limit = json.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as u32;
    let offset = json.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let fusion = json.get("fusion").and_then(json_to_fusion_algorithm);
    let lexical_params = json.get("lexical_params").and_then(json_to_lexical_params);
    let vector_params = json.get("vector_params").and_then(json_to_vector_params);

    let field_boosts = json
        .get("field_boosts")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f as f32)))
                .collect()
        })
        .unwrap_or_default();

    Ok(v1::SearchRequest {
        query,
        query_vectors,
        limit,
        offset,
        fusion,
        lexical_params,
        vector_params,
        field_boosts,
    })
}

fn json_to_query_vector(json: &Value) -> Option<v1::QueryVector> {
    let vector = json
        .get("vector")?
        .as_array()?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();
    let weight = json.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let fields = json
        .get("fields")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    Some(v1::QueryVector {
        vector,
        weight,
        fields,
    })
}

fn json_to_fusion_algorithm(json: &Value) -> Option<v1::FusionAlgorithm> {
    use v1::fusion_algorithm::Algorithm;
    if let Some(rrf) = json.get("rrf") {
        let k = rrf.get("k").and_then(|v| v.as_f64()).unwrap_or(60.0);
        Some(v1::FusionAlgorithm {
            algorithm: Some(Algorithm::Rrf(v1::Rrf { k })),
        })
    } else if let Some(ws) = json.get("weighted_sum") {
        let lexical_weight = ws
            .get("lexical_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        let vector_weight = ws
            .get("vector_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        Some(v1::FusionAlgorithm {
            algorithm: Some(Algorithm::WeightedSum(v1::WeightedSum {
                lexical_weight,
                vector_weight,
            })),
        })
    } else {
        None
    }
}

fn json_to_lexical_params(json: &Value) -> Option<v1::LexicalParams> {
    let obj = json.as_object()?;
    Some(v1::LexicalParams {
        min_score: obj.get("min_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        timeout_ms: obj.get("timeout_ms").and_then(|v| v.as_u64()),
        parallel: obj
            .get("parallel")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        sort_by: obj.get("sort_by").and_then(|v| {
            let field = v.get("field")?.as_str()?.to_string();
            let order = match v.get("order").and_then(|o| o.as_str()).unwrap_or("asc") {
                "desc" => v1::SortOrder::Desc as i32,
                _ => v1::SortOrder::Asc as i32,
            };
            Some(v1::SortSpec { field, order })
        }),
    })
}

fn json_to_vector_params(json: &Value) -> Option<v1::VectorParams> {
    let obj = json.as_object()?;
    Some(v1::VectorParams {
        fields: obj
            .get("fields")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        score_mode: obj
            .get("score_mode")
            .and_then(|v| v.as_str())
            .map(|s| match s.to_lowercase().as_str() {
                "max_sim" => v1::VectorScoreMode::MaxSim as i32,
                "late_interaction" => v1::VectorScoreMode::LateInteraction as i32,
                _ => v1::VectorScoreMode::WeightedSum as i32,
            })
            .unwrap_or(v1::VectorScoreMode::WeightedSum as i32),
        overfetch: obj.get("overfetch").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        min_score: obj.get("min_score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        // Issue #481 Stage 2 (rerank). Optional in proto so older
        // gateway clients can omit the field; the engine forwards it
        // to the HNSW searcher, which honors it when the queried
        // field has rerank_storage enabled and otherwise silently
        // falls back to Stage 1 ranking.
        rerank_factor: obj
            .get("rerank_factor")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
        // Issue #644: per-query override for HNSW `ef_search`.
        ef_search: obj
            .get("ef_search")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
    })
}

/// Converts a proto `SearchResult` to a JSON value: `{"id", "score",
/// "fields"}` — the same `"fields"` key (no `"document"` wrapper) that
/// [`proto_document_to_json`] and laurus-cli's search output use, so a hit
/// can be re-submitted through `put doc` / `PUT /v1/documents/:id` as-is.
pub fn proto_search_result_to_json(result: &v1::SearchResult) -> Value {
    let mut obj = json!({
        "id": result.id,
        "score": result.score,
    });
    if let Some(doc) = &result.document {
        obj["fields"] = Value::Object(proto_document_fields_to_json(doc));
    }
    obj
}

// ---------------------------------------------------------------------------
// Analyzer definition conversion
// ---------------------------------------------------------------------------

fn json_to_proto_analyzer_definition(json: &Value) -> Result<v1::AnalyzerDefinition, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "analyzer definition must be an object".to_string())?;

    let tokenizer = obj
        .get("tokenizer")
        .ok_or("analyzer definition missing 'tokenizer'")?;
    let tokenizer = json_to_proto_component(tokenizer)?;

    let char_filters = obj
        .get("char_filters")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(json_to_proto_component)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    let token_filters = obj
        .get("token_filters")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(json_to_proto_component)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

    Ok(v1::AnalyzerDefinition {
        char_filters,
        tokenizer: Some(tokenizer),
        token_filters,
    })
}

fn json_to_proto_component(json: &Value) -> Result<v1::ComponentConfig, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "component config must be an object".to_string())?;
    let type_name = obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or("component config missing 'type'")?
        .to_string();

    let mut params = HashMap::new();
    for (k, v) in obj {
        if k == "type" {
            continue;
        }
        match v {
            Value::String(s) => {
                params.insert(k.clone(), s.clone());
            }
            Value::Bool(b) => {
                params.insert(k.clone(), b.to_string());
            }
            Value::Number(n) => {
                params.insert(k.clone(), n.to_string());
            }
            Value::Array(arr) => {
                // Encode arrays as comma-separated strings.
                let items: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                params.insert(k.clone(), items.join(","));
            }
            Value::Object(map) => {
                // Encode nested objects as key=value entries with prefixed keys.
                for (mk, mv) in map {
                    if let Some(s) = mv.as_str() {
                        params.insert(mk.clone(), s.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    Ok(v1::ComponentConfig {
        r#type: type_name,
        params,
    })
}

fn proto_analyzer_definition_to_json(def: &v1::AnalyzerDefinition) -> Value {
    let mut obj = json!({});
    if !def.char_filters.is_empty() {
        let cfs: Vec<Value> = def
            .char_filters
            .iter()
            .map(proto_component_to_json)
            .collect();
        obj["char_filters"] = json!(cfs);
    }
    if let Some(tok) = &def.tokenizer {
        obj["tokenizer"] = proto_component_to_json(tok);
    }
    if !def.token_filters.is_empty() {
        let tfs: Vec<Value> = def
            .token_filters
            .iter()
            .map(proto_component_to_json)
            .collect();
        obj["token_filters"] = json!(tfs);
    }
    obj
}

fn proto_component_to_json(comp: &v1::ComponentConfig) -> Value {
    let mut obj = json!({"type": comp.r#type});
    for (k, v) in &comp.params {
        obj[k] = json!(v);
    }
    obj
}

// ---------------------------------------------------------------------------
// Embedder definition conversion
// ---------------------------------------------------------------------------

fn json_to_proto_embedder_definition(json: &Value) -> Result<v1::EmbedderConfig, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "embedder definition must be an object".to_string())?;

    let type_name = obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or("embedder definition missing 'type'")?
        .to_string();

    let mut params = HashMap::new();
    for (k, v) in obj {
        if k == "type" {
            continue;
        }
        if let Some(s) = v.as_str() {
            params.insert(k.clone(), s.to_string());
        }
    }

    Ok(v1::EmbedderConfig {
        r#type: type_name,
        params,
    })
}

fn proto_embedder_definition_to_json(def: &v1::EmbedderConfig) -> Value {
    let mut obj = json!({"type": def.r#type});
    for (k, v) in &def.params {
        obj[k] = json!(v);
    }
    obj
}

#[cfg(test)]
mod tests {
    use laurus::{InferredValue, infer_from_json};

    use super::*;
    use crate::convert::document::data_value_to_proto;

    /// Regression coverage for the JSON -> `DataValue` -> proto `Value`
    /// pipeline (`infer_from_json` + `data_value_to_proto`). No longer used
    /// by [`json_to_proto_document`] itself, which now delegates to
    /// [`laurus::json_to_document`] + [`to_proto`], but kept here as a
    /// single-value round-trip check against `proto_value_to_json`.
    fn json_value_to_proto(json: &Value) -> Result<v1::Value, String> {
        use v1::value::Kind;
        match infer_from_json(json).map_err(|e| e.to_string())? {
            InferredValue::Skip => Ok(v1::Value {
                kind: Some(Kind::NullValue(true)),
            }),
            InferredValue::Inferred { value, .. } => Ok(data_value_to_proto(&value)),
        }
    }

    #[test]
    fn test_json_value_roundtrip_null() {
        let json = Value::Null;
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, Value::Null);
    }

    #[test]
    fn test_json_value_roundtrip_bool() {
        let json = json!(true);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!(true));
    }

    #[test]
    fn test_json_value_roundtrip_int() {
        let json = json!(42);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!(42));
    }

    #[test]
    fn test_json_value_roundtrip_float() {
        let json = json!(1.23);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!(1.23));
    }

    #[test]
    fn test_json_value_roundtrip_text() {
        let json = json!("hello");
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!("hello"));
    }

    #[test]
    fn test_json_value_integer_array_roundtrip() {
        // Integer-only arrays now roundtrip as Int64ArrayValue (multi-valued integer field).
        let json = json!([10, 20, 30]);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!([10, 20, 30]));
    }

    #[test]
    fn test_json_value_float_array_roundtrip() {
        // Mixed-precision numeric arrays roundtrip as Float64ArrayValue.
        let json = json!([1.0, 2.5, 3.0]);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, json!([1.0, 2.5, 3.0]));
    }

    #[test]
    fn test_json_value_empty_array_skips_to_null() {
        // Empty arrays are inferred as Skip → emitted as NullValue.
        let json = json!([]);
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back, Value::Null);
    }

    #[test]
    fn test_json_value_mixed_array_errors() {
        let json = json!([1, "x"]);
        let err = json_value_to_proto(&json).unwrap_err();
        assert!(err.contains("array"), "unexpected error: {err}");
    }

    #[test]
    fn test_json_value_geo_canonical_keys() {
        let json = json!({"latitude": 35.6762, "longitude": 139.6503});
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back["latitude"], json!(35.6762));
        assert_eq!(back["longitude"], json!(139.6503));
    }

    #[test]
    fn test_json_value_geo_short_aliases() {
        // The engine accepts `lat` / `lon` aliases.
        let json = json!({"lat": 35.6762, "lon": 139.6503});
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back["latitude"], json!(35.6762));
        assert_eq!(back["longitude"], json!(139.6503));
    }

    #[test]
    fn test_json_value_geo_lng_alias() {
        let json = json!({"latitude": 35.6762, "lng": 139.6503});
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back["latitude"], json!(35.6762));
        assert_eq!(back["longitude"], json!(139.6503));
    }

    #[test]
    fn test_json_value_geo_out_of_range_errors() {
        let json = json!({"latitude": 200.0, "longitude": 0.0});
        let err = json_value_to_proto(&json).unwrap_err();
        assert!(err.contains("latitude"), "unexpected error: {err}");
    }

    #[test]
    fn test_json_value_non_geo_object_errors() {
        let json = json!({"foo": "bar"});
        let err = json_value_to_proto(&json).unwrap_err();
        assert!(err.contains("geographic"), "unexpected error: {err}");
    }

    #[test]
    fn test_json_value_geo3d_roundtrip() {
        // {x, y, z} JSON inputs are inferred as a Geo3d (ECEF) point and
        // round-trip back through proto without loss.
        let json = json!({"x": -3_955_182.0, "y": 3_350_553.0, "z": 3_700_276.0});
        let proto = json_value_to_proto(&json).unwrap();
        let back = proto_value_to_json(&proto);
        assert_eq!(back["x"], json!(-3_955_182.0));
        assert_eq!(back["y"], json!(3_350_553.0));
        assert_eq!(back["z"], json!(3_700_276.0));
    }

    #[test]
    fn test_json_value_geo_geo3d_mix_errors() {
        // Mixing 2D (lat/lon) and 3D (x/y/z) markers in the same object is
        // ambiguous and must be rejected.
        let json = json!({"lat": 35.0, "x": 1.0, "y": 2.0, "z": 3.0});
        let err = json_value_to_proto(&json).unwrap_err();
        assert!(err.contains("mix"), "unexpected error: {err}");
    }

    #[test]
    fn test_json_to_proto_document() {
        let json = json!({
            "fields": {
                "title": "hello",
                "count": 42
            }
        });
        let doc = json_to_proto_document(&json).unwrap();
        assert_eq!(doc.fields.len(), 2);
    }

    /// The gateway document endpoints accept a bare `{"fields": {...}}`
    /// body (the `id` comes from the URL path) — `json_to_proto_document`
    /// must not require a `document` wrapper around it.
    #[test]
    fn test_json_to_proto_document_accepts_bare_fields_body() {
        let json = json!({"fields": {"title": "hello"}});
        let doc = json_to_proto_document(&json).unwrap();
        assert_eq!(doc.fields.len(), 1);
    }

    /// Round-trips a proto `Document` through `proto_document_to_json`,
    /// confirming the output is `{"fields": {...}}` (no `document`
    /// wrapper) and feeds straight back into `json_to_proto_document`.
    #[test]
    fn test_proto_document_to_json_round_trips_through_json_to_proto_document() {
        let original = json!({"fields": {"title": "hello", "count": 42}});
        let doc = json_to_proto_document(&original).unwrap();
        let rendered = proto_document_to_json(&doc);
        assert_eq!(rendered["fields"]["title"], json!("hello"));
        assert_eq!(rendered["fields"]["count"], json!(42));
        assert!(rendered.get("document").is_none());

        let doc2 = json_to_proto_document(&rendered).unwrap();
        assert_eq!(doc2.fields.len(), doc.fields.len());
    }

    /// A search result's document is exposed under `"fields"`, not
    /// `"document"`, matching `proto_document_to_json` and laurus-cli's
    /// search output — so a hit can be resubmitted through `put doc`.
    #[test]
    fn test_proto_search_result_to_json_uses_fields_not_document() {
        let doc = json_to_proto_document(&json!({"fields": {"title": "hello"}})).unwrap();
        let result = v1::SearchResult {
            id: "doc1".to_string(),
            score: 0.5,
            document: Some(doc),
        };
        let json = proto_search_result_to_json(&result);
        assert_eq!(json["fields"]["title"], json!("hello"));
        assert!(
            json.get("document").is_none(),
            "search result JSON must not have a \"document\" key: {json}"
        );
    }

    #[test]
    fn test_json_to_proto_document_field_error_includes_name() {
        let json = json!({
            "fields": {
                "good": "ok",
                "bad": [1, "x"]
            }
        });
        let err = json_to_proto_document(&json).unwrap_err();
        assert!(err.contains("field \"bad\":"), "unexpected error: {err}");
    }

    #[test]
    fn test_json_to_proto_schema() {
        let json = json!({
            "fields": {
                "title": { "text": { "indexed": true, "stored": true } },
                "embedding": { "hnsw": { "dimension": 768, "distance": "cosine" } }
            },
            "default_fields": ["title"]
        });
        let schema = json_to_proto_schema(&json).unwrap();
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(schema.default_fields, vec!["title"]);
    }

    /// Issue #631: `pq_codebook_path` survives the HTTP gateway's
    /// JSON -> proto -> JSON round-trip, and an absent key stays absent.
    #[test]
    fn test_hnsw_pq_codebook_path_round_trips_through_json() {
        let json = json!({
            "dimension": 32,
            "distance": "euclidean",
            "quantizer": { "product_quantization": { "subvector_count": 4 } },
            "pq_codebook_path": "embedding.pqcb"
        });
        let opt = json_to_hnsw_option(&json).unwrap();
        assert_eq!(opt.pq_codebook_path.as_deref(), Some("embedding.pqcb"));

        let back = hnsw_option_to_json(&opt);
        assert_eq!(
            back.get("pq_codebook_path").and_then(|v| v.as_str()),
            Some("embedding.pqcb")
        );

        // Absent in JSON -> absent in proto -> no key in the JSON view.
        let plain = json_to_hnsw_option(&json!({ "dimension": 32 })).unwrap();
        assert_eq!(plain.pq_codebook_path, None);
        assert!(
            hnsw_option_to_json(&plain)
                .get("pq_codebook_path")
                .is_none(),
            "an unset pq_codebook_path must not emit a key"
        );

        // An explicitly empty string names a codebook that can never exist —
        // treat it as unset.
        let empty =
            json_to_hnsw_option(&json!({ "dimension": 32, "pq_codebook_path": "" })).unwrap();
        assert_eq!(empty.pq_codebook_path, None, "empty must normalize to None");
    }

    /// Issue #1084: `base_weight` survives the HTTP gateway's
    /// JSON -> proto -> JSON round-trip when explicitly set, and an absent
    /// key stays absent (rather than materializing as `0.0`, which would
    /// silently zero out every vector score for a client that never sent
    /// it).
    #[test]
    fn test_base_weight_round_trips_through_json_and_absent_key_stays_absent() {
        let json = json!({ "dimension": 32, "base_weight": 2.5 });
        let opt = json_to_hnsw_option(&json).unwrap();
        assert_eq!(opt.base_weight, Some(2.5));

        let back = hnsw_option_to_json(&opt);
        assert_eq!(back.get("base_weight").and_then(|v| v.as_f64()), Some(2.5));

        let plain = json_to_hnsw_option(&json!({ "dimension": 32 })).unwrap();
        assert_eq!(plain.base_weight, None);
        assert!(
            hnsw_option_to_json(&plain).get("base_weight").is_none(),
            "an unset base_weight must not emit a key"
        );

        // Flat and Ivf follow the same contract.
        let flat = json_to_flat_option(&json!({ "dimension": 32 })).unwrap();
        assert_eq!(flat.base_weight, None);
        assert!(flat_option_to_json(&flat).get("base_weight").is_none());

        let ivf = json_to_ivf_option(&json!({ "dimension": 32 })).unwrap();
        assert_eq!(ivf.base_weight, None);
        assert!(ivf_option_to_json(&ivf).get("base_weight").is_none());
    }

    #[test]
    fn test_json_to_proto_search_request() {
        let json = json!({
            "query": "body:test",
            "limit": 10,
            "offset": 0,
            "field_boosts": { "title": 2.0 }
        });
        let req = json_to_proto_search_request(&json).unwrap();
        assert_eq!(req.query, "body:test");
        assert_eq!(req.limit, 10);
        assert_eq!(req.offset, 0);
        assert_eq!(*req.field_boosts.get("title").unwrap(), 2.0);
    }

    /// Every concrete `FieldChangeKind` value maps to its own distinct
    /// JSON name, and `Unspecified` (the proto3 zero-default, never sent
    /// by the server) maps to `None`.
    #[test]
    fn test_proto_field_change_kind_to_json() {
        assert_eq!(
            proto_field_change_kind_to_json(v1::FieldChangeKind::MetadataOnly as i32),
            Some("metadata_only")
        );
        assert_eq!(
            proto_field_change_kind_to_json(v1::FieldChangeKind::Reindex as i32),
            Some("reindex")
        );
        assert_eq!(
            proto_field_change_kind_to_json(v1::FieldChangeKind::Destructive as i32),
            Some("destructive")
        );
        assert_eq!(
            proto_field_change_kind_to_json(v1::FieldChangeKind::Unspecified as i32),
            None
        );
        assert_eq!(proto_field_change_kind_to_json(9999), None);
    }
}
