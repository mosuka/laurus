//! Conversion helpers between JSON (`serde_json::Value`) and proto messages.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::convert::document::to_proto;
use crate::convert::json::{
    json_optional_array, json_optional_bool, json_optional_f64, json_optional_object,
    json_optional_str, json_optional_string_array, json_optional_u32, json_optional_u64,
};
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
        Some(Kind::DatetimeValue(us)) => datetime_micros_to_json(*us),
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
        Some(Kind::GeoArrayValue(arr)) => Value::Array(
            arr.values
                .iter()
                .map(|g| json!({"latitude": g.latitude, "longitude": g.longitude}))
                .collect(),
        ),
        Some(Kind::Geo3dArrayValue(arr)) => Value::Array(
            arr.values
                .iter()
                .map(|p| json!({"x": p.x, "y": p.y, "z": p.z}))
                .collect(),
        ),
        Some(Kind::DatetimeArrayValue(arr)) => Value::Array(
            arr.values
                .iter()
                .map(|us| datetime_micros_to_json(*us))
                .collect(),
        ),
        Some(Kind::BoolArrayValue(arr)) => {
            Value::Array(arr.values.iter().map(|b| Value::Bool(*b)).collect())
        }
        Some(Kind::TextArrayValue(arr)) => Value::Array(
            arr.values
                .iter()
                .map(|s| Value::String(s.clone()))
                .collect(),
        ),
        // Same plain base64-string encoding as the scalar `BytesValue` arm
        // above, one element per array entry. The per-element MIME type is
        // not carried over the wire (see `data_value_to_proto`), so it has
        // no JSON representation here either.
        Some(Kind::BytesArrayValue(arr)) => Value::Array(
            arr.values
                .iter()
                .map(|b| Value::String(base64_encode(&base64_engine(), b)))
                .collect(),
        ),
        // Nested arrays, the same shape JSON input uses for a MultiVector
        // field (#1177).
        Some(Kind::VectorArrayValue(arr)) => {
            json!(crate::convert::document::vector_array_from_proto(arr))
        }
        None => Value::Null,
    }
}

/// Render proto Unix micro-seconds as an RFC 3339 string, or as the raw
/// number when the value is outside chrono's range. `from_timestamp_micros`
/// handles negative (pre-1970) values correctly, unlike the previous `/` +
/// `%` split, which truncated toward zero and wrapped the nanosecond part.
fn datetime_micros_to_json(us: i64) -> Value {
    match chrono::DateTime::from_timestamp_micros(us) {
        Some(dt) => Value::String(dt.to_rfc3339()),
        None => json!(us),
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
///
/// # Errors
///
/// Returns an error when `json` is not an object or has no `fields` object,
/// when a field option, analyzer or embedder is invalid, or when
/// `default_fields` is present and not `null` but not an array of strings,
/// or `analyzers` or `embedders` not an object, instead of reading it as
/// omitted or dropping an item (Issue #1524).
pub fn json_to_proto_schema(json: &Value) -> Result<v1::Schema, String> {
    let json = json
        .as_object()
        .ok_or_else(|| "schema must be an object".to_string())?;
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

    let default_fields = json_optional_string_array(json, "default_fields")?.unwrap_or_default();

    let analyzers = json_optional_object(json, "analyzers")?
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

    let embedders = json_optional_object(json, "embedders")?
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

    // Absent or `null` means the server default; a non-string value is a
    // 400 like an unknown name, not the default (Issue #1473).
    let dynamic_field_policy = match json.get("dynamic_field_policy") {
        None | Some(Value::Null) => v1::DynamicFieldPolicy::Unspecified,
        Some(Value::String(s)) => json_to_proto_dynamic_field_policy(s)?,
        Some(other) => {
            return Err(format!(
                "dynamic_field_policy must be a string, one of \"strict\", \"dynamic\", \
                 or \"ignore\", got {other}"
            ));
        }
    };

    Ok(v1::Schema {
        fields,
        default_fields,
        analyzers,
        embedders,
        dynamic_field_policy: dynamic_field_policy as i32,
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

/// Convert a JSON analyzer reference into the proto wire form.
///
/// Accepts either a bare string (`"standard"`) or a structured object
/// (`{"language": "japanese", "dict": "..."}`). The structured form
/// currently supports only the Japanese preset.
///
/// # Errors
///
/// Returns an error when `value` is neither a string nor an object, when the
/// object has no string `language` or names an unsupported one, when a
/// Japanese analyzer has no string `dict`, or when its `mode` or `user_dict`
/// is present and not `null` but not a string, instead of using the default
/// (Issue #1524).
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
                let error = |message: String| format!("analyzer: {message}");
                let dict = obj
                    .get("dict")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "japanese analyzer requires a \"dict\" path".to_string())?;
                let mode = json_optional_str(obj, "mode")
                    .map_err(error)?
                    .unwrap_or("normal")
                    .to_string();
                let user_dict = json_optional_str(obj, "user_dict")
                    .map_err(error)?
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
/// Returns an error string if the JSON is not an object, contains no known
/// field option type or more than one whose value is not `null` (Issue
/// #1524), or has a type whose value is not an object (`{"text": 1}`, or
/// `{"text": null}` with no other type, which used to read as all defaults;
/// Issue #1524), or when a key of the option is invalid: a boolean key
/// (`indexed`, `stored`, `multi_valued`, `term_vectors`, `doc_values`) that
/// is present and not `null` but not `true` or `false` (Issue #1511; see
/// [`json_optional_bool`]), or an integer, `analyzer` or vector option key
/// that cannot be read. A `null` `analyzer`, or a `null` type next to
/// another type, is the same as omitting it.
pub fn json_to_proto_field_option(json: &Value) -> Result<v1::FieldOption, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "field option must be an object".to_string())?;

    // The type is the one type key whose value is not `null`: a `null` one
    // is the same as omitting it, as for every other key. Two type keys
    // used to take the first one an `if` chain read, without an error
    // (Issue #1524).
    let mut set = FIELD_OPTION_TYPES
        .into_iter()
        .filter(|name| obj.get(*name).is_some_and(|v| !v.is_null()));
    let kind = match (set.next(), set.next()) {
        (Some(kind), None) => kind,
        (Some(first), Some(second)) => {
            return Err(format!(
                "a field option must have one type, got \"{first}\" and \"{second}\""
            ));
        }
        // A type key that is only `null` still names the type the client
        // meant, so the error names it.
        (None, _) => {
            return Err(
                match FIELD_OPTION_TYPES
                    .into_iter()
                    .find(|name| obj.contains_key(*name))
                {
                    Some(kind) => format!("{kind} option must be an object"),
                    None => "unknown field option type".to_string(),
                },
            );
        }
    };
    let value = &obj[kind];

    use v1::field_option::Option as Opt;

    let option = match kind {
        "text" => {
            let v = option_object(value, kind)?;
            Opt::Text(v1::TextOption {
                indexed: json_optional_bool(v, "indexed")?.unwrap_or(false),
                stored: json_optional_bool(v, "stored")?.unwrap_or(false),
                // Unset (omitted by the client) is distinct from an explicit
                // `false`: it means "use the engine's default" (#1083), unlike
                // `indexed`/`stored` above, which have no such tri-state and
                // are `false` when omitted.
                term_vectors: json_optional_bool(v, "term_vectors")?,
                // Same tri-state treatment for the same reason (#1047).
                doc_values: json_optional_bool(v, "doc_values")?,
                // `null` is the same as omitting it, as for every other key
                // (Issue #1524).
                analyzer: v
                    .get("analyzer")
                    .filter(|a| !a.is_null())
                    .map(json_to_analyzer_spec)
                    .transpose()?,
                multi_valued: json_optional_bool(v, "multi_valued")?.unwrap_or(false),
                // Tri-state like `term_vectors` (#1175): an omitted gap means the
                // engine default, never 0. Anything but an integer that fits
                // `u32` is rejected rather than read as omitted (#1490).
                position_increment_gap: json_optional_u32(v, "position_increment_gap")?,
            })
        }
        "integer" => {
            let v = option_object(value, kind)?;
            Opt::Integer(v1::IntegerOption {
                indexed: json_optional_bool(v, "indexed")?.unwrap_or(false),
                stored: json_optional_bool(v, "stored")?.unwrap_or(false),
                multi_valued: json_optional_bool(v, "multi_valued")?.unwrap_or(false),
                doc_values: json_optional_bool(v, "doc_values")?,
            })
        }
        "float" => {
            let v = option_object(value, kind)?;
            Opt::Float(v1::FloatOption {
                indexed: json_optional_bool(v, "indexed")?.unwrap_or(false),
                stored: json_optional_bool(v, "stored")?.unwrap_or(false),
                multi_valued: json_optional_bool(v, "multi_valued")?.unwrap_or(false),
                doc_values: json_optional_bool(v, "doc_values")?,
            })
        }
        "boolean" => {
            let v = option_object(value, kind)?;
            Opt::Boolean(v1::BooleanOption {
                indexed: json_optional_bool(v, "indexed")?.unwrap_or(false),
                stored: json_optional_bool(v, "stored")?.unwrap_or(false),
                multi_valued: json_optional_bool(v, "multi_valued")?.unwrap_or(false),
                doc_values: json_optional_bool(v, "doc_values")?,
            })
        }
        "date_time" => {
            let v = option_object(value, kind)?;
            Opt::DateTime(v1::DateTimeOption {
                indexed: json_optional_bool(v, "indexed")?.unwrap_or(false),
                stored: json_optional_bool(v, "stored")?.unwrap_or(false),
                multi_valued: json_optional_bool(v, "multi_valued")?.unwrap_or(false),
                doc_values: json_optional_bool(v, "doc_values")?,
            })
        }
        "geo" => {
            let v = option_object(value, kind)?;
            Opt::Geo(v1::GeoOption {
                indexed: json_optional_bool(v, "indexed")?.unwrap_or(false),
                stored: json_optional_bool(v, "stored")?.unwrap_or(false),
                multi_valued: json_optional_bool(v, "multi_valued")?.unwrap_or(false),
                doc_values: json_optional_bool(v, "doc_values")?,
            })
        }
        "geo3d" => {
            let v = option_object(value, kind)?;
            Opt::Geo3d(v1::Geo3dOption {
                indexed: json_optional_bool(v, "indexed")?.unwrap_or(false),
                stored: json_optional_bool(v, "stored")?.unwrap_or(false),
                multi_valued: json_optional_bool(v, "multi_valued")?.unwrap_or(false),
                doc_values: json_optional_bool(v, "doc_values")?,
            })
        }
        "bytes" => {
            let v = option_object(value, kind)?;
            Opt::Bytes(v1::BytesOption {
                stored: json_optional_bool(v, "stored")?.unwrap_or(false),
                multi_valued: json_optional_bool(v, "multi_valued")?.unwrap_or(false),
            })
        }
        "hnsw" => Opt::Hnsw(json_to_hnsw_option(value)?),
        "flat" => Opt::Flat(json_to_flat_option(value)?),
        "ivf" => Opt::Ivf(json_to_ivf_option(value)?),
        "multi_vector" => Opt::MultiVector(json_to_multi_vector_option(value)?),
        other => unreachable!("FIELD_OPTION_TYPES has no type \"{other}\""),
    };

    Ok(v1::FieldOption {
        option: Some(option),
    })
}

/// The type keys [`json_to_proto_field_option`] reads. It takes the one whose
/// value is not `null` and rejects a field option with more than one.
const FIELD_OPTION_TYPES: [&str; 12] = [
    "text",
    "integer",
    "float",
    "boolean",
    "date_time",
    "geo",
    "geo3d",
    "bytes",
    "hnsw",
    "flat",
    "ivf",
    "multi_vector",
];

/// The value of a field option's type key `kind` as an object.
///
/// # Errors
///
/// Returns an error naming `kind` when `value` is not an object, `null`
/// included, instead of reading it as an option with every key omitted
/// (Issue #1524).
fn option_object<'a>(value: &'a Value, kind: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{kind} option must be an object"))
}

fn proto_field_option_to_json(opt: &v1::FieldOption) -> Value {
    use v1::field_option::Option as Opt;
    match &opt.option {
        Some(Opt::Text(v)) => {
            // `multi_valued` is always surfaced, like on every other option
            // that carries it (#1175; see the comment on the BKD-backed arms
            // below).
            let mut text_obj = json!({
                "indexed": v.indexed,
                "stored": v.stored,
                "multi_valued": v.multi_valued,
            });
            // Surfaced only when explicitly set (#1083): an absent
            // `term_vectors` means "use the engine's default", same as
            // `analyzer` below, rather than a `null` standing in for `false`.
            if let Some(term_vectors) = v.term_vectors {
                text_obj["term_vectors"] = json!(term_vectors);
            }
            // Same tri-state treatment for the same reason (#1047).
            if let Some(doc_values) = v.doc_values {
                text_obj["doc_values"] = json!(doc_values);
            }
            // Same tri-state treatment again (#1175): `None` is the engine
            // default, and `0` is a deliberate, meaningful setting.
            if let Some(gap) = v.position_increment_gap {
                text_obj["position_increment_gap"] = json!(gap);
            }
            if let Some(spec) = v.analyzer.as_ref().and_then(analyzer_spec_to_json) {
                text_obj["analyzer"] = spec;
            }
            json!({ "text": text_obj })
        }
        // `multi_valued` is a plain (non-optional) proto bool on every option
        // that carries it (the BKD-backed Integer/Float/Geo/Geo3d/DateTime
        // and, since #1180, the term-only Boolean), so it is always surfaced,
        // like `indexed` and `stored`. Integer/Float used to omit it, so
        // `GET .../schema` could not tell a multi-valued field from a
        // single-valued one (#1174).
        Some(Opt::Integer(v)) => {
            let mut obj = json!({
                "indexed": v.indexed,
                "stored": v.stored,
                "multi_valued": v.multi_valued,
            });
            if let Some(doc_values) = v.doc_values {
                obj["doc_values"] = json!(doc_values);
            }
            json!({ "integer": obj })
        }
        Some(Opt::Float(v)) => {
            let mut obj = json!({
                "indexed": v.indexed,
                "stored": v.stored,
                "multi_valued": v.multi_valued,
            });
            if let Some(doc_values) = v.doc_values {
                obj["doc_values"] = json!(doc_values);
            }
            json!({ "float": obj })
        }
        Some(Opt::Boolean(v)) => {
            let mut obj = json!({
                "indexed": v.indexed,
                "stored": v.stored,
                "multi_valued": v.multi_valued,
            });
            if let Some(doc_values) = v.doc_values {
                obj["doc_values"] = json!(doc_values);
            }
            json!({ "boolean": obj })
        }
        Some(Opt::DateTime(v)) => {
            let mut obj = json!({
                "indexed": v.indexed,
                "stored": v.stored,
                "multi_valued": v.multi_valued,
            });
            if let Some(doc_values) = v.doc_values {
                obj["doc_values"] = json!(doc_values);
            }
            json!({ "date_time": obj })
        }
        Some(Opt::Geo(v)) => {
            let mut obj = json!({
                "indexed": v.indexed,
                "stored": v.stored,
                "multi_valued": v.multi_valued,
            });
            if let Some(doc_values) = v.doc_values {
                obj["doc_values"] = json!(doc_values);
            }
            json!({ "geo": obj })
        }
        Some(Opt::Geo3d(v)) => {
            let mut obj = json!({
                "indexed": v.indexed,
                "stored": v.stored,
                "multi_valued": v.multi_valued,
            });
            if let Some(doc_values) = v.doc_values {
                obj["doc_values"] = json!(doc_values);
            }
            json!({ "geo3d": obj })
        }
        // `multi_valued` is always surfaced, like on every other option
        // that carries it (#1176; see the comment on the BKD-backed arms
        // above). `BytesOption` has no `indexed` / `doc_values` flags.
        Some(Opt::Bytes(v)) => json!({
            "bytes": { "stored": v.stored, "multi_valued": v.multi_valued }
        }),
        Some(Opt::Hnsw(v)) => json!({ "hnsw": hnsw_option_to_json(v) }),
        Some(Opt::Flat(v)) => json!({ "flat": flat_option_to_json(v) }),
        Some(Opt::Ivf(v)) => json!({ "ivf": ivf_option_to_json(v) }),
        Some(Opt::MultiVector(v)) => {
            let mut obj = json!({
                "dimension": v.dimension,
                "distance": distance_metric_to_string(v.distance),
            });
            if !v.embedder.is_empty() {
                obj["embedder"] = json!(v.embedder);
            }
            if let Some(storage) = multi_vector_storage_to_json(v.storage) {
                obj["storage"] = json!(storage);
            }
            // Omitted at 1 (no pooling), the default.
            if let Some(factor) = v.pool_factor.filter(|&f| f != 1) {
                obj["pool_factor"] = json!(factor);
            }
            json!({ "multi_vector": obj })
        }
        None => Value::Null,
    }
}

// ---------------------------------------------------------------------------
// Vector field option conversion
// ---------------------------------------------------------------------------

/// The `distance` strings [`json_to_distance`] accepts, as quoted in its
/// error message.
const DISTANCE_METRICS: &str =
    "\"cosine\", \"euclidean\", \"manhattan\", \"dot_product\", or \"angular\"";

/// Parse a vector option's JSON `distance` string into the proto enum value.
///
/// Returns `COSINE`, the default, when `distance` is absent or `null`. The
/// string is matched case-insensitively.
///
/// # Arguments
///
/// * `obj` - The vector option object that may carry `distance`.
///
/// # Errors
///
/// Returns an error when `distance` is not a string or names a metric
/// other than `cosine`, `euclidean`, `manhattan`, `dot_product` or
/// `angular`, so a typo is a 400 instead of a cosine index (Issue #1473).
fn json_to_distance(obj: &Map<String, Value>) -> Result<i32, String> {
    let name = match obj.get("distance") {
        None | Some(Value::Null) => return Ok(v1::DistanceMetric::Cosine as i32),
        Some(Value::String(name)) => name,
        Some(other) => {
            return Err(format!(
                "distance must be a string, one of {DISTANCE_METRICS}, got {other}"
            ));
        }
    };
    let metric = match name.to_lowercase().as_str() {
        "cosine" => v1::DistanceMetric::Cosine,
        "euclidean" => v1::DistanceMetric::Euclidean,
        "manhattan" => v1::DistanceMetric::Manhattan,
        "dot_product" => v1::DistanceMetric::DotProduct,
        "angular" => v1::DistanceMetric::Angular,
        _ => {
            return Err(format!(
                "unknown distance \"{name}\" (expected {DISTANCE_METRICS})"
            ));
        }
    };
    Ok(metric as i32)
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

/// The `method` strings [`json_to_quantizer`] accepts, as quoted in its
/// error message.
const QUANTIZER_METHODS: &str =
    "\"scalar_8bit\", \"product_quantization\", \"product_quantization_fastscan\", or \"float16\"";

/// Parse a vector option's JSON `quantizer` object
/// (`{ "method": "product_quantization", "subvector_count": 4 }`) into the
/// proto message.
///
/// Returns `Ok(None)` when `quantizer` is absent or `null`, so the server
/// default (`Scalar8Bit`) applies. `method` is matched case-insensitively.
///
/// # Arguments
///
/// * `option` - The vector option object that may carry `quantizer`.
///
/// # Errors
///
/// Returns an error when `quantizer` is not an object, has no string
/// `method`, or names a method other than `scalar_8bit`,
/// `product_quantization`, `product_quantization_fastscan` or `float16`,
/// so a typo is a 400 instead of int8 storage (Issue #1459). `none` is
/// rejected too: unquantized storage was removed in Issue #481 Stage 1.
/// Also returns an error when `subvector_count` is present and not an
/// integer that fits `u32` (Issue #1490; see [`json_optional_u32`]).
fn json_to_quantizer(
    option: &Map<String, Value>,
) -> Result<Option<v1::QuantizationConfig>, String> {
    let obj = match option.get("quantizer") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(obj)) => obj,
        Some(other) => {
            return Err(format!(
                "quantizer must be an object with a \"method\" of {QUANTIZER_METHODS}, got {other}"
            ));
        }
    };
    let method_str = obj.get("method").and_then(|v| v.as_str()).ok_or_else(|| {
        format!("quantizer \"method\" must be a string, one of {QUANTIZER_METHODS}")
    })?;
    let method = match method_str.to_lowercase().as_str() {
        "scalar_8bit" => v1::QuantizationMethod::Scalar8bit,
        "product_quantization" => v1::QuantizationMethod::ProductQuantization,
        "product_quantization_fastscan" => v1::QuantizationMethod::ProductQuantizationFastscan,
        // Issue #623: IEEE 754 half precision (HNSW / Flat / IVF).
        "float16" => v1::QuantizationMethod::Float16,
        _ => {
            return Err(format!(
                "unknown quantizer method \"{method_str}\" (expected {QUANTIZER_METHODS})"
            ));
        }
    };
    // Absent or `null` is 0, which the engine rejects for the two product
    // quantization methods (Issue #1491); anything else must be an integer
    // that fits `u32` rather than read as 0 or wrapped (Issue #1490).
    let subvector_count = json_optional_u32(obj, "subvector_count")
        .map_err(|e| format!("quantizer: {e}"))?
        .unwrap_or(0);
    Ok(Some(v1::QuantizationConfig {
        method: method as i32,
        subvector_count,
    }))
}

fn quantizer_to_json(q: &v1::QuantizationConfig) -> Value {
    let method = match v1::QuantizationMethod::try_from(q.method) {
        Ok(v1::QuantizationMethod::None) => "none",
        Ok(v1::QuantizationMethod::Scalar8bit) => "scalar_8bit",
        Ok(v1::QuantizationMethod::ProductQuantization) => "product_quantization",
        Ok(v1::QuantizationMethod::ProductQuantizationFastscan) => "product_quantization_fastscan",
        Ok(v1::QuantizationMethod::Float16) => "float16",
        Err(_) => "none",
    };
    json!({
        "method": method,
        "subvector_count": q.subvector_count,
    })
}

/// Parse a vector option's JSON `rerank_storage` string (`"F32"`, matched
/// case-insensitively) into the proto enum value (Issue #793).
///
/// Returns `None` (no sidecar) when `rerank_storage` is absent or `null`.
///
/// # Arguments
///
/// * `obj` - The vector option object that may carry `rerank_storage`.
///
/// # Errors
///
/// Returns an error when `rerank_storage` is not a string or is a string
/// other than `"F32"`, so a typo is a 400 instead of no sidecar
/// (Issue #1473).
fn json_to_rerank_storage(obj: &Map<String, Value>) -> Result<Option<i32>, String> {
    match obj.get("rerank_storage") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.eq_ignore_ascii_case("f32") => {
            Ok(Some(v1::RerankStorageKind::F32 as i32))
        }
        Some(Value::String(s)) => Err(format!("unknown rerank_storage \"{s}\" (expected \"F32\")")),
        Some(other) => Err(format!(
            "rerank_storage must be the string \"F32\", got {other}"
        )),
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

/// An HNSW field option.
///
/// An omitted or `null` `m` / `ef_construction` stays unset, which the
/// server reads as the engine's default (Issue #1479), and so does
/// `default_ef_search`. An omitted `dimension` is 0, which the engine
/// rejects: it has no default.
///
/// # Errors
///
/// Returns an error when `json` is not an object, when `dimension`, `m`,
/// `ef_construction` or `default_ef_search` is present and not an integer
/// that fits `u32` (see [`json_optional_u32`]; Issues #1479, #1490), when
/// `base_weight` is present and not a number, or `embedder` or
/// `pq_codebook_path` not a string (Issue #1524), or when `distance`,
/// `quantizer` or `rerank_storage` is invalid.
fn json_to_hnsw_option(json: &Value) -> Result<v1::HnswOption, String> {
    let obj = option_object(json, "hnsw")?;
    Ok(v1::HnswOption {
        dimension: json_optional_u32(obj, "dimension")?.unwrap_or(0),
        distance: json_to_distance(obj)?,
        m: json_optional_u32(obj, "m")?,
        ef_construction: json_optional_u32(obj, "ef_construction")?,
        // Unset (omitted by the client) means "use the engine's default"
        // (#1084).
        base_weight: json_optional_f64(obj, "base_weight")?.map(|v| v as f32),
        quantizer: json_to_quantizer(obj)?,
        embedder: json_optional_str(obj, "embedder")?
            .unwrap_or("")
            .to_string(),
        // Issue #644: optional schema-level `ef_search` default.
        default_ef_search: json_optional_u32(obj, "default_ef_search")?,
        // Issue #793: optional Stage-2 rerank sidecar storage.
        rerank_storage: json_to_rerank_storage(obj)?,
        // Issue #631: optional shared PQ codebook file name. An explicitly
        // empty string would configure a codebook that can never exist;
        // treat it as unset.
        pq_codebook_path: json_optional_str(obj, "pq_codebook_path")?
            .filter(|s| !s.is_empty())
            .map(str::to_string),
    })
}

/// A Flat field option. An omitted `dimension` is 0, which the engine
/// rejects (Issue #1479).
///
/// # Errors
///
/// Returns an error when `json` is not an object, when `dimension` is
/// present and not an integer that fits `u32` (see [`json_optional_u32`]),
/// when `base_weight` is present and not a number or `embedder` not a
/// string (Issue #1524), or when `distance`, `quantizer` or
/// `rerank_storage` is invalid.
fn json_to_flat_option(json: &Value) -> Result<v1::FlatOption, String> {
    let obj = option_object(json, "flat")?;
    Ok(v1::FlatOption {
        dimension: json_optional_u32(obj, "dimension")?.unwrap_or(0),
        distance: json_to_distance(obj)?,
        // Unset means "use the engine's default" (#1084).
        base_weight: json_optional_f64(obj, "base_weight")?.map(|v| v as f32),
        quantizer: json_to_quantizer(obj)?,
        embedder: json_optional_str(obj, "embedder")?
            .unwrap_or("")
            .to_string(),
        // Issue #793: optional Stage-2 rerank sidecar storage.
        rerank_storage: json_to_rerank_storage(obj)?,
    })
}

/// A multi-vector field option (#1177); the engine rejects a zero
/// dimension, a distance other than cosine / dot product, or `"binary"`
/// storage with a distance other than cosine when the index is created.
///
/// # Errors
///
/// Returns an error when `storage` is present and not `null` but not one of
/// `"f32"` / `"f16"` / `"int8"` / `"binary"` (case-insensitive), so a typo
/// or a non-string value 400s instead of silently defaulting to f32
/// (Issues #1346, #1473), and when `distance` is invalid (see
/// [`json_to_distance`]) or `dimension` or `pool_factor` is present but not
/// a non-negative integer that fits in `u32` (Issues #1346, #1479; see
/// [`json_optional_u32`]). A `dimension` or `pool_factor` of 0 passes here
/// and is rejected by the engine. Also returns an error when `json` is not
/// an object or `embedder` is present and not a string (Issue #1524).
fn json_to_multi_vector_option(json: &Value) -> Result<v1::MultiVectorOption, String> {
    let obj = option_object(json, "multi_vector")?;
    let storage = match obj.get("storage") {
        None | Some(Value::Null) => v1::MultiVectorStorage::F32 as i32,
        Some(Value::String(s)) if s.eq_ignore_ascii_case("f32") => {
            v1::MultiVectorStorage::F32 as i32
        }
        Some(Value::String(s)) if s.eq_ignore_ascii_case("f16") => {
            v1::MultiVectorStorage::F16 as i32
        }
        Some(Value::String(s)) if s.eq_ignore_ascii_case("int8") => {
            v1::MultiVectorStorage::Int8 as i32
        }
        Some(Value::String(s)) if s.eq_ignore_ascii_case("binary") => {
            v1::MultiVectorStorage::Binary as i32
        }
        Some(Value::String(s)) => return Err(format!("unknown multi_vector storage '{s}'")),
        Some(other) => {
            return Err(format!(
                "multi_vector storage must be a string, one of \"f32\", \"f16\", \"int8\", \
                 or \"binary\", got {other}"
            ));
        }
    };
    Ok(v1::MultiVectorOption {
        dimension: json_optional_u32(obj, "dimension")?.unwrap_or(0),
        distance: json_to_distance(obj)?,
        embedder: json_optional_str(obj, "embedder")?
            .unwrap_or("")
            .to_string(),
        storage: Some(storage),
        pool_factor: json_optional_u32(obj, "pool_factor")?,
    })
}

/// Render a proto `storage` value as a lowercase JSON string (Issue
/// #1346). Returns `None` for F32/`UNSPECIFIED`/unknown, the default, so
/// the field is omitted from the JSON object. Every kind is matched by
/// name, so a new one cannot silently render as the default.
fn multi_vector_storage_to_json(value: Option<i32>) -> Option<&'static str> {
    match value.and_then(|v| v1::MultiVectorStorage::try_from(v).ok())? {
        v1::MultiVectorStorage::F16 => Some("f16"),
        v1::MultiVectorStorage::Int8 => Some("int8"),
        v1::MultiVectorStorage::Binary => Some("binary"),
        v1::MultiVectorStorage::F32 | v1::MultiVectorStorage::Unspecified => None,
    }
}

/// An IVF field option.
///
/// An omitted or `null` `n_clusters` / `n_probe` stays unset, which the
/// server reads as the engine's default (Issue #1479). An omitted
/// `dimension` is 0, which the engine rejects: it has no default.
///
/// # Errors
///
/// Returns an error when `json` is not an object, when `dimension`,
/// `n_clusters` or `n_probe` is present and not an integer that fits `u32`
/// (see [`json_optional_u32`]), when `base_weight` is present and not a
/// number or `embedder` not a string (Issue #1524), or when `distance`,
/// `quantizer` or `rerank_storage` is invalid.
fn json_to_ivf_option(json: &Value) -> Result<v1::IvfOption, String> {
    let obj = option_object(json, "ivf")?;
    Ok(v1::IvfOption {
        dimension: json_optional_u32(obj, "dimension")?.unwrap_or(0),
        distance: json_to_distance(obj)?,
        n_clusters: json_optional_u32(obj, "n_clusters")?,
        n_probe: json_optional_u32(obj, "n_probe")?,
        // Unset means "use the engine's default" (#1084).
        base_weight: json_optional_f64(obj, "base_weight")?.map(|v| v as f32),
        quantizer: json_to_quantizer(obj)?,
        embedder: json_optional_str(obj, "embedder")?
            .unwrap_or("")
            .to_string(),
        // Issue #793: optional Stage-2 rerank sidecar storage.
        rerank_storage: json_to_rerank_storage(obj)?,
    })
}

fn hnsw_option_to_json(opt: &v1::HnswOption) -> Value {
    let mut obj = json!({
        "dimension": opt.dimension,
        "distance": distance_metric_to_string(opt.distance),
    });
    // Present whenever the server sends them (Issue #1479); an unset value
    // is left out rather than rendered as `null`.
    if let Some(m) = opt.m {
        obj["m"] = json!(m);
    }
    if let Some(ef_construction) = opt.ef_construction {
        obj["ef_construction"] = json!(ef_construction);
    }
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
    // Written like `m` (Issue #1490): gRPC GetSchema keeps it, and a client
    // that copies a field from this response into `PATCH
    // /v1/schema/fields/:name` would otherwise clear it.
    if let Some(default_ef_search) = opt.default_ef_search {
        obj["default_ef_search"] = json!(default_ef_search);
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
    });
    // See `hnsw_option_to_json` (Issue #1479).
    if let Some(n_clusters) = opt.n_clusters {
        obj["n_clusters"] = json!(n_clusters);
    }
    if let Some(n_probe) = opt.n_probe {
        obj["n_probe"] = json!(n_probe);
    }
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
///
/// A key that is absent or `null` keeps its default.
///
/// # Errors
///
/// Returns a message for HTTP 400 naming the offending key when `json` is
/// not an object or a key has the wrong type, instead of reading the value
/// as omitted or dropping it (Issues #1366, #1367, #1490, #1511, #1524).
/// `query` used to be `""` when it was not a string, which turned a hybrid
/// request into a vector-only one.
pub fn json_to_proto_search_request(json: &Value) -> Result<v1::SearchRequest, String> {
    let json = json
        .as_object()
        .ok_or_else(|| "search request must be an object".to_string())?;
    let query = json_optional_str(json, "query")?.unwrap_or("").to_string();

    // A malformed `query_vectors` is rejected (400) rather than dropped,
    // shortened or defaulted (Issue #1366).
    let query_vectors = match json_optional_array(json, "query_vectors")? {
        None => Vec::new(),
        Some(entries) => entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                json_to_query_vector(entry).map_err(|e| format!("query_vectors[{i}]: {e}"))
            })
            .collect::<Result<_, _>>()?,
    };

    // Absent or `null` keeps the default; anything else must fit `u32`
    // rather than wrap around (Issue #1367).
    let page_value = |key: &str, default: u32| -> Result<u32, String> {
        json_optional_u32(json, key).map(|value| value.unwrap_or(default))
    };
    let limit = page_value("limit", 10)?;
    let offset = page_value("offset", 0)?;

    // A malformed `fusion`, `lexical_params`, `field_boosts` or `highlight`
    // is rejected (400) rather than dropped: see `fusion_algorithm_from_json`,
    // `json_to_lexical_params`, `field_boosts_from_json` and
    // `highlight_params_from_json` (Issues #1490, #1511, #1524). The MCP
    // search tools share the three `*_from_json` parsers (Issue #1523).
    let fusion = json
        .get("fusion")
        .filter(|v| !v.is_null())
        .map(crate::convert::search::fusion_algorithm_from_json)
        .transpose()?;
    let lexical_params = json_optional_object(json, "lexical_params")?
        .map(json_to_lexical_params)
        .transpose()?;
    // A malformed `vector_params` is rejected (400) rather than dropped:
    // see `json_to_vector_params`.
    let vector_params = json
        .get("vector_params")
        .filter(|v| !v.is_null())
        .map(json_to_vector_params)
        .transpose()?;

    let field_boosts = json
        .get("field_boosts")
        .filter(|v| !v.is_null())
        .map(crate::convert::search::field_boosts_from_json)
        .transpose()?
        .unwrap_or_default();

    let highlight = json
        .get("highlight")
        .map(crate::convert::search::highlight_params_from_json)
        .transpose()?
        .flatten();

    // A malformed `rescore` is rejected (400) rather than dropped: see
    // `rescore_params_from_json`.
    let rescore = json
        .get("rescore")
        .filter(|v| !v.is_null())
        .map(crate::convert::search::rescore_params_from_json)
        .transpose()?;

    // A malformed `group_by` is rejected (400) rather than dropped: see
    // `group_by_params_from_json`.
    let group_by = json
        .get("group_by")
        .filter(|v| !v.is_null())
        .map(crate::convert::search::group_by_params_from_json)
        .transpose()?;

    Ok(v1::SearchRequest {
        query,
        query_vectors,
        limit,
        offset,
        fusion,
        lexical_params,
        vector_params,
        field_boosts,
        highlight,
        rescore,
        group_by,
    })
}

/// Converts one JSON `query_vectors` entry to a proto `QueryVector`.
///
/// `vector` must be a non-empty array of numbers. `weight` (a number) and
/// `fields` (an array of strings) are optional; absent or `null` keeps the
/// defaults (weight 1.0, no fields).
///
/// # Errors
///
/// Returns a message for HTTP 400 describing the offending key.
fn json_to_query_vector(json: &Value) -> Result<v1::QueryVector, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "must be an object".to_string())?;
    let get = |key: &str| obj.get(key).filter(|v| !v.is_null());

    let vector: Vec<f32> = get("vector")
        .and_then(|v| v.as_array())
        .filter(|items| !items.is_empty())
        .and_then(|items| {
            items
                .iter()
                .map(|item| item.as_f64().map(|f| f as f32))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| "vector must be a non-empty array of numbers".to_string())?;

    let weight = json_optional_f64(obj, "weight")?.map_or(1.0, |weight| weight as f32);
    let fields = json_optional_string_array(obj, "fields")?.unwrap_or_default();

    Ok(v1::QueryVector {
        vector,
        weight,
        fields,
    })
}

/// Converts a JSON `lexical_params` object to proto `LexicalParams`.
///
/// # Errors
///
/// Returns an error when `timeout_ms` is present and not `null` but not an
/// integer that fits `u64`, instead of searching without a timeout (Issue
/// #1490; see [`json_optional_u64`]), when `parallel` is present and not
/// `null` but not `true` or `false`, instead of searching serially (Issue
/// #1511; see [`json_optional_bool`]), or when `min_score` is present and
/// not `null` but not a number, instead of `0.0`, or `sort_by` is invalid
/// (Issue #1524; see [`json_to_sort_spec`]).
fn json_to_lexical_params(obj: &Map<String, Value>) -> Result<v1::LexicalParams, String> {
    let error = |message: String| format!("lexical_params: {message}");
    Ok(v1::LexicalParams {
        min_score: json_optional_f64(obj, "min_score")
            .map_err(error)?
            .unwrap_or(0.0) as f32,
        timeout_ms: json_optional_u64(obj, "timeout_ms").map_err(error)?,
        parallel: json_optional_bool(obj, "parallel")
            .map_err(error)?
            .unwrap_or(false),
        sort_by: json_optional_object(obj, "sort_by")
            .map_err(error)?
            .map(json_to_sort_spec)
            .transpose()
            .map_err(error)?,
    })
}

/// Converts a JSON `sort_by` object (`{"field": "year", "order": "desc"}`)
/// to proto `SortSpec`.
///
/// `order` is `"asc"` or `"desc"`, matched case-insensitively like the
/// other enum strings; absent or `null` is ascending. An empty `field`
/// means a sort by score, as over gRPC.
///
/// # Errors
///
/// Returns an error when `field` is not a string, absent or `null`
/// included, or `order` is present and not `null` but not `"asc"` or
/// `"desc"`. A `sort_by` without a string `field` used to be dropped and
/// any other `order`, such as `"DESC"`, sorted ascending (Issues #1476,
/// #1524).
fn json_to_sort_spec(obj: &Map<String, Value>) -> Result<v1::SortSpec, String> {
    let error = |message: String| format!("sort_by: {message}");
    let field = json_optional_str(obj, "field")
        .map_err(error)?
        .ok_or_else(|| error("field must be a string".to_string()))?
        .to_string();
    let order = match obj.get("order") {
        None | Some(Value::Null) => v1::SortOrder::Asc,
        Some(Value::String(s)) if s.eq_ignore_ascii_case("asc") => v1::SortOrder::Asc,
        Some(Value::String(s)) if s.eq_ignore_ascii_case("desc") => v1::SortOrder::Desc,
        Some(other) => {
            return Err(error(format!(
                "order must be \"asc\" or \"desc\", got {other}"
            )));
        }
    };
    Ok(v1::SortSpec {
        field,
        order: order as i32,
    })
}

/// Converts a JSON `vector_params` object to proto `VectorParams`.
///
/// A key that is absent or `null` keeps its proto default. A key with the
/// wrong type, an unknown `score_mode` or an integer that does not fit
/// `u32` is rejected, so a value is never silently dropped or wrapped
/// around (Issue #1342). Range checks shared with gRPC happen in
/// [`crate::convert::search::from_proto`].
///
/// # Errors
///
/// Returns a message for HTTP 400 describing the offending key.
fn json_to_vector_params(json: &Value) -> Result<v1::VectorParams, String> {
    let error = |message: String| format!("vector_params: {message}");
    let obj = json
        .as_object()
        .ok_or_else(|| error("must be an object".to_string()))?;
    let fields = json_optional_string_array(obj, "fields")
        .map_err(error)?
        .unwrap_or_default();

    let score_mode = match obj.get("score_mode").filter(|v| !v.is_null()) {
        None => v1::VectorScoreMode::WeightedSum,
        Some(value) => match value.as_str().map(str::to_lowercase).as_deref() {
            Some("weighted_sum") => v1::VectorScoreMode::WeightedSum,
            Some("max_sim") => v1::VectorScoreMode::MaxSim,
            Some("late_interaction") => v1::VectorScoreMode::LateInteraction,
            _ => {
                return Err(error(format!(
                    "score_mode must be \"weighted_sum\", \"max_sim\" or \"late_interaction\", \
                     got {value}"
                )));
            }
        },
    };

    let number = |key: &str| -> Result<f32, String> {
        json_optional_f64(obj, key)
            .map(|n| n.unwrap_or(0.0) as f32)
            .map_err(error)
    };
    // Issue #481 Stage 2 (`rerank_factor`) and Issue #644 (`ef_search`).
    let uint32 = |key: &str| json_optional_u32(obj, key).map_err(error);

    Ok(v1::VectorParams {
        fields,
        score_mode: score_mode as i32,
        overfetch: number("overfetch")?,
        min_score: number("min_score")?,
        rerank_factor: uint32("rerank_factor")?,
        ef_search: uint32("ef_search")?,
    })
}

/// Converts a proto `SearchResult` to a JSON value: `{"id", "score",
/// "fields"}` — the same `"fields"` key (no `"document"` wrapper) that
/// [`proto_document_to_json`] and laurus-cli's search output use, so a hit
/// can be re-submitted through `put doc` / `PUT /v1/documents/:id` as-is.
///
/// A grouped search's result also carries `"group_key"` (when set) and
/// `"inner_hits"` (when non-empty, each converted the same way), so an
/// ungrouped result's shape is unchanged (Issue #1297).
pub fn proto_search_result_to_json(result: &v1::SearchResult) -> Value {
    let mut obj = json!({
        "id": result.id,
        "score": result.score,
    });
    if let Some(doc) = &result.document {
        obj["fields"] = Value::Object(proto_document_fields_to_json(doc));
    }
    if !result.highlights.is_empty() {
        obj["highlights"] = Value::Object(
            result
                .highlights
                .iter()
                .map(|(field, highlights)| {
                    (
                        field.clone(),
                        Value::Array(
                            highlights
                                .fragments
                                .iter()
                                .map(|fragment| Value::String(fragment.clone()))
                                .collect(),
                        ),
                    )
                })
                .collect(),
        );
    }
    if let Some(group_key) = &result.group_key {
        obj["group_key"] = proto_value_to_json(group_key);
    }
    if !result.inner_hits.is_empty() {
        obj["inner_hits"] = Value::Array(
            result
                .inner_hits
                .iter()
                .map(proto_search_result_to_json)
                .collect(),
        );
    }
    obj
}

/// Converts a proto `SearchResponse` to the JSON body of `POST /v1/search`
/// and of each entry of `POST /v1/search/batch`: `{"results": [...]}`, each
/// result converted by [`proto_search_result_to_json`].
///
/// The body carries no total: `total_hits` held only the number of returned
/// results and was removed (Issue #1455).
pub fn proto_search_response_to_json(response: &v1::SearchResponse) -> Value {
    json!({
        "results": response
            .results
            .iter()
            .map(proto_search_result_to_json)
            .collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------
// Analyzer definition conversion
// ---------------------------------------------------------------------------

/// Converts a JSON analyzer definition (`{"char_filters": [...],
/// "tokenizer": {...}, "token_filters": [...]}`) to the proto message.
///
/// # Errors
///
/// Returns an error when `json` is not an object, has no `tokenizer`, or a
/// component is invalid, or when `char_filters` or `token_filters` is
/// present and not `null` but not an array, which used to be the same as
/// omitting it (Issue #1524).
fn json_to_proto_analyzer_definition(json: &Value) -> Result<v1::AnalyzerDefinition, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "analyzer definition must be an object".to_string())?;

    let tokenizer = obj
        .get("tokenizer")
        .ok_or("analyzer definition missing 'tokenizer'")?;
    let tokenizer = json_to_proto_component(tokenizer)?;

    let components = |key: &str| -> Result<Vec<v1::ComponentConfig>, String> {
        json_optional_array(obj, key)?
            .map(|arr| arr.iter().map(json_to_proto_component).collect())
            .transpose()
            .map(Option::unwrap_or_default)
    };
    let char_filters = components("char_filters")?;
    let token_filters = components("token_filters")?;

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

    // The proto carries every parameter as a string, so scalars are written
    // in their JSON form (`32` -> "32"); only nested values are rejected.
    let mut params = HashMap::new();
    for (k, v) in obj {
        if k == "type" {
            continue;
        }
        let value = match v {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Null => continue,
            Value::Array(_) | Value::Object(_) => {
                return Err(format!(
                    "embedder parameter '{k}' must be a string, number or boolean"
                ));
            }
        };
        params.insert(k.clone(), value);
    }

    Ok(v1::EmbedderConfig {
        r#type: type_name,
        params,
    })
}

/// Embedder parameters rendered as JSON numbers rather than strings.
const NUMERIC_EMBEDDER_PARAMS: [&str; 2] = ["query_maxlen", "doc_maxlen"];

fn proto_embedder_definition_to_json(def: &v1::EmbedderConfig) -> Value {
    let mut obj = json!({"type": def.r#type});
    for (k, v) in &def.params {
        obj[k] = match v.parse::<u64>() {
            Ok(n) if NUMERIC_EMBEDDER_PARAMS.contains(&k.as_str()) => json!(n),
            _ => json!(v),
        };
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

    /// #1174: an array of geo objects is inferred as a multi-valued geo
    /// value (`GeoArrayValue`) and rendered back as an array of
    /// canonical-key objects; aliases are accepted per element.
    #[test]
    fn test_json_value_geo_array_roundtrip() {
        let json = json!([
            {"lat": 35.6762, "lon": 139.6503},
            {"latitude": -33.86, "longitude": 151.21},
        ]);
        let proto = json_value_to_proto(&json).unwrap();
        assert!(matches!(
            proto.kind,
            Some(v1::value::Kind::GeoArrayValue(_))
        ));
        let back = proto_value_to_json(&proto);
        assert_eq!(
            back,
            json!([
                {"latitude": 35.6762, "longitude": 139.6503},
                {"latitude": -33.86, "longitude": 151.21},
            ])
        );
    }

    #[test]
    fn test_json_value_geo3d_array_roundtrip() {
        let json = json!([{"x": 1.0, "y": 2.0, "z": 3.0}, {"x": -4.0, "y": 5.0, "z": -6.0}]);
        let proto = json_value_to_proto(&json).unwrap();
        assert!(matches!(
            proto.kind,
            Some(v1::value::Kind::Geo3dArrayValue(_))
        ));
        assert_eq!(proto_value_to_json(&proto), json);
    }

    #[test]
    fn test_json_value_mixed_geo_dimension_array_errors() {
        let json = json!([{"lat": 35.0, "lon": 139.0}, {"x": 1.0, "y": 2.0, "z": 3.0}]);
        let err = json_value_to_proto(&json).unwrap_err();
        assert!(err.contains("2D geo points"), "unexpected error: {err}");
    }

    /// #1184: an array of RFC 3339 strings is inferred as a multi-valued
    /// datetime (`DatetimeArrayValue`, Unix micros) and renders back as RFC
    /// 3339 strings normalized to UTC.
    #[test]
    fn test_json_value_datetime_array_roundtrip() {
        let json = json!(["2024-01-01T00:00:00Z", "2024-06-15T21:00:00.5+09:00"]);
        let proto = json_value_to_proto(&json).unwrap();
        match &proto.kind {
            Some(v1::value::Kind::DatetimeArrayValue(a)) => {
                assert_eq!(a.values, vec![1_704_067_200_000_000, 1_718_452_800_500_000]);
            }
            other => panic!("expected DatetimeArrayValue, got {other:?}"),
        }
        let back = proto_value_to_json(&proto);
        let rendered: Vec<chrono::DateTime<chrono::Utc>> = back
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                chrono::DateTime::parse_from_rfc3339(v.as_str().unwrap())
                    .unwrap()
                    .with_timezone(&chrono::Utc)
            })
            .collect();
        assert_eq!(
            rendered,
            vec![
                chrono::DateTime::from_timestamp_micros(1_704_067_200_000_000).unwrap(),
                chrono::DateTime::from_timestamp_micros(1_718_452_800_500_000).unwrap(),
            ]
        );
    }

    /// #1175: a string array that is not entirely RFC 3339 is a multi-valued
    /// text field (`TextArrayValue`) rather than an error, and renders back
    /// as the same JSON strings. An all-RFC-3339 array still infers a
    /// datetime array (see `test_json_value_datetime_array_roundtrip`).
    #[test]
    fn test_json_value_non_datetime_string_array_infers_text_array() {
        for json in [
            json!(["2024-01-01T00:00:00Z", "tomorrow"]),
            json!(["a", "b"]),
            json!(["https://example.com/a", "https://example.com/b"]),
        ] {
            let proto = json_value_to_proto(&json).unwrap();
            match &proto.kind {
                Some(v1::value::Kind::TextArrayValue(a)) => {
                    assert_eq!(json!(a.values), json, "{json}");
                }
                other => panic!("{json}: expected TextArrayValue, got {other:?}"),
            }
            assert_eq!(proto_value_to_json(&proto), json);
        }
    }

    /// #1175: the `text` option reads `multi_valued` and the tri-state
    /// `position_increment_gap`, and surfaces the gap only when set.
    #[test]
    fn test_text_option_position_increment_gap_round_trips_through_json() {
        let json = json!({ "text": {"indexed": true, "stored": true, "multi_valued": true, "position_increment_gap": 0} });
        let proto = json_to_proto_field_option(&json).unwrap();
        let back = proto_field_option_to_json(&proto);
        assert_eq!(back["text"]["multi_valued"], json!(true));
        assert_eq!(back["text"]["position_increment_gap"], json!(0));

        // Omitted on input: engine default, so nothing is surfaced.
        let json = json!({ "text": {"indexed": true, "stored": true} });
        let proto = json_to_proto_field_option(&json).unwrap();
        let back = proto_field_option_to_json(&proto);
        assert_eq!(back["text"]["multi_valued"], json!(false));
        assert!(back["text"].get("position_increment_gap").is_none());
    }

    /// Regression: pre-1970 micros used to render as the epoch (or the raw
    /// number) because the seconds/nanos split truncated toward zero.
    #[test]
    fn test_proto_datetime_pre_1970_renders_rfc3339() {
        let proto = v1::Value {
            kind: Some(v1::value::Kind::DatetimeValue(-86_400_000_001)),
        };
        assert_eq!(
            proto_value_to_json(&proto),
            json!("1969-12-30T23:59:59.999999+00:00")
        );
    }

    /// #1180: an array of JSON booleans is inferred as a multi-valued boolean
    /// (`BoolArrayValue`) and renders back as JSON booleans.
    #[test]
    fn test_json_value_bool_array_roundtrip() {
        let json = json!([true, false]);
        let proto = json_value_to_proto(&json).unwrap();
        match &proto.kind {
            Some(v1::value::Kind::BoolArrayValue(a)) => {
                assert_eq!(a.values, vec![true, false]);
            }
            other => panic!("expected BoolArrayValue, got {other:?}"),
        }
        assert_eq!(proto_value_to_json(&proto), json);
    }

    #[test]
    fn test_json_value_mixed_bool_array_errors() {
        // The boolean gate needs every element to be a boolean; a mix keeps
        // the numeric-array error.
        let err = json_value_to_proto(&json!([true, 1])).unwrap_err();
        assert!(err.contains("numeric"), "unexpected error: {err}");
    }

    /// #1176: a `BytesArrayValue` proto value (as produced for a
    /// multi-valued bytes field) renders as a JSON array of plain base64
    /// strings -- the same encoding the scalar `BytesValue` arm already
    /// uses, just one element per array entry. Unlike the geo/datetime/bool
    /// array tests above, this cannot go through `json_value_to_proto`:
    /// `infer_from_json` never infers `BytesArray` from raw JSON (Issue
    /// #1176; a multi-valued bytes field must be declared explicitly), so
    /// the proto value is built directly here.
    #[test]
    fn test_proto_bytes_array_value_renders_as_base64_array() {
        let proto = v1::Value {
            kind: Some(v1::value::Kind::BytesArrayValue(v1::BytesArrayValue {
                values: vec![b"hi".to_vec(), b"there".to_vec(), Vec::new()],
            })),
        };
        assert_eq!(proto_value_to_json(&proto), json!(["aGk=", "dGhlcmU=", ""]));
    }

    /// #1177: a packed `VectorArrayValue` renders as nested arrays, the
    /// shape JSON input uses for a MultiVector field.
    #[test]
    fn test_proto_vector_array_value_renders_as_nested_arrays() {
        let proto = v1::Value {
            kind: Some(v1::value::Kind::VectorArrayValue(v1::VectorArrayValue {
                dimension: 2,
                values: vec![0.5, 1.0, -2.0, 0.25],
            })),
        };
        assert_eq!(
            proto_value_to_json(&proto),
            json!([[0.5, 1.0], [-2.0, 0.25]])
        );
    }

    /// #1174: `multi_valued` is read from and written to the JSON schema
    /// shape for every option that carries it (the BKD-backed ones and,
    /// since #1180, Boolean). Integer/Float used to accept it on input but
    /// never surface it on output.
    #[test]
    fn test_field_option_multi_valued_round_trips_through_json() {
        for kind in [
            "integer",
            "float",
            "geo",
            "geo3d",
            "date_time",
            "boolean",
            "text",
        ] {
            let json = json!({ kind: {"indexed": true, "stored": true, "multi_valued": true} });
            let proto = json_to_proto_field_option(&json).unwrap();
            let back = proto_field_option_to_json(&proto);
            assert_eq!(back[kind]["multi_valued"], json!(true), "{kind}");
            assert_eq!(back[kind]["indexed"], json!(true), "{kind}");

            // Absent on input means single-valued, and is still surfaced.
            let json = json!({ kind: {"indexed": true, "stored": true} });
            let proto = json_to_proto_field_option(&json).unwrap();
            let back = proto_field_option_to_json(&proto);
            assert_eq!(back[kind]["multi_valued"], json!(false), "{kind}");
        }
    }

    /// #1176: `BytesOption.multi_valued` round-trips through the JSON schema
    /// shape too, even though bytes fields have no `indexed` flag, so they
    /// cannot join the loop above.
    #[test]
    fn test_bytes_option_multi_valued_round_trips_through_json() {
        let json = json!({ "bytes": {"stored": true, "multi_valued": true} });
        let proto = json_to_proto_field_option(&json).unwrap();
        let back = proto_field_option_to_json(&proto);
        assert_eq!(back["bytes"]["multi_valued"], json!(true));
        assert_eq!(back["bytes"]["stored"], json!(true));

        // Absent on input means single-valued, and is still surfaced.
        let json = json!({ "bytes": {"stored": true} });
        let proto = json_to_proto_field_option(&json).unwrap();
        let back = proto_field_option_to_json(&proto);
        assert_eq!(back["bytes"]["multi_valued"], json!(false));
    }

    /// #1177: the `multi_vector` field option round-trips its dimension and
    /// distance, and an omitted distance means cosine.
    #[test]
    fn test_multi_vector_option_round_trips_through_json() {
        let json = json!({ "multi_vector": {"dimension": 128, "distance": "dot_product"} });
        let proto = json_to_proto_field_option(&json).unwrap();
        assert!(matches!(
            proto.option,
            Some(v1::field_option::Option::MultiVector(_))
        ));
        let back = proto_field_option_to_json(&proto);
        assert_eq!(back, json);

        let json = json!({ "multi_vector": {"dimension": 96} });
        let back = proto_field_option_to_json(&json_to_proto_field_option(&json).unwrap());
        assert_eq!(back["multi_vector"]["distance"], json!("cosine"));
        assert!(back["multi_vector"].get("embedder").is_none());

        let json = json!({
            "multi_vector": {"dimension": 128, "distance": "cosine", "embedder": "colbert"}
        });
        let back = proto_field_option_to_json(&json_to_proto_field_option(&json).unwrap());
        assert_eq!(back, json);
    }

    /// #1346: `storage` round-trips for every kind, case-insensitively,
    /// is omitted from the rendered JSON when f32 (the default), and an
    /// unknown value is a 400-worthy error rather than a silent default.
    #[test]
    fn test_multi_vector_storage_round_trips_through_json() {
        for (input, expected) in [
            ("f32", "f32"),
            ("F16", "f16"),
            ("int8", "int8"),
            ("Binary", "binary"),
        ] {
            let json = json!({ "multi_vector": {"dimension": 32, "storage": input} });
            let proto = json_to_proto_field_option(&json).unwrap();
            let back = proto_field_option_to_json(&proto);
            if input.eq_ignore_ascii_case("f32") {
                assert!(
                    back["multi_vector"].get("storage").is_none(),
                    "f32 is the default and should be omitted: {back}"
                );
            } else {
                assert_eq!(back["multi_vector"]["storage"], json!(expected));
            }
        }

        let json = json!({ "multi_vector": {"dimension": 32} });
        let back = proto_field_option_to_json(&json_to_proto_field_option(&json).unwrap());
        assert!(back["multi_vector"].get("storage").is_none());

        let json = json!({ "multi_vector": {"dimension": 32, "storage": "bf16"} });
        let err = json_to_proto_field_option(&json).unwrap_err();
        assert!(err.contains("bf16"), "{err}");
    }

    /// #1346: `pool_factor` round-trips, is omitted when 1 (the default)
    /// or absent, and a value that is not a non-negative integer is a
    /// 400-worthy error.
    #[test]
    fn test_multi_vector_pool_factor_round_trips_through_json() {
        let json = json!({ "multi_vector": {"dimension": 32, "pool_factor": 3} });
        let back = proto_field_option_to_json(&json_to_proto_field_option(&json).unwrap());
        assert_eq!(back["multi_vector"]["pool_factor"], json!(3));

        for json in [
            json!({ "multi_vector": {"dimension": 32, "pool_factor": 1} }),
            json!({ "multi_vector": {"dimension": 32} }),
        ] {
            let back = proto_field_option_to_json(&json_to_proto_field_option(&json).unwrap());
            assert!(back["multi_vector"].get("pool_factor").is_none(), "{back}");
        }

        for bad in [json!(2.5), json!(-1), json!("2"), json!(5_000_000_000u64)] {
            let json = json!({ "multi_vector": {"dimension": 32, "pool_factor": bad} });
            let err = json_to_proto_field_option(&json).unwrap_err();
            assert!(err.contains("pool_factor"), "{err}");
        }
    }

    /// Issue #1479: an omitted or `null` HNSW `m` / `ef_construction` or
    /// IVF `n_clusters` / `n_probe` stays unset, which the server reads as
    /// the engine's default; an explicit value, 0 included, is passed on
    /// for the engine to check. A set value renders back, an unset one is
    /// left out.
    #[test]
    fn test_vector_option_parameters_omitted_null_and_explicit() {
        for json in [
            json!({"dimension": 4}),
            json!({"dimension": 4, "m": null, "ef_construction": null}),
        ] {
            let opt = json_to_hnsw_option(&json).unwrap();
            assert_eq!((opt.m, opt.ef_construction), (None, None), "{json}");
            let back = hnsw_option_to_json(&opt);
            assert!(back.get("m").is_none(), "{back}");
            assert!(back.get("ef_construction").is_none(), "{back}");
        }
        let opt =
            json_to_hnsw_option(&json!({"dimension": 4, "m": 0, "ef_construction": 32})).unwrap();
        assert_eq!((opt.m, opt.ef_construction), (Some(0), Some(32)));
        let back = hnsw_option_to_json(&opt);
        assert_eq!(back["m"], json!(0));
        assert_eq!(back["ef_construction"], json!(32));

        for json in [
            json!({"dimension": 4}),
            json!({"dimension": 4, "n_clusters": null, "n_probe": null}),
        ] {
            let opt = json_to_ivf_option(&json).unwrap();
            assert_eq!((opt.n_clusters, opt.n_probe), (None, None), "{json}");
            let back = ivf_option_to_json(&opt);
            assert!(back.get("n_clusters").is_none(), "{back}");
            assert!(back.get("n_probe").is_none(), "{back}");
        }
        let opt =
            json_to_ivf_option(&json!({"dimension": 4, "n_clusters": 8, "n_probe": 0})).unwrap();
        assert_eq!((opt.n_clusters, opt.n_probe), (Some(8), Some(0)));
        let back = ivf_option_to_json(&opt);
        assert_eq!(back["n_clusters"], json!(8));
        assert_eq!(back["n_probe"], json!(0));

        // `dimension` has no default: omitted, it is 0 for the engine to
        // reject.
        assert_eq!(json_to_hnsw_option(&json!({})).unwrap().dimension, 0);
    }

    /// Issue #1479: a vector option's integer parameter that is present
    /// must be an integer that fits `u32`; it used to read as 0 (or be
    /// truncated) instead.
    #[test]
    fn test_vector_option_parameters_reject_non_integers() {
        let keys = [
            ("hnsw", "dimension"),
            ("hnsw", "m"),
            ("hnsw", "ef_construction"),
            ("flat", "dimension"),
            ("ivf", "dimension"),
            ("ivf", "n_clusters"),
            ("ivf", "n_probe"),
            ("multi_vector", "dimension"),
        ];
        for (kind, key) in keys {
            for bad in [json!(16.0), json!("16"), json!(-1), json!(4_294_967_296u64)] {
                let mut option = json!({"dimension": 4});
                option[key] = bad.clone();
                let err = json_to_proto_field_option(&json!({ kind: option })).unwrap_err();
                assert!(
                    err.contains(&format!(
                        "{key} must be an integer between 0 and 4294967295"
                    )),
                    "{kind}.{key} = {bad}: {err}"
                );
            }
        }
    }

    /// Issue #1490: `text.position_increment_gap`, `hnsw.default_ef_search`
    /// and a quantizer's `subvector_count` must be integers that fit `u32`.
    /// They used to be dropped (or read as 0), or wrapped (`4294967296`
    /// became 0).
    #[test]
    fn test_field_option_integer_keys_reject_non_integers() {
        // Builds a field option with the key set to a value.
        type WithKey = fn(Value) -> Value;
        // Each key as the error names it, and its field option.
        let keys: [(&str, WithKey); 3] = [
            (
                "position_increment_gap",
                |value| json!({"text": {"indexed": true, "position_increment_gap": value}}),
            ),
            (
                "default_ef_search",
                |value| json!({"hnsw": {"dimension": 8, "default_ef_search": value}}),
            ),
            ("quantizer: subvector_count", |value| {
                json!({"hnsw": {"dimension": 8, "quantizer": {
                    "method": "product_quantization", "subvector_count": value,
                }}})
            }),
        ];
        // Every value that is not rejected with the expected message, so a
        // failure lists all of them rather than the first.
        let mut wrong = Vec::new();
        for (key, option) in keys {
            let expected = format!("{key} must be an integer between 0 and 4294967295");
            for bad in [json!("16"), json!(16.0), json!(-1), json!(4_294_967_296u64)] {
                let error = json_to_proto_field_option(&option(bad.clone())).err();
                if error.as_deref() != Some(expected.as_str()) {
                    wrong.push(format!("{key} = {bad}: {error:?}"));
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:?}");
    }

    /// Issue #1490: an omitted or `null` key keeps the value it had before
    /// the keys were made strict (unset, or 0 for `subvector_count`), and an
    /// integer is passed on as is.
    #[test]
    fn test_field_option_integer_keys_omitted_null_and_explicit() {
        let gap = |text: Value| match json_to_proto_field_option(&json!({ "text": text }))
            .unwrap()
            .option
        {
            Some(v1::field_option::Option::Text(text)) => text.position_increment_gap,
            other => panic!("expected a text option, got {other:?}"),
        };
        assert_eq!(gap(json!({"indexed": true})), None);
        assert_eq!(
            gap(json!({"indexed": true, "position_increment_gap": null})),
            None
        );
        assert_eq!(
            gap(json!({"indexed": true, "position_increment_gap": 4})),
            Some(4)
        );

        let ef = |hnsw: Value| json_to_hnsw_option(&hnsw).unwrap().default_ef_search;
        assert_eq!(ef(json!({"dimension": 8})), None);
        assert_eq!(ef(json!({"dimension": 8, "default_ef_search": null})), None);
        assert_eq!(ef(json!({"dimension": 8, "default_ef_search": 4})), Some(4));

        let count = |quantizer: Value| {
            json_to_hnsw_option(&json!({"dimension": 8, "quantizer": quantizer}))
                .unwrap()
                .quantizer
                .unwrap()
                .subvector_count
        };
        let pq = "product_quantization";
        assert_eq!(count(json!({"method": pq})), 0);
        assert_eq!(count(json!({"method": pq, "subvector_count": null})), 0);
        assert_eq!(count(json!({"method": pq, "subvector_count": 4})), 4);
    }

    /// Issue #1490: `default_ef_search` renders back like `m`, so a field
    /// copied from `GET /v1/schema` keeps it, and an unset one is left out.
    #[test]
    fn test_hnsw_default_ef_search_round_trips_through_json() {
        let opt = json_to_hnsw_option(&json!({"dimension": 8, "default_ef_search": 64})).unwrap();
        assert_eq!(opt.default_ef_search, Some(64));
        let back = hnsw_option_to_json(&opt);
        assert_eq!(back["default_ef_search"], json!(64));
        assert_eq!(json_to_hnsw_option(&back).unwrap(), opt);

        let opt = json_to_hnsw_option(&json!({"dimension": 8})).unwrap();
        let back = hnsw_option_to_json(&opt);
        assert!(back.get("default_ef_search").is_none(), "{back}");
    }

    /// Every non-vector field option with the boolean keys it reads. The
    /// vector options have none.
    const FIELD_OPTION_BOOLEAN_KEYS: [(&str, &[&str]); 8] = [
        (
            "text",
            &[
                "indexed",
                "stored",
                "multi_valued",
                "term_vectors",
                "doc_values",
            ],
        ),
        (
            "integer",
            &["indexed", "stored", "multi_valued", "doc_values"],
        ),
        (
            "float",
            &["indexed", "stored", "multi_valued", "doc_values"],
        ),
        (
            "boolean",
            &["indexed", "stored", "multi_valued", "doc_values"],
        ),
        (
            "date_time",
            &["indexed", "stored", "multi_valued", "doc_values"],
        ),
        ("geo", &["indexed", "stored", "multi_valued", "doc_values"]),
        (
            "geo3d",
            &["indexed", "stored", "multi_valued", "doc_values"],
        ),
        ("bytes", &["stored", "multi_valued"]),
    ];

    /// Issue #1511: a boolean key of a field option must be `true`, `false`
    /// or `null`. Anything else used to be read as omitted (`false`, or
    /// unset for `term_vectors` and `doc_values`), so
    /// `{"text": {"indexed": "true"}}` created a field that is not indexed.
    #[test]
    fn test_field_option_boolean_keys_reject_non_booleans() {
        // Every value that is not rejected with the expected message, so a
        // failure lists all of them rather than the first.
        let mut wrong = Vec::new();
        for (kind, keys) in FIELD_OPTION_BOOLEAN_KEYS {
            for &key in keys {
                let expected = format!("{key} must be true or false");
                for bad in [json!("true"), json!("false"), json!(1), json!(0)] {
                    let option = json!({ kind: { key: bad } });
                    let error = json_to_proto_field_option(&option).err();
                    if error.as_deref() != Some(expected.as_str()) {
                        wrong.push(format!("{kind}.{key} = {bad}: {error:?}"));
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:?}");
    }

    /// Issue #1511: `null` is the same as omitting a boolean key of a field
    /// option, and `true` and `false` are passed on as is.
    #[test]
    fn test_field_option_boolean_keys_omitted_null_and_explicit() {
        let mut wrong = Vec::new();
        for (kind, keys) in FIELD_OPTION_BOOLEAN_KEYS {
            let omitted = json_to_proto_field_option(&json!({ kind: {} })).unwrap();
            for &key in keys {
                let null = json_to_proto_field_option(&json!({ kind: { key: null } }));
                if null.as_ref() != Ok(&omitted) {
                    wrong.push(format!("{kind}.{key} = null: {null:?}"));
                }
                for value in [true, false] {
                    let option = json_to_proto_field_option(&json!({ kind: { key: value } }));
                    let back = option.as_ref().map(proto_field_option_to_json);
                    if back.as_ref().map(|back| &back[kind][key]) != Ok(&json!(value)) {
                        wrong.push(format!("{kind}.{key} = {value}: {back:?}"));
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:?}");
    }

    /// #1349: numeric and boolean embedder parameters reach the proto as
    /// strings instead of being dropped, the ColBERT lengths render back as
    /// numbers, and nested values are rejected.
    #[test]
    fn test_embedder_definition_scalar_params_round_trip() {
        let json = json!({
            "type": "candle_colbert",
            "model": "colbert-ir/colbertv2.0",
            "revision": "c1e84128",
            "query_maxlen": 32,
            "doc_maxlen": 180,
        });
        let proto = json_to_proto_embedder_definition(&json).unwrap();
        assert_eq!(proto.params["query_maxlen"], "32");
        assert_eq!(proto.params["doc_maxlen"], "180");
        assert_eq!(proto_embedder_definition_to_json(&proto), json);

        let proto = json_to_proto_embedder_definition(&json!({"type": "x", "flag": true})).unwrap();
        assert_eq!(proto.params["flag"], "true");

        let err =
            json_to_proto_embedder_definition(&json!({"type": "x", "bad": [1, 2]})).unwrap_err();
        assert!(err.contains("'bad'"), "{err}");
    }

    /// #1177: nested numeric arrays in a document become a packed
    /// `VectorArrayValue` and render back as the same nested arrays.
    #[test]
    fn test_json_nested_arrays_round_trip_as_vector_array_value() {
        let input = json!([[0.5, 1.0], [-2.0, 0.25]]);
        let proto = json_value_to_proto(&input).unwrap();
        match &proto.kind {
            Some(v1::value::Kind::VectorArrayValue(a)) => {
                assert_eq!(a.dimension, 2);
                assert_eq!(a.values, vec![0.5, 1.0, -2.0, 0.25]);
            }
            other => panic!("expected VectorArrayValue, got {other:?}"),
        }
        assert_eq!(proto_value_to_json(&proto), input);
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
            highlights: Default::default(),
            ..Default::default()
        };
        let json = proto_search_result_to_json(&result);
        assert_eq!(json["fields"]["title"], json!("hello"));
        assert!(
            json.get("document").is_none(),
            "search result JSON must not have a \"document\" key: {json}"
        );
    }

    #[test]
    fn test_proto_search_result_to_json_includes_highlights_when_present() {
        let mut highlights = HashMap::new();
        highlights.insert(
            "body".to_string(),
            v1::Highlights {
                fragments: vec!["<mark>Rust</mark> is great".to_string()],
            },
        );
        let result = v1::SearchResult {
            id: "doc1".to_string(),
            score: 0.5,
            document: None,
            highlights,
            ..Default::default()
        };

        let json = proto_search_result_to_json(&result);
        assert_eq!(
            json["highlights"]["body"],
            json!(["<mark>Rust</mark> is great"])
        );
    }

    #[test]
    fn test_proto_search_result_to_json_omits_highlights_key_when_empty() {
        let result = v1::SearchResult {
            id: "doc1".to_string(),
            score: 0.5,
            document: None,
            highlights: HashMap::new(),
            ..Default::default()
        };

        let json = proto_search_result_to_json(&result);
        assert!(
            json.get("highlights").is_none(),
            "empty highlights must not add a key: {json}"
        );
    }

    /// Issue #1297: a grouped result carries its group key and its inner
    /// hits, each in the same shape; an ungrouped one gains neither key.
    #[test]
    fn test_proto_search_result_to_json_includes_groups_when_present() {
        let key = v1::Value {
            kind: Some(v1::value::Kind::TextValue("doc".to_string())),
        };
        let hit = |id: &str| v1::SearchResult {
            id: id.to_string(),
            score: 0.5,
            group_key: Some(key.clone()),
            ..Default::default()
        };
        let leader = v1::SearchResult {
            inner_hits: vec![hit("b"), hit("c")],
            ..hit("a")
        };

        let json = proto_search_result_to_json(&leader);
        assert_eq!(json["group_key"], json!("doc"));
        assert_eq!(json["inner_hits"][0]["id"], json!("b"));
        assert_eq!(json["inner_hits"][1]["group_key"], json!("doc"));
        assert!(json["inner_hits"][0].get("inner_hits").is_none());

        let ungrouped = proto_search_result_to_json(&v1::SearchResult {
            id: "a".to_string(),
            ..Default::default()
        });
        assert!(ungrouped.get("group_key").is_none(), "{ungrouped}");
        assert!(ungrouped.get("inner_hits").is_none(), "{ungrouped}");
    }

    /// Issue #1455: the search body is `{"results": [...]}` and nothing
    /// else. `total_hits` held only the number of returned results, so no
    /// total comes back under any key.
    #[test]
    fn test_proto_search_response_to_json_has_only_results() {
        let hit = |id: &str| v1::SearchResult {
            id: id.to_string(),
            score: 0.5,
            ..Default::default()
        };
        let response = v1::SearchResponse {
            results: vec![hit("a"), hit("b")],
        };

        assert_eq!(
            proto_search_response_to_json(&response),
            json!({
                "results": [
                    proto_search_result_to_json(&hit("a")),
                    proto_search_result_to_json(&hit("b")),
                ]
            })
        );
    }

    /// Issue #1297: `group_by` reaches the proto request, `null` leaves it
    /// unset, and a malformed value is a 400 rather than dropped.
    #[test]
    fn test_json_to_proto_search_request_group_by() {
        let request = json_to_proto_search_request(&json!({
            "query": "title:rust",
            "group_by": {"field": "_id", "group_size": 3}
        }))
        .unwrap();
        assert_eq!(
            request.group_by,
            Some(v1::GroupByParams {
                field: "_id".to_string(),
                group_size: Some(3),
            })
        );

        let request =
            json_to_proto_search_request(&json!({"query": "title:rust", "group_by": null}))
                .unwrap();
        assert_eq!(request.group_by, None);

        let err = json_to_proto_search_request(&json!({
            "query": "title:rust",
            "group_by": {"group_size": 3}
        }))
        .unwrap_err();
        assert!(err.contains("group_by.field"), "{err}");
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
    /// The PQ quantizer it is meant for round-trips with it (Issue #1460).
    #[test]
    fn test_hnsw_pq_codebook_path_round_trips_through_json() {
        let json = json!({
            "dimension": 32,
            "distance": "euclidean",
            "quantizer": { "method": "product_quantization", "subvector_count": 4 },
            "pq_codebook_path": "embedding.pqcb"
        });
        let opt = json_to_hnsw_option(&json).unwrap();
        assert_eq!(opt.pq_codebook_path.as_deref(), Some("embedding.pqcb"));
        assert_eq!(
            opt.quantizer,
            Some(v1::QuantizationConfig {
                method: v1::QuantizationMethod::ProductQuantization as i32,
                subvector_count: 4,
            })
        );

        let back = hnsw_option_to_json(&opt);
        assert_eq!(
            back.get("pq_codebook_path").and_then(|v| v.as_str()),
            Some("embedding.pqcb")
        );
        assert_eq!(back["quantizer"], json["quantizer"]);

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

    /// Issue #623: a `{ "method": "float16" }` quantizer survives the HTTP
    /// gateway's JSON -> proto -> JSON round-trip for HNSW and Flat fields,
    /// mapping to `QUANTIZATION_METHOD_FLOAT16` rather than being rejected
    /// as an unrecognized method string (Issue #1459).
    #[test]
    fn test_float16_quantizer_round_trips_through_json() {
        let json = json!({ "dimension": 32, "quantizer": { "method": "float16" } });

        let hnsw = json_to_hnsw_option(&json).unwrap();
        assert_eq!(
            hnsw.quantizer.as_ref().map(|q| q.method),
            Some(v1::QuantizationMethod::Float16 as i32)
        );
        assert_eq!(
            hnsw_option_to_json(&hnsw)["quantizer"]["method"].as_str(),
            Some("float16")
        );

        let flat = json_to_flat_option(&json).unwrap();
        assert_eq!(
            flat.quantizer.as_ref().map(|q| q.method),
            Some(v1::QuantizationMethod::Float16 as i32)
        );
        assert_eq!(
            flat_option_to_json(&flat)["quantizer"]["method"].as_str(),
            Some("float16")
        );
    }

    /// Issue #1460: every `method` string the gateway accepts maps to its
    /// proto `QuantizationMethod` and back, with its `subvector_count`, for
    /// each vector field kind that carries a quantizer. The gateway does not
    /// check a method against the field kind or the build's features (see
    /// #1457, #1459), so every pair passes through unchanged. The method is
    /// matched case-insensitively and written back in lower case.
    #[test]
    fn test_quantizer_methods_round_trip_through_json() {
        let methods = [
            ("scalar_8bit", v1::QuantizationMethod::Scalar8bit, 0),
            (
                "product_quantization",
                v1::QuantizationMethod::ProductQuantization,
                8,
            ),
            (
                "product_quantization_fastscan",
                v1::QuantizationMethod::ProductQuantizationFastscan,
                8,
            ),
            ("float16", v1::QuantizationMethod::Float16, 0),
        ];
        for kind in ["hnsw", "flat", "ivf"] {
            for (name, method, subvector_count) in methods {
                let quantizer = json!({ "method": name, "subvector_count": subvector_count });
                let json = json!({ kind: { "dimension": 32, "quantizer": quantizer.clone() } });
                let proto = json_to_proto_field_option(&json).unwrap();
                let parsed = match &proto.option {
                    Some(v1::field_option::Option::Hnsw(o)) => o.quantizer.as_ref(),
                    Some(v1::field_option::Option::Flat(o)) => o.quantizer.as_ref(),
                    Some(v1::field_option::Option::Ivf(o)) => o.quantizer.as_ref(),
                    other => panic!("{kind}: unexpected option {other:?}"),
                };
                assert_eq!(
                    parsed,
                    Some(&v1::QuantizationConfig {
                        method: method as i32,
                        subvector_count,
                    }),
                    "{kind}/{name}"
                );
                assert_eq!(
                    proto_field_option_to_json(&proto)[kind]["quantizer"],
                    quantizer,
                    "{kind}/{name}"
                );

                let upper = json!({ kind: {
                    "dimension": 32,
                    "quantizer": { "method": name.to_uppercase(), "subvector_count": subvector_count }
                } });
                let back = proto_field_option_to_json(&json_to_proto_field_option(&upper).unwrap());
                assert_eq!(
                    back[kind]["quantizer"], quantizer,
                    "{kind}/{name} upper case"
                );
            }
        }
    }

    /// Issue #1459: a `quantizer` the gateway cannot map to a method is a
    /// 400 instead of `NONE`, which the server would store as int8, for each
    /// vector field kind; the schema conversion names the field. `none` is
    /// rejected too: unquantized storage was removed in Issue #481.
    #[test]
    fn test_unknown_quantizer_is_rejected() {
        let cases = [
            (json!({ "method": "pq" }), "unknown quantizer method \"pq\""),
            (
                json!({ "method": "none" }),
                "unknown quantizer method \"none\"",
            ),
            (
                json!({ "subvector_count": 4 }),
                "quantizer \"method\" must be a string",
            ),
            (
                json!({ "method": 2 }),
                "quantizer \"method\" must be a string",
            ),
            (json!("float16"), "quantizer must be an object"),
        ];
        for kind in ["hnsw", "flat", "ivf"] {
            for (quantizer, expected) in &cases {
                let option = json!({ kind: { "dimension": 32, "quantizer": quantizer } });
                let err = json_to_proto_field_option(&option).unwrap_err();
                assert!(err.starts_with(expected), "{kind}/{quantizer}: {err}");

                let schema = json!({ "fields": { "vec": option } });
                let err = json_to_proto_schema(&schema).unwrap_err();
                assert!(
                    err.starts_with(&format!("field \"vec\": {expected}")),
                    "{kind}/{quantizer}: {err}"
                );
            }
        }
    }

    /// Issue #1459: an absent or `null` quantizer is left unset, so the
    /// server default (`Scalar8Bit`) applies.
    #[test]
    fn test_absent_or_null_quantizer_is_left_unset() {
        for kind in ["hnsw", "flat", "ivf"] {
            for option in [
                json!({ kind: { "dimension": 32 } }),
                json!({ kind: { "dimension": 32, "quantizer": null } }),
            ] {
                let proto = json_to_proto_field_option(&option).unwrap();
                let quantizer = match &proto.option {
                    Some(v1::field_option::Option::Hnsw(o)) => o.quantizer.as_ref(),
                    Some(v1::field_option::Option::Flat(o)) => o.quantizer.as_ref(),
                    Some(v1::field_option::Option::Ivf(o)) => o.quantizer.as_ref(),
                    other => panic!("{kind}: unexpected option {other:?}"),
                };
                assert_eq!(quantizer, None, "{option}");
            }
        }
    }

    /// The `distance` of a converted vector field option.
    fn proto_distance_of(option: &v1::FieldOption) -> i32 {
        match &option.option {
            Some(v1::field_option::Option::Hnsw(o)) => o.distance,
            Some(v1::field_option::Option::Flat(o)) => o.distance,
            Some(v1::field_option::Option::Ivf(o)) => o.distance,
            Some(v1::field_option::Option::MultiVector(o)) => o.distance,
            other => panic!("not a vector option: {other:?}"),
        }
    }

    /// Issue #1473: `distance` is matched case-insensitively, and an absent
    /// or `null` distance is cosine, the default, on every vector field
    /// kind.
    #[test]
    fn test_distance_parses_or_defaults_to_cosine() {
        for kind in ["hnsw", "flat", "ivf", "multi_vector"] {
            for (distance, expected) in [
                (json!("cosine"), v1::DistanceMetric::Cosine),
                (json!("Euclidean"), v1::DistanceMetric::Euclidean),
                (json!("manhattan"), v1::DistanceMetric::Manhattan),
                (json!("DOT_PRODUCT"), v1::DistanceMetric::DotProduct),
                (json!("angular"), v1::DistanceMetric::Angular),
                (Value::Null, v1::DistanceMetric::Cosine),
            ] {
                let option = json!({ kind: { "dimension": 32, "distance": distance } });
                let proto = json_to_proto_field_option(&option).unwrap();
                assert_eq!(proto_distance_of(&proto), expected as i32, "{option}");
            }
            let option = json!({ kind: { "dimension": 32 } });
            let proto = json_to_proto_field_option(&option).unwrap();
            assert_eq!(
                proto_distance_of(&proto),
                v1::DistanceMetric::Cosine as i32,
                "{option}"
            );
        }
    }

    /// Issue #1473: a `distance` the gateway cannot map to a metric is a 400
    /// instead of cosine, for each vector field kind; the schema conversion
    /// names the field.
    #[test]
    fn test_unknown_distance_is_rejected() {
        let cases = [
            (json!("euclidian"), "unknown distance \"euclidian\""),
            (json!("l2"), "unknown distance \"l2\""),
            (json!(1), "distance must be a string"),
            (json!({ "metric": "cosine" }), "distance must be a string"),
        ];
        for kind in ["hnsw", "flat", "ivf", "multi_vector"] {
            for (distance, expected) in &cases {
                let option = json!({ kind: { "dimension": 32, "distance": distance } });
                let err = json_to_proto_field_option(&option).unwrap_err();
                assert!(err.starts_with(expected), "{kind}/{distance}: {err}");

                let schema = json!({ "fields": { "vec": option } });
                let err = json_to_proto_schema(&schema).unwrap_err();
                assert!(
                    err.starts_with(&format!("field \"vec\": {expected}")),
                    "{kind}/{distance}: {err}"
                );
            }
        }

        let option = json!({ "hnsw": { "dimension": 32, "distance": "euclidian" } });
        assert_eq!(
            json_to_proto_field_option(&option).unwrap_err(),
            "unknown distance \"euclidian\" (expected \"cosine\", \"euclidean\", \
             \"manhattan\", \"dot_product\", or \"angular\")"
        );
    }

    /// The `rerank_storage` of a converted HNSW, Flat or IVF field option.
    fn proto_rerank_storage_of(option: &v1::FieldOption) -> Option<i32> {
        match &option.option {
            Some(v1::field_option::Option::Hnsw(o)) => o.rerank_storage,
            Some(v1::field_option::Option::Flat(o)) => o.rerank_storage,
            Some(v1::field_option::Option::Ivf(o)) => o.rerank_storage,
            other => panic!("not an HNSW, Flat or IVF option: {other:?}"),
        }
    }

    /// Issue #1473: `rerank_storage` is matched case-insensitively, and an
    /// absent or `null` value is left unset (no sidecar).
    #[test]
    fn test_rerank_storage_parses_or_is_left_unset() {
        let f32 = Some(v1::RerankStorageKind::F32 as i32);
        for kind in ["hnsw", "flat", "ivf"] {
            for (rerank_storage, expected) in [
                (json!("F32"), f32),
                (json!("f32"), f32),
                (Value::Null, None),
            ] {
                let option = json!({ kind: { "dimension": 32, "rerank_storage": rerank_storage } });
                let proto = json_to_proto_field_option(&option).unwrap();
                assert_eq!(proto_rerank_storage_of(&proto), expected, "{option}");
            }
            let option = json!({ kind: { "dimension": 32 } });
            let proto = json_to_proto_field_option(&option).unwrap();
            assert_eq!(proto_rerank_storage_of(&proto), None, "{option}");
        }
    }

    /// Issue #1473: a `rerank_storage` other than `"F32"` is a 400 instead of
    /// no sidecar, for each vector field kind that carries it; the schema
    /// conversion names the field.
    #[test]
    fn test_unknown_rerank_storage_is_rejected() {
        let cases = [
            (
                json!("f16"),
                "unknown rerank_storage \"f16\" (expected \"F32\")",
            ),
            (
                json!("none"),
                "unknown rerank_storage \"none\" (expected \"F32\")",
            ),
            (json!(1), "rerank_storage must be the string \"F32\", got 1"),
            (
                json!(true),
                "rerank_storage must be the string \"F32\", got true",
            ),
        ];
        for kind in ["hnsw", "flat", "ivf"] {
            for (rerank_storage, expected) in &cases {
                let option = json!({ kind: { "dimension": 32, "rerank_storage": rerank_storage } });
                let err = json_to_proto_field_option(&option).unwrap_err();
                assert_eq!(&err, expected, "{kind}/{rerank_storage}");

                let schema = json!({ "fields": { "vec": option } });
                let err = json_to_proto_schema(&schema).unwrap_err();
                assert_eq!(
                    err,
                    format!("field \"vec\": {expected}"),
                    "{kind}/{rerank_storage}"
                );
            }
        }
    }

    /// Issue #1473: a multi-vector `storage` that is not a string is a 400
    /// instead of `f32`, like an unknown name; `null` keeps the default.
    #[test]
    fn test_non_string_multi_vector_storage_is_rejected() {
        for storage in [json!(3), json!(["int8"])] {
            let option = json!({ "multi_vector": { "dimension": 32, "storage": storage } });
            let err = json_to_proto_field_option(&option).unwrap_err();
            assert!(
                err.starts_with("multi_vector storage must be a string"),
                "{storage}: {err}"
            );
        }

        let option = json!({ "multi_vector": { "dimension": 32, "storage": null } });
        match json_to_proto_field_option(&option).unwrap().option {
            Some(v1::field_option::Option::MultiVector(o)) => {
                assert_eq!(o.storage, Some(v1::MultiVectorStorage::F32 as i32));
            }
            other => panic!("expected a multi-vector option, got {other:?}"),
        }
    }

    /// Issue #1473: the schema's `dynamic_field_policy` parses its three
    /// names, an absent or `null` value is left `UNSPECIFIED` (the server
    /// default, dynamic), and an unknown name or a value that is not a
    /// string is a 400 instead of the default.
    #[test]
    fn test_dynamic_field_policy_parses_or_is_rejected() {
        for (policy, expected) in [
            (json!("strict"), v1::DynamicFieldPolicy::Strict),
            (json!("dynamic"), v1::DynamicFieldPolicy::Dynamic),
            (json!("ignore"), v1::DynamicFieldPolicy::Ignore),
            (Value::Null, v1::DynamicFieldPolicy::Unspecified),
        ] {
            let schema = json!({ "fields": {}, "dynamic_field_policy": policy });
            let proto = json_to_proto_schema(&schema).unwrap();
            assert_eq!(proto.dynamic_field_policy, expected as i32, "{policy}");
        }
        let proto = json_to_proto_schema(&json!({ "fields": {} })).unwrap();
        assert_eq!(
            proto.dynamic_field_policy,
            v1::DynamicFieldPolicy::Unspecified as i32
        );

        for (policy, expected) in [
            (json!("lenient"), "unknown dynamic_field_policy \"lenient\""),
            (json!(1), "dynamic_field_policy must be a string"),
            (json!(false), "dynamic_field_policy must be a string"),
        ] {
            let schema = json!({ "fields": {}, "dynamic_field_policy": policy });
            let err = json_to_proto_schema(&schema).unwrap_err();
            assert!(err.starts_with(expected), "{policy}: {err}");
        }
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

    /// #1047: `doc_values` follows the same tri-state contract as
    /// `base_weight`/`term_vectors` above -- an absent key must round-trip
    /// as absent (not a `false`/zero-value standing in for "unset"), for
    /// both `Text` (which also carries `term_vectors`) and a non-`Text`
    /// variant.
    #[test]
    fn test_doc_values_round_trips_through_json_and_absent_key_stays_absent() {
        let with_dv = json!({ "text": { "indexed": true, "stored": true, "doc_values": false } });
        let proto = json_to_proto_field_option(&with_dv).unwrap();
        match &proto.option {
            Some(v1::field_option::Option::Text(t)) => assert_eq!(t.doc_values, Some(false)),
            other => panic!("expected Text option, got {other:?}"),
        }
        let back = proto_field_option_to_json(&proto);
        assert_eq!(
            back["text"].get("doc_values").and_then(|v| v.as_bool()),
            Some(false)
        );

        let without_dv = json!({ "text": { "indexed": true, "stored": true } });
        let proto = json_to_proto_field_option(&without_dv).unwrap();
        match &proto.option {
            Some(v1::field_option::Option::Text(t)) => assert_eq!(t.doc_values, None),
            other => panic!("expected Text option, got {other:?}"),
        }
        let back = proto_field_option_to_json(&proto);
        assert!(
            back["text"].get("doc_values").is_none(),
            "an unset doc_values must not emit a key: {back:?}"
        );

        // Non-Text variant: same contract.
        let integer_dv =
            json!({ "integer": { "indexed": true, "stored": true, "doc_values": true } });
        let proto = json_to_proto_field_option(&integer_dv).unwrap();
        match &proto.option {
            Some(v1::field_option::Option::Integer(i)) => assert_eq!(i.doc_values, Some(true)),
            other => panic!("expected Integer option, got {other:?}"),
        }
        let integer_unset = json!({ "integer": { "indexed": true, "stored": true } });
        let proto = json_to_proto_field_option(&integer_unset).unwrap();
        match &proto.option {
            Some(v1::field_option::Option::Integer(i)) => assert_eq!(i.doc_values, None),
            other => panic!("expected Integer option, got {other:?}"),
        }
        let back = proto_field_option_to_json(&proto);
        assert!(back["integer"].get("doc_values").is_none());
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
        assert!(req.highlight.is_none());
    }

    /// The array shorthand (`"highlight": ["title", "body"]`) is just a
    /// field list — no extra config.
    #[test]
    fn test_json_to_proto_search_request_highlight_array_shorthand() {
        let json = json!({
            "query": "body:test",
            "highlight": ["title", "body"],
        });
        let req = json_to_proto_search_request(&json).unwrap();
        let highlight = req.highlight.expect("highlight set");
        assert_eq!(highlight.fields, ["title", "body"]);
        assert_eq!(highlight.max_fragments, None);
    }

    /// The full object form carries `fields` plus any of the optional
    /// `HighlightConfig` knobs.
    #[test]
    fn test_json_to_proto_search_request_highlight_object_form() {
        let json = json!({
            "query": "body:test",
            "highlight": {
                "fields": ["body"],
                "max_fragments": 2,
                "fragment_size": 80,
                "tag": "em",
                "css_class": "hl",
                "require_field_match": false,
                "max_analyzed_chars": 500,
                "return_entire_field_if_no_highlight": true,
            },
        });
        let req = json_to_proto_search_request(&json).unwrap();
        let highlight = req.highlight.expect("highlight set");
        assert_eq!(highlight.fields, ["body"]);
        assert_eq!(highlight.max_fragments, Some(2));
        assert_eq!(highlight.fragment_size, Some(80));
        assert_eq!(highlight.tag.as_deref(), Some("em"));
        assert_eq!(highlight.css_class.as_deref(), Some("hl"));
        assert_eq!(highlight.require_field_match, Some(false));
        assert_eq!(highlight.max_analyzed_chars, Some(500));
        assert_eq!(highlight.return_entire_field_if_no_highlight, Some(true));
    }

    /// An object with no (or an empty) `fields` list produces no
    /// `HighlightParams` at all — `from_proto` treats that the same as the
    /// key being absent, rather than surfacing a gateway-level error for a
    /// shorthand that just requests nothing.
    #[test]
    fn test_json_to_proto_search_request_highlight_missing_fields_is_ignored() {
        for highlight_json in [json!({}), json!({"fields": []}), json!([])] {
            let json = json!({ "query": "body:test", "highlight": highlight_json });
            let req = json_to_proto_search_request(&json).unwrap();
            assert!(req.highlight.is_none(), "expected no highlight for {json}");
        }
    }

    /// #1351: `rescore` reaches the proto request, and a malformed one is
    /// an error (400) instead of being dropped.
    #[test]
    fn test_json_to_proto_search_request_rescore() {
        let json = json!({
            "query": "title:rust",
            "rescore": {"late_interaction": {"field": "tokens", "text": "rust"}}
        });
        let req = json_to_proto_search_request(&json).unwrap();
        assert!(matches!(
            req.rescore.and_then(|r| r.rescorer),
            Some(v1::rescore_params::Rescorer::LateInteraction(ref late)) if late.field == "tokens"
        ));

        let req = json_to_proto_search_request(&json!({"query": "q", "rescore": null})).unwrap();
        assert!(req.rescore.is_none());

        let err = json_to_proto_search_request(&json!({
            "query": "title:rust",
            "rescore": {"late_interaction": {"field": "tokens"}}
        }))
        .unwrap_err();
        assert!(err.contains("rescore"), "{err}");
    }

    /// Issue #1342: every `vector_params` key reaches the engine's vector
    /// options, for `query_vectors` and for a DSL query.
    #[test]
    fn test_json_to_proto_search_request_vector_params() {
        let json = json!({
            "query_vectors": [{"vector": [1.0, 0.0]}],
            "vector_params": {
                "fields": ["a"],
                "score_mode": "MAX_SIM",
                "overfetch": 3.0,
                "min_score": 0.25,
                "rerank_factor": 4,
                "ef_search": 80
            }
        });
        let req = json_to_proto_search_request(&json).unwrap();
        let vp = req.vector_params.as_ref().unwrap();
        assert_eq!(vp.fields, ["a"]);
        assert_eq!(vp.score_mode, v1::VectorScoreMode::MaxSim as i32);
        assert_eq!(vp.overfetch, 3.0);
        assert_eq!(vp.min_score, 0.25);
        assert_eq!(vp.rerank_factor, Some(4));
        assert_eq!(vp.ef_search, Some(80));

        let options = crate::convert::search::from_proto(&req)
            .unwrap()
            .vector_options;
        assert!(matches!(
            options.score_mode,
            laurus::VectorScoreMode::MaxSim
        ));
        assert_eq!(options.overfetch, 3.0);
        assert_eq!(options.min_score, 0.25);
        assert_eq!(options.rerank_factor, Some(4));
        assert_eq!(options.ef_search, Some(80));
        assert!(matches!(
            options.fields.as_deref(),
            Some([laurus::vector::store::request::FieldSelector::Exact(f)]) if f == "a"
        ));

        let dsl = json_to_proto_search_request(&json!({
            "query": "vec:\"rust\"",
            "vector_params": {"score_mode": "weighted_sum", "ef_search": 32}
        }))
        .unwrap();
        let options = crate::convert::search::from_proto(&dsl)
            .unwrap()
            .vector_options;
        assert!(matches!(
            options.score_mode,
            laurus::VectorScoreMode::WeightedSum
        ));
        assert_eq!(options.ef_search, Some(32));
    }

    /// Issue #1366: `query_vectors` entries keep their values; `null` keeps
    /// the defaults (no vectors, weight 1.0, no fields).
    #[test]
    fn test_json_to_proto_search_request_query_vectors() {
        let req = json_to_proto_search_request(&json!({
            "query_vectors": [
                {"vector": [1.0, 0.5], "weight": 2.0, "fields": ["vec"]},
                {"vector": [0.0, 1.0], "weight": null, "fields": null}
            ]
        }))
        .unwrap();
        assert_eq!(req.query_vectors.len(), 2);
        assert_eq!(req.query_vectors[0].vector, [1.0, 0.5]);
        assert_eq!(req.query_vectors[0].weight, 2.0);
        assert_eq!(req.query_vectors[0].fields, ["vec"]);
        assert_eq!(req.query_vectors[1].weight, 1.0);
        assert!(req.query_vectors[1].fields.is_empty());

        let req =
            json_to_proto_search_request(&json!({"query": "q", "query_vectors": null})).unwrap();
        assert!(req.query_vectors.is_empty());
    }

    /// Issue #1366: a malformed `query_vectors` value is rejected (400)
    /// instead of being dropped, shortened or defaulted.
    #[test]
    fn test_json_to_proto_search_request_rejects_malformed_query_vectors() {
        let vector = "query_vectors[0]: vector must be a non-empty array of numbers";
        let fields = "query_vectors[0]: fields must be an array of strings";
        let cases = [
            (json!({"vector": [1.0]}), "query_vectors must be an array"),
            (json!([1.0]), "query_vectors[0]: must be an object"),
            (json!([{}]), vector),
            (json!([{"vector": "systems programming"}]), vector),
            (json!([{"vector": []}]), vector),
            (json!([{"vector": [1.0, "x"]}]), vector),
            (
                json!([{"vector": [1.0], "weight": "2"}]),
                "query_vectors[0]: weight must be a number",
            ),
            (json!([{"vector": [1.0], "fields": "vec"}]), fields),
            (json!([{"vector": [1.0], "fields": [1]}]), fields),
        ];
        // Every value that is not rejected with the expected message, so a
        // failure lists all of them rather than the first.
        let mut wrong = Vec::new();
        for (query_vectors, expected) in cases {
            let request = json!({ "query_vectors": query_vectors });
            let error = json_to_proto_search_request(&request).err();
            if error.as_deref() != Some(expected) {
                wrong.push(format!("{query_vectors}: {error:?}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1367: `limit` / `offset` keep their defaults when absent or
    /// `null`, and a value of the wrong type or beyond `u32` is rejected
    /// (400) instead of falling back to the default or wrapping around.
    #[test]
    fn test_json_to_proto_search_request_limit_and_offset() {
        let req = json_to_proto_search_request(&json!({"query": "q"})).unwrap();
        assert_eq!((req.limit, req.offset), (10, 0));
        let req =
            json_to_proto_search_request(&json!({"query": "q", "limit": null, "offset": null}))
                .unwrap();
        assert_eq!((req.limit, req.offset), (10, 0));
        let req = json_to_proto_search_request(&json!({"query": "q", "limit": 25, "offset": 50}))
            .unwrap();
        assert_eq!((req.limit, req.offset), (25, 50));

        let cases = [
            ("limit", json!(4_294_967_296u64)),
            ("limit", json!(-1)),
            ("limit", json!("10")),
            ("offset", json!(4_294_967_296u64)),
            ("offset", json!(1.5)),
        ];
        for (key, value) in cases {
            let err =
                json_to_proto_search_request(&json!({"query": "q", key: value})).expect_err(key);
            assert!(err.contains(key), "{key} = {value}: {err}");
        }
    }

    /// Issue #1490: `lexical_params.timeout_ms` and the highlight numbers
    /// must be integers in the range of their proto type. They used to be
    /// dropped (no timeout, or the highlight default), and `max_fragments`
    /// and `fragment_size` wrapped (`4294967298` became 2).
    #[test]
    fn test_json_to_proto_search_request_rejects_malformed_integer_keys() {
        // Above `u64::MAX`: `serde_json` parses it as a float, as it does
        // in a request body.
        let above_u64: Value = serde_json::from_str("18446744073709551616").unwrap();
        let uint32 = [json!("2"), json!(2.0), json!(-1), json!(4_294_967_296u64)];
        let uint64 = [json!("500"), json!(500.0), json!(-1), above_u64];
        let cases = [
            ("lexical_params", "timeout_ms", u64::MAX, &uint64),
            ("highlight", "max_fragments", u32::MAX.into(), &uint32),
            ("highlight", "fragment_size", u32::MAX.into(), &uint32),
            ("highlight", "max_analyzed_chars", u64::MAX, &uint64),
        ];
        // Every value that is not rejected with the expected message, so a
        // failure lists all of them rather than the first.
        let mut wrong = Vec::new();
        for (object, key, max, bad_values) in cases {
            let expected = format!("{object}: {key} must be an integer between 0 and {max}");
            for bad in bad_values {
                let mut params = match object {
                    "highlight" => json!({"fields": ["body"]}),
                    _ => json!({}),
                };
                params[key] = bad.clone();
                let request = json!({"query": "body:rust", object: params});
                let error = json_to_proto_search_request(&request).err();
                if error.as_deref() != Some(expected.as_str()) {
                    wrong.push(format!("{object}.{key} = {bad}: {error:?}"));
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:?}");

        // Checked even when the highlight is dropped for lack of fields.
        let err = json_to_proto_search_request(&json!({
            "query": "body:rust",
            "highlight": {"fields": [], "max_fragments": "2"}
        }))
        .expect_err("highlight without fields");
        assert!(err.contains("highlight: max_fragments"), "{err}");
    }

    /// Issue #1490: an omitted or `null` `lexical_params.timeout_ms` or
    /// highlight number stays unset, and an integer in range is passed on,
    /// up to `u64` for `timeout_ms` and `max_analyzed_chars`.
    #[test]
    fn test_json_to_proto_search_request_integer_keys_omitted_null_and_explicit() {
        for (lexical_params, highlight) in [
            (json!({}), json!({"fields": ["body"]})),
            (
                json!({"timeout_ms": null}),
                json!({
                    "fields": ["body"],
                    "max_fragments": null,
                    "fragment_size": null,
                    "max_analyzed_chars": null
                }),
            ),
        ] {
            let req = json_to_proto_search_request(&json!({
                "query": "body:rust",
                "lexical_params": lexical_params,
                "highlight": highlight
            }))
            .unwrap();
            assert_eq!(req.lexical_params.unwrap().timeout_ms, None);
            let highlight = req.highlight.unwrap();
            assert_eq!(
                (
                    highlight.max_fragments,
                    highlight.fragment_size,
                    highlight.max_analyzed_chars
                ),
                (None, None, None)
            );
        }

        let req = json_to_proto_search_request(&json!({
            "query": "body:rust",
            "lexical_params": {"timeout_ms": 4_294_967_296u64},
            "highlight": {
                "fields": ["body"],
                "max_fragments": 2,
                "fragment_size": 80,
                "max_analyzed_chars": 4_294_967_296u64
            }
        }))
        .unwrap();
        assert_eq!(req.lexical_params.unwrap().timeout_ms, Some(4_294_967_296));
        let highlight = req.highlight.unwrap();
        assert_eq!(
            (
                highlight.max_fragments,
                highlight.fragment_size,
                highlight.max_analyzed_chars
            ),
            (Some(2), Some(80), Some(4_294_967_296))
        );
    }

    /// Issue #1511: `lexical_params.parallel` and the highlight booleans
    /// must be `true`, `false` or `null`. Anything else used to be read as
    /// omitted (a serial search, or the highlight default).
    #[test]
    fn test_json_to_proto_search_request_rejects_malformed_boolean_keys() {
        // Each key with the object that holds it, the highlight both with
        // and without fields: a highlight without fields is dropped, but its
        // keys are checked first.
        let cases = [
            ("lexical_params", "parallel", json!({})),
            (
                "highlight",
                "require_field_match",
                json!({"fields": ["body"]}),
            ),
            ("highlight", "require_field_match", json!({"fields": []})),
            (
                "highlight",
                "return_entire_field_if_no_highlight",
                json!({"fields": ["body"]}),
            ),
            (
                "highlight",
                "return_entire_field_if_no_highlight",
                json!({"fields": []}),
            ),
        ];
        // Every value that is not rejected with the expected message, so a
        // failure lists all of them rather than the first.
        let mut wrong = Vec::new();
        for (object, key, params) in cases {
            let expected = format!("{object}: {key} must be true or false");
            for bad in [json!("true"), json!("false"), json!(1), json!(0)] {
                let mut params = params.clone();
                params[key] = bad.clone();
                let request = json!({"query": "body:rust", object: params});
                let error = json_to_proto_search_request(&request).err();
                if error.as_deref() != Some(expected.as_str()) {
                    wrong.push(format!("{object}.{key} = {bad} in {params}: {error:?}"));
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:?}");
    }

    /// Issue #1511: an omitted or `null` `lexical_params.parallel` is
    /// `false` and an omitted or `null` highlight boolean stays unset, and
    /// `true` and `false` are passed on as is.
    #[test]
    fn test_json_to_proto_search_request_boolean_keys_omitted_null_and_explicit() {
        // `parallel`, `require_field_match` and
        // `return_entire_field_if_no_highlight` of the converted request.
        let flags = |lexical_params: Value, highlight: Value| {
            let req = json_to_proto_search_request(&json!({
                "query": "body:rust",
                "lexical_params": lexical_params,
                "highlight": highlight
            }))
            .unwrap();
            let highlight = req.highlight.unwrap();
            (
                req.lexical_params.unwrap().parallel,
                highlight.require_field_match,
                highlight.return_entire_field_if_no_highlight,
            )
        };
        assert_eq!(
            flags(json!({}), json!({"fields": ["body"]})),
            (false, None, None)
        );
        assert_eq!(
            flags(
                json!({"parallel": null}),
                json!({
                    "fields": ["body"],
                    "require_field_match": null,
                    "return_entire_field_if_no_highlight": null
                })
            ),
            (false, None, None)
        );
        for value in [true, false] {
            assert_eq!(
                flags(
                    json!({"parallel": value}),
                    json!({
                        "fields": ["body"],
                        "require_field_match": value,
                        "return_entire_field_if_no_highlight": value
                    })
                ),
                (value, Some(value), Some(value))
            );
        }
    }

    /// Issue #1342: a malformed `vector_params` value is rejected (400)
    /// instead of being dropped or wrapped around.
    #[test]
    fn test_json_to_proto_search_request_rejects_malformed_vector_params() {
        let cases = [
            ("not an object", json!("max_sim")),
            ("fields is not an array", json!({"fields": "a"})),
            ("a field is not a string", json!({"fields": ["a", 1]})),
            ("unknown score_mode", json!({"score_mode": "best"})),
            ("score_mode is not a string", json!({"score_mode": 1})),
            ("overfetch is not a number", json!({"overfetch": "2"})),
            ("min_score is not a number", json!({"min_score": true})),
            (
                "rerank_factor does not fit u32",
                json!({"rerank_factor": 4_294_967_296u64}),
            ),
            (
                "ef_search does not fit u32",
                json!({"ef_search": 4_294_967_296u64}),
            ),
            ("negative ef_search", json!({"ef_search": -1})),
            ("fractional rerank_factor", json!({"rerank_factor": 1.5})),
        ];
        for (case, vector_params) in cases {
            let err = json_to_proto_search_request(&json!({
                "query_vectors": [{"vector": [1.0, 0.0]}],
                "vector_params": vector_params
            }))
            .expect_err(case);
            assert!(err.contains("vector_params"), "{case}: {err}");
        }
    }

    /// Issue #1524: every other key of a search request rejects a value of
    /// the wrong type with a message naming the key. Such a value used to be
    /// read as omitted or dropped: `"order": "DESC"` sorted ascending,
    /// `{"title": "2"}` boosted nothing and `"k": "1"` was k 60.
    #[test]
    fn test_json_to_proto_search_request_rejects_values_of_the_wrong_type() {
        let shape = "highlight must be an array of strings or an object";
        let exactly_one = "fusion must have exactly one of \"rrf\" and \"weighted_sum\"";
        // A whole request, or a key with its value added to a request with
        // a `query`, and the expected message.
        let bodies = [
            (json!(5), "search request must be an object"),
            (json!([]), "search request must be an object"),
            (json!("body:rust"), "search request must be an object"),
            (json!(null), "search request must be an object"),
            (json!({"query": 5}), "query must be a string"),
            (json!({"query": ["body:rust"]}), "query must be a string"),
            (
                json!({"query": 5, "query_vectors": [{"vector": [1.0, 0.0]}]}),
                "query must be a string",
            ),
        ];
        let keys = [
            (
                "lexical_params",
                json!("x"),
                "lexical_params must be an object",
            ),
            (
                "lexical_params",
                json!([{"min_score": 100}]),
                "lexical_params must be an object",
            ),
            (
                "lexical_params",
                json!({"min_score": "100"}),
                "lexical_params: min_score must be a number",
            ),
            (
                "lexical_params",
                json!({"sort_by": 5}),
                "lexical_params: sort_by must be an object",
            ),
            (
                "lexical_params",
                json!({"sort_by": "year"}),
                "lexical_params: sort_by must be an object",
            ),
            (
                "lexical_params",
                json!({"sort_by": {"order": "desc"}}),
                "lexical_params: sort_by: field must be a string",
            ),
            (
                "lexical_params",
                json!({"sort_by": {"field": null}}),
                "lexical_params: sort_by: field must be a string",
            ),
            (
                "lexical_params",
                json!({"sort_by": {"field": 1, "order": "desc"}}),
                "lexical_params: sort_by: field must be a string",
            ),
            (
                "lexical_params",
                json!({"sort_by": {"field": "year", "order": "descending"}}),
                "lexical_params: sort_by: order must be \"asc\" or \"desc\", got \"descending\"",
            ),
            (
                "lexical_params",
                json!({"sort_by": {"field": "year", "order": 1}}),
                "lexical_params: sort_by: order must be \"asc\" or \"desc\", got 1",
            ),
            (
                "field_boosts",
                json!([["title", 2]]),
                "field_boosts must be an object",
            ),
            (
                "field_boosts",
                json!("title"),
                "field_boosts must be an object",
            ),
            (
                "field_boosts",
                json!({"title": "2"}),
                "field_boosts: \"title\" must be a number",
            ),
            (
                "field_boosts",
                json!({"title": [2]}),
                "field_boosts: \"title\" must be a number",
            ),
            ("fusion", json!("weighted_sum"), "fusion must be an object"),
            ("fusion", json!({}), exactly_one),
            ("fusion", json!({"rrf": null}), exactly_one),
            (
                "fusion",
                json!({"rrf": {"k": 1}, "weighted_sum": {}}),
                exactly_one,
            ),
            ("fusion", json!({"rrf": 5}), "fusion: rrf must be an object"),
            (
                "fusion",
                json!({"weighted_sum": "x"}),
                "fusion: weighted_sum must be an object",
            ),
            (
                "fusion",
                json!({"rrf": {"k": "1"}}),
                "fusion: rrf: k must be a number",
            ),
            (
                "fusion",
                json!({"weighted_sum": {"lexical_weight": "1"}}),
                "fusion: weighted_sum: lexical_weight must be a number",
            ),
            (
                "fusion",
                json!({"weighted_sum": {"vector_weight": "0"}}),
                "fusion: weighted_sum: vector_weight must be a number",
            ),
            ("highlight", json!("body"), shape),
            ("highlight", json!(5), shape),
            ("highlight", json!(true), shape),
            ("highlight", json!(["body", 1]), shape),
            ("highlight", json!([null]), shape),
            (
                "highlight",
                json!({"fields": "body"}),
                "highlight: fields must be an array of strings",
            ),
            (
                "highlight",
                json!({"fields": ["body", 1]}),
                "highlight: fields must be an array of strings",
            ),
            (
                "highlight",
                json!({"fields": ["body"], "tag": 1}),
                "highlight: tag must be a string",
            ),
            // Checked even when the field list is empty.
            (
                "highlight",
                json!({"fields": [], "tag": 1}),
                "highlight: tag must be a string",
            ),
            (
                "highlight",
                json!({"fields": ["body"], "css_class": 1}),
                "highlight: css_class must be a string",
            ),
        ];
        let requests =
            bodies
                .into_iter()
                .chain(keys.into_iter().map(|(key, value, expected)| {
                    (json!({"query": "body:rust", key: value}), expected)
                }));
        // Every request that is not rejected with the expected message, so
        // a failure lists all of them rather than the first.
        let mut wrong = Vec::new();
        for (request, expected) in requests {
            let error = json_to_proto_search_request(&request).err();
            if error.as_deref() != Some(expected) {
                wrong.push(format!("{request}: {error:?}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1524: an omitted or `null` key of a search request keeps its
    /// default, and a value of the right type is passed on.
    #[test]
    fn test_json_to_proto_search_request_value_types_omitted_null_and_explicit() {
        let request = |extra: Value| {
            let mut request = json!({"query": "body:rust"});
            request
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            json_to_proto_search_request(&request).unwrap()
        };

        let omitted = request(json!({}));
        let null = request(json!({
            "lexical_params": null,
            "field_boosts": null,
            "fusion": null,
            "highlight": null
        }));
        assert_eq!(null, omitted);
        let null_query = json_to_proto_search_request(&json!({"query": null})).unwrap();
        assert_eq!(null_query.query, "");

        let lexical = |params: Value| request(json!({"lexical_params": params})).lexical_params;
        let defaults = lexical(json!({})).unwrap();
        assert_eq!(defaults.min_score, 0.0);
        assert_eq!(defaults.sort_by, None);
        assert_eq!(
            lexical(json!({"min_score": null, "sort_by": null})),
            Some(defaults)
        );
        assert_eq!(lexical(json!({"min_score": 0.5})).unwrap().min_score, 0.5);

        let order = |order: Value| {
            let sort_by = json!({"sort_by": {"field": "year", "order": order}});
            lexical(sort_by).unwrap().sort_by.unwrap().order
        };
        let asc = v1::SortOrder::Asc as i32;
        let desc = v1::SortOrder::Desc as i32;
        let orders = [
            (json!(null), asc),
            (json!("asc"), asc),
            (json!("ASC"), asc),
            (json!("desc"), desc),
            (json!("DESC"), desc),
            (json!("Desc"), desc),
        ];
        let wrong: Vec<_> = orders
            .into_iter()
            .filter(|(value, expected)| order(value.clone()) != *expected)
            .collect();
        assert!(wrong.is_empty(), "{wrong:?}");
        let sort_by = lexical(json!({"sort_by": {"field": "year"}}))
            .unwrap()
            .sort_by;
        assert_eq!(
            sort_by,
            Some(v1::SortSpec {
                field: "year".to_string(),
                order: asc
            })
        );
        // An empty field is a sort by score, as over gRPC.
        let sort_by = lexical(json!({"sort_by": {"field": ""}})).unwrap().sort_by;
        assert_eq!(sort_by.unwrap().field, "");

        let boosts = request(json!({"field_boosts": {"title": 2, "body": null}})).field_boosts;
        assert_eq!(boosts, HashMap::from([("title".to_string(), 2.0)]));

        use v1::fusion_algorithm::Algorithm;
        let fusion = |fusion: Value| {
            request(json!({"fusion": fusion}))
                .fusion
                .unwrap()
                .algorithm
                .unwrap()
        };
        assert_eq!(
            fusion(json!({"rrf": {}})),
            Algorithm::Rrf(v1::Rrf { k: 60.0 })
        );
        assert_eq!(
            fusion(json!({"rrf": {"k": null}})),
            Algorithm::Rrf(v1::Rrf { k: 60.0 })
        );
        assert_eq!(
            fusion(json!({"rrf": {"k": 1}})),
            Algorithm::Rrf(v1::Rrf { k: 1.0 })
        );
        assert_eq!(
            fusion(json!({"rrf": null, "weighted_sum": {"vector_weight": null}})),
            Algorithm::WeightedSum(v1::WeightedSum {
                lexical_weight: 0.5,
                vector_weight: 0.5
            })
        );
        assert_eq!(
            fusion(json!({"weighted_sum": {"lexical_weight": 1, "vector_weight": 0}})),
            Algorithm::WeightedSum(v1::WeightedSum {
                lexical_weight: 1.0,
                vector_weight: 0.0
            })
        );

        let highlight = request(json!({
            "highlight": {"fields": ["body"], "tag": null, "css_class": null}
        }))
        .highlight
        .unwrap();
        assert_eq!((highlight.tag, highlight.css_class), (None, None));
        let highlight = request(json!({
            "highlight": {"fields": ["body"], "tag": "em", "css_class": "hl"}
        }))
        .highlight
        .unwrap();
        assert_eq!(
            (highlight.tag.as_deref(), highlight.css_class.as_deref()),
            (Some("em"), Some("hl"))
        );
        let highlight = request(json!({"highlight": {"fields": null}})).highlight;
        assert_eq!(highlight, None);
    }

    /// Issue #1524: a field option whose type has a value that is not an
    /// object, or with two types, is rejected, and so is a vector or
    /// Japanese analyzer key of the wrong type. `{"text": 1}` used to read
    /// as a `text` option with every key omitted.
    #[test]
    fn test_field_option_rejects_values_of_the_wrong_type() {
        let mut cases = Vec::new();
        for kind in FIELD_OPTION_TYPES {
            for bad in [json!(1), json!(null), json!("x"), json!([])] {
                cases.push((
                    json!({ kind: bad }),
                    format!("{kind} option must be an object"),
                ));
            }
        }
        cases.push((
            json!({"text": {}, "integer": {}}),
            "a field option must have one type, got \"text\" and \"integer\"".to_string(),
        ));
        cases.push((
            json!({"flat": {"dimension": 2}, "hnsw": {"dimension": 2}}),
            "a field option must have one type, got \"hnsw\" and \"flat\"".to_string(),
        ));
        // Only `null` types: the error names the first.
        cases.push((
            json!({"integer": null, "text": null}),
            "text option must be an object".to_string(),
        ));
        for kind in ["hnsw", "flat", "ivf"] {
            cases.push((
                json!({ kind: {"dimension": 2, "base_weight": "2"} }),
                "base_weight must be a number".to_string(),
            ));
        }
        for kind in ["hnsw", "flat", "ivf", "multi_vector"] {
            cases.push((
                json!({ kind: {"dimension": 2, "embedder": 1} }),
                "embedder must be a string".to_string(),
            ));
        }
        cases.push((
            json!({"hnsw": {"dimension": 2, "pq_codebook_path": 1}}),
            "pq_codebook_path must be a string".to_string(),
        ));
        for key in ["mode", "user_dict"] {
            let analyzer = json!({"language": "japanese", "dict": "d", key: 1});
            cases.push((
                json!({"text": {"analyzer": analyzer}}),
                format!("analyzer: {key} must be a string"),
            ));
        }
        // Every option that is not rejected with the expected message, so a
        // failure lists all of them rather than the first.
        let mut wrong = Vec::new();
        for (option, expected) in cases {
            let error = json_to_proto_field_option(&option).err();
            if error.as_deref() != Some(expected.as_str()) {
                wrong.push(format!("{option}: {error:?}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1524: every type [`json_to_proto_field_option`] counts is one
    /// it reads, an omitted or `null` key of a field option keeps its
    /// default (`"analyzer": null` included, which used to be rejected),
    /// and a value of the right type is passed on.
    #[test]
    fn test_field_option_value_types_omitted_null_and_explicit() {
        let wrong: Vec<_> = FIELD_OPTION_TYPES
            .into_iter()
            .filter_map(|kind| {
                json_to_proto_field_option(&json!({ kind: {} }))
                    .err()
                    .map(|e| (kind, e))
            })
            .collect();
        assert!(wrong.is_empty(), "{wrong:?}");

        let option = |json: Value| {
            json_to_proto_field_option(&json).unwrap_or_else(|e| panic!("{json}: {e}"))
        };
        assert_eq!(
            option(json!({"text": {"indexed": true, "analyzer": null}})),
            option(json!({"text": {"indexed": true}}))
        );
        // A `null` type next to another type is the same as omitting it.
        assert_eq!(
            option(json!({"text": {"indexed": true}, "integer": null, "hnsw": null})),
            option(json!({"text": {"indexed": true}}))
        );
        assert_eq!(
            option(json!({"text": null, "hnsw": {"dimension": 2}})),
            option(json!({"hnsw": {"dimension": 2}}))
        );
        for kind in ["hnsw", "flat", "ivf"] {
            assert_eq!(
                option(json!({ kind: {"dimension": 2, "base_weight": null, "embedder": null} })),
                option(json!({ kind: {"dimension": 2} })),
                "{kind}"
            );
        }
        assert_eq!(
            option(json!({"hnsw": {"dimension": 2, "pq_codebook_path": null}})),
            option(json!({"hnsw": {"dimension": 2}}))
        );

        let hnsw = json_to_hnsw_option(&json!({
            "dimension": 2,
            "base_weight": 2,
            "embedder": "e",
            "pq_codebook_path": "p"
        }))
        .unwrap();
        assert_eq!(hnsw.base_weight, Some(2.0));
        assert_eq!(hnsw.embedder, "e");
        assert_eq!(hnsw.pq_codebook_path.as_deref(), Some("p"));

        let japanese = |mode: Value, user_dict: Value| {
            let analyzer = json!({
                "language": "japanese",
                "dict": "d",
                "mode": mode,
                "user_dict": user_dict
            });
            match json_to_analyzer_spec(&analyzer).unwrap().spec {
                Some(v1::analyzer_spec::Spec::Builtin(builtin)) => match builtin.preset {
                    Some(v1::builtin_analyzer_spec::Preset::Japanese(jp)) => {
                        (jp.mode, jp.user_dict)
                    }
                    other => panic!("not a Japanese preset: {other:?}"),
                },
                other => panic!("not a builtin analyzer: {other:?}"),
            }
        };
        assert_eq!(
            japanese(json!(null), json!(null)),
            ("normal".to_string(), String::new())
        );
        assert_eq!(
            japanese(json!("search"), json!("u")),
            ("search".to_string(), "u".to_string())
        );
    }

    /// Issue #1524: a schema that is not an object, and a `default_fields`,
    /// `analyzers`, `embedders`, `char_filters` or `token_filters` of the
    /// wrong type, are rejected instead of read as omitted.
    #[test]
    fn test_json_to_proto_schema_rejects_values_of_the_wrong_type() {
        let title = json!({"title": {"text": {"indexed": true}}});
        let analyzer = |key: &str, value: Value| json!({"ws": {"tokenizer": {"type": "whitespace"}, key: value}});
        let cases = [
            (json!(5), "schema must be an object"),
            (json!(null), "schema must be an object"),
            (
                json!({"fields": title, "default_fields": "title"}),
                "default_fields must be an array of strings",
            ),
            (
                json!({"fields": title, "default_fields": [1, "title"]}),
                "default_fields must be an array of strings",
            ),
            (
                json!({"fields": title, "analyzers": "x"}),
                "analyzers must be an object",
            ),
            (
                json!({"fields": title, "embedders": "x"}),
                "embedders must be an object",
            ),
            (
                json!({"fields": title, "analyzers": analyzer("token_filters", json!("lowercase"))}),
                "analyzer \"ws\": token_filters must be an array",
            ),
            (
                json!({"fields": title, "analyzers": analyzer("char_filters", json!({}))}),
                "analyzer \"ws\": char_filters must be an array",
            ),
            (
                json!({"fields": {"title": {"text": 1}}}),
                "field \"title\": text option must be an object",
            ),
        ];
        let mut wrong = Vec::new();
        for (schema, expected) in cases {
            let error = json_to_proto_schema(&schema).err();
            if error.as_deref() != Some(expected) {
                wrong.push(format!("{schema}: {error:?}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1524: an omitted or `null` key of a schema or an analyzer
    /// definition keeps its default, and a value of the right type is
    /// passed on.
    #[test]
    fn test_json_to_proto_schema_value_types_omitted_null_and_explicit() {
        let title = json!({"title": {"text": {"indexed": true}}});
        let omitted = json_to_proto_schema(&json!({"fields": title})).unwrap();
        let null = json_to_proto_schema(&json!({
            "fields": title,
            "default_fields": null,
            "analyzers": null,
            "embedders": null
        }))
        .unwrap();
        assert_eq!(null, omitted);

        let schema = json_to_proto_schema(&json!({
            "fields": title,
            "default_fields": ["title"],
            "analyzers": {
                "ws": {
                    "tokenizer": {"type": "whitespace"},
                    "char_filters": null,
                    "token_filters": [{"type": "lowercase"}]
                }
            }
        }))
        .unwrap();
        assert_eq!(schema.default_fields, ["title"]);
        let ws = &schema.analyzers["ws"];
        assert!(ws.char_filters.is_empty());
        assert_eq!(ws.token_filters.len(), 1);
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
