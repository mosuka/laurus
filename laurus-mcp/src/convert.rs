//! Conversions between protobuf types and JSON values.
//!
//! These helpers translate between the laurus-server proto `Document` / `Value`
//! types and `serde_json::Value` for the MCP tool input/output.

use base64::Engine as _;
use laurus_server::proto::laurus::v1;
use serde_json::{Value, json};

/// Convert a proto [`v1::Document`]'s fields into a JSON fields map:
/// `{field: value, ...}`, with no `"fields"` wrapper and no `id`.
///
/// Shared by [`document_to_json`] (wraps the result under `"fields"`) and
/// the `search`/`search_batch` tool handlers (insert it directly under a
/// search result's own `"fields"` key), so every MCP tool agrees on the
/// same `{"fields": {...}}` shape [`json_to_document`] accepts back.
pub fn document_fields_to_json(doc: &v1::Document) -> Value {
    let fields: serde_json::Map<String, Value> = doc
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), proto_value_to_json(v)))
        .collect();
    Value::Object(fields)
}

/// Convert a proto [`v1::Document`] to a [`serde_json::Value`] of the shape
/// `{"fields": {...}}` — the same shape [`json_to_document`] accepts back,
/// and the same one laurus-cli and the HTTP gateway use.
///
/// # Arguments
///
/// * `doc` - The proto document to convert.
pub fn document_to_json(doc: &v1::Document) -> Value {
    json!({ "fields": document_fields_to_json(doc) })
}

/// Convert a JSON value of the shape `{"fields": {...}}` into a proto
/// [`v1::Document`].
///
/// Delegates to [`laurus::json_to_document`] — the same canonical converter
/// used by laurus-cli and the HTTP gateway, so all three JSON-accepting
/// transports agree on one document shape — and lowers the resulting
/// [`laurus::Document`] to the proto wire form via
/// [`laurus_server::convert::document::to_proto`].
///
/// # Arguments
///
/// * `value` - A JSON value of the shape `{"fields": {...}}`.
///
/// # Errors
///
/// Returns an error when `value` lacks a `fields` object or any individual
/// field value cannot be inferred.
pub fn json_to_document(value: &Value) -> anyhow::Result<v1::Document> {
    let doc = laurus::json_to_document(value)?;
    Ok(laurus_server::convert::document::to_proto(&doc))
}

fn proto_value_to_json(val: &v1::Value) -> Value {
    use v1::value::Kind;
    match &val.kind {
        None | Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(*b),
        Some(Kind::Int64Value(i)) => json!(i),
        Some(Kind::Float64Value(f)) => json!(f),
        Some(Kind::TextValue(s)) => Value::String(s.clone()),
        Some(Kind::BytesValue(b)) => {
            // Base64, matching the HTTP gateway's BytesValue -> JSON
            // encoding and the plain-string shape `coerce_to_bytes`
            // decodes back into a declared `Bytes` field.
            Value::String(base64::engine::general_purpose::STANDARD.encode(b))
        }
        Some(Kind::VectorValue(v)) => json!(v.values),
        Some(Kind::DatetimeValue(us)) => datetime_micros_to_json(*us),
        Some(Kind::GeoValue(g)) => json!({ "lat": g.latitude, "lon": g.longitude }),
        Some(Kind::Geo3dValue(p)) => json!({ "x": p.x, "y": p.y, "z": p.z }),
        Some(Kind::Int64ArrayValue(arr)) => json!(arr.values),
        Some(Kind::Float64ArrayValue(arr)) => json!(arr.values),
        Some(Kind::GeoArrayValue(arr)) => Value::Array(
            arr.values
                .iter()
                .map(|g| json!({ "lat": g.latitude, "lon": g.longitude }))
                .collect(),
        ),
        Some(Kind::Geo3dArrayValue(arr)) => Value::Array(
            arr.values
                .iter()
                .map(|p| json!({ "x": p.x, "y": p.y, "z": p.z }))
                .collect(),
        ),
        Some(Kind::DatetimeArrayValue(arr)) => Value::Array(
            arr.values
                .iter()
                .map(|us| datetime_micros_to_json(*us))
                .collect(),
        ),
    }
}

/// Convert Unix microseconds to an ISO 8601 / RFC 3339 string, or the raw
/// number when the value is outside chrono's range. `from_timestamp_micros`
/// handles pre-1970 values correctly, unlike the previous truncating
/// seconds/nanos split.
fn datetime_micros_to_json(us: i64) -> Value {
    match chrono::DateTime::from_timestamp_micros(us) {
        Some(dt) => Value::String(dt.to_rfc3339()),
        None => json!(us),
    }
}

/// Parse a JSON string into a proto [`v1::FusionAlgorithm`].
///
/// Accepts two formats:
/// - `{"rrf": {"k": 60.0}}`
/// - `{"weighted_sum": {"lexical_weight": 0.7, "vector_weight": 0.3}}`
///
/// # Arguments
///
/// * `json_str` - JSON string representing the fusion algorithm.
///
/// # Errors
///
/// Returns an error if the JSON is malformed or does not match either format.
pub fn json_to_fusion_algorithm(json_str: &str) -> anyhow::Result<v1::FusionAlgorithm> {
    let val: Value = serde_json::from_str(json_str)?;

    if let Some(rrf) = val.get("rrf") {
        let k = rrf.get("k").and_then(|v| v.as_f64()).unwrap_or(60.0);
        Ok(v1::FusionAlgorithm {
            algorithm: Some(v1::fusion_algorithm::Algorithm::Rrf(v1::Rrf { k })),
        })
    } else if let Some(ws) = val.get("weighted_sum") {
        let lexical_weight = ws
            .get("lexical_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        let vector_weight = ws
            .get("vector_weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;
        Ok(v1::FusionAlgorithm {
            algorithm: Some(v1::fusion_algorithm::Algorithm::WeightedSum(
                v1::WeightedSum {
                    lexical_weight,
                    vector_weight,
                },
            )),
        })
    } else {
        Err(anyhow::anyhow!(
            "fusion must contain either \"rrf\" or \"weighted_sum\" key"
        ))
    }
}

/// Parse a JSON string into a field boost map for the proto
/// [`SearchRequest`](v1::SearchRequest).
///
/// Expects a JSON object mapping field names to numeric boost values.
///
/// # Arguments
///
/// * `json_str` - JSON string like `{"title": 2.0, "body": 1.0}`.
///
/// # Errors
///
/// Returns an error if the JSON is malformed or not an object.
pub fn json_to_field_boosts(
    json_str: &str,
) -> anyhow::Result<std::collections::HashMap<String, f32>> {
    let val: Value = serde_json::from_str(json_str)?;
    let obj = val
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("field_boosts must be a JSON object"))?;
    Ok(obj
        .iter()
        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f as f32)))
        .collect())
}

/// Parse a JSON string into a proto [`v1::HighlightParams`] (Issue #1134),
/// for the `search` / `search_batch` tools' `highlight` parameter.
///
/// Accepts two formats, matching the HTTP gateway's `highlight` field:
/// - The array shorthand — just a field list: `["title", "body"]`.
/// - The full object form, adding any of the optional `HighlightConfig`
///   knobs: `{"fields": ["body"], "max_fragments": 2, "tag": "em",
///   "css_class": "hl", "require_field_match": false,
///   "max_analyzed_chars": 500, "return_entire_field_if_no_highlight": true}`.
///
/// # Arguments
///
/// * `json_str` - JSON string in either of the two formats above.
///
/// # Errors
///
/// Returns an error if the JSON is malformed, not an array or object, or an
/// object form has no non-empty `fields` array. Validation that only
/// `from_proto` can perform (e.g. rejecting a zero `max_fragments`) is left
/// to the server; this function only parses the shape.
pub fn json_to_highlight_params(json_str: &str) -> anyhow::Result<v1::HighlightParams> {
    let val: Value = serde_json::from_str(json_str)?;

    let fields_from = |v: &Value| -> Option<Vec<String>> {
        v.as_array().map(|arr| {
            arr.iter()
                .filter_map(|f| f.as_str().map(str::to_string))
                .collect()
        })
    };

    if let Some(fields) = fields_from(&val) {
        if fields.is_empty() {
            return Err(anyhow::anyhow!("highlight field list must not be empty"));
        }
        return Ok(v1::HighlightParams {
            fields,
            ..Default::default()
        });
    }

    let obj = val
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("highlight must be a field array or an object"))?;
    let fields = obj.get("fields").and_then(fields_from).unwrap_or_default();
    if fields.is_empty() {
        return Err(anyhow::anyhow!(
            "highlight.fields must be a non-empty array of field names"
        ));
    }
    Ok(v1::HighlightParams {
        fields,
        max_fragments: obj
            .get("max_fragments")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
        fragment_size: obj
            .get("fragment_size")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32),
        tag: obj.get("tag").and_then(|v| v.as_str()).map(str::to_string),
        css_class: obj
            .get("css_class")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        require_field_match: obj.get("require_field_match").and_then(|v| v.as_bool()),
        max_analyzed_chars: obj.get("max_analyzed_chars").and_then(|v| v.as_u64()),
        return_entire_field_if_no_highlight: obj
            .get("return_entire_field_if_no_highlight")
            .and_then(|v| v.as_bool()),
    })
}

/// Convert a proto [`v1::SearchResult`] to the JSON shape every MCP search
/// tool returns: `{"id", "score", "fields", "highlights"?}`. `"highlights"`
/// is present only when at least one field actually highlighted, matching
/// the HTTP gateway's `proto_search_result_to_json` (Issue #1134).
pub fn search_result_to_json(result: &v1::SearchResult) -> Value {
    let mut obj = json!({
        "id": result.id,
        "score": result.score,
        "fields": result.document.as_ref().map(document_fields_to_json),
    });
    if !result.highlights.is_empty() {
        obj["highlights"] = Value::Object(
            result
                .highlights
                .iter()
                .map(|(field, highlights)| (field.clone(), json!(highlights.fragments)))
                .collect(),
        );
    }
    obj
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn text_value(s: &str) -> v1::Value {
        v1::Value {
            kind: Some(v1::value::Kind::TextValue(s.to_string())),
        }
    }

    fn int_value(i: i64) -> v1::Value {
        v1::Value {
            kind: Some(v1::value::Kind::Int64Value(i)),
        }
    }

    fn float_value(f: f64) -> v1::Value {
        v1::Value {
            kind: Some(v1::value::Kind::Float64Value(f)),
        }
    }

    #[test]
    fn test_document_to_json() {
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), text_value("hello"));
        fields.insert("score".to_string(), float_value(1.5));
        fields.insert("count".to_string(), int_value(42));
        let doc = v1::Document { fields };

        let json = document_to_json(&doc);
        assert_eq!(json["fields"]["title"], "hello");
        assert_eq!(json["fields"]["score"], 1.5);
        assert_eq!(json["fields"]["count"], 42);
    }

    #[test]
    fn test_json_to_fusion_algorithm_rrf() {
        let json = r#"{"rrf": {"k": 30.0}}"#;
        let fusion = json_to_fusion_algorithm(json).unwrap();
        match fusion.algorithm {
            Some(v1::fusion_algorithm::Algorithm::Rrf(rrf)) => {
                assert!((rrf.k - 30.0).abs() < f64::EPSILON);
            }
            _ => panic!("Expected RRF"),
        }
    }

    #[test]
    fn test_json_to_fusion_algorithm_rrf_default_k() {
        let json = r#"{"rrf": {}}"#;
        let fusion = json_to_fusion_algorithm(json).unwrap();
        match fusion.algorithm {
            Some(v1::fusion_algorithm::Algorithm::Rrf(rrf)) => {
                assert!((rrf.k - 60.0).abs() < f64::EPSILON);
            }
            _ => panic!("Expected RRF with default k"),
        }
    }

    #[test]
    fn test_json_to_fusion_algorithm_weighted_sum() {
        let json = r#"{"weighted_sum": {"lexical_weight": 0.7, "vector_weight": 0.3}}"#;
        let fusion = json_to_fusion_algorithm(json).unwrap();
        match fusion.algorithm {
            Some(v1::fusion_algorithm::Algorithm::WeightedSum(ws)) => {
                assert!((ws.lexical_weight - 0.7).abs() < f32::EPSILON);
                assert!((ws.vector_weight - 0.3).abs() < f32::EPSILON);
            }
            _ => panic!("Expected WeightedSum"),
        }
    }

    #[test]
    fn test_json_to_fusion_algorithm_invalid() {
        let json = r#"{"unknown": {}}"#;
        assert!(json_to_fusion_algorithm(json).is_err());
    }

    #[test]
    fn test_json_to_field_boosts() {
        let json = r#"{"title": 2.0, "body": 1.0}"#;
        let boosts = json_to_field_boosts(json).unwrap();
        assert_eq!(boosts.len(), 2);
        assert!((boosts["title"] - 2.0).abs() < f32::EPSILON);
        assert!((boosts["body"] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_json_to_field_boosts_empty() {
        let json = r#"{}"#;
        let boosts = json_to_field_boosts(json).unwrap();
        assert!(boosts.is_empty());
    }

    #[test]
    fn test_json_to_field_boosts_invalid() {
        let json = r#"[1, 2, 3]"#;
        assert!(json_to_field_boosts(json).is_err());
    }

    #[test]
    fn test_json_to_highlight_params_array_shorthand() {
        let params = json_to_highlight_params(r#"["title", "body"]"#).unwrap();
        assert_eq!(params.fields, ["title", "body"]);
        assert_eq!(params.max_fragments, None);
    }

    #[test]
    fn test_json_to_highlight_params_object_form() {
        let json = r#"{
            "fields": ["body"],
            "max_fragments": 2,
            "fragment_size": 80,
            "tag": "em",
            "css_class": "hl",
            "require_field_match": false,
            "max_analyzed_chars": 500,
            "return_entire_field_if_no_highlight": true
        }"#;
        let params = json_to_highlight_params(json).unwrap();
        assert_eq!(params.fields, ["body"]);
        assert_eq!(params.max_fragments, Some(2));
        assert_eq!(params.fragment_size, Some(80));
        assert_eq!(params.tag.as_deref(), Some("em"));
        assert_eq!(params.css_class.as_deref(), Some("hl"));
        assert_eq!(params.require_field_match, Some(false));
        assert_eq!(params.max_analyzed_chars, Some(500));
        assert_eq!(params.return_entire_field_if_no_highlight, Some(true));
    }

    #[test]
    fn test_json_to_highlight_params_rejects_empty_array() {
        assert!(json_to_highlight_params("[]").is_err());
    }

    #[test]
    fn test_json_to_highlight_params_rejects_missing_fields() {
        assert!(json_to_highlight_params(r#"{"max_fragments": 2}"#).is_err());
        assert!(json_to_highlight_params(r#"{"fields": []}"#).is_err());
    }

    #[test]
    fn test_json_to_highlight_params_rejects_non_array_non_object() {
        assert!(json_to_highlight_params("42").is_err());
    }

    #[test]
    fn test_search_result_to_json_includes_highlights_when_present() {
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
        };

        let json = search_result_to_json(&result);
        assert_eq!(
            json["highlights"]["body"],
            json!(["<mark>Rust</mark> is great"])
        );
    }

    #[test]
    fn test_search_result_to_json_omits_highlights_key_when_empty() {
        let result = v1::SearchResult {
            id: "doc1".to_string(),
            score: 0.5,
            document: None,
            highlights: HashMap::new(),
        };

        let json = search_result_to_json(&result);
        assert!(json.get("highlights").is_none());
    }

    #[test]
    fn test_json_to_document() {
        let json_val = json!({
            "fields": {
                "text_field": "hello",
                "int_field": 10,
                "float_field": 2.78,
                "bool_field": true,
                "null_field": null,
                "vec_field": [0.1_f32, 0.2_f32, 0.3_f32]
            }
        });

        let doc = json_to_document(&json_val).unwrap();
        assert!(matches!(
            doc.fields["text_field"].kind,
            Some(v1::value::Kind::TextValue(_))
        ));
        assert!(matches!(
            doc.fields["int_field"].kind,
            Some(v1::value::Kind::Int64Value(10))
        ));
        assert!(matches!(
            doc.fields["bool_field"].kind,
            Some(v1::value::Kind::BoolValue(true))
        ));
        assert!(
            !doc.fields.contains_key("null_field"),
            "a null field must be omitted, not inserted as NullValue"
        );
        // A plain numeric JSON array is inferred as Int64ArrayValue /
        // Float64ArrayValue (json_to_document has no schema access, so it
        // never produces VectorValue directly — see type_inference.rs).
        assert!(matches!(
            doc.fields["vec_field"].kind,
            Some(v1::value::Kind::Float64ArrayValue(_))
        ));
    }

    #[test]
    fn json_to_proto_geo_3d_object() {
        // `{ x, y, z }` JSON input must be encoded as a Geo3dValue proto
        // kind so MCP `put_document` can pass ECEF coordinates through
        // to laurus-server.
        let json_val = json!({
            "fields": { "position": { "x": 1_000_000.0, "y": 2_000_000.0, "z": 3_000_000.0 } }
        });
        let doc = json_to_document(&json_val).unwrap();
        match &doc.fields["position"].kind {
            Some(v1::value::Kind::Geo3dValue(p)) => {
                assert_eq!(p.x, 1_000_000.0);
                assert_eq!(p.y, 2_000_000.0);
                assert_eq!(p.z, 3_000_000.0);
            }
            other => panic!("expected Geo3dValue, got {other:?}"),
        }
    }

    #[test]
    fn json_to_proto_geo_2d_object() {
        // `{ lat, lon }` (and the long-form `latitude` / `longitude`)
        // produces a 2D GeoValue. Confirms #305 wiring did not break
        // the historical 2D shape.
        let json_val = json!({
            "fields": {
                "spot_a": { "lat": 35.6, "lon": 139.7 },
                "spot_b": { "latitude": -33.86, "longitude": 151.21 },
            }
        });
        let doc = json_to_document(&json_val).unwrap();
        for field in ["spot_a", "spot_b"] {
            assert!(
                matches!(doc.fields[field].kind, Some(v1::value::Kind::GeoValue(_))),
                "{field} must be GeoValue"
            );
        }
    }

    #[test]
    fn proto_to_json_geo_3d_round_trip() {
        // Reverse direction: a Geo3dValue from the server is surfaced
        // to MCP clients as `{ x, y, z }`.
        let mut fields = HashMap::new();
        fields.insert(
            "position".to_string(),
            v1::Value {
                kind: Some(v1::value::Kind::Geo3dValue(v1::Geo3dPoint {
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                })),
            },
        );
        let doc = v1::Document { fields };
        let json = document_to_json(&doc);
        assert_eq!(
            json["fields"]["position"],
            json!({"x": 4.0, "y": 5.0, "z": 6.0})
        );
    }

    #[test]
    fn json_to_proto_geo_arrays_and_back() {
        // #1174: an array of geo objects becomes a `GeoArrayValue` (or
        // `Geo3dArrayValue`) and is surfaced back to MCP clients as an
        // array of `{ lat, lon }` / `{ x, y, z }` objects.
        let json_val = json!({
            "fields": {
                "spots": [
                    { "lat": 35.6, "lon": 139.7 },
                    { "latitude": -33.86, "longitude": 151.21 },
                ],
                "positions": [{ "x": 1.0, "y": 2.0, "z": 3.0 }],
            }
        });
        let doc = json_to_document(&json_val).unwrap();
        assert!(matches!(
            doc.fields["spots"].kind,
            Some(v1::value::Kind::GeoArrayValue(_))
        ));
        assert!(matches!(
            doc.fields["positions"].kind,
            Some(v1::value::Kind::Geo3dArrayValue(_))
        ));

        let json = document_to_json(&doc);
        assert_eq!(
            json["fields"]["spots"],
            json!([{ "lat": 35.6, "lon": 139.7 }, { "lat": -33.86, "lon": 151.21 }])
        );
        assert_eq!(
            json["fields"]["positions"],
            json!([{ "x": 1.0, "y": 2.0, "z": 3.0 }])
        );
    }

    #[test]
    fn json_to_proto_datetime_array_and_back() {
        // #1184: an array of RFC 3339 strings becomes a `DatetimeArrayValue`
        // and is surfaced back to MCP clients as RFC 3339 strings (UTC).
        let json_val = json!({
            "fields": {
                "times": ["2024-01-01T00:00:00Z", "2024-06-15T21:00:00+09:00"],
            }
        });
        let doc = json_to_document(&json_val).unwrap();
        match &doc.fields["times"].kind {
            Some(v1::value::Kind::DatetimeArrayValue(a)) => {
                assert_eq!(a.values, vec![1_704_067_200_000_000, 1_718_452_800_000_000]);
            }
            other => panic!("expected DatetimeArrayValue, got {other:?}"),
        }
        let json = document_to_json(&doc);
        assert_eq!(
            json["fields"]["times"],
            json!(["2024-01-01T00:00:00+00:00", "2024-06-15T12:00:00+00:00"])
        );
    }

    /// Regression: pre-1970 micros used to collapse onto the epoch because
    /// the seconds/nanos split truncated toward zero.
    #[test]
    fn proto_to_json_datetime_pre_1970() {
        let mut fields = HashMap::new();
        fields.insert(
            "t".to_string(),
            v1::Value {
                kind: Some(v1::value::Kind::DatetimeValue(-86_400_000_001)),
            },
        );
        let json = document_to_json(&v1::Document { fields });
        assert_eq!(
            json["fields"]["t"],
            json!("1969-12-30T23:59:59.999999+00:00")
        );
    }

    #[test]
    fn json_to_proto_bytes_base64() {
        let json_val = json!({"fields": {"thumb": {"data": "aGk=", "mime": "image/jpeg"}}});
        let doc = json_to_document(&json_val).unwrap();
        match &doc.fields["thumb"].kind {
            Some(v1::value::Kind::BytesValue(b)) => assert_eq!(b, b"hi"),
            other => panic!("expected BytesValue, got {other:?}"),
        }
    }

    #[test]
    fn proto_bytes_to_json_is_base64_string() {
        let mut fields = HashMap::new();
        fields.insert(
            "thumb".to_string(),
            v1::Value {
                kind: Some(v1::value::Kind::BytesValue(b"hi".to_vec())),
            },
        );
        let doc = v1::Document { fields };
        let json = document_to_json(&doc);
        assert_eq!(json["fields"]["thumb"], json!("aGk="));
    }
}
