//! Conversions between protobuf types and JSON values.
//!
//! These helpers translate between the laurus-server proto `Document` / `Value`
//! types and `serde_json::Value` for the MCP tool input/output.

use base64::Engine as _;
use laurus_server::convert::document::vector_array_from_proto;
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
        Some(Kind::BoolArrayValue(arr)) => json!(arr.values),
        Some(Kind::TextArrayValue(arr)) => json!(arr.values),
        // Array of the same plain base64 strings the scalar `BytesValue` arm
        // produces (#1176); per-element MIME is not on the wire, matching
        // the scalar arm.
        Some(Kind::BytesArrayValue(arr)) => Value::Array(
            arr.values
                .iter()
                .map(|b| Value::String(base64::engine::general_purpose::STANDARD.encode(b)))
                .collect(),
        ),
        // Nested arrays, the same shape JSON input uses for a MultiVector
        // field (#1177).
        Some(Kind::VectorArrayValue(arr)) => json!(vector_array_from_proto(arr)),
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

/// Parse the `fusion` parameter of the `search` tool into a proto
/// [`v1::FusionAlgorithm`]: the same shape as the HTTP gateway's `fusion`,
/// exactly one of `{"rrf": {"k": 60.0}}` and
/// `{"weighted_sum": {"lexical_weight": 0.7, "vector_weight": 0.3}}`.
///
/// Delegates to [`laurus_server::convert::search::fusion_algorithm_from_json`],
/// which the gateway uses too (Issue #1523).
///
/// # Arguments
///
/// * `json_str` - JSON string representing the fusion algorithm.
///
/// # Errors
///
/// Returns an error if the JSON is malformed, does not hold exactly one of
/// the two algorithms, or has a key of the wrong type, such as
/// `{"rrf": {"k": "1"}}`, which used to fall back to the default.
pub fn json_to_fusion_algorithm(json_str: &str) -> anyhow::Result<v1::FusionAlgorithm> {
    let val: Value = serde_json::from_str(json_str)?;
    laurus_server::convert::search::fusion_algorithm_from_json(&val).map_err(anyhow::Error::msg)
}

/// Parse the `field_boosts` parameter of the `search` tool into the field
/// boost map of the proto [`SearchRequest`](v1::SearchRequest): the same
/// shape as the HTTP gateway's `field_boosts`, an object mapping field names
/// to numbers. A `null` boost is the same as omitting the field.
///
/// Delegates to [`laurus_server::convert::search::field_boosts_from_json`]
/// (Issue #1523).
///
/// # Arguments
///
/// * `json_str` - JSON string like `{"title": 2.0, "body": 1.0}`.
///
/// # Errors
///
/// Returns an error if the JSON is malformed, not an object, or has a boost
/// that is not a number, such as `{"title": "2"}`, which used to be dropped.
pub fn json_to_field_boosts(
    json_str: &str,
) -> anyhow::Result<std::collections::HashMap<String, f32>> {
    let val: Value = serde_json::from_str(json_str)?;
    laurus_server::convert::search::field_boosts_from_json(&val).map_err(anyhow::Error::msg)
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
/// Delegates to [`laurus_server::convert::search::highlight_params_from_json`],
/// which the gateway uses too (Issue #1523), so a `null` key is the same as
/// omitting it.
///
/// # Arguments
///
/// * `json_str` - JSON string in either of the two formats above.
///
/// # Errors
///
/// Returns an error if the JSON is malformed or not an array of strings or
/// an object, if a key has the wrong type or an integer is out of the range
/// of its proto type (such as `"max_fragments": "2"`, or `4294967297`, which
/// used to wrap to 1), or if it names no field (`[]`, `{}`,
/// `{"fields": []}`, `null`), which the gateway reads as no highlighting.
/// Validation that only `from_proto` can perform (e.g. rejecting a zero
/// `max_fragments`) is left to the server.
pub fn json_to_highlight_params(json_str: &str) -> anyhow::Result<v1::HighlightParams> {
    let val: Value = serde_json::from_str(json_str)?;
    laurus_server::convert::search::highlight_params_from_json(&val)
        .map_err(anyhow::Error::msg)?
        .ok_or_else(|| anyhow::anyhow!("highlight must name at least one field"))
}

/// Parse a JSON string into [`v1::RescoreParams`] (Issue #1351), in the
/// same shape as the HTTP gateway's `rescore`:
/// `{"window_size": 100, "late_interaction": {"field": "...", "vectors": [[...]] | "text": "..."}}`.
///
/// # Errors
///
/// Returns an error if the JSON is malformed or does not have that shape;
/// the server validates the values (window range, field, query).
pub fn json_to_rescore_params(json_str: &str) -> anyhow::Result<v1::RescoreParams> {
    let val: Value = serde_json::from_str(json_str)?;
    laurus_server::convert::search::rescore_params_from_json(&val).map_err(anyhow::Error::msg)
}

/// Parse the `group_by` parameter of the `search` and `search_batch` tools
/// (Issue #1297): the same shape as the HTTP gateway's `group_by`,
/// `{"field": "_id", "group_size": 3}`.
///
/// # Errors
///
/// Returns an error if the JSON is malformed or does not have that shape;
/// the server validates the values (field, size).
pub fn json_to_group_by_params(json_str: &str) -> anyhow::Result<v1::GroupByParams> {
    let val: Value = serde_json::from_str(json_str)?;
    laurus_server::convert::search::group_by_params_from_json(&val).map_err(anyhow::Error::msg)
}

/// Convert a proto [`v1::SearchResult`] to the JSON shape every MCP search
/// tool returns: `{"id", "score", "fields", "highlights"?, "group_key"?,
/// "inner_hits"?}`. `"highlights"` is present only when at least one field
/// actually highlighted (Issue #1134); `"group_key"` only when set and
/// `"inner_hits"` only when non-empty (Issue #1297), matching the HTTP
/// gateway's `proto_search_result_to_json`.
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
    if let Some(group_key) = &result.group_key {
        obj["group_key"] = proto_value_to_json(group_key);
    }
    if !result.inner_hits.is_empty() {
        obj["inner_hits"] = Value::Array(
            result
                .inner_hits
                .iter()
                .map(search_result_to_json)
                .collect(),
        );
    }
    obj
}

/// Convert a proto [`v1::SearchResponse`] to the JSON the `search` tool
/// returns, and `search_batch` returns per query: `{"results": [...]}`, each
/// result converted by [`search_result_to_json`].
///
/// There is no `"total"`: it held only the number of returned results, as
/// the server's `total_hits` did, and was removed (Issue #1455).
pub fn search_response_to_json(response: &v1::SearchResponse) -> Value {
    json!({
        "results": response
            .results
            .iter()
            .map(search_result_to_json)
            .collect::<Vec<_>>(),
    })
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

    /// Issue #1523: a `fusion` value of the wrong type is an error naming
    /// the key, as in the HTTP gateway. It used to fall back to the default
    /// (k 60, weights 0.5), and `rrf` won over `weighted_sum`.
    #[test]
    fn test_json_to_fusion_algorithm_rejects_values_of_the_wrong_type() {
        let exactly_one = "fusion must have exactly one of \"rrf\" and \"weighted_sum\"";
        let cases = [
            (r#"{"rrf": {"k": "1"}}"#, "fusion: rrf: k must be a number"),
            (r#"{"rrf": 5}"#, "fusion: rrf must be an object"),
            (r#"{"rrf": []}"#, "fusion: rrf must be an object"),
            (
                r#"{"weighted_sum": 5}"#,
                "fusion: weighted_sum must be an object",
            ),
            (
                r#"{"weighted_sum": {"lexical_weight": "1", "vector_weight": 0}}"#,
                "fusion: weighted_sum: lexical_weight must be a number",
            ),
            (
                r#"{"weighted_sum": {"vector_weight": true}}"#,
                "fusion: weighted_sum: vector_weight must be a number",
            ),
            (r#"{"rrf": {"k": 1}, "weighted_sum": {}}"#, exactly_one),
            (r#"{"rrf": null}"#, exactly_one),
            ("{}", exactly_one),
            ("5", "fusion must be an object"),
            ("null", "fusion must be an object"),
        ];
        // Every value that is not rejected with the expected message, so a
        // failure lists all of them rather than the first.
        let wrong: Vec<_> = cases
            .iter()
            .filter_map(|(json, expected)| {
                let error = json_to_fusion_algorithm(json).err().map(|e| e.to_string());
                (error.as_deref() != Some(expected)).then(|| format!("{json}: {error:?}"))
            })
            .collect();
        assert!(wrong.is_empty(), "{wrong:?}");
    }

    /// Issue #1523: a `null` key of `fusion` is the same as omitting it.
    #[test]
    fn test_json_to_fusion_algorithm_null_keys_are_omitted() {
        let rrf = |json: &str| match json_to_fusion_algorithm(json).unwrap().algorithm {
            Some(v1::fusion_algorithm::Algorithm::Rrf(rrf)) => rrf.k,
            other => panic!("{json}: expected RRF, got {other:?}"),
        };
        assert_eq!(rrf(r#"{"rrf": {"k": null}}"#), 60.0);
        assert_eq!(rrf(r#"{"rrf": {}, "weighted_sum": null}"#), 60.0);

        match json_to_fusion_algorithm(
            r#"{"weighted_sum": {"lexical_weight": null, "vector_weight": null}, "rrf": null}"#,
        )
        .unwrap()
        .algorithm
        {
            Some(v1::fusion_algorithm::Algorithm::WeightedSum(ws)) => {
                assert_eq!((ws.lexical_weight, ws.vector_weight), (0.5, 0.5));
            }
            other => panic!("expected WeightedSum, got {other:?}"),
        }
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

    /// Issue #1523: a boost that is not a number is an error naming the
    /// field, as in the HTTP gateway, instead of being dropped, and a `null`
    /// boost is the same as leaving the field out.
    #[test]
    fn test_json_to_field_boosts_rejects_values_of_the_wrong_type() {
        let cases = [
            (
                r#"{"title": "2"}"#,
                "field_boosts: \"title\" must be a number",
            ),
            (
                r#"{"title": true}"#,
                "field_boosts: \"title\" must be a number",
            ),
            (
                r#"{"body": 1, "title": [2]}"#,
                "field_boosts: \"title\" must be a number",
            ),
            ("[1]", "field_boosts must be an object"),
            ("null", "field_boosts must be an object"),
        ];
        let wrong: Vec<_> = cases
            .iter()
            .filter_map(|(json, expected)| {
                let error = json_to_field_boosts(json).err().map(|e| e.to_string());
                (error.as_deref() != Some(expected)).then(|| format!("{json}: {error:?}"))
            })
            .collect();
        assert!(wrong.is_empty(), "{wrong:?}");

        let boosts = json_to_field_boosts(r#"{"title": null, "body": 1}"#).unwrap();
        assert_eq!(boosts, HashMap::from([("body".to_string(), 1.0)]));
    }

    #[test]
    fn test_json_to_rescore_params() {
        let params = json_to_rescore_params(
            r#"{"window_size": 20, "late_interaction": {"field": "body_colbert", "text": "rust lifetimes"}}"#,
        )
        .unwrap();
        assert_eq!(params.window_size, Some(20));
        assert!(matches!(
            params.rescorer,
            Some(v1::rescore_params::Rescorer::LateInteraction(ref late))
                if late.field == "body_colbert"
                    && late.query == Some(v1::late_interaction_rescore::Query::Text("rust lifetimes".into()))
        ));

        assert!(json_to_rescore_params("not json").is_err());
        let err = json_to_rescore_params(r#"{"late_interaction": {"field": "t"}}"#).unwrap_err();
        assert!(
            err.to_string().contains("exactly one of vectors and text"),
            "{err}"
        );
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

    /// Issue #1523: a `highlight` key of the wrong type, or an integer out
    /// of the range of its proto type, is an error naming the key, as in
    /// the HTTP gateway. Such a value used to be read as omitted or dropped,
    /// and `max_fragments` wrapped (`4294967297` became 1).
    #[test]
    fn test_json_to_highlight_params_rejects_values_of_the_wrong_type() {
        // The JSON text of each value, so that a number above `u64::MAX`
        // reaches the parser as `serde_json` reads it from a tool call.
        let uint32 = [r#""1""#, "1.0", "-1", "4294967296", "4294967297"];
        let uint64 = [r#""3""#, "3.0", "-1", "18446744073709551616"];
        let boolean = [r#""true""#, "1"];
        let string = ["1", r#"["em"]"#];
        let keys: [(&str, &[&str], String); 7] = [
            (
                "max_fragments",
                &uint32,
                format!("an integer between 0 and {}", u32::MAX),
            ),
            (
                "fragment_size",
                &uint32,
                format!("an integer between 0 and {}", u32::MAX),
            ),
            (
                "max_analyzed_chars",
                &uint64,
                format!("an integer between 0 and {}", u64::MAX),
            ),
            ("require_field_match", &boolean, "true or false".to_string()),
            (
                "return_entire_field_if_no_highlight",
                &boolean,
                "true or false".to_string(),
            ),
            ("tag", &string, "a string".to_string()),
            ("css_class", &string, "a string".to_string()),
        ];
        let mut cases: Vec<(String, String)> = keys
            .iter()
            .flat_map(|(key, bad_values, what)| {
                bad_values.iter().map(move |bad| {
                    (
                        format!(r#"{{"fields": ["body"], "{key}": {bad}}}"#),
                        format!("highlight: {key} must be {what}"),
                    )
                })
            })
            .collect();
        let shape = "highlight must be an array of strings or an object";
        let fields = "highlight: fields must be an array of strings";
        for (json, expected) in [
            (r#"["body", 1]"#, shape),
            ("42", shape),
            (r#""body""#, shape),
            (r#"{"fields": ["body", 1]}"#, fields),
            (r#"{"fields": "body"}"#, fields),
            // Checked even when the highlight names no field.
            (
                r#"{"fields": [], "max_fragments": "1"}"#,
                "highlight: max_fragments must be an integer between 0 and 4294967295",
            ),
        ] {
            cases.push((json.to_string(), expected.to_string()));
        }

        let wrong: Vec<_> = cases
            .iter()
            .filter_map(|(json, expected)| {
                let error = json_to_highlight_params(json).err().map(|e| e.to_string());
                (error.as_ref() != Some(expected)).then(|| format!("{json}: {error:?}"))
            })
            .collect();
        assert!(wrong.is_empty(), "{wrong:?}");
    }

    /// Issue #1523: a `null` key of `highlight` is the same as omitting it,
    /// and a `highlight` that names no field is still an error (the HTTP
    /// gateway reads it as no highlighting).
    #[test]
    fn test_json_to_highlight_params_null_keys_and_no_field() {
        let params = json_to_highlight_params(
            r#"{
                "fields": ["body"],
                "max_fragments": null,
                "fragment_size": null,
                "tag": null,
                "css_class": null,
                "require_field_match": null,
                "max_analyzed_chars": null,
                "return_entire_field_if_no_highlight": null
            }"#,
        )
        .unwrap();
        assert_eq!(
            params,
            v1::HighlightParams {
                fields: vec!["body".to_string()],
                ..Default::default()
            }
        );

        for json in [
            "[]",
            "{}",
            r#"{"fields": []}"#,
            r#"{"fields": null}"#,
            "null",
        ] {
            let error = json_to_highlight_params(json).map_err(|e| e.to_string());
            assert_eq!(
                error,
                Err("highlight must name at least one field".to_string()),
                "{json}"
            );
        }
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
            ..Default::default()
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
            ..Default::default()
        };

        let json = search_result_to_json(&result);
        assert!(json.get("highlights").is_none());
    }

    /// Issue #1297: the `group_by` parameter has the HTTP gateway's shape.
    #[test]
    fn test_json_to_group_by_params() {
        let params = json_to_group_by_params(r#"{"field": "_id", "group_size": 3}"#).unwrap();
        assert_eq!(params.field, "_id");
        assert_eq!(params.group_size, Some(3));

        assert!(json_to_group_by_params("not json").is_err());
        let err = json_to_group_by_params(r#"{"group_size": 3}"#).unwrap_err();
        assert!(err.to_string().contains("group_by.field"), "{err}");
    }

    /// Issue #1297: a grouped result carries its group key and its inner
    /// hits, each in the same shape; an ungrouped one gains neither key.
    #[test]
    fn test_search_result_to_json_includes_groups_when_present() {
        let hit = |id: &str| v1::SearchResult {
            id: id.to_string(),
            score: 0.5,
            group_key: Some(int_value(2024)),
            ..Default::default()
        };
        let leader = v1::SearchResult {
            inner_hits: vec![hit("b")],
            ..hit("a")
        };

        let json = search_result_to_json(&leader);
        assert_eq!(json["group_key"], json!(2024));
        assert_eq!(json["inner_hits"][0]["id"], json!("b"));
        assert_eq!(json["inner_hits"][0]["group_key"], json!(2024));

        let ungrouped = search_result_to_json(&v1::SearchResult {
            id: "a".to_string(),
            ..Default::default()
        });
        assert!(ungrouped.get("group_key").is_none(), "{ungrouped}");
        assert!(ungrouped.get("inner_hits").is_none(), "{ungrouped}");
    }

    /// Issue #1455: a search's JSON is `{"results": [...]}` and nothing
    /// else. `"total"` held only the number of returned results, so no total
    /// comes back under any key.
    #[test]
    fn test_search_response_to_json_has_only_results() {
        let hit = |id: &str| v1::SearchResult {
            id: id.to_string(),
            score: 0.5,
            ..Default::default()
        };
        let response = v1::SearchResponse {
            results: vec![hit("a"), hit("b")],
        };

        assert_eq!(
            search_response_to_json(&response),
            json!({
                "results": [
                    search_result_to_json(&hit("a")),
                    search_result_to_json(&hit("b")),
                ]
            })
        );
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

    #[test]
    fn json_to_proto_bool_array_and_back() {
        // #1180: an array of booleans becomes a `BoolArrayValue` and is
        // surfaced back to MCP clients as JSON booleans.
        let json_val = json!({ "fields": { "flags": [true, false] } });
        let doc = json_to_document(&json_val).unwrap();
        match &doc.fields["flags"].kind {
            Some(v1::value::Kind::BoolArrayValue(a)) => {
                assert_eq!(a.values, vec![true, false]);
            }
            other => panic!("expected BoolArrayValue, got {other:?}"),
        }
        let json = document_to_json(&doc);
        assert_eq!(json["fields"]["flags"], json!([true, false]));
    }

    #[test]
    fn json_to_proto_text_array_and_back() {
        // #1175: a string array that is not all RFC 3339 becomes a
        // `TextArrayValue` and is surfaced back to MCP clients as strings.
        let json_val = json!({ "fields": { "links": ["https://a.example", "https://b.example"] } });
        let doc = json_to_document(&json_val).unwrap();
        match &doc.fields["links"].kind {
            Some(v1::value::Kind::TextArrayValue(a)) => {
                assert_eq!(a.values, vec!["https://a.example", "https://b.example"]);
            }
            other => panic!("expected TextArrayValue, got {other:?}"),
        }
        let json = document_to_json(&doc);
        assert_eq!(
            json["fields"]["links"],
            json!(["https://a.example", "https://b.example"])
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

    #[test]
    fn proto_bytes_array_to_json_is_base64_string_array() {
        // #1176: a `BytesArrayValue` is surfaced to MCP clients as an array
        // of the same plain base64 strings the scalar `BytesValue` arm
        // produces, with no per-element MIME wrapper (matching the scalar
        // arm, which also drops it).
        let mut fields = HashMap::new();
        fields.insert(
            "thumbs".to_string(),
            v1::Value {
                kind: Some(v1::value::Kind::BytesArrayValue(v1::BytesArrayValue {
                    values: vec![b"hi".to_vec(), b"bye".to_vec()],
                })),
            },
        );
        let doc = v1::Document { fields };
        let json = document_to_json(&doc);
        assert_eq!(json["fields"]["thumbs"], json!(["aGk=", "Ynll"]));
    }

    #[test]
    fn proto_vector_array_to_json_is_nested_arrays() {
        // #1177: a packed `VectorArrayValue` is surfaced as nested arrays,
        // the shape JSON input uses for a MultiVector field.
        let mut fields = HashMap::new();
        fields.insert(
            "tokens".to_string(),
            v1::Value {
                kind: Some(v1::value::Kind::VectorArrayValue(v1::VectorArrayValue {
                    dimension: 2,
                    values: vec![1.0, 0.5, -1.0, 2.0],
                })),
            },
        );
        let doc = v1::Document { fields };
        let json = document_to_json(&doc);
        assert_eq!(json["fields"]["tokens"], json!([[1.0, 0.5], [-1.0, 2.0]]));
    }
}
