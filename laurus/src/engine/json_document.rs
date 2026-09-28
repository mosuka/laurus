//! Canonical JSON ⇄ [`Document`] conversion for transport layers.
//!
//! This module is the single source of truth for how a document's field
//! values are represented in JSON across every transport that accepts JSON
//! (laurus-cli, the laurus-server HTTP gateway, and laurus-mcp). Each
//! transport is expected to call [`json_to_document`] rather than
//! hand-rolling its own JSON → [`Document`] conversion.
//!
//! # Format
//!
//! ```json
//! {"fields": {"title": "Hello World", "views": 42, "score": 4.5}}
//! ```
//!
//! Field values are plain JSON — no type tags. The declared schema type of
//! each field (if any) resolves ambiguity downstream, inside the engine: see
//! [`type_inference::infer_from_json`] for how a bare JSON value maps to a
//! [`DataValue`], and [`type_coercion::coerce_value`] for how that value is
//! then coerced to the field's declared type during ingestion
//! (`Engine::apply_dynamic_schema`). This module only performs the JSON →
//! `DataValue` step; it does not need schema access.
//!
//! [`type_inference`]: super::type_inference
//! [`type_coercion`]: super::type_coercion

use std::collections::HashMap;

use serde_json::Value as JsonValue;

use crate::data::{DataValue, Document};
use crate::error::{LaurusError, Result};

use super::type_inference::{InferredValue, infer_from_json};

/// Convert a JSON value of the shape `{"fields": {...}}` into a [`Document`].
///
/// Each entry in `fields` is run through
/// [`infer_from_json`](super::type_inference::infer_from_json), so the
/// resulting [`DataValue`]s follow the same inference rules as every other
/// JSON-based ingestion path (geo aliases, range checks, multi-valued
/// numerics, the bytes `{data, mime}` shape, mixed-array rejection).
///
/// A field whose value infers to [`InferredValue::Skip`] (JSON `null`, an
/// empty array) is **not** inserted into the resulting document — the
/// caller never sees a placeholder value for it. This differs from the
/// gateway's older `NullValue`-based encoding, which round-tripped `Skip`
/// as an explicit null and could turn a declared `Text` field into an empty
/// string; omitting the field entirely lets `Engine::apply_dynamic_schema`
/// treat it exactly as if the field had not been supplied.
///
/// The per-field [`FieldOption`](super::schema::FieldOption) that
/// `infer_from_json` also produces is discarded here: schema registration
/// for undeclared fields is the engine's responsibility
/// (`Engine::apply_dynamic_schema`, via
/// [`infer_option_from_data_value`](super::type_inference::infer_option_from_data_value)),
/// not the transport layer's.
///
/// # Arguments
///
/// * `json` - The JSON value to convert. Must be an object with a `fields`
///   key whose value is itself an object.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] when `json` is not an object,
/// when it lacks a `fields` key, when `fields` is not an object, or when any
/// individual field value cannot be inferred (the error message is
/// prefixed with `field "<name>": ` to identify the offending field).
pub fn json_to_document(json: &JsonValue) -> Result<Document> {
    let fields_obj = json
        .get("fields")
        .ok_or_else(|| LaurusError::invalid_argument("missing \"fields\" key"))?
        .as_object()
        .ok_or_else(|| LaurusError::invalid_argument("\"fields\" must be an object"))?;

    let mut fields: HashMap<String, DataValue> = HashMap::with_capacity(fields_obj.len());
    for (name, value) in fields_obj {
        match infer_from_json(value).map_err(|e| field_error(name, e))? {
            InferredValue::Skip => {}
            InferredValue::Inferred { value, .. } => {
                fields.insert(name.clone(), value);
            }
        }
    }

    Ok(Document { fields })
}

/// Prefix a per-field inference error with the field name.
///
/// `infer_from_json`'s errors are all built via `LaurusError::invalid_argument`,
/// which renders as `"Invalid argument: <msg>"`. Re-wrapping that rendered
/// string with `invalid_argument` again (i.e. `format!("field \"{name}\":
/// {e}")`) would double the `"Invalid argument: "` prefix, so the inner
/// message is unwrapped from `LaurusError::InvalidArgument` first.
fn field_error(name: &str, e: LaurusError) -> LaurusError {
    let inner = match e {
        LaurusError::InvalidArgument(msg) => msg,
        other => other.to_string(),
    };
    LaurusError::invalid_argument(format!("field \"{name}\": {inner}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn converts_plain_values() {
        let doc = json_to_document(&json!({
            "fields": {
                "title": "Hello World",
                "views": 42,
                "score": 4.5,
                "published": true
            }
        }))
        .unwrap();
        assert_eq!(
            doc.fields.get("title"),
            Some(&DataValue::Text("Hello World".into()))
        );
        assert_eq!(doc.fields.get("views"), Some(&DataValue::Int64(42)));
        assert_eq!(doc.fields.get("score"), Some(&DataValue::Float64(4.5)));
        assert_eq!(doc.fields.get("published"), Some(&DataValue::Bool(true)));
    }

    #[test]
    fn null_field_is_omitted_not_inserted() {
        let doc = json_to_document(&json!({"fields": {"a": "x", "b": null}})).unwrap();
        assert!(doc.fields.contains_key("a"));
        assert!(
            !doc.fields.contains_key("b"),
            "a Skip-inferred field must not appear in the document at all"
        );
    }

    #[test]
    fn empty_array_field_is_omitted() {
        let doc = json_to_document(&json!({"fields": {"tags": []}})).unwrap();
        assert!(!doc.fields.contains_key("tags"));
    }

    #[test]
    fn bytes_object_shape_round_trips() {
        // base64("hi") == "aGk="
        let doc = json_to_document(&json!({
            "fields": {"thumb": {"data": "aGk=", "mime": "image/jpeg"}}
        }))
        .unwrap();
        assert_eq!(
            doc.fields.get("thumb"),
            Some(&DataValue::Bytes(b"hi".to_vec(), Some("image/jpeg".into())))
        );
    }

    #[test]
    fn geo_object_shape_round_trips() {
        let doc =
            json_to_document(&json!({"fields": {"loc": {"lat": 35.1, "lon": 139.0}}})).unwrap();
        assert_eq!(
            doc.fields.get("loc"),
            Some(&DataValue::Geo(crate::data::GeoPoint::new(35.1, 139.0)))
        );
    }

    #[test]
    fn missing_fields_key_rejected() {
        let err = json_to_document(&json!({"id": "x"})).unwrap_err();
        assert!(err.to_string().contains("\"fields\""));
    }

    #[test]
    fn non_object_fields_rejected() {
        let err = json_to_document(&json!({"fields": "not an object"})).unwrap_err();
        assert!(err.to_string().contains("\"fields\" must be an object"));
    }

    #[test]
    fn field_error_includes_field_name() {
        let err =
            json_to_document(&json!({"fields": {"loc": {"lat": 999.0, "lon": 0.0}}})).unwrap_err();
        assert!(err.to_string().contains("\"loc\""));
        assert!(err.to_string().contains("latitude"));
    }

    #[test]
    fn field_error_does_not_double_wrap_the_error_prefix() {
        // `infer_from_json`'s own errors already render as `"Invalid
        // argument: <msg>"`; naively re-wrapping that whole rendered string
        // (`format!("field \"{name}\": {e}")` fed straight into another
        // `invalid_argument`) would duplicate the prefix.
        let err =
            json_to_document(&json!({"fields": {"loc": {"lat": 999.0, "lon": 0.0}}})).unwrap_err();
        assert!(
            matches!(err, LaurusError::InvalidArgument(_)),
            "a bad field value is a caller error: {err:?}"
        );
        let msg = err.to_string();
        assert_eq!(
            msg.matches("Invalid argument:").count(),
            1,
            "message must not double-wrap the \"Invalid argument: \" prefix: {msg}"
        );
    }

    #[test]
    fn old_tagged_format_rejected() {
        // A pre-migration document value like `{"Text": "hello"}` must not
        // be silently misinterpreted — it should error out clearly.
        let err = json_to_document(&json!({"fields": {"title": {"Text": "hello"}}})).unwrap_err();
        assert!(err.to_string().contains("\"title\""));
    }
}
