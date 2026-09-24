//! Conversions between serde_json values and Laurus types.
//!
//! napi-rs v3 with `serde-json` feature automatically converts `serde_json::Value`
//! to/from JS values, so we work at the serde_json level.

use chrono::DateTime;
use laurus::{DataValue, Document};
use serde_json::Value;

/// Convert a `serde_json::Value` (object) to a [`Document`].
///
/// The input must be a JSON object whose keys are field names.
///
/// # Arguments
///
/// * `value` - A JSON object value.
///
/// # Returns
///
/// A [`Document`] with fields populated from the JSON object.
pub fn json_to_document(value: &Value) -> napi::Result<Document> {
    let obj = value
        .as_object()
        .ok_or_else(|| napi::Error::from_reason("Document must be a JSON object"))?;

    let mut builder = Document::builder();
    for (field, val) in obj {
        let dv = json_to_data_value(val)?;
        builder = builder.add_field(field, dv);
    }
    Ok(builder.build())
}

/// Convert a `serde_json::Value` to a [`DataValue`].
///
/// Type mapping:
/// - `null`                  -> `DataValue::Null`
/// - `bool`                  -> `DataValue::Bool`
/// - `number` (integer)      -> `DataValue::Int64`
/// - `number` (float)        -> `DataValue::Float64`
/// - `string`                -> `DataValue::Text` (or `DateTime` if ISO8601)
/// - `array` of integers     -> `DataValue::Int64Array`
/// - `array` of numbers      -> `DataValue::Float64Array` (vector fields
///   cast either array to `Vector` downstream; an empty array is an empty `Int64Array`)
/// - `array` of `{ "lat", "lon" }` -> `DataValue::GeoArray` (multi-valued geo, #1174)
/// - `array` of `{ "x", "y", "z" }` -> `DataValue::GeoEcefArray`
/// - `array` of RFC 3339 strings -> `DataValue::DateTimeArray` (multi-valued datetime, #1184)
/// - `array` of booleans     -> `DataValue::BoolArray` (multi-valued boolean, #1180)
/// - `array` of other strings -> `DataValue::TextArray` (multi-valued text, #1175)
/// - `{ "lat", "lon" }`      -> `DataValue::Geo`
/// - `{ "x", "y", "z" }`     -> `DataValue::GeoEcef` (3D ECEF Cartesian, meters)
///
/// # Arguments
///
/// * `value` - A JSON value.
///
/// # Returns
///
/// The corresponding [`DataValue`].
pub fn json_to_data_value(value: &Value) -> napi::Result<DataValue> {
    match value {
        Value::Null => Ok(DataValue::Null),
        Value::Bool(b) => Ok(DataValue::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(DataValue::Int64(i))
            } else if let Some(f) = n.as_f64() {
                Ok(DataValue::Float64(f))
            } else {
                Err(napi::Error::from_reason("Invalid number value"))
            }
        }
        Value::String(s) => {
            // Try parsing as DateTime
            if let Ok(dt) = s.parse::<DateTime<chrono::Utc>>() {
                return Ok(DataValue::DateTime(dt));
            }
            Ok(DataValue::Text(s.clone()))
        }
        Value::Array(arr) => {
            // Non-empty arrays go through the same inference the server
            // gateway and CLI already use (`laurus::infer_from_json`):
            // all-integer -> `Int64Array`, otherwise numeric -> `Float64Array`,
            // all geo objects -> `GeoArray` / `GeoEcefArray` (#1174),
            // non-numeric elements -> error. The core's schema-aware
            // `coerce_value` then routes the result — vector fields cast it
            // to `Vector`, multi-valued numeric fields keep it, single-valued
            // fields reject it (#1178). Emitting `Vector` here unconditionally
            // made multi-valued numeric fields unreachable from this binding.
            // An empty array becomes an empty `Int64Array`: `coerce_to_vector`
            // casts it to the same empty `Vector` as before, while multi-valued
            // numeric fields — which reject `Vector` outright — now accept it.
            if arr.is_empty() {
                return Ok(DataValue::Int64Array(Vec::new()));
            }
            match laurus::infer_from_json(value)
                .map_err(|e| napi::Error::from_reason(e.to_string()))?
            {
                laurus::InferredValue::Inferred {
                    value: inferred, ..
                } => Ok(inferred),
                laurus::InferredValue::Skip => Ok(DataValue::Vector(Vec::new())),
            }
        }
        Value::Object(obj) => {
            // Check for geo { lat, lon }
            if let (Some(lat), Some(lon)) = (
                obj.get("lat").and_then(|v| v.as_f64()),
                obj.get("lon").and_then(|v| v.as_f64()),
            ) {
                let point = laurus::lexical::GeoPoint::try_new(lat, lon)
                    .map_err(|e| napi::Error::from_reason(format!("invalid geo point: {e}")))?;
                return Ok(DataValue::Geo(point));
            }
            // Check for geo3d { x, y, z } (must come after the 2D Geo check
            // so existing { lat, lon } semantics are preserved).
            if let (Some(x), Some(y), Some(z)) = (
                obj.get("x").and_then(|v| v.as_f64()),
                obj.get("y").and_then(|v| v.as_f64()),
                obj.get("z").and_then(|v| v.as_f64()),
            ) {
                return Ok(DataValue::GeoEcef(laurus::GeoEcefPoint::new(x, y, z)));
            }
            Err(napi::Error::from_reason(
                "Cannot convert JSON object to DataValue: expected { lat, lon } or { x, y, z }",
            ))
        }
    }
}

/// Convert a [`Document`] to a `serde_json::Value`.
///
/// # Arguments
///
/// * `doc` - The document to convert.
///
/// # Returns
///
/// A JSON object with fields from the document.
pub fn document_to_json(doc: &Document) -> Value {
    let mut map = serde_json::Map::new();
    for (field, value) in &doc.fields {
        map.insert(field.clone(), data_value_to_json(value));
    }
    Value::Object(map)
}

/// Convert a [`DataValue`] to a `serde_json::Value`.
///
/// # Arguments
///
/// * `value` - The data value to convert.
///
/// # Returns
///
/// The corresponding JSON value.
pub fn data_value_to_json(value: &DataValue) -> Value {
    match value {
        DataValue::Null => Value::Null,
        DataValue::Bool(b) => Value::Bool(*b),
        DataValue::Int64(i) => serde_json::json!(*i),
        DataValue::Float64(f) => serde_json::json!(*f),
        DataValue::Text(s) => Value::String(s.clone()),
        DataValue::Bytes(b, _) => {
            Value::Array(b.iter().map(|byte| serde_json::json!(*byte)).collect())
        }
        DataValue::Vector(v) => Value::Array(v.iter().map(|f| serde_json::json!(*f)).collect()),
        DataValue::DateTime(dt) => Value::String(dt.to_rfc3339()),
        DataValue::Geo(p) => {
            serde_json::json!({ "lat": p.lat, "lon": p.lon })
        }
        DataValue::GeoEcef(p) => {
            serde_json::json!({ "x": p.x, "y": p.y, "z": p.z })
        }
        DataValue::Int64Array(arr) => {
            Value::Array(arr.iter().map(|v| serde_json::json!(*v)).collect())
        }
        DataValue::Float64Array(arr) => {
            Value::Array(arr.iter().map(|v| serde_json::json!(*v)).collect())
        }
        // Arrays of the same objects the single-valued arms produce, so the
        // output feeds back into `json_to_data_value` unchanged.
        DataValue::GeoArray(arr) => Value::Array(
            arr.iter()
                .map(|p| serde_json::json!({ "lat": p.lat, "lon": p.lon }))
                .collect(),
        ),
        DataValue::GeoEcefArray(arr) => Value::Array(
            arr.iter()
                .map(|p| serde_json::json!({ "x": p.x, "y": p.y, "z": p.z }))
                .collect(),
        ),
        // RFC 3339 strings, which `infer_from_json` reads back as a
        // multi-valued datetime (#1184).
        DataValue::DateTimeArray(arr) => Value::Array(
            arr.iter()
                .map(|dt| Value::String(dt.to_rfc3339()))
                .collect(),
        ),
        // JSON booleans, which `infer_from_json` reads back as a
        // multi-valued boolean (#1180).
        DataValue::BoolArray(arr) => Value::Array(arr.iter().map(|b| Value::Bool(*b)).collect()),
        // JSON strings, which `infer_from_json` reads back as a multi-valued
        // text field (#1175) — or a datetime array if every element is RFC 3339.
        DataValue::TextArray(arr) => {
            Value::Array(arr.iter().map(|s| Value::String(s.clone())).collect())
        }
    }
}
