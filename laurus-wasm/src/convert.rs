//! Conversions between serde_json values and Laurus types, and of the
//! numbers passed to integer parameters.

use chrono::DateTime;
use laurus::{DataValue, Document};
use serde_json::Value;
use wasm_bindgen::JsValue;

/// Convert a number passed to an unsigned integer parameter to a `u32`.
///
/// Such a parameter was a `u32`. The wasm-bindgen glue passed an optional
/// one as `(value) >>> 0` and a required one as it is, for WebAssembly to
/// convert with `ToInt32`; either way only the low 32 bits of the number
/// were kept, without an error, so `-1` became 4294967295, `1.5` became 1,
/// 4294967296 became 0, and `NaN` and the infinities became 0 (Issue
/// #1612). The parameter is now a number, checked here. An option field
/// read with `serde_wasm_bindgen` is a number too, so that its error names
/// the field.
///
/// A number parameter has no type check in the glue, so a non-number
/// argument is converted as JavaScript's `Number()` does: `"5"` becomes 5
/// and `true` 1 (and `null` 0 for a required parameter; an optional one
/// takes `null` as omitted), as a release build's glue did before, while
/// `"abc"` becomes `NaN`, which is rejected here. Only a `--dev` build's
/// glue used to reject a non-number for a 32-bit integer parameter.
///
/// `value.fract() == 0.0` rejects a fraction, and `NaN` and the infinities
/// as well, whose `fract()` is `NaN`; the range rejects the rest. `-0` is
/// taken as 0.
///
/// # Arguments
///
/// * `context` - The method, such as `Schema.addHnswField`, which starts
///   the error message.
/// * `name` - The parameter's JavaScript name, such as `efConstruction`,
///   or an option field's path, such as `highlight.fragmentSize`.
/// * `value` - The number passed.
///
/// # Errors
///
/// Returns an error string such as
/// `Schema.addHnswField: m must be an integer from 0 to 4294967295, got -1`
/// unless `value` is an integer from 0 to 4294967295.
pub fn unsigned_param(context: &str, name: &str, value: f64) -> Result<u32, JsValue> {
    if value.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&value) {
        Ok(value as u32)
    } else {
        Err(JsValue::from_str(&format!(
            "{context}: {name} must be an integer from 0 to {}, got {}",
            u32::MAX,
            js_number(value)
        )))
    }
}

/// [`unsigned_param`] for an optional parameter or option field: `None`
/// (omitted, `undefined` or `null`) stays `None`.
///
/// # Errors
///
/// Returns the error string of [`unsigned_param`].
pub fn optional_unsigned_param(
    context: &str,
    name: &str,
    value: Option<f64>,
) -> Result<Option<u32>, JsValue> {
    value.map(|v| unsigned_param(context, name, v)).transpose()
}

/// Write a number as JavaScript writes `NaN`, `Infinity` and `-Infinity`,
/// where Rust writes `NaN`, `inf` and `-inf`. Other numbers are written by
/// Rust, which writes a large integer in full (`1e21` as 22 digits).
fn js_number(value: f64) -> String {
    if value == f64::INFINITY {
        "Infinity".to_string()
    } else if value == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        value.to_string()
    }
}

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
pub fn json_to_document(value: &Value) -> Result<Document, JsValue> {
    let obj = value
        .as_object()
        .ok_or_else(|| JsValue::from_str("Document must be a JSON object"))?;

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
/// - `array` of other strings -> `DataValue::TextArray` (multi-valued text, #1175); on a
///   multi-valued `Bytes` field the schema-aware coercion further base64-decodes each
///   element into a `DataValue::BytesArray` (#1176) — this function still returns
///   `TextArray`, the same as the scalar `Bytes` case (see `DataValue::Bytes` doc)
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
pub fn json_to_data_value(value: &Value) -> Result<DataValue, JsValue> {
    match value {
        Value::Null => Ok(DataValue::Null),
        Value::Bool(b) => Ok(DataValue::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(DataValue::Int64(i))
            } else if let Some(f) = n.as_f64() {
                Ok(DataValue::Float64(f))
            } else {
                Err(JsValue::from_str("Invalid number value"))
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
            match laurus::infer_from_json(value).map_err(|e| JsValue::from_str(&e.to_string()))? {
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
                    .map_err(|e| JsValue::from_str(&format!("invalid geo point: {e}")))?;
                return Ok(DataValue::Geo(point));
            }
            // Check for geo3d { x, y, z } (must come after the {lat, lon}
            // check so existing 2D Geo semantics are preserved).
            if let (Some(x), Some(y), Some(z)) = (
                obj.get("x").and_then(|v| v.as_f64()),
                obj.get("y").and_then(|v| v.as_f64()),
                obj.get("z").and_then(|v| v.as_f64()),
            ) {
                return Ok(DataValue::GeoEcef(laurus::GeoEcefPoint::new(x, y, z)));
            }
            Err(JsValue::from_str(
                "Cannot convert JSON object to DataValue: expected { lat, lon } or { x, y, z }",
            ))
        }
    }
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
        // Each element rendered the same way the scalar `Bytes` arm above
        // renders it (mime dropped; #1176). Not expected to round-trip back
        // into a `BytesArray` through `json_to_data_value` — same
        // intentional asymmetry as the scalar case, which also renders as a
        // byte-integer array but only accepts base64 text as input.
        DataValue::BytesArray(arr) => Value::Array(
            arr.iter()
                .map(|(data, _mime)| {
                    Value::Array(data.iter().map(|b| serde_json::json!(*b)).collect())
                })
                .collect(),
        ),
        // Nested number arrays, which `infer_from_json` reads back as a
        // MultiVector value (#1177).
        DataValue::VectorArray(arr) => serde_json::json!(arr),
    }
}

/// Values and the check shared by the tests of the integer parameters
/// (Issue #1612).
#[cfg(test)]
pub(crate) mod unsigned_param_testing {
    use wasm_bindgen::JsValue;

    /// A negative number, a fraction, a number above 4294967295, `NaN` and
    /// an infinity, each with the way the error message writes it. The
    /// glue turned each into another number before Issue #1612.
    pub(crate) const BAD: [(f64, &str); 5] = [
        (-1.0, "-1"),
        (1.5, "1.5"),
        (4_294_967_296.0, "4294967296"),
        (f64::NAN, "NaN"),
        (f64::INFINITY, "Infinity"),
    ];

    /// Record in `mismatches` a call, passing a value of [`BAD`] (written
    /// as `written`) to `name` of `context`, that did not fail with the
    /// expected message.
    pub(crate) fn check<T>(
        mismatches: &mut Vec<String>,
        context: &str,
        name: &str,
        (_, written): (f64, &str),
        result: Result<T, JsValue>,
    ) {
        let expected =
            format!("{context}: {name} must be an integer from 0 to 4294967295, got {written}");
        match result {
            Ok(_) => mismatches.push(format!("{context} {name} {written}: no error")),
            Err(e) => {
                let got = e.as_string().unwrap_or_else(|| format!("{e:?}"));
                if got != expected {
                    mismatches.push(format!("{context} {name} {written}: {got}"));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wasm_bindgen_test::wasm_bindgen_test;

    /// Issue #1612: an integer from 0 to 4294967295 converts as it is, and
    /// `-0` is 0.
    #[wasm_bindgen_test]
    fn unsigned_param_takes_an_integer_from_0_to_u32_max() {
        for (value, expected) in [(0.0, 0), (-0.0, 0), (1.0, 1), (4_294_967_295.0, u32::MAX)] {
            assert_eq!(
                unsigned_param("C", "n", value).unwrap(),
                expected,
                "{value}"
            );
        }
        assert_eq!(optional_unsigned_param("C", "n", None).unwrap(), None);
        assert_eq!(
            optional_unsigned_param("C", "n", Some(7.0)).unwrap(),
            Some(7)
        );
    }

    /// Issue #1612: anything else fails with a message naming the context
    /// and the parameter and writing the value as JavaScript does.
    #[wasm_bindgen_test]
    fn unsigned_param_rejects_a_value_that_is_not_an_integer_from_0_to_u32_max() {
        use super::unsigned_param_testing::{BAD, check};

        let mut mismatches = Vec::new();
        let more = [
            (f64::NEG_INFINITY, "-Infinity"),
            (-0.5, "-0.5"),
            (4_294_967_295.5, "4294967295.5"),
        ];
        for case in BAD.into_iter().chain(more) {
            check(
                &mut mismatches,
                "C",
                "n",
                case,
                unsigned_param("C", "n", case.0),
            );
            check(
                &mut mismatches,
                "C",
                "n",
                case,
                optional_unsigned_param("C", "n", Some(case.0)),
            );
        }
        assert!(mismatches.is_empty(), "{mismatches:#?}");
    }

    /// Issue #1184: an array of RFC 3339 strings is a multi-valued datetime
    /// and renders back as RFC 3339 strings normalized to UTC.
    #[wasm_bindgen_test]
    fn datetime_string_array_round_trips_as_datetime_array() {
        let json = json!(["2024-01-01T00:00:00Z", "2024-06-15T21:00:00+09:00"]);
        let dv = json_to_data_value(&json).unwrap();
        match &dv {
            DataValue::DateTimeArray(v) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[1].timestamp(), 1_718_452_800);
            }
            other => panic!("expected DateTimeArray, got {other:?}"),
        }
        assert_eq!(
            data_value_to_json(&dv),
            json!(["2024-01-01T00:00:00+00:00", "2024-06-15T12:00:00+00:00"])
        );
    }

    /// Issue #1175: a string array that is not entirely RFC 3339 is a
    /// multi-valued text value and renders back as the same strings.
    #[wasm_bindgen_test]
    fn non_datetime_string_array_becomes_text_array() {
        for json in [
            json!(["2024-01-01T00:00:00Z", "tomorrow"]),
            json!(["a", "b c"]),
        ] {
            let dv = json_to_data_value(&json).unwrap();
            assert!(
                matches!(&dv, DataValue::TextArray(v) if v.len() == 2),
                "{json}: {dv:?}"
            );
            assert_eq!(data_value_to_json(&dv), json);
        }
    }

    /// Issue #1178: arrays used to become `Vector` unconditionally, which the
    /// core's multi-valued integer/float coercion rejects. They must now
    /// arrive as the most informative numeric array shape instead.
    #[wasm_bindgen_test]
    fn integer_array_becomes_int64_array() {
        match json_to_data_value(&json!([1, 2, 3])) {
            Ok(DataValue::Int64Array(v)) => assert_eq!(v, vec![1, 2, 3]),
            other => panic!("expected Int64Array, got {other:?}"),
        }
    }

    #[wasm_bindgen_test]
    fn float_array_becomes_float64_array() {
        match json_to_data_value(&json!([1.5, 2.0])) {
            Ok(DataValue::Float64Array(v)) => assert_eq!(v, vec![1.5, 2.0]),
            other => panic!("expected Float64Array, got {other:?}"),
        }
    }

    /// One non-integer element widens the whole array to floats.
    #[wasm_bindgen_test]
    fn mixed_array_becomes_float64_array() {
        match json_to_data_value(&json!([1, 2.5])) {
            Ok(DataValue::Float64Array(v)) => assert_eq!(v, vec![1.0, 2.5]),
            other => panic!("expected Float64Array, got {other:?}"),
        }
    }

    /// An empty array is an empty `Int64Array`: vector fields cast it to the
    /// same empty `Vector` as before, and multi-valued numeric fields (which
    /// reject `Vector`) now accept it.
    #[wasm_bindgen_test]
    fn empty_array_becomes_empty_int64_array() {
        match json_to_data_value(&json!([])) {
            Ok(DataValue::Int64Array(v)) => assert!(v.is_empty()),
            other => panic!("expected empty Int64Array, got {other:?}"),
        }
    }

    #[wasm_bindgen_test]
    fn non_numeric_array_is_rejected() {
        assert!(json_to_data_value(&json!([1, "x"])).is_err());
    }

    /// Issue #1174: an array of geo objects is a multi-valued geo value and
    /// renders back as the same array of `{ lat, lon }` objects.
    #[wasm_bindgen_test]
    fn geo_object_array_round_trips_as_geo_array() {
        let json = json!([{ "lat": 35.68, "lon": 139.76 }, { "lat": 34.69, "lon": 135.5 }]);
        let dv = json_to_data_value(&json).unwrap();
        match &dv {
            DataValue::GeoArray(pts) => {
                assert_eq!(pts.len(), 2);
                assert_eq!(pts[1].lon, 135.5);
            }
            other => panic!("expected GeoArray, got {other:?}"),
        }
        assert_eq!(data_value_to_json(&dv), json);
    }

    #[wasm_bindgen_test]
    fn geo3d_object_array_round_trips_as_geo_ecef_array() {
        let json = json!([{ "x": 1.0, "y": 2.0, "z": 3.0 }, { "x": -4.0, "y": 5.0, "z": -6.0 }]);
        let dv = json_to_data_value(&json).unwrap();
        assert!(matches!(&dv, DataValue::GeoEcefArray(pts) if pts.len() == 2));
        assert_eq!(data_value_to_json(&dv), json);
    }

    #[wasm_bindgen_test]
    fn mixed_geo_dimension_array_is_rejected() {
        assert!(
            json_to_data_value(
                &json!([{ "lat": 35.0, "lon": 139.0 }, { "x": 1.0, "y": 2.0, "z": 3.0 }])
            )
            .is_err()
        );
    }

    /// Issue #1180: an array of booleans is a multi-valued boolean and
    /// renders back as the same array.
    #[wasm_bindgen_test]
    fn bool_array_round_trips_as_bool_array() {
        let json = json!([true, false]);
        let dv = json_to_data_value(&json).unwrap();
        assert_eq!(dv, DataValue::BoolArray(vec![true, false]));
        assert_eq!(data_value_to_json(&dv), json);
    }

    #[wasm_bindgen_test]
    fn mixed_bool_and_number_array_is_rejected() {
        assert!(json_to_data_value(&json!([true, 1])).is_err());
    }

    /// Issue #1176: a `BytesArray` renders as an array of byte-integer
    /// arrays, each element in the same shape the scalar `Bytes` arm
    /// renders (mime dropped). This is not expected to round-trip back
    /// through `json_to_data_value` — the scalar `Bytes` case has the same
    /// asymmetry, accepting base64 text as input but rendering as an
    /// integer array on output.
    #[wasm_bindgen_test]
    fn bytes_array_renders_as_array_of_byte_arrays() {
        let dv = DataValue::BytesArray(vec![
            (vec![1, 2, 3], None),
            (vec![4, 5], Some("image/png".to_string())),
        ]);
        assert_eq!(data_value_to_json(&dv), json!([[1, 2, 3], [4, 5]]));
    }
}
