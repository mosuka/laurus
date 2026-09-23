//! Type inference for the dynamic schema feature.
//!
//! When a document is ingested against a schema with
//! [`DynamicFieldPolicy::Dynamic`](super::schema::DynamicFieldPolicy::Dynamic),
//! this module infers a [`FieldOption`] for each undeclared field from the
//! value that the user provided.
//!
//! Two entry points cover the two ingestion paths:
//!
//! - [`infer_option_from_data_value`] — engine-side path; the document already
//!   carries a [`DataValue`] (gRPC `add_document`, native bindings).
//! - [`infer_from_json`] — transport-side path; used by the HTTP gateway so
//!   JSON inputs follow the same inference rules as the engine.
//!
//! Supported inferences:
//!
//! - `string` → [`FieldOption::Text`]
//! - `integer` → [`FieldOption::Integer`]
//! - `float` → [`FieldOption::Float`]
//! - `bool` → [`FieldOption::Boolean`]
//! - numeric array (all `i64`) → [`FieldOption::Integer`] with `multi_valued = true`
//! - numeric array (any non-`i64` number) → [`FieldOption::Float`] with `multi_valued = true`
//! - `object` with `lat|latitude` and `lon|lng|longitude` keys (values in
//!   range) → [`FieldOption::Geo`]
//! - `object` with all three numeric keys `x`, `y`, `z` → [`FieldOption::Geo3d`]
//! - `object` with a `data` key (base64-encoded string, optional `mime`
//!   string) → [`DataValue::Bytes`]
//!
//! Vector fields are never inferred; a numeric array always maps to a
//! multi-valued Integer/Float field, never to [`DataValue::Vector`]. This is
//! about **value parsing** only, not schema auto-registration: even though
//! this module can now parse the `{data, mime}` bytes shape, an *undeclared*
//! field carrying a [`DataValue::Bytes`] value is still rejected by
//! [`infer_option_from_data_value`] — bytes fields must be declared
//! explicitly in the schema (dimension-free, but still explicit, unlike
//! Text/Integer/etc. which can be auto-registered). Mixing geographic keys
//! (`lat`/`lon`, `x`/`y`/`z`) with the bytes `data` key in the same object is
//! rejected as ambiguous.

use base64::Engine as _;
use serde_json::Value as JsonValue;

use crate::data::DataValue;
use crate::error::{LaurusError, Result};
use crate::lexical::core::field::{
    BooleanOption, BytesOption, FloatOption, GeoOption, IntegerOption, TextOption,
};

use super::schema::FieldOption;

/// Result of attempting to infer a [`DataValue`] and [`FieldOption`] from a
/// raw JSON value.
#[derive(Debug, Clone)]
pub enum InferredValue {
    /// The value was inferred successfully.
    Inferred {
        /// The converted [`DataValue`] to store in the document.
        value: DataValue,
        /// The [`FieldOption`] to register in the schema if the field is new.
        option: FieldOption,
    },
    /// The value is known but should be silently skipped (e.g. `null`, empty
    /// array). Callers should not add the field.
    Skip,
}

/// Infer a [`FieldOption`] from an existing [`DataValue`].
///
/// Used during document ingestion when the schema's
/// [`DynamicFieldPolicy`](super::schema::DynamicFieldPolicy) is `Dynamic` and
/// the field has not been declared yet. The document already carries a
/// [`DataValue`] (the JSON → [`DataValue`] conversion happened earlier, at the
/// transport layer), so this entry point does not re-parse the value.
///
/// Supported variants:
///
/// - [`DataValue::Text`] → [`FieldOption::Text`]
/// - [`DataValue::Int64`] → [`FieldOption::Integer`]
/// - [`DataValue::Float64`] → [`FieldOption::Float`]
/// - [`DataValue::Bool`] → [`FieldOption::Boolean`]
/// - [`DataValue::DateTime`] → [`FieldOption::DateTime`]
/// - [`DataValue::Geo`] → [`FieldOption::Geo`]
/// - [`DataValue::Null`] → `Ok(None)` (caller should skip the field)
///
/// # Arguments
///
/// * `value` - The data value to inspect.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] for variants that are not
/// supported by the dynamic schema:
///
/// - [`DataValue::Vector`]: vector fields must be declared explicitly.
/// - [`DataValue::Bytes`]: bytes fields must be declared explicitly.
pub fn infer_option_from_data_value(value: &DataValue) -> Result<Option<FieldOption>> {
    match value {
        DataValue::Null => Ok(None),
        DataValue::Text(_) => Ok(Some(FieldOption::Text(TextOption::default()))),
        DataValue::Int64(_) => Ok(Some(FieldOption::Integer(IntegerOption::default()))),
        DataValue::Float64(_) => Ok(Some(FieldOption::Float(FloatOption::default()))),
        DataValue::Bool(_) => Ok(Some(FieldOption::Boolean(BooleanOption::default()))),
        DataValue::DateTime(_) => Ok(Some(FieldOption::DateTime(
            crate::lexical::core::field::DateTimeOption::default(),
        ))),
        DataValue::Geo(_) => Ok(Some(FieldOption::Geo(GeoOption::default()))),
        DataValue::GeoEcef(_) => Ok(Some(FieldOption::Geo3d(
            crate::lexical::core::field::Geo3dOption::default(),
        ))),
        DataValue::Int64Array(_) => Ok(Some(FieldOption::Integer(IntegerOption {
            multi_valued: true,
            ..Default::default()
        }))),
        DataValue::Float64Array(_) => Ok(Some(FieldOption::Float(FloatOption {
            multi_valued: true,
            ..Default::default()
        }))),
        DataValue::GeoArray(_) => Ok(Some(FieldOption::Geo(GeoOption {
            multi_valued: true,
            ..Default::default()
        }))),
        DataValue::GeoEcefArray(_) => Ok(Some(FieldOption::Geo3d(
            crate::lexical::core::field::Geo3dOption {
                multi_valued: true,
                ..Default::default()
            },
        ))),
        DataValue::DateTimeArray(_) => Ok(Some(FieldOption::DateTime(
            crate::lexical::core::field::DateTimeOption {
                multi_valued: true,
                ..Default::default()
            },
        ))),
        DataValue::Vector(_) => Err(LaurusError::invalid_argument(
            "vector values require an explicit vector field declaration \
             (Hnsw, Flat, or Ivf) in the schema",
        )),
        DataValue::Bytes(_, _) => Err(LaurusError::invalid_argument(
            "bytes values require an explicit bytes field declaration in the schema",
        )),
    }
}

/// Infer a [`DataValue`] and [`FieldOption`] from a JSON value.
///
/// The mapping is:
///
/// | JSON value | DataValue | FieldOption |
/// | --- | --- | --- |
/// | `string` | [`DataValue::Text`] | [`FieldOption::Text`] |
/// | `integer` (fits in i64) | [`DataValue::Int64`] | [`FieldOption::Integer`] |
/// | `float` / large number | [`DataValue::Float64`] | [`FieldOption::Float`] |
/// | `bool` | [`DataValue::Bool`] | [`FieldOption::Boolean`] |
/// | `object` with `lat|latitude` + `lon|lng|longitude` | [`DataValue::Geo`] | [`FieldOption::Geo`] |
/// | `object` with `x` + `y` + `z` (all numeric) | [`DataValue::GeoEcef`] | [`FieldOption::Geo3d`] |
/// | `object` with `data` (base64 string) + optional `mime` | [`DataValue::Bytes`] | [`FieldOption::Bytes`] |
/// | `null` | (none) | (none) — returns [`InferredValue::Skip`] |
/// | `array` of integers | [`DataValue::Int64Array`] | [`FieldOption::Integer`] with `multi_valued = true` |
/// | `array` containing any non-i64 number | [`DataValue::Float64Array`] | [`FieldOption::Float`] with `multi_valued = true` |
/// | `array` of `lat`/`lon` objects | [`DataValue::GeoArray`] | [`FieldOption::Geo`] with `multi_valued = true` |
/// | `array` of `x`/`y`/`z` objects | [`DataValue::GeoEcefArray`] | [`FieldOption::Geo3d`] with `multi_valued = true` |
/// | `array` of RFC 3339 strings | [`DataValue::DateTimeArray`] | [`FieldOption::DateTime`] with `multi_valued = true` (a *single* string stays `Text`) |
/// | empty `array` | (none) | (none) — returns [`InferredValue::Skip`] |
///
/// # Arguments
///
/// * `value` - The JSON value to infer from.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] if the value cannot be inferred
/// (e.g. mixed-type array, non-geo object, out-of-range geo values).
pub fn infer_from_json(value: &JsonValue) -> Result<InferredValue> {
    match value {
        JsonValue::Null => Ok(InferredValue::Skip),
        JsonValue::Bool(b) => Ok(InferredValue::Inferred {
            value: DataValue::Bool(*b),
            option: FieldOption::Boolean(BooleanOption::default()),
        }),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(InferredValue::Inferred {
                    value: DataValue::Int64(i),
                    option: FieldOption::Integer(IntegerOption::default()),
                })
            } else if let Some(f) = n.as_f64() {
                Ok(InferredValue::Inferred {
                    value: DataValue::Float64(f),
                    option: FieldOption::Float(FloatOption::default()),
                })
            } else {
                Err(LaurusError::invalid_argument(format!(
                    "number {n} cannot be represented as i64 or f64"
                )))
            }
        }
        JsonValue::String(s) => Ok(InferredValue::Inferred {
            value: DataValue::Text(s.clone()),
            option: FieldOption::Text(TextOption::default()),
        }),
        JsonValue::Array(arr) => infer_from_array(arr),
        JsonValue::Object(map) => infer_from_object(map),
    }
}

/// Infer a value from a JSON array.
///
/// Numeric-only arrays become multi-valued numeric fields. Arrays whose
/// elements all fit in `i64` map to [`DataValue::Int64Array`] backed by an
/// [`IntegerOption`] with `multi_valued = true`. Arrays containing any
/// non-`i64` number map to [`DataValue::Float64Array`] backed by a
/// [`FloatOption`] with `multi_valued = true`. Arrays whose elements are all
/// geo objects — every element inferring, via [`infer_from_object`], to the
/// same one of [`DataValue::Geo`] / [`DataValue::GeoEcef`] — map to
/// [`DataValue::GeoArray`] / [`DataValue::GeoEcefArray`] backed by a
/// [`GeoOption`] / `Geo3dOption` with `multi_valued = true` (Issue #1174).
/// Arrays whose elements are all RFC 3339 strings map to
/// [`DataValue::DateTimeArray`] backed by a `DateTimeOption` with
/// `multi_valued = true` (Issue #1184; a single string still infers
/// [`DataValue::Text`]). Empty arrays return [`InferredValue::Skip`]
/// because their element type cannot be determined.
///
/// # Arguments
///
/// * `arr` - The JSON array to inspect.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] when the array mixes numbers
/// with non-numbers, mixes 2D and 3D geo objects, contains an object that
/// is not a geo point, or contains a string that is not an RFC 3339
/// datetime.
fn infer_from_array(arr: &[JsonValue]) -> Result<InferredValue> {
    if arr.is_empty() {
        return Ok(InferredValue::Skip);
    }

    // Gated on *every* element being an object (or, below, a string) so a
    // mixed array such as `[1, {"lat": ..}]` still falls through to the
    // numeric path's error below, which existing callers and tests rely on.
    if arr.iter().all(JsonValue::is_object) {
        return infer_geo_array(arr);
    }
    if arr.iter().all(JsonValue::is_string) {
        return infer_datetime_array(arr);
    }

    let mut all_i64 = true;
    let mut all_numeric = true;
    for elem in arr {
        match elem {
            JsonValue::Number(n) => {
                if n.as_i64().is_none() {
                    all_i64 = false;
                }
            }
            _ => {
                all_numeric = false;
                break;
            }
        }
    }

    if !all_numeric {
        return Err(LaurusError::invalid_argument(
            "array fields must contain only numeric values \
             (mixed or non-numeric arrays are not supported)",
        ));
    }

    if all_i64 {
        let values: Vec<i64> = arr
            .iter()
            .map(|v| v.as_i64().expect("checked above"))
            .collect();
        Ok(InferredValue::Inferred {
            value: DataValue::Int64Array(values),
            option: FieldOption::Integer(IntegerOption {
                multi_valued: true,
                ..Default::default()
            }),
        })
    } else {
        let values: Vec<f64> = arr
            .iter()
            .map(|v| {
                v.as_f64()
                    .expect("numeric JSON values are always representable as f64")
            })
            .collect();
        Ok(InferredValue::Inferred {
            value: DataValue::Float64Array(values),
            option: FieldOption::Float(FloatOption {
                multi_valued: true,
                ..Default::default()
            }),
        })
    }
}

/// Infer a multi-valued datetime field from an array whose elements are all
/// strings (Issue #1184). Every element must be an RFC 3339 datetime — the
/// same grammar the single-value coercion accepts — and offsets are
/// normalized to UTC. This is deliberately the only string shape that
/// infers a `DateTime`: a single string keeps inferring `Text` (changing
/// that would re-type existing Text fields under the Dynamic policy), and
/// there is no multi-valued text field for other string arrays to fall
/// back to yet (#1175).
fn infer_datetime_array(arr: &[JsonValue]) -> Result<InferredValue> {
    let mut values = Vec::with_capacity(arr.len());
    for elem in arr {
        let s = elem.as_str().expect("gated on all-strings by the caller");
        let dt = chrono::DateTime::parse_from_rfc3339(s.trim()).map_err(|e| {
            LaurusError::invalid_argument(format!(
                "array of strings must be all RFC 3339 datetimes \
                 (multi-valued text fields are not supported): {s:?}: {e}"
            ))
        })?;
        values.push(dt.with_timezone(&chrono::Utc));
    }
    Ok(InferredValue::Inferred {
        value: DataValue::DateTimeArray(values),
        option: FieldOption::DateTime(crate::lexical::core::field::DateTimeOption {
            multi_valued: true,
            ..Default::default()
        }),
    })
}

/// Infer a multi-valued geo field from an array whose elements are all
/// objects (Issue #1174). Each element goes through [`infer_from_object`],
/// so key aliases, range validation, and the mixed-marker rejection are
/// exactly those of a single-valued geo object.
fn infer_geo_array(arr: &[JsonValue]) -> Result<InferredValue> {
    let mixed = || {
        LaurusError::invalid_argument(
            "array of objects must be all 2D geo points or all 3D ECEF points",
        )
    };
    let mut geo: Vec<crate::data::GeoPoint> = Vec::with_capacity(arr.len());
    let mut ecef: Vec<crate::data::GeoEcefPoint> = Vec::with_capacity(arr.len());
    for elem in arr {
        let map = elem
            .as_object()
            .expect("gated on all-objects by the caller");
        match infer_from_object(map)? {
            InferredValue::Inferred {
                value: DataValue::Geo(p),
                ..
            } => geo.push(p),
            InferredValue::Inferred {
                value: DataValue::GeoEcef(p),
                ..
            } => ecef.push(p),
            _ => return Err(mixed()),
        }
    }
    match (geo.is_empty(), ecef.is_empty()) {
        (false, true) => Ok(InferredValue::Inferred {
            value: DataValue::GeoArray(geo),
            option: FieldOption::Geo(GeoOption {
                multi_valued: true,
                ..Default::default()
            }),
        }),
        (true, false) => Ok(InferredValue::Inferred {
            value: DataValue::GeoEcefArray(ecef),
            option: FieldOption::Geo3d(crate::lexical::core::field::Geo3dOption {
                multi_valued: true,
                ..Default::default()
            }),
        }),
        _ => Err(mixed()),
    }
}

/// Infer a value from a JSON object.
///
/// Three object shapes are accepted:
///
/// - **2D geographic point** ([`DataValue::Geo`]): an object with a latitude
///   key (`lat` or `latitude`) and a longitude key (`lon`, `lng`, or
///   `longitude`). Both values must be numeric, with latitude in `[-90, 90]`
///   and longitude in `[-180, 180]`.
/// - **3D ECEF point** ([`DataValue::GeoEcef`]): an object with all three
///   numeric keys `x`, `y`, `z` (meters from the Earth's centre, in the
///   ECEF Cartesian frame). Coordinates must be finite (no `NaN` / `inf`);
///   no range check is applied because ECEF values are unbounded.
/// - **Bytes** ([`DataValue::Bytes`]): an object with a `data` key holding a
///   base64-encoded string, and an optional `mime` string key. This is the
///   disambiguating shape for multimodal vector fields (e.g. an `Hnsw` field
///   backed by an image embedder), where a bare base64 string would be
///   ambiguous between "text to embed" and "image bytes to decode" — see
///   [`crate::engine::type_coercion::coerce_value`] for the field-level
///   Bytes/Vector coercion rules that consume this shape.
///
/// Mixing markers from more than one of these shapes in the same object
/// (e.g. supplying both `lat` and `x`, or `lat` and `data`) is rejected as
/// ambiguous. Other object shapes — including partial key sets like `{lat}`
/// alone or `{x, y}` without `z` — are rejected with the generic
/// "geographic points" error.
///
/// # Arguments
///
/// * `map` - The JSON object entries.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] when the object is not a
/// valid 2D/3D geographic point or bytes value, when markers from more than
/// one shape are mixed, when a coordinate is non-numeric or out of range,
/// when an ECEF coordinate is not finite, or when `data` is not a valid
/// base64 string.
fn infer_from_object(map: &serde_json::Map<String, JsonValue>) -> Result<InferredValue> {
    const LAT_KEYS: &[&str] = &["lat", "latitude"];
    const LON_KEYS: &[&str] = &["lon", "lng", "longitude"];

    let lat_val = LAT_KEYS.iter().find_map(|k| map.get(*k));
    let lon_val = LON_KEYS.iter().find_map(|k| map.get(*k));
    let x_val = map.get("x");
    let y_val = map.get("y");
    let z_val = map.get("z");
    let data_val = map.get("data");

    let has_2d_keys = lat_val.is_some() || lon_val.is_some();
    let has_3d_keys = x_val.is_some() || y_val.is_some() || z_val.is_some();
    let has_bytes_key = data_val.is_some();

    // Reject ambiguous mixing of markers from more than one shape.
    if (has_2d_keys as u8 + has_3d_keys as u8 + has_bytes_key as u8) > 1 {
        return Err(LaurusError::invalid_argument(
            "object cannot mix geographic keys (lat/lon, x/y/z) with a bytes \"data\" key; \
             use exactly one of {lat, lon}, {x, y, z}, or {data, mime?}",
        ));
    }

    // Bytes path.
    if let Some(data_val) = data_val {
        let encoded = data_val.as_str().ok_or_else(|| {
            LaurusError::invalid_argument("bytes \"data\" must be a base64-encoded string")
        })?;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|e| {
                LaurusError::invalid_argument(format!("bytes \"data\" is not valid base64: {e}"))
            })?;
        let mime = match map.get("mime") {
            None | Some(JsonValue::Null) => None,
            Some(JsonValue::String(s)) => Some(s.clone()),
            Some(_) => {
                return Err(LaurusError::invalid_argument(
                    "bytes \"mime\" must be a string",
                ));
            }
        };
        return Ok(InferredValue::Inferred {
            value: DataValue::Bytes(decoded, mime),
            option: FieldOption::Bytes(BytesOption::default()),
        });
    }

    // 2D Geo path.
    if let (Some(lat_val), Some(lon_val)) = (lat_val, lon_val) {
        let lat = lat_val
            .as_f64()
            .ok_or_else(|| LaurusError::invalid_argument("geo latitude must be a number"))?;
        let lon = lon_val
            .as_f64()
            .ok_or_else(|| LaurusError::invalid_argument("geo longitude must be a number"))?;
        if !(-90.0..=90.0).contains(&lat) {
            return Err(LaurusError::invalid_argument(format!(
                "geo latitude {lat} is out of range [-90, 90]"
            )));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(LaurusError::invalid_argument(format!(
                "geo longitude {lon} is out of range [-180, 180]"
            )));
        }
        return Ok(InferredValue::Inferred {
            value: DataValue::Geo(crate::data::GeoPoint::new(lat, lon)),
            option: FieldOption::Geo(GeoOption::default()),
        });
    }

    // 3D Geo3d path: require all three of x, y, z.
    if let (Some(x_val), Some(y_val), Some(z_val)) = (x_val, y_val, z_val) {
        let x = x_val
            .as_f64()
            .ok_or_else(|| LaurusError::invalid_argument("Geo3d x must be a number"))?;
        let y = y_val
            .as_f64()
            .ok_or_else(|| LaurusError::invalid_argument("Geo3d y must be a number"))?;
        let z = z_val
            .as_f64()
            .ok_or_else(|| LaurusError::invalid_argument("Geo3d z must be a number"))?;
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(LaurusError::invalid_argument(
                "Geo3d coordinates (x, y, z) must be finite numbers",
            ));
        }
        return Ok(InferredValue::Inferred {
            value: DataValue::GeoEcef(crate::data::GeoEcefPoint::new(x, y, z)),
            option: FieldOption::Geo3d(crate::lexical::core::field::Geo3dOption::default()),
        });
    }

    // Partial keys (e.g. only `lat`, or `x`+`y` without `z`) or unrelated object.
    Err(LaurusError::invalid_argument(
        "object values are only supported as geographic points or bytes \
         (expected keys: lat|latitude, lon|lng|longitude for 2D geo; \
         x+y+z for 3D ECEF geo; or data (+ optional mime) for bytes)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn inferred(v: InferredValue) -> (DataValue, FieldOption) {
        match v {
            InferredValue::Inferred { value, option } => (value, option),
            InferredValue::Skip => panic!("expected Inferred, got Skip"),
        }
    }

    #[test]
    fn infer_string_to_text() {
        let (v, o) = inferred(infer_from_json(&json!("hello")).unwrap());
        assert_eq!(v, DataValue::Text("hello".into()));
        assert!(matches!(o, FieldOption::Text(_)));
    }

    #[test]
    fn infer_integer_to_integer() {
        let (v, o) = inferred(infer_from_json(&json!(42)).unwrap());
        assert_eq!(v, DataValue::Int64(42));
        assert!(matches!(o, FieldOption::Integer(_)));
    }

    #[test]
    fn infer_negative_integer() {
        let (v, o) = inferred(infer_from_json(&json!(-7)).unwrap());
        assert_eq!(v, DataValue::Int64(-7));
        assert!(matches!(o, FieldOption::Integer(_)));
    }

    #[test]
    fn infer_float_to_float() {
        let (v, o) = inferred(infer_from_json(&json!(4.5)).unwrap());
        assert_eq!(v, DataValue::Float64(4.5));
        assert!(matches!(o, FieldOption::Float(_)));
    }

    #[test]
    fn infer_bool_to_boolean() {
        let (v, o) = inferred(infer_from_json(&json!(true)).unwrap());
        assert_eq!(v, DataValue::Bool(true));
        assert!(matches!(o, FieldOption::Boolean(_)));
    }

    #[test]
    fn infer_null_skips() {
        assert!(matches!(
            infer_from_json(&JsonValue::Null).unwrap(),
            InferredValue::Skip
        ));
    }

    #[test]
    fn infer_empty_array_skips() {
        assert!(matches!(
            infer_from_json(&json!([])).unwrap(),
            InferredValue::Skip
        ));
    }

    #[test]
    fn infer_integer_array_to_int64_array() {
        let (v, o) = inferred(infer_from_json(&json!([1, 2, 3])).unwrap());
        assert_eq!(v, DataValue::Int64Array(vec![1, 2, 3]));
        match o {
            FieldOption::Integer(opt) => assert!(opt.multi_valued),
            other => panic!("expected Integer with multi_valued=true, got {other:?}"),
        }
    }

    #[test]
    fn infer_float_array_to_float64_array() {
        // Mixed integer + float (any non-i64 element flips to float array)
        let (v, o) = inferred(infer_from_json(&json!([1.0, 2.5, 3])).unwrap());
        assert_eq!(v, DataValue::Float64Array(vec![1.0, 2.5, 3.0]));
        match o {
            FieldOption::Float(opt) => assert!(opt.multi_valued),
            other => panic!("expected Float with multi_valued=true, got {other:?}"),
        }
    }

    #[test]
    fn infer_mixed_array_rejected() {
        let err = infer_from_json(&json!([1, "a"])).unwrap_err();
        assert!(err.to_string().contains("only numeric"));
    }

    #[test]
    fn infer_geo_lat_lon() {
        let (v, o) = inferred(infer_from_json(&json!({"lat": 35.1, "lon": 139.0})).unwrap());
        assert_eq!(v, DataValue::Geo(crate::data::GeoPoint::new(35.1, 139.0)));
        assert!(matches!(o, FieldOption::Geo(_)));
    }

    #[test]
    fn infer_geo_latitude_longitude() {
        let (v, _) =
            inferred(infer_from_json(&json!({"latitude": 35.1, "longitude": 139.0})).unwrap());
        assert_eq!(v, DataValue::Geo(crate::data::GeoPoint::new(35.1, 139.0)));
    }

    #[test]
    fn infer_geo_lng_alias() {
        let (v, _) = inferred(infer_from_json(&json!({"lat": 35.1, "lng": 139.0})).unwrap());
        assert_eq!(v, DataValue::Geo(crate::data::GeoPoint::new(35.1, 139.0)));
    }

    #[test]
    fn infer_geo_out_of_range_lat() {
        let err = infer_from_json(&json!({"lat": 100.0, "lon": 139.0})).unwrap_err();
        assert!(err.to_string().contains("latitude"));
    }

    #[test]
    fn infer_geo_out_of_range_lon() {
        let err = infer_from_json(&json!({"lat": 35.1, "lon": 200.0})).unwrap_err();
        assert!(err.to_string().contains("longitude"));
    }

    #[test]
    fn infer_option_from_data_value_geo_ecef_to_geo3d() {
        // The dynamic-schema path: a `GeoEcef` value with no declared
        // option must auto-infer `FieldOption::Geo3d`. Without this, a
        // ECEF document with a missing schema entry would either fail or
        // (worse) be silently classified as 2D Geo.
        let p = crate::data::GeoEcefPoint::new(1.0, 2.0, 3.0);
        let inferred = infer_option_from_data_value(&DataValue::GeoEcef(p))
            .unwrap()
            .expect("GeoEcef must infer some FieldOption");
        assert!(
            matches!(inferred, FieldOption::Geo3d(_)),
            "expected FieldOption::Geo3d, got {inferred:?}"
        );
    }

    #[test]
    fn infer_option_from_data_value_geo_to_geo() {
        let p = crate::data::GeoPoint::new(35.1, 139.0);
        let inferred = infer_option_from_data_value(&DataValue::Geo(p))
            .unwrap()
            .expect("Geo must infer some FieldOption");
        assert!(
            matches!(inferred, FieldOption::Geo(_)),
            "expected FieldOption::Geo, got {inferred:?}"
        );
    }

    #[test]
    fn infer_unknown_object_rejected() {
        let err = infer_from_json(&json!({"foo": 1, "bar": 2})).unwrap_err();
        assert!(err.to_string().contains("geographic"));
    }

    #[test]
    fn infer_geo_missing_lon_rejected() {
        let err = infer_from_json(&json!({"lat": 35.1})).unwrap_err();
        assert!(err.to_string().contains("geographic"));
    }

    #[test]
    fn infer_geo3d_xyz() {
        let (v, o) = inferred(infer_from_json(&json!({"x": 1.0, "y": 2.0, "z": 3.0})).unwrap());
        assert_eq!(
            v,
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(1.0, 2.0, 3.0))
        );
        assert!(matches!(o, FieldOption::Geo3d(_)));
    }

    #[test]
    fn infer_geo3d_integer_xyz() {
        // JSON integers must also coerce to f64 for ECEF coordinates.
        let (v, _) = inferred(infer_from_json(&json!({"x": 1, "y": 2, "z": 3})).unwrap());
        assert_eq!(
            v,
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(1.0, 2.0, 3.0))
        );
    }

    #[test]
    fn infer_geo3d_real_ecef_values() {
        // Tokyo Tower-ish coordinates (negative x, positive y/z) — make sure
        // the inference does not mishandle large or negative ECEF values.
        let (v, _) = inferred(
            infer_from_json(&json!({
                "x": -3_955_182.0,
                "y": 3_350_553.0,
                "z": 3_700_276.0
            }))
            .unwrap(),
        );
        assert_eq!(
            v,
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3_955_182.0,
                3_350_553.0,
                3_700_276.0
            ))
        );
    }

    #[test]
    fn infer_geo3d_partial_keys_rejected() {
        let err = infer_from_json(&json!({"x": 1.0, "y": 2.0})).unwrap_err();
        // Falls through to the generic "geographic points" error because
        // a Geo3d object requires all three of x, y, z.
        assert!(err.to_string().contains("geographic"));
    }

    #[test]
    fn infer_geo3d_non_numeric_rejected() {
        let err = infer_from_json(&json!({"x": "not a number", "y": 2.0, "z": 3.0})).unwrap_err();
        assert!(err.to_string().contains("Geo3d"));
    }

    #[test]
    fn infer_geo3d_2d_3d_mix_rejected() {
        let err = infer_from_json(&json!({"lat": 35.0, "x": 1.0, "y": 2.0, "z": 3.0})).unwrap_err();
        assert!(err.to_string().contains("mix"));
    }

    #[test]
    fn infer_geo3d_partial_2d_3d_mix_rejected() {
        // Even a single 2D key plus a single 3D key should be rejected,
        // because the user's intent is ambiguous.
        let err = infer_from_json(&json!({"lat": 35.0, "x": 1.0})).unwrap_err();
        assert!(err.to_string().contains("mix"));
    }

    #[test]
    fn infer_bytes_data_only() {
        // base64("hi") == "aGk="
        let (v, o) = inferred(infer_from_json(&json!({"data": "aGk="})).unwrap());
        assert_eq!(v, DataValue::Bytes(b"hi".to_vec(), None));
        assert!(matches!(o, FieldOption::Bytes(_)));
    }

    #[test]
    fn infer_bytes_data_with_mime() {
        let (v, _) =
            inferred(infer_from_json(&json!({"data": "aGk=", "mime": "image/jpeg"})).unwrap());
        assert_eq!(
            v,
            DataValue::Bytes(b"hi".to_vec(), Some("image/jpeg".to_string()))
        );
    }

    #[test]
    fn infer_bytes_invalid_base64_rejected() {
        let err = infer_from_json(&json!({"data": "not valid base64!!"})).unwrap_err();
        assert!(err.to_string().contains("base64"));
    }

    #[test]
    fn infer_bytes_non_string_data_rejected() {
        let err = infer_from_json(&json!({"data": 42})).unwrap_err();
        assert!(err.to_string().contains("base64-encoded string"));
    }

    #[test]
    fn infer_bytes_non_string_mime_rejected() {
        let err = infer_from_json(&json!({"data": "aGk=", "mime": 42})).unwrap_err();
        assert!(err.to_string().contains("mime"));
    }

    #[test]
    fn infer_bytes_geo_mix_rejected() {
        let err = infer_from_json(&json!({"data": "aGk=", "lat": 35.0, "lon": 139.0})).unwrap_err();
        assert!(err.to_string().contains("mix"));
    }

    #[test]
    fn infer_bytes_geo3d_mix_rejected() {
        let err =
            infer_from_json(&json!({"data": "aGk=", "x": 1.0, "y": 2.0, "z": 3.0})).unwrap_err();
        assert!(err.to_string().contains("mix"));
    }

    #[test]
    fn infer_unknown_object_still_rejects_with_geographic_hint() {
        // Regression guard: an old tagged-format document value like
        // `{"Text": "hello"}` must still be rejected outright, not silently
        // misinterpreted as some other shape.
        let err = infer_from_json(&json!({"Text": "hello"})).unwrap_err();
        assert!(err.to_string().contains("geographic"));
    }

    // ---- Multi-valued geo arrays (#1174) ----

    #[test]
    fn infer_geo_object_array_to_geo_array() {
        // Key aliases are those of a single geo object.
        let (v, o) = inferred(
            infer_from_json(&json!([
                {"lat": 35.1, "lon": 139.0},
                {"latitude": -33.9, "longitude": 151.2},
            ]))
            .unwrap(),
        );
        assert_eq!(
            v,
            DataValue::GeoArray(vec![
                crate::data::GeoPoint::new(35.1, 139.0),
                crate::data::GeoPoint::new(-33.9, 151.2),
            ])
        );
        assert!(matches!(
            o,
            FieldOption::Geo(GeoOption {
                multi_valued: true,
                ..
            })
        ));
    }

    #[test]
    fn infer_geo3d_object_array_to_geo_ecef_array() {
        let (v, o) = inferred(
            infer_from_json(&json!([{"x": 1.0, "y": 2.0, "z": 3.0}, {"x": 4, "y": 5, "z": 6}]))
                .unwrap(),
        );
        assert_eq!(
            v,
            DataValue::GeoEcefArray(vec![
                crate::data::GeoEcefPoint::new(1.0, 2.0, 3.0),
                crate::data::GeoEcefPoint::new(4.0, 5.0, 6.0),
            ])
        );
        assert!(matches!(
            o,
            FieldOption::Geo3d(crate::lexical::core::field::Geo3dOption {
                multi_valued: true,
                ..
            })
        ));
    }

    #[test]
    fn infer_mixed_geo_dimension_array_rejected() {
        let err = infer_from_json(&json!([
            {"lat": 35.1, "lon": 139.0},
            {"x": 1.0, "y": 2.0, "z": 3.0},
        ]))
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("all 2D geo points or all 3D ECEF points"),
            "{err}"
        );
    }

    #[test]
    fn infer_array_with_non_geo_object_rejected() {
        // The per-element check is `infer_from_object`, so a non-geo object
        // fails the same way it would on its own.
        assert!(infer_from_json(&json!([{"lat": 35.1, "lon": 139.0}, {"foo": 1}])).is_err());
        // ...including its range validation.
        assert!(infer_from_json(&json!([{"lat": 95.0, "lon": 0.0}])).is_err());
    }

    #[test]
    fn infer_number_and_object_mix_still_reports_numeric_error() {
        // The geo branch is gated on *every* element being an object, so a
        // number/object mix keeps the pre-#1174 numeric-array error text.
        let err = infer_from_json(&json!([1, {"lat": 35.1, "lon": 139.0}])).unwrap_err();
        assert!(err.to_string().contains("only numeric"), "{err}");
    }

    // ---- Multi-valued datetime arrays (#1184) ----

    #[test]
    fn infer_rfc3339_string_array_to_datetime_array() {
        use chrono::TimeZone;
        let (v, o) = inferred(
            infer_from_json(&json!([
                "2024-01-01T00:00:00Z",
                "2024-06-15T21:00:00+09:00",
            ]))
            .unwrap(),
        );
        assert_eq!(
            v,
            DataValue::DateTimeArray(vec![
                chrono::Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                // Offsets are normalized to UTC.
                chrono::Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap(),
            ])
        );
        assert!(matches!(
            o,
            FieldOption::DateTime(crate::lexical::core::field::DateTimeOption {
                multi_valued: true,
                ..
            })
        ));
    }

    #[test]
    fn infer_string_array_with_non_datetime_rejected() {
        // Every element must be RFC 3339; naive / date-only forms are query
        // literals, not ingestion values.
        for bad in [
            json!(["2024-01-01T00:00:00Z", "yesterday"]),
            json!(["2024-01-01"]),
            json!(["2024-01-01T00:00:00"]),
            json!(["a", "b"]),
        ] {
            let err = infer_from_json(&bad).unwrap_err();
            assert!(
                err.to_string().contains("all RFC 3339 datetimes"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn infer_single_rfc3339_string_still_infers_text() {
        // The documented asymmetry: only *arrays* of RFC 3339 strings infer
        // a DateTime; a lone string keeps inferring Text.
        let (v, o) = inferred(infer_from_json(&json!("2024-01-01T00:00:00Z")).unwrap());
        assert_eq!(v, DataValue::Text("2024-01-01T00:00:00Z".into()));
        assert!(matches!(o, FieldOption::Text(_)));
    }

    #[test]
    fn infer_option_from_datetime_array_is_multi_valued_datetime() {
        let opt = infer_option_from_data_value(&DataValue::DateTimeArray(Vec::new())).unwrap();
        assert!(matches!(
            opt,
            Some(FieldOption::DateTime(
                crate::lexical::core::field::DateTimeOption {
                    multi_valued: true,
                    ..
                }
            ))
        ));
    }
}
