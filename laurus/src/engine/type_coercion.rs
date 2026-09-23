//! Type coercion for existing schema fields.
//!
//! When a document is ingested, each provided value is checked against the
//! declared [`FieldOption`] of its field. If the value does not already match
//! the field's type, this module decides whether it can be coerced (converted
//! without rejecting the document) and, if so, returns the coerced value.
//!
//! The coercion rules honour the user-facing **Dynamic Schema** semantics
//! documented in [`docs/src/concepts/schema_and_fields.md`]. In particular:
//!
//! - Integer fields **truncate** incoming float values (`3.14` → `3`). This
//!   is a **silent information loss**, preferred over rejecting the document.
//! - Numeric and boolean values are parsed from their canonical string
//!   representations when they are unambiguous (`"42"`, `"true"`).
//! - Text fields accept any scalar value by stringifying it.
//! - Geographic and (future) multi-valued numeric fields have strict typing.
//! - Bytes fields accept a plain string as base64-encoded data, in addition
//!   to a pass-through `Bytes` value.
//!
//! Values that cannot be coerced produce an error. The caller (see
//! [`Engine`](crate::Engine)) decides what to do with the error based on the
//! [`DynamicFieldPolicy`](super::schema::DynamicFieldPolicy):
//!
//! - `Strict` → propagates the error, aborting ingestion.
//! - `Dynamic` → propagates the error. The policy's "convert when possible"
//!   rule is already embedded in this module; truly incompatible values are
//!   reported back.
//! - `Ignore` → drops the field and continues.

use base64::Engine as _;

use crate::data::DataValue;
use crate::error::{LaurusError, Result};

use super::schema::FieldOption;

/// Attempt to coerce `value` into the type declared by `option`.
///
/// Returns the coerced (or unchanged) [`DataValue`] on success.
///
/// # Arguments
///
/// * `field_name` - The name of the field, used for error messages.
/// * `option` - The declared field option describing the target type.
/// * `value` - The incoming value from the user-supplied document.
///
/// # Errors
///
/// Returns [`LaurusError::invalid_argument`] if the value cannot be coerced
/// to the field's declared type.
pub fn coerce_value(field_name: &str, option: &FieldOption, value: DataValue) -> Result<DataValue> {
    match option {
        FieldOption::Text(_) => coerce_to_text(field_name, value),
        FieldOption::Integer(opt) => coerce_to_integer(field_name, opt, value),
        FieldOption::Float(opt) => coerce_to_float(field_name, opt, value),
        FieldOption::Boolean(opt) => coerce_to_boolean(field_name, opt, value),
        FieldOption::DateTime(opt) => coerce_to_datetime(field_name, opt, value),
        FieldOption::Geo(opt) => coerce_to_geo(field_name, opt, value),
        FieldOption::Geo3d(opt) => coerce_to_geo3d(field_name, opt, value),
        FieldOption::Bytes(_) => coerce_to_bytes(field_name, value),
        FieldOption::Hnsw(_) | FieldOption::Flat(_) | FieldOption::Ivf(_) => {
            coerce_to_vector(field_name, value)
        }
    }
}

fn coerce_to_text(_field_name: &str, value: DataValue) -> Result<DataValue> {
    Ok(match value {
        DataValue::Text(s) => DataValue::Text(s),
        DataValue::Int64(i) => DataValue::Text(i.to_string()),
        DataValue::Float64(f) => DataValue::Text(f.to_string()),
        DataValue::Bool(b) => DataValue::Text(b.to_string()),
        DataValue::DateTime(dt) => DataValue::Text(dt.to_rfc3339()),
        DataValue::Null => DataValue::Text(String::new()),
        // Other variants (Bytes, Vector, Geo, arrays) don't have a meaningful
        // string representation for a text field.
        other => {
            return Err(LaurusError::invalid_argument(format!(
                "cannot coerce {} to a text value",
                describe(&other)
            )));
        }
    })
}

fn coerce_to_integer(
    field_name: &str,
    option: &crate::lexical::core::field::IntegerOption,
    value: DataValue,
) -> Result<DataValue> {
    if option.multi_valued {
        // Multi-valued integer field. Single-value inputs are auto-wrapped
        // into a one-element array. Float inputs (single or array) are
        // truncated element-wise (Lucene-style information-losing).
        match value {
            DataValue::Int64Array(arr) => Ok(DataValue::Int64Array(arr)),
            DataValue::Float64Array(arr) => Ok(DataValue::Int64Array(
                arr.iter().map(|f| *f as i64).collect(),
            )),
            DataValue::Int64(i) => Ok(DataValue::Int64Array(vec![i])),
            DataValue::Float64(f) => Ok(DataValue::Int64Array(vec![f as i64])),
            DataValue::Bool(b) => Ok(DataValue::Int64Array(vec![if b { 1 } else { 0 }])),
            // Element-wise extension of the scalar `Bool` -> 0 / 1 rule
            // above (#1180): bindings hand a list of bools over as a
            // `BoolArray` before the field type is known.
            DataValue::BoolArray(arr) => Ok(DataValue::Int64Array(
                arr.iter().map(|b| i64::from(*b)).collect(),
            )),
            DataValue::Text(s) => s
                .trim()
                .parse::<i64>()
                .map(|n| DataValue::Int64Array(vec![n]))
                .map_err(|_| {
                    LaurusError::invalid_argument(format!(
                        "field '{field_name}': cannot parse '{s}' as an integer"
                    ))
                }),
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to a multi-valued integer",
                describe(&other)
            ))),
        }
    } else {
        match value {
            DataValue::Int64(i) => Ok(DataValue::Int64(i)),
            // Information-losing truncation: documented as intentional.
            DataValue::Float64(f) => Ok(DataValue::Int64(f as i64)),
            DataValue::Bool(b) => Ok(DataValue::Int64(if b { 1 } else { 0 })),
            DataValue::Text(s) => s.trim().parse::<i64>().map(DataValue::Int64).map_err(|_| {
                LaurusError::invalid_argument(format!(
                    "field '{field_name}': cannot parse '{s}' as an integer"
                ))
            }),
            // Multi-valued input to a single-valued field is rejected
            // rather than silently truncating to one element.
            DataValue::Int64Array(_) | DataValue::Float64Array(_) | DataValue::BoolArray(_) => {
                Err(LaurusError::invalid_argument(format!(
                    "field '{field_name}': received an array but the field is single-valued; \
                     declare the field with multi_valued = true to accept arrays"
                )))
            }
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to an integer",
                describe(&other)
            ))),
        }
    }
}

fn coerce_to_float(
    field_name: &str,
    option: &crate::lexical::core::field::FloatOption,
    value: DataValue,
) -> Result<DataValue> {
    if option.multi_valued {
        match value {
            DataValue::Float64Array(arr) => Ok(DataValue::Float64Array(arr)),
            DataValue::Int64Array(arr) => Ok(DataValue::Float64Array(
                arr.iter().map(|i| *i as f64).collect(),
            )),
            DataValue::Float64(f) => Ok(DataValue::Float64Array(vec![f])),
            DataValue::Int64(i) => Ok(DataValue::Float64Array(vec![i as f64])),
            DataValue::Bool(b) => Ok(DataValue::Float64Array(vec![if b { 1.0 } else { 0.0 }])),
            // Element-wise extension of the scalar `Bool` -> 0.0 / 1.0 rule
            // above (#1180).
            DataValue::BoolArray(arr) => Ok(DataValue::Float64Array(
                arr.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect(),
            )),
            DataValue::Text(s) => s
                .trim()
                .parse::<f64>()
                .map(|n| DataValue::Float64Array(vec![n]))
                .map_err(|_| {
                    LaurusError::invalid_argument(format!(
                        "field '{field_name}': cannot parse '{s}' as a float"
                    ))
                }),
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to a multi-valued float",
                describe(&other)
            ))),
        }
    } else {
        match value {
            DataValue::Float64(f) => Ok(DataValue::Float64(f)),
            DataValue::Int64(i) => Ok(DataValue::Float64(i as f64)),
            DataValue::Bool(b) => Ok(DataValue::Float64(if b { 1.0 } else { 0.0 })),
            DataValue::Text(s) => s
                .trim()
                .parse::<f64>()
                .map(DataValue::Float64)
                .map_err(|_| {
                    LaurusError::invalid_argument(format!(
                        "field '{field_name}': cannot parse '{s}' as a float"
                    ))
                }),
            DataValue::Int64Array(_) | DataValue::Float64Array(_) | DataValue::BoolArray(_) => {
                Err(LaurusError::invalid_argument(format!(
                    "field '{field_name}': received an array but the field is single-valued; \
                     declare the field with multi_valued = true to accept arrays"
                )))
            }
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to a float",
                describe(&other)
            ))),
        }
    }
}

/// The scalar Boolean rule shared by both [`coerce_to_boolean`] branches
/// (#1180): a typed bool, the integers `0` / `1`, or the text `true` /
/// `false` (trimmed, case-insensitive). Anything else is an error naming
/// the field.
fn parse_bool_scalar(field_name: &str, value: &DataValue) -> Result<bool> {
    match value {
        DataValue::Bool(b) => Ok(*b),
        DataValue::Int64(0) => Ok(false),
        DataValue::Int64(1) => Ok(true),
        DataValue::Int64(n) => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': cannot coerce integer {n} to bool (only 0 and 1 are accepted)"
        ))),
        DataValue::Text(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot parse '{s}' as a bool (expected 'true' or 'false')"
            ))),
        },
        other => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': cannot coerce {} to a bool",
            describe(other)
        ))),
    }
}

fn coerce_to_boolean(
    field_name: &str,
    option: &crate::lexical::core::field::BooleanOption,
    value: DataValue,
) -> Result<DataValue> {
    if option.multi_valued {
        // Multi-valued boolean field (#1180). A single value is auto-wrapped
        // under the scalar rule; an integer array is widened element-wise
        // under the same 0 / 1 rule — which also covers the empty
        // `Int64Array` every binding sends for `[]` before the field type
        // is known (#1178); an empty float array is that same `[]`.
        match value {
            DataValue::BoolArray(arr) => Ok(DataValue::BoolArray(arr)),
            DataValue::Int64Array(arr) => arr
                .iter()
                .map(|i| parse_bool_scalar(field_name, &DataValue::Int64(*i)))
                .collect::<Result<Vec<bool>>>()
                .map(DataValue::BoolArray),
            DataValue::Float64Array(a) if a.is_empty() => Ok(DataValue::BoolArray(Vec::new())),
            scalar @ (DataValue::Bool(_) | DataValue::Int64(_) | DataValue::Text(_)) => Ok(
                DataValue::BoolArray(vec![parse_bool_scalar(field_name, &scalar)?]),
            ),
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to a multi-valued bool",
                describe(&other)
            ))),
        }
    } else {
        match value {
            // Multi-valued input to a single-valued field is rejected
            // rather than silently truncating to one element.
            DataValue::BoolArray(_) => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': received an array but the field is single-valued; \
                 declare the field with multi_valued = true to accept arrays"
            ))),
            other => Ok(DataValue::Bool(parse_bool_scalar(field_name, &other)?)),
        }
    }
}

fn coerce_to_datetime(
    field_name: &str,
    option: &crate::lexical::core::field::DateTimeOption,
    value: DataValue,
) -> Result<DataValue> {
    let parse = |s: &str| {
        chrono::DateTime::parse_from_rfc3339(s.trim())
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .map_err(|e| {
                LaurusError::invalid_argument(format!(
                    "field '{field_name}': cannot parse '{s}' as an RFC 3339 datetime: {e}"
                ))
            })
    };
    if option.multi_valued {
        // Multi-valued datetime field (#1184). A single instant (typed or
        // RFC 3339 text) is auto-wrapped; an empty *numeric* array is an
        // empty instant list because every binding turns `[]` into
        // `Int64Array(vec![])` before the field type is known (#1178).
        match value {
            DataValue::DateTimeArray(arr) => Ok(DataValue::DateTimeArray(arr)),
            DataValue::DateTime(dt) => Ok(DataValue::DateTimeArray(vec![dt])),
            DataValue::Text(s) => Ok(DataValue::DateTimeArray(vec![parse(&s)?])),
            DataValue::Int64Array(a) if a.is_empty() => Ok(DataValue::DateTimeArray(Vec::new())),
            DataValue::Float64Array(a) if a.is_empty() => Ok(DataValue::DateTimeArray(Vec::new())),
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to a multi-valued datetime",
                describe(&other)
            ))),
        }
    } else {
        match value {
            DataValue::DateTime(dt) => Ok(DataValue::DateTime(dt)),
            DataValue::Text(s) => Ok(DataValue::DateTime(parse(&s)?)),
            // Multi-valued input to a single-valued field is rejected
            // rather than silently truncating to one element.
            DataValue::DateTimeArray(_) => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': received an array but the field is single-valued; \
                 declare the field with multi_valued = true to accept arrays"
            ))),
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to a datetime",
                describe(&other)
            ))),
        }
    }
}

fn coerce_to_geo(
    field_name: &str,
    option: &crate::lexical::core::field::GeoOption,
    value: DataValue,
) -> Result<DataValue> {
    if option.multi_valued {
        // Multi-valued geo field (#1174). A single point is auto-wrapped
        // into a one-element array. An empty *numeric* array is accepted as
        // an empty point list: every binding turns `[]` into
        // `Int64Array(vec![])` before the field type is known (#1178), the
        // same shape `coerce_to_vector` accommodates.
        match value {
            DataValue::GeoArray(arr) => Ok(DataValue::GeoArray(arr)),
            DataValue::Geo(p) => Ok(DataValue::GeoArray(vec![p])),
            DataValue::Int64Array(a) if a.is_empty() => Ok(DataValue::GeoArray(Vec::new())),
            DataValue::Float64Array(a) if a.is_empty() => Ok(DataValue::GeoArray(Vec::new())),
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to a multi-valued geographic point",
                describe(&other)
            ))),
        }
    } else {
        match value {
            DataValue::Geo(p) => Ok(DataValue::Geo(p)),
            // Multi-valued input to a single-valued field is rejected
            // rather than silently truncating to one element.
            DataValue::GeoArray(_) => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': received an array but the field is single-valued; \
                 declare the field with multi_valued = true to accept arrays"
            ))),
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to a geographic point",
                describe(&other)
            ))),
        }
    }
}

fn coerce_to_geo3d(
    field_name: &str,
    option: &crate::lexical::core::field::Geo3dOption,
    value: DataValue,
) -> Result<DataValue> {
    if option.multi_valued {
        // See `coerce_to_geo` for the empty-numeric-array accommodation.
        match value {
            DataValue::GeoEcefArray(arr) => Ok(DataValue::GeoEcefArray(arr)),
            DataValue::GeoEcef(p) => Ok(DataValue::GeoEcefArray(vec![p])),
            DataValue::Int64Array(a) if a.is_empty() => Ok(DataValue::GeoEcefArray(Vec::new())),
            DataValue::Float64Array(a) if a.is_empty() => Ok(DataValue::GeoEcefArray(Vec::new())),
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to a multi-valued 3D ECEF geo point",
                describe(&other)
            ))),
        }
    } else {
        match value {
            DataValue::GeoEcef(p) => Ok(DataValue::GeoEcef(p)),
            DataValue::GeoEcefArray(_) => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': received an array but the field is single-valued; \
                 declare the field with multi_valued = true to accept arrays"
            ))),
            other => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': cannot coerce {} to a 3D ECEF geo point",
                describe(&other)
            ))),
        }
    }
}

fn coerce_to_bytes(field_name: &str, value: DataValue) -> Result<DataValue> {
    match value {
        DataValue::Bytes(data, mime) => Ok(DataValue::Bytes(data, mime)),
        // A plain string on a declared `Bytes` field is unambiguous — the
        // schema already says "this is bytes" — so treat it as base64,
        // matching the `{"data": "<base64>"}` object shape that
        // `type_inference::infer_from_object` produces for undeclared
        // fields. Unlike a bytes-typed `Hnsw`/`Flat`/`Ivf` field (see
        // `coerce_to_vector`), there is no text-vs-bytes ambiguity here to
        // preserve, since a declared `Bytes` field never accepts a
        // to-be-embedded string.
        DataValue::Text(s) => base64::engine::general_purpose::STANDARD
            .decode(s.trim())
            .map(|data| DataValue::Bytes(data, None))
            .map_err(|e| {
                LaurusError::invalid_argument(format!(
                    "field '{field_name}': cannot decode '{s}' as base64: {e}"
                ))
            }),
        other => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': cannot coerce {} to bytes",
            describe(&other)
        ))),
    }
}

fn coerce_to_vector(field_name: &str, value: DataValue) -> Result<DataValue> {
    // Vector fields accept these input shapes, resolved downstream by the
    // vector store:
    //
    // - `Vector`: a pre-computed embedding, indexed verbatim.
    // - `Text` / `Bytes`: passed through unchanged so the vector store's
    //   configured embedder can turn them into vectors. The engine itself
    //   never auto-embeds, but an explicitly-configured embedder may.
    // - `Int64Array` / `Float64Array`: pre-computed embeddings supplied as
    //   numeric arrays via JSON or bindings. Cast element-wise to f32 since
    //   the vector store stores 32-bit floats.
    match value {
        DataValue::Vector(v) => Ok(DataValue::Vector(v)),
        DataValue::Text(s) => Ok(DataValue::Text(s)),
        DataValue::Bytes(data, mime) => Ok(DataValue::Bytes(data, mime)),
        DataValue::Float64Array(arr) => {
            Ok(DataValue::Vector(arr.iter().map(|v| *v as f32).collect()))
        }
        DataValue::Int64Array(arr) => {
            Ok(DataValue::Vector(arr.iter().map(|v| *v as f32).collect()))
        }
        other => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': vector fields accept Vector, Text, Bytes, \
             or numeric arrays; got {}",
            describe(&other)
        ))),
    }
}

fn describe(value: &DataValue) -> &'static str {
    match value {
        DataValue::Null => "null",
        DataValue::Bool(_) => "bool",
        DataValue::Int64(_) => "integer",
        DataValue::Float64(_) => "float",
        DataValue::Text(_) => "text",
        DataValue::Bytes(_, _) => "bytes",
        DataValue::Vector(_) => "vector",
        DataValue::DateTime(_) => "datetime",
        DataValue::Geo(_) => "geo",
        DataValue::GeoEcef(_) => "geo3d",
        DataValue::Int64Array(_) => "integer array",
        DataValue::Float64Array(_) => "float array",
        DataValue::GeoArray(_) => "geo array",
        DataValue::GeoEcefArray(_) => "geo3d array",
        DataValue::DateTimeArray(_) => "datetime array",
        DataValue::BoolArray(_) => "bool array",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::core::field::{
        BooleanOption, FloatOption, Geo3dOption, GeoOption, IntegerOption, TextOption,
    };

    fn integer() -> FieldOption {
        FieldOption::Integer(IntegerOption::default())
    }

    fn float() -> FieldOption {
        FieldOption::Float(FloatOption::default())
    }

    fn boolean() -> FieldOption {
        FieldOption::Boolean(BooleanOption::default())
    }

    fn text() -> FieldOption {
        FieldOption::Text(TextOption::default())
    }

    fn geo() -> FieldOption {
        FieldOption::Geo(GeoOption::default())
    }

    fn geo3d() -> FieldOption {
        FieldOption::Geo3d(Geo3dOption::default())
    }

    fn geo_multi() -> FieldOption {
        FieldOption::Geo(GeoOption {
            multi_valued: true,
            ..Default::default()
        })
    }

    fn geo3d_multi() -> FieldOption {
        FieldOption::Geo3d(Geo3dOption {
            multi_valued: true,
            ..Default::default()
        })
    }

    fn dt_single() -> FieldOption {
        FieldOption::DateTime(crate::lexical::core::field::DateTimeOption::default())
    }

    fn dt_multi() -> FieldOption {
        FieldOption::DateTime(crate::lexical::core::field::DateTimeOption {
            multi_valued: true,
            ..Default::default()
        })
    }

    // ---- Multi-valued datetime (#1184) ----

    #[test]
    fn datetime_multi_valued_accepts_array_and_wraps_single() {
        use chrono::TimeZone;
        let a = chrono::Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let b = chrono::Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
        assert_eq!(
            coerce_value("t", &dt_multi(), DataValue::DateTimeArray(vec![a, b])).unwrap(),
            DataValue::DateTimeArray(vec![a, b])
        );
        assert_eq!(
            coerce_value("t", &dt_multi(), DataValue::DateTime(a)).unwrap(),
            DataValue::DateTimeArray(vec![a])
        );
        // RFC 3339 text is parsed like the single-valued path, then wrapped.
        assert_eq!(
            coerce_value(
                "t",
                &dt_multi(),
                DataValue::Text("2024-06-15T21:00:00+09:00".to_string())
            )
            .unwrap(),
            DataValue::DateTimeArray(vec![b])
        );
        assert!(coerce_value("t", &dt_multi(), DataValue::Text("yesterday".to_string())).is_err());
    }

    #[test]
    fn datetime_multi_valued_accepts_empty_numeric_array_only() {
        // Bindings turn `[]` into an empty numeric array before the field
        // type is known (#1178); that must read as "no instants".
        assert_eq!(
            coerce_value("t", &dt_multi(), DataValue::Int64Array(Vec::new())).unwrap(),
            DataValue::DateTimeArray(Vec::new())
        );
        assert_eq!(
            coerce_value("t", &dt_multi(), DataValue::Float64Array(Vec::new())).unwrap(),
            DataValue::DateTimeArray(Vec::new())
        );
        assert!(
            coerce_value("t", &dt_multi(), DataValue::Int64Array(vec![1_700_000_000])).is_err(),
            "a non-empty numeric array is not an instant list"
        );
        assert!(coerce_value("t", &dt_multi(), DataValue::Bool(true)).is_err());
    }

    #[test]
    fn datetime_single_valued_rejects_arrays() {
        use chrono::TimeZone;
        let a = chrono::Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let err = coerce_value("t", &dt_single(), DataValue::DateTimeArray(vec![a])).unwrap_err();
        assert!(err.to_string().contains("multi_valued = true"), "{err}");
        // The single-valued contract is otherwise untouched.
        assert_eq!(
            coerce_value("t", &dt_single(), DataValue::DateTime(a)).unwrap(),
            DataValue::DateTime(a)
        );
        assert_eq!(
            coerce_value(
                "t",
                &dt_single(),
                DataValue::Text("2024-01-01T00:00:00Z".to_string())
            )
            .unwrap(),
            DataValue::DateTime(a)
        );
    }

    // ---- Multi-valued boolean (#1180) ----

    fn multi_boolean() -> FieldOption {
        FieldOption::Boolean(BooleanOption {
            multi_valued: true,
            ..Default::default()
        })
    }

    fn multi_integer() -> FieldOption {
        FieldOption::Integer(IntegerOption {
            multi_valued: true,
            ..Default::default()
        })
    }

    fn multi_float() -> FieldOption {
        FieldOption::Float(FloatOption {
            multi_valued: true,
            ..Default::default()
        })
    }

    #[test]
    fn boolean_multi_valued_accepts_array_and_wraps_single() {
        assert_eq!(
            coerce_value(
                "f",
                &multi_boolean(),
                DataValue::BoolArray(vec![true, false])
            )
            .unwrap(),
            DataValue::BoolArray(vec![true, false])
        );
        // Every scalar the single-valued path accepts is wrapped.
        assert_eq!(
            coerce_value("f", &multi_boolean(), DataValue::Bool(true)).unwrap(),
            DataValue::BoolArray(vec![true])
        );
        assert_eq!(
            coerce_value("f", &multi_boolean(), DataValue::Int64(1)).unwrap(),
            DataValue::BoolArray(vec![true])
        );
        assert_eq!(
            coerce_value("f", &multi_boolean(), DataValue::Text(" TRUE ".to_string())).unwrap(),
            DataValue::BoolArray(vec![true])
        );
        assert!(coerce_value("f", &multi_boolean(), DataValue::Text("yes".to_string())).is_err());
        assert!(coerce_value("f", &multi_boolean(), DataValue::Float64(1.0)).is_err());
    }

    #[test]
    fn boolean_multi_valued_accepts_int_array_of_zero_one() {
        assert_eq!(
            coerce_value("f", &multi_boolean(), DataValue::Int64Array(vec![0, 1, 1])).unwrap(),
            DataValue::BoolArray(vec![false, true, true])
        );
        let err =
            coerce_value("f", &multi_boolean(), DataValue::Int64Array(vec![0, 2])).unwrap_err();
        assert!(err.to_string().contains("only 0 and 1"), "{err}");
        // Bindings turn `[]` into an empty numeric array before the field
        // type is known (#1178); that must read as "no flags".
        assert_eq!(
            coerce_value("f", &multi_boolean(), DataValue::Int64Array(Vec::new())).unwrap(),
            DataValue::BoolArray(Vec::new())
        );
        assert_eq!(
            coerce_value("f", &multi_boolean(), DataValue::Float64Array(Vec::new())).unwrap(),
            DataValue::BoolArray(Vec::new())
        );
        assert!(
            coerce_value("f", &multi_boolean(), DataValue::Float64Array(vec![1.0])).is_err(),
            "a non-empty float array is not a flag list (the scalar rule rejects floats too)"
        );
    }

    #[test]
    fn boolean_single_valued_rejects_arrays() {
        let err = coerce_value("f", &boolean(), DataValue::BoolArray(vec![true])).unwrap_err();
        assert!(err.to_string().contains("multi_valued = true"), "{err}");
        // The single-valued contract is otherwise untouched.
        assert_eq!(
            coerce_value("f", &boolean(), DataValue::Bool(false)).unwrap(),
            DataValue::Bool(false)
        );
        assert_eq!(
            coerce_value("f", &boolean(), DataValue::Int64(0)).unwrap(),
            DataValue::Bool(false)
        );
        assert_eq!(
            coerce_value("f", &boolean(), DataValue::Text("false".to_string())).unwrap(),
            DataValue::Bool(false)
        );
        let err = coerce_value("f", &boolean(), DataValue::Int64(2)).unwrap_err();
        assert!(err.to_string().contains("only 0 and 1"), "{err}");
    }

    /// A `BoolArray` arriving at a multi-valued numeric field is widened
    /// element-wise, exactly like a scalar `Bool` is today.
    #[test]
    fn numeric_multi_valued_widens_bool_array() {
        assert_eq!(
            coerce_value(
                "n",
                &multi_integer(),
                DataValue::BoolArray(vec![true, false])
            )
            .unwrap(),
            DataValue::Int64Array(vec![1, 0])
        );
        assert_eq!(
            coerce_value("n", &multi_float(), DataValue::BoolArray(vec![true, false])).unwrap(),
            DataValue::Float64Array(vec![1.0, 0.0])
        );
    }

    #[test]
    fn numeric_single_valued_rejects_bool_array_with_hint() {
        for opt in [integer(), float()] {
            let err = coerce_value("n", &opt, DataValue::BoolArray(vec![true])).unwrap_err();
            assert!(err.to_string().contains("multi_valued = true"), "{err}");
        }
    }

    fn bytes() -> FieldOption {
        FieldOption::Bytes(crate::lexical::core::field::BytesOption::default())
    }

    #[test]
    fn integer_passthrough() {
        assert_eq!(
            coerce_value("n", &integer(), DataValue::Int64(5)).unwrap(),
            DataValue::Int64(5)
        );
    }

    #[test]
    fn integer_truncates_float() {
        assert_eq!(
            coerce_value("n", &integer(), DataValue::Float64(4.7)).unwrap(),
            DataValue::Int64(4)
        );
        assert_eq!(
            coerce_value("n", &integer(), DataValue::Float64(-3.9)).unwrap(),
            DataValue::Int64(-3)
        );
    }

    #[test]
    fn integer_parses_text() {
        assert_eq!(
            coerce_value("n", &integer(), DataValue::Text("42".into())).unwrap(),
            DataValue::Int64(42)
        );
    }

    #[test]
    fn integer_rejects_unparseable_text() {
        assert!(coerce_value("n", &integer(), DataValue::Text("abc".into())).is_err());
    }

    #[test]
    fn integer_from_bool() {
        assert_eq!(
            coerce_value("n", &integer(), DataValue::Bool(true)).unwrap(),
            DataValue::Int64(1)
        );
    }

    #[test]
    fn float_from_integer() {
        assert_eq!(
            coerce_value("x", &float(), DataValue::Int64(42)).unwrap(),
            DataValue::Float64(42.0)
        );
    }

    #[test]
    fn float_parses_text() {
        assert_eq!(
            coerce_value("x", &float(), DataValue::Text("4.5".into())).unwrap(),
            DataValue::Float64(4.5)
        );
    }

    #[test]
    fn float_rejects_unparseable_text() {
        assert!(coerce_value("x", &float(), DataValue::Text("abc".into())).is_err());
    }

    #[test]
    fn boolean_from_int_zero_one() {
        assert_eq!(
            coerce_value("b", &boolean(), DataValue::Int64(0)).unwrap(),
            DataValue::Bool(false)
        );
        assert_eq!(
            coerce_value("b", &boolean(), DataValue::Int64(1)).unwrap(),
            DataValue::Bool(true)
        );
    }

    #[test]
    fn boolean_rejects_other_ints() {
        assert!(coerce_value("b", &boolean(), DataValue::Int64(2)).is_err());
    }

    #[test]
    fn boolean_from_text() {
        assert_eq!(
            coerce_value("b", &boolean(), DataValue::Text("true".into())).unwrap(),
            DataValue::Bool(true)
        );
        assert_eq!(
            coerce_value("b", &boolean(), DataValue::Text("False".into())).unwrap(),
            DataValue::Bool(false)
        );
    }

    #[test]
    fn boolean_rejects_yes() {
        assert!(coerce_value("b", &boolean(), DataValue::Text("yes".into())).is_err());
    }

    #[test]
    fn text_stringifies_any_scalar() {
        assert_eq!(
            coerce_value("t", &text(), DataValue::Int64(42)).unwrap(),
            DataValue::Text("42".into())
        );
        assert_eq!(
            coerce_value("t", &text(), DataValue::Float64(4.5)).unwrap(),
            DataValue::Text("4.5".into())
        );
        assert_eq!(
            coerce_value("t", &text(), DataValue::Bool(true)).unwrap(),
            DataValue::Text("true".into())
        );
    }

    #[test]
    fn geo_passthrough_only() {
        assert_eq!(
            coerce_value(
                "g",
                &geo(),
                DataValue::Geo(crate::data::GeoPoint::new(35.1, 139.0))
            )
            .unwrap(),
            DataValue::Geo(crate::data::GeoPoint::new(35.1, 139.0))
        );
        assert!(coerce_value("g", &geo(), DataValue::Int64(35)).is_err());
    }

    #[test]
    fn geo3d_passthrough_only() {
        let p = crate::data::GeoEcefPoint::new(1_234_567.0, 2_345_678.0, 3_456_789.0);
        assert_eq!(
            coerce_value("g3", &geo3d(), DataValue::GeoEcef(p)).unwrap(),
            DataValue::GeoEcef(p)
        );
        // 2D geo cannot coerce to 3D ECEF: the schema declared the field
        // as Geo3d, so the writer must reject a 2D value rather than
        // silently lose the altitude dimension.
        let p2d = crate::data::GeoPoint::new(35.1, 139.0);
        assert!(coerce_value("g3", &geo3d(), DataValue::Geo(p2d)).is_err());
        assert!(coerce_value("g3", &geo3d(), DataValue::Int64(35)).is_err());
    }

    // ---- Multi-valued geo (#1174) ----

    #[test]
    fn multi_valued_geo_accepts_arrays_and_wraps_singles() {
        use crate::data::GeoPoint;
        let pts = vec![GeoPoint::new(35.1, 139.0), GeoPoint::new(-33.9, 151.2)];
        assert_eq!(
            coerce_value("g", &geo_multi(), DataValue::GeoArray(pts.clone())).unwrap(),
            DataValue::GeoArray(pts)
        );
        // A single point is auto-wrapped, mirroring Integer/Float.
        assert_eq!(
            coerce_value(
                "g",
                &geo_multi(),
                DataValue::Geo(GeoPoint::new(35.1, 139.0))
            )
            .unwrap(),
            DataValue::GeoArray(vec![GeoPoint::new(35.1, 139.0)])
        );
        // Bindings turn `[]` into an empty numeric array before the field
        // type is known (#1178); that must read as "no points", not an error.
        assert_eq!(
            coerce_value("g", &geo_multi(), DataValue::Int64Array(Vec::new())).unwrap(),
            DataValue::GeoArray(Vec::new())
        );
        assert_eq!(
            coerce_value("g", &geo_multi(), DataValue::Float64Array(Vec::new())).unwrap(),
            DataValue::GeoArray(Vec::new())
        );
        // ...but a non-empty numeric array is not a point list.
        assert!(coerce_value("g", &geo_multi(), DataValue::Int64Array(vec![35, 139])).is_err());
        assert!(coerce_value("g", &geo_multi(), DataValue::Text("x".into())).is_err());
        // Dimension mismatch is rejected rather than truncated.
        assert!(
            coerce_value(
                "g",
                &geo_multi(),
                DataValue::GeoEcefArray(vec![crate::data::GeoEcefPoint::new(1.0, 2.0, 3.0)])
            )
            .is_err()
        );
    }

    #[test]
    fn single_valued_geo_rejects_arrays() {
        let err = coerce_value(
            "g",
            &geo(),
            DataValue::GeoArray(vec![crate::data::GeoPoint::new(35.1, 139.0)]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("multi_valued = true"), "{err}");
    }

    #[test]
    fn multi_valued_geo3d_accepts_arrays_and_wraps_singles() {
        use crate::data::GeoEcefPoint;
        let pts = vec![
            GeoEcefPoint::new(1.0, 2.0, 3.0),
            GeoEcefPoint::new(-4.0, 5.0, -6.0),
        ];
        assert_eq!(
            coerce_value("g3", &geo3d_multi(), DataValue::GeoEcefArray(pts.clone())).unwrap(),
            DataValue::GeoEcefArray(pts)
        );
        assert_eq!(
            coerce_value(
                "g3",
                &geo3d_multi(),
                DataValue::GeoEcef(GeoEcefPoint::new(1.0, 2.0, 3.0))
            )
            .unwrap(),
            DataValue::GeoEcefArray(vec![GeoEcefPoint::new(1.0, 2.0, 3.0)])
        );
        assert_eq!(
            coerce_value("g3", &geo3d_multi(), DataValue::Int64Array(Vec::new())).unwrap(),
            DataValue::GeoEcefArray(Vec::new())
        );
        assert_eq!(
            coerce_value("g3", &geo3d_multi(), DataValue::Float64Array(Vec::new())).unwrap(),
            DataValue::GeoEcefArray(Vec::new())
        );
        assert!(
            coerce_value(
                "g3",
                &geo3d_multi(),
                DataValue::Float64Array(vec![1.0, 2.0, 3.0])
            )
            .is_err()
        );
        assert!(
            coerce_value(
                "g3",
                &geo3d_multi(),
                DataValue::GeoArray(vec![crate::data::GeoPoint::new(35.1, 139.0)])
            )
            .is_err()
        );
    }

    #[test]
    fn single_valued_geo3d_rejects_arrays() {
        let err = coerce_value(
            "g3",
            &geo3d(),
            DataValue::GeoEcefArray(vec![crate::data::GeoEcefPoint::new(1.0, 2.0, 3.0)]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("multi_valued = true"), "{err}");
    }

    #[test]
    fn bytes_passthrough() {
        assert_eq!(
            coerce_value("f", &bytes(), DataValue::Bytes(b"hi".to_vec(), None)).unwrap(),
            DataValue::Bytes(b"hi".to_vec(), None)
        );
        assert_eq!(
            coerce_value(
                "f",
                &bytes(),
                DataValue::Bytes(b"hi".to_vec(), Some("image/jpeg".to_string()))
            )
            .unwrap(),
            DataValue::Bytes(b"hi".to_vec(), Some("image/jpeg".to_string()))
        );
    }

    #[test]
    fn bytes_decodes_base64_text() {
        // base64("hi") == "aGk="
        assert_eq!(
            coerce_value("f", &bytes(), DataValue::Text("aGk=".to_string())).unwrap(),
            DataValue::Bytes(b"hi".to_vec(), None)
        );
    }

    #[test]
    fn bytes_rejects_invalid_base64_text() {
        assert!(
            coerce_value(
                "f",
                &bytes(),
                DataValue::Text("not valid base64!!".to_string())
            )
            .is_err()
        );
    }

    #[test]
    fn bytes_rejects_other_types() {
        assert!(coerce_value("f", &bytes(), DataValue::Int64(42)).is_err());
        assert!(coerce_value("f", &bytes(), DataValue::Bool(true)).is_err());
    }
}
