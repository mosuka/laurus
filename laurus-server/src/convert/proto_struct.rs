//! Conversions between JSON and `google.protobuf.Struct`, the form
//! [`ComponentConfig::config`] carries an analyzer component in (Issue #1478).
//!
//! JSON goes into a `Struct` as it is. A `Struct` coming in is read strictly
//! ([`proto_struct_to_json`]), since the server deserializes the result into
//! the core component types; one going out to an HTTP response is read
//! leniently ([`proto_struct_to_json_lenient`]), so that a response never
//! fails. Both write a whole number as a JSON integer, since a protobuf
//! number is always a double and an integer field such as `min_gram`
//! rejects a JSON float.
//!
//! [`ComponentConfig::config`]: crate::proto::laurus::v1::ComponentConfig::config

use prost_types::value::Kind;
use prost_types::{ListValue, Struct, Value as ProtoValue};
use serde_json::{Map, Number, Value};

/// 2^64, the first whole number `u64` cannot hold. A whole number is
/// checked against it before `as u64`, which would saturate instead.
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

/// -2^63, the smallest whole number `i64` can hold.
const MINUS_TWO_POW_63: f64 = -9_223_372_036_854_775_808.0;

/// Convert a JSON object to a `Struct`.
pub(crate) fn json_object_to_proto_struct(obj: &Map<String, Value>) -> Struct {
    Struct {
        fields: obj
            .iter()
            .map(|(key, value)| (key.clone(), json_to_proto_value(value)))
            .collect(),
    }
}

/// Convert a JSON value to a protobuf `Value`. A JSON number becomes a
/// double, which loses precision only above 2^53.
fn json_to_proto_value(json: &Value) -> ProtoValue {
    let kind = match json {
        Value::Null => Kind::NullValue(0),
        Value::Bool(b) => Kind::BoolValue(*b),
        Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(f64::NAN)),
        Value::String(s) => Kind::StringValue(s.clone()),
        Value::Array(items) => Kind::ListValue(ListValue {
            values: items.iter().map(json_to_proto_value).collect(),
        }),
        Value::Object(obj) => Kind::StructValue(json_object_to_proto_struct(obj)),
    };
    ProtoValue { kind: Some(kind) }
}

/// Convert a `Struct` the server received to a JSON object, strictly.
///
/// # Errors
///
/// Returns an error, naming the key (`mapping.a`, `words[1]`), for a number
/// that is not finite or a value with no kind, which JSON cannot hold.
pub(crate) fn proto_struct_to_json(proto: &Struct) -> Result<Map<String, Value>, String> {
    proto
        .fields
        .iter()
        .map(|(key, value)| {
            proto_value_to_json(value)
                .map(|json| (key.clone(), json))
                .map_err(|e| prefix_key(key, &e))
        })
        .collect()
}

fn proto_value_to_json(proto: &ProtoValue) -> Result<Value, String> {
    match &proto.kind {
        None => Err(": the value has no kind".to_string()),
        Some(Kind::NullValue(_)) => Ok(Value::Null),
        Some(Kind::BoolValue(b)) => Ok(Value::Bool(*b)),
        Some(Kind::NumberValue(n)) => {
            number_to_json(*n).ok_or_else(|| format!(": the number must be finite, got {n}"))
        }
        Some(Kind::StringValue(s)) => Ok(Value::String(s.clone())),
        Some(Kind::ListValue(list)) => list
            .values
            .iter()
            .enumerate()
            .map(|(i, value)| proto_value_to_json(value).map_err(|e| format!("[{i}]{e}")))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Some(Kind::StructValue(s)) => s
            .fields
            .iter()
            .map(|(key, value)| {
                proto_value_to_json(value)
                    .map(|json| (key.clone(), json))
                    .map_err(|e| format!(".{}", prefix_key(key, &e)))
            })
            .collect::<Result<Map<_, _>, _>>()
            .map(Value::Object),
    }
}

/// Put `key` in front of an error that starts with the rest of the path
/// (`[1]: ...`, `.a: ...` or `: ...`).
fn prefix_key(key: &str, error: &str) -> String {
    format!("{key}{error}")
}

/// Convert a `Struct` going out to an HTTP response to a JSON object. A
/// number that is not finite and a value with no kind become `null`.
pub(crate) fn proto_struct_to_json_lenient(proto: &Struct) -> Map<String, Value> {
    proto
        .fields
        .iter()
        .map(|(key, value)| (key.clone(), proto_value_to_json_lenient(value)))
        .collect()
}

fn proto_value_to_json_lenient(proto: &ProtoValue) -> Value {
    match &proto.kind {
        None | Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(*b),
        Some(Kind::NumberValue(n)) => number_to_json(*n).unwrap_or(Value::Null),
        Some(Kind::StringValue(s)) => Value::String(s.clone()),
        Some(Kind::ListValue(list)) => Value::Array(
            list.values
                .iter()
                .map(proto_value_to_json_lenient)
                .collect(),
        ),
        Some(Kind::StructValue(s)) => Value::Object(proto_struct_to_json_lenient(s)),
    }
}

/// Convert a double to a JSON number: a whole number that `u64` or `i64`
/// holds becomes an integer, anything else stays a float. `None` if the
/// double is not finite.
fn number_to_json(n: f64) -> Option<Value> {
    if !n.is_finite() {
        return None;
    }
    if n.fract() == 0.0 {
        if (0.0..TWO_POW_64).contains(&n) {
            return Some(Value::from(n as u64));
        }
        if (MINUS_TWO_POW_63..0.0).contains(&n) {
            return Some(Value::from(n as i64));
        }
    }
    Number::from_f64(n).map(Value::Number)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn number(n: f64) -> ProtoValue {
        ProtoValue {
            kind: Some(Kind::NumberValue(n)),
        }
    }

    fn one_field(key: &str, value: ProtoValue) -> Struct {
        Struct {
            fields: [(key.to_string(), value)].into_iter().collect(),
        }
    }

    #[test]
    fn json_round_trips_through_a_struct() {
        let json = json!({
            "type": "stop",
            "words": ["new,york", " padded ", ""],
            "mapping": {"type": "t", "a": "b"},
            "flag": false,
            "n": 2,
            "x": 0.1,
            "none": null,
            "empty_list": [],
            "empty_table": {},
        });
        let proto = json_object_to_proto_struct(json.as_object().unwrap());
        assert_eq!(Value::Object(proto_struct_to_json(&proto).unwrap()), json);
        assert_eq!(Value::Object(proto_struct_to_json_lenient(&proto)), json);
    }

    #[test]
    fn whole_numbers_become_integers_without_saturating() {
        for (n, expected) in [
            (2.0, json!(2)),
            (0.0, json!(0)),
            (-1.0, json!(-1)),
            (2.5, json!(2.5)),
            (1e20, json!(1e20)),
            (TWO_POW_64, json!(TWO_POW_64)),
            (MINUS_TWO_POW_63, json!(i64::MIN)),
            (-1e20, json!(-1e20)),
        ] {
            let proto = one_field("n", number(n));
            let json = proto_struct_to_json(&proto).unwrap();
            assert_eq!(json["n"], expected, "{n}");
            assert_eq!(json["n"].is_f64(), expected.is_f64(), "{n}");
        }
    }

    #[test]
    fn a_non_finite_number_is_rejected_naming_its_key() {
        for n in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let list = ProtoValue {
                kind: Some(Kind::ListValue(ListValue {
                    values: vec![number(1.0), number(n)],
                })),
            };
            let nested = ProtoValue {
                kind: Some(Kind::StructValue(one_field("words", list))),
            };
            let err = proto_struct_to_json(&one_field("mapping", nested)).unwrap_err();
            assert!(err.starts_with("mapping.words[1]: "), "{err}");
            assert!(err.contains("finite"), "{err}");
            let lenient = proto_struct_to_json_lenient(&one_field("n", number(n)));
            assert_eq!(lenient["n"], Value::Null);
        }
    }

    #[test]
    fn a_value_with_no_kind_is_rejected() {
        let proto = one_field("form", ProtoValue { kind: None });
        let err = proto_struct_to_json(&proto).unwrap_err();
        assert_eq!(err, "form: the value has no kind");
        assert_eq!(proto_struct_to_json_lenient(&proto)["form"], Value::Null);
    }
}
