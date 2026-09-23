//! Conversion between [`laurus::Document`] and the protobuf `Document` message.
//!
//! [`to_proto`] and [`from_proto`] handle the top-level document.
//! [`data_value_to_proto`] and [`data_value_from_proto`] convert individual
//! [`DataValue`] fields and are also reused by the HTTP gateway when it
//! lowers JSON-derived [`DataValue`]s into the proto wire format.

use std::collections::HashMap;

use laurus::{DataValue, Document};

use crate::proto::laurus::v1;

/// Convert a laurus Document into a proto Document.
///
/// The internal `_id` system field is excluded from the output since it
/// is already available as the `id` field on the enclosing message.
pub fn to_proto(doc: &Document) -> v1::Document {
    let fields: HashMap<String, v1::Value> = doc
        .fields
        .iter()
        .filter(|(k, _)| k.as_str() != "_id")
        .map(|(k, v)| (k.clone(), data_value_to_proto(v)))
        .collect();
    v1::Document { fields }
}

/// Convert a proto Document into a laurus Document.
pub fn from_proto(proto: &v1::Document) -> Document {
    let fields: HashMap<String, DataValue> = proto
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), data_value_from_proto(v)))
        .collect();
    Document { fields }
}

/// Convert a [`DataValue`] into a proto `Value`.
///
/// Used by both the gRPC document path ([`to_proto`]) and the HTTP gateway
/// JSON path (after [`laurus::engine::type_inference::infer_from_json`]).
///
/// # Arguments
///
/// * `val` - The [`DataValue`] to convert.
pub fn data_value_to_proto(val: &DataValue) -> v1::Value {
    use v1::value::Kind;
    let kind = match val {
        DataValue::Null => Some(Kind::NullValue(true)),
        DataValue::Bool(b) => Some(Kind::BoolValue(*b)),
        DataValue::Int64(i) => Some(Kind::Int64Value(*i)),
        DataValue::Float64(f) => Some(Kind::Float64Value(*f)),
        DataValue::Text(s) => Some(Kind::TextValue(s.clone())),
        DataValue::Bytes(b, _mime) => Some(Kind::BytesValue(b.clone())),
        DataValue::Vector(v) => Some(Kind::VectorValue(v1::VectorValue { values: v.clone() })),
        DataValue::DateTime(dt) => Some(Kind::DatetimeValue(dt.timestamp_micros())),
        DataValue::Geo(p) => Some(Kind::GeoValue(v1::GeoPoint {
            latitude: p.lat,
            longitude: p.lon,
        })),
        DataValue::GeoEcef(p) => Some(Kind::Geo3dValue(v1::Geo3dPoint {
            x: p.x,
            y: p.y,
            z: p.z,
        })),
        DataValue::Int64Array(arr) => Some(Kind::Int64ArrayValue(v1::Int64ArrayValue {
            values: arr.clone(),
        })),
        DataValue::Float64Array(arr) => Some(Kind::Float64ArrayValue(v1::Float64ArrayValue {
            values: arr.clone(),
        })),
        DataValue::GeoArray(arr) => Some(Kind::GeoArrayValue(v1::GeoArrayValue {
            values: arr
                .iter()
                .map(|p| v1::GeoPoint {
                    latitude: p.lat,
                    longitude: p.lon,
                })
                .collect(),
        })),
        DataValue::GeoEcefArray(arr) => Some(Kind::Geo3dArrayValue(v1::Geo3dArrayValue {
            values: arr
                .iter()
                .map(|p| v1::Geo3dPoint {
                    x: p.x,
                    y: p.y,
                    z: p.z,
                })
                .collect(),
        })),
        DataValue::DateTimeArray(arr) => Some(Kind::DatetimeArrayValue(v1::DatetimeArrayValue {
            values: arr.iter().map(|dt| dt.timestamp_micros()).collect(),
        })),
    };
    v1::Value { kind }
}

/// Convert a proto `Value` into a [`DataValue`].
///
/// # Arguments
///
/// * `val` - The proto `Value` to convert.
pub fn data_value_from_proto(val: &v1::Value) -> DataValue {
    use v1::value::Kind;
    match &val.kind {
        Some(Kind::NullValue(_)) => DataValue::Null,
        Some(Kind::BoolValue(b)) => DataValue::Bool(*b),
        Some(Kind::Int64Value(i)) => DataValue::Int64(*i),
        Some(Kind::Float64Value(f)) => DataValue::Float64(*f),
        Some(Kind::TextValue(s)) => DataValue::Text(s.clone()),
        Some(Kind::BytesValue(b)) => DataValue::Bytes(b.clone(), None),
        Some(Kind::VectorValue(v)) => DataValue::Vector(v.values.clone()),
        Some(Kind::DatetimeValue(us)) => DataValue::DateTime(datetime_from_micros(*us)),
        Some(Kind::GeoValue(g)) => DataValue::Geo(geo_point_from_proto(g)),
        Some(Kind::Geo3dValue(p)) => DataValue::GeoEcef(laurus::GeoEcefPoint::new(p.x, p.y, p.z)),
        Some(Kind::Int64ArrayValue(arr)) => DataValue::Int64Array(arr.values.clone()),
        Some(Kind::Float64ArrayValue(arr)) => DataValue::Float64Array(arr.values.clone()),
        Some(Kind::GeoArrayValue(arr)) => {
            DataValue::GeoArray(arr.values.iter().map(geo_point_from_proto).collect())
        }
        Some(Kind::Geo3dArrayValue(arr)) => DataValue::GeoEcefArray(
            arr.values
                .iter()
                .map(|p| laurus::GeoEcefPoint::new(p.x, p.y, p.z))
                .collect(),
        ),
        Some(Kind::DatetimeArrayValue(arr)) => DataValue::DateTimeArray(
            arr.values
                .iter()
                .map(|us| datetime_from_micros(*us))
                .collect(),
        ),
        None => DataValue::Null,
    }
}

/// Convert proto Unix micro-seconds into a UTC datetime, falling back to the
/// epoch for a value outside chrono's range — the lenient behavior the
/// `DatetimeValue` arm has always had. Uses `from_timestamp_micros` rather
/// than a hand-rolled `/` + `%` split, which truncated toward zero and
/// collapsed every pre-1970 instant onto the epoch.
fn datetime_from_micros(us: i64) -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::from_timestamp_micros(us).unwrap_or_default()
}

/// Convert a proto `GeoPoint` into a [`laurus::GeoPoint`], falling back to
/// `(0, 0)` for out-of-range coordinates — the lenient behavior the
/// single-valued `GeoValue` arm has always had; each element of a
/// `GeoArrayValue` is treated the same way.
fn geo_point_from_proto(g: &v1::GeoPoint) -> laurus::GeoPoint {
    laurus::GeoPoint::try_new(g.latitude, g.longitude)
        .unwrap_or_else(|_| laurus::GeoPoint::new(0.0, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use laurus::GeoEcefPoint;

    /// `DataValue::GeoEcef` round-trips through the new `Geo3dValue`
    /// proto kind added in #305 (it used to be encoded as a `GeoValue`
    /// fallback that quietly lost the `z` component).
    #[test]
    fn data_value_geo_ecef_round_trip() {
        let original =
            DataValue::GeoEcef(GeoEcefPoint::new(1_234_567.0, -2_345_678.0, 3_456_789.5));
        let proto = data_value_to_proto(&original);
        match &proto.kind {
            Some(v1::value::Kind::Geo3dValue(p)) => {
                assert_eq!(p.x, 1_234_567.0);
                assert_eq!(p.y, -2_345_678.0);
                assert_eq!(p.z, 3_456_789.5);
            }
            other => panic!("expected Geo3dValue, got {other:?}"),
        }
        let back = data_value_from_proto(&proto);
        assert_eq!(back, original);
    }

    /// #1174: multi-valued geo values use the dedicated `GeoArrayValue` /
    /// `Geo3dArrayValue` kinds and round-trip element-wise, including the
    /// empty list.
    #[test]
    fn data_value_geo_arrays_round_trip() {
        let geo = DataValue::GeoArray(vec![
            laurus::GeoPoint::new(35.1, 139.0),
            laurus::GeoPoint::new(-33.9, 151.2),
        ]);
        let proto = data_value_to_proto(&geo);
        match &proto.kind {
            Some(v1::value::Kind::GeoArrayValue(a)) => {
                assert_eq!(a.values.len(), 2);
                assert_eq!(a.values[1].latitude, -33.9);
                assert_eq!(a.values[1].longitude, 151.2);
            }
            other => panic!("expected GeoArrayValue, got {other:?}"),
        }
        assert_eq!(data_value_from_proto(&proto), geo);

        let ecef = DataValue::GeoEcefArray(vec![
            GeoEcefPoint::new(1.0, 2.0, 3.0),
            GeoEcefPoint::new(-4.0, 5.0, -6.0),
        ]);
        let proto = data_value_to_proto(&ecef);
        match &proto.kind {
            Some(v1::value::Kind::Geo3dArrayValue(a)) => {
                assert_eq!(a.values.len(), 2);
                assert_eq!(a.values[1].z, -6.0);
            }
            other => panic!("expected Geo3dArrayValue, got {other:?}"),
        }
        assert_eq!(data_value_from_proto(&ecef_proto_clone(&proto)), ecef);

        for empty in [
            DataValue::GeoArray(Vec::new()),
            DataValue::GeoEcefArray(Vec::new()),
        ] {
            assert_eq!(data_value_from_proto(&data_value_to_proto(&empty)), empty);
        }
    }

    fn ecef_proto_clone(v: &v1::Value) -> v1::Value {
        v.clone()
    }

    /// #1184: multi-valued datetimes use the dedicated `DatetimeArrayValue`
    /// kind (Unix micros per element) and round-trip element-wise, including
    /// sub-second, pre-1970 and empty.
    #[test]
    fn data_value_datetime_arrays_round_trip() {
        let value = DataValue::DateTimeArray(vec![
            chrono::DateTime::from_timestamp_micros(1_700_000_000_500_000).unwrap(),
            chrono::DateTime::from_timestamp_micros(-86_400_000_001).unwrap(),
        ]);
        let proto = data_value_to_proto(&value);
        match &proto.kind {
            Some(v1::value::Kind::DatetimeArrayValue(a)) => {
                assert_eq!(a.values, vec![1_700_000_000_500_000, -86_400_000_001]);
            }
            other => panic!("expected DatetimeArrayValue, got {other:?}"),
        }
        assert_eq!(data_value_from_proto(&proto), value);

        let empty = DataValue::DateTimeArray(Vec::new());
        assert_eq!(data_value_from_proto(&data_value_to_proto(&empty)), empty);
    }

    /// Regression: the scalar `DatetimeValue` decoder split micros with
    /// truncating `/` and `%`, so any pre-1970 instant produced a negative
    /// nanosecond count and collapsed onto the epoch.
    #[test]
    fn data_value_datetime_pre_1970_round_trips() {
        let original =
            DataValue::DateTime(chrono::DateTime::from_timestamp_micros(-86_400_000_001).unwrap());
        let back = data_value_from_proto(&data_value_to_proto(&original));
        assert_eq!(back, original);
    }

    /// `DataValue::Geo` continues to use the 2D `GeoValue` proto kind,
    /// so adding the `Geo3dValue` variant in #305 cannot disturb the
    /// existing 2D wire format.
    #[test]
    fn data_value_geo_still_uses_2d_kind() {
        let original = DataValue::Geo(laurus::lexical::GeoPoint::new(35.1, 139.0));
        let proto = data_value_to_proto(&original);
        assert!(matches!(proto.kind, Some(v1::value::Kind::GeoValue(_))));
        let back = data_value_from_proto(&proto);
        assert_eq!(back, original);
    }
}
