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
        DataValue::BoolArray(arr) => Some(Kind::BoolArrayValue(v1::BoolArrayValue {
            values: arr.clone(),
        })),
        DataValue::TextArray(arr) => Some(Kind::TextArrayValue(v1::TextArrayValue {
            values: arr.clone(),
        })),
        // The per-element MIME type is dropped, matching the scalar
        // `DataValue::Bytes` arm above -- the wire format has no place to
        // carry it for either shape (Issue #1176).
        DataValue::BytesArray(arr) => Some(Kind::BytesArrayValue(v1::BytesArrayValue {
            values: arr.iter().map(|(b, _mime)| b.clone()).collect(),
        })),
        DataValue::VectorArray(arr) => Some(Kind::VectorArrayValue(v1::VectorArrayValue {
            dimension: arr.first().map_or(0, Vec::len) as u32,
            values: arr.iter().flatten().copied().collect(),
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
        Some(Kind::BoolArrayValue(arr)) => DataValue::BoolArray(arr.values.clone()),
        Some(Kind::TextArrayValue(arr)) => DataValue::TextArray(arr.values.clone()),
        // Mime is always `None`, matching the scalar `BytesValue` arm above.
        Some(Kind::BytesArrayValue(arr)) => {
            DataValue::BytesArray(arr.values.iter().map(|b| (b.clone(), None)).collect())
        }
        Some(Kind::VectorArrayValue(arr)) => DataValue::VectorArray(vector_array_from_proto(arr)),
        None => DataValue::Null,
    }
}

/// Split a packed `VectorArrayValue` into its vectors.
///
/// A malformed value is not repaired: a `values` length that is not a
/// multiple of `dimension` leaves a shorter last vector, and a zero
/// `dimension` keeps all values as one vector, so the engine's dimension
/// check rejects the document instead of indexing silently altered data.
///
/// # Arguments
///
/// * `arr` - The packed proto value.
///
/// # Returns
///
/// The vectors in their original order.
pub fn vector_array_from_proto(arr: &v1::VectorArrayValue) -> Vec<Vec<f32>> {
    match arr.dimension as usize {
        _ if arr.values.is_empty() => Vec::new(),
        0 => vec![arr.values.clone()],
        dim => arr.values.chunks(dim).map(<[f32]>::to_vec).collect(),
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

    /// #1180: multi-valued booleans use the dedicated `BoolArrayValue` kind
    /// and round-trip element-wise, including the empty list.
    #[test]
    fn data_value_bool_arrays_round_trip() {
        let value = DataValue::BoolArray(vec![true, false, true]);
        let proto = data_value_to_proto(&value);
        match &proto.kind {
            Some(v1::value::Kind::BoolArrayValue(a)) => {
                assert_eq!(a.values, vec![true, false, true]);
            }
            other => panic!("expected BoolArrayValue, got {other:?}"),
        }
        assert_eq!(data_value_from_proto(&proto), value);

        let empty = DataValue::BoolArray(Vec::new());
        assert_eq!(data_value_from_proto(&data_value_to_proto(&empty)), empty);
    }

    /// #1175: multi-valued text uses the dedicated `TextArrayValue` kind and
    /// round-trips element-wise, including Unicode, an empty string and the
    /// empty list.
    #[test]
    fn data_value_text_arrays_round_trip() {
        let value = DataValue::TextArray(vec![
            "hello world".to_string(),
            String::new(),
            "日本語".to_string(),
        ]);
        let proto = data_value_to_proto(&value);
        match &proto.kind {
            Some(v1::value::Kind::TextArrayValue(a)) => {
                assert_eq!(a.values, vec!["hello world", "", "日本語"]);
            }
            other => panic!("expected TextArrayValue, got {other:?}"),
        }
        assert_eq!(data_value_from_proto(&proto), value);

        let empty = DataValue::TextArray(Vec::new());
        assert_eq!(data_value_from_proto(&data_value_to_proto(&empty)), empty);
    }

    /// #1176: multi-valued bytes use the dedicated `BytesArrayValue` kind
    /// and round-trip element-wise, including the empty list. The MIME type
    /// carried per element is dropped over the wire, matching the existing
    /// (lossy) behavior of the scalar `DataValue::Bytes` <-> `BytesValue`
    /// conversion just above.
    #[test]
    fn data_value_bytes_arrays_round_trip() {
        let value = DataValue::BytesArray(vec![
            (b"hello".to_vec(), Some("text/plain".to_string())),
            (Vec::new(), None),
        ]);
        let proto = data_value_to_proto(&value);
        match &proto.kind {
            Some(v1::value::Kind::BytesArrayValue(a)) => {
                assert_eq!(a.values, vec![b"hello".to_vec(), Vec::new()]);
            }
            other => panic!("expected BytesArrayValue, got {other:?}"),
        }
        let back = data_value_from_proto(&proto);
        assert_eq!(
            back,
            DataValue::BytesArray(vec![(b"hello".to_vec(), None), (Vec::new(), None)]),
            "mime is not carried over the wire, same as the scalar Bytes arm"
        );

        let empty = DataValue::BytesArray(Vec::new());
        assert_eq!(data_value_from_proto(&data_value_to_proto(&empty)), empty);
    }

    /// #1177: token vectors are packed row-major into `VectorArrayValue`
    /// and split back by `dimension`, including the empty list.
    #[test]
    fn data_value_vector_arrays_round_trip() {
        let value = DataValue::VectorArray(vec![vec![0.5, -1.0, 2.0], vec![3.0, 0.0, -0.25]]);
        let proto = data_value_to_proto(&value);
        match &proto.kind {
            Some(v1::value::Kind::VectorArrayValue(a)) => {
                assert_eq!(a.dimension, 3);
                assert_eq!(a.values, vec![0.5, -1.0, 2.0, 3.0, 0.0, -0.25]);
            }
            other => panic!("expected VectorArrayValue, got {other:?}"),
        }
        assert_eq!(data_value_from_proto(&proto), value);

        let empty = DataValue::VectorArray(Vec::new());
        assert_eq!(data_value_from_proto(&data_value_to_proto(&empty)), empty);
    }

    /// #1177: a malformed packed value is passed on in a shape the engine
    /// rejects (ragged, or one vector of the wrong length), never repaired.
    #[test]
    fn malformed_vector_array_value_is_not_repaired() {
        let ragged = v1::Value {
            kind: Some(v1::value::Kind::VectorArrayValue(v1::VectorArrayValue {
                dimension: 2,
                values: vec![1.0, 2.0, 3.0],
            })),
        };
        assert_eq!(
            data_value_from_proto(&ragged),
            DataValue::VectorArray(vec![vec![1.0, 2.0], vec![3.0]])
        );

        let no_dimension = v1::Value {
            kind: Some(v1::value::Kind::VectorArrayValue(v1::VectorArrayValue {
                dimension: 0,
                values: vec![1.0, 2.0],
            })),
        };
        assert_eq!(
            data_value_from_proto(&no_dimension),
            DataValue::VectorArray(vec![vec![1.0, 2.0]])
        );
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
