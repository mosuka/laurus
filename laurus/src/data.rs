use std::collections::HashMap;

use chrono::{DateTime, Utc};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

use crate::error::LaurusError;

/// Helper for archiving DateTime as micros timestamp (i64)
pub struct MicroSeconds;

impl rkyv::with::ArchiveWith<DateTime<Utc>> for MicroSeconds {
    type Archived = rkyv::Archived<i64>;
    type Resolver = ();

    fn resolve_with(field: &DateTime<Utc>, _: (), out: rkyv::Place<Self::Archived>) {
        let ts = field.timestamp_micros();
        ts.resolve((), out);
    }
}

impl<S: rkyv::rancor::Fallible + ?Sized> rkyv::with::SerializeWith<DateTime<Utc>, S>
    for MicroSeconds
{
    fn serialize_with(
        field: &DateTime<Utc>,
        serializer: &mut S,
    ) -> Result<Self::Resolver, S::Error> {
        RkyvSerialize::serialize(&field.timestamp_micros(), serializer)
    }
}

impl<D: rkyv::rancor::Fallible + ?Sized>
    rkyv::with::DeserializeWith<rkyv::Archived<i64>, DateTime<Utc>, D> for MicroSeconds
{
    fn deserialize_with(
        archived: &rkyv::Archived<i64>,
        _deserializer: &mut D,
    ) -> Result<DateTime<Utc>, D::Error> {
        let ts: i64 = (*archived).into();
        // DateTime::from_timestamp_micros returns None only for out-of-range values.
        // Fall back to UNIX epoch for corrupted timestamps to avoid panics.
        Ok(DateTime::from_timestamp_micros(ts).unwrap_or_default())
    }
}

/// A geographical point on the Earth's surface, in WGS84 latitude /
/// longitude degrees.
///
/// `lat` is bounded to `[-90, 90]` and `lon` to `[-180, 180]`. Use
/// [`GeoPoint::try_new`] to validate at construction; the infallible
/// [`GeoPoint::new`] only debug-asserts and is intended for callers that
/// have already validated their input (or for hot paths inside the engine
/// where validation has happened upstream).
#[derive(
    Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize,
)]
pub struct GeoPoint {
    /// Latitude in degrees, bounded to `[-90, 90]`.
    pub lat: f64,
    /// Longitude in degrees, bounded to `[-180, 180]`.
    pub lon: f64,
}

impl GeoPoint {
    /// Construct a `GeoPoint` without validation.
    ///
    /// Debug builds assert that `lat` and `lon` are inside their valid
    /// ranges; release builds skip the check, on the assumption that the
    /// caller validated the values upstream (or constructed them from a
    /// trusted source like a previous serialized index). Prefer
    /// [`GeoPoint::try_new`] when handling user-supplied input.
    #[inline]
    pub fn new(lat: f64, lon: f64) -> Self {
        debug_assert!(
            (-90.0..=90.0).contains(&lat),
            "GeoPoint latitude {lat} out of range [-90, 90]"
        );
        debug_assert!(
            (-180.0..=180.0).contains(&lon),
            "GeoPoint longitude {lon} out of range [-180, 180]"
        );
        GeoPoint { lat, lon }
    }

    /// Construct a `GeoPoint`, validating that `lat` and `lon` lie inside
    /// their canonical WGS84 ranges. Returns `LaurusError::other` for
    /// out-of-range or `NaN` inputs.
    pub fn try_new(lat: f64, lon: f64) -> crate::error::Result<Self> {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(LaurusError::other(format!(
                "Invalid latitude: {lat} (must be between -90 and 90)"
            )));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(LaurusError::other(format!(
                "Invalid longitude: {lon} (must be between -180 and 180)"
            )));
        }
        Ok(GeoPoint { lat, lon })
    }

    /// Great-circle distance to another point in meters, using the
    /// Haversine formula on a sphere of mean Earth radius (6 371 000 m).
    pub fn distance_to(&self, other: &GeoPoint) -> f64 {
        const EARTH_RADIUS_M: f64 = 6_371_000.0;

        let lat1_rad = self.lat.to_radians();
        let lat2_rad = other.lat.to_radians();
        let delta_lat = (other.lat - self.lat).to_radians();
        let delta_lon = (other.lon - self.lon).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        EARTH_RADIUS_M * c
    }

    /// Initial bearing toward `other`, in degrees in `[0, 360)` clockwise
    /// from true north.
    pub fn bearing_to(&self, other: &GeoPoint) -> f64 {
        let lat1_rad = self.lat.to_radians();
        let lat2_rad = other.lat.to_radians();
        let delta_lon = (other.lon - self.lon).to_radians();

        let y = delta_lon.sin() * lat2_rad.cos();
        let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * delta_lon.cos();

        let bearing_rad = y.atan2(x);
        (bearing_rad.to_degrees() + 360.0) % 360.0
    }

    /// Whether this point falls inside the closed lat/lon rectangle
    /// `[min_lat, max_lat] × [min_lon, max_lon]`.
    pub fn within_bounds(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> bool {
        self.lat >= min_lat && self.lat <= max_lat && self.lon >= min_lon && self.lon <= max_lon
    }
}

/// A 3D point in Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates.
///
/// All three components are in meters, with the origin at the Earth's
/// center of mass. Suitable for true 3D geospatial queries (drone
/// proximity, satellite tracking, indoor 3D positioning) where a 2D
/// `GeoPoint` would lose the altitude dimension or wrap incorrectly near
/// the poles.
#[derive(
    Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize,
)]
pub struct GeoEcefPoint {
    /// X axis: meters along the equatorial plane through 0° longitude.
    pub x: f64,
    /// Y axis: meters along the equatorial plane through 90°E longitude.
    pub y: f64,
    /// Z axis: meters along the Earth's rotation axis toward the North Pole.
    pub z: f64,
}

impl GeoEcefPoint {
    /// Construct a `GeoEcefPoint` from raw Cartesian components.
    ///
    /// No range validation is performed: ECEF coordinates have no fixed
    /// upper bound (a satellite is a valid 3D point well outside the
    /// surface), and the natural sentinels `±INFINITY` are accepted as
    /// "unbounded" markers in queries.
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        GeoEcefPoint { x, y, z }
    }
}

/// The unified value type for fields in a document.
///
/// This enum merges the concepts of `FieldValue` (from Lexical Index) and
/// `VectorValue` (from Vector Index).
#[derive(
    Debug, Clone, PartialEq, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize,
)]
pub enum DataValue {
    // --- Primitive Types ---
    Null,
    Bool(bool),
    Int64(i64),
    Float64(f64),

    // --- Complex / Searchable Types ---
    /// Text content. Whether this is tokenized or treated as a keyword
    /// is determined by the schema's [`FieldOption`](crate::lexical::core::field::FieldOption)
    /// and the configured [`Analyzer`](crate::analysis::analyzer::analyzer::Analyzer).
    Text(String),

    /// Binary content (image, audio, etc.) to be embedded.
    /// Contains the raw bytes and an optional MIME type.
    Bytes(Vec<u8>, Option<String>),

    /// Pre-computed vector.
    Vector(Vec<f32>),

    /// Date and time in UTC.
    DateTime(#[rkyv(with = MicroSeconds)] chrono::DateTime<chrono::Utc>),

    /// 2D geographical point (WGS84 latitude / longitude).
    Geo(GeoPoint),

    /// 3D Earth-Centered Earth-Fixed (ECEF) Cartesian point in meters.
    GeoEcef(GeoEcefPoint),

    /// Multi-valued 64-bit signed integers.
    ///
    /// Used by fields declared with
    /// [`IntegerOption::multi_valued`](crate::lexical::core::field::IntegerOption::multi_valued)
    /// set to `true`. Range queries match a document if **any** value in
    /// the array satisfies the predicate (Lucene-style "any match"
    /// semantics with constant scoring).
    Int64Array(Vec<i64>),

    /// Multi-valued 64-bit floating-point numbers.
    ///
    /// Used by fields declared with
    /// [`FloatOption::multi_valued`](crate::lexical::core::field::FloatOption::multi_valued)
    /// set to `true`. Range queries match a document if **any** value in
    /// the array satisfies the predicate (Lucene-style "any match"
    /// semantics with constant scoring).
    Float64Array(Vec<f64>),

    // New variants must be APPENDED here, never inserted above: this enum is
    // rkyv-archived on disk (DocValues `.dv`, WAL v3) and rkyv numbers the
    // archived discriminants in declaration order, so inserting a variant
    // mid-enum would make existing segments decode as the wrong variant.
    /// Multi-valued 2D geographical points (Issue #1174).
    ///
    /// Used by fields declared with
    /// [`GeoOption::multi_valued`](crate::lexical::core::field::GeoOption::multi_valued)
    /// set to `true`. Distance and bounding-box queries match a document if
    /// **any** point satisfies the predicate, scoring it by its closest
    /// matching point.
    GeoArray(Vec<GeoPoint>),

    /// Multi-valued 3D ECEF Cartesian points (Issue #1174).
    ///
    /// Used by fields declared with
    /// [`Geo3dOption::multi_valued`](crate::lexical::core::field::Geo3dOption::multi_valued)
    /// set to `true`. Distance, bounding-box, and nearest queries match a
    /// document if **any** point satisfies the predicate, scoring it by its
    /// closest matching point.
    GeoEcefArray(Vec<GeoEcefPoint>),

    /// Multi-valued UTC datetimes (Issue #1184).
    ///
    /// Used by fields declared with
    /// [`DateTimeOption::multi_valued`](crate::lexical::core::field::DateTimeOption::multi_valued)
    /// set to `true`. Range queries match a document if **any** instant
    /// satisfies the predicate (Lucene-style "any match" semantics with
    /// constant scoring). Archived element-wise with the same
    /// micro-second wrapper as [`DataValue::DateTime`].
    DateTimeArray(#[rkyv(with = rkyv::with::Map<MicroSeconds>)] Vec<DateTime<Utc>>),

    /// Multi-valued booleans (Issue #1180).
    ///
    /// Used by fields declared with
    /// [`BooleanOption::multi_valued`](crate::lexical::core::field::BooleanOption::multi_valued)
    /// set to `true`. Unlike the other multi-valued types there are no BKD
    /// points: every element is indexed as its own `"true"` / `"false"`
    /// term posting, so a term query matches a document if **any** element
    /// equals the queried value, and repeated elements raise the term
    /// frequency (Lucene multi-valued parity) rather than the hit count.
    BoolArray(Vec<bool>),
}

impl DataValue {
    /// Returns the text value if this is a Text variant.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            DataValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the integer value if this is an Int64 variant.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            DataValue::Int64(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the float value if this is a Float64 variant.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            DataValue::Float64(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the boolean value if this is a Bool variant.
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            DataValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the datetime value if this is a DateTime variant.
    pub fn as_datetime(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        match self {
            DataValue::DateTime(dt) => Some(*dt),
            _ => None,
        }
    }

    /// Returns the vector data if this is a Vector variant.
    pub fn as_vector(&self) -> Option<&Vec<f32>> {
        match self {
            DataValue::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the bytes data if this is a Bytes variant.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            DataValue::Bytes(b, _) => Some(b),
            _ => None,
        }
    }

    /// Returns the geographical point if this is a `Geo` variant.
    pub fn as_geo(&self) -> Option<GeoPoint> {
        match self {
            DataValue::Geo(p) => Some(*p),
            _ => None,
        }
    }

    /// Returns the ECEF Cartesian point if this is a `GeoEcef` variant.
    pub fn as_geo_ecef(&self) -> Option<GeoEcefPoint> {
        match self {
            DataValue::GeoEcef(p) => Some(*p),
            _ => None,
        }
    }

    /// Returns the multi-valued integer slice if this is an `Int64Array` variant.
    pub fn as_int64_array(&self) -> Option<&[i64]> {
        match self {
            DataValue::Int64Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Returns the multi-valued float slice if this is a `Float64Array` variant.
    pub fn as_float64_array(&self) -> Option<&[f64]> {
        match self {
            DataValue::Float64Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Returns the multi-valued geo point slice if this is a `GeoArray` variant.
    pub fn as_geo_array(&self) -> Option<&[GeoPoint]> {
        match self {
            DataValue::GeoArray(arr) => Some(arr),
            _ => None,
        }
    }

    /// Returns the multi-valued ECEF point slice if this is a `GeoEcefArray` variant.
    pub fn as_geo_ecef_array(&self) -> Option<&[GeoEcefPoint]> {
        match self {
            DataValue::GeoEcefArray(arr) => Some(arr),
            _ => None,
        }
    }

    /// Returns the multi-valued datetime slice if this is a `DateTimeArray` variant.
    pub fn as_datetime_array(&self) -> Option<&[DateTime<Utc>]> {
        match self {
            DataValue::DateTimeArray(arr) => Some(arr),
            _ => None,
        }
    }

    /// Returns the multi-valued boolean slice if this is a `BoolArray` variant.
    pub fn as_bool_array(&self) -> Option<&[bool]> {
        match self {
            DataValue::BoolArray(arr) => Some(arr),
            _ => None,
        }
    }
}

// --- Conversions ---

impl From<String> for DataValue {
    fn from(v: String) -> Self {
        DataValue::Text(v)
    }
}

impl From<&str> for DataValue {
    fn from(v: &str) -> Self {
        DataValue::Text(v.to_string())
    }
}

impl From<i64> for DataValue {
    fn from(v: i64) -> Self {
        DataValue::Int64(v)
    }
}

impl From<i32> for DataValue {
    fn from(v: i32) -> Self {
        DataValue::Int64(v as i64)
    }
}

impl From<f64> for DataValue {
    fn from(v: f64) -> Self {
        DataValue::Float64(v)
    }
}

impl From<f32> for DataValue {
    fn from(v: f32) -> Self {
        DataValue::Float64(v as f64)
    }
}

impl From<bool> for DataValue {
    fn from(v: bool) -> Self {
        DataValue::Bool(v)
    }
}

impl From<chrono::DateTime<chrono::Utc>> for DataValue {
    fn from(dt: chrono::DateTime<chrono::Utc>) -> Self {
        DataValue::DateTime(dt)
    }
}

impl From<Vec<f32>> for DataValue {
    fn from(v: Vec<f32>) -> Self {
        DataValue::Vector(v)
    }
}

impl From<Vec<i64>> for DataValue {
    fn from(v: Vec<i64>) -> Self {
        DataValue::Int64Array(v)
    }
}

impl From<Vec<f64>> for DataValue {
    fn from(v: Vec<f64>) -> Self {
        DataValue::Float64Array(v)
    }
}

impl From<Vec<GeoPoint>> for DataValue {
    fn from(v: Vec<GeoPoint>) -> Self {
        DataValue::GeoArray(v)
    }
}

impl From<Vec<GeoEcefPoint>> for DataValue {
    fn from(v: Vec<GeoEcefPoint>) -> Self {
        DataValue::GeoEcefArray(v)
    }
}

impl From<Vec<DateTime<Utc>>> for DataValue {
    fn from(v: Vec<DateTime<Utc>>) -> Self {
        DataValue::DateTimeArray(v)
    }
}

impl From<Vec<bool>> for DataValue {
    fn from(v: Vec<bool>) -> Self {
        DataValue::BoolArray(v)
    }
}

impl From<GeoPoint> for DataValue {
    fn from(p: GeoPoint) -> Self {
        DataValue::Geo(p)
    }
}

impl From<GeoEcefPoint> for DataValue {
    fn from(p: GeoEcefPoint) -> Self {
        DataValue::GeoEcef(p)
    }
}

/// Unified Document structure.
///
/// A document is a pure data container — a collection of named fields,
/// each containing a [`DataValue`]. Document identity (external ID) is
/// managed by the [`Engine`](crate::Engine), not by the document itself.
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct Document {
    /// Field data.
    pub fields: HashMap<String, DataValue>,
}

impl Document {
    /// Create a new empty document.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    /// Get a reference to a field's value.
    pub fn get(&self, name: &str) -> Option<&DataValue> {
        self.fields.get(name)
    }

    /// Alias for [`get`](Self::get), provided for API consistency with field-based
    /// access patterns. Behaves identically to `get`.
    pub fn get_field(&self, name: &str) -> Option<&DataValue> {
        self.get(name)
    }

    /// Check if the document has a field.
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Get all field names.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if the document is empty.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn builder() -> DocumentBuilder {
        DocumentBuilder::default()
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
pub struct DocumentBuilder {
    fields: HashMap<String, DataValue>,
}

impl DocumentBuilder {
    /// Add a field to the document.
    pub fn add_field(mut self, name: impl Into<String>, value: impl Into<DataValue>) -> Self {
        self.fields.insert(name.into(), value.into());
        self
    }

    /// Add a text field.
    pub fn add_text(self, name: impl Into<String>, text: impl Into<String>) -> Self {
        self.add_field(name.into(), DataValue::Text(text.into()))
    }

    /// Add an integer field.
    pub fn add_integer(self, name: impl Into<String>, value: i64) -> Self {
        self.add_field(name.into(), DataValue::Int64(value))
    }

    /// Add a float field.
    pub fn add_float(self, name: impl Into<String>, value: f64) -> Self {
        self.add_field(name.into(), DataValue::Float64(value))
    }

    /// Add a boolean field.
    pub fn add_boolean(self, name: impl Into<String>, value: bool) -> Self {
        self.add_field(name.into(), DataValue::Bool(value))
    }

    /// Add a datetime field.
    pub fn add_datetime(self, name: impl Into<String>, value: DateTime<Utc>) -> Self {
        self.add_field(name.into(), DataValue::DateTime(value))
    }

    /// Add a vector field.
    pub fn add_vector(self, name: impl Into<String>, vector: Vec<f32>) -> Self {
        self.add_field(name.into(), DataValue::Vector(vector))
    }

    /// Add a 2D geographical field from `(lat, lon)` degrees.
    ///
    /// The values are passed through [`GeoPoint::new`], which debug-asserts
    /// the canonical WGS84 ranges. Use [`add_field`](Self::add_field) with
    /// `GeoPoint::try_new(...)?` if you need validation against
    /// user-supplied input.
    pub fn add_geo(self, name: impl Into<String>, lat: f64, lon: f64) -> Self {
        self.add_field(name.into(), DataValue::Geo(GeoPoint::new(lat, lon)))
    }

    /// Add a 3D Earth-Centered Earth-Fixed (ECEF) Cartesian field from
    /// raw `(x, y, z)` meters.
    pub fn add_geo_ecef(self, name: impl Into<String>, x: f64, y: f64, z: f64) -> Self {
        self.add_field(name.into(), DataValue::GeoEcef(GeoEcefPoint::new(x, y, z)))
    }

    /// Add a multi-valued integer field.
    ///
    /// The schema field must be declared with
    /// [`IntegerOption::multi_valued`](crate::lexical::core::field::IntegerOption::multi_valued)
    /// set to `true`. Range queries match if any value satisfies the
    /// predicate.
    pub fn add_int64_array(self, name: impl Into<String>, values: Vec<i64>) -> Self {
        self.add_field(name.into(), DataValue::Int64Array(values))
    }

    /// Add a multi-valued float field.
    ///
    /// The schema field must be declared with
    /// [`FloatOption::multi_valued`](crate::lexical::core::field::FloatOption::multi_valued)
    /// set to `true`. Range queries match if any value satisfies the
    /// predicate.
    pub fn add_float64_array(self, name: impl Into<String>, values: Vec<f64>) -> Self {
        self.add_field(name.into(), DataValue::Float64Array(values))
    }

    /// Add a multi-valued 2D geographical field.
    ///
    /// The schema field must be declared with
    /// [`GeoOption::multi_valued`](crate::lexical::core::field::GeoOption::multi_valued)
    /// set to `true`. Distance and bounding-box queries match if any point
    /// satisfies the predicate.
    pub fn add_geo_array(self, name: impl Into<String>, points: Vec<GeoPoint>) -> Self {
        self.add_field(name.into(), DataValue::GeoArray(points))
    }

    /// Add a multi-valued 3D ECEF Cartesian field.
    ///
    /// The schema field must be declared with
    /// [`Geo3dOption::multi_valued`](crate::lexical::core::field::Geo3dOption::multi_valued)
    /// set to `true`. Distance, bounding-box, and nearest queries match if
    /// any point satisfies the predicate.
    pub fn add_geo_ecef_array(self, name: impl Into<String>, points: Vec<GeoEcefPoint>) -> Self {
        self.add_field(name.into(), DataValue::GeoEcefArray(points))
    }

    /// Add a multi-valued datetime field.
    ///
    /// The schema field must be declared with
    /// [`DateTimeOption::multi_valued`](crate::lexical::core::field::DateTimeOption::multi_valued)
    /// set to `true`. Range queries match if any instant satisfies the
    /// predicate.
    pub fn add_datetime_array(self, name: impl Into<String>, values: Vec<DateTime<Utc>>) -> Self {
        self.add_field(name.into(), DataValue::DateTimeArray(values))
    }

    /// Add a multi-valued boolean field.
    ///
    /// The schema field must be declared with
    /// [`BooleanOption::multi_valued`](crate::lexical::core::field::BooleanOption::multi_valued)
    /// set to `true`. A term query matches if any element equals the
    /// queried value.
    pub fn add_bool_array(self, name: impl Into<String>, values: Vec<bool>) -> Self {
        self.add_field(name.into(), DataValue::BoolArray(values))
    }

    /// Add a binary data field with no MIME type.
    ///
    /// The MIME type is set to `None`. If a MIME type is needed (e.g. for
    /// multimodal embedding), use [`add_field`](Self::add_field) directly with
    /// `DataValue::Bytes(data, Some(mime))`.
    pub fn add_bytes(self, name: impl Into<String>, data: Vec<u8>) -> Self {
        self.add_field(name.into(), DataValue::Bytes(data, None))
    }

    pub fn build(self) -> Document {
        Document {
            fields: self.fields,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive(value: &DataValue) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(value)
            .expect("rkyv serialization")
            .to_vec()
    }

    /// `DataValue` is rkyv-archived on disk (DocValues `.dv`, WAL v3), so
    /// its archived layout is a persistence contract. The archived size is
    /// pinned so a variant with a wider payload trips a test instead of
    /// silently changing the layout of every persisted value.
    #[test]
    fn archived_data_value_size_is_pinned() {
        assert_eq!(std::mem::size_of::<ArchivedDataValue>(), 32);
    }

    /// Golden bytes for the variants adjacent to the #1174 additions. rkyv
    /// numbers archived discriminants in declaration order, so inserting a
    /// variant anywhere but at the end of the enum renumbers the ones after
    /// it and makes existing segments decode as the wrong variant — a
    /// mistake `size_of` cannot see. If this fails after a deliberate
    /// format change, update the bytes together with a note on Issue #1040.
    #[test]
    fn archived_discriminant_order_is_pinned() {
        let snapshots: Vec<(&str, Vec<u8>)> = vec![
            ("Geo", archive(&DataValue::Geo(GeoPoint::new(1.0, 2.0)))),
            ("Int64Array", archive(&DataValue::Int64Array(vec![7]))),
            ("Float64Array", archive(&DataValue::Float64Array(vec![1.5]))),
            (
                "GeoArray",
                archive(&DataValue::GeoArray(vec![GeoPoint::new(1.0, 2.0)])),
            ),
            (
                "GeoEcefArray",
                archive(&DataValue::GeoEcefArray(vec![GeoEcefPoint::new(
                    1.0, 2.0, 3.0,
                )])),
            ),
            (
                "DateTimeArray",
                archive(&DataValue::DateTimeArray(vec![
                    DateTime::from_timestamp_micros(7).unwrap(),
                ])),
            ),
            ("BoolArray", archive(&DataValue::BoolArray(vec![true]))),
        ];
        // Little-endian rkyv 0.8 layout: the root enum sits at the end of
        // the buffer, its first byte being the archived discriminant (Geo =
        // 8, Int64Array = 10, Float64Array = 11, GeoArray = 12, GeoEcefArray
        // = 13, DateTimeArray = 14, BoolArray = 15), followed by the payload
        // — `ArchivedVec` is a relative pointer to the element data written
        // before the root, plus a length. `DateTimeArray` archives its
        // elements as micro-second `i64`s, so its bytes are `Int64Array`'s
        // with the discriminant changed; a `bool` element is a single byte,
        // so `BoolArray`'s element data is followed by seven padding bytes
        // before the 8-aligned root.
        let expected: Vec<(&str, Vec<u8>)> = vec![
            (
                "Geo",
                vec![
                    8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64, 0,
                    0, 0, 0, 0, 0, 0, 0,
                ],
            ),
            (
                "Int64Array",
                vec![
                    7, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 244, 255, 255, 255, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
            (
                "Float64Array",
                vec![
                    0, 0, 0, 0, 0, 0, 248, 63, 11, 0, 0, 0, 244, 255, 255, 255, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
            (
                "GeoArray",
                vec![
                    0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64, 12, 0, 0, 0, 236, 255, 255,
                    255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
            (
                "GeoEcefArray",
                vec![
                    0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 8, 64,
                    13, 0, 0, 0, 228, 255, 255, 255, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
            (
                "DateTimeArray",
                vec![
                    7, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 244, 255, 255, 255, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
            (
                "BoolArray",
                vec![
                    1, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 244, 255, 255, 255, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
            ),
        ];
        assert_eq!(snapshots, expected);
    }

    /// #1184: instants round-trip at micro-second precision (the persisted
    /// precision of a scalar `DateTime`), including pre-1970 and empty.
    #[test]
    fn datetime_arrays_round_trip_through_rkyv() {
        use chrono::TimeZone;
        for value in [
            DataValue::DateTimeArray(vec![
                Utc.timestamp_opt(1_700_000_000, 500_000).unwrap(),
                Utc.timestamp_opt(-86_400, 0).unwrap(),
            ]),
            DataValue::DateTimeArray(Vec::new()),
        ] {
            let bytes = archive(&value);
            let back = rkyv::from_bytes::<DataValue, rkyv::rancor::Error>(&bytes)
                .expect("rkyv deserialization");
            assert_eq!(back, value);
        }
    }

    #[test]
    fn datetime_array_accessors_and_builders() {
        use chrono::TimeZone;
        let instants = vec![
            Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
            Utc.timestamp_opt(1_700_003_600, 0).unwrap(),
        ];
        let doc = Document::builder()
            .add_datetime_array("times", instants.clone())
            .build();
        assert_eq!(
            doc.get_field("times")
                .and_then(DataValue::as_datetime_array),
            Some(instants.as_slice())
        );
        assert_eq!(
            doc.get_field("times").and_then(DataValue::as_datetime),
            None
        );
        assert_eq!(DataValue::DateTime(instants[0]).as_datetime_array(), None);
        assert_eq!(
            DataValue::from(instants.clone()),
            DataValue::DateTimeArray(instants)
        );
    }

    /// #1180: `bool` is an rkyv archive-self primitive, so a `Vec<bool>`
    /// round-trips element-wise with no wrapper, including the empty list.
    #[test]
    fn bool_arrays_round_trip_through_rkyv() {
        for value in [
            DataValue::BoolArray(vec![true, false, true]),
            DataValue::BoolArray(Vec::new()),
        ] {
            let bytes = archive(&value);
            let back = rkyv::from_bytes::<DataValue, rkyv::rancor::Error>(&bytes)
                .expect("rkyv deserialization");
            assert_eq!(back, value);
        }
    }

    #[test]
    fn bool_array_accessors_and_builders() {
        let flags = vec![true, false];
        let doc = Document::builder()
            .add_bool_array("flags", flags.clone())
            .build();
        assert_eq!(
            doc.get_field("flags").and_then(DataValue::as_bool_array),
            Some(flags.as_slice())
        );
        assert_eq!(doc.get_field("flags").and_then(DataValue::as_boolean), None);
        assert_eq!(DataValue::Bool(true).as_bool_array(), None);
        assert_eq!(DataValue::from(flags.clone()), DataValue::BoolArray(flags));
    }

    #[test]
    fn geo_arrays_round_trip_through_rkyv() {
        for value in [
            DataValue::GeoArray(vec![
                GeoPoint::new(35.6, 139.7),
                GeoPoint::new(-33.9, 151.2),
            ]),
            DataValue::GeoArray(Vec::new()),
            DataValue::GeoEcefArray(vec![
                GeoEcefPoint::new(1.0, 2.0, 3.0),
                GeoEcefPoint::new(-4.0, 5.0, -6.0),
            ]),
            DataValue::GeoEcefArray(Vec::new()),
        ] {
            let bytes = archive(&value);
            let back = rkyv::from_bytes::<DataValue, rkyv::rancor::Error>(&bytes)
                .expect("rkyv deserialization");
            assert_eq!(back, value);
        }
    }

    #[test]
    fn geo_array_accessors_and_builders() {
        let pts = vec![GeoPoint::new(35.6, 139.7)];
        let ecef = vec![GeoEcefPoint::new(1.0, 2.0, 3.0)];
        let doc = Document::builder()
            .add_geo_array("locations", pts.clone())
            .add_geo_ecef_array("positions", ecef.clone())
            .build();
        assert_eq!(
            doc.get_field("locations").and_then(DataValue::as_geo_array),
            Some(pts.as_slice())
        );
        assert_eq!(
            doc.get_field("positions")
                .and_then(DataValue::as_geo_ecef_array),
            Some(ecef.as_slice())
        );
        // Single-valued accessors do not see arrays and vice versa.
        assert_eq!(doc.get_field("locations").and_then(DataValue::as_geo), None);
        assert_eq!(DataValue::Geo(pts[0]).as_geo_array(), None);
        assert_eq!(DataValue::from(pts.clone()), DataValue::GeoArray(pts));
        assert_eq!(DataValue::from(ecef.clone()), DataValue::GeoEcefArray(ecef));
    }
}
