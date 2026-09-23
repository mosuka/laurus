//! Ruby wrappers for all Laurus query types.
//!
//! Each Ruby query class stores the data needed to construct the Rust query.
//! Vector query classes produce [`VectorSearchQuery`] instead.

use std::cell::RefCell;

use laurus::GeoEcefPoint;
use laurus::lexical::span::{SpanQueryBuilder, SpanQueryWrapper};
use laurus::lexical::{
    BooleanQuery, DateTimeRangeQuery, FuzzyQuery, Geo3dBoundingBoxQuery, Geo3dDistanceQuery,
    Geo3dNearestQuery, GeoBoundingBoxQuery, GeoDistanceQuery, NumericRangeQuery, PhraseQuery,
    TermQuery, WildcardQuery,
};
use laurus::vector::Vector;
use laurus::vector::store::request::QueryVector;
use laurus::{DataValue, LexicalSearchQuery, QueryPayload, VectorSearchQuery};
use magnus::prelude::*;
use magnus::scan_args::{get_kwargs, scan_args};
use magnus::{Error, RArray, RHash, RModule, Ruby, Value};

// ---------------------------------------------------------------------------
// Helper: extract a lexical query from any Ruby query object
// ---------------------------------------------------------------------------

/// Extract a Laurus lexical query from an arbitrary Ruby value.
///
/// Supports: `TermQuery`, `PhraseQuery`, `FuzzyQuery`, `WildcardQuery`,
/// `NumericRangeQuery`, `GeoDistanceQuery`, `GeoBoundingBoxQuery`, `Geo3dDistanceQuery`,
/// `Geo3dBoundingBoxQuery`, `Geo3dNearestQuery`, `BooleanQuery`, `SpanQuery`.
///
/// # Arguments
///
/// * `value` - Ruby value that should be one of the query types.
///
/// # Returns
///
/// A boxed `dyn Query` implementing the lexical query.
pub fn extract_lexical_query(value: Value) -> Result<Box<dyn laurus::lexical::Query>, Error> {
    let ruby = Ruby::get().expect("called from Ruby thread");
    if let Ok(q) = <&RbTermQuery>::try_convert(value) {
        return Ok(Box::new(TermQuery::new(&q.field, &q.term)));
    }
    if let Ok(q) = <&RbPhraseQuery>::try_convert(value) {
        return Ok(Box::new(PhraseQuery::new(&q.field, q.terms.clone())));
    }
    if let Ok(q) = <&RbFuzzyQuery>::try_convert(value) {
        return Ok(Box::new(
            FuzzyQuery::new(&q.field, &q.term).max_edits(q.max_edits),
        ));
    }
    if let Ok(q) = <&RbWildcardQuery>::try_convert(value) {
        return Ok(Box::new(WildcardQuery::new(&q.field, &q.pattern).map_err(
            |e| Error::new(ruby.exception_arg_error(), e.to_string()),
        )?));
    }
    if let Ok(q) = <&RbNumericRangeQuery>::try_convert(value) {
        return Ok(q.build());
    }
    if let Ok(q) = <&RbDateTimeRangeQuery>::try_convert(value) {
        return q
            .build()
            .map_err(|e| Error::new(ruby.exception_arg_error(), e.to_string()));
    }
    if let Ok(q) = <&RbGeoDistanceQuery>::try_convert(value) {
        return q
            .build()
            .map_err(|e| Error::new(ruby.exception_arg_error(), e.to_string()));
    }
    if let Ok(q) = <&RbGeoBoundingBoxQuery>::try_convert(value) {
        return q
            .build()
            .map_err(|e| Error::new(ruby.exception_arg_error(), e.to_string()));
    }
    if let Ok(q) = <&RbGeo3dDistanceQuery>::try_convert(value) {
        return Ok(q.build());
    }
    if let Ok(q) = <&RbGeo3dBoundingBoxQuery>::try_convert(value) {
        return q
            .build()
            .map_err(|e| Error::new(ruby.exception_arg_error(), e.to_string()));
    }
    if let Ok(q) = <&RbGeo3dNearestQuery>::try_convert(value) {
        return Ok(q.build());
    }
    if let Ok(q) = <&RbBooleanQuery>::try_convert(value) {
        return q.build_query();
    }
    if let Ok(q) = <&RbSpanQuery>::try_convert(value) {
        return Ok(Box::new(SpanQueryWrapper::new(q.kind.build(&q.field))));
    }
    Err(Error::new(
        ruby.exception_arg_error(),
        format!(
            "Expected a lexical query type (TermQuery, BooleanQuery, …), got {}",
            value.class()
        ),
    ))
}

/// Wrap an arbitrary Ruby query value as a `LexicalSearchQuery::Obj`.
///
/// # Arguments
///
/// * `value` - Ruby query object.
///
/// # Returns
///
/// A `LexicalSearchQuery` wrapping the extracted query.
pub fn rb_to_lexical_search_query(value: Value) -> Result<LexicalSearchQuery, Error> {
    Ok(LexicalSearchQuery::Obj(extract_lexical_query(value)?))
}

/// Check whether the Ruby value is a vector query type.
///
/// # Arguments
///
/// * `value` - Ruby value to check.
///
/// # Returns
///
/// `true` if the value is a `VectorQuery` or `VectorTextQuery`.
pub fn is_vector_query(value: Value) -> bool {
    <&RbVectorQuery>::try_convert(value).is_ok() || <&RbVectorTextQuery>::try_convert(value).is_ok()
}

/// Convert a vector query Ruby value into a [`VectorSearchQuery`].
///
/// # Arguments
///
/// * `value` - Ruby value that should be `VectorQuery` or `VectorTextQuery`.
///
/// # Returns
///
/// The corresponding `VectorSearchQuery`.
pub fn rb_to_vector_search_query(value: Value) -> Result<VectorSearchQuery, Error> {
    let ruby = Ruby::get().expect("called from Ruby thread");
    if let Ok(q) = <&RbVectorQuery>::try_convert(value) {
        return Ok(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(q.vector.clone()),
            weight: 1.0,
            fields: Some(vec![q.field.clone()]),
        }]));
    }
    if let Ok(q) = <&RbVectorTextQuery>::try_convert(value) {
        return Ok(VectorSearchQuery::Payloads(vec![QueryPayload::new(
            &q.field,
            DataValue::Text(q.text.clone()),
        )]));
    }
    Err(Error::new(
        ruby.exception_arg_error(),
        "Expected VectorQuery or VectorTextQuery",
    ))
}

// ---------------------------------------------------------------------------
// Internal span-query recipe enum (Clone so it can be nested)
// ---------------------------------------------------------------------------

/// Internal representation of span query structure for deferred building.
#[derive(Clone)]
pub enum SpanKind {
    /// Single term span.
    Term(String),
    /// Near proximity span with slop and ordering.
    Near(Vec<SpanKind>, u32, bool),
    /// Containing span: big span that contains little span.
    Containing(Box<SpanKind>, Box<SpanKind>),
    /// Within span: include span within exclude span at a maximum distance.
    Within(Box<SpanKind>, Box<SpanKind>, u32),
}

impl SpanKind {
    /// Build a concrete `SpanQuery` from this recipe.
    ///
    /// # Arguments
    ///
    /// * `field` - Field name to search within.
    pub fn build(&self, field: &str) -> Box<dyn laurus::lexical::span::SpanQuery> {
        let sb = SpanQueryBuilder::new(field);
        match self {
            SpanKind::Term(t) => Box::new(sb.term(t)),
            SpanKind::Near(clauses, slop, ordered) => {
                let built: Vec<Box<dyn laurus::lexical::span::SpanQuery>> =
                    clauses.iter().map(|c| c.build(field)).collect();
                Box::new(sb.near(built, *slop, *ordered))
            }
            SpanKind::Containing(big, little) => {
                Box::new(sb.containing(big.build(field), little.build(field)))
            }
            SpanKind::Within(include, exclude, dist) => {
                Box::new(sb.within(include.build(field), exclude.build(field), *dist))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TermQuery
// ---------------------------------------------------------------------------

/// Exact single-term lexical query (`Laurus::TermQuery`).
#[magnus::wrap(class = "Laurus::TermQuery")]
pub struct RbTermQuery {
    pub field: String,
    pub term: String,
}

impl RbTermQuery {
    /// Create a new term query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field name to search.
    /// * `term` - Exact term to match.
    fn new(field: String, term: String) -> Self {
        Self { field, term }
    }

    fn inspect(&self) -> String {
        format!("TermQuery(field='{}', term='{}')", self.field, self.term)
    }
}

// ---------------------------------------------------------------------------
// PhraseQuery
// ---------------------------------------------------------------------------

/// Exact phrase (word-sequence) lexical query (`Laurus::PhraseQuery`).
#[magnus::wrap(class = "Laurus::PhraseQuery")]
pub struct RbPhraseQuery {
    pub field: String,
    pub terms: Vec<String>,
}

impl RbPhraseQuery {
    /// Create a new phrase query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field name to search.
    /// * `terms` - Array of terms forming the phrase.
    fn new(field: String, terms: RArray) -> Result<Self, Error> {
        let terms: Vec<String> = terms.to_vec()?;
        Ok(Self { field, terms })
    }

    fn inspect(&self) -> String {
        format!(
            "PhraseQuery(field='{}', terms={:?})",
            self.field, self.terms
        )
    }
}

// ---------------------------------------------------------------------------
// FuzzyQuery
// ---------------------------------------------------------------------------

/// Approximate (typo-tolerant) lexical query (`Laurus::FuzzyQuery`).
#[magnus::wrap(class = "Laurus::FuzzyQuery")]
pub struct RbFuzzyQuery {
    pub field: String,
    pub term: String,
    pub max_edits: u32,
}

impl RbFuzzyQuery {
    /// Create a new fuzzy query.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `field` (String): Field name.
    ///   - `term` (String): Term to match approximately.
    ///   - `max_edits:` (u32, default 2): Maximum edit distance.
    fn new(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(String, String), (), (), (), RHash, ()>(args)?;
        let (field, term) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<u32>,), ()>(args.keywords, &[], &["max_edits"])?;
        let (max_edits,) = kwargs.optional;
        Ok(Self {
            field,
            term,
            max_edits: max_edits.unwrap_or(2),
        })
    }

    fn inspect(&self) -> String {
        format!(
            "FuzzyQuery(field='{}', term='{}', max_edits={})",
            self.field, self.term, self.max_edits
        )
    }
}

// ---------------------------------------------------------------------------
// WildcardQuery
// ---------------------------------------------------------------------------

/// Wildcard pattern lexical query (`Laurus::WildcardQuery`).
///
/// `*` matches any sequence, `?` matches any single character.
#[magnus::wrap(class = "Laurus::WildcardQuery")]
pub struct RbWildcardQuery {
    pub field: String,
    pub pattern: String,
}

impl RbWildcardQuery {
    /// Create a new wildcard query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field name to search.
    /// * `pattern` - Wildcard pattern.
    fn new(field: String, pattern: String) -> Self {
        Self { field, pattern }
    }

    fn inspect(&self) -> String {
        format!(
            "WildcardQuery(field='{}', pattern='{}')",
            self.field, self.pattern
        )
    }
}

// ---------------------------------------------------------------------------
// NumericRangeQuery
// ---------------------------------------------------------------------------

/// Internal representation of numeric range kind.
#[derive(Clone)]
pub enum NumericKind {
    /// Integer range with optional min and max.
    Integer(Option<i64>, Option<i64>),
    /// Float range with optional min and max.
    Float(Option<f64>, Option<f64>),
}

/// Numeric range filter query (`Laurus::NumericRangeQuery`).
///
/// The type (integer or float) is inferred from the Ruby type of `min`/`max`.
#[magnus::wrap(class = "Laurus::NumericRangeQuery")]
pub struct RbNumericRangeQuery {
    pub field: String,
    pub kind: NumericKind,
}

impl RbNumericRangeQuery {
    /// Create a new numeric range query.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `field` (String): Field name.
    ///   - `min:` (Integer or Float, optional): Lower bound.
    ///   - `max:` (Integer or Float, optional): Upper bound.
    fn new(args: &[Value]) -> Result<Self, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let args = scan_args::<(String,), (), (), (), RHash, ()>(args)?;
        let (field,) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<Value>, Option<Value>), ()>(
            args.keywords,
            &[],
            &["min", "max"],
        )?;
        let (min_val, max_val) = kwargs.optional;

        let is_float = min_val
            .as_ref()
            .map(|v| v.is_kind_of(ruby.class_float()))
            .unwrap_or(false)
            || max_val
                .as_ref()
                .map(|v| v.is_kind_of(ruby.class_float()))
                .unwrap_or(false);

        let kind = if is_float {
            NumericKind::Float(
                min_val
                    .as_ref()
                    .map(|v| <f64>::try_convert(*v))
                    .transpose()?,
                max_val
                    .as_ref()
                    .map(|v| <f64>::try_convert(*v))
                    .transpose()?,
            )
        } else {
            NumericKind::Integer(
                min_val
                    .as_ref()
                    .map(|v| <i64>::try_convert(*v))
                    .transpose()?,
                max_val
                    .as_ref()
                    .map(|v| <i64>::try_convert(*v))
                    .transpose()?,
            )
        };
        Ok(Self { field, kind })
    }

    fn inspect(&self) -> String {
        match &self.kind {
            NumericKind::Integer(min, max) => {
                format!(
                    "NumericRangeQuery(field='{}', min={:?}, max={:?}, type='integer')",
                    self.field, min, max
                )
            }
            NumericKind::Float(min, max) => {
                format!(
                    "NumericRangeQuery(field='{}', min={:?}, max={:?}, type='float')",
                    self.field, min, max
                )
            }
        }
    }
}

impl RbNumericRangeQuery {
    /// Build the underlying Rust `NumericRangeQuery`.
    pub fn build(&self) -> Box<dyn laurus::lexical::Query> {
        match &self.kind {
            NumericKind::Float(min, max) => {
                Box::new(NumericRangeQuery::f64_range(&self.field, *min, *max))
            }
            NumericKind::Integer(min, max) => {
                Box::new(NumericRangeQuery::i64_range(&self.field, *min, *max))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DateTimeRangeQuery
// ---------------------------------------------------------------------------

/// DateTime range filter query (`Laurus::DateTimeRangeQuery`, Issue #1179).
///
/// Both bounds are inclusive; omit a keyword (or pass `nil`) to leave that
/// side open. A bound is a `String` in any form the query DSL accepts —
/// RFC 3339 (`"2024-01-01T09:00:00+09:00"`, normalized to UTC), a naive
/// `"YYYY-MM-DDTHH:MM:SS[.fff]"` (UTC), or a date `"YYYY-MM-DD"` (midnight
/// UTC) — or any object responding to `iso8601` (`Time`, `DateTime`), the
/// same conversion document ingestion uses.
#[magnus::wrap(class = "Laurus::DateTimeRangeQuery")]
pub struct RbDateTimeRangeQuery {
    pub field: String,
    pub min: Option<String>,
    pub max: Option<String>,
}

impl RbDateTimeRangeQuery {
    /// Create a new datetime range query.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `field` (String): Field name.
    ///   - `min:` (String or Time, optional): Lower bound (inclusive).
    ///   - `max:` (String or Time, optional): Upper bound (inclusive).
    fn new(args: &[Value]) -> Result<Self, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let args = scan_args::<(String,), (), (), (), RHash, ()>(args)?;
        let (field,) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<Value>, Option<Value>), ()>(
            args.keywords,
            &[],
            &["min", "max"],
        )?;
        let (min_val, max_val) = kwargs.optional;
        let query = Self {
            field,
            min: min_val
                .map(|v| datetime_literal(&ruby, v))
                .transpose()?
                .flatten(),
            max: max_val
                .map(|v| datetime_literal(&ruby, v))
                .transpose()?
                .flatten(),
        };
        // Validate eagerly so a bad bound raises at construction, not at search.
        query
            .build()
            .map_err(|e| Error::new(ruby.exception_arg_error(), e.to_string()))?;
        Ok(query)
    }

    fn inspect(&self) -> String {
        format!(
            "DateTimeRangeQuery(field='{}', min={:?}, max={:?})",
            self.field, self.min, self.max
        )
    }

    /// Build the underlying Rust `DateTimeRangeQuery`.
    pub fn build(&self) -> laurus::Result<Box<dyn laurus::lexical::Query>> {
        Ok(Box::new(DateTimeRangeQuery::from_literals(
            &self.field,
            self.min.as_deref(),
            self.max.as_deref(),
            true,
            true,
        )?))
    }
}

/// A datetime bound as text: a `String` as given, or the `iso8601` of any
/// object responding to it (`Time`, `DateTime`). `nil` means "open".
fn datetime_literal(ruby: &Ruby, value: Value) -> Result<Option<String>, Error> {
    if value.is_nil() {
        return Ok(None);
    }
    if value.is_kind_of(ruby.class_string()) {
        return Ok(Some(String::try_convert(value)?));
    }
    if let Ok(s) = value.funcall::<_, _, String>("iso8601", ()) {
        return Ok(Some(s));
    }
    Err(Error::new(
        ruby.exception_type_error(),
        format!(
            "DateTimeRangeQuery: min/max must be a String or respond to iso8601, got {}",
            value.class()
        ),
    ))
}

// ---------------------------------------------------------------------------
// GeoDistanceQuery
// ---------------------------------------------------------------------------

/// Geographic distance (radius) search query (`Laurus::GeoDistanceQuery`).
#[magnus::wrap(class = "Laurus::GeoDistanceQuery")]
pub struct RbGeoDistanceQuery {
    pub field: String,
    pub lat: f64,
    pub lon: f64,
    pub distance_m: f64,
}

impl RbGeoDistanceQuery {
    /// Create a radius-based geo distance query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo field name.
    /// * `lat` - Center latitude.
    /// * `lon` - Center longitude.
    /// * `distance_m` - Maximum distance from the centre in meters.
    fn within_radius(field: String, lat: f64, lon: f64, distance_m: f64) -> Self {
        Self {
            field,
            lat,
            lon,
            distance_m,
        }
    }

    fn inspect(&self) -> String {
        format!(
            "GeoDistanceQuery.within_radius(field='{}', lat={}, lon={}, distance_m={})",
            self.field, self.lat, self.lon, self.distance_m
        )
    }
}

impl RbGeoDistanceQuery {
    /// Build the underlying Rust `GeoDistanceQuery`.
    pub fn build(&self) -> laurus::Result<Box<dyn laurus::lexical::Query>> {
        Ok(Box::new(GeoDistanceQuery::within_radius(
            &self.field,
            self.lat,
            self.lon,
            self.distance_m,
        )?))
    }
}

// ---------------------------------------------------------------------------
// GeoBoundingBoxQuery
// ---------------------------------------------------------------------------

/// Geographic bounding-box search query (`Laurus::GeoBoundingBoxQuery`).
#[magnus::wrap(class = "Laurus::GeoBoundingBoxQuery")]
pub struct RbGeoBoundingBoxQuery {
    pub field: String,
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
}

impl RbGeoBoundingBoxQuery {
    /// Create a bounding-box geo query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo field name.
    /// * `min_lat` - Southern boundary.
    /// * `min_lon` - Western boundary.
    /// * `max_lat` - Northern boundary.
    /// * `max_lon` - Eastern boundary.
    fn within_bounding_box(
        field: String,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Self {
        Self {
            field,
            min_lat,
            min_lon,
            max_lat,
            max_lon,
        }
    }

    fn inspect(&self) -> String {
        format!(
            "GeoBoundingBoxQuery.within_bounding_box(field='{}', min_lat={}, min_lon={}, max_lat={}, max_lon={})",
            self.field, self.min_lat, self.min_lon, self.max_lat, self.max_lon
        )
    }
}

impl RbGeoBoundingBoxQuery {
    /// Build the underlying Rust `GeoBoundingBoxQuery`.
    pub fn build(&self) -> laurus::Result<Box<dyn laurus::lexical::Query>> {
        Ok(Box::new(GeoBoundingBoxQuery::within_bounding_box(
            &self.field,
            self.min_lat,
            self.min_lon,
            self.max_lat,
            self.max_lon,
        )?))
    }
}

// ---------------------------------------------------------------------------
// Geo3dDistanceQuery
// ---------------------------------------------------------------------------

/// 3D ECEF sphere query (`Laurus::Geo3dDistanceQuery`).
///
/// Matches every document whose stored `(x, y, z)` point lies within
/// `distance_m` meters of the given centre.
#[magnus::wrap(class = "Laurus::Geo3dDistanceQuery")]
pub struct RbGeo3dDistanceQuery {
    pub field: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub distance_m: f64,
}

impl RbGeo3dDistanceQuery {
    /// Create a 3D distance (sphere) query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo3d field name.
    /// * `x`, `y`, `z` - Centre coordinates in ECEF meters.
    /// * `distance_m` - Maximum distance from the centre in meters
    ///   (i.e. the search sphere's radius).
    fn within_sphere(field: String, x: f64, y: f64, z: f64, distance_m: f64) -> Self {
        Self {
            field,
            x,
            y,
            z,
            distance_m,
        }
    }

    fn inspect(&self) -> String {
        format!(
            "Geo3dDistanceQuery.within_sphere(field='{}', x={}, y={}, z={}, distance_m={})",
            self.field, self.x, self.y, self.z, self.distance_m
        )
    }
}

impl RbGeo3dDistanceQuery {
    /// Build the underlying Rust `Geo3dDistanceQuery`.
    pub fn build(&self) -> Box<dyn laurus::lexical::Query> {
        Box::new(Geo3dDistanceQuery::new(
            &self.field,
            GeoEcefPoint::new(self.x, self.y, self.z),
            self.distance_m,
        ))
    }
}

// ---------------------------------------------------------------------------
// Geo3dBoundingBoxQuery
// ---------------------------------------------------------------------------

/// 3D ECEF axis-aligned bounding box query (`Laurus::Geo3dBoundingBoxQuery`).
#[magnus::wrap(class = "Laurus::Geo3dBoundingBoxQuery")]
pub struct RbGeo3dBoundingBoxQuery {
    pub field: String,
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl RbGeo3dBoundingBoxQuery {
    /// Create a 3D bounding-box query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo3d field name.
    /// * `min_x`, `min_y`, `min_z` - Lower corner of the box.
    /// * `max_x`, `max_y`, `max_z` - Upper corner of the box.
    #[allow(clippy::too_many_arguments)]
    fn within_box(
        field: String,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) -> Self {
        Self {
            field,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        }
    }

    fn inspect(&self) -> String {
        format!(
            "Geo3dBoundingBoxQuery.within_box(field='{}', min=({}, {}, {}), max=({}, {}, {}))",
            self.field, self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z
        )
    }
}

impl RbGeo3dBoundingBoxQuery {
    /// Build the underlying Rust `Geo3dBoundingBoxQuery`.
    pub fn build(&self) -> laurus::Result<Box<dyn laurus::lexical::Query>> {
        Ok(Box::new(Geo3dBoundingBoxQuery::new(
            &self.field,
            GeoEcefPoint::new(self.min_x, self.min_y, self.min_z),
            GeoEcefPoint::new(self.max_x, self.max_y, self.max_z),
        )?))
    }
}

// ---------------------------------------------------------------------------
// Geo3dNearestQuery
// ---------------------------------------------------------------------------

/// 3D ECEF k-nearest-neighbours query (`Laurus::Geo3dNearestQuery`).
#[magnus::wrap(class = "Laurus::Geo3dNearestQuery")]
pub struct RbGeo3dNearestQuery {
    pub field: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub k: u32,
    pub initial_radius_m: Option<f64>,
    pub max_radius_m: Option<f64>,
}

impl RbGeo3dNearestQuery {
    /// Create a 3D k-NN query.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `field` (String): Geo3d field name.
    ///   - `x`, `y`, `z` (Float): Centre coordinates in ECEF meters.
    ///   - `k` (Integer): Number of nearest neighbours to return.
    ///   - `initial_radius_m:` (Float, optional): Starting radius for the
    ///     expanding-radius search (default 1000 m).
    ///   - `max_radius_m:` (Float, optional): Hard cap on the search radius
    ///     (default 1e10 m).
    fn k_nearest(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(String, f64, f64, f64, u32), (), (), (), RHash, ()>(args)?;
        let (field, x, y, z, k) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<f64>, Option<f64>), ()>(
            args.keywords,
            &[],
            &["initial_radius_m", "max_radius_m"],
        )?;
        let (initial_radius_m, max_radius_m) = kwargs.optional;
        Ok(Self {
            field,
            x,
            y,
            z,
            k,
            initial_radius_m,
            max_radius_m,
        })
    }

    fn inspect(&self) -> String {
        format!(
            "Geo3dNearestQuery.k_nearest(field='{}', x={}, y={}, z={}, k={})",
            self.field, self.x, self.y, self.z, self.k
        )
    }
}

impl RbGeo3dNearestQuery {
    /// Build the underlying Rust `Geo3dNearestQuery`.
    pub fn build(&self) -> Box<dyn laurus::lexical::Query> {
        let mut q = Geo3dNearestQuery::new(
            &self.field,
            GeoEcefPoint::new(self.x, self.y, self.z),
            self.k as usize,
        );
        if let Some(r) = self.initial_radius_m {
            q = q.with_initial_radius(r);
        }
        if let Some(r) = self.max_radius_m {
            q = q.with_max_radius(r);
        }
        Box::new(q)
    }
}

// ---------------------------------------------------------------------------
// BooleanQuery - stores extracted Rust queries, not Ruby Values
// ---------------------------------------------------------------------------

/// Boolean combination query (`Laurus::BooleanQuery`).
///
/// Supports AND (must), OR (should), and NOT (must_not) clauses.
/// Stores extracted Rust query objects to avoid Send/Sync issues with Ruby Values.
#[magnus::wrap(class = "Laurus::BooleanQuery")]
pub struct RbBooleanQuery {
    pub musts: RefCell<Vec<Box<dyn laurus::lexical::Query>>>,
    pub shoulds: RefCell<Vec<Box<dyn laurus::lexical::Query>>>,
    pub must_nots: RefCell<Vec<Box<dyn laurus::lexical::Query>>>,
}

impl RbBooleanQuery {
    /// Create a new empty boolean query.
    fn new() -> Self {
        Self {
            musts: RefCell::new(Vec::new()),
            shoulds: RefCell::new(Vec::new()),
            must_nots: RefCell::new(Vec::new()),
        }
    }

    /// Add a MUST (required) clause.
    ///
    /// # Arguments
    ///
    /// * `query` - A lexical query object.
    fn must(&self, query: Value) -> Result<(), Error> {
        let q = extract_lexical_query(query)?;
        self.musts.borrow_mut().push(q);
        Ok(())
    }

    /// Add a SHOULD (optional, boosts score) clause.
    ///
    /// # Arguments
    ///
    /// * `query` - A lexical query object.
    fn should(&self, query: Value) -> Result<(), Error> {
        let q = extract_lexical_query(query)?;
        self.shoulds.borrow_mut().push(q);
        Ok(())
    }

    /// Add a MUST_NOT (exclusion) clause.
    ///
    /// # Arguments
    ///
    /// * `query` - A lexical query object.
    fn must_not(&self, query: Value) -> Result<(), Error> {
        let q = extract_lexical_query(query)?;
        self.must_nots.borrow_mut().push(q);
        Ok(())
    }

    fn inspect(&self) -> String {
        format!(
            "BooleanQuery(musts={}, shoulds={}, must_nots={})",
            self.musts.borrow().len(),
            self.shoulds.borrow().len(),
            self.must_nots.borrow().len()
        )
    }
}

impl RbBooleanQuery {
    /// Build the underlying Rust [`BooleanQuery`].
    pub fn build_query(&self) -> Result<Box<dyn laurus::lexical::Query>, Error> {
        let mut bq = BooleanQuery::new();
        for q in self.musts.borrow_mut().drain(..) {
            bq.add_must(q);
        }
        for q in self.shoulds.borrow_mut().drain(..) {
            bq.add_should(q);
        }
        for q in self.must_nots.borrow_mut().drain(..) {
            bq.add_must_not(q);
        }
        Ok(Box::new(bq))
    }
}

// ---------------------------------------------------------------------------
// SpanQuery
// ---------------------------------------------------------------------------

/// Positional / proximity span query (`Laurus::SpanQuery`).
///
/// Use the class methods to construct span queries, which can be nested to
/// build complex positional expressions.
#[magnus::wrap(class = "Laurus::SpanQuery")]
pub struct RbSpanQuery {
    pub field: String,
    pub kind: SpanKind,
}

impl RbSpanQuery {
    /// Single-term span query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to search.
    /// * `term` - Term to match.
    fn term(field: String, term: String) -> Self {
        Self {
            field,
            kind: SpanKind::Term(term),
        }
    }

    /// SpanNear: terms appearing within `slop` positions of each other.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `field` (String): Field to search.
    ///   - `terms` (Array): List of term strings.
    ///   - `slop:` (u32, default 0): Maximum token distance between terms.
    ///   - `ordered:` (bool, default true): Whether terms must appear in order.
    fn near(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(String, RArray), (), (), (), RHash, ()>(args)?;
        let (field, terms_arr) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<u32>, Option<bool>), ()>(
            args.keywords,
            &[],
            &["slop", "ordered"],
        )?;
        let (slop, ordered) = kwargs.optional;
        let terms: Vec<String> = terms_arr.to_vec()?;
        let kinds = terms.into_iter().map(SpanKind::Term).collect();
        Ok(Self {
            field,
            kind: SpanKind::Near(kinds, slop.unwrap_or(0), ordered.unwrap_or(true)),
        })
    }

    /// SpanNear with nested SpanQuery clauses instead of plain terms.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `field` (String): Field to search.
    ///   - `clauses` (Array): List of SpanQuery objects.
    ///   - `slop:` (u32, default 0): Maximum token distance.
    ///   - `ordered:` (bool, default true): Whether clauses must appear in order.
    fn near_spans(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(String, RArray), (), (), (), RHash, ()>(args)?;
        let (field, clauses_arr) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<u32>, Option<bool>), ()>(
            args.keywords,
            &[],
            &["slop", "ordered"],
        )?;
        let (slop, ordered) = kwargs.optional;
        // Manually iterate to extract SpanQuery references
        let len = clauses_arr.len();
        let mut kinds = Vec::with_capacity(len);
        for i in 0..len {
            let val: Value = clauses_arr.entry(i as isize)?;
            let span: &RbSpanQuery = <&RbSpanQuery>::try_convert(val)?;
            kinds.push(span.kind.clone());
        }
        Ok(Self {
            field,
            kind: SpanKind::Near(kinds, slop.unwrap_or(0), ordered.unwrap_or(true)),
        })
    }

    /// SpanContaining: a span that contains another span.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to search.
    /// * `big` - Outer span query.
    /// * `little` - Inner span query.
    fn containing(field: String, big: &RbSpanQuery, little: &RbSpanQuery) -> Self {
        Self {
            field,
            kind: SpanKind::Containing(Box::new(big.kind.clone()), Box::new(little.kind.clone())),
        }
    }

    /// SpanWithin: a span included within another span at a maximum distance.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to search.
    /// * `include` - Span to include.
    /// * `exclude` - Span to exclude.
    /// * `distance` - Maximum distance.
    fn within(field: String, include: &RbSpanQuery, exclude: &RbSpanQuery, distance: u32) -> Self {
        Self {
            field,
            kind: SpanKind::Within(
                Box::new(include.kind.clone()),
                Box::new(exclude.kind.clone()),
                distance,
            ),
        }
    }

    fn inspect(&self) -> String {
        format!("SpanQuery(field='{}')", self.field)
    }
}

// ---------------------------------------------------------------------------
// VectorQuery (pre-computed vector)
// ---------------------------------------------------------------------------

/// Vector search query using a pre-computed embedding vector (`Laurus::VectorQuery`).
#[magnus::wrap(class = "Laurus::VectorQuery")]
pub struct RbVectorQuery {
    pub field: String,
    pub vector: Vec<f32>,
}

impl RbVectorQuery {
    /// Create a new vector query.
    ///
    /// # Arguments
    ///
    /// * `field` - Vector field name.
    /// * `vector` - Pre-computed embedding vector as Array of floats.
    fn new(field: String, vector: RArray) -> Result<Self, Error> {
        let vector: Vec<f32> = vector.to_vec()?;
        Ok(Self { field, vector })
    }

    fn inspect(&self) -> String {
        format!(
            "VectorQuery(field='{}', dims={})",
            self.field,
            self.vector.len()
        )
    }
}

// ---------------------------------------------------------------------------
// VectorTextQuery (text → Rust Embedder → vector)
// ---------------------------------------------------------------------------

/// Vector search query where text is embedded by the Rust-side Embedder
/// (`Laurus::VectorTextQuery`).
#[magnus::wrap(class = "Laurus::VectorTextQuery")]
pub struct RbVectorTextQuery {
    pub field: String,
    pub text: String,
}

impl RbVectorTextQuery {
    /// Create a new vector text query.
    ///
    /// # Arguments
    ///
    /// * `field` - Vector field name.
    /// * `text` - Text to be embedded by the Rust-side embedder.
    fn new(field: String, text: String) -> Self {
        Self { field, text }
    }

    fn inspect(&self) -> String {
        format!(
            "VectorTextQuery(field='{}', text='{}')",
            self.field, self.text
        )
    }
}

// ---------------------------------------------------------------------------
// Class registration
// ---------------------------------------------------------------------------

/// Register all query classes under the `Laurus` module.
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `module` - The `Laurus` module.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    // TermQuery
    let term_q = module.define_class("TermQuery", ruby.class_object())?;
    term_q.define_singleton_method("new", magnus::function!(RbTermQuery::new, 2))?;
    term_q.define_method("inspect", magnus::method!(RbTermQuery::inspect, 0))?;
    term_q.define_method("to_s", magnus::method!(RbTermQuery::inspect, 0))?;

    // PhraseQuery
    let phrase_q = module.define_class("PhraseQuery", ruby.class_object())?;
    phrase_q.define_singleton_method("new", magnus::function!(RbPhraseQuery::new, 2))?;
    phrase_q.define_method("inspect", magnus::method!(RbPhraseQuery::inspect, 0))?;
    phrase_q.define_method("to_s", magnus::method!(RbPhraseQuery::inspect, 0))?;

    // FuzzyQuery
    let fuzzy_q = module.define_class("FuzzyQuery", ruby.class_object())?;
    fuzzy_q.define_singleton_method("new", magnus::function!(RbFuzzyQuery::new, -1))?;
    fuzzy_q.define_method("inspect", magnus::method!(RbFuzzyQuery::inspect, 0))?;
    fuzzy_q.define_method("to_s", magnus::method!(RbFuzzyQuery::inspect, 0))?;

    // WildcardQuery
    let wc_q = module.define_class("WildcardQuery", ruby.class_object())?;
    wc_q.define_singleton_method("new", magnus::function!(RbWildcardQuery::new, 2))?;
    wc_q.define_method("inspect", magnus::method!(RbWildcardQuery::inspect, 0))?;
    wc_q.define_method("to_s", magnus::method!(RbWildcardQuery::inspect, 0))?;

    // NumericRangeQuery
    let nr_q = module.define_class("NumericRangeQuery", ruby.class_object())?;
    nr_q.define_singleton_method("new", magnus::function!(RbNumericRangeQuery::new, -1))?;
    nr_q.define_method("inspect", magnus::method!(RbNumericRangeQuery::inspect, 0))?;
    nr_q.define_method("to_s", magnus::method!(RbNumericRangeQuery::inspect, 0))?;

    // DateTimeRangeQuery
    let dt_q = module.define_class("DateTimeRangeQuery", ruby.class_object())?;
    dt_q.define_singleton_method("new", magnus::function!(RbDateTimeRangeQuery::new, -1))?;
    dt_q.define_method("inspect", magnus::method!(RbDateTimeRangeQuery::inspect, 0))?;
    dt_q.define_method("to_s", magnus::method!(RbDateTimeRangeQuery::inspect, 0))?;

    // GeoDistanceQuery
    let geo_dist_q = module.define_class("GeoDistanceQuery", ruby.class_object())?;
    geo_dist_q.define_singleton_method(
        "within_radius",
        magnus::function!(RbGeoDistanceQuery::within_radius, 4),
    )?;
    geo_dist_q.define_method("inspect", magnus::method!(RbGeoDistanceQuery::inspect, 0))?;
    geo_dist_q.define_method("to_s", magnus::method!(RbGeoDistanceQuery::inspect, 0))?;

    // GeoBoundingBoxQuery
    let geo_bbox_q = module.define_class("GeoBoundingBoxQuery", ruby.class_object())?;
    geo_bbox_q.define_singleton_method(
        "within_bounding_box",
        magnus::function!(RbGeoBoundingBoxQuery::within_bounding_box, 5),
    )?;
    geo_bbox_q.define_method(
        "inspect",
        magnus::method!(RbGeoBoundingBoxQuery::inspect, 0),
    )?;
    geo_bbox_q.define_method("to_s", magnus::method!(RbGeoBoundingBoxQuery::inspect, 0))?;

    // Geo3dDistanceQuery
    let g3d_dist = module.define_class("Geo3dDistanceQuery", ruby.class_object())?;
    g3d_dist.define_singleton_method(
        "within_sphere",
        magnus::function!(RbGeo3dDistanceQuery::within_sphere, 5),
    )?;
    g3d_dist.define_method("inspect", magnus::method!(RbGeo3dDistanceQuery::inspect, 0))?;
    g3d_dist.define_method("to_s", magnus::method!(RbGeo3dDistanceQuery::inspect, 0))?;

    // Geo3dBoundingBoxQuery
    let g3d_bbox = module.define_class("Geo3dBoundingBoxQuery", ruby.class_object())?;
    g3d_bbox.define_singleton_method(
        "within_box",
        magnus::function!(RbGeo3dBoundingBoxQuery::within_box, 7),
    )?;
    g3d_bbox.define_method(
        "inspect",
        magnus::method!(RbGeo3dBoundingBoxQuery::inspect, 0),
    )?;
    g3d_bbox.define_method("to_s", magnus::method!(RbGeo3dBoundingBoxQuery::inspect, 0))?;

    // Geo3dNearestQuery
    let g3d_near = module.define_class("Geo3dNearestQuery", ruby.class_object())?;
    g3d_near.define_singleton_method(
        "k_nearest",
        magnus::function!(RbGeo3dNearestQuery::k_nearest, -1),
    )?;
    g3d_near.define_method("inspect", magnus::method!(RbGeo3dNearestQuery::inspect, 0))?;
    g3d_near.define_method("to_s", magnus::method!(RbGeo3dNearestQuery::inspect, 0))?;

    // BooleanQuery
    let bool_q = module.define_class("BooleanQuery", ruby.class_object())?;
    bool_q.define_singleton_method("new", magnus::function!(RbBooleanQuery::new, 0))?;
    bool_q.define_method("must", magnus::method!(RbBooleanQuery::must, 1))?;
    bool_q.define_method("should", magnus::method!(RbBooleanQuery::should, 1))?;
    bool_q.define_method("must_not", magnus::method!(RbBooleanQuery::must_not, 1))?;
    bool_q.define_method("inspect", magnus::method!(RbBooleanQuery::inspect, 0))?;
    bool_q.define_method("to_s", magnus::method!(RbBooleanQuery::inspect, 0))?;

    // SpanQuery
    let span_q = module.define_class("SpanQuery", ruby.class_object())?;
    span_q.define_singleton_method("term", magnus::function!(RbSpanQuery::term, 2))?;
    span_q.define_singleton_method("near", magnus::function!(RbSpanQuery::near, -1))?;
    span_q.define_singleton_method("near_spans", magnus::function!(RbSpanQuery::near_spans, -1))?;
    span_q.define_singleton_method("containing", magnus::function!(RbSpanQuery::containing, 3))?;
    span_q.define_singleton_method("within", magnus::function!(RbSpanQuery::within, 4))?;
    span_q.define_method("inspect", magnus::method!(RbSpanQuery::inspect, 0))?;
    span_q.define_method("to_s", magnus::method!(RbSpanQuery::inspect, 0))?;

    // VectorQuery
    let vec_q = module.define_class("VectorQuery", ruby.class_object())?;
    vec_q.define_singleton_method("new", magnus::function!(RbVectorQuery::new, 2))?;
    vec_q.define_method("inspect", magnus::method!(RbVectorQuery::inspect, 0))?;
    vec_q.define_method("to_s", magnus::method!(RbVectorQuery::inspect, 0))?;

    // VectorTextQuery
    let vt_q = module.define_class("VectorTextQuery", ruby.class_object())?;
    vt_q.define_singleton_method("new", magnus::function!(RbVectorTextQuery::new, 2))?;
    vt_q.define_method("inspect", magnus::method!(RbVectorTextQuery::inspect, 0))?;
    vt_q.define_method("to_s", magnus::method!(RbVectorTextQuery::inspect, 0))?;

    Ok(())
}
