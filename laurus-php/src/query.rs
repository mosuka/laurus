//! PHP wrappers for all Laurus query types.
//!
//! Each PHP query class stores the data needed to construct the Rust query.
//! Vector query classes produce [`VectorSearchQuery`] instead.

use std::cell::RefCell;

use ext_php_rs::convert::FromZval;
use ext_php_rs::prelude::*;
use ext_php_rs::types::{ZendClassObject, Zval};
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

// ---------------------------------------------------------------------------
// Helper: extract a lexical query from any PHP query object
// ---------------------------------------------------------------------------

/// Extract a Laurus lexical query from an arbitrary PHP Zval.
///
/// Supports: `TermQuery`, `PhraseQuery`, `FuzzyQuery`, `WildcardQuery`,
/// `NumericRangeQuery`, `GeoDistanceQuery`, `GeoBoundingBoxQuery`, `Geo3dDistanceQuery`,
/// `Geo3dBoundingBoxQuery`, `Geo3dNearestQuery`, `BooleanQuery`, `SpanQuery`.
///
/// # Arguments
///
/// * `zv` - PHP Zval that should be one of the query types.
///
/// # Returns
///
/// A boxed `dyn Query` implementing the lexical query.
pub fn extract_lexical_query(zv: &Zval) -> PhpResult<Box<dyn laurus::lexical::Query>> {
    if let Some(obj) = <&ZendClassObject<PhpTermQuery>>::from_zval(zv) {
        let q: &PhpTermQuery = obj;
        return Ok(Box::new(TermQuery::new(&q.field, &q.term)));
    }
    if let Some(obj) = <&ZendClassObject<PhpPhraseQuery>>::from_zval(zv) {
        let q: &PhpPhraseQuery = obj;
        return Ok(Box::new(PhraseQuery::new(&q.field, q.terms.clone())));
    }
    if let Some(obj) = <&ZendClassObject<PhpFuzzyQuery>>::from_zval(zv) {
        let q: &PhpFuzzyQuery = obj;
        return Ok(Box::new(
            FuzzyQuery::new(&q.field, &q.term).max_edits(q.max_edits),
        ));
    }
    if let Some(obj) = <&ZendClassObject<PhpWildcardQuery>>::from_zval(zv) {
        let q: &PhpWildcardQuery = obj;
        return Ok(Box::new(WildcardQuery::new(&q.field, &q.pattern).map_err(
            |e| ext_php_rs::exception::PhpException::default(e.to_string()),
        )?));
    }
    if let Some(obj) = <&ZendClassObject<PhpNumericRangeQuery>>::from_zval(zv) {
        let q: &PhpNumericRangeQuery = obj;
        return Ok(q.build());
    }
    if let Some(obj) = <&ZendClassObject<PhpDateTimeRangeQuery>>::from_zval(zv) {
        let q: &PhpDateTimeRangeQuery = obj;
        return q
            .build()
            .map_err(|e| ext_php_rs::exception::PhpException::default(e.to_string()));
    }
    if let Some(obj) = <&ZendClassObject<PhpGeoDistanceQuery>>::from_zval(zv) {
        let q: &PhpGeoDistanceQuery = obj;
        return q
            .build()
            .map_err(|e| ext_php_rs::exception::PhpException::default(e.to_string()));
    }
    if let Some(obj) = <&ZendClassObject<PhpGeoBoundingBoxQuery>>::from_zval(zv) {
        let q: &PhpGeoBoundingBoxQuery = obj;
        return q
            .build()
            .map_err(|e| ext_php_rs::exception::PhpException::default(e.to_string()));
    }
    if let Some(obj) = <&ZendClassObject<PhpGeo3dDistanceQuery>>::from_zval(zv) {
        let q: &PhpGeo3dDistanceQuery = obj;
        return Ok(q.build());
    }
    if let Some(obj) = <&ZendClassObject<PhpGeo3dBoundingBoxQuery>>::from_zval(zv) {
        let q: &PhpGeo3dBoundingBoxQuery = obj;
        return q
            .build()
            .map_err(|e| ext_php_rs::exception::PhpException::default(e.to_string()));
    }
    if let Some(obj) = <&ZendClassObject<PhpGeo3dNearestQuery>>::from_zval(zv) {
        let q: &PhpGeo3dNearestQuery = obj;
        return Ok(q.build());
    }
    if let Some(obj) = <&ZendClassObject<PhpBooleanQuery>>::from_zval(zv) {
        let q: &PhpBooleanQuery = obj;
        return q.build_query();
    }
    if let Some(obj) = <&ZendClassObject<PhpSpanQuery>>::from_zval(zv) {
        let q: &PhpSpanQuery = obj;
        return Ok(Box::new(SpanQueryWrapper::new(q.kind.build(&q.field))));
    }
    Err("Expected a lexical query type (TermQuery, BooleanQuery, …)".into())
}

/// Wrap an arbitrary PHP query Zval as a `LexicalSearchQuery::Obj`.
///
/// # Arguments
///
/// * `zv` - PHP Zval query object.
///
/// # Returns
///
/// A `LexicalSearchQuery` wrapping the extracted query.
pub fn zval_to_lexical_search_query(zv: &Zval) -> PhpResult<LexicalSearchQuery> {
    Ok(LexicalSearchQuery::Obj(extract_lexical_query(zv)?))
}

/// Check whether the PHP Zval is a vector query type.
///
/// # Arguments
///
/// * `zv` - PHP Zval to check.
///
/// # Returns
///
/// `true` if the value is a `VectorQuery` or `VectorTextQuery`.
pub fn is_vector_query(zv: &Zval) -> bool {
    <&ZendClassObject<PhpVectorQuery>>::from_zval(zv).is_some()
        || <&ZendClassObject<PhpVectorTextQuery>>::from_zval(zv).is_some()
}

/// Convert a vector query PHP Zval into a [`VectorSearchQuery`].
///
/// # Arguments
///
/// * `zv` - PHP Zval that should be `VectorQuery` or `VectorTextQuery`.
///
/// # Returns
///
/// The corresponding `VectorSearchQuery`.
pub fn zval_to_vector_search_query(zv: &Zval) -> PhpResult<VectorSearchQuery> {
    if let Some(obj) = <&ZendClassObject<PhpVectorQuery>>::from_zval(zv) {
        let q: &PhpVectorQuery = obj;
        return Ok(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(q.vector.clone()),
            weight: 1.0,
            fields: Some(vec![q.field.clone()]),
        }]));
    }
    if let Some(obj) = <&ZendClassObject<PhpVectorTextQuery>>::from_zval(zv) {
        let q: &PhpVectorTextQuery = obj;
        return Ok(VectorSearchQuery::Payloads(vec![QueryPayload::new(
            &q.field,
            DataValue::Text(q.text.clone()),
        )]));
    }
    Err("Expected VectorQuery or VectorTextQuery".into())
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

/// Exact single-term lexical query (`Laurus\TermQuery`).
#[php_class]
#[php(name = "Laurus\\TermQuery")]
pub struct PhpTermQuery {
    pub field: String,
    pub term: String,
}

#[php_impl]
impl PhpTermQuery {
    /// Create a new term query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field name to search.
    /// * `term` - Exact term to match.
    pub fn __construct(field: String, term: String) -> Self {
        Self { field, term }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!("TermQuery(field='{}', term='{}')", self.field, self.term)
    }
}

// ---------------------------------------------------------------------------
// PhraseQuery
// ---------------------------------------------------------------------------

/// Exact phrase (word-sequence) lexical query (`Laurus\PhraseQuery`).
#[php_class]
#[php(name = "Laurus\\PhraseQuery")]
pub struct PhpPhraseQuery {
    pub field: String,
    pub terms: Vec<String>,
}

#[php_impl]
impl PhpPhraseQuery {
    /// Create a new phrase query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field name to search.
    /// * `terms` - Array of terms forming the phrase.
    pub fn __construct(field: String, terms: Vec<String>) -> Self {
        Self { field, terms }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "PhraseQuery(field='{}', terms={:?})",
            self.field, self.terms
        )
    }
}

// ---------------------------------------------------------------------------
// FuzzyQuery
// ---------------------------------------------------------------------------

/// Approximate (typo-tolerant) lexical query (`Laurus\FuzzyQuery`).
#[php_class]
#[php(name = "Laurus\\FuzzyQuery")]
pub struct PhpFuzzyQuery {
    pub field: String,
    pub term: String,
    pub max_edits: u32,
}

#[php_impl]
impl PhpFuzzyQuery {
    /// Create a new fuzzy query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field name.
    /// * `term` - Term to match approximately.
    /// * `max_edits` - Maximum edit distance (default: 2).
    #[php(defaults(max_edits = 2))]
    pub fn __construct(field: String, term: String, max_edits: i64) -> Self {
        Self {
            field,
            term,
            max_edits: max_edits as u32,
        }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "FuzzyQuery(field='{}', term='{}', max_edits={})",
            self.field, self.term, self.max_edits
        )
    }
}

// ---------------------------------------------------------------------------
// WildcardQuery
// ---------------------------------------------------------------------------

/// Wildcard pattern lexical query (`Laurus\WildcardQuery`).
///
/// `*` matches any sequence, `?` matches any single character.
#[php_class]
#[php(name = "Laurus\\WildcardQuery")]
pub struct PhpWildcardQuery {
    pub field: String,
    pub pattern: String,
}

#[php_impl]
impl PhpWildcardQuery {
    /// Create a new wildcard query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field name to search.
    /// * `pattern` - Wildcard pattern.
    pub fn __construct(field: String, pattern: String) -> Self {
        Self { field, pattern }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
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

/// Numeric range filter query (`Laurus\NumericRangeQuery`).
///
/// Use `numeric_type` parameter to specify "integer" or "float".
#[php_class]
#[php(name = "Laurus\\NumericRangeQuery")]
pub struct PhpNumericRangeQuery {
    pub field: String,
    pub kind: NumericKind,
}

#[php_impl]
impl PhpNumericRangeQuery {
    /// Create a new numeric range query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field name.
    /// * `min` - Lower bound (optional, pass null for unbounded).
    /// * `max` - Upper bound (optional, pass null for unbounded).
    /// * `numeric_type` - "integer" or "float" (default: "integer").
    pub fn __construct(
        field: String,
        min: &Zval,
        max: &Zval,
        numeric_type: Option<String>,
    ) -> Self {
        let nt = numeric_type.unwrap_or_else(|| "integer".to_string());
        let kind = if nt == "float" {
            let min_f = if min.is_null() {
                None
            } else {
                f64::from_zval(min)
            };
            let max_f = if max.is_null() {
                None
            } else {
                f64::from_zval(max)
            };
            NumericKind::Float(min_f, max_f)
        } else {
            let min_i = if min.is_null() {
                None
            } else {
                i64::from_zval(min)
            };
            let max_i = if max.is_null() {
                None
            } else {
                i64::from_zval(max)
            };
            NumericKind::Integer(min_i, max_i)
        };
        Self { field, kind }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
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

impl PhpNumericRangeQuery {
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

/// DateTime range filter query (`Laurus\DateTimeRangeQuery`, Issue #1179).
///
/// Both bounds are inclusive; pass `null` to leave a side open. Bounds are
/// strings in any form the query DSL accepts: RFC 3339
/// (`"2024-01-01T09:00:00+09:00"`, normalized to UTC), a naive
/// `"YYYY-MM-DDTHH:MM:SS[.fff]"` (UTC), or a date `"YYYY-MM-DD"` (midnight
/// UTC). A `DateTimeInterface` can be passed as `$dt->format(DATE_RFC3339)`.
///
/// Constructors cannot fail in ext-php-rs, so a malformed bound is reported
/// when the query is used (`Index::search`, `BooleanQuery` clauses).
#[php_class]
#[php(name = "Laurus\\DateTimeRangeQuery")]
pub struct PhpDateTimeRangeQuery {
    pub field: String,
    pub min: Option<String>,
    pub max: Option<String>,
}

#[php_impl]
impl PhpDateTimeRangeQuery {
    /// Create a new datetime range query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field name.
    /// * `min` - Lower bound literal (optional, pass null for unbounded).
    /// * `max` - Upper bound literal (optional, pass null for unbounded).
    pub fn __construct(field: String, min: &Zval, max: &Zval) -> Self {
        let literal = |zv: &Zval| {
            if zv.is_null() {
                None
            } else {
                String::from_zval(zv)
            }
        };
        Self {
            field,
            min: literal(min),
            max: literal(max),
        }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "DateTimeRangeQuery(field='{}', min={:?}, max={:?})",
            self.field, self.min, self.max
        )
    }
}

impl PhpDateTimeRangeQuery {
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

// ---------------------------------------------------------------------------
// GeoDistanceQuery
// ---------------------------------------------------------------------------

/// Geographic distance (radius) search query (`Laurus\GeoDistanceQuery`).
#[php_class]
#[php(name = "Laurus\\GeoDistanceQuery")]
pub struct PhpGeoDistanceQuery {
    pub field: String,
    pub lat: f64,
    pub lon: f64,
    pub distance_m: f64,
}

#[php_impl]
impl PhpGeoDistanceQuery {
    /// Create a radius-based geo distance query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo field name.
    /// * `lat` - Center latitude.
    /// * `lon` - Center longitude.
    /// * `distance_m` - Maximum distance from the centre in meters.
    pub fn within_radius(field: String, lat: f64, lon: f64, distance_m: f64) -> Self {
        Self {
            field,
            lat,
            lon,
            distance_m,
        }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "GeoDistanceQuery.within_radius(field='{}', lat={}, lon={}, distance_m={})",
            self.field, self.lat, self.lon, self.distance_m
        )
    }
}

impl PhpGeoDistanceQuery {
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

/// Geographic bounding-box search query (`Laurus\GeoBoundingBoxQuery`).
#[php_class]
#[php(name = "Laurus\\GeoBoundingBoxQuery")]
pub struct PhpGeoBoundingBoxQuery {
    pub field: String,
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
}

#[php_impl]
impl PhpGeoBoundingBoxQuery {
    /// Create a bounding-box geo query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo field name.
    /// * `min_lat` - Southern boundary.
    /// * `min_lon` - Western boundary.
    /// * `max_lat` - Northern boundary.
    /// * `max_lon` - Eastern boundary.
    pub fn within_bounding_box(
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

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "GeoBoundingBoxQuery.within_bounding_box(field='{}', min_lat={}, min_lon={}, max_lat={}, max_lon={})",
            self.field, self.min_lat, self.min_lon, self.max_lat, self.max_lon
        )
    }
}

impl PhpGeoBoundingBoxQuery {
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

/// 3D ECEF sphere query (`Laurus\Geo3dDistanceQuery`).
///
/// Returns documents whose 3D ECEF point lies within `distance_m` meters of the
/// query centre. Construct via the `withinSphere` static factory.
#[php_class]
#[php(name = "Laurus\\Geo3dDistanceQuery")]
pub struct PhpGeo3dDistanceQuery {
    pub field: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub distance_m: f64,
}

#[php_impl]
impl PhpGeo3dDistanceQuery {
    /// Create a sphere-based 3D geo query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo3d field name.
    /// * `x` - Centre ECEF X coordinate (meters).
    /// * `y` - Centre ECEF Y coordinate (meters).
    /// * `z` - Centre ECEF Z coordinate (meters).
    /// * `distance_m` - Maximum distance from the centre in meters
    ///   (i.e. the search sphere's radius).
    pub fn within_sphere(field: String, x: f64, y: f64, z: f64, distance_m: f64) -> Self {
        Self {
            field,
            x,
            y,
            z,
            distance_m,
        }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "Geo3dDistanceQuery(field='{}', x={}, y={}, z={}, distance_m={})",
            self.field, self.x, self.y, self.z, self.distance_m
        )
    }
}

impl PhpGeo3dDistanceQuery {
    /// Build the underlying Rust [`Geo3dDistanceQuery`].
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

/// 3D ECEF axis-aligned bounding box query (`Laurus\Geo3dBoundingBoxQuery`).
///
/// Returns documents whose 3D ECEF point lies inside the AABB defined by
/// `[min_x, max_x] x [min_y, max_y] x [min_z, max_z]`. Construct via the
/// `withinBox` static factory.
#[php_class]
#[php(name = "Laurus\\Geo3dBoundingBoxQuery")]
pub struct PhpGeo3dBoundingBoxQuery {
    pub field: String,
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

#[php_impl]
impl PhpGeo3dBoundingBoxQuery {
    /// Create a 3D AABB geo query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo3d field name.
    /// * `min_x` - Minimum ECEF X (meters).
    /// * `min_y` - Minimum ECEF Y (meters).
    /// * `min_z` - Minimum ECEF Z (meters).
    /// * `max_x` - Maximum ECEF X (meters).
    /// * `max_y` - Maximum ECEF Y (meters).
    /// * `max_z` - Maximum ECEF Z (meters).
    #[allow(clippy::too_many_arguments)]
    pub fn within_box(
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

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "Geo3dBoundingBoxQuery(field='{}', min=({}, {}, {}), max=({}, {}, {}))",
            self.field, self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z
        )
    }
}

impl PhpGeo3dBoundingBoxQuery {
    /// Build the underlying Rust [`Geo3dBoundingBoxQuery`].
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

/// 3D ECEF k-nearest-neighbours query (`Laurus\Geo3dNearestQuery`).
///
/// Returns the `k` documents whose 3D ECEF points are closest to the query
/// centre. Construct via the `kNearest` static factory.
#[php_class]
#[php(name = "Laurus\\Geo3dNearestQuery")]
pub struct PhpGeo3dNearestQuery {
    pub field: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub k: u32,
    pub initial_radius_m: Option<f64>,
    pub max_radius_m: Option<f64>,
}

#[php_impl]
impl PhpGeo3dNearestQuery {
    /// Create a k-NN 3D geo query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo3d field name.
    /// * `x` - Centre ECEF X coordinate (meters).
    /// * `y` - Centre ECEF Y coordinate (meters).
    /// * `z` - Centre ECEF Z coordinate (meters).
    /// * `k` - Number of nearest neighbours to return.
    /// * `initial_radius_m` - Starting radius for the expanding-radius search
    ///   in meters (optional, default 1000.0).
    /// * `max_radius_m` - Hard cap on the search radius in meters (optional,
    ///   default 1e10).
    #[allow(clippy::too_many_arguments)]
    pub fn k_nearest(
        field: String,
        x: f64,
        y: f64,
        z: f64,
        k: i64,
        initial_radius_m: Option<f64>,
        max_radius_m: Option<f64>,
    ) -> Self {
        Self {
            field,
            x,
            y,
            z,
            k: k as u32,
            initial_radius_m,
            max_radius_m,
        }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "Geo3dNearestQuery.kNearest(field='{}', x={}, y={}, z={}, k={})",
            self.field, self.x, self.y, self.z, self.k
        )
    }
}

impl PhpGeo3dNearestQuery {
    /// Build the underlying Rust [`Geo3dNearestQuery`].
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
// BooleanQuery
// ---------------------------------------------------------------------------

/// Boolean combination query (`Laurus\BooleanQuery`).
///
/// Supports AND (must), OR (should), and NOT (must_not) clauses.
#[php_class]
#[php(name = "Laurus\\BooleanQuery")]
pub struct PhpBooleanQuery {
    pub musts: RefCell<Vec<Box<dyn laurus::lexical::Query>>>,
    pub shoulds: RefCell<Vec<Box<dyn laurus::lexical::Query>>>,
    pub must_nots: RefCell<Vec<Box<dyn laurus::lexical::Query>>>,
}

#[php_impl]
impl PhpBooleanQuery {
    /// Create a new empty boolean query.
    pub fn __construct() -> Self {
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
    pub fn must(&self, query: &Zval) -> PhpResult<()> {
        let q = extract_lexical_query(query)?;
        self.musts.borrow_mut().push(q);
        Ok(())
    }

    /// Add a SHOULD (optional, boosts score) clause.
    ///
    /// # Arguments
    ///
    /// * `query` - A lexical query object.
    pub fn should(&self, query: &Zval) -> PhpResult<()> {
        let q = extract_lexical_query(query)?;
        self.shoulds.borrow_mut().push(q);
        Ok(())
    }

    /// Add a MUST_NOT (exclusion) clause.
    ///
    /// # Arguments
    ///
    /// * `query` - A lexical query object.
    pub fn must_not(&self, query: &Zval) -> PhpResult<()> {
        let q = extract_lexical_query(query)?;
        self.must_nots.borrow_mut().push(q);
        Ok(())
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "BooleanQuery(musts={}, shoulds={}, must_nots={})",
            self.musts.borrow().len(),
            self.shoulds.borrow().len(),
            self.must_nots.borrow().len()
        )
    }
}

impl PhpBooleanQuery {
    /// Build the underlying Rust [`BooleanQuery`].
    pub fn build_query(&self) -> PhpResult<Box<dyn laurus::lexical::Query>> {
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

/// Positional / proximity span query (`Laurus\SpanQuery`).
///
/// Use the static methods to construct span queries, which can be nested to
/// build complex positional expressions.
#[php_class]
#[php(name = "Laurus\\SpanQuery")]
pub struct PhpSpanQuery {
    pub field: String,
    pub kind: SpanKind,
}

#[php_impl]
impl PhpSpanQuery {
    /// Single-term span query.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to search.
    /// * `term` - Term to match.
    pub fn term(field: String, term: String) -> Self {
        Self {
            field,
            kind: SpanKind::Term(term),
        }
    }

    /// SpanNear: terms appearing within `slop` positions of each other.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to search.
    /// * `terms` - Array of term strings.
    /// * `slop` - Maximum token distance between terms (default: 0).
    /// * `ordered` - Whether terms must appear in order (default: true).
    #[php(defaults(slop = 0, ordered = true))]
    pub fn near(field: String, terms: Vec<String>, slop: i64, ordered: bool) -> Self {
        let kinds = terms.into_iter().map(SpanKind::Term).collect();
        Self {
            field,
            kind: SpanKind::Near(kinds, slop as u32, ordered),
        }
    }

    /// SpanNear with nested `SpanQuery` clauses instead of plain terms.
    ///
    /// Use this when you need a `SpanNear` that contains other span queries
    /// (e.g. another `SpanNear`, a `SpanContaining`, or a `SpanWithin`)
    /// rather than just term clauses.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to search.
    /// * `clauses` - Array of `Laurus\\SpanQuery` instances. Each clause's
    ///   `field` is replaced by `field` at build time, so nested clauses are
    ///   re-rooted onto the outer field.
    /// * `slop` - Maximum token distance between clauses (default: 0).
    /// * `ordered` - Whether clauses must appear in order (default: true).
    #[php(defaults(slop = 0, ordered = true))]
    pub fn near_spans(
        field: String,
        clauses: Vec<&PhpSpanQuery>,
        slop: i64,
        ordered: bool,
    ) -> Self {
        let kinds: Vec<SpanKind> = clauses.iter().map(|c| c.kind.clone()).collect();
        Self {
            field,
            kind: SpanKind::Near(kinds, slop as u32, ordered),
        }
    }

    /// SpanContaining: a span that contains another span.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to search.
    /// * `big` - Outer span query.
    /// * `little` - Inner span query.
    pub fn containing(field: String, big: &PhpSpanQuery, little: &PhpSpanQuery) -> Self {
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
    pub fn within(
        field: String,
        include: &PhpSpanQuery,
        exclude: &PhpSpanQuery,
        distance: i64,
    ) -> Self {
        Self {
            field,
            kind: SpanKind::Within(
                Box::new(include.kind.clone()),
                Box::new(exclude.kind.clone()),
                distance as u32,
            ),
        }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!("SpanQuery(field='{}')", self.field)
    }
}

// ---------------------------------------------------------------------------
// VectorQuery (pre-computed vector)
// ---------------------------------------------------------------------------

/// Vector search query using a pre-computed embedding vector (`Laurus\VectorQuery`).
#[php_class]
#[php(name = "Laurus\\VectorQuery")]
pub struct PhpVectorQuery {
    pub field: String,
    pub vector: Vec<f32>,
}

#[php_impl]
impl PhpVectorQuery {
    /// Create a new vector query.
    ///
    /// # Arguments
    ///
    /// * `field` - Vector field name.
    /// * `vector` - Pre-computed embedding vector as array of floats.
    pub fn __construct(field: String, vector: Vec<f64>) -> Self {
        Self {
            field,
            vector: vector.into_iter().map(|f| f as f32).collect(),
        }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
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
/// (`Laurus\VectorTextQuery`).
#[php_class]
#[php(name = "Laurus\\VectorTextQuery")]
pub struct PhpVectorTextQuery {
    pub field: String,
    pub text: String,
}

#[php_impl]
impl PhpVectorTextQuery {
    /// Create a new vector text query.
    ///
    /// # Arguments
    ///
    /// * `field` - Vector field name.
    /// * `text` - Text to be embedded by the Rust-side embedder.
    pub fn __construct(field: String, text: String) -> Self {
        Self { field, text }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "VectorTextQuery(field='{}', text='{}')",
            self.field, self.text
        )
    }
}
