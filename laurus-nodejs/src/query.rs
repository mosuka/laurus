//! Node.js wrappers for all Laurus query types.
//!
//! Each query class stores the data needed to construct the Rust query.
//! Vector query classes produce `VectorSearchQuery` instead of lexical queries.

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
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// Helper: extract a lexical query from a JsQuery enum
// ---------------------------------------------------------------------------

/// Extract a Laurus lexical query from a [`JsQuery`] enum variant.
///
/// # Arguments
///
/// * `query` - A reference to a `JsQuery` enum.
///
/// # Returns
///
/// A boxed Laurus lexical query trait object.
pub fn extract_lexical_query(query: &JsQuery) -> Result<Box<dyn laurus::lexical::Query>> {
    match query {
        JsQuery::TermQuery(q) => Ok(Box::new(TermQuery::new(&q.field, &q.term))),
        JsQuery::PhraseQuery(q) => Ok(Box::new(PhraseQuery::new(&q.field, q.terms.clone()))),
        JsQuery::FuzzyQuery(q) => Ok(Box::new(
            FuzzyQuery::new(&q.field, &q.term).max_edits(q.max_edits),
        )),
        JsQuery::WildcardQuery(q) => Ok(Box::new(
            WildcardQuery::new(&q.field, &q.pattern)
                .map_err(|e| napi::Error::from_reason(e.to_string()))?,
        )),
        JsQuery::NumericRangeQuery(q) => Ok(q.build()),
        JsQuery::DateTimeRangeQuery(q) => q
            .build()
            .map_err(|e| napi::Error::from_reason(e.to_string())),
        JsQuery::GeoDistanceQuery(q) => q
            .build()
            .map_err(|e| napi::Error::from_reason(e.to_string())),
        JsQuery::GeoBoundingBoxQuery(q) => q
            .build()
            .map_err(|e| napi::Error::from_reason(e.to_string())),
        JsQuery::Geo3dDistanceQuery(q) => Ok(q.build()),
        JsQuery::Geo3dBoundingBoxQuery(q) => q
            .build()
            .map_err(|e| napi::Error::from_reason(e.to_string())),
        JsQuery::Geo3dNearestQuery(q) => Ok(q.build()),
        JsQuery::BooleanQuery(q) => q.build_query(),
        JsQuery::SpanQuery(q) => Ok(Box::new(SpanQueryWrapper::new(q.kind.build(&q.field)))),
    }
}

/// Convert a [`JsQuery`] into a [`LexicalSearchQuery`].
///
/// # Arguments
///
/// * `query` - A reference to a `JsQuery` enum.
///
/// # Returns
///
/// A `LexicalSearchQuery::Obj` wrapping the extracted query.
pub fn query_to_lexical_search_query(query: &JsQuery) -> Result<LexicalSearchQuery> {
    Ok(LexicalSearchQuery::Obj(extract_lexical_query(query)?))
}

/// Convert a [`JsVectorQuery`] into a [`VectorSearchQuery`].
///
/// # Arguments
///
/// * `query` - A reference to a `JsVectorQuery` enum.
///
/// # Returns
///
/// A `VectorSearchQuery` for the engine.
pub fn vector_query_to_search_query(query: &JsVectorQuery) -> VectorSearchQuery {
    match query {
        JsVectorQuery::VectorQuery(q) => VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(q.vector.clone()),
            weight: 1.0,
            fields: Some(vec![q.field.clone()]),
        }]),
        JsVectorQuery::VectorTextQuery(q) => VectorSearchQuery::Payloads(vec![QueryPayload::new(
            &q.field,
            DataValue::Text(q.text.clone()),
        )]),
    }
}

// ---------------------------------------------------------------------------
// Query union types (napi does not support trait objects, so we use enums)
// ---------------------------------------------------------------------------

/// Enum wrapping all supported lexical query types.
///
/// Used internally to pass query objects across the JS/Rust boundary.
#[derive(Clone)]
pub enum JsQuery {
    TermQuery(JsTermQuery),
    PhraseQuery(JsPhraseQuery),
    FuzzyQuery(JsFuzzyQuery),
    WildcardQuery(JsWildcardQuery),
    NumericRangeQuery(JsNumericRangeQuery),
    DateTimeRangeQuery(JsDateTimeRangeQuery),
    GeoDistanceQuery(JsGeoDistanceQuery),
    GeoBoundingBoxQuery(JsGeoBoundingBoxQuery),
    Geo3dDistanceQuery(JsGeo3dDistanceQuery),
    Geo3dBoundingBoxQuery(JsGeo3dBoundingBoxQuery),
    Geo3dNearestQuery(JsGeo3dNearestQuery),
    BooleanQuery(JsBooleanQuery),
    SpanQuery(JsSpanQuery),
}

/// Enum wrapping all supported vector query types.
pub enum JsVectorQuery {
    VectorQuery(JsVectorQueryInner),
    VectorTextQuery(JsVectorTextQuery),
}

// ---------------------------------------------------------------------------
// Internal span-query recipe enum (Clone so it can be nested)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum SpanKind {
    Term(String),
    Near(Vec<SpanKind>, u32, bool),
    Containing(Box<SpanKind>, Box<SpanKind>),
    Within(Box<SpanKind>, Box<SpanKind>, u32),
}

impl SpanKind {
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

/// Exact single-term lexical query.
///
/// ## Example
///
/// ```javascript
/// const { TermQuery } = require("laurus-nodejs");
/// const q = new TermQuery("body", "rust");
/// const results = await index.search(q, { limit: 5 });
/// ```
#[derive(Clone)]
#[napi(js_name = "TermQuery")]
pub struct JsTermQuery {
    pub(crate) field: String,
    pub(crate) term: String,
}

#[napi]
impl JsTermQuery {
    /// Create a new term query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name to search in.
    /// * `term` - The exact term to match.
    #[napi(constructor)]
    pub fn new(field: String, term: String) -> Self {
        Self { field, term }
    }
}

// ---------------------------------------------------------------------------
// PhraseQuery
// ---------------------------------------------------------------------------

/// Exact phrase (word-sequence) lexical query.
///
/// ## Example
///
/// ```javascript
/// const { PhraseQuery } = require("laurus-nodejs");
/// const q = new PhraseQuery("body", ["machine", "learning"]);
/// ```
#[derive(Clone)]
#[napi(js_name = "PhraseQuery")]
pub struct JsPhraseQuery {
    pub(crate) field: String,
    pub(crate) terms: Vec<String>,
}

#[napi]
impl JsPhraseQuery {
    /// Create a new phrase query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name to search in.
    /// * `terms` - The ordered list of terms that form the phrase.
    #[napi(constructor)]
    pub fn new(field: String, terms: Vec<String>) -> Self {
        Self { field, terms }
    }
}

// ---------------------------------------------------------------------------
// FuzzyQuery
// ---------------------------------------------------------------------------

/// Approximate (typo-tolerant) lexical query.
///
/// ## Example
///
/// ```javascript
/// const { FuzzyQuery } = require("laurus-nodejs");
/// const q = new FuzzyQuery("body", "programing", 2);
/// ```
#[derive(Clone)]
#[napi(js_name = "FuzzyQuery")]
pub struct JsFuzzyQuery {
    pub(crate) field: String,
    pub(crate) term: String,
    pub(crate) max_edits: u32,
}

#[napi]
impl JsFuzzyQuery {
    /// Create a new fuzzy query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name to search in.
    /// * `term` - The approximate term to match.
    /// * `max_edits` - Maximum edit distance (default 2).
    #[napi(constructor)]
    pub fn new(field: String, term: String, max_edits: Option<u32>) -> Self {
        Self {
            field,
            term,
            max_edits: max_edits.unwrap_or(2),
        }
    }
}

// ---------------------------------------------------------------------------
// WildcardQuery
// ---------------------------------------------------------------------------

/// Wildcard pattern lexical query (`*` = any sequence, `?` = any character).
///
/// ## Example
///
/// ```javascript
/// const { WildcardQuery } = require("laurus-nodejs");
/// const q = new WildcardQuery("filename", "*.pdf");
/// ```
#[derive(Clone)]
#[napi(js_name = "WildcardQuery")]
pub struct JsWildcardQuery {
    pub(crate) field: String,
    pub(crate) pattern: String,
}

#[napi]
impl JsWildcardQuery {
    /// Create a new wildcard query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name to search in.
    /// * `pattern` - The wildcard pattern (`*` = any sequence, `?` = single character).
    #[napi(constructor)]
    pub fn new(field: String, pattern: String) -> Self {
        Self { field, pattern }
    }
}

// ---------------------------------------------------------------------------
// NumericRangeQuery
// ---------------------------------------------------------------------------

/// Numeric range filter query (integer or float).
///
/// Use the `numericType` parameter to choose between an integer range
/// (default `"integer"`) and a float range (`"float"`). The same string
/// discriminator is used by the PHP binding's `NumericRangeQuery`, so the
/// two bindings now agree on the call shape.
///
/// ## Example
///
/// ```javascript
/// const { NumericRangeQuery } = require("laurus-nodejs");
///
/// // Integer range (default).
/// const q1 = new NumericRangeQuery("year", 2020, 2023);
/// const q2 = new NumericRangeQuery("year", 2020, 2023, "integer");
///
/// // Float range.
/// const q3 = new NumericRangeQuery("price", 9.99, 19.99, "float");
/// ```
#[derive(Clone)]
#[napi(js_name = "NumericRangeQuery")]
pub struct JsNumericRangeQuery {
    pub(crate) field: String,
    pub(crate) min: Option<f64>,
    pub(crate) max: Option<f64>,
    pub(crate) is_float: bool,
}

#[napi]
impl JsNumericRangeQuery {
    /// Create a new numeric range query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name to filter on.
    /// * `min` - Minimum value (inclusive), or `null` for unbounded.
    /// * `max` - Maximum value (inclusive), or `null` for unbounded.
    /// * `numeric_type` - Either `"integer"` or `"float"`. Defaults to
    ///   `"integer"` when omitted. Anything else throws an `Error`.
    #[napi(constructor)]
    pub fn new(
        field: String,
        min: Option<f64>,
        max: Option<f64>,
        numeric_type: Option<String>,
    ) -> Result<Self> {
        let is_float = match numeric_type.as_deref() {
            Some("integer") | None => false,
            Some("float") => true,
            Some(other) => {
                return Err(napi::Error::from_reason(format!(
                    "numericType must be \"integer\" or \"float\", got {other:?}"
                )));
            }
        };
        Ok(Self {
            field,
            min,
            max,
            is_float,
        })
    }
}

impl JsNumericRangeQuery {
    pub fn build(&self) -> Box<dyn laurus::lexical::Query> {
        if self.is_float {
            Box::new(NumericRangeQuery::f64_range(
                &self.field,
                self.min,
                self.max,
            ))
        } else {
            Box::new(NumericRangeQuery::i64_range(
                &self.field,
                self.min.map(|v| v as i64),
                self.max.map(|v| v as i64),
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// DateTimeRangeQuery
// ---------------------------------------------------------------------------

/// DateTime range query (Issue #1179).
///
/// Both bounds are inclusive; pass `null` to leave a side open. Bounds are
/// strings in any form the query DSL accepts: RFC 3339
/// (`"2024-01-01T09:00:00+09:00"`, normalized to UTC), a naive
/// `"YYYY-MM-DDTHH:MM:SS[.fff]"` (interpreted as UTC), or a date
/// `"YYYY-MM-DD"` (midnight UTC). A JS `Date` can be passed as
/// `date.toISOString()`.
///
/// ```javascript
/// const q = new DateTimeRangeQuery("createdAt", "2024-01-01", "2024-12-31");
/// const open = new DateTimeRangeQuery("createdAt", "2024-06-01T00:00:00Z", null);
/// ```
#[derive(Clone)]
#[napi(js_name = "DateTimeRangeQuery")]
pub struct JsDateTimeRangeQuery {
    pub(crate) field: String,
    pub(crate) min: Option<String>,
    pub(crate) max: Option<String>,
}

#[napi]
impl JsDateTimeRangeQuery {
    /// Create a new datetime range query.
    ///
    /// # Arguments
    ///
    /// * `field` - The DateTime field name to filter on.
    /// * `min` - Lower bound (inclusive), or `null` for unbounded.
    /// * `max` - Upper bound (inclusive), or `null` for unbounded.
    ///
    /// Throws an `Error` when a bound is not a recognized datetime literal.
    #[napi(constructor)]
    pub fn new(field: String, min: Option<String>, max: Option<String>) -> Result<Self> {
        let query = Self { field, min, max };
        query
            .build()
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(query)
    }
}

impl JsDateTimeRangeQuery {
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

/// Geographic distance (radius) search query.
///
/// ## Example
///
/// ```javascript
/// const { GeoDistanceQuery } = require("laurus-nodejs");
///
/// // Radius search: within 100 km of San Francisco
/// const q = GeoDistanceQuery.withinRadius("location", 37.77, -122.42, 100_000.0);
/// ```
#[derive(Clone)]
#[napi(js_name = "GeoDistanceQuery")]
pub struct JsGeoDistanceQuery {
    pub(crate) field: String,
    pub(crate) lat: f64,
    pub(crate) lon: f64,
    pub(crate) distance_m: f64,
}

#[napi]
impl JsGeoDistanceQuery {
    /// Create a radius-based geo distance query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo field name.
    /// * `lat` - Center latitude.
    /// * `lon` - Center longitude.
    /// * `distance_m` - Maximum distance from the centre in meters.
    #[napi(factory)]
    pub fn within_radius(field: String, lat: f64, lon: f64, distance_m: f64) -> Self {
        Self {
            field,
            lat,
            lon,
            distance_m,
        }
    }
}

impl JsGeoDistanceQuery {
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

/// Geographic bounding-box search query.
///
/// ## Example
///
/// ```javascript
/// const { GeoBoundingBoxQuery } = require("laurus-nodejs");
///
/// const q = GeoBoundingBoxQuery.withinBoundingBox(
///   "location", 33.0, -123.0, 48.0, -117.0,
/// );
/// ```
#[derive(Clone)]
#[napi(js_name = "GeoBoundingBoxQuery")]
pub struct JsGeoBoundingBoxQuery {
    pub(crate) field: String,
    pub(crate) min_lat: f64,
    pub(crate) min_lon: f64,
    pub(crate) max_lat: f64,
    pub(crate) max_lon: f64,
}

#[napi]
impl JsGeoBoundingBoxQuery {
    /// Create a bounding-box geo query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo field name.
    /// * `min_lat` - Southern boundary.
    /// * `min_lon` - Western boundary.
    /// * `max_lat` - Northern boundary.
    /// * `max_lon` - Eastern boundary.
    #[napi(factory)]
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
}

impl JsGeoBoundingBoxQuery {
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

/// 3D ECEF sphere query: matches documents whose stored `(x, y, z)` point
/// lies within `distance_m` meters of the given centre.
///
/// ## Example
///
/// ```javascript
/// const { Geo3dDistanceQuery } = require("laurus-nodejs");
/// const q = Geo3dDistanceQuery.withinSphere(
///     "position", -3955182.0, 3350553.0, 3700276.0, 5000.0,
/// );
/// ```
#[derive(Clone)]
#[napi(js_name = "Geo3dDistanceQuery")]
pub struct JsGeo3dDistanceQuery {
    pub(crate) field: String,
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
    pub(crate) distance_m: f64,
}

#[napi]
impl JsGeo3dDistanceQuery {
    /// Create a 3D distance (sphere) query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo3d field name.
    /// * `x`, `y`, `z` - Centre coordinates in ECEF meters.
    /// * `distance_m` - Maximum distance from the centre in meters
    ///   (i.e. the search sphere's radius).
    #[napi(factory)]
    pub fn within_sphere(field: String, x: f64, y: f64, z: f64, distance_m: f64) -> Self {
        Self {
            field,
            x,
            y,
            z,
            distance_m,
        }
    }
}

impl JsGeo3dDistanceQuery {
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

/// 3D ECEF axis-aligned bounding box query.
///
/// ## Example
///
/// ```javascript
/// const { Geo3dBoundingBoxQuery } = require("laurus-nodejs");
/// const q = Geo3dBoundingBoxQuery.withinBox(
///     "position",
///     -3962000.0, 3340000.0, 3690000.0,
///     -3954000.0, 3360000.0, 3710000.0,
/// );
/// ```
#[derive(Clone)]
#[napi(js_name = "Geo3dBoundingBoxQuery")]
pub struct JsGeo3dBoundingBoxQuery {
    pub(crate) field: String,
    pub(crate) min_x: f64,
    pub(crate) min_y: f64,
    pub(crate) min_z: f64,
    pub(crate) max_x: f64,
    pub(crate) max_y: f64,
    pub(crate) max_z: f64,
}

#[napi]
impl JsGeo3dBoundingBoxQuery {
    /// Create a 3D bounding-box query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo3d field name.
    /// * `min_x`, `min_y`, `min_z` - Lower corner of the box.
    /// * `max_x`, `max_y`, `max_z` - Upper corner of the box.
    #[napi(factory)]
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
}

impl JsGeo3dBoundingBoxQuery {
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

/// 3D ECEF k-nearest-neighbours query.
///
/// ## Example
///
/// ```javascript
/// const { Geo3dNearestQuery } = require("laurus-nodejs");
/// const q = Geo3dNearestQuery.kNearest(
///     "position", -3955182.0, 3350553.0, 3700276.0, 10,
/// );
/// ```
#[derive(Clone)]
#[napi(js_name = "Geo3dNearestQuery")]
pub struct JsGeo3dNearestQuery {
    pub(crate) field: String,
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
    pub(crate) k: u32,
    pub(crate) initial_radius_m: Option<f64>,
    pub(crate) max_radius_m: Option<f64>,
}

#[napi]
impl JsGeo3dNearestQuery {
    /// Create a 3D k-NN query.
    ///
    /// # Arguments
    ///
    /// * `field` - Geo3d field name.
    /// * `x`, `y`, `z` - Centre coordinates in ECEF meters.
    /// * `k` - Number of nearest neighbours to return.
    /// * `initial_radius_m` - Starting radius for the expanding-radius
    ///   search (default 1000 m). Optional.
    /// * `max_radius_m` - Hard cap on the search radius (default 1e10 m).
    ///   Optional.
    #[napi(factory)]
    pub fn k_nearest(
        field: String,
        x: f64,
        y: f64,
        z: f64,
        k: u32,
        initial_radius_m: Option<f64>,
        max_radius_m: Option<f64>,
    ) -> Self {
        Self {
            field,
            x,
            y,
            z,
            k,
            initial_radius_m,
            max_radius_m,
        }
    }
}

impl JsGeo3dNearestQuery {
    pub fn build(&self) -> Box<dyn laurus::lexical::Query> {
        let centre = GeoEcefPoint::new(self.x, self.y, self.z);
        let mut q = Geo3dNearestQuery::new(&self.field, centre, self.k as usize);
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

/// Boolean combination query (AND / OR / NOT).
///
/// ## Example
///
/// ```javascript
/// const { BooleanQuery, TermQuery } = require("laurus-nodejs");
///
/// const bq = new BooleanQuery();
/// bq.must(new TermQuery("body", "programming"));
/// bq.mustNot(new TermQuery("body", "python"));
/// bq.should(new TermQuery("category", "data-science"));
/// const results = await index.search(bq, { limit: 5 });
/// ```
#[derive(Clone)]
#[napi(js_name = "BooleanQuery")]
pub struct JsBooleanQuery {
    pub(crate) musts: Vec<JsQuery>,
    pub(crate) shoulds: Vec<JsQuery>,
    pub(crate) must_nots: Vec<JsQuery>,
}

#[napi]
impl JsBooleanQuery {
    /// Create a new empty boolean query.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            musts: Vec::new(),
            shoulds: Vec::new(),
            must_nots: Vec::new(),
        }
    }

    // The 36 methods below cover {must, should, must_not} × 12 query
    // types. We have to expose one method per concrete query type rather
    // than a single polymorphic `must(query)` taking a union, because
    // napi-derive's auto-generated `ValidateNapiValue` impl for `&T`
    // looks the constructor up by the **Rust struct name** instead of
    // the JS class name (set via `#[napi(js_name = ...)]`). That bug
    // makes any `Either<&T, ...>`-style polymorphic argument fail at
    // runtime ("Value is non of these types `&JsTermQuery`, …") even
    // when the caller passes a correctly constructed instance. Direct
    // `&T` arguments still work because napi-derive special-cases them
    // at the function-signature level via `FromNapiRef`.

    /// Add a MUST clause from a [`JsTermQuery`].
    #[napi]
    pub fn must_term(&mut self, query: &JsTermQuery) {
        self.musts.push(JsQuery::TermQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsPhraseQuery`].
    #[napi]
    pub fn must_phrase(&mut self, query: &JsPhraseQuery) {
        self.musts.push(JsQuery::PhraseQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsFuzzyQuery`].
    #[napi]
    pub fn must_fuzzy(&mut self, query: &JsFuzzyQuery) {
        self.musts.push(JsQuery::FuzzyQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsWildcardQuery`].
    #[napi]
    pub fn must_wildcard(&mut self, query: &JsWildcardQuery) {
        self.musts.push(JsQuery::WildcardQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsNumericRangeQuery`].
    #[napi]
    pub fn must_numeric_range(&mut self, query: &JsNumericRangeQuery) {
        self.musts.push(JsQuery::NumericRangeQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsDateTimeRangeQuery`].
    #[napi]
    pub fn must_date_time_range(&mut self, query: &JsDateTimeRangeQuery) {
        self.musts.push(JsQuery::DateTimeRangeQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsGeoDistanceQuery`].
    #[napi]
    pub fn must_geo_distance(&mut self, query: &JsGeoDistanceQuery) {
        self.musts.push(JsQuery::GeoDistanceQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsGeoBoundingBoxQuery`].
    #[napi]
    pub fn must_geo_bounding_box(&mut self, query: &JsGeoBoundingBoxQuery) {
        self.musts.push(JsQuery::GeoBoundingBoxQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsGeo3dDistanceQuery`].
    #[napi(js_name = "mustGeo3dDistance")]
    pub fn must_geo3d_distance(&mut self, query: &JsGeo3dDistanceQuery) {
        self.musts.push(JsQuery::Geo3dDistanceQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsGeo3dBoundingBoxQuery`].
    #[napi(js_name = "mustGeo3dBoundingBox")]
    pub fn must_geo3d_bounding_box(&mut self, query: &JsGeo3dBoundingBoxQuery) {
        self.musts
            .push(JsQuery::Geo3dBoundingBoxQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsGeo3dNearestQuery`].
    #[napi(js_name = "mustGeo3dNearest")]
    pub fn must_geo3d_nearest(&mut self, query: &JsGeo3dNearestQuery) {
        self.musts.push(JsQuery::Geo3dNearestQuery(query.clone()));
    }

    /// Add a MUST clause from another [`JsBooleanQuery`] (nested).
    #[napi]
    pub fn must_boolean(&mut self, query: &JsBooleanQuery) {
        self.musts.push(JsQuery::BooleanQuery(query.clone()));
    }

    /// Add a MUST clause from a [`JsSpanQuery`].
    #[napi]
    pub fn must_span(&mut self, query: &JsSpanQuery) {
        self.musts.push(JsQuery::SpanQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsTermQuery`].
    #[napi]
    pub fn should_term(&mut self, query: &JsTermQuery) {
        self.shoulds.push(JsQuery::TermQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsPhraseQuery`].
    #[napi]
    pub fn should_phrase(&mut self, query: &JsPhraseQuery) {
        self.shoulds.push(JsQuery::PhraseQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsFuzzyQuery`].
    #[napi]
    pub fn should_fuzzy(&mut self, query: &JsFuzzyQuery) {
        self.shoulds.push(JsQuery::FuzzyQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsWildcardQuery`].
    #[napi]
    pub fn should_wildcard(&mut self, query: &JsWildcardQuery) {
        self.shoulds.push(JsQuery::WildcardQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsNumericRangeQuery`].
    #[napi]
    pub fn should_numeric_range(&mut self, query: &JsNumericRangeQuery) {
        self.shoulds.push(JsQuery::NumericRangeQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsDateTimeRangeQuery`].
    #[napi]
    pub fn should_date_time_range(&mut self, query: &JsDateTimeRangeQuery) {
        self.shoulds
            .push(JsQuery::DateTimeRangeQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsGeoDistanceQuery`].
    #[napi]
    pub fn should_geo_distance(&mut self, query: &JsGeoDistanceQuery) {
        self.shoulds.push(JsQuery::GeoDistanceQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsGeoBoundingBoxQuery`].
    #[napi]
    pub fn should_geo_bounding_box(&mut self, query: &JsGeoBoundingBoxQuery) {
        self.shoulds
            .push(JsQuery::GeoBoundingBoxQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsGeo3dDistanceQuery`].
    #[napi(js_name = "shouldGeo3dDistance")]
    pub fn should_geo3d_distance(&mut self, query: &JsGeo3dDistanceQuery) {
        self.shoulds
            .push(JsQuery::Geo3dDistanceQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsGeo3dBoundingBoxQuery`].
    #[napi(js_name = "shouldGeo3dBoundingBox")]
    pub fn should_geo3d_bounding_box(&mut self, query: &JsGeo3dBoundingBoxQuery) {
        self.shoulds
            .push(JsQuery::Geo3dBoundingBoxQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsGeo3dNearestQuery`].
    #[napi(js_name = "shouldGeo3dNearest")]
    pub fn should_geo3d_nearest(&mut self, query: &JsGeo3dNearestQuery) {
        self.shoulds.push(JsQuery::Geo3dNearestQuery(query.clone()));
    }

    /// Add a SHOULD clause from another [`JsBooleanQuery`] (nested).
    #[napi]
    pub fn should_boolean(&mut self, query: &JsBooleanQuery) {
        self.shoulds.push(JsQuery::BooleanQuery(query.clone()));
    }

    /// Add a SHOULD clause from a [`JsSpanQuery`].
    #[napi]
    pub fn should_span(&mut self, query: &JsSpanQuery) {
        self.shoulds.push(JsQuery::SpanQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsTermQuery`].
    #[napi]
    pub fn must_not_term(&mut self, query: &JsTermQuery) {
        self.must_nots.push(JsQuery::TermQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsPhraseQuery`].
    #[napi]
    pub fn must_not_phrase(&mut self, query: &JsPhraseQuery) {
        self.must_nots.push(JsQuery::PhraseQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsFuzzyQuery`].
    #[napi]
    pub fn must_not_fuzzy(&mut self, query: &JsFuzzyQuery) {
        self.must_nots.push(JsQuery::FuzzyQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsWildcardQuery`].
    #[napi]
    pub fn must_not_wildcard(&mut self, query: &JsWildcardQuery) {
        self.must_nots.push(JsQuery::WildcardQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsNumericRangeQuery`].
    #[napi]
    pub fn must_not_numeric_range(&mut self, query: &JsNumericRangeQuery) {
        self.must_nots
            .push(JsQuery::NumericRangeQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsDateTimeRangeQuery`].
    #[napi]
    pub fn must_not_date_time_range(&mut self, query: &JsDateTimeRangeQuery) {
        self.must_nots
            .push(JsQuery::DateTimeRangeQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsGeoDistanceQuery`].
    #[napi]
    pub fn must_not_geo_distance(&mut self, query: &JsGeoDistanceQuery) {
        self.must_nots
            .push(JsQuery::GeoDistanceQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsGeoBoundingBoxQuery`].
    #[napi]
    pub fn must_not_geo_bounding_box(&mut self, query: &JsGeoBoundingBoxQuery) {
        self.must_nots
            .push(JsQuery::GeoBoundingBoxQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsGeo3dDistanceQuery`].
    #[napi(js_name = "mustNotGeo3dDistance")]
    pub fn must_not_geo3d_distance(&mut self, query: &JsGeo3dDistanceQuery) {
        self.must_nots
            .push(JsQuery::Geo3dDistanceQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsGeo3dBoundingBoxQuery`].
    #[napi(js_name = "mustNotGeo3dBoundingBox")]
    pub fn must_not_geo3d_bounding_box(&mut self, query: &JsGeo3dBoundingBoxQuery) {
        self.must_nots
            .push(JsQuery::Geo3dBoundingBoxQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsGeo3dNearestQuery`].
    #[napi(js_name = "mustNotGeo3dNearest")]
    pub fn must_not_geo3d_nearest(&mut self, query: &JsGeo3dNearestQuery) {
        self.must_nots
            .push(JsQuery::Geo3dNearestQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from another [`JsBooleanQuery`] (nested).
    #[napi]
    pub fn must_not_boolean(&mut self, query: &JsBooleanQuery) {
        self.must_nots.push(JsQuery::BooleanQuery(query.clone()));
    }

    /// Add a MUST_NOT clause from a [`JsSpanQuery`].
    #[napi]
    pub fn must_not_span(&mut self, query: &JsSpanQuery) {
        self.must_nots.push(JsQuery::SpanQuery(query.clone()));
    }
}

impl JsBooleanQuery {
    /// Build the underlying Rust [`BooleanQuery`].
    pub fn build_query(&self) -> Result<Box<dyn laurus::lexical::Query>> {
        let mut bq = BooleanQuery::new();
        for q in &self.musts {
            bq.add_must(extract_lexical_query(q)?);
        }
        for q in &self.shoulds {
            bq.add_should(extract_lexical_query(q)?);
        }
        for q in &self.must_nots {
            bq.add_must_not(extract_lexical_query(q)?);
        }
        Ok(Box::new(bq))
    }
}

// ---------------------------------------------------------------------------
// SpanQuery
// ---------------------------------------------------------------------------

/// Positional / proximity span query.
///
/// Use the static factory methods to construct span queries.
///
/// ## Example
///
/// ```javascript
/// const { SpanQuery } = require("laurus-nodejs");
///
/// // SpanNear: "quick" within 1 position of "fox", in order
/// const q = SpanQuery.near("body", ["quick", "fox"], 1, true);
/// ```
#[derive(Clone)]
#[napi(js_name = "SpanQuery")]
pub struct JsSpanQuery {
    pub(crate) field: String,
    pub(crate) kind: SpanKind,
}

#[napi]
impl JsSpanQuery {
    /// Single-term span query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name to search in.
    /// * `term` - The term to match.
    #[napi(factory)]
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
    /// * `terms` - List of term strings.
    /// * `slop` - Maximum token distance between terms (default 0).
    /// * `ordered` - Whether terms must appear in the given order (default `true`).
    #[napi(factory)]
    pub fn near(
        field: String,
        terms: Vec<String>,
        slop: Option<u32>,
        ordered: Option<bool>,
    ) -> Self {
        let kinds = terms.into_iter().map(SpanKind::Term).collect();
        Self {
            field,
            kind: SpanKind::Near(kinds, slop.unwrap_or(0), ordered.unwrap_or(true)),
        }
    }

    /// SpanNear with nested SpanQuery clauses instead of plain terms.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to search.
    /// * `clauses` - List of SpanQuery objects.
    /// * `slop` - Maximum token distance (default 0).
    /// * `ordered` - Whether clauses must appear in order (default `true`).
    #[napi(factory)]
    pub fn near_spans(
        field: String,
        clauses: Vec<&JsSpanQuery>,
        slop: Option<u32>,
        ordered: Option<bool>,
    ) -> Self {
        let kinds: Vec<SpanKind> = clauses.iter().map(|c| c.kind.clone()).collect();
        Self {
            field,
            kind: SpanKind::Near(kinds, slop.unwrap_or(0), ordered.unwrap_or(true)),
        }
    }

    /// SpanContaining: a span that contains another span.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to search.
    /// * `big` - The outer span query.
    /// * `little` - The inner span query that must be contained.
    #[napi(factory)]
    pub fn containing(field: String, big: &JsSpanQuery, little: &JsSpanQuery) -> Self {
        Self {
            field,
            kind: SpanKind::Containing(Box::new(big.kind.clone()), Box::new(little.kind.clone())),
        }
    }

    /// SpanWithin: a span included within another span, at a maximum distance.
    ///
    /// # Arguments
    ///
    /// * `field` - Field to search.
    /// * `include` - The span to include.
    /// * `exclude` - The span to measure distance from.
    /// * `distance` - Maximum distance.
    #[napi(factory)]
    pub fn within(
        field: String,
        include: &JsSpanQuery,
        exclude: &JsSpanQuery,
        distance: u32,
    ) -> Self {
        Self {
            field,
            kind: SpanKind::Within(
                Box::new(include.kind.clone()),
                Box::new(exclude.kind.clone()),
                distance,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// VectorQuery (pre-computed vector)
// ---------------------------------------------------------------------------

/// Vector search query using a pre-computed embedding vector.
///
/// ## Example
///
/// ```javascript
/// const { VectorQuery } = require("laurus-nodejs");
/// const results = await index.search(new VectorQuery("text_vec", [0.1, 0.2, ...]));
/// ```
#[derive(Clone)]
#[napi(js_name = "VectorQuery")]
pub struct JsVectorQueryInner {
    pub(crate) field: String,
    pub(crate) vector: Vec<f32>,
}

#[napi]
impl JsVectorQueryInner {
    /// Create a new vector query with a pre-computed embedding.
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `vector` - The embedding vector as an array of numbers.
    #[napi(constructor)]
    pub fn new(field: String, vector: Vec<f64>) -> Self {
        Self {
            field,
            vector: vector.into_iter().map(|v| v as f32).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// VectorTextQuery (text → Rust Embedder → vector)
// ---------------------------------------------------------------------------

/// Vector search query where the text is embedded by the Rust-side Embedder.
///
/// ## Example
///
/// ```javascript
/// const { VectorTextQuery } = require("laurus-nodejs");
/// const results = await index.search(new VectorTextQuery("text_vec", "memory safety"));
/// ```
#[derive(Clone)]
#[napi(js_name = "VectorTextQuery")]
pub struct JsVectorTextQuery {
    pub(crate) field: String,
    pub(crate) text: String,
}

#[napi]
impl JsVectorTextQuery {
    /// Create a new text-based vector query.
    ///
    /// The text will be automatically embedded by the registered embedder.
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `text` - The text to embed and search with.
    #[napi(constructor)]
    pub fn new(field: String, text: String) -> Self {
        Self { field, text }
    }
}
