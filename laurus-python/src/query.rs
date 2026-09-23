//! Python wrappers for all Laurus query types.
//!
//! Each Python query class stores the data needed to construct the Rust query
//! and provides a `build()` method that materializes it into a `Box<dyn Query>`.
//! Vector query classes produce `VectorSearchQuery` instead.

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
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Helper: extract a lexical LexicalSearchQuery from any Python query object
// ---------------------------------------------------------------------------

/// Extract a Laurus lexical query from an arbitrary Python object.
///
/// Supports: `TermQuery`, `PhraseQuery`, `FuzzyQuery`, `WildcardQuery`,
/// `NumericRangeQuery`, `GeoDistanceQuery`, `GeoBoundingBoxQuery`,
/// `Geo3dDistanceQuery`, `Geo3dBoundingBoxQuery`, `Geo3dNearestQuery`,
/// `BooleanQuery`, `SpanQuery`.
pub fn extract_lexical_query(
    py: Python,
    obj: &Bound<PyAny>,
) -> PyResult<Box<dyn laurus::lexical::Query>> {
    if let Ok(q) = obj.extract::<PyRef<PyTermQuery>>() {
        return Ok(Box::new(TermQuery::new(&q.field, &q.term)));
    }
    if let Ok(q) = obj.extract::<PyRef<PyPhraseQuery>>() {
        return Ok(Box::new(PhraseQuery::new(&q.field, q.terms.clone())));
    }
    if let Ok(q) = obj.extract::<PyRef<PyFuzzyQuery>>() {
        return Ok(Box::new(
            FuzzyQuery::new(&q.field, &q.term).max_edits(q.max_edits),
        ));
    }
    if let Ok(q) = obj.extract::<PyRef<PyWildcardQuery>>() {
        return Ok(Box::new(
            WildcardQuery::new(&q.field, &q.pattern)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        ));
    }
    if let Ok(q) = obj.extract::<PyRef<PyNumericRangeQuery>>() {
        return Ok(q.build());
    }
    if let Ok(q) = obj.extract::<PyRef<PyDateTimeRangeQuery>>() {
        return q.build().map_err(|e| PyValueError::new_err(e.to_string()));
    }
    if let Ok(q) = obj.extract::<PyRef<PyGeoDistanceQuery>>() {
        return q.build().map_err(|e| PyValueError::new_err(e.to_string()));
    }
    if let Ok(q) = obj.extract::<PyRef<PyGeoBoundingBoxQuery>>() {
        return q.build().map_err(|e| PyValueError::new_err(e.to_string()));
    }
    if let Ok(q) = obj.extract::<PyRef<PyGeo3dDistanceQuery>>() {
        return Ok(q.build());
    }
    if let Ok(q) = obj.extract::<PyRef<PyGeo3dBoundingBoxQuery>>() {
        return q.build().map_err(|e| PyValueError::new_err(e.to_string()));
    }
    if let Ok(q) = obj.extract::<PyRef<PyGeo3dNearestQuery>>() {
        return Ok(q.build());
    }
    if let Ok(q) = obj.extract::<PyRef<PyBooleanQuery>>() {
        return q.build_query(py);
    }
    if let Ok(q) = obj.extract::<PyRef<PySpanQuery>>() {
        return Ok(Box::new(SpanQueryWrapper::new(q.kind.build(&q.field))));
    }
    Err(PyValueError::new_err(format!(
        "Expected a lexical query type (TermQuery, BooleanQuery, …), got {}",
        obj.get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    )))
}

/// Try to wrap an arbitrary Python query object as a `LexicalSearchQuery::Obj`.
pub fn py_to_lexical_search_query(py: Python, obj: &Bound<PyAny>) -> PyResult<LexicalSearchQuery> {
    Ok(LexicalSearchQuery::Obj(extract_lexical_query(py, obj)?))
}

/// Check whether the Python object is a vector query type.
pub fn is_vector_query(obj: &Bound<PyAny>) -> bool {
    obj.is_instance_of::<PyVectorQuery>() || obj.is_instance_of::<PyVectorTextQuery>()
}

/// Convert a vector query Python object into a [`VectorSearchQuery`].
pub fn py_to_vector_search_query(obj: &Bound<PyAny>) -> PyResult<VectorSearchQuery> {
    if let Ok(q) = obj.extract::<PyRef<PyVectorQuery>>() {
        // Use Vectors variant directly for pre-computed embeddings.
        // Payloads with DataValue::Vector are silently skipped by the engine
        // (it only embeds Text/Bytes payloads), so we must use Vectors here.
        return Ok(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(q.vector.clone()),
            weight: 1.0,
            fields: Some(vec![q.field.clone()]),
        }]));
    }
    if let Ok(q) = obj.extract::<PyRef<PyVectorTextQuery>>() {
        return Ok(VectorSearchQuery::Payloads(vec![QueryPayload::new(
            &q.field,
            DataValue::Text(q.text.clone()),
        )]));
    }
    Err(PyValueError::new_err(
        "Expected VectorQuery or VectorTextQuery",
    ))
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
/// ```python
/// q = laurus.TermQuery("body", "rust")
/// results = index.search(q, limit=5)
/// ```
#[pyclass(name = "TermQuery")]
pub struct PyTermQuery {
    pub field: String,
    pub term: String,
}

#[pymethods]
impl PyTermQuery {
    #[new]
    pub fn new(field: String, term: String) -> Self {
        Self { field, term }
    }
    fn __repr__(&self) -> String {
        format!("TermQuery(field='{}', term='{}')", self.field, self.term)
    }
}

// ---------------------------------------------------------------------------
// PhraseQuery
// ---------------------------------------------------------------------------

/// Exact phrase (word-sequence) lexical query.
///
/// ## Example
///
/// ```python
/// q = laurus.PhraseQuery("body", ["machine", "learning"])
/// ```
#[pyclass(name = "PhraseQuery")]
pub struct PyPhraseQuery {
    pub field: String,
    pub terms: Vec<String>,
}

#[pymethods]
impl PyPhraseQuery {
    #[new]
    pub fn new(field: String, terms: Vec<String>) -> Self {
        Self { field, terms }
    }
    fn __repr__(&self) -> String {
        format!(
            "PhraseQuery(field='{}', terms={:?})",
            self.field, self.terms
        )
    }
}

// ---------------------------------------------------------------------------
// FuzzyQuery
// ---------------------------------------------------------------------------

/// Approximate (typo-tolerant) lexical query.
///
/// ## Example
///
/// ```python
/// q = laurus.FuzzyQuery("body", "programing", max_edits=2)
/// ```
#[pyclass(name = "FuzzyQuery")]
pub struct PyFuzzyQuery {
    pub field: String,
    pub term: String,
    pub max_edits: u32,
}

#[pymethods]
impl PyFuzzyQuery {
    #[new]
    #[pyo3(signature = (field, term, *, max_edits=2))]
    pub fn new(field: String, term: String, max_edits: u32) -> Self {
        Self {
            field,
            term,
            max_edits,
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "FuzzyQuery(field='{}', term='{}', max_edits={})",
            self.field, self.term, self.max_edits
        )
    }
}

// ---------------------------------------------------------------------------
// WildcardQuery
// ---------------------------------------------------------------------------

/// Wildcard pattern lexical query (`*` = any sequence, `?` = any character).
///
/// ## Example
///
/// ```python
/// q = laurus.WildcardQuery("filename", "*.pdf")
/// ```
#[pyclass(name = "WildcardQuery")]
pub struct PyWildcardQuery {
    pub field: String,
    pub pattern: String,
}

#[pymethods]
impl PyWildcardQuery {
    #[new]
    pub fn new(field: String, pattern: String) -> Self {
        Self { field, pattern }
    }
    fn __repr__(&self) -> String {
        format!(
            "WildcardQuery(field='{}', pattern='{}')",
            self.field, self.pattern
        )
    }
}

// ---------------------------------------------------------------------------
// NumericRangeQuery
// ---------------------------------------------------------------------------

/// Numeric range filter query (integer or float).
///
/// The type is inferred from the Python type of `min`/`max`:
/// pass `int` values for integer range, `float` for float range.
///
/// ## Example
///
/// ```python
/// q = laurus.NumericRangeQuery("year", min=2020, max=2023)   # integer
/// q = laurus.NumericRangeQuery("price", min=40.0, max=60.0)  # float
/// ```
#[pyclass(name = "NumericRangeQuery")]
pub struct PyNumericRangeQuery {
    pub field: String,
    pub kind: NumericKind,
}

#[derive(Clone)]
pub enum NumericKind {
    Integer(Option<i64>, Option<i64>),
    Float(Option<f64>, Option<f64>),
}

#[pymethods]
impl PyNumericRangeQuery {
    #[new]
    #[pyo3(signature = (field, *, min=None, max=None))]
    pub fn new(
        field: String,
        min: Option<&Bound<PyAny>>,
        max: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        use pyo3::types::{PyFloat, PyInt};
        let is_float = min.map(|v| v.is_instance_of::<PyFloat>()).unwrap_or(false)
            || max.map(|v| v.is_instance_of::<PyFloat>()).unwrap_or(false);

        let kind = if is_float {
            NumericKind::Float(
                min.and_then(|v| v.extract::<f64>().ok()),
                max.and_then(|v| v.extract::<f64>().ok()),
            )
        } else {
            let is_int = min.map(|v| v.is_instance_of::<PyInt>()).unwrap_or(false)
                || max.map(|v| v.is_instance_of::<PyInt>()).unwrap_or(false);
            if !is_int && (min.is_some() || max.is_some()) {
                return Err(PyValueError::new_err(
                    "NumericRangeQuery: min/max must be int or float",
                ));
            }
            NumericKind::Integer(
                min.and_then(|v| v.extract::<i64>().ok()),
                max.and_then(|v| v.extract::<i64>().ok()),
            )
        };
        Ok(Self { field, kind })
    }

    fn __repr__(&self) -> String {
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

impl PyNumericRangeQuery {
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

/// Python wrapper for a DateTime range query (Issue #1179).
///
/// Both bounds are inclusive; pass `None` to leave a side open. A bound may
/// be a `str` in any form the query DSL accepts — RFC 3339
/// (`"2024-01-01T09:00:00+09:00"`, normalized to UTC), a naive
/// `"YYYY-MM-DDTHH:MM:SS[.fff]"` (interpreted as UTC), or a date
/// `"YYYY-MM-DD"` (midnight UTC) — or a `datetime.datetime` / `datetime.date`
/// object, converted through `isoformat()` like document ingestion does.
///
/// ```python
/// q = laurus.DateTimeRangeQuery("created_at", min="2024-01-01", max="2024-12-31")
/// q = laurus.DateTimeRangeQuery("created_at", min=datetime(2024, 1, 1, tzinfo=timezone.utc))
/// ```
#[pyclass(name = "DateTimeRangeQuery")]
pub struct PyDateTimeRangeQuery {
    pub field: String,
    pub min: Option<String>,
    pub max: Option<String>,
}

#[pymethods]
impl PyDateTimeRangeQuery {
    #[new]
    #[pyo3(signature = (field, *, min=None, max=None))]
    pub fn new(
        field: String,
        min: Option<&Bound<PyAny>>,
        max: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        let query = Self {
            field,
            min: min.map(datetime_literal).transpose()?,
            max: max.map(datetime_literal).transpose()?,
        };
        // Validate eagerly so a bad bound raises at construction, not at search.
        query
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(query)
    }

    fn __repr__(&self) -> String {
        format!(
            "DateTimeRangeQuery(field='{}', min={:?}, max={:?})",
            self.field, self.min, self.max
        )
    }
}

impl PyDateTimeRangeQuery {
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

/// A datetime bound as text: a `str` as given, or any object exposing
/// `isoformat()` (`datetime.datetime`, `datetime.date`).
fn datetime_literal(obj: &Bound<PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    if let Ok(iso) = obj.call_method0("isoformat")
        && let Ok(s) = iso.extract::<String>()
    {
        return Ok(s);
    }
    Err(PyValueError::new_err(format!(
        "DateTimeRangeQuery: min/max must be a str or a datetime, got {}",
        obj.get_type().name()?
    )))
}

// ---------------------------------------------------------------------------
// GeoDistanceQuery
// ---------------------------------------------------------------------------

/// Geographic distance (radius) search query.
///
/// ## Example
///
/// ```python
/// # Within 100 km of San Francisco
/// q = laurus.GeoDistanceQuery.within_radius("location", 37.77, -122.42, 100_000.0)
/// ```
#[pyclass(name = "GeoDistanceQuery")]
pub struct PyGeoDistanceQuery {
    pub field: String,
    pub lat: f64,
    pub lon: f64,
    pub distance_m: f64,
}

#[pymethods]
impl PyGeoDistanceQuery {
    /// Create a radius-based geo distance query.
    ///
    /// Args:
    ///     field: Geo field name.
    ///     lat: Center latitude.
    ///     lon: Center longitude.
    ///     distance_m: Maximum distance from the centre in meters.
    #[staticmethod]
    pub fn within_radius(field: String, lat: f64, lon: f64, distance_m: f64) -> Self {
        Self {
            field,
            lat,
            lon,
            distance_m,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GeoDistanceQuery.within_radius(field='{}', lat={}, lon={}, distance_m={})",
            self.field, self.lat, self.lon, self.distance_m
        )
    }
}

impl PyGeoDistanceQuery {
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
/// ```python
/// q = laurus.GeoBoundingBoxQuery.within_bounding_box(
///     "location", 33.0, -123.0, 48.0, -117.0,
/// )
/// ```
#[pyclass(name = "GeoBoundingBoxQuery")]
pub struct PyGeoBoundingBoxQuery {
    pub field: String,
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
}

#[pymethods]
impl PyGeoBoundingBoxQuery {
    /// Create a bounding-box geo query.
    ///
    /// Args:
    ///     field: Geo field name.
    ///     min_lat: Southern boundary.
    ///     min_lon: Western boundary.
    ///     max_lat: Northern boundary.
    ///     max_lon: Eastern boundary.
    #[staticmethod]
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

    fn __repr__(&self) -> String {
        format!(
            "GeoBoundingBoxQuery.within_bounding_box(field='{}', min_lat={}, min_lon={}, max_lat={}, max_lon={})",
            self.field, self.min_lat, self.min_lon, self.max_lat, self.max_lon
        )
    }
}

impl PyGeoBoundingBoxQuery {
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
/// ```python
/// q = laurus.Geo3dDistanceQuery.within_sphere(
///     "position", -3955182.0, 3350553.0, 3700276.0, 5000.0,
/// )
/// ```
#[pyclass(name = "Geo3dDistanceQuery")]
pub struct PyGeo3dDistanceQuery {
    pub field: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub distance_m: f64,
}

#[pymethods]
impl PyGeo3dDistanceQuery {
    /// Create a 3D distance (sphere) query.
    ///
    /// Args:
    ///     field: Geo3d field name.
    ///     x, y, z: Centre coordinates in ECEF meters.
    ///     distance_m: Maximum distance from the centre in meters
    ///         (i.e. the search sphere's radius).
    #[staticmethod]
    pub fn within_sphere(field: String, x: f64, y: f64, z: f64, distance_m: f64) -> Self {
        Self {
            field,
            x,
            y,
            z,
            distance_m,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Geo3dDistanceQuery.within_sphere(field='{}', x={}, y={}, z={}, distance_m={})",
            self.field, self.x, self.y, self.z, self.distance_m
        )
    }
}

impl PyGeo3dDistanceQuery {
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
/// ```python
/// q = laurus.Geo3dBoundingBoxQuery.within_box(
///     "position",
///     -3962000.0, 3340000.0, 3690000.0,
///     -3958000.0, 3360000.0, 3710000.0,
/// )
/// ```
#[pyclass(name = "Geo3dBoundingBoxQuery")]
pub struct PyGeo3dBoundingBoxQuery {
    pub field: String,
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

#[pymethods]
impl PyGeo3dBoundingBoxQuery {
    /// Create a 3D bounding-box query.
    ///
    /// Args:
    ///     field: Geo3d field name.
    ///     min_x, min_y, min_z: Lower corner of the box.
    ///     max_x, max_y, max_z: Upper corner of the box.
    #[staticmethod]
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

    fn __repr__(&self) -> String {
        format!(
            "Geo3dBoundingBoxQuery.within_box(field='{}', min=({}, {}, {}), max=({}, {}, {}))",
            self.field, self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z
        )
    }
}

impl PyGeo3dBoundingBoxQuery {
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
/// ```python
/// q = laurus.Geo3dNearestQuery.k_nearest(
///     "position", -3955182.0, 3350553.0, 3700276.0, 10,
/// )
/// ```
#[pyclass(name = "Geo3dNearestQuery")]
pub struct PyGeo3dNearestQuery {
    pub field: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub k: usize,
    pub initial_radius_m: Option<f64>,
    pub max_radius_m: Option<f64>,
}

#[pymethods]
impl PyGeo3dNearestQuery {
    /// Create a 3D k-NN query.
    ///
    /// Args:
    ///     field: Geo3d field name.
    ///     x, y, z: Centre coordinates in ECEF meters.
    ///     k: Number of nearest neighbours to return.
    ///     initial_radius_m: Starting radius for the expanding-radius
    ///         search (default 1000 m). Optional.
    ///     max_radius_m: Hard cap on the search radius (default 1e10 m).
    ///         Optional.
    #[staticmethod]
    #[pyo3(signature = (field, x, y, z, k, *, initial_radius_m=None, max_radius_m=None))]
    pub fn k_nearest(
        field: String,
        x: f64,
        y: f64,
        z: f64,
        k: usize,
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

    fn __repr__(&self) -> String {
        format!(
            "Geo3dNearestQuery.k_nearest(field='{}', x={}, y={}, z={}, k={})",
            self.field, self.x, self.y, self.z, self.k
        )
    }
}

impl PyGeo3dNearestQuery {
    pub fn build(&self) -> Box<dyn laurus::lexical::Query> {
        let centre = GeoEcefPoint::new(self.x, self.y, self.z);
        let mut q = Geo3dNearestQuery::new(&self.field, centre, self.k);
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
/// ```python
/// bq = laurus.BooleanQuery()
/// bq.must(laurus.TermQuery("body", "programming"))
/// bq.must_not(laurus.TermQuery("body", "python"))
/// bq.should(laurus.TermQuery("category", "data-science"))
/// results = index.search(bq, limit=5)
/// ```
#[pyclass(name = "BooleanQuery")]
pub struct PyBooleanQuery {
    pub musts: Vec<Py<PyAny>>,
    pub shoulds: Vec<Py<PyAny>>,
    pub must_nots: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyBooleanQuery {
    #[new]
    pub fn new() -> Self {
        Self {
            musts: Vec::new(),
            shoulds: Vec::new(),
            must_nots: Vec::new(),
        }
    }

    /// Add a MUST (required) clause.
    pub fn must(&mut self, query: Py<PyAny>) {
        self.musts.push(query);
    }

    /// Add a SHOULD (optional, boosts score) clause.
    pub fn should(&mut self, query: Py<PyAny>) {
        self.shoulds.push(query);
    }

    /// Add a MUST_NOT (exclusion) clause.
    pub fn must_not(&mut self, query: Py<PyAny>) {
        self.must_nots.push(query);
    }

    fn __repr__(&self) -> String {
        format!(
            "BooleanQuery(musts={}, shoulds={}, must_nots={})",
            self.musts.len(),
            self.shoulds.len(),
            self.must_nots.len()
        )
    }
}

impl PyBooleanQuery {
    /// Build the underlying Rust [`BooleanQuery`].
    pub fn build_query(&self, py: Python) -> PyResult<Box<dyn laurus::lexical::Query>> {
        let mut bq = BooleanQuery::new();
        for obj in &self.musts {
            let bound: &Bound<'_, PyAny> = obj.bind(py);
            bq.add_must(extract_lexical_query(py, bound)?);
        }
        for obj in &self.shoulds {
            let bound: &Bound<'_, PyAny> = obj.bind(py);
            bq.add_should(extract_lexical_query(py, bound)?);
        }
        for obj in &self.must_nots {
            let bound: &Bound<'_, PyAny> = obj.bind(py);
            bq.add_must_not(extract_lexical_query(py, bound)?);
        }
        Ok(Box::new(bq))
    }
}

// ---------------------------------------------------------------------------
// SpanQuery
// ---------------------------------------------------------------------------

/// Positional / proximity span query.
///
/// Use the static factory methods to construct span queries, which can be
/// nested to build complex positional expressions.
///
/// ## Example
///
/// ```python
/// # SpanNear: "quick" within 1 position of "fox", in order
/// q = laurus.SpanQuery.near("body", ["quick", "fox"], slop=1, ordered=True)
///
/// # SpanContaining: span containing another span
/// big   = laurus.SpanQuery.near("body", ["quick", "fox"], slop=1, ordered=True)
/// small = laurus.SpanQuery.term("body", "brown")
/// q     = laurus.SpanQuery.containing("body", big, small)
/// ```
#[pyclass(name = "SpanQuery")]
pub struct PySpanQuery {
    pub field: String,
    pub kind: SpanKind,
}

#[pymethods]
impl PySpanQuery {
    /// Single-term span query.
    #[staticmethod]
    pub fn term(field: String, term: String) -> Self {
        Self {
            field,
            kind: SpanKind::Term(term),
        }
    }

    /// SpanNear: terms appearing within `slop` positions of each other.
    ///
    /// Args:
    ///     field: Field to search.
    ///     terms: List of term strings (each becomes a SpanTermQuery).
    ///     slop: Maximum token distance between terms.
    ///     ordered: Whether terms must appear in the given order.
    #[staticmethod]
    #[pyo3(signature = (field, terms, *, slop=0, ordered=true))]
    pub fn near(field: String, terms: Vec<String>, slop: u32, ordered: bool) -> Self {
        let kinds = terms.into_iter().map(SpanKind::Term).collect();
        Self {
            field,
            kind: SpanKind::Near(kinds, slop, ordered),
        }
    }

    /// SpanNear with nested SpanQuery clauses instead of plain terms.
    #[staticmethod]
    #[pyo3(signature = (field, clauses, *, slop=0, ordered=true))]
    pub fn near_spans(
        field: String,
        clauses: Vec<PyRef<PySpanQuery>>,
        slop: u32,
        ordered: bool,
    ) -> Self {
        let kinds: Vec<SpanKind> = clauses.iter().map(|c| c.kind.clone()).collect();
        Self {
            field,
            kind: SpanKind::Near(kinds, slop, ordered),
        }
    }

    /// SpanContaining: a span that contains another span.
    #[staticmethod]
    pub fn containing(field: String, big: PyRef<PySpanQuery>, little: PyRef<PySpanQuery>) -> Self {
        Self {
            field,
            kind: SpanKind::Containing(Box::new(big.kind.clone()), Box::new(little.kind.clone())),
        }
    }

    /// SpanWithin: a span included within another span, at a maximum distance.
    #[staticmethod]
    pub fn within(
        field: String,
        include: PyRef<PySpanQuery>,
        exclude: PyRef<PySpanQuery>,
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

    fn __repr__(&self) -> String {
        format!("SpanQuery(field='{}')", self.field)
    }
}

// ---------------------------------------------------------------------------
// VectorQuery  (pre-computed vector)
// ---------------------------------------------------------------------------

/// Vector search query using a pre-computed embedding vector.
///
/// Use this when you compute embeddings on the Python side (e.g. with
/// `sentence-transformers`) and want to pass the vector directly.
///
/// ## Example
///
/// ```python
/// from sentence_transformers import SentenceTransformer
/// model = SentenceTransformer("all-MiniLM-L6-v2")
/// vec = model.encode("memory safety").tolist()
///
/// results = index.search(laurus.VectorQuery("text_vec", vec), limit=5)
/// ```
#[pyclass(name = "VectorQuery")]
pub struct PyVectorQuery {
    pub field: String,
    pub vector: Vec<f32>,
}

#[pymethods]
impl PyVectorQuery {
    #[new]
    pub fn new(field: String, vector: Vec<f32>) -> Self {
        Self { field, vector }
    }
    fn __repr__(&self) -> String {
        format!(
            "VectorQuery(field='{}', dims={})",
            self.field,
            self.vector.len()
        )
    }
}

// ---------------------------------------------------------------------------
// VectorTextQuery  (text → Rust Embedder → vector)
// ---------------------------------------------------------------------------

/// Vector search query where the text is embedded by the Rust-side Embedder.
///
/// Use this when the `Index` was created with a built-in embedder (e.g.
/// `OpenAIEmbedder` or `CandleBertEmbedder`) so that Laurus converts the
/// query text into a vector automatically at search time.
///
/// ## Example
///
/// ```python
/// results = index.search(laurus.VectorTextQuery("text_vec", "memory safety"), limit=3)
/// ```
#[pyclass(name = "VectorTextQuery")]
pub struct PyVectorTextQuery {
    pub field: String,
    pub text: String,
}

#[pymethods]
impl PyVectorTextQuery {
    #[new]
    pub fn new(field: String, text: String) -> Self {
        Self { field, text }
    }
    fn __repr__(&self) -> String {
        format!(
            "VectorTextQuery(field='{}', text='{}')",
            self.field, self.text
        )
    }
}
