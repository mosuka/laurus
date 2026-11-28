//! The queries the WASM search methods build from their arguments.
//!
//! JavaScript has no query classes: `Index.search` takes the query DSL, and
//! the other search methods (`searchTerm`, `searchDateTimeRange`, the
//! polygon and polyline searches, the Geo3d searches, `searchVector` and
//! `searchVectorText`) build one of these.

use laurus::GeoEcefPoint;
use laurus::lexical::{
    DateTimeRangeQuery, Geo3dBoundingBoxQuery, Geo3dDistanceQuery, Geo3dNearestQuery,
    Geo3dPolyline, Geo3dPolylineQuery, GeoPolygonQuery, GeoPolylineQuery, TermQuery,
};
use laurus::vector::Vector;
use laurus::vector::store::request::QueryVector;
use laurus::{DataValue, QueryPayload, VectorSearchQuery};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use wasm_bindgen::JsValue;

use crate::errors::laurus_err;

// ---------------------------------------------------------------------------
// Internal enum types (not exported to JS)
// ---------------------------------------------------------------------------

/// The lexical query a search method builds.
pub enum JsQuery {
    TermQuery(JsTermQuery),
    DateTimeRangeQuery(JsDateTimeRangeQuery),
    Geo3dDistanceQuery(JsGeo3dDistanceQuery),
    Geo3dBoundingBoxQuery(JsGeo3dBoundingBoxQuery),
    Geo3dNearestQuery(JsGeo3dNearestQuery),
    GeoPolygonQuery(GeoPolygonQuery),
    GeoPolylineQuery(GeoPolylineQuery),
    Geo3dPolylineQuery(Geo3dPolylineQuery),
}

/// The vector query a search method builds.
pub enum JsVectorQuery {
    VectorQuery(JsVectorQueryInner),
    VectorTextQuery(JsVectorTextQuery),
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a Laurus lexical query from a [`JsQuery`] enum variant.
pub fn extract_lexical_query(query: &JsQuery) -> Result<Box<dyn laurus::lexical::Query>, JsValue> {
    match query {
        JsQuery::TermQuery(q) => Ok(Box::new(TermQuery::new(&q.field, &q.term))),
        JsQuery::DateTimeRangeQuery(q) => q.build().map_err(|e| JsValue::from_str(&e.to_string())),
        JsQuery::Geo3dDistanceQuery(q) => Ok(q.build()),
        JsQuery::Geo3dBoundingBoxQuery(q) => {
            q.build().map_err(|e| JsValue::from_str(&e.to_string()))
        }
        JsQuery::Geo3dNearestQuery(q) => Ok(q.build()),
        JsQuery::GeoPolygonQuery(q) => Ok(Box::new(q.clone())),
        JsQuery::GeoPolylineQuery(q) => Ok(Box::new(q.clone())),
        JsQuery::Geo3dPolylineQuery(q) => Ok(Box::new(q.clone())),
    }
}

// ---------------------------------------------------------------------------
// Vertices of a polygon or a polyline (Issue #1596)
// ---------------------------------------------------------------------------

/// A vertex of a `Geo` polygon or polyline: `{ lat, lon }`, as a point in
/// a document is written. Other keys are not read.
#[derive(Deserialize)]
struct GeoVertex {
    lat: f64,
    lon: f64,
}

/// A vertex of a `Geo3d` polyline: `{ x, y, z }` in ECEF meters.
#[derive(Deserialize)]
struct Geo3dVertex {
    x: f64,
    y: f64,
    z: f64,
}

/// Read `value` as `T`, naming `what` of `context` in the error.
fn vertices_from<T: DeserializeOwned>(
    context: &str,
    what: &str,
    shape: &str,
    value: JsValue,
) -> Result<T, JsValue> {
    serde_wasm_bindgen::from_value(value)
        .map_err(|e| JsValue::from_str(&format!("{context}: {what} must be {shape}: {e}")))
}

/// The `(lat, lon)` pairs of `vertices`, an array of `{ lat, lon }`
/// objects.
fn lat_lon_vertices(
    context: &str,
    what: &str,
    vertices: JsValue,
) -> Result<Vec<(f64, f64)>, JsValue> {
    let vertices: Vec<GeoVertex> =
        vertices_from(context, what, "an array of { lat, lon } objects", vertices)?;
    Ok(vertices.into_iter().map(|v| (v.lat, v.lon)).collect())
}

/// The polygon query `Index.searchGeoPolygon` builds: `exterior` is an
/// array of `{ lat, lon }` objects, and `holes`, unless it is missing,
/// `null` or `undefined`, an array of such arrays.
pub fn geo_polygon_query(
    context: &str,
    field: String,
    exterior: JsValue,
    holes: Option<JsValue>,
) -> Result<GeoPolygonQuery, JsValue> {
    let exterior = lat_lon_vertices(context, "exterior", exterior)?;
    let holes: Vec<Vec<(f64, f64)>> = match holes {
        Some(holes) if !holes.is_null() && !holes.is_undefined() => {
            let holes: Vec<Vec<GeoVertex>> = vertices_from(
                context,
                "holes",
                "an array of arrays of { lat, lon } objects",
                holes,
            )?;
            holes
                .into_iter()
                .map(|hole| hole.into_iter().map(|v| (v.lat, v.lon)).collect())
                .collect()
        }
        _ => Vec::new(),
    };
    let holes: Vec<&[(f64, f64)]> = holes.iter().map(Vec::as_slice).collect();
    GeoPolygonQuery::within_polygon(field, &exterior, &holes).map_err(laurus_err)
}

/// The polyline query `Index.searchGeoPolyline` builds: `vertices` is an
/// array of `{ lat, lon }` objects.
pub fn geo_polyline_query(
    context: &str,
    field: String,
    vertices: JsValue,
    distance_m: f64,
) -> Result<GeoPolylineQuery, JsValue> {
    let vertices = lat_lon_vertices(context, "vertices", vertices)?;
    GeoPolylineQuery::within_distance(field, &vertices, distance_m).map_err(laurus_err)
}

/// The polyline query `Index.searchGeo3dPolyline` builds: `vertices` is an
/// array of `{ x, y, z }` objects in ECEF meters.
pub fn geo3d_polyline_query(
    context: &str,
    field: String,
    vertices: JsValue,
    distance_m: f64,
) -> Result<Geo3dPolylineQuery, JsValue> {
    let vertices: Vec<Geo3dVertex> = vertices_from(
        context,
        "vertices",
        "an array of { x, y, z } objects",
        vertices,
    )?;
    let polyline = Geo3dPolyline::new(
        vertices
            .into_iter()
            .map(|v| GeoEcefPoint::new(v.x, v.y, v.z))
            .collect(),
    )
    .map_err(laurus_err)?;
    Ok(Geo3dPolylineQuery::new(field, polyline, distance_m))
}

/// Convert a [`JsVectorQuery`] into a [`VectorSearchQuery`].
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
// Query data structs (used internally by search.rs and index.rs)
// ---------------------------------------------------------------------------

/// Term query (internal; built by `Index.searchTerm`).
pub struct JsTermQuery {
    pub field: String,
    pub term: String,
}

/// DateTime range query (internal; reached through `Index.searchDateTimeRange`,
/// Issue #1179). Bounds are inclusive datetime literals in any form the
/// query DSL accepts (RFC 3339, naive `YYYY-MM-DDTHH:MM:SS[.fff]` as UTC,
/// or `YYYY-MM-DD` as midnight UTC); `None` leaves a side open.
pub struct JsDateTimeRangeQuery {
    pub field: String,
    pub min: Option<String>,
    pub max: Option<String>,
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

/// 3D ECEF sphere query (internal; built by `Index.searchGeo3dDistance`).
pub struct JsGeo3dDistanceQuery {
    pub field: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub distance_m: f64,
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

/// 3D ECEF axis-aligned bounding-box query (internal; built by
/// `Index.searchGeo3dBoundingBox`).
pub struct JsGeo3dBoundingBoxQuery {
    pub field: String,
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
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

/// 3D ECEF k-nearest-neighbours query (internal; built by
/// `Index.searchGeo3dNearest`).
pub struct JsGeo3dNearestQuery {
    pub field: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub k: u32,
    pub initial_radius_m: Option<f64>,
    pub max_radius_m: Option<f64>,
}

impl JsGeo3dNearestQuery {
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

/// Vector query with a precomputed vector (internal; built by
/// `Index.searchVector`).
pub struct JsVectorQueryInner {
    pub field: String,
    pub vector: Vec<f32>,
}

/// Vector query with text to embed (internal; built by
/// `Index.searchVectorText`).
pub struct JsVectorTextQuery {
    pub field: String,
    pub text: String,
}

#[cfg(test)]
mod datetime_range_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    /// Issue #1179: literals in every accepted form build a query; a bad
    /// bound surfaces as an error rather than a silent empty result.
    #[wasm_bindgen_test]
    fn datetime_range_builds_from_literals() {
        for (min, max) in [
            (Some("2024-01-01"), Some("2024-12-31")),
            (Some("2024-01-01T09:00:00+09:00"), None),
            (None, Some("2024-06-15T12:34:56.5")),
        ] {
            let query = JsDateTimeRangeQuery {
                field: "created_at".to_string(),
                min: min.map(str::to_string),
                max: max.map(str::to_string),
            };
            assert!(query.build().is_ok(), "{min:?}..{max:?}");
        }
        let bad = JsDateTimeRangeQuery {
            field: "created_at".to_string(),
            min: Some("yesterday".to_string()),
            max: None,
        };
        assert!(bad.build().is_err());
        assert!(
            extract_lexical_query(&JsQuery::DateTimeRangeQuery(bad)).is_err(),
            "the dispatcher must surface the error"
        );
    }
}
