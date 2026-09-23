//! WASM wrappers for all Laurus query types.
//!
//! Each query struct stores the data needed to construct the Rust query.

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
use wasm_bindgen::JsValue;

// ---------------------------------------------------------------------------
// Internal enum types (not exported to JS)
// ---------------------------------------------------------------------------

/// Enum wrapping all supported lexical query types.
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
// Helpers
// ---------------------------------------------------------------------------

/// Extract a Laurus lexical query from a [`JsQuery`] enum variant.
pub fn extract_lexical_query(query: &JsQuery) -> Result<Box<dyn laurus::lexical::Query>, JsValue> {
    match query {
        JsQuery::TermQuery(q) => Ok(Box::new(TermQuery::new(&q.field, &q.term))),
        JsQuery::PhraseQuery(q) => Ok(Box::new(PhraseQuery::new(&q.field, q.terms.clone()))),
        JsQuery::FuzzyQuery(q) => Ok(Box::new(
            FuzzyQuery::new(&q.field, &q.term).max_edits(q.max_edits),
        )),
        JsQuery::WildcardQuery(q) => Ok(Box::new(
            WildcardQuery::new(&q.field, &q.pattern)
                .map_err(|e| JsValue::from_str(&e.to_string()))?,
        )),
        JsQuery::NumericRangeQuery(q) => Ok(q.build()),
        JsQuery::DateTimeRangeQuery(q) => q.build().map_err(|e| JsValue::from_str(&e.to_string())),
        JsQuery::GeoDistanceQuery(q) => q.build().map_err(|e| JsValue::from_str(&e.to_string())),
        JsQuery::GeoBoundingBoxQuery(q) => q.build().map_err(|e| JsValue::from_str(&e.to_string())),
        JsQuery::Geo3dDistanceQuery(q) => Ok(q.build()),
        JsQuery::Geo3dBoundingBoxQuery(q) => {
            q.build().map_err(|e| JsValue::from_str(&e.to_string()))
        }
        JsQuery::Geo3dNearestQuery(q) => Ok(q.build()),
        JsQuery::BooleanQuery(q) => q.build_query(),
        JsQuery::SpanQuery(q) => Ok(Box::new(SpanQueryWrapper::new(q.kind.build(&q.field)))),
    }
}

/// Convert a [`JsQuery`] into a [`LexicalSearchQuery`].
pub fn query_to_lexical_search_query(query: &JsQuery) -> Result<LexicalSearchQuery, JsValue> {
    Ok(LexicalSearchQuery::Obj(extract_lexical_query(query)?))
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
// Span-query recipe enum (Clone so it can be nested)
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
// Query data structs (used internally by search.rs and index.rs)
// ---------------------------------------------------------------------------

pub struct JsTermQuery {
    pub field: String,
    pub term: String,
}

pub struct JsPhraseQuery {
    pub field: String,
    pub terms: Vec<String>,
}

pub struct JsFuzzyQuery {
    pub field: String,
    pub term: String,
    pub max_edits: u32,
}

pub struct JsWildcardQuery {
    pub field: String,
    pub pattern: String,
}

pub struct JsNumericRangeQuery {
    pub field: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub is_float: bool,
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

/// 2D Geo distance (radius) query (internal).
///
/// Construction is internal — JS callers reach 2D Geo through the DSL
/// (`field:geo_distance(...)`) since `JsGeoDistanceQuery` is not exposed
/// as a JS class.
pub struct JsGeoDistanceQuery {
    pub field: String,
    pub lat: f64,
    pub lon: f64,
    pub distance_m: f64,
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

/// 2D Geo bounding-box query (internal).
///
/// Construction is internal — JS callers reach 2D Geo through the DSL
/// (`field:geo_bbox(...)`) since `JsGeoBoundingBoxQuery` is not exposed
/// as a JS class.
pub struct JsGeoBoundingBoxQuery {
    pub field: String,
    pub min_lat: f64,
    pub min_lon: f64,
    pub max_lat: f64,
    pub max_lon: f64,
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

/// 3D ECEF sphere query (internal).
///
/// Constructed by `Index::searchGeo3dDistance`. JS callers cannot
/// instantiate this directly; the method-on-Index pattern matches the
/// existing 2D `JsGeoDistanceQuery` and `JsTermQuery` flows.
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

/// 3D ECEF axis-aligned bounding-box query (internal).
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

/// 3D ECEF k-nearest-neighbours query (internal).
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

pub struct JsBooleanQuery {
    pub musts: Vec<JsQuery>,
    pub shoulds: Vec<JsQuery>,
    pub must_nots: Vec<JsQuery>,
}

impl JsBooleanQuery {
    pub fn build_query(&self) -> Result<Box<dyn laurus::lexical::Query>, JsValue> {
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

pub struct JsSpanQuery {
    pub field: String,
    pub kind: SpanKind,
}

pub struct JsVectorQueryInner {
    pub field: String,
    pub vector: Vec<f32>,
}

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
