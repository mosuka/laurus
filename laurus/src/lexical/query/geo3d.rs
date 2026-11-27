//! 3D geographical search functionality (ECEF Cartesian queries).
//!
//! This module is the 3D counterpart of [`super::geo`]: where the 2D module
//! reasons about WGS84 latitude / longitude pairs and uses Haversine
//! distance, the 3D module reasons about Earth-Centered Earth-Fixed (ECEF)
//! Cartesian points and uses straight-line Euclidean distance. ECEF is
//! the right primitive for true 3D queries (drone proximity, satellite
//! tracking, indoor 3D positioning) where altitude is a first-class
//! dimension and pole-wrap or date-line concerns disappear.
//!
//! # Implemented queries
//!
//! - [`Geo3dDistanceQuery`] — sphere-radius search: every doc whose stored
//!   ECEF point lies within `distance_m` meters of the query center.
//! - [`Geo3dBoundingBoxQuery`] — 3D axis-aligned box search: every doc
//!   whose stored ECEF point falls inside the closed `[min, max]` box.
//! - [`Geo3dNearestQuery`] — k-nearest-neighbour search: the `k` docs
//!   whose stored ECEF points lie closest to the query center, returned
//!   in distance-ascending order. Built on an expanding-radius loop on
//!   top of the same BKD `intersect` primitive.
//!
//! # How sphere queries reach the BKD
//!
//! [`Geo3dDistanceQuery::find_matches`] grabs the per-field
//! [`BKDTree`](crate::lexical::index::structures::bkd_tree::BKDTree) from
//! the reader and runs `tree.intersect(&mut SphereVisitor)`. The visitor's
//! [`compare`](IntersectVisitor::compare) implementation classifies each
//! cell as Inside / Outside / Crosses by comparing the cell's nearest /
//! farthest squared distance to the sphere's center against `radius²`,
//! enabling the BKD reader to skip whole subtrees that cannot intersect
//! the sphere.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::data::GeoEcefPoint;
use crate::error::Result;
use crate::lexical::index::structures::aabb::AABB;
use crate::lexical::index::structures::visitor::{CellRelation, IntersectVisitor};
use crate::lexical::query::Query;
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::scorer::Scorer;
use crate::lexical::reader::LexicalIndexReader;

/// A 3D distance (sphere) query against an ECEF-typed field.
///
/// Matches every document whose stored [`GeoEcefPoint`] lies within
/// `distance_m` meters of `center`. Distance is straight-line Euclidean in
/// ECEF space; a `distance_m` of 1000 means "within 1 km of `center`",
/// regardless of latitude or altitude.
///
/// Scoring follows the same shape as the 2D
/// [`GeoDistanceQuery`](super::geo::GeoDistanceQuery): score is
/// `1.0 - distance / radius`, clamped to `[0, 1]`, computed from every
/// matched point's real coordinates, so the result order and the scores
/// do not depend on how the BKD tree happens to be split (#1594).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geo3dDistanceQuery {
    /// Field containing 3D ECEF coordinates.
    field: String,
    /// Center point of the search sphere.
    center: GeoEcefPoint,
    /// Maximum distance from `center` in meters (sphere radius).
    distance_m: f64,
    /// Boost factor applied to the final score.
    boost: f32,
}

impl Geo3dDistanceQuery {
    /// Construct a new query.
    ///
    /// `distance_m` is interpreted as straight-line distance in ECEF meters
    /// and represents the maximum distance from `center` (i.e. the search
    /// sphere's radius). Negative or zero values produce a query that matches
    /// no documents (see [`Geo3dDistanceQuery::is_empty`]).
    pub fn new<F: Into<String>>(field: F, center: GeoEcefPoint, distance_m: f64) -> Self {
        Self {
            field: field.into(),
            center,
            distance_m,
            boost: 1.0,
        }
    }

    /// Set the boost factor applied to every match's score.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Field name this query targets.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Search center.
    pub fn center(&self) -> GeoEcefPoint {
        self.center
    }

    /// Maximum distance from `center` in meters (sphere radius).
    pub fn distance_m(&self) -> f64 {
        self.distance_m
    }

    /// Run the query against `reader` and return the matches sorted by
    /// distance ascending (closest first).
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<Geo3dMatch>> {
        let Some(bkd) = reader.get_bkd_tree(&self.field)? else {
            return Ok(Vec::new());
        };

        let mut visitor = SphereVisitor::new(self.center, self.distance_m);
        bkd.intersect(&mut visitor)?;
        Ok(scored_matches(visitor.hits, self.distance_m))
    }
}

/// The matches of `hits` within `radius`: each scored `1 - distance /
/// radius`, clamped to `[0, 1]` (`0` for a radius of 0 or less), the
/// closest one per document only, sorted by distance ascending. Shared by
/// [`Geo3dDistanceQuery`] and [`Geo3dPolylineQuery`].
fn scored_matches(hits: Vec<SphereHit>, radius: f64) -> Vec<Geo3dMatch> {
    let mut matches: Vec<Geo3dMatch> = hits
        .into_iter()
        .map(|hit| {
            let distance = hit.distance_sq.sqrt();
            let score = if radius <= 0.0 {
                0.0
            } else {
                (1.0 - distance / radius).clamp(0.0, 1.0) as f32
            };
            Geo3dMatch {
                doc_id: hit.doc_id,
                distance_m: distance,
                score,
            }
        })
        .collect();

    // Multi-segment readers and multi-valued fields can produce duplicates;
    // keep the closest.
    matches.sort_by(|a, b| {
        a.doc_id
            .cmp(&b.doc_id)
            .then_with(|| a.distance_m.total_cmp(&b.distance_m))
    });
    matches.dedup_by_key(|m| m.doc_id);
    // Final order: distance ascending.
    matches.sort_by(|a, b| a.distance_m.total_cmp(&b.distance_m));
    matches
}

/// A document matched by a [`Geo3dDistanceQuery`], with its distance to
/// the query center and a `[0, 1]` relevance score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geo3dMatch {
    /// Document id.
    pub doc_id: u64,
    /// Straight-line distance to the query center in meters.
    pub distance_m: f64,
    /// Final relevance score in `[0, 1]`, before the query's boost is
    /// applied by the [`Scorer`].
    pub score: f32,
}

/// `IntersectVisitor` that collects every doc whose stored point lies
/// within a sphere of radius `distance_m` centered at `center`, with its
/// squared distance to the center.
///
/// `compare` prunes cells with [`AABB::min_distance_sq_to_point`] in
/// squared-distance space (no `sqrt`) and never returns
/// [`CellRelation::Inside`]: the `Inside` path reports doc ids via
/// `visit_inside` **without** coordinates, and the query scores and
/// ranks by distance. Before #1594 a cell whose farthest corner lay
/// within the sphere was reported `Inside` and every point under it got
/// distance `0.0` and score `1.0`, ranking it above closer points and
/// making a document's score depend on the segment's BKD layout.
/// Returning `Crosses` for a contained cell costs decoding its points,
/// never correctness (same pattern as the 2D `GeoBoxPointsVisitor` and
/// [`NearestVisitor`]).
struct SphereVisitor {
    center: [f64; 3],
    radius_sq: f64,
    hits: Vec<SphereHit>,
}

#[derive(Debug, Clone, Copy)]
struct SphereHit {
    doc_id: u64,
    /// Squared distance from `center` to the doc's stored point.
    distance_sq: f64,
}

impl SphereVisitor {
    fn new(center: GeoEcefPoint, radius: f64) -> Self {
        let radius_clamped = radius.max(0.0);
        Self {
            center: [center.x, center.y, center.z],
            radius_sq: radius_clamped * radius_clamped,
            hits: Vec::new(),
        }
    }
}

impl IntersectVisitor for SphereVisitor {
    fn compare(&self, cell: &AABB) -> CellRelation {
        debug_assert_eq!(cell.num_dims(), 3, "SphereVisitor expects a 3D BKD");
        let min_d_sq = cell.min_distance_sq_to_point(&self.center);
        if min_d_sq > self.radius_sq {
            return CellRelation::Outside;
        }
        // Always Crosses, even for a cell entirely within the sphere:
        // the query needs the per-point distance the Inside path
        // withholds (see the type-level note).
        CellRelation::Crosses
    }

    fn visit_inside(&mut self, _doc_id: u64) {
        // Unreachable in practice — `compare` never returns Inside — but
        // if a future BKD impl chooses to call this we ignore the hit (we
        // have no coordinates to score it with).
    }

    fn visit(&mut self, doc_id: u64, point: &[f64]) {
        debug_assert_eq!(point.len(), 3, "SphereVisitor expects a 3D BKD");
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let d_sq = dx * dx + dy * dy + dz * dz;
        if d_sq <= self.radius_sq {
            self.hits.push(SphereHit {
                doc_id,
                distance_sq: d_sq,
            });
        }
    }
}

/// Matcher for [`Geo3dDistanceQuery`]. Iterates over the matches in
/// distance-ascending order (mirrors [`super::geo::GeoMatcher`]).
#[derive(Debug)]
pub struct Geo3dMatcher {
    matches: Vec<Geo3dMatch>,
    cursor: usize,
}

impl Geo3dMatcher {
    /// Wrap a match list, re-sorting it into ascending document-id order.
    ///
    /// [`Geo3dDistanceQuery::find_matches`] produces distance-ascending
    /// lists for ranking, but `Matcher` iteration order is a trait
    /// contract — boolean conjunction / disjunction drivers interleave
    /// `skip_to` calls and assume monotonically increasing doc ids.
    /// Distance ranking stays with [`Geo3dScorer`], which looks scores up
    /// by doc id.
    pub fn new(mut matches: Vec<Geo3dMatch>) -> Self {
        matches.sort_by_key(|m| m.doc_id);
        Self { matches, cursor: 0 }
    }
}

impl Matcher for Geo3dMatcher {
    fn doc_id(&self) -> u64 {
        if self.cursor >= self.matches.len() {
            u64::MAX
        } else {
            self.matches[self.cursor].doc_id
        }
    }

    fn next(&mut self) -> Result<bool> {
        self.cursor += 1;
        if self.cursor < self.matches.len() {
            Ok(true)
        } else {
            self.cursor = self.matches.len();
            Ok(false)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        // Forward scan for the first doc id >= target; valid because
        // `Geo3dMatcher::new` sorts matches by ascending doc id.
        while self.cursor < self.matches.len() {
            if self.matches[self.cursor].doc_id >= target {
                return Ok(true);
            }
            self.cursor += 1;
        }
        Ok(false)
    }

    fn cost(&self) -> u64 {
        self.matches.len() as u64
    }

    fn is_exhausted(&self) -> bool {
        self.cursor >= self.matches.len()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Scorer for [`Geo3dDistanceQuery`]. Looks up the precomputed score
/// per doc id and applies the query's boost.
#[derive(Debug)]
pub struct Geo3dScorer {
    doc_scores: HashMap<u64, f32>,
    boost: f32,
}

impl Geo3dScorer {
    /// Build a scorer from the matches returned by
    /// [`Geo3dDistanceQuery::find_matches`].
    pub fn new(matches: Vec<Geo3dMatch>, boost: f32) -> Self {
        let mut doc_scores = HashMap::with_capacity(matches.len());
        for m in matches {
            doc_scores.insert(m.doc_id, m.score);
        }
        Self { doc_scores, boost }
    }
}

impl Scorer for Geo3dScorer {
    fn score(&self, doc_id: u64, _term_freq: f32, _field_length: Option<f32>) -> f32 {
        self.doc_scores.get(&doc_id).copied().unwrap_or(0.0) * self.boost
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        self.doc_scores.values().copied().fold(0.0_f32, f32::max) * self.boost
    }

    fn name(&self) -> &'static str {
        "Geo3dScorer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Query for Geo3dDistanceQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(Geo3dMatcher::new(self.find_matches(reader)?)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        Ok(Box::new(Geo3dScorer::new(
            self.find_matches(reader)?,
            self.boost,
        )))
    }

    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        // Run the BKD sphere traversal once and share it (#996);
        // ranking is identical to building matcher and scorer
        // independently.
        let matches = self.find_matches(reader)?;
        Ok((
            Box::new(Geo3dMatcher::new(matches.clone())),
            Box::new(Geo3dScorer::new(matches, self.boost)),
        ))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn description(&self) -> String {
        format!(
            "Geo3dDistanceQuery(field: {}, center: {:?}, distance: {}m)",
            self.field, self.center, self.distance_m
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.distance_m <= 0.0)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        // Same shape as 2D GeoDistanceQuery::cost — moderate.
        let doc_count = reader.doc_count();
        Ok(doc_count.saturating_mul(2))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field + center + radius determine the matched set; boost is
        // score-only and excluded.
        Some(format!(
            "geo3ddist|{:?}|{:?}|{:?}",
            self.field, self.center, self.distance_m
        ))
    }
}

/// How much farther than the query's distance a cell may be and still be
/// visited, in metres: rounding in ECEF coordinates of millions of metres is
/// about a nanometre, so no point at the distance is pruned.
const CELL_SLACK_M: f64 = 1e-6;

/// A polyline of ECEF points joined in order by straight segments, for
/// [`Geo3dPolylineQuery`] (Issue #1596).
///
/// The distance from a point to the polyline is the Euclidean distance to
/// its closest segment, as [`Geo3dDistanceQuery`]'s to its center. A segment
/// is a chord: between two points on the surface it runs below it, by about
/// `L² / 8R` midway for a segment of length `L` (196 m for 100 km, about
/// 20 km for 1,000 km). A single vertex is a point, and repeated vertices
/// are fine: a segment of length 0 is its point.
#[derive(Debug, Clone, PartialEq)]
pub struct Geo3dPolyline {
    /// The vertices as given.
    vertices: Vec<GeoEcefPoint>,
}

impl Geo3dPolyline {
    /// Create a polyline through `vertices`, in order.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`](crate::error::LaurusError)
    /// if there is no vertex, or a coordinate is NaN or infinite.
    pub fn new(vertices: Vec<GeoEcefPoint>) -> Result<Self> {
        if vertices.is_empty() {
            return Err(crate::error::LaurusError::invalid_argument(
                "a polyline needs at least one vertex".to_string(),
            ));
        }
        for (i, vertex) in vertices.iter().enumerate() {
            if !(vertex.x.is_finite() && vertex.y.is_finite() && vertex.z.is_finite()) {
                return Err(crate::error::LaurusError::invalid_argument(format!(
                    "vertex {i} of the polyline has a coordinate that is not finite: \
                     ({}, {}, {})",
                    vertex.x, vertex.y, vertex.z
                )));
            }
        }
        Ok(Geo3dPolyline { vertices })
    }

    /// The vertices, in order.
    pub fn vertices(&self) -> &[GeoEcefPoint] {
        &self.vertices
    }

    /// The Euclidean distance in metres from `point` to the polyline. For a
    /// single vertex it is the distance [`Geo3dDistanceQuery`] measures.
    pub fn distance_to(&self, point: &GeoEcefPoint) -> f64 {
        self.distance_sq_to(&[point.x, point.y, point.z]).sqrt()
    }

    /// The squared distance from `point` to the polyline.
    fn distance_sq_to(&self, point: &[f64]) -> f64 {
        if let [vertex] = self.vertices.as_slice() {
            return point_distance_sq(point, vertex);
        }
        self.vertices
            .windows(2)
            .map(|edge| segment_distance_sq(point, &edge[0], &edge[1]))
            .fold(f64::INFINITY, f64::min)
    }

    /// Whether a segment comes within `distance_m` of `cell`, by clipping
    /// it to the cell widened by `distance_m` (and [`CELL_SLACK_M`]) on
    /// every side. The widened box holds every point within the distance of
    /// the cell, so this never misses one; it only gives up pruning near
    /// the cell's corners, where the box reaches farther than the distance.
    fn meets_cell(&self, cell: &AABB, distance_m: f64) -> bool {
        let reach = distance_m + CELL_SLACK_M;
        let (min, max) = (cell.min(), cell.max());
        let low = [min[0] - reach, min[1] - reach, min[2] - reach];
        let high = [max[0] + reach, max[1] + reach, max[2] + reach];
        if let [vertex] = self.vertices.as_slice() {
            return segment_meets_box(vertex, vertex, &low, &high);
        }
        self.vertices
            .windows(2)
            .any(|edge| segment_meets_box(&edge[0], &edge[1], &low, &high))
    }
}

/// The squared distance from `point` to `vertex`, summed in the order of
/// `SphereVisitor::visit`, so that a single vertex measures exactly what
/// [`Geo3dDistanceQuery`] does.
fn point_distance_sq(point: &[f64], vertex: &GeoEcefPoint) -> f64 {
    let dx = point[0] - vertex.x;
    let dy = point[1] - vertex.y;
    let dz = point[2] - vertex.z;
    dx * dx + dy * dy + dz * dz
}

/// The squared distance from `point` to the segment from `a` to `b`.
fn segment_distance_sq(point: &[f64], a: &GeoEcefPoint, b: &GeoEcefPoint) -> f64 {
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let length_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if length_sq == 0.0 {
        return point_distance_sq(point, a);
    }
    let ap = [point[0] - a.x, point[1] - a.y, point[2] - a.z];
    let t = ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / length_sq).clamp(0.0, 1.0);
    let closest = GeoEcefPoint::new(a.x + t * ab[0], a.y + t * ab[1], a.z + t * ab[2]);
    point_distance_sq(point, &closest)
}

/// Whether the segment from `a` to `b` meets the closed box from `low` to
/// `high`, by clipping it to the box (Liang-Barsky); a segment of length 0
/// meets it when its point is inside.
fn segment_meets_box(a: &GeoEcefPoint, b: &GeoEcefPoint, low: &[f64; 3], high: &[f64; 3]) -> bool {
    let (start, end) = ([a.x, a.y, a.z], [b.x, b.y, b.z]);
    let (mut enter, mut leave) = (0.0_f64, 1.0_f64);
    for dim in 0..3 {
        let d = end[dim] - start[dim];
        for (p, q) in [(-d, start[dim] - low[dim]), (d, high[dim] - start[dim])] {
            if p == 0.0 {
                if q < 0.0 {
                    return false;
                }
            } else {
                let t = q / p;
                if p < 0.0 {
                    if t > leave {
                        return false;
                    }
                    enter = enter.max(t);
                } else {
                    if t < enter {
                        return false;
                    }
                    leave = leave.min(t);
                }
            }
        }
    }
    true
}

/// [`IntersectVisitor`] collecting the points within a distance of a
/// [`Geo3dPolyline`], with their squared distances. The score needs every
/// point's distance, so `compare` never returns [`CellRelation::Inside`],
/// as [`SphereVisitor`] does not.
struct Geo3dPolylineVisitor<'a> {
    polyline: &'a Geo3dPolyline,
    distance_m: f64,
    radius_sq: f64,
    hits: Vec<SphereHit>,
    /// How many points were tested one by one, for the pruning test.
    #[cfg(test)]
    visited: usize,
}

impl<'a> Geo3dPolylineVisitor<'a> {
    /// A visitor for the points within `radius` of `polyline`; a negative
    /// radius is 0, as in [`SphereVisitor`].
    fn new(polyline: &'a Geo3dPolyline, radius: f64) -> Self {
        let radius_clamped = radius.max(0.0);
        Geo3dPolylineVisitor {
            polyline,
            distance_m: radius_clamped,
            radius_sq: radius_clamped * radius_clamped,
            hits: Vec::new(),
            #[cfg(test)]
            visited: 0,
        }
    }
}

impl IntersectVisitor for Geo3dPolylineVisitor<'_> {
    fn compare(&self, cell: &AABB) -> CellRelation {
        debug_assert_eq!(cell.num_dims(), 3, "Geo3dPolylineVisitor expects a 3D BKD");
        if self.polyline.meets_cell(cell, self.distance_m) {
            // Never Inside: the score needs each point's distance.
            CellRelation::Crosses
        } else {
            CellRelation::Outside
        }
    }

    fn visit_inside(&mut self, _doc_id: u64) {
        // Unreachable: `compare` never returns Inside, and a doc id alone
        // has no distance to score by.
    }

    fn visit(&mut self, doc_id: u64, point: &[f64]) {
        debug_assert_eq!(point.len(), 3, "Geo3dPolylineVisitor expects a 3D BKD");
        #[cfg(test)]
        {
            self.visited += 1;
        }
        let distance_sq = self.polyline.distance_sq_to(point);
        if distance_sq <= self.radius_sq {
            self.hits.push(SphereHit {
                doc_id,
                distance_sq,
            });
        }
    }
}

/// A query for the documents with a point of a `Geo3d` field within a
/// distance of a [`Geo3dPolyline`] (Issue #1596): a flight path, a tunnel,
/// a cable.
///
/// The distance is Euclidean in ECEF, as [`Geo3dDistanceQuery`]'s, and the
/// score is the same: `1 - d / distance_m`, clamped to `[0, 1]`, `d` being
/// the distance of the document's closest point to the polyline. A single
/// vertex finds the documents, and gives the scores, that
/// [`Geo3dDistanceQuery`] does. Like the other 3D queries, it reads the
/// field's BKD only, so a field indexed with `indexed: false` matches
/// nothing.
///
/// # Example
///
/// ```rust
/// use laurus::lexical::query::{Geo3dPolyline, Geo3dPolylineQuery};
/// use laurus::util::ecef::wgs84_to_ecef;
///
/// // Within 500 m of a flight path at 1,000 m.
/// let path = Geo3dPolyline::new(vec![
///     wgs84_to_ecef(35.0, 139.0, 1_000.0),
///     wgs84_to_ecef(35.4, 139.6, 1_000.0),
/// ])
/// .unwrap();
/// let query = Geo3dPolylineQuery::new("position", path, 500.0);
/// ```
#[derive(Debug, Clone)]
pub struct Geo3dPolylineQuery {
    /// Field containing 3D ECEF coordinates.
    field: String,
    /// The polyline.
    polyline: Geo3dPolyline,
    /// The maximum distance from the polyline, in metres.
    distance_m: f64,
    /// Boost factor applied to the final score.
    boost: f32,
}

impl Geo3dPolylineQuery {
    /// Create a query for the points within `distance_m` metres of
    /// `polyline`. A distance of 0 or less is reported empty by
    /// [`is_empty`](Query::is_empty), as for [`Geo3dDistanceQuery`].
    pub fn new<F: Into<String>>(field: F, polyline: Geo3dPolyline, distance_m: f64) -> Self {
        Geo3dPolylineQuery {
            field: field.into(),
            polyline,
            distance_m,
            boost: 1.0,
        }
    }

    /// Set the boost factor applied to every match's score.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Field name this query targets.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// The polyline.
    pub fn polyline(&self) -> &Geo3dPolyline {
        &self.polyline
    }

    /// The maximum distance from the polyline, in metres.
    pub fn distance_m(&self) -> f64 {
        self.distance_m
    }

    /// Run the query against `reader` and return the matches, each document
    /// once with its closest point, sorted by distance ascending.
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<Geo3dMatch>> {
        let Some(bkd) = reader.get_bkd_tree(&self.field)? else {
            return Ok(Vec::new());
        };

        let mut visitor = Geo3dPolylineVisitor::new(&self.polyline, self.distance_m);
        bkd.intersect(&mut visitor)?;
        Ok(scored_matches(visitor.hits, self.distance_m))
    }
}

impl Query for Geo3dPolylineQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(Geo3dMatcher::new(self.find_matches(reader)?)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        Ok(Box::new(Geo3dScorer::new(
            self.find_matches(reader)?,
            self.boost,
        )))
    }

    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        // One traversal for both, as `Geo3dDistanceQuery` does (#996).
        let matches = self.find_matches(reader)?;
        Ok((
            Box::new(Geo3dMatcher::new(matches.clone())),
            Box::new(Geo3dScorer::new(matches, self.boost)),
        ))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn description(&self) -> String {
        format!(
            "Geo3dPolylineQuery(field: {}, vertices: {:?}, distance: {}m)",
            self.field, self.polyline.vertices, self.distance_m
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.distance_m <= 0.0)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        // The same shape as `Geo3dDistanceQuery::cost`.
        Ok(reader.doc_count().saturating_mul(2))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field, vertices and distance determine the matched set; the boost
        // is score-only and excluded.
        Some(format!(
            "geo3dpolyline|{:?}|{:?}|{:?}",
            self.field, self.polyline.vertices, self.distance_m
        ))
    }
}

/// A 3D bounding-box query against an ECEF-typed field.
///
/// Matches every document whose stored [`GeoEcefPoint`] falls inside the
/// closed axis-aligned box `[min, max]` (inclusive on every axis). The
/// constructor enforces `min.x <= max.x` (and likewise for `y` / `z`); a
/// degenerate box where `min == max` on one or more axes is allowed and
/// matches points that lie exactly on that face / edge / corner.
///
/// Implemented as a thin wrapper over [`BKDTree::range_search`] with the
/// 3D bounds — no custom visitor is needed because the BKD's existing
/// axis-aligned range path already does the right thing for AABBs. As a
/// result every match comes back with `distance_m = 0.0` and `score =
/// 1.0` (the BKD only surfaces doc ids, not the points themselves, so
/// per-doc distance scoring would require an extra read pass; users who
/// want distance-based ranking should use [`Geo3dDistanceQuery`]).
///
/// [`BKDTree::range_search`]: crate::lexical::index::structures::bkd_tree::BKDTree::range_search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geo3dBoundingBoxQuery {
    /// Field containing 3D ECEF coordinates.
    field: String,
    /// South-west-bottom corner of the box (per-axis minimum).
    min: GeoEcefPoint,
    /// North-east-top corner of the box (per-axis maximum).
    max: GeoEcefPoint,
    /// Boost factor applied to the constant per-doc score.
    boost: f32,
}

impl Geo3dBoundingBoxQuery {
    /// Construct a new bounding-box query.
    ///
    /// # Errors
    /// Returns `LaurusError::other` when `min` exceeds `max` on any axis.
    /// A degenerate box where they are equal on one or more axes is
    /// accepted (it matches points lying exactly on that face / edge /
    /// corner).
    pub fn new<F: Into<String>>(field: F, min: GeoEcefPoint, max: GeoEcefPoint) -> Result<Self> {
        if min.x > max.x {
            return Err(crate::error::LaurusError::other(format!(
                "Geo3dBoundingBoxQuery: min.x ({}) must be <= max.x ({})",
                min.x, max.x
            )));
        }
        if min.y > max.y {
            return Err(crate::error::LaurusError::other(format!(
                "Geo3dBoundingBoxQuery: min.y ({}) must be <= max.y ({})",
                min.y, max.y
            )));
        }
        if min.z > max.z {
            return Err(crate::error::LaurusError::other(format!(
                "Geo3dBoundingBoxQuery: min.z ({}) must be <= max.z ({})",
                min.z, max.z
            )));
        }
        Ok(Self {
            field: field.into(),
            min,
            max,
            boost: 1.0,
        })
    }

    /// Set the boost factor applied to every match's score.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Field name this query targets.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// SW-bottom corner (per-axis minimum).
    pub fn min(&self) -> GeoEcefPoint {
        self.min
    }

    /// NE-top corner (per-axis maximum).
    pub fn max(&self) -> GeoEcefPoint {
        self.max
    }

    /// Run the query against `reader` and return the matches in
    /// doc-id-ascending order. Every match has `distance_m = 0.0` and
    /// `score = 1.0` — see the type docs for the rationale.
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<Geo3dMatch>> {
        let Some(bkd) = reader.get_bkd_tree(&self.field)? else {
            return Ok(Vec::new());
        };
        let mins = [Some(self.min.x), Some(self.min.y), Some(self.min.z)];
        let maxs = [Some(self.max.x), Some(self.max.y), Some(self.max.z)];
        // Closed bounds on both ends: a point sitting exactly on a face
        // of the box is a hit (consistent with the constructor's
        // tolerance for degenerate boxes).
        let doc_ids = bkd.range_search(&mins, &maxs, true, true)?;
        Ok(doc_ids
            .into_iter()
            .map(|doc_id| Geo3dMatch {
                doc_id,
                distance_m: 0.0,
                score: 1.0,
            })
            .collect())
    }
}

impl Query for Geo3dBoundingBoxQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(Geo3dMatcher::new(self.find_matches(reader)?)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        Ok(Box::new(Geo3dScorer::new(
            self.find_matches(reader)?,
            self.boost,
        )))
    }

    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        // Run the BKD range search once and share it (#996); ranking is
        // identical to building matcher and scorer independently.
        let matches = self.find_matches(reader)?;
        Ok((
            Box::new(Geo3dMatcher::new(matches.clone())),
            Box::new(Geo3dScorer::new(matches, self.boost)),
        ))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn description(&self) -> String {
        format!(
            "Geo3dBoundingBoxQuery(field: {}, min: {:?}, max: {:?})",
            self.field, self.min, self.max
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        // The constructor guarantees min <= max on every axis, so the box
        // is always non-empty in the geometric sense (it contains at
        // least one point). Whether any document falls inside is a
        // runtime question; we leave that to the matcher.
        Ok(false)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        let doc_count = reader.doc_count();
        Ok(doc_count.saturating_mul(2))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field + min/max corners determine the matched set; boost is
        // score-only and excluded.
        Some(format!(
            "geo3dbbox|{:?}|{:?}|{:?}",
            self.field, self.min, self.max
        ))
    }
}

/// k-nearest-neighbour query against an ECEF-typed field.
///
/// Returns the `k` documents whose stored [`GeoEcefPoint`] lies closest
/// to `center`, sorted distance-ascending. Implemented as a sphere
/// query whose radius doubles until at least `k` candidates have been
/// collected (or the index is exhausted, or `max_radius_m` is reached).
///
/// Unlike [`Geo3dDistanceQuery`], the visitor here never lets the BKD
/// short-circuit a subtree as Inside: k-NN ordering needs the *exact*
/// per-point distance, and the Inside path skips reading the
/// coordinates. The trade-off is slightly less aggressive pruning in
/// exchange for stable ordering.
///
/// # Configuration
///
/// - `initial_radius_m`: first radius the loop probes. Default `1000.0`
///   (1 km). Picking this close to the expected nearest-neighbour
///   distance avoids wasted work.
/// - `max_radius_m`: ceiling for the doubling loop. Default `1e10`
///   meters (well beyond Earth radius), large enough that the loop
///   terminates because either `k` candidates are found or the index
///   has been visited in full.
///
/// # Termination
///
/// The loop exits as soon as one of these holds:
/// 1. The number of distinct candidates collected reaches `k`.
/// 2. Doubling the radius produced no additional hits — the index is
///    exhausted.
/// 3. The next radius would exceed `max_radius_m`.
///
/// In every case the result is truncated to the top-`k` by distance,
/// so a smaller-than-`k` result simply means the index does not contain
/// `k` indexed points within `max_radius_m`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geo3dNearestQuery {
    field: String,
    center: GeoEcefPoint,
    k: usize,
    initial_radius_m: f64,
    max_radius_m: f64,
    boost: f32,
}

impl Geo3dNearestQuery {
    /// Initial sphere radius used for the first probe (meters).
    pub const DEFAULT_INITIAL_RADIUS_M: f64 = 1_000.0;
    /// Upper bound for the expanding radius loop (meters). 10⁷ km is
    /// well beyond any real ECEF distance.
    pub const DEFAULT_MAX_RADIUS_M: f64 = 1.0e10;

    /// Construct a new k-NN query with the default radius schedule.
    pub fn new<F: Into<String>>(field: F, center: GeoEcefPoint, k: usize) -> Self {
        Self {
            field: field.into(),
            center,
            k,
            initial_radius_m: Self::DEFAULT_INITIAL_RADIUS_M,
            max_radius_m: Self::DEFAULT_MAX_RADIUS_M,
            boost: 1.0,
        }
    }

    /// Override the initial probe radius.
    pub fn with_initial_radius(mut self, radius_m: f64) -> Self {
        self.initial_radius_m = radius_m.max(0.0);
        self
    }

    /// Override the doubling-loop ceiling.
    pub fn with_max_radius(mut self, radius_m: f64) -> Self {
        self.max_radius_m = radius_m.max(0.0);
        self
    }

    /// Set the boost factor applied to every match's score.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Field name this query targets.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Search center.
    pub fn center(&self) -> GeoEcefPoint {
        self.center
    }

    /// Number of neighbours requested.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Initial probe radius (meters).
    pub fn initial_radius_m(&self) -> f64 {
        self.initial_radius_m
    }

    /// Doubling-loop ceiling (meters).
    pub fn max_radius_m(&self) -> f64 {
        self.max_radius_m
    }

    /// Run the query and return the top-`k` matches in distance-ascending
    /// order. Score is normalized so that the closest hit gets `1.0` and
    /// the farthest hit in the returned set gets `0.0`.
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<Geo3dMatch>> {
        if self.k == 0 {
            return Ok(Vec::new());
        }
        let Some(bkd) = reader.get_bkd_tree(&self.field)? else {
            return Ok(Vec::new());
        };

        // Expanding-radius loop. Start at `initial_radius_m`, double the
        // radius until we collect at least k distinct hits or the radius
        // ceiling is reached. We deliberately do NOT short-circuit on
        // "no growth this iteration": a doubling can fail to add any
        // hits and still reveal new ones a few doublings later (e.g.
        // points clustered far away beyond an empty annulus). Bounding
        // the loop only by `max_radius_m` is safe because doubling from
        // `initial_radius_m` (default 1 km) to `max_radius_m` (default
        // 10⁷ km) takes about 23 iterations — and once we hit k we
        // exit immediately, which is the usual case in practice.
        let mut radius = self.initial_radius_m.max(0.0);
        let visitor: NearestVisitor;

        loop {
            let mut current = NearestVisitor::new(self.center, radius);
            bkd.intersect(&mut current)?;

            // Deduplicate by doc_id (multi-segment readers can produce
            // duplicates) to get an accurate "unique candidates" count.
            let mut deduped = current.hits.clone();
            deduped.sort_by(|a, b| {
                a.doc_id
                    .cmp(&b.doc_id)
                    .then_with(|| a.distance_sq.total_cmp(&b.distance_sq))
            });
            deduped.dedup_by_key(|h| h.doc_id);
            let unique_count = deduped.len();

            // Termination #1: collected enough.
            if unique_count >= self.k {
                current.hits = deduped;
                visitor = current;
                break;
            }

            // Termination #2: hit the radius ceiling — return whatever
            // we found (fewer than k means the index does not contain
            // k points within `max_radius_m`).
            if radius >= self.max_radius_m {
                current.hits = deduped;
                visitor = current;
                break;
            }

            // Double the radius (capped at the ceiling). A starting
            // radius of zero gets bumped to a small positive value so
            // doubling can make progress.
            let next = if radius == 0.0 {
                self.max_radius_m.min(1.0)
            } else {
                (radius * 2.0).min(self.max_radius_m)
            };
            if next == radius {
                current.hits = deduped;
                visitor = current;
                break;
            }
            radius = next;
        }

        // Final sort by distance ascending and truncation to top-k.
        let mut hits = visitor.hits;
        hits.sort_by(|a, b| a.distance_sq.total_cmp(&b.distance_sq));
        hits.truncate(self.k);

        // Normalize scores against the farthest distance in the returned
        // set: closest gets 1.0, farthest gets 0.0. Single-hit results
        // and all-coincident results normalize to 1.0.
        let max_distance = hits.last().map(|h| h.distance_sq.sqrt()).unwrap_or(0.0);

        Ok(hits
            .into_iter()
            .map(|h| {
                let distance = h.distance_sq.sqrt();
                let score = if max_distance == 0.0 {
                    1.0
                } else {
                    (1.0 - distance / max_distance).clamp(0.0, 1.0) as f32
                };
                Geo3dMatch {
                    doc_id: h.doc_id,
                    distance_m: distance,
                    score,
                }
            })
            .collect())
    }
}

/// `IntersectVisitor` for k-NN. Always returns `Outside` or `Crosses`
/// from `compare` — never `Inside` — so every candidate goes through
/// `visit` with its exact coordinates and the visitor can record an
/// exact distance for k-NN ordering.
struct NearestVisitor {
    center: [f64; 3],
    radius_sq: f64,
    hits: Vec<NearestHit>,
}

#[derive(Debug, Clone, Copy)]
struct NearestHit {
    doc_id: u64,
    distance_sq: f64,
}

impl NearestVisitor {
    fn new(center: GeoEcefPoint, radius_m: f64) -> Self {
        let r = radius_m.max(0.0);
        Self {
            center: [center.x, center.y, center.z],
            radius_sq: r * r,
            hits: Vec::new(),
        }
    }
}

impl IntersectVisitor for NearestVisitor {
    fn compare(&self, cell: &AABB) -> CellRelation {
        debug_assert_eq!(cell.num_dims(), 3, "NearestVisitor expects a 3D BKD");
        let min_d_sq = cell.min_distance_sq_to_point(&self.center);
        if min_d_sq > self.radius_sq {
            CellRelation::Outside
        } else {
            // Always Crosses: k-NN needs the exact per-point distance,
            // which the Inside path withholds.
            CellRelation::Crosses
        }
    }

    fn visit_inside(&mut self, _doc_id: u64) {
        // Unreachable in practice — `compare` never returns Inside —
        // but if a future BKD impl chooses to call this we ignore the
        // hit (we have no exact distance to record for k-NN ordering).
    }

    fn visit(&mut self, doc_id: u64, point: &[f64]) {
        debug_assert_eq!(point.len(), 3, "NearestVisitor expects a 3D BKD");
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let d_sq = dx * dx + dy * dy + dz * dz;
        if d_sq <= self.radius_sq {
            self.hits.push(NearestHit {
                doc_id,
                distance_sq: d_sq,
            });
        }
    }
}

impl Query for Geo3dNearestQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(Geo3dMatcher::new(self.find_matches(reader)?)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        Ok(Box::new(Geo3dScorer::new(
            self.find_matches(reader)?,
            self.boost,
        )))
    }

    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        // Run the expanding-radius k-NN search (up to ~23 BKD passes)
        // once and share it (#996); ranking is identical to building
        // matcher and scorer independently.
        let matches = self.find_matches(reader)?;
        Ok((
            Box::new(Geo3dMatcher::new(matches.clone())),
            Box::new(Geo3dScorer::new(matches, self.boost)),
        ))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn description(&self) -> String {
        format!(
            "Geo3dNearestQuery(field: {}, center: {:?}, k: {}, initial_radius: {}m, max_radius: {}m)",
            self.field, self.center, self.k, self.initial_radius_m, self.max_radius_m
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.k == 0)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        // The expanding-radius loop visits the index roughly once per
        // doubling iteration in the worst case; bound at doc_count * 4
        // to give the planner a moderate-cost signal.
        let doc_count = reader.doc_count();
        Ok(doc_count.saturating_mul(4))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field + center + k + the radius schedule fully determine which k
        // documents are returned within a snapshot; boost is score-only and
        // excluded.
        Some(format!(
            "geo3dnear|{:?}|{:?}|{}|{:?}|{:?}",
            self.field, self.center, self.k, self.initial_radius_m, self.max_radius_m
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(min: [f64; 3], max: [f64; 3]) -> AABB {
        AABB::new(min.to_vec(), max.to_vec()).unwrap()
    }

    /// Build a `Geo3dMatch` whose distance deliberately disagrees with its
    /// doc-id order, to exercise the matcher's re-sort.
    fn geo3d_match(doc_id: u64, distance_m: f64) -> Geo3dMatch {
        Geo3dMatch {
            doc_id,
            distance_m,
            score: 1.0,
        }
    }

    /// `Matcher` iteration must be in ascending doc-id order even when the
    /// input list is distance-ordered (regression for hits dropped from
    /// boolean conjunctions, GitHub issue #982).
    #[test]
    fn geo3d_matcher_iterates_in_doc_id_order() {
        let mut matcher = Geo3dMatcher::new(vec![
            geo3d_match(7, 10.0),
            geo3d_match(1, 20.0),
            geo3d_match(4, 30.0),
        ]);

        let mut seen = vec![matcher.doc_id()];
        while matcher.next().unwrap() {
            seen.push(matcher.doc_id());
        }
        assert_eq!(seen, vec![1, 4, 7]);
    }

    /// `skip_to` must find every doc id >= target, exactly as a zig-zag
    /// conjunction driver expects (regression for GitHub issue #982).
    #[test]
    fn geo3d_matcher_skip_to_honors_doc_id_order() {
        let mut matcher = Geo3dMatcher::new(vec![
            geo3d_match(7, 10.0),
            geo3d_match(1, 20.0),
            geo3d_match(4, 30.0),
        ]);

        assert!(matcher.skip_to(2).unwrap());
        assert_eq!(matcher.doc_id(), 4);
        assert!(matcher.skip_to(5).unwrap());
        assert_eq!(matcher.doc_id(), 7);
        assert!(!matcher.skip_to(8).unwrap());
        assert!(matcher.is_exhausted());
    }

    fn visitor(cx: f64, cy: f64, cz: f64, radius: f64) -> SphereVisitor {
        SphereVisitor::new(GeoEcefPoint::new(cx, cy, cz), radius)
    }

    #[test]
    fn sphere_visitor_compare_outside() {
        // Sphere radius 5 around origin; cell sits 100 units away on x.
        let v = visitor(0.0, 0.0, 0.0, 5.0);
        let c = cell([100.0, 0.0, 0.0], [110.0, 10.0, 10.0]);
        assert_eq!(v.compare(&c), CellRelation::Outside);
    }

    /// #1594: a cell entirely within the sphere is reported `Crosses`,
    /// never `Inside`, so its points are decoded and scored by distance.
    #[test]
    fn sphere_visitor_compare_contained_cell_crosses() {
        // Sphere radius 100 around origin; cell tightly centered fits
        // entirely inside (corner distance √(3) * 1 < 100).
        let v = visitor(0.0, 0.0, 0.0, 100.0);
        let c = cell([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert_eq!(v.compare(&c), CellRelation::Crosses);
    }

    #[test]
    fn sphere_visitor_compare_crosses_at_boundary() {
        // Sphere radius 5; cell straddles the boundary.
        let v = visitor(0.0, 0.0, 0.0, 5.0);
        let c = cell([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        assert_eq!(v.compare(&c), CellRelation::Crosses);
    }

    #[test]
    fn sphere_visitor_visit_filters_by_radius() {
        let mut v = visitor(0.0, 0.0, 0.0, 5.0);
        // Inside the sphere
        v.visit(1, &[1.0, 2.0, 2.0]); // sqrt(9) = 3
        // On the boundary (5² + 0 + 0 = 25 == radius²)
        v.visit(2, &[5.0, 0.0, 0.0]);
        // Outside the sphere
        v.visit(3, &[10.0, 0.0, 0.0]);
        assert_eq!(v.hits.len(), 2);
        assert_eq!(v.hits[0].doc_id, 1);
        assert_eq!(v.hits[1].doc_id, 2);
    }

    /// #1594: a coordinate-less `visit_inside` call records nothing, so
    /// no hit can ever carry a fabricated distance of `0.0`.
    #[test]
    fn sphere_visitor_visit_inside_records_nothing() {
        let mut v = visitor(0.0, 0.0, 0.0, 100.0);
        v.visit_inside(42);
        assert!(v.hits.is_empty());
    }

    #[test]
    fn sphere_visitor_negative_radius_matches_nothing() {
        let mut v = visitor(0.0, 0.0, 0.0, -1.0);
        // radius is clamped to 0, so radius² = 0; only a hit at the
        // exact center qualifies.
        v.visit(1, &[0.0, 0.0, 0.0]);
        v.visit(2, &[1.0, 0.0, 0.0]);
        assert_eq!(v.hits.len(), 1);
        assert_eq!(v.hits[0].doc_id, 1);
    }

    #[test]
    fn geo3d_distance_query_basics() {
        let q = Geo3dDistanceQuery::new("position", GeoEcefPoint::new(1.0, 2.0, 3.0), 500.0)
            .with_boost(2.0);
        assert_eq!(q.field(), "position");
        assert_eq!(q.center(), GeoEcefPoint::new(1.0, 2.0, 3.0));
        assert_eq!(q.distance_m(), 500.0);
        assert_eq!(q.boost(), 2.0);
        let cloned = q.clone_box();
        assert!(cloned.description().contains("Geo3dDistanceQuery"));
    }

    #[test]
    fn geo3d_bbox_query_basics() {
        let q = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(0.0, 0.0, 0.0),
            GeoEcefPoint::new(10.0, 20.0, 30.0),
        )
        .unwrap()
        .with_boost(3.0);
        assert_eq!(q.field(), "position");
        assert_eq!(q.min(), GeoEcefPoint::new(0.0, 0.0, 0.0));
        assert_eq!(q.max(), GeoEcefPoint::new(10.0, 20.0, 30.0));
        assert_eq!(q.boost(), 3.0);
        let cloned = q.clone_box();
        assert!(cloned.description().contains("Geo3dBoundingBoxQuery"));
    }

    #[test]
    fn geo3d_bbox_query_accepts_degenerate_box() {
        // Zero-volume box (single point); valid per the type docs.
        let q = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(5.0, 5.0, 5.0),
            GeoEcefPoint::new(5.0, 5.0, 5.0),
        );
        assert!(q.is_ok(), "degenerate (zero-volume) box must be accepted");
    }

    #[test]
    fn geo3d_bbox_query_accepts_zero_volume_axis() {
        // Zero on a single axis (a face) — also valid.
        let q = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(0.0, 0.0, 5.0),
            GeoEcefPoint::new(10.0, 10.0, 5.0),
        );
        assert!(
            q.is_ok(),
            "box with zero extent on one axis must be accepted"
        );
    }

    #[test]
    fn geo3d_nearest_query_basics() {
        let q = Geo3dNearestQuery::new("position", GeoEcefPoint::new(1.0, 2.0, 3.0), 5)
            .with_initial_radius(100.0)
            .with_max_radius(1_000_000.0)
            .with_boost(2.5);
        assert_eq!(q.field(), "position");
        assert_eq!(q.center(), GeoEcefPoint::new(1.0, 2.0, 3.0));
        assert_eq!(q.k(), 5);
        assert_eq!(q.initial_radius_m(), 100.0);
        assert_eq!(q.max_radius_m(), 1_000_000.0);
        assert_eq!(q.boost(), 2.5);
        let cloned = q.clone_box();
        assert!(cloned.description().contains("Geo3dNearestQuery"));
    }

    #[test]
    fn nearest_visitor_compare_outside_when_cell_too_far() {
        let v = NearestVisitor::new(GeoEcefPoint::new(0.0, 0.0, 0.0), 5.0);
        let c = cell([100.0, 100.0, 100.0], [110.0, 110.0, 110.0]);
        assert_eq!(v.compare(&c), CellRelation::Outside);
    }

    #[test]
    fn nearest_visitor_never_returns_inside() {
        // Tight cell wholly inside a huge sphere — `SphereVisitor`
        // would say Inside; `NearestVisitor` must say Crosses so the
        // BKD streams the points and we can compute exact distances.
        let v = NearestVisitor::new(GeoEcefPoint::new(0.0, 0.0, 0.0), 10_000.0);
        let c = cell([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert_eq!(v.compare(&c), CellRelation::Crosses);
    }

    #[test]
    fn nearest_visitor_visit_records_exact_distance() {
        let mut v = NearestVisitor::new(GeoEcefPoint::new(0.0, 0.0, 0.0), 100.0);
        v.visit(1, &[3.0, 4.0, 0.0]); // dist 5
        v.visit(2, &[0.0, 0.0, 6.0]); // dist 6
        v.visit(3, &[200.0, 0.0, 0.0]); // outside
        assert_eq!(v.hits.len(), 2);
        assert_eq!(v.hits[0].doc_id, 1);
        assert_eq!(v.hits[0].distance_sq, 25.0);
        assert_eq!(v.hits[1].doc_id, 2);
        assert_eq!(v.hits[1].distance_sq, 36.0);
    }

    #[test]
    fn geo3d_bbox_query_rejects_inverted_box() {
        // min.x > max.x → reject.
        let err = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(10.0, 0.0, 0.0),
            GeoEcefPoint::new(5.0, 10.0, 10.0),
        )
        .unwrap_err();
        assert!(format!("{err:?}").contains("min.x"));

        // min.y > max.y → reject.
        let err_y = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(0.0, 10.0, 0.0),
            GeoEcefPoint::new(10.0, 5.0, 10.0),
        )
        .unwrap_err();
        assert!(format!("{err_y:?}").contains("min.y"));

        // min.z > max.z → reject.
        let err_z = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(0.0, 0.0, 10.0),
            GeoEcefPoint::new(10.0, 10.0, 5.0),
        )
        .unwrap_err();
        assert!(format!("{err_z:?}").contains("min.z"));
    }

    /// Reader counting `get_bkd_tree` lookups. `find_matches` calls it
    /// exactly once per invocation, so the counter observes how many
    /// times the candidate search ran (#996).
    #[derive(Debug)]
    struct BkdCountingReader {
        get_bkd_tree_calls: std::sync::atomic::AtomicU64,
    }

    impl BkdCountingReader {
        fn new() -> Self {
            BkdCountingReader {
                get_bkd_tree_calls: std::sync::atomic::AtomicU64::new(0),
            }
        }

        fn calls(&self) -> u64 {
            self.get_bkd_tree_calls
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl crate::lexical::reader::LexicalIndexReader for BkdCountingReader {
        fn doc_count(&self) -> u64 {
            1
        }
        fn max_doc(&self) -> u64 {
            1
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(
            &self,
            _doc_id: u64,
        ) -> crate::error::Result<Option<crate::lexical::core::document::Document>> {
            Ok(None)
        }
        fn get_bkd_tree(
            &self,
            _field: &str,
        ) -> crate::error::Result<
            Option<std::sync::Arc<dyn crate::lexical::index::structures::bkd_tree::BKDTree>>,
        > {
            self.get_bkd_tree_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(None)
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>>
        {
            Ok(None)
        }
        fn field_stats(
            &self,
            _field: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> crate::error::Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// #996 regression: `matcher_scorer()` must run `find_matches` once
    /// (one `get_bkd_tree` lookup), not once for the matcher and once
    /// for the scorer.
    #[test]
    fn matcher_scorer_shares_one_bkd_traversal() {
        let reader = BkdCountingReader::new();
        let query = Geo3dDistanceQuery::new("position", GeoEcefPoint::new(0.0, 0.0, 0.0), 1000.0);

        let (_matcher, _scorer) = query.matcher_scorer(&reader).unwrap();

        assert_eq!(
            reader.calls(),
            1,
            "find_matches must run exactly once for the matcher/scorer pair"
        );
    }

    /// Reader whose 3D field is a real [`BKDReader`] written over
    /// `points` with small blocks, so that whole leaves and index nodes
    /// fall inside a query sphere and `compare` pruning runs.
    #[derive(Debug)]
    struct RealBkdReader {
        tree: std::sync::Arc<dyn crate::lexical::index::structures::bkd_tree::BKDTree>,
        max_doc: u64,
    }

    impl RealBkdReader {
        fn new(points: &[(u64, [f64; 3])]) -> Self {
            use crate::lexical::index::structures::bkd_tree::{BKDReader, BKDWriter};
            use crate::storage::Storage;
            use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

            let storage = std::sync::Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            let flat: Vec<f64> = points.iter().flat_map(|(_, p)| p.iter().copied()).collect();
            let doc_ids: Vec<u64> = points.iter().map(|(id, _)| *id).collect();
            {
                let output = storage.create_output("geo3d.bkd").unwrap();
                let mut writer = BKDWriter::new(output, 3).with_block_size(16);
                writer.write(&flat, &doc_ids).unwrap();
                writer.finish().unwrap();
            }
            let reader = BKDReader::open(storage, "geo3d.bkd").unwrap();
            RealBkdReader {
                tree: std::sync::Arc::new(reader),
                max_doc: doc_ids.iter().max().map_or(0, |m| m + 1),
            }
        }
    }

    impl LexicalIndexReader for RealBkdReader {
        fn doc_count(&self) -> u64 {
            self.max_doc
        }
        fn max_doc(&self) -> u64 {
            self.max_doc
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(
            &self,
            _doc_id: u64,
        ) -> crate::error::Result<Option<crate::lexical::core::document::Document>> {
            Ok(None)
        }
        fn get_bkd_tree(
            &self,
            _field: &str,
        ) -> crate::error::Result<
            Option<std::sync::Arc<dyn crate::lexical::index::structures::bkd_tree::BKDTree>>,
        > {
            Ok(Some(self.tree.clone()))
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>>
        {
            Ok(None)
        }
        fn field_stats(
            &self,
            _field: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> crate::error::Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// #1594: with a sphere large enough to contain whole BKD cells, every
    /// hit is still scored `1 - d / r` from its real coordinates, the
    /// reported distance is real, and the order is by true distance —
    /// exactly the set a linear scan finds.
    #[test]
    fn geo3d_distance_query_scores_points_of_contained_cells_by_real_distance() {
        // Deterministic LCG so a failure reproduces.
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };
        // 4 000 points uniform in a 200 km cube around the centre; with
        // 16-point leaves, a 120 km sphere contains many whole cells.
        let center = GeoEcefPoint::new(-3_955_182.0, 3_350_553.0, 3_700_276.0);
        let points: Vec<(u64, [f64; 3])> = (0..4_000u64)
            .map(|i| {
                (
                    i,
                    [
                        center.x + (next() - 0.5) * 200_000.0,
                        center.y + (next() - 0.5) * 200_000.0,
                        center.z + (next() - 0.5) * 200_000.0,
                    ],
                )
            })
            .collect();
        let true_distance = |doc_id: u64| -> f64 {
            let p = points[doc_id as usize].1;
            ((p[0] - center.x).powi(2) + (p[1] - center.y).powi(2) + (p[2] - center.z).powi(2))
                .sqrt()
        };
        let radius = 120_000.0;
        let reader = RealBkdReader::new(&points);

        // Sanity: the sphere really does contain whole cells.
        let mut probe = SphereVisitor::new(center, radius);
        let contained = cell(
            [center.x - 1.0, center.y - 1.0, center.z - 1.0],
            [center.x + 1.0, center.y + 1.0, center.z + 1.0],
        );
        assert_eq!(probe.compare(&contained), CellRelation::Crosses);
        probe.visit_inside(0);
        assert!(probe.hits.is_empty());

        let matches = Geo3dDistanceQuery::new("position", center, radius)
            .find_matches(&reader)
            .unwrap();

        let mut expected: Vec<u64> = (0..points.len() as u64)
            .filter(|&id| true_distance(id) <= radius)
            .collect();
        expected.sort_unstable();
        let mut got: Vec<u64> = matches.iter().map(|m| m.doc_id).collect();
        got.sort_unstable();
        assert_eq!(got, expected, "same set as the linear scan");
        assert!(
            expected.len() > 1_000,
            "the sphere should be well populated"
        );

        for m in &matches {
            let d = true_distance(m.doc_id);
            assert!(
                (m.distance_m - d).abs() < 1e-6,
                "doc {}: reported distance {} vs real {d}",
                m.doc_id,
                m.distance_m
            );
            let expected_score = (1.0 - d / radius) as f32;
            assert!(
                (m.score - expected_score).abs() < 1e-6,
                "doc {}: score {} vs expected {expected_score} (d = {d})",
                m.doc_id,
                m.score
            );
        }
        assert!(
            matches
                .windows(2)
                .all(|w| w[0].distance_m <= w[1].distance_m),
            "results are ordered by true distance"
        );
        let nearest = (0..points.len() as u64)
            .min_by(|&a, &b| true_distance(a).total_cmp(&true_distance(b)))
            .unwrap();
        assert_eq!(
            matches[0].doc_id, nearest,
            "the true nearest point is first"
        );
    }

    /// Deterministic pseudo-random numbers in `[0, 1)` (xorshift64*).
    fn uniform(state: &mut u64) -> f64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn ecef(x: f64, y: f64, z: f64) -> GeoEcefPoint {
        GeoEcefPoint::new(x, y, z)
    }

    /// Issue #1596: distances to segments: beside one, past an end, to a
    /// segment of length 0, and to a single vertex exactly as
    /// `SphereVisitor` measures.
    #[test]
    fn a_geo3d_polyline_measures_distances() {
        let line = Geo3dPolyline::new(vec![
            ecef(0.0, 0.0, 0.0),
            ecef(10.0, 0.0, 0.0),
            ecef(10.0, 0.0, 0.0),
            ecef(10.0, 10.0, 0.0),
        ])
        .unwrap();
        let mut wrong = Vec::new();
        for (point, expected) in [
            (ecef(5.0, 3.0, 4.0), 5.0),
            (ecef(-3.0, 0.0, 4.0), 5.0),
            (ecef(13.0, 14.0, 0.0), 5.0),
            (ecef(10.0, 5.0, 0.0), 0.0),
            (ecef(7.0, 4.0, 0.0), 3.0),
        ] {
            let found = line.distance_to(&point);
            if (found - expected).abs() > 1e-12 {
                wrong.push(format!("{point:?}: {found}, expected {expected}"));
            }
        }
        let center = ecef(-3_955_182.0, 3_350_553.0, 3_700_276.0);
        let point = ecef(-3_955_100.5, 3_350_600.25, 3_700_200.125);
        let single = Geo3dPolyline::new(vec![center]).unwrap();
        let mut sphere = SphereVisitor::new(center, 1_000.0);
        sphere.visit(1, &[point.x, point.y, point.z]);
        let expected = sphere.hits[0].distance_sq;
        if single.distance_sq_to(&[point.x, point.y, point.z]) != expected {
            wrong.push("a single vertex measures otherwise than SphereVisitor".to_string());
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1596: no cell holding a point within the distance of a
    /// segment is pruned: points at the distance or less from a random point
    /// of a random segment, in random directions, in cells of random sizes
    /// around them.
    #[test]
    fn no_cell_near_a_segment_is_pruned() {
        let mut state = 31_u64;
        let random_point = |state: &mut u64, scale: f64| {
            ecef(
                scale * (2.0 * uniform(state) - 1.0),
                scale * (2.0 * uniform(state) - 1.0),
                scale * (2.0 * uniform(state) - 1.0),
            )
        };
        let mut wrong = Vec::new();
        for i in 0..2_000 {
            let a = random_point(&mut state, 6_400_000.0);
            let b = if i % 10 == 0 {
                a
            } else {
                let step = random_point(&mut state, 50_000.0);
                ecef(a.x + step.x, a.y + step.y, a.z + step.z)
            };
            let line = Geo3dPolyline::new(vec![a, b]).unwrap();
            let distance_m = [0.0, 1.0, 500.0, 20_000.0][i % 4];
            let t = uniform(&mut state);
            let on_segment = [
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
            ];
            let direction = random_point(&mut state, 1.0);
            let length = (direction.x.powi(2) + direction.y.powi(2) + direction.z.powi(2)).sqrt();
            let walk = if i % 3 == 0 {
                distance_m
            } else {
                distance_m * uniform(&mut state)
            };
            let point = [
                on_segment[0] + walk * direction.x / length,
                on_segment[1] + walk * direction.y / length,
                on_segment[2] + walk * direction.z / length,
            ];
            let size = [0.0, 1.0, 1_000.0, 100_000.0][(i / 4) % 4];
            let offset = [
                size * uniform(&mut state),
                size * uniform(&mut state),
                size * uniform(&mut state),
            ];
            let low = [
                point[0] - offset[0],
                point[1] - offset[1],
                point[2] - offset[2],
            ];
            let high = [low[0] + size, low[1] + size, low[2] + size];
            if !line.meets_cell(&cell(low, high), distance_m) {
                wrong.push(format!(
                    "{a:?} to {b:?} within {distance_m} m: the cell {low:?} to {high:?} \
                     holding {point:?} pruned"
                ));
            }
        }
        assert!(
            wrong.is_empty(),
            "{} mistakes: {:#?}",
            wrong.len(),
            &wrong[..wrong.len().min(20)]
        );
    }

    /// Issue #1596: through a real BKD, a polyline finds exactly the points
    /// within the distance, testing far fewer points one by one than there
    /// are.
    #[test]
    fn a_geo3d_polyline_prunes_the_bkd() {
        use crate::lexical::index::structures::bkd_tree::{BKDReader, BKDTree, BKDWriter};
        use crate::storage::Storage;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
        use crate::util::ecef::wgs84_to_ecef;

        let n = 20_000_u64;
        let mut state = 32_u64;
        let points: Vec<GeoEcefPoint> = (0..n)
            .map(|_| {
                wgs84_to_ecef(
                    -60.0 + 120.0 * uniform(&mut state),
                    -180.0 + 360.0 * uniform(&mut state),
                    3_000.0 * uniform(&mut state),
                )
            })
            .collect();
        let storage = std::sync::Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("geo3d.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 3).with_block_size(64);
            let flat: Vec<f64> = points.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
            let doc_ids: Vec<u64> = (0..n).collect();
            writer.write(&flat, &doc_ids).unwrap();
            writer.finish().unwrap();
        }
        let tree = BKDReader::open(storage, "geo3d.bkd").unwrap();

        let line = Geo3dPolyline::new(
            [
                (-20.0, -40.0),
                (-10.0, -30.0),
                (-15.0, -20.0),
                (-5.0, -10.0),
            ]
            .iter()
            .map(|&(lat, lon)| wgs84_to_ecef(lat, lon, 1_000.0))
            .collect(),
        )
        .unwrap();
        let distance_m = 100_000.0;
        let mut visitor = Geo3dPolylineVisitor::new(&line, distance_m);
        tree.intersect(&mut visitor).unwrap();
        let visited = visitor.visited;
        let mut found: Vec<u64> = visitor.hits.iter().map(|hit| hit.doc_id).collect();
        found.sort_unstable();
        let expected: Vec<u64> = (0..n)
            .filter(|&id| line.distance_to(&points[id as usize]) <= distance_m)
            .collect();
        assert!(
            !expected.is_empty(),
            "the test needs points within the distance"
        );
        assert_eq!(found, expected);
        assert!(
            visited * 10 < n as usize,
            "tested {visited} of {n} points one by one, for {} hits",
            found.len()
        );
    }
}
