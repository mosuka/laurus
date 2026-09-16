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
/// `1.0 - distance / radius`, clamped to `[0, 1]`. Documents that the BKD
/// reader classified as "fully inside" (their enclosing cell sits
/// entirely within the sphere) are reported with the maximum score `1.0`
/// because the visitor never observes their exact coordinates — see the
/// trade-off note on [`SphereVisitor::visit_inside`].
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
        let mut matches: Vec<Geo3dMatch> = Vec::new();
        let Some(bkd) = reader.get_bkd_tree(&self.field)? else {
            return Ok(matches);
        };

        let mut visitor = SphereVisitor::new(self.center, self.distance_m);
        bkd.intersect(&mut visitor)?;

        let radius = self.distance_m;
        for hit in visitor.hits {
            let distance = hit.distance_sq.sqrt();
            let score = if hit.from_inside_cell {
                1.0
            } else if radius <= 0.0 {
                0.0
            } else {
                (1.0 - distance / radius).clamp(0.0, 1.0) as f32
            };
            matches.push(Geo3dMatch {
                doc_id: hit.doc_id,
                distance_m: distance,
                score,
            });
        }

        // Multi-segment readers can produce duplicates; keep the closest.
        matches.sort_by(|a, b| {
            a.doc_id
                .cmp(&b.doc_id)
                .then_with(|| a.distance_m.total_cmp(&b.distance_m))
        });
        matches.dedup_by_key(|m| m.doc_id);
        // Final order: distance ascending.
        matches.sort_by(|a, b| a.distance_m.total_cmp(&b.distance_m));

        Ok(matches)
    }
}

/// A document matched by a [`Geo3dDistanceQuery`], with its distance to
/// the query center and a `[0, 1]` relevance score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geo3dMatch {
    /// Document id.
    pub doc_id: u64,
    /// Straight-line distance to the query center in meters. `0.0` for
    /// hits reported via [`SphereVisitor::visit_inside`] (the visitor
    /// does not observe the point's coordinates in that case).
    pub distance_m: f64,
    /// Final relevance score in `[0, 1]`, before the query's boost is
    /// applied by the [`Scorer`].
    pub score: f32,
}

/// `IntersectVisitor` that collects every doc whose stored point lies
/// within a sphere of radius `distance_m` centered at `center`.
///
/// The classification logic uses [`AABB::min_distance_sq_to_point`] (for
/// the Outside test) and [`AABB::max_distance_sq_to_point`] (for the
/// Inside test) so both checks happen in squared-distance space and
/// avoid `sqrt`.
struct SphereVisitor {
    center: [f64; 3],
    radius_sq: f64,
    hits: Vec<SphereHit>,
}

#[derive(Debug, Clone, Copy)]
struct SphereHit {
    doc_id: u64,
    /// Squared distance from `center` to the doc's stored point. `0.0`
    /// for hits arriving via `visit_inside` (the visitor does not
    /// observe the coordinates in that case).
    distance_sq: f64,
    /// `true` if the hit was reported via `visit_inside` rather than
    /// `visit`. The query layer treats Inside-cell hits as having a
    /// score of `1.0` regardless of their unknown distance.
    from_inside_cell: bool,
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
        let max_d_sq = cell.max_distance_sq_to_point(&self.center);
        if max_d_sq <= self.radius_sq {
            return CellRelation::Inside;
        }
        CellRelation::Crosses
    }

    /// Trade-off note: `visit_inside` is invoked for every doc beneath a
    /// subtree whose AABB is fully inside the sphere, which by definition
    /// means the doc's point is inside the sphere too — but the visitor
    /// does not get the coordinates, so the exact distance is unknown.
    /// We mark the hit accordingly; the scorer surfaces it as `1.0`. If
    /// callers need exact per-doc distances they should consume
    /// [`Geo3dMatch::distance_m`] only when `score < 1.0`, or fall back
    /// to the `Geo3dNearestQuery` k-NN flavour from #302.
    fn visit_inside(&mut self, doc_id: u64) {
        self.hits.push(SphereHit {
            doc_id,
            distance_sq: 0.0,
            from_inside_cell: true,
        });
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
                from_inside_cell: false,
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

    #[test]
    fn sphere_visitor_compare_inside() {
        // Sphere radius 100 around origin; cell tightly centered fits
        // entirely inside (corner distance √(3) * 1 < 100).
        let v = visitor(0.0, 0.0, 0.0, 100.0);
        let c = cell([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert_eq!(v.compare(&c), CellRelation::Inside);
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
        assert!(!v.hits[0].from_inside_cell);
        assert_eq!(v.hits[1].doc_id, 2);
    }

    #[test]
    fn sphere_visitor_visit_inside_marks_hit_uniformly() {
        let mut v = visitor(0.0, 0.0, 0.0, 100.0);
        v.visit_inside(42);
        assert_eq!(v.hits.len(), 1);
        assert_eq!(v.hits[0].doc_id, 42);
        assert_eq!(v.hits[0].distance_sq, 0.0);
        assert!(v.hits[0].from_inside_cell);
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
}
