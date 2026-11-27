//! Geographical search functionality for location-based queries.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// Canonical GeoPoint definition lives in `crate::data` alongside DataValue.
// We re-export it here so that `use laurus::lexical::query::GeoPoint` keeps
// working for downstream callers.
pub use crate::data::GeoPoint;

use crate::error::Result;
use crate::lexical::index::structures::aabb::AABB;
use crate::lexical::index::structures::visitor::{CellRelation, IntersectVisitor};
use crate::lexical::query::Query;
use crate::lexical::query::matcher::{Matcher, PreComputedMatcher};
use crate::lexical::query::scorer::{ConstantScorer, Scorer};
use crate::lexical::reader::{LexicalIndexReader, scan_doc_ids};

/// A 2D geographical bounding box, expressed as the south-west `min` corner
/// and the north-east `max` corner of a rectangle of latitudes and
/// longitudes.
///
/// The latitudes are the closed interval `[min.lat, max.lat]`; the
/// constructor returns `LaurusError::other` when `min.lat > max.lat`.
///
/// The longitudes run eastward from `min.lon` to `max.lon`. When
/// `min.lon > max.lon` the box crosses the antimeridian (±180°) and covers
/// `[min.lon, 180]` and `[-180, max.lon]`, the convention of GeoJSON
/// ([RFC 7946 section 5.2](https://datatracker.ietf.org/doc/html/rfc7946#section-5.2)),
/// so a map viewport over the Pacific is one box (Issue #1595). An edge on
/// exactly ±180° covers that meridian whichever sign a point is stored
/// with, as a circle in [`GeoDistanceQuery`] does.
///
/// The previous `top_left` / `bottom_right` representation was removed in
/// #297.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoBoundingBox {
    /// South-west corner: minimum latitude and the western edge's longitude.
    pub min: GeoPoint,
    /// North-east corner: maximum latitude and the eastern edge's longitude.
    pub max: GeoPoint,
}

/// The longitudes of a [`GeoBoundingBox`], as one or two disjoint closed
/// ranges, each within `[-180, 180]`. Returned by
/// [`GeoBoundingBox::lon_ranges`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct LonRanges {
    /// The `len` ranges, as `(min_lon, max_lon)`.
    ranges: [(f64, f64); 2],
    len: usize,
}

impl LonRanges {
    fn one(range: (f64, f64)) -> Self {
        LonRanges {
            ranges: [range, range],
            len: 1,
        }
    }

    fn two(first: (f64, f64), second: (f64, f64)) -> Self {
        LonRanges {
            ranges: [first, second],
            len: 2,
        }
    }

    /// The ranges, the one that starts at the box's western edge first.
    fn iter(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.ranges[..self.len].iter().copied()
    }
}

impl GeoBoundingBox {
    /// Create a new bounding box from its south-west and north-east corners.
    ///
    /// Errors if `min.lat > max.lat`. `min.lon > max.lon` is a box that
    /// crosses the antimeridian; see [`GeoBoundingBox`].
    pub fn new(min: GeoPoint, max: GeoPoint) -> Result<Self> {
        if min.lat > max.lat {
            return Err(crate::error::LaurusError::other(format!(
                "min latitude {} must be <= max latitude {}",
                min.lat, max.lat
            )));
        }
        Ok(GeoBoundingBox { min, max })
    }

    /// The box's longitudes as closed ranges: `[min.lon, max.lon]`, or
    /// `[min.lon, 180]` and `[-180, max.lon]` when the box crosses the
    /// antimeridian. An edge on ±180° also covers the other sign of the
    /// same meridian (a point can be stored with either), unless the box
    /// is the whole circle `[-180, 180]`.
    fn lon_ranges(&self) -> LonRanges {
        let (min, max) = (self.min.lon, self.max.lon);
        if min > max {
            // Holds both signs of ±180 already.
            LonRanges::two((min, 180.0), (-180.0, max))
        } else if max >= 180.0 && min > -180.0 {
            LonRanges::two((min, max), (-180.0, -180.0))
        } else if min <= -180.0 && max < 180.0 {
            LonRanges::two((min, max), (180.0, 180.0))
        } else {
            LonRanges::one((min, max))
        }
    }

    /// Whether `point` lies inside the box, edges included.
    pub fn contains(&self, point: &GeoPoint) -> bool {
        self.lon_ranges().iter().any(|(min_lon, max_lon)| {
            point.within_bounds(self.min.lat, self.max.lat, min_lon, max_lon)
        })
    }

    /// Center of the bounding box, clamped to the valid lat/lon ranges as
    /// a defensive measure (a malformed box could in theory carry
    /// out-of-range corners). For a box that crosses the antimeridian the
    /// longitude is half the width east of the south-west corner, past
    /// ±180° if need be, so the center is inside the box.
    pub fn center(&self) -> GeoPoint {
        let center_lat = ((self.min.lat + self.max.lat) / 2.0).clamp(-90.0, 90.0);
        let mut center_lon = if self.min.lon > self.max.lon {
            self.min.lon + self.dimensions().0 / 2.0
        } else {
            (self.min.lon + self.max.lon) / 2.0
        };
        if center_lon > 180.0 {
            center_lon -= 360.0;
        }
        // Clamped to valid ranges, so the infallible constructor is fine.
        GeoPoint::new(center_lat, center_lon.clamp(-180.0, 180.0))
    }

    /// Width (lon span) and height (lat span) in degrees. The width of a
    /// box that crosses the antimeridian is the arc from its western edge
    /// eastward to its eastern edge.
    pub fn dimensions(&self) -> (f64, f64) {
        let mut width = self.max.lon - self.min.lon;
        if width < 0.0 {
            width += 360.0;
        }
        let height = self.max.lat - self.min.lat;
        (width, height)
    }

    /// Greatest great-circle distance from the box's center to any of its
    /// four corners, in meters.
    pub fn max_distance_from_center(&self) -> f64 {
        let center = self.center();
        let sw = self.min;
        let ne = self.max;
        // The bounding box invariant guarantees that swapping corners
        // produces in-range GeoPoints, so the infallible constructor is
        // appropriate here.
        let nw = GeoPoint::new(self.max.lat, self.min.lon);
        let se = GeoPoint::new(self.min.lat, self.max.lon);
        [&sw, &se, &nw, &ne]
            .iter()
            .map(|corner| center.distance_to(corner))
            .fold(0.0, f64::max)
    }
}

/// [`IntersectVisitor`] collecting `(doc_id, GeoPoint)` pairs for every
/// indexed point inside a closed 2D `[lat, lon]` box, reading the
/// coordinates straight from the BKD leaves (#1000).
///
/// `compare` never returns [`CellRelation::Inside`]: the `Inside` path
/// reports doc ids via `visit_inside` **without** coordinates, which the
/// geo queries need for exact haversine filtering and distance-based
/// scoring. Returning `Crosses` for a fully-contained cell only costs
/// pruning efficiency, never correctness (same pattern as geo3d's
/// `NearestVisitor` and the merge engine's `CollectPointsVisitor`).
#[derive(Debug)]
struct GeoBoxPointsVisitor {
    /// Closed lower bounds: `[min_lat, min_lon]`.
    min: [f64; 2],
    /// Closed upper bounds: `[max_lat, max_lon]`.
    max: [f64; 2],
    /// Collected in-box points, in BKD traversal order.
    hits: Vec<(u64, GeoPoint)>,
}

impl GeoBoxPointsVisitor {
    /// Create a visitor for the closed box `[min, max]` (both bounds
    /// inclusive, matching the previous `range_search(.., true, true)`).
    fn new(min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Self {
        GeoBoxPointsVisitor {
            min: [min_lat, min_lon],
            max: [max_lat, max_lon],
            hits: Vec::new(),
        }
    }

    /// Consume the visitor and return the collected `(doc_id, point)`
    /// pairs, sorted by doc id. Deliberately **not** deduplicated: a
    /// multi-valued field (#1174) legitimately contributes several points
    /// per document, and the distance query pre-filters by a rectangle
    /// around the circle, so keeping only the first BKD-traversal point
    /// could drop a document whose first point lies in a rectangle corner
    /// while a later one lies inside the circle. `find_matches` dedups
    /// after evaluating every point, keeping the closest.
    fn into_candidates(mut self) -> Vec<(u64, GeoPoint)> {
        self.hits.sort_by_key(|(doc_id, _)| *doc_id);
        self.hits
    }
}

impl IntersectVisitor for GeoBoxPointsVisitor {
    fn compare(&self, cell: &AABB) -> CellRelation {
        debug_assert_eq!(cell.num_dims(), 2, "GeoBoxPointsVisitor expects a 2D BKD");
        let cell_min = cell.min();
        let cell_max = cell.max();
        let disjoint =
            (0..2).any(|dim| cell_max[dim] < self.min[dim] || cell_min[dim] > self.max[dim]);
        if disjoint {
            CellRelation::Outside
        } else {
            // Always Crosses, even for contained cells: the geo queries
            // need the per-point coordinates the Inside path withholds.
            CellRelation::Crosses
        }
    }

    fn visit_inside(&mut self, _doc_id: u64) {
        // Unreachable in practice — `compare` never returns Inside —
        // but if a future BKD impl chooses to call this we ignore the
        // hit (we have no coordinates to record).
    }

    fn visit(&mut self, doc_id: u64, point: &[f64]) {
        let inside = (0..2).all(|dim| point[dim] >= self.min[dim] && point[dim] <= self.max[dim]);
        if inside {
            self.hits.push((doc_id, GeoPoint::new(point[0], point[1])));
        }
    }
}

/// A geographical distance query that finds documents within a certain distance of a point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoDistanceQuery {
    /// Field containing geographical coordinates
    field: String,
    /// Center point for the search
    center: GeoPoint,
    /// Maximum distance in meters
    distance_m: f64,
    /// Boost factor for the query
    boost: f32,
}

impl GeoDistanceQuery {
    /// Create a new geo distance query.
    pub fn new<F: Into<String>>(field: F, center: GeoPoint, distance_m: f64) -> Self {
        GeoDistanceQuery {
            field: field.into(),
            center,
            distance_m,
            boost: 1.0,
        }
    }

    /// Create a geo distance query from raw lat/lon coordinates.
    ///
    /// Validates the latitude / longitude bounds via [`GeoPoint::try_new`]
    /// (`-90 <= lat <= 90`, `-180 <= lon <= 180`) and returns
    /// `LaurusError::other` for out-of-range values.
    ///
    /// # Arguments
    ///
    /// * `field` - The field containing geographical coordinates.
    /// * `lat` - Center latitude in degrees (`-90` to `90`).
    /// * `lon` - Center longitude in degrees (`-180` to `180`).
    /// * `distance_m` - Maximum distance from the centre in meters.
    ///
    /// # Example
    ///
    /// ```rust
    /// use laurus::lexical::query::GeoDistanceQuery;
    ///
    /// let query =
    ///     GeoDistanceQuery::within_radius("location", 40.7128, -74.0060, 10_000.0).unwrap();
    /// ```
    pub fn within_radius<F: Into<String>>(
        field: F,
        lat: f64,
        lon: f64,
        distance_m: f64,
    ) -> Result<Self> {
        let center = GeoPoint::try_new(lat, lon)?;
        Ok(GeoDistanceQuery::new(field, center, distance_m))
    }

    /// Set the boost factor.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the center point.
    pub fn center(&self) -> GeoPoint {
        self.center
    }

    /// Get the search distance in meters.
    pub fn distance_m(&self) -> f64 {
        self.distance_m
    }

    /// Find matching documents and their distances using spatial indexing.
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<GeoMatch>> {
        let mut matches = Vec::new();

        // The lat/lon rectangle(s) enclosing the circle, for BKD pruning.
        let bounds = self.cap_bounds();

        // Get candidates from the index. Every point is evaluated — a
        // document may appear several times (multi-valued field, #1174;
        // multi-segment reader) and is deduplicated *after* the radius
        // test below so a far-away point never shadows a matching one.
        let candidates = self.get_spatial_candidates(reader, &bounds)?;

        for (doc_id, point) in candidates {
            let distance = self.center.distance_to(&point);
            if distance <= self.distance_m {
                matches.push(GeoMatch {
                    doc_id,
                    point,
                    distance_m: distance,
                    relevance_score: distance_score(distance, self.distance_m),
                });
            }
        }

        Ok(closest_per_document(matches))
    }

    /// The lat/lon rectangle(s) enclosing the spherical cap of radius
    /// `distance_m` around `center`, used to prune the BKD before the exact
    /// haversine test in [`find_matches`](Self::find_matches).
    ///
    /// The cap is bounded exactly, on the same sphere as
    /// [`GeoPoint::distance_to`] (#1593):
    ///
    /// * latitude spans `center.lat ± δ` where `δ = distance_m / R` in degrees;
    /// * if that span reaches a pole the cap wraps around it and every
    ///   longitude is in range;
    /// * otherwise the cap's widest longitude extent is
    ///   `Δλ = asin(sin δ / cos φ)` (not at the centre's latitude but nearer the
    ///   pole, so `δ / cos φ` would be too narrow), and `center.lon ± Δλ` is
    ///   split in two when it crosses the antimeridian.
    ///
    /// The previous `111 km/°` box clamped longitude to `[-180, 180]`, so a
    /// circle crossing the antimeridian, containing a pole, or large at high
    /// latitude lost the matches outside the box before the exact test ran.
    fn cap_bounds(&self) -> CapBounds {
        if self.distance_m.is_nan() || self.distance_m <= 0.0 {
            // Zero, negative or NaN radius: only the centre itself can match.
            return CapBounds {
                min_lat: self.center.lat,
                max_lat: self.center.lat,
                lon_ranges: vec![(self.center.lon, self.center.lon)],
            };
        }

        let delta_rad = self.distance_m / EARTH_RADIUS_M;
        if delta_rad >= std::f64::consts::PI {
            return CapBounds::whole_sphere();
        }
        let delta_deg = delta_rad.to_degrees();

        let min_lat = self.center.lat - delta_deg;
        let max_lat = self.center.lat + delta_deg;
        if min_lat <= -90.0 || max_lat >= 90.0 {
            // The cap contains a pole: every meridian passes through it.
            return CapBounds {
                min_lat: min_lat.max(-90.0),
                max_lat: max_lat.min(90.0),
                lon_ranges: vec![(-180.0, 180.0)],
            };
        }

        // cos φ > 0 here because |center.lat| < 90 - δ. The ratio is < 1 for
        // the same reason, but clamp it so rounding never yields NaN.
        let ratio = (delta_rad.sin() / self.center.lat.to_radians().cos()).min(1.0);
        let lon_delta = ratio.asin().to_degrees();
        if lon_delta >= 180.0 {
            return CapBounds {
                min_lat,
                max_lat,
                lon_ranges: vec![(-180.0, 180.0)],
            };
        }

        CapBounds::with_lon_span(
            min_lat,
            max_lat,
            self.center.lon - lon_delta,
            self.center.lon + lon_delta,
        )
    }

    /// Get spatial candidates from the index within the cap bounds.
    fn get_spatial_candidates(
        &self,
        reader: &dyn LexicalIndexReader,
        bounds: &CapBounds,
    ) -> Result<Vec<(u64, GeoPoint)>> {
        let mut candidates = Vec::new();

        // Try to use BKD tree for efficient candidate retrieval. The
        // coordinates come straight from the BKD leaves via the visitor
        // — no stored-document probes, so `indexed = true, stored =
        // false` geo fields work too (#1000). One traversal per longitude
        // range: two when the cap crosses the antimeridian.
        if let Some(bkd_tree) = reader.get_bkd_tree(&self.field)? {
            for &(min_lon, max_lon) in &bounds.lon_ranges {
                let mut visitor =
                    GeoBoxPointsVisitor::new(bounds.min_lat, min_lon, bounds.max_lat, max_lon);
                bkd_tree.intersect(&mut visitor)?;
                candidates.extend(visitor.into_candidates());
            }
            candidates.sort_by_key(|(doc_id, _)| *doc_id);
            return Ok(candidates);
        }

        // Fallback: scan the stored documents present in this reader
        // (segment-bounded under the fanout; correct for sparse id
        // spaces — #996).
        for doc_id in scan_doc_ids(reader)? {
            // Get the document
            if let Some(doc) = reader.document(doc_id)? {
                // Get the geo field value
                if let Some(field_value) = doc.get_field(&self.field) {
                    // A multi-valued field (#1174) contributes every point;
                    // `find_matches` keeps the closest one per document.
                    for geo_point in stored_geo_points(field_value) {
                        // First check bounding box for efficiency, then exact distance
                        if bounds.contains(&geo_point) {
                            let distance = self.center.distance_to(&geo_point);
                            // Double-check with exact distance calculation
                            if distance <= self.distance_m {
                                candidates.push((doc_id, geo_point));
                            }
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }
}

/// The score of a match at `distance` metres of a query reaching `radius`
/// metres: `1` at distance 0, falling linearly to `0` at the radius.
fn distance_score(distance: f64, radius: f64) -> f32 {
    if distance == 0.0 {
        1.0
    } else {
        (1.0 - (distance / radius)).max(0.0) as f32
    }
}

/// `matches` with the closest one per document only, closest first.
/// Multi-segment readers and multi-valued fields can produce several
/// matches per document.
fn closest_per_document(mut matches: Vec<GeoMatch>) -> Vec<GeoMatch> {
    matches.sort_by(|a, b| {
        a.doc_id
            .cmp(&b.doc_id)
            .then_with(|| a.distance_m.total_cmp(&b.distance_m))
    });
    matches.dedup_by_key(|m| m.doc_id);
    // Sort by distance (closest first), then by relevance score
    matches.sort_by(|a, b| {
        a.distance_m
            .total_cmp(&b.distance_m)
            .then_with(|| b.relevance_score.total_cmp(&a.relevance_score))
    });
    matches
}

/// Lat/lon bounds of a spherical cap: one latitude span and one or two
/// closed longitude ranges (two when the cap crosses the antimeridian).
/// Built by [`GeoDistanceQuery::cap_bounds`], and for each edge of a
/// [`GeoPolyline`].
#[derive(Debug, Clone, PartialEq)]
struct CapBounds {
    min_lat: f64,
    max_lat: f64,
    /// Closed `(min_lon, max_lon)` ranges, each within `[-180, 180]`.
    lon_ranges: Vec<(f64, f64)>,
}

impl CapBounds {
    fn whole_sphere() -> Self {
        CapBounds {
            min_lat: -90.0,
            max_lat: 90.0,
            lon_ranges: vec![(-180.0, 180.0)],
        }
    }

    /// The bounds of the latitudes `[min_lat, max_lat]` and of the
    /// longitudes eastward from `min_lon` to `max_lon`, which may go past
    /// ±180 on one side and are split there; a span of 360 degrees or more
    /// is every longitude.
    fn with_lon_span(min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Self {
        // `<=` / `>=`: a range ending exactly on the antimeridian also
        // covers the point stored with the other sign (-180 == 180).
        let lon_ranges = if max_lon - min_lon >= 360.0 {
            vec![(-180.0, 180.0)]
        } else if min_lon <= -180.0 {
            vec![(-180.0, max_lon), (min_lon + 360.0, 180.0)]
        } else if max_lon >= 180.0 {
            vec![(min_lon, 180.0), (-180.0, max_lon - 360.0)]
        } else {
            vec![(min_lon, max_lon)]
        };
        CapBounds {
            min_lat,
            max_lat,
            lon_ranges,
        }
    }

    /// Whether the bounds meet the closed cell from `min` to `max`
    /// (`[lat, lon]`, longitudes within `[-180, 180]`).
    fn meets(&self, min: &[f64], max: &[f64]) -> bool {
        max[0] >= self.min_lat
            && min[0] <= self.max_lat
            && self
                .lon_ranges
                .iter()
                .any(|&(min_lon, max_lon)| max[1] >= min_lon && min[1] <= max_lon)
    }

    /// Whether `point` lies inside the bounds (closed on every edge).
    fn contains(&self, point: &GeoPoint) -> bool {
        point.lat >= self.min_lat
            && point.lat <= self.max_lat
            && self
                .lon_ranges
                .iter()
                .any(|&(min_lon, max_lon)| point.lon >= min_lon && point.lon <= max_lon)
    }
}

/// The 2-D points held by a stored geo field value: one for
/// [`DataValue::Geo`](crate::data::DataValue::Geo), each element of a
/// [`DataValue::GeoArray`](crate::data::DataValue::GeoArray) (#1174), and
/// none for any other variant. Shared by the stored-fields fallbacks of the
/// distance and bounding-box queries.
fn stored_geo_points(value: &crate::data::DataValue) -> impl Iterator<Item = GeoPoint> + '_ {
    use crate::data::DataValue;

    let points: &[GeoPoint] = match value {
        DataValue::Geo(p) => std::slice::from_ref(p),
        DataValue::GeoArray(arr) => arr,
        _ => &[],
    };
    points.iter().copied()
}

#[cfg(test)]
impl GeoDistanceQuery {
    /// Calculate relevance score based on distance (closer = higher score).
    fn calculate_distance_score(&self, distance_m: f64) -> f32 {
        if distance_m > self.distance_m {
            return 0.0;
        }

        // Linear decay: score = 1.0 at center, 0.0 at max distance
        let normalized_distance = distance_m / self.distance_m;
        (1.0 - normalized_distance) as f32
    }

    /// Calculate enhanced relevance score with multiple factors.
    fn calculate_distance_score_enhanced(&self, distance_m: f64, point: &GeoPoint) -> f32 {
        if distance_m > self.distance_m {
            return 0.0;
        }

        // Base distance score (exponential decay for better distance weighting)
        let normalized_distance = distance_m / self.distance_m;
        let base_score = (-2.0 * normalized_distance).exp() as f32;

        // Precision bonus for exact location matches (within 100 m)
        let precision_bonus = if distance_m < 100.0 { 0.1 } else { 0.0 };

        // Geographic relevance bonus (e.g., prefer points in certain regions)
        let geo_bonus = self.calculate_geographic_relevance(point);

        // Population density estimation (simulated)
        let density_bonus = self.estimate_population_density(point) * 0.05;

        (base_score + precision_bonus + geo_bonus + density_bonus).min(1.0)
    }

    /// Calculate geographic relevance based on point characteristics.
    fn calculate_geographic_relevance(&self, point: &GeoPoint) -> f32 {
        // Bonus for points in certain latitudinal zones (e.g., temperate zones)
        let lat_abs = point.lat.abs();
        let temperate_bonus = if lat_abs > 23.5 && lat_abs < 66.5 {
            0.05
        } else {
            0.0
        };

        // Bonus for points near major meridians or equator
        let meridian_bonus = if point.lon.abs() % 15.0 < 1.0 {
            0.02
        } else {
            0.0
        };
        let equator_bonus = if point.lat.abs() < 5.0 { 0.03 } else { 0.0 };

        temperate_bonus + meridian_bonus + equator_bonus
    }

    /// Estimate population density bonus (simplified simulation).
    fn estimate_population_density(&self, point: &GeoPoint) -> f32 {
        // Simplified heuristic: higher density near major coordinates
        let lat_density = (1.0 - (point.lat.abs() / 90.0)) as f32;
        let lon_density = (1.0 - (point.lon.abs() / 180.0)) as f32;

        // Coastal bonus (simplified: points near 0° longitude or specific latitudes)
        let coastal_bonus = if point.lon.abs() < 10.0 || (point.lat.abs() - 40.0).abs() < 5.0 {
            0.2
        } else {
            0.0
        };

        ((lat_density + lon_density) / 2.0 + coastal_bonus).min(1.0)
    }
}

impl Query for GeoDistanceQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        let matches = self.find_matches(reader)?;
        Ok(Box::new(GeoMatcher::new(matches)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        let matches = self.find_matches(reader)?;
        Ok(Box::new(GeoScorer::new(matches, self.boost)))
    }

    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        // Run the candidate scan once and share it (#996): the scorer's
        // per-doc distance scores derive from the same matches the
        // matcher iterates, so ranking is identical to building them
        // independently.
        let matches = self.find_matches(reader)?;
        Ok((
            Box::new(GeoMatcher::new(matches.clone())),
            Box::new(GeoScorer::new(matches, self.boost)),
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
            "GeoDistanceQuery(field: {}, center: {:?}, distance: {}m)",
            self.field, self.center, self.distance_m
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.distance_m <= 0.0)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        // Geo queries can be expensive depending on the spatial index
        let doc_count = reader.doc_count() as u32;
        Ok(doc_count as u64 * 2) // Moderate cost
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field + center + radius determine the matched set; boost is
        // score-only and excluded. `{:?}` renders the f64 coordinates
        // deterministically.
        Some(format!(
            "geodist|{:?}|{:?}|{:?}",
            self.field, self.center, self.distance_m
        ))
    }
}

/// A geographical bounding box query.
///
/// This query matches documents with coordinates within a rectangular bounding box
/// defined by its south-west and north-east corners, edges included. A box with
/// `min.lon > max.lon` crosses the antimeridian, and an edge on ±180° also covers
/// points stored with the other sign; see [`GeoBoundingBox`].
///
/// # Scoring Behavior
///
/// Documents within the bounding box are scored based on their distance from the
/// bounding box center. Documents closer to the center receive higher scores.
/// This allows for relevance ranking even when all results are within the box.
///
/// - Score range: 0.0 to 1.0
/// - Documents at the center: score ≈ 1.0
/// - Documents at the edges: score decreases based on distance from center
///
/// If you need all documents within the box to have equal scores, consider using
/// a constant score wrapper or filtering approach instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoBoundingBoxQuery {
    /// Field containing geographical coordinates
    field: String,
    /// Bounding box for the search
    bounding_box: GeoBoundingBox,
    /// Boost factor for the query
    boost: f32,
}

impl GeoBoundingBoxQuery {
    /// Create a new geo bounding box query.
    pub fn new<F: Into<String>>(field: F, bounding_box: GeoBoundingBox) -> Self {
        GeoBoundingBoxQuery {
            field: field.into(),
            bounding_box,
            boost: 1.0,
        }
    }

    /// Create a geo bounding box query from raw lat/lon corner coordinates.
    ///
    /// Validates that each corner falls within the latitude / longitude
    /// bounds (via [`GeoPoint::try_new`]) and that `min_lat <= max_lat`
    /// (via [`GeoBoundingBox::new`]). `min_lon > max_lon` is a box that
    /// crosses the antimeridian.
    ///
    /// # Arguments
    ///
    /// * `field` - The field containing geographical coordinates.
    /// * `min_lat` - Minimum (south) latitude.
    /// * `min_lon` - Western edge: the longitude the box starts at.
    /// * `max_lat` - Maximum (north) latitude.
    /// * `max_lon` - Eastern edge: the longitude the box ends at, going east
    ///   from `min_lon`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use laurus::lexical::query::GeoBoundingBoxQuery;
    ///
    /// let query = GeoBoundingBoxQuery::within_bounding_box(
    ///     "location", 40.0, -75.0, 41.0, -74.0,
    /// ).unwrap();
    ///
    /// // Over the Pacific, from 175 degrees east to 175 degrees west.
    /// let across = GeoBoundingBoxQuery::within_bounding_box(
    ///     "location", -20.0, 175.0, -15.0, -175.0,
    /// ).unwrap();
    /// ```
    pub fn within_bounding_box<F: Into<String>>(
        field: F,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<Self> {
        let min = GeoPoint::try_new(min_lat, min_lon)?;
        let max = GeoPoint::try_new(max_lat, max_lon)?;
        let bbox = GeoBoundingBox::new(min, max)?;
        Ok(GeoBoundingBoxQuery::new(field, bbox))
    }

    /// Set the boost factor.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the bounding box.
    pub fn bounding_box(&self) -> &GeoBoundingBox {
        &self.bounding_box
    }

    /// Find matching documents within the bounding box.
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<GeoMatch>> {
        let mut matches = Vec::new();

        // Get candidates from the spatial index. Every point is evaluated
        // and documents are deduplicated afterwards (see the distance
        // query for why).
        let candidates = self.get_candidates_in_bounds(reader)?;

        for (doc_id, point) in candidates {
            if self.bounding_box.contains(&point) {
                let center = self.bounding_box.center();
                let distance = center.distance_to(&point);

                // Simple scoring based on position within bounding box.
                // Every point of a box with zero size is its center, even
                // when rounding makes the distance nonzero (±180° stored
                // with the other sign).
                let max_distance = self.bounding_box.max_distance_from_center();
                let relevance_score = if distance == 0.0 || max_distance == 0.0 {
                    1.0
                } else {
                    // Closer to center gets higher score
                    ((max_distance - distance) / max_distance).max(0.0) as f32
                };

                matches.push(GeoMatch {
                    doc_id,
                    point,
                    distance_m: distance,
                    relevance_score,
                });
            }
        }

        // Multi-segment readers and multi-valued fields can produce
        // duplicates; keep the point closest to the box center per
        // document (the same distance the score is derived from).
        matches.sort_by(|a, b| {
            a.doc_id
                .cmp(&b.doc_id)
                .then_with(|| a.distance_m.total_cmp(&b.distance_m))
        });
        matches.dedup_by_key(|m| m.doc_id);

        // Sort by relevance score (highest first), then by distance to center
        matches.sort_by(|a, b| {
            b.relevance_score
                .total_cmp(&a.relevance_score)
                .then_with(|| a.distance_m.total_cmp(&b.distance_m))
        });

        Ok(matches)
    }

    /// Get candidate points that might be within the bounding box.
    fn get_candidates_in_bounds(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<Vec<(u64, GeoPoint)>> {
        let mut candidates = Vec::new();

        // Try to use BKD tree for efficient candidate retrieval. The
        // coordinates come straight from the BKD leaves via the visitor
        // — no stored-document probes, so `indexed = true, stored =
        // false` geo fields work too (#1000). One traversal per longitude
        // range: two when the box crosses the antimeridian or has an edge
        // on it (#1595).
        if let Some(bkd_tree) = reader.get_bkd_tree(&self.field)? {
            let (min_lat, max_lat) = (self.bounding_box.min.lat, self.bounding_box.max.lat);
            for (min_lon, max_lon) in self.bounding_box.lon_ranges().iter() {
                let mut visitor = GeoBoxPointsVisitor::new(min_lat, min_lon, max_lat, max_lon);
                bkd_tree.intersect(&mut visitor)?;
                candidates.extend(visitor.into_candidates());
            }
            candidates.sort_by_key(|(doc_id, _)| *doc_id);
            return Ok(candidates);
        }

        // Fallback: scan the stored documents present in this reader
        // (segment-bounded under the fanout; correct for sparse id
        // spaces — #996).
        for doc_id in scan_doc_ids(reader)? {
            // Get the document
            if let Some(doc) = reader.document(doc_id)? {
                // Get the geo field value
                if let Some(field_value) = doc.get_field(&self.field) {
                    // A multi-valued field (#1174) contributes every point;
                    // `find_matches` keeps the closest one per document.
                    for geo_point in stored_geo_points(field_value) {
                        // Check if the point is within the bounding box
                        if self.bounding_box.contains(&geo_point) {
                            candidates.push((doc_id, geo_point));
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }
}

#[cfg(test)]
impl GeoBoundingBoxQuery {
    /// Generate candidates within and around the bounding box.
    fn generate_bounding_box_candidates(&self) -> Vec<(u64, GeoPoint)> {
        let mut candidates = Vec::new();
        let (width, height) = self.bounding_box.dimensions();

        // Generate grid points within the bounding box
        let grid_size = 20;
        for i in 0..grid_size {
            for j in 0..grid_size {
                let lat_ratio = i as f64 / (grid_size - 1) as f64;
                let lon_ratio = j as f64 / (grid_size - 1) as f64;

                let lat = self.bounding_box.min.lat + lat_ratio * height;
                // Past +180 a box that crosses the antimeridian continues
                // from -180.
                let mut lon = self.bounding_box.min.lon + lon_ratio * width;
                if lon > 180.0 {
                    lon -= 360.0;
                }

                if let Ok(point) = GeoPoint::try_new(lat, lon) {
                    let doc_id = (i * grid_size + j + 2000) as u64;
                    candidates.push((doc_id, point));
                }
            }
        }

        // Add some points outside the box for testing boundary conditions
        let expansion_factor = 0.1;
        let expanded_width = width * (1.0 + expansion_factor);
        let expanded_height = height * (1.0 + expansion_factor);

        for i in 0..10 {
            let angle = (i as f64 / 10.0) * 2.0 * std::f64::consts::PI;
            let lat_offset = angle.sin() * expanded_height / 2.0;
            let lon_offset = angle.cos() * expanded_width / 2.0;

            let center = self.bounding_box.center();
            if let Ok(point) = GeoPoint::try_new(center.lat + lat_offset, center.lon + lon_offset) {
                let doc_id = (i + 3000) as u64;
                candidates.push((doc_id, point));
            }
        }

        candidates
    }

    /// Calculate relevance score for points within the bounding box.
    fn calculate_bounding_box_score(&self, point: &GeoPoint) -> f32 {
        let center = self.bounding_box.center();
        let (width, height) = self.bounding_box.dimensions();

        // Distance from center as a fraction of the bounding box diagonal
        let distance_to_center = center.distance_to(point);
        let diagonal_m = ((width * 111_000.0).powi(2) + (height * 111_000.0).powi(2)).sqrt();
        let normalized_distance = distance_to_center / diagonal_m;

        // Base score: higher for points closer to center
        let base_score = (1.0 - normalized_distance.min(1.0)) as f32;

        // Bonus for points near edges or corners (depending on use case)
        let edge_bonus = self.calculate_edge_proximity_bonus(point);

        // Corner bonus for strategic locations
        let corner_bonus = self.calculate_corner_bonus(point);

        (base_score + edge_bonus + corner_bonus).min(1.0)
    }

    /// Calculate bonus for points near edges of the bounding box.
    fn calculate_edge_proximity_bonus(&self, point: &GeoPoint) -> f32 {
        let (width, height) = self.bounding_box.dimensions();
        let edge_threshold = 0.1; // 10% of dimension

        let lat_distance_from_edge =
            (point.lat - self.bounding_box.min.lat).min(self.bounding_box.max.lat - point.lat);
        let lon_distance_from_edge =
            (point.lon - self.bounding_box.min.lon).min(self.bounding_box.max.lon - point.lon);

        let lat_edge_proximity = if lat_distance_from_edge < height * edge_threshold {
            0.05
        } else {
            0.0
        };
        let lon_edge_proximity = if lon_distance_from_edge < width * edge_threshold {
            0.05
        } else {
            0.0
        };

        lat_edge_proximity + lon_edge_proximity
    }

    /// Calculate bonus for points near corners of the bounding box.
    fn calculate_corner_bonus(&self, point: &GeoPoint) -> f32 {
        let bbox = &self.bounding_box;
        let corners = [
            bbox.min,                                  // SW
            GeoPoint::new(bbox.max.lat, bbox.min.lon), // NW
            bbox.max,                                  // NE
            GeoPoint::new(bbox.min.lat, bbox.max.lon), // SE
        ];

        let corner_threshold_m = 1_000.0; // Within 1 km (= 1 000 m) of corner
        let mut min_corner_distance = f64::INFINITY;

        for corner in &corners {
            let distance = point.distance_to(corner);
            min_corner_distance = min_corner_distance.min(distance);
        }

        if min_corner_distance < corner_threshold_m {
            0.1 // Corner bonus
        } else {
            0.0
        }
    }
}

impl Query for GeoBoundingBoxQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        let matches = self.find_matches(reader)?;
        Ok(Box::new(GeoMatcher::new(matches)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        let matches = self.find_matches(reader)?;
        Ok(Box::new(GeoScorer::new(matches, self.boost)))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        // Run the candidate scan once and share it (#996): the scorer's
        // per-doc center-distance scores derive from the same matches
        // the matcher iterates, so ranking is identical to building
        // them independently.
        let matches = self.find_matches(reader)?;
        Ok((
            Box::new(GeoMatcher::new(matches.clone())),
            Box::new(GeoScorer::new(matches, self.boost)),
        ))
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn description(&self) -> String {
        format!(
            "GeoBoundingBoxQuery(field: {}, bounds: {:?})",
            self.field, self.bounding_box
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        // A box the constructor accepts is never empty: one with zero width
        // or height matches the points on it, as `GeoBoundingBox::contains`
        // says (#1595). The search skips a query that is empty.
        Ok(false)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        let doc_count = reader.doc_count() as u32;
        Ok(doc_count as u64)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field + bounding box determine the matched set; boost is score-only
        // and excluded.
        Some(format!("geobbox|{:?}|{:?}", self.field, self.bounding_box))
    }
}

/// How near an edge a point is on it, in degrees (about 0.1 mm).
const ON_EDGE_DEGREES: f64 = 1e-9;

/// A polygon on the surface: an exterior ring of latitudes and longitudes,
/// with holes, for [`GeoPolygonQuery`] (Issue #1596).
///
/// Its edges are straight lines in latitude and longitude, as Lucene's
/// `Polygon`. It reads as Lucene's does on its boundary: a point on an edge
/// or vertex of the exterior is inside, and a point on an edge or vertex of
/// a hole is not, so a hole's boundary belongs to the hole.
///
/// An edge takes the shorter way between its two vertices: the way whose
/// longitudes differ by 180 degrees or less. An edge from 170 to -170 is
/// 20 degrees wide and crosses the antimeridian, so a polygon over the
/// Pacific needs no splitting; an edge wider than 180 degrees is written
/// with a vertex in between. A ring around a pole, whose edges add up to a
/// whole turn of longitude, and an edge across exactly 180 degrees, which
/// could go either way, are rejected.
///
/// A ring is closed if its last point is not its first. Self-intersection is
/// not checked, as in Lucene; a polygon that crosses itself matches by the
/// crossing number.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoPolygon {
    /// The exterior ring as given, closed.
    exterior: Vec<GeoPoint>,
    /// The holes as given, closed.
    holes: Vec<Vec<GeoPoint>>,
    /// Every ring as `[lat, lon]`, the exterior first, with the longitudes
    /// unwrapped: each edge's longitudes differ by at most 180 degrees, and
    /// may go past ±180. The exterior starts within `[-180, 180)`, and each
    /// hole lies with it.
    rings: Vec<Vec<[f64; 2]>>,
    /// The exterior's bounding box `[lat, lon]`, in unwrapped longitudes.
    min: [f64; 2],
    max: [f64; 2],
}

impl GeoPolygon {
    /// Create a polygon from its exterior ring and holes.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`](crate::error::LaurusError)
    /// if a ring has fewer than 3 distinct points, a coordinate is not a
    /// latitude in `[-90, 90]` or a longitude in `[-180, 180]`, an edge
    /// spans exactly 180 degrees of longitude, or a ring goes around a pole.
    pub fn new(exterior: Vec<GeoPoint>, holes: Vec<Vec<GeoPoint>>) -> Result<Self> {
        let exterior = close_ring(exterior, "the exterior")?;
        let holes = holes
            .into_iter()
            .enumerate()
            .map(|(i, hole)| close_ring(hole, &format!("hole {i}")))
            .collect::<Result<Vec<_>>>()?;

        let mut outer = unwrap_ring(&exterior, "the exterior")?;
        let (mut min, mut max) = ring_bounds(&outer);
        // Start the exterior within [-180, 180).
        let shift = 360.0 * ((min[1] + 180.0) / 360.0).floor();
        shift_ring(&mut outer, -shift);
        min[1] -= shift;
        max[1] -= shift;

        let mut rings = vec![outer];
        let center = (min[1] + max[1]) / 2.0;
        for (i, hole) in holes.iter().enumerate() {
            let mut ring = unwrap_ring(hole, &format!("hole {i}"))?;
            let (hole_min, hole_max) = ring_bounds(&ring);
            // Lay the hole where the exterior is: the shift by 360 degrees
            // that brings its center nearest the exterior's.
            let shift = 360.0 * (((hole_min[1] + hole_max[1]) / 2.0 - center) / 360.0).round();
            shift_ring(&mut ring, -shift);
            rings.push(ring);
        }

        Ok(GeoPolygon {
            exterior,
            holes,
            rings,
            min,
            max,
        })
    }

    /// The exterior ring, closed.
    pub fn exterior(&self) -> &[GeoPoint] {
        &self.exterior
    }

    /// The holes, each closed.
    pub fn holes(&self) -> &[Vec<GeoPoint>] {
        &self.holes
    }

    /// Whether `point` is inside the polygon: inside or on the exterior,
    /// and neither inside nor on a hole.
    pub fn contains(&self, point: &GeoPoint) -> bool {
        self.contains_lat_lon(point.lat, point.lon)
    }

    /// [`contains`](Self::contains) for a latitude and a longitude in
    /// `[-180, 180]`, which is tried at every shift of 360 degrees that
    /// reaches the polygon's unwrapped longitudes.
    fn contains_lat_lon(&self, lat: f64, lon: f64) -> bool {
        if lat < self.min[0] - ON_EDGE_DEGREES || lat > self.max[0] + ON_EDGE_DEGREES {
            return false;
        }
        [-360.0, 0.0, 360.0].iter().any(|shift| {
            let lon = lon + shift;
            lon >= self.min[1] - ON_EDGE_DEGREES
                && lon <= self.max[1] + ON_EDGE_DEGREES
                && self.contains_unwrapped(lat, lon)
        })
    }

    /// Whether the unwrapped `(lat, lon)` is inside the polygon.
    fn contains_unwrapped(&self, lat: f64, lon: f64) -> bool {
        ring_contains(&self.rings[0], lat, lon)
            && !self.rings[1..]
                .iter()
                .any(|hole| ring_contains(hole, lat, lon))
    }

    /// How a cell of the BKD, the box from `min` to `max` (`[lat, lon]`,
    /// longitudes within `[-180, 180]`), relates to the polygon: `Inside`
    /// when every point of it is inside, `Outside` when none is, and
    /// `Crosses` when an edge of a ring comes within [`ON_EDGE_DEGREES`] of
    /// it. Tried at every shift of 360 degrees: one shift with the cell
    /// inside is enough.
    fn relate(&self, min: &[f64], max: &[f64]) -> CellRelation {
        let mut crosses = false;
        for shift in [-360.0, 0.0, 360.0] {
            let cell = [min[0], min[1] + shift, max[0], max[1] + shift];
            match self.relate_unwrapped(cell) {
                CellRelation::Inside => return CellRelation::Inside,
                CellRelation::Crosses => crosses = true,
                CellRelation::Outside => {}
            }
        }
        if crosses {
            CellRelation::Crosses
        } else {
            CellRelation::Outside
        }
    }

    /// [`relate`](Self::relate) for a box in unwrapped longitudes,
    /// `[min_lat, min_lon, max_lat, max_lon]`.
    fn relate_unwrapped(&self, cell: [f64; 4]) -> CellRelation {
        // The box widened by the tolerance of an edge, so that no point
        // closer than that to an edge is left to the corner below.
        let near = [
            cell[0] - ON_EDGE_DEGREES,
            cell[1] - ON_EDGE_DEGREES,
            cell[2] + ON_EDGE_DEGREES,
            cell[3] + ON_EDGE_DEGREES,
        ];
        if near[2] < self.min[0]
            || near[0] > self.max[0]
            || near[3] < self.min[1]
            || near[1] > self.max[1]
        {
            return CellRelation::Outside;
        }
        let meets = self
            .rings
            .iter()
            .flat_map(|ring| ring.windows(2))
            .any(|edge| segment_meets_box(edge[0], edge[1], near));
        if meets {
            return CellRelation::Crosses;
        }
        // No edge comes near the box, so every point of it is on the same
        // side of every ring as its corner.
        if self.contains_unwrapped(cell[0], cell[1]) {
            CellRelation::Inside
        } else {
            CellRelation::Outside
        }
    }
}

/// Check the points of `ring` and close it: its last point is its first.
fn close_ring(mut ring: Vec<GeoPoint>, name: &str) -> Result<Vec<GeoPoint>> {
    for point in &ring {
        if !(-90.0..=90.0).contains(&point.lat) {
            return Err(crate::error::LaurusError::invalid_argument(format!(
                "{name} of the polygon has the latitude {}, which is not within [-90, 90]",
                point.lat
            )));
        }
        if !(-180.0..=180.0).contains(&point.lon) {
            return Err(crate::error::LaurusError::invalid_argument(format!(
                "{name} of the polygon has the longitude {}, which is not within [-180, 180]",
                point.lon
            )));
        }
    }
    if ring.first() != ring.last() {
        ring.push(ring[0]);
    }
    let mut distinct: Vec<GeoPoint> = Vec::new();
    for point in &ring[..ring.len().saturating_sub(1)] {
        if !distinct.contains(point) {
            distinct.push(*point);
        }
    }
    if distinct.len() < 3 {
        return Err(crate::error::LaurusError::invalid_argument(format!(
            "{name} of the polygon needs at least 3 distinct points, and has {}",
            distinct.len()
        )));
    }
    Ok(ring)
}

/// `ring` (closed) as `[lat, lon]` with its longitudes unwrapped, so that
/// each edge takes the shorter way.
fn unwrap_ring(ring: &[GeoPoint], name: &str) -> Result<Vec<[f64; 2]>> {
    let mut out = Vec::with_capacity(ring.len());
    let mut lon = ring[0].lon;
    out.push([ring[0].lat, lon]);
    for edge in ring.windows(2) {
        let mut delta = edge[1].lon - edge[0].lon;
        if delta > 180.0 {
            delta -= 360.0;
        } else if delta < -180.0 {
            delta += 360.0;
        }
        if delta.abs() == 180.0 {
            return Err(crate::error::LaurusError::invalid_argument(format!(
                "an edge of {name} of the polygon spans exactly 180 degrees of longitude, \
                 which could go either way; add a point between its ends"
            )));
        }
        lon += delta;
        out.push([edge[1].lat, lon]);
    }
    if (lon - out[0][1]).abs() > ON_EDGE_DEGREES {
        return Err(crate::error::LaurusError::invalid_argument(format!(
            "{name} of the polygon goes around a pole, which a polygon cannot hold"
        )));
    }
    // The closing point exactly on the first.
    let first = out[0];
    if let Some(last) = out.last_mut() {
        *last = first;
    }
    Ok(out)
}

/// The bounding box of `ring`, `[lat, lon]`.
fn ring_bounds(ring: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for point in ring {
        for dim in 0..2 {
            min[dim] = min[dim].min(point[dim]);
            max[dim] = max[dim].max(point[dim]);
        }
    }
    (min, max)
}

fn shift_ring(ring: &mut [[f64; 2]], lon: f64) {
    for point in ring {
        point[1] += lon;
    }
}

/// Whether `(lat, lon)` is inside the closed `ring` or on it.
fn ring_contains(ring: &[[f64; 2]], lat: f64, lon: f64) -> bool {
    let mut inside = false;
    for edge in ring.windows(2) {
        let (a, b) = (edge[0], edge[1]);
        if on_segment(a, b, lat, lon) {
            return true;
        }
        // The crossing number, with a ray toward greater longitudes.
        if (a[0] > lat) != (b[0] > lat) {
            let crossing = a[1] + (lat - a[0]) * (b[1] - a[1]) / (b[0] - a[0]);
            if lon < crossing {
                inside = !inside;
            }
        }
    }
    inside
}

/// Whether `(lat, lon)` is within [`ON_EDGE_DEGREES`] of the segment from
/// `a` to `b`.
fn on_segment(a: [f64; 2], b: [f64; 2], lat: f64, lon: f64) -> bool {
    if lat < a[0].min(b[0]) - ON_EDGE_DEGREES
        || lat > a[0].max(b[0]) + ON_EDGE_DEGREES
        || lon < a[1].min(b[1]) - ON_EDGE_DEGREES
        || lon > a[1].max(b[1]) + ON_EDGE_DEGREES
    {
        return false;
    }
    let (d_lat, d_lon) = (b[0] - a[0], b[1] - a[1]);
    let length = d_lat.hypot(d_lon);
    if length == 0.0 {
        return true;
    }
    let cross = d_lon * (lat - a[0]) - d_lat * (lon - a[1]);
    (cross / length).abs() <= ON_EDGE_DEGREES
}

/// Whether the segment from `a` to `b` (`[lat, lon]`) meets the closed box
/// `[min_lat, min_lon, max_lat, max_lon]`, by clipping it to the box
/// (Liang-Barsky).
fn segment_meets_box(a: [f64; 2], b: [f64; 2], cell: [f64; 4]) -> bool {
    let (d_lat, d_lon) = (b[0] - a[0], b[1] - a[1]);
    let (mut enter, mut leave) = (0.0_f64, 1.0_f64);
    for (p, q) in [
        (-d_lat, a[0] - cell[0]),
        (d_lat, cell[2] - a[0]),
        (-d_lon, a[1] - cell[1]),
        (d_lon, cell[3] - a[1]),
    ] {
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
    true
}

/// [`IntersectVisitor`] collecting the doc ids of the points inside a
/// [`GeoPolygon`]. The polygon's score is constant, so a cell inside it is
/// taken whole through `visit_inside`, without its coordinates.
#[derive(Debug)]
struct GeoPolygonVisitor<'a> {
    polygon: &'a GeoPolygon,
    hits: Vec<u64>,
    /// How many points were tested one by one, for the pruning test.
    #[cfg(test)]
    visited: usize,
}

impl<'a> GeoPolygonVisitor<'a> {
    fn new(polygon: &'a GeoPolygon) -> Self {
        GeoPolygonVisitor {
            polygon,
            hits: Vec::new(),
            #[cfg(test)]
            visited: 0,
        }
    }
}

impl IntersectVisitor for GeoPolygonVisitor<'_> {
    fn compare(&self, cell: &AABB) -> CellRelation {
        debug_assert_eq!(cell.num_dims(), 2, "GeoPolygonVisitor expects a 2D BKD");
        self.polygon.relate(cell.min(), cell.max())
    }

    fn visit_inside(&mut self, doc_id: u64) {
        self.hits.push(doc_id);
    }

    fn visit(&mut self, doc_id: u64, point: &[f64]) {
        #[cfg(test)]
        {
            self.visited += 1;
        }
        if self.polygon.contains_lat_lon(point[0], point[1]) {
            self.hits.push(doc_id);
        }
    }
}

/// A query for the documents with a point of a `Geo` field inside a
/// [`GeoPolygon`] (Issue #1596).
///
/// Every hit scores the boost (`1.0` unless set), as Lucene's polygon query
/// does: a polygon has no natural center to score by. A multi-valued field
/// matches when any of its points is inside.
///
/// # Example
///
/// ```rust
/// use laurus::lexical::query::GeoPolygonQuery;
///
/// // A square with a square hole, latitude first.
/// let query = GeoPolygonQuery::within_polygon(
///     "location",
///     &[(35.0, 139.0), (35.0, 140.0), (36.0, 140.0), (36.0, 139.0)],
///     &[&[(35.4, 139.4), (35.4, 139.6), (35.6, 139.6), (35.6, 139.4)]],
/// )
/// .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GeoPolygonQuery {
    /// Field containing geographical coordinates.
    field: String,
    /// The polygon.
    polygon: GeoPolygon,
    /// Boost factor for the query.
    boost: f32,
}

impl GeoPolygonQuery {
    /// Create a polygon query.
    pub fn new<F: Into<String>>(field: F, polygon: GeoPolygon) -> Self {
        GeoPolygonQuery {
            field: field.into(),
            polygon,
            boost: 1.0,
        }
    }

    /// Create a polygon query from `(lat, lon)` pairs: the exterior ring and
    /// the holes, each closed or not.
    ///
    /// # Errors
    ///
    /// Returns what [`GeoPolygon::new`] returns.
    pub fn within_polygon<F: Into<String>>(
        field: F,
        exterior: &[(f64, f64)],
        holes: &[&[(f64, f64)]],
    ) -> Result<Self> {
        let ring = |points: &[(f64, f64)]| -> Vec<GeoPoint> {
            points
                .iter()
                .map(|&(lat, lon)| GeoPoint { lat, lon })
                .collect()
        };
        let polygon = GeoPolygon::new(
            ring(exterior),
            holes.iter().map(|hole| ring(hole)).collect(),
        )?;
        Ok(GeoPolygonQuery::new(field, polygon))
    }

    /// Set the boost factor.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the polygon.
    pub fn polygon(&self) -> &GeoPolygon {
        &self.polygon
    }

    /// The ids of the documents with a point inside the polygon, sorted and
    /// once each.
    pub fn find_doc_ids(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<u64>> {
        let mut doc_ids = Vec::new();
        if let Some(bkd_tree) = reader.get_bkd_tree(&self.field)? {
            // One traversal, whichever side of the antimeridian a cell is
            // on: `GeoPolygon::relate` tries every shift of 360 degrees.
            let mut visitor = GeoPolygonVisitor::new(&self.polygon);
            bkd_tree.intersect(&mut visitor)?;
            doc_ids = visitor.hits;
        } else {
            // Without a BKD, the stored values of the documents present in
            // this reader.
            for doc_id in scan_doc_ids(reader)? {
                if let Some(doc) = reader.document(doc_id)?
                    && let Some(value) = doc.get_field(&self.field)
                    && stored_geo_points(value).any(|point| self.polygon.contains(&point))
                {
                    doc_ids.push(doc_id);
                }
            }
        }
        doc_ids.sort_unstable();
        doc_ids.dedup();
        Ok(doc_ids)
    }
}

impl Query for GeoPolygonQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(PreComputedMatcher::new(
            self.find_doc_ids(reader)?,
        )))
    }

    fn scorer(&self, _reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        Ok(Box::new(ConstantScorer::with_boost(1.0, self.boost)))
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
            "GeoPolygonQuery(field: {}, exterior: {:?}, holes: {:?})",
            self.field, self.polygon.exterior, self.polygon.holes
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        // A polygon the constructor accepts has a boundary, which it
        // matches, so it is never empty.
        Ok(false)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        Ok(reader.doc_count())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field and rings determine the matched set; the boost is
        // score-only and excluded.
        Some(format!(
            "geopolygon|{:?}|{:?}|{:?}",
            self.field, self.polygon.exterior, self.polygon.holes
        ))
    }
}

/// The radius of the sphere [`GeoPoint::distance_to`] measures on, in
/// metres: the mean radius of the Earth.
const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// At or below this `|a × b|`, with `a · b > 0`, the two ends of an edge
/// are one point (about 6 µm apart on the surface).
const SAME_POINT_SINE: f64 = 1e-12;

/// Below this `|a × b|`, with `a · b < 0`, the ends of an edge are within
/// about 640 m of antipodal: no single shortest arc joins antipodal points,
/// and the plane through ends this near them is too ill-conditioned.
const NEAR_ANTIPODAL_SINE: f64 = 1e-4;

type Vec3 = [f64; 3];

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// The unit vector of `point` on the sphere.
fn unit_vector(point: &GeoPoint) -> Vec3 {
    let (lat, lon) = (point.lat.to_radians(), point.lon.to_radians());
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}

/// The great circle of an edge of a [`GeoPolyline`] whose ends are apart.
#[derive(Debug, Clone, PartialEq)]
struct EdgeArc {
    /// The unit normal of the plane of the arc, the direction of `a × b`.
    normal: Vec3,
    /// `normal × a`: a point is past the start of the arc when it is on the
    /// non-negative side of this.
    after_start: Vec3,
    /// `b × normal`: a point is before the end of the arc when it is on the
    /// non-negative side of this.
    before_end: Vec3,
}

/// An edge of a [`GeoPolyline`]: the shorter great-circle arc from `start`
/// to `end`, or a point when they are one.
#[derive(Debug, Clone, PartialEq)]
struct PolylineEdge {
    start: GeoPoint,
    end: GeoPoint,
    /// `None` for an edge of length 0.
    arc: Option<EdgeArc>,
}

impl PolylineEdge {
    /// The edge from `start` to `end`, the `index`-th of its polyline.
    fn new(index: usize, start: GeoPoint, end: GeoPoint) -> Result<Self> {
        let (a, b) = (unit_vector(&start), unit_vector(&end));
        // `(b + a) × (b − a)` is `2 (a × b)`, and keeps a rounding error near
        // the machine epsilon for a short edge (`b − a` is exact) and for
        // one near antipodal (`b + a` is exact), where `a × b` loses it:
        // the idea of S2's `RobustCrossProd`.
        let doubled = cross(
            [b[0] + a[0], b[1] + a[1], b[2] + a[2]],
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
        );
        let length = norm(doubled);
        let (sine, cosine) = (length / 2.0, dot(a, b));
        let arc = if cosine > 0.0 && sine <= SAME_POINT_SINE {
            None
        } else if cosine < 0.0 && sine < NEAR_ANTIPODAL_SINE {
            return Err(crate::error::LaurusError::invalid_argument(format!(
                "edge {index} of the polyline, from ({}, {}) to ({}, {}), joins antipodal \
                 points or points within about 640 m of them, which no single shortest arc \
                 joins; add a vertex between them",
                start.lat, start.lon, end.lat, end.lon
            )));
        } else {
            let normal = [
                doubled[0] / length,
                doubled[1] / length,
                doubled[2] / length,
            ];
            Some(EdgeArc {
                normal,
                after_start: cross(normal, a),
                before_end: cross(b, normal),
            })
        };
        Ok(PolylineEdge { start, end, arc })
    }

    /// The distance in metres from `point`, whose unit vector is `p`, to
    /// the edge: to the arc's closest point when the foot of the
    /// perpendicular is on the arc, and to the nearer end in any case.
    fn distance_to(&self, point: &GeoPoint, p: Vec3) -> f64 {
        let ends = self
            .start
            .distance_to(point)
            .min(self.end.distance_to(point));
        match &self.arc {
            Some(arc) if dot(p, arc.after_start) >= 0.0 && dot(p, arc.before_end) >= 0.0 => {
                // The angle between `p` and the plane of the arc.
                let across = dot(p, arc.normal).abs();
                let along = norm(cross(p, arc.normal));
                ends.min(across.atan2(along) * EARTH_RADIUS_M)
            }
            _ => ends,
        }
    }

    /// Conservative lat/lon bounds of the points within `delta` radians of
    /// the edge on the sphere, padded by [`ON_EDGE_DEGREES`].
    fn bounds(&self, delta: f64) -> CapBounds {
        if delta >= std::f64::consts::PI {
            return CapBounds::whole_sphere();
        }
        let delta_deg = delta.to_degrees();
        let mut south = self.start.lat.min(self.end.lat);
        let mut north = self.start.lat.max(self.end.lat);
        if let Some(arc) = &self.arc {
            // The great circle reaches `acos(|n_z|)` at its northernmost
            // point, the north pole projected on its plane, `z − n_z n`.
            // That point is on the arc when it is on the non-negative side
            // of both `n × a` and `b × n`, and its dot product with them is
            // their `z` component; the southernmost point is its opposite.
            let top = arc.normal[2].abs().min(1.0).acos().to_degrees();
            if arc.after_start[2] >= 0.0 && arc.before_end[2] >= 0.0 {
                north = north.max(top);
            }
            if arc.after_start[2] <= 0.0 && arc.before_end[2] <= 0.0 {
                south = south.min(-top);
            }
        }
        let min_lat = (south - delta_deg - ON_EDGE_DEGREES).max(-90.0);
        let max_lat = (north + delta_deg + ON_EDGE_DEGREES).min(90.0);
        let mut d_lon = self.end.lon - self.start.lon;
        if d_lon > 180.0 {
            d_lon -= 360.0;
        } else if d_lon < -180.0 {
            d_lon += 360.0;
        }
        let highest = south.abs().max(north.abs());
        if highest + delta_deg >= 90.0 - ON_EDGE_DEGREES || d_lon.abs() >= 180.0 - ON_EDGE_DEGREES {
            // The bounds reach a pole, or the edge goes over one: every
            // meridian passes near it.
            return CapBounds {
                min_lat,
                max_lat,
                lon_ranges: vec![(-180.0, 180.0)],
            };
        }
        // Along an arc that does not pass a pole the longitude changes
        // monotonically, the shorter way. A cap of radius δ at latitude φ
        // spans `asin(sin δ / cos φ)` either way, which grows with |φ|.
        let widen = (delta.sin() / highest.to_radians().cos())
            .min(1.0)
            .asin()
            .to_degrees()
            + ON_EDGE_DEGREES;
        let (west, east) = if d_lon >= 0.0 {
            (self.start.lon, self.start.lon + d_lon)
        } else {
            (self.start.lon + d_lon, self.start.lon)
        };
        CapBounds::with_lon_span(min_lat, max_lat, west - widen, east + widen)
    }
}

/// A polyline on the surface: vertices of latitudes and longitudes joined in
/// order, for [`GeoPolylineQuery`] (Issue #1596).
///
/// Each edge is the shorter great-circle arc between its vertices, so the
/// distance from a point to the polyline is on the surface, as
/// [`GeoPoint::distance_to`]'s: to the closest point of an arc, or to an end
/// past it, so the ends and corners are round. A polygon's edges, unlike
/// these, are straight lines in latitude and longitude (see
/// [`GeoPolygon`]): the arc between two points at 10°N, 170 degrees of
/// longitude apart, rises to about 63.7°N midway.
///
/// A single vertex is a point, and repeated vertices are fine: an edge of
/// length 0 is its point. An edge between antipodal points, or points
/// within about 640 m of them, is rejected, since no single shortest arc
/// joins them; an edge across the antimeridian needs nothing special.
#[derive(Debug, Clone, PartialEq)]
pub struct GeoPolyline {
    /// The vertices as given.
    vertices: Vec<GeoPoint>,
    /// The edges in order; a single vertex is one edge of length 0.
    edges: Vec<PolylineEdge>,
}

impl GeoPolyline {
    /// Create a polyline through `vertices`, in order.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`](crate::error::LaurusError)
    /// if there is no vertex, a coordinate is not a latitude in `[-90, 90]`
    /// or a longitude in `[-180, 180]`, or an edge joins antipodal points
    /// or points within about 640 m of them.
    pub fn new(vertices: Vec<GeoPoint>) -> Result<Self> {
        if vertices.is_empty() {
            return Err(crate::error::LaurusError::invalid_argument(
                "a polyline needs at least one vertex".to_string(),
            ));
        }
        for (i, vertex) in vertices.iter().enumerate() {
            if !(-90.0..=90.0).contains(&vertex.lat) {
                return Err(crate::error::LaurusError::invalid_argument(format!(
                    "vertex {i} of the polyline has the latitude {}, which is not within [-90, 90]",
                    vertex.lat
                )));
            }
            if !(-180.0..=180.0).contains(&vertex.lon) {
                return Err(crate::error::LaurusError::invalid_argument(format!(
                    "vertex {i} of the polyline has the longitude {}, which is not within \
                     [-180, 180]",
                    vertex.lon
                )));
            }
        }
        let edges = if let [vertex] = vertices.as_slice() {
            vec![PolylineEdge::new(0, *vertex, *vertex)?]
        } else {
            vertices
                .windows(2)
                .enumerate()
                .map(|(i, edge)| PolylineEdge::new(i, edge[0], edge[1]))
                .collect::<Result<Vec<_>>>()?
        };
        Ok(GeoPolyline { vertices, edges })
    }

    /// The vertices, in order.
    pub fn vertices(&self) -> &[GeoPoint] {
        &self.vertices
    }

    /// The distance in metres from `point` to the polyline, on the sphere
    /// of [`GeoPoint::distance_to`]. For a single vertex it is exactly
    /// `vertex.distance_to(point)`.
    pub fn distance_to(&self, point: &GeoPoint) -> f64 {
        let p = unit_vector(point);
        self.edges
            .iter()
            .map(|edge| edge.distance_to(point, p))
            .fold(f64::INFINITY, f64::min)
    }

    /// For each edge, the bounds of the points within `distance_m` of it.
    fn edge_bounds(&self, distance_m: f64) -> Vec<CapBounds> {
        let delta = distance_m / EARTH_RADIUS_M;
        self.edges.iter().map(|edge| edge.bounds(delta)).collect()
    }

    /// The distance from `point` to the edges whose `bounds` hold it, or
    /// `None` when none does: such a point is farther from every edge than
    /// the distance the bounds were made for.
    fn distance_within(&self, point: &GeoPoint, bounds: &[CapBounds]) -> Option<f64> {
        let p = unit_vector(point);
        self.edges
            .iter()
            .zip(bounds)
            .filter(|(_, bounds)| bounds.contains(point))
            .map(|(edge, _)| edge.distance_to(point, p))
            .reduce(f64::min)
    }
}

/// [`IntersectVisitor`] collecting the points within a distance of a
/// [`GeoPolyline`], with their distances. The score needs every point's
/// distance, so `compare` never returns [`CellRelation::Inside`], as
/// [`GeoBoxPointsVisitor`] does not.
#[derive(Debug)]
struct GeoPolylineVisitor<'a> {
    polyline: &'a GeoPolyline,
    /// The bounds of each edge, from [`GeoPolyline::edge_bounds`].
    bounds: &'a [CapBounds],
    /// The latitudes every bound lies within.
    min_lat: f64,
    max_lat: f64,
    distance_m: f64,
    /// `(doc_id, point, distance)` of every point within the distance.
    hits: Vec<(u64, GeoPoint, f64)>,
    /// How many points were tested one by one, for the pruning test.
    #[cfg(test)]
    visited: usize,
}

impl<'a> GeoPolylineVisitor<'a> {
    fn new(polyline: &'a GeoPolyline, bounds: &'a [CapBounds], distance_m: f64) -> Self {
        GeoPolylineVisitor {
            polyline,
            bounds,
            min_lat: bounds
                .iter()
                .map(|b| b.min_lat)
                .fold(f64::INFINITY, f64::min),
            max_lat: bounds
                .iter()
                .map(|b| b.max_lat)
                .fold(f64::NEG_INFINITY, f64::max),
            distance_m,
            hits: Vec::new(),
            #[cfg(test)]
            visited: 0,
        }
    }
}

impl IntersectVisitor for GeoPolylineVisitor<'_> {
    fn compare(&self, cell: &AABB) -> CellRelation {
        debug_assert_eq!(cell.num_dims(), 2, "GeoPolylineVisitor expects a 2D BKD");
        let (min, max) = (cell.min(), cell.max());
        if max[0] < self.min_lat || min[0] > self.max_lat {
            return CellRelation::Outside;
        }
        if self.bounds.iter().any(|bounds| bounds.meets(min, max)) {
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
        #[cfg(test)]
        {
            self.visited += 1;
        }
        let point = GeoPoint::new(point[0], point[1]);
        if let Some(distance) = self.polyline.distance_within(&point, self.bounds)
            && distance <= self.distance_m
        {
            self.hits.push((doc_id, point, distance));
        }
    }
}

/// A query for the documents with a point of a `Geo` field within a
/// distance of a [`GeoPolyline`] (Issue #1596): along a road, a railway, a
/// river or a route.
///
/// The distance is measured on the surface, as [`GeoDistanceQuery`]'s, and
/// the score is the same: `1 - d / distance_m`, `d` being the distance of
/// the document's closest point to the polyline. A single vertex finds the
/// documents, and gives the scores, that [`GeoDistanceQuery`] does.
///
/// # Example
///
/// ```rust
/// use laurus::lexical::query::GeoPolylineQuery;
///
/// // Within 2 km of a route from Tokyo through Nagoya to Osaka, latitude
/// // first.
/// let query = GeoPolylineQuery::within_distance(
///     "location",
///     &[(35.68, 139.76), (35.17, 136.91), (34.69, 135.50)],
///     2_000.0,
/// )
/// .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GeoPolylineQuery {
    /// Field containing geographical coordinates.
    field: String,
    /// The polyline.
    polyline: GeoPolyline,
    /// The maximum distance from the polyline, in metres.
    distance_m: f64,
    /// Boost factor for the query.
    boost: f32,
}

impl GeoPolylineQuery {
    /// Create a query for the points within `distance_m` metres of
    /// `polyline`.
    pub fn new<F: Into<String>>(field: F, polyline: GeoPolyline, distance_m: f64) -> Self {
        GeoPolylineQuery {
            field: field.into(),
            polyline,
            distance_m,
            boost: 1.0,
        }
    }

    /// Create a query from `(lat, lon)` vertices, in order, and a distance
    /// in metres.
    ///
    /// # Errors
    ///
    /// Returns what [`GeoPolyline::new`] returns.
    pub fn within_distance<F: Into<String>>(
        field: F,
        vertices: &[(f64, f64)],
        distance_m: f64,
    ) -> Result<Self> {
        let polyline = GeoPolyline::new(
            vertices
                .iter()
                .map(|&(lat, lon)| GeoPoint { lat, lon })
                .collect(),
        )?;
        Ok(GeoPolylineQuery::new(field, polyline, distance_m))
    }

    /// Set the boost factor.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the polyline.
    pub fn polyline(&self) -> &GeoPolyline {
        &self.polyline
    }

    /// Get the maximum distance from the polyline, in metres.
    pub fn distance_m(&self) -> f64 {
        self.distance_m
    }

    /// The documents with a point within the distance, each once with its
    /// closest point, closest first.
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<GeoMatch>> {
        let mut found: Vec<(u64, GeoPoint, f64)> = Vec::new();
        // A negative or NaN distance holds nothing, as for
        // `GeoDistanceQuery`.
        if self.distance_m >= 0.0 {
            let bounds = self.polyline.edge_bounds(self.distance_m);
            if let Some(bkd_tree) = reader.get_bkd_tree(&self.field)? {
                let mut visitor = GeoPolylineVisitor::new(&self.polyline, &bounds, self.distance_m);
                bkd_tree.intersect(&mut visitor)?;
                found = visitor.hits;
            } else {
                // Without a BKD, the stored values of the documents present
                // in this reader; a multi-valued field gives every point.
                for doc_id in scan_doc_ids(reader)? {
                    if let Some(doc) = reader.document(doc_id)?
                        && let Some(value) = doc.get_field(&self.field)
                    {
                        for point in stored_geo_points(value) {
                            if let Some(distance) = self.polyline.distance_within(&point, &bounds)
                                && distance <= self.distance_m
                            {
                                found.push((doc_id, point, distance));
                            }
                        }
                    }
                }
            }
        }
        let matches = found
            .into_iter()
            .map(|(doc_id, point, distance)| GeoMatch {
                doc_id,
                point,
                distance_m: distance,
                relevance_score: distance_score(distance, self.distance_m),
            })
            .collect();
        Ok(closest_per_document(matches))
    }
}

impl Query for GeoPolylineQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(GeoMatcher::new(self.find_matches(reader)?)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        Ok(Box::new(GeoScorer::new(
            self.find_matches(reader)?,
            self.boost,
        )))
    }

    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        // One traversal for both, as `GeoDistanceQuery` does (#996).
        let matches = self.find_matches(reader)?;
        Ok((
            Box::new(GeoMatcher::new(matches.clone())),
            Box::new(GeoScorer::new(matches, self.boost)),
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
            "GeoPolylineQuery(field: {}, vertices: {:?}, distance: {}m)",
            self.field, self.polyline.vertices, self.distance_m
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.distance_m <= 0.0)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        // The same shape as `GeoDistanceQuery::cost`.
        Ok(reader.doc_count().saturating_mul(2))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field, vertices and distance determine the matched set; the boost
        // is score-only and excluded.
        Some(format!(
            "geopolyline|{:?}|{:?}|{:?}",
            self.field, self.polyline.vertices, self.distance_m
        ))
    }
}

/// A match found by geographical search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoMatch {
    /// Document ID
    pub doc_id: u64,
    /// Geographical point of the document
    pub point: GeoPoint,
    /// Distance from query center in meters
    pub distance_m: f64,
    /// Relevance score based on distance
    pub relevance_score: f32,
}

/// Matcher for geographical queries.
#[derive(Debug)]
pub struct GeoMatcher {
    /// Matching documents in order of relevance (distance-sorted)
    matches: Vec<GeoMatch>,
    /// Current iteration position
    current_index: usize,
}

impl GeoMatcher {
    /// Create a new geo matcher.
    ///
    /// Sorts matches into ascending document-id order: `Matcher` iteration
    /// order is a trait contract — boolean conjunction / disjunction drivers
    /// interleave `skip_to` calls and assume monotonically increasing doc
    /// ids. Distance-based ranking is the scorer's concern ([`GeoScorer`]
    /// looks scores up by doc id), so standalone geo ranking is unaffected.
    pub fn new(mut matches: Vec<GeoMatch>) -> Self {
        matches.sort_by_key(|m| m.doc_id);

        GeoMatcher {
            matches,
            current_index: 0,
        }
    }
}

impl Matcher for GeoMatcher {
    fn doc_id(&self) -> u64 {
        if self.current_index >= self.matches.len() {
            u64::MAX
        } else {
            self.matches[self.current_index].doc_id
        }
    }

    fn next(&mut self) -> Result<bool> {
        self.current_index += 1;
        if self.current_index < self.matches.len() {
            Ok(true)
        } else {
            self.current_index = self.matches.len();
            Ok(false)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        // Forward scan for the first doc id >= target; valid because
        // `GeoMatcher::new` sorts matches by ascending doc id.
        while self.current_index < self.matches.len() {
            let doc_id = self.matches[self.current_index].doc_id;
            if doc_id >= target {
                return Ok(true);
            }
            self.current_index += 1;
        }
        Ok(false)
    }

    fn cost(&self) -> u64 {
        self.matches.len() as u64
    }

    fn is_exhausted(&self) -> bool {
        self.current_index >= self.matches.len()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Scorer for geographical queries.
#[derive(Debug)]
pub struct GeoScorer {
    /// Document scores based on geographical relevance
    doc_scores: HashMap<u64, f32>,
    /// Query boost factor
    boost: f32,
}

impl GeoScorer {
    /// Create a new geo scorer.
    pub fn new(matches: Vec<GeoMatch>, boost: f32) -> Self {
        let mut doc_scores = HashMap::new();

        for geo_match in matches {
            doc_scores.insert(geo_match.doc_id, geo_match.relevance_score);
        }

        GeoScorer { doc_scores, boost }
    }
}

impl Scorer for GeoScorer {
    fn score(&self, doc_id: u64, _term_freq: f32, _field_length: Option<f32>) -> f32 {
        self.doc_scores.get(&doc_id).unwrap_or(&0.0) * self.boost
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        self.doc_scores
            .values()
            .fold(0.0_f32, |max, &score| max.max(score))
            * self.boost
    }

    fn name(&self) -> &'static str {
        "GeoScorer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `GeoMatch` whose distance deliberately disagrees with its
    /// doc-id order, to exercise the matcher's re-sort.
    fn geo_match(doc_id: u64, distance_m: f64) -> GeoMatch {
        GeoMatch {
            doc_id,
            point: GeoPoint::new(0.0, 0.0),
            distance_m,
            relevance_score: 1.0,
        }
    }

    /// `Matcher` iteration must be in ascending doc-id order even when the
    /// input list is distance-ordered (regression for hits dropped from
    /// boolean conjunctions, GitHub issue #982).
    #[test]
    fn test_geo_matcher_iterates_in_doc_id_order() {
        // Distance order: 7, 1, 4 — doc-id order must win.
        let mut matcher = GeoMatcher::new(vec![
            geo_match(7, 10.0),
            geo_match(1, 20.0),
            geo_match(4, 30.0),
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
    fn test_geo_matcher_skip_to_honors_doc_id_order() {
        let mut matcher = GeoMatcher::new(vec![
            geo_match(7, 10.0),
            geo_match(1, 20.0),
            geo_match(4, 30.0),
        ]);

        assert!(matcher.skip_to(2).unwrap());
        assert_eq!(matcher.doc_id(), 4);
        assert!(matcher.skip_to(4).unwrap());
        assert_eq!(matcher.doc_id(), 4);
        assert!(matcher.skip_to(5).unwrap());
        assert_eq!(matcher.doc_id(), 7);
        assert!(!matcher.skip_to(8).unwrap());
        assert!(matcher.is_exhausted());
    }

    #[test]
    fn test_geo_point_creation() {
        let point = GeoPoint::try_new(40.7128, -74.0060).unwrap(); // New York City
        assert_eq!(point.lat, 40.7128);
        assert_eq!(point.lon, -74.0060);

        // Test invalid coordinates
        assert!(GeoPoint::try_new(91.0, 0.0).is_err()); // Invalid latitude
        assert!(GeoPoint::try_new(0.0, 181.0).is_err()); // Invalid longitude
    }

    #[test]
    fn test_geo_distance_calculation() {
        let nyc = GeoPoint::try_new(40.7128, -74.0060).unwrap();
        let la = GeoPoint::try_new(34.0522, -118.2437).unwrap();

        let distance = nyc.distance_to(&la);
        // Distance between NYC and LA is approximately 3 944 km (3 944 000 m)
        assert!((distance - 3_944_000.0).abs() < 100_000.0); // Allow ~100 km tolerance
    }

    #[test]
    fn test_geo_bearing() {
        let nyc = GeoPoint::try_new(40.7128, -74.0060).unwrap();
        let la = GeoPoint::try_new(34.0522, -118.2437).unwrap();

        let bearing = nyc.bearing_to(&la);
        // Bearing from NYC to LA should be roughly west (around 270 degrees)
        assert!(bearing > 200.0 && bearing < 300.0);
    }

    #[test]
    fn test_geo_bounding_box() {
        let min = GeoPoint::try_new(40.0, -75.0).unwrap();
        let max = GeoPoint::try_new(41.0, -74.0).unwrap();
        let bbox = GeoBoundingBox::new(min, max).unwrap();

        let inside_point = GeoPoint::try_new(40.5, -74.5).unwrap();
        let outside_point = GeoPoint::try_new(42.0, -73.0).unwrap();

        assert!(bbox.contains(&inside_point));
        assert!(!bbox.contains(&outside_point));

        let center = bbox.center();
        assert_eq!(center.lat, 40.5);
        assert_eq!(center.lon, -74.5);
    }

    /// #1595: the longitudes of a box as closed ranges: its own, two for a
    /// box that crosses the antimeridian, and the other sign of ±180° for an
    /// edge that lies on it (not for the whole circle).
    #[test]
    fn test_geo_bounding_box_lon_ranges() {
        let ranges = |min_lon: f64, max_lon: f64| {
            GeoBoundingBox::new(GeoPoint::new(0.0, min_lon), GeoPoint::new(1.0, max_lon))
                .unwrap()
                .lon_ranges()
                .iter()
                .collect::<Vec<_>>()
        };
        assert_eq!(ranges(-75.0, -74.0), [(-75.0, -74.0)]);
        assert_eq!(ranges(175.0, -175.0), [(175.0, 180.0), (-180.0, -175.0)]);
        assert_eq!(ranges(175.0, 180.0), [(175.0, 180.0), (-180.0, -180.0)]);
        assert_eq!(ranges(-180.0, -175.0), [(-180.0, -175.0), (180.0, 180.0)]);
        assert_eq!(ranges(-180.0, 180.0), [(-180.0, 180.0)]);
        // Zero width on the antimeridian, and the arc that starts and ends
        // on it.
        assert_eq!(ranges(180.0, 180.0), [(180.0, 180.0), (-180.0, -180.0)]);
        assert_eq!(ranges(-180.0, -180.0), [(-180.0, -180.0), (180.0, 180.0)]);
        assert_eq!(ranges(180.0, -180.0), [(180.0, 180.0), (-180.0, -180.0)]);
    }

    /// #1595: the center of a box that crosses the antimeridian is the
    /// midpoint of its arc, inside the box, and its width is the arc.
    #[test]
    fn test_geo_bounding_box_center_and_dimensions_across_the_antimeridian() {
        for (min_lon, max_lon, width) in [
            (175.0, -175.0, 10.0),
            (170.0, -100.0, 90.0),
            (100.0, -100.0, 160.0),
            (0.5, -0.5, 359.0),
            (180.0, -175.0, 5.0),
            (175.0, -180.0, 5.0),
            // Boxes that do not cross it keep their values.
            (-75.0, -74.0, 1.0),
            (139.0, 139.0, 0.0),
        ] {
            let bbox =
                GeoBoundingBox::new(GeoPoint::new(-20.0, min_lon), GeoPoint::new(-15.0, max_lon))
                    .unwrap();
            let (got_width, height) = bbox.dimensions();
            assert!(
                (got_width - width).abs() < 1e-9,
                "width of [{min_lon}, {max_lon}]: {got_width}"
            );
            assert_eq!(height, 5.0);

            let center = bbox.center();
            assert!(
                bbox.contains(&center),
                "center {center:?} of [{min_lon}, {max_lon}] must be inside it"
            );
            // Half the arc east of the western edge.
            let east_of_min = (center.lon - min_lon).rem_euclid(360.0);
            assert!(
                (east_of_min - width / 2.0).abs() < 1e-9,
                "center {center:?} of [{min_lon}, {max_lon}]"
            );
            let max_distance = bbox.max_distance_from_center();
            assert!(max_distance > 0.0 && max_distance <= std::f64::consts::PI * 6_371_000.0);
        }
    }

    /// #1595: no box the constructor accepts is empty, not even one with
    /// zero width, which the search used to skip.
    #[test]
    fn test_geo_bounding_box_query_is_never_empty() {
        let reader = BkdBackedReader::new(Vec::new());
        for (min_lat, min_lon, max_lat, max_lon) in [
            (40.0, -75.0, 41.0, -74.0),
            (-20.0, 175.0, -15.0, -175.0),
            (30.0, 139.0, 40.0, 139.0),
            (35.0, 130.0, 35.0, 150.0),
            (35.0, 139.0, 35.0, 139.0),
        ] {
            let query = GeoBoundingBoxQuery::within_bounding_box(
                "location", min_lat, min_lon, max_lat, max_lon,
            )
            .unwrap();
            assert!(!query.is_empty(&reader).unwrap());
        }
    }

    #[test]
    fn test_geo_distance_query() {
        let center = GeoPoint::try_new(40.7128, -74.0060).unwrap();
        let query = GeoDistanceQuery::new("location", center, 10_000.0).with_boost(1.5);

        assert_eq!(query.field(), "location");
        assert_eq!(query.center(), center);
        assert_eq!(query.distance_m(), 10_000.0);
        assert_eq!(query.boost(), 1.5);
    }

    #[test]
    fn test_geo_distance_scoring() {
        let center = GeoPoint::try_new(0.0, 0.0).unwrap();
        let query = GeoDistanceQuery::new("location", center, 10_000.0);

        // Test scoring at different distances (meters)
        assert_eq!(query.calculate_distance_score(0.0), 1.0); // At center
        assert_eq!(query.calculate_distance_score(5_000.0), 0.5); // Half distance
        assert_eq!(query.calculate_distance_score(10_000.0), 0.0); // At max distance
        assert_eq!(query.calculate_distance_score(15_000.0), 0.0); // Beyond max distance
    }

    #[test]
    fn test_geo_bounding_box_query() {
        let min = GeoPoint::try_new(40.0, -75.0).unwrap();
        let max = GeoPoint::try_new(41.0, -74.0).unwrap();
        let bbox = GeoBoundingBox::new(min, max).unwrap();
        let query = GeoBoundingBoxQuery::new("location", bbox);

        assert_eq!(query.field(), "location");
        assert_eq!(query.bounding_box().min, min);
        assert_eq!(query.bounding_box().max, max);
    }

    #[test]
    fn test_geo_distance_query_within_radius_factory() {
        let query =
            GeoDistanceQuery::within_radius("location", 40.7128, -74.0060, 10_000.0).unwrap();

        assert_eq!(query.field(), "location");
        assert_eq!(query.center().lat, 40.7128);
        assert_eq!(query.center().lon, -74.0060);
        assert_eq!(query.distance_m(), 10_000.0);
    }

    #[test]
    fn test_geo_distance_query_within_radius_invalid_lat() {
        // Latitude outside [-90, 90] is rejected by GeoPoint::try_new.
        let err = GeoDistanceQuery::within_radius("location", 95.0, 0.0, 10_000.0).unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("lat"));
    }

    #[test]
    fn test_geo_bounding_box_query_within_bounding_box_factory() {
        let query =
            GeoBoundingBoxQuery::within_bounding_box("location", 40.0, -75.0, 41.0, -74.0).unwrap();

        assert_eq!(query.field(), "location");
        assert_eq!(query.bounding_box().min.lat, 40.0);
        assert_eq!(query.bounding_box().min.lon, -75.0);
        assert_eq!(query.bounding_box().max.lat, 41.0);
        assert_eq!(query.bounding_box().max.lon, -74.0);
    }

    #[test]
    fn test_geo_bounding_box_query_within_bounding_box_inverted() {
        // min > max is rejected by GeoBoundingBox::new.
        let err = GeoBoundingBoxQuery::within_bounding_box("location", 50.0, 0.0, 40.0, 10.0)
            .unwrap_err();
        // Error mentions either bounding-box or min/max.
        let msg = format!("{err}").to_lowercase();
        assert!(msg.contains("bounding") || msg.contains("min") || msg.contains("max"));
    }

    #[test]
    fn test_geo_matcher() {
        let matches = vec![
            GeoMatch {
                doc_id: 3,
                point: GeoPoint::try_new(0.0, 0.0).unwrap(),
                distance_m: 1_000.0,
                relevance_score: 0.9,
            },
            GeoMatch {
                doc_id: 1,
                point: GeoPoint::try_new(0.0, 0.0).unwrap(),
                distance_m: 2_000.0,
                relevance_score: 0.8,
            },
        ];

        let mut matcher = GeoMatcher::new(matches);

        // Iterates in ascending doc-id order (the `Matcher` contract used
        // by boolean drivers) — distance ranking is the scorer's job.
        assert_eq!(matcher.doc_id(), 1);

        assert!(matcher.next().unwrap()); // Move to next
        assert_eq!(matcher.doc_id(), 3);

        assert!(!matcher.next().unwrap()); // No more documents
    }

    #[test]
    fn test_geo_scorer() {
        let matches = vec![GeoMatch {
            doc_id: 1,
            point: GeoPoint::try_new(0.0, 0.0).unwrap(),
            distance_m: 1_000.0,
            relevance_score: 0.9,
        }];

        let scorer = GeoScorer::new(matches, 2.0);

        assert_eq!(scorer.score(1, 1.0, None), 0.9 * 2.0);
        assert_eq!(scorer.score(999, 1.0, None), 0.0); // Non-existent document
        assert_eq!(scorer.max_score(), 0.9 * 2.0);
        assert_eq!(scorer.name(), "GeoScorer");
    }

    #[test]
    fn test_enhanced_distance_scoring() {
        let center = GeoPoint::try_new(40.7128, -74.0060).unwrap(); // NYC
        let query = GeoDistanceQuery::new("location", center, 10_000.0);

        // Test point very close to centre (50 m away)
        let close_point = GeoPoint::try_new(40.7130, -74.0062).unwrap();
        let close_score = query.calculate_distance_score_enhanced(50.0, &close_point);

        // Test point at moderate distance (1 km)
        let mid_point = GeoPoint::try_new(40.7200, -74.0100).unwrap();
        let mid_score = query.calculate_distance_score_enhanced(1_000.0, &mid_point);

        // Test point at almost max distance (9 km)
        let far_point = GeoPoint::try_new(40.8000, -74.1000).unwrap();
        let far_score = query.calculate_distance_score_enhanced(9_000.0, &far_point);

        // Scores should decrease with distance, and close points should get precision bonus
        assert!(close_score > mid_score);
        assert!(mid_score > far_score);
        assert!(close_score > 0.9); // Should get precision bonus
    }

    #[test]
    fn test_bounding_box_enhanced_functionality() {
        let min = GeoPoint::try_new(40.0, -75.0).unwrap();
        let max = GeoPoint::try_new(41.0, -74.0).unwrap();
        let bbox = GeoBoundingBox::new(min, max).unwrap();
        let query = GeoBoundingBoxQuery::new("location", bbox);

        // Test that generated candidates include points within bounds
        let candidates = query.generate_bounding_box_candidates();
        assert!(!candidates.is_empty());

        // Test that some candidates are within the bounding box
        let within_count = candidates
            .iter()
            .filter(|(_, point)| query.bounding_box().contains(point))
            .count();
        assert!(within_count > 0);

        // Test scoring for points within bounding box
        let center_point = query.bounding_box().center();
        let center_score = query.calculate_bounding_box_score(&center_point);

        let corner_point = query.bounding_box().min;
        let corner_score = query.calculate_bounding_box_score(&corner_point);

        // Center should generally score higher than corners
        assert!(center_score >= corner_score);
    }

    #[test]
    fn test_spatial_bounding_box_creation() {
        let center = GeoPoint::try_new(40.7128, -74.0060).unwrap(); // NYC
        let query = GeoDistanceQuery::new("location", center, 5_000.0); // 5 km radius

        let bounds = query.cap_bounds();

        // Check that the bounds contain the center
        assert!(bounds.contains(&center));

        // Check that the bounds are roughly the right size
        assert_eq!(bounds.lon_ranges.len(), 1, "no antimeridian crossing");
        let (min_lon, max_lon) = bounds.lon_ranges[0];
        let width = max_lon - min_lon;
        let height = bounds.max_lat - bounds.min_lat;
        assert!(width > 0.0 && width < 1.0); // Should be less than 1 degree
        assert!(height > 0.0 && height < 1.0);

        // The center should be approximately in the middle of the bounds
        // (within ~1 km).
        let bounds_center = GeoPoint::new(
            (bounds.min_lat + bounds.max_lat) / 2.0,
            (min_lon + max_lon) / 2.0,
        );
        let center_distance = center.distance_to(&bounds_center);
        assert!(center_distance < 1_000.0);
    }

    #[test]
    fn test_geographic_relevance_calculation() {
        let center = GeoPoint::try_new(40.7128, -74.0060).unwrap();
        let query = GeoDistanceQuery::new("location", center, 10_000.0);

        // Test temperate zone bonus
        let temperate_point = GeoPoint::try_new(45.0, 0.0).unwrap(); // Temperate zone
        let tropical_point = GeoPoint::try_new(10.0, 0.0).unwrap(); // Tropical zone

        let temperate_bonus = query.calculate_geographic_relevance(&temperate_point);
        let tropical_bonus = query.calculate_geographic_relevance(&tropical_point);

        assert!(temperate_bonus > tropical_bonus);

        // Test equator bonus
        let equator_point = GeoPoint::try_new(2.0, 0.0).unwrap(); // Near equator
        let non_equator_point = GeoPoint::try_new(45.0, 0.0).unwrap();

        let equator_geo_bonus = query.calculate_geographic_relevance(&equator_point);
        let non_equator_geo_bonus = query.calculate_geographic_relevance(&non_equator_point);

        // Both should have some bonus, but for different reasons
        assert!(equator_geo_bonus > 0.0);
        assert!(non_equator_geo_bonus > 0.0);
    }

    /// Reader with a configurable stored-doc id set that counts
    /// `document()` probes. `max_doc()` deliberately disagrees with the
    /// id set to model sparse id spaces (post-merge segments, per-segment
    /// fanout views reporting the global `max_doc`) — see #996 / #994.
    /// Every present doc carries a `location` geo point near Tokyo.
    #[derive(Debug)]
    struct GeoCountingReader {
        ids: Vec<u64>,
        max_doc: u64,
        document_calls: std::sync::atomic::AtomicU64,
    }

    impl GeoCountingReader {
        fn new(ids: Vec<u64>, max_doc: u64) -> Self {
            GeoCountingReader {
                ids,
                max_doc,
                document_calls: std::sync::atomic::AtomicU64::new(0),
            }
        }

        fn calls(&self) -> u64 {
            self.document_calls
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl LexicalIndexReader for GeoCountingReader {
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
            doc_id: u64,
        ) -> crate::error::Result<Option<crate::lexical::core::document::Document>> {
            self.document_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if self.ids.contains(&doc_id) {
                let doc = crate::data::Document::builder()
                    .add_field(
                        "location",
                        crate::data::DataValue::Geo(GeoPoint::new(35.68, 139.76)),
                    )
                    .build();
                Ok(Some(doc))
            } else {
                Ok(None)
            }
        }
        fn doc_ids(&self) -> crate::error::Result<Vec<u64>> {
            Ok(self.ids.clone())
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

    fn drain_matcher(mut matcher: Box<dyn Matcher>) -> Vec<u64> {
        let mut found = Vec::new();
        while !matcher.is_exhausted() {
            let doc_id = matcher.doc_id();
            if doc_id == u64::MAX {
                break;
            }
            found.push(doc_id);
            if !matcher.next().unwrap() {
                break;
            }
        }
        found
    }

    /// #996 regression: the BKD-less bounding-box fallback must probe
    /// only the doc ids actually present — not the dense `0..max_doc()`
    /// range, which both over-scans (fanout-global `max_doc`) and
    /// under-scans (ids above `max_doc` in sparse post-merge segments).
    #[test]
    fn bbox_fallback_scans_only_present_doc_ids() {
        let reader = GeoCountingReader::new(vec![5, 1500], 10);
        let query =
            GeoBoundingBoxQuery::within_bounding_box("location", 34.0, 138.0, 37.0, 141.0).unwrap();

        let found = drain_matcher(query.matcher(&reader).unwrap());

        assert_eq!(
            found,
            vec![5, 1500],
            "id above max_doc() must not be missed"
        );
        assert_eq!(
            reader.calls(),
            2,
            "fallback must probe only the present ids"
        );
    }

    /// #1000: a cell wholly inside the box would be classified `Inside`
    /// by a plain range visitor — `GeoBoxPointsVisitor` must say
    /// `Crosses` so the BKD streams the points and the coordinates stay
    /// available (2D twin of `nearest_visitor_never_returns_inside`).
    #[test]
    fn geo_box_points_visitor_never_returns_inside() {
        let visitor = GeoBoxPointsVisitor::new(0.0, 0.0, 10.0, 10.0);

        let contained = AABB::new(vec![4.0, 4.0], vec![6.0, 6.0]).unwrap();
        assert_eq!(visitor.compare(&contained), CellRelation::Crosses);

        let overlapping = AABB::new(vec![-5.0, -5.0], vec![5.0, 5.0]).unwrap();
        assert_eq!(visitor.compare(&overlapping), CellRelation::Crosses);

        let disjoint = AABB::new(vec![20.0, 20.0], vec![30.0, 30.0]).unwrap();
        assert_eq!(visitor.compare(&disjoint), CellRelation::Outside);
    }

    /// #1000: `visit` records only in-box points, with closed bounds on
    /// both ends (matching the previous `range_search(.., true, true)`),
    /// and `into_candidates` sorts by doc id. Since #1174 it keeps every
    /// point of a document (multi-valued fields); the per-document dedup
    /// happens in `find_matches`, after the radius / box test, keeping the
    /// closest point.
    #[test]
    fn geo_box_points_visitor_filters_and_orders() {
        let mut visitor = GeoBoxPointsVisitor::new(0.0, 0.0, 10.0, 10.0);

        visitor.visit(7, &[5.0, 5.0]); // inside
        visitor.visit(3, &[0.0, 10.0]); // on the boundary — inclusive
        visitor.visit(9, &[10.5, 5.0]); // outside (lat)
        visitor.visit(1, &[5.0, -0.1]); // outside (lon)
        visitor.visit(7, &[6.0, 6.0]); // second point of doc 7 — kept
        visitor.visit_inside(2); // must be ignored (no coordinates)

        let candidates = visitor.into_candidates();
        let ids: Vec<u64> = candidates.iter().map(|(doc_id, _)| *doc_id).collect();
        assert_eq!(
            ids,
            vec![3, 7, 7],
            "sorted, boundary-inclusive, every point kept"
        );
    }

    /// #1174: the distance query must not lose a document whose first
    /// traversed point lies inside the enclosing rectangle but outside the
    /// circle while a later point lies inside the circle — and it reports
    /// each document once, by its closest point.
    #[test]
    fn distance_query_keeps_the_closest_point_per_document() {
        let center = GeoPoint::new(0.0, 0.0);
        let query = GeoDistanceQuery::new("location", center, 150_000.0);
        let bounds = query.cap_bounds();
        // A rectangle corner: inside the bounds, outside the circle.
        let corner = [bounds.max_lat - 1e-6, bounds.lon_ranges[0].1 - 1e-6];
        assert!(center.distance_to(&GeoPoint::new(corner[0], corner[1])) > 150_000.0);

        let reader = BkdBackedReader::new(vec![
            (1, corner),
            (1, [0.5, 0.0]), // ≈ 55 km — the closest point of doc 1
            (1, [1.0, 0.0]), // ≈ 111 km
            (2, corner),     // doc 2 has no in-circle point
            (3, [1.0, 0.0]), // single point, ≈ 111 km
        ]);

        let matches = query.find_matches(&reader).unwrap();
        let ids: Vec<u64> = matches.iter().map(|m| m.doc_id).collect();
        assert_eq!(ids, vec![1, 3], "closest first, doc 1 once, doc 2 absent");
        assert!(
            (matches[0].distance_m - center.distance_to(&GeoPoint::new(0.5, 0.0))).abs() < 1e-6,
            "doc 1 is reported by its closest point"
        );
    }

    /// In-memory `BKDTree` that streams its fixed points through
    /// `visit()` (always-`Crosses` semantics — a legal tree behavior).
    #[derive(Debug)]
    struct FakeBkdTree {
        points: Vec<(u64, [f64; 2])>,
    }

    impl crate::lexical::index::structures::bkd_tree::BKDTree for FakeBkdTree {
        fn intersect(
            &self,
            visitor: &mut dyn crate::lexical::index::structures::visitor::IntersectVisitor,
        ) -> Result<()> {
            for (doc_id, point) in &self.points {
                visitor.visit(*doc_id, point);
            }
            Ok(())
        }
    }

    /// Reader whose geo field is backed by a [`FakeBkdTree`] and whose
    /// stored documents are absent (`document()` returns `None`, as for
    /// an `indexed = true, stored = false` field) while counting probes.
    #[derive(Debug)]
    struct BkdBackedReader {
        points: Vec<(u64, [f64; 2])>,
        document_calls: std::sync::atomic::AtomicU64,
    }

    impl BkdBackedReader {
        fn new(points: Vec<(u64, [f64; 2])>) -> Self {
            BkdBackedReader {
                points,
                document_calls: std::sync::atomic::AtomicU64::new(0),
            }
        }

        fn calls(&self) -> u64 {
            self.document_calls
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl LexicalIndexReader for BkdBackedReader {
        fn doc_count(&self) -> u64 {
            self.points.len() as u64
        }
        fn max_doc(&self) -> u64 {
            self.points.iter().map(|(id, _)| id + 1).max().unwrap_or(0)
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(
            &self,
            _doc_id: u64,
        ) -> crate::error::Result<Option<crate::lexical::core::document::Document>> {
            self.document_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(None)
        }
        fn get_bkd_tree(
            &self,
            _field: &str,
        ) -> crate::error::Result<
            Option<std::sync::Arc<dyn crate::lexical::index::structures::bkd_tree::BKDTree>>,
        > {
            Ok(Some(std::sync::Arc::new(FakeBkdTree {
                points: self.points.clone(),
            })))
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

    /// #1000 regression: the BKD path must take coordinates from the
    /// tree itself and never probe stored documents (which do not exist
    /// for an `indexed = true, stored = false` field).
    #[test]
    fn bkd_path_uses_tree_coordinates_without_stored_doc_probes() {
        let reader = BkdBackedReader::new(vec![
            (5, [35.68, 139.76]), // Tokyo
            (7, [34.69, 135.50]), // Osaka
            (9, [50.0, 10.0]),    // outside the Japan box
        ]);

        let bbox =
            GeoBoundingBoxQuery::within_bounding_box("location", 30.0, 128.0, 46.0, 146.0).unwrap();
        assert_eq!(drain_matcher(bbox.matcher(&reader).unwrap()), vec![5, 7]);

        let distance =
            GeoDistanceQuery::within_radius("location", 35.68, 139.76, 50_000.0).unwrap();
        assert_eq!(drain_matcher(distance.matcher(&reader).unwrap()), vec![5]);

        assert_eq!(
            reader.calls(),
            0,
            "the BKD path must not probe stored documents"
        );
    }

    /// #996 regression: `matcher_scorer()` must run the candidate scan
    /// once, shared between the matcher and the scorer — not once each.
    #[test]
    fn matcher_scorer_shares_one_candidate_pass() {
        let reader = GeoCountingReader::new(vec![5, 1500], 10);
        let query =
            GeoBoundingBoxQuery::within_bounding_box("location", 34.0, 138.0, 37.0, 141.0).unwrap();

        let (matcher, scorer) = query.matcher_scorer(&reader).unwrap();

        assert_eq!(
            reader.calls(),
            2,
            "one probe per present id — the candidate pass must run exactly once"
        );
        assert_eq!(drain_matcher(matcher), vec![5, 1500]);
        // The shared result must still feed distance-based scores.
        assert!(scorer.score(5, 1.0, None) > 0.0);
    }

    /// #996 regression: same gate for the distance-query fallback.
    #[test]
    fn distance_fallback_scans_only_present_doc_ids() {
        let reader = GeoCountingReader::new(vec![5, 1500], 10);
        let query = GeoDistanceQuery::within_radius("location", 35.68, 139.76, 50_000.0).unwrap();

        let found = drain_matcher(query.matcher(&reader).unwrap());

        assert_eq!(
            found,
            vec![5, 1500],
            "id above max_doc() must not be missed"
        );
        assert_eq!(
            reader.calls(),
            2,
            "fallback must probe only the present ids"
        );
    }

    /// Reader whose geo field is a real [`BKDReader`] written over
    /// `points`, so `compare` pruning runs (the [`FakeBkdTree`] above
    /// visits every point and never prunes).
    #[derive(Debug)]
    struct RealBkdReader {
        tree: std::sync::Arc<dyn crate::lexical::index::structures::bkd_tree::BKDTree>,
        max_doc: u64,
    }

    impl RealBkdReader {
        fn new(points: &[(u64, [f64; 2])]) -> Self {
            use crate::lexical::index::structures::bkd_tree::{BKDReader, BKDWriter};
            use crate::storage::Storage;
            use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

            let storage = std::sync::Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            let flat: Vec<f64> = points.iter().flat_map(|(_, p)| p.iter().copied()).collect();
            let doc_ids: Vec<u64> = points.iter().map(|(id, _)| *id).collect();
            {
                let output = storage.create_output("geo.bkd").unwrap();
                // Small blocks so that a few hundred points span many
                // leaves and the index-node pruning is exercised.
                let mut writer = BKDWriter::new(output, 2).with_block_size(16);
                writer.write(&flat, &doc_ids).unwrap();
                writer.finish().unwrap();
            }
            let reader = BKDReader::open(storage, "geo.bkd").unwrap();
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

    /// Doc ids within `distance_m` of `center` by a linear haversine scan:
    /// the reference every pruned search below is compared against.
    fn brute_force_within(
        center: GeoPoint,
        distance_m: f64,
        points: &[(u64, [f64; 2])],
    ) -> Vec<u64> {
        let mut ids: Vec<u64> = points
            .iter()
            .filter(|(_, p)| center.distance_to(&GeoPoint::new(p[0], p[1])) <= distance_m)
            .map(|(id, _)| *id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn matched_ids(query: &GeoDistanceQuery, reader: &dyn LexicalIndexReader) -> Vec<u64> {
        let mut ids: Vec<u64> = query
            .find_matches(reader)
            .unwrap()
            .iter()
            .map(|m| m.doc_id)
            .collect();
        ids.sort_unstable();
        ids
    }

    /// #1593: a circle crossing the antimeridian matches the points on
    /// both sides of it.
    #[test]
    fn distance_query_crosses_the_antimeridian() {
        let points = vec![
            (1, [0.0, 179.8]),  // ≈ 11 km east of the centre
            (2, [0.0, -179.9]), // ≈ 22 km, on the far side of ±180
            (3, [0.0, -179.0]), // ≈ 122 km, outside
            (4, [0.0, 179.0]),  // ≈ 100 km, outside
            (5, [0.1, -180.0]), // ≈ 15 km, stored with the -180 sign
        ];
        let center = GeoPoint::new(0.0, 179.9);
        let query = GeoDistanceQuery::new("location", center, 50_000.0);
        assert_eq!(brute_force_within(center, 50_000.0, &points), vec![1, 2, 5]);

        let bounds = query.cap_bounds();
        assert_eq!(
            bounds.lon_ranges.len(),
            2,
            "split at the antimeridian: {bounds:?}"
        );

        assert_eq!(
            matched_ids(&query, &RealBkdReader::new(&points)),
            vec![1, 2, 5]
        );
        assert_eq!(
            matched_ids(&query, &BkdBackedReader::new(points)),
            vec![1, 2, 5]
        );
    }

    /// #1593: a circle containing a pole covers every longitude.
    #[test]
    fn distance_query_wraps_around_the_pole() {
        let points = vec![
            (1, [89.9, 160.0]),  // ≈ 2 km
            (2, [89.9, -170.0]), // ≈ 4 km, "behind" the pole in longitude
            (3, [89.9, -10.0]),  // ≈ 22 km, through the pole
            (4, [89.0, 170.0]),  // ≈ 100 km, outside
            (5, [-89.9, 170.0]), // the other pole, outside
        ];
        let center = GeoPoint::new(89.9, 170.0);
        let query = GeoDistanceQuery::new("location", center, 50_000.0);
        assert_eq!(brute_force_within(center, 50_000.0, &points), vec![1, 2, 3]);

        let bounds = query.cap_bounds();
        assert_eq!(bounds.lon_ranges, vec![(-180.0, 180.0)], "{bounds:?}");
        assert_eq!(bounds.max_lat, 90.0);

        assert_eq!(
            matched_ids(&query, &RealBkdReader::new(&points)),
            vec![1, 2, 3]
        );
        assert_eq!(
            matched_ids(&query, &BkdBackedReader::new(points)),
            vec![1, 2, 3]
        );
    }

    /// #1593: at high latitude the circle's widest longitude extent lies
    /// nearer the pole than the centre, so a box of `δ / cos(center.lat)`
    /// degrees cuts off the circle's flanks.
    #[test]
    fn distance_query_keeps_the_flanks_of_a_high_latitude_circle() {
        let points = vec![
            (1, [80.0, 20.0]),    // ≈ 384 km, inside the old box
            (2, [81.059, 26.5]),  // ≈ 494 km, inside the circle, outside the old box
            (3, [81.059, -26.5]), // mirror of 2
            (4, [80.0, 30.0]),    // ≈ 572 km, outside
        ];
        let center = GeoPoint::new(80.0, 0.0);
        let query = GeoDistanceQuery::new("location", center, 500_000.0);
        assert_eq!(
            brute_force_within(center, 500_000.0, &points),
            vec![1, 2, 3]
        );

        assert_eq!(
            matched_ids(&query, &RealBkdReader::new(&points)),
            vec![1, 2, 3]
        );
        assert_eq!(
            matched_ids(&query, &BkdBackedReader::new(points)),
            vec![1, 2, 3]
        );
    }

    /// #1593: the pruned search returns exactly the linear-scan set for
    /// random centres (including near the poles and the antimeridian) and
    /// radii up to 1 000 km, against a real BKD tree.
    #[test]
    fn distance_query_matches_brute_force_everywhere() {
        // Deterministic LCG so a failure reproduces.
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };

        // Points everywhere, denser near the poles and the antimeridian
        // where the old box failed.
        let mut points: Vec<(u64, [f64; 2])> = Vec::new();
        for i in 0..1_500u64 {
            let (lat, lon) = match i % 3 {
                0 => (next() * 180.0 - 90.0, next() * 360.0 - 180.0),
                1 => (
                    (next() * 10.0 + 80.0) * if next() < 0.5 { 1.0 } else { -1.0 },
                    next() * 360.0 - 180.0,
                ),
                _ => (
                    next() * 180.0 - 90.0,
                    (next() * 20.0 + 170.0) * if next() < 0.5 { 1.0 } else { -1.0 },
                ),
            };
            points.push((i, [lat, lon.clamp(-180.0, 180.0)]));
        }
        let reader = RealBkdReader::new(&points);

        let mut nonempty = 0;
        for _ in 0..300 {
            let (lat, lon) = match (next() * 3.0) as u32 {
                0 => (next() * 180.0 - 90.0, next() * 360.0 - 180.0),
                1 => (
                    (next() * 10.0 + 80.0) * if next() < 0.5 { 1.0 } else { -1.0 },
                    next() * 360.0 - 180.0,
                ),
                _ => (
                    next() * 180.0 - 90.0,
                    (next() * 10.0 + 170.0) * if next() < 0.5 { 1.0 } else { -1.0 },
                ),
            };
            let center = GeoPoint::new(lat, lon.clamp(-180.0, 180.0));
            let distance_m = next() * 1_000_000.0;
            let query = GeoDistanceQuery::new("location", center, distance_m);
            let expected = brute_force_within(center, distance_m, &points);
            nonempty += usize::from(!expected.is_empty());
            assert_eq!(
                matched_ids(&query, &reader),
                expected,
                "center=({lat}, {lon}) radius={distance_m}"
            );
        }
        assert!(
            nonempty > 100,
            "the random queries should mostly hit something"
        );
    }

    /// A radius covering the whole sphere or more is bounded by the whole
    /// sphere, not by an inverted box.
    #[test]
    fn cap_bounds_degenerate_radii() {
        let center = GeoPoint::new(10.0, 20.0);
        let whole = GeoDistanceQuery::new("location", center, 25_000_000.0).cap_bounds();
        assert_eq!(whole, CapBounds::whole_sphere());

        let zero = GeoDistanceQuery::new("location", center, 0.0).cap_bounds();
        assert!(zero.contains(&center));
        assert!(!zero.contains(&GeoPoint::new(10.0, 20.001)));
    }

    fn square_with_a_hole() -> GeoPolygon {
        let ring = |points: &[(f64, f64)]| -> Vec<GeoPoint> {
            points
                .iter()
                .map(|&(lat, lon)| GeoPoint::new(lat, lon))
                .collect()
        };
        GeoPolygon::new(
            ring(&[(10.0, 10.0), (10.0, 30.0), (30.0, 30.0), (30.0, 10.0)]),
            vec![ring(&[
                (15.0, 15.0),
                (15.0, 25.0),
                (25.0, 25.0),
                (25.0, 15.0),
            ])],
        )
        .unwrap()
    }

    fn across_the_antimeridian() -> GeoPolygon {
        GeoPolygonQuery::within_polygon(
            "location",
            &[
                (-20.0, 170.0),
                (-20.0, -170.0),
                (-10.0, -170.0),
                (-10.0, 170.0),
            ],
            &[],
        )
        .unwrap()
        .polygon()
        .clone()
    }

    /// Deterministic pseudo-random numbers in `[0, 1)` (xorshift64*).
    fn uniform(state: &mut u64) -> f64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Issue #1596: a point on an exterior edge or vertex is inside, one on
    /// a hole's edge or vertex is not, and a polygon across the antimeridian
    /// holds both signs of ±180.
    #[test]
    fn a_polygon_contains_its_exterior_boundary_and_not_its_holes() {
        let polygon = square_with_a_hole();
        let mut wrong = Vec::new();
        for (lat, lon, expected) in [
            (10.0, 20.0, true),  // an exterior edge
            (30.0, 30.0, true),  // an exterior vertex
            (12.0, 12.0, true),  // inside
            (15.0, 20.0, false), // a hole's edge
            (25.0, 25.0, false), // a hole's vertex
            (20.0, 20.0, false), // in the hole
            (9.0, 20.0, false),  // outside
            (20.0, 30.0 + 1e-6, false),
        ] {
            if polygon.contains(&GeoPoint::new(lat, lon)) != expected {
                wrong.push(format!("({lat}, {lon}) should be {expected}"));
            }
        }
        let polygon = across_the_antimeridian();
        for (lat, lon, expected) in [
            (-15.0, 175.0, true),
            (-15.0, -175.0, true),
            (-15.0, 180.0, true),
            (-15.0, -180.0, true),
            (-15.0, 0.0, false),
            (-15.0, 165.0, false),
        ] {
            if polygon.contains(&GeoPoint::new(lat, lon)) != expected {
                wrong.push(format!("across: ({lat}, {lon}) should be {expected}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1596: `relate` never says `Inside` of a cell with a point
    /// outside, nor `Outside` of one with a point inside, on random cells.
    #[test]
    fn a_polygon_relates_cells_without_mistakes() {
        let mut state = 7_u64;
        let mut wrong = Vec::new();
        let mut seen = [0_usize; 3];
        for polygon in [square_with_a_hole(), across_the_antimeridian()] {
            for _ in 0..2_000 {
                let lat = -40.0 + 80.0 * uniform(&mut state);
                let lon = -180.0 + 360.0 * uniform(&mut state);
                let height = 15.0 * uniform(&mut state);
                let width = 15.0 * uniform(&mut state);
                let min = [lat, lon];
                let max = [lat + height, (lon + width).min(180.0)];
                let relation = polygon.relate(&min, &max);
                seen[relation as usize] += 1;
                if relation == CellRelation::Crosses {
                    continue;
                }
                // A grid of points of the cell, its corners and edges too.
                for i in 0..=8 {
                    for j in 0..=8 {
                        let p_lat = min[0] + (max[0] - min[0]) * f64::from(i) / 8.0;
                        let p_lon = min[1] + (max[1] - min[1]) * f64::from(j) / 8.0;
                        let inside = polygon.contains_lat_lon(p_lat, p_lon);
                        if inside != (relation == CellRelation::Inside) {
                            wrong.push(format!(
                                "{relation:?} cell {min:?}..{max:?} has ({p_lat}, {p_lon}) \
                                 inside: {inside}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|count| *count > 0),
            "every relation should come up: {seen:?}"
        );
        assert!(
            wrong.is_empty(),
            "{} mistakes: {:#?}",
            wrong.len(),
            &wrong[..wrong.len().min(10)]
        );
    }

    /// Issue #1596: through a BKD of many points, a polygon tests far fewer
    /// points one by one than the BKD holds, taking the cells inside it
    /// whole, and finds what a linear scan finds.
    #[test]
    fn a_polygon_prunes_the_bkd() {
        use crate::lexical::index::structures::bkd_tree::{BKDReader, BKDTree, BKDWriter};
        use crate::storage::Storage;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let n = 20_000_u64;
        let mut state = 11_u64;
        let points: Vec<[f64; 2]> = (0..n)
            .map(|_| {
                [
                    -60.0 + 120.0 * uniform(&mut state),
                    -180.0 + 360.0 * uniform(&mut state),
                ]
            })
            .collect();
        let storage = std::sync::Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("geo.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2).with_block_size(64);
            let flat: Vec<f64> = points.iter().flatten().copied().collect();
            let doc_ids: Vec<u64> = (0..n).collect();
            writer.write(&flat, &doc_ids).unwrap();
            writer.finish().unwrap();
        }
        let tree = BKDReader::open(storage, "geo.bkd").unwrap();

        // A large one, whose hits are mostly in cells inside it. It is 300
        // degrees wide, so its long edges have a vertex in the middle: an
        // edge takes the shorter way, and from -150 to 150 that is the 60
        // degrees across the antimeridian.
        let large = GeoPolygonQuery::within_polygon(
            "location",
            &[
                (-50.0, -150.0),
                (-50.0, 0.0),
                (-50.0, 150.0),
                (50.0, 150.0),
                (50.0, 0.0),
                (50.0, -150.0),
            ],
            &[&[(-10.0, -10.0), (-10.0, 10.0), (10.0, 10.0), (10.0, -10.0)]],
        )
        .unwrap()
        .polygon()
        .clone();

        let mut wrong = Vec::new();
        for (name, polygon, is_large) in [
            ("square with a hole", square_with_a_hole(), false),
            ("across the antimeridian", across_the_antimeridian(), false),
            ("large with a hole", large, true),
        ] {
            let mut visitor = GeoPolygonVisitor::new(&polygon);
            tree.intersect(&mut visitor).unwrap();
            let visited = visitor.visited;
            let mut found = visitor.hits;
            found.sort_unstable();
            let expected: Vec<u64> = (0..n)
                .filter(|&id| {
                    let [lat, lon] = points[id as usize];
                    polygon.contains_lat_lon(lat, lon)
                })
                .collect();
            if found != expected {
                wrong.push(format!(
                    "{name}: found {} ids, expected {}",
                    found.len(),
                    expected.len()
                ));
            }
            // Only the points of the cells across an edge are tested one
            // by one: a tenth of the points at most for the small polygons,
            // and fewer than half of the hits for the large one, whose
            // other hits come from cells taken whole.
            let pruned = if is_large {
                visited * 2 < found.len()
            } else {
                visited * 10 < n as usize
            };
            if !pruned {
                wrong.push(format!(
                    "{name}: tested {visited} of {n} points one by one, for {} hits",
                    found.len()
                ));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    fn polyline(vertices: &[(f64, f64)]) -> GeoPolyline {
        GeoPolyline::new(
            vertices
                .iter()
                .map(|&(lat, lon)| GeoPoint::new(lat, lon))
                .collect(),
        )
        .unwrap()
    }

    /// The point `distance` radians from `from` in the direction `bearing`
    /// (radians clockwise from north), on the unit sphere.
    fn destination(from: &GeoPoint, bearing: f64, distance: f64) -> GeoPoint {
        let (lat, lon) = (from.lat.to_radians(), from.lon.to_radians());
        let lat2 = (lat.sin() * distance.cos() + lat.cos() * distance.sin() * bearing.cos())
            .clamp(-1.0, 1.0)
            .asin();
        let lon2 = lon
            + (bearing.sin() * distance.sin() * lat.cos())
                .atan2(distance.cos() - lat.sin() * lat2.sin());
        let mut lon2 = lon2.to_degrees();
        while lon2 > 180.0 {
            lon2 -= 360.0;
        }
        while lon2 < -180.0 {
            lon2 += 360.0;
        }
        GeoPoint::new(lat2.to_degrees().clamp(-90.0, 90.0), lon2)
    }

    /// The point a fraction `t` of the way along the arc from `a` to `b`.
    fn along(a: &GeoPoint, b: &GeoPoint, t: f64) -> GeoPoint {
        let (va, vb) = (unit_vector(a), unit_vector(b));
        let angle = norm(cross(va, vb)).atan2(dot(va, vb));
        if angle < 1e-15 {
            return *a;
        }
        let (wa, wb) = (
            ((1.0 - t) * angle).sin() / angle.sin(),
            (t * angle).sin() / angle.sin(),
        );
        let v = [
            wa * va[0] + wb * vb[0],
            wa * va[1] + wb * vb[1],
            wa * va[2] + wb * vb[2],
        ];
        GeoPoint::new(
            v[2].atan2(v[0].hypot(v[1])).to_degrees(),
            v[1].atan2(v[0]).to_degrees(),
        )
    }

    /// Issue #1596: distances with an exact answer: over the inside of an
    /// equatorial edge, beside a meridian, past an end, to a single vertex,
    /// to an edge of length 0, and beside an edge 1 cm long.
    #[test]
    fn a_polyline_measures_exact_distances() {
        let radius = EARTH_RADIUS_M;
        let mut wrong = Vec::new();
        let mut check = |name: &str, found: f64, expected: f64, tolerance: f64| {
            if (found - expected).abs() > tolerance {
                wrong.push(format!("{name}: {found}, expected {expected}"));
            }
        };
        let equator = polyline(&[(0.0, 10.0), (0.0, 40.0)]);
        check(
            "over an equatorial edge",
            equator.distance_to(&GeoPoint::new(1.5, 25.0)),
            1.5_f64.to_radians() * radius,
            1e-6,
        );
        check(
            "past its end",
            equator.distance_to(&GeoPoint::new(0.0, 45.0)),
            GeoPoint::new(0.0, 40.0).distance_to(&GeoPoint::new(0.0, 45.0)),
            1e-6,
        );
        let meridian = polyline(&[(10.0, 20.0), (50.0, 20.0)]);
        check(
            "beside a meridian",
            meridian.distance_to(&GeoPoint::new(30.0, 21.0)),
            (30.0_f64.to_radians().cos() * 1.0_f64.to_radians().sin()).asin() * radius,
            1e-6,
        );
        let vertex = GeoPoint::new(35.5, 139.5);
        let point = GeoPoint::new(35.7, 139.9);
        check(
            "a single vertex, exactly",
            polyline(&[(35.5, 139.5)]).distance_to(&point),
            vertex.distance_to(&point),
            0.0,
        );
        check(
            "an edge of length 0 on the antimeridian",
            polyline(&[(0.0, 180.0), (0.0, -180.0)]).distance_to(&GeoPoint::new(0.0, 170.0)),
            10.0_f64.to_radians() * radius,
            1e-6,
        );
        // About 111 m north of the middle of an edge 9 mm long: as far as
        // from its ends, within a millimetre.
        let short = polyline(&[(35.0, 139.0), (35.0, 139.000_000_1)]);
        let beside = GeoPoint::new(35.001, 139.000_000_05);
        check(
            "beside an edge 9 mm long",
            short.distance_to(&beside),
            GeoPoint::new(35.0, 139.0).distance_to(&beside),
            1e-3,
        );
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1596: an edge's bounds hold the arc's highest latitude between
    /// its vertices, every longitude for an edge over or at a pole, and both
    /// sides of the antimeridian for an edge across it.
    #[test]
    fn an_edge_s_bounds_hold_its_arc() {
        let delta = 1_000.0 / EARTH_RADIUS_M;
        let bounds = |vertices: &[(f64, f64)]| polyline(vertices).edge_bounds(1_000.0)[0].clone();
        let mut wrong = Vec::new();

        let high = bounds(&[(10.0, 0.0), (10.0, 170.0)]);
        if high.max_lat < 63.69 || !high.contains(&GeoPoint::new(63.69, 85.0)) {
            wrong.push(format!("an arc rising to 63.7°N: {high:?}"));
        }
        let south = bounds(&[(-10.0, 0.0), (-10.0, 170.0)]);
        if south.min_lat > -63.69 || !south.contains(&GeoPoint::new(-63.69, 85.0)) {
            wrong.push(format!("an arc sinking to 63.7°S: {south:?}"));
        }
        for vertices in [
            [(70.0, 30.0), (70.0, -150.0)],
            [(-70.0, 30.0), (-70.0, -150.0)],
            [(90.0, 123.0), (70.0, -40.0)],
            [(-90.0, 0.0), (-89.9999, 10.0)],
        ] {
            let b = bounds(&vertices);
            if b.lon_ranges != [(-180.0, 180.0)] {
                wrong.push(format!("{vertices:?} over or at a pole: {b:?}"));
            }
        }
        let across = bounds(&[(-15.0, 179.5), (-15.0, -179.5)]);
        for lon in [179.6, 180.0, -180.0, -179.6] {
            if !across.contains(&GeoPoint::new(-15.0, lon)) {
                wrong.push(format!("across the antimeridian, at {lon}: {across:?}"));
            }
        }
        if across.contains(&GeoPoint::new(-15.0, 0.0)) {
            wrong.push(format!("across the antimeridian, holding 0: {across:?}"));
        }
        let equator = bounds(&[(0.0, 10.0), (0.0, 40.0)]);
        let half = delta.to_degrees();
        if equator.max_lat > half + 1e-6 || equator.min_lat < -half - 1e-6 {
            wrong.push(format!("an equatorial edge: {equator:?}"));
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1596: every point within the distance of an edge lies inside
    /// the edge's bounds, so no cell holding one is pruned, and is measured
    /// within the distance: points walked from a random point of the arc in
    /// a random direction, by the distance or less, over awkward edges and
    /// random ones.
    #[test]
    fn no_point_within_the_distance_is_outside_its_bounds() {
        let mut state = 21_u64;
        let mut edges: Vec<[(f64, f64); 2]> = vec![
            [(10.0, 0.0), (10.0, 170.0)],
            [(70.0, 30.0), (70.0, -150.0)],
            [(-70.0, 30.0), (-70.0, -150.0)],
            [(90.0, 123.0), (70.0, -40.0)],
            [(-90.0, 0.0), (-60.0, 77.0)],
            [(0.0, 10.0), (0.0, 40.0)],
            [(10.0, 20.0), (50.0, 20.0)],
            [(-15.0, 179.5), (-15.0, -179.5)],
            [(0.0, 180.0), (0.0, -180.0)],
            [(35.0, 139.0), (35.0, 139.000_000_1)],
            [(0.0, 0.0), (0.0, 179.99)],
            [(20.0, 60.0), (20.0, 60.0)],
        ];
        while edges.len() < 300 {
            let a = (
                -90.0 + 180.0 * uniform(&mut state),
                -180.0 + 360.0 * uniform(&mut state),
            );
            let b = (
                -90.0 + 180.0 * uniform(&mut state),
                -180.0 + 360.0 * uniform(&mut state),
            );
            let (va, vb) = (
                unit_vector(&GeoPoint::new(a.0, a.1)),
                unit_vector(&GeoPoint::new(b.0, b.1)),
            );
            if dot(va, vb) > -0.999 {
                edges.push([a, b]);
            }
        }
        let mut wrong = Vec::new();
        for (i, edge) in edges.iter().enumerate() {
            let line = polyline(edge);
            let (a, b) = (line.vertices[0], line.vertices[1]);
            for distance_m in [0.0, 50.0, 30_000.0, 800_000.0, 5_000_000.0] {
                let bounds = line.edge_bounds(distance_m);
                for _ in 0..40 {
                    let on_arc = along(&a, &b, uniform(&mut state));
                    let walk = if uniform(&mut state) < 0.2 {
                        distance_m
                    } else {
                        distance_m * uniform(&mut state)
                    };
                    let bearing = std::f64::consts::TAU * uniform(&mut state);
                    let point = destination(&on_arc, bearing, walk / EARTH_RADIUS_M);
                    let measured = line.distance_within(&point, &bounds);
                    match measured {
                        None => wrong.push(format!(
                            "edge {i} {edge:?} within {distance_m} m: {point:?}, {walk} m from \
                             {on_arc:?}, outside {:?}",
                            bounds[0]
                        )),
                        Some(d) if d > walk + 1e-3 => wrong.push(format!(
                            "edge {i} {edge:?}: {point:?} measured {d} m, walked {walk} m"
                        )),
                        Some(_) => {}
                    }
                }
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
    fn a_polyline_prunes_the_bkd() {
        use crate::lexical::index::structures::bkd_tree::{BKDReader, BKDTree, BKDWriter};
        use crate::storage::Storage;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let n = 20_000_u64;
        let mut state = 12_u64;
        let points: Vec<[f64; 2]> = (0..n)
            .map(|_| {
                [
                    -60.0 + 120.0 * uniform(&mut state),
                    -180.0 + 360.0 * uniform(&mut state),
                ]
            })
            .collect();
        let storage = std::sync::Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("geo.bkd").unwrap();
            let mut writer = BKDWriter::new(output, 2).with_block_size(64);
            let flat: Vec<f64> = points.iter().flatten().copied().collect();
            let doc_ids: Vec<u64> = (0..n).collect();
            writer.write(&flat, &doc_ids).unwrap();
            writer.finish().unwrap();
        }
        let tree = BKDReader::open(storage, "geo.bkd").unwrap();

        let mut wrong = Vec::new();
        for (name, line, distance_m) in [
            (
                "zigzag",
                polyline(&[
                    (-20.0, -40.0),
                    (-10.0, -30.0),
                    (-15.0, -20.0),
                    (-5.0, -10.0),
                ]),
                100_000.0,
            ),
            (
                "across the antimeridian",
                polyline(&[(-30.0, 165.0), (-10.0, 179.0), (0.0, -170.0)]),
                100_000.0,
            ),
        ] {
            let bounds = line.edge_bounds(distance_m);
            let mut visitor = GeoPolylineVisitor::new(&line, &bounds, distance_m);
            tree.intersect(&mut visitor).unwrap();
            let visited = visitor.visited;
            let mut found: Vec<u64> = visitor.hits.iter().map(|(id, _, _)| *id).collect();
            found.sort_unstable();
            let expected: Vec<u64> = (0..n)
                .filter(|&id| {
                    let [lat, lon] = points[id as usize];
                    line.distance_to(&GeoPoint::new(lat, lon)) <= distance_m
                })
                .collect();
            if found != expected || expected.is_empty() {
                wrong.push(format!(
                    "{name}: found {} ids, expected {}",
                    found.len(),
                    expected.len()
                ));
            }
            if visited * 10 >= n as usize {
                wrong.push(format!(
                    "{name}: tested {visited} of {n} points one by one, for {} hits",
                    found.len()
                ));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }
}
