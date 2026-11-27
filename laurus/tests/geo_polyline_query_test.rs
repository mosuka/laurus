//! Issue #1596: `GeoPolylineQuery` and `Geo3dPolylineQuery` match the
//! documents with a point within a distance of a polyline, scoring
//! `1 - d/r` by the closest point: exactly what a linear scan finds, through
//! the BKD (with and without stored values) and, for `Geo`, over the stored
//! documents, in one segment and in two.
//!
//! The references here are written for these tests and share nothing with
//! the queries' code: an arc of a `Geo` polyline is `a cos t + u sin t` for
//! `t` from 0 to its angle, its closest point to a point is found in closed
//! form, and the distance to it is the haversine. A document within 1 mm of
//! the distance is left out of the comparison, since the two ways of
//! measuring may round it to either side.

use std::sync::Arc;

use laurus::lexical::core::field::{FieldOption, Geo3dOption, GeoOption, TextOption};
use laurus::lexical::query::geo::{GeoDistanceQuery, GeoPolyline, GeoPolylineQuery};
use laurus::lexical::query::geo3d::{Geo3dDistanceQuery, Geo3dPolyline, Geo3dPolylineQuery};
use laurus::lexical::query::{GeoPoint, Query};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::util::ecef::wgs84_to_ecef;
use laurus::{DataValue, Document, GeoEcefPoint};

/// The radius of the sphere `GeoPoint::distance_to` measures on, in metres.
const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// How near the distance a document may be and still be compared, in metres.
const AMBIGUOUS_M: f64 = 1e-3;

/// How a store indexes its geo field, and how many commits it makes.
#[derive(Clone, Copy, Debug)]
struct Mode {
    indexed: bool,
    stored: bool,
    segments: usize,
}

/// For `Geo`: the BKD path with and without stored values and the fallback
/// over stored documents, in one segment and in two.
fn geo_modes() -> Vec<Mode> {
    let mut modes = Vec::new();
    for (indexed, stored) in [(true, true), (true, false), (false, true)] {
        for segments in [1, 2] {
            modes.push(Mode {
                indexed,
                stored,
                segments,
            });
        }
    }
    modes
}

/// For `Geo3d`, which every 3D query reads through the BKD only: with and
/// without stored values, in one segment and in two.
fn geo3d_modes() -> Vec<Mode> {
    geo_modes()
        .into_iter()
        .filter(|mode| mode.indexed)
        .collect()
}

fn geo_doc(points: &[(f64, f64)]) -> Document {
    let points: Vec<GeoPoint> = points
        .iter()
        .map(|&(lat, lon)| GeoPoint::try_new(lat, lon).unwrap())
        .collect();
    match points.as_slice() {
        [point] => Document::builder()
            .add_field("location", DataValue::Geo(*point))
            .build(),
        _ => Document::builder()
            .add_geo_array("location", points)
            .build(),
    }
}

fn geo3d_doc(points: &[GeoEcefPoint]) -> Document {
    match points {
        [point] => Document::builder()
            .add_field("position", DataValue::GeoEcef(*point))
            .build(),
        _ => Document::builder()
            .add_geo_ecef_array("position", points.to_vec())
            .build(),
    }
}

/// An empty store with a `Geo` field `location` and a `Geo3d` field
/// `position`, both indexed and stored as `mode` says, and a text field
/// `body`.
fn empty_store(mode: Mode, multi_valued: bool) -> LexicalStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "location",
            FieldOption::Geo(GeoOption {
                indexed: mode.indexed,
                stored: mode.stored,
                multi_valued,
                doc_values: true,
            }),
        )
        .add_field(
            "position",
            FieldOption::Geo3d(Geo3dOption {
                indexed: mode.indexed,
                stored: mode.stored,
                multi_valued,
                doc_values: true,
            }),
        )
        .add_field("body", FieldOption::Text(TextOption::default()))
        .build();
    LexicalStore::new(storage, config).unwrap()
}

/// A store holding `docs`, committed in `mode.segments` commits.
fn store_with(mode: Mode, multi_valued: bool, docs: &[(u64, Document)]) -> LexicalStore {
    let store = empty_store(mode, multi_valued);
    let per_segment = docs.len().div_ceil(mode.segments);
    for chunk in docs.chunks(per_segment) {
        for (id, doc) in chunk {
            store.upsert_document(*id, doc.clone()).unwrap();
        }
        store.commit().unwrap();
    }
    store
}

/// The ids and scores a query finds, sorted by id.
fn hits(store: &LexicalStore, query: Box<dyn Query>) -> Vec<(u64, f32)> {
    let mut hits: Vec<(u64, f32)> = store
        .search(LexicalSearchRequest::new(query).limit(100_000))
        .unwrap()
        .hits
        .iter()
        .map(|hit| (hit.doc_id, hit.score))
        .collect();
    hits.sort_by_key(|(id, _)| *id);
    hits
}

/// Deterministic pseudo-random numbers (xorshift64*), so that a failure
/// reproduces.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn between(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }
}

/// `n` random points in the given ranges, longitudes past 180 wrapped.
fn random_points(seed: u64, n: usize, lats: (f64, f64), lons: (f64, f64)) -> Vec<(f64, f64)> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let lat = rng.between(lats.0, lats.1);
            let mut lon = rng.between(lons.0, lons.1);
            if lon > 180.0 {
                lon -= 360.0;
            }
            (lat, lon)
        })
        .collect()
}

/// The points around the poles and on the antimeridian, with both signs.
const SPECIAL_POINTS: [(f64, f64); 8] = [
    (90.0, 0.0),
    (-90.0, 0.0),
    (90.0, 180.0),
    (0.0, 180.0),
    (0.0, -180.0),
    (45.0, 180.0),
    (-45.0, -180.0),
    (89.999, -180.0),
];

type Vec3 = [f64; 3];

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn axpy(x: f64, a: Vec3, y: f64, b: Vec3) -> Vec3 {
    [
        x * a[0] + y * b[0],
        x * a[1] + y * b[1],
        x * a[2] + y * b[2],
    ]
}

fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

fn unit(point: (f64, f64)) -> Vec3 {
    let (lat, lon) = (point.0.to_radians(), point.1.to_radians());
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}

fn lat_lon(v: Vec3) -> (f64, f64) {
    (
        v[2].atan2(v[0].hypot(v[1])).to_degrees(),
        v[1].atan2(v[0]).to_degrees(),
    )
}

/// The haversine distance on the sphere of `GeoPoint::distance_to`.
fn haversine(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (lat1, lat2) = (a.0.to_radians(), b.0.to_radians());
    let d_lat = (b.0 - a.0).to_radians();
    let d_lon = (b.1 - a.1).to_radians();
    let h = (d_lat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (d_lon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * h.sqrt().atan2((1.0 - h).sqrt())
}

/// The distance from `p` to the minor arc from `a` to `b`: the arc is
/// `a cos t + u sin t` for `t` in `[0, θ]`, with `u` the unit vector in the
/// plane of the arc perpendicular to `a`. `p · (a cos t + u sin t)` is
/// `A cos(t - t0)` with `t0 = atan2(p·u, p·a)`, so the arc's closest point
/// is at `t0` if `t0` is on the arc, and at an end otherwise.
fn reference_to_arc(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let ends = haversine(p, a).min(haversine(p, b));
    let (va, vb, vp) = (unit(a), unit(b), unit(p));
    let across = axpy(1.0, vb, -dot(va, vb), va);
    let length = norm(across);
    if length < 1e-15 {
        return ends;
    }
    let u = axpy(1.0 / length, across, 0.0, across);
    let theta = length.atan2(dot(va, vb));
    let t = dot(vp, u).atan2(dot(vp, va));
    if t > 0.0 && t < theta {
        let closest = axpy(t.cos(), va, t.sin(), u);
        ends.min(haversine(p, lat_lon(closest)))
    } else {
        ends
    }
}

/// The distance from `p` to the polyline through `vertices`.
fn reference_to_polyline(p: (f64, f64), vertices: &[(f64, f64)]) -> f64 {
    if vertices.len() == 1 {
        return haversine(p, vertices[0]);
    }
    vertices
        .windows(2)
        .map(|edge| reference_to_arc(p, edge[0], edge[1]))
        .fold(f64::INFINITY, f64::min)
}

fn ecef(point: GeoEcefPoint) -> Vec3 {
    [point.x, point.y, point.z]
}

/// The distance from `p` to the straight segments through `vertices`.
fn reference_to_segments(p: GeoEcefPoint, vertices: &[GeoEcefPoint]) -> f64 {
    let p = ecef(p);
    let to_point = |v: Vec3| norm(axpy(1.0, p, -1.0, v));
    if vertices.len() == 1 {
        return to_point(ecef(vertices[0]));
    }
    vertices
        .windows(2)
        .map(|edge| {
            let (a, b) = (ecef(edge[0]), ecef(edge[1]));
            let ab = axpy(1.0, b, -1.0, a);
            let length_sq = dot(ab, ab);
            if length_sq == 0.0 {
                return to_point(a);
            }
            let t = (dot(axpy(1.0, p, -1.0, a), ab) / length_sq).clamp(0.0, 1.0);
            to_point(axpy(1.0, a, t, ab))
        })
        .fold(f64::INFINITY, f64::min)
}

/// What is wrong with `found` against `distances` (each document's distance
/// by the reference): a document within `distance_m` missing, one beyond it
/// found, or a score other than `boost × (1 - d/r)`. Documents within
/// [`AMBIGUOUS_M`] of `distance_m` are not compared; their count is
/// returned with the mistakes.
fn mistakes(
    found: &[(u64, f32)],
    distances: &[(u64, f64)],
    distance_m: f64,
    boost: f32,
) -> (Vec<String>, usize) {
    let mut wrong = Vec::new();
    let mut ambiguous = 0;
    for &(id, d) in distances {
        if (d - distance_m).abs() < AMBIGUOUS_M {
            ambiguous += 1;
            continue;
        }
        let hit = found.iter().find(|(hit, _)| *hit == id);
        match (d <= distance_m, hit) {
            (true, None) => wrong.push(format!("doc {id} at {d} m missing")),
            (false, Some(_)) => wrong.push(format!("doc {id} at {d} m found")),
            (true, Some(&(_, score))) => {
                let expected = boost * (1.0 - d / distance_m).clamp(0.0, 1.0) as f32;
                if (score - expected).abs() > 1e-5 * boost.max(1.0) {
                    wrong.push(format!(
                        "doc {id} at {d} m scored {score}, expected {expected}"
                    ));
                }
            }
            (false, None) => {}
        }
    }
    (wrong, ambiguous)
}

/// `points` as documents with one point each, ids from 1.
fn geo_docs(points: &[(f64, f64)]) -> Vec<(u64, Document)> {
    (1..)
        .zip(points)
        .map(|(id, &point)| (id, geo_doc(&[point])))
        .collect()
}

/// The mistakes of `GeoPolylineQuery` over `points` in every `Geo` mode,
/// against the reference.
fn wrong_geo_modes(points: &[(f64, f64)], vertices: &[(f64, f64)], distance_m: f64) -> Vec<String> {
    let docs = geo_docs(points);
    let distances: Vec<(u64, f64)> = (1..)
        .zip(points)
        .map(|(id, &p)| (id, reference_to_polyline(p, vertices)))
        .collect();
    let mut wrong = Vec::new();
    for mode in geo_modes() {
        let store = store_with(mode, false, &docs);
        let query = GeoPolylineQuery::within_distance("location", vertices, distance_m).unwrap();
        let found = hits(&store, Box::new(query));
        let (mistakes, ambiguous) = mistakes(&found, &distances, distance_m, 1.0);
        if ambiguous * 100 > points.len() {
            wrong.push(format!(
                "{mode:?}: {ambiguous} documents too near the distance"
            ));
        }
        for mistake in mistakes {
            wrong.push(format!(
                "{mode:?} {vertices:?} within {distance_m} m: {mistake}"
            ));
        }
    }
    wrong
}

/// Issue #1596: a zigzag polyline over random points finds exactly the
/// points the reference puts within the distance, each scoring `1 - d/r`.
#[test]
fn a_zigzag_polyline_matches_a_linear_scan() {
    let points = random_points(11, 2000, (30.0, 45.0), (130.0, 145.0));
    let zigzag = [
        (32.0, 131.0),
        (40.0, 134.0),
        (33.0, 138.0),
        (43.0, 141.0),
        (36.0, 144.0),
    ];
    let wrong = wrong_geo_modes(&points, &zigzag, 50_000.0);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: edges across the antimeridian reach the points on both
/// sides of it, and those stored at ±180.
#[test]
fn a_polyline_across_the_antimeridian_matches_a_linear_scan() {
    let mut points = random_points(12, 2000, (-30.0, 0.0), (160.0, 200.0));
    points.extend(SPECIAL_POINTS);
    points.extend([
        (-15.0, 180.0),
        (-15.0, -180.0),
        (-12.0, 179.9),
        (-12.0, -179.9),
    ]);
    let across = [
        (-25.0, 165.0),
        (-10.0, 178.0),
        (-20.0, -170.0),
        (-5.0, -162.0),
    ];
    let wrong = wrong_geo_modes(&points, &across, 100_000.0);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: edges near the poles: one whose arc rises far above its
/// ends, one through the north pole (its ends 180 degrees of longitude
/// apart), one ending at the south pole, whose longitude is any.
#[test]
fn a_polyline_near_a_pole_matches_a_linear_scan() {
    let mut points = random_points(13, 1500, (60.0, 90.0), (-180.0, 180.0));
    points.extend(random_points(14, 500, (-90.0, -60.0), (-180.0, 180.0)));
    points.extend(SPECIAL_POINTS);
    let mut wrong = Vec::new();
    for vertices in [
        vec![(75.0, 0.0), (75.0, 170.0)],
        vec![(70.0, 30.0), (70.0, -150.0)],
        vec![(-90.0, 123.0), (-70.0, -40.0), (-65.0, 60.0)],
    ] {
        wrong.extend(wrong_geo_modes(&points, &vertices, 200_000.0));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: awkward edges over points all over the sphere: on the
/// equator, along a meridian, 1 cm long, nearly antipodal, between two
/// ways of writing one point on the antimeridian, a repeated vertex; and a
/// distance of half the circumference or more, which every point is within.
#[test]
fn awkward_edges_match_a_linear_scan() {
    let mut points = random_points(15, 1500, (-90.0, 90.0), (-180.0, 180.0));
    points.extend(SPECIAL_POINTS);
    points.extend([(0.0, 25.0), (0.5, 25.0), (30.0, 20.0), (30.0, 20.5)]);
    let mut wrong = Vec::new();
    for (vertices, distance_m) in [
        (vec![(0.0, 10.0), (0.0, 40.0)], 100_000.0),
        (vec![(10.0, 20.0), (50.0, 20.0)], 100_000.0),
        (vec![(35.0, 139.0), (35.0, 139.000_000_1)], 100_000.0),
        (vec![(0.0, 0.0), (0.5, 179.5)], 100_000.0),
        (vec![(0.0, 180.0), (0.0, -180.0)], 300_000.0),
        (vec![(20.0, 60.0), (20.0, 60.0), (25.0, 70.0)], 300_000.0),
        (vec![(10.0, 10.0), (20.0, 20.0)], 2.1e7),
    ] {
        wrong.extend(wrong_geo_modes(&points, &vertices, distance_m));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: an arc between two points at 10°N, 170 degrees of longitude
/// apart, rises to about 63.7°N midway: the points by its top match, though
/// both ends are far south of them.
#[test]
fn an_edge_reaches_its_highest_latitude() {
    let (a, b) = ((10.0, 0.0), (10.0, 170.0));
    let top = lat_lon(axpy(1.0, unit(a), 1.0, unit(b)));
    let points = [
        top,
        (top.0 - 0.004, top.1),
        (top.0 + 0.004, top.1),
        (top.0 - 0.02, top.1),
        (10.0, 85.0),
    ];
    let wrong = wrong_geo_modes(&points, &[a, b], 1_000.0);
    assert!(wrong.is_empty(), "{wrong:#?}");
    let near = reference_to_polyline(points[1], &[a, b]);
    assert!(
        near < 1_000.0,
        "the test needs a point near the top: {near}"
    );
}

/// Issue #1596: a single vertex finds the documents, and gives the scores,
/// that `GeoDistanceQuery` does, a radius of 0 included.
#[test]
fn a_single_vertex_is_a_geo_distance() {
    let mut points = random_points(16, 1000, (34.0, 37.0), (138.0, 141.0));
    points.push((35.5, 139.5));
    let docs = geo_docs(&points);
    let mut wrong = Vec::new();
    for mode in geo_modes() {
        let store = store_with(mode, false, &docs);
        for distance_m in [0.0, 1.0, 30_000.0, 2.5e7] {
            let circle =
                GeoDistanceQuery::within_radius("location", 35.5, 139.5, distance_m).unwrap();
            let expected = hits(&store, Box::new(circle));
            // One vertex, and the same vertex twice: an edge of length 0.
            for vertices in [&[(35.5, 139.5)][..], &[(35.5, 139.5), (35.5, 139.5)]] {
                let polyline =
                    GeoPolylineQuery::within_distance("location", vertices, distance_m).unwrap();
                let found = hits(&store, Box::new(polyline));
                if found != expected {
                    wrong.push(format!(
                        "{mode:?} {vertices:?} within {distance_m} m: {} hits, geo_distance {}",
                        found.len(),
                        expected.len()
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a multi-valued document matches when any point is within
/// the distance, and scores by its closest point.
#[test]
fn a_multi_valued_document_scores_its_closest_point() {
    let line = [(35.0, 139.0), (35.0, 140.0)];
    let docs = vec![
        (1, geo_doc(&[(10.0, 10.0), (35.01, 139.5), (35.05, 139.5)])),
        (2, geo_doc(&[(35.05, 139.5), (35.01, 139.5), (10.0, 10.0)])),
        (3, geo_doc(&[(10.0, 10.0), (36.0, 139.5)])),
    ];
    let distance_m = 10_000.0;
    let closest = reference_to_polyline((35.01, 139.5), &line);
    let mut wrong = Vec::new();
    for mode in geo_modes() {
        let store = store_with(mode, true, &docs);
        let query = GeoPolylineQuery::within_distance("location", &line, distance_m).unwrap();
        let found = hits(&store, Box::new(query));
        let (mistakes, _) = mistakes(
            &found,
            &[(1, closest), (2, closest), (3, 1e9)],
            distance_m,
            1.0,
        );
        for mistake in mistakes {
            wrong.push(format!("{mode:?}: {mistake}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a document replaced in a later commit matches by its new
/// point only: the BKD's deletion filter drops the old one.
#[test]
fn a_replaced_document_matches_by_its_new_point() {
    let line = [(36.0, 138.0), (37.0, 141.0)];
    let distance_m = 60_000.0;
    let points = random_points(17, 1500, (34.0, 39.0), (136.0, 143.0));
    let mut wrong = Vec::new();
    for mode in geo_modes().into_iter().filter(|mode| mode.segments == 1) {
        let store = store_with(mode, false, &geo_docs(&points));
        let mut moved = points.clone();
        for (i, point) in moved.iter_mut().enumerate() {
            if i % 3 == 0 {
                *point = (73.0 - point.0, point.1);
            }
        }
        for (id, point) in (1..).zip(&moved).step_by(3) {
            store.upsert_document(id, geo_doc(&[*point])).unwrap();
        }
        store.commit().unwrap();
        let distances: Vec<(u64, f64)> = (1..)
            .zip(&moved)
            .map(|(id, &p)| (id, reference_to_polyline(p, &line)))
            .collect();
        let query = GeoPolylineQuery::within_distance("location", &line, distance_m).unwrap();
        let found = hits(&store, Box::new(query));
        for mistake in mistakes(&found, &distances, distance_m, 1.0).0 {
            wrong.push(format!("{mode:?}: {mistake}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: segments without the field neither fail nor match.
#[test]
fn a_field_in_some_segments_only() {
    let line = [(20.0, 20.0), (20.0, 21.0)];
    let body = |text: &str| Document::builder().add_text("body", text).build();
    let mut wrong = Vec::new();
    for mode in geo_modes().into_iter().filter(|mode| mode.segments == 1) {
        let store = empty_store(mode, false);
        store.upsert_document(1, body("alpha")).unwrap();
        store.upsert_document(2, body("bravo")).unwrap();
        store.commit().unwrap();
        store.upsert_document(3, geo_doc(&[(20.0, 20.5)])).unwrap();
        store.upsert_document(4, geo_doc(&[(0.0, 0.0)])).unwrap();
        store.commit().unwrap();
        store.upsert_document(5, body("charlie")).unwrap();
        store.commit().unwrap();
        let query = GeoPolylineQuery::within_distance("location", &line, 1_000.0).unwrap();
        let found: Vec<u64> = hits(&store, Box::new(query))
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        if found != [3] {
            wrong.push(format!("{mode:?}: found {found:?}, expected [3]"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: every hit scores `boost × (1 - d/r)`.
#[test]
fn every_hit_scores_the_boost_times_its_closeness() {
    let line = [(35.0, 139.0), (35.5, 139.5), (35.0, 140.0)];
    let distance_m = 40_000.0;
    let points = random_points(18, 800, (34.5, 36.0), (138.5, 140.5));
    let distances: Vec<(u64, f64)> = (1..)
        .zip(&points)
        .map(|(id, &p)| (id, reference_to_polyline(p, &line)))
        .collect();
    let mut wrong = Vec::new();
    for mode in geo_modes() {
        let store = store_with(mode, false, &geo_docs(&points));
        let query = GeoPolylineQuery::within_distance("location", &line, distance_m)
            .unwrap()
            .with_boost(2.5);
        let found = hits(&store, Box::new(query));
        for mistake in mistakes(&found, &distances, distance_m, 2.5).0 {
            wrong.push(format!("{mode:?}: {mistake}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a polyline needs a vertex, takes coordinates in range only,
/// and no edge between antipodal points; repeated vertices are fine, and so
/// is an "edge" between two ways of writing one point.
#[test]
fn a_polyline_is_validated() {
    let polyline = |vertices: &[(f64, f64)]| {
        GeoPolyline::new(
            vertices
                .iter()
                .map(|&(lat, lon)| GeoPoint { lat, lon })
                .collect(),
        )
    };
    let mut wrong = Vec::new();
    for (vertices, reason) in [
        (vec![], "at least one vertex"),
        (vec![(35.0, 139.0), (95.0, 139.0)], "latitude"),
        (vec![(35.0, 139.0), (35.0, 181.0)], "longitude"),
        (vec![(35.0, f64::NAN)], "longitude"),
        (vec![(0.0, 0.0), (0.0, 180.0)], "antipodal"),
        (vec![(10.0, 20.0), (-10.0, -160.0)], "antipodal"),
        (vec![(90.0, 0.0), (-90.0, 77.0)], "antipodal"),
        // About 111 m from antipodal.
        (vec![(0.0, 0.0), (0.0, 179.999)], "antipodal"),
    ] {
        match polyline(&vertices) {
            Ok(_) => wrong.push(format!("{vertices:?} accepted")),
            Err(e) if e.to_string().contains(reason) => {}
            Err(e) => wrong.push(format!("{vertices:?}: {e}, expected {reason:?}")),
        }
    }
    for vertices in [
        vec![(35.0, 139.0)],
        vec![(35.0, 139.0), (35.0, 139.0)],
        vec![(0.0, 180.0), (0.0, -180.0)],
        vec![(90.0, 0.0), (90.0, 45.0)],
        vec![(0.0, 0.0), (0.5, 179.5)],
        // About 1.1 km from antipodal.
        vec![(0.0, 0.0), (0.0, 179.99)],
    ] {
        if let Err(e) = polyline(&vertices) {
            wrong.push(format!("{vertices:?}: {e}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `n` random ECEF points from latitudes, longitudes and heights in the
/// given ranges.
fn random_ecef(seed: u64, n: usize, lats: (f64, f64), lons: (f64, f64)) -> Vec<GeoEcefPoint> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let lat = rng.between(lats.0, lats.1);
            let lon = rng.between(lons.0, lons.1);
            let height = rng.between(0.0, 3_000.0);
            wgs84_to_ecef(lat, lon, height)
        })
        .collect()
}

fn geo3d_docs(points: &[GeoEcefPoint]) -> Vec<(u64, Document)> {
    (1..)
        .zip(points)
        .map(|(id, &point)| (id, geo3d_doc(&[point])))
        .collect()
}

/// A flight path at 1,000 m, a vertex repeated (an edge of length 0).
fn flight_path() -> Vec<GeoEcefPoint> {
    [
        (35.0, 139.0),
        (35.4, 139.6),
        (35.4, 139.6),
        (35.1, 140.2),
        (35.8, 140.5),
    ]
    .iter()
    .map(|&(lat, lon)| wgs84_to_ecef(lat, lon, 1_000.0))
    .collect()
}

/// Issue #1596: a `Geo3d` polyline finds exactly the points within the
/// distance of its straight segments, each scoring `1 - d/r`.
#[test]
fn a_geo3d_polyline_matches_a_linear_scan() {
    let points = random_ecef(21, 2000, (34.5, 36.5), (138.5, 141.0));
    let vertices = flight_path();
    let distance_m = 20_000.0;
    let distances: Vec<(u64, f64)> = (1..)
        .zip(&points)
        .map(|(id, &p)| (id, reference_to_segments(p, &vertices)))
        .collect();
    let mut wrong = Vec::new();
    for mode in geo3d_modes() {
        let store = store_with(mode, false, &geo3d_docs(&points));
        let polyline = Geo3dPolyline::new(vertices.clone()).unwrap();
        let query = Geo3dPolylineQuery::new("position", polyline, distance_m);
        let found = hits(&store, Box::new(query));
        let (mistakes, ambiguous) = mistakes(&found, &distances, distance_m, 1.0);
        if ambiguous * 100 > points.len() {
            wrong.push(format!(
                "{mode:?}: {ambiguous} documents too near the distance"
            ));
        }
        for mistake in mistakes {
            wrong.push(format!("{mode:?}: {mistake}"));
        }
    }
    let within = distances.iter().filter(|(_, d)| *d <= distance_m).count();
    assert!(
        within > 50,
        "the test needs points within the distance: {within}"
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a single `Geo3d` vertex finds the documents, and gives the
/// scores, that `Geo3dDistanceQuery` does.
#[test]
fn a_single_vertex_is_a_geo3d_distance() {
    let points = random_ecef(22, 1000, (34.5, 36.5), (138.5, 141.0));
    let center = wgs84_to_ecef(35.5, 139.8, 500.0);
    let mut wrong = Vec::new();
    for mode in geo3d_modes() {
        let store = store_with(mode, false, &geo3d_docs(&points));
        for distance_m in [1.0, 30_000.0, 1e7] {
            let expected = hits(
                &store,
                Box::new(Geo3dDistanceQuery::new("position", center, distance_m)),
            );
            // One vertex, and the same vertex twice: a segment of length 0.
            for vertices in [vec![center], vec![center, center]] {
                let count = vertices.len();
                let polyline = Geo3dPolyline::new(vertices).unwrap();
                let found = hits(
                    &store,
                    Box::new(Geo3dPolylineQuery::new("position", polyline, distance_m)),
                );
                if found != expected {
                    wrong.push(format!(
                        "{mode:?} {count} vertices within {distance_m} m: {} hits, \
                         geo3d_distance {}",
                        found.len(),
                        expected.len()
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a multi-valued `Geo3d` document matches when any point is
/// within the distance, scores by its closest point, times the boost.
#[test]
fn a_multi_valued_geo3d_document_scores_its_closest_point() {
    let vertices = flight_path();
    let near = wgs84_to_ecef(35.2, 139.3, 2_000.0);
    let nearer = wgs84_to_ecef(35.2, 139.3, 1_200.0);
    let far = wgs84_to_ecef(10.0, 10.0, 0.0);
    let docs = vec![
        (1, geo3d_doc(&[far, near, nearer])),
        (2, geo3d_doc(&[nearer, near, far])),
        (3, geo3d_doc(&[far])),
    ];
    let distance_m = 10_000.0;
    let closest = reference_to_segments(nearer, &vertices);
    let mut wrong = Vec::new();
    for mode in geo3d_modes() {
        let store = store_with(mode, true, &docs);
        let polyline = Geo3dPolyline::new(vertices.clone()).unwrap();
        let query = Geo3dPolylineQuery::new("position", polyline, distance_m).with_boost(1.5);
        let found = hits(&store, Box::new(query));
        let reference = [(1, closest), (2, closest), (3, 1e9)];
        for mistake in mistakes(&found, &reference, distance_m, 1.5).0 {
            wrong.push(format!("{mode:?}: {mistake}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a `Geo3d` polyline needs a vertex and finite coordinates.
#[test]
fn a_geo3d_polyline_is_validated() {
    let mut wrong = Vec::new();
    for (vertices, reason) in [
        (vec![], "at least one vertex"),
        (vec![GeoEcefPoint::new(0.0, f64::NAN, 0.0)], "finite"),
        (
            vec![
                GeoEcefPoint::new(0.0, 0.0, 0.0),
                GeoEcefPoint::new(f64::INFINITY, 0.0, 0.0),
            ],
            "finite",
        ),
    ] {
        match Geo3dPolyline::new(vertices.clone()) {
            Ok(_) => wrong.push(format!("{vertices:?} accepted")),
            Err(e) if e.to_string().contains(reason) => {}
            Err(e) => wrong.push(format!("{vertices:?}: {e}, expected {reason:?}")),
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
