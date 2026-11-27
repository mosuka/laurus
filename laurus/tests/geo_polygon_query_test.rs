//! Issue #1596: `GeoPolygonQuery` matches the points of a `Geo` field inside
//! a polygon with holes: exactly what a linear scan finds, through the BKD
//! (with and without stored values) and over the stored documents, in one
//! segment and in two.
//!
//! The reference here is a crossing-number test written for these tests,
//! not the query's own code. The random points are never on an edge, so the
//! boundary is checked separately, with points placed on it.

use std::sync::Arc;

use laurus::lexical::core::field::{FieldOption, GeoOption, TextOption};
use laurus::lexical::query::geo::{GeoPolygon, GeoPolygonQuery};
use laurus::lexical::query::{GeoPoint, Query};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};

/// How a store indexes its geo field, and how many commits it makes.
#[derive(Clone, Copy, Debug)]
struct Mode {
    indexed: bool,
    stored: bool,
    segments: usize,
}

/// The BKD path with and without stored values and the fallback over stored
/// documents, in one segment and in two.
fn modes() -> Vec<Mode> {
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

/// An empty store whose geo field `location` is indexed and stored as
/// `mode` says, with a text field `body` too.
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
        .add_field("body", FieldOption::Text(TextOption::default()))
        .build();
    LexicalStore::new(storage, config).unwrap()
}

/// A store whose geo field is indexed and stored as `mode` says, holding
/// `docs`, committed in `mode.segments` commits.
fn store_with(mode: Mode, multi_valued: bool, docs: &[(u64, Vec<(f64, f64)>)]) -> LexicalStore {
    let store = empty_store(mode, multi_valued);
    let per_segment = docs.len().div_ceil(mode.segments);
    for chunk in docs.chunks(per_segment) {
        for (id, points) in chunk {
            store.upsert_document(*id, geo_doc(points)).unwrap();
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

fn ids(store: &LexicalStore, query: Box<dyn Query>) -> Vec<u64> {
    hits(store, query).into_iter().map(|(id, _)| id).collect()
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

/// `n` documents with one random point each, in the given ranges; ids from 1.
fn random_docs(
    seed: u64,
    n: u64,
    lats: (f64, f64),
    lons: (f64, f64),
) -> Vec<(u64, Vec<(f64, f64)>)> {
    let mut rng = Rng(seed);
    (1..=n)
        .map(|id| {
            let lat = rng.between(lats.0, lats.1);
            let lon = rng.between(lons.0, lons.1);
            (id, vec![(lat, lon)])
        })
        .collect()
}

/// Whether `(lat, lon)` is inside `ring` (`(lat, lon)` vertices, closed), by
/// the crossing number; for points that are not on an edge.
fn reference_in_ring(ring: &[(f64, f64)], lat: f64, lon: f64) -> bool {
    let mut inside = false;
    for pair in ring.windows(2) {
        let ((lat_a, lon_a), (lat_b, lon_b)) = (pair[0], pair[1]);
        if (lat_a > lat) != (lat_b > lat) {
            let crossing = lon_a + (lat - lat_a) * (lon_b - lon_a) / (lat_b - lat_a);
            if lon < crossing {
                inside = !inside;
            }
        }
    }
    inside
}

/// The ids whose point is inside `exterior` and in none of `holes`.
fn reference_ids(
    docs: &[(u64, Vec<(f64, f64)>)],
    exterior: &[(f64, f64)],
    holes: &[&[(f64, f64)]],
    lon_of: impl Fn(f64) -> f64,
) -> Vec<u64> {
    docs.iter()
        .filter(|(_, points)| {
            points.iter().any(|&(lat, lon)| {
                let lon = lon_of(lon);
                reference_in_ring(exterior, lat, lon)
                    && !holes.iter().any(|hole| reference_in_ring(hole, lat, lon))
            })
        })
        .map(|(id, _)| *id)
        .collect()
}

fn polygon(exterior: &[(f64, f64)], holes: &[&[(f64, f64)]]) -> GeoPolygon {
    let ring = |points: &[(f64, f64)]| -> Vec<GeoPoint> {
        points
            .iter()
            .map(|&(lat, lon)| GeoPoint::new(lat, lon))
            .collect()
    };
    GeoPolygon::new(
        ring(exterior),
        holes.iter().map(|hole| ring(hole)).collect(),
    )
    .unwrap()
}

fn query(polygon: GeoPolygon) -> Box<dyn Query> {
    Box::new(GeoPolygonQuery::new("location", polygon))
}

/// The modes in which a polygon does not find what the reference finds.
fn wrong_modes(
    docs: &[(u64, Vec<(f64, f64)>)],
    multi_valued: bool,
    shape: &GeoPolygon,
    expected: &[u64],
) -> Vec<String> {
    let mut wrong = Vec::new();
    for mode in modes() {
        let store = store_with(mode, multi_valued, docs);
        let found = ids(&store, query(shape.clone()));
        if found != expected {
            let missing: Vec<&u64> = expected.iter().filter(|id| !found.contains(id)).collect();
            let extra: Vec<&u64> = found.iter().filter(|id| !expected.contains(id)).collect();
            wrong.push(format!("{mode:?}: missing {missing:?}, extra {extra:?}"));
        }
    }
    wrong
}

/// Issue #1596: a polygon with a hole matches what a linear scan finds.
#[test]
fn a_polygon_with_a_hole_matches_a_linear_scan() {
    let exterior = [
        (10.0, 10.0),
        (10.0, 30.0),
        (30.0, 30.0),
        (30.0, 10.0),
        (10.0, 10.0),
    ];
    let hole = [
        (15.0, 15.0),
        (15.0, 25.0),
        (25.0, 25.0),
        (25.0, 15.0),
        (15.0, 15.0),
    ];
    let docs = random_docs(1, 2_000, (0.0, 40.0), (0.0, 40.0));
    let expected = reference_ids(&docs, &exterior, &[&hole], |lon| lon);
    assert!(
        expected.len() > 100,
        "too few points inside: {}",
        expected.len()
    );
    let wrong = wrong_modes(&docs, false, &polygon(&exterior, &[&hole]), &expected);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a concave polygon, a U, matches what a linear scan finds,
/// and nothing between its arms.
#[test]
fn a_concave_polygon_matches_a_linear_scan() {
    let exterior = [
        (0.0, 0.0),
        (0.0, 30.0),
        (30.0, 30.0),
        (30.0, 20.0),
        (10.0, 20.0),
        (10.0, 10.0),
        (30.0, 10.0),
        (30.0, 0.0),
        (0.0, 0.0),
    ];
    let docs = random_docs(2, 2_000, (-5.0, 35.0), (-5.0, 35.0));
    let expected = reference_ids(&docs, &exterior, &[], |lon| lon);
    let between_the_arms = docs
        .iter()
        .filter(|(_, points)| {
            let (lat, lon) = points[0];
            lat > 10.0 && lat < 30.0 && lon > 10.0 && lon < 20.0
        })
        .count();
    assert!(
        between_the_arms > 50,
        "the test needs points between the arms"
    );
    let wrong = wrong_modes(&docs, false, &polygon(&exterior, &[]), &expected);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: an edge from 170 to -170 degrees takes the shorter way,
/// across the antimeridian, and the polygon matches what a linear scan finds
/// with the western longitudes shifted by 360 degrees.
#[test]
fn a_polygon_across_the_antimeridian_matches_a_linear_scan() {
    let exterior = [
        (-20.0, 170.0),
        (-20.0, -170.0),
        (-10.0, -170.0),
        (-10.0, 170.0),
    ];
    let shifted: Vec<(f64, f64)> = exterior
        .iter()
        .chain(std::iter::once(&exterior[0]))
        .map(|&(lat, lon)| (lat, if lon < 0.0 { lon + 360.0 } else { lon }))
        .collect();
    let mut docs = random_docs(3, 1_000, (-25.0, -5.0), (160.0, 180.0));
    docs.extend(
        random_docs(4, 1_000, (-25.0, -5.0), (-180.0, -160.0))
            .into_iter()
            .map(|(id, points)| (id + 1_000, points)),
    );
    let shift = |lon: f64| if lon < 0.0 { lon + 360.0 } else { lon };
    let expected = reference_ids(&docs, &shifted, &[], shift);
    assert!(
        expected.iter().any(|id| *id <= 1_000) && expected.iter().any(|id| *id > 1_000),
        "the test needs points on both sides"
    );
    let wrong = wrong_modes(&docs, false, &polygon(&exterior, &[]), &expected);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a point on an exterior edge or vertex matches, and one on a
/// hole's edge or vertex does not, as in Lucene. Points on ±180 degrees
/// match a polygon across the antimeridian whichever sign they are stored
/// with.
#[test]
fn the_boundary_belongs_to_the_exterior_and_to_the_holes() {
    let exterior = [(10.0, 10.0), (10.0, 30.0), (30.0, 30.0), (30.0, 10.0)];
    let hole = [(15.0, 15.0), (15.0, 25.0), (25.0, 25.0), (25.0, 15.0)];
    let docs: Vec<(u64, Vec<(f64, f64)>)> = vec![
        (1, vec![(10.0, 20.0)]), // on an exterior edge
        (2, vec![(30.0, 30.0)]), // on an exterior vertex
        (3, vec![(20.0, 15.0)]), // on a hole's edge
        (4, vec![(25.0, 25.0)]), // on a hole's vertex
        (5, vec![(12.0, 12.0)]), // inside
        (6, vec![(20.0, 20.0)]), // in the hole
        (7, vec![(9.0, 20.0)]),  // outside
    ];
    let wrong = wrong_modes(&docs, false, &polygon(&exterior, &[&hole]), &[1, 2, 5]);
    assert!(wrong.is_empty(), "{wrong:#?}");

    let across = [
        (-20.0, 170.0),
        (-20.0, -170.0),
        (-10.0, -170.0),
        (-10.0, 170.0),
    ];
    let docs: Vec<(u64, Vec<(f64, f64)>)> = vec![
        (1, vec![(-15.0, 180.0)]),
        (2, vec![(-15.0, -180.0)]),
        (3, vec![(-15.0, 0.0)]),
    ];
    let wrong = wrong_modes(&docs, false, &polygon(&across, &[]), &[1, 2]);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: in a polygon across the antimeridian, a hole on the western
/// side of it, written with western longitudes, and a hole across the
/// antimeridian too, written from either side, each lie where they are
/// written, in the exterior.
#[test]
fn a_hole_across_or_beside_the_antimeridian_lies_in_the_exterior() {
    let exterior = [
        (-20.0, 170.0),
        (-20.0, -170.0),
        (-10.0, -170.0),
        (-10.0, 170.0),
    ];
    let west = [
        (-17.0, -178.0),
        (-17.0, -172.0),
        (-13.0, -172.0),
        (-13.0, -178.0),
    ];
    let across_from_the_east = [
        (-17.0, 178.0),
        (-17.0, -178.0),
        (-13.0, -178.0),
        (-13.0, 178.0),
    ];
    let across_from_the_west = [
        (-17.0, -178.0),
        (-17.0, 178.0),
        (-13.0, 178.0),
        (-13.0, -178.0),
    ];
    let docs: Vec<(u64, Vec<(f64, f64)>)> = vec![
        (1, vec![(-15.0, -175.0)]), // in the western hole
        (2, vec![(-15.0, -171.0)]), // east of the western hole
        (3, vec![(-15.0, 175.0)]),  // in the exterior
        (4, vec![(-15.0, 180.0)]),  // in the hole across the antimeridian
        (5, vec![(-15.0, -179.0)]), // in the hole across the antimeridian
        (6, vec![(-15.0, 179.0)]),  // in the hole across the antimeridian
    ];
    let mut wrong = Vec::new();
    for (name, hole, expected) in [
        ("west", &west, vec![2, 3, 4, 5, 6]),
        (
            "across, from the east",
            &across_from_the_east,
            vec![1, 2, 3],
        ),
        (
            "across, from the west",
            &across_from_the_west,
            vec![1, 2, 3],
        ),
    ] {
        for found in wrong_modes(&docs, false, &polygon(&exterior, &[hole]), &expected) {
            wrong.push(format!("{name}: {found}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a multi-valued document matches when any of its points is
/// inside, and is reported once.
#[test]
fn a_multi_valued_document_matches_when_any_point_is_inside() {
    let exterior = [(10.0, 10.0), (10.0, 30.0), (30.0, 30.0), (30.0, 10.0)];
    let docs: Vec<(u64, Vec<(f64, f64)>)> = vec![
        (1, vec![(0.0, 0.0), (20.0, 20.0)]),   // the second point inside
        (2, vec![(20.0, 20.0), (21.0, 21.0)]), // both inside
        (3, vec![(0.0, 0.0), (40.0, 40.0)]),   // neither
    ];
    let wrong = wrong_modes(&docs, true, &polygon(&exterior, &[]), &[1, 2]);
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a document replaced in a later commit matches by its new
/// point only, though the earlier segment still holds the old one. There
/// are enough points for the BKD to have cells inside the polygon, which it
/// takes whole, so the old points are dropped both from those cells and
/// from the cells an edge crosses.
#[test]
fn a_replaced_document_matches_by_its_new_point() {
    let exterior = [
        (10.0, 10.0),
        (10.0, 30.0),
        (30.0, 30.0),
        (30.0, 10.0),
        (10.0, 10.0),
    ];
    let before = random_docs(6, 20_000, (0.0, 40.0), (0.0, 40.0));
    // Every third document moves: one inside moves out, and one outside
    // moves in.
    let after: Vec<(u64, Vec<(f64, f64)>)> = before
        .iter()
        .filter(|(id, _)| id % 3 == 0)
        .map(|(id, points)| {
            let (lat, lon) = points[0];
            let moved = if reference_in_ring(&exterior, lat, lon) {
                // Latitudes 10 to 30 go to -15 to 5, south of the polygon.
                (lat - 25.0, lon)
            } else {
                (20.0 + (lat - 20.0) / 4.0, 20.0 + (lon - 20.0) / 4.0)
            };
            (*id, vec![moved])
        })
        .collect();
    let mut current = before.clone();
    for (id, points) in &after {
        current[*id as usize - 1].1 = points.clone();
    }
    let expected = reference_ids(&current, &exterior, &[], |lon| lon);

    let mut wrong = Vec::new();
    for mode in modes().into_iter().filter(|mode| mode.segments == 1) {
        let store = empty_store(mode, false);
        for (id, points) in &before {
            store.upsert_document(*id, geo_doc(points)).unwrap();
        }
        store.commit().unwrap();
        for (id, points) in &after {
            store.upsert_document(*id, geo_doc(points)).unwrap();
        }
        store.commit().unwrap();
        let found = ids(&store, query(polygon(&exterior, &[])));
        if found != expected {
            let missing = expected.iter().filter(|id| !found.contains(id)).count();
            let extra = found.iter().filter(|id| !expected.contains(id)).count();
            wrong.push(format!("{mode:?}: {missing} missing, {extra} extra"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: with the geo field in some segments only, the segments
/// without it match nothing and the others match their points.
#[test]
fn a_field_in_some_segments_only() {
    let exterior = [(10.0, 10.0), (10.0, 30.0), (30.0, 30.0), (30.0, 10.0)];
    let body = |text: &str| Document::builder().add_text("body", text).build();
    let mut wrong = Vec::new();
    for mode in modes().into_iter().filter(|mode| mode.segments == 1) {
        let store = empty_store(mode, false);
        store.upsert_document(1, body("alpha")).unwrap();
        store.upsert_document(2, body("bravo")).unwrap();
        store.commit().unwrap();
        store.upsert_document(3, geo_doc(&[(20.0, 20.0)])).unwrap();
        store.upsert_document(4, geo_doc(&[(0.0, 0.0)])).unwrap();
        store.commit().unwrap();
        store.upsert_document(5, body("charlie")).unwrap();
        store.commit().unwrap();
        let found = ids(&store, query(polygon(&exterior, &[])));
        if found != [3] {
            wrong.push(format!("{mode:?}: found {found:?}, expected [3]"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: every hit scores the boost, whatever its place in the
/// polygon.
#[test]
fn every_hit_scores_the_boost() {
    let exterior = [(10.0, 10.0), (10.0, 30.0), (30.0, 30.0), (30.0, 10.0)];
    let docs = random_docs(5, 500, (0.0, 40.0), (0.0, 40.0));
    let mut wrong = Vec::new();
    for mode in modes() {
        let store = store_with(mode, false, &docs);
        let boosted = GeoPolygonQuery::new("location", polygon(&exterior, &[])).with_boost(2.5);
        let found = hits(&store, Box::new(boosted));
        if found.is_empty() || found.iter().any(|(_, score)| (score - 2.5).abs() > 1e-6) {
            wrong.push(format!("{mode:?}: {:?}", &found[..found.len().min(5)]));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a ring needs 3 distinct points and coordinates in range, an
/// unclosed ring is closed, a ring around a pole and an edge across exactly
/// 180 degrees of longitude are rejected.
#[test]
fn a_polygon_is_validated() {
    // Not `GeoPoint::new`, which asserts the range in debug builds: the
    // polygon checks it itself, for points built without it.
    let ring = |points: &[(f64, f64)]| -> Vec<GeoPoint> {
        points
            .iter()
            .map(|&(lat, lon)| GeoPoint { lat, lon })
            .collect()
    };
    let rejected = [
        ("two points", ring(&[(0.0, 0.0), (1.0, 1.0)])),
        (
            "two points, closed",
            ring(&[(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
        ),
        (
            "a latitude out of range",
            ring(&[(0.0, 0.0), (95.0, 1.0), (1.0, 0.0)]),
        ),
        (
            "a longitude out of range",
            ring(&[(0.0, 0.0), (1.0, 181.0), (1.0, 0.0)]),
        ),
        ("a NaN", ring(&[(0.0, 0.0), (f64::NAN, 1.0), (1.0, 0.0)])),
        (
            "around the north pole",
            ring(&[(80.0, 0.0), (80.0, 90.0), (80.0, 180.0), (80.0, -90.0)]),
        ),
        (
            "an edge across 180 degrees",
            ring(&[(0.0, 0.0), (0.0, 180.0), (10.0, 90.0)]),
        ),
    ];
    let mut wrong = Vec::new();
    for (name, exterior) in rejected {
        if let Ok(polygon) = GeoPolygon::new(exterior, Vec::new()) {
            wrong.push(format!("{name}: accepted as {polygon:?}"));
        }
    }
    let square = [(10.0, 10.0), (10.0, 30.0), (30.0, 30.0), (30.0, 10.0)];
    let mut closed = square.to_vec();
    closed.push(square[0]);
    match (
        GeoPolygon::new(ring(&square), Vec::new()),
        GeoPolygon::new(ring(&closed), Vec::new()),
    ) {
        (Ok(unclosed), Ok(closed)) if unclosed == closed => {}
        (unclosed, closed) => wrong.push(format!("unclosed {unclosed:?}, closed {closed:?}")),
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
