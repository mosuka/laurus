//! Integration tests for issue #1595 — a 2D bounding box that crosses the
//! antimeridian (`min.lon > max.lon`, RFC 7946 section 5.2), a box whose
//! edge lies on ±180°, and a box with zero width or height.
//!
//! The cases that check which documents match run over the BKD path
//! (`indexed = true`, also without stored values) and over the
//! stored-document fallback (`indexed = false`), with the documents in one
//! segment and in two.

use std::sync::Arc;

use laurus::lexical::core::field::{FieldOption, GeoOption};
use laurus::lexical::query::geo::{GeoBoundingBox, GeoBoundingBoxQuery};
use laurus::lexical::query::{GeoPoint, Query};
use laurus::lexical::search::searcher::LexicalSearchQuery;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{
    DataValue, Document, Engine, FieldOption as EngineFieldOption, Schema, SearchRequestBuilder,
    TextOption,
};

/// The documents, by id.
const DOCS: [(u64, f64, f64); 9] = [
    (1, -17.0, 179.0),  // just west of the antimeridian
    (2, -17.0, -179.0), // just east of it
    (3, -17.0, 0.0),    // far from it
    (4, -17.0, 180.0),  // on it
    (5, -17.0, -180.0), // on it, stored with the other sign
    (6, -17.0, 174.0),  // farther west of it
    (7, -17.0, -174.0), // farther east of it
    (8, -25.0, 179.0),  // outside the latitudes
    (9, 35.68, 139.76), // Tokyo
];

/// How a store indexes its geo field, and how many commits it makes.
#[derive(Clone, Copy, Debug)]
struct Mode {
    indexed: bool,
    stored: bool,
    segments: usize,
}

/// Every mode: the BKD path with and without stored values and the fallback
/// over stored documents, in one segment and in two.
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

/// A store whose geo field is indexed and stored as `mode` says, holding
/// `docs`, committed in `mode.segments` commits.
fn store_with(mode: Mode, multi_valued: bool, docs: &[(u64, Vec<(f64, f64)>)]) -> LexicalStore {
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
        .build();
    let store = LexicalStore::new(storage, config).unwrap();
    let per_segment = docs.len().div_ceil(mode.segments);
    for chunk in docs.chunks(per_segment) {
        for (id, points) in chunk {
            store.upsert_document(*id, geo_doc(points)).unwrap();
        }
        store.commit().unwrap();
    }
    store
}

fn default_docs() -> Vec<(u64, Vec<(f64, f64)>)> {
    DOCS.iter()
        .map(|&(id, lat, lon)| (id, vec![(lat, lon)]))
        .collect()
}

fn bbox(min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> Box<dyn Query> {
    Box::new(
        GeoBoundingBoxQuery::within_bounding_box("location", min_lat, min_lon, max_lat, max_lon)
            .unwrap(),
    )
}

/// The ids a query finds, sorted.
fn hits(store: &LexicalStore, query: Box<dyn Query>) -> Vec<u64> {
    let mut ids: Vec<u64> = store
        .search(LexicalSearchRequest::new(query))
        .unwrap()
        .hits
        .iter()
        .map(|hit| hit.doc_id)
        .collect();
    ids.sort_unstable();
    ids
}

/// #1595: `min.lon > max.lon` is a box that crosses the antimeridian. It
/// matches the points on both sides of it and on it, whichever sign they
/// are stored with, and neither the points outside the width nor those
/// outside the latitudes.
#[test]
fn a_box_across_the_antimeridian_matches_both_sides() {
    for mode in modes() {
        let store = store_with(mode, false, &default_docs());
        assert_eq!(
            hits(&store, bbox(-20.0, 175.0, -15.0, -175.0)),
            [1, 2, 4, 5],
            "{mode:?}"
        );
        // A narrower one, and one that reaches farther west than east.
        assert_eq!(
            hits(&store, bbox(-20.0, 178.5, -15.0, -178.5)),
            [1, 2, 4, 5],
            "{mode:?}"
        );
        assert_eq!(
            hits(&store, bbox(-20.0, 173.0, -15.0, -179.5)),
            [1, 4, 5, 6],
            "{mode:?}"
        );
        // The same edges swapped are the rest of the globe: `[-175, 175]`
        // holds -174, 0 and 174, and not ±180.
        assert_eq!(
            hits(&store, bbox(-20.0, -175.0, -15.0, 175.0)),
            [3, 6, 7],
            "{mode:?}"
        );
    }
}

/// #1595: an edge on exactly ±180° covers the meridian whichever sign a
/// point is stored with, as `geo_distance` does.
#[test]
fn a_box_edge_on_the_antimeridian_matches_both_signs() {
    for mode in modes() {
        let store = store_with(mode, false, &default_docs());
        assert_eq!(
            hits(&store, bbox(-20.0, 175.0, -15.0, 180.0)),
            [1, 4, 5],
            "{mode:?}"
        );
        assert_eq!(
            hits(&store, bbox(-20.0, -180.0, -15.0, -175.0)),
            [2, 4, 5],
            "{mode:?}"
        );
        // The whole circle matches each point once.
        assert_eq!(
            hits(&store, bbox(-20.0, -180.0, -15.0, 180.0)),
            [1, 2, 3, 4, 5, 6, 7],
            "{mode:?}"
        );
    }
}

/// #1595: a box with zero width, zero height or zero size matches the
/// points on it, as `GeoBoundingBox::contains` says. It used to be reported
/// as empty, and the search stopped there.
#[test]
fn a_degenerate_box_matches_the_points_on_it() {
    for mode in modes() {
        let store = store_with(mode, false, &default_docs());
        // Zero width: the 139.76 meridian, between latitudes 30 and 40.
        assert_eq!(
            hits(&store, bbox(30.0, 139.76, 40.0, 139.76)),
            [9],
            "{mode:?}"
        );
        // Zero height: the parallel of latitude 35.68.
        assert_eq!(
            hits(&store, bbox(35.68, 130.0, 35.68, 150.0)),
            [9],
            "{mode:?}"
        );
        // Zero size: one point, and one point where nothing is.
        assert_eq!(
            hits(&store, bbox(35.68, 139.76, 35.68, 139.76)),
            [9],
            "{mode:?}"
        );
        assert_eq!(
            hits(&store, bbox(10.0, 10.0, 10.0, 10.0)),
            [] as [u64; 0],
            "{mode:?}"
        );
        // Zero width on ±180: the two points stored with either sign.
        assert_eq!(
            hits(&store, bbox(-20.0, 180.0, -15.0, 180.0)),
            [4, 5],
            "{mode:?}"
        );
        assert_eq!(
            hits(&store, bbox(-20.0, -180.0, -15.0, -180.0)),
            [4, 5],
            "{mode:?}"
        );
    }
}

/// #1595: a multi-valued document with points on both sides of the
/// antimeridian is reported once.
#[test]
fn a_multi_valued_document_is_reported_once_for_a_box_across_the_antimeridian() {
    for mode in modes() {
        let store = store_with(
            mode,
            true,
            &[
                (1, vec![(-17.0, 179.0), (-17.0, -179.0)]),
                (2, vec![(-17.0, 0.0), (-17.0, -178.0)]),
                (3, vec![(-17.0, 10.0), (-17.0, 20.0)]),
            ],
        );
        assert_eq!(
            hits(&store, bbox(-20.0, 175.0, -15.0, -175.0)),
            [1, 2],
            "{mode:?}"
        );
    }
}

/// #1595: the score falls with the distance from the center of the box,
/// which for this box is on the antimeridian. The points at equal distances
/// from it get equal scores, and the point on it gets the highest.
#[test]
fn a_box_across_the_antimeridian_scores_from_its_center() {
    let store = store_with(
        Mode {
            indexed: true,
            stored: true,
            segments: 1,
        },
        false,
        &default_docs(),
    );
    let hits = store
        .search(LexicalSearchRequest::new(bbox(-22.0, 175.0, -12.0, -175.0)))
        .unwrap()
        .hits;
    let score = |id: u64| {
        hits.iter()
            .find(|hit| hit.doc_id == id)
            .unwrap_or_else(|| panic!("document {id} must match"))
            .score
    };
    assert!(score(4) > score(1), "the point on it scores highest");
    assert_eq!(score(4), score(5), "either sign of ±180 is the same point");
    assert!((score(1) - score(2)).abs() < 1e-4, "east and west mirror");
}

/// #1595: every point of a box with zero size is its center, so it gets the
/// full score, also when ±180° is stored with the other sign (the distance
/// between the two signs is not exactly 0 in floating point).
#[test]
fn a_box_with_zero_size_scores_its_points_as_its_center() {
    for mode in modes() {
        let store = store_with(mode, false, &default_docs());
        for lon in [180.0, -180.0] {
            let hits = store
                .search(LexicalSearchRequest::new(bbox(-17.0, lon, -17.0, lon)))
                .unwrap()
                .hits;
            let mut scores: Vec<(u64, f32)> =
                hits.iter().map(|hit| (hit.doc_id, hit.score)).collect();
            scores.sort_by_key(|&(id, _)| id);
            assert_eq!(scores, [(4, 1.0), (5, 1.0)], "lon {lon}, {mode:?}");
        }
    }
}

/// #1595: only `min.lat > max.lat` is rejected.
#[test]
fn a_box_rejects_inverted_latitudes_only() {
    let point = |lat, lon| GeoPoint::try_new(lat, lon).unwrap();
    assert!(GeoBoundingBox::new(point(-20.0, 175.0), point(-15.0, -175.0)).is_ok());
    let err = GeoBoundingBox::new(point(-15.0, 175.0), point(-20.0, -175.0))
        .unwrap_err()
        .to_string();
    assert!(err.contains("min latitude"), "unexpected error: {err}");
}

/// #1595 through the engine and the query DSL: the issue's example, and an
/// inverted latitude is still an error.
#[tokio::test]
async fn the_dsl_accepts_a_box_across_the_antimeridian() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let schema = Schema::builder()
        .add_field("title", EngineFieldOption::Text(TextOption::default()))
        .add_field("loc", EngineFieldOption::Geo(GeoOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await.unwrap();
    for (id, lat, lon) in [
        ("east", -17.0, 179.0),
        ("west", -17.0, -179.0),
        ("zero", -17.0, 0.0),
    ] {
        let doc = Document::builder()
            .add_text("title", id)
            .add_geo("loc", lat, lon)
            .build();
        engine.put_document(id, doc).await.unwrap();
        engine.commit().await.unwrap();
    }

    let search = async |dsl: &str| {
        let request = SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(dsl))
            .limit(10)
            .build();
        engine.search(request).await.map(|hits| {
            let mut ids: Vec<String> = hits.into_iter().map(|hit| hit.id).collect();
            ids.sort();
            ids
        })
    };
    assert_eq!(
        search("loc:geo_bbox(-20, 175, -15, -175)").await.unwrap(),
        ["east", "west"]
    );
    assert_eq!(
        search("loc:geo_bbox(-20, -175, -15, 175)").await.unwrap(),
        ["zero"]
    );
    let err = search("loc:geo_bbox(-15, 175, -20, -175)")
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("min latitude"),
        "an inverted latitude must be rejected, got: {err}"
    );
}
