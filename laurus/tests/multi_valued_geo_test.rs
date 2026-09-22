//! End-to-end tests for multi-valued geo fields (Issue #1174).
//!
//! Drives `LexicalStore` the way production search does and verifies the
//! Lucene-style "any point matches" semantics for `GeoArray` (2-D) and
//! `GeoEcefArray` (3-D ECEF) fields, including:
//!
//! - a document is reported **once** even when several of its points
//!   match, scored by its **closest** point;
//! - the distance query's rectangle pre-filter no longer drops a document
//!   whose first BKD-traversed point lies in a rectangle corner outside the
//!   circle while a later point lies inside it (the first-occurrence dedup
//!   that `GeoBoxPointsVisitor::into_candidates` used to apply);
//! - the stored-fields fallback (`indexed = false`) sees every point;
//! - single-valued fields behave exactly as before.

use std::sync::Arc;

use laurus::lexical::core::field::{FieldOption, Geo3dOption, GeoOption, TextOption};
use laurus::lexical::query::geo::{GeoBoundingBoxQuery, GeoDistanceQuery};
use laurus::lexical::query::{
    Geo3dBoundingBoxQuery, Geo3dDistanceQuery, Geo3dNearestQuery, GeoPoint, Query,
};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document, GeoEcefPoint};

const TOKYO: (f64, f64) = (35.68, 139.76);
const YOKOHAMA: (f64, f64) = (35.44, 139.64);
const OSAKA: (f64, f64) = (34.69, 135.50);
const SAPPORO: (f64, f64) = (43.06, 141.35);

fn point((lat, lon): (f64, f64)) -> GeoPoint {
    GeoPoint::try_new(lat, lon).unwrap()
}

fn geo_doc(p: (f64, f64)) -> Document {
    Document::builder()
        .add_field("location", DataValue::Geo(point(p)))
        .build()
}

fn geo_array_doc(points: &[(f64, f64)]) -> Document {
    Document::builder()
        .add_geo_array("location", points.iter().copied().map(point).collect())
        .build()
}

fn geo_store(indexed: bool, stored: bool, multi_valued: bool) -> LexicalStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "location",
            FieldOption::Geo(GeoOption {
                indexed,
                stored,
                multi_valued,
                doc_values: true,
            }),
        )
        .add_field("tag", FieldOption::Text(TextOption::default()))
        .build();
    LexicalStore::new(storage, config).unwrap()
}

fn geo3d_store() -> LexicalStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "position",
            FieldOption::Geo3d(Geo3dOption {
                multi_valued: true,
                ..Default::default()
            }),
        )
        .build();
    LexicalStore::new(storage, config).unwrap()
}

/// Sorted doc ids of every hit.
fn search_ids(store: &LexicalStore, query: Box<dyn Query>) -> Vec<u64> {
    let mut ids: Vec<u64> = search_hits(store, query)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    ids.sort_unstable();
    ids
}

/// `(doc_id, score)` of every hit in ranking order.
fn search_hits(store: &LexicalStore, query: Box<dyn Query>) -> Vec<(u64, f32)> {
    store
        .search(LexicalSearchRequest::new(query).limit(1_000))
        .unwrap()
        .hits
        .iter()
        .map(|hit| (hit.doc_id, hit.score))
        .collect()
}

/// The stored `field` of `doc_id`, fetched through a match-all box query
/// with `load_documents`.
fn stored_field(store: &LexicalStore, doc_id: u64, field: &str) -> DataValue {
    let results = store
        .search(
            LexicalSearchRequest::new(within_box((-90.0, -180.0), (90.0, 180.0)))
                .limit(1_000)
                .load_documents(true),
        )
        .unwrap();
    results
        .hits
        .iter()
        .find(|hit| hit.doc_id == doc_id)
        .unwrap_or_else(|| panic!("doc {doc_id} not among hits"))
        .document
        .as_ref()
        .expect("document loaded")
        .get_field(field)
        .cloned()
        .unwrap_or_else(|| panic!("doc {doc_id} has no stored field {field}"))
}

fn within_radius(center: (f64, f64), radius_m: f64) -> Box<dyn Query> {
    Box::new(GeoDistanceQuery::within_radius("location", center.0, center.1, radius_m).unwrap())
}

fn within_box(min: (f64, f64), max: (f64, f64)) -> Box<dyn Query> {
    Box::new(
        GeoBoundingBoxQuery::within_bounding_box("location", min.0, min.1, max.0, max.1).unwrap(),
    )
}

// ---------------------------------------------------------------------------
// 2-D GeoArray
// ---------------------------------------------------------------------------

/// Any-match: a document matches if at least one of its points is within
/// the radius, and the array as a whole is stored and read back intact.
#[test]
fn geo_array_distance_query_matches_if_any_point_is_within_radius() {
    let store = geo_store(true, true, true);

    store
        .upsert_document(1, geo_array_doc(&[OSAKA, TOKYO]))
        .unwrap(); // Tokyo point matches
    store
        .upsert_document(2, geo_array_doc(&[OSAKA, SAPPORO]))
        .unwrap(); // nothing near Tokyo
    store
        .upsert_document(3, geo_array_doc(&[YOKOHAMA]))
        .unwrap(); // single element
    store.commit().unwrap();

    assert_eq!(
        search_ids(&store, within_radius(TOKYO, 50_000.0)),
        vec![1, 3]
    );
    assert_eq!(
        search_ids(&store, within_radius(OSAKA, 50_000.0)),
        vec![1, 2]
    );
    assert_eq!(
        search_ids(&store, within_radius(SAPPORO, 50_000.0)),
        vec![2]
    );

    assert_eq!(
        stored_field(&store, 1, "location"),
        DataValue::GeoArray(vec![point(OSAKA), point(TOKYO)])
    );
}

/// Regression for the rectangle-corner false negative. The distance query
/// pre-filters by the rectangle enclosing the circle; a point in the
/// rectangle's corner passes the pre-filter but fails the radius test. With
/// first-occurrence dedup at candidate level, a document whose corner point
/// was traversed first lost its in-circle point and vanished from the
/// result. The corner is the south-west one (smallest lat *and* lon) and is
/// listed first, so it precedes the in-circle point under insertion order
/// and under either single-axis ordering of the BKD leaf.
#[test]
fn geo_array_document_is_found_when_a_corner_point_precedes_the_in_circle_point() {
    let store = geo_store(true, true, true);

    let radius_m = 50_000.0;
    // ~0.40° lat / ~0.50° lon ≈ 44 km / 45 km from the center on each axis:
    // inside the enclosing rectangle, but ≈ 63 km away — outside the circle.
    let sw_corner = (TOKYO.0 - 0.40, TOKYO.1 - 0.50);
    let ne_corner = (TOKYO.0 + 0.40, TOKYO.1 + 0.50);
    assert!(point(TOKYO).distance_to(&point(sw_corner)) > radius_m);
    assert!(point(TOKYO).distance_to(&point(ne_corner)) > radius_m);

    store
        .upsert_document(1, geo_array_doc(&[sw_corner, TOKYO]))
        .unwrap();
    store
        .upsert_document(2, geo_array_doc(&[TOKYO, ne_corner]))
        .unwrap();
    store
        .upsert_document(3, geo_array_doc(&[sw_corner, ne_corner]))
        .unwrap(); // no in-circle point
    store
        .upsert_document(4, geo_array_doc(&[sw_corner]))
        .unwrap();
    store.commit().unwrap();

    assert_eq!(
        search_ids(&store, within_radius(TOKYO, radius_m)),
        vec![1, 2],
        "a document must match when any of its points is inside the circle, \
         regardless of which point the BKD traversal yields first"
    );
}

/// A document with several matching points is reported once, with the
/// score of its closest point — identical to a single-valued document at
/// that closest point, and better than one holding only the farther point.
#[test]
fn geo_array_document_is_reported_once_with_its_closest_point() {
    let store = geo_store(true, true, true);

    store.upsert_document(1, geo_doc(TOKYO)).unwrap();
    store
        .upsert_document(2, geo_array_doc(&[YOKOHAMA, TOKYO]))
        .unwrap();
    store
        .upsert_document(3, geo_array_doc(&[YOKOHAMA]))
        .unwrap();
    store.commit().unwrap();

    let hits = search_hits(&store, within_radius(TOKYO, 50_000.0));
    let mut ids: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3], "doc 2 must appear exactly once");

    let score = |id: u64| hits.iter().find(|(d, _)| *d == id).unwrap().1;
    assert!(
        (score(1) - score(2)).abs() < 1e-6,
        "doc 2's closest point is Tokyo itself, so it scores like doc 1: {} vs {}",
        score(1),
        score(2)
    );
    assert!(
        score(2) > score(3),
        "doc 2 must outscore doc 3, whose only point is the farther Yokohama: {} vs {}",
        score(2),
        score(3)
    );
}

/// Bounding-box queries use the same any-match semantics and dedup.
#[test]
fn geo_array_bounding_box_query_matches_any_point() {
    let store = geo_store(true, true, true);

    store
        .upsert_document(1, geo_array_doc(&[OSAKA, TOKYO]))
        .unwrap();
    store.upsert_document(2, geo_array_doc(&[SAPPORO])).unwrap();
    store
        .upsert_document(3, geo_array_doc(&[TOKYO, YOKOHAMA]))
        .unwrap(); // both inside
    store.commit().unwrap();

    // Greater Tokyo box.
    let tokyo_box = || within_box((35.0, 139.0), (36.0, 140.5));
    assert_eq!(search_ids(&store, tokyo_box()), vec![1, 3]);
    let hits = search_hits(&store, tokyo_box());
    assert_eq!(
        hits.len(),
        2,
        "doc 3 has two points in the box but is reported once"
    );

    // Osaka box.
    assert_eq!(
        search_ids(&store, within_box((34.0, 135.0), (35.0, 136.0))),
        vec![1]
    );
    // Whole Japan.
    assert_eq!(
        search_ids(&store, within_box((30.0, 128.0), (46.0, 146.0))),
        vec![1, 2, 3]
    );
}

/// `indexed = false, stored = true`: no BKD tree exists, so every hit flows
/// through the stored-document fallback, which must see every point of a
/// `GeoArray`, not only single `Geo` values.
#[test]
fn stored_only_geo_array_field_matches_via_fallback() {
    let store = geo_store(false, true, true);

    store
        .upsert_document(1, geo_array_doc(&[OSAKA, TOKYO]))
        .unwrap();
    store.upsert_document(2, geo_array_doc(&[SAPPORO])).unwrap();
    store.upsert_document(3, geo_doc(YOKOHAMA)).unwrap(); // single point on a multi-valued field
    store.commit().unwrap();

    assert_eq!(
        search_ids(&store, within_radius(TOKYO, 50_000.0)),
        vec![1, 3]
    );
    assert_eq!(
        search_ids(&store, within_box((34.0, 135.0), (35.0, 136.0))),
        vec![1]
    );
    assert_eq!(
        search_ids(&store, within_box((30.0, 128.0), (46.0, 146.0))),
        vec![1, 2, 3]
    );
}

/// `indexed = true, stored = false`: the field lives solely in the BKD
/// tree, so every point of a `GeoArray` must reach the tree — there is no
/// stored-document fallback to mask a point that was never indexed (the
/// fallback rescues the `stored = true` cases above when no BKD exists).
#[test]
fn index_only_geo_array_field_matches_via_bkd() {
    let store = geo_store(true, false, true);

    store
        .upsert_document(1, geo_array_doc(&[OSAKA, TOKYO]))
        .unwrap();
    store.upsert_document(2, geo_array_doc(&[SAPPORO])).unwrap();
    store
        .upsert_document(3, geo_array_doc(&[YOKOHAMA, OSAKA]))
        .unwrap();
    store.commit().unwrap();

    assert_eq!(
        search_ids(&store, within_radius(TOKYO, 50_000.0)),
        vec![1, 3],
        "BKD-only coordinates must produce hits on a stored=false field"
    );
    assert_eq!(
        search_ids(&store, within_radius(OSAKA, 50_000.0)),
        vec![1, 3]
    );
    assert_eq!(
        search_ids(&store, within_box((30.0, 128.0), (46.0, 146.0))),
        vec![1, 2, 3]
    );
}

/// Multi-valued documents spread over several segments are each found once,
/// and the multi-segment dedup keeps the closest point.
#[test]
fn geo_array_queries_span_segments() {
    let store = geo_store(true, true, true);

    store
        .upsert_document(1, geo_array_doc(&[OSAKA, TOKYO]))
        .unwrap();
    store.commit().unwrap();
    store
        .upsert_document(2, geo_array_doc(&[SAPPORO, YOKOHAMA]))
        .unwrap();
    store.upsert_document(100, geo_array_doc(&[OSAKA])).unwrap(); // id above Σ doc_count
    store.commit().unwrap();
    store
        .upsert_document(3, geo_array_doc(&[TOKYO, YOKOHAMA]))
        .unwrap();
    store.commit().unwrap();

    assert_eq!(
        search_ids(&store, within_radius(TOKYO, 50_000.0)),
        vec![1, 2, 3]
    );
    assert_eq!(
        search_ids(&store, within_radius(OSAKA, 50_000.0)),
        vec![1, 100]
    );
    let hits = search_hits(&store, within_box((30.0, 128.0), (46.0, 146.0)));
    assert_eq!(hits.len(), 4, "every document exactly once across segments");
}

/// An empty point list is a valid value: the document is stored, reads back
/// as an empty array, and matches no spatial query.
#[test]
fn empty_geo_array_is_stored_and_never_matches() {
    let store = geo_store(true, true, true);

    store
        .upsert_document(
            1,
            Document::builder()
                .add_geo_array("location", Vec::new())
                .add_text("tag", "empty")
                .build(),
        )
        .unwrap();
    store.upsert_document(2, geo_array_doc(&[TOKYO])).unwrap();
    store.commit().unwrap();

    assert_eq!(
        search_ids(&store, within_box((-90.0, -180.0), (90.0, 180.0))),
        vec![2]
    );

    // The empty array is still stored: reach it through a term query on a
    // companion field, since no spatial query can match it.
    let results = store
        .search(
            LexicalSearchRequest::from_dsl("tag:empty")
                .limit(10)
                .load_documents(true),
        )
        .unwrap();
    assert_eq!(results.hits.len(), 1);
    assert_eq!(results.hits[0].doc_id, 1);
    let doc = results.hits[0].document.as_ref().expect("document loaded");
    assert_eq!(
        doc.get_field("location"),
        Some(&DataValue::GeoArray(Vec::new()))
    );
}

/// A single-valued field's results and per-document uniqueness are
/// unchanged by the candidate-level dedup removal.
#[test]
fn single_valued_geo_field_is_unchanged() {
    let store = geo_store(true, true, false);

    store.upsert_document(1, geo_doc(TOKYO)).unwrap();
    store.upsert_document(2, geo_doc(YOKOHAMA)).unwrap();
    store.commit().unwrap();
    store.upsert_document(3, geo_doc(OSAKA)).unwrap();
    store.upsert_document(4, geo_doc(SAPPORO)).unwrap();
    store.commit().unwrap();

    let hits = search_hits(&store, within_radius(TOKYO, 50_000.0));
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].0, 1, "closest first");
    assert_eq!(hits[1].0, 2);
    assert_eq!(
        search_ids(&store, within_box((30.0, 128.0), (46.0, 146.0))),
        vec![1, 2, 3, 4]
    );
}

// ---------------------------------------------------------------------------
// 3-D GeoEcefArray
// ---------------------------------------------------------------------------

/// Distance, bounding-box, and nearest queries on a `GeoEcefArray` field:
/// any-match, one hit per document, ranked by the closest point.
#[test]
fn geo_ecef_array_queries_match_any_point_and_rank_by_closest() {
    let store = geo3d_store();

    let center = GeoEcefPoint::new(1_000_000.0, 2_000_000.0, 3_000_000.0);
    let near = GeoEcefPoint::new(1_000_100.0, 2_000_000.0, 3_000_000.0); // 100 m
    let mid = GeoEcefPoint::new(1_000_500.0, 2_000_000.0, 3_000_000.0); // 500 m
    let far = GeoEcefPoint::new(2_000_000.0, 4_000_000.0, 6_000_000.0); // ~4 000 km

    let doc = |pts: Vec<GeoEcefPoint>| {
        Document::builder()
            .add_geo_ecef_array("position", pts)
            .build()
    };
    store.upsert_document(1, doc(vec![far, near])).unwrap(); // near point matches
    store.upsert_document(2, doc(vec![far])).unwrap(); // nothing close
    store.upsert_document(3, doc(vec![mid, mid])).unwrap(); // duplicate points
    store
        .upsert_document(
            4,
            Document::builder()
                .add_geo_ecef("position", near.x, near.y, near.z)
                .build(),
        )
        .unwrap(); // single point on a multi-valued field
    store.commit().unwrap();

    let search = |q: Box<dyn Query>| search_hits(&store, q);

    // Distance.
    let hits = search(Box::new(Geo3dDistanceQuery::new(
        "position", center, 1_000.0,
    )));
    let mut ids: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 3, 4]);
    assert_eq!(
        hits.len(),
        3,
        "doc 3's two identical points collapse to one hit"
    );

    // Nearest-k: doc 1's closest point (100 m) ties doc 4; doc 3 (500 m) is
    // next; the far-only doc 2 comes last. Every document appears once.
    let hits = search(Box::new(Geo3dNearestQuery::new("position", center, 10)));
    assert_eq!(hits.len(), 4, "one hit per document");
    let ids: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
    assert!(ids[..2].contains(&1) && ids[..2].contains(&4), "{ids:?}");
    assert_eq!(ids[2], 3, "{ids:?}");
    assert_eq!(ids[3], 2, "{ids:?}");

    // Bounding box around the center (±1 km on every axis).
    let bbox = Geo3dBoundingBoxQuery::new(
        "position",
        GeoEcefPoint::new(center.x - 1_000.0, center.y - 1_000.0, center.z - 1_000.0),
        GeoEcefPoint::new(center.x + 1_000.0, center.y + 1_000.0, center.z + 1_000.0),
    )
    .unwrap();
    let mut ids: Vec<u64> = search(Box::new(bbox)).iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 3, 4]);

    // Stored value round-trips as an array.
    let results = store
        .search(
            LexicalSearchRequest::new(Box::new(Geo3dNearestQuery::new("position", center, 10)))
                .limit(10)
                .load_documents(true),
        )
        .unwrap();
    let hit = results
        .hits
        .iter()
        .find(|h| h.doc_id == 1)
        .expect("doc 1 hit");
    assert_eq!(
        hit.document
            .as_ref()
            .and_then(|d| d.get_field("position"))
            .and_then(DataValue::as_geo_ecef_array),
        Some(&[far, near][..])
    );
}
