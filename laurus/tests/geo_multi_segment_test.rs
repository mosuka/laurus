//! Integration tests for issue #996 — 2D geo queries on multi-segment
//! indices.
//!
//! Guards two regressions:
//!
//! - the BKD-less fallback must stay segment-bounded and correct for
//!   sparse doc-id spaces (a stored-only geo field has no BKD tree in
//!   any segment, so every hit flows through the fallback — this gate
//!   fails if the fallback misses ids above `max_doc()` or stops
//!   matching);
//! - mixed BKD / fallback fan-out (only some segments carry the geo
//!   field) must return the correct hits.

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::core::field::{FieldOption, GeoOption, TextOption};
use laurus::lexical::query::geo::{GeoBoundingBoxQuery, GeoDistanceQuery};
use laurus::lexical::query::{GeoPoint, Query};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

fn geo_doc(lat: f64, lon: f64) -> Document {
    Document::builder()
        .add_field(
            "location",
            laurus::DataValue::Geo(GeoPoint::try_new(lat, lon).unwrap()),
        )
        .build()
}

fn body_doc(text: &str) -> Document {
    Document::builder().add_text("body", text).build()
}

fn search_hits(store: &LexicalStore, query: Box<dyn Query>) -> Vec<u64> {
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

/// #996: a stored-only (`indexed = false`) geo field has no BKD tree in
/// any segment, so every hit must come through the stored-document
/// fallback. Doc ids are placed above `max_doc()` (= Σ doc_count) to
/// catch the dense `0..max_doc()` under-scan directly.
#[test]
fn stored_only_geo_field_matches_via_fallback_across_segments() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "location",
            FieldOption::Geo(GeoOption {
                indexed: false,
                stored: true,
                doc_values: true,
            }),
        )
        .build();
    let store = LexicalStore::new(storage, config).unwrap();

    // Segment 1: Tokyo and Yokohama, low ids.
    store.upsert_document(1, geo_doc(35.68, 139.76)).unwrap();
    store.upsert_document(2, geo_doc(35.44, 139.64)).unwrap();
    store.commit().unwrap();
    // Segment 2: Osaka and Sapporo, ids far above Σ doc_count (= 4).
    store.upsert_document(100, geo_doc(34.69, 135.50)).unwrap();
    store.upsert_document(101, geo_doc(43.06, 141.35)).unwrap();
    store.commit().unwrap();

    // Whole-Japan box: every doc, including the high ids the dense
    // 0..max_doc() scan used to miss.
    let all_japan = Box::new(
        GeoBoundingBoxQuery::within_bounding_box("location", 30.0, 128.0, 46.0, 146.0).unwrap(),
    );
    assert_eq!(search_hits(&store, all_japan), vec![1, 2, 100, 101]);

    // Osaka-only box: exactly the high-id doc.
    let osaka_box = Box::new(
        GeoBoundingBoxQuery::within_bounding_box("location", 34.0, 135.0, 35.0, 136.0).unwrap(),
    );
    assert_eq!(search_hits(&store, osaka_box), vec![100]);

    // Distance query around Tokyo station (50 km): Tokyo + Yokohama.
    let near_tokyo =
        Box::new(GeoDistanceQuery::within_radius("location", 35.68, 139.76, 50_000.0).unwrap());
    assert_eq!(search_hits(&store, near_tokyo), vec![1, 2]);
}

/// #1000: an index-only geo field (`indexed = true, stored = false`)
/// lives solely in the BKD tree. The BKD path must recover coordinates
/// from the tree itself — the old stored-document probe silently
/// dropped every candidate, returning zero hits guaranteed (the BKD
/// branch returns early, so the fallback never ran either).
#[test]
fn index_only_geo_field_matches_via_bkd() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "location",
            FieldOption::Geo(GeoOption {
                indexed: true,
                stored: false,
                doc_values: true,
            }),
        )
        .build();
    let store = LexicalStore::new(storage, config).unwrap();

    // Two segments, both carrying the geo field.
    store.upsert_document(1, geo_doc(35.68, 139.76)).unwrap(); // Tokyo
    store.upsert_document(2, geo_doc(35.44, 139.64)).unwrap(); // Yokohama
    store.commit().unwrap();
    store.upsert_document(3, geo_doc(34.69, 135.50)).unwrap(); // Osaka
    store.commit().unwrap();

    let all_japan = Box::new(
        GeoBoundingBoxQuery::within_bounding_box("location", 30.0, 128.0, 46.0, 146.0).unwrap(),
    );
    assert_eq!(
        search_hits(&store, all_japan),
        vec![1, 2, 3],
        "BKD-only coordinates must produce hits on a stored=false field"
    );

    let near_tokyo =
        Box::new(GeoDistanceQuery::within_radius("location", 35.68, 139.76, 50_000.0).unwrap());
    assert_eq!(search_hits(&store, near_tokyo), vec![1, 2]);

    let osaka_box = Box::new(
        GeoBoundingBoxQuery::within_bounding_box("location", 34.0, 135.0, 35.0, 136.0).unwrap(),
    );
    assert_eq!(search_hits(&store, osaka_box), vec![3]);
}

/// #996: mixed BKD / fallback fan-out — only one of three segments
/// carries the (indexed) geo field; the field-less segments take the
/// fallback path and must not disturb the result.
#[test]
fn geo_query_on_sparse_field_across_segments() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "location",
            FieldOption::Geo(GeoOption {
                indexed: true,
                stored: true,
                doc_values: true,
            }),
        )
        .add_field("body", FieldOption::Text(TextOption::default()))
        .build();
    let store = LexicalStore::new(storage, config).unwrap();

    // Segment 1: no geo field (no .bkd → fallback path under fanout).
    store.upsert_document(1, body_doc("alpha")).unwrap();
    store.upsert_document(2, body_doc("bravo")).unwrap();
    store.commit().unwrap();
    // Segment 2: geo docs (BKD path).
    store.upsert_document(3, geo_doc(35.68, 139.76)).unwrap();
    store.upsert_document(4, geo_doc(34.69, 135.50)).unwrap();
    store.commit().unwrap();
    // Segment 3: no geo field again.
    store.upsert_document(5, body_doc("charlie")).unwrap();
    store.commit().unwrap();

    let all_japan = Box::new(
        GeoBoundingBoxQuery::within_bounding_box("location", 30.0, 128.0, 46.0, 146.0).unwrap(),
    );
    assert_eq!(search_hits(&store, all_japan), vec![3, 4]);

    let near_tokyo =
        Box::new(GeoDistanceQuery::within_radius("location", 35.68, 139.76, 50_000.0).unwrap());
    assert_eq!(search_hits(&store, near_tokyo), vec![3]);
}
