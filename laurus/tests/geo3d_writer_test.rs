//! End-to-end integration test for 3D ECEF geo field handling in the
//! inverted index writer/reader pair.
//!
//! Covers the full path that #299 wires up:
//! - `DataValue::GeoEcef` ingested via `add_geo_ecef` flows through the
//!   document parser into `point_values` of length 3.
//! - `BKDWriter` materializes a 3D BKD because the points are 3-element.
//! - The stored-field deserializer reads tag 12 back as
//!   `FieldValue::GeoEcef`.
//! - A `range_search` over the BKD returns the expected doc ids and the
//!   round-tripped `GeoEcefPoint` values match what was indexed.
//!
//! Sample points are deliberately spread across an order-of-magnitude
//! range on each axis so a narrow query box exercises the BKD's pruning
//! paths and confirms that all three dimensions are honoured.

use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::query::Query;
use laurus::lexical::query::{Geo3dBoundingBoxQuery, Geo3dDistanceQuery, Geo3dNearestQuery};
use laurus::lexical::writer::LexicalIndexWriter;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document, GeoEcefPoint};
use std::sync::Arc;

/// An index and a writer registered with it (#1024): a standalone
/// `InvertedIndexWriter` is ephemeral — its segments enter no manifest, so
/// no reader sees them — so durable fixtures go through
/// `InvertedIndex::create` + `writer()` and read back through
/// `index.reader()`.
fn index_and_writer(
    storage: Arc<dyn laurus::storage::Storage>,
) -> (InvertedIndex, Box<dyn LexicalIndexWriter>) {
    let index = InvertedIndex::create(
        storage,
        // Loose layout, explicitly: this suite pins the on-disk BKD
        // FILE format (header bytes, per-field `.bkd` names), which
        // only exists as loose files. Compound-layout BKD behavior is
        // covered by `compound_segment_test.rs`.
        laurus::lexical::InvertedIndexConfig {
            use_compound: false,
            ..Default::default()
        },
    )
    .unwrap();
    let writer = index.writer().unwrap();
    (index, writer)
}

#[test]
fn geo3d_round_trip_through_writer_and_reader() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (index, mut writer) = index_and_writer(storage.clone());

    // Three ECEF points, picked so that the per-axis ranges differ by an
    // order of magnitude (this is what makes widest-axis splitting
    // observable, and what would expose any 2D-vs-3D dim mismatch).
    let p_a = GeoEcefPoint::new(1_000_000.0, 2_000_000.0, 3_000_000.0);
    let p_b = GeoEcefPoint::new(1_000_500.0, 2_005_000.0, 3_050_000.0);
    let p_c = GeoEcefPoint::new(2_000_000.0, 4_000_000.0, 6_000_000.0);

    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_a))
                .add_field("name", DataValue::Text("A".into()))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_b))
                .add_field("name", DataValue::Text("B".into()))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_c))
                .add_field("name", DataValue::Text("C".into()))
                .build(),
        )
        .unwrap();

    writer.commit().unwrap();

    // The per-field BKD is named after the segment + field name. Confirm
    // it landed on disk before going further; if we miss this the next
    // assertion will fail with a less obvious message.
    assert!(
        storage.file_exists("segment_000000.position.bkd"),
        "expected the writer to materialize a 3D BKD for the position field"
    );

    let reader = index.reader().unwrap();

    // Stored-field round-trip: docs come back with their GeoEcef intact.
    // This exercises the new tag-12 path on the reader side.
    let doc_a = reader.document(0).unwrap().expect("doc 0 must exist");
    let doc_b = reader.document(1).unwrap().expect("doc 1 must exist");
    let doc_c = reader.document(2).unwrap().expect("doc 2 must exist");
    assert_eq!(
        doc_a.get("position").and_then(|v| v.as_geo_ecef()),
        Some(p_a)
    );
    assert_eq!(
        doc_b.get("position").and_then(|v| v.as_geo_ecef()),
        Some(p_b)
    );
    assert_eq!(
        doc_c.get("position").and_then(|v| v.as_geo_ecef()),
        Some(p_c)
    );

    // BKD round-trip: a narrow 3D range that should match only A and B
    // (their x/y/z all sit in the lower half of every axis).
    let bkd_tree = reader
        .get_bkd_tree("position")
        .unwrap()
        .expect("position field must have a BKD tree");
    let mut hits = bkd_tree
        .range_search(
            &[Some(0.0), Some(0.0), Some(0.0)],
            &[Some(1_500_000.0), Some(2_500_000.0), Some(3_500_000.0)],
            true,
            true,
        )
        .unwrap();
    hits.sort_unstable();
    assert_eq!(hits, vec![0, 1]);

    // Wider query: include C (which lives in the upper octant).
    let mut all_hits = bkd_tree
        .range_search(
            &[Some(0.0), Some(0.0), Some(0.0)],
            &[Some(3_000_000.0), Some(5_000_000.0), Some(7_000_000.0)],
            true,
            true,
        )
        .unwrap();
    all_hits.sort_unstable();
    assert_eq!(all_hits, vec![0, 1, 2]);
}

#[test]
fn geo3d_dimension_observable_through_bkd_header() {
    use laurus::lexical::index::structures::bkd_tree::BKDReader;

    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (_index, mut writer) = index_and_writer(storage.clone());

    writer
        .add_document(
            Document::builder()
                .add_field(
                    "position",
                    DataValue::GeoEcef(GeoEcefPoint::new(1.0, 2.0, 3.0)),
                )
                .build(),
        )
        .unwrap();
    writer.commit().unwrap();

    // Pop the BKD reader off the segment directly so we can inspect the
    // dimensionality the writer chose. A 3D BKD is exactly what we want;
    // a 2D one would mean the GeoEcef branch fell back to the 2D Geo flow.
    let bkd = BKDReader::open(
        storage.clone() as Arc<dyn Storage>,
        "segment_000000.position.bkd",
    )
    .unwrap();
    let header = bkd.header();
    assert_eq!(header.num_dims, 3, "ECEF must produce a 3D BKD");
    assert_eq!(header.bytes_per_dim, 8);
    assert_eq!(header.total_point_count, 1);
}

#[test]
fn geo3d_distance_query_finds_docs_within_radius() {
    // Index three ECEF points and run a sphere query that should match
    // exactly two of them. The third sits well outside the radius.
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (index, mut writer) = index_and_writer(storage.clone());

    let center = GeoEcefPoint::new(1_000_000.0, 2_000_000.0, 3_000_000.0);
    // Inside (offset 100m on x)
    let p_close = GeoEcefPoint::new(1_000_100.0, 2_000_000.0, 3_000_000.0);
    // Inside (offset ~141m, sqrt(50000² + 0 + 0) wait — let me pick
    // (50, 50, 50) → sqrt(7500) ≈ 86.6m, well inside 1km)
    let p_mid = GeoEcefPoint::new(1_000_050.0, 2_000_050.0, 3_000_050.0);
    // Outside (offset 5km on x)
    let p_far = GeoEcefPoint::new(1_005_000.0, 2_000_000.0, 3_000_000.0);

    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_close))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_mid))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_far))
                .build(),
        )
        .unwrap();
    writer.commit().unwrap();

    let reader = index.reader().unwrap();

    // 1km sphere query catches the two close docs but not the 5km-away
    // one. Matches arrive distance-ascending — doc 1 (~87m) is closer
    // than doc 0 (100m), so doc 1 comes first.
    let query = Geo3dDistanceQuery::new("position", center, 1_000.0);
    let matches = query.find_matches(&*reader).unwrap();
    let doc_ids: Vec<u64> = matches.iter().map(|m| m.doc_id).collect();
    assert_eq!(doc_ids, vec![1, 0]);

    let d_close = ((p_close.x - center.x).powi(2)
        + (p_close.y - center.y).powi(2)
        + (p_close.z - center.z).powi(2))
    .sqrt();
    let d_mid = ((p_mid.x - center.x).powi(2)
        + (p_mid.y - center.y).powi(2)
        + (p_mid.z - center.z).powi(2))
    .sqrt();
    assert!(d_mid < d_close, "doc 1 must be closer than doc 0");
    assert!((matches[0].distance_m - d_mid).abs() < 1e-6);
    assert!((matches[1].distance_m - d_close).abs() < 1e-6);
    // Scores decrease with distance.
    assert!(matches[0].score > matches[1].score);

    // Wider radius (10km) catches all three, including the far doc.
    let wide = Geo3dDistanceQuery::new("position", center, 10_000.0);
    let mut all_doc_ids: Vec<u64> = wide
        .find_matches(&*reader)
        .unwrap()
        .iter()
        .map(|m| m.doc_id)
        .collect();
    all_doc_ids.sort_unstable();
    assert_eq!(all_doc_ids, vec![0, 1, 2]);

    // Empty radius matches nothing.
    let empty = Geo3dDistanceQuery::new("position", center, 0.0);
    assert!(empty.find_matches(&*reader).unwrap().is_empty());
    assert!(empty.is_empty(&*reader).unwrap());
}

#[test]
fn geo3d_bbox_query_finds_docs_inside_box() {
    // Index three ECEF points in distinct octants and verify various
    // bounding-box queries pick the right ones.
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (index, mut writer) = index_and_writer(storage.clone());

    let p_a = GeoEcefPoint::new(100.0, 100.0, 100.0); // doc 0 — interior
    let p_b = GeoEcefPoint::new(200.0, 200.0, 200.0); // doc 1 — interior
    let p_c = GeoEcefPoint::new(1_000.0, 1_000.0, 1_000.0); // doc 2 — outside

    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_a))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_b))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("position", DataValue::GeoEcef(p_c))
                .build(),
        )
        .unwrap();
    writer.commit().unwrap();

    let reader = index.reader().unwrap();

    // Basic case: a box that contains exactly doc 0 and doc 1.
    let q = Geo3dBoundingBoxQuery::new(
        "position",
        GeoEcefPoint::new(0.0, 0.0, 0.0),
        GeoEcefPoint::new(300.0, 300.0, 300.0),
    )
    .unwrap();
    let mut hits: Vec<u64> = q
        .find_matches(&*reader)
        .unwrap()
        .into_iter()
        .map(|m| m.doc_id)
        .collect();
    hits.sort_unstable();
    assert_eq!(hits, vec![0, 1]);

    // Edge-aligned: a box whose maximum corner sits exactly on doc 1.
    // Closed bounds → doc 1 is included.
    let q_edge = Geo3dBoundingBoxQuery::new(
        "position",
        GeoEcefPoint::new(0.0, 0.0, 0.0),
        GeoEcefPoint::new(200.0, 200.0, 200.0),
    )
    .unwrap();
    let mut edge_hits: Vec<u64> = q_edge
        .find_matches(&*reader)
        .unwrap()
        .into_iter()
        .map(|m| m.doc_id)
        .collect();
    edge_hits.sort_unstable();
    assert_eq!(edge_hits, vec![0, 1]);

    // Degenerate (zero-volume) box: matches the single point at doc 0.
    let q_point = Geo3dBoundingBoxQuery::new("position", p_a, p_a).unwrap();
    let pt_hits: Vec<u64> = q_point
        .find_matches(&*reader)
        .unwrap()
        .into_iter()
        .map(|m| m.doc_id)
        .collect();
    assert_eq!(pt_hits, vec![0]);

    // Wide-open box: catches every doc.
    let q_wide = Geo3dBoundingBoxQuery::new(
        "position",
        GeoEcefPoint::new(-1e9, -1e9, -1e9),
        GeoEcefPoint::new(1e9, 1e9, 1e9),
    )
    .unwrap();
    let mut wide_hits: Vec<u64> = q_wide
        .find_matches(&*reader)
        .unwrap()
        .into_iter()
        .map(|m| m.doc_id)
        .collect();
    wide_hits.sort_unstable();
    assert_eq!(wide_hits, vec![0, 1, 2]);

    // Per-axis exclusion: shrinking only the z axis enough to drop
    // doc 1 still keeps doc 0.
    let q_z = Geo3dBoundingBoxQuery::new(
        "position",
        GeoEcefPoint::new(0.0, 0.0, 0.0),
        GeoEcefPoint::new(300.0, 300.0, 150.0),
    )
    .unwrap();
    let z_hits: Vec<u64> = q_z
        .find_matches(&*reader)
        .unwrap()
        .into_iter()
        .map(|m| m.doc_id)
        .collect();
    assert_eq!(z_hits, vec![0]);

    // Scoring: every hit gets the constant score 1.0 from the bbox path.
    for m in q.find_matches(&*reader).unwrap() {
        assert_eq!(m.score, 1.0);
        assert_eq!(m.distance_m, 0.0);
    }
}

#[test]
fn geo3d_nearest_query_finds_top_k_in_distance_order() {
    // Index five points along the x-axis at increasing distances from
    // the origin, then exercise the k-NN query at various k values and
    // initial-radius settings.
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (index, mut writer) = index_and_writer(storage.clone());

    let center = GeoEcefPoint::new(0.0, 0.0, 0.0);
    // Doc i sits at (10^i, 0, 0): 1, 10, 100, 1000, 10000.
    let positions = [1.0_f64, 10.0, 100.0, 1_000.0, 10_000.0];
    for &x in &positions {
        writer
            .add_document(
                Document::builder()
                    .add_field(
                        "position",
                        DataValue::GeoEcef(GeoEcefPoint::new(x, 0.0, 0.0)),
                    )
                    .build(),
            )
            .unwrap();
    }
    writer.commit().unwrap();

    let reader = index.reader().unwrap();

    // k=1: closest doc is doc 0 (distance 1).
    let q1 = Geo3dNearestQuery::new("position", center, 1);
    let m1 = q1.find_matches(&*reader).unwrap();
    assert_eq!(m1.len(), 1);
    assert_eq!(m1[0].doc_id, 0);
    assert_eq!(m1[0].distance_m, 1.0);

    // k=3: docs 0, 1, 2 in distance order.
    let q3 = Geo3dNearestQuery::new("position", center, 3);
    let m3 = q3.find_matches(&*reader).unwrap();
    let ids3: Vec<u64> = m3.iter().map(|m| m.doc_id).collect();
    assert_eq!(ids3, vec![0, 1, 2]);
    // Distance-ascending: each subsequent hit is farther than the last.
    for w in m3.windows(2) {
        assert!(w[0].distance_m <= w[1].distance_m);
    }
    // Score: monotonically decreasing as distance grows, with the
    // farthest hit normalized to 0.0 by `1 - distance/max_distance`.
    assert!(m3[0].score > m3[1].score);
    assert!(m3[1].score > m3[2].score);
    assert_eq!(m3[2].score, 0.0);
    // m3[0] is the closest of the returned top-k; with positions
    // {1, 10, 100} it scores `1 - 1/100 = 0.99`, very close to 1.0.
    assert!((m3[0].score - 0.99).abs() < 1e-4);

    // k > #docs: returns all docs (5).
    let q_all = Geo3dNearestQuery::new("position", center, 100);
    let m_all = q_all.find_matches(&*reader).unwrap();
    assert_eq!(m_all.len(), 5);
    let ids_all: Vec<u64> = m_all.iter().map(|m| m.doc_id).collect();
    assert_eq!(ids_all, vec![0, 1, 2, 3, 4]);

    // k=0: empty result.
    let q0 = Geo3dNearestQuery::new("position", center, 0);
    assert!(q0.find_matches(&*reader).unwrap().is_empty());
    assert!(q0.is_empty(&*reader).unwrap());

    // Convergence with a tiny initial radius: even though the loop
    // starts at 1m (smaller than the closest doc), it must double up
    // until 5 hits are collected.
    let q_small_init = Geo3dNearestQuery::new("position", center, 5).with_initial_radius(0.5);
    let m_si = q_small_init.find_matches(&*reader).unwrap();
    assert_eq!(m_si.len(), 5);
    let ids_si: Vec<u64> = m_si.iter().map(|m| m.doc_id).collect();
    assert_eq!(ids_si, vec![0, 1, 2, 3, 4]);

    // Sparse: query center far from every point. With default initial
    // radius (1km), the first probe finds nothing — the loop must keep
    // doubling until the index is exhausted within max_radius.
    let far_center = GeoEcefPoint::new(1_000_000.0, 0.0, 0.0);
    let q_far = Geo3dNearestQuery::new("position", far_center, 2);
    let m_far = q_far.find_matches(&*reader).unwrap();
    assert_eq!(m_far.len(), 2);
    // Closest to (1e6, 0, 0) is doc 4 at (10000, 0, 0), then doc 3 at
    // (1000, 0, 0).
    assert_eq!(m_far[0].doc_id, 4);
    assert_eq!(m_far[1].doc_id, 3);
}
