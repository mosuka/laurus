use chrono::{TimeZone, Utc};
use laurus::lexical::NumericRangeQuery;
use laurus::lexical::NumericType;
use laurus::lexical::Query;
use laurus::lexical::{GeoDistanceQuery, GeoPoint};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};
use std::sync::Arc;

/// A writer registered with a real index (#1024): a standalone
/// `InvertedIndexWriter` is ephemeral — its segments enter no manifest and
/// `build_reader` sees nothing — so durable fixtures go through
/// `InvertedIndex::create` + `writer()`.
fn index_writer(
    storage: Arc<dyn laurus::storage::Storage>,
) -> Box<dyn laurus::lexical::writer::LexicalIndexWriter> {
    let index = laurus::lexical::index::inverted::InvertedIndex::create(
        storage,
        // Loose layout, explicitly: this suite pins per-field `.bkd` FILE
        // creation on disk, which only exists as loose files. Compound-layout
        // BKD behavior is covered by `compound_segment_test.rs`.
        laurus::lexical::InvertedIndexConfig {
            use_compound: false,
            ..Default::default()
        },
    )
    .unwrap();
    use laurus::lexical::index::LexicalIndex;
    index.writer().unwrap()
}

#[test]
fn test_bkd_file_creation_and_query() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut writer = index_writer(storage.clone());

    // Doc 1: age=30, score=95.5
    let doc1 = Document::builder()
        .add_field("age", DataValue::Int64(30))
        .add_field("score", DataValue::Float64(95.5))
        .add_field(
            "created_at",
            DataValue::DateTime(Utc.timestamp_opt(1600000000, 0).unwrap()),
        )
        .add_field("description", DataValue::Text("User profile 1".into()))
        .build();
    writer.add_document(doc1).unwrap();

    // Doc 2: age=20, score=80.0
    let doc2 = Document::builder()
        .add_field("age", DataValue::Int64(20))
        .add_field("score", DataValue::Float64(80.0))
        .add_field(
            "created_at",
            DataValue::DateTime(Utc.timestamp_opt(1500000000, 0).unwrap()),
        )
        .add_field("description", DataValue::Text("User profile 2".into()))
        .build();
    writer.add_document(doc2).unwrap();

    // Doc 3: age=40, score=100.0
    let doc3 = Document::builder()
        .add_field("age", DataValue::Int64(40))
        .add_field("score", DataValue::Float64(100.0))
        .add_field(
            "created_at",
            DataValue::DateTime(Utc.timestamp_opt(1700000000, 0).unwrap()),
        )
        .add_field("description", DataValue::Text("User profile 3".into()))
        .build();
    writer.add_document(doc3).unwrap();

    // Commit to flush segment
    writer.commit().unwrap();

    // Verify files existed
    let age_bkd = "segment_000000.age.bkd";
    assert!(
        storage.file_exists(age_bkd),
        "BKD file for age should exist"
    );

    // Open Reader
    let reader = writer.build_reader().unwrap();

    // Query 1: Age [25, 35] -> Should match Doc 1 (age 30) -> ID 0
    let query_age = NumericRangeQuery::new(
        "age",
        NumericType::Integer,
        Some(25.0),
        Some(35.0),
        true,
        true,
    );

    let matched_age = collect_matcher_results(query_age.matcher(&*reader).unwrap());

    assert_eq!(matched_age, vec![0]);

    // Query 2: Score >= 90.0 -> Doc 1 (95.5), Doc 3 (100.0) -> IDs 0, 2
    let query_score =
        NumericRangeQuery::new("score", NumericType::Float, Some(90.0), None, true, true);
    let matched_score = collect_matcher_results(query_score.matcher(&*reader).unwrap());

    assert_eq!(matched_score, vec![0, 2]);

    // Query 3: Created At < 1600000000 -> Doc 2 (1500000000) -> ID 1
    let query_date = NumericRangeQuery::new(
        "created_at",
        NumericType::Integer,
        None,
        Some(1600000000.0),
        false,
        false,
    );
    let matched_date = collect_matcher_results(query_date.matcher(&*reader).unwrap());

    assert_eq!(matched_date, vec![1]);
}

#[test]
fn test_geo_bkd_query() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut writer = index_writer(storage.clone());

    // Tokyo: 35.6812, 139.7671
    let tokyo = GeoPoint::new(35.6812, 139.7671);
    // Yokohama: 35.4437, 139.6380
    let yokohama = GeoPoint::new(35.4437, 139.6380);
    // Osaka: 34.6937, 135.5023
    let osaka = GeoPoint::new(34.6937, 135.5023);

    writer
        .add_document(
            Document::builder()
                .add_field("location", DataValue::Geo(tokyo))
                .add_field("city", DataValue::Text("Tokyo".into()))
                .build(),
        )
        .unwrap();

    writer
        .add_document(
            Document::builder()
                .add_field("location", DataValue::Geo(yokohama))
                .add_field("city", DataValue::Text("Yokohama".into()))
                .build(),
        )
        .unwrap();

    writer
        .add_document(
            Document::builder()
                .add_field("location", DataValue::Geo(osaka))
                .add_field("city", DataValue::Text("Osaka".into()))
                .build(),
        )
        .unwrap();

    writer.commit().unwrap();

    // Verify BKD file existed
    assert!(storage.file_exists("segment_000000.location.bkd"));

    let reader = writer.build_reader().unwrap();

    // Distance query: Near Tokyo (within 50 km) -> Should match Tokyo (0 km) and Yokohama (~30 km)
    let query = GeoDistanceQuery::new("location", tokyo, 50_000.0);
    let matched_docs = collect_matcher_results(query.matcher(&*reader).unwrap());
    assert_eq!(matched_docs, vec![0, 1]);

    // Near Osaka (within 20 km) -> Should match Osaka
    let query_osaka = GeoDistanceQuery::new("location", osaka, 20_000.0);
    let matched_osaka = collect_matcher_results(query_osaka.matcher(&*reader).unwrap());
    assert_eq!(matched_osaka, vec![2]);
}

fn collect_matcher_results(mut m: Box<dyn laurus::lexical::query::matcher::Matcher>) -> Vec<u64> {
    let mut docs = Vec::new();
    while !m.is_exhausted() {
        let doc_id = m.doc_id();
        if doc_id == u64::MAX {
            break;
        }
        docs.push(doc_id);
        if !m.next().unwrap() {
            break;
        }
    }
    docs
}

/// Regression test for Issue #557's `write_bkd_trees` rewrite: a field
/// carrying an empty array (`Int64Array(vec![])`) contributes zero BKD
/// points and must not produce a `.bkd` part at all. Before the rewrite
/// this was enforced by an `if doc_ids.is_empty() { continue; }` guard that
/// was, in practice, dead code (the single-pass loop only ever created a
/// bucket from inside the per-point loop, so an empty array never created
/// one to begin with) — the two-pass discovery/write split makes that
/// guard load-bearing for the first time, so it needs its own coverage.
#[test]
fn empty_array_point_field_produces_no_bkd_part() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut writer = index_writer(storage.clone());

    writer
        .add_document(
            Document::builder()
                .add_int64_array("scores", vec![])
                .add_int64_array("never_populated", vec![])
                .add_field("name", DataValue::Text("empty scores".into()))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_int64_array("scores", vec![10, 20])
                .add_int64_array("never_populated", vec![])
                .add_field("name", DataValue::Text("has scores".into()))
                .build(),
        )
        .unwrap();
    writer.commit().unwrap();

    // The field itself does produce a `.bkd` part, since doc 1 has real
    // points — this asserts the empty-array doc doesn't suppress that.
    assert!(
        storage.file_exists("segment_000000.scores.bkd"),
        "a field with at least one real point must still get a .bkd part"
    );
    // `never_populated` is empty in EVERY document, so it must never
    // produce a `.bkd` part at all (this is the case a mutation that
    // registers a field from an empty array wouldn't catch if some other
    // document happened to have real points for the same field name).
    assert!(
        !storage.file_exists("segment_000000.never_populated.bkd"),
        "a field with no real points in any document must not get a .bkd part"
    );

    let reader = writer.build_reader().unwrap();
    let query = NumericRangeQuery::new(
        "scores",
        NumericType::Integer,
        Some(0.0),
        Some(100.0),
        true,
        true,
    );
    let matched = collect_matcher_results(query.matcher(&*reader).unwrap());
    assert_eq!(
        matched,
        vec![1],
        "only the document with real points should match a range query"
    );
}

/// Regression test for Issue #557's `write_bkd_trees` rewrite: with two
/// distinct point-bearing fields on the same documents (an `Integer` field
/// and a `Geo` field), each field's per-field buffer must hold exactly and
/// only that field's points — no cross-contamination between fields
/// processed in the same discovery/write pass.
#[test]
fn multiple_point_bearing_fields_do_not_cross_contaminate() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut writer = index_writer(storage.clone());

    let tokyo = GeoPoint::new(35.6812, 139.7671);
    let osaka = GeoPoint::new(34.6937, 135.5023);

    writer
        .add_document(
            Document::builder()
                .add_field("price", DataValue::Int64(100))
                .add_field("location", DataValue::Geo(tokyo))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("price", DataValue::Int64(200))
                .add_field("location", DataValue::Geo(osaka))
                .build(),
        )
        .unwrap();
    writer.commit().unwrap();

    assert!(storage.file_exists("segment_000000.price.bkd"));
    assert!(storage.file_exists("segment_000000.location.bkd"));

    let reader = writer.build_reader().unwrap();

    let price_query =
        NumericRangeQuery::new("price", NumericType::Integer, Some(150.0), None, true, true);
    let matched_price = collect_matcher_results(price_query.matcher(&*reader).unwrap());
    assert_eq!(matched_price, vec![1], "price field must hold only prices");

    let geo_query = GeoDistanceQuery::new("location", tokyo, 50_000.0);
    let matched_geo = collect_matcher_results(geo_query.matcher(&*reader).unwrap());
    assert_eq!(
        matched_geo,
        vec![0],
        "location field must hold only geo points, unaffected by price"
    );
}
