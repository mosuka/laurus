//! End-to-end tests for multi-valued DateTime fields (Issue #1184).
//!
//! Drives `LexicalStore` the way production search does and verifies the
//! Lucene-style "any instant matches" semantics for `DateTimeArray`
//! fields, on the BKD path and the stored-document fallback, through the
//! typed `DateTimeRangeQuery`, the numeric API, and the query DSL.

use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use laurus::lexical::NumericType;
use laurus::lexical::core::field::{DateTimeOption, FieldOption, TextOption};
use laurus::lexical::query::{DateTimeRangeQuery, NumericRangeQuery, Query};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};

fn utc(y: i32, m: u32, d: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
}

fn dt_doc(dt: DateTime<Utc>) -> Document {
    Document::builder()
        .add_field("seen_at", DataValue::DateTime(dt))
        .build()
}

fn dt_array_doc(instants: &[DateTime<Utc>]) -> Document {
    Document::builder()
        .add_datetime_array("seen_at", instants.to_vec())
        .build()
}

fn store(indexed: bool, stored: bool, multi_valued: bool) -> LexicalStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "seen_at",
            FieldOption::DateTime(DateTimeOption {
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

fn hits(store: &LexicalStore, request: LexicalSearchRequest) -> Vec<u64> {
    let mut ids: Vec<u64> = store
        .search(request.limit(100))
        .unwrap()
        .hits
        .iter()
        .map(|hit| hit.doc_id)
        .collect();
    ids.sort_unstable();
    ids
}

fn query_hits(store: &LexicalStore, query: impl Query + 'static) -> Vec<u64> {
    hits(store, LexicalSearchRequest::new(Box::new(query)))
}

fn dsl_hits(store: &LexicalStore, dsl: &str) -> Vec<u64> {
    hits(store, LexicalSearchRequest::from_dsl(dsl))
}

fn between(start: DateTime<Utc>, end: DateTime<Utc>) -> DateTimeRangeQuery {
    DateTimeRangeQuery::between("seen_at", start, end)
}

/// Doc 1: Jan + Jun; doc 2: Dec + next Jan; doc 3: Mar only; doc 4: next
/// Mar only.
fn seed(store: &LexicalStore) {
    store
        .upsert_document(1, dt_array_doc(&[utc(2024, 1, 15), utc(2024, 6, 15)]))
        .unwrap();
    store
        .upsert_document(2, dt_array_doc(&[utc(2024, 12, 15), utc(2025, 1, 15)]))
        .unwrap();
    store
        .upsert_document(3, dt_array_doc(&[utc(2024, 3, 15)]))
        .unwrap();
    store
        .upsert_document(4, dt_array_doc(&[utc(2025, 3, 15)]))
        .unwrap();
    store.commit().unwrap();
}

#[test]
fn datetime_array_range_query_matches_if_any_instant_is_in_range() {
    let store = store(true, true, true);
    seed(&store);

    // Q2 2024: only doc 1 (via its June instant).
    assert_eq!(
        query_hits(&store, between(utc(2024, 4, 1), utc(2024, 6, 30))),
        vec![1]
    );
    // Whole 2024: docs 1, 2 (Dec), 3.
    assert_eq!(
        query_hits(&store, between(utc(2024, 1, 1), utc(2024, 12, 31))),
        vec![1, 2, 3]
    );
    // 2025 onwards: docs 2 (next Jan) and 4.
    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::on_or_after("seen_at", utc(2025, 1, 1))
        ),
        vec![2, 4]
    );
}

/// A document with several matching instants is reported once, and the
/// result count equals the document count.
#[test]
fn datetime_array_document_is_reported_once() {
    let store = store(true, true, true);
    seed(&store);

    let results = store
        .search(
            LexicalSearchRequest::new(Box::new(between(utc(2024, 1, 1), utc(2024, 12, 31))))
                .limit(100),
        )
        .unwrap();
    let mut ids: Vec<u64> = results.hits.iter().map(|h| h.doc_id).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3]);
    assert_eq!(
        results.hits.len(),
        3,
        "doc 1 has two instants in 2024 but is reported once"
    );
}

/// Sub-second instants are indexed at micro-second precision, so a window
/// inside one second selects the right element.
#[test]
fn datetime_array_honors_sub_second_instants() {
    let store = store(true, true, true);
    let base = 1_700_000_000;
    let at = |nanos: u32| Utc.timestamp_opt(base, nanos).unwrap();
    store
        .upsert_document(1, dt_array_doc(&[at(100_000_000), at(900_000_000)]))
        .unwrap();
    store
        .upsert_document(2, dt_array_doc(&[at(500_000_000)]))
        .unwrap();
    store.commit().unwrap();

    assert_eq!(
        query_hits(&store, between(at(400_000_000), at(600_000_000))),
        vec![2]
    );
    assert_eq!(
        query_hits(&store, between(at(850_000_000), at(950_000_000))),
        vec![1]
    );
}

/// `indexed = true, stored = false`: the field lives solely in the BKD
/// tree, so every instant of a `DateTimeArray` must reach it — no
/// stored-document fallback can mask an element that was never indexed.
#[test]
fn index_only_datetime_array_field_matches_via_bkd() {
    let store = store(true, false, true);
    seed(&store);

    assert_eq!(
        query_hits(&store, between(utc(2024, 4, 1), utc(2024, 6, 30))),
        vec![1],
        "BKD-only instants must produce hits on a stored=false field"
    );
    assert_eq!(
        query_hits(&store, between(utc(2024, 1, 1), utc(2024, 12, 31))),
        vec![1, 2, 3]
    );
}

/// `indexed = false, stored = true`: every hit comes through the
/// stored-document fallback, which must scan every instant of the array —
/// for the typed query, the numeric API, and the DSL alike.
#[test]
fn stored_only_datetime_array_field_matches_via_fallback() {
    let store = store(false, true, true);
    seed(&store);

    assert_eq!(
        query_hits(&store, between(utc(2024, 4, 1), utc(2024, 6, 30))),
        vec![1]
    );
    assert_eq!(
        query_hits(
            &store,
            NumericRangeQuery::new(
                "seen_at",
                NumericType::Float,
                Some(utc(2025, 1, 1).timestamp() as f64),
                None,
                true,
                true,
            )
        ),
        vec![2, 4]
    );
    assert_eq!(
        dsl_hits(&store, "seen_at:[2024-01-01 TO 2024-12-31]"),
        vec![1, 2, 3]
    );
}

/// The DSL date range (all surfaces' path) sees every instant.
#[test]
fn dsl_date_range_over_datetime_array() {
    let store = store(true, true, true);
    seed(&store);

    assert_eq!(
        dsl_hits(&store, "seen_at:[2024-04-01 TO 2024-06-30]"),
        vec![1]
    );
    assert_eq!(dsl_hits(&store, "seen_at:{2024-12-31 TO *}"), vec![2, 4]);
    assert_eq!(
        dsl_hits(
            &store,
            "seen_at:[2024-03-15T00:00:00Z TO 2024-03-15T00:00:00Z]"
        ),
        vec![3]
    );
}

#[test]
fn datetime_array_queries_span_segments() {
    let store = store(true, true, true);
    store
        .upsert_document(1, dt_array_doc(&[utc(2024, 1, 15), utc(2024, 6, 15)]))
        .unwrap();
    store
        .upsert_document(10, Document::builder().add_text("tag", "no date").build())
        .unwrap();
    store.commit().unwrap();
    store
        .upsert_document(2, dt_array_doc(&[utc(2024, 12, 15), utc(2025, 1, 15)]))
        .unwrap();
    store.commit().unwrap();
    store
        .upsert_document(3, dt_array_doc(&[utc(2024, 3, 15)]))
        .unwrap();
    store.commit().unwrap();

    assert_eq!(
        query_hits(&store, between(utc(2024, 1, 1), utc(2024, 12, 31))),
        vec![1, 2, 3]
    );
    assert_eq!(dsl_hits(&store, "seen_at:[2024-06-01 TO *]"), vec![1, 2]);
}

/// An empty instant list is a valid value: stored, read back as an empty
/// array, and matching no range.
#[test]
fn empty_datetime_array_is_stored_and_never_matches() {
    let store = store(true, true, true);
    store
        .upsert_document(
            1,
            Document::builder()
                .add_datetime_array("seen_at", Vec::new())
                .add_text("tag", "empty")
                .build(),
        )
        .unwrap();
    store
        .upsert_document(2, dt_array_doc(&[utc(2024, 3, 15)]))
        .unwrap();
    store.commit().unwrap();

    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::new("seen_at", None, None, true, true)
        ),
        vec![2]
    );
    let results = store
        .search(
            LexicalSearchRequest::from_dsl("tag:empty")
                .limit(10)
                .load_documents(true),
        )
        .unwrap();
    assert_eq!(results.hits.len(), 1);
    let doc = results.hits[0].document.as_ref().expect("document loaded");
    assert_eq!(
        doc.get_field("seen_at"),
        Some(&DataValue::DateTimeArray(Vec::new()))
    );
}

/// A single-valued DateTime field behaves exactly as before.
#[test]
fn single_valued_datetime_field_is_unchanged() {
    let store = store(true, true, false);
    store.upsert_document(1, dt_doc(utc(2024, 1, 15))).unwrap();
    store.upsert_document(2, dt_doc(utc(2024, 6, 15))).unwrap();
    store.commit().unwrap();
    store.upsert_document(3, dt_doc(utc(2025, 1, 15))).unwrap();
    store.commit().unwrap();

    assert_eq!(
        query_hits(&store, between(utc(2024, 1, 1), utc(2024, 12, 31))),
        vec![1, 2]
    );
    assert_eq!(dsl_hits(&store, "seen_at:[2025-01-01 TO *]"), vec![3]);
    let results = store
        .search(
            LexicalSearchRequest::new(Box::new(DateTimeRangeQuery::new(
                "seen_at", None, None, true, true,
            )))
            .limit(10)
            .load_documents(true),
        )
        .unwrap();
    assert_eq!(results.hits.len(), 3);
    for hit in &results.hits {
        let doc = hit.document.as_ref().unwrap();
        assert!(matches!(
            doc.get_field("seen_at"),
            Some(DataValue::DateTime(_))
        ));
    }
}
