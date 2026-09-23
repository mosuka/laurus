//! End-to-end tests for DateTime range queries (Issue #1179).
//!
//! `DateTimeRangeQuery` used to be a stub whose matcher always errored, and
//! the query DSL turned `created_at:[2024-01-01 TO 2024-12-31]` into a
//! literal `TermQuery` that matched nothing. These tests drive
//! `LexicalStore` the way production search does and cover:
//!
//! - the BKD path (inclusive / exclusive edges, one-sided bounds,
//!   **sub-second bounds** — the writer must index the fractional part);
//! - the stored-document fallback for `indexed = false, stored = true`;
//! - multi-segment fan-out;
//! - the DSL forms the documentation promises;
//! - the pre-#1179 programmatic path (`NumericRangeQuery` over the field's
//!   timestamp point), which must keep working.

use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use laurus::lexical::NumericType;
use laurus::lexical::core::field::{DateTimeOption, FieldOption, TextOption};
use laurus::lexical::query::{DateTimeRangeQuery, NumericRangeQuery, Query};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};

fn utc(y: i32, m: u32, d: u32, hh: u32, mm: u32, ss: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(y, m, d, hh, mm, ss).unwrap()
}

fn dt_doc(dt: DateTime<Utc>) -> Document {
    Document::builder()
        .add_field("created_at", DataValue::DateTime(dt))
        .build()
}

fn store(indexed: bool) -> LexicalStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "created_at",
            FieldOption::DateTime(DateTimeOption {
                indexed,
                stored: true,
                doc_values: true,
            }),
        )
        .add_field("body", FieldOption::Text(TextOption::default()))
        .build();
    LexicalStore::new(storage, config).unwrap()
}

/// Four docs: New Year, mid-June, New Year's Eve, and next New Year.
fn seed(store: &LexicalStore) {
    store
        .upsert_document(1, dt_doc(utc(2024, 1, 1, 0, 0, 0)))
        .unwrap();
    store
        .upsert_document(2, dt_doc(utc(2024, 6, 15, 12, 0, 0)))
        .unwrap();
    store
        .upsert_document(3, dt_doc(utc(2024, 12, 31, 0, 0, 0)))
        .unwrap();
    store
        .upsert_document(4, dt_doc(utc(2025, 1, 1, 0, 0, 0)))
        .unwrap();
    store.commit().unwrap();
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

#[test]
fn bkd_path_matches_inclusive_and_exclusive_edges() {
    let store = store(true);
    seed(&store);

    let start = utc(2024, 1, 1, 0, 0, 0);
    let end = utc(2024, 12, 31, 0, 0, 0);
    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::between("created_at", start, end)
        ),
        vec![1, 2, 3]
    );
    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::new("created_at", Some(start), Some(end), false, false)
        ),
        vec![2]
    );
    assert_eq!(
        query_hits(&store, DateTimeRangeQuery::after("created_at", end)),
        vec![4]
    );
    assert_eq!(
        query_hits(&store, DateTimeRangeQuery::on_or_after("created_at", end)),
        vec![3, 4]
    );
    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::before("created_at", utc(2024, 6, 15, 12, 0, 0))
        ),
        vec![1]
    );
    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::on_or_before("created_at", utc(2024, 6, 15, 12, 0, 0))
        ),
        vec![1, 2]
    );
}

/// The writer must index the fractional second: with the pre-#1179
/// whole-second point all three docs collapse onto `1700000000.0` and the
/// `[.4, .6]` window matches nothing.
#[test]
fn bkd_path_honors_sub_second_bounds() {
    let store = store(true);
    let base = 1_700_000_000;
    let at = |nanos: u32| Utc.timestamp_opt(base, nanos).unwrap();
    store.upsert_document(1, dt_doc(at(250_000_000))).unwrap();
    store.upsert_document(2, dt_doc(at(500_000_000))).unwrap();
    store.upsert_document(3, dt_doc(at(750_000_000))).unwrap();
    store.commit().unwrap();

    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::between("created_at", at(400_000_000), at(600_000_000))
        ),
        vec![2]
    );
    // An exact instant is expressible as a degenerate inclusive range.
    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::between("created_at", at(750_000_000), at(750_000_000))
        ),
        vec![3]
    );
    // Exclusive edges stay exclusive at micro-second precision.
    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::new(
                "created_at",
                Some(at(250_000_000)),
                Some(at(750_000_000)),
                false,
                false
            )
        ),
        vec![2]
    );
}

/// `indexed = false, stored = true`: no BKD tree exists, so every hit
/// comes through the stored-document fallback, which must understand
/// `DataValue::DateTime`.
#[test]
fn stored_only_datetime_field_matches_via_fallback() {
    let store = store(false);
    seed(&store);

    let start = utc(2024, 1, 1, 0, 0, 0);
    let end = utc(2024, 12, 31, 0, 0, 0);
    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::between("created_at", start, end)
        ),
        vec![1, 2, 3]
    );
    assert_eq!(
        query_hits(&store, DateTimeRangeQuery::after("created_at", end)),
        vec![4]
    );
    // The numeric API over the same stored values goes through the same
    // fallback (it used to ignore `DateTime` and return nothing).
    assert_eq!(
        query_hits(
            &store,
            NumericRangeQuery::new(
                "created_at",
                NumericType::Float,
                Some(utc(2024, 6, 15, 12, 0, 0).timestamp() as f64),
                None,
                true,
                true,
            )
        ),
        vec![2, 3, 4]
    );
    assert_eq!(
        dsl_hits(&store, "created_at:[2024-06-01 TO 2024-12-31]"),
        vec![2, 3]
    );
}

#[test]
fn datetime_range_across_segments() {
    let store = store(true);
    store
        .upsert_document(1, dt_doc(utc(2024, 1, 1, 0, 0, 0)))
        .unwrap();
    store
        .upsert_document(10, Document::builder().add_text("body", "no date").build())
        .unwrap();
    store.commit().unwrap();
    store
        .upsert_document(2, dt_doc(utc(2024, 6, 15, 12, 0, 0)))
        .unwrap();
    store.commit().unwrap();
    store
        .upsert_document(3, dt_doc(utc(2024, 12, 31, 0, 0, 0)))
        .unwrap();
    store
        .upsert_document(4, dt_doc(utc(2025, 1, 1, 0, 0, 0)))
        .unwrap();
    store.commit().unwrap();

    assert_eq!(
        query_hits(
            &store,
            DateTimeRangeQuery::between(
                "created_at",
                utc(2024, 1, 1, 0, 0, 0),
                utc(2024, 12, 31, 0, 0, 0)
            )
        ),
        vec![1, 2, 3]
    );
    assert_eq!(
        dsl_hits(&store, "created_at:[2024-06-01 TO *]"),
        vec![2, 3, 4]
    );
}

/// The forms the documentation promises, through the DSL string path
/// every surface (server, CLI, bindings) uses.
#[test]
fn dsl_date_range_through_lexical_search_request() {
    let store = store(true);
    seed(&store);

    assert_eq!(
        dsl_hits(&store, "created_at:[2024-01-01 TO 2024-12-31]"),
        vec![1, 2, 3]
    );
    // Exclusive braces (the `docs/src/concepts/query_dsl.md` example).
    assert_eq!(
        dsl_hits(&store, "created_at:{2024-01-01 TO 2024-12-31}"),
        vec![2]
    );
    // RFC 3339 with an offset, normalized to UTC; naive datetimes are UTC.
    assert_eq!(
        dsl_hits(
            &store,
            "created_at:[2024-06-15T21:00:00+09:00 TO 2025-01-01T00:00:00]"
        ),
        vec![2, 3, 4]
    );
    assert_eq!(
        dsl_hits(&store, "created_at:[* TO 2024-06-15T12:00:00Z]"),
        vec![1, 2]
    );
    // Bare numbers are epoch seconds and keep going through the numeric path.
    let june = utc(2024, 6, 15, 12, 0, 0).timestamp();
    assert_eq!(
        dsl_hits(&store, &format!("created_at:[{june} TO *]")),
        vec![2, 3, 4]
    );
    // Mixed shapes are an error, not a silent empty result.
    assert!(
        store
            .search(LexicalSearchRequest::from_dsl(
                "created_at:[100 TO 2024-12-31]"
            ))
            .is_err()
    );
    assert!(
        store
            .search(LexicalSearchRequest::from_dsl(
                "created_at:[2024-01-01 TO yesterday]"
            ))
            .is_err()
    );
}

/// `bkd_integration_test.rs` queries a DateTime field with
/// `NumericRangeQuery` over epoch seconds; that must keep working.
#[test]
fn numeric_range_query_over_datetime_field_still_works() {
    let store = store(true);
    seed(&store);

    let june = utc(2024, 6, 15, 12, 0, 0).timestamp() as f64;
    assert_eq!(
        query_hits(
            &store,
            NumericRangeQuery::new(
                "created_at",
                NumericType::Float,
                None,
                Some(june),
                true,
                false
            )
        ),
        vec![1]
    );
    assert_eq!(
        query_hits(
            &store,
            NumericRangeQuery::new(
                "created_at",
                NumericType::Integer,
                Some(june),
                None,
                true,
                true
            )
        ),
        vec![2, 3, 4]
    );
}
