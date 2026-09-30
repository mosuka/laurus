//! End-to-end tests for multi-valued numeric fields.
//!
//! These tests drive the inverted index writer / reader pair the way
//! production search does and verify the Lucene-style "any value
//! matches" semantics for both `Int64Array` and `Float64Array` fields.

use laurus::lexical::NumericRangeQuery;
use laurus::lexical::NumericType;
use laurus::lexical::Query;
use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::writer::LexicalIndexWriter;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};
use std::sync::Arc;

/// Drain a matcher into a Vec of doc IDs in iteration order.
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

/// An index and a writer registered with it (#1024): a standalone
/// `InvertedIndexWriter` is ephemeral — its segments enter no manifest, so
/// no reader sees them — so durable fixtures go through
/// `InvertedIndex::create` + `writer()` and read back through
/// `index.reader()`.
fn index_and_writer(
    storage: Arc<dyn laurus::storage::Storage>,
) -> (InvertedIndex, Box<dyn LexicalIndexWriter>) {
    let index = InvertedIndex::create(storage, Default::default()).unwrap();
    let writer = index.writer().unwrap();
    (index, writer)
}

#[test]
fn int64_array_any_value_in_range() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (index, mut writer) = index_and_writer(storage.clone());

    // Doc 0: scores = [85, 72, 95]  (95 in [80, 100])
    // Doc 1: scores = [60, 65]      (no value in [80, 100])
    // Doc 2: scores = [88]          (88 in [80, 100])
    writer
        .add_document(
            Document::builder()
                .add_int64_array("scores", vec![85, 72, 95])
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_int64_array("scores", vec![60, 65])
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_int64_array("scores", vec![88])
                .build(),
        )
        .unwrap();

    writer.commit().unwrap();

    let reader = index.reader().unwrap();

    // Range [80, 100] inclusive.
    let q = NumericRangeQuery::new(
        "scores",
        NumericType::Integer,
        Some(80.0),
        Some(100.0),
        true,
        true,
    );

    let matched = collect_matcher_results(q.matcher(&*reader).unwrap());
    assert_eq!(
        matched,
        vec![0, 2],
        "doc 0 (95) and doc 2 (88) should match"
    );

    // Range [10, 20] — no document has any value in this range.
    let q_low = NumericRangeQuery::new(
        "scores",
        NumericType::Integer,
        Some(10.0),
        Some(20.0),
        true,
        true,
    );
    let matched_low = collect_matcher_results(q_low.matcher(&*reader).unwrap());
    assert!(
        matched_low.is_empty(),
        "no doc should match a disjoint range, got {matched_low:?}",
    );
}

#[test]
fn int64_array_dedups_doc_when_multiple_values_match() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (index, mut writer) = index_and_writer(storage.clone());

    // Doc 0 has THREE values inside [50, 100]: 60, 80, 90.
    // The doc must be reported only once (Lucene dedup contract).
    writer
        .add_document(
            Document::builder()
                .add_int64_array("scores", vec![60, 80, 90])
                .build(),
        )
        .unwrap();

    writer.commit().unwrap();

    let reader = index.reader().unwrap();

    let q = NumericRangeQuery::new(
        "scores",
        NumericType::Integer,
        Some(50.0),
        Some(100.0),
        true,
        true,
    );
    let matched = collect_matcher_results(q.matcher(&*reader).unwrap());
    assert_eq!(matched, vec![0], "doc must be reported exactly once");
}

#[test]
fn float64_array_any_value_in_range() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (index, mut writer) = index_and_writer(storage.clone());

    // Doc 0: prices = [12.5, 99.9, 7.0]  (99.9 in [50.0, 100.0])
    // Doc 1: prices = [200.0, 250.0]     (none in range)
    writer
        .add_document(
            Document::builder()
                .add_float64_array("prices", vec![12.5, 99.9, 7.0])
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_float64_array("prices", vec![200.0, 250.0])
                .build(),
        )
        .unwrap();

    writer.commit().unwrap();

    let reader = index.reader().unwrap();

    let q = NumericRangeQuery::new(
        "prices",
        NumericType::Float,
        Some(50.0),
        Some(100.0),
        true,
        true,
    );
    let matched = collect_matcher_results(q.matcher(&*reader).unwrap());
    assert_eq!(matched, vec![0]);
}

#[test]
fn single_valued_field_unchanged_by_multi_valued_changes() {
    // Regression: existing single-valued integer fields must keep their
    // pre-existing behaviour after the multi-value plumbing change.
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (index, mut writer) = index_and_writer(storage.clone());

    writer
        .add_document(
            Document::builder()
                .add_field("age", DataValue::Int64(30))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("age", DataValue::Int64(20))
                .build(),
        )
        .unwrap();
    writer
        .add_document(
            Document::builder()
                .add_field("age", DataValue::Int64(40))
                .build(),
        )
        .unwrap();
    writer.commit().unwrap();

    let reader = index.reader().unwrap();
    let q = NumericRangeQuery::new(
        "age",
        NumericType::Integer,
        Some(25.0),
        Some(35.0),
        true,
        true,
    );
    let matched = collect_matcher_results(q.matcher(&*reader).unwrap());
    assert_eq!(matched, vec![0]);
}
