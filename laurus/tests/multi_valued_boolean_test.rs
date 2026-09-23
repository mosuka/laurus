//! End-to-end tests for multi-valued Boolean fields (Issue #1180).
//!
//! Drives `LexicalStore` the way production search does. Unlike the
//! BKD-backed multi-valued types, a `BoolArray` is indexed purely as one
//! `"true"` / `"false"` term posting per element, so "any element matches"
//! falls out of ordinary term-query semantics: a document is reported once
//! per matching term, and repeated elements raise the term frequency
//! (Lucene multi-valued parity) rather than the hit count.

use std::sync::Arc;

use laurus::lexical::core::field::{BooleanOption, FieldOption, TextOption};
use laurus::lexical::query::{Query, TermQuery};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};

fn bool_doc(flag: bool) -> Document {
    Document::builder().add_boolean("flags", flag).build()
}

fn bool_array_doc(flags: &[bool]) -> Document {
    Document::builder()
        .add_bool_array("flags", flags.to_vec())
        .build()
}

fn store(indexed: bool, stored: bool, multi_valued: bool) -> LexicalStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "flags",
            FieldOption::Boolean(BooleanOption {
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

/// `flags:true` / `flags:false` as the typed query every surface lowers to.
fn is(flag: bool) -> TermQuery {
    TermQuery::new("flags", if flag { "true" } else { "false" })
}

/// Doc 1: `[true, false]`; doc 2: `[false]`; doc 3: `[true]`.
fn seed(store: &LexicalStore) {
    store
        .upsert_document(1, bool_array_doc(&[true, false]))
        .unwrap();
    store.upsert_document(2, bool_array_doc(&[false])).unwrap();
    store.upsert_document(3, bool_array_doc(&[true])).unwrap();
    store.commit().unwrap();
}

#[test]
fn bool_array_term_query_matches_if_any_element_matches() {
    let store = store(true, true, true);
    seed(&store);

    assert_eq!(query_hits(&store, is(true)), vec![1, 3]);
    assert_eq!(query_hits(&store, is(false)), vec![1, 2]);
}

/// A document whose array carries the queried value several times is
/// reported once: the writer aggregates postings per (doc, term).
#[test]
fn bool_array_document_is_reported_once() {
    let store = store(true, true, true);
    store
        .upsert_document(1, bool_array_doc(&[true, true, false]))
        .unwrap();
    store.upsert_document(2, bool_array_doc(&[false])).unwrap();
    store.commit().unwrap();

    let results = store
        .search(LexicalSearchRequest::new(Box::new(is(true))).limit(100))
        .unwrap();
    assert_eq!(
        results.hits.len(),
        1,
        "doc 1 carries `true` twice but is one hit"
    );
    assert_eq!(results.hits[0].doc_id, 1);
    assert_eq!(query_hits(&store, is(false)), vec![1, 2]);
}

/// Pins the "no dedupe" decision: repeated elements raise the term
/// frequency, so under BM25 `[true, true]` outranks `[true]` — Lucene
/// multi-valued parity, not constant scoring (that is #580).
#[test]
fn duplicate_elements_raise_term_frequency_not_hit_count() {
    let store = store(true, true, true);
    store
        .upsert_document(1, bool_array_doc(&[true, true]))
        .unwrap();
    store.upsert_document(2, bool_array_doc(&[true])).unwrap();
    store.commit().unwrap();

    let results = store
        .search(LexicalSearchRequest::new(Box::new(is(true))).limit(100))
        .unwrap();
    assert_eq!(results.hits.len(), 2);
    let score_of = |doc_id: u64| {
        results
            .hits
            .iter()
            .find(|h| h.doc_id == doc_id)
            .map(|h| h.score)
            .expect("both documents are hits")
    };
    assert!(
        score_of(1) > score_of(2),
        "tf=2 must outscore tf=1: {} vs {}",
        score_of(1),
        score_of(2)
    );
}

/// `indexed = true, stored = false`: the field lives solely in the
/// postings, so every element of a `BoolArray` must reach them.
#[test]
fn index_only_bool_array_field_matches_via_postings() {
    let store = store(true, false, true);
    seed(&store);

    assert_eq!(query_hits(&store, is(true)), vec![1, 3]);
    assert_eq!(query_hits(&store, is(false)), vec![1, 2]);
}

/// `indexed = false, stored = true`: a term query has no stored-document
/// fallback (unlike range queries), so — exactly as for a scalar Bool — it
/// finds nothing, while the array is still stored intact.
#[test]
fn stored_only_bool_array_field_has_no_term_hits() {
    let store = store(false, true, true);
    store
        .upsert_document(
            1,
            Document::builder()
                .add_bool_array("flags", vec![true, false])
                .add_text("tag", "kept")
                .build(),
        )
        .unwrap();
    store.commit().unwrap();

    assert_eq!(query_hits(&store, is(true)), Vec::<u64>::new());
    let results = store
        .search(
            LexicalSearchRequest::from_dsl("tag:kept")
                .limit(10)
                .load_documents(true),
        )
        .unwrap();
    assert_eq!(results.hits.len(), 1);
    let doc = results.hits[0].document.as_ref().expect("document loaded");
    assert_eq!(
        doc.get_field("flags"),
        Some(&DataValue::BoolArray(vec![true, false]))
    );
}

/// The DSL term form (all surfaces' path) sees every element.
#[test]
fn dsl_term_over_bool_array() {
    let store = store(true, true, true);
    seed(&store);

    assert_eq!(dsl_hits(&store, "flags:true"), vec![1, 3]);
    assert_eq!(dsl_hits(&store, "flags:false"), vec![1, 2]);
}

#[test]
fn bool_array_queries_span_segments() {
    let store = store(true, true, true);
    store
        .upsert_document(1, bool_array_doc(&[true, false]))
        .unwrap();
    store
        .upsert_document(10, Document::builder().add_text("tag", "no flags").build())
        .unwrap();
    store.commit().unwrap();
    store.upsert_document(2, bool_array_doc(&[false])).unwrap();
    store.commit().unwrap();
    store.upsert_document(3, bool_array_doc(&[true])).unwrap();
    store.commit().unwrap();

    assert_eq!(query_hits(&store, is(true)), vec![1, 3]);
    assert_eq!(dsl_hits(&store, "flags:false"), vec![1, 2]);
}

/// An empty flag list is a valid value: stored, read back as an empty
/// array, and matching neither term.
#[test]
fn empty_bool_array_is_stored_and_never_matches() {
    let store = store(true, true, true);
    store
        .upsert_document(
            1,
            Document::builder()
                .add_bool_array("flags", Vec::new())
                .add_text("tag", "empty")
                .build(),
        )
        .unwrap();
    store.upsert_document(2, bool_array_doc(&[true])).unwrap();
    store.commit().unwrap();

    assert_eq!(query_hits(&store, is(true)), vec![2]);
    assert_eq!(query_hits(&store, is(false)), Vec::<u64>::new());
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
        doc.get_field("flags"),
        Some(&DataValue::BoolArray(Vec::new()))
    );
}

/// A single-valued Boolean field behaves exactly as before (the array
/// rejection itself is enforced by the engine's coercion, covered in
/// `dynamic_schema_test.rs`).
#[test]
fn single_valued_boolean_field_is_unchanged() {
    let store = store(true, true, false);
    store.upsert_document(1, bool_doc(true)).unwrap();
    store.upsert_document(2, bool_doc(false)).unwrap();
    store.commit().unwrap();
    store.upsert_document(3, bool_doc(true)).unwrap();
    store.commit().unwrap();

    assert_eq!(query_hits(&store, is(true)), vec![1, 3]);
    assert_eq!(dsl_hits(&store, "flags:false"), vec![2]);
    let results = store
        .search(
            LexicalSearchRequest::new(Box::new(is(true)))
                .limit(10)
                .load_documents(true),
        )
        .unwrap();
    assert_eq!(results.hits.len(), 2);
    for hit in &results.hits {
        let doc = hit.document.as_ref().unwrap();
        assert_eq!(doc.get_field("flags"), Some(&DataValue::Bool(true)));
    }
}
