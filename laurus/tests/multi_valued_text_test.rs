//! End-to-end tests for multi-valued Text fields (Issue #1175).
//!
//! Drives `LexicalStore` the way production search does. A `TextArray`
//! analyzes every element separately onto one ascending position sequence,
//! with `TextOption::position_increment_gap` positions between elements:
//! term queries match if any element carries the term, while a phrase
//! query cannot cross an element boundary unless its slop reaches the gap.

use std::sync::Arc;

use laurus::lexical::core::field::{DEFAULT_POSITION_INCREMENT_GAP, FieldOption, TextOption};
use laurus::lexical::query::{PhraseQuery, Query, TermQuery};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};

fn text_doc(text: &str) -> Document {
    Document::builder().add_text("body", text).build()
}

fn text_array_doc(values: &[&str]) -> Document {
    Document::builder()
        .add_text_array("body", values.iter().map(|s| s.to_string()).collect())
        .build()
}

/// `term_vectors: true` comes from the default, which every phrase test
/// relies on.
fn store_with(
    indexed: bool,
    stored: bool,
    multi_valued: bool,
    position_increment_gap: u32,
    term_vectors: bool,
) -> LexicalStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "body",
            FieldOption::Text(TextOption {
                indexed,
                stored,
                multi_valued,
                position_increment_gap,
                term_vectors,
                ..Default::default()
            }),
        )
        .add_field("tag", FieldOption::Text(TextOption::default()))
        .build();
    LexicalStore::new(storage, config).unwrap()
}

fn store(indexed: bool, stored: bool, multi_valued: bool) -> LexicalStore {
    store_with(
        indexed,
        stored,
        multi_valued,
        DEFAULT_POSITION_INCREMENT_GAP,
        true,
    )
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

fn term(t: &str) -> TermQuery {
    TermQuery::new("body", t)
}

fn phrase(terms: &[&str], slop: u32) -> PhraseQuery {
    PhraseQuery::new("body", terms.iter().map(|s| s.to_string()).collect()).with_slop(slop)
}

/// Doc 1: `["hello world", "foo bar"]`; doc 2: `["foo bar"]`; doc 3:
/// `["hello world"]`.
fn seed(store: &LexicalStore) {
    store
        .upsert_document(1, text_array_doc(&["hello world", "foo bar"]))
        .unwrap();
    store
        .upsert_document(2, text_array_doc(&["foo bar"]))
        .unwrap();
    store
        .upsert_document(3, text_array_doc(&["hello world"]))
        .unwrap();
    store.commit().unwrap();
}

#[test]
fn text_array_term_query_matches_if_any_element_matches() {
    let store = store(true, true, true);
    seed(&store);

    assert_eq!(query_hits(&store, term("hello")), vec![1, 3]);
    assert_eq!(query_hits(&store, term("foo")), vec![1, 2]);
}

/// A document whose elements repeat the queried term is reported once.
#[test]
fn text_array_document_is_reported_once() {
    let store = store(true, true, true);
    store
        .upsert_document(1, text_array_doc(&["rust rocks", "rust rolls"]))
        .unwrap();
    store.upsert_document(2, text_array_doc(&["rust"])).unwrap();
    store.commit().unwrap();

    let results = store
        .search(LexicalSearchRequest::new(Box::new(term("rust"))).limit(100))
        .unwrap();
    assert_eq!(
        results.hits.len(),
        2,
        "doc 1 carries `rust` twice but is one hit"
    );
}

/// Repeated terms across elements raise the term frequency, not the hit
/// count — Lucene multi-valued parity, like `BoolArray` (#1180).
#[test]
fn duplicate_terms_across_elements_raise_term_frequency_not_hit_count() {
    let store = store(true, true, true);
    store
        .upsert_document(1, text_array_doc(&["rust", "rust"]))
        .unwrap();
    store.upsert_document(2, text_array_doc(&["rust"])).unwrap();
    store.commit().unwrap();

    let results = store
        .search(LexicalSearchRequest::new(Box::new(term("rust"))).limit(100))
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
/// postings, so every element must reach them.
#[test]
fn index_only_text_array_field_matches_via_postings() {
    let store = store(true, false, true);
    seed(&store);

    assert_eq!(query_hits(&store, term("hello")), vec![1, 3]);
    assert_eq!(query_hits(&store, phrase(&["foo", "bar"], 0)), vec![1, 2]);
}

/// `indexed = false, stored = true`: no term hits (a term query has no
/// stored-document fallback), while the array is still stored intact.
#[test]
fn stored_only_text_array_field_has_no_term_hits() {
    let store = store(false, true, true);
    store
        .upsert_document(
            1,
            Document::builder()
                .add_text_array("body", vec!["hello world".into(), "foo bar".into()])
                .add_text("tag", "kept")
                .build(),
        )
        .unwrap();
    store.commit().unwrap();

    assert_eq!(query_hits(&store, term("hello")), Vec::<u64>::new());
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
        doc.get_field("body"),
        Some(&DataValue::TextArray(vec![
            "hello world".into(),
            "foo bar".into()
        ]))
    );
}

#[test]
fn dsl_term_over_text_array() {
    let store = store(true, true, true);
    seed(&store);

    assert_eq!(dsl_hits(&store, "body:hello"), vec![1, 3]);
    assert_eq!(dsl_hits(&store, "body:bar"), vec![1, 2]);
}

/// Positions — gap included — survive a segment merge verbatim: the
/// cross-boundary phrase must still miss after the segments are merged.
#[test]
fn text_array_queries_span_segments() {
    let store = store(true, true, true);
    store
        .upsert_document(1, text_array_doc(&["hello world", "foo bar"]))
        .unwrap();
    store
        .upsert_document(10, Document::builder().add_text("tag", "no body").build())
        .unwrap();
    store.commit().unwrap();
    store
        .upsert_document(2, text_array_doc(&["foo bar"]))
        .unwrap();
    store.commit().unwrap();
    store
        .upsert_document(3, text_array_doc(&["hello world"]))
        .unwrap();
    store.commit().unwrap();

    assert_eq!(query_hits(&store, term("hello")), vec![1, 3]);
    assert_eq!(dsl_hits(&store, "body:bar"), vec![1, 2]);
    assert_eq!(
        query_hits(&store, phrase(&["world", "foo"], 0)),
        Vec::<u64>::new()
    );

    store.optimize().unwrap();
    assert_eq!(query_hits(&store, term("hello")), vec![1, 3]);
    assert_eq!(
        query_hits(&store, phrase(&["hello", "world"], 0)),
        vec![1, 3]
    );
    assert_eq!(
        query_hits(&store, phrase(&["world", "foo"], 0)),
        Vec::<u64>::new(),
        "the position gap must survive the merge"
    );
}

/// An empty list is a valid value: stored, read back as an empty array,
/// and matching no term.
#[test]
fn empty_text_array_is_stored_and_never_matches() {
    let store = store(true, true, true);
    store
        .upsert_document(
            1,
            Document::builder()
                .add_text_array("body", Vec::new())
                .add_text("tag", "empty")
                .build(),
        )
        .unwrap();
    store
        .upsert_document(2, text_array_doc(&["hello"]))
        .unwrap();
    store.commit().unwrap();

    assert_eq!(query_hits(&store, term("hello")), vec![2]);
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
        doc.get_field("body"),
        Some(&DataValue::TextArray(Vec::new()))
    );
}

/// A single-valued Text field behaves exactly as before (the array
/// rejection itself is enforced by the engine's coercion, covered in
/// `dynamic_schema_test.rs`).
#[test]
fn single_valued_text_field_is_unchanged() {
    let store = store(true, true, false);
    store.upsert_document(1, text_doc("hello world")).unwrap();
    store.upsert_document(2, text_doc("foo bar")).unwrap();
    store.commit().unwrap();

    assert_eq!(query_hits(&store, term("hello")), vec![1]);
    assert_eq!(query_hits(&store, phrase(&["hello", "world"], 0)), vec![1]);
    assert_eq!(
        query_hits(&store, phrase(&["world", "foo"], 0)),
        Vec::<u64>::new()
    );
    let results = store
        .search(
            LexicalSearchRequest::new(Box::new(term("hello")))
                .limit(10)
                .load_documents(true),
        )
        .unwrap();
    let doc = results.hits[0].document.as_ref().unwrap();
    assert!(matches!(doc.get_field("body"), Some(DataValue::Text(_))));
}

// ---- Phrase queries and the position-increment gap ----

#[test]
fn phrase_within_one_element_still_matches() {
    let store = store(true, true, true);
    seed(&store);

    assert_eq!(
        query_hits(&store, phrase(&["hello", "world"], 0)),
        vec![1, 3]
    );
    assert_eq!(query_hits(&store, phrase(&["foo", "bar"], 0)), vec![1, 2]);
}

/// The reason the gap exists: `["hello world", "foo bar"]` must not match
/// the phrase "world foo".
#[test]
fn phrase_across_element_boundary_does_not_match() {
    let store = store(true, true, true);
    seed(&store);

    assert_eq!(
        query_hits(&store, phrase(&["world", "foo"], 0)),
        Vec::<u64>::new()
    );
    assert_eq!(
        query_hits(&store, phrase(&["world", "foo"], 50)),
        Vec::<u64>::new()
    );
}

/// Pins the exact arithmetic: the element after "hello world" (positions
/// 0, 1) starts at `1 + 1 + gap`, and the matcher accepts
/// `pos <= expected + slop` with `expected = 2`, so the phrase crosses at
/// slop == gap and not one less.
#[test]
fn phrase_across_element_boundary_matches_at_slop_equal_to_gap() {
    let store = store(true, true, true);
    seed(&store);

    let gap = DEFAULT_POSITION_INCREMENT_GAP;
    assert_eq!(
        query_hits(&store, phrase(&["world", "foo"], gap - 1)),
        Vec::<u64>::new()
    );
    assert_eq!(query_hits(&store, phrase(&["world", "foo"], gap)), vec![1]);
}

/// The gap is configurable: at 0 the elements are numbered as if they had
/// been concatenated, so the cross-boundary phrase matches — and the
/// within-element phrase still matches too, which proves positions are
/// contiguous rather than restarted at 0 per element (a restart would
/// silently corrupt the delta-encoded posting list).
#[test]
fn zero_position_increment_gap_lets_the_cross_boundary_phrase_match() {
    let store = store_with(true, true, true, 0, true);
    seed(&store);

    assert_eq!(query_hits(&store, phrase(&["world", "foo"], 0)), vec![1]);
    assert_eq!(
        query_hits(&store, phrase(&["hello", "world"], 0)),
        vec![1, 3]
    );
    assert_eq!(query_hits(&store, phrase(&["foo", "bar"], 0)), vec![1, 2]);
}

/// The gap is charged per element (Lucene's per-value
/// `positionIncrementGap`), so an element that analyzes to nothing still
/// pushes the next element another gap further away.
#[test]
fn empty_element_still_consumes_a_gap() {
    let store = store_with(true, true, true, 100, true);
    store
        .upsert_document(1, text_array_doc(&["hello world", "", "foo bar"]))
        .unwrap();
    store.commit().unwrap();

    assert_eq!(
        query_hits(&store, phrase(&["world", "foo"], 199)),
        Vec::<u64>::new()
    );
    assert_eq!(query_hits(&store, phrase(&["world", "foo"], 200)), vec![1]);
}

/// Without positions a phrase query silently matches nothing — the
/// behaviour `lexical_term_vectors_test.rs` pins for scalar Text — and a
/// multi-valued field is no different.
#[test]
fn text_array_without_term_vectors_has_no_phrase_matches() {
    let store = store_with(true, true, true, DEFAULT_POSITION_INCREMENT_GAP, false);
    seed(&store);

    assert_eq!(query_hits(&store, term("hello")), vec![1, 3]);
    assert_eq!(
        query_hits(&store, phrase(&["hello", "world"], 0)),
        Vec::<u64>::new()
    );
}
