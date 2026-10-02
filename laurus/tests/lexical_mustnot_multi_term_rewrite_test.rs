//! Regression test for Issue #1313: a `BooleanQuery` `MustNot` clause
//! holding a multi-term query (prefix / wildcard / fuzzy / regexp)
//! excluded nothing on a multi-segment index, because
//! `BooleanQuery::rewrite` only lowered `Should` / `Must` clauses and
//! `MustNot` clauses matched directly against the per-segment fanout's
//! `PerSegmentReaderView`, which cannot enumerate terms.

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::query::Query;
use laurus::lexical::query::advanced_query::AdvancedQuery;
use laurus::lexical::query::boolean::{BooleanClause, BooleanQuery, Occur};
use laurus::lexical::query::prefix::PrefixQuery;
use laurus::lexical::query::term::TermQuery;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

/// The issue's own reproduction corpus: ids 0-2 in one commit, 3-5 in
/// another, so `max_segments(1000)` keeps them as two segments.
const CORPUS: [(u64, &str); 6] = [
    (0, "apple red fresh"),
    (1, "apple green"),
    (2, "banana fresh"),
    (3, "apple fresh"),
    (4, "apple ripe"),
    (5, "cherry"),
];

fn doc(body: &str) -> Document {
    Document::builder().add_text("body", body).build()
}

fn store_with_segments(n_segments: usize) -> LexicalStore {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder().max_segments(1000).build();
    let store = LexicalStore::new(storage, config).unwrap();

    let chunk = CORPUS.len().div_ceil(n_segments);
    for group in CORPUS.chunks(chunk) {
        for (id, body) in group {
            store.upsert_document(*id, doc(body)).unwrap();
        }
        store.commit().unwrap();
    }
    store
}

/// `Must(body:apple)` + `MustNot(PrefixQuery(body, "fr"))`. "fr" prefixes
/// "fresh" only, so the negative excludes docs 0, 2 and 3; combined with
/// the positive `apple` match (0, 1, 3, 4) the expected hit set is `[1, 4]`.
fn must_not_prefix_query() -> Box<dyn Query> {
    let mut query = BooleanQuery::new();
    query.add_clause(BooleanClause::must(Box::new(TermQuery::new(
        "body", "apple",
    ))));
    query.add_clause(BooleanClause::new(
        Box::new(PrefixQuery::new("body", "fr")),
        Occur::MustNot,
    ));
    Box::new(query)
}

fn search_ids(store: &LexicalStore, request: LexicalSearchRequest) -> Vec<u64> {
    let mut ids: Vec<u64> = store
        .search(request)
        .unwrap()
        .hits
        .iter()
        .map(|h| h.doc_id)
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn must_not_prefix_excludes_on_single_segment() {
    let store = store_with_segments(1);
    let ids = search_ids(
        &store,
        LexicalSearchRequest::new(must_not_prefix_query()).limit(10),
    );
    assert_eq!(ids, vec![1, 4]);
}

#[test]
fn must_not_prefix_excludes_on_multi_segment_score_sorted() {
    let store = store_with_segments(2);
    let ids = search_ids(
        &store,
        LexicalSearchRequest::new(must_not_prefix_query()).limit(10),
    );
    assert_eq!(ids, vec![1, 4]);
}

#[test]
fn must_not_prefix_excludes_on_multi_segment_parallel() {
    let store = store_with_segments(2);
    let ids = search_ids(
        &store,
        LexicalSearchRequest::new(must_not_prefix_query())
            .limit(10)
            .parallel(true),
    );
    assert_eq!(ids, vec![1, 4]);
}

/// Since #1305, an `AdvancedQuery`'s negative filters become `MustNot`
/// clauses too, so a multi-term negative filter has the same gap.
#[test]
fn advanced_query_multi_term_negative_filter_excludes_on_multi_segment() {
    let store = store_with_segments(2);
    let query = AdvancedQuery::new(Box::new(TermQuery::new("body", "apple")))
        .with_negative_filter(Box::new(PrefixQuery::new("body", "fr")));
    let ids = search_ids(&store, LexicalSearchRequest::new(Box::new(query)).limit(10));
    assert_eq!(ids, vec![1, 4]);
}
