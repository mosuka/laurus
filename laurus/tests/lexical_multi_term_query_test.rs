//! Index-backed end-to-end tests for the multi-term query types
//! (Prefix / Wildcard / Fuzzy / Regexp) — Issue #613.
//!
//! These are the first hit-level tests for these query types (prior
//! coverage was structural only). The multi-segment cases pin the
//! Issue #613 bug: before the searcher-level rewrite, the per-segment
//! fanout handed each segment a `PerSegmentReaderView`, which cannot
//! enumerate the term dictionary, so a raw multi-term query against a
//! ≥2-segment index silently returned 0 hits.

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::query::Query;
use laurus::lexical::query::fuzzy::FuzzyQuery;
use laurus::lexical::query::prefix::PrefixQuery;
use laurus::lexical::query::regexp::RegexpQuery;
use laurus::lexical::query::wildcard::WildcardQuery;
use laurus::lexical::{
    BooleanQueryBuilder, LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery,
};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

fn doc(body: &str) -> Document {
    Document::builder().add_text("body", body).build()
}

/// Corpus with distinctive stems. Expected matches:
/// - prefix/wildcard/regexp `program…`: docs 1, 2, 3
/// - fuzzy "programing" (1 edit): doc 1 ("programming")
const CORPUS: [(u64, &str); 4] = [
    (1, "programming in rust"),
    (2, "a programmer at work"),
    (3, "program notes for the concert"),
    (4, "python coding session"),
];

/// Build a store with the corpus split across `n_segments` commits
/// (one commit per chunk; auto-merge disabled via a high threshold).
fn store_with_segments(n_segments: usize) -> (LexicalStore, Arc<dyn Storage>) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder().max_segments(1000).build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    let chunk = CORPUS.len().div_ceil(n_segments);
    for group in CORPUS.chunks(chunk) {
        for (id, body) in group {
            store.upsert_document(*id, doc(body)).unwrap();
        }
        store.commit().unwrap();
    }
    (store, storage)
}

fn search_ids(store: &LexicalStore, query: Box<dyn Query>) -> Vec<u64> {
    let mut ids: Vec<u64> = store
        .search(LexicalSearchRequest::new(query).limit(10))
        .unwrap()
        .hits
        .iter()
        .map(|h| h.doc_id)
        .collect();
    ids.sort_unstable();
    ids
}

/// Run the four query types against a store and assert hit sets.
fn assert_multi_term_hits(store: &LexicalStore, label: &str) {
    assert_eq!(
        search_ids(store, Box::new(PrefixQuery::new("body", "program"))),
        vec![1, 2, 3],
        "{label}: PrefixQuery(body, program)"
    );
    assert_eq!(
        search_ids(
            store,
            Box::new(WildcardQuery::new("body", "program*").unwrap())
        ),
        vec![1, 2, 3],
        "{label}: WildcardQuery(body, program*)"
    );
    assert_eq!(
        search_ids(
            store,
            Box::new(FuzzyQuery::new("body", "programing").max_edits(1))
        ),
        vec![1],
        "{label}: FuzzyQuery(body, programing, 1 edit)"
    );
    assert_eq!(
        search_ids(
            store,
            Box::new(RegexpQuery::new("body", "program.*").unwrap())
        ),
        vec![1, 2, 3],
        "{label}: RegexpQuery(body, program.*)"
    );
}

/// Single segment: all four multi-term query types return the expected
/// hits (this passed before #613's fix as well).
#[test]
fn multi_term_queries_hit_on_single_segment() {
    let (store, storage) = store_with_segments(1);
    let segs = storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.starts_with("segment_") && f.ends_with(".cfs"))
        .count();
    assert_eq!(segs, 1, "harness must produce exactly 1 segment");
    assert_multi_term_hits(&store, "single-segment");
}

/// Multi-segment: the same queries over a 2-segment index. Before the
/// Issue #613 searcher-level rewrite this returned 0 hits (the fanout's
/// `PerSegmentReaderView` cannot enumerate terms); it must return the
/// same hits as the single-segment case.
#[test]
fn multi_term_queries_hit_on_multi_segment() {
    let (store, storage) = store_with_segments(2);
    let segs = storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.starts_with("segment_") && f.ends_with(".cfs"))
        .count();
    assert_eq!(segs, 2, "harness must produce exactly 2 segments");
    assert_multi_term_hits(&store, "multi-segment");
}

/// A document written again under the same id is in two segments, deleted in
/// the first and live in the second. It matches by the terms of the second
/// only, and the deletion in the first does not hide it there (Issue #1740,
/// as #1211 for other queries).
#[test]
fn a_replaced_document_matches_by_its_live_version_only() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder().max_segments(1000).build();
    let store = LexicalStore::new(storage, config).unwrap();
    for (id, body) in [
        (1u64, "zqold every"),
        (2, "zqold every"),
        (3, "orange every"),
    ] {
        store.upsert_document(id, doc(body)).unwrap();
    }
    store.commit().unwrap();
    for (id, body) in [(1u64, "zqnew every"), (4, "zqnew every")] {
        store.upsert_document(id, doc(body)).unwrap();
    }
    store.commit().unwrap();

    for (prefix, expected) in [
        ("zqold", vec![2]),
        ("zqnew", vec![1, 4]),
        ("zq", vec![1, 2, 4]),
        ("o", vec![3]),
    ] {
        let query = || PrefixQuery::new("body", prefix);
        assert_eq!(search_ids(&store, Box::new(query())), expected, "{prefix}");
        // A filter and a count see the segments as one.
        let filtered = BooleanQueryBuilder::new()
            .must(Box::new(TermQuery::new("body", "every")))
            .filter(Box::new(query()))
            .build();
        assert_eq!(
            search_ids(&store, Box::new(filtered)),
            expected,
            "{prefix}, as a filter"
        );
        assert_eq!(
            store
                .count(LexicalSearchRequest::new(Box::new(query())))
                .unwrap(),
            expected.len() as u64,
            "{prefix}, counted"
        );
    }
}

/// Lists too long to be gathered with others, and more short ones than are
/// gathered at once, are all in the set (Issue #1740).
#[test]
fn long_lists_and_many_short_ones_are_all_in_the_set() {
    const DOCUMENTS: u64 = 1500;
    const WORDS: u64 = 100;
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage, LexicalIndexConfig::builder().build()).unwrap();
    for id in 1..=DOCUMENTS {
        // `zqlong` is in two documents of three. One document of four has
        // 200 of 800 short words, 75,000 postings in all, and one of a
        // hundred has the last short word.
        let mut body = String::from("every");
        if id % 3 != 0 {
            body.push_str(" zqlong");
        }
        if id % 4 == 0 {
            for word in 0..WORDS {
                body.push_str(&format!(" zqshort{:03}", (id + word * 3) % 400));
                body.push_str(&format!(" zqshort{:03}x", (id + word * 7) % 400));
            }
        }
        if id % 100 == 1 {
            body.push_str(" zqshortz");
        }
        store.upsert_document(id, doc(&body)).unwrap();
    }
    store.commit().unwrap();

    let ids = |keep: &dyn Fn(u64) -> bool| -> Vec<u64> {
        (1..=DOCUMENTS).filter(|id| keep(*id)).collect()
    };
    let short = |id: u64| id.is_multiple_of(4) || id % 100 == 1;
    for (prefix, expected) in [
        ("zqlong", ids(&|id| !id.is_multiple_of(3))),
        ("zqshort", ids(&short)),
        ("zq", ids(&|id| short(id) || !id.is_multiple_of(3))),
        ("every", ids(&|_| true)),
    ] {
        let query = Box::new(PrefixQuery::new("body", prefix));
        let mut found: Vec<u64> = store
            .search(LexicalSearchRequest::new(query).limit(DOCUMENTS as usize))
            .unwrap()
            .hits
            .iter()
            .map(|hit| hit.doc_id)
            .collect();
        found.sort_unstable();
        assert_eq!(found, expected, "{prefix}");
    }
}

/// The dictionary holds the terms of every field, sorted. Those of the fields
/// before and after the one searched do not match, whatever the pattern
/// (Issue #1740).
#[test]
fn the_terms_of_other_fields_do_not_match() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage, LexicalIndexConfig::builder().build()).unwrap();
    // Names of one length, so that their terms start at the same place in the
    // dictionary's keys.
    let fields = ["aody", "body", "cody"];
    for (id, field) in (1u64..).zip(fields) {
        for (word, offset) in [("zqone", 0), ("xx", 10)] {
            let document = Document::builder().add_text(field, word).build();
            store.upsert_document(id + offset, document).unwrap();
        }
    }
    store.commit().unwrap();

    for (id, field) in (1u64..).zip(fields) {
        for (what, query, expected) in [
            (
                "prefix",
                Box::new(PrefixQuery::new(field, "zq")) as Box<dyn Query>,
                vec![id],
            ),
            (
                "wildcard",
                Box::new(WildcardQuery::new(field, "z*e").unwrap()),
                vec![id],
            ),
            (
                "wildcard without a prefix",
                Box::new(WildcardQuery::new(field, "*").unwrap()),
                vec![id, id + 10],
            ),
            (
                "regexp without a prefix",
                Box::new(RegexpQuery::new(field, "q|x").unwrap()),
                vec![id, id + 10],
            ),
            (
                "fuzzy",
                Box::new(FuzzyQuery::new(field, "zqonf").max_edits(1)),
                vec![id],
            ),
        ] {
            assert_eq!(search_ids(&store, query), expected, "{field}, {what}");
        }
    }
}
