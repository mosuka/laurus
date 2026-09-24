//! Term and phrase queries against a segment that has **no inverted index**
//! (Issue #1194): `SegmentReader::postings` falls back to scanning the
//! stored documents when the segment's `.post` file is missing, and that
//! scan must see multi-valued values the way the index would have.
//!
//! The writer always emits a `.post` part, so the situation is reproduced
//! the way `lexical_doc_values_mixed_segments_test.rs` reproduces a missing
//! DocValues column: two loose (non-compound) segments, then segment 0's
//! `.post` **and** `.dict` are deleted outright. Two consequences shape the
//! assertions below:
//!
//! - `TermQuery::is_empty` and `count`'s `doc_freq` fast path consult the
//!   term dictionaries, which the index-less segment is absent from. Since
//!   #1196 the reader reports `term_info_is_authoritative() == false` in
//!   that state, so both fall through to the matcher and the scanned
//!   documents are found and counted like any other.
//! - A scan hit has no `term_info`, so it scores 0.0. It is collected as
//!   long as the top-k heap is not full, hence the generous `limit`.

use std::sync::Arc;

use laurus::Document;
use laurus::analysis::analyzer::keyword::KeywordAnalyzer;
use laurus::lexical::index::config::InvertedIndexConfig;
use laurus::lexical::query::{PhraseQuery, TermQuery};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

const MULTI_VALUED_DOC: u64 = 1;
const SCALAR_DOC: u64 = 2;

/// Loose (non-compound) config so each segment's parts are standalone
/// files this test can delete; a high `max_segments` keeps auto-merge from
/// folding the two segments together.
fn loose_config() -> LexicalIndexConfig {
    LexicalIndexConfig::Inverted(InvertedIndexConfig {
        use_compound: false,
        max_segments: 1000,
        ..Default::default()
    })
}

/// Files in `storage` ending with `suffix`, sorted by name (= commit order,
/// since segment names carry an increasing zero-padded counter).
fn files_with_suffix(storage: &Arc<dyn Storage>, suffix: &str) -> Vec<String> {
    let mut files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(suffix))
        .collect();
    files.sort();
    files
}

/// Segment 0: the multi-valued document `body = ["hello world", "rust search"]`
/// (its `.post` and `.dict` are deleted afterwards). Segment 1: the scalar
/// document `body = "rust"`, whose dictionary keeps `TermQuery("body", "rust")`
/// non-empty.
fn store_with_an_index_less_first_segment() -> (Arc<dyn Storage>, LexicalStore) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), loose_config()).unwrap();

    store
        .upsert_document(
            MULTI_VALUED_DOC,
            Document::builder()
                .add_text_array(
                    "body",
                    vec!["hello world".to_string(), "rust search".to_string()],
                )
                .build(),
        )
        .unwrap();
    store.commit().unwrap();

    store
        .upsert_document(
            SCALAR_DOC,
            Document::builder().add_text("body", "rust").build(),
        )
        .unwrap();
    store.commit().unwrap();

    let post_files = files_with_suffix(&storage, ".post");
    let dict_files = files_with_suffix(&storage, ".dict");
    assert_eq!(
        post_files.len(),
        2,
        "one .post per segment, found {post_files:?}"
    );
    assert_eq!(
        dict_files.len(),
        2,
        "one .dict per segment, found {dict_files:?}"
    );
    storage.delete_file(&post_files[0]).unwrap();
    storage.delete_file(&dict_files[0]).unwrap();
    assert!(!storage.file_exists(&post_files[0]));
    assert!(!storage.file_exists(&dict_files[0]));

    (storage, store)
}

fn hit_ids(store: &LexicalStore, request: LexicalSearchRequest) -> Vec<u64> {
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

fn term(t: &str) -> Box<TermQuery> {
    Box::new(TermQuery::new("body", t))
}

fn phrase(terms: &[&str], slop: u32) -> Box<PhraseQuery> {
    Box::new(
        PhraseQuery::new("body", terms.iter().map(|s| (*s).to_string()).collect()).with_slop(slop),
    )
}

#[test]
fn term_query_reaches_a_multi_valued_document_through_the_scan_fallback() {
    let (_storage, store) = store_with_an_index_less_first_segment();

    // The multi-valued document comes from segment 0's scan, the scalar one
    // from segment 1's postings. Before #1194 the scan skipped `TextArray`
    // and only the scalar document was found.
    assert_eq!(
        hit_ids(&store, LexicalSearchRequest::new(term("rust")).limit(100)),
        vec![MULTI_VALUED_DOC, SCALAR_DOC]
    );
    let ids = store.matching_doc_ids(term("rust")).unwrap();
    assert!(ids.contains(MULTI_VALUED_DOC) && ids.contains(SCALAR_DOC));

    // `count` would take the #610 `doc_freq` fast path, which the
    // index-less segment cannot contribute to; the reader is not
    // authoritative here (#1196), so `count` walks the matcher and agrees
    // with `search`.
    assert_eq!(
        store
            .count(LexicalSearchRequest::new(term("rust")).limit(100))
            .unwrap(),
        2
    );
}

#[test]
fn a_term_known_only_to_the_index_less_segment_is_found() {
    let (_storage, store) = store_with_an_index_less_first_segment();

    // `search` short-circuits on `TermQuery::is_empty`, which no dictionary
    // can answer for this term. With a segment lacking its dictionary the
    // reader is not authoritative (#1196), so the query is not considered
    // empty and the scan finds the document.
    assert_eq!(
        hit_ids(&store, LexicalSearchRequest::new(term("search")).limit(100)),
        vec![MULTI_VALUED_DOC]
    );
    assert_eq!(
        store
            .count(LexicalSearchRequest::new(term("search")).limit(100))
            .unwrap(),
        1
    );
    // `matching_doc_ids` runs the matcher directly and reaches the scan too.
    let ids = store.matching_doc_ids(term("search")).unwrap();
    assert!(ids.contains(MULTI_VALUED_DOC));
    assert!(!ids.contains(SCALAR_DOC));
}

#[test]
fn phrase_query_honours_the_position_gap_through_the_scan_fallback() {
    let (_storage, store) = store_with_an_index_less_first_segment();

    // `PhraseQuery::is_empty` never consults the reader, so every phrase
    // reaches the scan. Within one element the phrase matches ...
    assert_eq!(
        hit_ids(
            &store,
            LexicalSearchRequest::new(phrase(&["rust", "search"], 0)).limit(100)
        ),
        vec![MULTI_VALUED_DOC]
    );
    // ... across the element boundary it does not at slop 0, because the
    // scan numbers the second element `position_increment_gap` (100) past
    // the first, exactly like the writer, and matches once the slop
    // reaches the gap.
    assert!(
        hit_ids(
            &store,
            LexicalSearchRequest::new(phrase(&["world", "rust"], 0)).limit(100)
        )
        .is_empty()
    );
    assert!(
        hit_ids(
            &store,
            LexicalSearchRequest::new(phrase(&["world", "rust"], 99)).limit(100)
        )
        .is_empty()
    );
    assert_eq!(
        hit_ids(
            &store,
            LexicalSearchRequest::new(phrase(&["world", "rust"], 100)).limit(100)
        ),
        vec![MULTI_VALUED_DOC]
    );
}

// ---- The index analyzer on the scan path, Issue #1196 ----------------------

/// The two-segment shape above, on an index whose analyzer is
/// `KeywordAnalyzer`: a whole value is one term, so `"rust search"` is a
/// term and `"rust"` is not. Segment 0 holds `body = ["rust search", "foo"]`
/// and loses its `.post` (and, when `strip_dict` is set, its `.dict`);
/// segment 1 holds the scalar `body = "rust search"`.
fn keyword_store(strip_dict: bool) -> (Arc<dyn Storage>, LexicalStore) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::Inverted(InvertedIndexConfig {
        analyzer: Arc::new(KeywordAnalyzer::new()),
        use_compound: false,
        max_segments: 1000,
        ..Default::default()
    });
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    store
        .upsert_document(
            MULTI_VALUED_DOC,
            Document::builder()
                .add_text_array("body", vec!["rust search".to_string(), "foo".to_string()])
                .build(),
        )
        .unwrap();
    store.commit().unwrap();
    store
        .upsert_document(
            SCALAR_DOC,
            Document::builder().add_text("body", "rust search").build(),
        )
        .unwrap();
    store.commit().unwrap();

    let post_files = files_with_suffix(&storage, ".post");
    assert_eq!(
        post_files.len(),
        2,
        "one .post per segment, found {post_files:?}"
    );
    storage.delete_file(&post_files[0]).unwrap();
    if strip_dict {
        let dict_files = files_with_suffix(&storage, ".dict");
        assert_eq!(
            dict_files.len(),
            2,
            "one .dict per segment, found {dict_files:?}"
        );
        storage.delete_file(&dict_files[0]).unwrap();
    }
    (storage, store)
}

#[test]
fn scan_fallback_uses_the_index_analyzer_end_to_end() {
    let (_storage, store) = keyword_store(true);

    // Under the keyword analyzer both documents carry the single term
    // `rust search` — the index-less one through the scan, which must
    // analyze with the index analyzer `InvertedIndexReader::new` hands the
    // segment readers ...
    assert_eq!(
        hit_ids(
            &store,
            LexicalSearchRequest::new(term("rust search")).limit(100)
        ),
        vec![MULTI_VALUED_DOC, SCALAR_DOC]
    );
    assert_eq!(
        hit_ids(&store, LexicalSearchRequest::new(term("foo")).limit(100)),
        vec![MULTI_VALUED_DOC]
    );
    // ... and neither carries `rust`. A scan analyzing with
    // `StandardAnalyzer` (the pre-#1196 behaviour) would have found the
    // multi-valued document here.
    assert!(hit_ids(&store, LexicalSearchRequest::new(term("rust")).limit(100)).is_empty());
}

#[test]
fn merge_replays_a_post_less_segment_with_the_index_analyzer() {
    let (storage, store) = keyword_store(false);

    // Segment 0 kept its `.dict`, so the merge replay enumerates
    // `rust search` and `foo` and re-reads each through `postings` — the
    // scan, driven by `MergeConfig::index_analyzer`. Force-merge both
    // segments into one.
    store.optimize().unwrap();
    assert_eq!(
        files_with_suffix(&storage, ".post").len(),
        1,
        "the force-merge must leave one segment with a real postings part"
    );

    // Both documents now come from the merged postings ...
    assert_eq!(
        hit_ids(
            &store,
            LexicalSearchRequest::new(term("rust search")).limit(100)
        ),
        vec![MULTI_VALUED_DOC, SCALAR_DOC]
    );
    // ... and document 0's own term survived the replay. A replay scanning
    // with `StandardAnalyzer` would have produced `rust` / `search`, matched
    // nothing for either dictionary key, and dropped both terms.
    assert_eq!(
        hit_ids(&store, LexicalSearchRequest::new(term("foo")).limit(100)),
        vec![MULTI_VALUED_DOC]
    );
}
