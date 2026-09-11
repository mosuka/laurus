//! Index-backed end-to-end tests for Issue #1047's mixed-segment DocValues
//! bug — the sort and facet paths both cached `has_doc_values` per field
//! across the whole (possibly multi-segment) index, then treated any
//! `get_doc_value` miss under a `true` cache as "the value is `Null`" /
//! "no contribution" instead of falling back to the stored document. That
//! collapses correctly whenever *no* segment has the column (already
//! covered by #1053's stored-document fallback), but a segment that lacks
//! the column while a sibling segment has it was unreachable by any
//! existing test.
//!
//! These tests reproduce that exact situation on a real, on-disk,
//! multi-segment index (not a mock reader): two segments are committed
//! with `use_compound: false` so each segment's `.dv` file exists
//! standalone, then one segment's `.dv` file is deleted outright via
//! `Storage::delete_file`. `has_doc_values` still reports `true`
//! index-wide (the surviving segment has the column), but every document
//! in the deleted segment now misses on `get_doc_value` — exactly the
//! `has_dv == true, get_doc_value == Ok(None)` case `TopFieldCollector`
//! and `FacetCollector` must fall back on.

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::config::InvertedIndexConfig;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::query::Query;
use laurus::lexical::search::features::facet::{FacetCollector, FacetConfig};
use laurus::lexical::writer::LexicalIndexWriter;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

/// `.dv` files present in `storage`, sorted by name. Segment names are
/// `{prefix}_{:06}`, assigned in increasing order as segments are
/// flushed, so a lexicographic sort is also commit order.
fn dv_files_sorted(storage: &Arc<dyn Storage>) -> Vec<String> {
    let mut files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".dv"))
        .collect();
    files.sort();
    files
}

fn doc_with_score(score: i64) -> Document {
    Document::builder()
        .add_text("body", "alpha")
        .add_integer("score", score)
        .build()
}

fn doc_with_brand(brand: &str) -> Document {
    Document::builder().add_text("brand", brand).build()
}

/// Loose (non-compound) config so each segment's DocValues live in a
/// standalone `{segment}.dv` file that this test can delete directly. A
/// high `max_segments` keeps auto-merge from folding the two segments
/// back into one before the deletion takes effect.
fn loose_config() -> InvertedIndexConfig {
    InvertedIndexConfig {
        use_compound: false,
        max_segments: 1000,
        ..Default::default()
    }
}

#[test]
fn field_sort_falls_back_when_a_segments_dv_file_is_missing() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::Inverted(loose_config());
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    // Segment 0: doc 1 -> 30, doc 2 -> 10, doc 3 -> 20.
    for (doc_id, score) in [(1u64, 30i64), (2, 10), (3, 20)] {
        store
            .upsert_document(doc_id, doc_with_score(score))
            .unwrap();
    }
    store.commit().unwrap();

    // Segment 1: doc 4 -> 5, doc 5 -> 60, doc 6 -> 40.
    for (doc_id, score) in [(4u64, 5i64), (5, 60), (6, 40)] {
        store
            .upsert_document(doc_id, doc_with_score(score))
            .unwrap();
    }
    store.commit().unwrap();

    let dv_files = dv_files_sorted(&storage);
    assert_eq!(
        dv_files.len(),
        2,
        "expected one standalone .dv file per segment, found {dv_files:?}"
    );
    // Delete segment 0's DocValues column entirely -- its docs (1, 2, 3)
    // now have no "score" column even though segment 1 still does.
    storage.delete_file(&dv_files[0]).unwrap();

    let query: Box<dyn Query> = Box::new(TermQuery::new("body", "alpha"));
    let results = store
        .search(
            LexicalSearchRequest::new(query)
                .limit(6)
                .sort_by_field_asc("score"),
        )
        .unwrap();

    // True ascending order by score: 4(5), 2(10), 3(20), 1(30), 6(40), 5(60).
    // Before the #1047 fix, docs 1-3 would collapse to `Null` (sorts last
    // under both directions) instead of falling back to their stored
    // "score" value, producing a different order.
    assert_eq!(
        results.hits.iter().map(|h| h.doc_id).collect::<Vec<_>>(),
        vec![4, 2, 3, 1, 6, 5],
        "docs in the segment with the deleted .dv file must still sort by \
         their true stored score, not collapse to Null"
    );
    assert_eq!(results.total_hits, 6);
}

#[test]
fn facet_falls_back_when_a_segments_dv_file_is_missing() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = InvertedIndex::create(storage.clone(), loose_config()).unwrap();
    let mut writer: Box<dyn LexicalIndexWriter> = index.writer().unwrap();

    // Segment 0: apple, apple, dell.
    let mut doc_ids = Vec::new();
    for brand in ["apple", "apple", "dell"] {
        doc_ids.push(writer.add_document(doc_with_brand(brand)).unwrap());
    }
    writer.commit().unwrap();

    // Segment 1: apple, dell, dell.
    for brand in ["apple", "dell", "dell"] {
        doc_ids.push(writer.add_document(doc_with_brand(brand)).unwrap());
    }
    writer.commit().unwrap();

    let dv_files = dv_files_sorted(&storage);
    assert_eq!(
        dv_files.len(),
        2,
        "expected one standalone .dv file per segment, found {dv_files:?}"
    );
    // Delete segment 0's DocValues column entirely.
    storage.delete_file(&dv_files[0]).unwrap();

    let reader = writer.build_reader().unwrap();
    assert!(
        reader.has_doc_values("brand"),
        "segment 1 still has the brand column, so this stays true index-wide"
    );

    let mut collector = FacetCollector::new(FacetConfig::default(), vec!["brand".to_string()]);
    for doc_id in &doc_ids {
        collector.collect_doc(*doc_id, reader.as_ref()).unwrap();
    }
    let results = collector.finalize().unwrap();

    let counts: std::collections::HashMap<String, u64> = results
        .get_field_facets("brand")
        .expect("brand facets must be present")
        .iter()
        .map(|c| (c.path.path[0].clone(), c.count))
        .collect();

    // True counts across both segments: apple = 3 (2 from segment 0 + 1
    // from segment 1), dell = 3 (1 from segment 0 + 2 from segment 1).
    // Before the #1047 fix, segment 0's docs would silently contribute
    // nothing (has_dv == true but get_doc_value misses, and the old code
    // never fell back), undercounting both to apple = 1, dell = 2.
    assert_eq!(counts.get("apple").copied(), Some(3));
    assert_eq!(counts.get("dell").copied(), Some(3));
}
