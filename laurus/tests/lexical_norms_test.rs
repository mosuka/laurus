//! Integration tests for Issue #555 (`.lens`/`.fstats` -> `.norms`
//! migration): search-visible behavior must be unaffected for ordinary
//! documents, and merging segments that mix long (quantised) and short
//! (exact) fields must not change hit counts or drift scores for the
//! documents whose lengths were never quantised in the first place.

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

fn doc_with_body(body: &str) -> Document {
    Document::builder().add_text("body", body).build()
}

fn search(store: &LexicalStore, term: &str) -> Vec<(u64, f32)> {
    let query = Box::new(TermQuery::new("body", term));
    let mut hits: Vec<(u64, f32)> = store
        .search(LexicalSearchRequest::new(query))
        .unwrap()
        .hits
        .into_iter()
        .map(|h| (h.doc_id, h.score))
        .collect();
    hits.sort_by_key(|(doc_id, _)| *doc_id);
    hits
}

/// A merge round-trips a long field through the quantised `.norms`
/// representation (write -> read back the decoded length -> write again),
/// but must never drop or duplicate a hit.
#[test]
fn merge_preserves_hit_counts_with_long_and_short_fields() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    // Segment 1: a short (exact, < 40 tokens) field.
    store
        .upsert_document(1, doc_with_body("widget alpha bravo"))
        .unwrap();
    store.commit().unwrap();

    // Segment 2: a long (quantised, > 40 tokens) field.
    let long_body = format!("widget {}", "filler ".repeat(5_000));
    store.upsert_document(2, doc_with_body(&long_body)).unwrap();
    store.commit().unwrap();

    // Segment 3: another long field, different length, to exercise
    // multiple quantised documents merging together.
    let another_long_body = format!("widget {}", "filler ".repeat(1_000));
    store
        .upsert_document(3, doc_with_body(&another_long_body))
        .unwrap();
    store.commit().unwrap();

    let before = search(&store, "widget");
    assert_eq!(
        before.len(),
        3,
        "all three documents must match before optimize"
    );

    store.optimize().unwrap();

    let after = search(&store, "widget");
    assert_eq!(
        after.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        before.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        "optimize must not drop or duplicate hits across the .norms merge"
    );

    // Idempotent: a second optimize on a single segment is a no-op.
    store.optimize().unwrap();
    let again = search(&store, "widget");
    assert_eq!(
        again, after,
        "re-optimizing a single segment must be a no-op"
    );
}

/// A merged segment's BM25 scores for documents whose field lengths never
/// left the exact (< 40 tokens) window must match what indexing the same
/// final document set into a single fresh segment would produce -- i.e.
/// the `.norms`-based merge reconstruction must not perturb documents it
/// never actually quantised.
///
/// This deliberately does *not* compare a document's score "before" and
/// "after" its own merge: BM25's per-segment `avg_length`/IDF are a
/// function of the whole segment's population, so merging two
/// single-document segments into one changes those statistics regardless
/// of `.norms` -- that is expected, pre-existing behavior, not something
/// this migration could introduce or fix.
#[test]
fn merged_short_fields_score_the_same_as_a_fresh_single_segment_build() {
    let via_merge = {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        store
            .upsert_document(1, doc_with_body("widget alpha bravo charlie"))
            .unwrap();
        store.commit().unwrap();
        store
            .upsert_document(2, doc_with_body("widget delta echo"))
            .unwrap();
        store.commit().unwrap();
        store.optimize().unwrap();

        search(&store, "widget")
    };

    let via_fresh_single_segment = {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        store
            .upsert_document(1, doc_with_body("widget alpha bravo charlie"))
            .unwrap();
        store
            .upsert_document(2, doc_with_body("widget delta echo"))
            .unwrap();
        store.commit().unwrap();

        search(&store, "widget")
    };

    assert_eq!(
        via_merge, via_fresh_single_segment,
        "a .norms-based merge of exact (unquantised) fields must reproduce \
         exactly what a fresh single-segment build of the same documents \
         would score"
    );
}
