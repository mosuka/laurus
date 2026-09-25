//! Integration test for the post-commit auto-merge hook (Issue #755).
//!
//! After each commit, `LexicalStore` invokes `maybe_merge`, which merges the
//! smallest `merge_factor` segments once the segment count exceeds
//! `max_segments`. This keeps the segment count bounded without a manual
//! `optimize()`, and is a no-op below the threshold.
//!
//! The writer's flush thresholds (`max_buffered_docs` / `max_buffer_memory`,
//! Issue #1200) decide how many segments one commit publishes, so they are
//! pinned here too, together with the guarantee that a merge ignores them.

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

fn doc(title: &str) -> Document {
    Document::builder()
        .add_text("title", title)
        .add_text("body", "lorem ipsum")
        .build()
}

fn segment_count(storage: &Arc<dyn Storage>) -> usize {
    // Count via the manifest (#1024): `.meta` files are gone, and
    // `segments.json` is the sole record of the committed segment set.
    let mut input = storage.open_input("segments.json").unwrap();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut input, &mut bytes).unwrap();
    let payload: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => {
            let mut len: u64 = 0;
            let mut shift = 0;
            let mut cursor = 0usize;
            loop {
                let byte = bytes[cursor];
                cursor += 1;
                len |= u64::from(byte & 0x7F) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            serde_json::from_slice(&bytes[cursor..cursor + len as usize]).unwrap()
        }
    };
    payload["segments"].as_array().unwrap().len()
}

fn hits(store: &LexicalStore, field: &str, term: &str) -> usize {
    let query = Box::new(TermQuery::new(field, term));
    store
        .search(LexicalSearchRequest::new(query))
        .unwrap()
        .hits
        .len()
}

/// With a low `max_segments`, repeated commits stay bounded: each commit adds a
/// segment, and `maybe_merge` compacts the smallest ones back down once the
/// threshold is crossed — without any manual `optimize()`.
#[test]
fn auto_merge_keeps_segment_count_bounded() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .max_segments(2)
        .merge_factor(2)
        .build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    // One doc per commit => one segment per commit, but auto-merge keeps the
    // count from growing past the threshold.
    let titles = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"];
    for (i, title) in titles.iter().enumerate() {
        store.upsert_document((i + 1) as u64, doc(title)).unwrap();
        store.commit().unwrap();
        assert!(
            segment_count(&storage) <= 2,
            "after commit {}: segment count {} must stay <= max_segments (2)",
            i + 1,
            segment_count(&storage),
        );
    }

    // Steady state: exactly `max_segments` segments after enough commits.
    assert_eq!(segment_count(&storage), 2);
    // Every document is still searchable, and a per-doc term survives.
    assert_eq!(hits(&store, "body", "lorem"), titles.len());
    assert_eq!(hits(&store, "title", "echo"), 1);
}

/// A high `max_segments` disables auto-merge: commits accumulate segments (the
/// `maybe_merge` no-op path), so users can opt out by raising the threshold.
#[test]
fn auto_merge_noop_above_threshold() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder().max_segments(1000).build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    for i in 1..=4u64 {
        store.upsert_document(i, doc("doc")).unwrap();
        store.commit().unwrap();
    }

    assert_eq!(
        segment_count(&storage),
        4,
        "no merge below threshold => one segment per commit"
    );
    assert_eq!(hits(&store, "body", "lorem"), 4);
}

const TITLES: [&str; 5] = ["alpha", "bravo", "charlie", "delta", "echo"];

/// Upsert one document per entry of [`TITLES`] into a store built from
/// `config`, then commit once.
fn store_with_one_commit(config: LexicalIndexConfig) -> (Arc<dyn Storage>, LexicalStore) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), config).unwrap();
    for (i, title) in TITLES.iter().enumerate() {
        store.upsert_document((i + 1) as u64, doc(title)).unwrap();
    }
    store.commit().unwrap();
    (storage, store)
}

/// `max_buffered_docs` set through the builder reaches the writer (Issue
/// #1200): five documents under a threshold of 2 flush after the second and
/// the fourth, and `commit()` flushes the fifth, so one commit publishes
/// three segments. Ignored, the 10,000-document default publishes one.
#[test]
fn max_buffered_docs_splits_one_commit_into_segments() {
    let config = LexicalIndexConfig::builder()
        .max_buffered_docs(2)
        .max_segments(1000)
        .build();
    let (storage, store) = store_with_one_commit(config);

    assert_eq!(segment_count(&storage), 3, "ceil(5 / 2) segments");
    assert_eq!(hits(&store, "body", "lorem"), TITLES.len());
    assert_eq!(hits(&store, "title", "echo"), 1);
}

/// `max_buffer_memory` set through the builder reaches the writer (Issue
/// #1200): every document exceeds a one-byte budget and flushes on its own,
/// leaving `commit()` nothing to flush — five segments, all searchable.
#[test]
fn max_buffer_memory_splits_one_commit_into_segments() {
    let config = LexicalIndexConfig::builder()
        .max_buffer_memory(1)
        .max_segments(1000)
        .build();
    let (storage, store) = store_with_one_commit(config);

    assert_eq!(
        segment_count(&storage),
        TITLES.len(),
        "one segment per document"
    );
    assert_eq!(hits(&store, "body", "lorem"), TITLES.len());
    assert_eq!(hits(&store, "title", "charlie"), 1);
}

/// A merge is not bound by the flush thresholds (Issue #1200): its writer is
/// unbounded, so `optimize()` still produces a single segment holding every
/// document. Were the threshold inherited, the merge would flush part of its
/// replay into unregistered files, and the #1166 check that the merged
/// segment holds every document it claims would fail the merge.
#[test]
fn optimize_ignores_the_flush_thresholds() {
    let config = LexicalIndexConfig::builder()
        .max_buffered_docs(2)
        .max_segments(1000)
        .build();
    let (storage, store) = store_with_one_commit(config);
    assert_eq!(segment_count(&storage), 3);

    store.optimize().unwrap();

    assert_eq!(
        segment_count(&storage),
        1,
        "optimize merges into one segment"
    );
    assert_eq!(hits(&store, "body", "lorem"), TITLES.len());
    assert_eq!(hits(&store, "title", "bravo"), 1);
}

/// `LexicalStore::stats` counts documents an automatic flush has already
/// written to an unpublished segment (Issue #1204): before the commit, five
/// upserts under a threshold of 2 are all counted, not just the one still
/// buffered.
#[test]
fn stats_count_documents_flushed_before_commit() {
    let config = LexicalIndexConfig::builder()
        .max_buffered_docs(2)
        .max_segments(1000)
        .build();
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), config).unwrap();
    for (i, title) in TITLES.iter().enumerate() {
        store.upsert_document((i + 1) as u64, doc(title)).unwrap();
    }

    assert_eq!(store.stats().unwrap().doc_count, TITLES.len() as u64);

    store.commit().unwrap();
    assert_eq!(store.stats().unwrap().doc_count, TITLES.len() as u64);
}
