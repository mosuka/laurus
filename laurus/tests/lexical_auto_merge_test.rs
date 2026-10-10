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
//!
//! `max_merged_segment_bytes` (Issue #1394) limits which segments one
//! auto-merge takes; its cap and convergence are pinned here as well.

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

fn segment_ids(storage: &Arc<dyn Storage>) -> Vec<String> {
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
    payload["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["segment_id"].as_str().unwrap().to_string())
        .collect()
}

/// Sum the on-disk size of every file belonging to `segment_id`, mirroring
/// `InvertedIndex::segment_size_bytes` (Issue #1394).
fn segment_size_bytes(storage: &Arc<dyn Storage>, segment_id: &str) -> u64 {
    let prefix = format!("{segment_id}.");
    storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.starts_with(&prefix))
        .map(|f| storage.metadata(f).unwrap().size)
        .sum()
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

/// An auto-merge of segments whose documents record no field lengths must
/// not fail the commit that triggers it (Issue #1213). The merge reads every
/// source segment's `.norms`, which the reader used to reject as corrupted
/// past 21 documents — failing `commit()` although its data was persisted.
#[test]
fn auto_merge_of_field_less_segments_does_not_fail_the_commit() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .max_segments(1)
        .merge_factor(2)
        .build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    for batch in [1..=30u64, 31..=60] {
        for id in batch {
            // `Bytes` fields are never indexed, so no field length is recorded.
            let document = Document::builder()
                .add_bytes("blob", id.to_le_bytes().to_vec())
                .build();
            store.upsert_document(id, document).unwrap();
        }
        store.commit().unwrap();
    }

    assert_eq!(segment_count(&storage), 1, "the second commit merged");
    assert_eq!(store.stats().unwrap().doc_count, 60);
}

/// A one-document segment's on-disk size, used to scale
/// `max_merged_segment_bytes` to a cap independent of the on-disk encoding.
fn one_doc_segment_bytes() -> u64 {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder().max_segments(1000).build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    let id = segment_ids(&storage).into_iter().next().unwrap();
    segment_size_bytes(&storage, &id)
}

const DOCS_PER_BATCH: u64 = 20;

/// A document whose 20 body terms are unique to `(batch, i)`, so a segment's
/// on-disk size grows with its document count. (A one-document segment of
/// [`doc`] is mostly fixed per-segment overhead: ten of them merged are still
/// under three times its size.)
fn batch_doc(batch: u64, i: u64) -> Document {
    let body: Vec<String> = (0..20).map(|w| format!("b{batch}d{i}w{w}")).collect();
    Document::builder().add_text("body", body.join(" ")).build()
}

/// Upsert batch `batch` of [`DOCS_PER_BATCH`] documents and commit it as one
/// segment.
fn commit_batch(store: &LexicalStore, batch: u64) {
    for i in 0..DOCS_PER_BATCH {
        let id = batch * DOCS_PER_BATCH + i + 1;
        store.upsert_document(id, batch_doc(batch, i)).unwrap();
    }
    store.commit().unwrap();
}

/// The on-disk size of the largest committed segment.
fn largest_segment_bytes(storage: &Arc<dyn Storage>) -> u64 {
    segment_ids(storage)
        .iter()
        .map(|id| segment_size_bytes(storage, id))
        .max()
        .unwrap_or(0)
}

/// Commit `batches` batches into a store whose auto-merge fires on every
/// commit with no `merge_factor` limit, and return the store together with
/// the largest segment seen after any commit.
fn largest_segment_over_batches(batches: u64, cap: u64) -> (LexicalStore, u64) {
    let config = LexicalIndexConfig::builder()
        .max_segments(1)
        .merge_factor(100)
        .max_merged_segment_bytes(cap)
        .build();
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), config).unwrap();
    let mut largest = 0;
    for batch in 0..batches {
        commit_batch(&store, batch);
        largest = largest.max(largest_segment_bytes(&storage));
    }
    (store, largest)
}

/// Issue #1394: `max_merged_segment_bytes` bounds the combined on-disk size
/// auto-merge will take, even though `merge_factor` is high enough to pull in
/// every segment every time.
#[test]
fn auto_merge_never_exceeds_the_merged_segment_cap() {
    const BATCHES: u64 = 8;

    let probe_storage: Arc<dyn Storage> =
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let probe_config = LexicalIndexConfig::builder().max_segments(1000).build();
    let probe = LexicalStore::new(probe_storage.clone(), probe_config).unwrap();
    commit_batch(&probe, 0);
    // Room for two batches merged together, but not three.
    let cap = largest_segment_bytes(&probe_storage) * 5 / 2;

    let (_, uncapped_largest) = largest_segment_over_batches(BATCHES, u64::MAX);
    assert!(
        uncapped_largest > cap,
        "sanity: without a cap, auto-merge must grow a segment past {cap} bytes \
         (largest was {uncapped_largest}) for this test to mean anything"
    );

    let (store, capped_largest) = largest_segment_over_batches(BATCHES, cap);
    assert!(
        capped_largest <= cap,
        "a segment reached {capped_largest} bytes, over the cap of {cap}"
    );
    assert_eq!(store.stats().unwrap().doc_count, BATCHES * DOCS_PER_BATCH);
    assert_eq!(hits(&store, "body", "b5d3w7"), 1);
}

/// Issue #1394: a segment at or over the cap is left out of every future
/// auto-merge -- it is never rewritten, and the smaller segments around it
/// keep converging on their own instead of looping on the same selection.
#[test]
fn auto_merge_leaves_segments_at_the_cap_alone() {
    let cap = one_doc_segment_bytes() * 5;
    let config = LexicalIndexConfig::builder()
        .max_segments(1)
        .merge_factor(100)
        .max_merged_segment_bytes(cap)
        .build();
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    // One commit of many documents makes one oversized segment (no flush
    // threshold is crossed mid-commit, so it never splits), comfortably over
    // the cap above.
    for id in 1..=30u64 {
        store
            .upsert_document(id, doc(&format!("big-{id}")))
            .unwrap();
    }
    store.commit().unwrap();
    let big_ids = segment_ids(&storage);
    assert_eq!(big_ids.len(), 1, "one commit => one segment");
    let big_id = big_ids.into_iter().next().unwrap();
    let big_size = segment_size_bytes(&storage, &big_id);
    assert!(
        big_size > cap,
        "the big segment ({big_size} bytes) must exceed the cap ({cap}) for this test to be meaningful"
    );

    // Repeated single-document commits: each one is small enough to merge
    // with its siblings, but the big segment above must never be touched.
    for id in 31..=40u64 {
        store
            .upsert_document(id, doc(&format!("small-{id}")))
            .unwrap();
        store.commit().unwrap();

        assert!(
            segment_ids(&storage).contains(&big_id),
            "the oversized segment must survive every commit unmerged"
        );
        assert_eq!(
            segment_size_bytes(&storage, &big_id),
            big_size,
            "the oversized segment's content must never be rewritten"
        );
    }

    // The small segments still converged among themselves instead of
    // accumulating one per commit (10 small commits, far fewer segments).
    assert!(
        segment_count(&storage) < 10,
        "segments below the cap must still merge together: got {} segments",
        segment_count(&storage)
    );
    assert_eq!(store.stats().unwrap().doc_count, 40);
}
