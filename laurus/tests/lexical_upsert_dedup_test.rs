//! Integration tests for `InvertedIndexWriter::remove_pending_document`
//! (Issue #570 — replace the per-upsert full-buffer scan with an O(1)
//! membership probe).
//!
//! `Engine::add_document` routes through `upsert_document(doc_id, ..)`, which
//! calls `remove_pending_document(doc_id)` before indexing. Previously that ran
//! `buffered_docs.retain(..)` — a full O(N) scan — on every call, even for
//! freshly assigned doc IDs that can never already be buffered, making an
//! `add × N` ingest O(N²). The fix keeps a `buffered_doc_ids` set in sync with
//! `buffered_docs` so the new-id case returns in O(1).
//!
//! These tests pin down both branches of the new code path and assert that the
//! externally observable behaviour is unchanged:
//! - new, never-seen doc IDs accumulate in the buffer (the O(1) early return);
//! - re-upserting an already-buffered doc ID still dedups in place and the
//!   in-memory index reflects only the latest version (the rebuild branch).

use std::sync::Arc;

use laurus::Document;
use laurus::lexical::{InvertedIndexWriter, InvertedIndexWriterConfig};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

/// Build a writer whose buffer never auto-flushes during a test, so
/// `buffered_docs` (and the new `buffered_doc_ids` set) stay populated and the
/// `remove_pending_document` paths are exercised directly.
fn writer() -> InvertedIndexWriter {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = InvertedIndexWriterConfig {
        // Far above the doc counts used here so no `flush_segment` clears the
        // buffer mid-test.
        max_buffered_docs: 1_000_000,
        ..Default::default()
    };
    InvertedIndexWriter::new(storage, config).unwrap()
}

fn doc(term: &str) -> Document {
    Document::builder().add_text("title", term).build()
}

/// New, distinct doc IDs take the O(1) early-return path: every upsert is
/// buffered, none is dropped, and `docs_added` counts each one exactly once.
#[test]
fn distinct_upserts_all_buffered() {
    let mut w = writer();
    let n = 500u64;

    for i in 0..n {
        w.upsert_document(i, doc(&format!("term{i}"))).unwrap();
    }

    assert_eq!(
        w.pending_docs(),
        n as usize,
        "every distinct upsert must remain buffered"
    );
    assert_eq!(
        w.stats().docs_added,
        n,
        "docs_added must equal the number of distinct upserts"
    );

    // A sampling of the ids is resolvable through the in-memory index.
    for i in [0u64, 1, 250, n - 1] {
        assert_eq!(
            w.find_doc_id_by_term("title", &format!("term{i}")).unwrap(),
            Some(i),
            "term{i} must resolve to its buffered doc id"
        );
    }
}

/// Re-upserting an already-buffered doc ID takes the rebuild branch: the buffer
/// does not grow, `docs_added` stays at one, and the in-memory index reflects
/// only the latest version (the old term is gone, the new term resolves).
#[test]
fn reupsert_same_id_dedups_and_reflects_latest() {
    let mut w = writer();

    w.upsert_document(7, doc("alpha")).unwrap();
    assert_eq!(w.pending_docs(), 1);
    assert_eq!(w.find_doc_id_by_term("title", "alpha").unwrap(), Some(7));

    // Re-upsert the SAME internal id with different content.
    w.upsert_document(7, doc("beta")).unwrap();

    assert_eq!(
        w.pending_docs(),
        1,
        "re-upserting the same id must not grow the buffer"
    );
    assert_eq!(
        w.stats().docs_added,
        1,
        "docs_added must net to one after an add + re-upsert of the same id"
    );
    assert_eq!(
        w.find_doc_id_by_term("title", "beta").unwrap(),
        Some(7),
        "the latest version's term must resolve"
    );
    assert_eq!(
        w.find_doc_id_by_term("title", "alpha").unwrap(),
        None,
        "the replaced version's term must be gone from the in-memory index"
    );
}

/// Interleaving fresh ids with a re-upsert keeps the buffer count and the
/// membership set consistent: the rebuild branch must not disturb the other
/// buffered docs, and a later fresh id still appends.
#[test]
fn interleaved_new_and_reupsert_stay_consistent() {
    let mut w = writer();

    w.upsert_document(1, doc("one")).unwrap();
    w.upsert_document(2, doc("two")).unwrap();
    w.upsert_document(3, doc("three")).unwrap();
    assert_eq!(w.pending_docs(), 3);

    // Re-upsert an existing id (rebuild branch) — count stays 3.
    w.upsert_document(2, doc("twotwo")).unwrap();
    assert_eq!(w.pending_docs(), 3);

    // The untouched neighbours survive the rebuild.
    assert_eq!(w.find_doc_id_by_term("title", "one").unwrap(), Some(1));
    assert_eq!(w.find_doc_id_by_term("title", "three").unwrap(), Some(3));
    // The re-upserted doc reflects its new term, not the old one.
    assert_eq!(w.find_doc_id_by_term("title", "twotwo").unwrap(), Some(2));
    assert_eq!(w.find_doc_id_by_term("title", "two").unwrap(), None);

    // A subsequent fresh id still appends via the O(1) path.
    w.upsert_document(4, doc("four")).unwrap();
    assert_eq!(w.pending_docs(), 4);
    assert_eq!(w.find_doc_id_by_term("title", "four").unwrap(), Some(4));
}

/// #828: removing a buffered doc defers the in-memory index rebuild, so the
/// removed doc's postings linger in `inverted_index` until flush. The NRT
/// lookup must still exclude them (filtered by the live buffer set) while
/// leaving untouched neighbours resolvable. This is the production
/// `delete_document(old_id)` path where the new version gets a *different* id.
#[test]
fn nrt_find_excludes_deferred_removed_doc() {
    let mut w = writer();

    w.upsert_document(1, doc("alpha")).unwrap();
    w.upsert_document(2, doc("beta")).unwrap();
    assert_eq!(w.pending_docs(), 2);

    // Delete a still-buffered doc by a DIFFERENT id than any survivor — this
    // takes the deferred path (no eager rebuild).
    w.delete_document(1).unwrap();

    assert_eq!(w.pending_docs(), 1, "the removed doc must leave the buffer");
    assert_eq!(
        w.find_doc_id_by_term("title", "alpha").unwrap(),
        None,
        "the deferred-removed doc's term must not resolve via NRT lookup"
    );
    assert_eq!(
        w.find_doc_id_by_term("title", "beta").unwrap(),
        Some(2),
        "the surviving doc must still resolve while the index is dirty"
    );
}

/// #828: a removal records the id as stale and defers purging its postings;
/// `rollback` must clear the buffer AND that deferred state so a later batch
/// starts clean and is not corrupted by it.
#[test]
fn rollback_clears_deferred_removed_state() {
    let mut w = writer();

    w.upsert_document(1, doc("alpha")).unwrap();
    w.delete_document(1).unwrap(); // records 1 as stale, defers the purge
    assert_eq!(w.pending_docs(), 0);

    w.rollback().unwrap();
    assert_eq!(w.pending_docs(), 0);

    // A fresh batch after rollback resolves normally and is unaffected by the
    // earlier deferred removal (no stale "alpha", clean new state).
    w.upsert_document(5, doc("gamma")).unwrap();
    assert_eq!(w.pending_docs(), 1);
    assert_eq!(w.find_doc_id_by_term("title", "gamma").unwrap(), Some(5));
    assert_eq!(w.find_doc_id_by_term("title", "alpha").unwrap(), None);
}
