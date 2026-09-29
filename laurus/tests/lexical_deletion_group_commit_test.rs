//! Integration tests for #875 — deferred (group-commit) deletion persistence.
//!
//! The delete-first path of an existing-id upsert used to persist deletion
//! state synchronously per delete: a `deletions.log` append plus a full
//! `.delmap` rewrite (each fsync'd on close) and a `.meta` JSON parse per
//! delete. #875 defers all of it to the writer's commit — mutations only
//! update in-memory bitmaps, and `flush_deletions` group-commits them once —
//! with crash safety provided by the engine WAL (the delete record precedes
//! every index mutation and replay is idempotent).
//!
//! These tests pin the externally observable contract:
//! - deletion state reaches storage exactly at commit (not per delete);
//! - within a commit, `.delmap` files land BEFORE the `metadata.json`
//!   `last_wal_seq` checkpoint (a mid-commit crash keeps the WAL delete
//!   records replayable) and no `deletions.log` is ever written;
//! - `optimize()` flushes buffered deletions before its force-merge, so they
//!   are never resurrected into the merged segment;
//! - `rollback()` keeps buffered deletions (they were never rollback-able);
//! - WAL recovery auto-commits, so replayed deletions are searchable on a
//!   freshly reopened engine without a manual commit.

use std::sync::{Arc, Mutex};

use laurus::Document;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{FileMetadata, Storage, StorageInput, StorageOutput};

/// The manifest as lossy text (it is framed: varint + JSON + CRC, so a
/// strict UTF-8 read fails on the trailer).
fn read_manifest_text(storage: &Arc<dyn Storage>) -> String {
    let mut input = storage.open_input("segments.json").unwrap();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut input, &mut bytes).unwrap();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn doc(title: &str) -> Document {
    Document::builder()
        .add_text("title", title)
        .add_text("body", "lorem ipsum")
        .build()
}

fn hits(store: &LexicalStore, term: &str) -> usize {
    let query = Box::new(TermQuery::new("title", term));
    store
        .search(LexicalSearchRequest::new(query))
        .unwrap()
        .hits
        .len()
}

fn delmap_files(storage: &Arc<dyn Storage>) -> Vec<String> {
    storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".delmap"))
        .collect()
}

/// Deletion state must reach storage exactly at commit: an existing-id upsert
/// buffers the deletion in memory (no `.delmap`, no `.meta` flip), and the
/// following commit persists both — after which the old version stays
/// invisible to search.
#[test]
fn deletions_persist_only_at_commit() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    assert_eq!(hits(&store, "alpha"), 1);
    assert!(delmap_files(&storage).is_empty());

    // Existing-id upsert -> delete-first marks the committed version deleted.
    store.upsert_document(1, doc("beta")).unwrap();

    // Deferred: nothing persisted yet, manifest flag untouched (#1024:
    // the flag lives in `segments.json` — `.meta` files are gone).
    assert!(
        delmap_files(&storage).is_empty(),
        "deletion bitmap must stay buffered until commit (#875)"
    );
    assert!(
        !read_manifest_text(&storage).contains("\"has_deletions\":true"),
        "has_deletions must not flip before commit (#875)"
    );

    store.commit().unwrap();

    // Persisted: bitmap on storage, flag flipped, old version deduped.
    assert!(
        !delmap_files(&storage).is_empty(),
        "commit must persist the deletion bitmap"
    );
    assert!(
        read_manifest_text(&storage).contains("\"has_deletions\":true"),
        "commit must flip has_deletions on the affected segment"
    );
    assert_eq!(hits(&store, "beta"), 1);
    assert_eq!(hits(&store, "alpha"), 0, "old version must stay deleted");
}

/// Storage decorator that records every `create_output` file name in order
/// and can inject write failures for names containing a chosen substring.
#[derive(Debug)]
struct RecordingStorage {
    inner: Arc<dyn Storage>,
    created: Mutex<Vec<String>>,
    /// While `Some(s)`, every `create_output(_append)` whose name contains
    /// `s` fails (persistent, unlike a one-shot fault — the writer's silent
    /// Drop-close retry must fail the same way a real full disk would).
    fail_contains: Mutex<Option<String>>,
}

impl RecordingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        RecordingStorage {
            inner,
            created: Mutex::new(Vec::new()),
            fail_contains: Mutex::new(None),
        }
    }

    fn set_fail_contains(&self, pattern: Option<&str>) {
        *self.fail_contains.lock().unwrap() = pattern.map(str::to_string);
    }

    fn maybe_fail(&self, name: &str) -> laurus::Result<()> {
        if let Some(pattern) = self.fail_contains.lock().unwrap().as_deref()
            && name.contains(pattern)
        {
            return Err(laurus::LaurusError::Storage(format!(
                "injected write failure for '{name}'"
            )));
        }
        Ok(())
    }
}

impl Storage for RecordingStorage {
    fn open_input(&self, name: &str) -> laurus::Result<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }
    fn create_output(&self, name: &str) -> laurus::Result<Box<dyn StorageOutput>> {
        self.maybe_fail(name)?;
        self.created.lock().unwrap().push(name.to_string());
        self.inner.create_output(name)
    }
    fn create_output_append(&self, name: &str) -> laurus::Result<Box<dyn StorageOutput>> {
        self.maybe_fail(name)?;
        self.created.lock().unwrap().push(name.to_string());
        self.inner.create_output_append(name)
    }
    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }
    fn delete_file(&self, name: &str) -> laurus::Result<()> {
        self.inner.delete_file(name)
    }
    fn list_files(&self) -> laurus::Result<Vec<String>> {
        self.inner.list_files()
    }
    fn file_size(&self, name: &str) -> laurus::Result<u64> {
        self.inner.file_size(name)
    }
    fn metadata(&self, name: &str) -> laurus::Result<FileMetadata> {
        self.inner.metadata(name)
    }
    fn rename_file(&self, old_name: &str, new_name: &str) -> laurus::Result<()> {
        self.inner.rename_file(old_name, new_name)
    }
    fn create_temp_output(&self, prefix: &str) -> laurus::Result<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }
    fn sync(&self) -> laurus::Result<()> {
        self.inner.sync()
    }
    fn close(&mut self) -> laurus::Result<()> {
        // The inner storage is shared behind `Arc`; closing is a no-op here.
        Ok(())
    }
}

/// Within a commit the deletion bitmap must be written BEFORE the
/// `metadata.json` checkpoint: `metadata.json` persists `last_wal_seq`, and
/// once that is durable a crash makes WAL replay skip the delete records —
/// so bitmaps landing after it would be lost forever. Also: the deletion log
/// is disabled (nothing replays it), so `deletions.log` must never be
/// created.
#[test]
fn commit_writes_delmap_before_metadata_checkpoint() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let recording = Arc::new(RecordingStorage::new(inner));
    let storage: Arc<dyn Storage> = recording.clone();
    let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();

    // Existing-id upsert (buffers a deletion), then commit.
    store.upsert_document(1, doc("beta")).unwrap();
    recording.created.lock().unwrap().clear();
    store.commit().unwrap();

    let created = recording.created.lock().unwrap().clone();
    // `.delmap` (since #1267) and `metadata.json` (since #1023) are both
    // published atomically, so what the recorder sees created is each one's
    // staging file. The ordering invariant this test exists for is unchanged
    // — only the names of the files that carry them into place.
    let delmap_pos = created
        .iter()
        .position(|f| f.ends_with(".delmap") || f.ends_with(".delmap.tmp"));
    let checkpoint_pos = created
        .iter()
        .position(|f| f == "metadata.json" || f == "metadata.json.tmp");
    assert!(
        delmap_pos.is_some(),
        "the deletion-flush commit must write a .delmap, got: {created:?}"
    );
    assert!(
        checkpoint_pos.is_some(),
        "the commit must write metadata.json, got: {created:?}"
    );
    assert!(
        delmap_pos.unwrap() < checkpoint_pos.unwrap(),
        ".delmap must land before the metadata.json last_wal_seq checkpoint \
         (#875), got: {created:?}"
    );
    assert!(
        !created.iter().any(|f| f == "deletions.log"),
        "the deletion log is disabled (#875) — nothing replays it, got: {created:?}"
    );
}

/// `optimize()` must flush buffered deletions before its force-merge: the
/// merge engine reads deletions from the on-disk `.delmap`, so an unflushed
/// deletion would be resurrected into the merged segment and then silently
/// discarded when the writer's deletion manager is dropped.
#[test]
fn optimize_does_not_resurrect_buffered_deletions() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    // Two commits -> two segments, so optimize actually merges.
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    store.upsert_document(2, doc("bravo")).unwrap();
    store.commit().unwrap();

    // Buffer a deletion of committed doc 1 (existing-id upsert), no commit.
    store.upsert_document(1, doc("gamma")).unwrap();

    store.optimize().unwrap();

    // The merged segment must not contain the deleted version.
    assert_eq!(
        hits(&store, "alpha"),
        0,
        "optimize must not resurrect a buffered deletion (#875)"
    );
    assert_eq!(hits(&store, "bravo"), 1, "unrelated doc survives the merge");

    // The buffered new version commits normally afterwards.
    store.commit().unwrap();
    assert_eq!(hits(&store, "gamma"), 1);
    assert_eq!(hits(&store, "alpha"), 0);
}

/// A FAILED commit must not destroy the buffered deletion state: the writer
/// stays cached with its deferred bitmaps/meta flips, and the retry commit
/// persists them. (Pre-fix, `commit()` took the writer out before committing,
/// so a transient storage error dropped it — and with it the only copy of the
/// deferred deletions — after which the retry "succeeded" trivially, the WAL
/// was truncated past the delete records, and the old version resurfaced
/// forever, no crash required.)
#[test]
fn failed_commit_preserves_buffered_deletions_for_retry() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let recording = Arc::new(RecordingStorage::new(inner));
    let storage: Arc<dyn Storage> = recording.clone();
    let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();

    // Buffer the deletion of committed doc 1 (existing-id upsert)...
    store.upsert_document(1, doc("beta")).unwrap();

    // ...and fail the commit at the deletion-bitmap write (transient error;
    // persistent while set, so the writer's silent Drop-close retry — if the
    // writer were wrongly dropped — cannot accidentally persist either).
    recording.set_fail_contains(Some(".delmap"));
    store
        .commit()
        .expect_err("the injected .delmap write failure must propagate");
    recording.set_fail_contains(None);

    // Retry: the buffered deletion (and the buffered new version) must have
    // survived the failed commit and persist now.
    store.commit().unwrap();
    assert_eq!(hits(&store, "beta"), 1, "the new version must be live");
    assert_eq!(
        hits(&store, "alpha"),
        0,
        "the deferred deletion must survive a failed commit and persist on \
         the retry (#875)"
    );
}

/// `add_field` invalidates the writer cache; it must commit the writer's
/// buffered state (including deferred deletions) with error propagation
/// instead of relying on the silent Drop-close commit. (Pre-fix, a transient
/// storage error during the drop made `add_field` return `Ok` while the only
/// copy of an acknowledged deletion was destroyed — the next successful
/// commit truncated the WAL past the delete records and the old version
/// resurfaced permanently. Reachable during normal ingest via the Dynamic
/// field policy.)
#[test]
fn add_field_propagates_writer_commit_failure_and_preserves_deletions() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let recording = Arc::new(RecordingStorage::new(inner));
    let storage: Arc<dyn Storage> = recording.clone();
    let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();

    // Buffer the deletion of committed doc 1 (existing-id upsert).
    store.upsert_document(1, doc("beta")).unwrap();

    // A transient failure while add_field commits the cached writer must
    // PROPAGATE (pre-fix: swallowed by the Drop-close) and must not destroy
    // the buffered state.
    recording.set_fail_contains(Some(".delmap"));
    let text_option = laurus::lexical::core::field::FieldOption::Text(Default::default());
    store
        .add_field("extra", text_option, None)
        .expect_err("add_field must propagate the writer-commit failure");
    recording.set_fail_contains(None);

    // The field registration itself preceded the failure; the buffered state
    // survived in the still-cached writer and persists on the next commit.
    store.commit().unwrap();
    assert_eq!(hits(&store, "beta"), 1, "the new version must be live");
    assert_eq!(
        hits(&store, "alpha"),
        0,
        "the deferred deletion must survive a failed add_field and persist (#875)"
    );
}

/// `rollback()` keeps buffered deletions: deletions were never rollback-able
/// (they used to be persisted synchronously at delete time) and their WAL
/// records are already acknowledged — so the next commit persists them.
/// Rollback is a writer-level API, so this drives the writer directly.
#[test]
fn rollback_keeps_buffered_deletions() {
    use laurus::lexical::{InvertedIndexWriter, InvertedIndexWriterConfig};

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut writer =
        InvertedIndexWriter::new(storage.clone(), InvertedIndexWriterConfig::default()).unwrap();

    writer.upsert_document(1, doc("alpha")).unwrap();
    writer.commit().unwrap();
    assert!(delmap_files(&storage).is_empty());

    // Existing-id upsert buffers the deletion AND the new version...
    writer.upsert_document(1, doc("beta")).unwrap();
    // ...rollback discards the buffered new version, not the deletion.
    writer.rollback().unwrap();
    writer.commit().unwrap();
    drop(writer);

    assert!(
        !delmap_files(&storage).is_empty(),
        "the commit after rollback must persist the buffered deletion (#875)"
    );

    // A fresh store over the same storage sees neither version: the old one
    // is deleted, the new one was rolled back.
    let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
    assert_eq!(
        hits(&store, "alpha"),
        0,
        "the deletion must survive rollback and persist at commit (#875)"
    );
    assert_eq!(hits(&store, "beta"), 0, "the rolled-back version is gone");
}
