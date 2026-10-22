//! Reproduction for #1032: a failed merge publishes a partially merged
//! segment through the merge writer's implicit `Drop`-commit.
//!
//! `MergeEngine::perform_merge` replays the source documents through an
//! internally constructed `InvertedIndexWriter` and writes the merged
//! segment via `flush_buffered_to_segment`. If that fails (or the replay
//! loop errors), the function returns early and the writer is dropped with
//! its buffers full — and `Drop` runs `close()` → `commit()`, whose
//! `flush_segment()` + `publish_pending_segments()` write a `segment_*`
//! holding the partially merged documents and flip it to `committed: true`
//! in the same call. The source segments still exist, so every document in
//! the orphaned segment is now live TWICE.

use std::io::Read;
use std::sync::Arc;

use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{Storage, StorageInput, StorageOutput};
use laurus::{Document, LaurusError, Result as LaurusResult};

/// Storage decorator failing the next `create_output` whose name matches
/// the armed `(prefix, suffix)` — aimed at a chosen file of the merged
/// segment, so the merge dies inside `flush_buffered_to_segment` with the
/// writer's buffers still full.
#[derive(Debug)]
struct FailingStorage {
    inner: Arc<dyn Storage>,
    fail_create_matching: parking_lot::Mutex<Option<(String, String)>>,
}

impl FailingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            fail_create_matching: parking_lot::Mutex::new(None),
        }
    }

    /// Arm a one-shot failure for the next `create_output` whose name has
    /// this prefix (suffix unconstrained).
    fn fail_next_create_with_prefix(&self, prefix: &str) {
        self.fail_next_create_matching(prefix, "");
    }

    /// Arm a one-shot failure for the next `create_output` whose name has
    /// both this prefix and this suffix.
    fn fail_next_create_matching(&self, prefix: &str, suffix: &str) {
        *self.fail_create_matching.lock() = Some((prefix.to_string(), suffix.to_string()));
    }

    /// Asserts the armed failure was hit. A failed merge no longer fails the
    /// commit (Issue #1500), so the commit's result cannot tell.
    fn assert_hit(&self) {
        assert!(
            self.fail_create_matching.lock().is_none(),
            "the injected merge failure was not hit"
        );
    }
}

impl Storage for FailingStorage {
    fn create_output(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
        let armed = {
            let mut guard = self.fail_create_matching.lock();
            if guard
                .as_ref()
                .is_some_and(|(p, x)| name.starts_with(p) && name.ends_with(x))
            {
                *guard = None;
                true
            } else {
                false
            }
        };
        if armed {
            return Err(LaurusError::storage(format!(
                "injected failure creating {name}"
            )));
        }
        self.inner.create_output(name)
    }

    fn create_output_append(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn open_input(&self, name: &str) -> LaurusResult<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }

    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }

    fn delete_file(&self, name: &str) -> LaurusResult<()> {
        self.inner.delete_file(name)
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> LaurusResult<()> {
        self.inner.rename_file(old_name, new_name)
    }

    fn list_files(&self) -> LaurusResult<Vec<String>> {
        self.inner.list_files()
    }

    fn file_size(&self, name: &str) -> LaurusResult<u64> {
        self.inner.file_size(name)
    }

    fn sync(&self) -> LaurusResult<()> {
        self.inner.sync()
    }

    fn metadata(&self, name: &str) -> LaurusResult<laurus::storage::FileMetadata> {
        self.inner.metadata(name)
    }

    fn create_temp_output(&self, prefix: &str) -> LaurusResult<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }

    fn close(&mut self) -> LaurusResult<()> {
        Ok(())
    }
}

fn doc(body: &str) -> Document {
    Document::builder().add_text("body", body).build()
}

/// Auto-merging config: the second commit's `maybe_merge` merges the two
/// single-doc segments.
fn merging_config() -> LexicalIndexConfig {
    LexicalIndexConfig::builder()
        .max_segments(1)
        .merge_factor(2)
        .build()
}

/// A failed merge must not leave a published segment holding the partially
/// merged documents while the source segments still exist.
#[test]
fn failed_merge_must_not_duplicate_documents() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = LexicalStore::new(storage.clone(), merging_config()).unwrap();
    store.upsert_document(1, doc("apple")).unwrap();
    store.commit().unwrap();
    store.upsert_document(2, doc("banana")).unwrap();

    // The second commit's maybe_merge merges the two segments; kill the
    // merged segment's first file write. The commit still succeeds: its own
    // segment was durable before the merge ran (Issue #1500).
    failing.fail_next_create_with_prefix("merged_");
    store.commit().unwrap();
    failing.assert_hit();
    drop(store);

    // Reopen over the clean storage: discovery sees whatever the failed
    // merge left behind.
    let reopened = LexicalStore::new(inner.clone(), merging_config()).unwrap();
    let hits = reopened
        .search(LexicalSearchRequest::new(Box::new(TermQuery::new(
            "body", "apple",
        ))))
        .unwrap();
    assert_eq!(
        hits.hits.len(),
        1,
        "a failed merge must not leave a second live copy of a document"
    );
}

/// Structural variant of the same defect: after a failed merge, no NEW
/// committed segment may exist beyond the sources.
#[test]
fn failed_merge_must_not_publish_a_segment() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = LexicalStore::new(storage.clone(), merging_config()).unwrap();
    store.upsert_document(1, doc("apple")).unwrap();
    store.commit().unwrap();

    let before = list_manifest_ids(&inner);

    store.upsert_document(2, doc("banana")).unwrap();
    failing.fail_next_create_with_prefix("merged_");
    store.commit().unwrap();
    failing.assert_hit();
    drop(store);

    // The new source segment from commit 2 is expected in the manifest —
    // the sole publication record since #1024; anything further would be
    // the Drop-published orphan.
    let after = list_manifest_ids(&inner);
    let added: Vec<&String> = after.iter().filter(|m| !before.contains(m)).collect();
    assert_eq!(
        added.len(),
        1,
        "a failed merge must add only commit 2's own segment, found: {added:?}"
    );
}

/// A failure in any data-file write must not leave a discoverable
/// segment (#1032, restated for #1024: discoverability now means a
/// manifest entry, and publication happens only after every data file
/// exists — there is no `.meta` to order against any more).
///
/// Kills the merged segment's `.cfs` container write — the one data file
/// a compound-layout merge produces; the manifest must not list the
/// merged segment afterwards.
#[test]
fn segment_meta_is_written_after_every_data_file() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = LexicalStore::new(storage.clone(), merging_config()).unwrap();
    // Numeric field so the container carries BKD data as well.
    let numbered = |body: &str, n: i64| {
        Document::builder()
            .add_text("body", body)
            .add_integer("rank", n)
            .build()
    };
    store.upsert_document(1, numbered("apple", 1)).unwrap();
    store.commit().unwrap();
    store.upsert_document(2, numbered("banana", 2)).unwrap();

    failing.fail_next_create_matching("merged_", ".cfs");
    store.commit().unwrap();
    failing.assert_hit();
    drop(store);

    let leaked: Vec<String> = list_manifest_ids(&inner)
        .into_iter()
        .filter(|id| id.starts_with("merged_"))
        .collect();
    assert!(
        leaked.is_empty(),
        "a failed data-file write must not leave a discoverable merged segment: {leaked:?}"
    );
}

/// The segment ids the manifest — the sole publication record — lists.
fn list_manifest_ids(storage: &Arc<dyn Storage>) -> Vec<String> {
    let mut input = storage.open_input("segments.json").unwrap();
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes).unwrap();
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
    let mut ids: Vec<String> = payload["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["segment_id"].as_str().unwrap().to_string())
        .collect();
    ids.sort();
    ids
}
