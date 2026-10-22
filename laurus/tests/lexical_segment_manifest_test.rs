//! #1021 / #1024 — the lexical segment manifest.
//!
//! `segments.json` is the sole record of the committed segment set:
//! discovery reads only the manifest (zero storage I/O per reader
//! construction), publication and the merge transition are single atomic
//! writes, the orphan sweep reclaims what the manifest does not list —
//! but only under an authoritative (version >= 2) manifest — and every
//! merge crash window resolves in the manifest's favor with no duplicate
//! hits. Since #1024 there are no `.meta` files at all: what used to be
//! the equivalence suite (manifest == committed-`.meta` scan) is now a
//! set of direct manifest-content assertions per mutation class.

use std::io::Read;
use std::sync::Arc;

use laurus::lexical::index::inverted::segment::SegmentInfo;
use laurus::lexical::{LexicalIndexConfig, LexicalStore};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{Storage, StorageInput, StorageOutput};
use laurus::{Document, LaurusError, Result as LaurusResult};

/// Storage decorator failing the next `rename_file` whose destination
/// matches the armed `(prefix, suffix)` — the atomic-write publish step of
/// either the manifest (`segments.json`) or an advisory `.meta` flip.
#[derive(Debug)]
struct FailingStorage {
    inner: Arc<dyn Storage>,
    /// `(prefix, suffix, skip)` — fail the destination match after
    /// letting `skip` matches through.
    fail_rename_to: parking_lot::Mutex<Option<(String, String, usize)>>,
    fail_delete_of: parking_lot::Mutex<Option<(String, String)>>,
    fail_create_of: parking_lot::Mutex<Option<(String, String, usize)>>,
}

impl FailingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            fail_rename_to: parking_lot::Mutex::new(None),
            fail_delete_of: parking_lot::Mutex::new(None),
            fail_create_of: parking_lot::Mutex::new(None),
        }
    }

    fn fail_next_rename_matching(&self, prefix: &str, suffix: &str) {
        self.fail_rename_matching_after(prefix, suffix, 0);
    }

    /// Arm a one-shot rename failure that skips the first `skip` matches —
    /// for aiming past an earlier publication of the same file.
    fn fail_rename_matching_after(&self, prefix: &str, suffix: &str, skip: usize) {
        *self.fail_rename_to.lock() = Some((prefix.to_string(), suffix.to_string(), skip));
    }

    fn fail_next_delete_matching(&self, prefix: &str, suffix: &str) {
        *self.fail_delete_of.lock() = Some((prefix.to_string(), suffix.to_string()));
    }

    /// Asserts every armed failure was hit, for a step whose failure does
    /// not surface: a failed merge no longer fails the commit (Issue #1500).
    fn assert_all_hit(&self) {
        assert!(
            self.fail_rename_to.lock().is_none(),
            "the rename failure was not hit"
        );
        assert!(
            self.fail_delete_of.lock().is_none(),
            "the delete failure was not hit"
        );
        assert!(
            self.fail_create_of.lock().is_none(),
            "the create failure was not hit"
        );
    }
}

impl Storage for FailingStorage {
    fn create_output(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
        let armed = {
            let mut guard = self.fail_create_of.lock();
            match guard.as_mut() {
                Some((p, x, skip)) if name.starts_with(&**p) && name.ends_with(&**x) => {
                    if *skip > 0 {
                        *skip -= 1;
                        false
                    } else {
                        *guard = None;
                        true
                    }
                }
                _ => false,
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
        let armed = {
            let mut guard = self.fail_delete_of.lock();
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
                "injected failure deleting {name}"
            )));
        }
        self.inner.delete_file(name)
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> LaurusResult<()> {
        let armed = {
            let mut guard = self.fail_rename_to.lock();
            match guard.as_mut() {
                Some((p, x, skip)) if new_name.starts_with(&**p) && new_name.ends_with(&**x) => {
                    if *skip > 0 {
                        *skip -= 1;
                        false
                    } else {
                        *guard = None;
                        true
                    }
                }
                _ => false,
            }
        };
        if armed {
            return Err(LaurusError::storage(format!(
                "injected failure renaming {old_name} to {new_name}"
            )));
        }
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

/// Read a file's bytes.
fn read_bytes(storage: &Arc<dyn Storage>, name: &str) -> Vec<u8> {
    let mut input = storage.open_input(name).unwrap();
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes).unwrap();
    bytes
}

/// Parse the checksummed manifest: `varint(len) || json || crc`.
fn read_manifest(storage: &Arc<dyn Storage>) -> Vec<SegmentInfo> {
    let bytes = read_bytes(storage, "segments.json");
    let payload = if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
        value
    } else {
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
    };
    let segments = payload["segments"].as_array().unwrap().clone();
    segments
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect()
}

/// Assert the manifest holds exactly `count` segments.
#[track_caller]
fn assert_manifest_len(storage: &Arc<dyn Storage>, count: usize) {
    let manifest = read_manifest(storage);
    assert_eq!(
        manifest.len(),
        count,
        "unexpected committed segment set: {manifest:?}"
    );
}

/// Write a pre-#1024 `.meta` for every current manifest entry — the shape
/// a legacy (pre-manifest-authority) index left on storage. Used to
/// construct legacy-migration fixtures now that no production path writes
/// `.meta` files.
fn plant_legacy_metas(storage: &Arc<dyn Storage>) {
    use std::io::Write;
    for info in read_manifest(storage) {
        let legacy = serde_json::json!({
            "segment_id": info.segment_id,
            "doc_count": info.doc_count,
            "min_doc_id": info.min_doc_id,
            "max_doc_id": info.max_doc_id,
            "generation": info.generation,
            "has_deletions": info.has_deletions,
            "shard_id": info.shard_id,
            "committed": true,
        });
        let name = format!("{}.meta", info.segment_id);
        let mut out = storage.create_output(&name).unwrap();
        out.write_all(&serde_json::to_vec(&legacy).unwrap())
            .unwrap();
        out.close().unwrap();
    }
}

fn doc(body: &str) -> Document {
    Document::builder().add_text("body", body).build()
}

fn memory_storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
}

/// Plain commits: every commit adds its segment to the manifest.
#[test]
fn manifest_matches_meta_after_plain_commits() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    assert_eq!(
        read_manifest(&storage).len(),
        0,
        "create() must write an empty manifest from birth"
    );

    for round in 1..=2u64 {
        store.upsert_document(round, doc("alpha")).unwrap();
        store.commit().unwrap();
        assert_manifest_len(&storage, round as usize);
    }
}

/// A deleting commit: the `has_deletions` flip reaches the manifest.
#[test]
fn manifest_matches_meta_after_a_deleting_commit() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    // Overwrite the committed doc: marks a deletion in its segment.
    store.upsert_document(1, doc("alpha-v2")).unwrap();
    store.commit().unwrap();

    assert_manifest_len(&storage, 2);
    assert!(
        read_manifest(&storage).iter().any(|s| s.has_deletions),
        "the deleting commit must flip has_deletions in the manifest"
    );
}

/// The BLOCKER-1 canary: a segment that is deleted-from and published in
/// the SAME commit must enter the manifest with `has_deletions: true`.
///
/// The pending `SegmentInfo` is captured at flush time, before the
/// deletion exists; without the `pending_publish` update in
/// `flush_deletions`, the manifest would say `false` while the `.meta`
/// (read back from disk at flip time) says `true` — and once discovery
/// trusts the manifest, the deleted version would resurrect.
#[test]
fn delete_and_publish_in_one_commit_keeps_has_deletions() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    // Past max_buffered_docs (10_000) so a segment auto-flushes,
    // uncommitted. Doc 1 lands in it.
    for i in 1..=10_050u64 {
        store.upsert_document(i, doc("alpha")).unwrap();
    }
    // Overwrite doc 1: the deletion targets the FLUSHED, unpublished
    // segment.
    store.upsert_document(1, doc("alpha-v2")).unwrap();
    store.commit().unwrap();

    assert_manifest_len(&storage, 2);
    assert!(
        read_manifest(&storage)
            .iter()
            .any(|s| s.has_deletions && s.segment_id.starts_with("segment_")),
        "the flushed-then-deleted-from segment must carry has_deletions in the manifest"
    );
}

/// Auto-merge: the merge transition (drop sources, add merged with its
/// final generation) reaches the manifest atomically.
#[test]
fn manifest_matches_meta_after_auto_merge() {
    let storage = memory_storage();
    let config = LexicalIndexConfig::builder()
        .max_segments(1)
        .merge_factor(2)
        .build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    for round in 1..=3u64 {
        store.upsert_document(round, doc("alpha")).unwrap();
        store.commit().unwrap();
        // max_segments = 1: every commit past the first merges back down.
        assert_manifest_len(&storage, 1);
    }
    assert!(
        read_manifest(&storage)[0].segment_id.starts_with("merged_"),
        "the surviving segment must be the merged one"
    );
}

/// Optimize (force-merge-all): same property on the other merge path.
#[test]
fn manifest_matches_meta_after_optimize() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    for round in 1..=3u64 {
        store.upsert_document(round, doc("alpha")).unwrap();
        store.commit().unwrap();
    }
    store.optimize().unwrap();
    assert_manifest_len(&storage, 1);
    assert!(
        read_manifest(&storage)[0].segment_id.starts_with("merged_"),
        "optimize must leave exactly the merged segment"
    );
}

/// save-then-swap: a failed manifest save leaves the previous manifest
/// intact, and the retained writer's retry republishes without
/// double-adding.
#[test]
fn failed_manifest_save_retries_cleanly() {
    let inner = memory_storage();
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    let before = read_bytes(&inner, "segments.json");

    store.upsert_document(2, doc("beta")).unwrap();
    failing.fail_next_rename_matching("segments.json", "");
    assert!(
        store.commit().is_err(),
        "the injected manifest-save failure must surface"
    );
    assert_eq!(
        read_bytes(&inner, "segments.json"),
        before,
        "a failed save must leave the previous manifest byte-identical"
    );

    // The retained writer retries (#875) and republishes.
    store.commit().unwrap();
    let manifest = read_manifest(&inner);
    let mut ids: Vec<&str> = manifest.iter().map(|s| s.segment_id.as_str()).collect();
    ids.sort_unstable();
    let unique = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), unique, "a retried publish must not double-add");
    assert_eq!(manifest.len(), 2);
}

/// Legacy migration: a pre-manifest index (committed `.meta` files, no
/// `segments.json`) opens through the one sanctioned `.meta` read, and the
/// first mutation persists the migrated manifest.
///
/// The legacy state is planted by hand — no production path writes `.meta`
/// files any more (#1024).
#[test]
fn legacy_index_without_manifest_migrates_on_first_mutation() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    drop(store);

    // Rewind to the pre-manifest world: `.meta` files describe the
    // segments, `segments.json` does not exist.
    plant_legacy_metas(&storage);
    storage.delete_file("segments.json").unwrap();

    let reopened = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    reopened.upsert_document(2, doc("beta")).unwrap();
    reopened.commit().unwrap();

    assert_manifest_len(&storage, 2);
    assert_eq!(
        hits(&reopened, "alpha") + hits(&reopened, "beta"),
        2,
        "the legacy segment and the new one must both be live"
    );
}

// ---------------------------------------------------------------------------
// PR 2 — discovery reads the manifest; the sweep enforces it.
// ---------------------------------------------------------------------------

use laurus::lexical::{LexicalSearchRequest, TermQuery};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Storage decorator counting `list_files` calls and `.meta` opens.
#[derive(Debug)]
struct CountingStorage {
    inner: Arc<dyn Storage>,
    list_files_calls: AtomicUsize,
    meta_opens: AtomicUsize,
}

impl CountingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            list_files_calls: AtomicUsize::new(0),
            meta_opens: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        self.list_files_calls.store(0, Ordering::SeqCst);
        self.meta_opens.store(0, Ordering::SeqCst);
    }
}

impl Storage for CountingStorage {
    fn create_output(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
        self.inner.create_output(name)
    }

    fn create_output_append(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn open_input(&self, name: &str) -> LaurusResult<Box<dyn StorageInput>> {
        if name.ends_with(".meta") {
            self.meta_opens.fetch_add(1, Ordering::SeqCst);
        }
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
        self.list_files_calls.fetch_add(1, Ordering::SeqCst);
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

fn hits(store: &LexicalStore, term: &str) -> usize {
    store
        .search(LexicalSearchRequest::new(Box::new(TermQuery::new(
            "body", term,
        ))))
        .unwrap()
        .hits
        .len()
}

/// The authority proof: with every `.meta` deleted, the index still opens
/// and serves identical results — discovery owes nothing to the scan.
#[test]
fn discovery_survives_without_any_meta_file() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    for i in 1..=2u64 {
        store.upsert_document(i, doc("alpha")).unwrap();
        store.commit().unwrap();
    }
    assert_eq!(hits(&store, "alpha"), 2);
    drop(store);

    for f in storage.list_files().unwrap() {
        if f.ends_with(".meta") && f != "index.meta" {
            storage.delete_file(&f).unwrap();
        }
    }

    let reopened = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    assert_eq!(
        hits(&reopened, "alpha"),
        2,
        "discovery must be fully served by the manifest"
    );
}

/// The O(1) gate: building a fresh searcher performs ZERO `list_files`
/// calls and ZERO `.meta` opens — where the scan paid O(segments) of both.
#[test]
fn reader_construction_does_no_discovery_io() {
    let inner = memory_storage();
    let counting = Arc::new(CountingStorage::new(inner));
    let storage: Arc<dyn Storage> = counting.clone();

    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    for i in 1..=3u64 {
        store.upsert_document(i, doc("alpha")).unwrap();
        store.commit().unwrap();
    }

    // Invalidate the cached searcher, then measure a fresh build + search.
    store.refresh().unwrap();
    counting.reset();
    assert_eq!(hits(&store, "alpha"), 3);
    assert_eq!(
        counting.list_files_calls.load(Ordering::SeqCst),
        0,
        "reader construction must not list storage"
    );
    assert_eq!(
        counting.meta_opens.load(Ordering::SeqCst),
        0,
        "reader construction must not open any .meta"
    );
}

/// Write a stray, committed-looking segment the manifest does not know.
///
/// `with_meta` controls whether the stray carries a committed `.meta`.
/// The gate cases (absent / version-1 manifest) must use a META-LESS
/// stray: with a committed `.meta`, the legacy migration scan would ADOPT
/// the stray as a real segment and keep it regardless of the sweep gate —
/// masking exactly what those cases exist to pin.
fn plant_stray_segment(storage: &Arc<dyn Storage>, stem: &str, with_meta: bool) {
    let mut files = vec![(format!("{stem}.post"), b"junk".to_vec())];
    if with_meta {
        // The pre-#1024 on-disk shape, committed included.
        let legacy = serde_json::json!({
            "segment_id": stem,
            "doc_count": 1,
            "min_doc_id": 900_000,
            "max_doc_id": 900_000,
            "generation": 99,
            "has_deletions": false,
            "shard_id": 0,
            "committed": true,
        });
        files.push((format!("{stem}.meta"), serde_json::to_vec(&legacy).unwrap()));
    }
    for (name, bytes) in files {
        let mut out = storage.create_output(&name).unwrap();
        out.write_all(&bytes).unwrap();
        out.close().unwrap();
    }
}

/// The sweep runs only under an authoritative (version >= 2) manifest:
/// v2 reclaims strays, an absent manifest never sweeps, and a version-1
/// manifest is treated exactly like an absent one until the next
/// publication upgrades it.
#[test]
fn sweep_is_gated_on_an_authoritative_manifest() {
    // Case 1: v2 manifest — the stray is reclaimed.
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    drop(store);
    plant_stray_segment(&storage, "segment_999999", /* with_meta */ true);
    let reopened = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    drop(reopened);
    assert!(
        !storage.file_exists("segment_999999.post"),
        "a v2 manifest must reclaim stray segment files at open"
    );

    // Case 2: no manifest, a legacy index (`.meta` files describe the
    // segments — a bare absence would refuse to open, see
    // `open_refuses_segment_files_with_nothing_to_describe_them`).
    // Nothing is swept.
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    drop(store);
    plant_legacy_metas(&storage);
    storage.delete_file("segments.json").unwrap();
    // Meta-less: the migration scan must not adopt it, so only the sweep
    // gate stands between it and deletion.
    plant_stray_segment(&storage, "segment_999999", /* with_meta */ false);
    let reopened = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    drop(reopened);
    assert!(
        storage.file_exists("segment_999999.post"),
        "an absent manifest must never be swept against"
    );

    // Case 3: a v1 manifest is a hint, not a deletion warrant — but the
    // next publication upgrades it to v2, after which the sweep applies.
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    drop(store);
    // Rewind to the PR-1 era: `.meta` files exist (a v1-era index always
    // wrote them) and the manifest says version 1, raw-JSON legacy form
    // (accepted by the loader), listing the real segments.
    let real = read_manifest(&storage);
    plant_legacy_metas(&storage);
    let v1 = serde_json::json!({ "version": 1u32, "segments": real });
    storage.delete_file("segments.json").unwrap();
    let mut out = storage.create_output("segments.json").unwrap();
    out.write_all(&serde_json::to_vec(&v1).unwrap()).unwrap();
    out.close().unwrap();
    // Meta-less, so only the gate protects it (see plant_stray_segment).
    plant_stray_segment(&storage, "segment_888888", /* with_meta */ false);

    let reopened = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    assert!(
        storage.file_exists("segment_888888.post"),
        "a version-1 manifest must not be used as a deletion warrant"
    );
    // The next commit publishes a v2 manifest (which does not know the
    // meta-less stray), and the reopen after THAT reclaims it.
    reopened.upsert_document(2, doc("beta")).unwrap();
    reopened.commit().unwrap();
    drop(reopened);
    let final_store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    drop(final_store);
    assert!(
        !storage.file_exists("segment_888888.post"),
        "under the upgraded v2 manifest the meta-less stray is an orphan and is reclaimed"
    );
}

/// A merge whose MANIFEST save fails: the sources stay authoritative, the
/// merged leftovers are reclaimed at the next open, and no document is
/// served twice.
#[test]
fn merge_crash_before_manifest_save_resolves_to_the_sources() {
    let inner = memory_storage();
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let config = LexicalIndexConfig::builder()
        .max_segments(1)
        .merge_factor(2)
        .build();
    let store = LexicalStore::new(storage.clone(), config.clone()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    store.upsert_document(2, doc("alpha")).unwrap();
    // Commit 2 renames segments.json twice: its own publish, then the
    // merge transition. Skip the publish; kill the merge. The commit
    // succeeds: its own publish went through (Issue #1500).
    failing.fail_rename_matching_after("segments.json", "", 1);
    store.commit().unwrap();
    failing.assert_all_hit();
    drop(store);

    let reopened = LexicalStore::new(inner.clone(), config).unwrap();
    assert_eq!(
        hits(&reopened, "alpha"),
        2,
        "every document exactly once, served by the source segments"
    );
    drop(reopened);
    assert!(
        !inner
            .list_files()
            .unwrap()
            .iter()
            .any(|f| f.starts_with("merged_")),
        "the failed merge's leftovers must be reclaimed"
    );
}

/// A merge that crashes after its manifest save — during the source file
/// deletions: the merged segment is authoritative, the source leftovers
/// are reclaimed at the next open, no duplicates.
///
/// (#1024 removed the other post-save steps this test used to exercise:
/// the advisory `.meta` generation rewrite no longer exists.)
#[test]
fn merge_crash_after_manifest_save_resolves_to_the_merged_segment() {
    let inner = memory_storage();
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let config = LexicalIndexConfig::builder()
        .max_segments(1)
        .merge_factor(2)
        .build();
    let store = LexicalStore::new(storage.clone(), config.clone()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    store.upsert_document(2, doc("alpha")).unwrap();
    // The first source data-file deletion fails after the manifest already
    // recorded the transition. The commit succeeds (Issue #1500).
    failing.fail_next_delete_matching("segment_", ".cfs");
    store.commit().unwrap();
    failing.assert_all_hit();
    drop(store);

    let reopened = LexicalStore::new(inner.clone(), config).unwrap();
    assert_eq!(
        hits(&reopened, "alpha"),
        2,
        "every document exactly once, served by the merged segment"
    );
    drop(reopened);
    let files = inner.list_files().unwrap();
    assert!(
        files.iter().any(|f| f.starts_with("merged_")),
        "the merged segment is the live copy"
    );
    assert!(
        !files.iter().any(|f| f.starts_with("segment_")),
        "the source leftovers must be reclaimed, found {files:?}"
    );
}

// ---------------------------------------------------------------------------
// #1024 PR A — writer state derives from the manifest; the scan is legacy.
// ---------------------------------------------------------------------------

/// The lost-manifest guard: segment files with neither a manifest nor any
/// `.meta` to migrate from must refuse to open — the alternative is a
/// silently empty index whose next commit publishes a fresh manifest that
/// the following open's sweep enforces by deleting every old segment.
#[test]
fn open_refuses_segment_files_with_nothing_to_describe_them() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    drop(store);

    // Simulate the lost-manifest state of a post-#1024 index: no
    // segments.json, no .meta, but the segment data files are there.
    storage.delete_file("segments.json").unwrap();
    for f in storage.list_files().unwrap() {
        if f.ends_with(".meta") && f != "index.meta" {
            storage.delete_file(&f).unwrap();
        }
    }

    let err = LexicalStore::new(storage.clone(), LexicalIndexConfig::default());
    assert!(
        err.is_err(),
        "segment files without a manifest or .meta must refuse to open, not \
         serve a silently empty index"
    );
}

/// The generation seed counts surviving segment-file stems, not just
/// manifest entries: an ordinal whose files survived (a failed best-effort
/// sweep deletion, a legacy crash) must never be reused — the new segment
/// would adopt stale foreign `.bkd`s / `.delmap`s by name prefix.
#[test]
fn generation_seed_never_reuses_a_surviving_file_ordinal() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();
    drop(store);

    // A stray high-ordinal data file the manifest does not know. The sweep
    // will reclaim it, but the seed is taken from the pre-sweep listing.
    plant_stray_segment(&storage, "segment_000009", /* with_meta */ false);

    let reopened = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    reopened.upsert_document(2, doc("beta")).unwrap();
    reopened.commit().unwrap();

    let new_entry_generation = read_manifest(&storage)
        .iter()
        .map(|s| s.generation)
        .max()
        .unwrap();
    assert_eq!(
        new_entry_generation, 10,
        "the next generation must clear the surviving stray ordinal (9), got {new_entry_generation}"
    );
}

/// Flush-time reservation from the shared counter: a writer surviving
/// `optimize` and the merge can no longer mint the same generation (both
/// used to compute `max + 1` independently), and the post-merge flush
/// sorts strictly NEWER than the merged segment.
#[test]
fn surviving_writer_and_merge_never_tie_on_generation() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    for i in 1..=2u64 {
        store.upsert_document(i, doc("alpha")).unwrap();
        store.commit().unwrap();
    }
    // A live writer with buffered docs survives optimize (#864/#1017).
    store.upsert_document(3, doc("alpha")).unwrap();
    store.optimize().unwrap();
    // The same writer flushes again after the merge.
    store.upsert_document(4, doc("alpha")).unwrap();
    store.commit().unwrap();

    let manifest = read_manifest(&storage);
    let mut generations: Vec<u64> = manifest.iter().map(|s| s.generation).collect();
    generations.sort_unstable();
    let unique = generations.len();
    generations.dedup();
    assert_eq!(
        generations.len(),
        unique,
        "generations must be unique across the merge and the surviving writer: {manifest:?}"
    );

    let merged_generation = manifest
        .iter()
        .find(|s| s.segment_id.starts_with("merged_"))
        .map(|s| s.generation)
        .expect("optimize must have produced a merged segment");
    let post_merge_flush = manifest
        .iter()
        .filter(|s| s.segment_id.starts_with("segment_"))
        .map(|s| s.generation)
        .max()
        .expect("the post-merge commit must have published a segment");
    assert!(
        post_merge_flush > merged_generation,
        "a segment flushed after the merge must sort newer than the merged \
         segment (merged={merged_generation}, flush={post_merge_flush})"
    );
}
