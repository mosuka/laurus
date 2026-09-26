//! Reproductions for #1023: the lexical `metadata.json` has several writers
//! that each believe they own it.
//!
//! PR 1 (#1027) and PR 2 (#1028) made every write atomic and checksummed;
//! these tests pin the *ownership* half — the lost updates that remain even
//! with perfectly durable writes:
//!
//! - `InvertedIndex::update_metadata` blind-writes its stale in-memory copy
//!   over the writer's fresher on-disk state at the end of `optimize` (R1);
//! - the merge engine's internally constructed writer re-adds the whole
//!   merged output to `doc_count` from its `Drop`, compounding on every
//!   auto-merging commit (R2);
//! - the writer applies its lifetime stats to a base frozen at construction,
//!   so any interleaved write is silently reverted (R3 guards the delta
//!   rewrite this forces);
//! - `InvertedIndex`'s WAL-checkpoint accessors are inherent methods, not
//!   trait members, so through `Box<dyn LexicalIndex>` the store hits the
//!   trait defaults: `last_wal_seq()` is always `0` and the writer-less
//!   `set_last_wal_seq` silently discards the value (R4);
//! - making the lexical checkpoint real activates `Engine::recover`'s
//!   skip-everything path, which must still leave the WAL truncatable (R5).

use std::io::Read;
use std::sync::Arc;

use laurus::lexical::{LexicalIndexConfig, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document, Engine, FieldOption, Schema};
use laurus::{LaurusError, Result as LaurusResult};

/// Read `metadata.json` and parse its JSON payload, accepting both the
/// checksummed framing (#1028) and bare legacy JSON.
fn read_meta(storage: &Arc<dyn Storage>) -> serde_json::Value {
    let mut input = storage.open_input("metadata.json").unwrap();
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes).unwrap();
    if let Ok(v) = serde_json::from_slice(&bytes) {
        return v;
    }
    // varint(len) || json || crc
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

/// A document with one text field.
fn doc(body: &str) -> Document {
    Document::builder().add_text("body", body).build()
}

fn memory_storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
}

/// R1 — `optimize` must not roll the persisted checkpoint back.
///
/// The writer's in-optimize commit persists the current `last_wal_seq` and
/// `doc_count`; `InvertedIndex::optimize` then calls `update_metadata`,
/// which persists the index's in-memory copy — last refreshed at the
/// previous commit. On main the file ends at the OLD values: a durable
/// rollback of the WAL checkpoint on the happy path, no crash required.
#[test]
fn optimize_must_not_roll_back_the_metadata_checkpoint() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    for i in 1..=2u64 {
        store.upsert_document(i, doc("alpha")).unwrap();
    }
    store.set_last_wal_seq(2).unwrap();
    store.commit().unwrap();

    for i in 3..=9u64 {
        store.upsert_document(i, doc("alpha")).unwrap();
    }
    store.set_last_wal_seq(9).unwrap();
    store.optimize().unwrap();

    let m = read_meta(&storage);
    assert_eq!(
        m["last_wal_seq"], 9,
        "optimize must persist the newest WAL checkpoint, not roll it back"
    );
    assert_eq!(
        m["doc_count"], 9,
        "optimize must persist the true document count, not roll it back"
    );
}

/// R2 — an auto-merging commit must not inflate `doc_count`.
///
/// Deliberately does NOT go through `optimize`: there, `update_metadata`
/// runs after the merge and its rollback (R1) overwrites the inflation —
/// the two defects mask each other, and a test driving `optimize` alone can
/// go green for the wrong reason.
///
/// On main the merge engine's internal writer re-adds the whole merged
/// output from its `Drop` on every commit whose `maybe_merge` fires, and
/// each merge reads an already-inflated base, so it compounds: 1 → 4 → 8
/// for three documents.
#[test]
fn auto_merge_must_not_inflate_doc_count() {
    let storage = memory_storage();
    let config = LexicalIndexConfig::builder()
        .max_segments(1)
        .merge_factor(2)
        .build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    for round in 1..=3u64 {
        store.upsert_document(round, doc("alpha")).unwrap();
        store.set_last_wal_seq(round).unwrap();
        store.commit().unwrap();
    }

    let m = read_meta(&storage);
    assert_eq!(
        m["doc_count"], 3,
        "three committed documents must persist doc_count 3, however many merges ran"
    );
    assert_eq!(
        m["last_wal_seq"], 3,
        "the checkpoint must survive the merges"
    );

    // `LexicalStore::commit` refreshes the in-memory copy from the file, so
    // the inflation also poisons the live stats.
    assert_eq!(
        store.stats().unwrap().doc_count,
        3,
        "the in-memory stats must not launder the inflated file back in"
    );
}

/// R3 — a writer surviving `optimize` must not corrupt the next commit.
///
/// `optimize` commits the cached writer and keeps it; the next `commit`
/// makes the same writer persist again, and `LexicalStore::commit` then
/// drops it — whose `Drop` runs `close()` → `commit()`, a SECOND full
/// ladder (`commit()` never sets `closed`).
///
/// Today both extra commits happen to be harmless because the writer writes
/// absolute values derived from a frozen base. Under #1023's delta design
/// they are only harmless if the deltas are consumed inside `commit()`
/// itself — applied once, exactly. This test pins that: green on main, and
/// it must stay green when the delta rewrite lands (disable the in-commit
/// stats reset to see it fail).
#[test]
fn post_optimize_commit_must_keep_exact_counts() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    for i in 1..=2u64 {
        store.upsert_document(i, doc("alpha")).unwrap();
    }
    store.set_last_wal_seq(2).unwrap();
    store.commit().unwrap();

    for i in 3..=5u64 {
        store.upsert_document(i, doc("alpha")).unwrap();
    }
    store.set_last_wal_seq(5).unwrap();
    store.optimize().unwrap();

    store.upsert_document(6, doc("alpha")).unwrap();
    store.set_last_wal_seq(6).unwrap();
    store.commit().unwrap();

    let m = read_meta(&storage);
    assert_eq!(
        m["doc_count"], 6,
        "six documents across optimize + commit must persist doc_count 6"
    );
    assert_eq!(m["last_wal_seq"], 6);

    // And a reopen agrees, so the file was not healed by an in-memory copy.
    drop(store);
    let reopened = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    assert_eq!(reopened.stats().unwrap().doc_count, 6);
}

/// R4 — the store-level WAL checkpoint accessors must be live.
///
/// `InvertedIndex::last_wal_seq`/`set_last_wal_seq` exist but as inherent
/// methods outside the `impl LexicalIndex` block, so through the store's
/// `Box<dyn LexicalIndex>` both resolve to the trait defaults: the getter
/// returns `0` in every state, and the writer-less setter silently returns
/// `Ok(())` while discarding the value.
#[test]
fn last_wal_seq_accessor_must_return_the_persisted_value() {
    let storage = memory_storage();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    store.upsert_document(1, doc("alpha")).unwrap();
    store.set_last_wal_seq(7).unwrap();
    store.commit().unwrap();

    assert_eq!(
        read_meta(&storage)["last_wal_seq"],
        7,
        "precondition: the write path persists the checkpoint correctly"
    );
    assert_eq!(
        store.last_wal_seq(),
        7,
        "the accessor must report the persisted checkpoint, not the trait default 0"
    );

    // A reopened store reads it back from disk.
    drop(store);
    let reopened = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    assert_eq!(
        reopened.last_wal_seq(),
        7,
        "a reopened store must see the persisted checkpoint"
    );

    // The writer-less setter path must persist rather than silently drop.
    reopened.set_last_wal_seq(8).unwrap();
    assert_eq!(
        reopened.last_wal_seq(),
        8,
        "the writer-less setter must take effect"
    );
    assert_eq!(
        read_meta(&storage)["last_wal_seq"],
        8,
        "the writer-less setter must persist the checkpoint"
    );
}

// ---------------------------------------------------------------------------
// #1041 — one metadata persist per committing commit, zero per no-op commit.
// ---------------------------------------------------------------------------

/// Storage decorator counting `create_output` calls that target
/// `metadata.json` (its atomic write creates `metadata.json.tmp` and
/// renames it, so the tmp create is the one observable per persist).
#[derive(Debug)]
struct MetadataCountingStorage {
    inner: Arc<dyn Storage>,
    metadata_creates: Arc<std::sync::atomic::AtomicU64>,
}

impl Storage for MetadataCountingStorage {
    fn create_output(&self, name: &str) -> LaurusResult<Box<dyn laurus::storage::StorageOutput>> {
        if name.starts_with("metadata.json") {
            self.metadata_creates
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        self.inner.create_output(name)
    }

    fn create_output_append(
        &self,
        name: &str,
    ) -> LaurusResult<Box<dyn laurus::storage::StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn open_input(&self, name: &str) -> LaurusResult<Box<dyn laurus::storage::StorageInput>> {
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

    fn create_temp_output(
        &self,
        prefix: &str,
    ) -> LaurusResult<(String, Box<dyn laurus::storage::StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }

    fn close(&mut self) -> LaurusResult<()> {
        Ok(())
    }
}

/// #1041 — a committing commit must persist `metadata.json` exactly once.
///
/// `LexicalStore::commit` runs the writer's explicit `commit()` and then
/// drops the writer; `Drop → close() → commit()` re-runs the ladder with
/// every delta already consumed, and `write_metadata_json` used to persist
/// that zero-delta snapshot anyway — a full create + rename + fsync per
/// commit for a byte-equivalent file (modulo `generation`, which nothing
/// reads).
#[test]
fn a_committing_commit_persists_metadata_exactly_once() {
    let creates = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let storage: Arc<dyn Storage> = Arc::new(MetadataCountingStorage {
        inner: memory_storage(),
        metadata_creates: creates.clone(),
    });
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    let before = creates.load(std::sync::atomic::Ordering::SeqCst);
    store.upsert_document(1, doc("alpha")).unwrap();
    store.commit().unwrap();

    assert_eq!(
        creates.load(std::sync::atomic::Ordering::SeqCst) - before,
        1,
        "a committing commit must persist metadata.json exactly once, \
         not once per ladder run"
    );
    assert_eq!(
        read_meta(&storage)["doc_count"],
        1,
        "the single persist must carry the commit's delta"
    );
}

/// #1041 — a no-op commit must not persist `metadata.json` at all.
///
/// A registered writer with nothing buffered, no deletions, and an
/// unchanged WAL seq has nothing to record; committing and dropping it
/// must leave the file untouched.
#[test]
fn a_noop_commit_skips_the_metadata_persist() {
    let creates = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let storage: Arc<dyn Storage> = Arc::new(MetadataCountingStorage {
        inner: memory_storage(),
        metadata_creates: creates.clone(),
    });
    let index = laurus::lexical::index::inverted::InvertedIndex::create(
        storage,
        laurus::lexical::InvertedIndexConfig::default(),
    )
    .unwrap();
    use laurus::lexical::index::LexicalIndex;

    let before = creates.load(std::sync::atomic::Ordering::SeqCst);
    let mut writer = index.writer().unwrap();
    writer.commit().unwrap();
    drop(writer);

    assert_eq!(
        creates.load(std::sync::atomic::Ordering::SeqCst) - before,
        0,
        "a zero-delta commit (and the Drop re-run) must skip the metadata persist"
    );
}

/// #1041 regression guard for the #875 retry contract: a FAILED persist
/// leaves the deltas unconsumed, so the retried commit is not zero-delta
/// and must still write.
#[test]
fn a_failed_metadata_persist_still_writes_on_the_retry() {
    let inner = memory_storage();
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    store.upsert_document(1, doc("alpha")).unwrap();
    failing.fail_next_create_of("metadata.json.tmp");
    assert!(
        store.commit().is_err(),
        "the injected metadata persist failure must surface"
    );

    store.commit().unwrap();
    assert_eq!(
        read_meta(&storage)["doc_count"],
        1,
        "the retried commit must persist the unconsumed delta, not skip as zero-delta"
    );
}

// ---------------------------------------------------------------------------
// R5 — Engine::recover with a REAL lexical checkpoint.
// ---------------------------------------------------------------------------

/// Storage decorator failing the next `create_output` of a named file, used
/// to abort the commit ladder at its final WAL-truncate step: every store
/// has durably committed, but the WAL still holds the covered records.
#[derive(Debug)]
struct FailingStorage {
    inner: Arc<dyn Storage>,
    fail_create_of: parking_lot::Mutex<Option<String>>,
}

impl FailingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            fail_create_of: parking_lot::Mutex::new(None),
        }
    }

    fn fail_next_create_of(&self, name: &str) {
        *self.fail_create_of.lock() = Some(name.to_string());
    }
}

impl Storage for FailingStorage {
    fn create_output(&self, name: &str) -> LaurusResult<Box<dyn laurus::storage::StorageOutput>> {
        let armed = {
            let mut guard = self.fail_create_of.lock();
            if guard.as_deref() == Some(name) {
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

    fn create_output_append(
        &self,
        name: &str,
    ) -> LaurusResult<Box<dyn laurus::storage::StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn open_input(&self, name: &str) -> LaurusResult<Box<dyn laurus::storage::StorageInput>> {
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

    fn create_temp_output(
        &self,
        prefix: &str,
    ) -> LaurusResult<(String, Box<dyn laurus::storage::StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }

    fn close(&mut self) -> LaurusResult<()> {
        Ok(())
    }
}

/// A schema with a text field and a Flat vector field.
///
/// The vector field is load-bearing: `recover`'s skip guard requires the
/// record to be covered by BOTH checkpoints, and on a lexical-only engine
/// the vector aggregate is `0`, so the guard can never fire and the hazard
/// this test pins is unreachable.
fn mixed_schema() -> Schema {
    use laurus::lexical::TextOption;
    use laurus::vector::FlatOption;
    Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field(
            "vec",
            FieldOption::Flat(FlatOption {
                dimension: 4,
                ..Default::default()
            }),
        )
        .build()
}

fn mixed_doc(i: u64) -> Document {
    Document::builder()
        .add_text("title", format!("doc {i}"))
        .add_field("vec", DataValue::Vector(vec![i as f32, 0.0, 1.0, 0.5]))
        .build()
}

/// R5 — after a recovery that skips every record (both checkpoints already
/// cover the WAL), the next commit must still empty the WAL.
///
/// The state is a crash between the commit ladder's materialize steps and
/// its WAL truncate: every store is durable at checkpoint N while the WAL
/// still holds records 1..=N. With a real lexical checkpoint (R4), recover
/// skips them all — and `applied_seq` must still be published for the
/// skipped records, otherwise `truncate_retaining_after(0)` re-retains the
/// entire dead WAL on the next commit, breaking the documented post-commit
/// invariant.
#[tokio::test(flavor = "multi_thread")]
async fn recover_skip_path_must_leave_the_wal_truncatable() -> laurus::Result<()> {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let engine = Engine::new(storage, mixed_schema()).await?;
    for i in 1..=3u64 {
        engine.put_document(&format!("id{i}"), mixed_doc(i)).await?;
    }

    // Fail the ladder at its final step: the fast-path truncate recreates
    // the WAL via `create_output`. Steps 1-4 (WAL barrier, lexical, vector,
    // documents) have all completed by then.
    failing.fail_next_create_of("engine.wal");
    let err = engine.commit().await;
    assert!(err.is_err(), "the injected truncate failure must surface");
    drop(engine);

    let wal_size = inner.file_size("engine.wal").unwrap_or(0);
    assert!(
        wal_size > 0,
        "precondition: the WAL still holds the covered records"
    );

    // Reopen over the clean storage: recover sees records that both
    // checkpoints already cover.
    let reopened = Engine::new(inner.clone(), mixed_schema()).await?;

    // A commit with no new mutations must uphold the post-commit invariant.
    reopened.commit().await?;
    assert_eq!(
        inner.file_size("engine.wal").unwrap_or(0),
        0,
        "a commit after a skip-everything recovery must empty the WAL"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// #1212 — the index's counts are derived from the segment manifest.
// ---------------------------------------------------------------------------

/// A crash between a commit's manifest save and its `metadata.json` write
/// must not skew the document count once the WAL is replayed.
///
/// The crashed commit published a new segment and a deletion, but its
/// metadata checkpoint never landed, so recovery replays both records: the
/// put re-upserts over the published copy and the delete re-marks an
/// already-set bit. Running deltas then counted the replay as one more
/// document and two more deletions — 2 live instead of 3. Counts summed
/// from the manifest describe what is actually stored.
#[tokio::test(flavor = "multi_thread")]
async fn counts_stay_exact_across_a_crash_before_the_metadata_persist() -> laurus::Result<()> {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let engine = Engine::new(storage, mixed_schema()).await?;
    for i in 1..=3u64 {
        engine.put_document(&format!("id{i}"), mixed_doc(i)).await?;
    }
    engine.commit().await?;
    engine.put_document("id4", mixed_doc(4)).await?;
    engine.delete_documents("id2").await?;
    failing.fail_next_create_of("lexical/metadata.json.tmp");
    assert!(
        engine.commit().await.is_err(),
        "the injected metadata persist failure must surface"
    );
    // The process dies here: nothing else runs, not even the writer's
    // Drop-time commit retry.
    std::mem::forget(engine);

    let reopened = Engine::new(inner.clone(), mixed_schema()).await?;
    assert_eq!(reopened.stats()?.document_count, 3, "id1, id3 and id4");
    Ok(())
}

/// A crash between a commit's `.delmap` write and its manifest save leaves
/// a persisted bit the manifest does not know about. Replaying the delete
/// re-marks that already-set bit; the writer must still queue the segment,
/// or its `has_deletions` flag and deletion count are never recorded — and
/// the deleted document would be served again.
#[tokio::test(flavor = "multi_thread")]
async fn a_deletion_persisted_only_in_its_bitmap_is_recorded_on_replay() -> laurus::Result<()> {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let engine = Engine::new(storage, mixed_schema()).await?;
    for i in 1..=3u64 {
        engine.put_document(&format!("id{i}"), mixed_doc(i)).await?;
    }
    engine.commit().await?;
    engine.delete_documents("id2").await?;
    failing.fail_next_create_of("lexical/segments.json.tmp");
    assert!(
        engine.commit().await.is_err(),
        "the injected manifest save failure must surface"
    );
    std::mem::forget(engine);

    let reopened = Engine::new(inner.clone(), mixed_schema()).await?;
    assert_eq!(reopened.stats()?.document_count, 2);
    assert!(reopened.get_documents("id2").await?.is_empty());
    Ok(())
}
