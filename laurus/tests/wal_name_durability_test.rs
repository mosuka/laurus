//! Issue #1566: a write is acknowledged only once the name of the WAL file
//! it went to survives a crash.
//!
//! The document log replaces the WAL by renaming a new file over it (a
//! commit's truncation that a write raced, the upgrade of an older framing,
//! and the cut of an incomplete record on open), and creates it on the
//! first append to a new index. Such a change of the directory is durable
//! only once the storage is synced: fsyncing the file, as every append
//! does, does not cover it. When the sync after a rename failed, the next
//! append went to the renamed file and was acknowledged, and a crash that
//! brought back the replaced file lost it. A reopened log, which cannot know
//! what an earlier one left, did the same.
//!
//! The tests run on [`CrashModelStorage`], which models what a crash
//! undoes: every change of the WAL's name that no successful sync has
//! followed yet.

use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use laurus::lexical::TextOption;
use laurus::storage::memory::MemoryStorage;
use laurus::storage::{FileMetadata, LoadingMode, StorageInput, StorageOutput};
use laurus::store::log::{DocumentLog, LogEntry};
use laurus::{DataValue, Document, Engine, FieldOption, LaurusError, Result, Schema, Storage};

/// The WAL file of the [`DocumentLog`] tests.
const LOG_WAL: &str = "test.wal";

/// The engine's WAL file, at the root of its storage.
const ENGINE_WAL: &str = "engine.wal";

/// A v4 WAL header (`LWAL`, version 4) followed by three bytes that do not
/// form a record's length prefix, as a crash partway through an append
/// leaves it: opening the log cuts the file back to the header.
const TORN_WAL: &[u8] = b"LWAL\x04\xde\xad\xbe";

/// What the WAL's name held before its oldest change that no successful
/// sync has followed.
#[derive(Debug, Clone)]
enum Before {
    /// Nothing: the change created the file.
    Absent,
    /// These bytes: the change renamed another file over them.
    Bytes(Vec<u8>),
}

/// The state of a [`CrashModelStorage`], shared with the test.
#[derive(Debug, Default)]
struct Model {
    /// The name's state before its oldest unsynced change, if one is
    /// pending.
    pending: Option<Before>,
    /// Whether `sync` fails while a change is pending.
    fail_sync: bool,
    /// How many syncs succeeded.
    syncs: usize,
}

/// A [`MemoryStorage`] that models which changes of one file's name, the
/// WAL's, a crash undoes.
///
/// A rename onto the name, and the creation of the file under it, are
/// pending until a `sync` succeeds, and [`Self::crash`] undoes a pending
/// one: it brings back the bytes the name held, or removes the file. File
/// contents are durable as soon as they are written, since the log fsyncs
/// the WAL file on every append anyway. It has no lock manager, so that an
/// engine dropped with `std::mem::forget`, as in a crash, does not keep the
/// index locked.
#[derive(Debug)]
struct CrashModelStorage {
    inner: Arc<dyn Storage>,
    name: String,
    model: Mutex<Model>,
}

impl CrashModelStorage {
    /// A model over a new [`MemoryStorage`] for the file `name`.
    fn new(name: &str) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(MemoryStorage::new(Default::default())),
            name: name.to_string(),
            model: Mutex::new(Model::default()),
        })
    }

    /// Record that the file is about to be created under the name, unless
    /// it exists or an older change is pending.
    fn note_creation(&self, name: &str) {
        if name == self.name && !self.inner.file_exists(name) {
            let mut model = self.model.lock().unwrap();
            model.pending.get_or_insert(Before::Absent);
        }
    }

    /// The bytes of `name` in the wrapped storage.
    fn read(&self, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.inner
            .open_input(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    /// Write `bytes` as the file `name`, directly to the wrapped storage,
    /// so the model records nothing.
    fn write_raw(&self, name: &str, bytes: &[u8]) {
        let mut out = self.inner.create_output(name).unwrap();
        out.write_all(bytes).unwrap();
        out.close().unwrap();
    }

    /// Make `sync` fail while a change is pending, or succeed again.
    fn fail_sync(&self, fail: bool) {
        self.model.lock().unwrap().fail_sync = fail;
    }

    /// Whether a change of the name is pending.
    fn pending(&self) -> bool {
        self.model.lock().unwrap().pending.is_some()
    }

    /// How many syncs succeeded.
    fn syncs(&self) -> usize {
        self.model.lock().unwrap().syncs
    }

    /// Undo the pending change of the name, as a crash does. A writer still
    /// open on the file would publish it again when it is dropped, so the
    /// log or engine must be dropped, or forgotten, before this.
    fn crash(&self) {
        let pending = self.model.lock().unwrap().pending.take();
        match pending {
            None => {}
            Some(Before::Absent) => {
                if self.inner.file_exists(&self.name) {
                    self.inner.delete_file(&self.name).unwrap();
                }
            }
            Some(Before::Bytes(bytes)) => self.write_raw(&self.name, &bytes),
        }
    }
}

impl Storage for CrashModelStorage {
    fn loading_mode(&self) -> LoadingMode {
        self.inner.loading_mode()
    }
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }
    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.note_creation(name);
        self.inner.create_output(name)
    }
    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.note_creation(name);
        self.inner.create_output_append(name)
    }
    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }
    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(name)
    }
    fn list_files(&self) -> Result<Vec<String>> {
        self.inner.list_files()
    }
    fn file_size(&self, name: &str) -> Result<u64> {
        self.inner.file_size(name)
    }
    fn metadata(&self, name: &str) -> Result<FileMetadata> {
        self.inner.metadata(name)
    }
    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        if new_name != self.name {
            return self.inner.rename_file(old_name, new_name);
        }
        let before = if self.inner.file_exists(new_name) {
            Before::Bytes(self.read(new_name))
        } else {
            Before::Absent
        };
        self.inner.rename_file(old_name, new_name)?;
        self.model.lock().unwrap().pending.get_or_insert(before);
        Ok(())
    }
    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }
    fn sync(&self) -> Result<()> {
        let mut model = self.model.lock().unwrap();
        if model.fail_sync && model.pending.is_some() {
            return Err(LaurusError::storage(format!(
                "injected sync failure while the name of '{}' is pending",
                self.name
            )));
        }
        self.inner.sync()?;
        model.pending = None;
        model.syncs += 1;
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A document log on `storage`, with its WAL at [`LOG_WAL`] and its
/// document store in a storage of its own.
fn open_log(storage: &Arc<CrashModelStorage>) -> DocumentLog {
    let doc_store: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
    DocumentLog::new(storage.clone(), LOG_WAL, doc_store).unwrap()
}

/// A document with one text field.
fn doc(text: &str) -> Document {
    Document::builder()
        .add_field("title", DataValue::Text(text.to_string()))
        .build()
}

/// The external ids of the upserts a new log on `storage` reads back.
fn upserted_ids(storage: &Arc<CrashModelStorage>) -> Vec<String> {
    let log = open_log(storage);
    log.read_all()
        .unwrap()
        .into_iter()
        .filter_map(|record| match record.entry {
            LogEntry::Upsert { external_id, .. } => Some(external_id),
            LogEntry::Delete { .. } => None,
        })
        .collect()
}

/// A log on `storage` holding the upserts `a`, `b` and `c` (sequence
/// numbers 1 to 3), whose commit then failed at the sync after it renamed
/// the truncated WAL over the old one: a write (`c`) raced the commit, so
/// the truncation kept the records after 1 in a new file.
fn log_after_a_failed_rewrite(storage: &Arc<CrashModelStorage>) -> DocumentLog {
    let log = open_log(storage);
    log.read_all().unwrap();
    for id in ["a", "b", "c"] {
        log.append(id, &doc(id)).unwrap();
    }
    assert!(
        !storage.pending(),
        "precondition: the WAL's creation synced"
    );

    storage.fail_sync(true);
    assert!(log.truncate_retaining_after(1).is_err());
    assert!(storage.pending(), "precondition: the rename is not durable");
    log
}

/// While the sync keeps failing after a commit's rewrite of the WAL, an
/// append is refused instead of being acknowledged on a file a crash can
/// take back.
#[test]
fn an_append_after_a_failed_rewrite_fails_while_the_sync_fails() {
    let storage = CrashModelStorage::new(LOG_WAL);
    let log = log_after_a_failed_rewrite(&storage);

    assert!(
        log.append("d", &doc("d")).is_err(),
        "an append must not be acknowledged before the rename is durable"
    );
    assert!(storage.pending());
}

/// Once the sync works again, the next append syncs the storage before it
/// is acknowledged, so a crash keeps it.
#[test]
fn an_append_after_a_failed_rewrite_survives_a_crash() {
    let storage = CrashModelStorage::new(LOG_WAL);
    let log = log_after_a_failed_rewrite(&storage);

    storage.fail_sync(false);
    log.append("d", &doc("d")).unwrap();
    drop(log);
    storage.crash();

    let ids = upserted_ids(&storage);
    assert!(ids.contains(&"d".to_string()), "lost the append: {ids:?}");
}

/// The first append to a WAL in an older framing rewrites it in v4 and
/// renames it over the old one. When the sync after that rename fails, the
/// append fails, and the next one syncs before it is acknowledged.
#[test]
fn an_append_after_a_failed_upgrade_survives_a_crash() {
    let storage = CrashModelStorage::new(LOG_WAL);
    // A legacy WAL (no header, `[u32 len][JSON]` frames) with one delete.
    let json = br#"{"seq":1,"entry":{"Delete":{"doc_id":7,"external_id":"x"}}}"#;
    let mut legacy = (json.len() as u32).to_le_bytes().to_vec();
    legacy.extend_from_slice(json);
    storage.write_raw(LOG_WAL, &legacy);

    let log = open_log(&storage);
    assert_eq!(log.read_all().unwrap().len(), 1, "precondition");
    storage.fail_sync(true);
    assert!(log.append("a", &doc("a")).is_err());
    assert!(storage.pending(), "precondition: the rename is not durable");

    storage.fail_sync(false);
    log.append("b", &doc("b")).unwrap();
    drop(log);
    storage.crash();

    let ids = upserted_ids(&storage);
    assert!(ids.contains(&"b".to_string()), "lost the append: {ids:?}");
}

/// When the sync after an open's cut of an incomplete record fails, the
/// open fails. A log opened again finds nothing to cut, and cannot tell
/// that the cut is not durable, so its first append syncs the storage.
#[test]
fn the_first_append_of_a_new_log_after_a_failed_cut_survives_a_crash() {
    let storage = CrashModelStorage::new(LOG_WAL);
    storage.write_raw(LOG_WAL, TORN_WAL);

    storage.fail_sync(true);
    let log = open_log(&storage);
    assert!(log.read_all().is_err());
    drop(log);
    assert!(storage.pending(), "precondition: the cut is not durable");

    storage.fail_sync(false);
    let log = open_log(&storage);
    assert!(log.read_all().unwrap().is_empty());
    assert!(storage.pending(), "precondition: nothing synced the cut");
    log.append("a", &doc("a")).unwrap();
    drop(log);
    storage.crash();

    let ids = upserted_ids(&storage);
    assert!(ids.contains(&"a".to_string()), "lost the append: {ids:?}");
}

/// The first append to a new log creates the WAL file, whose name only a
/// storage sync makes durable.
#[test]
fn the_first_append_that_creates_the_wal_survives_a_crash() {
    let storage = CrashModelStorage::new(LOG_WAL);
    let log = open_log(&storage);
    assert!(log.read_all().unwrap().is_empty());

    log.append("a", &doc("a")).unwrap();
    drop(log);
    storage.crash();

    let ids = upserted_ids(&storage);
    assert!(ids.contains(&"a".to_string()), "lost the append: {ids:?}");
}

/// A cut by a `read_all` after appends, whose sync fails, is not durable
/// either, so the next append syncs the storage first, although an earlier
/// append of this log already did.
#[test]
fn an_append_after_a_later_failed_cut_survives_a_crash() {
    let storage = CrashModelStorage::new(LOG_WAL);
    let log = open_log(&storage);
    log.read_all().unwrap();
    log.append("a", &doc("a")).unwrap();
    log.truncate().unwrap();
    storage.write_raw(LOG_WAL, TORN_WAL);
    assert!(!storage.pending(), "precondition");

    storage.fail_sync(true);
    assert!(log.read_all().is_err());
    assert!(storage.pending(), "precondition: the cut is not durable");

    storage.fail_sync(false);
    log.append("b", &doc("b")).unwrap();
    drop(log);
    storage.crash();

    let ids = upserted_ids(&storage);
    assert!(ids.contains(&"b".to_string()), "lost the append: {ids:?}");
}

/// The storage is synced once before the first append of a log, and then
/// only when the WAL is replaced: appends to an open writer, and the
/// appends after a commit's truncation, whose own sync succeeded, add none.
#[test]
fn a_log_syncs_the_storage_only_when_the_wal_name_may_not_be_durable() {
    let storage = CrashModelStorage::new(LOG_WAL);
    let log = open_log(&storage);
    log.read_all().unwrap();
    assert_eq!(storage.syncs(), 0);

    log.append("a", &doc("a")).unwrap();
    assert_eq!(storage.syncs(), 1, "the first append syncs");
    log.append("b", &doc("b")).unwrap();
    assert_eq!(storage.syncs(), 1, "an open writer does not sync");

    // The fast path of a commit's truncation syncs once itself.
    log.truncate().unwrap();
    assert_eq!(storage.syncs(), 2);
    log.append("c", &doc("c")).unwrap();
    assert_eq!(storage.syncs(), 2, "after the fast path");

    // The slow path renames the kept records over the WAL and syncs.
    log.append("d", &doc("d")).unwrap();
    log.truncate_retaining_after(log.last_seq() - 1).unwrap();
    assert_eq!(storage.syncs(), 3);
    log.append("e", &doc("e")).unwrap();
    assert_eq!(storage.syncs(), 3, "after the slow path");
    drop(log);

    // An open whose cut synced needs no other sync before its first append.
    storage.write_raw(LOG_WAL, TORN_WAL);
    let log = open_log(&storage);
    log.read_all().unwrap();
    assert_eq!(storage.syncs(), 4, "the cut syncs");
    log.append("f", &doc("f")).unwrap();
    assert_eq!(storage.syncs(), 4, "after a cut that synced");
}

/// An engine over `storage` with one text field, `title`.
async fn text_engine(storage: &Arc<CrashModelStorage>) -> Result<Engine> {
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .build();
    Engine::new(storage.clone(), schema).await
}

/// How many documents `engine` holds under `id`.
async fn count(engine: &Engine, id: &str) -> usize {
    engine.get_documents(id).await.unwrap().len()
}

/// The first put to a new index creates its WAL, and a crash right after
/// the put returned keeps it.
#[tokio::test]
async fn the_first_put_to_a_new_index_survives_a_crash() {
    let storage = CrashModelStorage::new(ENGINE_WAL);
    let engine = text_engine(&storage).await.unwrap();
    engine.put_document("a", doc("alpha")).await.unwrap();
    std::mem::forget(engine);
    storage.crash();

    let reopened = text_engine(&storage).await.unwrap();
    assert_eq!(count(&reopened, "a").await, 1, "lost the put");
}

/// An index whose open failed at the sync after the cut of an incomplete
/// WAL record opens again, and the first put of that engine survives a
/// crash, although nothing in the second open synced the cut.
#[tokio::test]
async fn the_first_put_after_an_open_that_failed_to_sync_its_cut_survives_a_crash() {
    let storage = CrashModelStorage::new(ENGINE_WAL);
    storage.write_raw(ENGINE_WAL, TORN_WAL);

    storage.fail_sync(true);
    assert!(text_engine(&storage).await.is_err());
    assert!(storage.pending(), "precondition: the cut is not durable");

    storage.fail_sync(false);
    let engine = text_engine(&storage).await.unwrap();
    assert!(storage.pending(), "precondition: the open synced nothing");
    engine.put_document("a", doc("alpha")).await.unwrap();
    std::mem::forget(engine);
    storage.crash();

    let reopened = text_engine(&storage).await.unwrap();
    assert_eq!(count(&reopened, "a").await, 1, "lost the put");
}
