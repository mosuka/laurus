//! Issue #1588: a temporary file of a replacement of the WAL never stays
//! behind.
//!
//! The document log replaces the WAL by writing a new file under a
//! temporary name (`<wal_path>_<n>.tmp`) and renaming it over the WAL: a
//! commit's rewrite of the WAL that a write raced, the upgrade of an older
//! framing, and the cut of an incomplete record on open. A failure before
//! the rename, or a crash, left the temporary file, and each failure left
//! one more. Now a failure deletes it, and opening the log deletes one that
//! a crash left.
//!
//! The tests run on [`FaultStorage`], a [`MemoryStorage`] that can fail
//! the writes to a temporary file, renames and deletions, and on a
//! `FileStorage` engine.

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use laurus::lexical::TextOption;
use laurus::storage::file::{FileStorage, FileStorageConfig};
use laurus::storage::memory::MemoryStorage;
use laurus::storage::{FileMetadata, LoadingMode, StorageInput, StorageOutput};
use laurus::store::log::{DocumentLog, LogEntry};
use laurus::{DataValue, Document, Engine, FieldOption, LaurusError, Result, Schema, Storage};

/// The WAL file of the [`DocumentLog`] tests.
const LOG_WAL: &str = "test.wal";

/// A v4 WAL header (`LWAL`, version 4) with no record: an empty WAL.
const EMPTY_WAL: &[u8] = b"LWAL\x04";

/// A v4 WAL header followed by three bytes that do not form a record's
/// length prefix, as a crash partway through an append leaves it: opening
/// the log cuts the file back to the header.
const TORN_WAL: &[u8] = b"LWAL\x04\xde\xad\xbe";

/// An output whose writes all fail, wrapping the temporary file's real
/// output, which publishes the file when it is dropped.
#[derive(Debug)]
struct FailingOutput {
    inner: Box<dyn StorageOutput>,
}

impl Write for FailingOutput {
    fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("injected write failure"))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for FailingOutput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl StorageOutput for FailingOutput {
    fn flush_and_sync(&mut self) -> Result<()> {
        self.inner.flush_and_sync()
    }
    fn position(&self) -> Result<u64> {
        self.inner.position()
    }
    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }
}

/// A [`MemoryStorage`] that fails, on request, the writes to a temporary
/// file, every rename, and the deletion of a temporary file.
#[derive(Debug)]
struct FaultStorage {
    inner: Arc<dyn Storage>,
    fail_temp_write: AtomicBool,
    fail_rename: AtomicBool,
    fail_temp_delete: AtomicBool,
}

impl FaultStorage {
    /// A storage over a new [`MemoryStorage`] that fails nothing.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(MemoryStorage::new(Default::default())),
            fail_temp_write: AtomicBool::new(false),
            fail_rename: AtomicBool::new(false),
            fail_temp_delete: AtomicBool::new(false),
        })
    }

    /// Write `bytes` as the file `name`, directly to the wrapped storage.
    fn write_raw(&self, name: &str, bytes: &[u8]) {
        let mut out = self.inner.create_output(name).unwrap();
        out.write_all(bytes).unwrap();
        out.close().unwrap();
    }

    /// The bytes of `name`.
    fn read(&self, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.inner
            .open_input(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    /// The names of every temporary file in the storage.
    fn temp_files(&self) -> Vec<String> {
        self.inner
            .list_files()
            .unwrap()
            .into_iter()
            .filter(|name| name.ends_with(".tmp"))
            .collect()
    }
}

impl Storage for FaultStorage {
    fn loading_mode(&self) -> LoadingMode {
        self.inner.loading_mode()
    }
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }
    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output(name)
    }
    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }
    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }
    fn delete_file(&self, name: &str) -> Result<()> {
        if name.ends_with(".tmp") && self.fail_temp_delete.load(Ordering::SeqCst) {
            return Err(LaurusError::storage(format!(
                "injected failure to delete '{name}'"
            )));
        }
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
        if self.fail_rename.load(Ordering::SeqCst) {
            return Err(LaurusError::storage(format!(
                "injected failure to rename '{old_name}' to '{new_name}'"
            )));
        }
        self.inner.rename_file(old_name, new_name)
    }
    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        let (name, out) = self.inner.create_temp_output(prefix)?;
        if self.fail_temp_write.load(Ordering::SeqCst) {
            return Ok((name, Box::new(FailingOutput { inner: out })));
        }
        Ok((name, out))
    }
    fn sync(&self) -> Result<()> {
        self.inner.sync()
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A document log on `storage`, with its WAL at [`LOG_WAL`] and its
/// document store in a storage of its own.
fn open_log(storage: &Arc<FaultStorage>) -> DocumentLog {
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
fn upserted_ids(storage: &Arc<FaultStorage>) -> Vec<String> {
    open_log(storage)
        .read_all()
        .unwrap()
        .into_iter()
        .filter_map(|record| match record.entry {
            LogEntry::Upsert { external_id, .. } => Some(external_id),
            LogEntry::Delete { .. } => None,
        })
        .collect()
}

/// A log on `storage` holding the upserts `a`, `b` and `c` (sequence
/// numbers 1 to 3), so that `truncate_retaining_after(1)` rewrites the WAL
/// with `b` and `c` through a temporary file, as a commit that a write
/// raced does.
fn log_with_three_upserts(storage: &Arc<FaultStorage>) -> DocumentLog {
    let log = open_log(storage);
    log.read_all().unwrap();
    for id in ["a", "b", "c"] {
        log.append(id, &doc(id)).unwrap();
    }
    log
}

/// A commit's rewrite of the WAL that fails writing the temporary file
/// returns the error and deletes the file, and the WAL keeps its records.
#[test]
fn a_rewrite_that_fails_writing_deletes_its_temporary_file() {
    let storage = FaultStorage::new();
    let log = log_with_three_upserts(&storage);

    storage.fail_temp_write.store(true, Ordering::SeqCst);
    let rewrite = log.truncate_retaining_after(1);
    assert!(rewrite.is_err(), "the rewrite must fail: {rewrite:?}");
    assert_eq!(
        storage.temp_files(),
        Vec::<String>::new(),
        "a failed rewrite must delete its temporary file"
    );

    storage.fail_temp_write.store(false, Ordering::SeqCst);
    drop(log);
    assert_eq!(upserted_ids(&storage), ["a", "b", "c"]);
}

/// A commit's rewrite of the WAL whose rename fails returns the error and
/// deletes the temporary file, and the WAL keeps its records.
#[test]
fn a_rewrite_whose_rename_fails_deletes_its_temporary_file() {
    let storage = FaultStorage::new();
    let log = log_with_three_upserts(&storage);

    storage.fail_rename.store(true, Ordering::SeqCst);
    let rewrite = log.truncate_retaining_after(1);
    assert!(rewrite.is_err(), "the rewrite must fail: {rewrite:?}");
    assert_eq!(
        storage.temp_files(),
        Vec::<String>::new(),
        "a failed rewrite must delete its temporary file"
    );

    storage.fail_rename.store(false, Ordering::SeqCst);
    drop(log);
    assert_eq!(upserted_ids(&storage), ["a", "b", "c"]);
}

/// Opening a log on a WAL that ends in an incomplete record cuts it; when
/// the cut's rename fails, the open returns the error, deletes the
/// temporary file and leaves the WAL as it was.
#[test]
fn a_cut_whose_rename_fails_deletes_its_temporary_file() {
    let storage = FaultStorage::new();
    storage.write_raw(LOG_WAL, TORN_WAL);

    storage.fail_rename.store(true, Ordering::SeqCst);
    let read = open_log(&storage).read_all();
    assert!(read.is_err(), "the cut must fail: {read:?}");
    assert_eq!(
        storage.temp_files(),
        Vec::<String>::new(),
        "a failed cut must delete its temporary file"
    );
    assert_eq!(storage.read(LOG_WAL), TORN_WAL, "the WAL must be unchanged");
}

/// Opening a log deletes the temporary files that an earlier replacement
/// of its WAL left, as a crash before the rename does, and no other file.
#[test]
fn opening_a_log_deletes_only_its_wal_s_leftover_temporary_files() {
    let storage = FaultStorage::new();
    storage.write_raw(LOG_WAL, EMPTY_WAL);
    let leftovers = ["test.wal_0.tmp", "test.wal_12.tmp"];
    // Not a temporary file of this WAL: no number, an empty number, and
    // another file's temporary file.
    let others = ["test.wal_x.tmp", "test.wal_.tmp", "other.wal_0.tmp"];
    for name in leftovers.iter().chain(&others) {
        storage.write_raw(name, EMPTY_WAL);
    }

    open_log(&storage).read_all().unwrap();

    let mut wrong = Vec::new();
    for name in leftovers {
        if storage.file_exists(name) {
            wrong.push(format!("{name} was kept"));
        }
    }
    for name in others.iter().chain(&[LOG_WAL]) {
        if !storage.file_exists(name) {
            wrong.push(format!("{name} was deleted"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:?}");
}

/// Deleting a leftover temporary file is best-effort: when it fails, the
/// log still opens.
#[test]
fn a_leftover_temporary_file_that_cannot_be_deleted_does_not_fail_the_open() {
    let storage = FaultStorage::new();
    storage.write_raw(LOG_WAL, EMPTY_WAL);
    storage.write_raw("test.wal_0.tmp", EMPTY_WAL);

    storage.fail_temp_delete.store(true, Ordering::SeqCst);
    let read = open_log(&storage).read_all();
    assert!(read.is_ok(), "the open must succeed: {read:?}");
    assert!(
        storage.file_exists("test.wal_0.tmp"),
        "precondition: the deletion failed"
    );
}

/// An engine on a `FileStorage` deletes the WAL's leftover temporary files
/// when it opens.
#[tokio::test]
async fn opening_an_engine_deletes_the_wal_s_leftover_temporary_files() {
    // Declared before the engine, so that the engine is dropped first.
    let dir = tempfile::TempDir::new().unwrap();
    let leftovers = ["engine.wal_0.tmp", "engine.wal_3.tmp"];
    for name in leftovers {
        std::fs::write(dir.path().join(name), EMPTY_WAL).unwrap();
    }
    let storage: Arc<dyn Storage> =
        Arc::new(FileStorage::new(dir.path(), FileStorageConfig::new(dir.path())).unwrap());
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .build();

    let engine = Engine::new(storage, schema).await.unwrap();

    let kept: Vec<_> = leftovers
        .iter()
        .filter(|name| dir.path().join(name).exists())
        .collect();
    assert!(kept.is_empty(), "kept {kept:?}");
    drop(engine);
}
