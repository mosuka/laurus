//! Issue #1522: a put, add or delete writes one WAL record, so a write that
//! fails is applied whole or not at all when the index is reopened.
//!
//! A put used to append one delete record per previous version, applying
//! each, before the document's own record, and a delete one record per
//! document. When a later append failed, writes stopped (#1492), the reopen
//! cut the partial record off (#1503) and replayed the records before it: a
//! put lost its previous versions without its new one, and a delete of
//! several chunks deleted only some. A commit taking its watermark between a
//! put's records persisted the same loss.
//!
//! The tests run each write against a storage whose WAL writes can be made
//! to fail, then drop the engine as a crash would (`std::mem::forget`, which
//! skips the drop-time commit) and reopen the index on the same storage.
//! They also cover a WAL an older build wrote (separate delete and upsert
//! records), and a WAL whose version this build does not know (#1507).

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use laurus::lexical::TextOption;
use laurus::storage::memory::MemoryStorage;
use laurus::storage::{FileMetadata, LoadingMode, StorageInput, StorageOutput};
use laurus::vector::{FlatOption, QueryVector, Vector, VectorSearchQuery};
use laurus::{DataValue, Document, Engine, FieldOption, LaurusError, Result, Schema};
use laurus::{SearchRequestBuilder, Storage};

/// The WAL file, at the root of the engine's storage.
const WAL: &str = "engine.wal";

/// Which WAL write to fail: the `skip`-th (0-based) write whose bytes
/// contain `marker`, after holding it until [`Faults::release`] when `hold`
/// is set.
#[derive(Debug)]
struct WalRule {
    marker: Vec<u8>,
    skip: usize,
    hold: bool,
}

/// Which file names a fault applies to.
type NameRule = fn(&str) -> bool;

/// The faults a [`FaultStorage`] injects, shared with its outputs.
#[derive(Debug, Default)]
struct Faults {
    wal: Mutex<Option<WalRule>>,
    /// `open_input` of a name this accepts fails.
    open: Mutex<Option<NameRule>>,
    /// How many writes and opens were failed so far.
    fired: AtomicU64,
    /// Whether a WAL write is being held.
    held: AtomicBool,
    released: Mutex<bool>,
    release_signal: Condvar,
}

impl Faults {
    /// Fail the `skip`-th WAL write whose bytes contain `marker`.
    fn fail_wal_write(&self, marker: &str, skip: usize) {
        *self.wal.lock().unwrap() = Some(WalRule {
            marker: marker.as_bytes().to_vec(),
            skip,
            hold: false,
        });
    }

    /// Hold the first WAL write whose bytes contain `marker` until
    /// [`Self::release`], then fail it.
    fn hold_then_fail_wal_write(&self, marker: &str) {
        *self.wal.lock().unwrap() = Some(WalRule {
            marker: marker.as_bytes().to_vec(),
            skip: 0,
            hold: true,
        });
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.release_signal.notify_all();
    }

    fn disarm(&self) {
        *self.wal.lock().unwrap() = None;
        *self.open.lock().unwrap() = None;
    }

    fn fired(&self) -> u64 {
        self.fired.load(Ordering::SeqCst)
    }

    /// Whether the WAL write of `buf` must fail, after holding it if the
    /// rule says so.
    fn wal_write_fails(&self, buf: &[u8]) -> bool {
        let hold = {
            let mut rule = self.wal.lock().unwrap();
            let Some(r) = rule.as_mut() else {
                return false;
            };
            if !buf
                .windows(r.marker.len())
                .any(|window| window == r.marker.as_slice())
            {
                return false;
            }
            if r.skip > 0 {
                r.skip -= 1;
                return false;
            }
            let hold = r.hold;
            *rule = None;
            hold
        };
        self.fired.fetch_add(1, Ordering::SeqCst);
        if hold {
            self.held.store(true, Ordering::SeqCst);
            let mut released = self.released.lock().unwrap();
            while !*released {
                released = self.release_signal.wait(released).unwrap();
            }
        }
        true
    }
}

/// A WAL append handle whose writes [`Faults::wal_write_fails`] decides.
#[derive(Debug)]
struct FaultOutput {
    inner: Box<dyn StorageOutput>,
    faults: Arc<Faults>,
}

impl Write for FaultOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.faults.wal_write_fails(buf) {
            return Err(std::io::Error::other("injected WAL write failure"));
        }
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for FaultOutput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl StorageOutput for FaultOutput {
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

/// A [`MemoryStorage`] that injects [`Faults`]: into the appends to the WAL
/// and into `open_input`.
#[derive(Debug)]
struct FaultStorage {
    inner: Arc<dyn Storage>,
    faults: Arc<Faults>,
}

/// A fresh [`FaultStorage`], and the faults to arm on it.
fn fault_storage() -> (Arc<dyn Storage>, Arc<Faults>) {
    let faults = Arc::new(Faults::default());
    let storage = Arc::new(FaultStorage {
        inner: Arc::new(MemoryStorage::new(Default::default())),
        faults: faults.clone(),
    });
    (storage, faults)
}

impl Storage for FaultStorage {
    fn loading_mode(&self) -> LoadingMode {
        self.inner.loading_mode()
    }
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        if self
            .faults
            .open
            .lock()
            .unwrap()
            .is_some_and(|rule| rule(name))
        {
            self.faults.fired.fetch_add(1, Ordering::SeqCst);
            return Err(LaurusError::storage(format!("injected failure on {name}")));
        }
        self.inner.open_input(name)
    }
    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output(name)
    }
    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        let inner = self.inner.create_output_append(name)?;
        if name != WAL {
            return Ok(inner);
        }
        Ok(Box::new(FaultOutput {
            inner,
            faults: self.faults.clone(),
        }))
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
        self.inner.rename_file(old_name, new_name)
    }
    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }
    fn sync(&self) -> Result<()> {
        self.inner.sync()
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// An engine over `storage` with one text field, `title`.
async fn text_engine(storage: &Arc<dyn Storage>) -> Result<Engine> {
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .build();
    Engine::new(storage.clone(), schema).await
}

/// An engine over `storage` with a text field, `title`, and a 2-d flat
/// vector field, `vec`.
async fn vector_engine(storage: &Arc<dyn Storage>) -> Engine {
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("vec", FieldOption::Flat(FlatOption::default().dimension(2)))
        .build();
    Engine::new(storage.clone(), schema).await.unwrap()
}

fn title_doc(title: &str) -> Document {
    Document::builder().add_text("title", title).build()
}

fn flat_doc(title: &str, vector: [f32; 2]) -> Document {
    Document::builder()
        .add_text("title", title)
        .add_vector("vec", vector.to_vec())
        .build()
}

/// The `title` of every live version of `id`, sorted.
async fn titles(engine: &Engine, id: &str) -> Vec<String> {
    let mut titles: Vec<String> = engine
        .get_documents(id)
        .await
        .unwrap()
        .iter()
        .filter_map(|doc| match doc.fields.get("title") {
            Some(DataValue::Text(title)) => Some(title.clone()),
            _ => None,
        })
        .collect();
    titles.sort();
    titles
}

/// Whether the nearest vector hit for `vector` belongs to `id`.
async fn nearest_is(engine: &Engine, id: &str, vector: [f32; 2]) -> bool {
    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vector.to_vec()),
            weight: 1.0,
            fields: None,
        }]))
        .limit(1)
        .build();
    engine
        .search(request)
        .await
        .unwrap()
        .iter()
        .any(|hit| hit.id == id)
}

fn read_wal(storage: &Arc<dyn Storage>) -> Vec<u8> {
    let mut bytes = Vec::new();
    storage
        .open_input(WAL)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}

fn write_wal(storage: &Arc<dyn Storage>, bytes: &[u8]) {
    let mut out = storage.create_output(WAL).unwrap();
    out.write_all(bytes).unwrap();
    out.flush_and_sync().unwrap();
    out.close().unwrap();
}

/// A put that fails at a WAL append keeps every previous version, whether
/// the id has one version or two chunks, and a put that succeeds leaves
/// only the new one. The test fails each WAL write of the put in turn (the
/// `k`-th write that holds the id). A put is one record, so only the first
/// fails; the others never happen and the put succeeds. When a put wrote a
/// delete record per previous version before its own, a failure at a later
/// record left the earlier deletes for the replay: one chunk, or no version
/// at all.
#[tokio::test]
async fn a_put_that_fails_at_its_wal_append_keeps_its_previous_versions() {
    const ID: &str = "doc-1522";
    for previous in [&["alpha"][..], &["alpha-1", "alpha-2"]] {
        // A put of an id with n versions used to write n + 1 records.
        for k in 0..=previous.len() {
            let case = format!("previous {previous:?}, failing write {k}");
            let (storage, faults) = fault_storage();
            let engine = text_engine(&storage).await.unwrap();
            if let [only] = previous {
                engine.put_document(ID, title_doc(only)).await.unwrap();
            } else {
                for chunk in previous {
                    engine.add_document(ID, title_doc(chunk)).await.unwrap();
                }
            }
            engine.commit().await.unwrap();

            faults.fail_wal_write(ID, k);
            let result = engine.put_document(ID, title_doc("bravo")).await;
            let failed = result.is_err();
            if let Err(err) = &result {
                assert_eq!(faults.fired(), 1, "{case}: {err}");
                assert!(err.to_string().contains("injected"), "{case}: {err}");
                assert!(engine.write_stop_reason().is_some(), "{case}");
            }
            std::mem::forget(engine);
            faults.disarm();

            let reopened = text_engine(&storage).await.unwrap();
            let expected: Vec<&str> = if failed {
                previous.to_vec()
            } else {
                vec!["bravo"]
            };
            assert_eq!(titles(&reopened, ID).await, expected, "{case}");
            if k == 0 {
                assert!(failed, "{case}: the put's first append must fail");
            }

            // The reopened index takes writes again, numbered after the
            // records it replayed.
            reopened
                .put_document(ID, title_doc("charlie"))
                .await
                .unwrap();
            reopened.commit().await.unwrap();
            assert_eq!(titles(&reopened, ID).await, ["charlie"], "{case}");
        }
    }
}

/// The case in the Issue: a commit takes its watermark while a put's WAL
/// append is held, and the append then fails. The commit succeeds, and the
/// reopen keeps the previous version. The put used to append and apply the
/// previous version's delete before its own record, so the commit persisted
/// the delete and truncated its record, and the reopen had neither version.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_put_whose_append_fails_while_a_commit_runs_keeps_its_previous_version() {
    let (storage, faults) = fault_storage();
    let engine = Arc::new(text_engine(&storage).await.unwrap());
    engine.put_document("a", title_doc("alpha")).await.unwrap();
    engine.commit().await.unwrap();

    faults.hold_then_fail_wal_write("bravo");
    let put = tokio::spawn({
        let engine = engine.clone();
        async move { engine.put_document("a", title_doc("bravo")).await }
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    while !faults.held.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the put's append must be held");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let commit = tokio::spawn({
        let engine = engine.clone();
        async move { engine.commit().await }
    });
    // Let the commit take its watermark; it then waits for the WAL, which
    // the held append locks.
    tokio::time::sleep(Duration::from_millis(200)).await;
    faults.release();

    let err = put.await.unwrap().unwrap_err();
    assert!(err.to_string().contains("injected"), "{err}");
    commit
        .await
        .unwrap()
        .expect("the commit started before writes stopped, so it runs");
    assert_eq!(faults.fired(), 1);
    std::mem::forget(engine);
    faults.disarm();

    let reopened = text_engine(&storage).await.unwrap();
    assert_eq!(titles(&reopened, "a").await, ["alpha"]);
}

/// A delete of an id with two chunks that fails at a WAL append keeps both
/// chunks, and one that succeeds deletes both. As for a put, each WAL write
/// of the delete is failed in turn. A delete used to write one record per
/// document, so a failure at the second left the first chunk deleted.
#[tokio::test]
async fn a_delete_that_fails_at_its_wal_append_keeps_every_chunk() {
    const ID: &str = "doc-1522";
    for k in 0..2 {
        let (storage, faults) = fault_storage();
        let engine = text_engine(&storage).await.unwrap();
        engine.add_document(ID, title_doc("one")).await.unwrap();
        engine.add_document(ID, title_doc("two")).await.unwrap();
        engine.commit().await.unwrap();

        faults.fail_wal_write(ID, k);
        let failed = engine.delete_documents(ID).await.is_err();
        std::mem::forget(engine);
        faults.disarm();

        let reopened = text_engine(&storage).await.unwrap();
        let expected: &[&str] = if failed { &["one", "two"] } else { &[] };
        assert_eq!(titles(&reopened, ID).await, expected, "failing write {k}");
        if k == 0 {
            assert!(failed, "the delete's first append must fail");
        }
    }
}

/// A put that replaces a version and fails after its WAL append, here
/// loading the vector field's deletion bitmap, is applied by the reopen:
/// the previous version is gone from both stores, and the new one is in
/// both. The put's one record names the version it replaces, and the replay
/// deletes it with the insert (Issue #1522).
#[tokio::test]
async fn a_put_that_fails_after_its_wal_append_is_applied_whole_on_reopen() {
    fn is_vector_delmap(name: &str) -> bool {
        name.starts_with("vector/") && name.ends_with(".delmap")
    }

    let (storage, faults) = fault_storage();
    let engine = vector_engine(&storage).await;
    engine
        .put_document("b", flat_doc("bravo", [0.0, 1.0]))
        .await
        .unwrap();
    // Enough documents in the segment that deleting one leaves a bitmap
    // rather than a compacted segment.
    for i in 0..20 {
        engine
            .put_document(&format!("fill-{i}"), flat_doc("fill", [0.7, 0.7]))
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    engine.delete_documents("fill-0").await.unwrap();
    engine.commit().await.unwrap();
    drop(engine);
    assert!(
        storage
            .list_files()
            .unwrap()
            .iter()
            .any(|name| is_vector_delmap(name)),
        "precondition: the vector field has a deletion bitmap"
    );

    // Nothing may load the bitmap before the put: no search, no stats.
    let engine = vector_engine(&storage).await;
    *faults.open.lock().unwrap() = Some(is_vector_delmap);
    let err = engine
        .put_document("b", flat_doc("bravo-2", [-1.0, 0.0]))
        .await
        .unwrap_err();
    assert!(faults.fired() > 0, "the injected failure must fire: {err}");
    assert!(err.to_string().contains("injected"), "{err}");
    assert!(engine.write_stop_reason().is_some());
    std::mem::forget(engine);
    faults.disarm();

    let reopened = vector_engine(&storage).await;
    assert_eq!(titles(&reopened, "b").await, ["bravo-2"]);
    assert!(
        nearest_is(&reopened, "b", [-1.0, 0.0]).await,
        "the new version's vector is in the vector store"
    );
    assert!(
        !nearest_is(&reopened, "b", [0.0, 1.0]).await,
        "the previous version's vector is gone from the vector store"
    );
}

/// A WAL a build before Issue #1522 wrote (`b757077c`), in v3: over a text
/// engine that had put `a` (`alpha`) and committed, a put of `a` (`bravo`)
/// as a delete record and an upsert record, an add of `c` (`charlie`), and
/// a delete of `c`, then a crash. It recovers as it did in that build, and
/// writes after it go on in v4.
#[tokio::test]
async fn a_wal_written_by_an_older_build_recovers_as_before() {
    let (storage, _faults) = fault_storage();
    let engine = text_engine(&storage).await.unwrap();
    engine.put_document("a", title_doc("alpha")).await.unwrap();
    engine.commit().await.unwrap();
    std::mem::forget(engine);
    write_wal(&storage, &hex_bytes(V3_WAL_FROM_B757077C));

    let reopened = text_engine(&storage).await.unwrap();
    assert_eq!(titles(&reopened, "a").await, ["bravo"]);
    assert!(titles(&reopened, "c").await.is_empty());

    reopened
        .put_document("a", title_doc("delta"))
        .await
        .unwrap();
    assert_eq!(read_wal(&storage)[..5], *b"LWAL\x04", "the WAL is v4");
    std::mem::forget(reopened);
    let reopened = text_engine(&storage).await.unwrap();
    assert_eq!(titles(&reopened, "a").await, ["delta"]);
}

/// Issue #1507: a WAL whose header names a version this build does not
/// know, which a newer build wrote, fails the open with an error that names
/// the version, and the file is left as it is. This build used to read it
/// as v2, recover nothing, append behind its records and truncate them at
/// the next commit.
#[tokio::test]
async fn a_wal_of_an_unknown_version_fails_the_open_and_is_left_as_it_is() {
    let (storage, _faults) = fault_storage();
    let engine = text_engine(&storage).await.unwrap();
    engine.put_document("a", title_doc("alpha")).await.unwrap();
    engine.commit().await.unwrap();
    drop(engine);
    // A version-5 header and one CRC-framed record this build cannot read.
    let body = b"a record of a newer build";
    let len = (body.len() as u32).to_le_bytes();
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&len);
    hasher.update(body);
    let bytes = [
        &b"LWAL\x05"[..],
        &len,
        &hasher.finalize().to_le_bytes(),
        body,
    ]
    .concat();
    write_wal(&storage, &bytes);

    let err = match text_engine(&storage).await {
        Ok(_) => panic!("the open must fail"),
        Err(err) => err,
    };
    assert!(
        matches!(&err, LaurusError::IncompatibleFormat(message) if message.contains("format version 5")),
        "{err}"
    );
    assert_eq!(read_wal(&storage), bytes);
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

/// `engine.wal` as `b757077c` left it in
/// `a_wal_written_by_an_older_build_recovers_as_before`'s scenario: 533
/// bytes, a v3 header and four records, seq 2 `Delete { doc_id: 1, "a" }`,
/// seq 3 `Upsert { doc_id: 2, "a", bravo }`, seq 4 `Upsert { doc_id: 3,
/// "c", charlie }` and seq 5 `Delete { doc_id: 3, "c" }`.
const V3_WAL_FROM_B757077C: &str = "4c57414c03300000004e94af6b02000000000000000100000000000000010000000000000061ffffffffffffff00000000000000000000000000000000c8000000bcc9e70b7469746c65ffffff04000000627261766fffffff0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000005f6964ffffffffff0400000061ffffffffffffff000000000000000000000000000000000000000003ff3403ff34ffffffffffffffffffffffffffffffffffffffffffffffffff0003000000000000000000000000000000020000000000000061ffffffffffffffc0ffffff020000000300000000000000c80000005e6615cb7469746c65ffffff04000000636861726c6965ff0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000005f6964ffffffffff0400000063ffffffffffffff000000000000000000000000000000000000000003ff3403ff34ffffffffffffffffffffffffffffffffffffffffffffffffff0004000000000000000000000000000000030000000000000063ffffffffffffffc0ffffff0200000003000000000000003000000040d3443a05000000000000000100000000000000030000000000000063ffffffffffffff00000000000000000000000000000000";
