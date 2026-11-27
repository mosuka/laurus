//! Crash-injection tests for the WAL / `last_wal_seq` co-durability ladder
//! (Issue #821, follow-up to #542 group commit).
//!
//! [`Engine::commit`] persists state in a fixed order:
//!
//! 1. `flush_wal()` — force the WAL durable (the #817 barrier).
//! 2. `lexical.commit()` — materialize + fsync the lexical store (this is where
//!    `last_wal_seq` is persisted).
//! 3. `vector.commit()` — materialize + fsync the vector store.
//! 4. `commit_documents()` — materialize + fsync the document store.
//! 5. `truncate()` — replace the WAL with an empty, fsync'd file.
//!
//! The invariant under test: a crash at *any* step never loses an acknowledged
//! record. Because the WAL is forced durable (step 1) before any store persists
//! its `last_wal_seq` (step 2+), and every store is fully fsync'd before the WAL
//! is truncated (step 5), recovery can always replay whatever a partial commit
//! left behind. Replay is idempotent (keyed by the recorded `doc_id`) and each
//! store tracks its own `last_wal_seq`, so a commit that fails midway recovers
//! cleanly with no lost or duplicated documents — under both the per-record and
//! the group-commit sync policies.
//!
//! These tests simulate a crash with [`FaultyStorage`], which makes a chosen
//! storage write fail at a precise point in the ladder. The engine is then
//! dropped (the crash) and re-opened on the same underlying data to drive WAL
//! recovery, mirroring the "drop without commit, then re-open" pattern in
//! `wal_recovery_test.rs`.

use std::sync::{Arc, Mutex};

use tempfile::TempDir;

use laurus::lexical::{TermQuery, TextOption};
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{
    FileMetadata, LoadingMode, Storage, StorageConfig, StorageFactory, StorageInput, StorageOutput,
};
use laurus::vector::FlatOption;
use laurus::{
    DataValue, Document, Engine, FieldOption, LaurusError, LexicalSearchQuery, Schema,
    SearchRequestBuilder, WalSyncPolicy,
};

/// Number of documents indexed per scenario.
const DOC_COUNT: usize = 5;

/// A shared term present in every document's `title`, used to count recovered
/// documents via a lexical search.
const SHARED_TERM: &str = "rust";

/// One-shot fault plan shared between the test body and the [`FaultyStorage`]
/// wrapper. While `armed`, the first file-creating operation whose (mapped) name
/// contains `target` fails once and records that it fired in `triggered`.
#[derive(Debug)]
struct FaultPlan {
    /// Whether the fault is active. The test arms it only right before the
    /// `commit()` under test, so engine setup and document indexing run cleanly.
    armed: bool,
    /// Substring matched against the underlying (prefixed) file name, e.g.
    /// `"lexical/"`, `"vector/"`, `"documents/"`, or `"engine.wal"`.
    target: String,
    /// Set to `true` once the fault has fired, so it injects exactly one failure.
    triggered: bool,
}

/// A [`Storage`] decorator that injects a single write failure at a chosen
/// point in the commit ladder, then delegates everything else to `inner`.
///
/// Only the file-*creating* operations (`create_output` / `create_temp_output`)
/// are intercepted; reads, syncs, renames, and the WAL append path pass through
/// untouched. Failing on the first creation whose name matches `target` models
/// a crash at the very start of a commit step, before that step has persisted
/// anything — the cleanest interleaving to reason about.
#[derive(Debug)]
struct FaultyStorage {
    /// The real storage backend that survives the simulated crash.
    inner: Arc<dyn Storage>,
    /// The shared, test-controlled fault plan.
    plan: Arc<Mutex<FaultPlan>>,
}

impl FaultyStorage {
    /// Returns `Err` if the fault is armed, untriggered, and `name` matches the
    /// configured target; otherwise `Ok(())`. Marks the fault as fired so it
    /// injects exactly one failure.
    fn maybe_fail(&self, name: &str) -> laurus::Result<()> {
        let mut plan = self
            .plan
            .lock()
            .map_err(|e| LaurusError::Storage(format!("fault plan lock poisoned: {e}")))?;
        if plan.armed && !plan.triggered && name.contains(&plan.target) {
            plan.triggered = true;
            return Err(LaurusError::Storage(format!(
                "injected crash before writing '{name}'"
            )));
        }
        Ok(())
    }
}

impl Storage for FaultyStorage {
    fn loading_mode(&self) -> LoadingMode {
        self.inner.loading_mode()
    }

    fn open_input(&self, name: &str) -> laurus::Result<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }

    fn create_output(&self, name: &str) -> laurus::Result<Box<dyn StorageOutput>> {
        self.maybe_fail(name)?;
        self.inner.create_output(name)
    }

    fn create_output_append(&self, name: &str) -> laurus::Result<Box<dyn StorageOutput>> {
        self.maybe_fail(name)?;
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
        self.maybe_fail(prefix)?;
        self.inner.create_temp_output(prefix)
    }

    fn sync(&self) -> laurus::Result<()> {
        self.inner.sync()
    }

    fn close(&mut self) -> laurus::Result<()> {
        // The wrapper does not own the underlying storage (the test keeps an
        // `Arc` to it for the recovery round), so closing is a no-op.
        Ok(())
    }
}

/// Build the test schema: one lexical text field and one flat vector field, so
/// every commit touches both the `lexical/` and `vector/` namespaces.
fn build_schema() -> Schema {
    Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("embedding", FieldOption::Flat(FlatOption::default()))
        .build()
}

/// Build document `i` with the shared lexical term and a vector payload.
fn make_doc(i: usize) -> Document {
    Document::builder()
        .add_field(
            "title",
            DataValue::Text(format!("{SHARED_TERM} programming entry {i}")),
        )
        .add_field("embedding", DataValue::Vector(vec![0.1 + i as f32; 128]))
        .build()
}

/// Assert that all `DOC_COUNT` documents are present exactly once after recovery.
async fn assert_all_recovered(engine: &Engine) -> laurus::Result<()> {
    // Every external id resolves to exactly one document (upsert, no chunks).
    for i in 0..DOC_COUNT {
        let docs = engine.get_documents(&format!("doc{i}")).await?;
        assert_eq!(
            docs.len(),
            1,
            "doc{i} must be recovered exactly once, got {}",
            docs.len()
        );
    }

    // A lexical search on the shared term must return every document and no
    // duplicates.
    let query = Box::new(TermQuery::new("title", SHARED_TERM));
    let request = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(query))
        .limit(DOC_COUNT * 4)
        .build();
    let results = engine.search(request).await?.results;
    assert_eq!(
        results.len(),
        DOC_COUNT,
        "lexical search must return all {DOC_COUNT} documents with no duplicates"
    );
    Ok(())
}

/// Drive one crash scenario: index `DOC_COUNT` docs under `policy`, arm a fault
/// targeting `target`, fail the `commit()` there (the simulated crash), then
/// re-open on the same data and verify every document recovers.
async fn run_crash_scenario(policy: WalSyncPolicy, target: &str) -> laurus::Result<()> {
    let temp = TempDir::new().expect("temp dir");
    let inner: Arc<dyn Storage> =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp.path())))?;

    let plan = Arc::new(Mutex::new(FaultPlan {
        armed: false,
        target: target.to_string(),
        triggered: false,
    }));
    let faulty: Arc<dyn Storage> = Arc::new(FaultyStorage {
        inner: inner.clone(),
        plan: plan.clone(),
    });

    let schema = build_schema();

    // Round 1: index, then crash at the chosen commit step.
    {
        let engine = Engine::builder(faulty.clone(), schema.clone())
            .wal_sync_policy(policy)
            .build()
            .await?;

        for i in 0..DOC_COUNT {
            engine.put_document(&format!("doc{i}"), make_doc(i)).await?;
        }

        // Arm the fault for the commit under test (indexing above ran cleanly).
        plan.lock().expect("plan lock").armed = true;

        let result = engine.commit().await;
        assert!(
            result.is_err(),
            "commit must fail at the injected crash point '{target}' (policy {policy:?})"
        );
        assert!(
            plan.lock().expect("plan lock").triggered,
            "the fault targeting '{target}' must have fired (policy {policy:?})"
        );
        // Dropping the engine here simulates the crash.
    }

    // Round 2: re-open on the same underlying storage (no fault) and recover.
    {
        let engine = Engine::new(inner.clone(), schema.clone()).await?;
        // A clean commit after recovery must succeed and truncate the WAL.
        engine.commit().await?;
        assert_all_recovered(&engine).await?;
    }

    // Round 3: re-open once more to prove the post-recovery commit is itself
    // durable and the WAL is in a clean, replayable state.
    {
        let engine = Engine::new(inner.clone(), schema.clone()).await?;
        assert_all_recovered(&engine).await?;
    }

    Ok(())
}

// ── Crash after flush_wal, before lexical.commit (target "lexical/") ──────────

#[tokio::test(flavor = "multi_thread")]
async fn crash_before_lexical_commit_per_record() -> laurus::Result<()> {
    run_crash_scenario(WalSyncPolicy::PerRecord, "lexical/").await
}

#[tokio::test(flavor = "multi_thread")]
async fn crash_before_lexical_commit_group() -> laurus::Result<()> {
    run_crash_scenario(WalSyncPolicy::group_with_defaults(), "lexical/").await
}

// ── Crash after lexical.commit, before vector.commit (target "vector/") ───────

#[tokio::test(flavor = "multi_thread")]
async fn crash_before_vector_commit_per_record() -> laurus::Result<()> {
    run_crash_scenario(WalSyncPolicy::PerRecord, "vector/").await
}

#[tokio::test(flavor = "multi_thread")]
async fn crash_before_vector_commit_group() -> laurus::Result<()> {
    run_crash_scenario(WalSyncPolicy::group_with_defaults(), "vector/").await
}

// ── Crash after vector.commit, before commit_documents (target "documents/") ──

#[tokio::test(flavor = "multi_thread")]
async fn crash_before_document_commit_per_record() -> laurus::Result<()> {
    run_crash_scenario(WalSyncPolicy::PerRecord, "documents/").await
}

#[tokio::test(flavor = "multi_thread")]
async fn crash_before_document_commit_group() -> laurus::Result<()> {
    run_crash_scenario(WalSyncPolicy::group_with_defaults(), "documents/").await
}

// ── Crash after commit_documents, before/at WAL truncate (target "engine.wal") ─

#[tokio::test(flavor = "multi_thread")]
async fn crash_before_wal_truncate_per_record() -> laurus::Result<()> {
    run_crash_scenario(WalSyncPolicy::PerRecord, "engine.wal").await
}

#[tokio::test(flavor = "multi_thread")]
async fn crash_before_wal_truncate_group() -> laurus::Result<()> {
    run_crash_scenario(WalSyncPolicy::group_with_defaults(), "engine.wal").await
}

// ── Harness sanity: an unarmed FaultyStorage commits cleanly ─────────────────

#[tokio::test(flavor = "multi_thread")]
async fn unarmed_faulty_storage_commits_cleanly() -> laurus::Result<()> {
    let temp = TempDir::new().expect("temp dir");
    let inner: Arc<dyn Storage> =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp.path())))?;
    let plan = Arc::new(Mutex::new(FaultPlan {
        armed: false,
        target: "lexical/".to_string(),
        triggered: false,
    }));
    let faulty: Arc<dyn Storage> = Arc::new(FaultyStorage {
        inner: inner.clone(),
        plan,
    });

    let engine = Engine::builder(faulty, build_schema())
        .wal_sync_policy(WalSyncPolicy::group_with_defaults())
        .build()
        .await?;
    for i in 0..DOC_COUNT {
        engine.put_document(&format!("doc{i}"), make_doc(i)).await?;
    }
    engine.commit().await?;
    assert_all_recovered(&engine).await?;
    Ok(())
}

/// #875: deletion persistence is deferred to commit, with crash safety from
/// the WAL. This crashes an engine that holds a *buffered* deletion (the
/// existing-id upsert's delete-first is deferred, and the writer's drop-commit
/// is made to fail so nothing persists), then reopens and searches WITHOUT a
/// manual commit: recovery must replay the delete and auto-commit, so the old
/// version stays invisible and the new version is searchable.
#[tokio::test(flavor = "multi_thread")]
async fn deferred_deletion_recovers_after_crash_without_manual_commit() -> laurus::Result<()> {
    let temp = TempDir::new().expect("temp dir");
    let inner: Arc<dyn Storage> =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp.path())))?;

    let plan = Arc::new(Mutex::new(FaultPlan {
        armed: false,
        target: "lexical/".to_string(),
        triggered: false,
    }));
    let faulty: Arc<dyn Storage> = Arc::new(FaultyStorage {
        inner: inner.clone(),
        plan: plan.clone(),
    });

    let schema = build_schema();

    // Round 1: commit v1, upsert v2 (buffers the deletion of v1), then crash.
    {
        let engine = Engine::new(faulty.clone(), schema.clone()).await?;
        engine.put_document("doc0", make_doc(0)).await?;
        engine.commit().await?;

        let v2 = Document::builder()
            .add_field("title", DataValue::Text("updated fresh entry".into()))
            .add_field("embedding", DataValue::Vector(vec![0.9; 128]))
            .build();
        engine.put_document("doc0", v2).await?;

        // Crash: fail the lexical writer's drop-commit so neither the new
        // segment nor the deferred `.delmap` / checkpoint persists — only the
        // WAL (whose delete record preceded the mutation) survives.
        plan.lock().expect("plan lock").armed = true;
    }
    plan.lock().expect("plan lock").armed = false;

    // Round 2: reopen and search immediately — NO manual commit. Recovery
    // replays the delete + upsert and auto-commits (#875), so the replayed
    // state is fully searchable.
    {
        let engine = Engine::new(faulty.clone(), schema.clone()).await?;

        let request = |term: &str| {
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "title", term,
                ))))
                .build()
        };
        let updated = engine.search(request("updated")).await?.results;
        assert_eq!(
            updated.len(),
            1,
            "the replayed new version must be searchable without a manual commit"
        );
        let old = engine.search(request(SHARED_TERM)).await?.results;
        assert_eq!(
            old.len(),
            0,
            "the replayed deletion of the old version must be visible without \
             a manual commit (#875 recovery auto-commit)"
        );
    }

    Ok(())
}
