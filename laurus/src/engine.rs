pub mod json_document;
pub mod query;
pub mod schema;
pub mod search;
pub mod type_coercion;
pub mod type_inference;

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::RwLock;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::keyword::KeywordAnalyzer;
use crate::analysis::analyzer::per_field::PerFieldAnalyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::data::Document;
use crate::embedding::cache::{EmbeddingCache, embed_batch_with_cache};
use crate::embedding::embedder::Embedder;
use crate::error::Result;
use crate::lexical::store::LexicalStore;
use crate::lexical::store::config::LexicalIndexConfig;
use crate::storage::Storage;
use crate::storage::prefixed::PrefixedStorage;
use crate::store::log::{DocumentLog, LogEntry, WalSyncPolicy};
use crate::vector::core::vector::Vector;
use crate::vector::store::VectorStore;
use crate::vector::store::config::VectorIndexConfig;

use self::schema::Schema;

/// Callback invoked to persist a [`Schema`] snapshot outside the engine's own
/// [`Storage`] (Issue #1078).
///
/// The engine's `storage` only ever covers its `store/` subdirectory —
/// `schema.toml` conventionally lives one level up, alongside it, and where
/// (or whether) to write it is a decision that belongs to the caller, not the
/// engine. Set via [`EngineBuilder::persist_schema_with`]; when set,
/// [`Engine::add_field`] and [`Engine::delete_field`] invoke it with the
/// updated schema instead of leaving persistence to the caller.
pub type SchemaPersistHook = Arc<dyn Fn(&Schema) -> Result<()> + Send + Sync>;

/// Policy controlling when the engine automatically runs the commit ladder
/// during ingestion (Issue #890).
///
/// The commit ladder ([`Engine::commit`]) materializes and fsyncs the lexical,
/// vector, and document stores and truncates the WAL. By default it runs only
/// when the caller invokes [`Engine::commit`] explicitly; a non-`Manual` policy
/// makes the engine run it automatically at an ingestion-driven cadence, one
/// full ladder per auto-commit (group-commit semantics are preserved).
///
/// This is orthogonal to [`WalSyncPolicy`]: that governs *WAL fsync
/// durability*, while `CommitPolicy` governs *when the stores materialize*. An
/// auto-commit works under any `WalSyncPolicy` because [`Engine::commit`] always
/// begins with a WAL flush.
///
/// The enum is `#[non_exhaustive]`, so downstream `match`es must carry a
/// wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CommitPolicy {
    /// Never auto-commit; the caller drives every commit explicitly. This is
    /// the default and preserves the historical behavior.
    #[default]
    Manual,
    /// Auto-commit after every `n` applied documents, across the singular
    /// ([`Engine::put_document`] / [`Engine::add_document`]) and batch
    /// ([`Engine::put_documents`] / [`Engine::add_documents`]) APIs. Within a
    /// batch the commit fires every `n` documents (chunked), so a large batch
    /// materializes incrementally rather than in one final ladder.
    ///
    /// The exact cadence, and the usual "acknowledged write is durable"
    /// guarantee, hold for **single-writer ingestion** (the model the engine's
    /// write path is built around — the CLI and language bindings drive a
    /// single ingest task). Under **concurrent writers on a shared engine**,
    /// auto-commit is best-effort: the commit ladder is not atomic with respect
    /// to another thread's in-flight write, so a write acknowledged while a
    /// concurrent auto-commit is running may only become durable at the
    /// following commit, and the cadence may drift. This is a property of
    /// commit-vs-concurrent-write in general (a concurrent manual `commit()`
    /// races the same way); auto-commit merely triggers it from inside the
    /// ingest path. Use explicit commits, or a single ingest task, when you
    /// need these guarantees under concurrency.
    ///
    /// `EveryDocs(0)` disables auto-commit (equivalent to
    /// [`CommitPolicy::Manual`]).
    EveryDocs(usize),
    /// Auto-commit at least every `Duration` via a background timer, so a
    /// trailing partial batch is committed even while ingestion is idle (the
    /// time-based counterpart of [`CommitPolicy::EveryDocs`]).
    ///
    /// The timer runs the full commit ladder on its own thread; the same
    /// best-effort concurrency caveat as `EveryDocs` applies (a commit racing a
    /// concurrent in-flight write may leave that write durable only at the next
    /// commit). **Native targets only** — on `wasm32` there are no background
    /// threads, so `Interval` is a no-op (the timer is never started), just like
    /// [`WalSyncPolicy::Group`]'s `max_interval`.
    Interval(Duration),
}

/// Combined statistics from both the lexical and vector stores.
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    /// Total number of documents in the index (from the lexical store).
    pub document_count: u64,
    /// Per-field vector statistics, keyed by field name.
    /// Empty when the schema contains no vector fields.
    pub vector_fields: HashMap<String, crate::vector::index::field::VectorFieldStats>,
}

/// Summary of a shared PQ codebook produced by
/// [`Engine::train_pq_codebook`] (Issue #631).
#[derive(Debug, Clone)]
pub struct PqCodebookInfo {
    /// Storage-relative file name the codebook was written to, inside the
    /// engine's vector storage namespace (e.g. `"embedding.pqcb"`). This is
    /// the value the field's
    /// [`HnswOption::pq_codebook_path`](crate::vector::core::field::HnswOption::pq_codebook_path)
    /// must be set to for subsequent commits to pick the codebook up.
    pub path: String,
    /// Number of sub-vectors (`m`) the codebook was trained for.
    pub subvector_count: usize,
    /// Centroids per sub-vector (`k`): `256` for standard 8-bit PQ
    /// fields, `16` for FastScan fields (Issue #920).
    pub centroids: usize,
    /// Sub-vector dimension (`dimension / subvector_count`).
    pub sub_dimension: usize,
    /// Original vector dimension the codebook encodes.
    pub dimension: usize,
    /// Number of vectors the codebook was trained on.
    pub training_vectors: usize,
}

/// Background timer that periodically forces the WAL durable under a
/// [`WalSyncPolicy::Group`] configured with a `max_interval` (Issue #542, Phase
/// 4b).
///
/// Runs a dedicated thread that calls [`DocumentLog::flush_wal`] every interval
/// — a no-op when nothing is pending, thanks to the dirty guard — so a trailing
/// partial batch under a low ingest rate is not left unsynced indefinitely (the
/// record/byte thresholds may never be reached). Dropping the timer wakes the
/// thread immediately and joins it. Native targets only; on `wasm32` (no
/// background threads) the timer is never constructed and the interval is
/// ignored.
#[cfg(not(target_arch = "wasm32"))]
struct WalFlushTimer {
    /// Sending on (or dropping) this channel signals the thread to stop.
    stop: std::sync::mpsc::Sender<()>,
    /// Join handle for the flush thread, taken and joined on drop.
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl WalFlushTimer {
    /// Spawn the flush thread for `doc_log`, forcing the WAL durable every
    /// `interval`.
    ///
    /// # Arguments
    ///
    /// * `doc_log` - The document log whose WAL is flushed on each tick.
    /// * `interval` - How often to flush the WAL.
    ///
    /// # Errors
    ///
    /// Returns an error if the OS thread cannot be spawned.
    fn spawn(doc_log: Arc<DocumentLog>, interval: std::time::Duration) -> Result<Self> {
        use std::sync::mpsc::RecvTimeoutError;

        let (stop, rx) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::Builder::new()
            .name("laurus-wal-flush".to_string())
            .spawn(move || {
                // Keep ticking while each wait ends in a timeout: flush the WAL
                // (a no-op when there is nothing pending). Any other outcome — a
                // stop signal (`Ok`) or the sender being dropped (`Disconnected`)
                // — ends the loop and the thread.
                while let Err(RecvTimeoutError::Timeout) = rx.recv_timeout(interval) {
                    if let Err(e) = doc_log.flush_wal() {
                        log::warn!("WAL flush timer: failed to flush WAL: {e}");
                    }
                }
            })?;
        Ok(Self {
            stop,
            handle: Some(handle),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for WalFlushTimer {
    fn drop(&mut self) {
        // Wake the thread immediately so it exits without waiting out the
        // current interval, then join it.
        let _ = self.stop.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Run the commit durability ladder over the borrowed store sub-parts and reset
/// the auto-commit counter (Issue #821, extracted for reuse by [`CommitTimer`]
/// in #892).
///
/// The fixed order makes a crash at any step recoverable: `flush_wal` (the hard
/// barrier) → `lexical.commit` (persists `last_wal_seq`) → `vector.commit` →
/// `commit_documents` → `truncate`. Both [`Engine::commit`] and the background
/// [`CommitTimer`] call this, so the timer needs only the `Arc` sub-parts — not
/// the whole `Engine` (which would be a reference cycle).
///
/// # Errors
///
/// Returns an error if any ladder step fails; the counter is reset only after a
/// fully successful commit.
async fn run_commit_ladder(
    lexical: &LexicalStore,
    vector: &VectorStore,
    log: &DocumentLog,
    docs_since_commit: &AtomicU64,
    applied_seq: &AtomicU64,
) -> Result<()> {
    // Snapshot the ingest high-water mark BEFORE any store materializes
    // (Issue #876). `applied_seq` is advanced only after a document has landed
    // in BOTH stores' NRT buffers, so every record at or below this snapshot is
    // guaranteed to be materialized by the `lexical.commit()` below — the store
    // commits hold the same writer lock those upserts take, so a document that
    // finished before this point cannot be excluded from the commit. Anything
    // appended (or applied) afterwards must survive the truncate.
    let applied_before = applied_seq.load(Ordering::Acquire);
    // Hard durability barrier: force the WAL durable before any store
    // materializes its state, so the WAL is never less durable than the
    // committed lexical/vector indexes. A near-no-op under the per-record
    // default (each append already synced, so the dirty guard skips the
    // fsync); the load-bearing step once group commit defers per-append
    // fsync (#542 Phase 3).
    log.flush_wal()?;
    lexical.commit()?;
    vector.commit().await?;
    log.commit_documents()?;
    // Truncate the log, but RETAIN every record this commit did not materialize
    // (Issue #876). The ladder is not serialized against ingestion, so a
    // mutation can append its WAL record while the commit runs — wiping the
    // whole file would destroy that record while its data lives only in the
    // fresh in-memory writer, losing an acknowledged write on the next crash.
    // Retaining from the pre-commit snapshot is conservative: a record that was
    // in fact materialized is merely replayed again on recovery, which is
    // idempotent.
    log.truncate_retaining_after(applied_before)?;
    // Reset the auto-commit counter so a manual commit and an `EveryDocs`
    // auto-commit keep the same cadence going forward (#890).
    docs_since_commit.store(0, Ordering::Release);
    Ok(())
}

/// Background timer that runs the full commit ladder at least every interval for
/// [`CommitPolicy::Interval`] (Issue #892), so a trailing partial batch is
/// committed even while ingestion is idle.
///
/// Mirrors [`WalFlushTimer`]: a dedicated thread with an mpsc stop channel,
/// joined on drop. Unlike the WAL timer (which calls the synchronous
/// `flush_wal` on `Arc<DocumentLog>`), the commit ladder is `async` and spans
/// the lexical + vector stores, so the thread owns a private single-threaded
/// tokio runtime and `block_on`s [`run_commit_ladder`] each tick. It holds only
/// the `Arc` sub-parts the ladder needs — never the `Engine` — so there is no
/// reference cycle and dropping the engine cleanly stops the timer. Native
/// targets only; on `wasm32` (no background threads) `Interval` is a no-op.
#[cfg(not(target_arch = "wasm32"))]
struct CommitTimer {
    /// Sending on (or dropping) this channel signals the thread to stop.
    stop: std::sync::mpsc::Sender<()>,
    /// Join handle for the commit thread, taken and joined on drop.
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl CommitTimer {
    /// Spawn the commit thread, running the ladder every `interval`.
    ///
    /// # Arguments
    ///
    /// * `lexical` / `vector` / `log` / `docs_since_commit` - The store
    ///   sub-parts the commit ladder operates on, shared with the engine.
    /// * `interval` - How often to run the commit ladder.
    ///
    /// # Errors
    ///
    /// Returns an error if the private runtime or the OS thread cannot be
    /// created.
    fn spawn(
        lexical: Arc<LexicalStore>,
        vector: Arc<VectorStore>,
        log: Arc<DocumentLog>,
        docs_since_commit: Arc<AtomicU64>,
        applied_seq: Arc<AtomicU64>,
        interval: Duration,
    ) -> Result<Self> {
        use std::sync::mpsc::RecvTimeoutError;

        let (stop, rx) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::Builder::new()
            .name("laurus-commit-timer".to_string())
            .spawn(move || {
                // A private current-thread runtime drives the async ladder. The
                // ladder's only runtime-bound await is a `tokio::sync::Mutex`
                // lock (in `VectorStore::commit`), which is runtime-agnostic, so
                // this is independent of whatever runtime the caller uses.
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        log::warn!("commit timer: failed to build runtime: {e}");
                        return;
                    }
                };
                // Tick while each wait ends in a timeout: run the commit ladder.
                // A stop signal (`Ok`) or a dropped sender (`Disconnected`) ends
                // the loop and the thread.
                while let Err(RecvTimeoutError::Timeout) = rx.recv_timeout(interval) {
                    // Catch a panic from deep in the ladder so a single failure
                    // does not silently kill the timer thread — which would stop
                    // auto-commit for the engine's whole lifetime with no
                    // diagnostic. The stores guard their own state behind locks,
                    // so the next tick simply retries. `AssertUnwindSafe` is
                    // sound here: after a caught panic the closure only re-reads
                    // the shared `Arc` sub-parts and retries the commit.
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        rt.block_on(run_commit_ladder(
                            &lexical,
                            &vector,
                            &log,
                            &docs_since_commit,
                            &applied_seq,
                        ))
                    }));
                    match outcome {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => log::warn!("commit timer: failed to commit: {e}"),
                        Err(_) => {
                            log::error!("commit timer: commit panicked; retrying on the next tick")
                        }
                    }
                }
            })?;
        Ok(Self {
            stop,
            handle: Some(handle),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for CommitTimer {
    fn drop(&mut self) {
        // Wake the thread immediately so it exits without waiting out the
        // current interval, then join it.
        let _ = self.stop.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Number of shards in [`Engine::id_locks`]. Fixed rather than derived from
/// the core count so behavior is identical everywhere; a power of two keeps
/// the hash-to-shard mapping cheap and even.
const ID_LOCK_SHARDS: usize = 64;

/// Unified Engine that manages both Lexical and Vector indices.
///
/// This engine acts as a facade, coordinating document ingestion and search
/// across the underlying specialized engines. All index mutations are
/// WAL-backed via [`DocumentLog`] for crash-recovery durability.
///
/// A system field `_id` is automatically injected into every indexed document
/// to track the external document identifier.
pub struct Engine {
    schema: RwLock<Schema>,
    /// The lexical store. `Arc` so the [`CommitTimer`] can share it as a
    /// sub-part (a timer holding the whole `Engine` would be a reference cycle);
    /// every access is through `&self` methods, so the `Arc` is transparent.
    lexical: Arc<LexicalStore>,
    /// The vector store. `Arc` for the same reason as [`Self::lexical`].
    vector: Arc<VectorStore>,
    log: Arc<DocumentLog>,
    /// Pre-constructed analyzers registered at build time and consulted
    /// before built-in names and `schema.analyzers` when resolving
    /// per-field analyzer references. See
    /// [`EngineBuilder::register_runtime_analyzer`].
    runtime_analyzers: HashMap<String, Arc<dyn Analyzer>>,
    /// Optional LRU cache for query-time embeddings (Issue #678).
    ///
    /// `None` when the cache is disabled (the default); enabled via
    /// [`EngineBuilder::embedding_cache_capacity`]. Shared (via `Arc`) with
    /// the [`VectorQueryParser`](crate::vector::query::parser::VectorQueryParser)
    /// built in [`Self::unified_query_parser`], so both the direct
    /// `Payloads` path and the DSL path hit the same cache.
    embedding_cache: Option<Arc<EmbeddingCache>>,
    /// Auto-commit policy (Issue #890). `Manual` by default; a non-`Manual`
    /// policy runs the commit ladder automatically during ingestion.
    commit_policy: CommitPolicy,
    /// Documents applied since the last commit, tracking progress toward the
    /// [`CommitPolicy::EveryDocs`] threshold. Incremented once per applied
    /// document in [`Self::index_internal`] and reset to `0` by
    /// [`Self::commit`], so both manual and auto commits keep the count
    /// consistent. Lock-free (mirrors [`DocumentLog`]'s counters); exact for
    /// serial ingestion, best-effort under concurrent writers. `Arc` so the
    /// [`CommitTimer`] shares the same counter and resets it on each timed
    /// commit.
    docs_since_commit: Arc<AtomicU64>,
    /// Highest WAL sequence number whose mutation has been applied to **both**
    /// stores' NRT buffers (Issue #876).
    ///
    /// Advanced in [`Self::index_internal`] / the delete path only after both
    /// stores accepted the mutation, so a commit can snapshot it up front and
    /// know that everything at or below it will be materialized by that commit.
    /// [`run_commit_ladder`] uses the snapshot as the WAL retain point, so a
    /// record appended by a mutation racing the commit is preserved instead of
    /// being wiped by the truncate. `Arc` so the [`CommitTimer`] shares it.
    applied_seq: Arc<AtomicU64>,
    /// Per-external-id serialization for the delete-then-add upsert (#1049).
    ///
    /// [`Self::index_internal`] deletes any existing versions of an id and
    /// then indexes the new one; every step takes the underlying store locks
    /// separately, so without this two threads upserting the same id both
    /// find-and-delete, then both add, and the id ends up with several live
    /// versions. Sharded by hash so only genuine same-id contention
    /// serializes — distinct ids keep ingesting in parallel.
    ///
    /// A `tokio::sync::Mutex` because the guard is held across `await`
    /// points. It is the OUTERMOST lock wherever it is taken, so it cannot
    /// invert against the lexical writer → searcher order that
    /// [`LexicalStore::find_doc_ids_by_term`] maintains.
    id_locks: Arc<Vec<tokio::sync::Mutex<()>>>,
    /// Background WAL flush timer for a [`WalSyncPolicy::Group`] configured with
    /// a `max_interval`. `None` when no interval is set. Held only to keep the
    /// timer thread alive for the engine's lifetime; dropping the engine stops
    /// it. Absent on `wasm32` (no background threads — the interval is ignored).
    #[cfg(not(target_arch = "wasm32"))]
    _wal_flush_timer: Option<WalFlushTimer>,
    /// Background auto-commit timer for [`CommitPolicy::Interval`] (Issue #892).
    /// `None` under any other policy. Held only to keep the timer thread alive
    /// for the engine's lifetime; dropping the engine stops and joins it.
    /// Absent on `wasm32` (no background threads — `Interval` is a no-op).
    #[cfg(not(target_arch = "wasm32"))]
    _commit_timer: Option<CommitTimer>,
    /// Optional callback that persists the schema outside the engine's own
    /// storage (Issue #1078). See [`SchemaPersistHook`].
    schema_persist_hook: Option<SchemaPersistHook>,
}

use crate::engine::search::{FusionAlgorithm, SearchResult};

impl Engine {
    /// Create a new Unified Engine with default analyzer and no embedder.
    ///
    /// For custom analyzer or embedder configuration, use [`Engine::builder`].
    ///
    /// # Errors
    ///
    /// Returns an error if storage initialization, index creation, or
    /// WAL recovery fails.
    pub async fn new(storage: Arc<dyn Storage>, schema: Schema) -> Result<Self> {
        EngineBuilder::new(storage, schema).build().await
    }

    /// Create an [`EngineBuilder`] for custom configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let engine = Engine::builder(storage, schema)
    ///     .analyzer(Arc::new(StandardAnalyzer::default()))
    ///     .embedder(Arc::new(MyEmbedder))
    ///     .build()
    ///     .await?;
    /// ```
    pub fn builder(storage: Arc<dyn Storage>, schema: Schema) -> EngineBuilder {
        EngineBuilder::new(storage, schema)
    }

    /// Recover index state from the document log.
    ///
    /// Replays every WAL record that is newer than each store's persisted
    /// `last_wal_seq` checkpoint. Recovery is **idempotent**: each record is
    /// re-applied under its originally recorded `doc_id`, so re-running it
    /// overwrites rather than duplicates. The lexical and vector stores track
    /// their checkpoints independently, so a commit that failed partway (leaving
    /// the stores at different `last_wal_seq` values) is reconciled here — each
    /// store re-applies only what it is missing. See [`Self::commit`] for the
    /// ordering guarantees that make this safe (Issue #821).
    async fn recover(&self) -> Result<()> {
        // read_all() internally syncs next_doc_id with doc_store segments.
        let records = self.log.read_all()?;

        if records.is_empty() {
            return Ok(());
        }

        // `vector.last_wal_seq()` is itself an aggregate when the vector
        // store is a `MultiFieldVectorIndex` (Issue #948): each vector
        // field has its own independent sub-index and therefore its own
        // checkpoint, and the store reports the MINIMUM across all of
        // them -- the least-caught-up field, never the most. A `max`
        // would let a lagging field's WAL records be skipped here forever
        // (`record.seq <= vector_last_seq` would already be true for them).
        // The `min` instead makes an already-caught-up field replay a few
        // extra records it does not need; that is safe because recovery
        // re-applies each record's upsert under its recorded `doc_id`
        // (delete-then-add), which is idempotent.
        let vector_last_seq = self.vector.last_wal_seq();
        let lexical_last_seq = self.lexical.last_wal_seq();

        let mut applied = false;
        for record in records {
            if record.seq <= vector_last_seq && record.seq <= lexical_last_seq {
                // Covered by both INDEX checkpoints — their replay can be
                // skipped. Before the lexical checkpoint became real this
                // branch was unreachable (the trait default pinned
                // `lexical_last_seq` to 0), so both obligations below are
                // new with #1023.
                //
                // The document store is NOT covered by the guard: it has no
                // checkpoint of its own, and the commit ladder persists it
                // AFTER the indexes — so a crash between those steps leaves
                // both indexes checkpointed at N while the documents exist
                // only in the WAL. Re-store the payload unconditionally;
                // for a document that did reach disk this rewrites identical
                // content, so it is idempotent. Deletes need nothing here:
                // the document store never deletes — liveness is decided by
                // the lexical index, which the checkpoint does cover.
                if let LogEntry::Upsert {
                    doc_id, document, ..
                } = &record.entry
                {
                    let stored_doc = self.filter_stored_fields(document);
                    self.log.store_document(*doc_id, stored_doc);
                    // Persist the restored documents via the commit below,
                    // exactly as a full replay would have.
                    applied = true;
                }
                // Publish the high-water mark for skipped records too:
                // without it a skip-everything recovery leaves `applied_seq`
                // at its build-time 0, and the next commit's
                // `truncate_retaining_after(0)` re-retains the whole dead
                // WAL — breaking the post-commit empty-WAL invariant.
                self.applied_seq.fetch_max(record.seq, Ordering::AcqRel);
                continue;
            }
            applied = true;

            match record.entry {
                LogEntry::Upsert {
                    doc_id,
                    external_id: _,
                    document,
                } => {
                    // Restore document into document store
                    let stored_doc = self.filter_stored_fields(&document);
                    self.log.store_document(doc_id, stored_doc);

                    // Re-index into both stores using the recorded doc_id.
                    // Update seq only after BOTH stores succeed to maintain atomicity.
                    if record.seq > lexical_last_seq {
                        self.lexical.upsert_document(doc_id, document.clone())?;
                    }

                    if record.seq > vector_last_seq {
                        // Filter for vector fields
                        let mut vector_doc = Document::new();
                        {
                            let schema = self.schema.read();
                            for (name, val) in &document.fields {
                                if schema.fields.get(name).is_some_and(|fc| fc.is_vector()) {
                                    vector_doc.fields.insert(name.clone(), val.clone());
                                }
                            }
                        }
                        self.vector
                            .upsert_document_by_internal_id(doc_id, vector_doc)
                            .await?;
                    }

                    // Both stores succeeded — now update seq trackers.
                    // `vector.set_last_wal_seq` propagates `record.seq` to
                    // EVERY field's sub-index identically (Issue #948); see
                    // `vector_last_seq`'s comment above for why the store's
                    // own `last_wal_seq()` aggregate is a `min`, not a `max`.
                    if record.seq > lexical_last_seq {
                        self.lexical.set_last_wal_seq(record.seq)?;
                    }
                    if record.seq > vector_last_seq {
                        self.vector.set_last_wal_seq(record.seq);
                    }
                    // Publish the high-water mark exactly as `index_internal`
                    // does (Issue #876): by this point both stores are at or
                    // above `record.seq`, whether just applied here or already
                    // covered by an earlier partial commit. Without this, the
                    // `self.commit()` below would snapshot `applied_seq` at its
                    // stale build-time 0 and its own truncate would wrongly
                    // retain every record it just durably committed.
                    self.applied_seq.fetch_max(record.seq, Ordering::AcqRel);
                }
                LogEntry::Delete {
                    doc_id,
                    external_id: _,
                } => {
                    if record.seq > lexical_last_seq {
                        self.lexical.delete_document_by_internal_id(doc_id)?;
                    }
                    if record.seq > vector_last_seq {
                        self.vector.delete_document_by_internal_id(doc_id).await?;
                    }

                    // Both stores succeeded — now update seq trackers (see
                    // the `Upsert` arm above for the `MultiFieldVectorIndex`
                    // min-aggregate rationale, Issue #948).
                    if record.seq > lexical_last_seq {
                        self.lexical.set_last_wal_seq(record.seq)?;
                    }
                    if record.seq > vector_last_seq {
                        self.vector.set_last_wal_seq(record.seq);
                    }
                    // See the matching comment in the `Upsert` arm above.
                    self.applied_seq.fetch_max(record.seq, Ordering::AcqRel);
                }
            }
        }

        // Persist the replayed state right away (Issue #875). Deletion
        // persistence is deferred to commit, so without this a freshly
        // reopened index would serve the *old* versions of upserted/deleted
        // documents to searchers until the caller's next commit — replayed
        // deletions only exist in the writer's in-memory bitmaps until then.
        // Committing here also bounds re-replay work: the WAL is truncated,
        // so a subsequent crash replays nothing instead of the same records
        // again. Safe against a crash *during* this commit: the WAL is only
        // truncated after every store has committed, and replay is
        // idempotent.
        if applied {
            self.commit().await?;
        }
        Ok(())
    }

    /// Put (upsert) a document.
    ///
    /// If a document with the same external ID exists, all its chunks are
    /// deleted before the new document is indexed. A `_id` field is
    /// automatically inserted into the document with the provided `id` value.
    /// A WAL entry is written before any index mutations to ensure durability.
    ///
    /// The document fields are routed to the appropriate underlying stores
    /// (lexical or vector) based on the schema field configuration. If the
    /// vector store indexing fails after the lexical store has already been
    /// updated, the lexical insert is rolled back to maintain cross-store
    /// consistency.
    ///
    /// # Parameters
    ///
    /// - `id` - The external document identifier.
    /// - `doc` - The document to index.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write, deletion of existing documents,
    /// or indexing into either the lexical or vector store fails.
    pub async fn put_document(&self, id: &str, doc: Document) -> Result<()> {
        let _ = self.index_internal(id, doc, false).await?;
        // Re-assert per-record durability: a concurrent batch may hold a WAL
        // sync-deferral scope, which would otherwise leave this acknowledged
        // write unsynced. A no-op when the append already self-synced.
        self.log.ensure_per_record_durability()?;
        // Auto-commit if the CommitPolicy threshold was reached (#890).
        self.maybe_auto_commit().await?;
        Ok(())
    }

    /// Add a document as a new chunk (always appends, never deletes existing).
    ///
    /// Unlike [`put_document`](Self::put_document), this method does **not**
    /// delete existing documents with the same external ID. Multiple chunks
    /// can share the same ID, which is useful for indexing parts of a large
    /// document (e.g. paragraphs or pages) separately while keeping them
    /// associated with the same logical document.
    ///
    /// A `_id` field is automatically inserted into the document with the
    /// provided `id` value. A WAL entry is written before any index mutations
    /// to ensure durability.
    ///
    /// # Parameters
    ///
    /// - `id` - The external document identifier (may duplicate existing IDs).
    /// - `doc` - The document chunk to index.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write or indexing into either the lexical
    /// or vector store fails.
    pub async fn add_document(&self, id: &str, doc: Document) -> Result<()> {
        let _ = self.index_internal(id, doc, true).await?;
        // Re-assert per-record durability: a concurrent batch may hold a WAL
        // sync-deferral scope, which would otherwise leave this acknowledged
        // write unsynced. A no-op when the append already self-synced.
        self.log.ensure_per_record_durability()?;
        // Auto-commit if the CommitPolicy threshold was reached (#890).
        self.maybe_auto_commit().await?;
        Ok(())
    }

    /// Index a batch of documents, replacing any existing documents that
    /// share an external ID (batched form of [`Self::put_document`]).
    ///
    /// Documents are applied **sequentially, in input order** — the WAL
    /// doc_id/seq allocation order is the crash-recovery replay order, and
    /// duplicate external IDs within one batch must dedup exactly like the
    /// equivalent sequence of [`Self::put_document`] calls (the last
    /// occurrence wins). Both underlying stores serialize writes behind
    /// mutexes, so unlike [`Self::search_batch`] there is nothing to gain
    /// from processing batch entries concurrently.
    ///
    /// Under the default [`WalSyncPolicy::PerRecord`](crate::store::log::WalSyncPolicy)
    /// the whole batch is made durable with a **single WAL fsync** at batch
    /// end instead of one per record: per-record fsync is suppressed for the
    /// duration of the call (via [`DocumentLog::defer_sync`](crate::store::log::DocumentLog::defer_sync))
    /// and [`Self::flush_wal`] runs once before returning, on both the
    /// success and the error path. When the call returns `Ok`, every
    /// document in the batch is as durable as a singular put.
    ///
    /// # Parameters
    ///
    /// - `docs` - `(external_id, document)` pairs, applied in order.
    ///   An empty batch is a no-op.
    ///
    /// # Errors
    ///
    /// Fails fast at the first document that cannot be applied, returning
    /// [`LaurusError::BatchIngest`](crate::error::LaurusError::BatchIngest) with the failing position, its external
    /// id, and the number of documents already applied. Applied documents
    /// are **not** rolled back: they stay in the WAL and NRT buffers,
    /// searchable immediately, durable at the next commit, and replayed on
    /// crash recovery — so retrying the batch (or its suffix from
    /// `failed_index`) is idempotent. The failing document inherits the
    /// singular-put semantics: its WAL record (if written) is retried by
    /// crash recovery and discarded by the next successful commit.
    pub async fn put_documents(&self, docs: Vec<(String, Document)>) -> Result<()> {
        self.index_batch_internal(docs, false).await
    }

    /// Index a batch of document chunks, always appending (batched form of
    /// [`Self::add_document`]).
    ///
    /// Unlike [`Self::put_documents`], existing documents with the same
    /// external ID are **not** deleted, so a batch may legitimately repeat
    /// an ID to add multiple chunks of the same logical document in one
    /// call.
    ///
    /// Ordering, durability (single WAL fsync per batch under the default
    /// per-record policy), and fail-fast error semantics are identical to
    /// [`Self::put_documents`].
    ///
    /// # Parameters
    ///
    /// - `docs` - `(external_id, document)` pairs, applied in order.
    ///   An empty batch is a no-op.
    ///
    /// # Errors
    ///
    /// Fails fast with [`LaurusError::BatchIngest`](crate::error::LaurusError::BatchIngest); see
    /// [`Self::put_documents`] for the exact semantics.
    pub async fn add_documents(&self, docs: Vec<(String, Document)>) -> Result<()> {
        self.index_batch_internal(docs, true).await
    }

    /// Shared sequential loop behind [`Self::put_documents`] /
    /// [`Self::add_documents`].
    ///
    /// Wraps per-document [`Self::index_internal`] calls in a WAL
    /// sync-deferral scope and flushes the WAL exactly once at batch end on
    /// both exit paths, converting the first per-document failure into
    /// [`LaurusError::BatchIngest`](crate::error::LaurusError::BatchIngest).
    ///
    /// # Parameters
    ///
    /// - `docs` - `(external_id, document)` pairs, applied in order.
    /// - `as_chunk` - `false` for put (delete-first) semantics, `true` for
    ///   add (append-chunk) semantics.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::BatchIngest`](crate::error::LaurusError::BatchIngest) on the first failing document,
    /// or the WAL flush error if the batch applied fully but the final
    /// fsync failed (retrying the whole batch is idempotent).
    async fn index_batch_internal(
        &self,
        docs: Vec<(String, Document)>,
        as_chunk: bool,
    ) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let deferral = self.log.defer_sync();
        for (index, (id, doc)) in docs.into_iter().enumerate() {
            if let Err(e) = self.index_internal(&id, doc, as_chunk).await {
                drop(deferral);
                // Make the applied prefix durable before reporting the
                // failure, keeping the per-record durability contract at
                // batch granularity. If the fsync itself fails the records
                // stay marked dirty and are retried by the next flush_wal /
                // commit, so the per-document error remains the primary one.
                let _ = self.log.flush_wal();
                return Err(crate::error::LaurusError::BatchIngest {
                    failed_index: index,
                    failed_id: id,
                    applied: index,
                    source: Box::new(e),
                });
            }
            // Auto-commit every `n` documents *within* the batch (#890). The
            // commit's own `flush_wal` is deferral-independent, so each chunk
            // becomes durable in one fsync + one ladder (group-commit per
            // chunk), leaving the sync-deferral scope intact for the rest of
            // the loop. A commit failure is distinct from a per-document
            // failure: surface it directly (the applied prefix stays in the
            // WAL and replays on recovery) rather than as a `BatchIngest`.
            if let Err(e) = self.maybe_auto_commit().await {
                drop(deferral);
                return Err(e);
            }
        }
        drop(deferral);
        self.log.flush_wal()
    }

    async fn index_internal(&self, id: &str, mut doc: Document, as_chunk: bool) -> Result<u64> {
        // 1. Inject _id field
        use crate::data::DataValue;
        doc.fields
            .insert("_id".to_string(), DataValue::Text(id.to_string()));

        // 1b. Validate reserved field-name namespace, then apply the schema's
        // DynamicFieldPolicy to add / coerce / drop user fields.
        self.apply_dynamic_schema(&mut doc).await?;

        // Serialize everything that follows against other mutations of this
        // same id (#1049). The delete and the add below take the store locks
        // separately, so without this guard two concurrent upserts of one id
        // both delete-then-add and leave two live versions. Held across the
        // whole sequence, including the WAL append, so the log's record order
        // for an id matches the order the stores applied them in.
        let _id_guard = self.id_lock(id).lock().await;

        if !as_chunk {
            self.delete_documents_internal(id).await?;
        }

        // 2. Write-Ahead Log: assign doc_id + persist (before any index updates)
        let (doc_id, seq) = self.log.append(id, doc.clone())?;

        // 3. Store only stored fields for retrieval (WAL has full data for recovery)
        let stored_doc = self.filter_stored_fields(&doc);
        self.log.store_document(doc_id, stored_doc);

        // 4. Prepare vector document (extract vector fields only)
        let mut vector_doc = Document::new();
        {
            let schema = self.schema.read();
            for (name, val) in &doc.fields {
                if schema.fields.get(name).is_some_and(|fc| fc.is_vector()) {
                    vector_doc.fields.insert(name.clone(), val.clone());
                }
            }
        }

        // 5. Index into Lexical and Vector stores
        self.lexical.upsert_document(doc_id, doc)?;
        if let Err(e) = self
            .vector
            .upsert_document_by_internal_id(doc_id, vector_doc)
            .await
        {
            // Rollback lexical insert to maintain consistency
            let _ = self.lexical.delete_document_by_internal_id(doc_id);
            return Err(e);
        }

        // 6. Update sub-stores sequence tracker AFTER both stores succeed.
        // This ensures failed index operations are retried on recovery.
        // `vector.set_last_wal_seq` fans `seq` out to every field's
        // sub-index identically (Issue #948); see `recover()`'s comment on
        // `vector_last_seq` for why the store's own `last_wal_seq()`
        // aggregate is a `min` across fields, not a `max`.
        self.lexical.set_last_wal_seq(seq)?;
        self.vector.set_last_wal_seq(seq);
        // Publish the ingest high-water mark only now that BOTH stores hold the
        // mutation, so a concurrent commit can safely treat everything at or
        // below it as materializable (Issue #876). `fetch_max` keeps it
        // monotonic under concurrent writers.
        self.applied_seq.fetch_max(seq, Ordering::AcqRel);

        // 7. Count the applied document toward the auto-commit threshold. The
        // trigger itself lives at the API boundary (see `maybe_auto_commit`),
        // not here, so this method's `Ok` keeps meaning "document applied" —
        // a commit failure never masquerades as a per-document index failure.
        self.docs_since_commit.fetch_add(1, Ordering::AcqRel);

        Ok(doc_id)
    }

    /// Run the commit ladder if the [`CommitPolicy`] threshold has been reached.
    ///
    /// A no-op under [`CommitPolicy::Manual`] and [`CommitPolicy::EveryDocs`]
    /// with a zero count. Under `EveryDocs(n)` with `n > 0` it commits once the
    /// documents applied since the last commit reach `n`; [`Self::commit`]
    /// resets the counter. Called at ingestion API boundaries (after each
    /// singular put/add, and after each document inside a batch), never inside
    /// [`Self::index_internal`], so its error is reported as a commit error
    /// rather than a document-index failure.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`Self::commit`].
    async fn maybe_auto_commit(&self) -> Result<()> {
        let CommitPolicy::EveryDocs(n) = self.commit_policy else {
            return Ok(());
        };
        if n > 0 && self.docs_since_commit.load(Ordering::Acquire) >= n as u64 {
            self.commit().await?;
        }
        Ok(())
    }

    /// Apply the schema's [`DynamicFieldPolicy`](schema::DynamicFieldPolicy)
    /// to an incoming document's fields.
    ///
    /// For each user-supplied field:
    ///
    /// - **Reserved names**: any field name starting with `_` other than
    ///   `_id` is rejected regardless of policy.
    /// - **Declared fields**: the value is coerced to the declared type (see
    ///   [`type_coercion::coerce_value`]).
    /// - **Undeclared fields**: handled according to the policy:
    ///   - `Strict`: ingest fails with an error.
    ///   - `Dynamic`: the field type is inferred (see
    ///     [`type_inference::infer_option_from_data_value`]) and the field
    ///     is added to the schema.
    ///   - `Ignore`: the field is silently dropped.
    ///
    /// # Arguments
    ///
    /// * `doc` - The document to normalise in place.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LaurusError::invalid_argument`] when:
    ///
    /// - A field name collides with the reserved namespace.
    /// - Strict policy is set and an undeclared field is encountered.
    /// - A declared field's value cannot be coerced to its type.
    /// - Dynamic policy is set and an undeclared field has a value whose
    ///   type cannot be inferred (e.g. raw vectors or bytes).
    async fn apply_dynamic_schema(&self, doc: &mut Document) -> Result<()> {
        // 1. Validate reserved field-name namespace for user-supplied keys.
        //    `_id` was just injected by the engine and is always allowed.
        for name in doc.fields.keys() {
            if name == schema::RESERVED_ID_FIELD {
                continue;
            }
            schema::validate_field_name(name)?;
        }

        // 2. Snapshot the current policy and declared-field set.
        let (policy, declared): (
            schema::DynamicFieldPolicy,
            std::collections::HashSet<String>,
        ) = {
            let s = self.schema.read();
            (s.dynamic_field_policy, s.fields.keys().cloned().collect())
        };

        // 3. Partition fields into declared vs undeclared.
        let mut undeclared: Vec<(String, crate::data::DataValue)> = Vec::new();
        let mut declared_updates: Vec<(String, crate::data::DataValue)> = Vec::new();
        for (name, value) in doc.fields.drain() {
            if name == schema::RESERVED_ID_FIELD || declared.contains(&name) {
                declared_updates.push((name, value));
            } else {
                undeclared.push((name, value));
            }
        }

        // 4. Handle undeclared fields per policy.
        match policy {
            schema::DynamicFieldPolicy::Strict => {
                if !undeclared.is_empty() {
                    let names: Vec<&str> = undeclared.iter().map(|(n, _)| n.as_str()).collect();
                    return Err(crate::error::LaurusError::invalid_argument(format!(
                        "undeclared fields {names:?} are not permitted \
                         (DynamicFieldPolicy::Strict)"
                    )));
                }
            }
            schema::DynamicFieldPolicy::Ignore => {
                // Silently drop undeclared fields.
                for (name, _) in &undeclared {
                    log::debug!(
                        target: "laurus::engine::dynamic_schema",
                        "dropping undeclared field '{name}' \
                         (DynamicFieldPolicy::Ignore)",
                    );
                }
                undeclared.clear();
            }
            schema::DynamicFieldPolicy::Dynamic => {
                // Infer a FieldOption for each undeclared field and add it to
                // the schema. Keep the original values on the document so they
                // are indexed under the newly-added fields.
                let mut kept: Vec<(String, crate::data::DataValue)> = Vec::new();
                for (name, value) in undeclared.drain(..) {
                    match type_inference::infer_option_from_data_value(&value)? {
                        Some(option) => {
                            match self.add_field(&name, option).await {
                                Ok(_) => {}
                                Err(e) => {
                                    // Another concurrent ingest may have added
                                    // this field in the meantime. Accept it
                                    // silently; any other failure propagates.
                                    let msg = e.to_string();
                                    if !msg.contains("already exists") {
                                        return Err(e);
                                    }
                                }
                            }
                            kept.push((name, value));
                        }
                        None => {
                            // Null value — skip this field entirely.
                        }
                    }
                }
                undeclared = kept;
            }
        }

        // 5. Coerce declared-field values to their declared types.
        let coerced_declared: Vec<(String, crate::data::DataValue)> = {
            let s = self.schema.read();
            let mut out = Vec::with_capacity(declared_updates.len());
            for (name, value) in declared_updates {
                if name == schema::RESERVED_ID_FIELD {
                    out.push((name, value));
                    continue;
                }
                // The field is declared (we partitioned above) so this lookup
                // is infallible in practice, but guard just in case.
                match s.fields.get(&name) {
                    Some(option) => {
                        let coerced = match type_coercion::coerce_value(&name, option, value) {
                            Ok(v) => v,
                            Err(e) => match policy {
                                schema::DynamicFieldPolicy::Ignore => {
                                    log::debug!(
                                        target: "laurus::engine::dynamic_schema",
                                        "dropping declared field '{name}' due to coercion \
                                         failure ({e}) (DynamicFieldPolicy::Ignore)",
                                    );
                                    continue;
                                }
                                _ => return Err(e),
                            },
                        };
                        out.push((name, coerced));
                    }
                    None => out.push((name, value)),
                }
            }
            out
        };

        // 6. Re-populate the document with processed fields.
        for (name, value) in coerced_declared {
            doc.fields.insert(name, value);
        }
        for (name, value) in undeclared {
            doc.fields.insert(name, value);
        }

        Ok(())
    }

    /// Delete all documents (including chunks) by external ID.
    ///
    /// Looks up all internal document IDs associated with the given external
    /// `id` via the `_id` field in the lexical index, then removes each one
    /// from both the lexical and vector stores. A WAL delete entry is written
    /// for each matched document before mutation.
    ///
    /// If no documents match the given ID, the operation completes
    /// successfully without error (non-existent IDs are silently ignored).
    ///
    /// # Parameters
    ///
    /// - `id` - The external document identifier to delete.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write, lexical deletion, or vector
    /// deletion fails for any matched document.
    pub async fn delete_documents(&self, id: &str) -> Result<()> {
        // Same-id serialization as the upsert path (#1049): without it a
        // delete can interleave between a concurrent upsert's own delete and
        // its add, so the upsert's new version outlives the delete that was
        // acknowledged after it.
        let _id_guard = self.id_lock(id).lock().await;
        self.delete_documents_internal(id).await?;
        // Re-assert per-record durability: a concurrent batch may hold a WAL
        // sync-deferral scope, which would otherwise leave these acknowledged
        // deletes unsynced. A no-op when the appends already self-synced.
        self.log.ensure_per_record_durability()?;
        Ok(())
    }

    /// Body of [`Self::delete_documents`] without the per-record durability
    /// re-assertion, so the batch-ingest path (whose put semantics
    /// delete-first per document inside a WAL sync-deferral scope) does not
    /// fsync once per deleted document.
    ///
    /// # Parameters
    ///
    /// - `id` - The external document identifier to delete.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL write, lexical deletion, or vector
    /// deletion fails for any matched document.
    /// The shard serializing mutations of `id` (#1049).
    ///
    /// # Parameters
    ///
    /// - `id` - The external document identifier to map to a shard.
    ///
    /// # Returns
    ///
    /// The [`tokio::sync::Mutex`] guarding every mutation of this id.
    fn id_lock(&self, id: &str) -> &tokio::sync::Mutex<()> {
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        id.hash(&mut hasher);
        // ID_LOCK_SHARDS is a power of two, so the mask is exact.
        &self.id_locks[hasher.finish() as usize & (ID_LOCK_SHARDS - 1)]
    }

    async fn delete_documents_internal(&self, id: &str) -> Result<()> {
        let doc_ids = self.lexical.find_doc_ids_by_term("_id", id)?;
        for doc_id in doc_ids {
            // 1. Write to log
            let seq = self.log.append_delete(doc_id, id)?;
            // 2. Delete from Lexical
            self.lexical.delete_document_by_internal_id(doc_id)?;
            // 3. Delete from Vector
            self.vector.delete_document_by_internal_id(doc_id).await?;
            // 4. Update trackers AFTER both deletes succeed.
            // This ensures failed deletes are retried on recovery. See
            // `index_internal`'s comment on `vector.set_last_wal_seq`
            // (Issue #948: fans out to every field identically; the
            // store's own `last_wal_seq()` is a min across fields).
            self.lexical.set_last_wal_seq(seq)?;
            self.vector.set_last_wal_seq(seq);
            // Publish the high-water mark only once both stores hold the delete
            // (Issue #876) — see `index_internal` for the rationale.
            self.applied_seq.fetch_max(seq, Ordering::AcqRel);
        }
        Ok(())
    }

    /// Commit changes to both stores and truncate the WAL.
    ///
    /// Persists state in a fixed order — the **commit durability ladder** — that
    /// makes a crash at any step recoverable (Issue #821):
    ///
    /// 1. `flush_wal()` — force the WAL durable (the hard barrier).
    /// 2. `lexical.commit()` — materialize + fsync the lexical store. This is
    ///    where the lexical `last_wal_seq` checkpoint is persisted.
    /// 3. `vector.commit()` — materialize + fsync the vector store.
    /// 4. `commit_documents()` — materialize + fsync the document store.
    /// 5. `truncate_retaining_after(applied_before)` — discard everything this
    ///    commit covered; **retain** anything appended (or still being applied)
    ///    concurrently (Issue #876).
    ///
    /// This order upholds two invariants. First, `last_wal_seq` is persisted
    /// only in step 2+, always *after* the step-1 barrier, so a committed index
    /// can never reference a WAL record that is not yet durable. Second, every
    /// store is fully fsync'd (steps 2–4) before the WAL is truncated (step 5),
    /// so the WAL is discarded only once the data it described is durable. A
    /// crash between any two steps therefore leaves enough in the WAL for the
    /// idempotent replay in [`Self::recover`] to reconstruct a consistent state.
    ///
    /// This method itself is **not** serialized against concurrent
    /// `put`/`add`/`delete` calls. After a successful commit with no concurrent
    /// mutation, the WAL is empty and all data is durable, exactly as before
    /// Issue #876. Under a mutation that raced this commit, the WAL is **not**
    /// necessarily empty afterward: it retains that mutation's record so the
    /// next crash can still recover it. `CommitPolicy::Interval`'s background
    /// timer runs this same ladder, so the same caveat applies to auto-commits.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing the WAL, committing the lexical store,
    /// vector store, document store, or truncating the WAL fails.
    pub async fn commit(&self) -> Result<()> {
        run_commit_ladder(
            &self.lexical,
            &self.vector,
            &self.log,
            &self.docs_since_commit,
            &self.applied_seq,
        )
        .await
    }

    /// Force every appended-but-unsynced WAL record durable, without a full
    /// [`commit`](Self::commit) (Issue #542).
    ///
    /// Under the default [`WalSyncPolicy::PerRecord`] this is a near-no-op: each
    /// `add`/`delete` already fsyncs, so there is nothing pending. Under
    /// [`WalSyncPolicy::Group`] appends defer their fsync, so this is the way to
    /// bound the crash-loss window at an application-chosen point — analogous to
    /// SQLite's manual WAL checkpoint — without paying the cost of materializing
    /// the lexical/vector indexes that [`commit`](Self::commit) entails.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or fsyncing the open WAL writer fails.
    pub fn flush_wal(&self) -> Result<()> {
        self.log.flush_wal()
    }

    /// Get combined index statistics from both the lexical and vector stores.
    ///
    /// Returns an [`EngineStats`] containing:
    /// - `document_count` from the lexical index (authoritative source).
    /// - Per-field vector statistics from the vector store (empty when no
    ///   vector fields are defined in the schema).
    ///
    /// # Errors
    ///
    /// Returns an error if the lexical index statistics cannot be retrieved.
    pub fn stats(&self) -> Result<EngineStats> {
        let lexical_stats = self.lexical.stats()?;

        let vector_fields = match self.vector.stats() {
            Ok(vs) => vs.fields,
            Err(_) => std::collections::HashMap::new(),
        };

        // doc_count includes deleted documents (soft-deleted, pending merge).
        // Subtract deleted_count for the live document count.
        let live_count = lexical_stats
            .doc_count
            .saturating_sub(lexical_stats.deleted_count);

        Ok(EngineStats {
            document_count: live_count,
            vector_fields,
        })
    }

    /// Return a clone of the current schema.
    ///
    /// This can be used to inspect the schema after dynamic field additions
    /// or to persist it to storage (e.g., `schema.toml`).
    pub fn schema(&self) -> Schema {
        self.schema.read().clone()
    }

    /// Returns the embedder used by the vector store.
    ///
    /// This is useful for constructing a [`VectorQueryParser`] or
    /// [`UnifiedQueryParser`] that shares the same embedder configuration
    /// as the engine.
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        self.vector.embedder()
    }

    /// Create a [`UnifiedQueryParser`] configured for this engine.
    ///
    /// The returned parser uses the engine's analyzer for lexical queries
    /// and the engine's embedder for vector queries. Default fields are
    /// derived from the schema: `default_fields` for lexical queries, and
    /// all vector fields for vector queries.
    ///
    /// # Errors
    ///
    /// Returns an error if the lexical query parser cannot be constructed
    /// (e.g. the analyzer is misconfigured).
    pub fn unified_query_parser(&self) -> Result<self::query::UnifiedQueryParser> {
        let lexical_parser = self.lexical.query_parser()?;
        let embedder = self.embedder();

        let schema = self.schema.read();
        let vector_fields: Vec<String> = schema
            .fields
            .iter()
            .filter(|(_, opt)| opt.is_vector())
            .map(|(name, _)| name.clone())
            .collect();

        let vector_field_set: std::collections::HashSet<String> =
            vector_fields.iter().cloned().collect();

        // All declared field names (lexical + vector), used by the parser to
        // reject typo'd field references at parse time.
        //
        // `_id` is injected by the engine at ingest and indexed with a
        // `KeywordAnalyzer` (see `split_schema`), but it is never present
        // in `schema.fields` — users cannot declare it. Add it explicitly
        // so that `_id:doc-001` keeps working instead of being rejected
        // as an unknown field.
        let mut known_fields: std::collections::HashSet<String> =
            schema.fields.keys().cloned().collect();
        known_fields.insert(schema::RESERVED_ID_FIELD.to_string());

        let mut vector_parser = crate::vector::query::parser::VectorQueryParser::new(embedder);
        if !vector_fields.is_empty() {
            vector_parser = vector_parser.with_default_fields(vector_fields);
        }
        if let Some(cache) = &self.embedding_cache {
            // Share the engine's cache so DSL queries hit the same entries
            // as the direct Payloads path (Issue #678).
            vector_parser = vector_parser.with_embedding_cache(cache.clone());
        }

        Ok(
            self::query::UnifiedQueryParser::new(lexical_parser, vector_parser, vector_field_set)
                .with_known_fields(known_fields),
        )
    }

    /// Persist the current schema via the configured hook, if any (Issue #1078).
    ///
    /// No-op when [`EngineBuilder::persist_schema_with`] was not called
    /// (the caller remains responsible for persisting the schema itself, as
    /// before). Called by [`Self::add_field`] and [`Self::delete_field`]
    /// after they update the in-memory schema.
    fn persist_schema(&self, schema: &Schema) -> Result<()> {
        if let Some(hook) = &self.schema_persist_hook {
            hook(schema)?;
        }
        Ok(())
    }

    /// Dynamically add a new field to the engine at runtime.
    ///
    /// This method registers the field in both the engine schema and the
    /// appropriate underlying store (lexical or vector). Only field addition
    /// is supported; removal or type changes are not allowed.
    ///
    /// After adding a field, new documents can include values for this field
    /// and searches can target it. Existing documents are unaffected (they
    /// simply do not have a value for the new field).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name. Must not collide with an existing field.
    /// * `option` - The field configuration (e.g., `FieldOption::Text`,
    ///   `FieldOption::Hnsw`, etc.).
    ///
    /// # Returns
    ///
    /// Returns the updated [`Schema`] on success. If
    /// [`EngineBuilder::persist_schema_with`] was configured, the returned
    /// schema has already been persisted via that hook; otherwise the
    /// caller remains responsible for persisting it (e.g., writing
    /// `schema.toml`).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A field with the same name already exists.
    /// - The field references an unknown analyzer or embedder.
    /// - The underlying store rejects the field.
    /// - A configured schema-persist hook fails (the in-memory schema has
    ///   already been updated at that point; this is not rolled back).
    pub async fn add_field(&self, name: &str, option: schema::FieldOption) -> Result<Schema> {
        // 1a. Reject reserved field names (e.g. `_`-prefixed except `_id`).
        schema::validate_field_name(name)?;

        // 1. Check for duplicates.
        {
            let schema = self.schema.read();
            if schema.fields.contains_key(name) {
                return Err(crate::error::LaurusError::invalid_argument(format!(
                    "Field '{name}' already exists in the schema"
                )));
            }
        }

        // 2. Register in the appropriate store.
        if option.is_lexical() {
            // Resolve the per-field analyzer if configured.
            let field_analyzer = if let schema::FieldOption::Text(ref text_opt) = option
                && let Some(ref analyzer_spec) = text_opt.analyzer
            {
                let schema = self.schema.read();
                let analyzer = crate::analysis::analyzer::registry::create_analyzer_from_spec(
                    analyzer_spec,
                    &schema.analyzers,
                    &self.runtime_analyzers,
                )
                .map_err(|e| {
                    crate::error::LaurusError::invalid_argument(format!(
                        "Failed to resolve analyzer for field '{name}': {e}"
                    ))
                })?;
                Some(analyzer)
            } else {
                None
            };

            let lexical_opt = option
                .to_lexical()
                .expect("is_lexical() was true but to_lexical() returned None");
            self.lexical.add_field(name, lexical_opt, field_analyzer)?;
        }

        if option.is_vector() {
            // Resolve the per-field embedder if configured.
            // Clone the embedder definition out of the schema lock before
            // calling the async factory so that the non-Send parking_lot
            // guard is not held across an await point.
            let field_embedder = if let Some(embedder_name) = option.embedder_name() {
                let embedder_def = {
                    let schema = self.schema.read();
                    schema.embedders.get(embedder_name).cloned()
                };
                if let Some(def) = embedder_def {
                    Some(
                        crate::embedding::registry::create_embedder_from_definition(
                            embedder_name,
                            &def,
                        )
                        .await?,
                    )
                } else {
                    None
                }
            } else {
                None
            };

            let vector_opt = option
                .to_vector()
                .expect("is_vector() was true but to_vector() returned None");
            self.vector
                .add_field(name, &vector_opt, field_embedder)
                .await?;
        }

        // 3. Update the schema.
        {
            let mut schema = self.schema.write();
            schema.fields.insert(name.to_string(), option);
        }

        let updated = self.schema.read().clone();
        self.persist_schema(&updated)?;
        Ok(updated)
    }

    /// Dynamically remove a field from the engine schema at runtime.
    ///
    /// This removes the field definition from the schema so that it is no longer
    /// available for indexing or searching. Existing data already stored in the
    /// index is **not** deleted; it simply becomes inaccessible through the
    /// normal query path.
    ///
    /// For lexical fields, the field is also removed from the underlying
    /// [`LexicalStore`] (if it was dynamically added) and any per-field analyzer
    /// is unregistered. For vector fields, the per-field embedder is
    /// unregistered and writer/searcher caches are invalidated.
    ///
    /// If the deleted field appears in [`Schema::default_fields`], it is removed
    /// from that list as well.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the field to delete
    ///
    /// # Returns
    ///
    /// The updated [`Schema`] after the field has been removed. If
    /// [`EngineBuilder::persist_schema_with`] was configured, it has
    /// already been persisted via that hook; otherwise the caller remains
    /// responsible for persisting it.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No field with the given name exists in the schema.
    /// - The underlying store rejects the deletion.
    /// - A configured schema-persist hook fails (the in-memory schema has
    ///   already been updated at that point; this is not rolled back).
    pub async fn delete_field(&self, name: &str) -> Result<Schema> {
        // 1. Check that the field exists.
        let option = {
            let schema = self.schema.read();
            schema.fields.get(name).cloned().ok_or_else(|| {
                crate::error::LaurusError::invalid_argument(format!(
                    "Field '{name}' does not exist in the schema"
                ))
            })?
        };

        // 2. Remove from the appropriate store.
        if option.is_lexical() {
            self.lexical.delete_field(name)?;
        }

        if option.is_vector() {
            self.vector.delete_field(name).await?;
        }

        // 3. Update the schema.
        {
            let mut schema = self.schema.write();
            schema.fields.remove(name);
            schema.default_fields.retain(|f| f != name);
        }

        let updated = self.schema.read().clone();
        self.persist_schema(&updated)?;
        Ok(updated)
    }

    /// Resolve a [`LexicalSearchQuery`] into a concrete [`Query`] object.
    ///
    /// If the query is already an `Obj` variant, it is returned as-is.
    /// If it is a `Dsl` string, it is parsed using the lexical store's
    /// query parser (which includes the configured analyzer and default fields).
    ///
    /// # Arguments
    ///
    /// * `query` - The query to resolve.
    ///
    /// # Errors
    ///
    /// Returns an error if the DSL string cannot be parsed.
    fn resolve_query(
        &self,
        query: crate::lexical::search::searcher::LexicalSearchQuery,
    ) -> Result<Box<dyn crate::lexical::query::Query>> {
        match query {
            crate::lexical::search::searcher::LexicalSearchQuery::Obj(q) => Ok(q),
            crate::lexical::search::searcher::LexicalSearchQuery::Dsl(dsl) => {
                let parser = self.lexical.query_parser()?;
                parser.parse(&dsl)
            }
        }
    }

    /// Resolve a [`SearchQuery`](self::search::SearchQuery) into internal
    /// search request types for the lexical and vector stores.
    ///
    /// This method converts the public query enum variants into the
    /// internal `LexicalSearchRequest` and `VectorSearchRequest` types,
    /// applying the relevant options.
    ///
    /// # Parameters
    ///
    /// * `query` - The search query to resolve.
    /// * `offset` - The pagination offset from the search request.
    /// * `limit` - The result limit from the search request.
    /// * `fusion_algorithm` - The caller-specified fusion algorithm, if any.
    /// * `lexical_options` - Lexical search options.
    /// * `vector_options` - Vector search options.
    ///
    /// # Errors
    ///
    /// Panics (via `unreachable!`) if called with `SearchQuery::Dsl`, which
    /// must be resolved before calling this method.
    #[allow(clippy::type_complexity)]
    fn resolve_search_query_from_parts(
        &self,
        query: self::search::SearchQuery,
        offset: usize,
        limit: usize,
        fusion_algorithm: Option<FusionAlgorithm>,
        lexical_options: &self::search::LexicalSearchOptions,
        vector_options: &self::search::VectorSearchOptions,
    ) -> Result<(
        Option<crate::lexical::search::searcher::LexicalSearchRequest>,
        Option<crate::vector::store::request::VectorSearchRequest>,
        Option<FusionAlgorithm>,
        self::search::HybridMode,
    )> {
        let fetch_count = offset.saturating_add(limit);

        match query {
            self::search::SearchQuery::Dsl(_) => {
                // DSL should be parsed by UnifiedQueryParser before calling this
                unreachable!("DSL should be resolved before resolve_search_query_from_parts")
            }
            self::search::SearchQuery::Lexical(lexical_query) => {
                let lex_req = crate::lexical::search::searcher::LexicalSearchRequest {
                    query: lexical_query,
                    params: crate::lexical::search::searcher::LexicalSearchParams {
                        limit: 0, // Controlled by engine
                        min_score: lexical_options.min_score,
                        load_documents: true,
                        timeout_ms: lexical_options.timeout_ms,
                        parallel: lexical_options.parallel,
                        sort_by: lexical_options.sort_by.clone(),
                    },
                    field_boosts: lexical_options.field_boosts.clone(),
                };
                Ok((Some(lex_req), None, None, self::search::HybridMode::Union))
            }
            self::search::SearchQuery::Vector(vector_query) => {
                let vec_req = self.build_vector_request(vector_query, vector_options, fetch_count);
                Ok((None, Some(vec_req), None, self::search::HybridMode::Union))
            }
            self::search::SearchQuery::Hybrid {
                lexical,
                vector,
                mode,
            } => {
                let lex_req = crate::lexical::search::searcher::LexicalSearchRequest {
                    query: lexical,
                    params: crate::lexical::search::searcher::LexicalSearchParams {
                        limit: 0, // Controlled by engine
                        min_score: lexical_options.min_score,
                        load_documents: true,
                        timeout_ms: lexical_options.timeout_ms,
                        parallel: lexical_options.parallel,
                        sort_by: lexical_options.sort_by.clone(),
                    },
                    field_boosts: lexical_options.field_boosts.clone(),
                };
                let vec_req = self.build_vector_request(vector, vector_options, fetch_count);
                let fusion = fusion_algorithm.or(Some(FusionAlgorithm::RRF { k: 60.0 }));
                Ok((Some(lex_req), Some(vec_req), fusion, mode))
            }
        }
    }

    /// Build a [`VectorSearchRequest`](crate::vector::store::request::VectorSearchRequest)
    /// from a [`VectorSearchQuery`](self::search::VectorSearchQuery) and options.
    ///
    /// # Parameters
    ///
    /// * `query` - The vector search query (payloads or pre-embedded vectors).
    /// * `opts` - Vector search options (score mode, min score).
    /// * `limit` - Maximum number of results to fetch.
    fn build_vector_request(
        &self,
        query: self::search::VectorSearchQuery,
        opts: &self::search::VectorSearchOptions,
        limit: usize,
    ) -> crate::vector::store::request::VectorSearchRequest {
        crate::vector::store::request::VectorSearchRequest {
            query,
            params: crate::vector::search::searcher::VectorSearchParams {
                fields: None,
                limit,
                score_mode: opts.score_mode,
                overfetch: 2.0,
                min_score: opts.min_score,
                allowed_ids: None,
                allowed_filter: None,
                rerank_factor: opts.rerank_factor,
                ef_search: opts.ef_search,
            },
        }
    }

    /// Get all documents (including chunks) by external ID.
    ///
    /// Only fields marked as stored in the schema are included in the
    /// returned documents. If no documents match the given ID, an empty
    /// `Vec` is returned (not an error).
    ///
    /// # Parameters
    ///
    /// - `id` - The external document identifier to look up.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal ID lookup or document retrieval fails.
    pub async fn get_documents(&self, id: &str) -> Result<Vec<Document>> {
        let doc_ids = self.lexical.find_doc_ids_by_term("_id", id)?;
        let mut docs = Vec::with_capacity(doc_ids.len());
        for doc_id in doc_ids {
            if let Some(doc) = self.get_document_by_internal_id(doc_id)? {
                docs.push(doc);
            }
        }
        Ok(docs)
    }

    /// Check if a field should be stored, decided against an
    /// already-acquired schema guard.
    ///
    /// - `_id`: always stored (system field)
    /// - Lexical fields: stored only if `stored=true`
    /// - Vector fields: always stored
    /// - Unknown fields: not stored
    ///
    /// Takes the guard rather than locking internally so callers that
    /// test many fields (i.e. [`Self::filter_stored_fields`] and the
    /// search-result resolver, which run once per returned hit) acquire
    /// the schema lock once instead of once per field (#1010).
    ///
    /// # Arguments
    ///
    /// * `schema` - The already-locked schema.
    /// * `name` - The field name to test.
    ///
    /// # Returns
    ///
    /// `true` when the field's value is kept in the document store.
    fn is_field_stored_in(schema: &Schema, name: &str) -> bool {
        use crate::engine::schema::FieldOption;

        if name == "_id" {
            return true;
        }
        if let Some(field_opt) = schema.fields.get(name) {
            match field_opt {
                FieldOption::Text(o) => o.stored,
                FieldOption::Integer(o) => o.stored,
                FieldOption::Float(o) => o.stored,
                FieldOption::Boolean(o) => o.stored,
                FieldOption::DateTime(o) => o.stored,
                FieldOption::Geo(o) => o.stored,
                FieldOption::Geo3d(o) => o.stored,
                FieldOption::Bytes(o) => o.stored,
                // Vector fields are always stored
                FieldOption::Hnsw(_) | FieldOption::Flat(_) | FieldOption::Ivf(_) => true,
            }
        } else {
            false
        }
    }

    /// Filter a document to only include fields that should be stored.
    ///
    /// The document log (WAL) stores ALL fields for recovery, but the
    /// document store only keeps stored fields to save space.
    fn filter_stored_fields(&self, doc: &Document) -> Document {
        // One schema lock for the whole document, not one per field
        // (#1010): this runs once per returned search hit.
        let schema = self.schema.read();
        let mut stored_doc = Document::new();
        for (name, val) in &doc.fields {
            if Self::is_field_stored_in(&schema, name) {
                stored_doc.fields.insert(name.clone(), val.clone());
            }
        }
        stored_doc
    }

    /// Get a document by its internal ID (private helper).
    ///
    /// Retrieves from the document log and filters out non-stored fields.
    fn get_document_by_internal_id(&self, doc_id: u64) -> Result<Option<Document>> {
        let doc = self.log.get_document(doc_id)?;

        if let Some(doc) = doc {
            Ok(Some(self.filter_stored_fields(&doc)))
        } else {
            Ok(None)
        }
    }

    /// Batch-resolve external IDs and documents for multiple internal IDs.
    ///
    /// Fetches all documents in one pass through the document store,
    /// reducing per-document lock acquisition overhead.
    ///
    /// # Arguments
    ///
    /// * `internal_ids` - Slice of internal document IDs.
    ///
    /// # Returns
    ///
    /// A map from internal ID to `(external_id, Option<Document>)`.
    fn resolve_ids_and_documents_batch(
        &self,
        internal_ids: &[u64],
    ) -> Result<HashMap<u64, (String, Option<Document>)>> {
        // One batched store call instead of one lookup per id (#1010):
        // the store opens each segment file once and seeks in offset
        // order, and consults / populates the document cache for the
        // whole batch under a single lock.
        let mut fetched = self.log.get_documents_batch(internal_ids)?;

        // One schema lock for every document in the batch, rather than
        // one per field per document (#1010).
        let schema = self.schema.read();
        let mut results = HashMap::with_capacity(internal_ids.len());
        for &id in internal_ids {
            match fetched.remove(&id) {
                Some(doc) => {
                    let external_id = doc
                        .fields
                        .get("_id")
                        .and_then(|v| v.as_text())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("unknown_{}", id));
                    let mut filtered = Document::new();
                    for (name, val) in doc.fields {
                        if Self::is_field_stored_in(&schema, &name) {
                            filtered.fields.insert(name, val);
                        }
                    }
                    results.insert(id, (external_id, Some(filtered)));
                }
                None => {
                    results.insert(id, (format!("unknown_{}", id), None));
                }
            }
        }
        Ok(results)
    }

    /// Split the unified schema into specialized configs.
    async fn split_schema(
        schema: &Schema,
        analyzer: Option<Arc<dyn Analyzer>>,
        embedder: Option<Arc<dyn Embedder>>,
        runtime_analyzers: &HashMap<String, Arc<dyn Analyzer>>,
    ) -> Result<(LexicalIndexConfig, VectorIndexConfig)> {
        // Construct Lexical Config
        let analyzer = match analyzer {
            Some(a) => a,
            None => Arc::new(StandardAnalyzer::new()?),
        };

        // If the user passed a PerFieldAnalyzer, clone it and ensure _id uses KeywordAnalyzer.
        // Otherwise, wrap the simple analyzer in a new PerFieldAnalyzer.
        let per_field_analyzer =
            if let Some(existing) = analyzer.as_any().downcast_ref::<PerFieldAnalyzer>() {
                let pfa = existing.clone();
                pfa.add_analyzer("_id", Arc::new(KeywordAnalyzer::new()));
                pfa
            } else {
                let pfa = PerFieldAnalyzer::new(analyzer);
                pfa.add_analyzer("_id", Arc::new(KeywordAnalyzer::new()));
                pfa
            };

        // Register per-field analyzers declared in the schema.
        // Resolution order: parameterized built-in → built-in name → custom
        // definition in schema.analyzers.
        for (name, field_option) in &schema.fields {
            if let schema::FieldOption::Text(text_opt) = field_option
                && let Some(spec) = &text_opt.analyzer
            {
                let field_analyzer =
                    crate::analysis::analyzer::registry::create_analyzer_from_spec(
                        spec,
                        &schema.analyzers,
                        runtime_analyzers,
                    )
                    .map_err(|e| {
                        crate::error::LaurusError::invalid_argument(format!(
                            "Failed to resolve analyzer for field '{name}': {e}"
                        ))
                    })?;
                per_field_analyzer.add_analyzer(name, field_analyzer);
            }
        }

        let mut lexical_builder =
            LexicalIndexConfig::builder().analyzer(Arc::new(per_field_analyzer));

        if !schema.default_fields.is_empty() {
            lexical_builder = lexical_builder.default_fields(schema.default_fields.clone());
        }

        for (name, field_option) in &schema.fields {
            if let Some(lexical_opt) = field_option.to_lexical() {
                lexical_builder = lexical_builder.add_field(name, lexical_opt);
            }
        }

        let lexical_config = lexical_builder.build();

        // Construct Vector Config — resolve embedder from schema if not explicitly provided.
        let embedder = if embedder.is_some() {
            embedder
        } else if !schema.embedders.is_empty() {
            // Build a PerFieldEmbedder from schema.embedders declarations.
            let mut embedder_cache: HashMap<String, Arc<dyn crate::embedding::embedder::Embedder>> =
                HashMap::new();
            let default_embedder: Arc<dyn crate::embedding::embedder::Embedder> =
                Arc::new(crate::embedding::precomputed::PrecomputedEmbedder::new());
            let per_field = crate::embedding::per_field::PerFieldEmbedder::new(default_embedder);

            for (name, field_option) in &schema.fields {
                if let Some(embedder_name) = field_option.embedder_name() {
                    let emb = if let Some(cached) = embedder_cache.get(embedder_name) {
                        cached.clone()
                    } else {
                        let def = schema.embedders.get(embedder_name).ok_or_else(|| {
                            crate::error::LaurusError::invalid_argument(format!(
                                "Unknown embedder '{embedder_name}' for field '{name}': \
                                 not defined in schema.embedders"
                            ))
                        })?;
                        let emb = crate::embedding::registry::create_embedder_from_definition(
                            embedder_name,
                            def,
                        )
                        .await?;
                        embedder_cache.insert(embedder_name.to_string(), emb.clone());
                        emb
                    };
                    per_field.add_embedder(name, emb);
                }
            }

            let emb: Arc<dyn crate::embedding::embedder::Embedder> = Arc::new(per_field);
            Some(emb)
        } else {
            None
        };

        let mut vector_builder = VectorIndexConfig::builder();
        if let Some(embedder) = &embedder {
            vector_builder = vector_builder.embedder(embedder.clone());
        }

        for (name, field_option) in &schema.fields {
            if let Some(vector_opt) = field_option.to_vector() {
                vector_builder = vector_builder.add_field(name, vector_opt)?;
            }
        }

        let vector_config = vector_builder.build()?;

        Ok((lexical_config, vector_config))
    }

    /// Warm the vector searcher so the first vector / hybrid query does not pay
    /// the searcher-construction and page-fault cost (Issue #677).
    ///
    /// Delegates to [`VectorStore::warmup`](crate::vector::VectorStore::warmup):
    /// it eagerly builds the cached searcher (loading the reader) and pre-faults
    /// on-disk vector data into the OS page cache where applicable (HNSW `Mmap`
    /// mode). Call once after building the engine, before serving traffic.
    /// Safe to call multiple times; lexical search needs no warming.
    ///
    /// # Errors
    ///
    /// Returns an error if building the vector searcher (reader load) fails.
    pub fn warmup(&self) -> Result<()> {
        self.vector.warmup()
    }

    /// Sample up to `sample_size` vectors already committed for `field`,
    /// suitable as [`Self::train_pq_codebook`] input without a separate
    /// JSONL export (Issue #920).
    ///
    /// Ordered by ascending doc_id for determinism — the same "first N"
    /// semantics `laurus train pq-codebook`'s JSONL path already uses,
    /// just drawn from committed segments instead of a training file. See
    /// [`VectorStore::sample_field_vectors`](crate::vector::store::VectorStore::sample_field_vectors)
    /// for the full ordering/emptiness contract.
    ///
    /// # Arguments
    ///
    /// * `field` - Vector field to sample. An unknown or vector-less
    ///   field yields an empty `Vec`, not an error.
    /// * `sample_size` - Maximum number of vectors to return. `None`
    ///   returns every committed vector for the field.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining the reader or reading the field's
    /// vectors fails.
    pub fn sample_committed_vectors(
        &self,
        field: &str,
        sample_size: Option<usize>,
    ) -> Result<Vec<Vector>> {
        Ok(self
            .vector
            .sample_field_vectors(field, sample_size)?
            .into_iter()
            .map(|(_, v)| v)
            .collect())
    }

    /// Train a shared PQ codebook for `field` and persist it into the
    /// engine's vector storage namespace (Issue #631).
    ///
    /// The codebook is trained once on `vectors` and then reused by every
    /// segment write instead of re-running k-means on every commit and
    /// merge. Training is synchronous and CPU-bound (like
    /// [`stats`](Self::stats) / [`warmup`](Self::warmup)); expect seconds
    /// for thousands of training vectors.
    ///
    /// The codebook file is picked up when the index is (re)opened with the
    /// field's
    /// [`HnswOption::pq_codebook_path`](crate::vector::core::field::HnswOption::pq_codebook_path)
    /// naming it — an engine instance that is already open does **not** hot-swap
    /// codebooks mid-flight (intentional: a codebook must not change between
    /// the commits of one session). Train first, then open the engine that
    /// will ingest.
    ///
    /// When the field's distance metric is Cosine, the training set is
    /// L2-normalized before k-means so the codebook matches the normalized
    /// vectors the writer encodes (the #794 basis-mismatch trap).
    ///
    /// # Arguments
    ///
    /// * `field` - Schema field to train for. Must be an HNSW vector field
    ///   configured with
    ///   [`QuantizationMethod::ProductQuantization`](crate::vector::core::quantization::QuantizationMethod::ProductQuantization).
    /// * `vectors` - Training vectors; each must have the field's configured
    ///   dimension. A representative sample of the corpus (thousands of
    ///   vectors) is enough — the full corpus is not required.
    /// * `output` - Optional storage-relative file name override. `None`
    ///   writes to the field's configured `pq_codebook_path`, falling back
    ///   to the default `"{field}.pqcb"`. Pass `Some` to train a new
    ///   codebook alongside a live one (e.g. `"embedding.v2.pqcb"`) and
    ///   flip the schema afterwards.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LaurusError::invalid_argument`] when `field`
    /// is not in the schema, is not an HNSW field, or does not use PQ
    /// quantization; forwards
    /// [`train_and_write_pq_codebook`](crate::vector::index::pq_codebook::train_and_write_pq_codebook)'s
    /// errors for an empty/mis-dimensioned training set or storage I/O
    /// failures.
    pub fn train_pq_codebook(
        &self,
        field: &str,
        vectors: &[Vector],
        output: Option<&str>,
    ) -> Result<PqCodebookInfo> {
        use crate::vector::core::distance::DistanceMetric;
        use crate::vector::core::quantization::QuantizationMethod;
        use crate::vector::index::pq_codebook::{
            default_codebook_name, train_and_write_pq_codebook,
        };

        let (dimension, subvector_count, k, normalize, configured_path) = {
            let schema = self.schema.read();
            let Some(option) = schema.fields.get(field) else {
                return Err(crate::error::LaurusError::invalid_argument(format!(
                    "cannot train a PQ codebook: field '{field}' is not defined in the schema"
                )));
            };
            let schema::FieldOption::Hnsw(o) = option else {
                return Err(crate::error::LaurusError::invalid_argument(format!(
                    "cannot train a PQ codebook: field '{field}' is not an HNSW vector field \
                     (shared PQ codebooks are HNSW-only)"
                )));
            };
            // The quantizer variant fixes the centroid count the segments
            // will encode against: k=256 for standard 8-bit PQ, k=16 for
            // the FastScan 4-bit variant (Issue #920).
            let (subvector_count, k): (usize, u16) = match o.quantizer {
                QuantizationMethod::ProductQuantization { subvector_count } => {
                    (subvector_count, 256)
                }
                #[cfg(feature = "pq-fastscan")]
                QuantizationMethod::ProductQuantizationFastScan { subvector_count } => {
                    (subvector_count, 16)
                }
                other => {
                    return Err(crate::error::LaurusError::invalid_argument(format!(
                        "cannot train a PQ codebook: field '{field}' does not use \
                         ProductQuantization (quantizer is {other:?})"
                    )));
                }
            };
            (
                o.dimension,
                subvector_count,
                k,
                o.distance == DistanceMetric::Cosine,
                o.pq_codebook_path.clone(),
            )
        };

        let name = output
            .map(str::to_string)
            .or(configured_path)
            .unwrap_or_else(|| default_codebook_name(field));

        let codebook = train_and_write_pq_codebook(
            self.vector.storage().as_ref(),
            &name,
            dimension,
            subvector_count,
            k,
            normalize,
            vectors,
        )?;

        Ok(PqCodebookInfo {
            path: name,
            subvector_count: codebook.params.m as usize,
            centroids: codebook.params.k as usize,
            sub_dimension: codebook.params.sub_dim as usize,
            dimension,
            training_vectors: vectors.len(),
        })
    }

    /// Search the index.
    ///
    /// Supports three modes depending on how the
    /// [`SearchRequest`](self::search::SearchRequest) is configured:
    ///
    /// - **Unified query DSL** (via `query_dsl`): The query string is
    ///   parsed using [`UnifiedQueryParser`](self::query::UnifiedQueryParser)
    ///   to automatically extract lexical and/or vector components. This is
    ///   the recommended approach for external callers.
    /// - **Structured fields** (via `lexical_search_request` /
    ///   `vector_search_request`): Lower-level API for programmatic use.
    ///
    /// When `query_dsl` is set, it is parsed first, and the resulting
    /// lexical/vector components replace any explicitly set fields. The
    /// `fusion_algorithm`, `limit`, `offset`, and `filter_query` fields
    /// from the original request are preserved.
    ///
    /// After resolving the query source, the engine executes the
    /// appropriate search mode:
    ///
    /// - **Lexical only**: BM25-scored inverted index search.
    /// - **Vector only**: Nearest-neighbor vector search.
    /// - **Hybrid**: Both searches run and results are merged using the
    ///   configured `fusion_algorithm` (defaults to
    ///   [`RRF { k: 60.0 }`](FusionAlgorithm::RRF)).
    ///
    /// When a `filter_query` is present, it is evaluated first to determine
    /// the set of candidate documents. For lexical search, the filter is
    /// combined with the user query via a boolean `must` + `filter` clause.
    /// For vector search, the filter produces an `allowed_ids` list that
    /// restricts candidate scoring. If the filter matches zero documents,
    /// an empty result is returned immediately.
    ///
    /// When both lexical and vector search requests are present, both fetch
    /// limits are doubled (2x overfetch) to improve fusion quality.
    ///
    /// Results are paginated via `offset` and `limit` on the
    /// [`SearchRequest`](self::search::SearchRequest).
    ///
    /// # Parameters
    ///
    /// - `request` - The unified search request.
    ///
    /// # Errors
    ///
    /// Returns an error if the unified query parsing, filter query
    /// execution, lexical search, vector search, embedding, or document
    /// retrieval fails.
    pub async fn search(
        &self,
        request: self::search::SearchRequest,
    ) -> Result<Vec<self::search::SearchResult>> {
        // 0a. Resolve query to internal search components
        //
        // When the query is a DSL string, parse it with UnifiedQueryParser to
        // extract both lexical and vector components. For other variants,
        // construct the internal request types from the query + options.
        //
        // Destructure the request upfront so that `query` can be moved
        // independently while the remaining fields stay available.
        let self::search::SearchRequest {
            query: request_query,
            limit: request_limit,
            offset: request_offset,
            fusion_algorithm: request_fusion,
            filter_query: request_filter,
            lexical_options,
            vector_options,
        } = request;

        let (lexical_search_request, vector_search_request, fusion_algorithm, hybrid_mode) =
            match request_query {
                self::search::SearchQuery::Dsl(ref dsl) => {
                    let parser = self.unified_query_parser()?;
                    let parser = if let Some(fusion) = request_fusion {
                        parser.with_fusion(fusion)
                    } else {
                        parser
                    };
                    let parsed = parser.parse(dsl).await?;
                    // UnifiedQueryParser now returns Lexical/Vector/Hybrid variants
                    self.resolve_search_query_from_parts(
                        parsed.query,
                        request_offset,
                        request_limit,
                        request_fusion,
                        &lexical_options,
                        &vector_options,
                    )?
                }
                other => self.resolve_search_query_from_parts(
                    other,
                    request_offset,
                    request_limit,
                    request_fusion,
                    &lexical_options,
                    &vector_options,
                )?,
            };

        // 0b. Pre-process Filter
        let (allowed_filter, lexical_query_override) = if let Some(filter_query) = &request_filter {
            // Evaluate the filter through the snapshot-scoped query/filter cache
            // (Issue #578): a repeated filter is served as a cached doc-id set
            // instead of re-walking posting lists. Unlike the previous path,
            // this is not capped at 1M matches. The resulting `Arc<RoaringTreemap>`
            // is handed to the vector side as-is (Issue #739) — no `Vec<u64>` /
            // `AHashSet` round trip.
            let allowed = self.lexical.matching_doc_ids(filter_query.clone_box())?;

            if allowed.is_empty() {
                return Ok(Vec::new());
            }

            let new_lexical_query: Option<Box<dyn crate::lexical::query::Query>> =
                if let Some(lex_req) = &lexical_search_request {
                    use crate::lexical::query::boolean::BooleanQueryBuilder;
                    let user_query = self.resolve_query(lex_req.query.clone())?;
                    let bool_query = BooleanQueryBuilder::new()
                        .must(user_query)
                        .filter(filter_query.clone_box())
                        .build();
                    Some(Box::new(bool_query))
                } else {
                    None
                };

            (Some(allowed), new_lexical_query)
        } else {
            (None, None)
        };

        // 1. Execute Lexical Search
        let mut lexical_query_to_use = if lexical_query_override.is_some() {
            lexical_query_override
        } else if let Some(lex_req) = &lexical_search_request {
            Some(self.resolve_query(lex_req.query.clone())?)
        } else {
            None
        };

        if let Some(query) = &mut lexical_query_to_use
            && let Some(lex_req) = &lexical_search_request
            && !lex_req.field_boosts.is_empty()
        {
            query.apply_field_boosts(&lex_req.field_boosts);
        }

        let fetch_count = request_offset.saturating_add(request_limit);

        // Build the lexical request; the search itself runs in parallel below.
        let lex_req = if let Some(query) = &lexical_query_to_use {
            let q = query.clone_box();
            let overfetch_limit = if vector_search_request.is_some() {
                fetch_count.saturating_mul(2)
            } else {
                fetch_count
            };
            let mut req = crate::lexical::search::searcher::LexicalSearchRequest::new(q)
                .limit(overfetch_limit)
                .load_documents(false);
            // Carry the caller's resolved lexical params through (#942):
            // rebuilding the request from scratch silently dropped them.
            // `limit` / `load_documents` stay engine-controlled (overfetch
            // and engine-side document resolution). `sort_by` applies only
            // to lexical-only searches: under hybrid fusion the ranking is
            // the fused score, and a field-sorted candidate set would
            // neither survive fusion nor be a relevance top-K.
            if let Some(src) = &lexical_search_request {
                req.params.min_score = src.params.min_score;
                req.params.timeout_ms = src.params.timeout_ms;
                req.params.parallel = src.params.parallel;
                if vector_search_request.is_none() {
                    req.params.sort_by = src.params.sort_by.clone();
                }
            }
            Some(req)
        } else {
            None
        };

        // 2. Build the vector request — including the async payload embedding,
        // which must complete before the (synchronous) search runs in parallel
        // below.
        let vec_req = if let Some(vector_req) = &vector_search_request {
            let mut vreq = vector_req.clone();
            if lexical_search_request.is_some() && vreq.params.limit < fetch_count.saturating_mul(2)
            {
                vreq.params.limit = fetch_count.saturating_mul(2);
            }
            if let Some(filter) = &allowed_filter {
                vreq.params.allowed_filter = Some(filter.clone());
            }
            // Embed Payloads into Vectors before searching.
            // NOTE: When using VectorQueryParser, query is already Vectors
            // at parse time, so this block is skipped. This fallback remains for
            // VectorSearchRequestBuilder users who populate Payloads directly.
            if let crate::vector::search::searcher::VectorSearchQuery::Payloads(ref payloads) =
                vreq.query
            {
                use crate::data::DataValue;
                use crate::embedding::embedder::EmbedInput;
                use crate::vector::store::request::QueryVector;

                // Owned payload data for the embeddable (Text / Bytes) payloads,
                // keeping each one's field and weight. Non-text / non-bytes
                // payloads are skipped, as before. Owned buffers must outlive the
                // borrowed `EmbedInput`s handed to the batch call below.
                enum Owned {
                    Text(String),
                    Bytes(Vec<u8>, Option<String>),
                }
                let mut owned: Vec<(String, f32, Owned)> = Vec::new();
                for payload in payloads {
                    let data = match &payload.payload {
                        DataValue::Text(t) => Owned::Text(t.clone()),
                        DataValue::Bytes(b, m) => Owned::Bytes(b.clone(), m.clone()),
                        _ => continue,
                    };
                    owned.push((payload.field.clone(), payload.weight, data));
                }

                // Embed every payload in one batch (Issue #671) so a
                // batch-capable embedder pays one round trip instead of one per
                // payload, while preserving cache and per-field routing.
                let items: Vec<(String, EmbedInput<'_>)> = owned
                    .iter()
                    .map(|(field, _, data)| {
                        let input = match data {
                            Owned::Text(t) => EmbedInput::Text(t),
                            Owned::Bytes(b, m) => EmbedInput::Bytes(b, m.as_deref()),
                        };
                        (field.clone(), input)
                    })
                    .collect();
                let embedder = self.vector.embedder();
                let vectors =
                    embed_batch_with_cache(self.embedding_cache.as_ref(), &embedder, &items)
                        .await?;

                let query_vectors: Vec<QueryVector> = owned
                    .iter()
                    .zip(vectors)
                    .map(|((field, weight, _), vector)| QueryVector {
                        vector,
                        weight: *weight,
                        fields: Some(vec![field.clone()]),
                    })
                    .collect();
                vreq.query =
                    crate::vector::search::searcher::VectorSearchQuery::Vectors(query_vectors);
            }
            Some(vreq)
        } else {
            None
        };

        // Run the independent lexical and vector searches (#659). On native
        // builds both synchronous searches overlap via `rayon::join`, so the
        // hybrid latency drops from `lex + vec` toward `max(lex, vec)`. The
        // closures take disjoint immutable borrows of `self.lexical` /
        // `self.vector` plus the moved requests, so they are `Send`. On wasm32
        // (no rayon) they run sequentially. Fusion below is order-independent,
        // so the result set is identical either way.
        let run_lexical = || lex_req.map(|r| self.lexical.search(r)).transpose();
        let run_vector = || vec_req.map(|r| self.vector.search(r)).transpose();
        #[cfg(feature = "native")]
        let (lex_res, vec_res) = rayon::join(run_lexical, run_vector);
        #[cfg(not(feature = "native"))]
        let (lex_res, vec_res) = (run_lexical(), run_vector());
        let lexical_hits = lex_res?.map(|r| r.hits).unwrap_or_default();
        let vector_hits = vec_res?.map(|r| r.hits).unwrap_or_default();

        // 3. Fusion
        if lexical_search_request.is_some() && vector_search_request.is_some() {
            let algorithm = fusion_algorithm.unwrap_or(FusionAlgorithm::RRF { k: 60.0 });
            let mut results = self.fuse_results(
                lexical_hits,
                vector_hits,
                algorithm,
                hybrid_mode,
                fetch_count,
            )?;
            if request_offset > 0 {
                results = results.into_iter().skip(request_offset).collect();
            }
            results.truncate(request_limit);
            Ok(results)
        } else if !vector_hits.is_empty() {
            // Only vector results — batch-resolve external IDs and documents.
            let paginated: Vec<_> = vector_hits
                .into_iter()
                .skip(request_offset)
                .take(request_limit)
                .collect();
            let ids: Vec<u64> = paginated.iter().map(|h| h.doc_id).collect();
            let mut resolved = self.resolve_ids_and_documents_batch(&ids)?;
            let mut results = Vec::with_capacity(paginated.len());
            for hit in paginated {
                // `remove` moves the id and document out instead of
                // cloning them per hit (#1010). Hits carry distinct doc
                // ids — they come from a collector's top-K — so no entry
                // is needed twice.
                if let Some((external_id, document)) = resolved.remove(&hit.doc_id) {
                    results.push(SearchResult {
                        id: external_id,
                        score: hit.score,
                        document,
                    });
                }
            }
            Ok(results)
        } else {
            // Only lexical results (or both empty)
            let paginated: Vec<_> = lexical_hits
                .into_iter()
                .skip(request_offset)
                .take(request_limit)
                .collect();
            let ids: Vec<u64> = paginated.iter().map(|h| h.doc_id).collect();
            let mut resolved = self.resolve_ids_and_documents_batch(&ids)?;
            let mut results = Vec::with_capacity(paginated.len());
            for hit in paginated {
                // `remove` moves the id and document out instead of
                // cloning them per hit (#1010). Hits carry distinct doc
                // ids — they come from a collector's top-K — so no entry
                // is needed twice.
                if let Some((external_id, document)) = resolved.remove(&hit.doc_id) {
                    results.push(SearchResult {
                        id: external_id,
                        score: hit.score,
                        document,
                    });
                }
            }
            Ok(results)
        }
    }

    /// Combine results from lexical and vector engines.
    fn fuse_results(
        &self,
        lexical_hits: Vec<crate::lexical::query::SearchHit>,
        vector_hits: Vec<crate::vector::store::response::VectorHit>,
        fusion: FusionAlgorithm,
        mode: self::search::HybridMode,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // Collect doc_id sets upfront for intersection filtering.
        let lexical_ids: HashSet<u64> = lexical_hits.iter().map(|h| h.doc_id).collect();
        let vector_ids: HashSet<u64> = vector_hits.iter().map(|h| h.doc_id).collect();

        let mut fused_scores: HashMap<u64, (f32, Option<crate::data::Document>)> = HashMap::new();

        match fusion {
            FusionAlgorithm::RRF { k } => {
                for (rank, hit) in lexical_hits.into_iter().enumerate() {
                    let rrf_score = 1.0 / (k + (rank + 1) as f64);
                    let entry = fused_scores
                        .entry(hit.doc_id)
                        .or_insert((0.0, hit.document));
                    entry.0 += rrf_score as f32;
                }
                for (rank, hit) in vector_hits.into_iter().enumerate() {
                    let rrf_score = 1.0 / (k + (rank + 1) as f64);
                    let entry = fused_scores.entry(hit.doc_id).or_insert((0.0, None));
                    entry.0 += rrf_score as f32;
                }
            }
            FusionAlgorithm::WeightedSum {
                lexical_weight,
                vector_weight,
            } => {
                let lexical_min = lexical_hits
                    .iter()
                    .map(|h| h.score)
                    .fold(f32::INFINITY, f32::min);
                let lexical_max = lexical_hits
                    .iter()
                    .map(|h| h.score)
                    .fold(f32::NEG_INFINITY, f32::max);

                for hit in lexical_hits {
                    let norm_score = if lexical_max > lexical_min {
                        (hit.score - lexical_min) / (lexical_max - lexical_min)
                    } else {
                        1.0
                    };
                    let entry = fused_scores
                        .entry(hit.doc_id)
                        .or_insert((0.0, hit.document));
                    entry.0 += norm_score * lexical_weight;
                }

                let vector_min = vector_hits
                    .iter()
                    .map(|h| h.score)
                    .fold(f32::INFINITY, f32::min);
                let vector_max = vector_hits
                    .iter()
                    .map(|h| h.score)
                    .fold(f32::NEG_INFINITY, f32::max);

                for hit in vector_hits {
                    let norm_score = if vector_max > vector_min {
                        (hit.score - vector_min) / (vector_max - vector_min)
                    } else {
                        1.0
                    };
                    let entry = fused_scores.entry(hit.doc_id).or_insert((0.0, None));
                    entry.0 += norm_score * vector_weight;
                }
            }
        }

        // Intersection mode: keep only documents appearing in BOTH result sets.
        if mode == self::search::HybridMode::Intersection {
            fused_scores.retain(|id, _| lexical_ids.contains(id) && vector_ids.contains(id));
        }

        let mut intermediate: Vec<(u64, f32, Option<crate::data::Document>)> = fused_scores
            .into_iter()
            .map(|(doc_id, (score, document))| (doc_id, score, document))
            .collect();

        // Sort by fused score descending
        intermediate.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Limit results
        if intermediate.len() > limit {
            intermediate.truncate(limit);
        }

        // Batch-resolve external IDs and fill missing documents.
        // Collect IDs that need resolution (either missing external ID or
        // missing document).
        let ids_to_resolve: Vec<u64> = intermediate.iter().map(|(doc_id, _, _)| *doc_id).collect();
        let mut resolved = self.resolve_ids_and_documents_batch(&ids_to_resolve)?;

        let mut results = Vec::with_capacity(intermediate.len());
        for (doc_id, score, document) in intermediate {
            // `remove` moves the id and document out instead of cloning
            // them per hit (#1010); the fused map is keyed by doc id, so
            // each entry is wanted exactly once.
            if let Some((external_id, resolved_doc)) = resolved.remove(&doc_id) {
                // Prefer the document already fetched by the lexical search;
                // fall back to the batch-resolved copy.
                let final_doc = if document.is_some() {
                    document
                } else {
                    resolved_doc
                };
                results.push(SearchResult {
                    id: external_id,
                    score,
                    document: final_doc,
                });
            }
        }

        Ok(results)
    }

    /// Execute multiple independent search requests in parallel.
    ///
    /// Batched form of [`Self::search`] that runs each request
    /// concurrently on the tokio runtime via
    /// [`futures::future::try_join_all`]. Internal vector-search work
    /// additionally parallelises per-request via rayon (Phase 1 of
    /// issue [#648](https://github.com/mosuka/laurus/issues/648), PR
    /// [#711](https://github.com/mosuka/laurus/pull/711)), so a batch
    /// of `B` requests benefits from two-level parallelism: `B`
    /// requests in parallel on tokio, each request's multi-vector
    /// path in parallel on rayon.
    ///
    /// External callers (gRPC service, REST gateway, language
    /// bindings) invoke this method to amortise IPC and serialisation
    /// overhead across multiple queries, in addition to the per-query
    /// parallelism already provided by Phases 1 and 2 of
    /// [#648](https://github.com/mosuka/laurus/issues/648).
    ///
    /// # Parameters
    ///
    /// - `requests` - The list of independent search requests. Order
    ///   is preserved in the output.
    ///
    /// # Returns
    ///
    /// A `Vec<Vec<SearchResult>>` where `results[i]` is the result of
    /// `requests[i]`. Empty input returns an empty `Vec` without
    /// invoking [`Self::search`] at all.
    ///
    /// # Errors
    ///
    /// Short-circuits with the first error encountered; the other
    /// in-flight requests are dropped per
    /// [`futures::future::try_join_all`] semantics.
    ///
    /// Issue [#715](https://github.com/mosuka/laurus/issues/715)
    /// (Phase 3 prerequisite of
    /// [#648](https://github.com/mosuka/laurus/issues/648)).
    pub async fn search_batch(
        &self,
        requests: Vec<self::search::SearchRequest>,
    ) -> Result<Vec<Vec<self::search::SearchResult>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        futures::future::try_join_all(requests.into_iter().map(|r| self.search(r))).await
    }
}

/// Builder for constructing an [`Engine`] with custom configuration.
///
/// Use this when you need to specify a custom text analyzer or embedding
/// model. For simple cases with default settings (StandardAnalyzer, no
/// embedder), use [`Engine::new`] directly.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
///
/// let schema = Schema::builder()
///     .add_field("content", FieldOption::Text(TextOption::default()))
///     .add_field("content_vec", FieldOption::Flat(FlatOption { dimension: 384, ..Default::default() }))
///     .build();
///
/// let engine = Engine::builder(storage, schema)
///     .analyzer(Arc::new(StandardAnalyzer::default()))
///     .embedder(Arc::new(MyEmbedder))
///     .build()
///     .await?;
/// ```
pub struct EngineBuilder {
    storage: Arc<dyn Storage>,
    schema: Schema,
    analyzer: Option<Arc<dyn Analyzer>>,
    embedder: Option<Arc<dyn Embedder>>,
    runtime_analyzers: HashMap<String, Arc<dyn Analyzer>>,
    embedding_cache_capacity: Option<usize>,
    wal_sync_policy: WalSyncPolicy,
    commit_policy: CommitPolicy,
    schema_persist_hook: Option<SchemaPersistHook>,
}

impl EngineBuilder {
    /// Create a new builder with the given storage and schema.
    pub fn new(storage: Arc<dyn Storage>, schema: Schema) -> Self {
        Self {
            storage,
            schema,
            analyzer: None,
            embedder: None,
            runtime_analyzers: HashMap::new(),
            embedding_cache_capacity: None,
            wal_sync_policy: WalSyncPolicy::default(),
            commit_policy: CommitPolicy::default(),
            schema_persist_hook: None,
        }
    }

    /// Register a callback that persists the schema whenever
    /// [`Engine::add_field`] or [`Engine::delete_field`] changes it
    /// (Issue #1078).
    ///
    /// Without this, callers are responsible for persisting the `Schema`
    /// those methods return (e.g. writing `schema.toml`) themselves, and
    /// forgetting to do so silently loses the change on the next restart.
    /// `laurus-cli` and `laurus-server` both set this to write
    /// `<index_dir>/schema.toml`.
    ///
    /// # Arguments
    ///
    /// * `hook` - Called with the updated schema after each successful
    ///   `add_field`/`delete_field`. An error from the hook is propagated to
    ///   the caller of `add_field`/`delete_field`; the in-memory schema has
    ///   already been updated by that point (this does not roll back).
    pub fn persist_schema_with(mut self, hook: SchemaPersistHook) -> Self {
        self.schema_persist_hook = Some(hook);
        self
    }

    /// Set the analyzer for text fields.
    ///
    /// Both simple analyzers (e.g., [`StandardAnalyzer`]) and [`PerFieldAnalyzer`] are
    /// supported. When a `PerFieldAnalyzer` is passed, it is used directly (with `_id`
    /// automatically set to `KeywordAnalyzer` if not already configured).
    ///
    /// If not set, [`StandardAnalyzer`] is used as the default.
    pub fn analyzer(mut self, analyzer: Arc<dyn Analyzer>) -> Self {
        self.analyzer = Some(analyzer);
        self
    }

    /// Register a pre-constructed analyzer under a name, resolved at
    /// build time before built-in names and `schema.analyzers`.
    ///
    /// Useful when an analyzer cannot be expressed as a serializable
    /// [`crate::AnalyzerSpec`] — for example, a Japanese analyzer
    /// constructed from raw dictionary bytes loaded from OPFS in a
    /// browser WASM context. Schema text fields can refer to the
    /// runtime-registered analyzer by its `Named` form.
    ///
    /// # Arguments
    ///
    /// * `name` - The name used in `TextOption.analyzer` (e.g.
    ///   `"ja-ipadic"`).
    /// * `analyzer` - The pre-built analyzer instance.
    pub fn register_runtime_analyzer(
        mut self,
        name: impl Into<String>,
        analyzer: Arc<dyn Analyzer>,
    ) -> Self {
        self.runtime_analyzers.insert(name.into(), analyzer);
        self
    }

    /// Set the embedder for vector fields.
    ///
    /// Both simple embedders and [`PerFieldEmbedder`](crate::embedding::per_field::PerFieldEmbedder)
    /// are supported. When a `PerFieldEmbedder` is passed, each vector field will use
    /// the embedder registered for that field name, falling back to the default.
    ///
    /// If not set, no embedder is configured.
    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Enable an LRU cache for query-time embeddings, holding up to
    /// `capacity` entries (Issue #678).
    ///
    /// When set, identical query payloads embedded by the same field /
    /// embedder are produced only once and reused on subsequent searches,
    /// avoiding repeated model inference (or network round trips for remote
    /// embedders). Disabled by default; `capacity = 0` is treated as
    /// disabled.
    ///
    /// The cache only affects query-time embedding in [`Engine::search`];
    /// document-ingestion embedding is unaffected.
    pub fn embedding_cache_capacity(mut self, capacity: usize) -> Self {
        self.embedding_cache_capacity = Some(capacity);
        self
    }

    /// Set the WAL durability policy (Issue #542).
    ///
    /// Defaults to [`WalSyncPolicy::PerRecord`], where every `add`/`delete`
    /// fsyncs the WAL before returning, so a successful write can never be lost
    /// to a crash. Switch to [`WalSyncPolicy::Group`] to defer and batch the
    /// fsync — much higher ingest throughput at the cost of losing up to the
    /// last unsynced batch on a crash. [`Engine::commit`] is a hard durability
    /// barrier under both policies, and [`Engine::flush_wal`] forces a flush on
    /// demand.
    ///
    /// # Arguments
    ///
    /// * `policy` - The durability policy. Use
    ///   [`WalSyncPolicy::group_with_defaults`] for group commit with the
    ///   default batch thresholds.
    pub fn wal_sync_policy(mut self, policy: WalSyncPolicy) -> Self {
        self.wal_sync_policy = policy;
        self
    }

    /// Set the auto-commit policy (Issue #890).
    ///
    /// Defaults to [`CommitPolicy::Manual`] — the engine commits only when the
    /// caller invokes [`Engine::commit`]. Use [`CommitPolicy::EveryDocs`] to
    /// run the commit ladder automatically every `n` applied documents (across
    /// the singular and batch ingest APIs, and every `n` documents *within* a
    /// batch). Orthogonal to [`Self::wal_sync_policy`]: an auto-commit works
    /// under any WAL sync policy because [`Engine::commit`] always begins with
    /// a WAL flush.
    ///
    /// # Arguments
    ///
    /// * `policy` - The auto-commit policy. `EveryDocs(0)` disables auto-commit
    ///   (equivalent to `Manual`).
    pub fn commit_policy(mut self, policy: CommitPolicy) -> Self {
        self.commit_policy = policy;
        self
    }

    /// Build the [`Engine`].
    ///
    /// Creates the lexical store, vector store, and document log (WAL),
    /// then runs WAL recovery to replay any uncommitted changes from a
    /// previous session.
    ///
    /// # Errors
    ///
    /// Returns an error if storage initialization, index creation, WAL
    /// opening, or recovery replay fails.
    pub async fn build(self) -> Result<Engine> {
        let (lexical_config, vector_config) = Engine::split_schema(
            &self.schema,
            self.analyzer,
            self.embedder,
            &self.runtime_analyzers,
        )
        .await?;

        let lexical_storage = Arc::new(PrefixedStorage::new("lexical", self.storage.clone()));
        let vector_storage = Arc::new(PrefixedStorage::new("vector", self.storage.clone()));
        let document_storage: Arc<dyn Storage> =
            Arc::new(PrefixedStorage::new("documents", self.storage.clone()));

        // The commit-able stores and counter are `Arc` so the background
        // `CommitTimer` (Issue #892) can share them as sub-parts without a
        // reference cycle back to the `Engine`.
        let lexical = Arc::new(LexicalStore::new(lexical_storage, lexical_config)?);
        let vector = Arc::new(VectorStore::new(vector_storage, vector_config)?);
        let docs_since_commit = Arc::new(AtomicU64::new(0));
        let applied_seq = Arc::new(AtomicU64::new(0));

        let log = Arc::new(DocumentLog::with_sync_policy(
            self.storage,
            "engine.wal",
            document_storage,
            self.wal_sync_policy,
        )?);

        let embedding_cache = self
            .embedding_cache_capacity
            .and_then(NonZeroUsize::new)
            .map(|cap| Arc::new(EmbeddingCache::new(cap)));

        // Start the periodic WAL flush timer when the policy is a group commit
        // with an interval. Native only; on wasm32 the interval is ignored.
        #[cfg(not(target_arch = "wasm32"))]
        let wal_flush_timer = match self.wal_sync_policy.flush_interval() {
            Some(interval) => Some(WalFlushTimer::spawn(Arc::clone(&log), interval)?),
            None => None,
        };

        // Start the auto-commit timer for `CommitPolicy::Interval`. Native
        // only; on wasm32 `Interval` is a no-op (no background threads).
        #[cfg(not(target_arch = "wasm32"))]
        let commit_timer = match self.commit_policy {
            CommitPolicy::Interval(interval) => Some(CommitTimer::spawn(
                Arc::clone(&lexical),
                Arc::clone(&vector),
                Arc::clone(&log),
                Arc::clone(&docs_since_commit),
                Arc::clone(&applied_seq),
                interval,
            )?),
            _ => None,
        };

        let engine = Engine {
            schema: RwLock::new(self.schema),
            lexical,
            vector,
            log,
            runtime_analyzers: self.runtime_analyzers,
            embedding_cache,
            commit_policy: self.commit_policy,
            docs_since_commit,
            applied_seq,
            id_locks: Arc::new(
                (0..ID_LOCK_SHARDS)
                    .map(|_| tokio::sync::Mutex::new(()))
                    .collect(),
            ),
            #[cfg(not(target_arch = "wasm32"))]
            _wal_flush_timer: wal_flush_timer,
            #[cfg(not(target_arch = "wasm32"))]
            _commit_timer: commit_timer,
            schema_persist_hook: self.schema_persist_hook,
        };

        engine.recover().await?;

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::per_field::PerFieldEmbedder;
    use crate::embedding::precomputed::PrecomputedEmbedder;
    use crate::storage::memory::MemoryStorage;

    #[tokio::test]
    async fn test_accepts_per_field_analyzer() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::default()));

        let result = Engine::builder(storage, schema)
            .analyzer(Arc::new(per_field))
            .build()
            .await;

        assert!(result.is_ok(), "Should accept PerFieldAnalyzer");
    }

    #[tokio::test]
    async fn test_accepts_per_field_embedder() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let dummy_embedder = Arc::new(PrecomputedEmbedder::new());
        let per_field = PerFieldEmbedder::new(dummy_embedder);

        let result = Engine::builder(storage, schema)
            .embedder(Arc::new(per_field))
            .build()
            .await;

        assert!(result.is_ok(), "Should accept PerFieldEmbedder");
    }

    #[tokio::test]
    async fn test_accepts_simple_analyzer() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let result = Engine::builder(storage, schema)
            .analyzer(Arc::new(StandardAnalyzer::default()))
            .build()
            .await;

        assert!(result.is_ok(), "Should accept StandardAnalyzer");
    }

    #[tokio::test]
    async fn test_accepts_simple_embedder() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let dummy_embedder = Arc::new(PrecomputedEmbedder::new());

        let result = Engine::builder(storage, schema)
            .embedder(dummy_embedder)
            .build()
            .await;

        assert!(result.is_ok(), "Should accept simple embedder");
    }

    #[tokio::test]
    async fn test_schema_per_field_analyzer() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        // "category" uses keyword analyzer (no tokenization).
        // "body" uses default (standard) analyzer.
        let schema = Schema::builder()
            .add_field(
                "category",
                FieldOption::Text(TextOption::default().analyzer("keyword")),
            )
            .add_field("body", FieldOption::Text(TextOption::default()))
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        let mut doc = crate::data::Document::new();
        doc.fields
            .insert("category".into(), DataValue::Text("Rust Lang".into()));
        doc.fields.insert(
            "body".into(),
            DataValue::Text("Rust is a systems programming language".into()),
        );
        engine.put_document("doc1", doc).await.unwrap();
        engine.commit().await.unwrap();

        // "Rust Lang" as keyword — exact match required.
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("category:\"Rust Lang\""))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert_eq!(
            results.len(),
            1,
            "Keyword analyzer should match exact phrase"
        );

        // Partial token "Rust" should NOT match keyword-analyzed category.
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("category:Rust"))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert!(
            results.is_empty(),
            "Keyword analyzer should not match partial tokens"
        );

        // Standard-analyzed "body" field should match single token "rust".
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("body:rust"))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert_eq!(
            results.len(),
            1,
            "Standard analyzer should tokenize and match"
        );
    }

    #[tokio::test]
    async fn test_custom_analyzer_definition_in_schema() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::engine::schema::analyzer::{
            AnalyzerDefinition, CharFilterConfig, TokenFilterConfig, TokenizerConfig,
        };
        use crate::lexical::core::field::TextOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        // Define a custom analyzer: whitespace + NFKC normalization + lowercase.
        let schema = Schema::builder()
            .add_analyzer(
                "my_custom",
                AnalyzerDefinition {
                    char_filters: vec![CharFilterConfig::UnicodeNormalization {
                        form: "nfkc".into(),
                    }],
                    tokenizer: TokenizerConfig::Whitespace,
                    token_filters: vec![TokenFilterConfig::Lowercase],
                },
            )
            .add_field(
                "content",
                FieldOption::Text(TextOption::default().analyzer("my_custom")),
            )
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        let mut doc = crate::data::Document::new();
        // Fullwidth "ＨＥＬＬＯ" should be normalized to "HELLO", then lowercased.
        doc.fields.insert(
            "content".into(),
            DataValue::Text("\u{ff28}\u{ff25}\u{ff2c}\u{ff2c}\u{ff2f} world".into()),
        );
        engine.put_document("doc1", doc).await.unwrap();
        engine.commit().await.unwrap();

        // Search for "hello" should match (NFKC + lowercase).
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("content:hello"))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert_eq!(
            results.len(),
            1,
            "Custom analyzer (NFKC + lowercase) should match normalized text"
        );
    }

    #[tokio::test]
    async fn test_add_lexical_field() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        // Start with a schema containing only "title".
        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        // Dynamically add a "category" field.
        let updated = engine
            .add_field(
                "category",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await
            .unwrap();

        assert!(updated.fields.contains_key("category"));
        assert!(updated.fields.contains_key("title"));

        // Index a document that uses the new field.
        engine
            .add_document(
                "doc1",
                Document::builder()
                    .add_text("title", "Rust Programming")
                    .add_text("category", "programming")
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();

        // Search on the dynamically added field.
        use crate::lexical::search::searcher::LexicalSearchQuery;
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("category:programming"))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert_eq!(
            results.len(),
            1,
            "Should find doc via dynamically added field"
        );
    }

    /// An engine built with [`WalSyncPolicy::Group`] plumbs the policy through to
    /// the WAL, accepts [`Engine::flush_wal`] as an on-demand durability barrier,
    /// and commits searchable results (Issue #542, Phase 4).
    #[tokio::test]
    async fn test_group_commit_policy_is_wired_and_searchable() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let engine = Engine::builder(storage, schema)
            .wal_sync_policy(WalSyncPolicy::group_with_defaults())
            .build()
            .await
            .unwrap();

        engine
            .add_document(
                "doc1",
                Document::builder()
                    .add_text("title", "group commit")
                    .build(),
            )
            .await
            .unwrap();

        // Under the group policy the append defers its fsync; an on-demand
        // flush_wal (no full commit) must succeed as a durability barrier.
        engine.flush_wal().unwrap();

        engine.commit().await.unwrap();

        use crate::lexical::search::searcher::LexicalSearchQuery;
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("title:group"))
            .limit(10)
            .build();
        let results = engine.search(request).await.unwrap();
        assert_eq!(results.len(), 1, "group-committed doc must be searchable");
    }

    /// The background flush timer forces a dirty (deferred) WAL writer durable
    /// within its interval, then stops cleanly on drop (Issue #542, Phase 4b).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn wal_flush_timer_flushes_dirty_writer_and_stops_on_drop() {
        use std::time::Duration;

        let wal_storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let doc_storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        // Thresholds high enough that a single append never trips them, so only
        // the timer can flush the writer.
        let log = Arc::new(
            DocumentLog::with_sync_policy(
                wal_storage,
                "engine.wal",
                doc_storage,
                WalSyncPolicy::Group {
                    max_records: usize::MAX,
                    max_bytes: usize::MAX,
                    max_interval: Some(Duration::from_millis(20)),
                },
            )
            .unwrap(),
        );

        log.append("doc1", Document::builder().add_text("title", "x").build())
            .unwrap();
        assert!(
            log.wal_is_dirty(),
            "the deferred append leaves the writer dirty"
        );

        let timer = WalFlushTimer::spawn(Arc::clone(&log), Duration::from_millis(20)).unwrap();

        // Poll up to ~2s for the timer to flush the writer.
        let mut flushed = false;
        for _ in 0..200 {
            if !log.wal_is_dirty() {
                flushed = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            flushed,
            "the timer should flush the dirty writer within its interval"
        );

        // Dropping the timer must return promptly (clean shutdown / thread join).
        drop(timer);
    }

    /// An engine built with a group-commit policy that includes a flush interval
    /// starts and stops its background timer without hanging on drop (Issue
    /// #542, Phase 4b).
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn test_group_commit_with_interval_builds_and_drops_cleanly() {
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let engine = Engine::builder(storage, schema)
            .wal_sync_policy(WalSyncPolicy::group_with_interval(Duration::from_millis(
                20,
            )))
            .build()
            .await
            .unwrap();

        engine
            .add_document(
                "doc1",
                Document::builder().add_text("title", "timer").build(),
            )
            .await
            .unwrap();

        // Dropping the engine must stop the background timer without hanging.
        drop(engine);
    }

    #[tokio::test]
    async fn test_add_field_duplicate_rejected() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        // Adding a field with the same name should fail.
        let result = engine
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await;
        assert!(result.is_err(), "Duplicate field should be rejected");
    }

    #[tokio::test]
    async fn test_add_vector_field() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let dummy_embedder = Arc::new(PrecomputedEmbedder::new());
        let per_field = PerFieldEmbedder::new(dummy_embedder);

        let engine = Engine::builder(storage, schema)
            .embedder(Arc::new(per_field))
            .build()
            .await
            .unwrap();

        // Dynamically add a vector field with dimension 128 (matching PrecomputedEmbedder default).
        let updated = engine
            .add_field(
                "embedding",
                schema::FieldOption::Flat(
                    crate::vector::core::field::FlatOption::default().dimension(128),
                ),
            )
            .await
            .unwrap();

        assert!(updated.fields.contains_key("embedding"));

        // Index a document with the vector field.
        let vec_data: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        engine
            .add_document(
                "doc1",
                Document::builder()
                    .add_text("title", "Hello")
                    .add_vector("embedding", vec_data)
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();

        // Verify document was indexed successfully.
        let docs = engine.get_documents("doc1").await.unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[tokio::test]
    async fn test_schema_returns_current_state() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let engine = Engine::new(storage, schema).await.unwrap();

        // Initially empty (no user fields).
        assert!(engine.schema().fields.is_empty());

        // Add a field.
        engine
            .add_field(
                "body",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await
            .unwrap();

        // schema() should reflect the addition.
        let current = engine.schema();
        assert!(current.fields.contains_key("body"));
    }

    #[tokio::test]
    async fn test_delete_lexical_field() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        // Dynamically add a "category" field, then delete it.
        engine
            .add_field(
                "category",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await
            .unwrap();
        assert!(engine.schema().fields.contains_key("category"));

        let updated = engine.delete_field("category").await.unwrap();
        assert!(!updated.fields.contains_key("category"));
        assert!(updated.fields.contains_key("title"));
    }

    #[tokio::test]
    async fn test_delete_field_removes_from_default_fields() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .add_default_field("title")
            .build();

        let engine = Engine::new(storage, schema).await.unwrap();

        let updated = engine.delete_field("title").await.unwrap();
        assert!(!updated.fields.contains_key("title"));
        assert!(!updated.default_fields.contains(&"title".to_string()));
    }

    #[tokio::test]
    async fn test_delete_field_nonexistent_rejected() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let engine = Engine::new(storage, schema).await.unwrap();

        let result = engine.delete_field("nonexistent").await;
        assert!(result.is_err(), "Deleting a nonexistent field should fail");
    }

    /// Issue #1078: `add_field`/`delete_field` invoke the configured
    /// schema-persist hook with the up-to-date schema, and propagate its
    /// error (rather than silently swallowing it) when it fails.
    #[tokio::test]
    async fn test_add_field_invokes_schema_persist_hook() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let persisted: Arc<parking_lot::Mutex<Vec<Schema>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));
        let persisted_clone = Arc::clone(&persisted);

        let engine = Engine::builder(storage, schema)
            .persist_schema_with(Arc::new(move |schema: &Schema| {
                persisted_clone.lock().push(schema.clone());
                Ok(())
            }))
            .build()
            .await
            .unwrap();

        engine
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await
            .unwrap();

        let calls = persisted.lock();
        assert_eq!(calls.len(), 1, "hook should be called exactly once");
        assert!(calls[0].fields.contains_key("title"));
    }

    #[tokio::test]
    async fn test_delete_field_invokes_schema_persist_hook() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let persisted: Arc<parking_lot::Mutex<Vec<Schema>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));
        let persisted_clone = Arc::clone(&persisted);

        let engine = Engine::builder(storage, schema)
            .persist_schema_with(Arc::new(move |schema: &Schema| {
                persisted_clone.lock().push(schema.clone());
                Ok(())
            }))
            .build()
            .await
            .unwrap();

        engine.delete_field("title").await.unwrap();

        let calls = persisted.lock();
        assert_eq!(calls.len(), 1, "hook should be called exactly once");
        assert!(!calls[0].fields.contains_key("title"));
    }

    #[tokio::test]
    async fn test_add_field_without_hook_does_not_persist() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        // No `persist_schema_with` configured: this is the pre-#1078
        // behavior, preserved for callers that persist the schema
        // themselves.
        let engine = Engine::new(storage, schema).await.unwrap();

        let updated = engine
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await
            .unwrap();
        assert!(updated.fields.contains_key("title"));
    }

    #[tokio::test]
    async fn test_add_field_propagates_schema_persist_hook_error() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::new();

        let engine = Engine::builder(storage, schema)
            .persist_schema_with(Arc::new(|_schema: &Schema| {
                Err(crate::error::LaurusError::invalid_argument(
                    "simulated persistence failure",
                ))
            }))
            .build()
            .await
            .unwrap();

        let result = engine
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .await;
        assert!(
            result.is_err(),
            "a failing schema-persist hook must fail add_field"
        );

        // The in-memory schema was already updated before the hook ran
        // (this phase does not add rollback — see #1077).
        assert!(engine.schema().fields.contains_key("title"));
    }

    #[tokio::test]
    async fn test_delete_vector_field() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));

        let schema = Schema::builder()
            .add_field(
                "title",
                schema::FieldOption::Text(crate::lexical::core::field::TextOption::default()),
            )
            .build();

        let dummy_embedder = Arc::new(PrecomputedEmbedder::new());
        let per_field = PerFieldEmbedder::new(dummy_embedder);

        let engine = Engine::builder(storage, schema)
            .embedder(Arc::new(per_field))
            .build()
            .await
            .unwrap();

        // Add then delete a vector field.
        engine
            .add_field(
                "embedding",
                schema::FieldOption::Hnsw(crate::vector::core::field::HnswOption {
                    dimension: 4,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert!(engine.schema().fields.contains_key("embedding"));

        let updated = engine.delete_field("embedding").await.unwrap();
        assert!(!updated.fields.contains_key("embedding"));
    }

    /// Regression test for the InvertedIndexWriter `delete_document` bug
    /// where the in-memory inverted index and DocValues were not rebuilt
    /// after a buffered doc was retained out, leaving ghost postings that
    /// survived into the next flushed segment.
    ///
    /// Symptom in callers: `put_document(id, doc1)` then
    /// `put_document(id, doc2)` in the same uncommitted batch ended up
    /// with two live docs sharing the same external `_id` after commit,
    /// and `get_documents(id)` / `engine.search(`_id:id`)` returned both.
    ///
    /// Fix lives in `lexical/index/inverted/writer.rs::delete_document`:
    /// it now calls `remove_pending_document` (which rebuilds the
    /// in-memory inverted index and DocValues) instead of doing a bare
    /// `buffered_docs.retain`.
    #[tokio::test]
    async fn test_put_document_replaces_within_uncommitted_batch() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        let mut doc1 = crate::data::Document::new();
        doc1.fields
            .insert("title".into(), DataValue::Text("first".into()));
        engine.put_document("X", doc1).await.unwrap();

        // Second put for the same external id, BEFORE commit. The first
        // doc must be fully replaced — not appended.
        let mut doc2 = crate::data::Document::new();
        doc2.fields
            .insert("title".into(), DataValue::Text("second".into()));
        engine.put_document("X", doc2).await.unwrap();

        engine.commit().await.unwrap();

        let docs = engine.get_documents("X").await.unwrap();
        assert_eq!(
            docs.len(),
            1,
            "exactly one doc should exist for id=X after two puts in the \
             same uncommitted batch (got {} docs: {:?})",
            docs.len(),
            docs.iter()
                .map(|d| d.fields.get("title").cloned())
                .collect::<Vec<_>>(),
        );

        let title = docs[0]
            .fields
            .get("title")
            .and_then(|v| v.as_text())
            .map(String::from);
        assert_eq!(
            title.as_deref(),
            Some("second"),
            "the surviving doc must be the latest put"
        );

        let stats = engine.stats().unwrap();
        assert_eq!(
            stats.document_count, 1,
            "engine.stats().document_count must agree with get_documents",
        );
    }

    /// Regression test for the same bug under a heavier put-pattern that
    /// mirrors the `laurus-wasm/examples/geo3d/` workload: put many docs,
    /// many of them carrying the same external id, in a single
    /// uncommitted batch.
    ///
    /// Before the fix, the engine reported `document_count` equal to the
    /// raw put count (with duplicates) instead of the unique-id count.
    #[tokio::test]
    async fn test_put_document_dedupes_duplicate_ids_in_batch() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // 10 unique ids, each put 3 times in a row before commit.
        for i in 0..10 {
            for rev in 0..3 {
                let mut doc = crate::data::Document::new();
                doc.fields
                    .insert("title".into(), DataValue::Text(format!("id{i}-rev{rev}")));
                engine.put_document(&format!("id{i}"), doc).await.unwrap();
            }
        }

        engine.commit().await.unwrap();

        let stats = engine.stats().unwrap();
        assert_eq!(
            stats.document_count, 10,
            "exactly 10 unique docs should be live; the 20 redundant puts \
             must have been replaced, not accumulated"
        );

        // Each id should resolve to exactly one doc — the last put wins.
        for i in 0..10 {
            let docs = engine.get_documents(&format!("id{i}")).await.unwrap();
            assert_eq!(docs.len(), 1, "id{i} should resolve to a single doc");
            let title = docs[0]
                .fields
                .get("title")
                .and_then(|v| v.as_text())
                .map(String::from);
            assert_eq!(
                title.as_deref(),
                Some(format!("id{i}-rev2").as_str()),
                "id{i} should retain the last put's title"
            );
        }
    }

    /// Build a `(id, doc)` batch entry with a single `title` text field, for
    /// the `put_documents` / `add_documents` tests below.
    fn batch_entry(id: &str, title: &str) -> (String, crate::data::Document) {
        use crate::data::DataValue;
        let mut doc = crate::data::Document::new();
        doc.fields
            .insert("title".into(), DataValue::Text(title.into()));
        (id.to_string(), doc)
    }

    /// Build an engine over `MemoryStorage` with a single `title` text field,
    /// for the `put_documents` / `add_documents` tests below.
    async fn title_only_engine() -> Engine {
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();
        Engine::new(storage, schema).await.unwrap()
    }

    /// #551: an empty batch is a no-op — no WAL activity, no error.
    #[tokio::test]
    async fn test_put_documents_empty_batch_is_noop() {
        let engine = title_only_engine().await;
        engine.put_documents(Vec::new()).await.unwrap();
        engine.add_documents(Vec::new()).await.unwrap();
        assert!(
            !engine.log.wal_is_dirty(),
            "an empty batch must not leave unsynced WAL bytes"
        );
        assert_eq!(engine.stats().unwrap().document_count, 0);
    }

    /// #551: `put_documents` must produce exactly the same final state as the
    /// equivalent sequence of singular `put_document` calls.
    #[tokio::test]
    async fn test_put_documents_matches_sequential_puts() {
        let batch_engine = title_only_engine().await;
        let sequential_engine = title_only_engine().await;

        let docs: Vec<_> = (0..20)
            .map(|i| batch_entry(&format!("id{i}"), &format!("title-{i}")))
            .collect();

        for (id, doc) in docs.clone() {
            sequential_engine.put_document(&id, doc).await.unwrap();
        }
        batch_engine.put_documents(docs).await.unwrap();

        batch_engine.commit().await.unwrap();
        sequential_engine.commit().await.unwrap();

        assert_eq!(
            batch_engine.stats().unwrap().document_count,
            sequential_engine.stats().unwrap().document_count,
        );
        for i in 0..20 {
            let batch_docs = batch_engine.get_documents(&format!("id{i}")).await.unwrap();
            let seq_docs = sequential_engine
                .get_documents(&format!("id{i}"))
                .await
                .unwrap();
            assert_eq!(batch_docs.len(), 1, "id{i} must resolve to one doc");
            assert_eq!(
                batch_docs[0].fields.get("title"),
                seq_docs[0].fields.get("title"),
                "id{i} must carry the same title on both paths"
            );
        }
    }

    /// #551: duplicate external ids **within one batch** must dedup exactly
    /// like the same puts issued sequentially (last occurrence wins) — the
    /// batch mirror of `test_put_document_dedupes_duplicate_ids_in_batch`.
    #[tokio::test]
    async fn test_put_documents_dedupes_duplicate_ids_in_batch() {
        let engine = title_only_engine().await;

        // 10 unique ids × 3 revisions interleaved in a single batch call.
        let mut docs = Vec::new();
        for rev in 0..3 {
            for i in 0..10 {
                docs.push(batch_entry(&format!("id{i}"), &format!("id{i}-rev{rev}")));
            }
        }
        engine.put_documents(docs).await.unwrap();
        engine.commit().await.unwrap();

        assert_eq!(
            engine.stats().unwrap().document_count,
            10,
            "exactly 10 unique docs should be live after in-batch dedup"
        );
        for i in 0..10 {
            let docs = engine.get_documents(&format!("id{i}")).await.unwrap();
            assert_eq!(docs.len(), 1, "id{i} should resolve to a single doc");
            let title = docs[0]
                .fields
                .get("title")
                .and_then(|v| v.as_text())
                .map(String::from);
            assert_eq!(
                title.as_deref(),
                Some(format!("id{i}-rev2").as_str()),
                "id{i} must retain the last occurrence in the batch"
            );
        }
    }

    /// #551: `add_documents` never delete-firsts, so repeating an id within
    /// one batch legitimately accumulates chunks.
    #[tokio::test]
    async fn test_add_documents_same_id_creates_chunks() {
        let engine = title_only_engine().await;

        let docs: Vec<_> = (0..4)
            .map(|i| batch_entry("doc", &format!("chunk-{i}")))
            .collect();
        engine.add_documents(docs).await.unwrap();
        engine.commit().await.unwrap();

        let chunks = engine.get_documents("doc").await.unwrap();
        assert_eq!(
            chunks.len(),
            4,
            "all four chunks sharing the external id must be live"
        );
    }

    /// #551: fail-fast semantics — the batch stops at the first failing doc,
    /// reports its position/id and the applied count, and the applied prefix
    /// stays NRT-visible (no rollback).
    #[tokio::test]
    async fn test_put_documents_failfast_reports_index_and_applied() {
        use crate::engine::schema::{DynamicFieldPolicy, FieldOption};
        use crate::lexical::core::field::TextOption;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .dynamic_field_policy(DynamicFieldPolicy::Strict)
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // Docs 0-2 are fine; doc 3 carries an undeclared field, which the
        // Strict policy rejects in apply_dynamic_schema (before any WAL /
        // store mutation for that doc).
        let mut docs: Vec<_> = (0..3)
            .map(|i| batch_entry(&format!("ok{i}"), &format!("title-{i}")))
            .collect();
        let mut bad = crate::data::Document::new();
        bad.fields.insert(
            "undeclared".into(),
            crate::data::DataValue::Text("boom".into()),
        );
        docs.push(("bad".to_string(), bad));
        docs.push(batch_entry("never", "never-applied"));

        let err = engine
            .put_documents(docs)
            .await
            .expect_err("Strict policy must fail the batch at doc 3");
        match err {
            crate::error::LaurusError::BatchIngest {
                failed_index,
                failed_id,
                applied,
                ..
            } => {
                assert_eq!(failed_index, 3);
                assert_eq!(failed_id, "bad");
                assert_eq!(applied, 3);
            }
            other => panic!("expected BatchIngest, got: {other}"),
        }
        assert!(
            !engine.log.wal_is_dirty(),
            "the applied prefix must be flushed durable on the error path"
        );

        // The applied prefix must be live (NRT) — doc 4 must not have been
        // attempted.
        for i in 0..3 {
            let docs = engine.get_documents(&format!("ok{i}")).await.unwrap();
            assert_eq!(docs.len(), 1, "applied doc ok{i} must stay visible");
        }
        assert!(
            engine.get_documents("never").await.unwrap().is_empty(),
            "docs after the failing one must not be applied"
        );
    }

    /// #551: under the default `PerRecord` policy a batch of N docs must fsync
    /// the WAL exactly once (deferred-fsync scope), not once per record, and
    /// leave nothing unsynced.
    #[tokio::test]
    async fn test_put_documents_single_wal_fsync_per_batch() {
        let engine = title_only_engine().await;

        // Prime the WAL writer so the sync counter exists, then baseline it.
        engine
            .put_document("prime", batch_entry("prime", "prime").1)
            .await
            .unwrap();
        let baseline = engine.log.wal_sync_count();

        let docs: Vec<_> = (0..50)
            .map(|i| batch_entry(&format!("id{i}"), &format!("title-{i}")))
            .collect();
        engine.put_documents(docs).await.unwrap();

        assert_eq!(
            engine.log.wal_sync_count(),
            baseline + 1,
            "a 50-doc batch must amortize to exactly one WAL fsync"
        );
        assert!(
            !engine.log.wal_is_dirty(),
            "the batch-end flush must leave no unsynced WAL bytes"
        );
    }

    /// #551: singular writes acknowledged while a **concurrent batch** holds
    /// the WAL sync-deferral scope must keep their per-record durability —
    /// the deferral flag is global, so without the explicit re-assertion in
    /// `put_document` / `add_document` / `delete_documents` their records
    /// would be left unsynced until the batch ends.
    #[tokio::test]
    async fn test_singular_writes_stay_durable_during_concurrent_batch_deferral() {
        let engine = title_only_engine().await;
        engine
            .put_document("seed", batch_entry("seed", "seed").1)
            .await
            .unwrap();

        // Simulate an in-flight put_documents batch on another task.
        let _foreign_batch = engine.log.defer_sync();

        engine
            .put_document("a", batch_entry("a", "a").1)
            .await
            .unwrap();
        assert!(
            !engine.log.wal_is_dirty(),
            "a singular put must be fsync'd before ack even during a batch"
        );

        engine
            .add_document("b", batch_entry("b", "b").1)
            .await
            .unwrap();
        assert!(
            !engine.log.wal_is_dirty(),
            "a singular add must be fsync'd before ack even during a batch"
        );

        engine.delete_documents("seed").await.unwrap();
        assert!(
            !engine.log.wal_is_dirty(),
            "a singular delete must be fsync'd before ack even during a batch"
        );
    }

    /// #828: updating many docs that are still in the uncommitted buffer must
    /// stay correct under the deferred in-memory-index rebuild. Each update goes
    /// through `delete_documents` → `delete_document(old_buffered_id)` →
    /// `remove_pending_document`, which now defers the rebuild to flush. After
    /// commit, every external id must resolve to exactly its latest version and
    /// the live doc count must equal the number of unique ids (no ghosts from
    /// the superseded buffered versions).
    #[tokio::test]
    async fn test_put_document_update_many_before_commit() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        let n = 200usize;

        // Phase 1: add N distinct docs into one uncommitted buffer.
        for i in 0..n {
            let mut doc = crate::data::Document::new();
            doc.fields
                .insert("title".into(), DataValue::Text(format!("v0-{i}")));
            engine.put_document(&format!("id{i}"), doc).await.unwrap();
        }

        // Phase 2: update every one of them BEFORE committing. Each update hits
        // the deferred-rebuild path (the old version is still buffered).
        for i in 0..n {
            let mut doc = crate::data::Document::new();
            doc.fields
                .insert("title".into(), DataValue::Text(format!("v1-{i}")));
            engine.put_document(&format!("id{i}"), doc).await.unwrap();
        }

        engine.commit().await.unwrap();

        let stats = engine.stats().unwrap();
        assert_eq!(
            stats.document_count, n as u64,
            "every external id must collapse to exactly one live doc after the \
             pre-commit updates (no ghost versions from the deferred rebuild)"
        );

        // Each id resolves to exactly one doc carrying the updated content.
        for i in [0usize, 1, n / 2, n - 1] {
            let docs = engine.get_documents(&format!("id{i}")).await.unwrap();
            assert_eq!(docs.len(), 1, "id{i} must resolve to a single doc");
            let title = docs[0]
                .fields
                .get("title")
                .and_then(|v| v.as_text())
                .map(String::from);
            assert_eq!(
                title.as_deref(),
                Some(format!("v1-{i}").as_str()),
                "id{i} must carry the updated (v1) content after commit"
            );
        }
    }

    /// Regression test for the geo3d demo's "departure + re-arrival"
    /// pattern: put → commit → delete → commit → put-with-same-id →
    /// commit. The post-commit search must return exactly one doc and
    /// `engine.stats().document_count` must agree.
    ///
    /// This exercises the path where the previous version of the
    /// document is in a *committed* segment (not the writer buffer)
    /// when the next put runs `delete_documents` internally — which
    /// is what happens when an aircraft drops out of an
    /// `airplanes.live` snapshot and re-enters on a later refresh.
    #[tokio::test]
    async fn test_put_document_replaces_after_delete_across_commits() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // Round 1: put + commit so the doc lives in a flushed segment.
        let mut doc1 = crate::data::Document::new();
        doc1.fields
            .insert("title".into(), DataValue::Text("first".into()));
        engine.put_document("X", doc1).await.unwrap();
        engine.commit().await.unwrap();

        // Round 2: delete + commit so the previous version is
        // soft-deleted in the segment.
        engine.delete_documents("X").await.unwrap();
        engine.commit().await.unwrap();
        assert!(
            engine.get_documents("X").await.unwrap().is_empty(),
            "after delete + commit, get_documents must return empty"
        );

        // Round 3: re-arrival with the same external id. The new put
        // must produce a single live doc; the soft-deleted segment
        // version must not resurface.
        let mut doc2 = crate::data::Document::new();
        doc2.fields
            .insert("title".into(), DataValue::Text("second".into()));
        engine.put_document("X", doc2).await.unwrap();
        engine.commit().await.unwrap();

        let docs = engine.get_documents("X").await.unwrap();
        assert_eq!(
            docs.len(),
            1,
            "exactly one doc should exist for id=X after departure + re-arrival"
        );
        assert_eq!(
            docs[0]
                .fields
                .get("title")
                .and_then(|v| v.as_text())
                .map(String::from)
                .as_deref(),
            Some("second"),
            "the surviving doc must be the latest put"
        );

        let stats = engine.stats().unwrap();
        assert_eq!(
            stats.document_count, 1,
            "engine.stats().document_count must agree with get_documents \
             across the departure + re-arrival cycle"
        );
    }

    /// Regression test for the geo3d-side stale-id bug: spatial queries
    /// (BKD-backed: Geo / Geo3d) used to return soft-deleted docs
    /// because the BKD tree itself does not consult the segment
    /// deletion bitmap. The fix lives in the
    /// `lexical/query/geo3d.rs::*::find_matches` helpers (and the 2D
    /// counterparts in `lexical/query/geo.rs`), which now skip
    /// `reader.is_deleted(doc_id)` hits.
    ///
    /// Steps:
    ///   1. Put a doc with a geo3d position, commit.
    ///   2. Run `geo3d_bbox(...)` over a region containing the point —
    ///      should find 1 hit (sanity check).
    ///   3. Delete the doc, commit. The BKD tree still contains the
    ///      point until a merge, but the deletion bitmap is set.
    ///   4. Run the same `geo3d_bbox(...)` query — must find 0 hits.
    #[tokio::test]
    async fn test_geo3d_query_filters_soft_deleted_docs() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::Geo3dOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("position", FieldOption::Geo3d(Geo3dOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // Tokyo Tower in ECEF (approx).
        let mut doc = crate::data::Document::new();
        doc.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3955182.0, 3350553.0, 3700276.0,
            )),
        );
        engine.put_document("FW52", doc).await.unwrap();
        engine.commit().await.unwrap();

        // Sanity check: a wide bbox around the point matches it.
        let bbox_dsl = "position:geo3d_bbox(-3956000.0, 3349000.0, 3699000.0, \
                       -3954000.0, 3352000.0, 3702000.0)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(bbox_dsl))
            .limit(10)
            .build();
        let before = engine.search(request).await.unwrap();
        assert_eq!(before.len(), 1, "live doc should match the bbox");

        // Delete + commit. The doc is soft-deleted in the segment but
        // its BKD entry stays in place until merge.
        engine.delete_documents("FW52").await.unwrap();
        engine.commit().await.unwrap();

        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(bbox_dsl))
            .limit(10)
            .build();
        let after = engine.search(request).await.unwrap();
        assert_eq!(
            after.len(),
            0,
            "soft-deleted doc must NOT be returned by geo3d_bbox \
             (BKD entry survives in-tree until merge)",
        );

        // Same expectation for geo3d_nearest.
        let nearest_dsl = "position:geo3d_nearest(-3955182.0, 3350553.0, 3700276.0, 5)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(nearest_dsl))
            .limit(10)
            .build();
        let after_nearest = engine.search(request).await.unwrap();
        assert_eq!(
            after_nearest.len(),
            0,
            "soft-deleted doc must NOT be returned by geo3d_nearest",
        );

        // And for geo3d_distance.
        let distance_dsl = "position:geo3d_distance(-3955182.0, 3350553.0, 3700276.0, 100000.0)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(distance_dsl))
            .limit(10)
            .build();
        let after_distance = engine.search(request).await.unwrap();
        assert_eq!(
            after_distance.len(),
            0,
            "soft-deleted doc must NOT be returned by geo3d_distance",
        );
    }

    /// Regression test for #480 (`e7c206ad`): the per-segment fanout
    /// path in [`InvertedIndexSearcher::search_with_collector_parallel`]
    /// wraps each segment in a [`PerSegmentReaderView`] that did not
    /// override `get_bkd_tree`, so the trait default (`Ok(None)`)
    /// silently disabled every BKD-backed query (geo / geo3d / numeric
    /// range) once an index accumulated two or more segments. Reported
    /// in production via the `laurus-wasm/examples/geo3d` demo, where
    /// the second auto-refresh commit added a second segment and every
    /// subsequent `geo3d_nearest` returned 0 hits.
    ///
    /// Steps:
    ///   1. Put one doc, commit — segment 0.
    ///   2. Put another doc, commit — segment 1. Reader now has
    ///      `segment_count() == 2`, the fanout condition triggers.
    ///   3. Run `geo3d_distance(...)` over a sphere covering both
    ///      points — must find 2 hits, NOT 0.
    #[tokio::test]
    async fn test_geo3d_distance_multi_segment_returns_hits() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::Geo3dOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("position", FieldOption::Geo3d(Geo3dOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // Two distinct ECEF points roughly 50 km apart near Tokyo.
        let mut doc_a = crate::data::Document::new();
        doc_a.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3955182.0, 3350553.0, 3700276.0,
            )),
        );
        engine.put_document("A", doc_a).await.unwrap();
        engine.commit().await.unwrap();

        let mut doc_b = crate::data::Document::new();
        doc_b.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3960000.0, 3350000.0, 3700000.0,
            )),
        );
        engine.put_document("B", doc_b).await.unwrap();
        engine.commit().await.unwrap();

        // 100 km sphere covers both points. With the bug, fanout makes
        // this return 0 because PerSegmentReaderView.get_bkd_tree falls
        // through to the trait default `Ok(None)`.
        let dsl = "position:geo3d_distance(-3957000.0, 3350000.0, 3700000.0, 100000.0)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(dsl))
            .limit(10)
            .build();
        let hits = engine.search(request).await.unwrap();
        assert_eq!(
            hits.len(),
            2,
            "geo3d_distance must find both docs across two segments; \
             got {} (bug: per-segment fanout drops BKD-backed queries)",
            hits.len()
        );
    }

    /// Regression test for #480: same as
    /// [`test_geo3d_distance_multi_segment_returns_hits`] but for
    /// `geo3d_nearest`, which is the query path the demo exercises.
    #[tokio::test]
    async fn test_geo3d_nearest_multi_segment_returns_hits() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::Geo3dOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("position", FieldOption::Geo3d(Geo3dOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        let mut doc_a = crate::data::Document::new();
        doc_a.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3955182.0, 3350553.0, 3700276.0,
            )),
        );
        engine.put_document("A", doc_a).await.unwrap();
        engine.commit().await.unwrap();

        let mut doc_b = crate::data::Document::new();
        doc_b.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3960000.0, 3350000.0, 3700000.0,
            )),
        );
        engine.put_document("B", doc_b).await.unwrap();
        engine.commit().await.unwrap();

        let dsl = "position:geo3d_nearest(-3957000.0, 3350000.0, 3700000.0, 5)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(dsl))
            .limit(10)
            .build();
        let hits = engine.search(request).await.unwrap();
        assert_eq!(
            hits.len(),
            2,
            "geo3d_nearest must find both docs across two segments; \
             got {} (bug: per-segment fanout returns no BKD tree)",
            hits.len()
        );
    }

    /// Regression test for #480 on numeric range queries. Same
    /// underlying cause — BKD-backed query through the fanout view.
    #[tokio::test]
    async fn test_numeric_range_multi_segment_returns_hits() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::IntegerOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("score", FieldOption::Integer(IntegerOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        let mut doc_a = crate::data::Document::new();
        doc_a.fields.insert("score".into(), DataValue::Int64(10));
        engine.put_document("A", doc_a).await.unwrap();
        engine.commit().await.unwrap();

        let mut doc_b = crate::data::Document::new();
        doc_b.fields.insert("score".into(), DataValue::Int64(20));
        engine.put_document("B", doc_b).await.unwrap();
        engine.commit().await.unwrap();

        let dsl = "score:[5 TO 25]";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(dsl))
            .limit(10)
            .build();
        let hits = engine.search(request).await.unwrap();
        assert_eq!(
            hits.len(),
            2,
            "numeric range query must find both docs across two \
             segments; got {} (bug: per-segment fanout drops BKD tree)",
            hits.len()
        );
    }

    /// Build an engine with `body` (text) + `popularity` (integer) and
    /// four docs whose popularity order differs from doc order (#942).
    async fn options_test_engine() -> Engine {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::{IntegerOption, TextOption};

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("body", FieldOption::Text(TextOption::default()))
            .add_field("popularity", FieldOption::Integer(IntegerOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        for (id, popularity) in [("A", 30i64), ("B", 10), ("C", 40), ("D", 20)] {
            let mut doc = crate::data::Document::new();
            doc.fields
                .insert("body".into(), DataValue::Text("alpha".into()));
            doc.fields
                .insert("popularity".into(), DataValue::Int64(popularity));
            engine.put_document(id, doc).await.unwrap();
        }
        engine.commit().await.unwrap();
        engine
    }

    /// #942 regression: `SearchRequestBuilder::sort_by` must take effect
    /// through `Engine::search` — it used to be a silent no-op because
    /// the execution path rebuilt the lexical request without params.
    #[tokio::test]
    async fn test_engine_search_honors_sort_by() {
        use crate::lexical::search::searcher::{LexicalSearchQuery, SortField, SortOrder};

        let engine = options_test_engine().await;

        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("body:alpha"))
            .sort_by(SortField::Field {
                name: "popularity".into(),
                order: SortOrder::Desc,
            })
            .limit(4)
            .build();
        let hits = engine.search(request).await.unwrap();

        let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["C", "A", "D", "B"],
            "results must be ordered by popularity desc, not by score"
        );
    }

    /// #942 regression: `lexical_min_score` must take effect through
    /// `Engine::search`.
    #[tokio::test]
    async fn test_engine_search_honors_min_score() {
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let engine = options_test_engine().await;

        let baseline = engine
            .search(
                crate::engine::search::SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::from("body:alpha"))
                    .limit(4)
                    .build(),
            )
            .await
            .unwrap();
        assert_eq!(baseline.len(), 4);
        let max_score = baseline
            .iter()
            .map(|h| h.score)
            .fold(f32::NEG_INFINITY, f32::max);

        // A threshold above every real score must exclude everything.
        let filtered = engine
            .search(
                crate::engine::search::SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::from("body:alpha"))
                    .lexical_min_score(max_score + 1.0)
                    .limit(4)
                    .build(),
            )
            .await
            .unwrap();
        assert!(
            filtered.is_empty(),
            "min_score above every score must exclude all hits; got {}",
            filtered.len()
        );
    }

    /// #942: `lexical_parallel` must be honored and produce the same
    /// result set as the serial path (wiring equality gate).
    #[tokio::test]
    async fn test_engine_search_parallel_matches_serial() {
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let engine = options_test_engine().await;

        let mut serial_ids: Vec<String> = engine
            .search(
                crate::engine::search::SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::from("body:alpha"))
                    .lexical_parallel(false)
                    .limit(4)
                    .build(),
            )
            .await
            .unwrap()
            .iter()
            .map(|h| h.id.clone())
            .collect();
        let mut parallel_ids: Vec<String> = engine
            .search(
                crate::engine::search::SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::from("body:alpha"))
                    .lexical_parallel(true)
                    .limit(4)
                    .build(),
            )
            .await
            .unwrap()
            .iter()
            .map(|h| h.id.clone())
            .collect();
        serial_ids.sort();
        parallel_ids.sort();
        assert_eq!(serial_ids, parallel_ids);
    }

    /// Combined regression test: per-segment fanout must restore BKD
    /// query hits (#480 fix) AND continue to filter out soft-deleted
    /// hits within each segment (#400 fix). Without per-segment
    /// deletion filtering in `PerSegmentReaderView::get_bkd_tree`,
    /// the #480 fix would re-introduce the #400 ghost-hit regression.
    #[tokio::test]
    async fn test_geo3d_distance_multi_segment_filters_deleted() {
        use crate::data::DataValue;
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::Geo3dOption;
        use crate::lexical::search::searcher::LexicalSearchQuery;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("position", FieldOption::Geo3d(Geo3dOption::default()))
            .build();
        let engine = Engine::new(storage, schema).await.unwrap();

        // Segment 0: two docs near Tokyo.
        for id in &["A", "B"] {
            let mut doc = crate::data::Document::new();
            doc.fields.insert(
                "position".into(),
                DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                    -3955182.0, 3350553.0, 3700276.0,
                )),
            );
            engine.put_document(id, doc).await.unwrap();
        }
        engine.commit().await.unwrap();

        // Soft-delete A in segment 0 and commit so segment 0 carries
        // a deletion bitmap when the fanout view consults it.
        engine.delete_documents("A").await.unwrap();
        engine.commit().await.unwrap();

        // Segment 1: one new doc.
        let mut doc_c = crate::data::Document::new();
        doc_c.fields.insert(
            "position".into(),
            DataValue::GeoEcef(crate::data::GeoEcefPoint::new(
                -3960000.0, 3350000.0, 3700000.0,
            )),
        );
        engine.put_document("C", doc_c).await.unwrap();
        engine.commit().await.unwrap();

        let dsl = "position:geo3d_distance(-3957000.0, 3350000.0, 3700000.0, 100000.0)";
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from(dsl))
            .limit(10)
            .build();
        let hits = engine.search(request).await.unwrap();
        assert_eq!(
            hits.len(),
            2,
            "must return live docs B and C across two segments; \
             got {} (regression: either #480 fanout BKD or #400 \
             per-segment deletion filter)",
            hits.len()
        );
    }

    // ---- #890 CommitPolicy (auto-commit) gates ----

    /// Build a `title`-only engine with the given auto-commit policy, returning
    /// the engine plus a handle to the root storage so tests can observe the
    /// WAL file directly. The WAL lives at `engine.wal` on the root storage;
    /// the commit ladder truncates it to zero bytes, so [`wal_bytes`] is a
    /// side-effect-free "are there uncommitted records?" probe.
    async fn commit_policy_engine(policy: CommitPolicy) -> (Engine, Arc<dyn Storage>) {
        use crate::engine::schema::FieldOption;
        use crate::lexical::core::field::TextOption;
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let schema = Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .build();
        let engine = Engine::builder(storage.clone(), schema)
            .commit_policy(policy)
            .build()
            .await
            .unwrap();
        (engine, storage)
    }

    /// Bytes in the WAL file: `0` right after a commit (the ladder truncates
    /// the WAL) or before the first append; `> 0` while uncommitted records are
    /// buffered.
    fn wal_bytes(storage: &Arc<dyn Storage>) -> u64 {
        storage.file_size("engine.wal").unwrap_or(0)
    }

    fn title_doc(title: &str) -> Document {
        Document::builder().add_text("title", title).build()
    }

    /// #890: `EveryDocs(n)` auto-commits after exactly every n-th applied
    /// document — the WAL is truncated right after each n-th put and non-empty
    /// in between (a deterministic commit-boundary counter).
    #[tokio::test]
    async fn every_docs_commits_at_exact_multiples() {
        let (engine, storage) = commit_policy_engine(CommitPolicy::EveryDocs(3)).await;
        for i in 1..=7u64 {
            engine
                .put_document(&format!("d{i}"), title_doc("x"))
                .await
                .unwrap();
            if i % 3 == 0 {
                assert_eq!(
                    wal_bytes(&storage),
                    0,
                    "an auto-commit must truncate the WAL at doc {i}"
                );
            } else {
                assert!(
                    wal_bytes(&storage) > 0,
                    "docs between commit points stay uncommitted (doc {i})"
                );
            }
        }
    }

    /// #890: within a batch the auto-commit fires every n documents *inside*
    /// the call (chunked), not once at the end. `EveryDocs(3)` over 7 docs
    /// commits at docs 3 and 6, leaving exactly the 1-document remainder
    /// uncommitted — a single end-of-batch commit would empty the WAL entirely.
    #[tokio::test]
    async fn every_docs_commits_within_batch() {
        let (engine, storage) = commit_policy_engine(CommitPolicy::EveryDocs(3)).await;
        let docs: Vec<_> = (1..=7u64)
            .map(|i| batch_entry(&format!("d{i}"), "x"))
            .collect();
        engine.put_documents(docs).await.unwrap();
        assert!(
            wal_bytes(&storage) > 0,
            "chunked auto-commit must leave the 1-doc remainder uncommitted; \
             an empty WAL would mean a single end-of-batch commit instead"
        );

        // A clean multiple leaves nothing uncommitted (both chunks committed).
        let (engine2, storage2) = commit_policy_engine(CommitPolicy::EveryDocs(3)).await;
        let docs2: Vec<_> = (1..=6u64)
            .map(|i| batch_entry(&format!("d{i}"), "x"))
            .collect();
        engine2.put_documents(docs2).await.unwrap();
        assert_eq!(
            wal_bytes(&storage2),
            0,
            "6 docs at EveryDocs(3) commits both chunks, leaving an empty WAL"
        );
    }

    /// #890: `EveryDocs(0)` disables auto-commit (equivalent to `Manual`).
    #[tokio::test]
    async fn every_docs_zero_is_manual() {
        let (engine, storage) = commit_policy_engine(CommitPolicy::EveryDocs(0)).await;
        for i in 1..=5u64 {
            engine
                .put_document(&format!("d{i}"), title_doc("x"))
                .await
                .unwrap();
        }
        assert!(wal_bytes(&storage) > 0, "EveryDocs(0) must not auto-commit");
        engine.commit().await.unwrap();
        assert_eq!(
            wal_bytes(&storage),
            0,
            "an explicit commit still truncates the WAL"
        );
    }

    /// #890: the default policy is `Manual` — no auto-commit until the caller
    /// commits explicitly.
    #[tokio::test]
    async fn manual_default_no_autocommit() {
        assert_eq!(CommitPolicy::default(), CommitPolicy::Manual);
        let (engine, storage) = commit_policy_engine(CommitPolicy::Manual).await;
        for i in 1..=5u64 {
            engine
                .put_document(&format!("d{i}"), title_doc("x"))
                .await
                .unwrap();
        }
        assert!(wal_bytes(&storage) > 0, "Manual must never auto-commit");
    }

    /// #890: a manual commit between auto-commits resets the counter, so the
    /// next auto-commit lands n documents after the reset — not at a stale
    /// offset carried over from before the manual commit.
    #[tokio::test]
    async fn commit_resets_autocommit_counter() {
        let (engine, storage) = commit_policy_engine(CommitPolicy::EveryDocs(3)).await;
        // 2 docs, then a manual commit resets the counter to 0.
        for i in 1..=2u64 {
            engine
                .put_document(&format!("d{i}"), title_doc("x"))
                .await
                .unwrap();
        }
        engine.commit().await.unwrap();
        assert_eq!(wal_bytes(&storage), 0);

        // The next auto-commit must be 3 docs after the reset (at d5), not 1
        // (which would happen if the counter still held its pre-commit value 2).
        engine.put_document("d3", title_doc("x")).await.unwrap();
        assert!(
            wal_bytes(&storage) > 0,
            "1 doc after a manual commit must not auto-commit (counter was reset)"
        );
        engine.put_document("d4", title_doc("x")).await.unwrap();
        assert!(wal_bytes(&storage) > 0);
        engine.put_document("d5", title_doc("x")).await.unwrap();
        assert_eq!(
            wal_bytes(&storage),
            0,
            "auto-commit lands exactly 3 docs after the manual-commit reset"
        );
    }

    /// #890: an auto-committed document is durably searchable without any
    /// explicit commit from the caller (end-to-end wiring through the ladder).
    #[tokio::test]
    async fn every_docs_autocommit_is_searchable_without_explicit_commit() {
        let (engine, _storage) = commit_policy_engine(CommitPolicy::EveryDocs(2)).await;
        engine.put_document("a", title_doc("alpha")).await.unwrap();
        engine.put_document("b", title_doc("bravo")).await.unwrap();
        // No explicit commit — the 2nd put reached the threshold and committed.
        use crate::lexical::search::searcher::LexicalSearchQuery;
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("title:alpha"))
            .limit(10)
            .build();
        assert_eq!(
            engine.search(request).await.unwrap().len(),
            1,
            "an auto-committed doc must be searchable without an explicit commit"
        );
    }

    /// #892: under `CommitPolicy::Interval` the background timer commits a
    /// trailing doc even while ingestion is idle — the WAL is truncated within
    /// the interval with no explicit commit from the caller.
    #[tokio::test(flavor = "multi_thread")]
    async fn interval_commits_while_idle() {
        let (engine, storage) =
            commit_policy_engine(CommitPolicy::Interval(Duration::from_millis(30))).await;
        engine.put_document("d1", title_doc("idle")).await.unwrap();
        assert!(
            wal_bytes(&storage) > 0,
            "the doc is uncommitted immediately after the put"
        );

        // Poll up to ~2s for the timer to run the commit ladder (WAL truncated).
        let mut committed = false;
        for _ in 0..200 {
            if wal_bytes(&storage) == 0 {
                committed = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            committed,
            "the Interval timer must commit the idle doc within its interval (#892)"
        );
    }

    /// #892: dropping an engine built with `CommitPolicy::Interval` returns
    /// promptly — the timer holds only `Arc` sub-parts (not the `Engine`), so
    /// there is no reference cycle and the drop stops and joins the thread.
    #[tokio::test(flavor = "multi_thread")]
    async fn interval_engine_drops_cleanly() {
        let (engine, _storage) =
            commit_policy_engine(CommitPolicy::Interval(Duration::from_millis(20))).await;
        engine.put_document("d1", title_doc("drop")).await.unwrap();
        // Let the timer tick at least once, then drop and time the shutdown.
        std::thread::sleep(Duration::from_millis(50));
        let started = std::time::Instant::now();
        drop(engine);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "dropping an Interval engine must not hang (no reference cycle, clean join)"
        );
    }

    /// #892: a doc committed by the Interval timer is durably searchable without
    /// any explicit commit from the caller (end-to-end through the ladder).
    #[tokio::test(flavor = "multi_thread")]
    async fn interval_commits_are_searchable() {
        let (engine, storage) =
            commit_policy_engine(CommitPolicy::Interval(Duration::from_millis(30))).await;
        engine.put_document("a", title_doc("alpha")).await.unwrap();
        // Wait for the timer to commit.
        for _ in 0..200 {
            if wal_bytes(&storage) == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        use crate::lexical::search::searcher::LexicalSearchQuery;
        let request = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("title:alpha"))
            .limit(10)
            .build();
        assert_eq!(
            engine.search(request).await.unwrap().len(),
            1,
            "an Interval-committed doc must be searchable without an explicit commit"
        );
    }

    /// #876: a mutation whose WAL record was appended but not yet applied to
    /// the stores when a commit starts must not lose that record to the WAL
    /// truncate.
    ///
    /// `index_internal` appends the WAL record for a mutation *before* applying
    /// it to the lexical/vector stores, so there is a real window — between the
    /// append and the apply — where a concurrent commit could run. This test
    /// makes that window deterministic instead of racing threads (which would
    /// deadlock: every ladder step holds a lock the "racing" apply needs): it
    /// appends "b"'s WAL record directly via `log.append`, exactly as
    /// `index_internal` does, but — like a mutation caught mid-flight — does
    /// NOT apply it to the stores before `commit()` runs. `commit()` must
    /// snapshot `applied_seq` (covering only "a", already applied) BEFORE
    /// materializing, so "b"'s un-applied record is retained by the truncate
    /// and replayed on the next recovery.
    #[tokio::test]
    async fn wal_record_appended_but_not_yet_applied_survives_commit_truncate() {
        let (engine, storage) = commit_policy_engine(CommitPolicy::Manual).await;

        // "a" goes through the normal path: WAL-appended AND applied to both
        // stores before `commit()` runs.
        engine.put_document("a", title_doc("alpha")).await.unwrap();

        // "b" simulates a mutation caught between its WAL append and its store
        // apply: append the WAL record directly (mirrors `index_internal`'s
        // step 2) without upserting it into `lexical`/`vector` (which would be
        // steps 4-5) — `applied_seq` is therefore NOT advanced for it.
        engine.log.append("b", title_doc("bravo")).unwrap();

        // The commit ladder must snapshot `applied_seq` before materializing,
        // so "b" — appended but never applied — is retained by the truncate
        // rather than wiped with the rest of the (now-covered) WAL.
        engine.commit().await.unwrap();
        assert!(
            wal_bytes(&storage) > 0,
            "the WAL record for the not-yet-applied mutation must survive the \
             commit's truncate (#876)"
        );

        // Recovery must replay it: reopen on the same storage.
        drop(engine);
        let reopened = Engine::new(storage.clone(), {
            use crate::engine::schema::FieldOption;
            use crate::lexical::core::field::TextOption;
            Schema::builder()
                .add_field("title", FieldOption::Text(TextOption::default()))
                .build()
        })
        .await
        .unwrap();

        use crate::lexical::search::searcher::LexicalSearchQuery;
        let alpha = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("title:alpha"))
            .limit(10)
            .build();
        assert_eq!(reopened.search(alpha).await.unwrap().len(), 1);

        let bravo = crate::engine::search::SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::from("title:bravo"))
            .limit(10)
            .build();
        assert_eq!(
            reopened.search(bravo).await.unwrap().len(),
            1,
            "the mutation caught mid-flight during commit must be recovered \
             from its retained WAL record (#876)"
        );
    }

    /// #876: `recover()`'s own internal `commit()` must actually empty the
    /// WAL, not just replay the records into the stores.
    ///
    /// `recover()` mutates the stores directly rather than through
    /// `index_internal`, so it must publish `applied_seq` itself; otherwise the
    /// `commit()` it calls at the end snapshots the stale build-time `0`,
    /// `truncate_retaining_after` takes the slow path unconditionally, and
    /// re-retains every record it just durably committed — the WAL never
    /// shrinks until an unrelated future mutation happens to raise
    /// `applied_seq` past it.
    #[tokio::test]
    async fn recover_commit_empties_the_wal() {
        let (engine, storage) = commit_policy_engine(CommitPolicy::Manual).await;
        engine.put_document("a", title_doc("alpha")).await.unwrap();
        // Crash: drop without an explicit commit, leaving "a" WAL-only.
        drop(engine);
        assert!(
            wal_bytes(&storage) > 0,
            "precondition: WAL has a pending record"
        );

        // Reopen: recover() replays "a" and commits internally.
        let schema = {
            use crate::engine::schema::FieldOption;
            use crate::lexical::core::field::TextOption;
            Schema::builder()
                .add_field("title", FieldOption::Text(TextOption::default()))
                .build()
        };
        let reopened = Engine::new(storage.clone(), schema).await.unwrap();
        assert_eq!(
            wal_bytes(&storage),
            0,
            "recovery's own commit must empty the WAL, matching the documented \
             post-commit invariant (#876)"
        );

        // An immediately-following no-op commit must stay a no-op (fast path),
        // not repeat a read-back/rewrite of the same stale tail forever.
        reopened.commit().await.unwrap();
        assert_eq!(wal_bytes(&storage), 0);
    }
}
