//! Python-facing [`Index`] class — the primary entry point for the laurus binding.

use std::path::Path;
use std::sync::Arc;

use crate::convert::{dict_to_document, document_to_dict};
use crate::errors::{closed_err, index_dir_err, laurus_err, reload_requires_path_err};
use crate::schema::PySchema;
use crate::search::{PySearchResult, build_request_from_py, to_py_search_result};
use laurus::{
    CommitPolicy, DEFAULT_GROUP_MAX_BYTES, DEFAULT_GROUP_MAX_RECORDS, Embedder, Engine,
    EngineStats, Schema, Storage, StorageConfig, StorageFactory, WalSyncPolicy,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// WalSyncPolicy
// ---------------------------------------------------------------------------

/// Durability policy that controls when Write-Ahead Log (WAL) appends are
/// flushed (`fsync`'d) to durable storage.
///
/// This is a value object wrapping the Rust [`laurus::WalSyncPolicy`]. It is
/// passed to [`Index`] at construction time via the `wal_sync_policy` keyword
/// argument. It trades the durability of an *individual* write against ingest
/// throughput; [`Index.commit`] is always a hard durability barrier regardless
/// of the policy in effect.
///
/// ## Constructing a policy
///
/// ```python
/// import laurus
///
/// # Per-record durability (the default): every append is fsync'd
/// # before it returns. Safest, lowest ingest throughput.
/// policy = laurus.WalSyncPolicy.per_record()
///
/// # Group-commit durability with default thresholds (batch fsyncs).
/// policy = laurus.WalSyncPolicy.group()
///
/// # Group-commit with explicit thresholds and a flush interval.
/// policy = laurus.WalSyncPolicy.group(
///     max_records=4096,
///     max_bytes=4 * 1024 * 1024,
///     max_interval_ms=1000,
/// )
///
/// index = laurus.Index(wal_sync_policy=policy)
/// ```
#[pyclass(name = "WalSyncPolicy", skip_from_py_object)]
#[derive(Clone)]
pub struct PyWalSyncPolicy {
    /// The wrapped Rust durability policy.
    pub inner: WalSyncPolicy,
}

#[pymethods]
impl PyWalSyncPolicy {
    /// Create a per-record durability policy.
    ///
    /// Every WAL append is fsync'd to durable storage before the write call
    /// returns. This is the safest policy and the default behaviour when no
    /// `wal_sync_policy` is supplied to [`Index`], at the cost of the lowest
    /// ingest throughput.
    ///
    /// Returns:
    ///     A `WalSyncPolicy` wrapping `WalSyncPolicy::PerRecord`.
    #[staticmethod]
    pub fn per_record() -> Self {
        Self {
            inner: WalSyncPolicy::PerRecord,
        }
    }

    /// Create a group-commit durability policy.
    ///
    /// WAL appends are batched and fsync'd together once any of the configured
    /// thresholds is reached, rather than one fsync per record. This increases
    /// ingest throughput at the cost of potentially losing the last unsynced
    /// batch on a crash. [`Index.commit`] remains a hard durability barrier,
    /// and [`Index.flush_wal`] forces a flush on demand.
    ///
    /// Args:
    ///     max_records: Flush after this many records accumulate since the last
    ///         flush. Defaults to laurus' built-in
    ///         `DEFAULT_GROUP_MAX_RECORDS` (1024) when `None`.
    ///     max_bytes: Flush after this many bytes accumulate since the last
    ///         flush. Defaults to laurus' built-in `DEFAULT_GROUP_MAX_BYTES`
    ///         (1 MiB) when `None`.
    ///     max_interval_ms: Optional time-based flush interval in milliseconds.
    ///         When set, a background timer flushes the WAL at least this often
    ///         even if the size thresholds have not been reached. When `None`
    ///         (the default) no time-based flushing occurs.
    ///
    /// Returns:
    ///     A `WalSyncPolicy` wrapping `WalSyncPolicy::Group { .. }`.
    #[staticmethod]
    #[pyo3(signature = (max_records=None, max_bytes=None, max_interval_ms=None))]
    pub fn group(
        max_records: Option<usize>,
        max_bytes: Option<usize>,
        max_interval_ms: Option<u64>,
    ) -> Self {
        Self {
            inner: WalSyncPolicy::Group {
                max_records: max_records.unwrap_or(DEFAULT_GROUP_MAX_RECORDS),
                max_bytes: max_bytes.unwrap_or(DEFAULT_GROUP_MAX_BYTES),
                max_interval: max_interval_ms.map(std::time::Duration::from_millis),
            },
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            WalSyncPolicy::PerRecord => "WalSyncPolicy.per_record()".to_string(),
            WalSyncPolicy::Group {
                max_records,
                max_bytes,
                max_interval,
            } => format!(
                "WalSyncPolicy.group(max_records={}, max_bytes={}, max_interval_ms={})",
                max_records,
                max_bytes,
                match max_interval {
                    Some(d) => d.as_millis().to_string(),
                    None => "None".to_string(),
                }
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// CommitPolicy
// ---------------------------------------------------------------------------

/// Auto-commit policy that controls when the engine automatically runs the
/// commit ladder during ingestion.
///
/// This is a value object wrapping the Rust [`laurus::CommitPolicy`]. It is
/// passed to [`Index`] at construction time via the `commit_policy` keyword
/// argument. By default the engine commits only when [`Index.commit`] is
/// called explicitly; a non-`manual` policy makes it commit automatically at
/// an ingestion-driven cadence. This is orthogonal to `wal_sync_policy`.
///
/// ## Constructing a policy
///
/// ```python
/// import laurus
///
/// # Manual (the default): the caller drives every commit().
/// policy = laurus.CommitPolicy.manual()
///
/// # Auto-commit after every 1000 applied documents.
/// policy = laurus.CommitPolicy.every_docs(1000)
///
/// index = laurus.Index(commit_policy=policy)
/// ```
#[pyclass(name = "CommitPolicy", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCommitPolicy {
    /// The wrapped Rust auto-commit policy.
    pub inner: CommitPolicy,
}

#[pymethods]
impl PyCommitPolicy {
    /// Create a manual (no auto-commit) policy.
    ///
    /// The engine commits only when [`Index.commit`] is called explicitly.
    /// This is the default when no `commit_policy` is supplied to [`Index`].
    ///
    /// Returns:
    ///     A `CommitPolicy` wrapping `CommitPolicy::Manual`.
    #[staticmethod]
    pub fn manual() -> Self {
        Self {
            inner: CommitPolicy::Manual,
        }
    }

    /// Create an auto-commit-every-`n`-documents policy.
    ///
    /// The engine runs the commit ladder after every `n` applied documents,
    /// across the singular and batch ingest APIs (and every `n` documents
    /// within a single batch). `every_docs(0)` disables auto-commit, which is
    /// equivalent to [`CommitPolicy.manual`].
    ///
    /// Args:
    ///     n: Commit after this many applied documents. `0` disables
    ///         auto-commit.
    ///
    /// Returns:
    ///     A `CommitPolicy` wrapping `CommitPolicy::EveryDocs(n)`.
    #[staticmethod]
    pub fn every_docs(n: usize) -> Self {
        Self {
            inner: CommitPolicy::EveryDocs(n),
        }
    }

    /// Create an auto-commit-every-`ms`-milliseconds policy.
    ///
    /// A background timer runs the commit ladder at least every `ms`
    /// milliseconds while ingestion is in progress. This is the time-based
    /// counterpart of [`CommitPolicy.every_docs`].
    ///
    /// Note:
    ///     This policy is native-only. On the `wasm32` target the engine never
    ///     starts the background timer, so this policy is a documented no-op
    ///     there (the value still constructs).
    ///
    /// Args:
    ///     ms: Commit at least this often, in milliseconds.
    ///
    /// Returns:
    ///     A `CommitPolicy` wrapping `CommitPolicy::Interval(Duration)`.
    #[staticmethod]
    pub fn interval_ms(ms: u64) -> Self {
        Self {
            inner: CommitPolicy::Interval(std::time::Duration::from_millis(ms)),
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            CommitPolicy::Manual => "CommitPolicy.manual()".to_string(),
            CommitPolicy::EveryDocs(n) => format!("CommitPolicy.every_docs({n})"),
            CommitPolicy::Interval(d) => {
                format!("CommitPolicy.interval_ms({})", d.as_millis())
            }
            // `CommitPolicy` is #[non_exhaustive]; a future variant renders
            // generically rather than failing to compile.
            _ => "CommitPolicy(<unknown>)".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Laurus search index — the main entry point for the Python binding.
///
/// ## Creating an index
///
/// ```python
/// import laurus
///
/// # In-memory (ephemeral, great for prototyping)
/// index = laurus.Index()
///
/// # File-based (persistent)
/// schema = laurus.Schema()
/// schema.add_text_field("title")
/// schema.add_text_field("body")
/// schema.add_hnsw_field("embedding", dimension=384)
/// index = laurus.Index(path="./myindex", schema=schema)
/// ```
///
/// ## Adding documents
///
/// ```python
/// index.put_document("doc1", {"title": "Hello", "body": "World"})
/// index.commit()
/// ```
///
/// ## Searching
///
/// ```python
/// # DSL string
/// results = index.search("title:hello", limit=10)
///
/// # Query object
/// results = index.search(laurus.TermQuery("body", "rust"), limit=5)
///
/// # Pre-computed vector
/// results = index.search(laurus.VectorQuery("embedding", vec), limit=5)
///
/// # Hybrid via SearchRequest
/// request = laurus.SearchRequest(
///     lexical_query=laurus.TermQuery("body", "async"),
///     vector_query=laurus.VectorTextQuery("embedding", "concurrent"),
///     fusion=laurus.RRF(k=60.0),
///     limit=3,
/// )
/// results = index.search(request)
/// ```
#[pyclass(name = "Index")]
pub struct PyIndex {
    engine: Option<Arc<Engine>>,
    rt: Arc<tokio::runtime::Runtime>,
    /// Directory path this index was constructed with, retained so
    /// [`Self::reload`] can reopen the same directory. `None` for an
    /// in-memory index.
    path: Option<String>,
    /// Durability policy this index was constructed with, retained so a
    /// [`Self::reload`] doesn't silently reset it back to the default.
    wal_sync_policy: Option<PyWalSyncPolicy>,
    /// Auto-commit policy this index was constructed with, retained for the
    /// same reason as `wal_sync_policy`.
    commit_policy: Option<PyCommitPolicy>,
    /// Schema of the most recently built `Engine` (construction or
    /// `reload`), used to decide whether [`Self::reload`] can reuse
    /// `last_embedder` instead of rebuilding the embedder(s) from scratch.
    last_schema: Option<Schema>,
    /// Embedder of the most recently built `Engine`, reused by
    /// [`Self::reload`] when `last_schema` matches the freshly-read schema.
    last_embedder: Option<Arc<dyn Embedder>>,
    /// Commit generation of the most recently built `Engine`, used as the
    /// baseline [`Self::reload`] compares against to report whether
    /// anything actually changed.
    last_generation: u64,
}

// Every engine call below releases the GIL via `py.detach`, so two Python
// threads can now hold `&PyIndex` at the same time. pyo3 only enforces
// `Send` on GIL-enabled builds, so assert `Sync` here: a future field that
// isn't `Sync` must not silently break that.
const _: () = {
    const fn assert_sync<T: Sync>() {}
    assert_sync::<PyIndex>();
};

#[pymethods]
impl PyIndex {
    /// Create a new index, or reopen an existing one.
    ///
    /// When `path` is given, the directory follows the same
    /// `<path>/schema.toml` + `<path>/store/` layout `laurus-cli create
    /// index`/`--index-dir` uses, so an index built here can be opened by
    /// the CLI (and vice versa) without any path juggling.
    ///
    /// * If `<path>/schema.toml` does not yet exist, this **creates** a new
    ///   index: the given `schema` (or an empty one, if omitted) is
    ///   persisted to `<path>/schema.toml`.
    /// * If `<path>/schema.toml` already exists, this **reopens** the
    ///   index: `schema` must be omitted (`None`) — the persisted schema
    ///   is loaded instead. Passing an explicit `schema` here raises
    ///   `ValueError`, since it would be ambiguous which one should win.
    ///
    /// Args:
    ///     path: Directory path for persistent storage.
    ///           Pass `None` (default) for an ephemeral in-memory index.
    ///     schema: Schema definition. Required (or optional) only when
    ///           *creating* a new index; must be omitted when reopening an
    ///           existing one. If omitted for both an in-memory index and a
    ///           brand-new file-backed one, an empty schema is used.
    ///     wal_sync_policy: Optional [`WalSyncPolicy`] controlling when WAL
    ///           appends are fsync'd. When `None` (the default), laurus uses
    ///           per-record durability (every append is fsync'd before it
    ///           returns).
    ///     commit_policy: Optional [`CommitPolicy`] controlling automatic
    ///           commits during ingestion. When `None` (the default), the
    ///           engine is manual — the caller drives every `commit()`.
    ///
    /// Raises:
    ///     ValueError: if `path` points at an existing index and `schema`
    ///         was also given, or if `path` contains an index in the
    ///         pre-existing (pre-Issue-1059) flat layout.
    #[new]
    #[pyo3(signature = (path=None, schema=None, wal_sync_policy=None, commit_policy=None))]
    pub fn new(
        py: Python,
        path: Option<String>,
        schema: Option<&PySchema>,
        wal_sync_policy: Option<&PyWalSyncPolicy>,
        commit_policy: Option<&PyCommitPolicy>,
    ) -> PyResult<Self> {
        let rt =
            tokio::runtime::Runtime::new().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let schema_arg = schema.map(|s| s.inner.clone());
        let wal_sync_policy_inner = wal_sync_policy.map(|p| p.inner);
        let commit_policy_inner = commit_policy.map(|p| p.inner);
        let path_for_resolve = path.clone();

        let engine = py.detach(|| -> PyResult<Engine> {
            let (schema_arg, storage) =
                resolve_storage_and_schema(path_for_resolve.as_deref(), schema_arg)?;

            let mut builder = Engine::builder(storage, schema_arg);
            if let Some(p) = wal_sync_policy_inner {
                builder = builder.wal_sync_policy(p);
            }
            if let Some(p) = commit_policy_inner {
                builder = builder.commit_policy(p);
            }

            rt.block_on(builder.build()).map_err(laurus_err)
        })?;

        Ok(Self {
            last_schema: Some(engine.schema()),
            last_embedder: Some(engine.embedder()),
            last_generation: engine.commit_generation(),
            engine: Some(Arc::new(engine)),
            rt: Arc::new(rt),
            path,
            wal_sync_policy: wal_sync_policy.cloned(),
            commit_policy: commit_policy.cloned(),
        })
    }

    /// Release this index's handle on the underlying engine, deterministically
    /// dropping its storage lock (Issue #1086/#1097) instead of waiting on
    /// CPython's garbage collector.
    ///
    /// Idempotent: calling `close()` more than once is a no-op. Every other
    /// method raises `RuntimeError` after `close()` has been called --
    /// **except** [`Self::reload`], which is documented to work after
    /// `close()` too.
    pub fn close(&mut self, py: Python) {
        let engine = self.engine.take();
        // Dropping the last `Arc<Engine>` releases the storage lock
        // (Issue #1086/#1097), which is blocking I/O -- do it with the GIL
        // released.
        py.detach(|| drop(engine));
    }

    /// Rebuild this index's `Engine` from the directory it was constructed
    /// with, picking up any changes committed by another process since it
    /// was last (re)opened -- without paying the cost of reconstructing the
    /// embedding model(s) when the schema's embedding configuration hasn't
    /// changed.
    ///
    /// Unlike every other method, `reload()` works whether this `Index` is
    /// currently open **or already `close()`d** -- it reopens the same
    /// directory either way. This is deliberate: it lets a caller hold onto
    /// one `Index` object across a full reload cycle instead of having to
    /// construct a new one and swap references, and it means "another
    /// process wrote while I wasn't looking" is exactly what `close()` then
    /// `reload()` naturally expresses.
    ///
    /// # Why the storage lock isn't held continuously across the swap
    ///
    /// Issue #1086's directory lock is a single, process-agnostic exclusive
    /// lock: while it is held, no other `Engine` -- in this process or any
    /// other -- can be built over the same directory. Holding it
    /// continuously across every `reload()` call would therefore be
    /// self-defeating: an external writer could never acquire it in the
    /// first place, so there would be nothing for `reload()` to ever pick
    /// up. `reload()` instead releases the lock and immediately reacquires
    /// it while rebuilding, back to back within this one call. This is safe
    /// from same-process races without any extra locking: `reload()` and
    /// `close()` take `&mut self`, which pyo3 only grants as an *exclusive*
    /// borrow (`PyRefMut`) -- it cannot be acquired while any other method
    /// call (an ordinary `&self` / shared borrow, held for that call's full
    /// duration, GIL released or not) is in flight on this object. A
    /// `reload()`/`close()` that overlaps another in-flight call instead
    /// fails fast with `RuntimeError: Already borrowed`, rather than racing
    /// silently; callers that see this are expected to retry.
    ///
    /// # Embedder reuse
    ///
    /// If the freshly-read schema is identical to the schema of the
    /// previously-built engine, the already-loaded embedder(s) are reused
    /// as-is instead of being reconstructed (which, for `CandleBertEmbedder`
    /// et al., means skipping a fresh HF-cache lookup, safetensors mmap, and
    /// tokenizer load). This comparison is on the *whole* schema, not just
    /// the embedding-relevant subset, so an unrelated schema change (e.g. a
    /// new non-vector field) forfeits this optimization even though it
    /// didn't strictly need to -- a deliberate simplification.
    ///
    /// # Errors
    ///
    /// Raises `ValueError` if this index has no directory (was constructed
    /// with `path=None`). Raises the usual construction errors if rebuilding
    /// the engine fails; in that case this index is left `close()`d (the
    /// old engine's lock was already released), but every cached value
    /// needed for a retry (path, schema, embedder) is preserved, so calling
    /// `reload()` again retries cleanly.
    ///
    /// Returns:
    ///     `True` if the commit generation advanced (something was actually
    ///     picked up), `False` if the index was already up to date.
    pub fn reload(&mut self, py: Python) -> PyResult<bool> {
        let path = self.path.clone().ok_or_else(reload_requires_path_err)?;

        // Read the generation off the *live* engine, if there is one, not
        // the `last_generation` cache: a commit made through this same
        // `Index` since it was last (re)opened already advanced the live
        // engine's generation without updating the cache (only `new`/
        // `reload` do that), so comparing against the cache here would
        // wrongly report changes the caller already knows about via their
        // own `commit()` call. Falls back to the cache when already
        // `close()`d, since there's no live engine to read at that point.
        let baseline_generation = self
            .engine
            .as_ref()
            .map_or(self.last_generation, |engine| engine.commit_generation());

        // Release the current engine's storage lock deterministically
        // before rebuilding -- see the doc comment above for why this
        // cannot instead hold the lock across the whole call.
        self.engine = None;

        let last_schema = self.last_schema.clone();
        let last_embedder = self.last_embedder.clone();
        let wal_sync_policy = self.wal_sync_policy.as_ref().map(|p| p.inner);
        let commit_policy = self.commit_policy.as_ref().map(|p| p.inner);
        let rt = self.rt.clone();

        let engine = py.detach(move || -> PyResult<Engine> {
            let (new_schema, storage) =
                laurus::index_dir::open_or_create(Path::new(&path), None).map_err(index_dir_err)?;

            let reuse_embedder = last_schema.as_ref().is_some_and(|old_schema| {
                schemas_match_for_embedder_reuse(old_schema, &new_schema)
            });

            let mut builder = Engine::builder(storage, new_schema);
            if reuse_embedder && let Some(embedder) = &last_embedder {
                builder = builder.embedder(embedder.clone());
            }
            if let Some(policy) = wal_sync_policy {
                builder = builder.wal_sync_policy(policy);
            }
            if let Some(policy) = commit_policy {
                builder = builder.commit_policy(policy);
            }

            rt.block_on(builder.build()).map_err(laurus_err)
        })?;

        let changed = engine.commit_generation() != baseline_generation;
        self.last_schema = Some(engine.schema());
        self.last_embedder = Some(engine.embedder());
        self.last_generation = engine.commit_generation();
        self.engine = Some(Arc::new(engine));
        Ok(changed)
    }

    /// Return the current commit generation (Issue #1088) in O(1).
    ///
    /// A monotonically increasing counter, persisted across restarts, that
    /// advances by 1 on every commit that actually applied a document
    /// (put/add/delete) since the previous one. Unlike reading
    /// `commit_generation` through [`Self::stats`], this does not scan any
    /// vector fields, so it's cheap to call as often as needed.
    ///
    /// This is a snapshot held in memory by the currently-loaded `Engine`,
    /// not re-read from disk on every call: it only reflects commits made
    /// through *this* `Index` object (confirming that [`Self::reload`] or
    /// your own `commit()` actually advanced the state), not commits made
    /// by another process. Only [`Self::reload`] can pick those up.
    pub fn commit_generation(&self) -> PyResult<u64> {
        Ok(self.engine()?.commit_generation())
    }

    // ── Document CRUD ─────────────────────────────────────────────────────

    /// Index a document, replacing any existing document with the same id.
    ///
    /// Args:
    ///     id: External document identifier (string).
    ///     doc: A `dict` mapping field names to values.
    ///
    /// Call [`commit`] to make the change visible to searches.
    pub fn put_document(&self, py: Python, id: &str, doc: &Bound<PyDict>) -> PyResult<()> {
        let document = dict_to_document(py, doc)?;
        let engine = self.engine()?;
        let id = id.to_string();
        py.detach(|| self.rt.block_on(engine.put_document(&id, document)))
            .map_err(laurus_err)
    }

    /// Append a document version without removing existing versions.
    ///
    /// Laurus supports multiple versions of the same id (chunk-per-document
    /// RAG pattern).  Use [`put_document`] to replace.
    ///
    /// Args:
    ///     id: External document identifier.
    ///     doc: A `dict` mapping field names to values.
    pub fn add_document(&self, py: Python, id: &str, doc: &Bound<PyDict>) -> PyResult<()> {
        let document = dict_to_document(py, doc)?;
        let engine = self.engine()?;
        let id = id.to_string();
        py.detach(|| self.rt.block_on(engine.add_document(&id, document)))
            .map_err(laurus_err)
    }

    /// Index many documents in one call, replacing existing documents by id.
    ///
    /// Batched form of [`put_document`]: the `(id, dict)` pairs are applied
    /// sequentially, in order, with a single WAL fsync for the whole batch.
    /// Duplicate ids within one batch deduplicate exactly like the same puts
    /// issued one by one (the last occurrence wins).
    ///
    /// Args:
    ///     docs: An iterable of `(id, dict)` pairs.
    ///
    /// Fails fast at the first document that cannot be indexed; the raised
    /// error names the failing position and id. Documents applied before the
    /// failure are **not** rolled back (retrying the batch is idempotent).
    /// Call [`commit`] to make the changes visible to searches.
    pub fn put_documents(&self, py: Python, docs: &Bound<PyAny>) -> PyResult<()> {
        let batch = pairs_to_documents(py, docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        let engine = self.engine()?;
        py.detach(|| self.rt.block_on(engine.put_documents(batch)))
            .map_err(laurus_err)
    }

    /// Append many document versions in one call, without removing existing
    /// versions.
    ///
    /// Batched form of [`add_document`]. Ordering, single-fsync durability,
    /// and fail-fast error semantics match [`put_documents`], but repeated
    /// ids accumulate as separate versions instead of deduplicating.
    ///
    /// Args:
    ///     docs: An iterable of `(id, dict)` pairs.
    pub fn add_documents(&self, py: Python, docs: &Bound<PyAny>) -> PyResult<()> {
        let batch = pairs_to_documents(py, docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        let engine = self.engine()?;
        py.detach(|| self.rt.block_on(engine.add_documents(batch)))
            .map_err(laurus_err)
    }

    /// Retrieve all document versions stored under `id`.
    ///
    /// Returns a list of dicts, one per indexed version.
    pub fn get_documents(&self, py: Python, id: &str) -> PyResult<Vec<Py<PyAny>>> {
        let engine = self.engine()?;
        let id = id.to_string();
        let docs = py
            .detach(|| self.rt.block_on(engine.get_documents(&id)))
            .map_err(laurus_err)?;
        docs.iter().map(|doc| document_to_dict(py, doc)).collect()
    }

    /// Delete all document versions stored under `id`.
    ///
    /// Call [`commit`] to make the deletion visible to searches.
    pub fn delete_documents(&self, py: Python, id: &str) -> PyResult<()> {
        let engine = self.engine()?;
        let id = id.to_string();
        py.detach(|| self.rt.block_on(engine.delete_documents(&id)))
            .map_err(laurus_err)
    }

    /// Flush buffered writes and make all pending changes searchable.
    pub fn commit(&self, py: Python) -> PyResult<()> {
        let engine = self.engine()?;
        py.detach(|| self.rt.block_on(engine.commit()))
            .map_err(laurus_err)
    }

    /// Force any buffered Write-Ahead Log (WAL) appends to be flushed
    /// (`fsync`'d) to durable storage.
    ///
    /// This matters only under a group-commit [`WalSyncPolicy`]
    /// ([`WalSyncPolicy.group`]), where individual appends are batched and not
    /// fsync'd immediately. Under the default per-record policy
    /// ([`WalSyncPolicy.per_record`]) every append is already durable, so this
    /// call is effectively a no-op.
    ///
    /// Durability trade-off: with group commit, a crash can lose the most
    /// recent unsynced batch of appends. Call `flush_wal()` to bound that
    /// window on demand without paying for a full [`commit`], which would also
    /// materialize the in-memory index state. Use `flush_wal()` when you want
    /// the WAL durable but do not yet need the pending changes to be
    /// searchable; use [`commit`] when you need both.
    ///
    /// This call is synchronous and does not make any pending changes
    /// searchable; use [`commit`] for that.
    ///
    /// Raises:
    ///     An exception if the underlying WAL flush fails (for example, an I/O
    ///     error while fsync'ing).
    pub fn flush_wal(&self, py: Python) -> PyResult<()> {
        let engine = self.engine()?;
        py.detach(|| engine.flush_wal()).map_err(laurus_err)
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Search the index and return a list of [`SearchResult`] objects.
    ///
    /// `query` may be:
    ///   - A **DSL string** (e.g. `"title:hello"`, `"~\"memory safety\""`)
    ///   - A **lexical query** object (`TermQuery`, `BooleanQuery`, `GeoDistanceQuery`, …)
    ///   - A **vector query** object (`VectorQuery`, `VectorTextQuery`)
    ///   - A **[`SearchRequest`]** for full control (hybrid, filter, fusion)
    ///
    /// Args:
    ///     query: The query to execute.
    ///     limit: Maximum number of results to return (default 10).
    ///     offset: Pagination offset (default 0).
    ///     highlight: Request highlighted fragments per field (Issue
    ///         #1134). Either a list of field names (`["body"]`) or a dict
    ///         adding `HighlightConfig` knobs (`{"fields": ["body"],
    ///         "max_fragments": 2, "tag": "em"}`). Highlighting follows
    ///         this query, and only `stored: true` text fields can be
    ///         highlighted. `None` (the default) leaves every result's
    ///         `highlights` empty.
    ///
    /// Returns:
    ///     A list of [`SearchResult`] objects with `.id`, `.score`,
    ///     `.document`, `.highlights`.
    #[pyo3(signature = (query, *, limit=10, offset=0, highlight=None))]
    pub fn search(
        &self,
        py: Python,
        query: &Bound<PyAny>,
        limit: usize,
        offset: usize,
        highlight: Option<&Bound<PyAny>>,
    ) -> PyResult<Vec<PySearchResult>> {
        let request = build_request_from_py(py, query, limit, offset, highlight)?;

        let engine = self.engine()?;
        let results = py
            .detach(|| self.rt.block_on(engine.search(request)))
            .map_err(laurus_err)?;

        results
            .into_iter()
            .map(|r| to_py_search_result(py, r))
            .collect()
    }

    /// Execute multiple independent searches in one call.
    ///
    /// Each query in `queries` is dispatched in parallel on the underlying
    /// tokio runtime via `laurus::Engine::search_batch`. Each entry can
    /// be the same kind of value `search()` accepts: a DSL string, a
    /// `LexicalQuery` / `VectorQuery` / `VectorTextQuery` object, or a
    /// `SearchRequest`. The same `limit` and `offset` are applied to
    /// every query in the batch.
    ///
    /// Args:
    ///     queries: A list of queries to execute. Order is preserved in
    ///         the output.
    ///     limit: Maximum number of results to return per query
    ///         (default 10).
    ///     offset: Pagination offset applied to each query (default 0).
    ///     highlight: Request highlighted fragments per field, applied
    ///         identically to every query in the batch — same format as
    ///         `search`'s `highlight` parameter (Issue #1134).
    ///
    /// Returns:
    ///     A list of lists: `results[i]` is the result list for
    ///     `queries[i]`. Empty input returns an empty list without
    ///     invoking the engine.
    ///
    /// Issue [#717](https://github.com/mosuka/laurus/issues/717)
    /// Phase 3b of [#648](https://github.com/mosuka/laurus/issues/648).
    #[pyo3(signature = (queries, *, limit=10, offset=0, highlight=None))]
    pub fn search_batch(
        &self,
        py: Python,
        queries: &Bound<PyAny>,
        limit: usize,
        offset: usize,
        highlight: Option<&Bound<PyAny>>,
    ) -> PyResult<Vec<Vec<PySearchResult>>> {
        let queries_seq = queries.try_iter().map_err(|_| {
            PyRuntimeError::new_err(
                "search_batch: expected an iterable of queries (DSL string, Query object, or SearchRequest)",
            )
        })?;

        let mut requests = Vec::new();
        for item in queries_seq {
            let item = item?;
            requests.push(build_request_from_py(py, &item, limit, offset, highlight)?);
        }

        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let engine = self.engine()?;
        let batch_results = py
            .detach(|| self.rt.block_on(engine.search_batch(requests)))
            .map_err(laurus_err)?;

        batch_results
            .into_iter()
            .map(|per_query_results| {
                per_query_results
                    .into_iter()
                    .map(|r| to_py_search_result(py, r))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect()
    }

    // ── Schema & stats ────────────────────────────────────────────────────

    /// Return index statistics.
    ///
    /// Returns a dict with keys:
    ///   - `document_count` (int): total indexed documents.
    ///   - `vector_fields` (dict): per-field vector statistics.
    ///   - `commit_generation` (int): monotonically increasing counter,
    ///     persisted across restarts, that advances by 1 on every commit
    ///     that actually applied a document (put/add/delete) since the
    ///     previous one (Issue #1088). Lets a separate process/instance
    ///     reopening this same index directory detect "something changed
    ///     since I last checked" in O(1) instead of hashing the whole
    ///     store directory on a timer. Does not reflect schema changes
    ///     made via `update_field`, and does not advance on a commit with
    ///     nothing new to apply (e.g. an idle auto-commit tick).
    pub fn stats(&self, py: Python) -> PyResult<Py<PyAny>> {
        let engine = self.engine()?;
        let stats: EngineStats = py.detach(|| engine.stats()).map_err(laurus_err)?;
        let dict = PyDict::new(py);
        dict.set_item("document_count", stats.document_count)?;
        let vf = PyDict::new(py);
        for (field, field_stats) in &stats.vector_fields {
            let fd = PyDict::new(py);
            fd.set_item("count", field_stats.vector_count)?;
            fd.set_item("dimension", field_stats.dimension)?;
            vf.set_item(field, fd)?;
        }
        dict.set_item("vector_fields", vf)?;
        dict.set_item("commit_generation", stats.commit_generation)?;
        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "Index()".to_string()
    }
}

impl PyIndex {
    /// Return a clone of the underlying engine handle, or `RuntimeError` if
    /// [`Self::close`] has already been called.
    fn engine(&self) -> PyResult<Arc<Engine>> {
        self.engine.clone().ok_or_else(closed_err)
    }
}

// ---------------------------------------------------------------------------
// Batch-ingestion helper
// ---------------------------------------------------------------------------

/// Convert a Python iterable of `(id, dict)` pairs into the engine's
/// `(String, Document)` batch, naming the offending position on any entry
/// that is not a two-element `(str, dict)` pair.
fn pairs_to_documents(
    py: Python,
    docs: &Bound<PyAny>,
) -> PyResult<Vec<(String, laurus::Document)>> {
    let iter = docs.try_iter().map_err(|_| {
        PyRuntimeError::new_err(
            "expected an iterable of (id, dict) pairs, e.g. [(\"doc1\", {\"title\": \"...\"}), ...]",
        )
    })?;

    let mut batch = Vec::new();
    for (index, item) in iter.enumerate() {
        let item = item?;
        let (id, doc): (String, Bound<PyDict>) = item.extract().map_err(|_| {
            PyRuntimeError::new_err(format!(
                "documents[{index}]: expected a (id: str, doc: dict) pair"
            ))
        })?;
        batch.push((id, dict_to_document(py, &doc)?));
    }
    Ok(batch)
}

// ---------------------------------------------------------------------------
// Storage factory helper
// ---------------------------------------------------------------------------

/// Resolve the `(Schema, Storage)` pair for [`PyIndex::new`].
///
/// `path=None` keeps the pre-existing in-memory behavior (schema defaults
/// to empty, no persistence, no conflict checking). `path=Some(p)` defers
/// to [`laurus::index_dir::open_or_create`], which applies the
/// `<p>/schema.toml` + `<p>/store/` convention shared with `laurus-cli`.
fn resolve_storage_and_schema(
    path: Option<&str>,
    schema: Option<Schema>,
) -> PyResult<(Schema, Arc<dyn Storage>)> {
    match path {
        None => {
            let storage = StorageFactory::create(StorageConfig::Memory(Default::default()))
                .map_err(laurus_err)?;
            Ok((schema.unwrap_or_default(), storage))
        }
        Some(p) => laurus::index_dir::open_or_create(Path::new(p), schema).map_err(index_dir_err),
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Read the persisted commit generation for `path` directly from disk,
/// without building an `Engine` -- no storage lock, no WAL recovery, no
/// embedder loading (Issue #1101).
///
/// Unlike [`PyIndex.reload`], this is not an `Index` method: it works even
/// when no `Index` for `path` has ever been constructed in this process,
/// which is the point -- it lets a caller cheaply decide whether opening
/// (or [`PyIndex.reload`]-ing) the index is worth doing at all.
///
/// Args:
///     path: Directory path of a file-backed index (the same value passed
///         to `Index(path=...)`).
///
/// Returns:
///     `0` if the index exists but nothing has been committed yet.
///
/// Raises:
///     ValueError: if `path` has no persisted schema at all (not a laurus
///         index directory).
#[pyfunction]
pub fn peek_commit_generation(py: Python, path: String) -> PyResult<u64> {
    py.detach(|| laurus::index_dir::peek_commit_generation(Path::new(&path)))
        .map_err(index_dir_err)
}

// ---------------------------------------------------------------------------
// Reload helper
// ---------------------------------------------------------------------------

/// Decide whether [`PyIndex::reload`] can reuse the previously-built
/// embedder instead of reconstructing it from `new_schema`.
///
/// Deliberately compares the *whole* schema (via `Serialize`, since `Schema`
/// has no `PartialEq`) rather than just the embedding-relevant subset: an
/// unrelated schema change (e.g. a new non-vector field) forfeits the reuse
/// optimization even though it didn't strictly need to, but this avoids
/// having to reason about `PerFieldEmbedder`'s field-to-embedder-name
/// routing across a partial schema diff.
fn schemas_match_for_embedder_reuse(old_schema: &Schema, new_schema: &Schema) -> bool {
    serde_json::to_value(old_schema).ok() == serde_json::to_value(new_schema).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_match_for_embedder_reuse_true_when_identical() {
        let schema = Schema::default();
        assert!(schemas_match_for_embedder_reuse(&schema, &schema.clone()));
    }

    #[test]
    fn schemas_match_for_embedder_reuse_false_when_a_field_is_added() {
        let old_schema = Schema::default();
        let new_schema = Schema::builder()
            .add_text_field("title", laurus::lexical::core::field::TextOption::default())
            .build();

        assert!(!schemas_match_for_embedder_reuse(&old_schema, &new_schema));
    }
}
