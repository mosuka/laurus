//! PHP-facing `Index` class — the primary entry point for the laurus binding.

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use ext_php_rs::convert::FromZval;
use ext_php_rs::prelude::*;
use ext_php_rs::types::{ZendHashTable, Zval};
use laurus::{
    CommitPolicy, DEFAULT_GROUP_MAX_BYTES, DEFAULT_GROUP_MAX_RECORDS, Engine, EngineStats, Storage,
    StorageConfig, StorageFactory, WalSyncPolicy,
};

use crate::convert::{document_to_hashtable, hashtable_to_document};
use crate::errors::{closed_err, index_dir_err, laurus_err};
use crate::schema::PhpSchema;
use crate::search::{
    PhpSearchResult, build_request_from_php, parse_highlight_option, to_php_search_result,
};

// ---------------------------------------------------------------------------
// WalSyncPolicy
// ---------------------------------------------------------------------------

/// Write-ahead-log (WAL) durability policy (`Laurus\WalSyncPolicy`).
///
/// This is a value object wrapping the Rust [`laurus::WalSyncPolicy`]. It is
/// passed to [`PhpIndex::__construct`] as the optional third argument and
/// controls when the engine forces appended records durable (fsync).
///
/// Construct one with the static factory methods rather than `new`:
///
/// ```php
/// use Laurus\Index;
/// use Laurus\WalSyncPolicy;
///
/// // Per-record durability (the default): every add/put is fsynced before
/// // it returns, so a successful write can never be lost to a crash.
/// $policy = WalSyncPolicy::perRecord();
///
/// // Group commit with the built-in default thresholds.
/// $policy = WalSyncPolicy::group();
///
/// // Group commit with custom thresholds and a 1 s periodic flush.
/// $policy = WalSyncPolicy::group(4096, 4 * 1024 * 1024, 1000);
///
/// $index = new Index(null, null, $policy);
/// ```
#[php_class]
#[php(name = "Laurus\\WalSyncPolicy")]
#[derive(Clone, Copy)]
pub struct PhpWalSyncPolicy {
    /// The wrapped Rust durability policy passed to the engine builder.
    pub inner: WalSyncPolicy,
}

#[php_impl]
impl PhpWalSyncPolicy {
    /// Create a per-record durability policy.
    ///
    /// Every `addDocument` / `putDocument` is fsynced before it returns, so a
    /// successful write can never be lost to a crash. This is the safest policy
    /// and the engine default when no `wal_sync_policy` is supplied to
    /// [`PhpIndex::__construct`], at the cost of the lowest ingest throughput.
    ///
    /// # Returns
    ///
    /// A `WalSyncPolicy` wrapping [`WalSyncPolicy::PerRecord`].
    pub fn per_record() -> Self {
        Self {
            inner: WalSyncPolicy::PerRecord,
        }
    }

    /// Create a group-commit durability policy.
    ///
    /// The engine defers the fsync and amortizes it over a batch: it flushes
    /// when **either** `max_records` records **or** `max_bytes` bytes have
    /// accumulated since the last sync (whichever comes first), and
    /// unconditionally at `commit()`. This trades per-record durability for
    /// ingest throughput — a crash can lose the most recent unsynced batch of
    /// appends. `commit()` is still a hard durability barrier, and `flushWal()`
    /// forces a flush on demand.
    ///
    /// # Arguments
    ///
    /// * `max_records` - Flush after this many records accumulate since the
    ///   last sync. Defaults to laurus' built-in
    ///   [`DEFAULT_GROUP_MAX_RECORDS`] (1024) when null.
    /// * `max_bytes` - Flush after this many appended bytes accumulate since
    ///   the last sync. Defaults to laurus' built-in
    ///   [`DEFAULT_GROUP_MAX_BYTES`] (1 MiB) when null.
    /// * `max_interval_ms` - Optional periodic flush interval in milliseconds.
    ///   When provided, the engine runs a background timer that forces the WAL
    ///   durable at least this often so a trailing partial batch under a low
    ///   ingest rate is not left unsynced indefinitely. Null disables the
    ///   timer.
    ///
    /// # Returns
    ///
    /// A `WalSyncPolicy` wrapping [`WalSyncPolicy::Group`].
    pub fn group(
        max_records: Option<i64>,
        max_bytes: Option<i64>,
        max_interval_ms: Option<i64>,
    ) -> Self {
        Self {
            inner: WalSyncPolicy::Group {
                max_records: max_records
                    .map(|v| v as usize)
                    .unwrap_or(DEFAULT_GROUP_MAX_RECORDS),
                max_bytes: max_bytes
                    .map(|v| v as usize)
                    .unwrap_or(DEFAULT_GROUP_MAX_BYTES),
                max_interval: max_interval_ms.map(|v| std::time::Duration::from_millis(v as u64)),
            },
        }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        match self.inner {
            WalSyncPolicy::PerRecord => "WalSyncPolicy.perRecord()".to_string(),
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
                    None => "null".to_string(),
                }
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// CommitPolicy
// ---------------------------------------------------------------------------

/// Auto-commit policy controlling when the engine automatically runs the commit
/// ladder during ingestion.
///
/// By default the engine commits only when `commit()` is called explicitly; a
/// non-`manual` policy makes it commit automatically at an ingestion-driven
/// cadence. This is orthogonal to [`PhpWalSyncPolicy`].
///
/// ```php
/// use Laurus\CommitPolicy;
/// use Laurus\Index;
///
/// // Manual (the default): the caller drives every commit().
/// $policy = CommitPolicy::manual();
///
/// // Auto-commit after every 1000 applied documents.
/// $policy = CommitPolicy::everyDocs(1000);
///
/// $index = new Index(null, null, null, $policy);
/// ```
#[php_class]
#[php(name = "Laurus\\CommitPolicy")]
#[derive(Clone, Copy)]
pub struct PhpCommitPolicy {
    /// The wrapped Rust auto-commit policy passed to the engine builder.
    pub inner: CommitPolicy,
}

#[php_impl]
impl PhpCommitPolicy {
    /// Create a manual (no auto-commit) policy.
    ///
    /// The engine commits only when `commit()` is called explicitly. This is
    /// the engine default when no `commit_policy` is supplied to
    /// [`PhpIndex::__construct`].
    ///
    /// # Returns
    ///
    /// A `CommitPolicy` wrapping [`CommitPolicy::Manual`].
    pub fn manual() -> Self {
        Self {
            inner: CommitPolicy::Manual,
        }
    }

    /// Create an auto-commit-every-`n`-documents policy.
    ///
    /// The engine runs the commit ladder after every `n` applied documents,
    /// across the singular and batch ingest APIs (and every `n` documents
    /// within a single batch). `everyDocs(0)` disables auto-commit, which is
    /// equivalent to [`PhpCommitPolicy::manual`].
    ///
    /// # Arguments
    ///
    /// * `n` - Commit after this many applied documents. `0` disables
    ///   auto-commit.
    ///
    /// # Returns
    ///
    /// A `CommitPolicy` wrapping [`CommitPolicy::EveryDocs`].
    pub fn every_docs(n: i64) -> Self {
        Self {
            inner: CommitPolicy::EveryDocs(n.max(0) as usize),
        }
    }

    /// Create an auto-commit-every-`ms`-milliseconds policy (#892).
    ///
    /// The engine runs the commit ladder at least every `ms` milliseconds via a
    /// background timer, so a trailing partial batch is flushed even when
    /// ingestion goes idle. This is the time-based counterpart of
    /// [`PhpCommitPolicy::every_docs`].
    ///
    /// # Platform behavior
    ///
    /// Native only: on `wasm32` the engine never starts the timer, so `Interval`
    /// is a documented no-op there. The factory still constructs the value.
    ///
    /// # Arguments
    ///
    /// * `ms` - Commit at least this often, in milliseconds. Negative values are
    ///   clamped to `0`.
    ///
    /// # Returns
    ///
    /// A `CommitPolicy` wrapping [`CommitPolicy::Interval`].
    pub fn interval_ms(ms: i64) -> Self {
        Self {
            inner: CommitPolicy::Interval(std::time::Duration::from_millis(ms.max(0) as u64)),
        }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        match self.inner {
            CommitPolicy::Manual => "CommitPolicy.manual()".to_string(),
            CommitPolicy::EveryDocs(n) => format!("CommitPolicy.everyDocs({n})"),
            CommitPolicy::Interval(d) => {
                format!("CommitPolicy.intervalMs({})", d.as_millis())
            }
            // `CommitPolicy` is #[non_exhaustive]; render a future variant
            // generically rather than failing to compile.
            _ => "CommitPolicy(<unknown>)".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Laurus search index — the main entry point for the PHP binding
/// (`Laurus\Index`).
///
/// # Creating an index
///
/// ```php
/// use Laurus\Index;
/// use Laurus\Schema;
///
/// // In-memory (ephemeral)
/// $index = new Index();
///
/// // File-based (persistent)
/// $schema = new Schema();
/// $schema->addTextField("title");
/// $index = new Index("./myindex", $schema);
/// ```
///
/// # Searching
///
/// ```php
/// $results = $index->search("title:hello", 10);
/// $results = $index->search(new \Laurus\TermQuery("body", "rust"), 5);
/// ```
#[php_class]
#[php(name = "Laurus\\Index")]
pub struct PhpIndex {
    engine: RefCell<Option<Arc<Engine>>>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[php_impl]
impl PhpIndex {
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
    ///   index: `schema` must be omitted (null) — the persisted schema is
    ///   loaded instead. Passing an explicit `schema` here throws a
    ///   `ValueError`, since it would be ambiguous which one should win.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path for persistent storage. Pass null (default)
    ///   for an ephemeral in-memory index.
    /// * `schema` - Schema definition. Required (or optional) only when
    ///   *creating* a new index; must be omitted when reopening an existing
    ///   one. If omitted for both an in-memory index and a brand-new
    ///   file-backed one, an empty schema is used.
    /// * `wal_sync_policy` - Optional [`PhpWalSyncPolicy`] controlling when the
    ///   write-ahead log is forced durable. Defaults to per-record durability
    ///   ([`WalSyncPolicy::PerRecord`]) when null. Use
    ///   [`PhpWalSyncPolicy::group`] for higher-throughput group commit.
    /// * `commit_policy` - Optional [`PhpCommitPolicy`] controlling automatic
    ///   commits during ingestion. Defaults to manual (caller-driven commits)
    ///   when null. Use [`PhpCommitPolicy::every_docs`] to auto-commit every
    ///   `n` documents.
    ///
    /// # Exceptions
    ///
    /// Throws a `ValueError` if `path` points at an existing index and
    /// `schema` was also given, or if `path` contains an index in the
    /// pre-existing (pre-Issue-1059) flat layout.
    pub fn __construct(
        path: Option<String>,
        schema: Option<&PhpSchema>,
        wal_sync_policy: Option<&PhpWalSyncPolicy>,
        commit_policy: Option<&PhpCommitPolicy>,
    ) -> PhpResult<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| ext_php_rs::exception::PhpException::default(e.to_string()))?;

        let schema_val = schema.map(|php_schema| php_schema.inner.borrow().clone());
        let (schema_val, storage) = resolve_storage_and_schema(path.as_deref(), schema_val)?;

        let mut builder = Engine::builder(storage, schema_val);
        if let Some(policy) = wal_sync_policy {
            builder = builder.wal_sync_policy(policy.inner);
        }
        if let Some(policy) = commit_policy {
            builder = builder.commit_policy(policy.inner);
        }

        let engine = rt.block_on(builder.build()).map_err(laurus_err)?;

        Ok(Self {
            engine: RefCell::new(Some(Arc::new(engine))),
            rt: Arc::new(rt),
        })
    }

    /// Release this index's handle on the underlying engine, deterministically
    /// dropping its storage lock (Issue #1086/#1097) instead of waiting on
    /// PHP's garbage collector's non-deterministic timing.
    ///
    /// Idempotent: calling `close()` more than once is a no-op. Every other
    /// method throws after `close()` has been called.
    pub fn close(&self) {
        self.engine.borrow_mut().take();
    }

    // ── Document CRUD ─────────────────────────────────────────────────────

    /// Index a document, replacing any existing document with the same id.
    ///
    /// Call `commit()` to make the change visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier (string).
    /// * `doc` - An associative array mapping field names to values.
    pub fn put_document(&self, id: String, doc: &ZendHashTable) -> PhpResult<()> {
        let document = hashtable_to_document(doc)?;
        let engine = self.engine()?;
        self.rt
            .block_on(engine.put_document(&id, document))
            .map_err(laurus_err)
    }

    /// Append a document version without removing existing versions.
    ///
    /// Laurus supports multiple versions of the same id (chunk-per-document
    /// RAG pattern). Use `putDocument()` to replace.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    /// * `doc` - An associative array mapping field names to values.
    pub fn add_document(&self, id: String, doc: &ZendHashTable) -> PhpResult<()> {
        let document = hashtable_to_document(doc)?;
        let engine = self.engine()?;
        self.rt
            .block_on(engine.add_document(&id, document))
            .map_err(laurus_err)
    }

    /// Index many documents in one call, replacing existing documents by id.
    ///
    /// Batched form of `putDocument()`: the `[id, doc]` pairs are applied
    /// sequentially, in order, with one WAL fsync for the whole batch.
    /// Duplicate ids within one batch deduplicate exactly like the same puts
    /// issued one by one (the last occurrence wins). Fails fast at the first
    /// document that cannot be indexed; documents applied before the failure
    /// are **not** rolled back (retrying the batch is idempotent).
    ///
    /// # Arguments
    ///
    /// * `docs` - An array of `[id, doc]` pairs.
    pub fn put_documents(&self, docs: &Zval) -> PhpResult<()> {
        let batch = pairs_to_documents(docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        let engine = self.engine()?;
        self.rt
            .block_on(engine.put_documents(batch))
            .map_err(laurus_err)
    }

    /// Append many document versions in one call, without removing existing
    /// versions.
    ///
    /// Batched form of `addDocument()`. Ordering, single-fsync durability, and
    /// fail-fast error semantics match `putDocuments()`, but repeated ids
    /// accumulate as separate versions instead of deduplicating.
    ///
    /// # Arguments
    ///
    /// * `docs` - An array of `[id, doc]` pairs.
    pub fn add_documents(&self, docs: &Zval) -> PhpResult<()> {
        let batch = pairs_to_documents(docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        let engine = self.engine()?;
        self.rt
            .block_on(engine.add_documents(batch))
            .map_err(laurus_err)
    }

    /// Retrieve all document versions stored under `id`.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    ///
    /// # Returns
    ///
    /// An array of associative arrays, one per indexed version.
    pub fn get_documents(&self, id: String) -> PhpResult<Zval> {
        let engine = self.engine()?;
        let docs = self
            .rt
            .block_on(engine.get_documents(&id))
            .map_err(laurus_err)?;
        let mut arr = ZendHashTable::new();
        for (i, doc) in docs.iter().enumerate() {
            let ht = document_to_hashtable(doc)?;
            let mut zv = Zval::new();
            zv.set_hashtable(ht);
            arr.insert_at_index(i as i64, zv)
                .map_err(|_| "failed to insert document")?;
        }
        let mut result = Zval::new();
        result.set_hashtable(arr);
        Ok(result)
    }

    /// Delete all document versions stored under `id`.
    ///
    /// Call `commit()` to make the deletion visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    pub fn delete_documents(&self, id: String) -> PhpResult<()> {
        let engine = self.engine()?;
        self.rt
            .block_on(engine.delete_documents(&id))
            .map_err(laurus_err)
    }

    /// Flush buffered writes and make all pending changes searchable.
    pub fn commit(&self) -> PhpResult<()> {
        let engine = self.engine()?;
        self.rt.block_on(engine.commit()).map_err(laurus_err)
    }

    /// Force the write-ahead log (WAL) durable without materializing the index.
    ///
    /// This synchronously fsyncs any appended-but-unsynced records and returns
    /// once they are on stable storage. It runs directly on the calling thread
    /// (the underlying `Engine::flush_wal` is synchronous), so unlike
    /// `commit()` it does not block on the tokio runtime.
    ///
    /// # Durability trade-off
    ///
    /// This matters only under a group-commit [`PhpWalSyncPolicy`]
    /// ([`PhpWalSyncPolicy::group`]), where individual appends are batched and
    /// not yet durable. Under the default per-record policy
    /// ([`PhpWalSyncPolicy::per_record`]) every append is already durable, so
    /// this call is a cheap no-op-ish flush.
    ///
    /// With group commit a crash can lose the most recent unsynced batch of
    /// appends. Call `flushWal()` to bound that window on demand without paying
    /// the cost of a full `commit()` (which also fsyncs the WAL but additionally
    /// materializes the in-memory index state). Use `flushWal()` when you want
    /// the WAL durable but do not yet need the new documents to be searchable.
    pub fn flush_wal(&self) -> PhpResult<()> {
        self.engine()?.flush_wal().map_err(laurus_err)
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Search the index and return an array of `SearchResult` objects.
    ///
    /// `$query` may be:
    ///   - A **DSL string** (e.g. `"title:hello"`)
    ///   - A **lexical query** object (`TermQuery`, `BooleanQuery`, etc.)
    ///   - A **vector query** object (`VectorQuery`, `VectorTextQuery`)
    ///   - A **`SearchRequest`** for full control
    ///
    /// # Arguments
    ///
    /// * `query` - The query to execute.
    /// * `limit` - Maximum number of results (default: 10).
    /// * `offset` - Pagination offset (default: 0).
    /// * `highlight` - Request highlighted fragments per field (Issue
    ///   #1134): a list of field names or an associative array adding
    ///   `HighlightConfig` knobs. Highlighting follows this query, and only
    ///   `stored: true` text fields can be highlighted.
    ///
    /// # Returns
    ///
    /// An array of `SearchResult` objects.
    #[php(defaults(limit = 10, offset = 0))]
    pub fn search(
        &self,
        query: &Zval,
        limit: i64,
        offset: i64,
        highlight: Option<&ZendHashTable>,
    ) -> PhpResult<Vec<PhpSearchResult>> {
        let highlight = parse_highlight_option(highlight)?;
        let request =
            build_request_from_php(query, limit as usize, offset as usize, highlight.as_ref())?;

        let engine = self.engine()?;
        let results = self
            .rt
            .block_on(engine.search(request))
            .map_err(laurus_err)?;

        Ok(results.into_iter().map(to_php_search_result).collect())
    }

    /// Execute multiple independent searches in one call.
    ///
    /// Each entry of `queries` is dispatched in parallel on the
    /// underlying tokio runtime via `laurus::Engine::search_batch`.
    /// The same `limit` and `offset` are applied to every query in the
    /// batch. Each entry accepts the same kinds of values as `search`:
    /// a DSL string, a lexical / vector query object, or a
    /// `SearchRequest`.
    ///
    /// # Arguments
    ///
    /// * `queries` - An array of queries to execute.
    /// * `limit` - Maximum number of results per query (default: 10).
    /// * `offset` - Pagination offset per query (default: 0).
    /// * `highlight` - Request highlighted fragments per field, applied
    ///   identically to every query in the batch — same format as
    ///   `search`'s `$highlight` (Issue #1134).
    ///
    /// # Returns
    ///
    /// An array of arrays: `results[i]` is the result array for
    /// `queries[i]`. Empty input returns an empty array without
    /// invoking the engine.
    ///
    /// Issue [#720](https://github.com/mosuka/laurus/issues/720)
    /// Phase 3e of [#648](https://github.com/mosuka/laurus/issues/648).
    #[php(defaults(limit = 10, offset = 0))]
    pub fn search_batch(
        &self,
        queries: &Zval,
        limit: i64,
        offset: i64,
        highlight: Option<&ZendHashTable>,
    ) -> PhpResult<Vec<Vec<PhpSearchResult>>> {
        let highlight = parse_highlight_option(highlight)?;
        let arr = queries.array().ok_or_else(|| {
            PhpException::from(
                "search_batch: expected an array of queries (DSL string, Query object, or SearchRequest)".to_string(),
            )
        })?;

        if arr.is_empty() {
            return Ok(Vec::new());
        }

        let mut requests = Vec::with_capacity(arr.len());
        for (_, value) in arr.iter() {
            requests.push(build_request_from_php(
                value,
                limit as usize,
                offset as usize,
                highlight.as_ref(),
            )?);
        }

        let engine = self.engine()?;
        let batch_results = self
            .rt
            .block_on(engine.search_batch(requests))
            .map_err(laurus_err)?;

        Ok(batch_results
            .into_iter()
            .map(|per_query_results| {
                per_query_results
                    .into_iter()
                    .map(to_php_search_result)
                    .collect()
            })
            .collect())
    }

    // ── Schema & stats ────────────────────────────────────────────────────

    /// Return index statistics as an associative array.
    ///
    /// # Returns
    ///
    /// An associative array with keys:
    ///   - `"documentCount"` (int): total indexed documents.
    ///   - `"vectorFields"` (array): per-field vector statistics.
    pub fn stats(&self) -> PhpResult<Zval> {
        let engine = self.engine()?;
        let stats: EngineStats = self
            .rt
            .block_on(async { engine.stats() })
            .map_err(laurus_err)?;

        let mut ht = ZendHashTable::new();
        let mut count_zv = Zval::new();
        count_zv.set_long(stats.document_count as i64);
        ht.insert("documentCount", count_zv)
            .map_err(|_| "failed to insert documentCount")?;

        let mut vf_ht = ZendHashTable::new();
        for (field, field_stats) in &stats.vector_fields {
            let mut fd_ht = ZendHashTable::new();
            let mut count_zv = Zval::new();
            count_zv.set_long(field_stats.vector_count as i64);
            fd_ht
                .insert("count", count_zv)
                .map_err(|_| "failed to insert count")?;
            let mut dim_zv = Zval::new();
            dim_zv.set_long(field_stats.dimension as i64);
            fd_ht
                .insert("dimension", dim_zv)
                .map_err(|_| "failed to insert dimension")?;
            let mut fd_zv = Zval::new();
            fd_zv.set_hashtable(fd_ht);
            vf_ht
                .insert(field.as_str(), fd_zv)
                .map_err(|_| "failed to insert vector field")?;
        }
        let mut vf_zv = Zval::new();
        vf_zv.set_hashtable(vf_ht);
        ht.insert("vectorFields", vf_zv)
            .map_err(|_| "failed to insert vectorFields")?;

        let mut result = Zval::new();
        result.set_hashtable(ht);
        Ok(result)
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        "Index()".to_string()
    }
}

impl PhpIndex {
    /// Return a clone of the underlying engine handle, or a `closed_err`
    /// exception if [`Self::close`] has already been called.
    fn engine(&self) -> Result<Arc<Engine>, PhpException> {
        self.engine.borrow().clone().ok_or_else(closed_err)
    }
}

// ---------------------------------------------------------------------------
// Storage factory helper
// ---------------------------------------------------------------------------

/// Convert an array of `[id, doc]` pairs into the engine's
/// `(String, Document)` batch, raising an exception that names the offending
/// position on any entry that is not a two-element `[string, array]` pair.
///
/// # Arguments
///
/// * `docs` - A PHP array of `[id, doc]` pairs.
fn pairs_to_documents(docs: &Zval) -> PhpResult<Vec<(String, laurus::Document)>> {
    let arr = docs
        .array()
        .ok_or_else(|| PhpException::from("expected an array of [id, doc] pairs".to_string()))?;

    let mut batch = Vec::with_capacity(arr.len());
    for (index, (_, pair_zv)) in arr.iter().enumerate() {
        let pair = pair_zv.array().ok_or_else(|| {
            PhpException::from(format!("documents[{index}]: expected an [id, doc] pair"))
        })?;
        let values: Vec<&Zval> = pair.iter().map(|(_, v)| v).collect();
        if values.len() != 2 {
            return Err(PhpException::from(format!(
                "documents[{index}]: expected a 2-element [id, doc] pair"
            )));
        }
        let id = String::from_zval(values[0]).ok_or_else(|| {
            PhpException::from(format!("documents[{index}]: id must be a string"))
        })?;
        let doc = values[1].array().ok_or_else(|| {
            PhpException::from(format!("documents[{index}]: document must be an array"))
        })?;
        batch.push((id, hashtable_to_document(doc)?));
    }
    Ok(batch)
}

/// Read the persisted commit generation for `path` directly from disk,
/// without building an `Engine` -- no storage lock, no WAL recovery, no
/// embedder loading (Issue #1101).
///
/// Unlike `Index::stats()`, this is not tied to any `Index` instance: it
/// works even when no `Index` for `path` has ever been constructed in this
/// process, which is the point -- it lets a caller cheaply decide whether
/// opening the index is worth doing at all.
///
/// # Arguments
///
/// * `path` - Directory path of a file-backed index (the same value passed
///   as the first argument to `new Index(...)`).
///
/// # Returns
///
/// `0` if the index exists but nothing has been committed yet.
///
/// # Exceptions
///
/// Throws a `ValueError` if `path` has no persisted schema at all (not a
/// laurus index directory).
#[php_function]
#[php(name = "Laurus\\peek_commit_generation")]
pub fn peek_commit_generation(path: String) -> PhpResult<u64> {
    laurus::index_dir::peek_commit_generation(Path::new(&path)).map_err(index_dir_err)
}

/// Build the PHP function entry for [`peek_commit_generation`], for
/// `lib.rs`'s `#[php_module]` registration.
///
/// `wrap_function!` expands to a reference to a `#[doc(hidden)]` struct
/// that `#[php_function]` generates in *this* module, so it must be called
/// from here rather than from `lib.rs` directly.
pub fn peek_commit_generation_function_entry() -> ext_php_rs::builders::FunctionBuilder<'static> {
    wrap_function!(peek_commit_generation)
}

/// Resolve the `(Schema, Storage)` pair for [`PhpIndex::__construct`].
///
/// `path=None` keeps the pre-existing in-memory behavior (schema defaults
/// to empty, no persistence, no conflict checking). `path=Some(p)` defers
/// to [`laurus::index_dir::open_or_create`], which applies the
/// `<p>/schema.toml` + `<p>/store/` convention shared with `laurus-cli`.
///
/// # Arguments
///
/// * `path` - Optional directory path. `None` means in-memory storage.
/// * `schema` - Schema explicitly passed by the caller, if any, prior to
///   defaulting. Must stay `None` (rather than a defaulted `Schema`) so
///   `open_or_create` can distinguish "no schema passed" from "schema
///   explicitly passed" for its reopen-conflict check.
///
/// # Returns
///
/// The resolved `(Schema, Storage)` pair for the engine builder.
fn resolve_storage_and_schema(
    path: Option<&str>,
    schema: Option<laurus::Schema>,
) -> PhpResult<(laurus::Schema, Arc<dyn Storage>)> {
    match path {
        None => {
            let storage = StorageFactory::create(StorageConfig::Memory(Default::default()))
                .map_err(laurus_err)?;
            Ok((schema.unwrap_or_default(), storage))
        }
        Some(p) => laurus::index_dir::open_or_create(Path::new(p), schema).map_err(index_dir_err),
    }
}
