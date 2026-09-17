//! Ruby-facing `Index` class — the primary entry point for the laurus binding.

use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::commit_policy::RbCommitPolicy;
use crate::convert::{document_to_hash, hash_to_document};
use crate::errors::{closed_err, index_dir_err, laurus_err};
use crate::gvl::without_gvl;
use crate::schema::RbSchema;
use crate::search::{build_request_from_rb, rb_to_highlight_options, to_rb_search_result};
use crate::wal::RbWalSyncPolicy;
use laurus::{Engine, EngineStats, Schema, Storage, StorageConfig, StorageFactory};
use magnus::prelude::*;
use magnus::scan_args::{get_kwargs, scan_args};
use magnus::{Error, RArray, RHash, RModule, Ruby, TryConvert, Value};

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Laurus search index — the main entry point for the Ruby binding
/// (`Laurus::Index`).
///
/// # Creating an index
///
/// ```ruby
/// require "laurus"
///
/// # In-memory (ephemeral)
/// index = Laurus::Index.new
///
/// # File-based (persistent)
/// schema = Laurus::Schema.new
/// schema.add_text_field("title")
/// index = Laurus::Index.new(path: "./myindex", schema: schema)
/// ```
///
/// # Searching
///
/// ```ruby
/// results = index.search("title:hello", limit: 10)
/// results = index.search(Laurus::TermQuery.new("body", "rust"), limit: 5)
/// ```
#[magnus::wrap(class = "Laurus::Index")]
pub struct RbIndex {
    engine: RwLock<Option<Arc<Engine>>>,
    rt: Arc<tokio::runtime::Runtime>,
}

// Every engine call below releases the GVL via `without_gvl`, so two Ruby
// threads can now hold `&RbIndex` at the same time. magnus only requires
// `Send` for a `#[magnus::wrap]` type (`Sync` is checked only for the
// opt-in `frozen_shareable` flag, which this type does not use), so assert
// `Sync` here: a future field that isn't `Sync` must not silently break
// that. `RwLock` (unlike the `RefCell` this replaces) is fine under
// concurrent access from multiple threads.
const _: () = {
    const fn assert_sync<T: Sync>() {}
    assert_sync::<RbIndex>();
};

impl RbIndex {
    /// Create a new index, or reopen an existing one.
    ///
    /// When `path:` is given, the directory follows the same
    /// `<path>/schema.toml` + `<path>/store/` layout `laurus-cli create
    /// index`/`--index-dir` uses, so an index built here can be opened by
    /// the CLI (and vice versa) without any path juggling.
    ///
    /// * If `<path>/schema.toml` does not yet exist, this **creates** a new
    ///   index: the given `schema:` (or an empty one, if omitted) is
    ///   persisted to `<path>/schema.toml`.
    /// * If `<path>/schema.toml` already exists, this **reopens** the
    ///   index: `schema:` must be omitted (`nil`) — the persisted schema is
    ///   loaded instead. Passing an explicit `schema:` here raises
    ///   `ArgumentError`, since it would be ambiguous which one should win.
    ///
    /// # Arguments
    ///
    /// * `args` - Keyword arguments:
    ///   - `path:` (String, optional): Directory path for persistent storage.
    ///     Pass `nil` (default) for an ephemeral in-memory index.
    ///   - `schema:` (Schema, optional): Schema definition. Required (or
    ///     optional) only when *creating* a new index; must be omitted when
    ///     reopening an existing one. If omitted for both an in-memory index
    ///     and a brand-new file-backed one, an empty schema is used.
    ///   - `wal_sync_policy:` (WalSyncPolicy, optional): WAL durability policy.
    ///     Defaults to per-record fsync (highest durability). Pass
    ///     `Laurus::WalSyncPolicy.group(...)` to enable group commit for higher
    ///     write throughput; use `flush_wal` to force durability on demand.
    ///   - `commit_policy:` (CommitPolicy, optional): auto-commit policy.
    ///     Defaults to manual (the caller drives every commit). Pass
    ///     `Laurus::CommitPolicy.every_docs(n)` to auto-commit every `n`
    ///     applied documents.
    ///
    /// # Errors
    ///
    /// Raises a Ruby `ArgumentError` if `path:` points at an existing index
    /// and `schema:` was also given, or if `path:` contains an index in the
    /// pre-existing (pre-Issue-1059) flat layout.
    fn new(args: &[Value]) -> Result<Self, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let args = scan_args::<(), (), (), (), RHash, ()>(args)?;
        let kwargs = get_kwargs::<
            _,
            (),
            (
                Option<Option<String>>,
                Option<Option<&RbSchema>>,
                Option<Option<&RbWalSyncPolicy>>,
                Option<Option<&RbCommitPolicy>>,
            ),
            (),
        >(
            args.keywords,
            &[],
            &["path", "schema", "wal_sync_policy", "commit_policy"],
        )?;
        let (path, schema, wal_sync_policy, commit_policy) = kwargs.optional;
        let path = path.flatten();
        let schema_ref = schema.flatten();
        let wal_sync_policy = wal_sync_policy.flatten();
        let commit_policy = commit_policy.flatten();

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| Error::new(ruby.exception_runtime_error(), e.to_string()))?;
        let rt = Arc::new(rt);

        let schema = schema_ref.map(|s| s.inner.borrow().clone());
        // Returns `magnus::Error` (holds a `Value`, so it's `!Send`); must
        // run with the GVL held, before entering `without_gvl` below.
        let (schema, storage) = resolve_storage_and_schema(path.as_deref(), schema)?;
        let wal_sync_policy = wal_sync_policy.map(|p| p.inner);
        let commit_policy = commit_policy.map(|p| p.inner);

        let rt_for_build = rt.clone();
        let engine = without_gvl(move || -> Result<Engine, laurus::LaurusError> {
            let mut builder = Engine::builder(storage, schema);
            if let Some(policy) = wal_sync_policy {
                builder = builder.wal_sync_policy(policy);
            }
            if let Some(policy) = commit_policy {
                builder = builder.commit_policy(policy);
            }
            rt_for_build.block_on(builder.build())
        })
        .map_err(laurus_err)?;

        Ok(Self {
            engine: RwLock::new(Some(Arc::new(engine))),
            rt,
        })
    }

    /// Return a clone of the underlying engine handle, or a `RuntimeError`
    /// if `close` has already been called.
    fn engine(&self) -> Result<Arc<Engine>, Error> {
        self.engine.read().unwrap().clone().ok_or_else(closed_err)
    }

    /// Release this index's handle on the underlying engine, deterministically
    /// dropping its storage lock (Issue #1086/#1097) instead of waiting on
    /// Ruby's garbage collector's non-deterministic timing.
    ///
    /// Idempotent: calling `close` more than once is a no-op. Every other
    /// method raises `RuntimeError` after `close` has been called.
    ///
    /// # Concurrent calls (Issue #1103)
    ///
    /// Unlike the Python binding's `close()`/`reload()`, this method takes
    /// `&self`, not an exclusive borrow -- magnus only ever hands out `&self`
    /// to `#[magnus::wrap]` methods, so there is no borrow-checker mechanism
    /// here to make `close()` wait for an in-flight call on another thread.
    /// If some other thread still holds a cloned `Arc<Engine>` (returned by
    /// [`Self::engine`]) when `close()` runs, this only drops *this* handle's
    /// reference: the storage lock is released once every in-flight call
    /// finishes and drops its own reference, not necessarily by the time
    /// `close()` returns.
    fn close(&self) {
        let engine = self.engine.write().unwrap().take();
        // Dropping the last `Arc<Engine>` releases the storage lock (Issue
        // #1086/#1097), which is blocking I/O -- do it with the GVL
        // released.
        without_gvl(|| drop(engine));
    }

    // ── Document CRUD ─────────────────────────────────────────────────────

    /// Index a document, replacing any existing document with the same id.
    ///
    /// Call `commit` to make the change visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier (String).
    /// * `doc` - A Hash mapping field names to values.
    fn put_document(&self, id: String, doc: RHash) -> Result<(), Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let document = hash_to_document(&ruby, doc)?;
        let engine = self.engine()?;
        without_gvl(|| self.rt.block_on(engine.put_document(&id, document))).map_err(laurus_err)
    }

    /// Append a document version without removing existing versions.
    ///
    /// Laurus supports multiple versions of the same id (chunk-per-document
    /// RAG pattern). Use `put_document` to replace.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    /// * `doc` - A Hash mapping field names to values.
    fn add_document(&self, id: String, doc: RHash) -> Result<(), Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let document = hash_to_document(&ruby, doc)?;
        let engine = self.engine()?;
        without_gvl(|| self.rt.block_on(engine.add_document(&id, document))).map_err(laurus_err)
    }

    /// Index many documents in one call, replacing existing documents by id.
    ///
    /// Batched form of `put_document`: the `[id, hash]` pairs are applied
    /// sequentially, in order, with one WAL fsync for the whole batch.
    /// Duplicate ids within one batch deduplicate exactly like the same puts
    /// issued one by one (the last occurrence wins). Fails fast at the first
    /// document that cannot be indexed; documents applied before the failure
    /// are **not** rolled back (retrying the batch is idempotent).
    ///
    /// # Arguments
    ///
    /// * `docs` - An Array of `[id, hash]` pairs.
    fn put_documents(&self, docs: RArray) -> Result<(), Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let batch = pairs_to_documents(&ruby, docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        let engine = self.engine()?;
        without_gvl(|| self.rt.block_on(engine.put_documents(batch))).map_err(laurus_err)
    }

    /// Append many document versions in one call, without removing existing
    /// versions.
    ///
    /// Batched form of `add_document`. Ordering, single-fsync durability, and
    /// fail-fast error semantics match `put_documents`, but repeated ids
    /// accumulate as separate versions instead of deduplicating.
    ///
    /// # Arguments
    ///
    /// * `docs` - An Array of `[id, hash]` pairs.
    fn add_documents(&self, docs: RArray) -> Result<(), Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let batch = pairs_to_documents(&ruby, docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        let engine = self.engine()?;
        without_gvl(|| self.rt.block_on(engine.add_documents(batch))).map_err(laurus_err)
    }

    /// Retrieve all document versions stored under `id`.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    ///
    /// # Returns
    ///
    /// An Array of Hashes, one per indexed version.
    fn get_documents(&self, id: String) -> Result<RArray, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let engine = self.engine()?;
        let docs =
            without_gvl(|| self.rt.block_on(engine.get_documents(&id))).map_err(laurus_err)?;
        let arr = ruby.ary_new_capa(docs.len());
        for doc in &docs {
            let hash = document_to_hash(&ruby, doc)?;
            arr.push(hash)?;
        }
        Ok(arr)
    }

    /// Delete all document versions stored under `id`.
    ///
    /// Call `commit` to make the deletion visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    fn delete_documents(&self, id: String) -> Result<(), Error> {
        let engine = self.engine()?;
        without_gvl(|| self.rt.block_on(engine.delete_documents(&id))).map_err(laurus_err)
    }

    /// Flush buffered writes and make all pending changes searchable.
    fn commit(&self) -> Result<(), Error> {
        let engine = self.engine()?;
        without_gvl(|| self.rt.block_on(engine.commit())).map_err(laurus_err)
    }

    /// Force any buffered Write-Ahead Log (WAL) appends to durable storage.
    ///
    /// This is an on-demand durability barrier and runs synchronously (no async
    /// runtime is involved). Its effect depends on the configured
    /// `wal_sync_policy`:
    ///
    /// * Under the default `Laurus::WalSyncPolicy.per_record`, every append is
    ///   already fsync'd as it happens, so this is effectively a no-op.
    /// * Under `Laurus::WalSyncPolicy.group(...)`, appends are buffered and
    ///   their fsync is deferred for throughput; `flush_wal` fsyncs the
    ///   outstanding batch immediately, guaranteeing those records survive a
    ///   crash without waiting for a threshold or the next `commit`.
    ///
    /// Unlike `commit`, this does **not** make pending documents searchable; it
    /// only guarantees their durability in the WAL. Call `commit` to publish
    /// changes to searches.
    ///
    /// # Returns
    ///
    /// `nil` on success, or raises if the underlying fsync fails.
    fn flush_wal(&self) -> Result<(), Error> {
        let engine = self.engine()?;
        without_gvl(|| engine.flush_wal()).map_err(laurus_err)
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Search the index and return an Array of `SearchResult` objects.
    ///
    /// `query` may be:
    ///   - A **DSL string** (e.g. `"title:hello"`)
    ///   - A **lexical query** object (`TermQuery`, `BooleanQuery`, etc.)
    ///   - A **vector query** object (`VectorQuery`, `VectorTextQuery`)
    ///   - A **`SearchRequest`** for full control
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `query`: The query to execute.
    ///   - `limit:` (Integer, default 10): Maximum number of results.
    ///   - `offset:` (Integer, default 0): Pagination offset.
    ///   - `highlight:` - Request highlighted fragments per field (Issue
    ///     #1134): an Array of field names or a config Hash. Highlighting
    ///     follows this query, and only `stored: true` text fields can be
    ///     highlighted.
    ///
    /// # Returns
    ///
    /// An Array of `SearchResult` objects.
    fn search(&self, args: &[Value]) -> Result<RArray, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let args = scan_args::<(Value,), (), (), (), RHash, ()>(args)?;
        let (query,) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<usize>, Option<usize>, Option<Value>), ()>(
            args.keywords,
            &[],
            &["limit", "offset", "highlight"],
        )?;
        let (limit, offset, highlight) = kwargs.optional;
        let limit = limit.unwrap_or(10);
        let offset = offset.unwrap_or(0);
        let highlight = highlight
            .map(|v| rb_to_highlight_options(&ruby, v))
            .transpose()?;

        let request = build_request_from_rb(query, limit, offset, highlight.as_ref())?;

        let engine = self.engine()?;
        let results =
            without_gvl(|| self.rt.block_on(engine.search(request))).map_err(laurus_err)?;

        let arr = ruby.ary_new_capa(results.len());
        for r in results {
            let rb_result = to_rb_search_result(r);
            arr.push(ruby.into_value(rb_result))?;
        }
        Ok(arr)
    }

    /// Execute multiple independent searches in one call.
    ///
    /// Each entry in `queries` is dispatched in parallel on the
    /// underlying tokio runtime via `laurus::Engine::search_batch`.
    /// The same `limit:` and `offset:` keyword arguments apply to every
    /// query in the batch. Each entry accepts the same kinds of values
    /// as `search`: a DSL string, a lexical / vector query object, or
    /// a `SearchRequest`.
    ///
    /// # Arguments
    ///
    /// * `args` - Positional and keyword arguments:
    ///   - `queries`: An Array of queries to execute.
    ///   - `limit:` (Integer, default 10): Maximum number of results per query.
    ///   - `offset:` (Integer, default 0): Pagination offset per query.
    ///   - `highlight:` - Request highlighted fragments per field, applied
    ///     identically to every query in the batch — same format as
    ///     `search`'s `highlight:` (Issue #1134).
    ///
    /// # Returns
    ///
    /// An Array of Arrays: `results[i]` is the result Array for
    /// `queries[i]`. Empty input returns an empty Array without
    /// invoking the engine.
    ///
    /// Issue [#719](https://github.com/mosuka/laurus/issues/719)
    /// Phase 3d of [#648](https://github.com/mosuka/laurus/issues/648).
    fn search_batch(&self, args: &[Value]) -> Result<RArray, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let args = scan_args::<(Value,), (), (), (), RHash, ()>(args)?;
        let (queries,) = args.required;
        let kwargs = get_kwargs::<_, (), (Option<usize>, Option<usize>, Option<Value>), ()>(
            args.keywords,
            &[],
            &["limit", "offset", "highlight"],
        )?;
        let (limit, offset, highlight) = kwargs.optional;
        let limit = limit.unwrap_or(10);
        let offset = offset.unwrap_or(0);
        let highlight = highlight
            .map(|v| rb_to_highlight_options(&ruby, v))
            .transpose()?;

        let queries_array: RArray = TryConvert::try_convert(queries).map_err(|_| {
            Error::new(
                ruby.exception_arg_error(),
                "search_batch: expected an Array of queries (DSL string, Query object, or SearchRequest)",
            )
        })?;

        if queries_array.is_empty() {
            return Ok(ruby.ary_new());
        }

        let mut requests = Vec::with_capacity(queries_array.len());
        for item in queries_array.into_iter() {
            requests.push(build_request_from_rb(
                item,
                limit,
                offset,
                highlight.as_ref(),
            )?);
        }

        let engine = self.engine()?;
        let batch_results =
            without_gvl(|| self.rt.block_on(engine.search_batch(requests))).map_err(laurus_err)?;

        let outer = ruby.ary_new_capa(batch_results.len());
        for per_query_results in batch_results {
            let inner = ruby.ary_new_capa(per_query_results.len());
            for r in per_query_results {
                let rb_result = to_rb_search_result(r);
                inner.push(ruby.into_value(rb_result))?;
            }
            outer.push(inner)?;
        }
        Ok(outer)
    }

    // ── Schema & stats ────────────────────────────────────────────────────

    /// Return index statistics.
    ///
    /// # Returns
    ///
    /// A Hash with keys:
    ///   - `"document_count"` (Integer): total indexed documents.
    ///   - `"vector_fields"` (Hash): per-field vector statistics.
    fn stats(&self) -> Result<RHash, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let engine = self.engine()?;
        let stats: EngineStats = without_gvl(|| engine.stats()).map_err(laurus_err)?;
        let dict = ruby.hash_new();
        dict.aset(ruby.str_new("document_count"), stats.document_count)?;
        let vf = ruby.hash_new();
        for (field, field_stats) in &stats.vector_fields {
            let fd = ruby.hash_new();
            fd.aset(ruby.str_new("count"), field_stats.vector_count)?;
            fd.aset(ruby.str_new("dimension"), field_stats.dimension)?;
            vf.aset(ruby.str_new(field), fd)?;
        }
        dict.aset(ruby.str_new("vector_fields"), vf)?;
        Ok(dict)
    }

    fn inspect(&self) -> String {
        "Index()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Batch-ingestion helper
// ---------------------------------------------------------------------------

/// Convert an Array of `[id, hash]` pairs into the engine's
/// `(String, Document)` batch, raising an ArgumentError that names the
/// offending position on any entry that is not a two-element `[String, Hash]`
/// pair.
///
/// # Arguments
///
/// * `ruby` - The current Ruby interpreter handle.
/// * `docs` - An Array of `[id, hash]` pairs.
fn pairs_to_documents(ruby: &Ruby, docs: RArray) -> Result<Vec<(String, laurus::Document)>, Error> {
    let mut batch = Vec::with_capacity(docs.len());
    for (index, item) in docs.into_iter().enumerate() {
        let pair: RArray = TryConvert::try_convert(item).map_err(|_| {
            Error::new(
                ruby.exception_arg_error(),
                format!("documents[{index}]: expected an [id, hash] pair"),
            )
        })?;
        if pair.len() != 2 {
            return Err(Error::new(
                ruby.exception_arg_error(),
                format!("documents[{index}]: expected a 2-element [id, hash] pair"),
            ));
        }
        let id: String = TryConvert::try_convert(pair.entry(0)?).map_err(|_| {
            Error::new(
                ruby.exception_arg_error(),
                format!("documents[{index}]: id must be a String"),
            )
        })?;
        let hash: RHash = TryConvert::try_convert(pair.entry(1)?).map_err(|_| {
            Error::new(
                ruby.exception_arg_error(),
                format!("documents[{index}]: document must be a Hash"),
            )
        })?;
        batch.push((id, hash_to_document(ruby, hash)?));
    }
    Ok(batch)
}

// ---------------------------------------------------------------------------
// Storage factory helper
// ---------------------------------------------------------------------------

/// Resolve the `(Schema, Storage)` pair for [`RbIndex::new`].
///
/// `path: nil` keeps the pre-existing in-memory behavior (schema defaults
/// to empty, no persistence, no conflict checking). `path: Some(p)` defers
/// to `laurus::index_dir::open_or_create`, which applies the
/// `<p>/schema.toml` + `<p>/store/` convention shared with `laurus-cli`.
///
/// # Arguments
///
/// * `path` - Optional directory path. `None` means in-memory storage.
/// * `schema` - Optional schema. Only meaningful when creating a new
///   file-backed index; see `laurus::index_dir::open_or_create` for the
///   reopen-conflict rule.
///
/// # Returns
///
/// The resolved `Schema` and an `Arc<dyn Storage>` for the engine.
fn resolve_storage_and_schema(
    path: Option<&str>,
    schema: Option<Schema>,
) -> Result<(Schema, Arc<dyn Storage>), Error> {
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
/// Unlike `Index#stats`, this is not tied to any `Index` instance: it works
/// even when no `Index` for `path` has ever been constructed in this
/// process, which is the point -- it lets a caller cheaply decide whether
/// opening the index is worth doing at all.
///
/// # Arguments
///
/// * `path` - Directory path of a file-backed index (the same value passed
///   as `path:` to `Laurus::Index.new`).
///
/// # Returns
///
/// `0` if the index exists but nothing has been committed yet.
///
/// # Errors
///
/// Raises a Ruby `ArgumentError` if `path` has no persisted schema at all
/// (not a laurus index directory).
fn peek_commit_generation(path: String) -> Result<u64, Error> {
    laurus::index_dir::peek_commit_generation(Path::new(&path)).map_err(index_dir_err)
}

// ---------------------------------------------------------------------------
// Class registration
// ---------------------------------------------------------------------------

/// Register the `Laurus::Index` class and its methods.
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `module` - The `Laurus` module.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    let class = module.define_class("Index", ruby.class_object())?;
    class.define_singleton_method("new", magnus::function!(RbIndex::new, -1))?;
    class.define_method("put_document", magnus::method!(RbIndex::put_document, 2))?;
    class.define_method("add_document", magnus::method!(RbIndex::add_document, 2))?;
    class.define_method("put_documents", magnus::method!(RbIndex::put_documents, 1))?;
    class.define_method("add_documents", magnus::method!(RbIndex::add_documents, 1))?;
    class.define_method("get_documents", magnus::method!(RbIndex::get_documents, 1))?;
    class.define_method(
        "delete_documents",
        magnus::method!(RbIndex::delete_documents, 1),
    )?;
    class.define_method("commit", magnus::method!(RbIndex::commit, 0))?;
    class.define_method("flush_wal", magnus::method!(RbIndex::flush_wal, 0))?;
    class.define_method("search", magnus::method!(RbIndex::search, -1))?;
    class.define_method("search_batch", magnus::method!(RbIndex::search_batch, -1))?;
    class.define_method("stats", magnus::method!(RbIndex::stats, 0))?;
    class.define_method("close", magnus::method!(RbIndex::close, 0))?;
    class.define_method("inspect", magnus::method!(RbIndex::inspect, 0))?;
    class.define_method("to_s", magnus::method!(RbIndex::inspect, 0))?;
    module.define_module_function(
        "peek_commit_generation",
        magnus::function!(peek_commit_generation, 1),
    )?;
    Ok(())
}
