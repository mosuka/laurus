//! Node.js-facing [`Index`] class — the primary entry point for the laurus binding.

use std::path::Path;
use std::sync::Arc;

use crate::commit::JsCommitPolicy;
use crate::convert::{data_value_to_json, json_to_document};
use crate::errors::{closed_err, index_dir_err, laurus_err};
use crate::query::{JsQuery, JsTermQuery, JsVectorQuery, JsVectorQueryInner, JsVectorTextQuery};
use crate::schema::JsSchema;
use crate::search::{
    JsHighlightOptions, JsSearchRequest, JsSearchResult, build_dsl_request, build_lexical_request,
    build_vector_request, js_highlight_options_to_core, to_js_search_result,
};
use crate::wal::JsWalSyncPolicy;
use laurus::{Engine, Schema, Storage, StorageConfig, StorageFactory};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Laurus search index — the main entry point for the Node.js binding.
///
/// ## Creating an index
///
/// ```javascript
/// const { Index, Schema } = require("laurus-nodejs");
///
/// // In-memory (ephemeral, great for prototyping)
/// const index = await Index.create();
///
/// // File-based (persistent)
/// const schema = new Schema();
/// schema.addTextField("title");
/// schema.addTextField("body");
/// schema.addHnswField("embedding", 384);
/// const index = await Index.create("./myindex", schema);
/// ```
///
/// ## Adding documents
///
/// ```javascript
/// await index.putDocument("doc1", { title: "Hello", body: "World" });
/// await index.commit();
/// ```
///
/// ## Searching
///
/// ```javascript
/// // DSL string
/// const results = await index.search("title:hello");
///
/// // Term query
/// const results = await index.searchTerm("body", "rust");
///
/// // Via SearchRequest for full control
/// const req = new SearchRequest();
/// req.setVectorTextQuery("embedding", "concurrent");
/// req.setRrfFusion();
/// const results = await index.searchWithRequest(req);
/// ```
#[napi(js_name = "Index")]
pub struct JsIndex {
    engine: Option<Arc<Engine>>,
}

#[napi]
impl JsIndex {
    /// Borrow the underlying engine, or a clear "Index is closed" error
    /// once [`Self::close`] has been called.
    fn engine(&self) -> Result<&Arc<Engine>> {
        self.engine.as_ref().ok_or_else(closed_err)
    }

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
    ///   index: `schema` must be omitted — the persisted schema is loaded
    ///   instead. Passing an explicit `schema` here throws, since it would
    ///   be ambiguous which one should win.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path for persistent storage.
    ///     Pass `null` or omit for an ephemeral in-memory index.
    /// * `schema` - Schema definition. Only meaningful when *creating* a
    ///     new index; must be omitted when reopening an existing one. If
    ///     omitted for both an in-memory index and a brand-new file-backed
    ///     one, an empty schema is used.
    /// * `wal_sync_policy` - Optional WAL durability policy (see
    ///     `WalSyncPolicy`). When omitted, the default per-record policy is
    ///     used, where every append is fsync'd before it returns.
    /// * `commit_policy` - Optional auto-commit policy (see `CommitPolicy`).
    ///     When omitted, the engine is manual — the caller drives every
    ///     `commit()`.
    ///
    /// # Returns
    ///
    /// A new `Index` instance.
    ///
    /// # Errors
    ///
    /// Throws if `path` points at an existing index and `schema` was also
    /// given, or if `path` contains an index in the pre-existing
    /// (pre-Issue-1059) flat layout.
    #[napi(factory)]
    pub async fn create(
        path: Option<String>,
        schema: Option<&JsSchema>,
        wal_sync_policy: Option<&JsWalSyncPolicy>,
        commit_policy: Option<&JsCommitPolicy>,
    ) -> Result<Self> {
        let schema = schema.map(|s| s.inner.clone());
        let (schema, storage) = resolve_storage_and_schema(path.as_deref(), schema)?;

        let mut builder = Engine::builder(storage, schema);
        if let Some(policy) = wal_sync_policy {
            builder = builder.wal_sync_policy(policy.inner);
        }
        if let Some(policy) = commit_policy {
            builder = builder.commit_policy(policy.inner);
        }
        let engine = builder.build().await.map_err(laurus_err)?;

        Ok(Self {
            engine: Some(Arc::new(engine)),
        })
    }

    /// Release this index's handle on the underlying engine, deterministically
    /// dropping its storage lock (Issue #1086/#1097) instead of waiting on the
    /// JS garbage collector's non-deterministic timing.
    ///
    /// Idempotent: calling `close()` more than once is a no-op. Every other
    /// method throws after `close()` has been called.
    #[napi]
    pub fn close(&mut self) {
        self.engine = None;
    }

    // ── Document CRUD ─────────────────────────────────────────────────────

    /// Index a document, replacing any existing document with the same id.
    ///
    /// Call `commit()` to make the change visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier (string).
    /// * `doc` - An object mapping field names to values.
    #[napi]
    pub async fn put_document(&self, id: String, doc: Value) -> Result<()> {
        let document = json_to_document(&doc)?;
        self.engine()?
            .put_document(&id, document)
            .await
            .map_err(laurus_err)
    }

    /// Append a document version without removing existing versions.
    ///
    /// Laurus supports multiple versions of the same id (chunk-per-document
    /// RAG pattern). Use `putDocument` to replace.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    /// * `doc` - An object mapping field names to values.
    #[napi]
    pub async fn add_document(&self, id: String, doc: Value) -> Result<()> {
        let document = json_to_document(&doc)?;
        self.engine()?
            .add_document(&id, document)
            .await
            .map_err(laurus_err)
    }

    /// Index many documents in one call, replacing existing documents by id.
    ///
    /// Batched form of `putDocument`: the `[id, doc]` pairs are applied
    /// sequentially, in order, with one WAL fsync for the whole batch.
    /// Duplicate ids within one batch deduplicate exactly like the same puts
    /// issued one by one (the last occurrence wins). Fails fast at the first
    /// document that cannot be indexed; documents applied before the failure
    /// are **not** rolled back (retrying the batch is idempotent).
    ///
    /// # Arguments
    ///
    /// * `docs` - An array of `[id, doc]` pairs.
    #[napi]
    pub async fn put_documents(&self, docs: Vec<(String, Value)>) -> Result<()> {
        let batch = pairs_to_documents(docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        self.engine()?
            .put_documents(batch)
            .await
            .map_err(laurus_err)
    }

    /// Append many document versions in one call, without removing existing
    /// versions.
    ///
    /// Batched form of `addDocument`. Ordering, single-fsync durability, and
    /// fail-fast error semantics match `putDocuments`, but repeated ids
    /// accumulate as separate versions instead of deduplicating.
    ///
    /// # Arguments
    ///
    /// * `docs` - An array of `[id, doc]` pairs.
    #[napi]
    pub async fn add_documents(&self, docs: Vec<(String, Value)>) -> Result<()> {
        let batch = pairs_to_documents(docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        self.engine()?
            .add_documents(batch)
            .await
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
    /// A list of document objects (one per indexed version).
    #[napi]
    pub async fn get_documents(&self, id: String) -> Result<Vec<Value>> {
        let docs = self
            .engine()?
            .get_documents(&id)
            .await
            .map_err(laurus_err)?;
        Ok(docs
            .iter()
            .map(|doc| {
                let mut map = serde_json::Map::new();
                for (field, value) in &doc.fields {
                    map.insert(field.clone(), data_value_to_json(value));
                }
                Value::Object(map)
            })
            .collect())
    }

    /// Delete all document versions stored under `id`.
    ///
    /// Call `commit()` to make the deletion visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    #[napi]
    pub async fn delete_documents(&self, id: String) -> Result<()> {
        self.engine()?
            .delete_documents(&id)
            .await
            .map_err(laurus_err)
    }

    /// Flush buffered writes and make all pending changes searchable.
    #[napi]
    pub async fn commit(&self) -> Result<()> {
        self.engine()?.commit().await.map_err(laurus_err)
    }

    /// Force the write-ahead log durable on demand.
    ///
    /// Under the default `WalSyncPolicy.perRecord()` policy every append is
    /// already fsync'd, so this is a no-op fast path. Under
    /// `WalSyncPolicy.group(...)` the fsync of each append is deferred and
    /// batched for throughput, which means a crash can lose the last unsynced
    /// batch; calling `flushWal()` forces the current batch durable without the
    /// heavier work of a full `commit()` (which also materializes the index).
    /// Use it to bound the durability window of a group-commit index — for
    /// example after a logical unit of ingest — when you do not yet want the
    /// changes to become searchable.
    ///
    /// # Returns
    ///
    /// Resolves once the WAL has been fsync'd, or rejects if the flush fails.
    #[napi]
    pub async fn flush_wal(&self) -> Result<()> {
        self.engine()?.flush_wal().map_err(laurus_err)
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Search using a DSL string query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query DSL string (e.g. `"title:hello"`, `"~\"memory safety\""`).
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    /// * `highlight` - Request highlighted fragments per field (Issue
    ///   #1134). Highlighting follows this query, and only `stored: true`
    ///   text fields can be highlighted.
    ///
    /// # Returns
    ///
    /// An array of SearchResult objects.
    #[napi]
    pub async fn search(
        &self,
        query: String,
        limit: Option<u32>,
        offset: Option<u32>,
        highlight: Option<JsHighlightOptions>,
    ) -> Result<Vec<JsSearchResult>> {
        let mut request = build_dsl_request(
            query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        request.lexical_options.highlight = highlight.as_ref().map(js_highlight_options_to_core);
        let results = self.engine()?.search(request).await.map_err(laurus_err)?;
        Ok(results.into_iter().map(to_js_search_result).collect())
    }

    /// Search using a term query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field to search in.
    /// * `term` - The exact term to match.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    /// * `highlight` - Same as `search`'s `highlight` parameter (Issue #1134).
    ///
    /// # Returns
    ///
    /// An array of SearchResult objects.
    #[napi]
    pub async fn search_term(
        &self,
        field: String,
        term: String,
        limit: Option<u32>,
        offset: Option<u32>,
        highlight: Option<JsHighlightOptions>,
    ) -> Result<Vec<JsSearchResult>> {
        let query = JsQuery::TermQuery(JsTermQuery { field, term });
        let mut request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        request.lexical_options.highlight = highlight.as_ref().map(js_highlight_options_to_core);
        let results = self.engine()?.search(request).await.map_err(laurus_err)?;
        Ok(results.into_iter().map(to_js_search_result).collect())
    }

    /// Search using a pre-computed embedding vector.
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `vector` - The embedding vector as an array of numbers.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    ///
    /// # Returns
    ///
    /// An array of SearchResult objects.
    #[napi]
    pub async fn search_vector(
        &self,
        field: String,
        vector: Vec<f64>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<JsSearchResult>> {
        let query = JsVectorQuery::VectorQuery(JsVectorQueryInner {
            field,
            vector: vector.into_iter().map(|v| v as f32).collect(),
        });
        let request = build_vector_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        let results = self.engine()?.search(request).await.map_err(laurus_err)?;
        Ok(results.into_iter().map(to_js_search_result).collect())
    }

    /// Search using a text-based vector query (embedded by the registered embedder).
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `text` - The text to embed and search with.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    ///
    /// # Returns
    ///
    /// An array of SearchResult objects.
    #[napi]
    pub async fn search_vector_text(
        &self,
        field: String,
        text: String,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<Vec<JsSearchResult>> {
        let query = JsVectorQuery::VectorTextQuery(JsVectorTextQuery { field, text });
        let request = build_vector_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        let results = self.engine()?.search(request).await.map_err(laurus_err)?;
        Ok(results.into_iter().map(to_js_search_result).collect())
    }

    /// Search using a full SearchRequest for advanced control.
    ///
    /// # Arguments
    ///
    /// * `request` - A `SearchRequest` object.
    ///
    /// # Returns
    ///
    /// An array of SearchResult objects.
    #[napi]
    pub async fn search_with_request(
        &self,
        request: &JsSearchRequest,
    ) -> Result<Vec<JsSearchResult>> {
        let req = request.build()?;
        let results = self.engine()?.search(req).await.map_err(laurus_err)?;
        Ok(results.into_iter().map(to_js_search_result).collect())
    }

    /// Execute multiple independent searches in one call.
    ///
    /// Each query is dispatched in parallel on the underlying tokio
    /// runtime via `laurus::Engine::search_batch`. The same `limit`
    /// and `offset` are applied to every query in the batch.
    ///
    /// # Arguments
    ///
    /// * `queries` - An array of query DSL strings.
    /// * `limit` - Maximum number of results per query (default 10).
    /// * `offset` - Pagination offset applied to each query (default 0).
    /// * `highlight` - Request highlighted fragments per field, applied
    ///   identically to every query in the batch (Issue #1134).
    ///
    /// # Returns
    ///
    /// An array of arrays: `results[i]` is the result list for
    /// `queries[i]`. Empty input returns `[]` without invoking the
    /// engine.
    ///
    /// Issue [#718](https://github.com/mosuka/laurus/issues/718)
    /// Phase 3c of [#648](https://github.com/mosuka/laurus/issues/648).
    #[napi]
    pub async fn search_batch(
        &self,
        queries: Vec<String>,
        limit: Option<u32>,
        offset: Option<u32>,
        highlight: Option<JsHighlightOptions>,
    ) -> Result<Vec<Vec<JsSearchResult>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        let limit = limit.unwrap_or(10) as usize;
        let offset = offset.unwrap_or(0) as usize;
        let highlight = highlight.as_ref().map(js_highlight_options_to_core);
        let requests: Vec<_> = queries
            .into_iter()
            .map(|q| {
                let mut request = build_dsl_request(q, limit, offset);
                request.lexical_options.highlight = highlight.clone();
                request
            })
            .collect();

        let batch_results = self
            .engine()?
            .search_batch(requests)
            .await
            .map_err(laurus_err)?;

        Ok(batch_results
            .into_iter()
            .map(|per_query_results| {
                per_query_results
                    .into_iter()
                    .map(to_js_search_result)
                    .collect()
            })
            .collect())
    }

    // ── Stats ─────────────────────────────────────────────────────────────

    /// Return index statistics.
    ///
    /// # Returns
    ///
    /// An object with:
    ///   - `documentCount` (number): total indexed documents.
    ///   - `vectorFields` (object): per-field vector statistics with `count` and `dimension`.
    #[napi]
    pub fn stats(&self) -> Result<Value> {
        let stats = self.engine()?.stats().map_err(laurus_err)?;
        let mut vector_fields = serde_json::Map::new();
        for (field, field_stats) in &stats.vector_fields {
            vector_fields.insert(
                field.clone(),
                serde_json::json!({
                    "count": field_stats.vector_count,
                    "dimension": field_stats.dimension,
                }),
            );
        }
        Ok(serde_json::json!({
            "documentCount": stats.document_count,
            "vectorFields": vector_fields,
        }))
    }
}

// ---------------------------------------------------------------------------
// Batch-ingestion helper
// ---------------------------------------------------------------------------

/// Convert an array of `(id, doc)` pairs into the engine's
/// `(String, Document)` batch, naming the offending position on any entry
/// whose document cannot be converted.
fn pairs_to_documents(docs: Vec<(String, Value)>) -> Result<Vec<(String, laurus::Document)>> {
    docs.into_iter()
        .enumerate()
        .map(|(index, (id, doc))| {
            let document = json_to_document(&doc)
                .map_err(|e| laurus::LaurusError::other(format!("documents[{index}]: {e}")))
                .map_err(laurus_err)?;
            Ok((id, document))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Read the persisted commit generation for `path` directly from disk,
/// without building an `Engine` -- no storage lock, no WAL recovery, no
/// embedder loading (Issue #1101).
///
/// Unlike `Index.stats()`, this is not tied to any `Index` instance: it
/// works even when no `Index` for `path` has ever been constructed in this
/// process, which is the point -- it lets a caller cheaply decide whether
/// opening the index is worth doing at all.
///
/// Returned as `number` (not `bigint`), matching every other numeric value
/// this binding exposes (e.g. `stats()`'s `documentCount`); a commit
/// generation will never realistically approach 2^53.
///
/// # Arguments
///
/// * `path` - Directory path of a file-backed index (the same value passed
///   to `Index.create(path, ...)`).
///
/// # Returns
///
/// `0` if the index exists but nothing has been committed yet.
///
/// # Errors
///
/// Throws if `path` has no persisted schema at all (not a laurus index
/// directory).
#[napi]
pub fn peek_commit_generation(path: String) -> Result<f64> {
    laurus::index_dir::peek_commit_generation(Path::new(&path))
        .map(|generation| generation as f64)
        .map_err(index_dir_err)
}

// ---------------------------------------------------------------------------
// Storage factory helper
// ---------------------------------------------------------------------------

/// Resolve the `(Schema, Storage)` pair for [`JsIndex::create`].
///
/// `path=None` keeps the pre-existing in-memory behavior (schema defaults
/// to empty, no persistence, no conflict checking). `path=Some(p)` defers
/// to [`laurus::index_dir::open_or_create`], which applies the
/// `<p>/schema.toml` + `<p>/store/` convention shared with `laurus-cli`.
fn resolve_storage_and_schema(
    path: Option<&str>,
    schema: Option<Schema>,
) -> Result<(Schema, Arc<dyn Storage>)> {
    match path {
        None => {
            let storage = StorageFactory::create(StorageConfig::Memory(Default::default()))
                .map_err(laurus_err)?;
            Ok((schema.unwrap_or_default(), storage))
        }
        Some(p) => laurus::index_dir::open_or_create(Path::new(p), schema).map_err(index_dir_err),
    }
}
