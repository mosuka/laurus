//! WASM-facing [`Index`] class — the primary entry point for the laurus-wasm binding.

use std::sync::Arc;

use crate::commit::WasmCommitPolicy;
use crate::convert::{data_value_to_json, json_to_document};
use crate::errors::laurus_err;
use crate::query::{
    JsDateTimeRangeQuery, JsGeo3dBoundingBoxQuery, JsGeo3dDistanceQuery, JsGeo3dNearestQuery,
    JsQuery, JsTermQuery, JsVectorQuery, JsVectorQueryInner, JsVectorTextQuery,
};
use crate::schema::WasmSchema;
use crate::search::{
    build_dsl_request, build_lexical_request, build_vector_request, parse_highlight_options,
    parse_rescore_options,
};
use crate::storage::OpfsPersistence;
use crate::wal::WasmWalSyncPolicy;
use laurus::embedding::embedder::Embedder;
use laurus::embedding::per_field::PerFieldEmbedder;
use laurus::embedding::precomputed::PrecomputedEmbedder;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{Engine, EngineBuilder};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Serialize search results to JS via JSON.parse(JSON string).
///
/// This avoids issues with `serde_wasm_bindgen` not correctly handling
/// nested `serde_json::Value` types. Instead, we serialize the entire
/// result to a JSON string and then parse it in JS.
fn search_results_to_js(results: Vec<laurus::SearchResult>) -> Result<JsValue, JsValue> {
    let json_results: Vec<serde_json::Value> = results
        .into_iter()
        .map(|r| {
            let document = r.document.map(|doc| {
                let mut map = serde_json::Map::new();
                for (field, value) in doc.fields {
                    map.insert(field, data_value_to_json(&value));
                }
                serde_json::Value::Object(map)
            });
            serde_json::json!({
                "id": r.id,
                "score": r.score,
                "document": document,
                "highlights": r.highlights,
            })
        })
        .collect();

    let json_str = serde_json::to_string(&json_results)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))?;
    js_sys::JSON::parse(&json_str)
}

/// Serialize documents to JS via JSON.parse(JSON string).
fn documents_to_js(docs: Vec<laurus::Document>) -> Result<JsValue, JsValue> {
    let json_docs: Vec<serde_json::Value> = docs
        .iter()
        .map(|doc| {
            let mut map = serde_json::Map::new();
            for (field, value) in &doc.fields {
                map.insert(field.clone(), data_value_to_json(value));
            }
            serde_json::Value::Object(map)
        })
        .collect();

    let json_str = serde_json::to_string(&json_docs)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))?;
    js_sys::JSON::parse(&json_str)
}

/// Convert a JS array of `[id, doc]` pairs into the engine's
/// `(String, Document)` batch, naming the offending position on any entry
/// that is not a `[string, object]` pair.
fn pairs_to_documents(docs: JsValue) -> Result<Vec<(String, laurus::Document)>, JsValue> {
    let pairs: Vec<(String, serde_json::Value)> =
        serde_wasm_bindgen::from_value(docs).map_err(|e| {
            JsValue::from_str(&format!(
                "Invalid documents: expected an array of [id, doc] pairs: {e}"
            ))
        })?;
    pairs
        .into_iter()
        .enumerate()
        .map(|(index, (id, doc))| {
            let document = json_to_document(&doc).map_err(|e| {
                JsValue::from_str(&format!("documents[{index}]: {}", js_error_string(&e)))
            })?;
            Ok((id, document))
        })
        .collect()
}

/// Best-effort string form of a `JsValue` error for message composition.
fn js_error_string(e: &JsValue) -> String {
    e.as_string().unwrap_or_else(|| format!("{e:?}"))
}

/// Return the registered embedder names that no schema field references,
/// sorted for deterministic output.
///
/// A registered-but-unreferenced embedder is almost always a call-site
/// mistake — most notoriously the embedder name passed at a stale
/// positional slot of `addHnswField` (GitHub issue #978), which JS
/// silently coerces into the wrong parameter. Registering an embedder
/// ahead of a later dynamic field addition is legitimate, so callers
/// should warn, not fail.
///
/// # Arguments
///
/// * `schema` - The schema whose vector fields reference embedders by name.
/// * `registered` - Names of all registered embedders.
///
/// # Returns
///
/// The subset of `registered` that no field's `embedder_name()` mentions.
fn unused_embedder_names<'a>(
    schema: &laurus::Schema,
    registered: impl Iterator<Item = &'a String>,
) -> Vec<String> {
    let referenced: std::collections::HashSet<&str> = schema
        .fields
        .values()
        .filter_map(|option| option.embedder_name())
        .collect();
    let mut unused: Vec<String> = registered
        .filter(|name| !referenced.contains(name.as_str()))
        .cloned()
        .collect();
    unused.sort();
    unused
}

/// Build a [`PerFieldEmbedder`] from JS callback embedders and the schema.
///
/// Reads the schema to find which vector fields reference which embedder name,
/// then maps field names to the corresponding JS callback embedder. Registered
/// embedders that no field references are reported with a console warning
/// (see [`unused_embedder_names`]).
fn build_per_field_embedder(
    schema: &laurus::Schema,
    embedder_map: std::collections::HashMap<String, Arc<dyn Embedder>>,
) -> Arc<dyn Embedder> {
    let default: Arc<dyn Embedder> = Arc::new(PrecomputedEmbedder::new());
    let per_field = PerFieldEmbedder::new(default);

    for name in unused_embedder_names(schema, embedder_map.keys()) {
        web_sys::console::warn_1(
            &format!(
                "laurus-wasm: embedder '{name}' is registered but no schema field references \
                 it, so it will never run. Check the `embedder` argument position in \
                 addHnswField / addFlatField / addIvfField / addMultiVectorField \
                 (https://github.com/mosuka/laurus/issues/978)."
            )
            .into(),
        );
    }

    // Map field_name -> embedder based on the schema's field → embedder_name mapping
    for (field_name, field_option) in &schema.fields {
        if let Some(embedder_name) = field_option.embedder_name()
            && let Some(emb) = embedder_map.get(embedder_name)
        {
            per_field.add_embedder(field_name, emb.clone());
        }
    }

    Arc::new(per_field)
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

/// Laurus search index — the main entry point for the WASM binding.
///
/// Supports two storage modes:
/// - **In-memory** (`Index.create(schema)`) — ephemeral, data lost on page reload
/// - **OPFS-persistent** (`Index.open(name, schema)`) — data survives page reloads
///
/// ```javascript
/// import { Index, Schema } from "laurus-wasm";
///
/// const schema = new Schema();
/// schema.addTextField("title");
/// schema.addTextField("body");
///
/// // In-memory (ephemeral)
/// const index = await Index.create(schema);
///
/// // OPFS-persistent (survives page reloads)
/// const index = await Index.open("my-index", schema);
///
/// await index.putDocument("doc1", { title: "Hello", body: "World" });
/// await index.commit(); // also persists to OPFS if opened with open()
///
/// const results = await index.search("title:hello");
/// ```
#[wasm_bindgen(js_name = "Index")]
pub struct WasmIndex {
    engine: Arc<Engine>,
    storage: Arc<MemoryStorage>,
    opfs: Option<OpfsPersistence>,
}

#[wasm_bindgen(js_class = "Index")]
impl WasmIndex {
    /// Create a new in-memory index (ephemeral, not persisted).
    ///
    /// # Arguments
    ///
    /// * `schema` - Schema definition. An empty schema is used when omitted.
    /// * `wal_sync_policy` - Optional WAL durability policy. Defaults to
    ///   per-record fsync when omitted. Pass `WalSyncPolicy.group()` to opt into
    ///   group-commit batching.
    /// * `commit_policy` - Optional auto-commit policy. Defaults to manual
    ///   (caller-driven commits) when omitted. Pass `CommitPolicy.everyDocs(n)`
    ///   to auto-commit every `n` documents.
    ///
    /// # Returns
    ///
    /// A new `Index` instance backed by in-memory storage.
    ///
    /// # Errors
    ///
    /// Rejects if `schema` defines a custom analyzer (via
    /// `addAnalyzerDefinition` or `fromToml`) under a name reserved for a
    /// built-in analyzer.
    #[wasm_bindgen]
    pub async fn create(
        schema: Option<WasmSchema>,
        wal_sync_policy: Option<WasmWalSyncPolicy>,
        commit_policy: Option<WasmCommitPolicy>,
    ) -> Result<WasmIndex, JsValue> {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let (js_embedders, runtime_analyzers, schema) = match schema {
            Some(s) => (s.js_embedders, s.runtime_analyzers, s.inner),
            None => (
                Default::default(),
                Default::default(),
                laurus::Schema::default(),
            ),
        };
        schema.validate_for_create().map_err(laurus_err)?;

        // Build embedder BEFORE moving schema into EngineBuilder
        let embedder = if js_embedders.is_empty() {
            None
        } else {
            Some(build_per_field_embedder(&schema, js_embedders))
        };

        let mut builder = EngineBuilder::new(storage.clone() as Arc<dyn Storage>, schema);
        if let Some(emb) = embedder {
            builder = builder.embedder(emb);
        }
        for (name, analyzer) in runtime_analyzers {
            builder = builder.register_runtime_analyzer(name, analyzer);
        }
        if let Some(policy) = wal_sync_policy {
            builder = builder.wal_sync_policy(policy.inner);
        }
        if let Some(policy) = commit_policy {
            builder = builder.commit_policy(policy.inner);
        }

        let engine = builder.build().await.map_err(laurus_err)?;

        Ok(Self {
            engine: Arc::new(engine),
            storage,
            opfs: None,
        })
    }

    /// Open or create a persistent index backed by OPFS.
    ///
    /// If an index with the given name already exists in OPFS, its data
    /// (and its persisted schema, if any) is loaded. Otherwise, a new
    /// empty index is created.
    ///
    /// The field-schema part of `schema` is only used when *creating* a
    /// new index (or backfilling one persisted before this method existed
    /// -- see below); once a schema is persisted for this index, it always
    /// wins and the `schema` argument's field definitions are ignored on
    /// subsequent opens. `schema`'s embedder callbacks and runtime
    /// analyzers, however, are **not** persisted (they can't be) and must
    /// still be supplied on every call that needs them, including reopens.
    ///
    /// Data is automatically persisted to OPFS on each `commit()` call.
    ///
    /// # Arguments
    ///
    /// * `name` - Index name (used as the OPFS subdirectory name).
    /// * `schema` - Schema definition, plus any embedder callbacks /
    ///   runtime analyzers this session needs. Required the first time an
    ///   index is created (or when opening one persisted before schema
    ///   tracking was added); optional afterwards.
    /// * `wal_sync_policy` - Optional WAL durability policy. Defaults to
    ///   per-record fsync when omitted. Pass `WalSyncPolicy.group()` to opt into
    ///   group-commit batching.
    /// * `commit_policy` - Optional auto-commit policy. Defaults to manual
    ///   (caller-driven commits) when omitted. Pass `CommitPolicy.everyDocs(n)`
    ///   to auto-commit every `n` documents.
    ///
    /// # Returns
    ///
    /// A new `Index` instance backed by OPFS-persistent storage.
    ///
    /// # Errors
    ///
    /// Rejects if this OPFS index has data from before schema tracking was
    /// added and `schema` was not supplied to complete the one-time
    /// migration.
    #[wasm_bindgen]
    pub async fn open(
        name: String,
        schema: Option<WasmSchema>,
        wal_sync_policy: Option<WasmWalSyncPolicy>,
        commit_policy: Option<WasmCommitPolicy>,
    ) -> Result<WasmIndex, JsValue> {
        let opfs = OpfsPersistence::open(&name).await?;
        let (js_embedders, runtime_analyzers, schema_candidate) = match schema {
            Some(s) => (s.js_embedders, s.runtime_analyzers, Some(s.inner)),
            None => (Default::default(), Default::default(), None),
        };
        let schema = opfs.resolve_schema(schema_candidate).await?;
        let storage = opfs.load().await?;

        let embedder = if js_embedders.is_empty() {
            None
        } else {
            Some(build_per_field_embedder(&schema, js_embedders))
        };

        let mut builder = EngineBuilder::new(storage.clone() as Arc<dyn Storage>, schema);
        if let Some(emb) = embedder {
            builder = builder.embedder(emb);
        }
        for (name, analyzer) in runtime_analyzers {
            builder = builder.register_runtime_analyzer(name, analyzer);
        }
        if let Some(policy) = wal_sync_policy {
            builder = builder.wal_sync_policy(policy.inner);
        }
        if let Some(policy) = commit_policy {
            builder = builder.commit_policy(policy.inner);
        }

        let engine = builder.build().await.map_err(laurus_err)?;

        Ok(Self {
            engine: Arc::new(engine),
            storage,
            opfs: Some(opfs),
        })
    }

    // ── Document CRUD ─────────────────────────────────────────────────────

    /// Index a document, replacing any existing document with the same id.
    ///
    /// Call `commit()` to make the change visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier (string).
    /// * `doc` - A JS object mapping field names to values.
    #[wasm_bindgen(js_name = "putDocument")]
    pub async fn put_document(&self, id: String, doc: JsValue) -> Result<(), JsValue> {
        let value: serde_json::Value = serde_wasm_bindgen::from_value(doc)
            .map_err(|e| JsValue::from_str(&format!("Invalid document: {e}")))?;
        let document = json_to_document(&value)?;
        self.engine
            .put_document(&id, document)
            .await
            .map_err(laurus_err)
    }

    /// Append a document version without removing existing versions.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    /// * `doc` - A JS object mapping field names to values.
    #[wasm_bindgen(js_name = "addDocument")]
    pub async fn add_document(&self, id: String, doc: JsValue) -> Result<(), JsValue> {
        let value: serde_json::Value = serde_wasm_bindgen::from_value(doc)
            .map_err(|e| JsValue::from_str(&format!("Invalid document: {e}")))?;
        let document = json_to_document(&value)?;
        self.engine
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
    /// * `docs` - A JS array of `[id, doc]` pairs.
    #[wasm_bindgen(js_name = "putDocuments")]
    pub async fn put_documents(&self, docs: JsValue) -> Result<(), JsValue> {
        let batch = pairs_to_documents(docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        self.engine.put_documents(batch).await.map_err(laurus_err)
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
    /// * `docs` - A JS array of `[id, doc]` pairs.
    #[wasm_bindgen(js_name = "addDocuments")]
    pub async fn add_documents(&self, docs: JsValue) -> Result<(), JsValue> {
        let batch = pairs_to_documents(docs)?;
        if batch.is_empty() {
            return Ok(());
        }
        self.engine.add_documents(batch).await.map_err(laurus_err)
    }

    /// Retrieve all document versions stored under `id`.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    ///
    /// # Returns
    ///
    /// A JS array of document objects.
    #[wasm_bindgen(js_name = "getDocuments")]
    pub async fn get_documents(&self, id: String) -> Result<JsValue, JsValue> {
        let docs = self.engine.get_documents(&id).await.map_err(laurus_err)?;
        documents_to_js(docs)
    }

    /// Delete all document versions stored under `id`.
    ///
    /// Call `commit()` to make the deletion visible to searches.
    ///
    /// # Arguments
    ///
    /// * `id` - External document identifier.
    #[wasm_bindgen(js_name = "deleteDocuments")]
    pub async fn delete_documents(&self, id: String) -> Result<(), JsValue> {
        self.engine.delete_documents(&id).await.map_err(laurus_err)
    }

    /// Flush buffered writes and make all pending changes searchable.
    ///
    /// If this index was opened with `Index.open()`, the data is also
    /// persisted to OPFS.
    #[wasm_bindgen]
    pub async fn commit(&self) -> Result<(), JsValue> {
        self.engine.commit().await.map_err(laurus_err)?;

        // Persist to OPFS if applicable
        if let Some(opfs) = &self.opfs {
            opfs.save(self.storage.as_ref()).await?;
        }

        Ok(())
    }

    /// Force every appended-but-unsynced WAL record durable, without a full
    /// `commit()` (Issue #542 / #820).
    ///
    /// Under the default per-record policy this is a near-no-op: each
    /// `putDocument` / `addDocument` / `deleteDocuments` already fsyncs, so there
    /// is nothing pending. Under `WalSyncPolicy.group()` appends defer their
    /// fsync, so this is the way to bound the crash-loss window at an
    /// application-chosen point without paying the cost of materializing the
    /// lexical/vector indexes that `commit()` entails.
    ///
    /// Unlike `commit()`, this does **not** rebuild the searchable indexes and
    /// does **not** persist to OPFS — OPFS durability still requires `commit()`.
    ///
    /// **wasm note:** the `maxIntervalMs` background flush timer configured via
    /// `WalSyncPolicy.group()` is a no-op under WebAssembly (it is native-only in
    /// the core), so `flushWal()` and `commit()` are the only ways to flush a
    /// deferred group-commit batch on this target.
    #[wasm_bindgen(js_name = "flushWal")]
    pub async fn flush_wal(&self) -> Result<(), JsValue> {
        self.engine.flush_wal().map_err(laurus_err)
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// Search using a DSL string query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query DSL string (e.g. `"title:hello"`).
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    /// * `highlight` - Optional highlight request (Issue #1134):
    ///   `{ fields: ["body"], maxFragments?, fragmentSize?, tag?, cssClass?,
    ///   requireFieldMatch? }`. Only `fields` is required. Highlighting
    ///   follows this query, and only `stored: true` text fields can be
    ///   highlighted.
    /// * `rescore` - Optional late-interaction rescore of the top results
    ///   (Issue #1351): `{ field, vectors?, text?, windowSize? }` with
    ///   exactly one of `vectors` (`number[][]`) and `text`. The top
    ///   `windowSize` results (default 100) are reordered by MaxSim against
    ///   the multi-vector `field`.
    ///
    /// # Returns
    ///
    /// A JS array of SearchResult objects `{ id, score, document, highlights }`.
    #[wasm_bindgen]
    pub async fn search(
        &self,
        query: String,
        limit: Option<u32>,
        offset: Option<u32>,
        highlight: Option<js_sys::Object>,
        rescore: Option<js_sys::Object>,
    ) -> Result<JsValue, JsValue> {
        let mut request = build_dsl_request(
            query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        request.lexical_options.highlight = parse_highlight_options(highlight)?;
        request.rescore = parse_rescore_options(rescore)?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a term query.
    ///
    /// # Arguments
    ///
    /// * `field` - The field to search in.
    /// * `term` - The exact term to match.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    /// * `highlight` - Optional highlight request; same shape as `search`'s
    ///   `highlight` argument (Issue #1134).
    #[wasm_bindgen(js_name = "searchTerm")]
    pub async fn search_term(
        &self,
        field: String,
        term: String,
        limit: Option<u32>,
        offset: Option<u32>,
        highlight: Option<js_sys::Object>,
    ) -> Result<JsValue, JsValue> {
        let query = JsQuery::TermQuery(JsTermQuery { field, term });
        let mut request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        request.lexical_options.highlight = parse_highlight_options(highlight)?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search a DateTime field for values within an inclusive range
    /// (Issue #1179).
    ///
    /// # Arguments
    ///
    /// * `field` - The DateTime field name.
    /// * `min` - Lower bound (inclusive) or `null`/`undefined` for unbounded.
    /// * `max` - Upper bound (inclusive) or `null`/`undefined` for unbounded.
    ///   Bounds are datetime literals in any form the query DSL accepts:
    ///   RFC 3339 (`"2024-01-01T09:00:00+09:00"`, normalized to UTC), a naive
    ///   `"YYYY-MM-DDTHH:MM:SS[.fff]"` (UTC), or a date `"YYYY-MM-DD"`
    ///   (midnight UTC). A `Date` can be passed as `date.toISOString()`.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    /// * `highlight` - Optional highlight request; same shape as `search`'s
    ///   `highlight` argument.
    ///
    /// Rejects with an error when a bound is not a recognized literal.
    #[wasm_bindgen(js_name = "searchDateTimeRange")]
    pub async fn search_date_time_range(
        &self,
        field: String,
        min: Option<String>,
        max: Option<String>,
        limit: Option<u32>,
        offset: Option<u32>,
        highlight: Option<js_sys::Object>,
    ) -> Result<JsValue, JsValue> {
        let query = JsQuery::DateTimeRangeQuery(JsDateTimeRangeQuery { field, min, max });
        let mut request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        request.lexical_options.highlight = parse_highlight_options(highlight)?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a 3D ECEF distance (sphere) query.
    ///
    /// # Arguments
    ///
    /// * `field` - The Geo3d field name.
    /// * `x`, `y`, `z` - Sphere centre in ECEF meters.
    /// * `distance_m` - Maximum distance from the centre in meters
    ///   (i.e. the search sphere's radius).
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    #[wasm_bindgen(js_name = "searchGeo3dDistance")]
    #[allow(clippy::too_many_arguments)]
    pub async fn search_geo3d_distance(
        &self,
        field: String,
        x: f64,
        y: f64,
        z: f64,
        distance_m: f64,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<JsValue, JsValue> {
        let query = JsQuery::Geo3dDistanceQuery(JsGeo3dDistanceQuery {
            field,
            x,
            y,
            z,
            distance_m,
        });
        let request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a 3D ECEF axis-aligned bounding-box query.
    ///
    /// # Arguments
    ///
    /// * `field` - The Geo3d field name.
    /// * `min_x`, `min_y`, `min_z` - Lower corner of the box.
    /// * `max_x`, `max_y`, `max_z` - Upper corner of the box.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    #[wasm_bindgen(js_name = "searchGeo3dBoundingBox")]
    #[allow(clippy::too_many_arguments)]
    pub async fn search_geo3d_bounding_box(
        &self,
        field: String,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<JsValue, JsValue> {
        let query = JsQuery::Geo3dBoundingBoxQuery(JsGeo3dBoundingBoxQuery {
            field,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        });
        let request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a 3D ECEF k-nearest-neighbours query.
    ///
    /// # Arguments
    ///
    /// * `field` - The Geo3d field name.
    /// * `x`, `y`, `z` - Centre coordinates in ECEF meters.
    /// * `k` - Number of nearest neighbours to return.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    #[wasm_bindgen(js_name = "searchGeo3dNearest")]
    #[allow(clippy::too_many_arguments)]
    pub async fn search_geo3d_nearest(
        &self,
        field: String,
        x: f64,
        y: f64,
        z: f64,
        k: u32,
        limit: Option<u32>,
        offset: Option<u32>,
        initial_radius_m: Option<f64>,
        max_radius_m: Option<f64>,
    ) -> Result<JsValue, JsValue> {
        let query = JsQuery::Geo3dNearestQuery(JsGeo3dNearestQuery {
            field,
            x,
            y,
            z,
            k,
            initial_radius_m,
            max_radius_m,
        });
        let request = build_lexical_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        )?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a pre-computed embedding vector.
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `vector` - The embedding vector as a Float64Array or number[].
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    /// * `rescore` - Optional late-interaction rescore; same shape as
    ///   `search`'s `rescore` argument (Issue #1351).
    #[wasm_bindgen(js_name = "searchVector")]
    pub async fn search_vector(
        &self,
        field: String,
        vector: Vec<f64>,
        limit: Option<u32>,
        offset: Option<u32>,
        rescore: Option<js_sys::Object>,
    ) -> Result<JsValue, JsValue> {
        let query = JsVectorQuery::VectorQuery(JsVectorQueryInner {
            field,
            vector: vector.into_iter().map(|v| v as f32).collect(),
        });
        let mut request = build_vector_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        request.rescore = parse_rescore_options(rescore)?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    /// Search using a text-based vector query (embedded by the registered embedder).
    ///
    /// # Arguments
    ///
    /// * `field` - The vector field name.
    /// * `text` - The text to embed and search with.
    /// * `limit` - Maximum number of results (default 10).
    /// * `offset` - Pagination offset (default 0).
    /// * `rescore` - Optional late-interaction rescore; same shape as
    ///   `search`'s `rescore` argument (Issue #1351).
    #[wasm_bindgen(js_name = "searchVectorText")]
    pub async fn search_vector_text(
        &self,
        field: String,
        text: String,
        limit: Option<u32>,
        offset: Option<u32>,
        rescore: Option<js_sys::Object>,
    ) -> Result<JsValue, JsValue> {
        let query = JsVectorQuery::VectorTextQuery(JsVectorTextQuery { field, text });
        let mut request = build_vector_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        request.rescore = parse_rescore_options(rescore)?;
        let results = self.engine.search(request).await.map_err(laurus_err)?;
        search_results_to_js(results)
    }

    // ── Stats ─────────────────────────────────────────────────────────────

    /// Return index statistics.
    ///
    /// # Returns
    ///
    /// An object with `documentCount` and `vectorFields`.
    #[wasm_bindgen]
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        let stats = self.engine.stats().map_err(laurus_err)?;
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
        let json = serde_json::json!({
            "documentCount": stats.document_count,
            "vectorFields": vector_fields,
        });
        let json_str = serde_json::to_string(&json)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {e}")))?;
        js_sys::JSON::parse(&json_str)
    }
}

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::wasm_bindgen_test;

    use laurus::lexical::TextOption;
    use laurus::{FieldOption, HnswOption, Schema};

    use super::unused_embedder_names;

    /// Build a schema with one text field and one HNSW field referencing
    /// `embedder_name` (or none).
    fn schema_with_embedder(embedder_name: Option<&str>) -> Schema {
        Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .add_field(
                "embedding",
                FieldOption::Hnsw(HnswOption {
                    dimension: 3,
                    embedder: embedder_name.map(str::to_string),
                    ..Default::default()
                }),
            )
            .build()
    }

    /// A registered embedder that no field references must be reported
    /// (the #978 stale-positional-argument failure class).
    #[wasm_bindgen_test]
    fn unreferenced_embedder_is_reported() {
        let schema = schema_with_embedder(None);
        let registered = ["minilm".to_string()];
        assert_eq!(
            unused_embedder_names(&schema, registered.iter()),
            ["minilm".to_string()],
        );
    }

    /// An embedder referenced by a field must not be flagged.
    #[wasm_bindgen_test]
    fn referenced_embedder_is_not_reported() {
        let schema = schema_with_embedder(Some("minilm"));
        let registered = ["minilm".to_string()];
        assert!(unused_embedder_names(&schema, registered.iter()).is_empty());
    }

    /// Mixed case: only the unreferenced name is reported, sorted.
    #[wasm_bindgen_test]
    fn only_unreferenced_names_are_reported_sorted() {
        let schema = schema_with_embedder(Some("minilm"));
        let registered = [
            "zeta".to_string(),
            "minilm".to_string(),
            "alpha".to_string(),
        ];
        assert_eq!(
            unused_embedder_names(&schema, registered.iter()),
            ["alpha".to_string(), "zeta".to_string()],
        );
    }

    // ── Highlighting (Issue #1134) ──────────────────────────────────────────

    use wasm_bindgen::{JsCast, JsValue};

    use super::WasmIndex;
    use crate::schema::WasmSchema;

    /// Build an in-memory index with one stored `body` text field and one
    /// document, ready to commit and search.
    async fn text_index() -> WasmIndex {
        let mut schema = WasmSchema::new();
        schema
            .add_text_field("body".to_string(), None, None, None, None, None, None, None)
            .expect("addTextField must succeed");
        let index = WasmIndex::create(Some(schema), None, None)
            .await
            .expect("index creation must succeed");
        let doc = serde_wasm_bindgen::to_value(&serde_json::json!({
            "body": "Rust is a systems programming language"
        }))
        .unwrap();
        index
            .put_document("doc1".to_string(), doc)
            .await
            .expect("put_document must succeed");
        index.commit().await.expect("commit must succeed");
        index
    }

    /// Build a genuine JS object (property access, not a `Map`) from a
    /// `serde_json::Value` — matching what a real caller passes from JS.
    /// `serde_wasm_bindgen::to_value` on a JSON object serializes to a JS
    /// `Map` by default, which `serde_wasm_bindgen::from_value` cannot
    /// deserialize into a plain struct the way `parse_highlight_options`
    /// expects; going through `JSON.parse` (as `search_results_to_js` does
    /// for its own output) sidesteps that entirely.
    fn highlight_object(value: serde_json::Value) -> js_sys::Object {
        let json = serde_json::to_string(&value).unwrap();
        js_sys::JSON::parse(&json)
            .unwrap()
            .dyn_into::<js_sys::Object>()
            .expect("highlight options must parse to a JS object")
    }

    #[wasm_bindgen_test]
    async fn search_with_highlight_returns_fragments_for_the_requested_field() {
        let index = text_index().await;
        let highlight = highlight_object(serde_json::json!({ "fields": ["body"] }));

        let js_results = index
            .search("body:rust".to_string(), None, None, Some(highlight), None)
            .await
            .expect("search must succeed");
        let results: serde_json::Value = serde_wasm_bindgen::from_value(js_results).unwrap();

        let fragments = results[0]["highlights"]["body"]
            .as_array()
            .expect("body must be highlighted");
        assert!(
            fragments[0].as_str().unwrap().contains("<mark>Rust</mark>"),
            "{results:?}"
        );
    }

    #[wasm_bindgen_test]
    async fn search_without_highlight_leaves_highlights_empty() {
        let index = text_index().await;

        let js_results = index
            .search("body:rust".to_string(), None, None, None, None)
            .await
            .expect("search must succeed");
        let results: serde_json::Value = serde_wasm_bindgen::from_value(js_results).unwrap();

        assert_eq!(results[0]["highlights"], serde_json::json!({}));
    }

    #[wasm_bindgen_test]
    async fn search_term_with_highlight_config_applies_the_tag() {
        let index = text_index().await;
        let highlight = highlight_object(serde_json::json!({
            "fields": ["body"],
            "tag": "em",
        }));

        let js_results = index
            .search_term(
                "body".to_string(),
                "rust".to_string(),
                None,
                None,
                Some(highlight),
            )
            .await
            .expect("searchTerm must succeed");
        let results: serde_json::Value = serde_wasm_bindgen::from_value(js_results).unwrap();

        let fragment = results[0]["highlights"]["body"][0].as_str().unwrap();
        assert!(fragment.contains("<em>Rust</em>"), "{fragment}");
    }

    // ── Analyzer definitions & TOML schema I/O (Issue #1062) ────────────────

    #[wasm_bindgen_test]
    async fn add_analyzer_definition_enables_ngram_substring_match() {
        let mut schema = WasmSchema::new();
        let definition = highlight_object(serde_json::json!({
            "tokenizer": { "type": "ngram", "min_gram": 3, "max_gram": 3 }
        }));
        schema
            .add_analyzer_definition("ngram3".to_string(), definition)
            .expect("addAnalyzerDefinition must succeed");
        schema
            .add_text_field(
                "title".to_string(),
                None,
                None,
                None,
                None,
                Some("ngram3".to_string()),
                None,
                None,
            )
            .expect("addTextField must succeed");
        // Default "standard" analyzer: whole-word tokens only.
        schema
            .add_text_field(
                "plain".to_string(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("addTextField must succeed");

        let index = WasmIndex::create(Some(schema), None, None)
            .await
            .expect("index creation must succeed");
        let doc = serde_wasm_bindgen::to_value(&serde_json::json!({
            "title": "hello",
            "plain": "hello"
        }))
        .unwrap();
        index
            .put_document("doc1".to_string(), doc)
            .await
            .expect("put_document must succeed");
        index.commit().await.expect("commit must succeed");

        // "ell" is a substring of "hello", only reachable via 3-grams (hel/ell/llo).
        let title_hits = index
            .search("title:ell".to_string(), None, None, None, None)
            .await
            .expect("search must succeed");
        let title_hits: serde_json::Value = serde_wasm_bindgen::from_value(title_hits).unwrap();
        assert_eq!(title_hits.as_array().unwrap().len(), 1);

        let plain_hits = index
            .search("plain:ell".to_string(), None, None, None, None)
            .await
            .expect("search must succeed");
        let plain_hits: serde_json::Value = serde_wasm_bindgen::from_value(plain_hits).unwrap();
        assert_eq!(plain_hits.as_array().unwrap().len(), 0);
    }

    /// This is the most direct proof that lifting the core's wasm32 gate on
    /// `Schema::from_toml`/`to_toml` (Issue #1062) actually took effect: if
    /// the gate were still present, `WasmSchema::from_toml`/`to_toml`
    /// wouldn't compile at all for this target.
    #[wasm_bindgen_test]
    async fn to_toml_then_from_toml_round_trip_preserves_fields_and_analyzers() {
        let mut schema = WasmSchema::new();
        let definition = highlight_object(serde_json::json!({
            "tokenizer": { "type": "ngram", "min_gram": 3, "max_gram": 3 }
        }));
        schema
            .add_analyzer_definition("ngram3".to_string(), definition)
            .expect("addAnalyzerDefinition must succeed");
        schema
            .add_text_field(
                "title".to_string(),
                None,
                None,
                None,
                None,
                Some("ngram3".to_string()),
                None,
                None,
            )
            .expect("addTextField must succeed");

        let toml_str = schema.to_toml().expect("toToml must succeed");
        let restored = WasmSchema::from_toml(toml_str).expect("fromToml must succeed");

        assert_eq!(restored.field_names(), schema.field_names());
        assert_eq!(restored.analyzer_names(), schema.analyzer_names());

        let index = WasmIndex::create(Some(restored), None, None)
            .await
            .expect("index creation must succeed");
        let doc = serde_wasm_bindgen::to_value(&serde_json::json!({ "title": "hello" })).unwrap();
        index
            .put_document("doc1".to_string(), doc)
            .await
            .expect("put_document must succeed");
        index.commit().await.expect("commit must succeed");

        let hits = index
            .search("title:ell".to_string(), None, None, None, None)
            .await
            .expect("search must succeed");
        let hits: serde_json::Value = serde_wasm_bindgen::from_value(hits).unwrap();
        assert_eq!(hits.as_array().unwrap().len(), 1);
    }

    #[wasm_bindgen_test]
    fn add_analyzer_definition_requires_a_tokenizer_key() {
        let mut schema = WasmSchema::new();
        let definition = highlight_object(serde_json::json!({}));
        let err = schema
            .add_analyzer_definition("bad".to_string(), definition)
            .expect_err("missing tokenizer must be rejected");
        let message = err.as_string().unwrap();
        assert!(message.contains("tokenizer"), "{message}");
    }

    #[wasm_bindgen_test]
    fn add_analyzer_definition_rejects_unknown_tokenizer_type() {
        let mut schema = WasmSchema::new();
        let definition = highlight_object(serde_json::json!({
            "tokenizer": { "type": "kuromoji" }
        }));
        let err = schema
            .add_analyzer_definition("bad".to_string(), definition)
            .expect_err("unknown tokenizer type must be rejected");
        let message = err.as_string().unwrap();
        assert!(message.contains("tokenizer"), "{message}");
    }

    // ── Names reserved for built-in analyzers (Issue #1310) ─────────────────

    #[wasm_bindgen_test]
    fn add_analyzer_definition_rejects_reserved_names() {
        for name in laurus::analysis::analyzer::registry::RESERVED_ANALYZER_NAMES {
            let mut schema = WasmSchema::new();
            let definition = highlight_object(serde_json::json!({
                "tokenizer": { "type": "whitespace" }
            }));
            let err = schema
                .add_analyzer_definition(name.to_string(), definition)
                .expect_err("a reserved name must be rejected");
            let message = err.as_string().unwrap();
            assert!(
                message.contains(&format!(
                    "Analyzer name '{name}' is reserved for a built-in analyzer"
                )),
                "{message}"
            );
            assert!(schema.analyzer_names().is_empty());
        }
    }

    #[wasm_bindgen_test]
    fn add_analyzer_definition_accepts_japanese_name() {
        let mut schema = WasmSchema::new();
        let definition = highlight_object(serde_json::json!({
            "tokenizer": { "type": "whitespace" }
        }));
        schema
            .add_analyzer_definition("japanese".to_string(), definition)
            .expect("'japanese' is not reserved");
        assert_eq!(schema.analyzer_names(), ["japanese".to_string()]);
    }

    #[wasm_bindgen_test]
    async fn create_rejects_reserved_analyzer_name() {
        let schema = WasmSchema::from_toml(
            r#"
            [analyzers.standard]
            tokenizer = { type = "whitespace" }

            [fields.body.Text]
            analyzer = "standard"
            "#
            .to_string(),
        )
        .expect("fromToml accepts the entry; Index.create rejects it");
        let Err(err) = WasmIndex::create(Some(schema), None, None).await else {
            panic!("a reserved analyzer name must be rejected");
        };
        let message = err.as_string().unwrap();
        assert!(
            message.contains("reserved for a built-in analyzer"),
            "{message}"
        );
    }

    #[wasm_bindgen_test]
    async fn create_rejects_reserved_field_name() {
        let schema = WasmSchema::from_toml(
            r#"
            [fields._secret.Text]
            "#
            .to_string(),
        )
        .expect("fromToml accepts the field; Index.create rejects it");
        let Err(err) = WasmIndex::create(Some(schema), None, None).await else {
            panic!("a reserved field name must be rejected");
        };
        let message = err.as_string().unwrap();
        assert!(
            message.contains("Field name '_secret' is reserved"),
            "{message}"
        );
    }

    /// Every `add*Field` method rejects a reserved name at call time and
    /// leaves the schema untouched (Issue #1331).
    #[wasm_bindgen_test]
    fn add_field_methods_reject_reserved_field_name() {
        type AddField = fn(&mut WasmSchema, String) -> Result<(), JsValue>;
        let methods: [(&str, AddField); 12] = [
            ("addTextField", |s, n| {
                s.add_text_field(n, None, None, None, None, None, None, None)
            }),
            ("addIntegerField", |s, n| {
                s.add_integer_field(n, None, None, None, None)
            }),
            ("addFloatField", |s, n| {
                s.add_float_field(n, None, None, None, None)
            }),
            ("addBooleanField", |s, n| {
                s.add_boolean_field(n, None, None, None, None)
            }),
            ("addDatetimeField", |s, n| {
                s.add_datetime_field(n, None, None, None, None)
            }),
            ("addGeoField", |s, n| {
                s.add_geo_field(n, None, None, None, None)
            }),
            ("addGeo3dField", |s, n| {
                s.add_geo3d_field(n, None, None, None, None)
            }),
            ("addBytesField", |s, n| s.add_bytes_field(n, None, None)),
            ("addHnswField", |s, n| {
                s.add_hnsw_field(
                    n, 3, None, None, None, None, None, None, None, None, None, None,
                )
            }),
            ("addFlatField", |s, n| {
                s.add_flat_field(n, 3, None, None, None)
            }),
            ("addIvfField", |s, n| {
                s.add_ivf_field(n, 3, None, None, None, None, None)
            }),
            ("addMultiVectorField", |s, n| {
                s.add_multi_vector_field(n, 3, None, None, None)
            }),
        ];
        for (method, add_field) in methods {
            let mut schema = WasmSchema::new();
            let Err(err) = add_field(&mut schema, "_secret".to_string()) else {
                panic!("{method} must reject a reserved field name");
            };
            let message = err.as_string().unwrap();
            assert!(
                message.contains("Field name '_secret' is reserved"),
                "{method}: {message}"
            );
            assert!(schema.field_names().is_empty(), "{method}");
        }
    }

    // ── Late-interaction rescore (Issue #1351) ─────────────────────────────
    //
    // Uses the same corpus as the Rust test
    // (`laurus/tests/late_interaction_rescore_test.rs`): against the query
    // token vectors [[1, 0], [0, 1]] the MaxSim scores are c 1.1, b 1.0,
    // d 0.9, a 0.1, e 0.05 — an order that matches neither the BM25 nor
    // the `vec` ranking — so the binding must rank exactly like the Rust
    // API.

    const EXPECTED: [&str; 5] = ["c", "b", "d", "a", "e"];

    /// `(id, title, vec, tokens)` of each corpus document.
    fn corpus() -> serde_json::Value {
        serde_json::json!([
            ["a", "rust", [1.0, 0.0], [[0.1, 0.0]]],
            ["b", "rust rust", [0.9, 0.1], [[0.5, 0.5], [0.0, 0.2]]],
            ["c", "rust language", [0.5, 0.5], [[0.9, 0.2]]],
            ["d", "rust rust rust", [0.2, 0.8], [[0.3, 0.3], [0.6, 0.0]]],
            ["e", "learning rust today", [0.0, 1.0], [[0.02, 0.03]]]
        ])
    }

    /// A token callback that returns the corpus token vectors of the
    /// document whose id is the text, and the test query for any query.
    /// `wrap` is the returned expression around `out`, so a test can return
    /// the vectors synchronously or as a Promise.
    fn token_callback(wrap: &str) -> js_sys::Function {
        let tokens: serde_json::Map<String, serde_json::Value> = corpus()
            .as_array()
            .unwrap()
            .iter()
            .map(|doc| (doc[0].as_str().unwrap().to_string(), doc[3].clone()))
            .collect();
        js_sys::Function::new_with_args(
            "text, role",
            &format!(
                "const tokens = {}; \
                 const out = role === 'query' ? [[1, 0], [0, 1]] : tokens[text]; \
                 return {wrap};",
                serde_json::Value::Object(tokens)
            ),
        )
    }

    /// A `token_callback` embedder config, with `embed` set when given.
    fn token_callback_config(embed: Option<&js_sys::Function>, dimension: Option<u32>) -> JsValue {
        let config = highlight_object(serde_json::json!({ "type": "token_callback" }));
        if let Some(embed) = embed {
            js_sys::Reflect::set(&config, &JsValue::from_str("embed"), embed).unwrap();
        }
        if let Some(dimension) = dimension {
            js_sys::Reflect::set(&config, &JsValue::from_str("dimension"), &dimension.into())
                .unwrap();
        }
        config.into()
    }

    /// A schema with the corpus fields; `tokens` is embedded by `callback`
    /// when given.
    fn rescore_schema(callback: Option<&js_sys::Function>) -> WasmSchema {
        let mut schema = WasmSchema::new();
        schema
            .add_text_field(
                "title".to_string(),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        schema
            .add_flat_field("vec".to_string(), 2, None, None, None)
            .unwrap();
        if let Some(func) = callback {
            schema
                .add_embedder(
                    "colbert".to_string(),
                    token_callback_config(Some(func), Some(2)),
                )
                .unwrap();
        }
        schema
            .add_multi_vector_field(
                "tokens".to_string(),
                2,
                Some("dot_product".to_string()),
                callback.map(|_| "colbert".to_string()),
                None,
            )
            .unwrap();
        schema
    }

    /// Build the corpus index. With a token callback, the `tokens` values
    /// are the document ids, embedded by the callback; otherwise they are
    /// the token vectors themselves.
    async fn rescore_index(callback: Option<js_sys::Function>) -> WasmIndex {
        let embedded = callback.is_some();
        let index = WasmIndex::create(Some(rescore_schema(callback.as_ref())), None, None)
            .await
            .expect("index creation must succeed");
        for doc in corpus().as_array().unwrap() {
            let tokens = if embedded {
                doc[0].clone()
            } else {
                doc[3].clone()
            };
            let value = serde_wasm_bindgen::to_value(&serde_json::json!({
                "title": doc[1], "vec": doc[2], "tokens": tokens
            }))
            .unwrap();
            index
                .put_document(doc[0].as_str().unwrap().to_string(), value)
                .await
                .expect("put_document must succeed");
        }
        index.commit().await.expect("commit must succeed");
        index
    }

    fn ids(js_results: JsValue) -> Vec<String> {
        let results: serde_json::Value = serde_wasm_bindgen::from_value(js_results).unwrap();
        results
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["id"].as_str().unwrap().to_string())
            .collect()
    }

    fn vectors_rescore() -> js_sys::Object {
        highlight_object(serde_json::json!({
            "field": "tokens",
            "vectors": [[1.0, 0.0], [0.0, 1.0]]
        }))
    }

    #[wasm_bindgen_test]
    async fn search_is_reordered_by_late_interaction() {
        let index = rescore_index(None).await;
        let baseline = index
            .search("title:rust".to_string(), None, None, None, None)
            .await
            .unwrap();
        assert_ne!(ids(baseline), EXPECTED);

        let results = index
            .search(
                "title:rust".to_string(),
                None,
                None,
                None,
                Some(vectors_rescore()),
            )
            .await
            .unwrap();
        let results: serde_json::Value = serde_wasm_bindgen::from_value(results).unwrap();
        let scores: Vec<(String, f64)> = results
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                (
                    r["id"].as_str().unwrap().to_string(),
                    r["score"].as_f64().unwrap(),
                )
            })
            .collect();
        let expected = [("c", 1.1), ("b", 1.0), ("d", 0.9), ("a", 0.1), ("e", 0.05)];
        assert_eq!(scores.len(), expected.len());
        for ((id, score), (expected_id, expected_score)) in scores.iter().zip(expected) {
            assert_eq!(id, expected_id);
            assert!((score - expected_score).abs() < 1e-5, "{scores:?}");
        }
    }

    #[wasm_bindgen_test]
    async fn vector_search_is_reordered_by_late_interaction() {
        let index = rescore_index(None).await;
        let results = index
            .search_vector(
                "vec".to_string(),
                vec![1.0, 0.0],
                None,
                None,
                Some(vectors_rescore()),
            )
            .await
            .unwrap();
        assert_eq!(ids(results), EXPECTED);
    }

    #[wasm_bindgen_test]
    async fn token_callback_embeds_documents_and_the_query_text() {
        for wrap in ["out", "Promise.resolve(out)"] {
            let index = rescore_index(Some(token_callback(wrap))).await;
            let rescore = highlight_object(serde_json::json!({
                "field": "tokens",
                "text": "how fast is rust"
            }));
            let results = index
                .search("title:rust".to_string(), None, None, None, Some(rescore))
                .await
                .unwrap();
            assert_eq!(ids(results), EXPECTED, "{wrap}");
        }
    }

    #[wasm_bindgen_test]
    async fn token_callback_dimension_must_match_the_field() {
        let mut schema = WasmSchema::new();
        schema
            .add_embedder(
                "colbert".to_string(),
                token_callback_config(Some(&token_callback("out")), Some(3)),
            )
            .unwrap();
        schema
            .add_multi_vector_field(
                "tokens".to_string(),
                2,
                None,
                Some("colbert".to_string()),
                None,
            )
            .unwrap();
        let Err(err) = WasmIndex::create(Some(schema), None, None).await else {
            panic!("a dimension mismatch must be rejected");
        };
        let message = err.as_string().unwrap();
        assert!(
            message.contains("produces 3-dimensional token vectors"),
            "{message}"
        );
    }

    #[wasm_bindgen_test]
    async fn token_callback_must_return_numbers() {
        let callback = js_sys::Function::new_with_args("text, role", "return [[1, 'x']];");
        let index = WasmIndex::create(Some(rescore_schema(Some(&callback))), None, None)
            .await
            .unwrap();
        let doc = serde_wasm_bindgen::to_value(&serde_json::json!({ "tokens": "a" })).unwrap();
        let Err(err) = index.put_document("a".to_string(), doc).await else {
            panic!("a non-numeric token vector must be rejected");
        };
        let message = err.as_string().unwrap();
        assert!(
            message.contains("returned a non-number in token vector 0"),
            "{message}"
        );
    }

    #[wasm_bindgen_test]
    fn token_callback_config_is_checked() {
        let embed = token_callback("out");
        let cases = [
            (None, Some(2), "'embed' function"),
            (Some(&embed), None, "positive integer 'dimension'"),
            (Some(&embed), Some(0), "positive integer 'dimension'"),
        ];
        for (embed, dimension, message) in cases {
            let config = token_callback_config(embed, dimension);
            let Err(err) = WasmSchema::new().add_embedder("colbert".to_string(), config) else {
                panic!("an invalid token_callback config must be rejected: {message}");
            };
            let error = err.as_string().unwrap();
            assert!(error.contains(message), "{error}");
        }
    }

    #[wasm_bindgen_test]
    async fn invalid_rescore_is_rejected() {
        let index = rescore_index(None).await;
        let cases = [
            (
                serde_json::json!({ "field": "tokens" }),
                "exactly one of vectors or text",
            ),
            (
                serde_json::json!({ "field": "tokens", "vectors": [[1.0, 0.0]], "text": "x" }),
                "exactly one of vectors or text",
            ),
            (
                serde_json::json!({ "field": "tokens", "vectors": [[1.0, "x"]] }),
                "Invalid rescore options",
            ),
            (
                serde_json::json!({ "field": "tokens", "text": "rust" }),
                "has no token-level embedder",
            ),
            (
                serde_json::json!({ "field": "tokens", "vectors": [[1.0, 0.0]], "windowSize": 0 }),
                "window_size must be between",
            ),
        ];
        for (rescore, message) in cases {
            let Err(err) = index
                .search(
                    "title:rust".to_string(),
                    None,
                    None,
                    None,
                    Some(highlight_object(rescore)),
                )
                .await
            else {
                panic!("an invalid rescore must be rejected: {message}");
            };
            let error = err.as_string().unwrap();
            assert!(error.contains(message), "{error}");
        }
    }

    #[wasm_bindgen_test]
    fn add_multi_vector_field_rejects_invalid_options() {
        let cases = [
            (0, None, None, "dimension must be greater than 0"),
            (
                2,
                Some("euclidean"),
                None,
                "distance must be Cosine or DotProduct",
            ),
            (2, None, Some("bogus"), "Unknown multi-vector storage"),
        ];
        for (dimension, distance, storage, message) in cases {
            let mut schema = WasmSchema::new();
            let Err(err) = schema.add_multi_vector_field(
                "tokens".to_string(),
                dimension,
                distance.map(str::to_string),
                None,
                storage.map(str::to_string),
            ) else {
                panic!("invalid multi-vector options must be rejected: {message}");
            };
            let error = err.as_string().unwrap();
            assert!(error.contains(message), "{error}");
            assert!(schema.field_names().is_empty());
        }
    }

    #[wasm_bindgen_test]
    fn add_multi_vector_field_accepts_valid_storage() {
        for storage in ["f32", "f16", "int8"] {
            let mut schema = WasmSchema::new();
            schema
                .add_multi_vector_field(
                    "tokens".to_string(),
                    2,
                    None,
                    None,
                    Some(storage.to_string()),
                )
                .unwrap_or_else(|err| panic!("storage={storage} must be accepted: {err:?}"));
            assert_eq!(schema.field_names(), vec!["tokens".to_string()]);
        }
    }
}
