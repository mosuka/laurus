//! WASM-facing [`Index`] class — the primary entry point for the laurus-wasm binding.

use std::sync::Arc;

use crate::commit::WasmCommitPolicy;
use crate::convert::{data_value_to_json, json_to_document};
use crate::errors::laurus_err;
use crate::query::{
    JsGeo3dBoundingBoxQuery, JsGeo3dDistanceQuery, JsGeo3dNearestQuery, JsQuery, JsTermQuery,
    JsVectorQuery, JsVectorQueryInner, JsVectorTextQuery,
};
use crate::schema::WasmSchema;
use crate::search::{
    build_dsl_request, build_lexical_request, build_vector_request, parse_highlight_options,
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
    js_embedders: std::collections::HashMap<String, crate::embedder::JsCallbackEmbedder>,
) -> Arc<dyn Embedder> {
    let default: Arc<dyn Embedder> = Arc::new(PrecomputedEmbedder::new());
    let per_field = PerFieldEmbedder::new(default);

    // Build a map: embedder_name -> Arc<dyn Embedder>
    let mut embedder_map: std::collections::HashMap<String, Arc<dyn Embedder>> =
        std::collections::HashMap::new();
    for (name, embedder) in js_embedders {
        embedder_map.insert(name, Arc::new(embedder));
    }

    for name in unused_embedder_names(schema, embedder_map.keys()) {
        web_sys::console::warn_1(
            &format!(
                "laurus-wasm: embedder '{name}' is registered but no schema field references \
                 it, so it will never run. Check the `embedder` argument position in \
                 addHnswField / addFlatField / addIvfField \
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
    ) -> Result<JsValue, JsValue> {
        let mut request = build_dsl_request(
            query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
        request.lexical_options.highlight = parse_highlight_options(highlight)?;
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
    #[wasm_bindgen(js_name = "searchVector")]
    pub async fn search_vector(
        &self,
        field: String,
        vector: Vec<f64>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<JsValue, JsValue> {
        let query = JsVectorQuery::VectorQuery(JsVectorQueryInner {
            field,
            vector: vector.into_iter().map(|v| v as f32).collect(),
        });
        let request = build_vector_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
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
    #[wasm_bindgen(js_name = "searchVectorText")]
    pub async fn search_vector_text(
        &self,
        field: String,
        text: String,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<JsValue, JsValue> {
        let query = JsVectorQuery::VectorTextQuery(JsVectorTextQuery { field, text });
        let request = build_vector_request(
            &query,
            limit.unwrap_or(10) as usize,
            offset.unwrap_or(0) as usize,
        );
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

    use wasm_bindgen::JsCast;

    use super::WasmIndex;
    use crate::schema::WasmSchema;

    /// Build an in-memory index with one stored `body` text field and one
    /// document, ready to commit and search.
    async fn text_index() -> WasmIndex {
        let mut schema = WasmSchema::new();
        schema.add_text_field("body".to_string(), None, None, None, None, None);
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
            .search("body:rust".to_string(), None, None, Some(highlight))
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
            .search("body:rust".to_string(), None, None, None)
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
}
