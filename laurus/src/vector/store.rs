//! VectorStore: Simplified vector storage following LexicalStore pattern.
//!
//! This module provides a vector storage component with a simple 3-member structure:
//! - `index`: The underlying vector index
//! - `writer_cache`: Cached writer for write operations (`tokio::sync::Mutex`)
//! - `searcher_cache`: Cached searcher for search operations (`parking_lot::RwLock`)
//!
//! # Concurrency Strategy
//!
//! - **Searcher cache** uses double-checked locking with `RwLockWriteGuard::downgrade()`
//!   so that only searcher *creation* (on cache miss) holds an exclusive lock; the actual
//!   search executes under a shared read lock, allowing concurrent queries.
//! - **Writer cache** is protected by a `tokio::sync::Mutex`. Embedding (potentially slow
//!   network I/O) is performed *outside* the lock; only the final `delete + add_vectors`
//!   step runs while the lock is held, keeping the critical section short.
//!
//! # Module Structure
//!
//! - [`config`] - Configuration types (VectorIndexConfig, VectorFieldConfig)
//! - [`embedding_writer`] - Embedding writer wrapper
//! - [`request`] - Search request types
//! - [`response`] - Search response types

pub mod config;
pub mod embedding_writer;
pub mod memory;
pub mod request;
pub mod response;

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::data::{DataValue, Document};
use crate::embedding::embedder::{EmbedInput, Embedder};
use crate::embedding::per_field::PerFieldEmbedder;
use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use crate::vector::index::VectorIndex;
use crate::vector::index::config::VectorIndexTypeConfig;
use crate::vector::index::factory::VectorIndexFactory;
use crate::vector::search::filter_set::FilterSet;
use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
use crate::vector::writer::VectorIndexWriter;

use self::config::VectorIndexConfig;
use self::request::{FieldSelector, QueryVector, VectorScoreMode, VectorSearchRequest};
use self::response::{VectorHit, VectorSearchResults, VectorStats};

/// A simplified vector storage component following the LexicalStore pattern.
///
/// This structure mirrors `LexicalStore` with only 3 members:
/// - `index`: The underlying vector index
/// - `writer_cache`: Cached writer for write operations
/// - `searcher_cache`: Cached searcher for search operations
pub struct VectorStore {
    /// The underlying vector index.
    index: Box<dyn VectorIndex>,
    /// Cached writer (created on-demand).
    writer_cache: Mutex<Option<Box<dyn VectorIndexWriter>>>,
    /// Cached searcher (invalidated after commit/optimize).
    searcher_cache: parking_lot::RwLock<Option<Box<dyn VectorIndexSearcher>>>,
    /// The collection-wide configuration this store was constructed with
    /// (Issue #948), retained so [`Self::add_field`] can build a properly
    /// parametrized [`VectorIndexTypeConfig`] (embedder, deletion policy)
    /// for a field added after construction. `None` for stores built via
    /// [`Self::with_index_type_config`] (the single-index constructor),
    /// which never had a collection-wide config to retain.
    config: Option<VectorIndexConfig>,
    /// Per-field relative scoring priority (Issue #1084), keyed by field
    /// name. Seeded from `config.fields` at construction and kept in sync
    /// by [`Self::add_field`]/[`Self::delete_field`]/[`Self::rebuild_field`].
    /// A field with no entry (including every field of a store built via
    /// [`Self::with_index_type_config`]) falls back to `1.0` in
    /// [`Self::base_weight_of`].
    base_weights: parking_lot::RwLock<HashMap<String, f32>>,
}

impl std::fmt::Debug for VectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStore")
            .field("index", &self.index)
            .finish()
    }
}

impl VectorStore {
    /// Create a new vector store with the given storage and high-level configuration.
    ///
    /// This constructor is compatible with Engine and accepts VectorIndexConfig.
    /// Every vector field gets its own independent sub-index (Issue
    /// [#948](https://github.com/mosuka/laurus/issues/948)) via
    /// [`VectorIndexConfig::field_index_configs`] and
    /// [`VectorIndexFactory::open_or_create_multi_field`] -- this replaces
    /// the old behavior of collapsing every field down to whichever one
    /// happened to come first out of `config.fields` (a `HashMap`, so
    /// non-deterministic), which silently discarded every other field's
    /// dimension, distance metric, and HNSW parameters, and could corrupt
    /// data when one document had vectors in more than one field.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend for persisting index data
    /// * `config` - High-level configuration (compatible with Engine)
    ///
    /// # Returns
    ///
    /// Returns a new `VectorStore` instance.
    pub fn new(storage: Arc<dyn Storage>, config: VectorIndexConfig) -> Result<Self> {
        let field_configs = config.field_index_configs();
        let index = VectorIndexFactory::open_or_create_multi_field(
            storage,
            &field_configs,
            config.embedder.clone(),
        )?;
        let base_weights = config
            .fields
            .iter()
            .filter_map(|(name, field_config)| {
                field_config
                    .vector
                    .as_ref()
                    .map(|opt| (name.clone(), Self::sanitize_base_weight(opt.base_weight())))
            })
            .collect();
        Ok(Self {
            index: Box::new(index),
            writer_cache: Mutex::new(None),
            searcher_cache: parking_lot::RwLock::new(None),
            config: Some(config),
            base_weights: parking_lot::RwLock::new(base_weights),
        })
    }

    /// Create a new vector store with explicit index type configuration.
    ///
    /// This is a lower-level constructor for when you have a specific
    /// VectorIndexTypeConfig.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend for persisting index data
    /// * `config` - Configuration for the vector index (Flat, HNSW, or IVF)
    ///
    /// # Returns
    ///
    /// Returns a new `VectorStore` instance.
    pub fn with_index_type_config(
        storage: Arc<dyn Storage>,
        config: VectorIndexTypeConfig,
    ) -> Result<Self> {
        let index = VectorIndexFactory::open_or_create(storage, "vector_index", config)?;
        Ok(Self {
            index,
            writer_cache: Mutex::new(None),
            searcher_cache: parking_lot::RwLock::new(None),
            config: None,
            base_weights: parking_lot::RwLock::new(HashMap::new()),
        })
    }

    /// Upsert a document by its internal ID.
    ///
    /// This method first deletes any existing vectors for the given `doc_id`,
    /// then iterates over all fields in the document and passes each field value
    /// to the writer's [`add_value()`](crate::vector::writer::VectorIndexWriter::add_value)
    /// method, which handles embedding automatically when the writer is wrapped
    /// in an `EmbeddingVectorIndexWriter`.
    ///
    /// It is primarily used during WAL recovery where the internal ID
    /// is already known.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID.
    /// * `doc` - The document whose fields will be indexed as vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining/creating the writer fails, if deleting the
    /// existing document fails, or if adding any field value fails.
    pub async fn upsert_document_by_internal_id(&self, doc_id: u64, doc: Document) -> Result<()> {
        // Phase 1: Embed all fields OUTSIDE the lock.
        // This allows multiple concurrent upserts to perform embedding in parallel
        // rather than being serialized by the writer Mutex.
        let embedder = self.index.embedder();
        let mut embedded_vectors: Vec<(u64, String, Vector)> = Vec::new();

        for (field_name, value) in &doc.fields {
            let vector = match value {
                DataValue::Vector(v) => Vector::new(v.clone()),
                DataValue::Text(_) | DataValue::Bytes(_, _) => {
                    Self::embed_value(&*embedder, field_name, value).await?
                }
                _ => continue,
            };
            embedded_vectors.push((doc_id, field_name.clone(), vector));
        }

        // Phase 2: Acquire lock and write pre-computed vectors (fast, sync-only).
        let mut guard = self.writer_cache.lock().await;
        if guard.is_none() {
            *guard = Some(self.index.writer()?);
        }
        let writer = guard.as_mut().unwrap();
        writer.delete_document(doc_id)?;
        writer.add_vectors(embedded_vectors)?;

        Ok(())
    }

    /// Validate input and embed a single field value into a vector.
    ///
    /// This is a helper extracted from `EmbeddingVectorIndexWriter::add_value()`
    /// to allow embedding to happen outside the writer lock.
    ///
    /// # Arguments
    ///
    /// * `embedder` - The embedder to use for converting content to vectors.
    /// * `field_name` - The name of the field being embedded.
    /// * `value` - The data value to embed (must be `Text` or `Bytes`).
    ///
    /// # Errors
    ///
    /// Returns an error if the embedder does not support the input type or if
    /// the embedding operation fails.
    async fn embed_value(
        embedder: &dyn Embedder,
        field_name: &str,
        value: &DataValue,
    ) -> Result<Vector> {
        // Validate input type compatibility
        match value {
            DataValue::Text(_) if !embedder.supports_text() => {
                return Err(LaurusError::invalid_argument(format!(
                    "Embedder '{}' does not support text input",
                    embedder.name()
                )));
            }
            DataValue::Bytes(_, mime)
                if !embedder.supports_image()
                    && mime.as_ref().is_some_and(|m| m.starts_with("image/")) =>
            {
                return Err(LaurusError::invalid_argument(format!(
                    "Embedder '{}' does not support image input",
                    embedder.name()
                )));
            }
            _ => {}
        }

        // Prepare owned data for the embed call
        let (text_owned, bytes_owned, mime_owned) = match value {
            DataValue::Text(t) => (Some(t.clone()), None, None),
            DataValue::Bytes(b, m) => (None, Some(b.clone()), m.clone()),
            _ => {
                return Err(LaurusError::invalid_argument(
                    "Unsupported data type for embedding",
                ));
            }
        };

        let input = if let Some(ref text) = text_owned {
            EmbedInput::Text(text)
        } else if let Some(ref bytes) = bytes_owned {
            EmbedInput::Bytes(bytes, mime_owned.as_deref())
        } else {
            return Err(LaurusError::internal("Unreachable state in embed_value"));
        };

        // Use field-specific embedder if PerFieldEmbedder, otherwise default.
        if let Some(per_field) = embedder.as_any().downcast_ref::<PerFieldEmbedder>() {
            per_field.embed_field(field_name, &input).await
        } else {
            embedder.embed(&input).await
        }
    }

    /// Delete a document by its internal ID.
    ///
    /// Obtains (or creates) the cached writer and removes all vectors
    /// associated with the given `doc_id` from the index buffer.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID to delete.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining/creating the writer fails or if the
    /// underlying delete operation fails.
    pub async fn delete_document_by_internal_id(&self, doc_id: u64) -> Result<()> {
        // Prefer logical (soft) deletion when the index supports it (Issue
        // #624): mark the deletion bitmap and invalidate the searcher cache so
        // the next search filters the document out via the deletion-aware
        // traversal (Issue #665). This avoids the full graph rebuild that the
        // writer-side delete triggers, and applies equally to updates (the
        // engine deletes the old internal id, which is monotonic and never
        // reused, before adding the new version).
        if self.index.supports_soft_delete() {
            self.index.soft_delete_document(doc_id)?;
            *self.searcher_cache.write() = None;
            return Ok(());
        }

        let mut guard = self.writer_cache.lock().await;
        if guard.is_none() {
            *guard = Some(self.index.writer()?);
        }
        let writer = guard.as_mut().unwrap();

        writer.delete_document(doc_id)?;

        Ok(())
    }

    /// Commit any pending changes to the index.
    ///
    /// If a cached writer exists, this method calls
    /// [`commit()`](crate::vector::writer::VectorIndexWriter::commit) on it
    /// (which finalizes the index and writes it to storage). It then syncs
    /// the underlying storage to ensure all file metadata is flushed to disk,
    /// refreshes the index metadata, and invalidates the searcher cache so
    /// that subsequent searches see the committed data.
    ///
    /// When the index opts in via
    /// [`VectorIndex::retain_writer_after_commit`] (Issue #572 / #864), the
    /// committed writer stays in the cache — its in-memory state is
    /// equivalent to the file it just wrote — so the first upsert after the
    /// commit does not reload the whole index from storage. The cache is
    /// still dropped when auto-compaction ran: compaction rewrites the index
    /// through a fresh writer and clears the deletion bitmap, so a retained
    /// writer would resurrect the physically reclaimed vectors on its next
    /// commit. The writer-cache lock is held across the whole ladder so a
    /// concurrent upsert cannot interleave with the commit.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer commit, deletion persistence,
    /// compaction, storage sync, or index refresh fails.
    pub async fn commit(&self) -> Result<()> {
        let mut writer_guard = self.writer_cache.lock().await;
        // commit() calls finalize() then write() to persist to storage. A
        // retained writer with no pending changes is skipped — its state was
        // already captured by the previous finalize+write, so re-committing
        // would only rewrite an identical index file.
        let flush_result = match writer_guard.as_mut() {
            Some(writer) if writer.has_pending_changes() => writer.commit(),
            _ => Ok(()),
        };
        // A FAILED flush must keep the writer cached (#882 review, the #875
        // lesson): its buffered mutations are the only in-process copy, and
        // the segmented index's pending WAL checkpoint may already cover
        // their sequence numbers — dropping the writer while a later
        // successful commit publishes that checkpoint would let recovery
        // skip records whose effects were silently discarded. The seal is
        // atomic (tmp+rename before manifest registration), so a retry from
        // the retained writer is sound.
        if let Err(e) = flush_result {
            drop(writer_guard);
            *self.searcher_cache.write() = None;
            return Err(e);
        }
        let ladder_result = Ok(())
            // Persist any pending logical deletions (Issue #624) so the
            // deletion bitmap survives restarts. The WAL also records
            // deletions, so this is a durability optimization rather than the
            // source of truth.
            .and_then(|_| self.index.persist_deletions())
            // Automatically compact when the deletion ratio crosses the
            // configured threshold (Issue #782), so logically deleted vectors
            // are physically reclaimed rather than accumulating indefinitely.
            // A no-op unless the index supports it and `auto_compaction` is
            // enabled.
            .and_then(|_| self.index.maybe_auto_compact());
        match ladder_result {
            Ok(compacted) => {
                if compacted || !self.index.retain_writer_after_commit() {
                    *writer_guard = None;
                }
            }
            Err(e) => {
                // A mid-ladder failure leaves the writer/disk agreement
                // unknown — compaction in particular may have partially
                // rewritten the index and cleared the deletion bitmap before
                // failing. Drop the cache so the next writer reloads ground
                // truth from storage (the pre-retention behavior on every
                // path), instead of retrying — or resurrecting from — a
                // stale writer.
                *writer_guard = None;
                return Err(e);
            }
        }
        drop(writer_guard);
        // Sync storage to ensure all file metadata (creation, rename, size) is
        // flushed to disk. This is critical on Windows where directory listings
        // and file visibility may be cached until the directory is synced.
        self.index.storage().sync()?;
        self.index.refresh()?;
        *self.searcher_cache.write() = None;
        Ok(())
    }

    /// Optimize the index for improved query performance.
    ///
    /// Delegates to the underlying [`VectorIndex::optimize()`] implementation
    /// and then invalidates the searcher cache so the next search creates a
    /// fresh searcher reflecting the optimized state.
    ///
    /// The cached writer is committed first when it holds buffered documents
    /// (so optimization compacts the full state and nothing is lost), and the
    /// cache is dropped in every case: `optimize()` rewrites the index
    /// through a fresh writer and clears the deletion bitmap, so a writer
    /// retained across it would resurrect the physically reclaimed vectors
    /// on its next commit (Issue #864).
    ///
    /// # Errors
    ///
    /// Returns an error if flushing the cached writer or the underlying index
    /// optimization fails.
    pub async fn optimize(&self) -> Result<()> {
        let mut writer_guard = self.writer_cache.lock().await;
        // Flush uncommitted mutations first. `has_pending_changes` (not
        // `pending_docs`) is the gate: a writer whose buffer was emptied by
        // deletions has zero pending docs but still holds an uncommitted
        // delete-everything mutation that dropping would silently discard.
        let flush_result = match writer_guard.as_mut() {
            Some(writer) if writer.has_pending_changes() => writer.commit(),
            _ => Ok(()),
        };
        // A failed flush keeps the writer cached (#882 review / #875): its
        // buffered mutations are the only in-process copy and their WAL
        // records may already be inside the pending checkpoint's range.
        flush_result?;
        // Drop the cache on success — optimize() rewrites the index through
        // a fresh writer and clears the deletion bitmap, so a retained
        // writer would resurrect reclaimed vectors (#864).
        *writer_guard = None;
        self.index.optimize()?;
        drop(writer_guard);
        *self.searcher_cache.write() = None;
        Ok(())
    }

    /// Invalidate the searcher cache.
    ///
    /// Clears the cached searcher so that the next search operation creates a
    /// new one. This is useful after external writes that bypass this store's
    /// commit path.
    pub fn refresh(&self) -> Result<()> {
        *self.searcher_cache.write() = None;
        Ok(())
    }

    /// Acquire a read lock on the cached searcher, populating the cache on miss.
    ///
    /// Uses double-checked locking: first tries a shared read lock (fast path),
    /// then falls back to an exclusive write lock to create the searcher and
    /// atomically downgrades it to a read lock so concurrent searches are not
    /// blocked while the actual query executes.
    fn acquire_searcher_guard(
        &self,
    ) -> Result<parking_lot::RwLockReadGuard<'_, Option<Box<dyn VectorIndexSearcher>>>> {
        // Fast path: cache hit under read lock.
        {
            let guard = self.searcher_cache.read();
            if guard.is_some() {
                return Ok(guard);
            }
        }

        // Slow path: populate under write lock, then downgrade.
        let mut guard = self.searcher_cache.write();
        if guard.is_none() {
            *guard = Some(self.index.searcher()?);
        }
        Ok(parking_lot::RwLockWriteGuard::downgrade(guard))
    }

    /// Warm the cached searcher so the first query does not pay its setup cost
    /// (Issue #677).
    ///
    /// Eagerly builds and caches the index searcher — which loads the reader
    /// (file → memory for `InMemory`, the offset table for `Mmap`) — and then
    /// invokes the searcher's
    /// [`warmup`](crate::vector::search::searcher::VectorIndexSearcher::warmup),
    /// which pre-faults on-disk vector data into the OS page cache where
    /// applicable (HNSW `Mmap` mode). This moves the searcher-construction and
    /// page-fault latency off the first query.
    ///
    /// Safe to call multiple times and from any index type (a no-op
    /// `warmup` for searchers that do not override it). Typically called once
    /// at startup via [`Engine::warmup`](crate::engine::Engine::warmup).
    ///
    /// # Errors
    ///
    /// Returns an error if building the searcher (reader load) fails.
    pub fn warmup(&self) -> Result<()> {
        let mut guard = self.searcher_cache.write();
        if guard.is_none() {
            *guard = Some(self.index.searcher()?);
        }
        if let Some(searcher) = guard.as_mut() {
            searcher.warmup()?;
        }
        Ok(())
    }

    /// Execute a low-level vector similarity search.
    pub fn search_index(
        &self,
        request: &VectorIndexQuery,
    ) -> Result<crate::vector::search::searcher::VectorIndexQueryResults> {
        let guard = self.acquire_searcher_guard()?;
        guard.as_ref().unwrap().search(request)
    }

    /// Execute a high-level vector search (compatible with Engine).
    ///
    /// This method extracts query vectors from the
    /// [`VectorSearchQuery`](crate::vector::search::searcher::VectorSearchQuery)
    /// inside the request, performs a similarity search against the index, and
    /// aggregates the per-vector scores according to the requested
    /// [`score_mode`](crate::vector::search::searcher::VectorSearchParams::score_mode).
    /// Results are filtered by
    /// [`allowed_ids`](crate::vector::search::searcher::VectorSearchParams::allowed_ids)
    /// and
    /// [`min_score`](crate::vector::search::searcher::VectorSearchParams::min_score),
    /// sorted by descending score, and truncated to
    /// [`limit`](crate::vector::search::searcher::VectorSearchParams::limit).
    ///
    /// Field routing (Issue #676): each query vector is routed to a set of
    /// target fields. A query's own
    /// [`fields`](crate::vector::store::request::QueryVector::fields) take
    /// precedence; otherwise the request-level
    /// [`fields`](crate::vector::search::searcher::VectorSearchParams::fields)
    /// selectors apply ([`Exact`](crate::vector::store::request::FieldSelector::Exact)
    /// by name, [`Prefix`](crate::vector::store::request::FieldSelector::Prefix)
    /// resolved against the reader's field names). When neither is set, all
    /// indexed fields are searched (the default).
    ///
    /// The per-field candidate pool is widened by
    /// [`overfetch`](crate::vector::search::searcher::VectorSearchParams::overfetch)
    /// via
    /// [`overfetch_top_k`](crate::vector::search::searcher::VectorSearchParams::overfetch_top_k)
    /// (Issue #675) so the score-mode merge has headroom before the final
    /// truncation to `limit`.
    ///
    /// **Note:** The following request field is currently **ignored** by this
    /// implementation:
    /// - `VectorSearchQuery::Payloads` -- callers must embed payloads into
    ///   vectors before calling this method.
    ///
    /// # Arguments
    ///
    /// * `request` - The search request containing query vectors, filters, and
    ///   scoring options.
    ///
    /// # Returns
    ///
    /// A [`VectorSearchResults`] containing hits sorted by descending score.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining the searcher or executing the underlying
    /// index search fails, or if the query contains unresolved payloads.
    pub fn search(&self, request: VectorSearchRequest) -> Result<VectorSearchResults> {
        self.search_impl(request, None)
    }

    /// Test-only variant of [`Self::search`] that lets the caller pin the
    /// multi-vector parallelisation threshold.
    ///
    /// When `parallel_threshold == 0` the multi-vector path always runs in
    /// parallel (when the `native` feature is on); when it is `usize::MAX`
    /// the path always runs serially. Production code goes through
    /// [`Self::search`], which uses the searcher's
    /// [`VectorIndexSearcher::parallel_threshold`] (default `4`).
    ///
    /// Issue [#710](https://github.com/mosuka/laurus/issues/710) Phase 1 of
    /// [#648](https://github.com/mosuka/laurus/issues/648); refactored in
    /// Phase 2 ([#712](https://github.com/mosuka/laurus/issues/712)) to
    /// dispatch through the trait method.
    #[doc(hidden)]
    pub fn search_with_threshold(
        &self,
        request: VectorSearchRequest,
        parallel_threshold: usize,
    ) -> Result<VectorSearchResults> {
        self.search_impl(request, Some(parallel_threshold))
    }

    /// Clamp a raw `base_weight` to a usable value (Issue #1084).
    ///
    /// `base_weight` is a multiplicative factor on similarity scores, so a
    /// non-positive or non-finite value would zero out, negate, or NaN-out
    /// every hit from the field. This also rescues indexes whose
    /// `schema.toml` already has `base_weight: 0.0` from a pre-#1084
    /// gRPC/REST-created schema (proto3's non-optional `float` used to
    /// default to `0.0`, silently distinct from core's `1.0` default).
    fn sanitize_base_weight(raw: f32) -> f32 {
        if raw.is_finite() && raw > 0.0 {
            raw
        } else {
            log::warn!(
                "base_weight must be a positive, finite value; got {raw}, falling back to 1.0"
            );
            1.0
        }
    }

    /// Look up a field's sanitized `base_weight`, defaulting to `1.0` when
    /// the field has no entry (schema-less fields, non-vector fields, and
    /// every field of a store built via [`Self::with_index_type_config`]).
    fn base_weight_of(&self, field: &str) -> f32 {
        self.base_weights.read().get(field).copied().unwrap_or(1.0)
    }

    /// Resolve the set of vector fields a single query should be routed to
    /// (Issue #676).
    ///
    /// Precedence:
    /// 1. `QueryVector.fields` (per-query) when set — the engine populates
    ///    this from the field a DSL clause names (e.g. `image_vec:"..."`).
    /// 2. Otherwise `VectorSearchParams.fields` (request-level), resolving
    ///    [`FieldSelector::Exact`] directly and [`FieldSelector::Prefix`]
    ///    against `reader_field_names`.
    /// 3. Otherwise an empty `Vec`, meaning "search all fields" (the
    ///    historical behaviour — one query with `field_name = None`).
    ///
    /// `reader_field_names` is only consulted for `Prefix` selectors; the
    /// caller passes an empty slice when no `Prefix` is present.
    fn resolve_target_fields(
        qv: &QueryVector,
        params_fields: Option<&[FieldSelector]>,
        reader_field_names: &[String],
    ) -> Vec<String> {
        if let Some(fields) = &qv.fields {
            return fields.clone();
        }
        if let Some(selectors) = params_fields {
            let mut out = Vec::new();
            for selector in selectors {
                match selector {
                    FieldSelector::Exact(name) => out.push(name.clone()),
                    FieldSelector::Prefix(prefix) => out.extend(
                        reader_field_names
                            .iter()
                            .filter(|n| n.starts_with(prefix))
                            .cloned(),
                    ),
                }
            }
            return out;
        }
        Vec::new()
    }

    /// Common implementation for [`Self::search`] and
    /// [`Self::search_with_threshold`].
    ///
    /// `threshold_override == None` uses the searcher's own
    /// [`VectorIndexSearcher::parallel_threshold`]; `Some(t)` pins the
    /// threshold for tests.
    fn search_impl(
        &self,
        request: VectorSearchRequest,
        threshold_override: Option<usize>,
    ) -> Result<VectorSearchResults> {
        use crate::vector::search::searcher::VectorSearchQuery;

        let query_vectors = match &request.query {
            VectorSearchQuery::Vectors(vecs) => vecs,
            VectorSearchQuery::Payloads(_) => {
                return Err(crate::error::LaurusError::invalid_argument(
                    "VectorStore::search requires pre-embedded vectors; \
                     Payloads must be embedded before calling this method",
                ));
            }
        };

        if query_vectors.is_empty() {
            return Ok(VectorSearchResults::default());
        }

        let searcher_guard = self.acquire_searcher_guard()?;
        let searcher = searcher_guard.as_ref().unwrap();

        // Resolve reader field names only when a Prefix selector is present
        // (Issue #676); Exact selectors and per-query fields need no lookup.
        let reader_field_names: Vec<String> = match &request.params.fields {
            Some(sels) if sels.iter().any(|s| matches!(s, FieldSelector::Prefix(_))) => {
                // Issue #948: `field_dimensions()` is an in-memory lookup,
                // so prefer it over building a full reader just to read
                // field names. Empty for index types that do not override
                // it (single-index `with_index_type_config` construction),
                // where the reader-based lookup below is still correct.
                let dims = self.index.field_dimensions();
                if !dims.is_empty() {
                    dims.into_keys().collect()
                } else {
                    self.index.reader()?.field_names().unwrap_or_default()
                }
            }
            _ => Vec::new(),
        };

        // Build the filter-aware allow-set once (Issues #645 / #739), shared
        // (by Arc clone) across the 1 → N field expansion below and reused by
        // the inline Flat / IVF filters and the post-filter. The Engine path
        // hands us a pre-built `Arc<RoaringTreemap>` (`allowed_filter`) from the
        // lexical filter cache, which we wrap as a `FilterSet::Bitmap` without
        // copying; external callers pass `allowed_ids: Vec<u64>`, from which we
        // pick a representation by shape.
        let filter_set: Option<std::sync::Arc<FilterSet>> =
            if let Some(bitmap) = request.params.allowed_filter.as_ref() {
                Some(std::sync::Arc::new(FilterSet::from_bitmap(bitmap.clone())))
            } else {
                request
                    .params
                    .allowed_ids
                    .as_ref()
                    .map(|ids| std::sync::Arc::new(FilterSet::from_doc_ids(ids)))
            };

        // Expand each query vector to its target fields (Issue #676). A query
        // with no resolved fields searches all fields (`field_name = None`);
        // one targeting N fields becomes N index queries. `query_weights`
        // runs parallel to `index_queries` so per-query weights survive the
        // 1 → N expansion.
        let mut index_queries: Vec<VectorIndexQuery> = Vec::new();
        let mut query_weights: Vec<f32> = Vec::new();
        for qv in query_vectors {
            let targets = Self::resolve_target_fields(
                qv,
                request.params.fields.as_deref(),
                &reader_field_names,
            );
            let make = |field: Option<&str>| {
                let mut q = VectorIndexQuery::new(qv.vector.clone())
                    .top_k(request.params.overfetch_top_k());
                if let Some(field) = field {
                    q = q.field_name(field.to_string());
                }
                if let Some(factor) = request.params.rerank_factor {
                    q = q.rerank_factor(factor);
                }
                if let Some(ef) = request.params.ef_search {
                    q = q.ef_search(ef);
                }
                if let Some(ref f) = filter_set {
                    q = q.filter(f.clone());
                }
                q
            };
            if targets.is_empty() {
                // Field-less fanout search (Issue #676): no single field is
                // known here, so `base_weight` cannot be applied (Issue
                // #1084 leaves this out of scope — see the store-level doc
                // comment on `base_weights`).
                index_queries.push(make(None));
                query_weights.push(qv.weight);
            } else {
                for field in &targets {
                    index_queries.push(make(Some(field)));
                    query_weights.push(qv.weight * self.base_weight_of(field));
                }
            }
        }

        // Fast path: a single index query (one query vector routed to one or
        // no specific field) — skip HashMap aggregation.
        if index_queries.len() == 1 {
            let weight = query_weights[0];
            let results = searcher.search(&index_queries[0])?;

            let mut hits: Vec<VectorHit> = results
                .results
                .into_iter()
                .filter(|r| {
                    if let Some(ref allowed) = filter_set
                        && !allowed.contains(r.doc_id)
                    {
                        return false;
                    }
                    r.similarity >= request.params.min_score
                })
                .map(|r| VectorHit {
                    doc_id: r.doc_id,
                    score: r.similarity * weight,
                    field_hits: vec![],
                })
                .collect();

            // Use partial sort for top-K selection when the result set is larger
            // than the requested limit.
            let limit = request.params.limit.min(hits.len());
            if limit > 0 && limit < hits.len() {
                hits.select_nth_unstable_by(limit - 1, |a, b| b.score.total_cmp(&a.score));
                hits.truncate(limit);
                hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
            } else if !hits.is_empty() {
                hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
            }

            return Ok(VectorSearchResults { hits });
        }

        // Multi-query / multi-field path (Phase 2 of #648, issue #712):
        // dispatch the batch via `search_batch_with_threshold` (parallelised
        // inside the trait method) and merge serially.
        let per_query_results = match threshold_override {
            Some(t) => searcher.search_batch_with_threshold(&index_queries, t)?,
            None => searcher.search_batch(&index_queries)?,
        };

        // Serial merge by score_mode (applies allowed_ids / min_score filter
        // and the per-query weight that the trait method intentionally does
        // not know about). `query_weights[i]` is the weight for the query that
        // produced `per_query_results[i]`.
        let mut all_hits: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();
        for (weight, results) in query_weights.iter().zip(per_query_results) {
            for result in results.results {
                if let Some(ref allowed) = filter_set
                    && !allowed.contains(result.doc_id)
                {
                    continue;
                }
                if result.similarity < request.params.min_score {
                    continue;
                }
                let weighted_score = result.similarity * weight;
                let entry = all_hits.entry(result.doc_id).or_insert(0.0);
                match request.params.score_mode {
                    VectorScoreMode::WeightedSum | VectorScoreMode::LateInteraction => {
                        // WeightedSum: sum of similarity * weight across all query vectors.
                        // LateInteraction: for each query vector, find the max similarity
                        // across document vectors, then sum. In the current single-vector-
                        // per-field architecture, this is equivalent to WeightedSum since
                        // each query vector already gets a single best match per document.
                        *entry += weighted_score;
                    }
                    VectorScoreMode::MaxSim => {
                        // MaxSim: take the maximum weighted similarity across query vectors.
                        if weighted_score > *entry {
                            *entry = weighted_score;
                        }
                    }
                }
            }
        }

        // Convert to VectorHit and sort by score with doc_id tiebreak for
        // parallel-deterministic ordering (issue #710 Phase 1 of #648).
        let mut hits: Vec<VectorHit> = all_hits
            .into_iter()
            .map(|(doc_id, score)| VectorHit {
                doc_id,
                score,
                field_hits: vec![],
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });

        // Apply limit
        if hits.len() > request.params.limit {
            hits.truncate(request.params.limit);
        }

        Ok(VectorSearchResults { hits })
    }

    /// Count the number of vectors matching the given search request.
    ///
    /// Delegates to the searcher's
    /// [`count()`](crate::vector::search::searcher::VectorIndexSearcher::count)
    /// method, which returns the total number of vectors that match the query
    /// criteria.
    ///
    /// # Arguments
    ///
    /// * `request` - A low-level vector index search request specifying the
    ///   query vector and parameters.
    ///
    /// # Returns
    ///
    /// The number of matching vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining the searcher or executing the count fails.
    pub fn count(&self, request: VectorIndexQuery) -> Result<u64> {
        let guard = self.acquire_searcher_guard()?;
        guard.as_ref().unwrap().count(request)
    }

    /// Get index statistics including per-field vector counts.
    ///
    /// Returns a [`VectorStats`] containing the total document count and
    /// per-field statistics (vector count and dimension) for each vector
    /// field in the index. The dimension is derived from the actual vectors
    /// stored for each field, falling back to the index-level dimension when
    /// no vectors are present.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining the reader fails.
    pub fn stats(&self) -> Result<VectorStats> {
        let reader = self.index.reader()?;
        let doc_count = reader.vector_count();
        let index_dimension = reader.dimension();
        // Issue #948: each field's OWN configured dimension, so a field
        // with zero vectors in a heterogeneous multi-field index (e.g. a
        // 384-dim `title_vec` alongside an empty 128-dim `other_vec`)
        // reports its own dimension instead of falling back to
        // `index_dimension` -- which for `MultiFieldVectorIndex` is just
        // the first field's dimension in iteration order, not this one's.
        // Empty for index types that do not override
        // `field_dimensions` (single-index `with_index_type_config`
        // construction), where `index_dimension` remains the only and
        // correct fallback, same as before this fix.
        let configured_dimensions = self.index.field_dimensions();

        let mut fields = std::collections::HashMap::new();
        if let Ok(field_names) = reader.field_names() {
            for name in field_names {
                // Issue #1089: `doc_ids_for_field` is an O(1)-ish lookup
                // into a pre-built per-field index for every concrete reader
                // that matters here (multi-field/HNSW/IVF/flat), unlike
                // `get_vectors_by_field`, which materializes and clones
                // every vector in the field just to count them.
                let doc_ids = reader.doc_ids_for_field(&name);
                let vector_count = doc_ids.len();
                // Derive dimension from one actual vector when present
                // (exact ground truth, and only a single lookup rather than
                // the whole field); otherwise this field's own configured
                // dimension; otherwise the index-level dimension.
                let dimension = doc_ids
                    .first()
                    .and_then(|&doc_id| reader.get_vector(doc_id, &name).ok().flatten())
                    .map(|v| v.data.len())
                    .or_else(|| configured_dimensions.get(&name).copied())
                    .unwrap_or(index_dimension);
                fields.insert(
                    name,
                    crate::vector::index::field::VectorFieldStats {
                        vector_count,
                        dimension,
                    },
                );
            }
        }

        Ok(VectorStats {
            document_count: doc_count,
            fields,
        })
    }

    /// Sample up to `limit` `(doc_id, vector)` pairs already committed for
    /// `field`.
    ///
    /// Ordered by ascending doc_id for determinism —
    /// [`get_vectors_by_field`](crate::vector::reader::VectorIndexReader::get_vectors_by_field)
    /// returns vectors in sealed-segment (newest-generation-first) order,
    /// not doc_id order, so this sorts before truncating (Issue #920:
    /// mirrors the `laurus train pq-codebook` JSONL path's "first N,
    /// deterministic" sampling semantics, just drawn from committed
    /// segments instead of a training file).
    ///
    /// # Arguments
    ///
    /// * `field` - Vector field to sample. An unknown or vector-less
    ///   field yields an empty `Vec`, not an error (matches
    ///   `get_vectors_by_field`'s own convention).
    /// * `limit` - Maximum number of pairs to return. `None` returns
    ///   every committed vector for the field.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining the reader or reading the field's
    /// vectors fails.
    pub fn sample_field_vectors(
        &self,
        field: &str,
        limit: Option<usize>,
    ) -> Result<Vec<(u64, Vector)>> {
        let reader = self.index.reader()?;
        let mut vectors = reader.get_vectors_by_field(field)?;
        vectors.sort_unstable_by_key(|(doc_id, _)| *doc_id);
        if let Some(limit) = limit {
            vectors.truncate(limit);
        }
        Ok(vectors)
    }

    /// Get the storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        self.index.storage()
    }

    /// Close the store.
    pub async fn close(&self) -> Result<()> {
        *self.writer_cache.lock().await = None;
        *self.searcher_cache.write() = None;
        self.index.close()
    }

    /// Check if the store is closed.
    pub fn is_closed(&self) -> bool {
        self.index.is_closed()
    }

    /// Get the embedder.
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        self.index.embedder()
    }

    /// Get the last processed WAL sequence number.
    pub fn last_wal_seq(&self) -> u64 {
        self.index.last_wal_seq()
    }

    /// Set the last processed WAL sequence number.
    ///
    /// Note: This method doesn't return Result for Engine compatibility.
    /// Errors are silently ignored.
    pub fn set_last_wal_seq(&self, seq: u64) {
        let _ = self.index.set_last_wal_seq(seq);
    }

    /// Add a dynamically-added vector field, creating its sub-index when
    /// needed and registering a field-specific embedder.
    ///
    /// Issue [#948](https://github.com/mosuka/laurus/issues/948): when the
    /// underlying index is a
    /// [`MultiFieldVectorIndex`](crate::vector::index::multi_field::MultiFieldVectorIndex)
    /// (`supports_dynamic_fields() == true`), a brand-new field needs its
    /// own sub-index created here BEFORE any vectors for it can be
    /// written -- its writer rejects unknown field names outright rather
    /// than silently collapsing them into another field's data (the bug
    /// this whole index type exists to prevent). Index types without field
    /// boundaries (the single-index
    /// [`Self::with_index_type_config`] construction) already accept any
    /// field name as a record-level tag with no schema change needed, so
    /// `supports_dynamic_fields()` is `false` there and this step is
    /// skipped.
    ///
    /// If the underlying index's embedder is a
    /// [`PerFieldEmbedder`](crate::embedding::per_field::PerFieldEmbedder),
    /// this method registers the given embedder for the specified field.
    /// The writer and searcher caches are invalidated afterwards.
    ///
    /// # Arguments
    ///
    /// * `name` - The vector field name
    /// * `vector_opt` - The field's schema-level vector option (dimension,
    ///   distance metric, index kind), used to build its sub-index config
    ///   when one must be created.
    /// * `embedder` - Optional field-specific embedder to register
    ///
    /// # Errors
    ///
    /// Returns an error if creating the field's sub-index fails (e.g. the
    /// field already exists, or a storage I/O error).
    pub async fn add_field(
        &self,
        name: &str,
        vector_opt: &crate::vector::core::field::FieldOption,
        embedder: Option<Arc<dyn crate::embedding::embedder::Embedder>>,
    ) -> Result<()> {
        if self.index.supports_dynamic_fields() {
            let deletion_config = self
                .config
                .as_ref()
                .map(|c| c.deletion_config.clone())
                .unwrap_or_default();
            let field_config = self::config::build_field_index_config(
                vector_opt,
                self.index.embedder(),
                &deletion_config,
            );
            self.index.add_field(name, field_config)?;
        }

        self.base_weights.write().insert(
            name.to_string(),
            Self::sanitize_base_weight(vector_opt.base_weight()),
        );

        if let Some(field_embedder) = embedder {
            let index_embedder = self.index.embedder();
            if let Some(pfe) = index_embedder
                .as_any()
                .downcast_ref::<crate::embedding::per_field::PerFieldEmbedder>()
            {
                pfe.add_embedder(name, field_embedder);
            }
        }

        // Invalidate caches so the next writer/searcher uses updated config.
        // Commit the cached writer first and KEEP it on failure (#882
        // review / #875): its buffered mutations are the only in-process
        // copy, and the segmented pending WAL checkpoint may already cover
        // them — a bare drop would let a later successful commit's
        // checkpoint hide the loss from recovery. A commit failure here is
        // intentionally not propagated (pre-existing behavior): leaving
        // the writer cached makes the next `commit()` retry and surface
        // it instead.
        {
            let mut writer_guard = self.writer_cache.lock().await;
            if let Some(writer) = writer_guard.as_mut()
                && writer.has_pending_changes()
                && writer.commit().is_err()
            {
                *self.searcher_cache.write() = None;
                return Ok(());
            }
            *writer_guard = None;
        }
        *self.searcher_cache.write() = None;
        Ok(())
    }

    /// Remove a field from the vector store.
    ///
    /// Unregisters the field from routing (Issue #948:
    /// [`VectorIndex::remove_field`], a no-op for index types without
    /// field boundaries) and any field-specific embedder from the
    /// `PerFieldEmbedder`, then invalidates writer/searcher caches.
    /// Existing vector data in the index is not deleted -- re-adding the
    /// field with the same name recovers it.
    ///
    /// # Arguments
    ///
    /// * `name` - The vector field name to remove
    ///
    /// # Errors
    ///
    /// Returns an error if unregistering the field from the underlying
    /// index fails.
    pub async fn delete_field(&self, name: &str) -> Result<()> {
        // `purge: false` -- this method's existing "unregister only"
        // contract (documented above) is unchanged; physical deletion is
        // opt-in via `Self::rebuild_field` (Issue #1080).
        self.index.remove_field(name, false)?;

        self.base_weights.write().remove(name);

        // Remove the field-specific embedder from the PerFieldEmbedder if present.
        let index_embedder = self.index.embedder();
        if let Some(pfe) = index_embedder
            .as_any()
            .downcast_ref::<crate::embedding::per_field::PerFieldEmbedder>()
        {
            pfe.remove_embedder(name);
        }

        // Invalidate caches so the next writer/searcher uses updated config.
        // Same commit-then-drop-on-success guard as `add_field` (#882
        // review): never rely on a bare drop while buffered mutations are
        // the only in-process copy.
        {
            let mut writer_guard = self.writer_cache.lock().await;
            if let Some(writer) = writer_guard.as_mut()
                && writer.has_pending_changes()
                && writer.commit().is_err()
            {
                *self.searcher_cache.write() = None;
                return Ok(());
            }
            *writer_guard = None;
        }
        *self.searcher_cache.write() = None;
        Ok(())
    }

    /// Rebuild a field's on-disk data under a new type/option configuration
    /// (Issue #1080: `Engine::update_field`'s `Reindex`/`Destructive`
    /// classified vector changes).
    ///
    /// # Arguments
    ///
    /// * `name` - The vector field to rebuild. Must already exist.
    /// * `new_option` - The field's new schema-level vector option.
    /// * `embedder` - Optional field-specific embedder to register under
    ///   the new option (same as [`Self::add_field`]).
    /// * `purge` - `false` for a [`FieldChangeKind::Reindex`]-classified
    ///   change: rebuilds in place from the field's existing segments via
    ///   [`VectorIndex::rebuild_field`]. `true` for a
    ///   [`FieldChangeKind::Destructive`]-classified change: physically
    ///   discards the field's existing data ([`VectorIndex::remove_field`]
    ///   with `purge: true`) and recreates it empty under the new config
    ///   ([`VectorIndex::add_field`]) -- `dimension`/`distance`/`embedder`
    ///   changes cannot be rebuilt from already-encoded data.
    ///
    /// [`FieldChangeKind::Reindex`]: crate::engine::schema::FieldChangeKind::Reindex
    /// [`FieldChangeKind::Destructive`]: crate::engine::schema::FieldChangeKind::Destructive
    ///
    /// # Errors
    ///
    /// Returns an error if flushing a cached writer's pending changes
    /// fails, or if the underlying rebuild (`purge: false`) or purge +
    /// recreate (`purge: true`) fails. Unlike [`Self::add_field`]/
    /// [`Self::delete_field`], a flush failure is **propagated**, not
    /// cached for a later retry — the writer's buffered state is not a
    /// safe starting point for a rebuild, the same reasoning
    /// [`Self::optimize`] already applies.
    pub async fn rebuild_field(
        &self,
        name: &str,
        new_option: &crate::vector::core::field::FieldOption,
        embedder: Option<Arc<dyn Embedder>>,
        purge: bool,
    ) -> Result<()> {
        let deletion_config = self
            .config
            .as_ref()
            .map(|c| c.deletion_config.clone())
            .unwrap_or_default();
        let new_config = self::config::build_field_index_config(
            new_option,
            self.index.embedder(),
            &deletion_config,
        );

        let mut writer_guard = self.writer_cache.lock().await;
        let flush_result = match writer_guard.as_mut() {
            Some(writer) if writer.has_pending_changes() => writer.commit(),
            _ => Ok(()),
        };
        flush_result?;
        *writer_guard = None;

        if purge {
            self.index.remove_field(name, true)?;
            self.index.add_field(name, new_config)?;
        } else {
            self.index.rebuild_field(name, new_config)?;
        }

        self.base_weights.write().insert(
            name.to_string(),
            Self::sanitize_base_weight(new_option.base_weight()),
        );

        if let Some(field_embedder) = embedder {
            let index_embedder = self.index.embedder();
            if let Some(pfe) = index_embedder.as_any().downcast_ref::<PerFieldEmbedder>() {
                pfe.add_embedder(name, field_embedder);
            }
        }

        drop(writer_guard);
        *self.searcher_cache.write() = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    #[test]
    fn test_vectorstore_creation() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let config = VectorIndexTypeConfig::default();
        let store = VectorStore::with_index_type_config(storage, config).unwrap();

        assert!(!store.is_closed());
    }

    #[tokio::test]
    async fn test_vectorstore_close() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let config = VectorIndexTypeConfig::default();
        let store = VectorStore::with_index_type_config(storage, config).unwrap();

        assert!(!store.is_closed());
        store.close().await.unwrap();
        assert!(store.is_closed());
    }

    /// Build a `VectorStore` (Flat, Euclidean — no normalization, so the
    /// dequantized round-trip is easy to bound) and commit `n` vectors
    /// under `field`, one dimension of value `doc_id as f32` (so exact
    /// identity is trivially checkable modulo int8 quantization error).
    async fn store_with_committed_vectors(field: &str, doc_ids: &[u64]) -> VectorStore {
        use crate::vector::index::config::FlatIndexConfig;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexTypeConfig::Flat(FlatIndexConfig {
            dimension: 4,
            distance_metric: crate::vector::core::distance::DistanceMetric::Euclidean,
            ..Default::default()
        });
        let store = VectorStore::with_index_type_config(storage, config).unwrap();

        for &doc_id in doc_ids {
            let doc = Document::builder()
                .add_field(field, DataValue::Vector(vec![doc_id as f32; 4]))
                .build();
            store
                .upsert_document_by_internal_id(doc_id, doc)
                .await
                .unwrap();
        }
        store.commit().await.unwrap();
        store
    }

    /// Issue #920: `sample_field_vectors` must return committed vectors
    /// sorted by ascending doc_id, regardless of commit/insertion order —
    /// `get_vectors_by_field`'s underlying sealed-segment order is
    /// newest-generation-first, not doc_id order.
    #[tokio::test]
    async fn sample_field_vectors_orders_by_ascending_doc_id() {
        let store = store_with_committed_vectors("embedding", &[30, 10, 20]).await;

        let sampled = store.sample_field_vectors("embedding", None).unwrap();
        let ids: Vec<u64> = sampled.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    /// `limit` truncates to the first N by doc_id; `None` returns all.
    #[tokio::test]
    async fn sample_field_vectors_respects_limit() {
        let store = store_with_committed_vectors("embedding", &[5, 1, 3, 2, 4]).await;

        let limited = store.sample_field_vectors("embedding", Some(2)).unwrap();
        let ids: Vec<u64> = limited.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![1, 2]);

        let all = store.sample_field_vectors("embedding", None).unwrap();
        assert_eq!(all.len(), 5);
    }

    /// An unknown field yields an empty `Vec`, not an error — matches
    /// `get_vectors_by_field`'s own convention.
    #[tokio::test]
    async fn sample_field_vectors_returns_empty_for_unknown_field() {
        let store = store_with_committed_vectors("embedding", &[1, 2]).await;

        let sampled = store.sample_field_vectors("no_such_field", None).unwrap();
        assert!(sampled.is_empty());
    }

    // Issue #948: `extract_index_type_config` (which collapsed every field
    // down to whichever one came first out of a `HashMap`) was removed and
    // replaced by `VectorIndexConfig::field_index_configs`, which converts
    // EVERY field. Its regression coverage -- including the Issue #790
    // (rerank/quantizer propagation) and #794 (metric-conditional
    // normalization) cases these two tests used to cover -- now lives in
    // `vector::store::config::tests`.
}
