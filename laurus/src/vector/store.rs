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
use crate::embedding::embedder::{EmbedInput, EmbedRole, Embedder};
use crate::embedding::per_field::{PerFieldEmbedder, embed_field_tokens};
use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::token_pooling::{PooledTokens, PoolingBudget, TokenPooling, pooled_count};
use crate::vector::core::vector::Vector;
use crate::vector::index::VectorIndex;
use crate::vector::index::config::VectorIndexTypeConfig;
use crate::vector::index::factory::VectorIndexFactory;
use crate::vector::search::filter_set::FilterSet;
use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
use crate::vector::writer::{FieldVectors, VectorIndexWriter};

use self::config::VectorIndexConfig;
use self::request::{FieldSelector, QueryVector, VectorScoreMode, VectorSearchRequest};
use self::response::{VectorHit, VectorSearchResults, VectorStats};

/// What [`VectorStore::rebuild_field`] does with the field's entry in the
/// store's [`PerFieldEmbedder`] (Issue #1354).
///
/// Every variant is a no-op when the store's embedder is not a
/// `PerFieldEmbedder`.
#[derive(Debug)]
pub enum FieldEmbedderChange {
    /// Leave the field's registration, if any, as it is. Used when neither
    /// the old nor the new option names a schema embedder, so an embedder
    /// registered directly through `EngineBuilder::embedder` is kept.
    Keep,
    /// Register this embedder for the field, replacing any registration.
    Register(Arc<dyn Embedder>),
    /// Remove the field's registration, so the field falls back to the
    /// `PerFieldEmbedder`'s default embedder. Used when the old option
    /// names a schema embedder and the new one does not; this also removes
    /// an embedder registered directly for the field.
    Unregister,
}

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
    /// Bounds the documents pooled at the same time before the writer
    /// lock is taken (Issue #1429; see [`Self::embed_fields`]).
    pooling_budget: PoolingBudget,
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
            pooling_budget: PoolingBudget::for_this_machine(),
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
            pooling_budget: PoolingBudget::for_this_machine(),
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
    /// A thin wrapper over [`Self::embed_document`] followed by
    /// [`Self::write_vectors`] (Issue #1326): callers that must not mutate
    /// anything else until embedding is known to succeed — `Engine::index_internal`
    /// before its WAL append, and `Engine::recover` before replaying a
    /// record's other side effects — call the two phases directly instead.
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
        let vectors = self.embed_document(&doc).await?;
        self.write_vectors(doc_id, vectors).await
    }

    /// Embed every vector-field value in `doc`, with no lock held.
    ///
    /// This is Phase 1 of the former single-method `upsert_document_by_internal_id`
    /// (Issue #1326), split out so a caller can run it *before* taking any
    /// action that would be hard to undo — a WAL append, a prior version's
    /// deletion — so that a document this store cannot apply leaves no
    /// trace anywhere. Also validates the resulting vectors against this
    /// index's configured field shapes, which used to be checked only deep
    /// inside the writer, after such side effects had already happened.
    ///
    /// # Arguments
    ///
    /// * `doc` - The document whose vector-field values will be embedded.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] if a field's embedder does
    /// not support the given input type, if a value names a field this
    /// index has no configuration for, or if a vector's dimension or values
    /// are invalid for its field. Returns the embedder's own error if the
    /// embedding call itself fails (e.g. a remote embedder is unreachable).
    ///
    /// # Returns
    ///
    /// One `(field_name, vectors)` pair per vector field of `doc`: the
    /// field's single vector, or a MultiVector field's token vectors in
    /// order. Each field name is carried once rather than once per token
    /// vector (Issue #1406).
    pub async fn embed_document(&self, doc: &Document) -> Result<Vec<(String, Vec<Vector>)>> {
        let fields: Vec<(&str, &DataValue)> = doc
            .fields
            .iter()
            .map(|(name, val)| (name.as_str(), val))
            .collect();
        self.embed_raw(&fields).await
    }

    /// [`Self::embed_document`] for already-borrowed `(field_name, value)`
    /// pairs instead of a whole `Document`, followed by pooling every
    /// multi-vector field that pools (Issue #1429; see [`Self::prepare`]),
    /// so that [`Self::write_prepared`] only has to buffer the result.
    ///
    /// Lets a caller that already knows which of a larger document's fields
    /// are vector fields (the engine, via its schema) embed them without
    /// first cloning those fields' values into a throwaway vector-only
    /// `Document` (Issue #1416) — for a `MultiVector` field, that clone
    /// copies every token's `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// The same as [`Self::embed_document`]'s.
    pub(crate) async fn embed_fields(
        &self,
        fields: &[(&str, &DataValue)],
    ) -> Result<Vec<FieldVectors>> {
        let embedded = self.embed_raw(fields).await?;
        Ok(self.prepare(embedded).await)
    }

    /// Embed and validate `(field_name, value)` pairs: the body of
    /// [`Self::embed_document`].
    async fn embed_raw(&self, fields: &[(&str, &DataValue)]) -> Result<Vec<(String, Vec<Vector>)>> {
        // Phase 1: Embed all fields OUTSIDE the lock.
        // This allows multiple concurrent upserts to perform embedding in parallel
        // rather than being serialized by the writer Mutex.
        let embedder = self.index.embedder();
        let late_interaction_fields = self.index.late_interaction_fields();
        let mut embedded_fields: Vec<(String, Vec<Vector>)> = Vec::with_capacity(fields.len());

        for (field_name, value) in fields {
            let field_name = *field_name;
            let value = *value;
            // A multi-vector field takes exactly a token-vector array, and
            // only a multi-vector field takes one (Issue #1177). Checked
            // here as well as in the engine's coercion: this store is public,
            // and a WAL record written before a field changed kind can carry
            // the old shape.
            let multi_vector = late_interaction_fields.contains(field_name);
            let vectors = match value {
                DataValue::VectorArray(vectors) if multi_vector => {
                    Self::check_token_vector_count(field_name, vectors.len())?;
                    vectors.iter().map(|v| Vector::new(v.clone())).collect()
                }
                DataValue::VectorArray(_) => {
                    return Err(LaurusError::invalid_argument(format!(
                        "field '{field_name}' is not a MultiVector field; only MultiVector \
                         fields accept an array of token vectors"
                    )));
                }
                DataValue::Text(text) if multi_vector => {
                    Self::embed_tokens(&embedder, field_name, EmbedInput::Text(text)).await?
                }
                DataValue::Bytes(bytes, mime) if multi_vector => {
                    Self::embed_tokens(
                        &embedder,
                        field_name,
                        EmbedInput::Bytes(bytes, mime.as_deref()),
                    )
                    .await?
                }
                _ if multi_vector => {
                    return Err(LaurusError::invalid_argument(format!(
                        "MultiVector field '{field_name}' accepts only an array of token vectors \
                         or a value for its token-level embedder"
                    )));
                }
                DataValue::Vector(v) => vec![Vector::new(v.clone())],
                DataValue::Text(_) | DataValue::Bytes(_, _) => {
                    vec![Self::embed_value(&*embedder, field_name, value).await?]
                }
                _ => continue,
            };
            embedded_fields.push((field_name.to_string(), vectors));
        }

        // Validate every resulting vector against this index's configured
        // field shapes before returning (Issue #1326): a dimension mismatch,
        // an unknown field, or NaN/infinity used to be caught only inside
        // the writer's `validate_vectors`, by which point the WAL append and
        // any prior-version deletion had already happened. `field_dimensions()`
        // is empty for stores built via `with_index_type_config` (no field
        // boundaries to check against, e.g. standalone tests/benches) — the
        // writer's own validation remains the backstop for those, as today.
        let dims = self.index.field_dimensions();
        if !dims.is_empty() {
            for (field_name, vectors) in &embedded_fields {
                let Some(&expected_dim) = dims.get(field_name) else {
                    return Err(LaurusError::invalid_argument(format!(
                        "unknown vector field '{field_name}': no index configured for it"
                    )));
                };
                for vector in vectors {
                    if vector.dimension() != expected_dim {
                        return Err(LaurusError::invalid_argument(format!(
                            "vector for field '{field_name}' has dimension {}, expected {expected_dim}",
                            vector.dimension()
                        )));
                    }
                    if !vector.is_valid() {
                        return Err(LaurusError::invalid_argument(format!(
                            "vector for field '{field_name}' contains invalid values (NaN or infinity)"
                        )));
                    }
                }
            }
        }

        Ok(embedded_fields)
    }

    /// Write already-embedded vectors for `doc_id` (Phase 2 of the former
    /// single-method `upsert_document_by_internal_id`, Issue #1326).
    ///
    /// Pools the token vectors of every multi-vector field that pools (see
    /// [`Self::prepare`]), then acquires the writer lock, deletes any
    /// existing vectors for `doc_id`, and adds the given ones. Callers are
    /// expected to have produced `fields` via [`Self::embed_document`],
    /// which has already validated them; this phase performs no further
    /// validation of its own.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID.
    /// * `fields` - `(field_name, vectors)` pairs as returned by
    ///   [`Self::embed_document`], already embedded and validated.
    ///
    /// # Errors
    ///
    /// Returns an error if obtaining/creating the writer fails, or if
    /// deleting the existing document or adding the new vectors fails.
    pub async fn write_vectors(
        &self,
        doc_id: u64,
        fields: Vec<(String, Vec<Vector>)>,
    ) -> Result<()> {
        let prepared = self.prepare(fields).await;
        self.write_prepared(doc_id, prepared).await
    }

    /// [`Self::write_vectors`] for fields [`Self::embed_fields`] has already
    /// embedded and pooled: only this step holds the writer lock.
    ///
    /// # Errors
    ///
    /// The same as [`Self::write_vectors`]'s.
    pub(crate) async fn write_prepared(
        &self,
        doc_id: u64,
        fields: Vec<FieldVectors>,
    ) -> Result<()> {
        // Phase 2: Acquire lock and write pre-computed vectors (fast, sync-only).
        let mut guard = self.writer_cache.lock().await;
        if guard.is_none() {
            *guard = Some(self.index.writer()?);
        }
        let writer = guard.as_mut().unwrap();
        writer.delete_document(doc_id)?;
        // One call for the whole document, so a field the writer does not
        // know (e.g. removed concurrently by `delete_field`) rejects every
        // field rather than leaving the others buffered.
        writer.add_prepared_document_vectors(doc_id, fields)?;

        Ok(())
    }

    /// Pool, before the writer lock is taken, the token vectors of every
    /// multi-vector field whose index pools (Issue #1429), so that
    /// concurrent puts no longer pool one after another under the lock.
    ///
    /// The writer keeps the pooled vectors only when the field still pools
    /// the same way when it buffers them, and pools the embedded vectors
    /// itself otherwise, so a field changed in between is handled exactly
    /// as before. Pooling never fails a document: whatever goes wrong here
    /// leaves the field without pooled vectors.
    async fn prepare(&self, fields: Vec<(String, Vec<Vector>)>) -> Vec<FieldVectors> {
        let mut prepared = Vec::with_capacity(fields.len());
        for (field, vectors) in fields {
            let pooled = match self.index.token_pooling(&field) {
                Some(pooling) => self.pool_before_writer(pooling, &vectors).await,
                None => None,
            };
            prepared.push(FieldVectors {
                field,
                vectors,
                pooled,
            });
        }
        prepared
    }

    /// Pool one field's token vectors within [`Self::pooling_budget`]. Under
    /// a tokio runtime (the `native` feature), the pooling runs on the
    /// blocking pool, so documents pooled at the same time do not hold up
    /// the runtime's workers; elsewhere it runs inline.
    async fn pool_before_writer(
        &self,
        pooling: TokenPooling,
        vectors: &[Vector],
    ) -> Option<PooledTokens> {
        if pooled_count(vectors.len(), pooling.factor()) == vectors.len() {
            return None;
        }
        let permit = self
            .pooling_budget
            .acquire(vectors.len(), pooling.dimension())
            .await;
        #[cfg(feature = "native")]
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            // Clones the rows' `Arc`s, not their values. The permit moves
            // along, so a put cancelled while it waits still holds the
            // budget until the pooling ends.
            let vectors = vectors.to_vec();
            return runtime
                .spawn_blocking(move || {
                    let _permit = permit;
                    pooling.apply(&vectors)
                })
                .await
                .ok()
                .flatten();
        }
        let _permit = permit;
        pooling.apply(vectors)
    }

    /// Embed a multi-vector field's value into token vectors with the
    /// field's token-level embedder, as a document (Issue #1349).
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] when the field has no
    /// token-level embedder or the embedder produces no or too many
    /// vectors, and the embedder's error when embedding fails.
    async fn embed_tokens(
        embedder: &Arc<dyn Embedder>,
        field_name: &str,
        input: EmbedInput<'_>,
    ) -> Result<Vec<Vector>> {
        let tokens = embed_field_tokens(
            embedder,
            field_name,
            std::slice::from_ref(&input),
            EmbedRole::Document,
        )
        .await?
        .swap_remove(0);
        Self::check_token_vector_count(field_name, tokens.len())?;
        Ok(tokens)
    }

    /// Reject a token-vector array that is empty or longer than a document
    /// may hold; each vector's dimension and values are checked with every
    /// other vector in [`Self::embed_document`].
    fn check_token_vector_count(field_name: &str, count: usize) -> Result<()> {
        use crate::vector::core::field::MultiVectorOption;

        if count == 0 {
            return Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': a multi-vector value needs at least one vector"
            )));
        }
        if count > MultiVectorOption::MAX_VECTORS_PER_DOCUMENT {
            return Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': a multi-vector value holds at most {} vectors, got {count}",
                MultiVectorOption::MAX_VECTORS_PER_DOCUMENT
            )));
        }
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
    /// underlying delete operation fails. The cached searcher is dropped on
    /// every return of the soft-delete path, as in [`Self::commit`] (Issue
    /// #1435): `MultiFieldVectorIndex::soft_delete_document` keeps marking
    /// the other fields after one fails, so a failure can still leave the
    /// document marked deleted (Issue #1445).
    pub async fn delete_document_by_internal_id(&self, doc_id: u64) -> Result<()> {
        // Prefer logical (soft) deletion when the index supports it (Issue
        // #624): mark the deletion bitmap and invalidate the searcher cache so
        // the next search filters the document out via the deletion-aware
        // traversal (Issue #665). This avoids the full graph rebuild that the
        // writer-side delete triggers, and applies equally to updates (the
        // engine deletes the old internal id, which is monotonic and never
        // reused, before adding the new version).
        //
        // The cache is dropped whatever the outcome (Issue #1445): a failure
        // in one field does not undo markings already applied to others, and
        // a field with no bitmap yet fixes it as `None` when its searcher is
        // built, so a cached searcher would not see a deletion applied after
        // it was built until the cache is dropped.
        if self.index.supports_soft_delete() {
            let result = self.index.soft_delete_document(doc_id);
            *self.searcher_cache.write() = None;
            return result;
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
    /// compaction, storage sync, or index refresh fails. The cached searcher
    /// is dropped on every return, so the next search sees whatever a
    /// failed commit had already published (Issue #1435).
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
        let finished = match ladder_result {
            Ok(compacted) => {
                if compacted || !self.index.retain_writer_after_commit() {
                    *writer_guard = None;
                }
                Ok(())
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
                Err(e)
            }
        };
        drop(writer_guard);
        let finished = finished
            // Sync storage to ensure all file metadata (creation, rename,
            // size) is flushed to disk. This is critical on Windows where
            // directory listings and file visibility may be cached until the
            // directory is synced.
            .and_then(|()| self.index.storage().sync())
            .and_then(|()| self.index.refresh());
        // On the error paths too (Issue #1435): the flush above, or a
        // compaction that failed partway, may already have changed the
        // segment set, and a searcher fixes its segment set when it is
        // built, so keeping it would hide those changes from every search
        // until the next successful commit.
        *self.searcher_cache.write() = None;
        finished
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
    /// optimization fails. The cached searcher is dropped on every return,
    /// as in [`Self::commit`] (Issue #1435).
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
        // records may already be inside the pending checkpoint's range. It
        // drops the cached searcher, as `commit` does (Issue #1435): the
        // multi-field writer commits the other fields after one fails, so a
        // failed flush may still have sealed segments.
        if let Err(e) = flush_result {
            drop(writer_guard);
            *self.searcher_cache.write() = None;
            return Err(e);
        }
        // Drop the cache on success — optimize() rewrites the index through
        // a fresh writer and clears the deletion bitmap, so a retained
        // writer would resurrect reclaimed vectors (#864).
        *writer_guard = None;
        let optimized = self.index.optimize();
        drop(writer_guard);
        // Whatever the outcome (Issue #1435): the flush may have sealed new
        // segments, and an optimization that failed partway may already have
        // replaced some.
        *self.searcher_cache.write() = None;
        optimized
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
    /// which pre-faults the vector rows read in place from every segment's
    /// mapped file into the OS page cache where applicable (`Scalar8Bit` and
    /// `Float16` rows of HNSW, Flat and IVF fields in `Mmap` mode). This
    /// moves the searcher-construction and page-fault latency off the first
    /// query.
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

    /// Select the top `limit` hits by descending score, breaking ties by
    /// ascending doc id so that identical requests return the same order on
    /// every run (Issue #1456) -- matching the multi-query path and the
    /// late-interaction rescore, which already tiebreak this way.
    fn select_top_k_by_score(mut hits: Vec<VectorHit>, limit: usize) -> Vec<VectorHit> {
        let cmp = |a: &VectorHit, b: &VectorHit| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        };
        let limit = limit.min(hits.len());
        if limit > 0 && limit < hits.len() {
            hits.select_nth_unstable_by(limit - 1, cmp);
            hits.truncate(limit);
            hits.sort_unstable_by(cmp);
        } else if !hits.is_empty() {
            hits.sort_unstable_by(cmp);
        }
        hits
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
                return Err(crate::error::LaurusError::internal(
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
                    // Multi-vector fields are not search targets (Issue
                    // #1177), so a prefix never expands to one.
                    let late_interaction = self.index.late_interaction_fields();
                    dims.into_keys()
                        .filter(|name| !late_interaction.contains(name))
                        .collect()
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
            let field_less = index_queries[0].field_name.is_none();

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

            // Issue #1343: a field-less query run on a single-index store
            // (`with_index_type_config`) is not routed through
            // `MultiFieldFanoutSearcher` (which already de-duplicates), so
            // the underlying searcher's unfiltered `field_name: None` scan
            // can still emit one hit per `(doc_id, field)` pair. Collapse to
            // one hit per document (best score wins) before ranking, so
            // `limit` counts documents.
            if field_less {
                use std::collections::hash_map::Entry;
                let mut best: std::collections::HashMap<u64, VectorHit> =
                    std::collections::HashMap::with_capacity(hits.len());
                for hit in hits {
                    match best.entry(hit.doc_id) {
                        Entry::Occupied(mut e) => {
                            if hit.score > e.get().score {
                                e.insert(hit);
                            }
                        }
                        Entry::Vacant(e) => {
                            e.insert(hit);
                        }
                    }
                }
                hits = best.into_values().collect();
            }

            // Use partial sort for top-K selection when the result set is larger
            // than the requested limit.
            let hits = Self::select_top_k_by_score(hits, request.params.limit);

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
        // Issue #1343: a single query vector must contribute at most one
        // term per document (that is what `score_mode`'s doc comment on
        // `VectorScoreMode` promises), but a field-less query — or one
        // routed to a single-index store — can still return the same
        // document through more than one field. Collapse each query's own
        // results to its best-similarity hit per document first, then fold
        // across queries by `score_mode`. Reused per query to avoid an
        // allocation per iteration.
        let mut per_query_best: std::collections::HashMap<u64, f32> =
            std::collections::HashMap::new();
        for (weight, results) in query_weights.iter().zip(per_query_results) {
            per_query_best.clear();
            for result in results.results {
                if let Some(ref allowed) = filter_set
                    && !allowed.contains(result.doc_id)
                {
                    continue;
                }
                if result.similarity < request.params.min_score {
                    continue;
                }
                per_query_best
                    .entry(result.doc_id)
                    .and_modify(|best| {
                        if result.similarity > *best {
                            *best = result.similarity;
                        }
                    })
                    .or_insert(result.similarity);
            }
            for (&doc_id, &similarity) in per_query_best.iter() {
                let weighted_score = similarity * weight;
                let entry = all_hits.entry(doc_id).or_insert(0.0);
                match request.params.score_mode {
                    VectorScoreMode::WeightedSum | VectorScoreMode::LateInteraction => {
                        // WeightedSum: sum of similarity * weight across all query vectors.
                        // LateInteraction: for each query vector, find the max similarity
                        // across document vectors, then sum. Each query vector contributes
                        // a single best match per document (collapsed above), so this is
                        // equivalent to WeightedSum.
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

    /// A point-in-time view of the committed token vectors of the
    /// multi-vector field `field` (Issue #1177).
    ///
    /// # Arguments
    ///
    /// * `field` - A multi-vector field.
    ///
    /// # Returns
    ///
    /// The snapshot; documents committed or deleted afterwards are not
    /// reflected in its segments (deletions may be).
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] when `field` is unknown or
    /// is not a multi-vector field, and an error when a segment fails to
    /// load.
    pub fn multi_vector_snapshot(
        &self,
        field: &str,
    ) -> Result<crate::vector::index::multivector::MultiVectorSnapshot> {
        self.index.multi_vector_snapshot(field)?.ok_or_else(|| {
            LaurusError::invalid_argument(format!("field '{field}' is not a MultiVector field"))
        })
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

        self.unregister_field_embedder(name);

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

    /// Remove `name`'s field-specific embedder from the store's
    /// `PerFieldEmbedder`, if the store has one.
    fn unregister_field_embedder(&self, name: &str) {
        let index_embedder = self.index.embedder();
        if let Some(pfe) = index_embedder.as_any().downcast_ref::<PerFieldEmbedder>() {
            pfe.remove_embedder(name);
        }
    }

    /// Rebuild a field's on-disk data under a new type/option configuration
    /// (Issue #1080: `Engine::update_field`'s `Reindex`/`Destructive`
    /// classified vector changes).
    ///
    /// # Arguments
    ///
    /// * `name` - The vector field to rebuild. Must already exist.
    /// * `new_option` - The field's new schema-level vector option.
    /// * `embedder` - What to do with the field's registration in the
    ///   store's `PerFieldEmbedder` (see [`FieldEmbedderChange`]). Applied
    ///   only after the rebuild succeeds, so a failed rebuild leaves the
    ///   registration as it was.
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
    /// [`Self::optimize`] already applies. The cached searcher is dropped on
    /// every return, as in [`Self::commit`] (Issue #1435); the field's base
    /// weight and embedder registration change only when the rebuild
    /// succeeds.
    pub async fn rebuild_field(
        &self,
        name: &str,
        new_option: &crate::vector::core::field::FieldOption,
        embedder: FieldEmbedderChange,
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

        // `purge` discards the field's data before re-creating it, so an
        // option the new index would reject must fail before that.
        match new_option {
            crate::vector::core::field::FieldOption::Flat(opt) => opt.validate(name)?,
            crate::vector::core::field::FieldOption::Ivf(opt) => opt.validate(name)?,
            crate::vector::core::field::FieldOption::MultiVector(opt) => opt.validate(name)?,
            crate::vector::core::field::FieldOption::Hnsw(_) => {}
        }

        let mut writer_guard = self.writer_cache.lock().await;
        let flush_result = match writer_guard.as_mut() {
            Some(writer) if writer.has_pending_changes() => writer.commit(),
            _ => Ok(()),
        };
        // A failed flush may still have sealed other fields' segments, as in
        // `optimize` (Issue #1435).
        if let Err(e) = flush_result {
            drop(writer_guard);
            *self.searcher_cache.write() = None;
            return Err(e);
        }
        *writer_guard = None;

        let rebuilt = if purge {
            self.index
                .remove_field(name, true)
                .and_then(|()| self.index.add_field(name, new_config))
        } else {
            self.index.rebuild_field(name, new_config)
        };

        if rebuilt.is_ok() {
            self.base_weights.write().insert(
                name.to_string(),
                Self::sanitize_base_weight(new_option.base_weight()),
            );

            match embedder {
                FieldEmbedderChange::Keep => {}
                FieldEmbedderChange::Register(field_embedder) => {
                    let index_embedder = self.index.embedder();
                    if let Some(pfe) = index_embedder.as_any().downcast_ref::<PerFieldEmbedder>() {
                        pfe.add_embedder(name, field_embedder);
                    }
                }
                FieldEmbedderChange::Unregister => self.unregister_field_embedder(name),
            }
        }

        drop(writer_guard);
        // Whatever the outcome (Issue #1435): the flush may have sealed new
        // segments, and a purge whose re-creation failed has already
        // unregistered the field, which a searcher built before it would
        // still search.
        *self.searcher_cache.write() = None;
        rebuilt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    /// Issue #1456: ties (identical score) must be ordered by ascending
    /// doc id on every call, matching the multi-query path and the
    /// late-interaction rescore, which already tiebreak this way. Exercises
    /// both the partial-sort branch (`limit < hits.len()`) and the full-sort
    /// branch.
    #[test]
    fn select_top_k_by_score_breaks_ties_by_ascending_doc_id() {
        let doc_ids: Vec<u64> = vec![
            17, 3, 42, 9, 25, 1, 38, 12, 30, 6, 44, 2, 19, 27, 33, 5, 21, 8, 36, 14,
        ];
        let hits: Vec<VectorHit> = doc_ids
            .iter()
            .map(|&doc_id| VectorHit {
                doc_id,
                score: 1.0,
                field_hits: vec![],
            })
            .collect();

        let top = VectorStore::select_top_k_by_score(hits.clone(), 7);
        let mut expected: Vec<u64> = doc_ids.clone();
        expected.sort_unstable();
        expected.truncate(7);
        let actual: Vec<u64> = top.iter().map(|h| h.doc_id).collect();
        assert_eq!(actual, expected);

        let all = VectorStore::select_top_k_by_score(hits, doc_ids.len());
        let mut expected_all = doc_ids;
        expected_all.sort_unstable();
        let actual_all: Vec<u64> = all.iter().map(|h| h.doc_id).collect();
        assert_eq!(actual_all, expected_all);
    }

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

    /// The [`VectorIndex`] call that a [`FailingIndex`] fails (Issue #1435,
    /// `SoftDelete` added for Issue #1445).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum IndexCall {
        PersistDeletions,
        MaybeAutoCompact,
        Refresh,
        Optimize,
        RemoveField,
        AddField,
        RebuildField,
        SoftDelete,
    }

    /// A [`VectorIndex`] that delegates every call to `inner`, except that
    /// the armed call fails once without reaching `inner`, and that
    /// [`VectorIndex::storage`] returns `storage` (Issue #1435).
    #[derive(Debug)]
    struct FailingIndex {
        inner: Box<dyn VectorIndex>,
        armed: Arc<parking_lot::Mutex<Option<IndexCall>>>,
        storage: Arc<dyn Storage>,
    }

    impl FailingIndex {
        fn check(&self, call: IndexCall) -> Result<()> {
            let mut armed = self.armed.lock();
            if *armed == Some(call) {
                *armed = None;
                return Err(LaurusError::internal(format!("injected {call:?} failure")));
            }
            Ok(())
        }
    }

    impl VectorIndex for FailingIndex {
        fn reader(&self) -> Result<Arc<dyn crate::vector::reader::VectorIndexReader>> {
            self.inner.reader()
        }
        fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
            self.inner.writer()
        }
        fn storage(&self) -> &Arc<dyn Storage> {
            &self.storage
        }
        fn close(&self) -> Result<()> {
            self.inner.close()
        }
        fn is_closed(&self) -> bool {
            self.inner.is_closed()
        }
        fn stats(&self) -> Result<crate::vector::index::VectorIndexStats> {
            self.inner.stats()
        }
        fn optimize(&self) -> Result<()> {
            self.check(IndexCall::Optimize)?;
            self.inner.optimize()
        }
        fn refresh(&self) -> Result<()> {
            self.check(IndexCall::Refresh)?;
            self.inner.refresh()
        }
        fn retain_writer_after_commit(&self) -> bool {
            self.inner.retain_writer_after_commit()
        }
        fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>> {
            self.inner.searcher()
        }
        fn embedder(&self) -> Arc<dyn Embedder> {
            self.inner.embedder()
        }
        fn last_wal_seq(&self) -> u64 {
            self.inner.last_wal_seq()
        }
        fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
            self.inner.set_last_wal_seq(seq)
        }
        fn supports_soft_delete(&self) -> bool {
            self.inner.supports_soft_delete()
        }
        fn soft_delete_document(&self, doc_id: u64) -> Result<()> {
            // The real call first (Issue #1445): a genuine partial failure
            // inside `MultiFieldVectorIndex::soft_delete_document`'s
            // `try_all` still marks whichever fields succeed before
            // reporting the first error, so the injected failure below must
            // not undo that real effect.
            self.inner.soft_delete_document(doc_id)?;
            self.check(IndexCall::SoftDelete)
        }
        fn persist_deletions(&self) -> Result<()> {
            self.check(IndexCall::PersistDeletions)?;
            self.inner.persist_deletions()
        }
        fn maybe_auto_compact(&self) -> Result<bool> {
            self.check(IndexCall::MaybeAutoCompact)?;
            self.inner.maybe_auto_compact()
        }
        fn supports_dynamic_fields(&self) -> bool {
            self.inner.supports_dynamic_fields()
        }
        fn add_field(&self, name: &str, config: VectorIndexTypeConfig) -> Result<()> {
            self.check(IndexCall::AddField)?;
            self.inner.add_field(name, config)
        }
        fn remove_field(&self, name: &str, purge: bool) -> Result<()> {
            self.check(IndexCall::RemoveField)?;
            self.inner.remove_field(name, purge)
        }
        fn rebuild_field(&self, name: &str, new_config: VectorIndexTypeConfig) -> Result<()> {
            self.check(IndexCall::RebuildField)?;
            self.inner.rebuild_field(name, new_config)
        }
        fn field_dimensions(&self) -> std::collections::BTreeMap<String, usize> {
            self.inner.field_dimensions()
        }
        fn multi_vector_snapshot(
            &self,
            field: &str,
        ) -> Result<Option<crate::vector::index::multivector::MultiVectorSnapshot>> {
            self.inner.multi_vector_snapshot(field)
        }
        fn late_interaction_fields(&self) -> std::collections::BTreeSet<String> {
            self.inner.late_interaction_fields()
        }

        fn token_pooling(&self, field: &str) -> Option<TokenPooling> {
            self.inner.token_pooling(field)
        }
    }

    /// A [`Storage`] over `inner` that fails the next `sync`, or the next
    /// `create_output` of a file whose name starts with the armed prefix
    /// (Issue #1435).
    #[derive(Debug)]
    struct FailingStorage {
        inner: Arc<dyn Storage>,
        fail_sync: std::sync::atomic::AtomicBool,
        fail_create_prefix: parking_lot::Mutex<Option<&'static str>>,
    }

    impl FailingStorage {
        fn new(inner: Arc<dyn Storage>) -> Self {
            Self {
                inner,
                fail_sync: std::sync::atomic::AtomicBool::new(false),
                fail_create_prefix: parking_lot::Mutex::new(None),
            }
        }
    }

    impl Storage for FailingStorage {
        fn open_input(&self, name: &str) -> Result<Box<dyn crate::storage::StorageInput>> {
            self.inner.open_input(name)
        }
        fn create_output(&self, name: &str) -> Result<Box<dyn crate::storage::StorageOutput>> {
            let mut prefix = self.fail_create_prefix.lock();
            if prefix.is_some_and(|prefix| name.starts_with(prefix)) {
                *prefix = None;
                return Err(LaurusError::storage(format!(
                    "injected failure creating {name}"
                )));
            }
            self.inner.create_output(name)
        }
        fn create_output_append(
            &self,
            name: &str,
        ) -> Result<Box<dyn crate::storage::StorageOutput>> {
            self.inner.create_output_append(name)
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
        fn metadata(&self, name: &str) -> Result<crate::storage::FileMetadata> {
            self.inner.metadata(name)
        }
        fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
            self.inner.rename_file(old_name, new_name)
        }
        fn create_temp_output(
            &self,
            prefix: &str,
        ) -> Result<(String, Box<dyn crate::storage::StorageOutput>)> {
            self.inner.create_temp_output(prefix)
        }
        fn sync(&self) -> Result<()> {
            if self
                .fail_sync
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                return Err(LaurusError::storage("injected sync failure"));
            }
            self.inner.sync()
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
        fn lock_manager(&self) -> Option<Arc<dyn crate::storage::LockManager>> {
            self.inner.lock_manager()
        }
    }

    /// A store from [`store_with_stale_searcher`] and the switches that make
    /// it fail.
    struct StaleSearcherFixture {
        store: VectorStore,
        /// Fails a call from the store to its index.
        armed: Arc<parking_lot::Mutex<Option<IndexCall>>>,
        /// The storage the index writes its files to.
        index_storage: Arc<FailingStorage>,
        /// The storage the store itself syncs ([`VectorIndex::storage`]).
        /// The index syncs `index_storage` instead, for example after saving
        /// a manifest, so failing this one fails only the store's sync.
        store_storage: Arc<FailingStorage>,
    }

    impl StaleSearcherFixture {
        /// Asserts that the armed failure was hit, so the operation failed
        /// where the test meant it to.
        fn assert_failure_hit(&self) {
            assert!(self.armed.lock().is_none(), "the index failure was not hit");
            assert!(
                self.index_storage.fail_create_prefix.lock().is_none(),
                "the create failure was not hit"
            );
            assert!(
                !self
                    .store_storage
                    .fail_sync
                    .load(std::sync::atomic::Ordering::SeqCst),
                "the sync failure was not hit"
            );
        }
    }

    fn flat_option() -> crate::vector::core::field::FlatOption {
        crate::vector::core::field::FlatOption {
            dimension: 4,
            ..Default::default()
        }
    }

    /// Buffers `doc_id` with a vector in both fields `a` and `b`.
    async fn put_doc(store: &VectorStore, doc_id: u64) {
        let vector = DataValue::Vector(vec![doc_id as f32; 4]);
        let doc = Document::builder()
            .add_field("a", vector.clone())
            .add_field("b", vector)
            .build();
        store
            .upsert_document_by_internal_id(doc_id, doc)
            .await
            .unwrap();
    }

    /// The ids a search of `field` returns, in ascending order.
    fn search_ids(store: &VectorStore, field: &str) -> Result<Vec<u64>> {
        use crate::vector::search::searcher::{VectorSearchParams, VectorSearchQuery};

        let request = VectorSearchRequest {
            query: VectorSearchQuery::Vectors(vec![QueryVector {
                vector: Vector::new(vec![1.0; 4]),
                weight: 1.0,
                fields: Some(vec![field.to_string()]),
            }]),
            params: VectorSearchParams {
                limit: 10,
                ..Default::default()
            },
        };
        let mut ids: Vec<u64> = store
            .search(request)?
            .hits
            .iter()
            .map(|hit| hit.doc_id)
            .collect();
        ids.sort_unstable();
        Ok(ids)
    }

    /// Issue #1435 fixture: a store over two Flat fields `a` and `b` that
    /// has committed doc 1, cached a searcher over that commit, and buffered
    /// doc 2 in its writer. The searcher fixed its segment set when it was
    /// built, so it cannot see doc 2 once doc 2 is committed.
    async fn store_with_stale_searcher() -> StaleSearcherFixture {
        let memory: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index_storage = Arc::new(FailingStorage::new(memory));
        let config = VectorIndexConfig::builder()
            .add_field("a", flat_option())
            .unwrap()
            .add_field("b", flat_option())
            .unwrap()
            .build()
            .unwrap();
        let VectorStore {
            index,
            writer_cache,
            searcher_cache,
            config,
            base_weights,
            pooling_budget,
        } = VectorStore::new(index_storage.clone(), config).unwrap();
        let armed = Arc::new(parking_lot::Mutex::new(None));
        let store_storage = Arc::new(FailingStorage::new(index_storage.clone()));
        let store = VectorStore {
            index: Box::new(FailingIndex {
                inner: index,
                armed: armed.clone(),
                storage: store_storage.clone(),
            }),
            writer_cache,
            searcher_cache,
            config,
            base_weights,
            pooling_budget,
        };

        put_doc(&store, 1).await;
        store.commit().await.unwrap();
        assert_eq!(search_ids(&store, "b").unwrap(), vec![1]);
        put_doc(&store, 2).await;

        StaleSearcherFixture {
            store,
            armed,
            index_storage,
            store_storage,
        }
    }

    /// Issue #1435 — a commit whose deletion persistence fails after the
    /// writer's commit must not keep the searcher built before that commit.
    #[tokio::test]
    async fn commit_failing_to_persist_deletions_drops_stale_searcher() {
        let fixture = store_with_stale_searcher().await;
        *fixture.armed.lock() = Some(IndexCall::PersistDeletions);
        assert!(fixture.store.commit().await.is_err());
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "b").unwrap(), vec![1, 2]);
    }

    /// Issue #1435 — as above, for a failed auto-compaction.
    #[tokio::test]
    async fn commit_failing_to_auto_compact_drops_stale_searcher() {
        let fixture = store_with_stale_searcher().await;
        *fixture.armed.lock() = Some(IndexCall::MaybeAutoCompact);
        assert!(fixture.store.commit().await.is_err());
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "b").unwrap(), vec![1, 2]);
    }

    /// Issue #1435 — as above, for a failed storage sync.
    #[tokio::test]
    async fn commit_failing_to_sync_drops_stale_searcher() {
        let fixture = store_with_stale_searcher().await;
        fixture
            .store_storage
            .fail_sync
            .store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(fixture.store.commit().await.is_err());
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "b").unwrap(), vec![1, 2]);
    }

    /// Issue #1435 — as above, for a failed index refresh.
    #[tokio::test]
    async fn commit_failing_to_refresh_drops_stale_searcher() {
        let fixture = store_with_stale_searcher().await;
        *fixture.armed.lock() = Some(IndexCall::Refresh);
        assert!(fixture.store.commit().await.is_err());
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "b").unwrap(), vec![1, 2]);
    }

    /// Issue #1435 — an optimization that fails after the writer's commit
    /// must not keep the searcher built before that commit.
    #[tokio::test]
    async fn optimize_failure_after_flush_drops_stale_searcher() {
        let fixture = store_with_stale_searcher().await;
        *fixture.armed.lock() = Some(IndexCall::Optimize);
        assert!(fixture.store.optimize().await.is_err());
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "b").unwrap(), vec![1, 2]);
    }

    /// Issue #1435 — a failed flush in `optimize` must drop the cached
    /// searcher, as `commit` does: the multi-field writer commits field `a`
    /// even though field `b` fails, so doc 2 is published in `a`.
    #[tokio::test]
    async fn optimize_flush_failure_drops_stale_searcher() {
        let fixture = store_with_stale_searcher().await;
        *fixture.index_storage.fail_create_prefix.lock() = Some("b/segment_");
        assert!(fixture.store.optimize().await.is_err());
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "a").unwrap(), vec![1, 2]);
    }

    /// Issue #1435 — a rebuild (`purge: false`) that fails after the writer's
    /// commit must not keep the searcher built before that commit.
    #[tokio::test]
    async fn rebuild_failure_after_flush_drops_stale_searcher() {
        let fixture = store_with_stale_searcher().await;
        *fixture.armed.lock() = Some(IndexCall::RebuildField);
        let rebuilt = fixture
            .store
            .rebuild_field("a", &flat_option().into(), FieldEmbedderChange::Keep, false)
            .await;
        assert!(rebuilt.is_err());
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "b").unwrap(), vec![1, 2]);
    }

    /// Issue #1435 — as above, for a purge that fails to remove the field.
    #[tokio::test]
    async fn purge_failing_to_remove_drops_stale_searcher() {
        let fixture = store_with_stale_searcher().await;
        *fixture.armed.lock() = Some(IndexCall::RemoveField);
        let rebuilt = fixture
            .store
            .rebuild_field("a", &flat_option().into(), FieldEmbedderChange::Keep, true)
            .await;
        assert!(rebuilt.is_err());
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "b").unwrap(), vec![1, 2]);
    }

    /// Issue #1435 — a purge that removed the field but failed to re-create
    /// it must not keep the searcher built before: besides missing doc 2,
    /// that searcher would still search the purged field.
    #[tokio::test]
    async fn purge_failing_to_recreate_drops_stale_searcher() {
        let fixture = store_with_stale_searcher().await;
        *fixture.armed.lock() = Some(IndexCall::AddField);
        let rebuilt = fixture
            .store
            .rebuild_field("a", &flat_option().into(), FieldEmbedderChange::Keep, true)
            .await;
        assert!(rebuilt.is_err());
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "b").unwrap(), vec![1, 2]);
        let err = search_ids(&fixture.store, "a").unwrap_err();
        assert!(
            err.to_string().contains("unknown vector field 'a'"),
            "unexpected error: {err}"
        );
    }

    /// Issue #1435 — a failed flush in `rebuild_field` must drop the cached
    /// searcher, as in `optimize_flush_failure_drops_stale_searcher`.
    #[tokio::test]
    async fn rebuild_flush_failure_drops_stale_searcher() {
        let fixture = store_with_stale_searcher().await;
        *fixture.index_storage.fail_create_prefix.lock() = Some("b/segment_");
        let rebuilt = fixture
            .store
            .rebuild_field("a", &flat_option().into(), FieldEmbedderChange::Keep, false)
            .await;
        assert!(rebuilt.is_err());
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "a").unwrap(), vec![1, 2]);
    }

    /// Issue #1457 — a purge to a Flat or IVF option with a product
    /// quantizer is rejected before the field's data is discarded.
    #[tokio::test]
    async fn rebuild_rejects_product_quantization_on_flat_and_ivf_before_purging() {
        use crate::vector::core::field::{FieldOption, IvfOption};
        use crate::vector::core::quantization::QuantizationMethod;

        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexConfig::builder()
            .add_field("a", flat_option())
            .unwrap()
            .add_field("b", flat_option())
            .unwrap()
            .build()
            .unwrap();
        let store = VectorStore::new(storage, config).unwrap();
        put_doc(&store, 1).await;
        store.commit().await.unwrap();

        let pq = QuantizationMethod::ProductQuantization { subvector_count: 2 };
        let options = [
            ("Flat", FieldOption::Flat(flat_option().quantizer(pq))),
            (
                "IVF",
                FieldOption::Ivf(IvfOption {
                    dimension: 4,
                    quantizer: pq,
                    ..Default::default()
                }),
            ),
        ];
        for (kind, option) in options {
            let err = store
                .rebuild_field("a", &option, FieldEmbedderChange::Keep, true)
                .await
                .unwrap_err();
            let expected = format!("field 'a': {kind} does not support quantizer");
            assert!(
                matches!(&err, LaurusError::InvalidArgument(m) if m.contains(&expected)),
                "unexpected error: {err:?}"
            );
            assert_eq!(search_ids(&store, "a").unwrap(), vec![1], "{kind}");
        }
    }

    /// Issue #1445 fixture: a store over two Flat fields `a` and `b` with doc
    /// 1 committed and a searcher already cached over that commit, like
    /// [`store_with_stale_searcher`] -- but doc 1 is inserted through the
    /// index's own writer directly, bypassing
    /// [`VectorStore::upsert_document_by_internal_id`]'s delete-before-add
    /// dance ([`write_vectors`](VectorStore::write_vectors) always calls
    /// `writer.delete_document` first, which eagerly creates a field's
    /// deletion bitmap even on its very first write, via
    /// `SegmentedCore::mark_deleted`'s `create_if_missing: true`). Both
    /// fields therefore have no deletion bitmap at all when the searcher
    /// below is built and cached -- `SegmentedIndex::searcher` captures
    /// `None` -- which is the condition the Issue's bug depends on: once a
    /// field's bitmap is created later, a `Some(Arc<DeletionBitmap>)`
    /// captured by an earlier searcher is a live, shared reference that
    /// sees later mutations on its own, but a searcher that captured `None`
    /// has no such reference to see the bitmap created after it.
    async fn store_with_searcher_cached_before_any_bitmap() -> StaleSearcherFixture {
        let memory: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index_storage = Arc::new(FailingStorage::new(memory));
        let config = VectorIndexConfig::builder()
            .add_field("a", flat_option())
            .unwrap()
            .add_field("b", flat_option())
            .unwrap()
            .build()
            .unwrap();
        let VectorStore {
            index,
            writer_cache,
            searcher_cache,
            config,
            base_weights,
            pooling_budget,
        } = VectorStore::new(index_storage.clone(), config).unwrap();
        let armed = Arc::new(parking_lot::Mutex::new(None));
        let store_storage = Arc::new(FailingStorage::new(index_storage.clone()));
        let store = VectorStore {
            index: Box::new(FailingIndex {
                inner: index,
                armed: armed.clone(),
                storage: store_storage.clone(),
            }),
            writer_cache,
            searcher_cache,
            config,
            base_weights,
            pooling_budget,
        };

        let vector = Vector::new(vec![1.0; 4]);
        let mut writer = store.index.writer().unwrap();
        writer
            .add_vectors(vec![
                (1, "a".to_string(), vector.clone()),
                (1, "b".to_string(), vector),
            ])
            .unwrap();
        writer.commit().unwrap();

        assert_eq!(search_ids(&store, "b").unwrap(), vec![1]);

        StaleSearcherFixture {
            store,
            armed,
            index_storage,
            store_storage,
        }
    }

    /// Issue #1445 — `MultiFieldVectorIndex::soft_delete_document`'s
    /// `try_all` keeps marking every field it can before returning the first
    /// error, so a soft delete that fails still leaves doc 1 marked deleted.
    /// Field `b` has no deletion bitmap before this call (see
    /// [`store_with_searcher_cached_before_any_bitmap`]), so the searcher
    /// cached before it captured `None` and will not see the mark unless
    /// `delete_document_by_internal_id` drops the cache on this error path
    /// too, same as `commit`/`optimize`/`rebuild_field` already do (Issue
    /// #1435).
    #[tokio::test]
    async fn soft_delete_failure_drops_stale_searcher() {
        let fixture = store_with_searcher_cached_before_any_bitmap().await;
        *fixture.armed.lock() = Some(IndexCall::SoftDelete);
        assert!(
            fixture
                .store
                .delete_document_by_internal_id(1)
                .await
                .is_err()
        );
        fixture.assert_failure_hit();

        assert_eq!(search_ids(&fixture.store, "b").unwrap(), Vec::<u64>::new());
    }

    /// A store with a multi-vector field `tokens` (4 dimensions, Cosine,
    /// pool factor 2) and a Flat field `flat`.
    fn pooling_store() -> VectorStore {
        use crate::vector::core::field::{FieldOption, MultiVectorOption};

        let config = VectorIndexConfig::builder()
            .add_field(
                "tokens",
                FieldOption::MultiVector(MultiVectorOption::new(4).pool_factor(2)),
            )
            .unwrap()
            .add_field("flat", flat_option())
            .unwrap()
            .build()
            .unwrap();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        VectorStore::new(storage, config).unwrap()
    }

    fn token_array(count: usize) -> DataValue {
        DataValue::VectorArray(
            (0..count)
                .map(|i| vec![1.0 + i as f32, 0.5, -0.25 * i as f32, 0.125])
                .collect(),
        )
    }

    /// Issue #1429: embedding pools a pooling multi-vector field's
    /// document before the writer is taken, exactly as
    /// `TokenPooling::apply` does, and leaves alone other fields and
    /// documents that pooling would not shorten.
    #[tokio::test]
    async fn embed_fields_pools_documents_that_pooling_shortens() {
        let store = pooling_store();
        let (long, short) = (token_array(9), token_array(2));
        let flat = DataValue::Vector(vec![1.0, 0.0, 0.0, 0.0]);
        let fields = store
            .embed_fields(&[("tokens", &long), ("flat", &flat)])
            .await
            .unwrap();
        assert_eq!(fields[0].field, "tokens");
        let pooled = fields[0].pooled.as_ref().expect("9 vectors pool to 5");
        let expected = TokenPooling::new(4, 2, true)
            .apply(&fields[0].vectors)
            .unwrap();
        assert_eq!(pooled.pooling(), expected.pooling());
        assert_eq!(pooled.vectors(), expected.vectors());
        assert_eq!(fields[1].field, "flat");
        assert!(fields[1].pooled.is_none());

        let fields = store.embed_fields(&[("tokens", &short)]).await.unwrap();
        assert!(fields[0].pooled.is_none());
    }

    /// Without a tokio runtime the pooling runs inline instead of on the
    /// runtime's blocking pool.
    #[test]
    fn embed_fields_pools_without_a_tokio_runtime() {
        let store = pooling_store();
        let long = token_array(9);
        let fields = futures::executor::block_on(store.embed_fields(&[("tokens", &long)])).unwrap();
        assert!(fields[0].pooled.is_some());
    }
}
