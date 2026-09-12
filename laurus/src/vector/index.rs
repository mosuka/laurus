//! Vector indexing module for building and maintaining vector indexes.
//!
//! This module handles all vector index construction, maintenance, and optimization:
//! - Building HNSW, Flat, and IVF indexes
//! - Text embedding generation
//! - Vector quantization and compression
//! - Index optimization and maintenance

pub mod config;
pub mod factory;
pub mod field;
pub mod flat;
pub mod format;
pub mod hnsw;
pub mod io;
pub mod ivf;
pub mod multi_field;
pub mod pq_codebook;
pub mod pq_fastscan_avx2;
#[cfg(feature = "pq-fastscan")]
pub mod pq_fastscan_io;
pub mod pq_fastscan_neon;
pub mod pq_fastscan_storage;
pub mod pq_io;
pub mod pq_storage;
pub mod quantized_io;
pub mod quantized_segment;
pub mod quantized_storage;
pub mod rerank_sidecar;
pub mod rerank_storage;
pub mod segment;
pub mod segmented_field;
pub mod storage;
pub mod wal;

use std::sync::Arc;

use parking_lot::RwLock;

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use crate::vector::index::config::{
    FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, VectorIndexTypeConfig,
};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::writer::VectorIndexWriter;

/// Trait for vector index implementations.
///
/// This trait defines the low-level interface for individual vector indexes
/// (Flat, HNSW, IVF, etc.). Each index type implements this trait to provide
/// reader/writer access and basic lifecycle management.
///
/// This is analogous to [`crate::lexical::index::LexicalIndex`] in the lexical module.
/// For high-level, document-centric operations, see
/// [`crate::vector::store::VectorStore`] which manages multiple
/// vector fields and handles document-level operations.
pub trait VectorIndex: Send + Sync + std::fmt::Debug {
    /// Get a reader for this index.
    ///
    /// Returns a reader that can be used to query the index.
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>>;

    /// Get a writer for this index.
    ///
    /// Returns a writer that can be used to add or update vectors.
    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>>;

    /// Get the storage backend for this index.
    ///
    /// Returns a reference to the underlying storage.
    fn storage(&self) -> &Arc<dyn Storage>;

    /// Close the index and release resources.
    ///
    /// This should flush any pending writes and release all resources.
    /// Uses interior mutability for thread-safe access.
    fn close(&self) -> Result<()>;

    /// Check if the index is closed.
    ///
    /// Returns true if the index has been closed.
    fn is_closed(&self) -> bool;

    /// Get index statistics.
    ///
    /// Returns statistics about the index such as vector count, dimension, etc.
    fn stats(&self) -> Result<VectorIndexStats>;

    /// Optimize the index.
    ///
    /// Performs index optimization to improve query performance.
    /// Uses interior mutability for thread-safe access.
    fn optimize(&self) -> Result<()>;

    /// Refresh the index metadata from storage.
    ///
    /// Should be called after external writes (e.g., by a Writer) to ensure
    /// the index state is up-to-date.
    fn refresh(&self) -> Result<()> {
        Ok(())
    }

    /// Whether a store may keep its cached writer across `commit()` calls
    /// (Issue #572 / #864) instead of dropping it, so the first upsert after
    /// a commit does not reload the whole index from storage.
    ///
    /// Retention is sound only when **both** hold for this index type:
    ///
    /// 1. The writer's post-commit in-memory state is equivalent to the file
    ///    it just wrote (idempotent `finalize()`, non-consuming `write()`,
    ///    incremental graph/buffer reuse), so subsequent commits from the
    ///    retained writer produce the same bytes a freshly loaded writer
    ///    would.
    /// 2. Every disk mutation that bypasses the writer — compaction via
    ///    [`Self::maybe_auto_compact`], [`Self::optimize`] — invalidates the
    ///    store's writer cache, otherwise a stale retained writer would
    ///    rewrite physically reclaimed (deleted) vectors back into the index
    ///    with no deletion bitmap marking them.
    ///
    /// Defaults to `false` (the store drops the writer on commit, today's
    /// behavior); `HnswIndex` opts in after auditing both conditions.
    fn retain_writer_after_commit(&self) -> bool {
        false
    }

    /// Create a searcher tailored for this index implementation.
    ///
    /// Returns a boxed [`VectorIndexSearcher`] capable of executing search/count operations.
    fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>>;

    /// Get the embedder associated with this index.
    ///
    /// Returns the embedder used to convert documents to vectors.
    fn embedder(&self) -> Arc<dyn Embedder>;

    /// Get the last processed WAL sequence number.
    fn last_wal_seq(&self) -> u64 {
        0
    }

    /// Set the last processed WAL sequence number.
    fn set_last_wal_seq(&self, _seq: u64) -> Result<()> {
        Ok(())
    }

    /// Whether this index supports logical (soft) deletion.
    ///
    /// When `true`, callers should route deletions through
    /// [`Self::soft_delete_document`] (mark a deletion bitmap, filtered at
    /// search time, physically reclaimed by [`Self::optimize`]) instead of the
    /// writer-side deletion path that rebuilds the whole index. Defaults to
    /// `false`, so existing index types keep their current behaviour.
    fn supports_soft_delete(&self) -> bool {
        false
    }

    /// Logically delete a document by its internal ID (Issue #624).
    ///
    /// Marks `doc_id` in the index's deletion bitmap so that subsequent
    /// searches exclude it, without rebuilding the underlying graph. The
    /// vector is physically removed only by a later [`Self::optimize`]
    /// (compaction). Implementations must persist durably no later than the
    /// next [`Self::persist_deletions`] call.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID to mark as deleted.
    ///
    /// # Errors
    ///
    /// The default implementation returns an error because soft deletion is
    /// unsupported; callers should gate on [`Self::supports_soft_delete`].
    fn soft_delete_document(&self, _doc_id: u64) -> Result<()> {
        Err(crate::error::LaurusError::InvalidOperation(
            "soft deletion is not supported by this index".to_string(),
        ))
    }

    /// Persist any pending logical deletions to storage (Issue #624).
    ///
    /// Called by the store on commit so the deletion bitmap survives restarts.
    /// Defaults to a no-op for indexes that do not buffer soft deletions.
    ///
    /// # Errors
    ///
    /// Returns an error if writing the deletion bitmap to storage fails.
    fn persist_deletions(&self) -> Result<()> {
        Ok(())
    }

    /// Compact the index automatically if the deletion ratio warrants it (Issue
    /// #782).
    ///
    /// Called by the store after a commit. Implementations that buffer logical
    /// deletions (Issue #624) should physically reclaim them via
    /// [`Self::optimize`] when the deleted fraction crosses their configured
    /// threshold, so tombstones do not accumulate unboundedly. Returns whether
    /// a compaction actually ran. Defaults to a no-op (`Ok(false)`).
    ///
    /// # Errors
    ///
    /// Returns an error if the triggered compaction fails.
    fn maybe_auto_compact(&self) -> Result<bool> {
        Ok(false)
    }

    /// Whether this index supports adding/removing vector fields after
    /// construction (Issue [#948](https://github.com/mosuka/laurus/issues/948)).
    ///
    /// Defaults to `false`; only
    /// [`MultiFieldVectorIndex`](crate::vector::index::multi_field::MultiFieldVectorIndex)
    /// (one independent sub-index per field) can grow a genuinely new field
    /// without disturbing the others. A monolithic single-field index
    /// (Flat/HNSW/IVF `open_or_create`) has no field boundary to add to.
    fn supports_dynamic_fields(&self) -> bool {
        false
    }

    /// Add a new vector field to this index (Issue #948: dynamic schema
    /// evolution).
    ///
    /// # Errors
    ///
    /// The default implementation always errors; callers must gate on
    /// [`Self::supports_dynamic_fields`] first.
    fn add_field(&self, _name: &str, _config: VectorIndexTypeConfig) -> Result<()> {
        Err(LaurusError::InvalidOperation(
            "this index type does not support adding fields dynamically".to_string(),
        ))
    }

    /// Remove a vector field from this index's routing (Issue #948: dynamic
    /// schema evolution).
    ///
    /// With `purge: false`, this only unregisters the field -- mirrors
    /// [`crate::vector::store::VectorStore`]'s existing "unregister only"
    /// contract for `delete_field`, so the data can be recovered by
    /// re-adding the field with the same name.
    ///
    /// With `purge: true` (Issue #1080: `Engine::update_field` rebuilding a
    /// field under a new type), the field's on-disk data is also
    /// physically deleted, so a same-name re-add under a different type
    /// cannot misread the old bytes.
    ///
    /// Defaults to `Ok(())` (a no-op): index types without field boundaries
    /// have nothing to unregister or purge.
    fn remove_field(&self, _name: &str, _purge: bool) -> Result<()> {
        Ok(())
    }

    /// Rebuild an existing vector field's on-disk data under a new
    /// configuration, in place (Issue #1080: `Engine::update_field`'s
    /// `Reindex`-classified vector changes -- e.g. HNSW `m`/
    /// `ef_construction`, any index kind's `quantizer`/`rerank_storage`,
    /// IVF `n_clusters`).
    ///
    /// The field keeps its identity and physical storage location; only
    /// the segments are rewritten under `new_config`. Implementations must
    /// leave the field's existing data completely untouched if the rebuild
    /// fails partway through (e.g. by writing new segments before
    /// atomically publishing them, the way [`Self::optimize`] already
    /// does).
    ///
    /// Only valid for parameters that do not change the meaning of
    /// already-written vectors (dimension, distance metric, and embedder
    /// must be unchanged -- callers gate this via
    /// `laurus::engine::schema::classify_change`). Rebuilding under a
    /// different `dimension`/`distance`/`embedder` is a
    /// [`crate::engine::schema::FieldChangeKind::Destructive`] change
    /// instead, handled by the caller as remove-then-add.
    ///
    /// # Errors
    ///
    /// The default implementation always errors; callers must gate on
    /// [`Self::supports_dynamic_fields`] first. Implementations also error
    /// if no field named `name` is registered, or if the rebuild itself
    /// fails.
    fn rebuild_field(&self, _name: &str, _new_config: VectorIndexTypeConfig) -> Result<()> {
        Err(LaurusError::InvalidOperation(
            "this index type does not support rebuilding fields dynamically".to_string(),
        ))
    }

    /// The configured vector dimension of every field, keyed by field name
    /// (Issue #948: multi-field stats/routing need per-field dimensions
    /// without paying for a full `reader()`/`stats()` round trip).
    ///
    /// Defaults to empty: index types without field boundaries report their
    /// single dimension through [`Self::stats`] instead.
    fn field_dimensions(&self) -> std::collections::BTreeMap<String, usize> {
        std::collections::BTreeMap::new()
    }
}

/// Statistics about a vector index.
#[derive(Debug, Clone)]
pub struct VectorIndexStats {
    /// Number of vectors in the index.
    pub vector_count: u64,

    /// Dimension of vectors.
    pub dimension: usize,

    /// Total size of the index in bytes.
    pub total_size: u64,

    /// Number of deleted vectors.
    pub deleted_count: u64,

    /// Last modified time (seconds since epoch).
    pub last_modified: u64,
}

// Add imports for readers
use crate::vector::index::flat::reader::FlatVectorIndexReader;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::ivf::reader::IvfIndexReader;

/// Internal implementation for managing vector index lifecycle.
///
/// This structure wraps a vector index writer and manages its state.
/// For most use cases, prefer using `VectorStore` which provides a higher-level interface.
///
/// # Note
///
/// This is an internal implementation detail. The public API for vector indexes
/// is defined by the `VectorIndex` trait and `VectorIndexFactory`.
pub struct ManagedVectorIndex {
    config: VectorIndexTypeConfig,
    builder: Arc<RwLock<Box<dyn VectorIndexWriter>>>,
    is_finalized: Arc<RwLock<bool>>,
    storage: Option<Arc<dyn Storage>>,
    index_path: Arc<RwLock<Option<String>>>,
}

impl ManagedVectorIndex {
    /// Create a new vector index with the given configuration and storage.
    ///
    /// # Arguments
    ///
    /// * `config` - Vector index type configuration (Flat, HNSW, or IVF)
    /// * `storage` - Storage backend (MemoryStorage, FileStorage, etc.)
    /// * `path` - Base path/name for the index files
    pub fn new(
        mut config: VectorIndexTypeConfig,
        storage: Arc<dyn Storage>,
        path: impl Into<String>,
    ) -> Result<Self> {
        let path = path.into();
        // Resolve a configured shared PQ codebook (Issue #631) before
        // constructing the writer -- mirrors `factory.rs`'s
        // `dispatch_hnsw` resolution point for the `VectorIndexFactory`
        // path; this constructor is the other (bench/direct-use) path
        // into an HNSW writer.
        if let VectorIndexTypeConfig::HNSW(hnsw_config) = &mut config {
            hnsw_config.resolve_pq_codebook(storage.as_ref())?;
        }
        // Create builder based on config type
        let builder: Box<dyn VectorIndexWriter> = match &config {
            VectorIndexTypeConfig::Flat(flat_config) => {
                let writer_config = Self::default_writer_config();
                Box::new(flat::writer::FlatIndexWriter::with_storage(
                    flat_config.clone(),
                    writer_config,
                    path.clone(),
                    storage.clone(),
                )?)
            }
            VectorIndexTypeConfig::HNSW(hnsw_config) => {
                let writer_config = Self::default_writer_config();
                Box::new(hnsw::writer::HnswIndexWriter::with_storage(
                    hnsw_config.clone(),
                    writer_config,
                    path.clone(),
                    storage.clone(),
                )?)
            }
            VectorIndexTypeConfig::IVF(ivf_config) => {
                let writer_config = Self::default_writer_config();
                Box::new(ivf::writer::IvfIndexWriter::with_storage(
                    ivf_config.clone(),
                    writer_config,
                    path.clone(),
                    storage.clone(),
                )?)
            }
        };

        Ok(Self {
            config,
            builder: Arc::new(RwLock::new(builder)),
            is_finalized: Arc::new(RwLock::new(false)),
            storage: Some(storage),
            index_path: Arc::new(RwLock::new(Some(path))),
        })
    }

    /// Helper to create a default writer config.
    fn default_writer_config() -> crate::vector::writer::VectorIndexWriterConfig {
        crate::vector::writer::VectorIndexWriterConfig::default()
    }

    /// Add vectors to the index.
    pub fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        let finalized = *self.is_finalized.read();
        if finalized {
            return Err(LaurusError::InvalidOperation(
                "Cannot add vectors to finalized index".to_string(),
            ));
        }

        let mut builder = self.builder.write();
        builder.add_vectors(vectors)?;
        Ok(())
    }

    /// Finalize the index construction.
    pub fn finalize(&mut self) -> Result<()> {
        let mut builder = self.builder.write();
        builder.finalize()?;
        *self.is_finalized.write() = true;
        Ok(())
    }

    /// Delete a document by its ID.
    pub fn delete_document(&self, doc_id: u64) -> Result<()> {
        let mut builder = self.builder.write();
        builder.delete_document(doc_id)
    }

    /// Get the configuration.
    pub fn config(&self) -> &VectorIndexTypeConfig {
        &self.config
    }

    /// Get build progress (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        let builder = self.builder.read();
        builder.progress()
    }

    /// Get estimated memory usage.
    pub fn estimated_memory_usage(&self) -> usize {
        let builder = self.builder.read();
        builder.estimated_memory_usage()
    }

    /// Check if the index is finalized.
    pub fn is_finalized(&self) -> bool {
        *self.is_finalized.read()
    }

    /// Get vectors from this index.
    /// Returns a copy of all vectors stored in the index.
    pub fn vectors(&self) -> Result<Vec<(u64, String, Vector)>> {
        let finalized = *self.is_finalized.read();
        if !finalized {
            return Err(LaurusError::InvalidOperation(
                "Index must be finalized before accessing vectors".to_string(),
            ));
        }

        let builder = self.builder.read();
        Ok(builder.vectors().to_vec())
    }

    /// Write the index to storage.
    /// The index must be finalized before calling this method.
    pub fn write(&self) -> Result<()> {
        let finalized = *self.is_finalized.read();
        if !finalized {
            return Err(LaurusError::InvalidOperation(
                "Index must be finalized before writing".to_string(),
            ));
        }

        let builder = self.builder.read();
        if !builder.has_storage() {
            return Err(LaurusError::InvalidOperation(
                "Index was not created with storage support".to_string(),
            ));
        }

        builder.write()?;
        Ok(())
    }

    /// Check if this index has storage configured.
    pub fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    /// Create a reader for this index.
    /// Returns a boxed VectorIndexReader that can be used for searching.
    pub fn reader(&self) -> Result<Arc<dyn crate::vector::reader::VectorIndexReader>> {
        let finalized = *self.is_finalized.read();
        if !finalized {
            return Err(LaurusError::InvalidOperation(
                "Index must be finalized before creating a reader".to_string(),
            ));
        }

        // Try loading from storage if available (index was written via write()).
        if let Some(storage) = &self.storage {
            let path_guard = self.index_path.read();
            if let Some(path) = &*path_guard {
                let storage_result: Result<Arc<dyn crate::vector::reader::VectorIndexReader>> =
                    match &self.config {
                        VectorIndexTypeConfig::Flat(c) => {
                            FlatVectorIndexReader::load(storage.clone(), path, c.distance_metric)
                                .map(|r| Arc::new(r) as _)
                        }
                        VectorIndexTypeConfig::HNSW(c) => {
                            HnswIndexReader::load(storage.clone(), path, c.distance_metric)
                                .map(|r| Arc::new(r) as _)
                        }
                        VectorIndexTypeConfig::IVF(c) => {
                            IvfIndexReader::load(storage.clone(), path, c.distance_metric)
                                .map(|r| Arc::new(r) as _)
                        }
                    };
                if let Ok(reader) = storage_result {
                    return Ok(reader);
                }
                // Fall through to in-memory reader if storage load fails
                // (e.g., finalize() was called but write() was not).
            }
        }

        // Create reader from in-memory vectors held by the writer.
        let vectors = self.vectors()?;
        let reader = crate::vector::reader::SimpleVectorReader::new(
            vectors,
            self.config.dimension(),
            self.config.distance_metric(),
        )?;
        Ok(Arc::new(reader))
    }
}
