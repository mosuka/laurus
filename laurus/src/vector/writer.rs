//! Vector index writer trait and common types.
//!
//! This module defines the `VectorIndexWriter` trait which all vector index writer
//! implementations must follow. The primary implementations are:
//! - `HnswIndexWriter` in the `hnsw` module for approximate nearest neighbor search
//! - `FlatIndexWriter` in the `flat` module for exact search
//! - `IvfIndexWriter` in the `ivf` module for memory-efficient search

use async_trait::async_trait;

use crate::error::Result;
use crate::vector::core::vector::Vector;
use serde::{Deserialize, Serialize};

/// Configuration for vector index writers common to all index types.
///
/// This configuration contains settings that are common across all vector index writer
/// implementations (Flat, HNSW, IVF), similar to `InvertedIndexWriterConfig` in the lexical module.
///
/// Type-specific settings (dimension, distance metric, HNSW parameters, etc.) are defined
/// in the respective index configs: `FlatIndexConfig`, `HnswIndexConfig`, `IvfIndexConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexWriterConfig {
    /// Maximum number of vectors to buffer before flushing to storage.
    pub max_buffered_vectors: usize,

    /// Maximum memory usage for buffering (in bytes).
    pub max_buffer_memory: usize,

    /// Segment name prefix.
    pub segment_prefix: String,

    /// Build index in parallel (when supported by the index type).
    pub parallel_build: bool,

    /// Memory limit for index construction (in bytes).
    /// If None, no explicit limit is enforced.
    pub memory_limit: Option<usize>,

    /// Auto-flush threshold: flush when buffer reaches this percentage (0.0-1.0).
    pub auto_flush_threshold: f32,
}

impl Default for VectorIndexWriterConfig {
    fn default() -> Self {
        Self {
            max_buffered_vectors: 10000,
            max_buffer_memory: 512 * 1024 * 1024, // 512 MB
            segment_prefix: "segment".to_string(),
            parallel_build: true,
            memory_limit: None,
            auto_flush_threshold: 0.9,
        }
    }
}

/// Trait for vector index writers.
///
/// This trait defines the common interface that all vector index writer implementations
/// must follow. Writers are responsible for:
/// - Building vector indexes from collections of vectors
/// - Adding vectors incrementally during construction
/// - Finalizing index construction
/// - Writing indexes to persistent storage
///
/// # Example
///
/// ```rust,no_run
/// use laurus::vector::index::hnsw::writer::HnswIndexWriter;
/// use laurus::vector::index::config::HnswIndexConfig;
/// use laurus::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
/// use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
/// use laurus::storage::StorageConfig;
/// use std::sync::Arc;
///
/// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
/// let index_config = HnswIndexConfig::default();
/// let writer_config = VectorIndexWriterConfig::default();
/// let mut writer = HnswIndexWriter::with_storage(index_config, writer_config, "test_index", storage).unwrap();
///
/// // Use VectorIndexWriter trait methods
/// // writer.add_vectors(vectors).unwrap();
/// // writer.finalize().unwrap();
/// // writer.write("my_index").unwrap();
/// ```
#[async_trait]
pub trait VectorIndexWriter: Send + Sync + std::fmt::Debug {
    /// Get the next available vector ID (for automatic ID assignment).
    fn next_vector_id(&self) -> u64;

    /// Build an index from a collection of vectors with field names.
    /// Each vector is a tuple of (vec_id, field_name, Vector).
    fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()>;

    /// Add vectors incrementally during construction.
    /// Each vector is a tuple of (vec_id, field_name, Vector).
    /// This allows field-specific vector search similar to lexical field search.
    fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()>;

    /// Finalize the index construction.
    fn finalize(&mut self) -> Result<()>;

    /// Get build progress (0.0 to 1.0).
    fn progress(&self) -> f32;

    /// Get estimated memory usage.
    fn estimated_memory_usage(&self) -> usize;

    /// Get access to the stored vectors with field names.
    /// Returns a reference to the vectors with their field names stored in the writer.
    fn vectors(&self) -> &[(u64, String, Vector)];

    /// Write the index to storage.
    /// This method must be called after finalize() to persist the index.
    fn write(&self) -> Result<()>;

    /// Check if this writer has storage configured.
    fn has_storage(&self) -> bool;

    /// Delete a document by its ID.
    ///
    /// This removes the document from the index buffer if it hasn't been finalized.
    fn delete_document(&mut self, doc_id: u64) -> Result<()>;

    /// Whether the writer holds changes that have not been written to storage
    /// yet (Issue #864).
    ///
    /// Unlike [`Self::pending_docs`] — which reports the number of *buffered
    /// documents* and is therefore `0` both for a clean just-committed writer
    /// and for a dirty writer whose buffer was emptied by deletions — this
    /// answers "would dropping the writer lose an uncommitted mutation?".
    /// Stores use it to skip the full index rewrite on a no-change commit of
    /// a retained writer, and to flush a dirty writer before dropping it in
    /// `optimize()`.
    ///
    /// The default implementation conservatively returns `true` (always
    /// commit). Writers that track a finalized flag should override it.
    fn has_pending_changes(&self) -> bool {
        true
    }

    /// Delete documents matching a metadata field value.
    ///
    /// This removes matching documents from the index buffer regarding the filtering criteria.
    /// Returns the number of documents removed.
    fn delete_documents(&mut self, field: &str, value: &str) -> Result<usize> {
        let _ = field;
        let _ = value;
        Err(crate::error::LaurusError::InvalidOperation(
            "delete_documents not implemented for this writer".to_string(),
        ))
    }

    /// Commit pending changes to the index.
    ///
    /// This method finalizes the index and writes it to storage.
    /// After commit, all changes are persisted and visible to readers.
    fn commit(&mut self) -> Result<()> {
        self.finalize()?;
        self.write()
    }

    /// Add a value to the index, embedding it automatically if supported.
    ///
    /// The default implementation only accepts
    /// [`DataValue::Vector`](crate::data::DataValue::Vector) and converts it
    /// directly into an index vector. For non-vector inputs (e.g., text or
    /// bytes), the writer must be wrapped in an `EmbeddingVectorIndexWriter`,
    /// which overrides this method to embed the payload before indexing.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID to associate the vector with.
    /// * `field_name` - The name of the field being indexed.
    /// * `value` - The data value to index. In the default implementation only
    ///   the `Vector` variant is accepted; all other variants return an error.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`](crate::error::LaurusError::InvalidArgument) if
    /// the value is not a `Vector` variant and auto-embedding is not supported.
    async fn add_value(
        &mut self,
        doc_id: u64,
        field_name: String,
        value: crate::data::DataValue,
    ) -> Result<()> {
        if let crate::data::DataValue::Vector(v) = value {
            self.add_vectors(vec![(doc_id, field_name, Vector::new(v))])
        } else {
            Err(crate::error::LaurusError::invalid_argument(
                "Auto-embedding not supported by this index writer. Wrap it in EmbeddingVectorIndexWriter.",
            ))
        }
    }

    /// Rollback pending changes.
    ///
    /// This method discards all pending changes that haven't been committed.
    fn rollback(&mut self) -> Result<()>;

    /// Get the number of pending documents not yet committed.
    fn pending_docs(&self) -> u64;

    /// Close the writer and release resources.
    fn close(&mut self) -> Result<()>;

    /// Check if the writer is closed.
    fn is_closed(&self) -> bool;

    /// Optimize the index.
    ///
    /// This method is optional and performs index-specific optimizations.
    /// Default implementation does nothing.
    fn optimize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Build a reader from the written index.
    ///
    /// This method allows creating a reader directly from the writer,
    /// enabling the "write-then-read" workflow used in hybrid search.
    fn build_reader(&self) -> Result<std::sync::Arc<dyn crate::vector::reader::VectorIndexReader>>;
}
