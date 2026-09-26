//! Lexical index writer trait and common types.
//!
//! This module defines the `LexicalIndexWriter` trait which all lexical index writer
//! implementations must follow. The primary implementation is `InvertedIndexWriter`.

use crate::error::Result;
use crate::lexical::core::analyzed::AnalyzedDocument;
use crate::lexical::core::document::Document;

/// Trait for lexical index writers.
///
/// This trait defines the common interface that all lexical index writer implementations
/// must follow. The primary implementation is
/// [`InvertedIndexWriter`](crate::lexical::index::inverted::writer::InvertedIndexWriter).
///
/// # Write Semantics
///
/// All write operations (`add_document`, `upsert_document`, `delete_document`, etc.)
/// are **batched** in an in-memory buffer. Changes are only persisted to storage when
/// [`commit()`](Self::commit) is called. Use [`rollback()`](Self::rollback) to discard
/// all pending changes without persisting.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync`. However, all mutating methods require
/// `&mut self`, so external synchronization (e.g., via `Mutex`) is needed if
/// shared across threads.
///
/// # Example
///
/// ```rust,no_run
/// use laurus::lexical::index::inverted::writer::{InvertedIndexWriter, InvertedIndexWriterConfig};
/// use laurus::lexical::writer::LexicalIndexWriter;
/// use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
/// use laurus::storage::StorageConfig;
/// use std::sync::Arc;
///
/// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
/// let config = InvertedIndexWriterConfig::default();
/// let mut writer = InvertedIndexWriter::new(storage, config).unwrap();
///
/// // Use LexicalIndexWriter trait methods
/// // writer.add_document(doc).unwrap();
/// // writer.commit().unwrap();
/// ```
pub trait LexicalIndexWriter: Send + Sync + std::fmt::Debug {
    /// Add a document to the index with automatic ID assignment.
    ///
    /// The document is analyzed (tokenized) and buffered in memory. It is not
    /// persisted until [`commit()`](Self::commit) is called.
    ///
    /// # Returns
    ///
    /// The automatically assigned document ID on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is closed or document analysis fails.
    fn add_document(&mut self, doc: Document) -> Result<u64>;

    /// Upsert a document to the index with a specific document ID.
    ///
    /// If a document with the given `doc_id` already exists, it is replaced.
    /// The document is analyzed and buffered in memory until [`commit()`](Self::commit).
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is closed or document analysis fails.
    fn upsert_document(&mut self, doc_id: u64, doc: Document) -> Result<()>;

    /// Add an already analyzed document to the index with automatic ID assignment.
    ///
    /// This allows adding pre-analyzed documents that were processed
    /// using [`DocumentParser`](crate::lexical::core::parser::DocumentParser) or
    /// from external tokenization systems. The document is buffered until
    /// [`commit()`](Self::commit).
    ///
    /// # Returns
    ///
    /// The automatically assigned document ID on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is closed.
    fn add_analyzed_document(&mut self, doc: AnalyzedDocument) -> Result<u64>;

    /// Upsert an already analyzed document to the index with a specific document ID.
    ///
    /// If a document with the given `doc_id` already exists, it is replaced.
    /// The document is buffered until [`commit()`](Self::commit).
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is closed.
    fn upsert_analyzed_document(&mut self, doc_id: u64, doc: AnalyzedDocument) -> Result<()>;

    /// Delete a document by ID.
    ///
    /// Removes the document from the in-memory buffer if it is still pending.
    /// For already-committed documents, the deletion is recorded and applied
    /// on the next [`commit()`](Self::commit).
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is closed.
    fn delete_document(&mut self, doc_id: u64) -> Result<()>;

    /// Commit all pending changes to the index.
    ///
    /// Flushes any buffered documents into a new segment and writes updated
    /// index metadata to storage. After a successful commit, the changes become
    /// visible to new readers/searchers.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is closed or if flushing/persisting fails.
    fn commit(&mut self) -> Result<()>;

    /// Rollback all pending changes.
    ///
    /// Discards every document added since the last commit — the in-memory
    /// buffer and any segment an automatic flush already wrote but no commit
    /// published — so the next commit publishes none of them. Deletions of
    /// already-committed documents are kept and persisted by the next commit;
    /// committed data is otherwise not affected, and the writer stays usable.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is closed.
    fn rollback(&mut self) -> Result<()>;

    /// Get the number of documents added since the last commit.
    ///
    /// Includes documents an automatic flush has already written to a
    /// segment that no commit has published yet, not only the in-memory
    /// buffer.
    fn pending_docs(&self) -> u64;

    /// Close the writer and release resources.
    ///
    /// After closing, all subsequent write operations will return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is already closed.
    fn close(&mut self) -> Result<()>;

    /// Check if the writer is closed.
    ///
    /// Returns `true` if [`close()`](Self::close) has been called.
    fn is_closed(&self) -> bool;

    /// Build a reader from the written index.
    ///
    /// This method allows creating a reader directly from the writer,
    /// enabling the "write-then-read" workflow used in hybrid search.
    ///
    /// # Returns
    ///
    /// An `Arc`-wrapped reader that can be shared across threads.
    ///
    /// # Errors
    ///
    /// Returns an error if the reader cannot be constructed.
    fn build_reader(
        &self,
    ) -> Result<std::sync::Arc<dyn crate::lexical::reader::LexicalIndexReader>>;

    /// Get the next available document ID.
    ///
    /// Returns the ID that will be assigned to the next document added
    /// via [`add_document()`](Self::add_document) or
    /// [`add_analyzed_document()`](Self::add_analyzed_document).
    fn next_doc_id(&self) -> u64;

    /// Find the internal document ID for a given term (field:value).
    ///
    /// Performs a near-real-time (NRT) lookup against both the in-memory buffer
    /// and committed segments. Returns the first matching document ID, or `None`
    /// if no document contains the given term in the specified field.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is closed.
    fn find_doc_id_by_term(&self, field: &str, term: &str) -> Result<Option<u64>>;

    /// Find all internal document IDs for a given term (field:value).
    ///
    /// Similar to [`find_doc_id_by_term()`](Self::find_doc_id_by_term) but returns
    /// all matching document IDs. This is useful for multi-chunk document lookups.
    /// Returns `None` if no documents match.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer is closed.
    fn find_doc_ids_by_term(&self, field: &str, term: &str) -> Result<Option<Vec<u64>>>;

    /// Set the last processed WAL sequence number.
    ///
    /// The default implementation is a no-op. Override this to track WAL progress.
    fn set_last_wal_seq(&mut self, _seq: u64) -> Result<()> {
        Ok(())
    }

    /// Whether this writer superseded the persisted copy of `doc_id` since
    /// the last commit — a deletion a searcher built before it cannot see
    /// yet. `false` again once a commit has published the deletions.
    ///
    /// The default implementation always returns `false`. Override this if the
    /// writer tracks pending deletions.
    fn is_updated_deleted(&self, _doc_id: u64) -> bool {
        false
    }

    /// Invalidate any cached view of the committed segments (Issue #864).
    ///
    /// Called by the store after an operation that rewrites segments behind a
    /// live writer (today only
    /// [`LexicalStore::optimize`](crate::lexical::store::LexicalStore::optimize)'s
    /// force-merge), so a writer that caches segment metadata rebuilds it
    /// from storage. The default implementation is a no-op for writers that
    /// hold no such cache.
    ///
    /// # Errors
    ///
    /// Returns an error if rebuilding the cached view from storage fails.
    fn invalidate_segment_cache(&mut self) -> Result<()> {
        Ok(())
    }

    /// Persist any buffered (deferred) deletion state to storage (Issue #875).
    ///
    /// Writers that defer deletion persistence for group-commit (the
    /// [`InvertedIndexWriter`](crate::lexical::index::inverted::writer::InvertedIndexWriter)
    /// buffers deletion-bitmap writes and `has_deletions` metadata flips and
    /// flushes them once per [`commit()`](Self::commit)) must also expose the
    /// flush for callers that consume deletion state from storage without
    /// committing — today that is
    /// [`LexicalStore::optimize`](crate::lexical::store::LexicalStore::optimize),
    /// whose force-merge reads the on-disk deletion bitmaps. The default
    /// implementation is a no-op for writers that persist deletions eagerly.
    ///
    /// # Errors
    ///
    /// Returns an error if persisting the buffered deletion state fails.
    fn flush_deletions(&mut self) -> Result<()> {
        Ok(())
    }
}
