//! High-level lexical search engine that combines indexing and searching.
//!
//! This module provides the core `LexicalStore` implementation.

pub mod config;

use std::sync::Arc;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::error::Result;
use crate::lexical::core::document::Document;
use crate::lexical::index::LexicalIndex;
use crate::lexical::index::factory::LexicalIndexFactory;
use crate::lexical::index::inverted::InvertedIndexStats;
use crate::lexical::query::LexicalSearchResults;
use crate::lexical::query::Query;
#[cfg(test)]
use crate::lexical::reader::LexicalIndexReader;
use crate::lexical::search::searcher::{LexicalSearchRequest, LexicalSearcher};
use crate::lexical::store::config::LexicalIndexConfig;
use crate::lexical::writer::LexicalIndexWriter;
use crate::storage::Storage;
use parking_lot::Mutex;
use parking_lot::RwLock;

/// A high-level lexical search engine that provides both indexing and searching capabilities.
///
/// The `LexicalStore` wraps a `LexicalIndex` trait object and provides a simplified,
/// unified interface for all lexical search operations. It manages the complexity of
/// coordinating between readers and writers while maintaining efficiency through caching.
///
/// # Features
///
/// - **Writer caching**: The writer is created on-demand and cached until commit
/// - **Searcher invalidation**: Searchers are automatically invalidated after commits/optimizations
/// - **Index abstraction**: Works with any `LexicalIndex` implementation (Inverted, etc.)
/// - **Simplified workflow**: Handles the lifecycle of readers and writers automatically
///
/// # Caching Strategy
///
/// - **Writer**: Created on first write operation, cached until `commit()` is called
/// - **Searcher**: Cached on first search after invalidation; invalidated after `commit()` or `optimize()`.
///   Uses double-checked locking with `RwLockWriteGuard::downgrade()` so that only
///   searcher *creation* holds an exclusive lock; the actual search runs under a shared
///   read lock, allowing concurrent queries.
/// - This design ensures that you always read committed data while minimizing object creation
///
/// # Usage Example
///
/// ```rust,no_run
/// use laurus::lexical::core::document::Document;
/// use laurus::lexical::store::LexicalStore;
/// use laurus::lexical::store::config::LexicalIndexConfig;
/// use laurus::lexical::search::searcher::LexicalSearchRequest;
/// use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
/// use std::sync::Arc;
///
/// // Create storage and engine
/// let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
/// let config = LexicalIndexConfig::default();
/// let engine = LexicalStore::new(storage, config).unwrap();
///
/// // Add documents
/// let doc = Document::builder()
///     .add_text("title", "Rust Programming")
///     .build();
/// engine.upsert_document(1, doc).unwrap();
/// engine.commit().unwrap();
///
/// // Search using DSL string
/// let results = engine.search(LexicalSearchRequest::from_dsl("title:rust")).unwrap();
/// ```
pub struct LexicalStore {
    /// The underlying lexical index.
    index: Box<dyn LexicalIndex>,
    /// Cached writer instance, created on-demand for write operations and cleared on commit.
    writer_cache: Mutex<Option<Box<dyn LexicalIndexWriter>>>,
    /// Cached searcher instance, invalidated after `commit()` or `optimize()` to ensure fresh data.
    searcher_cache: RwLock<Option<Box<dyn LexicalSearcher>>>,
}

impl std::fmt::Debug for LexicalStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LexicalStore")
            .field("index", &self.index)
            .finish()
    }
}

impl LexicalStore {
    /// Create a new lexical search engine with the given storage and configuration.
    ///
    /// This constructor creates a `LexicalIndex` internally using the provided storage
    /// and configuration, then wraps it with lazy-initialized caches for the reader,
    /// writer, and searcher.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend for persisting index data
    /// * `config` - Configuration for the lexical index (schema, analyzer, etc.)
    ///
    /// # Returns
    ///
    /// Returns a new `LexicalStore` instance.
    ///
    /// # Example with Memory Storage
    ///
    /// ```rust,no_run
    /// use laurus::lexical::store::LexicalStore;
    /// use laurus::lexical::store::config::LexicalIndexConfig;
    /// use laurus::storage::{Storage, StorageConfig, StorageFactory};
    /// use laurus::storage::memory::MemoryStorageConfig;
    /// use std::sync::Arc;
    ///
    /// let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// let storage = StorageFactory::create(storage_config).unwrap();
    /// let engine = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
    /// ```
    ///
    /// # Example with File Storage
    ///
    /// ```rust,no_run
    /// use laurus::lexical::store::LexicalStore;
    /// use laurus::lexical::store::config::LexicalIndexConfig;
    /// use laurus::storage::{Storage, StorageConfig, StorageFactory};
    /// use laurus::storage::file::FileStorageConfig;
    /// use std::sync::Arc;
    ///
    /// let storage_config = StorageConfig::File(FileStorageConfig::new("/tmp/index"));
    /// let storage = StorageFactory::create(storage_config).unwrap();
    /// let engine = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
    /// ```
    pub fn new(storage: Arc<dyn Storage>, config: LexicalIndexConfig) -> Result<Self> {
        let index = LexicalIndexFactory::open_or_create(storage, config)?;
        Ok(Self {
            index,
            writer_cache: Mutex::new(None),
            searcher_cache: RwLock::new(None),
        })
    }

    /// Upsert a document with a specific internal ID.
    ///
    /// The caller is responsible for doc_id generation (via [`DocumentLog`](crate::store::log::DocumentLog)).
    /// Changes are not persisted until you call `commit()`.
    pub fn upsert_document(&self, internal_id: u64, doc: Document) -> Result<()> {
        let mut guard = self.writer_cache.lock();
        if guard.is_none() {
            *guard = Some(self.index.writer()?);
        }
        guard.as_mut().unwrap().upsert_document(internal_id, doc)
    }

    /// Delete a document by internal ID.
    ///
    /// Note: You must call `commit()` to persist the changes.
    pub(crate) fn delete_document_by_internal_id(&self, internal_id: u64) -> Result<()> {
        // Delete from doc store
        // UnifiedDocumentStore doesn't have delete? It might have a bitmap or tombstone.
        // Actually deletion is usually handled by a separate DeletionPolicy / Bitmap.
        // But if DocumentStore is source of truth, maybe it has delete?
        // Let's leave DocumentStore alone for deletion for now (soft delete/vacuum handled elsewhere).
        // Or check if it has delete.
        // Assuming no delete in doc store for now (compaction handles it).

        let mut guard = self.writer_cache.lock();
        if guard.is_none() {
            *guard = Some(self.index.writer()?);
        }
        guard.as_mut().unwrap().delete_document(internal_id)
    }

    /// Find all internal document IDs for a given term (field:value).
    ///
    /// This searches both the uncommitted in-memory buffer (via Writer) and
    /// the committed index (via Searcher).
    pub(crate) fn find_doc_ids_by_term(&self, field: &str, term: &str) -> Result<Vec<u64>> {
        let mut ids = Vec::new();
        let guard = self.writer_cache.lock();

        // 1. Check writer (NRT - Uncommitted)
        if let Some(writer) = guard.as_ref()
            && let Some(writer_ids) = writer.find_doc_ids_by_term(field, term)?
        {
            ids.extend(writer_ids);
        }

        // 2. Check reader (Committed)
        use crate::lexical::query::Query;
        use crate::lexical::query::term::TermQuery;

        let query = Box::new(TermQuery::new(field, term)) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query)
            .limit(usize::MAX) // Retrieve all matches
            .load_documents(false);

        // Safe to call search while holding the writer lock: the lock order
        // is writer_cache -> searcher_cache everywhere, and search() takes
        // only searcher_cache. Every path that retires the writer (commit,
        // optimize, the field operations, close) also drops the cached
        // searcher before releasing the writer lock (Issue #1207), so this
        // search never pairs a pre-commit snapshot with a missing writer.
        let results = self.search(request)?;
        for hit in results.hits {
            if !ids.contains(&hit.doc_id) {
                // Check if marked as deleted in pending set
                let is_deleted = if let Some(writer) = guard.as_ref() {
                    writer.is_updated_deleted(hit.doc_id)
                } else {
                    false
                };

                if !is_deleted {
                    ids.push(hit.doc_id);
                }
            }
        }

        Ok(ids)
    }

    /// Commit any pending changes to the index.
    ///
    /// This method flushes all pending write operations to storage and makes them
    /// visible to subsequent searches. The cached writer is consumed and the reader
    /// cache is invalidated to ensure fresh data on the next search.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the commit fails.
    ///
    /// # Important
    ///
    /// - All write operations (add, update, delete) are not persisted until commit
    /// - After commit, the reader cache is invalidated automatically
    /// - The writer cache is cleared and will be recreated on the next write operation
    /// - An auto-merge runs after the commit (Issue #755): once the segment
    ///   count exceeds `max_segments`, the smallest `merge_factor` segments are
    ///   merged so the count stays bounded without a manual `optimize()`. It is
    ///   a no-op below the threshold; raise `max_segments` to disable it.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use laurus::lexical::core::document::Document;
    /// # use laurus::lexical::store::LexicalStore;
    /// # use laurus::lexical::store::config::LexicalIndexConfig;
    /// # use laurus::storage::{StorageConfig, StorageFactory};
    /// use laurus::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let engine = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
    ///
    /// // Add multiple documents
    /// for i in 0..10 {
    ///     let doc = Document::builder()
    ///         .add_text("id", &i.to_string())
    ///         .add_text("title", &format!("Document {}", i))
    ///         .build();
    ///     engine.upsert_document(i + 1, doc).unwrap();
    /// }
    ///
    /// // Commit all changes at once
    /// engine.commit().unwrap();
    /// ```
    pub fn commit(&self) -> Result<()> {
        // Hold the writer-cache lock across the whole ladder (Issue #864): a
        // writer constructed concurrently while `maybe_merge` replaces
        // segments would seed its segment-range cache from a racing,
        // mid-merge `.meta` scan and never be invalidated. With the lock
        // held, new writers can only be constructed strictly before or
        // strictly after the merge, where the scan sees a consistent set.
        let mut writer_guard = self.writer_cache.lock();
        // Commit through a borrow and drop the writer only on SUCCESS (Issue
        // #875): taking it out first would destroy its buffered state — the
        // deferred deletion bitmaps and `has_deletions` meta flips — on a
        // failed commit (the writer's silent Drop-close retry usually fails
        // the same way), after which a later successful commit would truncate
        // the WAL past the delete records and make the loss permanent without
        // any crash. Keeping the writer cached preserves everything for the
        // retry the WAL contract assumes.
        if let Some(writer) = writer_guard.as_mut() {
            writer.commit()?;
        }
        *writer_guard = None;
        let finished = self.finish_commit();
        // Drop the cached searcher BEFORE releasing the writer lock, on the
        // error path too (Issue #1207). The writer is gone, so an `_id`
        // lookup that took the lock in between would search this pre-commit
        // snapshot with no pending-deletion filter: it would return the
        // superseded copy and miss the version just committed, and a
        // concurrent put of that `_id` would then leave a duplicate.
        *self.searcher_cache.write() = None;
        drop(writer_guard);
        finished
    }

    /// The steps of [`Self::commit`] that follow the writer's commit.
    fn finish_commit(&self) -> Result<()> {
        // Sync storage to ensure all file metadata (creation, rename, size) is
        // flushed to disk. This is critical on Windows where directory listings
        // and file visibility may be cached until the directory is synced.
        self.index.storage().sync()?;
        // Auto-merge after the new segment is visible (Issue #755): keeps the
        // segment count bounded without a manual `optimize()`. A no-op below the
        // configured threshold, so most commits pay only a segment-count check.
        // Sync again afterwards so any merge output is durable/visible.
        self.index.maybe_merge()?;
        self.index.storage().sync()?;
        // For `InvertedIndex` this is a no-op (#1023): the writer applied
        // its commit deltas straight to the index's shared metadata, so
        // there is nothing fresher on disk. Kept for other `LexicalIndex`
        // implementations whose refresh does real work.
        self.index.refresh()
    }

    /// Optimize the index by force-merging all segments into one (Issue #754).
    ///
    /// This method delegates to [`LexicalIndex::optimize()`], which for the default
    /// [`InvertedIndex`](crate::lexical::index::inverted::InvertedIndex) implementation
    /// force-merges every current segment into a single new segment (the classic
    /// `optimize` / force-merge semantics): it rewrites the live documents into one
    /// segment, reclaiming logically deleted documents and removing the source
    /// segment files. This bounds the per-query cost that otherwise grows with the
    /// number of commits. It is a no-op when fewer than two segments exist.
    /// After optimization, the searcher cache is invalidated so subsequent searches
    /// reflect the merged state.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if optimization fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use laurus::lexical::core::document::Document;
    /// # use laurus::lexical::store::LexicalStore;
    /// # use laurus::lexical::store::config::LexicalIndexConfig;
    /// # use laurus::storage::{StorageConfig, StorageFactory};
    /// use laurus::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let mut engine = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
    ///
    /// // Add and commit many documents
    /// for i in 0..1000 {
    ///     let doc = Document::builder()
    ///         .add_text("id", &i.to_string())
    ///         .build();
    ///     engine.upsert_document(i + 1, doc).unwrap();
    /// }
    /// engine.commit().unwrap();
    ///
    /// // Optimize the index for better performance
    /// engine.optimize().unwrap();
    /// ```
    pub fn optimize(&self) -> Result<()> {
        // Hold the writer-cache lock across the force-merge (Issue #864):
        // without it a concurrent upsert can run against the live writer's
        // stale segment cache while the merge deletes those segments, marking
        // deletions in ghost segments (lost dedup / duplicate versions).
        let mut writer_guard = self.writer_cache.lock();
        // Commit the live writer BEFORE the merge (#1017).
        //
        // The merge consumes the published segments, so a segment flushed but
        // not yet committed would sit outside it — while
        // `invalidate_segment_cache` below still drops the writer's NRT
        // readers for it. That is exactly the #1016 failure mode: `get` and
        // `delete` silently stop finding those documents and an upsert leaves
        // a duplicate. Committing first means no unpublished segment exists
        // and the question does not arise.
        //
        // It also subsumes the older `flush_deletions` call this replaces
        // (Issue #875): the merge engine reads deletions from the on-disk
        // `.delmap` files, so unflushed ones would be resurrected into the
        // merged segment and then discarded with the writer's deletion
        // manager. `commit` flushes them as part of its ladder.
        //
        // Failing here aborts the optimize with the writer state intact,
        // which is the same contract as before. Lucene's `forceMerge`
        // likewise operates on a flushed writer.
        if let Some(writer) = writer_guard.as_mut() {
            writer.commit()?;
        }
        let merge_result = self.index.optimize();
        // The force-merge replaces committed segments (and their deletion
        // bitmaps) behind any live writer — rebuild its cached segment view
        // even when the merge errored partway, since segments may already
        // have been replaced by then. If the rebuild itself fails, drop the
        // writer entirely: an unrebuildable cache must not survive to serve
        // stale (or, worse, empty) ranges.
        if let Some(writer) = writer_guard.as_mut()
            && let Err(e) = writer.invalidate_segment_cache()
        {
            *writer_guard = None;
            *self.searcher_cache.write() = None;
            drop(writer_guard);
            merge_result?;
            return Err(e);
        }
        // Before the writer lock is released, as in `commit` (Issue #1207).
        *self.searcher_cache.write() = None;
        drop(writer_guard);
        merge_result
    }

    /// Refresh the reader to see latest changes.
    ///
    /// Invalidates the cached searcher so that the next search operation will
    /// create a new searcher reflecting the most recent committed data.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    pub fn refresh(&self) -> Result<()> {
        *self.searcher_cache.write() = None;
        Ok(())
    }

    /// Borrow the underlying [`LexicalIndexReader`] (#476 Phase 1).
    ///
    /// Exposed for crate-internal tests that need to drive the
    /// inverted searcher directly with a custom collector — e.g. to
    /// bypass the per-segment fanout and exercise the legacy
    /// matcher-driven path against the same multi-segment store.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn reader_for_tests(&self) -> Result<Arc<dyn LexicalIndexReader>> {
        self.index.reader()
    }

    /// Get index statistics.
    ///
    /// Returns aggregated statistics including document count and deleted document
    /// count from the index metadata. The `doc_count` field also includes any
    /// documents pending in the writer cache that have not yet been committed.
    ///
    /// # Current Limitations
    ///
    /// In the current [`InvertedIndex`](crate::lexical::index::inverted::InvertedIndex)
    /// implementation, the following fields are always returned as `0`:
    /// - `term_count`
    /// - `segment_count`
    /// - `total_size`
    ///
    /// # Returns
    ///
    /// An [`InvertedIndexStats`] snapshot on success, or an error if the
    /// underlying index cannot provide statistics.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`](crate::error::LaurusError) if the index stats
    /// cannot be retrieved (e.g., the index is closed).
    pub fn stats(&self) -> Result<InvertedIndexStats> {
        let mut stats = self.index.stats()?;

        // Add pending docs from writer cache
        let guard = self.writer_cache.lock();
        if let Some(writer) = guard.as_ref() {
            stats.doc_count += writer.pending_docs();
        }

        Ok(stats)
    }

    /// Get the storage backend.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        self.index.storage()
    }

    /// Search with the given request.
    ///
    /// This method executes a search query against the index using a cached searcher
    /// for improved performance.
    ///
    /// # Arguments
    ///
    /// * `request` - The search request containing the query and search parameters
    ///
    /// # Returns
    ///
    /// Returns `SearchResults` containing matching documents, scores, and metadata.
    ///
    /// # Example with TermQuery
    ///
    /// ```rust,no_run
    /// use laurus::lexical::core::document::Document;
    /// use laurus::lexical::search::searcher::LexicalSearchRequest;
    /// use laurus::lexical::query::term::TermQuery;
    /// # use laurus::lexical::store::LexicalStore;
    /// # use laurus::lexical::store::config::LexicalIndexConfig;
    /// # use laurus::storage::{StorageConfig, StorageFactory};
    /// use laurus::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let engine = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
    /// # let doc = Document::builder().add_text("title", "hello world").build();
    /// # engine.upsert_document(1, doc).unwrap();
    /// # engine.commit().unwrap();
    ///
    /// // Using DSL string
    /// let request = LexicalSearchRequest::from_dsl("title:hello")
    ///     .limit(10)
    ///     .min_score(0.5);
    /// let results = engine.search(request).unwrap();
    ///
    /// println!("Found {} documents", results.total_hits);
    /// for hit in results.hits {
    ///     println!("Doc ID: {}, Score: {}", hit.doc_id, hit.score);
    /// }
    /// ```
    ///
    /// # Example with QueryParser
    ///
    /// ```rust,no_run
    /// use laurus::lexical::query::parser::LexicalQueryParser;
    /// use laurus::lexical::search::searcher::LexicalSearchRequest;
    /// # use laurus::lexical::core::document::Document;
    /// # use laurus::lexical::store::LexicalStore;
    /// # use laurus::lexical::store::config::LexicalIndexConfig;
    /// # use laurus::storage::{StorageConfig, StorageFactory};
    /// use laurus::storage::memory::MemoryStorageConfig;
    /// use laurus::analysis::analyzer::standard::StandardAnalyzer;
    /// # use std::sync::Arc;
    /// # let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    /// # let storage = StorageFactory::create(storage_config).unwrap();
    /// # let engine = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
    ///
    /// let analyzer = Arc::new(StandardAnalyzer::default());
    /// let parser = LexicalQueryParser::new(analyzer).with_default_field("title");
    /// let query = parser.parse("rust AND programming").unwrap();
    /// let results = engine.search(LexicalSearchRequest::new(query)).unwrap();
    /// ```
    pub fn search(&self, request: LexicalSearchRequest) -> Result<LexicalSearchResults> {
        // Fast path: read lock, cache hit — concurrent searches proceed in parallel.
        {
            let guard = self.searcher_cache.read();
            if let Some(ref searcher) = *guard {
                return searcher.search(request);
            }
        }

        // Slow path: write lock to populate, then downgrade to read lock so that
        // the actual search executes under a shared read lock rather than an
        // exclusive write lock. This allows other readers to proceed as soon as
        // the searcher is created.
        let mut guard = self.searcher_cache.write();
        if guard.is_none() {
            *guard = Some(self.index.searcher()?);
        }
        let guard = parking_lot::RwLockWriteGuard::downgrade(guard);
        guard.as_ref().unwrap().search(request)
    }

    /// Count documents matching the request.
    ///
    /// Uses a cached searcher for improved performance.
    /// If `min_score` is specified in the request parameters, only documents
    /// with a score equal to or greater than the threshold are counted.
    ///
    /// # Arguments
    ///
    /// * `request` - Search request containing the query and search parameters.
    ///   Use `LexicalSearchRequest::new(query)` to create a request.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use laurus::lexical::store::LexicalStore;
    /// # use laurus::lexical::store::config::LexicalIndexConfig;
    /// # use laurus::lexical::search::searcher::LexicalSearchRequest;
    /// # use laurus::storage::memory::MemoryStorage;
    /// # use laurus::storage::memory::MemoryStorageConfig;
    /// # use std::sync::Arc;
    /// # let config = LexicalIndexConfig::default();
    /// # let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    /// # let engine = LexicalStore::new(storage, config).unwrap();
    /// // Count all matching documents
    /// let count = engine.count(LexicalSearchRequest::from_dsl("title:hello")).unwrap();
    /// println!("Found {} documents", count);
    ///
    /// // Count with min_score threshold
    /// let count = engine.count(
    ///     LexicalSearchRequest::from_dsl("title:hello").min_score(0.5)
    /// ).unwrap();
    /// println!("Found {} documents with score >= 0.5", count);
    /// ```
    pub fn count(&self, request: LexicalSearchRequest) -> Result<u64> {
        // Fast path: read lock, cache hit.
        {
            let guard = self.searcher_cache.read();
            if let Some(ref searcher) = *guard {
                return searcher.count(request);
            }
        }

        // Slow path: populate under write lock, then downgrade to read lock.
        let mut guard = self.searcher_cache.write();
        if guard.is_none() {
            *guard = Some(self.index.searcher()?);
        }
        let guard = parking_lot::RwLockWriteGuard::downgrade(guard);
        guard.as_ref().unwrap().count(request)
    }

    /// Return the set of document ids matching `query`, ignoring score
    /// thresholds, via the snapshot-scoped query / filter cache (Issue
    /// [#578](https://github.com/mosuka/laurus/issues/578)).
    ///
    /// Filters select documents without affecting relevance, so the result is a
    /// score-independent doc-id set. This goes through the cached searcher (and
    /// thus the reader's filter cache), so a repeated filter is served without
    /// re-walking posting lists. The cache is invalidated automatically by
    /// `commit()` / `optimize()` / `refresh()`, which drop the cached searcher.
    ///
    /// # Arguments
    ///
    /// * `query` - The query whose matching document set is requested.
    ///
    /// # Returns
    ///
    /// An `Arc<roaring::RoaringTreemap>` of matching document ids.
    pub fn matching_doc_ids(&self, query: Box<dyn Query>) -> Result<Arc<roaring::RoaringTreemap>> {
        // Fast path: read lock, cache hit.
        {
            let guard = self.searcher_cache.read();
            if let Some(ref searcher) = *guard {
                return searcher.matching_doc_ids(query);
            }
        }

        // Slow path: populate under write lock, then downgrade to read lock.
        let mut guard = self.searcher_cache.write();
        if guard.is_none() {
            *guard = Some(self.index.searcher()?);
        }
        let guard = parking_lot::RwLockWriteGuard::downgrade(guard);
        guard.as_ref().unwrap().matching_doc_ids(query)
    }

    /// Close the search engine and release resources.
    ///
    /// Drops the cached writer and searcher, then marks the underlying index
    /// as closed. After this call, subsequent operations on the store will
    /// fail with a "closed" error.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the underlying index fails to close.
    pub fn close(&self) -> Result<()> {
        let mut writer_guard = self.writer_cache.lock();
        *writer_guard = None;
        *self.searcher_cache.write() = None;
        drop(writer_guard);
        self.index.close()
    }

    /// Check if the engine is closed.
    ///
    /// Delegates to the underlying [`LexicalIndex::is_closed()`] method.
    /// Returns `true` if [`close()`](Self::close) has been called, `false` otherwise.
    pub fn is_closed(&self) -> bool {
        self.index.is_closed()
    }

    /// Get the analyzer used by this engine.
    ///
    /// Returns the analyzer from the underlying index reader.
    /// This is useful for query parsing and term normalization.
    ///
    /// # Returns
    ///
    /// Returns `Result<Arc<dyn Analyzer>>` containing the analyzer.
    ///
    /// # Errors
    ///
    /// Returns an error if the reader cannot be created or the index type
    /// doesn't support analyzers.
    pub fn analyzer(&self) -> Result<Arc<dyn Analyzer>> {
        use crate::lexical::index::inverted::reader::InvertedIndexReader;

        let reader = self.index.reader()?;

        // Downcast to InvertedIndexReader to access analyzer
        if let Some(inverted_reader) = reader.as_any().downcast_ref::<InvertedIndexReader>() {
            Ok(Arc::clone(inverted_reader.analyzer()))
        } else {
            // For other index types, return StandardAnalyzer as default
            use crate::analysis::analyzer::standard::StandardAnalyzer;
            Ok(Arc::new(StandardAnalyzer::new()?))
        }
    }

    /// Create a query parser configured for this index.
    ///
    /// The parser uses the index's analyzer and default fields configuration.
    ///
    /// # Returns
    ///
    /// Returns `Result<QueryParser>` containing the configured parser.
    pub fn query_parser(&self) -> Result<crate::lexical::query::parser::LexicalQueryParser> {
        let analyzer = self.analyzer()?;
        let mut parser = crate::lexical::query::parser::LexicalQueryParser::new(analyzer);

        if let Ok(fields) = self.index.default_fields()
            && !fields.is_empty()
        {
            parser = parser.with_default_fields(fields);
        }

        Ok(parser)
    }

    /// Get the last processed WAL sequence number.
    pub fn last_wal_seq(&self) -> u64 {
        self.index.last_wal_seq()
    }

    /// Set the last processed WAL sequence number.
    ///
    /// If a writer is cached, it sets the sequence on the writer.
    /// Otherwise, it sets it on the underlying index.
    pub fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
        if let Some(writer) = self.writer_cache.lock().as_mut() {
            writer.set_last_wal_seq(seq)?;
        } else {
            self.index.set_last_wal_seq(seq)?;
        }
        Ok(())
    }

    /// Dynamically add a new lexical field to the index at runtime.
    ///
    /// This registers the field in the underlying index so that subsequent
    /// writers will include the new field in their configuration. It also
    /// registers the field-specific analyzer if the index's analyzer is a
    /// [`PerFieldAnalyzer`](crate::analysis::analyzer::per_field::PerFieldAnalyzer).
    ///
    /// After adding a field, the writer and searcher caches are invalidated
    /// so the next operation uses updated configurations.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name
    /// * `option` - The field configuration
    /// * `analyzer` - Optional field-specific analyzer to register
    ///
    /// # Errors
    ///
    /// Returns an error if the field already exists or the underlying index
    /// does not support dynamic field addition.
    pub fn add_field(
        &self,
        name: &str,
        option: crate::lexical::core::field::FieldOption,
        analyzer: Option<Arc<dyn Analyzer>>,
    ) -> Result<()> {
        // Register the field in the underlying index.
        self.index.add_field(name, option)?;

        // If a field-specific analyzer is provided, register it in the
        // PerFieldAnalyzer (if the index's analyzer supports it).
        if let Some(field_analyzer) = analyzer
            && let Ok(index_analyzer) = self.analyzer()
            && let Some(pfa) = index_analyzer
                .as_any()
                .downcast_ref::<crate::analysis::analyzer::per_field::PerFieldAnalyzer>(
            )
        {
            pfa.add_analyzer(name, field_analyzer);
        }

        // Invalidate caches so the next writer/searcher uses updated config.
        // Commit the cached writer through a borrow and drop it only on
        // SUCCESS (Issue #875): a bare `= None` relies on the writer's silent
        // Drop-close commit, whose failure would destroy the buffered state —
        // including deferred deletion bitmaps — while this method still
        // returns `Ok`; a later successful commit would then truncate the WAL
        // past the acknowledged delete records, resurrecting old document
        // versions permanently. Reachable during normal ingest: the Dynamic
        // field policy calls `add_field` automatically for unseen fields.
        {
            let mut writer_guard = self.writer_cache.lock();
            if let Some(writer) = writer_guard.as_mut() {
                writer.commit()?;
            }
            *writer_guard = None;
            // Before the writer lock is released, as in `commit` (Issue #1207).
            *self.searcher_cache.write() = None;
        }

        Ok(())
    }

    /// Rebuild an existing lexical field's on-disk data under a new
    /// option/analyzer (Issue #1081: `Engine::update_field`'s `Reindex`
    /// classification for lexical fields — e.g. a text field's
    /// `analyzer`, or `indexed: false -> true`).
    ///
    /// # Arguments
    ///
    /// * `name` - The field name. Must already exist.
    /// * `option` - The field's new configuration.
    /// * `analyzer` - The field-specific analyzer to register for `name`.
    ///   `None` removes any existing per-field registration (e.g. the
    ///   field is being switched to `indexed: false`, or has no
    ///   analyzer).
    ///
    /// # Ordering (this is the fix for Issue #1081's race)
    ///
    /// The segment rebuild ([`LexicalIndex::rebuild_field`]) runs FIRST;
    /// the per-field analyzer swap
    /// ([`PerFieldAnalyzer::add_analyzer`](crate::analysis::analyzer::per_field::PerFieldAnalyzer::add_analyzer)/
    /// `remove_analyzer`) only happens after it succeeds. [`Self::add_field`]
    /// does the opposite (analyzer first) because a brand-new field has no
    /// existing postings to race against; a rebuild does, so swapping the
    /// analyzer first would let a query issued mid-rebuild parse against
    /// the NEW analyzer while on-disk postings are still built with the
    /// OLD one.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing a cached writer's pending changes
    /// fails, or if the underlying rebuild fails. Unlike
    /// [`Self::add_field`]/[`Self::delete_field`], a flush failure is
    /// **propagated**, not cached for a later retry — the writer's
    /// buffered state is not a safe starting point for a rebuild that is
    /// about to replace every segment (same reasoning [`Self::optimize`]
    /// already applies).
    pub fn rebuild_field(
        &self,
        name: &str,
        option: crate::lexical::core::field::FieldOption,
        analyzer: Option<Arc<dyn Analyzer>>,
    ) -> Result<()> {
        // Flush pending writes BEFORE the rebuild (mirrors `optimize()`):
        // the rebuild reads segments straight from storage, so anything
        // still buffered in the cached writer must be durable first, or
        // the rebuild would silently miss it. A failure here is
        // propagated, not swallowed — see the doc comment above.
        let mut writer_guard = self.writer_cache.lock();
        if let Some(writer) = writer_guard.as_mut() {
            writer.commit()?;
        }

        self.index.rebuild_field(name, option, analyzer.clone())?;

        // Only NOW does the analyzer swap become visible to the query
        // parser — after the rebuild has fully replaced the field's
        // postings under the new analyzer (see the ordering note above).
        if let Ok(index_analyzer) = self.analyzer()
            && let Some(pfa) = index_analyzer
                .as_any()
                .downcast_ref::<crate::analysis::analyzer::per_field::PerFieldAnalyzer>(
            )
        {
            match analyzer {
                Some(field_analyzer) => pfa.add_analyzer(name, field_analyzer),
                None => pfa.remove_analyzer(name),
            }
        }

        // The rebuild replaced every segment, so the writer's cached
        // segment view (if it survived the flush above) is now stale —
        // mirrors `optimize()`'s `invalidate_segment_cache` handling.
        if let Some(writer) = writer_guard.as_mut()
            && let Err(e) = writer.invalidate_segment_cache()
        {
            *writer_guard = None;
            *self.searcher_cache.write() = None;
            drop(writer_guard);
            return Err(e);
        }
        *writer_guard = None;
        // Before the writer lock is released, as in `commit` (Issue #1207).
        *self.searcher_cache.write() = None;
        drop(writer_guard);

        Ok(())
    }

    /// Remove a field from the lexical store.
    ///
    /// Removes the field from the underlying index (if it was dynamically added)
    /// and unregisters any field-specific analyzer from the `PerFieldAnalyzer`.
    /// After this call, the field will no longer be available for indexing or
    /// searching, but existing data in the index is not deleted.
    ///
    /// # Arguments
    ///
    /// * `name` - The field name to remove
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying index does not support dynamic field
    /// deletion.
    pub fn delete_field(&self, name: &str) -> Result<()> {
        // Remove the field from the underlying index.
        self.index.delete_field(name)?;

        // Remove the field-specific analyzer from the PerFieldAnalyzer if present.
        if let Ok(index_analyzer) = self.analyzer()
            && let Some(pfa) = index_analyzer
                .as_any()
                .downcast_ref::<crate::analysis::analyzer::per_field::PerFieldAnalyzer>(
            )
        {
            pfa.remove_analyzer(name);
        }

        // Invalidate caches so the next writer/searcher uses updated config.
        // Same commit-then-drop-on-success guard as `add_field` (Issue #875):
        // never rely on the silent Drop-close commit to persist buffered
        // deletion state.
        {
            let mut writer_guard = self.writer_cache.lock();
            if let Some(writer) = writer_guard.as_mut() {
                writer.commit()?;
            }
            *writer_guard = None;
            // Before the writer lock is released, as in `commit` (Issue #1207).
            *self.searcher_cache.write() = None;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::query::Query;
    use crate::lexical::query::term::TermQuery;
    use crate::lexical::store::config::LexicalIndexConfig;
    use crate::storage::file::{FileStorage, FileStorageConfig};
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use std::sync::Arc;
    use tempfile::TempDir;

    fn create_test_document(title: &str, body: &str) -> Document {
        Document::builder()
            .add_text("title", title)
            .add_text("body", body)
            .build()
    }

    /// A store whose writer flushes after every second document.
    fn store_flushing_every_two_docs() -> LexicalStore {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = LexicalIndexConfig::builder().max_buffered_docs(2).build();
        LexicalStore::new(storage, config).unwrap()
    }

    fn id_doc(id: &str) -> Document {
        Document::builder().add_text("_id", id).build()
    }

    /// Issue #1207 — re-upserting a doc id whose flushed copy was superseded
    /// must still resolve the new version once that version is flushed too.
    ///
    /// The flushed-segment lookup skipped every id in the writer's flat
    /// `pending_deletions` set, so it hid the new copy along with the old
    /// one. It now asks each flushed segment's own deletion bitmap.
    #[test]
    fn reupserted_version_resolves_after_its_own_flush() {
        let store = store_flushing_every_two_docs();
        store.upsert_document(1, id_doc("ida")).unwrap();
        store.upsert_document(2, id_doc("idb")).unwrap(); // flushes 1, 2
        store.upsert_document(1, id_doc("ida2")).unwrap(); // supersedes 1
        store.upsert_document(3, id_doc("idc")).unwrap(); // flushes 1 (new), 3

        assert_eq!(store.find_doc_ids_by_term("_id", "ida2").unwrap(), vec![1]);
        assert!(
            store.find_doc_ids_by_term("_id", "ida").unwrap().is_empty(),
            "the superseded copy must stay hidden"
        );
    }

    /// Issue #1207 — a re-upserted version published by `optimize()` must
    /// resolve right away. `optimize` commits but keeps its writer, whose
    /// `pending_deletions` still named the id and so filtered the new,
    /// committed copy out of the lookup until the next commit.
    #[test]
    fn reupserted_version_resolves_after_optimize() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        store.upsert_document(1, id_doc("ida")).unwrap();
        store.commit().unwrap();

        store.upsert_document(1, id_doc("ida2")).unwrap();
        store.optimize().unwrap();

        assert_eq!(store.find_doc_ids_by_term("_id", "ida2").unwrap(), vec![1]);
        assert!(store.find_doc_ids_by_term("_id", "ida").unwrap().is_empty());
    }

    /// #1016 — a document written before the writer's automatic
    /// `flush_segment` must still resolve by `_id` before commit.
    ///
    /// The flush empties the in-memory buffer, which used to be the only
    /// thing the NRT lookup searched, so the document silently stopped
    /// being findable — which is what made `get` return nothing and
    /// `delete` do nothing for it.
    #[test]
    fn doc_written_before_auto_flush_still_resolves_by_id() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        // Past `max_buffered_docs` (10_000) so the writer auto-flushes.
        //
        // Resolve before every write, exactly as the ingest path does —
        // `Engine::index_internal` runs `delete_documents_internal`, and so
        // `find_doc_ids_by_term`, on every put (engine.rs:787). That warms
        // the searcher cache well before the flush, which is the state the
        // bug actually occurs in; a test that only resolves at the end gets
        // a cold cache and a freshly built searcher, and would pass without
        // the fix.
        let total = 10_050u64;
        for i in 0..total {
            let id = format!("id{i:06}");
            let _ = store.find_doc_ids_by_term("_id", &id).unwrap();
            let doc = Document::builder()
                .add_text("_id", &id)
                .add_text("body", "alpha")
                .build();
            store.upsert_document(i + 1, doc).unwrap();
        }

        // No commit at any point.
        let buffered = store.find_doc_ids_by_term("_id", "id010040").unwrap();
        assert_eq!(buffered.len(), 1, "a buffered doc must resolve");

        let flushed = store.find_doc_ids_by_term("_id", "id000000").unwrap();
        assert_eq!(
            flushed,
            vec![1u64],
            "a doc written before the automatic flush must still resolve"
        );
    }

    /// #1016 — after an automatic flush, `_id` resolution must still
    /// surface the flushed copy so a later upsert can supersede it.
    ///
    /// This is the mechanism behind the duplicate-document symptom.
    /// `Engine::index_internal` implements upsert as "resolve the `_id` to
    /// its existing internal ids, delete those, then write the new
    /// version" (engine.rs:1065). When resolution silently returned
    /// nothing, step one found no old version, nothing was deleted, and
    /// the batch ended with two live copies of the same `_id`.
    ///
    /// `docs/src/laurus/deletions.md` states that "upsert deduplication
    /// within an uncommitted batch is handled separately and is always
    /// correct"; this pins that sentence.
    #[test]
    fn upsert_across_auto_flush_supersedes_the_flushed_version() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();

        // Resolve before every write, warming the searcher cache the way
        // the real ingest path does (see the sibling test).
        for i in 0..10_050u64 {
            let id = format!("id{i:06}");
            let _ = store.find_doc_ids_by_term("_id", &id).unwrap();
            let doc = Document::builder()
                .add_text("_id", &id)
                .add_text("body", "alpha")
                .build();
            store.upsert_document(i + 1, doc).unwrap();
        }

        // Step 1 of the engine's upsert: resolve the existing copies. This
        // is what used to come back empty.
        let existing = store.find_doc_ids_by_term("_id", "id000000").unwrap();
        assert_eq!(
            existing,
            vec![1u64],
            "the flushed copy must be visible to the upsert that supersedes it"
        );

        // Steps 2 and 3: delete what was found, then write the new version
        // under a fresh internal id.
        for doc_id in &existing {
            store.delete_document_by_internal_id(*doc_id).unwrap();
        }
        let replacement = Document::builder()
            .add_text("_id", "id000000")
            .add_text("body", "beta")
            .build();
        store.upsert_document(90_001, replacement).unwrap();

        let after = store.find_doc_ids_by_term("_id", "id000000").unwrap();
        assert_eq!(
            after,
            vec![90_001u64],
            "only the new version may remain — no duplicate"
        );
    }

    #[test]
    fn test_search_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        assert!(!engine.is_closed());
    }

    #[test]
    fn test_search_engine_in_memory() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let engine = LexicalStore::new(storage, config).unwrap();

        // Add some documents
        let docs = vec![
            create_test_document("Test Document 1", "Content of test document 1"),
            create_test_document("Test Document 2", "Content of test document 2"),
        ];
        for (i, doc) in docs.into_iter().enumerate() {
            engine.upsert_document((i + 1) as u64, doc).unwrap();
        }
        engine.commit().unwrap();

        // Search for documents
        let query = Box::new(TermQuery::new("title", "Test")) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query);
        let _results = engine.search(request).unwrap();

        assert!(!engine.is_closed());
    }

    #[test]
    fn test_search_engine_open() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        // Create engine
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config.clone()).unwrap();
        engine.close().unwrap();

        // Open engine
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        assert!(!engine.is_closed());
    }

    #[test]
    fn test_upsert_document() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        let doc = create_test_document("Hello World", "This is a test document");
        engine.upsert_document(1, doc).unwrap();
        engine.commit().unwrap();

        let _stats = engine.stats().unwrap();
    }

    #[test]
    fn test_upsert_multiple_documents() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        let docs = vec![
            create_test_document("First Document", "Content of first document"),
            create_test_document("Second Document", "Content of second document"),
            create_test_document("Third Document", "Content of third document"),
        ];

        for (i, doc) in docs.into_iter().enumerate() {
            engine.upsert_document((i + 1) as u64, doc).unwrap();
        }
        engine.commit().unwrap();

        let _stats = engine.stats().unwrap();
    }

    #[test]
    fn test_search_empty_index() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query);
        let results = engine.search(request).unwrap();

        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
        assert_eq!(results.max_score, 0.0);
    }

    #[test]
    fn test_search_with_documents() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        // Add some documents
        let docs = vec![
            create_test_document("Hello World", "This is a test document"),
            create_test_document("Goodbye World", "This is another test document"),
        ];
        for (i, doc) in docs.into_iter().enumerate() {
            engine.upsert_document((i + 1) as u64, doc).unwrap();
        }
        engine.commit().unwrap();

        // Search for documents
        let query = Box::new(TermQuery::new("title", "Hello")) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query);
        let _results = engine.search(request).unwrap();
    }

    #[test]
    fn test_count_query() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;
        let count = engine.count(LexicalSearchRequest::new(query)).unwrap();

        // Should return 0 for empty index
        assert_eq!(count, 0);
    }

    /// Helper: a `LexicalStore` over file storage with `titles` indexed (one
    /// document per title, internal id = index + 1), committed.
    fn store_with_titles(temp_dir: &TempDir, titles: &[&str]) -> LexicalStore {
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for (i, title) in titles.iter().enumerate() {
            store
                .upsert_document(
                    (i + 1) as u64,
                    create_test_document(title, "shared body text"),
                )
                .unwrap();
        }
        store.commit().unwrap();
        store
    }

    fn term_count(store: &LexicalStore, field: &str, term: &str) -> u64 {
        let q = Box::new(TermQuery::new(field, term)) as Box<dyn Query>;
        store.count(LexicalSearchRequest::new(q)).unwrap()
    }

    /// Issue #610: `TermQuery::count` over an index with no deletions returns
    /// the term's document frequency (the O(1) fast path), matching the true
    /// number of matching documents.
    #[test]
    fn test_count_term_query_o1_no_deletions() {
        let temp_dir = TempDir::new().unwrap();
        // "world" appears in 3 titles, "hello" in 1.
        let store = store_with_titles(&temp_dir, &["Hello World", "Goodbye World", "Big World"]);
        assert_eq!(term_count(&store, "title", "world"), 3);
        assert_eq!(term_count(&store, "title", "hello"), 1);
    }

    /// Issue #610: with deletions present the fast path must NOT fire — the
    /// raw `doc_freq` still counts the deleted posting, so the count must come
    /// from the deletion-aware slow path. Deleting one of three "world"
    /// documents yields a count of 2, not 3.
    #[test]
    fn test_count_term_query_excludes_deleted() {
        let temp_dir = TempDir::new().unwrap();
        let store = store_with_titles(&temp_dir, &["Hello World", "Goodbye World", "Big World"]);
        assert_eq!(term_count(&store, "title", "world"), 3);

        store.delete_document_by_internal_id(2).unwrap();
        store.commit().unwrap();

        assert_eq!(
            term_count(&store, "title", "world"),
            2,
            "deleted document must not be counted"
        );
    }

    /// Issue #610: a non-`TermQuery` (here a Boolean AND) bypasses the fast
    /// path and is counted correctly by the slow path.
    #[test]
    fn test_count_boolean_query_falls_back() {
        use crate::lexical::query::boolean::BooleanQueryBuilder;

        let temp_dir = TempDir::new().unwrap();
        let store = store_with_titles(&temp_dir, &["Hello World", "Goodbye World", "Big World"]);

        // title:world AND title:hello → only "Hello World".
        let q = Box::new(
            BooleanQueryBuilder::new()
                .must(Box::new(TermQuery::new("title", "world")))
                .must(Box::new(TermQuery::new("title", "hello")))
                .build(),
        ) as Box<dyn Query>;
        let count = store.count(LexicalSearchRequest::new(q)).unwrap();
        assert_eq!(count, 1);
    }

    /// Issue #610: a positive `min_score` forces the scoring slow path (the
    /// fast path cannot honour a score threshold). An unreachable threshold
    /// yields zero; a zero threshold counts every match.
    #[test]
    fn test_count_with_min_score_falls_back() {
        let temp_dir = TempDir::new().unwrap();
        let store = store_with_titles(&temp_dir, &["Hello World", "Goodbye World", "Big World"]);

        let all = term_count(&store, "title", "world");
        assert_eq!(all, 3);

        let q = Box::new(TermQuery::new("title", "world")) as Box<dyn Query>;
        let thresholded = store
            .count(LexicalSearchRequest::new(q).min_score(f32::MAX))
            .unwrap();
        assert_eq!(thresholded, 0, "no document scores above f32::MAX");
    }

    /// Issue #610: a term absent from the index counts as zero (handled by the
    /// `is_empty` guard before the fast path).
    #[test]
    fn test_count_nonexistent_term() {
        let temp_dir = TempDir::new().unwrap();
        let store = store_with_titles(&temp_dir, &["Hello World"]);
        assert_eq!(term_count(&store, "title", "nonexistent"), 0);
    }

    /// Issue #1211: a segment with gaps in its id range reports its true live
    /// count once it has a deletion, and the count fast path does not return
    /// the deleted document. `optimize` merges {1,2,3} (2 deleted) and {4}
    /// into {1,3,4} over range 1..4; deleting 3 then made the range width
    /// minus the deletions (3) equal `max_doc`, so the fast path answered
    /// from `doc_freq`, which still counts doc 3.
    #[test]
    fn a_gapped_segment_counts_its_live_documents() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for id in 1..=3u64 {
            store
                .upsert_document(id, create_test_document("world", "body"))
                .unwrap();
        }
        store.commit().unwrap();
        store
            .upsert_document(4, create_test_document("world", "body"))
            .unwrap();
        store.commit().unwrap();
        store.delete_document_by_internal_id(2).unwrap();
        store.commit().unwrap();
        store.optimize().unwrap();
        store.delete_document_by_internal_id(3).unwrap();
        store.commit().unwrap();

        assert_eq!(store.reader_for_tests().unwrap().doc_count(), 2);
        assert_eq!(term_count(&store, "title", "world"), 2);
    }

    /// Issue #1211: when a segment's membership is unknown (no `.ids` and no
    /// `.norms`, as before #555), its live count is an upper bound that never
    /// under-counts — and the count fast path does not take that bound for
    /// "no deletions". Segment {1, 3} with 3 deleted: the bound is 2 (it
    /// cannot tell whether the bit is on a held id), which equals `max_doc`.
    #[test]
    fn unknown_membership_bounds_the_live_count_and_skips_the_fast_path() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config =
            LexicalIndexConfig::Inverted(crate::lexical::index::config::InvertedIndexConfig {
                use_compound: false,
                ..Default::default()
            });
        let store = LexicalStore::new(storage.clone(), config).unwrap();
        for id in [1u64, 3] {
            store
                .upsert_document(id, create_test_document("world", "body"))
                .unwrap();
        }
        store.commit().unwrap();
        store.delete_document_by_internal_id(3).unwrap();
        store.commit().unwrap();
        for file in storage.list_files().unwrap() {
            if file.ends_with(".ids") || file.ends_with(".norms") {
                storage.delete_file(&file).unwrap();
            }
        }
        store.refresh().unwrap();

        let live = store.reader_for_tests().unwrap().doc_count();
        assert!(live >= 1, "the bound must not under-count: {live}");
        assert_eq!(term_count(&store, "title", "world"), 1);
    }

    #[test]
    fn test_engine_refresh() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        // Add a document
        let doc = create_test_document("Test Document", "Test content");
        engine.upsert_document(1, doc).unwrap();
        engine.commit().unwrap();

        // Refresh should not fail
        engine.refresh().unwrap();

        // Search should still work
        let query = Box::new(TermQuery::new("title", "Test")) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query);
        let _results = engine.search(request).unwrap();
    }

    #[test]
    fn test_engine_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        let stats = engine.stats().unwrap();
        assert!(stats.last_modified > 0);
    }

    #[test]
    fn test_engine_close() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        assert!(!engine.is_closed());

        engine.close().unwrap();

        assert!(engine.is_closed());
    }

    #[test]
    fn test_search_request_configuration() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        let query = Box::new(TermQuery::new("title", "hello")) as Box<dyn Query>;
        let request = LexicalSearchRequest::new(query)
            .limit(5)
            .min_score(0.5)
            .load_documents(false);

        let results = engine.search(request).unwrap();

        assert_eq!(results.hits.len(), 0);
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_search_with_query_parser() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        // Add some documents with lowercase titles for testing
        let docs = vec![
            create_test_document("hello world", "This is a test document"),
            create_test_document("goodbye world", "This is another test document"),
        ];
        for (i, doc) in docs.into_iter().enumerate() {
            engine.upsert_document((i + 1) as u64, doc).unwrap();
        }
        engine.commit().unwrap();

        // Search with QueryParser (Lucene style)
        use crate::lexical::query::parser::LexicalQueryParser;
        let parser = LexicalQueryParser::with_standard_analyzer()
            .unwrap()
            .with_default_field("title");

        // QueryParser analyzes "Hello" to "hello" before creating TermQuery
        let query = parser.parse("Hello").unwrap();
        let results = engine.search(LexicalSearchRequest::new(query)).unwrap();

        // Should find the document
        assert_eq!(results.hits.len(), 1);
        assert_eq!(results.total_hits, 1);
    }

    #[test]
    fn test_search_field_with_string() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();

        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        // Search specific field
        use crate::analysis::analyzer::standard::StandardAnalyzer;
        use crate::lexical::query::parser::LexicalQueryParser;
        let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        let parser = LexicalQueryParser::new(analyzer);
        let query = parser.parse_field("title", "hello world").unwrap();
        let results = engine.search(LexicalSearchRequest::new(query)).unwrap();

        // Should not find anything (empty index)
        assert_eq!(results.hits.len(), 0);
    }

    #[test]
    fn test_find_doc_ids_by_term() {
        let temp_dir = TempDir::new().unwrap();
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap(),
        );
        let engine = LexicalStore::new(storage, config).unwrap();

        // Index document with external ID
        let doc = Document::builder()
            .add_text("title", "Test Doc")
            .add_text("_id", "ext_1")
            .build();
        engine.upsert_document(1, doc).unwrap();
        engine.commit().unwrap();

        // Verify find_doc_ids_by_term
        let found_ids = engine.find_doc_ids_by_term("_id", "ext_1").unwrap();
        assert_eq!(found_ids, vec![1]);

        // Non-existent
        let not_found = engine.find_doc_ids_by_term("_id", "ext_999").unwrap();
        assert!(not_found.is_empty());
    }
}
