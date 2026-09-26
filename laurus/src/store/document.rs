//! Segmented storage for documents (Unified).
//!
//! This module provides a way to store and retrieve documents in segments,
//! avoiding the need to keep all documents in memory or a single massive JSON file.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use serde::{Deserialize, Serialize};

use crate::data::Document;
use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::storage::manifest as manifest_io;
use crate::storage::structured::{StructReader, StructWriter};
use crate::util::alloc_bounds::checked_capacity_u64;

/// Default capacity for the document LRU cache.
const DEFAULT_DOC_CACHE_CAPACITY: usize = 1024;

/// A segment of stored documents.
///
/// Each segment represents a contiguous batch of documents that have been flushed
/// to persistent storage as a single binary file. The segment tracks the range of
/// document IDs it contains, enabling efficient lookup without scanning every file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSegment {
    /// Unique identifier for this segment, used to derive the segment file name.
    pub id: u32,
    /// Lowest (inclusive) document ID stored in this segment.
    pub start_doc_id: u64,
    /// Highest (inclusive) document ID stored in this segment.
    pub end_doc_id: u64,
    /// Number of documents stored in this segment.
    pub doc_count: usize,
}

impl DocumentSegment {
    /// Returns the file name for this segment's binary data file.
    ///
    /// The file name is derived from the segment [`id`](Self::id) with zero-padded
    /// formatting (e.g. `doc_segment_000042.docs`).
    ///
    /// # Returns
    ///
    /// A `String` containing the segment file name.
    pub fn file_name(&self) -> String {
        format!("doc_segment_{:06}.docs", self.id)
    }

    /// Checks whether the given document ID falls within this segment's range.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to check.
    ///
    /// # Returns
    ///
    /// `true` if `doc_id` is between [`start_doc_id`](Self::start_doc_id) and
    /// [`end_doc_id`](Self::end_doc_id) (inclusive), `false` otherwise.
    pub fn contains(&self, doc_id: u64) -> bool {
        doc_id >= self.start_doc_id && doc_id <= self.end_doc_id
    }
}

/// Writer for document segments.
#[derive(Debug)]
pub struct DocumentSegmentWriter {
    storage: Arc<dyn Storage>,
}

impl DocumentSegmentWriter {
    /// Creates a new `DocumentSegmentWriter` backed by the given storage.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend used to persist segment files.
    ///
    /// # Returns
    ///
    /// A new `DocumentSegmentWriter` instance.
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }

    /// Writes a set of documents to a new segment file and returns the resulting
    /// [`DocumentSegment`] metadata.
    ///
    /// Documents are serialized to JSON and written in ascending document-ID order
    /// using a simple binary format: `[u32: doc_count] ([u64: doc_id][bytes: json_data])*`.
    ///
    /// # Arguments
    ///
    /// * `segment_id` - The unique ID to assign to the new segment.
    /// * `docs` - A map of document IDs to [`Document`] values to be written.
    ///
    /// # Returns
    ///
    /// A [`DocumentSegment`] describing the segment that was written.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] if `docs` is empty, serialization fails, or the
    /// underlying storage I/O fails.
    pub fn write_segment(
        &self,
        segment_id: u32,
        docs: &HashMap<u64, Document>,
    ) -> Result<DocumentSegment> {
        if docs.is_empty() {
            return Err(LaurusError::invalid_argument(
                "cannot write empty document segment",
            ));
        }

        let mut sorted_ids: Vec<_> = docs.keys().cloned().collect();
        sorted_ids.sort();

        let start_doc_id = *sorted_ids.first().unwrap();
        let end_doc_id = *sorted_ids.last().unwrap();
        let doc_count = docs.len();

        let segment = DocumentSegment {
            id: segment_id,
            start_doc_id,
            end_doc_id,
            doc_count,
        };

        let file_name = segment.file_name();
        let output = self.storage.create_output(&file_name)?;
        let mut writer = StructWriter::new(output);

        // Simple binary format using StructWriter:
        // [u32: doc_count]
        // [u64: doc_id][bytes: json_data] * doc_count

        let doc_count_u32: u32 = doc_count.try_into().map_err(|_| {
            LaurusError::InvalidOperation(format!("document count {doc_count} exceeds u32::MAX"))
        })?;
        writer.write_u32(doc_count_u32)?;
        for id in sorted_ids {
            let doc = docs.get(&id).unwrap();
            let json = serde_json::to_vec(doc)
                .map_err(|e| LaurusError::index(format!("failed to serialize document: {e}")))?;
            writer.write_u64(id)?;
            writer.write_bytes(&json)?;
        }

        writer.close()?;
        Ok(segment)
    }
}

/// Reader for document segments.
///
/// On construction an in-memory offset index (`doc_id -> byte position`) can
/// optionally be built so that subsequent lookups can seek directly to the
/// target document in O(1) instead of performing a linear scan.
#[derive(Debug)]
pub struct DocumentSegmentReader {
    storage: Arc<dyn Storage>,
    segment: DocumentSegment,
    /// doc_id -> byte position of the doc_id field in the segment file.
    /// Built once via [`with_index`](Self::with_index) and reused for all lookups.
    offsets: HashMap<u64, u64>,
}

impl DocumentSegmentReader {
    /// Creates a new `DocumentSegmentReader` for the specified segment.
    ///
    /// The offset index is **not** built; lookups will use linear scan.
    /// Use [`with_index`](Self::with_index) for O(1) lookups.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend from which segment files are read.
    /// * `segment` - The [`DocumentSegment`] metadata describing the segment to read.
    ///
    /// # Returns
    ///
    /// A new `DocumentSegmentReader` instance.
    pub fn new(storage: Arc<dyn Storage>, segment: DocumentSegment) -> Self {
        Self {
            storage,
            segment,
            offsets: HashMap::new(),
        }
    }

    /// Creates a `DocumentSegmentReader` with a pre-built offset index.
    ///
    /// The segment file is scanned once during construction. Each document's
    /// byte offset is recorded so that subsequent lookups can seek directly
    /// rather than scanning.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend from which segment files are read.
    /// * `segment` - The [`DocumentSegment`] metadata describing the segment to read.
    ///
    /// # Returns
    ///
    /// A `DocumentSegmentReader` with an offset index that enables O(1) lookups.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] if the segment file cannot be opened or read.
    pub fn with_index(storage: Arc<dyn Storage>, segment: DocumentSegment) -> Result<Self> {
        let offsets = Self::build_index(&*storage, &segment)?;
        Ok(Self {
            storage,
            segment,
            offsets,
        })
    }

    /// Scans the segment file once and records the byte offset of each document
    /// entry (positioned at the `doc_id` field).
    fn build_index(storage: &dyn Storage, segment: &DocumentSegment) -> Result<HashMap<u64, u64>> {
        let input = storage.open_input(&segment.file_name())?;
        let mut reader = StructReader::new(input)?;
        let doc_count = reader.read_u32()?;
        // Each entry is at least a u64 doc id and a one-byte length prefix
        // (Issue #1220).
        let doc_count = checked_capacity_u64(
            u64::from(doc_count),
            8 + 1,
            reader.remaining(),
            "document segment doc count",
        )?;

        let mut offsets = HashMap::with_capacity(doc_count);
        for _ in 0..doc_count {
            let offset = reader.stream_position()?;
            let doc_id = reader.read_u64()?;
            // Skip the document bytes (varint-prefixed)
            let _json = reader.read_bytes()?;
            offsets.insert(doc_id, offset);
        }
        Ok(offsets)
    }

    /// Retrieves a single document by its internal document ID.
    ///
    /// When an offset index is available the method seeks directly to the
    /// target document in O(1).  Otherwise it falls back to a linear scan.
    ///
    /// If the `doc_id` is outside this segment's range the method returns `Ok(None)`
    /// without performing any I/O.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID to look up.
    ///
    /// # Returns
    ///
    /// `Ok(Some(document))` if found, `Ok(None)` if the document is not in this segment.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on storage I/O or deserialization failure.
    pub fn get_document(&self, doc_id: u64) -> Result<Option<Document>> {
        if !self.segment.contains(doc_id) {
            return Ok(None);
        }

        // Fast path: use offset index for O(1) seek
        if let Some(&offset) = self.offsets.get(&doc_id) {
            let input = self.storage.open_input(&self.segment.file_name())?;
            let mut reader = StructReader::new(input)?;
            reader.seek(std::io::SeekFrom::Start(offset))?;
            let current_id = reader.read_u64()?;
            debug_assert_eq!(current_id, doc_id);
            let json = reader.read_bytes()?;
            let doc: Document = serde_json::from_slice(&json)
                .map_err(|e| LaurusError::index(format!("failed to deserialize document: {e}")))?;
            return Ok(Some(doc));
        }

        // Fallback: linear scan
        let input = self.storage.open_input(&self.segment.file_name())?;
        let mut reader = StructReader::new(input)?;
        let doc_count = reader.read_u32()?;

        for _ in 0..doc_count {
            let current_id = reader.read_u64()?;
            let json = reader.read_bytes()?;
            if current_id == doc_id {
                let doc: Document = serde_json::from_slice(&json).map_err(|e| {
                    LaurusError::index(format!("failed to deserialize document: {e}"))
                })?;
                return Ok(Some(doc));
            }
        }

        Ok(None)
    }

    /// Retrieve multiple documents from this segment.
    ///
    /// When an offset index is available each document is looked up via direct
    /// seek (offsets are sorted for sequential I/O).  Otherwise the method
    /// falls back to a single-pass linear scan.
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Set of document IDs to retrieve.
    ///
    /// # Returns
    ///
    /// A map of doc_id to [`Document`] for all found documents in this segment.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on storage I/O or deserialization failure.
    pub fn get_documents_batch(
        &self,
        doc_ids: &std::collections::HashSet<u64>,
    ) -> Result<HashMap<u64, Document>> {
        let mut results = HashMap::with_capacity(doc_ids.len());
        if doc_ids.is_empty() {
            return Ok(results);
        }

        // Quick check: are any requested IDs within this segment's range?
        if !doc_ids.iter().any(|id| self.segment.contains(*id)) {
            return Ok(results);
        }

        // Fast path: use offset index for direct seeks
        if !self.offsets.is_empty() {
            let input = self.storage.open_input(&self.segment.file_name())?;
            let mut reader = StructReader::new(input)?;

            // Sort offsets for sequential I/O
            let mut indexed: Vec<(u64, u64)> = doc_ids
                .iter()
                .filter_map(|id| self.offsets.get(id).map(|&off| (*id, off)))
                .collect();
            indexed.sort_unstable_by_key(|&(_, off)| off);

            for (doc_id, offset) in indexed {
                reader.seek(std::io::SeekFrom::Start(offset))?;
                let current_id = reader.read_u64()?;
                debug_assert_eq!(current_id, doc_id);
                let json = reader.read_bytes()?;
                let doc: Document = serde_json::from_slice(&json).map_err(|e| {
                    LaurusError::index(format!("failed to deserialize document: {e}"))
                })?;
                results.insert(doc_id, doc);
            }
            return Ok(results);
        }

        // Fallback: linear scan
        let input = self.storage.open_input(&self.segment.file_name())?;
        let mut reader = StructReader::new(input)?;
        let doc_count = reader.read_u32()?;

        let mut remaining = doc_ids.len();
        for _ in 0..doc_count {
            if remaining == 0 {
                break; // All requested docs found, stop early.
            }
            let current_id = reader.read_u64()?;
            let json = reader.read_bytes()?;
            if doc_ids.contains(&current_id) {
                let doc: Document = serde_json::from_slice(&json).map_err(|e| {
                    LaurusError::index(format!("failed to deserialize document: {e}"))
                })?;
                results.insert(current_id, doc);
                remaining -= 1;
            }
        }
        Ok(results)
    }

    /// Finds the first internal document ID whose `_id` field matches the given external ID.
    ///
    /// The method performs a linear scan over all documents in this segment.
    ///
    /// # Arguments
    ///
    /// * `external_id` - The external document identifier to search for (value of the `_id` field).
    ///
    /// # Returns
    ///
    /// `Ok(Some(doc_id))` if a matching document is found, `Ok(None)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on storage I/O or deserialization failure.
    pub fn find_by_external_id(&self, external_id: &str) -> Result<Option<u64>> {
        let input = self.storage.open_input(&self.segment.file_name())?;
        let mut reader = StructReader::new(input)?;
        let doc_count = reader.read_u32()?;

        for _ in 0..doc_count {
            let current_id = reader.read_u64()?;
            let json = reader.read_bytes()?;
            let doc: Document = serde_json::from_slice(&json)
                .map_err(|e| LaurusError::index(format!("failed to deserialize document: {e}")))?;
            if doc.fields.get("_id").and_then(|v| v.as_text()) == Some(external_id) {
                return Ok(Some(current_id));
            }
        }

        Ok(None)
    }

    /// Finds all internal document IDs whose `_id` field matches the given external ID.
    ///
    /// Unlike [`find_by_external_id`](Self::find_by_external_id) this method does not
    /// stop at the first match and returns every matching document ID in the segment.
    ///
    /// # Arguments
    ///
    /// * `external_id` - The external document identifier to search for (value of the `_id` field).
    ///
    /// # Returns
    ///
    /// A `Vec<u64>` of all matching internal document IDs (may be empty).
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on storage I/O or deserialization failure.
    pub fn find_all_by_external_id(&self, external_id: &str) -> Result<Vec<u64>> {
        let input = self.storage.open_input(&self.segment.file_name())?;
        let mut reader = StructReader::new(input)?;
        let doc_count = reader.read_u32()?;
        let mut results = Vec::new();

        for _ in 0..doc_count {
            let current_id = reader.read_u64()?;
            let json = reader.read_bytes()?;
            let doc: Document = serde_json::from_slice(&json)
                .map_err(|e| LaurusError::index(format!("failed to deserialize document: {e}")))?;
            if doc.fields.get("_id").and_then(|v| v.as_text()) == Some(external_id) {
                results.push(current_id);
            }
        }

        Ok(results)
    }
}

const MANIFEST_FILE: &str = "segments.json";

#[derive(Debug, Serialize, Deserialize)]
struct StoreManifest {
    version: u32,
    segments: Vec<DocumentSegment>,
    next_segment_id: u32,
}

/// Unified segmented document store.
///
/// `UnifiedDocumentStore` manages document persistence across multiple binary segment
/// files. Newly added documents are held in an in-memory pending buffer until
/// [`commit`](Self::commit) is called, at which point they are flushed to a new segment
/// file and the manifest is atomically updated.
///
/// A JSON manifest (`segments.json`) tracks all committed segments and the next
/// segment ID so that the store can be re-opened across process restarts.
#[derive(Debug)]
pub struct UnifiedDocumentStore {
    storage: Arc<dyn Storage>,
    segments: Vec<DocumentSegment>,
    next_segment_id: u32,
    pending_docs: HashMap<u64, Document>,
    next_doc_id: u64,
    /// Cached segment readers with pre-built offset indexes.
    /// Keyed by segment ID to avoid rebuilding the index on every lookup.
    reader_cache: HashMap<u32, DocumentSegmentReader>,
    /// LRU cache for recently accessed documents, avoiding repeated I/O
    /// for hot documents.  Wrapped in `parking_lot::Mutex` so that
    /// [`get_document`](Self::get_document) can remain `&self`.
    doc_cache: parking_lot::Mutex<LruCache<u64, Document>>,
}

impl UnifiedDocumentStore {
    /// Creates a new, empty `UnifiedDocumentStore`.
    ///
    /// No manifest file is read or written; the store starts with zero segments and
    /// document IDs beginning at 1.
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend for segment and manifest files.
    ///
    /// # Returns
    ///
    /// A fresh `UnifiedDocumentStore` instance.
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        // SAFETY: DEFAULT_DOC_CACHE_CAPACITY is a compile-time constant > 0.
        let cap = NonZeroUsize::new(DEFAULT_DOC_CACHE_CAPACITY).unwrap();
        Self {
            storage,
            segments: Vec::new(),
            next_segment_id: 0,
            pending_docs: HashMap::new(),
            next_doc_id: 1,
            reader_cache: HashMap::new(),
            doc_cache: parking_lot::Mutex::new(LruCache::new(cap)),
        }
    }

    /// Opens an existing document store from the given storage backend.
    ///
    /// If a manifest file (`segments.json`) exists it is read and the segment list and
    /// ID counters are restored. Otherwise a fresh, empty store is returned (equivalent
    /// to calling [`new`](Self::new)).
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend containing the manifest and segment files.
    ///
    /// # Returns
    ///
    /// A `UnifiedDocumentStore` populated from the persisted manifest.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] if the manifest file exists but cannot be read or
    /// deserialized.
    pub fn open(storage: Arc<dyn Storage>) -> Result<Self> {
        // The CRC-32 trailer this writes is now verified on the way back in
        // (#1022): it used to be written and never checked, so a corrupted
        // manifest was read as though it were valid.
        if let Some((manifest, _format)) = manifest_io::load_checksummed_json::<StoreManifest>(
            storage.as_ref(),
            MANIFEST_FILE,
            None,
        )? {
            let mut next_doc_id = 1;
            for segment in &manifest.segments {
                if segment.end_doc_id >= next_doc_id {
                    next_doc_id = segment.end_doc_id + 1;
                }
            }

            // SAFETY: DEFAULT_DOC_CACHE_CAPACITY is a compile-time constant > 0.
            let cap = NonZeroUsize::new(DEFAULT_DOC_CACHE_CAPACITY).unwrap();
            Ok(Self {
                storage,
                segments: manifest.segments,
                next_segment_id: manifest.next_segment_id,
                pending_docs: HashMap::new(),
                next_doc_id,
                reader_cache: HashMap::new(),
                doc_cache: parking_lot::Mutex::new(LruCache::new(cap)),
            })
        } else {
            Ok(Self::new(storage))
        }
    }

    /// Flushes pending documents to a new segment and atomically updates the manifest.
    ///
    /// If there are no pending documents the manifest is still written so that any
    /// previously added segments are persisted. After the manifest is written the
    /// storage is synced to ensure durability.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on serialization or storage I/O failure.
    pub fn commit(&mut self) -> Result<()> {
        // Flush pending documents if any
        if !self.pending_docs.is_empty() {
            let docs = std::mem::take(&mut self.pending_docs);
            self.add_segment(&docs)?;
        }

        // Build offset indexes for any segments not yet cached.
        for segment in &self.segments {
            if self.reader_cache.contains_key(&segment.id) {
                continue;
            }
            if let Ok(reader) =
                DocumentSegmentReader::with_index(self.storage.clone(), segment.clone())
            {
                self.reader_cache.insert(segment.id, reader);
            }
        }

        let manifest = StoreManifest {
            version: 1,
            segments: self.segments.clone(),
            next_segment_id: self.next_segment_id,
        };

        // temp + CRC-32 trailer + rename + sync, shared with every other
        // control file in the tree (#1022). The sync matters beyond
        // durability here: on Windows a cached directory listing would hide
        // the renamed manifest and the new segment files from later reads.
        manifest_io::save_checksummed_json(self.storage.as_ref(), MANIFEST_FILE, None, &manifest)
    }

    /// Adds a document to the pending buffer and assigns it a new internal document ID.
    ///
    /// The document is **not** written to storage until [`commit`](Self::commit) is called.
    ///
    /// # Arguments
    ///
    /// * `doc` - The [`Document`] to add.
    ///
    /// # Returns
    ///
    /// The newly assigned internal document ID.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for forward compatibility.
    pub fn add_document(&mut self, doc: Document) -> Result<u64> {
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;
        self.pending_docs.insert(doc_id, doc);
        // Flushing is intentionally left to the caller via `commit()` to give full
        // control over transaction boundaries and batch sizes.
        Ok(doc_id)
    }

    /// Get the current next_doc_id counter.
    ///
    /// Used by [`DocumentLog`](super::log::DocumentLog) to sync its own
    /// counter with committed document store segments on startup.
    pub fn next_doc_id(&self) -> u64 {
        self.next_doc_id
    }

    /// Insert a document with a specific doc_id (used during WAL recovery).
    ///
    /// Updates `next_doc_id` if the given `doc_id` is >= current counter
    /// to avoid ID conflicts on subsequent `add_document()` calls.
    pub fn put_document_with_id(&mut self, doc_id: u64, doc: Document) {
        self.pending_docs.insert(doc_id, doc);
        // Invalidate stale cache entry
        self.doc_cache.get_mut().pop(&doc_id);
        if doc_id >= self.next_doc_id {
            self.next_doc_id = doc_id + 1;
        }
    }

    /// Writes a set of documents into a new segment file and registers it in the store.
    ///
    /// This is a lower-level method; most callers should use [`add_document`](Self::add_document)
    /// followed by [`commit`](Self::commit) instead.
    ///
    /// # Arguments
    ///
    /// * `docs` - A map of internal document IDs to [`Document`] values.
    ///
    /// # Returns
    ///
    /// The [`DocumentSegment`] metadata for the newly created segment.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] if `docs` is empty or the segment write fails.
    pub fn add_segment(&mut self, docs: &HashMap<u64, Document>) -> Result<DocumentSegment> {
        let writer = DocumentSegmentWriter::new(self.storage.clone());
        let segment = writer.write_segment(self.next_segment_id, docs)?;
        self.segments.push(segment.clone());
        self.next_segment_id += 1;
        Ok(segment)
    }

    /// Retrieves a document by its internal document ID.
    ///
    /// Pending (uncommitted) documents are checked first, followed by committed
    /// segments in reverse order (newest first).
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID.
    ///
    /// # Returns
    ///
    /// `Ok(Some(document))` if found, `Ok(None)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on storage I/O or deserialization failure.
    pub fn get_document(&self, doc_id: u64) -> Result<Option<Document>> {
        // Check pending docs first
        if let Some(doc) = self.pending_docs.get(&doc_id) {
            return Ok(Some(doc.clone()));
        }

        // Check LRU cache
        {
            let mut cache = self.doc_cache.lock();
            if let Some(doc) = cache.get(&doc_id) {
                return Ok(Some(doc.clone()));
            }
        }

        // Search segments; prefer cached reader with offset index.
        for segment in self.segments.iter().rev() {
            if segment.contains(doc_id) {
                let doc_opt = if let Some(reader) = self.reader_cache.get(&segment.id) {
                    reader.get_document(doc_id)?
                } else {
                    let reader = DocumentSegmentReader::new(self.storage.clone(), segment.clone());
                    reader.get_document(doc_id)?
                };

                if let Some(doc) = doc_opt {
                    // Insert into LRU cache
                    self.doc_cache.lock().put(doc_id, doc.clone());
                    return Ok(Some(doc));
                }
            }
        }
        Ok(None)
    }

    /// Retrieve multiple documents by their internal IDs in a single batch.
    ///
    /// More efficient than individual [`get_document()`](Self::get_document) calls because
    /// each segment file is opened and scanned only once.
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Slice of internal document IDs to retrieve.
    ///
    /// # Returns
    ///
    /// A map of doc_id to [`Document`] for all found documents.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on storage I/O or deserialization failure.
    pub fn get_documents_batch(&self, doc_ids: &[u64]) -> Result<HashMap<u64, Document>> {
        let mut results = HashMap::with_capacity(doc_ids.len());
        if doc_ids.is_empty() {
            return Ok(results);
        }

        let id_set: std::collections::HashSet<u64> = doc_ids.iter().copied().collect();

        // Check pending docs first.
        for &doc_id in doc_ids {
            if let Some(doc) = self.pending_docs.get(&doc_id) {
                results.insert(doc_id, doc.clone());
            }
        }

        // Then the LRU cache, under a single lock for the whole batch
        // rather than one lock per document as `get_document` takes
        // (#1010). Without this the batch path bypassed the cache
        // entirely, so wiring it into the search path would have made
        // repeated / paginated queries slower.
        {
            let mut cache = self.doc_cache.lock();
            for &doc_id in &id_set {
                if results.contains_key(&doc_id) {
                    continue;
                }
                if let Some(doc) = cache.get(&doc_id) {
                    results.insert(doc_id, doc.clone());
                }
            }
        }

        // Find remaining IDs not yet resolved.
        let remaining: std::collections::HashSet<u64> = id_set
            .iter()
            .filter(|id| !results.contains_key(id))
            .copied()
            .collect();

        if remaining.is_empty() {
            return Ok(results);
        }

        // Batch-load from segments (one file open per segment). Segments
        // are ordered oldest-first and a later one overwrites an earlier
        // hit, matching `get_document`'s newest-wins resolution (which
        // reaches the same answer by scanning `.rev()` and taking the
        // first hit).
        let mut loaded: HashMap<u64, Document> = HashMap::new();
        for segment in &self.segments {
            let segment_ids: std::collections::HashSet<u64> = remaining
                .iter()
                .filter(|id| segment.contains(**id))
                .copied()
                .collect();
            if segment_ids.is_empty() {
                continue;
            }

            if let Some(reader) = self.reader_cache.get(&segment.id) {
                let batch = reader.get_documents_batch(&segment_ids)?;
                loaded.extend(batch);
            } else {
                let reader = DocumentSegmentReader::new(self.storage.clone(), segment.clone());
                let batch = reader.get_documents_batch(&segment_ids)?;
                loaded.extend(batch);
            }
        }

        // Populate the LRU with what we just read, mirroring
        // `get_document` (#1010), so a repeat of the same query is served
        // from cache.
        if !loaded.is_empty() {
            let mut cache = self.doc_cache.lock();
            for (doc_id, doc) in &loaded {
                cache.put(*doc_id, doc.clone());
            }
        }
        results.extend(loaded);

        Ok(results)
    }

    /// Finds the first internal document ID whose `_id` field matches the given external ID.
    ///
    /// Pending documents are searched first, then committed segments in reverse order.
    ///
    /// # Arguments
    ///
    /// * `external_id` - The external document identifier to search for.
    ///
    /// # Returns
    ///
    /// `Ok(Some(doc_id))` if a matching document is found, `Ok(None)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on storage I/O or deserialization failure.
    pub fn find_by_external_id(&self, external_id: &str) -> Result<Option<u64>> {
        // Check pending docs first
        for (id, doc) in &self.pending_docs {
            if doc.fields.get("_id").and_then(|v| v.as_text()) == Some(external_id) {
                return Ok(Some(*id));
            }
        }

        for segment in self.segments.iter().rev() {
            let reader = DocumentSegmentReader::new(self.storage.clone(), segment.clone());
            if let Some(id) = reader.find_by_external_id(external_id)? {
                return Ok(Some(id));
            }
        }
        Ok(None)
    }

    /// Finds all internal document IDs whose `_id` field matches the given external ID.
    ///
    /// Both pending documents and all committed segments are searched.
    ///
    /// # Arguments
    ///
    /// * `external_id` - The external document identifier to search for.
    ///
    /// # Returns
    ///
    /// A `Vec<u64>` of all matching internal document IDs (may be empty).
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on storage I/O or deserialization failure.
    pub fn find_all_by_external_id(&self, external_id: &str) -> Result<Vec<u64>> {
        let mut results = Vec::new();

        // Check pending docs
        for (id, doc) in &self.pending_docs {
            if doc.fields.get("_id").and_then(|v| v.as_text()) == Some(external_id) {
                results.push(*id);
            }
        }

        for segment in self.segments.iter() {
            let reader = DocumentSegmentReader::new(self.storage.clone(), segment.clone());
            results.extend(reader.find_all_by_external_id(external_id)?);
        }
        Ok(results)
    }

    /// Marks a document as deleted.
    ///
    /// Logical deletion is handled externally by the deletion bitmap / deletion manager;
    /// this method is a no-op placeholder that exists for API symmetry.
    ///
    /// # Arguments
    ///
    /// * `_doc_id` - The internal document ID to delete (currently unused).
    ///
    /// # Errors
    ///
    /// Currently infallible.
    pub fn delete_document(&mut self, _doc_id: u64) -> Result<()> {
        // Logical deletion is handled by DeletionBitmap/DeletionManager.
        Ok(())
    }

    /// Returns a slice of all committed [`DocumentSegment`]s.
    ///
    /// # Returns
    ///
    /// A borrowed slice of segment metadata, ordered by creation time.
    pub fn segments(&self) -> &[DocumentSegment] {
        &self.segments
    }

    /// Deletes the underlying data file for the segment with the given ID.
    ///
    /// If no segment with `segment_id` exists in the store the call is a no-op.
    ///
    /// # Arguments
    ///
    /// * `segment_id` - The ID of the segment whose file should be removed.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] if the storage backend fails to delete the file.
    pub fn delete_segment_files(&self, segment_id: u32) -> Result<()> {
        if let Some(segment) = self.segments.iter().find(|s| s.id == segment_id) {
            self.storage.delete_file(&segment.file_name())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};

    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    /// A segment file's document count is bounded by the file before the
    /// offset index is sized from it (Issue #1220). One million documents
    /// cannot fit in a file holding one; an unbounded reader reserves the
    /// index and then runs out of file instead.
    #[test]
    fn a_doc_count_the_segment_file_cannot_hold_is_rejected() {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let docs = HashMap::from([(1u64, Document::builder().add_text("title", "a").build())]);
        let segment = DocumentSegmentWriter::new(storage.clone())
            .write_segment(0, &docs)
            .unwrap();
        assert!(DocumentSegmentReader::with_index(storage.clone(), segment.clone()).is_ok());

        let mut bytes = Vec::new();
        storage
            .open_input(&segment.file_name())
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes[0..4].copy_from_slice(&1_000_000u32.to_le_bytes());
        let mut output = storage.create_output(&segment.file_name()).unwrap();
        output.write_all(&bytes).unwrap();
        output.close().unwrap();

        match DocumentSegmentReader::with_index(storage, segment) {
            Err(LaurusError::Index(msg)) => {
                assert!(msg.contains("document segment doc count"), "{msg}");
            }
            Err(other) => panic!("expected Index error, got {other:?}"),
            Ok(_) => panic!("a doc count the file cannot hold must be rejected"),
        }
    }
}
