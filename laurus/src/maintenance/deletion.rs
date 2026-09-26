//! Document deletion and compaction system.
//!
//! This module provides efficient document deletion using set-based
//! logical deletion and periodic compaction for space reclamation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use ahash::AHashMap;
use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::storage::structured::{StructReader, StructWriter};
use crate::storage::{Storage, StorageInput, StorageOutput};

/// Configuration for deletion management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionConfig {
    /// Compaction threshold (deletion ratio 0.0-1.0).
    pub compaction_threshold: f64,

    /// Enable automatic compaction.
    pub auto_compaction: bool,

    /// Compaction check interval in seconds.
    pub compaction_interval_secs: u64,

    /// Maximum memory for deletion bitmaps (in MB).
    pub max_bitmap_memory_mb: u64,

    /// Batch size for deletion operations.
    pub deletion_batch_size: usize,

    /// Enable deletion log for recovery.
    pub enable_deletion_log: bool,
}

impl Default for DeletionConfig {
    fn default() -> Self {
        DeletionConfig {
            compaction_threshold: 0.3,
            auto_compaction: true,
            compaction_interval_secs: 300, // 5 minutes
            max_bitmap_memory_mb: 64,
            deletion_batch_size: 1000,
            enable_deletion_log: true,
        }
    }
}

/// A hash set-based deletion tracker for a segment.
#[derive(Debug)]
pub struct DeletionBitmap {
    /// Segment ID this bitmap belongs to.
    pub segment_id: String,

    /// Set of deleted document IDs, stored as a Roaring bitmap (Issue #684).
    ///
    /// `RoaringTreemap` (u64) is far more compact than the previous
    /// `AHashSet<u64>` for the dense deletion sets that accumulate over a
    /// segment's life, and `contains` is a branch-light bit test rather than a
    /// hashed probe — measurable on the per-doc / per-neighbour `is_deleted`
    /// hot paths.
    pub deleted_docs: RwLock<RoaringTreemap>,

    /// Total number of documents in the segment.
    pub total_docs: AtomicU64,

    /// Minimum document ID in this segment.
    pub min_doc_id: u64,

    /// Maximum document ID in this segment.
    pub max_doc_id: u64,

    /// Number of deleted documents.
    pub deleted_count: AtomicU64,

    /// Timestamp of last modification.
    pub last_modified: AtomicU64,

    /// Version number for consistency.
    pub version: AtomicU64,
}

impl DeletionBitmap {
    /// Create a new deletion bitmap for a segment.
    pub fn new(segment_id: String, min_doc_id: u64, max_doc_id: u64) -> Self {
        let total_docs = if max_doc_id >= min_doc_id {
            max_doc_id - min_doc_id + 1
        } else {
            0
        };
        DeletionBitmap {
            segment_id,
            deleted_docs: RwLock::new(RoaringTreemap::new()),
            total_docs: AtomicU64::new(total_docs),
            min_doc_id,
            max_doc_id,
            deleted_count: AtomicU64::new(0),
            last_modified: AtomicU64::new(crate::util::time::now_secs()),
            version: AtomicU64::new(1),
        }
    }

    /// Mark a document as deleted.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to mark as deleted. Must be within the
    ///   `[min_doc_id, max_doc_id]` range of this segment.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the document was newly deleted, or `Ok(false)` if
    /// it was already marked as deleted (idempotent). Returns `Err` if the
    /// document ID is outside this segment's range.
    pub fn delete_document(&self, doc_id: u64) -> Result<bool> {
        // Range check
        if doc_id < self.min_doc_id || doc_id > self.max_doc_id {
            return Err(LaurusError::index(format!(
                "Document ID {doc_id} is out of range [{}, {}] for segment {}",
                self.min_doc_id, self.max_doc_id, self.segment_id
            )));
        }

        let mut docs = self.deleted_docs.write().unwrap();
        // `RoaringTreemap::insert` returns `true` when the id was newly added.
        let newly_deleted = docs.insert(doc_id);
        if newly_deleted {
            self.deleted_count.fetch_add(1, Ordering::SeqCst);
            self.last_modified
                .store(crate::util::time::now_secs(), Ordering::SeqCst);
            self.version.fetch_add(1, Ordering::SeqCst);
        }

        Ok(newly_deleted)
    }

    /// Clear a document's deletion mark (Issue #880).
    ///
    /// Used by the segmented vector path's same-id upsert dance: the
    /// delete-first step marks the id so sealed-segment copies stop matching,
    /// and the following re-add clears the mark so the *new* copy is not
    /// shadowed by its own delete once flushed. The revived old copies are
    /// masked by newest-generation-wins deduplication at search and removed
    /// physically at merge.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to unmark. Must be within the
    ///   `[min_doc_id, max_doc_id]` range of this segment.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the document had been marked deleted, or
    /// `Ok(false)` if it was not marked (idempotent). Returns `Err` if the
    /// document ID is outside this segment's range.
    pub fn undelete_document(&self, doc_id: u64) -> Result<bool> {
        // Range check
        if doc_id < self.min_doc_id || doc_id > self.max_doc_id {
            return Err(LaurusError::index(format!(
                "Document ID {doc_id} is out of range [{}, {}] for segment {}",
                self.min_doc_id, self.max_doc_id, self.segment_id
            )));
        }

        let mut docs = self.deleted_docs.write().unwrap();
        // `RoaringTreemap::remove` returns `true` when the id was present.
        let was_deleted = docs.remove(doc_id);
        if was_deleted {
            self.deleted_count.fetch_sub(1, Ordering::SeqCst);
            self.last_modified
                .store(crate::util::time::now_secs(), Ordering::SeqCst);
            self.version.fetch_add(1, Ordering::SeqCst);
        }

        Ok(was_deleted)
    }

    /// Resize the bitmap to accommodate more documents.
    pub fn resize(&self, new_size: u64) {
        self.total_docs.store(new_size, Ordering::SeqCst);
    }

    /// Check if a document is deleted.
    pub fn is_deleted(&self, doc_id: u64) -> bool {
        self.deleted_docs.read().unwrap().contains(doc_id)
    }

    /// Get deletion ratio (0.0 to 1.0).
    pub fn deletion_ratio(&self) -> f64 {
        if self.total_docs.load(Ordering::SeqCst) == 0 {
            0.0
        } else {
            self.deleted_count.load(Ordering::SeqCst) as f64
                / self.total_docs.load(Ordering::SeqCst) as f64
        }
    }

    /// Number of deleted ids inside `[min_doc_id, max_doc_id]` (Issue #1211).
    ///
    /// Counted from the set itself, not from the header's `deleted_count`: a
    /// v1/v2 `.delmap` infers its range from its own bits, and nothing checks
    /// a loaded bitmap against its segment's range, so bits outside it (or a
    /// stale counter) are possible.
    pub fn deleted_in_range(&self, min_doc_id: u64, max_doc_id: u64) -> u64 {
        self.deleted_docs
            .read()
            .unwrap()
            .range_cardinality(min_doc_id..=max_doc_id)
    }

    /// Number of `ids` marked deleted — the deletions that hit documents a
    /// segment actually holds, when `ids` is its doc-id set (Issue #1211).
    pub fn deleted_count_in(&self, ids: &RoaringTreemap) -> u64 {
        self.deleted_docs.read().unwrap().intersection_len(ids)
    }

    /// Get number of live (non-deleted) documents, assuming the segment
    /// holds every id of its range.
    ///
    /// `total_docs` is the range width, so this over-counts a segment whose
    /// ids have gaps; `SegmentReader::doc_count` computes the live count from
    /// the documents the segment holds instead (Issue #1211).
    pub fn live_count(&self) -> u64 {
        self.total_docs.load(Ordering::SeqCst) - self.deleted_count.load(Ordering::SeqCst)
    }

    /// Check if compaction is needed.
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.deletion_ratio() > threshold
    }

    /// Get all deleted document IDs, in ascending order.
    pub fn get_deleted_docs(&self) -> Vec<u64> {
        let docs = self.deleted_docs.read().unwrap();
        docs.iter().collect()
    }

    /// Consume this bitmap and hand over its deleted-id set directly.
    ///
    /// Unlike [`Self::get_deleted_docs`], which collects the ids into a
    /// `Vec<u64>`, this moves the underlying `RoaringTreemap` out — no
    /// clone, no intermediate allocation. Use it when the caller only
    /// needs membership tests and a count, which the treemap answers
    /// directly and far more compactly (#541).
    ///
    /// # Returns
    ///
    /// The owned set of deleted document IDs.
    pub fn into_deleted_docs(self) -> RoaringTreemap {
        // A poisoned lock still holds a valid bitmap — only a writer
        // panicked — so recover rather than propagate.
        self.deleted_docs
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Get an approximate memory usage of this deletion tracker in bytes.
    ///
    /// The estimate includes the struct itself, the segment ID string buffer,
    /// and the Roaring bitmap's serialized size (Issue #684) as a proxy for its
    /// in-memory footprint — far more accurate, and far smaller for dense
    /// deletion sets, than the previous `AHashSet::capacity()` heuristic.
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.deleted_docs.read().unwrap().serialized_size()
            + self.segment_id.capacity()
    }

    /// Write bitmap to storage.
    ///
    /// Writes the v4 format (Issue #684): the metadata header is unchanged from
    /// v3, but the deleted-id set is a `RoaringTreemap::serialize_into` payload
    /// rather than a raw `u64` list — orders of magnitude smaller for dense
    /// deletion sets. v1/v2/v3 remain readable (see [`Self::read_from_storage`]).
    pub fn write_to_storage<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        // Write header
        writer.write_u32(0x44454C42)?; // "DELB" - Deletion Bitmap
        writer.write_u32(4)?; // Version 4 (Roaring bitmap with min/max doc_id)

        // Write metadata
        writer.write_string(&self.segment_id)?;
        writer.write_u64(self.total_docs.load(Ordering::SeqCst))?;
        writer.write_u64(self.deleted_count.load(Ordering::SeqCst))?;
        writer.write_u64(self.last_modified.load(Ordering::SeqCst))?;
        writer.write_u64(self.version.load(Ordering::SeqCst))?;
        writer.write_u64(self.min_doc_id)?;
        writer.write_u64(self.max_doc_id)?;

        // Write the deleted-id set as a Roaring payload (length-prefixed bytes).
        let docs = self.deleted_docs.read().unwrap();
        let mut payload = Vec::with_capacity(docs.serialized_size());
        docs.serialize_into(&mut payload)
            .map_err(|e| LaurusError::index(format!("Failed to serialize deletion bitmap: {e}")))?;
        writer.write_bytes(&payload)?;

        Ok(())
    }

    /// Read bitmap from storage.
    pub fn read_from_storage<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        // Read header
        let magic = reader.read_u32()?;
        if magic != 0x44454C42 {
            return Err(LaurusError::index("Invalid deletion bitmap format"));
        }

        let version = reader.read_u32()?;
        if version == 1 {
            // Legacy BitVec format
            let segment_id = reader.read_string()?;
            let total_docs = reader.read_u64()?;
            let deleted_count = reader.read_u64()?;
            let last_modified = reader.read_u64()?;
            let bitmap_version = reader.read_u64()?;

            let _bitmap_size = reader.read_varint()? as usize;
            let bitmap_bytes = reader.read_bytes()?;
            let bitvec = bit_vec::BitVec::from_bytes(&bitmap_bytes);

            let mut deleted_docs = RoaringTreemap::new();
            let mut min_doc_id = u64::MAX;
            let mut max_doc_id = 0;
            for (idx, bit) in bitvec.iter().enumerate() {
                if bit {
                    let doc_id = idx as u64;
                    deleted_docs.insert(doc_id);
                    min_doc_id = min_doc_id.min(doc_id);
                    max_doc_id = max_doc_id.max(doc_id);
                }
            }
            // If no docs were deleted, min/max might be default values,
            // but total_docs should give a hint for the range.
            // For version 1, we don't have explicit min/max, so we infer.
            // If total_docs is 0, min/max can be 0. Otherwise, assume 0 to total_docs-1.
            if total_docs > 0 && deleted_docs.is_empty() {
                min_doc_id = 0;
                max_doc_id = total_docs - 1;
            } else if total_docs == 0 {
                min_doc_id = 0;
                max_doc_id = 0;
            }

            Ok(DeletionBitmap {
                segment_id,
                deleted_docs: RwLock::new(deleted_docs),
                total_docs: AtomicU64::new(total_docs),
                min_doc_id,
                max_doc_id,
                deleted_count: AtomicU64::new(deleted_count),
                last_modified: AtomicU64::new(last_modified),
                version: AtomicU64::new(bitmap_version),
            })
        } else if version == 2 {
            // New HashSet format
            let segment_id = reader.read_string()?;
            let total_docs = reader.read_u64()?;
            let deleted_count = reader.read_u64()?;
            let last_modified = reader.read_u64()?;
            let bitmap_version = reader.read_u64()?;

            let deleted_id_count = reader.read_varint()? as usize;
            let _ = deleted_id_count; // count is informational; Roaring grows as needed
            let mut deleted_docs = RoaringTreemap::new();
            let mut min_doc_id = u64::MAX;
            let mut max_doc_id = 0;
            for _ in 0..deleted_id_count {
                let doc_id = reader.read_u64()?;
                deleted_docs.insert(doc_id);
                min_doc_id = min_doc_id.min(doc_id);
                max_doc_id = max_doc_id.max(doc_id);
            }
            // For version 2, we don't have explicit min/max, so we infer.
            // If total_docs is 0, min/max can be 0. Otherwise, assume 0 to total_docs-1.
            if total_docs > 0 && deleted_docs.is_empty() {
                min_doc_id = 0;
                max_doc_id = total_docs - 1;
            } else if total_docs == 0 {
                min_doc_id = 0;
                max_doc_id = 0;
            }

            Ok(DeletionBitmap {
                segment_id,
                deleted_docs: RwLock::new(deleted_docs),
                total_docs: AtomicU64::new(total_docs),
                min_doc_id,
                max_doc_id,
                deleted_count: AtomicU64::new(deleted_count),
                last_modified: AtomicU64::new(last_modified),
                version: AtomicU64::new(bitmap_version),
            })
        } else if version == 3 {
            // Version 3 (HashSet based with min/max doc_id)
            let segment_id = reader.read_string()?;
            let total_docs = reader.read_u64()?;
            let deleted_count = reader.read_u64()?;
            let last_modified = reader.read_u64()?;
            let bitmap_version = reader.read_u64()?;
            let min_doc_id = reader.read_u64()?;
            let max_doc_id = reader.read_u64()?;

            let deleted_id_count = reader.read_varint()? as usize;
            let _ = deleted_id_count; // count is informational; Roaring grows as needed
            let mut deleted_docs = RoaringTreemap::new();
            for _ in 0..deleted_id_count {
                deleted_docs.insert(reader.read_u64()?);
            }

            Ok(DeletionBitmap {
                segment_id,
                deleted_docs: RwLock::new(deleted_docs),
                total_docs: AtomicU64::new(total_docs),
                min_doc_id,
                max_doc_id,
                deleted_count: AtomicU64::new(deleted_count),
                last_modified: AtomicU64::new(last_modified),
                version: AtomicU64::new(bitmap_version),
            })
        } else if version == 4 {
            // Version 4 (Roaring bitmap with min/max doc_id) — Issue #684.
            let segment_id = reader.read_string()?;
            let total_docs = reader.read_u64()?;
            let deleted_count = reader.read_u64()?;
            let last_modified = reader.read_u64()?;
            let bitmap_version = reader.read_u64()?;
            let min_doc_id = reader.read_u64()?;
            let max_doc_id = reader.read_u64()?;

            let payload = reader.read_bytes()?;
            let deleted_docs = RoaringTreemap::deserialize_from(&payload[..]).map_err(|e| {
                LaurusError::index(format!("Failed to deserialize deletion bitmap: {e}"))
            })?;

            Ok(DeletionBitmap {
                segment_id,
                deleted_docs: RwLock::new(deleted_docs),
                total_docs: AtomicU64::new(total_docs),
                min_doc_id,
                max_doc_id,
                deleted_count: AtomicU64::new(deleted_count),
                last_modified: AtomicU64::new(last_modified),
                version: AtomicU64::new(bitmap_version),
            })
        } else {
            Err(LaurusError::index(format!(
                "Unsupported bitmap version: {version}"
            )))
        }
    }
}

/// Entry in the deletion log for recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionLogEntry {
    /// Timestamp of the deletion.
    pub timestamp: u64,

    /// Segment ID.
    pub segment_id: String,

    /// Document ID that was deleted.
    pub doc_id: u64,

    /// Reason for deletion.
    pub reason: String,

    /// Log sequence number.
    pub sequence: u64,
}

/// Log for tracking deletion operations.
#[derive(Debug)]
pub struct DeletionLog {
    /// Storage backend.
    storage: Arc<dyn Storage>,

    /// Current sequence number.
    sequence: std::sync::atomic::AtomicU64,

    /// Log file path.
    log_path: String,
}

impl DeletionLog {
    /// Create a new deletion log.
    pub fn new(storage: Arc<dyn Storage>, log_path: String) -> Result<Self> {
        let log = DeletionLog {
            storage,
            sequence: std::sync::atomic::AtomicU64::new(0),
            log_path,
        };

        // Load existing sequence number
        log.load_sequence()?;

        Ok(log)
    }

    /// Log a deletion operation.
    pub fn log_deletion(&self, segment_id: &str, doc_id: u64, reason: &str) -> Result<()> {
        let entry = DeletionLogEntry {
            timestamp: crate::util::time::now_secs(),
            segment_id: segment_id.to_string(),
            doc_id,
            reason: reason.to_string(),
            sequence: self
                .sequence
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        };

        // Append to log file
        let output = self.storage.create_output_append(&self.log_path)?;
        let mut writer = StructWriter::new(output);

        // Write entry
        let json = serde_json::to_string(&entry)?;
        writer.write_string(&json)?;
        writer.write_u8(b'\n')?; // Newline separator
        writer.close()?;

        Ok(())
    }

    /// Load sequence number from existing log.
    fn load_sequence(&self) -> Result<()> {
        if let Ok(input) = self.storage.open_input(&self.log_path) {
            let mut reader = StructReader::new(input)?;
            let mut max_sequence = 0;

            // Read all entries to find max sequence
            while !reader.is_eof() {
                if let Ok(json) = reader.read_string() {
                    if let Ok(entry) = serde_json::from_str::<DeletionLogEntry>(&json) {
                        max_sequence = max_sequence.max(entry.sequence);
                    }
                    // Skip newline
                    if reader.read_u8().is_err() {
                        // EOF or error after string
                        break;
                    }
                } else {
                    // Failed to read string (EOF or corruption)
                    break;
                }
            }

            self.sequence
                .store(max_sequence + 1, std::sync::atomic::Ordering::SeqCst);
        }

        Ok(())
    }
}

/// Core deletion manager.
#[derive(Debug)]
pub struct DeletionManager {
    /// Storage backend.
    storage: Arc<dyn Storage>,

    /// Deletion bitmaps per segment.
    bitmaps: RwLock<AHashMap<String, Arc<DeletionBitmap>>>,

    /// Deletion log for recovery.
    deletion_log: Option<DeletionLog>,

    /// Segments whose in-memory bitmap has changed since the last
    /// [`flush`](Self::flush) (Issue #875).
    ///
    /// Mutations ([`delete_document`](Self::delete_document) and
    /// [`initialize_segment`](Self::initialize_segment)) only update the in-memory
    /// bitmap and record the segment here; the `.delmap` files are written
    /// once per group by [`flush`](Self::flush) instead of once per delete.
    /// This removes the per-delete full-bitmap rewrite (+ fsync) from the
    /// upsert hot path — the caller is responsible for flushing at its
    /// durability point (the lexical writer flushes on commit).
    dirty_segments: RwLock<ahash::AHashSet<String>>,
}

impl DeletionManager {
    /// Create a new deletion manager.
    pub fn new(config: DeletionConfig, storage: Arc<dyn Storage>) -> Result<Self> {
        let deletion_log = if config.enable_deletion_log {
            Some(DeletionLog::new(
                storage.clone(),
                "deletions.log".to_string(),
            )?)
        } else {
            None
        };

        let manager = DeletionManager {
            storage,
            bitmaps: RwLock::new(AHashMap::new()),
            deletion_log,
            dirty_segments: RwLock::new(ahash::AHashSet::new()),
        };

        // Load existing bitmaps
        manager.load_bitmaps()?;

        Ok(manager)
    }

    /// Initialize deletion tracking for a segment.
    pub fn initialize_segment(
        &self,
        segment_id: &str,
        min_doc_id: u64,
        max_doc_id: u64,
    ) -> Result<()> {
        let bitmaps = self.bitmaps.read().unwrap();
        if bitmaps.contains_key(segment_id) {
            // If segment already exists, update its min/max if necessary, or just return.
            // For now, we assume it's already correctly initialized.
            // A more robust system might check if min/max changed and update.
            return Ok(());
        }
        drop(bitmaps);

        let bitmap = Arc::new(DeletionBitmap::new(
            segment_id.to_string(),
            min_doc_id,
            max_doc_id,
        ));

        {
            let mut bitmaps = self.bitmaps.write().unwrap();
            bitmaps.insert(segment_id.to_string(), bitmap);
        }

        // The empty bitmap is persisted by the next `flush` together with the
        // deletions that prompted the initialization (Issue #875).
        self.mark_dirty(segment_id);

        Ok(())
    }

    /// Mark a document as deleted.
    ///
    /// The deletion is applied to the in-memory bitmap immediately but is only
    /// persisted to the segment's `.delmap` file by the next
    /// [`flush`](Self::flush) (Issue #875) — callers must flush at their
    /// durability point (the lexical writer flushes on commit; crash recovery
    /// is covered by the engine WAL, which records every delete before the
    /// index mutation).
    pub fn delete_document(&self, segment_id: &str, doc_id: u64, reason: &str) -> Result<bool> {
        let was_deleted = {
            let bitmaps = self.bitmaps.read().unwrap();

            if let Some(bitmap) = bitmaps.get(segment_id) {
                bitmap.delete_document(doc_id)?
            } else {
                return Err(LaurusError::index(format!(
                    "Segment {segment_id} not found in deletion manager"
                )));
            }
        };

        // Log the deletion
        if let Some(ref log) = self.deletion_log {
            log.log_deletion(segment_id, doc_id, reason)?;
        }

        // Defer bitmap persistence to the next `flush` (Issue #875)
        if was_deleted {
            self.mark_dirty(segment_id);
        }

        Ok(was_deleted)
    }

    /// Forget a segment's in-memory deletion state without persisting it.
    ///
    /// For a segment the caller is discarding — the lexical writer's
    /// `rollback()` drops segments an automatic flush wrote but no commit
    /// published (Issue #1204). The bitmap and its pending
    /// [`flush`](Self::flush) are dropped; a `.delmap` already on storage is
    /// left for the caller, which deletes the segment's files itself.
    /// Forgetting an unknown segment is a no-op.
    pub fn forget_segment(&self, segment_id: &str) {
        self.bitmaps.write().unwrap().remove(segment_id);
        self.dirty_segments.write().unwrap().remove(segment_id);
    }

    /// `segment_id`'s deletion bitmap — its persisted `.delmap` plus every
    /// deletion made through this manager since — if it tracks one (Issue
    /// #1212).
    pub fn bitmap(&self, segment_id: &str) -> Option<Arc<DeletionBitmap>> {
        self.bitmaps.read().unwrap().get(segment_id).cloned()
    }

    /// Whether `doc_id` is marked deleted in `segment_id`'s bitmap.
    ///
    /// `false` for a segment this manager tracks no bitmap for — nothing has
    /// been deleted from it. The lexical writer asks this of the segments it
    /// flushed but has not committed, to skip exactly the copies superseded
    /// since the flush (Issue #1207).
    pub fn is_deleted(&self, segment_id: &str, doc_id: u64) -> bool {
        self.bitmaps
            .read()
            .unwrap()
            .get(segment_id)
            .is_some_and(|bitmap| bitmap.is_deleted(doc_id))
    }

    /// Record that a segment's in-memory bitmap diverged from its `.delmap`
    /// file and needs persisting by the next [`flush`](Self::flush).
    fn mark_dirty(&self, segment_id: &str) {
        self.dirty_segments
            .write()
            .unwrap()
            .insert(segment_id.to_string());
    }

    /// Persist every dirty segment's deletion bitmap to its `.delmap` file
    /// (Issue #875).
    ///
    /// This is the single durability point for deletion state: mutations only
    /// update the in-memory bitmaps and mark their segment dirty, and this
    /// method group-commits all of them (one `.delmap` write per dirty
    /// segment instead of one per delete). Idempotent — a second call without
    /// intervening mutations writes nothing.
    ///
    /// # Returns
    ///
    /// The IDs of the segments whose bitmap was written, in unspecified
    /// order (empty when nothing was dirty).
    ///
    /// # Errors
    ///
    /// Returns an error if writing a bitmap fails; segments not yet written
    /// (including the failed one) remain marked dirty so a retry or a later
    /// flush persists them.
    pub fn flush(&self) -> Result<Vec<String>> {
        // Claim the dirty set up front: a mutation racing with this flush
        // re-marks its segment dirty on its own, so draining first can never
        // drop an unpersisted change (removing after the save could).
        let dirty: Vec<String> = self.dirty_segments.write().unwrap().drain().collect();

        let mut flushed = Vec::with_capacity(dirty.len());
        for (i, segment_id) in dirty.iter().enumerate() {
            if let Err(e) = self.save_bitmap(segment_id) {
                // Restore the failed segment and every not-yet-written one so
                // a retry (or the next flush) persists them.
                let mut dirty_guard = self.dirty_segments.write().unwrap();
                for seg in &dirty[i..] {
                    dirty_guard.insert(seg.clone());
                }
                return Err(e);
            }
            flushed.push(segment_id.clone());
        }
        Ok(flushed)
    }

    /// Save bitmap to storage.
    fn save_bitmap(&self, segment_id: &str) -> Result<()> {
        let bitmaps = self.bitmaps.read().unwrap();

        if let Some(bitmap) = bitmaps.get(segment_id) {
            let bitmap_file = format!("{segment_id}.delmap");
            let output = self.storage.create_output(&bitmap_file)?;
            let mut writer = StructWriter::new(output);
            bitmap.write_to_storage(&mut writer)?;
            writer.close()?;
        }

        Ok(())
    }

    /// Load existing bitmaps from storage.
    fn load_bitmaps(&self) -> Result<()> {
        let files = self.storage.list_files()?;

        for file in files {
            if file.ends_with(".delmap") {
                let input = self.storage.open_input(&file)?;
                let mut reader = StructReader::new(input)?;

                if let Ok(bitmap) = DeletionBitmap::read_from_storage(&mut reader) {
                    let mut bitmaps = self.bitmaps.write().unwrap();
                    bitmaps.insert(bitmap.segment_id.clone(), Arc::new(bitmap));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;

    #[test]
    fn test_deletion_bitmap_creation() {
        let bitmap = DeletionBitmap::new("seg001".to_string(), 0, 999);

        assert_eq!(bitmap.segment_id, "seg001");
        assert_eq!(bitmap.total_docs.load(Ordering::SeqCst), 1000);
        assert_eq!(bitmap.deleted_count.load(Ordering::SeqCst), 0);
        assert_eq!(bitmap.deletion_ratio(), 0.0);
        assert_eq!(bitmap.live_count(), 1000);
    }

    #[test]
    fn test_deletion_bitmap_operations() {
        let bitmap = DeletionBitmap::new("seg001".to_string(), 0, 99);

        // Delete some documents
        assert!(bitmap.delete_document(5).unwrap());
        assert!(bitmap.delete_document(10).unwrap());
        assert!(bitmap.delete_document(15).unwrap());

        // Check deletion status
        assert!(bitmap.is_deleted(5));
        assert!(bitmap.is_deleted(10));
        assert!(bitmap.is_deleted(15));
        assert!(!bitmap.is_deleted(20));

        // Check counts
        assert_eq!(bitmap.deleted_count.load(Ordering::SeqCst), 3);
        assert_eq!(bitmap.live_count(), 97);
        assert_eq!(bitmap.deletion_ratio(), 0.03);

        // Try to delete same document again
        assert!(!bitmap.delete_document(5).unwrap());
        assert_eq!(bitmap.deleted_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_deletion_bitmap_out_of_range() {
        let bitmap = DeletionBitmap::new("seg001".to_string(), 0, 99);

        let result = bitmap.delete_document(150);
        assert!(result.is_err());

        assert!(!bitmap.is_deleted(150));
    }

    /// v4 (Roaring) `.delmap` round-trips: write then read yields the same
    /// deleted set, ordering, and metadata (Issue #684).
    #[test]
    fn test_deletion_bitmap_v4_round_trip() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());

        let bitmap = DeletionBitmap::new("seg-v4".to_string(), 0, 999);
        for id in [3u64, 7, 42, 900, 999] {
            bitmap.delete_document(id).unwrap();
        }

        {
            let output = storage.create_output("seg-v4.delmap").unwrap();
            let mut writer = StructWriter::new(output);
            bitmap.write_to_storage(&mut writer).unwrap();
            writer.close().unwrap();
        }

        let loaded = {
            let input = storage.open_input("seg-v4.delmap").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            DeletionBitmap::read_from_storage(&mut reader).unwrap()
        };

        assert_eq!(loaded.segment_id, "seg-v4");
        assert_eq!(loaded.min_doc_id, 0);
        assert_eq!(loaded.max_doc_id, 999);
        assert_eq!(loaded.deleted_count.load(Ordering::SeqCst), 5);
        // `get_deleted_docs` is ascending for a Roaring bitmap.
        assert_eq!(loaded.get_deleted_docs(), vec![3, 7, 42, 900, 999]);
        assert!(loaded.is_deleted(42));
        assert!(!loaded.is_deleted(43));
    }

    /// Legacy v3 (raw-`u64`-list) `.delmap` payloads must still be readable
    /// after the Roaring migration (Issue #684 back-compat).
    #[test]
    fn test_deletion_bitmap_reads_v3_format() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());

        // Hand-write a v3 payload (the format prior to this change).
        {
            let output = storage.create_output("seg-v3.delmap").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_u32(0x44454C42).unwrap(); // magic "DELB"
            writer.write_u32(3).unwrap(); // version 3
            writer.write_string("seg-v3").unwrap();
            writer.write_u64(1000).unwrap(); // total_docs
            writer.write_u64(3).unwrap(); // deleted_count
            writer.write_u64(12345).unwrap(); // last_modified
            writer.write_u64(7).unwrap(); // bitmap version
            writer.write_u64(0).unwrap(); // min_doc_id
            writer.write_u64(999).unwrap(); // max_doc_id
            writer.write_varint(3).unwrap(); // deleted id count
            for id in [11u64, 222, 888] {
                writer.write_u64(id).unwrap();
            }
            writer.close().unwrap();
        }

        let loaded = {
            let input = storage.open_input("seg-v3.delmap").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            DeletionBitmap::read_from_storage(&mut reader).unwrap()
        };

        assert_eq!(loaded.segment_id, "seg-v3");
        assert_eq!(loaded.min_doc_id, 0);
        assert_eq!(loaded.max_doc_id, 999);
        assert_eq!(loaded.get_deleted_docs(), vec![11, 222, 888]);
        assert!(loaded.is_deleted(222));
        assert!(!loaded.is_deleted(223));
    }

    /// Deferred persistence (Issue #875): mutations must not write the
    /// `.delmap` file; `flush` persists every dirty segment and is
    /// idempotent. Rewritten on the live mutation API after #1024 removed
    /// the speculative `DeletionManager` surface.
    #[test]
    fn flush_persists_dirty_bitmaps_deferred() {
        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = DeletionManager::new(DeletionConfig::default(), storage.clone()).unwrap();
        manager.initialize_segment("seg001", 0, 999).unwrap();
        manager.delete_document("seg001", 5, "test").unwrap();

        assert!(
            !storage.file_exists("seg001.delmap"),
            "mutations must not write the .delmap before flush (#875)"
        );
        manager.flush().unwrap();
        assert!(
            storage.file_exists("seg001.delmap"),
            "flush must persist the dirty bitmap"
        );
        // Idempotent: a second flush finds nothing dirty and succeeds.
        manager.flush().unwrap();
    }

    /// `deleted_in_range` counts the set's bits inside the range, not the
    /// header counter (Issue #1211): a v1/v2 `.delmap` infers its range from
    /// its own bits, so a loaded bitmap can hold ids outside the segment's.
    #[test]
    fn deleted_in_range_counts_only_the_range() {
        let bitmap = DeletionBitmap::new("seg".to_string(), 0, 9);
        bitmap.delete_document(3).unwrap();
        bitmap.deleted_docs.write().unwrap().insert(50);

        assert_eq!(bitmap.deleted_in_range(0, 9), 1);
        let held: RoaringTreemap = [3u64, 4].into_iter().collect();
        assert_eq!(bitmap.deleted_count_in(&held), 1);
    }

    /// `forget_segment` drops a segment's unflushed deletions (Issue #1204):
    /// a later flush writes no `.delmap` for it, and deleting from it again
    /// is an error until it is re-initialised — while another segment's
    /// pending deletions still flush.
    #[test]
    fn forget_segment_drops_unflushed_deletions() {
        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = DeletionManager::new(DeletionConfig::default(), storage.clone()).unwrap();
        manager.initialize_segment("seg001", 0, 999).unwrap();
        manager.initialize_segment("seg002", 1000, 1999).unwrap();
        manager.delete_document("seg001", 5, "test").unwrap();
        manager.delete_document("seg002", 1005, "test").unwrap();

        manager.forget_segment("seg001");
        manager.forget_segment("unknown"); // no-op
        manager.flush().unwrap();

        assert!(!storage.file_exists("seg001.delmap"));
        assert!(storage.file_exists("seg002.delmap"));
        assert!(manager.delete_document("seg001", 6, "test").is_err());
    }

    /// `is_deleted` answers per segment (Issue #1207): the same id can be
    /// deleted in one segment and live in another, and an untracked segment
    /// has nothing deleted.
    #[test]
    fn is_deleted_is_scoped_to_the_segment() {
        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = DeletionManager::new(DeletionConfig::default(), storage).unwrap();
        manager.initialize_segment("seg001", 0, 999).unwrap();
        manager.initialize_segment("seg002", 0, 999).unwrap();
        manager.delete_document("seg001", 5, "test").unwrap();

        assert!(manager.is_deleted("seg001", 5));
        assert!(!manager.is_deleted("seg001", 6));
        assert!(!manager.is_deleted("seg002", 5));
        assert!(!manager.is_deleted("untracked", 5));
    }
}
