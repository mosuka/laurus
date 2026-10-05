//! Reader-independent lifecycle state shared by a segmented vector index.
//!
//! [`SegmentedCore`] owns the parts of a segment-per-commit index that do
//! not depend on how a segment is read: the segment manifest, the
//! index-level logical-deletion bitmap (`{name}.delmap`), and the
//! pending/published WAL checkpoint. Flat, HNSW and IVF still carry their
//! own copies of this logic (`flat/segmented.rs` and friends); new segmented
//! indexes build on this type instead of adding another copy.
//!
//! Unlike those copies, this type never attaches the bitmap to segment
//! readers: callers consult [`SegmentedCore::is_deleted`] at lookup time, so
//! there is no reader cache to invalidate when the bitmap is first created.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::storage::structured::{StructReader, StructWriter};
use crate::vector::index::segment::manager::SegmentManager;

/// Deletion and WAL-checkpoint state of one segmented index.
#[derive(Debug)]
pub(crate) struct SegmentedCore {
    /// Index name; the prefix of the deletion bitmap file.
    name: String,

    /// Storage backend shared with the segment manager.
    storage: Arc<dyn Storage>,

    /// Segment registry (atomic manifest).
    manager: Arc<SegmentManager>,

    /// Index-level logical-deletion bitmap. `None` means "not loaded yet, or
    /// no deletions"; it is loaded lazily from `{name}.delmap`.
    deletion: RwLock<Option<Arc<DeletionBitmap>>>,

    /// Highest WAL sequence number applied to this index but not yet
    /// published to the manifest. [`Self::persist_deletions`] publishes it
    /// once every covered mutation is durable.
    pending_wal_seq: AtomicU64,
}

impl SegmentedCore {
    /// Create the state for the index `name` over an opened segment manager.
    pub(crate) fn new(name: &str, storage: Arc<dyn Storage>, manager: Arc<SegmentManager>) -> Self {
        Self {
            name: name.to_string(),
            storage,
            manager,
            deletion: RwLock::new(None),
            pending_wal_seq: AtomicU64::new(0),
        }
    }

    /// The storage backend.
    pub(crate) fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// The segment manager.
    pub(crate) fn manager(&self) -> &Arc<SegmentManager> {
        &self.manager
    }

    fn delmap_file_name(&self) -> String {
        format!("{}.delmap", self.name)
    }

    /// Load the deletion bitmap, creating an empty one when
    /// `create_if_missing` is set and none exists yet.
    fn load_or_get_bitmap(&self, create_if_missing: bool) -> Result<Option<Arc<DeletionBitmap>>> {
        if let Some(bitmap) = self.deletion.read().as_ref() {
            return Ok(Some(bitmap.clone()));
        }

        let mut guard = self.deletion.write();
        if let Some(bitmap) = guard.as_ref() {
            return Ok(Some(bitmap.clone()));
        }

        let file = self.delmap_file_name();
        if self.storage.file_exists(&file) {
            let input = self.storage.open_input(&file)?;
            let mut reader = StructReader::new(input)?;
            let bitmap = Arc::new(DeletionBitmap::read_from_storage(&mut reader)?);
            *guard = Some(bitmap.clone());
            return Ok(Some(bitmap));
        }

        if create_if_missing {
            let bitmap = Arc::new(DeletionBitmap::new(self.name.clone(), 0, u64::MAX - 1));
            *guard = Some(bitmap.clone());
            return Ok(Some(bitmap));
        }

        Ok(None)
    }

    /// The current deletion bitmap, if any document was ever deleted.
    pub(crate) fn bitmap(&self) -> Result<Option<Arc<DeletionBitmap>>> {
        self.load_or_get_bitmap(false)
    }

    /// Whether `doc_id` is logically deleted.
    pub(crate) fn is_deleted(&self, doc_id: u64) -> Result<bool> {
        Ok(self
            .bitmap()?
            .is_some_and(|bitmap| bitmap.is_deleted(doc_id)))
    }

    /// Logically delete `doc_id` in every sealed segment.
    ///
    /// # Returns
    ///
    /// `true` when the document was not already marked deleted.
    pub(crate) fn mark_deleted(&self, doc_id: u64) -> Result<bool> {
        let bitmap = self
            .load_or_get_bitmap(true)?
            .ok_or_else(|| LaurusError::internal("deletion bitmap unexpectedly missing"))?;
        let newly_deleted = bitmap.delete_document(doc_id)?;
        if newly_deleted && !self.manager.list_segments().is_empty() {
            // Segment infos carry no doc-id range, and the flag only feeds
            // merge-policy prioritization, so every segment is flagged.
            self.manager.mark_all_has_deletions()?;
        }
        Ok(newly_deleted)
    }

    /// Clear the deletion marks of re-added documents.
    ///
    /// An upsert first marks the document deleted (hiding its sealed
    /// copies), then re-adds it; the new copy must not be hidden by its own
    /// delete once its segment seals. Stale copies in older segments are
    /// masked by newest-segment-wins lookups and collapsed by merges.
    pub(crate) fn unmark_deleted(&self, doc_ids: impl IntoIterator<Item = u64>) -> Result<()> {
        if let Some(bitmap) = self.bitmap()? {
            for doc_id in doc_ids {
                bitmap.undelete_document(doc_id)?;
            }
        }
        Ok(())
    }

    /// Number of logically deleted documents.
    pub(crate) fn deleted_count(&self) -> Result<u64> {
        Ok(self
            .bitmap()?
            .map_or(0, |bitmap| bitmap.deleted_count.load(Ordering::Relaxed)))
    }

    /// Drop the deletion state after a merge physically removed every
    /// deleted document.
    pub(crate) fn clear_deletions(&self) -> Result<()> {
        *self.deletion.write() = None;
        let file = self.delmap_file_name();
        if self.storage.file_exists(&file) {
            self.storage.delete_file(&file)?;
        }
        Ok(())
    }

    /// Persist the deletion bitmap, then publish the pending WAL checkpoint.
    ///
    /// This is the last vector step of the store's commit sequence: sealed
    /// segments are already durable and registered, so once the bitmap is
    /// written, every record up to the pending sequence number is on
    /// storage and the checkpoint may be published.
    pub(crate) fn persist_deletions(&self) -> Result<()> {
        let guard = self.deletion.read();
        if let Some(bitmap) = guard.as_ref() {
            let file = self.delmap_file_name();
            if bitmap.deleted_count.load(Ordering::Relaxed) > 0 {
                let tmp = format!("{file}.tmp");
                let output = self.storage.create_output(&tmp)?;
                let mut writer = StructWriter::new(output);
                bitmap.write_to_storage(&mut writer)?;
                writer.close()?;
                self.storage.rename_file(&tmp, &file)?;
            } else if self.storage.file_exists(&file) {
                // An upsert can clear every mark; a stale delmap would then
                // hide the committed re-add on reopen.
                self.storage.delete_file(&file)?;
            }
        }
        drop(guard);

        let pending = self.pending_wal_seq.load(Ordering::Acquire);
        if pending > self.manager.last_wal_seq() {
            self.manager.set_last_wal_seq(pending);
            self.manager.save_state()?;
        }
        Ok(())
    }

    /// The published WAL checkpoint. Recovery skips records at or below it,
    /// so it only ever reflects state that is already on storage.
    pub(crate) fn last_wal_seq(&self) -> u64 {
        self.manager.last_wal_seq()
    }

    /// Record `seq` as applied but not yet durable.
    pub(crate) fn set_pending_wal_seq(&self, seq: u64) {
        self.pending_wal_seq.fetch_max(seq, Ordering::Release);
    }

    /// Discard the pending checkpoint after its mutations were rolled back
    /// or dropped, so their WAL records stay replayable.
    pub(crate) fn rollback_pending_wal_seq(&self) {
        self.pending_wal_seq
            .store(self.manager.last_wal_seq(), Ordering::Release);
    }

    /// Deleted documents divided by the documents counted in the manifest
    /// (`0.0` when the manifest is empty).
    ///
    /// The manifest counts stale same-id copies once per segment until a
    /// merge collapses them, so the ratio slightly under-estimates.
    pub(crate) fn deletion_ratio(&self) -> Result<f64> {
        let total = self.manager.total_vectors();
        if total == 0 {
            return Ok(0.0);
        }
        Ok(self.deleted_count()? as f64 / total as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::index::segment::manager::{
        ManagedSegmentInfo, SegmentFileLayout, SegmentManagerConfig,
    };

    const TEST_LAYOUT: SegmentFileLayout = SegmentFileLayout {
        primary: ".test",
        sidecars: &[],
        tmp: ".test.tmp",
    };

    fn open(storage: &Arc<dyn Storage>) -> SegmentedCore {
        let manager = Arc::new(
            SegmentManager::new(
                SegmentManagerConfig::default(),
                storage.clone(),
                TEST_LAYOUT,
            )
            .unwrap(),
        );
        SegmentedCore::new("index", storage.clone(), manager)
    }

    fn memory_storage() -> Arc<dyn Storage> {
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
    }

    #[test]
    fn test_deletions_survive_persist_and_reopen() {
        let storage = memory_storage();
        let core = open(&storage);
        assert!(core.mark_deleted(7).unwrap());
        assert!(!core.mark_deleted(7).unwrap());
        core.persist_deletions().unwrap();

        let reopened = open(&storage);
        assert!(reopened.is_deleted(7).unwrap());
        assert!(!reopened.is_deleted(8).unwrap());
        assert_eq!(reopened.deleted_count().unwrap(), 1);
    }

    #[test]
    fn test_unmarking_every_deletion_removes_the_delmap() {
        let storage = memory_storage();
        let core = open(&storage);
        core.mark_deleted(1).unwrap();
        core.persist_deletions().unwrap();
        assert!(storage.file_exists("index.delmap"));

        core.unmark_deleted([1]).unwrap();
        core.persist_deletions().unwrap();
        assert!(!storage.file_exists("index.delmap"));
        assert!(!open(&storage).is_deleted(1).unwrap());
    }

    #[test]
    fn test_pending_wal_seq_is_published_only_by_persist() {
        let storage = memory_storage();
        let core = open(&storage);
        core.set_pending_wal_seq(5);
        core.set_pending_wal_seq(3);
        assert_eq!(core.last_wal_seq(), 0);

        core.persist_deletions().unwrap();
        assert_eq!(core.last_wal_seq(), 5);
        assert_eq!(open(&storage).last_wal_seq(), 5);
    }

    #[test]
    fn test_rollback_restores_the_published_wal_seq() {
        let storage = memory_storage();
        let core = open(&storage);
        core.set_pending_wal_seq(4);
        core.persist_deletions().unwrap();

        core.set_pending_wal_seq(9);
        core.rollback_pending_wal_seq();
        core.persist_deletions().unwrap();
        assert_eq!(core.last_wal_seq(), 4);
    }

    #[test]
    fn test_clear_deletions_drops_state_and_file() {
        let storage = memory_storage();
        let core = open(&storage);
        core.mark_deleted(2).unwrap();
        core.persist_deletions().unwrap();

        core.clear_deletions().unwrap();
        assert!(!storage.file_exists("index.delmap"));
        assert!(!core.is_deleted(2).unwrap());
        assert_eq!(core.deleted_count().unwrap(), 0);
    }

    #[test]
    fn test_deletion_ratio_counts_against_manifest_total() {
        let storage = memory_storage();
        let core = open(&storage);
        assert_eq!(core.deletion_ratio().unwrap(), 0.0);

        storage
            .create_output("segment_000000.test")
            .unwrap()
            .close()
            .unwrap();
        core.manager()
            .add_segment(ManagedSegmentInfo::new("segment_000000".into(), 4, 0, 0))
            .unwrap();
        core.mark_deleted(1).unwrap();
        assert_eq!(core.deletion_ratio().unwrap(), 0.25);
    }
}
