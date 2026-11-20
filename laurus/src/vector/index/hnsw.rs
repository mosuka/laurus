//! HNSW vector index implementation.

pub mod graph;
pub mod reader;
pub mod searcher;
pub mod segment;
pub mod segmented;
pub mod writer;

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::storage::manifest as manifest_io;
use crate::storage::structured::{StructReader, StructWriter};
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::searcher::HnswSearcher;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::store::embedding_writer::EmbeddingVectorIndexWriter;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// Magic marker for the CRC-framed `metadata.json` layout (Issue #786).
///
/// A new `metadata.json` is written through [`StructWriter`] as
/// `[magic u32][json bytes][crc-32 trailer]`. The magic lets the reader tell a
/// checksummed file from a legacy raw-JSON one (which starts with `{`), so old
/// indexes keep loading without migration.
const METADATA_MAGIC: u32 = 0x4C4D_4431; // "LMD1"

/// Magic marker for the CRC-32 footer appended to a `.hnsw` segment (Issue #786).
///
/// New segments end with `[magic u32][crc-32 u32]` over all preceding bytes.
/// The footer is detected by file size (legacy segments have none), so old
/// `.hnsw` files still load.
pub(crate) const HNSW_FOOTER_MAGIC: u32 = 0x4C56_4331; // "LVC1"

/// Byte length of the `.hnsw` CRC footer (`magic u32` + `crc u32`).
pub(crate) const HNSW_FOOTER_LEN: u64 = 8;

/// Metadata for the HNSW index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct IndexMetadata {
    /// Number of vectors in the index.
    vector_count: u64,
    /// Vector dimension.
    dimension: usize,
    /// Creation timestamp.
    created: u64,
    /// Last modification timestamp.
    modified: u64,
}

impl Default for IndexMetadata {
    fn default() -> Self {
        let now = crate::util::time::now_secs();
        Self {
            vector_count: 0,
            dimension: 0,
            created: now,
            modified: now,
        }
    }
}

/// A concrete HNSW vector index implementation.
pub struct HnswIndex {
    /// The name of the index.
    name: String,

    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// HNSW index specific configuration.
    config: HnswIndexConfig,

    /// Whether the index is closed (thread-safe).
    closed: AtomicBool,

    /// Index metadata (thread-safe).
    metadata: RwLock<IndexMetadata>,

    /// Logical deletion bitmap for this single-segment index (Issue #624).
    ///
    /// Lazily loaded from `<name>.delmap` (or created on the first soft
    /// delete). When present and non-empty it is attached to readers so the
    /// deletion-aware HNSW traversal (Issue #665) excludes deleted documents
    /// without rebuilding the graph; [`Self::optimize`] physically reclaims
    /// them. `None` means "not yet loaded / no deletions".
    deletion: RwLock<Option<Arc<DeletionBitmap>>>,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("name", &self.name)
            .field("storage", &self.storage)
            .field("config", &self.config)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .field("metadata", &*self.metadata.read())
            .finish()
    }
}

impl HnswIndex {
    /// Create a new HNSW index in the given storage.
    pub fn create(storage: Arc<dyn Storage>, name: &str, config: HnswIndexConfig) -> Result<Self> {
        let metadata = IndexMetadata {
            dimension: config.dimension,
            ..Default::default()
        };

        let index = HnswIndex {
            name: name.to_string(),
            storage,
            config,
            closed: AtomicBool::new(false),
            metadata: RwLock::new(metadata),
            deletion: RwLock::new(None),
        };

        index.write_metadata()?;
        Ok(index)
    }

    /// Open an existing HNSW index from storage.
    pub fn open(storage: Arc<dyn Storage>, name: &str, config: HnswIndexConfig) -> Result<Self> {
        if !storage.file_exists("metadata.json") {
            return Err(LaurusError::index("Index does not exist"));
        }

        let metadata = Self::read_metadata(storage.as_ref(), name)?;

        // Validate dimension consistency between stored metadata and config.
        if metadata.dimension != 0 && metadata.dimension != config.dimension {
            return Err(LaurusError::index(format!(
                "Dimension mismatch: stored {}, config {}",
                metadata.dimension, config.dimension
            )));
        }

        Ok(HnswIndex {
            name: name.to_string(),
            storage,
            config,
            closed: AtomicBool::new(false),
            metadata: RwLock::new(metadata),
            deletion: RwLock::new(None),
        })
    }

    /// Create an index in a directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create_in_dir<P: AsRef<Path>>(
        dir: P,
        name: &str,
        config: HnswIndexConfig,
    ) -> Result<Self> {
        use crate::storage::file::{FileStorage, FileStorageConfig};

        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::create(storage, name, config)
    }

    /// Open an index from a directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_dir<P: AsRef<Path>>(dir: P, name: &str, config: HnswIndexConfig) -> Result<Self> {
        use crate::storage::file::{FileStorage, FileStorageConfig};

        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::open(storage, name, config)
    }

    /// Write metadata to storage.
    fn write_metadata(&self) -> Result<()> {
        let metadata = self.metadata.read();
        let metadata_json = serde_json::to_string_pretty(&*metadata)
            .map_err(|e| LaurusError::index(format!("Failed to serialize metadata: {e}")))?;
        drop(metadata);

        // temp + CRC-32 trailer + rename + sync (#784/#786, shared since
        // #1022): a crash mid-write leaves the previous metadata intact, and
        // a corrupted file is rejected on load. The shared helper also syncs
        // after the rename, which this call site used to omit — without it a
        // crash can lose the rename itself.
        manifest_io::save_checksummed(
            self.storage.as_ref(),
            "metadata.json",
            Some(METADATA_MAGIC),
            metadata_json.as_bytes(),
        )
    }

    /// Read metadata from storage.
    ///
    /// Reads the CRC-framed layout (Issue #786) and verifies the checksum,
    /// falling back to the legacy raw-JSON format for pre-existing files so old
    /// indexes keep loading.
    fn read_metadata(storage: &dyn Storage, _: &str) -> Result<IndexMetadata> {
        let loaded = manifest_io::load_checksummed_json::<IndexMetadata>(
            storage,
            "metadata.json",
            Some(METADATA_MAGIC),
        )?;
        match loaded {
            Some((metadata, _format)) => Ok(metadata),
            None => Err(LaurusError::index("metadata.json is missing or empty")),
        }
    }

    /// Update metadata.
    fn update_metadata(&self) -> Result<()> {
        {
            let mut metadata = self.metadata.write();
            metadata.modified = crate::util::time::now_secs();
        }
        self.write_metadata()
    }

    /// Check if the index is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(LaurusError::InvalidOperation("Index is closed".to_string()));
        }
        Ok(())
    }

    /// File name holding this index's deletion bitmap (`<name>.delmap`).
    fn delmap_file_name(&self) -> String {
        format!("{}.delmap", self.name)
    }

    /// Resolve this index's deletion bitmap (Issue #624).
    ///
    /// Returns the cached bitmap if present; otherwise loads it from
    /// `<name>.delmap` when that file exists. When `create_if_missing` is
    /// `true` and no bitmap exists yet, a fresh empty bitmap is created and
    /// cached. The returned `Arc` shares interior mutability with the cached
    /// instance, so callers see subsequent marks.
    ///
    /// The bitmap is created with an unbounded id range (`[0, u64::MAX - 1]`)
    /// because the single-segment index has no fixed doc-id span; the range
    /// check is therefore a no-op and `total_docs` is not used (compaction is
    /// driven explicitly by [`Self::optimize`], not by deletion ratio).
    ///
    /// # Arguments
    ///
    /// * `create_if_missing` - Create and cache an empty bitmap when none exists.
    ///
    /// # Errors
    ///
    /// Returns an error if reading an existing `.delmap` file fails.
    fn load_or_get_bitmap(&self, create_if_missing: bool) -> Result<Option<Arc<DeletionBitmap>>> {
        if let Some(bitmap) = self.deletion.read().as_ref() {
            return Ok(Some(bitmap.clone()));
        }

        let mut guard = self.deletion.write();
        self.load_into(&mut guard, create_if_missing)?;
        Ok(guard.clone())
    }

    /// Fill an empty `slot`, held under the `deletion` write lock, from
    /// `<name>.delmap` when the file exists, or with a new empty bitmap when
    /// `create_if_missing` is set. A `slot` another thread has filled is
    /// left as it is.
    ///
    /// # Errors
    ///
    /// Returns an error if reading an existing `.delmap` file fails.
    fn load_into(
        &self,
        slot: &mut Option<Arc<DeletionBitmap>>,
        create_if_missing: bool,
    ) -> Result<()> {
        if slot.is_some() {
            return Ok(());
        }
        let file = self.delmap_file_name();
        if self.storage.file_exists(&file) {
            let input = self.storage.open_input(&file)?;
            let mut reader = StructReader::new(input)?;
            *slot = Some(Arc::new(DeletionBitmap::read_from_storage(&mut reader)?));
        } else if create_if_missing {
            *slot = Some(Arc::new(DeletionBitmap::new(
                self.name.clone(),
                0,
                u64::MAX - 1,
            )));
        }
        Ok(())
    }

    /// Apply `change` to the deletion bitmap while holding the `deletion`
    /// lock, loading the bitmap first, or creating it when
    /// `create_if_missing` is set (Issue #1526, as Issue #1518 does for the
    /// segmented indexes).
    ///
    /// [`Self::forget_deletions`] replaces the bitmap under the write lock.
    /// A change made through a clone of the old `Arc` after the lock was
    /// released would land on the replaced bitmap and be lost.
    ///
    /// # Returns
    ///
    /// `None`, without calling `change`, when there is no bitmap and
    /// `create_if_missing` is not set.
    ///
    /// # Errors
    ///
    /// Returns an error if loading the bitmap or `change` fails.
    fn change_bitmap<T>(
        &self,
        create_if_missing: bool,
        change: impl FnOnce(&DeletionBitmap) -> Result<T>,
    ) -> Result<Option<T>> {
        {
            let guard = self.deletion.read();
            if let Some(bitmap) = guard.as_ref() {
                return change(bitmap).map(Some);
            }
        }

        let mut guard = self.deletion.write();
        self.load_into(&mut guard, create_if_missing)?;
        guard.as_deref().map(change).transpose()
    }

    /// Number of vectors in the committed graph, including logically deleted
    /// ones (Issue #782).
    ///
    /// Read cheaply from the leading `u64` of `<name>.hnsw` (the same vector
    /// count the reader loads, `hnsw/reader.rs`) without materializing the
    /// graph. Returns `0` when no segment has been written yet. Deleted nodes
    /// remain in the graph until compaction, so this is the correct denominator
    /// for the deletion ratio (the `DeletionBitmap`'s synthetic `total_docs` is
    /// not).
    ///
    /// # Errors
    ///
    /// Returns an error if the segment file exists but cannot be read.
    fn committed_node_count(&self) -> Result<u64> {
        use std::io::Read;

        let file = format!("{}.hnsw", self.name);
        if !self.storage.file_exists(&file) {
            return Ok(0);
        }
        let mut input = self.storage.open_input(&file)?;
        let mut buf = [0u8; 8];
        input.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Forget the deletions an `optimize` rebuild physically removed (Issue
    /// #1526, as Issue #1518 does for the segmented indexes).
    ///
    /// `reclaimed` is the copy of the bitmap whose documents the rebuild
    /// dropped. Under the `deletion` write lock, the bitmap is replaced by
    /// one with only the marks made since, or by `None` when there are none,
    /// and `<name>.delmap` is rewritten or removed to match before the lock
    /// is released, so neither a reopen nor a lazy load brings the reclaimed
    /// marks back. A document deleted while the rebuild ran is still in the
    /// new graph, and stays marked.
    ///
    /// The bitmap is replaced rather than edited: a reader built before the
    /// rebuild holds it together with the old graph, which still holds the
    /// reclaimed documents.
    ///
    /// # Errors
    ///
    /// Returns an error if writing or removing `<name>.delmap` fails; the
    /// bitmap is then left as it was.
    fn forget_deletions(&self, reclaimed: &DeletionBitmap) -> Result<()> {
        let mut guard = self.deletion.write();
        let remaining = match guard.as_ref() {
            Some(bitmap) => bitmap.difference(reclaimed),
            None => return Ok(()),
        };
        self.write_delmap(&remaining)?;
        *guard = (remaining.deleted_count.load(Ordering::Relaxed) > 0).then(|| Arc::new(remaining));
        Ok(())
    }

    /// Write `bitmap` to `<name>.delmap`, or remove the file when `bitmap`
    /// marks nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if writing, renaming or removing the file fails.
    fn write_delmap(&self, bitmap: &DeletionBitmap) -> Result<()> {
        let file = self.delmap_file_name();
        if bitmap.deleted_count.load(Ordering::Relaxed) > 0 {
            // Temp-then-rename for crash safety (Issue #784).
            let tmp = format!("{file}.tmp");
            let output = self.storage.create_output(&tmp)?;
            let mut writer = StructWriter::new(output);
            bitmap.write_to_storage(&mut writer)?;
            writer.close()?;
            self.storage.rename_file(&tmp, &file)?;
        } else if self.storage.file_exists(&file) {
            self.storage.delete_file(&file)?;
        }
        Ok(())
    }
}

impl VectorIndex for HnswIndex {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.check_closed()?;

        use crate::vector::index::hnsw::reader::HnswIndexReader;

        let mut reader = HnswIndexReader::load(
            self.storage.clone(),
            &self.name,
            self.config.distance_metric,
        )?;
        // Attach the logical-deletion bitmap so the deletion-aware traversal
        // (Issue #665) excludes soft-deleted documents (Issue #624).
        if let Some(bitmap) = self.load_or_get_bitmap(false)?
            && bitmap.deleted_count.load(Ordering::Relaxed) > 0
        {
            reader.set_deletion_bitmap(bitmap);
        }
        Ok(Arc::new(reader))
    }

    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
        self.check_closed()?;

        let inner_writer = HnswIndexWriter::with_storage(
            self.config.clone(),
            VectorIndexWriterConfig::default(),
            self.name.clone(),
            self.storage.clone(),
        )?;

        // Wrap with EmbeddingVectorIndexWriter for automatic text/image embedding
        let embedder = self.embedder();
        let writer = EmbeddingVectorIndexWriter::new(Box::new(inner_writer), embedder);
        Ok(Box::new(writer))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// Mark the index as closed.
    ///
    /// Callers must call `commit()` before `close()` to persist pending data.
    /// This method only sets the closed flag and releases resources.
    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn stats(&self) -> Result<VectorIndexStats> {
        self.check_closed()?;

        let metadata = self.metadata.read();
        Ok(VectorIndexStats {
            vector_count: metadata.vector_count,
            dimension: metadata.dimension,
            total_size: 0,
            deleted_count: 0,
            last_modified: metadata.modified,
        })
    }

    fn retain_writer_after_commit(&self) -> bool {
        // Audited for Issue #864: `finalize()` is idempotent and appends
        // incrementally to the existing graph, `write(&self)` is
        // non-consuming, and `delete_document` only invalidates the graph
        // when a buffered vector was actually removed — so a retained
        // writer's state stays equivalent to the file it just wrote. The
        // bypassing mutations (`optimize` / auto-compaction) are handled by
        // `VectorStore`, which invalidates its writer cache on both.
        true
    }

    fn optimize(&self) -> Result<()> {
        self.check_closed()?;

        // Physically reclaim soft-deleted documents (Issue #624). Reuse the
        // writer's existing full-rebuild path: load the current graph +
        // vectors, drop the deleted ids (this removes them from the buffer and
        // invalidates the graph), then commit to rebuild a clean graph without
        // them. The expensive rebuild happens here (intentional compaction),
        // not on every delete.
        //
        // A soft delete takes no writer lock, so it can mark a document while
        // the rebuild runs (Issue #1526). The rebuild drops the documents of
        // a frozen copy of the bitmap, and only the copy's marks are
        // forgotten afterwards: a document deleted meanwhile stays in the
        // new graph, and stays marked.
        let reclaimed = match self.load_or_get_bitmap(false)? {
            Some(bitmap) if bitmap.deleted_count.load(Ordering::Relaxed) > 0 => bitmap.snapshot(),
            _ => {
                self.update_metadata()?;
                return Ok(());
            }
        };

        let mut writer = HnswIndexWriter::with_storage(
            self.config.clone(),
            VectorIndexWriterConfig::default(),
            self.name.clone(),
            self.storage.clone(),
        )?;
        for doc_id in reclaimed.get_deleted_docs() {
            writer.delete_document(doc_id)?;
        }
        writer.commit()?;

        self.forget_deletions(&reclaimed)?;
        self.update_metadata()?;
        Ok(())
    }

    fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>> {
        self.check_closed()?;
        let reader = self.reader()?;
        // Pull the schema-level default `ef_search` from the config so the
        // monolithic search path honours `HnswOption.default_ef_search`
        // (Issue #644). Per-query overrides via
        // `VectorIndexQueryParams.ef_search` still take precedence.
        Ok(Box::new(HnswSearcher::with_default_ef_search(
            reader,
            self.config.default_ef_search,
        )?))
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.config.embedder)
    }

    fn supports_soft_delete(&self) -> bool {
        true
    }

    /// Mark `doc_id` deleted, holding the `deletion` lock across the mark so
    /// that an `optimize` replacing the bitmap meanwhile cannot drop it
    /// (Issue #1526).
    fn soft_delete_document(&self, doc_id: u64) -> Result<()> {
        self.check_closed()?;
        self.change_bitmap(true, |bitmap| bitmap.delete_document(doc_id))?
            .ok_or_else(|| LaurusError::internal("deletion bitmap unexpectedly missing"))?;
        Ok(())
    }

    fn persist_deletions(&self) -> Result<()> {
        let guard = self.deletion.read();
        if let Some(bitmap) = guard.as_ref()
            && bitmap.deleted_count.load(Ordering::Relaxed) > 0
        {
            let file = self.delmap_file_name();
            // Temp-then-rename for crash safety (Issue #784); the payload is
            // already CRC-32 protected by `StructWriter` (Issue #684).
            let tmp = format!("{file}.tmp");
            let output = self.storage.create_output(&tmp)?;
            let mut writer = StructWriter::new(output);
            bitmap.write_to_storage(&mut writer)?;
            writer.close()?;
            self.storage.rename_file(&tmp, &file)?;
        }
        Ok(())
    }

    fn write_deletions(&self) -> Result<()> {
        // This index has no WAL checkpoint to hold back.
        self.persist_deletions()
    }

    fn maybe_auto_compact(&self) -> Result<bool> {
        if !self.config.auto_compaction {
            return Ok(false);
        }
        let deleted = match self.load_or_get_bitmap(false)? {
            Some(bitmap) => bitmap.deleted_count.load(Ordering::Relaxed),
            None => 0,
        };
        if deleted == 0 {
            return Ok(false);
        }
        // Denominator is the committed graph size (deleted nodes still live in
        // the graph until compaction), not the bitmap's synthetic `total_docs`.
        let total = self.committed_node_count()?;
        if total == 0 {
            return Ok(false);
        }
        let ratio = deleted as f64 / total as f64;
        if ratio >= self.config.compaction_threshold {
            self.optimize()?;
            return Ok(true);
        }
        Ok(false)
    }
}
#[cfg(test)]
mod tests;
