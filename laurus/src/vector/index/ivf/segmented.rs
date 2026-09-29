//! Segment-per-commit IVF vector index (Issue #889 PR-6, mirroring HNSW's
//! #634/#882 design and Flat's PR-4 — see
//! [`crate::vector::index::hnsw::segmented`] and
//! [`crate::vector::index::flat::segmented`]).
//!
//! [`SegmentedIvfIndex`] implements [`VectorIndex`] over a set of immutable
//! per-segment `.ivf` files registered in an atomic, checksummed
//! `segments.json` manifest, instead of the monolithic single-file layout
//! of [`super::IvfIndex`]: each commit trains centroids and seals only the
//! newly added vectors as a new segment, turning the per-commit cost from a
//! full corpus-wide re-training into training over just the new documents.
//!
//! Each segment trains its own centroids independently with an adaptive
//! cluster count (Issue #889 PR-5's `configured_n_clusters` ceiling) — no
//! cross-segment cluster-id agreement is needed, since search fans out per
//! segment (via the shared [`crate::vector::index::segment::fanout`] layer)
//! and merges top-k results, exactly how HNSW's per-segment graphs need no
//! cross-segment agreement either. A merge does not attempt to reconcile
//! source cluster ids: it flattens the deduplicated survivors and retrains
//! from scratch (see [`crate::vector::index::ivf::segment::merge_engine`]).
//!
//! Search fans out over the sealed segments newest-generation-first with
//! containment-based newest-wins deduplication (mirroring #880): a hit from
//! an older segment is dropped when any newer segment contains the same
//! `(doc_id, field)` — same-id upserts replayed from the WAL leave stale
//! copies behind until a merge physically collapses them. Uncommitted adds
//! are invisible until commit, matching the monolithic semantics.
//!
//! Deletions are logical: a shared index-level `DeletionBitmap` is marked
//! by [`VectorIndex::soft_delete_document`], persisted to `{name}.delmap`
//! by [`VectorIndex::persist_deletions`], filtered by every segment reader,
//! and physically reclaimed by [`VectorIndex::optimize`] (a force-merge).
//! This is a first-time implementation for IVF — the monolithic
//! [`super::IvfIndex`] has no soft-delete/compaction wiring at all.
//!
//! [`IvfIndexConfig::segmented`] defaults to `true` (Issue #907, mirroring
//! HNSW's own #882 flip): this is now the default IVF layout, with the
//! monolithic single-file layout still available via `segmented: false`. A
//! legacy monolithic index is migrated zero-copy on first open with the
//! flag on (its `.ivf` becomes segment 0 of the manifest, no data
//! movement).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::storage::structured::{StructReader, StructWriter};
use crate::vector::core::vector::Vector;
use crate::vector::index::config::IvfIndexConfig;
use crate::vector::index::ivf::reader::IvfIndexReader;
use crate::vector::index::ivf::searcher::IvfSearcher;
use crate::vector::index::ivf::segment::merge_engine::MergeEngine;
use crate::vector::index::ivf::writer::IvfIndexWriter;
use crate::vector::index::segment::fanout::{SegmentFanoutSearcher, SegmentedReaderFacade};
use crate::vector::index::segment::manager::{
    ManagedSegmentInfo, MergeCandidate, SegmentManager, SegmentManagerConfig,
};
use crate::vector::index::segment::reader_cache::SegmentedReaderCache;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::store::embedding_writer::EmbeddingVectorIndexWriter;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// State shared between the index handle, its writers, and its searchers.
#[derive(Debug)]
struct SegmentedShared {
    /// Index name — the prefix for the deletion bitmap file. Segment files
    /// are named by the manager (`segment_NNNNNN.ivf`).
    name: String,

    /// Storage backend.
    storage: Arc<dyn Storage>,

    /// Segment registry (atomic manifest).
    manager: Arc<SegmentManager>,

    /// Per-segment reader cache; invalidated on merge.
    reader_cache: Arc<SegmentedReaderCache<IvfIndexReader>>,

    /// Index-level logical-deletion bitmap (doc-scoped). `None` means "not
    /// yet loaded / no deletions"; lazily loaded from `{name}.delmap`.
    deletion: RwLock<Option<Arc<DeletionBitmap>>>,

    /// Bumped when the bitmap is CREATED (first soft delete). Readers cached
    /// before that moment are not attached to it; the reader-building loop
    /// re-checks this epoch after populating from the cache and rebuilds if
    /// a create raced it — clearing the cache alone is not enough, because
    /// a concurrent build could repopulate it with bitmap-less readers
    /// after the clear.
    bitmap_epoch: std::sync::atomic::AtomicU64,

    /// Highest WAL sequence number applied to this index but NOT yet
    /// published to the manifest. Published (and persisted) by
    /// [`VectorIndex::persist_deletions`] at the end of the store's commit
    /// ladder, once every covered mutation is durable.
    pending_wal_seq: std::sync::atomic::AtomicU64,
}

impl SegmentedShared {
    fn delmap_file_name(&self) -> String {
        format!("{}.delmap", self.name)
    }

    /// Load (or lazily create) the shared deletion bitmap.
    ///
    /// Mirrors `SegmentedFlatIndex`'s bitmap handling. When a bitmap is
    /// created for the first time, the reader cache is cleared so
    /// subsequently loaded segment readers attach it (cached readers hold
    /// the bitmap by `Arc`, so later marks on an *already attached* bitmap
    /// are always visible).
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
            self.reader_cache.clear();
            self.bitmap_epoch.fetch_add(1, Ordering::Release);
            return Ok(Some(bitmap));
        }

        Ok(None)
    }

    /// Sealed segment readers, newest generation first, with the deletion
    /// bitmap attached.
    fn sealed_readers_newest_first(
        &self,
        config: &IvfIndexConfig,
    ) -> Result<Vec<Arc<IvfIndexReader>>> {
        let mut segments = self.manager.list_segments();
        segments.sort_by_key(|s| std::cmp::Reverse(s.generation));

        loop {
            let epoch = self.bitmap_epoch.load(Ordering::Acquire);
            let bitmap = self.load_or_get_bitmap(false)?;
            let mut readers = Vec::with_capacity(segments.len());
            for info in &segments {
                let storage = self.storage.clone();
                let metric = config.distance_metric;
                let bitmap_for_loader = bitmap.clone();
                let segment_id = info.segment_id.clone();
                let reader = self.reader_cache.get_or_load(&segment_id, || {
                    let mut r = IvfIndexReader::load(storage, &segment_id, metric)?;
                    if let Some(b) = bitmap_for_loader {
                        r.set_deletion_bitmap(b);
                    }
                    Ok(r)
                })?;
                readers.push(reader);
            }
            // A bitmap create that raced this build may have been missed by
            // readers loaded from our (pre-create) snapshot; the epoch bump
            // detects it. Creation happens at most once per index lifetime,
            // so the loop reruns at most once.
            if self.bitmap_epoch.load(Ordering::Acquire) == epoch {
                return Ok(readers);
            }
            self.reader_cache.clear();
        }
    }
}

/// Segment-per-commit IVF index (see the module docs).
#[derive(Debug)]
pub struct SegmentedIvfIndex {
    shared: Arc<SegmentedShared>,
    config: IvfIndexConfig,
    closed: AtomicBool,
}

impl SegmentedIvfIndex {
    /// Open an existing segmented index or create a new one.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend.
    /// * `name` - Index name (bitmap file prefix; segment files are named
    ///   by the segment manager).
    /// * `config` - IVF configuration (with [`IvfIndexConfig::segmented`]
    ///   set).
    ///
    /// A legacy monolithic index is migrated **zero-copy**: the existing
    /// `{name}.ivf` is registered verbatim as the first segment — its
    /// segment id is the index name, which every reader treats as a plain
    /// path prefix — with a single atomic manifest write and no data
    /// movement. A crash before the manifest save leaves the legacy layout
    /// intact, so the migration simply re-runs on the next open. The
    /// pre-existing `{name}.delmap` keeps its meaning: it is the
    /// index-level deletion bitmap in both layouts (and
    /// `delete_segment_files` never touches `.delmap` files for exactly
    /// this reason).
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest fails to load (corruption fails
    /// loudly), when reading the legacy file's header during migration
    /// fails, or when the legacy file's persisted dimension does not match
    /// `config.dimension`.
    pub fn open_or_create(
        storage: Arc<dyn Storage>,
        name: &str,
        config: IvfIndexConfig,
    ) -> Result<Self> {
        // The generated-segment namespace is reserved: an index named like
        // a generated id would collide with sealed segments (same file
        // names, orphan-sweep patterns, and merge GC) — reject it loudly.
        if let Some(ordinal) = name.strip_prefix("segment_")
            && !ordinal.is_empty()
            && ordinal.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(LaurusError::invalid_config(format!(
                "index name '{name}' collides with the reserved segment-id \
                 namespace (segment_<digits>)"
            )));
        }

        let legacy_file = format!("{name}.ivf");
        let migrate = storage.file_exists(&legacy_file) && !storage.file_exists("segments.json");

        // The manager's orphan sweep only touches `segment_NNNNNN.*` names,
        // so an unmigrated `{name}.ivf` is never swept.
        let manager = Arc::new(SegmentManager::new(
            SegmentManagerConfig::default(),
            storage.clone(),
            crate::vector::index::ivf::segment::LAYOUT,
        )?);

        if migrate {
            // Read the committed vector count and dimension from the
            // legacy file's leading two u32 fields (same header every
            // reader loads; `n_clusters`/`n_probe` follow but migration
            // needs neither).
            let (vector_count, dimension) = {
                use std::io::Read;
                let mut input = storage.open_input(&legacy_file)?;
                let mut count_buf = [0u8; 4];
                input.read_exact(&mut count_buf)?;
                let mut dim_buf = [0u8; 4];
                input.read_exact(&mut dim_buf)?;
                (
                    u32::from_le_bytes(count_buf) as u64,
                    u32::from_le_bytes(dim_buf) as usize,
                )
            };
            if dimension != config.dimension {
                return Err(LaurusError::index(format!(
                    "Dimension mismatch during migration: stored {dimension}, config {}",
                    config.dimension
                )));
            }
            // `add_segment` stamps generation 1, measures the on-storage
            // size, and saves the manifest atomically — the moment it
            // returns, the index is segmented; before that, it is still a
            // valid legacy index. A legacy delmap (same file in both
            // layouts) means the segment carries deletions.
            let mut info = ManagedSegmentInfo::new(name.to_string(), vector_count, 0, 0);
            info.has_deletions = storage.file_exists(&format!("{name}.delmap"));
            manager.add_segment(info)?;

            // Drop the monolithic index's now-stale `metadata.json` so the
            // factory's open/create routing can never again mistake this
            // directory for a monolithic index (mirrors HNSW's #882
            // review fix).
            let _ = storage.delete_file("metadata.json");
        }

        Ok(Self {
            shared: Arc::new(SegmentedShared {
                name: name.to_string(),
                storage,
                manager,
                reader_cache: Arc::new(SegmentedReaderCache::new()),
                deletion: RwLock::new(None),
                bitmap_epoch: std::sync::atomic::AtomicU64::new(0),
                pending_wal_seq: std::sync::atomic::AtomicU64::new(0),
            }),
            config,
            closed: AtomicBool::new(false),
        })
    }

    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(LaurusError::InvalidOperation("Index is closed".to_string()));
        }
        Ok(())
    }

    /// Merge one policy-selected window of segments.
    ///
    /// Uses the generation-contiguous, size-similar `TieredMergePolicy`
    /// through the deletion-filtering, re-clustering merge engine; source
    /// readers are invalidated afterwards. Returns whether a merge actually
    /// ran.
    fn merge_once(&self) -> Result<bool> {
        use crate::vector::index::segment::merge_policy::TieredMergePolicy;

        let Some(candidate) = self.shared.manager.check_merge(&TieredMergePolicy::new()) else {
            return Ok(false);
        };

        let mut engine = MergeEngine::new(
            self.shared.storage.clone(),
            self.config.clone(),
            VectorIndexWriterConfig::default(),
        );
        if let Some(bitmap) = self.shared.load_or_get_bitmap(false)? {
            engine.set_deletion_bitmap(bitmap);
        }

        let new_segment_id = self.shared.manager.generate_segment_id();
        let result = engine.merge_segments(candidate.segments.clone(), new_segment_id)?;

        let source_ids: Vec<String> = candidate
            .segments
            .iter()
            .map(|s| s.segment_id.clone())
            .collect();
        self.shared
            .manager
            .apply_merge(candidate, result.merged_segment)?;
        for id in &source_ids {
            self.shared.reader_cache.invalidate(id);
        }
        Ok(true)
    }

    /// Drop the deletion state after a compaction physically reclaimed the
    /// deleted documents.
    fn clear_deletions(&self) -> Result<()> {
        *self.shared.deletion.write() = None;
        let file = self.shared.delmap_file_name();
        let _ = self.shared.storage.delete_file(&file);
        Ok(())
    }
}

impl VectorIndex for SegmentedIvfIndex {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.check_closed()?;
        let readers = self.shared.sealed_readers_newest_first(&self.config)?;
        let readers: Vec<Arc<dyn VectorIndexReader>> = readers
            .into_iter()
            .map(|r| r as Arc<dyn VectorIndexReader>)
            .collect();
        let bitmap = self.shared.load_or_get_bitmap(false)?;
        Ok(Arc::new(SegmentedReaderFacade::new(
            readers,
            bitmap,
            self.config.dimension,
            self.config.distance_metric,
        )))
    }

    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
        self.check_closed()?;

        // A fresh, EMPTY active-segment writer per call: constructing it does
        // not touch the existing segments — the whole point of the layout.
        let segment_id = self.shared.manager.generate_segment_id();
        let inner = IvfIndexWriter::with_storage(
            self.config.clone(),
            VectorIndexWriterConfig::default(),
            &segment_id,
            self.shared.storage.clone(),
        )?;
        let writer = SegmentedIvfWriter {
            shared: self.shared.clone(),
            segment_id,
            inner,
            sealed_len: None,
        };
        let embedder = self.embedder();
        Ok(Box::new(EmbeddingVectorIndexWriter::new(
            Box::new(writer),
            embedder,
        )))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        &self.shared.storage
    }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn stats(&self) -> Result<VectorIndexStats> {
        self.check_closed()?;
        // `total` sums per-segment row counts: stale same-id upsert copies
        // awaiting a merge are counted once per segment, so this (and the
        // auto-compaction ratio derived from it) slightly over-counts until
        // the next merge. Exact distinct-key counting would cost O(total
        // ids) per call; the approximation is intentional (mirrors HNSW).
        let total = self.shared.manager.total_vectors();
        let deleted = match self.shared.load_or_get_bitmap(false)? {
            Some(bitmap) => bitmap.deleted_count.load(Ordering::Relaxed),
            None => 0,
        };
        let stats = self.shared.manager.stats();
        Ok(VectorIndexStats {
            vector_count: total.saturating_sub(deleted),
            dimension: self.config.dimension,
            total_size: stats.total_size,
            deleted_count: deleted,
            last_modified: 0,
        })
    }

    fn retain_writer_after_commit(&self) -> bool {
        // Retention exists only to avoid a monolithic writer's full
        // corpus-wide re-training after commit. A segmented writer starts
        // EMPTY — there is nothing to retrain — so the store can drop it
        // and construct a fresh one per commit cycle; each commit then
        // seals exactly one new segment.
        false
    }

    fn optimize(&self) -> Result<()> {
        self.check_closed()?;

        let segments = self.shared.manager.list_segments();
        if segments.is_empty() {
            return Ok(());
        }
        // Force-merge everything into one segment; the merge engine filters
        // through the deletion bitmap (physical reclamation), collapses
        // same-key duplicates newest-generation-first, and re-clusters the
        // survivors from scratch.
        let mut engine = MergeEngine::new(
            self.shared.storage.clone(),
            self.config.clone(),
            VectorIndexWriterConfig::default(),
        );
        if let Some(bitmap) = self.shared.load_or_get_bitmap(false)? {
            engine.set_deletion_bitmap(bitmap);
        }

        let new_segment_id = self.shared.manager.generate_segment_id();
        let result = engine.merge_segments(segments.clone(), new_segment_id)?;

        let candidate = MergeCandidate {
            total_vectors: segments.iter().map(|s| s.vector_count).sum(),
            total_size: segments.iter().map(|s| s.size_bytes).sum(),
            segments,
        };
        let source_ids: Vec<String> = candidate
            .segments
            .iter()
            .map(|s| s.segment_id.clone())
            .collect();
        self.shared
            .manager
            .apply_merge(candidate, result.merged_segment)?;
        for id in &source_ids {
            self.shared.reader_cache.invalidate(id);
        }

        // Every logically deleted doc was physically dropped by the merge.
        self.clear_deletions()?;
        Ok(())
    }

    // `refresh` keeps its trait default (no-op), like the monolithic index:
    // sealed segment files are immutable, segment-set changes are picked up
    // from the shared manager on every `searcher()`/`reader()` call, and
    // merges invalidate their source readers explicitly.

    fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>> {
        self.check_closed()?;
        let readers = self.shared.sealed_readers_newest_first(&self.config)?;
        let readers: Vec<Arc<dyn VectorIndexReader>> = readers
            .into_iter()
            .map(|r| r as Arc<dyn VectorIndexReader>)
            .collect();
        let bitmap = self.shared.load_or_get_bitmap(false)?;
        let n_probe = self.config.n_probe;
        Ok(Box::new(SegmentFanoutSearcher::new(
            readers,
            bitmap,
            move |reader| {
                Ok(Box::new(IvfSearcher::with_n_probe(reader, n_probe)?)
                    as Box<dyn VectorIndexSearcher>)
            },
        )))
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.config.embedder)
    }

    fn last_wal_seq(&self) -> u64 {
        // The PUBLISHED checkpoint: the value loaded from (or last saved
        // to) the manifest — never the pending one, which may not be
        // durable yet. Recovery skips WAL records at or below this seq, so
        // it must only ever reflect state that is already on storage.
        self.shared.manager.last_wal_seq()
    }

    fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
        // Recorded as PENDING only. It is published into the manifest at
        // the end of `persist_deletions` — the last vector step of the
        // store's commit ladder — at which point the sealed segment files
        // AND the deletion bitmap for every record up to `seq` are
        // durable, and the engine has not yet truncated the WAL (a
        // checkpoint must never outrun the durability of the state it
        // covers).
        self.shared
            .pending_wal_seq
            .fetch_max(seq, Ordering::Release);
        Ok(())
    }

    fn supports_soft_delete(&self) -> bool {
        true
    }

    fn soft_delete_document(&self, doc_id: u64) -> Result<()> {
        self.check_closed()?;
        let bitmap = self
            .shared
            .load_or_get_bitmap(true)?
            .ok_or_else(|| LaurusError::internal("deletion bitmap unexpectedly missing"))?;
        let newly_deleted = bitmap.delete_document(doc_id)?;
        if newly_deleted && !self.shared.manager.list_segments().is_empty() {
            // Conservative-all flagging: segment infos carry no doc-id
            // range, and the flag only feeds merge-policy prioritization.
            self.shared.manager.mark_all_has_deletions()?;
        }
        Ok(())
    }

    fn persist_deletions(&self) -> Result<()> {
        let guard = self.shared.deletion.read();
        if let Some(bitmap) = guard.as_ref() {
            let file = self.shared.delmap_file_name();
            if bitmap.deleted_count.load(Ordering::Relaxed) > 0 {
                // Temp-then-rename for crash safety; the payload is CRC-32
                // protected by `StructWriter`.
                let tmp = format!("{file}.tmp");
                let output = self.shared.storage.create_output(&tmp)?;
                let mut writer = StructWriter::new(output);
                bitmap.write_to_storage(&mut writer)?;
                writer.close()?;
                self.shared.storage.rename_file(&tmp, &file)?;
            } else if self.shared.storage.file_exists(&file) {
                // Undelete-to-zero: the upsert dance's re-add can clear
                // every mark, and a previously persisted delmap would then
                // be STALE — on reopen it would mask the committed upsert
                // in every segment. "Nothing deleted" must persist as "no
                // delmap file", not "keep whatever file exists".
                self.shared.storage.delete_file(&file)?;
            }
        }
        drop(guard);

        // Publish the WAL checkpoint. This is the last vector step of the
        // store's commit ladder: the sealed segment files (fsynced +
        // manifest-registered by `writer.commit()`) and the deletion
        // bitmap (written above) are durable for every record up to the
        // pending seq, and `Engine::commit` truncates the WAL only after
        // all stores finish — so a crash on either side of this save is
        // safe: before it, recovery replays idempotently from the old
        // checkpoint; after it, everything skipped is already on storage.
        let pending = self.shared.pending_wal_seq.load(Ordering::Acquire);
        if pending > self.shared.manager.last_wal_seq() {
            self.shared.manager.set_last_wal_seq(pending);
            self.shared.manager.save_state()?;
        }
        Ok(())
    }

    fn maybe_auto_compact(&self) -> Result<bool> {
        // Tiered auto-merge: the policy itself is the trigger — it returns
        // a candidate when a generation-contiguous, size-similar window
        // fills up (steady-state: every `merge_factor` commits the newest
        // tier collapses upward, giving each vector O(log N) rewrites over
        // its lifetime) or when the hard segment-count bound is exceeded.
        // Runs regardless of `auto_compaction`, which gates deletion
        // *reclamation*, not the structural segment bound.
        if self.merge_once()? {
            return Ok(true);
        }

        if !self.config.auto_compaction {
            return Ok(false);
        }
        let deleted = match self.shared.load_or_get_bitmap(false)? {
            Some(bitmap) => bitmap.deleted_count.load(Ordering::Relaxed),
            None => 0,
        };
        if deleted == 0 {
            return Ok(false);
        }
        let total = self.shared.manager.total_vectors();
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

/// Active-segment writer for [`SegmentedIvfIndex`].
///
/// Buffers new vectors in an ordinary [`IvfIndexWriter`] whose path is a
/// fresh `segment_NNNNNN`; [`VectorIndexWriter::commit`] seals it as a new
/// immutable segment file (training centroids from just this segment's own
/// buffer) and registers it in the manifest exactly once. A sealed writer is
/// DONE: committing again with new changes is rejected — re-writing a
/// registered segment would carry its original generation while newer
/// segments may have sealed in between, inverting the newest-wins ordering.
/// The store never does this (`retain_writer_after_commit` is false);
/// direct API users get a loud error instead of silent metadata corruption.
#[derive(Debug)]
struct SegmentedIvfWriter {
    shared: Arc<SegmentedShared>,
    segment_id: String,
    inner: IvfIndexWriter,
    /// Buffer length at the moment this writer's segment was sealed and
    /// registered; `None` until then. Used to make the post-commit
    /// `close()` a no-op (the inner buffer is retained after commit, so a
    /// plain "buffer non-empty" check would re-enter `commit`).
    sealed_len: Option<usize>,
}

impl VectorIndexWriter for SegmentedIvfWriter {
    fn next_vector_id(&self) -> u64 {
        self.inner.next_vector_id()
    }

    fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        self.add_vectors(vectors)
    }

    fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        // Same-id upsert completion: the delete-first step marked the ids
        // so sealed copies stop matching; clear the marks so the NEW
        // copies are not shadowed by their own delete once this segment
        // seals. Revived stale copies are masked by newest-wins containment
        // at search and collapsed at merge.
        if let Some(bitmap) = self.shared.load_or_get_bitmap(false)? {
            for (doc_id, _, _) in &vectors {
                let _ = bitmap.undelete_document(*doc_id)?;
            }
        }
        self.inner.add_vectors(vectors)
    }

    fn finalize(&mut self) -> Result<()> {
        self.inner.finalize()
    }

    fn progress(&self) -> f32 {
        self.inner.progress()
    }

    fn estimated_memory_usage(&self) -> usize {
        self.inner.estimated_memory_usage()
    }

    fn vectors(&self) -> &[(u64, String, Vector)] {
        self.inner.vectors()
    }

    fn write(&self) -> Result<()> {
        if self.inner.vectors().is_empty() {
            // Nothing buffered — sealing would create an empty segment file
            // per commit. Deletion persistence is handled by
            // `persist_deletions` on the index.
            return Ok(());
        }
        // Writes `{segment_id}.ivf`; manifest registration happens in
        // `commit()` (`&mut self`), which is the store's entry point.
        self.inner.write()
    }

    fn has_storage(&self) -> bool {
        self.inner.has_storage()
    }

    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        // Upsert delete-first: remove any buffered copy AND mark the
        // shared bitmap so copies in sealed segments stop matching. A pure
        // delete stays marked; a following `add_vectors` with the same id
        // clears the mark.
        if let Some(bitmap) = self.shared.load_or_get_bitmap(true)? {
            let newly_deleted = bitmap.delete_document(doc_id)?;
            if newly_deleted && !self.shared.manager.list_segments().is_empty() {
                self.shared.manager.mark_all_has_deletions()?;
            }
        }
        let _ = self.inner.delete_document(doc_id);
        Ok(())
    }

    fn has_pending_changes(&self) -> bool {
        match self.sealed_len {
            // Sealed: pending only if the buffer changed since the seal.
            Some(sealed) => self.inner.vectors().len() != sealed,
            // `IvfIndexWriter` doesn't override `has_pending_changes` (the
            // trait default is an unconditional `true`), so checking it
            // here would make this branch meaningless.
            None => !self.inner.vectors().is_empty(),
        }
    }

    fn commit(&mut self) -> Result<()> {
        if self.inner.vectors().is_empty() {
            return Ok(());
        }
        if let Some(sealed) = self.sealed_len {
            if self.inner.vectors().len() == sealed {
                // Post-commit close(): the segment is already sealed and the
                // buffer is unchanged — nothing to do.
                return Ok(());
            }
            // See the struct docs: re-sealing a registered segment would
            // invert the newest-wins generation ordering.
            return Err(LaurusError::InvalidOperation(format!(
                "segment '{}' is already sealed; obtain a fresh writer for further changes",
                self.segment_id
            )));
        }
        self.inner.finalize()?;
        self.inner.write()?;
        let vector_count = self.inner.vectors().len() as u64;
        // generation 0 → the manager stamps max+1; size_bytes 0 → measured
        // from storage by `add_segment`.
        let info = ManagedSegmentInfo::new(self.segment_id.clone(), vector_count, 0, 0);
        self.shared.manager.add_segment(info)?;
        self.sealed_len = Some(self.inner.vectors().len());
        self.shared.reader_cache.invalidate(&self.segment_id);
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        // Rolled-back mutations are discarded, so the pending WAL
        // checkpoint must not keep covering their sequence numbers: roll
        // it back to the published value; their WAL records then stay
        // replayable.
        self.shared
            .pending_wal_seq
            .store(self.shared.manager.last_wal_seq(), Ordering::Release);
        self.inner.rollback()
    }

    fn pending_docs(&self) -> u64 {
        self.inner.pending_docs()
    }

    fn close(&mut self) -> Result<()> {
        if self.has_pending_changes() {
            self.commit()?;
        }
        self.inner.close()
    }

    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    fn build_reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.inner.build_reader()
    }
}

impl Drop for SegmentedIvfWriter {
    fn drop(&mut self) {
        // Backstop: dropping a writer whose buffered mutations were never
        // sealed discards the only in-process copy, while the pending WAL
        // checkpoint may already cover their sequence numbers — publishing
        // it later would hide the loss from recovery. Roll the pending
        // value back to the published checkpoint so those records stay
        // replayable. The store never drops a dirty writer (its
        // commit/optimize/add_field paths retain it on failure), so this
        // fires only on direct API use.
        if self.has_pending_changes() {
            self.shared
                .pending_wal_seq
                .store(self.shared.manager.last_wal_seq(), Ordering::Release);
        }
    }
}
