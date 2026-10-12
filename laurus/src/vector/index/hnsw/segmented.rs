//! Segment-per-commit HNSW vector index (Issue #634, PR-3 / #881).
//!
//! [`SegmentedHnswIndex`] implements [`VectorIndex`] over a set of immutable
//! per-segment `.hnsw` files registered in an atomic, checksummed
//! `segments.json` manifest (#879), instead of the monolithic single-file
//! layout of [`super::HnswIndex`]: each commit seals only the newly added
//! vectors as a new segment, turning the per-commit cost from O(index) into
//! O(new docs) — no full rewrite, no re-quantization of the existing corpus.
//!
//! Search fans out over the sealed segments newest-generation-first with
//! containment-based newest-wins deduplication (#880): a hit from an older
//! segment is dropped when any newer segment contains the same
//! `(doc_id, field)` — same-id upserts replayed from the WAL leave stale
//! copies behind until a merge physically collapses them. Uncommitted adds
//! are invisible until commit, matching the monolithic semantics.
//!
//! Deletions are logical: a shared index-level
//! [`DeletionBitmap`](crate::maintenance::deletion::DeletionBitmap) is marked
//! by [`VectorIndex::soft_delete_document`], persisted to `{name}.delmap` by
//! [`VectorIndex::persist_deletions`], filtered by every segment reader, and
//! physically reclaimed by [`VectorIndex::optimize`] (a force-merge). The
//! bitmap is doc-scoped, which is consistent here because a document's
//! fields always land in the same segment.
//!
//! Since #882 this is the DEFAULT HNSW layout ([`HnswIndexConfig::segmented`]
//! defaults to `true`): a legacy monolithic index is migrated zero-copy on
//! first open (its `.hnsw` becomes segment 0 of the manifest, no data
//! movement), and the manifest's `last_wal_seq` is published with the
//! durability ordering the engine's recovery relies on (data before
//! checkpoint, checkpoint before WAL truncate). Set the flag to `false` to
//! keep the monolithic layout.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::vector::Vector;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::searcher::HnswSearcher;
use crate::vector::index::hnsw::segment::merge_engine::MergeEngine;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::index::segment::fanout::{SegmentFanoutSearcher, SegmentedReaderFacade};
use crate::vector::index::segment::manager::{
    ManagedSegmentInfo, SegmentManager, SegmentManagerConfig,
};
use crate::vector::index::segment::reader_cache::SegmentedReaderCache;
use crate::vector::index::segment::segmented_core::SegmentedCore;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::store::embedding_writer::EmbeddingVectorIndexWriter;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// State shared between the index handle, its writers, and its searchers.
#[derive(Debug)]
struct SegmentedShared {
    /// Segment manifest (#879), index-level deletion bitmap
    /// (`{name}.delmap`) and WAL checkpoint (#882). Segment files are named
    /// by the manager (`segment_NNNNNN.hnsw`).
    core: SegmentedCore,

    /// Per-segment reader cache (#660); invalidated on merge, and rebuilt
    /// once the deletion bitmap is created (#881 review).
    reader_cache: SegmentedReaderCache<HnswIndexReader>,
}

impl SegmentedShared {
    /// Sealed segment readers, newest generation first, with the deletion
    /// bitmap attached (#880 containment/dedup ordering).
    fn sealed_readers_newest_first(
        &self,
        config: &HnswIndexConfig,
    ) -> Result<Vec<Arc<dyn VectorIndexReader>>> {
        let metric = config.distance_metric;
        let readers =
            self.reader_cache
                .sealed_readers_newest_first(&self.core, |segment_id, bitmap| {
                    let mut reader =
                        HnswIndexReader::load(self.core.storage().clone(), segment_id, metric)?;
                    if let Some(bitmap) = bitmap {
                        reader.set_deletion_bitmap(bitmap);
                    }
                    Ok(reader)
                })?;
        Ok(readers
            .into_iter()
            .map(|r| r as Arc<dyn VectorIndexReader>)
            .collect())
    }

    /// Merge `segments` into one new segment and drop their cached readers.
    ///
    /// The merge engine filters through the deletion bitmap (physical
    /// reclamation) and collapses same-key duplicates
    /// newest-generation-first (#880).
    fn merge(&self, config: &HnswIndexConfig, segments: Vec<ManagedSegmentInfo>) -> Result<()> {
        let source_ids =
            self.core
                .merge_segments(segments, |bitmap, segments, new_segment_id| {
                    let mut engine = MergeEngine::new(
                        self.core.storage().clone(),
                        config.clone(),
                        VectorIndexWriterConfig::default(),
                    );
                    if let Some(bitmap) = bitmap {
                        engine.set_deletion_bitmap(bitmap);
                    }
                    Ok(engine
                        .merge_segments(segments, new_segment_id)?
                        .merged_segment)
                })?;
        for id in &source_ids {
            self.reader_cache.invalidate(id);
        }
        Ok(())
    }
}

/// Segment-per-commit HNSW index (see the module docs).
#[derive(Debug)]
pub struct SegmentedHnswIndex {
    shared: Arc<SegmentedShared>,
    config: HnswIndexConfig,
    closed: AtomicBool,
}

impl SegmentedHnswIndex {
    /// Open an existing segmented index or create a new one.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend.
    /// * `name` - Index name (bitmap file prefix; segment files are named by
    ///   the segment manager).
    /// * `config` - HNSW configuration (with [`HnswIndexConfig::segmented`]
    ///   set).
    ///
    /// A legacy monolithic index is migrated **zero-copy** (#882): the
    /// existing `{name}.hnsw` is registered verbatim as the first segment —
    /// its segment id is the index name, which every reader treats as a
    /// plain path prefix — with a single atomic manifest write and no data
    /// movement. A crash before the manifest save leaves the legacy layout
    /// intact, so the migration simply re-runs on the next open. The
    /// pre-existing `{name}.delmap` keeps its meaning: it is the index-level
    /// deletion bitmap in both layouts (and `delete_segment_files` never
    /// touches `.delmap` files for exactly this reason).
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest fails to load (corruption fails
    /// loudly, #879), or when reading the legacy file's header during
    /// migration fails.
    pub fn open_or_create(
        storage: Arc<dyn Storage>,
        name: &str,
        config: HnswIndexConfig,
    ) -> Result<Self> {
        // The generated-segment namespace is reserved: an index named like
        // a generated id would collide with sealed segments (same file
        // names, orphan-sweep patterns, and merge GC) — reject it loudly
        // (#882 review).
        if let Some(ordinal) = name.strip_prefix("segment_")
            && !ordinal.is_empty()
            && ordinal.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(LaurusError::invalid_config(format!(
                "index name '{name}' collides with the reserved segment-id \
                 namespace (segment_<digits>)"
            )));
        }

        let legacy_file = format!("{name}.hnsw");
        let migrate = storage.file_exists(&legacy_file) && !storage.file_exists("segments.json");

        // The manager's orphan sweep only touches `segment_NNNNNN.*` names,
        // so an unmigrated `{name}.hnsw` is never swept.
        let manager = Arc::new(SegmentManager::new(
            SegmentManagerConfig::default(),
            storage.clone(),
            crate::vector::index::hnsw::segment::LAYOUT,
        )?);

        if migrate {
            // Read the committed vector count from the leading u64 of the
            // legacy file (same header every reader loads).
            let vector_count = {
                use std::io::Read;
                let mut input = storage.open_input(&legacy_file)?;
                let mut buf = [0u8; 8];
                input.read_exact(&mut buf)?;
                u64::from_le_bytes(buf)
            };
            // `add_segment` stamps generation 1, measures the on-storage
            // size, and saves the manifest atomically (#879) — the moment it
            // returns, the index is segmented; before that, it is still a
            // valid legacy index. A legacy delmap (same file in both
            // layouts) means the segment carries deletions.
            let mut info = ManagedSegmentInfo::new(name.to_string(), vector_count, 0, 0);
            info.has_deletions = storage.file_exists(&format!("{name}.delmap"));
            manager.add_segment(info)?;

            // Drop the monolithic index's now-stale `metadata.json` so the
            // factory's open/create routing can never again mistake this
            // directory for a monolithic index (#882 review: a stale
            // metadata.json flipped a migrated index back to the monolithic
            // view, silently hiding every post-migration segment).
            let _ = storage.delete_file("metadata.json");
        }

        Ok(Self {
            shared: Arc::new(SegmentedShared {
                core: SegmentedCore::new(name, storage, manager),
                reader_cache: SegmentedReaderCache::new(),
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

    /// Default `ef_search` for the multi-segment fan-out: the schema-level
    /// value when configured, else `ef_construction.max(50) * 2` — the
    /// compensation that kept multi-segment recall stable (#644 / #880).
    fn default_ef(&self) -> usize {
        self.config
            .default_ef_search
            .unwrap_or_else(|| self.config.ef_construction.max(50) * 2)
    }

    /// Merge one policy-selected window of segments (#882).
    ///
    /// Uses the generation-contiguous, size-similar `TieredMergePolicy`
    /// (#883) through the deletion-filtering, duplicate-collapsing merge
    /// engine; source readers are invalidated afterwards. Returns whether a
    /// merge actually ran.
    fn merge_once(&self) -> Result<bool> {
        use crate::vector::index::segment::merge_policy::TieredMergePolicy;

        let Some(candidate) = self
            .shared
            .core
            .manager()
            .check_merge(&TieredMergePolicy::new())
        else {
            return Ok(false);
        };
        self.shared.merge(&self.config, candidate.segments)?;
        Ok(true)
    }
}

impl VectorIndex for SegmentedHnswIndex {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.check_closed()?;
        let readers = self.shared.sealed_readers_newest_first(&self.config)?;
        let bitmap = self.shared.core.bitmap()?;
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
        let segment_id = self.shared.core.manager().generate_segment_id();
        let inner = HnswIndexWriter::with_storage(
            self.config.clone(),
            VectorIndexWriterConfig::default(),
            &segment_id,
            self.shared.core.storage().clone(),
        )?;
        let writer = SegmentedHnswWriter {
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
        self.shared.core.storage()
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
        // ids) per call; the approximation is intentional (#881 review).
        let manager = self.shared.core.manager();
        let total = manager.total_vectors();
        let deleted = self.shared.core.deleted_count()?;
        let stats = manager.stats();
        Ok(VectorIndexStats {
            vector_count: total.saturating_sub(deleted),
            dimension: self.config.dimension,
            total_size: stats.total_size,
            deleted_count: deleted,
            last_modified: 0,
        })
    }

    fn retain_writer_after_commit(&self) -> bool {
        // Retention existed to avoid the monolithic writer's O(index) reload
        // after commit (#864). A segmented writer starts EMPTY — there is
        // nothing to reload — so the store can drop it and construct a fresh
        // one per commit cycle; each commit then seals exactly one new
        // segment.
        false
    }

    fn optimize(&self) -> Result<()> {
        self.check_closed()?;

        let segments = self.shared.core.manager().list_segments();
        if segments.is_empty() {
            return Ok(());
        }
        // Force-merge everything into one segment.
        self.shared.merge(&self.config, segments)?;

        // Every logically deleted doc was physically dropped by the merge.
        self.shared.core.clear_deletions()
    }

    // `refresh` keeps its trait default (no-op), like the monolithic index:
    // sealed segment files are immutable, segment-set changes are picked up
    // from the shared manager on every `searcher()`/`reader()` call, and
    // merges invalidate their source readers explicitly. Clearing the reader
    // cache here would force an O(all segments) reload on every commit —
    // defeating the O(delta) design this index exists for (#660/#881).

    fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>> {
        self.check_closed()?;
        let readers = self.shared.sealed_readers_newest_first(&self.config)?;
        let bitmap = self.shared.core.bitmap()?;
        let default_ef = self.default_ef();
        Ok(Box::new(SegmentFanoutSearcher::new(
            readers,
            bitmap,
            move |reader| {
                Ok(Box::new(HnswSearcher::with_default_ef_search(
                    reader,
                    Some(default_ef),
                )?) as Box<dyn VectorIndexSearcher>)
            },
        )))
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.config.embedder)
    }

    fn last_wal_seq(&self) -> u64 {
        // The PUBLISHED checkpoint: the value loaded from (or last saved to)
        // the manifest — never the pending one, which may not be durable
        // yet. Recovery skips WAL records at or below this seq, so it must
        // only ever reflect state that is already on storage (#882).
        self.shared.core.last_wal_seq()
    }

    fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
        // Recorded as PENDING only. It is published into the manifest at the
        // end of `persist_deletions` — the last vector step of the store's
        // commit ladder — at which point the sealed segment files AND the
        // deletion bitmap for every record up to `seq` are durable, and the
        // engine has not yet truncated the WAL (#882 ordering; the #875
        // lesson: a checkpoint must never outrun the durability of the state
        // it covers).
        self.shared.core.set_pending_wal_seq(seq);
        Ok(())
    }

    fn supports_soft_delete(&self) -> bool {
        true
    }

    fn soft_delete_document(&self, doc_id: u64) -> Result<()> {
        self.check_closed()?;
        self.shared.core.mark_deleted(doc_id)?;
        Ok(())
    }

    fn persist_deletions(&self) -> Result<()> {
        // The last vector step of the store's commit ladder (#882): the
        // sealed segment files are durable and registered, so once the
        // bitmap is written the pending WAL checkpoint is published.
        // `Engine::commit` truncates the WAL only after all stores finish,
        // so a crash on either side of the publish is safe.
        self.shared.core.persist_deletions()
    }

    fn maybe_auto_compact(&self) -> Result<bool> {
        // Tiered auto-merge (#883): the policy itself is the trigger — it
        // returns a candidate when a generation-contiguous, size-similar
        // window fills up (steady-state: every `merge_factor` commits the
        // newest tier collapses upward, giving each vector O(log N)
        // rewrites over its lifetime) or when the hard segment-count bound
        // is exceeded. Runs regardless of `auto_compaction`, which gates
        // deletion *reclamation*, not the structural segment bound.
        if self.merge_once()? {
            return Ok(true);
        }

        if self.config.auto_compaction
            && self
                .shared
                .core
                .needs_compaction(self.config.compaction_threshold)?
        {
            self.optimize()?;
            return Ok(true);
        }
        Ok(false)
    }
}

/// Active-segment writer for [`SegmentedHnswIndex`].
///
/// Buffers new vectors in an ordinary [`HnswIndexWriter`] whose path is a
/// fresh `segment_NNNNNN`; [`VectorIndexWriter::commit`] seals it as a new
/// immutable segment file and registers it in the manifest exactly once.
/// A sealed writer is DONE: committing again with new changes is rejected
/// (#881 review) — re-writing a registered segment would carry its original
/// generation while newer segments may have sealed in between, inverting
/// the newest-wins ordering. The store never does this
/// (`retain_writer_after_commit` is false); direct API users get a loud
/// error instead of silent metadata corruption.
#[derive(Debug)]
struct SegmentedHnswWriter {
    shared: Arc<SegmentedShared>,
    segment_id: String,
    inner: HnswIndexWriter,
    /// Buffer length at the moment this writer's segment was sealed and
    /// registered; `None` until then. Used to make the post-commit
    /// `close()` a no-op (the inner buffer is retained after commit, so a
    /// plain "buffer non-empty" check would re-enter `commit`).
    sealed_len: Option<usize>,
}

impl VectorIndexWriter for SegmentedHnswWriter {
    fn next_vector_id(&self) -> u64 {
        self.inner.next_vector_id()
    }

    fn build(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        self.add_vectors(vectors)
    }

    fn add_vectors(&mut self, vectors: Vec<(u64, String, Vector)>) -> Result<()> {
        // Same-id upsert completion (#880): the delete-first step marked the
        // ids so sealed copies stop matching; clear the marks so the NEW
        // copies are not shadowed by their own delete once this segment
        // seals. Revived stale copies are masked by newest-wins containment
        // at search and collapsed at merge.
        self.shared
            .core
            .unmark_deleted(vectors.iter().map(|(doc_id, _, _)| *doc_id))?;
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
        // Writes `{segment_id}.hnsw`; manifest registration happens in
        // `commit()` (`&mut self`), which is the store's entry point.
        self.inner.write()
    }

    fn has_storage(&self) -> bool {
        self.inner.has_storage()
    }

    fn delete_document(&mut self, doc_id: u64) -> Result<()> {
        // Upsert delete-first (#880): remove any buffered copy AND mark the
        // shared bitmap so copies in sealed segments stop matching. A pure
        // delete stays marked; a following `add_vectors` with the same id
        // clears the mark.
        self.shared.core.mark_deleted(doc_id)?;
        let _ = self.inner.delete_document(doc_id);
        Ok(())
    }

    fn has_pending_changes(&self) -> bool {
        match self.sealed_len {
            // Sealed: pending only if the buffer changed since the seal.
            Some(sealed) => self.inner.vectors().len() != sealed,
            None => !self.inner.vectors().is_empty() || self.inner.has_pending_changes(),
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
        // generation 0 → the manager stamps max+1 (#879); size_bytes 0 →
        // measured from storage by `add_segment`.
        let info = ManagedSegmentInfo::new(self.segment_id.clone(), vector_count, 0, 0);
        self.shared.core.manager().add_segment(info)?;
        self.sealed_len = Some(self.inner.vectors().len());
        self.shared.reader_cache.invalidate(&self.segment_id);
        Ok(())
    }

    fn rollback(&mut self) -> Result<()> {
        // Rolled-back mutations are discarded, so the pending WAL
        // checkpoint must not keep covering their sequence numbers (#882):
        // roll it back to the published value; their WAL records then stay
        // replayable.
        self.shared.core.rollback_pending_wal_seq();
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

impl Drop for SegmentedHnswWriter {
    fn drop(&mut self) {
        // Backstop (#882 review): dropping a writer whose buffered
        // mutations were never sealed discards the only in-process copy,
        // while the pending WAL checkpoint may already cover their
        // sequence numbers — publishing it later would hide the loss from
        // recovery. Roll the pending value back to the published
        // checkpoint so those records stay replayable. The store never
        // drops a dirty writer (its commit/optimize/add_field paths retain
        // it on failure), so this fires only on direct API use.
        if self.has_pending_changes() {
            self.shared.core.rollback_pending_wal_seq();
        }
    }
}

// The multi-segment fan-out searcher and read facade (#880/#883 semantics)
// are shared across index types since Issue #889 — see
// `crate::vector::index::segment::fanout`.
