//! Generic segment-per-commit vector index and writer, shared by Flat, HNSW
//! and IVF (Issue #1409).
//!
//! [`SegmentedIndex<B>`] and its writer implement [`VectorIndex`] /
//! [`VectorIndexWriter`] once; a [`SegmentedBackend`] implementation supplies
//! only what differs between backends — concrete reader/writer/config types,
//! the on-disk [`SegmentFileLayout`], per-segment searcher construction, the
//! merge engine, and two narrow behavioral exceptions (the legacy-migration
//! header format and whether the inner writer's own pending-changes flag
//! matters before a segment is sealed).
//!
//! [`SegmentedCore`] (deletion bitmap, WAL checkpoint, merge publish) and
//! [`SegmentedReaderCache`] are unchanged by this generic layer — they are
//! also used by the MultiVector index (out of scope for #1409), so their
//! public signatures stay as they are.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::embedding::embedder::Embedder;
use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::vector::DistanceMetric;
use crate::vector::index::segment::fanout::{SegmentFanoutSearcher, SegmentedReaderFacade};
use crate::vector::index::segment::field_record::{FieldRecord, check_field_record};
use crate::vector::index::segment::manager::{
    ManagedSegmentInfo, SegmentFileLayout, SegmentManager, SegmentManagerConfig,
};
use crate::vector::index::segment::merge_policy::TieredMergePolicy;
use crate::vector::index::segment::reader_cache::SegmentedReaderCache;
use crate::vector::index::segment::segmented_core::SegmentedCore;
use crate::vector::index::{VectorIndex, VectorIndexStats};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::store::embedding_writer::EmbeddingVectorIndexWriter;
use crate::vector::writer::VectorIndexWriter;

/// The parts of a segment-per-commit vector index that differ between Flat,
/// HNSW and IVF.
///
/// Every method is an associated function (no `self`): a backend is a
/// zero-sized witness type (e.g. `FlatSegmentedBackend`) used only to select
/// an `impl`, never instantiated.
pub trait SegmentedBackend: fmt::Debug + Send + Sync + 'static {
    /// Backend-specific index configuration (`FlatIndexConfig` and so on).
    type Config: Clone + fmt::Debug + Send + Sync + 'static;

    /// Backend-specific segment reader.
    type Reader: VectorIndexReader + 'static;

    /// Backend-specific active-segment writer.
    type Writer: VectorIndexWriter + 'static;

    /// On-disk file suffixes for this backend's segments.
    const LAYOUT: SegmentFileLayout;

    /// Vector dimension, from `config`.
    fn dimension(config: &Self::Config) -> usize;

    /// Distance metric, from `config`.
    fn distance_metric(config: &Self::Config) -> DistanceMetric;

    /// Embedder, from `config`.
    fn embedder(config: &Self::Config) -> Arc<dyn Embedder>;

    /// Whether automatic deletion-reclamation compaction is enabled.
    fn auto_compaction(config: &Self::Config) -> bool;

    /// The deletion ratio at which compaction triggers.
    fn compaction_threshold(config: &Self::Config) -> f64;

    /// Load one sealed segment's reader, with the deletion bitmap attached
    /// when `bitmap` is `Some`.
    fn load_segment_reader(
        config: &Self::Config,
        storage: &Arc<dyn Storage>,
        segment_id: &str,
        bitmap: Option<Arc<DeletionBitmap>>,
    ) -> Result<Self::Reader>;

    /// Build a fresh, empty active-segment writer at `segment_id`.
    fn new_segment_writer(
        config: &Self::Config,
        storage: &Arc<dyn Storage>,
        segment_id: &str,
    ) -> Result<Self::Writer>;

    /// Build the per-segment searcher used by the multi-segment fan-out.
    fn segment_searcher(
        config: &Self::Config,
        reader: Arc<dyn VectorIndexReader>,
    ) -> Result<Box<dyn VectorIndexSearcher>>;

    /// Merge `segments` into one new segment named `new_segment_id`, filtered
    /// through `bitmap` when `Some`, and return its info.
    fn merge_segments(
        config: &Self::Config,
        storage: &Arc<dyn Storage>,
        bitmap: Option<Arc<DeletionBitmap>>,
        segments: Vec<ManagedSegmentInfo>,
        new_segment_id: String,
    ) -> Result<ManagedSegmentInfo>;

    /// Read the vector count and the dimension from the header of the
    /// primary file `file`, which a sealed segment and a legacy monolithic
    /// index share.
    ///
    /// The default matches Flat's and IVF's on-disk header: a `u32` count
    /// followed by a `u32` dimension.
    fn read_segment_header(storage: &Arc<dyn Storage>, file: &str) -> Result<(u64, usize)> {
        use std::io::Read;

        let mut input = storage.open_input(file)?;
        let mut count_buf = [0u8; 4];
        input.read_exact(&mut count_buf)?;
        let mut dim_buf = [0u8; 4];
        input.read_exact(&mut dim_buf)?;
        Ok((
            u32::from_le_bytes(count_buf) as u64,
            u32::from_le_bytes(dim_buf) as usize,
        ))
    }

    /// Read the committed vector count from a legacy monolithic index's
    /// header ([`Self::read_segment_header`]), validating its dimension
    /// against `config`.
    fn read_legacy_vector_count(
        config: &Self::Config,
        storage: &Arc<dyn Storage>,
        legacy_file: &str,
    ) -> Result<u64> {
        let (vector_count, dimension) = Self::read_segment_header(storage, legacy_file)?;

        let expected = Self::dimension(config);
        if dimension != expected {
            return Err(LaurusError::index(format!(
                "Dimension mismatch during migration: stored {dimension}, config {expected}"
            )));
        }
        Ok(vector_count)
    }

    /// Whether an unsealed writer's own `has_pending_changes` should be
    /// consulted in addition to its buffer being non-empty.
    ///
    /// The default is `false`: a backend's inner writer that never overrides
    /// `VectorIndexWriter::has_pending_changes` (the trait default is
    /// unconditionally `true`) must not be asked, or every unsealed writer
    /// would report pending changes regardless of its buffer.
    fn writer_tracks_pending_changes(_writer: &Self::Writer) -> bool {
        false
    }
}

/// State shared between the index handle, its writers, and its searchers.
struct SegmentedShared<B: SegmentedBackend> {
    /// Segment manifest, index-level deletion bitmap (`{name}.delmap`) and
    /// WAL checkpoint. Segment files are named by the manager
    /// (`segment_NNNNNN{B::LAYOUT.primary}`).
    core: SegmentedCore,

    /// Per-segment reader cache; invalidated on merge, and rebuilt once the
    /// deletion bitmap is created.
    reader_cache: SegmentedReaderCache<B::Reader>,
}

impl<B: SegmentedBackend> fmt::Debug for SegmentedShared<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentedShared")
            .field("core", &self.core)
            .field("reader_cache", &self.reader_cache)
            .finish()
    }
}

impl<B: SegmentedBackend> SegmentedShared<B> {
    /// Sealed segment readers, newest generation first, with the deletion
    /// bitmap attached.
    fn sealed_readers_newest_first(
        &self,
        config: &B::Config,
    ) -> Result<Vec<Arc<dyn VectorIndexReader>>> {
        let readers =
            self.reader_cache
                .sealed_readers_newest_first(&self.core, |segment_id, bitmap| {
                    B::load_segment_reader(config, self.core.storage(), segment_id, bitmap)
                })?;
        Ok(readers
            .into_iter()
            .map(|r| r as Arc<dyn VectorIndexReader>)
            .collect())
    }

    /// Merge `segments` into one new segment, dropping the documents
    /// `deletions` (a [`SegmentedCore::deletion_snapshot`]) marks, and drop
    /// their cached readers.
    fn merge(
        &self,
        config: &B::Config,
        segments: Vec<ManagedSegmentInfo>,
        deletions: Option<Arc<DeletionBitmap>>,
    ) -> Result<()> {
        let source_ids = self.core.merge_segments(
            segments,
            deletions,
            |deletions, segments, new_segment_id| {
                B::merge_segments(
                    config,
                    self.core.storage(),
                    deletions,
                    segments,
                    new_segment_id,
                )
            },
        )?;
        for id in &source_ids {
            self.reader_cache.invalidate(id);
        }
        Ok(())
    }
}

/// Segment-per-commit vector index, generic over its backend (see the module
/// docs).
pub struct SegmentedIndex<B: SegmentedBackend> {
    shared: Arc<SegmentedShared<B>>,
    config: B::Config,
    closed: AtomicBool,
}

impl<B: SegmentedBackend> fmt::Debug for SegmentedIndex<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentedIndex")
            .field("shared", &self.shared)
            .field("config", &self.config)
            .field("closed", &self.closed)
            .finish()
    }
}

impl<B: SegmentedBackend> SegmentedIndex<B> {
    /// Open an existing segmented index or create a new one.
    ///
    /// A legacy monolithic index is migrated **zero-copy**: the existing
    /// `{name}{LAYOUT.primary}` is registered verbatim as the first segment
    /// — its segment id is the index name, which every reader treats as a
    /// plain path prefix — with a single atomic manifest write and no data
    /// movement. A crash before the manifest save leaves the legacy layout
    /// intact, so the migration simply re-runs on the next open. The
    /// pre-existing `{name}.delmap` keeps its meaning: it is the index-level
    /// deletion bitmap in both layouts (and `delete_segment_files` never
    /// touches `.delmap` files for exactly this reason).
    ///
    /// The manifest records the kind, dimension and distance the segments
    /// are written with, and an index that has segments only opens under
    /// the same ones (Issue #1547; see [`check_field_record`]).
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest fails to load (corruption fails
    /// loudly), [`LaurusError::InvalidArgument`] when the segments were
    /// written with another kind, dimension or distance than `config`'s,
    /// and an error when reading the legacy file's header during migration
    /// fails, or when the legacy file's persisted dimension does not match
    /// `config`'s.
    pub fn open_or_create(
        storage: Arc<dyn Storage>,
        name: &str,
        config: B::Config,
    ) -> Result<Self> {
        // The generated-segment namespace is reserved: an index named like a
        // generated id would collide with sealed segments (same file names,
        // orphan-sweep patterns, and merge GC) — reject it loudly.
        if let Some(ordinal) = name.strip_prefix("segment_")
            && !ordinal.is_empty()
            && ordinal.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(LaurusError::invalid_config(format!(
                "index name '{name}' collides with the reserved segment-id \
                 namespace (segment_<digits>)"
            )));
        }

        let legacy_file = format!("{name}{}", B::LAYOUT.primary);
        let migrate = storage.file_exists(&legacy_file) && !storage.file_exists("segments.json");

        // The manager's orphan sweep only touches `segment_NNNNNN.*` names,
        // so an unmigrated legacy file is never swept.
        let manager = Arc::new(SegmentManager::new(
            SegmentManagerConfig::default(),
            storage.clone(),
            B::LAYOUT,
        )?);

        // Before a migration registers the legacy file: until then there is
        // no segment, so the check only records `config`, which the
        // migration's manifest save then writes.
        check_field_record(
            &manager,
            storage.as_ref(),
            FieldRecord::new(
                &B::LAYOUT,
                B::dimension(&config),
                B::distance_metric(&config),
            ),
            |file| B::read_segment_header(&storage, file).map(|(_, dimension)| dimension),
        )?;

        if migrate {
            let vector_count = B::read_legacy_vector_count(&config, &storage, &legacy_file)?;
            // `add_segment` stamps generation 1, measures the on-storage
            // size, and saves the manifest atomically — the moment it
            // returns, the index is segmented; before that, it is still a
            // valid legacy index.
            let info = ManagedSegmentInfo::new(name.to_string(), vector_count, 0, 0);
            manager.add_segment(info)?;

            // Drop the monolithic index's now-stale `metadata.json` so the
            // factory's open/create routing can never again mistake this
            // directory for a monolithic index.
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

    /// Merge one policy-selected window of segments.
    ///
    /// Uses the generation-contiguous, size-similar `TieredMergePolicy`
    /// through the deletion-filtering, duplicate-collapsing merge engine;
    /// source readers are invalidated afterwards. Returns whether a merge
    /// actually ran.
    fn merge_once(&self) -> Result<bool> {
        let Some(candidate) = self
            .shared
            .core
            .manager()
            .check_merge(&TieredMergePolicy::new())
        else {
            return Ok(false);
        };
        let deletions = self.shared.core.deletion_snapshot()?;
        self.shared
            .merge(&self.config, candidate.segments, deletions)?;
        Ok(true)
    }
}

impl<B: SegmentedBackend> VectorIndex for SegmentedIndex<B> {
    fn reader(&self) -> Result<Arc<dyn VectorIndexReader>> {
        self.check_closed()?;
        let readers = self.shared.sealed_readers_newest_first(&self.config)?;
        let bitmap = self.shared.core.bitmap()?;
        Ok(Arc::new(SegmentedReaderFacade::new(
            readers,
            bitmap,
            B::dimension(&self.config),
            B::distance_metric(&self.config),
        )))
    }

    fn writer(&self) -> Result<Box<dyn VectorIndexWriter>> {
        self.check_closed()?;

        // A fresh, EMPTY active-segment writer per call: constructing it does
        // not touch the existing segments — the whole point of the layout.
        let segment_id = self.shared.core.manager().generate_segment_id();
        let inner = B::new_segment_writer(&self.config, self.shared.core.storage(), &segment_id)?;
        let writer = SegmentedWriter::<B> {
            shared: self.shared.clone(),
            segment_id,
            inner,
            changed_since_seal: None,
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
        // ids) per call; the approximation is intentional (mirrors the
        // pre-#1409 per-backend code).
        let manager = self.shared.core.manager();
        let total = manager.total_vectors();
        let deleted = self.shared.core.deleted_count()?;
        let stats = manager.stats();
        Ok(VectorIndexStats {
            vector_count: total.saturating_sub(deleted),
            dimension: B::dimension(&self.config),
            total_size: stats.total_size,
            deleted_count: deleted,
            last_modified: 0,
        })
    }

    fn retain_writer_after_commit(&self) -> bool {
        // Retention exists only to avoid a monolithic writer's O(index)
        // reload after commit. A segmented writer starts EMPTY — there is
        // nothing to reload — so the store can drop it and construct a
        // fresh one per commit cycle; each commit then seals exactly one
        // new segment.
        false
    }

    fn optimize(&self) -> Result<()> {
        self.optimize_until_published()?.map_or(Ok(()), Err)
    }

    fn optimize_until_published(&self) -> Result<Option<LaurusError>> {
        self.check_closed()?;

        let segments = self.shared.core.manager().list_segments();
        if segments.is_empty() {
            return Ok(None);
        }
        // Force-merge everything into one segment, filtering with a frozen
        // copy of the deletions: a soft delete takes no writer lock and can
        // land while the merge runs (Issue #1518).
        let deletions = self.shared.core.deletion_snapshot()?;
        self.shared
            .merge(&self.config, segments, deletions.clone())?;

        // Every document the copy marks was physically dropped by the merge;
        // a delete marked since stays marked. The merge has published, so a
        // failure here is reported apart (Issue #1538): the bitmap then keeps
        // the copy's marks, which name only documents the merge dropped.
        let forgotten = self.shared.core.forget_deletions(deletions.as_deref());
        Ok(forgotten.err())
    }

    // `refresh` keeps its trait default (no-op), like the monolithic index:
    // sealed segment files are immutable, segment-set changes are picked up
    // from the shared manager on every `searcher()`/`reader()` call, and
    // merges invalidate their source readers explicitly. Clearing the reader
    // cache here would force an O(all segments) reload on every commit —
    // defeating the O(delta) design this index exists for.

    fn searcher(&self) -> Result<Box<dyn VectorIndexSearcher>> {
        self.check_closed()?;
        let readers = self.shared.sealed_readers_newest_first(&self.config)?;
        let bitmap = self.shared.core.bitmap()?;
        // `SegmentFanoutSearcher::new`'s per-segment builder must be `'static`,
        // so the config is cloned into it rather than borrowed; its heavy
        // fields (embedder, PQ codebook) are `Arc`s, so this is cheap.
        let config = self.config.clone();
        Ok(Box::new(SegmentFanoutSearcher::new(
            readers,
            bitmap,
            move |reader| B::segment_searcher(&config, reader),
        )))
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        B::embedder(&self.config)
    }

    fn last_wal_seq(&self) -> u64 {
        // The PUBLISHED checkpoint: the value loaded from (or last saved
        // to) the manifest — never the pending one, which may not be
        // durable yet. Recovery skips WAL records at or below this seq, so
        // it must only ever reflect state that is already on storage.
        self.shared.core.last_wal_seq()
    }

    fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
        // Recorded as PENDING only. It is published into the manifest at
        // the end of `persist_deletions` — the last vector step of the
        // store's commit ladder — at which point the sealed segment files
        // AND the deletion bitmap for every record up to `seq` are
        // durable, and the engine has not yet truncated the WAL (a
        // checkpoint must never outrun the durability of the state it
        // covers).
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
        // The last vector step of the store's commit ladder: the sealed
        // segment files are durable and registered, so once the bitmap is
        // written the pending WAL checkpoint is published. `Engine::commit`
        // truncates the WAL only after all stores finish, so a crash on
        // either side of the publish is safe.
        self.shared.core.persist_deletions()
    }

    fn write_deletions(&self) -> Result<()> {
        self.shared.core.write_deletions()
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

        if B::auto_compaction(&self.config)
            && self
                .shared
                .core
                .needs_compaction(B::compaction_threshold(&self.config))?
        {
            self.optimize()?;
            return Ok(true);
        }
        Ok(false)
    }
}

/// Active-segment writer for [`SegmentedIndex<B>`].
///
/// Buffers new vectors in an ordinary backend writer whose path is a fresh
/// `segment_NNNNNN`; [`VectorIndexWriter::commit`] seals it as a new
/// immutable segment file and registers it in the manifest exactly once. A
/// sealed writer is DONE, even after a rollback: committing again with new
/// changes is rejected — re-writing a registered segment would carry its
/// original generation while newer segments may have sealed in between,
/// inverting the newest-wins ordering. The store never does this
/// (`retain_writer_after_commit` is false); direct API users get a loud
/// error instead of silent metadata corruption.
struct SegmentedWriter<B: SegmentedBackend> {
    shared: Arc<SegmentedShared<B>>,
    segment_id: String,
    inner: B::Writer,
    /// `None` until this writer's segment is sealed and registered, then
    /// whether the buffer has changed since the seal or the last rollback.
    /// The inner buffer is retained after commit, so this keeps the
    /// post-commit `close()` a no-op while any later change makes `commit`
    /// fail — including a same-size replacement, which a buffer-length
    /// comparison would miss.
    changed_since_seal: Option<bool>,
}

impl<B: SegmentedBackend> SegmentedWriter<B> {
    /// Record a buffer change; after sealing, it makes `commit` fail.
    fn mark_changed(&mut self) {
        if let Some(changed) = &mut self.changed_since_seal {
            *changed = true;
        }
    }
}

impl<B: SegmentedBackend> fmt::Debug for SegmentedWriter<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentedWriter")
            .field("shared", &self.shared)
            .field("segment_id", &self.segment_id)
            .field("inner", &self.inner)
            .field("changed_since_seal", &self.changed_since_seal)
            .finish()
    }
}

impl<B: SegmentedBackend> VectorIndexWriter for SegmentedWriter<B> {
    fn next_vector_id(&self) -> u64 {
        self.inner.next_vector_id()
    }

    fn build(
        &mut self,
        vectors: Vec<(u64, String, crate::vector::core::vector::Vector)>,
    ) -> Result<()> {
        self.add_vectors(vectors)
    }

    fn add_vectors(
        &mut self,
        vectors: Vec<(u64, String, crate::vector::core::vector::Vector)>,
    ) -> Result<()> {
        // Same-id upsert completion: the delete-first step marked the ids
        // so sealed copies stop matching; clear the marks so the NEW
        // copies are not shadowed by their own delete once this segment
        // seals. Revived stale copies are masked by newest-wins containment
        // at search and collapsed at merge.
        self.shared
            .core
            .unmark_deleted(vectors.iter().map(|(doc_id, _, _)| *doc_id))?;
        if !vectors.is_empty() {
            self.mark_changed();
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

    fn vectors(&self) -> &[(u64, String, crate::vector::core::vector::Vector)] {
        self.inner.vectors()
    }

    fn write(&self) -> Result<()> {
        if self.inner.vectors().is_empty() {
            // Nothing buffered — sealing would create an empty segment file
            // per commit. Deletion persistence is handled by
            // `persist_deletions` on the index.
            return Ok(());
        }
        // Writes the segment's primary file; manifest registration happens
        // in `commit()` (`&mut self`), which is the store's entry point.
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
        self.shared.core.mark_deleted(doc_id)?;
        // The inner writers report success whether or not the id was
        // buffered; only a shrunken buffer means a buffered copy was removed.
        let buffered = self.inner.vectors().len();
        let _ = self.inner.delete_document(doc_id);
        if self.inner.vectors().len() != buffered {
            self.mark_changed();
        }
        Ok(())
    }

    fn has_pending_changes(&self) -> bool {
        match self.changed_since_seal {
            Some(changed) => changed,
            None => {
                !self.inner.vectors().is_empty() || B::writer_tracks_pending_changes(&self.inner)
            }
        }
    }

    fn commit(&mut self) -> Result<()> {
        if self.inner.vectors().is_empty() {
            return Ok(());
        }
        if let Some(changed) = self.changed_since_seal {
            if !changed {
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
        self.shared.core.manager().add_segment(info)?;
        self.changed_since_seal = Some(false);
        self.shared.reader_cache.invalidate(&self.segment_id);
        Ok(())
    }

    /// Discard the changes buffered since the last commit. A sealed writer
    /// stays sealed: its segment is registered, so a later change must still
    /// make `commit` fail, while `has_pending_changes` is `false` right after
    /// the rollback, which emptied the buffer.
    fn rollback(&mut self) -> Result<()> {
        // Rolled-back mutations are discarded, so the pending WAL
        // checkpoint must not keep covering their sequence numbers: roll
        // it back to the published value; their WAL records then stay
        // replayable.
        self.shared.core.rollback_pending_wal_seq();
        self.inner.rollback()?;
        if let Some(changed) = &mut self.changed_since_seal {
            *changed = false;
        }
        Ok(())
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

impl<B: SegmentedBackend> Drop for SegmentedWriter<B> {
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
            self.shared.core.rollback_pending_wal_seq();
        }
    }
}
