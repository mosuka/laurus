//! Segment-per-commit IVF vector index (Issue #889 PR-6, mirroring HNSW's
//! #634/#882 design and Flat's PR-4 — see
//! [`crate::vector::index::hnsw::segmented`] and
//! [`crate::vector::index::flat::segmented`]).
//!
//! [`SegmentedIvfIndex`] implements [`VectorIndex`](crate::vector::index::VectorIndex) over a set of immutable
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
//! by [`VectorIndex::soft_delete_document`](crate::vector::index::VectorIndex::soft_delete_document), persisted to `{name}.delmap`
//! by [`VectorIndex::persist_deletions`](crate::vector::index::VectorIndex::persist_deletions), filtered by every segment reader,
//! and physically reclaimed by [`VectorIndex::optimize`](crate::vector::index::VectorIndex::optimize) (a force-merge).
//! This is a first-time implementation for IVF — the monolithic
//! [`super::IvfIndex`] has no soft-delete/compaction wiring at all.
//!
//! [`IvfIndexConfig::segmented`] defaults to `true` (Issue #907, mirroring
//! HNSW's own #882 flip): this is now the default IVF layout, with the
//! monolithic single-file layout still available via `segmented: false`. A
//! legacy monolithic index is migrated zero-copy on first open with the
//! flag on (its `.ivf` becomes segment 0 of the manifest, no data
//! movement).
//!
//! The generic implementation ([`SegmentedIndex`] and its writer) lives
//! in [`crate::vector::index::segment::segmented_index`] (Issue #1409); this
//! module supplies only [`IvfSegmentedBackend`], the type that tells the
//! generic layer how to build IVF's reader, writer, searcher and merge
//! engine.

use std::sync::Arc;

use crate::embedding::embedder::Embedder;
use crate::error::Result;
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::vector::DistanceMetric;
use crate::vector::index::config::IvfIndexConfig;
use crate::vector::index::ivf::reader::IvfIndexReader;
use crate::vector::index::ivf::searcher::IvfSearcher;
use crate::vector::index::ivf::segment::merge_engine::MergeEngine;
use crate::vector::index::ivf::writer::IvfIndexWriter;
use crate::vector::index::segment::manager::{ManagedSegmentInfo, SegmentFileLayout};
use crate::vector::index::segment::segmented_index::{SegmentedBackend, SegmentedIndex};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::writer::VectorIndexWriterConfig;

/// Supplies IVF's concrete types and construction logic to
/// [`SegmentedIndex`]. A zero-sized witness type, never instantiated.
#[derive(Debug)]
pub struct IvfSegmentedBackend;

impl SegmentedBackend for IvfSegmentedBackend {
    type Config = IvfIndexConfig;
    type Reader = IvfIndexReader;
    type Writer = IvfIndexWriter;

    const LAYOUT: SegmentFileLayout = crate::vector::index::ivf::segment::LAYOUT;

    fn dimension(config: &Self::Config) -> usize {
        config.dimension
    }

    fn distance_metric(config: &Self::Config) -> DistanceMetric {
        config.distance_metric
    }

    fn embedder(config: &Self::Config) -> Arc<dyn Embedder> {
        Arc::clone(&config.embedder)
    }

    fn auto_compaction(config: &Self::Config) -> bool {
        config.auto_compaction
    }

    fn compaction_threshold(config: &Self::Config) -> f64 {
        config.compaction_threshold
    }

    fn load_segment_reader(
        config: &Self::Config,
        storage: &Arc<dyn Storage>,
        segment_id: &str,
        bitmap: Option<Arc<DeletionBitmap>>,
    ) -> Result<Self::Reader> {
        let mut reader = IvfIndexReader::load(storage.clone(), segment_id, config.distance_metric)?;
        if let Some(bitmap) = bitmap {
            reader.set_deletion_bitmap(bitmap);
        }
        Ok(reader)
    }

    fn new_segment_writer(
        config: &Self::Config,
        storage: &Arc<dyn Storage>,
        segment_id: &str,
    ) -> Result<Self::Writer> {
        IvfIndexWriter::with_storage(
            config.clone(),
            VectorIndexWriterConfig::default(),
            segment_id,
            storage.clone(),
        )
    }

    fn segment_searcher(
        config: &Self::Config,
        reader: Arc<dyn VectorIndexReader>,
    ) -> Result<Box<dyn VectorIndexSearcher>> {
        Ok(Box::new(IvfSearcher::with_n_probe(reader, config.n_probe)?)
            as Box<dyn VectorIndexSearcher>)
    }

    fn merge_segments(
        config: &Self::Config,
        storage: &Arc<dyn Storage>,
        bitmap: Option<Arc<DeletionBitmap>>,
        segments: Vec<ManagedSegmentInfo>,
        new_segment_id: String,
    ) -> Result<ManagedSegmentInfo> {
        let mut engine = MergeEngine::new(
            storage.clone(),
            config.clone(),
            VectorIndexWriterConfig::default(),
        );
        if let Some(bitmap) = bitmap {
            engine.set_deletion_bitmap(bitmap);
        }
        Ok(engine
            .merge_segments(segments, new_segment_id)?
            .merged_segment)
    }
}

/// Segment-per-commit IVF index (see the module docs).
pub type SegmentedIvfIndex = SegmentedIndex<IvfSegmentedBackend>;
