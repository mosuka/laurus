//! Segment-per-commit HNSW vector index (Issue #634, PR-3 / #881).
//!
//! [`SegmentedHnswIndex`] implements [`VectorIndex`](crate::vector::index::VectorIndex) over a set of immutable
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
//! by [`VectorIndex::soft_delete_document`](crate::vector::index::VectorIndex::soft_delete_document), persisted to `{name}.delmap` by
//! [`VectorIndex::persist_deletions`](crate::vector::index::VectorIndex::persist_deletions), filtered by every segment reader, and
//! physically reclaimed by [`VectorIndex::optimize`](crate::vector::index::VectorIndex::optimize) (a force-merge). The
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
//!
//! The generic implementation ([`SegmentedIndex`] and its writer) lives
//! in [`crate::vector::index::segment::segmented_index`] (Issue #1409); this
//! module supplies only [`HnswSegmentedBackend`], the type that tells the
//! generic layer how to build HNSW's reader, writer, searcher and merge
//! engine, and the one behavior HNSW does not share with Flat/IVF: the inner
//! writer's own `has_pending_changes` must be consulted before a segment is
//! sealed.

use std::sync::Arc;

use crate::embedding::embedder::Embedder;
use crate::error::Result;
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::vector::DistanceMetric;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::searcher::HnswSearcher;
use crate::vector::index::hnsw::segment::merge_engine::MergeEngine;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::index::segment::manager::{ManagedSegmentInfo, SegmentFileLayout};
use crate::vector::index::segment::segmented_index::{SegmentedBackend, SegmentedIndex};
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::VectorIndexSearcher;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

/// Supplies HNSW's concrete types and construction logic to
/// [`SegmentedIndex`]. A zero-sized witness type, never instantiated.
#[derive(Debug)]
pub struct HnswSegmentedBackend;

impl SegmentedBackend for HnswSegmentedBackend {
    type Config = HnswIndexConfig;
    type Reader = HnswIndexReader;
    type Writer = HnswIndexWriter;

    const LAYOUT: SegmentFileLayout = crate::vector::index::hnsw::segment::LAYOUT;

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
        let mut reader =
            HnswIndexReader::load(storage.clone(), segment_id, config.distance_metric)?;
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
        HnswIndexWriter::with_storage(
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
        // Default `ef_search` for the multi-segment fan-out: the
        // schema-level value when configured, else
        // `ef_construction.max(50) * 2` — the compensation that kept
        // multi-segment recall stable (#644 / #880).
        let default_ef = config
            .default_ef_search
            .unwrap_or_else(|| config.ef_construction.max(50) * 2);
        Ok(Box::new(HnswSearcher::with_default_ef_search(
            reader,
            Some(default_ef),
        )?) as Box<dyn VectorIndexSearcher>)
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

    fn read_segment_header(storage: &Arc<dyn Storage>, file: &str) -> Result<(u64, usize)> {
        // The `.hnsw` header is `u64 num_vectors` followed by `u32
        // dimension` (then `m` and `ef_construction`, unneeded here).
        // Unlike the trait default's `u32` count, HNSW's count is `u64`, so
        // this cannot reuse the default implementation.
        use std::io::Read;

        let mut input = storage.open_input(file)?;
        let mut count_buf = [0u8; 8];
        input.read_exact(&mut count_buf)?;
        let mut dim_buf = [0u8; 4];
        input.read_exact(&mut dim_buf)?;
        Ok((
            u64::from_le_bytes(count_buf),
            u32::from_le_bytes(dim_buf) as usize,
        ))
    }

    fn writer_tracks_pending_changes(writer: &Self::Writer) -> bool {
        // `HnswIndexWriter` is the only backend writer that overrides
        // `has_pending_changes` (`!self.is_finalized`, `false` once
        // finalize+write has captured its in-memory state) instead of
        // leaving the trait's unconditional `true` default — so only here
        // does consulting it (in addition to the buffer check) add
        // information instead of making every unsealed writer permanently
        // "pending".
        writer.has_pending_changes()
    }
}

/// Segment-per-commit HNSW index (see the module docs).
pub type SegmentedHnswIndex = SegmentedIndex<HnswSegmentedBackend>;
