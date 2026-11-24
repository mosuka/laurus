//! Merge engine for IVF vector index segments.
//!
//! The merge is the one Flat, HNSW and IVF share, in
//! [`crate::vector::index::segment::merge`]; this engine supplies IVF's
//! segment reader and writer.
//!
//! Unlike Flat/HNSW, an IVF segment's inverted-list structure is meaningless
//! across a merge — cluster ids are assigned independently per segment (each
//! trains its own centroids from adaptive K over its own buffer, Issue #889),
//! so there is no way to reconcile cluster `i` in one source with cluster `i`
//! in another. A merge therefore discards every source's inverted lists
//! entirely, reduces to a flat, deduplicated, deletion-filtered
//! `(doc_id, field, Vector)` list (same as Flat's merge), and re-clusters
//! that union from scratch through the ordinary
//! [`IvfIndexWriter`]/`add_vectors`/`finalize` pipeline. No special
//! "expected count" setter is needed: `finalize` -> `train_centroids`
//! already recomputes the effective cluster count as
//! `configured_n_clusters.min(vectors.len()).max(1)` (Issue #889 PR-5) from
//! whatever `index_config.n_clusters` the writer was constructed with, so
//! constructing it with the same schema-level config the segmented index
//! already uses is sufficient to re-derive the adaptive K for the merged
//! union's size — including the zero-vectors case (Issue #889 PR-6), which
//! trains zero clusters instead of erroring.

use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;

use crate::vector::index::segment::manager::ManagedSegmentInfo;
use crate::vector::index::segment::merge::{self, MergeResult, MergeSource};

use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::index::config::IvfIndexConfig;
use crate::vector::index::ivf::reader::IvfIndexReader;
use crate::vector::index::ivf::writer::IvfIndexWriter;
use crate::vector::writer::VectorIndexWriterConfig;

/// Engine for merging IVF vector index segments.
pub struct MergeEngine {
    storage: Arc<dyn Storage>,
    index_config: IvfIndexConfig,
    writer_config: VectorIndexWriterConfig,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
}

impl MergeEngine {
    /// Create a new merge engine.
    pub fn new(
        storage: Arc<dyn Storage>,
        index_config: IvfIndexConfig,
        writer_config: VectorIndexWriterConfig,
    ) -> Self {
        Self {
            storage,
            index_config,
            writer_config,
            deletion_bitmap: None,
        }
    }

    /// Set deletion bitmap for filtering deleted vectors during merge.
    pub fn set_deletion_bitmap(&mut self, bitmap: Arc<DeletionBitmap>) {
        self.deletion_bitmap = Some(bitmap);
    }

    /// Merge multiple segments into a single, freshly-reclustered segment.
    ///
    /// Reads every live vector from the source segments (deletion-filtered
    /// via the configured bitmap), deduplicates cross-segment duplicates of
    /// the same `(doc_id, field)` with **newest-generation wins** semantics
    /// (same reasoning as Issue #880 for Flat/HNSW), then trains a fresh set
    /// of centroids over the deduplicated union and rebuilds the inverted
    /// lists from scratch — the per-segment cluster ids being merged carry
    /// no cross-segment meaning, so there is nothing to reconcile.
    pub fn merge_segments(
        &self,
        segments: Vec<ManagedSegmentInfo>,
        new_segment_id: String,
    ) -> Result<MergeResult> {
        merge::merge_segments(
            segments,
            new_segment_id,
            self.deletion_bitmap.as_deref(),
            |segment| {
                let reader = IvfIndexReader::load(
                    self.storage.clone(),
                    &segment.segment_id,
                    self.index_config.distance_metric,
                )?;
                let rerank = reader.rerank_storage().cloned();
                Ok(MergeSource {
                    reader: Box::new(reader),
                    rerank,
                })
            },
            |segment_id| {
                IvfIndexWriter::with_storage(
                    self.index_config.clone(),
                    self.writer_config.clone(),
                    segment_id,
                    self.storage.clone(),
                )
            },
        )
    }

    /// Get storage reference.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }
}
