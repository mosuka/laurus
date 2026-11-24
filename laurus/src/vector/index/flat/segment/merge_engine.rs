//! Merge engine for Flat vector index segments.
//!
//! The merge is the one Flat, HNSW and IVF share, in
//! [`crate::vector::index::segment::merge`]; this engine supplies Flat's
//! segment writer. Flat has no graph, so the merged segment is the live
//! vectors of the sources and nothing more.

use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;

use crate::vector::index::segment::kind::SegmentKind;
use crate::vector::index::segment::manager::ManagedSegmentInfo;
use crate::vector::index::segment::merge::{self, MergeResult, MergeSources};

use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::index::config::FlatIndexConfig;
use crate::vector::index::flat::writer::FlatIndexWriter;
use crate::vector::writer::VectorIndexWriterConfig;

/// Engine for merging Flat vector index segments.
pub struct MergeEngine {
    storage: Arc<dyn Storage>,
    index_config: FlatIndexConfig,
    writer_config: VectorIndexWriterConfig,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
}

impl MergeEngine {
    /// Create a new merge engine.
    pub fn new(
        storage: Arc<dyn Storage>,
        index_config: FlatIndexConfig,
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

    /// Merge multiple segments into a single segment.
    ///
    /// Reads every live vector from the source segments (deletion-filtered
    /// via the configured bitmap), deduplicates cross-segment duplicates of
    /// the same `(doc_id, field)` with **newest-generation wins** semantics
    /// (Issue #880 — same-id upserts replayed from the WAL land in newer
    /// segments and must shadow the stale copies), and writes the
    /// survivors into a new segment.
    pub fn merge_segments(
        &self,
        segments: Vec<ManagedSegmentInfo>,
        new_segment_id: String,
    ) -> Result<MergeResult> {
        merge::merge_segments(
            segments,
            new_segment_id,
            self.deletion_bitmap.as_deref(),
            &MergeSources {
                storage: &self.storage,
                kind: SegmentKind::Flat,
                distance: self.index_config.distance_metric,
            },
            |segment_id| {
                FlatIndexWriter::with_storage(
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
