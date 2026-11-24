//! Merge engine for HNSW vector index segments.
//!
//! The merge is the one Flat, HNSW and IVF share, in
//! [`crate::vector::index::segment::merge`]; this engine supplies HNSW's
//! segment reader and writer. The writer builds a new graph over the live
//! vectors of the sources.

use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;

use crate::vector::index::segment::manager::ManagedSegmentInfo;
use crate::vector::index::segment::merge::{self, MergeResult, MergeSource};

use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::writer::VectorIndexWriterConfig;

/// Engine for merging vector index segments.
pub struct MergeEngine {
    storage: Arc<dyn Storage>,
    index_config: HnswIndexConfig,
    writer_config: VectorIndexWriterConfig,
    deletion_bitmap: Option<Arc<DeletionBitmap>>,
}

impl MergeEngine {
    /// Create a new merge engine.
    pub fn new(
        storage: Arc<dyn Storage>,
        index_config: HnswIndexConfig,
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
    /// via the configured bitmap, losslessly sourced from the f32 rerank
    /// sidecar when present — Issue #795), deduplicates cross-segment
    /// duplicates of the same `(doc_id, field)` with **newest-generation
    /// wins** semantics (Issue #880 — same-id upserts replayed from the WAL
    /// land in newer segments and must shadow the stale copies), and writes
    /// the survivors into a new segment.
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
                // `HnswIndexReader::load` expects the path without its extension.
                let reader = HnswIndexReader::load(
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
                HnswIndexWriter::with_storage(
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
