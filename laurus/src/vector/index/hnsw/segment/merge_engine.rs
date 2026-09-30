//! Merge engine for vector index segments.
//!
//! This module handles the actual merging of segments. [`MergeStats`] and
//! [`MergeResult`] are the shared, index-type-agnostic data shapes defined
//! in [`crate::vector::index::segment::merge`]; this engine's own logic
//! (below) is HNSW-graph-typed.

use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::core::vector::Vector;

use crate::vector::index::segment::manager::ManagedSegmentInfo;
use crate::vector::index::segment::merge::{MergeResult, MergeStats};

use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::reader::VectorIndexReader;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

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
        let start_time = crate::util::time::Timer::now();

        // Calculate statistics
        let segments_merged = segments.len() as u32;
        let mut deletions_removed = 0;
        let mut duplicates_removed = 0u64;

        let mut all_vectors: Vec<(u64, String, Vector)> = Vec::new();

        // Newest generation FIRST, so the first occurrence of a
        // `(doc_id, field)` key is the authoritative (newest) copy and every
        // later one is a stale duplicate (Issue #880).
        let mut sources = segments.clone();
        sources.sort_by_key(|s| std::cmp::Reverse(s.generation));
        let mut seen: std::collections::HashSet<(u64, String)> = std::collections::HashSet::new();

        // 1. Read all live vectors from source segments (newest first)
        for segment in &sources {
            // Note: HnswIndexReader::load expects path without extension
            let reader = HnswIndexReader::load(
                self.storage.clone(),
                &segment.segment_id,
                self.index_config.distance_metric,
            )?;

            // Issue #795: prefer the source segment's original f32 rerank
            // sidecar (lossless) over the int8-dequantized iterator value,
            // so a merge does not bake one round of quantization error into
            // the merged sidecar. The pool is already loaded by the reader
            // (Eager mode, zero extra I/O); when absent (no sidecar, or Lazy
            // mode) we keep the int8-dequantized vector — the best available
            // source.
            let rerank_pool = reader.rerank_storage().cloned();

            let mut iterator = reader.vector_iterator()?;
            while let Some((doc_id, field, vector)) = iterator.next()? {
                if let Some(bitmap) = &self.deletion_bitmap
                    && bitmap.is_deleted(doc_id)
                {
                    deletions_removed += 1;
                    continue;
                }
                if !seen.insert((doc_id, field.clone())) {
                    // A newer segment already contributed this key.
                    duplicates_removed += 1;
                    continue;
                }
                let vector = match rerank_pool
                    .as_ref()
                    .and_then(|pool| pool.get_f32_slice(doc_id, &field))
                {
                    Some(f32_vector) => Vector::new(f32_vector.to_vec()),
                    None => vector,
                };
                all_vectors.push((doc_id, field, vector));
            }
        }

        // 2. Write to new segment
        // We use with_storage to ensure it writes to the correct location
        let vectors_merged = all_vectors.len() as u64;
        let total_size = vectors_merged * 128; // Dummy estimate; measured from storage by `add_segment`.
        let mut writer = HnswIndexWriter::with_storage(
            self.index_config.clone(),
            self.writer_config.clone(),
            &new_segment_id,
            self.storage.clone(),
        )?;

        // Issue #636: move instead of clone -- `all_vectors` is not read
        // again after this call, and merges are routine under
        // segment-per-commit's tiered policy rather than rare full
        // rewrites, so the clone's peak-memory doubling recurs on every
        // merge instead of occasionally.
        writer.add_vectors(all_vectors)?;
        writer.finalize()?;
        writer.write()?;

        let merge_time_ms = start_time.elapsed_ms();

        let merged_segment = ManagedSegmentInfo {
            segment_id: new_segment_id,
            vector_count: vectors_merged,
            vector_offset: 0,
            // The merged output represents data no newer than its newest
            // source, so it inherits max(source generations) — NOT max+1
            // (Issue #880): with +1 a merge over non-adjacent sources would
            // out-rank untouched newer segments, laundering a stale copy
            // from an old source above a genuinely newer one under the
            // newest-generation-wins dedup. Generations are unique (stamped
            // max+1 at flush) and the sources are removed with the merge, so
            // inheriting the maximum cannot collide with a live segment.
            generation: segments.iter().map(|s| s.generation).max().unwrap_or(0),
            has_deletions: false,
            size_bytes: total_size,
        };

        let stats = MergeStats {
            segments_merged,
            vectors_merged,
            deletions_removed,
            duplicates_removed,
            merge_time_ms,
            merged_size_bytes: total_size,
        };

        Ok(MergeResult {
            merged_segment,
            stats,
            merged_segment_ids: segments.iter().map(|s| s.segment_id.clone()).collect(),
        })
    }

    /// Get storage reference.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_stats() {
        let stats = MergeStats {
            segments_merged: 3,
            vectors_merged: 1000,
            deletions_removed: 200,
            duplicates_removed: 0,
            merge_time_ms: 100,
            merged_size_bytes: 102400,
        };

        assert_eq!(stats.compression_ratio(), 0.8);
    }
}
