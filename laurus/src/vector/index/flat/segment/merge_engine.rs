//! Merge engine for Flat vector index segments.
//!
//! This module handles the actual merging of segments. [`MergeStats`] and
//! [`MergeResult`] are the shared, index-type-agnostic data shapes defined
//! in [`crate::vector::index::segment::merge`]; this
//! engine's own logic (below) is Flat-typed. Unlike HNSW, Flat has no f32
//! rerank sidecar (Issue #795 does not apply) and no graph, so a merge is
//! just: read every live vector from the source segments (newest
//! generation first, deletion-filtered, deduplicated), then write the
//! survivors into a new segment.

use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::core::vector::Vector;

use crate::vector::index::segment::manager::ManagedSegmentInfo;
use crate::vector::index::segment::merge::{MergeResult, MergeStats};

use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::index::config::FlatIndexConfig;
use crate::vector::index::flat::reader::FlatVectorIndexReader;
use crate::vector::index::flat::writer::FlatIndexWriter;
use crate::vector::reader::VectorIndexReader;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

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
        let start_time = crate::util::time::Timer::now();

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

        // 1. Read all live vectors from source segments (newest first).
        for segment in &sources {
            let reader = FlatVectorIndexReader::load(
                self.storage.clone(),
                &segment.segment_id,
                self.index_config.distance_metric,
            )?;

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
                all_vectors.push((doc_id, field, vector));
            }
        }

        // 2. Write to new segment.
        let vectors_merged = all_vectors.len() as u64;
        let total_size = vectors_merged * 128; // Dummy estimate; measured from storage by `add_segment`.
        let mut writer = FlatIndexWriter::with_storage(
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
