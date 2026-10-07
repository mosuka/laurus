//! Merge engine for IVF vector index segments.
//!
//! This module handles the actual merging of segments. [`MergeStats`] and
//! [`MergeResult`] are the shared, index-type-agnostic data shapes defined
//! in [`crate::vector::index::segment::merge`]; this engine's own logic
//! (below) is IVF-typed.
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
//!
//! Like HNSW (Issue #795), a source segment's f32 rerank sidecar value is
//! preferred over the iterator's int8-dequantized one when present, so a
//! merge does not bake quantization error into the merged sidecar (Issue
//! #1348).

use std::sync::Arc;

use crate::error::Result;
use crate::storage::Storage;
use crate::vector::core::vector::Vector;

use crate::vector::index::segment::manager::ManagedSegmentInfo;
use crate::vector::index::segment::merge::{MergeResult, MergeStats};

use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::index::config::IvfIndexConfig;
use crate::vector::index::ivf::reader::IvfIndexReader;
use crate::vector::index::ivf::writer::IvfIndexWriter;
use crate::vector::reader::VectorIndexReader;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};

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

        // 1. Read all live vectors from source segments (newest first),
        //    discarding their inverted-list/cluster structure entirely.
        for segment in &sources {
            let reader = IvfIndexReader::load(
                self.storage.clone(),
                &segment.segment_id,
                self.index_config.distance_metric,
            )?;

            // Issue #1348 (mirrors #795): prefer the source segment's
            // original f32 rerank sidecar (lossless) over the
            // int8-dequantized iterator value, so a merge does not bake
            // one round of quantization error into the merged sidecar.
            // The pool is already loaded by the reader; when absent (no
            // sidecar configured, or a source predating this fix) we
            // keep the int8-dequantized vector -- the best available
            // source. The iterator's own doc-id re-sort (Issue #1152)
            // does not matter here: the sidecar is addressed by
            // `(doc_id, field)`, never by iterator position.
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

        // 2. Re-cluster the deduplicated union from scratch and write it as
        //    a new segment. `finalize` -> `train_centroids` derives the
        //    adaptive cluster count from `all_vectors.len()` (see the
        //    module docs); the empty case (Issue #889 PR-6) trains zero
        //    clusters instead of erroring.
        let vectors_merged = all_vectors.len() as u64;
        let mut writer = IvfIndexWriter::with_storage(
            self.index_config.clone(),
            self.writer_config.clone(),
            &new_segment_id,
            self.storage.clone(),
        )?;
        writer.add_vectors(all_vectors)?;
        writer.finalize()?;
        writer.write()?;

        let total_size = vectors_merged * 128; // Dummy estimate; measured from storage by `add_segment`.

        let merge_time_ms = start_time.elapsed_ms();

        let merged_segment = ManagedSegmentInfo {
            segment_id: new_segment_id,
            vector_count: vectors_merged,
            vector_offset: 0,
            // The merged output represents data no newer than its newest
            // source, so it inherits max(source generations) — NOT max+1
            // (same reasoning as Issue #880 for Flat/HNSW): with +1 a merge
            // over non-adjacent sources would out-rank untouched newer
            // segments, laundering a stale copy from an old source above a
            // genuinely newer one under the newest-generation-wins dedup.
            // Generations are unique (stamped max+1 at flush) and the
            // sources are removed with the merge, so inheriting the maximum
            // cannot collide with a live segment.
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
