//! The merge of vector index segments that Flat, HNSW and IVF share (Issue
//! #1579).
//!
//! A merge reads every live vector of its source segments, each through the
//! reader of the kind it was written as, and writes the survivors into one
//! new segment. Only the writer of the new segment depends on the index
//! type, so each type's own merge engine (e.g.
//! `hnsw::segment::merge_engine::MergeEngine`) supplies it and leaves the
//! rest to `merge_segments`. Merging every segment of a field as another
//! kind is how `update_field` changes the kind. [`MergeStats`] and
//! [`MergeResult`] are what a merge returns.

use std::collections::HashSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;
use crate::vector::index::segment::kind::{LoadedSegment, SegmentKind};
use crate::vector::index::segment::manager::ManagedSegmentInfo;
use crate::vector::writer::VectorIndexWriter;

/// Statistics about a merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeStats {
    /// Number of segments merged.
    pub segments_merged: u32,

    /// Number of vectors in merged segment.
    pub vectors_merged: u64,

    /// Number of deleted vectors removed.
    pub deletions_removed: u64,

    /// Number of stale cross-segment duplicates removed — copies of a
    /// `(doc_id, field)` key shadowed by a newer segment's copy
    /// (Issue #880). Defaults to 0 when deserializing older stats.
    #[serde(default)]
    pub duplicates_removed: u64,

    /// Time taken for merge (in milliseconds).
    pub merge_time_ms: u64,

    /// Size of merged segment (in bytes).
    pub merged_size_bytes: u64,
}

impl MergeStats {
    /// Create new merge stats.
    pub fn new() -> Self {
        Self {
            segments_merged: 0,
            vectors_merged: 0,
            deletions_removed: 0,
            duplicates_removed: 0,
            merge_time_ms: 0,
            merged_size_bytes: 0,
        }
    }

    /// Calculate compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.vectors_merged == 0 {
            return 1.0;
        }
        1.0 - (self.deletions_removed as f64 / self.vectors_merged as f64)
    }
}

impl Default for MergeStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Information about the new merged segment.
    pub merged_segment: ManagedSegmentInfo,

    /// Statistics about the merge.
    pub stats: MergeStats,

    /// IDs of segments that were merged.
    pub merged_segment_ids: Vec<String>,
}

/// Where a merge reads its source segments, and as what.
pub(crate) struct MergeSources<'a> {
    /// The storage the segments are in.
    pub(crate) storage: &'a Arc<dyn Storage>,

    /// The kind of a segment that does not say its own.
    pub(crate) kind: SegmentKind,

    /// The distance metric the vectors are read under.
    pub(crate) distance: DistanceMetric,
}

impl MergeSources<'_> {
    /// Load `segment` through the reader of its kind. The reader filters no
    /// deletions, or the merge could not count them.
    fn load(&self, segment: &ManagedSegmentInfo) -> Result<LoadedSegment> {
        let kind = match segment.kind.as_deref() {
            Some(name) => SegmentKind::of_segment(&segment.segment_id, name)?,
            None => self.kind,
        };
        kind.load(self.storage, &segment.segment_id, self.distance, None)
    }
}

/// The vectors of a merge's source segments that the merged segment holds.
struct LiveVectors {
    vectors: Vec<(u64, String, Vector)>,
    deletions_removed: u64,
    duplicates_removed: u64,
}

/// Read every live vector of `segments`, the newest generation first.
///
/// A vector of a document `deletions` marks is dropped, and so is a copy of
/// a `(doc_id, field)` that a newer segment holds (Issue #880: a same-id
/// upsert replayed from the WAL lands in a newer segment and must shadow the
/// stale copy).
///
/// A segment's f32 rerank sidecar is preferred over the iterator's value,
/// which is decoded from the quantized one, so a merge does not bake a round
/// of quantization error into the merged segment (Issues #795, #1348). It is
/// addressed by `(doc_id, field)`, never by the iterator's position. Without
/// a sidecar (none configured, or a segment written before it was), the
/// decoded value is the best there is.
fn collect_live_vectors(
    segments: &[ManagedSegmentInfo],
    deletions: Option<&DeletionBitmap>,
    sources: &MergeSources<'_>,
) -> Result<LiveVectors> {
    let mut live = LiveVectors {
        vectors: Vec::new(),
        deletions_removed: 0,
        duplicates_removed: 0,
    };

    // Newest generation FIRST, so the first occurrence of a
    // `(doc_id, field)` key is the authoritative (newest) copy and every
    // later one is a stale duplicate (Issue #880).
    let mut newest_first: Vec<&ManagedSegmentInfo> = segments.iter().collect();
    newest_first.sort_by_key(|s| std::cmp::Reverse(s.generation));
    let mut seen: HashSet<(u64, String)> = HashSet::new();

    for segment in newest_first {
        let source = sources.load(segment)?;
        let mut iterator = source.reader.vector_iterator()?;
        while let Some((doc_id, field, vector)) = iterator.next()? {
            if deletions.is_some_and(|bitmap| bitmap.is_deleted(doc_id)) {
                live.deletions_removed += 1;
                continue;
            }
            if !seen.insert((doc_id, field.clone())) {
                // A newer segment already contributed this key.
                live.duplicates_removed += 1;
                continue;
            }
            let vector = match source
                .rerank
                .as_ref()
                .and_then(|pool| pool.get_f32_slice(doc_id, &field))
            {
                Some(f32_vector) => Vector::new(f32_vector.to_vec()),
                None => vector,
            };
            live.vectors.push((doc_id, field, vector));
        }
    }
    Ok(live)
}

/// Merge `segments` into one new segment named `new_segment_id`, without the
/// documents `deletions` marks.
///
/// # Arguments
///
/// * `segments` - The source segments, in any order.
/// * `new_segment_id` - The id of the merged segment.
/// * `deletions` - The documents to leave out, if any.
/// * `sources` - Where and as what to read the source segments.
/// * `new_writer` - Creates the empty writer of the merged segment, called
///   once the sources have been read.
///
/// # Errors
///
/// Returns an error if a source cannot be read or the merged segment cannot
/// be written.
pub(crate) fn merge_segments<W: VectorIndexWriter>(
    segments: Vec<ManagedSegmentInfo>,
    new_segment_id: String,
    deletions: Option<&DeletionBitmap>,
    sources: &MergeSources<'_>,
    new_writer: impl FnOnce(&str) -> Result<W>,
) -> Result<MergeResult> {
    let start_time = crate::util::time::Timer::now();

    let live = collect_live_vectors(&segments, deletions, sources)?;
    let vectors_merged = live.vectors.len() as u64;
    let total_size = vectors_merged * 128; // Dummy estimate; measured from storage by `add_segment`.

    // Issue #636: move instead of clone -- the vectors are not read again
    // after this call, and merges are routine under segment-per-commit's
    // tiered policy rather than rare full rewrites, so the clone's
    // peak-memory doubling recurs on every merge instead of occasionally.
    let mut writer = new_writer(&new_segment_id)?;
    writer.add_vectors(live.vectors)?;
    writer.finalize()?;
    writer.write()?;

    let stats = MergeStats {
        segments_merged: segments.len() as u32,
        vectors_merged,
        deletions_removed: live.deletions_removed,
        duplicates_removed: live.duplicates_removed,
        merge_time_ms: start_time.elapsed_ms(),
        merged_size_bytes: total_size,
    };
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
        size_bytes: total_size,
        // The manager that publishes the merge makes it one of its own kind.
        kind: None,
    };

    Ok(MergeResult {
        merged_segment,
        stats,
        merged_segment_ids: segments.into_iter().map(|s| s.segment_id).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::core::rerank::RerankStorageKind;
    use crate::vector::index::config::FlatIndexConfig;
    use crate::vector::index::flat::reader::FlatVectorIndexReader;
    use crate::vector::index::flat::segment::merge_engine::MergeEngine;
    use crate::vector::index::flat::writer::FlatIndexWriter;
    use crate::vector::writer::VectorIndexWriterConfig;

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

    /// The merged segment takes the generation of its newest source, neither
    /// above it nor below (Issue #880), and the result counts what the merge
    /// left out.
    #[test]
    fn a_merge_takes_its_newest_generation_and_counts_what_it_left_out() {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = FlatIndexConfig {
            dimension: 2,
            distance_metric: DistanceMetric::Euclidean,
            normalize_vectors: false,
            rerank_storage: Some(RerankStorageKind::F32),
            ..Default::default()
        };
        // (id, generation, [(doc, x)]), listed the older first.
        let sources = [
            ("segment_0", 2, vec![(3, 30.0), (4, 40.0), (5, 50.0)]),
            ("segment_1", 5, vec![(1, 1.0), (2, 2.0), (3, 3.0)]),
        ];
        let mut segments = Vec::new();
        for (id, generation, docs) in &sources {
            let mut writer = FlatIndexWriter::with_storage(
                config.clone(),
                VectorIndexWriterConfig::default(),
                *id,
                storage.clone(),
            )
            .unwrap();
            let vectors = docs
                .iter()
                .map(|&(doc, x)| (doc, "v".to_string(), Vector::new(vec![x, 1.0])))
                .collect();
            writer.add_vectors(vectors).unwrap();
            writer.finalize().unwrap();
            writer.write().unwrap();
            segments.push(ManagedSegmentInfo::new(
                id.to_string(),
                docs.len() as u64,
                0,
                *generation,
            ));
        }
        let deletions = DeletionBitmap::new("index".to_string(), 0, u64::MAX - 1);
        for doc in [2, 4] {
            assert!(deletions.delete_document(doc).unwrap());
        }

        let mut engine = MergeEngine::new(
            storage.clone(),
            config.clone(),
            VectorIndexWriterConfig::default(),
        );
        engine.set_deletion_bitmap(Arc::new(deletions));
        let result = engine
            .merge_segments(segments, "segment_2".to_string())
            .unwrap();

        assert_eq!(result.merged_segment.segment_id, "segment_2");
        assert_eq!(result.merged_segment.generation, 5);
        assert_eq!(result.merged_segment.vector_count, 3);
        assert_eq!(result.merged_segment_ids, ["segment_0", "segment_1"]);
        assert_eq!(result.stats.segments_merged, 2);
        assert_eq!(result.stats.vectors_merged, 3);
        assert_eq!(result.stats.deletions_removed, 2);
        assert_eq!(result.stats.duplicates_removed, 1);

        let reader =
            FlatVectorIndexReader::load(storage, "segment_2", config.distance_metric).unwrap();
        let pool = reader.rerank_storage().unwrap();
        let held: Vec<(u64, Option<f32>)> = (1..=5)
            .map(|doc| (doc, pool.get_f32_slice(doc, "v").map(|v| v[0])))
            .collect();
        assert_eq!(
            held,
            [
                (1, Some(1.0)),
                (2, None),
                (3, Some(3.0)),
                (4, None),
                (5, Some(50.0))
            ]
        );
    }
}
