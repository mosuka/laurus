//! Shared merge-operation data types for vector index segments.
//!
//! [`MergeStats`] and [`MergeResult`] are index-type-agnostic; each index
//! type's own merge engine (e.g. `hnsw::segment::merge_engine::MergeEngine`)
//! performs the actual type-specific segment-reader/segment-writer I/O and
//! returns these shared shapes.

use serde::{Deserialize, Serialize};

use crate::vector::index::segment::manager::ManagedSegmentInfo;

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
