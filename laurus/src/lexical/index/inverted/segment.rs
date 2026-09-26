//! Segment management for inverted indexes.
//!
//! This module handles segment operations for inverted indexes:
//! - Segment manager for coordinating segments
//! - Merge engine for combining segments
//! - Merge policy for determining when to merge

use std::sync::Arc;

use roaring::RoaringTreemap;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::maintenance::deletion::DeletionBitmap;

/// Information about a segment in the inverted index.
///
/// This structure contains metadata about an individual segment,
/// including document counts, offsets, and deletion status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// Segment identifier.
    pub segment_id: String,

    /// Number of documents in this segment.
    pub doc_count: u64,

    /// Minimum document ID in this segment.
    pub min_doc_id: u64,

    /// Maximum document ID in this segment.
    pub max_doc_id: u64,

    /// Generation number of this segment.
    pub generation: u64,

    /// Whether this segment has deletions.
    pub has_deletions: bool,

    /// How many of this segment's documents are deleted (Issue #1212).
    ///
    /// Recorded by the writer whenever it persists deletions against the
    /// segment, from the segment's deletion bitmap and the ids it holds, so
    /// the index's document and deletion counts can be summed from the
    /// manifest rather than accumulated as deltas that drift. `None` means
    /// not recorded — an entry written by an older build, which open
    /// recounts. Flushed and merged segments start at `Some(0)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deleted_count: Option<u64>,

    /// Shard ID for this segment.
    pub shard_id: u16,
}

/// What a segment is known to hold (Issue #1211).
///
/// Deletions and counts address global doc ids, and a segment's
/// `[min_doc_id, max_doc_id]` range can contain ids it does not hold — a
/// merge drops deleted documents and can span segments it left out,
/// concurrent puts can reach the writer out of id order, and callers may
/// choose ids — so the range alone does not say which ids are the
/// segment's. Shared by the writer (which segments a deletion marks), the
/// reader (live counts and deletion checks) and the index's recount of
/// per-segment deletion counts (Issue #1212).
#[derive(Debug, Clone)]
pub(crate) enum Membership {
    /// No documents: a merge whose every source document was deleted.
    Empty,
    /// Every id of the range: the range has no gaps.
    Range,
    /// Exactly these ids, from the `.ids` part (or `.norms` for a segment
    /// written before it existed).
    Set(Arc<RoaringTreemap>),
    /// Not recorded (a pre-#555 segment), or a recorded set that disagrees
    /// with the segment and so is not trusted.
    Unknown,
}

impl Membership {
    /// What a segment holds when its shape alone decides it, without
    /// reading any part: an empty segment holds nothing, and one whose
    /// range has no gaps holds all of it. `None` for a segment with gaps.
    pub(crate) fn from_shape(min_doc_id: u64, max_doc_id: u64, doc_count: u64) -> Option<Self> {
        if doc_count == 0 {
            Some(Membership::Empty)
        } else if range_width(min_doc_id, max_doc_id) == Some(doc_count) {
            Some(Membership::Range)
        } else {
            None
        }
    }

    /// What a segment holds, from a doc-id set loaded for it.
    ///
    /// The set is trusted only if it matches the segment's recorded count
    /// and range: a wrong set would, for the writer, skip a real copy's
    /// deletion and leave a duplicate, and for the reader, mis-state its
    /// live documents. A mismatch or a read error is warned about and
    /// yields [`Membership::Unknown`]; a missing set yields it silently.
    pub(crate) fn from_loaded(
        segment_id: &str,
        min_doc_id: u64,
        max_doc_id: u64,
        doc_count: u64,
        loaded: Result<Option<RoaringTreemap>>,
    ) -> Self {
        match loaded {
            Ok(Some(ids))
                if ids.len() == doc_count
                    && ids.min() == Some(min_doc_id)
                    && ids.max() == Some(max_doc_id) =>
            {
                Membership::Set(Arc::new(ids))
            }
            Ok(Some(ids)) => {
                log::warn!(
                    "segment {segment_id} records {} doc ids in [{:?}, {:?}] but claims \
                     {doc_count} in [{min_doc_id}, {max_doc_id}]; treating the ids it holds as \
                     unknown",
                    ids.len(),
                    ids.min(),
                    ids.max()
                );
                Membership::Unknown
            }
            Ok(None) => Membership::Unknown,
            Err(e) => {
                log::warn!(
                    "cannot read the doc-id set of segment {segment_id}: {e}; treating the ids it \
                     holds as unknown"
                );
                Membership::Unknown
            }
        }
    }

    /// Whether the segment holds `doc_id` (which must be inside its range);
    /// `None` when that is unknown.
    pub(crate) fn holds(&self, doc_id: u64) -> Option<bool> {
        match self {
            Membership::Empty => Some(false),
            Membership::Range => Some(true),
            Membership::Set(ids) => Some(ids.contains(doc_id)),
            Membership::Unknown => None,
        }
    }

    /// How many of the documents the segment holds `bitmap` marks deleted
    /// (Issues #1211 / #1212).
    ///
    /// Exact unless the membership is unknown, where it is a lower bound:
    /// at most `width − doc_count` bits can sit on ids the segment does not
    /// hold, so at least `bits − gaps` deletions are real. The live count
    /// `doc_count − held_deletions` is then an upper bound that never
    /// under-counts.
    pub(crate) fn held_deletions(
        &self,
        bitmap: &DeletionBitmap,
        min_doc_id: u64,
        max_doc_id: u64,
        doc_count: u64,
    ) -> u64 {
        match self {
            Membership::Empty => 0,
            Membership::Range => bitmap.deleted_in_range(min_doc_id, max_doc_id),
            Membership::Set(ids) => bitmap.deleted_count_in(ids),
            Membership::Unknown => {
                let gaps = range_width(min_doc_id, max_doc_id)
                    .unwrap_or(u64::MAX)
                    .saturating_sub(doc_count);
                bitmap
                    .deleted_in_range(min_doc_id, max_doc_id)
                    .saturating_sub(gaps)
            }
        }
    }
}

/// Width of an id range, `max − min + 1`; `None` on overflow or an inverted
/// range.
pub(crate) fn range_width(min_doc_id: u64, max_doc_id: u64) -> Option<u64> {
    max_doc_id
        .checked_sub(min_doc_id)
        .and_then(|w| w.checked_add(1))
}

pub mod merge_engine;

// ---------------------------------------------------------------------------
// Merge data types (#1024).
//
// Extracted from the deleted `segment/manager.rs`: that file's
// `SegmentManager` and its binary `segments.manifest` ("SEGS") machinery
// were a complete parallel segment-management architecture that production
// never adopted — discovery and publication run on `segments.json` (#1021).
// These three types are the parts the live merge path genuinely uses.
// ---------------------------------------------------------------------------

/// Extended segment information with management metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManagedSegmentInfo {
    /// Core segment information.
    pub segment_info: SegmentInfo,

    /// Size of the segment in bytes.
    pub size_bytes: u64,

    /// Number of deleted documents in this segment.
    pub deleted_count: u64,

    /// Timestamp when segment was created.
    pub created_at: u64,

    /// Timestamp when segment was last modified.
    pub last_modified: u64,

    /// Merge tier (for tiered merge policy).
    pub tier: u8,

    /// Whether this segment is currently being merged.
    pub is_merging: bool,

    /// Segment file paths for cleanup.
    pub file_paths: Vec<String>,
}

impl ManagedSegmentInfo {
    /// Create new managed segment info.
    pub fn new(segment_info: SegmentInfo) -> Self {
        let now = crate::util::time::now_secs();

        ManagedSegmentInfo {
            segment_info,
            size_bytes: 0,
            deleted_count: 0,
            created_at: now,
            last_modified: now,
            tier: 0,
            is_merging: false,
            file_paths: Vec::new(),
        }
    }

    /// Get deletion ratio (deleted docs / total docs).
    pub fn deletion_ratio(&self) -> f64 {
        if self.segment_info.doc_count == 0 {
            0.0
        } else {
            self.deleted_count as f64 / self.segment_info.doc_count as f64
        }
    }

    /// Get effective document count (total - deleted).
    pub fn effective_doc_count(&self) -> u64 {
        self.segment_info
            .doc_count
            .saturating_sub(self.deleted_count)
    }

    /// Check if segment needs compaction.
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.deletion_ratio() > threshold
    }
}

/// Merge candidate representing segments to be merged.
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// Segments to merge.
    pub segments: Vec<String>,

    /// Priority score (higher = more urgent).
    pub priority: f64,

    /// Expected size after merge.
    pub estimated_size: u64,

    /// Merge strategy to use.
    pub strategy: MergeStrategy,
}

/// Merge strategy options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeStrategy {
    /// Size-based merging (small segments first).
    SizeBased,

    /// Deletion-based merging (high deletion ratio first).
    DeletionBased,

    /// Time-based merging (oldest segments first).
    TimeBased,

    /// Balanced approach considering multiple factors.
    Balanced,
}
