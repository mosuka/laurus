//! Merge policy for vector index segments.
//!
//! This module defines the strategy for selecting which segments to merge.

use std::fmt::Debug;

use crate::vector::index::segment::manager::{ManagedSegmentInfo, SegmentManagerConfig};

/// Trait for merge policies.
pub trait MergePolicy: Debug + Send + Sync {
    /// Select segments to merge.
    ///
    /// Returns a list of segment IDs to merge, or `None` if no merge is needed.
    fn candidates(
        &self,
        segments: &[ManagedSegmentInfo],
        config: &SegmentManagerConfig,
    ) -> Option<Vec<String>>;
}

/// Simple merge policy based on segment count and size.
#[derive(Debug, Default)]
pub struct SimpleMergePolicy;

impl SimpleMergePolicy {
    /// Create a new simple merge policy.
    pub fn new() -> Self {
        Self
    }
}

impl MergePolicy for SimpleMergePolicy {
    fn candidates(
        &self,
        segments: &[ManagedSegmentInfo],
        config: &SegmentManagerConfig,
    ) -> Option<Vec<String>> {
        // If we don't have enough segments, don't merge.
        // We trigger merge only when we exceed max_segments or soft limit?
        // Let's say we trigger if we have more than max_segments / 2?
        // Or strictly strictly max_segments?
        // Currently config.max_segments is the trigger threshold.

        if segments.len() < config.max_segments as usize {
            return None;
        }

        let merge_factor = config.merge_factor as usize;
        if segments.len() < merge_factor {
            return None;
        }

        // Generation-CONTIGUOUS window selection (Issue #880): the merged
        // segment inherits max(source generations), which is only correct
        // when no non-source segment's generation lies inside the sources'
        // generation range — a stale copy from an old source would otherwise
        // be laundered above that segment under newest-generation-wins
        // dedup. Picking the N globally-smallest segments (the previous
        // strategy) routinely produced non-adjacent sets, forcing the
        // caller's gap expansion to inflate a bounded merge into a
        // near-total rewrite. Instead, slide a window of `merge_factor`
        // generation-adjacent segments and pick the cheapest one.
        let mut by_generation: Vec<&ManagedSegmentInfo> = segments.iter().collect();
        by_generation.sort_by_key(|s| s.generation);

        let best = by_generation
            .windows(merge_factor)
            .min_by_key(|w| w.iter().map(|s| s.vector_count).sum::<u64>())?;

        Some(best.iter().map(|s| s.segment_id.clone()).collect())
    }
}

/// Tiered, metadata-driven merge policy (#883).
///
/// The classic Lucene/tantivy shape adapted to the newest-wins constraint
/// (#880): candidates must be **generation-contiguous**, so the policy
/// slides windows over the generation-sorted segment list and picks the
/// cheapest window whose members are *size-similar* — all within
/// [`Self::tier_ratio`] of the window's smallest member. Merging similar
/// sizes keeps the total write amplification logarithmic (each vector is
/// rewritten O(log N) times over its lifetime); merging a tiny segment into
/// a huge one repeatedly would be quadratic.
///
/// Triggers:
/// - **Tier full** — some window of `merge_factor` generation-adjacent,
///   size-similar segments exists (steady-state ingest: the newest tier
///   fills up every `merge_factor` commits and collapses upward).
/// - **Hard bound** — the segment count exceeds
///   [`SegmentManagerConfig::max_segments`]; falls back to the cheapest
///   contiguous window regardless of similarity, so growth stays bounded
///   even under adversarial size distributions.
///
/// Windows whose merged size would exceed
/// [`SegmentManagerConfig::max_vectors_per_segment`] are skipped (except
/// under the hard bound, where boundedness wins).
#[derive(Debug)]
pub struct TieredMergePolicy {
    /// Maximum allowed ratio between the largest and smallest member of a
    /// candidate window. Defaults to 8.0.
    pub tier_ratio: f64,
}

impl TieredMergePolicy {
    /// Create a tiered policy with the default tier ratio.
    pub fn new() -> Self {
        Self { tier_ratio: 8.0 }
    }
}

impl Default for TieredMergePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl MergePolicy for TieredMergePolicy {
    fn candidates(
        &self,
        segments: &[ManagedSegmentInfo],
        config: &SegmentManagerConfig,
    ) -> Option<Vec<String>> {
        let merge_factor = (config.merge_factor as usize).max(2);
        if segments.len() < merge_factor {
            return None;
        }

        let mut by_generation: Vec<&ManagedSegmentInfo> = segments.iter().collect();
        by_generation.sort_by_key(|s| s.generation);

        // Preferred: the cheapest size-similar window (a "full tier").
        let similar = by_generation
            .windows(merge_factor)
            .filter(|w| {
                let min = w.iter().map(|s| s.vector_count.max(1)).min().unwrap_or(1);
                let max = w.iter().map(|s| s.vector_count.max(1)).max().unwrap_or(1);
                max as f64 <= min as f64 * self.tier_ratio
            })
            .filter(|w| {
                w.iter().map(|s| s.vector_count).sum::<u64>() <= config.max_vectors_per_segment
            })
            .min_by_key(|w| w.iter().map(|s| s.vector_count).sum::<u64>());
        if let Some(window) = similar {
            return Some(window.iter().map(|s| s.segment_id.clone()).collect());
        }

        // Hard bound: over the segment cap, merge the cheapest contiguous
        // window regardless of similarity so the count stays bounded.
        if segments.len() > config.max_segments as usize {
            let best = by_generation
                .windows(merge_factor)
                .min_by_key(|w| w.iter().map(|s| s.vector_count).sum::<u64>())?;
            return Some(best.iter().map(|s| s.segment_id.clone()).collect());
        }

        None
    }
}

/// Policy that forces validation/merging of all segments.
#[derive(Debug, Default)]
pub struct ForceMergePolicy;

impl ForceMergePolicy {
    /// Create a new force merge policy.
    pub fn new() -> Self {
        Self
    }
}

impl MergePolicy for ForceMergePolicy {
    fn candidates(
        &self,
        segments: &[ManagedSegmentInfo],
        _config: &SegmentManagerConfig,
    ) -> Option<Vec<String>> {
        if segments.is_empty() {
            return None;
        }
        // Force merge all segments
        Some(segments.iter().map(|s| s.segment_id.clone()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::index::segment::manager::ManagedSegmentInfo;

    fn create_info(id: &str, count: u64) -> ManagedSegmentInfo {
        ManagedSegmentInfo {
            segment_id: id.to_string(),
            vector_count: count,
            vector_offset: 0,
            generation: 1,
            size_bytes: count * 100,
        }
    }

    #[test]
    fn test_simple_merge_policy_candidates() {
        let policy = SimpleMergePolicy::new();
        let config = SegmentManagerConfig {
            max_segments: 5,
            merge_factor: 3,
            ..Default::default()
        };

        // Case 1: Not enough segments
        let segments = vec![create_info("1", 100), create_info("2", 100)];
        assert!(policy.candidates(&segments, &config).is_none());

        // Case 2: Enough segments, trigger merge
        let segments = vec![
            create_info("1", 1000), // Large
            create_info("2", 100),  // Small
            create_info("3", 100),  // Small
            create_info("4", 100),  // Small
            create_info("5", 1000), // Large
            create_info("6", 1000), // Large
        ]; // Total 6 > max 5

        let candidates = policy.candidates(&segments, &config).unwrap();
        assert_eq!(candidates.len(), 3);
        // Should pick smallest: 2, 3, 4
        assert!(candidates.contains(&"2".to_string()));
        assert!(candidates.contains(&"3".to_string()));
        assert!(candidates.contains(&"4".to_string()));
    }
}
