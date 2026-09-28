//! Segment manager for vector indexes.
//!
//! This module manages vector index segments, including segment metadata,
//! merging strategies, and segment lifecycle.
//!
//! The segment registry is persisted in a `segments.json` manifest. Since
//! #634 (PR-1 / #879) the manifest is **versioned, checksummed, and written
//! atomically** (temp file + CRC-32 trailer + fsync + rename — the same
//! #784/#786 standard the segment data files follow), so it can serve as
//! the commit pivot for the segment-per-commit pipeline: a crash mid-save
//! leaves the previous manifest intact, and a corrupted manifest fails the
//! open loudly instead of silently starting empty.
//!
//! Originally HNSW-only, this manager is shared across index types since
//! Issue #889 via [`SegmentFileLayout`] — the only piece of its behavior
//! that varies by on-disk file suffix.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::error::Result;
use crate::storage::Storage;
use crate::storage::manifest as manifest_io;

use super::merge_policy::MergePolicy;

/// Manifest file name for the segment registry.
const MANIFEST_FILE: &str = "segments.json";

/// Temporary file the manifest is staged in before the atomic rename.
const MANIFEST_TMP_FILE: &str = "segments.json.tmp";

/// Current manifest format version (see [`SegmentManifest`]).
const MANIFEST_VERSION: u32 = 1;

/// Filename suffixes a segment's on-disk files use, parameterizing
/// [`SegmentManager`] across index types that share this manifest/merge
/// machinery but write different file extensions (Issue #889).
#[derive(Debug, Clone, Copy)]
pub struct SegmentFileLayout {
    /// Primary segment data file suffix, e.g. `.hnsw`.
    pub primary: &'static str,

    /// Sidecar file suffixes beyond the primary, e.g. `.hnsw.f32` for
    /// HNSW's f32 rerank sidecar. Counted by [`SegmentManager::add_segment`]
    /// size measurement and removed by
    /// [`SegmentManager::delete_segment_files`] alongside the primary file.
    pub sidecars: &'static [&'static str],

    /// Suffix for a segment's own atomic-write temp-staging file, e.g.
    /// `.hnsw.tmp`. Recognized (and swept if orphaned) by the segment
    /// manager's internal orphan sweep on open.
    pub tmp: &'static str,
}

impl SegmentFileLayout {
    /// All suffixes this layout recognizes for a segment's own files, plus
    /// the layout-independent `.delmap` suffix, ordered **longest first**.
    ///
    /// A shorter suffix that is itself the tail of a longer one (e.g.
    /// `.hnsw` is the tail of `.hnsw.tmp`) must be tried last, or the longer
    /// file gets mis-stemmed (`foo.hnsw.tmp` stripped of `.hnsw` leaves
    /// `foo` + a dangling `.tmp`, not the segment id `foo`).
    fn ordered_suffixes(&self) -> Vec<&'static str> {
        let mut suffixes: Vec<&'static str> = self.sidecars.to_vec();
        suffixes.push(self.primary);
        suffixes.push(self.tmp);
        suffixes.push(".delmap");
        suffixes.sort_by_key(|s| std::cmp::Reverse(s.len()));
        suffixes
    }
}

/// Versioned payload of the `segments.json` manifest (#634 PR-1 / #879).
///
/// Serialized as JSON and framed with a length prefix plus a CRC-32 trailer
/// (via `StructWriter`), so corruption is detected at load time. The legacy
/// format — a bare JSON array of [`ManagedSegmentInfo`] written in place —
/// is still readable for backward compatibility and is upgraded on the next
/// save.
#[derive(Debug, Serialize, Deserialize)]
struct SegmentManifest {
    /// Format version (currently [`MANIFEST_VERSION`]).
    version: u32,

    /// Next segment ordinal handed out by
    /// [`SegmentManager::generate_segment_id`], persisted so a reopen never
    /// re-issues an ID even if the highest-numbered segment was merged away.
    next_segment_id: u64,

    /// Last WAL sequence number whose effects are fully durable in this
    /// manifest's segments and the index deletion bitmap (#882).
    ///
    /// Published by the segmented index's `persist_deletions` — the last
    /// vector step of the store's commit ladder, before the engine
    /// truncates the WAL — so recovery can safely skip records at or below
    /// it. Intermediate manifest saves (segment seals) carry the previous
    /// value.
    last_wal_seq: u64,

    /// The registered segments.
    segments: Vec<ManagedSegmentInfo>,
}

/// Configuration for segment manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentManagerConfig {
    /// Maximum number of vectors per segment.
    pub max_vectors_per_segment: u64,

    /// Minimum number of vectors per segment before merging.
    pub min_vectors_per_segment: u64,

    /// Maximum number of segments before triggering merge.
    pub max_segments: u32,

    /// Merge factor (how many segments to merge at once).
    pub merge_factor: u32,
}

impl Default for SegmentManagerConfig {
    fn default() -> Self {
        Self {
            max_vectors_per_segment: 1000000,
            min_vectors_per_segment: 10000,
            max_segments: 100,
            merge_factor: 10,
        }
    }
}

/// Information about a managed segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedSegmentInfo {
    /// Segment identifier.
    pub segment_id: String,

    /// Number of vectors in this segment.
    pub vector_count: u64,

    /// Vector offset for this segment.
    pub vector_offset: u64,

    /// Generation number of this segment.
    pub generation: u64,

    /// Whether this segment has deletions.
    pub has_deletions: bool,

    /// Size of the segment in bytes.
    pub size_bytes: u64,
}

impl ManagedSegmentInfo {
    /// Create a new managed segment info.
    pub fn new(segment_id: String, vector_count: u64, vector_offset: u64, generation: u64) -> Self {
        Self {
            segment_id,
            vector_count,
            vector_offset,
            generation,
            has_deletions: false,
            size_bytes: 0,
        }
    }

    /// Check if this segment should be merged based on config.
    pub fn should_merge(&self, config: &SegmentManagerConfig) -> bool {
        self.vector_count < config.min_vectors_per_segment
    }
}

/// Candidate segments for merging.
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// Segments to merge.
    pub segments: Vec<ManagedSegmentInfo>,

    /// Total vector count.
    pub total_vectors: u64,

    /// Total size in bytes.
    pub total_size: u64,
}

/// Statistics about segment manager.
#[derive(Debug, Clone)]
pub struct SegmentManagerStats {
    /// Total number of segments.
    pub segment_count: u32,

    /// Total number of vectors across all segments.
    pub total_vectors: u64,

    /// Total size of all segments in bytes.
    pub total_size: u64,

    /// Number of segments with deletions.
    pub segments_with_deletions: u32,

    /// Average vectors per segment.
    pub avg_vectors_per_segment: f64,
}

/// Manages segments for vector indexes.
#[derive(Debug)]
pub struct SegmentManager {
    config: SegmentManagerConfig,
    layout: SegmentFileLayout,
    storage: Arc<dyn Storage>,
    segments: Arc<RwLock<Vec<ManagedSegmentInfo>>>,
    next_segment_id: Arc<RwLock<u64>>,
    /// Last WAL sequence number recorded in the manifest (see
    /// [`SegmentManifest::last_wal_seq`]). Persisted by the next save.
    last_wal_seq: AtomicU64,
}

impl SegmentManager {
    /// Create a new segment manager with the given configuration.
    ///
    /// Loads the `segments.json` manifest when present and sweeps orphaned
    /// segment files (a torn flush can leave a `segment_*` file on storage
    /// that never made it into the manifest — its documents are covered by
    /// WAL replay, so the file is garbage).
    ///
    /// # Errors
    ///
    /// Returns an error if an existing manifest fails to load or fails its
    /// CRC check — a corrupted registry must fail the open loudly rather
    /// than silently starting empty (which would orphan, and then sweep,
    /// every live segment).
    pub fn new(
        config: SegmentManagerConfig,
        storage: Arc<dyn Storage>,
        layout: SegmentFileLayout,
    ) -> Result<Self> {
        let manager = Self {
            config,
            layout,
            storage,
            segments: Arc::new(RwLock::new(Vec::new())),
            next_segment_id: Arc::new(RwLock::new(0)),
            last_wal_seq: AtomicU64::new(0),
        };

        manager.load_state()?;
        manager.cleanup_orphans();

        Ok(manager)
    }

    fn load_state(&self) -> Result<()> {
        // The versioned form is length-prefixed JSON with a CRC-32 trailer;
        // the legacy form is a bare pretty-printed JSON array written in
        // place, with no framing and no checksum. The loader tells them apart
        // and a checksum mismatch is a hard error rather than a silent empty
        // start (#1022).
        let Some((payload, format)) =
            manifest_io::load_checksummed(self.storage.as_ref(), MANIFEST_FILE, None)?
        else {
            return Ok(());
        };

        let (segments_info, next_id, wal_seq) = match format {
            manifest_io::ManifestFormat::Legacy => {
                // Pre-versioning manifests carried only the segment list; the
                // next save upgrades the file.
                let segments: Vec<ManagedSegmentInfo> = serde_json::from_slice(&payload)?;
                (segments, 0, 0)
            }
            manifest_io::ManifestFormat::Checksummed => {
                let manifest: SegmentManifest = serde_json::from_slice(&payload)?;
                (
                    manifest.segments,
                    manifest.next_segment_id,
                    manifest.last_wal_seq,
                )
            }
        };

        let mut segments = self.segments.write();
        *segments = segments_info;

        // Zero generations in a loaded manifest need stamping, or the
        // newest-generation-wins dedup ordering is meaningless (Issue #880).
        // Two populations exist:
        // - Legacy all-zero manifests (pre-#879 nothing stamped): list
        //   position is flush order, so stamp 1..N in list order.
        // - MIXED manifests (stamped flush segments + a zero-generation
        //   merged segment from the brief window where `apply_merge` did
        //   not stamp): the zero entry's true age is unrecoverable, so
        //   stamp it BELOW every stamped generation (shift the stamped ones
        //   up). Treating it as oldest is the loss-minimizing direction —
        //   stamping it newest would let a stale copy inside it claim the
        //   `(doc_id, field)` key at the next merge and physically drop the
        //   genuinely newer copy.
        let zero_count = segments.iter().filter(|s| s.generation == 0).count() as u64;
        if zero_count > 0 {
            let all_zero = zero_count == segments.len() as u64;
            if !all_zero {
                for segment in segments.iter_mut() {
                    if segment.generation != 0 {
                        segment.generation += zero_count;
                    }
                }
            }
            let mut next = 0u64;
            for segment in segments.iter_mut() {
                if segment.generation == 0 {
                    next += 1;
                    segment.generation = next;
                }
            }
        }

        // The persisted counter wins over the max-scan, but never goes
        // backwards: a merged-away highest segment must not cause ID reuse.
        let max_id = segments
            .iter()
            .filter_map(|s| s.segment_id.strip_prefix("segment_"))
            .filter_map(|s| s.parse::<u64>().ok())
            .map(|id| id + 1)
            .max()
            .unwrap_or(0);
        *self.next_segment_id.write() = next_id.max(max_id);
        self.last_wal_seq.store(wal_seq, Ordering::Release);

        Ok(())
    }

    /// Persist the manifest atomically: temp file + length-prefixed JSON +
    /// CRC-32 trailer + fsync + rename (#784/#786 parity).
    ///
    /// Takes the segment list as a parameter so mutators can call it while
    /// still holding the `segments` write lock — saving after dropping the
    /// lock let two racing mutators publish manifests out of order.
    fn save_state_locked(&self, segments: &[ManagedSegmentInfo]) -> Result<()> {
        let manifest = SegmentManifest {
            version: MANIFEST_VERSION,
            next_segment_id: *self.next_segment_id.read(),
            last_wal_seq: self.last_wal_seq.load(Ordering::Acquire),
            segments: segments.to_vec(),
        };
        // temp + CRC-32 trailer + rename + sync, shared with every other
        // control file in the tree (#1022).
        manifest_io::save_checksummed_json(self.storage.as_ref(), MANIFEST_FILE, None, &manifest)
    }

    /// Persist the current manifest state.
    ///
    /// # Errors
    ///
    /// Returns an error if serializing or writing the manifest fails; the
    /// previously published manifest stays intact (atomic rename).
    pub fn save_state(&self) -> Result<()> {
        let segments = self.segments.read();
        self.save_state_locked(&segments)
    }

    /// Record the last applied WAL sequence number; persisted by the next
    /// save (see [`SegmentManifest::last_wal_seq`]).
    pub fn set_last_wal_seq(&self, seq: u64) {
        self.last_wal_seq.store(seq, Ordering::Release);
    }

    /// Last WAL sequence number recorded in the manifest.
    pub fn last_wal_seq(&self) -> u64 {
        self.last_wal_seq.load(Ordering::Acquire)
    }

    /// Best-effort sweep of files that belong to no registered segment.
    ///
    /// Only touches names this manager itself generates (`segment_NNNNNN`
    /// plus its [`SegmentFileLayout`] suffixes) and the manifest temp file,
    /// so foreign files — e.g. a monolithic `vector_index.hnsw` in the same
    /// directory — are never affected. Runs only after a successful
    /// manifest load.
    fn cleanup_orphans(&self) {
        let Ok(files) = self.storage.list_files() else {
            return;
        };
        let segments = self.segments.read();
        let suffixes = self.layout.ordered_suffixes();
        for file in files {
            if file == MANIFEST_TMP_FILE {
                let _ = self.storage.delete_file(&file);
                continue;
            }
            // Strip a known suffix (longest first); skip files that are not
            // segment-shaped.
            let Some(stem) = suffixes.iter().find_map(|suffix| file.strip_suffix(suffix)) else {
                continue;
            };
            let Some(ordinal) = stem.strip_prefix("segment_") else {
                continue;
            };
            if ordinal.is_empty() || !ordinal.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            if !segments.iter().any(|s| s.segment_id == stem) {
                let _ = self.storage.delete_file(&file);
            }
        }
    }

    /// Add a new segment.
    ///
    /// Fills in bookkeeping metadata the caller did not provide (#879):
    /// a zero `size_bytes` is replaced by the on-storage size of the segment
    /// file plus its sidecars, and a zero `generation` is stamped with the
    /// next generation after the currently registered maximum — both feed
    /// the metadata-driven merge policy.
    ///
    /// # Errors
    ///
    /// Returns an error if persisting the manifest fails.
    pub fn add_segment(&self, mut info: ManagedSegmentInfo) -> Result<()> {
        if info.size_bytes == 0 {
            info.size_bytes = self.measure_segment_size(&info.segment_id);
        }
        let mut segments = self.segments.write();
        if info.generation == 0 {
            info.generation = segments.iter().map(|s| s.generation).max().unwrap_or(0) + 1;
        }
        segments.push(info);
        self.save_state_locked(&segments)
    }

    /// Sum the on-storage sizes of a segment's primary file and sidecars.
    ///
    /// `.delmap` is deliberately excluded (#882): per-segment deletion
    /// bitmaps do not exist — the bitmap is index-level — and for a
    /// legacy-migrated segment (whose id equals the index name) including
    /// it would misattribute the index bitmap's size to the segment.
    fn measure_segment_size(&self, segment_id: &str) -> u64 {
        std::iter::once(self.layout.primary)
            .chain(self.layout.sidecars.iter().copied())
            .map(|suffix| format!("{segment_id}{suffix}"))
            .filter_map(|name| self.storage.file_size(&name).ok())
            .sum()
    }

    /// Remove a segment.
    pub fn remove_segment(&self, segment_id: &str) -> Result<()> {
        let mut segments = self.segments.write();
        if let Some(pos) = segments.iter().position(|s| s.segment_id == segment_id) {
            segments.remove(pos);
            self.save_state_locked(&segments)
        } else {
            Ok(())
        }
    }

    /// Delete physical files associated with a segment, including its
    /// sidecars — leaving them behind orphans storage after every merge
    /// (#879).
    ///
    /// Deliberately does NOT touch `{segment_id}.delmap` (#882): the
    /// segmented index keeps ONE index-level deletion bitmap named
    /// `{index_name}.delmap` — per-segment bitmaps do not exist — and a
    /// legacy-migrated segment's id EQUALS the index name, so deleting the
    /// segment's "delmap" here would destroy live deletion state when a
    /// merge consumes that segment.
    pub fn delete_segment_files(&self, segment_id: &str) -> Result<()> {
        // Best effort deletion - ignore if a file doesn't exist
        for suffix in
            std::iter::once(self.layout.primary).chain(self.layout.sidecars.iter().copied())
        {
            let _ = self.storage.delete_file(&format!("{segment_id}{suffix}"));
        }
        Ok(())
    }

    /// Flag every registered segment as containing deletions (Issue #880).
    ///
    /// The vector-side [`ManagedSegmentInfo`] carries no per-segment doc-id
    /// range, so a delete that may target a sealed segment cannot be
    /// attributed to a specific one; flagging all of them is conservative
    /// but correct — the flag only feeds the merge policy's prioritization,
    /// while actual filtering always goes through the shared deletion
    /// bitmap. Saves the manifest once (not per segment). No-op when every
    /// segment is already flagged.
    ///
    /// # Errors
    ///
    /// Returns an error if persisting the manifest fails.
    pub fn mark_all_has_deletions(&self) -> Result<()> {
        let mut segments = self.segments.write();
        let mut changed = false;
        for segment in segments.iter_mut() {
            if !segment.has_deletions {
                segment.has_deletions = true;
                changed = true;
            }
        }
        if changed {
            self.save_state_locked(&segments)
        } else {
            Ok(())
        }
    }

    /// Update a segment info.
    pub fn update_segment(&self, info: ManagedSegmentInfo) -> Result<()> {
        let mut segments = self.segments.write();
        if let Some(idx) = segments
            .iter()
            .position(|s| s.segment_id == info.segment_id)
        {
            segments[idx] = info;
        }
        self.save_state_locked(&segments)
    }

    /// Get segment information.
    pub fn get_segment(&self, segment_id: &str) -> Option<ManagedSegmentInfo> {
        let segments = self.segments.read();
        segments
            .iter()
            .find(|s| s.segment_id == segment_id)
            .cloned()
    }

    /// List all segments.
    pub fn list_segments(&self) -> Vec<ManagedSegmentInfo> {
        let segments = self.segments.read();
        segments.clone()
    }

    /// Check if any segments need merging.
    pub fn check_merge(&self, policy: &dyn MergePolicy) -> Option<MergeCandidate> {
        let segments_lock = self.segments.read();

        if let Some(candidate_ids) = policy.candidates(&segments_lock, &self.config) {
            let mut total_vectors = 0;
            let mut total_size = 0;
            let mut candidates = Vec::new();

            for id in &candidate_ids {
                if let Some(segment) = segments_lock.iter().find(|s| s.segment_id == *id) {
                    total_vectors += segment.vector_count;
                    total_size += segment.size_bytes;
                    candidates.push(segment.clone());
                }
            }

            return Some(MergeCandidate {
                segments: candidates,
                total_vectors,
                total_size,
            });
        }
        None
    }

    /// Apply a merge result by replacing source segments with the merged segment.
    pub fn apply_merge(
        &self,
        candidate: MergeCandidate,
        mut merged_segment: ManagedSegmentInfo,
    ) -> Result<()> {
        // Replace the engine's estimated size with the on-storage size of
        // the merged segment file + sidecars (mirrors `add_segment`, #879).
        merged_segment.size_bytes = self.measure_segment_size(&merged_segment.segment_id);
        let mut segments_lock = self.segments.write();

        // 1. Remove source segments
        let ids_to_remove: std::collections::HashSet<_> =
            candidate.segments.iter().map(|s| &s.segment_id).collect();

        segments_lock.retain(|s| !ids_to_remove.contains(&s.segment_id));

        // 2. Add new segment
        segments_lock.push(merged_segment);

        // 3. Save state (lock held — see `save_state_locked`)
        self.save_state_locked(&segments_lock)?;
        drop(segments_lock);

        // 4. Cleanup physical files of source segments
        for segment in candidate.segments {
            self.delete_segment_files(&segment.segment_id)?;
        }

        Ok(())
    }
    pub fn total_vectors(&self) -> u64 {
        self.segments.read().iter().map(|s| s.vector_count).sum()
    }

    /// Generate a new segment ID.
    pub fn generate_segment_id(&self) -> String {
        let mut next_id = self.next_segment_id.write();
        let id = *next_id;
        *next_id += 1;
        format!("segment_{:06}", id)
    }

    /// Check if merging is needed.
    pub fn needs_merge(&self) -> bool {
        let segments = self.segments.read();
        segments.len() as u32 > self.config.max_segments
    }

    /// Get statistics.
    pub fn stats(&self) -> SegmentManagerStats {
        let segments = self.segments.read();
        let segment_count = segments.len() as u32;
        let total_vectors: u64 = segments.iter().map(|s| s.vector_count).sum();
        let total_size: u64 = segments.iter().map(|s| s.size_bytes).sum();
        let segments_with_deletions = segments.iter().filter(|s| s.has_deletions).count() as u32;
        let avg_vectors_per_segment = if segment_count > 0 {
            total_vectors as f64 / segment_count as f64
        } else {
            0.0
        };

        SegmentManagerStats {
            segment_count,
            total_vectors,
            total_size,
            segments_with_deletions,
            avg_vectors_per_segment,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::index::segment::merge_policy::SimpleMergePolicy;

    /// The layout exercised by every test below except the
    /// layout-parameterization test itself: mirrors HNSW's real on-disk
    /// suffixes, matching these tests' `.hnsw`-shaped fixture names.
    const TEST_LAYOUT: SegmentFileLayout = SegmentFileLayout {
        primary: ".hnsw",
        sidecars: &[".hnsw.f32"],
        tmp: ".hnsw.tmp",
    };

    fn create_info(id: &str, count: u64) -> ManagedSegmentInfo {
        ManagedSegmentInfo {
            segment_id: id.to_string(),
            vector_count: count,
            vector_offset: 0,
            generation: 1,
            has_deletions: false,
            size_bytes: count * 100,
        }
    }

    #[test]
    fn test_segment_manager_basic() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = SegmentManager::new(config, storage, TEST_LAYOUT).unwrap();

        let segment_id = manager.generate_segment_id();
        assert_eq!(segment_id, "segment_000000");

        let info = ManagedSegmentInfo::new(segment_id.clone(), 1000, 0, 0);
        manager.add_segment(info.clone()).unwrap();

        let retrieved = manager.get_segment(&segment_id).unwrap();
        assert_eq!(retrieved.vector_count, 1000);
    }

    // Additional tests for persistence?
    #[test]
    fn test_persistence() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager =
                SegmentManager::new(config.clone(), storage.clone(), TEST_LAYOUT).unwrap();
            let info = ManagedSegmentInfo::new("segment_000000".to_string(), 1000, 0, 0);
            manager.add_segment(info).unwrap();
            // Saves automatically
        }

        // Reload
        {
            let manager = SegmentManager::new(config, storage.clone(), TEST_LAYOUT).unwrap();
            let segments = manager.list_segments();
            assert_eq!(segments.len(), 1);
            assert_eq!(segments[0].segment_id, "segment_000000");
        }
    }

    /// #879: the versioned manifest round-trips segments, the ID counter,
    /// and the WAL checkpoint through save + reload.
    #[test]
    fn test_versioned_manifest_round_trip() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager =
                SegmentManager::new(config.clone(), storage.clone(), TEST_LAYOUT).unwrap();
            let id0 = manager.generate_segment_id();
            let id1 = manager.generate_segment_id();
            manager
                .add_segment(ManagedSegmentInfo::new(id0, 100, 0, 0))
                .unwrap();
            manager
                .add_segment(ManagedSegmentInfo::new(id1.clone(), 200, 100, 0))
                .unwrap();
            manager.set_last_wal_seq(42);
            // Remove the highest-numbered segment: the persisted counter must
            // still prevent its ID from being reused after reload.
            manager.remove_segment(&id1).unwrap();
            manager.save_state().unwrap();
        }

        {
            let manager = SegmentManager::new(config, storage, TEST_LAYOUT).unwrap();
            let segments = manager.list_segments();
            assert_eq!(segments.len(), 1);
            assert_eq!(segments[0].segment_id, "segment_000000");
            assert_eq!(manager.last_wal_seq(), 42, "WAL checkpoint must persist");
            assert_eq!(
                manager.generate_segment_id(),
                "segment_000002",
                "the persisted counter must win over the max-scan (no ID reuse)"
            );
        }
    }

    /// #879: the legacy bare-JSON-array manifest still loads and is upgraded
    /// to the versioned format by the next save.
    #[test]
    fn test_legacy_manifest_loads_and_upgrades() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Hand-write the pre-#879 format: a bare pretty JSON array.
        let legacy = serde_json::to_vec_pretty(&vec![create_info("segment_000007", 100)]).unwrap();
        {
            let mut out = storage.create_output("segments.json").unwrap();
            std::io::Write::write_all(&mut out, &legacy).unwrap();
            out.close().unwrap();
        }

        let manager = SegmentManager::new(config.clone(), storage.clone(), TEST_LAYOUT).unwrap();
        assert_eq!(manager.list_segments().len(), 1);
        assert_eq!(
            manager.generate_segment_id(),
            "segment_000008",
            "legacy load must derive the counter from the max scan"
        );

        // A save upgrades the file; a reload parses the versioned format.
        manager.save_state().unwrap();
        let manager = SegmentManager::new(config, storage, TEST_LAYOUT).unwrap();
        assert_eq!(manager.list_segments().len(), 1);
        assert_eq!(manager.list_segments()[0].segment_id, "segment_000007");
    }

    /// #879: a corrupted manifest fails the open loudly — silently starting
    /// empty would orphan (and then sweep) every live segment.
    #[test]
    fn test_corrupted_manifest_fails_open() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager =
                SegmentManager::new(config.clone(), storage.clone(), TEST_LAYOUT).unwrap();
            manager
                .add_segment(create_info("segment_000000", 100))
                .unwrap();
        }

        // Flip one byte inside the JSON payload (skip the length prefix).
        let mut content = Vec::new();
        {
            let mut input = storage.open_input("segments.json").unwrap();
            input.read_to_end(&mut content).unwrap();
        }
        let mid = content.len() / 2;
        content[mid] ^= 0xFF;
        {
            let mut out = storage.create_output("segments.json").unwrap();
            std::io::Write::write_all(&mut out, &content).unwrap();
            out.close().unwrap();
        }

        assert!(
            SegmentManager::new(config, storage, TEST_LAYOUT).is_err(),
            "a corrupted manifest must fail the open, not silently start empty"
        );
    }

    /// #880: legacy manifests carry all-zero generations; the load stamps
    /// them in list order (flush order = age) so newest-generation-wins
    /// dedup has a meaningful ordering.
    #[test]
    fn test_legacy_zero_generations_are_stamped_in_list_order() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Hand-write the pre-#879 format with generation 0 everywhere.
        let mut a = create_info("segment_000000", 100);
        a.generation = 0;
        let mut b = create_info("segment_000001", 100);
        b.generation = 0;
        let legacy = serde_json::to_vec_pretty(&vec![a, b]).unwrap();
        {
            let mut out = storage.create_output("segments.json").unwrap();
            std::io::Write::write_all(&mut out, &legacy).unwrap();
            out.close().unwrap();
        }

        let manager = SegmentManager::new(config, storage, TEST_LAYOUT).unwrap();
        let segments = manager.list_segments();
        assert_eq!(segments[0].generation, 1, "stamped in list order");
        assert_eq!(segments[1].generation, 2, "monotone with position");
    }

    /// #880: a MIXED manifest (stamped flush segments + a zero-generation
    /// merged segment from the pre-#880 `apply_merge`) stamps the zero
    /// entry as the OLDEST — its true age is unrecoverable, and treating it
    /// as newest would let a stale copy inside it physically drop a newer
    /// copy at the next merge.
    #[test]
    fn test_mixed_manifest_zero_generation_stamped_oldest() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut flushed = create_info("segment_000002", 100);
        flushed.generation = 5;
        let mut merged = create_info("segment_000003", 100);
        merged.generation = 0;
        let legacy = serde_json::to_vec_pretty(&vec![flushed, merged]).unwrap();
        {
            let mut out = storage.create_output("segments.json").unwrap();
            std::io::Write::write_all(&mut out, &legacy).unwrap();
            out.close().unwrap();
        }

        let manager = SegmentManager::new(config, storage, TEST_LAYOUT).unwrap();
        let segments = manager.list_segments();
        let gen_of = |id: &str| {
            segments
                .iter()
                .find(|s| s.segment_id == id)
                .unwrap()
                .generation
        };
        assert!(
            gen_of("segment_000003") < gen_of("segment_000002"),
            "the zero-generation merged segment must be stamped OLDEST, got {:?}",
            segments
                .iter()
                .map(|s| (s.segment_id.clone(), s.generation))
                .collect::<Vec<_>>()
        );
    }

    /// #879: the CRC verify itself is load-bearing — corrupting only the
    /// trailer (payload still valid JSON) must fail the open.
    #[test]
    fn test_crc_trailer_mismatch_fails_open() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager =
                SegmentManager::new(config.clone(), storage.clone(), TEST_LAYOUT).unwrap();
            manager
                .add_segment(create_info("segment_000000", 100))
                .unwrap();
        }

        let mut content = Vec::new();
        {
            let mut input = storage.open_input("segments.json").unwrap();
            input.read_to_end(&mut content).unwrap();
        }
        let rewrite = |bytes: &[u8]| {
            let mut out = storage.create_output("segments.json").unwrap();
            std::io::Write::write_all(&mut out, bytes).unwrap();
            out.close().unwrap();
        };

        // Flip a bit in the footer's CRC only (the u32 before the 4-byte
        // magic) — the JSON payload stays intact, so only the checksum
        // comparison can catch this.
        let mut crc_flipped = content.clone();
        let crc_at = crc_flipped.len() - 8;
        crc_flipped[crc_at] ^= 0x01;
        rewrite(&crc_flipped);
        let err = SegmentManager::new(config.clone(), storage.clone(), TEST_LAYOUT).unwrap_err();
        assert!(
            err.to_string().contains("checksum mismatch"),
            "the CRC verify must reject a trailer-only corruption, got: {err}"
        );

        // A damaged magic hides the framing, so the file no longer reads as
        // framed at all — but it is still refused, never loaded.
        let mut magic_flipped = content;
        let last = magic_flipped.len() - 1;
        magic_flipped[last] ^= 0x01;
        rewrite(&magic_flipped);
        assert!(SegmentManager::new(config, storage.clone(), TEST_LAYOUT).is_err());
    }

    /// #879: a torn manifest write (garbage left in the staging file) must
    /// not affect the published manifest; the temp file is swept on open.
    #[test]
    fn test_torn_manifest_write_survives() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        {
            let manager =
                SegmentManager::new(config.clone(), storage.clone(), TEST_LAYOUT).unwrap();
            manager
                .add_segment(create_info("segment_000000", 100))
                .unwrap();
        }
        // Simulate a crash mid-save: garbage staged, rename never happened.
        {
            let mut out = storage.create_output("segments.json.tmp").unwrap();
            std::io::Write::write_all(&mut out, b"\x00garbage").unwrap();
            out.close().unwrap();
        }

        let manager = SegmentManager::new(config, storage.clone(), TEST_LAYOUT).unwrap();
        assert_eq!(
            manager.list_segments().len(),
            1,
            "published manifest intact"
        );
        assert!(
            !storage.file_exists("segments.json.tmp"),
            "the torn staging file must be swept on open"
        );
    }

    /// #879: files belonging to no registered segment are swept on open;
    /// foreign files (e.g. a monolithic index in the same directory) are
    /// never touched.
    #[test]
    fn test_orphan_sweep_spares_foreign_files() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let write = |name: &str| {
            let mut out = storage.create_output(name).unwrap();
            std::io::Write::write_all(&mut out, b"x").unwrap();
            out.close().unwrap();
        };

        {
            let manager =
                SegmentManager::new(config.clone(), storage.clone(), TEST_LAYOUT).unwrap();
            manager
                .add_segment(create_info("segment_000000", 100))
                .unwrap();
        }
        // Registered segment's files: must survive.
        write("segment_000000.hnsw");
        write("segment_000000.hnsw.f32");
        // Orphans from a torn flush: must be swept.
        write("segment_000042.hnsw");
        write("segment_000042.hnsw.f32");
        write("segment_000042.delmap");
        write("segment_000042.hnsw.tmp");
        // Foreign files: must never be touched.
        write("vector_index.hnsw");
        write("not_a_segment.hnsw");

        let _manager = SegmentManager::new(config, storage.clone(), TEST_LAYOUT).unwrap();
        assert!(storage.file_exists("segment_000000.hnsw"));
        assert!(storage.file_exists("segment_000000.hnsw.f32"));
        assert!(!storage.file_exists("segment_000042.hnsw"), "orphan swept");
        assert!(!storage.file_exists("segment_000042.hnsw.f32"));
        assert!(!storage.file_exists("segment_000042.delmap"));
        assert!(!storage.file_exists("segment_000042.hnsw.tmp"));
        assert!(storage.file_exists("vector_index.hnsw"), "foreign spared");
        assert!(storage.file_exists("not_a_segment.hnsw"), "foreign spared");
    }

    /// #889: a distinct layout (Flat-shaped: no sidecars, a different
    /// primary/tmp suffix) sweeps its own orphans and spares foreign/HNSW
    /// files — proving the layout parameterization, not just the HNSW
    /// default, actually drives the sweep.
    #[test]
    fn test_cleanup_orphans_respects_a_different_layout() {
        const FLAT_LAYOUT: SegmentFileLayout = SegmentFileLayout {
            primary: ".flat",
            sidecars: &[],
            tmp: ".flat.tmp",
        };
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let write = |name: &str| {
            let mut out = storage.create_output(name).unwrap();
            std::io::Write::write_all(&mut out, b"x").unwrap();
            out.close().unwrap();
        };

        {
            let manager =
                SegmentManager::new(config.clone(), storage.clone(), FLAT_LAYOUT).unwrap();
            manager
                .add_segment(create_info("segment_000000", 100))
                .unwrap();
        }
        // Registered segment's file: must survive.
        write("segment_000000.flat");
        // An orphaned Flat segment (a torn flush): must be swept.
        write("segment_000042.flat");
        write("segment_000042.flat.tmp");
        // Foreign / other-index-type files: must never be touched, even
        // though they share the `segment_NNNNNN` numbering shape.
        write("segment_000099.hnsw");
        write("vector_index.flat");

        let _manager = SegmentManager::new(config, storage.clone(), FLAT_LAYOUT).unwrap();
        assert!(storage.file_exists("segment_000000.flat"));
        assert!(!storage.file_exists("segment_000042.flat"), "orphan swept");
        assert!(!storage.file_exists("segment_000042.flat.tmp"));
        assert!(
            storage.file_exists("segment_000099.hnsw"),
            "a different index type's segment file must never be swept by this layout"
        );
        assert!(storage.file_exists("vector_index.flat"), "foreign spared");
    }

    /// #879/#882: deleting a segment's files removes its sidecars along
    /// with the primary file — but never a `.delmap`: the deletion bitmap
    /// is index-level, and a legacy-migrated segment shares the index's
    /// name (#882), so touching it would destroy live deletion state.
    #[test]
    fn test_delete_segment_files_includes_sidecars() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = SegmentManager::new(config, storage.clone(), TEST_LAYOUT).unwrap();

        for name in [
            "segment_000001.hnsw",
            "segment_000001.hnsw.f32",
            "segment_000001.delmap",
        ] {
            let mut out = storage.create_output(name).unwrap();
            std::io::Write::write_all(&mut out, b"x").unwrap();
            out.close().unwrap();
        }

        manager.delete_segment_files("segment_000001").unwrap();
        assert!(!storage.file_exists("segment_000001.hnsw"));
        assert!(
            !storage.file_exists("segment_000001.hnsw.f32"),
            "sidecar GC"
        );
        assert!(
            storage.file_exists("segment_000001.delmap"),
            "a .delmap must never be deleted with segment files — the \
             deletion bitmap is index-level (#882)"
        );
    }

    /// #879: `add_segment` stamps a zero generation with max+1 and measures a
    /// zero `size_bytes` from storage — the merge policy's inputs are real.
    #[test]
    fn test_add_segment_fills_metadata() {
        let config = SegmentManagerConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = SegmentManager::new(config, storage.clone(), TEST_LAYOUT).unwrap();

        {
            let mut out = storage.create_output("segment_000000.hnsw").unwrap();
            std::io::Write::write_all(&mut out, &[0u8; 128]).unwrap();
            out.close().unwrap();
        }

        manager
            .add_segment(ManagedSegmentInfo::new(
                "segment_000000".to_string(),
                10,
                0,
                0,
            ))
            .unwrap();
        manager
            .add_segment(ManagedSegmentInfo::new(
                "segment_000001".to_string(),
                10,
                10,
                0,
            ))
            .unwrap();

        let segments = manager.list_segments();
        assert_eq!(segments[0].generation, 1, "zero generation stamped");
        assert_eq!(segments[1].generation, 2, "monotonically increasing");
        assert_eq!(
            segments[0].size_bytes, 128,
            "zero size_bytes measured from storage"
        );
    }

    #[test]
    fn test_check_merge() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = SegmentManagerConfig {
            max_segments: 5,
            merge_factor: 3,
            ..Default::default()
        };

        // We use a temporary config for the manager
        let manager = SegmentManager::new(config, storage, TEST_LAYOUT).unwrap();

        // 1. Add segments (not enough for merge)
        manager.add_segment(create_info("1", 100)).unwrap();
        manager.add_segment(create_info("2", 100)).unwrap();

        assert!(manager.check_merge(&SimpleMergePolicy::new()).is_none());

        // 2. Add more segments to trigger merge
        manager.add_segment(create_info("3", 100)).unwrap();
        manager.add_segment(create_info("4", 100)).unwrap();
        manager.add_segment(create_info("5", 100)).unwrap();
        manager.add_segment(create_info("6", 100)).unwrap(); // Total 6 > 5

        let candidate = manager.check_merge(&SimpleMergePolicy::new());
        assert!(candidate.is_some());

        let candidate = candidate.unwrap();
        assert_eq!(candidate.segments.len(), 3);
        // Expect smallest: 1, 2, 3, 4, 5, 6 are all 100?
        // Wait, simple policy sort by vector_count.
        // If all equal, it picks stable sort order? Or arbitrary.
        // SimpleMergePolicy uses `segments.iter().enumerate()` then sort_by_key.
        // `sort_by_key` is stable. So it picks first 3: 1, 2, 3.

        let ids: Vec<String> = candidate
            .segments
            .iter()
            .map(|s| s.segment_id.clone())
            .collect();
        assert!(ids.contains(&"1".to_string()));
        assert!(ids.contains(&"2".to_string()));
        assert!(ids.contains(&"3".to_string()));
    }
}
