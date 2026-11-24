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
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::storage::manifest as manifest_io;

use super::field_record::FieldRecord;
use super::kind::{kind_of_layout, layouts_read_with};
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
    /// A sidecar is staged through its suffix plus `.tmp` (e.g.
    /// `.hnsw.f32.tmp`), which the orphan sweep on open also reclaims
    /// (Issue #1532).
    pub sidecars: &'static [&'static str],

    /// Suffix for a segment's own atomic-write temp-staging file, e.g.
    /// `.hnsw.tmp`. Recognized (and swept if orphaned) by the segment
    /// manager's internal orphan sweep on open.
    pub tmp: &'static str,
}

impl SegmentFileLayout {
    /// The suffixes of the files a segment of this layout consists of: its
    /// primary file and its sidecars.
    fn file_suffixes(&self) -> impl Iterator<Item = &'static str> {
        std::iter::once(self.primary).chain(self.sidecars.iter().copied())
    }

    /// All suffixes this layout recognizes for a segment's own files.
    ///
    /// They include each sidecar's staging file, the sidecar suffix plus
    /// `.tmp` (e.g. `.hnsw.f32.tmp`), which a write that fails or crashes
    /// before its rename leaves (Issue #1532).
    fn own_suffixes(&self) -> impl Iterator<Item = String> {
        let sidecars = self.sidecars.iter();
        sidecars
            .flat_map(|sidecar| [sidecar.to_string(), format!("{sidecar}.tmp")])
            .chain([self.primary.to_string(), self.tmp.to_string()])
    }
}

/// The suffixes the files of a segment of any of `layouts` can have, each
/// with the kind of its layout, plus the layout-independent `.delmap`
/// suffix, ordered **longest first**.
///
/// A shorter suffix that is itself the tail of a longer one (e.g. `.hnsw`
/// is the tail of `.hnsw.tmp`) must be tried last, or the longer file gets
/// mis-stemmed (`foo.hnsw.tmp` stripped of `.hnsw` leaves `foo` + a dangling
/// `.tmp`, not the segment id `foo`).
fn ordered_suffixes(layouts: &[SegmentFileLayout]) -> Vec<(String, Option<&'static str>)> {
    let mut suffixes: Vec<(String, Option<&'static str>)> = layouts
        .iter()
        .flat_map(|layout| {
            let kind = kind_of_layout(layout);
            layout
                .own_suffixes()
                .map(move |suffix| (suffix, Some(kind)))
        })
        .collect();
    suffixes.push((".delmap".to_string(), None));
    suffixes.sort_by_key(|(suffix, _)| std::cmp::Reverse(suffix.len()));
    suffixes
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

    /// The kind, dimension and distance the segments were written with
    /// (Issue #1547), checked when the field opens. Absent in a manifest
    /// written before, or saved again by a build that does not know the
    /// key, and kept as raw JSON so that a value this build cannot read
    /// counts as absent instead of failing the load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    field: Option<serde_json::Value>,
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

    /// Size of the segment in bytes.
    pub size_bytes: u64,

    /// The index kind the segment was written as, as the suffix of its
    /// primary file without the dot: `flat`, `hnsw`, `ivf` or `mv` (Issue
    /// #1579).
    ///
    /// `None` until a [`SegmentManager`] takes the segment, which then is of
    /// the manager's own kind. Absent in a manifest written before, whose
    /// segments are given one when it is loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

impl ManagedSegmentInfo {
    /// Create a new managed segment info.
    pub fn new(segment_id: String, vector_count: u64, vector_offset: u64, generation: u64) -> Self {
        Self {
            segment_id,
            vector_count,
            vector_offset,
            generation,
            size_bytes: 0,
            kind: None,
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

    /// Average vectors per segment.
    pub avg_vectors_per_segment: f64,
}

/// Manages segments for vector indexes.
#[derive(Debug)]
pub struct SegmentManager {
    config: SegmentManagerConfig,
    /// The layout of every kind whose segments the index can hold (Issue
    /// #1579), the layout of its own kind first.
    layouts: Vec<SegmentFileLayout>,
    storage: Arc<dyn Storage>,
    segments: Arc<RwLock<Vec<ManagedSegmentInfo>>>,
    next_segment_id: Arc<RwLock<u64>>,
    /// Last WAL sequence number recorded in the manifest (see
    /// [`SegmentManifest::last_wal_seq`]). Persisted by the next save.
    last_wal_seq: AtomicU64,
    /// What the segments were written with (see
    /// [`SegmentManifest::field`]). Persisted by the next save.
    field_record: RwLock<Option<FieldRecord>>,
    /// Orders reader builds before the deletion of the segment files they
    /// listed (Issue #1678): a build holds it for reading from listing the
    /// segments until it has loaded their readers (see
    /// [`Self::pinned_segments`]), and [`Self::delete_segment_files`] holds
    /// it for writing.
    build_gate: RwLock<()>,
}

/// The segments a reader build reads, listed by
/// [`SegmentManager::pinned_segments`], with their files kept on storage
/// while it lives (Issue #1678).
///
/// A merge can publish the segment that replaces some of them meanwhile,
/// but [`SegmentManager::delete_segment_files`] waits until every
/// `PinnedSegments` taken before has been dropped. A build holds it until
/// it has loaded the readers of the listed segments.
pub(crate) struct PinnedSegments<'a> {
    segments: Vec<ManagedSegmentInfo>,
    _gate: parking_lot::RwLockReadGuard<'a, ()>,
}

impl std::ops::Deref for PinnedSegments<'_> {
    type Target = [ManagedSegmentInfo];

    fn deref(&self) -> &[ManagedSegmentInfo] {
        &self.segments
    }
}

/// Lets a build order the listed segments in place (newest generation
/// first, say) instead of copying them.
impl std::ops::DerefMut for PinnedSegments<'_> {
    fn deref_mut(&mut self) -> &mut [ManagedSegmentInfo] {
        &mut self.segments
    }
}

impl SegmentManager {
    /// Create a new segment manager with the given configuration.
    ///
    /// Loads the `segments.json` manifest when present and sweeps orphaned
    /// segment files (a torn flush can leave a `segment_*` file on storage
    /// that never made it into the manifest — its documents are covered by
    /// WAL replay, so the file is garbage).
    ///
    /// `layout` is the one of the index's own kind, which the segments the
    /// manager takes are of. A Flat, HNSW or IVF index can also hold segments
    /// of the other two kinds (Issue #1579), and the sweep covers the files
    /// of all three.
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
            layouts: layouts_read_with(&layout),
            storage,
            segments: Arc::new(RwLock::new(Vec::new())),
            next_segment_id: Arc::new(RwLock::new(0)),
            last_wal_seq: AtomicU64::new(0),
            field_record: RwLock::new(None),
            build_gate: RwLock::new(()),
        };

        let inferred = manager.load_state()?;
        manager.cleanup_orphans(&inferred);

        Ok(manager)
    }

    /// The name of the index's own kind, which the segments it writes are
    /// of.
    fn own_kind(&self) -> &'static str {
        kind_of_layout(&self.layouts[0])
    }

    /// The name of the kind `segment` was written as, which is the index's
    /// own when it does not say.
    pub(crate) fn kind_of<'a>(&self, segment: &'a ManagedSegmentInfo) -> &'a str {
        segment.kind.as_deref().unwrap_or(self.own_kind())
    }

    /// How many of the segments were written as another kind than the
    /// index's own (Issue #1579).
    pub(crate) fn segments_of_other_kinds(&self) -> usize {
        let own = self.own_kind();
        self.segments
            .read()
            .iter()
            .filter(|segment| self.kind_of(segment) != own)
            .count()
    }

    /// The kind of segment `segment_id`, listed by a manifest that records
    /// none for it (written before Issue #1579, or saved again by a build
    /// that does not know the key): the first of the kind `recorded` for the
    /// field, the index's own and the others it can hold whose primary file
    /// exists.
    fn infer_kind(&self, segment_id: &str, recorded: Option<&str>) -> String {
        let preferred = recorded.unwrap_or(self.own_kind());
        std::iter::once(preferred)
            .chain(
                self.layouts
                    .iter()
                    .map(|layout| -> &str { kind_of_layout(layout) }),
            )
            .find(|kind| self.storage.file_exists(&format!("{segment_id}.{kind}")))
            .unwrap_or(preferred)
            .to_string()
    }

    /// Load the manifest, when there is one.
    ///
    /// # Returns
    ///
    /// The ids of the segments whose kind it does not record, which
    /// [`Self::infer_kind`] gave them.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest cannot be read.
    fn load_state(&self) -> Result<HashSet<String>> {
        // The versioned form is length-prefixed JSON with a CRC-32 trailer;
        // the legacy form is a bare pretty-printed JSON array written in
        // place, with no framing and no checksum. The loader tells them apart
        // and a checksum mismatch is a hard error rather than a silent empty
        // start (#1022).
        let Some((payload, format)) =
            manifest_io::load_checksummed(self.storage.as_ref(), MANIFEST_FILE, None)?
        else {
            return Ok(HashSet::new());
        };

        let (segments_info, next_id, wal_seq, field) = match format {
            manifest_io::ManifestFormat::Legacy => {
                // Pre-versioning manifests carried only the segment list; the
                // next save upgrades the file.
                let segments: Vec<ManagedSegmentInfo> = serde_json::from_slice(&payload)?;
                (segments, 0, 0, None)
            }
            manifest_io::ManifestFormat::Checksummed => {
                let manifest: SegmentManifest = serde_json::from_slice(&payload)?;
                (
                    manifest.segments,
                    manifest.next_segment_id,
                    manifest.last_wal_seq,
                    manifest.field,
                )
            }
        };
        *self.field_record.write() = field.and_then(|value| {
            serde_json::from_value(value)
                .inspect_err(|e| {
                    log::warn!("ignoring the unreadable field record of '{MANIFEST_FILE}': {e}")
                })
                .ok()
        });

        let mut segments = self.segments.write();
        *segments = segments_info;

        let mut inferred = HashSet::new();
        let recorded = self.field_record.read().as_ref().map(|r| r.kind.clone());
        for segment in segments.iter_mut().filter(|s| s.kind.is_none()) {
            segment.kind = Some(self.infer_kind(&segment.segment_id, recorded.as_deref()));
            inferred.insert(segment.segment_id.clone());
        }

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

        Ok(inferred)
    }

    /// Persist the manifest atomically: temp file + length-prefixed JSON +
    /// CRC-32 trailer + fsync + rename (#784/#786 parity).
    ///
    /// Takes the segment list as a parameter so mutators can call it while
    /// still holding the `segments` write lock — saving after dropping the
    /// lock let two racing mutators publish manifests out of order.
    fn save_state_locked(&self, segments: &[ManagedSegmentInfo]) -> Result<()> {
        self.save_state_until_published(segments)?
            .map_or(Ok(()), Err)
    }

    /// Save `next` as the manifest and make it the in-memory list, only
    /// once the save fully succeeds (Issue #1583).
    ///
    /// The in-memory list mirrors the last manifest saved in full, as the
    /// lexical manifest's does: a failed save leaves it unchanged, so a
    /// caller that retries the mutation (a commit retried from a retained
    /// writer) applies it once. When only the sync after the rename fails,
    /// the published manifest is ahead of memory until the next save
    /// rewrites it from memory. The WAL checkpoint the manifest carries is
    /// published only by a commit that succeeded (`persist_deletions`), so
    /// a save in between never covers the records of the failed mutation.
    ///
    /// Called with the `segments` write lock held (see
    /// [`Self::save_state_locked`]).
    fn publish_locked(
        &self,
        segments: &mut Vec<ManagedSegmentInfo>,
        next: Vec<ManagedSegmentInfo>,
    ) -> Result<()> {
        self.save_state_locked(&next)?;
        *segments = next;
        Ok(())
    }

    /// [`Self::save_state_locked`], telling a save that published nothing
    /// apart from one that published the manifest but failed to sync it
    /// (Issue #1542, see
    /// [`manifest_io::save_checksummed_until_published`]).
    ///
    /// # Returns
    ///
    /// `None` when the manifest is published and synced, or the error of
    /// the sync that failed after it was published.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest was not published; the previous one
    /// is then left intact.
    fn save_state_until_published(
        &self,
        segments: &[ManagedSegmentInfo],
    ) -> Result<Option<LaurusError>> {
        let field = self
            .field_record
            .read()
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?;
        let manifest = SegmentManifest {
            version: MANIFEST_VERSION,
            next_segment_id: *self.next_segment_id.read(),
            last_wal_seq: self.last_wal_seq.load(Ordering::Acquire),
            segments: segments.to_vec(),
            field,
        };
        // temp + CRC-32 trailer + rename + sync, shared with every other
        // control file in the tree (#1022).
        manifest_io::save_checksummed_json_until_published(
            self.storage.as_ref(),
            MANIFEST_FILE,
            None,
            &manifest,
        )
    }

    /// Persist the current manifest state.
    ///
    /// # Errors
    ///
    /// Returns an error if serializing or writing the manifest fails. A
    /// failure up to the rename leaves the previously published manifest
    /// intact (atomic rename); a failed sync after the rename leaves the new
    /// one published, though a crash may still bring back the previous one
    /// (Issue #1542).
    pub fn save_state(&self) -> Result<()> {
        let segments = self.segments.read();
        self.save_state_locked(&segments)
    }

    /// Record the last applied WAL sequence number; persisted by the next
    /// save (see `SegmentManifest::last_wal_seq`).
    pub fn set_last_wal_seq(&self, seq: u64) {
        self.last_wal_seq.store(seq, Ordering::Release);
    }

    /// Last WAL sequence number recorded in the manifest.
    pub fn last_wal_seq(&self) -> u64 {
        self.last_wal_seq.load(Ordering::Acquire)
    }

    /// What the manifest records the segments were written with, if it
    /// records it (Issue #1547).
    pub(crate) fn field_record(&self) -> Option<FieldRecord> {
        self.field_record.read().clone()
    }

    /// Record what the segments are written with; persisted by the next
    /// save (Issue #1547).
    pub(crate) fn set_field_record(&self, record: FieldRecord) {
        *self.field_record.write() = Some(record);
    }

    /// Best-effort sweep of files that belong to no registered segment.
    ///
    /// Only touches names a manager of one of the kinds the index can hold
    /// generates (`segment_NNNNNN` plus the suffixes of its
    /// [`SegmentFileLayout`], the staging files of the primary and of each
    /// sidecar among them) and the manifest temp file, so foreign files —
    /// e.g. a monolithic `vector_index.hnsw` in the same directory, or a
    /// MultiVector segment — are never affected. Runs only after a
    /// successful manifest load.
    ///
    /// A file of a registered segment's id belongs to the segment only when
    /// it is of the segment's kind (Issue #1579): a rebuild as another kind
    /// that failed leaves its file under the id that the next segment of the
    /// field takes. It stays when the kind of the segment is among the ones
    /// `inferred` from the files.
    fn cleanup_orphans(&self, inferred: &HashSet<String>) {
        let Ok(files) = self.storage.list_files() else {
            return;
        };
        let segments = self.segments.read();
        let suffixes = ordered_suffixes(&self.layouts);
        for file in files {
            if file == MANIFEST_TMP_FILE {
                let _ = self.storage.delete_file(&file);
                continue;
            }
            // Strip a known suffix (longest first); skip files that are not
            // segment-shaped.
            let Some((stem, kind)) = suffixes.iter().find_map(|(suffix, kind)| {
                file.strip_suffix(suffix.as_str()).map(|stem| (stem, *kind))
            }) else {
                continue;
            };
            let Some(ordinal) = stem.strip_prefix("segment_") else {
                continue;
            };
            if ordinal.is_empty() || !ordinal.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let keep = segments
                .iter()
                .find(|segment| segment.segment_id == stem)
                .is_some_and(|segment| {
                    inferred.contains(stem) || kind.is_none_or(|kind| kind == self.kind_of(segment))
                });
            if !keep {
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
    /// the metadata-driven merge policy. A segment with no `kind` is of the
    /// index's own (Issue #1579).
    ///
    /// The segment joins the in-memory list only once the manifest that
    /// lists it is saved in full, so a commit retried after a failed save
    /// registers it once (Issue #1583).
    ///
    /// # Errors
    ///
    /// Returns an error if persisting the manifest fails. The in-memory
    /// list is then unchanged; see `Self::publish_locked` for a save that
    /// published the manifest but failed to sync it.
    pub fn add_segment(&self, mut info: ManagedSegmentInfo) -> Result<()> {
        self.stamp_kind(&mut info);
        if info.size_bytes == 0 {
            info.size_bytes = self.measure_segment_size(&info);
        }
        let mut segments = self.segments.write();
        if info.generation == 0 {
            info.generation = segments.iter().map(|s| s.generation).max().unwrap_or(0) + 1;
        }
        let mut next = segments.clone();
        next.push(info);
        self.publish_locked(&mut segments, next)
    }

    /// Make a segment that says no kind one of the index's own (Issue
    /// #1579).
    fn stamp_kind(&self, info: &mut ManagedSegmentInfo) {
        info.kind.get_or_insert_with(|| self.own_kind().to_string());
    }

    /// Sum the on-storage sizes of a segment's primary file and sidecars,
    /// of the segment's kind.
    ///
    /// `.delmap` is deliberately excluded (#882): per-segment deletion
    /// bitmaps do not exist — the bitmap is index-level — and for a
    /// legacy-migrated segment (whose id equals the index name) including
    /// it would misattribute the index bitmap's size to the segment.
    fn measure_segment_size(&self, segment: &ManagedSegmentInfo) -> u64 {
        let kind = self.kind_of(segment);
        self.layouts
            .iter()
            .filter(|layout| kind_of_layout(layout) == kind)
            .flat_map(SegmentFileLayout::file_suffixes)
            .map(|suffix| format!("{}{suffix}", segment.segment_id))
            .filter_map(|name| self.storage.file_size(&name).ok())
            .sum()
    }

    /// Remove a segment.
    ///
    /// The segment leaves the in-memory list only once the manifest without
    /// it is saved in full (Issue #1583). Removing an id that is not listed
    /// does nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if persisting the manifest fails. The in-memory
    /// list then still lists the segment.
    pub fn remove_segment(&self, segment_id: &str) -> Result<()> {
        let mut segments = self.segments.write();
        if let Some(pos) = segments.iter().position(|s| s.segment_id == segment_id) {
            let mut next = segments.clone();
            next.remove(pos);
            self.publish_locked(&mut segments, next)
        } else {
            Ok(())
        }
    }

    /// Delete physical files associated with a segment, including its
    /// sidecars — leaving them behind orphans storage after every merge
    /// (#879).
    ///
    /// The caller has published a manifest that does not list the segment,
    /// so its files are deleted whatever kind they are of (Issue #1579).
    ///
    /// Deliberately does NOT touch `{segment_id}.delmap` (#882): the
    /// segmented index keeps ONE index-level deletion bitmap named
    /// `{index_name}.delmap` — per-segment bitmaps do not exist — and a
    /// legacy-migrated segment's id EQUALS the index name, so deleting the
    /// segment's "delmap" here would destroy live deletion state when a
    /// merge consumes that segment.
    ///
    /// Waits until every reader build of the index that listed the segments
    /// before this call has loaded its readers (Issue #1678). A build that
    /// lists them afterwards finds the manifest the caller published,
    /// without the segment.
    pub fn delete_segment_files(&self, segment_id: &str) -> Result<()> {
        let _gate = match self.build_gate.try_write() {
            Some(gate) => gate,
            None => {
                #[cfg(test)]
                test_hooks::run_when_deletion_waits();
                self.build_gate.write()
            }
        };
        // Best effort deletion - ignore if a file doesn't exist
        for suffix in self
            .layouts
            .iter()
            .flat_map(SegmentFileLayout::file_suffixes)
        {
            let _ = self.storage.delete_file(&format!("{segment_id}{suffix}"));
        }
        Ok(())
    }

    /// Update a segment info.
    ///
    /// The new info replaces the listed one in memory only once the
    /// manifest with it is saved in full (Issue #1583). An id that is not
    /// listed changes nothing, but the manifest is still saved. An info with
    /// no `kind` keeps the listed one's.
    ///
    /// # Errors
    ///
    /// Returns an error if persisting the manifest fails. The in-memory
    /// list then keeps the previous info.
    pub fn update_segment(&self, mut info: ManagedSegmentInfo) -> Result<()> {
        let mut segments = self.segments.write();
        let mut next = segments.clone();
        if let Some(listed) = next.iter_mut().find(|s| s.segment_id == info.segment_id) {
            info.kind = info.kind.or_else(|| listed.kind.take());
            *listed = info;
        }
        self.publish_locked(&mut segments, next)
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

    /// List all segments for a reader build, keeping their files on storage
    /// until the result is dropped (Issue #1678).
    ///
    /// A build that may run while a merge replaces segments lists them
    /// here and holds the result until it has loaded their readers:
    /// [`Self::delete_segment_files`] waits for it, so every listed
    /// segment's files are there to load. A merge's reads of its own
    /// sources need not, since only merges delete segment files.
    ///
    /// Take it only where the segments are listed, never again on the same
    /// thread while holding one: the lock is fair, so a second read waits
    /// behind a waiting deletion, which waits for the first.
    pub(crate) fn pinned_segments(&self) -> PinnedSegments<'_> {
        // The gate before the list: a merge that publishes after this
        // listing deletes the replaced files only after the build.
        let gate = self.build_gate.read();
        PinnedSegments {
            segments: self.list_segments(),
            _gate: gate,
        }
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
    ///
    /// [`Self::publish_merge`] publishes the manifest that lists the merged
    /// segment instead of the sources. The sources' files are deleted only
    /// once that manifest is also synced: when the sync fails, a crash may
    /// bring back the previous manifest, which lists the sources, so their
    /// files stay for the next open's orphan sweep, which removes whichever
    /// segments the manifest it reads does not list (Issue #1542). The
    /// deletion waits for the reader builds that listed the sources before
    /// the publish (Issue #1678, see [`Self::delete_segment_files`]).
    ///
    /// # Returns
    ///
    /// `None` when the merge is published and synced, or the error of the
    /// sync that failed after it was published. The merged segment replaces
    /// the sources in memory either way.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest was not published. The in-memory
    /// list and the manifest then still list the sources, and the merged
    /// segment's files are left for the next open's orphan sweep.
    pub fn apply_merge(
        &self,
        candidate: MergeCandidate,
        merged_segment: ManagedSegmentInfo,
    ) -> Result<Option<LaurusError>> {
        let source_ids: Vec<String> = candidate
            .segments
            .iter()
            .map(|s| s.segment_id.clone())
            .collect();
        if let Some(sync_error) = self.publish_merge(candidate, merged_segment)? {
            return Ok(Some(sync_error));
        }
        for segment_id in &source_ids {
            self.delete_segment_files(segment_id)?;
        }
        Ok(None)
    }

    /// Publish a merge result: the manifest that lists the merged segment
    /// instead of the sources is saved, and the in-memory list takes it only
    /// once it is.
    ///
    /// Unlike [`Self::apply_merge`], the sources' files stay on storage
    /// (Issue #1543): another manager opened over the same storage before
    /// the merge still lists them, and a reader built through it loads them.
    /// The caller deletes them with [`Self::delete_segment_files`] once
    /// nothing can read them through such a manager, and only when this
    /// returned no sync error.
    ///
    /// # Returns
    ///
    /// `None` when the merge is published and synced, or the error of the
    /// sync that failed after it was published. The merged segment replaces
    /// the sources in memory either way.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest was not published. The in-memory
    /// list and the manifest then still list the sources, and the merged
    /// segment's files are left for the next open's orphan sweep.
    pub fn publish_merge(
        &self,
        candidate: MergeCandidate,
        mut merged_segment: ManagedSegmentInfo,
    ) -> Result<Option<LaurusError>> {
        // Replace the engine's estimated size with the on-storage size of
        // the merged segment file + sidecars (mirrors `add_segment`, #879).
        self.stamp_kind(&mut merged_segment);
        merged_segment.size_bytes = self.measure_segment_size(&merged_segment);
        let mut segments_lock = self.segments.write();

        // 1. Replace the sources with the merged segment, on a copy until
        //    the manifest is published.
        let ids_to_remove: HashSet<_> = candidate.segments.iter().map(|s| &s.segment_id).collect();
        let mut merged_list = segments_lock.clone();
        merged_list.retain(|s| !ids_to_remove.contains(&s.segment_id));
        merged_list.push(merged_segment);

        // 2. Publish it (lock held — see `save_state_locked`).
        let sync_error = self.save_state_until_published(&merged_list)?;
        *segments_lock = merged_list;
        Ok(sync_error)
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
        let avg_vectors_per_segment = if segment_count > 0 {
            total_vectors as f64 / segment_count as f64
        } else {
            0.0
        };

        SegmentManagerStats {
            segment_count,
            total_vectors,
            total_size,
            avg_vectors_per_segment,
        }
    }
}

/// Points where a test can act in the middle of a reader build or a merge
/// on the same thread (Issue #1678). Each hook runs once, in the next build
/// or deletion that reaches its point.
#[cfg(test)]
pub(crate) mod test_hooks {
    use std::cell::RefCell;

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static AFTER_LISTING: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static WHEN_DELETION_WAITS: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static AFTER_BUILD: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    fn run(slot: &'static std::thread::LocalKey<RefCell<Option<Hook>>>) {
        // Taken out before it runs, so a build inside the hook runs none.
        let hook = slot.with(|slot| slot.borrow_mut().take());
        if let Some(hook) = hook {
            hook();
        }
    }

    /// Run `hook` in the next reader build on this thread, right after it
    /// has listed the segments and before it loads their readers.
    pub(crate) fn set_after_listing(hook: impl FnOnce() + 'static) {
        AFTER_LISTING.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    pub(crate) fn run_after_listing() {
        run(&AFTER_LISTING);
    }

    /// Run `hook` when a deletion of segment files on this thread has to
    /// wait for reader builds, before it waits.
    pub(crate) fn set_when_deletion_waits(hook: impl FnOnce() + 'static) {
        WHEN_DELETION_WAITS.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    pub(super) fn run_when_deletion_waits() {
        run(&WHEN_DELETION_WAITS);
    }

    /// Run `hook` in the next reader build on this thread, once it has
    /// loaded the readers and released the listed segments, before it
    /// pairs them with a deletion bitmap.
    pub(crate) fn set_after_build(hook: impl FnOnce() + 'static) {
        AFTER_BUILD.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    pub(crate) fn run_after_build() {
        run(&AFTER_BUILD);
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
            size_bytes: count * 100,
            kind: None,
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

    /// #889: a distinct layout sweeps its own orphans and spares foreign
    /// files — proving the layout parameterization, not just the HNSW
    /// default, actually drives the sweep. This is a synthetic no-sidecar
    /// layout, of a kind no other reads.
    #[test]
    fn test_cleanup_orphans_respects_a_different_layout() {
        const OTHER_LAYOUT: SegmentFileLayout = SegmentFileLayout {
            primary: ".other",
            sidecars: &[],
            tmp: ".other.tmp",
        };
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        register(&storage, OTHER_LAYOUT, "segment_000000");
        let spared = [
            // Registered segment's file.
            "segment_000000.other",
            // Foreign files, also of the numbering shape.
            "segment_000099.hnsw",
            "segment_000099.mv",
            "vector_index.other",
        ];
        // An orphan (a torn flush).
        let swept = ["segment_000042.other", "segment_000042.other.tmp"];
        check_sweep(&storage, OTHER_LAYOUT, &spared, &swept);
    }

    /// Register segment `id`, which says no kind, with a manager for
    /// `layout`.
    fn register(storage: &Arc<MemoryStorage>, layout: SegmentFileLayout, id: &str) {
        let config = SegmentManagerConfig::default();
        let manager = SegmentManager::new(config, storage.clone(), layout).unwrap();
        manager.add_segment(create_info(id, 100)).unwrap();
    }

    fn saved_manifest(storage: &Arc<MemoryStorage>) -> serde_json::Value {
        manifest_io::load_checksummed_json(storage.as_ref(), MANIFEST_FILE, None)
            .unwrap()
            .unwrap()
            .0
    }

    fn write(storage: &Arc<MemoryStorage>, name: &str, bytes: &[u8]) {
        let mut out = storage.create_output(name).unwrap();
        std::io::Write::write_all(&mut out, bytes).unwrap();
        out.close().unwrap();
    }

    /// Write `spared` and `swept`, open a manager for `layout`, and check
    /// that it removes exactly the latter.
    fn check_sweep(
        storage: &Arc<MemoryStorage>,
        layout: SegmentFileLayout,
        spared: &[&str],
        swept: &[&str],
    ) {
        for name in spared.iter().chain(swept) {
            write(storage, name, b"x");
        }
        let config = SegmentManagerConfig::default();
        let _manager = SegmentManager::new(config, storage.clone(), layout).unwrap();
        let wrong: Vec<&&str> = spared
            .iter()
            .filter(|name| !storage.file_exists(name))
            .chain(swept.iter().filter(|name| storage.file_exists(name)))
            .collect();
        assert!(wrong.is_empty(), "{:?}: {wrong:?}", layout.primary);
    }

    const FAMILY: [SegmentFileLayout; 3] = [
        crate::vector::index::flat::segment::LAYOUT,
        crate::vector::index::hnsw::segment::LAYOUT,
        crate::vector::index::ivf::segment::LAYOUT,
    ];

    /// Issue #1579: a manager of one of the three kinds sweeps the files of
    /// all three that its manifest does not list, under an id of no segment
    /// and under the id of a segment of another kind, with the sidecars and
    /// the staging files. It spares MultiVector files, the deletion bitmap,
    /// a codebook, the directory of a nested field and the single-file
    /// layout of every kind.
    #[test]
    fn the_sweep_covers_the_files_of_every_kind_the_index_can_hold() {
        for layout in FAMILY {
            let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            register(&storage, layout, "segment_000000");
            // The second segment is of another kind than the index's own.
            let other = FAMILY
                .into_iter()
                .find(|other| other.primary != layout.primary)
                .unwrap();
            {
                let config = SegmentManagerConfig::default();
                let manager = SegmentManager::new(config, storage.clone(), layout).unwrap();
                let mut info = create_info("segment_000001", 100);
                info.kind = Some(kind_of_layout(&other).to_string());
                manager.add_segment(info).unwrap();
            }
            let mut spared = vec![
                "segment_000042.mv".to_string(),
                "index.delmap".to_string(),
                "codebook.pqcb".to_string(),
                "b/segment_000042.hnsw".to_string(),
            ];
            let mut swept = vec![
                "segment_000042.delmap".to_string(),
                "segments.json.tmp".to_string(),
            ];
            for kind in FAMILY {
                let primary = kind.primary;
                spared.push(format!("index{primary}"));
                for suffix in ["", ".f32"] {
                    for (id, of_kind) in [("segment_000000", layout), ("segment_000001", other)] {
                        let file = format!("{id}{primary}{suffix}");
                        if primary == of_kind.primary {
                            spared.push(file);
                        } else {
                            swept.push(file);
                        }
                    }
                    swept.push(format!("segment_000042{primary}{suffix}"));
                    swept.push(format!("segment_000042{primary}{suffix}.tmp"));
                }
            }
            check_sweep(
                &storage,
                layout,
                &spared.iter().map(String::as_str).collect::<Vec<_>>(),
                &swept.iter().map(String::as_str).collect::<Vec<_>>(),
            );
        }
    }

    /// Issue #1579: a segment that joins a manager and says no kind is of
    /// the manager's, also one a merge wrote; one that says another keeps
    /// it, also when it is updated with an info that says none. The manifest
    /// records it.
    #[test]
    fn a_segment_is_recorded_with_its_kind() {
        for layout in FAMILY {
            let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            let config = SegmentManagerConfig::default();
            let manager = SegmentManager::new(config, storage.clone(), layout).unwrap();
            let own = kind_of_layout(&layout);
            let other = if own == "hnsw" { "ivf" } else { "hnsw" };
            manager
                .add_segment(create_info("segment_000000", 100))
                .unwrap();
            let mut info = create_info("segment_000001", 100);
            info.kind = Some(other.to_string());
            manager.add_segment(info).unwrap();
            assert_eq!(manager.segments_of_other_kinds(), 1);
            manager
                .update_segment(create_info("segment_000001", 50))
                .unwrap();
            let kinds = |manager: &SegmentManager| -> Vec<(String, Option<String>)> {
                manager
                    .list_segments()
                    .into_iter()
                    .map(|segment| (segment.segment_id, segment.kind))
                    .collect()
            };
            let expected = vec![
                ("segment_000000".to_string(), Some(own.to_string())),
                ("segment_000001".to_string(), Some(other.to_string())),
            ];
            assert_eq!(kinds(&manager), expected);
            let saved = saved_manifest(&storage);
            assert_eq!(saved["segments"][0]["kind"], own);
            assert_eq!(saved["segments"][1]["kind"], other);
            let config = SegmentManagerConfig::default();
            let reopened = SegmentManager::new(config, storage.clone(), layout).unwrap();
            assert_eq!(kinds(&reopened), expected);

            let candidate = MergeCandidate {
                segments: reopened.list_segments(),
                total_vectors: 150,
                total_size: 0,
            };
            reopened
                .publish_merge(candidate, create_info("segment_000002", 150))
                .unwrap();
            assert_eq!(
                kinds(&reopened),
                [("segment_000002".to_string(), Some(own.to_string()))]
            );
            assert_eq!(reopened.segments_of_other_kinds(), 0);
        }
    }

    /// Issue #1579: a manifest written before it records no kind of its
    /// segments. Each is of the kind recorded for the field, the index's own
    /// or another it can hold, the first whose primary file exists, and of
    /// the recorded or the own one when none does. A file of another kind
    /// under such an id stays at that load. The next save records the kind.
    #[test]
    fn a_segment_without_a_kind_is_given_one_by_its_files() {
        let hnsw = crate::vector::index::hnsw::segment::LAYOUT;
        // (recorded for the field, files of the segment, expected kind).
        let cases: [(Option<&str>, &[&str], &str); 8] = [
            (None, &[], "hnsw"),
            (None, &["hnsw"], "hnsw"),
            (None, &["flat"], "flat"),
            (None, &["ivf"], "ivf"),
            (None, &["flat", "hnsw"], "hnsw"),
            (Some("flat"), &["flat", "hnsw"], "flat"),
            (Some("flat"), &["hnsw"], "hnsw"),
            (Some("ivf"), &[], "ivf"),
        ];
        for (recorded, files, expected) in cases {
            let what = format!("recorded {recorded:?}, files {files:?}");
            let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            register(&storage, hnsw, "segment_000000");
            let mut old = saved_manifest(&storage);
            old["segments"][0].as_object_mut().unwrap().remove("kind");
            if let Some(kind) = recorded {
                old["field"] =
                    serde_json::json!({"kind": kind, "dimension": 8, "distance": "Cosine"});
            }
            manifest_io::save_checksummed_json(storage.as_ref(), MANIFEST_FILE, None, &old)
                .unwrap();
            let files: Vec<String> = files
                .iter()
                .map(|kind| format!("segment_000000.{kind}"))
                .collect();
            for file in &files {
                write(&storage, file, b"x");
            }

            let config = SegmentManagerConfig::default();
            let manager = SegmentManager::new(config, storage.clone(), hnsw).unwrap();
            let segment = &manager.list_segments()[0];
            assert_eq!(segment.kind.as_deref(), Some(expected), "{what}");
            for file in &files {
                assert!(storage.file_exists(file), "{what}: {file} must stay");
            }
            manager.save_state().unwrap();
            assert_eq!(
                saved_manifest(&storage)["segments"][0]["kind"],
                expected,
                "{what}"
            );
        }
    }

    /// Issue #1579: the size of a segment is that of the files of its kind,
    /// and its deletion removes the files of every kind under its id.
    #[test]
    fn a_segments_size_is_of_its_kind_and_its_deletion_of_every_kind() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = SegmentManager::new(
            SegmentManagerConfig::default(),
            storage.clone(),
            crate::vector::index::hnsw::segment::LAYOUT,
        )
        .unwrap();
        for (file, bytes) in [
            ("segment_000000.hnsw", 1),
            ("segment_000000.hnsw.f32", 2),
            ("segment_000000.flat", 4),
            ("segment_000000.flat.f32", 8),
            ("segment_000000.ivf", 16),
            ("segment_000000.mv", 32),
        ] {
            write(&storage, file, &vec![0; bytes]);
        }
        for (kind, size) in [(None, 3), (Some("flat"), 12), (Some("ivf"), 16)] {
            let mut info = create_info("segment_000000", 1);
            info.size_bytes = 0;
            info.kind = kind.map(str::to_string);
            manager.add_segment(info).unwrap();
            let listed = manager.get_segment("segment_000000").unwrap();
            assert_eq!(listed.size_bytes, size, "{kind:?}");

            // A merge's segment is measured when it is published.
            let candidate = MergeCandidate {
                segments: vec![listed],
                total_vectors: 1,
                total_size: 0,
            };
            let mut merged = create_info("segment_000000", 1);
            merged.kind = kind.map(str::to_string);
            manager.publish_merge(candidate, merged).unwrap();
            let listed = manager.get_segment("segment_000000").unwrap();
            assert_eq!(listed.size_bytes, size, "merged, {kind:?}");
            manager.remove_segment("segment_000000").unwrap();
        }
        manager.delete_segment_files("segment_000000").unwrap();
        let mut left = storage.list_files().unwrap();
        left.retain(|file| file.starts_with("segment_"));
        assert_eq!(left, ["segment_000000.mv"]);
    }

    /// Issue #1532: the Flat, HNSW and IVF writers stage a rerank sidecar as
    /// the sidecar's name plus `.tmp` and rename it after the primary file,
    /// so a write that fails or crashes in between leaves
    /// `segment_N.{flat,hnsw,ivf}.f32.tmp`. Under each real layout the sweep
    /// reclaims it for an unregistered segment, and keeps a registered
    /// segment's primary and sidecar.
    #[test]
    fn test_cleanup_orphans_reclaims_a_sidecar_staging_file() {
        for (layout, kind) in [
            (crate::vector::index::flat::segment::LAYOUT, "flat"),
            (crate::vector::index::hnsw::segment::LAYOUT, "hnsw"),
            (crate::vector::index::ivf::segment::LAYOUT, "ivf"),
        ] {
            let config = SegmentManagerConfig::default();
            let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            let write = |name: &str| {
                let mut out = storage.create_output(name).unwrap();
                std::io::Write::write_all(&mut out, b"x").unwrap();
                out.close().unwrap();
            };

            {
                let manager = SegmentManager::new(config.clone(), storage.clone(), layout).unwrap();
                manager
                    .add_segment(create_info("segment_000000", 100))
                    .unwrap();
            }
            // The names the writers give the files (`sidecar_tmp` in
            // `{flat,hnsw,ivf}/writer.rs`).
            let registered = [
                format!("segment_000000.{kind}"),
                format!("segment_000000.{kind}.f32"),
            ];
            let orphans = [
                format!("segment_000042.{kind}"),
                format!("segment_000042.{kind}.f32.tmp"),
            ];
            for name in registered.iter().chain(&orphans) {
                write(name);
            }

            let _manager = SegmentManager::new(config, storage.clone(), layout).unwrap();
            for name in &registered {
                assert!(storage.file_exists(name), "{name} must stay");
            }
            for name in &orphans {
                assert!(!storage.file_exists(name), "{name} must be swept");
            }
        }
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

    /// Manifests written before #1411 carry a `has_deletions` key on every
    /// segment. They must still load: `ManagedSegmentInfo` does not deny
    /// unknown fields, so serde skips the key.
    #[test]
    fn test_manifest_with_removed_has_deletions_field_still_loads() {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let json = br#"{"version":1,"next_segment_id":1,"last_wal_seq":0,"segments":[{"segment_id":"segment_000000","vector_count":2,"vector_offset":0,"generation":1,"has_deletions":true,"size_bytes":42}]}"#;
        crate::storage::manifest::save_checksummed(storage.as_ref(), MANIFEST_FILE, None, json)
            .unwrap();

        let manager =
            SegmentManager::new(SegmentManagerConfig::default(), storage, TEST_LAYOUT).unwrap();
        let segments = manager.list_segments();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].segment_id, "segment_000000");
        assert_eq!(segments[0].vector_count, 2);
        assert_eq!(segments[0].generation, 1);
    }

    fn hnsw_record(dimension: usize) -> FieldRecord {
        FieldRecord::new(
            &TEST_LAYOUT,
            dimension,
            crate::vector::core::distance::DistanceMetric::Cosine,
        )
    }

    /// Issue #1547: the field record a manager holds is written by the next
    /// save and read back by the next open.
    #[test]
    fn test_field_record_round_trips_through_the_manifest() {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let manager = SegmentManager::new(
            SegmentManagerConfig::default(),
            storage.clone(),
            TEST_LAYOUT,
        )
        .unwrap();
        assert_eq!(manager.field_record(), None);
        manager.set_field_record(hnsw_record(8));
        manager
            .add_segment(create_info("segment_000000", 2))
            .unwrap();

        let reopened =
            SegmentManager::new(SegmentManagerConfig::default(), storage, TEST_LAYOUT).unwrap();
        assert_eq!(reopened.field_record(), Some(hnsw_record(8)));
        assert_eq!(
            hnsw_record(8),
            FieldRecord {
                kind: "hnsw".to_string(),
                dimension: 8,
                distance: crate::vector::core::distance::DistanceMetric::Cosine,
            }
        );
    }

    /// Issue #1547: a manifest written before the record existed loads, with
    /// no record, and a save under a manager that holds none adds no key.
    #[test]
    fn test_manifest_without_field_record_loads_without_one() {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let json = br#"{"version":1,"next_segment_id":1,"last_wal_seq":0,"segments":[{"segment_id":"segment_000000","vector_count":2,"vector_offset":0,"generation":1,"size_bytes":42}]}"#;
        crate::storage::manifest::save_checksummed(storage.as_ref(), MANIFEST_FILE, None, json)
            .unwrap();

        let manager = SegmentManager::new(
            SegmentManagerConfig::default(),
            storage.clone(),
            TEST_LAYOUT,
        )
        .unwrap();
        assert_eq!(manager.list_segments().len(), 1);
        assert_eq!(manager.field_record(), None);

        manager.save_state().unwrap();
        let (payload, _) =
            crate::storage::manifest::load_checksummed(storage.as_ref(), MANIFEST_FILE, None)
                .unwrap()
                .unwrap();
        let saved: serde_json::Value = serde_json::from_slice(&payload).unwrap();
        assert!(saved.get("field").is_none(), "{saved}");
    }

    /// Issue #1547: a record this build cannot read, such as a distance a
    /// later build added, counts as no record instead of failing the load.
    #[test]
    fn test_unreadable_field_record_counts_as_none() {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let json = br#"{"version":1,"next_segment_id":1,"last_wal_seq":0,"segments":[{"segment_id":"segment_000000","vector_count":2,"vector_offset":0,"generation":1,"size_bytes":42}],"field":{"kind":"hnsw","dimension":8,"distance":"Hamming"}}"#;
        crate::storage::manifest::save_checksummed(storage.as_ref(), MANIFEST_FILE, None, json)
            .unwrap();

        let manager =
            SegmentManager::new(SegmentManagerConfig::default(), storage, TEST_LAYOUT).unwrap();
        assert_eq!(manager.list_segments().len(), 1);
        assert_eq!(manager.field_record(), None);
    }

    const MERGE_SOURCES: [&str; 2] = ["segment_000000", "segment_000001"];
    const MERGED: &str = "segment_000002";

    /// A manager over `storage` whose manifest lists the two
    /// [`MERGE_SOURCES`], with the files of those and of [`MERGED`] written
    /// to `inner`, the storage under `storage`.
    fn manager_ready_to_merge(
        inner: &Arc<dyn Storage>,
        storage: Arc<dyn Storage>,
    ) -> SegmentManager {
        let manager =
            SegmentManager::new(SegmentManagerConfig::default(), storage, TEST_LAYOUT).unwrap();
        for id in MERGE_SOURCES.iter().chain([&MERGED]) {
            for name in segment_files(id) {
                crate::storage::manifest::test_support::write_file(inner.as_ref(), &name, b"x");
            }
        }
        for id in MERGE_SOURCES {
            manager.add_segment(create_info(id, 10)).unwrap();
        }
        manager
    }

    fn segment_files(id: &str) -> [String; 2] {
        [format!("{id}.hnsw"), format!("{id}.hnsw.f32")]
    }

    fn segment_ids(manager: &SegmentManager) -> Vec<String> {
        manager
            .list_segments()
            .into_iter()
            .map(|s| s.segment_id)
            .collect()
    }

    fn merge_candidate(manager: &SegmentManager) -> MergeCandidate {
        MergeCandidate {
            segments: manager.list_segments(),
            total_vectors: 20,
            total_size: 0,
        }
    }

    /// Issue #1542: a merge whose manifest sync fails after the rename is
    /// published. The merged segment replaces the sources in memory, and
    /// the sources' files stay, so the next open finds every file the
    /// manifest it reads lists: the published one, or the previous one a
    /// crash brings back. Its orphan sweep removes the others.
    #[test]
    fn test_apply_merge_keeps_the_sources_when_the_manifest_sync_fails() {
        use crate::storage::manifest::test_support::{FailSyncAfterRename, read_file, write_file};
        use std::sync::atomic::Ordering;

        for crash in [false, true] {
            let inner: Arc<dyn Storage> =
                Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            let failing = Arc::new(FailSyncAfterRename::new(inner.clone(), MANIFEST_FILE));
            let manager = manager_ready_to_merge(&inner, failing.clone());
            let previous_manifest = read_file(inner.as_ref(), MANIFEST_FILE);

            failing.armed.store(true, Ordering::SeqCst);
            let merged = manager.apply_merge(merge_candidate(&manager), create_info(MERGED, 20));
            failing.armed.store(false, Ordering::SeqCst);
            let sync_error = merged
                .expect("the merge is published")
                .expect("the failed sync is reported");
            assert!(
                sync_error.to_string().contains("injected sync failure"),
                "{sync_error}"
            );
            assert_eq!(segment_ids(&manager), [MERGED]);
            for id in MERGE_SOURCES.iter().chain([&MERGED]) {
                for name in segment_files(id) {
                    assert!(
                        inner.file_exists(&name),
                        "{name} must stay until the next open"
                    );
                }
            }

            if crash {
                write_file(inner.as_ref(), MANIFEST_FILE, &previous_manifest);
            }
            let reopened =
                SegmentManager::new(SegmentManagerConfig::default(), inner.clone(), TEST_LAYOUT)
                    .unwrap();
            let (live, swept) = if crash {
                (MERGE_SOURCES.to_vec(), vec![MERGED])
            } else {
                (vec![MERGED], MERGE_SOURCES.to_vec())
            };
            assert_eq!(segment_ids(&reopened), live, "crash: {crash}");
            for id in &live {
                for name in segment_files(id) {
                    assert!(inner.file_exists(&name), "crash: {crash}: {name} must stay");
                }
            }
            for id in &swept {
                for name in segment_files(id) {
                    assert!(
                        !inner.file_exists(&name),
                        "crash: {crash}: {name} must be swept"
                    );
                }
            }
        }
    }

    /// Issue #1542: a merge whose manifest is not published changes
    /// nothing: memory and the manifest still list the sources, and their
    /// files stay.
    #[test]
    fn test_apply_merge_changes_nothing_when_the_manifest_is_not_published() {
        use crate::storage::manifest::test_support::{FailSyncAfterRename, read_file};
        use std::sync::atomic::Ordering;

        let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let failing = Arc::new(FailSyncAfterRename::new(inner.clone(), MANIFEST_FILE));
        let manager = manager_ready_to_merge(&inner, failing.clone());
        let previous_manifest = read_file(inner.as_ref(), MANIFEST_FILE);

        failing.armed.store(true, Ordering::SeqCst);
        failing.fail_rename.store(true, Ordering::SeqCst);
        let error = manager
            .apply_merge(merge_candidate(&manager), create_info(MERGED, 20))
            .unwrap_err();
        assert!(error.to_string().contains("injected failure"), "{error}");
        assert_eq!(segment_ids(&manager), MERGE_SOURCES);
        assert_eq!(read_file(inner.as_ref(), MANIFEST_FILE), previous_manifest);
        for id in MERGE_SOURCES {
            for name in segment_files(id) {
                assert!(inner.file_exists(&name), "{name} must stay");
            }
        }
    }

    /// A manager over a storage that can fail its manifest saves, with
    /// `segment_000000` registered, and the label of the failure point
    /// `fail_rename` selects: the rename (nothing published) or the sync
    /// after it (published).
    fn manager_with_failing_saves(
        fail_rename: bool,
    ) -> (
        Arc<dyn Storage>,
        Arc<crate::storage::manifest::test_support::FailSyncAfterRename>,
        SegmentManager,
        &'static str,
    ) {
        use crate::storage::manifest::test_support::FailSyncAfterRename;

        let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let failing = Arc::new(FailSyncAfterRename::new(inner.clone(), MANIFEST_FILE));
        failing
            .fail_rename
            .store(fail_rename, std::sync::atomic::Ordering::SeqCst);
        let manager = SegmentManager::new(
            SegmentManagerConfig::default(),
            failing.clone(),
            TEST_LAYOUT,
        )
        .unwrap();
        manager
            .add_segment(create_info("segment_000000", 10))
            .unwrap();
        let case = if fail_rename {
            "rename failed"
        } else {
            "sync failed"
        };
        (inner, failing, manager, case)
    }

    /// Run `mutation` with one injected manifest save failure, and check
    /// that it failed on it.
    fn fail_once(
        failing: &crate::storage::manifest::test_support::FailSyncAfterRename,
        case: &str,
        mutation: impl FnOnce() -> Result<()>,
    ) {
        use std::sync::atomic::Ordering;

        failing.armed.store(true, Ordering::SeqCst);
        let result = mutation();
        failing.armed.store(false, Ordering::SeqCst);
        assert!(result.is_err(), "{case}: the save must fail");
        assert_eq!(
            failing.failures.load(Ordering::SeqCst),
            1,
            "{case}: the injected failure must be hit"
        );
    }

    /// Issue #1583: an `add_segment` whose manifest save fails leaves the
    /// in-memory list unchanged, so calling it again, as a commit retried
    /// from its writer does, lists the segment once.
    #[test]
    fn add_segment_whose_save_fails_leaves_the_list_unchanged() {
        for fail_rename in [true, false] {
            let (inner, failing, manager, case) = manager_with_failing_saves(fail_rename);

            fail_once(&failing, case, || {
                manager.add_segment(create_info("segment_000001", 20))
            });
            assert_eq!(segment_ids(&manager), ["segment_000000"], "{case}");

            manager
                .add_segment(create_info("segment_000001", 20))
                .unwrap();
            assert_eq!(
                segment_ids(&manager),
                ["segment_000000", "segment_000001"],
                "{case}"
            );
            let reopened =
                SegmentManager::new(SegmentManagerConfig::default(), inner, TEST_LAYOUT).unwrap();
            assert_eq!(
                segment_ids(&reopened),
                ["segment_000000", "segment_000001"],
                "{case}"
            );
        }
    }

    /// Issue #1583: a `remove_segment` whose manifest save fails keeps the
    /// segment in memory, so calling it again removes it and saves.
    #[test]
    fn remove_segment_whose_save_fails_keeps_the_segment() {
        for fail_rename in [true, false] {
            let (inner, failing, manager, case) = manager_with_failing_saves(fail_rename);

            fail_once(&failing, case, || manager.remove_segment("segment_000000"));
            assert_eq!(segment_ids(&manager), ["segment_000000"], "{case}");

            manager.remove_segment("segment_000000").unwrap();
            assert!(segment_ids(&manager).is_empty(), "{case}");
            let reopened =
                SegmentManager::new(SegmentManagerConfig::default(), inner, TEST_LAYOUT).unwrap();
            assert!(segment_ids(&reopened).is_empty(), "{case}");
        }
    }

    /// Issue #1583: an `update_segment` whose manifest save fails keeps the
    /// previous info in memory.
    #[test]
    fn update_segment_whose_save_fails_keeps_the_previous_entry() {
        for fail_rename in [true, false] {
            let (inner, failing, manager, case) = manager_with_failing_saves(fail_rename);

            fail_once(&failing, case, || {
                manager.update_segment(create_info("segment_000000", 99))
            });
            assert_eq!(
                manager.get_segment("segment_000000").unwrap().vector_count,
                10,
                "{case}"
            );

            manager
                .update_segment(create_info("segment_000000", 99))
                .unwrap();
            assert_eq!(
                manager.get_segment("segment_000000").unwrap().vector_count,
                99,
                "{case}"
            );
            let reopened =
                SegmentManager::new(SegmentManagerConfig::default(), inner, TEST_LAYOUT).unwrap();
            assert_eq!(
                reopened.get_segment("segment_000000").unwrap().vector_count,
                99,
                "{case}"
            );
        }
    }
}
