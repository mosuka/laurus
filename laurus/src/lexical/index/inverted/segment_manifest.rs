//! Atomic segment manifest for the lexical inverted index (#1021).
//!
//! `segments.json` is the single, atomically replaced record of the
//! committed segment set — the counterpart of the vector side's
//! `SegmentManager` manifest, built on the shared
//! [`storage::manifest`](crate::storage::manifest) helpers (CRC-verified,
//! written via temp + rename + directory sync).
//!
//! # Authority and the save-then-swap rule
//!
//! The in-memory copy (owned by
//! [`InvertedIndex`](super::InvertedIndex), shared with its writer) is a
//! mirror of the last **successfully persisted** manifest, never ahead of
//! it. Every mutation goes through [`publish_with`]: build a candidate
//! under the write guard, persist it, and only on success swap it into
//! memory. This deliberately deviates from the vector manager's
//! mutate-then-save: there, a failed save leaves memory ahead of disk,
//! which is survivable when every later save rewrites the full list — but
//! here a retried commit would find its pending state consumed, publish
//! nothing, and still advance the WAL checkpoint, after which a reopen
//! would neither replay nor discover the segments.
//!
//! The manifest lock is a **leaf**: nothing under its guard may construct
//! readers, list storage, or take the index's metadata lock. The guard is
//! intentionally held across the save so two racing mutators cannot
//! publish manifests out of order (the same argument as
//! `SegmentManager::save_state_locked`); this is safe only while no other
//! code path — `Debug` impls included — blocks on this lock.

use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::storage::Storage;
use crate::storage::manifest as manifest_io;

use super::segment::SegmentInfo;

/// Manifest file name inside the lexical storage namespace.
pub(crate) const MANIFEST_FILE: &str = "segments.json";

/// On-disk manifest version written by this build.
///
/// Version 1 was written while segment discovery still ran on the `.meta`
/// scan (#1021 PR 1) — those manifests were never the authority, so a
/// loader treats them exactly like an absent manifest (rebuild from the
/// committed-`.meta` scan) and never uses them as a deletion warrant.
/// Version 2 means discovery reads the manifest and the orphan sweep may
/// trust it.
pub(crate) const MANIFEST_VERSION: u32 = 2;

/// The first version whose manifest is authoritative for discovery and
/// for the orphan sweep — see [`MANIFEST_VERSION`].
pub(crate) const AUTHORITATIVE_VERSION: u32 = 2;

/// In-memory manifest state shared between the index and its writers.
#[derive(Debug)]
pub(crate) struct ManifestState {
    /// The committed segments — the mirror of the last successfully
    /// persisted `segments.json`.
    pub(crate) segments: Vec<SegmentInfo>,

    /// Next segment generation ordinal to hand out (#1024).
    ///
    /// **In-memory only, never persisted.** Persistence is unnecessary
    /// because a lexical merge always inserts its entry with a generation
    /// strictly above its sources, so re-deriving from the entries can
    /// never regress past a live segment; crash-lost reservations are
    /// covered because the seed also scans surviving segment-file stems
    /// (see `InvertedIndex::open`), which is strictly stronger than a
    /// persisted counter (a memory-only bump would not have been saved
    /// anyway). Reserved under this lock at flush/merge time via
    /// [`reserve_generation`], which is what keeps a surviving writer and
    /// a concurrent merge from minting the same generation.
    pub(crate) next_generation: u64,
}

/// Shared handle to the in-memory manifest state.
pub(crate) type SharedSegmentManifest = Arc<RwLock<ManifestState>>;

/// Serialized form of [`MANIFEST_FILE`].
#[derive(Debug, Serialize, Deserialize)]
struct SegmentManifest {
    /// Format version — see [`MANIFEST_VERSION`].
    version: u32,
    /// The committed segments, in publication order.
    segments: Vec<SegmentInfo>,
}

/// Load the manifest from `storage`.
///
/// # Arguments
///
/// * `storage` - The lexical index's storage.
///
/// # Returns
///
/// `Ok(Some((version, segments)))` when the manifest exists, `Ok(None)`
/// when it does not (a legacy index — the caller falls back to the
/// `.meta` scan). Callers must treat a version below
/// [`AUTHORITATIVE_VERSION`] like an absent manifest for discovery and
/// sweeping purposes.
///
/// # Errors
///
/// Returns an error if the file exists but is corrupted (checksum
/// mismatch) or fails to parse — a torn or damaged manifest must surface,
/// not silently degrade to an empty segment set.
pub(crate) fn load(storage: &dyn Storage) -> Result<Option<(u32, Vec<SegmentInfo>)>> {
    match manifest_io::load_checksummed_json::<SegmentManifest>(storage, MANIFEST_FILE, None)? {
        Some((manifest, _format)) => Ok(Some((manifest.version, manifest.segments))),
        None => Ok(None),
    }
}

/// Persist `segments` as the new manifest, atomically.
///
/// # Arguments
///
/// * `storage` - The lexical index's storage.
/// * `segments` - The complete committed segment set to record.
///
/// # Errors
///
/// Returns an error if serializing or persisting fails; the previous
/// manifest is left intact in that case (temp + rename).
pub(crate) fn save(storage: &dyn Storage, segments: &[SegmentInfo]) -> Result<()> {
    let manifest = SegmentManifest {
        version: MANIFEST_VERSION,
        segments: segments.to_vec(),
    };
    manifest_io::save_checksummed_json(storage, MANIFEST_FILE, None, &manifest)
}

/// Mutate the shared manifest under the save-then-swap rule.
///
/// Clones the current list, applies `mutate` to the clone, persists the
/// candidate while still holding the write guard, and swaps it into
/// memory **only if the save succeeded**. On failure the in-memory copy
/// (and the caller's pending state) is untouched, so a retried commit
/// republishes the same mutation.
///
/// # Arguments
///
/// * `storage` - The lexical index's storage.
/// * `shared` - The shared in-memory manifest.
/// * `mutate` - Delta to apply to the candidate list. It receives the
///   candidate, never the live list.
///
/// # Errors
///
/// Returns the persistence error, leaving memory unchanged.
pub(crate) fn publish_with<F>(
    storage: &dyn Storage,
    shared: &RwLock<ManifestState>,
    mutate: F,
) -> Result<()>
where
    F: FnOnce(&mut Vec<SegmentInfo>),
{
    let mut guard = shared.write();
    let mut candidate = guard.segments.clone();
    mutate(&mut candidate);
    save(storage, &candidate)?;
    guard.segments = candidate;
    Ok(())
}

/// The index's document and deletion counts, summed over the committed
/// segments' manifest entries (Issue #1212).
///
/// `doc_count` is every document the committed segments hold, deleted ones
/// included; `deleted_count` is how many of those are deleted, so their
/// difference is the live count. Both come from persisted state — the
/// manifest records each segment's count whenever its deletions are
/// persisted — instead of deltas accumulated per commit, which drifted on
/// re-upserts, crash-recovery replays and merges. An entry whose count is
/// still unknown (unreadable at open) contributes no deletions, as its
/// readers also treat an unreadable bitmap. No I/O.
pub(crate) fn index_counts(shared: &RwLock<ManifestState>) -> (u64, u64) {
    let guard = shared.read();
    guard
        .segments
        .iter()
        .fold((0, 0), |(docs, deleted), segment| {
            (
                docs + segment.doc_count,
                deleted + segment.deleted_count.unwrap_or(0),
            )
        })
}

/// Hand out the next segment generation ordinal (#1024).
///
/// Taken under the manifest write lock so a flushing writer and a
/// concurrent merge can never mint the same generation — the tie the old
/// scan-derived numbering allowed (a writer surviving `optimize` and the
/// merged segment both computed `max + 1` independently).
pub(crate) fn reserve_generation(shared: &RwLock<ManifestState>) -> u64 {
    let mut guard = shared.write();
    let generation = guard.next_generation;
    guard.next_generation += 1;
    generation
}

/// The generation ordinal encoded in a segment-shaped file name, if any.
///
/// Accepts the two stems this index mints — `segment_<digits>` and
/// `merged_<digits>` — taking the stem before the first `.` so data
/// files, `.delmap`s and per-field `.bkd`s all count.
pub(crate) fn stem_ordinal(file_name: &str) -> Option<u64> {
    let stem = file_name.split('.').next()?;
    let ordinal = stem
        .strip_prefix("segment_")
        .or_else(|| stem.strip_prefix("merged_"))?;
    if ordinal.is_empty() || !ordinal.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    ordinal.parse().ok()
}

/// Derive the generation-counter seed for an opening index (#1024).
///
/// `max(entry generations, surviving file-stem ordinals) + 1`. The file
/// stems are load-bearing: a crash can lose an in-memory reservation
/// while its flushed files survive (a legacy directory the sweep never
/// touches, or a best-effort sweep deletion that failed), and reusing
/// such an ordinal would let the new segment adopt stale foreign
/// per-field `.bkd`s or a stale `.delmap` by name-prefix.
pub(crate) fn derive_next_generation(entries: &[SegmentInfo], files: &[String]) -> u64 {
    let from_entries = entries.iter().map(|s| s.generation + 1).max().unwrap_or(0);
    let from_files = files
        .iter()
        .filter_map(|f| stem_ordinal(f))
        .map(|ordinal| ordinal + 1)
        .max()
        .unwrap_or(0);
    from_entries.max(from_files)
}

/// Insert `info` into `list`, replacing any entry with the same
/// `segment_id`.
///
/// The dedup is what makes a retried publication idempotent: a commit
/// whose manifest save succeeded but whose later ladder step failed
/// re-runs the whole publish, and must not double-add.
pub(crate) fn upsert_entry(list: &mut Vec<SegmentInfo>, info: SegmentInfo) {
    if let Some(existing) = list
        .iter_mut()
        .find(|entry| entry.segment_id == info.segment_id)
    {
        *existing = info;
    } else {
        list.push(info);
    }
}

#[cfg(test)]
mod tests {

    /// `SegmentInfo::deleted_count` (Issue #1212) reads as `None` from an
    /// entry written by an older build, and `None` is not written back.
    #[test]
    fn deleted_count_is_optional_in_the_manifest() {
        let old = r#"{"segment_id":"s","doc_count":3,"min_doc_id":0,"max_doc_id":2,"generation":0,"has_deletions":true,"shard_id":0}"#;
        let info: SegmentInfo = serde_json::from_str(old).unwrap();
        assert_eq!(info.deleted_count, None);
        assert!(
            !serde_json::to_string(&info)
                .unwrap()
                .contains("deleted_count")
        );

        let recorded = SegmentInfo {
            deleted_count: Some(1),
            ..info
        };
        let json = serde_json::to_string(&recorded).unwrap();
        let back: SegmentInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.deleted_count, Some(1));
    }
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn seg(id: &str, generation: u64) -> SegmentInfo {
        SegmentInfo {
            segment_id: id.to_string(),
            doc_count: 1,
            min_doc_id: 0,
            max_doc_id: 0,
            generation,
            has_deletions: false,
            deleted_count: None,
            shard_id: 0,
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        assert!(load(&storage).unwrap().is_none(), "no manifest yet");

        let segments = vec![seg("segment_000001", 1), seg("merged_2", 2)];
        save(&storage, &segments).unwrap();
        assert_eq!(
            load(&storage).unwrap().unwrap(),
            (MANIFEST_VERSION, segments)
        );
    }

    #[test]
    fn upsert_entry_replaces_by_id() {
        let mut list = vec![seg("segment_000001", 1)];
        let mut updated = seg("segment_000001", 1);
        updated.has_deletions = true;
        upsert_entry(&mut list, updated.clone());
        assert_eq!(list, vec![updated], "same id must replace, not append");

        upsert_entry(&mut list, seg("segment_000002", 2));
        assert_eq!(list.len(), 2, "new id must append");
    }

    /// The merge-transition shape must be a delta against the LIVE list:
    /// an entry added between computing a merge's sources and publishing
    /// it survives. A snapshot replacement would silently drop it.
    #[test]
    fn publish_with_applies_deltas_to_the_live_list() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let shared = RwLock::new(ManifestState {
            segments: vec![seg("segment_000001", 1)],
            next_generation: 2,
        });

        // A concurrent publication lands first.
        publish_with(&storage, &shared, |list| {
            upsert_entry(list, seg("segment_000002", 2));
        })
        .unwrap();

        // The merge delta (computed when only segment_000001 existed)
        // drops its source and inserts the merged segment.
        publish_with(&storage, &shared, |list| {
            list.retain(|s| s.segment_id != "segment_000001");
            upsert_entry(list, seg("merged_3", 3));
        })
        .unwrap();

        let ids: Vec<String> = shared
            .read()
            .segments
            .iter()
            .map(|s| s.segment_id.clone())
            .collect();
        assert_eq!(
            ids,
            vec!["segment_000002".to_string(), "merged_3".to_string()],
            "the interleaved publication must survive the merge delta"
        );
    }

    #[test]
    fn failed_save_leaves_memory_and_disk_untouched() {
        // A storage that refuses every write.
        #[derive(Debug)]
        struct ReadOnly(MemoryStorage);
        impl Storage for ReadOnly {
            fn create_output(&self, _n: &str) -> Result<Box<dyn crate::storage::StorageOutput>> {
                Err(crate::LaurusError::storage("read-only"))
            }
            fn create_output_append(
                &self,
                n: &str,
            ) -> Result<Box<dyn crate::storage::StorageOutput>> {
                self.0.create_output_append(n)
            }
            fn open_input(&self, n: &str) -> Result<Box<dyn crate::storage::StorageInput>> {
                self.0.open_input(n)
            }
            fn file_exists(&self, n: &str) -> bool {
                self.0.file_exists(n)
            }
            fn delete_file(&self, n: &str) -> Result<()> {
                self.0.delete_file(n)
            }
            fn rename_file(&self, a: &str, b: &str) -> Result<()> {
                self.0.rename_file(a, b)
            }
            fn list_files(&self) -> Result<Vec<String>> {
                self.0.list_files()
            }
            fn file_size(&self, n: &str) -> Result<u64> {
                self.0.file_size(n)
            }
            fn sync(&self) -> Result<()> {
                Ok(())
            }
            fn metadata(&self, n: &str) -> Result<crate::storage::FileMetadata> {
                self.0.metadata(n)
            }
            fn create_temp_output(
                &self,
                p: &str,
            ) -> Result<(String, Box<dyn crate::storage::StorageOutput>)> {
                self.0.create_temp_output(p)
            }
            fn close(&mut self) -> Result<()> {
                Ok(())
            }
        }

        let storage = ReadOnly(MemoryStorage::new(MemoryStorageConfig::default()));
        let shared = RwLock::new(ManifestState {
            segments: vec![seg("segment_000001", 1)],
            next_generation: 2,
        });
        let err = publish_with(&storage, &shared, |list| {
            upsert_entry(list, seg("segment_000002", 2));
        });
        assert!(err.is_err());
        assert_eq!(
            shared.read().segments.len(),
            1,
            "a failed save must not advance the in-memory manifest"
        );
    }
}
