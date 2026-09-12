//! Inverted index implementation for full-text search.
//!
//! This module provides the core inverted index implementation:
//! - Core data structures (posting lists, term enumeration)
//! - Index creation and management
//! - Writer for building the index
//! - Reader for querying the index
//! - Searcher for executing searches
//! - Segment management and merging
//! - Index maintenance operations
//! - Query types for searching

use std::collections::HashMap;
use std::io::Read;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use serde::{Deserialize, Serialize};

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::error::{LaurusError, Result};
use crate::lexical::core::field::FieldOption;
use crate::lexical::index::LexicalIndex;
use crate::lexical::index::config::InvertedIndexConfig;
use crate::lexical::reader::LexicalIndexReader;
use crate::lexical::search::searcher::LexicalSearcher;
use crate::lexical::writer::LexicalIndexWriter;
use crate::storage::Storage;
#[cfg(not(target_arch = "wasm32"))]
use crate::storage::file::{FileStorage, FileStorageConfig};
use crate::storage::manifest as manifest_io;

pub(crate) mod bmw;
pub(crate) mod compound;
pub mod core;
pub mod parsed_query_cache;
pub(crate) mod per_segment_view;
pub mod posting_cache;
pub mod query_cache;
pub mod reader;
pub mod searcher;
pub mod segment;
pub(crate) mod segment_manifest;
pub mod writer;

use self::reader::{InvertedIndexReader, InvertedIndexReaderConfig};
use self::searcher::InvertedIndexSearcher;
use self::segment::SegmentInfo;
use self::writer::{InvertedIndexWriter, InvertedIndexWriterConfig};

/// Metadata about an inverted index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Version of the index format.
    pub version: u32,

    /// Creation time (seconds since epoch).
    pub created: u64,

    /// Last modified time (seconds since epoch).
    pub modified: u64,

    /// Number of documents indexed.
    pub doc_count: u64,

    /// Generation number for updates.
    pub generation: u64,

    /// Number of deleted documents.
    #[serde(default)]
    pub deleted_count: u64,

    /// Last processed WAL sequence number.
    #[serde(default)]
    pub last_wal_seq: u64,
}

/// Statistics about an inverted index.
#[derive(Debug, Clone)]
pub struct InvertedIndexStats {
    /// Number of documents in the index.
    pub doc_count: u64,

    /// Number of unique terms in the index.
    pub term_count: u64,

    /// Number of segments in the index.
    pub segment_count: u32,

    /// Total size of the index in bytes.
    pub total_size: u64,

    /// Number of deleted documents.
    pub deleted_count: u64,

    /// Last modified time (seconds since epoch).
    pub last_modified: u64,
}

impl Default for IndexMetadata {
    fn default() -> Self {
        let now = crate::util::time::now_secs();

        IndexMetadata {
            version: 1,
            created: now,
            modified: now,
            doc_count: 0,
            generation: 0,
            deleted_count: 0,
            last_wal_seq: 0,
        }
    }
}

/// A concrete inverted index implementation for schema-less lexical indexing.
pub struct InvertedIndex {
    /// The storage backend.
    storage: Arc<dyn Storage>,

    /// Inverted index specific configuration.
    config: InvertedIndexConfig,

    /// Fields added dynamically at runtime via [`add_field()`](Self::add_field).
    /// These are merged with `config.fields` when creating a new writer.
    extra_fields: RwLock<HashMap<String, FieldOption>>,

    /// Whether the index is closed (thread-safe).
    closed: AtomicBool,

    /// Index metadata (thread-safe).
    ///
    /// This in-memory copy is the AUTHORITY over `metadata.json` (#1023):
    /// the handle is cloned into every writer this index constructs (see
    /// [`Self::writer`]), which applies its per-commit deltas under this
    /// lock and persists a snapshot of it. Nothing re-reads the file into
    /// this lock after `open`, so disk can never clobber fresher state.
    metadata: Arc<RwLock<IndexMetadata>>,

    /// The committed segment set, mirroring `segments.json` (#1021).
    ///
    /// Mutated only through
    /// [`segment_manifest::publish_with`] (save-then-swap), so it is never
    /// ahead of the persisted manifest. Shared with writers built by
    /// [`Self::writer`]; writers built through the public
    /// `InvertedIndexWriter::new` get no handle. Deliberately NOT part of
    /// the `Debug` impl: publication holds the write guard across storage
    /// I/O, and a blocking `.read()` in `Debug` would deadlock any error
    /// path that formats the index mid-publish.
    segment_manifest: segment_manifest::SharedSegmentManifest,
}

impl std::fmt::Debug for InvertedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndex")
            .field("storage", &self.storage)
            .field("config", &self.config)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .field("metadata", &*self.metadata.read())
            .finish()
    }
}

impl InvertedIndex {
    /// Create a new index in the given storage.
    pub fn create(storage: Arc<dyn Storage>, config: InvertedIndexConfig) -> Result<Self> {
        let metadata = IndexMetadata::default();

        let index = InvertedIndex {
            storage,
            config,
            extra_fields: RwLock::new(HashMap::new()),
            closed: AtomicBool::new(false),
            metadata: Arc::new(RwLock::new(metadata)),
            segment_manifest: Arc::new(RwLock::new(segment_manifest::ManifestState {
                segments: Vec::new(),
                next_generation: 0,
            })),
        };

        index.write_metadata()?;
        // An empty manifest from birth (#1021): "manifest present" then
        // reliably means "authoritative record exists", and a fresh index
        // is never mistaken for a legacy one.
        segment_manifest::save(index.storage.as_ref(), &[])?;
        Ok(index)
    }

    /// Open an existing index from storage.
    pub fn open(storage: Arc<dyn Storage>, config: InvertedIndexConfig) -> Result<Self> {
        if !storage.file_exists("metadata.json") {
            return Err(LaurusError::index("Index does not exist"));
        }

        let metadata = Self::read_metadata(storage.as_ref())?;

        // Manifest, or legacy migration (#1021). Two legacy shapes read
        // through the committed `.meta` scan instead: no manifest at all
        // (pre-#1021), and a version-1 manifest (written while discovery
        // still ran on the scan — it was never the authority, so it is a
        // hint at best and NEVER a deletion warrant). The migration is in
        // memory only — `open` writes nothing (read-only and WASM storages
        // must keep opening) — and the first mutation persists it at the
        // current version.
        let loaded = segment_manifest::load(storage.as_ref())?;
        let authoritative = matches!(
            &loaded,
            Some((version, _)) if *version >= segment_manifest::AUTHORITATIVE_VERSION
        );
        // One listing serves the lost-manifest guard, the generation-counter
        // seed and the sweep. Taken BEFORE the sweep on purpose: seeding
        // from pre-sweep stems is conservative (a swept ordinal leaves a
        // harmless gap), and it is what covers ordinals whose files survive
        // a failed best-effort deletion.
        let files = storage.list_files()?;
        let segments = match loaded {
            Some((version, segments)) if version >= segment_manifest::AUTHORITATIVE_VERSION => {
                segments
            }
            _ => {
                // Lost-manifest guard (#1024): segment-shaped files with
                // neither a manifest nor any `.meta` to migrate from means
                // the manifest was deleted or lost. Opening anyway would
                // serve a silently empty index, and the first commit would
                // publish a fresh manifest that the NEXT open's sweep
                // enforces by deleting every pre-existing segment —
                // delayed, permanent data loss the WAL cannot cover (the
                // metadata.json checkpoint is intact, so replay skips
                // everything). Refuse loudly instead.
                let has_meta = files
                    .iter()
                    .any(|f| f.ends_with(".meta") && segment_manifest::stem_ordinal(f).is_some());
                let has_segment_files = files
                    .iter()
                    .any(|f| segment_manifest::stem_ordinal(f).is_some());
                if loaded.is_none() && !has_meta && has_segment_files {
                    return Err(LaurusError::index(
                        "segments.json is missing but segment files are present; refusing to                          open — restore the manifest (or remove the segment files) instead of                          losing the segments to the next sweep",
                    ));
                }
                Self::scan_segment_metas_from(storage.as_ref())?
            }
        };
        let next_generation = segment_manifest::derive_next_generation(&segments, &files);

        // Orphan sweep (#1021): reclaim segment files the manifest does not
        // know — crash-orphaned uncommitted flushes, and the leftovers of a
        // merge whose cleanup did not finish. Gated on an authoritative
        // manifest: a legacy or version-1 index is never swept, so nothing
        // is deleted on the word of a record that was never the authority.
        // Best-effort by design: failures (read-only storage included) are
        // ignored, and correctness never depends on the sweep — WAL replay
        // covers anything a crash left unpublished.
        if authoritative {
            Self::sweep_orphans(storage.as_ref(), &segments, &files);
        }

        Ok(InvertedIndex {
            storage,
            config,
            extra_fields: RwLock::new(HashMap::new()),
            closed: AtomicBool::new(false),
            metadata: Arc::new(RwLock::new(metadata)),
            segment_manifest: Arc::new(RwLock::new(segment_manifest::ManifestState {
                segments,
                next_generation,
            })),
        })
    }

    /// Create an index in a directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create_in_dir<P: AsRef<Path>>(dir: P, config: InvertedIndexConfig) -> Result<Self> {
        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::create(storage, config)
    }

    /// Open an index from a directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_dir<P: AsRef<Path>>(dir: P, config: InvertedIndexConfig) -> Result<Self> {
        let storage_config = FileStorageConfig::new(&dir);
        let storage = Arc::new(FileStorage::new(&dir, storage_config)?);
        Self::open(storage, config)
    }

    /// Open or create an index.
    pub fn open_or_create(storage: Arc<dyn Storage>, config: InvertedIndexConfig) -> Result<Self> {
        if storage.file_exists("metadata.json") {
            Self::open(storage, config)
        } else {
            Self::create(storage, config)
        }
    }

    /// Write metadata to storage.
    ///
    /// Atomic and checksummed (#1023). This file is what `open` gates on, so
    /// a torn write does not lose a counter — it stops the whole engine from
    /// opening, vector and document data included, with nothing to recreate
    /// it. Serialization happens under the lock and the write outside it,
    /// because holding a `parking_lot::RwLock` across I/O would deadlock the
    /// moment an error path formatted `self`.
    fn write_metadata(&self) -> Result<()> {
        let metadata_json = {
            let metadata = self.metadata.read();
            serde_json::to_vec(&*metadata)
                .map_err(|e| LaurusError::index(format!("Failed to serialize metadata: {e}")))?
        };

        manifest_io::save_checksummed(self.storage.as_ref(), "metadata.json", None, &metadata_json)
    }

    /// Read metadata from storage.
    pub(crate) fn read_metadata(storage: &dyn Storage) -> Result<IndexMetadata> {
        // Verifies the checksum and refuses a corrupted file rather than
        // reading it as valid; pre-#1023 raw-JSON files still load (#1023).
        match manifest_io::load_checksummed_json::<IndexMetadata>(storage, "metadata.json", None)? {
            Some((metadata, _format)) => Ok(metadata),
            None => Err(LaurusError::index("metadata.json is missing or empty")),
        }
    }

    /// Update metadata and write to storage.
    fn update_metadata(&self) -> Result<()> {
        {
            let mut metadata = self.metadata.write();
            metadata.modified = crate::util::time::now_secs();
        }

        self.write_metadata()
    }

    /// Check if the index is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            Err(LaurusError::index("Index is closed"))
        } else {
            Ok(())
        }
    }

    /// Load the segments a reader may see, i.e. the published ones.
    ///
    /// The writer flushes a segment as soon as its buffer fills, so a
    /// segment can be fully written while its documents are not committed.
    /// Skipping those here is what makes the documented contract hold —
    /// documents become searchable only after `commit()` — instead of
    /// depending on whether a searcher happened to be built in between
    /// (#1017).
    ///
    /// # Returns
    ///
    /// Published segments, ordered by generation.
    fn load_segments(&self) -> Result<Vec<SegmentInfo>> {
        // A pure in-memory read since #1021: the shared manifest mirrors
        // the last successfully persisted `segments.json` (save-then-swap),
        // and every entry in it is committed by construction. Zero storage
        // I/O per reader construction, where the `.meta` scan paid
        // O(segments) opens + parses.
        let mut segments = self.segment_manifest.read().segments.clone();
        segments.sort_by_key(|s| s.generation);
        Ok(segments)
    }

    /// Delete segment-shaped files the manifest does not list (#1021).
    ///
    /// A file participates when the stem before its first `.` is
    /// `segment_` or `merged_` followed by ASCII digits only — the two
    /// shapes this index mints — which also covers their `.delmap`s and
    /// per-field `.bkd`s while rejecting foreign names. A stem present in
    /// `manifest` is live and kept. Deletions are best-effort (`let _`):
    /// the sweep is an optimization, never a correctness requirement.
    ///
    /// Reclaiming a segment whose `.meta` says `committed: true` is logged
    /// at `warn`: it usually means a crash in a publication window (the
    /// manifest wins by design), but it can also mean segments committed
    /// through a standalone `InvertedIndexWriter` into a directory owned
    /// by a manifest-bearing index — a documented misuse.
    fn sweep_orphans(storage: &dyn Storage, manifest: &[SegmentInfo], files: &[String]) {
        for file in files {
            if file == "segments.json.tmp" {
                let _ = storage.delete_file(file);
                continue;
            }
            let Some(stem) = file.split('.').next() else {
                continue;
            };
            let ordinal = stem
                .strip_prefix("segment_")
                .or_else(|| stem.strip_prefix("merged_"));
            let Some(ordinal) = ordinal else {
                continue;
            };
            if ordinal.is_empty() || !ordinal.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            if manifest.iter().any(|s| s.segment_id == stem) {
                continue;
            }
            // No content inspection (#1024): `.meta` files are legacy
            // artifacts, and the manifest is the sole authority — a file
            // the manifest does not list is an orphan by definition
            // (a crash in a publication window, an unfinished merge
            // cleanup, or a segment written by a standalone writer).
            log::debug!("sweeping unreferenced segment file {file}");
            let _ = storage.delete_file(file);
        }
    }

    /// The legacy-migration scan (#1021/#1024): read every pre-manifest
    /// `.meta` on storage — the ONE sanctioned `.meta` read left, used by
    /// `open` when no authoritative manifest exists. Unpublished records
    /// (`committed: false`, written by pre-#1024 builds for
    /// flushed-but-uncommitted segments) are skipped.
    fn scan_segment_metas_from(storage: &dyn Storage) -> Result<Vec<SegmentInfo>> {
        /// The on-disk shape of a pre-#1024 `.meta`, parsed only here.
        /// `committed` defaulted to true for records older than #1017.
        #[derive(serde::Deserialize)]
        struct LegacyMetaRecord {
            segment_id: String,
            doc_count: u64,
            min_doc_id: u64,
            max_doc_id: u64,
            generation: u64,
            has_deletions: bool,
            shard_id: u16,
            #[serde(default = "legacy_committed_default")]
            committed: bool,
        }
        fn legacy_committed_default() -> bool {
            true
        }

        let files = storage.list_files()?;
        let mut segments = Vec::new();

        for file in &files {
            // Both freshly flushed segments (`segment_*`) and segments produced
            // by a merge (`merged_*`, Issue #754) are discovered here.
            if (file.starts_with("segment_") || file.starts_with("merged_"))
                && file.ends_with(".meta")
            {
                let mut input = storage.open_input(file)?;
                let mut data = Vec::new();
                Read::read_to_end(&mut input, &mut data)?;

                let record: LegacyMetaRecord = serde_json::from_slice(&data).map_err(|e| {
                    LaurusError::index(format!("Failed to parse segment metadata: {e}"))
                })?;
                if !record.committed {
                    continue;
                }
                segments.push(SegmentInfo {
                    segment_id: record.segment_id,
                    doc_count: record.doc_count,
                    min_doc_id: record.min_doc_id,
                    max_doc_id: record.max_doc_id,
                    generation: record.generation,
                    has_deletions: record.has_deletions,
                    shard_id: record.shard_id,
                });
            }
        }

        Ok(segments)
    }

    /// Force-merge every current segment into a single new segment (Issue
    /// #754), the classic `optimize()` / force-merge semantics.
    ///
    /// Discovers the current segments, merges them with the (correct, typed)
    /// [`MergeEngine`](self::segment::merge_engine::MergeEngine), rewrites the
    /// merged segment's metadata generation so it sorts as the newest segment,
    /// and deletes the now-merged source segments' files so segment discovery
    /// ([`Self::load_segments`]) sees only the merged result. A no-op when
    /// fewer than two segments exist.
    fn force_merge_all(&self) -> Result<()> {
        let segments = self.load_segments()?;
        if segments.len() < 2 {
            // Zero or one segment: nothing to compact.
            return Ok(());
        }
        // The merged segment must sort as the newest, so its generation is
        // one past the highest on storage — including segments flushed but
        // not yet published, which a merge must not collide with (#1017).
        let next_generation = self.next_generation()?;
        self.merge_segment_set(&segments, next_generation)
    }

    /// Auto-merge implementation behind the [`LexicalIndex::maybe_merge`] hook
    /// run after each commit (Issue #755).
    ///
    /// Keeps the segment count bounded without a manual
    /// [`optimize()`](LexicalIndex::optimize): when the number of segments
    /// exceeds [`InvertedIndexConfig::max_segments`], the smallest
    /// [`merge_factor`](InvertedIndexConfig::merge_factor) segments are merged
    /// into one (Lucene-style "merge small segments first"). A single merge is
    /// performed per call; repeated commits converge the count. Cheap when no
    /// merge is needed (a segment count check). Disable by raising
    /// `max_segments`.
    fn auto_merge(&self) -> Result<()> {
        let segments = self.load_segments()?;
        if segments.len() <= self.config.max_segments as usize {
            // Under the threshold: nothing to do.
            return Ok(());
        }

        // Merge the smallest `merge_factor` segments (small-first keeps merge
        // cost low and bounds per-commit latency).
        let mut by_size: Vec<(SegmentInfo, u64)> = segments
            .iter()
            .map(|s| (s.clone(), self.segment_size_bytes(&s.segment_id)))
            .collect();
        by_size.sort_by_key(|(_, size)| *size);

        let take = (self.config.merge_factor as usize).clamp(2, segments.len());
        let subset: Vec<SegmentInfo> = by_size.into_iter().take(take).map(|(s, _)| s).collect();

        let next_generation = self.next_generation()?;
        self.merge_segment_set(&subset, next_generation)
    }

    /// Reserve the generation for a newly merged segment (#1024).
    ///
    /// Taken from the shared in-memory counter, which every flush also
    /// reserves from — so a merge can no longer collide with a segment
    /// flushed but not yet committed (#1017), nor tie with a surviving
    /// writer's next flush (both used to compute `max + 1` independently).
    /// A reservation consumed by a merge that later fails leaves a gap in
    /// the numbering, which is harmless.
    ///
    /// # Returns
    ///
    /// The generation a newly merged segment should take.
    fn next_generation(&self) -> Result<u64> {
        Ok(segment_manifest::reserve_generation(&self.segment_manifest))
    }

    /// Merge a set of source segments into a single new segment.
    ///
    /// Shared by [`Self::force_merge_all`] (all segments) and
    /// [`Self::maybe_merge`] (a policy-selected subset). Runs the (correct,
    /// typed) [`MergeEngine`](self::segment::merge_engine::MergeEngine), rewrites
    /// the merged segment's generation to `next_generation` so it sorts as the
    /// newest, and deletes the source segments (their `.meta` first, so they
    /// drop out of `.meta` file-scan discovery before their now-orphaned data
    /// files are removed — minimizing any window in which a document could be
    /// seen in both a source and the merged segment). A no-op for fewer than
    /// two sources.
    /// Every currently-declared field's `doc_values` setting, merging the
    /// initial schema (`config.fields`) with fields added at runtime
    /// (`extra_fields`) -- the same union [`Self::writer`] builds for
    /// `InvertedIndexWriterConfig::fields`. Feeds
    /// [`MergeConfig::field_doc_values`](self::segment::merge_engine::MergeConfig::field_doc_values)
    /// so a merge or field rebuild resolves DocValues from the CURRENT
    /// schema first (Issue #1047), not from whatever a source segment
    /// happens to have on disk.
    fn current_field_doc_values(&self) -> HashMap<String, bool> {
        let mut out = HashMap::new();
        for (name, option) in &self.config.fields {
            if let Some(dv) = option.doc_values() {
                out.insert(name.clone(), dv);
            }
        }
        // Runtime-added fields win over the initial schema on a name
        // clash, mirroring `writer()`'s `fields.extend(extra_fields)`.
        for (name, option) in self.extra_fields.read().iter() {
            if let Some(dv) = option.doc_values() {
                out.insert(name.clone(), dv);
            }
        }
        out
    }

    fn merge_segment_set(&self, sources: &[SegmentInfo], next_generation: u64) -> Result<()> {
        use self::segment::merge_engine::{MergeConfig, MergeEngine};
        use self::segment::{ManagedSegmentInfo, MergeCandidate, MergeStrategy};

        if sources.len() < 2 {
            return Ok(());
        }

        let managed: Vec<ManagedSegmentInfo> = sources
            .iter()
            .map(|info| {
                let mut mi = ManagedSegmentInfo::new(info.clone());
                mi.size_bytes = self.segment_size_bytes(&info.segment_id);
                mi
            })
            .collect();
        let candidate = MergeCandidate {
            segments: sources.iter().map(|s| s.segment_id.clone()).collect(),
            priority: 1.0,
            estimated_size: 0,
            strategy: MergeStrategy::SizeBased,
        };

        let engine = MergeEngine::new(
            MergeConfig {
                use_compound: self.config.use_compound,
                field_doc_values: self.current_field_doc_values(),
                default_doc_values: self.config.store_doc_values,
                ..MergeConfig::default()
            },
            self.storage.clone(),
        );
        let result = engine.merge_segments(&candidate, &managed, next_generation)?;

        // Publish the merge transition as ONE manifest write (#1021): drop
        // the sources and insert the merged segment with its final
        // generation, applied as a delta to the LIVE list — never a
        // snapshot replacement, which would lose a mutation that landed
        // since this merge computed its sources. Everything after this is
        // advisory (`.meta`) or physical cleanup, and every crash window
        // below resolves in the manifest's favor.
        let mut merged_info = result.new_segment.segment_info.clone();
        merged_info.generation = next_generation;
        let source_ids: Vec<String> = sources.iter().map(|s| s.segment_id.clone()).collect();
        segment_manifest::publish_with(self.storage.as_ref(), &self.segment_manifest, |list| {
            list.retain(|entry| !source_ids.contains(&entry.segment_id));
            segment_manifest::upsert_entry(list, merged_info);
        })?;

        // Physical cleanup only (#1024): the manifest write above already
        // removed the sources from discovery atomically, so file deletion
        // order no longer matters — anything a crash leaves behind is an
        // unreferenced orphan the next open's sweep reclaims.
        for info in sources {
            self.delete_segment_files(&info.segment_id)?;
        }

        Ok(())
    }

    /// Sum the on-disk size of every file belonging to `segment_id`.
    fn segment_size_bytes(&self, segment_id: &str) -> u64 {
        let prefix = format!("{segment_id}.");
        self.storage
            .list_files()
            .map(|files| {
                files
                    .iter()
                    .filter(|f| f.starts_with(&prefix))
                    .map(|f| self.storage.metadata(f).map(|m| m.size).unwrap_or(0))
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Delete every file belonging to `segment_id`.
    fn delete_segment_files(&self, segment_id: &str) -> Result<()> {
        let prefix = format!("{segment_id}.");
        let files = self.storage.list_files()?;
        for file in files.iter().filter(|f| f.starts_with(&prefix)) {
            self.storage.delete_file(file)?;
        }
        Ok(())
    }

    /// Check if an index exists in the given directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn exists_in_dir<P: AsRef<Path>>(dir: P) -> bool {
        let metadata_path = dir.as_ref().join("metadata.json");
        metadata_path.exists()
    }

    /// Delete an index from the given directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn delete_in_dir<P: AsRef<Path>>(dir: P) -> Result<()> {
        let storage_config = FileStorageConfig::new(&dir);
        let storage = FileStorage::new(&dir, storage_config)?;

        for file in storage.list_files()? {
            storage.delete_file(&file)?;
        }

        Ok(())
    }

    /// List all files in the index.
    pub fn list_files(&self) -> Result<Vec<String>> {
        self.check_closed()?;
        self.storage.list_files()
    }

    /// Returns the last WAL (Write-Ahead Log) sequence number recorded in the index metadata.
    ///
    /// Also exposed through the [`LexicalIndex`] trait (#1023): before that
    /// override existed, calls through `Box<dyn LexicalIndex>` resolved to
    /// the trait default `0`, making the checkpoint invisible to
    /// [`LexicalStore`](crate::lexical::store::LexicalStore) and
    /// `Engine::recover`.
    ///
    /// # Returns
    ///
    /// The last WAL sequence number as a `u64`.
    pub fn last_wal_seq(&self) -> u64 {
        self.metadata.read().last_wal_seq
    }

    /// Sets the last WAL (Write-Ahead Log) sequence number in the index metadata
    /// and persists the updated metadata to storage.
    ///
    /// # Arguments
    ///
    /// * `seq` - The new WAL sequence number to record.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the index is closed or the metadata write fails.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] if the index has been closed
    /// or if persisting the metadata fails.
    pub fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
        self.check_closed()?;
        {
            let mut metadata = self.metadata.write();
            metadata.last_wal_seq = seq;
        }
        self.update_metadata()
    }
}

impl LexicalIndex for InvertedIndex {
    fn reader(&self) -> Result<Arc<dyn LexicalIndexReader>> {
        self.check_closed()?;

        let segments = self.load_segments()?;

        // Use analyzer from index config. The query/filter cache capacity must
        // be set explicitly here: `InvertedIndexReaderConfig::default()` would
        // otherwise mask the value configured on the index (Issue #578).
        let reader_config = InvertedIndexReaderConfig {
            analyzer: self.config.analyzer.clone(),
            query_filter_cache_capacity: self.config.query_filter_cache_capacity,
            ..InvertedIndexReaderConfig::default()
        };

        let reader = InvertedIndexReader::new(segments, self.storage.clone(), reader_config)?;
        Ok(Arc::new(reader))
    }

    fn writer(&self) -> Result<Box<dyn LexicalIndexWriter>> {
        self.check_closed()?;

        // Merge base config fields with dynamically added fields.
        let mut fields = self.config.fields.clone();
        fields.extend(
            self.extra_fields
                .read()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone())),
        );

        // Use analyzer and shard_id from index config
        let writer_config = InvertedIndexWriterConfig {
            analyzer: self.config.analyzer.clone(),
            shard_id: self.config.shard_id,
            fields,
            use_compound: self.config.use_compound,
            store_term_positions: self.config.store_term_vectors,
            store_doc_values: self.config.store_doc_values,
            ..Default::default()
        };
        // Hand the writer the shared metadata and manifest handles
        // (#1023 / #1021). This is the ONLY constructor that does: writers
        // built through the public `InvertedIndexWriter::new` — the merge
        // engine's internal replay writer included — get neither handle
        // and therefore can touch neither `metadata.json` nor
        // `segments.json`.
        let writer = InvertedIndexWriter::with_shared_state(
            self.storage.clone(),
            writer_config,
            Arc::clone(&self.metadata),
            Arc::clone(&self.segment_manifest),
        )?;
        Ok(Box::new(writer))
    }

    fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    fn stats(&self) -> Result<InvertedIndexStats> {
        self.check_closed()?;

        let metadata = self.metadata.read();
        Ok(InvertedIndexStats {
            doc_count: metadata.doc_count,
            term_count: 0,
            segment_count: 0,
            total_size: 0,
            deleted_count: metadata.deleted_count,
            last_modified: metadata.modified,
        })
    }

    fn optimize(&self) -> Result<()> {
        self.check_closed()?;
        self.force_merge_all()?;
        self.update_metadata()?;
        Ok(())
    }

    fn maybe_merge(&self) -> Result<()> {
        self.check_closed()?;
        self.auto_merge()
    }

    fn refresh(&self) -> Result<()> {
        // Deliberately does NOT re-read `metadata.json` (#1023): the
        // in-memory copy is the authority — every writer this index hands
        // out applies its commit deltas directly to the shared lock, so the
        // lock is always at least as fresh as the file. Re-reading used to
        // be how `LexicalStore::commit` picked up the writer's values, and
        // it was also how the merge engine's inflated counts were laundered
        // back into memory.
        self.check_closed()?;
        Ok(())
    }

    fn last_wal_seq(&self) -> u64 {
        InvertedIndex::last_wal_seq(self)
    }

    fn set_last_wal_seq(&self, seq: u64) -> Result<()> {
        InvertedIndex::set_last_wal_seq(self, seq)
    }

    fn searcher(&self) -> Result<Box<dyn LexicalSearcher>> {
        self.check_closed()?;
        let reader = self.reader()?;
        let searcher = InvertedIndexSearcher::from_arc(reader)
            .with_default_fields(self.config.default_fields.clone())
            .with_parsed_query_cache_capacity(self.config.parsed_query_cache_capacity);
        Ok(Box::new(searcher))
    }

    fn default_fields(&self) -> Result<Vec<String>> {
        Ok(self.config.default_fields.clone())
    }

    fn add_field(&self, name: &str, option: FieldOption) -> Result<()> {
        // Check for duplicates in both base config and extra fields.
        if self.config.fields.contains_key(name) || self.extra_fields.read().contains_key(name) {
            return Err(LaurusError::invalid_argument(format!(
                "Field '{name}' already exists in the lexical index"
            )));
        }
        self.extra_fields.write().insert(name.to_string(), option);
        Ok(())
    }

    fn delete_field(&self, name: &str) -> Result<()> {
        // Only dynamically added fields (in extra_fields) can be removed at
        // the index level. Fields from the initial config remain in the
        // underlying index data but will be hidden from the engine-level schema.
        self.extra_fields.write().remove(name);
        Ok(())
    }

    fn rebuild_field(
        &self,
        name: &str,
        option: FieldOption,
        analyzer: Option<Arc<dyn Analyzer>>,
    ) -> Result<()> {
        self.check_closed()?;

        // #1083: `option` is the field's NEW setting. For anything other
        // than `Text`, term positions fall back to the index-wide default,
        // matching `InvertedIndexWriterConfig::stores_term_positions`'s
        // 3-level resolution.
        let target_term_vectors = match &option {
            FieldOption::Text(text_option) => text_option.term_vectors,
            _ => self.config.store_term_vectors,
        };
        // #1047: same idea for DocValues -- `option.doc_values()` is
        // `None` only for `Bytes` (which has no such flag), so this falls
        // back to the index-wide default in exactly that one case.
        let target_doc_values = option.doc_values().unwrap_or(self.config.store_doc_values);

        let segments = self.load_segments()?;
        if !segments.is_empty() {
            use self::segment::ManagedSegmentInfo;
            use self::segment::merge_engine::{MergeConfig, MergeEngine};

            let managed: Vec<ManagedSegmentInfo> = segments
                .iter()
                .map(|info| {
                    let mut mi = ManagedSegmentInfo::new(info.clone());
                    mi.size_bytes = self.segment_size_bytes(&info.segment_id);
                    mi
                })
                .collect();

            // Reserve one fresh generation per source segment up front, so
            // every rebuilt segment has a final ID before any I/O runs —
            // the merge engine writes to `new_segment_ids` directly, no
            // renumbering needed after the fact.
            let mut new_segment_ids = Vec::with_capacity(segments.len());
            for _ in &segments {
                let generation = self.next_generation()?;
                new_segment_ids.push(format!("merged_{generation}"));
            }

            let engine = MergeEngine::new(
                MergeConfig {
                    use_compound: self.config.use_compound,
                    field_doc_values: self.current_field_doc_values(),
                    default_doc_values: self.config.store_doc_values,
                    ..MergeConfig::default()
                },
                self.storage.clone(),
            );
            // Nothing is published yet: a failure here leaves every source
            // segment and the manifest completely untouched (Issue #1081's
            // acceptance criterion). Only a partially-written new segment
            // file might exist on disk, which is not referenced by any
            // manifest and is swept up as an orphan on next open, the same
            // as an aborted `perform_merge`.
            let results = engine.rebuild_field_across_segments(
                &managed,
                name,
                analyzer.as_ref(),
                target_term_vectors,
                target_doc_values,
                &new_segment_ids,
            )?;

            // Publish the whole batch as ONE manifest write (mirrors
            // `merge_segment_set`): drop every source and insert every
            // rebuilt segment, each keeping the generation its own ID
            // already encodes.
            let mut new_infos: Vec<SegmentInfo> = Vec::with_capacity(results.len());
            for (result, new_segment_id) in results.into_iter().zip(&new_segment_ids) {
                let mut info = result.new_segment.segment_info;
                if let Some(generation) = segment_manifest::stem_ordinal(new_segment_id) {
                    info.generation = generation;
                }
                new_infos.push(info);
            }
            let source_ids: Vec<String> = segments.iter().map(|s| s.segment_id.clone()).collect();
            segment_manifest::publish_with(
                self.storage.as_ref(),
                &self.segment_manifest,
                |list| {
                    list.retain(|entry| !source_ids.contains(&entry.segment_id));
                    for info in new_infos {
                        segment_manifest::upsert_entry(list, info);
                    }
                },
            )?;

            // Physical cleanup only: the manifest write above already
            // removed the sources from discovery atomically.
            for info in &segments {
                self.delete_segment_files(&info.segment_id)?;
            }
        }

        // Update the field's option — same "extra_fields overrides base
        // config" mechanism `add_field`/`writer()` already rely on (#1081:
        // this also covers a field originally declared in the base config,
        // which `add_field` cannot touch since it rejects duplicates).
        self.extra_fields.write().insert(name.to_string(), option);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::core::document::Document;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use std::sync::Arc;

    #[allow(dead_code)]
    fn create_test_document(title: &str, body: &str) -> Document {
        Document::builder()
            .add_text("title", title)
            .add_text("body", body)
            .build()
    }

    #[test]
    fn test_inverted_index_writer_creation() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let writer = InvertedIndexWriter::new(storage, config).unwrap();

        assert_eq!(writer.pending_docs(), 0);
        assert_eq!(writer.stats().docs_added, 0);
    }

    #[test]
    fn test_add_document() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();
        let doc = create_test_document("Test Title", "This is test content");

        writer.add_document(doc).unwrap();

        assert_eq!(writer.pending_docs(), 1);
        assert_eq!(writer.stats().docs_added, 1);
        assert!(writer.stats().unique_terms > 0);
    }

    #[test]
    fn test_auto_flush() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig {
            max_buffered_docs: 2,
            ..Default::default()
        };

        let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();

        // Add first document
        writer
            .add_document(create_test_document("Doc 1", "Content 1"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 1);

        // Add second document - should trigger flush
        writer
            .add_document(create_test_document("Doc 2", "Content 2"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 0); // Flushed
        assert_eq!(writer.stats().segments_created, 1);

        // Check that files were created
        let files = storage.list_files().unwrap();
        assert!(files.iter().any(|f| f.contains("segment_000000")));
    }

    /// #557: the flush memory budget must account for a document's actual
    /// buffered size, not a flat per-document constant.
    ///
    /// `estimate_memory_usage` charged 1 KB per document plus 256 bytes per
    /// distinct term, so a document carrying thousands of BKD points — a
    /// multi-valued numeric or geo field — slipped past `max_buffer_memory`
    /// entirely. The repeated values keep the term vocabulary small and
    /// constant, so the term half of the old estimate cannot compensate:
    /// only the point/term *instances* grow, and those were uncounted.
    #[test]
    fn flush_budget_counts_multi_valued_point_data() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig {
            // Effectively disable the count trigger so only the memory
            // budget can fire.
            max_buffered_docs: 100_000,
            max_buffer_memory: 1024 * 1024,
            ..Default::default()
        };

        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();

        // Each document carries 2000 points drawn from a 100-value
        // vocabulary: ~64 KB of real point data per document, while the
        // distinct-term count stays pinned at 100 for the whole run.
        let values: Vec<i64> = (0..2000).map(|i| i % 100).collect();
        for _ in 0..40 {
            let doc = Document::builder()
                .add_field(
                    "readings",
                    crate::data::DataValue::Int64Array(values.clone()),
                )
                .build();
            writer.add_document(doc).unwrap();
        }

        // 40 documents x ~64 KB is far past the 1 MB budget, so the writer
        // must have flushed at least once instead of buffering all of them.
        assert!(
            writer.stats().segments_created >= 1,
            "the memory budget must fire on point-heavy documents: \
             {} docs still buffered, no segment flushed",
            writer.pending_docs()
        );
    }

    #[test]
    fn test_commit() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();

        writer
            .add_document(create_test_document("Test", "Content"))
            .unwrap();
        writer.commit().unwrap();

        assert_eq!(writer.pending_docs(), 0);

        // A standalone writer flushes data files; it registers them
        // nowhere (#1024 — index.meta is gone, and only an index-owned
        // writer publishes into segments.json).
        let files = storage.list_files().unwrap();
        assert!(
            files
                .iter()
                .any(|f| f.starts_with("segment_") && f.ends_with(".cfs")),
            "the default layout is the compound container (#554)"
        );
    }

    #[test]
    fn test_rollback() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();

        writer
            .add_document(create_test_document("Test", "Content"))
            .unwrap();
        assert_eq!(writer.pending_docs(), 1);

        writer.rollback().unwrap();
        assert_eq!(writer.pending_docs(), 0);
        assert_eq!(writer.stats().docs_added, 1); // Stats don't rollback
    }

    #[test]
    fn test_multiple_field_types() {
        // Schema-less mode: fields are inferred from document
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexWriterConfig::default();

        let mut writer = InvertedIndexWriter::new(storage, config).unwrap();

        let doc = Document::builder()
            .add_text("title", "Test Document")
            .add_text("id", "doc1")
            .add_float("count", 42.0)
            .build();

        writer.add_document(doc).unwrap();
        writer.commit().unwrap();

        assert_eq!(writer.stats().docs_added, 1);
        assert!(writer.stats().unique_terms >= 3); // At least title, id, count fields
    }
}
