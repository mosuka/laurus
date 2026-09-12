//! Merge engine for combining segments efficiently.
//!
//! This module provides the core functionality for merging multiple segments
//! into a single optimized segment with proper handling of deletions and updates.

use std::collections::HashMap;
use std::sync::Arc;

use ahash::AHashMap;
use roaring::RoaringTreemap;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::error::{LaurusError, Result};
use crate::lexical::core::analyzed::{AnalyzedDocument, AnalyzedTerm};
use crate::lexical::index::inverted::reader::{InvertedIndexReader, SegmentReader};
use crate::lexical::index::inverted::segment::SegmentInfo;
use crate::lexical::index::inverted::segment::{ManagedSegmentInfo, MergeCandidate, MergeStrategy};
use crate::lexical::index::inverted::writer::{
    InvertedIndexWriter, InvertedIndexWriterConfig, analyze_field_value,
};
use crate::lexical::index::structures::aabb::AABB;
use crate::lexical::index::structures::visitor::{CellRelation, IntersectVisitor};
use crate::lexical::reader::LexicalIndexReader;
use crate::storage::Storage;

/// Configuration for merge operations.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Write the merged segment as a compound `.cfs` container (#554).
    ///
    /// Set from the owning index's `use_compound`, so the merged output
    /// follows the same layout as fresh flushes.
    pub use_compound: bool,

    /// Maximum memory usage during merge (in bytes).
    pub max_memory_mb: u64,

    /// Number of documents to process in each batch.
    pub batch_size: usize,

    /// Enable compression during merge.
    pub enable_compression: bool,

    /// Remove deleted documents during merge.
    pub remove_deleted_docs: bool,

    /// Sort documents by ID during merge for better locality.
    pub sort_by_doc_id: bool,

    /// Verify integrity after merge.
    pub verify_after_merge: bool,

    /// The current schema's per-field `doc_values` setting, keyed by
    /// field name (Issue #1047).
    ///
    /// Unlike term positions (which are detected from what a source
    /// segment already has on disk, since a discarded position cannot be
    /// recovered -- see [`Self::default_doc_values`]'s doc comment for
    /// why this is deliberately different), DocValues columns can always
    /// be regenerated from `stored_fields`. So the CURRENT schema wins
    /// here: a field this map declares is written (or not) according to
    /// the schema regardless of what any source segment happened to have.
    /// Only a field this map does *not* mention falls back to detecting
    /// each source segment's existing column.
    ///
    /// Populated by the caller from the owning index's `config.fields` +
    /// `extra_fields` (`InvertedIndex::merge_segment_set` /
    /// `InvertedIndex::rebuild_field`); empty for a bare
    /// `MergeConfig::default()`, which makes every field fall through to
    /// pure detection, then [`Self::default_doc_values`].
    pub field_doc_values: HashMap<String, bool>,

    /// Index-wide default for whether a field's value is written to
    /// DocValues, used only for a field neither `field_doc_values` nor
    /// per-segment detection resolves (i.e. no source segment ever had a
    /// candidate value for it at all -- so this is rarely, if ever,
    /// actually consulted). Mirrors
    /// [`InvertedIndexConfig::store_doc_values`](crate::lexical::index::config::InvertedIndexConfig::store_doc_values).
    pub default_doc_values: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        MergeConfig {
            use_compound: crate::lexical::index::inverted::compound::default_use_compound(),
            max_memory_mb: 256,
            batch_size: 10000,
            enable_compression: true,
            remove_deleted_docs: true,
            sort_by_doc_id: true,
            verify_after_merge: true,
            field_doc_values: HashMap::new(),
            default_doc_values: true,
        }
    }
}

/// Statistics about a merge operation.
#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    /// Number of segments merged.
    pub segments_merged: usize,

    /// Number of documents processed.
    pub docs_processed: u64,

    /// Number of deleted documents removed.
    pub deleted_docs_removed: u64,

    /// Size before merge (in bytes).
    pub size_before: u64,

    /// Size after merge (in bytes).
    pub size_after: u64,

    /// Time taken for merge (in milliseconds).
    pub merge_time_ms: u64,

    /// Compression ratio achieved.
    pub compression_ratio: f64,

    /// Terms merged.
    pub terms_merged: u64,

    /// Postings merged.
    pub postings_merged: u64,

    /// Shard ID for the merged segment.
    pub shard_id: u16,
}

impl MergeStats {
    /// Calculate space savings percentage.
    pub fn space_savings(&self) -> f64 {
        if self.size_before == 0 {
            0.0
        } else {
            ((self.size_before - self.size_after) as f64 / self.size_before as f64) * 100.0
        }
    }
}

/// Result of a merge operation.
#[derive(Debug)]
pub struct MergeResult {
    /// Information about the new merged segment.
    pub new_segment: ManagedSegmentInfo,

    /// Statistics about the merge operation.
    pub stats: MergeStats,

    /// File paths of the new segment.
    pub file_paths: Vec<String>,
}

/// Core merge engine for segment operations (schema-less mode).
#[derive(Debug)]
pub struct MergeEngine {
    /// Configuration for merge operations.
    config: MergeConfig,

    /// Storage backend.
    storage: Arc<dyn Storage>,
}

impl MergeEngine {
    /// Create a new merge engine (schema-less mode).
    pub fn new(config: MergeConfig, storage: Arc<dyn Storage>) -> Self {
        MergeEngine { config, storage }
    }

    /// Merge segments according to the merge candidate.
    pub fn merge_segments(
        &self,
        candidate: &MergeCandidate,
        segments: &[ManagedSegmentInfo],
        next_generation: u64,
    ) -> Result<MergeResult> {
        let start_millis = crate::util::time::now_millis();

        // Filter segments to merge
        let segments_to_merge: Vec<_> = segments
            .iter()
            .filter(|seg| candidate.segments.contains(&seg.segment_info.segment_id))
            .collect();

        if segments_to_merge.is_empty() {
            return Err(LaurusError::index("No segments found to merge"));
        }

        // Create new segment ID
        let new_segment_id = format!("merged_{next_generation}");

        // Initialize merge statistics
        let mut stats = MergeStats {
            segments_merged: segments_to_merge.len(),
            size_before: segments_to_merge.iter().map(|s| s.size_bytes).sum(),
            ..Default::default()
        };

        // Perform merge based on strategy
        let merge_result = match candidate.strategy {
            MergeStrategy::SizeBased => self.merge_by_size(&segments_to_merge, &new_segment_id)?,
            MergeStrategy::DeletionBased => {
                self.merge_by_deletion(&segments_to_merge, &new_segment_id)?
            }
            MergeStrategy::TimeBased => self.merge_by_time(&segments_to_merge, &new_segment_id)?,
            MergeStrategy::Balanced => self.merge_balanced(&segments_to_merge, &new_segment_id)?,
        };

        // Calculate final statistics
        let end_millis = crate::util::time::now_millis();
        stats.merge_time_ms = end_millis.saturating_sub(start_millis);

        stats.size_after = merge_result.new_segment.size_bytes;
        stats.compression_ratio = if stats.size_before > 0 {
            stats.size_after as f64 / stats.size_before as f64
        } else {
            1.0
        };

        // Update merge result stats
        let mut final_result = merge_result;
        final_result.stats = stats;

        // Verify merge if configured
        if self.config.verify_after_merge {
            self.verify_merged_segment(&final_result.new_segment)?;
        }

        Ok(final_result)
    }

    /// Merge segments prioritizing size efficiency.
    fn merge_by_size(
        &self,
        segments: &[&ManagedSegmentInfo],
        new_segment_id: &str,
    ) -> Result<MergeResult> {
        // Sort segments by size (smallest first for better merging efficiency)
        let mut sorted_segments = segments.to_vec();
        sorted_segments.sort_by_key(|s| s.size_bytes);

        self.perform_merge(&sorted_segments, new_segment_id)
    }

    /// Merge segments prioritizing deletion removal.
    fn merge_by_deletion(
        &self,
        segments: &[&ManagedSegmentInfo],
        new_segment_id: &str,
    ) -> Result<MergeResult> {
        // Sort by deletion ratio (highest first for better compaction)
        let mut sorted_segments = segments.to_vec();
        sorted_segments.sort_by(|a, b| b.deletion_ratio().total_cmp(&a.deletion_ratio()));

        self.perform_merge(&sorted_segments, new_segment_id)
    }

    /// Merge segments prioritizing age.
    fn merge_by_time(
        &self,
        segments: &[&ManagedSegmentInfo],
        new_segment_id: &str,
    ) -> Result<MergeResult> {
        // Sort by creation time (oldest first)
        let mut sorted_segments = segments.to_vec();
        sorted_segments.sort_by_key(|s| s.created_at);

        self.perform_merge(&sorted_segments, new_segment_id)
    }

    /// Balanced merge considering multiple factors.
    fn merge_balanced(
        &self,
        segments: &[&ManagedSegmentInfo],
        new_segment_id: &str,
    ) -> Result<MergeResult> {
        // Calculate composite score for each segment
        let mut scored_segments: Vec<_> = segments
            .iter()
            .map(|seg| {
                let size_score = 1.0 / (seg.size_bytes as f64 + 1.0); // Prefer smaller
                let deletion_score = seg.deletion_ratio() * 2.0; // Prefer high deletion
                let age_score = 1.0 / (seg.created_at as f64 + 1.0); // Prefer older

                let composite_score = size_score + deletion_score + age_score;
                (*seg, composite_score)
            })
            .collect();

        // Sort by composite score (highest first)
        scored_segments.sort_by(|a, b| b.1.total_cmp(&a.1));

        let sorted_segments: Vec<_> = scored_segments.into_iter().map(|(seg, _)| seg).collect();

        self.perform_merge(&sorted_segments, new_segment_id)
    }

    /// Core merge implementation.
    fn perform_merge(
        &self,
        segments: &[&ManagedSegmentInfo],
        new_segment_id: &str,
    ) -> Result<MergeResult> {
        let mut stats = MergeStats {
            segments_merged: segments.len(),
            shard_id: segments
                .first()
                .map(|s| s.segment_info.shard_id)
                .unwrap_or(0),
            ..Default::default()
        };

        // Reconstruct every live document's analyzed form from each source
        // segment's postings + stored fields (no re-tokenization; #753). The
        // postings are the source of truth for the inverted index, so
        // index-only (non-stored) fields are preserved, and original doc_ids
        // are kept (they encode the shard and are referenced by deletion
        // bitmaps / external-id maps).
        //
        // `docs` keyed by doc_id with `order` tracking first-seen order makes
        // doc_id collisions across segments (an update re-wrote a doc in a
        // newer segment) resolve to the last-processed version.
        let mut order: Vec<u64> = Vec::new();
        let mut docs: AHashMap<u64, AnalyzedDocument> = AHashMap::new();
        // Whether the source segments stored term positions, per field;
        // detected from the first posting seen for each field so the merged
        // segment reproduces every field's positions state independently
        // (#1083).
        let mut positions_by_field: HashMap<String, bool> = HashMap::new();
        // Whether the merged segment should write a DocValues column, per
        // field (#1047). Seeded from the CURRENT schema so it wins over
        // whatever a source segment happens to have on disk; a field the
        // schema does not mention falls back to detection inside
        // `reconstruct_segment` (`.entry().or_insert_with()` only fills
        // gaps `field_doc_values` left open).
        let mut doc_values_by_field: HashMap<String, bool> = self.config.field_doc_values.clone();

        for segment in segments {
            let reader = SegmentReader::open(segment.segment_info.clone(), self.storage.clone())?;
            let deleted = self.load_deleted_docs(&segment.segment_info)?;
            let reconstructed = self.reconstruct_segment(
                &reader,
                &deleted,
                &mut positions_by_field,
                &mut doc_values_by_field,
            )?;
            stats.deleted_docs_removed += deleted.len();
            for (doc_id, analyzed) in reconstructed {
                if docs.insert(doc_id, analyzed).is_none() {
                    order.push(doc_id);
                }
            }
        }

        if self.config.sort_by_doc_id {
            order.sort_unstable();
        }

        let doc_count = order.len() as u64;
        let min_doc_id = order.iter().copied().min().unwrap_or(0);
        let max_doc_id = order.iter().copied().max().unwrap_or(0);

        // Replay the reconstructed analyzed documents through a writer so the
        // merged segment is written by the same complete, typed write path as a
        // normal flush, then flush to the merged segment's name. Buffers are
        // unbounded so the merge produces exactly one output segment.
        let writer_config = InvertedIndexWriterConfig {
            field_term_positions: positions_by_field,
            field_doc_values: doc_values_by_field,
            store_doc_values: self.config.default_doc_values,
            shard_id: stats.shard_id,
            max_buffered_docs: usize::MAX,
            max_buffer_memory: usize::MAX,
            use_compound: self.config.use_compound,
            ..Default::default()
        };
        // Deliberately `new`, not `with_shared_metadata` (#1023): this
        // writer exists only to replay documents into the merged segment.
        // With no metadata handle, its implicit Drop-commit at the end of
        // this function cannot touch `metadata.json` — the historical bug
        // here re-added the whole merged output to `doc_count` on every
        // merge, compounding on each auto-merging commit.
        let mut writer = InvertedIndexWriter::new(self.storage.clone(), writer_config)?;
        // Replay + flush as one fallible unit. On ANY error the writer is
        // aborted before it can drop: `Drop` would otherwise commit the
        // partially replayed buffer into a fresh `segment_*` and publish it
        // (#1032) — silent document duplication, since the source segments
        // are only deleted after a successful merge.
        let replayed = (|| -> Result<Vec<String>> {
            for doc_id in &order {
                if let Some(analyzed) = docs.remove(doc_id) {
                    writer.upsert_analyzed_document(*doc_id, analyzed)?;
                }
            }
            writer.flush_buffered_to_segment(new_segment_id)
        })();
        let file_paths = match replayed {
            Ok(paths) => paths,
            Err(e) => {
                writer.abort();
                return Err(e);
            }
        };

        stats.docs_processed = doc_count;
        stats.postings_merged = doc_count;

        // Create new segment info
        let segment_info = SegmentInfo {
            segment_id: new_segment_id.to_string(),
            doc_count,
            min_doc_id,
            max_doc_id,
            generation: 0,        // Will be assigned by segment manager
            has_deletions: false, // New merged segment has no deleted docs until updated
            shard_id: stats.shard_id,
        };

        // Calculate segment size
        let size_bytes = file_paths
            .iter()
            .map(|path| {
                self.storage
                    .metadata(path)
                    .map(|meta| meta.size)
                    .unwrap_or(0)
            })
            .sum();

        // Create managed segment info
        let mut managed_info = ManagedSegmentInfo::new(segment_info);
        managed_info.size_bytes = size_bytes;
        managed_info.file_paths = file_paths.clone();

        Ok(MergeResult {
            new_segment: managed_info,
            stats,
            file_paths,
        })
    }

    /// Rebuild every one of `segments` into a same-count set of NEW
    /// segments, with `target_field` re-derived per
    /// [`Self::reconstruct_segment_with_field_override`] and every other
    /// field carried over unchanged (Issue #1081: `Engine::update_field`
    /// rebuilding a lexical field's analyzer/indexed setting).
    ///
    /// Unlike [`Self::merge_segments`] (N sources -> 1 output), this is a
    /// 1:1 transform — one new segment per source, each keeping that
    /// source's document set — so the caller
    /// ([`InvertedIndex::rebuild_field`](crate::lexical::index::inverted::InvertedIndex::rebuild_field))
    /// can publish the whole batch as a single atomic manifest swap
    /// without disturbing segment count or merge policy.
    ///
    /// `new_segment_ids` must have the same length as `segments`, in the
    /// same order (the caller reserves one fresh ID per source via
    /// `InvertedIndex`'s segment ID generator).
    ///
    /// `target_term_vectors` is `target_field`'s new `term_vectors` setting
    /// (#1083): since `target_field`'s own postings are discarded and
    /// re-derived rather than detected from what is on disk, this is
    /// seeded into the per-field positions map up front so every rebuilt
    /// segment stores (or omits) `target_field`'s positions according to
    /// the NEW schema, not the old one. Every other field keeps whatever
    /// positions state it already had, detected the same way
    /// [`Self::perform_merge`] does.
    ///
    /// `target_doc_values` is `target_field`'s new `doc_values` setting
    /// (#1047), seeded the same way into the per-field DocValues map for
    /// the same reason. Every other field resolves from
    /// [`MergeConfig::field_doc_values`] first, detection second, exactly
    /// as [`Self::perform_merge`] does.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered and aborts the writer that hit
    /// it (no partial segment is left committed). The caller is expected
    /// to treat any error here as "nothing published yet" and leave the
    /// existing segments completely untouched — this function never
    /// touches a manifest itself.
    ///
    /// # Panics
    ///
    /// Panics if `new_segment_ids.len() != segments.len()`.
    pub fn rebuild_field_across_segments(
        &self,
        segments: &[ManagedSegmentInfo],
        target_field: &str,
        analyzer: Option<&Arc<dyn Analyzer>>,
        target_term_vectors: bool,
        target_doc_values: bool,
        new_segment_ids: &[String],
    ) -> Result<Vec<MergeResult>> {
        assert_eq!(
            segments.len(),
            new_segment_ids.len(),
            "one new segment ID is required per source segment"
        );

        // Every other field's positions state is detected from the first
        // posting seen across ALL segments (shared with
        // `reconstruct_segment_with_field_override`'s signature), so every
        // rebuilt segment agrees. `target_field` is seeded up front so its
        // own detection (skipped in Pass 1, since its postings are stale)
        // never overrides the new schema's setting.
        let mut positions_by_field: HashMap<String, bool> = HashMap::new();
        positions_by_field.insert(target_field.to_string(), target_term_vectors);
        // Same idea for DocValues (#1047): seed the CURRENT schema (minus
        // `target_field`, whose stale on-disk column must not leak
        // through), then pin `target_field` to its NEW setting so
        // detection can never override either.
        let mut doc_values_by_field: HashMap<String, bool> = self.config.field_doc_values.clone();
        doc_values_by_field.insert(target_field.to_string(), target_doc_values);
        let mut results = Vec::with_capacity(segments.len());

        for (segment, new_segment_id) in segments.iter().zip(new_segment_ids) {
            let reader = SegmentReader::open(segment.segment_info.clone(), self.storage.clone())?;
            let deleted = self.load_deleted_docs(&segment.segment_info)?;
            let reconstructed = self.reconstruct_segment_with_field_override(
                &reader,
                &deleted,
                &mut positions_by_field,
                &mut doc_values_by_field,
                target_field,
                analyzer,
            )?;

            let doc_count = reconstructed.len() as u64;
            let min_doc_id = reconstructed.iter().map(|(id, _)| *id).min().unwrap_or(0);
            let max_doc_id = reconstructed.iter().map(|(id, _)| *id).max().unwrap_or(0);

            let writer_config = InvertedIndexWriterConfig {
                field_term_positions: positions_by_field.clone(),
                field_doc_values: doc_values_by_field.clone(),
                store_doc_values: self.config.default_doc_values,
                shard_id: segment.segment_info.shard_id,
                max_buffered_docs: usize::MAX,
                max_buffer_memory: usize::MAX,
                use_compound: self.config.use_compound,
                ..Default::default()
            };
            // Deliberately `new`, not `with_shared_metadata` (#1023): see
            // `perform_merge`'s identical comment above.
            let mut writer = InvertedIndexWriter::new(self.storage.clone(), writer_config)?;
            let replayed = (|| -> Result<Vec<String>> {
                for (doc_id, analyzed) in reconstructed {
                    writer.upsert_analyzed_document(doc_id, analyzed)?;
                }
                writer.flush_buffered_to_segment(new_segment_id)
            })();
            let file_paths = match replayed {
                Ok(paths) => paths,
                Err(e) => {
                    writer.abort();
                    return Err(e);
                }
            };

            let segment_info = SegmentInfo {
                segment_id: new_segment_id.clone(),
                doc_count,
                min_doc_id,
                max_doc_id,
                generation: 0, // The caller assigns the final generation.
                has_deletions: false,
                shard_id: segment.segment_info.shard_id,
            };
            let size_bytes = file_paths
                .iter()
                .map(|path| {
                    self.storage
                        .metadata(path)
                        .map(|meta| meta.size)
                        .unwrap_or(0)
                })
                .sum();
            let mut managed_info = ManagedSegmentInfo::new(segment_info);
            managed_info.size_bytes = size_bytes;
            managed_info.file_paths = file_paths.clone();

            results.push(MergeResult {
                new_segment: managed_info,
                stats: MergeStats {
                    segments_merged: 1,
                    docs_processed: doc_count,
                    postings_merged: doc_count,
                    shard_id: segment.segment_info.shard_id,
                    ..Default::default()
                },
                file_paths,
            });
        }

        Ok(results)
    }

    /// Load the set of deleted doc_ids for a segment from its `.delmap`.
    fn load_deleted_docs(&self, segment_info: &SegmentInfo) -> Result<RoaringTreemap> {
        if !segment_info.has_deletions {
            return Ok(RoaringTreemap::new());
        }
        let bitmap_file = format!("{}.delmap", segment_info.segment_id);
        if let Ok(input) = self.storage.open_input(&bitmap_file) {
            use crate::maintenance::deletion::DeletionBitmap;
            use crate::storage::structured::StructReader;

            if let Ok(mut reader) = StructReader::new(input)
                && let Ok(bitmap) = DeletionBitmap::read_from_storage(&mut reader)
            {
                // The `.delmap` payload already *is* a Roaring bitmap, and
                // the merge only ever asks it for a count and membership —
                // both of which it answers directly. Expanding it into a
                // `Vec` and then a hash set turned ~125 KB into tens of
                // megabytes of transient allocation for a segment with a
                // million deletions, and replaced a bit test with a hashed
                // probe on the merge's innermost loops (#541).
                return Ok(bitmap.into_deleted_docs());
            }
        }
        Ok(RoaringTreemap::new())
    }

    /// Load a segment reader for the given segment.
    fn load_segment_reader(
        &self,
        segment_info: &SegmentInfo,
    ) -> Result<Box<dyn LexicalIndexReader>> {
        // Create segment list with single segment
        let segments = vec![segment_info.clone()];

        // Use default config for reader
        let config = crate::lexical::index::inverted::reader::InvertedIndexReaderConfig::default();

        let reader = InvertedIndexReader::new(segments, self.storage.clone(), config)?;
        Ok(Box::new(reader) as Box<dyn LexicalIndexReader>)
    }

    /// Reconstruct every live document's [`AnalyzedDocument`] from one source
    /// segment, without re-tokenizing (Issue #753).
    ///
    /// `field_terms` are rebuilt from the segment's postings (the authoritative
    /// source for the inverted index, so index-only fields survive);
    /// `stored_fields` from the stored documents; `point_values` (BKD entries)
    /// are read back from the segment's BKD trees — the authoritative source
    /// for numeric/geo points, so index-only (`stored=false`) and multi-valued
    /// numeric fields are preserved (Issue #758); `field_lengths` are read back
    /// from the segment. Deleted docs are excluded.
    ///
    /// `positions_by_field` is populated, per field, from the first posting
    /// seen for that field (whether the source stored term positions) so
    /// the merged segment reproduces each field's positions state
    /// independently — fields can disagree, e.g. one `term_vectors: true`
    /// and one `false` (#1083).
    ///
    /// `doc_values_by_field` (#1047) works the same way but only fills
    /// gaps the caller's schema-derived seed left open (see
    /// [`MergeConfig::field_doc_values`]): for a field this map does not
    /// already mention, the first document in *this* segment with a
    /// DocValues-candidate value for it records whether this segment's
    /// `.dv` currently has a column for that field.
    fn reconstruct_segment(
        &self,
        reader: &SegmentReader,
        deleted: &RoaringTreemap,
        positions_by_field: &mut HashMap<String, bool>,
        doc_values_by_field: &mut HashMap<String, bool>,
    ) -> Result<Vec<(u64, AnalyzedDocument)>> {
        // Pass 1: bucket postings into per-doc analyzed terms.
        let mut field_terms: AHashMap<u64, AHashMap<String, Vec<AnalyzedTerm>>> = AHashMap::new();
        if let Some(dict) = reader.term_dictionary()? {
            for (term_key, _info) in dict.iter() {
                let Some((field, term)) = term_key.split_once(':') else {
                    continue;
                };
                if let Some(mut iter) = reader.postings(field, term)? {
                    while iter.next()? {
                        // No deletion check here: `SegmentReader::postings`
                        // already excludes deleted documents on both of its
                        // paths — `filter_deleted_soa` for the normal one and
                        // `scan_documents_for_term` for the no-inverted-index
                        // fallback — gating on the same `has_deletions` flag
                        // and the same `.delmap` this merge reads.
                        //
                        // That invariant is pinned by
                        // `postings_never_yields_a_deleted_document` in
                        // reader.rs, which is stronger protection than a
                        // second test that can never fire, on the innermost
                        // loop of the merge. The BKD and stored-document
                        // loops below get no such filtering and do check
                        // (#541).
                        let doc_id = iter.doc_id();
                        let positions = iter.positions()?;
                        let freq = iter.term_freq();
                        positions_by_field
                            .entry(field.to_string())
                            .or_insert_with(|| !positions.is_empty());
                        let terms = field_terms
                            .entry(doc_id)
                            .or_default()
                            .entry(field.to_string())
                            .or_default();
                        if positions.is_empty() {
                            // Frequency-only segment: one analyzed term carries
                            // the whole frequency.
                            terms.push(AnalyzedTerm {
                                term: term.to_string(),
                                position: 0,
                                frequency: freq as u32,
                                offset: (0, 0),
                            });
                        } else {
                            // One analyzed term per stored position so the
                            // rebuilt posting list reproduces the positions.
                            for pos in positions {
                                terms.push(AnalyzedTerm {
                                    term: term.to_string(),
                                    position: pos as u32,
                                    frequency: freq as u32,
                                    offset: (0, 0),
                                });
                            }
                        }
                    }
                }
            }
        }

        // Collect BKD points per (doc, field) straight from the segment's BKD
        // trees — the authoritative source for numeric/geo points. This keeps
        // index-only (`stored=false`) and multi-valued numeric fields, which a
        // stored-field derivation would miss (Issue #758).
        let mut points: AHashMap<u64, AHashMap<String, Vec<Vec<f64>>>> = AHashMap::new();
        // Enumerate through the reader (#554): a raw `list_files` scan
        // finds no `.bkd` files once the parts live inside a compound
        // container, and the merged segment would silently drop every
        // numeric/geo point (`verify_after_merge` only checks doc_count).
        for field in reader.bkd_field_names()? {
            let field = field.as_str();
            if let Some(tree) = reader.get_bkd_tree(field)? {
                let mut visitor = CollectPointsVisitor::default();
                tree.intersect(&mut visitor)?;
                for (doc_id, point) in visitor.entries {
                    // Load-bearing: `get_bkd_tree` hands back the raw
                    // `BKDReader`, which knows nothing about deletions.
                    if deleted.contains(doc_id) {
                        continue;
                    }
                    points
                        .entry(doc_id)
                        .or_default()
                        .entry(field.to_string())
                        .or_default()
                        .push(point);
                }
            }
        }

        // Pass 2: assemble each live document.
        let mut out = Vec::new();
        for doc_id in reader.doc_ids()? {
            // Load-bearing: `doc_ids()` returns every stored key, deleted
            // ones included.
            if deleted.contains(doc_id) {
                continue;
            }
            let Some(stored) = reader.document(doc_id)? else {
                continue;
            };

            let mut analyzed = AnalyzedDocument::new();
            analyzed.field_terms = field_terms.remove(&doc_id).unwrap_or_default();
            analyzed.point_values = points.remove(&doc_id).unwrap_or_default();

            for (field_name, value) in &stored.fields {
                if InvertedIndexWriter::is_doc_values_candidate(value) {
                    // Only fills a gap the schema-derived seed left open
                    // (#1047) -- a field `doc_values_by_field` already
                    // covers (the current schema decided it) is never
                    // touched here.
                    doc_values_by_field
                        .entry(field_name.clone())
                        .or_insert_with(|| reader.has_doc_values(field_name));
                }
                analyzed
                    .stored_fields
                    .insert(field_name.clone(), value.clone());
            }

            // Field lengths are read back from the segment so BM25 length
            // normalization is preserved exactly. Only indexed fields have a
            // recorded length.
            let indexed_fields: Vec<String> = analyzed.field_terms.keys().cloned().collect();
            for field_name in indexed_fields {
                if let Some(len) = reader.field_length(doc_id, &field_name)? {
                    analyzed.field_lengths.insert(field_name, len);
                }
            }

            out.push((doc_id, analyzed));
        }

        Ok(out)
    }

    /// Reconstruct a segment's analyzed documents like
    /// [`Self::reconstruct_segment`], but with `target_field`'s existing
    /// postings/BKD points discarded and re-derived from its stored value
    /// using `analyzer` (Issue #1081).
    ///
    /// Every other field is carried over unchanged, exactly as
    /// [`Self::reconstruct_segment`] does — this is the "field-conversion
    /// hook" that lets a rebuild change one field's analyzer/indexed
    /// setting without perturbing the rest of the segment. `analyzer` is
    /// `None` when the field is being switched to `indexed: false`: its
    /// terms/points are simply omitted (skipped in Pass 1/1.5 below), the
    /// same as [`InvertedIndexWriter::analyze_document`]'s `should_index`
    /// gate for a fresh document.
    ///
    /// A live document whose stored fields lack `target_field` entirely is
    /// left with no terms/points for it, exactly as if the field were
    /// absent from the original document — this function does not
    /// validate that `target_field` is actually `stored: true` in the
    /// schema; callers (`InvertedIndex::rebuild_field`, gated by
    /// `Engine::update_field`'s `classify_change` call) are responsible
    /// for that.
    fn reconstruct_segment_with_field_override(
        &self,
        reader: &SegmentReader,
        deleted: &RoaringTreemap,
        positions_by_field: &mut HashMap<String, bool>,
        doc_values_by_field: &mut HashMap<String, bool>,
        target_field: &str,
        analyzer: Option<&Arc<dyn Analyzer>>,
    ) -> Result<Vec<(u64, AnalyzedDocument)>> {
        // Pass 1: bucket postings into per-doc analyzed terms, EXCEPT
        // `target_field` — its old postings were built under the previous
        // analyzer/indexed setting and are stale under the new one.
        let mut field_terms: AHashMap<u64, AHashMap<String, Vec<AnalyzedTerm>>> = AHashMap::new();
        if let Some(dict) = reader.term_dictionary()? {
            for (term_key, _info) in dict.iter() {
                let Some((field, term)) = term_key.split_once(':') else {
                    continue;
                };
                if field == target_field {
                    continue;
                }
                if let Some(mut iter) = reader.postings(field, term)? {
                    while iter.next()? {
                        // See `reconstruct_segment`'s identical comment:
                        // `postings` already excludes deleted documents.
                        let doc_id = iter.doc_id();
                        let positions = iter.positions()?;
                        let freq = iter.term_freq();
                        positions_by_field
                            .entry(field.to_string())
                            .or_insert_with(|| !positions.is_empty());
                        let terms = field_terms
                            .entry(doc_id)
                            .or_default()
                            .entry(field.to_string())
                            .or_default();
                        if positions.is_empty() {
                            terms.push(AnalyzedTerm {
                                term: term.to_string(),
                                position: 0,
                                frequency: freq as u32,
                                offset: (0, 0),
                            });
                        } else {
                            for pos in positions {
                                terms.push(AnalyzedTerm {
                                    term: term.to_string(),
                                    position: pos as u32,
                                    frequency: freq as u32,
                                    offset: (0, 0),
                                });
                            }
                        }
                    }
                }
            }
        }

        // BKD points, EXCEPT `target_field` for the same reason (a numeric
        // field switching `indexed: false -> true`, or being re-derived
        // for consistency, needs fresh points from its stored value).
        let mut points: AHashMap<u64, AHashMap<String, Vec<Vec<f64>>>> = AHashMap::new();
        for field in reader.bkd_field_names()? {
            let field = field.as_str();
            if field == target_field {
                continue;
            }
            if let Some(tree) = reader.get_bkd_tree(field)? {
                let mut visitor = CollectPointsVisitor::default();
                tree.intersect(&mut visitor)?;
                for (doc_id, point) in visitor.entries {
                    if deleted.contains(doc_id) {
                        continue;
                    }
                    points
                        .entry(doc_id)
                        .or_default()
                        .entry(field.to_string())
                        .or_default()
                        .push(point);
                }
            }
        }

        // Pass 2: assemble each live document, re-deriving `target_field`
        // from its stored value via the SAME per-value analysis
        // `InvertedIndexWriter::analyze_document` uses for fresh ingestion
        // (`analyze_field_value`), so a rebuilt field is indistinguishable
        // from one indexed fresh under the new option.
        let mut out = Vec::new();
        for doc_id in reader.doc_ids()? {
            if deleted.contains(doc_id) {
                continue;
            }
            let Some(stored) = reader.document(doc_id)? else {
                continue;
            };

            let mut analyzed = AnalyzedDocument::new();
            analyzed.field_terms = field_terms.remove(&doc_id).unwrap_or_default();
            analyzed.point_values = points.remove(&doc_id).unwrap_or_default();

            for (field_name, value) in &stored.fields {
                if InvertedIndexWriter::is_doc_values_candidate(value) {
                    // `target_field` is pre-seeded by
                    // `rebuild_field_across_segments` with the field's NEW
                    // `doc_values` setting, so this never overrides it
                    // with the stale on-disk state (#1047, mirrors
                    // `positions_by_field`'s identical pre-seeding for
                    // `target_term_vectors`).
                    doc_values_by_field
                        .entry(field_name.clone())
                        .or_insert_with(|| reader.has_doc_values(field_name));
                }
                analyzed
                    .stored_fields
                    .insert(field_name.clone(), value.clone());
            }

            if let (Some(analyzer), Some(target_value)) =
                (analyzer, stored.fields.get(target_field))
            {
                let (terms, pts) = analyze_field_value(target_field, target_value, analyzer)?;
                if !terms.is_empty() {
                    analyzed.field_terms.insert(target_field.to_string(), terms);
                }
                if !pts.is_empty() {
                    analyzed.point_values.insert(target_field.to_string(), pts);
                }
            }
            // `analyzer.is_none()` (switching to `indexed: false`):
            // `target_field`'s terms/points are already absent (Pass 1/1.5
            // skipped it above), so there is nothing further to do here.

            let indexed_fields: Vec<String> = analyzed.field_terms.keys().cloned().collect();
            for field_name in indexed_fields {
                if field_name == target_field {
                    let len = analyzed.field_terms[&field_name].len() as u32;
                    analyzed.field_lengths.insert(field_name, len);
                } else if let Some(len) = reader.field_length(doc_id, &field_name)? {
                    analyzed.field_lengths.insert(field_name, len);
                }
            }

            out.push((doc_id, analyzed));
        }

        Ok(out)
    }

    /// Verify the integrity of a merged segment.
    fn verify_merged_segment(&self, segment: &ManagedSegmentInfo) -> Result<()> {
        // Load the segment and perform basic checks
        let reader = self.load_segment_reader(&segment.segment_info)?;

        // Check document count matches
        if reader.doc_count() != segment.segment_info.doc_count {
            return Err(LaurusError::index("Document count mismatch after merge"));
        }

        // TODO: Add more verification checks
        // - Term dictionary integrity
        // - Posting list consistency
        // - Document field validation

        Ok(())
    }

    /// Get merge configuration.
    pub fn get_config(&self) -> &MergeConfig {
        &self.config
    }
}

/// BKD visitor that enumerates **every** `(doc_id, point)` entry in a tree.
///
/// Used by the merge to read back all stored points (Issue #758). `compare`
/// always returns [`CellRelation::Crosses`] so the traversal descends to every
/// leaf and yields each point through `visit` (a `CellRelation::Inside` verdict
/// would report doc ids via `visit_inside` *without* the point coordinates,
/// which the merge needs). `visit_inside` is therefore never called.
#[derive(Default)]
struct CollectPointsVisitor {
    entries: Vec<(u64, Vec<f64>)>,
}

impl IntersectVisitor for CollectPointsVisitor {
    fn compare(&self, _cell: &AABB) -> CellRelation {
        // Force a full descent so every point is reported via `visit`.
        CellRelation::Crosses
    }

    fn visit_inside(&mut self, _doc_id: u64) {
        // Unreachable: `compare` never returns `Inside`. Points (not just doc
        // ids) are required, so all entries must arrive through `visit`.
    }

    fn visit(&mut self, doc_id: u64, point: &[f64]) {
        self.entries.push((doc_id, point.to_vec()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::inverted::segment::ManagedSegmentInfo;
    use crate::lexical::index::inverted::segment::SegmentInfo;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;

    /// #541 — the deterministic gate.
    ///
    /// The merge used to expand the `.delmap`'s Roaring bitmap into a
    /// `Vec<u64>` and then an `AHashSet<u64>`, both discarded when the
    /// merge finished. This pins what that cost, without a stopwatch:
    /// wall-clock benchmarks are noise-dominated on this host, and no
    /// benchmark reaches `MergeEngine` at all.
    ///
    /// `serialized_size()` is the repository's own yardstick for this
    /// comparison — `DeletionBitmap::memory_usage` is defined as exactly
    /// that, and its doc comment records that it replaced "the previous
    /// `AHashSet::capacity()` heuristic".
    ///
    /// The point asserted is structural rather than a single ratio: over a
    /// dense run of deletions the bitmap's size **does not grow at all**
    /// while the hash set grows with the deletion count. A ratio measured
    /// at one size would have hidden that, and would also have been
    /// measured at whichever container boundary happened to be worst.
    ///
    /// The hash-set figure, `capacity() * size_of::<u64>()`, deliberately
    /// UNDERSTATES the old cost: it ignores hashbrown's one control byte
    /// per bucket and its 8/7 over-allocation, and ignores the `Vec<u64>`
    /// materialised alongside it. Every number here is a floor.
    #[test]
    fn roaring_deletion_set_is_far_smaller_than_the_hash_set_it_replaced() {
        use ahash::AHashSet;
        use roaring::RoaringTreemap;

        let mut measured: Vec<(u64, usize, usize)> = Vec::new();

        // Dense runs — the shape a segment accumulates over its life.
        for deleted_count in [4_096u64, 16_384, 65_536] {
            let mut bitmap = RoaringTreemap::new();
            for doc_id in 0..deleted_count {
                bitmap.insert(doc_id);
            }

            let old: AHashSet<u64> = bitmap.iter().collect();
            assert_eq!(
                old.len() as u64,
                bitmap.len(),
                "the substitution must be lossless at {deleted_count} deletions"
            );

            measured.push((
                deleted_count,
                bitmap.serialized_size(),
                old.capacity() * std::mem::size_of::<u64>(),
            ));
        }

        // The bitmap is flat across a 16x growth in deletions; the hash set
        // is not. This is the mechanism, and it is what makes the saving
        // scale with segment size rather than being a fixed discount.
        let bitmap_sizes: Vec<usize> = measured.iter().map(|(_, b, _)| *b).collect();
        assert!(
            bitmap_sizes.iter().all(|b| *b == bitmap_sizes[0]),
            "a dense run must stay one container regardless of length: {measured:?}"
        );

        let first = measured[0];
        let last = measured[measured.len() - 1];
        assert!(
            last.2 >= first.2 * 8,
            "the hash set must grow with the deletion count: {measured:?}"
        );

        // Even at the worst container boundary the hash set alone — before
        // counting the transient Vec — is several times the bitmap.
        for (count, roaring_bytes, hash_set_bytes) in &measured {
            let discarded_vec_bytes = (*count as usize) * std::mem::size_of::<u64>();
            assert!(
                hash_set_bytes >= &(6 * roaring_bytes),
                "at {count} deletions: hash set {hash_set_bytes} B vs bitmap {roaring_bytes} B, \
                 plus a further {discarded_vec_bytes} B of transient Vec"
            );
        }
    }

    /// #541 — `into_deleted_docs` must hand over exactly what
    /// `get_deleted_docs` used to collect, so switching the merge to the
    /// bitmap changes what it allocates and nothing else.
    #[test]
    fn into_deleted_docs_matches_the_vec_it_replaces() {
        use crate::maintenance::deletion::DeletionBitmap;

        let bitmap = DeletionBitmap::new("seg_x".to_string(), 0, 999);
        for doc_id in [3u64, 7, 42, 900, 999] {
            bitmap.delete_document(doc_id).unwrap();
        }

        let as_vec = bitmap.get_deleted_docs();
        let as_bitmap = bitmap.into_deleted_docs();

        assert_eq!(as_bitmap.len(), as_vec.len() as u64);
        assert_eq!(as_bitmap.iter().collect::<Vec<u64>>(), as_vec);
        for doc_id in &as_vec {
            assert!(as_bitmap.contains(*doc_id));
        }
        assert!(!as_bitmap.contains(4));
    }

    #[allow(dead_code)]
    fn create_test_segment(id: &str, doc_count: u64) -> ManagedSegmentInfo {
        let segment_info = SegmentInfo {
            segment_id: id.to_string(),
            doc_count,
            min_doc_id: 0,
            max_doc_id: doc_count.saturating_sub(1),
            generation: 1,
            has_deletions: false,
            shard_id: 0, // Added shard_id for test segments
        };

        ManagedSegmentInfo::new(segment_info)
    }

    #[test]
    fn test_merge_engine_creation() {
        let config = MergeConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let engine = MergeEngine::new(config, storage);
        assert_eq!(engine.config.batch_size, 10000);
        assert!(engine.config.remove_deleted_docs);
    }

    #[test]
    fn test_merge_config_default() {
        let config = MergeConfig::default();

        assert_eq!(config.max_memory_mb, 256);
        assert_eq!(config.batch_size, 10000);
        assert!(config.enable_compression);
        assert!(config.remove_deleted_docs);
        assert!(config.sort_by_doc_id);
        assert!(config.verify_after_merge);
    }

    #[test]
    fn test_merge_stats_space_savings() {
        let stats = MergeStats {
            size_before: 1000,
            size_after: 800,
            ..Default::default()
        };

        assert_eq!(stats.space_savings(), 20.0);

        let stats_zero = MergeStats {
            size_before: 0,
            size_after: 0,
            ..Default::default()
        };
        assert_eq!(stats_zero.space_savings(), 0.0);
    }

    use crate::data::{DataValue, Document};
    use crate::lexical::index::inverted::reader::{InvertedIndexReader, SegmentReader};
    use crate::lexical::index::inverted::writer::{InvertedIndexWriter, InvertedIndexWriterConfig};
    use crate::lexical::reader::LexicalIndexReader;

    fn text_int_doc(title: &str, num: i64) -> Document {
        Document::builder()
            .add_field("title", DataValue::Text(title.to_string()))
            .add_field("num", DataValue::Int64(num))
            .build()
    }

    /// Describe a segment a standalone writer just flushed (#1024: such
    /// writers register their segments nowhere, so the test provides the
    /// descriptor the manifest would normally hold).
    fn segment_info(
        seg_id: &str,
        doc_count: u64,
        min: u64,
        max: u64,
        generation: u64,
    ) -> SegmentInfo {
        SegmentInfo {
            segment_id: seg_id.to_string(),
            doc_count,
            min_doc_id: min,
            max_doc_id: max,
            generation,
            has_deletions: false,
            shard_id: 0,
        }
    }

    /// End-to-end correctness of the rewritten merge (Issue #753, closes #556):
    /// merging two segments must produce one segment that preserves the
    /// documents, their *typed* stored fields (int stays int — the #556 bug
    /// stringified everything), and their searchable postings.
    #[test]
    fn merge_preserves_docs_typed_fields_and_postings() {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Two segments via two commits.
        let mut writer =
            InvertedIndexWriter::new(storage.clone(), InvertedIndexWriterConfig::default())
                .unwrap();
        let d0 = writer
            .add_document(text_int_doc("alpha bravo", 10))
            .unwrap();
        let d1 = writer
            .add_document(text_int_doc("bravo charlie", 20))
            .unwrap();
        writer.commit().unwrap(); // segment_000000
        let d2 = writer
            .add_document(text_int_doc("charlie delta", 30))
            .unwrap();
        writer.commit().unwrap(); // segment_000001
        drop(writer);

        let si0 = segment_info("segment_000000", 2, d0, d1, 0);
        let si1 = segment_info("segment_000001", 1, d2, d2, 1);
        let candidate = MergeCandidate {
            segments: vec![si0.segment_id.clone(), si1.segment_id.clone()],
            priority: 1.0,
            estimated_size: 0,
            strategy: MergeStrategy::SizeBased,
        };
        let engine = MergeEngine::new(MergeConfig::default(), storage.clone());
        let result = engine
            .merge_segments(
                &candidate,
                &[ManagedSegmentInfo::new(si0), ManagedSegmentInfo::new(si1)],
                1,
            )
            .unwrap();

        // All three docs survive the merge (verify_after_merge also checks this).
        assert_eq!(result.new_segment.segment_info.doc_count, 3);

        // Typed stored fields round-trip: `num` stays an Int64 (the #556 bug
        // wrote it as a stringified value).
        let merged =
            SegmentReader::open(result.new_segment.segment_info.clone(), storage.clone()).unwrap();
        for (doc_id, expected) in [(d0, 10i64), (d1, 20), (d2, 30)] {
            let doc = merged
                .document(doc_id)
                .unwrap()
                .unwrap_or_else(|| panic!("doc {doc_id} missing after merge"));
            match doc.fields.get("num") {
                Some(DataValue::Int64(n)) => assert_eq!(*n, expected, "doc {doc_id} num"),
                other => panic!("doc {doc_id} `num` not Int64 after merge: {other:?}"),
            }
        }

        // Postings are reconstructed (not empty): "bravo" appears in d0 and d1.
        let reader = InvertedIndexReader::new(
            vec![result.new_segment.segment_info.clone()],
            storage.clone(),
            Default::default(),
        )
        .unwrap();
        let mut got = Vec::new();
        if let Some(mut it) = reader.postings("title", "bravo").unwrap() {
            while it.next().unwrap() {
                got.push(it.doc_id());
            }
        }
        got.sort_unstable();
        let mut want = vec![d0, d1];
        want.sort_unstable();
        assert_eq!(got, want, "`title:bravo` postings after merge");
    }

    /// #1083: after merging two segments, each field must keep its OWN
    /// `term_vectors` state independently — one field can carry positions
    /// while another does not, and the merged segment must not collapse
    /// them into a single index-wide setting. Run with both field-name
    /// orderings so the result cannot depend on `HashMap`/dictionary
    /// iteration order.
    #[test]
    fn merge_preserves_per_field_term_vectors_independently() {
        use crate::lexical::core::field::{FieldOption, TextOption};
        use crate::lexical::query::Query;
        use crate::lexical::query::phrase::PhraseQuery;

        let run = |vec_field: &str, novec_field: &str| {
            let storage: Arc<dyn Storage> =
                Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

            let mut fields = std::collections::HashMap::new();
            fields.insert(
                vec_field.to_string(),
                FieldOption::Text(TextOption {
                    term_vectors: true,
                    ..Default::default()
                }),
            );
            fields.insert(
                novec_field.to_string(),
                FieldOption::Text(TextOption {
                    term_vectors: false,
                    ..Default::default()
                }),
            );
            let config = InvertedIndexWriterConfig {
                fields,
                ..Default::default()
            };

            let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();
            let d0 = writer
                .add_document(
                    Document::builder()
                        .add_field(vec_field, DataValue::Text("quick brown fox".to_string()))
                        .add_field(novec_field, DataValue::Text("quick brown fox".to_string()))
                        .build(),
                )
                .unwrap();
            writer.commit().unwrap(); // segment_000000
            let d1 = writer
                .add_document(
                    Document::builder()
                        .add_field(vec_field, DataValue::Text("lazy dog".to_string()))
                        .add_field(novec_field, DataValue::Text("lazy dog".to_string()))
                        .build(),
                )
                .unwrap();
            writer.commit().unwrap(); // segment_000001
            drop(writer);

            let si0 = segment_info("segment_000000", 1, d0, d0, 0);
            let si1 = segment_info("segment_000001", 1, d1, d1, 1);
            let candidate = MergeCandidate {
                segments: vec![si0.segment_id.clone(), si1.segment_id.clone()],
                priority: 1.0,
                estimated_size: 0,
                strategy: MergeStrategy::SizeBased,
            };
            let engine = MergeEngine::new(MergeConfig::default(), storage.clone());
            let result = engine
                .merge_segments(
                    &candidate,
                    &[ManagedSegmentInfo::new(si0), ManagedSegmentInfo::new(si1)],
                    1,
                )
                .unwrap();

            let reader = InvertedIndexReader::new(
                vec![result.new_segment.segment_info.clone()],
                storage.clone(),
                Default::default(),
            )
            .unwrap();

            let with_vectors =
                PhraseQuery::new(vec_field, vec!["quick".to_string(), "brown".to_string()]);
            let matcher = with_vectors.matcher(&reader).unwrap();
            assert!(
                !matcher.is_exhausted(),
                "{vec_field} must keep its positions after merge"
            );

            let without_vectors =
                PhraseQuery::new(novec_field, vec!["quick".to_string(), "brown".to_string()]);
            let matcher = without_vectors.matcher(&reader).unwrap();
            assert!(
                matcher.is_exhausted(),
                "{novec_field} must not have positions after merge"
            );
        };

        // Both orderings, so the result cannot depend on field-name sort order.
        run("a_vec", "b_novec");
        run("a_novec", "b_vec");
    }

    /// #1047: mirrors [`merge_preserves_per_field_term_vectors_independently`]
    /// for DocValues -- after merging two segments, each field must keep its
    /// OWN `doc_values` state independently, detected per field from what
    /// each source segment actually has on disk (`MergeConfig::default()`,
    /// no schema override). Run with both field-name orderings.
    #[test]
    fn merge_preserves_per_field_doc_values_independently() {
        use crate::lexical::core::field::{FieldOption, TextOption};

        let run = |dv_field: &str, nodv_field: &str| {
            let storage: Arc<dyn Storage> =
                Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

            let mut fields = std::collections::HashMap::new();
            fields.insert(
                dv_field.to_string(),
                FieldOption::Text(TextOption {
                    doc_values: true,
                    ..Default::default()
                }),
            );
            fields.insert(
                nodv_field.to_string(),
                FieldOption::Text(TextOption {
                    doc_values: false,
                    ..Default::default()
                }),
            );
            let config = InvertedIndexWriterConfig {
                fields,
                ..Default::default()
            };

            let mut writer = InvertedIndexWriter::new(storage.clone(), config).unwrap();
            let d0 = writer
                .add_document(
                    Document::builder()
                        .add_field(dv_field, DataValue::Text("alpha".to_string()))
                        .add_field(nodv_field, DataValue::Text("alpha".to_string()))
                        .build(),
                )
                .unwrap();
            writer.commit().unwrap(); // segment_000000
            let d1 = writer
                .add_document(
                    Document::builder()
                        .add_field(dv_field, DataValue::Text("bravo".to_string()))
                        .add_field(nodv_field, DataValue::Text("bravo".to_string()))
                        .build(),
                )
                .unwrap();
            writer.commit().unwrap(); // segment_000001
            drop(writer);

            let si0 = segment_info("segment_000000", 1, d0, d0, 0);
            let si1 = segment_info("segment_000001", 1, d1, d1, 1);
            let candidate = MergeCandidate {
                segments: vec![si0.segment_id.clone(), si1.segment_id.clone()],
                priority: 1.0,
                estimated_size: 0,
                strategy: MergeStrategy::SizeBased,
            };
            let engine = MergeEngine::new(MergeConfig::default(), storage.clone());
            let result = engine
                .merge_segments(
                    &candidate,
                    &[ManagedSegmentInfo::new(si0), ManagedSegmentInfo::new(si1)],
                    1,
                )
                .unwrap();

            let merged =
                SegmentReader::open(result.new_segment.segment_info.clone(), storage.clone())
                    .unwrap();
            assert!(
                merged.has_doc_values(dv_field),
                "{dv_field} must keep its DocValues column after merge"
            );
            assert!(
                !merged.has_doc_values(nodv_field),
                "{nodv_field} must not have a DocValues column after merge"
            );
        };

        // Both orderings, so the result cannot depend on field-name sort order.
        run("a_dv", "b_nodv");
        run("a_nodv", "b_dv");
    }

    /// #1047: the merge's CURRENT schema (`MergeConfig::field_doc_values`)
    /// must win over a source segment's stale on-disk DocValues state --
    /// the deliberate deviation from `term_vectors`' detection-only
    /// resolution (Issue #1047's design rationale: a DocValues column can
    /// always be regenerated from `stored_fields`, so there is no harm in
    /// re-deriving it from the current schema on every merge).
    #[test]
    fn merge_resolves_doc_values_from_the_current_schema_over_stale_segments() {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Written with a bare (pre-flag-change) writer, so "notes" gets a
        // DocValues column the ordinary way (doc_values defaults to true).
        let mut writer =
            InvertedIndexWriter::new(storage.clone(), InvertedIndexWriterConfig::default())
                .unwrap();
        let d0 = writer
            .add_document(
                Document::builder()
                    .add_field("title", DataValue::Text("alpha".to_string()))
                    .add_field("notes", DataValue::Text("legacy note".to_string()))
                    .build(),
            )
            .unwrap();
        writer.commit().unwrap(); // segment_000000
        drop(writer);

        let si0 = segment_info("segment_000000", 1, d0, d0, 0);
        assert!(
            SegmentReader::open(si0.clone(), storage.clone())
                .unwrap()
                .has_doc_values("notes"),
            "sanity: the source segment must have the column before the merge"
        );

        // The schema has SINCE changed "notes" to `doc_values: false`. Even
        // though the only source segment still has the column on disk, the
        // merge must honor the new schema.
        let mut field_doc_values = std::collections::HashMap::new();
        field_doc_values.insert("notes".to_string(), false);
        let config = MergeConfig {
            field_doc_values,
            ..MergeConfig::default()
        };
        let candidate = MergeCandidate {
            segments: vec![si0.segment_id.clone()],
            priority: 1.0,
            estimated_size: 0,
            strategy: MergeStrategy::SizeBased,
        };
        let engine = MergeEngine::new(config, storage.clone());
        let result = engine
            .merge_segments(&candidate, &[ManagedSegmentInfo::new(si0)], 1)
            .unwrap();

        let merged =
            SegmentReader::open(result.new_segment.segment_info.clone(), storage.clone()).unwrap();
        assert!(
            !merged.has_doc_values("notes"),
            "the current schema's doc_values: false must win over the \
             source segment's on-disk column"
        );
        assert!(
            merged.has_doc_values("title"),
            "a field the schema doesn't mention still detects from the source segment"
        );
    }

    /// #1047: a source segment with no value at all for some field must
    /// not be misread as that field having opted out of DocValues --
    /// another segment's real column, and its values, must still survive
    /// the merge intact. Deliberately processes the value-less segment
    /// FIRST so a naive "first segment seen decides" bug (as opposed to
    /// "first segment that actually HAS the field decides") would be
    /// caught.
    #[test]
    fn merge_does_not_misdetect_a_valueless_field_as_opted_out() {
        let storage: Arc<dyn Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut writer =
            InvertedIndexWriter::new(storage.clone(), InvertedIndexWriterConfig::default())
                .unwrap();
        // segment_000000: no "extra" field at all.
        let d0 = writer
            .add_document(
                Document::builder()
                    .add_field("title", DataValue::Text("alpha".to_string()))
                    .build(),
            )
            .unwrap();
        writer.commit().unwrap();
        // segment_000001: "extra" is present, with its own real column.
        let d1 = writer
            .add_document(
                Document::builder()
                    .add_field("title", DataValue::Text("bravo".to_string()))
                    .add_field("extra", DataValue::Text("present".to_string()))
                    .build(),
            )
            .unwrap();
        writer.commit().unwrap();
        drop(writer);

        let si0 = segment_info("segment_000000", 1, d0, d0, 0);
        let si1 = segment_info("segment_000001", 1, d1, d1, 1);
        let candidate = MergeCandidate {
            segments: vec![si0.segment_id.clone(), si1.segment_id.clone()],
            priority: 1.0,
            estimated_size: 0,
            strategy: MergeStrategy::SizeBased,
        };
        let engine = MergeEngine::new(MergeConfig::default(), storage.clone());
        let result = engine
            .merge_segments(
                &candidate,
                &[ManagedSegmentInfo::new(si0), ManagedSegmentInfo::new(si1)],
                1,
            )
            .unwrap();

        let merged =
            SegmentReader::open(result.new_segment.segment_info.clone(), storage.clone()).unwrap();
        assert!(
            merged.has_doc_values("extra"),
            "segment_000000 having no value for \"extra\" at all must not be \
             misread as an opt-out; segment_000001's real column must still win"
        );
        assert_eq!(
            merged.get_doc_value("extra", d1).unwrap(),
            Some(DataValue::Text("present".to_string())),
            "the merged value itself must be intact"
        );
    }
}
