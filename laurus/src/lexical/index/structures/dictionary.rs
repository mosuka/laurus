//! Term dictionary data structures for mapping terms to posting list metadata.
//!
//! This module provides [`BlockTermDictionary`] — a Lucene
//! `BlockTreeTermsWriter`-style dictionary that maps each term to its
//! [`TermInfo`] (posting list offset, length, document frequency, total
//! frequency, and Block-Max-WAND metadata). The dictionary is built
//! through [`TermDictionaryBuilder`] from an in-memory `BTreeMap`.
//!
//! See Issue [#487](https://github.com/mosuka/laurus/issues/487) for
//! the design rationale and the migration path away from the legacy
//! parallel-array / `AHashMap` representation that this module replaced.

use std::collections::BTreeMap;
use std::sync::Arc;

use ahash::AHashMap;
use fst::Map as FstMap;

use crate::error::{LaurusError, Result};
use crate::storage::structured::{StructReader, StructWriter};
use crate::storage::{StorageInput, StorageOutput};
use crate::util::alloc_bounds::checked_capacity_u64;

// Sub-modules backing the new Lucene BlockTreeTerms-style
// implementation (Issue #487). These will gradually take over as the
// legacy `Hybrid` / `Sorted` / `Hash` dictionaries are decommissioned
// in Phase 9.
mod block_max_data;
mod block_reader;
mod builder;
mod front_coding;
mod term_info_block;

/// Magic number for the v1 BlockTermDictionary on-disk format ("LTDD"
/// = Laurus Term Dictionary, block-tree). Issue #487.
const MAGIC_LTDD: u32 = 0x4C544444;
/// Legacy magic for the removed sorted-array dictionary ("STDC").
/// Reading a file with this magic now yields an explicit "rebuild
/// required" error.
const LEGACY_MAGIC_STDC: u32 = 0x53544443;
/// Legacy magic for the removed hash-table dictionary ("HTDC").
/// Reading a file with this magic now yields an explicit "rebuild
/// required" error.
const LEGACY_MAGIC_HTDC: u32 = 0x48544443;

/// One block's max-impact metadata for Block-Max-WAND (#403 PR-C).
///
/// Each block covers up to [`BLOCK_SIZE`] consecutive postings of a
/// term. The pair records:
///
/// - `last_doc_id` — the doc id of the last posting in the block; used
///   to locate the block containing a target doc id by binary search,
///   and to seek to the next block during WAND skipping.
/// - `max_factor` — the tightest possible BM25 TF-component value over
///   the postings in this block, computed with default BM25 parameters
///   (`k1 = 1.2`, `b = 0.75`) and the segment's average field length.
///   `BM25Scorer::block_max_score_at` multiplies it by `boost · idf`
///   to expose a per-block upper bound to the searcher loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockMax {
    /// Last doc id covered by this block.
    pub last_doc_id: u64,
    /// Block-level TF-component upper bound (default BM25 parameters).
    pub max_factor: f32,
}

/// Posting blocks are 128 entries wide, matching Lucene 9 / Tantivy.
pub const BLOCK_SIZE: usize = 128;

/// Information about a term in the dictionary.
#[derive(Debug, Clone, PartialEq)]
pub struct TermInfo {
    /// Offset to the posting list in the posting file.
    pub posting_offset: u64,
    /// Length of the posting list in bytes.
    pub posting_length: u64,
    /// Document frequency (number of documents containing this term).
    pub doc_frequency: u64,
    /// Total frequency across all documents.
    pub total_frequency: u64,
    /// Term-level tightened BM25 TF-component upper bound (#403 PR-B2).
    ///
    /// Maximum of `(tf · (k1 + 1)) / (tf + k1 · (1 - b + b · (L / avg_L)))`
    /// over every posting in the list. `0.0` means "unset"; scorers
    /// fall back to the loose `k1 + 1` synthetic ceiling.
    pub max_score_factor: f32,
    /// Per-block (`BLOCK_SIZE`-wide) max-impact metadata used by
    /// Block-Max-WAND (#403 PR-C). Empty if not available
    /// (legacy v2 dictionaries or aggregated cross-segment views) —
    /// scorers then fall back to [`Self::max_score_factor`].
    pub block_max: Vec<BlockMax>,
}

impl TermInfo {
    /// Create new term info with `max_score_factor = 0.0` (loose bound).
    ///
    /// Use [`TermInfo::with_max_score_factor`] to build a value with the
    /// tightened block-max bound from the index. Existing call sites
    /// keep working — the searcher silently falls back to the synthetic
    /// upper bound when the field is `0.0`.
    pub fn new(
        posting_offset: u64,
        posting_length: u64,
        doc_frequency: u64,
        total_frequency: u64,
    ) -> Self {
        TermInfo {
            posting_offset,
            posting_length,
            doc_frequency,
            total_frequency,
            max_score_factor: 0.0,
            block_max: Vec::new(),
        }
    }

    /// Create new term info with a precomputed tightened max-score
    /// factor (#403 PR-B2). Pass `0.0` if the factor is not available
    /// — scorers will fall back to the loose `k1 + 1` bound.
    pub fn with_max_score_factor(
        posting_offset: u64,
        posting_length: u64,
        doc_frequency: u64,
        total_frequency: u64,
        max_score_factor: f32,
    ) -> Self {
        TermInfo {
            posting_offset,
            posting_length,
            doc_frequency,
            total_frequency,
            max_score_factor,
            block_max: Vec::new(),
        }
    }

    /// Create new term info with both the term-level factor and the
    /// per-block max-impact metadata used by Block-Max-WAND (#403 PR-C).
    pub fn with_block_max(
        posting_offset: u64,
        posting_length: u64,
        doc_frequency: u64,
        total_frequency: u64,
        max_score_factor: f32,
        block_max: Vec<BlockMax>,
    ) -> Self {
        TermInfo {
            posting_offset,
            posting_length,
            doc_frequency,
            total_frequency,
            max_score_factor,
            block_max,
        }
    }
}

/// Lucene `BlockTreeTermsWriter`-style block term dictionary
/// (Issue [#487](https://github.com/mosuka/laurus/issues/487)).
///
/// Replaces the legacy parallel-array / `AHashMap`-backed dictionary
/// triad (removed in #487 Phase 9) with a two-layer structure designed
/// for production scale (10M-100M+ terms / segment):
///
/// - an [`fst::Map`] keyed by each block's **last** term, valued by the
///   block's start offset within the BlockSection — typically 1-2
///   orders of magnitude smaller than a flat per-term FST
/// - a flat BlockSection holding 128-term blocks, each with
///   front-coded term bytes, a bit-packed
///   [`term_info_block::FixedTermInfoBlock`], and a
///   [`block_max_data::BlockMaxData`] for variable-length per-term
///   block_max arrays
///
/// Lookup: `fst.range().ge(target).into_stream().next()` identifies the
/// block containing `target`; an in-block linear scan over the
/// front-coded term bytes finds the exact match.
///
/// `iter()` walks the BlockSection sequentially without consulting the
/// FST, so per-step cost is the front-coding decode (≈ 5–10 ns) rather
/// than FST DFA traversal.
///
/// # Two-layer storage
///
/// The dictionary keeps **two equivalent representations** of the same
/// data, optimised for different access patterns:
///
/// 1. **Disk-format scratch** (`fst` + `block_section`) — the compact
///    FST + 128-term BlockSection layout written to `.dict`. Kept in
///    memory so [`Self::write_to_storage`] can re-emit the segment
///    without re-encoding from scratch.
/// 2. **In-memory query layer** (`map` + `sorted_terms` + `term_infos`)
///    — populated at build / load time so that `get` / `iter` /
///    `find_prefix` operate at parallel-array / `AHashMap` speed
///    rather than paying the in-block linear scan that the
///    disk-format alone would require.
///
/// Both representations share a single source of truth: the
/// `term_infos` array is `Arc<[TermInfo]>` (single copy across `map`'s
/// ordinal index and any reader), and `sorted_terms` parallels it in
/// ascending term order.
#[derive(Clone, Debug)]
pub struct BlockTermDictionary {
    // ----- Disk-format scratch -----
    /// FST: each block's last term bytes → block start offset within
    /// `block_section`.
    fst: Arc<FstMap<Vec<u8>>>,
    /// Concatenated block bytes (one entry per block). Shared via
    /// `Arc` so that cloning a `BlockTermDictionary` does not copy
    /// the term data.
    block_section: Arc<[u8]>,

    // ----- In-memory query layer -----
    /// `term -> ordinal (0..total_term_count)` index for `O(1)` `get`.
    map: AHashMap<String, u32>,
    /// `ordinal -> term`, in ascending term order. Used by `iter`,
    /// `find_prefix`, and `find_range`.
    sorted_terms: Vec<String>,
    /// `ordinal -> TermInfo`. Single copy shared via `Arc`.
    term_infos: Arc<[TermInfo]>,

    /// Total number of terms across all blocks.
    total_term_count: u64,
    /// Number of blocks in `block_section`.
    block_count: u32,
    /// On-disk version of the segment's posting-list format implied by
    /// this dictionary (#503). The dictionary itself does not contain
    /// the posting lists, but it is the only segment-level file with a
    /// magic + version, so its version implies the format used by the
    /// sibling `.post` file. `1` = legacy (no on-disk skip levels);
    /// `2` = multi-level skip table embedded per posting list (#503);
    /// `3` = the weights section is gated by an `any_weights` header
    /// byte and omitted when every weight is `1.0` (#553).
    ///
    /// Because this drives which decoder `SegmentReader::postings`
    /// selects, bumping the posting format **requires** bumping this
    /// version in the same change — otherwise new payloads are handed to
    /// an old decoder and misparse without erroring.
    posting_format_version: u32,
}

impl BlockTermDictionary {
    /// Look up a term and return a borrowed [`TermInfo`] reference.
    ///
    /// Uses the in-memory `AHashMap` index for `O(1)` lookup. The
    /// disk-format fields (`fst`, `block_section`) are not consulted —
    /// they exist only so the dictionary can be re-serialised via
    /// [`Self::write_to_storage`].
    ///
    /// Returns `None` if `term` is not present in the dictionary.
    /// Callers that need an owned `TermInfo` should call `.cloned()`.
    pub fn get(&self, term: &str) -> Option<&TermInfo> {
        let ordinal = *self.map.get(term)? as usize;
        Some(&self.term_infos[ordinal])
    }

    /// Iterate `(term, term_info)` borrowed pairs in ascending term
    /// order. Yields references — no per-term `String` / `TermInfo`
    /// clones — so iteration cost matches the legacy parallel-array
    /// representation.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TermInfo)> + '_ {
        self.sorted_terms
            .iter()
            .zip(self.term_infos.iter())
            .map(|(term, info)| (term.as_str(), info))
    }

    /// Ordinal of the first term `>= target` (Issue #845).
    ///
    /// A cursor entry point over the ascending `sorted_terms` array:
    /// binary search returning the smallest index whose term is not
    /// less than `target` (`sorted_terms.len()` when every term is
    /// smaller). Together with [`Self::entry_at`] this lets callers
    /// iterate a bounded range lazily instead of collecting.
    ///
    /// # Arguments
    ///
    /// * `target` - The full dictionary key (e.g. `"field:term"`).
    ///
    /// # Returns
    ///
    /// The ordinal of the first term `>= target`.
    pub(crate) fn seek_index(&self, target: &str) -> usize {
        self.sorted_terms
            .partition_point(|term| term.as_str() < target)
    }

    /// Borrowed `(term, term_info)` at ordinal `idx` (Issue #845).
    ///
    /// # Arguments
    ///
    /// * `idx` - Ordinal into the ascending term order.
    ///
    /// # Returns
    ///
    /// `None` when `idx` is past the end of the dictionary.
    pub(crate) fn entry_at(&self, idx: usize) -> Option<(&str, &TermInfo)> {
        let term = self.sorted_terms.get(idx)?;
        Some((term.as_str(), &self.term_infos[idx]))
    }

    /// Collect borrowed `(term, term_info)` pairs whose term begins
    /// with `prefix`, in ascending term order.
    ///
    /// Implemented eagerly via binary search over `sorted_terms`. The
    /// FST is not consulted on this path.
    pub fn find_prefix(&self, prefix: &str) -> Vec<(&str, &TermInfo)> {
        // Binary-search for the first ordinal whose term ≥ prefix.
        let start = self
            .sorted_terms
            .binary_search_by(|probe| probe.as_str().cmp(prefix))
            .unwrap_or_else(|insert_at| insert_at);

        let mut result = Vec::new();
        for i in start..self.sorted_terms.len() {
            let term = &self.sorted_terms[i];
            if term.starts_with(prefix) {
                result.push((term.as_str(), &self.term_infos[i]));
            } else {
                break;
            }
        }
        result
    }

    /// Collect borrowed `(term, term_info)` pairs in the half-open
    /// range `[start, end)`, in ascending term order.
    ///
    /// Implemented eagerly via binary search over `sorted_terms`.
    pub fn find_range(&self, start: &str, end: &str) -> Vec<(&str, &TermInfo)> {
        if start >= end {
            return Vec::new();
        }

        let start_idx = self
            .sorted_terms
            .binary_search_by(|probe| probe.as_str().cmp(start))
            .unwrap_or_else(|insert_at| insert_at);
        let end_idx = self
            .sorted_terms
            .binary_search_by(|probe| probe.as_str().cmp(end))
            .unwrap_or_else(|insert_at| insert_at);

        let mut result = Vec::with_capacity(end_idx.saturating_sub(start_idx));
        for i in start_idx..end_idx.min(self.sorted_terms.len()) {
            result.push((self.sorted_terms[i].as_str(), &self.term_infos[i]));
        }
        result
    }

    /// Total number of terms in the dictionary.
    pub fn len(&self) -> u64 {
        self.total_term_count
    }

    /// Returns `true` if the dictionary contains no terms.
    pub fn is_empty(&self) -> bool {
        self.total_term_count == 0
    }

    /// Number of blocks in the dictionary's BlockSection.
    pub fn block_count(&self) -> u32 {
        self.block_count
    }

    /// Read the dictionary from storage.
    ///
    /// Format (`LTDD` layout):
    ///
    /// ```text
    /// [magic:              u32 = 0x4C544444 "LTDD"]
    /// [version:            u32 = 1 | 2 | 3]
    /// [fst_bytes_len:      u32]
    /// [fst_bytes:          u8 × fst_bytes_len]
    /// [block_section_len:  u32]
    /// [block_section:      u8 × block_section_len]
    /// [total_term_count:   u64]
    /// [block_count:        u32]
    /// [reserved:           u32 = 0]
    /// ```
    ///
    /// Rejects legacy `STDC` (sorted) / `HTDC` (hash) magic numbers
    /// with an explicit error — pre-release semantics apply.
    pub fn read_from_storage<R: StorageInput>(reader: &mut StructReader<R>) -> Result<Self> {
        let magic = reader.read_u32()?;
        match magic {
            MAGIC_LTDD => {} // proceed
            LEGACY_MAGIC_STDC | LEGACY_MAGIC_HTDC => {
                return Err(LaurusError::index(
                    "Unsupported legacy term dictionary format. Rebuild required.",
                ));
            }
            _ => {
                return Err(LaurusError::index(format!(
                    "Invalid term dictionary magic: 0x{magic:08X}"
                )));
            }
        }

        let version = reader.read_u32()?;
        if !matches!(version, 1..=3) {
            return Err(LaurusError::index(format!(
                "Unsupported BlockTermDictionary version: {version}"
            )));
        }

        let fst_bytes_len = reader.read_u32()? as usize;
        let fst_bytes = reader.read_raw(fst_bytes_len)?;
        let fst = FstMap::new(fst_bytes)
            .map_err(|e| LaurusError::index(format!("FST parse error: {e}")))?;

        let block_section_len = reader.read_u32()? as usize;
        let block_section_vec = reader.read_raw(block_section_len)?;

        let total_term_count = reader.read_u64()?;
        let block_count = reader.read_u32()?;
        let _reserved = reader.read_u32()?;

        let block_section: Arc<[u8]> = Arc::from(block_section_vec.into_boxed_slice());
        // Every term costs at least its 4-byte block-max offset, so the
        // count cannot exceed a quarter of the BlockSection (Issue #1220).
        let expected_terms = checked_capacity_u64(
            total_term_count,
            4,
            block_section.len() as u64,
            "term dictionary term count",
        )?;

        // Populate the in-memory query layer by streaming the
        // BlockSection once. After this point, all hot-path queries go
        // through the in-memory structures; the FST + block_section
        // remain only for `write_to_storage`.
        let (map, sorted_terms, term_infos) =
            populate_in_memory_layer(&block_section, block_count, expected_terms)?;

        Ok(BlockTermDictionary {
            fst: Arc::new(fst),
            block_section,
            map,
            sorted_terms,
            term_infos,
            total_term_count,
            block_count,
            posting_format_version: version,
        })
    }

    /// Return the on-disk posting-list format version implied by this
    /// dictionary (#503). Readers dispatch between
    /// [`super::super::inverted::core::posting::PostingList::decode_soa`]
    /// (v1) and `decode_soa_v2` (v2) based on this value.
    pub fn posting_format_version(&self) -> u32 {
        self.posting_format_version
    }

    /// Write the dictionary to storage in the v3 `LTDD` layout.
    /// See [`Self::read_from_storage`] for the byte layout.
    pub fn write_to_storage<W: StorageOutput>(&self, writer: &mut StructWriter<W>) -> Result<()> {
        writer.write_u32(MAGIC_LTDD)?;
        // Stamp the version this dictionary actually carries rather than
        // a literal. Fresh builds set it to the current format in
        // `build()`; a dictionary loaded from disk keeps whatever its
        // segment was written with, so re-serialising one can never
        // claim a newer format than its sibling `.post` file holds
        // (#553).
        //
        // v1 = no on-disk skip levels; v2 = skip table per posting list
        // (#503); v3 = weights section gated by `any_weights` (#553).
        writer.write_u32(self.posting_format_version)?;

        let fst_bytes = self.fst.as_fst().as_inner();
        writer.write_u32(
            u32::try_from(fst_bytes.len())
                .map_err(|_| LaurusError::index("FST bytes length exceeds u32::MAX"))?,
        )?;
        writer.write_raw(fst_bytes)?;

        writer.write_u32(
            u32::try_from(self.block_section.len())
                .map_err(|_| LaurusError::index("BlockSection length exceeds u32::MAX"))?,
        )?;
        writer.write_raw(&self.block_section)?;

        writer.write_u64(self.total_term_count)?;
        writer.write_u32(self.block_count)?;
        writer.write_u32(0)?; // reserved

        Ok(())
    }

    /// Get aggregate statistics about the dictionary.
    ///
    /// Walks the dictionary once to compute term-length statistics and
    /// frequency totals. `memory_size` reports the encoded size of the
    /// FST plus the BlockSection in bytes.
    pub fn stats(&self) -> DictionaryStats {
        let term_count = self.total_term_count as usize;

        let mut total_term_length = 0usize;
        let mut total_doc_frequency = 0u64;
        let mut total_term_frequency = 0u64;

        for (term, info) in self.iter() {
            total_term_length += term.len();
            total_doc_frequency += info.doc_frequency;
            total_term_frequency += info.total_frequency;
        }

        let avg_term_length = if term_count > 0 {
            total_term_length as f64 / term_count as f64
        } else {
            0.0
        };

        let memory_size = self.fst.as_fst().as_inner().len() + self.block_section.len();

        DictionaryStats {
            term_count,
            memory_size,
            avg_term_length,
            total_doc_frequency,
            total_term_frequency,
        }
    }
}

/// Builder for creating term dictionaries.
pub struct TermDictionaryBuilder {
    terms: BTreeMap<String, TermInfo>,
}

impl TermDictionaryBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        TermDictionaryBuilder {
            terms: BTreeMap::new(),
        }
    }

    /// Add a term with its info.
    pub fn add_term(&mut self, term: String, info: TermInfo) {
        self.terms.insert(term, info);
    }

    /// Build a [`BlockTermDictionary`] (Lucene BlockTreeTerms-style,
    /// Issue [#487](https://github.com/mosuka/laurus/issues/487)).
    ///
    /// Splits the sorted term map into 128-term blocks, encodes each
    /// block via [`builder::encode_block_into`], and indexes the
    /// blocks' last terms in an FST keyed by block byte offset.
    ///
    /// Returns an empty dictionary (no FST keys, empty BlockSection)
    /// for an empty builder.
    pub fn build(self) -> Result<BlockTermDictionary> {
        let total_term_count = self.terms.len() as u64;

        if self.terms.is_empty() {
            // Build an empty FST so callers can still call `get` /
            // `iter` without special-casing.
            let fst_builder = fst::MapBuilder::memory();
            let bytes = fst_builder
                .into_inner()
                .map_err(|e| LaurusError::index(format!("FST finish error: {e}")))?;
            let fst = FstMap::new(bytes)
                .map_err(|e| LaurusError::index(format!("FST construct error: {e}")))?;
            return Ok(BlockTermDictionary {
                fst: Arc::new(fst),
                block_section: Arc::from(Vec::<u8>::new().into_boxed_slice()),
                map: AHashMap::new(),
                sorted_terms: Vec::new(),
                term_infos: Arc::from(Vec::<TermInfo>::new().into_boxed_slice()),
                total_term_count: 0,
                block_count: 0,
                // Fresh builds always emit the latest format (#503).
                posting_format_version: 3,
            });
        }

        // BTreeMap iteration is already ascending key order.
        let entries: Vec<(String, TermInfo)> = self.terms.into_iter().collect();

        let mut block_section: Vec<u8> = Vec::new();
        let mut fst_builder = fst::MapBuilder::memory();
        let mut block_count: u32 = 0;

        // While we encode the disk-format scratch we also populate the
        // in-memory query layer in lock-step. This avoids walking the
        // BlockSection again post-build to materialise the AHashMap +
        // sorted_terms + term_infos arrays.
        let mut map = AHashMap::with_capacity(entries.len());
        let mut sorted_terms: Vec<String> = Vec::with_capacity(entries.len());
        let mut term_infos: Vec<TermInfo> = Vec::with_capacity(entries.len());
        let mut next_ordinal: u32 = 0;

        for chunk in entries.chunks(term_info_block::BLOCK_TERM_COUNT) {
            let block_offset = block_section.len() as u64;

            let term_byte_refs: Vec<&[u8]> = chunk.iter().map(|(t, _)| t.as_bytes()).collect();
            let fixed_infos: Vec<term_info_block::FixedTermInfo> = chunk
                .iter()
                .map(|(_, info)| term_info_block::FixedTermInfo {
                    posting_offset: info.posting_offset,
                    posting_length: info.posting_length,
                    doc_frequency: info.doc_frequency,
                    total_frequency: info.total_frequency,
                    max_score_factor: info.max_score_factor,
                })
                .collect();
            let block_max_per_term: Vec<Vec<BlockMax>> = chunk
                .iter()
                .map(|(_, info)| info.block_max.clone())
                .collect();

            builder::encode_block_into(
                &mut block_section,
                &term_byte_refs,
                &fixed_infos,
                &block_max_per_term,
            );

            // Mirror the chunk into the in-memory layer.
            for (term, info) in chunk {
                map.insert(term.clone(), next_ordinal);
                sorted_terms.push(term.clone());
                term_infos.push(info.clone());
                next_ordinal += 1;
            }

            let last_term = chunk.last().expect("chunk non-empty").0.as_bytes();
            fst_builder
                .insert(last_term, block_offset)
                .map_err(|e| LaurusError::index(format!("FST insert error: {e}")))?;
            block_count += 1;
        }

        let fst_bytes = fst_builder
            .into_inner()
            .map_err(|e| LaurusError::index(format!("FST finish error: {e}")))?;
        let fst = FstMap::new(fst_bytes)
            .map_err(|e| LaurusError::index(format!("FST construct error: {e}")))?;

        Ok(BlockTermDictionary {
            fst: Arc::new(fst),
            block_section: Arc::from(block_section.into_boxed_slice()),
            map,
            sorted_terms,
            term_infos: Arc::from(term_infos.into_boxed_slice()),
            total_term_count,
            block_count,
            // Fresh builds always emit the latest format (#503).
            posting_format_version: 3,
        })
    }

    /// Get the current number of terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl Default for TermDictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A loaded dictionary's in-memory query layer: the term → ordinal map,
/// the terms in sorted order, and their [`TermInfo`]s by ordinal.
type InMemoryLayer = (AHashMap<String, u32>, Vec<String>, Arc<[TermInfo]>);

/// Stream the BlockSection bytes once, building the in-memory query
/// layer (`map`, `sorted_terms`, `term_infos`) for a freshly-loaded
/// [`BlockTermDictionary`].
///
/// Used by [`BlockTermDictionary::read_from_storage`] only. Build-time
/// population happens inline in [`TermDictionaryBuilder::build`] to
/// avoid a redundant BlockSection walk.
///
/// # Errors
///
/// Returns an error when the BlockSection yields a different number of
/// terms than `total_term_count`, the header's count.
fn populate_in_memory_layer(
    block_section: &[u8],
    block_count: u32,
    total_term_count: usize,
) -> Result<InMemoryLayer> {
    let mut map = AHashMap::with_capacity(total_term_count);
    let mut sorted_terms: Vec<String> = Vec::with_capacity(total_term_count);
    let mut term_infos: Vec<TermInfo> = Vec::with_capacity(total_term_count);

    let iter = block_reader::BlockSectionIter::new(block_section, block_count);
    for (ordinal, (term, info)) in iter.enumerate() {
        map.insert(term.clone(), ordinal as u32);
        sorted_terms.push(term);
        term_infos.push(info);
    }

    // `BlockSectionIter` ends silently at a block it cannot parse, and the
    // writer records exactly the terms it encodes. A dictionary cut short
    // would otherwise load without its later terms — and a merge walking
    // it would drop them for good (Issue #1220).
    if sorted_terms.len() != total_term_count {
        return Err(LaurusError::index(format!(
            "term dictionary: read {} terms but the header declares {total_term_count} — \
             segment is corrupted",
            sorted_terms.len()
        )));
    }

    Ok((map, sorted_terms, Arc::from(term_infos.into_boxed_slice())))
}

/// Dictionary statistics.
#[derive(Debug, Clone)]
pub struct DictionaryStats {
    /// Number of terms.
    pub term_count: usize,
    /// Total size in memory (bytes).
    pub memory_size: usize,
    /// Average term length.
    pub avg_term_length: f64,
    /// Total document frequency.
    pub total_doc_frequency: u64,
    /// Total term frequency.
    pub total_term_frequency: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;
    use std::sync::Arc;

    fn create_test_term_info(offset: u64) -> TermInfo {
        TermInfo::new(offset, 100, 5, 20)
    }

    #[test]
    fn test_dictionary_builder_basic() {
        let mut builder = TermDictionaryBuilder::new();
        assert!(builder.is_empty());

        builder.add_term("test".to_string(), create_test_term_info(0));
        assert_eq!(builder.len(), 1);

        let dict = builder.build().unwrap();
        assert_eq!(dict.len(), 1);
        assert!(dict.get("test").is_some());
    }

    // ----- Posting format version (#553) -----

    /// Copy a dictionary file, overwriting only its `version` header
    /// field.
    ///
    /// Everything after the header is version-independent, so a body
    /// written by the current code with an older stamp is byte-identical
    /// to what that older release produced — which is what makes this a
    /// genuine backward-compatibility fixture rather than a re-encoding.
    /// The trailing CRC is not recomputed because `read_from_storage`
    /// does not verify it (checksum validation is opt-in via
    /// `StructReader::verify_checksum`).
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage holding both files.
    /// * `src` - Path of the file to copy.
    /// * `dst` - Path to write the restamped copy to.
    /// * `version` - Version number to stamp.
    fn stamp_version(storage: &Arc<MemoryStorage>, src: &str, dst: &str, version: u32) {
        use std::io::{Read, Write};

        let mut bytes = Vec::new();
        {
            let mut input = storage.open_input(src).unwrap();
            input.read_to_end(&mut bytes).unwrap();
        }
        // [magic: u32][version: u32], little-endian per `StructWriter`.
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        {
            let mut output = storage.create_output(dst).unwrap();
            output.write_all(&bytes).unwrap();
            output.flush_and_sync().unwrap();
            output.close().unwrap();
        }
    }

    /// #553 — the `.dict` header version is what tells the reader which
    /// `.post` decoder to use, so a fresh build must stamp the current
    /// version and a round trip must carry it back unchanged.
    ///
    /// Nothing asserted this value before, which is how the two version
    /// gates could have drifted apart unnoticed.
    #[test]
    fn fresh_dictionary_stamps_and_round_trips_posting_format_v3() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let mut builder = TermDictionaryBuilder::new();
        builder.add_term("alpha".to_string(), create_test_term_info(0));
        builder.add_term("beta".to_string(), create_test_term_info(100));
        let dict = builder.build().unwrap();

        assert_eq!(
            dict.posting_format_version(),
            3,
            "fresh builds must emit the current posting format"
        );

        let path = "dict_v3.bin";
        {
            let output = storage.create_output(path).unwrap();
            let mut writer = StructWriter::new(output);
            dict.write_to_storage(&mut writer).unwrap();
            writer.close().unwrap();
        }
        let input = storage.open_input(path).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let loaded = BlockTermDictionary::read_from_storage(&mut reader).unwrap();

        assert_eq!(loaded.posting_format_version(), 3);
        assert!(loaded.get("alpha").is_some());
        assert!(loaded.get("beta").is_some());
    }

    /// #553 — an existing v2 dictionary must keep loading and must keep
    /// reporting **2**, so its sibling `.post` file is decoded with the
    /// v2 decoder rather than the v3 one.
    ///
    /// This is the backward-compatibility guarantee for every index
    /// written before this change. `index-interop` CI cannot cover it —
    /// both of its jobs build the same commit, so it tests platforms,
    /// not versions.
    #[test]
    fn v2_dictionary_still_loads_and_reports_v2() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let mut builder = TermDictionaryBuilder::new();
        builder.add_term("gamma".to_string(), create_test_term_info(0));
        let dict = builder.build().unwrap();

        // Rewrite the version field in place: everything after the
        // header is version-independent, so a v3 body with a v2 stamp is
        // byte-identical to what the previous release produced.
        let path = "dict_v2.bin";
        {
            let output = storage.create_output(path).unwrap();
            let mut writer = StructWriter::new(output);
            dict.write_to_storage(&mut writer).unwrap();
            writer.close().unwrap();
        }
        let patched = "dict_v2_patched.bin";
        stamp_version(&storage, path, patched, 2);

        let input = storage.open_input(patched).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let loaded = BlockTermDictionary::read_from_storage(&mut reader).unwrap();

        assert_eq!(
            loaded.posting_format_version(),
            2,
            "a v2 dictionary must not be silently promoted to v3"
        );
        assert!(loaded.get("gamma").is_some());

        // Re-serialising it must keep the v2 stamp. Writing a literal
        // here instead of the carried version would tell the reader to
        // use the v3 posting decoder on a sibling `.post` file that is
        // still v2 — a silent misparse, and the reason
        // `write_to_storage` stamps `self.posting_format_version`.
        let rewritten = "dict_v2_rewritten.bin";
        {
            let output = storage.create_output(rewritten).unwrap();
            let mut writer = StructWriter::new(output);
            loaded.write_to_storage(&mut writer).unwrap();
            writer.close().unwrap();
        }
        let input = storage.open_input(rewritten).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let again = BlockTermDictionary::read_from_storage(&mut reader).unwrap();
        assert_eq!(
            again.posting_format_version(),
            2,
            "re-serialising a v2 dictionary must not promote it to v3"
        );
    }

    /// #553 — an unknown version must be rejected rather than assumed
    /// compatible, which is what keeps the exact-match dispatch in
    /// `SegmentReader::postings` sound.
    #[test]
    fn unknown_dictionary_version_is_rejected() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let mut builder = TermDictionaryBuilder::new();
        builder.add_term("delta".to_string(), create_test_term_info(0));
        let dict = builder.build().unwrap();

        let path = "dict_v99_src.bin";
        {
            let output = storage.create_output(path).unwrap();
            let mut writer = StructWriter::new(output);
            dict.write_to_storage(&mut writer).unwrap();
            writer.close().unwrap();
        }
        let patched = "dict_v99.bin";
        stamp_version(&storage, path, patched, 99);

        let input = storage.open_input(patched).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let err = BlockTermDictionary::read_from_storage(&mut reader).unwrap_err();
        assert!(
            err.to_string()
                .contains("Unsupported BlockTermDictionary version"),
            "unexpected error: {err}"
        );
    }

    // ----- BlockTermDictionary tests (#487 PR1) -----

    fn make_test_term_info_with_block_max(offset: u64, block_max: Vec<BlockMax>) -> TermInfo {
        TermInfo {
            posting_offset: offset,
            posting_length: 100,
            doc_frequency: 5,
            total_frequency: 20,
            max_score_factor: 1.0,
            block_max,
        }
    }

    #[test]
    fn block_dict_empty_builder_yields_empty_dict() {
        let builder = TermDictionaryBuilder::new();
        let dict = builder.build().unwrap();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
        assert_eq!(dict.block_count(), 0);
        assert!(dict.get("anything").is_none());
        assert_eq!(dict.iter().count(), 0);
    }

    #[test]
    fn block_dict_single_term_round_trip() {
        let mut builder = TermDictionaryBuilder::new();
        builder.add_term("hello".to_string(), create_test_term_info(42));
        let dict = builder.build().unwrap();
        assert_eq!(dict.len(), 1);
        assert_eq!(dict.block_count(), 1);

        let info = dict.get("hello").unwrap();
        assert_eq!(info.posting_offset, 42);
        assert!(dict.get("missing").is_none());
    }

    #[test]
    fn block_dict_get_within_single_block() {
        let mut builder = TermDictionaryBuilder::new();
        for (i, term) in ["apple", "banana", "cherry", "date"].iter().enumerate() {
            builder.add_term(term.to_string(), create_test_term_info(i as u64 * 100));
        }
        let dict = builder.build().unwrap();
        assert_eq!(dict.len(), 4);
        assert_eq!(dict.block_count(), 1);

        for (i, term) in ["apple", "banana", "cherry", "date"].iter().enumerate() {
            assert_eq!(dict.get(term).unwrap().posting_offset, i as u64 * 100);
        }
        assert!(dict.get("aardvark").is_none());
        assert!(dict.get("blueberry").is_none());
        assert!(dict.get("zulu").is_none());
    }

    #[test]
    fn block_dict_iter_yields_in_sorted_order() {
        let mut builder = TermDictionaryBuilder::new();
        for term in ["zulu", "alpha", "mike", "bravo"] {
            builder.add_term(term.to_string(), create_test_term_info(0));
        }
        let dict = builder.build().unwrap();
        let collected: Vec<String> = dict.iter().map(|(t, _)| t.to_string()).collect();
        assert_eq!(collected, vec!["alpha", "bravo", "mike", "zulu"]);
    }

    /// Issue #845: `seek_index` returns the first ordinal `>= target`
    /// and `entry_at` reads it, including boundaries.
    #[test]
    fn block_dict_seek_index_and_entry_at() {
        let mut builder = TermDictionaryBuilder::new();
        for term in ["a:x", "a:y", "b:m", "b:n", "c:z"] {
            builder.add_term(term.to_string(), create_test_term_info(1));
        }
        let dict = builder.build().unwrap();

        // Exact hit, in-between miss, prefix boundary, past-the-end.
        assert_eq!(dict.seek_index("a:x"), 0);
        assert_eq!(dict.seek_index("a:xx"), 1, "between a:x and a:y");
        assert_eq!(dict.seek_index("b:"), 2, "field-prefix start of b");
        assert_eq!(dict.seek_index("d:"), 5, "past every term");

        assert_eq!(dict.entry_at(2).map(|(t, _)| t), Some("b:m"));
        assert_eq!(dict.entry_at(4).map(|(t, _)| t), Some("c:z"));
        assert!(dict.entry_at(5).is_none(), "past-the-end yields None");
    }

    #[test]
    fn block_dict_multi_block_get_hit_and_miss() {
        // 300 terms → 3 blocks of 128/128/44 (BLOCK_TERM_COUNT = 128).
        let mut builder = TermDictionaryBuilder::new();
        for i in 0..300 {
            builder.add_term(format!("term{i:04}"), create_test_term_info(i as u64));
        }
        let dict = builder.build().unwrap();
        assert_eq!(dict.len(), 300);
        assert_eq!(dict.block_count(), 3);

        // Hits across all blocks.
        for i in [0, 1, 100, 127, 128, 129, 200, 299] {
            let key = format!("term{i:04}");
            assert_eq!(
                dict.get(&key).unwrap().posting_offset,
                i as u64,
                "miss on hit probe {key}"
            );
        }

        // Misses (after, between, before existing keys).
        assert!(dict.get("term0300").is_none());
        assert!(dict.get("term9999").is_none());
        assert!(dict.get("aaa").is_none());
        assert!(dict.get("zzz").is_none());
    }

    #[test]
    fn block_dict_iter_walks_multi_block() {
        let mut builder = TermDictionaryBuilder::new();
        for i in 0..200 {
            builder.add_term(format!("term{i:04}"), create_test_term_info(i as u64));
        }
        let dict = builder.build().unwrap();
        let collected: Vec<(String, u64)> = dict
            .iter()
            .map(|(t, info)| (t.to_string(), info.posting_offset))
            .collect();
        assert_eq!(collected.len(), 200);
        for (i, (term, offset)) in collected.iter().enumerate() {
            assert_eq!(term, &format!("term{i:04}"));
            assert_eq!(*offset, i as u64);
        }
    }

    #[test]
    fn block_dict_find_prefix_within_block() {
        let mut builder = TermDictionaryBuilder::new();
        for term in ["alpha", "apple", "apricot", "axis", "banana"] {
            builder.add_term(term.to_string(), create_test_term_info(0));
        }
        let dict = builder.build().unwrap();

        let ap = dict.find_prefix("ap");
        let ap_terms: Vec<&str> = ap.iter().map(|(t, _)| *t).collect();
        assert_eq!(ap_terms, vec!["apple", "apricot"]);

        // Empty prefix → all terms.
        let all = dict.find_prefix("");
        assert_eq!(all.len(), 5);

        // No matches.
        let zzz = dict.find_prefix("zzz");
        assert!(zzz.is_empty());
    }

    #[test]
    fn block_dict_find_prefix_across_block_boundary() {
        // Force a prefix to span the block boundary at 128.
        let mut builder = TermDictionaryBuilder::new();
        for i in 0..200 {
            builder.add_term(format!("term{i:04}"), create_test_term_info(i as u64));
        }
        let dict = builder.build().unwrap();
        // "term01" matches term0100..term0199 → 100 entries spanning
        // both blocks (128-term boundary at "term0127"/"term0128").
        let matches = dict.find_prefix("term01");
        assert_eq!(matches.len(), 100);
        assert_eq!(matches[0].0, "term0100");
        assert_eq!(matches[99].0, "term0199");
    }

    #[test]
    fn block_dict_find_range_basic() {
        let mut builder = TermDictionaryBuilder::new();
        for term in ["apple", "banana", "cherry", "date", "fig"] {
            builder.add_term(term.to_string(), create_test_term_info(0));
        }
        let dict = builder.build().unwrap();
        let r = dict.find_range("banana", "fig");
        let r_terms: Vec<&str> = r.iter().map(|(t, _)| *t).collect();
        assert_eq!(r_terms, vec!["banana", "cherry", "date"]);

        // Empty range when start >= end.
        assert!(dict.find_range("date", "banana").is_empty());
        assert!(dict.find_range("date", "date").is_empty());
    }

    #[test]
    fn block_dict_term_info_with_block_max_round_trip() {
        let mut builder = TermDictionaryBuilder::new();
        let bm = vec![
            BlockMax {
                last_doc_id: 5,
                max_factor: 0.5,
            },
            BlockMax {
                last_doc_id: 10,
                max_factor: 1.5,
            },
        ];
        builder.add_term(
            "hello".to_string(),
            make_test_term_info_with_block_max(99, bm.clone()),
        );
        let dict = builder.build().unwrap();
        let info = dict.get("hello").unwrap();
        assert_eq!(info.posting_offset, 99);
        assert_eq!(info.block_max.len(), 2);
        assert_eq!(info.block_max[0].last_doc_id, 5);
        assert_eq!(info.block_max[1].last_doc_id, 10);
    }

    #[test]
    fn block_dict_clone_shares_storage() {
        let mut builder = TermDictionaryBuilder::new();
        for i in 0..50 {
            builder.add_term(format!("term{i:03}"), create_test_term_info(i as u64));
        }
        let dict = builder.build().unwrap();
        let dict2 = dict.clone();
        assert_eq!(dict.len(), dict2.len());
        assert_eq!(dict.get("term025").unwrap(), dict2.get("term025").unwrap());
        // Arc strong count should be > 1 because of clone.
        assert!(Arc::strong_count(&dict.fst) >= 2);
    }

    #[test]
    fn block_dict_stats() {
        let mut builder = TermDictionaryBuilder::new();
        builder.add_term("short".to_string(), TermInfo::new(0, 50, 1, 1));
        builder.add_term("longer_term".to_string(), TermInfo::new(50, 100, 5, 10));
        builder.add_term(
            "longest_term_here".to_string(),
            TermInfo::new(150, 200, 3, 8),
        );

        let dict = builder.build().unwrap();
        let stats = dict.stats();

        assert_eq!(stats.term_count, 3);
        assert!(stats.avg_term_length > 0.0);
        assert_eq!(stats.total_doc_frequency, 9); // 1 + 5 + 3
        assert_eq!(stats.total_term_frequency, 19); // 1 + 10 + 8
        assert!(stats.memory_size > 0);
    }

    /// Writes a 200-term dictionary to `name` and returns its bytes.
    fn written_dictionary_bytes(storage: &Arc<MemoryStorage>, name: &str) -> Vec<u8> {
        use std::io::Read;

        let mut builder = TermDictionaryBuilder::new();
        for i in 0..200u64 {
            builder.add_term(format!("term{i:04}"), create_test_term_info(i * 16));
        }
        let dict = builder.build().unwrap();
        let output = storage.create_output(name).unwrap();
        let mut writer = StructWriter::new(output);
        dict.write_to_storage(&mut writer).unwrap();
        writer.close().unwrap();

        let mut bytes = Vec::new();
        storage
            .open_input(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    /// Stores `total_term_count` into a dictionary file's trailer — the u64
    /// before `block_count`, `reserved` and the CRC — and loads it back.
    fn load_with_term_count(total_term_count: u64) -> Result<BlockTermDictionary> {
        use std::io::Write;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let mut bytes = written_dictionary_bytes(&storage, "dict.bin");
        let at = bytes.len() - 20;
        bytes[at..at + 8].copy_from_slice(&total_term_count.to_le_bytes());
        let mut output = storage.create_output("patched.bin").unwrap();
        output.write_all(&bytes).unwrap();
        output.close().unwrap();

        let input = storage.open_input("patched.bin").unwrap();
        BlockTermDictionary::read_from_storage(&mut StructReader::new(input).unwrap())
    }

    fn assert_corrupted<T>(result: Result<T>, label: &str) {
        match result {
            Ok(_) => panic!("expected a corruption error mentioning {label:?}"),
            Err(LaurusError::Index(msg)) => {
                assert!(msg.contains("corrupted") && msg.contains(label), "{msg}");
            }
            Err(other) => panic!("expected Index error, got {other:?}"),
        }
    }

    /// The term count sizes three in-memory tables, so it is bounded by the
    /// BlockSection before they are allocated (Issue #1220).
    #[test]
    fn a_term_count_the_block_section_cannot_hold_is_rejected() {
        assert!(
            load_with_term_count(200).is_ok(),
            "the file as written loads"
        );
        assert_corrupted(load_with_term_count(u64::MAX), "term dictionary term count");
    }

    /// A dictionary must yield every term its header declares. The block
    /// walk stops silently at a block it cannot parse, so a dictionary cut
    /// short would load without its later terms — and a merge walking it
    /// would drop them for good (Issue #1220).
    #[test]
    fn a_dictionary_yielding_fewer_terms_than_declared_is_rejected() {
        assert_corrupted(load_with_term_count(201), "read 200 terms");
    }

    #[test]
    fn block_dict_round_trip_via_storage() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut builder = TermDictionaryBuilder::new();
        for i in 0..200u64 {
            // 200 terms exercises 2 blocks (BLOCK_TERM_COUNT = 128 +
            // partial second block).
            let mut info = create_test_term_info(i * 16);
            // Sprinkle non-trivial block_max on every 5th term to
            // exercise the variable-length BlockMaxData encoding.
            if i.is_multiple_of(5) {
                info.block_max = vec![BlockMax {
                    last_doc_id: i,
                    max_factor: 1.0 + (i as f32) * 0.01,
                }];
            }
            builder.add_term(format!("term{i:04}"), info);
        }
        let original_dict = builder.build().unwrap();

        // Write
        {
            let output = storage.create_output("test_block_dict.bin").unwrap();
            let mut writer = StructWriter::new(output);
            original_dict.write_to_storage(&mut writer).unwrap();
            writer.close().unwrap();
        }

        // Read
        let loaded_dict = {
            let input = storage.open_input("test_block_dict.bin").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            BlockTermDictionary::read_from_storage(&mut reader).unwrap()
        };

        assert_eq!(loaded_dict.len(), original_dict.len());
        assert_eq!(loaded_dict.block_count(), original_dict.block_count());

        // All-term spot check.
        for i in 0..200 {
            let key = format!("term{i:04}");
            let orig = original_dict.get(&key).unwrap();
            let loaded = loaded_dict.get(&key).unwrap();
            assert_eq!(orig, loaded, "mismatch for {key}");
        }
        // iter order
        let orig_iter: Vec<_> = original_dict.iter().collect();
        let loaded_iter: Vec<_> = loaded_dict.iter().collect();
        assert_eq!(orig_iter.len(), loaded_iter.len());
        for (a, b) in orig_iter.iter().zip(loaded_iter.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn block_dict_read_rejects_legacy_stdc_magic() {
        // Synthesise a file starting with the legacy `STDC` magic.
        // Real legacy writers no longer exist (#487 Phase 9); we emit
        // just the magic so `read_from_storage` short-circuits before
        // trying to decode any payload.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("legacy_stdc.bin").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_u32(LEGACY_MAGIC_STDC).unwrap();
            writer.write_u32(3).unwrap(); // dummy version
            writer.close().unwrap();
        }

        let input = storage.open_input("legacy_stdc.bin").unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let err = BlockTermDictionary::read_from_storage(&mut reader);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("Unsupported legacy term dictionary format"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn block_dict_read_rejects_legacy_htdc_magic() {
        // Same idea as `..._stdc_magic`: emit only the legacy magic.
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("legacy_htdc.bin").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_u32(LEGACY_MAGIC_HTDC).unwrap();
            writer.write_u32(3).unwrap();
            writer.close().unwrap();
        }

        let input = storage.open_input("legacy_htdc.bin").unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let err = BlockTermDictionary::read_from_storage(&mut reader);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("Unsupported legacy term dictionary format"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn block_dict_read_rejects_unknown_magic() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        {
            let output = storage.create_output("garbage.bin").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_u32(0xDEADBEEF).unwrap(); // bogus magic
            writer.write_u32(0).unwrap();
            writer.close().unwrap();
        }
        let input = storage.open_input("garbage.bin").unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let err = BlockTermDictionary::read_from_storage(&mut reader);
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("Invalid term dictionary magic"));
    }

    #[test]
    fn block_dict_round_trip_empty() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let original_dict = TermDictionaryBuilder::new().build().unwrap();
        {
            let output = storage.create_output("empty.bin").unwrap();
            let mut writer = StructWriter::new(output);
            original_dict.write_to_storage(&mut writer).unwrap();
            writer.close().unwrap();
        }
        let input = storage.open_input("empty.bin").unwrap();
        let mut reader = StructReader::new(input).unwrap();
        let loaded = BlockTermDictionary::read_from_storage(&mut reader).unwrap();
        assert!(loaded.is_empty());
        assert_eq!(loaded.block_count(), 0);
    }

    #[test]
    fn block_dict_exact_block_boundary_term_count() {
        // Exactly 128 terms → 1 block, fully filled.
        let mut builder = TermDictionaryBuilder::new();
        for i in 0..128 {
            builder.add_term(format!("term{i:03}"), create_test_term_info(i as u64));
        }
        let dict = builder.build().unwrap();
        assert_eq!(dict.len(), 128);
        assert_eq!(dict.block_count(), 1);

        // 129 terms → 2 blocks (128 + 1).
        let mut builder2 = TermDictionaryBuilder::new();
        for i in 0..129 {
            builder2.add_term(format!("term{i:03}"), create_test_term_info(i as u64));
        }
        let dict2 = builder2.build().unwrap();
        assert_eq!(dict2.len(), 129);
        assert_eq!(dict2.block_count(), 2);
        assert_eq!(dict2.get("term128").unwrap().posting_offset, 128);
    }
}
