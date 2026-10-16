//! Inverted index reader implementation.
//!
//! This module provides a production-ready inverted index reader that efficiently
//! handles multiple segments, caching, and optimized posting list access.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use ahash::AHashMap;
use lru::LruCache;
use parking_lot::Mutex;
use roaring::RoaringTreemap;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::error::{LaurusError, Result};
use crate::lexical::core::document::Document;
use crate::lexical::core::field::FieldValue;
use crate::lexical::index::inverted::core::posting::{DecodedPostingList, Posting, PostingList};
use crate::lexical::index::inverted::core::terms::{
    InvertedIndexTerms, MergedInvertedIndexTerms, TermDictionaryAccess, Terms,
};
use crate::lexical::index::inverted::deleted_docs::DeletedDocs;
use crate::lexical::index::inverted::id_filter::{CommittedIdFilter, ID_FIELD, IdCheck};
use crate::lexical::index::inverted::posting_cache::PostingCache;
use crate::lexical::index::inverted::query_cache::QueryFilterCache;
use crate::lexical::index::inverted::segment::{Membership, SegmentInfo, SegmentKey};
use crate::lexical::index::inverted::segment_reader_cache::SegmentReaderCache;
use crate::lexical::index::structures::bkd_tree::{BKDReader, BKDTree};
use crate::lexical::index::structures::dictionary::BLOCK_SIZE;
use crate::lexical::index::structures::dictionary::BlockTermDictionary;
use crate::lexical::index::structures::dictionary::TermInfo;
use crate::lexical::index::structures::doc_values::DocValuesReader;
use crate::lexical::index::structures::norms::{NormsColumn, NormsReader};
use crate::lexical::query::Query;
use crate::lexical::reader::FieldStats;
use crate::lexical::reader::PostingIterator;
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::storage::structured::{
    ChecksumStatus, StructReader, ends_in_footer, verify_file_checksum,
};

/// Default [`InvertedIndexReaderConfig::max_cache_memory`] (128 MiB), shared
/// with `InvertedIndexConfig` so the index-level default cannot drift from
/// the reader's (Issue #1200).
pub(crate) const DEFAULT_MAX_CACHE_MEMORY: usize = 128 * 1024 * 1024;

/// Advanced index reader configuration.
#[derive(Clone)]
pub struct InvertedIndexReaderConfig {
    /// Maximum memory for caching (in bytes): the budget of the reader's
    /// term-info cache, and of each segment reader's posting cache. The
    /// caches grow as they fill; none of the budget is reserved up front.
    /// Default: 128 MiB.
    pub max_cache_memory: usize,

    /// Enable posting cache.
    pub enable_posting_cache: bool,

    /// Preload segments on open.
    pub preload_segments: bool,

    /// Maximum number of entries in the snapshot-scoped query / filter result
    /// cache (Issue #578). `0` disables the cache. See
    /// [`QueryFilterCache`](crate::lexical::index::inverted::query_cache::QueryFilterCache).
    pub query_filter_cache_capacity: usize,

    /// Analyzer for query term analysis.
    pub analyzer: Arc<dyn Analyzer>,
}

impl std::fmt::Debug for InvertedIndexReaderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndexReaderConfig")
            .field("max_cache_memory", &self.max_cache_memory)
            .field("enable_posting_cache", &self.enable_posting_cache)
            .field("preload_segments", &self.preload_segments)
            .field(
                "query_filter_cache_capacity",
                &self.query_filter_cache_capacity,
            )
            .field("analyzer", &self.analyzer.name())
            .finish()
    }
}

impl Default for InvertedIndexReaderConfig {
    fn default() -> Self {
        InvertedIndexReaderConfig {
            max_cache_memory: DEFAULT_MAX_CACHE_MEMORY,
            enable_posting_cache: true,
            preload_segments: false,
            query_filter_cache_capacity: 1024,
            analyzer: Arc::new(
                StandardAnalyzer::new().expect("StandardAnalyzer should be creatable"),
            ),
        }
    }
}

/// Advanced posting iterator for efficiently reading postings from the index.
///
/// # Purpose
/// Used when executing queries against the actual index.
///
/// # Storage layout
///
/// Internally backed by **structure-of-arrays** parallel slices
/// (`doc_ids: Vec<u32>`, `frequencies: Vec<u32>`, optional positions
/// sidecar). This avoids the AoS `Vec<Posting>` reassembly that
/// `PostingList::decode` previously paid: 4 bytes per doc-id instead of a
/// 40-byte `Posting` struct, and `next()` advances a single integer cursor
/// over a dense `&[u32]` slice. Per-segment doc-ids fit in `u32` by the same
/// invariant the encoder enforces.
///
/// # Implemented Traits
/// - `reader::PostingIterator` trait
///
/// # Features
/// - `next()`: Move to the next document
/// - `skip_to(target)`: Efficiently skip to a specified document ID
/// - Block-based optimization for fast skip operations
/// - Position information retrieval
/// - Cost calculation for optimization
///
/// # Use Cases
/// - Returned as `Box<dyn reader::PostingIterator>` from `InvertedIndexReader.postings()`
/// - Used during query execution (BooleanQuery, FuzzyQuery, etc.)
/// - When efficient processing of multiple query conditions is needed
///
/// # Difference from `posting::PostingIterator`
/// - `posting::PostingIterator`: Simple in-memory iteration
/// - `InvertedIndexPostingIterator`: Advanced iterator for index queries
#[derive(Debug)]
pub struct InvertedIndexPostingIterator {
    /// Shared, immutable decoded posting list backing this iterator. Holds the
    /// SoA arrays (`doc_ids` / `frequencies` / `positions`) and the multi-level
    /// `skip_levels` table (#503); the iterator only adds a cursor over them.
    ///
    /// The posting cache (#612) stores `Arc<DecodedPostingList>`, so both the
    /// cache-hit and cache-insert paths hand the iterator an `Arc::clone`
    /// (refcount bump) instead of deep-copying the SoA `Vec<u32>` arrays — the
    /// deep clone was ~60% of multi-segment BM25 search wall-time (#576).
    data: Arc<DecodedPostingList>,

    /// Current position in the parallel arrays.
    position: usize,

    /// Whether `next()` has been called at least once.
    started: bool,
}

/// Parallel-array form returned by [`InvertedIndexPostingIterator::soa_from_aos`].
/// Tuple is `(doc_ids, frequencies, optional positions sidecar)`.
type SoaArrays = (Vec<u32>, Vec<u32>, Option<Vec<Option<Vec<u32>>>>);

impl InvertedIndexPostingIterator {
    /// Create a new advanced posting iterator from an AoS [`Vec<Posting>`].
    /// Performs an AoS→SoA conversion eagerly; prefer
    /// [`Self::from_decoded_soa`] in the query hot path to skip this copy.
    pub fn new(postings: Vec<Posting>) -> Self {
        let (doc_ids, frequencies, positions) = Self::soa_from_aos(&postings);
        let skip_levels =
            crate::lexical::index::inverted::core::posting::build_skip_levels(&doc_ids);
        // Wrap the AoS-derived arrays in a `DecodedPostingList` so the iterator
        // shares one representation with the SoA hot path. `weights` / `term` /
        // the frequency aggregates are not read by the iterator, so they are
        // left at their defaults.
        let doc_frequency = doc_ids.len() as u64;
        Self::from_decoded_soa(DecodedPostingList {
            term: String::new(),
            doc_ids,
            frequencies,
            weights: Vec::new(),
            positions,
            skip_levels,
            total_frequency: 0,
            doc_frequency,
        })
    }

    /// Create a posting iterator from AoS postings with multi-level
    /// skip table for O(log_8 N) `skip_to` (#503).
    ///
    /// The `_block_size` argument is kept for source compatibility with
    /// callers that previously tuned the legacy single-level
    /// `block_cache`; the multi-level skip layout is now controlled by
    /// the fixed [`crate::lexical::index::inverted::core::posting::SKIP_INTERVAL`]
    /// constant (Lucene-90 compatible branching factor 8), so this
    /// argument is ignored. Prefer [`Self::from_decoded_soa_with_blocks`]
    /// in the query hot path to skip the AoS→SoA copy.
    pub fn with_blocks(postings: Vec<Posting>, _block_size: usize) -> Self {
        Self::new(postings)
    }

    /// Construct an iterator directly from a SoA-decoded posting list,
    /// without paying an AoS reassembly. This is the fast path used by
    /// [`SegmentReader::postings`] / `term_postings` after
    /// [`PostingList::decode_soa`].
    ///
    /// The iterator inherits the skip table from `decoded`, which is
    /// either decoded straight from a v2 segment or rebuilt at load
    /// time for v1 segments (#503).
    ///
    /// # Arguments
    ///
    /// * `decoded` - SoA-decoded posting data.
    pub fn from_decoded_soa(decoded: DecodedPostingList) -> Self {
        Self::from_decoded_soa_arc(Arc::new(decoded))
    }

    /// Construct an iterator that shares an already-`Arc`-wrapped decoded
    /// posting list. This is the query hot path: [`SegmentReader::postings`]
    /// hands the iterator an `Arc::clone` of the cached list, so no SoA array
    /// is copied (#576).
    ///
    /// # Arguments
    ///
    /// * `data` - Shared SoA-decoded posting data (typically the same `Arc`
    ///   held by the per-segment posting cache).
    pub fn from_decoded_soa_arc(data: Arc<DecodedPostingList>) -> Self {
        InvertedIndexPostingIterator {
            data,
            position: 0,
            started: false,
        }
    }

    /// Like [`Self::from_decoded_soa`]. The `_block_size` argument is
    /// kept for source compatibility but ignored — the skip table is
    /// determined by the global [`crate::lexical::index::inverted::core::posting::SKIP_INTERVAL`]
    /// branching factor (#503).
    ///
    /// # Arguments
    ///
    /// * `decoded` - SoA-decoded posting data.
    /// * `_block_size` - Ignored; retained for source-level compat.
    pub fn from_decoded_soa_with_blocks(decoded: DecodedPostingList, _block_size: usize) -> Self {
        Self::from_decoded_soa(decoded)
    }

    /// Convert AoS postings to parallel SoA arrays; the positions sidecar is
    /// allocated only when at least one posting carries position data.
    fn soa_from_aos(postings: &[Posting]) -> SoaArrays {
        let n = postings.len();
        let mut doc_ids = Vec::with_capacity(n);
        let mut frequencies = Vec::with_capacity(n);
        let any_positions = postings.iter().any(|p| p.positions.is_some());
        let mut positions: Option<Vec<Option<Vec<u32>>>> = if any_positions {
            Some(Vec::with_capacity(n))
        } else {
            None
        };
        for p in postings {
            doc_ids.push(p.doc_id as u32);
            frequencies.push(p.frequency);
            if let Some(out) = positions.as_mut() {
                out.push(p.positions.clone());
            }
        }
        (doc_ids, frequencies, positions)
    }

    /// Walk the multi-level skip table from the top down to find the
    /// smallest `doc_ids` index that is **guaranteed not to exceed** the
    /// position of `target_u32` (#503). The returned index is the
    /// starting point for a final linear scan inside `skip_to`.
    ///
    /// Each level descent bounds its `partition_point` window to at
    /// most [`SKIP_INTERVAL`] entries — the bucket identified at the
    /// parent level. The total work is `O(SKIP_INTERVAL · log_SKIP_INTERVAL N)`
    /// comparisons per call (Lucene 90 / Tantivy compatible) instead
    /// of the `O(N / SKIP_INTERVAL)` scan the legacy single-level
    /// `block_cache` paid.
    ///
    /// The walk respects the current `self.position`: the search never
    /// regresses below where the iterator already sits, so repeated
    /// `skip_to(x); skip_to(y)` calls keep advancing monotonically
    /// without redoing work behind the cursor.
    fn skip_via_levels(&self, target_u32: u32) -> usize {
        use crate::lexical::index::inverted::core::posting::SKIP_INTERVAL;

        let n = self.data.doc_ids.len();
        let cursor = self.position;
        if cursor >= n {
            return n;
        }
        if self.data.skip_levels.is_empty() {
            // Posting list shorter than SKIP_INTERVAL — the tail scan
            // inside `skip_to` handles the whole list.
            return cursor;
        }

        let top = self.data.skip_levels.len() - 1;
        // step at the current level = SKIP_INTERVAL^(level + 1).
        let mut step = SKIP_INTERVAL.saturating_pow((top + 1) as u32);

        // Top level: `build_skip_levels` guarantees ≤ SKIP_INTERVAL
        // entries here, so a single `partition_point` already runs in
        // ≤ log_2(SKIP_INTERVAL) comparisons.
        let top_lvl = &self.data.skip_levels[top];
        let bucket_lo = cursor / step;
        if bucket_lo >= top_lvl.len() {
            // Cursor is past every entry on the top level — descend
            // straight into the linear-scan tail.
            return cursor;
        }
        let slice = &top_lvl[bucket_lo..];
        let local = slice.partition_point(|&x| x < target_u32);
        let mut bucket_index = bucket_lo + local;
        let mut lower = bucket_index * step;

        // Descend: at each lower level, restrict `partition_point` to
        // the SKIP_INTERVAL-wide window corresponding to the parent's
        // bucket. This bounds per-level work to log_2(SKIP_INTERVAL)
        // comparisons (≈ 3 for SKIP_INTERVAL = 8) instead of the
        // unbounded slice the naïve descent would search.
        for level in (0..top).rev() {
            step /= SKIP_INTERVAL;
            let lvl = &self.data.skip_levels[level];

            let parent_lo = bucket_index * SKIP_INTERVAL;
            let parent_hi = (parent_lo + SKIP_INTERVAL).min(lvl.len());
            // Skip entries strictly behind the cursor.
            let lo = (cursor / step).max(parent_lo);
            if lo >= parent_hi {
                // No useful entry left in this bucket; keep `lower`
                // monotone with `cursor` and prepare the next level.
                lower = lower.max(cursor);
                bucket_index = parent_hi.saturating_sub(1);
                continue;
            }
            let slice = &lvl[lo..parent_hi];
            let local = slice.partition_point(|&x| x < target_u32);
            bucket_index = lo + local;
            lower = bucket_index * step;
        }

        // Monotonic progress: never regress below the cursor, and
        // clamp to the posting-list length so the tail scan inside
        // `skip_to` does not run past the array.
        lower.max(cursor).min(n)
    }
}

impl crate::lexical::reader::PostingIterator for InvertedIndexPostingIterator {
    fn doc_id(&self) -> u64 {
        if self.position < self.data.doc_ids.len() {
            self.data.doc_ids[self.position] as u64
        } else {
            u64::MAX // Convention for exhausted iterator
        }
    }

    fn term_freq(&self) -> u64 {
        if self.position < self.data.frequencies.len() {
            self.data.frequencies[self.position] as u64
        } else {
            0
        }
    }

    fn positions(&self) -> Result<Vec<u64>> {
        if self.position >= self.data.doc_ids.len() {
            return Ok(Vec::new());
        }
        match &self.data.positions {
            Some(per_doc) => match &per_doc[self.position] {
                Some(p) => Ok(p.iter().map(|&v| v as u64).collect()),
                None => Ok(Vec::new()),
            },
            None => Ok(Vec::new()),
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.data.doc_ids.is_empty() {
            return Ok(false);
        }

        if !self.started {
            // First call - position at first document
            self.started = true;
            Ok(true)
        } else {
            // Move to next document
            self.position += 1;
            Ok(self.position < self.data.doc_ids.len())
        }
    }

    fn skip_to(&mut self, target_doc_id: u64) -> Result<bool> {
        // Mark as started
        self.started = true;

        let n = self.data.doc_ids.len();
        if n == 0 {
            return Ok(false);
        }

        // Per-segment doc ids are bounded to u32::MAX (matches
        // `PostingList::encode`). A target beyond u32::MAX cannot match
        // any posting in this segment, so we exhaust the iterator
        // straight away.
        let target_u32 = match u32::try_from(target_doc_id) {
            Ok(t) => t,
            Err(_) => {
                self.position = n;
                return Ok(false);
            }
        };

        // Descend the multi-level skip table to land on a small window
        // (≤ SKIP_INTERVAL postings). The final linear scan below
        // bounds the comparisons to that window — total work is
        // O(log_8 N + SKIP_INTERVAL) per call (#503).
        self.position = self.skip_via_levels(target_u32);

        while self.position < n {
            if self.data.doc_ids[self.position] >= target_u32 {
                return Ok(true);
            }
            self.position += 1;
        }
        Ok(false)
    }

    fn cost(&self) -> u64 {
        self.data.doc_ids.len() as u64
    }
}

/// The parts of a segment that its files alone decide (Issue #1423).
///
/// A segment's files never change once written, except its `.delmap`, so
/// everything read from them can serve every reader of the segment, whatever
/// its snapshot. [`SegmentReader`] holds one of these in an `Arc` beside the
/// state its snapshot decides (the deletions and what depends on them), the
/// split Lucene makes between `SegmentCoreReaders` and `SegmentReader`. Each
/// part is loaded on first use.
#[derive(Debug)]
pub(crate) struct SegmentCore {
    /// Which segment this is. Holds only what the files decide: a reader's
    /// own [`SegmentInfo`] also says whether the segment has deletions.
    key: SegmentKey,

    /// Storage backend: the compound facade for a compound segment.
    storage: Arc<dyn Storage>,

    /// The compound-container facade when this segment is a `.cfs`
    /// (#554); `None` for loose segments. Kept typed alongside the
    /// type-erased `storage` clone so part enumeration (`bkd_field_names`)
    /// can consult the table.
    compound: Option<Arc<crate::lexical::index::inverted::compound::CompoundSegmentStorage>>,

    /// Term dictionary for efficient term lookup.
    term_dictionary: RwLock<Option<Arc<BlockTermDictionary>>>,

    /// Cached stored documents, deleted ones included.
    stored_documents: RwLock<Option<BTreeMap<u64, Document>>>,

    /// Cached field-length/statistics source for this segment (#555): its
    /// `.norms` part. See [`Self::load_norms`].
    norms: RwLock<Option<Arc<NormsReader>>>,

    /// DocValues reader for this segment.
    doc_values: RwLock<Option<Arc<DocValuesReader>>>,

    /// Cached BKD trees: field -> tree. Unfiltered; a reader applies its
    /// deletions when it hands one out.
    bkd_trees: RwLock<AHashMap<String, Arc<dyn BKDTree>>>,

    /// Whether the segment is loaded.
    loaded: AtomicBool,

    /// What the segment holds, resolved once (Issue #1211) — see
    /// [`Self::membership`]. It depends only on the segment's files, so the
    /// answer cannot go stale.
    membership: OnceLock<Membership>,
}

/// Reader for a single segment (schema-less mode).
///
/// Describes the segment as of one snapshot: its deletions are read when the
/// reader opens and fixed for its life. What the segment's files decide lives
/// in a shared [`SegmentCore`]. An index hands the same segment reader to
/// every reader it builds until the segment's deletions change, and then a
/// new one over the same core (Issue #1423).
#[derive(Debug)]
pub struct SegmentReader {
    /// Segment information.
    info: SegmentInfo,

    /// The parts the segment's files decide.
    core: Arc<SegmentCore>,

    /// The segment's deleted documents, read by [`Self::open`] and fixed for
    /// the reader's life, so a check takes no lock (Issue #1301); `None` when
    /// the segment has no deletions.
    deleted_docs: Option<Arc<DeletedDocs>>,

    /// Decoded posting-list cache (Issue #612). Disabled by default
    /// (`open` builds it with a zero budget); query readers enable it via
    /// [`Self::with_posting_cache_bytes`]. It holds lists with this reader's
    /// deletions applied, so it belongs to the reader, not to the core, and
    /// stays warm across commits that leave the segment unchanged.
    posting_cache: PostingCache,

    /// Index-time analyzer used by the `.post`-less scan fallback (Issue
    /// #1196). `None` (the state `open` leaves) falls back to
    /// `StandardAnalyzer`; query and merge readers chain
    /// [`Self::with_analyzer`] so the scan analyzes stored values exactly
    /// like the writer did — including per-field analyzers and the
    /// `_id → KeywordAnalyzer` mapping the engine installs.
    analyzer: Option<Arc<dyn Analyzer>>,

    /// Set once the first `postings` call found no `.post` file and logged
    /// the warning (Issue #1196), so a damaged segment warns once per reader
    /// instance instead of once per term.
    warned_missing_postings: AtomicBool,

    /// The live document count, computed once per reader (Issue #1211) —
    /// see [`Self::doc_count`].
    live_doc_count: OnceLock<u64>,
}

impl SegmentCore {
    /// Open the core of the segment `info` describes. Reads nothing but the
    /// compound container's part table; every part loads on first use.
    fn open(info: &SegmentInfo, storage: Arc<dyn Storage>) -> Result<Self> {
        // Layout detection (#554): a `{segment_id}.cfs` container routes
        // every part read through a windowed facade; its absence means a
        // loose (pre-#554) segment and the storage is used as-is. Loose
        // and compound segments therefore coexist freely in one index.
        let compound = crate::lexical::index::inverted::compound::CompoundSegmentStorage::try_open(
            Arc::clone(&storage),
            &info.segment_id,
        )?;
        let storage: Arc<dyn Storage> = match &compound {
            Some(facade) => Arc::clone(facade) as Arc<dyn Storage>,
            None => storage,
        };
        Ok(SegmentCore {
            key: SegmentKey::of(info),
            storage,
            compound,
            term_dictionary: RwLock::new(None),
            stored_documents: RwLock::new(None),
            norms: RwLock::new(None),
            doc_values: RwLock::new(None),
            bkd_trees: RwLock::new(AHashMap::new()),
            loaded: AtomicBool::new(false),
            membership: OnceLock::new(),
        })
    }

    /// See [`SegmentReader::bkd_field_names`].
    fn bkd_field_names(&self) -> Result<Vec<String>> {
        if let Some(facade) = &self.compound {
            return Ok(facade.bkd_field_names());
        }
        let prefix = format!("{}.", self.key.segment_id);
        Ok(self
            .storage
            .list_files()?
            .into_iter()
            .filter_map(|file| {
                file.strip_prefix(&prefix)
                    .and_then(|rest| rest.strip_suffix(".bkd"))
                    .map(str::to_string)
            })
            .collect())
    }

    /// See [`SegmentReader::check_integrity`].
    fn check_integrity(&self, strict: bool) -> Result<()> {
        let segment_id = &self.key.segment_id;
        let mut parts = vec![format!("{segment_id}.post")];
        parts.extend(
            self.bkd_field_names()?
                .into_iter()
                .map(|field| format!("{segment_id}.{field}.bkd")),
        );
        if strict {
            parts.extend(["dict", "docs", "norms", "ids"].map(|s| format!("{segment_id}.{s}")));
        }

        let dict = format!("{segment_id}.dict");
        let mut footed = strict
            || (self.storage.file_exists(&dict)
                && ends_in_footer(self.storage.open_input(&dict)?)?);
        let mut legacy_part = None;
        for part in parts {
            if !self.storage.file_exists(&part) {
                continue;
            }
            match verify_file_checksum(self.storage.open_input(&part)?)? {
                ChecksumStatus::Verified => footed = true,
                ChecksumStatus::Legacy(_) => {
                    legacy_part.get_or_insert(part);
                }
                ChecksumStatus::Mismatch => {
                    return Err(LaurusError::index(format!(
                        "{part}: checksum mismatch — the file is corrupted"
                    )));
                }
            }
        }
        match legacy_part {
            Some(part) if footed => Err(LaurusError::index(format!(
                "{part}: has no checksum footer, though its segment was written with them — the \
                 file is corrupted"
            ))),
            _ => Ok(()),
        }
    }

    /// Load the term dictionary, the stored documents and the doc values.
    fn load(&self) -> Result<()> {
        if self.loaded.load(Ordering::Acquire) {
            return Ok(());
        }
        self.load_term_dictionary()?;
        self.load_stored_documents()?;
        self.load_doc_values()?;
        self.loaded.store(true, Ordering::Release);
        Ok(())
    }

    /// Load the term dictionary for this segment.
    fn load_term_dictionary(&self) -> Result<()> {
        let dict_file = format!("{}.dict", self.key.segment_id);

        if let Ok(input) = self.storage.open_input(&dict_file) {
            let mut reader = StructReader::new(input)?;
            // Bounded by the documents the segment holds, deleted ones
            // included, as the posting decode is (Issue #1235). Not
            // `SegmentReader::doc_count`, which is the live count.
            let dictionary =
                BlockTermDictionary::read_from_storage_for_segment(&mut reader, self.key.doc_count)
                    .map_err(|e| {
                        LaurusError::index(format!(
                            "Failed to read term dictionary from {dict_file}: {e}"
                        ))
                    })?;
            *self.term_dictionary.write().unwrap() = Some(Arc::new(dictionary));
        }

        Ok(())
    }

    /// This segment's term dictionary, loaded on first use (Issues #1420,
    /// #1423). Once loaded, or once the segment has been loaded whole,
    /// `.dict` is not read again.
    ///
    /// # Returns
    ///
    /// `None` when the segment has no `.dict` or it could not be opened.
    ///
    /// # Errors
    ///
    /// Returns an error if `.dict` opens but cannot be parsed.
    fn loaded_term_dictionary(&self) -> Result<Option<Arc<BlockTermDictionary>>> {
        self.ensure_term_dictionary()?;
        Ok(self.term_dictionary.read().unwrap().clone())
    }

    /// Load the term dictionary unless it is loaded already, or the segment
    /// has been loaded whole.
    fn ensure_term_dictionary(&self) -> Result<()> {
        if self.term_dictionary.read().unwrap().is_none() && !self.loaded.load(Ordering::Acquire) {
            self.load_term_dictionary()?;
        }
        Ok(())
    }

    /// Load stored documents for this segment into the
    /// `stored_documents` cache.
    ///
    /// Callers gate on the cache being `None`; after this returns the
    /// cache is always `Some` (an empty map when the segment has no
    /// stored-documents file), so misses stay O(1) instead of re-probing
    /// storage on every lookup.
    fn load_stored_documents(&self) -> Result<()> {
        // Primary: typed, chunked, LZ4-compressed `.docs` (Issue #548), which
        // records the real doc_id per document (correct for non-contiguous
        // ids, e.g. merged segments). See
        // `crate::lexical::index::structures::stored_fields` for the format.
        let docs_file = format!("{}.docs", self.key.segment_id);
        if let Ok(input) = self.storage.open_input(&docs_file) {
            let mut reader = StructReader::new(input)?;
            let documents =
                crate::lexical::index::structures::stored_fields::StoredFieldsReader::load(
                    &mut reader,
                )?;
            *self.stored_documents.write().unwrap() = Some(documents);
            return Ok(());
        }

        // Legacy fallback: segments written before the binary `.docs` format
        // stored fields as a positional JSON mirror (Issue #756 stopped writing
        // it). Doc ids are assigned positionally — valid only for the
        // contiguous ids those legacy segments used.
        let json_file = format!("{}.json", self.key.segment_id);
        if self.storage.file_exists(&json_file) {
            let mut input = self.storage.open_input(&json_file)?;
            let mut json_data = String::new();
            std::io::Read::read_to_string(&mut input, &mut json_data)?;

            let docs: Vec<Document> = serde_json::from_str(&json_data)
                .map_err(|e| LaurusError::index(format!("Failed to parse JSON documents: {e}")))?;

            let mut documents = BTreeMap::new();
            for (idx, doc) in docs.into_iter().enumerate() {
                let doc_id = self.key.min_doc_id + idx as u64;
                documents.insert(doc_id, doc);
            }

            *self.stored_documents.write().unwrap() = Some(documents);
        }

        // No stored-documents file (or the primary `.docs` failed to
        // open): cache an empty map so later lookups don't re-probe
        // storage per call.
        let mut docs = self.stored_documents.write().unwrap();
        if docs.is_none() {
            *docs = Some(BTreeMap::new());
        }

        Ok(())
    }

    /// Load the stored documents unless they are loaded already.
    fn ensure_stored_documents(&self) -> Result<()> {
        if self.stored_documents.read().unwrap().is_none() {
            self.load_stored_documents()?;
        }
        Ok(())
    }

    /// Load DocValues for this segment.
    fn load_doc_values(&self) -> Result<()> {
        // Load DocValues file (required for field sorting)
        let reader = DocValuesReader::load(self.storage.clone(), &self.key.segment_id)?;

        let mut doc_values = self.doc_values.write().unwrap();
        *doc_values = Some(Arc::new(reader));

        Ok(())
    }

    /// See [`SegmentReader::membership`].
    fn membership(&self) -> &Membership {
        self.membership.get_or_init(|| {
            let key = &self.key;
            Membership::from_shape(key.min_doc_id, key.max_doc_id, key.doc_count)
                .unwrap_or_else(|| {
                    Membership::from_loaded(
                        &key.segment_id,
                        key.min_doc_id,
                        key.max_doc_id,
                        key.doc_count,
                        crate::lexical::index::structures::doc_id_set::load_doc_ids_from_segment_storage(
                            self.storage.as_ref(),
                            &key.segment_id,
                        ),
                    )
                })
        })
    }

    /// Load this segment's field-length/statistics source (#555), its
    /// `.norms` part, and return what it stored. A segment with no `.norms`
    /// and no `.lens`/`.fstats` either gets empty norms, as a segment with
    /// any other part missing still opens.
    ///
    /// # Errors
    ///
    /// Returns an error when `.norms` is corrupt, or when it is missing but
    /// `.lens` or `.fstats` exists: the segment predates `.norms` (written
    /// by v0.13.2 or earlier), and its lengths are no longer read (#1121).
    fn load_norms(&self) -> Result<Arc<NormsReader>> {
        let segment_id = &self.key.segment_id;
        if let Some(reader) = NormsReader::load(self.storage.as_ref(), segment_id)? {
            return Ok(self.store_norms(reader));
        }
        if ["lens", "fstats"]
            .iter()
            .any(|suffix| self.storage.file_exists(&format!("{segment_id}.{suffix}")))
        {
            return Err(LaurusError::index(format!(
                "Segment {segment_id} stores its field lengths in .lens/.fstats, the format \
                 before .norms (v0.13.2 or earlier). Pre-release format changes do not support \
                 older revisions; rebuild the index."
            )));
        }
        Ok(self.store_norms(NormsReader::empty()))
    }

    fn store_norms(&self, norms: NormsReader) -> Arc<NormsReader> {
        let norms = Arc::new(norms);
        *self.norms.write().unwrap() = Some(Arc::clone(&norms));
        norms
    }

    /// This segment's field-length source, loaded on first use.
    fn norms(&self) -> Result<Arc<NormsReader>> {
        if let Some(norms) = self.norms.read().unwrap().as_ref() {
            return Ok(Arc::clone(norms));
        }
        self.load_norms()
    }

    /// See [`SegmentReader::stored_doc_count`].
    fn stored_doc_count(&self) -> Result<Option<u64>> {
        let docs_file = format!("{}.docs", self.key.segment_id);
        if !self.storage.file_exists(&docs_file) {
            return Ok(None);
        }
        let input = self.storage.open_input(&docs_file)?;
        let mut reader = StructReader::new(input)?;
        Ok(Some(
            crate::lexical::index::structures::stored_fields::StoredFieldsReader::read_doc_count(
                &mut reader,
            )?,
        ))
    }

    /// See [`SegmentReader::get_bkd_tree`].
    fn get_bkd_tree(&self, field: &str) -> Result<Option<Arc<dyn BKDTree>>> {
        // Check cache
        if let Some(tree) = self.bkd_trees.read().unwrap().get(field) {
            return Ok(Some(tree.clone()));
        }

        // Try to open file
        let bkd_file = format!("{}.{}.bkd", self.key.segment_id, field);
        if self.storage.file_exists(&bkd_file) {
            let reader = BKDReader::open(self.storage.clone(), &bkd_file)?;
            let tree: Arc<dyn BKDTree> = Arc::new(reader);

            // Update cache
            self.bkd_trees
                .write()
                .unwrap()
                .insert(field.to_string(), tree.clone());

            return Ok(Some(tree));
        }

        Ok(None)
    }
}

impl SegmentReader {
    /// Return a reference to the segment metadata.
    ///
    /// This is useful for callers that need to inspect segment boundaries
    /// (e.g., `min_doc_id` / `max_doc_id`) without acquiring interior locks.
    pub fn segment_info(&self) -> &SegmentInfo {
        &self.info
    }

    /// Open a segment reader (schema-less mode).
    ///
    /// Readers that answer queries or feed a merge should chain
    /// [`Self::with_analyzer`] with the index analyzer so the `.post`-less
    /// scan fallback analyzes stored values like the writer did (Issue
    /// #1196); without it the fallback uses `StandardAnalyzer`.
    pub fn open(info: SegmentInfo, storage: Arc<dyn Storage>) -> Result<Self> {
        let core = Arc::new(SegmentCore::open(&info, storage)?);
        // Unlike the other parts, `.delmap` is read here rather than on first
        // use (Issue #1265). The deletion checks behind `doc_count` and
        // `has_effective_deletions` cannot return an error, so a lazy load
        // could only hide an unreadable bitmap as "no deletions"; reading it
        // now fails the open instead. It also fixes the deletions for the
        // reader's life, as a snapshot.
        let deleted_docs = Self::read_deleted_ids(&info, core.storage.as_ref())?
            .map(|ids| Self::deleted_docs(&info, ids));
        Ok(Self::with_deletions(info, core, deleted_docs))
    }

    /// A reader of the segment `info` describes over `core`, the same
    /// segment's, with `deleted_docs` as its deletions.
    fn with_deletions(
        info: SegmentInfo,
        core: Arc<SegmentCore>,
        deleted_docs: Option<Arc<DeletedDocs>>,
    ) -> Self {
        debug_assert_eq!(core.key, SegmentKey::of(&info));
        SegmentReader {
            info,
            core,
            deleted_docs,
            // Disabled by default; query readers enable it (Issue #612).
            posting_cache: PostingCache::new(0),
            analyzer: None,
            warned_missing_postings: AtomicBool::new(false),
            live_doc_count: OnceLock::new(),
        }
    }

    /// The parts of the segment its files decide (test hook, Issue #1423).
    #[cfg(test)]
    pub(crate) fn core(&self) -> &Arc<SegmentCore> {
        &self.core
    }

    /// The numeric/geo field names this segment carries BKD trees for
    /// (#554).
    ///
    /// Compound segments answer from the container table; loose segments
    /// scan storage for `{segment_id}.{field}.bkd` files — the enumeration
    /// the merge engine used to do against raw storage, which would find
    /// nothing once the parts moved into a container and silently drop
    /// every point at the first merge.
    ///
    /// # Returns
    ///
    /// The field names, in no particular order.
    ///
    /// # Errors
    ///
    /// Returns an error if listing a loose segment's storage fails.
    pub fn bkd_field_names(&self) -> Result<Vec<String>> {
        self.core.bkd_field_names()
    }

    /// Verify this segment's parts against their checksum footers, one
    /// sequential pass per part (Issue #1214).
    ///
    /// `.post` and every `.bkd` are read by random access, so the checks
    /// made while loading the other parts never reach them. A merge calls
    /// this on each source before copying it, so that corruption is not
    /// carried into the merged segment under a fresh, valid checksum.
    ///
    /// `strict` is for a segment this binary just wrote: it also covers
    /// `.dict`, `.docs`, `.norms` and `.ids`, and refuses any part without a
    /// footer. Otherwise a part with a legacy trailer passes only while the
    /// rest of its segment has legacy trailers too — one binary writes all of
    /// a segment's parts, so a footed part beside an unfooted one means that
    /// part's footer was damaged.
    ///
    /// # Arguments
    ///
    /// * `strict` - Whether the segment was written with footers throughout.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::Index`] naming the first part that fails, and
    /// any error from reading the parts.
    pub fn check_integrity(&self, strict: bool) -> Result<()> {
        self.core.check_integrity(strict)
    }

    /// Enable (or resize) this segment's decoded posting-list cache with a byte
    /// budget (Issue #612). `0` keeps it disabled. Returns `self` for chaining
    /// after [`Self::open`].
    ///
    /// # Arguments
    ///
    /// * `max_bytes` - Soft heap budget for cached posting lists in this segment.
    pub fn with_posting_cache_bytes(mut self, max_bytes: usize) -> Self {
        self.posting_cache = PostingCache::new(max_bytes);
        self
    }

    /// Set the index analyzer the `.post`-less scan fallback analyzes stored
    /// values with (Issue #1196). Pass the same `Arc<dyn Analyzer>` the
    /// writer indexed with — a `PerFieldAnalyzer` is resolved per field, so
    /// custom field analyzers and `_id → KeywordAnalyzer` match the
    /// postings. Returns `self` for chaining after [`Self::open`].
    pub fn with_analyzer(mut self, analyzer: Arc<dyn Analyzer>) -> Self {
        self.analyzer = Some(analyzer);
        self
    }

    /// Whether the `.post`-missing warning has been logged by this reader
    /// instance (test hook for the once-per-segment guard, Issue #1196).
    #[cfg(test)]
    pub(crate) fn has_warned_missing_postings(&self) -> bool {
        self.warned_missing_postings.load(Ordering::Relaxed)
    }

    /// Replace the segment's term dictionary (test hook for dictionaries the
    /// segment load would refuse, Issue #1235). It is the core's, so every
    /// reader sharing the core sees it, and nothing reads `.dict` over it.
    #[cfg(test)]
    pub(crate) fn set_term_dictionary(&self, dictionary: Arc<BlockTermDictionary>) {
        *self.core.term_dictionary.write().unwrap() = Some(dictionary);
    }

    /// Snapshot of this segment's posting-cache hit / miss counters (Issue #612).
    pub fn posting_cache_stats(
        &self,
    ) -> crate::lexical::index::inverted::posting_cache::PostingCacheStats {
        self.posting_cache.stats()
    }

    /// Get all document IDs in this segment, deleted ones included.
    pub fn doc_ids(&self) -> Result<Vec<u64>> {
        self.core.ensure_stored_documents()?;
        let docs = self.core.stored_documents.read().unwrap();
        if let Some(ref documents) = *docs {
            Ok(documents.keys().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Deprecated: Use `open()` instead. Schema is no longer required.
    #[deprecated(
        since = "0.2.0",
        note = "Use `open()` instead. Schema is no longer required."
    )]
    pub fn open_with_schema(
        info: SegmentInfo,
        _schema: Arc<()>,
        storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        Self::open(info, storage)
    }

    /// Load the segment data: the term dictionary, the stored documents and
    /// the doc values. The deletion bitmap is already loaded by `open`.
    pub fn load(&mut self) -> Result<()> {
        self.core.load()
    }

    /// A reader of this segment as `info`, a later snapshot's description of
    /// it, has it (Issue #1423).
    ///
    /// The segment's files have not changed, so the new reader shares this
    /// one's [`SegmentCore`] and reads only the `.delmap`, which the writer
    /// rewrites in place as the segment gains deletions. The deletions are
    /// compared by content, because `info` can be unchanged while the bitmap
    /// is not: `deleted_count` counts only the deletions of documents the
    /// segment holds.
    ///
    /// # Arguments
    ///
    /// * `info` - The segment as the later snapshot describes it.
    ///
    /// # Returns
    ///
    /// `None` when this reader already describes the segment as `info` does,
    /// with the same metadata and the same deletions. Otherwise a reader of
    /// `info` with an empty posting cache and no analyzer, as
    /// [`Self::open`] returns one.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::Index`] naming the segment if the `.delmap`
    /// exists but cannot be read or is corrupted.
    pub(crate) fn reopen(&self, info: &SegmentInfo) -> Result<Option<SegmentReader>> {
        debug_assert_eq!(self.core.key, SegmentKey::of(info));
        let ids = Self::read_deleted_ids(info, self.core.storage.as_ref())?;
        let deleted_docs = if ids.as_ref() == self.deleted_docs.as_ref().map(|d| d.ids()) {
            if *info == self.info {
                return Ok(None);
            }
            self.deleted_docs.clone()
        } else {
            ids.map(|ids| Self::deleted_docs(info, ids))
        };
        Ok(Some(Self::with_deletions(
            info.clone(),
            Arc::clone(&self.core),
            deleted_docs,
        )))
    }

    /// Read the ids the segment's deletion bitmap marks deleted.
    ///
    /// # Returns
    ///
    /// `None` when the segment has no deletions, or when its metadata
    /// records deletions but no `.delmap` file exists — a missing bitmap is
    /// not a corrupted one.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::Index`] naming the segment if the `.delmap`
    /// exists but cannot be read or is corrupted, as the merge's
    /// `load_deleted_docs` does.
    fn read_deleted_ids(
        info: &SegmentInfo,
        storage: &dyn Storage,
    ) -> Result<Option<RoaringTreemap>> {
        if !info.has_deletions {
            return Ok(None);
        }
        let bitmap_file = format!("{}.delmap", info.segment_id);
        if !storage.file_exists(&bitmap_file) {
            return Ok(None);
        }
        let mut reader = StructReader::new(storage.open_input(&bitmap_file)?)?;
        let bitmap = DeletionBitmap::read_from_storage(&mut reader).map_err(|e| {
            LaurusError::index(format!(
                "segment {}: failed to load its deletion bitmap: {e}",
                info.segment_id
            ))
        })?;
        Ok(Some(bitmap.into_deleted_docs()))
    }

    /// `ids` as the deleted documents of the segment `info` describes.
    fn deleted_docs(info: &SegmentInfo, ids: RoaringTreemap) -> Arc<DeletedDocs> {
        Arc::new(DeletedDocs::new(
            ids,
            info.min_doc_id,
            info.max_doc_id,
            info.doc_count,
        ))
    }

    /// The segment's deleted documents; `None` when it has no deletions.
    fn deletions(&self) -> Option<Arc<DeletedDocs>> {
        self.deleted_docs.clone()
    }

    /// What this segment holds (Issue #1211), resolved once for the segment.
    ///
    /// Most segments answer without I/O: an empty one holds nothing and one
    /// without gaps holds its whole range. Only a segment with gaps reads its
    /// doc-id set — `.ids`, or `.norms` for a segment written before that
    /// part existed — through the segment's own storage (the compound facade
    /// windows the part without copying the container). A set that
    /// disagrees with the segment's recorded count and range is not trusted.
    pub(crate) fn membership(&self) -> &Membership {
        self.core.membership()
    }

    /// Whether any document this segment holds is deleted (Issue #1211).
    ///
    /// Exact where the live count may only be a bound: a segment whose
    /// membership is unknown reports `true` as soon as its range carries a
    /// bit, so a caller relying on "no deletions" (the count fast path)
    /// never takes a bound for the truth.
    pub(crate) fn has_effective_deletions(&self) -> bool {
        let Some(deleted) = self.deletions() else {
            return false;
        };
        let deleted = deleted.ids();
        let info = &self.info;
        match self.membership() {
            // The held count is only a lower bound here: any bit in range
            // might be a real deletion.
            Membership::Unknown => deleted.range_cardinality(info.min_doc_id..=info.max_doc_id) > 0,
            membership => {
                membership.held_deletions(deleted, info.min_doc_id, info.max_doc_id, info.doc_count)
                    > 0
            }
        }
    }

    /// The ids of the live documents this segment holds, ascending (Issue
    /// #1211): the ids it holds minus its own deletions. A segment of
    /// unknown membership lists its stored documents instead.
    ///
    /// # Errors
    ///
    /// Returns an error if a segment of unknown membership cannot read its
    /// stored documents.
    pub(crate) fn live_doc_ids(&self) -> Result<RoaringTreemap> {
        let info = &self.info;
        let held: RoaringTreemap = match self.membership() {
            Membership::Empty => return Ok(RoaringTreemap::new()),
            Membership::Range => {
                let mut range = RoaringTreemap::new();
                range.insert_range(info.min_doc_id..=info.max_doc_id);
                range
            }
            Membership::Set(ids) => (**ids).clone(),
            Membership::Unknown => self.doc_ids()?.into_iter().collect(),
        };
        Ok(match self.deletions() {
            Some(deleted) => &held - deleted.ids(),
            None => held,
        })
    }

    /// Check whether a global doc_id is marked as deleted in this segment.
    pub fn is_deleted(&self, doc_id: u64) -> Result<bool> {
        // Hot on the scoring path, which probes deletion status per scored
        // doc (often redundantly, since the posting iterator is already
        // deletion-filtered at decode time via `filter_deleted_soa`). The
        // deletions were read by `open`, so this is a lookup with no I/O and
        // no lock.
        Ok(self
            .deleted_docs
            .as_ref()
            .is_some_and(|deleted| deleted.contains(doc_id)))
    }

    /// Drop deleted entries from a SoA-decoded posting list in lockstep
    /// across the parallel arrays (`doc_ids`, `frequencies`, optional
    /// positions). Returns the same list unchanged when the segment has no
    /// deletions, avoiding any allocation in the common case.
    fn filter_deleted_soa(&self, decoded: DecodedPostingList) -> Result<DecodedPostingList> {
        // Fast path: nothing to filter.
        let Some(deleted) = self.deleted_docs.as_ref() else {
            return Ok(decoded);
        };

        let n = decoded.doc_ids.len();
        let mut doc_ids = Vec::with_capacity(n);
        let mut frequencies = Vec::with_capacity(n);
        let mut positions: Option<Vec<Option<Vec<u32>>>> =
            decoded.positions.as_ref().map(|_| Vec::with_capacity(n));

        for i in 0..n {
            let did = decoded.doc_ids[i] as u64;
            if deleted.contains(did) {
                continue;
            }
            doc_ids.push(decoded.doc_ids[i]);
            frequencies.push(decoded.frequencies[i]);
            if let (Some(out), Some(src)) = (positions.as_mut(), decoded.positions.as_ref()) {
                out.push(src[i].clone());
            }
        }

        // After deletion filtering the doc_ids may shrink, so rebuild
        // the skip table over the surviving entries. This is the same
        // path #503's load-time fallback exercises.
        let skip_levels =
            crate::lexical::index::inverted::core::posting::build_skip_levels(&doc_ids);

        Ok(DecodedPostingList {
            term: decoded.term,
            doc_ids,
            frequencies,
            weights: Vec::new(), // weights are not consumed by the iterator API
            positions,
            skip_levels,
            total_frequency: decoded.total_frequency,
            doc_frequency: decoded.doc_frequency,
        })
    }

    /// Get a DocValues field value for a document.
    pub(crate) fn get_doc_value(&self, field: &str, doc_id: u64) -> Result<Option<FieldValue>> {
        // Mirror `document()`: a soft-deleted doc (e.g. the pre-upsert
        // copy in an older segment) must not surface its stale value
        // through the cross-segment first-hit resolution (#943).
        if self.is_deleted(doc_id)? {
            return Ok(None);
        }

        // Load once, on demand; the cache itself is the load gate (the
        // `loaded` flag is only set by the optional bulk `load()` path,
        // so gating on it re-parsed the whole `.dv` file per call —
        // #943, same class as #995). A missing `.dv` file loads as an
        // empty reader, so misses stay O(1).
        if self.core.doc_values.read().unwrap().is_none() {
            self.core.load_doc_values()?;
        }

        let doc_values = self.core.doc_values.read().unwrap();
        if let Some(reader) = doc_values.as_ref() {
            reader.get_value(field, doc_id)
        } else {
            Ok(None)
        }
    }

    /// Check if DocValues are available for a field.
    pub(crate) fn has_doc_values(&self, field: &str) -> bool {
        // Load on demand so availability is answered correctly even as
        // the first operation on a fresh reader (#943); previously this
        // reported `false` until something else loaded the cache.
        if self.core.doc_values.read().unwrap().is_none() && self.core.load_doc_values().is_err() {
            return false;
        }
        let doc_values = self.core.doc_values.read().unwrap();
        if let Some(reader) = doc_values.as_ref() {
            reader.has_field(field)
        } else {
            false
        }
    }

    /// This segment's field-length source, loaded on first use.
    fn norms(&self) -> Result<Arc<NormsReader>> {
        self.core.norms()
    }

    /// Get field statistics for a specific field.
    pub fn field_stats(&self, field: &str) -> Result<Option<FieldStats>> {
        if self.core.norms.read().unwrap().is_none() {
            self.core.load_norms()?;
        }
        let norms = self.core.norms.read().unwrap();
        Ok(norms.as_ref().and_then(|n| n.field_stats(field)))
    }

    /// Get field length for a specific document and field.
    ///
    /// Uses a fast path with a single `RwLock` acquisition when the
    /// field-length source is already loaded (hot path). Falls back to
    /// loading on the first call (cold path), which requires a second
    /// acquisition.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The document ID to look up.
    /// * `field` - The field name whose length is requested.
    ///
    /// # Returns
    ///
    /// `Ok(Some(length))` if the document exists and has the field,
    /// `Ok(None)` if the document is deleted or the field is absent.
    pub fn field_length(&self, doc_id: u64, field: &str) -> Result<Option<u32>> {
        if self.is_deleted(doc_id)? {
            return Ok(None);
        }

        // Fast path: try to read with a single lock acquisition.
        let norms = self.core.norms.read().unwrap();
        if let Some(n) = norms.as_ref() {
            return Ok(n.field_length(doc_id, field));
        }
        drop(norms);

        // Cold path: load (one-time), then retry.
        self.core.load_norms()?;
        let norms = self.core.norms.read().unwrap();
        Ok(norms.as_ref().and_then(|n| n.field_length(doc_id, field)))
    }

    /// All field names this segment recorded a length for (#1122).
    ///
    /// Mirrors [`Self::bkd_field_names`]'s role for BKD trees: the merge
    /// engine needs this to reconstruct a length for a field that analyzed
    /// to zero tokens for a given document, which leaves no term postings
    /// and so would otherwise be invisible to a `field_terms`-keyed
    /// enumeration.
    pub(crate) fn norms_field_names(&self) -> Result<Vec<String>> {
        if self.core.norms.read().unwrap().is_none() {
            self.core.load_norms()?;
        }
        let norms = self.core.norms.read().unwrap();
        Ok(norms.as_ref().map(|n| n.field_names()).unwrap_or_default())
    }

    /// Get a document by ID from this segment.
    pub fn document(&self, doc_id: u64) -> Result<Option<Document>> {
        // Load once, on demand; the cache itself is the load gate (the
        // `loaded` flag is only set by the optional bulk `load()` path,
        // so gating on it re-decoded the whole segment per call — #994).
        if self.core.stored_documents.read().unwrap().is_none() {
            self.core.load_stored_documents()?;
        }

        if self.is_deleted(doc_id)? {
            return Ok(None);
        }

        let docs = self.core.stored_documents.read().unwrap();
        if let Some(ref documents) = *docs {
            Ok(documents.get(&doc_id).cloned())
        } else {
            Ok(None)
        }
    }

    /// Fetch a subset of a document's stored fields without cloning
    /// the rest of the [`Document`] map (#410).
    ///
    /// Wide-schema search requests typically retrieve only a handful
    /// of stored fields; the default
    /// [`document()`](Self::document) path clones every field's
    /// `DataValue` — including byte arrays / vector payloads — before
    /// the caller filters them. This method clones only the requested
    /// fields out of the cached in-memory map.
    pub fn document_fields(
        &self,
        doc_id: u64,
        field_names: &[&str],
    ) -> Result<Option<std::collections::HashMap<String, crate::data::DataValue>>> {
        if self.core.stored_documents.read().unwrap().is_none() {
            self.core.load_stored_documents()?;
        }

        if self.is_deleted(doc_id)? {
            return Ok(None);
        }

        let docs = self.core.stored_documents.read().unwrap();
        if let Some(ref documents) = *docs
            && let Some(doc) = documents.get(&doc_id)
        {
            let mut out = std::collections::HashMap::with_capacity(field_names.len());
            for &name in field_names {
                if let Some(value) = doc.fields.get(name) {
                    out.insert(name.to_string(), value.clone());
                }
            }
            return Ok(Some(out));
        }
        Ok(None)
    }

    /// This segment's term dictionary, loaded on first use (Issues #1420,
    /// #1423): once loaded, or once the segment has been loaded whole,
    /// `.dict` is not read again.
    ///
    /// # Returns
    ///
    /// `None` when the segment has no `.dict` or it could not be opened.
    ///
    /// # Errors
    ///
    /// Returns an error if `.dict` opens but cannot be parsed.
    fn loaded_term_dictionary(&self) -> Result<Option<Arc<BlockTermDictionary>>> {
        self.core.loaded_term_dictionary()
    }

    /// Get term information for a field and term.
    pub fn term_info(&self, field: &str, term: &str) -> Result<Option<TermInfo>> {
        self.core.ensure_term_dictionary()?;

        if let Some(ref dict) = *self.core.term_dictionary.read().unwrap() {
            let full_term = format!("{field}:{term}");
            Ok(dict.get(&full_term).cloned())
        } else {
            Ok(None)
        }
    }

    /// Return this segment's term dictionary, loading it on demand.
    ///
    /// Exposes the dictionary so the segment merge (Issue #753) can enumerate
    /// every `"field:term"` key (via [`BlockTermDictionary::iter`]) and re-read
    /// each term's postings through [`Self::postings`] without re-tokenizing.
    /// Returns `None` when the segment has no on-disk term dictionary. A
    /// dictionary already loaded is returned as is (Issue #1423).
    ///
    /// # Errors
    ///
    /// Returns an error if `.dict` opens but cannot be parsed.
    pub fn term_dictionary(&self) -> Result<Option<Arc<BlockTermDictionary>>> {
        self.loaded_term_dictionary()
    }

    /// The number of documents this segment's stored-fields part declares,
    /// read from the `.docs` header only (Issue #1166). `None` when the
    /// segment has no `.docs` part — which is what a merge writes when every
    /// source document was deleted, or a pre-`.docs` legacy segment.
    ///
    /// Unlike [`Self::doc_count`], which reports the segment metadata's own
    /// `doc_count` whenever the segment has no deletions, this reflects what
    /// was actually written.
    ///
    /// # Errors
    ///
    /// Returns an error if the part exists but cannot be opened or its
    /// header is not a supported stored-fields header.
    pub fn stored_doc_count(&self) -> Result<Option<u64>> {
        self.core.stored_doc_count()
    }

    /// Whether this segment has an on-disk term dictionary (Issue #1196).
    ///
    /// A segment without one still answers `postings` through the
    /// stored-document scan, but contributes nothing to `term_info`, so a
    /// reader containing such a segment cannot treat `term_info` as the
    /// authority on which terms exist — see
    /// [`InvertedIndexReader::term_info_is_authoritative`]. A dictionary that
    /// fails to load counts as absent. Asking does not read `.dict` again
    /// once it is loaded (Issue #1423).
    pub fn has_term_dictionary(&self) -> bool {
        self.loaded_term_dictionary()
            .map(|dict| dict.is_some())
            .unwrap_or(false)
    }

    /// Get the posting list for a field and term, with this reader's
    /// deletions applied.
    pub fn postings(&self, field: &str, term: &str) -> Result<Option<Box<dyn PostingIterator>>> {
        // Load postings from storage
        let postings_file = format!("{}.post", self.info.segment_id);

        if !self.core.storage.file_exists(&postings_file) {
            // No postings part: answer from the stored documents. Warn once
            // per reader instance, not per term. The index keeps a segment's
            // reader across commits until the segment's deletions change
            // (Issue #1423), and every index opened builds its own.
            if !self.warned_missing_postings.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "segment {} has no .post file; term queries against it are answered by \
                     scanning its stored documents (Issue #1196)",
                    self.info.segment_id
                );
            }
            return self.scan_documents_for_term(field, term);
        }

        // Posting cache (#612): a repeated `(field, term)` lookup on this
        // segment reader reuses the decoded, deletion-filtered list instead of
        // re-opening + re-decoding the `.post` file (the read dominates on
        // remote storage). The key allocation, lookup, and the clone are
        // skipped entirely when the cache is disabled (budget 0), so the
        // uncached path — merge / test readers — is byte-for-byte unchanged.
        let cache_key = self
            .posting_cache
            .is_enabled()
            .then(|| format!("{field}\u{1}{term}"));
        if let Some(key) = &cache_key
            && let Some(cached) = self.posting_cache.get(key)
        {
            // Share the cached `Arc<DecodedPostingList>` with the iterator
            // instead of deep-copying the SoA arrays (#576).
            return Ok(Some(Box::new(
                InvertedIndexPostingIterator::from_decoded_soa_arc(cached),
            )));
        }

        if let Some(term_info) = self.term_info(field, term)? {
            let input = self.core.storage.open_input(&postings_file)?;
            let mut reader = StructReader::new(input)?;

            // Seek directly to the posting position
            if term_info.posting_offset > 0 {
                reader.seek(std::io::SeekFrom::Start(term_info.posting_offset))?;
            }

            // Decode the posting list in SoA-native form to skip the
            // intermediate `Vec<Posting>` reassembly and keep the iterator
            // backed by parallel `Vec<u32>` slices. Dispatched by on-disk
            // posting format version: v2 segments carry the multi-level
            // skip table inline (#503) while v1 segments rebuild it from
            // `doc_ids` at load time; v3 additionally gates the weights
            // section on a header byte (#553).
            //
            // Matched exactly rather than with an ordered comparison. A
            // `>=` would route a newer payload into an older decoder,
            // which reads the added header byte as the next field and
            // corrupts the list silently instead of failing.
            let posting_format = self
                .core
                .term_dictionary
                .read()
                .unwrap()
                .as_ref()
                .map(|dict| dict.posting_format_version())
                .unwrap_or(3);
            // Capped by the documents the segment holds, deleted ones
            // included — they stay in the postings until
            // `filter_deleted_soa` below — so a corrupt count cannot size
            // the decode buffers (Issue #1220). Not `self.doc_count()`,
            // which is the live count.
            let decoded = PostingList::decode_soa_for_segment(
                &mut reader,
                posting_format,
                self.info.doc_count,
            )
            // A list that decodes must still be the one the dictionary
            // entry describes, inside this segment (Issue #1235).
            .and_then(|decoded| {
                self.check_posting_list(&decoded, field, term, &term_info, reader.position())?;
                Ok(decoded)
            })
            // Name the segment and term a corruption was found in; other
            // errors (I/O) keep their variant.
            .map_err(|e| match e {
                LaurusError::Index(msg) => LaurusError::Index(format!(
                    "segment {}: {field}:{term}: {msg}",
                    self.info.segment_id
                )),
                other => other,
            })?;
            let filtered = self.filter_deleted_soa(decoded)?;

            if filtered.is_empty() {
                // Empty lists are not cached — `None` is cheap to recompute.
                Ok(None)
            } else if let Some(key) = cache_key {
                // Cache the shared decoded list and back the iterator with the
                // same `Arc` — both the cache and the iterator point at one
                // copy of the SoA arrays, so building the iterator is an
                // `Arc::clone` (refcount bump) rather than a `Vec` deep copy
                // (#576).
                let shared = Arc::new(filtered);
                self.posting_cache.put(key, Arc::clone(&shared));
                Ok(Some(Box::new(
                    InvertedIndexPostingIterator::from_decoded_soa_arc(shared),
                )))
            } else {
                // Cache disabled — build directly from the owned list (a single
                // `Arc::new`, no array copy).
                Ok(Some(Box::new(
                    InvertedIndexPostingIterator::from_decoded_soa(filtered),
                )))
            }
        } else {
            Ok(None)
        }
    }

    /// Check a posting list decoded for `field:term` against the dictionary
    /// entry that located it and against this segment (Issue #1235).
    ///
    /// The decoder can only check what a list says about itself. A list
    /// that decodes cleanly can still be another term's, shorter or longer
    /// than the one written, or hold ids outside the segment, and would then
    /// be served as this term's postings. Every check holds for a list the
    /// writer produced: it records the term's own list, its byte span and
    /// its posting count, derives the block-max entries from the same
    /// postings, and the segment's id range from the documents it writes.
    ///
    /// `end` is the reader's position after the decode. Positions are
    /// relative to the `.post` part in both layouts: a compound container
    /// serves the part through a window that seeks and reports positions
    /// relative to the part's start, as the writer's were.
    fn check_posting_list(
        &self,
        decoded: &DecodedPostingList,
        field: &str,
        term: &str,
        term_info: &TermInfo,
        end: u64,
    ) -> Result<()> {
        // A dictionary offset that points at another term's list decodes
        // cleanly, so the list's own key is compared.
        if decoded
            .term
            .strip_prefix(field)
            .and_then(|rest| rest.strip_prefix(':'))
            != Some(term)
        {
            return Err(LaurusError::index(format!(
                "posting list: the dictionary entry leads to the list of {:?} — segment is \
                 corrupted",
                decoded.term
            )));
        }
        let n = decoded.doc_ids.len();
        if n as u64 != term_info.doc_frequency {
            return Err(LaurusError::index(format!(
                "posting list: holds {n} postings but the dictionary declares {} documents — \
                 segment is corrupted",
                term_info.doc_frequency
            )));
        }
        // A corrupt byte can change how much the decoder reads — a varint
        // continuation bit, a block's bit width — and leave a list that
        // still parses, with its values shifted. Only its end tells.
        if end.checked_sub(term_info.posting_offset) != Some(term_info.posting_length) {
            return Err(LaurusError::index(format!(
                "posting list: the decode ended at byte {end}, but the dictionary places the list \
                 at byte {} with {} bytes — segment is corrupted",
                term_info.posting_offset, term_info.posting_length
            )));
        }
        // The doc ids are strictly ascending (the decoder checks it), so the
        // ends bound the rest. An id outside the range is not this segment's
        // document.
        if let (Some(&first), Some(&last)) = (decoded.doc_ids.first(), decoded.doc_ids.last()) {
            if u64::from(first) < self.info.min_doc_id {
                return Err(LaurusError::index(format!(
                    "posting list: doc id {first} is below the segment's min_doc_id {} — \
                     segment is corrupted",
                    self.info.min_doc_id
                )));
            }
            if u64::from(last) > self.info.max_doc_id {
                return Err(LaurusError::index(format!(
                    "posting list: doc id {last} is above the segment's max_doc_id {} — \
                     segment is corrupted",
                    self.info.max_doc_id
                )));
            }
        }
        // The scorer finds a document's block as the first entry whose
        // `last_doc_id` is not below it, and treats a document past the last
        // entry as unable to score (`current_block_max_score`,
        // `next_block_boundary`, `block_max_score_at` in the BM25 scorer).
        // A last entry that ends too early, or an array cut short, would make
        // it skip the term's remaining documents; a boundary off the list's
        // own would apply one block's bound to another's documents. The
        // entries' ascending order and factors are checked when the
        // dictionary loads. Empty stays allowed: the scorer then uses the
        // term-level bound, and it is what the writer stores for an empty
        // list.
        let block_max = &term_info.block_max;
        if !block_max.is_empty() {
            let blocks = n.div_ceil(BLOCK_SIZE);
            if block_max.len() != blocks {
                return Err(LaurusError::index(format!(
                    "posting list: the dictionary has {} block-max entries for {n} postings, \
                     not {blocks} — segment is corrupted",
                    block_max.len()
                )));
            }
            for (i, entry) in block_max.iter().enumerate() {
                // `i < blocks` and `n > 0` here, so the index is in range.
                let block_last = decoded.doc_ids[((i + 1) * BLOCK_SIZE).min(n) - 1];
                if entry.last_doc_id != u64::from(block_last) {
                    return Err(LaurusError::index(format!(
                        "posting list: block-max entry {i} ends at doc id {}, but block {i} of \
                         the list ends at {block_last} — segment is corrupted",
                        entry.last_doc_id
                    )));
                }
            }
        }
        Ok(())
    }

    /// Answer a term query from the stored documents when this segment has
    /// no `.post` file (the only condition under which [`Self::postings`]
    /// calls this). The writer always emits the postings part and compound
    /// segments are the default, so the path is reached only for a loose
    /// segment whose `.post` was lost or a pre-compound stored-only segment
    /// — and, through the merge engine's replay, for carrying such a
    /// segment's terms into a merged segment.
    ///
    /// Every stored value is re-analyzed with the writer's own
    /// `analyze_field_value` (Issue #1194), so each `DataValue` variant —
    /// `Text`, a multi-valued `TextArray` with its position-increment gap,
    /// `Bool` / `BoolArray` as `"true"` / `"false"`, numeric and datetime
    /// terms — yields exactly the terms and positions its postings would
    /// have carried. The analyzer is the one chained through
    /// [`Self::with_analyzer`] — the index analyzer, so a `PerFieldAnalyzer`
    /// resolves per field and `_id` is analyzed as a keyword (Issue #1196) —
    /// or `StandardAnalyzer` for a reader opened without one.
    ///
    /// Remaining limitations, because the segment reader is schema-less:
    /// the position gap is `position_increment_gap_for(None)` rather than
    /// the field's own, `indexed: false` cannot be honoured, the result is
    /// not cached (a phrase query rescans once per term), scores are zero
    /// without a `.dict`, and prefix / wildcard / fuzzy / regexp queries —
    /// which expand through dictionary enumeration — never see the scanned
    /// terms.
    fn scan_documents_for_term(
        &self,
        field: &str,
        term: &str,
    ) -> Result<Option<Box<dyn PostingIterator>>> {
        // Ensure documents are loaded
        if !self.core.loaded.load(Ordering::Acquire) {
            // Load documents on-demand
            self.core.load_stored_documents()?;
        }

        let docs = self.core.stored_documents.read().unwrap();

        if let Some(ref documents) = *docs {
            let mut postings = Vec::new();
            let analyzer: Arc<dyn Analyzer> = match &self.analyzer {
                Some(analyzer) => Arc::clone(analyzer),
                None => Arc::new(StandardAnalyzer::new()?),
            };
            let position_increment_gap = super::writer::position_increment_gap_for(None);

            for (doc_id, doc) in documents.iter() {
                if self.is_deleted(*doc_id)? {
                    continue;
                }
                let Some(field_value) = doc.get_field(field) else {
                    continue;
                };

                // Same term derivation as indexing; the BKD points half of
                // the tuple has no meaning for a term lookup.
                let (terms, _points) = super::writer::analyze_field_value(
                    field,
                    field_value,
                    &analyzer,
                    position_increment_gap,
                )?;
                // `AnalyzedTerm::frequency` is a running count per token, so
                // the term frequency is the number of matching entries.
                let positions: Vec<u32> = terms
                    .iter()
                    .filter(|analyzed| analyzed.term == term)
                    .map(|analyzed| analyzed.position)
                    .collect();

                if !positions.is_empty() {
                    postings.push(Posting {
                        doc_id: *doc_id,
                        frequency: positions.len() as u32,
                        positions: Some(positions),
                        weight: 1.0,
                    });
                }
            }

            if postings.is_empty() {
                Ok(None)
            } else {
                Ok(Some(Box::new(InvertedIndexPostingIterator::with_blocks(
                    postings, 64,
                ))))
            }
        } else {
            Ok(None)
        }
    }

    /// Get the number of live documents in this segment.
    ///
    /// The documents the segment holds minus the deleted ones (Issue #1211).
    /// It used to be the bitmap's `live_count`, the id-range width minus the
    /// deletions, which over-counts a segment with gaps in its range (a merge
    /// drops deleted documents) — skewing BM25's N, and satisfying the count
    /// fast path's "no deletions" test while deletions existed.
    ///
    /// When the segment's membership is unknown, the value is an upper bound
    /// that never under-counts: at most `width − doc_count` bits can sit on
    /// ids the segment does not hold, so at least `bits − gaps` deletions are
    /// real. Computed once per reader.
    pub fn doc_count(&self) -> u64 {
        *self.live_doc_count.get_or_init(|| {
            let info = &self.info;
            let Some(deleted_docs) = self.deletions() else {
                return info.doc_count;
            };
            let deleted = self.membership().held_deletions(
                deleted_docs.ids(),
                info.min_doc_id,
                info.max_doc_id,
                info.doc_count,
            );
            info.doc_count.saturating_sub(deleted)
        })
    }

    /// Get BKD Tree for a field, loading it if necessary.
    pub fn get_bkd_tree(&self, field: &str) -> Result<Option<Arc<dyn BKDTree>>> {
        self.core.get_bkd_tree(field)
    }

    /// Return this segment's BKD tree wrapped in a deletion filter that
    /// drops hits whose doc-id is recorded in the segment's deletion
    /// bitmap.
    ///
    /// Used by the per-segment fanout path
    /// ([`super::per_segment_view::PerSegmentReaderView`]) where the
    /// caller only sees one segment at a time, so the cross-segment
    /// snapshot built by [`InvertedIndexReader::get_bkd_tree`] is not
    /// applicable. Without per-segment filtering here, the fanout
    /// path would either drop every BKD hit (when the wrapper falls
    /// back to the trait default returning `None`) or resurrect
    /// soft-deleted hits — re-introducing the #400 ghost-hit
    /// regression on top of the #480 fanout-path failure.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name whose per-segment BKD tree to return.
    ///
    /// # Returns
    ///
    /// `Ok(None)` if this segment has no BKD entries for `field`,
    /// `Ok(Some(...))` otherwise. The returned tree is wrapped in
    /// [`DeletionFilteringBKDTree`] only when the segment carries
    /// recorded deletions; otherwise the raw tree is returned with
    /// zero overhead.
    pub(crate) fn get_filtered_bkd_tree(&self, field: &str) -> Result<Option<Arc<dyn BKDTree>>> {
        let Some(tree) = self.get_bkd_tree(field)? else {
            return Ok(None);
        };
        // No bitmap — no deletions, or a `.delmap` that `open` found missing
        // — forwards the raw tree.
        let Some(deleted) = self.deletions() else {
            return Ok(Some(tree));
        };
        let snapshot = Arc::new(DeletionSnapshot {
            bitmaps: vec![(self.info.min_doc_id, self.info.max_doc_id, deleted)],
        });
        Ok(Some(Arc::new(DeletionFilteringBKDTree {
            inner: tree,
            snapshot,
        })))
    }
}

#[derive(Debug)]
struct MultiSegmentBKDTree {
    trees: Vec<Arc<dyn BKDTree>>,
}

impl BKDTree for MultiSegmentBKDTree {
    /// Forward the visitor to every per-segment tree in order. The visitor
    /// accumulates hits across segments; the trait's default
    /// `range_search` then sorts and dedups the combined output.
    fn intersect(
        &self,
        visitor: &mut dyn crate::lexical::index::structures::visitor::IntersectVisitor,
    ) -> Result<()> {
        for tree in &self.trees {
            tree.intersect(visitor)?;
        }
        Ok(())
    }
}

/// Lock-free snapshot of every segment deletion bitmap that has any
/// recorded deletions, captured at the time
/// [`InvertedIndexReader::get_bkd_tree`] returns the wrapper. Lookups
/// take no locks, fall through quickly for segments whose `(min, max)`
/// doc-id window does not contain the queried id, and short-circuit
/// to a no-op when no segment has deletions at all.
#[derive(Debug, Clone)]
struct DeletionSnapshot {
    /// `(min_doc_id, max_doc_id, deleted)` for each segment that has
    /// any recorded deletion. Segments with no deletions are not
    /// stored — they cannot contribute hits.
    bitmaps: Vec<(u64, u64, Arc<DeletedDocs>)>,
}

impl DeletionSnapshot {
    /// `true` when no segment in the reader has any deletion. The BKD
    /// wrapper checks this first and avoids wrapping the visitor at
    /// all in the common (no-deletions) case.
    #[inline]
    fn is_empty(&self) -> bool {
        self.bitmaps.is_empty()
    }

    /// Lock-free deletion check. Doc-ids outside any segment's
    /// `(min, max)` window are dispatched in O(num_segments_with_deletions)
    /// without touching the bitmap.
    #[inline]
    fn is_deleted(&self, doc_id: u64) -> bool {
        for (min, max, deleted) in &self.bitmaps {
            if doc_id >= *min && doc_id <= *max && deleted.contains(doc_id) {
                return true;
            }
        }
        false
    }
}

/// `BKDTree` decorator that drops doc-id hits whose underlying document
/// has been soft-deleted.
///
/// Why this layer exists: a `BKDTree` is a primitive over flat point /
/// doc-id buffers and does **not** know about per-segment deletion
/// bitmaps. A `delete_documents(_id)` followed by `commit()` records
/// the deletion in the segment's bitmap but the BKD entry survives in
/// the tree until the next merge — so without this decorator a
/// subsequent `range_search` / `intersect` would surface "ghost" hits
/// for deleted docs (manifesting as stale ids in the geo / geo3d /
/// numeric range query paths). Wrapping every tree returned by
/// [`InvertedIndexReader::get_bkd_tree`] makes every BKD-backed query
/// filter soft-deletes uniformly without per-query glue.
///
/// Performance: the snapshot is captured **once** when the wrapper is
/// constructed, so per-hit checks are lock-free vector lookups. When
/// no segment has any deletion, [`BKDTree::intersect`] forwards
/// verbatim and pays no overhead at all.
struct DeletionFilteringBKDTree {
    inner: Arc<dyn BKDTree>,
    snapshot: Arc<DeletionSnapshot>,
}

impl std::fmt::Debug for DeletionFilteringBKDTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeletionFilteringBKDTree")
            .field("inner", &self.inner)
            .field(
                "snapshot_segments_with_deletions",
                &self.snapshot.bitmaps.len(),
            )
            .finish()
    }
}

impl BKDTree for DeletionFilteringBKDTree {
    fn intersect(
        &self,
        visitor: &mut dyn crate::lexical::index::structures::visitor::IntersectVisitor,
    ) -> Result<()> {
        // Common case: no segment has any deletion. Skip wrapping
        // entirely so the inner BKD tree gets the user's visitor
        // verbatim and we pay zero overhead.
        if self.snapshot.is_empty() {
            return self.inner.intersect(visitor);
        }
        let mut wrapped = DeletionFilteringVisitor {
            inner: visitor,
            snapshot: &self.snapshot,
        };
        self.inner.intersect(&mut wrapped)
    }
}

/// `IntersectVisitor` decorator that drops `visit` / `visit_inside`
/// callbacks for doc-ids that the snapshot marks as deleted.
struct DeletionFilteringVisitor<'a> {
    inner: &'a mut dyn crate::lexical::index::structures::visitor::IntersectVisitor,
    snapshot: &'a DeletionSnapshot,
}

impl crate::lexical::index::structures::visitor::IntersectVisitor for DeletionFilteringVisitor<'_> {
    fn compare(
        &self,
        cell: &crate::lexical::index::structures::aabb::AABB,
    ) -> crate::lexical::index::structures::visitor::CellRelation {
        // Subtree pruning is purely a geometry decision — deletions do
        // not change cell extents, so we forward verbatim.
        self.inner.compare(cell)
    }

    fn visit_inside(&mut self, doc_id: u64) {
        if !self.snapshot.is_deleted(doc_id) {
            self.inner.visit_inside(doc_id);
        }
    }

    fn visit(&mut self, doc_id: u64, point: &[f64]) {
        if !self.snapshot.is_deleted(doc_id) {
            self.inner.visit(doc_id, point);
        }
    }
}

/// Rough per-entry footprint used to derive the term cache's entry capacity
/// from the byte-based memory limit and to report `memory_usage` in
/// [`CacheStats`]. The term cache is bounded by entry count (a proper LRU),
/// not by exact bytes.
const EST_TERM_ENTRY_BYTES: usize = 64;

/// Cache manager for efficient data access.
#[derive(Debug)]
pub struct CacheManager {
    /// Term information cache — a proper LRU (Issue #593).
    ///
    /// Keyed by `"field:term"`, valued by `Arc<TermInfo>` so a hit is a
    /// refcount bump shared with the caller rather than a deep clone of the
    /// `block_max` vector. A `Mutex` (not `RwLock`) guards it because
    /// [`LruCache::get`] takes `&mut self` to update recency.
    term_cache: Mutex<LruCache<String, Arc<TermInfo>>>,

    /// Maximum memory limit in bytes (informational; also derives the term
    /// cache's entry capacity).
    memory_limit: usize,

    /// Cache statistics.
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
}

impl CacheManager {
    /// Create a new cache manager.
    ///
    /// The term cache starts empty and grows with the entries it holds; none
    /// of the budget is reserved up front (Issue #1431), so building a reader
    /// costs the same for any budget.
    ///
    /// # Arguments
    ///
    /// * `memory_limit` - Soft memory budget in bytes; the term cache's entry
    ///   capacity is derived as `memory_limit / EST_TERM_ENTRY_BYTES` (at
    ///   least one entry).
    pub fn new(memory_limit: usize) -> Self {
        let capacity = NonZeroUsize::new((memory_limit / EST_TERM_ENTRY_BYTES).max(1))
            .unwrap_or(NonZeroUsize::MIN);
        CacheManager {
            term_cache: Mutex::new(LruCache::sparse(capacity)),
            memory_limit,
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
        }
    }

    /// Get term information from cache, bumping its recency on a hit.
    ///
    /// Returns a shared `Arc<TermInfo>` (refcount bump) on a hit, or `None` on
    /// a miss. Records the lookup in the hit / miss statistics.
    pub fn get_term_info(&self, key: &str) -> Option<Arc<TermInfo>> {
        let hit = self.term_cache.lock().get(key).cloned();
        if hit.is_some() {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    /// Cache term information. The LRU evicts the least-recently-used entry
    /// when the capacity is reached (Issue #593 — replaced the previous
    /// random ~25% eviction).
    pub fn cache_term_info(&self, key: String, info: TermInfo) {
        self.term_cache.lock().put(key, Arc::new(info));
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let entries = self.term_cache.lock().len();
        CacheStats {
            hits: self.cache_hits.load(Ordering::Relaxed),
            misses: self.cache_misses.load(Ordering::Relaxed),
            memory_usage: entries * EST_TERM_ENTRY_BYTES,
            memory_limit: self.memory_limit,
        }
    }
}

/// Cache performance statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: usize,

    /// Number of cache misses.
    pub misses: usize,

    /// Current memory usage.
    pub memory_usage: usize,

    /// Memory limit.
    pub memory_limit: usize,
}

impl CacheStats {
    /// Calculate hit ratio.
    pub fn hit_ratio(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }
}

/// Advanced index reader with multi-segment support (schema-less mode).
#[derive(Debug, Clone)]
pub struct InvertedIndexReader {
    /// Segment readers.
    segment_readers: Vec<Arc<RwLock<SegmentReader>>>,

    /// Segment metadata cached at construction time.
    ///
    /// Stored separately so that doc_id range checks can be performed
    /// without acquiring the per-segment `RwLock`.
    segment_infos: Vec<SegmentInfo>,

    /// Cache manager.
    cache_manager: Arc<CacheManager>,

    /// Snapshot-scoped query / filter result cache (Issue #578).
    ///
    /// `Arc` so that `#[derive(Clone)]` shares a single cache across clones of
    /// this reader rather than deep-cloning an empty one. The cache is bound to
    /// this reader's snapshot and is dropped when a new reader is built after a
    /// commit / optimize / refresh.
    query_cache: Arc<QueryFilterCache>,

    /// Reader configuration.
    config: InvertedIndexReaderConfig,

    /// Whether the reader is closed.
    closed: Arc<AtomicBool>,

    /// Total document count across all segments.
    total_doc_count: u64,

    /// Live document count summed over the segments, computed once per
    /// reader (Issue #1211). `Arc` so clones share it.
    live_doc_count: Arc<OnceLock<u64>>,

    /// Lazily computed: does every segment have a term dictionary, so that
    /// `term_info` / `term_doc_freq` account for every document (Issue
    /// #1196)? Filled on first use — the first `is_empty` of a search
    /// already loads every segment's `.dict`, so this costs no extra I/O —
    /// and shared by clones like the other snapshot-scoped state.
    term_info_complete: Arc<OnceLock<bool>>,

    /// Every segment's term dictionary, in segment order, collected on the
    /// first [`Self::live_doc_ids_for_term`] (Issue #1420); the inner `None`
    /// when some segment has none. Holding the `Arc`s lets a lookup probe
    /// each dictionary without taking the segment's locks.
    term_dictionaries: Arc<OnceLock<Option<Vec<Arc<BlockTermDictionary>>>>>,

    /// The index's filter over committed `_id` keys, if this reader was
    /// built by the index (Issue #1420); see [`Self::with_id_filter`].
    id_filter: Option<Arc<CommittedIdFilter>>,

    /// The filter generation this reader's segments were last covered at,
    /// 0 for never. Shared by clones like the other snapshot-scoped state.
    id_filter_generation: Arc<AtomicU64>,
}

impl InvertedIndexReader {
    /// Whether `term_info` and `term_doc_freq` reflect every document in this
    /// reader (Issue #1196).
    ///
    /// `false` when some segment has no term dictionary: its documents are
    /// still reachable through `postings` (the stored-document scan) but are
    /// absent from the dictionaries, so "the term has no entry" no longer
    /// means "the term matches nothing". `TermQuery::is_empty` and the
    /// `count` fast path consult this before trusting the dictionary.
    /// Vacuously `true` for a reader with no segments.
    pub fn term_info_is_authoritative(&self) -> bool {
        *self.term_info_complete.get_or_init(|| {
            self.segment_readers
                .iter()
                .all(|segment| segment.read().unwrap().has_term_dictionary())
        })
    }

    /// Create a new advanced index reader (schema-less mode).
    ///
    /// Opens every segment anew; see [`Self::with_segment_cache`] for the
    /// reader an index builds after a commit.
    pub fn new(
        segments: Vec<SegmentInfo>,
        storage: Arc<dyn Storage>,
        config: InvertedIndexReaderConfig,
    ) -> Result<Self> {
        let segment_readers = segments
            .iter()
            .map(|info| {
                Self::configured(SegmentReader::open(info.clone(), storage.clone())?, &config)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::from_segment_readers(
            segments,
            segment_readers,
            config,
        ))
    }

    /// Like [`Self::new`], for the segments `snapshot` lists, reusing the
    /// segment readers `cache` holds from the readers built before (Issue
    /// #1423).
    ///
    /// A segment whose metadata and deletions are unchanged keeps its reader,
    /// with everything it has loaded and its posting cache. A segment whose
    /// deletions changed gets a new reader over the same
    /// [`SegmentCore`], so it reads only its `.delmap`. Any other segment is
    /// opened. A reader built earlier keeps its own segment readers, so it
    /// still sees exactly its snapshot.
    ///
    /// The cached readers were configured by an earlier call, so every call
    /// with one `cache` must pass the same `config`, as the index's
    /// `reader()` does.
    ///
    /// # Arguments
    ///
    /// * `storage` - The index's storage.
    /// * `config` - Reader configuration.
    /// * `cache` - The segment readers of the reader built before.
    /// * `snapshot` - Lists the segments to read. Called with `cache` locked.
    ///
    /// # Errors
    ///
    /// Returns an error if `snapshot` fails, or if a segment cannot be opened
    /// or its `.delmap` read. `cache` then keeps the readers it held.
    pub(crate) fn with_segment_cache(
        storage: Arc<dyn Storage>,
        config: InvertedIndexReaderConfig,
        cache: &SegmentReaderCache,
        snapshot: impl FnOnce() -> Result<Vec<SegmentInfo>>,
    ) -> Result<Self> {
        let (segments, segment_readers) = cache.build(snapshot, |info, cached| {
            let reader = match cached {
                Some(cached) => match cached.read().unwrap().reopen(info)? {
                    None => return Ok(Arc::clone(cached)),
                    Some(reader) => reader,
                },
                None => SegmentReader::open(info.clone(), storage.clone())?,
            };
            Self::configured(reader, &config)
        })?;
        Ok(Self::from_segment_readers(
            segments,
            segment_readers,
            config,
        ))
    }

    /// `reader` set up for a reader with `config`, ready to share.
    fn configured(
        reader: SegmentReader,
        config: &InvertedIndexReaderConfig,
    ) -> Result<Arc<RwLock<SegmentReader>>> {
        // Enable the per-segment posting cache (Issue #612) for query readers,
        // gated by `enable_posting_cache` and budgeted by `max_cache_memory`.
        let posting_cache_bytes = if config.enable_posting_cache {
            config.max_cache_memory
        } else {
            0
        };
        let mut reader = reader
            .with_posting_cache_bytes(posting_cache_bytes)
            // The `.post`-less scan fallback must analyze like the
            // writer did (Issue #1196); the per-segment fanout shares
            // these same readers, so this covers every query path.
            .with_analyzer(config.analyzer.clone());
        if config.preload_segments {
            reader.load()?;
        }
        Ok(Arc::new(RwLock::new(reader)))
    }

    /// A reader over `segment_readers`, one per entry of `segments`.
    fn from_segment_readers(
        segments: Vec<SegmentInfo>,
        segment_readers: Vec<Arc<RwLock<SegmentReader>>>,
        config: InvertedIndexReaderConfig,
    ) -> Self {
        let total_doc_count = segments.iter().map(|info| info.doc_count).sum();
        InvertedIndexReader {
            segment_readers,
            segment_infos: segments,
            cache_manager: Arc::new(CacheManager::new(config.max_cache_memory)),
            query_cache: Arc::new(QueryFilterCache::new(config.query_filter_cache_capacity)),
            config,
            closed: Arc::new(AtomicBool::new(false)),
            total_doc_count,
            term_info_complete: Arc::new(OnceLock::new()),
            live_doc_count: Arc::new(OnceLock::new()),
            term_dictionaries: Arc::new(OnceLock::new()),
            id_filter: None,
            id_filter_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Answer `_id` lookups for absent ids from `filter` (Issue #1420).
    ///
    /// The filter is shared by every reader of one index and outlives them:
    /// a reader inserts the keys of its segments the filter has not seen yet
    /// on its first `_id` lookup, so after a commit only the new segments are
    /// read. Without a filter, [`Self::live_doc_ids_for_term`] probes every
    /// segment's dictionary.
    ///
    /// # Arguments
    ///
    /// * `filter` - The index's filter over committed `_id` keys.
    pub(crate) fn with_id_filter(mut self, filter: Arc<CommittedIdFilter>) -> Self {
        self.id_filter = Some(filter);
        self
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache_manager.stats()
    }

    /// Snapshot of the query / filter result cache hit / miss counters (Issue
    /// #578).
    pub fn query_cache_stats(
        &self,
    ) -> crate::lexical::index::inverted::query_cache::QueryFilterCacheStats {
        self.query_cache.stats()
    }

    /// Return the set of document ids matching `query` within this reader
    /// snapshot, consulting the snapshot-scoped query / filter cache (Issue
    /// [#578](https://github.com/mosuka/laurus/issues/578)).
    ///
    /// On a cache hit the stored [`RoaringTreemap`] is returned as a refcount
    /// bump. On a miss — or for an uncacheable query, i.e. one whose
    /// [`Query::cache_key`] is `None` — the query's matcher is drained into a
    /// fresh bitmap; cacheable results are then stored for reuse. The returned
    /// set is **score-independent** and excludes deleted documents (deletions
    /// are filtered at the posting-iterator level, so a posting-derived matcher
    /// never emits them).
    ///
    /// # Arguments
    ///
    /// * `query` - The query whose matching document set is requested.
    ///
    /// # Returns
    ///
    /// An `Arc<RoaringTreemap>` of matching document ids, shared with the cache
    /// when the query is cacheable.
    pub fn matching_doc_ids(&self, query: &dyn Query) -> Result<Arc<RoaringTreemap>> {
        if let Some(key) = query.cache_key() {
            if let Some(cached) = self.query_cache.get(&key) {
                return Ok(cached);
            }
            let bitmap = Arc::new(self.drain_matching(query)?);
            self.query_cache.put(key, Arc::clone(&bitmap));
            Ok(bitmap)
        } else {
            Ok(Arc::new(self.drain_matching(query)?))
        }
    }

    /// Drain `query`'s matcher over this reader into a [`RoaringTreemap`].
    fn drain_matching(&self, query: &dyn Query) -> Result<RoaringTreemap> {
        let matcher = query.matcher(self)?;
        crate::lexical::index::inverted::query_cache::drain_matcher(matcher)
    }

    /// Get the analyzer from configuration.
    pub fn analyzer(&self) -> &Arc<dyn Analyzer> {
        &self.config.analyzer
    }

    /// Number of segments backing this reader (#476 Phase 1).
    pub fn segment_count(&self) -> usize {
        self.segment_readers.len()
    }

    /// Borrow the per-segment readers (#476 Phase 1). Used by the
    /// inverted searcher's per-segment fanout path to run a query
    /// against each segment independently so PR-F's BMW pivot loop
    /// can fire on each segment's local `block_max` table.
    pub fn segment_readers(&self) -> &[Arc<RwLock<SegmentReader>>] {
        &self.segment_readers
    }

    /// Every segment's term dictionary, collected once per reader (Issue
    /// #1420); `None` when some segment has none.
    ///
    /// Also settles [`Self::term_info_is_authoritative`] when it is still
    /// open, so the search path does not read every `.dict` a second time.
    ///
    /// # Errors
    ///
    /// Returns an error if a segment's `.dict` opens but cannot be parsed.
    /// Nothing is cached then, so the next call tries again.
    fn term_dictionaries(&self) -> Result<Option<&[Arc<BlockTermDictionary>]>> {
        if let Some(cached) = self.term_dictionaries.get() {
            return Ok(cached.as_deref());
        }
        let mut dictionaries = Vec::with_capacity(self.segment_readers.len());
        for segment in &self.segment_readers {
            match segment.read().unwrap().loaded_term_dictionary()? {
                Some(dictionary) => dictionaries.push(dictionary),
                None => {
                    dictionaries.clear();
                    break;
                }
            }
        }
        let complete = dictionaries.len() == self.segment_readers.len();
        let _ = self.term_info_complete.set(complete);
        let _ = self.term_dictionaries.set(complete.then_some(dictionaries));
        Ok(self.term_dictionaries.get().and_then(Option::as_deref))
    }

    /// The live documents whose `field` holds exactly `term`, read from the
    /// segments' term dictionaries and postings (Issue #1420).
    ///
    /// Gives the same documents as a
    /// [`TermQuery`](crate::lexical::query::term::TermQuery) search for the
    /// term, without the query machinery: the key is built once, and only a
    /// segment whose dictionary has it decodes a posting list, which comes
    /// back without deleted documents. A term no segment holds costs one
    /// dictionary probe per segment, except an `_id` the index's
    /// [`CommittedIdFilter`] has never seen, which costs one filter probe in
    /// all (see [`Self::with_id_filter`]).
    ///
    /// # Arguments
    ///
    /// * `field` - Field the term belongs to.
    /// * `term` - The exact term, as indexed.
    ///
    /// # Returns
    ///
    /// The document ids in segment order, or `None` when some segment has no
    /// term dictionary (Issue #1196): it can still match through its stored
    /// documents, so the caller has to search instead.
    ///
    /// # Errors
    ///
    /// Returns an error if the reader is closed, or if a dictionary or a
    /// posting list cannot be read.
    pub fn live_doc_ids_for_term(&self, field: &str, term: &str) -> Result<Option<Vec<u64>>> {
        self.check_closed()?;
        let key = format!("{field}:{term}");
        if field == ID_FIELD
            && let Some(filter) = &self.id_filter
        {
            let check = filter.check(
                &self.id_filter_generation,
                &self.segment_infos,
                |index| {
                    self.segment_readers[index]
                        .read()
                        .unwrap()
                        .loaded_term_dictionary()
                },
                &key,
            )?;
            if check == IdCheck::Absent {
                return Ok(Some(Vec::new()));
            }
        }
        let Some(dictionaries) = self.term_dictionaries()? else {
            return Ok(None);
        };
        let mut doc_ids = Vec::new();
        for (segment, dictionary) in self.segment_readers.iter().zip(dictionaries) {
            if dictionary.get(&key).is_none() {
                continue;
            }
            if let Some(mut postings) = segment.read().unwrap().postings(field, term)? {
                while postings.next()? {
                    doc_ids.push(postings.doc_id());
                }
            }
        }
        Ok(Some(doc_ids))
    }

    /// Check if the reader is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(LaurusError::index("Reader is closed"))
        } else {
            Ok(())
        }
    }

    /// Get the field length for a specific document and field.
    ///
    /// Skips segments whose `[min_doc_id, max_doc_id]` range does not
    /// contain the requested `doc_id`, avoiding unnecessary lock
    /// acquisitions.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - The internal document ID.
    /// * `field` - The field name whose length is requested.
    ///
    /// # Returns
    ///
    /// `Ok(Some(length))` if found, `Ok(None)` otherwise.
    pub fn field_length(&self, doc_id: u64, field: &str) -> Result<Option<u32>> {
        self.check_closed()?;
        Ok(segment_field_length(self.segment_ranges(), doc_id, field))
    }

    /// Each segment with the `[min_doc_id, max_doc_id]` range its cached
    /// info says it can hold (the whole id space when that info is missing).
    fn segment_ranges(&self) -> impl Iterator<Item = (u64, u64, &Arc<RwLock<SegmentReader>>)> {
        self.segment_readers.iter().enumerate().map(|(i, segment)| {
            match self.segment_infos.get(i) {
                Some(info) => (info.min_doc_id, info.max_doc_id, segment),
                None => (0, u64::MAX, segment),
            }
        })
    }
}

/// `doc_id`'s length of `field`, from the first segment whose range holds
/// `doc_id` and that has a live value for it. The range check skips a segment
/// without taking its lock; a deleted copy (the old copy of a re-added id)
/// yields no value, so the search moves on to the live one.
fn segment_field_length<'a>(
    segments: impl Iterator<Item = (u64, u64, &'a Arc<RwLock<SegmentReader>>)>,
    doc_id: u64,
    field: &str,
) -> Option<u32> {
    for (min_doc_id, max_doc_id, segment) in segments {
        if doc_id < min_doc_id || doc_id > max_doc_id {
            continue;
        }
        if let Ok(Some(length)) = segment.read().unwrap().field_length(doc_id, field) {
            return Some(length);
        }
    }
    None
}

/// The per-document lengths of one field across a set of segments (#1287).
///
/// Each segment's length source is resolved when this is built, once per
/// scorer (#1298), so a lookup takes neither the segment's lock nor its norms
/// lock. It owns what it resolved, so a `'static` scorer can keep it for a
/// whole query.
#[derive(Debug)]
pub(crate) struct SegmentFieldLengths {
    segments: Vec<SegmentLengths>,
}

/// One segment's part of a [`SegmentFieldLengths`].
#[derive(Debug)]
struct SegmentLengths {
    /// The `[min_doc_id, max_doc_id]` range the segment can hold.
    min_doc_id: u64,
    max_doc_id: u64,
    /// `None` when the segment has no deletions.
    deletions: Option<Arc<DeletedDocs>>,
    norms: Arc<NormsReader>,
    /// The field's `.norms` column.
    column: NormsColumn,
}

impl SegmentLengths {
    /// `doc_id`'s length of the field; `None` when the segment cannot hold
    /// `doc_id`, holds only a deleted copy of it (the old copy of a re-added
    /// id, whose live copy is in another segment), or records no length.
    fn get(&self, doc_id: u64) -> Option<u32> {
        if doc_id < self.min_doc_id || doc_id > self.max_doc_id {
            return None;
        }
        if self
            .deletions
            .as_ref()
            .is_some_and(|deletions| deletions.contains(doc_id))
        {
            return None;
        }
        self.norms.column_length(self.column, doc_id)
    }
}

impl SegmentFieldLengths {
    /// Lengths of `field` across `segments`, each given with the doc id range
    /// it can hold. A segment no lookup could find a length in is left out:
    /// one whose length source fails to load, or one with no column for
    /// `field`. Leaving out a segment that fails to load is safe only
    /// because a scorer reads `field_stats` first, which reports that
    /// failure (such as a pre-`.norms` segment's rebuild error).
    pub(crate) fn new<'a>(
        field: &str,
        segments: impl IntoIterator<Item = (u64, u64, &'a Arc<RwLock<SegmentReader>>)>,
    ) -> Self {
        let segments = segments
            .into_iter()
            .filter_map(|(min_doc_id, max_doc_id, segment)| {
                let segment = segment.read().unwrap();
                let norms = segment.norms().ok()?;
                let column = norms.column(field)?;
                Some(SegmentLengths {
                    min_doc_id,
                    max_doc_id,
                    deletions: segment.deletions(),
                    norms,
                    column,
                })
            })
            .collect();
        SegmentFieldLengths { segments }
    }
}

impl crate::lexical::reader::FieldLengths for SegmentFieldLengths {
    fn get(&self, doc_id: u64) -> Option<u32> {
        self.segments.iter().find_map(|segment| segment.get(doc_id))
    }
}

impl crate::lexical::reader::LexicalIndexReader for InvertedIndexReader {
    fn term_info_is_authoritative(&self) -> bool {
        InvertedIndexReader::term_info_is_authoritative(self)
    }

    fn doc_count(&self) -> u64 {
        // Sum live doc counts from each segment (accounts for deletions).
        // Computed once per reader: a scorer asks for it per clause, and
        // every segment's count is fixed for the snapshot (Issue #1211).
        *self.live_doc_count.get_or_init(|| {
            self.segment_readers
                .iter()
                .map(|sr| sr.read().unwrap().doc_count())
                .sum()
        })
    }

    fn has_effective_deletions(&self) -> bool {
        self.segment_readers
            .iter()
            .any(|sr| sr.read().unwrap().has_effective_deletions())
    }

    fn max_doc(&self) -> u64 {
        // max_doc reflects the total allocated doc space (including deleted).
        self.total_doc_count
    }

    fn is_deleted(&self, doc_id: u64) -> bool {
        // Decided by the segments that hold the document (Issue #1211): the
        // same id can be deleted in one segment and live in another (a
        // re-added id), and a segment can carry a stray bit for an id it
        // does not hold. A bit counts only where the segment may hold the
        // id; a segment known to hold it without a bit keeps it live, and
        // one whose membership is unknown and has no bit does not vote.
        let covering: Vec<_> = self
            .segment_readers
            .iter()
            .filter(|sr| {
                let info = &sr.read().unwrap().info;
                doc_id >= info.min_doc_id && doc_id <= info.max_doc_id
            })
            .collect();
        let marked: Vec<bool> = covering
            .iter()
            .map(|sr| matches!(sr.read().unwrap().is_deleted(doc_id), Ok(true)))
            .collect();
        if !marked.iter().any(|&bit| bit) {
            // No covering segment marks it: no membership needs loading.
            return false;
        }
        let mut deleted = false;
        for (sr, bit) in covering.iter().zip(marked) {
            let reader = sr.read().unwrap();
            match (bit, reader.membership().holds(doc_id)) {
                (false, Some(true)) => return false,
                (true, Some(true) | None) => deleted = true,
                _ => {}
            }
        }
        deleted
    }

    fn live_doc_ids(&self) -> Result<Vec<u64>> {
        self.check_closed()?;
        let mut live = RoaringTreemap::new();
        for segment_reader in &self.segment_readers {
            live |= segment_reader.read().unwrap().live_doc_ids()?;
        }
        Ok(live.iter().collect())
    }

    fn document(&self, doc_id: u64) -> Result<Option<Document>> {
        self.check_closed()?;

        // Search across all segments
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            if let Ok(Some(doc)) = reader.document(doc_id) {
                return Ok(Some(doc));
            }
        }

        Ok(None)
    }

    fn document_fields(
        &self,
        doc_id: u64,
        field_names: &[&str],
    ) -> Result<Option<std::collections::HashMap<String, crate::data::DataValue>>> {
        self.check_closed()?;

        // Search across all segments — first hit wins, matching
        // `document()`'s behaviour. The per-segment override clones
        // only the requested fields, so wide schemas avoid the
        // whole-document clone (#410).
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            if let Ok(Some(fields)) = reader.document_fields(doc_id, field_names) {
                return Ok(Some(fields));
            }
        }

        Ok(None)
    }

    fn doc_ids(&self) -> Result<Vec<u64>> {
        self.check_closed()?;

        let mut all_ids = Vec::new();
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            all_ids.extend(reader.doc_ids()?);
        }
        Ok(all_ids)
    }

    fn term_info(
        &self,
        field: &str,
        term: &str,
    ) -> Result<Option<crate::lexical::reader::ReaderTermInfo>> {
        self.check_closed()?;

        let cache_key = format!("{field}:{term}");

        // Check cache first
        if let Some(cached_info) = self.cache_manager.get_term_info(&cache_key) {
            return Ok(Some(crate::lexical::reader::ReaderTermInfo {
                field: field.to_string(),
                term: term.to_string(),
                doc_freq: cached_info.doc_frequency,
                total_freq: cached_info.total_frequency,
                posting_offset: cached_info.posting_offset,
                posting_size: cached_info.posting_length,
                max_score_factor: cached_info.max_score_factor,
                block_max: cached_info.block_max.clone(),
            }));
        }

        // Search across all segments. Aggregate by taking the **max**
        // of per-segment factors — each segment computed
        // `max_score_factor` against its own `avg_field_length`, but
        // `max(seg_max)` remains a valid upper bound on any
        // individual posting's TF-component contribution **as long as
        // the query-time `avg_field_length` does not exceed the local
        // average each factor was anchored against** (#403 PR-B2;
        // soundness gap tracked and closed as #1120 — see the guard
        // below).
        //
        // Block-max metadata is concatenated across segments
        // (#403 PR-D). The inverted writer assigns segments
        // monotonically-increasing doc-id ranges, so segment-order
        // concatenation preserves the `last_doc_id` ordering that
        // [`BM25Scorer::block_max_score_at`]'s binary search relies
        // on.
        //
        // Per-block `max_factor` was computed against each segment's
        // local `avg_field_length`. `BM25Scorer::tf`'s TF component is
        // monotonically *increasing* in `avg_field_length` (a larger
        // average shrinks `field_length / avg_length`, which shrinks
        // the denominator), so a factor computed against a smaller
        // local average understates what the same posting would score
        // under a larger cross-segment average -- the precomputed
        // bound would then no longer be an upper bound. The guard
        // after this loop drops both `max_score_factor` and
        // `block_max` whenever that cannot be ruled out, falling back
        // to `BM25Scorer`'s always-valid loose `k1 + 1` ceiling
        // (`scorer.rs`'s `max_score`/`current_block_max_score`/
        // `block_max_score_at`). Tightening this (storing enough
        // per-block raw data to re-anchor the factor at query time) is
        // a format change tracked separately.
        let mut total_doc_freq = 0u64;
        let mut total_term_freq = 0u64;
        let mut max_score_factor: f32 = 0.0;
        let mut matched_count = 0_usize;
        let mut combined_block_max: Vec<crate::lexical::index::structures::dictionary::BlockMax> =
            Vec::new();
        // Smallest local `avg_field_length` among segments that matched
        // this term -- the soundness guard below needs the *tightest*
        // (smallest) local average, since the bound was computed against
        // whichever segment anchors the weakest.
        let mut min_matched_avg: Option<f64> = None;

        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            if let Some(term_info) = reader.term_info(field, term)? {
                // Saturating: on-disk statistics a corrupt segment can push
                // past u64 (Issue #1224).
                total_doc_freq = total_doc_freq.saturating_add(term_info.doc_frequency);
                total_term_freq = total_term_freq.saturating_add(term_info.total_frequency);
                max_score_factor = max_score_factor.max(term_info.max_score_factor);
                matched_count += 1;
                combined_block_max.extend(term_info.block_max.iter().copied());
                if let Some(stats) = reader.field_stats(field)? {
                    min_matched_avg =
                        Some(min_matched_avg.map_or(stats.avg_length, |m| m.min(stats.avg_length)));
                }
            }
        }

        let found = matched_count > 0;
        // Pass per-block metadata through for the single-segment case
        // only. With more than one matching segment, two costs offset
        // the bound's tightness:
        //
        // 1. The binary search inside `BM25Scorer::block_max_score_at`
        //    walks `O(log Σ blocks)` instead of `O(log blocks_in_one_segment)`,
        //    and on uniform corpora the per-block factor degenerates
        //    to the term-level `max_score_factor` anyway — leaving
        //    only the search overhead.
        // 2. Per-block factors were computed against per-segment
        //    `avg_field_length`. For corpora whose segments diverge
        //    in average length, the segment-local factor can drop
        //    below the cross-segment-anchored BM25 contribution and
        //    the searcher's break would fire too early.
        //
        // Falling back to the term-level `max_score_factor` in the
        // multi-segment case sidesteps both issues -- but does not by
        // itself restore soundness, since `max_score_factor` is still
        // carried forward: the guard below is what actually enforces
        // it (#1120).
        let aggregated_block_max = if matched_count == 1 {
            combined_block_max
        } else {
            Vec::new()
        };

        // Soundness guard (#1120): `max_score_factor`/`aggregated_block_max`
        // are only valid upper bounds if the `avg_field_length` a caller
        // will actually score against (`Self::field_stats`, which
        // `TermQuery::scorer` reads from this same reader) does not exceed
        // the smallest local average any matched segment anchored its
        // bound to. Compared at `f32` precision because that is the
        // granularity `BM25Scorer::tf` actually computes at
        // (`avg_field_length as f32`) -- comparing at `f64` could reject a
        // bound that is provably safe once narrowed to `f32`.
        let (max_score_factor, aggregated_block_max) = if found {
            match min_matched_avg {
                Some(min_avg) => match self.field_stats(field)? {
                    Some(global_stats) if (global_stats.avg_length as f32) <= (min_avg as f32) => {
                        (max_score_factor, aggregated_block_max)
                    }
                    _ => (0.0, Vec::new()),
                },
                None => (0.0, Vec::new()),
            }
        } else {
            (max_score_factor, aggregated_block_max)
        };

        if found {
            let reader_info = crate::lexical::reader::ReaderTermInfo {
                field: field.to_string(),
                term: term.to_string(),
                doc_freq: total_doc_freq,
                total_freq: total_term_freq,
                posting_offset: 0, // Aggregated value, not meaningful for multi-segment
                posting_size: 0,   // Aggregated value, not meaningful for multi-segment
                max_score_factor,
                block_max: aggregated_block_max.clone(),
            };

            let term_info = TermInfo {
                posting_offset: 0,
                posting_length: 0,
                doc_frequency: total_doc_freq,
                total_frequency: total_term_freq,
                max_score_factor,
                block_max: aggregated_block_max,
            };
            self.cache_manager.cache_term_info(cache_key, term_info);

            Ok(Some(reader_info))
        } else {
            Ok(None)
        }
    }

    fn postings(
        &self,
        field: &str,
        term: &str,
    ) -> Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>> {
        self.check_closed()?;

        let mut iterators = Vec::new();

        // Collect posting iterators from all segments
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            if let Some(iter) = reader.postings(field, term)? {
                iterators.push(iter);
            }
        }

        if iterators.is_empty() {
            Ok(None)
        } else if iterators.len() == 1 {
            // Single segment case
            Ok(Some(iterators.into_iter().next().unwrap()))
        } else {
            // Multi-segment case - merge iterators
            let merged = MergedPostingIterator::new(iterators)?;
            // If the merged iterator has no documents (all underlying iterators empty),
            // it's effectively empty, but we return it anyway as it handles logic correctly.
            Ok(Some(Box::new(merged)))
        }
    }

    fn field_stats(&self, field: &str) -> Result<Option<crate::lexical::reader::FieldStats>> {
        self.check_closed()?;

        let mut total_doc_count = 0u64;
        // Sum of (avg_length * doc_count) for the weighted average. `f64`,
        // not `u64` (#1120): truncating each segment's contribution before
        // summing made even a single-segment index's aggregate diverge
        // slightly from that segment's own `avg_length` -- and the
        // `term_info` soundness guard below relies on the two being
        // exactly equal in that case.
        let mut total_length_sum = 0.0f64;
        let mut min_length = u64::MAX;
        let mut max_length = 0u64;
        let mut found = false;

        // Aggregate statistics from all segments
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();

            // Get field stats from this segment
            if let Some(segment_stats) = reader.field_stats(field)? {
                total_doc_count += segment_stats.doc_count;
                total_length_sum += segment_stats.avg_length * segment_stats.doc_count as f64;
                min_length = min_length.min(segment_stats.min_length);
                max_length = max_length.max(segment_stats.max_length);
                found = true;
            }
        }

        if found {
            Ok(Some(crate::lexical::reader::FieldStats {
                field: field.to_string(),
                unique_terms: 0, // Not aggregated
                total_terms: 0,  // Not aggregated
                doc_count: total_doc_count,
                avg_length: if total_doc_count > 0 {
                    total_length_sum / total_doc_count as f64
                } else {
                    0.0
                },
                min_length: if min_length == u64::MAX {
                    0
                } else {
                    min_length
                },
                max_length,
            }))
        } else {
            Ok(None)
        }
    }

    fn close(&mut self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Searches every segment for `doc_id`'s value, returning the first
    /// hit. `Ok(None)` means either no segment has a DocValues column for
    /// `field`, or every segment that does simply lacks a value for this
    /// particular doc — the two cases are indistinguishable from this
    /// return value alone. Callers that also consult
    /// [`Self::has_doc_values`] to decide whether to read DocValues at
    /// all must still treat `Ok(None)` from this method as "fall back to
    /// the stored document", not as "the value is absent" (Issue #1047):
    /// segments can disagree on whether they have the column (mixed
    /// old/new segments, or a field with `doc_values: false`), so
    /// `has_doc_values() == true` index-wide does not guarantee this
    /// specific document's segment has it.
    fn get_doc_value(&self, field: &str, doc_id: u64) -> Result<Option<FieldValue>> {
        // Search across all segments
        for segment_lock in &self.segment_readers {
            let segment = segment_lock.read().unwrap();
            if let Ok(Some(value)) = segment.get_doc_value(field, doc_id) {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    /// Returns whether ANY segment has a DocValues column for `field` —
    /// an index-wide, not per-document, answer. `true` does not mean
    /// every document has a value via [`Self::get_doc_value`]: segments
    /// can disagree (Issue #1047), so a caller must still fall back to
    /// the stored document on an `Ok(None)` miss from `get_doc_value`
    /// rather than treating this method's `true` as a per-doc guarantee.
    fn has_doc_values(&self, field: &str) -> bool {
        // Check if any segment has DocValues for this field
        self.segment_readers.iter().any(|seg_lock| {
            let seg = seg_lock.read().unwrap();
            seg.has_doc_values(field)
        })
    }

    fn field_lengths(&self, field: &str) -> Option<Arc<dyn crate::lexical::reader::FieldLengths>> {
        self.check_closed().ok()?;
        Some(Arc::new(SegmentFieldLengths::new(
            field,
            self.segment_ranges(),
        )))
    }

    fn get_bkd_tree(&self, field: &str) -> Result<Option<Arc<dyn BKDTree>>> {
        self.check_closed()?;

        // Each segment's tree is filtered by that segment's own deletion
        // bitmap (Issue #1211). A tree emits only the ids its segment holds,
        // so its own bits decide exactly; one snapshot ORing every segment's
        // bits instead let another segment's bit for the same id — the old
        // copy of a re-added id, or a stray bit — hide a live hit. Segments
        // without deletions forward their raw tree, so an index with none
        // still pays nothing per hit.
        let mut trees = Vec::new();
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            if let Some(tree) = reader.get_filtered_bkd_tree(field)? {
                trees.push(tree);
            }
        }

        if trees.is_empty() {
            return Ok(None);
        }
        Ok(Some(Arc::new(MultiSegmentBKDTree { trees })))
    }
}

// Implementation of TermDictionaryAccess for InvertedIndexReader
impl TermDictionaryAccess for InvertedIndexReader {
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms>>> {
        // Collect term dictionaries from ALL segments and merge them.
        let mut dicts = Vec::new();
        for seg_lock in &self.segment_readers {
            if let Some(dict) = seg_lock.read().unwrap().loaded_term_dictionary()? {
                dicts.push(dict);
            }
        }

        if dicts.is_empty() {
            return Ok(None);
        }

        // If only one segment, use the fast path
        if dicts.len() == 1 {
            let terms = InvertedIndexTerms::new(field, dicts.into_iter().next().unwrap());
            return Ok(Some(Box::new(terms)));
        }

        // Merge terms across all segments
        let terms = MergedInvertedIndexTerms::new(field, &dicts);
        Ok(Some(Box::new(terms)))
    }
}

/// Iterator that merges multiple posting iterators into a single stream.
///
/// This iterator maintains a priority queue of active iterators, always
/// processing the one with the smallest document ID first. This ensures
/// that document IDs are returned in ascending order across all segments.
///
/// Two storage strategies are chosen at construction time (#412):
///
/// - **Linear scan** for small segment counts (≤ [`LINEAR_THRESHOLD`]).
///   Sub-iterators sit in a `Vec` and the current minimum is found
///   with an `O(n)` scan at each `advance`. For typical multi-segment
///   reads (`n` between 2 and 8) this beats the heap by 1.5–2× — the
///   constant factor of a heap pop / push (vtable indirection,
///   reordering writes) outweighs the algorithmic `O(log n)` benefit
///   when `n` is small.
/// - **Heap** for larger segment counts (`> LINEAR_THRESHOLD`). The
///   pre-#412 implementation, kept verbatim for big merges where the
///   heap's algorithmic edge dominates.
#[derive(Debug)]
pub struct MergedPostingIterator {
    /// Storage strategy chosen at construction time based on segment count.
    inner: MergeImpl,

    /// The current document ID of the merged stream.
    current_doc: u64,

    /// Whether next() has been called at least once.
    /// Matches the same protocol as InvertedIndexPostingIterator:
    /// first next() positions at the first document without advancing.
    started: bool,
}

/// Segment-count threshold above which the merger switches from the
/// linear-scan path to the heap path (#412). Picked at 8 to match
/// Lucene's `MultiBits` heuristic — empirically the crossover sits
/// between 4 and 8 on the `posting_merge_bench` scenarios.
const LINEAR_THRESHOLD: usize = 8;

/// Storage strategy used by [`MergedPostingIterator`] (#412).
#[derive(Debug)]
enum MergeImpl {
    /// Small-`n` path: linear scan over a `Vec<IteratorWrapper>`.
    /// `min_idx` is the index of the wrapper currently at the
    /// minimum doc id; `advance` advances `wrappers[min_idx]` and
    /// re-finds the minimum.
    Linear {
        wrappers: Vec<IteratorWrapper>,
        min_idx: usize,
    },
    /// Large-`n` path: standard min-heap.
    Heap(std::collections::BinaryHeap<IteratorWrapper>),
}

/// Wrapper for PostingIterator to make it orderable for BinaryHeap.
#[derive(Debug)]
struct IteratorWrapper {
    iter: Box<dyn crate::lexical::reader::PostingIterator>,
    current_doc: u64,
}

impl PartialEq for IteratorWrapper {
    fn eq(&self, other: &Self) -> bool {
        self.current_doc == other.current_doc
    }
}

impl Eq for IteratorWrapper {}

impl PartialOrd for IteratorWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IteratorWrapper {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for Min-Heap (smallest doc_id at top)
        other.current_doc.cmp(&self.current_doc)
    }
}

/// Linear scan for the index of the minimum-`current_doc` wrapper.
///
/// Caller must ensure `wrappers` is non-empty; the search starts at
/// index 0 and updates the running minimum on every step. Used by
/// the `Linear` storage strategy of `MergedPostingIterator`.
#[inline]
fn find_min_idx(wrappers: &[IteratorWrapper]) -> usize {
    let mut min_idx = 0;
    let mut min_doc = wrappers[0].current_doc;
    for (i, w) in wrappers.iter().enumerate().skip(1) {
        if w.current_doc < min_doc {
            min_doc = w.current_doc;
            min_idx = i;
        }
    }
    min_idx
}

impl MergedPostingIterator {
    /// Create a new merged iterator from a list of iterators.
    ///
    /// Each sub-iterator is advanced to its first document during construction
    /// so the heap can be properly ordered. However, `next()` must still be called
    /// once before reading `doc_id()`, matching the `InvertedIndexPostingIterator`
    /// protocol (via the `started` flag).
    pub fn new(iterators: Vec<Box<dyn crate::lexical::reader::PostingIterator>>) -> Result<Self> {
        let mut wrappers = Vec::with_capacity(iterators.len());

        for mut iter in iterators {
            if iter.next()? {
                let doc_id = iter.doc_id();
                wrappers.push(IteratorWrapper {
                    iter,
                    current_doc: doc_id,
                });
            }
        }

        let inner = if wrappers.len() <= LINEAR_THRESHOLD {
            let min_idx = if wrappers.is_empty() {
                0
            } else {
                find_min_idx(&wrappers)
            };
            MergeImpl::Linear { wrappers, min_idx }
        } else {
            let mut heap = std::collections::BinaryHeap::with_capacity(wrappers.len());
            for w in wrappers {
                heap.push(w);
            }
            MergeImpl::Heap(heap)
        };

        let current_doc = match &inner {
            MergeImpl::Linear { wrappers, min_idx } => {
                if wrappers.is_empty() {
                    u64::MAX
                } else {
                    wrappers[*min_idx].current_doc
                }
            }
            MergeImpl::Heap(heap) => heap.peek().map_or(u64::MAX, |w| w.current_doc),
        };

        Ok(MergedPostingIterator {
            inner,
            current_doc,
            started: false,
        })
    }

    /// Internal advance: move the current minimum's underlying
    /// iterator forward, then re-locate the minimum across the
    /// remaining iterators. Used by both `next()` (after the started
    /// check) and `skip_to()`.
    fn advance(&mut self) -> Result<bool> {
        match &mut self.inner {
            MergeImpl::Linear { wrappers, min_idx } => {
                if wrappers.is_empty() {
                    self.current_doc = u64::MAX;
                    return Ok(false);
                }
                let idx = *min_idx;
                if wrappers[idx].iter.next()? {
                    wrappers[idx].current_doc = wrappers[idx].iter.doc_id();
                } else {
                    // Iterator exhausted — drop it from the active set
                    // via swap_remove (O(1)) since order is restored
                    // by the next `find_min_idx` scan anyway.
                    wrappers.swap_remove(idx);
                }
                if wrappers.is_empty() {
                    self.current_doc = u64::MAX;
                    Ok(false)
                } else {
                    *min_idx = find_min_idx(wrappers);
                    self.current_doc = wrappers[*min_idx].current_doc;
                    Ok(true)
                }
            }
            MergeImpl::Heap(heap) => {
                if let Some(mut wrapper) = heap.pop() {
                    if wrapper.iter.next()? {
                        wrapper.current_doc = wrapper.iter.doc_id();
                        heap.push(wrapper);
                    }
                    if let Some(new_top) = heap.peek() {
                        self.current_doc = new_top.current_doc;
                        Ok(true)
                    } else {
                        self.current_doc = u64::MAX;
                        Ok(false)
                    }
                } else {
                    self.current_doc = u64::MAX;
                    Ok(false)
                }
            }
        }
    }

    /// Reference to the wrapper currently at the merged stream's
    /// minimum, used by `term_freq()` and `positions()` to delegate
    /// to the active sub-iterator.
    fn current_wrapper(&self) -> Option<&IteratorWrapper> {
        match &self.inner {
            MergeImpl::Linear { wrappers, min_idx } => {
                if wrappers.is_empty() {
                    None
                } else {
                    Some(&wrappers[*min_idx])
                }
            }
            MergeImpl::Heap(heap) => heap.peek(),
        }
    }
}

impl crate::lexical::reader::PostingIterator for MergedPostingIterator {
    fn doc_id(&self) -> u64 {
        self.current_doc
    }

    fn term_freq(&self) -> u64 {
        self.current_wrapper().map_or(0, |w| w.iter.term_freq())
    }

    fn positions(&self) -> Result<Vec<u64>> {
        self.current_wrapper()
            .map_or(Ok(Vec::new()), |w| w.iter.positions())
    }

    fn next(&mut self) -> Result<bool> {
        if !self.started {
            // First call: just mark as started without advancing.
            // The merger is already positioned at the first document from new().
            self.started = true;
            let exhausted = match &self.inner {
                MergeImpl::Linear { wrappers, .. } => wrappers.is_empty(),
                MergeImpl::Heap(heap) => heap.is_empty(),
            };
            return Ok(!exhausted);
        }

        self.advance()
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        // Ensure started before skipping
        if !self.started {
            self.started = true;
        }

        // Naive implementation: just call next until we reach or pass target
        // (call the advancing logic directly, not via next() which checks started)
        while self.doc_id() < target {
            if !self.advance()? {
                return Ok(false);
            }
        }

        Ok(self.doc_id() != u64::MAX)
    }

    fn cost(&self) -> u64 {
        match &self.inner {
            MergeImpl::Linear { wrappers, .. } => wrappers.iter().map(|w| w.iter.cost()).sum(),
            MergeImpl::Heap(heap) => heap.iter().map(|w| w.iter.cost()).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::reader::PostingIterator;

    // ---- `.post`-less segments: the `scan_documents_for_term` fallback,
    // ---- Issue #1194 ---------------------------------------------------

    /// Hand-build a segment holding ONLY a `.docs` part — no `.post`, `.dict`
    /// or `.norms` — so [`SegmentReader::postings`] has to take the
    /// `scan_documents_for_term` fallback. `docs` are
    /// `(doc_id, [(field, stored value)])`. Returns the storage and the
    /// `SegmentInfo` so a test can add a `.delmap` before opening the reader.
    fn docs_only_segment(
        segment_id: &str,
        docs: &[(u64, Vec<(&str, crate::data::DataValue)>)],
        has_deletions: bool,
    ) -> (Arc<dyn crate::storage::Storage>, SegmentInfo) {
        use crate::lexical::core::analyzed::BufferedDocument;
        use crate::lexical::index::structures::stored_fields::StoredFieldsWriter;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
        use crate::storage::structured::StructWriter;

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let analyzed: Vec<(u64, BufferedDocument)> = docs
            .iter()
            .map(|(doc_id, fields)| {
                let mut doc = BufferedDocument::default();
                for (field, value) in fields {
                    doc.stored_fields
                        .insert((*field).to_string(), value.clone());
                }
                (*doc_id, doc)
            })
            .collect();
        {
            let output = storage
                .create_output(&format!("{segment_id}.docs"))
                .unwrap();
            let mut w = StructWriter::new(output);
            StoredFieldsWriter::write_to(&mut w, &analyzed).unwrap();
            w.close().unwrap();
        }
        assert!(
            !storage.file_exists(&format!("{segment_id}.post")),
            "the fixture must not have a postings part, or the scan fallback is never taken"
        );
        let info = SegmentInfo {
            segment_id: segment_id.to_string(),
            doc_count: docs.len() as u64,
            min_doc_id: docs.iter().map(|d| d.0).min().unwrap_or(0),
            max_doc_id: docs.iter().map(|d| d.0).max().unwrap_or(0),
            generation: 0,
            has_deletions,
            deleted_count: None,
            shard_id: 0,
        };
        (storage, info)
    }

    /// `(doc_id, term_freq, positions)` for every posting `postings(field, term)`
    /// yields on `reader`, in doc-id order.
    fn scan_hits(reader: &SegmentReader, field: &str, term: &str) -> Vec<(u64, u64, Vec<u64>)> {
        let mut hits = Vec::new();
        if let Some(mut iter) = reader.postings(field, term).unwrap() {
            while iter.next().unwrap() {
                hits.push((iter.doc_id(), iter.term_freq(), iter.positions().unwrap()));
            }
        }
        hits
    }

    fn texts(items: &[&str]) -> crate::data::DataValue {
        crate::data::DataValue::TextArray(items.iter().map(|s| (*s).to_string()).collect())
    }

    fn text(s: &str) -> crate::data::DataValue {
        crate::data::DataValue::Text(s.to_string())
    }

    #[test]
    fn scan_fallback_matches_any_element_of_a_text_array() {
        let (storage, info) = docs_only_segment(
            "scan_any",
            &[
                (0, vec![("tags", texts(&["rust", "search engine"]))]),
                (1, vec![("tags", text("rust"))]),
                (2, vec![("other", text("rust"))]),
            ],
            false,
        );
        let reader = SegmentReader::open(info, storage).unwrap();

        // Array element and scalar both match; another field does not.
        assert_eq!(
            scan_hits(&reader, "tags", "rust"),
            vec![(0, 1, vec![0]), (1, 1, vec![0])]
        );
        // Elements are analyzed individually; the second element's tokens
        // start `position_increment_gap` past the first element's last one.
        assert_eq!(
            scan_hits(&reader, "tags", "search"),
            vec![(0, 1, vec![101])]
        );
        assert_eq!(
            scan_hits(&reader, "tags", "engine"),
            vec![(0, 1, vec![102])]
        );
        assert!(scan_hits(&reader, "tags", "missing").is_empty());
        assert!(reader.postings("tags", "missing").unwrap().is_none());
    }

    #[test]
    fn scan_fallback_positions_span_elements_with_the_gap() {
        // Parity with the writer's `TextArray` arm: `foo` sits at
        // 2 (tokens of "hello world") + 100 (gap) = 102, so a phrase across
        // the element boundary needs slop >= gap exactly as on the indexed
        // path (multi_valued_text_test.rs).
        let (storage, info) = docs_only_segment(
            "scan_gap",
            &[(0, vec![("body", texts(&["hello world", "foo bar"]))])],
            false,
        );
        let reader = SegmentReader::open(info, storage).unwrap();
        assert_eq!(scan_hits(&reader, "body", "hello"), vec![(0, 1, vec![0])]);
        assert_eq!(scan_hits(&reader, "body", "world"), vec![(0, 1, vec![1])]);
        assert_eq!(scan_hits(&reader, "body", "foo"), vec![(0, 1, vec![102])]);
        assert_eq!(scan_hits(&reader, "body", "bar"), vec![(0, 1, vec![103])]);
    }

    #[test]
    fn scan_fallback_tf_accumulates_across_elements() {
        // One posting per document; repeated elements raise the term
        // frequency, not the hit count.
        let (storage, info) = docs_only_segment(
            "scan_tf",
            &[(0, vec![("tags", texts(&["rust", "rust tooling"]))])],
            false,
        );
        let reader = SegmentReader::open(info, storage).unwrap();
        assert_eq!(
            scan_hits(&reader, "tags", "rust"),
            vec![(0, 2, vec![0, 101])]
        );
    }

    #[test]
    fn scan_fallback_matches_bool_and_numeric_terms_like_the_writer() {
        use crate::data::DataValue;
        // Before #1194 only `Text` could match on this path; the writer
        // indexes booleans and numbers as terms, so the scan must too.
        let (storage, info) = docs_only_segment(
            "scan_terms",
            &[
                (
                    0,
                    vec![("flag", DataValue::Bool(true)), ("n", DataValue::Int64(42))],
                ),
                (1, vec![("flags", DataValue::BoolArray(vec![true, false]))]),
                (
                    2,
                    vec![
                        ("flags", DataValue::BoolArray(vec![false])),
                        ("f", DataValue::Float64(2.5)),
                    ],
                ),
            ],
            false,
        );
        let reader = SegmentReader::open(info, storage).unwrap();
        assert_eq!(scan_hits(&reader, "flag", "true"), vec![(0, 1, vec![0])]);
        assert_eq!(scan_hits(&reader, "flags", "true"), vec![(1, 1, vec![0])]);
        assert_eq!(
            scan_hits(&reader, "flags", "false"),
            vec![(1, 1, vec![1]), (2, 1, vec![0])]
        );
        assert_eq!(scan_hits(&reader, "n", "42"), vec![(0, 1, vec![0])]);
        assert_eq!(scan_hits(&reader, "f", "2.5"), vec![(2, 1, vec![0])]);
    }

    #[test]
    fn scan_fallback_renumbers_positions_like_the_writer() {
        // `StandardAnalyzer` drops "the" as a stop word; the writer then
        // renumbers the surviving tokens densely (`tokens_to_analyzed_terms`),
        // so `search` is at 1, not at the tokenizer's 2. The old scan used
        // the tokenizer positions and disagreed with the postings.
        let (storage, info) = docs_only_segment(
            "scan_renumber",
            &[(0, vec![("body", text("the rust search"))])],
            false,
        );
        let reader = SegmentReader::open(info, storage).unwrap();
        assert_eq!(scan_hits(&reader, "body", "rust"), vec![(0, 1, vec![0])]);
        assert_eq!(scan_hits(&reader, "body", "search"), vec![(0, 1, vec![1])]);
        assert!(scan_hits(&reader, "body", "the").is_empty());
    }

    /// The `.post`-less half of the #541 invariant (see
    /// `postings_never_yields_a_deleted_document` for the normal path).
    #[test]
    fn scan_fallback_never_yields_a_deleted_document() {
        use crate::maintenance::deletion::{DeletionConfig, DeletionManager};

        let (storage, info) = docs_only_segment(
            "scan_deleted",
            &[
                (0, vec![("body", text("alpha"))]),
                (1, vec![("body", text("alpha"))]),
                (2, vec![("body", texts(&["alpha", "beta"]))]),
            ],
            true,
        );
        let manager = DeletionManager::new(
            DeletionConfig {
                enable_deletion_log: false,
                ..Default::default()
            },
            storage.clone(),
        )
        .unwrap();
        manager
            .initialize_segment(&info.segment_id, info.min_doc_id, info.max_doc_id)
            .unwrap();
        manager
            .delete_document(&info.segment_id, 1, "test")
            .unwrap();
        manager.flush().unwrap();

        let reader = SegmentReader::open(info, storage).unwrap();
        assert_eq!(
            scan_hits(&reader, "body", "alpha"),
            vec![(0, 1, vec![0]), (2, 1, vec![0])]
        );
    }

    // ---- An unreadable `.delmap` is corruption, not "no deletions",
    // ---- Issue #1265 ---------------------------------------------------

    /// A two-document segment with `has_deletions` set and no `.delmap` yet.
    fn segment_marked_deleted(segment_id: &str) -> (Arc<dyn crate::storage::Storage>, SegmentInfo) {
        docs_only_segment(
            segment_id,
            &[
                (0, vec![("body", text("alpha"))]),
                (1, vec![("body", text("alpha"))]),
            ],
            true,
        )
    }

    /// Flushes a real `.delmap` for `info`'s segment with doc 1 deleted, and
    /// returns its bytes.
    fn flush_delmap(storage: &Arc<dyn crate::storage::Storage>, info: &SegmentInfo) -> Vec<u8> {
        use std::io::Read;

        use crate::maintenance::deletion::{DeletionConfig, DeletionManager};

        let manager = DeletionManager::new(
            DeletionConfig {
                enable_deletion_log: false,
                ..Default::default()
            },
            storage.clone(),
        )
        .unwrap();
        manager
            .initialize_segment(&info.segment_id, info.min_doc_id, info.max_doc_id)
            .unwrap();
        manager
            .delete_document(&info.segment_id, 1, "test")
            .unwrap();
        manager.flush().unwrap();

        let mut bytes = Vec::new();
        storage
            .open_input(&format!("{}.delmap", info.segment_id))
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    fn write_delmap(storage: &Arc<dyn crate::storage::Storage>, info: &SegmentInfo, bytes: &[u8]) {
        use std::io::Write;

        let mut output = storage
            .create_output(&format!("{}.delmap", info.segment_id))
            .unwrap();
        output.write_all(bytes).unwrap();
        output.close().unwrap();
    }

    /// Asserts that `info`'s segment refuses to open with a corruption error
    /// naming the segment and containing `what`. Opening it as "no
    /// deletions" would bring its deleted documents back into search results
    /// and counts.
    fn assert_open_refused(
        storage: Arc<dyn crate::storage::Storage>,
        info: SegmentInfo,
        what: &str,
    ) {
        let expected = format!(
            "segment {}: failed to load its deletion bitmap",
            info.segment_id
        );
        match SegmentReader::open(info, storage) {
            Err(LaurusError::Index(msg)) => {
                assert!(
                    msg.contains(&expected),
                    "the error names the segment: {msg}"
                );
                assert!(msg.contains(what), "{msg}");
            }
            Err(other) => panic!("expected an index corruption error, got {other:?}"),
            Ok(_) => panic!("an unreadable deletion bitmap must not open as \"no deletions\""),
        }
    }

    #[test]
    fn a_segment_whose_deletion_bitmap_cannot_be_parsed_refuses_to_open() {
        use crate::storage::structured::StructWriter;

        let (storage, info) = segment_marked_deleted("delmap_bad_magic");
        let output = storage
            .create_output(&format!("{}.delmap", info.segment_id))
            .unwrap();
        let mut w = StructWriter::new(output);
        w.write_u32(0xDEAD_BEEF).unwrap(); // not the "DELB" magic
        w.close().unwrap();

        assert_open_refused(storage, info, "Invalid deletion bitmap format");
    }

    /// The case #1214 made common: one flipped bit in a field nothing else
    /// checks still parses, and only the checksum catches it.
    #[test]
    fn a_segment_whose_deletion_bitmap_fails_its_checksum_refuses_to_open() {
        let (storage, info) = segment_marked_deleted("delmap_flipped");
        let mut bytes = flush_delmap(&storage, &info);
        let reader = SegmentReader::open(info.clone(), storage.clone()).unwrap();
        assert_eq!(reader.doc_count(), 1, "the intact bitmap hides doc 1");

        // magic (4) + version (4) + segment id (1 + len) + total_docs (8) +
        // deleted_count (8), then last_modified.
        let last_modified = 4 + 4 + 1 + info.segment_id.len() + 8 + 8;
        bytes[last_modified] ^= 0x01;
        write_delmap(&storage, &info, &bytes);

        assert_open_refused(storage, info, "checksum mismatch");
    }

    /// A real `.delmap` cut short — what a torn write leaves.
    #[test]
    fn a_segment_whose_deletion_bitmap_is_truncated_refuses_to_open() {
        let (storage, info) = segment_marked_deleted("delmap_truncated");
        let bytes = flush_delmap(&storage, &info);
        write_delmap(&storage, &info, &bytes[..bytes.len() / 2]);

        assert_open_refused(storage, info, "");
    }

    /// `has_deletions` without a `.delmap` file still opens as a segment with
    /// no deletions: a missing bitmap is not a corrupted one.
    #[test]
    fn a_segment_with_no_deletion_bitmap_file_still_opens_as_undeleted() {
        let (storage, info) = segment_marked_deleted("delmap_absent");
        assert!(!storage.file_exists("delmap_absent.delmap"));

        let reader = SegmentReader::open(info, storage).unwrap();
        assert_eq!(reader.doc_count(), 2);
        assert!(!reader.has_effective_deletions());
        assert_eq!(
            scan_hits(&reader, "body", "alpha"),
            vec![(0, 1, vec![0]), (1, 1, vec![0])]
        );
    }

    // ---- Analyzer wiring and the once-per-segment warning, Issue #1196 --

    #[test]
    fn scan_fallback_uses_the_configured_analyzer() {
        use crate::analysis::analyzer::keyword::KeywordAnalyzer;

        // Under `KeywordAnalyzer` the whole value is one term, so the scan
        // must hit `"rust search"` and miss `"rust"` — the inverse of the
        // standard-analyzer result the other tests assert.
        let (storage, info) = docs_only_segment(
            "scan_keyword",
            &[(0, vec![("tags", text("rust search"))])],
            false,
        );
        let reader = SegmentReader::open(info, storage)
            .unwrap()
            .with_analyzer(Arc::new(KeywordAnalyzer::new()));
        assert_eq!(
            scan_hits(&reader, "tags", "rust search"),
            vec![(0, 1, vec![0])]
        );
        assert!(scan_hits(&reader, "tags", "rust").is_empty());
    }

    #[test]
    fn scan_fallback_defaults_to_standard_without_an_analyzer() {
        // A reader opened without `with_analyzer` keeps the historical
        // `StandardAnalyzer` behaviour.
        let (storage, info) = docs_only_segment(
            "scan_default",
            &[(0, vec![("tags", text("rust search"))])],
            false,
        );
        let reader = SegmentReader::open(info, storage).unwrap();
        assert_eq!(scan_hits(&reader, "tags", "rust"), vec![(0, 1, vec![0])]);
        assert_eq!(scan_hits(&reader, "tags", "search"), vec![(0, 1, vec![1])]);
        assert!(scan_hits(&reader, "tags", "rust search").is_empty());
    }

    #[test]
    fn scan_fallback_resolves_a_per_field_analyzer() {
        use crate::analysis::analyzer::keyword::KeywordAnalyzer;
        use crate::analysis::analyzer::per_field::PerFieldAnalyzer;

        // The engine's analyzer: standard by default, keyword for `_id`
        // (`Engine::split_schema`). `_id` lookups against a `.post`-less
        // segment must match the whole id while other fields tokenize.
        let per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
        per_field.add_analyzer("_id", Arc::new(KeywordAnalyzer::new()));
        let (storage, info) = docs_only_segment(
            "scan_per_field",
            &[(0, vec![("_id", text("doc-1")), ("body", text("doc-1"))])],
            false,
        );
        let reader = SegmentReader::open(info, storage)
            .unwrap()
            .with_analyzer(Arc::new(per_field));
        assert_eq!(scan_hits(&reader, "_id", "doc-1"), vec![(0, 1, vec![0])]);
        assert!(scan_hits(&reader, "_id", "doc").is_empty());
        assert_eq!(scan_hits(&reader, "body", "doc"), vec![(0, 1, vec![0])]);
        assert!(scan_hits(&reader, "body", "doc-1").is_empty());
    }

    #[test]
    fn scan_fallback_warns_once_per_segment() {
        let (storage, info) =
            docs_only_segment("scan_warn", &[(0, vec![("tags", text("rust"))])], false);
        let reader = SegmentReader::open(info, storage).unwrap();
        assert!(!reader.has_warned_missing_postings());
        let _ = reader.postings("tags", "rust").unwrap();
        assert!(reader.has_warned_missing_postings());
        // A second lookup must not re-arm the warning: the flag is a
        // one-way latch per reader instance.
        let _ = reader.postings("tags", "missing").unwrap();
        assert!(reader.has_warned_missing_postings());
    }

    // ---- Term-dictionary authority, Issue #1196 ---------------------------

    #[test]
    fn has_term_dictionary_reflects_the_dict_file() {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        // The `.docs`-only fixture has no dictionary ...
        let (storage, info) =
            docs_only_segment("scan_no_dict", &[(0, vec![("tags", text("rust"))])], false);
        assert!(
            !SegmentReader::open(info, storage)
                .unwrap()
                .has_term_dictionary()
        );

        // ... while a writer-built segment always carries one, so a reader
        // over it stays authoritative.
        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage.clone(), InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();
        writer
            .add_document(crate::Document::builder().add_text("body", "alpha").build())
            .unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        assert!(
            inverted.segment_readers()[0]
                .read()
                .unwrap()
                .has_term_dictionary()
        );
        assert!(inverted.term_info_is_authoritative());
    }

    /// Fetching a segment's dictionary, asking whether it has one, settling
    /// term-info authority and enumerating terms all use the dictionary
    /// loaded first (Issue #1423). `term_dictionary` and
    /// `has_term_dictionary` used to read `.dict` again on every call, so a
    /// reader settling authority parsed every segment's dictionary anew.
    #[test]
    fn the_term_dictionary_is_read_once() {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage, InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();
        writer
            .add_document(crate::Document::builder().add_text("body", "alpha").build())
            .unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let dictionary = || {
            inverted.segment_readers()[0]
                .read()
                .unwrap()
                .term_dictionary()
                .unwrap()
                .unwrap()
        };

        let first = dictionary();
        assert!(Arc::ptr_eq(&first, &dictionary()));
        assert!(
            inverted.segment_readers()[0]
                .read()
                .unwrap()
                .has_term_dictionary()
        );
        assert!(inverted.term_info_is_authoritative());
        assert!(inverted.terms("body").unwrap().is_some());
        assert!(Arc::ptr_eq(&first, &dictionary()));
    }

    // ---- Segment readers kept across the readers an index builds,
    // ---- Issue #1423 ---------------------------------------------------

    /// A `.delmap` rewritten under an unchanged `SegmentInfo` is read again:
    /// `deleted_count` counts only the deletions of documents the segment
    /// holds, so the metadata alone cannot say the bitmap is unchanged. The
    /// reader built before keeps the deletions of its snapshot, both share
    /// the segment's core, and a build with nothing changed keeps the reader.
    #[test]
    fn a_rewritten_delmap_under_an_unchanged_segment_info_is_read_again() {
        let (storage, info) = segment_marked_deleted("rewritten_delmap");
        let cache = SegmentReaderCache::default();
        let segment = || {
            let reader = InvertedIndexReader::with_segment_cache(
                storage.clone(),
                InvertedIndexReaderConfig::default(),
                &cache,
                || Ok(vec![info.clone()]),
            )
            .unwrap();
            Arc::clone(&reader.segment_readers()[0])
        };

        let before = segment();
        flush_delmap(&storage, &info);
        let after = segment();

        assert!(!before.read().unwrap().is_deleted(1).unwrap());
        assert!(after.read().unwrap().is_deleted(1).unwrap());
        assert!(Arc::ptr_eq(
            before.read().unwrap().core(),
            after.read().unwrap().core()
        ));
        assert!(Arc::ptr_eq(&after, &segment()));
    }

    /// A cache build keeps exactly the segments it was given, so a segment
    /// that left the snapshot is freed once no reader uses it, while a failed
    /// build keeps what the cache held.
    #[test]
    fn a_cache_build_keeps_only_the_segments_it_was_given() {
        let (a_storage, a) = docs_only_segment("cache_a", &[(0, vec![("body", text("x"))])], false);
        let (b_storage, b) = docs_only_segment("cache_b", &[(1, vec![("body", text("y"))])], false);
        let open = |info: &SegmentInfo,
                    cached: Option<&Arc<RwLock<SegmentReader>>>|
         -> Result<Arc<RwLock<SegmentReader>>> {
            if let Some(cached) = cached {
                return Ok(Arc::clone(cached));
            }
            let storage = if info.segment_id == a.segment_id {
                &a_storage
            } else {
                &b_storage
            };
            Ok(Arc::new(RwLock::new(SegmentReader::open(
                info.clone(),
                storage.clone(),
            )?)))
        };
        let cache = SegmentReaderCache::default();

        let (_, first) = cache
            .build(|| Ok(vec![a.clone(), b.clone()]), open)
            .unwrap();
        let a_reader = Arc::downgrade(&first[0]);
        let b_reader = Arc::clone(&first[1]);
        drop(first);

        let failed = cache.build(|| Err(LaurusError::index("no snapshot")), open);
        assert!(failed.is_err());
        assert_eq!(cache.len(), 2);
        assert!(a_reader.upgrade().is_some());

        let (_, second) = cache.build(|| Ok(vec![b.clone()]), open).unwrap();
        assert_eq!(cache.len(), 1);
        assert!(a_reader.upgrade().is_none());
        assert!(Arc::ptr_eq(&second[0], &b_reader));
    }

    /// `InvertedIndexReader::new` opens every segment anew, so the readers it
    /// builds share nothing; only an index's own readers share segments.
    #[test]
    fn inverted_index_reader_new_does_not_share_segment_readers() {
        let (storage, info) =
            docs_only_segment("unshared", &[(0, vec![("body", text("x"))])], false);
        let open = || {
            let reader = InvertedIndexReader::new(
                vec![info.clone()],
                storage.clone(),
                InvertedIndexReaderConfig::default(),
            )
            .unwrap();
            Arc::clone(&reader.segment_readers()[0])
        };
        let (first, second) = (open(), open());
        assert!(!Arc::ptr_eq(
            first.read().unwrap().core(),
            second.read().unwrap().core()
        ));
    }

    #[test]
    fn reader_with_a_dictionary_less_segment_is_not_authoritative() {
        use crate::lexical::query::term::TermQuery;

        let (storage, info) =
            docs_only_segment("scan_partial", &[(0, vec![("tags", text("rust"))])], false);
        let partial = InvertedIndexReader::new(
            vec![info],
            storage.clone(),
            InvertedIndexReaderConfig::default(),
        )
        .unwrap();
        assert!(!partial.term_info_is_authoritative());
        // The dictionary cannot prove emptiness here, so the query must be
        // handed to the matcher (which then takes the scan).
        assert!(!TermQuery::new("tags", "rust").is_empty(&partial).unwrap());
        assert!(
            !TermQuery::new("tags", "missing")
                .is_empty(&partial)
                .unwrap()
        );

        // No segments at all: vacuously authoritative, and an unknown term
        // is empty as before.
        let empty = InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default())
            .unwrap();
        assert!(empty.term_info_is_authoritative());
        assert!(TermQuery::new("tags", "rust").is_empty(&empty).unwrap());
    }

    // ---- Direct term lookup, Issue #1420 -----------------------------------

    /// An index with one committed segment per entry of `commits`, each
    /// listing `(doc_id, _id)` pairs, then `deletions` committed on their own.
    fn index_with_id_segments(
        commits: &[&[(u64, &str)]],
        deletions: &[u64],
    ) -> crate::lexical::index::inverted::InvertedIndex {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage, InvertedIndexConfig::default()).unwrap();
        for commit in commits {
            let mut writer = index.writer().unwrap();
            for (doc_id, id) in *commit {
                let doc = crate::Document::builder()
                    .add_text("_id", *id)
                    .add_text("body", "alpha")
                    .build();
                writer.upsert_document(*doc_id, doc).unwrap();
            }
            writer.commit().unwrap();
        }
        if !deletions.is_empty() {
            let mut writer = index.writer().unwrap();
            for doc_id in deletions {
                writer.delete_document(*doc_id).unwrap();
            }
            writer.commit().unwrap();
        }
        index
    }

    /// The doc ids an unbounded `TermQuery` search returns, sorted.
    fn searched_ids(
        index: &crate::lexical::index::inverted::InvertedIndex,
        field: &str,
        term: &str,
    ) -> Vec<u64> {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::query::term::TermQuery;
        use crate::lexical::search::searcher::LexicalSearchRequest;

        let request = LexicalSearchRequest::new(Box::new(TermQuery::new(field, term)))
            .limit(usize::MAX)
            .load_documents(false);
        let mut ids: Vec<u64> = index
            .searcher()
            .unwrap()
            .search(request)
            .unwrap()
            .hits
            .into_iter()
            .map(|hit| hit.doc_id)
            .collect();
        ids.sort_unstable();
        ids
    }

    /// What `live_doc_ids_for_term` returns over a fresh reader, sorted.
    fn looked_up_ids(
        index: &crate::lexical::index::inverted::InvertedIndex,
        field: &str,
        term: &str,
    ) -> Option<Vec<u64>> {
        use crate::lexical::index::LexicalIndex;

        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let mut ids = inverted.live_doc_ids_for_term(field, term).unwrap()?;
        ids.sort_unstable();
        Some(ids)
    }

    #[test]
    fn live_doc_ids_for_term_matches_the_term_query_search() {
        // "ida" has a copy in the first and the third segment (two chunks of
        // one id), "idb" only in the second; "idc" is in no segment.
        let index = index_with_id_segments(
            &[&[(0, "ida"), (1, "idb")], &[(2, "idb")], &[(3, "ida")]],
            &[],
        );
        for (term, expected) in [("ida", vec![0, 3]), ("idb", vec![1, 2]), ("idc", vec![])] {
            assert_eq!(
                searched_ids(&index, "_id", term),
                expected,
                "search for {term}"
            );
            assert_eq!(
                looked_up_ids(&index, "_id", term),
                Some(expected),
                "lookup for {term}"
            );
        }
        // Another field's term is not mistaken for an `_id`.
        assert_eq!(looked_up_ids(&index, "_id", "alpha"), Some(vec![]));
        assert_eq!(
            looked_up_ids(&index, "body", "alpha"),
            Some(vec![0, 1, 2, 3])
        );
    }

    #[test]
    fn live_doc_ids_for_term_skips_committed_deletions() {
        let index = index_with_id_segments(&[&[(0, "ida")], &[(1, "ida")]], &[0]);
        assert_eq!(searched_ids(&index, "_id", "ida"), vec![1]);
        assert_eq!(looked_up_ids(&index, "_id", "ida"), Some(vec![1]));
    }

    #[test]
    fn live_doc_ids_for_term_defers_to_a_search_without_every_dictionary() {
        let (storage, info) =
            docs_only_segment("lookup_no_dict", &[(0, vec![("_id", text("ida"))])], false);
        let reader =
            InvertedIndexReader::new(vec![info], storage, InvertedIndexReaderConfig::default())
                .unwrap();
        assert_eq!(reader.live_doc_ids_for_term("_id", "ida").unwrap(), None);
        assert_eq!(reader.term_info_complete.get(), Some(&false));
    }

    #[test]
    fn live_doc_ids_for_term_settles_term_info_authority() {
        use crate::lexical::index::LexicalIndex;

        let index = index_with_id_segments(&[&[(0, "ida")], &[(1, "idb")]], &[]);
        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        assert!(inverted.term_info_complete.get().is_none());
        inverted.live_doc_ids_for_term("_id", "ida").unwrap();
        // The search path's authority check now answers without reading the
        // dictionaries again.
        assert_eq!(inverted.term_info_complete.get(), Some(&true));
        assert!(inverted.term_info_is_authoritative());
    }

    #[test]
    fn live_doc_ids_for_term_fails_on_a_closed_reader() {
        use crate::lexical::reader::LexicalIndexReader;

        let index = index_with_id_segments(&[&[(0, "ida")]], &[]);
        let reader = crate::lexical::index::LexicalIndex::reader(&index).unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        // Clones share the closed flag.
        inverted.clone().close().unwrap();
        assert!(inverted.live_doc_ids_for_term("_id", "ida").is_err());
    }

    // ---- Committed `_id` filter, Issue #1420 -------------------------------

    /// `(generation, covered segments, keys inserted)` of a filter.
    fn filter_state(filter: &CommittedIdFilter) -> (u64, usize, u64) {
        let stats = filter.stats();
        (stats.generation, stats.covered_segments, stats.inserted)
    }

    #[test]
    fn id_filter_answers_an_absent_id_and_covers_only_new_segments() {
        use crate::lexical::index::LexicalIndex;

        let index = index_with_id_segments(&[&[(0, "ida"), (1, "idb")], &[(2, "idc")]], &[]);
        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();

        // An id no segment holds is answered by the filter alone: the
        // reader never collects its dictionaries for the exact lookup.
        assert_eq!(
            inverted.live_doc_ids_for_term("_id", "zzz").unwrap(),
            Some(vec![])
        );
        assert!(inverted.term_dictionaries.get().is_none());
        assert_eq!(filter_state(&index.id_filter), (1, 2, 3));

        // A held id goes on to the exact lookup.
        assert_eq!(
            inverted.live_doc_ids_for_term("_id", "idc").unwrap(),
            Some(vec![2])
        );
        assert!(inverted.term_dictionaries.get().is_some());

        // A commit adds one segment; the next reader inserts only its key.
        let mut writer = index.writer().unwrap();
        writer
            .upsert_document(3, crate::Document::builder().add_text("_id", "idd").build())
            .unwrap();
        writer.commit().unwrap();
        assert_eq!(looked_up_ids(&index, "_id", "idd"), Some(vec![3]));
        assert_eq!(looked_up_ids(&index, "_id", "zzz"), Some(vec![]));
        assert_eq!(filter_state(&index.id_filter), (1, 3, 4));
    }

    /// A rebuild keeps only the rebuilding reader's segments, so a reader
    /// that covered an earlier generation must cover its own segments again
    /// before it trusts an "absent".
    #[test]
    fn id_filter_reader_covers_again_after_another_reader_rebuilt_it() {
        use crate::lexical::index::LexicalIndex;

        // Two segments of 1,000 ids each. A filter sized for its minimum of
        // 1,024 keys takes about 1,600 before it rebuilds, so covering the
        // second segment after the first one rebuilds it.
        let first: Vec<(u64, String)> = (0..1_000).map(|i| (i, format!("a{i}"))).collect();
        let second: Vec<(u64, String)> = (1_000..2_000).map(|i| (i, format!("b{i}"))).collect();
        fn as_refs(ids: &[(u64, String)]) -> Vec<(u64, &str)> {
            ids.iter()
                .map(|(doc_id, id)| (*doc_id, id.as_str()))
                .collect()
        }
        let index = index_with_id_segments(&[&as_refs(&first), &as_refs(&second)], &[]);
        let full = index.reader().unwrap();
        let infos = full
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap()
            .segment_infos
            .clone();
        assert_eq!(infos.len(), 2);

        // Two readers over one segment each, sharing a fresh filter.
        let filter = Arc::new(CommittedIdFilter::new());
        let reader_over = |info: &SegmentInfo| {
            InvertedIndexReader::new(
                vec![info.clone()],
                index.storage().clone(),
                InvertedIndexReaderConfig::default(),
            )
            .unwrap()
            .with_id_filter(Arc::clone(&filter))
        };
        let reader_a = reader_over(&infos[0]);
        let reader_b = reader_over(&infos[1]);

        assert_eq!(
            reader_a.live_doc_ids_for_term("_id", "a0").unwrap(),
            Some(vec![0])
        );
        assert_eq!(filter_state(&filter), (1, 1, 1_000));
        assert_eq!(
            reader_b.live_doc_ids_for_term("_id", "b1000").unwrap(),
            Some(vec![1_000])
        );
        let (generation, covered, _) = filter_state(&filter);
        assert_eq!(
            (generation, covered),
            (2, 1),
            "B's keys overflow it: rebuilt from B alone"
        );

        for (doc_id, id) in &first {
            assert_eq!(
                reader_a.live_doc_ids_for_term("_id", id).unwrap(),
                Some(vec![*doc_id]),
                "{id} must still resolve through the reader that covered generation 1"
            );
        }
        for (doc_id, id) in &second {
            assert_eq!(
                reader_b.live_doc_ids_for_term("_id", id).unwrap(),
                Some(vec![*doc_id]),
                "{id}"
            );
        }
    }

    #[test]
    fn id_filter_is_not_used_when_a_segment_has_no_dictionary() {
        let (storage, info) =
            docs_only_segment("filter_no_dict", &[(0, vec![("_id", text("ida"))])], false);
        let filter = Arc::new(CommittedIdFilter::new());
        let reader =
            InvertedIndexReader::new(vec![info], storage, InvertedIndexReaderConfig::default())
                .unwrap()
                .with_id_filter(Arc::clone(&filter));
        // Neither a held nor an absent id may be answered from the filter.
        assert_eq!(reader.live_doc_ids_for_term("_id", "ida").unwrap(), None);
        assert_eq!(reader.live_doc_ids_for_term("_id", "zzz").unwrap(), None);
        assert_eq!(filter_state(&filter), (1, 0, 0));
    }

    /// #1047: `has_doc_values` must reflect the schema's per-field
    /// `doc_values` flag on a real, on-disk segment -- a field declared
    /// `doc_values: false` gets no column, one left at the default does.
    #[test]
    fn has_doc_values_reflects_the_schemas_doc_values_flag() {
        use crate::lexical::core::field::{FieldOption, TextOption};
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let mut fields = std::collections::HashMap::new();
        fields.insert(
            "title".to_string(),
            FieldOption::Text(TextOption::default()),
        );
        fields.insert(
            "internal_note".to_string(),
            FieldOption::Text(TextOption {
                doc_values: false,
                ..Default::default()
            }),
        );
        let config = InvertedIndexConfig {
            fields,
            ..Default::default()
        };

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage, config).unwrap();
        let mut writer = index.writer().unwrap();
        writer
            .add_document(
                crate::Document::builder()
                    .add_text("title", "hello")
                    .add_text("internal_note", "shh")
                    .build(),
            )
            .unwrap();
        writer.commit().unwrap();

        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let segment = inverted.segment_readers()[0].read().unwrap();

        assert!(
            segment.has_doc_values("title"),
            "a field without doc_values: false must get a column"
        );
        assert!(
            !segment.has_doc_values("internal_note"),
            "doc_values: false must keep the field out of the segment's \
             DocValues directory entirely"
        );
    }

    /// #541 — `SegmentReader::postings` must never yield a deleted
    /// document, on either of its paths.
    ///
    /// The segment merge relies on this: its per-posting loop no longer
    /// re-checks the deletion set, because doing so could not ever fire
    /// and cost a membership test on the merge's innermost loop. That
    /// makes this invariant load-bearing rather than incidental, so it is
    /// pinned here — a test that fails when the invariant breaks is
    /// stronger protection than a runtime check that cannot.
    ///
    /// This test covers the normal path through `filter_deleted_soa`: the
    /// fixture is written by the real writer, which always emits a `.post`
    /// part, so the `scan_documents_for_term` fallback never runs here. The
    /// fallback's half of the invariant is pinned by
    /// `scan_fallback_never_yields_a_deleted_document` on a hand-built
    /// segment without a `.post` file (Issue #1194).
    #[test]
    fn postings_never_yields_a_deleted_document() {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::maintenance::deletion::{DeletionConfig, DeletionManager};
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage.clone(), InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();

        let doc_count = 60u64;
        for _ in 0..doc_count {
            writer
                .add_document(crate::Document::builder().add_text("body", "alpha").build())
                .unwrap();
        }
        writer.commit().unwrap();

        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let info = inverted.segment_readers()[0]
            .read()
            .unwrap()
            .segment_info()
            .clone();

        // Delete every third document straight into the segment's `.delmap`.
        let manager = DeletionManager::new(
            DeletionConfig {
                enable_deletion_log: false,
                ..Default::default()
            },
            storage.clone(),
        )
        .unwrap();
        manager
            .initialize_segment(&info.segment_id, info.min_doc_id, info.max_doc_id)
            .unwrap();
        let mut deleted_ids = Vec::new();
        for doc_id in (info.min_doc_id..=info.max_doc_id).step_by(3) {
            manager
                .delete_document(&info.segment_id, doc_id, "test")
                .unwrap();
            deleted_ids.push(doc_id);
        }
        manager.flush().unwrap();
        assert!(!deleted_ids.is_empty(), "the fixture must delete something");

        // Re-open with `has_deletions` set, the state the merge sees.
        let mut info_with_deletions = info.clone();
        info_with_deletions.has_deletions = true;
        let segment = SegmentReader::open(info_with_deletions, storage).unwrap();

        let mut iter = segment
            .postings("body", "alpha")
            .unwrap()
            .expect("term must have postings");
        let mut seen = Vec::new();
        while iter.next().unwrap() {
            seen.push(iter.doc_id());
        }

        assert_eq!(
            seen.len(),
            (doc_count as usize) - deleted_ids.len(),
            "postings must yield exactly the live documents"
        );
        for id in &deleted_ids {
            assert!(
                !seen.contains(id),
                "postings yielded deleted doc {id}; the merge relies on it not doing so"
            );
        }
    }

    /// #553 — the production decoder selector, end to end.
    ///
    /// `SegmentReader::postings` is the only place that chooses a
    /// posting decoder for a real segment, and it had **no test at all**
    /// before this change — which is how the `posting_format >= 2`
    /// ordered comparison could have survived a format bump and silently
    /// misparsed v3 payloads.
    ///
    /// This drives the whole production path: `InvertedIndexWriter`
    /// writes a real segment (v3 postings + a v3 dictionary), and the
    /// reader dispatches on the dictionary version to decode it back.
    /// The unit tests around `encode_v3` / `decode_soa_v3` never touch
    /// this wiring.
    /// A posting list cannot hold more postings than its segment has
    /// documents. The reader caps the decode by the segment's document
    /// count — deleted documents included — so a list longer than that is
    /// refused as corruption before its buffers are sized (Issue #1220).
    #[test]
    fn postings_are_capped_by_the_segments_document_count() {
        let (storage, info) = segment_with_three_alphas();
        let segment = SegmentReader::open(info.clone(), storage.clone()).unwrap();
        assert!(
            segment.postings("body", "alpha").unwrap().is_some(),
            "the segment as written decodes"
        );
        // Loaded against the understated count, the dictionary would be
        // refused first (Issue #1235). The one loaded against the true count
        // lets the decode reach the posting-list bound.
        let dictionary = segment
            .term_dictionary()
            .unwrap()
            .expect("the segment has a term dictionary");

        let understated = SegmentInfo {
            doc_count: 1,
            ..info
        };
        let segment = SegmentReader::open(understated, storage).unwrap();
        segment.set_term_dictionary(dictionary);
        match segment.postings("body", "alpha") {
            Err(LaurusError::Index(msg)) => {
                assert!(
                    msg.contains(
                        "posting list: header declares 3 postings but the segment holds only 1 \
                         documents"
                    ),
                    "{msg}"
                );
                assert!(
                    msg.contains("body:alpha"),
                    "the error names the term: {msg}"
                );
            }
            Err(other) => panic!("expected Index error, got {other:?}"),
            Ok(_) => panic!("a list longer than its segment must be rejected"),
        }
    }

    /// Writes one segment of three documents, each holding `body:alpha`,
    /// and returns its storage and metadata.
    fn segment_with_three_alphas() -> (Arc<dyn crate::storage::Storage>, SegmentInfo) {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage.clone(), InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();
        for _ in 0..3 {
            writer
                .add_document(crate::Document::builder().add_text("body", "alpha").build())
                .unwrap();
        }
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let info = inverted.segment_readers()[0].read().unwrap().info.clone();
        (storage, info)
    }

    /// A term cannot occur in more documents than its segment holds, so a
    /// dictionary claiming so is refused when the segment loads it, naming
    /// the term (Issue #1235).
    #[test]
    fn a_dictionary_term_in_more_documents_than_its_segment_holds_is_rejected() {
        let (storage, info) = segment_with_three_alphas();
        let understated = SegmentInfo {
            doc_count: 1,
            ..info
        };
        let segment = SegmentReader::open(understated, storage).unwrap();
        match segment.term_info("body", "alpha") {
            Err(LaurusError::Index(msg)) => {
                assert!(
                    msg.contains(
                        "term dictionary: term \"body:alpha\" declares 3 documents but the \
                         segment holds only 1 documents — segment is corrupted"
                    ),
                    "{msg}"
                );
            }
            Err(other) => panic!("expected Index error, got {other:?}"),
            Ok(_) => panic!("a term in more documents than its segment must be rejected"),
        }
    }

    /// A term's statistics summed across segments saturate rather than
    /// overflow on the huge per-segment values a corrupt dictionary can
    /// carry (Issue #1224).
    #[test]
    fn cross_segment_term_statistics_saturate_at_u64_max() {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::lexical::index::structures::dictionary::{TermDictionaryBuilder, TermInfo};
        use crate::lexical::reader::LexicalIndexReader;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage.clone(), InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();
        for _ in 0..2 {
            writer
                .add_document(crate::Document::builder().add_text("body", "alpha").build())
                .unwrap();
            writer.commit().unwrap();
        }
        // A reader with segment readers of its own: the index shares its
        // segment readers with the readers it builds later (Issue #1423), and
        // the dictionaries injected below must not reach them.
        let inverted = InvertedIndexReader::new(
            index.load_segments().unwrap(),
            storage,
            InvertedIndexReaderConfig::default(),
        )
        .unwrap();
        assert_eq!(inverted.segment_readers().len(), 2);

        // Both segments' dictionaries claim 2^63 documents and occurrences.
        let mut builder = TermDictionaryBuilder::new();
        builder.add_term(
            "body:alpha".to_string(),
            TermInfo::new(0, 0, 1 << 63, 1 << 63),
        );
        let dict = Arc::new(builder.build().unwrap());
        for segment in inverted.segment_readers() {
            segment.read().unwrap().set_term_dictionary(dict.clone());
        }

        let info = inverted.term_info("body", "alpha").unwrap().unwrap();
        assert_eq!((info.doc_freq, info.total_freq), (u64::MAX, u64::MAX));
    }

    // ---- A decoded posting list checked against its dictionary entry and
    // ---- its segment, Issue #1235 ---------------------------------------

    /// Writes one segment of 200 documents, all holding `body:alpha` and
    /// every second one `body:beta`, in the given layout. `alpha` spans two
    /// posting blocks (128 + 72), so its block-max array has two entries.
    /// Returns the storage, the segment's metadata and its term dictionary as
    /// the segment loads it.
    fn segment_with_alphas_and_betas(
        use_compound: bool,
    ) -> (
        Arc<dyn crate::storage::Storage>,
        SegmentInfo,
        Arc<BlockTermDictionary>,
    ) {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexConfig {
            use_compound,
            ..Default::default()
        };
        let index = InvertedIndex::create(storage.clone(), config).unwrap();
        let mut writer = index.writer().unwrap();
        for i in 0..200 {
            let body = if i % 2 == 0 { "alpha beta" } else { "alpha" };
            writer
                .add_document(crate::Document::builder().add_text("body", body).build())
                .unwrap();
        }
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let info = inverted.segment_readers()[0].read().unwrap().info.clone();
        assert_eq!(
            (info.doc_count, info.max_doc_id - info.min_doc_id),
            (200, 199),
            "the fixture's ids must be one contiguous range"
        );
        let dictionary = SegmentReader::open(info.clone(), storage.clone())
            .unwrap()
            .term_dictionary()
            .unwrap()
            .expect("the segment has a term dictionary");
        assert_eq!(
            dictionary.get("body:alpha").unwrap().block_max.len(),
            2,
            "the writer records one block-max entry per posting block"
        );
        (storage, info, dictionary)
    }

    /// `dictionary` rebuilt with `body:alpha`'s entry passed through `edit`.
    fn with_alpha_entry(
        dictionary: &BlockTermDictionary,
        edit: impl FnOnce(&mut TermInfo),
    ) -> Arc<BlockTermDictionary> {
        use crate::lexical::index::structures::dictionary::TermDictionaryBuilder;

        let mut edit = Some(edit);
        let mut builder = TermDictionaryBuilder::new();
        for (key, info) in dictionary.iter() {
            let mut info = info.clone();
            if key == "body:alpha" {
                (edit.take().expect("one entry per key"))(&mut info);
            }
            builder.add_term(key.to_string(), info);
        }
        assert!(edit.is_none(), "the dictionary holds body:alpha");
        Arc::new(builder.build().unwrap())
    }

    /// Opens the segment described by `info`, with `dictionary` in place of
    /// the one it would load — which also skips the load's own checks.
    fn open_with_dictionary(
        storage: &Arc<dyn crate::storage::Storage>,
        info: SegmentInfo,
        dictionary: Arc<BlockTermDictionary>,
    ) -> SegmentReader {
        let segment = SegmentReader::open(info, storage.clone()).unwrap();
        segment.set_term_dictionary(dictionary);
        segment
    }

    /// `(doc_id, term_freq)` for every posting `postings(field, term)` yields.
    fn posting_entries(segment: &SegmentReader, field: &str, term: &str) -> Vec<(u64, u64)> {
        let mut iter = segment
            .postings(field, term)
            .unwrap()
            .unwrap_or_else(|| panic!("{field}:{term} must have postings"));
        let mut entries = Vec::new();
        while iter.next().unwrap() {
            entries.push((iter.doc_id(), iter.term_freq()));
        }
        entries
    }

    /// Asserts that `body:alpha`'s postings are refused as corruption with a
    /// message naming the segment, the term and `what`.
    fn assert_alpha_rejected(segment: &SegmentReader, what: &str) {
        match segment.postings("body", "alpha") {
            Err(LaurusError::Index(msg)) => {
                assert!(msg.contains(what), "{msg}");
                assert!(msg.contains("segment is corrupted"), "{msg}");
                assert!(
                    msg.contains(&format!("segment {}: body:alpha:", segment.info.segment_id)),
                    "the error names the segment and the term: {msg}"
                );
            }
            Err(other) => panic!("expected Index error, got {other:?}"),
            Ok(_) => panic!("postings refused for `{what}` were accepted"),
        }
    }

    /// A segment the writer produced passes every check against its
    /// dictionary entry and yields the postings it was written with, in both
    /// layouts: a compound container's part reports positions relative to
    /// the part, as the loose file does (Issue #1235). A dictionary rebuilt
    /// from the loaded one passes too, so the tests below that doctor one
    /// entry of it are refused for that entry alone.
    #[test]
    fn postings_matching_their_dictionary_entry_and_segment_decode_unchanged() {
        for use_compound in [false, true] {
            let (storage, info, dictionary) = segment_with_alphas_and_betas(use_compound);
            let all: Vec<(u64, u64)> = (info.min_doc_id..=info.max_doc_id)
                .map(|id| (id, 1))
                .collect();
            let every_second: Vec<(u64, u64)> = all.iter().copied().step_by(2).collect();

            let loaded = SegmentReader::open(info.clone(), storage.clone()).unwrap();
            assert_eq!(posting_entries(&loaded, "body", "alpha"), all);
            assert_eq!(posting_entries(&loaded, "body", "beta"), every_second);

            let rebuilt =
                open_with_dictionary(&storage, info, with_alpha_entry(&dictionary, |_| {}));
            assert_eq!(posting_entries(&rebuilt, "body", "alpha"), all);
            assert_eq!(posting_entries(&rebuilt, "body", "beta"), every_second);
        }
    }

    /// A dictionary offset pointing at another term's list decodes cleanly;
    /// the list's own key gives it away (Issue #1235).
    #[test]
    fn a_dictionary_entry_leading_to_another_terms_list_is_rejected() {
        let (storage, info, dictionary) = segment_with_alphas_and_betas(true);
        let beta = dictionary.get("body:beta").unwrap().clone();
        let doctored = with_alpha_entry(&dictionary, |alpha| {
            alpha.posting_offset = beta.posting_offset;
            alpha.posting_length = beta.posting_length;
        });
        let segment = open_with_dictionary(&storage, info, doctored);
        assert_alpha_rejected(
            &segment,
            "the dictionary entry leads to the list of \"body:beta\"",
        );
    }

    /// A list must hold as many postings as its dictionary entry counts
    /// documents (Issue #1235). The doctored count stays within the segment's
    /// document count, which the dictionary load bounds.
    #[test]
    fn a_posting_count_differing_from_the_document_frequency_is_rejected() {
        let (storage, info, dictionary) = segment_with_alphas_and_betas(true);
        let doctored = with_alpha_entry(&dictionary, |alpha| alpha.doc_frequency -= 1);
        let segment = open_with_dictionary(&storage, info, doctored);
        assert_alpha_rejected(
            &segment,
            "holds 200 postings but the dictionary declares 199 documents",
        );
    }

    /// The decode must end exactly where the dictionary entry says the list
    /// does, one byte short or long included (Issue #1235).
    #[test]
    fn a_posting_list_ending_off_its_recorded_length_is_rejected() {
        let (storage, info, dictionary) = segment_with_alphas_and_betas(true);
        let alpha = dictionary.get("body:alpha").unwrap();
        let (offset, length) = (alpha.posting_offset, alpha.posting_length);
        for doctored_length in [length - 1, length + 1] {
            let doctored =
                with_alpha_entry(&dictionary, |alpha| alpha.posting_length = doctored_length);
            let segment = open_with_dictionary(&storage, info.clone(), doctored);
            assert_alpha_rejected(
                &segment,
                &format!(
                    "the decode ended at byte {}, but the dictionary places the list at byte \
                     {offset} with {doctored_length} bytes",
                    offset + length
                ),
            );
        }
    }

    /// A doc id outside the segment's `[min_doc_id, max_doc_id]` is not the
    /// segment's document (Issue #1235). The dictionary is the one loaded
    /// against the true metadata, so only the range can refuse the list.
    #[test]
    fn a_posting_outside_the_segments_doc_id_range_is_rejected() {
        let (storage, info, dictionary) = segment_with_alphas_and_betas(true);
        let (min, max) = (info.min_doc_id, info.max_doc_id);

        let raised_min = SegmentInfo {
            min_doc_id: min + 1,
            ..info.clone()
        };
        let segment = open_with_dictionary(&storage, raised_min, dictionary.clone());
        assert_alpha_rejected(
            &segment,
            &format!("doc id {min} is below the segment's min_doc_id {}", min + 1),
        );

        let lowered_max = SegmentInfo {
            max_doc_id: max - 1,
            ..info
        };
        let segment = open_with_dictionary(&storage, lowered_max, dictionary);
        assert_alpha_rejected(
            &segment,
            &format!("doc id {max} is above the segment's max_doc_id {}", max - 1),
        );
    }

    /// Each block-max entry must end where its posting block does (Issue
    /// #1235). A last entry ending one id early would make the scorer treat
    /// the term's last document as past its final block, unable to score.
    #[test]
    fn a_block_max_entry_off_its_blocks_last_doc_id_is_rejected() {
        let (storage, info, dictionary) = segment_with_alphas_and_betas(true);
        let max = info.max_doc_id;
        let doctored = with_alpha_entry(&dictionary, |alpha| {
            alpha.block_max.last_mut().unwrap().last_doc_id -= 1;
        });
        let segment = open_with_dictionary(&storage, info, doctored);
        assert_alpha_rejected(
            &segment,
            &format!(
                "block-max entry 1 ends at doc id {}, but block 1 of the list ends at {max}",
                max - 1
            ),
        );
    }

    /// A block-max array must have one entry per posting block (Issue
    /// #1235): one cut short leaves the last block's documents past the
    /// array, where the scorer skips them.
    #[test]
    fn a_block_max_array_of_the_wrong_length_is_rejected() {
        use crate::lexical::index::structures::dictionary::BlockMax;

        let (storage, info, dictionary) = segment_with_alphas_and_betas(true);

        let truncated = with_alpha_entry(&dictionary, |alpha| {
            alpha.block_max.pop();
        });
        let segment = open_with_dictionary(&storage, info.clone(), truncated);
        assert_alpha_rejected(
            &segment,
            "the dictionary has 1 block-max entries for 200 postings, not 2",
        );

        let extended = with_alpha_entry(&dictionary, |alpha| {
            alpha.block_max.push(BlockMax {
                last_doc_id: info.max_doc_id + 1,
                max_factor: 1.0,
            });
        });
        let segment = open_with_dictionary(&storage, info, extended);
        assert_alpha_rejected(
            &segment,
            "the dictionary has 3 block-max entries for 200 postings, not 2",
        );
    }

    /// An empty block-max array means "no per-block bounds" — the scorer
    /// falls back to the term-level one — so it is accepted for a non-empty
    /// list (Issue #1235).
    #[test]
    fn an_empty_block_max_array_is_accepted() {
        let (storage, info, dictionary) = segment_with_alphas_and_betas(true);
        let all: Vec<(u64, u64)> = (info.min_doc_id..=info.max_doc_id)
            .map(|id| (id, 1))
            .collect();
        let doctored = with_alpha_entry(&dictionary, |alpha| alpha.block_max.clear());
        let segment = open_with_dictionary(&storage, info, doctored);
        assert_eq!(posting_entries(&segment, "body", "alpha"), all);
    }

    #[test]
    fn segment_reader_decodes_postings_written_by_the_writer() {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage, InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();

        // Enough documents to push the term past the bit-packed block
        // boundary, so the decode exercises full blocks plus a tail.
        let doc_count = 200u64;
        for i in 0..doc_count {
            writer
                .add_document(
                    crate::Document::builder()
                        .add_text("body", if i % 2 == 0 { "alpha beta" } else { "alpha" })
                        .build(),
                )
                .unwrap();
        }
        writer.commit().unwrap();

        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let segment = inverted.segment_readers()[0].clone();
        let segment = segment.read().unwrap();

        // The segment the writer just produced must be stamped v3, or
        // the dispatch below is not testing what it claims to.
        let dict = segment
            .term_dictionary()
            .unwrap()
            .expect("segment must have a term dictionary");
        assert_eq!(
            dict.posting_format_version(),
            3,
            "the writer must produce v3 segments"
        );

        // "alpha" is in every document; "beta" in the even ones.
        let mut iter = segment
            .postings("body", "alpha")
            .unwrap()
            .expect("alpha must have postings");
        // `doc_id()` is only valid after a `next()` that returned true,
        // so the iterator starts positioned before the first posting.
        let mut seen = Vec::new();
        while iter.next().unwrap() {
            seen.push(iter.doc_id());
        }
        assert_eq!(
            seen.len(),
            doc_count as usize,
            "every document contains 'alpha'"
        );
        assert!(
            seen.windows(2).all(|w| w[0] < w[1]),
            "doc ids must come back strictly ascending"
        );

        let mut iter = segment
            .postings("body", "beta")
            .unwrap()
            .expect("beta must have postings");
        let mut even = Vec::new();
        while iter.next().unwrap() {
            even.push(iter.doc_id());
        }
        assert_eq!(
            even.len(),
            (doc_count / 2) as usize,
            "'beta' is only in the even documents"
        );
    }

    #[test]
    fn test_advanced_posting_iterator() {
        let postings = vec![
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 1,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 3,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 5,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 7,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 9,
                frequency: 1,
                positions: Some(vec![0]),
                weight: 1.0,
            },
        ];

        let mut iter = InvertedIndexPostingIterator::with_blocks(postings, 2);

        // Test skip_to functionality
        assert!(iter.skip_to(5).unwrap());
        assert_eq!(iter.doc_id(), 5);

        // Test next
        assert!(iter.next().unwrap());
        assert_eq!(iter.doc_id(), 7);

        // Test skip past end
        assert!(!iter.skip_to(15).unwrap());
        assert_eq!(iter.doc_id(), u64::MAX);
    }

    /// #576: building an iterator from an `Arc<DecodedPostingList>` must share
    /// the backing arrays (an `Arc::clone` refcount bump), not deep-copy them —
    /// this is what removes the per-query SoA clone that dominated multi-segment
    /// BM25 search. Two iterators over the same shared list must keep
    /// independent cursors and return identical results.
    #[test]
    fn from_decoded_soa_arc_shares_backing_without_deep_clone() {
        use crate::lexical::index::inverted::core::posting::{
            DecodedPostingList, build_skip_levels,
        };

        let doc_ids = vec![2u32, 4, 6, 8, 10];
        let skip_levels = build_skip_levels(&doc_ids);
        let shared = Arc::new(DecodedPostingList {
            term: "t".to_string(),
            doc_ids: doc_ids.clone(),
            frequencies: vec![1, 1, 1, 1, 1],
            weights: Vec::new(),
            positions: None,
            skip_levels,
            total_frequency: 5,
            doc_frequency: 5,
        });
        assert_eq!(Arc::strong_count(&shared), 1);

        // Each iterator must hold an `Arc::clone`, not a deep copy.
        let mut it_a = InvertedIndexPostingIterator::from_decoded_soa_arc(Arc::clone(&shared));
        let mut it_b = InvertedIndexPostingIterator::from_decoded_soa_arc(Arc::clone(&shared));
        assert_eq!(
            Arc::strong_count(&shared),
            3,
            "both iterators must share the cached Arc (no deep clone)"
        );

        // Independent cursors: advancing one must not move the other.
        assert!(it_a.skip_to(6).unwrap());
        assert_eq!(it_a.doc_id(), 6);
        assert!(it_b.next().unwrap());
        assert_eq!(it_b.doc_id(), 2);

        // Full sweep over `it_b` matches the source doc ids.
        let mut seen = vec![it_b.doc_id()];
        while it_b.next().unwrap() {
            seen.push(it_b.doc_id());
        }
        assert_eq!(seen, vec![2, 4, 6, 8, 10]);

        // Dropping the iterators releases their Arc references.
        drop(it_a);
        drop(it_b);
        assert_eq!(Arc::strong_count(&shared), 1);
    }

    /// `skip_to` must agree with a naive linear scan across the full
    /// sweep of corpus sizes and target positions — #503 multi-level
    /// skip table must not change observable behaviour. Covers below
    /// SKIP_INTERVAL (table empty, tail-only path), exact stride
    /// boundaries, and several multi-level cases.
    #[test]
    fn test_skip_to_matches_linear_scan() {
        use crate::lexical::index::inverted::core::posting::{Posting, SKIP_INTERVAL};

        for &n in &[
            1usize,
            SKIP_INTERVAL - 1,
            SKIP_INTERVAL,
            SKIP_INTERVAL + 1,
            SKIP_INTERVAL * SKIP_INTERVAL,
            5_000,
        ] {
            // Build posting list with doc_id = i * 3 + 7 — gaps + offset
            // so equality-on-boundary cases get exercised, not just
            // contiguous ranges.
            let postings: Vec<Posting> = (0..n as u64)
                .map(|i| Posting::with_frequency(i * 3 + 7, 1))
                .collect();
            let doc_ids: Vec<u64> = postings.iter().map(|p| p.doc_id).collect();

            // Pick a handful of target doc ids: before first, exactly
            // first/last, one past every level boundary, and well past
            // end-of-list.
            let mut targets: Vec<u64> = vec![0, doc_ids[0]];
            if n > 1 {
                targets.push(doc_ids[n / 2]);
                targets.push(doc_ids[n / 2] + 1);
            }
            targets.push(doc_ids[n - 1]);
            targets.push(doc_ids[n - 1] + 1);
            targets.push(doc_ids[n - 1] + 1000);
            // Anything beyond u32::MAX must exhaust the iterator.
            targets.push(u64::from(u32::MAX) + 1);

            for &target in &targets {
                let mut iter = InvertedIndexPostingIterator::new(postings.clone());
                let got = iter.skip_to(target).unwrap();

                // Linear-scan reference: the first doc id >= target.
                let want_idx = doc_ids.iter().position(|&d| d >= target);
                match want_idx {
                    Some(idx) => {
                        assert!(got, "expected hit at target={target} n={n}");
                        assert_eq!(
                            iter.doc_id(),
                            doc_ids[idx],
                            "wrong doc_id at target={target} n={n}"
                        );
                    }
                    None => {
                        assert!(!got, "expected miss at target={target} n={n}");
                        assert_eq!(
                            iter.doc_id(),
                            u64::MAX,
                            "exhausted iter should report u64::MAX (target={target} n={n})"
                        );
                    }
                }
            }
        }
    }

    /// Repeated `skip_to` calls must advance monotonically without
    /// regressing. After `skip_to(x)` lands at index `i`, a subsequent
    /// `skip_to(y)` with `y > x` must land at index ≥ `i`. This is the
    /// invariant the BMW pivot loop and conjunction matchers rely on.
    #[test]
    fn test_skip_to_is_monotonic() {
        use crate::lexical::index::inverted::core::posting::Posting;

        let n: usize = 2_048;
        let postings: Vec<Posting> = (0..n as u64)
            .map(|i| Posting::with_frequency(i * 2 + 1, 1))
            .collect();
        let mut iter = InvertedIndexPostingIterator::new(postings);
        let mut prev_doc: u64 = 0;
        for target in (50..n as u64 * 2).step_by(101) {
            assert!(iter.skip_to(target).unwrap(), "target={target}");
            let current = iter.doc_id();
            assert!(
                current >= prev_doc,
                "regressed: prev={prev_doc} current={current} target={target}"
            );
            assert!(
                current >= target,
                "landed before target: current={current} target={target}"
            );
            prev_doc = current;
        }
    }

    #[test]
    fn test_cache_manager() {
        let cache = CacheManager::new(1024);
        let key = "field:term".to_string();
        let term_info = TermInfo::new(100, 50, 5, 10);

        // Test cache miss
        assert!(cache.get_term_info(&key).is_none());

        // Test cache insertion and hit
        cache.cache_term_info(key.clone(), term_info.clone());
        let cached = cache.get_term_info(&key).unwrap();
        assert_eq!(cached.doc_frequency, term_info.doc_frequency);

        // Test cache statistics
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(stats.hit_ratio() > 0.0);
    }

    /// The term cache must evict the least-recently-used entry — not a random
    /// one — when it reaches capacity (Issue #593).
    #[test]
    fn test_cache_manager_lru_eviction() {
        // `memory_limit / EST_TERM_ENTRY_BYTES` = 128 / 64 = 2 entries.
        let cache = CacheManager::new(2 * EST_TERM_ENTRY_BYTES);

        cache.cache_term_info("a".to_string(), TermInfo::new(1, 1, 1, 1));
        cache.cache_term_info("b".to_string(), TermInfo::new(2, 2, 2, 2));

        // Touch "a" so "b" becomes the least-recently-used entry.
        assert!(cache.get_term_info("a").is_some());

        // Inserting "c" must evict "b" (the LRU victim), keeping "a" and "c".
        cache.cache_term_info("c".to_string(), TermInfo::new(3, 3, 3, 3));

        assert!(
            cache.get_term_info("a").is_some(),
            "recently-used 'a' survives"
        );
        assert!(
            cache.get_term_info("c").is_some(),
            "just-inserted 'c' survives"
        );
        assert!(
            cache.get_term_info("b").is_none(),
            "least-recently-used 'b' must be evicted, not a random entry"
        );
    }

    /// The term cache reserves nothing for its budget (Issue #1431): a
    /// `usize::MAX` budget, whose up-front table could never be allocated,
    /// still builds a cache that holds entries.
    #[test]
    fn test_cache_manager_does_not_reserve_its_budget() {
        let cache = CacheManager::new(usize::MAX);
        cache.cache_term_info("a".to_string(), TermInfo::new(1, 1, 7, 1));

        assert_eq!(cache.get_term_info("a").unwrap().doc_frequency, 7);
        assert_eq!(cache.stats().memory_usage, EST_TERM_ENTRY_BYTES);
    }

    /// A repeated `postings(field, term)` within a snapshot is served from the
    /// per-segment posting cache (Issue #612), and a commit (new snapshot) does
    /// not serve a stale, pre-deletion list.
    #[test]
    fn posting_cache_hit_and_snapshot_invalidation() {
        use crate::Document;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        for id in 0..5u64 {
            store
                .upsert_document(
                    id,
                    Document::builder().add_text("body", "shared term").build(),
                )
                .unwrap();
        }
        store.commit().unwrap();

        let drain = |it: Option<Box<dyn crate::lexical::reader::PostingIterator>>| -> Vec<u64> {
            let mut ids = Vec::new();
            if let Some(mut it) = it {
                while it.next().unwrap() {
                    ids.push(it.doc_id());
                }
            }
            ids.sort_unstable();
            ids
        };

        // First snapshot: the second `postings` call is a cache hit.
        {
            let reader = store.reader_for_tests().unwrap();
            let inverted = reader
                .as_any()
                .downcast_ref::<InvertedIndexReader>()
                .unwrap();
            let seg = inverted.segment_readers()[0].read().unwrap();

            let first = drain(seg.postings("body", "shared").unwrap());
            let second = drain(seg.postings("body", "shared").unwrap());
            assert_eq!(first, vec![0, 1, 2, 3, 4]);
            assert_eq!(first, second, "cached postings must match the decoded list");

            let stats = seg.posting_cache_stats();
            assert_eq!(stats.misses, 1, "the first decode is a cache miss");
            assert!(stats.hits >= 1, "the repeat lookup hits the cache");
        }

        // Delete a doc + commit: the fresh snapshot must exclude it (a new
        // segment reader with an empty cache re-decodes against the new
        // deletions — no stale cached list).
        store.delete_document_by_internal_id(2).unwrap();
        store.commit().unwrap();
        let reader2 = store.reader_for_tests().unwrap();
        let after = drain(reader2.postings("body", "shared").unwrap());
        assert_eq!(
            after,
            vec![0, 1, 3, 4],
            "deleted doc 2 must be excluded in the new snapshot"
        );
    }

    #[test]
    fn test_segment_info() {
        let info = SegmentInfo {
            segment_id: "seg_000001".to_string(),
            doc_count: 1000,
            min_doc_id: 0,
            max_doc_id: 999,
            generation: 1,
            has_deletions: false,
            deleted_count: None,
            shard_id: 0,
        };

        assert_eq!(info.segment_id, "seg_000001");
        assert_eq!(info.doc_count, 1000);
        assert_eq!(info.min_doc_id, 0);
        assert_eq!(info.max_doc_id, 999);
        assert!(!info.has_deletions);
    }

    /// Writes `{segment_id}.{suffix}` in the layout v0.13.2 and earlier
    /// wrote before `.norms` (#555): `lens` holds per-document field lengths,
    /// `fstats` per-field statistics, here for document 0 with a `body` of
    /// length 7.
    fn write_pre_norms_part(output: impl crate::storage::StorageOutput, suffix: &str) {
        use crate::storage::structured::StructWriter;

        let mut w = StructWriter::new(output);
        match suffix {
            "lens" => {
                w.write_varint(1).unwrap(); // doc_count
                w.write_u64(0).unwrap(); // doc_id
                w.write_varint(1).unwrap(); // field_count
                w.write_string("body").unwrap();
                w.write_u32(7).unwrap(); // length
            }
            "fstats" => {
                w.write_varint(1).unwrap(); // field_count
                w.write_string("body").unwrap();
                w.write_u64(1).unwrap(); // doc_count
                w.write_f64(7.0).unwrap(); // avg_length
                w.write_u64(7).unwrap(); // min_length
                w.write_u64(7).unwrap(); // max_length
            }
            other => panic!("not a pre-.norms part: {other}"),
        }
        w.close().unwrap();
    }

    /// Metadata for a one-document segment holding doc id 0.
    fn one_doc_segment_info(segment_id: &str) -> SegmentInfo {
        SegmentInfo {
            segment_id: segment_id.to_string(),
            doc_count: 1,
            min_doc_id: 0,
            max_doc_id: 0,
            generation: 0,
            has_deletions: false,
            deleted_count: None,
            shard_id: 0,
        }
    }

    /// #1121: a segment from v0.13.2 or earlier -- `.lens` and/or `.fstats`,
    /// no `.norms` -- is refused with the rebuild error by every read of its
    /// lengths, loose or compound, instead of being read as recording none.
    #[test]
    fn pre_norms_segment_is_refused_with_a_rebuild_error() {
        use crate::lexical::index::inverted::compound::CompoundSegmentWriter;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let segment_id = "pre_norms_seg";
        for parts in [&["lens"][..], &["fstats"], &["lens", "fstats"]] {
            for use_compound in [false, true] {
                let storage: Arc<dyn crate::storage::Storage> =
                    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
                if use_compound {
                    let mut writer =
                        CompoundSegmentWriter::create(storage.as_ref(), segment_id).unwrap();
                    for suffix in parts {
                        write_pre_norms_part(writer.begin_part(suffix).unwrap(), suffix);
                        writer.end_part().unwrap();
                    }
                    writer.finish().unwrap();
                } else {
                    for suffix in parts {
                        let output = storage
                            .create_output(&format!("{segment_id}.{suffix}"))
                            .unwrap();
                        write_pre_norms_part(output, suffix);
                    }
                }

                let reader =
                    SegmentReader::open(one_doc_segment_info(segment_id), storage).unwrap();
                let errors = [
                    reader.field_stats("body").map(drop),
                    reader.field_length(0, "body").map(drop),
                    reader.norms_field_names().map(drop),
                ];
                for result in errors {
                    let context = format!("parts {parts:?}, compound {use_compound}");
                    let Err(error) = result else {
                        panic!("{context}: the read succeeded");
                    };
                    let message = error.to_string();
                    assert!(
                        message.contains("rebuild the index"),
                        "{context}: {message}"
                    );
                }
            }
        }
    }

    /// A segment with no `.norms` and no `.lens`/`.fstats` opens with empty
    /// norms, as a segment missing any other part still opens.
    #[test]
    fn segment_without_any_length_part_records_no_lengths() {
        use crate::lexical::reader::FieldLengths;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let reader = SegmentReader::open(one_doc_segment_info("bare_seg"), storage).unwrap();

        assert!(reader.field_stats("body").unwrap().is_none());
        assert_eq!(reader.field_length(0, "body").unwrap(), None);
        assert!(reader.norms_field_names().unwrap().is_empty());
        let segment = Arc::new(RwLock::new(reader));
        assert_eq!(
            SegmentFieldLengths::new("body", [(0, 0, &segment)]).get(0),
            None
        );
    }

    /// #1121, through the index: over an index holding a segment from v0.13.2
    /// or earlier, term and phrase queries fail with the rebuild error rather
    /// than scoring with a default length, and so does merging it. The same
    /// segment without any length part still answers the queries and merges.
    #[test]
    fn search_and_merge_over_a_pre_norms_segment_fail_with_a_rebuild_error() {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::lexical::query::graph_phrase::{GraphPhraseQuery, PhraseArc};
        use crate::lexical::query::phrase::PhraseQuery;
        use crate::lexical::query::term::TermQuery;
        use crate::lexical::search::searcher::LexicalSearchRequest;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        for pre_norms in [false, true] {
            let storage: Arc<dyn crate::storage::Storage> =
                Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            let config = InvertedIndexConfig {
                use_compound: false,
                ..Default::default()
            };
            let segment_ids: Vec<String> = {
                let index = InvertedIndex::create(storage.clone(), config.clone()).unwrap();
                for body in ["alpha", "alpha beta"] {
                    let mut writer = index.writer().unwrap();
                    writer
                        .add_document(crate::Document::builder().add_text("body", body).build())
                        .unwrap();
                    writer.commit().unwrap();
                }
                let reader = index.reader().unwrap();
                let inverted = reader
                    .as_any()
                    .downcast_ref::<InvertedIndexReader>()
                    .unwrap();
                let ids: Vec<String> = inverted
                    .segment_readers()
                    .iter()
                    .map(|segment| segment.read().unwrap().info.segment_id.clone())
                    .collect();
                assert_eq!(ids.len(), 2, "the fixture must hold two segments");
                ids
            };

            // Turn one segment into what v0.13.2 left on disk.
            let segment_id = &segment_ids[0];
            storage.delete_file(&format!("{segment_id}.norms")).unwrap();
            if pre_norms {
                for suffix in ["lens", "fstats"] {
                    let output = storage
                        .create_output(&format!("{segment_id}.{suffix}"))
                        .unwrap();
                    write_pre_norms_part(output, suffix);
                }
            }

            let index = InvertedIndex::open(storage, config).unwrap();
            let search = |query: Box<dyn Query>| {
                let request = LexicalSearchRequest::new(query).load_documents(false);
                index.searcher().unwrap().search(request)
            };
            // Each query scores through its own path: BM25 over per-document
            // lengths, or a phrase score over the field's average length.
            let term = || Box::new(TermQuery::new("body", "alpha")) as Box<dyn Query>;
            let phrase = || {
                Box::new(PhraseQuery::new(
                    "body",
                    vec!["alpha".to_string(), "beta".to_string()],
                )) as Box<dyn Query>
            };
            let graph_phrase = || {
                let arcs = vec![
                    PhraseArc::new(0, 1, vec!["alpha".to_string()]),
                    PhraseArc::new(1, 2, vec!["beta".to_string(), "delta".to_string()]),
                ];
                Box::new(GraphPhraseQuery::from_arcs("body", arcs)) as Box<dyn Query>
            };
            if pre_norms {
                let results = [
                    ("term search", search(term()).map(drop)),
                    ("phrase search", search(phrase()).map(drop)),
                    ("graph phrase search", search(graph_phrase()).map(drop)),
                    ("optimize", index.optimize()),
                ];
                for (operation, result) in results {
                    let Err(error) = result else {
                        panic!("{operation} succeeded");
                    };
                    let message = error.to_string();
                    assert!(
                        message.contains("rebuild the index"),
                        "{operation}: {message}"
                    );
                }
            } else {
                let hits = |query| search(query).unwrap().hits.len();
                assert_eq!(
                    (hits(term()), hits(phrase()), hits(graph_phrase())),
                    (2, 1, 1)
                );
                index.optimize().unwrap();
                assert_eq!(
                    (hits(term()), hits(phrase()), hits(graph_phrase())),
                    (2, 1, 1)
                );
            }
        }
    }

    /// #1298: `SegmentFieldLengths` resolves each segment once, then answers
    /// every lookup as the per-document path does -- across segments, for a
    /// re-added id whose deleted old copy sits in an earlier segment, and for
    /// a field only some documents have.
    #[test]
    fn field_lengths_match_each_segments_own_lengths() {
        use crate::Document;
        use crate::lexical::reader::{FieldLengths, LexicalIndexReader};
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        let words = |n: u64| vec!["word"; n as usize].join(" ");
        // Lengths stay below 40, which `.norms` stores exactly.
        for id in 0..30u64 {
            let mut doc = Document::builder().add_text("body", words(1 + id % 25));
            if id % 3 == 0 {
                doc = doc.add_text("title", words(2 + id % 5));
            }
            store.upsert_document(id, doc.build()).unwrap();
            if id % 10 == 9 {
                store.commit().unwrap();
            }
        }
        store
            .upsert_document(5, Document::builder().add_text("body", words(33)).build())
            .unwrap();
        store.commit().unwrap();

        let reader = store.reader_for_tests().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let segments = inverted.segment_readers();
        assert!(segments.len() >= 2);
        assert!(
            segments
                .iter()
                .any(|segment| segment.read().unwrap().is_deleted(5).unwrap()),
            "the fixture must keep a deleted copy of id 5"
        );

        for field in ["body", "title", "missing"] {
            let lengths = inverted.field_lengths(field).unwrap();
            for id in 0..32u64 {
                assert_eq!(
                    lengths.get(id),
                    inverted.field_length(id, field).unwrap(),
                    "{field}: doc {id}"
                );
            }
            // One segment on its own, as `PerSegmentReaderView` builds it.
            for segment in segments {
                let lengths = SegmentFieldLengths::new(field, [(0, u64::MAX, segment)]);
                let segment = segment.read().unwrap();
                for id in 0..32u64 {
                    assert_eq!(
                        lengths.get(id),
                        segment.field_length(id, field).unwrap(),
                        "{field}: doc {id} in one segment"
                    );
                }
            }
        }

        let body = inverted.field_lengths("body").unwrap();
        assert_eq!(body.get(5), Some(33), "the live copy of a re-added id");
        assert_eq!(body.get(6), Some(7));
        let title = inverted.field_lengths("title").unwrap();
        assert_eq!(title.get(3), Some(5));
        assert_eq!(title.get(4), None);
    }

    /// #1301: a segment whose id range is too wide for a deletion bitset
    /// answers deletion checks, the live count and field lengths like a
    /// segment with one.
    #[test]
    fn a_segment_without_a_deletion_bitset_answers_like_one_with_it() {
        use crate::Document;
        use crate::lexical::reader::LexicalIndexReader;
        use crate::lexical::store::LexicalStore;
        use crate::lexical::store::config::LexicalIndexConfig;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let store = LexicalStore::new(storage, LexicalIndexConfig::default()).unwrap();
        let words = |n: u64| vec!["word"; n as usize].join(" ");
        let upsert = |id: u64| {
            store
                .upsert_document(
                    id,
                    Document::builder()
                        .add_text("body", words(1 + id % 7))
                        .build(),
                )
                .unwrap()
        };
        // A segment without gaps: [200, 209].
        (200..210).for_each(upsert);
        store.commit().unwrap();
        // Two documents in [5, 100]: 96 ids is wider than their 16 bits.
        [5, 100].into_iter().for_each(upsert);
        store.commit().unwrap();
        store.delete_document_by_internal_id(100).unwrap();
        store.delete_document_by_internal_id(203).unwrap();
        store.commit().unwrap();

        let reader = store.reader_for_tests().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let bitsets: Vec<(u64, bool)> = inverted
            .segment_readers()
            .iter()
            .map(|segment| {
                let segment = segment.read().unwrap();
                let deleted = segment.deletions().expect("both segments carry a deletion");
                (segment.segment_info().min_doc_id, deleted.has_bits())
            })
            .collect();
        assert_eq!(bitsets, vec![(200, true), (5, false)]);

        let lengths = inverted.field_lengths("body").unwrap();
        for id in 0..=210u64 {
            let held = id == 5 || (200..210).contains(&id);
            let deleted = id == 100 || id == 203;
            assert_eq!(
                LexicalIndexReader::is_deleted(inverted, id),
                deleted,
                "doc {id}"
            );
            let expected = (held && !deleted).then(|| (1 + id % 7) as u32);
            assert_eq!(lengths.get(id), expected, "length of doc {id}");
        }
        assert_eq!(LexicalIndexReader::doc_count(inverted), 10);
    }

    /// `.norms` must win even when stale `.lens`/`.fstats` are also
    /// present (e.g. leftover from a partially-completed migration) --
    /// [`SegmentCore::load_norms`] must not refuse the segment as
    /// pre-`.norms` just because those files happen to exist.
    #[test]
    fn norms_file_takes_precedence_over_stale_legacy_files() {
        use crate::lexical::core::analyzed::BufferedDocument;
        use crate::lexical::index::structures::norms::NormsBuilder;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
        use crate::storage::structured::StructWriter;

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let segment_id = "mixed_seg";

        // Stale legacy files claiming length 7.
        for suffix in ["lens", "fstats"] {
            let output = storage
                .create_output(&format!("{segment_id}.{suffix}"))
                .unwrap();
            write_pre_norms_part(output, suffix);
        }

        // The current .norms file claiming length 25 -- this must win.
        {
            let mut doc = BufferedDocument::default();
            doc.field_lengths.insert("body".to_string(), 25); // within the exact window (< 40)
            let norms = NormsBuilder::from_buffered(&[(0u64, doc)]);
            let output = storage
                .create_output(&format!("{segment_id}.norms"))
                .unwrap();
            let mut w = StructWriter::new(output);
            norms.write_to(&mut w).unwrap();
            w.close().unwrap();
        }

        let reader = SegmentReader::open(one_doc_segment_info(segment_id), storage).unwrap();

        assert_eq!(reader.field_length(0, "body").unwrap(), Some(25));
        assert_eq!(reader.field_stats("body").unwrap().unwrap().max_length, 25);
    }

    /// #555 Phase 3/4 end-to-end: the BM25 score-bound precomputed at
    /// flush time (`TermInfo::max_score_factor`/`block_max`) must remain a
    /// true upper bound once the reader substitutes back the *quantised*
    /// length via `.norms`. This is the invariant Block-Max-WAND's pruning
    /// correctness depends on; if it is violated a real match can be
    /// silently skipped.
    ///
    /// Each fixture document gets its own unique term (not shared with any
    /// other document), so each term's `max_score_factor`/`block_max` is
    /// derived **solely from that one document's length** -- a shared term
    /// would let a short, unquantised document's larger factor mask a long
    /// document's quantisation error, defeating the test.
    #[test]
    fn block_max_bound_is_never_violated_after_norms_quantisation() {
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::{InvertedIndex, InvertedIndexConfig};
        use crate::lexical::query::scorer::{BM25Scorer, Scorer};
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

        let storage: Arc<dyn crate::storage::Storage> =
            Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage, InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();

        // (unique term, term frequency, filler-word count). Field length =
        // tf + filler. Spans well below and well above
        // EXACT_LENGTH_BOUND (40) so quantisation actually applies to some.
        let fixtures: Vec<(&str, u32, u32)> = vec![
            ("shortterm", 2, 3),         // length 5, exact
            ("boundaryterm", 1, 38),     // length 39, exact (boundary)
            ("overterm", 5, 50),         // length 55, just above the window
            ("longterm", 1, 999),        // length 1_000
            ("verylongterm", 20, 4_980), // length 5_000
        ];
        for &(term, tf, filler) in &fixtures {
            let mut body = format!("{term} ").repeat(tf as usize);
            body.push_str(&"filler ".repeat(filler as usize));
            writer
                .add_document(crate::Document::builder().add_text("body", body).build())
                .unwrap();
        }
        writer.commit().unwrap();

        let reader = index.reader().unwrap();
        let inverted = reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .unwrap();
        let segment = inverted.segment_readers()[0].read().unwrap();

        let stats = segment.field_stats("body").unwrap().unwrap();
        let total_docs = segment.segment_info().doc_count;

        for (doc_id, &(term, tf, _filler)) in fixtures.iter().enumerate() {
            let doc_id = doc_id as u64;
            let term_info = segment.term_info("body", term).unwrap().unwrap();
            let scorer = BM25Scorer::with_block_max(
                term_info.doc_frequency,
                term_info.total_frequency,
                stats.doc_count,
                stats.avg_length,
                total_docs,
                1.0,
                term_info.max_score_factor,
                term_info.block_max.into(),
            );

            let field_length = segment.field_length(doc_id, "body").unwrap().unwrap();
            let score = scorer.score(doc_id, tf as f32, Some(field_length as f32));

            assert!(
                score <= scorer.max_score() + 1e-4,
                "doc {doc_id} ({term}): score {score} exceeds the term-level bound {}",
                scorer.max_score()
            );
            assert!(
                score <= scorer.block_max_score_at(doc_id) + 1e-4,
                "doc {doc_id} ({term}): score {score} exceeds the block-max bound {}",
                scorer.block_max_score_at(doc_id)
            );
        }
    }
}
