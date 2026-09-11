//! Inverted index reader implementation.
//!
//! This module provides a production-ready inverted index reader that efficiently
//! handles multiple segments, caching, and optimized posting list access.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use ahash::AHashMap;
use lru::LruCache;
use parking_lot::Mutex;
use roaring::RoaringTreemap;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::analysis::token::Token;
use crate::error::{LaurusError, Result};
use crate::lexical::core::document::Document;
use crate::lexical::core::field::FieldValue;
use crate::lexical::index::inverted::core::posting::{DecodedPostingList, Posting, PostingList};
use crate::lexical::index::inverted::core::terms::{
    InvertedIndexTerms, MergedInvertedIndexTerms, TermDictionaryAccess, Terms,
};
use crate::lexical::index::inverted::posting_cache::PostingCache;
use crate::lexical::index::inverted::query_cache::QueryFilterCache;
use crate::lexical::index::inverted::segment::SegmentInfo;
use crate::lexical::index::structures::bkd_tree::{BKDReader, BKDTree};
use crate::lexical::index::structures::dictionary::BlockTermDictionary;
use crate::lexical::index::structures::dictionary::TermInfo;
use crate::lexical::index::structures::doc_values::DocValuesReader;
use crate::lexical::query::Query;
use crate::lexical::reader::FieldStats;
use crate::lexical::reader::PostingIterator;
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::storage::structured::StructReader;

/// Advanced index reader configuration.
#[derive(Clone)]
pub struct InvertedIndexReaderConfig {
    /// Maximum memory for caching (in bytes).
    pub max_cache_memory: usize,

    /// Enable term caching.
    pub enable_term_cache: bool,

    /// Enable posting cache.
    pub enable_posting_cache: bool,

    /// Preload segments on open.
    pub preload_segments: bool,

    /// Maximum number of cached terms per field.
    pub max_cached_terms_per_field: usize,

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
            .field("enable_term_cache", &self.enable_term_cache)
            .field("enable_posting_cache", &self.enable_posting_cache)
            .field("preload_segments", &self.preload_segments)
            .field(
                "max_cached_terms_per_field",
                &self.max_cached_terms_per_field,
            )
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
            max_cache_memory: 128 * 1024 * 1024, // 128MB
            enable_term_cache: true,
            enable_posting_cache: true,
            preload_segments: false,
            max_cached_terms_per_field: 10000,
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

/// Reader for a single segment (schema-less mode).
#[derive(Debug)]
pub struct SegmentReader {
    /// Segment information.
    info: SegmentInfo,

    /// Storage backend.
    storage: Arc<dyn Storage>,

    /// The compound-container facade when this segment is a `.cfs`
    /// (#554); `None` for loose segments. Kept typed alongside the
    /// type-erased `storage` clone so part enumeration (`bkd_field_names`)
    /// can consult the table.
    compound: Option<Arc<crate::lexical::index::inverted::compound::CompoundSegmentStorage>>,

    /// Term dictionary for efficient term lookup.
    term_dictionary: RwLock<Option<Arc<BlockTermDictionary>>>,

    /// Cached stored documents.
    stored_documents: RwLock<Option<BTreeMap<u64, Document>>>,

    /// Cached field lengths: doc_id -> (field_name -> length).
    field_lengths: RwLock<Option<BTreeMap<u64, AHashMap<String, u32>>>>,

    /// Cached field statistics: field_name -> FieldStats.
    field_stats: RwLock<Option<AHashMap<String, crate::lexical::reader::FieldStats>>>,

    /// DocValues reader for this segment.
    doc_values: RwLock<Option<Arc<DocValuesReader>>>,

    /// Optional deletion bitmap for this segment.
    deletion_bitmap: RwLock<Option<Arc<DeletionBitmap>>>,

    /// Cached BKD trees: field -> tree
    bkd_trees: RwLock<AHashMap<String, Arc<dyn BKDTree>>>,

    /// Decoded posting-list cache (Issue #612). Disabled by default
    /// (`open` builds it with a zero budget); query readers enable it via
    /// [`Self::with_posting_cache_bytes`]. Per-segment because a segment is
    /// immutable for a reader snapshot.
    posting_cache: PostingCache,

    /// Whether the segment is loaded.
    loaded: AtomicBool,
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
    pub fn open(info: SegmentInfo, storage: Arc<dyn Storage>) -> Result<Self> {
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
        let reader = SegmentReader {
            info,
            storage,
            compound,
            term_dictionary: RwLock::new(None),
            stored_documents: RwLock::new(None),
            field_lengths: RwLock::new(None),
            field_stats: RwLock::new(None),
            doc_values: RwLock::new(None),
            deletion_bitmap: RwLock::new(None),
            bkd_trees: RwLock::new(AHashMap::new()),
            // Disabled by default; query readers enable it (Issue #612).
            posting_cache: PostingCache::new(0),
            loaded: AtomicBool::new(false),
        };

        Ok(reader)
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
        if let Some(facade) = &self.compound {
            return Ok(facade.bkd_field_names());
        }
        let prefix = format!("{}.", self.info.segment_id);
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

    /// Snapshot of this segment's posting-cache hit / miss counters (Issue #612).
    pub fn posting_cache_stats(
        &self,
    ) -> crate::lexical::index::inverted::posting_cache::PostingCacheStats {
        self.posting_cache.stats()
    }

    /// Get all document IDs in this segment.
    pub fn doc_ids(&self) -> Result<Vec<u64>> {
        if self.stored_documents.read().unwrap().is_none() {
            self.load_stored_documents()?;
        }
        let docs = self.stored_documents.read().unwrap();
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

    /// Load the segment data.
    pub fn load(&mut self) -> Result<()> {
        if self.loaded.load(Ordering::Acquire) {
            return Ok(());
        }

        // Load term dictionary
        self.load_term_dictionary()?;

        // Load stored documents
        self.load_stored_documents()?;

        // Load DocValues
        self.load_doc_values()?;

        // Load deletion bitmap if present
        self.load_deletion_bitmap()?;

        self.loaded.store(true, Ordering::Release);
        Ok(())
    }

    /// Load the term dictionary for this segment.
    fn load_term_dictionary(&self) -> Result<()> {
        let dict_file = format!("{}.dict", self.info.segment_id);

        if let Ok(input) = self.storage.open_input(&dict_file) {
            let mut reader = StructReader::new(input)?;
            let dictionary = BlockTermDictionary::read_from_storage(&mut reader).map_err(|e| {
                LaurusError::index(format!(
                    "Failed to read term dictionary from {dict_file}: {e}"
                ))
            })?;
            *self.term_dictionary.write().unwrap() = Some(Arc::new(dictionary));
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
        // Primary: typed binary `.docs`, which records the real doc_id per
        // document (correct for non-contiguous ids, e.g. merged segments).
        let docs_file = format!("{}.docs", self.info.segment_id);
        if let Ok(input) = self.storage.open_input(&docs_file) {
            let mut reader = StructReader::new(input)?;
            let doc_count = reader.read_varint()? as usize;
            let mut documents = BTreeMap::new();

            for _ in 0..doc_count {
                let doc_id = reader.read_u64()?;
                let field_count = reader.read_varint()? as usize;
                let mut doc = Document::new();

                for _ in 0..field_count {
                    let field_name = reader.read_string()?;

                    // Read type tag
                    let type_tag = reader.read_u8()?;

                    // Read value based on type tag
                    let field_value = match type_tag {
                        0 => {
                            // Text
                            let text = reader.read_string()?;
                            FieldValue::Text(text)
                        }
                        1 => {
                            // Integer
                            // Stored as u64 via `i64 as u64` (bit-preserving). Reverse with `u64 as i64`.
                            let num = reader.read_u64()? as i64;
                            FieldValue::Int64(num)
                        }
                        2 => {
                            // Float
                            let num = reader.read_f64()?;
                            FieldValue::Float64(num)
                        }
                        3 => {
                            // Boolean
                            let b = reader.read_u8()? != 0;
                            FieldValue::Bool(b)
                        }
                        4 => {
                            // Bytes (MIME type + Data)
                            let mime = reader.read_string()?;
                            let data = reader.read_bytes()?;
                            FieldValue::Bytes(data, if mime.is_empty() { None } else { Some(mime) })
                        }
                        5 => {
                            // DateTime
                            let dt_str = reader.read_string()?;
                            let dt = chrono::DateTime::parse_from_rfc3339(&dt_str)
                                .map_err(|e| {
                                    LaurusError::index(format!("Failed to parse DateTime: {e}"))
                                })?
                                .with_timezone(&chrono::Utc);
                            FieldValue::DateTime(dt)
                        }
                        6 => {
                            // Geo
                            let lat = reader.read_f64()?;
                            let lon = reader.read_f64()?;
                            FieldValue::Geo(crate::data::GeoPoint::new(lat, lon))
                        }
                        7 => {
                            // Null
                            FieldValue::Null
                        }
                        10 => {
                            // Int64Array
                            let len = reader.read_varint()? as usize;
                            let mut arr = Vec::with_capacity(len);
                            for _ in 0..len {
                                arr.push(reader.read_u64()? as i64);
                            }
                            FieldValue::Int64Array(arr)
                        }
                        11 => {
                            // Float64Array
                            let len = reader.read_varint()? as usize;
                            let mut arr = Vec::with_capacity(len);
                            for _ in 0..len {
                                arr.push(reader.read_f64()?);
                            }
                            FieldValue::Float64Array(arr)
                        }
                        12 => {
                            // 3D ECEF point. Tag 12 (not 11) — see the
                            // matching writer comment for why ECEF was
                            // remapped during #299.
                            let x = reader.read_f64()?;
                            let y = reader.read_f64()?;
                            let z = reader.read_f64()?;
                            FieldValue::GeoEcef(crate::data::GeoEcefPoint::new(x, y, z))
                        }
                        _ => {
                            return Err(LaurusError::index(format!(
                                "Unknown field type tag: {type_tag}"
                            )));
                        }
                    };

                    doc.fields.insert(field_name, field_value);
                }

                documents.insert(doc_id, doc);
            }

            *self.stored_documents.write().unwrap() = Some(documents);
            return Ok(());
        }

        // Legacy fallback: segments written before the binary `.docs` format
        // stored fields as a positional JSON mirror (Issue #756 stopped writing
        // it). Doc ids are assigned positionally — valid only for the
        // contiguous ids those legacy segments used.
        let json_file = format!("{}.json", self.info.segment_id);
        if self.storage.file_exists(&json_file) {
            let mut input = self.storage.open_input(&json_file)?;
            let mut json_data = String::new();
            std::io::Read::read_to_string(&mut input, &mut json_data)?;

            let docs: Vec<Document> = serde_json::from_str(&json_data)
                .map_err(|e| LaurusError::index(format!("Failed to parse JSON documents: {e}")))?;

            let mut documents = BTreeMap::new();
            for (idx, doc) in docs.into_iter().enumerate() {
                let doc_id = self.info.min_doc_id + idx as u64;
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

    /// Load DocValues for this segment.
    fn load_doc_values(&self) -> Result<()> {
        // Load DocValues file (required for field sorting)
        let reader = DocValuesReader::load(self.storage.clone(), &self.info.segment_id)?;

        let mut doc_values = self.doc_values.write().unwrap();
        *doc_values = Some(Arc::new(reader));

        Ok(())
    }

    /// Load deletion bitmap if present for this segment.
    fn load_deletion_bitmap(&self) -> Result<()> {
        if !self.info.has_deletions {
            return Ok(());
        }

        // Already loaded
        if self.deletion_bitmap.read().unwrap().is_some() {
            return Ok(());
        }

        let bitmap_file = format!("{}.delmap", self.info.segment_id);
        if !self.storage.file_exists(&bitmap_file) {
            // Metadata says we have deletions but bitmap is missing; treat as no deletions.
            return Ok(());
        }

        let input = self.storage.open_input(&bitmap_file)?;
        let mut reader = StructReader::new(input)?;
        let bitmap = DeletionBitmap::read_from_storage(&mut reader)?;
        *self.deletion_bitmap.write().unwrap() = Some(Arc::new(bitmap));
        Ok(())
    }

    /// Check whether a global doc_id is marked as deleted in this segment.
    pub fn is_deleted(&self, doc_id: u64) -> Result<bool> {
        // Lock-free fast path: a segment with no deletions can never mark a doc
        // deleted, so skip the `deletion_bitmap` RwLock acquire entirely. This
        // is hot on the scoring path, which probes deletion status per scored
        // doc (often redundantly, since the posting iterator is already
        // deletion-filtered at decode time via `filter_deleted_soa`).
        if !self.info.has_deletions {
            return Ok(false);
        }

        // Find deletion bitmap (load on demand the first time).
        if self.deletion_bitmap.read().unwrap().is_none() {
            self.load_deletion_bitmap()?;
        }

        let bitmap_lock = self.deletion_bitmap.read().unwrap();
        if let Some(ref bitmap) = *bitmap_lock {
            Ok(bitmap.is_deleted(doc_id))
        } else {
            Ok(false)
        }
    }

    /// Drop deleted entries from a SoA-decoded posting list in lockstep
    /// across the parallel arrays (`doc_ids`, `frequencies`, optional
    /// positions). Returns the same list unchanged when the segment has no
    /// deletions, avoiding any allocation in the common case.
    fn filter_deleted_soa(&self, decoded: DecodedPostingList) -> Result<DecodedPostingList> {
        // Fast path: nothing to filter.
        if !self.info.has_deletions {
            return Ok(decoded);
        }
        // Materialise the bitmap once (load on demand) so the inner loop is a
        // pure index lookup.
        if self.deletion_bitmap.read().unwrap().is_none() {
            self.load_deletion_bitmap()?;
        }
        let bitmap_lock = self.deletion_bitmap.read().unwrap();
        let bitmap = match bitmap_lock.as_ref() {
            Some(b) => b,
            None => return Ok(decoded),
        };

        let n = decoded.doc_ids.len();
        let mut doc_ids = Vec::with_capacity(n);
        let mut frequencies = Vec::with_capacity(n);
        let mut positions: Option<Vec<Option<Vec<u32>>>> =
            decoded.positions.as_ref().map(|_| Vec::with_capacity(n));

        for i in 0..n {
            let did = decoded.doc_ids[i] as u64;
            if bitmap.is_deleted(did) {
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
        if self.doc_values.read().unwrap().is_none() {
            self.load_doc_values()?;
        }

        let doc_values = self.doc_values.read().unwrap();
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
        if self.doc_values.read().unwrap().is_none() && self.load_doc_values().is_err() {
            return false;
        }
        let doc_values = self.doc_values.read().unwrap();
        if let Some(reader) = doc_values.as_ref() {
            reader.has_field(field)
        } else {
            false
        }
    }

    /// Load field lengths from the segment.
    fn load_field_lengths(&self) -> Result<()> {
        let lens_file = format!("{}.lens", self.info.segment_id);

        // Check if file exists (for backward compatibility with old indexes)
        if !self.storage.file_exists(&lens_file) {
            // Old index without field lengths - initialize empty
            *self.field_lengths.write().unwrap() = Some(BTreeMap::new());
            return Ok(());
        }

        let lens_input = self.storage.open_input(&lens_file)?;
        let mut lens_reader = StructReader::new(lens_input)?;

        let doc_count = lens_reader.read_varint()? as usize;
        let mut all_field_lengths = BTreeMap::new();

        for _ in 0..doc_count {
            let doc_id = lens_reader.read_u64()?;
            let field_count = lens_reader.read_varint()? as usize;

            let mut field_lens = AHashMap::new();
            for _ in 0..field_count {
                let field_name = lens_reader.read_string()?;
                let length = lens_reader.read_u32()?;
                field_lens.insert(field_name, length);
            }

            all_field_lengths.insert(doc_id, field_lens);
        }

        *self.field_lengths.write().unwrap() = Some(all_field_lengths);
        Ok(())
    }

    /// Load field statistics from the segment.
    fn load_field_stats(&self) -> Result<()> {
        let fstats_file = format!("{}.fstats", self.info.segment_id);

        // Check if file exists (for backward compatibility with old indexes)
        if !self.storage.file_exists(&fstats_file) {
            // Old index without field stats - initialize empty
            *self.field_stats.write().unwrap() = Some(AHashMap::new());
            return Ok(());
        }

        let fstats_input = self.storage.open_input(&fstats_file)?;
        let mut fstats_reader = StructReader::new(fstats_input)?;

        let field_count = fstats_reader.read_varint()? as usize;
        let mut all_field_stats = AHashMap::new();

        for _ in 0..field_count {
            let field_name = fstats_reader.read_string()?;
            let doc_count = fstats_reader.read_u64()?;
            let avg_length = fstats_reader.read_f64()?;
            let min_length = fstats_reader.read_u64()?;
            let max_length = fstats_reader.read_u64()?;

            all_field_stats.insert(
                field_name.clone(),
                crate::lexical::reader::FieldStats {
                    field: field_name,
                    unique_terms: 0, // Not stored, not needed for BM25
                    total_terms: 0,  // Not stored, not needed for BM25
                    doc_count,
                    avg_length,
                    min_length,
                    max_length,
                },
            );
        }

        *self.field_stats.write().unwrap() = Some(all_field_stats);
        Ok(())
    }

    /// Get field statistics for a specific field.
    pub fn field_stats(&self, field: &str) -> Result<Option<FieldStats>> {
        // Ensure field stats are loaded
        if self.field_stats.read().unwrap().is_none() {
            self.load_field_stats()?;
        }

        let field_stats = self.field_stats.read().unwrap();
        if let Some(ref stats_map) = *field_stats {
            return Ok(stats_map.get(field).cloned());
        }
        Ok(None)
    }

    /// Get field length for a specific document and field.
    ///
    /// Uses a fast path with a single `RwLock` acquisition when field lengths
    /// are already loaded (hot path). Falls back to loading on the first call
    /// (cold path), which requires a second acquisition.
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
        let field_lengths = self.field_lengths.read().unwrap();
        if let Some(ref lengths_map) = *field_lengths {
            return Ok(lengths_map
                .get(&doc_id)
                .and_then(|doc_lengths| doc_lengths.get(field).copied()));
        }
        drop(field_lengths);

        // Cold path: load field lengths (one-time), then retry.
        self.load_field_lengths()?;
        let field_lengths = self.field_lengths.read().unwrap();
        if let Some(ref lengths_map) = *field_lengths {
            return Ok(lengths_map
                .get(&doc_id)
                .and_then(|doc_lengths| doc_lengths.get(field).copied()));
        }
        Ok(None)
    }

    /// Get a document by ID from this segment.
    pub fn document(&self, doc_id: u64) -> Result<Option<Document>> {
        // Load once, on demand; the cache itself is the load gate (the
        // `loaded` flag is only set by the optional bulk `load()` path,
        // so gating on it re-decoded the whole segment per call — #994).
        if self.stored_documents.read().unwrap().is_none() {
            self.load_stored_documents()?;
        }

        if self.is_deleted(doc_id)? {
            return Ok(None);
        }

        let docs = self.stored_documents.read().unwrap();
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
        if self.stored_documents.read().unwrap().is_none() {
            self.load_stored_documents()?;
        }

        if self.is_deleted(doc_id)? {
            return Ok(None);
        }

        let docs = self.stored_documents.read().unwrap();
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

    /// Get term information for a field and term.
    pub fn term_info(&self, field: &str, term: &str) -> Result<Option<TermInfo>> {
        // Lazy load term dictionary if not loaded
        if self.term_dictionary.read().unwrap().is_none() && !self.loaded.load(Ordering::Acquire) {
            self.load_term_dictionary()?;
        }

        if let Some(ref dict) = *self.term_dictionary.read().unwrap() {
            let full_term = format!("{field}:{term}");
            Ok(dict.get(&full_term).cloned())
        } else {
            Ok(None)
        }
    }

    /// Get posting list for a field and term.
    /// Return this segment's term dictionary, loading it on demand.
    ///
    /// Exposes the dictionary so the segment merge (Issue #753) can enumerate
    /// every `"field:term"` key (via [`BlockTermDictionary::iter`]) and re-read
    /// each term's postings through [`Self::postings`] without re-tokenizing.
    /// Returns `None` when the segment has no on-disk term dictionary.
    pub fn term_dictionary(&self) -> Result<Option<Arc<BlockTermDictionary>>> {
        self.load_term_dictionary()?;
        Ok(self.term_dictionary.read().unwrap().clone())
    }

    pub fn postings(&self, field: &str, term: &str) -> Result<Option<Box<dyn PostingIterator>>> {
        // Load postings from storage
        let postings_file = format!("{}.post", self.info.segment_id);

        if !self.storage.file_exists(&postings_file) {
            // No inverted index, fall back to document scanning
            return self.scan_documents_for_term(field, term);
        }

        // Posting cache (#612): a repeated `(field, term)` lookup within this
        // reader snapshot reuses the decoded, deletion-filtered list instead of
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
            let input = self.storage.open_input(&postings_file)?;
            let mut reader = StructReader::new(input)?;

            // Seek directly to the posting position
            if term_info.posting_offset > 0 {
                reader.seek(std::io::SeekFrom::Start(term_info.posting_offset))?;
            }

            // Decode the posting list in SoA-native form to skip the
            // intermediate `Vec<Posting>` reassembly and keep the iterator
            // backed by parallel `Vec<u32>` slices. Dispatch by on-disk
            // posting format version: v2 segments carry the multi-level
            // skip table inline (#503) while v1 segments rebuild it from
            // `doc_ids` at load time inside `decode_soa`; v3 additionally
            // gates the weights section on a header byte (#553).
            //
            // Matched exactly rather than with an ordered comparison. A
            // `>=` would route a newer payload into an older decoder,
            // which reads the added header byte as the next field and
            // corrupts the list silently instead of failing.
            let posting_format = self
                .term_dictionary
                .read()
                .unwrap()
                .as_ref()
                .map(|dict| dict.posting_format_version())
                .unwrap_or(3);
            let decoded = match posting_format {
                1 => PostingList::decode_soa(&mut reader)?,
                2 => PostingList::decode_soa_v2(&mut reader)?,
                _ => PostingList::decode_soa_v3(&mut reader)?,
            };
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

    /// Scan documents for a term (fallback when no inverted index).
    fn scan_documents_for_term(
        &self,
        field: &str,
        term: &str,
    ) -> Result<Option<Box<dyn PostingIterator>>> {
        // Ensure documents are loaded
        if !self.loaded.load(Ordering::Acquire) {
            // Load documents on-demand
            self.load_stored_documents()?;
        }

        let docs = self.stored_documents.read().unwrap();

        if let Some(ref documents) = *docs {
            let mut postings = Vec::new();
            let default_analyzer = StandardAnalyzer::new()?;

            for (doc_id, doc) in documents.iter() {
                if self.is_deleted(*doc_id)? {
                    continue;
                }

                if let Some(field_value) = doc.get_field(field)
                    && let Some(text) = field_value.as_text()
                {
                    // Use default analyzer (analyzers are configured at writer level)
                    let token_stream = default_analyzer.analyze(text)?;
                    let tokens: Vec<Token> = token_stream.collect();

                    let mut positions = Vec::new();
                    for token in tokens.iter() {
                        if token.text == term {
                            positions.push(token.position as u32);
                        }
                    }

                    if !positions.is_empty() {
                        postings.push(Posting {
                            doc_id: *doc_id,
                            frequency: positions.len() as u32,
                            positions: Some(positions),
                            weight: 1.0,
                        });
                    }
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

    /// Get the number of documents in this segment.
    pub fn doc_count(&self) -> u64 {
        if !self.info.has_deletions {
            return self.info.doc_count;
        }

        if let Some(bitmap) = self.deletion_bitmap.read().unwrap().clone() {
            return bitmap.live_count();
        }

        // Lazy load bitmap if needed
        if self.load_deletion_bitmap().is_ok()
            && let Some(bitmap) = self.deletion_bitmap.read().unwrap().clone()
        {
            return bitmap.live_count();
        }

        self.info.doc_count
    }

    /// Get BKD Tree for a field, loading it if necessary.
    pub fn get_bkd_tree(&self, field: &str) -> Result<Option<Arc<dyn BKDTree>>> {
        // Check cache
        if let Some(tree) = self.bkd_trees.read().unwrap().get(field) {
            return Ok(Some(tree.clone()));
        }

        // Try to open file
        let bkd_file = format!("{}.{}.bkd", self.info.segment_id, field);
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
        if !self.info.has_deletions {
            return Ok(Some(tree));
        }
        // Ensure the bitmap is loaded; the load is idempotent and
        // tolerates the "metadata says deletions but file is missing"
        // case by leaving the bitmap unset, in which case we forward
        // the raw tree.
        self.load_deletion_bitmap()?;
        let Some(bitmap) = self.deletion_bitmap.read().unwrap().clone() else {
            return Ok(Some(tree));
        };
        let snapshot = Arc::new(DeletionSnapshot {
            bitmaps: vec![(self.info.min_doc_id, self.info.max_doc_id, bitmap)],
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
    /// `(min_doc_id, max_doc_id, bitmap)` for each segment that has
    /// any recorded deletion. Segments with no deletions are not
    /// stored — they cannot contribute hits.
    bitmaps: Vec<(u64, u64, Arc<DeletionBitmap>)>,
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
        for (min, max, bitmap) in &self.bitmaps {
            if doc_id >= *min && doc_id <= *max && bitmap.is_deleted(doc_id) {
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
    /// # Arguments
    ///
    /// * `memory_limit` - Soft memory budget in bytes; the term cache's entry
    ///   capacity is derived as `memory_limit / EST_TERM_ENTRY_BYTES` (at
    ///   least one entry).
    pub fn new(memory_limit: usize) -> Self {
        let capacity = NonZeroUsize::new((memory_limit / EST_TERM_ENTRY_BYTES).max(1))
            .unwrap_or(NonZeroUsize::MIN);
        CacheManager {
            term_cache: Mutex::new(LruCache::new(capacity)),
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
}

impl InvertedIndexReader {
    /// Create a new advanced index reader (schema-less mode).
    pub fn new(
        segments: Vec<SegmentInfo>,
        storage: Arc<dyn Storage>,
        config: InvertedIndexReaderConfig,
    ) -> Result<Self> {
        let cache_manager = Arc::new(CacheManager::new(config.max_cache_memory));
        let query_cache = Arc::new(QueryFilterCache::new(config.query_filter_cache_capacity));
        let mut segment_readers = Vec::new();
        let mut total_doc_count = 0;

        // Enable the per-segment posting cache (Issue #612) for query readers,
        // gated by `enable_posting_cache` and budgeted by `max_cache_memory`.
        let posting_cache_bytes = if config.enable_posting_cache {
            config.max_cache_memory
        } else {
            0
        };
        for segment_info in &segments {
            total_doc_count += segment_info.doc_count;
            let mut reader = SegmentReader::open(segment_info.clone(), storage.clone())?
                .with_posting_cache_bytes(posting_cache_bytes);

            if config.preload_segments {
                reader.load()?;
            }

            segment_readers.push(Arc::new(RwLock::new(reader)));
        }

        Ok(InvertedIndexReader {
            segment_readers,
            segment_infos: segments,
            cache_manager,
            query_cache,
            config,
            closed: Arc::new(AtomicBool::new(false)),
            total_doc_count,
        })
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

        // Search across segments, skipping those that cannot contain doc_id.
        for (i, segment_reader) in self.segment_readers.iter().enumerate() {
            // Use cached segment info to skip out-of-range segments
            // without acquiring the reader lock.
            if let Some(info) = self.segment_infos.get(i)
                && (doc_id < info.min_doc_id || doc_id > info.max_doc_id)
            {
                continue;
            }
            let reader = segment_reader.read().unwrap();
            if let Ok(Some(length)) = reader.field_length(doc_id, field) {
                return Ok(Some(length));
            }
        }

        Ok(None)
    }
}

impl crate::lexical::reader::LexicalIndexReader for InvertedIndexReader {
    fn doc_count(&self) -> u64 {
        // Sum live doc counts from each segment (accounts for deletions).
        self.segment_readers
            .iter()
            .map(|sr| sr.read().unwrap().doc_count())
            .sum()
    }

    fn max_doc(&self) -> u64 {
        // max_doc reflects the total allocated doc space (including deleted).
        self.total_doc_count
    }

    fn is_deleted(&self, doc_id: u64) -> bool {
        // Find the segment containing this document
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            // In Stable ID mode, we ask the reader directly.
            // A reader returns false if it doesn't own the document.
            if let Ok(true) = reader.is_deleted(doc_id) {
                return true;
            }
        }
        false
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
        // individual posting's TF-component contribution (#403 PR-B2).
        //
        // Block-max metadata is concatenated across segments
        // (#403 PR-D). The inverted writer assigns segments
        // monotonically-increasing doc-id ranges, so segment-order
        // concatenation preserves the `last_doc_id` ordering that
        // [`BM25Scorer::block_max_score_at`]'s binary search relies
        // on.
        //
        // Per-block `max_factor` was computed against each segment's
        // local `avg_field_length`. The cross-segment BM25 scorer
        // uses the global average; the per-block factor is therefore
        // approximate. For corpora whose segments have similar
        // average field lengths (the common case — segments are
        // sized in docs, not field-length bytes) the divergence is
        // small and the factor remains a usable bound. For corpora
        // with widely-varying segment averages a future PR will need
        // to either re-index or store enough per-block raw data
        // (max-tf + min-field-length) to re-anchor the factor at
        // query time.
        let mut total_doc_freq = 0;
        let mut total_term_freq = 0;
        let mut max_score_factor: f32 = 0.0;
        let mut matched_count = 0_usize;
        let mut combined_block_max: Vec<crate::lexical::index::structures::dictionary::BlockMax> =
            Vec::new();

        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            if let Some(term_info) = reader.term_info(field, term)? {
                total_doc_freq += term_info.doc_frequency;
                total_term_freq += term_info.total_frequency;
                max_score_factor = max_score_factor.max(term_info.max_score_factor);
                matched_count += 1;
                combined_block_max.extend(term_info.block_max.iter().copied());
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
        // multi-segment case sidesteps both issues. Cross-segment
        // block-max passthrough is tracked as a follow-up.
        let aggregated_block_max = if matched_count == 1 {
            combined_block_max
        } else {
            Vec::new()
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
        let mut total_length_sum = 0u64; // Sum of (avg_length * doc_count) for weighted average
        let mut min_length = u64::MAX;
        let mut max_length = 0u64;
        let mut found = false;

        // Aggregate statistics from all segments
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();

            // Get field stats from this segment
            if let Some(segment_stats) = reader.field_stats(field)? {
                total_doc_count += segment_stats.doc_count;
                total_length_sum +=
                    (segment_stats.avg_length * segment_stats.doc_count as f64) as u64;
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
                    total_length_sum as f64 / total_doc_count as f64
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

    fn get_bkd_tree(&self, field: &str) -> Result<Option<Arc<dyn BKDTree>>> {
        self.check_closed()?;

        let mut trees = Vec::new();
        for segment_reader in &self.segment_readers {
            let reader = segment_reader.read().unwrap();
            if let Some(tree) = reader.get_bkd_tree(field)? {
                trees.push(tree);
            }
        }

        if trees.is_empty() {
            return Ok(None);
        }

        let multi: Arc<dyn BKDTree> = Arc::new(MultiSegmentBKDTree { trees });

        // Capture a lock-free snapshot of every segment's deletion
        // bitmap *once* here, so per-hit checks during the search
        // never reach for a `RwLock`. Segments without any deletion
        // are skipped, and if no segment has any deletion at all the
        // wrapper's `intersect` short-circuits and forwards verbatim
        // to the inner BKD tree (zero overhead in the common case).
        let mut bitmaps = Vec::new();
        for sr in &self.segment_readers {
            let reader = sr.read().unwrap();
            if !reader.info.has_deletions {
                continue;
            }
            // Make sure the bitmap is loaded; the load is idempotent.
            reader.load_deletion_bitmap()?;
            if let Some(bitmap) = reader.deletion_bitmap.read().unwrap().clone() {
                bitmaps.push((reader.info.min_doc_id, reader.info.max_doc_id, bitmap));
            }
        }
        let snapshot = Arc::new(DeletionSnapshot { bitmaps });

        Ok(Some(Arc::new(DeletionFilteringBKDTree {
            inner: multi,
            snapshot,
        })))
    }
}

// Implementation of TermDictionaryAccess for InvertedIndexReader
impl TermDictionaryAccess for InvertedIndexReader {
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms>>> {
        // Collect term dictionaries from ALL segments and merge them.
        let mut dicts = Vec::new();
        for seg_lock in &self.segment_readers {
            let seg = seg_lock.read().unwrap();
            if seg.term_dictionary.read().unwrap().is_none() {
                seg.load_term_dictionary()?;
            }
            if let Some(dict) = seg.term_dictionary.read().unwrap().clone() {
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
    /// Both paths are covered: the normal one through `filter_deleted_soa`,
    /// and the `scan_documents_for_term` fallback taken when a segment has
    /// no `.post` file.
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

        let reader = writer.build_reader().unwrap();
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

        let reader = writer.build_reader().unwrap();
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
            shard_id: 0,
        };

        assert_eq!(info.segment_id, "seg_000001");
        assert_eq!(info.doc_count, 1000);
        assert_eq!(info.min_doc_id, 0);
        assert_eq!(info.max_doc_id, 999);
        assert!(!info.has_deletions);
    }
}
