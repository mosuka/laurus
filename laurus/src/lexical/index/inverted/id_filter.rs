//! A Bloom filter over every committed `_id` key of an index (Issue #1420).
//!
//! Every put resolves its `_id` against the committed segments to find the
//! versions it replaces. For an id no segment holds, probing each segment's
//! term dictionary still costs one hash and one probe per segment, so the
//! put grows with the segment count. [`CommittedIdFilter`] answers that case
//! once for the whole index: a key the filter has never seen is in no
//! segment, and only a "maybe" falls through to the per-segment lookup.
//!
//! The filter is owned by the [`InvertedIndex`](super::InvertedIndex) and
//! shared by every reader it builds. It never misses a key of a segment it
//! covers (a Bloom filter has no false negatives), and a reader makes sure
//! the filter covers all of its segments before trusting an "absent".

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ahash::{AHashSet, RandomState};
use parking_lot::Mutex;

use crate::error::Result;
use crate::lexical::index::inverted::segment::SegmentInfo;
use crate::lexical::index::structures::dictionary::BlockTermDictionary;

/// The field whose keys the filter holds: the engine's document id.
pub(crate) const ID_FIELD: &str = crate::engine::schema::RESERVED_ID_FIELD;

/// Prefix of every dictionary key of [`ID_FIELD`] (`"field:term"` keys).
const ID_KEY_PREFIX: &str = "_id:";

/// The first key past every [`ID_KEY_PREFIX`] key: `;` follows `:` in byte
/// order, and dictionary keys are sorted by bytes.
const ID_KEY_END: &str = "_id;";

/// Bits per key a filter is built with, for the keys it is built from.
const BITS_PER_KEY: u64 = 16;

/// Bits each key sets, all within one block.
const BITS_SET_PER_KEY: u32 = 7;

/// Fill (share of bits set) a filter may reach before it is rebuilt.
///
/// At 0.5 with seven bits per key a plain Bloom filter answers "maybe" for
/// about 0.8% of absent keys; the blocked layout adds a little. The tests pin
/// the measured rate.
const MAX_FILL: f64 = 0.5;

/// Smallest number of keys a filter is sized for.
const MIN_KEYS: u64 = 1024;

/// Bits in a block: one 64-byte cache line.
const BLOCK_BITS: u64 = 512;

/// One cache line of filter bits.
type Block = [u64; 8];

/// Result of [`CommittedIdFilter::check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IdCheck {
    /// No covered segment holds the key.
    Absent,
    /// Some segment may hold the key; look it up exactly.
    Maybe,
    /// Some segment of the reader has no term dictionary, so the filter
    /// cannot cover it.
    Unavailable,
}

/// What identifies a segment's contents: its files never change once
/// written, and a name alone could in principle be reused.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SegmentKey {
    segment_id: String,
    min_doc_id: u64,
    max_doc_id: u64,
    doc_count: u64,
}

impl SegmentKey {
    fn of(info: &SegmentInfo) -> Self {
        Self {
            segment_id: info.segment_id.clone(),
            min_doc_id: info.min_doc_id,
            max_doc_id: info.max_doc_id,
            doc_count: info.doc_count,
        }
    }
}

/// A Bloom filter whose keys each set [`BITS_SET_PER_KEY`] bits inside one
/// [`BLOCK_BITS`]-bit block, so a probe touches one cache line.
struct BlockedBloom {
    blocks: Vec<Block>,
    /// Bits currently set.
    ones: u64,
    /// Keys inserted since the filter was built, repeats included.
    inserted: u64,
    /// Keys the filter takes before its fill passes [`MAX_FILL`].
    max_keys: u64,
}

impl BlockedBloom {
    /// A filter sized for `keys` keys at [`BITS_PER_KEY`].
    fn with_capacity(keys: u64) -> Self {
        let keys = keys.max(MIN_KEYS);
        let block_count = (keys * BITS_PER_KEY).div_ceil(BLOCK_BITS);
        let bits = block_count * BLOCK_BITS;
        // Expected fill after n keys is 1 - exp(-k n / bits); solve for n.
        let max_keys =
            (bits as f64 * (1.0 / (1.0 - MAX_FILL)).ln() / f64::from(BITS_SET_PER_KEY)) as u64;
        Self {
            blocks: vec![[0; 8]; block_count as usize],
            ones: 0,
            inserted: 0,
            max_keys,
        }
    }

    /// The block a hash selects, and the masks of the bits it sets there.
    fn locate(&self, hash: u64) -> (usize, [(usize, u64); BITS_SET_PER_KEY as usize]) {
        let block = ((u128::from(hash) * self.blocks.len() as u128) >> 64) as usize;
        // The block takes the high bits of `hash`; mix it again so the bit
        // positions do not depend on the block.
        let positions = splitmix64(hash);
        let mut bits = [(0, 0); BITS_SET_PER_KEY as usize];
        for (i, bit) in bits.iter_mut().enumerate() {
            let position = (positions >> (9 * i)) & (BLOCK_BITS - 1);
            *bit = ((position / 64) as usize, 1u64 << (position % 64));
        }
        (block, bits)
    }

    fn insert(&mut self, hash: u64) {
        let (block, bits) = self.locate(hash);
        let block = &mut self.blocks[block];
        for (word, mask) in bits {
            if block[word] & mask == 0 {
                block[word] |= mask;
                self.ones += 1;
            }
        }
        self.inserted += 1;
    }

    fn may_contain(&self, hash: u64) -> bool {
        let (block, bits) = self.locate(hash);
        let block = &self.blocks[block];
        bits.iter().all(|&(word, mask)| block[word] & mask != 0)
    }

    /// Whether `more` keys still fit before the fill passes [`MAX_FILL`].
    /// Counts repeats, so it errs toward rebuilding early.
    fn fits(&self, more: u64) -> bool {
        self.inserted + more <= self.max_keys
    }

    #[cfg(test)]
    fn bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Block>()
    }
}

/// The splitmix64 finalizer: a bijective mix of all 64 bits.
fn splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The keys of [`ID_FIELD`] in a segment's dictionary.
fn id_keys(dictionary: &BlockTermDictionary) -> impl Iterator<Item = &str> + '_ {
    let start = dictionary.seek_index(ID_KEY_PREFIX);
    let end = dictionary.seek_index(ID_KEY_END);
    (start..end).filter_map(|ordinal| dictionary.entry_at(ordinal).map(|(key, _)| key))
}

/// How many keys of [`ID_FIELD`] a segment's dictionary holds.
fn id_key_count(dictionary: &BlockTermDictionary) -> u64 {
    (dictionary.seek_index(ID_KEY_END) - dictionary.seek_index(ID_KEY_PREFIX)) as u64
}

struct FilterState {
    bloom: BlockedBloom,
    /// The segments whose keys `bloom` holds.
    covered: AHashSet<SegmentKey>,
    /// Bumped on every rebuild; a reader that covered an older generation
    /// covers its segments again. Starts at 1, so 0 means "never covered".
    generation: u64,
}

/// A Bloom filter over the `_id` keys of every committed segment an index's
/// readers have seen (Issue #1420).
///
/// In memory only: built on the first `_id` lookup after the index is
/// opened, grown at the lookup after a commit by the new segments' keys, and
/// rebuilt from the current segments when it fills up. A rebuild also drops
/// the keys of segments merged away since the last one.
pub(crate) struct CommittedIdFilter {
    hasher: RandomState,
    state: Mutex<FilterState>,
}

impl std::fmt::Debug for CommittedIdFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommittedIdFilter").finish_non_exhaustive()
    }
}

impl Default for CommittedIdFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl CommittedIdFilter {
    /// An empty filter that covers no segment yet.
    pub(crate) fn new() -> Self {
        Self {
            hasher: RandomState::new(),
            state: Mutex::new(FilterState {
                bloom: BlockedBloom::with_capacity(0),
                covered: AHashSet::new(),
                generation: 1,
            }),
        }
    }

    /// The hash every insert and probe uses. Hashing the full key in one
    /// call keeps the two equal; hashing the prefix and the id separately
    /// would give a different value.
    fn hash(&self, key: &str) -> u64 {
        self.hasher.hash_one(key)
    }

    /// Whether any of a reader's segments may hold `key`, covering the
    /// segments the filter has not seen first.
    ///
    /// # Arguments
    ///
    /// * `covered_generation` - The filter generation the reader last
    ///   covered its segments at (0 if never); updated here.
    /// * `segments` - The reader's segments, in order.
    /// * `dictionary` - The term dictionary of the segment at an index into
    ///   `segments`, loaded on demand; `None` if it has none.
    /// * `key` - The full dictionary key, `"_id:<id>"`.
    ///
    /// # Returns
    ///
    /// [`IdCheck::Absent`] only if no segment in `segments` holds `key`.
    ///
    /// # Errors
    ///
    /// Returns what `dictionary` returns; nothing is marked covered then.
    pub(crate) fn check(
        &self,
        covered_generation: &AtomicU64,
        segments: &[SegmentInfo],
        dictionary: impl Fn(usize) -> Result<Option<Arc<BlockTermDictionary>>>,
        key: &str,
    ) -> Result<IdCheck> {
        let mut state = self.state.lock();
        if covered_generation.load(Ordering::Acquire) != state.generation {
            if !self.cover(&mut state, segments, &dictionary)? {
                return Ok(IdCheck::Unavailable);
            }
            covered_generation.store(state.generation, Ordering::Release);
        }
        Ok(if state.bloom.may_contain(self.hash(key)) {
            IdCheck::Maybe
        } else {
            IdCheck::Absent
        })
    }

    /// Insert the keys of `segments` the filter does not cover yet, or
    /// rebuild it from all of `segments` when they would overfill it.
    ///
    /// Returns `false`, leaving the filter as it was, when a segment has no
    /// term dictionary.
    fn cover(
        &self,
        state: &mut FilterState,
        segments: &[SegmentInfo],
        dictionary: &impl Fn(usize) -> Result<Option<Arc<BlockTermDictionary>>>,
    ) -> Result<bool> {
        let mut new = Vec::new();
        for (index, info) in segments.iter().enumerate() {
            let key = SegmentKey::of(info);
            if state.covered.contains(&key) {
                continue;
            }
            let Some(segment_dictionary) = dictionary(index)? else {
                return Ok(false);
            };
            new.push((key, segment_dictionary));
        }
        let more: u64 = new.iter().map(|(_, d)| id_key_count(d)).sum();
        if !state.bloom.fits(more) {
            return self.rebuild(state, segments, dictionary);
        }
        for (key, segment_dictionary) in new {
            for id_key in id_keys(&segment_dictionary) {
                state.bloom.insert(self.hash(id_key));
            }
            state.covered.insert(key);
        }
        Ok(true)
    }

    /// Replace the filter with one built from exactly `segments`, sized for
    /// their keys, and bump the generation. The old filter stays in place if
    /// a dictionary is missing or fails to load.
    fn rebuild(
        &self,
        state: &mut FilterState,
        segments: &[SegmentInfo],
        dictionary: &impl Fn(usize) -> Result<Option<Arc<BlockTermDictionary>>>,
    ) -> Result<bool> {
        let mut dictionaries = Vec::with_capacity(segments.len());
        for index in 0..segments.len() {
            let Some(segment_dictionary) = dictionary(index)? else {
                return Ok(false);
            };
            dictionaries.push(segment_dictionary);
        }
        let total: u64 = dictionaries.iter().map(|d| id_key_count(d)).sum();
        let mut bloom = BlockedBloom::with_capacity(total);
        for segment_dictionary in &dictionaries {
            for id_key in id_keys(segment_dictionary) {
                bloom.insert(self.hash(id_key));
            }
        }
        state.bloom = bloom;
        state.covered = segments.iter().map(SegmentKey::of).collect();
        state.generation += 1;
        Ok(true)
    }

    /// The filter's size and state, for tests and measurements.
    #[cfg(test)]
    pub(crate) fn stats(&self) -> IdFilterStats {
        let state = self.state.lock();
        IdFilterStats {
            generation: state.generation,
            covered_segments: state.covered.len(),
            inserted: state.bloom.inserted,
            bytes: state.bloom.bytes(),
        }
    }
}

/// A snapshot of a [`CommittedIdFilter`]'s state.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IdFilterStats {
    /// Rebuild count plus one.
    pub generation: u64,
    /// Segments whose keys the filter holds.
    pub covered_segments: usize,
    /// Keys inserted since the last rebuild, repeats included.
    pub inserted: u64,
    /// Size of the filter's bits.
    pub bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hashed(filter: &CommittedIdFilter, key: &str) -> u64 {
        filter.hash(key)
    }

    #[test]
    fn id_key_bounds_bracket_exactly_the_id_field() {
        assert!(ID_KEY_PREFIX.starts_with(ID_FIELD));
        assert_eq!(ID_KEY_PREFIX, format!("{ID_FIELD}:"));
        // Every `_id:` key sorts in [prefix, end); nothing else does.
        let id_keys = ID_KEY_PREFIX..ID_KEY_END;
        for key in ["_id:", "_id:a", "_id:\u{10FFFF}", "_id:~~~"] {
            assert!(id_keys.contains(&key), "{key}");
        }
        for key in ["_ic:z", "_id", "_id;", "_ida:x", "body:_id:x"] {
            assert!(!id_keys.contains(&key), "{key}");
        }
    }

    #[test]
    fn blocked_bloom_never_misses_an_inserted_key() {
        let filter = CommittedIdFilter::new();
        let mut bloom = BlockedBloom::with_capacity(100_000);
        for i in 0..100_000 {
            bloom.insert(hashed(&filter, &format!("_id:doc-{i}")));
        }
        for i in 0..100_000 {
            assert!(bloom.may_contain(hashed(&filter, &format!("_id:doc-{i}"))));
        }
    }

    /// The false-positive rate at the fill a filter is rebuilt at, its worst
    /// state, stays under 2%.
    #[test]
    fn blocked_bloom_false_positive_rate_at_max_fill() {
        let filter = CommittedIdFilter::new();
        let mut bloom = BlockedBloom::with_capacity(100_000);
        let mut i = 0u64;
        while bloom.fits(1) {
            bloom.insert(hashed(&filter, &format!("_id:doc-{i}")));
            i += 1;
        }
        let fill = bloom.ones as f64 / (bloom.blocks.len() as f64 * BLOCK_BITS as f64);
        assert!((0.45..=0.55).contains(&fill), "fill {fill}");

        let probes = 100_000;
        let false_positives = (0..probes)
            .filter(|j| bloom.may_contain(hashed(&filter, &format!("_id:absent-{j}"))))
            .count();
        let rate = false_positives as f64 / f64::from(probes);
        assert!(rate < 0.02, "false-positive rate {rate} after {i} keys");
    }

    #[test]
    fn blocked_bloom_is_sized_by_bits_per_key() {
        let bloom = BlockedBloom::with_capacity(100_000);
        assert_eq!(
            bloom.bytes(),
            (100_000 * BITS_PER_KEY / 8).next_multiple_of(64) as usize
        );
        assert_eq!(
            BlockedBloom::with_capacity(0).bytes(),
            (MIN_KEYS * BITS_PER_KEY / 8) as usize
        );
    }
}
