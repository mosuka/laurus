//! Decoded posting-list cache (Issue
//! [#612](https://github.com/mosuka/laurus/issues/612)).
//!
//! Without it, every `SegmentReader::postings` re-opens the segment's `.post`
//! file, re-decodes the varint posting list, re-applies deletions, and rebuilds
//! the skip table — on every query. For cloud / remote storage the read alone
//! dominates. [`PostingCache`] memoises the decoded, deletion-filtered
//! [`DecodedPostingList`] per `(field, term)` so a repeated lookup reuses it.
//!
//! # Lifetime and key
//!
//! The cache lives on a [`SegmentReader`](crate::lexical::index::inverted::reader::SegmentReader),
//! whose deletions are fixed for its life (a commit that changes them builds
//! a new segment reader, Issue #1423), so caching the **post-deletion** list
//! is sound. The key is `"field\u{1}term"` (the segment is
//! implicit — the cache is per-segment).
//!
//! # Eviction
//!
//! Posting lists range from a few bytes to many megabytes, so the cache is
//! bounded by an **estimated byte budget**, not an entry count: on insert it
//! evicts the least-recently-used entries until back under budget. An entry
//! larger than the whole budget is not cached (it would evict everything else
//! for a single list). A budget of `0` disables the cache.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lru::LruCache;
use parking_lot::Mutex;

use crate::lexical::index::inverted::core::posting::DecodedPostingList;

/// Estimate the heap footprint of a decoded posting list, used for the byte
/// budget. Approximate — counts the parallel `u32` arrays, the skip table, the
/// optional positions sidecar, and the term string.
fn estimated_bytes(list: &DecodedPostingList) -> usize {
    let mut bytes = std::mem::size_of::<DecodedPostingList>();
    bytes += list.term.len();
    bytes += list.doc_ids.len() * std::mem::size_of::<u32>();
    bytes += list.frequencies.len() * std::mem::size_of::<u32>();
    bytes += list.weights.len() * std::mem::size_of::<f32>();
    for level in &list.skip_levels {
        bytes += level.len() * std::mem::size_of::<u32>();
    }
    if let Some(positions) = &list.positions {
        for p in positions.iter().flatten() {
            bytes += p.len() * std::mem::size_of::<u32>();
        }
    }
    bytes
}

/// Mutable cache state guarded by the [`PostingCache`] mutex.
#[derive(Debug)]
struct PostingCacheInner {
    /// Unbounded LRU; eviction is driven by `cur_bytes` against `max_bytes`.
    lru: LruCache<String, Arc<DecodedPostingList>>,
    /// Sum of `estimated_bytes` over the cached entries.
    cur_bytes: usize,
    /// Soft byte budget. Eviction runs until `cur_bytes <= max_bytes`.
    max_bytes: usize,
}

/// A byte-budget LRU cache of decoded, deletion-filtered posting lists, scoped
/// to one [`SegmentReader`](crate::lexical::index::inverted::reader::SegmentReader).
///
/// A [`Mutex`] guards the state because [`LruCache::get`] takes `&mut self` to
/// update recency and the byte accounting mutates on every insert.
#[derive(Debug)]
pub struct PostingCache {
    /// `None` when disabled (budget 0).
    inner: Option<Mutex<PostingCacheInner>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl PostingCache {
    /// Create a cache with the given byte budget. `0` disables it (every
    /// [`get`](Self::get) misses and [`put`](Self::put) is a no-op).
    ///
    /// # Arguments
    ///
    /// * `max_bytes` - Soft heap budget for cached posting lists.
    pub fn new(max_bytes: usize) -> Self {
        let inner = (max_bytes > 0).then(|| {
            Mutex::new(PostingCacheInner {
                lru: LruCache::unbounded(),
                cur_bytes: 0,
                max_bytes,
            })
        });
        PostingCache {
            inner,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Returns `true` if caching is enabled (budget was non-zero).
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Look up the decoded posting list cached for `key`, bumping its recency
    /// on a hit. Records a hit or miss in the statistics.
    ///
    /// # Arguments
    ///
    /// * `key` - The `"field\u{1}term"` cache key.
    pub fn get(&self, key: &str) -> Option<Arc<DecodedPostingList>> {
        let hit = self
            .inner
            .as_ref()
            .and_then(|inner| inner.lock().lru.get(key).cloned());
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    /// Insert a decoded posting list for `key`, evicting least-recently-used
    /// entries until back under the byte budget. A no-op when the cache is
    /// disabled or when this single entry exceeds the whole budget.
    ///
    /// # Arguments
    ///
    /// * `key` - The `"field\u{1}term"` cache key.
    /// * `list` - The decoded, deletion-filtered posting list to share.
    pub fn put(&self, key: String, list: Arc<DecodedPostingList>) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        let size = estimated_bytes(&list);
        let mut guard = inner.lock();
        // A single list bigger than the whole budget is not cached — caching it
        // would evict everything else and still not fit.
        if size > guard.max_bytes {
            return;
        }
        if let Some(old) = guard.lru.put(key, list) {
            guard.cur_bytes = guard.cur_bytes.saturating_sub(estimated_bytes(&old));
        }
        guard.cur_bytes += size;
        // Evict LRU entries until under budget. The just-inserted entry is the
        // MRU, and `size <= max_bytes`, so it is never the one evicted.
        while guard.cur_bytes > guard.max_bytes {
            match guard.lru.pop_lru() {
                Some((_, evicted)) => {
                    guard.cur_bytes = guard.cur_bytes.saturating_sub(estimated_bytes(&evicted));
                }
                None => break,
            }
        }
    }

    /// Number of cached entries (0 when disabled). For tests / observability.
    pub fn len(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(0, |inner| inner.lock().lru.len())
    }

    /// Returns `true` if the cache holds no entries (or is disabled).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot of the cache hit / miss counters.
    pub fn stats(&self) -> PostingCacheStats {
        PostingCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

/// Hit / miss counters for a [`PostingCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostingCacheStats {
    /// Number of lookups served from the cache.
    pub hits: u64,
    /// Number of lookups that had to decode from storage.
    pub misses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a decoded list of `n` doc ids (each entry ≈ `n * 8` bytes from the
    /// doc_ids + frequencies arrays, plus fixed overhead).
    fn list(term: &str, n: u32) -> Arc<DecodedPostingList> {
        Arc::new(DecodedPostingList {
            term: term.to_string(),
            doc_ids: (0..n).collect(),
            frequencies: vec![1; n as usize],
            weights: Vec::new(),
            positions: None,
            skip_levels: Vec::new(),
            total_frequency: n as u64,
            doc_frequency: n as u64,
        })
    }

    #[test]
    fn put_then_get_returns_cached_list() {
        let cache = PostingCache::new(1 << 20);
        cache.put("body\u{1}rust".to_string(), list("rust", 4));
        let got = cache.get("body\u{1}rust").expect("present");
        assert_eq!(got.doc_ids, vec![0, 1, 2, 3]);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn miss_increments_miss_counter() {
        let cache = PostingCache::new(1 << 20);
        assert!(cache.get("absent").is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn capacity_zero_disables_cache() {
        let cache = PostingCache::new(0);
        assert!(!cache.is_enabled());
        cache.put("k".to_string(), list("k", 4));
        assert!(cache.get("k").is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn byte_budget_evicts_least_recently_used() {
        // Budget fits ~2 lists of 64 ids; a 3rd insert evicts the LRU victim.
        let one = estimated_bytes(&list("x", 64));
        let cache = PostingCache::new(one * 2 + one / 2);

        cache.put("a".to_string(), list("a", 64));
        cache.put("b".to_string(), list("b", 64));
        // Touch "a" so "b" is the LRU victim.
        assert!(cache.get("a").is_some());
        cache.put("c".to_string(), list("c", 64));

        assert!(cache.get("a").is_some(), "recently used 'a' survives");
        assert!(cache.get("c").is_some(), "just-inserted 'c' survives");
        assert!(cache.get("b").is_none(), "LRU 'b' must be evicted");
    }

    #[test]
    fn oversized_entry_is_not_cached() {
        let big = list("big", 1024);
        let cache = PostingCache::new(estimated_bytes(&big) / 2);
        cache.put("big".to_string(), big);
        assert!(
            cache.get("big").is_none(),
            "entry larger than budget is skipped"
        );
        assert!(cache.is_empty());
    }
}
