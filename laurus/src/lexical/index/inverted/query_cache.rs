//! Snapshot-scoped query / filter result cache (Issue
//! [#578](https://github.com/mosuka/laurus/issues/578)).
//!
//! Filter clauses (tenancy, category, status flags, …) are frequently reused
//! across many search requests, yet each request re-decodes every posting list
//! to re-evaluate them. [`QueryFilterCache`] memoises the **set of document
//! ids** a query matches so a repeated filter becomes a single map probe plus
//! an [`Arc`] clone instead of a full posting walk.
//!
//! # Lifetime and invalidation
//!
//! The cache lives on
//! [`InvertedIndexReader`](crate::lexical::index::inverted::reader::InvertedIndexReader),
//! which is rebuilt on every `commit()` / `optimize()` / `refresh()` (the
//! cached searcher in [`LexicalStore`](crate::lexical::store::LexicalStore) is
//! dropped). Each reader is therefore a point-in-time snapshot and its cache
//! requires **no explicit invalidation** — when the index changes, a fresh
//! reader starts with an empty cache. This mirrors Lucene's per-reader cache
//! model.
//!
//! # What is safe to cache
//!
//! Entries are keyed by [`Query::cache_key`](crate::lexical::query::Query::cache_key),
//! which returns `Some(key)` only for queries whose matched set is canonically
//! identified by the key (and `None` otherwise, bypassing the cache). The
//! cached set is **score-independent**, so it is sound for filters and counts.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lru::LruCache;
use parking_lot::Mutex;
use roaring::RoaringTreemap;

use crate::error::Result;
use crate::lexical::query::matcher::Matcher;

/// Drain a fully-constructed matcher into a [`RoaringTreemap`] of document ids.
///
/// Shared by the reader's cache-backed
/// [`matching_doc_ids`](crate::lexical::index::inverted::reader::InvertedIndexReader::matching_doc_ids)
/// and the searcher's uncached fallback so the matcher-iteration protocol lives
/// in one place. Mirrors the searcher's scoring loop: the matcher is
/// pre-positioned at its first match, so `doc_id` is read before each `next`,
/// and `u64::MAX` marks exhaustion. Deleted documents are already filtered at
/// the posting-iterator level, so they never appear here.
///
/// # Arguments
///
/// * `matcher` - A matcher positioned at its first match (as returned by
///   [`Query::matcher`](crate::lexical::query::Query::matcher)).
///
/// # Returns
///
/// The set of document ids the matcher yields.
pub(crate) fn drain_matcher(mut matcher: Box<dyn Matcher>) -> Result<RoaringTreemap> {
    let mut bitmap = RoaringTreemap::new();
    while !matcher.is_exhausted() {
        let doc_id = matcher.doc_id();
        if doc_id == u64::MAX {
            break;
        }
        bitmap.insert(doc_id);
        if !matcher.next()? {
            break;
        }
    }
    Ok(bitmap)
}

/// A bounded LRU cache mapping a query's canonical cache key to the set of
/// document ids it matches within one reader snapshot.
///
/// A [`Mutex`] (not an `RwLock`) guards the map because [`LruCache::get`] takes
/// `&mut self` to update recency; critical sections are a single map probe plus
/// an [`Arc`] clone. When the configured capacity is zero the cache is disabled
/// and every operation is a no-op.
#[derive(Debug)]
pub struct QueryFilterCache {
    /// The LRU map, or `None` when caching is disabled (capacity 0).
    inner: Option<Mutex<LruCache<String, Arc<RoaringTreemap>>>>,
    /// Number of lookups that returned a cached set.
    hits: AtomicU64,
    /// Number of lookups that missed (including lookups while disabled).
    misses: AtomicU64,
}

impl QueryFilterCache {
    /// Create a cache holding up to `capacity` entries.
    ///
    /// A `capacity` of `0` disables the cache: [`get`](Self::get) always misses
    /// and [`put`](Self::put) is a no-op, so callers always compute fresh.
    ///
    /// The map starts empty and grows with the entries it holds; nothing is
    /// reserved for `capacity` up front (Issue #1431), so building a reader
    /// costs the same for any capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of `(query key, doc-id set)` entries to
    ///   retain. The least-recently-used entry is evicted when full.
    pub fn new(capacity: usize) -> Self {
        let inner = NonZeroUsize::new(capacity).map(|c| Mutex::new(LruCache::sparse(c)));
        QueryFilterCache {
            inner,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Returns `true` if caching is enabled (capacity was non-zero).
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Look up the doc-id set cached for `key`, bumping its recency on a hit.
    ///
    /// Records a hit or miss in the cache statistics. Returns `None` when the
    /// key is absent or the cache is disabled; the cloned [`Arc`] on a hit is a
    /// refcount bump, not a copy of the bitmap.
    ///
    /// # Arguments
    ///
    /// * `key` - The query's canonical cache key.
    ///
    /// # Returns
    ///
    /// `Some(set)` on a hit, `None` on a miss.
    pub fn get(&self, key: &str) -> Option<Arc<RoaringTreemap>> {
        let hit = self
            .inner
            .as_ref()
            .and_then(|inner| inner.lock().get(key).cloned());
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    /// Insert a doc-id set for `key`, evicting the least-recently-used entry
    /// when full. A no-op when the cache is disabled.
    ///
    /// # Arguments
    ///
    /// * `key` - The query's canonical cache key.
    /// * `value` - The set of matching document ids for this snapshot.
    pub fn put(&self, key: String, value: Arc<RoaringTreemap>) {
        if let Some(inner) = self.inner.as_ref() {
            inner.lock().put(key, value);
        }
    }

    /// Number of entries currently cached (0 when disabled).
    pub fn len(&self) -> usize {
        self.inner.as_ref().map_or(0, |inner| inner.lock().len())
    }

    /// Returns `true` if the cache holds no entries (or is disabled).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot of the cache hit / miss counters.
    pub fn stats(&self) -> QueryFilterCacheStats {
        QueryFilterCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

/// Hit / miss counters for a [`QueryFilterCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryFilterCacheStats {
    /// Number of lookups served from the cache.
    pub hits: u64,
    /// Number of lookups that had to compute the result.
    pub misses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(ids: &[u64]) -> Arc<RoaringTreemap> {
        Arc::new(ids.iter().copied().collect())
    }

    #[test]
    fn put_then_get_returns_same_set() {
        let cache = QueryFilterCache::new(4);
        cache.put("k".to_string(), set(&[1, 2, 3]));

        let got = cache.get("k").expect("entry should be present");
        assert_eq!(got.iter().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn miss_increments_miss_counter() {
        let cache = QueryFilterCache::new(4);
        assert!(cache.get("absent").is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn capacity_zero_disables_cache() {
        let cache = QueryFilterCache::new(0);
        assert!(!cache.is_enabled());
        cache.put("k".to_string(), set(&[1]));
        // Disabled cache never retains anything.
        assert!(cache.get("k").is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let cache = QueryFilterCache::new(2);
        cache.put("a".to_string(), set(&[1]));
        cache.put("b".to_string(), set(&[2]));
        // Touch "a" so "b" becomes the LRU victim.
        assert!(cache.get("a").is_some());
        cache.put("c".to_string(), set(&[3]));

        assert!(cache.get("a").is_some(), "a was recently used");
        assert!(cache.get("c").is_some(), "c was just inserted");
        assert!(cache.get("b").is_none(), "b should have been evicted");
        assert_eq!(cache.len(), 2);
    }

    /// Nothing is reserved for the capacity (Issue #1431): a `usize::MAX`
    /// capacity, whose up-front table could never be allocated, still
    /// caches.
    #[test]
    fn capacity_is_not_reserved_up_front() {
        let cache = QueryFilterCache::new(usize::MAX);
        cache.put("k".to_string(), set(&[1, 2]));

        let got = cache.get("k").expect("entry should be present");
        assert_eq!(got.iter().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(cache.len(), 1);
    }
}
