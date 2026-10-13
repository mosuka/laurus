//! Parsed-query (DSL) cache (Issue
//! [#590](https://github.com/mosuka/laurus/issues/590)).
//!
//! Every `LexicalSearchQuery::Dsl(string)` is otherwise re-parsed on each
//! search — the pest grammar runs and the analyzer re-tokenises the query
//! terms. Autocomplete / popular-query servers pay that cost per call.
//! [`ParsedQueryCache`] memoises `dsl string -> Arc<dyn Query>` so a repeated
//! DSL string is parsed once and then reused via
//! [`Query::clone_box`](crate::lexical::query::Query::clone_box) (a cheap
//! refcount bump for boolean clause subtrees, #413).
//!
//! # Lifetime and key
//!
//! The cache lives on
//! [`InvertedIndexSearcher`](crate::lexical::index::inverted::searcher::InvertedIndexSearcher),
//! which the store rebuilds on every `commit()` / `optimize()` / `refresh()`.
//! The analyzer and `default_fields` are fixed for that searcher's lifetime, so
//! the DSL string alone keys the cache; a schema / analyzer change yields a
//! fresh searcher with an empty cache (no manual invalidation).

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lru::LruCache;
use parking_lot::Mutex;

use crate::lexical::query::Query;

/// A bounded LRU cache mapping a DSL query string to its parsed
/// `Arc<dyn Query>` for one searcher snapshot.
///
/// A [`Mutex`] (not an `RwLock`) guards the map because [`LruCache::get`] takes
/// `&mut self` to update recency; the critical section is a single map probe
/// plus an [`Arc`] clone. When the configured capacity is zero the cache is
/// disabled and every operation is a no-op.
#[derive(Debug)]
pub struct ParsedQueryCache {
    /// The LRU map, or `None` when caching is disabled (capacity 0).
    inner: Option<Mutex<LruCache<String, Arc<dyn Query>>>>,
    /// Number of lookups served from the cache.
    hits: AtomicU64,
    /// Number of lookups that missed (including lookups while disabled).
    misses: AtomicU64,
}

impl ParsedQueryCache {
    /// Create a cache holding up to `capacity` parsed queries.
    ///
    /// A `capacity` of `0` disables the cache: [`get`](Self::get) always misses
    /// and [`put`](Self::put) is a no-op, so callers always parse fresh.
    ///
    /// The map starts empty and grows with the entries it holds; nothing is
    /// reserved for `capacity` up front (Issue #1431). The per-segment search
    /// fanout builds a searcher, and with it this cache, for every segment of
    /// every query.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of `(dsl string, parsed query)` entries to
    ///   retain. The least-recently-used entry is evicted when full.
    pub fn new(capacity: usize) -> Self {
        let inner = NonZeroUsize::new(capacity).map(|c| Mutex::new(LruCache::sparse(c)));
        ParsedQueryCache {
            inner,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Look up the parsed query cached for `dsl`, bumping its recency on a hit.
    ///
    /// Records a hit or miss in the statistics. Returns `None` when the DSL
    /// string is absent or the cache is disabled; the cloned [`Arc`] on a hit
    /// is a refcount bump, not a re-parse.
    ///
    /// # Arguments
    ///
    /// * `dsl` - The DSL query string.
    pub fn get(&self, dsl: &str) -> Option<Arc<dyn Query>> {
        let hit = self
            .inner
            .as_ref()
            .and_then(|inner| inner.lock().get(dsl).cloned());
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    /// Insert a parsed query for `dsl`, evicting the least-recently-used entry
    /// when full. A no-op when the cache is disabled.
    ///
    /// # Arguments
    ///
    /// * `dsl` - The DSL query string.
    /// * `query` - The parsed query to share.
    pub fn put(&self, dsl: String, query: Arc<dyn Query>) {
        if let Some(inner) = self.inner.as_ref() {
            inner.lock().put(dsl, query);
        }
    }

    /// Returns `true` if caching is enabled (capacity was non-zero).
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Snapshot of the cache hit / miss counters.
    pub fn stats(&self) -> ParsedQueryCacheStats {
        ParsedQueryCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

/// Hit / miss counters for a [`ParsedQueryCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedQueryCacheStats {
    /// Number of lookups served from the cache.
    pub hits: u64,
    /// Number of lookups that had to parse the DSL.
    pub misses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::query::term::TermQuery;

    fn term(field: &str, t: &str) -> Arc<dyn Query> {
        Arc::new(TermQuery::new(field, t))
    }

    #[test]
    fn put_then_get_returns_cached_query() {
        let cache = ParsedQueryCache::new(4);
        cache.put("title:rust".to_string(), term("title", "rust"));

        let got = cache.get("title:rust").expect("entry should be present");
        assert_eq!(got.description(), "title:rust");
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn miss_increments_miss_counter() {
        let cache = ParsedQueryCache::new(4);
        assert!(cache.get("absent:x").is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn capacity_zero_disables_cache() {
        let cache = ParsedQueryCache::new(0);
        assert!(!cache.is_enabled());
        cache.put("title:rust".to_string(), term("title", "rust"));
        assert!(cache.get("title:rust").is_none());
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let cache = ParsedQueryCache::new(2);
        cache.put("a:1".to_string(), term("a", "1"));
        cache.put("b:1".to_string(), term("b", "1"));
        // Touch "a:1" so "b:1" becomes the LRU victim.
        assert!(cache.get("a:1").is_some());
        cache.put("c:1".to_string(), term("c", "1"));

        assert!(cache.get("a:1").is_some(), "recently used");
        assert!(cache.get("c:1").is_some(), "just inserted");
        assert!(cache.get("b:1").is_none(), "should have been evicted");
    }

    /// Nothing is reserved for the capacity (Issue #1431): a `usize::MAX`
    /// capacity, whose up-front table could never be allocated, still
    /// caches.
    #[test]
    fn capacity_is_not_reserved_up_front() {
        let cache = ParsedQueryCache::new(usize::MAX);
        cache.put("title:rust".to_string(), term("title", "rust"));

        let got = cache.get("title:rust").expect("entry should be present");
        assert_eq!(got.description(), "title:rust");
    }
}
