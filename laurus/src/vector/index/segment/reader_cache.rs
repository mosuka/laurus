//! Per-segment reader cache shared across segment-per-commit vector index
//! types (Issue #889; originated in HNSW's segment-per-commit design,
//! Issue #634).
//!
//! Before this cache existed, every search against a segmented index
//! reloaded every managed segment from disk on each call, re-parsing the
//! segment header and rebuilding its in-memory search state. For a
//! multi-segment index the per-query I/O often dominated the actual search
//! cost.
//!
//! This cache holds an `Arc<R>` per `segment_id` (`R` is the index type's
//! concrete reader — `HnswIndexReader`, `FlatVectorIndexReader`,
//! `IvfIndexReader`) so that the second and subsequent searches against the
//! same segment hit memory only. Entries are invalidated by the owning
//! index's merge path immediately after the segment manager removes a
//! source segment as part of a merge, and dropped all at once when the
//! index's deletion bitmap is created (see
//! `SegmentedReaderCache::sealed_readers_newest_first`).
//!
//! Issue [#660](https://github.com/mosuka/laurus/issues/660). Reuses the
//! storage-level per-segment input cache landed in
//! [#522](https://github.com/mosuka/laurus/issues/522); the two layers are
//! complementary — #522 caches the open file handle, #660 caches the fully
//! parsed reader state on top.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ahash::AHashMap;
use parking_lot::RwLock;

use crate::error::Result;
use crate::maintenance::deletion::DeletionBitmap;
use crate::vector::index::segment::segmented_core::SegmentedCore;

/// Caches `Arc<R>` per `segment_id`.
///
/// The cache is intentionally minimal: it stores every requested reader for
/// the lifetime of the owning index (i.e. no LRU eviction or memory
/// budget). A future change can swap the inner map for an LRU-backed
/// implementation without touching call sites.
#[derive(Debug)]
pub struct SegmentedReaderCache<R> {
    inner: RwLock<AHashMap<String, Arc<R>>>,

    /// The [`SegmentedCore::bitmap_epoch`] the cached readers were loaded
    /// under; only maintained by [`Self::sealed_readers_newest_first`].
    bitmap_epoch: AtomicU64,
}

impl<R> Default for SegmentedReaderCache<R> {
    // Not derived: `#[derive(Default)]` on a generic struct adds an `R:
    // Default` bound, but `R` here is only ever stored behind an `Arc` and
    // never itself defaulted.
    fn default() -> Self {
        Self {
            inner: RwLock::new(AHashMap::default()),
            bitmap_epoch: AtomicU64::new(0),
        }
    }
}

impl<R> SegmentedReaderCache<R> {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the cached reader for `segment_id`, invoking `loader` on miss.
    ///
    /// Concurrent callers requesting the same `segment_id` may race; the
    /// double-checked lock pattern below ensures only one of them runs
    /// `loader`, the rest pick up the already-inserted entry.
    ///
    /// # Arguments
    ///
    /// * `segment_id` - The segment identifier to look up.
    /// * `loader` - Invoked only on a cache miss; returns the freshly-built
    ///   reader.
    ///
    /// # Returns
    ///
    /// `Ok(Arc::clone(&reader))` if the reader is found or successfully
    /// loaded. Propagates `loader`'s error otherwise.
    pub fn get_or_load<F>(&self, segment_id: &str, loader: F) -> Result<Arc<R>>
    where
        F: FnOnce() -> Result<R>,
    {
        if let Some(reader) = self.inner.read().get(segment_id) {
            return Ok(Arc::clone(reader));
        }

        // Slow path: take the write lock and re-check before loading so that
        // concurrent misses on the same `segment_id` don't load twice.
        let mut guard = self.inner.write();
        if let Some(reader) = guard.get(segment_id) {
            return Ok(Arc::clone(reader));
        }

        let reader = Arc::new(loader()?);
        guard.insert(segment_id.to_string(), Arc::clone(&reader));
        Ok(reader)
    }

    /// Remove the cached reader for `segment_id`.
    ///
    /// Safe to call for unknown segment ids — a missing entry is a no-op.
    pub fn invalidate(&self, segment_id: &str) {
        self.inner.write().remove(segment_id);
    }

    /// Remove all cached readers.
    ///
    /// `sealed_readers_newest_first` calls it when the deletion bitmap is
    /// created, so readers loaded without it are rebuilt.
    pub fn clear(&self) {
        self.inner.write().clear();
    }

    /// Readers of `core`'s sealed segments, newest generation first, each
    /// loaded with the current deletion bitmap attached.
    ///
    /// `load` builds the reader of one segment and attaches the bitmap when
    /// it is `Some`. Segment readers filter deletions through the bitmap
    /// they hold, so a reader loaded before the bitmap existed would miss
    /// every later deletion. Readers cached under an older
    /// [`SegmentedCore::bitmap_epoch`] are therefore dropped before the
    /// build, and the build is redone if a bitmap is created while it runs.
    ///
    /// # Errors
    ///
    /// Returns the first error from reading the bitmap or from `load`.
    pub(crate) fn sealed_readers_newest_first(
        &self,
        core: &SegmentedCore,
        load: impl Fn(&str, Option<Arc<DeletionBitmap>>) -> Result<R>,
    ) -> Result<Vec<Arc<R>>> {
        let mut segments = core.manager().list_segments();
        segments.sort_by_key(|s| std::cmp::Reverse(s.generation));

        loop {
            let epoch = core.bitmap_epoch();
            if self.bitmap_epoch.load(Ordering::Acquire) != epoch {
                // Clear before publishing the epoch: a concurrent build that
                // sees the new epoch must not find a reader cached under the
                // old one. A build that read a stale epoch may move it back;
                // that only costs another clear.
                self.clear();
                self.bitmap_epoch.store(epoch, Ordering::Release);
            }

            let bitmap = core.bitmap()?;
            let readers = segments
                .iter()
                .map(|info| {
                    self.get_or_load(&info.segment_id, || load(&info.segment_id, bitmap.clone()))
                })
                .collect::<Result<Vec<_>>>()?;

            // A bitmap created while this build ran is missing from the
            // readers loaded under the `None` snapshot, which may already be
            // cached: drop them and rebuild. Creation happens at most once
            // between `clear_deletions` calls, so the loop rarely repeats.
            if core.bitmap_epoch() == epoch {
                return Ok(readers);
            }
            self.clear();
        }
    }

    /// Return the number of cached entries.
    ///
    /// Primarily exposed for tests. The value may be stale by the time the
    /// caller reads it because other threads may concurrently mutate the
    /// cache, so do not rely on it for control flow.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Return `true` if the cache currently holds no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Return `true` if the cache currently holds an entry for `segment_id`.
    ///
    /// Primarily exposed for tests.
    pub fn contains(&self, segment_id: &str) -> bool {
        self.inner.read().contains_key(segment_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LaurusError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The cache only stores `Arc<R>` and these tests don't need the reader
    /// to be functional — they only need to assert that the loader is
    /// invoked the right number of times for a given access pattern.
    ///
    /// Instead, return an `Err` from the loader and assert the call count.
    /// `get_or_load` propagates the error without inserting anything, so we
    /// can use call counters to distinguish hits from misses. `()` stands in
    /// for a concrete reader type here.

    #[test]
    fn get_or_load_invokes_loader_on_miss_only() {
        let cache: SegmentedReaderCache<()> = SegmentedReaderCache::new();
        let calls = AtomicUsize::new(0);

        // First call: loader is invoked, returns Err so nothing is cached.
        let r1 = cache.get_or_load("seg-a", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(LaurusError::other("intentional miss"))
        });
        assert!(r1.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(cache.len(), 0, "no entry inserted on loader error");

        // Second call: same segment, loader still invoked (no entry cached).
        let r2 = cache.get_or_load("seg-a", || {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(LaurusError::other("still missing"))
        });
        assert!(r2.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn invalidate_is_noop_for_unknown_segment() {
        let cache: SegmentedReaderCache<()> = SegmentedReaderCache::new();
        cache.invalidate("never-cached");
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn clear_empties_the_cache() {
        let cache: SegmentedReaderCache<()> = SegmentedReaderCache::new();
        assert!(cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn contains_reports_membership() {
        let cache: SegmentedReaderCache<()> = SegmentedReaderCache::new();
        assert!(!cache.contains("seg-a"));
        // No way to insert via the public API without a real reader; that
        // case is covered by the integration test in `segmented_field`.
    }

    #[test]
    fn get_or_load_caches_successful_loads() {
        let cache: SegmentedReaderCache<u32> = SegmentedReaderCache::new();
        let calls = AtomicUsize::new(0);
        let load = || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(42)
        };

        let r1 = cache.get_or_load("seg-a", load).unwrap();
        assert_eq!(*r1, 42);
        assert_eq!(cache.len(), 1);

        let r2 = cache.get_or_load("seg-a", load).unwrap();
        assert_eq!(*r2, 42);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "second call must hit the cache, not reinvoke the loader"
        );
    }

    mod sealed_readers {
        use std::sync::atomic::AtomicBool;

        use super::*;
        use crate::storage::Storage;
        use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
        use crate::vector::index::segment::manager::{
            ManagedSegmentInfo, SegmentFileLayout, SegmentManager, SegmentManagerConfig,
        };

        const TEST_LAYOUT: SegmentFileLayout = SegmentFileLayout {
            primary: ".test",
            sidecars: &[],
            tmp: ".test.tmp",
        };

        /// A core over `count` empty segments, added oldest first.
        fn core_with_segments(count: usize) -> (SegmentedCore, Vec<String>) {
            let storage: Arc<dyn Storage> =
                Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            let manager = Arc::new(
                SegmentManager::new(
                    SegmentManagerConfig::default(),
                    storage.clone(),
                    TEST_LAYOUT,
                )
                .unwrap(),
            );
            let mut ids = Vec::new();
            for _ in 0..count {
                let id = manager.generate_segment_id();
                storage
                    .create_output(&format!("{id}.test"))
                    .unwrap()
                    .close()
                    .unwrap();
                manager
                    .add_segment(ManagedSegmentInfo::new(id.clone(), 1, 0, 0))
                    .unwrap();
                ids.push(id);
            }
            (SegmentedCore::new("index", storage, manager), ids)
        }

        /// Readers are `true` when they were loaded with a bitmap attached.
        fn build(
            cache: &SegmentedReaderCache<bool>,
            core: &SegmentedCore,
            loads: &AtomicUsize,
        ) -> Vec<bool> {
            cache
                .sealed_readers_newest_first(core, |_, bitmap| {
                    loads.fetch_add(1, Ordering::SeqCst);
                    Ok(bitmap.is_some())
                })
                .unwrap()
                .iter()
                .map(|reader| **reader)
                .collect()
        }

        #[test]
        fn readers_come_newest_generation_first() {
            let (core, ids) = core_with_segments(3);
            let cache: SegmentedReaderCache<String> = SegmentedReaderCache::new();
            let readers = cache
                .sealed_readers_newest_first(&core, |id, _| Ok(id.to_string()))
                .unwrap();
            let order: Vec<String> = readers.iter().map(|r| r.to_string()).collect();
            let expected: Vec<String> = ids.into_iter().rev().collect();
            assert_eq!(order, expected);
        }

        #[test]
        fn readers_cached_before_the_bitmap_are_rebuilt_once() {
            let (core, _) = core_with_segments(2);
            let cache = SegmentedReaderCache::new();
            let loads = AtomicUsize::new(0);

            assert_eq!(build(&cache, &core, &loads), vec![false, false]);
            assert_eq!(build(&cache, &core, &loads), vec![false, false]);
            assert_eq!(loads.load(Ordering::SeqCst), 2, "the second build hits");

            core.mark_deleted(7).unwrap();
            assert_eq!(
                build(&cache, &core, &loads),
                vec![true, true],
                "readers loaded without the bitmap must be rebuilt"
            );
            assert_eq!(build(&cache, &core, &loads), vec![true, true]);
            assert_eq!(loads.load(Ordering::SeqCst), 4, "rebuilt exactly once");
        }

        #[test]
        fn a_bitmap_recreated_after_clear_deletions_rebuilds_again() {
            let (core, _) = core_with_segments(1);
            let cache = SegmentedReaderCache::new();
            let loads = AtomicUsize::new(0);
            core.mark_deleted(1).unwrap();
            assert_eq!(build(&cache, &core, &loads), vec![true]);

            // Optimize drops the bitmap; until the next delete, the readers
            // keep the old one (nothing they hold is still deleted).
            core.clear_deletions().unwrap();
            assert_eq!(build(&cache, &core, &loads), vec![true]);
            assert_eq!(loads.load(Ordering::SeqCst), 1);

            core.mark_deleted(2).unwrap();
            build(&cache, &core, &loads);
            assert_eq!(
                loads.load(Ordering::SeqCst),
                2,
                "the new bitmap must reach the readers"
            );
        }

        #[test]
        fn a_bitmap_created_during_the_build_restarts_it() {
            let (core, _) = core_with_segments(2);
            let cache = SegmentedReaderCache::new();
            let created = AtomicBool::new(false);

            // The first load races a first delete: it was handed the `None`
            // snapshot, but the bitmap exists once it returns.
            let readers = cache
                .sealed_readers_newest_first(&core, |_, bitmap| {
                    if !created.swap(true, Ordering::SeqCst) {
                        core.mark_deleted(1).unwrap();
                    }
                    Ok(bitmap.is_some())
                })
                .unwrap();
            assert!(readers.iter().all(|reader| **reader));
            assert_eq!(cache.len(), 2);
        }
    }
}
