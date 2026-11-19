//! The segment readers an index keeps between the readers it builds (Issue
//! #1423).
//!
//! The index builds a new reader after every commit. A segment the commit
//! did not touch would otherwise be opened again, and everything it loads
//! on first use — above all its term dictionary — read again, so the first
//! search or `_id` lookup after a commit grew with the whole index.
//! [`SegmentReaderCache`] keeps the segment readers of the last reader built
//! and hands each to the next one:
//!
//! - A segment whose metadata and deletions are unchanged keeps its reader,
//!   posting cache included.
//! - A segment that only gained deletions gets a new reader over the same
//!   [`SegmentCore`](super::reader::SegmentCore), so it reads only its
//!   `.delmap`.
//! - A segment the cache does not hold is opened.
//!
//! A reader built earlier keeps the segment readers it was built with, so it
//! still sees exactly its own snapshot.
//!
//! A segment reader opens its files only when it first needs a part, and
//! holds no handle between reads, so that a merge can delete the files of a
//! segment it replaced, on Windows too. The files of a replaced segment
//! therefore stay until no reader can read them any more (Issue #1680): each
//! segment the cache opens gets a [`SegmentLease`], shared by every core of
//! that segment, and [`SegmentReaderCache::retire`] has the lease delete the
//! files when the last reader holding it is dropped.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock, Weak};

use ahash::{AHashMap, AHashSet};
use parking_lot::Mutex;

use crate::error::Result;
use crate::lexical::index::inverted::reader::SegmentReader;
use crate::lexical::index::inverted::segment::{SegmentInfo, SegmentKey};
use crate::storage::Storage;

/// A segment reader as the readers of an index share it.
pub(crate) type SharedSegmentReader = Arc<RwLock<SegmentReader>>;

/// Delete every file of the segment `segment_id`: the files named
/// `{segment_id}.*`, which covers a compound container, every loose part and
/// the segment's `.delmap`.
///
/// # Errors
///
/// Returns the first error from listing or deleting a file.
pub(crate) fn delete_segment_files(storage: &dyn Storage, segment_id: &str) -> Result<()> {
    let prefix = format!("{segment_id}.");
    let files = storage.list_files()?;
    for file in files.iter().filter(|f| f.starts_with(&prefix)) {
        storage.delete_file(file)?;
    }
    Ok(())
}

/// The claim the readers of one segment hold on its files (Issue #1680).
///
/// Every [`SegmentCore`](super::reader::SegmentCore) the cache opens for a
/// segment holds the segment's lease. Once a merge or a field rewrite has
/// replaced the segment, [`SegmentReaderCache::retire`] marks the lease
/// retired, and the files are deleted when the last core holding it is
/// dropped.
#[derive(Debug)]
pub(crate) struct SegmentLease {
    segment_id: String,
    storage: Arc<dyn Storage>,
    retired: AtomicBool,
}

impl SegmentLease {
    fn new(segment_id: &str, storage: &Arc<dyn Storage>) -> Self {
        Self {
            segment_id: segment_id.to_string(),
            storage: Arc::clone(storage),
            retired: AtomicBool::new(false),
        }
    }
}

impl Drop for SegmentLease {
    /// Delete the segment's files if it was retired. `Drop` cannot report an
    /// error, so a failure is logged and the files are left for the next
    /// open's orphan sweep, which removes every segment the manifest does
    /// not list. It takes no lock of the cache: it can run inside
    /// [`SegmentReaderCache::build`] or [`SegmentReaderCache::retire`].
    fn drop(&mut self) {
        if !self.retired.load(Ordering::Acquire) {
            return;
        }
        if let Err(e) = delete_segment_files(self.storage.as_ref(), &self.segment_id) {
            log::warn!(
                "cannot delete the files of segment {}, replaced by a merge or a field \
                 rewrite, once its last reader was dropped: {e}; they are left for the next \
                 open's orphan sweep",
                self.segment_id
            );
        }
    }
}

/// What the cache's lock guards.
#[derive(Debug, Default)]
struct CacheState {
    /// The segment readers of the last reader built, by segment.
    readers: AHashMap<SegmentKey, SharedSegmentReader>,

    /// The lease of every segment a core may still hold, by segment id. A
    /// retire removes the replaced segments' entries; an entry whose lease
    /// is gone otherwise (after `clear()`, or a field rewrite whose manifest
    /// sync failed) is pruned by the next retire, not by every build, which
    /// would cost a pass over the map per build.
    leases: AHashMap<String, Weak<SegmentLease>>,
}

/// The segment readers of the last reader an index built, by segment.
///
/// Owned by the [`InvertedIndex`](super::InvertedIndex) and used only by its
/// `reader()`. It holds the segments of the last snapshot only: each build
/// replaces them, and the index drops the segments a merge removed as soon as
/// it publishes the merge. It also keeps the leases of the segments its
/// readers opened (see [`SegmentLease`]).
#[derive(Debug, Default)]
pub(crate) struct SegmentReaderCache {
    state: Mutex<CacheState>,
}

impl SegmentReaderCache {
    /// The segments of a snapshot and a reader for each, keeping exactly
    /// these readers for the next build.
    ///
    /// `snapshot` is called with the cache locked, so builds take their
    /// snapshots in the order they replace the cache: a build of an older
    /// snapshot can never overwrite the readers of a newer one. A merge
    /// retires the segments it replaced under the same lock, so a build that
    /// listed them has opened their leases by then.
    ///
    /// # Arguments
    ///
    /// * `storage` - The index's storage, where a retired segment's files are
    ///   deleted.
    /// * `snapshot` - Lists the segments of the snapshot to read.
    /// * `open` - The reader of a segment, given the one the cache holds for
    ///   it, if any, and a function returning the segment's lease, for a
    ///   reader over a new core.
    ///
    /// # Returns
    ///
    /// The segments, in the order `snapshot` listed them, and their readers.
    ///
    /// # Errors
    ///
    /// Returns what `snapshot` or `open` returns. The cache then keeps the
    /// readers it held.
    pub(crate) fn build(
        &self,
        storage: &Arc<dyn Storage>,
        snapshot: impl FnOnce() -> Result<Vec<SegmentInfo>>,
        mut open: impl FnMut(
            &SegmentInfo,
            Option<&SharedSegmentReader>,
            &mut dyn FnMut() -> Arc<SegmentLease>,
        ) -> Result<SharedSegmentReader>,
    ) -> Result<(Vec<SegmentInfo>, Vec<SharedSegmentReader>)> {
        let mut state = self.state.lock();
        let segments = snapshot()?;
        #[cfg(test)]
        test_hooks::run_after_snapshot();
        let CacheState { readers, leases } = &mut *state;
        let mut kept = AHashMap::with_capacity(segments.len());
        let mut opened = Vec::with_capacity(segments.len());
        for info in &segments {
            let key = SegmentKey::of(info);
            let mut lease = || lease_of(leases, &info.segment_id, storage);
            let reader = open(info, readers.get(&key), &mut lease)?;
            kept.insert(key, Arc::clone(&reader));
            opened.push(reader);
        }
        *readers = kept;
        Ok((segments, opened))
    }

    /// Drop the readers of segments not in `segments`, such as the ones a
    /// merge has just replaced, so their parts are freed once no reader uses
    /// them.
    pub(crate) fn retain(&self, segments: &[SegmentInfo]) {
        let live: AHashSet<SegmentKey> = segments.iter().map(SegmentKey::of).collect();
        self.state
            .lock()
            .readers
            .retain(|key, _| live.contains(key));
    }

    /// Drop the readers of segments not in `live`, as [`Self::retain`]
    /// does, and retire the segments `replaced`, which a merge or a field
    /// rewrite has replaced in the published manifest (Issue #1680).
    ///
    /// A replaced segment that some reader can still read keeps its files
    /// until the last such reader is dropped; its lease deletes them then.
    /// When this call held the last reference, they are deleted before it
    /// returns.
    ///
    /// # Returns
    ///
    /// The ids of the replaced segments no reader can read, whose files the
    /// caller deletes.
    pub(crate) fn retire(&self, live: &[SegmentInfo], replaced: &[String]) -> Vec<String> {
        let live: AHashSet<SegmentKey> = live.iter().map(SegmentKey::of).collect();
        let mut held = Vec::new();
        let mut unread = Vec::new();
        {
            let mut state = match self.state.try_lock() {
                Some(state) => state,
                None => {
                    #[cfg(test)]
                    test_hooks::run_when_retire_waits();
                    self.state.lock()
                }
            };
            state.readers.retain(|key, _| live.contains(key));
            for segment_id in replaced {
                match state
                    .leases
                    .remove(segment_id)
                    .and_then(|lease| lease.upgrade())
                {
                    Some(lease) => {
                        lease.retired.store(true, Ordering::Release);
                        held.push(lease);
                    }
                    None => unread.push(segment_id.clone()),
                }
            }
            state.leases.retain(|_, lease| lease.strong_count() > 0);
        }
        // After the lock is released: a lease only this call still held
        // deletes the files here, and `Drop` must not run under the lock.
        drop(held);
        unread
    }

    /// Drop every reader. The leases stay, so a merge that publishes later
    /// still finds the segments readers built before may read.
    pub(crate) fn clear(&self) {
        self.state.lock().readers.clear();
    }

    /// The number of segments the cache holds a reader for.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.state.lock().readers.len()
    }
}

/// The live lease of `segment_id` in `leases`, or a new one registered there.
fn lease_of(
    leases: &mut AHashMap<String, Weak<SegmentLease>>,
    segment_id: &str,
    storage: &Arc<dyn Storage>,
) -> Arc<SegmentLease> {
    if let Some(lease) = leases.get(segment_id).and_then(Weak::upgrade) {
        return lease;
    }
    let lease = Arc::new(SegmentLease::new(segment_id, storage));
    leases.insert(segment_id.to_string(), Arc::downgrade(&lease));
    lease
}

/// Points where a test can act in the middle of a reader build or a retire
/// on the same thread (Issue #1680). Each hook runs once, in the next build
/// or retire that reaches its point.
#[cfg(test)]
pub(crate) mod test_hooks {
    use std::cell::RefCell;

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static AFTER_SNAPSHOT: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static WHEN_RETIRE_WAITS: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static AFTER_MERGE: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    fn run(slot: &'static std::thread::LocalKey<RefCell<Option<Hook>>>) {
        let hook = slot.with(|slot| slot.borrow_mut().take());
        if let Some(hook) = hook {
            hook();
        }
    }

    /// Run `hook` in the next build on this thread, right after it has
    /// listed the segments and before it opens them, with the cache locked.
    pub(crate) fn set_after_snapshot(hook: impl FnOnce() + 'static) {
        AFTER_SNAPSHOT.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    pub(super) fn run_after_snapshot() {
        run(&AFTER_SNAPSHOT);
    }

    /// Run `hook` when a retire on this thread has to wait for the cache's
    /// lock, before it waits.
    pub(crate) fn set_when_retire_waits(hook: impl FnOnce() + 'static) {
        WHEN_RETIRE_WAITS.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    pub(super) fn run_when_retire_waits() {
        run(&WHEN_RETIRE_WAITS);
    }

    /// Run `hook` when the next merge on this thread has published and
    /// retired or deleted the segments it replaced.
    pub(crate) fn set_after_merge(hook: impl FnOnce() + 'static) {
        AFTER_MERGE.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    pub(crate) fn run_after_merge() {
        run(&AFTER_MERGE);
    }
}
