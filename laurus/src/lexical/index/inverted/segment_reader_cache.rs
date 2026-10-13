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

use std::sync::{Arc, RwLock};

use ahash::{AHashMap, AHashSet};
use parking_lot::Mutex;

use crate::error::Result;
use crate::lexical::index::inverted::reader::SegmentReader;
use crate::lexical::index::inverted::segment::{SegmentInfo, SegmentKey};

/// A segment reader as the readers of an index share it.
pub(crate) type SharedSegmentReader = Arc<RwLock<SegmentReader>>;

/// The segment readers of the last reader an index built, by segment.
///
/// Owned by the [`InvertedIndex`](super::InvertedIndex) and used only by its
/// `reader()`. It holds the segments of the last snapshot only: each build
/// replaces them, and the index drops the segments a merge removed as soon as
/// it publishes the merge.
#[derive(Debug, Default)]
pub(crate) struct SegmentReaderCache {
    readers: Mutex<AHashMap<SegmentKey, SharedSegmentReader>>,
}

impl SegmentReaderCache {
    /// The segments of a snapshot and a reader for each, keeping exactly
    /// these readers for the next build.
    ///
    /// `snapshot` is called with the cache locked, so builds take their
    /// snapshots in the order they replace the cache: a build of an older
    /// snapshot can never overwrite the readers of a newer one.
    ///
    /// # Arguments
    ///
    /// * `snapshot` - Lists the segments of the snapshot to read.
    /// * `open` - The reader of a segment, given the one the cache holds for
    ///   it, if any.
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
        snapshot: impl FnOnce() -> Result<Vec<SegmentInfo>>,
        mut open: impl FnMut(&SegmentInfo, Option<&SharedSegmentReader>) -> Result<SharedSegmentReader>,
    ) -> Result<(Vec<SegmentInfo>, Vec<SharedSegmentReader>)> {
        let mut readers = self.readers.lock();
        let segments = snapshot()?;
        let mut kept = AHashMap::with_capacity(segments.len());
        let mut opened = Vec::with_capacity(segments.len());
        for info in &segments {
            let key = SegmentKey::of(info);
            let reader = open(info, readers.get(&key))?;
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
        self.readers.lock().retain(|key, _| live.contains(key));
    }

    /// Drop every reader.
    pub(crate) fn clear(&self) {
        self.readers.lock().clear();
    }

    /// The number of segments the cache holds a reader for.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.readers.lock().len()
    }
}
