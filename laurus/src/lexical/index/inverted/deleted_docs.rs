//! The deleted documents a segment reader sees (Issue #1301).

use roaring::RoaringTreemap;

/// A segment reader's deleted documents: its `.delmap`, read when the reader
/// opens and fixed for the reader's life.
///
/// Unlike the writer's [`DeletionBitmap`](crate::maintenance::deletion::DeletionBitmap),
/// which deletions keep updating behind a lock, nothing writes to this, so a
/// check takes no lock — it runs per scored document and per decoded posting.
#[derive(Debug)]
pub(crate) struct DeletedDocs {
    /// Every deleted id.
    ids: RoaringTreemap,
}

impl DeletedDocs {
    /// The deleted `ids` of a segment.
    pub(crate) fn new(ids: RoaringTreemap) -> Self {
        DeletedDocs { ids }
    }

    /// Whether `doc_id` is marked deleted.
    #[inline]
    pub(crate) fn contains(&self, doc_id: u64) -> bool {
        self.ids.contains(doc_id)
    }

    /// Every deleted id, for set operations such as counting the deletions
    /// that fall on documents the segment holds.
    pub(crate) fn ids(&self) -> &RoaringTreemap {
        &self.ids
    }
}
