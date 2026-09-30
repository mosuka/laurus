//! The deleted documents a segment reader sees (Issue #1301).

use roaring::RoaringTreemap;

use crate::lexical::index::inverted::segment::range_width;

/// The most bits per document a segment holds that its bitset may take: one
/// byte per document, the size of one `.norms` column. Doc ids are global, so
/// a range with many gaps (such as a merge of segments that are not adjacent)
/// can be far wider than the segment; such a segment keeps only the treemap.
const MAX_BITS_PER_DOC: u64 = 8;

/// A segment reader's deleted documents: its `.delmap`, read when the reader
/// opens and fixed for the reader's life.
///
/// Unlike the writer's [`DeletionBitmap`](crate::maintenance::deletion::DeletionBitmap),
/// which deletions keep updating behind a lock, nothing writes to this, so a
/// check takes no lock — it runs per scored document and per decoded posting.
/// Inside the segment's range a check tests one bit of a plain bitset, as
/// Lucene's `FixedBitSet` and Tantivy's `AliveBitSet` do, instead of searching
/// the Roaring containers. The treemap stays for the set operations (counts,
/// the live ids) and for the ids the bitset does not cover.
#[derive(Debug)]
pub(crate) struct DeletedDocs {
    /// Every deleted id.
    ids: RoaringTreemap,
    /// `ids` from the segment's `min_doc_id` on, one bit per id; `None` when
    /// the range is too wide for the documents it holds.
    bits: Option<RangeBits>,
}

impl DeletedDocs {
    /// The deleted `ids` of a segment holding `doc_count` documents in
    /// `[min_doc_id, max_doc_id]`.
    pub(crate) fn new(
        ids: RoaringTreemap,
        min_doc_id: u64,
        max_doc_id: u64,
        doc_count: u64,
    ) -> Self {
        let bits = range_width(min_doc_id, max_doc_id)
            .filter(|&width| width <= doc_count.saturating_mul(MAX_BITS_PER_DOC))
            .and_then(|width| RangeBits::new(&ids, min_doc_id, width));
        DeletedDocs { ids, bits }
    }

    /// Whether `doc_id` is marked deleted.
    #[inline]
    pub(crate) fn contains(&self, doc_id: u64) -> bool {
        match self.bits.as_ref().and_then(|bits| bits.get(doc_id)) {
            Some(deleted) => deleted,
            None => self.contains_uncovered(doc_id),
        }
    }

    /// [`Self::contains`] for an id the bitset does not cover: one outside the
    /// segment's range, or any id of a segment without a bitset. Kept out of
    /// line: inlined, the treemap search bloats the per-document callers.
    #[cold]
    #[inline(never)]
    fn contains_uncovered(&self, doc_id: u64) -> bool {
        self.ids.contains(doc_id)
    }

    /// Every deleted id, for set operations such as counting the deletions
    /// that fall on documents the segment holds.
    pub(crate) fn ids(&self) -> &RoaringTreemap {
        &self.ids
    }

    /// Whether checks inside the range test the bitset.
    #[cfg(test)]
    pub(crate) fn has_bits(&self) -> bool {
        self.bits.is_some()
    }
}

/// The deleted ids of `[min_doc_id, min_doc_id + 64 × words.len())`, one bit
/// per id. The span covers the segment's range rounded up to whole words; the
/// bits past `max_doc_id` mirror the treemap too, so every covered id answers
/// exactly as the treemap does.
#[derive(Debug)]
struct RangeBits {
    min_doc_id: u64,
    words: Box<[u64]>,
}

impl RangeBits {
    /// The bits of `ids` over at least `width` ids from `min_doc_id`; `None`
    /// when the words cannot be addressed.
    fn new(ids: &RoaringTreemap, min_doc_id: u64, width: u64) -> Option<Self> {
        let len = usize::try_from(width.div_ceil(64)).ok()?;
        let span = (len as u64).checked_mul(64)?;
        let mut words = vec![0u64; len].into_boxed_slice();
        for offset in ids
            .iter()
            .skip_while(|&doc_id| doc_id < min_doc_id)
            .map(|doc_id| doc_id - min_doc_id)
            .take_while(|&offset| offset < span)
        {
            words[(offset / 64) as usize] |= 1 << (offset % 64);
        }
        Some(RangeBits { min_doc_id, words })
    }

    /// Whether `doc_id` is set; `None` when the span does not cover it.
    #[inline]
    fn get(&self, doc_id: u64) -> Option<bool> {
        // An id below `min_doc_id` wraps to an offset past every word.
        let offset = doc_id.wrapping_sub(self.min_doc_id);
        let word = self.words.get(usize::try_from(offset / 64).ok()?)?;
        Some((word >> (offset % 64)) & 1 == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deleted_docs(
        ids: &[u64],
        min_doc_id: u64,
        max_doc_id: u64,
        doc_count: u64,
    ) -> (RoaringTreemap, DeletedDocs) {
        let ids: RoaringTreemap = ids.iter().copied().collect();
        let deleted = DeletedDocs::new(ids.clone(), min_doc_id, max_doc_id, doc_count);
        (ids, deleted)
    }

    /// Inside the range, at the word boundaries, and for stray bits on either
    /// side of it (a v1/v2 `.delmap` can hold ids outside its segment's
    /// range), the bitset answers exactly as the treemap.
    #[test]
    fn contains_matches_the_treemap_around_the_range() {
        // Range [100, 299]: 200 ids, 4 words, so ids 300..=355 are covered
        // past `max_doc_id`.
        let (ids, deleted) = deleted_docs(
            &[3, 99, 100, 163, 164, 227, 228, 299, 300, 355, 356, 400],
            100,
            299,
            200,
        );
        assert!(deleted.has_bits());
        for doc_id in 0..=500 {
            assert_eq!(
                deleted.contains(doc_id),
                ids.contains(doc_id),
                "doc {doc_id}"
            );
        }
        assert!(!deleted.contains(u64::MAX));
    }

    /// A range much wider than the documents it holds gets no bitset, and
    /// every check searches the treemap.
    #[test]
    fn a_range_too_wide_for_its_documents_keeps_only_the_treemap() {
        // 2 documents may take 16 bits; the range is 17 wide.
        let (ids, deleted) = deleted_docs(&[5, 21], 5, 21, 2);
        assert!(!deleted.has_bits());
        for doc_id in 0..=30 {
            assert_eq!(
                deleted.contains(doc_id),
                ids.contains(doc_id),
                "doc {doc_id}"
            );
        }

        // At the limit, the bitset is built.
        let (_, deleted) = deleted_docs(&[5, 20], 5, 20, 2);
        assert!(deleted.has_bits());
        assert!(deleted.contains(20) && !deleted.contains(19));
    }

    /// An empty segment and a range whose width overflows get no bitset.
    #[test]
    fn an_empty_or_overflowing_range_keeps_only_the_treemap() {
        let (_, deleted) = deleted_docs(&[7], 0, 9, 0);
        assert!(!deleted.has_bits());
        assert!(deleted.contains(7));

        let (_, deleted) = deleted_docs(&[0, u64::MAX], 0, u64::MAX, u64::MAX);
        assert!(!deleted.has_bits());
        assert!(deleted.contains(0) && deleted.contains(u64::MAX) && !deleted.contains(1));
    }

    /// A range ending at `u64::MAX` covers its last id without overflowing.
    #[test]
    fn a_range_at_the_top_of_the_id_space_covers_its_last_id() {
        let min_doc_id = u64::MAX - 9;
        let (ids, deleted) = deleted_docs(&[min_doc_id, u64::MAX], min_doc_id, u64::MAX, 10);
        assert!(deleted.has_bits());
        for doc_id in min_doc_id - 5..=u64::MAX {
            assert_eq!(
                deleted.contains(doc_id),
                ids.contains(doc_id),
                "doc {doc_id}"
            );
        }
    }
}
