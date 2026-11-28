//! Matcher implementations for query execution.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::Debug;

use crate::error::Result;
use crate::lexical::reader::PostingIterator;

/// Trait for document matchers.
pub trait Matcher: Send + Debug {
    /// Get the current document ID.
    fn doc_id(&self) -> u64;

    /// Move to the next matching document.
    fn next(&mut self) -> Result<bool>;

    /// Skip to the first document >= target.
    fn skip_to(&mut self, target: u64) -> Result<bool>;

    /// Get the cost of iterating through this matcher.
    fn cost(&self) -> u64;

    /// Check if this matcher is exhausted.
    fn is_exhausted(&self) -> bool;

    /// Get the term frequency for the current document.
    fn term_freq(&self) -> u64 {
        1 // Default implementation for matchers that don't track term frequency
    }

    /// Get the frequency to score the current document with: the term
    /// frequency, with each occurrence weighted by its term's weight.
    ///
    /// Every place that passes a matcher's frequency to
    /// [`Scorer::score`](super::scorer::Scorer::score) reads this one. It is
    /// [`term_freq`](Self::term_freq) for every matcher but a weighted
    /// synonym's, whose alternatives weigh their occurrences by their token
    /// boosts (Issue #1704).
    #[inline]
    fn weighted_term_freq(&self) -> f32 {
        self.term_freq() as f32
    }

    /// Erased self-reference for runtime downcasting (#466).
    /// Used by [`super::scorer::LeafScorer::from_box`] / `LeafMatcher`
    /// dispatch helpers to specialise production hot-path matcher
    /// types (`PostingMatcher`, `AllMatcher`, `EmptyMatcher`) so
    /// `BooleanScorer` / `BlockMaxOrExecutor` inner loops can match-
    /// dispatch into the concrete matcher without the per-doc vtable
    /// lookup.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// A matcher that matches no documents.
#[derive(Debug)]
pub struct EmptyMatcher {
    exhausted: bool,
}

impl EmptyMatcher {
    /// Create a new empty matcher.
    pub fn new() -> Self {
        EmptyMatcher { exhausted: true }
    }
}

impl Default for EmptyMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Matcher for EmptyMatcher {
    fn doc_id(&self) -> u64 {
        u64::MAX
    }

    fn next(&mut self) -> Result<bool> {
        Ok(false)
    }

    fn skip_to(&mut self, _target: u64) -> Result<bool> {
        Ok(false)
    }

    fn cost(&self) -> u64 {
        0
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A matcher that matches every doc id in the dense range `[0, max_doc)`.
///
/// Note: this yields *ids*, not documents — the range includes ids that
/// were never assigned, ids owned by other segments (when `max_doc` is a
/// fanout-global value), and soft-deleted docs. #997 therefore replaced
/// its production use (the MustNot-only universe) with a
/// [`PreComputedMatcher`] over the reader's present, live doc ids; this
/// type remains for API compatibility and for callers that genuinely
/// want the dense range.
#[derive(Debug)]
pub struct AllMatcher {
    current_doc: u64,
    max_doc: u64,
}

impl AllMatcher {
    /// Create a new all matcher.
    pub fn new(max_doc: u64) -> Self {
        AllMatcher {
            current_doc: 0,
            max_doc,
        }
    }
}

impl Matcher for AllMatcher {
    fn doc_id(&self) -> u64 {
        if self.current_doc >= self.max_doc {
            u64::MAX
        } else {
            self.current_doc
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.current_doc >= self.max_doc {
            Ok(false)
        } else {
            self.current_doc += 1;
            Ok(self.current_doc < self.max_doc)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if target >= self.max_doc {
            self.current_doc = self.max_doc;
            Ok(false)
        } else {
            self.current_doc = target;
            Ok(true)
        }
    }

    fn cost(&self) -> u64 {
        self.max_doc
    }

    fn is_exhausted(&self) -> bool {
        self.current_doc >= self.max_doc
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A matcher based on a posting iterator.
#[derive(Debug)]
pub struct PostingMatcher {
    posting_iter: Box<dyn PostingIterator>,
    exhausted: bool,
    cost: u64,
}

impl PostingMatcher {
    /// Create a new posting matcher.
    pub fn new(mut posting_iter: Box<dyn PostingIterator>) -> Self {
        let cost = posting_iter.cost();

        // Position the iterator at the first document
        let exhausted = !posting_iter.next().unwrap_or(false);

        PostingMatcher {
            posting_iter,
            exhausted,
            cost,
        }
    }

    /// Create an exhausted posting matcher.
    pub fn exhausted() -> Self {
        // Create a dummy iterator that's already exhausted
        let postings = vec![];
        let posting_iter = Box::new(
            crate::lexical::index::inverted::reader::InvertedIndexPostingIterator::new(postings),
        );
        PostingMatcher {
            posting_iter,
            exhausted: true,
            cost: 0,
        }
    }
}

impl Matcher for PostingMatcher {
    fn doc_id(&self) -> u64 {
        if self.exhausted {
            u64::MAX
        } else {
            self.posting_iter.doc_id()
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.exhausted {
            Ok(false)
        } else {
            let has_next = self.posting_iter.next()?;
            if !has_next {
                self.exhausted = true;
            }
            Ok(has_next)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.exhausted {
            Ok(false)
        } else {
            let result = self.posting_iter.skip_to(target)?;
            if !result {
                self.exhausted = true;
            }
            Ok(result)
        }
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    fn term_freq(&self) -> u64 {
        if self.exhausted {
            0
        } else {
            self.posting_iter.term_freq()
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A helper struct for tracking matchers in the disjunction heap.
#[derive(Debug)]
struct MatcherEntry {
    matcher: Box<dyn Matcher>,
}

impl PartialEq for MatcherEntry {
    fn eq(&self, other: &Self) -> bool {
        self.matcher.doc_id() == other.matcher.doc_id()
    }
}

impl Eq for MatcherEntry {}

impl PartialOrd for MatcherEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MatcherEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower doc IDs come first
        other.matcher.doc_id().cmp(&self.matcher.doc_id())
    }
}

/// A matcher that implements disjunction (OR) of multiple matchers.
#[derive(Debug)]
pub struct DisjunctionMatcher {
    /// Min-heap of active matchers, ordered by current doc_id.
    heap: BinaryHeap<MatcherEntry>,
    /// Current document ID.
    current_doc: u64,
    /// Whether this matcher is exhausted.
    exhausted: bool,
    /// Total cost estimate.
    cost: u64,
    /// Reusable buffer for matchers to reinsert during advance operations.
    reinsert_buffer: Vec<MatcherEntry>,
}

impl DisjunctionMatcher {
    /// Create a new disjunction matcher from multiple matchers.
    pub fn new(mut matchers: Vec<Box<dyn Matcher>>) -> Self {
        let mut heap = BinaryHeap::new();
        let mut cost = 0;

        // Initialize heap with all non-empty matchers
        for matcher in matchers.drain(..) {
            if !matcher.is_exhausted() {
                cost += matcher.cost();
                heap.push(MatcherEntry { matcher });
            }
        }

        let current_doc = heap
            .peek()
            .map(|entry| entry.matcher.doc_id())
            .unwrap_or(u64::MAX);
        let exhausted = heap.is_empty();

        DisjunctionMatcher {
            heap,
            current_doc,
            exhausted,
            cost,
            reinsert_buffer: Vec::new(),
        }
    }

    /// Create an empty disjunction matcher.
    pub fn empty() -> Self {
        DisjunctionMatcher {
            heap: BinaryHeap::new(),
            current_doc: u64::MAX,
            exhausted: true,
            cost: 0,
            reinsert_buffer: Vec::new(),
        }
    }

    /// Advance to the next document, skipping duplicates.
    fn advance_to_next_doc(&mut self) -> Result<()> {
        if self.exhausted {
            return Ok(());
        }

        let current_doc = self.current_doc;

        // Reuse buffer to avoid allocation on every call.
        self.reinsert_buffer.clear();

        // Advance all matchers that are at the current document
        while let Some(entry) = self.heap.peek() {
            if entry.matcher.doc_id() == current_doc {
                let mut entry = self.heap.pop().unwrap();
                if entry.matcher.next()? && !entry.matcher.is_exhausted() {
                    self.reinsert_buffer.push(entry);
                }
            } else {
                break;
            }
        }

        // Reinsert advanced matchers
        while let Some(entry) = self.reinsert_buffer.pop() {
            self.heap.push(entry);
        }

        // Update current document
        if let Some(entry) = self.heap.peek() {
            self.current_doc = entry.matcher.doc_id();
        } else {
            self.current_doc = u64::MAX;
            self.exhausted = true;
        }

        Ok(())
    }
}

impl Matcher for DisjunctionMatcher {
    fn doc_id(&self) -> u64 {
        self.current_doc
    }

    fn next(&mut self) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }

        self.advance_to_next_doc()?;
        Ok(!self.exhausted)
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.exhausted || target <= self.current_doc {
            return Ok(!self.exhausted);
        }

        // Reuse buffer to avoid allocation on every call.
        self.reinsert_buffer.clear();

        // Skip all matchers to target or beyond
        while let Some(mut entry) = self.heap.pop() {
            if entry.matcher.skip_to(target)? && !entry.matcher.is_exhausted() {
                self.reinsert_buffer.push(entry);
            }
        }

        // Reinsert matchers that successfully skipped
        while let Some(entry) = self.reinsert_buffer.pop() {
            self.heap.push(entry);
        }

        // Update current document
        if let Some(entry) = self.heap.peek() {
            self.current_doc = entry.matcher.doc_id();
            self.exhausted = false;
        } else {
            self.current_doc = u64::MAX;
            self.exhausted = true;
        }

        Ok(!self.exhausted)
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn term_freq(&self) -> u64 {
        let mut total_freq = 0;
        for entry in self.heap.iter() {
            if entry.matcher.doc_id() == self.current_doc {
                total_freq += entry.matcher.term_freq();
            }
        }
        total_freq
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A matcher that implements conjunction (AND) of multiple matchers.
#[derive(Debug)]
pub struct ConjunctionMatcher {
    /// The matchers that must all match.
    matchers: Vec<Box<dyn Matcher>>,
    /// Current document ID.
    current_doc: u64,
    /// Whether this matcher is exhausted.
    exhausted: bool,
    /// Total cost estimate.
    cost: u64,
}

impl ConjunctionMatcher {
    /// Create a new conjunction matcher from multiple matchers.
    pub fn new(matchers: Vec<Box<dyn Matcher>>) -> Self {
        if matchers.is_empty() {
            return ConjunctionMatcher {
                matchers: vec![],
                current_doc: u64::MAX,
                exhausted: true,
                cost: 0,
            };
        }

        let cost = matchers.iter().map(|m| m.cost()).sum();
        let mut matcher = ConjunctionMatcher {
            matchers,
            current_doc: 0,
            exhausted: false,
            cost,
        };

        // Advance to the first matching document
        if let Ok(found) = matcher.advance_to_alignment() {
            if !found {
                matcher.exhausted = true;
                matcher.current_doc = u64::MAX;
            }
        } else {
            matcher.exhausted = true;
            matcher.current_doc = u64::MAX;
        }

        matcher
    }

    /// Advance all matchers to be aligned on the same document.
    fn advance_to_alignment(&mut self) -> Result<bool> {
        if self.matchers.is_empty() || self.exhausted {
            return Ok(false);
        }

        loop {
            // Find the maximum doc ID among all matchers
            let mut max_doc = 0;
            for matcher in &self.matchers {
                let doc_id = matcher.doc_id();
                if doc_id == u64::MAX {
                    self.exhausted = true;
                    return Ok(false);
                }
                if doc_id > max_doc {
                    max_doc = doc_id;
                }
            }

            // Try to advance all matchers to max_doc
            let mut all_aligned = true;
            for matcher in &mut self.matchers {
                let doc_id = matcher.doc_id();
                if doc_id < max_doc {
                    if !matcher.skip_to(max_doc)? {
                        self.exhausted = true;
                        return Ok(false);
                    }
                    if matcher.doc_id() != max_doc {
                        all_aligned = false;
                    }
                }
            }

            if all_aligned {
                self.current_doc = max_doc;
                return Ok(true);
            }
        }
    }
}

impl Matcher for ConjunctionMatcher {
    fn doc_id(&self) -> u64 {
        if self.exhausted {
            u64::MAX
        } else {
            self.current_doc
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.exhausted || self.matchers.is_empty() {
            return Ok(false);
        }

        // Advance the first matcher
        if !self.matchers[0].next()? {
            self.exhausted = true;
            self.current_doc = u64::MAX;
            return Ok(false);
        }

        // Realign all matchers
        self.advance_to_alignment()
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.exhausted || target <= self.current_doc {
            return Ok(!self.exhausted && self.current_doc >= target);
        }

        // Skip the first matcher to target
        if !self.matchers[0].skip_to(target)? {
            self.exhausted = true;
            self.current_doc = u64::MAX;
            return Ok(false);
        }

        // Realign all matchers
        self.advance_to_alignment()
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A matcher that excludes documents matched by negative matchers.
#[derive(Debug)]
pub struct ConjunctionNotMatcher {
    /// The positive matcher (documents must match this).
    positive: Box<dyn Matcher>,
    /// The negative matchers (documents must NOT match any of these).
    negatives: Vec<Box<dyn Matcher>>,
    /// Current document ID.
    current_doc: u64,
    /// Whether this matcher is exhausted.
    exhausted: bool,
    /// Total cost estimate.
    cost: u64,
}

impl ConjunctionNotMatcher {
    /// Create a new conjunction-not matcher.
    pub fn new(positive: Box<dyn Matcher>, negatives: Vec<Box<dyn Matcher>>) -> Self {
        let cost = positive.cost();
        let mut matcher = ConjunctionNotMatcher {
            positive,
            negatives,
            current_doc: 0,
            exhausted: false,
            cost,
        };

        // Advance to the first valid document
        if let Ok(found) = matcher.advance_to_next_valid() {
            if !found {
                matcher.exhausted = true;
                matcher.current_doc = u64::MAX;
            }
        } else {
            matcher.exhausted = true;
            matcher.current_doc = u64::MAX;
        }

        matcher
    }

    /// Check if the current document in positive matcher is excluded by any negative matcher.
    fn is_excluded(&mut self, doc_id: u64) -> Result<bool> {
        for negative in &mut self.negatives {
            // Skip negative matcher to the document
            if negative.doc_id() < doc_id {
                negative.skip_to(doc_id)?;
            }
            // If negative matcher is at the same document, it's excluded
            if negative.doc_id() == doc_id {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Advance to the next valid document (not excluded by negatives).
    fn advance_to_next_valid(&mut self) -> Result<bool> {
        loop {
            if self.positive.is_exhausted() {
                self.exhausted = true;
                self.current_doc = u64::MAX;
                return Ok(false);
            }

            let doc_id = self.positive.doc_id();
            if doc_id == u64::MAX {
                self.exhausted = true;
                self.current_doc = u64::MAX;
                return Ok(false);
            }

            if !self.is_excluded(doc_id)? {
                self.current_doc = doc_id;
                return Ok(true);
            }

            // Current document is excluded, move to next
            if !self.positive.next()? {
                self.exhausted = true;
                self.current_doc = u64::MAX;
                return Ok(false);
            }
        }
    }
}

impl Matcher for ConjunctionNotMatcher {
    fn doc_id(&self) -> u64 {
        if self.exhausted {
            u64::MAX
        } else {
            self.current_doc
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }

        // Advance positive matcher
        if !self.positive.next()? {
            self.exhausted = true;
            self.current_doc = u64::MAX;
            return Ok(false);
        }

        // Find next valid document
        self.advance_to_next_valid()
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.exhausted || target <= self.current_doc {
            return Ok(!self.exhausted && self.current_doc >= target);
        }

        // Skip positive matcher to target
        if !self.positive.skip_to(target)? {
            self.exhausted = true;
            self.current_doc = u64::MAX;
            return Ok(false);
        }

        // Find next valid document
        self.advance_to_next_valid()
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn term_freq(&self) -> u64 {
        self.positive.term_freq()
    }

    fn weighted_term_freq(&self) -> f32 {
        self.positive.weighted_term_freq()
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A matcher that matches every doc id in the dense range `[0, max_doc)`
/// except those matched by the negative matcher.
///
/// Note: like [`AllMatcher`], the dense universe includes never-assigned,
/// cross-segment, and soft-deleted ids. #997 replaced its production use
/// (MustNot-only boolean queries) with a [`ConjunctionNotMatcher`] over a
/// present-live-docs [`PreComputedMatcher`] universe; this type remains
/// for API compatibility.
#[derive(Debug)]
pub struct NotMatcher {
    /// The matcher for documents to exclude.
    negative: Box<dyn Matcher>,
    /// Maximum document ID.
    max_doc: u64,
    /// Current document ID.
    current_doc: u64,
    /// Whether this matcher is exhausted.
    exhausted: bool,
}

impl NotMatcher {
    /// Create a new NOT matcher.
    pub fn new(negative: Box<dyn Matcher>, max_doc: u64) -> Self {
        let mut matcher = NotMatcher {
            negative,
            max_doc,
            current_doc: 0,
            exhausted: false,
        };

        // Advance to the first valid document
        if let Ok(found) = matcher.advance_to_next_valid() {
            if !found {
                matcher.exhausted = true;
            }
        } else {
            matcher.exhausted = true;
        }

        matcher
    }

    /// Advance to the next document not matched by the negative matcher.
    fn advance_to_next_valid(&mut self) -> Result<bool> {
        while self.current_doc < self.max_doc {
            let neg_doc = self.negative.doc_id();

            if neg_doc > self.current_doc || neg_doc == u64::MAX {
                // Negative matcher is ahead or exhausted, current doc is valid
                return Ok(true);
            } else if neg_doc == self.current_doc {
                // Current doc is excluded, move to next
                self.current_doc += 1;
                // Also advance negative matcher if needed
                if self.negative.doc_id() < self.current_doc && !self.negative.is_exhausted() {
                    self.negative.skip_to(self.current_doc)?;
                }
            } else {
                // Negative matcher is behind, skip it forward
                self.negative.skip_to(self.current_doc)?;
            }
        }

        self.current_doc = u64::MAX;
        self.exhausted = true;
        Ok(false)
    }
}

impl Matcher for NotMatcher {
    fn doc_id(&self) -> u64 {
        if self.exhausted || self.current_doc >= self.max_doc {
            u64::MAX
        } else {
            self.current_doc
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.exhausted || self.current_doc >= self.max_doc {
            return Ok(false);
        }

        self.current_doc += 1;
        self.advance_to_next_valid()
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.exhausted || target <= self.current_doc {
            return Ok(!self.exhausted && self.current_doc < self.max_doc);
        }

        if target >= self.max_doc {
            self.current_doc = self.max_doc;
            self.exhausted = true;
            return Ok(false);
        }

        self.current_doc = target;
        self.advance_to_next_valid()
    }

    fn cost(&self) -> u64 {
        self.max_doc
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted || self.current_doc >= self.max_doc
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_matcher() {
        let mut matcher = EmptyMatcher::new();

        assert_eq!(matcher.doc_id(), u64::MAX);
        assert!(matcher.is_exhausted());
        assert_eq!(matcher.cost(), 0);
        assert!(!matcher.next().unwrap());
        assert!(!matcher.skip_to(5).unwrap());
    }

    #[test]
    fn test_all_matcher() {
        let mut matcher = AllMatcher::new(5);

        assert_eq!(matcher.doc_id(), 0);
        assert!(!matcher.is_exhausted());
        assert_eq!(matcher.cost(), 5);

        // Test iteration
        assert!(matcher.next().unwrap());
        assert_eq!(matcher.doc_id(), 1);

        assert!(matcher.next().unwrap());
        assert_eq!(matcher.doc_id(), 2);

        // Test skip_to
        assert!(matcher.skip_to(4).unwrap());
        assert_eq!(matcher.doc_id(), 4);

        assert!(!matcher.skip_to(10).unwrap());
        assert_eq!(matcher.doc_id(), u64::MAX);
        assert!(matcher.is_exhausted());
    }

    #[test]
    fn test_posting_matcher() {
        let postings = vec![
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 0,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 1,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 2,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 3,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 4,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
        ];
        let posting_iter = Box::new(
            crate::lexical::index::inverted::reader::InvertedIndexPostingIterator::new(postings),
        );
        let mut matcher = PostingMatcher::new(posting_iter);

        assert_eq!(matcher.doc_id(), 0);
        assert!(!matcher.is_exhausted());
        assert_eq!(matcher.cost(), 5);

        // Test iteration
        for i in 1..5 {
            assert!(matcher.next().unwrap());
            assert_eq!(matcher.doc_id(), i);
        }

        assert!(!matcher.next().unwrap());
        assert!(matcher.is_exhausted());
    }

    #[test]
    fn test_exhausted_posting_matcher() {
        let matcher = PostingMatcher::exhausted();

        assert_eq!(matcher.doc_id(), u64::MAX);
        assert!(matcher.is_exhausted());
        assert_eq!(matcher.cost(), 0);
    }

    #[test]
    fn test_conjunction_matcher() {
        // Create two posting matchers with overlapping documents
        let postings1 = vec![
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 0,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 2,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 4,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 6,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 8,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
        ];
        let matcher1 = Box::new(PostingMatcher::new(Box::new(
            crate::lexical::index::inverted::reader::InvertedIndexPostingIterator::new(postings1),
        )));

        let postings2 = vec![
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 1,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 2,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 3,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 4,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 5,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 6,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
        ];
        let matcher2 = Box::new(PostingMatcher::new(Box::new(
            crate::lexical::index::inverted::reader::InvertedIndexPostingIterator::new(postings2),
        )));

        let mut conjunction = ConjunctionMatcher::new(vec![matcher1, matcher2]);

        // Should match documents 2, 4, 6 (intersection)
        assert_eq!(conjunction.doc_id(), 2);
        assert!(!conjunction.is_exhausted());

        assert!(conjunction.next().unwrap());
        assert_eq!(conjunction.doc_id(), 4);

        assert!(conjunction.next().unwrap());
        assert_eq!(conjunction.doc_id(), 6);

        assert!(!conjunction.next().unwrap());
        assert!(conjunction.is_exhausted());
    }

    #[test]
    fn test_conjunction_not_matcher() {
        // Positive matcher: documents 0, 1, 2, 3, 4, 5
        let postings_pos = vec![
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 0,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 1,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 2,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 3,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 4,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 5,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
        ];
        let positive = Box::new(PostingMatcher::new(Box::new(
            crate::lexical::index::inverted::reader::InvertedIndexPostingIterator::new(
                postings_pos,
            ),
        )));

        // Negative matcher: documents 1, 3, 5
        let postings_neg = vec![
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 1,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 3,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 5,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
        ];
        let negative = Box::new(PostingMatcher::new(Box::new(
            crate::lexical::index::inverted::reader::InvertedIndexPostingIterator::new(
                postings_neg,
            ),
        )));

        let mut conj_not = ConjunctionNotMatcher::new(positive, vec![negative]);

        // Should match documents 0, 2, 4 (positive minus negative)
        assert_eq!(conj_not.doc_id(), 0);
        assert!(!conj_not.is_exhausted());

        assert!(conj_not.next().unwrap());
        assert_eq!(conj_not.doc_id(), 2);

        assert!(conj_not.next().unwrap());
        assert_eq!(conj_not.doc_id(), 4);

        assert!(!conj_not.next().unwrap());
        assert!(conj_not.is_exhausted());
    }

    #[test]
    fn test_not_matcher() {
        // Negative matcher: documents 1, 3, 5
        let postings_neg = vec![
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 1,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 3,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 5,
                frequency: 1,
                positions: Some(vec![]),
                weight: 1.0,
            },
        ];
        let negative = Box::new(PostingMatcher::new(Box::new(
            crate::lexical::index::inverted::reader::InvertedIndexPostingIterator::new(
                postings_neg,
            ),
        )));

        let mut not_matcher = NotMatcher::new(negative, 7);

        // Should match documents 0, 2, 4, 6 (all except 1, 3, 5)
        assert_eq!(not_matcher.doc_id(), 0);
        assert!(!not_matcher.is_exhausted());

        assert!(not_matcher.next().unwrap());
        assert_eq!(not_matcher.doc_id(), 2);

        assert!(not_matcher.next().unwrap());
        assert_eq!(not_matcher.doc_id(), 4);

        assert!(not_matcher.next().unwrap());
        assert_eq!(not_matcher.doc_id(), 6);

        assert!(!not_matcher.next().unwrap());
        assert!(not_matcher.is_exhausted());
    }

    #[test]
    fn test_conjunction_matcher_empty() {
        let conjunction = ConjunctionMatcher::new(vec![]);
        assert_eq!(conjunction.doc_id(), u64::MAX);
        assert!(conjunction.is_exhausted());
    }

    #[test]
    fn test_conjunction_matcher_no_overlap() {
        // Create two posting matchers with no overlapping documents
        let postings1 = vec![
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 0,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 2,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 4,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
        ];
        let matcher1 = Box::new(PostingMatcher::new(Box::new(
            crate::lexical::index::inverted::reader::InvertedIndexPostingIterator::new(postings1),
        )));

        let postings2 = vec![
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 1,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 3,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
            crate::lexical::index::inverted::core::posting::Posting {
                doc_id: 5,
                frequency: 1_u32,
                positions: Some(vec![]),
                weight: 1.0,
            },
        ];
        let matcher2 = Box::new(PostingMatcher::new(Box::new(
            crate::lexical::index::inverted::reader::InvertedIndexPostingIterator::new(postings2),
        )));

        let conjunction = ConjunctionMatcher::new(vec![matcher1, matcher2]);

        // Should have no matches
        assert_eq!(conjunction.doc_id(), u64::MAX);
        assert!(conjunction.is_exhausted());
    }
}

/// A matcher that works with pre-computed document IDs.
/// This is used for BKD Tree results where document IDs are already filtered.
#[derive(Debug)]
pub struct PreComputedMatcher {
    /// Pre-sorted document IDs that match the query.
    doc_ids: Vec<u64>,
    /// Current position in the doc_ids vector.
    current_index: usize,
    /// Whether this matcher is exhausted.
    exhausted: bool,
}

impl PreComputedMatcher {
    /// Create a new pre-computed matcher with sorted document IDs.
    pub fn new(mut doc_ids: Vec<u64>) -> Self {
        doc_ids.sort_unstable();
        doc_ids.dedup();

        let exhausted = doc_ids.is_empty();

        PreComputedMatcher {
            doc_ids,
            current_index: 0,
            exhausted,
        }
    }

    /// Create an empty pre-computed matcher.
    pub fn empty() -> Self {
        PreComputedMatcher {
            doc_ids: Vec::new(),
            current_index: 0,
            exhausted: true,
        }
    }
}

impl Matcher for PreComputedMatcher {
    fn doc_id(&self) -> u64 {
        if self.exhausted || self.current_index >= self.doc_ids.len() {
            u64::MAX
        } else {
            self.doc_ids[self.current_index]
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }

        self.current_index += 1;
        if self.current_index >= self.doc_ids.len() {
            self.exhausted = true;
            Ok(false)
        } else {
            Ok(true)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }

        // Binary search for the first doc_id >= target
        let mut left = self.current_index;
        let mut right = self.doc_ids.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if self.doc_ids[mid] >= target {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        self.current_index = left;
        if self.current_index >= self.doc_ids.len() {
            self.exhausted = true;
            Ok(false)
        } else {
            Ok(true)
        }
    }

    fn cost(&self) -> u64 {
        self.doc_ids.len() as u64
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Concrete-type dispatch enum for the production matcher hot path
/// (#466). Used by `BooleanScorer` and `BlockMaxOrExecutor` so that
/// the per-doc `skip_to` / `next` / `term_freq` calls in the inner
/// loop dispatch by `match` (BM25-friendly inlining) instead of
/// through a vtable.
#[derive(Debug)]
pub enum LeafMatcher {
    /// `PostingMatcher` — production hot-path leaf matcher. Boxed
    /// so the enum stays narrow and `Vec<(LeafScorer, LeafMatcher)>`
    /// remains cache-friendly inside the BMW pivot loop.
    Posting(Box<PostingMatcher>),
    /// `EmptyMatcher` — empty-result clause.
    Empty(Box<EmptyMatcher>),
    /// `AllMatcher` — match-all (filter) clause.
    All(Box<AllMatcher>),
    /// Trait-object fallback for compound matchers (`DisjunctionMatcher`,
    /// `ConjunctionMatcher`, etc.) and any custom `Matcher` impl.
    Other(Box<dyn Matcher>),
}

impl LeafMatcher {
    /// Specialise a `Box<dyn Matcher>` into a concrete enum variant
    /// when the matcher is one of the production hot-path types.
    /// Falls back to [`LeafMatcher::Other`] otherwise.
    pub fn from_box(b: Box<dyn Matcher>) -> Self {
        use std::any::TypeId;
        let tid = b.as_any().type_id();
        if tid == TypeId::of::<PostingMatcher>() {
            // SAFETY: TypeId equality guarantees the trait object's
            // concrete type. Convert via raw pointer cast.
            let raw: *mut dyn Matcher = Box::into_raw(b);
            let m: Box<PostingMatcher> = unsafe { Box::from_raw(raw as *mut PostingMatcher) };
            LeafMatcher::Posting(m)
        } else if tid == TypeId::of::<EmptyMatcher>() {
            let raw: *mut dyn Matcher = Box::into_raw(b);
            let m: Box<EmptyMatcher> = unsafe { Box::from_raw(raw as *mut EmptyMatcher) };
            LeafMatcher::Empty(m)
        } else if tid == TypeId::of::<AllMatcher>() {
            let raw: *mut dyn Matcher = Box::into_raw(b);
            let m: Box<AllMatcher> = unsafe { Box::from_raw(raw as *mut AllMatcher) };
            LeafMatcher::All(m)
        } else {
            LeafMatcher::Other(b)
        }
    }

    #[inline(always)]
    pub fn doc_id(&self) -> u64 {
        match self {
            LeafMatcher::Posting(m) => m.doc_id(),
            LeafMatcher::Empty(m) => m.doc_id(),
            LeafMatcher::All(m) => m.doc_id(),
            LeafMatcher::Other(m) => m.doc_id(),
        }
    }

    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<bool> {
        match self {
            LeafMatcher::Posting(m) => m.next(),
            LeafMatcher::Empty(m) => m.next(),
            LeafMatcher::All(m) => m.next(),
            LeafMatcher::Other(m) => m.next(),
        }
    }

    #[inline(always)]
    pub fn skip_to(&mut self, target: u64) -> Result<bool> {
        match self {
            LeafMatcher::Posting(m) => m.skip_to(target),
            LeafMatcher::Empty(m) => m.skip_to(target),
            LeafMatcher::All(m) => m.skip_to(target),
            LeafMatcher::Other(m) => m.skip_to(target),
        }
    }

    #[inline(always)]
    pub fn is_exhausted(&self) -> bool {
        match self {
            LeafMatcher::Posting(m) => m.is_exhausted(),
            LeafMatcher::Empty(m) => m.is_exhausted(),
            LeafMatcher::All(m) => m.is_exhausted(),
            LeafMatcher::Other(m) => m.is_exhausted(),
        }
    }

    #[inline(always)]
    pub fn term_freq(&self) -> u64 {
        match self {
            LeafMatcher::Posting(m) => m.term_freq(),
            LeafMatcher::Empty(m) => m.term_freq(),
            LeafMatcher::All(m) => m.term_freq(),
            LeafMatcher::Other(m) => m.term_freq(),
        }
    }

    /// The frequency to score the current document with
    /// ([`Matcher::weighted_term_freq`]).
    #[inline(always)]
    pub fn weighted_term_freq(&self) -> f32 {
        match self {
            LeafMatcher::Posting(m) => m.weighted_term_freq(),
            LeafMatcher::Empty(m) => m.weighted_term_freq(),
            LeafMatcher::All(m) => m.weighted_term_freq(),
            LeafMatcher::Other(m) => m.weighted_term_freq(),
        }
    }
}
