//! Term dictionary enumeration API.
//!
//! This module provides traits and types for efficiently enumerating terms in the index,
//! similar to Lucene's Terms and TermsEnum.

use crate::error::Result;

/// Statistics about a term in the index.
#[derive(Debug, Clone)]
pub struct TermStats {
    /// The term text
    pub term: String,
    /// Number of documents containing this term
    pub doc_freq: u64,
    /// Total number of occurrences across all documents
    pub total_term_freq: u64,
}

/// Iterator over terms in a field's term dictionary.
///
/// This is similar to Lucene's TermsEnum, providing sequential access to
/// indexed terms in sorted order.
///
/// # Example (conceptual - not yet implemented)
///
/// ```ignore
/// let terms = reader.terms("content")?;
/// let mut terms_enum = terms.iterator()?;
///
/// while let Some(term) = terms_enum.next()? {
///     println!("Term: {}, DocFreq: {}", term.term, term.doc_freq);
/// }
/// ```ignore
pub trait TermsEnum: Send + Sync {
    /// Advance to the next term in the enumeration.
    ///
    /// Returns `None` when there are no more terms.
    fn next(&mut self) -> Result<Option<TermStats>>;

    /// Seek to the first term greater than or equal to the target.
    ///
    /// Returns `true` if an exact match was found, `false` if positioned
    /// at the next term greater than target, or error if no such term exists.
    ///
    /// This is useful for implementing prefix queries efficiently.
    fn seek(&mut self, target: &str) -> Result<bool>;

    /// Seek to the exact term.
    ///
    /// Returns `true` if the term exists, `false` otherwise.
    fn seek_exact(&mut self, term: &str) -> Result<bool>;

    /// Get the current term without advancing the iterator.
    ///
    /// Returns `None` if the iterator hasn't been advanced or is exhausted.
    fn current(&self) -> Option<&TermStats>;

    /// Get statistics for the current term.
    ///
    /// This is equivalent to `current()` but returns a copy.
    fn term_stats(&self) -> Option<TermStats> {
        self.current().cloned()
    }
}

/// Access to the term dictionary for a specific field.
///
/// This is similar to Lucene's Terms, representing all terms indexed in a field.
///
/// # Example (conceptual - not yet implemented)
///
/// ```ignore
/// let terms = reader.terms("content")?;
/// println!("Total terms: {}", terms.size());
///
/// // Iterate over all terms
/// let mut iter = terms.iterator()?;
/// while let Some(term) = iter.next()? {
///     println!("{}: {} docs", term.term, term.doc_freq);
/// }
///
/// // Or seek to specific position
/// let mut iter = terms.iterator()?;
/// if iter.seek("prefix")? {
///     println!("Found exact match");
/// }
/// ```ignore
pub trait Terms: Send + Sync {
    /// Get an iterator over all terms in this field.
    fn iterator(&self) -> Result<Box<dyn TermsEnum>>;

    /// Get the number of unique terms in this field.
    ///
    /// Returns `None` if the count is not available or too expensive to compute.
    fn size(&self) -> Option<u64>;

    /// Get the sum of document frequencies across all terms.
    ///
    /// This is the total number of term-document pairs.
    fn sum_doc_freq(&self) -> Option<u64>;

    /// Get the sum of total term frequencies across all terms.
    ///
    /// This is the total number of term occurrences in the index.
    fn sum_total_term_freq(&self) -> Option<u64>;

    /// Check if this field has term frequencies stored.
    fn has_freqs(&self) -> bool {
        true
    }

    /// Check if this field has positions stored.
    fn has_positions(&self) -> bool {
        false
    }

    /// Check if this field has offsets stored.
    fn has_offsets(&self) -> bool {
        false
    }

    /// Check if this field has payloads stored.
    fn has_payloads(&self) -> bool {
        false
    }
}

/// Extension trait for LexicalIndexReader to provide term dictionary access.
///
/// This will eventually be added to the LexicalIndexReader trait, but is defined
/// separately here to avoid breaking changes during development.
///
/// # Example (conceptual - not yet implemented)
///
/// ```ignore
/// use laurus::lexical::terms::TermDictionaryAccess;
///
/// let reader = index.reader()?;
/// let terms = reader.terms("content")?;
/// let mut iter = terms.iterator()?;
///
/// while let Some(term_stats) = iter.next()? {
///     println!("{}: {} docs", term_stats.term, term_stats.doc_freq);
/// }
/// ```ignore
pub trait TermDictionaryAccess {
    /// Get access to the term dictionary for the specified field.
    ///
    /// Returns `None` if the field doesn't exist in the index.
    fn terms(&self, field: &str) -> Result<Option<Box<dyn Terms>>>;

    /// Check if a specific term exists in a field.
    ///
    /// This is a convenience method equivalent to:
    /// ```ignore
    /// reader.terms(field)?.and_then(|terms| {
    ///     let mut iter = terms.iterator()?;
    ///     iter.seek_exact(term)
    /// })
    /// ```ignore
    fn term_exists(&self, field: &str, term: &str) -> Result<bool> {
        if let Some(terms) = self.terms(field)? {
            let mut iter = terms.iterator()?;
            iter.seek_exact(term)
        } else {
            Ok(false)
        }
    }
}

// Implement TermsEnum for Box<dyn TermsEnum> to allow composition
impl TermsEnum for Box<dyn TermsEnum> {
    fn next(&mut self) -> Result<Option<TermStats>> {
        (**self).next()
    }

    fn seek(&mut self, target: &str) -> Result<bool> {
        (**self).seek(target)
    }

    fn seek_exact(&mut self, term: &str) -> Result<bool> {
        (**self).seek_exact(term)
    }

    fn current(&self) -> Option<&TermStats> {
        (**self).current()
    }
}

// TODO: Add automaton intersection support: terms.intersect(automaton)
// TODO: Add range query support: terms.range(min, max)

// ============================================================================
// Concrete Implementations for InvertedIndex
// ============================================================================

use std::sync::Arc;

use crate::lexical::index::structures::dictionary::BlockTermDictionary;

/// Lazy iterator over one field's terms in a term dictionary
/// (Issue #845).
///
/// Terms are stored under `"field:term"` keys in the dictionary's
/// ascending `sorted_terms` array, so one field occupies a contiguous
/// range. Construction is a single binary search to the range start —
/// **no dictionary walk and no term cloning** (the previous
/// implementation copied every matching term into a `Vec` per
/// `iterator()` call, after walking the whole cross-field dictionary).
/// `next()` yields entries while keys keep the field prefix, cloning
/// only each **yielded** term's text.
pub struct InvertedIndexTermsEnum {
    /// The shared dictionary backing this cursor.
    dict: Arc<BlockTermDictionary>,
    /// `"{field}:"` — bounds the cursor to the field's contiguous range.
    prefix: String,
    /// Ordinal of the next entry [`TermsEnum::next`] will yield.
    position: usize,
    /// Current term stats (cached)
    current: Option<TermStats>,
}

impl InvertedIndexTermsEnum {
    /// Create a new lazy terms enum for a field.
    ///
    /// # Arguments
    ///
    /// * `field` - The field whose terms to enumerate.
    /// * `dict` - The segment's term dictionary (refcount-shared).
    pub fn new(field: &str, dict: &Arc<BlockTermDictionary>) -> Self {
        let prefix = format!("{}:", field);
        let position = dict.seek_index(&prefix);
        InvertedIndexTermsEnum {
            dict: Arc::clone(dict),
            prefix,
            position,
            current: None,
        }
    }

    /// Stats for the entry at `idx`, or `None` when `idx` is past the
    /// dictionary end or the entry no longer belongs to this field.
    fn stats_at(&self, idx: usize) -> Option<TermStats> {
        let (key, info) = self.dict.entry_at(idx)?;
        let term = key.strip_prefix(&self.prefix)?;
        Some(TermStats {
            term: term.to_string(),
            doc_freq: info.doc_frequency,
            total_term_freq: info.total_frequency,
        })
    }
}

impl TermsEnum for InvertedIndexTermsEnum {
    fn next(&mut self) -> Result<Option<TermStats>> {
        match self.stats_at(self.position) {
            Some(stats) => {
                self.current = Some(stats.clone());
                self.position += 1;
                Ok(Some(stats))
            }
            None => {
                self.current = None;
                Ok(None)
            }
        }
    }

    fn seek(&mut self, target: &str) -> Result<bool> {
        // Binary search within the field's keyspace; positions at the
        // target or the next greater term (the following `next()` call
        // yields the entry `current` now points at, as before).
        let full_key = format!("{}{}", self.prefix, target);
        self.position = self.dict.seek_index(&full_key);
        match self.stats_at(self.position) {
            Some(stats) => {
                let exact = stats.term == target;
                self.current = Some(stats);
                Ok(exact)
            }
            None => {
                self.current = None;
                Ok(false)
            }
        }
    }

    fn seek_exact(&mut self, term: &str) -> Result<bool> {
        // O(1) point probe via the dictionary's hash index; only an
        // exact hit repositions the cursor (matching the previous
        // implementation, which left the position unchanged on a miss).
        let full_key = format!("{}{}", self.prefix, term);
        if self.dict.get(&full_key).is_none() {
            self.current = None;
            return Ok(false);
        }
        self.position = self.dict.seek_index(&full_key);
        self.current = self.stats_at(self.position);
        Ok(true)
    }

    fn current(&self) -> Option<&TermStats> {
        self.current.as_ref()
    }
}

/// Implementation of Terms trait for a specific field.
pub struct InvertedIndexTerms {
    field: String,
    dict: Arc<BlockTermDictionary>,
    // Cached statistics
    size: Option<u64>,
    sum_doc_freq: Option<u64>,
    sum_total_term_freq: Option<u64>,
}

impl InvertedIndexTerms {
    /// Create a new Terms instance for a field.
    pub fn new(field: &str, dict: Arc<BlockTermDictionary>) -> Self {
        let field_prefix = format!("{}:", field);

        // Calculate statistics over the field's contiguous sorted range
        // only (Issue #845): binary-search to the range start and stop
        // at the first key without the prefix, instead of walking the
        // whole cross-field dictionary.
        let mut size = 0u64;
        let mut sum_doc_freq = 0u64;
        let mut sum_total_term_freq = 0u64;

        let mut idx = dict.seek_index(&field_prefix);
        while let Some((key, info)) = dict.entry_at(idx) {
            if !key.starts_with(&field_prefix) {
                break;
            }
            size += 1;
            // Saturating: on-disk statistics a corrupt segment can push
            // past u64 (Issue #1224).
            sum_doc_freq = sum_doc_freq.saturating_add(info.doc_frequency);
            sum_total_term_freq = sum_total_term_freq.saturating_add(info.total_frequency);
            idx += 1;
        }

        InvertedIndexTerms {
            field: field.to_string(),
            dict,
            size: Some(size),
            sum_doc_freq: Some(sum_doc_freq),
            sum_total_term_freq: Some(sum_total_term_freq),
        }
    }
}

impl Terms for InvertedIndexTerms {
    fn iterator(&self) -> Result<Box<dyn TermsEnum>> {
        Ok(Box::new(InvertedIndexTermsEnum::new(
            &self.field,
            &self.dict,
        )))
    }

    fn size(&self) -> Option<u64> {
        self.size
    }

    fn sum_doc_freq(&self) -> Option<u64> {
        self.sum_doc_freq
    }

    fn sum_total_term_freq(&self) -> Option<u64> {
        self.sum_total_term_freq
    }
}

// ============================================================================
// Multi-segment merged Terms implementation
// ============================================================================

/// Terms merged from multiple segment term dictionaries.
///
/// Collects terms from all segments, accumulating `doc_freq` and
/// `total_term_freq` for the same term across segments.
pub struct MergedInvertedIndexTerms {
    /// The field whose terms this view merges.
    field: String,
    /// Per-segment dictionaries (refcount-shared).
    dicts: Vec<Arc<BlockTermDictionary>>,
    // Pre-computed statistics (field-range-bounded merge pass).
    size: u64,
    sum_doc_freq: u64,
    sum_total_term_freq: u64,
}

impl MergedInvertedIndexTerms {
    /// Create merged terms for `field` from multiple dictionaries.
    ///
    /// Statistics come from one field-range-bounded k-way merge pass
    /// (Issue #845) instead of the previous full walk of every
    /// segment's whole cross-field dictionary into a `BTreeMap`;
    /// [`Terms::iterator`] then hands out fresh lazy merge cursors
    /// instead of cloning a pre-merged `Vec` per call.
    pub fn new(field: &str, dicts: &[Arc<BlockTermDictionary>]) -> Self {
        let mut stats_cursor = MergedTermsEnum::new(field, dicts);
        let mut size = 0u64;
        let mut sum_doc_freq = 0u64;
        let mut sum_total_term_freq = 0u64;
        // A merge pass is required for `size`: it counts DISTINCT terms
        // across segments, which per-segment stats cannot provide.
        while let Ok(Some(stats)) = stats_cursor.next() {
            size += 1;
            // Saturating, as in `InvertedIndexTerms::new` (Issue #1224).
            sum_doc_freq = sum_doc_freq.saturating_add(stats.doc_freq);
            sum_total_term_freq = sum_total_term_freq.saturating_add(stats.total_term_freq);
        }

        MergedInvertedIndexTerms {
            field: field.to_string(),
            dicts: dicts.to_vec(),
            size,
            sum_doc_freq,
            sum_total_term_freq,
        }
    }
}

impl Terms for MergedInvertedIndexTerms {
    fn iterator(&self) -> Result<Box<dyn TermsEnum>> {
        Ok(Box::new(MergedTermsEnum::new(&self.field, &self.dicts)))
    }

    fn size(&self) -> Option<u64> {
        Some(self.size)
    }

    fn sum_doc_freq(&self) -> Option<u64> {
        Some(self.sum_doc_freq)
    }

    fn sum_total_term_freq(&self) -> Option<u64> {
        Some(self.sum_total_term_freq)
    }
}

/// Lazy k-way merge over per-segment field cursors (Issue #845).
///
/// Holds one [`InvertedIndexTermsEnum`] per segment plus its pulled
/// head entry. `next()` emits the minimum head term with
/// `doc_freq` / `total_term_freq` summed across every segment holding
/// that term (the previous pre-merged behavior), then refills the
/// consumed heads. Segment counts are small, so heads are compared
/// linearly (mirroring `MergedPostingIterator`'s linear mode).
struct MergedTermsEnum {
    /// Per-segment lazy cursors.
    children: Vec<InvertedIndexTermsEnum>,
    /// Pulled-but-unemitted head entry per child.
    heads: Vec<Option<TermStats>>,
    /// Current merged term stats (cached).
    current: Option<TermStats>,
}

impl MergedTermsEnum {
    /// Build the merge cursor: one field-bounded lazy cursor per
    /// segment, heads primed with each cursor's first entry.
    fn new(field: &str, dicts: &[Arc<BlockTermDictionary>]) -> Self {
        let mut children: Vec<InvertedIndexTermsEnum> = dicts
            .iter()
            .map(|dict| InvertedIndexTermsEnum::new(field, dict))
            .collect();
        let heads = children
            .iter_mut()
            .map(|child| child.next().unwrap_or(None))
            .collect();
        MergedTermsEnum {
            children,
            heads,
            current: None,
        }
    }

    /// Merge the minimum-term heads into one `TermStats` without
    /// consuming them (peek): equal terms sum df / ttf across segments.
    fn peek_merged(&self) -> Option<TermStats> {
        let min_term = self
            .heads
            .iter()
            .flatten()
            .map(|stats| stats.term.as_str())
            .min()?;
        let mut merged = TermStats {
            term: min_term.to_string(),
            doc_freq: 0,
            total_term_freq: 0,
        };
        for stats in self.heads.iter().flatten() {
            if stats.term == merged.term {
                // Saturating, as in `InvertedIndexTerms::new` (Issue #1224).
                merged.doc_freq = merged.doc_freq.saturating_add(stats.doc_freq);
                merged.total_term_freq =
                    merged.total_term_freq.saturating_add(stats.total_term_freq);
            }
        }
        Some(merged)
    }

    /// Refill every head whose term equals `term` from its child.
    fn advance_heads_matching(&mut self, term: &str) -> Result<()> {
        for (child, head) in self.children.iter_mut().zip(self.heads.iter_mut()) {
            if head.as_ref().is_some_and(|stats| stats.term == term) {
                *head = child.next()?;
            }
        }
        Ok(())
    }
}

impl TermsEnum for MergedTermsEnum {
    fn next(&mut self) -> Result<Option<TermStats>> {
        match self.peek_merged() {
            Some(stats) => {
                self.advance_heads_matching(&stats.term)?;
                self.current = Some(stats.clone());
                Ok(Some(stats))
            }
            None => {
                self.current = None;
                Ok(None)
            }
        }
    }

    fn seek(&mut self, target: &str) -> Result<bool> {
        // Seek every child, then re-prime its head with the entry the
        // child now points at (child `seek` peeks without consuming, so
        // the following child `next()` pulls exactly that entry).
        for (child, head) in self.children.iter_mut().zip(self.heads.iter_mut()) {
            child.seek(target)?;
            *head = child.next()?;
        }
        self.current = self.peek_merged();
        Ok(self
            .current
            .as_ref()
            .is_some_and(|stats| stats.term == target))
    }

    fn seek_exact(&mut self, term: &str) -> Result<bool> {
        let found = self.seek(term)?;
        if !found {
            self.current = None;
        }
        Ok(found)
    }

    fn current(&self) -> Option<&TermStats> {
        self.current.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::structures::dictionary::{TermDictionaryBuilder, TermInfo};

    /// Build a dictionary with keys spanning three fields so field
    /// boundaries are exercised on both sides.
    fn mixed_field_dict() -> Arc<BlockTermDictionary> {
        let mut builder = TermDictionaryBuilder::new();
        for (key, df, ttf) in [
            ("alpha:apple", 2, 5),
            ("alpha:apricot", 1, 1),
            ("body:program", 3, 7),
            ("body:programmer", 1, 2),
            ("body:programming", 4, 9),
            ("body:python", 2, 3),
            ("title:zebra", 1, 1),
        ] {
            builder.add_term(key.to_string(), TermInfo::new(0, 0, df, ttf));
        }
        Arc::new(builder.build().unwrap())
    }

    /// Issue #845: the lazy enum must yield exactly the field's terms in
    /// sorted order — no leakage from lexically adjacent fields.
    #[test]
    fn lazy_enum_isolates_field_boundaries() {
        let dict = mixed_field_dict();
        let mut terms_enum = InvertedIndexTermsEnum::new("body", &dict);
        let mut drained = Vec::new();
        while let Some(stats) = terms_enum.next().unwrap() {
            drained.push((stats.term, stats.doc_freq, stats.total_term_freq));
        }
        assert_eq!(
            drained,
            vec![
                ("program".to_string(), 3, 7),
                ("programmer".to_string(), 1, 2),
                ("programming".to_string(), 4, 9),
                ("python".to_string(), 2, 3),
            ]
        );

        // A field with no terms yields nothing.
        let mut empty = InvertedIndexTermsEnum::new("missing", &dict);
        assert!(empty.next().unwrap().is_none());
    }

    /// Equivalence with the dictionary's own prefix range: the lazy
    /// drain must match `find_prefix("{field}:")` exactly.
    #[test]
    fn lazy_enum_matches_find_prefix() {
        let dict = mixed_field_dict();
        let expected: Vec<String> = dict
            .find_prefix("body:")
            .into_iter()
            .map(|(key, _)| key.strip_prefix("body:").unwrap().to_string())
            .collect();
        let mut terms_enum = InvertedIndexTermsEnum::new("body", &dict);
        let mut drained = Vec::new();
        while let Some(stats) = terms_enum.next().unwrap() {
            drained.push(stats.term);
        }
        assert_eq!(drained, expected);
    }

    /// Seek semantics: exact hit, positioned-at-next miss, past-the-end,
    /// and the peek contract (the next `next()` yields the sought entry).
    #[test]
    fn lazy_enum_seek_semantics() {
        let dict = mixed_field_dict();
        let mut terms_enum = InvertedIndexTermsEnum::new("body", &dict);

        assert!(terms_enum.seek("programmer").unwrap(), "exact hit");
        assert_eq!(terms_enum.current().unwrap().term, "programmer");
        assert_eq!(terms_enum.next().unwrap().unwrap().term, "programmer");

        assert!(!terms_enum.seek("prog").unwrap(), "miss -> next greater");
        assert_eq!(terms_enum.current().unwrap().term, "program");

        assert!(!terms_enum.seek("zzz").unwrap(), "past the field range");
        assert!(terms_enum.current().is_none());

        assert!(terms_enum.seek_exact("python").unwrap());
        assert_eq!(terms_enum.current().unwrap().term, "python");
        assert!(!terms_enum.seek_exact("nope").unwrap());
    }

    /// Merged view: distinct terms in sorted order with df/ttf summed
    /// across segments; stats match a full drain.
    #[test]
    fn merged_enum_aggregates_across_segments() {
        let mut b1 = TermDictionaryBuilder::new();
        b1.add_term("body:apple".to_string(), TermInfo::new(0, 0, 2, 4));
        b1.add_term("body:cherry".to_string(), TermInfo::new(0, 0, 1, 1));
        let mut b2 = TermDictionaryBuilder::new();
        b2.add_term("body:apple".to_string(), TermInfo::new(0, 0, 3, 5));
        b2.add_term("body:banana".to_string(), TermInfo::new(0, 0, 1, 2));
        let dicts = vec![Arc::new(b1.build().unwrap()), Arc::new(b2.build().unwrap())];

        let merged = MergedInvertedIndexTerms::new("body", &dicts);
        assert_eq!(merged.size(), Some(3), "3 distinct terms");
        assert_eq!(merged.sum_doc_freq(), Some(2 + 3 + 1 + 1));
        assert_eq!(merged.sum_total_term_freq(), Some(4 + 5 + 2 + 1));

        let mut iter = merged.iterator().unwrap();
        let mut drained = Vec::new();
        while let Some(stats) = iter.next().unwrap() {
            drained.push((stats.term, stats.doc_freq, stats.total_term_freq));
        }
        assert_eq!(
            drained,
            vec![
                ("apple".to_string(), 5, 9), // summed across both segments
                ("banana".to_string(), 1, 2),
                ("cherry".to_string(), 1, 1),
            ]
        );

        // Seek: exact on a term present in only one segment, and the
        // peek contract across the merge.
        let mut iter = merged.iterator().unwrap();
        assert!(iter.seek("banana").unwrap());
        assert_eq!(iter.next().unwrap().unwrap().term, "banana");
        assert_eq!(iter.next().unwrap().unwrap().term, "cherry");
        assert!(!iter.seek("aaa").unwrap(), "miss -> positioned at apple");
        assert_eq!(iter.current().unwrap().term, "apple");
    }

    /// A dictionary whose `keys` each claim 2^63 documents and occurrences,
    /// as a corrupt segment can.
    fn huge_stats_dict(keys: &[&str]) -> Arc<BlockTermDictionary> {
        let mut builder = TermDictionaryBuilder::new();
        for key in keys {
            builder.add_term(key.to_string(), TermInfo::new(0, 0, 1 << 63, 1 << 63));
        }
        Arc::new(builder.build().unwrap())
    }

    /// Statistic sums saturate rather than overflow on huge per-term values
    /// (Issue #1224): over a field's terms, and — for the merged view —
    /// over one term's segments too.
    #[test]
    fn statistic_sums_saturate_at_u64_max() {
        let a = huge_stats_dict(&["body:a", "body:b"]);
        let terms = InvertedIndexTerms::new("body", a.clone());
        assert_eq!(terms.sum_doc_freq(), Some(u64::MAX));
        assert_eq!(terms.sum_total_term_freq(), Some(u64::MAX));

        // "a" is in both dictionaries, so its merged statistics are summed
        // across them before the field's two terms are summed again.
        let b = huge_stats_dict(&["body:a"]);
        let merged = MergedInvertedIndexTerms::new("body", &[a, b]);
        assert_eq!(merged.sum_doc_freq(), Some(u64::MAX));
        assert_eq!(merged.sum_total_term_freq(), Some(u64::MAX));
        let first = merged.iterator().unwrap().next().unwrap().unwrap();
        assert_eq!(
            (first.doc_freq, first.total_term_freq),
            (u64::MAX, u64::MAX)
        );
    }
}
