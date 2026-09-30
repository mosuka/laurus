//! Synonym query implementation: several alternatives at one position,
//! scored as one blended term (Issue #1257).
//!
//! `SynonymGraphFilter` stacks a synonym on the word it expands, at the
//! same position (`position_increment = 0`). Querying such a position with
//! `alt1 OR alt2 OR ...` (independent `TermQuery` clauses) double-counts a
//! document that holds every alternative, since index-time synonyms mean
//! the document contains all of them. `SynonymQuery` instead matches the
//! union of postings but scores it as one virtual term, like Lucene's
//! `SynonymQuery`: the alternatives' `doc_freq` is blended as the maximum
//! (the group is at least as common as its most common member) and their
//! `total_term_freq` is summed, and one [`BM25Scorer`] evaluates the
//! combined term frequency [`DisjunctionMatcher::term_freq`] already
//! reports for a document matching several clauses.

use crate::error::Result;
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::matcher::{DisjunctionMatcher, EmptyMatcher, Matcher, PostingMatcher};
use crate::lexical::query::scorer::{BM25Scorer, Scorer};
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::LexicalIndexReader;

/// A query that matches any of several alternative terms at one field
/// position, scoring them as one blended term rather than summing each
/// alternative's independent score.
#[derive(Debug, Clone)]
pub struct SynonymQuery {
    /// The field to search in.
    field: String,
    /// The alternative terms, deduplicated. Order is preserved for
    /// `description`/highlighting but does not affect matching or scoring.
    terms: Vec<String>,
    /// The boost factor for this query.
    boost: f32,
    /// Blended `(doc_freq, total_term_freq)`, frozen by [`rewrite`](Query::rewrite)
    /// against the top-level reader so every segment of a multi-segment
    /// fanout scores against the same statistics instead of each
    /// re-blending from only the alternatives it happens to hold.
    ///
    /// Frozen statistics belong to the reader snapshot they were taken
    /// from, and `rewrite` deliberately never refreshes them, so a
    /// rewritten query must not be cached and reused across a `commit`:
    /// it would keep scoring against the pre-commit statistics. Parse the
    /// query again instead — [`LexicalQueryParser`](crate::lexical::query::parser::LexicalQueryParser)
    /// always builds an unfrozen one.
    stats: Option<(u64, u64)>,
}

impl SynonymQuery {
    /// Create a new synonym query over `terms`, the alternatives at one
    /// position. Duplicate terms are removed (first occurrence wins) so a
    /// repeated alternative cannot double its contribution to the combined
    /// term frequency.
    pub fn new<F, T>(field: F, terms: Vec<T>) -> Self
    where
        F: Into<String>,
        T: Into<String>,
    {
        // A linear scan over the accumulator: an alternative list holds a
        // handful of terms, too few to earn a hash set (and its clone per
        // probe).
        let mut deduped: Vec<String> = Vec::with_capacity(terms.len());
        for term in terms {
            let term = term.into();
            if !deduped.contains(&term) {
                deduped.push(term);
            }
        }
        SynonymQuery {
            field: field.into(),
            terms: deduped,
            boost: 1.0,
            stats: None,
        }
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the alternative terms.
    pub fn terms(&self) -> &[String] {
        &self.terms
    }

    /// Set the boost factor.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Blend `doc_freq` (the maximum across alternatives) and
    /// `total_term_freq` (their sum) from `reader`, Lucene's approach:
    /// the group is at least as common as its most common member, and its
    /// total occurrence count is every alternative's occurrences combined.
    fn blend_stats(&self, reader: &dyn LexicalIndexReader) -> Result<(u64, u64)> {
        let mut doc_freq = 0u64;
        let mut total_term_freq = 0u64;
        for term in &self.terms {
            if let Some(info) = reader.term_info(&self.field, term)? {
                doc_freq = doc_freq.max(info.doc_freq);
                total_term_freq = total_term_freq.saturating_add(info.total_freq);
            }
        }
        Ok((doc_freq, total_term_freq))
    }
}

impl Query for SynonymQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        let mut matchers: Vec<Box<dyn Matcher>> = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            if let Some(posting_iter) = reader.postings(&self.field, term)? {
                matchers.push(Box::new(PostingMatcher::new(posting_iter)));
            }
        }
        match matchers.len() {
            0 => Ok(Box::new(EmptyMatcher::new())),
            // A single surviving alternative keeps the specialized
            // `LeafMatcher::Posting` arm in a parent `BooleanScorer`
            // instead of the generic vtable arm `DisjunctionMatcher` would
            // force.
            1 => Ok(matchers.remove(0)),
            _ => Ok(Box::new(DisjunctionMatcher::new(matchers))),
        }
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        let (doc_freq, total_term_freq) = match self.stats {
            Some(stats) => stats,
            None => self.blend_stats(reader)?,
        };
        let field_stats = reader.field_stats(&self.field)?;

        match field_stats {
            Some(field_stats) if doc_freq > 0 => {
                // Deliberately no block-max metadata: the combined term
                // frequency `DisjunctionMatcher::term_freq` reports can
                // exceed any single alternative's per-block maximum, so a
                // per-term `max_score_factor` would be an unsound upper
                // bound. `BM25Scorer::new`'s loose `k1 + 1` ceiling is
                // valid for any term frequency.
                let scorer = BM25Scorer::new(
                    doc_freq,
                    total_term_freq,
                    field_stats.doc_count,
                    field_stats.avg_length,
                    reader.doc_count(),
                    self.boost,
                )
                .with_field_lengths(reader.field_lengths(&self.field));
                Ok(Box::new(scorer))
            }
            _ => {
                // No alternative survives, or the field is absent: a
                // zero-scoring scorer, mirroring `TermQuery`'s fallback.
                let scorer = BM25Scorer::new(0, 0, 0, 0.0, 0, self.boost);
                Ok(Box::new(scorer))
            }
        }
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        let alternatives = self.terms.join("|");
        if self.boost == 1.0 {
            format!("{}:synonym({})", self.field, alternatives)
        } else {
            format!("{}:synonym({})^{}", self.field, alternatives, self.boost)
        }
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, reader: &dyn LexicalIndexReader) -> Result<bool> {
        // A segment without a term dictionary still matches through the
        // stored-document scan (Issue #1196), so the dictionary cannot
        // prove emptiness; let the matcher decide (mirrors `TermQuery`).
        if !reader.term_info_is_authoritative() {
            return Ok(false);
        }
        // The blended `doc_freq` is the maximum across alternatives, so it
        // is 0 exactly when every alternative is absent. Reusing it once
        // frozen saves a second dictionary lookup per alternative.
        if let Some((doc_freq, _)) = self.stats {
            return Ok(doc_freq == 0);
        }
        for term in &self.terms {
            if let Some(info) = reader.term_info(&self.field, term)?
                && info.doc_freq > 0
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        // Frozen, the blended `doc_freq` is the largest alternative's, a
        // lower bound on the union's size and a good enough estimate to
        // order clauses by — worth one lookup less per alternative.
        if let Some((doc_freq, _)) = self.stats {
            return Ok(doc_freq);
        }
        let mut total = 0u64;
        for term in &self.terms {
            if let Some(info) = reader.term_info(&self.field, term)? {
                total = total.saturating_add(info.doc_freq);
            }
        }
        Ok(total)
    }

    fn rewrite(&self, reader: &dyn LexicalIndexReader) -> Result<Option<Box<dyn Query>>> {
        // Already frozen: the per-segment fanout re-enters `rewrite` with
        // the rewritten query, so this keeps the recursion a cheap no-op
        // (Issue #613's documented idempotence requirement).
        if self.stats.is_some() {
            return Ok(None);
        }
        // Only the top-level (aggregate) reader can blend correctly: its
        // `term_info` sums `doc_freq`/`total_freq` across every segment.
        // A per-segment fanout view only knows the alternatives that
        // particular segment holds, which would blend a smaller `doc_freq`
        // for a segment missing some alternatives and so score identical
        // content differently across segments. Keep the original query
        // (which re-blends locally in `scorer`) when the reader cannot
        // provide global stats.
        if reader
            .as_any()
            .downcast_ref::<InvertedIndexReader>()
            .is_none()
        {
            return Ok(None);
        }
        let mut rewritten = self.clone();
        rewritten.stats = Some(self.blend_stats(reader)?);
        Ok(Some(Box::new(rewritten)))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        if field.is_none_or(|f| f == self.field) {
            for term in &self.terms {
                if !term.is_empty() {
                    out.push(HighlightTerm::Exact(term.clone()));
                }
            }
        }
    }

    fn cache_key(&self) -> Option<String> {
        // Field + the alternative set fully determine the matched document
        // set; boost only scales scores and is excluded, as `TermQuery`
        // does. Sorting a copy means two equal alternative sets in
        // different orders share a cache entry.
        let mut sorted = self.terms.clone();
        sorted.sort();
        Some(format!("synonym|{:?}|{:?}", self.field, sorted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::inverted::reader::{InvertedIndexReader, InvertedIndexReaderConfig};
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use std::sync::Arc;

    fn empty_reader() -> InvertedIndexReader {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        InvertedIndexReader::new(vec![], storage, InvertedIndexReaderConfig::default()).unwrap()
    }

    #[test]
    fn dedups_terms_preserving_first_occurrence() {
        let query = SynonymQuery::new("title", vec!["big", "large", "big"]);
        assert_eq!(query.terms(), &["big".to_string(), "large".to_string()]);
    }

    #[test]
    fn reports_its_field_and_boost() {
        let query = SynonymQuery::new("title", vec!["big", "large"]).with_boost(2.0);
        assert_eq!(query.field(), "title");
        assert_eq!(Query::field(&query), Some("title"));
        assert_eq!(query.boost(), 2.0);
        assert!(query.description().contains("title:synonym("));
        assert!(query.description().ends_with("^2"));
    }

    #[test]
    fn empty_reader_matches_nothing() {
        let reader = empty_reader();
        let query = SynonymQuery::new("title", vec!["big", "large"]);

        assert!(query.is_empty(&reader).unwrap());
        assert_eq!(query.cost(&reader).unwrap(), 0);

        let matcher = query.matcher(&reader).unwrap();
        assert!(matcher.is_exhausted() || matcher.doc_id() == u64::MAX);

        let scorer = query.scorer(&reader).unwrap();
        assert!(scorer.score(0, 1.0, None) >= 0.0);
    }

    #[test]
    fn highlight_terms_cover_every_alternative() {
        let query = SynonymQuery::new("title", vec!["big", "large"]);
        let mut out = Vec::new();
        query.collect_highlight_terms(None, &mut out);
        let exact: Vec<&str> = out
            .iter()
            .map(|t| match t {
                HighlightTerm::Exact(text) => text.as_str(),
                other => panic!("expected Exact, got {other:?}"),
            })
            .collect();
        assert_eq!(exact, ["big", "large"]);

        let mut other_field = Vec::new();
        query.collect_highlight_terms(Some("body"), &mut other_field);
        assert!(other_field.is_empty());
    }

    #[test]
    fn cache_key_is_order_independent() {
        let a = SynonymQuery::new("title", vec!["big", "large"]);
        let b = SynonymQuery::new("title", vec!["large", "big"]);
        assert_eq!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn rewrite_against_a_non_inverted_reader_keeps_the_original() {
        let reader = empty_reader();
        let query = SynonymQuery::new("title", vec!["big", "large"]);
        // `empty_reader` IS an `InvertedIndexReader`, so this exercises the
        // successful-blend path (all-zero stats, since nothing is
        // indexed) and confirms `rewrite` is idempotent afterwards.
        let rewritten = query.rewrite(&reader).unwrap().expect("should rewrite");
        assert!(rewritten.rewrite(&reader).unwrap().is_none());
    }
}
