//! Phrase query implementation for exact phrase matching.

use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::Result;
use crate::lexical::query::matcher::{EmptyMatcher, Matcher};
use crate::lexical::query::scorer::{BM25Scorer, Scorer};
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::LexicalIndexReader;

/// A matcher that finds documents containing phrase matches.
#[derive(Debug)]
pub struct PhraseMatcher {
    /// Matching document IDs with phrase frequencies.
    matches: Vec<PhraseMatch>,
    /// Current position in the matches.
    current_index: usize,
    /// Current document ID.
    current_doc_id: u64,
}

/// A phrase match in a specific document.
#[derive(Debug, Clone)]
pub struct PhraseMatch {
    /// Document ID.
    pub doc_id: u64,
    /// Number of phrase occurrences in this document.
    pub phrase_freq: u32,
    /// Positions where the phrase occurs.
    pub positions: Vec<u64>,
}

impl PhraseMatcher {
    /// Create a new phrase matcher.
    pub fn new(
        reader: &dyn LexicalIndexReader,
        field: &str,
        terms: &[String],
        slop: u32,
    ) -> Result<Self> {
        let matches = Self::find_phrase_matches(reader, field, terms, slop)?;

        let current_doc_id = if matches.is_empty() {
            u64::MAX // Invalid state when no matches
        } else {
            matches[0].doc_id
        };

        Ok(PhraseMatcher {
            matches,
            current_index: 0,
            current_doc_id,
        })
    }

    /// Find all documents containing the phrase.
    pub fn find_phrase_matches(
        reader: &dyn LexicalIndexReader,
        field: &str,
        terms: &[String],
        slop: u32,
    ) -> Result<Vec<PhraseMatch>> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // Get posting iterators for all terms
        let mut iterators = Vec::new();
        for term in terms {
            match reader.postings(field, term)? {
                Some(iter) => iterators.push(iter),
                None => return Ok(Vec::new()), // If any term is missing, no phrase matches
            }
        }

        let mut phrase_matches = Vec::new();
        let mut doc_candidates = std::collections::HashMap::new();

        // Find documents that contain all terms
        for (term_idx, iter) in iterators.iter_mut().enumerate() {
            while iter.next()? {
                let doc_id = iter.doc_id();
                if doc_id == u64::MAX {
                    break;
                }

                let positions = iter.positions()?;
                doc_candidates
                    .entry(doc_id)
                    .or_insert_with(Vec::new)
                    .push((term_idx, positions));
            }
        }

        // Check each candidate document for valid phrase matches
        for (doc_id, term_positions) in doc_candidates {
            // Sort by term index to ensure correct order
            let mut term_positions = term_positions;
            term_positions.sort_by_key(|(term_idx, _)| *term_idx);

            // Skip if we don't have all terms
            if term_positions.len() != terms.len() {
                continue;
            }

            // Find valid phrase occurrences in this document
            let phrase_positions = Self::find_phrase_positions(&term_positions, slop);

            if !phrase_positions.is_empty() {
                phrase_matches.push(PhraseMatch {
                    doc_id,
                    phrase_freq: phrase_positions.len() as u32,
                    positions: phrase_positions,
                });
            }
        }

        // Sort matches by document ID
        phrase_matches.sort_by_key(|m| m.doc_id);
        Ok(phrase_matches)
    }

    /// Find valid phrase positions within a document.
    /// Returns the starting positions of valid phrases.
    fn find_phrase_positions(term_positions: &[(usize, Vec<u64>)], slop: u32) -> Vec<u64> {
        if term_positions.is_empty() {
            return Vec::new();
        }

        let mut valid_positions = Vec::new();

        // Get positions for the first term as starting points
        for &start_pos in &term_positions[0].1 {
            if Self::is_valid_phrase_at_position(term_positions, start_pos, slop) {
                valid_positions.push(start_pos);
            }
        }

        valid_positions
    }

    /// Check if there's a valid phrase starting at the given position.
    fn is_valid_phrase_at_position(
        term_positions: &[(usize, Vec<u64>)],
        start_pos: u64,
        slop: u32,
    ) -> bool {
        let mut expected_pos = start_pos;

        for (term_idx, positions) in term_positions {
            if *term_idx == 0 {
                // First term - must be at start_pos
                if !positions.contains(&start_pos) {
                    return false;
                }
            } else {
                // Subsequent terms - must be within slop distance
                expected_pos += 1; // Next expected position

                // Use binary search since positions are sorted.
                let idx = positions.partition_point(|&pos| pos < expected_pos);
                let found_pos = positions
                    .get(idx)
                    .copied()
                    .filter(|&pos| pos <= expected_pos + slop as u64);

                if let Some(actual_pos) = found_pos {
                    expected_pos = actual_pos;
                } else {
                    return false;
                }
            }
        }

        true
    }
}

impl Matcher for PhraseMatcher {
    fn doc_id(&self) -> u64 {
        if self.current_index >= self.matches.len() {
            u64::MAX
        } else {
            self.current_doc_id
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.current_index >= self.matches.len() {
            return Ok(false);
        }

        self.current_index += 1;

        if self.current_index >= self.matches.len() {
            self.current_doc_id = u64::MAX;
            Ok(false)
        } else {
            self.current_doc_id = self.matches[self.current_index].doc_id;
            Ok(true)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.matches.is_empty() {
            return Ok(false);
        }

        // Find the first match >= target
        while self.current_index < self.matches.len()
            && self.matches[self.current_index].doc_id < target
        {
            self.current_index += 1;
        }

        if self.current_index >= self.matches.len() {
            self.current_doc_id = u64::MAX;
            Ok(false)
        } else {
            self.current_doc_id = self.matches[self.current_index].doc_id;
            Ok(true)
        }
    }

    fn is_exhausted(&self) -> bool {
        self.current_index >= self.matches.len()
    }

    fn cost(&self) -> u64 {
        self.matches.len() as u64
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A scorer specialized for phrase queries.
#[derive(Debug, Clone)]
pub struct PhraseScorer {
    /// Document frequencies for phrase matches.
    phrase_doc_freq: HashMap<u64, u32>,
    /// Total number of documents.
    total_docs: u64,
    /// Average field length.
    avg_field_length: f64,
    /// Boost factor.
    boost: f32,
    /// BM25 parameters.
    k1: f32,
    b: f32,
}

impl PhraseScorer {
    /// Create a new phrase scorer with phrase match information.
    pub fn new(
        phrase_matches: &[PhraseMatch],
        total_docs: u64,
        avg_field_length: f64,
        boost: f32,
    ) -> Self {
        let mut phrase_doc_freq = HashMap::new();

        // Calculate phrase frequency for each document
        for phrase_match in phrase_matches {
            phrase_doc_freq.insert(phrase_match.doc_id, phrase_match.phrase_freq);
        }

        PhraseScorer {
            phrase_doc_freq,
            total_docs,
            avg_field_length,
            boost,
            k1: 1.2,
            b: 0.75,
        }
    }

    /// Calculate IDF for the phrase.
    fn phrase_idf(&self) -> f32 {
        let phrase_doc_count = self.phrase_doc_freq.len() as f32;
        if phrase_doc_count == 0.0 || self.total_docs == 0 {
            return 0.0;
        }

        let n = self.total_docs as f32;
        let df = phrase_doc_count;

        // Modified IDF calculation for phrases (typically more selective)
        // Ensure the calculation never produces NaN by clamping values
        let base_idf = ((n - df + 0.5) / (df + 0.5)).ln();
        let epsilon = 0.1;

        // Check for NaN and clamp to valid range
        if base_idf.is_nan() || base_idf.is_infinite() {
            return epsilon * 1.2;
        }

        (base_idf + epsilon).max(epsilon) * 1.2 // Boost phrase IDF
    }

    /// Calculate TF component for phrase frequency.
    fn phrase_tf(&self, phrase_freq: f32, field_length: f32) -> f32 {
        if phrase_freq == 0.0 {
            return 0.0;
        }

        let avg_len = self.avg_field_length.max(1.0) as f32; // Ensure avg_len is at least 1
        let field_len = field_length.max(1.0); // Ensure field_length is at least 1
        let norm_factor = 1.0 - self.b + self.b * (field_len / avg_len);

        // Ensure norm_factor is never zero or negative
        let norm_factor = norm_factor.max(0.1);

        // Phrase TF calculation - phrases are more valuable than individual terms
        let enhanced_phrase_freq = phrase_freq * 1.5; // Boost phrase frequency
        let tf = (enhanced_phrase_freq * (self.k1 + 1.0))
            / (enhanced_phrase_freq + self.k1 * norm_factor);

        // Check for NaN and return safe value
        if tf.is_nan() || tf.is_infinite() {
            return 1.0;
        }

        tf
    }
}

impl Scorer for PhraseScorer {
    fn score(&self, doc_id: u64, _term_freq: f32, _field_length: Option<f32>) -> f32 {
        // Use phrase frequency instead of term frequency
        let phrase_freq = self
            .phrase_doc_freq
            .get(&doc_id)
            .map(|&f| f as f32)
            .unwrap_or(0.0);

        if phrase_freq == 0.0 {
            return 0.0;
        }

        let idf = self.phrase_idf();
        let field_length = self.avg_field_length as f32; // Simplified - would be per-document in full implementation
        let tf = self.phrase_tf(phrase_freq, field_length);

        let score = self.boost * idf * tf;

        // Final check: ensure score is never NaN
        if score.is_nan() || score.is_infinite() {
            // Return a reasonable default score for phrase matches
            return self.boost * 1.0;
        }

        score
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        if self.phrase_doc_freq.is_empty() {
            return 0.0;
        }

        let idf = self.phrase_idf();
        let max_tf = self.k1 + 1.0;
        self.boost * idf * max_tf
    }

    fn name(&self) -> &'static str {
        "PhraseScorer"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A query that matches documents containing an exact phrase.
///
/// A phrase query finds documents where the specified terms appear
/// in the exact order with no other terms between them.
#[derive(Debug, Clone)]
pub struct PhraseQuery {
    /// The field to search in.
    field: String,
    /// The terms that make up the phrase, in order.
    terms: Vec<String>,
    /// The boost factor for this query.
    boost: f32,
    /// Optional slop - maximum allowed distance between terms (0 = exact phrase).
    slop: u32,
}

impl PhraseQuery {
    /// Create a new phrase query.
    pub fn new<S: Into<String>>(field: S, terms: Vec<String>) -> Self {
        PhraseQuery {
            field: field.into(),
            terms,
            boost: 1.0,
            slop: 0,
        }
    }

    /// Create a phrase query from a phrase string.
    pub fn from_phrase<S: Into<String>>(field: S, phrase: &str) -> Self {
        let terms: Vec<String> = phrase.split_whitespace().map(|s| s.to_string()).collect();
        Self::new(field, terms)
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Set the slop (maximum distance between terms).
    ///
    /// A slop of 0 means exact phrase match.
    /// A slop of 1 allows one word between phrase terms.
    pub fn with_slop(mut self, slop: u32) -> Self {
        self.slop = slop;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the phrase terms.
    pub fn terms(&self) -> &[String] {
        &self.terms
    }

    /// Get the slop value.
    pub fn slop(&self) -> u32 {
        self.slop
    }
}

impl Query for PhraseQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        if self.terms.is_empty() {
            return Ok(Box::new(EmptyMatcher::new()));
        }

        // Create a proper phrase matcher that checks position adjacency
        let phrase_matcher = PhraseMatcher::new(reader, &self.field, &self.terms, self.slop)?;
        Ok(Box::new(phrase_matcher))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        if self.terms.is_empty() {
            return Ok(Box::new(BM25Scorer::new(0, 0, 0, 1.0, 1, self.boost)));
        }

        let total_docs = reader.doc_count();
        if total_docs == 0 {
            return Ok(Box::new(BM25Scorer::new(0, 0, 0, 1.0, 1, self.boost)));
        }

        // Get actual phrase matches to create accurate scorer
        let phrase_matches =
            PhraseMatcher::find_phrase_matches(reader, &self.field, &self.terms, self.slop)?;

        // Get field statistics
        let avg_field_length = match reader.field_statistics(&self.field) {
            Ok(field_stats) => field_stats.avg_field_length,
            Err(_) => 10.0, // Default fallback
        };

        // Apply boost multiplier for phrase queries (phrases are generally more valuable)
        let phrase_boost = self.boost * (1.0 + 0.2 * (self.terms.len() as f32 - 1.0));

        // Create specialized phrase scorer
        Ok(Box::new(PhraseScorer::new(
            &phrase_matches,
            total_docs,
            avg_field_length,
            phrase_boost,
        )))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        format!(
            "PhraseQuery(field:{}, terms:{:?}, slop:{})",
            self.field, self.terms, self.slop
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.terms.is_empty())
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        Ok(self.terms.len() as u64 * 100) // Rough estimate
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        if !self.terms.is_empty() && field.is_none_or(|f| f == self.field) {
            out.push(HighlightTerm::Phrase {
                terms: self.terms.clone(),
                slop: self.slop,
            });
        }
    }

    fn cache_key(&self) -> Option<String> {
        // Field + ordered terms + slop determine the matched set; boost is
        // score-only and excluded. `{:?}` on the term vector is unambiguous.
        Some(format!(
            "phrase|{:?}|{:?}|{}",
            self.field, self.terms, self.slop
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phrase_query_creation() {
        let query = PhraseQuery::new("content", vec!["hello".to_string(), "world".to_string()]);

        assert_eq!(query.field(), "content");
        assert_eq!(query.terms(), &["hello", "world"]);
        assert_eq!(query.slop(), 0);
        assert_eq!(query.boost(), 1.0);
    }

    #[test]
    fn test_phrase_query_from_phrase() {
        let query = PhraseQuery::from_phrase("content", "hello world test");

        assert_eq!(query.field(), "content");
        assert_eq!(query.terms(), &["hello", "world", "test"]);
    }

    #[test]
    fn test_phrase_query_with_boost() {
        let query = PhraseQuery::new("content", vec!["hello".to_string()]).with_boost(2.5);

        assert_eq!(query.boost(), 2.5);
    }

    #[test]
    fn test_phrase_query_with_slop() {
        let query = PhraseQuery::new("content", vec!["hello".to_string(), "world".to_string()])
            .with_slop(2);

        assert_eq!(query.slop(), 2);
    }
}
