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
    ///
    /// `positions` lists the phrase's positions in order, each with the
    /// terms any of which may appear there (see [`PhraseQuery::positions`]).
    pub fn new(
        reader: &dyn LexicalIndexReader,
        field: &str,
        positions: &[Vec<String>],
        slop: u32,
    ) -> Result<Self> {
        let matches = Self::find_phrase_matches(reader, field, positions, slop)?;

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
    ///
    /// A term missing from the index only drops that alternative; the
    /// phrase can match nothing only when every alternative at some
    /// position is missing.
    pub fn find_phrase_matches(
        reader: &dyn LexicalIndexReader,
        field: &str,
        positions: &[Vec<String>],
        slop: u32,
    ) -> Result<Vec<PhraseMatch>> {
        match positions {
            [] => return Ok(Vec::new()),
            [alternatives] if alternatives.len() == 1 => {
                return Self::find_single_term_matches(reader, field, &alternatives[0]);
            }
            _ => {}
        }

        // Per candidate document, the positions of each phrase position's
        // alternatives, merged.
        let mut doc_candidates: HashMap<u64, Vec<Vec<u64>>> = HashMap::new();
        for (slot, alternatives) in positions.iter().enumerate() {
            let mut any_indexed = false;
            for term in alternatives {
                let Some(mut iter) = reader.postings(field, term)? else {
                    continue;
                };
                any_indexed = true;
                while iter.next()? {
                    let doc_id = iter.doc_id();
                    if doc_id == u64::MAX {
                        break;
                    }
                    doc_candidates
                        .entry(doc_id)
                        .or_insert_with(|| vec![Vec::new(); positions.len()])[slot]
                        .extend(iter.positions()?);
                }
            }
            if !any_indexed {
                return Ok(Vec::new());
            }
        }

        let mut phrase_matches = Vec::new();
        for (doc_id, mut slots) in doc_candidates {
            // Also skips a document indexed without positions, whose
            // postings carry none.
            if slots.iter().any(Vec::is_empty) {
                continue;
            }
            // Stacked alternatives (synonyms) share positions.
            for slot in &mut slots {
                slot.sort_unstable();
                slot.dedup();
            }

            // Find valid phrase occurrences in this document
            let phrase_positions = Self::find_phrase_positions(&slots, slop);

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

    /// Match a one-term phrase straight from the term's posting list.
    ///
    /// A single term has no adjacency to check, so every document holding
    /// it matches, `term_freq` times. Requiring positions instead would
    /// find nothing in a field indexed with `term_vectors: false` (#1247),
    /// whose `positions` stay empty. Where positions are stored, their
    /// count is the term frequency, so the phrase frequency, and therefore
    /// the score, is the same either way.
    fn find_single_term_matches(
        reader: &dyn LexicalIndexReader,
        field: &str,
        term: &str,
    ) -> Result<Vec<PhraseMatch>> {
        let Some(mut iter) = reader.postings(field, term)? else {
            return Ok(Vec::new());
        };

        let mut matches = Vec::new();
        while iter.next()? {
            let doc_id = iter.doc_id();
            if doc_id == u64::MAX {
                break;
            }
            matches.push(PhraseMatch {
                doc_id,
                phrase_freq: u32::try_from(iter.term_freq()).unwrap_or(u32::MAX),
                positions: iter.positions()?,
            });
        }
        Ok(matches)
    }

    /// Find valid phrase positions within a document.
    ///
    /// `slots` holds, per phrase position, the sorted positions of its
    /// alternatives in this document. Returns the starting positions of
    /// valid phrases.
    fn find_phrase_positions(slots: &[Vec<u64>], slop: u32) -> Vec<u64> {
        let Some((first, rest)) = slots.split_first() else {
            return Vec::new();
        };
        first
            .iter()
            .copied()
            .filter(|&start_pos| Self::is_valid_phrase_at_position(rest, start_pos, slop))
            .collect()
    }

    /// Check if the phrase positions after the first (`rest`) follow a
    /// phrase starting at `start_pos`: each must occur at the first
    /// position within `slop` after the previous one's.
    fn is_valid_phrase_at_position(rest: &[Vec<u64>], start_pos: u64, slop: u32) -> bool {
        let mut expected_pos = start_pos;

        for positions in rest {
            expected_pos += 1;

            // Use binary search since positions are sorted.
            let idx = positions.partition_point(|&pos| pos < expected_pos);
            let found_pos = positions
                .get(idx)
                .copied()
                .filter(|&pos| pos <= expected_pos + slop as u64);

            match found_pos {
                Some(actual_pos) => expected_pos = actual_pos,
                None => return false,
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
///
/// Each phrase position may also hold several alternative terms, any of
/// which matches there (Lucene's `MultiPhraseQuery`). The query parser
/// builds such a phrase when the analyzer stacks synonyms on a word.
#[derive(Debug, Clone)]
pub struct PhraseQuery {
    /// The field to search in.
    field: String,
    /// The phrase's positions, in order, each with its alternative terms.
    positions: Vec<Vec<String>>,
    /// The boost factor for this query.
    boost: f32,
    /// Optional slop - maximum allowed distance between terms (0 = exact phrase).
    slop: u32,
}

impl PhraseQuery {
    /// Create a new phrase query with one term per position.
    pub fn new<S: Into<String>>(field: S, terms: Vec<String>) -> Self {
        Self::from_positions(field, terms.into_iter().map(|term| vec![term]).collect())
    }

    /// Create a phrase query whose positions each hold the given
    /// alternative terms.
    pub(crate) fn from_positions<S: Into<String>>(field: S, positions: Vec<Vec<String>>) -> Self {
        PhraseQuery {
            field: field.into(),
            positions,
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

    /// Get the phrase's positions, in order, each with its alternative
    /// terms. A phrase built by [`Self::new`] has one term per position.
    pub fn positions(&self) -> &[Vec<String>] {
        &self.positions
    }

    /// Get the slop value.
    pub fn slop(&self) -> u32 {
        self.slop
    }
}

impl Query for PhraseQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        if self.positions.is_empty() {
            return Ok(Box::new(EmptyMatcher::new()));
        }

        // Create a proper phrase matcher that checks position adjacency
        let phrase_matcher = PhraseMatcher::new(reader, &self.field, &self.positions, self.slop)?;
        Ok(Box::new(phrase_matcher))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        if self.positions.is_empty() {
            return Ok(Box::new(BM25Scorer::new(0, 0, 0, 1.0, 1, self.boost)));
        }

        let total_docs = reader.doc_count();
        if total_docs == 0 {
            return Ok(Box::new(BM25Scorer::new(0, 0, 0, 1.0, 1, self.boost)));
        }

        // Get actual phrase matches to create accurate scorer
        let phrase_matches =
            PhraseMatcher::find_phrase_matches(reader, &self.field, &self.positions, self.slop)?;

        // Get field statistics
        let avg_field_length = match reader.field_statistics(&self.field) {
            Ok(field_stats) => field_stats.avg_field_length,
            Err(_) => 10.0, // Default fallback
        };

        // Apply boost multiplier for phrase queries (phrases are generally more valuable)
        let phrase_boost = self.boost * (1.0 + 0.2 * (self.positions.len() as f32 - 1.0));

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
            "PhraseQuery(field:{}, positions:{:?}, slop:{})",
            self.field, self.positions, self.slop
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.positions.is_empty())
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        let terms: usize = self.positions.iter().map(Vec::len).sum();
        Ok(terms as u64 * 100) // Rough estimate
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn collect_positional_field_refs(&self, out: &mut std::collections::HashSet<String>) {
        if self.positions.len() >= 2 {
            out.insert(self.field.clone());
        }
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        if !self.positions.is_empty() && field.is_none_or(|f| f == self.field) {
            out.push(HighlightTerm::Phrase {
                positions: self.positions.clone(),
                slop: self.slop,
            });
        }
    }

    fn cache_key(&self) -> Option<String> {
        // Field + ordered positions + slop determine the matched set; boost
        // is score-only and excluded. `{:?}` on the nested vectors is
        // unambiguous.
        Some(format!(
            "phrase|{:?}|{:?}|{}",
            self.field, self.positions, self.slop
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::analysis::analyzer::analyzer::Analyzer;
    use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
    use crate::analysis::synonym::dictionary::SynonymDictionary;
    use crate::analysis::token_filter::synonym_graph::SynonymGraphFilter;
    use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;
    use crate::data::Document;
    use crate::lexical::index::LexicalIndex;
    use crate::lexical::index::config::InvertedIndexConfig;
    use crate::lexical::index::inverted::InvertedIndex;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn whitespace_analyzer() -> PipelineAnalyzer {
        PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new()))
    }

    /// Index each text as one document of field `body`; doc ids follow the
    /// order of `texts`.
    fn index(analyzer: Arc<dyn Analyzer>, texts: &[&str]) -> Arc<dyn LexicalIndexReader> {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = InvertedIndexConfig {
            analyzer,
            ..Default::default()
        };
        let index = InvertedIndex::create(storage, config).unwrap();
        let mut writer = index.writer().unwrap();
        for text in texts {
            writer
                .add_document(Document::builder().add_text("body", *text).build())
                .unwrap();
        }
        writer.commit().unwrap();
        writer.build_reader().unwrap()
    }

    fn slots(positions: &[&[&str]]) -> Vec<Vec<String>> {
        positions
            .iter()
            .map(|alternatives| alternatives.iter().map(|t| t.to_string()).collect())
            .collect()
    }

    fn matches(
        reader: &dyn LexicalIndexReader,
        positions: &[&[&str]],
        slop: u32,
    ) -> Vec<(u64, u32)> {
        PhraseMatcher::find_phrase_matches(reader, "body", &slots(positions), slop)
            .unwrap()
            .into_iter()
            .map(|m| (m.doc_id, m.phrase_freq))
            .collect()
    }

    const DOGS: &[&str] = &[
        "a big dog",
        "a large dog",
        "a huge cat",
        "a big cat",
        "a very large dog",
    ];

    /// #1252: a position may hold alternatives (stacked synonyms), any of
    /// which matches there.
    #[test]
    fn any_alternative_matches_at_its_position() {
        let reader = index(Arc::new(whitespace_analyzer()), DOGS);
        assert_eq!(
            matches(reader.as_ref(), &[&["a"], &["big", "large"], &["dog"]], 0),
            vec![(0, 1), (1, 1)]
        );
        assert_eq!(
            matches(reader.as_ref(), &[&["a"], &["big", "large"], &["dog"]], 1),
            vec![(0, 1), (1, 1), (4, 1)]
        );
    }

    /// An alternative missing from the index does not rule the phrase out;
    /// only a position whose alternatives are all missing does.
    #[test]
    fn a_position_matches_nothing_only_when_every_alternative_is_missing() {
        let reader = index(Arc::new(whitespace_analyzer()), DOGS);
        assert_eq!(
            matches(
                reader.as_ref(),
                &[&["a"], &["big", "enormous"], &["dog"]],
                0
            ),
            vec![(0, 1)]
        );
        assert!(
            matches(
                reader.as_ref(),
                &[&["a"], &["enormous", "tiny"], &["dog"]],
                0
            )
            .is_empty()
        );
    }

    /// With synonyms stacked at index time, "big" and "large" share one
    /// position; the phrase occurs once there, not once per alternative.
    #[test]
    fn stacked_alternatives_count_one_occurrence() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);
        let analyzer =
            whitespace_analyzer().add_filter(Arc::new(SynonymGraphFilter::new(dict, true)));
        let reader = index(Arc::new(analyzer), &["a big dog", "a large dog"]);

        let found = PhraseMatcher::find_phrase_matches(
            reader.as_ref(),
            "body",
            &slots(&[&["a"], &["big", "large"], &["dog"]]),
            0,
        )
        .unwrap();
        let found: Vec<(u64, u32, Vec<u64>)> = found
            .into_iter()
            .map(|m| (m.doc_id, m.phrase_freq, m.positions))
            .collect();
        assert_eq!(found, vec![(0, 1, vec![0]), (1, 1, vec![0])]);
    }

    /// A position count, not a term count, decides whether positions are
    /// needed and how the phrase is described.
    #[test]
    fn alternatives_are_one_position() {
        let query = PhraseQuery::from_positions("body", slots(&[&["big", "large"], &["dog"]]));
        let mut fields = std::collections::HashSet::new();
        query.collect_positional_field_refs(&mut fields);
        assert!(fields.contains("body"));

        let single = PhraseQuery::from_positions("body", slots(&[&["big", "large"]]));
        let mut fields = std::collections::HashSet::new();
        single.collect_positional_field_refs(&mut fields);
        assert!(fields.is_empty(), "one position needs no positions");

        assert_ne!(query.cache_key(), single.cache_key());
        let mut highlight = Vec::new();
        query.collect_highlight_terms(None, &mut highlight);
        match highlight.as_slice() {
            [HighlightTerm::Phrase { positions, slop: 0 }] => {
                assert_eq!(positions, &slots(&[&["big", "large"], &["dog"]]))
            }
            other => panic!("expected one Phrase, got {other:?}"),
        }
    }

    #[test]
    fn test_phrase_query_creation() {
        let query = PhraseQuery::new("content", vec!["hello".to_string(), "world".to_string()]);

        assert_eq!(query.field(), "content");
        assert_eq!(query.positions(), slots(&[&["hello"], &["world"]]));
        assert_eq!(query.slop(), 0);
        assert_eq!(query.boost(), 1.0);
    }

    #[test]
    fn test_phrase_query_from_phrase() {
        let query = PhraseQuery::from_phrase("content", "hello world test");

        assert_eq!(query.field(), "content");
        assert_eq!(
            query.positions(),
            slots(&[&["hello"], &["world"], &["test"]])
        );
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
