//! Span queries for positional and proximity-based searching.
//!
//! Span queries provide advanced search capabilities based on term positions
//! within documents, enabling complex proximity and phrase searches.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::scorer::Scorer;
use crate::lexical::query::{HighlightTerm, Query};
use crate::lexical::reader::LexicalIndexReader;

/// A span represents a term occurrence with position information.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Span {
    /// Start position (inclusive)
    pub start: u32,
    /// End position (exclusive)
    pub end: u32,
    /// Term that generated this span
    pub term: String,
}

impl Span {
    /// Create a new span.
    pub fn new(start: u32, end: u32, term: String) -> Self {
        Span { start, end, term }
    }

    /// Get the length of this span.
    pub fn length(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    /// Check if this span overlaps with another span.
    pub fn overlaps(&self, other: &Span) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if this span contains another span.
    pub fn contains(&self, other: &Span) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Get the distance between this span and another span.
    pub fn distance_to(&self, other: &Span) -> u32 {
        if self.overlaps(other) {
            0
        } else if self.end <= other.start {
            other.start - self.end
        } else {
            self.start.saturating_sub(other.end)
        }
    }
}

/// Base trait for span queries.
pub trait SpanQuery: Send + Sync + std::fmt::Debug {
    /// Get spans for a document.
    fn get_spans(&self, doc_id: u64, reader: &dyn LexicalIndexReader) -> Result<Vec<Span>>;

    /// Return the set of candidate doc IDs that *might* match this span query.
    /// The result is an upper bound; actual matches are confirmed by [`get_spans`].
    fn candidate_doc_ids(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<u64>>;

    /// Get the field name this span query operates on.
    fn field_name(&self) -> &str;

    /// Clone this span query.
    fn clone_box(&self) -> Box<dyn SpanQuery>;

    /// Collect the terms this span query matches, for highlighting (#594).
    ///
    /// Composites recurse into every child: a `SpanWithinQuery`'s `exclude`
    /// is a positive proximity requirement, not a negation. Field gating is
    /// the wrapper's job (`SpanQueryWrapper`).
    fn collect_highlight_terms(&self, out: &mut Vec<HighlightTerm>) {
        let _ = out;
    }
}

/// A span query that matches a single term.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanTermQuery {
    /// Field to search in
    field: String,
    /// Term to search for
    term: String,
    /// Boost factor
    boost: f32,
}

impl SpanTermQuery {
    /// Create a new span term query.
    pub fn new<F: Into<String>, T: Into<String>>(field: F, term: T) -> Self {
        SpanTermQuery {
            field: field.into(),
            term: term.into(),
            boost: 1.0,
        }
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the term.
    pub fn term(&self) -> &str {
        &self.term
    }
}

impl SpanQuery for SpanTermQuery {
    fn get_spans(&self, doc_id: u64, reader: &dyn LexicalIndexReader) -> Result<Vec<Span>> {
        let mut spans = Vec::new();

        if let Some(mut iter) = reader.postings(&self.field, &self.term)? {
            // Skip to the target document
            if iter.skip_to(doc_id)? && iter.doc_id() == doc_id {
                // Determine positions if available
                let positions = iter.positions()?;
                for pos in positions {
                    // Span for a single term has length 1
                    spans.push(Span::new(pos as u32, pos as u32 + 1, self.term.clone()));
                }
            }
        }

        Ok(spans)
    }

    fn candidate_doc_ids(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<u64>> {
        let mut ids = Vec::new();
        if let Some(mut iter) = reader.postings(&self.field, &self.term)? {
            while iter.next()? {
                ids.push(iter.doc_id());
            }
        }
        Ok(ids)
    }

    fn field_name(&self) -> &str {
        &self.field
    }

    fn clone_box(&self) -> Box<dyn SpanQuery> {
        Box::new(self.clone())
    }

    fn collect_highlight_terms(&self, out: &mut Vec<HighlightTerm>) {
        if !self.term.is_empty() {
            out.push(HighlightTerm::Exact(self.term.clone()));
        }
    }
}

/// A span query that matches terms near each other.
#[derive(Debug)]
pub struct SpanNearQuery {
    /// Field to search in
    field: String,
    /// Clauses that must appear near each other
    clauses: Vec<Box<dyn SpanQuery>>,
    /// Maximum distance between terms
    slop: u32,
    /// Whether terms must appear in order
    in_order: bool,
    /// Boost factor
    boost: f32,
}

impl SpanNearQuery {
    /// Create a new span near query.
    pub fn new<F: Into<String>>(
        field: F,
        clauses: Vec<Box<dyn SpanQuery>>,
        slop: u32,
        in_order: bool,
    ) -> Self {
        SpanNearQuery {
            field: field.into(),
            clauses,
            slop,
            in_order,
            boost: 1.0,
        }
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the slop (maximum distance).
    pub fn slop(&self) -> u32 {
        self.slop
    }

    /// Check if terms must be in order.
    pub fn is_in_order(&self) -> bool {
        self.in_order
    }

    /// Get the clauses.
    pub fn clauses(&self) -> &[Box<dyn SpanQuery>] {
        &self.clauses
    }
}

impl SpanQuery for SpanNearQuery {
    fn collect_highlight_terms(&self, out: &mut Vec<HighlightTerm>) {
        for clause in &self.clauses {
            clause.collect_highlight_terms(out);
        }
    }

    fn get_spans(&self, doc_id: u64, reader: &dyn LexicalIndexReader) -> Result<Vec<Span>> {
        let mut all_clause_spans = Vec::new();

        // Get spans for each clause
        for clause in &self.clauses {
            let clause_spans = clause.get_spans(doc_id, reader)?;
            all_clause_spans.push(clause_spans);
        }

        // Find combinations of spans that satisfy the proximity requirements
        let mut result_spans = Vec::new();
        self.find_near_spans(&all_clause_spans, 0, Vec::new(), &mut result_spans);

        Ok(result_spans)
    }

    fn candidate_doc_ids(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<u64>> {
        if self.clauses.is_empty() {
            return Ok(Vec::new());
        }
        // SpanNear requires all clauses to match — intersect candidates
        let mut candidates: std::collections::HashSet<u64> = self.clauses[0]
            .candidate_doc_ids(reader)?
            .into_iter()
            .collect();
        for clause in &self.clauses[1..] {
            let clause_ids: std::collections::HashSet<u64> =
                clause.candidate_doc_ids(reader)?.into_iter().collect();
            candidates = candidates.intersection(&clause_ids).copied().collect();
        }
        let mut result: Vec<u64> = candidates.into_iter().collect();
        result.sort_unstable();
        Ok(result)
    }

    fn field_name(&self) -> &str {
        &self.field
    }

    fn clone_box(&self) -> Box<dyn SpanQuery> {
        let cloned_clauses: Vec<Box<dyn SpanQuery>> = self
            .clauses
            .iter()
            .map(|clause| clause.clone_box())
            .collect();

        Box::new(SpanNearQuery {
            field: self.field.clone(),
            clauses: cloned_clauses,
            slop: self.slop,
            in_order: self.in_order,
            boost: self.boost,
        })
    }
}

impl SpanNearQuery {
    /// Recursively find combinations of spans that satisfy proximity requirements.
    fn find_near_spans(
        &self,
        all_clause_spans: &[Vec<Span>],
        clause_index: usize,
        current_spans: Vec<Span>,
        result_spans: &mut Vec<Span>,
    ) {
        if clause_index >= all_clause_spans.len() {
            // We have spans for all clauses, check if they satisfy proximity
            if self.spans_satisfy_proximity(&current_spans)
                && let Some(combined_span) = self.combine_spans(&current_spans)
            {
                result_spans.push(combined_span);
            }
            return;
        }

        // Try each span from the current clause
        for span in &all_clause_spans[clause_index] {
            let mut new_current = current_spans.clone();
            new_current.push(span.clone());
            self.find_near_spans(
                all_clause_spans,
                clause_index + 1,
                new_current,
                result_spans,
            );
        }
    }

    /// Check if a set of spans satisfies the proximity requirements.
    fn spans_satisfy_proximity(&self, spans: &[Span]) -> bool {
        if spans.len() < 2 {
            return true;
        }

        // Check order requirement
        if self.in_order {
            for i in 0..spans.len() - 1 {
                if spans[i].start > spans[i + 1].start {
                    return false;
                }
            }
        }

        let mut sorted_spans = spans.to_vec();
        sorted_spans.sort_by_key(|s| s.start);

        // Check slop requirement
        let total_span = Span::new(
            sorted_spans[0].start,
            sorted_spans.last().unwrap().end,
            "combined".to_string(),
        );

        let term_length: u32 = sorted_spans.iter().map(|s| s.length()).sum();
        let gaps = total_span.length().saturating_sub(term_length);

        gaps <= self.slop
    }

    /// Combine multiple spans into a single span covering all of them.
    fn combine_spans(&self, spans: &[Span]) -> Option<Span> {
        if spans.is_empty() {
            return None;
        }

        let start = spans.iter().map(|s| s.start).min().unwrap();
        let end = spans.iter().map(|s| s.end).max().unwrap();
        let terms: Vec<String> = spans.iter().map(|s| s.term.clone()).collect();
        let combined_term = format!("near({})", terms.join(","));

        Some(Span::new(start, end, combined_term))
    }
}

/// A span query that matches the first span that contains the second.
#[derive(Debug)]
pub struct SpanContainingQuery {
    /// Field to search in
    field: String,
    /// The containing span query
    big: Box<dyn SpanQuery>,
    /// The contained span query
    little: Box<dyn SpanQuery>,
    /// Boost factor
    boost: f32,
}

impl SpanContainingQuery {
    /// Create a new span containing query.
    pub fn new<F: Into<String>>(
        field: F,
        big: Box<dyn SpanQuery>,
        little: Box<dyn SpanQuery>,
    ) -> Self {
        SpanContainingQuery {
            field: field.into(),
            big,
            little,
            boost: 1.0,
        }
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }
}

impl SpanQuery for SpanContainingQuery {
    fn collect_highlight_terms(&self, out: &mut Vec<HighlightTerm>) {
        self.big.collect_highlight_terms(out);
        self.little.collect_highlight_terms(out);
    }

    fn get_spans(&self, doc_id: u64, reader: &dyn LexicalIndexReader) -> Result<Vec<Span>> {
        let big_spans = self.big.get_spans(doc_id, reader)?;
        let little_spans = self.little.get_spans(doc_id, reader)?;

        let mut result = Vec::new();

        for big_span in &big_spans {
            for little_span in &little_spans {
                if big_span.contains(little_span) {
                    result.push(big_span.clone());
                    break; // Only add each big span once
                }
            }
        }

        Ok(result)
    }

    fn candidate_doc_ids(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<u64>> {
        // Union: a document is a candidate if big OR little matches
        let mut candidates: std::collections::HashSet<u64> =
            self.big.candidate_doc_ids(reader)?.into_iter().collect();
        for id in self.little.candidate_doc_ids(reader)? {
            candidates.insert(id);
        }
        let mut result: Vec<u64> = candidates.into_iter().collect();
        result.sort_unstable();
        Ok(result)
    }

    fn field_name(&self) -> &str {
        &self.field
    }

    fn clone_box(&self) -> Box<dyn SpanQuery> {
        Box::new(SpanContainingQuery {
            field: self.field.clone(),
            big: self.big.clone_box(),
            little: self.little.clone_box(),
            boost: self.boost,
        })
    }
}

/// A span query that matches spans within a certain distance of another span.
#[derive(Debug)]
pub struct SpanWithinQuery {
    /// Field to search in
    field: String,
    /// The span query to match
    include: Box<dyn SpanQuery>,
    /// The span query that defines the boundaries
    exclude: Box<dyn SpanQuery>,
    /// Maximum distance from exclude spans
    distance: u32,
    /// Boost factor
    boost: f32,
}

impl SpanWithinQuery {
    /// Create a new span within query.
    pub fn new<F: Into<String>>(
        field: F,
        include: Box<dyn SpanQuery>,
        exclude: Box<dyn SpanQuery>,
        distance: u32,
    ) -> Self {
        SpanWithinQuery {
            field: field.into(),
            include,
            exclude,
            distance,
            boost: 1.0,
        }
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }
}

impl SpanQuery for SpanWithinQuery {
    fn collect_highlight_terms(&self, out: &mut Vec<HighlightTerm>) {
        // `exclude` bounds where `include` may appear; both are matched.
        self.include.collect_highlight_terms(out);
        self.exclude.collect_highlight_terms(out);
    }

    fn get_spans(&self, doc_id: u64, reader: &dyn LexicalIndexReader) -> Result<Vec<Span>> {
        let include_spans = self.include.get_spans(doc_id, reader)?;
        let exclude_spans = self.exclude.get_spans(doc_id, reader)?;

        let mut result = Vec::new();

        for include_span in &include_spans {
            let mut within_distance = false;

            for exclude_span in &exclude_spans {
                if include_span.distance_to(exclude_span) <= self.distance {
                    within_distance = true;
                    break;
                }
            }

            if within_distance {
                result.push(include_span.clone());
            }
        }

        Ok(result)
    }

    fn candidate_doc_ids(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<u64>> {
        // Union: include candidates union exclude candidates
        let mut candidates: std::collections::HashSet<u64> = self
            .include
            .candidate_doc_ids(reader)?
            .into_iter()
            .collect();
        for id in self.exclude.candidate_doc_ids(reader)? {
            candidates.insert(id);
        }
        let mut result: Vec<u64> = candidates.into_iter().collect();
        result.sort_unstable();
        Ok(result)
    }

    fn field_name(&self) -> &str {
        &self.field
    }

    fn clone_box(&self) -> Box<dyn SpanQuery> {
        Box::new(SpanWithinQuery {
            field: self.field.clone(),
            include: self.include.clone_box(),
            exclude: self.exclude.clone_box(),
            distance: self.distance,
            boost: self.boost,
        })
    }
}

/// A wrapper that adapts a SpanQuery to the regular Query interface.
#[derive(Debug)]
pub struct SpanQueryWrapper {
    /// The underlying span query
    span_query: Box<dyn SpanQuery>,
    /// Boost factor
    boost: f32,
}

impl SpanQueryWrapper {
    /// Create a new span query wrapper.
    pub fn new(span_query: Box<dyn SpanQuery>) -> Self {
        SpanQueryWrapper {
            span_query,
            boost: 1.0,
        }
    }

    /// Set the boost factor.
    pub fn boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the underlying span query.
    pub fn span_query(&self) -> &dyn SpanQuery {
        self.span_query.as_ref()
    }
}

impl SpanMatcher {
    pub fn new(span_query: Box<dyn SpanQuery>, reader: &dyn LexicalIndexReader) -> Result<Self> {
        // Enumerate actual doc_ids from posting lists and check if spans exist.
        let candidates = span_query.candidate_doc_ids(reader)?;
        let mut matches = Vec::new();
        for doc_id in candidates {
            let spans = span_query.get_spans(doc_id, reader)?;
            if !spans.is_empty() {
                matches.push(doc_id);
            }
        }

        // Initialize current_index=0 like PhraseMatcher so that is_exhausted() works correctly from the start.
        let current_doc_id = matches.first().copied().unwrap_or(u64::MAX);
        Ok(SpanMatcher {
            matches,
            current_index: 0,
            current_doc_id,
        })
    }
}

/// A matcher that finds documents containing spans.
#[derive(Debug)]
pub struct SpanMatcher {
    /// Matching document IDs.
    matches: Vec<u64>,
    /// Current position in the matches.
    current_index: usize,
    /// Current document ID.
    current_doc_id: u64,
}

impl Matcher for SpanMatcher {
    fn doc_id(&self) -> u64 {
        if self.current_index >= self.matches.len() {
            u64::MAX
        } else {
            self.current_doc_id
        }
    }

    fn next(&mut self) -> Result<bool> {
        self.current_index += 1;

        if self.current_index >= self.matches.len() {
            self.current_doc_id = u64::MAX;
            Ok(false)
        } else {
            self.current_doc_id = self.matches[self.current_index];
            Ok(true)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        while self.current_index < self.matches.len() && self.matches[self.current_index] < target {
            self.current_index += 1;
        }

        if self.current_index >= self.matches.len() {
            self.current_doc_id = u64::MAX;
            Ok(false)
        } else {
            self.current_doc_id = self.matches[self.current_index];
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

/// A scorer for span queries.
#[derive(Debug)]
pub struct SpanScorer {
    #[allow(dead_code)]
    span_query: Box<dyn SpanQuery>,
    // We cannot store reader here either!
    // Query::scorer returns Box<dyn Scorer>.
    // Scorer::score(doc_id, ...) does NOT take reader.
    // So SpanScorer CANNOT call `get_spans(doc_id, reader)`.
    // This implies `SpanScorer` must ALSO pre-calculate scores or have access to pre-calculated data?
    // PhraseScorer calculates `phrase_doc_freq` map upfront!
    // So `SpanScorer` must behave similarly: Pre-calculate everything in `new`.
    scores: HashMap<u64, f32>,
    boost: f32,
}

impl SpanScorer {
    pub fn new(
        span_query: Box<dyn SpanQuery>,
        reader: &dyn LexicalIndexReader,
        boost: f32,
    ) -> Result<Self> {
        // Enumerate actual doc_ids from posting lists and pre-compute scores.
        let candidates = span_query.candidate_doc_ids(reader)?;
        let mut scores = HashMap::new();
        for doc_id in candidates {
            let spans = span_query.get_spans(doc_id, reader)?;
            if !spans.is_empty() {
                let score = (spans.len() as f32) * boost;
                scores.insert(doc_id, score);
            }
        }

        Ok(SpanScorer {
            span_query,
            scores,
            boost,
        })
    }
}

impl Scorer for SpanScorer {
    fn score(&self, doc_id: u64, _term_freq: f32, _field_length: Option<f32>) -> f32 {
        *self.scores.get(&doc_id).unwrap_or(&0.0)
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        // Approximate
        self.boost * 100.0
    }

    fn name(&self) -> &'static str {
        "SpanScorer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Query for SpanQueryWrapper {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        let matches = SpanMatcher::new(self.span_query.clone_box(), reader)?;
        Ok(Box::new(matches))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        let scorer = SpanScorer::new(self.span_query.clone_box(), reader, self.boost)?;
        Ok(Box::new(scorer))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(SpanQueryWrapper {
            span_query: self.span_query.clone_box(),
            boost: self.boost,
        })
    }

    fn description(&self) -> String {
        format!("SpanQueryWrapper({})", self.span_query.field_name())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        // For now, assume span queries are never empty
        Ok(false)
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        // Return a default cost for now
        Ok(1)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn collect_field_refs(&self, out: &mut std::collections::HashSet<String>) {
        out.insert(self.span_query.field_name().to_string());
    }

    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        if field.is_none_or(|f| f == self.span_query.field_name()) {
            self.span_query.collect_highlight_terms(out);
        }
    }
}

/// Builder for creating complex span queries.
#[derive(Debug)]
pub struct SpanQueryBuilder {
    field: String,
}

impl SpanQueryBuilder {
    /// Create a new span query builder.
    pub fn new<F: Into<String>>(field: F) -> Self {
        SpanQueryBuilder {
            field: field.into(),
        }
    }

    /// Create a span term query.
    pub fn term<T: Into<String>>(&self, term: T) -> SpanTermQuery {
        SpanTermQuery::new(&self.field, term)
    }

    /// Create a span near query.
    pub fn near(
        &self,
        clauses: Vec<Box<dyn SpanQuery>>,
        slop: u32,
        in_order: bool,
    ) -> SpanNearQuery {
        SpanNearQuery::new(&self.field, clauses, slop, in_order)
    }

    /// Create a span containing query.
    pub fn containing(
        &self,
        big: Box<dyn SpanQuery>,
        little: Box<dyn SpanQuery>,
    ) -> SpanContainingQuery {
        SpanContainingQuery::new(&self.field, big, little)
    }

    /// Create a span within query.
    pub fn within(
        &self,
        include: Box<dyn SpanQuery>,
        exclude: Box<dyn SpanQuery>,
        distance: u32,
    ) -> SpanWithinQuery {
        SpanWithinQuery::new(&self.field, include, exclude, distance)
    }

    /// Create a phrase query using span near with zero slop.
    pub fn phrase(&self, terms: Vec<String>) -> SpanNearQuery {
        let clauses: Vec<Box<dyn SpanQuery>> = terms
            .into_iter()
            .map(|term| Box::new(self.term(term)) as Box<dyn SpanQuery>)
            .collect();

        self.near(clauses, 0, true)
    }

    /// Create a proximity query using span near.
    pub fn proximity(&self, terms: Vec<String>, slop: u32) -> SpanNearQuery {
        let clauses: Vec<Box<dyn SpanQuery>> = terms
            .into_iter()
            .map(|term| Box::new(self.term(term)) as Box<dyn SpanQuery>)
            .collect();

        self.near(clauses, slop, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_operations() {
        let span1 = Span::new(5, 10, "hello".to_string());
        let span2 = Span::new(8, 12, "world".to_string());
        let span3 = Span::new(15, 20, "test".to_string());

        assert_eq!(span1.length(), 5);
        assert!(span1.overlaps(&span2));
        assert!(!span1.overlaps(&span3));
        assert_eq!(span1.distance_to(&span3), 5);
        assert_eq!(span1.distance_to(&span2), 0); // overlapping
    }

    #[test]
    fn test_span_containment() {
        let big_span = Span::new(0, 20, "sentence".to_string());
        let small_span = Span::new(5, 10, "word".to_string());
        let outside_span = Span::new(25, 30, "other".to_string());

        assert!(big_span.contains(&small_span));
        assert!(!big_span.contains(&outside_span));
        assert!(!small_span.contains(&big_span));
    }

    #[test]
    fn test_span_term_query() {
        let query = SpanTermQuery::new("content", "hello").boost(2.0);

        assert_eq!(query.field_name(), "content");
        assert_eq!(query.term(), "hello");
        assert_eq!(query.boost, 2.0);
    }

    #[test]
    fn test_span_near_query() {
        let term1 = Box::new(SpanTermQuery::new("content", "hello")) as Box<dyn SpanQuery>;
        let term2 = Box::new(SpanTermQuery::new("content", "world")) as Box<dyn SpanQuery>;

        let near_query = SpanNearQuery::new("content", vec![term1, term2], 5, true);

        assert_eq!(near_query.field_name(), "content");
        assert_eq!(near_query.slop(), 5);
        assert!(near_query.is_in_order());
        assert_eq!(near_query.clauses().len(), 2);
    }

    #[test]
    fn test_span_query_builder() {
        let builder = SpanQueryBuilder::new("content");

        // Test term query
        let term_query = builder.term("hello");
        assert_eq!(term_query.field_name(), "content");
        assert_eq!(term_query.term(), "hello");

        // Test phrase query
        let phrase_query = builder.phrase(vec!["hello".to_string(), "world".to_string()]);
        assert_eq!(phrase_query.field_name(), "content");
        assert_eq!(phrase_query.slop(), 0);
        assert!(phrase_query.is_in_order());

        // Test proximity query
        let proximity_query = builder.proximity(vec!["hello".to_string(), "world".to_string()], 10);
        assert_eq!(proximity_query.field_name(), "content");
        assert_eq!(proximity_query.slop(), 10);
        assert!(!proximity_query.is_in_order());
    }

    #[test]
    fn test_span_query_wrapper() {
        let span_query = Box::new(SpanTermQuery::new("content", "hello")) as Box<dyn SpanQuery>;
        let wrapper = SpanQueryWrapper::new(span_query).boost(1.5);

        assert_eq!(wrapper.boost, 1.5);
        assert_eq!(wrapper.span_query().field_name(), "content");
    }

    #[test]
    fn test_span_proximity_checking() {
        let query = SpanNearQuery::new(
            "content",
            vec![], // Empty for this test
            3,      // slop = 3
            false,  // not in order
        );

        // Test spans that should satisfy proximity (distance = 2, which is <= slop of 3)
        let spans = vec![
            Span::new(0, 1, "hello".to_string()),
            Span::new(3, 4, "world".to_string()),
        ];
        assert!(query.spans_satisfy_proximity(&spans));

        // Test spans that should not satisfy proximity (distance = 5, which is > slop of 3)
        let spans = vec![
            Span::new(0, 1, "hello".to_string()),
            Span::new(6, 7, "world".to_string()),
        ];
        assert!(!query.spans_satisfy_proximity(&spans));
    }
}
