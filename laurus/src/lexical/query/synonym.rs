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
//!
//! An alternative can weigh its occurrences less than the others
//! ([`SynonymQuery::weighted`], Issue #1704), as Lucene's
//! `SynonymQuery.Builder.addTerm(term, boost)` does: the combined frequency
//! is then the sum of each matching alternative's weight times its term
//! frequency. The query parser gives each alternative its token boost
//! divided by the position's largest, which it puts on the query's boost,
//! so a synonym that `SynonymGraphFilter::with_boost` weighs at 0.8
//! counts each of its occurrences as 0.8 of the original's.

use crate::error::{LaurusError, Result};
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::matcher::{DisjunctionMatcher, EmptyMatcher, Matcher, PostingMatcher};
use crate::lexical::query::scorer::{BM25Scorer, Scorer};
use crate::lexical::query::{HighlightTerm, Query, check_query_boost, is_valid_boost};
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
    /// Each alternative's weight, in the order of `terms`, or `None` when
    /// every alternative weighs 1 (an unweighted query).
    weights: Option<Vec<f32>>,
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
            weights: None,
            stats: None,
        }
    }

    /// Create a synonym query whose alternatives weigh their occurrences by
    /// `weight`, as Lucene's `SynonymQuery.Builder.addTerm(term, boost)`
    /// does: a document's combined term frequency is the sum, over the
    /// alternatives it holds, of the alternative's weight times its term
    /// frequency (Issue #1704).
    ///
    /// A weight is expected in [0, 1]; [`Query::validate_boosts`] rejects
    /// one that is not finite or is out of that range, so a search with it
    /// fails. A duplicated term keeps its larger weight. When every weight
    /// is 1, the query is the same as [`SynonymQuery::new`]'s.
    pub fn weighted<F, T>(field: F, terms: Vec<(T, f32)>) -> Self
    where
        F: Into<String>,
        T: Into<String>,
    {
        let mut deduped: Vec<String> = Vec::with_capacity(terms.len());
        let mut weights: Vec<f32> = Vec::with_capacity(terms.len());
        for (term, weight) in terms {
            let term = term.into();
            match deduped.iter().position(|t| *t == term) {
                Some(i) => {
                    if weight > weights[i] {
                        weights[i] = weight;
                    }
                }
                None => {
                    deduped.push(term);
                    weights.push(weight);
                }
            }
        }
        let weights = (!weights.iter().all(|&w| w == 1.0)).then_some(weights);
        SynonymQuery {
            field: field.into(),
            terms: deduped,
            boost: 1.0,
            weights,
            stats: None,
        }
    }

    /// Get the alternatives' weights, in the order of
    /// [`terms`](Self::terms), or `None` when every alternative weighs 1.
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// The weight of the alternative at `index`.
    fn weight(&self, index: usize) -> f32 {
        self.weights.as_ref().map_or(1.0, |weights| weights[index])
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
        if self.weights.is_some() {
            let mut survivors: Vec<(PostingMatcher, f32)> = Vec::with_capacity(self.terms.len());
            for (i, term) in self.terms.iter().enumerate() {
                if let Some(posting_iter) = reader.postings(&self.field, term)? {
                    survivors.push((PostingMatcher::new(posting_iter), self.weight(i)));
                }
            }
            // The surviving alternatives may all weigh 1 (the lighter ones
            // absent from this reader); then the unweighted matchers below
            // serve, and the matcher is the same as an unweighted query's.
            if survivors.iter().any(|&(_, weight)| weight != 1.0) {
                // The matcher has to report the weighted frequency (Issue
                // #1704), even for a single survivor.
                return Ok(Box::new(WeightedSynonymMatcher::new(survivors)));
            }
            return Ok(unweighted_matcher(
                survivors
                    .into_iter()
                    .map(|(matcher, _)| Box::new(matcher) as Box<dyn Matcher>)
                    .collect(),
            ));
        }
        let mut matchers: Vec<Box<dyn Matcher>> = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            if let Some(posting_iter) = reader.postings(&self.field, term)? {
                matchers.push(Box::new(PostingMatcher::new(posting_iter)));
            }
        }
        Ok(unweighted_matcher(matchers))
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
                // valid for any term frequency, the weighted one
                // (`WeightedSynonymMatcher`, Issue #1704) included.
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
        // A weight shows only when it is not 1, so an unweighted query's
        // description is the same as before weights existed.
        let alternatives = match &self.weights {
            None => self.terms.join("|"),
            Some(weights) => self
                .terms
                .iter()
                .zip(weights)
                .map(|(term, &weight)| {
                    if weight == 1.0 {
                        term.clone()
                    } else {
                        format!("{term}^{weight}")
                    }
                })
                .collect::<Vec<_>>()
                .join("|"),
        };
        if self.boost == 1.0 {
            format!("{}:synonym({})", self.field, alternatives)
        } else {
            format!("{}:synonym({})^{}", self.field, alternatives, self.boost)
        }
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    /// Check the query's boost (Issue #1550) and each alternative's weight,
    /// which must be finite and in [0, 1] (Issue #1704). Lucene's
    /// `SynonymQuery` takes (0, 1]; a weight of 0 is accepted here because
    /// a token boost of 0 is.
    fn validate_boosts(&self) -> Result<()> {
        check_query_boost(self.boost, || self.description())?;
        for (term, &weight) in self.terms.iter().zip(self.weights.iter().flatten()) {
            if !(is_valid_boost(weight) && weight <= 1.0) {
                return Err(LaurusError::invalid_argument(format!(
                    "synonym weight must be finite and between 0 and 1, got {weight} for \
                     {term:?}: {}",
                    self.description()
                )));
            }
        }
        Ok(())
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
        // set; boost and the alternatives' weights only scale scores (an
        // alternative of weight 0 still matches) and are excluded, as
        // `TermQuery` excludes its boost. Sorting a copy means two equal
        // alternative sets in different orders share a cache entry.
        let mut sorted = self.terms.clone();
        sorted.sort();
        Some(format!("synonym|{:?}|{:?}", self.field, sorted))
    }
}

/// The matcher of an unweighted query over the matchers of its surviving
/// alternatives.
fn unweighted_matcher(mut matchers: Vec<Box<dyn Matcher>>) -> Box<dyn Matcher> {
    match matchers.len() {
        0 => Box::new(EmptyMatcher::new()),
        // A single surviving alternative keeps the specialized
        // `LeafMatcher::Posting` arm in a parent `BooleanScorer` instead of
        // the generic vtable arm `DisjunctionMatcher` would force.
        1 => matchers.remove(0),
        _ => Box::new(DisjunctionMatcher::new(matchers)),
    }
}

/// Matches the union of a weighted synonym's alternatives, as
/// [`DisjunctionMatcher`] does, and reports the weighted frequency: the
/// sum, over the alternatives on the current document, of each one's weight
/// times its term frequency (Issue #1704).
///
/// A synonym holds a handful of alternatives, so a linear scan for the
/// smallest document replaces `DisjunctionMatcher`'s heap. Exhausted
/// alternatives are dropped in place, keeping the others in term order, so
/// the weighted sum adds them in the same order on every document and in
/// every segment.
#[derive(Debug)]
pub(crate) struct WeightedSynonymMatcher {
    /// The alternatives not yet exhausted, in term order, with their
    /// weights.
    alternatives: Vec<(PostingMatcher, f32)>,
    /// The smallest document of the alternatives, `u64::MAX` once every
    /// one is exhausted.
    current_doc: u64,
    /// The sum of the alternatives' costs.
    cost: u64,
}

impl WeightedSynonymMatcher {
    /// Create the matcher over the alternatives' posting matchers, each
    /// positioned on its first document, with their weights.
    pub(crate) fn new(mut alternatives: Vec<(PostingMatcher, f32)>) -> Self {
        alternatives.retain(|(matcher, _)| !matcher.is_exhausted());
        let cost = alternatives.iter().map(|(matcher, _)| matcher.cost()).sum();
        let mut matcher = WeightedSynonymMatcher {
            alternatives,
            current_doc: u64::MAX,
            cost,
        };
        matcher.settle();
        matcher
    }

    /// Drop the exhausted alternatives and move to the smallest document of
    /// the others.
    fn settle(&mut self) {
        self.alternatives
            .retain(|(matcher, _)| !matcher.is_exhausted());
        self.current_doc = self
            .alternatives
            .iter()
            .map(|(matcher, _)| matcher.doc_id())
            .min()
            .unwrap_or(u64::MAX);
    }
}

impl Matcher for WeightedSynonymMatcher {
    fn doc_id(&self) -> u64 {
        self.current_doc
    }

    fn next(&mut self) -> Result<bool> {
        if self.alternatives.is_empty() {
            return Ok(false);
        }
        let current_doc = self.current_doc;
        for (matcher, _) in &mut self.alternatives {
            if matcher.doc_id() == current_doc {
                matcher.next()?;
            }
        }
        self.settle();
        Ok(!self.alternatives.is_empty())
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        if self.alternatives.is_empty() || target <= self.current_doc {
            return Ok(!self.alternatives.is_empty());
        }
        for (matcher, _) in &mut self.alternatives {
            if matcher.doc_id() < target {
                matcher.skip_to(target)?;
            }
        }
        self.settle();
        Ok(!self.alternatives.is_empty())
    }

    fn cost(&self) -> u64 {
        self.cost
    }

    fn is_exhausted(&self) -> bool {
        self.alternatives.is_empty()
    }

    fn term_freq(&self) -> u64 {
        self.alternatives
            .iter()
            .filter(|(matcher, _)| matcher.doc_id() == self.current_doc)
            .map(|(matcher, _)| matcher.term_freq())
            .sum()
    }

    fn weighted_term_freq(&self) -> f32 {
        let mut total = 0.0_f32;
        for (matcher, weight) in &self.alternatives {
            if matcher.doc_id() == self.current_doc {
                total += weight * matcher.term_freq() as f32;
            }
        }
        total
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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

    /// A posting matcher over `(doc_id, term_freq)` postings.
    fn posting_matcher(postings: &[(u64, u32)]) -> PostingMatcher {
        use crate::lexical::index::inverted::core::posting::Posting;
        use crate::lexical::index::inverted::reader::InvertedIndexPostingIterator;
        PostingMatcher::new(Box::new(InvertedIndexPostingIterator::new(
            postings
                .iter()
                .map(|&(doc_id, freq)| Posting::with_frequency(doc_id, freq))
                .collect(),
        )))
    }

    /// A reader over one segment of `docs`, each in the field `title`.
    fn reader_of(docs: &[&str]) -> Arc<dyn LexicalIndexReader> {
        use crate::Document;
        use crate::lexical::InvertedIndexConfig;
        use crate::lexical::index::LexicalIndex;
        use crate::lexical::index::inverted::InvertedIndex;

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let index = InvertedIndex::create(storage, InvertedIndexConfig::default()).unwrap();
        let mut writer = index.writer().unwrap();
        for text in docs {
            writer
                .add_document(Document::builder().add_text("title", *text).build())
                .unwrap();
        }
        writer.commit().unwrap();
        index.reader().unwrap()
    }

    /// Every (document, score) the query's own matcher and scorer give.
    fn own_scores(query: &SynonymQuery, reader: &dyn LexicalIndexReader) -> Vec<(u64, f32)> {
        let mut matcher = query.matcher(reader).unwrap();
        let scorer = query.scorer(reader).unwrap();
        let mut scores = Vec::new();
        while !matcher.is_exhausted() {
            let doc_id = matcher.doc_id();
            scores.push((
                doc_id,
                scorer.score(doc_id, matcher.weighted_term_freq(), None),
            ));
            matcher.next().unwrap();
        }
        scores
    }

    #[test]
    fn the_weighted_matcher_matches_as_a_disjunction_and_weighs_each_alternative() {
        // xorshift: deterministic postings for three alternatives.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let weights = [1.0_f32, 0.8, 0.25];
        for round in 0..200 {
            let lists: Vec<Vec<(u64, u32)>> = (0..3)
                .map(|_| {
                    let mut doc_id = 0;
                    (0..next() % 12)
                        .map(|_| {
                            doc_id += 1 + next() % 4;
                            (doc_id, 1 + (next() % 3) as u32)
                        })
                        .collect()
                })
                .collect();
            let mut plain = DisjunctionMatcher::new(
                lists
                    .iter()
                    .map(|list| Box::new(posting_matcher(list)) as Box<dyn Matcher>)
                    .collect(),
            );
            let mut weighted = WeightedSynonymMatcher::new(
                lists
                    .iter()
                    .zip(weights)
                    .map(|(list, weight)| (posting_matcher(list), weight))
                    .collect(),
            );
            loop {
                assert_eq!(weighted.doc_id(), plain.doc_id(), "round {round}");
                assert_eq!(
                    weighted.is_exhausted(),
                    plain.is_exhausted(),
                    "round {round}"
                );
                if plain.is_exhausted() {
                    break;
                }
                let doc_id = plain.doc_id();
                assert_eq!(weighted.term_freq(), plain.term_freq(), "round {round}");
                let expected: f32 = lists
                    .iter()
                    .zip(weights)
                    .filter_map(|(list, weight)| {
                        list.iter()
                            .find(|&&(d, _)| d == doc_id)
                            .map(|&(_, freq)| weight * freq as f32)
                    })
                    .sum();
                assert_eq!(weighted.weighted_term_freq(), expected, "round {round}");
                if next() % 3 == 0 {
                    let target = doc_id + 1 + next() % 5;
                    assert_eq!(
                        weighted.skip_to(target).unwrap(),
                        plain.skip_to(target).unwrap(),
                        "round {round}"
                    );
                } else {
                    assert_eq!(
                        weighted.next().unwrap(),
                        plain.next().unwrap(),
                        "round {round}"
                    );
                }
            }
        }
    }

    /// A `ConjunctionNotMatcher` (a clause with a must-not clause) reports
    /// the frequency its positive matcher weighs, as it reports its term
    /// frequency.
    #[test]
    fn a_conjunction_not_matcher_forwards_the_weighted_frequency() {
        use crate::lexical::query::matcher::ConjunctionNotMatcher;

        let positive =
            WeightedSynonymMatcher::new(vec![(posting_matcher(&[(1, 2), (3, 1)]), 0.25)]);
        let negative = posting_matcher(&[(3, 1)]);
        let matcher = ConjunctionNotMatcher::new(Box::new(positive), vec![Box::new(negative)]);
        assert_eq!(matcher.doc_id(), 1);
        assert_eq!(matcher.term_freq(), 2);
        assert_eq!(matcher.weighted_term_freq(), 0.5);
    }

    #[test]
    fn the_matcher_is_weighted_only_when_a_surviving_alternative_is() {
        let reader = reader_of(&["big dog", "large dog"]);
        let kind = |query: SynonymQuery| {
            let matcher = query.matcher(reader.as_ref()).unwrap();
            let any = matcher.as_any();
            if any.is::<WeightedSynonymMatcher>() {
                "weighted"
            } else if any.is::<DisjunctionMatcher>() {
                "disjunction"
            } else if any.is::<PostingMatcher>() {
                "posting"
            } else {
                "other"
            }
        };
        let cases = [
            (
                SynonymQuery::new("title", vec!["big", "large"]),
                "disjunction",
            ),
            (
                SynonymQuery::weighted("title", vec![("big", 1.0), ("large", 1.0)]),
                "disjunction",
            ),
            (
                SynonymQuery::weighted("title", vec![("big", 1.0), ("large", 0.5)]),
                "weighted",
            ),
            // Only the full-weight alternative is in the index.
            (
                SynonymQuery::weighted("title", vec![("big", 1.0), ("huge", 0.5)]),
                "posting",
            ),
            // Only a lighter one is: it still weighs its occurrences.
            (
                SynonymQuery::weighted("title", vec![("huge", 1.0), ("large", 0.5)]),
                "weighted",
            ),
        ];
        let actual: Vec<&str> = cases.iter().map(|(query, _)| kind(query.clone())).collect();
        let expected: Vec<&str> = cases.iter().map(|&(_, kind)| kind).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn validate_boosts_checks_each_weight() {
        let check =
            |weight: f32| SynonymQuery::weighted("title", vec![("big", 1.0), ("large", weight)]);
        let mut wrong = Vec::new();
        for weight in [f32::NAN, f32::INFINITY, -1.0, -0.0, 1.5] {
            if check(weight).validate_boosts().is_ok() {
                wrong.push(format!("accepted {weight}"));
            }
        }
        for weight in [0.0, 0.5, 1.0] {
            if let Err(e) = check(weight).validate_boosts() {
                wrong.push(format!("rejected {weight}: {e}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:?}");
        let message = check(1.5).validate_boosts().unwrap_err().to_string();
        assert!(
            message.contains("synonym weight") && message.contains("\"large\""),
            "{message}"
        );
    }

    #[test]
    fn a_weighted_query_keeps_the_larger_weight_and_shows_its_weights() {
        let query = SynonymQuery::weighted(
            "title",
            vec![("big", 1.0), ("large", 0.5), ("big", 0.25), ("large", 0.8)],
        );
        assert_eq!(query.terms(), &["big".to_string(), "large".to_string()]);
        assert_eq!(query.weights(), Some(&[1.0, 0.8][..]));
        assert_eq!(query.description(), "title:synonym(big|large^0.8)");
        // The weights do not change the matched set, so not the cache key.
        assert_eq!(
            query.cache_key(),
            SynonymQuery::new("title", vec!["large", "big"]).cache_key()
        );
        assert_eq!(
            SynonymQuery::weighted("title", vec![("big", 1.0), ("large", 1.0)]).weights(),
            None
        );
        let reader = empty_reader();
        let rewritten = query.rewrite(&reader).unwrap().expect("should rewrite");
        let rewritten = rewritten
            .as_any()
            .downcast_ref::<SynonymQuery>()
            .expect("a synonym query");
        assert_eq!(rewritten.weights(), Some(&[1.0, 0.8][..]));
    }

    /// The scorers that score a clause's matcher read its weighted
    /// frequency: `DisjunctionMaxScorer` and `AdvancedQuery::execute` give
    /// a weighted synonym's own scores (Issue #1704).
    #[test]
    fn dis_max_and_advanced_query_score_the_weighted_frequency() {
        use crate::lexical::query::advanced_query::AdvancedQuery;
        use crate::lexical::query::scorer::DisjunctionMaxScorer;

        let reader = reader_of(&["big dog", "large dog", "small cat"]);
        let query = SynonymQuery::weighted("title", vec![("big", 1.0), ("large", 0.5)]);
        let own = own_scores(&query, reader.as_ref());
        assert_eq!(own.len(), 2, "{own:?}");
        assert!(
            own[1].1 < own[0].1,
            "the lighter alternative scores less: {own:?}"
        );

        let dis_max = DisjunctionMaxScorer::new(
            reader.as_ref(),
            vec![Box::new(query.clone()) as Box<dyn Query>],
            0.0,
            1.0,
        )
        .unwrap();
        let dis_max_scores: Vec<(u64, f32)> = own
            .iter()
            .map(|&(doc_id, _)| (doc_id, dis_max.score(doc_id, 0.0, None)))
            .collect();
        assert_eq!(dis_max_scores, own, "DisjunctionMaxScorer");

        let mut advanced: Vec<(u64, f32)> = AdvancedQuery::new(Box::new(query))
            .execute(reader.as_ref())
            .unwrap()
            .into_iter()
            .map(|result| (result.doc_id, result.score))
            .collect();
        advanced.sort_by_key(|&(doc_id, _)| doc_id);
        assert_eq!(advanced, own, "AdvancedQuery::execute");
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
