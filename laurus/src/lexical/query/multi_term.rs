//! Multi-term query support.
//!
//! This module provides traits and utilities for queries that match multiple terms,
//! similar to Lucene's MultiTermQuery.
//!
//! Which documents a multi-term query matches does not depend on how it is
//! scored (Issue #1740): its matcher covers every term its automaton matches,
//! whatever their number, and each segment looks those terms up in its own
//! dictionary. The [`RewriteMethod`] decides the scores: the boost for every
//! match, or the BM25 scores of some or all of the terms. A search for the
//! query itself returns the documents that are scored, so a method that limits
//! the terms limits its hits too.

use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::lexical::index::inverted::core::automaton::Automaton;
use crate::lexical::index::inverted::reader::InvertedIndexReader;
use crate::lexical::query::boolean::{BooleanClause, BooleanQuery, Occur};
use crate::lexical::query::matcher::{EmptyMatcher, Matcher, PreComputedMatcher};
use crate::lexical::query::scorer::{ConstantScorer, Scorer};
use crate::lexical::query::term::TermQuery;
use crate::lexical::query::{Query, check_query_boost};
use crate::lexical::reader::LexicalIndexReader;

/// A query that matches multiple terms based on some pattern or criteria.
///
/// This is the base trait for queries like:
/// - `FuzzyQuery`: matches terms within edit distance
/// - `PrefixQuery`: matches terms with a common prefix
/// - `WildcardQuery`: matches terms matching a wildcard pattern
/// - `RegexpQuery`: matches terms matching a regular expression
///
/// Similar to Lucene's MultiTermQuery, this trait provides a common interface
/// for queries that need to enumerate and match against the term dictionary.
///
/// # Design
///
/// An implementation says which terms match, with its
/// [`automaton`](Self::automaton), and how the matches are scored, with its
/// [`rewrite_method`](Self::rewrite_method).
///
/// # Example (conceptual)
///
/// ```ignore
/// use laurus::lexical::query::multi_term::MultiTermQuery;
/// use laurus::lexical::query::prefix::PrefixQuery;
///
/// let query = PrefixQuery::new("content", "hel");
///
/// // Every term of the field that starts with "hel"
/// let reader = index.reader()?;
/// let matching_terms = query.enumerate_terms(&reader)?;
/// println!("Found {} matching terms", matching_terms.len());
/// ```
pub trait MultiTermQuery: Query {
    /// Get the field name this query searches in.
    fn field(&self) -> &str;

    /// Get the rewrite method.
    fn rewrite_method(&self) -> RewriteMethod;

    /// Build the automaton that decides which terms this query matches.
    ///
    /// # Errors
    ///
    /// Returns an error if the query's pattern is invalid.
    fn automaton(&self) -> Result<Box<dyn Automaton>>;

    /// The factor of the score of a term that this query matches, in the
    /// rewrite methods that score terms. The terms with the highest factors
    /// are scored first. The default is the same for every term, `1.0`.
    ///
    /// # Arguments
    ///
    /// * `term` - A term the query's automaton matches.
    ///
    /// # Returns
    ///
    /// A finite factor of at least `0`.
    fn term_boost(&self, term: &str) -> f32 {
        let _ = term;
        1.0
    }

    /// Enumerate every term of the index that this query matches.
    ///
    /// # Arguments
    ///
    /// * `reader` - The index reader to enumerate terms from
    ///
    /// # Returns
    ///
    /// A vector of tuples in ascending order of the terms, containing:
    /// - `term`: The matching term text
    /// - `doc_freq`: Number of documents containing this term
    /// - `boost`: The factor of the term's score (see
    ///   [`term_boost`](Self::term_boost))
    ///
    /// It is empty if the reader cannot enumerate terms.
    ///
    /// # Errors
    ///
    /// Returns an error if the automaton cannot be built or a term dictionary
    /// cannot be read.
    fn enumerate_terms(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<(String, u64, f32)>> {
        let automaton = self.automaton()?;
        let terms = reader
            .matching_terms(MultiTermQuery::field(self), automaton.as_ref())?
            .unwrap_or_default();
        Ok(terms
            .into_iter()
            .map(|(term, doc_freq)| {
                let boost = self.term_boost(&term);
                (term, doc_freq, boost)
            })
            .collect())
    }

    /// Rewrite the query into a `BooleanQuery` of the terms its
    /// [`rewrite_method`](Self::rewrite_method) scores: the top ones of a
    /// method with a limit, otherwise every term.
    ///
    /// # Arguments
    ///
    /// * `reader` - The index reader to enumerate terms from
    ///
    /// # Errors
    ///
    /// Returns an error if the terms cannot be enumerated.
    fn rewrite(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Query>> {
        let mut terms = self.enumerate_terms(reader)?;
        if let Some(limit) = self.rewrite_method().max_expansions() {
            keep_top_terms(&mut terms, limit);
        }
        let blended = match self.rewrite_method() {
            RewriteMethod::TopTermsBlended { .. } => {
                terms.iter().map(|(_, doc_freq, _)| *doc_freq).max()
            }
            _ => None,
        };
        // Without terms, the query produces an EmptyMatcher.
        let mut boolean_query = BooleanQuery::new();
        boolean_query.set_boost(self.boost());
        let field = MultiTermQuery::field(self);
        for (term, _, boost) in terms {
            let mut term_query = TermQuery::new(field, term).with_boost(boost);
            if let Some(doc_freq) = blended {
                term_query = term_query.with_doc_freq(doc_freq);
            }
            boolean_query.add_clause(BooleanClause::new(Box::new(term_query), Occur::Should));
        }
        Ok(Box::new(boolean_query))
    }
}

/// Keep the `limit` terms that rank best: by boost, then by document
/// frequency, then in the order of the terms.
fn keep_top_terms(terms: &mut Vec<(String, u64, f32)>, limit: usize) {
    if terms.len() <= limit {
        return;
    }
    if limit > 0 {
        terms.select_nth_unstable_by(limit - 1, |a, b| {
            b.2.total_cmp(&a.2)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| a.0.cmp(&b.0))
        });
    }
    terms.truncate(limit);
}

/// The matcher of a multi-term query: every document that holds a term its
/// automaton matches.
///
/// # Arguments
///
/// * `query` - The multi-term query.
/// * `reader` - The reader whose documents are matched.
///
/// # Errors
///
/// Returns an error if the automaton cannot be built, or a dictionary or a
/// posting list cannot be read.
pub(crate) fn multi_term_matcher<Q: MultiTermQuery>(
    query: &Q,
    reader: &dyn LexicalIndexReader,
) -> Result<Box<dyn Matcher>> {
    let automaton = query.automaton()?;
    let field = MultiTermQuery::field(query);
    Ok(
        match reader.docs_of_matching_terms(field, automaton.as_ref())? {
            Some(docs) => Box::new(PreComputedMatcher::new(docs.iter().collect())),
            None => Box::new(EmptyMatcher::new()),
        },
    )
}

/// The scorer of a multi-term query.
///
/// # Arguments
///
/// * `query` - The multi-term query.
/// * `reader` - The reader whose documents are scored.
///
/// # Errors
///
/// Returns an error if the terms to score cannot be enumerated.
pub(crate) fn multi_term_scorer<Q: MultiTermQuery>(
    query: &Q,
    reader: &dyn LexicalIndexReader,
) -> Result<Box<dyn Scorer>> {
    if query.rewrite_method().is_constant_score() {
        return Ok(Box::new(ConstantScorer::with_boost(1.0, query.boost())));
    }
    MultiTermQuery::rewrite(query, reader)?.scorer(reader)
}

/// What [`Query::rewrite`] returns for a multi-term query.
///
/// `None` for a constant score, which needs no statistics of the index, and
/// for a reader of one segment, whose statistics are not the index's: the
/// matcher and the scorer then work on the query itself.
///
/// # Arguments
///
/// * `query` - The multi-term query.
/// * `reader` - The reader the search runs against.
///
/// # Errors
///
/// Returns an error if the terms cannot be enumerated.
pub(crate) fn rewrite_multi_term<Q: MultiTermQuery>(
    query: &Q,
    reader: &dyn LexicalIndexReader,
) -> Result<Option<Box<dyn Query>>> {
    if query.rewrite_method().is_constant_score() || !reader.as_any().is::<InvertedIndexReader>() {
        return Ok(None);
    }
    Ok(Some(MultiTermQuery::rewrite(query, reader)?))
}

/// What [`Query::validate_boosts`] checks for a multi-term query: its boost,
/// and that its rewrite method scores any term.
///
/// # Arguments
///
/// * `query` - The multi-term query.
///
/// # Errors
///
/// Returns [`LaurusError::InvalidArgument`] for a boost that is not finite or
/// is negative, or for a limit of no terms, which would find nothing without
/// an error (Issue #1740).
pub(crate) fn validate_multi_term<Q: MultiTermQuery>(query: &Q) -> Result<()> {
    check_query_boost(query.boost(), || query.description())?;
    if query.rewrite_method().max_expansions() == Some(0) {
        return Err(LaurusError::invalid_argument(format!(
            "max_expansions must be at least 1, got 0: {}",
            query.description()
        )));
    }
    Ok(())
}

/// Rewrite strategies for multi-term queries.
///
/// Similar to Lucene's RewriteMethod, these strategies determine how the
/// matches of a multi-term query are scored.
///
/// # Strategies
///
/// - **ConstantScore** (the default of a prefix, wildcard or regexp query):
///   every match scores the query's boost
/// - **TopTermsScoring** and **TopTermsBlended**: the top N terms are scored
/// - **BooleanQuery**: every term is scored
///
/// The choice affects the scores, the cost of a search, and with a limit the
/// documents a search for the query returns. A filter and an exclusion use the
/// set of matches alone, which covers every term with each of them.
///
/// A limit of `0` is rejected by a search (see [`Query::validate_boosts`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewriteMethod {
    /// Score the top N terms as the clauses of a BooleanQuery, each with its
    /// factor (see [`MultiTermQuery::term_boost`]): the terms with the highest
    /// factors, among equal ones the most frequent, and among equally frequent
    /// ones the first.
    ///
    /// Similar to Lucene's TopTermsScoringBooleanQueryRewrite.
    TopTermsScoring { max_expansions: usize },

    /// Score the top N terms as `TopTermsScoring` does, but each as if it were
    /// in as many documents as the most frequent of them. Otherwise the rarest
    /// term typically ranks highest. The default of a fuzzy query, with 50.
    ///
    /// Similar to Lucene's TopTermsBlendedFreqScoringRewrite.
    TopTermsBlended { max_expansions: usize },

    /// Every match scores the query's boost, whatever its term and however
    /// many terms match.
    ///
    /// Similar to Lucene's CONSTANT_SCORE_REWRITE.
    ConstantScore,

    /// Score every matching term as a clause of a BooleanQuery.
    ///
    /// The cost of a search grows faster than the number of terms, to
    /// minutes for 10,000 terms in a large segment, and nothing limits it.
    ///
    /// Similar to Lucene's SCORING_BOOLEAN_REWRITE, which has a limit on the
    /// clauses.
    BooleanQuery,
}

impl Default for RewriteMethod {
    /// The default of a prefix, wildcard or regexp query, as in Lucene.
    fn default() -> Self {
        RewriteMethod::ConstantScore
    }
}

impl RewriteMethod {
    /// Get the maximum number of terms that are scored, if the method has one.
    pub fn max_expansions(&self) -> Option<usize> {
        match self {
            RewriteMethod::TopTermsScoring { max_expansions } => Some(*max_expansions),
            RewriteMethod::TopTermsBlended { max_expansions } => Some(*max_expansions),
            RewriteMethod::ConstantScore => None,
            RewriteMethod::BooleanQuery => None,
        }
    }

    /// Check if this rewrite method uses constant scoring.
    pub fn is_constant_score(&self) -> bool {
        matches!(self, RewriteMethod::ConstantScore)
    }

    /// Check if this rewrite method limits the number of expanded terms.
    pub fn is_top_terms(&self) -> bool {
        matches!(
            self,
            RewriteMethod::TopTermsScoring { .. } | RewriteMethod::TopTermsBlended { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The boost ranks first, then the document frequency, then the order of
    /// the terms.
    #[test]
    fn top_terms_rank_by_boost_then_frequency_then_term() {
        let terms = || {
            vec![
                ("a".to_string(), 1, 1.0),
                ("b".to_string(), 9, 0.5),
                ("c".to_string(), 3, 1.0),
                ("d".to_string(), 3, 1.0),
                ("e".to_string(), 2, 1.0),
            ]
        };
        for (limit, expected) in [
            (0, vec![]),
            (1, vec!["c"]),
            (2, vec!["c", "d"]),
            (3, vec!["c", "d", "e"]),
            (4, vec!["a", "c", "d", "e"]),
            (5, vec!["a", "b", "c", "d", "e"]),
            (9, vec!["a", "b", "c", "d", "e"]),
        ] {
            let mut kept = terms();
            keep_top_terms(&mut kept, limit);
            let mut kept: Vec<&str> = kept.iter().map(|(term, _, _)| term.as_str()).collect();
            kept.sort_unstable();
            assert_eq!(kept, expected, "limit {limit}");
        }
    }
}
