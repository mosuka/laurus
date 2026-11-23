//! Multi-term query support.
//!
//! This module provides traits and utilities for queries that match multiple terms,
//! similar to Lucene's MultiTermQuery.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::lexical::index::inverted::core::terms::TermsEnum;
use crate::lexical::query::Query;
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
/// MultiTermQuery implementations should:
/// 1. Enumerate matching terms from the index's term dictionary
/// 2. Apply rewrite strategies to convert to simpler queries (e.g., BooleanQuery)
/// 3. Limit the number of expanded terms to prevent resource exhaustion
///
/// # Example (conceptual - not fully implemented yet)
///
/// ```ignore
/// use laurus::lexical::query::multi_term::MultiTermQuery;
/// use laurus::lexical::query::fuzzy::FuzzyQuery;
///
/// let fuzzy_query = FuzzyQuery::new("content", "hello").max_edits(2);
///
/// // The query will enumerate terms from the index that match within edit distance 2
/// let reader = index.reader()?;
/// let matching_terms = fuzzy_query.enumerate_terms(&reader)?;
///
/// // Results are limited by max_expansions (default 50)
/// println!("Found {} matching terms", matching_terms.len());
/// ```
pub trait MultiTermQuery: Query {
    /// Get the field name this query searches in.
    fn field(&self) -> &str;

    /// Get the rewrite method.
    fn rewrite_method(&self) -> RewriteMethod;

    /// Enumerate terms from the index that match this query's criteria.
    ///
    /// This method should:
    /// 1. Access the term dictionary for the field
    /// 2. Iterate over terms that potentially match
    /// 3. Filter terms based on the query's criteria (e.g., edit distance, pattern)
    /// 4. Limit results to max_expansions
    /// 5. Return terms sorted by relevance/score
    ///
    /// # Arguments
    ///
    /// * `reader` - The index reader to enumerate terms from
    ///
    /// # Returns
    ///
    /// A vector of tuples containing:
    /// - `term`: The matching term text
    /// - `doc_freq`: Number of documents containing this term
    /// - `boost`: Optional boost factor for this term (default 1.0)
    ///
    /// # Performance
    ///
    /// Implementations should use efficient term dictionary enumeration rather than
    /// scanning all documents. See the `TermsEnum` trait for the proper API.
    fn enumerate_terms(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<(String, u64, f32)>>;

    /// Get the maximum number of terms this query will expand to.
    ///
    /// This prevents queries from matching too many terms and consuming
    /// excessive resources. Default should be 50, same as Lucene.
    fn max_expansions(&self) -> usize {
        50
    }

    /// Create a TermsEnum that filters terms according to this query's criteria.
    ///
    /// This is the preferred method for implementing efficient multi-term queries.
    /// Instead of enumerating all terms and filtering in memory, this creates
    /// a TermsEnum that only yields matching terms.
    ///
    /// # Example (conceptual)
    ///
    /// ```ignore
    /// // For a FuzzyQuery:
    /// fn get_terms_enum(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn TermsEnum>> {
    ///     let terms = reader.terms(self.field)?;
    ///     let automaton = LevenshteinAutomaton::build(&self.term, self.max_edits);
    ///     Ok(Box::new(AutomatonTermsEnum::new(terms, automaton)))
    /// }
    /// ```
    fn get_terms_enum(
        &self,
        _reader: &dyn LexicalIndexReader,
    ) -> Result<Option<Box<dyn TermsEnum>>> {
        // Default implementation returns None, indicating that enumerate_terms()
        // should be used instead. Implementations should override this for better performance.
        Ok(None)
    }

    /// Rewrite the query into a simpler form (e.g., BooleanQuery).
    fn rewrite(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Query>> {
        let rewrite_method = self.rewrite_method();

        // Try to get TermsEnum
        let terms_enum_opt = self.get_terms_enum(reader)?;

        let matching_terms = if let Some(mut terms_enum) = terms_enum_opt {
            match rewrite_method {
                RewriteMethod::TopTermsScoring { max_expansions }
                | RewriteMethod::TopTermsBlended { max_expansions } => {
                    // Use priority queue to collect top terms
                    collect_top_terms(&mut *terms_enum, max_expansions)?
                }
                _ => {
                    // Collect all terms
                    let mut terms = Vec::new();
                    while let Some(term_stats) = terms_enum.next()? {
                        terms.push((term_stats.term.clone(), term_stats.doc_freq, 1.0));
                    }
                    terms
                }
            }
        } else {
            // Fallback to legacy enumerate_terms
            self.enumerate_terms(reader)?
        };

        if matching_terms.is_empty() {
            // We return a BooleanQuery which will produce EmptyMatcher if empty.
            use crate::lexical::query::boolean::BooleanQuery;
            return Ok(Box::new(BooleanQuery::new()));
        }

        use crate::lexical::query::boolean::{BooleanClause, BooleanQuery, Occur};
        use crate::lexical::query::term::TermQuery;

        let mut boolean_query = BooleanQuery::new();
        boolean_query.set_boost(self.boost());

        match rewrite_method {
            RewriteMethod::TopTermsScoring { .. } => {
                for (term, _, _) in matching_terms {
                    let term_query = TermQuery::new(MultiTermQuery::field(self).to_string(), term);
                    boolean_query
                        .add_clause(BooleanClause::new(Box::new(term_query), Occur::Should));
                }
            }
            RewriteMethod::TopTermsBlended { .. } => {
                // TODO: Implement proper blended scoring (adjusting IDF across terms)
                // For now, treat same as TopTermsScoring but typically boosts might vary
                for (term, _, _) in matching_terms {
                    let term_query = TermQuery::new(MultiTermQuery::field(self).to_string(), term);
                    boolean_query
                        .add_clause(BooleanClause::new(Box::new(term_query), Occur::Should));
                }
            }
            RewriteMethod::ConstantScore => {
                for (term, _, _) in matching_terms {
                    let term_query = TermQuery::new(MultiTermQuery::field(self).to_string(), term);
                    boolean_query
                        .add_clause(BooleanClause::new(Box::new(term_query), Occur::Should));
                }
            }
            RewriteMethod::BooleanQuery => {
                for (term, _, _) in matching_terms {
                    let term_query = TermQuery::new(MultiTermQuery::field(self).to_string(), term);
                    boolean_query
                        .add_clause(BooleanClause::new(Box::new(term_query), Occur::Should));
                }
            }
        }

        Ok(Box::new(boolean_query))
    }
}

// Helper struct for priority queue
#[derive(PartialEq)]
struct ScoredTerm {
    term: String,
    doc_freq: u64,
    boost: f32,
}

impl Eq for ScoredTerm {}

impl PartialOrd for ScoredTerm {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredTerm {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse doc_freq comparison: Lower DF > Higher DF
        other
            .doc_freq
            .cmp(&self.doc_freq)
            .then_with(|| self.term.cmp(&other.term)) // Tie-break by term
    }
}

fn collect_top_terms(
    terms_enum: &mut dyn TermsEnum,
    max_expansions: usize,
) -> Result<Vec<(String, u64, f32)>> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut heap = BinaryHeap::with_capacity(max_expansions + 1);

    while let Some(term_stats) = terms_enum.next()? {
        let scored_term = ScoredTerm {
            term: term_stats.term.clone(),
            doc_freq: term_stats.doc_freq,
            boost: 1.0,
        };

        heap.push(Reverse(scored_term));

        if heap.len() > max_expansions {
            heap.pop(); // Remove the smallest item (Lowest Score)
        }
    }

    let mut results = Vec::with_capacity(heap.len());
    while let Some(Reverse(scored_term)) = heap.pop() {
        results.push((scored_term.term, scored_term.doc_freq, scored_term.boost));
    }

    Ok(results)
}

/// Rewrite strategies for multi-term queries.
///
/// Similar to Lucene's RewriteMethod, these strategies determine how a
/// multi-term query is converted into a simpler form for execution.
///
/// # Strategies
///
/// - **TopTermsRewrite**: Collect the top N scoring terms (default)
/// - **ConstantScoreRewrite**: All matching terms get the same score
/// - **BooleanRewrite**: Convert to BooleanQuery with all matching terms
///
/// The choice of strategy affects both performance and scoring behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewriteMethod {
    /// Collect the top N terms by score, then create a BooleanQuery.
    /// This is the default and most performant for scoring queries.
    ///
    /// Similar to Lucene's TopTermsScoringBooleanQueryRewrite.
    TopTermsScoring { max_expansions: usize },

    /// Collect the top N terms by document frequency, assign constant score.
    /// Good for filtering without needing accurate scores.
    ///
    /// Similar to Lucene's TopTermsBlendedFreqScoringRewrite.
    TopTermsBlended { max_expansions: usize },

    /// All matching terms get a constant score equal to the query boost.
    /// Most efficient when you don't need term-specific scoring.
    ///
    /// Similar to Lucene's CONSTANT_SCORE_REWRITE.
    ConstantScore,

    /// Convert to BooleanQuery with all matching terms.
    /// May hit max clause count limits for queries matching many terms.
    ///
    /// Similar to Lucene's SCORING_BOOLEAN_REWRITE.
    BooleanQuery,
}

impl Default for RewriteMethod {
    fn default() -> Self {
        // Use TopTermsBlended as default, same as Lucene's FuzzyQuery
        RewriteMethod::TopTermsBlended { max_expansions: 50 }
    }
}

impl RewriteMethod {
    /// Get the maximum number of terms to expand to, if applicable.
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
