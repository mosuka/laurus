//! Query system for searching documents in inverted indexes.

pub mod advanced_query;
pub mod boolean;
pub mod collector;
pub mod fuzzy;
pub mod geo;
pub mod geo3d;
pub mod matcher;
pub mod multi_term;
pub mod parser;
pub mod phrase;
pub mod prefix;
pub mod range;
pub mod regexp;
pub mod scorer;
pub mod span;
pub mod term;
pub mod wildcard;

// Re-exports for cleaner API
pub use advanced_query::AdvancedQuery;
pub use boolean::{BooleanQuery, BooleanQueryBuilder};
pub use fuzzy::FuzzyQuery;
pub use geo::{GeoBoundingBox, GeoBoundingBoxQuery, GeoDistanceQuery, GeoPoint};
pub use geo3d::{
    Geo3dBoundingBoxQuery, Geo3dDistanceQuery, Geo3dMatch, Geo3dMatcher, Geo3dNearestQuery,
    Geo3dScorer,
};
pub use multi_term::MultiTermQuery;
pub use parser::LexicalQueryParser;
pub use phrase::PhraseQuery;
pub use prefix::PrefixQuery;
pub use range::{DateTimeRangeQuery, NumericRangeQuery};
pub use regexp::RegexpQuery;
pub use span::{SpanNearQuery, SpanQuery, SpanTermQuery};
pub use term::TermQuery;
pub use wildcard::WildcardQuery;

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::Arc;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::lexical::index::inverted::core::automaton::LevenshteinAutomaton;

use crate::error::Result;
#[allow(unused_imports)]
use crate::lexical::core::document::Document;
use crate::lexical::reader::LexicalIndexReader;

use self::matcher::Matcher;
use self::scorer::Scorer;

/// A search hit containing a document and its score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    /// The document ID.
    pub doc_id: u64,
    /// The relevance score.
    pub score: f32,
    /// The document fields (if retrieved).
    pub fields: HashMap<String, String>,
}

/// A single search hit containing a matched document and its relevance score.
///
/// Returned as part of [`LexicalSearchResults`] to represent each document
/// that matched the search query. The `score` reflects the relevance ranking
/// computed by the scorer (e.g., BM25), and the `document` field optionally
/// holds the stored fields if they were requested.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// The internal document ID.
    pub doc_id: u64,
    /// The relevance score.
    pub score: f32,
    /// The document (if retrieved).
    pub document: Option<Document>,
}

/// Aggregated results from a lexical search query.
///
/// Contains the ranked list of matching documents along with summary statistics.
///
/// # Fields
///
/// - `hits` - Ranked list of [`SearchHit`] entries, ordered by descending score.
/// - `total_hits` - Total number of documents that matched the query (may exceed `hits.len()`
///   when a limit is applied).
/// - `max_score` - The highest relevance score among all results, useful for normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalSearchResults {
    /// The search hits.
    pub hits: Vec<SearchHit>,
    /// Total number of matching documents.
    pub total_hits: u64,
    /// Maximum score in the results.
    pub max_score: f32,
}

/// Query result wrapper for different result types.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Document ID.
    pub doc_id: u64,
    /// Score.
    pub score: f32,
}

/// What a query would highlight, expressed over analyzed tokens so the
/// highlighter needs no index access. Produced by
/// [`Query::collect_highlight_terms`].
#[derive(Debug, Clone)]
pub enum HighlightTerm {
    /// One analyzed term, matched against each token.
    Exact(String),
    /// Ordered terms matched on token positions with the same in-order,
    /// per-gap `slop` rule as the phrase matcher.
    Phrase {
        /// The phrase's terms, in order.
        terms: Vec<String>,
        /// Maximum position gap allowed between consecutive terms.
        slop: u32,
    },
    /// Tokens starting with this prefix.
    Prefix(String),
    /// Tokens matching this compiled regex (wildcard and regexp queries).
    Regex(Arc<Regex>),
    /// Tokens within the automaton's edit distance (fuzzy queries).
    Fuzzy(LevenshteinAutomaton),
}

/// Trait for search queries.
pub trait Query: Send + Sync + Debug {
    /// Create a matcher for this query.
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>>;

    /// Create a scorer for this query.
    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>>;

    /// Create the matcher and scorer for this query in one pass.
    ///
    /// The default implementation builds them independently via
    /// [`matcher`](Query::matcher) and [`scorer`](Query::scorer),
    /// preserving the historical behavior. Queries whose matcher and
    /// scorer derive from the same expensive candidate computation
    /// (e.g. geo queries computing per-document distances) override
    /// this to run that computation once and build both from the
    /// shared result (#996).
    ///
    /// # Arguments
    ///
    /// * `reader` - The index reader to search.
    ///
    /// # Returns
    ///
    /// The `(matcher, scorer)` pair for this query.
    fn matcher_scorer(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<(Box<dyn Matcher>, Box<dyn Scorer>)> {
        Ok((self.matcher(reader)?, self.scorer(reader)?))
    }

    /// Get the boost factor for this query.
    fn boost(&self) -> f32;

    /// Set the boost factor for this query.
    fn set_boost(&mut self, boost: f32);

    /// Get a human-readable description of this query.
    fn description(&self) -> String;

    /// Clone this query.
    fn clone_box(&self) -> Box<dyn Query>;

    /// Returns `true` if this query would match no documents in the given reader.
    ///
    /// Each implementor defines its own emptiness semantics. For example:
    /// - [`TermQuery`] checks whether
    ///   the term exists in the index via the reader.
    /// - [`BooleanQuery`] returns `true`
    ///   when it has no clauses or all of its clauses are empty.
    ///
    /// # Parameters
    ///
    /// - `reader` - The index reader used to check whether the query's terms exist.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if this query would not match any documents, `Ok(false)` otherwise.
    /// Returns an error if the reader cannot be queried.
    fn is_empty(&self, reader: &dyn LexicalIndexReader) -> Result<bool>;

    /// Get the estimated cost of executing this query.
    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64>;

    /// Rewrite this query against the reader (Issue #613).
    ///
    /// Called once by the searcher at the top of query execution,
    /// against the top-level reader — before the per-segment fanout and
    /// BMW gates. Multi-term queries (prefix / wildcard / fuzzy /
    /// regexp) override this to lower themselves into a `BooleanQuery`
    /// of `TermQuery` clauses via one term-dictionary enumeration, so
    /// the subsequent `matcher` + `scorer` construction (and every
    /// per-segment execution) reuses the lowered form instead of
    /// re-enumerating. [`crate::lexical::query::boolean::BooleanQuery`]
    /// recurses into its scoring clauses.
    ///
    /// # Arguments
    ///
    /// * `reader` - The index reader the query will execute against.
    ///
    /// # Returns
    ///
    /// `Ok(Some(query))` with the rewritten query, or `Ok(None)` when
    /// this query has nothing to rewrite (the default — zero cost for
    /// term / phrase / range queries) or the reader does not support
    /// term enumeration (the caller keeps the original query, whose
    /// `matcher` / `scorer` fallback behavior is unchanged).
    fn rewrite(&self, reader: &dyn LexicalIndexReader) -> Result<Option<Box<dyn Query>>> {
        let _ = reader;
        Ok(None)
    }

    /// Get this query as Any for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Get the field name this query searches in, if applicable.
    /// Returns None for queries that don't target a specific field (e.g., BooleanQuery).
    fn field(&self) -> Option<&str> {
        None
    }

    /// Collect every field name referenced by this query and its sub-queries.
    ///
    /// The default implementation inserts `self.field()` into `out` if it
    /// returns `Some`. Composite queries that hold child queries (e.g.
    /// [`BooleanQuery`](crate::lexical::query::BooleanQuery), span queries)
    /// override this method to recurse into each child so every leaf field
    /// reference contributes.
    ///
    /// This is the authoritative way for callers (such as
    /// [`UnifiedQueryParser`](crate::engine::query::UnifiedQueryParser)) to
    /// validate that every field a query references is declared in the
    /// schema, replacing earlier regex-based heuristics.
    ///
    /// # Arguments
    ///
    /// * `out` - The set to populate with field names. Existing entries are
    ///   preserved; new names are inserted.
    fn collect_field_refs(&self, out: &mut HashSet<String>) {
        if let Some(f) = self.field() {
            out.insert(f.to_string());
        }
    }

    /// Collect what this query would highlight, expressed over analyzed
    /// tokens (#594).
    ///
    /// `field == Some(f)` keeps only leaves targeting `f`; `None` keeps
    /// every leaf. Composites recurse into positive clauses only: a
    /// `MustNot` clause or a negative filter describes what a hit does
    /// *not* contain. Queries without text terms (range, numeric, geo)
    /// keep this no-op default.
    ///
    /// # Arguments
    ///
    /// * `field` - Restrict to leaves targeting this field, or `None` for all.
    /// * `out` - Terms are appended in query-tree order.
    fn collect_highlight_terms(&self, field: Option<&str>, out: &mut Vec<HighlightTerm>) {
        let _ = (field, out);
    }

    /// Apply field-level boosts to this query and its sub-queries.
    fn apply_field_boosts(&mut self, boosts: &HashMap<String, f32>) {
        if let Some(f) = self.field()
            && let Some(&b) = boosts.get(f)
        {
            self.set_boost(self.boost() * b);
        }
    }

    /// Return a stable, canonical cache key identifying the **set of documents**
    /// this query matches within a reader snapshot, or `None` if the query must
    /// not be cached.
    ///
    /// This key backs the snapshot-scoped filter/doc-id result cache
    /// (see [`QueryFilterCache`](crate::lexical::index::inverted::query_cache::QueryFilterCache)).
    /// It is **score-independent**: two queries that match the same document set
    /// must produce the same key, and the query's boost — which only scales
    /// scores, never changes membership — is deliberately excluded.
    ///
    /// # Correctness contract
    ///
    /// Returning `Some(key)` is a promise that the key is *canonical*: any two
    /// query instances that would match different document sets MUST yield
    /// different keys. Implementors that cannot guarantee this (e.g. a key built
    /// from a non-deterministic `HashMap` iteration order, or one that omits a
    /// membership-affecting parameter) MUST return `None` so the query bypasses
    /// the cache and is evaluated fresh. The default is `None` (not cacheable),
    /// which is always safe.
    ///
    /// Keys are namespaced with a per-type tag (e.g. `"term|…"`) so that two
    /// different query types can never collide in the shared cache map.
    ///
    /// # Returns
    ///
    /// `Some(canonical_key)` if this query may be cached, `None` otherwise.
    fn cache_key(&self) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod highlight_term_tests {
    use super::*;
    use crate::lexical::index::inverted::core::automaton::Automaton;
    use crate::lexical::query::advanced_query::MultiFieldQuery;
    use crate::lexical::query::range::RangeQuery;
    use crate::lexical::query::span::{
        SpanNearQuery, SpanQueryWrapper, SpanTermQuery, SpanWithinQuery,
    };

    fn collect(query: &dyn Query, field: Option<&str>) -> Vec<HighlightTerm> {
        let mut out = Vec::new();
        query.collect_highlight_terms(field, &mut out);
        out
    }

    fn exact(terms: &[HighlightTerm]) -> Vec<&str> {
        terms
            .iter()
            .map(|term| match term {
                HighlightTerm::Exact(text) => text.as_str(),
                other => panic!("expected Exact, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn term_query_is_gated_by_field_and_skips_empty_terms() {
        let query = TermQuery::new("body", "rust");
        assert_eq!(exact(&collect(&query, Some("body"))), ["rust"]);
        assert_eq!(exact(&collect(&query, None)), ["rust"]);
        assert!(collect(&query, Some("title")).is_empty());
        assert!(collect(&TermQuery::new("body", ""), None).is_empty());
    }

    #[test]
    fn phrase_query_keeps_terms_and_slop() {
        let query = PhraseQuery::new("body", vec!["hello".into(), "world".into()]).with_slop(2);
        match collect(&query, Some("body")).as_slice() {
            [HighlightTerm::Phrase { terms, slop }] => {
                assert_eq!(terms, &["hello", "world"]);
                assert_eq!(*slop, 2);
            }
            other => panic!("expected one Phrase, got {other:?}"),
        }
        assert!(collect(&query, Some("title")).is_empty());
        assert!(collect(&PhraseQuery::new("body", Vec::new()), None).is_empty());
    }

    #[test]
    fn multi_term_queries_yield_their_matcher_shapes() {
        assert!(matches!(
            collect(&PrefixQuery::new("body", "ru"), Some("body")).as_slice(),
            [HighlightTerm::Prefix(prefix)] if prefix == "ru"
        ));
        assert!(collect(&PrefixQuery::new("body", ""), None).is_empty());

        let wildcard = WildcardQuery::new("body", "r?st").unwrap();
        assert!(matches!(
            collect(&wildcard, Some("body")).as_slice(),
            [HighlightTerm::Regex(regex)] if regex.is_match("rust") && !regex.is_match("roast")
        ));

        let regexp = RegexpQuery::new("body", "^ru.*").unwrap();
        assert!(matches!(
            collect(&regexp, Some("body")).as_slice(),
            [HighlightTerm::Regex(regex)] if regex.is_match("rusty") && !regex.is_match("trust")
        ));

        let fuzzy = FuzzyQuery::new("body", "rust").max_edits(1);
        assert!(matches!(
            collect(&fuzzy, Some("body")).as_slice(),
            [HighlightTerm::Fuzzy(automaton)] if automaton.matches("rusty") && !automaton.matches("roast")
        ));
        assert!(collect(&fuzzy, Some("title")).is_empty());
    }

    #[test]
    fn boolean_query_skips_must_not_but_keeps_filters() {
        let mut query = BooleanQuery::new();
        query.add_must(Box::new(TermQuery::new("body", "rust")));
        query.add_must_not(Box::new(TermQuery::new("body", "java")));
        query.add_filter(Box::new(TermQuery::new("body", "search")));
        let mut nested = BooleanQuery::new();
        nested.add_should(Box::new(TermQuery::new("body", "engine")));
        nested.add_should(Box::new(TermQuery::new("title", "engine")));
        query.add_should(Box::new(nested));

        assert_eq!(
            exact(&collect(&query, Some("body"))),
            ["rust", "search", "engine"]
        );
        assert_eq!(
            exact(&collect(&query, None)),
            ["rust", "search", "engine", "engine"]
        );
    }

    #[test]
    fn advanced_query_skips_negative_filters() {
        let query = AdvancedQuery::new(Box::new(TermQuery::new("body", "rust")))
            .with_filter(Box::new(TermQuery::new("body", "fast")))
            .with_negative_filter(Box::new(TermQuery::new("body", "slow")))
            .with_post_filter(Box::new(TermQuery::new("body", "safe")));
        assert_eq!(
            exact(&collect(&query, Some("body"))),
            ["rust", "fast", "safe"]
        );
    }

    #[test]
    fn multi_field_query_yields_its_text_for_configured_fields_only() {
        let query = MultiFieldQuery::new("rust".to_string())
            .add_field("title".to_string(), 2.0)
            .add_field("body".to_string(), 1.0);
        assert_eq!(exact(&collect(&query, Some("body"))), ["rust"]);
        assert_eq!(exact(&collect(&query, None)), ["rust"]);
        assert!(collect(&query, Some("tags")).is_empty());
    }

    #[test]
    fn span_wrapper_collects_every_span_term_in_its_field() {
        let clauses: Vec<Box<dyn SpanQuery>> = vec![
            Box::new(SpanTermQuery::new("body", "rust")),
            Box::new(SpanTermQuery::new("body", "safety")),
        ];
        let near = SpanNearQuery::new("body", clauses, 3, true);
        let within = SpanWithinQuery::new(
            "body",
            Box::new(near),
            Box::new(SpanTermQuery::new("body", "memory")),
            5,
        );
        let query = SpanQueryWrapper::new(Box::new(within));
        assert_eq!(
            exact(&collect(&query, Some("body"))),
            ["rust", "safety", "memory"]
        );
        assert!(collect(&query, Some("title")).is_empty());
    }

    #[test]
    fn range_query_yields_nothing() {
        let query = RangeQuery::new("body", Some("a".to_string()), Some("z".to_string()));
        assert!(collect(&query, None).is_empty());
    }
}

#[cfg(test)]
mod leaf_field_tests {
    use super::*;
    use crate::data::GeoEcefPoint;
    use crate::lexical::core::field::NumericType;
    use crate::lexical::query::range::{DateTimeRangeQuery, RangeQuery};

    /// Every single-field leaf query type that is neither a term, phrase,
    /// prefix nor fuzzy query, built on `field` (#1131).
    fn leaf_queries(field: &str) -> Vec<(&'static str, Box<dyn Query>)> {
        let point = || GeoPoint {
            lat: 35.0,
            lon: 139.0,
        };
        let ecef = |v: f64| GeoEcefPoint { x: v, y: v, z: v };
        let bbox = GeoBoundingBox::new(
            GeoPoint {
                lat: 34.0,
                lon: 138.0,
            },
            GeoPoint {
                lat: 36.0,
                lon: 140.0,
            },
        )
        .unwrap();

        vec![
            (
                "WildcardQuery",
                Box::new(WildcardQuery::new(field, "fo?").unwrap()),
            ),
            (
                "RegexpQuery",
                Box::new(RegexpQuery::new(field, "^fo.*").unwrap()),
            ),
            (
                "RangeQuery",
                Box::new(RangeQuery::new(
                    field,
                    Some("a".to_string()),
                    Some("z".to_string()),
                )),
            ),
            (
                "NumericRangeQuery",
                Box::new(NumericRangeQuery::new(
                    field,
                    NumericType::Float,
                    Some(1.0),
                    Some(10.0),
                    true,
                    true,
                )),
            ),
            (
                "DateTimeRangeQuery",
                Box::new(DateTimeRangeQuery::new(field, None, None, true, true)),
            ),
            (
                "GeoDistanceQuery",
                Box::new(GeoDistanceQuery::new(field, point(), 1_000.0)),
            ),
            (
                "GeoBoundingBoxQuery",
                Box::new(GeoBoundingBoxQuery::new(field, bbox)),
            ),
            (
                "Geo3dDistanceQuery",
                Box::new(Geo3dDistanceQuery::new(field, ecef(1.0), 1_000.0)),
            ),
            (
                "Geo3dBoundingBoxQuery",
                Box::new(Geo3dBoundingBoxQuery::new(field, ecef(0.0), ecef(1.0)).unwrap()),
            ),
            (
                "Geo3dNearestQuery",
                Box::new(Geo3dNearestQuery::new(field, ecef(1.0), 10)),
            ),
        ]
    }

    /// `Query::field()` feeds `collect_field_refs`, which schema validation
    /// walks: a leaf that stays at the `None` default is invisible to it.
    #[test]
    fn every_single_field_leaf_query_reports_its_field() {
        for (name, query) in leaf_queries("price") {
            assert_eq!(query.field(), Some("price"), "{name}::field()");

            let mut refs = HashSet::new();
            query.collect_field_refs(&mut refs);
            assert_eq!(
                refs,
                HashSet::from(["price".to_string()]),
                "{name}::collect_field_refs"
            );
        }
    }

    /// `apply_field_boosts` scales a leaf's boost only when its field is
    /// in the map — so it must know its field.
    #[test]
    fn field_boosts_reach_every_single_field_leaf_query() {
        let matching: HashMap<String, f32> = HashMap::from([("price".to_string(), 3.0)]);
        for (name, mut query) in leaf_queries("price") {
            let before = query.boost();
            query.apply_field_boosts(&matching);
            assert_eq!(query.boost(), before * 3.0, "{name}: boost must be scaled");
        }

        let other: HashMap<String, f32> = HashMap::from([("title".to_string(), 3.0)]);
        for (name, mut query) in leaf_queries("price") {
            let before = query.boost();
            query.apply_field_boosts(&other);
            assert_eq!(
                query.boost(),
                before,
                "{name}: a boost for another field must not apply"
            );
        }
    }
}
