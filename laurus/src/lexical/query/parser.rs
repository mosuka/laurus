//! Query parser using pest.
//!
//! This parser supports the full query syntax including:
//! - Field-specific queries: `title:hello`
//! - Boolean operators: `AND`, `OR`, only in capitals and as whole words
//!   (#1747): `and`, `Or` and `ANDROID` are terms
//! - Required/prohibited: `+required`, `-forbidden`; `NOT forbidden`, in
//!   capitals, is `-forbidden` (#1747), and `NOT` with no clause after it is
//!   a term
//! - Phrases: `"hello world"` (quoted → `PhraseQuery`; a quoted value that
//!   analyzes to one token → `TermQuery`). Tokens the analyzer stacks at one
//!   position (synonyms) are alternatives there, and a multi-word synonym
//!   gives one query per path, OR'd.
//! - Bare terms that the analyzer splits into several tokens are OR'd
//!   (`BooleanQuery` with `Should` clauses), matching Lucene's `match`
//!   query. Use quotes for phrase semantics.
//! - Terms removed entirely by analysis match no documents in that field.
//! - Proximity search: `"hello world"~10`
//! - Fuzzy search: `roam~2`
//! - Range queries: `[100 TO 500]`, `{A TO Z}`, and datetime ranges such as
//!   `[2024-01-01 TO 2024-12-31]` or `[2024-01-01T00:00:00Z TO *]` (#1179)
//! - Wildcards: `te?t`, `test*`
//! - Boosting: `jakarta^4`; a group's boost multiplies the boosts inside
//!   it, so `(jakarta^2 apache)^3` boosts `jakarta` by 6 (#1568)
//! - Grouping: `(title:hello OR body:world)`

use std::collections::BTreeMap;
use std::sync::Arc;

use pest::Parser;
use pest_derive::Parser;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::per_field::PerFieldAnalyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::analysis::token::{Token, flatten_token_graph, token_arcs, token_positions};
use crate::data::GeoEcefPoint;
use crate::error::{LaurusError, Result};
use crate::lexical::core::datetime::parse_datetime_literal;
use crate::lexical::core::field::NumericType;
use crate::lexical::query::Query;
use crate::lexical::query::boolean::{BooleanClause, BooleanQuery, Occur};
use crate::lexical::query::fuzzy::FuzzyQuery;
use crate::lexical::query::geo::{GeoBoundingBoxQuery, GeoDistanceQuery};
use crate::lexical::query::geo3d::{Geo3dBoundingBoxQuery, Geo3dDistanceQuery, Geo3dNearestQuery};
use crate::lexical::query::graph_phrase::{
    GraphPhraseQuery, PhraseArc, complete_paths_with_origins,
};
use crate::lexical::query::phrase::PhraseQuery;
use crate::lexical::query::range::{Bound, DateTimeRangeQuery, NumericRangeQuery, RangeQuery};
use crate::lexical::query::synonym::SynonymQuery;
use crate::lexical::query::term::TermQuery;
use crate::lexical::query::wildcard::WildcardQuery;

#[derive(Parser)]
#[grammar = "lexical/query/parser.pest"]
struct QueryStringParser;

/// Query parser.
///
/// Similar to Lucene's QueryParser, this requires an Analyzer to properly
/// normalize query terms before matching against the index.
pub struct LexicalQueryParser {
    /// Analyzer for tokenizing and normalizing query terms.
    /// Required - following Lucene's design where Analyzer is mandatory.
    analyzer: Arc<dyn Analyzer>,
    /// Default fields to search when no field is specified.
    default_fields: Vec<String>,
    default_occur: Occur,
}

impl std::fmt::Debug for LexicalQueryParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryParser")
            .field("analyzer", &self.analyzer.name())
            .field("default_fields", &self.default_fields)
            .field("default_occur", &self.default_occur)
            .finish()
    }
}

impl LexicalQueryParser {
    /// Creates a new query parser with the given analyzer.
    ///
    /// Following Lucene's design, an Analyzer is required.
    ///
    /// # Arguments
    /// * `analyzer` - The analyzer to use for tokenizing and normalizing query terms
    ///
    /// # Example
    /// ```
    /// use laurus::analysis::analyzer::standard::StandardAnalyzer;
    /// use laurus::lexical::query::parser::LexicalQueryParser;
    /// use std::sync::Arc;
    ///
    /// let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
    /// let parser = LexicalQueryParser::new(analyzer);
    /// ```
    pub fn new(analyzer: Arc<dyn Analyzer>) -> Self {
        Self {
            analyzer,
            default_fields: Vec::new(),
            default_occur: Occur::Should,
        }
    }

    /// Create a query parser with the standard analyzer.
    ///
    /// This is a convenience method for the common case.
    pub fn with_standard_analyzer() -> Result<Self> {
        Ok(LexicalQueryParser::new(Arc::new(StandardAnalyzer::new()?)))
    }

    /// Sets the default field.
    ///
    /// This overwrites any existing default fields.
    pub fn with_default_field(mut self, field: impl Into<String>) -> Self {
        self.default_fields = vec![field.into()];
        self
    }

    /// Sets multiple default fields.
    pub fn with_default_fields(mut self, fields: Vec<String>) -> Self {
        self.default_fields = fields;
        self
    }

    /// Sets the default occur.
    pub fn with_default_occur(mut self, occur: Occur) -> Self {
        self.default_occur = occur;
        self
    }

    /// Get the default fields.
    pub fn default_fields(&self) -> &[String] {
        &self.default_fields
    }

    fn create_query_over_fields<F>(&self, field: Option<&str>, creator: F) -> Result<Box<dyn Query>>
    where
        F: Fn(&str) -> Result<Box<dyn Query>>,
    {
        if let Some(field_name) = field {
            return creator(field_name);
        }

        if self.default_fields.is_empty() {
            return Err(LaurusError::parse("No field specified".to_string()));
        }

        if self.default_fields.len() == 1 {
            return creator(&self.default_fields[0]);
        }

        let mut bool_query = BooleanQuery::new();
        for field_name in &self.default_fields {
            let q = creator(field_name)?;
            bool_query.add_clause(BooleanClause::new(q, Occur::Should));
        }
        Ok(Box::new(bool_query))
    }

    /// Parse a field-specific query.
    ///
    /// Constructs a query targeting a single field. If `query_str` contains spaces
    /// and is not already quoted, it is automatically wrapped in double quotes to
    /// form a phrase query.
    ///
    /// # Arguments
    ///
    /// * `field` - The name of the field to search.
    /// * `query_str` - The query string value to search for in the given field.
    ///
    /// # Returns
    ///
    /// A boxed [`Query`] targeting the specified field, or an error if parsing fails.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] if the constructed
    /// `field:query_str` expression cannot be parsed. This delegates to
    /// [`parse()`](Self::parse), so the same error conditions apply.
    pub fn parse_field(&self, field: &str, query_str: &str) -> Result<Box<dyn Query>> {
        // Handle phrase queries specially (preserve quotes).
        // Escape embedded double quotes to prevent query injection.
        let full_query = if query_str.contains(' ') && !query_str.starts_with('"') {
            let escaped = query_str.replace('"', "\\\"");
            format!("{field}:\"{escaped}\"")
        } else {
            format!("{field}:{query_str}")
        };
        self.parse(&full_query)
    }

    /// Parses a query string into a [`Query`] object.
    ///
    /// The query string follows a Lucene-like syntax supporting boolean operators,
    /// field-specific queries, phrase queries, wildcard queries, fuzzy queries, and
    /// range queries. When no field is specified, the parser's default fields are used.
    ///
    /// # Arguments
    ///
    /// * `query_str` - The query string to parse (e.g., `"title:rust AND body:search"`).
    ///
    /// # Returns
    ///
    /// A boxed [`Query`] representing the parsed query, or an error if the query
    /// string is malformed.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] in the following cases:
    /// - The query string has invalid syntax (e.g., unbalanced quotes or brackets).
    /// - The parsed input contains no valid query (e.g., an empty string).
    /// - A sub-query (boolean clause, range, etc.) fails to parse.
    /// - A phrase's slop (`"a b"~N`) or a fuzzy term's fuzziness (`a~N`) is
    ///   too large for a `u32` ([`LaurusError::InvalidArgument`], Issue
    ///   #1570).
    /// - A range bound is a number too large for an `f64`, such as `1e400`
    ///   ([`LaurusError::InvalidArgument`], Issue #1591).
    /// - A range mixes a numeric bound with a bound that is not a number,
    ///   such as `[5 TO abc]` or `[nan TO 5]` (Issue #1591).
    pub fn parse(&self, query_str: &str) -> Result<Box<dyn Query>> {
        let pairs = QueryStringParser::parse(Rule::query, query_str)
            .map_err(|e| LaurusError::parse(format!("Parse error: {e}")))?;

        for pair in pairs {
            if pair.as_rule() == Rule::query {
                for inner_pair in pair.into_inner() {
                    if inner_pair.as_rule() == Rule::boolean_query {
                        return self.parse_boolean_query(inner_pair);
                    }
                }
            }
        }

        Err(LaurusError::parse("No valid query found".to_string()))
    }

    fn parse_boolean_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<Box<dyn Query>> {
        let mut current_occur = self.default_occur;
        let mut terms: Vec<(Occur, Box<dyn Query>)> = Vec::new();

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::boolean_op => {
                    // The grammar takes the operators in capitals only
                    // (#1747), so the text is `AND` or `OR` as written.
                    let new_occur = match inner_pair.as_str() {
                        "AND" => Occur::Must,
                        _ => Occur::Should,
                    };
                    current_occur = new_occur;
                    // `AND` is symmetric: it makes both the preceding and the
                    // following clause required. The preceding clause has
                    // already been pushed under the default `Should` occur,
                    // so retroactively promote it to `Must` here. Skip the
                    // promotion if the previous clause was explicitly marked
                    // with `+` (already `Must`) or `-` (`MustNot`) so we
                    // don't override the user's intent.
                    if matches!(new_occur, Occur::Must)
                        && let Some(last) = terms.last_mut()
                        && matches!(last.0, Occur::Should)
                    {
                        last.0 = Occur::Must;
                    }
                }
                Rule::clause => {
                    let (occur, query) = self.parse_clause(inner_pair, current_occur)?;
                    terms.push((occur, query));
                    // Reset to default so that the operator only applies to
                    // the clauses on either side of it (e.g. `a AND b c` →
                    // `a Must`, `b Must`, `c Should`).
                    current_occur = self.default_occur;
                }
                _ => {}
            }
        }

        // If only one term, return it directly — unless it is prohibited
        // (`-term`): dropping the `Occur` here would silently turn the
        // negation into a positive query (#997). MustNot must stay
        // wrapped in a BooleanQuery so the matcher applies it against
        // the present-live universe.
        if terms.len() == 1 && !matches!(terms[0].0, Occur::MustNot) {
            return Ok(terms.into_iter().next().unwrap().1);
        }

        // Build boolean query
        let mut bool_query = BooleanQuery::new();
        for (occur, query) in terms {
            bool_query.add_clause(BooleanClause::new(query, occur));
        }

        Ok(Box::new(bool_query))
    }

    fn parse_clause(
        &self,
        pair: pest::iterators::Pair<Rule>,
        default_occur: Occur,
    ) -> Result<(Occur, Box<dyn Query>)> {
        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::required_clause => {
                    for sub_pair in inner_pair.into_inner() {
                        if sub_pair.as_rule() == Rule::sub_clause {
                            let query = self.parse_sub_clause(sub_pair)?;
                            return Ok((Occur::Must, query));
                        }
                    }
                }
                Rule::prohibited_clause => {
                    // `-clause` or `NOT clause` (#1747); the `not_op` pair
                    // before the sub-clause is skipped.
                    for sub_pair in inner_pair.into_inner() {
                        if sub_pair.as_rule() == Rule::sub_clause {
                            let query = self.parse_sub_clause(sub_pair)?;
                            return Ok((Occur::MustNot, query));
                        }
                    }
                }
                Rule::sub_clause => {
                    let query = self.parse_sub_clause(inner_pair)?;
                    return Ok((default_occur, query));
                }
                _ => {}
            }
        }

        Err(LaurusError::parse("Invalid clause".to_string()))
    }

    fn parse_sub_clause(&self, pair: pest::iterators::Pair<Rule>) -> Result<Box<dyn Query>> {
        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::grouped_query => return self.parse_grouped_query(inner_pair),
                Rule::field_query => return self.parse_field_query(inner_pair),
                Rule::term_query => return self.parse_term_query(inner_pair),
                _ => {}
            }
        }

        Err(LaurusError::parse("Invalid sub-clause".to_string()))
    }

    /// Parse a parenthesized group, `( ... )` with an optional `^boost`.
    ///
    /// The group's boost multiplies the boost of the query it holds,
    /// whatever the number of clauses or the nesting (Issue #1568), as
    /// Lucene's `BoostQuery` does: `(title:hello^2)^3` and
    /// `(title:hello^2 title:world)^3` both boost `hello` by 6, and
    /// `((title:hello^2)^3)^4` by 24. [`Self::parse_boolean_query`] returns
    /// a group of one clause unwrapped, so setting the group's boost would
    /// replace the clause's own. Multiplying gives the same score as a
    /// `BooleanQuery` around the clause would, since every query's score
    /// and bounds are proportional to its boost: the searcher runs a
    /// one-clause `BooleanQuery` by multiplying its boost into the clause
    /// the same way (Issue #1290). A boost of 1 changes nothing, so it
    /// needs no special case. A product too large for an `f32` is
    /// infinite, which [`Query::validate_boosts`] rejects.
    ///
    /// # Errors
    ///
    /// Returns an error if the group's contents do not parse.
    fn parse_grouped_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<Box<dyn Query>> {
        let mut boost = 1.0;
        let mut query: Option<Box<dyn Query>> = None;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::boolean_query => {
                    query = Some(self.parse_boolean_query(inner_pair)?);
                }
                Rule::boost => {
                    boost = self.parse_boost(inner_pair)?;
                }
                _ => {}
            }
        }

        if let Some(mut q) = query {
            q.set_boost(q.boost() * boost);
            Ok(q)
        } else {
            Err(LaurusError::parse("Invalid grouped query".to_string()))
        }
    }

    fn parse_field_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<Box<dyn Query>> {
        let mut field: Option<String> = None;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::field => {
                    field = Some(inner_pair.as_str().to_string());
                }
                Rule::field_value => {
                    let field_name = field
                        .ok_or_else(|| LaurusError::parse("Missing field name".to_string()))?;
                    return self.parse_field_value(inner_pair, Some(&field_name));
                }
                _ => {}
            }
        }

        Err(LaurusError::parse("Invalid field query".to_string()))
    }

    fn parse_term_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<Box<dyn Query>> {
        for inner_pair in pair.into_inner() {
            if inner_pair.as_rule() == Rule::field_value {
                return self.parse_field_value(inner_pair, None);
            }
        }

        Err(LaurusError::parse("Invalid term query".to_string()))
    }

    fn parse_field_value(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::geo3d_query => return self.parse_geo3d_query(inner_pair, field),
                Rule::geo_query => return self.parse_geo_query(inner_pair, field),
                Rule::range_query => return self.parse_range_query(inner_pair, field),
                Rule::phrase_query => return self.parse_phrase_query(inner_pair, field),
                Rule::fuzzy_term => return self.parse_fuzzy_term(inner_pair, field),
                Rule::wildcard_term => return self.parse_wildcard_term(inner_pair, field),
                Rule::simple_term => return self.parse_simple_term(inner_pair, field),
                _ => {}
            }
        }

        Err(LaurusError::parse("Invalid field value".to_string()))
    }

    /// Parse a 3D geo query (`geo3d_distance`, `geo3d_bbox`, or
    /// `geo3d_nearest`). All three require a target field; the DSL
    /// rejects bare invocations like `geo3d_distance(...)` without a
    /// `field:` prefix.
    fn parse_geo3d_query(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let field = field.ok_or_else(|| {
            LaurusError::parse(
                "3D geo queries require an explicit field prefix \
                 (e.g. `position:geo3d_distance(...)`)"
                    .to_string(),
            )
        })?;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::geo3d_distance => return self.parse_geo3d_distance(inner, field),
                Rule::geo3d_bbox => return self.parse_geo3d_bbox(inner, field),
                Rule::geo3d_nearest => return self.parse_geo3d_nearest(inner, field),
                _ => {}
            }
        }
        Err(LaurusError::parse(
            "Invalid 3D geo query (expected geo3d_distance / geo3d_bbox / geo3d_nearest)"
                .to_string(),
        ))
    }

    fn parse_geo3d_distance(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: &str,
    ) -> Result<Box<dyn Query>> {
        // Grammar guarantees four signed_float arguments: x, y, z, distance_m.
        let args = collect_signed_floats(pair)?;
        if args.len() != 4 {
            return Err(LaurusError::parse(format!(
                "geo3d_distance expects 4 numeric arguments (x, y, z, distance_m), got {}",
                args.len()
            )));
        }
        let center = GeoEcefPoint::new(args[0], args[1], args[2]);
        let distance_m = args[3];
        Ok(Box::new(Geo3dDistanceQuery::new(field, center, distance_m)))
    }

    fn parse_geo3d_bbox(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: &str,
    ) -> Result<Box<dyn Query>> {
        // Grammar: min_x, min_y, min_z, max_x, max_y, max_z.
        let args = collect_signed_floats(pair)?;
        if args.len() != 6 {
            return Err(LaurusError::parse(format!(
                "geo3d_bbox expects 6 numeric arguments \
                 (min_x, min_y, min_z, max_x, max_y, max_z), got {}",
                args.len()
            )));
        }
        let min = GeoEcefPoint::new(args[0], args[1], args[2]);
        let max = GeoEcefPoint::new(args[3], args[4], args[5]);
        Ok(Box::new(
            Geo3dBoundingBoxQuery::new(field, min, max)
                .map_err(|e| LaurusError::parse(format!("invalid geo3d_bbox arguments: {e}")))?,
        ))
    }

    fn parse_geo3d_nearest(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: &str,
    ) -> Result<Box<dyn Query>> {
        // Grammar: x (signed_float), y, z, k (unsigned_int).
        let mut floats: Vec<f64> = Vec::with_capacity(3);
        let mut k: Option<usize> = None;
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::signed_float => {
                    let s = inner.as_str();
                    let v = s.parse::<f64>().map_err(|e| {
                        LaurusError::parse(format!("geo3d_nearest: invalid float '{s}': {e}"))
                    })?;
                    floats.push(v);
                }
                Rule::unsigned_int => {
                    let s = inner.as_str();
                    let parsed = s.parse::<usize>().map_err(|e| {
                        LaurusError::parse(format!("geo3d_nearest: invalid integer '{s}': {e}"))
                    })?;
                    k = Some(parsed);
                }
                _ => {}
            }
        }
        if floats.len() != 3 {
            return Err(LaurusError::parse(format!(
                "geo3d_nearest expects 3 coordinate arguments before k, got {}",
                floats.len()
            )));
        }
        let k = k.ok_or_else(|| {
            LaurusError::parse("geo3d_nearest expects k as the 4th argument".to_string())
        })?;
        let center = GeoEcefPoint::new(floats[0], floats[1], floats[2]);
        Ok(Box::new(Geo3dNearestQuery::new(field, center, k)))
    }

    /// Parse a 2D geo query (`geo_distance` or `geo_bbox`). Both require a
    /// target field; the DSL rejects bare invocations like
    /// `geo_distance(...)` without a `field:` prefix.
    fn parse_geo_query(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let field = field.ok_or_else(|| {
            LaurusError::parse(
                "2D geo queries require an explicit field prefix \
                 (e.g. `location:geo_distance(...)`)"
                    .to_string(),
            )
        })?;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::geo_distance => return self.parse_geo_distance(inner, field),
                Rule::geo_bbox => return self.parse_geo_bbox(inner, field),
                _ => {}
            }
        }
        Err(LaurusError::parse(
            "Invalid 2D geo query (expected geo_distance / geo_bbox)".to_string(),
        ))
    }

    fn parse_geo_distance(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: &str,
    ) -> Result<Box<dyn Query>> {
        // Grammar: lat, lon, distance_m.
        let args = collect_signed_floats(pair)?;
        if args.len() != 3 {
            return Err(LaurusError::parse(format!(
                "geo_distance expects 3 numeric arguments (lat, lon, distance_m), got {}",
                args.len()
            )));
        }
        Ok(Box::new(
            GeoDistanceQuery::within_radius(field, args[0], args[1], args[2])
                .map_err(|e| LaurusError::parse(format!("invalid geo_distance arguments: {e}")))?,
        ))
    }

    fn parse_geo_bbox(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: &str,
    ) -> Result<Box<dyn Query>> {
        // Grammar: min_lat, min_lon, max_lat, max_lon.
        let args = collect_signed_floats(pair)?;
        if args.len() != 4 {
            return Err(LaurusError::parse(format!(
                "geo_bbox expects 4 numeric arguments (min_lat, min_lon, max_lat, max_lon), got {}",
                args.len()
            )));
        }
        Ok(Box::new(
            GeoBoundingBoxQuery::within_bounding_box(field, args[0], args[1], args[2], args[3])
                .map_err(|e| LaurusError::parse(format!("invalid geo_bbox arguments: {e}")))?,
        ))
    }

    /// Parse a range (`[a TO b]` inclusive, `{a TO b}` exclusive) on `field`,
    /// or on the default fields when `field` is `None`.
    ///
    /// The parser has no schema, so the range takes its type from its
    /// bounds, `*` leaving a side open:
    ///
    /// - a bound written in digits ([`parse_numeric_bound`]) builds a
    ///   [`NumericRangeQuery`];
    /// - otherwise a datetime literal builds a [`DateTimeRangeQuery`]
    ///   (#1179);
    /// - otherwise the bounds are strings and build a [`RangeQuery`], which
    ///   matches nothing.
    ///
    /// A range that mixes those kinds is an error rather than a silent
    /// empty or widened result.
    ///
    /// # Arguments
    ///
    /// * `pair` - The `range_query` pair.
    /// * `field` - The field the range names, if any.
    ///
    /// # Returns
    ///
    /// The range query, or a boolean query over the default fields.
    ///
    /// # Errors
    ///
    /// - [`LaurusError::InvalidArgument`] if a bound is a number that does
    ///   not fit an `f64`, such as `1e400` (Issue #1591).
    /// - [`LaurusError::Query`] if the range mixes a numeric bound with a
    ///   datetime bound or with a string (Issue #1591), or a datetime bound
    ///   with a string.
    fn parse_range_query(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let mut lower_inclusive = true;
        let mut upper_inclusive = true;
        let mut lower: Option<String> = None;
        let mut upper: Option<String> = None;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::range_inclusive => {
                    lower_inclusive = true;
                    upper_inclusive = true;
                    for range_part in inner_pair.into_inner() {
                        if range_part.as_rule() == Rule::range_value {
                            if lower.is_none() {
                                lower = Some(self.parse_range_value(range_part)?);
                            } else {
                                upper = Some(self.parse_range_value(range_part)?);
                            }
                        }
                    }
                }
                Rule::range_exclusive => {
                    lower_inclusive = false;
                    upper_inclusive = false;
                    for range_part in inner_pair.into_inner() {
                        if range_part.as_rule() == Rule::range_value {
                            if lower.is_none() {
                                lower = Some(self.parse_range_value(range_part)?);
                            } else {
                                upper = Some(self.parse_range_value(range_part)?);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Bound shape is inferred from the literal (the parser has no schema):
        // a number written in digits first (Issue #1591), so `[100 TO 500]`
        // and epoch-second bounds keep their meaning, then datetime literals
        // (#1179), then plain strings. A number too large for an `f64` is an
        // error rather than an infinite bound.
        let lower_num = lower
            .as_deref()
            .map(parse_numeric_bound)
            .transpose()?
            .flatten();
        let upper_num = upper
            .as_deref()
            .map(parse_numeric_bound)
            .transpose()?
            .flatten();
        let is_open = |bound: &Option<String>| bound.as_deref().is_none_or(|s| s == "*");
        let datetime_bound = |bound: &Option<String>| {
            bound
                .as_deref()
                .filter(|s| *s != "*")
                .and_then(parse_datetime_literal)
        };
        let lower_dt = datetime_bound(&lower);
        let upper_dt = datetime_bound(&upper);
        let describe = || {
            format!(
                "{}{} TO {}{}",
                if lower_inclusive { "[" } else { "{" },
                lower.as_deref().unwrap_or("*"),
                upper.as_deref().unwrap_or("*"),
                if upper_inclusive { "]" } else { "}" }
            )
        };

        self.create_query_over_fields(field, |field_name| {
            if lower_num.is_some() || upper_num.is_some() {
                if lower_dt.is_some() || upper_dt.is_some() {
                    return Err(crate::error::LaurusError::parse(format!(
                        "range {} on field '{field_name}' mixes a numeric bound with a \
                         datetime bound",
                        describe()
                    )));
                }
                // With no datetime bound here, a bound that is neither `*`
                // nor a number is a string. It used to become an open bound,
                // so `[5 TO abc]` searched `[5 TO *]` (Issue #1591), and
                // `[nan TO 5]` would search `[* TO 5]` now that `nan` is a
                // word.
                let is_string = |bound: &Option<String>, number: Option<f64>| {
                    !is_open(bound) && number.is_none()
                };
                if is_string(&lower, lower_num) || is_string(&upper, upper_num) {
                    return Err(crate::error::LaurusError::parse(format!(
                        "range {} on field '{field_name}' mixes a numeric bound with a \
                         bound that is not a number; a number is written in digits \
                         (nan, inf and infinity are words), and * leaves a side open",
                        describe()
                    )));
                }
                // Numeric range query
                let query = NumericRangeQuery::new(
                    field_name,
                    NumericType::Float,
                    lower_num,
                    upper_num,
                    lower_inclusive,
                    upper_inclusive,
                );
                Ok(Box::new(query))
            } else if lower_dt.is_some() || upper_dt.is_some() {
                // Datetime range (#1179): every non-open bound must be a
                // datetime literal — a silent "matches nothing" would be
                // worse than an error here.
                if (!is_open(&lower) && lower_dt.is_none())
                    || (!is_open(&upper) && upper_dt.is_none())
                {
                    return Err(crate::error::LaurusError::parse(format!(
                        "range {} on field '{field_name}' mixes a datetime bound with a \
                         bound that is not a datetime literal (expected RFC 3339, \
                         YYYY-MM-DDTHH:MM:SS[.fff], YYYY-MM-DD, or *)",
                        describe()
                    )));
                }
                Ok(Box::new(DateTimeRangeQuery::new(
                    field_name,
                    lower_dt,
                    upper_dt,
                    lower_inclusive,
                    upper_inclusive,
                )))
            } else {
                // String range. `RangeQuery`'s matcher is currently empty, so
                // this matches nothing (as the previous literal `TermQuery`
                // fallback did), but the parsed tree now reports the field
                // and bounds honestly.
                let to_bound = |bound: &Option<String>, inclusive: bool| match bound.as_deref() {
                    None | Some("*") => Bound::Unbounded,
                    Some(s) if inclusive => Bound::Included(s.to_string()),
                    Some(s) => Bound::Excluded(s.to_string()),
                };
                Ok(Box::new(RangeQuery::with_bounds(
                    field_name,
                    to_bound(&lower, lower_inclusive),
                    to_bound(&upper, upper_inclusive),
                )))
            }
        })
    }

    fn parse_range_value(&self, pair: pest::iterators::Pair<Rule>) -> Result<String> {
        let value = pair.as_str();
        if value == "*" {
            Ok("*".to_string())
        } else {
            Ok(value.trim_matches('"').to_string())
        }
    }

    /// Parse a quoted phrase (`"..."`, optionally `~slop`).
    ///
    /// Unlike a bare (unquoted) term whose analyzed tokens are OR'd (see
    /// `parse_simple_term`), a quoted phrase of two or more tokens builds a
    /// `PhraseQuery` requiring them to appear in sequence (within `slop`).
    /// Quoting is how a caller opts into strict adjacency.
    ///
    /// A quoted value that analyzes to one token has no adjacency to check,
    /// so it builds a `TermQuery`, as Lucene's `QueryBuilder` does. Quoting
    /// is also the only DSL form for a value containing `.`, `/` or `:`, such
    /// as a path or URL on a keyword field, and a `TermQuery` matches it
    /// without the positions a `term_vectors: false` field does not store
    /// (#1247).
    ///
    /// The analyzer's output is read as a token graph (#1252; see
    /// [`phrase_graph`]). Tokens stacked at one position (synonyms) are
    /// alternatives: one position becomes a single alternative's
    /// `TermQuery`, or several alternatives' [`SynonymQuery`], which scores
    /// them as one blended term instead of summing each alternative's
    /// independent score (Issue #1257) — the same relationship Lucene's
    /// `SynonymQuery` has to `TermQuery`. A longer phrase becomes a
    /// `PhraseQuery` with the alternatives at that position. A multi-word
    /// synonym makes several paths; the one-position paths are terms or
    /// synonyms as above, and the others are one `PhraseQuery` when there
    /// is one of them and a `GraphPhraseQuery` otherwise, which matches
    /// them all without listing them (#1271); the query is a `Should` of
    /// these, each carrying the boost.
    fn parse_phrase_query(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let mut phrase_content = String::new();
        let mut slop: Option<u32> = None;
        let mut boost = 1.0;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::phrase_content => {
                    phrase_content = inner_pair.as_str().to_string();
                }
                Rule::proximity => {
                    for prox_pair in inner_pair.into_inner() {
                        if prox_pair.as_rule() == Rule::number {
                            slop = Some(parse_count(prox_pair.as_str(), "phrase slop")?);
                        }
                    }
                }
                Rule::boost => {
                    boost = self.parse_boost(inner_pair)?;
                }
                _ => {}
            }
        }

        self.create_query_over_fields(field, |field_name| {
            let tokens = self.analyze_tokens(Some(field_name), &phrase_content)?;
            if tokens.is_empty() {
                // An empty phrase matches nothing and retains the field
                // reference for schema validation.
                return Ok(Box::new(
                    PhraseQuery::new(field_name, Vec::new()).with_boost(boost),
                ));
            }

            let slop = slop.unwrap_or(0);
            // The boost goes on each clause here.
            let one_position = |alternatives: Vec<String>, boost: f32| {
                one_position_query(field_name, alternatives, boost)
            };
            let phrase = |positions: Vec<Vec<String>>, boost: f32| -> Box<dyn Query> {
                Box::new(
                    PhraseQuery::from_positions(field_name, positions)
                        .with_slop(slop)
                        .with_boost(boost),
                )
            };
            let mut clauses: Vec<Box<dyn Query>> = match phrase_graph(&tokens) {
                // The tokens' boosts multiply the query's (Issue #1701), as
                // in Lucene's `QueryBuilder`: one position's boost, or the
                // product over a phrase's positions.
                PhraseGraph::Flat(mut positions) if positions.len() == 1 => {
                    let position = positions.remove(0);
                    vec![one_position(position.terms, boost * position.boost)]
                }
                PhraseGraph::Flat(positions) => {
                    let tokens_boost: f32 = positions.iter().map(|p| p.boost).product();
                    let positions = positions.into_iter().map(|p| p.terms).collect();
                    vec![phrase(positions, boost * tokens_boost)]
                }
                PhraseGraph::Graph { direct, arcs } => {
                    // `direct` and any phrase/graph-path clause below stay
                    // sibling `Should` clauses: a document matching both
                    // "nyc" and "new york city" genuinely contains two
                    // different phrase paths and both contribute, unlike
                    // the stacked-alternative case `one_position` folds
                    // into a single term.
                    let mut clauses: Vec<Box<dyn Query>> = match direct {
                        None => Vec::new(),
                        Some(direct) => {
                            vec![one_position(direct.terms, boost * direct.boost)]
                        }
                    };
                    if arcs.windows(2).all(|pair| pair[0].0.to == pair[1].0.from) {
                        // At most one path of two or more positions, as
                        // without multi-word synonyms: the phrase carries
                        // the product of its arcs' boosts.
                        if !arcs.is_empty() {
                            let tokens_boost: f32 = arcs.iter().map(|(_, b)| b).product();
                            clauses.push(phrase(
                                arcs.into_iter().map(|(arc, _)| arc.terms).collect(),
                                boost * tokens_boost,
                            ));
                        }
                    } else {
                        // Several paths of two or more positions, which
                        // `GraphPhraseQuery` matches without listing them,
                        // so there is no path to multiply the boosts of:
                        // it keeps the query's boost only. Lucene's
                        // multi-phrase path does not use them either.
                        clauses.push(Box::new(
                            GraphPhraseQuery::from_arcs(
                                field_name,
                                arcs.into_iter().map(|(arc, _)| arc).collect(),
                            )
                            .with_slop(slop)
                            .with_boost(boost),
                        ));
                    }
                    clauses
                }
            };

            if clauses.len() == 1 {
                return Ok(clauses.remove(0));
            }
            let mut bool_query = BooleanQuery::new();
            for clause in clauses {
                bool_query.add_should(clause);
            }
            Ok(Box::new(bool_query))
        })
    }

    fn parse_fuzzy_term(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let mut term = String::new();
        let mut fuzziness: u32 = 2; // Default fuzziness
        let mut boost = 1.0;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::term => {
                    term = inner_pair.as_str().to_string();
                }
                Rule::fuzziness => {
                    for fuzz_pair in inner_pair.into_inner() {
                        if fuzz_pair.as_rule() == Rule::number {
                            fuzziness = parse_count(fuzz_pair.as_str(), "fuzziness")?;
                        }
                    }
                }
                // The grammar always accepted `term~2^5`; the boost used to
                // be dropped here (Issue #1550).
                Rule::boost => {
                    boost = self.parse_boost(inner_pair)?;
                }
                _ => {}
            }
        }

        self.create_query_over_fields(field, |field_name| {
            // ✅ Normalize the term using the analyzer (like Lucene does)
            // This ensures the query term is in the same form as indexed terms
            let terms = self.analyze_term(Some(field_name), &term)?;
            let normalized_term = if terms.is_empty() {
                // Fallback to original term if analyzer produces no tokens
                &term
            } else {
                // Use the first token (following Lucene's behavior)
                &terms[0]
            };

            Ok(Box::new(
                FuzzyQuery::new(field_name, normalized_term)
                    .max_edits(fuzziness)
                    .with_boost(boost),
            ))
        })
    }

    fn parse_wildcard_term(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let mut pattern = String::new();
        let mut boost = 1.0;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::wildcard_pattern => {
                    pattern = inner_pair.as_str().to_string();
                }
                // The grammar always accepted `wild*^5`; the boost used to
                // be dropped here (Issue #1550).
                Rule::boost => {
                    boost = self.parse_boost(inner_pair)?;
                }
                _ => {}
            }
        }

        self.create_query_over_fields(field, |field_name| {
            Ok(Box::new(
                WildcardQuery::new(field_name, &pattern)?.with_boost(boost),
            ))
        })
    }

    fn parse_simple_term(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let mut term = String::new();
        let mut boost = 1.0;

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::term => {
                    term = inner_pair.as_str().to_string();
                }
                Rule::boost => {
                    boost = self.parse_boost(inner_pair)?;
                }
                _ => {}
            }
        }

        self.create_query_over_fields(field, |field_name| {
            let tokens = self.analyze_tokens(Some(field_name), &term)?;

            if tokens.is_empty() {
                // An empty phrase already matches nothing and retains the field
                // reference for schema validation, unlike an empty BooleanQuery.
                return Ok(Box::new(
                    PhraseQuery::new(field_name, Vec::new()).with_boost(boost),
                ));
            }

            // Grouped by position rather than flattened to text: a
            // position with several stacked alternatives (synonyms) must
            // be scored as one blended term instead of letting each
            // alternative add its own score (Issue #1257).
            let mut positions = group_tokens_by_position(tokens);

            if positions.len() == 1 {
                // The tokens' boost multiplies the query's (Issue #1701).
                let position = positions.remove(0);
                Ok(one_position_query(
                    field_name,
                    position.terms,
                    boost * position.boost,
                ))
            } else {
                // The analyzer split one *bare* (unquoted) term into
                // tokens at several positions. This happens for a
                // Japanese sentence under a morphological tokenizer, and
                // for inputs like `rust-lang` under a `\w+` tokenizer.
                //
                // Each position — a single token, or several stacked
                // alternatives scored as one term above — is OR'd with
                // the others, matching Lucene's `match` query and
                // Elasticsearch's default. The previous slop-0
                // `PhraseQuery` required the document to repeat the exact
                // morpheme sequence, which made unquoted Japanese queries
                // return almost nothing.
                //
                // Explicit phrase semantics are still available — a quoted
                // `"..."` goes through `parse_phrase_query`, which keeps
                // `PhraseQuery` (and its `~slop` modifier).
                let mut bool_query = BooleanQuery::new();
                for position in positions {
                    // The query's boost goes on the `BooleanQuery` below, as
                    // it did before this path grouped by position, so each
                    // clause carries only its tokens' boost (Issue #1701).
                    bool_query.add_clause(BooleanClause::new(
                        one_position_query(field_name, position.terms, position.boost),
                        Occur::Should,
                    ));
                }

                // `minimum_should_match` stays at its default of 0, so the
                // matcher builds a plain disjunction and BM25 scores add up
                // across the matching positions.
                if boost != 1.0 {
                    bool_query = bool_query.with_boost(boost);
                }
                Ok(Box::new(bool_query))
            }
        })
    }

    fn parse_boost(&self, pair: pest::iterators::Pair<Rule>) -> Result<f32> {
        for inner_pair in pair.into_inner() {
            if inner_pair.as_rule() == Rule::boost_value {
                return Ok(inner_pair.as_str().parse().unwrap_or(1.0));
            }
        }
        Ok(1.0)
    }

    fn analyze_term(&self, field: Option<&str>, term: &str) -> Result<Vec<String>> {
        Ok(self
            .analyze_tokens(field, term)?
            .into_iter()
            .map(|t| t.text)
            .collect())
    }

    fn analyze_tokens(&self, field: Option<&str>, term: &str) -> Result<Vec<Token>> {
        let token_stream = if let Some(field_name) = field {
            // Use field-specific analyzer if available (PerFieldAnalyzer)
            if let Some(per_field) = self.analyzer.as_any().downcast_ref::<PerFieldAnalyzer>() {
                per_field.analyze_field(field_name, term)?
            } else {
                self.analyzer.analyze(term)?
            }
        } else {
            self.analyzer.analyze(term)?
        };
        Ok(token_stream.collect())
    }
}

/// A term with no field in a query string, as the grammar reads it, and
/// whether a `-` or `NOT` applies to it ([`bare_terms`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BareTerm {
    /// The term as written.
    pub(crate) text: String,
    /// Whether a prohibited clause (`-` or `NOT`) holds the term, itself or
    /// a group around it.
    pub(crate) prohibited: bool,
}

/// The terms with no field in `query_str`, in order, as the grammar alone
/// reads them: nothing is analyzed and no query is built (Issue #1751).
///
/// Such a term is a clause of its own, or of a group: a `term_query` whose
/// value is a `simple_term`. The text of a phrase, a range or a field's
/// value is not a clause, and is not returned. The unified parser asks this
/// which of the vector clauses it finds in a query string are clauses, and
/// which of them a `-` or `NOT` applies to.
///
/// # Errors
///
/// Returns an error if the grammar cannot read `query_str`, as
/// [`LexicalQueryParser::parse`] does.
pub(crate) fn bare_terms(query_str: &str) -> Result<Vec<BareTerm>> {
    let pairs = QueryStringParser::parse(Rule::query, query_str)
        .map_err(|e| LaurusError::parse(format!("Parse error: {e}")))?;
    let mut terms = Vec::new();
    for pair in pairs {
        collect_bare_terms(pair, false, &mut terms);
    }
    Ok(terms)
}

/// Adds the bare terms under `pair` to `terms` ([`bare_terms`]).
/// `prohibited` is whether a prohibited clause holds `pair`.
fn collect_bare_terms(
    pair: pest::iterators::Pair<Rule>,
    prohibited: bool,
    terms: &mut Vec<BareTerm>,
) {
    match pair.as_rule() {
        Rule::prohibited_clause => {
            for inner in pair.into_inner() {
                collect_bare_terms(inner, true, terms);
            }
        }
        // `term_query` > `field_value` > `simple_term` > `term`.
        Rule::term_query => {
            let term = pair
                .into_inner()
                .next()
                .and_then(|value| value.into_inner().next())
                .filter(|value| value.as_rule() == Rule::simple_term)
                .and_then(|simple| simple.into_inner().find(|p| p.as_rule() == Rule::term));
            if let Some(term) = term {
                terms.push(BareTerm {
                    text: term.as_str().to_string(),
                    prohibited,
                });
            }
        }
        // A field's value holds no clause.
        Rule::field_query => {}
        _ => {
            for inner in pair.into_inner() {
                collect_bare_terms(inner, prohibited, terms);
            }
        }
    }
}

/// The graph a quoted value's tokens form ([`phrase_graph`]).
#[derive(Debug, PartialEq)]
enum PhraseGraph {
    /// A graph with no complete path, as the flat positions the index
    /// stores, each with its alternative terms and their boost.
    Flat(Vec<PositionTerms>),
    /// A graph with a complete path.
    Graph {
        /// The arc from the first node straight to the last, the paths of
        /// one position: its terms and their boost.
        direct: Option<PositionTerms>,
        /// The other arcs on a complete path, as
        /// [`complete_paths`](crate::lexical::query::graph_phrase::complete_paths)
        /// numbers them, each with the largest boost of its tokens.
        arcs: Vec<(PhraseArc, f32)>,
    },
}

/// The graph a quoted value's tokens form.
///
/// A token is an arc from its position (as the index numbers it) to its
/// position + `position_length` ([`token_arcs`]). Arcs with the
/// same start and end are one position of alternatives, such as a word and
/// its one-word synonyms. Every path from the first node to the last is
/// one phrase; a multi-word synonym adds one (`"ml"` → `[[ml]]` and
/// `[[machine], [learning]]`). Without multi-word synonyms there is a
/// single path. Arcs on no complete path are dropped.
///
/// A graph with no complete path (a malformed one, such as a hand-built
/// token stream can make) falls back to its flat positions
/// ([`flatten_token_graph`]), which is what the index stores.
fn phrase_graph(tokens: &[Token]) -> PhraseGraph {
    let token_arcs = token_arcs(tokens);

    // (from, to) → alternatives, and their largest boost (Issue #1701).
    let mut arcs: BTreeMap<(u32, u32), Vec<String>> = BTreeMap::new();
    let mut boosts: BTreeMap<(u32, u32), f32> = BTreeMap::new();
    for (token, &arc) in tokens.iter().zip(&token_arcs) {
        let boost = boosts.entry(arc).or_insert(token.boost);
        *boost = boost.max(token.boost);
        let alternatives = arcs.entry(arc).or_default();
        if !alternatives.contains(&token.text) {
            alternatives.push(token.text.clone());
        }
    }
    let first = token_arcs[0].0;
    let last = arcs.keys().map(|&(_, to)| to).max().unwrap_or(first);
    let arcs = arcs
        .into_iter()
        .map(|((from, to), terms)| PhraseArc::new(from, to, terms))
        .collect();
    let arcs = complete_paths_with_origins(arcs, first, last);

    if arcs.is_empty() {
        return PhraseGraph::Flat(group_tokens_by_position(tokens.to_vec()));
    }

    let last = arcs.iter().map(|(arc, _)| arc.to).max().unwrap_or(0);
    let mut direct = None;
    let mut others = Vec::with_capacity(arcs.len());
    for (arc, origin) in arcs {
        let boost = boosts[&origin];
        if arc.from == 0 && arc.to == last {
            // Each (from, to) is one arc, so there is at most one.
            direct = Some(PositionTerms {
                terms: arc.terms,
                boost,
            });
        } else {
            others.push((arc, boost));
        }
    }
    PhraseGraph::Graph {
        direct,
        arcs: others,
    }
}

/// The query for one field position's `alternatives`.
///
/// Several alternatives (synonyms the analyzer stacked there) become one
/// [`SynonymQuery`], which scores them as one blended term (Issue #1257)
/// rather than as an OR of independent `TermQuery`s: with index-time
/// synonyms a matching document holds every alternative, so summing their
/// scores would double-count it. A single alternative stays a plain
/// `TermQuery`. Shared by the quoted ([`LexicalQueryParser::parse_phrase_query`])
/// and bare ([`LexicalQueryParser::parse_simple_term`]) paths; each passes
/// the `boost` its own shape puts on a clause, with the position's tokens'
/// boost multiplied in ([`PositionTerms::boost`], Issue #1701).
fn one_position_query(
    field_name: &str,
    mut alternatives: Vec<String>,
    boost: f32,
) -> Box<dyn Query> {
    if alternatives.len() == 1 {
        Box::new(TermQuery::new(field_name, alternatives.remove(0)).with_boost(boost))
    } else {
        Box::new(SynonymQuery::new(field_name, alternatives).with_boost(boost))
    }
}

/// One field position of analyzed text: the alternative terms the analyzer
/// put there and the largest of their tokens' boosts.
///
/// The boost multiplies the query's when the position becomes a query
/// (Issue #1701), as Lucene's `QueryBuilder` applies a token's
/// `BoostAttribute`. Lucene gives each alternative of a `SynonymQuery` its
/// own boost; one factor for the position applies a uniform boost, such as
/// a `boost` token filter's, exactly, but not a synonym weight that differs
/// between the alternatives (`SynonymGraphFilter::with_boost`).
#[derive(Debug, PartialEq)]
struct PositionTerms {
    /// The alternative terms, deduplicated, in token order.
    terms: Vec<String>,
    /// The largest boost of the tokens at the position.
    boost: f32,
}

/// Group `tokens` by field position, each position paired with its
/// alternative terms — several when the analyzer stacks synonyms there
/// (`position_increment = 0`) — and their largest boost. Shared by
/// [`phrase_graph`]'s flat fallback and
/// [`LexicalQueryParser::parse_simple_term`], the two places that need a
/// bare term's tokens grouped this way without a full phrase graph.
fn group_tokens_by_position(tokens: Vec<Token>) -> Vec<PositionTerms> {
    let flat = flatten_token_graph(tokens);
    let positions = token_positions(&flat);
    let mut by_position: Vec<PositionTerms> = Vec::new();
    for (token, (i, &position)) in flat.iter().zip(positions.iter().enumerate()) {
        if i == 0 || positions[i - 1] != position {
            by_position.push(PositionTerms {
                terms: Vec::new(),
                boost: token.boost,
            });
        }
        let current = by_position.last_mut().expect("pushed above");
        current.boost = current.boost.max(token.boost);
        if !current.terms.contains(&token.text) {
            current.terms.push(token.text.clone());
        }
    }
    by_position
}

/// Walk a `geo3d_*` pair and collect every nested `signed_float` token
/// as an `f64`. Used by [`LexicalQueryParser::parse_geo3d_distance`]
/// and [`LexicalQueryParser::parse_geo3d_bbox`] which both take only
/// floats; `geo3d_nearest` mixes floats with an `unsigned_int` and
/// inlines its own walk.
fn collect_signed_floats(pair: pest::iterators::Pair<Rule>) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::signed_float {
            let s = inner.as_str();
            let v = s
                .parse::<f64>()
                .map_err(|e| LaurusError::parse(format!("invalid float '{s}': {e}")))?;
            out.push(v);
        }
    }
    Ok(out)
}

/// Parse a `number` token (digits only) as a `u32`: a phrase's slop or a
/// fuzzy term's fuzziness (Issue #1570). A number too large for a `u32` used
/// to become the default silently, `0` for a slop and `2` for a fuzziness;
/// it is rejected instead. The grammar admits nothing but digits, so that is
/// the only way the parse can fail.
///
/// # Arguments
///
/// * `text` - The token's text.
/// * `what` - What the number is, for the error message.
///
/// # Errors
///
/// Returns [`LaurusError::InvalidArgument`] if `text` does not fit a `u32`.
fn parse_count(text: &str, what: &str) -> Result<u32> {
    text.parse::<u32>().map_err(|_| {
        LaurusError::invalid_argument(format!("{what} must be at most {}, got {text}", u32::MAX))
    })
}

/// Read a range bound as a number (Issue #1591).
///
/// The parser has no schema, so a range takes its type from its bounds. A
/// bound is a number when it is written in digits: after one optional `+`
/// or `-`, it starts with an ASCII digit or `.` and parses as an `f64`
/// (`100`, `-1.5`, `+5`, `.5`, `5.`, `1e3`). Rust's `f64` parser also
/// accepts the words `nan`, `inf` and `infinity`, in any case and with a
/// sign. They are not numbers here, so they stay string bounds instead of
/// becoming a `NaN` bound, which matches nothing, or an infinite one.
///
/// A number too small to tell from `0`, such as `1e-400`, rounds to `0`,
/// as any `f64` literal rounds to the nearest `f64`.
///
/// # Arguments
///
/// * `text` - The bound, with its quotes removed.
///
/// # Returns
///
/// `Some(value)` for a number, and `None` for anything else: `*`, a
/// datetime literal or a string.
///
/// # Errors
///
/// Returns [`LaurusError::InvalidArgument`] if `text` is a number that does
/// not fit an `f64`, such as `1e400`, which `f64` parsing turns into an
/// infinity.
fn parse_numeric_bound(text: &str) -> Result<Option<f64>> {
    let unsigned = text.strip_prefix(['+', '-']).unwrap_or(text);
    if !unsigned.starts_with(|c: char| c.is_ascii_digit() || c == '.') {
        return Ok(None);
    }
    match text.parse::<f64>() {
        Ok(value) if value.is_finite() => Ok(Some(value)),
        Ok(_) => Err(LaurusError::invalid_argument(format!(
            "range bound {text} does not fit a 64-bit float"
        ))),
        Err(_) => Ok(None),
    }
}

/// Builder for constructing a [`LexicalQueryParser`] with a fluent API.
///
/// Allows configuring the analyzer, default search fields, and default boolean
/// occurrence before building the final [`LexicalQueryParser`] instance.
pub struct QueryParserBuilder {
    analyzer: Arc<dyn Analyzer>,
    default_fields: Vec<String>,
    default_occur: Occur,
}

impl QueryParserBuilder {
    /// Creates a new builder with the given analyzer.
    pub fn new(analyzer: Arc<dyn Analyzer>) -> Self {
        Self {
            analyzer,
            default_fields: Vec::new(),
            default_occur: Occur::Should,
        }
    }

    /// Sets the default field.
    pub fn default_field(mut self, field: impl Into<String>) -> Self {
        self.default_fields = vec![field.into()];
        self
    }

    /// Sets multiple default fields.
    pub fn default_fields(mut self, fields: Vec<String>) -> Self {
        self.default_fields = fields;
        self
    }

    /// Sets the default occur.
    pub fn default_occur(mut self, occur: Occur) -> Self {
        self.default_occur = occur;
        self
    }

    /// Builds the parser.
    pub fn build(self) -> Result<LexicalQueryParser> {
        Ok(LexicalQueryParser {
            analyzer: self.analyzer,
            default_fields: self.default_fields,
            default_occur: self.default_occur,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyzer::standard::StandardAnalyzer;

    /// Helper function to create a test parser with StandardAnalyzer
    fn create_test_parser() -> LexicalQueryParser {
        let analyzer = Arc::new(StandardAnalyzer::new().unwrap());
        LexicalQueryParser::new(analyzer)
    }

    // ---- Group boosts (#1568) ----

    /// A group's boost multiplies the boost of the query it holds, whatever
    /// the number of clauses or the nesting, instead of replacing the boost
    /// of a group of one clause (#1568).
    #[test]
    fn a_group_s_boost_multiplies_the_boosts_inside_it() {
        let parser = create_test_parser();
        let mut wrong = Vec::new();
        for (dsl, kind, boost) in [
            ("(title:hello^2)^3", "term", 6.0),
            ("(title:hello)^3", "term", 3.0),
            ("(title:hello^2)^1", "term", 2.0),
            ("((title:hello^2)^3)^4", "term", 24.0),
            ("((title:apple title:banana)^2)^3", "boolean", 6.0),
            ("(+title:hello^2)^3", "term", 6.0),
            // One prohibited clause stays in a `BooleanQuery` (#997).
            ("(-title:hello)^3", "boolean", 3.0),
        ] {
            let query = parser.parse(dsl).unwrap();
            let got = if query.as_any().is::<TermQuery>() {
                "term"
            } else if query.as_any().is::<BooleanQuery>() {
                "boolean"
            } else {
                "other"
            };
            if got != kind || query.boost() != boost {
                wrong.push(format!(
                    "{dsl}: {got} with boost {}, expected {kind} with boost {boost}",
                    query.boost()
                ));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    // ---- Range queries (#1179: datetime bounds) ----

    fn utc(y: i32, m: u32, d: u32, hh: u32, mm: u32, ss: u32) -> chrono::DateTime<chrono::Utc> {
        use chrono::TimeZone;
        chrono::Utc.with_ymd_and_hms(y, m, d, hh, mm, ss).unwrap()
    }

    fn parse_datetime_range(dsl: &str) -> DateTimeRangeQuery {
        let query = create_test_parser().parse(dsl).unwrap();
        query
            .as_any()
            .downcast_ref::<DateTimeRangeQuery>()
            .unwrap_or_else(|| panic!("{dsl} must parse to a DateTimeRangeQuery: {query:?}"))
            .clone()
    }

    #[test]
    fn range_date_only_bounds_build_datetime_range_query() {
        let q = parse_datetime_range("created_at:[2024-01-01 TO 2024-12-31]");
        assert_eq!(q.field(), "created_at");
        assert_eq!(q.lower_bound(), Some(utc(2024, 1, 1, 0, 0, 0)));
        assert_eq!(q.upper_bound(), Some(utc(2024, 12, 31, 0, 0, 0)));
        assert!(q.lower_inclusive());
        assert!(q.upper_inclusive());
    }

    #[test]
    fn range_rfc3339_offset_is_normalized_to_utc() {
        let q = parse_datetime_range(
            "created_at:[2024-01-01T09:00:00+09:00 TO 2024-01-02T00:00:00.5Z]",
        );
        assert_eq!(q.lower_bound(), Some(utc(2024, 1, 1, 0, 0, 0)));
        assert_eq!(
            q.upper_bound().unwrap().timestamp_subsec_millis(),
            500,
            "fractional seconds survive"
        );
        // Naive datetimes are UTC.
        let q = parse_datetime_range("created_at:[2024-06-15T12:34:56 TO *]");
        assert_eq!(q.lower_bound(), Some(utc(2024, 6, 15, 12, 34, 56)));
    }

    #[test]
    fn range_star_lower_with_date_upper() {
        let q = parse_datetime_range("created_at:[* TO 2024-12-31]");
        assert_eq!(q.lower_bound(), None);
        assert_eq!(q.upper_bound(), Some(utc(2024, 12, 31, 0, 0, 0)));
        let q = parse_datetime_range("created_at:[2024-12-31 TO *]");
        assert_eq!(q.lower_bound(), Some(utc(2024, 12, 31, 0, 0, 0)));
        assert_eq!(q.upper_bound(), None);
    }

    #[test]
    fn range_exclusive_braces_set_exclusive_flags() {
        let q = parse_datetime_range("created_at:{2024-01-01 TO 2024-12-31}");
        assert!(!q.lower_inclusive());
        assert!(!q.upper_inclusive());
    }

    #[test]
    fn range_mixed_numeric_and_date_is_an_error() {
        let err = create_test_parser()
            .parse("created_at:[100 TO 2024-12-31]")
            .unwrap_err();
        assert!(err.to_string().contains("numeric bound"), "{err}");
    }

    #[test]
    fn range_date_and_garbage_is_an_error() {
        let err = create_test_parser()
            .parse("created_at:[2024-01-01 TO yesterday]")
            .unwrap_err();
        assert!(err.to_string().contains("datetime literal"), "{err}");
    }

    #[test]
    fn range_numeric_bounds_still_build_numeric_range_query() {
        for dsl in [
            "price:[100 TO 500]",
            "price:[* TO 100]",
            "ts:[1700000000 TO *]",
        ] {
            let query = create_test_parser().parse(dsl).unwrap();
            assert!(
                query.as_any().downcast_ref::<NumericRangeQuery>().is_some(),
                "{dsl} must stay a NumericRangeQuery: {query:?}"
            );
        }
    }

    #[test]
    fn range_text_bounds_build_string_range_query() {
        let query = create_test_parser().parse("title:[A TO Z]").unwrap();
        let range = query
            .as_any()
            .downcast_ref::<RangeQuery>()
            .unwrap_or_else(|| panic!("expected a RangeQuery: {query:?}"));
        assert_eq!(range.field(), "title");
        assert_eq!(range.lower_bound(), &Bound::Included("A".to_string()));
        assert_eq!(range.upper_bound(), &Bound::Included("Z".to_string()));
        let query = create_test_parser().parse("title:{A TO *}").unwrap();
        let range = query.as_any().downcast_ref::<RangeQuery>().unwrap();
        assert_eq!(range.lower_bound(), &Bound::Excluded("A".to_string()));
        assert_eq!(range.upper_bound(), &Bound::Unbounded);
    }

    // ---- Range queries (#1591: numeric bounds) ----

    /// Issue #1591: `nan`, `inf` and `infinity` are words, not numbers, in
    /// any case, with a sign or quoted, so a range of them is a string range
    /// like `title:[A TO Z]`. Rust's `f64` parser read them as a `NaN` or an
    /// infinite numeric bound.
    #[test]
    fn range_words_nan_inf_infinity_are_string_bounds() {
        let text = |s: &str| s.to_string();
        let mut wrong = Vec::new();
        for (dsl, lower, upper) in [
            (
                "name:[nan TO inf]",
                Bound::Included(text("nan")),
                Bound::Included(text("inf")),
            ),
            (
                "name:[-Infinity TO zzz]",
                Bound::Included(text("-Infinity")),
                Bound::Included(text("zzz")),
            ),
            (
                "name:{NaN TO *}",
                Bound::Excluded(text("NaN")),
                Bound::Unbounded,
            ),
            (
                "name:[\"inf\" TO z]",
                Bound::Included(text("inf")),
                Bound::Included(text("z")),
            ),
        ] {
            match create_test_parser().parse(dsl) {
                Ok(query) => match query.as_any().downcast_ref::<RangeQuery>() {
                    Some(range)
                        if range.lower_bound() == &lower && range.upper_bound() == &upper => {}
                    _ => wrong.push(format!("{dsl}: {query:?}")),
                },
                Err(e) => wrong.push(format!("{dsl}: {e}")),
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1591: a numeric bound next to a bound that is not a number is
    /// a parse error, as a numeric or datetime bound next to a datetime
    /// bound or a string already is (#1179). The other bound used to become
    /// open, so `[5 TO abc]` searched `[5 TO *]`, and `[nan TO 5]` would
    /// search `[* TO 5]` now that `nan` is a word.
    #[test]
    fn range_number_next_to_a_string_bound_is_an_error() {
        let expected = "mixes a numeric bound with a bound that is not a number";
        let mut wrong = Vec::new();
        for dsl in [
            "price:[5 TO abc]",
            "price:[abc TO 5]",
            "price:[nan TO 5]",
            "price:[-inf TO 5]",
            "price:{0 TO Infinity}",
            "price:[1_000 TO 5]",
        ] {
            let result = create_test_parser().parse(dsl);
            if !matches!(&result, Err(LaurusError::Query(m)) if m.contains(expected)) {
                wrong.push(format!("{dsl}: {result:?}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1591: a number too large for an `f64` is rejected with
    /// `InvalidArgument`, as a boost too large for an `f32` (#1550) and a
    /// slop too large for a `u32` (#1570) are, instead of becoming an
    /// infinite bound.
    #[test]
    fn range_number_too_large_for_an_f64_is_invalid_argument() {
        let mut wrong = Vec::new();
        for (dsl, bound) in [
            ("price:[1e400 TO *]", "1e400"),
            ("price:[-1e309 TO 5]", "-1e309"),
            ("price:[* TO 1e999]", "1e999"),
        ] {
            let expected = format!("range bound {bound} does not fit a 64-bit float");
            let result = create_test_parser().parse(dsl);
            if !matches!(&result, Err(LaurusError::InvalidArgument(m)) if m.contains(&expected)) {
                wrong.push(format!("{dsl}: {result:?}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1591: every way of writing a number in digits stays a numeric
    /// bound with its value: a sign, a leading or trailing `.`, an exponent
    /// in either case, and a number too small to tell from `0`.
    #[test]
    fn range_numbers_in_every_digit_form_stay_numeric() {
        let mut wrong = Vec::new();
        for (dsl, min, max) in [
            ("price:[+5 TO .5e1]", 5.0, 5.0),
            ("price:[-1.5 TO 5.]", -1.5, 5.0),
            ("price:[-.5E1 TO 1e3]", -5.0, 1000.0),
            ("price:[1e-400 TO 1]", 0.0, 1.0),
        ] {
            match create_test_parser().parse(dsl) {
                Ok(query) => match query.as_any().downcast_ref::<NumericRangeQuery>() {
                    Some(range) if range.min_f64() == Some(min) && range.max_f64() == Some(max) => {
                    }
                    _ => wrong.push(format!("{dsl}: {query:?}")),
                },
                Err(e) => wrong.push(format!("{dsl}: {e}")),
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    #[test]
    fn test_simple_term() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("hello").unwrap();
        assert!(format!("{query:?}").contains("TermQuery"));
    }

    #[test]
    fn test_field_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("title:hello").unwrap();
        assert!(format!("{query:?}").contains("TermQuery"));
    }

    #[test]
    fn test_boolean_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("hello AND world").unwrap();
        assert!(format!("{query:?}").contains("BooleanQuery"));
    }

    /// Helper that downcasts the parsed query to a `BooleanQuery`
    /// and returns the per-clause occurs in source order.
    fn parsed_occurs(query_str: &str) -> Vec<Occur> {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse(query_str).unwrap();
        let bool_q = query
            .as_any()
            .downcast_ref::<crate::lexical::query::BooleanQuery>()
            .unwrap_or_else(|| panic!("expected BooleanQuery for '{query_str}', got {query:?}"));
        bool_q.clauses().iter().map(|c| c.occur).collect()
    }

    #[test]
    fn test_and_is_symmetric() {
        // `a AND b` should mark BOTH clauses as Must — not just the
        // right-hand side. Before the fix the left clause stayed at
        // the default `Should`, which made `(text) AND geo_bbox(...)`
        // match every doc inside the bbox regardless of the text.
        assert_eq!(
            parsed_occurs("hello AND world"),
            vec![Occur::Must, Occur::Must],
        );
    }

    #[test]
    fn test_and_chain_promotes_all_to_must() {
        // `a AND b AND c` → all three required.
        assert_eq!(
            parsed_occurs("hello AND world AND foo"),
            vec![Occur::Must, Occur::Must, Occur::Must],
        );
    }

    #[test]
    fn test_and_then_default_clause_keeps_default() {
        // `a AND b c`: the AND only governs the pair around it.
        // The trailing `c` (with no operator) keeps the default
        // (Should) — this matches the behaviour described in the
        // inline parser comment.
        assert_eq!(
            parsed_occurs("hello AND world foo"),
            vec![Occur::Must, Occur::Must, Occur::Should],
        );
    }

    #[test]
    fn test_or_does_not_change_should() {
        // `a OR b` is the default operator anyway: both Should.
        assert_eq!(
            parsed_occurs("hello OR world"),
            vec![Occur::Should, Occur::Should],
        );
    }

    #[test]
    fn test_explicit_required_clause_is_preserved_through_and() {
        // `+a AND b`: the explicit `+` already marked `a` as Must,
        // so the AND promotion is a no-op for it. `b` becomes Must
        // through the normal AND path.
        assert_eq!(
            parsed_occurs("+hello AND world"),
            vec![Occur::Must, Occur::Must],
        );
    }

    #[test]
    fn test_explicit_prohibited_clause_survives_and() {
        // `-a AND b`: do not silently turn the prohibited clause
        // into a required one. `a` stays MustNot, `b` is promoted
        // to Must by the AND.
        assert_eq!(
            parsed_occurs("-hello AND world"),
            vec![Occur::MustNot, Occur::Must],
        );
    }

    #[test]
    fn test_and_promotes_only_immediate_predecessor() {
        // `a b AND c`: only the clause directly to the left of
        // the AND is promoted to Must. The earlier `a` stays at
        // the default `Should`.
        assert_eq!(
            parsed_occurs("hello world AND foo"),
            vec![Occur::Should, Occur::Must, Occur::Must],
        );
    }

    // ---- NOT, and the operators in capitals as whole words (#1747) ----

    /// The clauses of `query_str` parsed on the field `content`, in source
    /// order: each one's occur, and its term when it is a `TermQuery`. A
    /// query that is not a `BooleanQuery` is one `Should` clause.
    fn parsed_clauses(query_str: &str) -> Vec<(Occur, Option<String>)> {
        let parser = create_test_parser().with_default_field("content");
        let query = parser
            .parse(query_str)
            .unwrap_or_else(|e| panic!("'{query_str}' must parse: {e}"));
        let term = |query: &dyn Query| {
            query
                .as_any()
                .downcast_ref::<TermQuery>()
                .map(|term| term.term().to_string())
        };
        match query.as_any().downcast_ref::<BooleanQuery>() {
            Some(bool_q) => bool_q
                .clauses()
                .iter()
                .map(|clause| (clause.occur, term(clause.query.as_ref())))
                .collect(),
            None => vec![(Occur::Should, term(query.as_ref()))],
        }
    }

    /// A query string, and the clauses expected from it: each one's occur,
    /// and its term when it is a `TermQuery`.
    type ClauseCase<'a> = (&'a str, Vec<(Occur, Option<&'a str>)>);

    /// The cases whose clauses are not the expected ones, each with what it
    /// parsed to.
    fn wrong_clauses(cases: &[ClauseCase]) -> Vec<String> {
        cases
            .iter()
            .filter_map(|(query_str, expected)| {
                let clauses = parsed_clauses(query_str);
                let expected: Vec<(Occur, Option<String>)> = expected
                    .iter()
                    .map(|(occur, term)| (*occur, term.map(str::to_string)))
                    .collect();
                (clauses != expected)
                    .then(|| format!("'{query_str}': {clauses:?}, expected {expected:?}"))
            })
            .collect()
    }

    /// Issue #1747: `NOT` prohibits the clause after it, as `-` does: alone,
    /// after a clause, after `AND` or `OR`, and before a group or a phrase.
    #[test]
    fn not_prohibits_the_clause_after_it() {
        use Occur::{Must, MustNot, Should};
        let wrong = wrong_clauses(&[
            ("NOT hello", vec![(MustNot, Some("hello"))]),
            (
                "hello NOT world",
                vec![(Should, Some("hello")), (MustNot, Some("world"))],
            ),
            (
                "hello AND NOT world",
                vec![(Must, Some("hello")), (MustNot, Some("world"))],
            ),
            (
                "hello OR NOT world",
                vec![(Should, Some("hello")), (MustNot, Some("world"))],
            ),
            (
                "NOT hello AND world",
                vec![(MustNot, Some("hello")), (Must, Some("world"))],
            ),
            ("NOT(hello world)", vec![(MustNot, None)]),
            ("NOT \"hello world\"", vec![(MustNot, None)]),
        ]);
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1747: `AND`, `OR` and `NOT` are operators only in capitals;
    /// in another case they are terms. `and`, `or` and `not` are stop words
    /// of `StandardAnalyzer`, so each becomes a query that matches nothing,
    /// but it stays a clause of its own.
    #[test]
    fn an_operator_in_another_case_is_a_term() {
        use Occur::Should;
        let cases: Vec<ClauseCase> = [
            "hello and world",
            "hello And world",
            "hello or world",
            "hello oR world",
            "hello not world",
            "hello Not world",
        ]
        .into_iter()
        .map(|query_str| {
            (
                query_str,
                vec![
                    (Should, Some("hello")),
                    (Should, None),
                    (Should, Some("world")),
                ],
            )
        })
        .collect();
        let wrong = wrong_clauses(&cases);
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1747: an operator is a whole word: a word that starts with
    /// `AND`, `OR` or `NOT`, in any case, is a term, also after a clause.
    #[test]
    fn a_word_that_starts_with_an_operator_is_a_term() {
        use Occur::Should;
        let cases: Vec<ClauseCase> = [
            ("hello ANDROID", "android"),
            ("hello android", "android"),
            ("hello ORACLE", "oracle"),
            ("hello orange", "orange"),
            ("hello NOTE", "note"),
            ("hello NOTHING", "nothing"),
        ]
        .into_iter()
        .map(|(query_str, word)| {
            (
                query_str,
                vec![(Should, Some("hello")), (Should, Some(word))],
            )
        })
        .collect();
        let wrong = wrong_clauses(&cases);
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1747: an operator with no clause after it is a term, as a
    /// trailing `AND` or `OR` already was (a stop word here).
    #[test]
    fn an_operator_with_nothing_after_it_is_a_term() {
        use Occur::Should;
        let cases: Vec<ClauseCase> = ["hello NOT", "hello AND", "hello OR"]
            .into_iter()
            .map(|query_str| (query_str, vec![(Should, Some("hello")), (Should, None)]))
            .collect();
        let wrong = wrong_clauses(&cases);
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1751: `bare_terms` reports the term `x` as prohibited when a
    /// `-` or `NOT` applies to it or to a group around it, and does not
    /// report it when it is the text of a phrase, a range or a field's value.
    #[test]
    fn bare_terms_report_whether_a_term_is_prohibited() {
        let prohibited = |query_str: &str| {
            bare_terms(query_str)
                .unwrap_or_else(|e| panic!("'{query_str}' must parse: {e}"))
                .into_iter()
                .find(|term| term.text == "x")
                .map(|term| term.prohibited)
        };
        let cases = [
            ("-x", Some(true)),
            ("NOT x", Some(true)),
            ("- x", Some(true)),
            ("-(a x)", Some(true)),
            ("NOT (a x)", Some(true)),
            ("(a -x)", Some(true)),
            ("-(-x)", Some(true)),
            ("-(+x)", Some(true)),
            ("x -a", Some(false)),
            ("+x", Some(false)),
            ("a AND x", Some(false)),
            ("(a x)", Some(false)),
            ("\"a x\"", None),
            ("title:[x TO y]", None),
            ("title:x", None),
        ];
        let wrong: Vec<String> = cases
            .iter()
            .filter(|(query_str, expected)| prohibited(query_str) != *expected)
            .map(|(query_str, expected)| {
                format!(
                    "'{query_str}': {:?}, expected {expected:?}",
                    prohibited(query_str)
                )
            })
            .collect();
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    #[test]
    fn test_phrase_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("\"hello world\"").unwrap();
        assert!(format!("{query:?}").contains("PhraseQuery"));
    }

    #[test]
    fn test_fuzzy_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("hello~2").unwrap();
        assert!(format!("{query:?}").contains("FuzzyQuery"));
    }

    #[test]
    fn test_wildcard_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("hel*").unwrap();
        assert!(format!("{query:?}").contains("WildcardQuery"));
    }

    #[test]
    fn test_required_clause() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("+hello world").unwrap();
        assert!(format!("{query:?}").contains("BooleanQuery"));
    }

    #[test]
    fn test_prohibited_clause() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("hello -world").unwrap();
        assert!(format!("{query:?}").contains("BooleanQuery"));
    }

    /// #997 regression: a lone prohibited clause must keep its negation.
    /// The single-term unwrap used to discard the `Occur`, silently
    /// turning `-term` into a positive query.
    #[test]
    fn test_prohibited_only_query_keeps_negation() {
        let parser = create_test_parser().with_default_field("content");

        let query = parser.parse("-content:spam").unwrap();
        let debug = format!("{query:?}");
        assert!(
            debug.contains("BooleanQuery"),
            "a lone prohibited clause must stay wrapped in a BooleanQuery: {debug}"
        );
        assert!(
            debug.contains("MustNot"),
            "the MustNot occur must survive parsing: {debug}"
        );

        // Same for the default-field form.
        let query = parser.parse("-spam").unwrap();
        let debug = format!("{query:?}");
        assert!(debug.contains("MustNot"), "default-field form: {debug}");
    }

    #[test]
    fn test_grouped_query() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("(hello OR world) AND test").unwrap();
        assert!(format!("{query:?}").contains("BooleanQuery"));
    }

    #[test]
    fn test_proximity_search() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("\"hello world\"~10").unwrap();
        assert!(format!("{query:?}").contains("PhraseQuery"));
    }

    #[test]
    fn test_unquoted_cjk_term() {
        // Bare CJK / Cyrillic / Greek terms must parse without quoting.
        // Previously the grammar restricted unescaped terms to
        // ASCII_ALPHANUMERIC, so `日本語` would surface a pest "expected
        // clause" error to the user.
        let parser = create_test_parser().with_default_field("content");

        for term in ["日本語", "形態素", "Привет", "Ελληνικά", "café"] {
            let query = parser
                .parse(term)
                .unwrap_or_else(|e| panic!("failed to parse {term:?}: {e}"));
            assert!(
                format!("{query:?}").contains("TermQuery"),
                "expected TermQuery for {term:?}, got {query:?}"
            );
        }
    }

    #[test]
    fn test_unquoted_cjk_field_query() {
        // CJK term following an ASCII field name and colon.
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("title:形態素").unwrap();
        assert!(format!("{query:?}").contains("TermQuery"));
    }

    #[test]
    fn test_apostrophe_in_unquoted_term() {
        // Issue #1111: `'` has no syntactic meaning in the DSL, but was
        // previously absent from both `unescaped_term` and `special_char`,
        // so it could not appear in a bare term at all -- not even
        // backslash-escaped. Contractions (`it's`, `don't`) and
        // possessives (`user's`) are extremely common in ordinary English
        // text, so this must parse without requiring the whole query to
        // be wrapped in quotes (which would silently change unquoted
        // OR-of-tokens semantics into an exact phrase match).
        let parser = create_test_parser().with_default_field("content");

        for q in ["It's Only the Himalayas", "don't", "user's book", "isn't"] {
            parser
                .parse(q)
                .unwrap_or_else(|e| panic!("failed to parse {q:?}: {e}"));
        }
    }

    #[test]
    fn test_apostrophe_grammar_consumes_the_whole_word() {
        // The `term` rule must consume the ENTIRE word including the
        // apostrophe, not stop short at it and strand the rest to be
        // re-parsed as a spurious second clause. This is the
        // `normal_char` half of the #1111 fix: `term`'s `escaped_char+`
        // alternative is tried before `unescaped_term`, and `escaped_char`
        // bottoms out at `normal_char` for any non-backslashed character
        // -- so `'` must be in `normal_char` too, or `escaped_char+`
        // would stop consuming right at the apostrophe.
        //
        // Verified directly against the grammar (not through `parse()`)
        // because the effect of the bug isn't a parse error: without this
        // fix, `don't` silently parses as two adjacent bare terms (`don`
        // and `'t`), which `analyze_term` OR's together into a
        // `BooleanQuery` just like a legitimate two-word query would --
        // there is no error to assert against, only the wrong split.
        let pairs = QueryStringParser::parse(Rule::term, "don't")
            .unwrap_or_else(|e| panic!("failed to parse term \"don't\": {e}"));
        let matched = pairs.as_str();
        assert_eq!(
            matched, "don't",
            "the term rule must consume the whole word in one match, got {matched:?}"
        );
    }

    #[test]
    fn test_unquoted_cjk_boolean() {
        // Whitespace-bounded boolean ops still kick in between CJK terms.
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("形態素 AND 解析").unwrap();
        assert!(format!("{query:?}").contains("BooleanQuery"));
    }

    #[test]
    fn unquoted_multi_token_term_builds_a_should_boolean_not_a_phrase() {
        // StandardAnalyzer's `\w+` tokenizer splits "rust-lang" into two
        // tokens. The previous implementation built a slop-0 PhraseQuery
        // here, requiring exact adjacency; it now OR's the tokens like
        // Lucene's `match` query.
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("rust-lang").unwrap();
        let bool_q = query
            .as_any()
            .downcast_ref::<crate::lexical::query::BooleanQuery>()
            .unwrap_or_else(|| panic!("expected BooleanQuery, got {query:?}"));
        let clauses = bool_q.clauses();
        assert_eq!(clauses.len(), 2, "expected 2 clauses, got {clauses:?}");
        assert!(
            clauses.iter().all(|c| c.occur == Occur::Should),
            "every clause must be Should (OR), got {clauses:?}"
        );
        assert!(
            !format!("{query:?}").contains("PhraseQuery"),
            "must not build a PhraseQuery for an unquoted multi-token term"
        );
    }

    #[test]
    fn unquoted_multi_token_term_keeps_its_boost() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("rust-lang^2").unwrap();
        assert_eq!(query.boost(), 2.0);
    }

    #[test]
    fn quoted_phrase_still_builds_a_phrase_query() {
        // Quoting is how a caller opts into strict phrase (adjacency)
        // semantics; this must be unaffected by the bare-term OR change.
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("\"hello world\"").unwrap();
        let phrase_q = query
            .as_any()
            .downcast_ref::<PhraseQuery>()
            .unwrap_or_else(|| panic!("expected PhraseQuery, got {query:?}"));
        assert_eq!(phrase_q.slop(), 0);
    }

    #[test]
    fn quoted_phrase_keeps_its_slop() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("\"hello world\"~5").unwrap();
        let phrase_q = query
            .as_any()
            .downcast_ref::<PhraseQuery>()
            .unwrap_or_else(|| panic!("expected PhraseQuery, got {query:?}"));
        assert_eq!(phrase_q.slop(), 5);
    }

    fn expect_term_query(query: &dyn Query) -> &TermQuery {
        query
            .as_any()
            .downcast_ref::<TermQuery>()
            .unwrap_or_else(|| panic!("expected TermQuery, got {query:?}"))
    }

    /// #1247: a quoted value that analyzes to one token is a term match. A
    /// one-term `PhraseQuery` needs positions, which a field with
    /// `term_vectors: false` does not store.
    #[test]
    fn quoted_single_token_builds_a_term_query() {
        let parser = create_test_parser().with_default_field("content");

        let query = parser.parse("title:\"Hello\"").unwrap();
        let term_q = expect_term_query(query.as_ref());
        assert_eq!(term_q.field(), "title");
        assert_eq!(term_q.term(), "hello");

        // The default-field form goes through the same path.
        let query = parser.parse("\"hello\"").unwrap();
        assert_eq!(expect_term_query(query.as_ref()).field(), "content");
    }

    /// #1247: quoting is the only DSL form for a value containing `.`, `/`
    /// or `:`, so a keyword field must get the whole value as one term.
    #[test]
    fn quoted_keyword_value_builds_a_term_query_for_the_whole_value() {
        use crate::analysis::analyzer::keyword::KeywordAnalyzer;

        let per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
        per_field.add_analyzer("path", Arc::new(KeywordAnalyzer::new()));
        let parser = LexicalQueryParser::new(Arc::new(per_field));

        let query = parser.parse("path:\"file:///m/a.md\"").unwrap();
        let term_q = expect_term_query(query.as_ref());
        assert_eq!(term_q.field(), "path");
        assert_eq!(term_q.term(), "file:///m/a.md");
    }

    #[test]
    fn quoted_single_token_keeps_its_boost() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("title:\"hello\"^2").unwrap();
        assert_eq!(expect_term_query(query.as_ref()).boost(), 2.0);
    }

    /// Slop is a distance between terms, so it has nothing to apply to.
    #[test]
    fn quoted_single_token_ignores_its_slop() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("title:\"hello\"~3").unwrap();
        assert_eq!(expect_term_query(query.as_ref()).term(), "hello");
    }

    /// Issue #1570: a slop too large for a `u32` used to become `0`, an
    /// exact phrase, without an error. It is rejected, also on a quoted
    /// single token, which ignores its slop; the largest `u32` still parses.
    #[test]
    fn a_slop_too_large_for_a_u32_is_rejected() {
        let parser = create_test_parser().with_default_field("content");
        let query = parser.parse("\"hello world\"~4294967295").unwrap();
        let phrase_q = query
            .as_any()
            .downcast_ref::<PhraseQuery>()
            .unwrap_or_else(|| panic!("expected PhraseQuery, got {query:?}"));
        assert_eq!(phrase_q.slop(), u32::MAX);

        let mut wrong = Vec::new();
        for (dsl, got) in [
            ("\"hello world\"~4294967296", "4294967296"),
            ("\"hello world\"~99999999999", "99999999999"),
            ("title:\"hello\"~4294967296", "4294967296"),
        ] {
            let expected = format!("phrase slop must be at most 4294967295, got {got}");
            let result = parser.parse(dsl);
            if !matches!(&result, Err(LaurusError::InvalidArgument(m)) if m.contains(&expected)) {
                wrong.push(format!("{dsl}: {result:?}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1570: a fuzziness parses as a `u32`, the type of
    /// `FuzzyQuery::max_edits`, rather than as a `u8` whose overflow (256
    /// and up) became the default `2` without an error. A number too large
    /// for a `u32` is rejected.
    #[test]
    fn a_fuzziness_parses_as_a_u32_and_rejects_a_larger_one() {
        let parser = create_test_parser().with_default_field("content");
        let mut wrong = Vec::new();
        for (dsl, expected) in [
            ("hello~", 2),
            ("hello~0", 0),
            ("hello~256", 256),
            ("hello~4294967295", u32::MAX),
        ] {
            let got = parser.parse(dsl).ok().and_then(|query| {
                query
                    .as_any()
                    .downcast_ref::<FuzzyQuery>()
                    .map(FuzzyQuery::get_max_edits)
            });
            if got != Some(expected) {
                wrong.push(format!("{dsl}: {got:?}, expected {expected}"));
            }
        }
        let result = parser.parse("hello~4294967296");
        if !matches!(&result, Err(LaurusError::InvalidArgument(m))
            if m.contains("fuzziness must be at most 4294967295, got 4294967296"))
        {
            wrong.push(format!("hello~4294967296: {result:?}"));
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// A Japanese sentence tokenized by Lindera into several morphemes
    /// must be OR'd when unquoted — the direct regression for the
    /// "unquoted Japanese queries return almost nothing" bug.
    #[test]
    fn unquoted_japanese_sentence_ors_its_morphemes() {
        use crate::analysis::analyzer::language::japanese::JapaneseAnalyzer;

        let analyzer =
            Arc::new(JapaneseAnalyzer::new("normal", "embedded://ipadic", None).unwrap());
        let parser = LexicalQueryParser::new(analyzer).with_default_field("content");

        // JapaneseAnalyzer's built-in stop-word filter drops the
        // particles ("は", "で", "ある"), leaving 2 content morphemes
        // ("吾輩", "猫") — still more than one, which is the point.
        let query = parser.parse("吾輩は猫である").unwrap();
        let bool_q = query
            .as_any()
            .downcast_ref::<crate::lexical::query::BooleanQuery>()
            .unwrap_or_else(|| panic!("expected BooleanQuery, got {query:?}"));
        assert!(
            bool_q.clauses().len() >= 2,
            "expected multiple morphemes OR'd, got {:?}",
            bool_q.clauses()
        );
        assert!(
            bool_q.clauses().iter().all(|c| c.occur == Occur::Should),
            "every morpheme clause must be Should (OR)"
        );
    }

    #[test]
    fn unquoted_japanese_sentence_in_a_field_query_ors_too() {
        use crate::analysis::analyzer::language::japanese::JapaneseAnalyzer;

        let analyzer =
            Arc::new(JapaneseAnalyzer::new("normal", "embedded://ipadic", None).unwrap());
        let parser = LexicalQueryParser::new(analyzer).with_default_field("content");

        let query = parser.parse("body:吾輩は猫である").unwrap();
        assert!(
            query
                .as_any()
                .downcast_ref::<crate::lexical::query::BooleanQuery>()
                .is_some(),
            "expected BooleanQuery, got {query:?}"
        );
    }

    /// A quoted Japanese phrase must still build a PhraseQuery — the
    /// morphemes are ordered tokens for `PhraseQuery`, not OR'd terms.
    #[test]
    fn quoted_japanese_phrase_still_builds_a_phrase_query() {
        use crate::analysis::analyzer::language::japanese::JapaneseAnalyzer;

        let analyzer =
            Arc::new(JapaneseAnalyzer::new("normal", "embedded://ipadic", None).unwrap());
        let parser = LexicalQueryParser::new(analyzer).with_default_field("content");

        let query = parser.parse("\"吾輩は猫である\"").unwrap();
        let phrase_q = query
            .as_any()
            .downcast_ref::<PhraseQuery>()
            .unwrap_or_else(|| panic!("expected PhraseQuery, got {query:?}"));
        assert!(
            phrase_q.positions().len() >= 2,
            "expected multiple morphemes, got {:?}",
            phrase_q.positions()
        );
    }

    #[test]
    fn test_multiple_default_fields() {
        let parser =
            create_test_parser().with_default_fields(vec!["title".to_string(), "body".to_string()]);
        let query = parser.parse("hello").unwrap();
        let query_debug = format!("{:?}", query);
        // Should be a BooleanQuery combining matches on title and body
        assert!(query_debug.contains("BooleanQuery"));

        // Unfortunately standard debug format might be opaque, but we can assume BooleanQuery is created if multiple fields.
        // If it was single field, it would be TermQuery.
    }

    #[test]
    fn test_geo3d_distance_basic() {
        let parser = create_test_parser().with_default_field("content");
        let q = parser
            .parse("position:geo3d_distance(1000, 2000, 3000, 500)")
            .unwrap();
        let dbg = format!("{q:?}");
        assert!(dbg.contains("Geo3dDistanceQuery"), "got: {dbg}");
        assert!(dbg.contains("position"), "got: {dbg}");
    }

    #[test]
    fn test_geo3d_distance_signed_and_scientific() {
        // Negative coordinates and scientific notation must round-trip
        // through the grammar.
        let parser = create_test_parser().with_default_field("content");
        let q = parser
            .parse("p:geo3d_distance(-1.5e6, 2.5e6, -3.0e6, 1e3)")
            .unwrap();
        assert!(format!("{q:?}").contains("Geo3dDistanceQuery"));
    }

    #[test]
    fn test_geo3d_bbox_basic() {
        let parser = create_test_parser().with_default_field("content");
        let q = parser
            .parse("position:geo3d_bbox(0, 0, 0, 100, 200, 300)")
            .unwrap();
        let dbg = format!("{q:?}");
        assert!(dbg.contains("Geo3dBoundingBoxQuery"), "got: {dbg}");
    }

    #[test]
    fn test_geo3d_bbox_inverted_rejected() {
        // The pest grammar accepts the syntactic form but the
        // GeoBoundingBoxQuery constructor rejects min > max with a
        // clear error.
        let parser = create_test_parser().with_default_field("content");
        let err = parser
            .parse("position:geo3d_bbox(100, 0, 0, 0, 200, 300)")
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("invalid geo3d_bbox arguments"), "got: {msg}");
    }

    #[test]
    fn test_geo3d_nearest_basic() {
        let parser = create_test_parser().with_default_field("content");
        let q = parser
            .parse("position:geo3d_nearest(1000.5, 2000.5, 3000.5, 5)")
            .unwrap();
        let dbg = format!("{q:?}");
        assert!(dbg.contains("Geo3dNearestQuery"), "got: {dbg}");
    }

    #[test]
    fn test_geo3d_query_combined_with_boolean() {
        // 3D geo queries can mix with the rest of the DSL: this
        // confirms the boolean composition still parses.
        let parser = create_test_parser().with_default_field("content");
        let q = parser
            .parse("city:Tokyo AND position:geo3d_distance(0, 0, 0, 1000)")
            .unwrap();
        let dbg = format!("{q:?}");
        assert!(dbg.contains("BooleanQuery"), "got: {dbg}");
        assert!(dbg.contains("Geo3dDistanceQuery"), "got: {dbg}");
    }

    #[test]
    fn test_geo3d_without_field_rejected() {
        // Parsing geo3d_* without a field prefix produces a parse
        // error — the spatial query needs to know which field to hit.
        let parser = create_test_parser().with_default_field("content");
        // The grammar still tries to match `geo3d_distance(...)` as a
        // bare term_query → field_value → geo3d_query path with no
        // field, which our parse_geo3d_query rejects.
        let err = parser.parse("geo3d_distance(0, 0, 0, 1000)").unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("3D geo queries require an explicit field prefix"),
            "got: {msg}"
        );
    }

    #[test]
    fn test_geo_distance_basic() {
        let parser = create_test_parser().with_default_field("content");
        let q = parser
            .parse("location:geo_distance(40.7128, -74.0060, 10)")
            .unwrap();
        let dbg = format!("{q:?}");
        assert!(dbg.contains("GeoDistanceQuery"), "got: {dbg}");
        assert!(dbg.contains("location"), "got: {dbg}");
    }

    #[test]
    fn test_geo_distance_invalid_lat_rejected() {
        // Latitude > 90 should be rejected by GeoPoint::try_new and
        // surface as a parse error.
        let parser = create_test_parser().with_default_field("content");
        let err = parser
            .parse("location:geo_distance(95.0, 0.0, 10)")
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("invalid geo_distance arguments"), "got: {msg}");
    }

    #[test]
    fn test_geo_bbox_basic() {
        let parser = create_test_parser().with_default_field("content");
        let q = parser
            .parse("location:geo_bbox(40.0, -75.0, 41.0, -74.0)")
            .unwrap();
        let dbg = format!("{q:?}");
        assert!(dbg.contains("GeoBoundingBoxQuery"), "got: {dbg}");
        assert!(dbg.contains("location"), "got: {dbg}");
    }

    #[test]
    fn test_geo_bbox_crosses_the_antimeridian() {
        // min_lon > max_lon is a box that crosses the antimeridian (#1595).
        let parser = create_test_parser().with_default_field("content");
        let query = parser
            .parse("location:geo_bbox(-20, 175, -15, -175)")
            .unwrap();
        let bbox = query
            .as_any()
            .downcast_ref::<GeoBoundingBoxQuery>()
            .expect("a geo_bbox builds a GeoBoundingBoxQuery")
            .bounding_box();
        assert_eq!((bbox.min.lat, bbox.min.lon), (-20.0, 175.0));
        assert_eq!((bbox.max.lat, bbox.max.lon), (-15.0, -175.0));
    }

    #[test]
    fn test_geo_bbox_inverted_rejected() {
        // The pest grammar accepts the syntactic form but the
        // GeoBoundingBox constructor rejects min_lat > max_lat.
        let parser = create_test_parser().with_default_field("content");
        let err = parser
            .parse("location:geo_bbox(50.0, 0.0, 40.0, 10.0)")
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("invalid geo_bbox arguments"), "got: {msg}");
    }

    #[test]
    fn test_geo_query_combined_with_boolean() {
        // 2D geo queries can mix with the rest of the DSL.
        let parser = create_test_parser().with_default_field("content");
        let q = parser
            .parse("city:Tokyo AND location:geo_distance(35.68, 139.76, 5)")
            .unwrap();
        let dbg = format!("{q:?}");
        assert!(dbg.contains("BooleanQuery"), "got: {dbg}");
        assert!(dbg.contains("GeoDistanceQuery"), "got: {dbg}");
    }

    #[test]
    fn test_geo_without_field_rejected() {
        // Bare `geo_distance(...)` (without a field prefix) is rejected.
        let parser = create_test_parser().with_default_field("content");
        let err = parser.parse("geo_distance(0, 0, 1)").unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("2D geo queries require an explicit field prefix"),
            "got: {msg}"
        );
    }

    // ---- Quoted values through stacked tokens (#1252) ----

    /// Whitespace + lowercase + synonyms {big, large} and {ml, machine
    /// learning}, keeping the originals.
    fn synonym_parser() -> LexicalQueryParser {
        synonym_parser_with(&[&["big", "large"], &["ml", "machine learning"]])
    }

    /// Whitespace + lowercase + the synonym `groups`, keeping the originals.
    fn synonym_analyzer(
        groups: &[&[&str]],
    ) -> crate::analysis::analyzer::pipeline::PipelineAnalyzer {
        use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
        use crate::analysis::synonym::dictionary::SynonymDictionary;
        use crate::analysis::token_filter::lowercase::LowercaseFilter;
        use crate::analysis::token_filter::synonym_graph::SynonymGraphFilter;
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

        let mut dict = SynonymDictionary::new(None).unwrap();
        for group in groups {
            dict.add_synonym_group(group.iter().map(|s| s.to_string()).collect());
        }
        PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new()))
            .add_filter(Arc::new(LowercaseFilter::new()))
            .add_filter(Arc::new(SynonymGraphFilter::with_tokenizer(
                dict,
                Box::new(WhitespaceTokenizer::new()),
                true,
            )))
    }

    /// A parser analyzing with [`synonym_analyzer`].
    fn synonym_parser_with(groups: &[&[&str]]) -> LexicalQueryParser {
        LexicalQueryParser::new(Arc::new(synonym_analyzer(groups))).with_default_field("body")
    }

    /// The Should clauses of `query`, each as `(description, boost)`,
    /// sorted; a query that is not a Should-only BooleanQuery is one clause.
    fn should_clauses(query: &dyn Query) -> Vec<(String, f32)> {
        let mut clauses: Vec<(String, f32)> = match query.as_any().downcast_ref::<BooleanQuery>() {
            Some(boolean) => {
                assert_eq!(boolean.boost(), 1.0, "the boost goes on each clause");
                boolean
                    .clauses()
                    .iter()
                    .map(|c| {
                        assert_eq!(c.occur, Occur::Should, "{query:?}");
                        (c.query.description(), c.query.boost())
                    })
                    .collect()
            }
            None => vec![(query.description(), query.boost())],
        };
        clauses.sort_by(|a, b| a.0.cmp(&b.0));
        clauses
    }

    fn term(term: &str, boost: f32) -> (String, f32) {
        (
            TermQuery::new("body", term).with_boost(boost).description(),
            boost,
        )
    }

    fn synonym(terms: &[&str], boost: f32) -> (String, f32) {
        let terms: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
        (
            SynonymQuery::new("body", terms)
                .with_boost(boost)
                .description(),
            boost,
        )
    }

    fn phrase(positions: &[&[&str]], slop: u32, boost: f32) -> (String, f32) {
        let positions = positions
            .iter()
            .map(|alternatives| alternatives.iter().map(|t| t.to_string()).collect())
            .collect();
        (
            PhraseQuery::from_positions("body", positions)
                .with_slop(slop)
                .description(),
            boost,
        )
    }

    fn sorted(mut clauses: Vec<(String, f32)>) -> Vec<(String, f32)> {
        clauses.sort_by(|a, b| a.0.cmp(&b.0));
        clauses
    }

    /// One position with stacked tokens is one blended `SynonymQuery`
    /// (Issue #1257), matching Lucene's `SynonymQuery`, and needs no
    /// positions — not an OR of independent `TermQuery`s, which would
    /// double-count a document holding every alternative.
    #[test]
    fn quoted_word_with_stacked_synonyms_becomes_a_synonym_query() {
        let query = synonym_parser().parse("\"big\"").unwrap();
        assert_eq!(
            should_clauses(query.as_ref()),
            vec![synonym(&["big", "large"], 1.0)]
        );
    }

    /// A quoted word absent from the synonym dictionary has only one
    /// alternative at its position, so it stays a plain `TermQuery`
    /// rather than a one-term `SynonymQuery`.
    #[test]
    fn quoted_word_without_a_synonym_stays_a_term_query() {
        let query = synonym_parser().parse("\"dog\"").unwrap();
        assert_eq!(should_clauses(query.as_ref()), vec![term("dog", 1.0)]);
    }

    /// The boost sits on each clause itself; slop has nothing to apply to.
    #[test]
    fn stacked_alternatives_carry_the_boost_on_each_clause() {
        let query = synonym_parser().parse("\"big\"~3^2").unwrap();
        assert_eq!(
            should_clauses(query.as_ref()),
            vec![synonym(&["big", "large"], 2.0)]
        );
    }

    #[test]
    fn phrase_through_a_stacked_position_has_alternatives_there() {
        let query = synonym_parser().parse("\"a Big dog\"~2^3").unwrap();
        assert_eq!(
            should_clauses(query.as_ref()),
            vec![phrase(&[&["a"], &["big", "large"], &["dog"]], 2, 3.0)]
        );
    }

    fn graph(arcs: &[(u32, u32, &[&str])], slop: u32, boost: f32) -> (String, f32) {
        let arcs = arcs
            .iter()
            .map(|&(from, to, terms)| {
                PhraseArc::new(from, to, terms.iter().map(|t| t.to_string()).collect())
            })
            .collect();
        (
            GraphPhraseQuery::from_arcs("body", arcs)
                .with_slop(slop)
                .description(),
            boost,
        )
    }

    /// The Should clauses of `query`, or `query` itself.
    fn clauses(query: &dyn Query) -> Vec<&dyn Query> {
        match query.as_any().downcast_ref::<BooleanQuery>() {
            Some(boolean) => boolean.clauses().iter().map(|c| c.query.as_ref()).collect(),
            None => vec![query],
        }
    }

    /// The kind of each clause of `query`, sorted.
    fn clause_kinds(query: &dyn Query) -> Vec<&'static str> {
        let mut kinds: Vec<&str> = clauses(query)
            .into_iter()
            .map(|clause| {
                let any = clause.as_any();
                if any.is::<TermQuery>() {
                    "term"
                } else if any.is::<SynonymQuery>() {
                    "synonym"
                } else if any.is::<PhraseQuery>() {
                    "phrase"
                } else if any.is::<GraphPhraseQuery>() {
                    "graph"
                } else {
                    panic!("unexpected clause {clause:?}")
                }
            })
            .collect();
        kinds.sort();
        kinds
    }

    /// The phrases `query` matches, sorted, each as its positions with
    /// their alternatives: a term is one position, a synonym query is one
    /// position with its (sorted, for a deterministic comparison)
    /// alternatives, and a graph phrase stands for every path through it.
    fn phrases(query: &dyn Query) -> Vec<Vec<Vec<String>>> {
        use crate::lexical::query::graph_phrase::test_support::paths;

        let mut phrases = Vec::new();
        for clause in clauses(query) {
            let any = clause.as_any();
            if let Some(term) = any.downcast_ref::<TermQuery>() {
                phrases.push(vec![vec![term.term().to_string()]]);
            } else if let Some(synonym) = any.downcast_ref::<SynonymQuery>() {
                let mut terms = synonym.terms().to_vec();
                terms.sort();
                phrases.push(vec![terms]);
            } else if let Some(phrase) = any.downcast_ref::<PhraseQuery>() {
                phrases.push(phrase.positions().to_vec());
            } else if let Some(graph) = any.downcast_ref::<GraphPhraseQuery>() {
                phrases.extend(paths(graph.arcs()));
            } else {
                panic!("unexpected clause {clause:?}");
            }
        }
        phrases.sort();
        phrases
    }

    fn sorted_phrases(phrases: &[&[&[&str]]]) -> Vec<Vec<Vec<String>>> {
        let mut phrases: Vec<Vec<Vec<String>>> = phrases
            .iter()
            .map(|positions| {
                positions
                    .iter()
                    .map(|alternatives| alternatives.iter().map(|t| t.to_string()).collect())
                    .collect()
            })
            .collect();
        phrases.sort();
        phrases
    }

    const ML_IS: &[(u32, u32, &[&str])] = &[
        (0, 1, &["machine"]),
        (0, 2, &["ml"]),
        (1, 2, &["learning"]),
        (2, 3, &["is"]),
    ];

    /// A multi-word synonym makes several paths. A one-position path is a
    /// term and a lone longer path a phrase, as without synonyms; two or
    /// more longer paths are one graph phrase (#1271).
    #[test]
    fn a_multi_word_synonym_makes_several_paths() {
        let expected = sorted(vec![
            term("ml", 1.0),
            phrase(&[&["machine"], &["learning"]], 0, 1.0),
        ]);
        for dsl in ["\"ml\"", "\"machine learning\""] {
            let query = synonym_parser().parse(dsl).unwrap();
            assert_eq!(should_clauses(query.as_ref()), expected, "{dsl}");
        }

        let query = synonym_parser().parse("\"ml is\"").unwrap();
        assert_eq!(should_clauses(query.as_ref()), vec![graph(ML_IS, 0, 1.0)]);
        assert_eq!(
            phrases(query.as_ref()),
            sorted_phrases(&[&[&["ml"], &["is"]], &[&["machine"], &["learning"], &["is"]]])
        );
    }

    /// #1262: each multi-word member has inner nodes of its own, so a
    /// quoted member expands into exactly its group's members, whatever
    /// their lengths, and never into a mix of their words.
    #[test]
    fn each_multi_word_member_is_its_own_path() {
        let parser =
            synonym_parser_with(&[&["ml", "machine learning", "statistical machine learning"]]);
        let expected = sorted_phrases(&[
            &[&["ml"]],
            &[&["machine"], &["learning"]],
            &[&["statistical"], &["machine"], &["learning"]],
        ]);
        for dsl in [
            "\"ml\"",
            "\"machine learning\"",
            "\"statistical machine learning\"",
        ] {
            let query = parser.parse(dsl).unwrap();
            assert_eq!(phrases(query.as_ref()), expected, "{dsl}");
            assert_eq!(clause_kinds(query.as_ref()), ["graph", "term"], "{dsl}");
        }

        let parser = synonym_parser_with(&[&["new york", "big apple"]]);
        let expected = sorted_phrases(&[&[&["new"], &["york"]], &[&["big"], &["apple"]]]);
        for dsl in ["\"new york\"", "\"big apple\""] {
            let query = parser.parse(dsl).unwrap();
            assert_eq!(phrases(query.as_ref()), expected, "{dsl}");
            assert_eq!(clause_kinds(query.as_ref()), ["graph"], "{dsl}");
        }
    }

    /// #1257: several single-word members of the same group span the
    /// whole quoted value (`direct`, no intermediate node), so they are
    /// one blended `SynonymQuery`, sibling to a longer member's own
    /// phrase clause — not one `TermQuery` per member, which would
    /// double-count a document holding every one of them.
    #[test]
    fn a_direct_synonym_group_becomes_one_synonym_query() {
        let parser = synonym_parser_with(&[&["ml", "ai", "machine learning"]]);
        let expected = sorted_phrases(&[&[&["ai", "ml"]], &[&["machine"], &["learning"]]]);
        for dsl in ["\"ml\"", "\"ai\"", "\"machine learning\""] {
            let query = parser.parse(dsl).unwrap();
            assert_eq!(phrases(query.as_ref()), expected, "{dsl}");
            assert_eq!(clause_kinds(query.as_ref()), ["phrase", "synonym"], "{dsl}");
        }
    }

    /// Paths multiply, so seven multi-word synonyms make 128 of them,
    /// which was past the old cap of 64. The graph phrase grows with the
    /// arcs instead, three per synonym (#1271).
    #[test]
    fn seven_multi_word_synonyms_build_one_graph_phrase() {
        let query = synonym_parser().parse("\"ml ml ml ml ml ml ml\"").unwrap();
        let graph = query
            .as_any()
            .downcast_ref::<GraphPhraseQuery>()
            .expect("one graph phrase");
        assert_eq!(graph.arcs().len(), 21);
        assert_eq!(graph.arcs().iter().map(|arc| arc.to).max(), Some(14));
        assert_eq!(phrases(query.as_ref()).len(), 128);
    }

    /// #1259: a stop word removed after the synonym filter leaves each
    /// member as a path without it, and no path of a member alone.
    #[test]
    fn a_stop_word_after_the_synonyms_leaves_each_member_a_path() {
        use crate::analysis::token_filter::stop::StopFilter;

        let parser = |groups: &[&[&str]]| {
            let analyzer = synonym_analyzer(groups)
                .add_filter(Arc::new(StopFilter::from_words(vec!["of", "the"])));
            LexicalQueryParser::new(Arc::new(analyzer)).with_default_field("body")
        };

        let query = parser(&[&["statue of liberty", "lady liberty"]])
            .parse("\"statue of liberty\"")
            .unwrap();
        assert_eq!(
            phrases(query.as_ref()),
            sorted_phrases(&[&[&["lady"], &["liberty"]], &[&["statue"], &["liberty"]]])
        );

        let query = parser(&[&["usa", "united states of america", "united states"]])
            .parse("\"usa rocks\"")
            .unwrap();
        assert_eq!(
            phrases(query.as_ref()),
            sorted_phrases(&[
                &[&["usa"], &["rocks"]],
                &[&["united"], &["states"], &["america"], &["rocks"]],
                &[&["united"], &["states"], &["rocks"]],
            ])
        );
    }

    #[test]
    fn graph_phrase_carries_slop_and_boost() {
        let query = synonym_parser().parse("\"ml is\"~2^3").unwrap();
        assert_eq!(should_clauses(query.as_ref()), vec![graph(ML_IS, 2, 3.0)]);
    }

    // ---- Bare (unquoted) values through stacked tokens (#1257) ----

    /// The same double-counting bug exists for a bare word — the most
    /// common way to hit it, since most searches are unquoted — so a
    /// stacked position becomes one blended `SynonymQuery` here too, not
    /// an OR of `TermQuery`s.
    #[test]
    fn unquoted_word_with_stacked_synonyms_becomes_a_synonym_query() {
        let query = synonym_parser().parse("big").unwrap();
        assert_eq!(
            should_clauses(query.as_ref()),
            vec![synonym(&["big", "large"], 1.0)]
        );
    }

    /// A bare multi-word input still OR's its positions (`parse_simple_term`'s
    /// pre-existing behavior for Japanese sentences and hyphenated words),
    /// but a stacked position among them is one `SynonymQuery` clause
    /// rather than one `TermQuery` per alternative.
    #[test]
    fn unquoted_multi_position_term_mixes_term_and_synonym_clauses() {
        let query = synonym_parser().parse("big dog").unwrap();
        assert_eq!(
            should_clauses(query.as_ref()),
            sorted(vec![synonym(&["big", "large"], 1.0), term("dog", 1.0)])
        );
    }

    /// An analyzer that always emits the same tokens.
    #[derive(Debug)]
    struct Fixed(Vec<Token>);

    impl Analyzer for Fixed {
        fn analyze(&self, _text: &str) -> Result<crate::analysis::token::TokenStream> {
            Ok(Box::new(self.0.clone().into_iter()))
        }
        fn name(&self) -> &'static str {
            "fixed"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// An arc on no complete path is dropped: here "a b" stops at node 2,
    /// short of the last node, so only "c" is left.
    #[test]
    fn arcs_off_every_complete_path_are_dropped() {
        let tokens = vec![
            Token::new("c", 0).with_position_length(3),
            Token::new("a", 0).with_position_increment(0),
            Token::new("b", 1),
        ];
        let parser = LexicalQueryParser::new(Arc::new(Fixed(tokens))).with_default_field("body");
        let query = parser.parse("\"anything\"").unwrap();
        assert_eq!(should_clauses(query.as_ref()), vec![term("c", 1.0)]);
    }

    /// A graph with no complete path (here "b" starts at a node nothing
    /// reaches) falls back to the positions the index stores, ignoring
    /// `position_length`.
    #[test]
    fn a_graph_without_a_complete_path_falls_back_to_positions() {
        let tokens = vec![
            Token::new("a", 0).with_position_length(2),
            Token::new("b", 1).with_position_length(3),
        ];
        let parser = LexicalQueryParser::new(Arc::new(Fixed(tokens))).with_default_field("body");
        let query = parser.parse("\"anything\"").unwrap();
        assert_eq!(
            should_clauses(query.as_ref()),
            vec![phrase(&[&["a"], &["b"]], 0, 1.0)]
        );
    }

    // ---- Token boosts reach the query (Issue #1701) ----

    /// A graph without a complete path falls back to its flat positions,
    /// and the phrase still carries the product of their token boosts:
    /// 2 × 3 = 6.
    #[test]
    fn the_flat_fallback_carries_the_product_of_its_token_boosts() {
        let tokens = vec![
            Token::new("a", 0).with_position_length(2).with_boost(2.0),
            Token::new("b", 1).with_position_length(3).with_boost(3.0),
        ];
        let parser = LexicalQueryParser::new(Arc::new(Fixed(tokens))).with_default_field("body");
        let query = parser.parse("\"anything\"").unwrap();
        assert_eq!(
            should_clauses(query.as_ref()),
            vec![phrase(&[&["a"], &["b"]], 0, 6.0)]
        );
    }

    /// A parser whose analyzer is `tokenizer` followed by a `BoostFilter`
    /// of `boost`.
    fn boost_filter_parser(
        tokenizer: Arc<dyn crate::analysis::tokenizer::Tokenizer>,
        boost: f32,
    ) -> LexicalQueryParser {
        use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
        use crate::analysis::token_filter::boost::BoostFilter;

        let analyzer =
            PipelineAnalyzer::new(tokenizer).add_filter(Arc::new(BoostFilter::new(boost)));
        LexicalQueryParser::new(Arc::new(analyzer)).with_default_field("body")
    }

    fn whitespace_boost_parser(boost: f32) -> LexicalQueryParser {
        use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;
        boost_filter_parser(Arc::new(WhitespaceTokenizer::new()), boost)
    }

    /// A bare term's token boost multiplies the query's: a `BoostFilter`
    /// of 2 makes `rust` a `TermQuery` of boost 2, and `rust^3` one of 6.
    /// The filter used to change no score.
    #[test]
    fn a_bare_term_carries_its_token_boost() {
        let parser = whitespace_boost_parser(2.0);
        for (input, expected) in [("rust", 2.0), ("rust^3", 6.0)] {
            let query = parser.parse(input).unwrap();
            assert_eq!(
                should_clauses(query.as_ref()),
                vec![term("rust", expected)],
                "{input}"
            );
        }
    }

    /// A bare term the analyzer splits into several positions carries each
    /// position's token boost on its clause, and the query's boost on the
    /// `BooleanQuery` around them.
    #[test]
    fn each_position_of_a_split_bare_term_carries_its_token_boost() {
        use crate::analysis::tokenizer::regex::RegexTokenizer;

        let parser = boost_filter_parser(Arc::new(RegexTokenizer::new().unwrap()), 2.0);
        for (input, query_boost) in [("rust-lang", 1.0), ("rust-lang^3", 3.0)] {
            let query = parser.parse(input).unwrap();
            let boolean = query
                .as_any()
                .downcast_ref::<BooleanQuery>()
                .unwrap_or_else(|| panic!("{input}: {query:?}"));
            assert_eq!(boolean.boost(), query_boost, "{input}");
            let clauses = sorted(
                boolean
                    .clauses()
                    .iter()
                    .map(|c| (c.query.description(), c.query.boost()))
                    .collect(),
            );
            assert_eq!(
                clauses,
                sorted(vec![term("rust", 2.0), term("lang", 2.0)]),
                "{input}"
            );
        }
    }

    /// A quoted phrase carries the product of its positions' token boosts,
    /// as in Lucene's `QueryBuilder::analyzePhrase`: 2 × 2 = 4, and 12 with
    /// `^3`.
    #[test]
    fn a_phrase_carries_the_product_of_its_token_boosts() {
        let parser = whitespace_boost_parser(2.0);
        for (input, expected) in [("\"rust lang\"", 4.0), ("\"rust lang\"^3", 12.0)] {
            let query = parser.parse(input).unwrap();
            assert_eq!(
                should_clauses(query.as_ref()),
                vec![phrase(&[&["rust"], &["lang"]], 0, expected)],
                "{input}"
            );
        }
    }

    /// The alternatives stacked at one position share their largest token
    /// boost, not the first token's: here the first is 0.5 and the largest
    /// 1.0.
    #[test]
    fn stacked_alternatives_share_their_largest_token_boost() {
        let tokens = vec![
            Token::new("large", 0).with_boost(0.5),
            Token::new("big", 0)
                .with_position_increment(0)
                .with_boost(1.0),
        ];
        let parser = LexicalQueryParser::new(Arc::new(Fixed(tokens))).with_default_field("body");
        for input in ["anything", "\"anything\""] {
            let query = parser.parse(input).unwrap();
            assert_eq!(
                should_clauses(query.as_ref()),
                vec![synonym(&["large", "big"], 1.0)],
                "{input}"
            );
        }
    }

    /// In a graph with one path of several positions next to the one-position
    /// path, as Lucene's `QueryBuilder::analyzeGraphPhrase` does for each
    /// path: the one-position path carries its token boost and the phrase
    /// the product of its arcs' boosts.
    #[test]
    fn each_path_of_a_one_phrase_graph_carries_its_token_boosts() {
        let tokens = vec![
            Token::new("nyc", 0).with_position_length(2).with_boost(2.0),
            Token::new("new", 0)
                .with_position_increment(0)
                .with_boost(2.0),
            Token::new("york", 1).with_boost(2.0),
        ];
        let parser = LexicalQueryParser::new(Arc::new(Fixed(tokens))).with_default_field("body");
        let query = parser.parse("\"anything\"").unwrap();
        assert_eq!(
            should_clauses(query.as_ref()),
            sorted(vec![
                term("nyc", 2.0),
                phrase(&[&["new"], &["york"]], 0, 4.0)
            ])
        );
    }

    /// A graph with several paths of two or more positions is one
    /// `GraphPhraseQuery`, which matches them without listing them, so it
    /// keeps the query's boost only: Lucene's multi-phrase path does not use
    /// token boosts either.
    #[test]
    fn a_graph_phrase_keeps_the_query_boost_only() {
        let tokens = vec![
            Token::new("ml", 0).with_position_length(2).with_boost(2.0),
            Token::new("machine", 0)
                .with_position_increment(0)
                .with_boost(2.0),
            Token::new("learning", 1).with_boost(2.0),
            Token::new("is", 2).with_boost(2.0),
        ];
        let parser = LexicalQueryParser::new(Arc::new(Fixed(tokens))).with_default_field("body");
        let query = parser.parse("\"anything\"").unwrap();
        assert_eq!(
            should_clauses(query.as_ref()),
            vec![graph(
                &[
                    (0, 1, &["machine"]),
                    (0, 2, &["ml"]),
                    (1, 2, &["learning"]),
                    (2, 3, &["is"]),
                ],
                0,
                1.0
            )]
        );
    }
}
