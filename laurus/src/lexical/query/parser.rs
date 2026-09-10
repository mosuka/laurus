//! Query parser using pest.
//!
//! This parser supports the full query syntax including:
//! - Field-specific queries: `title:hello`
//! - Boolean operators: `AND`, `OR`
//! - Required/prohibited: `+required`, `-forbidden`
//! - Phrases: `"hello world"` (quoted → `PhraseQuery`)
//! - Bare terms that the analyzer splits into several tokens are OR'd
//!   (`BooleanQuery` with `Should` clauses), matching Lucene's `match`
//!   query. Use quotes for phrase semantics.
//! - Terms removed entirely by analysis match no documents in that field.
//! - Proximity search: `"hello world"~10`
//! - Fuzzy search: `roam~2`
//! - Range queries: `[100 TO 500]`, `{A TO Z}`
//! - Wildcards: `te?t`, `test*`
//! - Boosting: `jakarta^4`
//! - Grouping: `(title:hello OR body:world)`

use std::sync::Arc;

use pest::Parser;
use pest_derive::Parser;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::analysis::analyzer::per_field::PerFieldAnalyzer;
use crate::analysis::analyzer::standard::StandardAnalyzer;
use crate::data::GeoEcefPoint;
use crate::error::{LaurusError, Result};
use crate::lexical::core::field::NumericType;
use crate::lexical::query::Query;
use crate::lexical::query::boolean::{BooleanClause, BooleanQuery, Occur};
use crate::lexical::query::fuzzy::FuzzyQuery;
use crate::lexical::query::geo::{GeoBoundingBoxQuery, GeoDistanceQuery};
use crate::lexical::query::geo3d::{Geo3dBoundingBoxQuery, Geo3dDistanceQuery, Geo3dNearestQuery};
use crate::lexical::query::phrase::PhraseQuery;
use crate::lexical::query::range::NumericRangeQuery;
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
                    let op = inner_pair.as_str();
                    let new_occur = match op.to_uppercase().as_str() {
                        "AND" => Occur::Must,
                        "OR" => Occur::Should,
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
            if boost != 1.0 {
                q.set_boost(boost);
            }
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

        let lower_num = lower.as_ref().and_then(|s| s.parse::<f64>().ok());
        let upper_num = upper.as_ref().and_then(|s| s.parse::<f64>().ok());

        self.create_query_over_fields(field, |field_name| {
            if lower_num.is_some() || upper_num.is_some() {
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
            } else {
                // Text range - use a term query as fallback
                let term = format!(
                    "{}{} TO {}{}",
                    if lower_inclusive { "[" } else { "{" },
                    lower.as_deref().unwrap_or("*"),
                    upper.as_deref().unwrap_or("*"),
                    if upper_inclusive { "]" } else { "}" }
                );
                Ok(Box::new(TermQuery::new(field_name, &term)))
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
    /// `parse_simple_term`), a quoted phrase always builds a `PhraseQuery`
    /// requiring the tokens to appear in sequence (within `slop`). Quoting
    /// is how a caller opts into strict adjacency.
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
                            slop = Some(prox_pair.as_str().parse().unwrap_or(0));
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
            let terms = self.analyze_term(Some(field_name), &phrase_content)?;
            let mut phrase_query = PhraseQuery::new(field_name, terms);

            if let Some(slop_value) = slop {
                phrase_query = phrase_query.with_slop(slop_value);
            }

            if boost != 1.0 {
                phrase_query = phrase_query.with_boost(boost);
            }

            Ok(Box::new(phrase_query))
        })
    }

    fn parse_fuzzy_term(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let mut term = String::new();
        let mut fuzziness: u8 = 2; // Default fuzziness

        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
                Rule::term => {
                    term = inner_pair.as_str().to_string();
                }
                Rule::fuzziness => {
                    for fuzz_pair in inner_pair.into_inner() {
                        if fuzz_pair.as_rule() == Rule::number {
                            fuzziness = fuzz_pair.as_str().parse().unwrap_or(2);
                        }
                    }
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
                FuzzyQuery::new(field_name, normalized_term).max_edits(fuzziness as u32),
            ))
        })
    }

    fn parse_wildcard_term(
        &self,
        pair: pest::iterators::Pair<Rule>,
        field: Option<&str>,
    ) -> Result<Box<dyn Query>> {
        let mut pattern = String::new();

        for inner_pair in pair.into_inner() {
            if inner_pair.as_rule() == Rule::wildcard_pattern {
                pattern = inner_pair.as_str().to_string();
            }
        }

        self.create_query_over_fields(field, |field_name| {
            Ok(Box::new(WildcardQuery::new(field_name, &pattern)?))
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
            let terms = self.analyze_term(Some(field_name), &term)?;

            if terms.is_empty() {
                // An empty phrase already matches nothing and retains the field
                // reference for schema validation, unlike an empty BooleanQuery.
                return Ok(Box::new(
                    PhraseQuery::new(field_name, terms).with_boost(boost),
                ));
            }

            if terms.len() == 1 {
                let query = TermQuery::new(field_name, &terms[0]);
                if boost != 1.0 {
                    Ok(Box::new(query.with_boost(boost)))
                } else {
                    Ok(Box::new(query))
                }
            } else {
                // The analyzer split one *bare* (unquoted) term into several
                // tokens. This happens for a Japanese sentence under a
                // morphological tokenizer, and for inputs like `rust-lang`
                // under a `\w+` tokenizer.
                //
                // Those tokens are OR'd, matching Lucene's `match` query and
                // Elasticsearch's default. The previous slop-0 `PhraseQuery`
                // required the document to repeat the exact morpheme
                // sequence, which made unquoted Japanese queries return
                // almost nothing.
                //
                // Explicit phrase semantics are still available — a quoted
                // `"..."` goes through `parse_phrase_query`, which keeps
                // `PhraseQuery` (and its `~slop` modifier).
                let mut bool_query = BooleanQuery::new();
                for token in &terms {
                    bool_query.add_clause(BooleanClause::new(
                        Box::new(TermQuery::new(field_name, token)),
                        Occur::Should,
                    ));
                }

                // `minimum_should_match` stays at its default of 0, so the
                // matcher builds a plain disjunction and BM25 scores add up
                // across the matching clauses.
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

        let tokens: Vec<String> = token_stream.into_iter().map(|t| t.text).collect();
        Ok(tokens)
    }
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

/// Builder for constructing a [`QueryParser`] with a fluent API.
///
/// Allows configuring the analyzer, default search fields, and default boolean
/// occurrence before building the final [`QueryParser`] instance.
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
            phrase_q.terms().len() >= 2,
            "expected multiple morphemes, got {:?}",
            phrase_q.terms()
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
    fn test_geo_bbox_inverted_rejected() {
        // The pest grammar accepts the syntactic form but the
        // GeoBoundingBox constructor rejects min > max.
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
}
