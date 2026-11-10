//! Range query implementation for querying within value ranges.

use std::fmt::Debug;

use chrono::{DateTime, Utc};

use crate::error::{LaurusError, Result};
use crate::lexical::core::datetime::{datetime_to_point, parse_datetime_literal};
use crate::lexical::core::field::NumericType;
use crate::lexical::index::structures::visitor::value_in_range;
use crate::lexical::query::Query;
use crate::lexical::query::matcher::{EmptyMatcher, Matcher, PreComputedMatcher};
use crate::lexical::query::scorer::{BM25Scorer, Scorer};
use crate::lexical::reader::{LexicalIndexReader, scan_doc_ids};

/// Bound type for range queries.
#[derive(Debug, Clone, PartialEq)]
pub enum Bound<T> {
    /// Inclusive bound.
    Included(T),
    /// Exclusive bound.
    Excluded(T),
    /// Unbounded (no limit).
    Unbounded,
}

impl<T: PartialOrd> Bound<T> {
    /// Check if a value satisfies this bound as a lower bound.
    pub fn contains_lower(&self, value: &T) -> bool {
        match self {
            Bound::Included(bound) => value >= bound,
            Bound::Excluded(bound) => value > bound,
            Bound::Unbounded => true,
        }
    }

    /// Check if a value satisfies this bound as an upper bound.
    pub fn contains_upper(&self, value: &T) -> bool {
        match self {
            Bound::Included(bound) => value <= bound,
            Bound::Excluded(bound) => value < bound,
            Bound::Unbounded => true,
        }
    }
}

/// A query that matches documents with field values within a specified range.
#[derive(Debug, Clone)]
pub struct RangeQuery {
    /// The field to search in.
    field: String,
    /// Lower bound of the range.
    lower_bound: Bound<String>,
    /// Upper bound of the range.
    upper_bound: Bound<String>,
    /// The boost factor for this query.
    boost: f32,
}

impl RangeQuery {
    /// Create a new range query with both bounds inclusive.
    pub fn new<S: Into<String>>(field: S, lower: Option<String>, upper: Option<String>) -> Self {
        let lower_bound = match lower {
            Some(val) => Bound::Included(val),
            None => Bound::Unbounded,
        };
        let upper_bound = match upper {
            Some(val) => Bound::Included(val),
            None => Bound::Unbounded,
        };

        RangeQuery {
            field: field.into(),
            lower_bound,
            upper_bound,
            boost: 1.0,
        }
    }

    /// Create a range query with custom bound types.
    pub fn with_bounds<S: Into<String>>(
        field: S,
        lower_bound: Bound<String>,
        upper_bound: Bound<String>,
    ) -> Self {
        RangeQuery {
            field: field.into(),
            lower_bound,
            upper_bound,
            boost: 1.0,
        }
    }

    /// Create a range query for values greater than or equal to the given value.
    pub fn greater_than_or_equal<S: Into<String>>(field: S, value: String) -> Self {
        Self::with_bounds(field, Bound::Included(value), Bound::Unbounded)
    }

    /// Create a range query for values greater than the given value.
    pub fn greater_than<S: Into<String>>(field: S, value: String) -> Self {
        Self::with_bounds(field, Bound::Excluded(value), Bound::Unbounded)
    }

    /// Create a range query for values less than or equal to the given value.
    pub fn less_than_or_equal<S: Into<String>>(field: S, value: String) -> Self {
        Self::with_bounds(field, Bound::Unbounded, Bound::Included(value))
    }

    /// Create a range query for values less than the given value.
    pub fn less_than<S: Into<String>>(field: S, value: String) -> Self {
        Self::with_bounds(field, Bound::Unbounded, Bound::Excluded(value))
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the lower bound.
    pub fn lower_bound(&self) -> &Bound<String> {
        &self.lower_bound
    }

    /// Get the upper bound.
    pub fn upper_bound(&self) -> &Bound<String> {
        &self.upper_bound
    }

    /// Check if a term falls within the range.
    pub fn contains(&self, term: &str) -> bool {
        self.lower_bound.contains_lower(&term.to_string())
            && self.upper_bound.contains_upper(&term.to_string())
    }
}

impl Query for RangeQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, _reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(EmptyMatcher::new()))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        // Range queries typically match fewer documents than term queries
        let total_docs = reader.doc_count();
        let estimated_doc_freq = total_docs / 10; // Estimate: 10% of docs might match a range
        let estimated_term_freq = estimated_doc_freq * 2; // Average 2 occurrences per doc

        Ok(Box::new(BM25Scorer::new(
            estimated_doc_freq.max(1),
            estimated_term_freq.max(1),
            total_docs,
            10.0, // Estimated average field length
            total_docs,
            self.boost,
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
            "RangeQuery(field:{}, lower:{:?}, upper:{:?})",
            self.field, self.lower_bound, self.upper_bound
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(false)
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        Ok(500) // Range queries are moderately expensive
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field + both bounds (inclusivity is encoded in the `Bound` variant)
        // determine the matched set; boost is score-only and excluded.
        Some(format!(
            "range|{:?}|{:?}|{:?}",
            self.field, self.lower_bound, self.upper_bound
        ))
    }
}

/// Matcher for range queries.
#[derive(Debug)]
pub struct RangeMatcher {
    /// Current document ID.
    current_doc: u64,
    /// Boost factor for scoring.
    #[allow(dead_code)]
    boost: f32,
    /// Whether we've reached the end.
    exhausted: bool,
}

impl RangeMatcher {
    /// Create a new range matcher.
    pub fn new(boost: f32) -> Self {
        RangeMatcher {
            current_doc: 0,
            boost,
            exhausted: true,
        }
    }
}

impl Matcher for RangeMatcher {
    fn doc_id(&self) -> u64 {
        if self.exhausted {
            u64::MAX
        } else {
            self.current_doc
        }
    }

    fn next(&mut self) -> Result<bool> {
        Ok(false)
    }

    fn skip_to(&mut self, _target: u64) -> Result<bool> {
        Ok(false)
    }

    fn cost(&self) -> u64 {
        0
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Specialized numeric range query for optimal performance.
#[derive(Debug, Clone)]
pub struct NumericRangeQuery {
    /// The field to search in.
    field: String,
    /// The numeric type of the field.
    numeric_type: NumericType,
    /// Lower bound as bytes (for efficient comparison).
    lower_bound: Option<Vec<u8>>,
    /// Upper bound as bytes (for efficient comparison).
    upper_bound: Option<Vec<u8>>,
    /// Whether lower bound is inclusive.
    lower_inclusive: bool,
    /// Whether upper bound is inclusive.
    upper_inclusive: bool,
    /// The boost factor for this query.
    boost: f32,
    /// Whether a bound given to [`Self::new`] was `NaN`, which
    /// [`Query::validate_boosts`] rejects at search (Issue #1601). The
    /// encoded bounds cannot tell: an `Integer` range truncates a `NaN`
    /// bound to 0.
    has_nan_bound: bool,
}

impl NumericRangeQuery {
    /// Create a new numeric range query.
    ///
    /// `None` leaves a side open. A bound compares as an `f64`, as the BKD
    /// tree does, so `f64::NEG_INFINITY` as the lower bound and
    /// `f64::INFINITY` as the upper bound are the same as `None`, while
    /// `f64::INFINITY` as the lower bound matches only that infinity. For
    /// [`NumericType::Integer`] a bound is truncated to an `i64`, an
    /// infinity to `i64::MIN` or `i64::MAX`.
    ///
    /// A `NaN` bound matches no value, so it is not an error here but at
    /// search: [`Query::validate_boosts`] rejects it with
    /// [`LaurusError::InvalidArgument`], whichever route the query comes
    /// from (Issue #1601).
    pub fn new<S: Into<String>>(
        field: S,
        numeric_type: NumericType,
        lower: Option<f64>,
        upper: Option<f64>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Self {
        let has_nan_bound = lower.is_some_and(f64::is_nan) || upper.is_some_and(f64::is_nan);
        let lower_bound = lower.map(|v| Self::encode_numeric(v, numeric_type));
        let upper_bound = upper.map(|v| Self::encode_numeric(v, numeric_type));

        NumericRangeQuery {
            field: field.into(),
            numeric_type,
            lower_bound,
            upper_bound,
            lower_inclusive,
            upper_inclusive,
            boost: 1.0,
            has_nan_bound,
        }
    }

    /// Create a range query for integers.
    pub fn i64_range<S: Into<String>>(field: S, lower: Option<i64>, upper: Option<i64>) -> Self {
        Self::new(
            field,
            NumericType::Integer,
            lower.map(|v| v as f64),
            upper.map(|v| v as f64),
            true,
            true,
        )
    }

    /// Create a range query for floats with inclusive bounds.
    pub fn f64_range<S: Into<String>>(field: S, lower: Option<f64>, upper: Option<f64>) -> Self {
        Self::new(field, NumericType::Float, lower, upper, true, true)
    }

    /// Create a range query for floats with exclusive upper bound.
    pub fn f64_range_exclusive_upper<S: Into<String>>(
        field: S,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Self {
        Self::new(field, NumericType::Float, lower, upper, true, false)
    }

    /// Create a range query for floats with exclusive lower bound.
    pub fn f64_range_exclusive_lower<S: Into<String>>(
        field: S,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Self {
        Self::new(field, NumericType::Float, lower, upper, false, true)
    }

    /// Create a range query for floats with both bounds exclusive.
    pub fn f64_range_exclusive<S: Into<String>>(
        field: S,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Self {
        Self::new(field, NumericType::Float, lower, upper, false, false)
    }

    /// Create a greater than query.
    pub fn greater_than<S: Into<String>>(field: S, numeric_type: NumericType, value: f64) -> Self {
        Self::new(field, numeric_type, Some(value), None, false, true)
    }

    /// Create a greater than or equal query.
    pub fn greater_than_or_equal<S: Into<String>>(
        field: S,
        numeric_type: NumericType,
        value: f64,
    ) -> Self {
        Self::new(field, numeric_type, Some(value), None, true, true)
    }

    /// Create a less than query.
    pub fn less_than<S: Into<String>>(field: S, numeric_type: NumericType, value: f64) -> Self {
        Self::new(field, numeric_type, None, Some(value), true, false)
    }

    /// Create a less than or equal query.
    pub fn less_than_or_equal<S: Into<String>>(
        field: S,
        numeric_type: NumericType,
        value: f64,
    ) -> Self {
        Self::new(field, numeric_type, None, Some(value), true, true)
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the numeric type.
    pub fn numeric_type(&self) -> NumericType {
        self.numeric_type
    }

    /// Encode a numeric value as the bytes this query holds a bound in: the
    /// big-endian bytes of the `f64` for [`NumericType::Float`], and of the
    /// value truncated to an `i64` for [`NumericType::Integer`].
    ///
    /// The bytes are not in numeric order: a negative number has its sign
    /// bit set, so its bytes compare above those of every non-negative
    /// number. Compare through [`Self::contains_numeric`] or
    /// [`Self::contains_encoded`], which decode the bytes and compare
    /// numbers (Issue #1600).
    pub fn encode_numeric(value: f64, numeric_type: NumericType) -> Vec<u8> {
        match numeric_type {
            NumericType::Integer => (value as i64).to_be_bytes().to_vec(),
            NumericType::Float => value.to_be_bytes().to_vec(),
        }
    }

    /// Check if an encoded value, as [`Self::encode_numeric`] returns it
    /// for this query's numeric type, falls within the range.
    ///
    /// The bytes are decoded to a number and compared by
    /// [`Self::contains_numeric`], not compared as bytes, since their byte
    /// order is not numeric order (Issue #1600). A value that is not 8 bytes
    /// long is not in the range.
    pub fn contains_encoded(&self, encoded_value: &[u8]) -> bool {
        let Ok(bytes) = <[u8; 8]>::try_from(encoded_value) else {
            return false;
        };
        let value = match self.numeric_type {
            NumericType::Integer => i64::from_be_bytes(bytes) as f64,
            NumericType::Float => f64::from_be_bytes(bytes),
        };
        self.contains_numeric(value)
    }

    /// Check if a numeric value falls within the range.
    ///
    /// The value is compared with the bounds as numbers, by the check the
    /// BKD tree's
    /// [`RangeQueryVisitor`](crate::lexical::index::structures::visitor::RangeQueryVisitor)
    /// applies to each point, so the stored-document scan of
    /// [`Query::matcher`] matches the same values as the tree
    /// (Issue #1600): negative values and bounds compare in
    /// numeric order, `-0.0` equals `0.0`, and a `NaN` value is in no
    /// range. An open side is `f64::NEG_INFINITY` or `f64::INFINITY`, as in
    /// the tree. For [`NumericType::Integer`] the value is not truncated:
    /// the tree, too, compares its `f64` point with the bounds.
    pub fn contains_numeric(&self, value: f64) -> bool {
        value_in_range(
            value,
            self.min_f64().unwrap_or(f64::NEG_INFINITY),
            self.max_f64().unwrap_or(f64::INFINITY),
            self.lower_inclusive,
            self.upper_inclusive,
        )
    }

    /// Get the minimum f64 value if the range has a lower bound.
    pub fn min_f64(&self) -> Option<f64> {
        self.lower_bound
            .as_ref()
            .map(|bytes| match self.numeric_type {
                NumericType::Float => f64::from_be_bytes(
                    bytes
                        .as_slice()
                        .try_into()
                        .expect("numeric bound must be 8 bytes"),
                ),
                NumericType::Integer => i64::from_be_bytes(
                    bytes
                        .as_slice()
                        .try_into()
                        .expect("numeric bound must be 8 bytes"),
                ) as f64,
            })
    }

    /// Get the maximum f64 value if the range has an upper bound.
    pub fn max_f64(&self) -> Option<f64> {
        self.upper_bound
            .as_ref()
            .map(|bytes| match self.numeric_type {
                NumericType::Float => f64::from_be_bytes(
                    bytes
                        .as_slice()
                        .try_into()
                        .expect("numeric bound must be 8 bytes"),
                ),
                NumericType::Integer => i64::from_be_bytes(
                    bytes
                        .as_slice()
                        .try_into()
                        .expect("numeric bound must be 8 bytes"),
                ) as f64,
            })
    }

    /// Get the minimum i64 value if the range has a lower bound.
    pub fn min_i64(&self) -> Option<i64> {
        self.min_f64().map(|v| v as i64)
    }

    /// Get the maximum i64 value if the range has an upper bound.
    pub fn max_i64(&self) -> Option<i64> {
        self.max_f64().map(|v| v as i64)
    }

    /// Count the number of documents that match this range query.
    ///
    /// For multi-valued fields, a document is counted at most once even
    /// when several of its values fall in the range (Lucene-style "any
    /// match" semantics).
    ///
    /// This probes every stored document present in the reader (O(n));
    /// it is no longer called by `scorer()` — range queries are
    /// constant-scored and need no matching-document count (#994) — but
    /// is retained as a public utility.
    ///
    /// # Arguments
    ///
    /// * `reader` - The index reader whose stored documents are probed.
    ///
    /// # Returns
    ///
    /// The number of documents with at least one value in this range.
    pub fn count_matching_documents(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        let mut count = 0u64;
        for doc_id in scan_doc_ids(reader)? {
            if let Ok(Some(doc)) = reader.document(doc_id)
                && let Some(val) = doc.get(&self.field)
                && self.value_matches_any(val)
            {
                count += 1;
            }
        }

        Ok(count)
    }

    /// Returns `true` if any numeric component of `val` is in this range.
    ///
    /// Single-valued fields contribute one numeric value; multi-valued
    /// (`Int64Array` / `Float64Array` / `DateTimeArray`) fields are scanned
    /// for any element that satisfies the predicate. A `DateTime`
    /// contributes its BKD point (#1179), so the stored-document fallback
    /// agrees with the tree. `Text` (and, since #1175, `TextArray`
    /// element-wise) is parsed as a number because stored docs lose type
    /// info in some paths. Other variants return `false`. Every number is
    /// tested by [`Self::contains_numeric`], with the tree's per-point
    /// check (Issue #1600).
    fn value_matches_any(&self, val: &crate::data::DataValue) -> bool {
        match val {
            crate::data::DataValue::Int64(i) => self.contains_numeric(*i as f64),
            crate::data::DataValue::Float64(f) => self.contains_numeric(*f),
            crate::data::DataValue::Int64Array(arr) => {
                arr.iter().any(|i| self.contains_numeric(*i as f64))
            }
            crate::data::DataValue::Float64Array(arr) => {
                arr.iter().any(|f| self.contains_numeric(*f))
            }
            crate::data::DataValue::DateTime(dt) => self.contains_numeric(datetime_to_point(dt)),
            crate::data::DataValue::DateTimeArray(arr) => arr
                .iter()
                .any(|dt| self.contains_numeric(datetime_to_point(dt))),
            crate::data::DataValue::Text(s) => s
                .parse::<f64>()
                .ok()
                .is_some_and(|n| self.contains_numeric(n)),
            // Symmetric with the scalar `Text` arm above (#1175).
            crate::data::DataValue::TextArray(arr) => arr.iter().any(|s| {
                s.parse::<f64>()
                    .ok()
                    .is_some_and(|n| self.contains_numeric(n))
            }),
            _ => false,
        }
    }
}

/// A scorer specialized for numeric range queries.
///
/// Range queries use Lucene-style **constant scoring**: every matching
/// document receives the same score (`boost`), regardless of how close
/// each value is to the range bounds or how many of a multi-valued
/// document's values fall in range. The previous IDF + proximity scoring
/// was removed when multi-valued numeric fields landed (#281), since per
/// per-value weighting interacts poorly with "any match" semantics and
/// produced surprising rankings.
#[derive(Debug, Clone)]
pub struct RangeScorer {
    /// Boost factor; the per-document score is exactly this value.
    boost: f32,
}

impl RangeScorer {
    /// Create a new range scorer.
    ///
    /// Bounds, matching-document count, and total document count were
    /// used by the legacy IDF / proximity scorer; they are accepted (and
    /// ignored) so existing call sites continue to compile. The
    /// matching-document count in particular required an O(total_docs)
    /// stored-document scan per segment that constant scoring never
    /// needed (#994).
    pub fn new(
        _lower_bound: Option<f64>,
        _upper_bound: Option<f64>,
        _matching_docs: u64,
        _total_docs: u64,
        boost: f32,
    ) -> Self {
        RangeScorer { boost }
    }
}

impl Scorer for RangeScorer {
    /// Constant score for range matches.
    ///
    /// Matches Lucene's
    /// [`PointRangeQuery`](https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/PointRangeQuery.html)
    /// behaviour: a range query is a binary "in range / not in range"
    /// predicate, scored uniformly by the boost factor. This applies to
    /// both single-valued and multi-valued numeric fields — when several
    /// of a multi-valued document's values fall in range, the document
    /// still contributes a single constant-scored hit (the BKD reader
    /// deduplicates `doc_id`s during range search).
    ///
    /// The previous IDF + proximity heuristic confused users when the same
    /// query produced different scores for the same document depending on
    /// surrounding queries; constant scoring keeps range query semantics
    /// predictable and aligns with the dynamic-schema spec.
    fn score(&self, _doc_id: u64, _value: f32, _field_length: Option<f32>) -> f32 {
        self.boost
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    /// Upper bound for pruning: the constant `boost`, unconditionally.
    ///
    /// An over-estimate (e.g. when the matcher turns out to be empty) is
    /// safe — pruning only compares against this bound, so a looser bound
    /// merely prunes less. Returning `0.0` for "no matches" (the previous
    /// behavior) required an exact O(total_docs) counting scan and could
    /// truncate results when the count disagreed with the matcher, e.g.
    /// on `indexed=true, stored=false` fields (#994).
    fn max_score(&self) -> f32 {
        self.boost
    }

    fn name(&self) -> &'static str {
        "RangeScorer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Query for NumericRangeQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        // Try to use BKD Tree if available
        if let Some(bkd_tree) = reader.get_bkd_tree(&self.field)? {
            let min_value = self.min_f64();
            let max_value = self.max_f64();

            let mins = [min_value];
            let maxs = [max_value];

            let doc_ids =
                bkd_tree.range_search(&mins, &maxs, self.lower_inclusive, self.upper_inclusive)?;
            return Ok(Box::new(PreComputedMatcher::new(doc_ids)));
        }

        // Fallback: scan the stored documents present in this reader and
        // filter by numeric range. Multi-valued fields match if ANY value
        // falls in the range (Lucene-style "any match"), and the document
        // is recorded at most once per query. The predicate is shared with
        // `count_matching_documents` so the two can never disagree, and it
        // compares numbers with the BKD tree's per-point check, so a
        // segment without a tree matches what a tree would (Issue #1600).
        let mut matching_docs = Vec::new();

        for doc_id in scan_doc_ids(reader)? {
            if let Ok(Some(doc)) = reader.document(doc_id)
                && let Some(val) = doc.get(&self.field)
                && self.value_matches_any(val)
            {
                matching_docs.push(doc_id);
            }
        }

        Ok(Box::new(PreComputedMatcher::new(matching_docs)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        let total_docs = reader.doc_count();
        if total_docs == 0 {
            return Ok(Box::new(BM25Scorer::new(0, 0, 0, 1.0, 1, self.boost)));
        }

        // Constant scoring needs no document statistics; counting
        // matches here would cost an O(total_docs) stored-document scan
        // per segment under the per-segment fanout (#994).
        Ok(Box::new(RangeScorer::new(
            self.min_f64(),
            self.max_f64(),
            0,
            total_docs,
            self.boost,
        )))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    /// Reject a bad boost, as every query does, and a `NaN` bound
    /// (Issue #1601).
    ///
    /// A `NaN` bound compares false with every value, so the range would
    /// match nothing without an error; an `Integer` range would even search
    /// it as 0. It is rejected here, where a query is received, so it fails
    /// the same way from the Rust API, a binding, a filter or a composite
    /// query.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] for a boost that is not
    /// finite or is negative, or for a `NaN` bound.
    fn validate_boosts(&self) -> Result<()> {
        crate::lexical::query::check_query_boost(self.boost, || self.description())?;
        if self.has_nan_bound {
            return Err(LaurusError::invalid_argument(format!(
                "NumericRangeQuery on field '{}' has a NaN bound; a bound must be a \
                 number, or None to leave that side open",
                self.field
            )));
        }
        Ok(())
    }

    fn description(&self) -> String {
        format!(
            "NumericRangeQuery(field:{}, type:{:?}, lower:{:?}, upper:{:?})",
            self.field, self.numeric_type, self.lower_bound, self.upper_bound
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(false)
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        Ok(100) // Numeric range queries are efficient
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field + numeric type + both byte-encoded bounds + inclusivity flags
        // determine the matched set; boost is score-only and excluded.
        Some(format!(
            "nrange|{:?}|{:?}|{:?}|{:?}|{}|{}",
            self.field,
            self.numeric_type,
            self.lower_bound,
            self.upper_bound,
            self.lower_inclusive,
            self.upper_inclusive
        ))
    }
}

/// Range query over a `DateTime` field (Issue #1179).
///
/// Bounds are kept as [`DateTime<Utc>`] and encoded with the same
/// [`datetime_to_point`] the writer uses for the field's one-dimensional
/// BKD point, so a bound and a stored value denoting the same instant
/// compare equal down to the persisted microsecond. Matching and scoring
/// delegate to [`NumericRangeQuery`] with [`NumericType::Float`]
/// (`Integer` would truncate sub-second bounds): the BKD tree is used when
/// the field is indexed, the stored-document fallback otherwise, and every
/// hit gets the same constant score like any other range query.
///
/// The query DSL builds this for bounds that parse as datetime literals
/// (see [`DateTimeRangeQuery::from_literals`]); the language bindings use
/// the same constructor.
#[derive(Debug, Clone)]
pub struct DateTimeRangeQuery {
    /// The field to search in.
    field: String,
    /// Lower bound, if any.
    lower_bound: Option<DateTime<Utc>>,
    /// Upper bound, if any.
    upper_bound: Option<DateTime<Utc>>,
    /// Whether lower bound is inclusive.
    lower_inclusive: bool,
    /// Whether upper bound is inclusive.
    upper_inclusive: bool,
    /// The boost factor for this query.
    boost: f32,
}

impl DateTimeRangeQuery {
    /// Create a new datetime range query.
    pub fn new<S: Into<String>>(
        field: S,
        lower: Option<DateTime<Utc>>,
        upper: Option<DateTime<Utc>>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Self {
        DateTimeRangeQuery {
            field: field.into(),
            lower_bound: lower,
            upper_bound: upper,
            lower_inclusive,
            upper_inclusive,
            boost: 1.0,
        }
    }

    /// Create a datetime range query from textual bounds.
    ///
    /// Each bound is parsed with the query-literal grammar shared with the
    /// query DSL: RFC 3339 (normalized to UTC), naive
    /// `YYYY-MM-DDTHH:MM:SS[.fff]` (UTC), or `YYYY-MM-DD` (midnight UTC).
    /// `None` leaves that side open.
    ///
    /// # Errors
    ///
    /// Returns a query error ([`LaurusError::parse`]) when a bound does not
    /// match any accepted form.
    pub fn from_literals<S: Into<String>>(
        field: S,
        lower: Option<&str>,
        upper: Option<&str>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Result<Self> {
        let parse = |bound: Option<&str>| -> Result<Option<DateTime<Utc>>> {
            match bound {
                None => Ok(None),
                Some(s) => parse_datetime_literal(s).map(Some).ok_or_else(|| {
                    LaurusError::parse(format!(
                        "invalid datetime bound {s:?}: expected RFC 3339 \
                         (e.g. 2024-01-01T00:00:00Z), YYYY-MM-DDTHH:MM:SS[.fff], \
                         or YYYY-MM-DD"
                    ))
                }),
            }
        };
        Ok(Self::new(
            field,
            parse(lower)?,
            parse(upper)?,
            lower_inclusive,
            upper_inclusive,
        ))
    }

    /// Create a datetime range with both bounds inclusive.
    pub fn between<S: Into<String>>(field: S, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self::new(field, Some(start), Some(end), true, true)
    }

    /// Create a query for dates after the given datetime.
    pub fn after<S: Into<String>>(field: S, datetime: DateTime<Utc>) -> Self {
        Self::new(field, Some(datetime), None, false, true)
    }

    /// Create a query for dates on or after the given datetime.
    pub fn on_or_after<S: Into<String>>(field: S, datetime: DateTime<Utc>) -> Self {
        Self::new(field, Some(datetime), None, true, true)
    }

    /// Create a query for dates before the given datetime.
    pub fn before<S: Into<String>>(field: S, datetime: DateTime<Utc>) -> Self {
        Self::new(field, None, Some(datetime), true, false)
    }

    /// Create a query for dates on or before the given datetime.
    pub fn on_or_before<S: Into<String>>(field: S, datetime: DateTime<Utc>) -> Self {
        Self::new(field, None, Some(datetime), true, true)
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the lower bound, if any.
    pub fn lower_bound(&self) -> Option<DateTime<Utc>> {
        self.lower_bound
    }

    /// Get the upper bound, if any.
    pub fn upper_bound(&self) -> Option<DateTime<Utc>> {
        self.upper_bound
    }

    /// Whether the lower bound is inclusive.
    pub fn lower_inclusive(&self) -> bool {
        self.lower_inclusive
    }

    /// Whether the upper bound is inclusive.
    pub fn upper_inclusive(&self) -> bool {
        self.upper_inclusive
    }

    /// The lower bound in BKD point space (see [`datetime_to_point`]).
    pub fn min_point(&self) -> Option<f64> {
        self.lower_bound.as_ref().map(datetime_to_point)
    }

    /// The upper bound in BKD point space (see [`datetime_to_point`]).
    pub fn max_point(&self) -> Option<f64> {
        self.upper_bound.as_ref().map(datetime_to_point)
    }

    /// The equivalent numeric range query over the field's BKD point.
    fn as_numeric(&self) -> NumericRangeQuery {
        NumericRangeQuery::new(
            self.field.clone(),
            NumericType::Float,
            self.min_point(),
            self.max_point(),
            self.lower_inclusive,
            self.upper_inclusive,
        )
        .with_boost(self.boost)
    }

    /// Check if a BKD point value falls within the range.
    ///
    /// Uses the per-point check of the BKD tree's
    /// [`RangeQueryVisitor`](crate::lexical::index::structures::visitor::RangeQueryVisitor),
    /// the same one the stored-document scan of [`Query::matcher`] uses, so
    /// this agrees with the documents the query matches (Issue #1600).
    pub fn contains_point(&self, point: f64) -> bool {
        value_in_range(
            point,
            self.min_point().unwrap_or(f64::NEG_INFINITY),
            self.max_point().unwrap_or(f64::INFINITY),
            self.lower_inclusive,
            self.upper_inclusive,
        )
    }

    /// Check if a whole-second Unix timestamp falls within the range.
    pub fn contains_timestamp(&self, timestamp: i64) -> bool {
        self.contains_point(timestamp as f64)
    }

    /// Check if a datetime falls within the range.
    pub fn contains_datetime(&self, datetime: &DateTime<Utc>) -> bool {
        self.contains_point(datetime_to_point(datetime))
    }
}

impl Query for DateTimeRangeQuery {
    fn field(&self) -> Option<&str> {
        Some(&self.field)
    }

    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        // BKD tree when the field is indexed, stored-document fallback
        // (which understands `DataValue::DateTime`) otherwise.
        self.as_numeric().matcher(reader)
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        // Constant scoring, like every other range query.
        self.as_numeric().scorer(reader)
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn description(&self) -> String {
        format!(
            "DateTimeRangeQuery(field:{}, lower:{}{:?}, upper:{:?}{})",
            self.field,
            if self.lower_inclusive { "[" } else { "{" },
            self.lower_bound.map(|dt| dt.to_rfc3339()),
            self.upper_bound.map(|dt| dt.to_rfc3339()),
            if self.upper_inclusive { "]" } else { "}" },
        )
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(false)
    }

    fn cost(&self, _reader: &dyn LexicalIndexReader) -> Result<u64> {
        Ok(100) // DateTime range queries are efficient
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn cache_key(&self) -> Option<String> {
        // Field + both bounds (exact micros) + inclusivity flags determine
        // the matched set; boost is score-only and excluded.
        Some(format!(
            "dtrange|{:?}|{:?}|{:?}|{}|{}",
            self.field,
            self.lower_bound.map(|dt| dt.timestamp_micros()),
            self.upper_bound.map(|dt| dt.timestamp_micros()),
            self.lower_inclusive,
            self.upper_inclusive
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bound_contains_lower() {
        let bound = Bound::Included("hello".to_string());
        assert!(bound.contains_lower(&"hello".to_string()));
        assert!(bound.contains_lower(&"world".to_string()));
        assert!(!bound.contains_lower(&"apple".to_string()));

        let bound = Bound::Excluded("hello".to_string());
        assert!(!bound.contains_lower(&"hello".to_string()));
        assert!(bound.contains_lower(&"world".to_string()));
        assert!(!bound.contains_lower(&"apple".to_string()));

        let bound: Bound<String> = Bound::Unbounded;
        assert!(bound.contains_lower(&"anything".to_string()));
    }

    #[test]
    fn test_bound_contains_upper() {
        let bound = Bound::Included("hello".to_string());
        assert!(bound.contains_upper(&"hello".to_string()));
        assert!(!bound.contains_upper(&"world".to_string()));
        assert!(bound.contains_upper(&"apple".to_string()));

        let bound = Bound::Excluded("hello".to_string());
        assert!(!bound.contains_upper(&"hello".to_string()));
        assert!(!bound.contains_upper(&"world".to_string()));
        assert!(bound.contains_upper(&"apple".to_string()));

        let bound: Bound<String> = Bound::Unbounded;
        assert!(bound.contains_upper(&"anything".to_string()));
    }

    #[test]
    fn test_range_query_creation() {
        let query = RangeQuery::new(
            "field",
            Some("apple".to_string()),
            Some("zebra".to_string()),
        );

        assert_eq!(query.field(), "field");
        assert_eq!(*query.lower_bound(), Bound::Included("apple".to_string()));
        assert_eq!(*query.upper_bound(), Bound::Included("zebra".to_string()));
        assert_eq!(query.boost(), 1.0);
    }

    #[test]
    fn test_range_query_contains() {
        let query = RangeQuery::new(
            "field",
            Some("apple".to_string()),
            Some("zebra".to_string()),
        );

        assert!(query.contains("apple"));
        assert!(query.contains("hello"));
        assert!(query.contains("zebra"));
        assert!(!query.contains("aardvark"));
        assert!(!query.contains("zoo"));
    }

    #[test]
    fn test_range_query_greater_than() {
        let query = RangeQuery::greater_than("field", "hello".to_string());

        assert!(!query.contains("hello"));
        assert!(query.contains("world"));
        assert!(!query.contains("apple"));
    }

    #[test]
    fn test_range_query_less_than_or_equal() {
        let query = RangeQuery::less_than_or_equal("field", "hello".to_string());

        assert!(query.contains("hello"));
        assert!(!query.contains("world"));
        assert!(query.contains("apple"));
    }

    #[test]
    fn test_range_query_with_boost() {
        let query = RangeQuery::new("field", None, None).with_boost(2.5);

        assert_eq!(query.boost(), 2.5);
    }

    #[test]
    fn test_numeric_range_query_creation() {
        let query = NumericRangeQuery::i64_range("price", Some(100), Some(200));

        assert_eq!(query.field(), "price");
        assert_eq!(query.numeric_type(), NumericType::Integer);
        assert!(query.contains_numeric(150.0));
        assert!(!query.contains_numeric(50.0));
        assert!(!query.contains_numeric(250.0));
    }

    #[test]
    fn test_numeric_range_query_f64() {
        let query = NumericRangeQuery::f64_range("price", Some(99.99), Some(199.99));

        assert!(query.contains_numeric(150.0));
        assert!(query.contains_numeric(99.99));
        assert!(query.contains_numeric(199.99));
        assert!(!query.contains_numeric(99.98));
        assert!(!query.contains_numeric(200.0));
    }

    #[test]
    fn test_numeric_range_query_greater_than() {
        let query = NumericRangeQuery::greater_than("rating", NumericType::Integer, 3.0);

        assert!(query.contains_numeric(4.0));
        assert!(query.contains_numeric(5.0));
        assert!(!query.contains_numeric(3.0));
        assert!(!query.contains_numeric(2.0));
    }

    #[test]
    fn test_numeric_range_query_less_than_or_equal() {
        let query = NumericRangeQuery::less_than_or_equal("rating", NumericType::Integer, 3.0);

        assert!(query.contains_numeric(3.0));
        assert!(query.contains_numeric(2.0));
        assert!(query.contains_numeric(1.0));
        assert!(!query.contains_numeric(4.0));
    }

    #[test]
    fn test_datetime_range_query_creation() {
        use chrono::{TimeZone, Utc};

        let start = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2023, 12, 31, 23, 59, 59).unwrap();

        let query = DateTimeRangeQuery::between("created_at", start, end);

        assert_eq!(query.field(), "created_at");
        assert!(query.contains_datetime(&Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 0).unwrap()));
        assert!(query.contains_datetime(&start));
        assert!(query.contains_datetime(&end));
        assert!(!query.contains_datetime(&Utc.with_ymd_and_hms(2022, 12, 31, 23, 59, 59).unwrap()));
        assert!(!query.contains_datetime(&Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()));
    }

    #[test]
    fn test_datetime_range_query_after() {
        use chrono::{TimeZone, Utc};

        let threshold = Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 0).unwrap();
        let query = DateTimeRangeQuery::after("updated_at", threshold);

        assert!(query.contains_datetime(&Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 1).unwrap()));
        assert!(query.contains_datetime(&Utc.with_ymd_and_hms(2023, 12, 31, 23, 59, 59).unwrap()));
        assert!(!query.contains_datetime(&threshold));
        assert!(!query.contains_datetime(&Utc.with_ymd_and_hms(2023, 6, 15, 11, 59, 59).unwrap()));
    }

    #[test]
    fn test_datetime_range_query_on_or_before() {
        use chrono::{TimeZone, Utc};

        let threshold = Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 0).unwrap();
        let query = DateTimeRangeQuery::on_or_before("updated_at", threshold);

        assert!(query.contains_datetime(&threshold));
        assert!(query.contains_datetime(&Utc.with_ymd_and_hms(2023, 6, 15, 11, 59, 59).unwrap()));
        assert!(query.contains_datetime(&Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()));
        assert!(!query.contains_datetime(&Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 1).unwrap()));
    }

    /// #1179: bounds keep their sub-second part, so two instants inside the
    /// same second are distinguishable.
    #[test]
    fn datetime_range_sub_second_bounds() {
        use chrono::{TimeZone, Utc};

        let base = 1_700_000_000;
        let lower = Utc.timestamp_opt(base, 400_000_000).unwrap();
        let upper = Utc.timestamp_opt(base, 600_000_000).unwrap();
        let query = DateTimeRangeQuery::between("created_at", lower, upper);

        assert!(query.contains_datetime(&Utc.timestamp_opt(base, 500_000_000).unwrap()));
        assert!(query.contains_datetime(&lower));
        assert!(query.contains_datetime(&upper));
        assert!(!query.contains_datetime(&Utc.timestamp_opt(base, 250_000_000).unwrap()));
        assert!(!query.contains_datetime(&Utc.timestamp_opt(base, 750_000_000).unwrap()));
        // Point space carries the fraction too.
        assert_eq!(query.min_point(), Some(1_700_000_000.4));
        assert_eq!(query.max_point(), Some(1_700_000_000.6));
        // Exclusive edges are exclusive at micro-second precision.
        let exclusive =
            DateTimeRangeQuery::new("created_at", Some(lower), Some(upper), false, false);
        assert!(!exclusive.contains_datetime(&lower));
        assert!(!exclusive.contains_datetime(&upper));
        assert!(exclusive.contains_datetime(&Utc.timestamp_opt(base, 400_001_000).unwrap()));
    }

    #[test]
    fn from_literals_accepts_all_literal_forms() {
        use chrono::{TimeZone, Utc};

        let q = DateTimeRangeQuery::from_literals(
            "created_at",
            Some("2024-01-01"),
            Some("2024-12-31T00:00:00Z"),
            true,
            false,
        )
        .unwrap();
        assert_eq!(
            q.lower_bound(),
            Some(Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap())
        );
        assert_eq!(
            q.upper_bound(),
            Some(Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap())
        );
        assert!(q.lower_inclusive());
        assert!(!q.upper_inclusive());

        let q = DateTimeRangeQuery::from_literals(
            "created_at",
            None,
            Some("2024-01-01T09:00:00+09:00"),
            true,
            true,
        )
        .unwrap();
        assert_eq!(q.lower_bound(), None);
        assert_eq!(
            q.upper_bound(),
            Some(Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap())
        );

        let q = DateTimeRangeQuery::from_literals(
            "created_at",
            Some("2024-06-15T12:34:56"),
            None,
            true,
            true,
        )
        .unwrap();
        assert_eq!(
            q.lower_bound(),
            Some(Utc.with_ymd_and_hms(2024, 6, 15, 12, 34, 56).unwrap())
        );
    }

    #[test]
    fn from_literals_rejects_garbage() {
        for bad in ["yesterday", "1700000000", "*", "", "2024-13-01"] {
            let err = DateTimeRangeQuery::from_literals("created_at", Some(bad), None, true, true)
                .unwrap_err();
            assert!(
                err.to_string().contains("invalid datetime bound"),
                "{bad:?}: {err}"
            );
        }
    }

    /// Reader whose stored documents carry a `created_at` DateTime and no
    /// BKD tree, exercising the stored-document fallback (#1179).
    #[derive(Debug)]
    struct DateTimeReader {
        docs: Vec<(u64, chrono::DateTime<chrono::Utc>)>,
    }

    impl LexicalIndexReader for DateTimeReader {
        fn doc_count(&self) -> u64 {
            self.docs.len() as u64
        }
        fn max_doc(&self) -> u64 {
            self.docs.iter().map(|(id, _)| id + 1).max().unwrap_or(0)
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(
            &self,
            doc_id: u64,
        ) -> crate::error::Result<Option<crate::lexical::core::document::Document>> {
            Ok(self
                .docs
                .iter()
                .find(|(id, _)| *id == doc_id)
                .map(|(_, dt)| {
                    crate::data::Document::builder()
                        .add_field("created_at", crate::data::DataValue::DateTime(*dt))
                        .build()
                }))
        }
        fn doc_ids(&self) -> crate::error::Result<Vec<u64>> {
            Ok(self.docs.iter().map(|(id, _)| *id).collect())
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>>
        {
            Ok(None)
        }
        fn field_stats(
            &self,
            _field: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> crate::error::Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn drain(mut matcher: Box<dyn Matcher>) -> Vec<u64> {
        let mut ids = Vec::new();
        while !matcher.is_exhausted() {
            ids.push(matcher.doc_id());
            if !matcher.next().unwrap() {
                break;
            }
        }
        ids
    }

    fn datetime_reader() -> DateTimeReader {
        use chrono::{TimeZone, Utc};
        DateTimeReader {
            docs: vec![
                (1, Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()),
                (2, Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap()),
                (3, Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap()),
                (4, Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap()),
            ],
        }
    }

    /// #1179: the numeric fallback understands stored `DateTime` values
    /// (it used to fall into `_ => false` and match nothing).
    #[test]
    fn matcher_fallback_matches_stored_datetime_values() {
        let reader = datetime_reader();
        let june = datetime_to_point(&reader.docs[1].1);
        let query = NumericRangeQuery::new(
            "created_at",
            NumericType::Float,
            Some(june),
            None,
            true,
            true,
        );
        assert_eq!(drain(query.matcher(&reader).unwrap()), vec![2, 3, 4]);
        assert_eq!(query.count_matching_documents(&reader).unwrap(), 3);
    }

    /// #1179: `DateTimeRangeQuery::matcher` used to error unconditionally;
    /// it now resolves through the same fallback as numeric ranges.
    #[test]
    fn datetime_range_query_matcher_uses_stored_fallback() {
        use chrono::{TimeZone, Utc};
        let reader = datetime_reader();
        let query = DateTimeRangeQuery::between(
            "created_at",
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap(),
        );
        assert_eq!(drain(query.matcher(&reader).unwrap()), vec![1, 2, 3]);
        let exclusive = DateTimeRangeQuery::new(
            "created_at",
            Some(Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()),
            Some(Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap()),
            false,
            false,
        );
        assert_eq!(drain(exclusive.matcher(&reader).unwrap()), vec![2]);
        // Constant scoring, like every other range query.
        let scorer = query.scorer(&reader).unwrap();
        assert!(scorer.as_any().downcast_ref::<RangeScorer>().is_some());
    }

    /// Reader whose stored documents carry one value of any type in the
    /// field `v` and no BKD tree, so a range query takes the
    /// stored-document scan (Issue #1600).
    #[derive(Debug)]
    struct StoredValueReader {
        docs: Vec<(u64, crate::data::DataValue)>,
    }

    impl LexicalIndexReader for StoredValueReader {
        fn doc_count(&self) -> u64 {
            self.docs.len() as u64
        }
        fn max_doc(&self) -> u64 {
            self.docs.iter().map(|(id, _)| id + 1).max().unwrap_or(0)
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(
            &self,
            doc_id: u64,
        ) -> crate::error::Result<Option<crate::lexical::core::document::Document>> {
            Ok(self
                .docs
                .iter()
                .find(|(id, _)| *id == doc_id)
                .map(|(_, value)| {
                    crate::data::Document::builder()
                        .add_field("v", value.clone())
                        .build()
                }))
        }
        fn doc_ids(&self) -> crate::error::Result<Vec<u64>> {
            Ok(self.docs.iter().map(|(id, _)| *id).collect())
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>>
        {
            Ok(None)
        }
        fn field_stats(
            &self,
            _field: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> crate::error::Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// A reader holding `values` as documents 1, 2, 3, ... in order.
    fn stored_values(values: Vec<crate::data::DataValue>) -> StoredValueReader {
        StoredValueReader {
            docs: (1u64..).zip(values).collect(),
        }
    }

    /// The `(name, query, documents)` cases whose scan over `reader` does
    /// not match exactly `documents`, described for an assertion message.
    fn scan_mismatches(
        reader: &StoredValueReader,
        cases: Vec<(&str, Box<dyn Query>, Vec<u64>)>,
    ) -> Vec<String> {
        cases
            .into_iter()
            .filter_map(|(name, query, want)| {
                let got = drain(query.matcher(reader).unwrap());
                (got != want).then(|| format!("{name}: got {got:?}, want {want:?}"))
            })
            .collect()
    }

    /// A `Float` range over `v` with the given bounds and inclusivity.
    fn float_range(
        lower: Option<f64>,
        upper: Option<f64>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Box<dyn Query> {
        Box::new(NumericRangeQuery::new(
            "v",
            NumericType::Float,
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
        ))
    }

    /// Documents 1 to 6 hold -5.0, -1.5, -0.0, 0.0, 2.5 and `NaN`.
    fn float_reader() -> StoredValueReader {
        use crate::data::DataValue::Float64;
        stored_values(vec![
            Float64(-5.0),
            Float64(-1.5),
            Float64(-0.0),
            Float64(0.0),
            Float64(2.5),
            Float64(f64::NAN),
        ])
    }

    /// Issue #1600: the scan compares `Float` values as numbers, as the BKD
    /// tree does. It used to compare the big-endian bytes, which put every
    /// negative number above every non-negative one, in reverse order. A
    /// `NaN` value is in no range, and `-0.0` equals `0.0`.
    #[test]
    fn scan_compares_float_values_as_numbers() {
        let reader = float_reader();
        let wrong = scan_mismatches(
            &reader,
            vec![
                (
                    "[-5, 5]",
                    float_range(Some(-5.0), Some(5.0), true, true),
                    vec![1, 2, 3, 4, 5],
                ),
                (
                    "[-3, 0]",
                    float_range(Some(-3.0), Some(0.0), true, true),
                    vec![2, 3, 4],
                ),
                (
                    "(-5, -1.5]",
                    float_range(Some(-5.0), Some(-1.5), false, true),
                    vec![2],
                ),
                (
                    "[-1.5, 0)",
                    float_range(Some(-1.5), Some(0.0), true, false),
                    vec![2],
                ),
                (
                    "[*, -2]",
                    float_range(None, Some(-2.0), true, true),
                    vec![1],
                ),
                (
                    "[-1, *]",
                    float_range(Some(-1.0), None, true, true),
                    vec![3, 4, 5],
                ),
                (
                    "[0, 0]",
                    float_range(Some(0.0), Some(0.0), true, true),
                    vec![3, 4],
                ),
                (
                    "[*, *]",
                    float_range(None, None, true, true),
                    vec![1, 2, 3, 4, 5],
                ),
            ],
        );
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1600: an `Integer` range, which the bindings build with
    /// `i64_range`, compares the values as numbers too. Its bytes used to
    /// put every negative value above every non-negative one. The value is
    /// not truncated, as the BKD tree does not truncate its point, so 2.5 is
    /// not in `[2, 2]`.
    #[test]
    fn scan_compares_integer_ranges_as_numbers() {
        use crate::data::DataValue::{Float64, Int64};
        let reader = stored_values(vec![Int64(-5), Int64(-1), Int64(0), Int64(3), Float64(2.5)]);
        let range = |lower: Option<i64>, upper: Option<i64>| -> Box<dyn Query> {
            Box::new(NumericRangeQuery::i64_range("v", lower, upper))
        };
        let wrong = scan_mismatches(
            &reader,
            vec![
                ("[-5, 5]", range(Some(-5), Some(5)), vec![1, 2, 3, 4, 5]),
                ("[-3, 0]", range(Some(-3), Some(0)), vec![2, 3]),
                ("[*, -2]", range(None, Some(-2)), vec![1]),
                ("[2, 2]", range(Some(2), Some(2)), vec![]),
                ("[2, 3]", range(Some(2), Some(3)), vec![4, 5]),
                (
                    "(-1, *]",
                    Box::new(NumericRangeQuery::greater_than(
                        "v",
                        NumericType::Integer,
                        -1.0,
                    )) as Box<dyn Query>,
                    vec![3, 4, 5],
                ),
            ],
        );
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1600: numbers parsed from `Text` and `TextArray` values, and
    /// the elements of `Float64Array` and `Int64Array` values, compare as
    /// numbers too.
    #[test]
    fn scan_compares_text_and_array_values_as_numbers() {
        use crate::data::DataValue::{Float64Array, Int64Array, Text, TextArray};
        let reader = stored_values(vec![
            Text("-1".to_string()),
            TextArray(vec!["-7".to_string(), "9".to_string()]),
            Float64Array(vec![-7.0, 9.0]),
            Int64Array(vec![-7, 9]),
            Text("3".to_string()),
        ]);
        let wrong = scan_mismatches(
            &reader,
            vec![
                (
                    "[-8, -6]",
                    float_range(Some(-8.0), Some(-6.0), true, true),
                    vec![2, 3, 4],
                ),
                (
                    "[-2, 0]",
                    float_range(Some(-2.0), Some(0.0), true, true),
                    vec![1],
                ),
                (
                    "[-1, -1]",
                    float_range(Some(-1.0), Some(-1.0), true, true),
                    vec![1],
                ),
                (
                    "[0, 5]",
                    float_range(Some(0.0), Some(5.0), true, true),
                    vec![5],
                ),
            ],
        );
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1600: a `DateTimeRangeQuery` over instants before 1970, whose
    /// BKD points are negative, scans like a numeric range, and
    /// `contains_datetime` agrees with the documents the matcher returns.
    #[test]
    fn datetime_scan_matches_instants_before_1970() {
        use crate::data::DataValue::{DateTime, DateTimeArray};
        use chrono::{TimeZone, Utc};
        let at = |y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32| {
            Utc.with_ymd_and_hms(y, mo, d, h, mi, s).unwrap()
        };
        let singles = [
            at(1960, 6, 1, 0, 0, 0),
            at(1969, 12, 31, 23, 59, 59),
            at(1970, 1, 1, 0, 0, 0),
            at(1975, 1, 1, 0, 0, 0),
        ];
        let mut values: Vec<_> = singles.iter().map(|dt| DateTime(*dt)).collect();
        values.push(DateTimeArray(vec![
            at(1950, 1, 1, 0, 0, 0),
            at(2000, 1, 1, 0, 0, 0),
        ]));
        let reader = stored_values(values);

        let queries = [
            (
                "[1960-01-01, 1980-01-01]",
                DateTimeRangeQuery::between("v", at(1960, 1, 1, 0, 0, 0), at(1980, 1, 1, 0, 0, 0)),
                vec![1, 2, 3, 4],
            ),
            (
                "[1960-01-01, 1969-12-31T23:59:59]",
                DateTimeRangeQuery::between(
                    "v",
                    at(1960, 1, 1, 0, 0, 0),
                    at(1969, 12, 31, 23, 59, 59),
                ),
                vec![1, 2],
            ),
            (
                "[*, 1965-01-01]",
                DateTimeRangeQuery::on_or_before("v", at(1965, 1, 1, 0, 0, 0)),
                // Document 5 holds 1950, so it matches: any element does.
                vec![1, 5],
            ),
            (
                "[1949-01-01, 1951-01-01]",
                DateTimeRangeQuery::between("v", at(1949, 1, 1, 0, 0, 0), at(1951, 1, 1, 0, 0, 0)),
                vec![5],
            ),
            (
                "(1969-12-31T23:59:59, *]",
                DateTimeRangeQuery::after("v", at(1969, 12, 31, 23, 59, 59)),
                vec![3, 4, 5],
            ),
        ];
        let mut wrong = Vec::new();
        for (name, query, want) in queries {
            let got = drain(query.matcher(&reader).unwrap());
            if got != want {
                wrong.push(format!("{name}: got {got:?}, want {want:?}"));
            }
            for (id, dt) in (1u64..).zip(singles.iter()) {
                if query.contains_datetime(dt) != got.contains(&id) {
                    wrong.push(format!(
                        "{name}: contains_datetime({dt}) disagrees with the matcher"
                    ));
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1600: `count_matching_documents` shares the scan's
    /// comparison, so it counts negative values in the range too.
    #[test]
    fn count_matching_documents_agrees_with_the_scan() {
        let reader = float_reader();
        let mut wrong = Vec::new();
        for (name, query, want) in [
            (
                "[-5, 5]",
                NumericRangeQuery::f64_range("v", Some(-5.0), Some(5.0)),
                5,
            ),
            (
                "[-3, 0]",
                NumericRangeQuery::f64_range("v", Some(-3.0), Some(0.0)),
                3,
            ),
            (
                "[*, -2]",
                NumericRangeQuery::f64_range("v", None, Some(-2.0)),
                1,
            ),
        ] {
            let count = query.count_matching_documents(&reader).unwrap();
            let matched = drain(query.matcher(&reader).unwrap()).len() as u64;
            if count != want || matched != want {
                wrong.push(format!(
                    "{name}: count {count}, matcher {matched}, want {want}"
                ));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1600: `contains_encoded` decodes the bytes for the query's
    /// type and compares numbers, so negative values are in order; bytes
    /// that are not 8 long are in no range.
    #[test]
    fn contains_encoded_compares_decoded_numbers() {
        let float = NumericRangeQuery::f64_range("v", Some(-5.0), Some(-1.0));
        let integer = NumericRangeQuery::i64_range("v", Some(-5), Some(-1));
        let mut wrong = Vec::new();
        for (query, values, want) in [
            (
                &float,
                vec![-6.0, -5.0, -3.5, -1.0, -0.5, 0.0, 2.0],
                vec![false, true, true, true, false, false, false],
            ),
            (
                &integer,
                vec![-6.0, -5.0, -3.0, -1.0, 0.0, 2.0],
                vec![false, true, true, true, false, false],
            ),
        ] {
            for (value, want) in values.into_iter().zip(want) {
                let encoded = NumericRangeQuery::encode_numeric(value, query.numeric_type());
                let got = query.contains_encoded(&encoded);
                if got != want || got != query.contains_numeric(value) {
                    wrong.push(format!(
                        "{:?} {value}: contains_encoded {got}, contains_numeric {}, want {want}",
                        query.numeric_type(),
                        query.contains_numeric(value)
                    ));
                }
            }
        }
        if float.contains_encoded(&[0u8; 4]) {
            wrong.push("4 bytes are in the range".to_string());
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    #[test]
    fn test_query_costs() {
        let string_range = RangeQuery::new("field", Some("a".to_string()), Some("z".to_string()));
        let numeric_range = NumericRangeQuery::i64_range("price", Some(0), Some(100));
        let datetime_range =
            DateTimeRangeQuery::between("created_at", chrono::Utc::now(), chrono::Utc::now());

        // These are placeholder implementations, but we can test the cost method exists
        assert!(string_range.cost(&EmptyReader).unwrap() > 0);
        assert!(numeric_range.cost(&EmptyReader).unwrap() > 0);
        assert!(datetime_range.cost(&EmptyReader).unwrap() > 0);

        // Numeric and datetime queries should be more efficient
        assert!(
            numeric_range.cost(&EmptyReader).unwrap() < string_range.cost(&EmptyReader).unwrap()
        );
        assert!(
            datetime_range.cost(&EmptyReader).unwrap() < string_range.cost(&EmptyReader).unwrap()
        );
    }

    // Placeholder reader for testing
    #[derive(Debug)]
    struct EmptyReader;

    impl LexicalIndexReader for EmptyReader {
        fn doc_count(&self) -> u64 {
            0
        }
        fn max_doc(&self) -> u64 {
            0
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(
            &self,
            _doc_id: u64,
        ) -> crate::error::Result<Option<crate::lexical::core::document::Document>> {
            Ok(None)
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>>
        {
            Ok(None)
        }
        fn field_stats(
            &self,
            _field: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> crate::error::Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Reader with a configurable stored-doc id set that counts
    /// `document()` probes. `max_doc()` deliberately disagrees with the
    /// id set to model sparse id spaces (post-merge segments, per-segment
    /// fanout views reporting the global `max_doc`) — see #994.
    #[derive(Debug)]
    struct CountingReader {
        ids: Vec<u64>,
        max_doc: u64,
        document_calls: std::sync::atomic::AtomicU64,
    }

    impl CountingReader {
        fn new(ids: Vec<u64>, max_doc: u64) -> Self {
            CountingReader {
                ids,
                max_doc,
                document_calls: std::sync::atomic::AtomicU64::new(0),
            }
        }

        fn calls(&self) -> u64 {
            self.document_calls
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl LexicalIndexReader for CountingReader {
        fn doc_count(&self) -> u64 {
            self.max_doc
        }
        fn max_doc(&self) -> u64 {
            self.max_doc
        }
        fn is_deleted(&self, _doc_id: u64) -> bool {
            false
        }
        fn document(
            &self,
            doc_id: u64,
        ) -> crate::error::Result<Option<crate::lexical::core::document::Document>> {
            self.document_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if self.ids.contains(&doc_id) {
                let doc = crate::data::Document::builder()
                    .add_integer("price", doc_id as i64)
                    .build();
                Ok(Some(doc))
            } else {
                Ok(None)
            }
        }
        fn doc_ids(&self) -> crate::error::Result<Vec<u64>> {
            Ok(self.ids.clone())
        }
        fn term_info(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::ReaderTermInfo>> {
            Ok(None)
        }
        fn postings(
            &self,
            _field: &str,
            _term: &str,
        ) -> crate::error::Result<Option<Box<dyn crate::lexical::reader::PostingIterator>>>
        {
            Ok(None)
        }
        fn field_stats(
            &self,
            _field: &str,
        ) -> crate::error::Result<Option<crate::lexical::reader::FieldStats>> {
            Ok(None)
        }
        fn close(&mut self) -> crate::error::Result<()> {
            Ok(())
        }
        fn is_closed(&self) -> bool {
            false
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// #994 regression: building the scorer must not scan stored
    /// documents — range queries are constant-scored and need no
    /// matching-document count.
    #[test]
    fn scorer_does_not_scan_stored_documents() {
        let reader = CountingReader::new(Vec::new(), 100);
        let query = NumericRangeQuery::i64_range("price", Some(0), Some(1000));

        let scorer = query.scorer(&reader).unwrap();

        assert_eq!(
            reader.calls(),
            0,
            "scorer() must not probe stored documents"
        );
        assert_eq!(scorer.max_score(), query.boost());
        assert_eq!(scorer.score(0, 1.0, None), query.boost());
    }

    /// #994 regression: the BKD-less matcher fallback must probe only
    /// the doc ids actually present in the reader — not the dense
    /// `0..max_doc()` range, which both over-scans (fanout-global
    /// `max_doc`) and under-scans (ids above `max_doc` in sparse
    /// post-merge segments).
    #[test]
    fn matcher_fallback_scans_only_present_doc_ids() {
        let reader = CountingReader::new(vec![5, 1500], 10);
        let query = NumericRangeQuery::i64_range("price", Some(0), Some(2000));

        let mut matcher = query.matcher(&reader).unwrap();
        let mut found = Vec::new();
        while !matcher.is_exhausted() {
            let doc_id = matcher.doc_id();
            if doc_id == u64::MAX {
                break;
            }
            found.push(doc_id);
            if !matcher.next().unwrap() {
                break;
            }
        }

        assert_eq!(
            found,
            vec![5, 1500],
            "id above max_doc() must not be missed"
        );
        assert_eq!(
            reader.calls(),
            2,
            "fallback must probe only the present ids"
        );
    }

    /// Issue #1601: a `NaN` bound, lower, upper or both, is rejected by
    /// `validate_boosts` for `Float` and for `Integer`, whose encoded
    /// bounds turn `NaN` into 0; the error names the field. Finite,
    /// infinite and open bounds pass.
    #[test]
    fn validate_boosts_rejects_a_nan_bound() {
        let nan = f64::NAN;
        let inf = f64::INFINITY;
        let mut wrong = Vec::new();
        for numeric_type in [NumericType::Float, NumericType::Integer] {
            for (lower, upper, rejected) in [
                (Some(nan), Some(5.0), true),
                (None, Some(nan), true),
                (Some(nan), None, true),
                (Some(nan), Some(nan), true),
                (Some(-5.0), Some(5.0), false),
                (Some(-inf), Some(inf), false),
                (Some(inf), None, false),
                (None, Some(-inf), false),
                (None, None, false),
            ] {
                let query = NumericRangeQuery::new("price", numeric_type, lower, upper, true, true);
                let result = query.validate_boosts();
                let ok = if rejected {
                    matches!(
                        &result,
                        Err(LaurusError::InvalidArgument(m))
                            if m.contains("field 'price' has a NaN bound")
                    )
                } else {
                    result.is_ok()
                };
                if !ok {
                    wrong.push(format!(
                        "{numeric_type:?} [{lower:?}, {upper:?}]: {result:?}"
                    ));
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// Issue #1601: `NumericRangeQuery` overrides `validate_boosts`, so it
    /// must still reject a bad boost as every query does (Issue #1550).
    #[test]
    fn validate_boosts_still_rejects_a_bad_boost_on_a_numeric_range() {
        let mut wrong = Vec::new();
        for boost in [-1.0, f32::NAN, f32::INFINITY] {
            let query =
                NumericRangeQuery::f64_range("price", Some(0.0), Some(5.0)).with_boost(boost);
            let result = query.validate_boosts();
            if !matches!(
                &result,
                Err(LaurusError::InvalidArgument(m)) if m.contains("query boost must be")
            ) {
                wrong.push(format!("boost {boost}: {result:?}"));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }
}
