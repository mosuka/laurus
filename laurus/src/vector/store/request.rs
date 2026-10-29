//! Vector store request value types.
//!
//! This module provides low-level value types used in vector search requests:
//! query vectors, query payloads, field selectors, and score combination modes.
//!
//! The high-level request and query types ([`VectorSearchRequest`],
//! [`VectorSearchQuery`]) live in [`crate::vector::search::searcher`], mirroring
//! the lexical module's organization where [`LexicalSearchRequest`] and
//! [`LexicalSearchQuery`] live in [`crate::lexical::search::searcher`].

use serde::{Deserialize, Serialize};

use crate::data::DataValue;
use crate::error::{LaurusError, Result};
use crate::vector::core::vector::Vector;

// Re-export high-level types from their canonical location for backward compatibility.
pub use crate::vector::search::searcher::{
    VectorSearchParams, VectorSearchQuery, VectorSearchRequest,
};

/// Selector for choosing which vector fields to include in a search.
///
/// Fields can be selected either by their exact name or by a name prefix,
/// allowing flexible targeting of specific vector fields within a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum FieldSelector {
    /// Select a field by its exact name (e.g., `"title_embedding"`).
    Exact(String),
    /// Select all fields whose names start with the given prefix
    /// (e.g., `"image_"` matches `"image_thumbnail"`, `"image_full"`, etc.).
    Prefix(String),
}

/// Strategy for combining similarity scores when a search uses multiple query vectors.
///
/// Different modes suit different retrieval scenarios. For example,
/// [`WeightedSum`](Self::WeightedSum) works well when all query vectors contribute
/// additively, while [`MaxSim`](Self::MaxSim) is better for alternative-interpretation
/// queries.
///
/// Combination always happens *across query vectors*: for a given document,
/// each query vector contributes at most one term, never one per matching
/// field. A field-less query vector that reaches a document through several
/// same-dimension fields collapses to that document's best-scoring field
/// before this combination runs (Issue #1343); a query vector targeting
/// fields by name keeps summing across them, weighted by
/// [`HnswOption::base_weight`](crate::vector::core::field::HnswOption::base_weight),
/// since it is explicitly querying several fields at once rather than
/// fanning out to find whichever one matches.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VectorScoreMode {
    /// Sum of `similarity * weight` across all query vectors.
    #[default]
    WeightedSum,
    /// Maximum `similarity * weight` across all query vectors.
    /// Useful when multiple query vectors represent alternative interpretations
    /// and only the best-matching one should determine the score.
    MaxSim,
    /// For each query vector, the max similarity across a document's vectors,
    /// summed. A vector field holds one vector per document, so this is
    /// currently the same as [`WeightedSum`](Self::WeightedSum).
    ///
    /// For ColBERT-style late interaction over a document's token vectors,
    /// use a [`RescoreOptions::late_interaction`](crate::RescoreOptions::late_interaction)
    /// rescore against a multi-vector field instead.
    LateInteraction,
}

/// A pre-embedded query vector with an optional weight and field restriction.
///
/// Each `QueryVector` carries a dense embedding that has already been produced
/// by an external embedding model. It is used directly for similarity
/// computation against the stored document vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryVector {
    /// Dense vector embedding representing the query.
    pub vector: Vector,
    /// Multiplicative weight applied to the similarity score produced by this
    /// vector. Defaults to `1.0`.
    ///
    /// Must be finite and at least `0` (`-0.0` is rejected); a search with any
    /// other weight fails with
    /// [`LaurusError::InvalidArgument`](crate::error::LaurusError::InvalidArgument)
    /// (Issue #1552). A weight of `0` gives the vector's hits a score of `0`;
    /// in a hybrid search they still take part in fusion (RRF ranks them, and
    /// `WeightedSum` normalizes a list whose scores are all equal to `1.0`).
    /// A negative weight is rejected because candidates are retrieved nearest
    /// first: it would only reverse the order of the nearest candidates,
    /// never find far documents.
    #[serde(default = "QueryVector::default_weight")]
    pub weight: f32,
    /// Optional list of fields to restrict this query vector to.
    /// If None, it applies to all target fields.
    #[serde(default)]
    pub fields: Option<Vec<String>>,
}

impl QueryVector {
    fn default_weight() -> f32 {
        1.0
    }
}

/// Query payload for a specific field (to be embedded internally).
///
/// This allows users to pass raw payloads (text, images, etc.) that will be
/// automatically embedded using the configured embedder during search.
///
/// Note: This type is not serializable because `Payload` contains
/// non-serializable data (e.g., `Arc<[u8]>`). Use `QueryVector` for
/// serialization scenarios with pre-embedded vectors.
#[derive(Debug, Clone)]
pub struct QueryPayload {
    /// The field name to search in.
    pub field: String,
    /// The payload to embed.
    pub payload: DataValue,
    /// Weight for this query vector (default: 1.0), in the range of
    /// [`QueryVector::weight`].
    pub weight: f32,
}

impl QueryPayload {
    /// Create a new query payload from a `DataValue`.
    pub fn new(field: impl Into<String>, payload: DataValue) -> Self {
        Self {
            field: field.into(),
            payload,
            weight: Self::default_weight(),
        }
    }

    /// Create a new query payload with a specific weight, in the range of
    /// [`QueryVector::weight`] (checked when the search runs).
    pub fn with_weight(field: impl Into<String>, payload: DataValue, weight: f32) -> Self {
        Self {
            field: field.into(),
            payload,
            weight,
        }
    }

    fn default_weight() -> f32 {
        1.0
    }
}

/// Check a query vector's weight (Issue #1552; see [`QueryVector::weight`]):
/// finite and at least `0`, with `-0.0` rejected, the rule of a query boost
/// ([`is_valid_boost`](crate::lexical::query::is_valid_boost), Issues #1540,
/// #1550).
///
/// `describe` names the query vector, payload or DSL clause, and runs only
/// when the weight is out of range.
///
/// # Errors
///
/// Returns [`LaurusError::InvalidArgument`] when the weight is not finite or
/// is negative.
pub(crate) fn check_query_vector_weight(
    weight: f32,
    describe: impl FnOnce() -> String,
) -> Result<()> {
    if crate::lexical::query::is_valid_boost(weight) {
        return Ok(());
    }
    Err(LaurusError::invalid_argument(format!(
        "query vector weight must be finite and at least 0, got {weight}: {}",
        describe()
    )))
}

/// Check a vector `min_score` (Issue #1552): finite, as the lexical one
/// (Issue #1540). Similarity is in `[0, 1]`, so `+inf` would drop every hit,
/// `-inf` would keep them all like any value of `0` or less, and `NaN` would
/// drop or keep every hit depending on the search path.
///
/// # Errors
///
/// Returns [`LaurusError::InvalidArgument`] when `min_score` is `NaN` or
/// infinite.
pub(crate) fn check_vector_min_score(min_score: f32) -> Result<()> {
    if min_score.is_finite() {
        return Ok(());
    }
    Err(LaurusError::invalid_argument(format!(
        "vector min_score must be finite, got {min_score}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The values of `values` that `check` accepts, so one assertion lists
    /// every out-of-range value that got through.
    fn accepted(values: &[f32], check: impl Fn(f32) -> Result<()>) -> Vec<f32> {
        values
            .iter()
            .copied()
            .filter(|&value| check(value).is_ok())
            .collect()
    }

    /// Issue #1552: a weight must be finite and at least `0`, `-0.0` rejected.
    #[test]
    fn query_vector_weight_range() {
        let check = |weight| check_query_vector_weight(weight, || "test".to_string());
        let bad = [-1.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        assert_eq!(accepted(&bad, check), Vec::<f32>::new());
        for weight in [0.0, 1.0, 2.5, 1e30] {
            check(weight).unwrap();
        }
        let err = check(f32::INFINITY).unwrap_err().to_string();
        assert!(
            err.contains("query vector weight must be finite and at least 0, got inf: test"),
            "{err}"
        );
    }

    /// Issue #1552: a vector `min_score` must be finite; a negative one is
    /// accepted (it is the same as `0`).
    #[test]
    fn vector_min_score_range() {
        let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        assert_eq!(accepted(&bad, check_vector_min_score), Vec::<f32>::new());
        for min_score in [-1.0, 0.0, 0.5, 1.0] {
            check_vector_min_score(min_score).unwrap();
        }
    }
}
