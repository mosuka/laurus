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
    /// Weight for this query vector (default: 1.0).
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

    /// Create a new query payload with a specific weight.
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
