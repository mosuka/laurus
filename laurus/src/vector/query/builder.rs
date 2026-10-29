//! Builder for VectorSearchRequest.
//!
//! This module provides a fluent API for constructing vector search requests.

use crate::data::DataValue;
use crate::vector::core::vector::Vector;

use crate::vector::store::request::{
    FieldSelector, QueryPayload, QueryVector, VectorScoreMode, VectorSearchParams,
    VectorSearchQuery, VectorSearchRequest,
};

/// Builder for constructing VectorSearchRequest.
///
/// # Example
///
/// ```
/// use laurus::vector::query::VectorSearchRequestBuilder;
///
/// let request = VectorSearchRequestBuilder::new()
///     .add_vector("content", vec![0.1, 0.2, 0.3])
///     .limit(5)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct VectorSearchRequestBuilder {
    query_vectors: Vec<QueryVector>,
    query_payloads: Vec<QueryPayload>,
    params: VectorSearchParams,
}

impl Default for VectorSearchRequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorSearchRequestBuilder {
    /// Create a new VectorSearchRequestBuilder.
    pub fn new() -> Self {
        Self {
            query_vectors: Vec::new(),
            query_payloads: Vec::new(),
            params: VectorSearchParams::default(),
        }
    }

    /// Add a raw query vector for a specific field.
    pub fn add_vector(mut self, field: impl Into<String>, vector: Vec<f32>) -> Self {
        self.query_vectors.push(QueryVector {
            vector: Vector::new(vector),
            weight: 1.0,
            fields: Some(vec![field.into()]),
        });
        self
    }

    /// Add a raw query vector with explicit weight for a specific field. The
    /// weight must be finite and at least `0` (see
    /// [`QueryVector::weight`](crate::vector::store::request::QueryVector::weight)).
    pub fn add_vector_with_weight(
        mut self,
        field: impl Into<String>,
        vector: Vec<f32>,
        weight: f32,
    ) -> Self {
        self.query_vectors.push(QueryVector {
            vector: Vector::new(vector),
            weight,
            fields: Some(vec![field.into()]),
        });
        self
    }

    /// Add a payload to be embedded.
    ///
    /// This is the unified method for all modalities (text, image, video, etc.).
    /// The bytes will be processed by the configured embedder.
    ///
    /// # Arguments
    ///
    /// * `field` - The target field name
    /// * `payload` - The payload to add
    ///
    /// This is the low-level method used by `add_text`, `add_image`, etc.
    pub fn add_payload(mut self, field: impl Into<String>, payload: DataValue) -> Self {
        self.query_payloads.push(QueryPayload::new(field, payload));
        self
    }

    /// Add a raw bytes payload (e.g. image bytes).
    pub fn add_bytes(
        self,
        field: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
        mime: Option<impl Into<String>>,
    ) -> Self {
        self.add_payload(
            field,
            DataValue::Bytes(bytes.into(), mime.map(|m| m.into())),
        )
    }

    /// Add a text payload to be embedded.
    pub fn add_text(self, field: impl Into<String>, text: impl Into<String>) -> Self {
        self.add_payload(field, DataValue::Text(text.into()))
    }

    /// Set the fields to search in.
    pub fn fields(mut self, fields: Vec<String>) -> Self {
        self.params.fields = Some(fields.into_iter().map(FieldSelector::Exact).collect());
        self
    }

    /// Add a field to search in.
    ///
    /// This is a convenience method to add a single field.
    pub fn field(mut self, field: impl Into<String>) -> Self {
        let field = field.into();
        if let Some(fields) = &mut self.params.fields {
            fields.push(FieldSelector::Exact(field));
        } else {
            self.params.fields = Some(vec![FieldSelector::Exact(field)]);
        }
        self
    }

    /// Set the search limit.
    pub fn limit(mut self, limit: usize) -> Self {
        self.params.limit = limit;
        self
    }

    /// Set the score mode.
    pub fn score_mode(mut self, mode: VectorScoreMode) -> Self {
        self.params.score_mode = mode;
        self
    }

    /// Set the overfetch factor.
    pub fn overfetch(mut self, overfetch: f32) -> Self {
        self.params.overfetch = overfetch;
        self
    }

    /// Set the minimum score threshold. It must be finite (see
    /// [`VectorSearchParams::min_score`](crate::vector::search::searcher::VectorSearchParams::min_score)).
    pub fn min_score(mut self, min_score: f32) -> Self {
        self.params.min_score = min_score;
        self
    }

    /// Build the VectorSearchRequest.
    ///
    /// If any pre-embedded vectors were added via [`add_vector`](Self::add_vector)
    /// or [`add_vector_with_weight`](Self::add_vector_with_weight), the query
    /// will use [`VectorSearchQuery::Vectors`]. Otherwise, if payloads were
    /// added via [`add_payload`](Self::add_payload), [`add_text`](Self::add_text),
    /// or [`add_bytes`](Self::add_bytes), the query will use
    /// [`VectorSearchQuery::Payloads`].
    pub fn build(self) -> VectorSearchRequest {
        let query = if !self.query_vectors.is_empty() {
            VectorSearchQuery::Vectors(self.query_vectors)
        } else {
            VectorSearchQuery::Payloads(self.query_payloads)
        };
        VectorSearchRequest {
            query,
            params: self.params,
        }
    }
}
