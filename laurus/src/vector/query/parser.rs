//! DSL parser for vector search queries.
//!
//! Parses `field:"text"` or `field:text` syntax into VectorSearchRequest,
//! embedding text into vectors at parse time. The caller (typically the
//! unified query parser) is responsible for routing only vector-field
//! clauses to this parser.

use std::sync::Arc;

use pest::Parser;
use pest_derive::Parser;

use crate::data::DataValue;
use crate::embedding::embedder::{EmbedInput, Embedder};
use crate::error::{LaurusError, Result};
use crate::vector::store::request::{
    QueryPayload, QueryVector, VectorSearchQuery, VectorSearchRequest,
};

/// Pest grammar parser for vector query DSL.
#[derive(Parser)]
#[grammar = "vector/query/parser.pest"]
struct VectorQueryStringParser;

/// Parser for vector search DSL.
///
/// Converts `field:"text"` or `field:text` syntax into `VectorSearchRequest`
/// with embedded vectors. Requires an `Embedder` to convert text into vectors
/// at parse time, following the same pattern as the lexical `QueryParser`
/// which requires an `Analyzer`.
///
/// # Supported Syntax
///
/// - `content:"cute kitten"` — field-specific quoted text query
/// - `content:python` — field-specific unquoted text query
/// - `content:"cute kitten"^0.8` — with weight (boost)
/// - `content:python^0.8` — unquoted with weight (boost)
/// - `"cute kitten"` — uses default field (quoted)
/// - `content:"cats" image:"dogs"^0.5` — multiple queries
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use laurus::vector::query::VectorQueryParser;
///
/// let parser = VectorQueryParser::new(embedder)
///     .with_default_field("content");
///
/// let request = parser.parse(r#"content:"cute kitten""#).await.unwrap();
/// // request.query is VectorSearchQuery::Vectors(vec![...])
/// ```
pub struct VectorQueryParser {
    embedder: Arc<dyn Embedder>,
    default_fields: Vec<String>,
    /// Optional query-time embedding cache, shared with the engine so DSL
    /// queries reuse embeddings produced by the direct payload path
    /// (Issue #678). `None` disables caching.
    embedding_cache: Option<Arc<crate::embedding::cache::EmbeddingCache>>,
}

impl VectorQueryParser {
    /// Create a new VectorQueryParser with the given embedder.
    ///
    /// Following the same pattern as `QueryParser::new(analyzer)`,
    /// an `Embedder` is required to convert query text into vectors.
    pub fn new(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedder,
            default_fields: Vec::new(),
            embedding_cache: None,
        }
    }

    /// Attach a shared query-time embedding cache (Issue #678).
    ///
    /// When set, identical query payloads embedded by the same field /
    /// embedder are produced only once. The cache is normally shared with
    /// the owning [`Engine`](crate::engine::Engine).
    pub fn with_embedding_cache(
        mut self,
        cache: Arc<crate::embedding::cache::EmbeddingCache>,
    ) -> Self {
        self.embedding_cache = Some(cache);
        self
    }

    /// Set a single default field for queries without explicit field prefix.
    pub fn with_default_field(mut self, field: impl Into<String>) -> Self {
        self.default_fields = vec![field.into()];
        self
    }

    /// Set multiple default fields for queries without explicit field prefix.
    pub fn with_default_fields(mut self, fields: Vec<String>) -> Self {
        self.default_fields = fields;
        self
    }

    /// Parse a vector query DSL string into a VectorSearchRequest.
    ///
    /// Text payloads are embedded into vectors at parse time using the
    /// configured embedder. The resulting `VectorSearchRequest` contains
    /// a `VectorSearchQuery::Vectors` query (not `Payloads`).
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidArgument`] when the string does not
    /// parse, when a clause has no field and no default field is set, or when
    /// a clause uses the removed `~` marker (`content:~"cute kitten"`).
    pub async fn parse(&self, query_str: &str) -> Result<VectorSearchRequest> {
        let pairs = VectorQueryStringParser::parse(Rule::query, query_str).map_err(|e| {
            LaurusError::invalid_argument(format!("Failed to parse vector query: {}", e))
        })?;

        let mut payloads = Vec::new();

        for pair in pairs {
            if pair.as_rule() == Rule::query {
                for inner in pair.into_inner() {
                    if inner.as_rule() == Rule::vector_clause {
                        let payload = self.parse_vector_clause(inner)?;
                        payloads.push(payload);
                    }
                }
            }
        }

        if payloads.is_empty() {
            return Err(LaurusError::invalid_argument(
                "Vector query must contain at least one clause",
            ));
        }

        // Embed every clause's payload in one batch (Issue #671) so a
        // batch-capable embedder pays one round trip instead of one per clause,
        // while preserving the shared embedding cache (#678) and per-field
        // routing. Non-text / non-bytes payloads are skipped, as before.
        let mut kept: Vec<&QueryPayload> = Vec::new();
        let mut items: Vec<(String, EmbedInput<'_>)> = Vec::new();
        for payload in &payloads {
            let input = match &payload.payload {
                DataValue::Text(t) => EmbedInput::Text(t),
                DataValue::Bytes(b, m) => EmbedInput::Bytes(b, m.as_deref()),
                _ => continue,
            };
            items.push((payload.field.clone(), input));
            kept.push(payload);
        }
        let vectors = crate::embedding::cache::embed_batch_with_cache(
            self.embedding_cache.as_ref(),
            &self.embedder,
            &items,
        )
        .await?;
        let query_vectors: Vec<QueryVector> = kept
            .into_iter()
            .zip(vectors)
            .map(|(payload, vector)| QueryVector {
                vector,
                weight: payload.weight,
                fields: Some(vec![payload.field.clone()]),
            })
            .collect();

        Ok(VectorSearchRequest {
            query: VectorSearchQuery::Vectors(query_vectors),
            params: Default::default(),
        })
    }

    /// Parse a single vector clause (e.g., `content:"cute kitten"^0.8` or `content:python`).
    fn parse_vector_clause(&self, pair: pest::iterators::Pair<Rule>) -> Result<QueryPayload> {
        let clause = pair.as_str();
        let mut field_name: Option<String> = None;
        let mut text: Option<String> = None;
        let mut weight: f32 = 1.0;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::field_prefix => {
                    // Extract field_name from field_prefix
                    for fp_inner in inner.into_inner() {
                        if fp_inner.as_rule() == Rule::field_name {
                            field_name = Some(fp_inner.as_str().to_string());
                        }
                    }
                }
                Rule::quoted_text => {
                    // Extract text from quoted_text → inner_text
                    for qt_inner in inner.into_inner() {
                        if qt_inner.as_rule() == Rule::inner_text {
                            text = Some(qt_inner.as_str().to_string());
                        }
                    }
                }
                Rule::plain_text => {
                    let value = inner.as_str();
                    // The grammar reads the removed marker of `field:~"text"`
                    // as an unquoted value `~`, so reject it here instead of
                    // embedding it (Issue #1495).
                    if value.starts_with('~') {
                        return Err(removed_marker_error(clause));
                    }
                    text = Some(value.to_string());
                }
                Rule::boost => {
                    // Extract weight from boost → float_value
                    for b_inner in inner.into_inner() {
                        if b_inner.as_rule() == Rule::float_value {
                            weight = b_inner.as_str().parse::<f32>().map_err(|e| {
                                LaurusError::invalid_argument(format!("Invalid boost value: {}", e))
                            })?;
                        }
                    }
                }
                _ => {}
            }
        }

        // Resolve field name.
        // NOTE: When no field is specified, only the first default field is used.
        // Multi-default-field support (generating a QueryVector per field) is not
        // yet implemented.
        let field = match field_name {
            Some(f) => f,
            None => {
                if self.default_fields.is_empty() {
                    return Err(LaurusError::invalid_argument(
                        "No field specified and no default field configured",
                    ));
                }
                self.default_fields[0].clone()
            }
        };

        let text =
            text.ok_or_else(|| LaurusError::invalid_argument("Missing text in vector clause"))?;

        Ok(QueryPayload::with_weight(
            field,
            DataValue::Text(text),
            weight,
        ))
    }
}

/// Build the error for a vector clause that uses the `~` marker
/// (`content:~"cute kitten"`), which the vector query syntax no longer has
/// (Issue #1495).
///
/// [`VectorQueryParser`] and the unified query parser both return it, so the
/// old syntax fails with an error that names the marker instead of being read
/// as different clauses or as a lexical query.
pub(crate) fn removed_marker_error(clause: &str) -> LaurusError {
    LaurusError::invalid_argument(format!(
        "Vector clause '{clause}' uses the '~' marker, which was removed from the \
         vector query syntax. Write the text right after the field (e.g. \
         content:\"cute kitten\", not content:~\"cute kitten\"), and quote text \
         that starts with '~' (e.g. content:\"~text\")."
    ))
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use async_trait::async_trait;

    use super::*;
    use crate::embedding::embedder::EmbedInputType;
    use crate::vector::core::vector::Vector;

    /// Mock embedder that returns a zero vector of the configured dimension.
    #[derive(Debug)]
    struct MockEmbedder {
        dimension: usize,
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
            Ok(Vector::new(vec![0.0; self.dimension]))
        }
        fn supported_input_types(&self) -> Vec<EmbedInputType> {
            vec![EmbedInputType::Text]
        }
        fn name(&self) -> &str {
            "mock"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn mock_embedder() -> Arc<dyn Embedder> {
        Arc::new(MockEmbedder { dimension: 4 })
    }

    /// Extract query vectors from a VectorSearchRequest, panicking if the
    /// query is not the `Vectors` variant.
    fn get_vectors(req: &VectorSearchRequest) -> &[QueryVector] {
        match &req.query {
            VectorSearchQuery::Vectors(v) => v,
            _ => panic!("Expected VectorSearchQuery::Vectors"),
        }
    }

    /// Helper to extract the first field name from a QueryVector.
    fn qv_field(qv: &QueryVector) -> &str {
        &qv.fields.as_ref().unwrap()[0]
    }

    #[tokio::test]
    async fn test_basic_quoted_query() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser.parse(r#"content:"cute kitten""#).await.unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(vecs.len(), 1);
        let qv = &vecs[0];
        assert_eq!(qv.fields.as_ref().unwrap()[0], "content");
        assert_eq!(qv.weight, 1.0);
        assert_eq!(qv.vector.dimension(), 4);
    }

    #[tokio::test]
    async fn test_basic_unquoted_query() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser.parse("content:python").await.unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(vecs.len(), 1);
        let qv = &vecs[0];
        assert_eq!(qv.fields.as_ref().unwrap()[0], "content");
        assert_eq!(qv.weight, 1.0);
        assert_eq!(qv.vector.dimension(), 4);
    }

    #[tokio::test]
    async fn test_quoted_boost() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser.parse(r#"content:"text"^0.8"#).await.unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(vecs.len(), 1);
        let qv = &vecs[0];
        assert_eq!(qv.fields.as_ref().unwrap()[0], "content");
        assert!((qv.weight - 0.8).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_unquoted_boost() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser.parse("content:python^0.8").await.unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(vecs.len(), 1);
        assert!((vecs[0].weight - 0.8).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_default_field_quoted() {
        let parser = VectorQueryParser::new(mock_embedder()).with_default_field("embedding");
        let request = parser.parse(r#""cute kitten""#).await.unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].fields.as_ref().unwrap()[0], "embedding");
    }

    #[tokio::test]
    async fn test_default_field_unquoted() {
        let parser = VectorQueryParser::new(mock_embedder()).with_default_field("embedding");
        let request = parser.parse("python").await.unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs[0].fields.as_ref().unwrap()[0], "embedding");
    }

    #[tokio::test]
    async fn test_multiple_clauses() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser
            .parse(r#"content:"cats" image:"dogs"^0.5"#)
            .await
            .unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(vecs.len(), 2);

        assert_eq!(vecs[0].fields.as_ref().unwrap()[0], "content");
        assert_eq!(vecs[0].weight, 1.0);

        assert_eq!(vecs[1].fields.as_ref().unwrap()[0], "image");
        assert!((vecs[1].weight - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_mixed_quoted_unquoted() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser
            .parse(r#"content:"cute kitten" image:dogs^0.5"#)
            .await
            .unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(vecs.len(), 2);
        assert_eq!(qv_field(&vecs[0]), "content");
        assert_eq!(qv_field(&vecs[1]), "image");
        assert!((vecs[1].weight - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_empty_query_error() {
        let parser = VectorQueryParser::new(mock_embedder());
        assert!(parser.parse("").await.is_err());
    }

    #[tokio::test]
    async fn test_no_field_no_default_error() {
        let parser = VectorQueryParser::new(mock_embedder()); // no default field
        assert!(parser.parse(r#""text""#).await.is_err());
    }

    #[tokio::test]
    async fn test_unicode_text() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser.parse(r#"content:"日本語テスト""#).await.unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(vecs.len(), 1);
        assert_eq!(qv_field(&vecs[0]), "content");
        assert_eq!(vecs[0].vector.dimension(), 4);
    }

    #[tokio::test]
    async fn test_integer_boost() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser.parse(r#"content:"text"^2"#).await.unwrap();

        let vecs = get_vectors(&request);
        assert!((vecs[0].weight - 2.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_field_with_underscore() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser.parse(r#"my_field:"text""#).await.unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(qv_field(&vecs[0]), "my_field");
    }

    #[tokio::test]
    async fn test_field_with_dot() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser.parse(r#"nested.field:"text""#).await.unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(qv_field(&vecs[0]), "nested.field");
    }

    /// Assert that `query` fails with the removed `~` marker error.
    async fn assert_removed_marker_error(parser: &VectorQueryParser, query: &str) {
        let err = parser.parse(query).await.unwrap_err();
        assert!(
            matches!(err, LaurusError::InvalidArgument(_)),
            "got: {err:?}"
        );
        assert!(err.to_string().contains("'~' marker"), "got: {err}");
    }

    /// `field:~"text"` is the syntax #275 removed. The parser used to fail
    /// on it with a missing-field error, or, with a default field, embed `~`
    /// and the text as two clauses (Issue #1495).
    #[tokio::test]
    async fn test_removed_tilde_marker_is_rejected() {
        let parser = VectorQueryParser::new(mock_embedder());
        assert_removed_marker_error(&parser, r#"text_vec:~"memory safety""#).await;

        let parser = VectorQueryParser::new(mock_embedder()).with_default_field("text_vec");
        assert_removed_marker_error(&parser, r#"text_vec:~"memory safety""#).await;
        assert_removed_marker_error(&parser, "content:~python").await;
        assert_removed_marker_error(&parser, r#"~"memory safety""#).await;
        assert_removed_marker_error(&parser, r#"content:"cats" image:~"dogs"^0.5"#).await;
    }

    /// Quoted text may start with `~`: only an unquoted value is the marker.
    #[tokio::test]
    async fn test_quoted_text_starting_with_tilde() {
        let parser = VectorQueryParser::new(mock_embedder());
        let request = parser.parse(r#"content:"~python""#).await.unwrap();

        let vecs = get_vectors(&request);
        assert_eq!(vecs.len(), 1);
        assert_eq!(qv_field(&vecs[0]), "content");
    }
}
