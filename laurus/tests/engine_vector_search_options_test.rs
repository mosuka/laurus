//! Issue #1342: `SearchRequestBuilder`'s vector options reach the vector
//! search, whether the vector part comes from pre-embedded vectors or from
//! a DSL vector clause.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;

use laurus::storage::memory::MemoryStorage;
use laurus::vector::store::request::FieldSelector;
use laurus::vector::{FlatOption, QueryVector, Vector, VectorSearchQuery};
use laurus::{DataValue, Document, EmbedInput, EmbedInputType, Embedder, Engine};
use laurus::{FieldOption, LaurusError, Result, Schema, SearchRequestBuilder};

/// Embeds any text as `[1, 0]`.
#[derive(Debug)]
struct UnitXEmbedder;

#[async_trait]
impl Embedder for UnitXEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(_) => Ok(Vector::new(vec![1.0, 0.0])),
            _ => Err(LaurusError::invalid_argument(
                "UnitXEmbedder only embeds text",
            )),
        }
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "unit_x"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn flat(dimension: usize) -> FieldOption {
    FieldOption::Flat(FlatOption::default().dimension(dimension))
}

async fn put_vector(engine: &Engine, id: &str, field: &str, vector: Vec<f32>) {
    engine
        .put_document(
            id,
            Document::builder()
                .add_field(field, DataValue::Vector(vector))
                .build(),
        )
        .await
        .unwrap();
}

fn sorted_ids(results: &[laurus::SearchResult]) -> Vec<String> {
    let mut ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
    ids.sort();
    ids
}

/// `vector_fields` routes a query vector that names no field.
#[tokio::test(flavor = "multi_thread")]
async fn vector_fields_routes_a_field_less_vector_query() {
    let schema = Schema::builder()
        .add_field("a", flat(2))
        .add_field("b", flat(2))
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    put_vector(&engine, "in_a", "a", vec![1.0, 0.0]).await;
    put_vector(&engine, "in_b", "b", vec![1.0, 0.0]).await;
    engine.commit().await.unwrap();

    let query = || {
        VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![1.0, 0.0]),
            weight: 1.0,
            fields: None,
        }])
    };

    let all = engine
        .search(SearchRequestBuilder::new().vector_query(query()).build())
        .await
        .unwrap();
    assert_eq!(sorted_ids(&all), vec!["in_a", "in_b"]);

    let only_a = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(query())
                .vector_fields(vec![FieldSelector::Exact("a".to_string())])
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(sorted_ids(&only_a), vec!["in_a"]);
}

/// `vector_min_score` applies to the vector part of a DSL query.
#[tokio::test(flavor = "multi_thread")]
async fn vector_min_score_applies_to_a_dsl_vector_clause() {
    let schema = Schema::builder().add_field("vec", flat(2)).build();
    let engine = Engine::builder(Arc::new(MemoryStorage::new(Default::default())), schema)
        .embedder(Arc::new(UnitXEmbedder))
        .build()
        .await
        .unwrap();
    put_vector(&engine, "near", "vec", vec![1.0, 0.0]).await;
    put_vector(&engine, "far", "vec", vec![0.0, 1.0]).await;
    engine.commit().await.unwrap();

    let all = engine
        .search(SearchRequestBuilder::new().query_dsl("vec:\"x\"").build())
        .await
        .unwrap();
    assert_eq!(sorted_ids(&all), vec!["far", "near"]);

    let filtered = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl("vec:\"x\"")
                .vector_min_score(0.9)
                .build(),
        )
        .await
        .unwrap();
    assert_eq!(sorted_ids(&filtered), vec!["near"]);
}
