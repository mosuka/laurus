//! Issue #1366: a DSL query set together with a vector query keeps both.
//! The vectors are added to the DSL's vector part (a lexical-only DSL
//! becomes a hybrid search) instead of being silently dropped.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;

use laurus::storage::memory::MemoryStorage;
use laurus::vector::store::request::QueryPayload;
use laurus::vector::{FlatOption, QueryVector, Vector, VectorSearchQuery};
use laurus::{DataValue, Document, EmbedInput, EmbedInputType, Embedder, Engine, FieldOption};
use laurus::{LaurusError, Result, Schema, SearchRequestBuilder, TextOption};

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

/// `lex_only` matches `title:rust` only; `vec_only` is close to `[1, 0]` in
/// `vec` only; `vec2_only` is close to `[1, 0]` in `vec2` only.
async fn engine() -> Engine {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field("vec", FieldOption::Flat(FlatOption::default().dimension(2)))
        .add_field(
            "vec2",
            FieldOption::Flat(FlatOption::default().dimension(2)),
        )
        .build();
    let engine = Engine::builder(Arc::new(MemoryStorage::new(Default::default())), schema)
        .embedder(Arc::new(UnitXEmbedder))
        .build()
        .await
        .unwrap();
    let docs = [
        ("lex_only", "rust", "vec", [0.0, 1.0]),
        ("vec_only", "go", "vec", [1.0, 0.0]),
        ("vec2_only", "python", "vec2", [1.0, 0.0]),
    ];
    for (id, title, field, vector) in docs {
        engine
            .put_document(
                id,
                Document::builder()
                    .add_text("title", title)
                    .add_field(field, DataValue::Vector(vector.to_vec()))
                    .build(),
            )
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

fn vectors(field: &str) -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0, 0.0]),
        weight: 1.0,
        fields: Some(vec![field.to_string()]),
    }])
}

async fn ids(engine: &Engine, builder: SearchRequestBuilder) -> Vec<String> {
    let mut ids: Vec<String> = engine
        .search(builder.build())
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    ids.sort();
    ids
}

/// A lexical-only DSL plus vectors becomes a hybrid search.
#[tokio::test(flavor = "multi_thread")]
async fn lexical_dsl_with_vectors_is_a_hybrid_search() {
    let engine = engine().await;
    let found = ids(
        &engine,
        SearchRequestBuilder::new()
            .query_dsl("title:rust")
            .vector_query(vectors("vec")),
    )
    .await;
    assert!(found.contains(&"lex_only".to_string()), "{found:?}");
    assert!(found.contains(&"vec_only".to_string()), "{found:?}");
}

/// Vectors are added to the vector clauses the DSL already has.
#[tokio::test(flavor = "multi_thread")]
async fn dsl_vector_clauses_and_extra_vectors_both_apply() {
    let engine = engine().await;
    let found = ids(
        &engine,
        SearchRequestBuilder::new()
            .query_dsl("vec:\"x\"")
            .vector_query(vectors("vec2")),
    )
    .await;
    assert!(found.contains(&"vec_only".to_string()), "{found:?}");
    assert!(found.contains(&"vec2_only".to_string()), "{found:?}");
}

/// Extra payloads are embedded before they are added.
#[tokio::test(flavor = "multi_thread")]
async fn extra_payloads_are_embedded() {
    let engine = engine().await;
    let found = ids(
        &engine,
        SearchRequestBuilder::new()
            .query_dsl("title:rust")
            .vector_query(VectorSearchQuery::Payloads(vec![QueryPayload {
                field: "vec2".to_string(),
                payload: DataValue::Text("x".to_string()),
                weight: 1.0,
            }])),
    )
    .await;
    assert!(found.contains(&"lex_only".to_string()), "{found:?}");
    assert!(found.contains(&"vec2_only".to_string()), "{found:?}");
}

/// No extra vectors means the same search as the DSL alone.
#[tokio::test(flavor = "multi_thread")]
async fn empty_extra_vectors_match_the_dsl_alone() {
    let engine = engine().await;
    let alone = ids(&engine, SearchRequestBuilder::new().query_dsl("title:rust")).await;
    let with_empty = ids(
        &engine,
        SearchRequestBuilder::new()
            .query_dsl("title:rust")
            .vector_query(VectorSearchQuery::Vectors(Vec::new())),
    )
    .await;
    assert_eq!(alone, ["lex_only"]);
    assert_eq!(with_empty, alone);
}
