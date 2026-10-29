//! Issue #1342: `SearchRequestBuilder`'s vector options reach the vector
//! search, whether the vector part comes from pre-embedded vectors or from
//! a DSL vector clause.
//!
//! Issue #1552: `Engine::search` rejects a query vector weight that is not
//! finite or is negative, and a vector `min_score` that is not finite,
//! whatever the data and before anything is embedded.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use laurus::lexical::{Query, TermQuery};
use laurus::storage::memory::MemoryStorage;
use laurus::vector::store::request::{FieldSelector, QueryPayload};
use laurus::vector::{FlatOption, QueryVector, Vector, VectorSearchQuery};
use laurus::{DataValue, Document, EmbedInput, EmbedInputType, Embedder, Engine};
use laurus::{FieldOption, LaurusError, Result, Schema, SearchRequestBuilder};
use laurus::{GroupBy, LexicalSearchQuery, SearchRequest, SearchResult, TextOption};

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

/// Embeds any text as `[1, 0]` and counts the inputs it embeds.
#[derive(Debug, Default)]
struct CountingEmbedder {
    calls: AtomicUsize,
}

#[async_trait]
impl Embedder for CountingEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        UnitXEmbedder.embed(input).await
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "counting"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An engine with a text field `title` and a 2-dimensional vector field
/// `vec`: doc1 (`rust`, `[1, 0]`) and doc2 (`go`, `[0.6, 0.8]`), committed.
async fn weight_engine(embedder: Arc<CountingEmbedder>) -> Engine {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field("vec", flat(2))
        .build();
    let engine = Engine::builder(Arc::new(MemoryStorage::new(Default::default())), schema)
        .embedder(embedder)
        .build()
        .await
        .unwrap();
    for (id, title, vector) in [("doc1", "rust", [1.0, 0.0]), ("doc2", "go", [0.6, 0.8])] {
        engine
            .put_document(
                id,
                Document::builder()
                    .add_field("title", DataValue::Text(title.to_string()))
                    .add_field("vec", DataValue::Vector(vector.to_vec()))
                    .build(),
            )
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

/// The query vector `[1, 0]` on `vec` with `weight`.
fn weighted(weight: f32) -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0, 0.0]),
        weight,
        fields: Some(vec!["vec".to_string()]),
    }])
}

/// A filter that matches no document.
fn nothing() -> Box<dyn Query> {
    Box::new(TermQuery::new("title", "nosuchterm"))
}

/// Weights out of range: negative (`-0.0` included) or not finite.
const BAD_WEIGHTS: [f32; 5] = [-1.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

/// Whether `result` is the `InvalidArgument` of an out-of-range weight.
fn is_weight_error(result: &Result<Vec<SearchResult>>) -> bool {
    matches!(
        result,
        Err(LaurusError::InvalidArgument(message))
            if message.contains("query vector weight must be finite and at least 0")
    )
}

/// Builds a search request around a query vector of the given weight.
type RequestWithWeight = Box<dyn Fn(f32) -> SearchRequest>;

/// Issue #1552: an out-of-range weight of a `Vector`, `Hybrid` or
/// `DslWithVectors` query is rejected even where the search returns early
/// without searching the vectors: under a filter that matches nothing, or
/// grouped with `limit` `0`.
#[tokio::test(flavor = "multi_thread")]
async fn a_bad_query_vector_weight_is_rejected_whatever_the_data() {
    let engine = weight_engine(Arc::new(CountingEmbedder::default())).await;
    let with = |weight| SearchRequestBuilder::new().vector_query(weighted(weight));
    let shapes: [(&str, RequestWithWeight); 4] = [
        (
            "vector, filter matches nothing",
            Box::new(move |weight| with(weight).filter_query(nothing()).build()),
        ),
        (
            "hybrid, filter matches nothing",
            Box::new(move |weight| {
                with(weight)
                    .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                        "title", "rust",
                    ))))
                    .filter_query(nothing())
                    .build()
            }),
        ),
        (
            "dsl with vectors, filter matches nothing",
            Box::new(move |weight| {
                with(weight)
                    .query_dsl("title:rust")
                    .filter_query(nothing())
                    .build()
            }),
        ),
        (
            "vector, grouped with limit 0",
            Box::new(move |weight| with(weight).group_by(GroupBy::new("_id")).limit(0).build()),
        ),
    ];

    let mut accepted = Vec::new();
    for (shape, request) in &shapes {
        for weight in BAD_WEIGHTS {
            let result = engine.search(request(weight)).await;
            if !is_weight_error(&result) {
                accepted.push(format!("{shape}: {weight} -> {result:?}"));
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "not rejected:\n{}",
        accepted.join("\n")
    );
}

/// Issue #1552: a payload's out-of-range weight is rejected before the
/// payload is embedded.
#[tokio::test(flavor = "multi_thread")]
async fn a_bad_payload_weight_is_rejected_before_embedding() {
    let embedder = Arc::new(CountingEmbedder::default());
    let engine = weight_engine(embedder.clone()).await;
    let before = embedder.calls.load(Ordering::SeqCst);

    let result = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(VectorSearchQuery::Payloads(vec![
                    QueryPayload::with_weight(
                        "vec",
                        DataValue::Text("payload text".to_string()),
                        f32::NAN,
                    ),
                ]))
                .build(),
        )
        .await;
    assert!(is_weight_error(&result), "{result:?}");
    assert_eq!(
        embedder.calls.load(Ordering::SeqCst),
        before,
        "nothing is embedded"
    );
}

/// Issue #1552: a DSL vector clause whose `^` is too large for an `f32` is
/// rejected before the clause is embedded, also under a filter that
/// matches nothing.
#[tokio::test(flavor = "multi_thread")]
async fn a_dsl_vector_weight_too_large_for_an_f32_is_rejected() {
    let embedder = Arc::new(CountingEmbedder::default());
    let engine = weight_engine(embedder.clone()).await;
    let before = embedder.calls.load(Ordering::SeqCst);

    let result = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl(format!("vec:\"dsl text\"^{}", "1".repeat(40)))
                .filter_query(nothing())
                .build(),
        )
        .await;
    assert!(is_weight_error(&result), "{result:?}");
    assert_eq!(
        embedder.calls.load(Ordering::SeqCst),
        before,
        "nothing is embedded"
    );
}

/// Issue #1552: a vector `min_score` that is not finite is rejected, even
/// by a request with no vector part.
#[tokio::test(flavor = "multi_thread")]
async fn a_vector_min_score_that_is_not_finite_is_rejected() {
    let engine = weight_engine(Arc::new(CountingEmbedder::default())).await;

    let mut accepted = Vec::new();
    for min_score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let result = engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl("title:rust")
                    .vector_min_score(min_score)
                    .build(),
            )
            .await;
        let rejected = matches!(
            &result,
            Err(LaurusError::InvalidArgument(message))
                if message.contains("vector min_score must be finite")
        );
        if !rejected {
            accepted.push(format!("{min_score} -> {result:?}"));
        }
    }
    assert!(
        accepted.is_empty(),
        "not rejected:\n{}",
        accepted.join("\n")
    );
}

/// Issue #1552: a weight of `0` gives every hit a score of `0`, and `2`
/// doubles the scores of a weight of `1`.
#[tokio::test(flavor = "multi_thread")]
async fn a_zero_or_positive_weight_scales_the_scores() {
    let engine = weight_engine(Arc::new(CountingEmbedder::default())).await;
    let scores = |weight| {
        let engine = &engine;
        async move {
            let mut hits: Vec<(String, f32)> = engine
                .search(
                    SearchRequestBuilder::new()
                        .vector_query(weighted(weight))
                        .build(),
                )
                .await
                .unwrap()
                .into_iter()
                .map(|hit| (hit.id, hit.score))
                .collect();
            hits.sort_by(|a, b| a.0.cmp(&b.0));
            hits
        }
    };

    let one = scores(1.0).await;
    assert_eq!(one.len(), 2, "{one:?}");
    assert!(one.iter().all(|(_, score)| *score > 0.0), "{one:?}");
    let zero = scores(0.0).await;
    assert_eq!(
        zero.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
        vec!["doc1", "doc2"]
    );
    assert!(zero.iter().all(|(_, score)| *score == 0.0), "{zero:?}");
    let two = scores(2.0).await;
    assert_eq!(
        two.iter().map(|(id, _)| id).collect::<Vec<_>>(),
        one.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    for ((id, score), (_, doubled)) in one.iter().zip(&two) {
        assert!(
            (doubled - 2.0 * score).abs() < 1e-5,
            "{id}: {score} -> {doubled}"
        );
    }
}
