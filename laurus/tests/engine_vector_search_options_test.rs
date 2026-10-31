//! Issue #1342: `SearchRequestBuilder`'s vector options reach the vector
//! search, whether the vector part comes from pre-embedded vectors or from
//! a DSL vector clause.
//!
//! Issue #1552: `Engine::search` rejects a query vector weight that is not
//! finite or is negative, and a vector `min_score` that is not finite,
//! whatever the data and before anything is embedded.
//!
//! Issue #1573: it also rejects a query vector with a `NaN` or infinite
//! component, whether the request gives it or the embedder returns it.
//!
//! Issue #1575: and a query vector whose dimension differs from a field it
//! is searched in, whether the request gives it or the embedder returns it.

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
/// The vectors are given, so `embedder` embeds queries only.
async fn weight_engine(embedder: Arc<dyn Embedder>) -> Engine {
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

/// Builds a search request around a query vector made from a value (its
/// weight, or one of its components).
type RequestWith = Box<dyn Fn(f32) -> SearchRequest>;

/// The requests that return early without searching the vectors, each
/// around `query(value)`: `Vector`, `Hybrid` and `DslWithVectors` under a
/// filter that matches nothing, and `Vector` grouped with `limit` `0`.
fn early_return_shapes(query: fn(f32) -> VectorSearchQuery) -> [(&'static str, RequestWith); 4] {
    let with = move |value| SearchRequestBuilder::new().vector_query(query(value));
    [
        (
            "vector, filter matches nothing",
            Box::new(move |value| with(value).filter_query(nothing()).build()),
        ),
        (
            "hybrid, filter matches nothing",
            Box::new(move |value| {
                with(value)
                    .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                        "title", "rust",
                    ))))
                    .filter_query(nothing())
                    .build()
            }),
        ),
        (
            "dsl with vectors, filter matches nothing",
            Box::new(move |value| {
                with(value)
                    .query_dsl("title:rust")
                    .filter_query(nothing())
                    .build()
            }),
        ),
        (
            "vector, grouped with limit 0",
            Box::new(move |value| with(value).group_by(GroupBy::new("_id")).limit(0).build()),
        ),
    ]
}

/// Issue #1552: an out-of-range weight of a `Vector`, `Hybrid` or
/// `DslWithVectors` query is rejected even where the search returns early
/// without searching the vectors: under a filter that matches nothing, or
/// grouped with `limit` `0`.
#[tokio::test(flavor = "multi_thread")]
async fn a_bad_query_vector_weight_is_rejected_whatever_the_data() {
    let engine = weight_engine(Arc::new(CountingEmbedder::default())).await;
    let shapes = early_return_shapes(weighted);

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

/// The query vector `[1, component]` on `vec`, with weight `1`.
fn with_component(component: f32) -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0, component]),
        weight: 1.0,
        fields: Some(vec!["vec".to_string()]),
    }])
}

/// Components that are not finite.
const BAD_COMPONENTS: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

/// Whether `result` is the `InvalidArgument` of the second component of the
/// request's first query vector.
fn is_component_error(result: &Result<Vec<SearchResult>>) -> bool {
    matches!(
        result,
        Err(LaurusError::InvalidArgument(message))
            if message.contains("query vector components must be finite, got ")
                && message.contains(" at component 1: query vector 0")
    )
}

/// Issue #1573: a query vector with a `NaN` or infinite component is
/// rejected in every shape of request: where the search returns early
/// without searching the vectors, and where there is data to search. A
/// finite component, however large, is not an error.
#[tokio::test(flavor = "multi_thread")]
async fn a_non_finite_query_vector_component_is_rejected_whatever_the_data() {
    let engine = weight_engine(Arc::new(CountingEmbedder::default())).await;
    let with = |component| SearchRequestBuilder::new().vector_query(with_component(component));
    let vector: RequestWith = Box::new(move |component| with(component).build());
    let hybrid: RequestWith = Box::new(move |component| {
        with(component)
            .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                "title", "rust",
            ))))
            .build()
    });
    let mut shapes = Vec::from(early_return_shapes(with_component));
    shapes.push(("vector", vector));
    shapes.push(("hybrid", hybrid));

    let mut accepted = Vec::new();
    for (shape, request) in &shapes {
        for component in BAD_COMPONENTS {
            let result = engine.search(request(component)).await;
            if !is_component_error(&result) {
                accepted.push(format!("{shape}: {component} -> {result:?}"));
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "not rejected:\n{}",
        accepted.join("\n")
    );

    engine
        .search(with(f32::MAX).build())
        .await
        .expect("a finite component is not an error");
}

/// Embeds any text as the vector it holds.
#[derive(Debug)]
struct FixedEmbedder(Vec<f32>);

#[async_trait]
impl Embedder for FixedEmbedder {
    async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
        Ok(Vector::new(self.0.clone()))
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "fixed"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A text payload on `vec`.
fn payload() -> VectorSearchQuery {
    VectorSearchQuery::Payloads(vec![QueryPayload::new(
        "vec",
        DataValue::Text("payload text".to_string()),
    )])
}

/// Issue #1573: a query vector the embedder returns with a `NaN` component
/// is rejected as soon as it is embedded, naming the field: a DSL vector
/// clause, also under a filter that matches nothing (the clause is embedded
/// while the DSL is parsed, before the filter runs), and a payload.
#[tokio::test(flavor = "multi_thread")]
async fn a_non_finite_embedded_query_vector_is_rejected() {
    let engine = weight_engine(Arc::new(FixedEmbedder(vec![f32::NAN, 0.0]))).await;
    let requests = [
        (
            "dsl clause",
            SearchRequestBuilder::new()
                .query_dsl("vec:\"dsl text\"")
                .build(),
        ),
        (
            "dsl clause, filter matches nothing",
            SearchRequestBuilder::new()
                .query_dsl("vec:\"dsl text\"")
                .filter_query(nothing())
                .build(),
        ),
        (
            "dsl with a payload, filter matches nothing",
            SearchRequestBuilder::new()
                .vector_query(payload())
                .query_dsl("title:rust")
                .filter_query(nothing())
                .build(),
        ),
        (
            "payload",
            SearchRequestBuilder::new().vector_query(payload()).build(),
        ),
    ];

    let mut accepted = Vec::new();
    for (shape, request) in requests {
        let result = engine.search(request).await;
        let rejected = matches!(
            &result,
            Err(LaurusError::InvalidArgument(message))
                if message.contains(
                    "query vector components must be finite, got NaN at component 0: \
                     the embedding of a query for field \"vec\""
                )
        );
        if !rejected {
            accepted.push(format!("{shape} -> {result:?}"));
        }
    }
    assert!(
        accepted.is_empty(),
        "not rejected:\n{}",
        accepted.join("\n")
    );
}

/// The query vector of `dimension` ones on `vec`. The dimension is given as
/// an `f32` so that it fits `early_return_shapes`.
fn named_of_dimension(dimension: f32) -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0; dimension as usize]),
        weight: 1.0,
        fields: Some(vec!["vec".to_string()]),
    }])
}

/// The query vector of `dimension` ones, naming no field.
fn unnamed_of_dimension(dimension: usize) -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0; dimension]),
        weight: 1.0,
        fields: None,
    }])
}

/// Dimensions other than `vec`'s `2`: one shorter, one longer.
const BAD_DIMENSIONS: [f32; 2] = [1.0, 3.0];

/// Whether `result` is the `InvalidArgument` of `subject` having `dimension`
/// where `vec` has `2`.
fn is_dimension_error(result: &Result<Vec<SearchResult>>, subject: &str, dimension: usize) -> bool {
    let expected = format!("{subject} has dimension {dimension}, but field 'vec' has 2");
    matches!(
        result,
        Err(LaurusError::InvalidArgument(message)) if *message == expected
    )
}

/// Issue #1575: a query vector of another dimension than the field it names,
/// or than the field the request's `vector_fields` pick for it, is rejected
/// in every shape of request: where the search returns early without
/// searching the vectors, and where there is data to search. A query vector
/// that names no field, with no `vector_fields`, is still searched only in
/// the fields of its own dimension.
#[tokio::test(flavor = "multi_thread")]
async fn a_query_vector_of_another_dimension_is_rejected_whatever_the_data() {
    let engine = weight_engine(Arc::new(CountingEmbedder::default())).await;
    let named = |dimension| SearchRequestBuilder::new().vector_query(named_of_dimension(dimension));
    let picked = |dimension: f32| {
        SearchRequestBuilder::new()
            .vector_query(unnamed_of_dimension(dimension as usize))
            .vector_fields(vec![FieldSelector::Exact("vec".to_string())])
    };
    let mut shapes = Vec::from(early_return_shapes(named_of_dimension));
    shapes.push((
        "vector",
        Box::new(move |dimension| named(dimension).build()),
    ));
    shapes.push((
        "hybrid",
        Box::new(move |dimension| {
            named(dimension)
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "title", "rust",
                ))))
                .build()
        }),
    ));
    shapes.push((
        "vector_fields, filter matches nothing",
        Box::new(move |dimension| picked(dimension).filter_query(nothing()).build()),
    ));
    shapes.push((
        "vector_fields, grouped with limit 0",
        Box::new(move |dimension| {
            picked(dimension)
                .group_by(GroupBy::new("_id"))
                .limit(0)
                .build()
        }),
    ));
    shapes.push((
        "vector_fields",
        Box::new(move |dimension| picked(dimension).build()),
    ));

    let mut accepted = Vec::new();
    for (shape, request) in &shapes {
        for dimension in BAD_DIMENSIONS {
            let result = engine.search(request(dimension)).await;
            if !is_dimension_error(&result, "query vector 0", dimension as usize) {
                accepted.push(format!("{shape}: {dimension} -> {result:?}"));
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "not rejected:\n{}",
        accepted.join("\n")
    );

    let unnamed = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(unnamed_of_dimension(3))
                .build(),
        )
        .await
        .expect("a query vector that names no field is not checked");
    assert!(unnamed.is_empty(), "no field has 3 dimensions: {unnamed:?}");
    let fitting = engine.search(named(2.0).build()).await.unwrap();
    assert_eq!(sorted_ids(&fitting), vec!["doc1", "doc2"]);
}

/// Issue #1575: a query vector the embedder returns with another dimension
/// than its field is rejected: a DSL vector clause and a `DslWithVectors`
/// payload once the DSL is parsed, also under a filter that matches nothing
/// (before the filter runs), and a `Vector` payload once it is embedded.
#[tokio::test(flavor = "multi_thread")]
async fn an_embedded_query_vector_of_another_dimension_is_rejected() {
    let engine = weight_engine(Arc::new(FixedEmbedder(vec![1.0, 0.0, 0.0]))).await;
    let requests = [
        (
            "dsl clause",
            SearchRequestBuilder::new()
                .query_dsl("vec:\"dsl text\"")
                .build(),
        ),
        (
            "dsl clause, filter matches nothing",
            SearchRequestBuilder::new()
                .query_dsl("vec:\"dsl text\"")
                .filter_query(nothing())
                .build(),
        ),
        (
            "dsl with a payload, filter matches nothing",
            SearchRequestBuilder::new()
                .vector_query(payload())
                .query_dsl("title:rust")
                .filter_query(nothing())
                .build(),
        ),
        (
            "payload",
            SearchRequestBuilder::new().vector_query(payload()).build(),
        ),
    ];

    let mut accepted = Vec::new();
    for (shape, request) in requests {
        let result = engine.search(request).await;
        if !is_dimension_error(&result, "the embedding of a query", 3) {
            accepted.push(format!("{shape} -> {result:?}"));
        }
    }
    assert!(
        accepted.is_empty(),
        "not rejected:\n{}",
        accepted.join("\n")
    );
}
