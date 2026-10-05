//! Integration tests for the query-time embedding cache (issue #678).
//!
//! These tests use a `CountingEmbedder` that records how many times `embed`
//! is called. Documents are indexed with pre-computed vectors (so document
//! ingestion does not embed), leaving the query path as the only source of
//! `embed` calls. The tests then assert the cache collapses repeated
//! identical query embeddings into a single `embed` call.

use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use laurus::Engine;
use laurus::LaurusError;
use laurus::SearchRequestBuilder;
use laurus::storage::memory::MemoryStorage;
use laurus::vector::FlatOption;
use laurus::vector::Vector;
use laurus::{DataValue, Document};
use laurus::{DistanceMetric, EmbedInput, EmbedInputType, Embedder, UpdateFieldOptions};
use laurus::{FieldOption, Schema};

/// Embedder that maps known texts to vectors and counts `embed` calls.
#[derive(Debug)]
struct CountingEmbedder {
    vectors: Mutex<HashMap<String, Vec<f32>>>,
    calls: AtomicUsize,
}

impl CountingEmbedder {
    fn new() -> Self {
        Self {
            vectors: Mutex::new(HashMap::new()),
            calls: AtomicUsize::new(0),
        }
    }

    fn add(&self, text: &str, vector: Vec<f32>) {
        self.vectors
            .lock()
            .unwrap()
            .insert(text.to_string(), vector);
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl Embedder for CountingEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> std::result::Result<Vector, LaurusError> {
        match input {
            EmbedInput::Text(text) => {
                self.calls.fetch_add(1, Ordering::Relaxed);
                let map = self.vectors.lock().unwrap();
                map.get(*text)
                    .map(|v| Vector::new(v.clone()))
                    .ok_or_else(|| {
                        LaurusError::invalid_argument(format!(
                            "CountingEmbedder: unknown text '{text}'"
                        ))
                    })
            }
            _ => Err(LaurusError::invalid_argument(
                "CountingEmbedder only supports text",
            )),
        }
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn supports_text(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "counting"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn schema() -> Schema {
    Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Flat(FlatOption::default().dimension(3)),
        )
        .build()
}

async fn index_corpus(engine: &Engine) {
    // Pre-computed vectors → no ingestion embedding, so the only embed calls
    // come from queries.
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("embedding", DataValue::Vector(vec![1.0, 0.0, 0.0]))
                .build(),
        )
        .await
        .unwrap();
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("embedding", DataValue::Vector(vec![0.0, 1.0, 0.0]))
                .build(),
        )
        .await
        .unwrap();
    engine.commit().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cache_disabled_by_default() {
    let storage = Arc::new(MemoryStorage::new(Default::default()));
    let embedder = Arc::new(CountingEmbedder::new());
    embedder.add("apple", vec![1.0, 0.0, 0.0]);

    // `Engine::new` (no builder knob) → cache disabled.
    let engine = Engine::builder(storage, schema())
        .embedder(embedder.clone())
        .build()
        .await
        .unwrap();
    index_corpus(&engine).await;

    let base = embedder.call_count();
    for _ in 0..3 {
        engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl("embedding:\"apple\"")
                    .build(),
            )
            .await
            .unwrap();
    }
    // No cache → one embed per search.
    assert_eq!(embedder.call_count() - base, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cache_hit_avoids_reembed() {
    let storage = Arc::new(MemoryStorage::new(Default::default()));
    let embedder = Arc::new(CountingEmbedder::new());
    embedder.add("apple", vec![1.0, 0.0, 0.0]);

    let engine = Engine::builder(storage, schema())
        .embedder(embedder.clone())
        .embedding_cache_capacity(16)
        .build()
        .await
        .unwrap();
    index_corpus(&engine).await;

    let base = embedder.call_count();
    for _ in 0..5 {
        engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl("embedding:\"apple\"")
                    .build(),
            )
            .await
            .unwrap();
    }
    // Cache → identical query embedded only once.
    assert_eq!(embedder.call_count() - base, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cache_distinguishes_payload() {
    let storage = Arc::new(MemoryStorage::new(Default::default()));
    let embedder = Arc::new(CountingEmbedder::new());
    embedder.add("apple", vec![1.0, 0.0, 0.0]);
    embedder.add("banana", vec![0.0, 1.0, 0.0]);

    let engine = Engine::builder(storage, schema())
        .embedder(embedder.clone())
        .embedding_cache_capacity(16)
        .build()
        .await
        .unwrap();
    index_corpus(&engine).await;

    let base = embedder.call_count();
    // Two distinct payloads, each searched twice → two embeds total.
    for _ in 0..2 {
        engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl("embedding:\"apple\"")
                    .build(),
            )
            .await
            .unwrap();
        engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl("embedding:\"banana\"")
                    .build(),
            )
            .await
            .unwrap();
    }
    assert_eq!(embedder.call_count() - base, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cache_hit_returns_correct_results() {
    let storage = Arc::new(MemoryStorage::new(Default::default()));
    let embedder = Arc::new(CountingEmbedder::new());
    embedder.add("apple", vec![1.0, 0.0, 0.0]);

    let engine = Engine::builder(storage, schema())
        .embedder(embedder.clone())
        .embedding_cache_capacity(16)
        .build()
        .await
        .unwrap();
    index_corpus(&engine).await;

    // First search populates the cache; second hits it. Both must return
    // doc1 (closest to [1,0,0]) as the top result.
    let first = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl("embedding:\"apple\"")
                .build(),
        )
        .await
        .unwrap();
    let second = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl("embedding:\"apple\"")
                .build(),
        )
        .await
        .unwrap();

    assert!(!first.is_empty());
    assert_eq!(first[0].id, "doc1");
    assert_eq!(first.len(), second.len());
    assert_eq!(first[0].id, second[0].id);
}

/// Adding, deleting or rebuilding a vector field may change the embedder
/// behind a field name, so each of them empties the cache (Issue #1349).
#[tokio::test(flavor = "multi_thread")]
async fn test_vector_field_changes_clear_the_cache() {
    let storage = Arc::new(MemoryStorage::new(Default::default()));
    let embedder = Arc::new(CountingEmbedder::new());
    embedder.add("apple", vec![1.0, 0.0, 0.0]);

    let engine = Engine::builder(storage, schema())
        .embedder(embedder.clone())
        .embedding_cache_capacity(16)
        .build()
        .await
        .unwrap();
    index_corpus(&engine).await;

    let search = || async {
        engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl("embedding:\"apple\"")
                    .build(),
            )
            .await
            .unwrap();
    };
    let base = embedder.call_count();
    search().await;
    search().await;
    assert_eq!(embedder.call_count() - base, 1);

    engine
        .add_field(
            "other",
            FieldOption::Flat(FlatOption::default().dimension(3)),
        )
        .await
        .unwrap();
    search().await;
    assert_eq!(
        embedder.call_count() - base,
        2,
        "add_field clears the cache"
    );

    engine.delete_field("other").await.unwrap();
    search().await;
    assert_eq!(
        embedder.call_count() - base,
        3,
        "delete_field clears the cache"
    );

    engine
        .update_field(
            "embedding",
            FieldOption::Flat(
                FlatOption::default()
                    .dimension(3)
                    .distance(DistanceMetric::DotProduct),
            ),
            UpdateFieldOptions {
                reindex: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    search().await;
    assert_eq!(
        embedder.call_count() - base,
        4,
        "update_field clears the cache"
    );
}
