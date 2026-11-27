use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use laurus::Engine;
use laurus::LaurusError;
use laurus::lexical::TermQuery;
use laurus::lexical::TextOption;
use laurus::storage::memory::MemoryStorage;
use laurus::vector::FlatOption;
use laurus::vector::QueryVector;
use laurus::vector::Vector;
use laurus::{DataValue, Document};
use laurus::{EmbedInput, EmbedInputType, Embedder};
use laurus::{FieldOption, Schema};
use laurus::{FusionAlgorithm, LexicalSearchQuery, SearchRequestBuilder, VectorSearchQuery};

#[derive(Debug, Clone)]
struct MockEmbedder {
    vectors: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}

impl MockEmbedder {
    fn new() -> Self {
        Self {
            vectors: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn add(&self, text: &str, vector: Vec<f32>) {
        self.vectors
            .lock()
            .unwrap()
            .insert(text.to_string(), vector);
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(
        &self,
        input: &EmbedInput<'_>,
    ) -> std::result::Result<laurus::vector::core::vector::Vector, LaurusError> {
        match input {
            EmbedInput::Text(text) => {
                let map = self.vectors.lock().unwrap();
                if let Some(vec) = map.get(*text) {
                    Ok(laurus::vector::core::vector::Vector::new(vec.clone()))
                } else {
                    Err(LaurusError::invalid_argument(format!(
                        "MockEmbedder: unknown text '{}'",
                        text
                    )))
                }
            }
            _ => Err(LaurusError::invalid_argument(
                "MockEmbedder only supports text",
            )),
        }
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn supports_text(&self) -> bool {
        true
    }
    fn supports_image(&self) -> bool {
        false
    }
    fn name(&self) -> &str {
        "MockEmbedder"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

async fn create_hybrid_engine() -> std::result::Result<Engine, Box<dyn std::error::Error>> {
    // Embedder setup
    let embedder = Arc::new(MockEmbedder::new());
    embedder.add("apple", vec![0.9, 0.1, 0.0]);
    embedder.add("banana", vec![0.0, 1.0, 0.0]);

    // Schema with separate fields for lexical and vector
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field(
            "title_vec",
            FieldOption::Flat(FlatOption {
                dimension: 3,
                ..Default::default()
            }),
        )
        .build();

    let storage = Arc::new(MemoryStorage::new(Default::default()));

    // Use simple embedder directly (PerFieldEmbedder is not supported)
    let engine = Engine::builder(storage, schema)
        .embedder(embedder)
        .build()
        .await?;
    Ok(engine)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hybrid_search_unification() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let engine = create_hybrid_engine().await?;

    // Index Doc 1: "apple" - both lexical and vector fields
    let payload1 = Document::builder()
        .add_field("title", DataValue::Text("apple".into()))
        .add_field("title_vec", DataValue::Text("apple".into()))
        .build();

    engine.add_document("1", payload1).await?;

    // Index Doc 2: "banana" - both lexical and vector fields
    let payload2 = Document::builder()
        .add_field("title", DataValue::Text("banana".into()))
        .add_field("title_vec", DataValue::Text("banana".into()))
        .build();

    engine.add_document("2", payload2).await?;

    engine.commit().await?;

    // Test 1: Vector Search (query closest to apple [0.9, 0.1, 0.0])
    let req_vector = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![0.9, 0.1, 0.0]),
            weight: 1.0,
            fields: None,
        }]))
        .limit(10)
        .build();

    let res_vector = engine.search(req_vector).await?.results;
    assert!(!res_vector.is_empty(), "Vector search should return hits");
    let top_id = res_vector[0].id.clone();

    // Test 2: Lexical Search ("banana")
    let req_lexical = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "title", "banana",
        ))))
        .limit(10)
        .build();

    let res_lexical = engine.search(req_lexical).await?.results;
    assert!(!res_lexical.is_empty(), "Lexical search should return hits");
    assert_ne!(
        res_lexical[0].id, top_id,
        "Banana should be different from Apple"
    );

    // Test 3: Hybrid Search (RRF)
    let req_hybrid = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![0.0, 1.0, 0.0]),
            weight: 1.0,
            fields: None,
        }]))
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "title", "banana",
        ))))
        .fusion_algorithm(FusionAlgorithm::RRF { k: 60.0 })
        .limit(10)
        .build();

    let res_hybrid = engine.search(req_hybrid).await?.results;
    assert!(!res_hybrid.is_empty());

    assert_eq!(
        res_hybrid[0].id, res_lexical[0].id,
        "Banana should win in hybrid search"
    );

    Ok(())
}
