//! Integration tests for per-field vector search routing (issue #676).
//!
//! The store holds two HNSW vector fields, `img_vec` and `txt_vec`. Each
//! document carries a vector for exactly one field, so a query routed to a
//! field can only return that field's documents. The tests exercise both
//! routing inputs — `QueryVector.fields` (per-query) and
//! `VectorSearchParams.fields` (request-level `Exact` / `Prefix`) — and the
//! "no fields → search all" default.

use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::store::request::{
    FieldSelector, QueryVector, VectorScoreMode, VectorSearchParams, VectorSearchRequest,
};
use laurus::vector::{FieldOption, VectorIndexConfig};
use laurus::{DataValue, Document};
use laurus::{EmbedInput, EmbedInputType, Embedder};
use laurus::{LaurusError, Result};

#[derive(Debug)]
struct MockEmbedder {
    dimension: usize,
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(_) => Ok(Vector::new(vec![0.0; self.dimension])),
            _ => Err(LaurusError::invalid_argument("text only")),
        }
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

fn hnsw(dimension: usize) -> FieldOption {
    FieldOption::Hnsw(HnswOption {
        dimension,
        distance: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 100,
        default_ef_search: None,
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
        pq_codebook_path: None,
    })
}

/// Store with two HNSW fields `img_vec` and `txt_vec`, each holding two docs.
/// doc1/doc2 → `img_vec`; doc3/doc4 → `txt_vec`.
async fn setup_store(dimension: usize) -> laurus::vector::VectorStore {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut field_configs = std::collections::HashMap::new();
    for name in ["img_vec", "txt_vec"] {
        field_configs.insert(
            name.to_string(),
            VectorFieldConfig {
                vector: Some(hnsw(dimension)),
                lexical: None,
            },
        );
    }
    let config = VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockEmbedder { dimension }),
        default_fields: vec!["img_vec".to_string(), "txt_vec".to_string()],
        metadata: std::collections::HashMap::new(),
        deletion_config: laurus::DeletionConfig::default(),
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    };
    let store = laurus::vector::VectorStore::new(storage, config).unwrap();

    let docs = [
        (1u64, "img_vec", vec![1.0, 0.0, 0.0]),
        (2, "img_vec", vec![0.0, 1.0, 0.0]),
        (3, "txt_vec", vec![1.0, 0.0, 0.0]),
        (4, "txt_vec", vec![0.0, 1.0, 0.0]),
    ];
    for (id, field, v) in docs {
        let doc = Document::builder()
            .add_field(field, DataValue::Vector(v))
            .build();
        store.upsert_document_by_internal_id(id, doc).await.unwrap();
    }
    store.commit().await.unwrap();
    store
}

fn request(
    query_fields: Option<Vec<String>>,
    params_fields: Option<Vec<FieldSelector>>,
) -> VectorSearchRequest {
    VectorSearchRequest {
        query: laurus::vector::VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![1.0, 0.0, 0.0]),
            weight: 1.0,
            fields: query_fields,
        }]),
        params: VectorSearchParams {
            limit: 10,
            score_mode: VectorScoreMode::WeightedSum,
            fields: params_fields,
            ..Default::default()
        },
    }
}

fn doc_ids(results: &laurus::vector::VectorSearchResults) -> std::collections::HashSet<u64> {
    results.hits.iter().map(|h| h.doc_id).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_vector_fields_routes() {
    let store = setup_store(3).await;
    // Per-query field = img_vec → only img_vec docs (1, 2).
    let results = store
        .search(request(Some(vec!["img_vec".into()]), None))
        .unwrap();
    let ids = doc_ids(&results);
    // Routing invariant (deterministic): the result contains only img_vec docs
    // and never leaks txt_vec docs. Whether *every* img_vec doc comes back is
    // an approximate-recall property of the randomized HNSW graph, so we assert
    // the routing guarantee (non-empty + subset) rather than exact recall, which
    // is environment-sensitively flaky (Issue #773).
    assert!(
        !ids.is_empty(),
        "routing must return at least one hit: {ids:?}"
    );
    assert!(
        ids.iter().all(|id| *id == 1 || *id == 2),
        "only img_vec docs (1, 2) may be returned: {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_params_fields_exact() {
    let store = setup_store(3).await;
    // Request-level Exact(txt_vec) → only txt_vec docs (3, 4).
    let results = store
        .search(request(
            None,
            Some(vec![FieldSelector::Exact("txt_vec".into())]),
        ))
        .unwrap();
    let ids = doc_ids(&results);
    // Routing invariant: only txt_vec docs may be returned (see #773 for why we
    // assert the routing guarantee instead of exact recall).
    assert!(
        !ids.is_empty(),
        "routing must return at least one hit: {ids:?}"
    );
    assert!(
        ids.iter().all(|id| *id == 3 || *id == 4),
        "only txt_vec docs (3, 4) may be returned: {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_params_fields_prefix() {
    let store = setup_store(3).await;
    // Prefix("img") → resolves to img_vec → only img_vec docs (1, 2).
    let results = store
        .search(request(
            None,
            Some(vec![FieldSelector::Prefix("img".into())]),
        ))
        .unwrap();
    let ids = doc_ids(&results);
    // Routing invariant: only img_vec docs may be returned (see #773).
    assert!(
        !ids.is_empty(),
        "routing must return at least one hit: {ids:?}"
    );
    assert!(
        ids.iter().all(|id| *id == 1 || *id == 2),
        "only img_vec docs (1, 2) may be returned: {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_no_fields_searches_all() {
    let store = setup_store(3).await;
    // No field selector anywhere → all fields searched (regression guard).
    let results = store.search(request(None, None)).unwrap();
    let ids = doc_ids(&results);
    // Both fields must be represented, proving neither was skipped. We anchor on
    // the docs identical to the query (doc 1 in img_vec, doc 3 in txt_vec), which
    // are the nearest in their field and reliably returned; requiring all four
    // docs would assert exact recall on a randomized HNSW graph (Issue #773).
    assert!(
        ids.iter().any(|id| *id == 1 || *id == 2),
        "img_vec field must be represented: {ids:?}"
    );
    assert!(
        ids.iter().any(|id| *id == 3 || *id == 4),
        "txt_vec field must be represented: {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_per_query_overrides_params() {
    let store = setup_store(3).await;
    // Per-query fields=[img_vec] must win over request-level Exact(txt_vec).
    let results = store
        .search(request(
            Some(vec!["img_vec".into()]),
            Some(vec![FieldSelector::Exact("txt_vec".into())]),
        ))
        .unwrap();
    let ids = doc_ids(&results);
    // Per-query img_vec wins → only img_vec docs may be returned, never the
    // request-level txt_vec docs (see #773 for the routing-vs-recall rationale).
    assert!(
        !ids.is_empty(),
        "routing must return at least one hit: {ids:?}"
    );
    assert!(
        ids.iter().all(|id| *id == 1 || *id == 2),
        "per-query img_vec wins, only docs (1, 2) may be returned: {ids:?}"
    );
}

/// [`setup_store`] plus doc 5, which holds a vector in BOTH fields:
/// `img_vec` matches the query exactly, `txt_vec` is orthogonal to it.
async fn setup_store_with_multi_field_doc(dimension: usize) -> laurus::vector::VectorStore {
    let store = setup_store(dimension).await;
    let doc = Document::builder()
        .add_field("img_vec", DataValue::Vector(vec![1.0, 0.0, 0.0]))
        .add_field("txt_vec", DataValue::Vector(vec![0.0, 1.0, 0.0]))
        .build();
    store.upsert_document_by_internal_id(5, doc).await.unwrap();
    store.commit().await.unwrap();
    store
}

fn hit_scores(results: &laurus::vector::VectorSearchResults, doc_id: u64) -> Vec<f32> {
    results
        .hits
        .iter()
        .filter(|h| h.doc_id == doc_id)
        .map(|h| h.score)
        .collect()
}

/// Issue #1343: a field-less query must not return a multi-field document
/// twice, and its score must be the best of the per-field scores (not their
/// sum).
#[tokio::test(flavor = "multi_thread")]
async fn test_no_fields_dedupes_multi_field_document() {
    let store = setup_store_with_multi_field_doc(3).await;

    let img_only = store
        .search(request(
            None,
            Some(vec![FieldSelector::Exact("img_vec".into())]),
        ))
        .unwrap();
    let img_scores = hit_scores(&img_only, 5);
    assert_eq!(
        img_scores.len(),
        1,
        "doc 5 must have exactly one img_vec hit: {img_scores:?}"
    );

    let txt_only = store
        .search(request(
            None,
            Some(vec![FieldSelector::Exact("txt_vec".into())]),
        ))
        .unwrap();
    let txt_scores = hit_scores(&txt_only, 5);
    assert_eq!(
        txt_scores.len(),
        1,
        "doc 5 must have exactly one txt_vec hit: {txt_scores:?}"
    );

    let fieldless = store.search(request(None, None)).unwrap();
    let fieldless_scores = hit_scores(&fieldless, 5);
    assert_eq!(
        fieldless_scores.len(),
        1,
        "a field-less query must return doc 5 exactly once, not once per \
         matching field: {fieldless_scores:?}"
    );
    let best = img_scores[0].max(txt_scores[0]);
    assert!(
        (fieldless_scores[0] - best).abs() < 1e-5,
        "the field-less score ({}) must equal the best per-field score \
         ({best}), not their sum",
        fieldless_scores[0]
    );
}
