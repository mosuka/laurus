//! Integration tests for #1084: `FieldOption::base_weight` wiring.
//!
//! `base_weight` is a per-field multiplicative factor on similarity scores,
//! applied only when a query is routed to one or more specific fields
//! (`QueryVector.fields` / `VectorSearchParams.fields`) — it has no effect
//! on a field-less fanout query (`fields: None` everywhere), which is out
//! of scope for this issue.
//!
//! Each test uses two or more HNSW fields sharing the same dimension, and
//! anchors on documents that carry the EXACT SAME vector as the query, so
//! int8 quantization error is identical across fields and any score
//! difference is attributable to `base_weight` alone (see #773 for why
//! exact-recall assertions on a randomized HNSW graph are avoided
//! elsewhere in this crate).

use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::store::request::{
    QueryVector, VectorScoreMode, VectorSearchParams, VectorSearchRequest,
};
use laurus::vector::{FieldOption, VectorIndexConfig, VectorStore};
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

fn hnsw(dimension: usize, base_weight: f32) -> FieldOption {
    FieldOption::Hnsw(HnswOption {
        dimension,
        distance: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 100,
        default_ef_search: None,
        base_weight,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
        pq_codebook_path: None,
    })
}

/// A store whose fields are given as `(name, base_weight)` pairs, each an
/// independent HNSW field of the given dimension.
async fn setup_store(dimension: usize, fields: &[(&str, f32)]) -> VectorStore {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut field_configs = HashMap::new();
    for (name, base_weight) in fields {
        field_configs.insert(
            name.to_string(),
            VectorFieldConfig {
                vector: Some(hnsw(dimension, *base_weight)),
                lexical: None,
            },
        );
    }
    let config = VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockEmbedder { dimension }),
        default_fields: fields.iter().map(|(n, _)| n.to_string()).collect(),
        metadata: HashMap::new(),
        deletion_config: laurus::DeletionConfig::default(),
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    };
    VectorStore::new(storage, config).unwrap()
}

async fn upsert(store: &VectorStore, doc_id: u64, field: &str, vector: Vec<f32>) {
    let doc = Document::builder()
        .add_field(field, DataValue::Vector(vector))
        .build();
    store
        .upsert_document_by_internal_id(doc_id, doc)
        .await
        .unwrap();
}

fn search_fields(store: &VectorStore, vector: Vec<f32>, fields: &[&str]) -> Vec<(u64, f32)> {
    let request = VectorSearchRequest {
        query: laurus::vector::VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vector),
            weight: 1.0,
            fields: Some(fields.iter().map(|f| f.to_string()).collect()),
        }]),
        params: VectorSearchParams {
            limit: 10,
            score_mode: VectorScoreMode::WeightedSum,
            ..Default::default()
        },
    };
    let results = store.search(request).unwrap();
    results
        .hits
        .into_iter()
        .map(|h| (h.doc_id, h.score))
        .collect()
}

/// #1084: base_weight reorders hits across vector fields when the same
/// query is routed to both fields.
#[tokio::test(flavor = "multi_thread")]
async fn base_weight_reorders_hits_across_vector_fields() {
    let store = setup_store(3, &[("a_vec", 1.0), ("b_vec", 3.0)]).await;
    upsert(&store, 1, "a_vec", vec![1.0, 0.0, 0.0]).await;
    upsert(&store, 2, "b_vec", vec![1.0, 0.0, 0.0]).await;
    store.commit().await.unwrap();

    let hits = search_fields(&store, vec![1.0, 0.0, 0.0], &["a_vec", "b_vec"]);
    let by_id: HashMap<u64, f32> = hits.into_iter().collect();

    let score_a = *by_id.get(&1).expect("doc 1 (a_vec) must be a hit");
    let score_b = *by_id.get(&2).expect("doc 2 (b_vec) must be a hit");

    assert!(
        score_b > score_a,
        "b_vec's base_weight (3.0) must outrank a_vec's (1.0): a={score_a}, b={score_b}"
    );
    let ratio = score_b / score_a;
    assert!(
        (ratio - 3.0).abs() < 0.05,
        "score ratio must track the base_weight ratio (~3.0), got {ratio}"
    );
}

/// #1084: a field whose `base_weight` is unset (the `HnswOption` default,
/// `1.0`) scores identically to one with an explicit `1.0`.
#[tokio::test(flavor = "multi_thread")]
async fn base_weight_defaults_to_one_when_unset() {
    let default_weight = HnswOption::default().base_weight;
    assert_eq!(default_weight, 1.0, "HnswOption::default() must be 1.0");

    let store = setup_store(3, &[("default_vec", default_weight), ("explicit_vec", 1.0)]).await;
    upsert(&store, 1, "default_vec", vec![1.0, 0.0, 0.0]).await;
    upsert(&store, 2, "explicit_vec", vec![1.0, 0.0, 0.0]).await;
    store.commit().await.unwrap();

    let hits = search_fields(
        &store,
        vec![1.0, 0.0, 0.0],
        &["default_vec", "explicit_vec"],
    );
    let by_id: HashMap<u64, f32> = hits.into_iter().collect();

    let score_default = *by_id.get(&1).expect("doc 1 must be a hit");
    let score_explicit = *by_id.get(&2).expect("doc 2 must be a hit");
    assert!(
        (score_default - score_explicit).abs() < 1e-6,
        "default and explicit 1.0 base_weight must score identically: \
         default={score_default}, explicit={score_explicit}"
    );
}

/// #1084: non-positive or non-finite `base_weight` values are clamped to
/// `1.0` rather than zeroing, negating, or NaN-ing out every hit from the
/// field — this rescues indexes whose schema already has `base_weight:
/// 0.0` persisted from before the proto/gateway fix.
#[tokio::test(flavor = "multi_thread")]
async fn non_positive_base_weight_falls_back_to_one() {
    let store = setup_store(
        3,
        &[
            ("baseline", 1.0),
            ("zero", 0.0),
            ("negative", -1.0),
            ("nan", f32::NAN),
        ],
    )
    .await;
    for (doc_id, field) in ["baseline", "zero", "negative", "nan"]
        .into_iter()
        .enumerate()
    {
        upsert(&store, doc_id as u64, field, vec![1.0, 0.0, 0.0]).await;
    }
    store.commit().await.unwrap();

    let baseline_hits = search_fields(&store, vec![1.0, 0.0, 0.0], &["baseline"]);
    let baseline_score = baseline_hits
        .first()
        .expect("baseline field must have a hit")
        .1;

    for field in ["zero", "negative", "nan"] {
        let hits = search_fields(&store, vec![1.0, 0.0, 0.0], &[field]);
        let score = hits
            .first()
            .unwrap_or_else(|| panic!("{field} field must have a hit"))
            .1;
        assert!(
            (score - baseline_score).abs() < 1e-6,
            "{field}'s base_weight must clamp to 1.0 like baseline: \
             baseline={baseline_score}, {field}={score}"
        );
    }
}

/// #1084: `min_score` compares the raw, unweighted similarity — a field
/// with a small `base_weight` that would push the final score below
/// `min_score` must still return the hit, because filtering happens
/// before weighting (matching the pre-existing `QueryVector.weight`
/// contract).
#[tokio::test(flavor = "multi_thread")]
async fn min_score_compares_unweighted_similarity() {
    let store = setup_store(3, &[("low_weight", 0.1)]).await;
    upsert(&store, 1, "low_weight", vec![1.0, 0.0, 0.0]).await;
    store.commit().await.unwrap();

    let request = VectorSearchRequest {
        query: laurus::vector::VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![1.0, 0.0, 0.0]),
            weight: 1.0,
            fields: Some(vec!["low_weight".to_string()]),
        }]),
        params: VectorSearchParams {
            limit: 10,
            score_mode: VectorScoreMode::WeightedSum,
            // The raw similarity for an identical vector is close to 1.0;
            // the base_weight-adjusted score is close to 0.1. A min_score
            // between the two proves which one the filter uses.
            min_score: 0.5,
            ..Default::default()
        },
    };
    let results = store.search(request).unwrap();
    assert_eq!(
        results.hits.len(),
        1,
        "min_score must be compared against the unweighted similarity (~1.0), \
         not the base_weight-adjusted score (~0.1): got {:?}",
        results.hits
    );
    assert!(
        results.hits[0].score < 0.5,
        "the returned score must still be base_weight-adjusted: {:?}",
        results.hits[0]
    );
}

/// #1084: a store built via `with_index_type_config` (no collection-wide
/// `VectorIndexConfig`, hence no `base_weights` entries) must not panic —
/// every field falls back to `1.0`.
#[tokio::test(flavor = "multi_thread")]
async fn single_index_store_ignores_base_weight() {
    use laurus::vector::index::config::{FlatIndexConfig, VectorIndexTypeConfig};

    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = VectorIndexTypeConfig::Flat(FlatIndexConfig {
        dimension: 3,
        ..Default::default()
    });
    let store = VectorStore::with_index_type_config(storage, config).unwrap();

    let doc = Document::builder()
        .add_field("vector", DataValue::Vector(vec![1.0, 0.0, 0.0]))
        .build();
    store.upsert_document_by_internal_id(1, doc).await.unwrap();
    store.commit().await.unwrap();

    let hits = search_fields(&store, vec![1.0, 0.0, 0.0], &["vector"]);
    assert!(
        !hits.is_empty(),
        "search must not panic and must find the doc"
    );
}
