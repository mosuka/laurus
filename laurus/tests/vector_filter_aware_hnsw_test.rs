//! Integration tests for filter-aware HNSW traversal (issue #645).
//!
//! The store holds one HNSW field `vec` whose documents lie on a 1-D angular
//! gradient: doc `i` points at angle `i * STEP` in the first two dimensions,
//! so cosine similarity to the query (angle 0) decreases monotonically with
//! `i` (nearest-neighbour order is ascending `i`).
//!
//! HNSW graph construction is randomised (`rand::rng()` per build), so the
//! *exact* set a search returns varies run to run. These tests therefore
//! assert only invariants that hold for any graph:
//!
//! - a filtered result is always a subset of the allow-set
//!   (`filter_excludes_non_matching`);
//! - a filtered result always contains at least what a post-filter on the
//!   same graph would yield (`filter_recall_at_least_post_filter`) — this is
//!   the recall guarantee #645 adds, since the filtered frontier explores a
//!   superset of the unfiltered frontier;
//! - an empty allow-set yields no hits; an absent filter still returns a full
//!   page.

use async_trait::async_trait;
use std::any::Any;
use std::collections::{HashMap, HashSet};
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
use laurus::vector::{FieldOption, VectorIndexConfig, VectorSearchQuery};
use laurus::{DataValue, Document};
use laurus::{EmbedInput, EmbedInputType, Embedder};
use laurus::{LaurusError, Result};

const DIM: usize = 16;
const N: u64 = 200;
/// Angular step between adjacent docs (~0.46°). `N * STEP < π` keeps cosine
/// similarity monotonically decreasing across the whole corpus.
const STEP: f32 = 0.008;

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

/// Vector for doc `i`: angle `i * STEP` in the first two dims, zero elsewhere.
fn doc_vec(i: u64) -> Vec<f32> {
    let theta = i as f32 * STEP;
    let mut v = vec![0.0; DIM];
    v[0] = theta.cos();
    v[1] = theta.sin();
    v
}

/// Query points at angle 0 (= doc 0's direction).
fn query_vec() -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[0] = 1.0;
    v
}

fn hnsw() -> FieldOption {
    FieldOption::Hnsw(HnswOption {
        dimension: DIM,
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

async fn setup_store() -> laurus::vector::VectorStore {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut field_configs = HashMap::new();
    field_configs.insert(
        "vec".to_string(),
        VectorFieldConfig {
            vector: Some(hnsw()),
            lexical: None,
        },
    );
    let config = VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockEmbedder { dimension: DIM }),
        default_fields: vec!["vec".to_string()],
        metadata: HashMap::new(),
        deletion_config: laurus::DeletionConfig::default(),
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    };
    let store = laurus::vector::VectorStore::new(storage, config).unwrap();

    for id in 0..N {
        let doc = Document::builder()
            .add_field("vec", DataValue::Vector(doc_vec(id)))
            .build();
        store.upsert_document_by_internal_id(id, doc).await.unwrap();
    }
    store.commit().await.unwrap();
    store
}

/// Build a request routed to field `vec` (so the HNSW graph path is taken,
/// not the linear-scan fallback) with an optional allow-set.
fn request(limit: usize, allowed: Option<Vec<u64>>) -> VectorSearchRequest {
    VectorSearchRequest {
        query: VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(query_vec()),
            weight: 1.0,
            fields: Some(vec!["vec".into()]),
        }]),
        params: VectorSearchParams {
            limit,
            score_mode: VectorScoreMode::WeightedSum,
            fields: None,
            allowed_ids: allowed,
            ..Default::default()
        },
    }
}

fn hit_ids(results: &laurus::vector::VectorSearchResults) -> Vec<u64> {
    results.hits.iter().map(|h| h.doc_id).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn filter_none_returns_full_page() {
    let store = setup_store().await;
    // Regression guard: with no filter the unfiltered path still returns a
    // full page of `limit` hits (exact membership is graph-dependent, so we
    // do not assert specific ids here).
    let results = store.search(request(10, None)).unwrap();
    assert_eq!(hit_ids(&results).len(), 10, "ten hits");
}

#[tokio::test(flavor = "multi_thread")]
async fn filter_excludes_non_matching() {
    let store = setup_store().await;
    // Invariant for any graph: a filtered result contains only allowed docs.
    let allowed: HashSet<u64> = [3u64, 7, 11].into_iter().collect();
    let results = store
        .search(request(10, Some(allowed.iter().copied().collect())))
        .unwrap();
    let ids: HashSet<u64> = hit_ids(&results).into_iter().collect();
    assert!(
        ids.is_subset(&allowed),
        "every hit must be in the allow-set: got {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn filter_recall_at_least_post_filter() {
    let store = setup_store().await;
    // The recall guarantee of #645, stated as a graph-independent invariant.
    //
    // On a fixed graph, the filtered frontier expands through a superset of
    // the nodes the unfiltered frontier visits (its result heap fills slowly,
    // so it prunes less). Hence every allowed doc a plain post-filter would
    // surface — i.e. an unfiltered hit that happens to be allowed — must also
    // appear in the filtered result. A post-filter can do no better; #645 can
    // do strictly better by reaching matches the unfiltered window misses.
    let allowed: HashSet<u64> = [5u64, 50, 150].into_iter().collect();

    // What a post-filter on the same graph would yield: unfiltered hits ∩ allow.
    let unfiltered: HashSet<u64> = hit_ids(&store.search(request(50, None)).unwrap())
        .into_iter()
        .collect();
    let post_filter: HashSet<u64> = unfiltered.intersection(&allowed).copied().collect();

    let filtered: HashSet<u64> = hit_ids(
        &store
            .search(request(10, Some(allowed.iter().copied().collect())))
            .unwrap(),
    )
    .into_iter()
    .collect();

    assert!(
        filtered.is_subset(&allowed),
        "filtered result must stay within the allow-set: {filtered:?}"
    );
    assert!(
        post_filter.is_subset(&filtered),
        "filtered recall must be at least the post-filter's: \
         post_filter={post_filter:?}, filtered={filtered:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_filter_returns_empty() {
    let store = setup_store().await;
    // An empty allow-set excludes everything; this must be empty, not an error.
    let results = store.search(request(10, Some(vec![]))).unwrap();
    assert!(results.hits.is_empty(), "empty allow-set => no hits");
}

// --- Issue #738: cardinality-driven brute-force mode ---
//
// When the allow-set is no larger than `ef_search` (200 for this store: the
// segmented fallback `ef_construction.max(50) * 2`, Issue #1519) the searcher
// scores the allowed documents directly instead of walking the graph. That
// path is *exact* and does not depend on the (randomised) graph structure, so
// unlike the approximate graph traversal it yields a deterministic, exactly
// ranked result — these tests assert that exactness.

#[tokio::test(flavor = "multi_thread")]
async fn brute_force_sparse_filter_is_exact() {
    let store = setup_store().await;
    // 3 allowed docs (< ef_search) → brute-force. Nearest-first is ascending
    // id on the gradient, so the result is exactly [5, 50, 150], every run.
    let results = store
        .search(request(10, Some(vec![5u64, 50, 150])))
        .unwrap();
    assert_eq!(hit_ids(&results), vec![5, 50, 150]);
}

#[tokio::test(flavor = "multi_thread")]
async fn brute_force_reaches_lone_distant_match() {
    let store = setup_store().await;
    // A single far-down-the-gradient match: the graph walk could miss it, but
    // the brute-force scan scores it directly, so it is always returned.
    let results = store.search(request(10, Some(vec![150u64]))).unwrap();
    assert_eq!(hit_ids(&results), vec![150]);
}
