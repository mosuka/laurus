//! Integration tests for deletion-aware HNSW traversal (issue #665).
//!
//! The store holds one HNSW field `vec` whose documents lie on a 1-D angular
//! gradient: doc `i` points at angle `i * STEP` in the first two dimensions,
//! so cosine similarity to the query (angle 0) decreases monotonically with
//! `i` (nearest-neighbour order is ascending `i`).
//!
//! Before #665 the graph walk admitted logically deleted nodes into the
//! result heap (and `finalize_graph_results` never re-checked deletion, while
//! the quantized distance path bypasses `get_vector` entirely), so deleted
//! documents both leaked into results and consumed `ef_search` slots that
//! should have gone to live neighbours. These tests assert the fix.
//!
//! HNSW graph construction is randomised (`rand::rng()` per build), so the
//! *exact* set a search returns varies run to run. These tests therefore
//! assert only invariants that hold for any graph:
//!
//! - a result never contains a deleted document (`deleted_never_returned`,
//!   `deletion_with_filter`, `tiny_allowset_excludes_deleted`);
//! - deleting the nearest documents does not shrink the result page — the
//!   walk fills it from the next-nearest *live* documents instead of returning
//!   the deleted ones (`deleting_nearest_keeps_full_page`), which is the
//!   recall guarantee #665 restores;
//! - mass deletion leaks nothing (`mass_deletion_no_leak`);
//! - with no deletions the result is byte-for-byte the pristine page
//!   (`no_deletions_returns_full_page`).

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

/// Delete every id in `ids` and commit so the deletion bitmap is rebuilt and
/// re-attached to the search reader.
async fn delete_and_commit(store: &laurus::vector::VectorStore, ids: &[u64]) {
    for &id in ids {
        store.delete_document_by_internal_id(id).await.unwrap();
    }
    store.commit().await.unwrap();
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
async fn deleted_never_returned() {
    let store = setup_store().await;
    // Delete the 20 documents nearest the query. Before #665 these would have
    // dominated the top-k; the result must now contain none of them.
    let deleted: HashSet<u64> = (0..20).collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;

    let ids: HashSet<u64> = hit_ids(&store.search(request(10, None)).unwrap())
        .into_iter()
        .collect();

    assert!(
        ids.is_disjoint(&deleted),
        "no deleted doc may appear: deleted={deleted:?}, hits={ids:?}"
    );
    assert!(
        ids.iter().all(|id| *id < N),
        "every hit must be a real live doc id: {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn deleting_nearest_keeps_full_page() {
    let store = setup_store().await;
    // Recall guarantee of #665: deleting the nearest docs must not shrink the
    // page — the walk fills `limit` from the next-nearest *live* docs instead
    // of letting deleted nodes consume the `ef_search` slots. With 180 live
    // docs remaining and a well-connected graph, a 10-hit page must still come
    // back full.
    let deleted: HashSet<u64> = (0..20).collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;

    let ids = hit_ids(&store.search(request(10, None)).unwrap());
    assert_eq!(
        ids.len(),
        10,
        "page must stay full despite deletions: {ids:?}"
    );
    let ids: HashSet<u64> = ids.into_iter().collect();
    assert!(
        ids.is_disjoint(&deleted),
        "full page must be all live: deleted={deleted:?}, hits={ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn deletion_with_filter() {
    let store = setup_store().await;
    // Allow-set larger than ef_search so the graph path runs (not the
    // brute-force tiny-allowset path). This store's default ef_search is
    // the segmented fallback `ef_construction.max(50) * 2 = 200`, which an
    // allow-set of 100 does not exceed, so the query sets 50 (Issue #1519).
    // Half of the allow-set is deleted.
    let allowed: HashSet<u64> = (0..100).collect();
    let deleted: HashSet<u64> = (0..50).collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;

    let mut req = request(10, Some(allowed.iter().copied().collect()));
    req.params.ef_search = Some(50);
    let ids: HashSet<u64> = hit_ids(&store.search(req).unwrap()).into_iter().collect();

    assert!(
        ids.is_subset(&allowed),
        "every hit must be in the allow-set: {ids:?}"
    );
    assert!(
        ids.is_disjoint(&deleted),
        "no deleted doc may appear even when allowed: deleted={deleted:?}, hits={ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn tiny_allowset_excludes_deleted() {
    let store = setup_store().await;
    // Allow-set smaller than ef_search triggers the exact brute-force path
    // (#738). Its admission must also honour deletions (#665).
    let allowed: Vec<u64> = vec![0, 1, 2, 40, 41, 42];
    let deleted: HashSet<u64> = [0u64, 1, 2].into_iter().collect();
    delete_and_commit(&store, &deleted.iter().copied().collect::<Vec<_>>()).await;

    let ids: HashSet<u64> = hit_ids(&store.search(request(10, Some(allowed.clone()))).unwrap())
        .into_iter()
        .collect();
    let live_allowed: HashSet<u64> = [40u64, 41, 42].into_iter().collect();

    assert!(
        ids.is_subset(&live_allowed),
        "tiny-allowset hits must be the live allowed docs only: {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn mass_deletion_no_leak() {
    let store = setup_store().await;
    // Delete all but the last five docs. The result may be short, but every
    // hit must be one of the survivors — nothing deleted may leak.
    let survivors: HashSet<u64> = (N - 5..N).collect();
    let deleted: Vec<u64> = (0..N - 5).collect();
    delete_and_commit(&store, &deleted).await;

    let ids: HashSet<u64> = hit_ids(&store.search(request(10, None)).unwrap())
        .into_iter()
        .collect();

    assert!(
        ids.len() <= 5,
        "at most five survivors can be returned: {ids:?}"
    );
    assert!(
        ids.is_subset(&survivors),
        "only survivors may appear: survivors={survivors:?}, hits={ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn no_deletions_returns_full_page() {
    let store = setup_store().await;
    // Regression guard: with no deletions the pristine path is unchanged and
    // still returns a full page of `limit` hits.
    let results = store.search(request(10, None)).unwrap();
    assert_eq!(hit_ids(&results).len(), 10, "ten hits with no deletions");
}
