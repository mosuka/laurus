//! Integration tests for searcher warmup (Issue #677).
//!
//! `VectorStore::warmup` eagerly builds the cached searcher and invokes the
//! searcher's `warmup`. Warmup must be transparent (results unchanged),
//! idempotent, and must not fail on an mmap-backed (`Lazy`) store, whose
//! vector rows are read in place from the mapped segment files. The
//! page-touching pass of `HnswSearcher::warmup` itself is tested in the
//! crate's `sq8_mapped_tests`, and that `Engine::warmup` runs it for every
//! segment of every HNSW, Flat and IVF field in the crate's engine tests
//! (Issues #1483 and #1485).
//! Cold-start *timing* is environment-dependent and intentionally not
//! asserted.

use std::collections::BTreeSet;
use std::sync::Arc;

use laurus::PrecomputedEmbedder;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::core::field::HnswOption;
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::{
    DistanceMetric, FieldOption, VectorIndexConfig, VectorSearchRequestBuilder, VectorStore,
};
use laurus::{DataValue, Document};

fn hnsw_field(dimension: usize) -> VectorFieldConfig {
    VectorFieldConfig {
        vector: Some(FieldOption::Hnsw(HnswOption {
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
        })),
        lexical: None,
    }
}

fn config(dimension: usize) -> VectorIndexConfig {
    VectorIndexConfig::builder()
        .embedder(Arc::new(PrecomputedEmbedder::new()))
        .field("v", hnsw_field(dimension))
        .build()
        .unwrap()
}

async fn add_docs(store: &VectorStore, vectors: Vec<Vec<f32>>) {
    for (i, vec) in vectors.into_iter().enumerate() {
        let doc = Document::builder()
            .add_field("v", DataValue::Vector(vec))
            .build();
        store
            .upsert_document_by_internal_id((i + 1) as u64, doc)
            .await
            .unwrap();
    }
    store.commit().await.unwrap();
}

fn search_ids(store: &VectorStore, query: Vec<f32>) -> BTreeSet<u64> {
    let request = VectorSearchRequestBuilder::new()
        .add_vector("v", query)
        .limit(3)
        .build();
    store
        .search(request)
        .unwrap()
        .hits
        .into_iter()
        .map(|h| h.doc_id)
        .collect()
}

/// Warmup is transparent (results unchanged) and idempotent on an in-memory
/// (Eager) HNSW store — exercises `VectorStore::warmup` and the `Owned*`
/// early-return in `HnswSearcher::warmup`.
#[tokio::test(flavor = "multi_thread")]
async fn warmup_is_transparent_and_idempotent() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = VectorStore::new(storage, config(3)).unwrap();
    add_docs(
        &store,
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ],
    )
    .await;

    let query = vec![1.0, 0.1, 0.0];
    let before = search_ids(&store, query.clone());
    assert!(!before.is_empty(), "baseline search must return hits");

    store.warmup().unwrap();
    store.warmup().unwrap(); // idempotent

    let after = search_ids(&store, query);
    assert_eq!(after, before, "warmup must not change search results");
}

/// Warmup on an mmap-backed HNSW store succeeds and leaves search working.
/// `use_mmap` is forced on so the storage is `Lazy` on every platform
/// (`LAURUS_NO_MMAP=1` would otherwise turn it off).
#[tokio::test(flavor = "multi_thread")]
async fn warmup_on_demand_hnsw_pre_faults() {
    let dir = tempfile::tempdir().unwrap();
    let mut file_config = FileStorageConfig::new(dir.path());
    file_config.use_mmap = true; // force Lazy (mapped rows) on all platforms
    let storage = StorageFactory::create(StorageConfig::File(file_config)).unwrap();
    let store = VectorStore::new(storage, config(3)).unwrap();
    add_docs(
        &store,
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
    )
    .await;

    // Warmup before any query: builds the cached searcher and runs its
    // warmup.
    store.warmup().unwrap();

    let hits = search_ids(&store, vec![1.0, 0.1, 0.0]);
    assert!(
        !hits.is_empty(),
        "search after warmup on an mmap-backed store must return hits"
    );
}
