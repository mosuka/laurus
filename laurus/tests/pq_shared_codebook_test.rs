//! End-to-end tests for the Issue #631 shared PQ codebook feature.
//!
//! Complements the internal determinism check in
//! `pq_io::tests::shared_codebook_produces_byte_identical_codes_to_fresh_training`
//! by exercising the real `HnswIndex` / `SegmentedHnswIndex` write and search
//! stack instead of calling `quantize_segment_pq` directly:
//!
//! 1. A shared-codebook index must return the exact same search results as
//!    one that trains PQ fresh on the identical corpus.
//! 2. A shared codebook must survive a segment-per-commit force-merge
//!    unchanged -- the merge engine re-quantizes every surviving vector
//!    through the same `write()` path, so this is the one path that could
//!    accidentally retrain against the (non-representative) merged subset
//!    instead of reusing the shared codebook.

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::index::config::{HnswIndexConfig, VectorIndexTypeConfig};
use laurus::vector::index::hnsw::HnswIndex;
use laurus::vector::index::hnsw::reader::HnswIndexReader;
use laurus::vector::index::hnsw::segmented::SegmentedHnswIndex;
use laurus::vector::index::pq_codebook::{default_codebook_name, train_and_write_pq_codebook};
use laurus::vector::index::storage::VectorStorage;
use laurus::vector::index::{ManagedVectorIndex, VectorIndex};
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};
use laurus::vector::{DistanceMetric, Vector};

const DIM: usize = 32;
const M: usize = 4;

/// Deterministic pseudo-random corpus of `count` vectors of dimension
/// [`DIM`], ids starting at `id_offset` -- distinct `seed`/`id_offset` pairs
/// give non-overlapping, reproducible corpora.
fn make_corpus(count: usize, seed: u64, id_offset: u64) -> Vec<(u64, String, Vector)> {
    let mut state = seed;
    (0..count)
        .map(|i| {
            let data: Vec<f32> = (0..DIM)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            (id_offset + i as u64, "v".to_string(), Vector::new(data))
        })
        .collect()
}

fn storage() -> Arc<MemoryStorage> {
    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
}

fn base_config() -> HnswIndexConfig {
    HnswIndexConfig {
        dimension: DIM,
        m: 16,
        ef_construction: 100,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantization { subvector_count: M },
        ..Default::default()
    }
}

fn query(v: &Vector, top_k: usize) -> VectorIndexQuery {
    VectorIndexQuery {
        query: v.clone(),
        params: VectorIndexQueryParams {
            top_k,
            ..Default::default()
        },
        field_name: Some("v".to_string()),
        filter: None,
    }
}

fn owned_pq_pool(
    reader: &dyn laurus::vector::reader::VectorIndexReader,
) -> Arc<laurus::vector::index::pq_storage::PqVectorPool> {
    let hnsw_reader = reader
        .as_any()
        .downcast_ref::<HnswIndexReader>()
        .expect("HnswIndexReader");
    match hnsw_reader.vectors() {
        VectorStorage::OwnedPq(pool) => pool.clone(),
        other => panic!("expected OwnedPq, got {other:?}"),
    }
}

/// A shared-codebook index must return byte-for-byte the same search
/// results as one that trains PQ fresh on the exact same corpus -- the
/// end-to-end counterpart of `pq_io`'s internal byte-identical-codes check.
#[test]
fn shared_codebook_search_results_match_fresh_training() {
    let docs = make_corpus(400, 0x1234_5678_9ABC_DEF0, 0);

    // Baseline: fresh per-segment training (today's existing behaviour).
    let trained_storage = storage();
    let trained_index =
        HnswIndex::create(trained_storage.clone(), "trained", base_config()).unwrap();
    {
        let mut writer = trained_index.writer().unwrap();
        writer.add_vectors(docs.clone()).unwrap();
        writer.commit().unwrap();
    }

    // Shared-codebook variant: train once up front, point the config at it.
    let shared_storage = storage();
    let codebook_name = default_codebook_name("v");
    let training_sample: Vec<Vector> = docs.iter().map(|(_, _, v)| v.clone()).collect();
    train_and_write_pq_codebook(
        shared_storage.as_ref() as &dyn Storage,
        &codebook_name,
        DIM,
        M,
        256,
        false,
        &training_sample,
    )
    .unwrap();

    let mut shared_config = base_config();
    shared_config.pq_codebook_path = Some(codebook_name);
    shared_config
        .resolve_pq_codebook(shared_storage.as_ref() as &dyn Storage)
        .unwrap();
    assert!(
        shared_config.pq_codebook.is_some(),
        "codebook must resolve before the index is created"
    );
    let shared_index = HnswIndex::create(shared_storage.clone(), "shared", shared_config).unwrap();
    {
        let mut writer = shared_index.writer().unwrap();
        writer.add_vectors(docs.clone()).unwrap();
        writer.commit().unwrap();
    }

    // Same codebook, byte for byte.
    let trained_pool = owned_pq_pool(trained_index.reader().unwrap().as_ref());
    let shared_pool = owned_pq_pool(shared_index.reader().unwrap().as_ref());
    assert_eq!(shared_pool.params, trained_pool.params);
    assert_eq!(shared_pool.codebook, trained_pool.codebook);

    // Same search results for several distinct queries.
    let trained_searcher = trained_index.searcher().unwrap();
    let shared_searcher = shared_index.searcher().unwrap();
    for (doc_id, _, vector) in docs.iter().step_by(37) {
        let trained_hits = trained_searcher.search(&query(vector, 5)).unwrap();
        let shared_hits = shared_searcher.search(&query(vector, 5)).unwrap();
        let trained_ids: Vec<u64> = trained_hits.results.iter().map(|r| r.doc_id).collect();
        let shared_ids: Vec<u64> = shared_hits.results.iter().map(|r| r.doc_id).collect();
        assert_eq!(
            trained_ids, shared_ids,
            "doc {doc_id}: shared-codebook search must match fresh-trained search"
        );
    }
}

/// A shared codebook must survive a segment-per-commit force-merge
/// unchanged, proving the merge engine's re-quantization pass reuses it
/// instead of silently retraining against the merged subset.
#[test]
fn shared_codebook_survives_a_forced_merge() {
    let storage = storage();

    // Train on a corpus DISTINCT from what gets indexed below, so an
    // accidental retrain against the indexed/merged data would produce
    // visibly different centroids -- making the failure mode unmistakable.
    let training_sample: Vec<Vector> = make_corpus(400, 0x0F0F_0F0F_0F0F_0F0F, 100_000)
        .into_iter()
        .map(|(_, _, v)| v)
        .collect();
    let codebook_name = default_codebook_name("v");
    let trained = train_and_write_pq_codebook(
        storage.as_ref() as &dyn Storage,
        &codebook_name,
        DIM,
        M,
        256,
        false,
        &training_sample,
    )
    .unwrap();

    let mut config = base_config();
    config.segmented = true;
    config.pq_codebook_path = Some(codebook_name);
    config
        .resolve_pq_codebook(storage.as_ref() as &dyn Storage)
        .unwrap();

    let index =
        SegmentedHnswIndex::open_or_create(storage.clone() as Arc<dyn Storage>, "vi", config)
            .unwrap();

    // 3 small commits, each well under PQ_MIN_TRAIN_VECTORS (256) -- exactly
    // the point: a shared codebook keeps even tiny per-commit segments on
    // PQ instead of falling back to Scalar8Bit.
    for batch in 0..3u64 {
        let docs = make_corpus(20, 0xA5A5_0000 + batch, batch * 20);
        let mut writer = index.writer().unwrap();
        writer.add_vectors(docs).unwrap();
        writer.commit().unwrap();
    }

    index.optimize().unwrap();

    let hnsw_files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".hnsw"))
        .collect();
    assert_eq!(
        hnsw_files.len(),
        1,
        "optimize must force-merge every commit's segment into one"
    );
    let segment_id = hnsw_files[0].trim_end_matches(".hnsw").to_string();

    let merged_reader = HnswIndexReader::load(
        storage.clone() as Arc<dyn Storage>,
        &segment_id,
        DistanceMetric::Euclidean,
    )
    .unwrap();
    let merged_pool = match merged_reader.vectors() {
        VectorStorage::OwnedPq(pool) => pool.clone(),
        other => panic!("expected the merged segment to stay on PQ, got {other:?}"),
    };
    assert_eq!(
        merged_pool.params, trained.params,
        "the merged segment's PQ params must match the shared codebook, not a retrain"
    );
    assert_eq!(
        merged_pool.codebook, trained.codebook,
        "the merged segment's codebook must be byte-identical to the shared codebook"
    );
    assert_eq!(merged_pool.vector_count, 60);

    // The merged segment is still searchable and returns sane neighbours.
    let searcher = index.searcher().unwrap();
    let (doc_id, _, vector) = &make_corpus(20, 0xA5A5_0001, 20)[0];
    let hits = searcher.search(&query(vector, 1)).unwrap();
    assert_eq!(hits.results[0].doc_id, *doc_id);
}

/// A [`ManagedVectorIndex`] loads the shared codebook its config names. Ten
/// vectors are too few to train one, and without the shared one the write
/// fails (Issue #1735).
#[test]
fn a_managed_index_loads_the_shared_codebook() {
    let storage = storage();
    let sample: Vec<Vector> = make_corpus(400, 1, 0)
        .into_iter()
        .map(|(_, _, v)| v)
        .collect();
    let trained =
        train_and_write_pq_codebook(storage.as_ref(), "v.pqcb", DIM, M, 256, false, &sample)
            .unwrap();
    let config = HnswIndexConfig {
        pq_codebook_path: Some("v.pqcb".to_string()),
        ..base_config()
    };
    let mut index = ManagedVectorIndex::new(
        VectorIndexTypeConfig::HNSW(config),
        storage.clone() as Arc<dyn Storage>,
        "managed",
    )
    .unwrap();
    index.add_vectors(make_corpus(10, 2, 0)).unwrap();
    index.finalize().unwrap();
    index.write().unwrap();

    let reader = HnswIndexReader::load(storage, "managed", DistanceMetric::Euclidean).unwrap();
    match reader.vectors() {
        VectorStorage::OwnedPq(pool) => assert_eq!(pool.codebook, trained.codebook),
        other => panic!("expected product quantization, got {other:?}"),
    }
}
