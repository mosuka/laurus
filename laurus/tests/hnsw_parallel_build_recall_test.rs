//! Regression test for #1238 — a parallel HNSW build must match a serial build.
//!
//! The parallel phase of `build_hnsw_graph` used to hand each rayon thread a
//! contiguous range of the sorted doc ids, so each thread grew its own
//! insertion front. On a corpus whose doc id order follows position in vector
//! space, the layer-0 graph then split into large components at the fronts'
//! boundaries (the #868 connectivity repair kept every node reachable, but
//! through a single edge per component), and which boundaries split depended
//! on thread interleaving. On this test's corpus, 4-thread builds scored a
//! mean self-recall@10 of 0.978 (minimum 0.899) against a serial 0.998, and
//! 39 of 40 builds scored below 0.99. Workers now claim the new nodes in input
//! order, 16 at a time, and 4-thread builds score 0.997 or more.
//!
//! The gate is **self-recall@10**: query each node with its own vector and
//! check it appears in its own top-10.

use std::sync::Arc;

use laurus::storage::file::{FileStorage, FileStorageConfig};
use laurus::vector::index::hnsw::searcher::HnswSearcher;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
use laurus::vector::{
    DistanceMetric, HnswIndexConfig, HnswIndexReader, HnswIndexWriter, Vector, VectorIndexWriter,
    VectorIndexWriterConfig,
};
use tempfile::tempdir;

/// Number of nodes in the corpus.
const N: u64 = 5000;

/// Worker threads for the parallel builds. Fixed rather than taken from the
/// machine, so the test exercises real concurrency on every runner.
const THREADS: usize = 4;

/// Independent parallel builds; every one must pass.
const TRIALS: usize = 3;

/// Lowest acceptable self-recall@10 for a single parallel build. The fixed
/// build's minimum over 20 local 4-thread builds was 0.9974; the old
/// contiguous-range split scored below this in 39 of 40 builds.
const MIN_SELF_RECALL: f32 = 0.99;

/// A smooth curve parameterized by doc id, so doc id order follows position
/// in vector space (the same corpus as `hnsw_coldstart_recall_test`).
fn doc_vec(i: u64) -> Vector {
    let mut v = vec![0.0f32; 16];
    let t = i as f32 * 0.001;
    v[0] = t.cos();
    v[1] = t.sin();
    v[2] = (t * 2.0).cos();
    v[3] = (t * 3.0).sin();
    Vector::new(v)
}

fn config() -> HnswIndexConfig {
    HnswIndexConfig {
        dimension: 16,
        m: 16,
        ef_construction: 100,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    }
}

/// Run `f` on a dedicated rayon pool with `threads` threads.
fn on_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build a rayon pool")
        .install(f)
}

/// Fresh full build of ids `0..N` in one shot, with `parallel_build` on.
fn build(path: &std::path::Path, name: &str) {
    let storage = Arc::new(FileStorage::new(path, FileStorageConfig::new(path)).unwrap());
    let writer_config = VectorIndexWriterConfig {
        parallel_build: true,
        ..Default::default()
    };
    let mut w = HnswIndexWriter::with_storage(config(), writer_config, name, storage).unwrap();
    let v: Vec<_> = (0..N).map(|i| (i, "v".to_string(), doc_vec(i))).collect();
    w.add_vectors(v).unwrap();
    w.finalize().unwrap();
    w.write().unwrap();
}

/// Fraction of ids in `0..N` that appear in their own vector's top-10.
fn self_recall_at_10(path: &std::path::Path, name: &str) -> f32 {
    let storage = Arc::new(FileStorage::new(path, FileStorageConfig::new(path)).unwrap());
    let reader = HnswIndexReader::load(storage, name, DistanceMetric::Cosine).unwrap();
    let mut searcher = HnswSearcher::new(Arc::new(reader)).unwrap();
    searcher.set_ef_search(200);
    let mut hits = 0u64;
    for id in 0..N {
        let req = VectorIndexQuery::new(doc_vec(id))
            .top_k(10)
            .field_name("v".to_string());
        if searcher
            .search(&req)
            .unwrap()
            .results
            .iter()
            .any(|r| r.doc_id == id)
        {
            hits += 1;
        }
    }
    hits as f32 / N as f32
}

/// #1238: every 4-thread build of a corpus inserted in spatial order must
/// reach serial-build quality.
///
/// Each build is an independent draw from the thread-interleaving
/// distribution, so the test asserts on every trial rather than the best
/// one: the old per-thread ranges scored below the threshold in 39 of 40
/// builds, so any trial catches that regression.
#[test]
fn parallel_build_recall_matches_serial_build() {
    let serial_dir = tempdir().unwrap();
    on_pool(1, || build(serial_dir.path(), "serial"));
    let serial = self_recall_at_10(serial_dir.path(), "serial");
    assert!(
        serial >= MIN_SELF_RECALL,
        "serial build self-recall@10 should be high, got {serial:.4}"
    );

    for trial in 0..TRIALS {
        let dir = tempdir().unwrap();
        on_pool(THREADS, || build(dir.path(), "parallel"));
        let parallel = self_recall_at_10(dir.path(), "parallel");
        assert!(
            parallel >= MIN_SELF_RECALL,
            "trial {trial}: {THREADS}-thread build self-recall@10 ({parallel:.4}) must be at \
             least {MIN_SELF_RECALL} (serial build: {serial:.4}) (#1238)",
        );
    }
}
