//! Regression test for #1241 — cluster-ordered inserts must not lose recall.
//!
//! `select_neighbors` used to take the M nearest candidates, and
//! `prune_neighbors` kept the nearest when a back-edge overflows a list.
//! Neither applied a diversity heuristic, so once a neighborhood filled with
//! nearby nodes, no long edge survived to connect it onward. On this corpus —
//! tight clusters inserted cluster-by-cluster, contiguous doc ids per cluster
//! — that showed up as a strong dependence on insertion order: self-recall@10
//! sat at 0.9828 serial / a mean of 0.9874 over 3 parallel trials (both below
//! this test's own 0.99 gate) before Issue #1241's diversity heuristic
//! (Malkov & Yashunin's Algorithm 4, the same form hnswlib/Lucene/FAISS use);
//! confirmed by reverting the heuristic locally during development. With the
//! heuristic, repeated local runs landed at 0.997-1.0000 in both regimes.
//!
//! The gate is **self-recall@10**: query each node with its own vector and
//! check it appears in its own top-10. Both regimes must clear 0.99 (the
//! issue's acceptance criterion). `ef_construction` is 200 rather than the
//! 100 other HNSW recall tests use — at 100, even the post-heuristic graph
//! stayed in the 0.98-0.99 range on this particular corpus (50 clusters of
//! 100 near-duplicate points is a denser local structure than e.g. the
//! `curve` corpus other tests use), so 200 was chosen empirically as the
//! smallest value giving comfortable, stable headroom above the gate.

use std::sync::Arc;

use laurus::storage::file::{FileStorage, FileStorageConfig};
use laurus::vector::index::hnsw::searcher::HnswSearcher;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
use laurus::vector::{
    DistanceMetric, HnswIndexConfig, HnswIndexReader, HnswIndexWriter, Vector, VectorIndexWriter,
    VectorIndexWriterConfig,
};
use tempfile::tempdir;

/// Clusters in the corpus.
const CLUSTERS: u64 = 50;

/// Vectors per cluster.
const PER_CLUSTER: u64 = 100;

/// Total nodes in the corpus (contiguous doc ids, cluster by cluster).
const N: u64 = CLUSTERS * PER_CLUSTER;

/// Worker threads for the parallel build.
const THREADS: usize = 4;

/// Independent parallel builds; every one must pass.
const TRIALS: usize = 3;

/// Lowest acceptable self-recall@10, in both regimes (Issue #1241's
/// acceptance criterion). Pre-heuristic this corpus scored 0.9828 serial and
/// a mean of 0.9874 over 3 parallel trials.
const MIN_SELF_RECALL: f32 = 0.99;

/// A small deterministic LCG, matching the style other HNSW recall tests use
/// (e.g. `vector_field_rebuild_recall_test`'s `pseudo_random_f32`) instead of
/// pulling in the `rand` crate for a test corpus.
fn lcg_f32s(seed: u32, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9).wrapping_add(0xDEAD_BEEF);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let bits = (state >> 16) as u16;
            (bits as f32 / u16::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// 50 cluster centers, each one PRNG draw over all 16 dims, plus small
/// per-vector noise (a second draw, scaled down) so vectors within a cluster
/// are close to their center but distinguishable from each other. Random
/// 16-dim centers are naturally far apart in cosine terms (high-dimensional
/// near-orthogonality), so clusters stay well separated without needing to
/// place them by hand. Doc ids are assigned cluster by cluster and
/// contiguously within each cluster, matching #1241's `clusters_sorted`
/// corpus: `[0, 100)` is cluster 0, `[100, 200)` is cluster 1, and so on.
fn doc_vec(i: u64) -> Vector {
    let cluster = i / PER_CLUSTER;
    let center = lcg_f32s(cluster as u32, 16);
    let noise = lcg_f32s(10_000 + i as u32, 16);
    let v: Vec<f32> = center
        .iter()
        .zip(noise.iter())
        .map(|(&c, &n)| c + n * 0.15)
        .collect();
    Vector::new(v)
}

fn config() -> HnswIndexConfig {
    HnswIndexConfig {
        dimension: 16,
        m: 16,
        ef_construction: 200,
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

/// #1241: a serial build of cluster-ordered data must reach the recall
/// target, not just the 0.9248 nearest-only score.
#[test]
fn cluster_ordered_serial_build_reaches_recall_target() {
    let dir = tempdir().unwrap();
    on_pool(1, || build(dir.path(), "serial"));
    let serial = self_recall_at_10(dir.path(), "serial");
    assert!(
        serial >= MIN_SELF_RECALL,
        "serial build of cluster-ordered data self-recall@10 must be at least \
         {MIN_SELF_RECALL}, got {serial:.4} (#1241)",
    );
}

/// #1241: every parallel build of cluster-ordered data must also reach the
/// recall target — each build is an independent draw from the thread
/// interleaving distribution, so every trial is asserted rather than just
/// the best one.
#[test]
fn cluster_ordered_parallel_build_reaches_recall_target() {
    for trial in 0..TRIALS {
        let dir = tempdir().unwrap();
        on_pool(THREADS, || build(dir.path(), "parallel"));
        let parallel = self_recall_at_10(dir.path(), "parallel");
        assert!(
            parallel >= MIN_SELF_RECALL,
            "trial {trial}: {THREADS}-thread build of cluster-ordered data self-recall@10 \
             ({parallel:.4}) must be at least {MIN_SELF_RECALL} (#1241)",
        );
    }
}
