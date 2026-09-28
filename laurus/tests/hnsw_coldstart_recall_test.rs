//! Regression tests for #872 — seed/incremental bulk-load recall parity.
//!
//! Appending a large batch onto a small HNSW base used to funnel every parallel
//! insert through the base's low-level entry point, fragmenting the graph:
//! reachability was still guaranteed (by the #868 connectivity repair), but the
//! appended nodes had far fewer/worse in-edges, so their search recall collapsed
//! to roughly half of a fresh full build of the same total size.
//!
//! The #872 fix has two regimes, and there is one test per regime:
//!   * base `< ef_construction` — the base is too shallow to navigate, so the
//!     append discards it and does a full rebuild (fresh quality). Covered by
//!     [`seed_then_bulk_load_recall_matches_fresh_build`] (base = 1).
//!   * base `>= ef_construction` — the incremental path is kept, but if the
//!     append promotes the entry point to a new top-level node, that node is
//!     inserted first and used as the build search start so inserts descend from
//!     the top layer instead of funneling through `old_ep`. Covered by
//!     [`incremental_append_recall_matches_fresh_build`] (base = 200).
//!
//! The gate is **self-recall@10**: query each node with its own vector and check
//! it appears in its own top-10. A well-built graph is ~1.0; a fragmented one is
//! much lower. Each test compares the appended nodes' self-recall against a fresh
//! full build of the same corpus, so it needs no fixed absolute threshold (both
//! builds see the identical vectors and query set).
//!
//! A `parallel_build` graph depends on thread interleaving, so each build's
//! self-recall is a random draw. Both tests here therefore run on a
//! single-thread rayon `ThreadPool`, which makes them deterministic (rebuild
//! regime: Issue #1233; incremental regime: Issue #1237, which found that the
//! incremental regime's earlier best-of-3 mitigation — Issue #886 — had
//! quietly lost most of its power to catch the regime's regression once a
//! later, unrelated change (#1242) narrowed the parallel-build recall
//! distribution). Real concurrent-insert coverage is provided separately by
//! `hnsw_reachability_test` and `hnsw_parallel_build_recall_test`.

use std::sync::Arc;

use laurus::storage::file::{FileStorage, FileStorageConfig};
use laurus::vector::index::hnsw::searcher::HnswSearcher;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
use laurus::vector::{
    DistanceMetric, HnswIndexConfig, HnswIndexReader, HnswIndexWriter, Vector, VectorIndexWriter,
    VectorIndexWriterConfig,
};
use tempfile::tempdir;

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
fn writer_config() -> VectorIndexWriterConfig {
    VectorIndexWriterConfig {
        parallel_build: true,
        ..Default::default()
    }
}

/// Run `f` on a dedicated single-thread rayon pool, so every `par_iter` inside
/// the writer runs serially and the build is deterministic (Issues #1233,
/// #1237).
///
/// The writer still takes its `parallel_build` code path; only the pool it
/// runs on changes. A panic inside `f` propagates to the caller.
fn on_single_thread<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("failed to build a single-thread rayon pool")
        .install(f)
}

/// Fraction of ids in `[lo, hi)` that appear in their own vector's top-10.
fn self_recall_at_10(path: &std::path::Path, name: &str, lo: u64, hi: u64) -> f32 {
    let storage = Arc::new(FileStorage::new(path, FileStorageConfig::new(path)).unwrap());
    let reader = HnswIndexReader::load(storage, name, DistanceMetric::Cosine).unwrap();
    let mut searcher = HnswSearcher::new(Arc::new(reader)).unwrap();
    searcher.set_ef_search(200);
    let mut hits = 0u64;
    for id in lo..hi {
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
    hits as f32 / (hi - lo) as f32
}

/// Fresh full build of ids `0..n` in one shot.
fn fresh_build(path: &std::path::Path, name: &str, n: u64) {
    let storage = Arc::new(FileStorage::new(path, FileStorageConfig::new(path)).unwrap());
    let mut w = HnswIndexWriter::with_storage(config(), writer_config(), name, storage).unwrap();
    let v: Vec<_> = (0..n).map(|i| (i, "v".to_string(), doc_vec(i))).collect();
    w.add_vectors(v).unwrap();
    w.finalize().unwrap();
    w.write().unwrap();
}

/// Build `0..base` first, commit, then append `base..n` in a second commit.
fn seed_then_append(path: &std::path::Path, name: &str, base: u64, n: u64) {
    {
        let storage = Arc::new(FileStorage::new(path, FileStorageConfig::new(path)).unwrap());
        let mut w =
            HnswIndexWriter::with_storage(config(), writer_config(), name, storage).unwrap();
        let v: Vec<_> = (0..base)
            .map(|i| (i, "v".to_string(), doc_vec(i)))
            .collect();
        w.add_vectors(v).unwrap();
        w.finalize().unwrap();
        w.write().unwrap();
    }
    {
        let storage = Arc::new(FileStorage::new(path, FileStorageConfig::new(path)).unwrap());
        let mut w = HnswIndexWriter::load(config(), writer_config(), storage, name).unwrap();
        let v: Vec<_> = (base..n)
            .map(|i| (i, "v".to_string(), doc_vec(i)))
            .collect();
        w.add_vectors(v).unwrap();
        w.finalize().unwrap();
        w.write().unwrap();
    }
}

/// Assert the appended nodes' self-recall matches a fresh full build's over the
/// same id range, within a small tolerance (pre-#872 the append was ~0.5x).
///
/// Both callers run this on [`on_single_thread`], so `fresh_build` and
/// `seed_then_append` are each a single deterministic build.
fn assert_append_matches_fresh(base: u64, n: u64) {
    let fresh_dir = tempdir().unwrap();
    fresh_build(fresh_dir.path(), "fresh", n);
    let fresh = self_recall_at_10(fresh_dir.path(), "fresh", base, n);

    // Sanity: the fresh build must itself be well-connected.
    assert!(
        fresh > 0.9,
        "fresh full-build self-recall@10 should be high, got {fresh:.4}"
    );

    let cold_dir = tempdir().unwrap();
    seed_then_append(cold_dir.path(), "cold", base, n);
    let cold = self_recall_at_10(cold_dir.path(), "cold", base, n);

    // Absolute-difference tolerance so a marginally-higher cold value (it can
    // slightly exceed fresh) also passes.
    assert!(
        cold >= fresh - 0.05,
        "base={base} seed-then-bulk-load self-recall@10 ({cold:.4}) must match \
         the fresh build ({fresh:.4}) within tolerance (#872)",
    );
}

/// #872, base `< ef_construction`: a seed-1-then-bulk-append build is rebuilt
/// fresh, so it must reach the same self-recall as a fresh full build.
///
/// Runs both builds on a single thread (Issue #1233). The cold and fresh sides
/// take the identical full-build code path (#872's discard-and-rebuild
/// branch), but under real parallelism each build is an independent draw from
/// the thread-interleaving distribution, so their variance does not cancel: a
/// single parallel trial failed ~2.6% of CI jobs once #1150's level draw
/// widened that distribution. A best-of-N mitigation (as #886 used for
/// [`incremental_append_recall_matches_fresh_build`]) would hide the very
/// regression this test guards: with the rebuild regime disabled, parallel
/// cold builds scatter across the tolerance (~0.85-0.97), so their max usually
/// passes. On a single thread the build is deterministic, the two sides agree
/// (both 0.998 locally), and that regression fails by a wide margin (~0.83).
/// Parallel builds stay covered by `hnsw_reachability_test` and
/// `hnsw_parallel_build_recall_test`.
#[test]
fn seed_then_bulk_load_recall_matches_fresh_build() {
    on_single_thread(|| assert_append_matches_fresh(1, 5000));
}

/// #872, base `>= ef_construction`: appending onto a real (non-trivial) base
/// keeps the incremental path; the promoted-entry-point-first fix must still
/// give the appended nodes fresh-build recall.
///
/// Runs both builds on a single thread (Issue #1237; this test used to keep
/// real parallelism and take the best of 3 trials, Issue #886). Locally, a
/// single thread gives fresh 0.9984 and cold 0.9987, both exactly reproducible
/// across repeated builds; disabling the `promoted_ep` fix gives a
/// deterministic 0.8983, which fails the tolerance on every run. Best-of-3 no
/// longer reliably catches that failure: at `RAYON_NUM_THREADS=4` the disabled
/// fix's self-recall scattered 0.9208-0.9540 across 8 builds, and 6 of those 8
/// individually clear the tolerance, so best-of-3's max is likely to land on a
/// passing draw. (This regime's parallel variance narrowed a lot after #1242
/// fixed a different, unrelated funneling defect, without narrowing the
/// disabled-fix build's variance by nearly as much — the two distributions
/// #886 found cleanly separated now overlap the tolerance.) Parallel builds
/// stay covered by `hnsw_reachability_test` and `hnsw_parallel_build_recall_test`.
#[test]
fn incremental_append_recall_matches_fresh_build() {
    on_single_thread(|| assert_append_matches_fresh(200, 5000));
}
