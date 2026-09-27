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
//! self-recall is a random draw. The rebuild-regime test therefore runs both
//! builds on a single thread, which makes them deterministic (Issue #1233); the
//! incremental-regime test keeps real parallelism and takes the best of 3
//! trials (Issue #886).

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
/// the writer runs serially and the build is deterministic (Issue #1233).
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
/// Takes the **best of `cold_trials` independent `seed_then_append` builds**
/// (Issue #886): under `parallel_build`, nondeterministic thread interleaving
/// occasionally produces a graph whose appended-node self-recall dips into a
/// tail below the tolerance even though the #872 fix is working — a single
/// build's result is not a reliable pass/fail signal on its own. `cold_trials
/// == 1` is for callers running on [`on_single_thread`], where every build is
/// identical and one trial is exact. A genuine regression (the #872 defect is
/// a structural, deterministic funneling through `old_ep`, not a scheduling
/// fluke) reproduces on **every** trial, so taking the max cannot mask it:
/// manually disabling the #872 `promoted_ep` fix under `RAYON_NUM_THREADS=2`
/// reproduced an identical failing self-recall@10 (~0.83, tolerance exceeded)
/// on all 3 trials. Not every regression behaves that way, though: disabling
/// the rebuild regime instead scatters parallel builds across the tolerance
/// (Issue #1233), so the rebuild-regime test runs single-threaded rather than
/// taking a best-of-N.
fn assert_append_matches_fresh(base: u64, n: u64, cold_trials: u32) {
    let fresh_dir = tempdir().unwrap();
    fresh_build(fresh_dir.path(), "fresh", n);
    let fresh = self_recall_at_10(fresh_dir.path(), "fresh", base, n);

    // Sanity: the fresh build must itself be well-connected.
    assert!(
        fresh > 0.9,
        "fresh full-build self-recall@10 should be high, got {fresh:.4}"
    );

    let mut best_cold = f32::NEG_INFINITY;
    for _ in 0..cold_trials {
        let cold_dir = tempdir().unwrap();
        seed_then_append(cold_dir.path(), "cold", base, n);
        let cold = self_recall_at_10(cold_dir.path(), "cold", base, n);
        best_cold = best_cold.max(cold);
    }

    // Absolute-difference tolerance so a marginally-higher cold value (it can
    // slightly exceed fresh) also passes.
    assert!(
        best_cold >= fresh - 0.05,
        "base={base} seed-then-bulk-load self-recall@10 (best of {cold_trials}: \
         {best_cold:.4}) must match the fresh build ({fresh:.4}) within tolerance (#872)",
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
/// widened that distribution. Best-of-N (the #886 fix) would hide the very
/// regression this test guards: with the rebuild regime disabled, parallel
/// cold builds scatter across the tolerance (~0.85-0.97), so their max usually
/// passes. On a single thread the build is deterministic, the two sides agree
/// (both 0.998 locally), and that regression fails by a wide margin (~0.83).
/// Parallel builds stay covered by `hnsw_reachability_test` and
/// [`incremental_append_recall_matches_fresh_build`].
#[test]
fn seed_then_bulk_load_recall_matches_fresh_build() {
    on_single_thread(|| assert_append_matches_fresh(1, 5000, 1));
}

/// #872, base `>= ef_construction`: appending onto a real (non-trivial) base
/// keeps the incremental path; the promoted-entry-point-first fix must still
/// give the appended nodes fresh-build recall (pre-fix ~0.73 vs ~0.98).
///
/// Best-of-3 (Issue #886): the incremental append is a structurally different
/// code path from the fresh build's one-shot construction, so their variance
/// does not cancel — on low-core CI runners, rare unlucky parallel-insertion
/// orderings can dip a single build's self-recall into a tail (observed:
/// ~0.91) even though the fix is working. Three independent trials make that
/// tail exponentially rarer while still failing hard on a genuine regression.
#[test]
fn incremental_append_recall_matches_fresh_build() {
    assert_append_matches_fresh(200, 5000, 3);
}
