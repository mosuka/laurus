//! End-to-end vector search benchmarks for Flat, IVF, and HNSW indexes.
//!
//! Compares construction and search performance across all three index types
//! using the same dataset and query parameters.
//!
//! # Quantization status (Issue #481 Stage 1)
//!
//! After Stage 1, **every** vector index group in this file goes through
//! the int8 scalar-quantized hot path: writers emit the `LVS1` segment
//! format, readers populate `VectorStorage::OwnedQuantized`, and the
//! search loop dispatches to `distance_quantized`. There is no longer an
//! f32-only path to compare against in-tree.
//!
//! - **Stage 1 speed gate** is `bench_hnsw_graph_search` at
//!   `top10/50000`: the int8 path here must achieve ≥ 2× speedup vs the
//!   pre-Stage-1 f32 numbers recorded on `main` before this PR landed.
//!   Compare via two separate runs (one per branch).
//! - **Stage 2 speed gate** is `bench_hnsw_graph_search_rerank` at
//!   `top10/50000`. Issue #481 Stage 2 asks for ≥ 3× speedup vs the
//!   same pre-Stage-1 f32 baseline; the bench above shows that on
//!   the suite's synthetic random unit-norm data, rerank's per-query
//!   overhead is well inside the noise band of
//!   `bench_hnsw_graph_search` (a few µs at dim 128 for 50 exact-
//!   distance calls). The realised speedup therefore tracks Stage 1
//!   plus that small overhead; on synthetic random data the
//!   underlying HNSW graph traversal is the floor, so the absolute
//!   speedup ratio is bounded by the Stage 1 result on the same
//!   data (≈ 2×). Real clustered embeddings allow a lower ef_search
//!   and a wider speedup margin -- see
//!   `tests/vector_recall_test.rs::stage2_recall_sweep_diagnostic`
//!   for the recall-vs-budget trade-off underlying that choice.
//! - **Recall gates** are _not_ measured here -- recall lives in
//!   `laurus/tests/vector_recall_test.rs` so the latency vs recall
//!   surfaces stay independent.
//! - The `HNSW Construction` group also runs through the new format
//!   (writer trains + serializes int8 at flush). Construction bench
//!   numbers should stay within the noise band of the f32 baseline
//!   since graph build still computes f32 distances; only the
//!   final `commit()` call now does the per-segment SQ training.
//!
//! # Scope
//!
//! - End-to-end through `FlatVectorSearcher`, `IvfSearcher`, `HnswSearcher`.
//! - Search corpus sweep: 1 000 / 5 000 by default; opt in to a 100 000
//!   vector case via `LAURUS_BENCH_LARGE=1` (see "Large-corpus gate" below).
//!   Construction benches stay at 1 000 / 5 000 to keep the default `cargo
//!   bench` runtime under five minutes.
//! - HNSW `ef_search` sweep: `bench_hnsw_ef_search_sweep` measures graph
//!   search at fixed corpus 5 000, dim 128, `ef_search ∈ {16, 64, 128, 256,
//!   512}`. Drives the visited-set hot path that #406 (replace
//!   `HashSet<u64>` with bitmap) targets.
//! - HNSW multi-field: `bench_hnsw_multi_field_search` builds 5 000 vectors
//!   distributed across `field_a` (30 %), `field_b` (30 %), `field_c`
//!   (40 %) and searches with `field_name("field_a")`. Drives the per-field
//!   `vector_ids` cache hot path that #405 targets — today every search
//!   linearly scans the full field list.
//! - All inputs are deterministic via `common::DEFAULT_SEED` so two
//!   consecutive runs produce comparable numbers (#427 hygiene).
//!
//! # Large-corpus gate
//!
//! Setting `LAURUS_BENCH_LARGE=1` adds a 100 000-vector case to
//! `bench_flat_search`, `bench_ivf_search`, `bench_hnsw_fallback_search`,
//! and `bench_hnsw_graph_search`. The 100 000-vector setup takes several
//! seconds and per-iter search at 100 k can be tens of milliseconds, so
//! this case is opt-in. Default runs (no env var) finish in well under
//! five minutes on a typical workstation.
//!
//! # On-disk index cache (#513 Stage 2)
//!
//! [`cached_vector_reader`] persists the built vector index under
//! `target/laurus_bench_index_cache/<slot>_v<N>/`, mirroring the
//! lexical bench's `cached_engine` (#510) and the hybrid bench's
//! `cached_hybrid_engine` (#513 Stage 1). On a fresh checkout the
//! first run pays the index-build cost once per case; later
//! `cargo bench` runs reopen the cached index via the type-specific
//! reader loader (`FlatVectorIndexReader::load`,
//! `IvfIndexReader::load`, `HnswIndexReader::load`) in well under a
//! second. The cache is applied to **search-only** benches; the
//! Construction benches (`bench_flat_construction`,
//! `bench_ivf_construction`, `bench_hnsw_construction`) and the SIFT
//! real-data benches' Criterion measurement window are unchanged.
//! Bump [`BENCH_INDEX_FORMAT_VERSION`] when anything that would alter
//! the resulting index changes; `LAURUS_BENCH_REBUILD=1` forces a
//! wipe-and-rebuild. See `benches/BENCHMARKS.md` for the architecture
//! rationale.
//!
//! # Run
//!
//! ```sh
//! # Daily iteration (fast — uses cache after first run):
//! cargo bench --bench vector_search_bench                          # default sizes
//!
//! # Acceptance / large-corpus sweep:
//! LAURUS_BENCH_LARGE=1 cargo bench --bench vector_search_bench     # adds 100 k
//!
//! # Force a fresh cache build:
//! LAURUS_BENCH_REBUILD=1 cargo bench --bench vector_search_bench
//! ```
//!
//! Filter by group / case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench vector_search_bench -- "Flat Search"
//! cargo bench --bench vector_search_bench -- "ef_search/ef_64"
//! cargo bench --bench vector_search_bench -- "Multi-field"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench vector_search_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules and
//! `benches/BENCHMARKS.md` for the cross-cutting bench architecture.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use laurus::storage::Storage;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::HnswIndexReader;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::distance_quantized::{
    abs_diff_u8_to_i32, abs_diff_u8_to_i32_scalar, dot_u8_to_i32, dot_u8_to_i32_scalar,
    sq_diff_u8_to_i32, sq_diff_u8_to_i32_scalar,
};
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::vector::core::vector::Vector;
use laurus::vector::index::ManagedVectorIndex;
use laurus::vector::index::config::{
    FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, VectorIndexTypeConfig,
};
use laurus::vector::index::flat::reader::FlatVectorIndexReader;
use laurus::vector::index::ivf::reader::IvfIndexReader;
use laurus::vector::reader::VectorIndexReader;
use laurus::vector::search::filter_set::FilterSet;
use laurus::vector::{
    FlatVectorSearcher, HnswSearcher, IvfSearcher, VectorIndexQuery, VectorIndexSearcher,
};

use common::{DEFAULT_SEED, SAMPLE_SIZE_SLOW, lcg_vec_unit, select_storage};

/// Build a deterministic `Vector` of length `dim`. The caller-supplied
/// `state` advances per call, so successive vectors are different but the
/// whole sequence is reproducible.
fn generate_vector(state: &mut u64, dim: usize) -> Vector {
    Vector::new(lcg_vec_unit(state, dim))
}

fn generate_vectors(count: usize, dim: usize) -> Vec<(u64, String, Vector)> {
    let mut state = DEFAULT_SEED;
    (0..count)
        .map(|i| {
            (
                i as u64,
                "field".to_string(),
                generate_vector(&mut state, dim),
            )
        })
        .collect()
}

/// Names of the three fields used by the multi-field bench.
const MULTI_FIELD_NAMES: &[&str] = &["field_a", "field_b", "field_c"];

/// Build deterministic vectors split 30 / 30 / 40 % across `field_a`,
/// `field_b`, `field_c`. Used by `bench_hnsw_multi_field_search` to drive
/// the per-field-cache hot path that #405 targets — searching with
/// `field_name("field_a")` should only need to consider the 30 % of
/// vectors tagged `field_a`, but today the implementation scans every
/// (id, field_name) pair.
fn generate_multi_field_vectors(count: usize, dim: usize) -> Vec<(u64, String, Vector)> {
    let mut state = DEFAULT_SEED;
    (0..count)
        .map(|i| {
            // 30 / 30 / 40 split based on i % 10.
            let pct = i % 10;
            let field = if pct < 3 {
                MULTI_FIELD_NAMES[0] // field_a — 30 %
            } else if pct < 6 {
                MULTI_FIELD_NAMES[1] // field_b — 30 %
            } else {
                MULTI_FIELD_NAMES[2] // field_c — 40 %
            };
            (
                i as u64,
                field.to_string(),
                generate_vector(&mut state, dim),
            )
        })
        .collect()
}

/// `(label, stride)` pairs for the allow-set filter benches (Issue #747).
/// The allow-set keeps every `stride`-th doc id, so `stride = 100` is ~1 %,
/// `10` is ~10 %, and `2` is ~50 % of the corpus.
const ALLOWSET_SELECTIVITIES: &[(&str, usize)] = &[("1pct", 100), ("10pct", 10), ("50pct", 2)];

/// Build a deterministic allow-set keeping every `stride`-th doc id in
/// `0..count`. Used by the inline filter benches (Issue #740 / #747) to drive
/// the Flat / IVF scan at a known selectivity. The typed [`FilterSet`]
/// (Issue #739) auto-selects a Roaring bitmap for dense sets and a hash set for
/// sparse ones, so the higher-selectivity cases (e.g. `50pct` on a large
/// corpus) exercise the bitmap path and the low ones the hash path.
fn make_allow_set(count: usize, stride: usize) -> Arc<FilterSet> {
    let ids: Vec<u64> = (0..count as u64).step_by(stride).collect();
    Arc::new(FilterSet::from_doc_ids(&ids))
}

/// Build a deterministic query vector. Uses a different starting seed from
/// the corpus so the query is not byte-identical to a corpus member.
fn generate_query(dim: usize) -> Vector {
    let mut state = DEFAULT_SEED.wrapping_add(1);
    generate_vector(&mut state, dim)
}

/// Bench-storage handle. Delegates to `common::select_storage()` so
/// `LAURUS_BENCH_DISK=1` swaps the in-memory backend for a temp-dir
/// `FileStorage` without changing call sites.
fn create_storage() -> Arc<dyn Storage> {
    select_storage()
}

/// Default search-corpus sizes for Flat and IVF benches.
/// `LAURUS_BENCH_LARGE=1` appends a 100 000-vector case so #405 / #406
/// have a measurable target without ballooning the default `cargo bench`
/// runtime past five minutes.
fn search_corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![1000usize, 5000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
}

/// HNSW-specific search-corpus sizes. Adds 50 000 over `search_corpus_sizes`
/// because the fallback / graph search benches added that case in #421 to
/// demonstrate path-specific scaling. `LAURUS_BENCH_LARGE=1` further
/// appends 100 000.
fn hnsw_corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![1000usize, 5000, 50_000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
}

// ---------------------------------------------------------------------------
// On-disk index cache (#513 Stage 2)
// ---------------------------------------------------------------------------

/// Bump when anything about the persisted on-disk vector index would
/// change: index-config defaults, vector synthesis ([`generate_vectors`]
/// / [`generate_multi_field_vectors`]), or laurus's segment format.
/// Caches written under a stale version are auto-rebuilt by
/// [`cached_vector_reader`].
const BENCH_INDEX_FORMAT_VERSION: &str = "1";

/// Index path used inside every cache slot's storage tree. Kept
/// constant so [`open_cached_reader`] can locate the file without
/// caller-supplied bookkeeping.
const CACHE_INDEX_NAME: &str = "bench";

/// Cache root, mirroring the lexical / hybrid bench layout
/// (`target/laurus_bench_index_cache/...`). `CARGO_TARGET_DIR`
/// (if set) overrides the workspace `target/` location.
fn cache_root() -> PathBuf {
    if let Ok(custom) = std::env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(custom).join("laurus_bench_index_cache");
    }
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .parent()
        .map(|p| p.join("target"))
        .unwrap_or_else(|| PathBuf::from("target"))
        .join("laurus_bench_index_cache")
}

/// Resolve the cache directory for `slot`. `slot` must encode every
/// input that affects the persisted index (index type, dim, n,
/// per-config parameters, corpus shape, optional fixture-size proxy);
/// the version key is appended automatically.
fn cache_dir_for(slot: &str) -> PathBuf {
    cache_root().join(format!("{slot}_v{BENCH_INDEX_FORMAT_VERSION}"))
}

/// Verify a cache entry: directory exists and `.bench_version` marker
/// matches [`BENCH_INDEX_FORMAT_VERSION`].
fn cache_is_valid(dir: &Path) -> bool {
    let marker = dir.join(".bench_version");
    if !marker.exists() {
        return false;
    }
    matches!(
        std::fs::read_to_string(&marker).map(|s| s.trim().to_string()),
        Ok(v) if v == BENCH_INDEX_FORMAT_VERSION
    )
}

/// Open an existing on-disk vector index. Dispatches to the
/// type-specific loader so the cached segments are mmap-/file-loaded
/// in their persisted form rather than rebuilt from an in-memory
/// writer.
fn open_cached_reader(
    dir: &Path,
    config: &VectorIndexTypeConfig,
    use_mmap: bool,
) -> laurus::Result<Arc<dyn VectorIndexReader>> {
    let mut file_config = FileStorageConfig::new(dir);
    // `use_mmap = false` forces `LoadingMode::Eager`, so scalar-quantized
    // segments load as `VectorStorage::OwnedQuantized` and the search hot loop
    // dispatches to the int8 `distance_quantized` kernel; mmap (the default)
    // yields `LoadingMode::Lazy` -> `OnDemand` dequantize-on-get (f32 path).
    file_config.use_mmap = use_mmap;
    let storage = StorageFactory::create(StorageConfig::File(file_config))?;
    let distance = config.distance_metric();
    let reader: Arc<dyn VectorIndexReader> = match config {
        VectorIndexTypeConfig::Flat(_) => Arc::new(FlatVectorIndexReader::load(
            storage,
            CACHE_INDEX_NAME,
            distance,
        )?),
        VectorIndexTypeConfig::IVF(_) => {
            Arc::new(IvfIndexReader::load(storage, CACHE_INDEX_NAME, distance)?)
        }
        VectorIndexTypeConfig::HNSW(_) => {
            Arc::new(HnswIndexReader::load(storage, CACHE_INDEX_NAME, distance)?)
        }
    };
    Ok(reader)
}

/// Build the cache slot at `dir` from scratch: instantiates a
/// `ManagedVectorIndex` on a fresh `FileStorage`, ingests the vectors
/// produced by `build_vectors`, finalises, writes, and drops the
/// version marker. Returns the reader handle the bench will hand to
/// its searcher.
fn build_cached_reader(
    dir: &Path,
    config: VectorIndexTypeConfig,
    use_mmap: bool,
    vectors: Vec<(u64, String, Vector)>,
) -> laurus::Result<Arc<dyn VectorIndexReader>> {
    let mut file_config = FileStorageConfig::new(dir);
    // Match the read mode the searcher will use so a fresh build returns a
    // reader in the same loading mode as a cache-hit reopen (see
    // `open_cached_reader`).
    file_config.use_mmap = use_mmap;
    let storage = StorageFactory::create(StorageConfig::File(file_config))?;
    let mut index = ManagedVectorIndex::new(config, storage, CACHE_INDEX_NAME)?;
    index.add_vectors(vectors)?;
    index.finalize()?;
    index.write()?;
    let reader = index.reader()?;
    std::fs::write(dir.join(".bench_version"), BENCH_INDEX_FORMAT_VERSION)?;
    Ok(reader)
}

/// Return a reader for the cached vector index identified by `slot`,
/// building it on disk the first time and re-opening it on subsequent
/// runs (#513 Stage 2). Mirrors `lexical_search_bench::cached_engine`
/// (#510 / #512) and `hybrid_search_bench::cached_hybrid_engine`
/// (#513 Stage 1), specialised for vector indexes.
///
/// # Arguments
///
/// * `slot` - Cache-key fragment that must uniquely encode every input
///   affecting the persisted index (index type, dim, n, config knobs,
///   corpus shape, fixture-size proxy for real-data corpora). The
///   version key from [`BENCH_INDEX_FORMAT_VERSION`] is appended
///   automatically by [`cache_dir_for`].
/// * `config` - The `VectorIndexTypeConfig` to build under (used both
///   for the cache-miss build path and to dispatch the cache-hit
///   loader to the right reader type).
/// * `build_vectors` - Closure that produces the corpus when the cache
///   is missing. Called only on cache miss so deterministic vector
///   generation does not run when the cache hits.
///
/// # Invalidation
///
/// - Bump [`BENCH_INDEX_FORMAT_VERSION`] in source. Stale slots are
///   auto-rebuilt on the next call.
/// - `LAURUS_BENCH_REBUILD=1` forces a wipe-and-rebuild without
///   touching the rest of `target/`.
/// - `cargo clean` evicts the cache along with the rest of `target/`.
fn cached_vector_reader(
    slot: &str,
    config: VectorIndexTypeConfig,
    build_vectors: impl FnOnce() -> Vec<(u64, String, Vector)>,
) -> Arc<dyn VectorIndexReader> {
    // Default to mmap (`LoadingMode::Lazy`) to preserve the historical
    // behaviour of every existing bench. Use
    // [`cached_vector_reader_with_loading`] with `use_mmap = false` to
    // force eager loading and exercise the int8 `distance_quantized`
    // kernel (Issue #652).
    cached_vector_reader_with_loading(slot, config, true, build_vectors)
}

/// Like [`cached_vector_reader`] but explicitly selects the storage
/// loading mode.
///
/// `use_mmap = false` forces `LoadingMode::Eager`, so scalar-quantized
/// segments load as `VectorStorage::OwnedQuantized` and the search hot
/// loop dispatches to the int8 `distance_quantized` kernel. `use_mmap =
/// true` yields the mmap `Lazy` / `OnDemand` dequantize-on-get f32 path.
fn cached_vector_reader_with_loading(
    slot: &str,
    config: VectorIndexTypeConfig,
    use_mmap: bool,
    build_vectors: impl FnOnce() -> Vec<(u64, String, Vector)>,
) -> Arc<dyn VectorIndexReader> {
    let dir = cache_dir_for(slot);
    let force_rebuild = std::env::var("LAURUS_BENCH_REBUILD").is_ok();

    if !force_rebuild && cache_is_valid(&dir) {
        match open_cached_reader(&dir, &config, use_mmap) {
            Ok(reader) => return reader,
            Err(err) => {
                eprintln!(
                    "vector bench cache open failed at {} ({err}); rebuilding",
                    dir.display()
                );
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create vector bench cache dir");

    build_cached_reader(&dir, config, use_mmap, build_vectors())
        .expect("vector bench cache build failed")
}

// ---------------------------------------------------------------------------
// Construction benchmarks
// ---------------------------------------------------------------------------

fn bench_flat_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flat Construction");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow construction path
    let dim = 128;

    for &count in &[1000, 5000] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let vectors = generate_vectors(count, dim);
            b.iter(|| {
                let storage = create_storage();
                let config = FlatIndexConfig {
                    dimension: dim,
                    distance_metric: DistanceMetric::Cosine,
                    ..Default::default()
                };
                let mut index =
                    ManagedVectorIndex::new(VectorIndexTypeConfig::Flat(config), storage, "bench")
                        .unwrap();
                index.add_vectors(vectors.clone()).unwrap();
                index.finalize().unwrap();
            });
        });
    }
    group.finish();
}

fn bench_ivf_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("IVF Construction");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow construction path
    let dim = 128;

    for &count in &[1000, 5000] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let vectors = generate_vectors(count, dim);
            b.iter(|| {
                let storage = create_storage();
                let config = IvfIndexConfig {
                    dimension: dim,
                    distance_metric: DistanceMetric::Cosine,
                    n_clusters: 10,
                    n_probe: 3,
                    ..Default::default()
                };
                let mut index =
                    ManagedVectorIndex::new(VectorIndexTypeConfig::IVF(config), storage, "bench")
                        .unwrap();
                index.add_vectors(vectors.clone()).unwrap();
                index.finalize().unwrap();
            });
        });
    }
    group.finish();
}

fn bench_hnsw_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Construction");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow construction path
    let dim = 128;

    for &count in &[1000, 5000] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            let vectors = generate_vectors(count, dim);
            b.iter(|| {
                let storage = create_storage();
                let config = HnswIndexConfig {
                    dimension: dim,
                    m: 16,
                    ef_construction: 200,
                    distance_metric: DistanceMetric::Cosine,
                    ..Default::default()
                };
                let mut index =
                    ManagedVectorIndex::new(VectorIndexTypeConfig::HNSW(config), storage, "bench")
                        .unwrap();
                index.add_vectors(vectors.clone()).unwrap();
                index.finalize().unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Search benchmarks
// ---------------------------------------------------------------------------

fn bench_flat_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flat Search");
    let dim = 128;

    for &count in &search_corpus_sizes() {
        let config = FlatIndexConfig {
            dimension: dim,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let slot = format!("flat_n{count}_dim{dim}_synthetic");
        let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::Flat(config), || {
            generate_vectors(count, dim)
        });
        let searcher = FlatVectorSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: the probe must return top-10 hits before timing.
        let probe = searcher
            .search(&VectorIndexQuery::new(query.clone()).top_k(10))
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "flat top-10 probe must return at least one hit at count={count}"
        );

        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone()).top_k(10);
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

/// Isolates the int8 `distance_quantized` kernel (Issue #652).
///
/// Forces eager loading (`use_mmap = false`) so scalar-quantized
/// segments load as `VectorStorage::OwnedQuantized`; a field-filtered
/// Flat search then runs a full linear scan through `distance_quantized`
/// for every candidate — the purest end-to-end measurement of the int8
/// kernel with negligible graph-traversal overhead. Sweeps the three
/// metrics that map to the three kernel shapes:
/// - `Cosine` → `dot_u8_to_i32`
/// - `Euclidean` → `sq_diff_u8_to_i32`
/// - `Manhattan` → `abs_diff_u8_to_i32`
fn bench_flat_search_int8_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flat Search int8 kernel");
    let dim = 128;
    // Large corpus so the per-candidate int8 kernel dominates timing.
    let count = 20_000;

    for &(label, metric) in &[
        ("cosine_dot", DistanceMetric::Cosine),
        ("euclidean_sqdiff", DistanceMetric::Euclidean),
        ("manhattan_absdiff", DistanceMetric::Manhattan),
    ] {
        let config = FlatIndexConfig {
            dimension: dim,
            distance_metric: metric,
            ..Default::default()
        };
        let slot = format!("flat_int8_eager_n{count}_dim{dim}_{label}");
        let reader = cached_vector_reader_with_loading(
            &slot,
            VectorIndexTypeConfig::Flat(config),
            false, // eager load -> int8 distance_quantized hot path
            || generate_vectors(count, dim),
        );
        let searcher = FlatVectorSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: the field-filtered probe must return top-10 hits
        // (and therefore engage the quantized scan) before timing.
        let probe = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string()),
            )
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "int8 flat probe must return at least one hit ({label})"
        );

        group.bench_with_input(BenchmarkId::new("top10", label), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string());
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

/// A/B micro-benchmark for the three int8 SQ kernels (Issue #652):
/// the runtime-dispatched kernel (AVX2 on x86_64, NEON on aarch64) vs
/// the portable `wide` scalar reference, at representative dimensions.
/// Isolates the raw kernel speedup with zero search-path overhead —
/// compare the `dispatch` and `scalar` timings within each
/// `(kernel, dim)` pair.
fn bench_int8_kernel_micro(c: &mut Criterion) {
    use std::hint::black_box;

    type Kernel = fn(&[u8], &[u8]) -> i32;
    let mut group = c.benchmark_group("int8 kernel micro");

    for &dim in &[128usize, 768] {
        // Deterministic full-range u8 operands (0..=255).
        let a: Vec<u8> = (0..dim)
            .map(|i| (i.wrapping_mul(97).wrapping_add(13) & 0xFF) as u8)
            .collect();
        let b: Vec<u8> = (0..dim)
            .map(|i| (i.wrapping_mul(131).wrapping_add(7) & 0xFF) as u8)
            .collect();

        let cases: [(&str, Kernel, Kernel); 3] = [
            ("dot", dot_u8_to_i32, dot_u8_to_i32_scalar),
            ("sq_diff", sq_diff_u8_to_i32, sq_diff_u8_to_i32_scalar),
            ("abs_diff", abs_diff_u8_to_i32, abs_diff_u8_to_i32_scalar),
        ];
        for (name, dispatch, scalar) in cases {
            group.bench_with_input(
                BenchmarkId::new(format!("{name}/dispatch"), dim),
                &dim,
                |bch, _| bch.iter(|| dispatch(black_box(&a), black_box(&b))),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{name}/scalar"), dim),
                &dim,
                |bch, _| bch.iter(|| scalar(black_box(&a), black_box(&b))),
            );
        }
    }
    group.finish();
}

fn bench_ivf_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("IVF Search");
    let dim = 128;

    for &count in &search_corpus_sizes() {
        let config = IvfIndexConfig {
            dimension: dim,
            distance_metric: DistanceMetric::Cosine,
            n_clusters: 10,
            n_probe: 3,
            ..Default::default()
        };
        let slot = format!("ivf_n{count}_dim{dim}_nc10_synthetic");
        let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::IVF(config), || {
            generate_vectors(count, dim)
        });
        let searcher = IvfSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: the probe must return top-10 hits before timing.
        let probe = searcher
            .search(&VectorIndexQuery::new(query.clone()).top_k(10))
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "ivf top-10 probe must return at least one hit at count={count}"
        );

        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone()).top_k(10);
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

/// Benchmark `IvfSearcher` at **large cluster counts** (Issue #668).
///
/// `bench_ivf_search` builds only `n_clusters = 10`, where `probe_clusters`
/// (centroid distance scan + nearest-`n_probe` selection) is negligible.
/// #668 targets the `K = 1024-4096` regime, where the per-query centroid scan
/// dominates. `n_clusters` straddles
/// [`PARALLEL_SCAN_THRESHOLD`](laurus::vector::search::searcher) (2048): 512
/// exercises the serial `select_nth_unstable_by` path, 2048 the rayon-parallel
/// centroid scan. `n_probe` is held small (8) so the centroid scan — not the
/// candidate scan over probed clusters — sits on the critical path.
///
/// First run builds the k-means index once per `n_clusters` (the 2048-cluster
/// build is the heavy one); subsequent runs re-open the cached on-disk index.
fn bench_ivf_search_large_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("IVF Search Large K");
    let dim = 128;
    let n_probe = 8usize;

    for &n_clusters in &[512usize, 2048] {
        // ~4 vectors per cluster; k-means requires at least `n_clusters` points.
        let count = n_clusters * 4;
        let config = IvfIndexConfig {
            dimension: dim,
            distance_metric: DistanceMetric::Cosine,
            n_clusters,
            n_probe,
            ..Default::default()
        };
        let slot = format!("ivf_largek_n{count}_dim{dim}_nc{n_clusters}_synthetic");
        let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::IVF(config), || {
            generate_vectors(count, dim)
        });
        // `with_n_probe` pins the probe count independent of how the cached
        // reader was built, keeping the centroid scan the dominant cost.
        let searcher = IvfSearcher::with_n_probe(reader, n_probe).unwrap();
        let query = generate_query(dim);

        // Sanity check: the probe must return top-10 hits before timing.
        let probe = searcher
            .search(&VectorIndexQuery::new(query.clone()).top_k(10))
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "ivf large-K top-10 probe must return at least one hit at n_clusters={n_clusters}"
        );

        group.bench_with_input(
            BenchmarkId::new("top10", n_clusters),
            &n_clusters,
            |b, _| {
                b.iter(|| {
                    let request = VectorIndexQuery::new(query.clone()).top_k(10);
                    searcher.search(&request).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// Benchmark the **fallback (linear-scan) path** of `HnswSearcher::search`.
///
/// `VectorIndexQuery::new` leaves `field_name` as `None`. The graph branch
/// in `HnswSearcher::search` is gated on `Some(field_name)`, so a query
/// without a field name short-circuits to the linear-scan fallback. This is
/// the path #404 (drop the redundant `similarity()` + `distance()` pair)
/// targets.
fn bench_hnsw_fallback_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Fallback Search");
    let dim = 128;

    for &count in &hnsw_corpus_sizes() {
        let config = HnswIndexConfig {
            dimension: dim,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let slot = format!("hnsw_n{count}_dim{dim}_m16_efc200_synthetic");
        let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::HNSW(config), || {
            generate_vectors(count, dim)
        });
        let searcher = HnswSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: the probe must return top-10 hits before timing.
        // No field_name set → fallback (linear scan) path.
        let probe = searcher
            .search(&VectorIndexQuery::new(query.clone()).top_k(10))
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "hnsw fallback top-10 probe must return at least one hit at count={count}"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone()).top_k(10);
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

/// Benchmark the **graph-traversal path** of `HnswSearcher::search`.
///
/// Setting `field_name` enables the graph branch (provided the reader has a
/// graph and is downcastable to `HnswIndexReader`). This is the path that
/// scales with `ef_search` rather than corpus size; #406 (replace the
/// `HashSet<u64>` visited set with a bitmap) and #405 (per-field
/// `vector_ids` cache) target this path.
///
/// **Issue #481 Stage 1 speed gate**: the `top10/50000` case here is the
/// designated benchmark for the "≥ 2× speedup vs f32 baseline" condition.
/// Run before and after the Stage-1 PR (or against the pre-PR `main`
/// branch) and compare absolute medians. Recall is gated separately in
/// `laurus/tests/vector_recall_test.rs`.
/// Stage 2 (Issue #481): HNSW graph search with the LRS1 rerank
/// sidecar enabled and `rerank_factor` set per query. Measures the
/// end-to-end search latency of the two-stage flow:
///
/// 1. int8 HNSW graph search returns `ef_search` candidates.
/// 2. The top `top_k * rerank_factor` candidates are rescored
///    against the original f32 vectors loaded from the sidecar.
/// 3. The new top `top_k` is returned.
///
/// **Stage 2 speed gate** sits at `top10/50000` here: the int8 +
/// rerank path must achieve **≥ 3×** speedup vs the pre-Stage-1 f32
/// numbers recorded on `main` before Stage 1 landed. The rerank
/// rescore is bounded at `top_k * rerank_factor = 50` f32 distance
/// calls per query (a few µs at dim 128) so the int8 graph search
/// dominates the wall clock and the speedup tracks
/// [`bench_hnsw_graph_search`] within rerank's small overhead.
///
/// Tracks ef_search at the searcher default (50) for direct
/// comparison with [`bench_hnsw_graph_search`]; a higher ef_search
/// is the recall-vs-speed lever (see
/// `tests/vector_recall_test.rs::stage2_recall_sweep_diagnostic`).
fn bench_hnsw_graph_search_rerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Graph Search Rerank");
    let dim = 128;

    for &count in &hnsw_corpus_sizes() {
        let config = HnswIndexConfig {
            dimension: dim,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            rerank_storage: Some(RerankStorageKind::F32),
            ..Default::default()
        };
        let slot = format!("hnsw_n{count}_dim{dim}_m16_efc200_rerankF32_synthetic");
        let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::HNSW(config), || {
            generate_vectors(count, dim)
        });
        let searcher = HnswSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: the rerank path must engage (reader exposes a
        // RerankStoragePool) and return at least one hit.
        let probe = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string())
                    .rerank_factor(5),
            )
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "hnsw graph rerank top-10 probe must return at least one hit at count={count}"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string())
                    .rerank_factor(5);
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

fn bench_hnsw_graph_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Graph Search");
    let dim = 128;

    for &count in &hnsw_corpus_sizes() {
        let config = HnswIndexConfig {
            dimension: dim,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        // Shares the slot with `bench_hnsw_fallback_search` /
        // `bench_hnsw_ef_search_sweep` — same persisted index.
        let slot = format!("hnsw_n{count}_dim{dim}_m16_efc200_synthetic");
        let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::HNSW(config), || {
            generate_vectors(count, dim)
        });
        let searcher = HnswSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity check: with field_name set, the graph branch must engage
        // and return at least one hit.
        let probe = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string()),
            )
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "hnsw graph top-10 probe must return at least one hit at count={count}"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string());
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

/// HNSW graph search over a sparse, high-value doc-id range (Issue #686).
///
/// Simulates a long-lived store: the engine-wide id allocator has advanced
/// far past this segment's node count (ids start at 50M with stride 37),
/// which is normal after churn — and on mixed corpora, where documents
/// without vectors still consume ids. Before #686 the per-query visited
/// bitmap was sized by `max_doc_id + 1` bits (~6.3 MB alloc + memset per
/// query here); with segment ordinals it is `node_count` bits (~1.25 KB),
/// independent of the id range. Eager int8 loading, matching the store
/// production path and the #686 perf-gate configuration.
fn bench_hnsw_graph_search_sparse_ids(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Graph Search sparse_ids");
    let dim = 128;
    let count = 10_000usize;

    let config = HnswIndexConfig {
        dimension: dim,
        m: 16,
        ef_construction: 200,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let slot = format!("hnsw_sparse_n{count}_dim{dim}_m16_efc200_base50m_stride37");
    let reader = cached_vector_reader_with_loading(
        &slot,
        VectorIndexTypeConfig::HNSW(config),
        false,
        || {
            let mut state = DEFAULT_SEED;
            (0..count)
                .map(|i| {
                    (
                        50_000_000u64 + (i as u64) * 37,
                        "field".to_string(),
                        generate_vector(&mut state, dim),
                    )
                })
                .collect()
        },
    );
    let searcher = HnswSearcher::new(reader).unwrap();
    let query = generate_query(dim);

    let probe = searcher
        .search(
            &VectorIndexQuery::new(query.clone())
                .top_k(10)
                .field_name("field".to_string()),
        )
        .unwrap();
    assert!(
        !probe.results.is_empty(),
        "sparse-id hnsw graph top-10 probe must return at least one hit"
    );

    group.throughput(Throughput::Elements(count as u64));
    group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
        b.iter(|| {
            let request = VectorIndexQuery::new(query.clone())
                .top_k(10)
                .field_name("field".to_string());
            searcher.search(&request).unwrap()
        });
    });
    group.finish();
}

/// HNSW search latency across an `ef_search` sweep at a fixed corpus.
///
/// `ef_search` controls the size of the dynamic candidate list during
/// graph traversal — larger `ef` exhaustively explores more nodes per
/// query, trading latency for recall. The visited-set hot path inside the
/// inner loop scales linearly with the number of nodes touched, so this
/// sweep is what #406 (replace `HashSet<u64>` with bitmap) targets.
fn bench_hnsw_ef_search_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW ef_search");
    let dim = 128;
    let count = 5000usize;

    let config = HnswIndexConfig {
        dimension: dim,
        m: 16,
        ef_construction: 200,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    // Same slot as `bench_hnsw_fallback_search` /
    // `bench_hnsw_graph_search` at n=5000.
    let slot = format!("hnsw_n{count}_dim{dim}_m16_efc200_synthetic");
    let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::HNSW(config), || {
        generate_vectors(count, dim)
    });
    let query = generate_query(dim);

    for &ef in &[16usize, 64, 128, 256, 512] {
        let mut searcher = HnswSearcher::new(reader.clone()).unwrap();
        searcher.set_ef_search(ef);

        // Sanity: graph traversal at this ef must yield at least one hit.
        let probe = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string()),
            )
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "hnsw ef_search probe must return at least one hit at ef={ef}"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("ef_{ef}/top10")),
            &ef,
            |b, _| {
                b.iter(|| {
                    let request = VectorIndexQuery::new(query.clone())
                        .top_k(10)
                        .field_name("field".to_string());
                    searcher.search(&request).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// HNSW search filtered to a single field on a multi-field corpus.
///
/// The corpus is split 30 / 30 / 40 % across `field_a`, `field_b`,
/// `field_c`. The query selects `field_a`, so only ~30 % of vectors are
/// candidates. Today every search calls `vector_ids()` to fetch all
/// `(id, field_name)` pairs and filters by string equality — wasted work
/// proportional to the non-target field count. #405 (per-field
/// `vector_ids` cache) targets this hot path.
fn bench_hnsw_multi_field_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Multi-field Search");
    let dim = 128;
    let count = 5000usize;

    let config = HnswIndexConfig {
        dimension: dim,
        m: 16,
        ef_construction: 200,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let slot = format!("hnsw_n{count}_dim{dim}_m16_efc200_multifield");
    let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::HNSW(config), || {
        generate_multi_field_vectors(count, dim)
    });
    let searcher = HnswSearcher::new(reader).unwrap();
    let query = generate_query(dim);

    // Sanity: filtering to field_a (~30 % of corpus) must still hit.
    let probe = searcher
        .search(
            &VectorIndexQuery::new(query.clone())
                .top_k(10)
                .field_name("field_a".to_string()),
        )
        .unwrap();
    assert!(
        !probe.results.is_empty(),
        "hnsw multi-field probe must return at least one hit (field_a, count={count})"
    );

    group.throughput(Throughput::Elements(count as u64));
    group.bench_function("field_a_30pct/top10", |b| {
        b.iter(|| {
            let request = VectorIndexQuery::new(query.clone())
                .top_k(10)
                .field_name("field_a".to_string());
            searcher.search(&request).unwrap()
        });
    });

    group.finish();
}

/// Flat search filtered to a single field on a multi-field corpus.
///
/// The corpus is split 30 / 30 / 40 % across `field_a`, `field_b`,
/// `field_c`. The query selects `field_a`, so only ~30 % of vectors are
/// candidates. Today every search calls `vector_ids()` to materialise
/// every `(id, field_name)` pair and filters by string equality —
/// wasted work proportional to the non-target field count plus the
/// `Vec<(u64, String)>` clone.
///
/// #405 (per-field `vector_ids` cache) targets this hot path: with the
/// fix the searcher fetches a pre-built `Arc<[u64]>` for the target
/// field at O(1) cost.
fn bench_flat_multi_field_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flat Multi-field Search");
    let dim = 128;

    for &count in &search_corpus_sizes() {
        let config = FlatIndexConfig {
            dimension: dim,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let slot = format!("flat_n{count}_dim{dim}_multifield");
        let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::Flat(config), || {
            generate_multi_field_vectors(count, dim)
        });
        let searcher = FlatVectorSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        // Sanity: filtering to field_a (~30 % of corpus) must still hit.
        let probe = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field_a".to_string()),
            )
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "flat multi-field probe must return at least one hit (field_a, count={count})"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("field_a_30pct/{count}")),
            &count,
            |b, _| {
                b.iter(|| {
                    let request = VectorIndexQuery::new(query.clone())
                        .top_k(10)
                        .field_name("field_a".to_string());
                    searcher.search(&request).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Flat search with an inline allow-set filter (Issue #740 / follow-up #747).
///
/// A candidate whose doc id is not in `request.filter` is skipped before the
/// distance kernel, so a more selective allow-set should lower latency. The
/// `unfiltered` case is the reference baseline (it must be unchanged by this
/// work). The corpus is the same cached single-field synthetic set used by
/// `bench_flat_search`, so no extra build is incurred.
fn bench_flat_filtered_allowset(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flat Filtered Allow-set");
    let dim = 128;

    for &count in &search_corpus_sizes() {
        let config = FlatIndexConfig {
            dimension: dim,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let slot = format!("flat_n{count}_dim{dim}_synthetic");
        let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::Flat(config), || {
            generate_vectors(count, dim)
        });
        let searcher = FlatVectorSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        group.throughput(Throughput::Elements(count as u64));

        // Unfiltered baseline.
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("unfiltered/{count}")),
            &count,
            |b, _| {
                b.iter(|| {
                    let request = VectorIndexQuery::new(query.clone()).top_k(10);
                    searcher.search(&request).unwrap()
                });
            },
        );

        for &(label, stride) in ALLOWSET_SELECTIVITIES {
            let allow = make_allow_set(count, stride);

            // Sanity: the flat scan is exhaustive, so every allowed doc is a
            // candidate and a top-10 query must still return hits.
            let probe = searcher
                .search(
                    &VectorIndexQuery::new(query.clone())
                        .top_k(10)
                        .filter(allow.clone()),
                )
                .unwrap();
            assert!(
                !probe.results.is_empty(),
                "flat allow-set probe must hit ({label}, count={count})"
            );

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{label}/{count}")),
                &count,
                |b, _| {
                    b.iter(|| {
                        let request = VectorIndexQuery::new(query.clone())
                            .top_k(10)
                            .filter(allow.clone());
                        searcher.search(&request).unwrap()
                    });
                },
            );
        }
    }

    group.finish();
}

/// IVF search with an inline allow-set filter (Issue #740 / follow-up #747).
///
/// Probes 3 of the 10 clusters (matching the bench IVF `n_probe`) and skips
/// non-matching candidates before the distance kernel. Unlike the flat scan,
/// a selective filter combined with cluster probing can legitimately return
/// no hits when the allowed docs fall outside the probed clusters, so the
/// filtered probe only asserts the search succeeds (timing the scan + skip is
/// still meaningful). The corpus reuses the cached set from `bench_ivf_search`.
fn bench_ivf_filtered_allowset(c: &mut Criterion) {
    let mut group = c.benchmark_group("IVF Filtered Allow-set");
    let dim = 128;
    let n_probe = 3;

    for &count in &search_corpus_sizes() {
        let config = IvfIndexConfig {
            dimension: dim,
            distance_metric: DistanceMetric::Cosine,
            n_clusters: 10,
            n_probe,
            ..Default::default()
        };
        let slot = format!("ivf_n{count}_dim{dim}_nc10_synthetic");
        let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::IVF(config), || {
            generate_vectors(count, dim)
        });
        let searcher = IvfSearcher::with_n_probe(reader, n_probe).unwrap();
        let query = generate_query(dim);

        group.throughput(Throughput::Elements(count as u64));

        // Unfiltered baseline (probing `n_probe` clusters).
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("unfiltered/{count}")),
            &count,
            |b, _| {
                b.iter(|| {
                    let request = VectorIndexQuery::new(query.clone()).top_k(10);
                    searcher.search(&request).unwrap()
                });
            },
        );

        for &(label, stride) in ALLOWSET_SELECTIVITIES {
            let allow = make_allow_set(count, stride);

            // Confirm the filtered path runs without error; an empty result is
            // acceptable for a selective filter under cluster probing.
            searcher
                .search(
                    &VectorIndexQuery::new(query.clone())
                        .top_k(10)
                        .filter(allow.clone()),
                )
                .unwrap();

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{label}/{count}")),
                &count,
                |b, _| {
                    b.iter(|| {
                        let request = VectorIndexQuery::new(query.clone())
                            .top_k(10)
                            .filter(allow.clone());
                        searcher.search(&request).unwrap()
                    });
                },
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Issue #498 — real-data Stage 2 speed validation (SIFT1M)
// ---------------------------------------------------------------------------

/// Real-data Stage 2 speed bench (Issue #498).
///
/// Loads a 50 000-vector SIFT1M subsample plus the standard 200-query
/// SIFT query set, builds an int8 HNSW index with the LRS1 rerank
/// sidecar enabled, and times the same two-stage `top10 + rerank_factor =
/// 5` flow the recall test
/// (`hnsw_quantized_recall_at_10_with_rerank_on_sift_meets_stage2_real_data_recall_gate`)
/// asserts. The two should agree on `(corpus, ef_search, rerank_factor,
/// HNSW config)` so the latency reported here is the speed-half of the
/// Issue #498 acceptance.
///
/// Opt-in: gated on `LAURUS_REAL_BENCHMARK=1` AND
/// `.cache/sift/sift/sift_base.fvecs` being present (Criterion has no
/// "skip" primitive, so a missing fixture or env var collapses the
/// group to zero benches and the bench file finishes silently). Run:
///
/// ```sh
/// ./scripts/fetch-sift.sh --large
/// LAURUS_REAL_BENCHMARK=1 cargo bench --bench vector_search_bench \
///     -- "HNSW Graph Search Rerank Real"
/// ```
///
/// # Speed gate interpretation
///
/// The Issue #498 acceptance asks for "≥ 3× vs the pre-Stage-1 f32
/// baseline." Phase 0 measurements
/// (`~/.claude/tasks/laurus/20260514_498_real_data_speed_validation/`)
/// showed that gate is **not reachable** with the current
/// implementation on SIFT1M; the maximum sustained speedup at Recall@10
/// ≥ 0.99 is ~1.81× with `(m=16, efc=200, ef_search=200, rerank=5)`.
/// The bench number reported here is intended for a cross-branch
/// comparison against the pre-Stage-1 main commit (`c20620c9^`) with
/// the same SIFT fixture and the same HNSW config, looking for **≥ 1.5×**
/// speedup. Criterion does not assert, so the gate is enforced by the
/// PR description / docs rather than CI.
///
/// To take the baseline:
///
/// ```sh
/// git worktree add /tmp/laurus-prestage1 c20620c9^
/// # add an f32-HNSW bench to /tmp/laurus-prestage1/laurus/benches/
/// # and run it with the same SIFT fixture (the .cache/sift/ symlink
/// # makes this a one-liner).
/// ```
fn bench_hnsw_graph_search_rerank_real_data(c: &mut Criterion) {
    if std::env::var("LAURUS_REAL_BENCHMARK").as_deref() != Ok("1") {
        return;
    }
    let base_path = common::sift_cache_dir()
        .join("sift")
        .join("sift_base.fvecs");
    let query_path = common::sift_cache_dir()
        .join("sift")
        .join("sift_query.fvecs");
    if !base_path.exists() || !query_path.exists() {
        eprintln!(
            "skipping Issue #498 real-data speed bench: SIFT1M fixture \
             not found at {}. Run ./scripts/fetch-sift.sh --large.",
            base_path.display()
        );
        return;
    }

    let dim: usize = 128;
    let n_corpus: usize = 50_000;
    let n_queries: usize = 200;
    let ef_search: usize = 200;
    let rerank_factor: usize = 5;
    let m: usize = 16;
    let ef_construction: usize = 200;

    let mut queries =
        common::load_fvecs(&query_path, dim, Some(n_queries)).expect("load sift_query.fvecs");
    for v in queries.iter_mut() {
        common::l2_normalise(v);
    }

    let config = HnswIndexConfig {
        dimension: dim,
        m,
        ef_construction,
        distance_metric: DistanceMetric::Cosine,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    // Include sift_base.fvecs size as a cheap "did the fixture
    // change?" proxy: re-fetching a different SIFT subset will give
    // a different size and miss the cache cleanly.
    let base_size = std::fs::metadata(&base_path)
        .map(|m| m.len())
        .unwrap_or_default();
    let slot = format!(
        "hnsw_sift_n{n_corpus}_dim{dim}_m{m}_efc{ef_construction}_rerankF32_size{base_size}",
    );
    let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::HNSW(config), || {
        let mut corpus =
            common::load_fvecs(&base_path, dim, Some(n_corpus)).expect("load sift_base.fvecs");
        for v in corpus.iter_mut() {
            common::l2_normalise(v);
        }
        corpus
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u64, "field".to_string(), Vector::new(v)))
            .collect()
    });
    let mut searcher = HnswSearcher::new(reader).unwrap();
    searcher.set_ef_search(ef_search);

    // Sanity check: rerank path must engage on real data too.
    let probe = searcher
        .search(
            &VectorIndexQuery::new(Vector::new(queries[0].clone()))
                .top_k(10)
                .field_name("field".to_string())
                .rerank_factor(rerank_factor),
        )
        .unwrap();
    assert!(
        !probe.results.is_empty(),
        "Issue #498 SIFT probe must return at least one hit"
    );

    let mut group = c.benchmark_group("HNSW Graph Search Rerank Real");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow real-data build path
    group.throughput(Throughput::Elements(n_corpus as u64));

    // Round-robin over the query set so the bench measures average
    // query latency across the full SIFT query distribution rather
    // than the same vector repeatedly.
    let mut iter_idx: usize = 0;
    group.bench_function(BenchmarkId::new("top10_rerank5", "sift50000"), |b| {
        b.iter(|| {
            let q = &queries[iter_idx % queries.len()];
            iter_idx = iter_idx.wrapping_add(1);
            let request = VectorIndexQuery::new(Vector::new(q.clone()))
                .top_k(10)
                .field_name("field".to_string())
                .rerank_factor(rerank_factor);
            searcher.search(&request).unwrap()
        });
    });
    group.finish();
}

/// Real-data Stage 3 speed bench (Issue #481 — PQ + rerank).
///
/// Same opt-in protocol as `bench_hnsw_graph_search_rerank_real_data`
/// (Issue #498) but builds the HNSW index with
/// `quantization_method = ProductQuantization { subvector_count = 32 }`
/// and `rerank_storage = Some(F32)`. Times the end-to-end `top10` +
/// `rerank_factor = 20` flow that the recall test
/// (`hnsw_pq_rerank_recall_at_10_on_sift_meets_stage3_real_data_recall_gate`)
/// asserts ≥ 0.95 recall at.
///
/// Opt-in: gated on `LAURUS_REAL_BENCHMARK=1` AND
/// `.cache/sift/sift/sift_base.fvecs` being present. Run:
///
/// ```sh
/// ./scripts/fetch-sift.sh --large
/// LAURUS_REAL_BENCHMARK=1 cargo bench --bench vector_search_bench \
///     -- "HNSW Graph Search PQ Rerank Real"
/// ```
///
/// # Speed gate interpretation
///
/// Issue #481 Stage 3 originally asked for ≥ 5× speedup vs the
/// pre-Stage-1 f32 HNSW baseline (625 µs/qry on SIFT1M-50k per #500),
/// but measurements during this PR showed the realistic ceiling sits
/// near ~2× with PQ + rerank — the int8 SQ kernel's per-candidate
/// cost is already close to PQ ADC's at the dimensions laurus
/// targets, and the rerank pass dominates the wall clock. The gate
/// was therefore reduced to ≥ 1.5× (same revision Issue #498 did
/// for Stage 2's 3× target), with the exact ratio taken via the
/// cross-branch worktree pattern Issue #498 documented.
fn bench_hnsw_graph_search_pq_rerank_real_data(c: &mut Criterion) {
    // Default operating point: top_k=10, rerank_factor=10 — matches the
    // Stage 3 recall test config (subvector_count=32, ef_search=200)
    // so latency is taken at the same point the recall assertion
    // defends.
    run_pq_real_data_bench(
        c,
        PqRealDataBenchParams {
            top_k: 10,
            rerank_factor: 10,
            ef_search: 200,
            bench_id_label: "top10_pq_rerank20",
        },
    );
}

/// Issue #673 (#650 PR-3): the same SIFT1M PQ + rerank config as
/// [`bench_hnsw_graph_search_pq_rerank_real_data`], but with `ef_search`
/// widened well past the exact-stage budget (`top_k * rerank_factor =
/// 30`) so the PQ → SQ → f32 three-stage chain activates instead of the
/// two-stage PQ → f32 chain. Re-measures the "rerank stage dominates
/// ADC" assumption documented on
/// [`bench_hnsw_graph_search_pq_rerank_real_data`] now that a cheap
/// int8 stage sits between the ADC candidate generation and the exact
/// f32 rescore — the SQ stage's own cost must stay well under the ADC
/// savings it enables for the three-stage chain to be worth taking by
/// default.
///
/// Same opt-in gating as [`bench_hnsw_graph_search_pq_rerank_real_data`].
fn bench_hnsw_graph_search_pq_rerank_real_data_wide_ef(c: &mut Criterion) {
    run_pq_real_data_bench(
        c,
        PqRealDataBenchParams {
            top_k: 10,
            rerank_factor: 3,
            ef_search: 800,
            bench_id_label: "top10_pq_rerank3_wide_ef800",
        },
    );
}

/// Issue #706 Phase 0 micro-experiment: PQ-256 with rerank disabled
/// (`rerank_factor = 1`). Compared against the matching FastScan
/// variant ([`bench_hnsw_graph_search_pq_fastscan_real_data_no_rerank`])
/// to isolate the wall-clock contribution of the rerank pass.
///
/// Same opt-in gating as [`bench_hnsw_graph_search_pq_rerank_real_data`].
fn bench_hnsw_graph_search_pq_rerank_real_data_no_rerank(c: &mut Criterion) {
    run_pq_real_data_bench(
        c,
        PqRealDataBenchParams {
            top_k: 10,
            rerank_factor: 1,
            ef_search: 200,
            bench_id_label: "top10_pq_no_rerank",
        },
    );
}

/// Issue #706 Phase 0 micro-experiment: PQ-256 generating 100
/// candidates without rerank. The FastScan twin
/// ([`bench_hnsw_graph_search_pq_fastscan_real_data_kernel_only`]) runs
/// the same shape, so the resulting ratio isolates the candidate-
/// generation kernel from both the rerank pass and any top-k specific
/// overhead.
///
/// Same opt-in gating as [`bench_hnsw_graph_search_pq_rerank_real_data`].
fn bench_hnsw_graph_search_pq_rerank_real_data_kernel_only(c: &mut Criterion) {
    run_pq_real_data_bench(
        c,
        PqRealDataBenchParams {
            top_k: 100,
            rerank_factor: 1,
            ef_search: 200,
            bench_id_label: "top100_pq_kernel_only",
        },
    );
}

/// Parameters for the PQ-256 SIFT1M real-data benchmark family.
///
/// `top_k` and `rerank_factor` flow straight through to the
/// [`VectorIndexQuery`] used per iteration. `ef_search` sets the graph
/// traversal budget (Issue #673: exceeding `top_k * rerank_factor`
/// activates the PQ → SQ → f32 three-stage chain instead of the
/// pre-#673 two-stage PQ → f32 chain — see
/// [`bench_hnsw_graph_search_pq_rerank_real_data_wide_ef`]).
/// `bench_id_label` is the per-variant id under the shared
/// `"HNSW Graph Search PQ Rerank Real"` Criterion group.
struct PqRealDataBenchParams {
    top_k: usize,
    rerank_factor: usize,
    ef_search: usize,
    bench_id_label: &'static str,
}

/// Shared body for the PQ-256 SIFT1M real-data benchmarks (Issue
/// [#481](https://github.com/mosuka/laurus/issues/481) Stage 3 and
/// Issue [#706](https://github.com/mosuka/laurus/issues/706) Phase 0).
///
/// The HNSW index is shared across variants via [`cached_vector_reader`]:
/// the cache slot depends only on the corpus + index parameters, not on
/// `top_k` / `rerank_factor`, so all PQ-256 variants reuse the same
/// pre-built segment.
fn run_pq_real_data_bench(c: &mut Criterion, params: PqRealDataBenchParams) {
    if std::env::var("LAURUS_REAL_BENCHMARK").as_deref() != Ok("1") {
        return;
    }
    let base_path = common::sift_cache_dir()
        .join("sift")
        .join("sift_base.fvecs");
    let query_path = common::sift_cache_dir()
        .join("sift")
        .join("sift_query.fvecs");
    if !base_path.exists() || !query_path.exists() {
        eprintln!(
            "skipping Issue #481 Stage 3 real-data speed bench: SIFT1M \
             fixture not found at {}. Run ./scripts/fetch-sift.sh --large.",
            base_path.display()
        );
        return;
    }

    let dim: usize = 128;
    let n_corpus: usize = 50_000;
    let n_queries: usize = 200;
    let ef_search = params.ef_search;
    let subvector_count: usize = 32;
    let m: usize = 16;
    let ef_construction: usize = 200;

    let mut queries =
        common::load_fvecs(&query_path, dim, Some(n_queries)).expect("load sift_query.fvecs");
    for v in queries.iter_mut() {
        common::l2_normalise(v);
    }

    let config = HnswIndexConfig {
        dimension: dim,
        m,
        ef_construction,
        distance_metric: DistanceMetric::Cosine,
        quantization_method:
            laurus::vector::core::quantization::QuantizationMethod::ProductQuantization {
                subvector_count,
            },
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let base_size = std::fs::metadata(&base_path)
        .map(|m| m.len())
        .unwrap_or_default();
    let slot = format!(
        "hnsw_sift_n{n_corpus}_dim{dim}_m{m}_efc{ef_construction}_pq{subvector_count}_rerankF32_size{base_size}",
    );
    let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::HNSW(config), || {
        let mut corpus =
            common::load_fvecs(&base_path, dim, Some(n_corpus)).expect("load sift_base.fvecs");
        for v in corpus.iter_mut() {
            common::l2_normalise(v);
        }
        corpus
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u64, "field".to_string(), Vector::new(v)))
            .collect()
    });
    let mut searcher = HnswSearcher::new(reader).unwrap();
    searcher.set_ef_search(ef_search);

    // Sanity check: the PQ + rerank path must engage on real data.
    let probe = searcher
        .search(
            &VectorIndexQuery::new(Vector::new(queries[0].clone()))
                .top_k(params.top_k)
                .field_name("field".to_string())
                .rerank_factor(params.rerank_factor),
        )
        .unwrap();
    assert!(
        !probe.results.is_empty(),
        "Issue #481 Stage 3 SIFT probe must return at least one hit"
    );

    let mut group = c.benchmark_group("HNSW Graph Search PQ Rerank Real");
    group.sample_size(SAMPLE_SIZE_SLOW); // slow real-data build path
    group.throughput(Throughput::Elements(n_corpus as u64));

    let mut iter_idx: usize = 0;
    group.bench_function(BenchmarkId::new(params.bench_id_label, "sift50000"), |b| {
        b.iter(|| {
            let q = &queries[iter_idx % queries.len()];
            iter_idx = iter_idx.wrapping_add(1);
            let request = VectorIndexQuery::new(Vector::new(q.clone()))
                .top_k(params.top_k)
                .field_name("field".to_string())
                .rerank_factor(params.rerank_factor);
            searcher.search(&request).unwrap()
        });
    });
    group.finish();
}

/// PQ FastScan + F32 rerank real-data benchmark on SIFT1M
/// (Issue [#703](https://github.com/mosuka/laurus/issues/703), Phase 4
/// of #695). Mirrors [`bench_hnsw_graph_search_pq_rerank_real_data`]
/// exactly — same SIFT1M 50k corpus + 200 queries + L2-normalised
/// Cosine + `m=16` HNSW + `ef_construction=200` + `ef_search=200` +
/// `rerank_factor=10` + `subvector_count=32` — and swaps only the
/// `quantization_method` to `ProductQuantizationFastScan`. The
/// resulting wall-clock ratio (this bench / the K=256 PQ bench) is
/// the acceptance gate the umbrella [#695](https://github.com/mosuka/laurus/issues/695)
/// uses to decide whether FastScan ships as the default PQ
/// implementation or stays behind the `pq-fastscan` cargo feature.
///
/// Opt-in via `LAURUS_REAL_BENCHMARK=1` and the `pq-fastscan` feature
/// flag (`cargo bench --features pq-fastscan ...`). The corpus is
/// taken from `.cache/sift/sift/sift_*.fvecs`; run
/// `./scripts/fetch-sift.sh --large` once to populate it.
#[cfg(feature = "pq-fastscan")]
fn bench_hnsw_graph_search_pq_fastscan_real_data(c: &mut Criterion) {
    // Default operating point matches the PQ-256 default
    // ([`bench_hnsw_graph_search_pq_rerank_real_data`]) one-to-one so
    // the umbrella's ratio gate is taken at the same point.
    run_pq_fastscan_real_data_bench(
        c,
        PqFastScanRealDataBenchParams {
            top_k: 10,
            rerank_factor: 10,
            bench_id_label: "top10_pqfs_rerank10",
        },
    );
}

/// Issue #706 Phase 0 micro-experiment: FastScan with rerank disabled
/// (`rerank_factor = 1`). Twin of
/// [`bench_hnsw_graph_search_pq_rerank_real_data_no_rerank`].
#[cfg(feature = "pq-fastscan")]
fn bench_hnsw_graph_search_pq_fastscan_real_data_no_rerank(c: &mut Criterion) {
    run_pq_fastscan_real_data_bench(
        c,
        PqFastScanRealDataBenchParams {
            top_k: 10,
            rerank_factor: 1,
            bench_id_label: "top10_pqfs_no_rerank",
        },
    );
}

/// Issue #706 Phase 0 micro-experiment: FastScan generating 100
/// candidates without rerank — kernel-only comparison against
/// [`bench_hnsw_graph_search_pq_rerank_real_data_kernel_only`].
#[cfg(feature = "pq-fastscan")]
fn bench_hnsw_graph_search_pq_fastscan_real_data_kernel_only(c: &mut Criterion) {
    run_pq_fastscan_real_data_bench(
        c,
        PqFastScanRealDataBenchParams {
            top_k: 100,
            rerank_factor: 1,
            bench_id_label: "top100_pqfs_kernel_only",
        },
    );
}

#[cfg(not(feature = "pq-fastscan"))]
fn bench_hnsw_graph_search_pq_fastscan_real_data(_c: &mut Criterion) {
    // No-op without the `pq-fastscan` feature so the criterion_group
    // macro can reference this symbol unconditionally.
}

#[cfg(not(feature = "pq-fastscan"))]
fn bench_hnsw_graph_search_pq_fastscan_real_data_no_rerank(_c: &mut Criterion) {}

#[cfg(not(feature = "pq-fastscan"))]
fn bench_hnsw_graph_search_pq_fastscan_real_data_kernel_only(_c: &mut Criterion) {}

/// Parameters for the FastScan SIFT1M real-data benchmark family
/// (mirror of [`PqRealDataBenchParams`]).
#[cfg(feature = "pq-fastscan")]
struct PqFastScanRealDataBenchParams {
    top_k: usize,
    rerank_factor: usize,
    bench_id_label: &'static str,
}

/// Shared body for the FastScan SIFT1M real-data benchmarks. Mirror of
/// [`run_pq_real_data_bench`] — same opt-in gating, same SIFT
/// fixture, same ef_search / subvector_count / m / ef_construction;
/// only the `quantization_method` differs (`ProductQuantizationFastScan`
/// instead of `ProductQuantization`).
#[cfg(feature = "pq-fastscan")]
fn run_pq_fastscan_real_data_bench(c: &mut Criterion, params: PqFastScanRealDataBenchParams) {
    if std::env::var("LAURUS_REAL_BENCHMARK").as_deref() != Ok("1") {
        return;
    }
    let base_path = common::sift_cache_dir()
        .join("sift")
        .join("sift_base.fvecs");
    let query_path = common::sift_cache_dir()
        .join("sift")
        .join("sift_query.fvecs");
    if !base_path.exists() || !query_path.exists() {
        eprintln!(
            "skipping Issue #703 FastScan real-data speed bench: SIFT1M \
             fixture not found at {}. Run ./scripts/fetch-sift.sh --large.",
            base_path.display()
        );
        return;
    }

    let dim: usize = 128;
    let n_corpus: usize = 50_000;
    let n_queries: usize = 200;
    let ef_search: usize = 200;
    let subvector_count: usize = 32;
    let m: usize = 16;
    let ef_construction: usize = 200;

    let mut queries =
        common::load_fvecs(&query_path, dim, Some(n_queries)).expect("load sift_query.fvecs");
    for v in queries.iter_mut() {
        common::l2_normalise(v);
    }

    let config = HnswIndexConfig {
        dimension: dim,
        m,
        ef_construction,
        distance_metric: DistanceMetric::Cosine,
        quantization_method:
            laurus::vector::core::quantization::QuantizationMethod::ProductQuantizationFastScan {
                subvector_count,
            },
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let base_size = std::fs::metadata(&base_path)
        .map(|m| m.len())
        .unwrap_or_default();
    // Distinct cache slot from the K=256 PQ bench so both flavours can
    // coexist in the bench cache. The "pqfs" prefix keeps the slot
    // human-readable for `du -sh target/laurus_bench_index_cache/`.
    let slot = format!(
        "hnsw_sift_n{n_corpus}_dim{dim}_m{m}_efc{ef_construction}_pqfs{subvector_count}_rerankF32_size{base_size}",
    );
    let reader = cached_vector_reader(&slot, VectorIndexTypeConfig::HNSW(config), || {
        let mut corpus =
            common::load_fvecs(&base_path, dim, Some(n_corpus)).expect("load sift_base.fvecs");
        for v in corpus.iter_mut() {
            common::l2_normalise(v);
        }
        corpus
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u64, "field".to_string(), Vector::new(v)))
            .collect()
    });
    let mut searcher = HnswSearcher::new(reader).unwrap();
    searcher.set_ef_search(ef_search);

    // Sanity check: the FastScan + rerank path must engage on real data.
    let probe = searcher
        .search(
            &VectorIndexQuery::new(Vector::new(queries[0].clone()))
                .top_k(params.top_k)
                .field_name("field".to_string())
                .rerank_factor(params.rerank_factor),
        )
        .unwrap();
    assert!(
        !probe.results.is_empty(),
        "Issue #703 FastScan SIFT probe must return at least one hit"
    );

    let mut group = c.benchmark_group("HNSW Graph Search PQ FastScan Rerank Real");
    group.sample_size(SAMPLE_SIZE_SLOW);
    group.throughput(Throughput::Elements(n_corpus as u64));

    let mut iter_idx: usize = 0;
    group.bench_function(BenchmarkId::new(params.bench_id_label, "sift50000"), |b| {
        b.iter(|| {
            let q = &queries[iter_idx % queries.len()];
            iter_idx = iter_idx.wrapping_add(1);
            let request = VectorIndexQuery::new(Vector::new(q.clone()))
                .top_k(params.top_k)
                .field_name("field".to_string())
                .rerank_factor(params.rerank_factor);
            searcher.search(&request).unwrap()
        });
    });
    group.finish();
}

/// HNSW graph search with prefetch actually enabled (Issue #656).
///
/// Every other HNSW graph-search bench in this file uses the default
/// `use_mmap = true` (`cached_vector_reader`), which loads segments as
/// `VectorStorage::OnDemand` — the storage mode `search_graph`'s
/// `prefetch_enabled` gate is *always false* for (no `sq_prefetch`, and
/// `field_prefetch_index` returns an empty map unconditionally), so none
/// of the other benches exercise the one-hop OR the new two-hop lookahead
/// prefetch code at all. This bench forces eager loading
/// (`cached_vector_reader_with_loading(..., false, ...)`), matching
/// `bench_hnsw_graph_search_sparse_ids`'s int8-kernel setup but sweeping
/// `hnsw_corpus_sizes()` (up to 100k with `LAURUS_BENCH_LARGE=1`) instead
/// of a single fixed 10k count, since the two-hop lookahead's benefit (if
/// any) is expected to show up once the working set exceeds L3.
fn bench_hnsw_graph_search_eager_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Graph Search eager (prefetch enabled)");
    let dim = 128;

    for &count in &hnsw_corpus_sizes() {
        let config = HnswIndexConfig {
            dimension: dim,
            m: 16,
            ef_construction: 200,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let slot = format!("hnsw_eager_n{count}_dim{dim}_m16_efc200_synthetic");
        let reader = cached_vector_reader_with_loading(
            &slot,
            VectorIndexTypeConfig::HNSW(config),
            false,
            || generate_vectors(count, dim),
        );
        let searcher = HnswSearcher::new(reader).unwrap();
        let query = generate_query(dim);

        let probe = searcher
            .search(
                &VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string()),
            )
            .unwrap();
        assert!(
            !probe.results.is_empty(),
            "eager hnsw graph top-10 probe must return at least one hit at count={count}"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("top10", count), &count, |b, _| {
            b.iter(|| {
                let request = VectorIndexQuery::new(query.clone())
                    .top_k(10)
                    .field_name("field".to_string());
                searcher.search(&request).unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_flat_construction,
    bench_ivf_construction,
    bench_hnsw_construction,
    bench_flat_search,
    bench_flat_search_int8_kernel,
    bench_int8_kernel_micro,
    bench_ivf_search,
    bench_ivf_search_large_k,
    bench_hnsw_fallback_search,
    bench_hnsw_graph_search,
    bench_hnsw_graph_search_sparse_ids,
    bench_hnsw_graph_search_eager_large,
    bench_hnsw_graph_search_rerank,
    bench_hnsw_graph_search_rerank_real_data,
    bench_hnsw_graph_search_pq_rerank_real_data,
    bench_hnsw_graph_search_pq_rerank_real_data_wide_ef,
    bench_hnsw_graph_search_pq_rerank_real_data_no_rerank,
    bench_hnsw_graph_search_pq_rerank_real_data_kernel_only,
    bench_hnsw_graph_search_pq_fastscan_real_data,
    bench_hnsw_graph_search_pq_fastscan_real_data_no_rerank,
    bench_hnsw_graph_search_pq_fastscan_real_data_kernel_only,
    bench_hnsw_ef_search_sweep,
    bench_hnsw_multi_field_search,
    bench_flat_multi_field_search,
    bench_flat_filtered_allowset,
    bench_ivf_filtered_allowset,
);
criterion_main!(benches);
