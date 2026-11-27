//! Hybrid (lexical + vector) search benchmarks.
//!
//! Targets the post-#429 coverage gap that `Engine::search` with both a
//! `lexical_query` and a `vector_query` set was never benchmarked. The
//! original audit (#416 item 8) flagged the fusion stage — particularly
//! the dedup + score-merge across two top-K candidate lists — as a
//! likely O(n²) hotspot. Without a hybrid surface, no fusion-side perf
//! change can post a credible before / after.
//!
//! # Scope
//!
//! Three measurement scenarios:
//!
//! 1. **`bench_hybrid_rrf`** — `lexical_query + vector_query` with
//!    `FusionAlgorithm::RRF { k: 60.0 }`, top-K = 10 at corpus 5 000.
//! 2. **`bench_hybrid_weighted_sum`** — same shape, but
//!    `FusionAlgorithm::WeightedSum { lexical_weight: 0.5,
//!    vector_weight: 0.5 }` so cost-difference between fusion strategies
//!    is measurable.
//! 3. **`bench_hybrid_top_k_sweep`** — RRF fusion at fixed corpus 5 000,
//!    K ∈ {10, 100, 500}. Larger K stresses the dedup / merge path the
//!    most.
//!
//! Optional: `LAURUS_BENCH_LARGE=1` adds a 100 000-doc case.
//!
//! # Vocabulary
//!
//! Mirrors the 3-tier Zipf vocabulary used by `lexical_search_bench` and
//! `lexical_indexing_bench`. Each document also carries a deterministic
//! 128-d vector in the `embedding` HNSW field. Inline copy with a
//! "keep in sync" comment; consolidation deferred.
//!
//! # On-disk index cache (#513 Stage 1)
//!
//! [`cached_hybrid_engine`] persists the built index under
//! `target/laurus_bench_index_cache/hybrid_<n>_dim<DIM>_v<N>/`. Mirrors
//! the lexical bench's `cached_engine` (#510): pay the Phase 2 build
//! cost (~tens of seconds for 5 k, minutes for 100 k including HNSW
//! graph construction) once per fresh checkout; later `cargo bench`
//! runs reopen the cached index in well under a second. Bump
//! [`BENCH_INDEX_FORMAT_VERSION`] when anything that would alter the
//! resulting index changes; `LAURUS_BENCH_REBUILD=1` forces a
//! wipe-and-rebuild. See `benches/BENCHMARKS.md` for the architecture
//! rationale.
//!
//! # Run
//!
//! ```sh
//! # Daily iteration (fast — uses cache after first run):
//! cargo bench --bench hybrid_search_bench
//!
//! # Acceptance / large-corpus sweep:
//! LAURUS_BENCH_LARGE=1 cargo bench --bench hybrid_search_bench
//!
//! # Force a fresh cache build:
//! LAURUS_BENCH_REBUILD=1 cargo bench --bench hybrid_search_bench
//! ```
//!
//! Filter by case (regex match against the criterion id):
//!
//! ```sh
//! cargo bench --bench hybrid_search_bench -- rrf
//! cargo bench --bench hybrid_search_bench -- 'top_k/k_500'
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench hybrid_search_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules and
//! `benches/BENCHMARKS.md` for the cross-cutting bench architecture.

mod common;

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tokio::runtime::Runtime;

use common::{DEFAULT_SEED, SAMPLE_SIZE_FAST, lcg_vec_unit, select_storage};

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::core::field::IntegerOption;
use laurus::lexical::{TermQuery, TextOption};
use laurus::storage::Storage;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::core::vector::Vector;
use laurus::vector::search::searcher::VectorSearchQuery;
use laurus::vector::store::request::QueryVector;
use laurus::{
    Document, Engine, FusionAlgorithm, LexicalSearchQuery, Result, Schema, SearchRequest,
    SearchRequestBuilder,
};

const DIM: usize = 128;
const VECTOR_FIELD: &str = "embedding";

const COMMON_TERMS: &str = "search engine system data query";

const TOPIC_PHRASES: &[&str] = &[
    "rust programming language systems safety concurrency memory ownership",
    "python data science machine learning artificial intelligence numpy",
    "javascript web development frontend backend node react framework",
    "database query optimization indexing performance search engine",
    "network protocol distributed systems cloud computing infrastructure",
    "security cryptography authentication authorization encryption",
    "algorithms data structures sorting searching graph traversal",
    "operating systems kernel processes threads scheduling memory",
];

const LONG_TAIL: &[&str] = &[
    "compaction",
    "histogram",
    "lattice",
    "registry",
    "compiler",
    "scheduler",
    "interpreter",
    "garbage",
    "collector",
    "allocator",
    "telemetry",
    "regression",
    "snapshot",
    "replication",
    "consensus",
    "quorum",
    "leader",
    "follower",
];

const LONG_TAIL_PER_DOC: usize = 5;

const CATEGORIES: &[&str] = &["programming", "data-science", "web", "database", "systems"];

fn build_body(i: usize) -> String {
    let topic = TOPIC_PHRASES[i % TOPIC_PHRASES.len()];

    let mut tail_words = Vec::with_capacity(LONG_TAIL_PER_DOC);
    for k in 0..LONG_TAIL_PER_DOC {
        let idx = (i.wrapping_mul(7) + k * 11) % LONG_TAIL.len();
        tail_words.push(LONG_TAIL[idx]);
    }
    let tail = tail_words.join(" ");

    format!("Document {i} {COMMON_TERMS} {topic} {tail} should match relevant terms")
}

/// Bench-storage handle. Delegates to `common::select_storage()` so
/// `LAURUS_BENCH_DISK=1` swaps the in-memory backend for a temp-dir
/// `FileStorage` without changing call sites. Used by the bench when
/// the on-disk cache is **not** in play (e.g. legacy callers); the
/// `cached_hybrid_engine` helper below routes through its own
/// deterministically-pathed `FileStorage` to persist across runs.
fn memory_storage() -> Result<Arc<dyn Storage>> {
    Ok(select_storage())
}

/// Build the hybrid bench schema. Kept in one place so the persistent
/// and ephemeral build paths cannot drift. Mirroring schema is what
/// lets the cache's `recover()` correctly reattach segments written
/// by an earlier bench run.
fn hybrid_schema() -> Schema {
    Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .add_text_field("category", TextOption::default())
        .add_integer_field("year", IntegerOption::default())
        .add_hnsw_field(
            VECTOR_FIELD,
            HnswOption::new(DIM).distance(DistanceMetric::Cosine),
        )
        .add_default_field("body")
        .build()
}

async fn build_hybrid_engine_into_storage(storage: Arc<dyn Storage>, n: usize) -> Result<Engine> {
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::default());
    let engine = Engine::builder(storage, hybrid_schema())
        .analyzer(analyzer)
        .build()
        .await?;

    let mut state = DEFAULT_SEED;
    for i in 0..n {
        let body = build_body(i);
        let vec = lcg_vec_unit(&mut state, DIM);
        let doc = Document::builder()
            .add_text("title", format!("Title for document {i}"))
            .add_text("body", &body)
            .add_text("category", CATEGORIES[i % CATEGORIES.len()])
            .add_integer("year", 2020 + (i % 5) as i64)
            .add_vector(VECTOR_FIELD, vec)
            .build();
        engine.add_document(&i.to_string(), doc).await?;
    }
    engine.commit().await?;
    Ok(engine)
}

#[allow(dead_code)]
async fn build_hybrid_engine(n: usize) -> Result<Engine> {
    build_hybrid_engine_into_storage(memory_storage()?, n).await
}

/// Bump when anything about the persisted on-disk index would change:
/// schema layout (`hybrid_schema`), analyzer defaults, doc-body /
/// vector synthesis, or laurus's segment format. Caches written under
/// a stale version are auto-rebuilt by [`cached_hybrid_engine`].
const BENCH_INDEX_FORMAT_VERSION: &str = "1";

/// Cache root, mirroring the lexical bench's layout
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

fn cache_dir_for(n: usize) -> PathBuf {
    cache_root().join(format!("hybrid_{n}_dim{DIM}_v{BENCH_INDEX_FORMAT_VERSION}"))
}

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

async fn open_persistent_hybrid_engine(dir: &Path) -> Result<Engine> {
    let config = FileStorageConfig::new(dir);
    let storage = StorageFactory::create(StorageConfig::File(config))?;
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::default());
    Engine::builder(storage, hybrid_schema())
        .analyzer(analyzer)
        .build()
        .await
}

/// Return an engine for the hybrid (lexical + vector) bench corpus
/// shape at size `n`, building it on disk the first time and re-opening
/// it on subsequent runs (#513 Stage 1). Mirrors the lexical bench's
/// `cached_engine` (#510 / #512), specialised for the hybrid schema:
/// cache key includes `DIM` so a follow-up bench at a different
/// embedding size lands in its own slot.
///
/// Invalidation works the same way as in `lexical_search_bench`:
/// bump [`BENCH_INDEX_FORMAT_VERSION`] in source, or set
/// `LAURUS_BENCH_REBUILD=1` for a one-off force-rebuild.
fn cached_hybrid_engine(rt: &Runtime, n: usize) -> Arc<Engine> {
    let dir = cache_dir_for(n);
    let force_rebuild = std::env::var("LAURUS_BENCH_REBUILD").is_ok();

    if !force_rebuild && cache_is_valid(&dir) {
        match rt.block_on(open_persistent_hybrid_engine(&dir)) {
            Ok(engine) => return Arc::new(engine),
            Err(err) => {
                eprintln!(
                    "hybrid bench cache open failed at {} ({err}); rebuilding",
                    dir.display()
                );
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create hybrid bench cache dir");

    let config = FileStorageConfig::new(&dir);
    let storage = StorageFactory::create(StorageConfig::File(config))
        .expect("create file storage for hybrid bench cache");
    let engine = rt
        .block_on(build_hybrid_engine_into_storage(storage, n))
        .expect("hybrid bench cache build failed");

    std::fs::write(dir.join(".bench_version"), BENCH_INDEX_FORMAT_VERSION)
        .expect("write hybrid .bench_version marker");

    Arc::new(engine)
}

fn corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![5_000usize];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
}

/// Build a deterministic dim-128 query vector. Uses a separate seed
/// offset so the query is not byte-identical to a corpus member.
fn build_query_vector() -> Vector {
    let mut state = DEFAULT_SEED.wrapping_add(1);
    Vector::new(lcg_vec_unit(&mut state, DIM))
}

/// Build a `SearchRequest` carrying both a lexical term query and a
/// vector query, with the given fusion algorithm and top-K.
fn build_hybrid_request(fusion: FusionAlgorithm, limit: usize) -> SearchRequest {
    let lexical = LexicalSearchQuery::Obj(Box::new(TermQuery::new("body", "search")));
    let vector = VectorSearchQuery::Vectors(vec![QueryVector {
        vector: build_query_vector(),
        weight: 1.0,
        fields: Some(vec![VECTOR_FIELD.to_string()]),
    }]);

    SearchRequestBuilder::new()
        .lexical_query(lexical)
        .vector_query(vector)
        .fusion_algorithm(fusion)
        .limit(limit)
        .build()
}

fn bench_hybrid_rrf(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("hybrid/rrf");
    group.sample_size(SAMPLE_SIZE_FAST);

    for &n in &corpus_sizes() {
        let engine = cached_hybrid_engine(&rt, n);

        // Sanity: hybrid search must produce non-empty fused results.
        let probe = rt.block_on(async {
            let req = build_hybrid_request(FusionAlgorithm::RRF { k: 60.0 }, 10);
            engine.search(req).await.unwrap().results
        });
        assert!(
            !probe.is_empty(),
            "rrf hybrid probe must return at least one fused hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let req = build_hybrid_request(FusionAlgorithm::RRF { k: 60.0 }, 10);
                    black_box(engine.search(req).await.unwrap().results)
                }
            });
        });
    }

    group.finish();
}

fn bench_hybrid_weighted_sum(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("hybrid/weighted_sum");
    group.sample_size(SAMPLE_SIZE_FAST);

    for &n in &corpus_sizes() {
        let engine = cached_hybrid_engine(&rt, n);

        let probe = rt.block_on(async {
            let req = build_hybrid_request(
                FusionAlgorithm::WeightedSum {
                    lexical_weight: 0.5,
                    vector_weight: 0.5,
                },
                10,
            );
            engine.search(req).await.unwrap().results
        });
        assert!(
            !probe.is_empty(),
            "weighted_sum hybrid probe must return at least one fused hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let req = build_hybrid_request(
                        FusionAlgorithm::WeightedSum {
                            lexical_weight: 0.5,
                            vector_weight: 0.5,
                        },
                        10,
                    );
                    black_box(engine.search(req).await.unwrap().results)
                }
            });
        });
    }

    group.finish();
}

fn bench_hybrid_top_k_sweep(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("hybrid/top_k");
    group.sample_size(SAMPLE_SIZE_FAST);

    let n = 5_000usize;
    let engine = cached_hybrid_engine(&rt, n);

    let probe = rt.block_on(async {
        let req = build_hybrid_request(FusionAlgorithm::RRF { k: 60.0 }, 10);
        engine.search(req).await.unwrap().results
    });
    assert!(
        !probe.is_empty(),
        "top_k hybrid probe must return at least one fused hit"
    );

    for &k in &[10usize, 100, 500] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("k_{k}")),
            &k,
            |b, &k| {
                b.to_async(&rt).iter(|| {
                    let engine = &engine;
                    async move {
                        let req = build_hybrid_request(FusionAlgorithm::RRF { k: 60.0 }, k);
                        black_box(engine.search(req).await.unwrap().results)
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hybrid_rrf,
    bench_hybrid_weighted_sum,
    bench_hybrid_top_k_sweep,
);
criterion_main!(benches);
