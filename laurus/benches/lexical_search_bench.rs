//! End-to-end lexical search benchmarks.
//!
//! Measures full query execution time (matching + scoring + collection)
//! for `TermQuery`, `BooleanQuery`, `PhraseQuery`, `FuzzyQuery`, and the
//! DSL parser. See [`laurus/benches/BENCHMARKS.md`](../BENCHMARKS.md) for
//! the broader architecture rationale (three-phase model, on-disk index
//! cache, why synthetic data instead of TREC/Wikipedia).
//!
//! # Three-phase model (#510)
//!
//! Each bench logically runs in three phases:
//!
//! 1. **Corpus generation** — synthetic, pure-functional ([`build_body`] /
//!    [`build_body_skewed`]). Cost ~1 sec for 100 k docs.
//! 2. **Index construction** — `engine.add_document` + `commit`. Cost
//!    ~17 minutes for 100 k uniform. **This is what the on-disk cache
//!    eliminates on second-and-later runs.**
//! 3. **Search measurement** — Criterion `b.iter` measuring
//!    `engine.search`. The only phase whose latency this binary cares
//!    about.
//!
//! For the indexing-side bench (`lexical_indexing_bench`), Phase 2 *is*
//! the measurement target and the cache is **not** used.
//!
//! # On-disk index cache
//!
//! [`cached_engine`] persists Phase 2's output under
//! `target/laurus_bench_index_cache/<shape>_<n>_segs<k>_v<N>/`. On a
//! fresh checkout the first run pays the build cost once; every later
//! `cargo bench` invocation reopens the cached index in <1 second. The
//! cache is invalidated by:
//!
//! - Bumping [`BENCH_INDEX_FORMAT_VERSION`] when schema, analyzer, or
//!   laurus's segment format change.
//! - `LAURUS_BENCH_REBUILD=1` to force a wipe-and-rebuild.
//! - `cargo clean` to drop the whole `target/` tree.
//!
//! # Corpus shape
//!
//! Document bodies follow a 3-tier Zipf-like distribution (a few very
//! common terms, several medium-frequency topic phrases, many rare
//! long-tail words):
//!
//! - [`COMMON_TERMS`] (5 words: `search`, `system`, `data`, `engine`,
//!   `query`): present in **every** document. High document-frequency,
//!   low IDF — the WAND-friendly terms that #403 (top-K
//!   early-termination) targets.
//! - [`TOPIC_PHRASES`] (8 phrases × 8 words): each phrase appears in
//!   `1/8` of documents (≈ 12.5 %).
//! - [`LONG_TAIL`] (~80 domain words): each document picks 5 words from
//!   this pool by stride; each individual word appears in roughly 5 % of
//!   documents.
//!
//! `bench_phrase_query` queries the phrase `"search engine"`; both words
//! come from `COMMON_TERMS`, so the phrase is guaranteed to occur in
//! every document.
//!
//! Why synthetic instead of TREC / Wikipedia: see
//! [`BENCHMARKS.md`](../BENCHMARKS.md) for the trade-off analysis. The
//! short version is that perf-PR comparisons (the bench's actual
//! workflow) only need ratio measurements, and synthetic data is
//! reproducible, deterministic, CI-friendly, and ships zero external
//! dependencies.
//!
//! # Size gates
//!
//! - [`corpus_sizes`] — uniform sweep used by `term_query`,
//!   `term_query_varying_limit`, `boolean_query`. Default
//!   `{100, 1k, 5k}`; `LAURUS_BENCH_LARGE=1` adds `100k`.
//! - [`skewed_corpus_sizes`] — used by `topk_or_skewed_tf`,
//!   `topk_or_multi_segment`. Default `{1k, 10k}`;
//!   `LAURUS_BENCH_LARGE=1` adds `100k`.
//! - [`seek_skewed_sizes`] — used by `seek_skewed`. Default `{10k}`;
//!   `LAURUS_BENCH_LARGE=1` adds `100k`.
//!
//! The default sweep finishes in single-digit minutes; the
//! `LAURUS_BENCH_LARGE` sweep is the acceptance-gate target and takes
//! tens of minutes on the **first** run only — subsequent runs reopen
//! the cache.
//!
//! # Run
//!
//! ```sh
//! # Daily iteration (fast — uses cache after first run):
//! cargo bench --bench lexical_search_bench
//!
//! # Acceptance gate (adds 100k cases):
//! LAURUS_BENCH_LARGE=1 cargo bench --bench lexical_search_bench
//!
//! # Force a fresh cache build (e.g. after pulling a major change):
//! LAURUS_BENCH_REBUILD=1 cargo bench --bench lexical_search_bench
//! ```
//!
//! Filter by group / case (regex match against the criterion id):
//!
//! ```sh
//! cargo bench --bench lexical_search_bench -- term_query
//! cargo bench --bench lexical_search_bench -- 'boolean_query/should_or_high_freq'
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench lexical_search_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules and
//! `benches/BENCHMARKS.md` for the bench architecture rationale.

mod common;

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use tokio::runtime::Runtime;

use common::{SAMPLE_SIZE_FAST, SAMPLE_SIZE_SLOW};

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::core::field::IntegerOption;
use laurus::lexical::{
    BooleanQuery, FuzzyQuery, PhraseQuery, SynonymQuery, TermQuery, TextOption, WildcardQuery,
};
use laurus::storage::Storage;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{Document, Engine, LexicalSearchQuery, Result, Schema, SearchRequestBuilder};

/// Words present in **every** document. Drives the high-document-frequency
/// region of the term distribution and gives WAND / MaxScore early-
/// termination (#403) something to prune against. The pair `search engine`
/// always appears in this exact order so `bench_phrase_query` has a
/// guaranteed match.
const COMMON_TERMS: &str = "search engine system data query";

/// Topic phrases — each appears in `1/8` of documents (≈ 12.5 %).
/// Mid-frequency tier. Each phrase is internally diverse so individual
/// words inside it get further-spread frequency.
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

/// Long-tail vocabulary. Each document picks `LONG_TAIL_PER_DOC` words
/// from this pool by stride, so any individual word appears in roughly
/// `LONG_TAIL_PER_DOC / LONG_TAIL.len()` ≈ 6 % of documents. Provides the
/// rare-term tail of the Zipf distribution.
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
    "hashicorp",
    "kubernetes",
    "docker",
    "mesos",
    "borg",
    "raft",
    "paxos",
    "tantivy",
    "lucene",
    "solr",
    "elasticsearch",
    "qdrant",
    "weaviate",
    "milvus",
    "embedding",
    "transformer",
    "attention",
    "softmax",
    "gradient",
    "backprop",
    "tensor",
    "matrix",
    "vector",
    "scalar",
    "tokenizer",
    "lemmatizer",
    "stemmer",
    "synonym",
    "antonym",
    "wordnet",
    "thesaurus",
    "ontology",
    "taxonomy",
    "shingle",
    "ngram",
    "unigram",
    "bigram",
    "trigram",
    "skipgram",
    "rocksdb",
    "leveldb",
    "bigtable",
    "hbase",
    "cassandra",
    "scylla",
    "kafka",
    "pulsar",
    "rabbitmq",
    "zeromq",
    "nats",
    "redis",
    "prometheus",
    "grafana",
    "jaeger",
    "zipkin",
    "opentelemetry",
    "wasm",
    "webassembly",
    "wasi",
    "nodejs",
    "deno",
    "bun",
    "criterion",
    "tarpaulin",
    "miri",
    "loom",
    "fuzzer",
    "afl",
];

/// Number of long-tail words appended to each document body.
const LONG_TAIL_PER_DOC: usize = 5;

const CATEGORIES: &[&str] = &["programming", "data-science", "web", "database", "systems"];

/// Build a deterministic document body following the 3-tier distribution
/// described at the top of this file.
fn build_body(i: usize) -> String {
    let topic = TOPIC_PHRASES[i % TOPIC_PHRASES.len()];

    // Pick LONG_TAIL_PER_DOC distinct long-tail words by stride. Using a
    // co-prime stride (7) and starting offset (i) means consecutive
    // documents share few long-tail words but the distribution stays
    // even-ish across the whole corpus.
    let mut tail_words = Vec::with_capacity(LONG_TAIL_PER_DOC);
    for k in 0..LONG_TAIL_PER_DOC {
        let idx = (i.wrapping_mul(7) + k * 11) % LONG_TAIL.len();
        tail_words.push(LONG_TAIL[idx]);
    }
    let tail = tail_words.join(" ");

    format!("Document {i} {COMMON_TERMS} {topic} {tail} should match relevant terms")
}

/// Build a skewed-TF document body (#403 PR-C fixture).
///
/// 90 % of documents look like a normal `build_body` result (`COMMON_TERMS`
/// once, plus topic / tail). The remaining 10 % — every 10th document —
/// repeat `COMMON_TERMS` an extra **15 times**, mimicking long-form
/// articles or product descriptions that mention common search terms
/// repeatedly. Concretely:
///
/// - For `search`, `system`, `data`, etc., 90 % of postings have
///   `tf = 1` (or `2` if the topic phrase contains the term).
/// - The remaining 10 % have `tf ≈ 16-17`.
///
/// The resulting per-block max-impact varies sharply: blocks whose
/// every doc has `tf = 1` have `block_max_factor ≈ 1.0`, while a
/// block containing one heavy-hitter has `block_max_factor ≈ 2.1`.
/// This is exactly the distribution Block-Max-WAND (#403 PR-C) is
/// designed to skip — the existing uniform `build_body` corpus does
/// not exercise that algorithm.
fn build_body_skewed(i: usize) -> String {
    let topic = TOPIC_PHRASES[i % TOPIC_PHRASES.len()];

    let mut tail_words = Vec::with_capacity(LONG_TAIL_PER_DOC);
    for k in 0..LONG_TAIL_PER_DOC {
        let idx = (i.wrapping_mul(7) + k * 11) % LONG_TAIL.len();
        tail_words.push(LONG_TAIL[idx]);
    }
    let tail = tail_words.join(" ");

    if i.is_multiple_of(10) {
        // Heavy hitter: repeat COMMON_TERMS 16x. The repetitions ride
        // the same byte-deterministic path as the normal builder so
        // two runs produce byte-identical input.
        let common_repeated = std::iter::repeat_n(COMMON_TERMS, 16)
            .collect::<Vec<_>>()
            .join(" ");
        format!("Document {i} {common_repeated} {topic} {tail} should match relevant terms")
    } else {
        format!("Document {i} {COMMON_TERMS} {topic} {tail} should match relevant terms")
    }
}

/// Core engine construction body. Same boilerplate as the legacy
/// `build_engine_with_segments`, parameterised on `storage` so the
/// on-disk cache helper ([`cached_engine`]) can pass a
/// [`FileStorageConfig`]-backed storage that points at a deterministic
/// directory under `target/`. The new architecture treats this function
/// as the canonical Phase 2 (index construction) primitive.
async fn build_engine_into_storage(
    storage: Arc<dyn Storage>,
    n: usize,
    body_fn: fn(usize) -> String,
    segment_count: usize,
) -> Result<Engine> {
    assert!(segment_count >= 1, "segment_count must be ≥ 1");
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::default());

    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .add_text_field("category", TextOption::default())
        .add_integer_field("year", IntegerOption::default())
        .add_integer_field("seq", IntegerOption::default())
        .add_default_field("body")
        .build();

    let engine = Engine::builder(storage, schema)
        .analyzer(analyzer)
        .build()
        .await?;

    let chunk_size = n.div_ceil(segment_count);
    let mut next_commit = chunk_size;
    for i in 0..n {
        let body = body_fn(i);
        let doc = Document::builder()
            .add_text("title", format!("Title for document {i}"))
            .add_text("body", &body)
            .add_text("category", CATEGORIES[i % CATEGORIES.len()])
            .add_integer("year", 2020 + (i % 5) as i64)
            // Monotonic with insertion order, so segment-per-commit
            // leaves each segment a disjoint range — the time-series
            // shape #944 Phase B's segment pruning targets. `year`
            // cycles every 5 docs and therefore overlaps everywhere.
            .add_integer("seq", i as i64)
            .build();

        engine.add_document(&i.to_string(), doc).await?;

        // Per-chunk commit so each chunk lands as its own segment
        // (auto-merge runs on a 60s interval and does not fire during
        // back-to-back commits in this build loop).
        if i + 1 == next_commit && i + 1 < n {
            engine.commit().await?;
            next_commit += chunk_size;
        }
    }

    engine.commit().await?;
    Ok(engine)
}

/// Body-content shape tag used as the cache key for [`cached_engine`].
/// `Uniform` documents come from [`build_body`] and feed the standard
/// TermQuery / BooleanQuery sweeps; `Skewed` documents come from
/// [`build_body_skewed`] and feed the BMW / skip-list / multi-segment
/// scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EngineShape {
    Uniform,
    Skewed,
}

impl EngineShape {
    fn body_fn(self) -> fn(usize) -> String {
        match self {
            EngineShape::Uniform => build_body,
            EngineShape::Skewed => build_body_skewed,
        }
    }
}

/// Bump this when anything that would alter the resulting index on
/// disk changes — schema layout, `StandardAnalyzer` defaults, the
/// `build_body` / `build_body_skewed` synthesis, or laurus's segment
/// format. Caches written under a stale version are rebuilt
/// automatically (the helper compares against `.bench_version`).
const BENCH_INDEX_FORMAT_VERSION: &str = "2";

/// Process-and-cargo-wide cache directory for pre-built indexes.
/// `target/laurus_bench_index_cache/<shape>_<n>_segs<k>_v<N>/` holds a
/// `FileStorage` tree plus a `.bench_version` file. Sits under
/// `target/` so `cargo clean` evicts the cache; `LAURUS_BENCH_REBUILD=1`
/// forces a wipe-and-rebuild without touching the rest of `target/`.
///
/// Workspace root is derived at compile time via `env!`; we use the
/// `laurus` crate's manifest dir and step up one level. `CARGO_TARGET_DIR`
/// (if set) overrides at runtime so custom target locations work too.
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

/// Path of the persisted index for `(shape, n, segment_count)`.
fn cache_dir_for(shape: EngineShape, n: usize, segment_count: usize) -> PathBuf {
    cache_root().join(format!(
        "{}_{n}_segs{segment_count}_v{BENCH_INDEX_FORMAT_VERSION}",
        match shape {
            EngineShape::Uniform => "uniform",
            EngineShape::Skewed => "skewed",
        }
    ))
}

/// Verify a cache entry: directory exists and `.bench_version` matches.
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

/// Open an existing on-disk index. The engine's [`Engine::build`]
/// internally runs `recover()` which replays the WAL + reattaches
/// segments persisted by an earlier bench run, so the returned engine
/// is immediately query-ready.
async fn open_persistent_engine(dir: &Path) -> Result<Engine> {
    let config = FileStorageConfig::new(dir);
    let storage = StorageFactory::create(StorageConfig::File(config))?;
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::default());
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .add_text_field("category", TextOption::default())
        .add_integer_field("year", IntegerOption::default())
        .add_integer_field("seq", IntegerOption::default())
        .add_default_field("body")
        .build();
    Engine::builder(storage, schema)
        .analyzer(analyzer)
        .build()
        .await
}

/// Return an engine for `(shape, n, segment_count)`, building it on
/// disk the first time and re-opening it on subsequent runs (#510).
///
/// On a fresh checkout this pays the legacy build cost once
/// (~minutes for 100 k uniform, ~tens of seconds for 10 k). Every
/// later `cargo bench` invocation opens the cached index in well under
/// a second, so iterative perf work — the workflow this helper exists
/// for — moves from "tens of minutes per cycle" to "Criterion warmup +
/// measurement only".
///
/// Cache invalidation is driven by [`BENCH_INDEX_FORMAT_VERSION`]; bump
/// it whenever any of the inputs that would alter the resulting index
/// change. `LAURUS_BENCH_REBUILD=1` forces a wipe-and-rebuild without
/// touching the rest of `target/`.
fn cached_engine(rt: &Runtime, shape: EngineShape, n: usize, segment_count: usize) -> Arc<Engine> {
    assert!(segment_count >= 1, "segment_count must be ≥ 1");
    let dir = cache_dir_for(shape, n, segment_count);

    let force_rebuild = std::env::var("LAURUS_BENCH_REBUILD").is_ok();

    if !force_rebuild && cache_is_valid(&dir) {
        // Try to open; if it fails (laurus internal format drift not
        // caught by the version key, partial cache, etc.) fall through
        // to rebuild rather than abort the bench run.
        match rt.block_on(open_persistent_engine(&dir)) {
            Ok(engine) => return Arc::new(engine),
            Err(err) => {
                eprintln!(
                    "bench cache open failed at {} ({err}); rebuilding",
                    dir.display()
                );
            }
        }
    }

    // Wipe + recreate so a partially-written previous run does not
    // contaminate the new build. `remove_dir_all` on a missing path is
    // an error we ignore — the create below handles both cases.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create bench cache dir");

    let config = FileStorageConfig::new(&dir);
    let storage =
        StorageFactory::create(StorageConfig::File(config)).expect("create file storage for cache");
    let engine = rt
        .block_on(build_engine_into_storage(
            storage,
            n,
            shape.body_fn(),
            segment_count,
        ))
        .expect("build_engine_into_storage failed");

    std::fs::write(dir.join(".bench_version"), BENCH_INDEX_FORMAT_VERSION)
        .expect("write .bench_version marker");

    Arc::new(engine)
}

/// Default corpus sizes for sweep cases. Setting `LAURUS_BENCH_LARGE=1`
/// in the environment appends a 100 000-document case so #403 (WAND /
/// MaxScore early-termination) has a measurable target without ballooning
/// the default `cargo bench` runtime.
fn corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![100usize, 1_000, 5_000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
}

/// Sizes for the always-scaled skewed-TF benches
/// ([`bench_topk_or_skewed_tf`], [`bench_topk_or_multi_segment`]).
/// Defaults stop at 10 000 so a `cargo bench` invocation does not
/// silently spend ~17 minutes per 100 k corpus build; the 100 k case
/// is the acceptance-gate target, gated on `LAURUS_BENCH_LARGE=1`.
fn skewed_corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![1_000usize, 10_000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
}

/// Sizes for [`bench_seek_skewed`]. The skip-list-targeted workload
/// only produces a meaningful signal once posting lists are long
/// enough that the binary-search hierarchy lights up, so the default
/// runs at 10 000 (~700 docs / posting list at 6 % long-tail
/// frequency) and `LAURUS_BENCH_LARGE=1` opts in to the
/// acceptance-gate 100 k case.
fn seek_skewed_sizes() -> Vec<usize> {
    let mut sizes = vec![10_000usize];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
}

/// Pick `sample_size` for a corpus of `n` documents. Large corpora use the
/// slow tier so wall time stays bounded.
fn sample_size_for(n: usize) -> usize {
    if n >= 50_000 {
        SAMPLE_SIZE_SLOW
    } else {
        SAMPLE_SIZE_FAST
    }
}

/// Apply the sample-size policy to a benchmark group. Picks the slowest
/// (= smallest) sample size required by any element of `sizes` so a single
/// group can host both small and large cases without losing precision on
/// the big end.
fn apply_sample_size(group: &mut BenchmarkGroup<'_, WallTime>, sizes: &[usize]) {
    let sample = sizes
        .iter()
        .copied()
        .map(sample_size_for)
        .min()
        .unwrap_or(SAMPLE_SIZE_FAST);
    group.sample_size(sample);
}

fn bench_term_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/term_query");
    let sizes = corpus_sizes();
    apply_sample_size(&mut group, &sizes);

    for &n in &sizes {
        let engine = cached_engine(&rt, EngineShape::Uniform, n, 1);

        // One-time sanity check: the probe query must return at least one
        // hit. If this fires, the corpus / query pair drifted out of sync.
        let probe = rt.block_on(async {
            let query = Box::new(TermQuery::new("body", "programming"));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "term_query probe must return at least one hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::new("search", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(TermQuery::new("body", "programming"));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });
    }
    group.finish();
}

fn bench_term_query_varying_limit(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    // Pick the largest corpus available (100 k under LAURUS_BENCH_LARGE,
    // 5 000 otherwise) — limit-sweep is most informative when there are
    // far more matching docs than the largest top-K asked for.
    let corpus_n = *corpus_sizes()
        .last()
        .expect("corpus_sizes() must be non-empty");
    let engine = cached_engine(&rt, EngineShape::Uniform, corpus_n, 1);

    // Sanity check: the largest limit must return more hits than the smallest.
    let probe_small = rt.block_on(async {
        let query = Box::new(TermQuery::new("body", "programming"));
        let request = SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Obj(query))
            .limit(10)
            .build();
        engine.search(request).await.unwrap()
    });
    assert!(
        !probe_small.is_empty(),
        "term_query_limit probe must return at least one hit (corpus={corpus_n})"
    );

    let mut group = c.benchmark_group("lexical/term_query_limit");
    apply_sample_size(&mut group, &[corpus_n]);

    for &limit in &[10, 50, 100, 500] {
        group.bench_with_input(
            BenchmarkId::new(format!("corpus_{corpus_n}/top"), limit),
            &limit,
            |b, &limit| {
                b.to_async(&rt).iter(|| {
                    let engine = &engine;
                    async move {
                        let query = Box::new(TermQuery::new("body", "programming"));
                        let request = SearchRequestBuilder::new()
                            .lexical_query(LexicalSearchQuery::Obj(query))
                            .limit(limit)
                            .build();
                        black_box(engine.search(request).await.unwrap())
                    }
                });
            },
        );
    }
    group.finish();
}

/// Field-sorted top-K (#944 Phase A). The corpus's `year` integer field
/// is DocValues-backed, so `sort_by(year)` exercises the
/// `TopFieldCollector` path end-to-end. Parameterised over segment
/// count: the single-segment case measures the scan itself; the
/// multi-segment case measures the per-segment fanout.
fn bench_field_sorted_query(c: &mut Criterion) {
    use laurus::lexical::search::searcher::{SortField, SortOrder};

    let rt = Runtime::new().unwrap();
    let corpus_n = *corpus_sizes()
        .last()
        .expect("corpus_sizes() must be non-empty");

    for &(sort_field, segment_count) in &[("year", 1usize), ("year", 4), ("seq", 4)] {
        let engine = cached_engine(&rt, EngineShape::Uniform, corpus_n, segment_count);

        // Sanity probe: the sorted search must return hits.
        let probe = rt.block_on(async {
            let query = Box::new(TermQuery::new("body", "programming"));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .sort_by(SortField::Field {
                    name: sort_field.into(),
                    order: SortOrder::Desc,
                })
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "field_sort probe must return at least one hit (corpus={corpus_n}, field={sort_field}, segments={segment_count})"
        );

        let mut group = c.benchmark_group(format!(
            "lexical/field_sort/{sort_field}/seg_{segment_count}"
        ));
        apply_sample_size(&mut group, &[corpus_n]);

        for &limit in &[10, 100] {
            group.bench_with_input(
                BenchmarkId::new(format!("corpus_{corpus_n}/top"), limit),
                &limit,
                |b, &limit| {
                    b.to_async(&rt).iter(|| {
                        let engine = &engine;
                        async move {
                            let query = Box::new(TermQuery::new("body", "programming"));
                            let request = SearchRequestBuilder::new()
                                .lexical_query(LexicalSearchQuery::Obj(query))
                                .sort_by(SortField::Field {
                                    name: sort_field.into(),
                                    order: SortOrder::Desc,
                                })
                                .limit(limit)
                                .build();
                            black_box(engine.search(request).await.unwrap())
                        }
                    });
                },
            );
        }
        group.finish();
    }
}

fn bench_boolean_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/boolean_query");
    let sizes = corpus_sizes();
    apply_sample_size(&mut group, &sizes);

    for &n in &sizes {
        let engine = cached_engine(&rt, EngineShape::Uniform, n, 1);

        // Sanity check: the OR probe over three known terms must hit.
        let probe = rt.block_on(async {
            let mut bq = BooleanQuery::new();
            bq.add_should(Box::new(TermQuery::new("body", "rust")));
            bq.add_should(Box::new(TermQuery::new("body", "python")));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "boolean_query OR probe must return at least one hit at n={n}"
        );

        // MUST + MUST (AND)
        group.bench_with_input(BenchmarkId::new("must_and", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let mut bq = BooleanQuery::new();
                    bq.add_must(Box::new(TermQuery::new("body", "programming")));
                    bq.add_must(Box::new(TermQuery::new("body", "language")));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        // SHOULD + SHOULD (OR) — three topic terms (≈ 12.5 % each).
        group.bench_with_input(BenchmarkId::new("should_or", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let mut bq = BooleanQuery::new();
                    bq.add_should(Box::new(TermQuery::new("body", "rust")));
                    bq.add_should(Box::new(TermQuery::new("body", "python")));
                    bq.add_should(Box::new(TermQuery::new("body", "javascript")));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        // SHOULD + SHOULD (OR) — three high-frequency COMMON_TERMS that
        // appear in every document. This is the workload WAND / MaxScore
        // (#403) targets: an OR over high-DF terms where most candidates
        // can be skipped once the top-K min-score is established.
        group.bench_with_input(BenchmarkId::new("should_or_high_freq", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let mut bq = BooleanQuery::new();
                    bq.add_should(Box::new(TermQuery::new("body", "search")));
                    bq.add_should(Box::new(TermQuery::new("body", "system")));
                    bq.add_should(Box::new(TermQuery::new("body", "data")));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        // MUST + MUST_NOT (AND NOT)
        group.bench_with_input(BenchmarkId::new("must_not", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let mut bq = BooleanQuery::new();
                    bq.add_must(Box::new(TermQuery::new("body", "programming")));
                    bq.add_must_not(Box::new(TermQuery::new("body", "python")));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });
    }
    group.finish();
}

/// Top-K OR benchmark over a **skewed-TF** corpus (#403 PR-C fixture).
///
/// The point of this scenario is to give Block-Max-WAND something to
/// actually skip — see [`build_body_skewed`] for the corpus shape.
/// Three high-frequency terms (`search`, `system`, `data`) appear in
/// every document, but 10 % of documents repeat `COMMON_TERMS` 16x,
/// so per-block max-impact varies sharply between blocks.
///
/// Sizes via [`skewed_corpus_sizes`]: default `{1k, 10k}`,
/// `LAURUS_BENCH_LARGE=1` adds `100k` (the acceptance-gate size). BMW
/// skip behaviour is observable at 10k already because the per-block
/// bound table is populated for any corpus that spans multiple 128-doc
/// blocks; the 100k case is reserved for PR acceptance runs.
fn bench_topk_or_skewed_tf(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/topk_or_skewed_tf");
    let sizes = skewed_corpus_sizes();
    // 100k case dominates wall time — pick the slow tier so the rest
    // of the sweep gets the same sample size for direct comparison.
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &sizes {
        let engine = cached_engine(&rt, EngineShape::Skewed, n, 1);

        // Sanity: high-freq OR must hit at least one heavy-hitter.
        let probe = rt.block_on(async {
            let mut bq = BooleanQuery::new();
            bq.add_should(Box::new(TermQuery::new("body", "search")));
            bq.add_should(Box::new(TermQuery::new("body", "system")));
            bq.add_should(Box::new(TermQuery::new("body", "data")));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "skewed-TF OR probe must return at least one hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::new("should_or_topk10", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let mut bq = BooleanQuery::new();
                    bq.add_should(Box::new(TermQuery::new("body", "search")));
                    bq.add_should(Box::new(TermQuery::new("body", "system")));
                    bq.add_should(Box::new(TermQuery::new("body", "data")));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        // The same OR with a `SynonymQuery` in place of `data`: the shape the
        // query parser builds for a position with stacked synonyms. A
        // `SynonymQuery` carries no block-max table (#1283).
        group.bench_with_input(
            BenchmarkId::new("should_or_synonym_topk10", n),
            &n,
            |b, _| {
                b.to_async(&rt).iter(|| {
                    let engine = &engine;
                    async move {
                        let mut bq = BooleanQuery::new();
                        bq.add_should(Box::new(TermQuery::new("body", "search")));
                        bq.add_should(Box::new(TermQuery::new("body", "system")));
                        bq.add_should(Box::new(SynonymQuery::new("body", vec!["data", "query"])));
                        let request = SearchRequestBuilder::new()
                            .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                            .limit(10)
                            .build();
                        black_box(engine.search(request).await.unwrap())
                    }
                });
            },
        );
    }
    group.finish();
}

/// Multi-segment baseline for #476. Same skewed-TF corpus and
/// should-OR query as [`bench_topk_or_skewed_tf`], but the corpus is
/// split across N commits so the index ends up with N segments
/// (subject to background merge). Compares 1 / 4 / 8 segment
/// constructions side-by-side to expose the cross-segment
/// `block_max_score_at` fallback overhead — the gap PR-G's
/// follow-up work would need to close for the audit-target 100k
/// speedup to transfer to multi-segment deployments.
fn bench_topk_or_multi_segment(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/topk_or_multi_segment");
    let sizes = skewed_corpus_sizes();
    let segment_counts: &[usize] = &[1, 4, 8];
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &sizes {
        for &seg_count in segment_counts {
            let engine = cached_engine(&rt, EngineShape::Skewed, n, seg_count);

            // Sanity probe: high-freq OR must hit at least one heavy-hitter.
            let probe = rt.block_on(async {
                let mut bq = BooleanQuery::new();
                bq.add_should(Box::new(TermQuery::new("body", "search")));
                bq.add_should(Box::new(TermQuery::new("body", "system")));
                bq.add_should(Box::new(TermQuery::new("body", "data")));
                let request = SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                    .limit(10)
                    .build();
                engine.search(request).await.unwrap()
            });
            assert!(
                !probe.is_empty(),
                "multi-seg probe must hit ≥1 doc at n={n} seg={seg_count}"
            );

            let id = format!("n={n}/segments={seg_count}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &n, |b, _| {
                b.to_async(&rt).iter(|| {
                    let engine = &engine;
                    async move {
                        let mut bq = BooleanQuery::new();
                        bq.add_should(Box::new(TermQuery::new("body", "search")));
                        bq.add_should(Box::new(TermQuery::new("body", "system")));
                        bq.add_should(Box::new(TermQuery::new("body", "data")));
                        let request = SearchRequestBuilder::new()
                            .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                            .limit(10)
                            .build();
                        black_box(engine.search(request).await.unwrap())
                    }
                });
            });
        }
    }
    group.finish();
}

/// Seek-heavy AND-conjunction benchmark for the multi-level skip list
/// (#503). Drives a `BooleanQuery::Must(common_term, rare_term)`. The
/// rare side has roughly 6 % document frequency (one [`LONG_TAIL`]
/// word), the common side hits every document — so the conjunction
/// matcher's leader is the rare side and the common side pays one
/// `skip_to` call per rare hit, exercising the worst-case posting-list
/// seek pattern that the linear-walk `find_block` path collapsed to
/// O(N) per call.
///
/// Default size: 10 000 (skip-list hierarchy already lights up at this
/// scale, see [`seek_skewed_sizes`]). `LAURUS_BENCH_LARGE=1` opts in to
/// the 100 000-doc acceptance-gate case from the #503 PR. The 1 M case
/// that existed previously is removed: it took ~3-5 hours to build and
/// did not produce information that the 100 k case did not already
/// expose (the binary-search hierarchy adds only one level between
/// 100 k and 1 M).
fn bench_seek_skewed(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/seek_skewed");
    let sizes = seek_skewed_sizes();
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &sizes {
        let engine = cached_engine(&rt, EngineShape::Uniform, n, 1);

        // Sanity probe: the AND must hit at least once at every
        // benched n — otherwise the conjunction would short-circuit
        // and we'd be measuring nothing.
        let probe = rt.block_on(async {
            let mut bq = BooleanQuery::new();
            bq.add_must(Box::new(TermQuery::new("body", "search")));
            bq.add_must(Box::new(TermQuery::new("body", "lattice")));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "seek_skewed AND probe must return at least one hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::new("and_common_rare", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let mut bq = BooleanQuery::new();
                    bq.add_must(Box::new(TermQuery::new("body", "search")));
                    bq.add_must(Box::new(TermQuery::new("body", "lattice")));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(Box::new(bq)))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });
    }
    group.finish();
}

fn bench_phrase_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/phrase_query");
    // Phrase / fuzzy / DSL benches stay on the fixed small-corpus sweep —
    // the audit's scale-up requirement (#422) only targets term and
    // boolean cases. Sample size is set explicitly so we do not rely on
    // criterion's implicit default.
    group.sample_size(SAMPLE_SIZE_FAST);

    for &n in &[100, 1000, 5000] {
        let engine = cached_engine(&rt, EngineShape::Uniform, n, 1);

        // Sanity check: the two-term phrase probe must hit at least once.
        let probe = rt.block_on(async {
            let query = Box::new(PhraseQuery::new(
                "body",
                vec!["search".into(), "engine".into()],
            ));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "phrase_query probe must return at least one hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::new("two_terms", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(PhraseQuery::new(
                        "body",
                        vec!["search".into(), "engine".into()],
                    ));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("three_terms", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(PhraseQuery::new(
                        "body",
                        vec!["practical".into(), "applications".into(), "in".into()],
                    ));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });
    }
    group.finish();
}

fn bench_fuzzy_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/fuzzy_query");
    group.sample_size(SAMPLE_SIZE_FAST);

    for &n in &[100, 1000, 5000] {
        let engine = cached_engine(&rt, EngineShape::Uniform, n, 1);

        // Sanity check: the edit1 probe must match `programming`.
        let probe = rt.block_on(async {
            let query = Box::new(FuzzyQuery::new("body", "programing").max_edits(1));
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "fuzzy_query probe must return at least one hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::new("edit1", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(FuzzyQuery::new("body", "programing").max_edits(1));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("edit2", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(FuzzyQuery::new("body", "progrming").max_edits(2));
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });
    }
    group.finish();
}

/// Wildcard search latency (Issue #613): like [`bench_fuzzy_query`],
/// exercises the multi-term enumeration path — the query is lowered to a
/// Boolean-of-TermQuery via one term-dictionary enumeration per request
/// (previously two: matcher and scorer re-enumerated independently).
fn bench_wildcard_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("lexical/wildcard_query");
    group.sample_size(SAMPLE_SIZE_FAST);

    for &n in &[100, 1000, 5000] {
        let engine = cached_engine(&rt, EngineShape::Uniform, n, 1);

        // Sanity check: the probe must match `programming`-family terms.
        let probe = rt.block_on(async {
            let query = Box::new(WildcardQuery::new("body", "program*").unwrap());
            let request = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .limit(10)
                .build();
            engine.search(request).await.unwrap()
        });
        assert!(
            !probe.is_empty(),
            "wildcard_query probe must return at least one hit at n={n}"
        );

        group.bench_with_input(BenchmarkId::new("prefix_star", n), &n, |b, _| {
            b.to_async(&rt).iter(|| {
                let engine = &engine;
                async move {
                    let query = Box::new(WildcardQuery::new("body", "program*").unwrap());
                    let request = SearchRequestBuilder::new()
                        .lexical_query(LexicalSearchQuery::Obj(query))
                        .limit(10)
                        .build();
                    black_box(engine.search(request).await.unwrap())
                }
            });
        });
    }
    group.finish();
}

fn bench_dsl_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = cached_engine(&rt, EngineShape::Uniform, 5000, 1);

    // Sanity check: the simple_term DSL probe must hit.
    let probe = rt.block_on(async {
        let request = SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Dsl("body:programming".to_string()))
            .limit(10)
            .build();
        engine.search(request).await.unwrap()
    });
    assert!(
        !probe.is_empty(),
        "dsl_query probe must return at least one hit"
    );

    let mut group = c.benchmark_group("lexical/dsl_query");
    group.sample_size(SAMPLE_SIZE_FAST);

    group.bench_function("simple_term", |b| {
        b.to_async(&rt).iter(|| {
            let engine = &engine;
            async move {
                let request = SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::Dsl("body:programming".to_string()))
                    .limit(10)
                    .build();
                black_box(engine.search(request).await.unwrap())
            }
        });
    });

    group.bench_function("boolean_and", |b| {
        b.to_async(&rt).iter(|| {
            let engine = &engine;
            async move {
                let request = SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::Dsl(
                        "body:programming AND body:language".to_string(),
                    ))
                    .limit(10)
                    .build();
                black_box(engine.search(request).await.unwrap())
            }
        });
    });

    group.bench_function("boolean_or", |b| {
        b.to_async(&rt).iter(|| {
            let engine = &engine;
            async move {
                let request = SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::Dsl(
                        "body:rust OR body:python OR body:javascript".to_string(),
                    ))
                    .limit(10)
                    .build();
                black_box(engine.search(request).await.unwrap())
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_term_query,
    bench_term_query_varying_limit,
    bench_field_sorted_query,
    bench_boolean_query,
    bench_topk_or_skewed_tf,
    bench_topk_or_multi_segment,
    bench_seek_skewed,
    bench_phrase_query,
    bench_fuzzy_query,
    bench_wildcard_query,
    bench_dsl_query,
);
criterion_main!(benches);
