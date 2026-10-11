//! Lexical indexing throughput benchmarks.
//!
//! Targets the audit gap that no bench previously measured the cost of
//! `Engine::add_document` or `Engine::commit`. `lexical_search_bench`
//! always called `build_engine(n)` inside its setup block, leaving the
//! ingest path invisible.
//!
//! # Scope
//!
//! Four measurement scenarios:
//!
//! 1. **`bench_add_documents`** — pure `add_document` × N (no `commit`).
//!    Reports per-doc latency for the in-memory buffering path. Sweep
//!    `N ∈ {1k, 10k}` (LARGE adds 100k).
//! 2. **`bench_commit`** — pre-buffer N docs, then time `commit()`. Shows
//!    the segment-flush cost without the `add_document` overhead.
//! 3. **`bench_bulk_ingest`** — `add × N + commit`. The user-visible
//!    "how long to ingest N docs?" number.
//! 4. **`bench_multi_segment_commit`** — `(add × K + commit) × M` so M
//!    segments accumulate. M ∈ {2, 4, 8} at K = 1 000. Exposes the
//!    cost of repeated commits. Auto-merge only runs once the segment
//!    count exceeds `max_segments` (default 100), so no M here triggers
//!    it.
//!
//! Each scenario uses `iter_batched` with `BatchSize::SmallInput` so
//! every timed iteration receives a fresh `Engine`. The Engine
//! construction sits in `iter_batched` setup, outside the timing window.
//!
//! # Vocabulary
//!
//! Mirrors the 3-tier Zipf vocabulary in `lexical_search_bench.rs`. Kept
//! local to this file for now; consolidate into `common.rs` if either
//! file's vocabulary needs updating.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench lexical_indexing_bench                       # default sizes
//! LAURUS_BENCH_LARGE=1 cargo bench --bench lexical_indexing_bench  # adds 100k
//! ```
//!
//! Filter by case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench lexical_indexing_bench -- "bulk_ingest"
//! cargo bench --bench lexical_indexing_bench -- "multi_segment/m_4"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench lexical_indexing_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokio::runtime::Runtime;

use common::{SAMPLE_SIZE_SLOW, select_storage};

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::core::field::IntegerOption;
use laurus::lexical::{TermQuery, TextOption};
use laurus::storage::Storage;
use laurus::{Document, Engine, LexicalSearchQuery, Result, Schema, SearchRequestBuilder};

// ----------------------------------------------------------------------------
// Vocabulary (mirrors lexical_search_bench.rs — keep in sync if either
// file's vocab changes).
// ----------------------------------------------------------------------------

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

/// Build `n` `(id, Document)` pairs deterministically.
fn build_documents(n: usize) -> Vec<(String, Document)> {
    (0..n)
        .map(|i| {
            let body = build_body(i);
            let doc = Document::builder()
                .add_text("title", format!("Title for document {i}"))
                .add_text("body", &body)
                .add_text("category", CATEGORIES[i % CATEGORIES.len()])
                .add_integer("year", 2020 + (i % 5) as i64)
                .build();
            (i.to_string(), doc)
        })
        .collect()
}

// ----------------------------------------------------------------------------
// Engine helpers
// ----------------------------------------------------------------------------

/// Bench-storage handle. Delegates to `common::select_storage()` so
/// `LAURUS_BENCH_DISK=1` swaps the in-memory backend for a temp-dir
/// `FileStorage` without changing call sites.
fn memory_storage() -> Result<Arc<dyn Storage>> {
    Ok(select_storage())
}

async fn build_empty_engine() -> Result<Engine> {
    let storage = memory_storage()?;
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::default());

    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .add_text_field("category", TextOption::default())
        .add_integer_field("year", IntegerOption::default())
        .add_default_field("body")
        .build();

    Engine::builder(storage, schema)
        .analyzer(analyzer)
        .build()
        .await
}

fn ingest_corpus_sizes() -> Vec<usize> {
    let mut sizes = vec![1000usize, 10_000];
    if std::env::var("LAURUS_BENCH_LARGE").is_ok() {
        sizes.push(100_000);
    }
    sizes
}

// ----------------------------------------------------------------------------
// Benches
// ----------------------------------------------------------------------------

/// `add_document` × N, without `commit`. Measures the in-memory buffering
/// path only.
fn bench_add_documents(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("ingest/add_documents");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &ingest_corpus_sizes() {
        // One-time sanity check: ingest once and confirm the engine
        // contains the buffered docs (not committed, but `add_document`
        // returned Ok). We do this by re-running setup + a single
        // add_document and asserting it does not error.
        {
            let engine = rt.block_on(build_empty_engine()).unwrap();
            let (id, doc) = build_documents(1).into_iter().next().unwrap();
            rt.block_on(engine.add_document(&id, doc)).unwrap();
        }

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let engine = rt.block_on(build_empty_engine()).unwrap();
                    let docs = build_documents(n);
                    (engine, docs)
                },
                |(engine, docs)| {
                    rt.block_on(async {
                        for (id, doc) in docs {
                            engine.add_document(&id, doc).await.unwrap();
                        }
                    });
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// `commit` latency after N docs are pre-buffered.
fn bench_commit(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("ingest/commit");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &[1000usize, 10_000] {
        // Sanity: a fresh engine with N adds + commit makes the docs
        // searchable.
        {
            let engine = rt.block_on(build_empty_engine()).unwrap();
            let docs = build_documents(n);
            rt.block_on(async {
                for (id, doc) in docs {
                    engine.add_document(&id, doc).await.unwrap();
                }
                engine.commit().await.unwrap();
                let request = SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                        "body", "search",
                    ))))
                    .limit(10)
                    .build();
                let hits = engine.search(request).await.unwrap();
                assert!(
                    !hits.is_empty(),
                    "commit probe must produce searchable docs at n={n}"
                );
            });
        }

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let engine = rt.block_on(build_empty_engine()).unwrap();
                    let docs = build_documents(n);
                    rt.block_on(async {
                        for (id, doc) in docs {
                            engine.add_document(&id, doc).await.unwrap();
                        }
                    });
                    engine
                },
                |engine| {
                    rt.block_on(engine.commit()).unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// End-to-end `add × N + commit`. The user-visible "ingest N docs" cost.
fn bench_bulk_ingest(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("ingest/bulk");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &ingest_corpus_sizes() {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let engine = rt.block_on(build_empty_engine()).unwrap();
                    let docs = build_documents(n);
                    (engine, docs)
                },
                |(engine, docs)| {
                    rt.block_on(async {
                        for (id, doc) in docs {
                            engine.add_document(&id, doc).await.unwrap();
                        }
                        engine.commit().await.unwrap();
                    });
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// End-to-end `add_documents(N) + commit` — the batch-API twin of
/// `ingest/bulk` (#551). One WAL fsync per batch instead of one per record,
/// so the delta vs `ingest/bulk` is only visible on the FileStorage backend
/// (`LAURUS_BENCH_DISK=1`); on MemoryStorage the two should be equal.
fn bench_bulk_ingest_batch_api(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("ingest/bulk_batch_api");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in &ingest_corpus_sizes() {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let engine = rt.block_on(build_empty_engine()).unwrap();
                    let docs = build_documents(n);
                    (engine, docs)
                },
                |(engine, docs)| {
                    rt.block_on(async {
                        engine.add_documents(docs).await.unwrap();
                        engine.commit().await.unwrap();
                    });
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// `(add × K + commit) × M` so M segments accumulate. Every M stays under
/// the default `max_segments` (100), so auto-merge never runs and the
/// M-sweep measures repeated commits alone.
fn bench_multi_segment_commit(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("ingest/multi_segment");
    group.sample_size(SAMPLE_SIZE_SLOW);

    let k_per_segment = 1000usize;

    for &m in &[2usize, 4, 8] {
        let total = m * k_per_segment;
        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(BenchmarkId::new("m", m), &m, |b, &m| {
            b.iter_batched(
                || {
                    let engine = rt.block_on(build_empty_engine()).unwrap();
                    let batches: Vec<Vec<(String, Document)>> = (0..m)
                        .map(|seg| {
                            let start = seg * k_per_segment;
                            (start..start + k_per_segment)
                                .map(|i| {
                                    let body = build_body(i);
                                    let doc = Document::builder()
                                        .add_text("title", format!("Title for document {i}"))
                                        .add_text("body", &body)
                                        .add_text("category", CATEGORIES[i % CATEGORIES.len()])
                                        .add_integer("year", 2020 + (i % 5) as i64)
                                        .build();
                                    (i.to_string(), doc)
                                })
                                .collect()
                        })
                        .collect();
                    (engine, batches)
                },
                |(engine, batches)| {
                    rt.block_on(async {
                        for batch in batches {
                            for (id, doc) in batch {
                                engine.add_document(&id, doc).await.unwrap();
                            }
                            engine.commit().await.unwrap();
                        }
                    });
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Thread counts swept by `ingest/concurrent`.
fn concurrency_levels() -> Vec<usize> {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4);
    // Include one level past the core count: the interesting question is
    // whether added threads help, plateau, or actively hurt.
    [1usize, 2, 4, 8, 16]
        .into_iter()
        .filter(|&t| t <= (cores * 2).max(2))
        .collect()
}

/// `add_document` × N spread over T threads sharing one `Engine` (#546).
///
/// This is the ceiling probe for parallel ingestion. Today the lexical
/// writer sits behind a single mutex in `LexicalStore` that is held across
/// tokenization, so the expected shape is flat-or-worse as T rises — the
/// baseline every later phase is measured against.
///
/// `iter_custom` rather than `iter`: the work has to be timed around a
/// `thread::scope`, and the per-iteration engine setup must stay outside
/// the measured span.
fn bench_concurrent_ingest(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("ingest/concurrent");
    group.sample_size(SAMPLE_SIZE_SLOW);

    // One corpus size only: the sweep of interest is T, not N, and the
    // cross product would make the bench unusably slow.
    const DOCS: usize = 2_000;

    for t in concurrency_levels() {
        group.throughput(Throughput::Elements(DOCS as u64));
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |b, &t| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let engine = Arc::new(rt.block_on(build_empty_engine()).unwrap());
                    let docs = build_documents(DOCS);
                    // Contiguous slices per thread: every document is
                    // ingested exactly once, by exactly one thread.
                    let per_thread = DOCS.div_ceil(t);
                    let chunks: Vec<Vec<(String, Document)>> =
                        docs.chunks(per_thread).map(<[_]>::to_vec).collect();

                    let start = std::time::Instant::now();
                    std::thread::scope(|scope| {
                        for chunk in chunks {
                            let engine = Arc::clone(&engine);
                            scope.spawn(move || {
                                // Each thread drives its own current-thread
                                // runtime, so the measurement reflects the
                                // engine's own serialization rather than a
                                // shared executor's scheduling.
                                let rt = tokio::runtime::Builder::new_current_thread()
                                    .enable_all()
                                    .build()
                                    .unwrap();
                                rt.block_on(async {
                                    for (id, doc) in chunk {
                                        engine.add_document(&id, doc).await.unwrap();
                                    }
                                });
                            });
                        }
                    });
                    rt.block_on(engine.commit()).unwrap();
                    total += start.elapsed();
                }
                total
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_add_documents,
    bench_commit,
    bench_bulk_ingest,
    bench_bulk_ingest_batch_api,
    bench_multi_segment_commit,
    bench_concurrent_ingest,
);
criterion_main!(benches);
