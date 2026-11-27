//! End-to-end throughput benchmark for [`laurus::Engine::search_batch`]
//! (issue [#715](https://github.com/mosuka/laurus/issues/715) +
//! [#716](https://github.com/mosuka/laurus/issues/716), Phase 3 of
//! [#648](https://github.com/mosuka/laurus/issues/648)).
//!
//! Compares two ways to run `B` independent search requests against a
//! shared engine:
//!
//! - **serial loop**: `for req in reqs { engine.search(req).await? }`
//!   — the pattern every caller used before Phase 3.
//! - **batched**: `engine.search_batch(reqs).await?` — the new API
//!   that dispatches the requests in parallel on the tokio runtime via
//!   `futures::future::try_join_all`.
//!
//! Speedup is bounded by the host's tokio worker thread count. On a
//! 4-core / 8-thread laptop CPU expect ~2× at `B = 64` (matching the
//! Phase 1/2 result observed at the `VectorStore` level).
//!
//! The bench uses a lexical-only corpus to keep the per-query cost
//! cheap and the bench duration manageable; the parallelisation it
//! exercises is identical to what hybrid / vector workloads see.
//!
//! Run:
//!
//! ```sh
//! cargo bench --bench engine_search_batch_bench
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

use laurus::Engine;
use laurus::SearchRequestBuilder;
use laurus::lexical::Query;
use laurus::lexical::TermQuery;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::{DataValue, Document};
use laurus::{FieldOption, LexicalSearchQuery, QueryVector, Schema, VectorSearchQuery};

const CORPUS_SIZE: usize = 1_000;
const VECTOR_CORPUS_SIZE: usize = 2_000;
const VECTOR_DIM: usize = 128;
const TERMS: &[&str] = &[
    "rust", "vector", "search", "engine", "index", "query", "field", "data", "system", "lexical",
];

fn build_engine_with_corpus(rt: &Runtime) -> (Arc<Engine>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config).expect("storage");

    let config = Schema::builder()
        .add_field("title", FieldOption::Text(Default::default()))
        .build();

    let engine = rt
        .block_on(async {
            let engine = Engine::new(storage, config).await?;
            for i in 0..CORPUS_SIZE {
                let term = TERMS[i % TERMS.len()];
                let companion = TERMS[(i + 3) % TERMS.len()];
                let doc = Document::builder()
                    .add_field(
                        "title",
                        DataValue::Text(format!("{} {} doc{}", term, companion, i)),
                    )
                    .build();
                engine.put_document(&format!("doc{i}"), doc).await?;
            }
            engine.commit().await?;
            laurus::Result::Ok(engine)
        })
        .expect("engine build");

    (Arc::new(engine), temp_dir)
}

fn build_queries(b: usize) -> Vec<laurus::SearchRequest> {
    (0..b)
        .map(|i| {
            let term = TERMS[i % TERMS.len()];
            let q = Box::new(TermQuery::new("title", term)) as Box<dyn Query>;
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(q))
                .limit(10)
                .build()
        })
        .collect()
}

fn make_vector(seed: u64, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    let hot = (seed as usize) % dim;
    v[hot] = 1.0;
    v[(hot + 1) % dim] = 0.5;
    v[(hot + 7) % dim] = 0.25;
    v
}

fn build_vector_engine_with_corpus(rt: &Runtime) -> (Arc<Engine>, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config).expect("storage");

    let hnsw = HnswOption {
        dimension: VECTOR_DIM,
        distance: DistanceMetric::Cosine,
        m: 16,
        ef_construction: 100,
        default_ef_search: None,
        base_weight: 1.0,
        quantizer: Default::default(),
        rerank_storage: None,
        embedder: None,
        pq_codebook_path: None,
    };
    let config = Schema::builder()
        .add_field("vec", FieldOption::Hnsw(hnsw))
        .build();

    let engine = rt
        .block_on(async {
            let engine = Engine::new(storage, config).await?;
            for i in 0..VECTOR_CORPUS_SIZE {
                let v = make_vector(i as u64, VECTOR_DIM);
                let doc = Document::builder()
                    .add_field("vec", DataValue::Vector(v))
                    .build();
                engine.put_document(&format!("doc{i}"), doc).await?;
            }
            engine.commit().await?;
            laurus::Result::Ok(engine)
        })
        .expect("vector engine build");

    (Arc::new(engine), temp_dir)
}

fn build_vector_queries(b: usize) -> Vec<laurus::SearchRequest> {
    (0..b)
        .map(|i| {
            let v = make_vector((i as u64).wrapping_mul(31) + 17, VECTOR_DIM);
            SearchRequestBuilder::new()
                .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
                    vector: Vector::new(v),
                    weight: 1.0,
                    fields: Some(vec!["vec".to_string()]),
                }]))
                .limit(10)
                .build()
        })
        .collect()
}

fn bench_engine_search_batch(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");

    // Lexical workload: per-query cost is low (~25 µs), so async parallelism
    // overhead can dominate at small B.
    let (engine, _temp_dir) = build_engine_with_corpus(&rt);
    let mut group = c.benchmark_group("engine_search_batch_lexical");
    for b in [1_usize, 4, 16, 64] {
        group.bench_with_input(BenchmarkId::new("serial_loop", b), &b, |bench, &b| {
            bench.iter(|| {
                rt.block_on(async {
                    let queries = build_queries(b);
                    let mut all = Vec::with_capacity(queries.len());
                    for q in queries {
                        let r = engine.search(q).await.expect("search").results;
                        all.push(r);
                    }
                    black_box(all);
                });
            });
        });

        group.bench_with_input(BenchmarkId::new("search_batch", b), &b, |bench, &b| {
            bench.iter(|| {
                rt.block_on(async {
                    let queries = build_queries(b);
                    let r = engine.search_batch(queries).await.expect("search_batch");
                    black_box(r);
                });
            });
        });
    }
    group.finish();

    // Vector workload: per-query HNSW search costs ~100-300 µs, so the
    // per-query work dominates over tokio dispatch overhead and async
    // parallelism actually pays off on multi-core hosts.
    let (vector_engine, _temp_dir_v) = build_vector_engine_with_corpus(&rt);
    let mut group = c.benchmark_group("engine_search_batch_vector");
    for b in [1_usize, 4, 16, 64] {
        group.bench_with_input(BenchmarkId::new("serial_loop", b), &b, |bench, &b| {
            bench.iter(|| {
                rt.block_on(async {
                    let queries = build_vector_queries(b);
                    let mut all = Vec::with_capacity(queries.len());
                    for q in queries {
                        let r = vector_engine.search(q).await.expect("search").results;
                        all.push(r);
                    }
                    black_box(all);
                });
            });
        });

        group.bench_with_input(BenchmarkId::new("search_batch", b), &b, |bench, &b| {
            bench.iter(|| {
                rt.block_on(async {
                    let queries = build_vector_queries(b);
                    let r = vector_engine
                        .search_batch(queries)
                        .await
                        .expect("search_batch");
                    black_box(r);
                });
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_engine_search_batch);
criterion_main!(benches);
