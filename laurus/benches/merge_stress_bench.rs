//! Criterion benchmark for `MergeEngine::perform_merge` (via
//! `LexicalStore::optimize()`), at a range of realistic corpus sizes
//! (Issue #1167).
//!
//! # Scope
//!
//! - Builds a multi-segment `LexicalStore` (4 segments, docs split evenly)
//!   with a text field, a single-valued numeric field, and a multi-valued
//!   numeric field, then force-merges it with `optimize()`.
//! - Sweeps `total_docs ∈ {1_000, 5_000, 20_000}`. Kept well below the
//!   integration test's 30,000-document stress scale (`lexical_merge_stress_test.rs`)
//!   because `iter_batched` rebuilds the whole multi-segment corpus once per
//!   sample (merge is non-idempotent — it deletes its source segments), and
//!   `SAMPLE_SIZE_SLOW` samples of very large rebuilds would make a full
//!   `cargo bench` run impractically slow (the same reason `bkd_bench.rs`'s
//!   own `bench_build` caps its sweep well below its query benches' 1M-point
//!   case).
//! - Two corpus shapes (Issue #1168): `short` (4-word bodies, the original
//!   workload) and `long` (200-word bodies). Short bodies make term data a
//!   small share of the peak, so changes to how analyzed terms and postings
//!   are held only show up clearly on `long`. `merge_optimize` keeps its
//!   original short-shape IDs (`merge_optimize/{n}`) so earlier results stay
//!   comparable, and adds one `merge_optimize/long/4000` case.
//! - `ingest` measures the normal write path — one segment's worth of
//!   documents upserted into a fresh store, then committed — for both
//!   shapes. The documents are built outside the measured closure.
//! - Peak-memory measurement: this is the first bench in the suite to
//!   instrument the allocator. `TrackingAllocator` (`#[global_allocator]`)
//!   tracks a high-water mark of net allocated bytes; it is scoped to this
//!   one bench binary only (a `#[global_allocator]` applies per compiled
//!   binary, and each `[[bench]]` entry in `Cargo.toml` is its own binary —
//!   this has zero effect on other benches, on `cargo test`, or on the
//!   library itself). Sound because the merge path (`MergeEngine`,
//!   `InvertedIndexWriter`) spawns no threads and uses no `rayon` parallelism
//!   — confirmed by grep — so a single global high-water mark cannot be
//!   corrupted by concurrent allocation activity from the code under test.
//!   The measurement is a standalone `println!` report (run once per size,
//!   in the same "call the routine once before `b.iter`" slot `common.rs`'s
//!   hygiene rule already reserves for a sanity check), not a full Criterion
//!   `Measurement` plugin — this scope only needs a directional
//!   before/after signal for #1163/#1164/#1165/#1168, not
//!   publication-grade profiling.
//!
//! # Running
//!
//! ```sh
//! cargo bench --bench merge_stress_bench
//! ```
//!
//! Filter by size:
//!
//! ```sh
//! cargo bench --bench merge_stress_bench -- merge_optimize/20000
//! ```
//!
//! Filter to the bounded-merge budget sweep (Issue #1164):
//!
//! ```sh
//! cargo bench --bench merge_stress_bench -- merge_bounded
//! ```
//!
//! Compile-only smoke check (skips the runtime, used by CI):
//!
//! ```sh
//! cargo bench --bench merge_stress_bench --no-run
//! ```
//!
//! Peak-memory reports print to stdout once per case, before the timed
//! Criterion loop starts for it — look for `peak bytes for ...` lines in the
//! `cargo bench` output. `merge_bounded` lines also report how many output
//! segments the merge produced. To print every peak without running any
//! timed loop, pass a filter that matches no benchmark ID:
//!
//! ```sh
//! cargo bench --bench merge_stress_bench -- peaks-only
//! ```

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use laurus::Document;
use laurus::lexical::{LexicalIndexConfig, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

use common::SAMPLE_SIZE_SLOW;

/// Net bytes currently outstanding (allocated - deallocated), tracked
/// across the whole process. See the file-level doc comment for why a
/// single non-thread-partitioned counter is sound here.
static CURRENT: AtomicI64 = AtomicI64::new(0);

/// High-water mark of [`CURRENT`] observed since it was last reset.
static PEAK: AtomicI64 = AtomicI64::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let cur =
                CURRENT.fetch_add(layout.size() as i64, Ordering::Relaxed) + layout.size() as i64;
            PEAK.fetch_max(cur, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size() as i64, Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

/// Reset the high-water mark to the current outstanding total, run `f`,
/// and return `(result, peak_bytes_above_baseline)`.
fn measure_peak_bytes<T>(f: impl FnOnce() -> T) -> (T, i64) {
    let baseline = CURRENT.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let result = f();
    let peak = PEAK.load(Ordering::Relaxed);
    (result, peak - baseline)
}

/// Total document counts to sweep. See the file-level doc comment for why
/// this stays well below the integration test's 30,000-document scale.
const SIZES: &[usize] = &[1_000, 5_000, 20_000];

/// Fixed segment count across every size (docs are split evenly).
const SEGMENTS: usize = 4;

/// Document body length for a corpus. See the file-level doc comment for
/// why both shapes exist.
#[derive(Clone, Copy)]
enum Shape {
    /// 4-word bodies — the original workload.
    Short,
    /// 200-word bodies, so analyzed terms and postings dominate.
    Long,
}

impl Shape {
    fn label(self) -> &'static str {
        match self {
            Shape::Short => "short",
            Shape::Long => "long",
        }
    }

    fn words_per_doc(self) -> u64 {
        match self {
            Shape::Short => 4,
            Shape::Long => 200,
        }
    }
}

/// Document `doc_id` of a corpus: a text body, a single-valued numeric
/// field, and a multi-valued numeric field — the same three-field shape
/// `lexical_merge_stress_test.rs` uses, so this bench and that test track
/// the same workload.
fn make_doc(doc_id: u64, shape: Shape) -> Document {
    let words: Vec<String> = (0..shape.words_per_doc())
        .map(|k| format!("word{}", (doc_id.wrapping_mul(7).wrapping_add(k)) % 2_000))
        .collect();
    let score = (doc_id.wrapping_mul(37) % 1000) as i64 - 500;
    let tags: Vec<i64> = (0..3u64)
        .map(|k| ((doc_id.wrapping_mul(11).wrapping_add(k)) % 2000) as i64)
        .collect();
    Document::builder()
        .add_text("body", words.join(" "))
        .add_integer("score", score)
        .add_int64_array("tags_num", tags)
        .build()
}

/// An empty `LexicalStore` over fresh in-memory storage.
fn empty_store() -> (std::sync::Arc<dyn Storage>, LexicalStore) {
    let storage: std::sync::Arc<dyn Storage> =
        std::sync::Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();
    (storage, store)
}

/// Build a `total_docs`-document, `SEGMENTS`-segment `LexicalStore` (one
/// commit per segment).
fn build_multi_segment_store(
    total_docs: usize,
    shape: Shape,
) -> (std::sync::Arc<dyn Storage>, LexicalStore) {
    let (storage, store) = empty_store();
    let docs_per_segment = total_docs / SEGMENTS;
    let mut doc_id = 0u64;
    for _ in 0..SEGMENTS {
        for _ in 0..docs_per_segment {
            store
                .upsert_document(doc_id, make_doc(doc_id, shape))
                .unwrap();
            doc_id += 1;
        }
        store.commit().unwrap();
    }
    (storage, store)
}

/// Number of distinct `merged_*` segments in `storage`, counted by file stem
/// so it works for both the loose and the compound (`.cfs`) layout.
fn merged_segment_count(storage: &std::sync::Arc<dyn Storage>) -> usize {
    let stems: std::collections::BTreeSet<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.starts_with("merged_"))
        .filter_map(|f| f.split('.').next().map(str::to_string))
        .collect();
    stems.len()
}

/// `(shape, total_docs)` cases for `merge_optimize`: the original short
/// sweep, plus one long-body case sized to build in about the same time as
/// the largest short one.
fn merge_optimize_cases() -> Vec<(Shape, usize)> {
    let mut cases: Vec<(Shape, usize)> = SIZES.iter().map(|&n| (Shape::Short, n)).collect();
    cases.push((Shape::Long, 4_000));
    cases
}

fn bench_merge_optimize(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_optimize");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for (shape, n) in merge_optimize_cases() {
        // One-time sanity call + peak-memory report (common.rs hygiene
        // rule #3's slot), before the timed Criterion loop for this size.
        let (_, store) = build_multi_segment_store(n, shape);
        let (result, peak_bytes) = measure_peak_bytes(|| store.optimize());
        result.unwrap();
        println!(
            "peak bytes for merge, shape={}, N={n}: {peak_bytes}",
            shape.label()
        );

        // The short sweep keeps its original IDs so earlier results stay
        // comparable.
        let id = match shape {
            Shape::Short => BenchmarkId::from_parameter(n),
            Shape::Long => BenchmarkId::new(shape.label(), n),
        };
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(id, &n, |b, &n| {
            b.iter_batched(
                || build_multi_segment_store(n, shape),
                |(_, store)| {
                    store.optimize().unwrap();
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// `(shape, docs)` cases for `ingest`: one segment's worth of documents,
/// each under the default `max_buffered_docs` so the whole batch is buffered
/// before the commit flushes it.
const INGEST_CASES: &[(Shape, usize)] = &[(Shape::Short, 5_000), (Shape::Long, 1_000)];

fn bench_ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest");
    group.sample_size(SAMPLE_SIZE_SLOW);

    let ingest = |store: &LexicalStore, docs: Vec<Document>| {
        for (doc_id, doc) in docs.into_iter().enumerate() {
            store.upsert_document(doc_id as u64, doc).unwrap();
        }
        store.commit().unwrap();
    };

    for &(shape, n) in INGEST_CASES {
        let make_docs = || -> Vec<Document> { (0..n as u64).map(|i| make_doc(i, shape)).collect() };

        let (_, store) = empty_store();
        let docs = make_docs();
        let ((), peak_bytes) = measure_peak_bytes(|| ingest(&store, docs));
        println!(
            "peak bytes for ingest, shape={}, N={n}: {peak_bytes}",
            shape.label()
        );

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new(shape.label(), n), &n, |b, _| {
            b.iter_batched(
                || (empty_store(), make_docs()),
                |((_, store), docs)| ingest(&store, docs),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// `(label, budget_bytes)` sweep for the bounded-merge peak-memory
/// comparison (Issue #1164). `usize::MAX` reproduces `bench_merge_optimize`
/// exactly (always one output segment); the smaller budgets are chosen to
/// force a rollover into multiple output segments at this bench's sizes, so
/// the `peak bytes for ...` lines printed for each show the directional
/// before/after signal this file's doc comment promises for #1164.
const BUDGETS: &[(&str, usize)] = &[
    ("8mib", 8 * 1024 * 1024),
    ("32mib", 32 * 1024 * 1024),
    ("unbounded", usize::MAX),
];

fn bench_merge_bounded(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_bounded");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in SIZES {
        for &(label, budget) in BUDGETS {
            let (storage, store) = build_multi_segment_store(n, Shape::Short);
            let (result, peak_bytes) = measure_peak_bytes(|| store.optimize_within_budget(budget));
            result.unwrap();
            println!(
                "peak bytes for merge_bounded, N={n}, budget={label}: {peak_bytes} \
                 ({} output segments)",
                merged_segment_count(&storage)
            );

            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(
                BenchmarkId::new(label, n),
                &(n, budget),
                |b, &(n, budget)| {
                    b.iter_batched(
                        || build_multi_segment_store(n, Shape::Short),
                        |(_, store)| {
                            store.optimize_within_budget(budget).unwrap();
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_merge_optimize,
    bench_merge_bounded,
    bench_ingest
);
criterion_main!(benches);
