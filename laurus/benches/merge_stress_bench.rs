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
//! Compile-only smoke check (skips the runtime, used by CI):
//!
//! ```sh
//! cargo bench --bench merge_stress_bench --no-run
//! ```
//!
//! Peak-memory reports print to stdout once per size, before the timed
//! Criterion loop starts for that size — look for `peak bytes for N=...`
//! lines in the `cargo bench` output.

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

/// Build a `total_docs`-document, `SEGMENTS`-segment `LexicalStore` (one
/// commit per segment), with a text field, a single-valued numeric field,
/// and a multi-valued numeric field — the same three-field shape
/// `lexical_merge_stress_test.rs` uses, so this bench and that test track
/// the same workload.
fn build_multi_segment_store(total_docs: usize) -> (std::sync::Arc<dyn Storage>, LexicalStore) {
    let storage: std::sync::Arc<dyn Storage> =
        std::sync::Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    let docs_per_segment = total_docs / SEGMENTS;
    let mut doc_id = 0u64;
    for _ in 0..SEGMENTS {
        for _ in 0..docs_per_segment {
            let words: Vec<String> = (0..4u64)
                .map(|k| format!("word{}", (doc_id.wrapping_mul(7).wrapping_add(k)) % 2_000))
                .collect();
            let score = (doc_id.wrapping_mul(37) % 1000) as i64 - 500;
            let tags: Vec<i64> = (0..3u64)
                .map(|k| ((doc_id.wrapping_mul(11).wrapping_add(k)) % 2000) as i64)
                .collect();
            let doc = Document::builder()
                .add_text("body", words.join(" "))
                .add_integer("score", score)
                .add_int64_array("tags_num", tags)
                .build();
            store.upsert_document(doc_id, doc).unwrap();
            doc_id += 1;
        }
        store.commit().unwrap();
    }
    (storage, store)
}

fn bench_merge_optimize(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_optimize");
    group.sample_size(SAMPLE_SIZE_SLOW);

    for &n in SIZES {
        // One-time sanity call + peak-memory report (common.rs hygiene
        // rule #3's slot), before the timed Criterion loop for this size.
        let (_, store) = build_multi_segment_store(n);
        let (result, peak_bytes) = measure_peak_bytes(|| store.optimize());
        result.unwrap();
        println!("peak bytes for N={n}: {peak_bytes}");

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || build_multi_segment_store(n),
                |(_, store)| {
                    store.optimize().unwrap();
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_merge_optimize);
criterion_main!(benches);
