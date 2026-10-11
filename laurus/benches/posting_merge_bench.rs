//! Criterion benchmark for the multi-segment posting iterator merger.
//!
//! Targets [`MergedPostingIterator::next`] from
//! `lexical::index::inverted::reader`. This is the audit target tracked
//! under #412 (use a linear-scan posting merge for small segment counts;
//! today every merge goes through `BinaryHeap`).
//!
//! # Scope
//!
//! - Synthetic micro-bench. Constructs `N` separate
//!   [`InvertedIndexPostingIterator`]s with overlapping doc IDs
//!   (interleaved across segments) so the merger has to interleave reads,
//!   then drives the merged iterator's `next()` to exhaustion.
//! - The bench bypasses `Engine` entirely so auto-merge (which runs after
//!   a commit once the segment count exceeds `max_segments`) cannot
//!   collapse segments under our feet.
//! - Sweep `N ∈ {1, 2, 4, 8, 16}` × fixed `postings_per_segment = 10_000`.
//!   For `N = 1` the merger holds a single iterator, exposing the
//!   `MergedPostingIterator` wrapper overhead even when no merging is
//!   needed.
//! - Inputs are deterministic: doc IDs are computed as
//!   `i * N + segment_index` so every iter has a strictly-ascending
//!   sequence and the global merged stream is also strictly ascending.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench posting_merge_bench
//! ```
//!
//! Filter by N:
//!
//! ```sh
//! cargo bench --bench posting_merge_bench -- "merge/8"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench posting_merge_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use laurus::lexical::index::inverted::core::posting::Posting;
use laurus::lexical::index::inverted::reader::{
    InvertedIndexPostingIterator, MergedPostingIterator,
};
use laurus::lexical::reader::PostingIterator;

/// Number of postings per synthetic segment. Picked large enough that the
/// per-call overhead of `next()` dominates iteration setup, but small
/// enough that the largest case (`N = 16` → 160k postings) runs in well
/// under a second per iter.
const POSTINGS_PER_SEGMENT: usize = 10_000;

/// Build a single segment's posting list. Document IDs are
/// `(start_doc_id, start_doc_id + stride, start_doc_id + 2 * stride, …)`
/// so each segment's stream is strictly ascending and segments interleave
/// when merged.
fn build_segment_postings(start_doc_id: u64, stride: u64, count: usize) -> Vec<Posting> {
    (0..count)
        .map(|i| Posting::new(start_doc_id + (i as u64) * stride))
        .collect()
}

/// Build `n` segment iterators ready to be passed to
/// `MergedPostingIterator::new`. The `i`th segment owns doc IDs
/// `{i, i + n, i + 2n, …}` so the merged stream visits doc IDs
/// `0, 1, 2, …` in order, exercising the cross-segment selection logic on
/// every step.
fn build_segment_iterators(n: usize) -> Vec<Box<dyn PostingIterator>> {
    (0..n)
        .map(|seg_idx| {
            let postings = build_segment_postings(seg_idx as u64, n as u64, POSTINGS_PER_SEGMENT);
            Box::new(InvertedIndexPostingIterator::new(postings)) as Box<dyn PostingIterator>
        })
        .collect()
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("posting_merge/next");

    for &n in &[1usize, 2, 4, 8, 16] {
        let total_postings = (n * POSTINGS_PER_SEGMENT) as u64;

        // One-time sanity check: a freshly built merged iterator must
        // produce strictly ascending doc IDs and exhaust at exactly
        // `n * POSTINGS_PER_SEGMENT` items. Failing this means the test
        // data generator drifted out of sync with the merger contract.
        let mut probe = MergedPostingIterator::new(build_segment_iterators(n))
            .expect("merger probe must construct");
        let mut last = None::<u64>;
        let mut count = 0u64;
        while probe
            .next()
            .expect("merger probe must not error during next()")
        {
            let doc = probe.doc_id();
            if let Some(prev) = last {
                assert!(
                    doc >= prev,
                    "merged stream must be non-decreasing (n={n}, prev={prev}, doc={doc})"
                );
            }
            last = Some(doc);
            count += 1;
        }
        assert_eq!(
            count, total_postings,
            "merged stream length mismatch at n={n}: expected {total_postings}, got {count}"
        );

        group.throughput(Throughput::Elements(total_postings));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("merge/{n}")),
            &n,
            |b, &n| {
                b.iter_batched(
                    || {
                        MergedPostingIterator::new(build_segment_iterators(n))
                            .expect("merger setup must succeed")
                    },
                    |mut merger| {
                        while merger.next().unwrap() {
                            black_box(merger.doc_id());
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_merge);
criterion_main!(benches);
