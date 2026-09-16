//! Criterion benchmarks for the highlighter.
//!
//! Targets `Highlighter::highlight` and `SimpleHighlighter::highlight_terms`
//! from `lexical::search::features::highlight`. These are the audit targets
//! tracked under #407 (pre-compile phrase regexes per query) and #408
//! (avoid re-tokenizing the full field text on every hit).
//!
//! # Scope
//!
//! Four measurement scenarios:
//!
//! 1. **`bench_simple_highlight_terms`** — `SimpleHighlighter::highlight_terms`.
//!    Sweeps text size {1 KB, 100 KB} × term count {1, 5, 20}. Each
//!    invocation compiles one `Regex` per term internally, exposing the
//!    cost #407 will reduce.
//! 2. **`bench_full_highlight_retokenize`** — `Highlighter::highlight` with
//!    a `TermQuery`. Sweeps text size {1 KB, 100 KB, 1 MB}. Each call
//!    invokes the analyzer over the full text inside `find_highlight_spans`,
//!    exposing the cost #408 will reduce.
//! 3. **`bench_full_highlight_top_k`** — `Highlighter::highlight` × K calls
//!    against a fixed-size text (~10 KB), simulating top-K result
//!    processing. Sweep K ∈ {1, 10, 50}. Reports `Throughput::Elements(K)`
//!    so per-hit cost is comparable.
//! 4. **`bench_full_highlight_dense_spans`** — `Highlighter::highlight` with
//!    a `PrefixQuery` over a text where every third token matches, so the
//!    fragment-grouping stage sees `n` spans (#595). Sweeps `n` on a 4×
//!    ladder {1k, 4k, 16k} so the growth exponent of that stage is
//!    visible directly.
//!
//! # Note on `extract_query_terms`
//!
//! `Highlighter::extract_query_terms` splits the query's `description()`
//! on whitespace and strips non-alphanumeric characters from the ends only.
//! `TermQuery::description()` returns `"field:term"`, which `trim_matches`
//! does not strip the embedded `:` from. The resulting term `"field:term"`
//! never matches an analyzer-produced token, so the highlight fragments
//! returned by scenarios 2 and 3 are typically empty. This is fine for the
//! purpose of this bench — the analyzer is still invoked over the whole
//! text inside `find_highlight_spans`, which is the cost we want to
//! measure. Scenario 1 (SimpleHighlighter) bypasses `extract_query_terms`
//! entirely and produces real `<mark>`-wrapped output, which is what the
//! sanity assert checks. Scenario 4 needs real spans and gets them through
//! `PrefixQuery`, whose description `"PrefixQuery(field: body, prefix: rust)"`
//! does leak the bare term `rust` past `trim_matches` — see its doc comment.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench highlight_bench
//! ```
//!
//! Filter by case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench highlight_bench -- "simple_highlight"
//! cargo bench --bench highlight_bench -- "retokenize/100KB"
//! cargo bench --bench highlight_bench -- "top_k/10"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench highlight_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use common::SAMPLE_SIZE_SLOW;

use laurus::lexical::search::features::highlight::{
    HighlightConfig, Highlighter, SimpleHighlighter,
};
use laurus::lexical::{PrefixQuery, TermQuery};

/// Vocabulary used for the synthetic English-like text. The set spans
/// common search-engine terminology so that terms picked from the same
/// vocabulary produce real matches in the SimpleHighlighter scenario.
const VOCAB: &[&str] = &[
    "search",
    "engine",
    "index",
    "document",
    "field",
    "term",
    "query",
    "rust",
    "performance",
    "latency",
    "throughput",
    "cluster",
    "node",
    "leader",
    "shard",
    "tokenize",
    "analyze",
    "vector",
    "similarity",
    "ranking",
];

/// Build a deterministic English-like text whose length is at least
/// `target_bytes`. Words are picked from `VOCAB` using a stride-based index
/// so two runs produce byte-identical input.
fn build_text(target_bytes: usize) -> String {
    let mut out = String::with_capacity(target_bytes + 16);
    let mut i = 0usize;
    while out.len() < target_bytes {
        if !out.is_empty() {
            out.push(' ');
        }
        let word_idx = (i * 7 + i / 5) % VOCAB.len();
        out.push_str(VOCAB[word_idx]);
        i += 1;
    }
    out
}

/// Pick `n` distinct terms deterministically from `VOCAB`.
fn pick_terms(n: usize) -> Vec<&'static str> {
    (0..n).map(|i| VOCAB[(i * 13) % VOCAB.len()]).collect()
}

fn bench_simple_highlight_terms(c: &mut Criterion) {
    let mut group = c.benchmark_group("highlight/simple_highlight");

    let highlighter = SimpleHighlighter::new(HighlightConfig::default());

    for &(label, target_bytes) in &[("1KB", 1024usize), ("100KB", 100 * 1024)] {
        let text = build_text(target_bytes);

        for &n_terms in &[1usize, 5, 20] {
            let terms = pick_terms(n_terms);

            // One-time sanity check: the result must contain at least one
            // <mark> tag, proving the regex compile + replace path produced
            // real output for the chosen vocabulary.
            let probe = highlighter.highlight_terms(&text, &terms);
            assert!(
                probe.contains("<mark>"),
                "simple_highlight probe must contain at least one <mark> tag (size={label}, n_terms={n_terms})"
            );

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{label}/n_terms_{n_terms}")),
                &(),
                |b, _| {
                    b.iter(|| {
                        let out = highlighter.highlight_terms(black_box(&text), black_box(&terms));
                        black_box(out);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Same workload shape as `bench_simple_highlight_terms`, but with the
/// regex patterns **pre-compiled outside the timed loop** via
/// `SimpleHighlighter::compile_patterns`. This is the case for callers
/// that reuse the same term set across many highlight calls (e.g. one
/// query × N search results); the per-call cost drops to
/// `replace_all` only.
///
/// Compare against `bench_simple_highlight_terms` at matching ids
/// (`1KB/n_terms_5` etc.) to see the regex-compile cost the
/// pre-compiled API avoids — this is the workload #407 reduces.
fn bench_simple_highlight_terms_precompiled(c: &mut Criterion) {
    let mut group = c.benchmark_group("highlight/simple_highlight_precompiled");

    let highlighter = SimpleHighlighter::new(HighlightConfig::default());

    for &(label, target_bytes) in &[("1KB", 1024usize), ("100KB", 100 * 1024)] {
        let text = build_text(target_bytes);

        for &n_terms in &[1usize, 5, 20] {
            let terms = pick_terms(n_terms);
            // Compile patterns ONCE, outside the timed loop.
            let patterns = SimpleHighlighter::compile_patterns(&terms);

            // Sanity check: the precompiled path must produce the same
            // <mark>-bearing output shape.
            let probe = highlighter.highlight_terms_compiled(&text, &patterns);
            assert!(
                probe.contains("<mark>"),
                "simple_highlight_precompiled probe must contain at least one <mark> tag (size={label}, n_terms={n_terms})"
            );

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{label}/n_terms_{n_terms}")),
                &(),
                |b, _| {
                    b.iter(|| {
                        let out = highlighter
                            .highlight_terms_compiled(black_box(&text), black_box(&patterns));
                        black_box(out);
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_full_highlight_retokenize(c: &mut Criterion) {
    let mut group = c.benchmark_group("highlight/retokenize");

    let highlighter = Highlighter::new(HighlightConfig::default());
    let query = TermQuery::new("body", "rust");

    for &(label, target_bytes) in &[
        ("1KB", 1024usize),
        ("100KB", 100 * 1024),
        ("1MB", 1024 * 1024),
    ] {
        let text = build_text(target_bytes);

        // Sanity check: highlight() must not error. Fragment count is
        // expected to be zero in this scenario because TermQuery's
        // description (`"body:rust"`) never matches an analyzer-produced
        // token; this is documented in the file header.
        let _probe = highlighter
            .highlight(&query, "body", &text)
            .expect("highlight probe must not error");

        group.bench_with_input(BenchmarkId::from_parameter(label), &(), |b, _| {
            b.iter(|| {
                let out = highlighter
                    .highlight(black_box(&query), black_box("body"), black_box(&text))
                    .unwrap();
                black_box(out);
            });
        });
    }

    group.finish();
}

fn bench_full_highlight_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("highlight/top_k");

    let highlighter = Highlighter::new(HighlightConfig::default());
    let query = TermQuery::new("body", "rust");
    let text = build_text(10 * 1024); // ~10 KB per hit, representative of a result snippet field

    // Sanity check: highlight() must not error.
    let _probe = highlighter
        .highlight(&query, "body", &text)
        .expect("highlight probe must not error");

    for &k in &[1usize, 10, 50] {
        group.throughput(Throughput::Elements(k as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("k_{k}")),
            &k,
            |b, &k| {
                b.iter(|| {
                    for _ in 0..k {
                        let out = highlighter
                            .highlight(black_box(&query), black_box("body"), black_box(&text))
                            .unwrap();
                        black_box(out);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Dense-span fragment grouping (#595). Every third token matches, so `n`
/// repetitions yield `n` merged spans and `group_spans_into_fragments` has
/// `n` windows to fill. Before the fix that stage rescanned every span per
/// window and rendered every candidate fragment before the top-
/// `max_fragments` cut (quadratic); after it, one binary search per window
/// and rendering only the survivors (`n log n`). The 4× ladder makes the
/// exponent readable: ~16× per step is quadratic, ~4× is linear
/// (tokenisation-bound).
///
/// Uses `PrefixQuery` rather than `TermQuery` on purpose: `PrefixQuery`'s
/// `description()` is `"PrefixQuery(field: body, prefix: rust)"`, from
/// which `extract_query_terms` recovers the bare `rust`, whereas a
/// `TermQuery`'s `"body:rust"` never splits (header note) — that is why
/// scenarios 2 and 3 return zero fragments. If #594 replaces the
/// description-string extraction, revisit this coupling.
fn bench_full_highlight_dense_spans(c: &mut Criterion) {
    let mut group = c.benchmark_group("highlight/dense_spans");
    // The pre-fix 16k case runs for hundreds of milliseconds per call.
    group.sample_size(SAMPLE_SIZE_SLOW);

    let config = HighlightConfig::default();
    let expected_fragments = config.max_fragments;
    let highlighter = Highlighter::new(config);
    let query = PrefixQuery::new("body", "rust");

    for &n in &[1_000usize, 4_000, 16_000] {
        let text = "alpha beta rust ".repeat(n);

        // Sanity: the term must leak through `extract_query_terms` and the
        // grouping stage must produce real fragments, otherwise this would
        // measure the zero-fragment path like scenarios 2 and 3.
        let probe = highlighter
            .highlight(&query, "body", &text)
            .expect("highlight probe must not error");
        assert_eq!(
            probe.fragments.len(),
            expected_fragments,
            "dense_spans probe must fill max_fragments (n={n})"
        );
        assert!(
            probe
                .fragments
                .iter()
                .all(|f| f.text.contains("<mark>rust</mark>")),
            "every dense_spans fragment must carry a highlighted term (n={n})"
        );

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("spans_{n}")),
            &(),
            |b, _| {
                b.iter(|| {
                    let out = highlighter
                        .highlight(black_box(&query), black_box("body"), black_box(&text))
                        .unwrap();
                    black_box(out);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simple_highlight_terms,
    bench_simple_highlight_terms_precompiled,
    bench_full_highlight_retokenize,
    bench_full_highlight_top_k,
    bench_full_highlight_dense_spans,
);
criterion_main!(benches);
