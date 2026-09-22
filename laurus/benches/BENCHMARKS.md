<!-- markdownlint-disable MD060 -->

# Benchmark architecture

This document captures the design rationale for the laurus benchmark
suite, with the focus on *why* it looks the way it does. Individual
bench files have file-level docstrings covering scope, vocabulary, and
how to run them; this file covers the cross-cutting concerns.

Japanese translation: [BENCHMARKS_ja.md](BENCHMARKS_ja.md).

## Three-phase model

Every search-side bench in this suite logically runs in three
distinct phases:

```
┌──────────────────────────┐
│ Phase 1 — corpus gen     │  Pure functions (`build_body*`)
│                          │  Input: i (integer)  Output: String
│ Cost: ~1 second / 100k   │  Deterministic, in-memory only.
└──────────────────────────┘
            ↓
┌──────────────────────────┐
│ Phase 2 — index build    │  engine.add_document × n + commit
│                          │  Input: corpus       Output: index files
│ Cost: ~17 min / 100k     │  ★ The expensive phase. Persisted to disk.
└──────────────────────────┘
            ↓
┌──────────────────────────┐
│ Phase 3 — search measure │  engine.search inside Criterion `b.iter`
│                          │  Input: index + query  Output: results + latency
│ Cost: ~10 sec / scenario │  The only phase whose latency we record.
└──────────────────────────┘
```

For the **search-side benches** (`lexical_search_bench`) the goal is
to measure Phase 3 in isolation. Phases 1 and 2 are setup, not
measurement.

For the **indexing-side benches** (`lexical_indexing_bench`) Phase 2
*is* the measurement target. Phase 1 is setup; Phase 3 does not exist
in that binary.

Microbenches (`bm25_simd_bench`, etc.) sit beside the three-phase
model: they exercise a single kernel directly with synthetic inputs,
no engine, no corpus, no Criterion-async wrapping.

## Why an on-disk index cache (#510)

Before the cache, each search-side bench rebuilt its index from
scratch every `cargo bench` invocation. Multiple bench functions
needed the same corpus shape (uniform 100k, skewed 100k, …) and each
function paid the build cost independently. A single full run hit
this multiplier 4–6 times for the 100k uniform case alone, taking
1–2 hours of pure rebuild before any latency was measured.

`cached_engine` writes the built index to
`target/laurus_bench_index_cache/<shape>_<n>_segs<k>_v<N>/` and
reopens it on subsequent runs via laurus's standard
`Engine::builder(file_storage, schema).build()` path (whose internal
`recover()` reattaches existing segments). After the first run, every
later `cargo bench` reaches Phase 3 in well under a second.

Invalidation:

- `BENCH_INDEX_FORMAT_VERSION` — bump in source when the schema /
  analyzer / corpus synthesis / segment format changes. Stale caches
  are auto-rebuilt.
- `LAURUS_BENCH_REBUILD=1` — manual override, useful when iterating
  on bench shape itself.
- `cargo clean` — evicts the cache along with the rest of `target/`.

The cache is **search-side only**. Indexing benches do not call
`cached_engine`; they build a fresh in-memory engine every iteration
because the build itself is what they measure.

## Why synthetic data, not TREC / Wikipedia

Lucene's benchmark framework supports external corpora (TREC,
Wikipedia dumps, `LineDocSource` over arbitrary `.txt` files) and is
the de-facto standard for "scale-out" search engine comparisons. The
laurus suite intentionally does not adopt that approach. The
trade-off:

|                         | External corpus              | Synthetic (current)         |
| ----------------------- | ---------------------------- | --------------------------- |
| Corpus size             | GB-scale (Wikipedia ~22 GB)  | ~few MB (100 k × ~300 B/doc)|
| Vocabulary              | Real (millions of unique terms) | Synthetic (~150 unique terms) |
| Reproducibility         | Depends on snapshot date     | Deterministic from source   |
| CI / fresh checkout cost | Multi-GB download, license review | Zero — in-process generation |
| Real-world realism      | High                         | Medium (Zipf-shaped synthetic) |
| Useful for **ratio** comparisons (PR before/after) | Yes | Yes |
| Useful for **absolute** throughput claims | Yes | No (synthetic vocab limits headline numbers) |

The laurus bench suite's actual workflow is **perf-PR ratio
comparison**: "did my change make the 100k search 1.2x faster?" For
that question, synthetic data is sufficient as long as:

1. Term-frequency distribution is realistic enough to exercise the
   relevant code paths (BMW pivot, skip list, top-K early termination,
   etc.). The 3-tier Zipf-like distribution (`COMMON_TERMS`,
   `TOPIC_PHRASES`, `LONG_TAIL`) achieves this — verified across #403
   (BMW), #466 (enum dispatch), #503 (skip list), #506 (SIMD batch).
2. Posting list lengths are long enough that the structures under
   test light up. At 100 k docs the rare-term posting list is ~6 k
   entries, which produces a `log_8(6_000) ≈ 4`-level skip-list
   hierarchy — within the same order as a 1 M-doc test would produce
   (`log_8(60_000) ≈ 5`).
3. Cache effects (postings in L2 vs DRAM) emerge somewhere in the
   sweep. 100k is enough to spill out of typical L3 (~32-64 MB) and
   exercise the DRAM path; the 10k case stays in cache and gives a
   clean control measurement.

Items the synthetic approach cannot answer:

- "How does laurus compare to Lucene at 10 M docs on enwiki?" — needs
  external corpus support, not addressed here. Open an issue if a
  real-corpus comparison becomes load-bearing.
- "Does the analyzer matter on real-world long-tail vocabulary?" —
  the synthetic vocab is small enough that analyzer cost is
  proportionally smaller than it would be on real text.

For the kind of perf work the suite is currently used for (#463
umbrella), synthetic data wins on every dimension that actually
matters: CI speed, determinism, no external dependencies, no license
encumbrance.

## Why we don't reuse Lucene's bench framework

Lucene's `benchmark/` module ships a task DSL (`.alg` files), content
sources, document makers, and a long list of measurable tasks
(`AddDocTask`, `SearchTravRetTask`, …). It is impressive and
appropriate for that project's scale (decades of contributors, hundreds
of variation points).

Laurus uses Criterion in plain Rust function form because:

1. Criterion already handles warmup, sample collection, statistical
   analysis, and the `--baseline` / `--save-baseline` workflow that
   perf PRs need.
2. The bench functions here are read top-to-bottom; there's no point
   serialising them through a DSL when the code is already
   declarative.
3. Onboarding cost — a contributor who knows Rust can read a bench
   file and immediately understand what it measures. The Lucene DSL
   adds another layer to learn.

If laurus ever needs Lucene-style "run a mixed indexing + searching
workload that injects N docs per second while M concurrent searchers
hit it" — that's a different kind of bench (a soak test) and would
warrant a dedicated harness. Single-shot perf benches don't need it.

## File-by-file scope (summary)

| Bench file                    | Phase measured | Cache used | Notes |
| ----------------------------- | -------------- | ---------- | ----- |
| `lexical_search_bench`        | Phase 3        | Yes        | Term / Boolean / Phrase / Fuzzy / DSL search throughput |
| `lexical_indexing_bench`      | Phase 2        | No         | `add_document` + `commit` cost |
| `bm25_simd_bench`             | Microbench     | n/a        | SIMD BM25 kernel only (no engine) |
| `posting_skip_to_bench`       | Microbench     | (own setup)| Skip-list seek microbench |
| `posting_merge_bench`         | Microbench     | (own setup)| Segment-merge microbench |
| `dict_lookup_bench`           | Phase 3 / micro | (own setup) | Dictionary lookup variants |
| `hybrid_search_bench`         | Phase 3        | Yes        | Lexical + vector hybrid (cached via `cached_hybrid_engine`, #513 Stage 1) |
| `bkd_bench`                   | Microbench     | (own setup)| Geo / numeric range |
| `distance_bench`              | Microbench     | n/a        | Distance kernels |
| `facet_bench`                 | Phase 3        | (own setup)| Facet collection |
| `highlight_bench`             | Phase 3        | (own setup)| Highlighter |
| `mutation_bench`              | Phase 2-like   | n/a        | Update / delete throughput |
| `search_perf`                 | Phase 3        | (own setup)| Legacy entrypoint |
| `spell_correction_bench`      | Phase 3        | n/a        | Spelling suggestion |
| `store_fetch_bench`           | Phase 2-like   | (own setup)| Document store fetch |
| `synonym_bench`               | Phase 3        | n/a        | Synonym expansion |
| `text_analysis_bench`         | Microbench     | n/a        | Analyzer pipeline |
| `vector_indexing_bench`       | Phase 2-like   | n/a        | Vector index build |
| `vector_search_bench`         | Phase 3        | Yes        | Vector kNN (search-only benches cached via `cached_vector_reader`, #513 Stage 2; construction benches unchanged) |
| `merge_stress_bench`          | Phase 2-like   | No         | `MergeEngine::perform_merge` via `optimize()`; own `TrackingAllocator` global allocator for peak-byte reports (#1167) |

Items marked "(own setup)" use a smaller corpus shape or a different
setup pattern that does not need a generalised cache. With #513
Stage 1 (`hybrid_search_bench`) and Stage 2 (`vector_search_bench`)
shipped, every search-side bench in the table above reuses an
on-disk cache. The cache helpers live next to the bench files that
use them rather than being lifted into `benches/common.rs`; that
extraction is a separate design discussion (issue #513 body).

## How to add a new search-side bench

1. Pick a corpus shape: `EngineShape::Uniform` for plain text,
   `EngineShape::Skewed` for the TF-skewed fixture.
2. Decide which size-gate helper to use:
   [`corpus_sizes`](lexical_search_bench.rs) for `(small … LARGE-adds-100k)`,
   [`skewed_corpus_sizes`](lexical_search_bench.rs) for `(1k, 10k …
   LARGE-adds-100k)`, or [`seek_skewed_sizes`](lexical_search_bench.rs) for
   the skip-list shape.
3. Call `cached_engine(&rt, shape, n, segment_count)` to obtain an
   `Arc<Engine>`. The cache is keyed on `(shape, n, segment_count,
   BENCH_INDEX_FORMAT_VERSION)`; first call builds, later calls reopen.
4. Run a `probe` query before the `b.iter` loop to assert the
   fixture / query pair returns at least one hit. This catches
   corpus / query drift cheaply.
5. Place `b.to_async(&rt).iter(|| { let engine = &engine; async move
   { … } })` inside the closure. Capturing the `Arc<Engine>` by
   reference is fine — auto-deref handles the method call.

When changing anything that would alter the resulting index — schema,
analyzer, corpus synthesis, segment format — bump
`BENCH_INDEX_FORMAT_VERSION` in `lexical_search_bench.rs`. Stale
caches will be auto-rebuilt on the next run.

## Peak-memory measurement (`merge_stress_bench`)

`merge_stress_bench` is the first bench in this suite to instrument the
allocator: it defines a `TrackingAllocator` (`#[global_allocator]`) that
tracks a high-water mark of net allocated bytes, reported via `println!`
once per size before the timed Criterion loop starts. A
`#[global_allocator]` applies per compiled binary, and each `[[bench]]`
entry is its own binary, so this has zero effect on any other bench,
on `cargo test`, or on the library itself when used elsewhere — it is
safe to define independently in future benches that need the same
signal, without touching `common.rs`.

This is a directional proxy for A/B comparisons (e.g. before/after a
memory-focused perf PR on the same machine), not publication-grade
profiling, and it assumes the code under measurement is single-threaded
(concurrent allocation activity would corrupt the single global
high-water mark) — check for `rayon`/thread spawning in whatever you
point it at before reusing the pattern.

## Background

- `#510` introduced the on-disk cache and size-gate cleanup for
  `lexical_search_bench`.
- `#513` extended the cache to `hybrid_search_bench` (Stage 1, PR #514)
  and `vector_search_bench` (Stage 2, search-only benches via
  `cached_vector_reader`).
- `#403` defined the BMW acceptance-gate that anchored 100k as the
  reference size.
- `#503` introduced `bench_seek_skewed`; the 1 M-doc case from #503's
  audit window was removed by #510 because the binary-search hierarchy
  it tested adds only one level vs 100k.
