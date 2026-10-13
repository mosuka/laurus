# Benchmarking

This guide describes how to run laurus benchmarks, how to capture and compare baselines, and how to report results in pull requests.

The benchmark suite lives in [`laurus/benches/`](https://github.com/mosuka/laurus/tree/main/laurus/benches) and is built on [Criterion](https://github.com/bheisler/criterion.rs). Hygiene rules — deterministic seeds, file-level documentation, sanity asserts, and `sample_size` policy — are codified in [`laurus/benches/common.rs`](https://github.com/mosuka/laurus/blob/main/laurus/benches/common.rs).

## Suite overview

| File | Scope |
| --- | --- |
| `bkd_bench.rs` | BKD tree range search, intersect, and build (1D / 2D / 3D, 10k / 100k / 1M points) |
| `distance_bench.rs` | `DistanceMetric::distance` for cosine, Euclidean, Manhattan, dot product (single dimension today; sweep tracked in #424) |
| `lexical_search_bench.rs` | End-to-end lexical search through `Engine::search` for term, boolean, phrase, fuzzy, and DSL queries |
| `search_perf.rs` | Posting iterator `skip_to`, `BM25Scorer::score`, SIMD batch scoring, compact posting conversion |
| `spell_correction_bench.rs` | `SpellingCorrector::correct` with a fresh corrector per iteration (cold-state measurement) |
| `synonym_bench.rs` | `SynonymDictionary::get_synonyms` lookup at 100 / 1k / 10k groups, plus build cost |
| `text_analysis_bench.rs` | `StandardAnalyzer::analyze` single-document and batch (100 docs) |
| `vector_search_bench.rs` | Flat / IVF / HNSW construction and search at 1k / 5k vectors, dim 128, top-10; plus a large-K IVF case (512 / 2048 clusters) for the cluster-selection path (#668) |

Each file declares its scope, scenarios, and how to filter in its top-of-file `//!` doc comment. Read it before running.

## Running benchmarks

Run a single bench file:

```bash
cargo bench -p laurus --bench distance_bench
```

Filter by criterion id (substring match):

```bash
cargo bench -p laurus --bench distance_bench -- cosine
cargo bench -p laurus --bench vector_search_bench -- "HNSW Search/top10"
```

Compile-only smoke check (useful in CI or during refactors):

```bash
cargo bench -p laurus --bench distance_bench --no-run
```

Run every bench file in the workspace:

```bash
cargo bench -p laurus
```

## Saving and comparing baselines

Criterion supports named baselines so you can compare a feature branch against `main` (or any other reference run).

Save a baseline named `main` from your current state:

```bash
cargo bench -p laurus --bench distance_bench -- --save-baseline main
```

Compare a subsequent run against that baseline:

```bash
cargo bench -p laurus --bench distance_bench -- --baseline main
```

The output prints a `change:` line per benchmark with a percentage and a verdict (`No change in performance detected`, `Performance has improved`, `Performance has regressed`). Criterion stores baselines under `target/criterion/<bench-id>/<baseline>/`.

Recommended flow for a perf PR:

1. On `main` (or before any change) — `cargo bench --bench RELEVANT -- --save-baseline main`.
2. Make the change on a branch.
3. On the branch — `cargo bench --bench RELEVANT -- --baseline main`.
4. Copy the `change:` lines into the PR description.

## Recommended environment

Microbenchmarks at the µs / ns scale are sensitive to system noise. For meaningful numbers:

- **CPU governor**: set to `performance` (Linux):

  ```bash
  sudo cpupower frequency-set -g performance
  ```

- **Turbo boost**: disable so frequency scaling does not skew results:

  ```bash
  echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo   # Intel
  ```

  AMD systems and BIOS-level switches differ; consult vendor docs.

- **Background load**: close browsers, IDEs, build watchers, and Docker. Anything sharing the CPU skews short-running benches.

- **Pinning (optional)**: pin to a fixed core if available:

  ```bash
  taskset -c 2 cargo bench -p laurus --bench distance_bench
  ```

  Or use the bundled wrapper that pins the bench process to one core via
  `taskset` and raises scheduling priority via `nice` in one step:

  ```bash
  ./scripts/bench-stable.sh --bench distance_bench
  ./scripts/bench-stable.sh --bench distance_bench -- --baseline main
  ```

  The wrapper is Linux-only. It does not touch the CPU governor or
  turbo state (those need root); set those up separately if you need
  the extra precision.

- **Repeat**: re-run twice and compare. Differences below ~5 % on a tuned machine are noise; differences above that on a shared workstation may also be noise. Do not over-interpret a single run.

If you cannot stabilise the environment, say so explicitly in the PR (e.g. "measured on a shared laptop, expect ~10 % noise") rather than presenting unstable numbers as authoritative.

## Make targets

The `Makefile` exposes the common entry points:

```bash
make bench             # cargo bench -p laurus
make bench-baseline    # cargo bench -p laurus -- --save-baseline main
make bench-compare     # cargo bench -p laurus -- --baseline main
```

For a single bench, pass `BENCH=name`:

```bash
make bench BENCH=distance_bench
make bench-baseline BENCH=distance_bench
make bench-compare BENCH=distance_bench
```

## PR description template

When a PR claims a measurable performance change, paste a table like the following into the description:

```markdown
## Performance

Environment: <CPU model>, governor=performance, turbo disabled, dedicated machine.

Baseline: `main` at <commit-sha>. After: this branch at <commit-sha>.

| Bench | Before | After | Δ | Verdict |
| --- | --- | --- | --- | --- |
| `distance_metrics/cosine` | 4.20 µs | 3.10 µs | -26 % | improved |
| `distance_metrics/euclidean` | 2.18 µs | 2.16 µs | -1 % | no change |

Reproduce: `cargo bench -p laurus --bench distance_bench -- --baseline main`
```

Always include the commit SHAs of the baseline and the after-state so the comparison is reproducible. State the environment explicitly even when running on a tuned machine.

## Adding a new benchmark

When adding a new bench file, follow the suite-wide hygiene rules from [`laurus/benches/common.rs`](https://github.com/mosuka/laurus/blob/main/laurus/benches/common.rs):

1. Use a deterministic seed via `common::DEFAULT_SEED` (or the `lcg_*` helpers). Never call `rand::rng()`.
2. Add a top-of-file `//!` doc comment listing scope, scenarios, run command, and filter examples.
3. Add a one-time sanity `assert!` outside the timed `b.iter` block so a regression that produces empty output cannot pass silently.
4. Pick `SAMPLE_SIZE_FAST` (default, for sub-50 ms operations) or `SAMPLE_SIZE_SLOW` (construction paths). Do not invent intermediate values.
5. Register the file in `laurus/Cargo.toml` with `[[bench]] name = "..." harness = false`. The crate sets `autobenches = false`, so files in `benches/` are not picked up automatically.

If your bench needs to share helpers across files, extend `benches/common.rs` rather than duplicating code.

## Retrieval-quality probes

Criterion measures speed; ranking quality is measured on BEIR datasets with [`laurus/examples/beir_multi_vector_probe.rs`](https://github.com/mosuka/laurus/blob/main/laurus/examples/beir_multi_vector_probe.rs). It reports nDCG@10 of ColBERT MaxSim rescoring for every [`MultiVector` storage kind](../concepts/indexing/vector_indexing.md#multivector-storage) and [token-pooling factor](../concepts/indexing/vector_indexing.md#token-pooling) it is configured with, along with each configuration's segment size.

```bash
./scripts/fetch-beir.sh scifact          # download and MD5-check the BEIR zip into .cache/beir/
cargo run --release --features embeddings-candle \
    --example beir_multi_vector_probe -- --dataset scifact
make beir-storage-eval BEIR_DATASET=nfcorpus   # both steps in one target
```

- The corpus is encoded once with `answerdotai/answerai-colbert-small-v1` (pinned revision) and the f32 token vectors are cached under `.cache/beir/<dataset>/`; later runs skip the encoding.
- Every storage kind ingests the same vectors into its own field of one engine, so the first stage is identical and only the on-disk element kind differs.
- Two conditions: an exhaustive MaxSim over the whole corpus (isolates the storage effect; the corpus must fit in `RescoreOptions::MAX_WINDOW_SIZE`), and a BM25 top-100 first stage rescored by MaxSim.
- nDCG@10 follows trec_eval's `ndcg_cut_10` as BEIR runs it. `--trec-run <dir>` writes TREC run files for an independent check with trec_eval or pytrec_eval.
- Differences from F32 come with a paired-bootstrap 95% confidence interval (10,000 resamples, fixed seed). The reported search times are informational only.

The probe's metric and cache code is unit-tested under `cargo test` (no download needed). The probe itself is not run in CI.

## Continuous integration

CI does not currently run a regression-detection bench job. Each perf-changing PR is expected to post baseline-vs-after numbers manually, captured under the recommended environment.

A future iteration may add a smoke-set bench job that fails on large regressions; this is tracked under the umbrella issue [#429](https://github.com/mosuka/laurus/issues/429).
