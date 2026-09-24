//! Criterion benchmarks for [`FacetCollector::collect_doc`] from
//! `lexical::search::features::facet`.
//!
//! Targets the audit issue #409 (replace `HashMap<FacetPath, _>` with
//! interned id-based counters). The hot path is `collect_doc` calling
//! `entry(facet_path.clone()).or_insert(0) += 1` per document, plus the
//! parent-walk for hierarchical facets.
//!
//! # Scope
//!
//! Three measurement scenarios:
//!
//! 1. **`bench_flat_single_field`** — one facet field with 50 distinct
//!    values. Establishes the dense-fast-path baseline.
//! 2. **`bench_multi_field`** — three independent facet fields each with
//!    50 values. Multiplies the per-doc work by 3.
//! 3. **`bench_hierarchical`** — one facet field whose values follow a
//!    `region/country/state/city` path layout (depth 4, 50 leaf paths).
//!    Each `collect_doc` call walks `parent()` four times, exercising the
//!    repeated-clone / repeated-hash-insert pattern that #409 will fix.
//! 4. **`bench_multi_valued`** — one facet field holding a three-element
//!    `TextArray` per document (Issue #1187). Each element expands to its
//!    own facet path, so the per-doc work is three flat increments plus the
//!    generation-stamp check that keeps counts per document.
//!
//! Each scenario sweeps `doc_count ∈ {1k, 10k, 100k}` so the gain at scale
//! is visible. The same 100k-document mock reader is reused across sweep
//! sizes; only the `0..n` doc-id range fed to the collector varies.
//!
//! # Mock reader
//!
//! `FacetCollector::collect_doc` only ever calls `reader.document(doc_id)`
//! on the [`LexicalIndexReader`] trait. The bench uses a minimal
//! `MockFacetReader` that owns a `Vec<Document>` indexed by `doc_id`; the
//! other trait methods are defaulted, return `None`, or `unimplemented!()`.
//! This avoids the cost of building a real `Engine` for every sweep size.
//!
//! # Sanity check
//!
//! For each scenario, before timing, the bench runs the collector once
//! over `0..n` and asserts that `total_facet_count >= n`. Every document
//! contributes at least one leaf increment, so the lower bound holds for
//! all three scenarios; for the hierarchical case the count is closer to
//! `5n` because each doc also bumps four ancestor paths.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench facet_bench
//! ```
//!
//! Filter by case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench facet_bench -- "flat_single/100000"
//! cargo bench --bench facet_bench -- "hierarchical"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench facet_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::any::Any;
use std::hint::black_box;
use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use laurus::lexical::core::field::FieldValue;
use laurus::lexical::index::structures::bkd_tree::BKDTree;
use laurus::lexical::reader::{FieldStats, LexicalIndexReader, PostingIterator, ReaderTermInfo};
use laurus::lexical::search::features::facet::{FacetCollector, FacetConfig};
use laurus::{Document, Result as LaurusResult};

/// Number of distinct facet values per field. Picked so the
/// `HashMap<FacetPath, _>` in `FacetCollector` has a non-trivial number of
/// keys (50 keys per field × 3 fields × 5 hierarchical levels = up to 250
/// distinct paths in the multi-field + hierarchical combined size).
const FACET_VALUES_PER_FIELD: usize = 50;

/// Maximum document count any sweep case requests. The mock reader is
/// built once at this size and reused across all bench cases by feeding
/// only `0..sweep_n` doc IDs to the collector.
const MAX_DOCS: usize = 100_000;

/// A minimal `LexicalIndexReader` that returns pre-built documents. Only
/// `document(doc_id)` is exercised by `FacetCollector::collect_doc`; the
/// rest of the trait is defaulted or stubbed.
#[derive(Debug)]
struct MockFacetReader {
    documents: Vec<Document>,
}

impl MockFacetReader {
    fn new(documents: Vec<Document>) -> Self {
        Self { documents }
    }
}

impl LexicalIndexReader for MockFacetReader {
    fn doc_count(&self) -> u64 {
        self.documents.len() as u64
    }

    fn max_doc(&self) -> u64 {
        self.documents.len() as u64
    }

    fn is_deleted(&self, _doc_id: u64) -> bool {
        false
    }

    fn document(&self, doc_id: u64) -> LaurusResult<Option<Document>> {
        Ok(self.documents.get(doc_id as usize).cloned())
    }

    fn term_info(&self, _field: &str, _term: &str) -> LaurusResult<Option<ReaderTermInfo>> {
        Ok(None)
    }

    fn postings(
        &self,
        _field: &str,
        _term: &str,
    ) -> LaurusResult<Option<Box<dyn PostingIterator>>> {
        Ok(None)
    }

    fn field_stats(&self, _field: &str) -> LaurusResult<Option<FieldStats>> {
        Ok(None)
    }

    fn close(&mut self) -> LaurusResult<()> {
        Ok(())
    }

    fn is_closed(&self) -> bool {
        false
    }

    fn get_bkd_tree(&self, _field: &str) -> LaurusResult<Option<Arc<dyn BKDTree>>> {
        Ok(None)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    // DocValues are available for every stored field (mirrors the real
    // writer, which stores every field into DocValues). This lets
    // `collect_doc` take the #597 fast path and read facet values directly
    // instead of decoding + cloning the whole document.
    fn has_doc_values(&self, _field: &str) -> bool {
        true
    }

    fn get_doc_value(&self, field: &str, doc_id: u64) -> LaurusResult<Option<FieldValue>> {
        Ok(self
            .documents
            .get(doc_id as usize)
            .and_then(|d| d.get(field).cloned()))
    }
}

/// Build `n` documents with a single flat facet field (`field_a`),
/// drawing values from a 50-value pool deterministically.
/// A non-facet stored-field payload (title + body) so each document is
/// "fat". The pre-#597 path decodes and clones *every* field via
/// `reader.document()`, while the #597 path reads only the facet field's
/// DocValue — this payload models the asymmetry between a whole-document
/// decode and a single-field DocValues read.
fn payload(i: usize) -> String {
    format!("doc {i}: {}", "lorem ipsum dolor sit amet ".repeat(10))
}

fn build_flat_documents(n: usize) -> Vec<Document> {
    (0..n)
        .map(|i| {
            let value = format!("value_{}", i % FACET_VALUES_PER_FIELD);
            Document::builder()
                .add_text("field_a", value)
                .add_text("title", format!("Title {i}"))
                .add_text("body", payload(i))
                .build()
        })
        .collect()
}

/// Build `n` documents with three facet fields, each with its own
/// 50-value pool. Values across fields are intentionally offset so the
/// three fields contribute to distinct `FacetPath` entries.
fn build_multi_field_documents(n: usize) -> Vec<Document> {
    (0..n)
        .map(|i| {
            let v_a = format!("a_{}", i % FACET_VALUES_PER_FIELD);
            let v_b = format!("b_{}", (i / 2) % FACET_VALUES_PER_FIELD);
            let v_c = format!("c_{}", (i / 3) % FACET_VALUES_PER_FIELD);
            Document::builder()
                .add_text("field_a", v_a)
                .add_text("field_b", v_b)
                .add_text("field_c", v_c)
                .add_text("title", format!("Title {i}"))
                .add_text("body", payload(i))
                .build()
        })
        .collect()
}

/// Build `n` documents with a single hierarchical facet field
/// (`hier_field`). Paths follow a `region/country/state/city` layout.
/// Each level has 5 distinct values, yielding 5 * 5 * 5 * 5 = 625
/// possible leaf paths; only the first 50 are used so the hierarchy
/// matches the flat / multi-field cardinality.
fn build_hierarchical_documents(n: usize) -> Vec<Document> {
    (0..n)
        .map(|i| {
            let leaf_idx = i % FACET_VALUES_PER_FIELD;
            // Distribute leaf_idx across four levels so each contributes.
            let region = leaf_idx % 5;
            let country = (leaf_idx / 5) % 5;
            let state = (leaf_idx / 25) % 5;
            let city = leaf_idx % 5;
            let path = format!("r{region}/c{country}/s{state}/v{city}");
            Document::builder()
                .add_text("hier_field", path)
                .add_text("title", format!("Title {i}"))
                .add_text("body", payload(i))
                .build()
        })
        .collect()
}

/// Build `n` documents whose facet field is a three-element `TextArray`
/// drawn from the same 50-value pool as `build_flat_documents` (Issue
/// #1187). The three elements of a document are distinct, so each expands
/// to its own facet path and the collector's generation stamp only has to
/// confirm — never refuse — an increment.
fn build_multi_valued_documents(n: usize) -> Vec<Document> {
    (0..n)
        .map(|i| {
            let tags: Vec<String> = (0..3)
                .map(|k| format!("value_{}", (i + k * 17) % FACET_VALUES_PER_FIELD))
                .collect();
            Document::builder()
                .add_text_array("field_a", tags)
                .add_text("title", format!("Title {i}"))
                .add_text("body", payload(i))
                .build()
        })
        .collect()
}

/// Materialise the doc list into a mock reader.
fn make_reader(documents: Vec<Document>) -> Arc<MockFacetReader> {
    Arc::new(MockFacetReader::new(documents))
}

/// Run the collector once over `0..n` and assert the sum of per-field
/// facet doc counts meets the lower bound for the scenario. Used as a
/// one-time sanity probe before each bench group.
///
/// `FacetResults::total_facet_count` only returns the number of distinct
/// `FacetPath` keys across all fields, not the sum of their doc counts.
/// We instead sum every `FacetCount::count` per field and assert
/// `>= n * fields.len()`. Hierarchical paths additionally contribute to
/// each ancestor and multi-valued arrays to each element, so the actual
/// sum is larger, but the lower bound is sufficient for catching
/// empty-result regressions. (It holds because every bench corpus gives
/// each document a facetable, non-empty value; a geo value or an empty
/// array would contribute nothing — Issue #1187.)
fn assert_collector_probe(fields: &[String], reader: &MockFacetReader, n: usize, label: &str) {
    let mut probe = FacetCollector::new(FacetConfig::default(), fields.to_vec());
    for doc_id in 0..n as u64 {
        probe
            .collect_doc(doc_id, reader)
            .expect("facet probe must not error");
    }
    let results = probe.finalize().expect("facet probe finalize must succeed");

    let total_count: u64 = fields
        .iter()
        .filter_map(|f| results.get_field_facets(f))
        .flat_map(|counts| counts.iter().map(|c| c.count))
        .sum();
    let lower_bound = (n as u64) * (fields.len() as u64);
    assert!(
        total_count >= lower_bound,
        "{label}: expected sum(facet_count) >= {lower_bound}, got {total_count} (each doc must contribute at least one increment per facet field)"
    );
}

fn bench_flat_single_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("facet/flat_single");
    let reader = make_reader(build_flat_documents(MAX_DOCS));
    let fields = vec!["field_a".to_string()];

    for &n in &[1000usize, 10_000, 100_000] {
        assert_collector_probe(&fields, &reader, n, "flat_single");

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || FacetCollector::new(FacetConfig::default(), fields.clone()),
                |mut collector| {
                    for doc_id in 0..n as u64 {
                        collector
                            .collect_doc(black_box(doc_id), reader.as_ref())
                            .unwrap();
                    }
                    black_box(collector);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_multi_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("facet/multi_field");
    let reader = make_reader(build_multi_field_documents(MAX_DOCS));
    let fields = vec![
        "field_a".to_string(),
        "field_b".to_string(),
        "field_c".to_string(),
    ];

    for &n in &[1000usize, 10_000, 100_000] {
        assert_collector_probe(&fields, &reader, n, "multi_field");

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || FacetCollector::new(FacetConfig::default(), fields.clone()),
                |mut collector| {
                    for doc_id in 0..n as u64 {
                        collector
                            .collect_doc(black_box(doc_id), reader.as_ref())
                            .unwrap();
                    }
                    black_box(collector);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_hierarchical(c: &mut Criterion) {
    let mut group = c.benchmark_group("facet/hierarchical");
    let reader = make_reader(build_hierarchical_documents(MAX_DOCS));
    let fields = vec!["hier_field".to_string()];

    for &n in &[1000usize, 10_000, 100_000] {
        assert_collector_probe(&fields, &reader, n, "hierarchical");

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || FacetCollector::new(FacetConfig::default(), fields.clone()),
                |mut collector| {
                    for doc_id in 0..n as u64 {
                        collector
                            .collect_doc(black_box(doc_id), reader.as_ref())
                            .unwrap();
                    }
                    black_box(collector);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_multi_valued(c: &mut Criterion) {
    let mut group = c.benchmark_group("facet/multi_valued");
    let reader = make_reader(build_multi_valued_documents(MAX_DOCS));
    let fields = vec!["field_a".to_string()];

    for &n in &[1000usize, 10_000, 100_000] {
        assert_collector_probe(&fields, &reader, n, "multi_valued");

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || FacetCollector::new(FacetConfig::default(), fields.clone()),
                |mut collector| {
                    for doc_id in 0..n as u64 {
                        collector
                            .collect_doc(black_box(doc_id), reader.as_ref())
                            .unwrap();
                    }
                    black_box(collector);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_flat_single_field,
    bench_multi_field,
    bench_hierarchical,
    bench_multi_valued,
);
criterion_main!(benches);
