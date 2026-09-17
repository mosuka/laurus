//! Criterion benchmarks for the document fetch + sparse-field selection
//! pattern that once lived in the (now-removed, #1134) `result_processor`
//! module as `retrieve_document_fields`.
//!
//! Targets the audit issue #410 (per-field document fetch). That code
//! called `reader.document(doc_id)`, which deserializes every field, then
//! filtered down to the requested subset via `should_retrieve_field`. The
//! unrequested fields are full waste on a wide-schema with sparse field
//! selection.
//!
//! # Scope
//!
//! Three measurement scenarios:
//!
//! 1. **`bench_wide_narrow`** — 50-field documents, top-K = 10 fetches per
//!    iteration. Sweep `fields_selected ∈ {1, 5, 50}`. Shows that the
//!    per-hit cost is dominated by the full-document deserialize, not by
//!    the selected-field count.
//! 2. **`bench_narrow_baseline`** — 5-field documents, top-K = 10, all 5
//!    fields selected. Sanity case where the #410 fix should be near-
//!    neutral (no unrequested fields to skip).
//! 3. **`bench_field_size_variance`** — 5-field documents, top-K = 10,
//!    1 field selected. Sweep per-field value size ∈ {100 B, 100 KB}.
//!    Demonstrates that skip-decode helps more when the unrequested
//!    fields are large.
//!
//! # Mock reader
//!
//! `LexicalIndexReader::document` is the only trait method exercised by
//! the fetch path. The bench uses a minimal `MockStoreReader` that owns
//! `Vec<Document>`; the other nine trait methods are defaulted or
//! stubbed (`Ok(None)`). This is the same pattern used by
//! `facet_bench.rs`.
//!
//! `Document` derives `Clone` and contains a `HashMap<String, DataValue>`,
//! so the mock's `document(doc_id)` returns a deep-cloned `Document` —
//! the per-field-count cost shape mirrors a real deserialize from
//! storage.
//!
//! # `fetch_and_filter`
//!
//! Reproduces the `retrieve_document_fields` body in the bench so we
//! measure the full pipeline (fetch + iterate fields + filter +
//! materialize `HashMap<String, String>`) without coupling to a private
//! method.
//!
//! # Sanity check
//!
//! Per the #420 acceptance criteria, the probe asserts that every
//! requested field is present in the result and every non-requested field
//! is absent.
//!
//! # Run
//!
//! ```sh
//! cargo bench --bench store_fetch_bench
//! ```
//!
//! Filter by case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench store_fetch_bench -- "wide_narrow/selected_1$"
//! cargo bench --bench store_fetch_bench -- "field_size/large"
//! ```
//!
//! Compile-only smoke check:
//!
//! ```sh
//! cargo bench --bench store_fetch_bench --no-run
//! ```
//!
//! See `benches/common.rs` for the suite-wide hygiene rules.

mod common;

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use laurus::lexical::index::structures::bkd_tree::BKDTree;
use laurus::lexical::reader::{FieldStats, LexicalIndexReader, PostingIterator, ReaderTermInfo};
use laurus::{DataValue, Document, Result as LaurusResult};

/// Number of documents kept in the mock reader. The bench fetches the
/// first `TOP_K` of them per iteration; building 100 lets us tag each
/// doc deterministically without overshooting useful range.
const N_DOCS: usize = 100;

/// Top-K = number of `fetch_and_filter` calls per timed iteration. Picks
/// 10 to mirror typical search result sizes.
const TOP_K: u64 = 10;

/// A minimal `LexicalIndexReader` that returns pre-built documents. Only
/// `document(doc_id)` is exercised by the fetch path; the rest of the
/// trait is defaulted or stubbed.
#[derive(Debug)]
struct MockStoreReader {
    documents: Vec<Document>,
}

impl MockStoreReader {
    fn new(documents: Vec<Document>) -> Self {
        Self { documents }
    }
}

impl LexicalIndexReader for MockStoreReader {
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

    /// Override the default trait impl so the bench measures the
    /// real-reader behaviour: only the requested fields are cloned out
    /// of the cached `Document`, not the whole map (#410). Without
    /// this override the default falls through to `document()` and
    /// post-filters, defeating the comparison.
    fn document_fields(
        &self,
        doc_id: u64,
        field_names: &[&str],
    ) -> LaurusResult<Option<HashMap<String, DataValue>>> {
        if let Some(doc) = self.documents.get(doc_id as usize) {
            let mut out = HashMap::with_capacity(field_names.len());
            for &name in field_names {
                if let Some(value) = doc.fields.get(name) {
                    out.insert(name.to_string(), value.clone());
                }
            }
            return Ok(Some(out));
        }
        Ok(None)
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
}

/// Reproduce the body of the removed `result_processor::retrieve_document_fields`:
/// fetch the document by ID, iterate every stored field, and pick the
/// subset matching `selected`. Returns a `HashMap<String, String>` whose
/// values are stringified `DataValue`s.
fn fetch_and_filter(
    reader: &dyn LexicalIndexReader,
    doc_id: u64,
    selected: &HashSet<&str>,
) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Ok(Some(doc)) = reader.document(doc_id) {
        for (name, value) in &doc.fields {
            if selected.contains(name.as_str()) {
                out.insert(name.clone(), data_value_to_string(value));
            }
        }
    }
    out
}

/// `fetch_and_filter`'s subset-aware counterpart (#410). Calls
/// [`LexicalIndexReader::document_fields`] so the reader clones only
/// the requested field values out of the cached `Document`. Returns
/// the same shape as `fetch_and_filter` for direct comparison.
fn fetch_subset(
    reader: &dyn LexicalIndexReader,
    doc_id: u64,
    selected: &[&str],
) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Ok(Some(fields)) = reader.document_fields(doc_id, selected) {
        for (name, value) in fields {
            out.insert(name, data_value_to_string(&value));
        }
    }
    out
}

/// Convert a `DataValue` to a stringified form. Mirrors the shape of the
/// removed `result_processor::field_value_to_string` — a String already in
/// Text form is cloned, other types fall back to `Debug`.
fn data_value_to_string(value: &DataValue) -> String {
    match value {
        DataValue::Text(s) => s.clone(),
        other => format!("{other:?}"),
    }
}

/// Generate `n_docs` documents, each with `n_fields` text fields named
/// `field_0`, `field_1`, …, `field_{n_fields-1}`. Every field's value is
/// a deterministic ASCII string of `value_bytes` length.
fn build_documents(n_docs: usize, n_fields: usize, value_bytes: usize) -> Vec<Document> {
    // The same value pattern in every doc is acceptable here — the bench
    // measures fetch / clone / filter cost, not value-content variance.
    let payload: String = (0..value_bytes)
        .map(|i| (b'a' + (i % 26) as u8) as char)
        .collect();

    (0..n_docs)
        .map(|_| {
            let mut builder = Document::builder();
            for f in 0..n_fields {
                builder = builder.add_text(format!("field_{f}"), payload.clone());
            }
            builder.build()
        })
        .collect()
}

/// One-time correctness probe. Fetches doc 0, filters by `selected`,
/// and asserts that every selected field name is present in the result
/// and every other field name is absent. Run before each bench group.
fn assert_selection_probe(
    reader: &dyn LexicalIndexReader,
    n_fields: usize,
    selected: &HashSet<&str>,
    label: &str,
) {
    let result = fetch_and_filter(reader, 0, selected);
    for &name in selected.iter() {
        assert!(
            result.contains_key(name),
            "{label}: selected field '{name}' must be present in fetch result"
        );
    }
    for f in 0..n_fields {
        let name = format!("field_{f}");
        if selected.contains(name.as_str()) {
            continue;
        }
        assert!(
            !result.contains_key(&name),
            "{label}: non-selected field '{name}' must be absent from fetch result"
        );
    }
}

fn bench_wide_narrow(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_fetch/wide_narrow");

    const N_FIELDS: usize = 50;
    const VALUE_BYTES: usize = 50;

    let reader = MockStoreReader::new(build_documents(N_DOCS, N_FIELDS, VALUE_BYTES));
    let all_field_names: Vec<String> = (0..N_FIELDS).map(|f| format!("field_{f}")).collect();

    for &n_selected in &[1usize, 5, 50] {
        let selected_owned: Vec<&str> = all_field_names
            .iter()
            .take(n_selected)
            .map(String::as_str)
            .collect();
        let selected: HashSet<&str> = selected_owned.iter().copied().collect();

        assert_selection_probe(&reader, N_FIELDS, &selected, "wide_narrow");

        group.throughput(Throughput::Elements(TOP_K));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("selected_{n_selected}")),
            &(),
            |b, _| {
                b.iter(|| {
                    for doc_id in 0..TOP_K {
                        let result = fetch_and_filter(
                            black_box(&reader),
                            black_box(doc_id),
                            black_box(&selected),
                        );
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_narrow_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_fetch/narrow_baseline");

    const N_FIELDS: usize = 5;
    const VALUE_BYTES: usize = 50;

    let reader = MockStoreReader::new(build_documents(N_DOCS, N_FIELDS, VALUE_BYTES));
    let all_field_names: Vec<String> = (0..N_FIELDS).map(|f| format!("field_{f}")).collect();

    let selected_owned: Vec<&str> = all_field_names.iter().map(String::as_str).collect();
    let selected: HashSet<&str> = selected_owned.iter().copied().collect();

    assert_selection_probe(&reader, N_FIELDS, &selected, "narrow_baseline");

    group.throughput(Throughput::Elements(TOP_K));
    group.bench_function("all_5", |b| {
        b.iter(|| {
            for doc_id in 0..TOP_K {
                let result =
                    fetch_and_filter(black_box(&reader), black_box(doc_id), black_box(&selected));
                black_box(result);
            }
        });
    });

    group.finish();
}

fn bench_field_size_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_fetch/field_size");

    const N_FIELDS: usize = 5;

    for &(label, value_bytes) in &[("small_100B", 100usize), ("large_100KB", 100 * 1024)] {
        let reader = MockStoreReader::new(build_documents(N_DOCS, N_FIELDS, value_bytes));
        let all_field_names: Vec<String> = (0..N_FIELDS).map(|f| format!("field_{f}")).collect();

        // Select only 1 field — the unrequested 4 fields are the wasted
        // deserialize work that #410 will skip.
        let selected_owned: Vec<&str> =
            all_field_names.iter().take(1).map(String::as_str).collect();
        let selected: HashSet<&str> = selected_owned.iter().copied().collect();

        assert_selection_probe(&reader, N_FIELDS, &selected, label);

        group.throughput(Throughput::Elements(TOP_K));
        group.bench_with_input(BenchmarkId::from_parameter(label), &(), |b, _| {
            b.iter(|| {
                for doc_id in 0..TOP_K {
                    let result = fetch_and_filter(
                        black_box(&reader),
                        black_box(doc_id),
                        black_box(&selected),
                    );
                    black_box(result);
                }
            });
        });
    }

    group.finish();
}

/// Side-by-side comparison of `fetch_and_filter` (full document clone +
/// post-filter) and `fetch_subset` (per-field clone via
/// [`LexicalIndexReader::document_fields`]) at the wide-schema +
/// narrow-selection workload that #410 targets. Two bench cases per
/// `n_selected` value so the criterion id makes the comparison
/// obvious from the output.
fn bench_wide_narrow_subset(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_fetch/wide_narrow_subset");

    const N_FIELDS: usize = 50;
    const VALUE_BYTES: usize = 50;

    let reader = MockStoreReader::new(build_documents(N_DOCS, N_FIELDS, VALUE_BYTES));
    let all_field_names: Vec<String> = (0..N_FIELDS).map(|f| format!("field_{f}")).collect();

    for &n_selected in &[1usize, 5, 50] {
        let selected_owned: Vec<&str> = all_field_names
            .iter()
            .take(n_selected)
            .map(String::as_str)
            .collect();
        let selected_set: HashSet<&str> = selected_owned.iter().copied().collect();

        // Full-doc-clone + filter (pre-#410 path).
        group.throughput(Throughput::Elements(TOP_K));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("full_clone/selected_{n_selected}")),
            &(),
            |b, _| {
                b.iter(|| {
                    for doc_id in 0..TOP_K {
                        let result = fetch_and_filter(
                            black_box(&reader),
                            black_box(doc_id),
                            black_box(&selected_set),
                        );
                        black_box(result);
                    }
                });
            },
        );

        // Per-field clone via `document_fields` (#410 path).
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("subset/selected_{n_selected}")),
            &(),
            |b, _| {
                b.iter(|| {
                    for doc_id in 0..TOP_K {
                        let result = fetch_subset(
                            black_box(&reader),
                            black_box(doc_id),
                            black_box(&selected_owned),
                        );
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_wide_narrow,
    bench_wide_narrow_subset,
    bench_narrow_baseline,
    bench_field_size_variance,
);
criterion_main!(benches);
