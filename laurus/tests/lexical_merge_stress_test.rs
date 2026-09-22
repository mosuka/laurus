//! Merge-path correctness at a realistic scale (Issue #1167).
//!
//! No test anywhere in the repo previously exercised `MergeEngine::perform_merge`
//! beyond a handful of documents — every `merge_engine.rs` unit test uses 1-3
//! docs, and the small `tests/lexical_*_test.rs` files are similarly tiny.
//! This is a prerequisite for safely validating #1163 (landed), #1164, #1165,
//! and #1168, none of which can be meaningfully regression-tested for their
//! actual memory/performance claims without a merge-path harness that
//! exercises realistic document/point/term counts.
//!
//! Uses only `LexicalStore`'s public API (matching `lexical_optimize_test.rs`'s
//! convention), not `MergeEngine`'s internals. The corpus is generated
//! deterministically so expected term hit-counts and numeric-range matches
//! are computed arithmetically during generation, rather than by building a
//! second, unmerged index purely for comparison.

use std::collections::HashMap;
use std::sync::Arc;

use laurus::Document;
use laurus::lexical::NumericRangeQuery;
use laurus::lexical::{
    LexicalIndexConfig, LexicalSearchRequest, LexicalStore, NumericType, TermQuery,
};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

/// Distinct words in the synthetic text vocabulary.
const VOCAB_SIZE: u64 = 2_000;

/// Deterministic word for vocabulary index `i` (`0..VOCAB_SIZE`).
fn word(i: u64) -> String {
    format!("word{i}")
}

/// Deterministic corpus generator shared by both tests below. Builds one
/// document per `doc_id` in `0..doc_count`, tracking ground truth (which
/// terms it carries, its numeric field values) as it goes, so later
/// assertions never need a second, separately-built index to compare
/// against.
struct Corpus {
    /// term -> set of doc_ids that carry it, exactly as generated.
    term_docs: HashMap<String, Vec<u64>>,
    /// doc_id -> single-valued "score" field.
    score: HashMap<u64, i64>,
    /// doc_id -> multi-valued "tags_num" field.
    tags_num: HashMap<u64, Vec<i64>>,
}

impl Corpus {
    fn new() -> Self {
        Corpus {
            term_docs: HashMap::new(),
            score: HashMap::new(),
            tags_num: HashMap::new(),
        }
    }

    /// Build document `doc_id`'s `Document` and record its ground truth.
    /// Every document also carries `kind: "doc"`, giving a single query
    /// that reports the total live document count.
    fn build_and_record(&mut self, doc_id: u64) -> Document {
        // 4 words per document, spread across the vocabulary via a fixed
        // stride+offset so the same doc_id always yields the same words,
        // and different doc_ids overlap enough to give every vocabulary
        // word a nontrivial, checkable posting list.
        let mut words = Vec::with_capacity(4);
        for k in 0..4u64 {
            let idx = (doc_id.wrapping_mul(7).wrapping_add(k.wrapping_mul(131))) % VOCAB_SIZE;
            let w = word(idx);
            self.term_docs.entry(w.clone()).or_default().push(doc_id);
            words.push(w);
        }
        let body = words.join(" ");

        let score = ((doc_id.wrapping_mul(37) % 1000) as i64) - 500; // [-500, 499]
        self.score.insert(doc_id, score);

        let tags_num: Vec<i64> = (0..3u64)
            .map(|k| ((doc_id.wrapping_mul(11).wrapping_add(k.wrapping_mul(53))) % 2000) as i64)
            .collect();
        self.tags_num.insert(doc_id, tags_num.clone());

        Document::builder()
            .add_text("kind", "doc")
            .add_text("body", body)
            .add_integer("score", score)
            .add_int64_array("tags_num", tags_num)
            .build()
    }

    /// Expected doc_ids matching an inclusive `[lo, hi]` range on `score`.
    fn expected_score_range(&self, lo: i64, hi: i64) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .score
            .iter()
            .filter(|&(_, &v)| v >= lo && v <= hi)
            .map(|(&id, _)| id)
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Expected doc_ids where ANY `tags_num` value falls in an inclusive
    /// `[lo, hi]` range (Lucene-style multi-valued semantics, #758).
    fn expected_tags_range(&self, lo: i64, hi: i64) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .tags_num
            .iter()
            .filter(|(_, values)| values.iter().any(|&v| v >= lo && v <= hi))
            .map(|(&id, _)| id)
            .collect();
        ids.sort_unstable();
        ids
    }
}

/// Large enough to return every match at this test's corpus scale --
/// `LexicalSearchParams::default().limit` is 10, which would otherwise
/// silently truncate every "how many total/matching docs" assertion below.
const MAX_HITS: usize = 100_000;

fn term_hits(store: &LexicalStore, field: &str, term: &str) -> Vec<u64> {
    let mut ids: Vec<u64> = store
        .search(LexicalSearchRequest::new(Box::new(TermQuery::new(field, term))).limit(MAX_HITS))
        .unwrap()
        .hits
        .into_iter()
        .map(|h| h.doc_id)
        .collect();
    ids.sort_unstable();
    ids
}

fn numeric_range_hits(
    store: &LexicalStore,
    field: &str,
    numeric_type: NumericType,
    lo: f64,
    hi: f64,
) -> Vec<u64> {
    let query = NumericRangeQuery::new(field, numeric_type, Some(lo), Some(hi), true, true);
    let mut ids: Vec<u64> = store
        .search(LexicalSearchRequest::new(Box::new(query)).limit(MAX_HITS))
        .unwrap()
        .hits
        .into_iter()
        .map(|h| h.doc_id)
        .collect();
    ids.sort_unstable();
    ids
}

/// Count discovered segments via the manifest (#1024), matching
/// `lexical_optimize_test.rs`'s helper.
fn segment_count(storage: &Arc<dyn Storage>) -> usize {
    let mut input = storage.open_input("segments.json").unwrap();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut input, &mut bytes).unwrap();
    let payload: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => {
            let mut len: u64 = 0;
            let mut shift = 0;
            let mut cursor = 0usize;
            loop {
                let byte = bytes[cursor];
                cursor += 1;
                len |= u64::from(byte & 0x7F) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            serde_json::from_slice(&bytes[cursor..cursor + len as usize]).unwrap()
        }
    };
    payload["segments"].as_array().unwrap().len()
}

/// Force-merging a 30,000-document, 6-segment index must preserve every
/// document, every term's posting list, and both numeric fields' range-query
/// semantics (including the multi-valued field's "any value matches"
/// contract) exactly.
#[test]
fn optimize_preserves_correctness_at_realistic_scale() {
    const SEGMENTS: u64 = 6;
    const DOCS_PER_SEGMENT: u64 = 5_000;
    const TOTAL_DOCS: u64 = SEGMENTS * DOCS_PER_SEGMENT;

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    let mut corpus = Corpus::new();
    let mut doc_id = 0u64;
    for _ in 0..SEGMENTS {
        for _ in 0..DOCS_PER_SEGMENT {
            let doc = corpus.build_and_record(doc_id);
            store.upsert_document(doc_id, doc).unwrap();
            doc_id += 1;
        }
        store.commit().unwrap();
    }
    assert_eq!(
        segment_count(&storage),
        SEGMENTS as usize,
        "sanity: one segment per commit before merging"
    );
    assert_eq!(
        term_hits(&store, "kind", "doc").len(),
        TOTAL_DOCS as usize,
        "sanity: every document indexed before merging"
    );

    store.optimize().unwrap();

    assert_eq!(
        segment_count(&storage),
        1,
        "optimize must force-merge every segment into one"
    );
    let leftover_sources = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.starts_with("segment_"))
        .count();
    assert_eq!(leftover_sources, 0, "source segment files must be deleted");

    // Total document count survives the merge.
    let mut all_docs = term_hits(&store, "kind", "doc");
    all_docs.sort_unstable();
    assert_eq!(
        all_docs.len(),
        TOTAL_DOCS as usize,
        "document count must be unchanged by the merge"
    );

    // Every vocabulary word's posting list survives the merge exactly.
    let mut mismatches: Vec<String> = Vec::new();
    for i in 0..VOCAB_SIZE {
        let w = word(i);
        let expected = corpus.term_docs.get(&w).cloned().unwrap_or_default();
        let mut expected_sorted = expected.clone();
        expected_sorted.sort_unstable();
        let actual = term_hits(&store, "body", &w);
        if actual != expected_sorted {
            mismatches.push(format!(
                "{w}: expected {} hits, got {} hits",
                expected_sorted.len(),
                actual.len()
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "term posting mismatches after merge (showing up to 10): {:?}",
        &mismatches[..mismatches.len().min(10)]
    );

    // Single-valued numeric range query (BKD, single value per doc).
    let (lo, hi) = (-100i64, 100i64);
    assert_eq!(
        numeric_range_hits(&store, "score", NumericType::Integer, lo as f64, hi as f64),
        corpus.expected_score_range(lo, hi),
        "single-valued numeric range results must survive the merge"
    );

    // Multi-valued numeric range query (BKD, several points per doc,
    // Lucene-style "any value matches" -- the merge path's multi-valued
    // handling, #758).
    let (lo, hi) = (500i64, 600i64);
    assert_eq!(
        numeric_range_hits(
            &store,
            "tags_num",
            NumericType::Integer,
            lo as f64,
            hi as f64
        ),
        corpus.expected_tags_range(lo, hi),
        "multi-valued numeric range results must survive the merge"
    );

    // Re-optimizing a single-segment index is a no-op, exactly as at small
    // scale (`lexical_optimize_test.rs`).
    store.optimize().unwrap();
    assert_eq!(segment_count(&storage), 1, "re-optimize is a no-op");
    assert_eq!(term_hits(&store, "kind", "doc").len(), TOTAL_DOCS as usize);
}

/// #1144's investigation found `auto_merge()` (triggered inside `commit()`
/// once `max_segments` is exceeded) reaches the same unbounded
/// `perform_merge` code path as explicit `optimize()`. This confirms that
/// path fires on its own -- not just when explicitly invoked -- and
/// preserves correctness across enough small commits to force it to run
/// more than once.
#[test]
fn auto_merge_fires_without_explicit_optimize_and_preserves_correctness() {
    const COMMITS: u64 = 20;
    const DOCS_PER_COMMIT: u64 = 25;
    const TOTAL_DOCS: u64 = COMMITS * DOCS_PER_COMMIT;

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .max_segments(3)
        .merge_factor(2)
        .build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    let mut corpus = Corpus::new();
    let mut doc_id = 0u64;
    for _ in 0..COMMITS {
        for _ in 0..DOCS_PER_COMMIT {
            let doc = corpus.build_and_record(doc_id);
            store.upsert_document(doc_id, doc).unwrap();
            doc_id += 1;
        }
        store.commit().unwrap();
    }

    assert!(
        segment_count(&storage) < COMMITS as usize,
        "auto_merge must have collapsed segments below one-per-commit \
         (got {} segments across {COMMITS} commits)",
        segment_count(&storage)
    );

    assert_eq!(
        term_hits(&store, "kind", "doc").len(),
        TOTAL_DOCS as usize,
        "auto-merged documents must all still be findable"
    );

    // Spot-check a handful of terms across the whole vocabulary range.
    let mut mismatches: Vec<String> = Vec::new();
    for i in (0..VOCAB_SIZE).step_by(97) {
        let w = word(i);
        let mut expected = corpus.term_docs.get(&w).cloned().unwrap_or_default();
        expected.sort_unstable();
        let actual = term_hits(&store, "body", &w);
        if actual != expected {
            mismatches.push(w);
        }
    }
    assert!(
        mismatches.is_empty(),
        "term posting mismatches after auto-merge: {mismatches:?}"
    );
}
