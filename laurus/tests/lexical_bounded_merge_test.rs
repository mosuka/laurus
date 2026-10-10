//! `LexicalStore::optimize_within_budget` correctness (Issue #1164).
//!
//! Reuses `lexical_merge_stress_test.rs`'s deterministic `Corpus` generator
//! and `segment_count` helper so content-preservation checks don't need a
//! second, separately-built index to compare against.

use std::collections::HashMap;
use std::sync::Arc;

use laurus::Document;
use laurus::lexical::NumericRangeQuery;
use laurus::lexical::index::config::InvertedIndexConfig;
use laurus::lexical::{
    LexicalIndexConfig, LexicalSearchRequest, LexicalStore, NumericType, TermQuery,
};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

/// Distinct words in the synthetic text vocabulary.
const VOCAB_SIZE: u64 = 2_000;

fn word(i: u64) -> String {
    format!("word{i}")
}

/// Same corpus generator as `lexical_merge_stress_test.rs` (duplicated: Rust
/// integration test binaries don't share a module tree, and this file's
/// scope is intentionally narrower -- a budget-focused companion, not a
/// shared test-support crate).
struct Corpus {
    term_docs: HashMap<String, Vec<u64>>,
    score: HashMap<u64, i64>,
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

    fn build_and_record(&mut self, doc_id: u64) -> Document {
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
}

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
/// `lexical_optimize_test.rs`/`lexical_merge_stress_test.rs`'s helper.
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

/// Builds `SEGMENTS` committed segments of `DOCS_PER_SEGMENT` documents
/// each, returning the storage, the store, and the recorded ground truth.
fn build_index(segments: u64, docs_per_segment: u64) -> (Arc<dyn Storage>, LexicalStore, Corpus) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage.clone(), LexicalIndexConfig::default()).unwrap();

    let mut corpus = Corpus::new();
    let mut doc_id = 0u64;
    for _ in 0..segments {
        for _ in 0..docs_per_segment {
            let doc = corpus.build_and_record(doc_id);
            store.upsert_document(doc_id, doc).unwrap();
            doc_id += 1;
        }
        store.commit().unwrap();
    }
    (storage, store, corpus)
}

/// A budget tiny enough that a single source segment's replayed buffer
/// always exceeds it (Issue #1164's rollover check only fires between
/// sources, never mid-replay, so this forces a rollover after every one).
const TINY_BUDGET: usize = 1;

/// A budget large enough that no realistic test corpus here ever reaches
/// it -- `usize::MAX` reproduces `optimize()` exactly.
const UNBOUNDED: usize = usize::MAX;

/// With an effectively-zero budget, a force-merge of `SEGMENTS` source
/// segments must roll over after every one of them, producing exactly
/// `SEGMENTS` output segments -- and must still preserve every document,
/// every term's posting list, and the numeric range query's results.
#[test]
fn tiny_budget_splits_into_one_segment_per_source_and_preserves_content() {
    const SEGMENTS: u64 = 6;
    const DOCS_PER_SEGMENT: u64 = 500;
    const TOTAL_DOCS: u64 = SEGMENTS * DOCS_PER_SEGMENT;

    let (storage, store, corpus) = build_index(SEGMENTS, DOCS_PER_SEGMENT);
    assert_eq!(segment_count(&storage), SEGMENTS as usize);

    store.optimize_within_budget(TINY_BUDGET).unwrap();

    assert_eq!(
        segment_count(&storage),
        SEGMENTS as usize,
        "a budget smaller than any single source segment's replayed buffer \
         must roll over after every source, producing one output segment \
         per source"
    );
    let leftover_sources = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.starts_with("segment_"))
        .count();
    assert_eq!(leftover_sources, 0, "source segment files must be deleted");

    // No documents lost or duplicated across bucket boundaries.
    assert_eq!(
        term_hits(&store, "kind", "doc").len(),
        TOTAL_DOCS as usize,
        "every document must survive a bounded merge exactly once"
    );

    let mut mismatches: Vec<String> = Vec::new();
    for i in 0..VOCAB_SIZE {
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
        "term posting mismatches after bounded merge: {:?}",
        &mismatches[..mismatches.len().min(10)]
    );

    let (lo, hi) = (-100i64, 100i64);
    assert_eq!(
        numeric_range_hits(&store, "score", NumericType::Integer, lo as f64, hi as f64),
        corpus.expected_score_range(lo, hi),
        "numeric range results must survive a bounded merge across \
         multiple output segments"
    );
}

/// `usize::MAX` must reproduce `optimize()`'s classic single-segment result
/// exactly -- the pre-#1164 behavior every merge path still defaults to.
#[test]
fn unbounded_budget_matches_plain_optimize() {
    const SEGMENTS: u64 = 4;
    const DOCS_PER_SEGMENT: u64 = 200;

    let (storage, store, _corpus) = build_index(SEGMENTS, DOCS_PER_SEGMENT);
    store.optimize_within_budget(UNBOUNDED).unwrap();

    assert_eq!(
        segment_count(&storage),
        1,
        "usize::MAX must always collapse into exactly one output segment"
    );
}

/// A single source segment is a no-op, exactly like `optimize()`.
#[test]
fn bounded_merge_is_a_noop_for_a_single_segment() {
    let (storage, store, _corpus) = build_index(1, 10);
    store.optimize_within_budget(TINY_BUDGET).unwrap();
    assert_eq!(segment_count(&storage), 1, "nothing to compact");
}

/// Issue #1164's rollover rebuilds the writer on every bucket boundary,
/// carrying its config (and with it the pinned `field_term_positions`
/// state) forward. Without that carry-over, only the FIRST bucket would
/// detect a field's positions state from its source segments -- every
/// later bucket would silently fall back to the index-wide default and
/// phrase search on it would degrade. Uses two fields with opposite
/// `term_vectors` settings, and a tiny budget that forces a rollover
/// between every source, so a dropped pin would surface as a mismatch on
/// at least one later bucket.
#[test]
fn bounded_merge_preserves_positions_pinning_across_bucket_boundaries() {
    use laurus::lexical::core::field::{FieldOption, TextOption};
    use laurus::lexical::query::phrase::PhraseQuery;

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "phrase_field",
            FieldOption::Text(TextOption {
                term_vectors: true,
                ..Default::default()
            }),
        )
        .build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();

    const SEGMENTS: u64 = 6;
    const DOCS_PER_SEGMENT: u64 = 20;
    let mut doc_id = 0u64;
    for _ in 0..SEGMENTS {
        for _ in 0..DOCS_PER_SEGMENT {
            let doc = Document::builder()
                .add_text("phrase_field", "the quick brown fox")
                .build();
            store.upsert_document(doc_id, doc).unwrap();
            doc_id += 1;
        }
        store.commit().unwrap();
    }
    let total_docs = doc_id as usize;

    store.optimize_within_budget(TINY_BUDGET).unwrap();
    assert_eq!(
        segment_count(&storage),
        SEGMENTS as usize,
        "sanity: the tiny budget must have produced several output segments"
    );

    // A phrase query needs positions. If any output bucket lost its pinned
    // `term_vectors: true` state, that bucket's documents would be
    // unsearchable by phrase (or, depending on the fallback, the query
    // itself could error) -- so finding every document here confirms the
    // pin survived every rollover.
    let hits = store
        .search(
            LexicalSearchRequest::new(Box::new(PhraseQuery::new(
                "phrase_field",
                vec!["quick".to_string(), "brown".to_string()],
            )))
            .limit(MAX_HITS),
        )
        .unwrap()
        .hits;
    assert_eq!(
        hits.len(),
        total_docs,
        "phrase search must find every document across every output bucket, \
         confirming the term_vectors pin survived each rollover"
    );
}

/// A later bucket's `.dv` must hold only its own documents' DocValues.
///
/// Found during Issue #1164's design review: the merge writer used to keep
/// a writer-wide DocValues buffer that `flush_buffered_to_segment` never
/// reset, so a writer reused across a rollover re-accumulated every earlier
/// bucket's entries on top of its own. No individual value was corrupted
/// (entries were keyed by doc_id); bucket K's `.dv` just held K times the
/// entries it needed, and that memory was never freed — the unbounded
/// growth #1164 exists to bound. Since #1168 the writer builds `.dv` from
/// the buffer at flush time and holds no DocValues state at all, so the
/// leak can no longer arise; this test keeps guarding the invariant.
///
/// With `TINY_BUDGET` forcing one source segment per bucket (so every
/// bucket holds the same `DOCS_PER_SEGMENT` live documents), every bucket's
/// `.dv` file is about the same size; a leak would make bucket K's grow
/// roughly linearly with K. Asserts the last bucket's `.dv` is not
/// meaningfully larger than the first's.
#[test]
fn bounded_merge_does_not_accumulate_doc_values_across_bucket_boundaries() {
    const SEGMENTS: u64 = 6;
    const DOCS_PER_SEGMENT: u64 = 300;

    // Loose files, not a compound `.cfs` container: this test inspects each
    // output bucket's standalone `.dv` file size directly.
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let inverted_config = InvertedIndexConfig {
        use_compound: false,
        ..Default::default()
    };
    let store = LexicalStore::new(
        storage.clone(),
        LexicalIndexConfig::Inverted(inverted_config),
    )
    .unwrap();
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

    store.optimize_within_budget(TINY_BUDGET).unwrap();
    assert_eq!(
        segment_count(&storage),
        SEGMENTS as usize,
        "sanity: the tiny budget must have produced one bucket per source, \
         each holding the same number of live documents"
    );

    // `merged_<generation>.dv`, ordered by the numeric generation (not
    // lexicographic: "merged_10" must sort after "merged_9").
    let mut dv_sizes: Vec<(u64, u64)> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter_map(|f| {
            let rest = f.strip_prefix("merged_")?.strip_suffix(".dv")?;
            let generation: u64 = rest.parse().ok()?;
            let size = storage.metadata(&f).unwrap().size;
            Some((generation, size))
        })
        .collect();
    dv_sizes.sort_unstable_by_key(|&(generation, _)| generation);
    assert_eq!(
        dv_sizes.len(),
        SEGMENTS as usize,
        "sanity: one .dv file per output bucket"
    );

    let first_size = dv_sizes.first().unwrap().1;
    let last_size = dv_sizes.last().unwrap().1;
    assert!(
        first_size > 0,
        "sanity: a bucket with live documents must have a non-empty .dv file"
    );
    assert!(
        last_size <= first_size * 2,
        "the last bucket's .dv file ({last_size} bytes) must stay roughly \
         the same size as the first's ({first_size} bytes), not grow with \
         the number of buckets already flushed -- sizes by generation: \
         {dv_sizes:?}"
    );
}
