//! Issue #1472: a lexical search counts its total exactly up to the
//! threshold `track_total_hits` chooses, on a corpus where block-max pruning
//! skips matches.
//!
//! The corpus puts high-scoring documents (short, the query term four times)
//! in the first posting block and many low-scoring ones (long, the term once)
//! after it, so that a top-10 search that counts nothing skips the later
//! blocks. Every test first checks that it does, against `count()`, which
//! visits every match, so that a count that only happens to be right cannot
//! pass. The searches go through `InvertedIndexSearcher::search`, the path
//! `LexicalStore::search` and the engine take.

use std::sync::Arc;

use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::index::inverted::reader::InvertedIndexReader;
use laurus::lexical::index::inverted::searcher::InvertedIndexSearcher;
use laurus::lexical::query::collector::{Collector, TopDocsCollector};
use laurus::lexical::query::{BooleanQuery, Query, TermQuery};
use laurus::lexical::{InvertedIndexConfig, LexicalSearchRequest};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{Document, TotalHits, TrackTotalHits};

const FIELD: &str = "body";
const LIMIT: usize = 10;
/// High-scoring documents at the start of each segment: one posting block.
const HIGH: u64 = 128;
/// Low-scoring documents after them in each segment.
const LOW: u64 = 1_500;

/// An index of `segments` segments, each with `HIGH` documents with `alpha`
/// four times in a short body, then `LOW` with `alpha` once in a long one.
/// Every third document also has `beta`. `seq` numbers the documents.
fn searcher(segments: u64) -> InvertedIndexSearcher {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = InvertedIndexConfig {
        analyzer: Arc::new(StandardAnalyzer::new().unwrap()),
        max_buffered_docs: 1_000_000,
        max_buffer_memory: usize::MAX,
        max_segments: 1_000,
        ..Default::default()
    };
    let index = InvertedIndex::create(storage, config).unwrap();
    let mut writer = index.writer().unwrap();
    let filler: String = (0..40).map(|i| format!(" filler{i}")).collect();
    let mut seq = 0i64;
    for _ in 0..segments {
        for i in 0..HIGH + LOW {
            let mut body = if i < HIGH {
                "alpha alpha alpha alpha".to_string()
            } else {
                format!("alpha{filler}")
            };
            if seq % 3 == 0 {
                body.push_str(" beta");
            }
            writer
                .add_document(
                    Document::builder()
                        .add_text(FIELD, body)
                        .add_integer("seq", seq)
                        .build(),
                )
                .unwrap();
            seq += 1;
        }
        writer.commit().unwrap();
    }
    drop(writer);
    let reader = index.reader().unwrap();
    let count = reader
        .as_any()
        .downcast_ref::<InvertedIndexReader>()
        .map(|r| r.segment_count())
        .unwrap();
    assert_eq!(count as u64, segments, "segments");
    InvertedIndexSearcher::from_arc(reader)
}

/// Builds a fresh query each time a search needs one.
type MakeQuery = dyn Fn() -> Box<dyn Query>;

fn term(t: &str) -> Box<dyn Query> {
    Box::new(TermQuery::new(FIELD, t))
}

/// `alpha OR beta`: Block-Max-WAND on one segment.
fn or_query() -> Box<dyn Query> {
    let mut q = BooleanQuery::new();
    q.add_should(term("alpha"));
    q.add_should(term("beta"));
    Box::new(q)
}

/// A `BooleanQuery` with a lone Must clause.
fn must_query() -> Box<dyn Query> {
    let mut q = BooleanQuery::new();
    q.add_must(term("alpha"));
    Box::new(q)
}

fn request(query: Box<dyn Query>, track: TrackTotalHits, min_score: f32) -> LexicalSearchRequest {
    LexicalSearchRequest::new(query)
        .limit(LIMIT)
        .min_score(min_score)
        .load_documents(false)
        .track_total_hits(track)
}

/// The total a search that counts up to `threshold` must report when
/// `exact` documents match.
fn expected(exact: u64, threshold: u64) -> TotalHits {
    if exact < threshold {
        TotalHits::exact(exact)
    } else {
        TotalHits::at_least(threshold)
    }
}

/// Check every threshold against `count()` for one query, sort and
/// `min_score`.
fn check(
    searcher: &InvertedIndexSearcher,
    label: &str,
    make: &MakeQuery,
    min_score: f32,
    sort_by_seq: bool,
) {
    let with_sort = |r: LexicalSearchRequest| {
        if sort_by_seq {
            r.sort_by_field_asc("seq")
        } else {
            r
        }
    };
    let exact = searcher
        .count(LexicalSearchRequest::new(make()).min_score(min_score))
        .unwrap();
    assert!(exact > 2 * LIMIT as u64, "{label}: {exact} matches");

    let hits_of = |track: TrackTotalHits| {
        let results = searcher
            .search(with_sort(request(make(), track, min_score)))
            .unwrap();
        let hits: Vec<(u64, u32)> = results
            .hits
            .iter()
            .map(|h| (h.doc_id, h.score.to_bits()))
            .collect();
        (hits, results.total_hits)
    };

    let (disabled_hits, disabled_total) = hits_of(TrackTotalHits::Disabled);
    assert_eq!(disabled_total, None, "{label}: no total asked for");

    let mut wrong = Vec::new();
    for threshold in [5, 50, exact, exact + 10] {
        let (hits, total) = hits_of(TrackTotalHits::UpTo(threshold));
        if total != Some(expected(exact, threshold)) {
            wrong.push(format!("UpTo({threshold}): {total:?}, {exact} matches"));
        }
        if hits != disabled_hits {
            wrong.push(format!(
                "UpTo({threshold}): other hits than without a total"
            ));
        }
    }
    let (hits, total) = hits_of(TrackTotalHits::Exact);
    if total != Some(TotalHits::exact(exact)) {
        wrong.push(format!("Exact: {total:?}, {exact} matches"));
    }
    if hits != disabled_hits {
        wrong.push("Exact: other hits than without a total".to_string());
    }
    assert!(wrong.is_empty(), "{label}:\n{}", wrong.join("\n"));
}

/// The precondition: a top-10 search that counts nothing skips matches, so
/// its raw count is below `count()`'s.
fn assert_prunes(searcher: &InvertedIndexSearcher, label: &str, make: &MakeQuery) {
    let exact = searcher.count(LexicalSearchRequest::new(make())).unwrap();
    let raw = searcher
        .search_with_collector(make(), TopDocsCollector::new(LIMIT))
        .unwrap()
        .total_hits();
    assert!(
        raw < exact,
        "{label}: counted {raw} of {exact} without pruning"
    );
}

/// A `min_score` between the scores of the high and the low documents,
/// far from both, so that rounding cannot move a document across it.
fn min_score_between_clusters(searcher: &InvertedIndexSearcher) -> f32 {
    let all = searcher
        .search(
            LexicalSearchRequest::new(term("alpha"))
                .limit(1_000_000)
                .load_documents(false),
        )
        .unwrap();
    let mut scores: Vec<f32> = all.hits.iter().map(|h| h.score).collect();
    scores.sort_by(f32::total_cmp);
    let (low_max, high_min) = scores
        .windows(2)
        .map(|w| (w[0], w[1]))
        .max_by(|a, b| (a.1 - a.0).total_cmp(&(b.1 - b.0)))
        .unwrap();
    // `alpha` is in every document, so its scores are small; the clusters
    // are told apart by their ratio.
    assert!(
        high_min > low_max * 1.5,
        "no clear gap: {low_max} {high_min}"
    );
    (low_max + high_min) / 2.0
}

fn check_all(segments: u64) {
    let searcher = searcher(segments);
    let queries: [(&str, &MakeQuery); 3] = [
        ("term", &|| term("alpha")),
        ("or", &or_query),
        ("must", &must_query),
    ];
    for (label, make) in queries {
        let label = format!("{segments} segment(s), {label}");
        assert_prunes(&searcher, &label, make);
        check(&searcher, &format!("{label}, score sort"), make, 0.0, false);
        check(&searcher, &format!("{label}, field sort"), make, 0.0, true);
    }
    let min_score = min_score_between_clusters(&searcher);
    check(
        &searcher,
        &format!("{segments} segment(s), term, min_score {min_score}"),
        &|| term("alpha"),
        min_score,
        false,
    );
}

#[test]
fn one_segment_counts_exactly_up_to_the_threshold() {
    check_all(1);
}

#[test]
fn several_segments_count_exactly_up_to_the_threshold_together() {
    check_all(3);
}

#[test]
fn an_empty_query_counts_zero_exactly_when_asked() {
    let searcher = searcher(1);
    let missing = || term("nowhere");
    let total = searcher
        .search(request(missing(), TrackTotalHits::UpTo(10), 0.0))
        .unwrap()
        .total_hits;
    assert_eq!(total, Some(TotalHits::exact(0)));
    let none = searcher
        .search(request(missing(), TrackTotalHits::Disabled, 0.0))
        .unwrap()
        .total_hits;
    assert_eq!(none, None);
}
