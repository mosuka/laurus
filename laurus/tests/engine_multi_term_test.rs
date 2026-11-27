//! Issue #1740: a prefix, wildcard or regexp query matches every document that
//! has a matching term, however many terms match. It used to find the
//! documents of 50 of them, the least frequent ones, without an error, and as
//! a `MUST_NOT` clause it excluded only those.
//!
//! They score every match with their boost, so every hit ties, and tied hits
//! come in the order of their document ids: the pages of a search used to
//! overlap where scores tied.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use laurus::lexical::Query;
use laurus::lexical::query::boolean::BooleanQueryBuilder;
use laurus::lexical::query::fuzzy::FuzzyQuery;
use laurus::lexical::query::multi_term::RewriteMethod;
use laurus::lexical::query::prefix::PrefixQuery;
use laurus::lexical::query::range::NumericRangeQuery;
use laurus::lexical::query::regexp::RegexpQuery;
use laurus::lexical::query::term::TermQuery;
use laurus::lexical::query::wildcard::WildcardQuery;
use laurus::storage::Storage;
use laurus::storage::memory::MemoryStorage;
use laurus::{
    DataValue, Document, Engine, IntegerOption, LexicalSearchQuery, Schema, SearchRequestBuilder,
    SortField, SortOrder, TextOption,
};
use regex::Regex;

/// Documents that hold `zqzcommon`, the most frequent of the matching terms.
const COMMON: usize = 20;

/// `n` distinct words of the prefix and two letters.
fn words(prefix: &str, n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let first = char::from(b'a' + (i / 26) as u8);
            let second = char::from(b'a' + (i % 26) as u8);
            format!("{prefix}{first}{second}")
        })
        .collect()
}

fn document(title: &str, n: i64) -> Document {
    Document::builder()
        .add_field("title", DataValue::Text(title.to_string()))
        .add_field("n", DataValue::Int64(n))
        .build()
}

/// An engine over `documents`, committed in `segments` parts. The documents
/// are numbered in the order given, in the field `n` too.
async fn engine(documents: &[(String, String)], segments: usize) -> Engine {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_integer_field("n", IntegerOption::default())
        .add_default_field("title")
        .build();
    let engine = Engine::new(storage.clone(), schema).await.unwrap();
    let mut n = 0;
    for part in documents.chunks(documents.len().div_ceil(segments)) {
        for (id, title) in part {
            n += 1;
            engine.put_document(id, document(title, n)).await.unwrap();
        }
        engine.commit().await.unwrap();
    }
    let files = storage.list_files().unwrap();
    let written = files.iter().filter(|f| f.ends_with(".cfs")).count();
    assert_eq!(written, segments, "{files:?}");
    engine
}

/// The hits of a request, best first.
async fn hits(engine: &Engine, request: SearchRequestBuilder) -> Vec<(String, f32)> {
    engine
        .search(request.build())
        .await
        .unwrap()
        .results
        .into_iter()
        .map(|hit| (hit.id, hit.score))
        .collect()
}

async fn found(engine: &Engine, request: SearchRequestBuilder) -> BTreeSet<String> {
    let hits = hits(engine, request.limit(1000)).await;
    let ids: BTreeSet<String> = hits.iter().map(|(id, _)| id.clone()).collect();
    assert_eq!(ids.len(), hits.len(), "a document is returned twice");
    ids
}

fn request(query: Box<dyn Query>) -> SearchRequestBuilder {
    SearchRequestBuilder::new().lexical_query(LexicalSearchQuery::Obj(query))
}

fn term(word: &str) -> Box<dyn Query> {
    Box::new(TermQuery::new("title", word))
}

/// The queries for the words that start with `zq`.
fn queries() -> Vec<(&'static str, Box<dyn Query>)> {
    vec![
        ("prefix", Box::new(PrefixQuery::new("title", "zq"))),
        (
            "wildcard",
            Box::new(WildcardQuery::new("title", "zq*").unwrap()),
        ),
        (
            "wildcard with a ?",
            Box::new(WildcardQuery::new("title", "z?*").unwrap()),
        ),
        (
            "regexp",
            Box::new(RegexpQuery::new("title", "^zq.*").unwrap()),
        ),
        (
            "regexp, not anchored",
            Box::new(RegexpQuery::new("title", "zq").unwrap()),
        ),
    ]
}

/// Every way a search uses a query for the matching words, with the documents
/// it should find: those that match, or those that do not.
fn searches(query: &dyn Query) -> Vec<(&'static str, SearchRequestBuilder, bool)> {
    let boolean = BooleanQueryBuilder::new;
    let must = boolean()
        .must(term("every"))
        .must(query.clone_box())
        .build();
    let must_not = boolean()
        .must(term("every"))
        .must_not(query.clone_box())
        .build();
    let alone_must_not = boolean().must_not(query.clone_box()).build();
    let should = boolean()
        .should(term("nosuchword"))
        .should(query.clone_box())
        .build();
    vec![
        ("alone", request(query.clone_box()), true),
        ("MUST", request(Box::new(must.clone())), true),
        ("MUST_NOT", request(Box::new(must_not.clone())), false),
        ("MUST_NOT alone", request(Box::new(alone_must_not)), false),
        ("SHOULD", request(Box::new(should)), true),
        (
            "a filter",
            request(term("every")).filter_query(query.clone_box()),
            true,
        ),
        (
            "sorted by a field",
            request(query.clone_box()).sort_by(SortField::Field {
                name: "n".to_string(),
                order: SortOrder::Desc,
            }),
            true,
        ),
        (
            "MUST_NOT, sorted by a field",
            request(Box::new(must_not.clone())).sort_by(SortField::Field {
                name: "n".to_string(),
                order: SortOrder::Asc,
            }),
            false,
        ),
        (
            "in parallel",
            request(query.clone_box()).lexical_parallel(true),
            true,
        ),
        (
            "MUST, in parallel",
            request(Box::new(must)).lexical_parallel(true),
            true,
        ),
        (
            "MUST_NOT, in parallel",
            request(Box::new(must_not)).lexical_parallel(true),
            false,
        ),
    ]
}

/// Compare what each search finds with `matching` and `others`.
async fn check(
    engine: &Engine,
    what: &str,
    matching: &BTreeSet<String>,
    others: &BTreeSet<String>,
    failures: &mut Vec<String>,
) {
    let mut report = |what: String, got: BTreeSet<String>, expected: &BTreeSet<String>| {
        if &got != expected {
            failures.push(format!(
                "{what}: {} documents, {} missing, {} too many",
                got.len(),
                expected.difference(&got).count(),
                got.difference(expected).count()
            ));
        }
    };
    for (name, query) in queries() {
        for (way, request, matches) in searches(query.as_ref()) {
            let expected = if matches { matching } else { others };
            let got = found(engine, request).await;
            report(format!("{what}, {name}, {way}"), got, expected);
        }
    }
    for (dsl, matches) in [
        ("title:zq*", true),
        ("+title:every +title:zq*", true),
        ("+title:every -title:zq*", false),
        ("z?*", true),
    ] {
        let expected = if matches { matching } else { others };
        let got = found(engine, SearchRequestBuilder::new().query_dsl(dsl)).await;
        report(format!("{what}, {dsl}"), got, expected);
    }
}

/// However many terms match, every document with one of them is found, in
/// every way a search uses the query, and the other documents are the ones a
/// `MUST_NOT` clause leaves.
#[tokio::test(flavor = "multi_thread")]
async fn every_document_with_a_matching_term_is_found() {
    let mut failures = Vec::new();
    for rare in [49, 50, 51, 200] {
        for segments in [1, 4] {
            // One document per rare word, and one word in `COMMON` documents.
            let mut documents: Vec<(String, String)> = words("zq", rare)
                .into_iter()
                .enumerate()
                .map(|(i, word)| (format!("rare{i:03}"), format!("{word} every")))
                .collect();
            documents.extend(
                (0..COMMON).map(|i| (format!("common{i:02}"), "zqzcommon every".to_string())),
            );
            let mut matching: BTreeSet<String> =
                documents.iter().map(|(id, _)| id.clone()).collect();
            let mut others = BTreeSet::new();
            for i in 0..5 {
                documents.insert(i * 7, (format!("other{i}"), "azp yq every".to_string()));
                others.insert(format!("other{i}"));
            }
            let engine = engine(&documents, segments).await;
            let what = format!("{rare} rare words, {segments} segments");
            check(&engine, &what, &matching, &others, &mut failures).await;

            // A document that was deleted, and one that lost its word. Their
            // old versions stay in the segments.
            engine.delete_documents("rare003").await.unwrap();
            matching.remove("rare003");
            engine
                .put_document("rare007", document("other every", 500))
                .await
                .unwrap();
            matching.remove("rare007");
            others.insert("rare007".to_string());
            engine.commit().await.unwrap();
            let what = format!("{what}, after a deletion and a change");
            check(&engine, &what, &matching, &others, &mut failures).await;
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A regexp finds what its pattern matches among the terms, whatever stands at
/// its start: the terms used to be sought by a literal prefix that an optional
/// character, an alternation or an escape does not have.
#[tokio::test(flavor = "multi_thread")]
async fn a_regexp_finds_every_term_its_pattern_matches() {
    let words = [
        "e", "ab", "abc", "abbc", "ac", "acc", "b", "ba", "d", "dd", "w1", "w12", "x1", "y2", "yz",
        "z", "1", "12", "9z", "über", "übung",
    ];
    let patterns = [
        "^abc?$",
        "^abb?c$",
        "^ab*c$",
        "^ab{0,1}c$",
        "^ab+c$",
        "^e?$",
        "^(ab)?c?$",
        "^x|z$",
        "^ab|^w1",
        "^(x|y).*",
        "^(?:ab|ac).*",
        r"^\d+$",
        r"^\w\d$",
        r"^\D+$",
        "^[a-c]c*$",
        "^[^a]",
        "^.b",
        "^",
        "^$",
        "(?i)^AB",
        "^(?i)AB",
        r"\Aab",
        r"^a\x62",
        "^über",
        "^üb",
        "b",
        "c$",
        "^a.*c$",
        "(^a)?b",
        "(?:^ab|c)$",
        "^ab|c",
        "(?m)^ab",
    ];
    let documents: Vec<(String, String)> = words
        .iter()
        .map(|word| (word.to_string(), word.to_string()))
        .collect();
    let mut failures = Vec::new();
    for segments in [1, 3] {
        let engine = engine(&documents, segments).await;
        for pattern in patterns {
            let regex = Regex::new(pattern).unwrap();
            let expected: BTreeSet<String> = words
                .iter()
                .filter(|word| regex.is_match(word))
                .map(|word| word.to_string())
                .collect();
            let query = RegexpQuery::new("title", pattern).unwrap();
            let got = found(&engine, request(Box::new(query))).await;
            if got != expected {
                failures.push(format!(
                    "{segments} segments, {pattern}: {got:?}, not {expected:?}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Each match scores the boost, whatever its term, and inside a `BooleanQuery`
/// it adds that to the other clauses.
#[tokio::test(flavor = "multi_thread")]
async fn each_match_scores_the_boost() {
    let mut documents: Vec<(String, String)> = words("zq", 60)
        .into_iter()
        .enumerate()
        .map(|(i, word)| (format!("rare{i:03}"), format!("{word} every")))
        .collect();
    documents.extend((0..COMMON).map(|i| (format!("common{i:02}"), "zqzcommon every".to_string())));
    let count = documents.len();
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let engine = engine(&documents, segments).await;
        let every: BTreeMap<String, f32> = hits(&engine, request(term("every")).limit(1000))
            .await
            .into_iter()
            .collect();
        for boost in [1.0, 2.0, 0.0] {
            let fuzzy = FuzzyQuery::new("title", "zqaa")
                .max_edits(5)
                .prefix_length(2)
                .with_rewrite_method(RewriteMethod::ConstantScore);
            let mut all = queries();
            all.push(("fuzzy, ConstantScore", Box::new(fuzzy)));
            for (name, mut query) in all {
                query.set_boost(boost);
                let what = format!("{segments} segments, boost {boost}, {name}");
                let got = hits(&engine, request(query.clone_box()).limit(1000)).await;
                // The edit distance reaches the words of four letters only.
                let expected = if name.starts_with("fuzzy") { 60 } else { count };
                if got.len() != expected || got.iter().any(|(_, score)| *score != boost) {
                    let scores: BTreeSet<String> =
                        got.iter().map(|(_, score)| score.to_string()).collect();
                    failures.push(format!("{what}: {} hits, scores {scores:?}", got.len()));
                }
                let both = BooleanQueryBuilder::new()
                    .must(term("every"))
                    .should(query.clone_box())
                    .build();
                for (id, score) in hits(&engine, request(Box::new(both)).limit(1000)).await {
                    let fuzzy_miss = name.starts_with("fuzzy") && id.starts_with("common");
                    let added = if fuzzy_miss { 0.0 } else { boost };
                    if (score - every[&id] - added).abs() > 1e-5 {
                        failures.push(format!("{what}, with every: {id} scores {score}"));
                        break;
                    }
                }
                for (threshold, kept) in [(boost, expected), (boost + 0.5, 0)] {
                    let got = hits(
                        &engine,
                        request(query.clone_box())
                            .lexical_min_score(threshold)
                            .limit(1000),
                    )
                    .await;
                    if got.len() != kept {
                        failures.push(format!("{what}, min_score {threshold}: {}", got.len()));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The rewrite methods that score a limited number of terms keep the most
/// frequent ones, and among equally frequent ones the first.
#[tokio::test(flavor = "multi_thread")]
async fn top_terms_are_the_most_frequent() {
    // `zqa` is in one document, `zqb` in two, and so on.
    let mut documents = Vec::new();
    for (i, letter) in ('a'..='h').enumerate() {
        for copy in 0..=i.min(5) {
            documents.push((format!("{letter}{copy}"), format!("zq{letter}")));
        }
    }
    let mut failures = Vec::new();
    for segments in [1, 3] {
        let engine = engine(&documents, segments).await;
        for (method, letters) in [
            (
                RewriteMethod::TopTermsScoring { max_expansions: 2 },
                vec!['f', 'g'],
            ),
            (
                RewriteMethod::TopTermsBlended { max_expansions: 4 },
                vec!['e', 'f', 'g', 'h'],
            ),
            (
                RewriteMethod::TopTermsScoring { max_expansions: 20 },
                ('a'..='h').collect(),
            ),
            (RewriteMethod::BooleanQuery, ('a'..='h').collect()),
            (RewriteMethod::ConstantScore, ('a'..='h').collect()),
        ] {
            let expected: BTreeSet<String> = documents
                .iter()
                .map(|(id, _)| id.clone())
                .filter(|id| letters.iter().any(|letter| id.starts_with(*letter)))
                .collect();
            for (name, query) in [
                (
                    "prefix",
                    Box::new(PrefixQuery::new("title", "zq").with_rewrite_method(method))
                        as Box<dyn Query>,
                ),
                (
                    "wildcard",
                    Box::new(
                        WildcardQuery::new("title", "zq?")
                            .unwrap()
                            .with_rewrite_method(method),
                    ),
                ),
                (
                    "regexp",
                    Box::new(
                        RegexpQuery::new("title", "^zq.$")
                            .unwrap()
                            .with_rewrite_method(method),
                    ),
                ),
            ] {
                let got = found(&engine, request(query)).await;
                if got != expected {
                    failures.push(format!(
                        "{segments} segments, {name}, {method:?}: {got:?}, not {expected:?}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The pages of a search show every hit once, and tied hits come in the order
/// of their documents.
#[tokio::test(flavor = "multi_thread")]
async fn the_pages_of_tied_hits_do_not_overlap() {
    const DOCUMENTS: usize = 60;
    const PAGE: usize = 7;
    let mut failures = Vec::new();
    for segments in [1, 4] {
        for (which, special) in [
            ("none", Vec::new()),
            ("the first", (0..10).collect::<Vec<usize>>()),
            ("the last", (50..60).collect()),
            ("every third", (0..60).step_by(3).collect()),
        ] {
            let documents: Vec<(String, String)> = (0..DOCUMENTS)
                .map(|i| {
                    let word = if special.contains(&i) {
                        "special"
                    } else {
                        "other"
                    };
                    (format!("d{i:03}"), format!("zq{:02} {word}", i % 20))
                })
                .collect();
            let engine = engine(&documents, segments).await;
            let range = || -> Box<dyn Query> {
                Box::new(NumericRangeQuery::i64_range("n", Some(0), Some(1000)))
            };
            let wildcard =
                || -> Box<dyn Query> { Box::new(WildcardQuery::new("title", "zq*").unwrap()) };
            for (name, query) in [("a numeric range", range()), ("a wildcard", wildcard())] {
                // Two levels of scores: with the word `special` and without.
                let query: Box<dyn Query> = Box::new(
                    BooleanQueryBuilder::new()
                        .must(query)
                        .should(term("special"))
                        .build(),
                );
                let mut listed = Vec::new();
                for offset in (0..DOCUMENTS).step_by(PAGE) {
                    let request = request(query.clone_box()).offset(offset).limit(PAGE);
                    listed.extend(hits(&engine, request).await);
                }
                let distinct: BTreeSet<&String> = listed.iter().map(|(id, _)| id).collect();
                let ordered = listed
                    .windows(2)
                    .all(|w| w[0].1 > w[1].1 || (w[0].1 == w[1].1 && w[0].0 < w[1].0));
                if distinct.len() != DOCUMENTS || listed.len() != DOCUMENTS || !ordered {
                    failures.push(format!(
                        "{segments} segments, {name}, special: {which}: {} documents in {} hits, \
                         in order: {ordered}",
                        distinct.len(),
                        listed.len()
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
