//! Issue #1740: a fuzzy query scores the terms closest to its own, up to
//! `max_expansions`, and among equally close ones the most frequent. A closer
//! term scores higher, and the exact one is never dropped. It used to score
//! every term within the edit distance, whatever `max_expansions` and the
//! rewrite method said, by BM25 alone, so that a rare misspelling outranked
//! the exact term.
//!
//! The limit is on the scores: an exclusion and a filter cover every term.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use laurus::lexical::Query;
use laurus::lexical::query::advanced_query::MultiFieldQuery;
use laurus::lexical::query::boolean::BooleanQueryBuilder;
use laurus::lexical::query::fuzzy::FuzzyQuery;
use laurus::lexical::query::multi_term::RewriteMethod;
use laurus::lexical::query::phrase::PhraseQuery;
use laurus::lexical::query::prefix::PrefixQuery;
use laurus::lexical::query::range::NumericRangeQuery;
use laurus::lexical::query::regexp::RegexpQuery;
use laurus::lexical::query::span::{SpanNearQuery, SpanQueryWrapper, SpanTermQuery};
use laurus::lexical::query::synonym::SynonymQuery;
use laurus::lexical::query::term::TermQuery;
use laurus::lexical::query::wildcard::WildcardQuery;
use laurus::storage::Storage;
use laurus::storage::memory::MemoryStorage;
use laurus::{
    DataValue, Document, Engine, IntegerOption, LaurusError, LexicalSearchQuery, Schema,
    SearchRequestBuilder, SortField, SortOrder, TextOption,
};

const TERM: &str = "abcdefgh";

/// Words that are one substitution away from [`TERM`], 128 at most.
fn one_edit_away(n: usize) -> Vec<String> {
    let mut words = Vec::new();
    for letter in 'k'..='z' {
        for position in 0..TERM.len() {
            let mut word: Vec<char> = TERM.chars().collect();
            word[position] = letter;
            words.push(word.into_iter().collect());
        }
    }
    words.truncate(n);
    words
}

/// Words that are two substitutions away from [`TERM`].
fn two_edits_away(n: usize) -> Vec<String> {
    let mut words = Vec::new();
    for letter in 'k'..='z' {
        for first in 0..4 {
            for second in 4..8 {
                let mut word: Vec<char> = TERM.chars().collect();
                word[first] = letter;
                word[second] = letter;
                words.push(word.into_iter().collect());
            }
        }
    }
    words.truncate(n);
    words
}

/// Number of substitutions between words of one length, as the corpora here
/// use no other edits.
fn substitutions(a: &str, b: &str) -> usize {
    assert_eq!(a.len(), b.len());
    a.chars().zip(b.chars()).filter(|(a, b)| a != b).count()
}

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
            let document = Document::builder()
                .add_field("title", DataValue::Text(title.clone()))
                .add_field("n", DataValue::Int64(n))
                .build();
            engine.put_document(id, document).await.unwrap();
        }
        engine.commit().await.unwrap();
    }
    let files = storage.list_files().unwrap();
    let written = files.iter().filter(|f| f.ends_with(".cfs")).count();
    assert_eq!(written, segments, "{files:?}");
    engine
}

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

fn fuzzy() -> FuzzyQuery {
    FuzzyQuery::new("title", TERM)
}

/// Documents of the exact term, 120 words one edit away and 60 two edits away.
/// The words one edit away are in one to three documents each, and those two
/// edits away in four, more than any closer one.
fn corpus() -> Vec<(String, String)> {
    let mut documents = Vec::new();
    let mut put = |word: &str, copies: usize| {
        for copy in 0..copies {
            documents.push((format!("{word}-{copy}"), format!("{word} every")));
        }
    };
    put(TERM, 2);
    for (i, word) in one_edit_away(120).iter().enumerate() {
        put(word, 1 + i % 3);
    }
    for word in two_edits_away(60) {
        put(&word, 4);
    }
    documents.push(("other".to_string(), "orange every".to_string()));
    documents
}

/// The documents of the `limit` words that rank best: by closeness, then by the
/// number of documents, then in the order of the words.
fn closest(documents: &[(String, String)], edits: usize, limit: usize) -> BTreeSet<String> {
    let mut frequency: BTreeMap<String, usize> = BTreeMap::new();
    for (_, title) in documents {
        let word = title.split(' ').next().unwrap();
        *frequency.entry(word.to_string()).or_default() += 1;
    }
    let mut words: Vec<(usize, std::cmp::Reverse<usize>, String)> = frequency
        .into_iter()
        .filter(|(word, _)| word.len() == TERM.len())
        .map(|(word, n)| (substitutions(TERM, &word), std::cmp::Reverse(n), word))
        .filter(|(distance, _, _)| *distance <= edits)
        .collect();
    words.sort();
    words.truncate(limit);
    let kept: BTreeSet<String> = words.into_iter().map(|(_, _, word)| word).collect();
    documents
        .iter()
        .filter(|(_, title)| kept.contains(title.split(' ').next().unwrap()))
        .map(|(id, _)| id.clone())
        .collect()
}

/// The closest terms are scored, up to the limit, by default 50.
#[tokio::test(flavor = "multi_thread")]
async fn the_closest_terms_are_scored() {
    let documents = corpus();
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let engine = engine(&documents, segments).await;
        for edits in [0usize, 1, 2] {
            let query = || fuzzy().max_edits(edits as u32);
            let mut cases: Vec<(String, SearchRequestBuilder, usize)> = vec![
                ("default".to_string(), request(Box::new(query())), 50),
                (
                    format!("title:{TERM}~{edits}"),
                    SearchRequestBuilder::new().query_dsl(format!("title:{TERM}~{edits}")),
                    50,
                ),
            ];
            for limit in [1, 2, 10, 50, 121, 130, 500] {
                cases.push((
                    format!("max_expansions({limit})"),
                    request(Box::new(query().max_expansions(limit))),
                    limit,
                ));
                for method in [
                    RewriteMethod::TopTermsScoring {
                        max_expansions: limit,
                    },
                    RewriteMethod::TopTermsBlended {
                        max_expansions: limit,
                    },
                ] {
                    cases.push((
                        format!("{method:?}"),
                        request(Box::new(query().with_rewrite_method(method))),
                        limit,
                    ));
                }
            }
            for method in [RewriteMethod::ConstantScore, RewriteMethod::BooleanQuery] {
                cases.push((
                    format!("{method:?}"),
                    request(Box::new(query().with_rewrite_method(method))),
                    usize::MAX,
                ));
            }
            for (what, request, limit) in cases {
                let expected = closest(&documents, edits, limit);
                let got = found(&engine, request).await;
                if got != expected {
                    failures.push(format!(
                        "{segments} segments, {edits} edits, {what}: {} documents, {} missing, \
                         {} too many",
                        got.len(),
                        expected.difference(&got).count(),
                        got.difference(&expected).count()
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The limit and the rewrite method are one setting: the later call wins, and
/// a limit leaves a method without one alone only when nothing is scored by
/// the terms.
#[tokio::test(flavor = "multi_thread")]
async fn the_later_of_the_limit_and_the_rewrite_method_wins() {
    let documents = corpus();
    let engine = engine(&documents, 1).await;
    let query = || fuzzy().max_edits(1);
    let scoring = |max_expansions| RewriteMethod::TopTermsScoring { max_expansions };
    let mut failures = Vec::new();
    for (what, query, limit) in [
        (
            "a limit, then a method",
            query().max_expansions(10).with_rewrite_method(scoring(20)),
            20,
        ),
        (
            "a method, then a limit",
            query().with_rewrite_method(scoring(20)).max_expansions(10),
            10,
        ),
        (
            "two limits",
            query().max_expansions(10).max_expansions(30),
            30,
        ),
        (
            "every term, then a limit",
            query()
                .with_rewrite_method(RewriteMethod::BooleanQuery)
                .max_expansions(10),
            10,
        ),
        (
            "a limit, then every term",
            query()
                .max_expansions(10)
                .with_rewrite_method(RewriteMethod::BooleanQuery),
            usize::MAX,
        ),
        (
            "a constant score, then a limit",
            query()
                .with_rewrite_method(RewriteMethod::ConstantScore)
                .max_expansions(10),
            usize::MAX,
        ),
    ] {
        let expected = closest(&documents, 1, limit);
        let got = found(&engine, request(Box::new(query))).await;
        if got != expected {
            failures.push(format!("{what}: {} documents", got.len()));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The exact term ranks before the words one edit away, and those before the
/// words two edits away, however rare each is. With the frequencies blended, a
/// word's score is the exact term's times `1 - edits / length`.
#[tokio::test(flavor = "multi_thread")]
async fn a_closer_term_ranks_higher() {
    let documents = corpus();
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let engine = engine(&documents, segments).await;
        let method = RewriteMethod::TopTermsBlended {
            max_expansions: 500,
        };
        for (what, query, boost) in [
            ("boost 1", fuzzy().with_rewrite_method(method), 1.0f32),
            (
                "boost 2",
                fuzzy().with_rewrite_method(method).with_boost(2.0),
                2.0,
            ),
        ] {
            let exact = hits(
                &engine,
                request(Box::new(fuzzy().max_edits(0).with_boost(boost))),
            )
            .await;
            let got = hits(&engine, request(Box::new(query)).limit(1000)).await;
            if got.len() != documents.len() - 1 {
                failures.push(format!(
                    "{segments} segments, {what}: {} documents",
                    got.len()
                ));
            }
            let mut scores: [BTreeSet<String>; 3] = Default::default();
            let mut ranks = Vec::new();
            for (id, score) in &got {
                let distance = substitutions(TERM, id.split('-').next().unwrap());
                scores[distance].insert(format!("{score:.5}"));
                ranks.push(distance);
            }
            if !ranks.is_sorted() {
                failures.push(format!(
                    "{segments} segments, {what}: not in the order of distance"
                ));
            }
            let best = got[0].1;
            for (distance, scores) in scores.iter().enumerate() {
                let expected = best * (1.0 - distance as f32 / TERM.len() as f32);
                if scores.len() != 1
                    || (scores.first().unwrap().parse::<f32>().unwrap() - expected).abs() > 1e-4
                {
                    failures.push(format!(
                        "{segments} segments, {what}, {distance} edits: scores {scores:?}, not \
                         {expected}"
                    ));
                }
            }
            // The exact term alone has its own, lower frequency.
            if exact[0].1 <= best {
                failures.push(format!(
                    "{segments} segments, {what}: the frequency is not blended"
                ));
            }
        }
        // Only the scored terms are blended: the more frequent words that the
        // limit leaves out do not count.
        let exact = hits(&engine, request(term(TERM))).await;
        let one = hits(&engine, request(Box::new(fuzzy().max_expansions(1)))).await;
        if one != exact {
            failures.push(format!(
                "{segments} segments, one term: {one:?} in place of {exact:?}"
            ));
        }
        // Without blending, the weights still apply.
        let method = RewriteMethod::TopTermsScoring {
            max_expansions: 500,
        };
        let got: BTreeMap<String, f32> = hits(
            &engine,
            request(Box::new(fuzzy().with_rewrite_method(method))).limit(1000),
        )
        .await
        .into_iter()
        .collect();
        for word in [
            one_edit_away(5).pop().unwrap(),
            two_edits_away(5).pop().unwrap(),
        ] {
            let alone = hits(&engine, request(term(&word))).await;
            let weight = 1.0 - substitutions(TERM, &word) as f32 / TERM.len() as f32;
            let id = format!("{word}-0");
            let score = alone.iter().find(|(found, _)| *found == id).unwrap().1;
            if (got[&id] - score * weight).abs() > 1e-4 {
                failures.push(format!(
                    "{segments} segments, {word}: {} in place of {score} times {weight}",
                    got[&id]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The common spelling of a word ranks before its rare misspellings, in the
/// DSL too.
#[tokio::test(flavor = "multi_thread")]
async fn a_common_term_ranks_before_its_rare_misspellings() {
    let mut documents: Vec<(String, String)> = one_edit_away(30)
        .into_iter()
        .enumerate()
        .map(|(i, word)| (format!("rare{i:02}"), word))
        .collect();
    documents.extend((0..20).map(|i| (format!("exact{i:02}"), TERM.to_string())));
    // So that the term is not in most of the documents.
    documents.extend((0..100).map(|i| (format!("other{i:03}"), "orange".to_string())));
    for segments in [1, 4] {
        let engine = engine(&documents, segments).await;
        for (what, request) in [
            ("a query", request(Box::new(fuzzy().max_edits(1)))),
            (
                "the DSL",
                SearchRequestBuilder::new().query_dsl(format!("title:{TERM}~1")),
            ),
            (
                "the DSL, 2 edits",
                SearchRequestBuilder::new().query_dsl(format!("{TERM}~")),
            ),
        ] {
            let got = hits(&engine, request.limit(100)).await;
            assert_eq!(got.len(), 50, "{segments} segments, {what}");
            let exact = got.iter().position(|(id, _)| id.starts_with("rare"));
            assert_eq!(exact, Some(20), "{segments} segments, {what}: {got:?}");
        }
    }
}

/// An exclusion and a filter use the documents of every matching term, not of
/// the scored ones.
#[tokio::test(flavor = "multi_thread")]
async fn exclusions_and_filters_cover_every_term() {
    let documents = corpus();
    let every: BTreeSet<String> = documents.iter().map(|(id, _)| id.clone()).collect();
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let engine = engine(&documents, segments).await;
        let prefix = PrefixQuery::new("title", "a")
            .with_rewrite_method(RewriteMethod::TopTermsScoring { max_expansions: 3 });
        let starting_with_a: BTreeSet<String> = every
            .iter()
            .filter(|id| id.starts_with('a'))
            .cloned()
            .collect();
        for (what, query, matching) in [
            (
                "fuzzy, 1 edit",
                Box::new(fuzzy().max_edits(1)) as Box<dyn Query>,
                closest(&documents, 1, usize::MAX),
            ),
            (
                "fuzzy, 2 edits",
                Box::new(fuzzy()),
                closest(&documents, 2, usize::MAX),
            ),
            (
                "fuzzy, one term",
                Box::new(fuzzy().max_expansions(1)),
                closest(&documents, 2, usize::MAX),
            ),
            ("prefix, three terms", Box::new(prefix), starting_with_a),
        ] {
            let others: BTreeSet<String> = every.difference(&matching).cloned().collect();
            let must_not = BooleanQueryBuilder::new()
                .must(term("every"))
                .must_not(query.clone_box())
                .build();
            let filter = BooleanQueryBuilder::new()
                .must(term("every"))
                .filter(query.clone_box())
                .build();
            let nested = BooleanQueryBuilder::new()
                .must(term("every"))
                .must_not(Box::new(
                    BooleanQueryBuilder::new()
                        .should(term("nosuchword"))
                        .should(query.clone_box())
                        .build(),
                ))
                .build();
            let inner = BooleanQueryBuilder::new()
                .must(Box::new(must_not.clone()))
                .must(term("every"))
                .build();
            let alone = BooleanQueryBuilder::new()
                .must_not(query.clone_box())
                .build();
            let filter_alone = BooleanQueryBuilder::new().filter(query.clone_box()).build();
            let sort = || SortField::Field {
                name: "n".to_string(),
                order: SortOrder::Desc,
            };
            for (way, request, expected) in [
                ("MUST_NOT", request(Box::new(must_not.clone())), &others),
                ("MUST_NOT alone", request(Box::new(alone)), &others),
                ("inside a MUST_NOT", request(Box::new(nested)), &others),
                ("MUST_NOT inside a MUST", request(Box::new(inner)), &others),
                (
                    "MUST_NOT, sorted by a field",
                    request(Box::new(must_not.clone())).sort_by(sort()),
                    &others,
                ),
                (
                    "MUST_NOT, in parallel",
                    request(Box::new(must_not)).lexical_parallel(true),
                    &others,
                ),
                (
                    "a filter clause",
                    request(Box::new(filter.clone())),
                    &matching,
                ),
                (
                    "a filter clause, in parallel",
                    request(Box::new(filter)).lexical_parallel(true),
                    &matching,
                ),
                (
                    "a filter clause alone, in parallel",
                    request(Box::new(filter_alone)).lexical_parallel(true),
                    &matching,
                ),
                (
                    "a filter",
                    request(term("every")).filter_query(query.clone_box()),
                    &matching,
                ),
                (
                    "a filter, in parallel",
                    request(term("every"))
                        .filter_query(query.clone_box())
                        .lexical_parallel(true),
                    &matching,
                ),
            ] {
                let got = found(&engine, request).await;
                if &got != expected {
                    failures.push(format!(
                        "{segments} segments, {what}, {way}: {} documents, {} missing, {} too \
                         many",
                        got.len(),
                        expected.difference(&got).count(),
                        got.difference(expected).count()
                    ));
                }
            }
        }
        let dsl = format!("+title:every -title:{TERM}~1");
        let matching = closest(&documents, 1, usize::MAX);
        let others: BTreeSet<String> = every.difference(&matching).cloned().collect();
        let got = found(&engine, SearchRequestBuilder::new().query_dsl(dsl.clone())).await;
        if got != others {
            failures.push(format!("{segments} segments, {dsl}: {}", got.len()));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A clause that is not scored selects the same documents whether the clauses
/// run in parallel or not, whatever its query, also after documents were
/// replaced and deleted.
#[tokio::test(flavor = "multi_thread")]
async fn a_clause_that_is_not_scored_is_the_same_in_parallel() {
    let documents: Vec<(String, String)> = (0..60)
        .map(|i| {
            let words = ["red", "green", "blue", "grey"];
            let title = format!("every {} {} w{}", words[i % 4], words[(i / 4) % 4], i % 7);
            (format!("d{i:02}"), title)
        })
        .collect();
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let engine = engine(&documents, segments).await;
        for i in (0..60).step_by(9) {
            engine.delete_documents(&format!("d{i:02}")).await.unwrap();
        }
        for i in (1..60).step_by(11) {
            let document = Document::builder()
                .add_field("title", DataValue::Text("every black".to_string()))
                .add_field("n", DataValue::Int64(1000 + i))
                .build();
            engine
                .put_document(&format!("d{i:02}"), document)
                .await
                .unwrap();
        }
        engine.commit().await.unwrap();
        let every = found(&engine, request(term("every"))).await;
        let span = |word: &str| -> Box<dyn laurus::lexical::query::span::SpanQuery> {
            Box::new(SpanTermQuery::new("title", word))
        };
        let near = SpanNearQuery::new("title", vec![span("red"), span("w0")], 3, true);
        let nested = BooleanQueryBuilder::new()
            .must(term("red"))
            .filter(term("green"))
            .must_not(term("w1"))
            .build();
        let queries: Vec<(&str, Box<dyn Query>)> = vec![
            ("term", term("red")),
            ("no such term", term("nosuchword")),
            (
                "phrase",
                Box::new(PhraseQuery::new(
                    "title",
                    vec!["red".to_string(), "green".to_string()],
                )),
            ),
            (
                "range",
                Box::new(NumericRangeQuery::i64_range("n", Some(10), Some(1000))),
            ),
            ("span", Box::new(SpanQueryWrapper::new(Box::new(near)))),
            ("prefix", Box::new(PrefixQuery::new("title", "gre"))),
            (
                "wildcard",
                Box::new(WildcardQuery::new("title", "?re*").unwrap()),
            ),
            (
                "regexp",
                Box::new(RegexpQuery::new("title", "^w[135]$").unwrap()),
            ),
            (
                "fuzzy",
                Box::new(FuzzyQuery::new("title", "grey").max_edits(2)),
            ),
            (
                "synonyms",
                Box::new(SynonymQuery::new("title", vec!["red", "w2"])),
            ),
            (
                "fields",
                Box::new(
                    MultiFieldQuery::new("blue".to_string()).add_field("title".to_string(), 1.0),
                ),
            ),
            ("boolean", Box::new(nested)),
        ];
        for (what, query) in queries {
            let mut selected = Vec::new();
            for parallel in [false, true] {
                let filter = BooleanQueryBuilder::new()
                    .must(term("every"))
                    .filter(query.clone_box())
                    .build();
                let must_not = BooleanQueryBuilder::new()
                    .must(term("every"))
                    .must_not(query.clone_box())
                    .build();
                let with = request(Box::new(filter)).lexical_parallel(parallel);
                let without = request(Box::new(must_not)).lexical_parallel(parallel);
                let with = found(&engine, with).await;
                let without = found(&engine, without).await;
                if with.union(&without).cloned().collect::<BTreeSet<_>>() != every
                    || !with.is_disjoint(&without)
                {
                    failures.push(format!(
                        "{segments} segments, {what}, parallel {parallel}: {} with and {} \
                         without it, of {}",
                        with.len(),
                        without.len(),
                        every.len()
                    ));
                }
                selected.push(with);
            }
            let expected = if what == "no such term" {
                0..1
            } else {
                1..every.len()
            };
            if selected[0] != selected[1] || !expected.contains(&selected[0].len()) {
                failures.push(format!(
                    "{segments} segments, {what}: {} documents, {} in parallel, of {}",
                    selected[0].len(),
                    selected[1].len(),
                    every.len()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The first hits of a search are the first of the whole ranking: the bounds
/// by which documents are skipped hold for the blended frequency.
#[tokio::test(flavor = "multi_thread")]
async fn the_first_hits_are_those_of_the_whole_ranking() {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    let near = one_edit_away(12);
    let far = two_edits_away(8);
    let documents: Vec<(String, String)> = (0..1500)
        .map(|i| {
            let mut words = Vec::new();
            for _ in 0..next(4) {
                let word = match next(12) {
                    0 => TERM,
                    n if n < 8 => &near[next(near.len())],
                    _ => &far[next(far.len())],
                };
                // Some words several times.
                for _ in 0..=next(3) {
                    words.push(word.to_string());
                }
            }
            for _ in 0..1 + next(20) {
                words.push("orange".to_string());
            }
            (format!("d{i:04}"), words.join(" "))
        })
        .collect();
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let engine = engine(&documents, segments).await;
        for method in [
            RewriteMethod::TopTermsBlended { max_expansions: 50 },
            RewriteMethod::TopTermsBlended { max_expansions: 5 },
            RewriteMethod::TopTermsScoring { max_expansions: 50 },
        ] {
            let query = || Box::new(fuzzy().with_rewrite_method(method)) as Box<dyn Query>;
            // Every document is wanted, so none is skipped.
            let whole = hits(&engine, request(query()).limit(documents.len())).await;
            assert!(whole.len() > 300, "{} hits", whole.len());
            for limit in [1, 2, 3, 10, 50, 200] {
                let first = hits(&engine, request(query()).limit(limit)).await;
                let expected = &whole[..limit];
                let same = first.len() == limit
                    && first
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| a.0 == b.0 && (a.1 - b.1).abs() < 1e-5);
                if !same {
                    failures.push(format!("{segments} segments, {method:?}, limit {limit}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A term no longer than the edit distance matches with a weight of zero, and
/// no score is negative or not a number.
#[tokio::test(flavor = "multi_thread")]
async fn a_short_term_gives_no_negative_score() {
    let documents: Vec<(String, String)> = ["x", "xy", "yz", "xyz", "wxyz", "ü", "üx"]
        .iter()
        .map(|word| (word.to_string(), word.to_string()))
        .collect();
    for segments in [1, 3] {
        let engine = engine(&documents, segments).await;
        for (query, edits, expected) in [
            // Closest first: by `1 - edits / min(length)`, down to zero.
            ("x", 0, vec!["x"]),
            ("x", 1, vec!["x", "xy", "ü", "üx"]),
            ("xy", 2, vec!["xy", "xyz", "x", "yz", "wxyz", "ü", "üx"]),
            ("ü", 1, vec!["ü", "x", "üx"]),
            ("xyz", 100, vec!["xyz", "wxyz", "xy", "yz", "x", "ü", "üx"]),
        ] {
            let fuzzy = FuzzyQuery::new("title", query).max_edits(edits);
            let got = hits(&engine, request(Box::new(fuzzy)).limit(100)).await;
            let what = format!("{segments} segments, {query}~{edits}: {got:?}");
            assert!(
                got.iter().all(|(_, s)| s.is_finite() && *s >= 0.0),
                "{what}"
            );
            let ids: BTreeSet<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
            assert_eq!(ids, expected.iter().copied().collect(), "{what}");
            assert_eq!(got[0].0, expected[0], "{what}");
            assert!(got[0].1 > 0.0, "{what}");
        }
    }
}

/// A limit of no terms would find nothing without an error, so it is rejected,
/// wherever the query stands.
#[tokio::test(flavor = "multi_thread")]
async fn a_limit_of_no_terms_is_rejected() {
    let engine = engine(&corpus(), 1).await;
    let none = RewriteMethod::TopTermsScoring { max_expansions: 0 };
    for (what, query) in [
        (
            "max_expansions",
            Box::new(fuzzy().max_expansions(0)) as Box<dyn Query>,
        ),
        (
            "a fuzzy query's rewrite method",
            Box::new(fuzzy().with_rewrite_method(none)),
        ),
        (
            "a prefix query's rewrite method",
            Box::new(PrefixQuery::new("title", "a").with_rewrite_method(none)),
        ),
        (
            "a wildcard query's rewrite method",
            Box::new(
                WildcardQuery::new("title", "a*")
                    .unwrap()
                    .with_rewrite_method(none),
            ),
        ),
        (
            "a regexp query's rewrite method",
            Box::new(
                RegexpQuery::new("title", "^a")
                    .unwrap()
                    .with_rewrite_method(none),
            ),
        ),
    ] {
        let must_not = BooleanQueryBuilder::new()
            .must(term("every"))
            .must_not(query.clone_box())
            .build();
        for (way, request) in [
            ("alone", request(query.clone_box())),
            ("as a MUST_NOT", request(Box::new(must_not))),
            (
                "as a filter",
                request(term("every")).filter_query(query.clone_box()),
            ),
        ] {
            match engine
                .search(request.build())
                .await
                .map(|response| response.results)
            {
                Err(LaurusError::InvalidArgument(message)) => {
                    assert!(
                        message.contains("max_expansions"),
                        "{what}, {way}: {message}"
                    )
                }
                other => panic!("{what}, {way}: {:?}", other.map(|hits| hits.len())),
            }
        }
    }
}

/// A boost that is not finite or is negative is rejected, for each query whose
/// terms are expanded.
#[tokio::test(flavor = "multi_thread")]
async fn a_boost_out_of_range_is_rejected() {
    let engine = engine(&corpus(), 1).await;
    for boost in [-1.0, f32::NAN, f32::INFINITY] {
        let queries: Vec<(&str, Box<dyn Query>)> = vec![
            ("fuzzy", Box::new(fuzzy().with_boost(boost))),
            (
                "prefix",
                Box::new(PrefixQuery::new("title", "a").with_boost(boost)),
            ),
            (
                "wildcard",
                Box::new(WildcardQuery::new("title", "a*").unwrap().with_boost(boost)),
            ),
            (
                "regexp",
                Box::new(RegexpQuery::new("title", "^a").unwrap().with_boost(boost)),
            ),
        ];
        for (what, query) in queries {
            match engine
                .search(request(query).build())
                .await
                .map(|response| response.results)
            {
                Err(LaurusError::InvalidArgument(message)) => {
                    assert!(message.contains("boost"), "{what}, {boost}: {message}")
                }
                other => panic!("{what}, {boost}: {:?}", other.map(|hits| hits.len())),
            }
        }
    }
}
