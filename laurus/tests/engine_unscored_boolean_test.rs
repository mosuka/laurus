//! Issue #1744: a `BooleanQuery` without a clause that scores (`Must` or
//! `Should`) gives every match the score 0, as Elasticsearch does for filters
//! and for negations alone. Its filter and `MustNot` clauses decide which
//! documents match and carry no score. It used to score each match as a term
//! that one document has once: a value that grew with the index, outranked the
//! clauses of an outer query, and made `min_score` keep or drop all hits.
//!
//! Every document has `every`. Every third also has `third`, every fifth
//! `fifth`. `#` marks a filter clause below.

use std::collections::BTreeMap;
use std::sync::Arc;

use laurus::lexical::Query;
use laurus::lexical::query::boolean::{BooleanQuery, BooleanQueryBuilder};
use laurus::lexical::query::term::TermQuery;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::MemoryStorage;
use laurus::{
    DataValue, Document, Engine, LexicalSearchQuery, Schema, SearchRequestBuilder, TextOption,
};

fn term(word: &str) -> Box<dyn Query> {
    Box::new(TermQuery::new("title", word))
}

fn title(id: u64) -> String {
    let mut title = String::from("every");
    if id.is_multiple_of(3) {
        title.push_str(" third");
    }
    if id.is_multiple_of(5) {
        title.push_str(" fifth");
    }
    title
}

/// A query without a scored clause: its name, itself, and which documents it
/// matches.
type Unscored = (&'static str, BooleanQuery, fn(u64) -> bool);

/// The queries without a scored clause.
fn unscored() -> Vec<Unscored> {
    let b = BooleanQueryBuilder::new;
    vec![
        ("#third", b().filter(term("third")).build(), |id| {
            id.is_multiple_of(3)
        }),
        (
            "#third #fifth",
            b().filter(term("third")).filter(term("fifth")).build(),
            |id| id.is_multiple_of(15),
        ),
        (
            "#third -fifth",
            b().filter(term("third")).must_not(term("fifth")).build(),
            |id| id.is_multiple_of(3) && !id.is_multiple_of(5),
        ),
        ("-third", b().must_not(term("third")).build(), |id| {
            !id.is_multiple_of(3)
        }),
        (
            "-third -fifth",
            b().must_not(term("third")).must_not(term("fifth")).build(),
            |id| !id.is_multiple_of(3) && !id.is_multiple_of(5),
        ),
        (
            "#third, boost 2",
            b().filter(term("third")).boost(2.0).build(),
            |id| id.is_multiple_of(3),
        ),
        (
            "-third, boost 2",
            b().must_not(term("third")).boost(2.0).build(),
            |id| !id.is_multiple_of(3),
        ),
    ]
}

/// The number of the documents 1 to `docs` that `matches` selects.
fn count(docs: u64, matches: fn(u64) -> bool) -> usize {
    (1..=docs).filter(|id| matches(*id)).count()
}

async fn engine(docs: u64, segments: u64) -> (Engine, Arc<dyn Storage>) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_default_field("title")
        .build();
    let engine = Engine::new(storage.clone(), schema).await.unwrap();
    for id in 1..=docs {
        let document = Document::builder()
            .add_field("title", DataValue::Text(title(id)))
            .build();
        engine
            .put_document(&id.to_string(), document)
            .await
            .unwrap();
        if id.is_multiple_of(docs / segments) {
            engine.commit().await.unwrap();
        }
    }
    let files = storage.list_files().unwrap();
    let written = files
        .iter()
        .filter(|f| f.starts_with("lexical/") && f.ends_with(".cfs"))
        .count();
    assert_eq!(written, segments as usize, "{files:?}");
    (engine, storage)
}

fn store(docs: u64, segments: u64) -> LexicalStore {
    let store = LexicalStore::new(
        Arc::new(MemoryStorage::new(Default::default())),
        LexicalIndexConfig::builder().build(),
    )
    .unwrap();
    for id in 1..=docs {
        store
            .upsert_document(id, Document::builder().add_text("title", title(id)).build())
            .unwrap();
        if id.is_multiple_of(docs / segments) {
            store.commit().unwrap();
        }
    }
    store
}

/// The scores of a search through `Engine`, by id.
async fn scores(engine: &Engine, request: SearchRequestBuilder) -> BTreeMap<u64, f32> {
    engine
        .search(request.limit(1000).build())
        .await
        .unwrap()
        .results
        .into_iter()
        .map(|hit| (hit.id.parse().unwrap(), hit.score))
        .collect()
}

fn request(query: &BooleanQuery) -> SearchRequestBuilder {
    SearchRequestBuilder::new().lexical_query(LexicalSearchQuery::Obj(Box::new(query.clone())))
}

/// Every match scores 0, with any boost, in parallel and not, on one segment
/// and several, through `Engine`, `LexicalStore` and the DSL, whatever the
/// number of documents.
#[tokio::test(flavor = "multi_thread")]
async fn a_query_without_a_scored_clause_scores_0() {
    let mut failures = Vec::new();
    for docs in [12u64, 60] {
        for segments in [1, 4] {
            let (engine, _) = engine(docs, segments).await;
            let store = store(docs, segments);
            for (what, query, matches) in unscored() {
                let expected = count(docs, matches);
                for parallel in [false, true] {
                    let what = format!(
                        "{docs} documents, {segments} segments, {what}, parallel {parallel}"
                    );
                    let got = scores(&engine, request(&query).lexical_parallel(parallel)).await;
                    if got.len() != expected || got.values().any(|s| *s != 0.0) {
                        failures.push(format!("{what}, Engine: {got:?}"));
                    }
                    let found = store
                        .search(
                            LexicalSearchRequest::new(Box::new(query.clone()) as Box<dyn Query>)
                                .limit(1000)
                                .parallel(parallel),
                        )
                        .unwrap();
                    let lexical: Vec<f32> = found.hits.iter().map(|hit| hit.score).collect();
                    if lexical.len() != expected || lexical.iter().any(|s| *s != 0.0) {
                        failures.push(format!("{what}, LexicalStore: {lexical:?}"));
                    }
                }
            }
            for dsl in ["-title:third", "-third", "-title:third -title:fifth"] {
                let got = scores(&engine, SearchRequestBuilder::new().query_dsl(dsl)).await;
                if got.is_empty() || got.values().any(|s| *s != 0.0) {
                    failures.push(format!(
                        "{docs} documents, {segments} segments, DSL {dsl}: {got:?}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Inside another query such a query adds 0: the outer clauses alone rank.
#[tokio::test(flavor = "multi_thread")]
async fn a_nested_query_without_a_scored_clause_adds_0() {
    let b = BooleanQueryBuilder::new;
    let filter = |word| -> Box<dyn Query> { Box::new(b().filter(term(word)).build()) };
    let negation = |word| -> Box<dyn Query> { Box::new(b().must_not(term(word)).build()) };
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let (engine, _) = engine(60, segments).await;
        for parallel in [false, true] {
            let what = |shape: &str| format!("{segments} segments, {shape}, parallel {parallel}");
            let run = |query: BooleanQuery| {
                let engine = &engine;
                async move { scores(engine, request(&query).lexical_parallel(parallel)).await }
            };
            // `+every +(#third)` is `+every #third`.
            let nested = run(b().must(term("every")).must(filter("third")).build()).await;
            let flat = run(b().must(term("every")).filter(term("third")).build()).await;
            if nested.len() != 20 || nested != flat {
                failures.push(what(&format!("+every +(#third): {nested:?}, not {flat:?}")));
            }
            // `fifth (#third)` and `fifth (-third)` score `fifth` alone.
            let fifth = run(b().should(term("fifth")).build()).await;
            for (shape, inner, matches) in [
                ("fifth (#third)", filter("third"), 28usize),
                ("fifth (-third)", negation("third"), 44),
            ] {
                let got = run(b().should(term("fifth")).should(inner).build()).await;
                let wrong: Vec<(u64, f32)> = got
                    .iter()
                    .filter(|(id, score)| match fifth.get(id) {
                        Some(alone) => (**score - alone).abs() > 1e-5,
                        None => **score != 0.0,
                    })
                    .map(|(id, score)| (*id, *score))
                    .collect();
                if got.len() != matches || !wrong.is_empty() {
                    failures.push(what(&format!(
                        "{shape}: {} hits, scored apart from fifth: {wrong:?}",
                        got.len()
                    )));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A `min_score` above 0 drops every match of such a query, and one of 0 keeps
/// them all, whatever the boost; `count` agrees.
#[tokio::test(flavor = "multi_thread")]
async fn min_score_and_count_treat_the_score_as_0() {
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let (engine, _) = engine(60, segments).await;
        let store = store(60, segments);
        for (what, query, matches) in unscored() {
            for (min_score, expected) in [(0.0f32, count(60, matches)), (0.5, 0)] {
                let what = format!("{segments} segments, {what}, min_score {min_score}");
                let got = scores(&engine, request(&query).lexical_min_score(min_score)).await;
                let request = || {
                    LexicalSearchRequest::new(Box::new(query.clone()) as Box<dyn Query>)
                        .limit(1000)
                        .min_score(min_score)
                };
                let found = store.search(request()).unwrap().hits.len();
                let count = store.count(request()).unwrap();
                if got.len() != expected || found != expected || count != expected as u64 {
                    failures.push(format!(
                        "{what}: Engine {}, LexicalStore {found}, count {count}, not {expected}",
                        got.len()
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
