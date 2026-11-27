//! Issue #1743: on an index of several segments, a `BooleanQuery` whose
//! required clauses are filters returns each matching document once, as on one
//! segment. Each segment used to take a filter's documents from the cached set
//! of the whole index, and so returned the documents of every segment: hits
//! repeated per segment, `Engine` lost their ids, and a `MustNot` clause
//! excluded only the segment's own.
//!
//! Every document has `every`. Every third also has `third`, every fifth
//! `fifth`. `#` marks a filter clause below.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use laurus::lexical::Query;
use laurus::lexical::query::boolean::{BooleanQuery, BooleanQueryBuilder};
use laurus::lexical::query::term::TermQuery;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::MemoryStorage;
use laurus::vector::Vector;
use laurus::vector::store::request::QueryVector;
use laurus::{
    DataValue, Document, Engine, FlatOption, FusionAlgorithm, IntegerOption, LexicalSearchQuery,
    Schema, SearchRequestBuilder, SortField, SortOrder, TextOption, VectorSearchQuery,
};

const DOCS: u64 = 60;

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

/// What a document of a title matches.
struct Words {
    third: bool,
    fifth: bool,
}

impl Words {
    fn of(title: &str) -> Self {
        Words {
            third: title.split(' ').any(|w| w == "third"),
            fifth: title.split(' ').any(|w| w == "fifth"),
        }
    }
}

/// A query and the documents it matches.
struct Shape {
    name: &'static str,
    query: BooleanQuery,
    matches: fn(&Words) -> bool,
    /// Whether a parallel search scores it as a serial one: not so for a query
    /// without a scored clause (Issue #1744).
    scored: bool,
}

fn shapes() -> Vec<Shape> {
    let b = BooleanQueryBuilder::new;
    let filter = |word| b().filter(term(word)).build();
    vec![
        Shape {
            name: "#third",
            query: filter("third"),
            matches: |w| w.third,
            scored: false,
        },
        Shape {
            name: "#third #fifth",
            query: b().filter(term("third")).filter(term("fifth")).build(),
            matches: |w| w.third && w.fifth,
            scored: false,
        },
        Shape {
            name: "#third -fifth",
            query: b().filter(term("third")).must_not(term("fifth")).build(),
            matches: |w| w.third && !w.fifth,
            scored: false,
        },
        Shape {
            name: "fifth #third",
            query: b().should(term("fifth")).filter(term("third")).build(),
            matches: |w| w.third,
            scored: true,
        },
        Shape {
            name: "+every +(#third -fifth)",
            query: b()
                .must(term("every"))
                .must(Box::new(
                    b().filter(term("third")).must_not(term("fifth")).build(),
                ))
                .build(),
            matches: |w| w.third && !w.fifth,
            scored: true,
        },
        Shape {
            name: "(#third) (#fifth)",
            query: b()
                .should(Box::new(filter("third")))
                .should(Box::new(filter("fifth")))
                .build(),
            matches: |w| w.third || w.fifth,
            scored: false,
        },
        Shape {
            name: "#(#third)",
            query: b().filter(Box::new(filter("third"))).build(),
            matches: |w| w.third,
            scored: false,
        },
        Shape {
            name: "#third -(#fifth)",
            query: b()
                .filter(term("third"))
                .must_not(Box::new(filter("fifth")))
                .build(),
            matches: |w| w.third && !w.fifth,
            scored: false,
        },
        Shape {
            name: "+every #third",
            query: b().must(term("every")).filter(term("third")).build(),
            matches: |w| w.third,
            scored: true,
        },
    ]
}

/// The ids that match, in the order of `n`.
fn expected(titles: &BTreeMap<u64, String>, matches: fn(&Words) -> bool) -> Vec<u64> {
    titles
        .iter()
        .filter(|(_, title)| matches(&Words::of(title)))
        .map(|(id, _)| *id)
        .collect()
}

fn initial_titles() -> BTreeMap<u64, String> {
    (1..=DOCS).map(|id| (id, title(id))).collect()
}

fn document(id: u64, title: &str) -> Document {
    Document::builder()
        .add_field("title", DataValue::Text(title.to_string()))
        .add_field("n", DataValue::Int64(id as i64))
        .add_field(
            "vector",
            DataValue::Vector(vec![1.0, id as f32 / DOCS as f32]),
        )
        .build()
}

async fn engine(segments: u64) -> (Engine, Arc<dyn Storage>) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_integer_field("n", IntegerOption::default())
        .add_flat_field("vector", FlatOption::default().dimension(2))
        .add_default_field("title")
        .build();
    let engine = Engine::new(storage.clone(), schema).await.unwrap();
    for id in 1..=DOCS {
        engine
            .put_document(&id.to_string(), document(id, &title(id)))
            .await
            .unwrap();
        if id % (DOCS / segments) == 0 {
            engine.commit().await.unwrap();
        }
    }
    assert_eq!(lexical_segments(&storage), segments as usize);
    (engine, storage)
}

/// The number of segments of the lexical index.
fn lexical_segments(storage: &Arc<dyn Storage>) -> usize {
    let files = storage.list_files().unwrap();
    files
        .iter()
        .filter(|f| f.starts_with("lexical/") && f.ends_with(".cfs"))
        .count()
}

/// The hits of a search through `Engine`, as `(id, score)`, each checked to
/// have a real id and its document.
async fn engine_hits(engine: &Engine, request: SearchRequestBuilder) -> Vec<(u64, f32)> {
    engine
        .search(request.build())
        .await
        .unwrap()
        .results
        .into_iter()
        .map(|hit| {
            let id = hit
                .id
                .parse()
                .unwrap_or_else(|_| panic!("a hit without its id: {:?}", hit.id));
            assert!(hit.document.is_some(), "hit {id} without its document");
            (id, hit.score)
        })
        .collect()
}

fn request(query: &BooleanQuery) -> SearchRequestBuilder {
    SearchRequestBuilder::new().lexical_query(LexicalSearchQuery::Obj(Box::new(query.clone())))
}

fn sort(order: SortOrder) -> SortField {
    SortField::Field {
        name: "n".to_string(),
        order,
    }
}

/// Checks that `got` holds the expected documents once each. With a limit,
/// it holds that many of them.
fn check(failures: &mut Vec<String>, what: String, got: &[u64], expected: &[u64], limit: usize) {
    let distinct: BTreeSet<u64> = got.iter().copied().collect();
    let all: BTreeSet<u64> = expected.iter().copied().collect();
    let ok = distinct.len() == got.len()
        && got.len() == expected.len().min(limit)
        && distinct.is_subset(&all);
    if !ok {
        failures.push(format!(
            "{what}: {} hits, {} distinct, {} not expected, of {} expected",
            got.len(),
            distinct.len(),
            distinct.difference(&all).count(),
            expected.len()
        ));
    }
}

/// Each shape returns its documents once each, through `Engine`, in parallel
/// and not, with a limit, and sorted by a field both ways.
#[tokio::test(flavor = "multi_thread")]
async fn every_shape_returns_each_document_once() {
    let titles = initial_titles();
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let (engine, _) = engine(segments).await;
        for shape in shapes() {
            let expected = expected(&titles, shape.matches);
            assert!(!expected.is_empty(), "{}", shape.name);
            let mut serial_scores = BTreeMap::new();
            for parallel in [false, true] {
                let what = |way: &str| {
                    format!(
                        "{segments} segments, {}, parallel {parallel}, {way}",
                        shape.name
                    )
                };
                for limit in [10, 1000] {
                    let hits = engine_hits(
                        &engine,
                        request(&shape.query)
                            .lexical_parallel(parallel)
                            .limit(limit),
                    )
                    .await;
                    let ids: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
                    check(
                        &mut failures,
                        what(&format!("limit {limit}")),
                        &ids,
                        &expected,
                        limit,
                    );
                    if limit == 1000 {
                        let scores: BTreeMap<u64, f32> = hits.into_iter().collect();
                        if !parallel {
                            serial_scores = scores;
                        } else if shape.scored
                            && scores.iter().any(|(id, s)| {
                                serial_scores.get(id).is_none_or(|t| (s - t).abs() > 1e-4)
                            })
                        {
                            failures.push(what("scores differ from a serial search"));
                        }
                    }
                }
                for (order, ordered) in [
                    (SortOrder::Asc, expected.clone()),
                    (SortOrder::Desc, expected.iter().rev().copied().collect()),
                ] {
                    for limit in [10, 1000] {
                        let hits = engine_hits(
                            &engine,
                            request(&shape.query)
                                .lexical_parallel(parallel)
                                .sort_by(sort(order))
                                .limit(limit),
                        )
                        .await;
                        let ids: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
                        let first: Vec<u64> = ordered.iter().take(limit).copied().collect();
                        if ids != first {
                            failures.push(what(&format!(
                                "sorted {order:?}, limit {limit}: {ids:?}, not {first:?}"
                            )));
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A filter-only query with a `filter_query`, and a `filter_query` that is a
/// filter-only query.
#[tokio::test(flavor = "multi_thread")]
async fn a_filter_query_and_a_filter_only_query() {
    let titles = initial_titles();
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let (engine, _) = engine(segments).await;
        let filter = || BooleanQueryBuilder::new().filter(term("third")).build();
        let filtered = || request(&filter()).filter_query(term("fifth"));
        let filtering = || {
            SearchRequestBuilder::new()
                .query_dsl("title:every")
                .filter_query(Box::new(filter()))
        };
        for (what, request, expected) in [
            (
                "#third, filter_query fifth",
                &filtered as &dyn Fn() -> SearchRequestBuilder,
                expected(&titles, |w| w.third && w.fifth),
            ),
            (
                "every, filter_query #third",
                &filtering,
                expected(&titles, |w| w.third),
            ),
        ] {
            for parallel in [false, true] {
                let hits =
                    engine_hits(&engine, request().lexical_parallel(parallel).limit(1000)).await;
                let ids: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
                check(
                    &mut failures,
                    format!("{segments} segments, {what}, parallel {parallel}"),
                    &ids,
                    &expected,
                    1000,
                );
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A hybrid search fuses one rank per list for a document, as on one segment.
#[tokio::test(flavor = "multi_thread")]
async fn a_hybrid_search_fuses_each_document_once() {
    let mut fused = Vec::new();
    for segments in [1, 4] {
        let (engine, _) = engine(segments).await;
        let filter = BooleanQueryBuilder::new().filter(term("third")).build();
        let hits = engine_hits(
            &engine,
            request(&filter)
                .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
                    vector: Vector::new(vec![1.0, 1.0]),
                    weight: 1.0,
                    fields: Some(vec!["vector".to_string()]),
                }]))
                .fusion_algorithm(FusionAlgorithm::RRF { k: 60.0 })
                .limit(20),
        )
        .await;
        let distinct: BTreeSet<u64> = hits.iter().map(|(id, _)| *id).collect();
        assert_eq!(distinct.len(), hits.len(), "{segments} segments: {hits:?}");
        fused.push(hits);
    }
    let rounded = |hits: &[(u64, f32)]| -> Vec<(u64, String)> {
        hits.iter()
            .map(|(id, s)| (*id, format!("{s:.6}")))
            .collect()
    };
    assert_eq!(rounded(&fused[0]), rounded(&fused[1]));
}

/// After documents were replaced and deleted, a filter selects the live
/// version of each document only, through `Engine` and `LexicalStore` (which
/// replaces them only, having no public deletion).
#[tokio::test(flavor = "multi_thread")]
async fn a_filter_selects_live_versions_only() {
    // Old and new versions: with `third` and with it (3), with it and without
    // (6, 15), without and with (1, 5); 9 and 10 are deleted.
    let changes: [(u64, Option<&str>); 7] = [
        (3, Some("every third")),
        (6, Some("every")),
        (15, Some("every fifth")),
        (1, Some("every third")),
        (5, Some("every third fifth")),
        (9, None),
        (10, None),
    ];
    let mut titles = initial_titles();
    let mut replaced = initial_titles();
    for (id, change) in changes {
        match change {
            Some(title) => {
                titles.insert(id, title.to_string());
                replaced.insert(id, title.to_string());
            }
            None => {
                titles.remove(&id);
            }
        };
    }
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let (engine, storage) = engine(segments).await;
        for (id, change) in changes {
            match change {
                Some(title) => engine
                    .put_document(&id.to_string(), document(id, title))
                    .await
                    .unwrap(),
                None => engine.delete_documents(&id.to_string()).await.unwrap(),
            }
        }
        engine.commit().await.unwrap();
        assert_eq!(lexical_segments(&storage), segments as usize + 1);
        let lexical = LexicalStore::new(
            Arc::new(MemoryStorage::new(Default::default())),
            LexicalIndexConfig::builder().build(),
        )
        .unwrap();
        for id in 1..=DOCS {
            lexical
                .upsert_document(id, Document::builder().add_text("title", title(id)).build())
                .unwrap();
            if id % (DOCS / segments) == 0 {
                lexical.commit().unwrap();
            }
        }
        for (id, change) in changes {
            if let Some(title) = change {
                lexical
                    .upsert_document(id, Document::builder().add_text("title", title).build())
                    .unwrap();
            }
        }
        lexical.commit().unwrap();
        for shape in shapes() {
            let in_lexical = expected(&replaced, shape.matches);
            let expected = expected(&titles, shape.matches);
            for parallel in [false, true] {
                let what = format!(
                    "{segments} segments and the changes, {}, parallel {parallel}",
                    shape.name
                );
                let hits = engine_hits(
                    &engine,
                    request(&shape.query)
                        .lexical_parallel(parallel)
                        .sort_by(sort(SortOrder::Asc))
                        .limit(1000),
                )
                .await;
                let ids: Vec<u64> = hits.iter().map(|(id, _)| *id).collect();
                if ids != expected {
                    failures.push(format!("{what}, Engine: {ids:?}, not {expected:?}"));
                }
                let found = lexical
                    .search(
                        LexicalSearchRequest::new(Box::new(shape.query.clone()) as Box<dyn Query>)
                            .limit(1000)
                            .parallel(parallel)
                            .sort_by_field_asc("_id"),
                    )
                    .unwrap();
                let mut ids: Vec<u64> = found.hits.iter().map(|hit| hit.doc_id).collect();
                ids.sort_unstable();
                if ids != in_lexical {
                    failures.push(format!("{what}, LexicalStore: {ids:?}, not {in_lexical:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Through `LexicalStore`, each shape returns its documents once each, and the
/// counts are those of its documents.
#[test]
fn a_lexical_store_returns_each_document_once() {
    let titles = initial_titles();
    let mut failures = Vec::new();
    for segments in [1, 4] {
        let store = LexicalStore::new(
            Arc::new(MemoryStorage::new(Default::default())),
            LexicalIndexConfig::builder().build(),
        )
        .unwrap();
        for id in 1..=DOCS {
            store
                .upsert_document(
                    id,
                    Document::builder()
                        .add_text("title", title(id))
                        .add_integer("n", id as i64)
                        .build(),
                )
                .unwrap();
            if id % (DOCS / segments) == 0 {
                store.commit().unwrap();
            }
        }
        for shape in shapes() {
            let expected = expected(&titles, shape.matches);
            let query = || Box::new(shape.query.clone()) as Box<dyn Query>;
            for parallel in [false, true] {
                let what = |way: &str| {
                    format!(
                        "{segments} segments, {}, parallel {parallel}, {way}",
                        shape.name
                    )
                };
                for limit in [10, 1000] {
                    let found = store
                        .search(
                            LexicalSearchRequest::new(query())
                                .limit(limit)
                                .parallel(parallel),
                        )
                        .unwrap();
                    let ids: Vec<u64> = found.hits.iter().map(|hit| hit.doc_id).collect();
                    check(
                        &mut failures,
                        what(&format!("limit {limit}")),
                        &ids,
                        &expected,
                        limit,
                    );
                }
                let found = store
                    .search(
                        LexicalSearchRequest::new(query())
                            .limit(1000)
                            .parallel(parallel)
                            .sort_by_field_asc("n")
                            .track_total_hits(laurus::TrackTotalHits::Exact),
                    )
                    .unwrap();
                let ids: Vec<u64> = found.hits.iter().map(|hit| hit.doc_id).collect();
                if ids != expected || found.total_hits.unwrap().value != expected.len() as u64 {
                    failures.push(what(&format!(
                        "sorted by a field: {} hits, total {}, of {}",
                        ids.len(),
                        found.total_hits.unwrap().value,
                        expected.len()
                    )));
                }
            }
            let count = store.count(LexicalSearchRequest::new(query())).unwrap();
            if count != expected.len() as u64 {
                failures.push(format!(
                    "{segments} segments, {}: count {count}, of {}",
                    shape.name,
                    expected.len()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
