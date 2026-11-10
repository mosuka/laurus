//! Issue #1601: a `NumericRangeQuery` with a `NaN` bound is rejected with
//! `InvalidArgument` at search, on every route a query can take.
//!
//! A `NaN` bound compares false with every value, so the range used to match
//! nothing without an error, and an `Integer` range truncated it to 0 and
//! searched `[0, ..]`. The bindings build their range queries through the
//! same constructors, so this covers them too once their bounds reach the
//! core. An infinite bound is accepted and compares as a number.

use std::sync::Arc;

use laurus::lexical::core::field::FieldOption;
use laurus::lexical::query::BooleanQueryBuilder;
use laurus::lexical::{
    LexicalIndexConfig, LexicalSearchRequest, LexicalStore, NumericRangeQuery, NumericType, Query,
    TermQuery,
};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{
    Document, Engine, FloatOption, LaurusError, LexicalSearchQuery, Schema, SearchRequestBuilder,
    TextOption,
};

/// The documents: `neg` (-1.5), `small` (3.0) and `big` (10.0).
const DOCS: [(&str, f64); 3] = [("neg", -1.5), ("small", 3.0), ("big", 10.0)];

/// An engine over a `name` text field and a `price` float field holding
/// [`DOCS`].
async fn engine() -> Engine {
    let schema = Schema::builder()
        .add_text_field("name", TextOption::default())
        .add_float_field("price", FloatOption::default())
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    for (id, price) in DOCS {
        let doc = Document::builder()
            .add_text("name", id)
            .add_float("price", price)
            .build();
        engine.put_document(id, doc).await.unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

/// Builds a fresh range query, so each route gets its own.
type BuildRange = fn() -> NumericRangeQuery;

/// The ranges with a `NaN` bound, by name.
fn nan_ranges() -> Vec<(&'static str, BuildRange)> {
    vec![
        ("Float [NaN, 5]", || {
            NumericRangeQuery::f64_range("price", Some(f64::NAN), Some(5.0))
        }),
        ("Float [*, NaN]", || {
            NumericRangeQuery::f64_range("price", None, Some(f64::NAN))
        }),
        ("Integer [NaN, 5]", || {
            NumericRangeQuery::new(
                "price",
                NumericType::Integer,
                Some(f64::NAN),
                Some(5.0),
                true,
                true,
            )
        }),
    ]
}

/// The requests that carry `range`: as the query, as a filter, as a `Must`
/// and a `Filter` clause of a `BooleanQuery`, and as the query under a
/// filter that matches nothing, where the search would return before it
/// runs the query.
fn requests(range: BuildRange) -> Vec<(&'static str, SearchRequestBuilder)> {
    let query =
        |q: Box<dyn Query>| SearchRequestBuilder::new().lexical_query(LexicalSearchQuery::Obj(q));
    let term = |t: &str| Box::new(TermQuery::new("name", t)) as Box<dyn Query>;
    vec![
        ("the query", query(Box::new(range()))),
        (
            "a filter",
            query(term("small")).filter_query(Box::new(range())),
        ),
        (
            "a Must clause",
            query(Box::new(
                BooleanQueryBuilder::new()
                    .must(term("small"))
                    .must(Box::new(range()))
                    .build(),
            )),
        ),
        (
            "a Filter clause",
            query(Box::new(
                BooleanQueryBuilder::new()
                    .must(term("small"))
                    .filter(Box::new(range()))
                    .build(),
            )),
        ),
        (
            "the query under a filter that matches nothing",
            query(Box::new(range())).filter_query(term("nosuchterm")),
        ),
    ]
}

/// Issue #1601: a `NaN` bound is `InvalidArgument` through `Engine::search`,
/// on every route.
#[tokio::test(flavor = "multi_thread")]
async fn a_nan_bound_is_rejected_by_engine_search() {
    let engine = engine().await;
    let mut wrong = Vec::new();
    for (name, range) in nan_ranges() {
        for (route, request) in requests(range) {
            let result = engine.search(request.build()).await;
            if !matches!(
                &result,
                Err(LaurusError::InvalidArgument(m)) if m.contains("field 'price' has a NaN bound")
            ) {
                wrong.push(format!("{name} as {route}: {:?}", result.map(|r| r.len())));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1601: a `NaN` bound is `InvalidArgument` through the lexical
/// searcher's two entry points, `LexicalStore::search` and
/// `LexicalStore::count`.
#[test]
fn a_nan_bound_is_rejected_by_the_lexical_search_and_count() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field("price", FieldOption::Float(FloatOption::default()))
        .build();
    let store = LexicalStore::new(storage, config).unwrap();
    for (doc_id, (_, price)) in (1u64..).zip(DOCS) {
        store
            .upsert_document(
                doc_id,
                Document::builder().add_float("price", price).build(),
            )
            .unwrap();
    }
    store.commit().unwrap();

    fn expected<T>(result: &Result<T, LaurusError>) -> bool {
        matches!(
            result,
            Err(LaurusError::InvalidArgument(m)) if m.contains("field 'price' has a NaN bound")
        )
    }
    let mut wrong = Vec::new();
    for (name, range) in nan_ranges() {
        let search = store
            .search(LexicalSearchRequest::new(Box::new(range())))
            .map(|r| r.hits.len());
        if !expected(&search) {
            wrong.push(format!("search {name}: {search:?}"));
        }
        let count = store.count(LexicalSearchRequest::new(Box::new(range())));
        if !expected(&count) {
            wrong.push(format!("count {name}: {count:?}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1601: an infinite bound is accepted and compares as a number:
/// `-INFINITY` below and `INFINITY` above find what an open side finds, for
/// `Float` and for `Integer`, which turns them into `i64::MIN` and
/// `i64::MAX`.
#[tokio::test(flavor = "multi_thread")]
async fn an_infinite_bound_is_the_same_as_an_open_side() {
    let engine = engine().await;
    let inf = f64::INFINITY;
    let integer = |lower, upper| {
        NumericRangeQuery::new("price", NumericType::Integer, lower, upper, true, true)
    };
    let cases = [
        (
            "Float [-inf, 5]",
            NumericRangeQuery::f64_range("price", Some(-inf), Some(5.0)),
            vec!["neg", "small"],
        ),
        (
            "Float [0, inf]",
            NumericRangeQuery::f64_range("price", Some(0.0), Some(inf)),
            vec!["big", "small"],
        ),
        (
            "Float [inf, *]",
            NumericRangeQuery::f64_range("price", Some(inf), None),
            vec![],
        ),
        (
            "Integer [-inf, 5]",
            integer(Some(-inf), Some(5.0)),
            vec!["neg", "small"],
        ),
        (
            "Integer [0, inf]",
            integer(Some(0.0), Some(inf)),
            vec!["big", "small"],
        ),
    ];
    let mut wrong = Vec::new();
    for (name, range, want) in cases {
        let found = engine
            .search(
                SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::Obj(Box::new(range)))
                    .build(),
            )
            .await
            .map(|results| {
                let mut ids: Vec<String> = results.into_iter().map(|r| r.id).collect();
                ids.sort();
                ids
            });
        if !matches!(&found, Ok(ids) if *ids == want) {
            wrong.push(format!("{name}: {found:?}, want {want:?}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
