//! Issue #1591: a DSL range bound is a number only when it is written in
//! digits, through both DSL routes of `Engine::search`: `query_dsl`, which
//! the unified parser reads, and a lexical DSL query.
//!
//! Rust's `f64` parser read `nan`, `inf` and `infinity` as numbers, so
//! `price:[nan TO 5]` matched nothing without an error and
//! `price:[-inf TO 5]` searched `price:[* TO 5]`; it turned `1e400` into an
//! infinity; and a bound that was not a number next to a numeric one became
//! open, so `price:[5 TO abc]` searched `price:[5 TO *]`. The words are now
//! strings, a range that mixes a numeric bound with a string is a parse
//! error, and a number too large for an `f64` is an `InvalidArgument`.

use std::sync::Arc;

use laurus::lexical::{Query, TermQuery};
use laurus::storage::memory::MemoryStorage;
use laurus::{
    Document, Engine, FloatOption, LaurusError, LexicalSearchQuery, Schema, SearchRequestBuilder,
    TextOption,
};

/// An engine over a `name` text field and a `price` float field, with
/// `neg` (-1.5), `small` (3.0) and `big` (10.0).
async fn engine() -> Engine {
    let schema = Schema::builder()
        .add_text_field("name", TextOption::default())
        .add_float_field("price", FloatOption::default())
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    for (id, price) in [("neg", -1.5), ("small", 3.0), ("big", 10.0)] {
        let doc = Document::builder()
            .add_text("name", id)
            .add_float("price", price)
            .build();
        engine.put_document(id, doc).await.unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

/// The requests that search `dsl`: through `query_dsl` and as a lexical DSL
/// query, each alone and under a filter that matches nothing, where the
/// search would return before it runs the query. The same routes as
/// `engine_query_dsl_counts_test.rs` (Issue #1570).
fn requests(dsl: &str) -> Vec<(&'static str, SearchRequestBuilder)> {
    let nothing = || Box::new(TermQuery::new("name", "nosuchterm")) as Box<dyn Query>;
    let lexical = || LexicalSearchQuery::Dsl(dsl.to_string());
    vec![
        ("query_dsl", SearchRequestBuilder::new().query_dsl(dsl)),
        (
            "query_dsl under a filter that matches nothing",
            SearchRequestBuilder::new()
                .query_dsl(dsl)
                .filter_query(nothing()),
        ),
        (
            "lexical DSL",
            SearchRequestBuilder::new().lexical_query(lexical()),
        ),
        (
            "lexical DSL under a filter that matches nothing",
            SearchRequestBuilder::new()
                .lexical_query(lexical())
                .filter_query(nothing()),
        ),
    ]
}

/// Issue #1591: a numeric bound next to a bound that is not a number, the
/// words `nan`, `inf` and `infinity` included, is a parse error through
/// every route.
#[tokio::test(flavor = "multi_thread")]
async fn a_numeric_bound_next_to_a_string_bound_is_rejected() {
    let engine = engine().await;
    let expected = "mixes a numeric bound with a bound that is not a number";
    let mut wrong = Vec::new();
    for dsl in [
        "price:[nan TO 5]",
        "price:[-inf TO 5]",
        "price:[0 TO Infinity]",
        "price:[5 TO abc]",
    ] {
        for (route, builder) in requests(dsl) {
            let result = engine.search(builder.build()).await;
            if !matches!(&result, Err(LaurusError::Query(m)) if m.contains(expected)) {
                wrong.push(format!("{route} {dsl}: {:?}", result.map(|r| r.len())));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1591: a number too large for an `f64` is rejected with
/// `InvalidArgument` through every route, instead of becoming an infinite
/// bound.
#[tokio::test(flavor = "multi_thread")]
async fn a_range_bound_too_large_for_an_f64_is_rejected() {
    let engine = engine().await;
    let mut wrong = Vec::new();
    for (dsl, bound) in [
        ("price:[1e400 TO *]", "1e400"),
        ("price:[-1e309 TO 5]", "-1e309"),
    ] {
        let expected = format!("range bound {bound} does not fit a 64-bit float");
        for (route, builder) in requests(dsl) {
            let result = engine.search(builder.build()).await;
            if !matches!(&result, Err(LaurusError::InvalidArgument(m)) if m.contains(&expected)) {
                wrong.push(format!("{route} {dsl}: {:?}", result.map(|r| r.len())));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1591: a number written with a sign, a leading `.` or an exponent
/// is still a numeric bound and searches.
#[tokio::test(flavor = "multi_thread")]
async fn numbers_written_in_digits_still_search() {
    let engine = engine().await;
    let mut wrong = Vec::new();
    for (dsl, hits) in [
        ("price:[-1.5 TO +5]", 2),
        ("price:[.5 TO 1e1]", 2),
        ("price:[* TO 5]", 2),
    ] {
        for (route, builder) in requests(dsl) {
            if route.ends_with("matches nothing") {
                continue;
            }
            let found = engine
                .search(builder.build())
                .await
                .map(|results| results.len());
            if !matches!(found, Ok(n) if n == hits) {
                wrong.push(format!("{route} {dsl}: {found:?}, expected {hits}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
