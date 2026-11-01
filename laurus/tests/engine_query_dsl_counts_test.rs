//! Issue #1570: a DSL phrase slop or fuzziness too large for a `u32` is
//! rejected with `InvalidArgument`, instead of silently becoming `0` (an
//! exact phrase) or `2`, through both DSL routes of `Engine::search`:
//! `query_dsl`, which the unified parser reads, and a lexical DSL query.
//! A fuzziness is a `u32` like `FuzzyQuery::max_edits`, so a distance of
//! 256 or more applies instead of becoming `2`.

use std::sync::Arc;

use laurus::lexical::{Query, TermQuery};
use laurus::storage::memory::MemoryStorage;
use laurus::{
    Document, Engine, LaurusError, LexicalSearchQuery, Schema, SearchRequestBuilder, TextOption,
};

/// An engine over a `title` text field with `d1` ("hello big world") and
/// `d2` ("rust").
async fn engine() -> Engine {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    for (id, title) in [("d1", "hello big world"), ("d2", "rust")] {
        engine
            .put_document(id, Document::builder().add_text("title", title).build())
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

/// The requests that search `dsl`: through `query_dsl` and as a lexical DSL
/// query, each alone and under a filter that matches nothing, where the
/// search would return before it runs the query.
fn requests(dsl: &str) -> Vec<(&'static str, SearchRequestBuilder)> {
    let nothing = || Box::new(TermQuery::new("title", "nosuchterm")) as Box<dyn Query>;
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

/// Issue #1570: a slop or a fuzziness that does not fit a `u32` is rejected
/// with a message naming it, through every route.
#[tokio::test(flavor = "multi_thread")]
async fn a_slop_or_fuzziness_too_large_for_a_u32_is_rejected() {
    let engine = engine().await;
    let mut wrong = Vec::new();
    for (dsl, expected) in [
        (
            "title:\"hello world\"~4294967296",
            "phrase slop must be at most 4294967295, got 4294967296",
        ),
        (
            "title:\"hello world\"~99999999999",
            "phrase slop must be at most 4294967295, got 99999999999",
        ),
        (
            "title:rusk~4294967296",
            "fuzziness must be at most 4294967295, got 4294967296",
        ),
    ] {
        for (route, builder) in requests(dsl) {
            let result = engine.search(builder.build()).await;
            if !matches!(&result, Err(LaurusError::InvalidArgument(m)) if m.contains(expected)) {
                wrong.push(format!("{route} {dsl}: {:?}", result.map(|r| r.len())));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1570: the largest slop and fuzziness search. A fuzziness of 300 is
/// 300 edits, so `rusk` matches every term of the field, `hello`, `big` and
/// `world` included, where it used to become `2` and match `rust` alone.
#[tokio::test(flavor = "multi_thread")]
async fn the_largest_slop_and_a_large_fuzziness_search() {
    let engine = engine().await;
    let mut wrong = Vec::new();
    for (dsl, hits) in [
        ("title:\"hello world\"~4294967295", 1),
        ("title:rusk~2", 1),
        ("title:rusk~300", 2),
        ("title:rusk~4294967295", 2),
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
