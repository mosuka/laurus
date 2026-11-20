//! Issue #1568: a DSL group's boost multiplies the boosts inside it,
//! whatever the number of clauses or the nesting, as in Lucene.
//!
//! A group of one clause, and the outer group of a nested group, used to
//! replace the boost it held: `(title:hello^2)^3` boosted by 3 while
//! `(title:hello^2 title:world)^3` boosted `hello` by 6, and
//! `((title:hello^2)^3)^4` boosted by 4. The tests search one document
//! through the engine's DSL and compare each score with the score of
//! `title:hello`.

use std::sync::Arc;

use laurus::storage::memory::MemoryStorage;
use laurus::{Document, Engine, LaurusError, Result, Schema};
use laurus::{SearchRequestBuilder, TextOption};

/// An engine holding one document whose `title` is `hello`.
async fn engine() -> Engine {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    engine
        .put_document("d1", Document::builder().add_text("title", "hello").build())
        .await
        .unwrap();
    engine.commit().await.unwrap();
    engine
}

/// The score of the one hit of `dsl`.
async fn search(engine: &Engine, dsl: &str) -> Result<f32> {
    let hits = engine
        .search(SearchRequestBuilder::new().query_dsl(dsl).limit(10).build())
        .await?;
    assert_eq!(hits.len(), 1, "{dsl}");
    Ok(hits[0].score)
}

/// Every boosted group multiplies the boost of what it holds into the
/// score: `title:nosuchterm` matches nothing, so it adds nothing.
#[tokio::test(flavor = "multi_thread")]
async fn a_group_s_boost_multiplies_the_score_of_what_it_holds() {
    let engine = engine().await;
    let base = search(&engine, "title:hello").await.unwrap();
    assert!(base > 0.0, "precondition: the document scores");

    let mut wrong = Vec::new();
    for (dsl, multiplier) in [
        ("(title:hello^2)^3", 6.0_f32),
        ("(title:hello^2)^1", 2.0),
        ("(title:hello^2 title:nosuchterm)^3", 6.0),
        ("((title:hello^2)^3)^4", 24.0),
        ("((title:hello title:nosuchterm)^2)^3", 6.0),
        ("((title:hello^2 title:nosuchterm)^3)^4", 24.0),
        ("(+title:hello^2)^3", 6.0),
    ] {
        let ratio = search(&engine, dsl).await.unwrap() / base;
        if ((ratio - multiplier) / multiplier).abs() > 1e-5 {
            wrong.push(format!("{dsl}: {ratio} times, expected {multiplier}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Two boosts whose product is too large for an `f32` are rejected, as one
/// boost that large is (Issue #1550), instead of the inner one being
/// dropped.
#[tokio::test(flavor = "multi_thread")]
async fn a_product_of_boosts_too_large_for_an_f32_is_rejected() {
    let engine = engine().await;
    // 1e20 each, 1e40 together; an `f32` holds up to about 3.4e38.
    let big = "100000000000000000000";
    let dsl = format!("(title:hello^{big})^{big}");
    let result = search(&engine, &dsl).await;
    assert!(
        matches!(&result, Err(LaurusError::InvalidArgument(m)) if m.contains("boost")),
        "{dsl}: {result:?}"
    );
}
