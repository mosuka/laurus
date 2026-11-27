//! Issue #1754: a plain or fuzzy term of the DSL drops the backslash of each
//! escape before the analyzer reads it, as Lucene's `discardEscapeChar`
//! does, so `title:a\:b` searches the term `a:b`. The backslash used to be
//! kept, which showed on a field whose analyzer keeps a value whole: the
//! term had the backslash in it and found only a value that had one too.
//!
//! Every case is searched through both DSL routes of `Engine::search`:
//! `query_dsl`, which the unified parser reads, and a lexical DSL query.

use std::sync::Arc;

use laurus::storage::memory::MemoryStorage;
use laurus::{Document, Engine, LexicalSearchQuery, Schema, SearchRequestBuilder, TextOption};

/// The values, one document each, whose id is `v<index>`, in both fields.
const VALUES: [&str; 9] = [
    "a:b", "te*t", "a\\:b", "te\\*t", "a-b", "(x)", "a b", "abc", "a\\b",
];

/// An engine over `kw`, a text field under the `keyword` analyzer and the
/// default field, and `std`, one under the `standard` analyzer, which splits
/// on punctuation, with one document per value in both.
async fn engine() -> Engine {
    let schema = Schema::builder()
        .add_text_field("kw", TextOption::default().analyzer("keyword"))
        .add_text_field("std", TextOption::default().analyzer("standard"))
        .add_default_field("kw")
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    for (i, value) in VALUES.iter().enumerate() {
        engine
            .put_document(
                &format!("v{i}"),
                Document::builder()
                    .add_text("kw", *value)
                    .add_text("std", *value)
                    .build(),
            )
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

fn sorted(mut values: Vec<&str>) -> Vec<&str> {
    values.sort();
    values
}

/// The cases that do not find exactly the expected values through one of
/// the two DSL routes, each with what it found or the error.
async fn wrong_cases(cases: &[(&str, Vec<&'static str>)]) -> Vec<String> {
    let engine = engine().await;
    let mut wrong = Vec::new();
    for (dsl, expected) in cases {
        let expected = sorted(expected.clone());
        let routes = [
            ("query_dsl", SearchRequestBuilder::new().query_dsl(*dsl)),
            (
                "lexical DSL",
                SearchRequestBuilder::new().lexical_query(LexicalSearchQuery::Dsl(dsl.to_string())),
            ),
        ];
        for (route, builder) in routes {
            match engine.search(builder.limit(100).build()).await {
                Ok(hits) => {
                    let found = sorted(
                        hits.iter()
                            .map(|hit| VALUES[hit.id[1..].parse::<usize>().unwrap()])
                            .collect(),
                    );
                    if found != expected {
                        wrong.push(format!(
                            "'{dsl}' through {route}: found {found:?}, expected {expected:?}"
                        ));
                    }
                }
                Err(e) => wrong.push(format!("'{dsl}' through {route}: {e}")),
            }
        }
    }
    wrong
}

/// Issue #1754: a plain term drops the backslash of each escape, with a
/// field and on the default field, an escaped backslash too.
#[tokio::test(flavor = "multi_thread")]
async fn a_term_drops_the_backslash_of_an_escape() {
    let wrong = wrong_cases(&[
        (r"kw:a\:b", vec!["a:b"]),
        (r"a\:b", vec!["a:b"]),
        (r"kw:te\*t", vec!["te*t"]),
        (r"kw:\(x\)", vec!["(x)"]),
        (r"kw:a\-b", vec!["a-b"]),
        (r"kw:a\\b", vec!["a\\b"]),
        (r"kw:a\\\:b", vec!["a\\:b"]),
    ])
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1754: a fuzzy term drops the backslash of each escape too.
#[tokio::test(flavor = "multi_thread")]
async fn a_fuzzy_term_drops_the_backslash_of_an_escape() {
    let wrong = wrong_cases(&[
        (r"kw:a\:b~0", vec!["a:b"]),
        (r"kw:te\*t~0", vec!["te*t"]),
        (r"kw:\(x\)~0", vec!["(x)"]),
    ])
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1754: a term without a backslash is read as before, and so is an
/// escaped term on a field whose analyzer splits on punctuation, which drops
/// the backslash either way.
#[tokio::test(flavor = "multi_thread")]
async fn a_term_without_an_escape_and_a_split_one_are_read_as_before() {
    let wrong = wrong_cases(&[
        ("kw:a-b", vec!["a-b"]),
        ("kw:abc~1", vec!["abc"]),
        (r"std:a\:b", vec!["a:b", "a\\:b", "a-b", "a b", "a\\b"]),
        (r"std:te\*t", vec!["te*t", "te\\*t"]),
    ])
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}
