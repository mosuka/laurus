//! Issue #1747: the DSL reads `NOT` as a prohibition, as `-`, and reads
//! `AND`, `OR` and `NOT` only in capitals and as whole words. A keyword in
//! another case (`and`, `Not`), a word that starts with one (`android`,
//! `ANDROID`, `NOTE`), and a keyword with no clause after it are terms.
//!
//! Every case is searched through both DSL routes of `Engine::search`:
//! `query_dsl`, which the unified parser reads, and a lexical DSL query.

use std::sync::Arc;

use laurus::storage::memory::MemoryStorage;
use laurus::{Document, Engine, LexicalSearchQuery, Schema, SearchRequestBuilder, TextOption};

/// The documents; each one's id is its title. The `simple` analyzer keeps
/// stop words and case, so `not`, `NOT`, `and` and `or` are terms of their
/// own. `roid` and `ange` are what `android` and `orange` used to be split
/// into after `AND` and `OR`.
const TITLES: [&str; 16] = [
    "apple",
    "banana",
    "cherry",
    "apple banana",
    "android",
    "orange",
    "notable",
    "ANDROID",
    "ORACLE",
    "NOTE",
    "not",
    "NOT",
    "and",
    "or",
    "roid",
    "ange",
];

/// An engine over a `title` text field, the default field, under the
/// `simple` analyzer, with one document per title.
async fn engine() -> Engine {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default().analyzer("simple"))
        .add_default_field("title")
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    for title in TITLES {
        engine
            .put_document(title, Document::builder().add_text("title", title).build())
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

/// Every title except `excluded`.
fn all_but(excluded: &[&'static str]) -> Vec<&'static str> {
    TITLES
        .into_iter()
        .filter(|title| !excluded.contains(title))
        .collect()
}

/// The cases that do not find exactly the expected titles through one of
/// the two DSL routes, each with what it found or the error.
async fn wrong_cases(engine: &Engine, cases: &[(&str, Vec<&'static str>)]) -> Vec<String> {
    let mut wrong = Vec::new();
    for (dsl, expected) in cases {
        let mut expected: Vec<String> = expected.iter().map(|title| title.to_string()).collect();
        expected.sort();
        let routes = [
            ("query_dsl", SearchRequestBuilder::new().query_dsl(*dsl)),
            (
                "lexical DSL",
                SearchRequestBuilder::new().lexical_query(LexicalSearchQuery::Dsl(dsl.to_string())),
            ),
        ];
        for (route, builder) in routes {
            match engine
                .search(builder.limit(100).build())
                .await
                .map(|response| response.results)
            {
                Ok(hits) => {
                    let mut found: Vec<String> = hits.into_iter().map(|hit| hit.id).collect();
                    found.sort();
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

/// Issue #1747: `NOT` prohibits the clause after it, as `-` does, alone,
/// after a clause, after `AND` or `OR`, and before a group. Each case is
/// listed with the form written with `-`, which finds the same titles.
#[tokio::test(flavor = "multi_thread")]
async fn not_prohibits_the_clause_after_it_as_minus_does() {
    let engine = engine().await;
    let without_apple = all_but(&["apple", "apple banana"]);
    let without_apple_or_banana = all_but(&["apple", "banana", "apple banana"]);
    let wrong = wrong_cases(
        &engine,
        &[
            ("NOT title:apple", without_apple.clone()),
            ("-title:apple", without_apple),
            ("title:banana NOT title:apple", vec!["banana"]),
            ("title:banana -title:apple", vec!["banana"]),
            ("title:banana AND NOT title:apple", vec!["banana"]),
            ("+title:banana -title:apple", vec!["banana"]),
            ("title:banana OR NOT title:apple", vec!["banana"]),
            (
                "NOT (title:apple OR title:banana)",
                without_apple_or_banana.clone(),
            ),
            (
                "NOT(title:apple OR title:banana)",
                without_apple_or_banana.clone(),
            ),
            ("-(title:apple OR title:banana)", without_apple_or_banana),
        ],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1747: the word `not` can still be searched: in lowercase, and in
/// capitals quoted, after a field, alone, or at the end, where no clause
/// follows the keyword.
#[tokio::test(flavor = "multi_thread")]
async fn the_word_not_can_be_searched() {
    let engine = engine().await;
    let wrong = wrong_cases(
        &engine,
        &[
            ("not", vec!["not"]),
            ("title:not", vec!["not"]),
            ("\"NOT\"", vec!["NOT"]),
            ("title:NOT", vec!["NOT"]),
            ("NOT", vec!["NOT"]),
            ("apple NOT", vec!["apple", "apple banana", "NOT"]),
        ],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1747: `and`, `or` and `not` in another case than capitals are
/// terms, so the clauses around them stay optional. `apple and banana`
/// used to require both words.
#[tokio::test(flavor = "multi_thread")]
async fn a_keyword_in_another_case_is_a_term() {
    let engine = engine().await;
    let wrong = wrong_cases(
        &engine,
        &[
            (
                "apple and banana",
                vec!["apple", "banana", "apple banana", "and"],
            ),
            ("apple And banana", vec!["apple", "banana", "apple banana"]),
            (
                "apple or cherry",
                vec!["apple", "apple banana", "cherry", "or"],
            ),
            ("apple Or cherry", vec!["apple", "apple banana", "cherry"]),
            (
                "banana not apple",
                vec!["apple", "banana", "apple banana", "not"],
            ),
            ("banana Not apple", vec!["apple", "banana", "apple banana"]),
        ],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1747: an operator is a whole word, so a word that starts with
/// `AND`, `OR` or `NOT`, in any case, is a term after another clause too.
/// `apple android` used to require `apple` and `roid`, and `apple orange`
/// searched `ange`.
#[tokio::test(flavor = "multi_thread")]
async fn a_word_that_starts_with_a_keyword_is_a_term() {
    let engine = engine().await;
    let wrong = wrong_cases(
        &engine,
        &[
            ("apple android", vec!["apple", "apple banana", "android"]),
            ("apple ANDROID", vec!["apple", "apple banana", "ANDROID"]),
            ("apple orange", vec!["apple", "apple banana", "orange"]),
            ("apple ORACLE", vec!["apple", "apple banana", "ORACLE"]),
            ("apple notable", vec!["apple", "apple banana", "notable"]),
            ("apple NOTE", vec!["apple", "apple banana", "NOTE"]),
        ],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}
