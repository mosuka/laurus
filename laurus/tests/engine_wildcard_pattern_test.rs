//! Issue #1745: a wildcard pattern reads as in Lucene's `WildcardQuery`. A
//! backslash takes the next character literally, whatever it is, and at the
//! end of a pattern is a backslash itself; `*` and `?` match any characters,
//! line breaks included; and no pattern is an error, short of one too long
//! for the regexp engine. A wildcard of the DSL
//! takes the characters of a term, `-`, `'` and an escaped special character
//! among them, and passes the pattern on as written.
//!
//! The values are whole terms (the `keyword` analyzer), so a pattern is
//! compared with each value as a whole. The cases that miss are gathered and
//! asserted at once.

use std::sync::Arc;

use laurus::lexical::Query;
use laurus::lexical::query::wildcard::WildcardQuery;
use laurus::storage::memory::MemoryStorage;
use laurus::{Document, Engine, LexicalSearchQuery, Schema, SearchRequestBuilder, TextOption};

/// The values that the patterns of the Rust API are searched in, those of
/// the table in the Issue among them.
const API_VALUES: [&str; 19] = [
    "tedt", "te1t", "te\\dt", "abc", "a-c", "anb", "a\nb", "a\nc", "axb", "atb", "a\tb", "te\\",
    "te$", "te$x", "te?", "apb", "te*", "tex", "a\\b",
];

/// The values that the DSL is searched in.
const DSL_VALUES: [&str; 17] = [
    "ab cd", "ab\ncd", "te*t", "te*tx", "te\\*t", "tedt", "te1t", "a-b", "a-bx", "it's", "it'sx",
    "a:b", "a:bx", "foo-bar", "foox", "foox-bar", "bar",
];

/// An engine over `title`, a text field under the `keyword` analyzer and the
/// default field, with one document per value, whose id is `v<index>`.
async fn engine(values: &[&str]) -> Engine {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default().analyzer("keyword"))
        .add_default_field("title")
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    for (i, value) in values.iter().enumerate() {
        engine
            .put_document(
                &format!("v{i}"),
                Document::builder().add_text("title", *value).build(),
            )
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

/// The value of the document `id`.
fn value_of<'a>(values: &[&'a str], id: &str) -> &'a str {
    values[id[1..].parse::<usize>().unwrap()]
}

fn sorted(mut values: Vec<&str>) -> Vec<&str> {
    values.sort();
    values
}

/// The patterns that do not match, find and highlight exactly the expected
/// values of [`API_VALUES`] through the Rust API, each with what it did.
async fn wrong_patterns(cases: &[(&str, Vec<&'static str>)]) -> Vec<String> {
    let engine = engine(&API_VALUES).await;
    let mut wrong = Vec::new();
    for (pattern, expected) in cases {
        let expected = sorted(expected.clone());
        let query = match WildcardQuery::new("title", *pattern) {
            Ok(query) => query,
            Err(e) => {
                wrong.push(format!("{pattern:?}: {e}"));
                continue;
            }
        };
        let matched = sorted(
            API_VALUES
                .iter()
                .copied()
                .filter(|value| query.matches(value))
                .collect(),
        );
        let request = SearchRequestBuilder::new()
            .lexical_query(LexicalSearchQuery::Obj(Box::new(query) as Box<dyn Query>))
            .highlight(vec!["title".to_string()])
            .limit(100)
            .build();
        let hits = match engine.search(request).await {
            Ok(hits) => hits,
            Err(e) => {
                wrong.push(format!("{pattern:?}: search failed: {e}"));
                continue;
            }
        };
        let found = sorted(
            hits.iter()
                .map(|hit| value_of(&API_VALUES, &hit.id))
                .collect(),
        );
        let highlighted = sorted(
            hits.iter()
                .filter(|hit| {
                    hit.highlights.get("title").is_some_and(|fragments| {
                        fragments.iter().any(|fragment| fragment.contains("<mark>"))
                    })
                })
                .map(|hit| value_of(&API_VALUES, &hit.id))
                .collect(),
        );
        for (what, got) in [
            ("matches()", matched),
            ("found", found),
            ("highlighted", highlighted),
        ] {
            if got != expected {
                wrong.push(format!(
                    "{pattern:?}: {what} {got:?}, expected {expected:?}"
                ));
            }
        }
    }
    wrong
}

/// The DSL queries that do not find exactly the expected values of
/// [`DSL_VALUES`] through one of the two DSL routes of `Engine::search`:
/// `query_dsl`, which the unified parser reads, and a lexical DSL query.
async fn wrong_dsl_cases(engine: &Engine, cases: &[(&str, Vec<&'static str>)]) -> Vec<String> {
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
                            .map(|hit| value_of(&DSL_VALUES, &hit.id))
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

/// Issue #1745: a backslash takes the next character literally, whatever it
/// is, and at the end of a pattern is a backslash itself. The rows of the
/// table in the Issue, and `\a`, the bell character in a regexp.
#[tokio::test(flavor = "multi_thread")]
async fn a_backslash_takes_the_next_character_literally() {
    let wrong = wrong_patterns(&[
        (r"te\dt", vec!["tedt"]),
        (r"a\bc", vec!["abc"]),
        (r"\abc", vec!["abc"]),
        (r"a\nb", vec!["anb"]),
        (r"a\tb", vec!["atb"]),
        (r"te\", vec!["te\\"]),
        (r"a\pb", vec!["apb"]),
        (r"a\xb", vec!["axb"]),
        (r"te\*", vec!["te*"]),
        (r"te\?", vec!["te?"]),
        (r"a\\b", vec!["a\\b"]),
    ])
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1745: `*` and `?` match any characters, a line break too, through
/// the Rust API and through the DSL.
#[tokio::test(flavor = "multi_thread")]
async fn star_and_question_mark_match_a_line_break() {
    let three_letters = vec!["anb", "a\nb", "axb", "atb", "a\tb", "apb", "a\\b"];
    let mut wrong = wrong_patterns(&[
        ("a*b", three_letters.clone()),
        ("a?b", three_letters),
        ("a?c", vec!["abc", "a-c", "a\nc"]),
    ])
    .await;
    let engine = engine(&DSL_VALUES).await;
    wrong.extend(
        wrong_dsl_cases(
            &engine,
            &[
                ("title:ab*", vec!["ab cd", "ab\ncd"]),
                ("title:ab?cd", vec!["ab cd", "ab\ncd"]),
                ("title:ab*cd", vec!["ab cd", "ab\ncd"]),
            ],
        )
        .await,
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1745: no pattern is an error. Each is read as in Lucene: it
/// matches the string it spells, with `\` taking the next character, and
/// not what it would match as a regexp.
#[test]
fn no_pattern_is_an_error() {
    let mut wrong = Vec::new();
    for (pattern, matching, not_matching) in [
        (r"a\pb", "apb", "a"),
        (r"a\xb", "axb", "a"),
        (r"\p{L}", "p{L}", "x"),
        (r"\", "\\", ""),
        (r"\\\", "\\\\", "\\"),
        ("(", "(", ""),
        ("(a)", "(a)", "a"),
        ("[a", "[a", "a"),
        ("[[:alpha:]]", "[[:alpha:]]", "x"),
        ("{2}", "{2}", "2"),
        ("a|b", "a|b", "a"),
        ("^$", "^$", ""),
        ("a.b", "a.b", "axb"),
        ("a+", "a+", "aa"),
        ("#&~-", "#&~-", ""),
    ] {
        match WildcardQuery::new("title", pattern) {
            Ok(query) => {
                if !query.matches(matching) || query.matches(not_matching) {
                    wrong.push(format!(
                        "{pattern:?}: matches {matching:?} {}, matches {not_matching:?} {}",
                        query.matches(matching),
                        query.matches(not_matching)
                    ));
                }
            }
            Err(e) => wrong.push(format!("{pattern:?}: {e}")),
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1745: a wildcard of the DSL takes the characters of a term, `-`,
/// `'` and an escaped special character among them, and is one wildcard
/// with them, where it used to end before them and leave a bare `*`, which
/// matched every value of the default field. `foo*-bar` is one wildcard
/// too, as in Lucene, no longer `foo*` and a prohibited `bar`.
#[tokio::test(flavor = "multi_thread")]
async fn a_dsl_wildcard_takes_the_characters_of_a_term() {
    let engine = engine(&DSL_VALUES).await;
    let wrong = wrong_dsl_cases(
        &engine,
        &[
            ("title:a-b*", vec!["a-b", "a-bx"]),
            (r"title:a\-b*", vec!["a-b", "a-bx"]),
            ("title:it's*", vec!["it's", "it'sx"]),
            (r"title:te\*t*", vec!["te*t", "te*tx"]),
            (r"title:a\:b*", vec!["a:b", "a:bx"]),
            ("title:foo*-bar", vec!["foo-bar", "foox-bar"]),
            ("title:te?t", vec!["te*t", "tedt", "te1t"]),
        ],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}
