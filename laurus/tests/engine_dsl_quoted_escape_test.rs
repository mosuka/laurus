//! Issue #1758: every quoted string of the DSL reads its escapes, as
//! Lucene's `QUOTED` does: a backslash takes the next character literally,
//! whatever it is, so `\"` does not end the string, and the backslash is
//! dropped before the analysis or the embedding. A quoted phrase had no
//! escapes: it ended at the next `"`, and a backslash in it was a character
//! of its own; so had a quoted vector value of the unified DSL.
//!
//! The lexical cases are searched through both DSL routes of
//! `Engine::search`: `query_dsl`, which the unified parser reads, and a
//! lexical DSL query.

use std::any::Any;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use laurus::storage::memory::MemoryStorage;
use laurus::vector::{FlatOption, Vector};
use laurus::{
    DataValue, Document, EmbedInput, EmbedInputType, Embedder, Engine, FieldOption, LaurusError,
    LexicalSearchQuery, Result, Schema, SearchRequestBuilder, TextOption,
};

/// The values, one document each, whose id is `v<index>`, in `kw` and `std`.
const VALUES: [&str; 10] = [
    "say \"hi\"",
    "a:b",
    "a\\:b",
    "ab",
    "a\\b",
    "C:\\temp",
    "C:temp",
    "x y",
    "x \" vec:y",
    "a b",
];

/// An embedder that records every text it embeds, and gives each the
/// vector `[1, 0]`.
#[derive(Debug, Default)]
struct RecordingEmbedder {
    texts: Mutex<Vec<String>>,
}

#[async_trait]
impl Embedder for RecordingEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(text) => {
                self.texts.lock().unwrap().push(text.to_string());
                Ok(Vector::new(vec![1.0, 0.0]))
            }
            _ => Err(LaurusError::invalid_argument("text only")),
        }
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "recording"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An engine over `kw`, a text field under the `keyword` analyzer and the
/// default field, `std`, one under the `standard` analyzer, and `vec`, a
/// vector field of 2 dimensions, with one document per value; and its
/// embedder.
async fn engine() -> (Engine, Arc<RecordingEmbedder>) {
    let embedder = Arc::new(RecordingEmbedder::default());
    let schema = Schema::builder()
        .add_text_field("kw", TextOption::default().analyzer("keyword"))
        .add_text_field("std", TextOption::default().analyzer("standard"))
        .add_field("vec", FieldOption::Flat(FlatOption::default().dimension(2)))
        .add_default_field("kw")
        .build();
    let engine = Engine::builder(Arc::new(MemoryStorage::new(Default::default())), schema)
        .embedder(embedder.clone())
        .build()
        .await
        .unwrap();
    for (i, value) in VALUES.iter().enumerate() {
        engine
            .put_document(
                &format!("v{i}"),
                Document::builder()
                    .add_text("kw", *value)
                    .add_text("std", *value)
                    .add_field("vec", DataValue::Vector(vec![0.0, 1.0]))
                    .build(),
            )
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    embedder.texts.lock().unwrap().clear();
    (engine, embedder)
}

fn sorted(mut values: Vec<&str>) -> Vec<&str> {
    values.sort();
    values
}

fn found(hits: &[laurus::SearchResult]) -> Vec<&'static str> {
    sorted(
        hits.iter()
            .map(|hit| VALUES[hit.id[1..].parse::<usize>().unwrap()])
            .collect(),
    )
}

/// The lexical cases that do not find exactly the expected values through
/// one of the two DSL routes, each with what it found or the error.
async fn wrong_lexical_cases(cases: &[(&str, Vec<&'static str>)]) -> Vec<String> {
    let (engine, _) = engine().await;
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
            match engine
                .search(builder.limit(100).build())
                .await
                .map(|response| response.results)
            {
                Ok(hits) => {
                    if found(&hits) != expected {
                        wrong.push(format!(
                            "'{dsl}' through {route}: found {:?}, expected {expected:?}",
                            found(&hits)
                        ));
                    }
                }
                Err(e) => wrong.push(format!("'{dsl}' through {route}: {e}")),
            }
        }
    }
    wrong
}

/// Issue #1758: a quoted phrase reads its escapes: `\"` is a quote in it,
/// and a backslash before any other character is dropped, so a backslash
/// itself is written `\\`.
#[tokio::test(flavor = "multi_thread")]
async fn a_phrase_reads_its_escapes() {
    let wrong = wrong_lexical_cases(&[
        (r#"kw:"say \"hi\"""#, vec!["say \"hi\""]),
        (r#""say \"hi\"""#, vec!["say \"hi\""]),
        (r#"kw:"a\:b""#, vec!["a:b"]),
        (r#"kw:"a\b""#, vec!["ab"]),
        (r#"kw:"a\\b""#, vec!["a\\b"]),
        (r#"kw:"C:\\temp""#, vec!["C:\\temp"]),
        (r#"kw:"C:\temp""#, vec!["C:temp"]),
    ])
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1758: a phrase without a backslash is read as before, with its
/// proximity and boost, and so is one on a field whose analyzer splits on
/// punctuation, which drops the backslash either way.
#[tokio::test(flavor = "multi_thread")]
async fn a_phrase_without_an_escape_is_read_as_before() {
    let wrong = wrong_lexical_cases(&[
        (r#"kw:"x y"~1"#, vec!["x y"]),
        (r#"kw:"ab"^2"#, vec!["ab"]),
        (r#"std:"x y""#, vec!["x y"]),
        (r#"std:"a\:b""#, vec!["a:b", "a\\:b", "a\\b", "a b"]),
    ])
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1758: a phrase whose last `"` is escaped is not closed, and the
/// query fails to parse, as in Lucene. It used to be the phrase `say \`.
#[tokio::test(flavor = "multi_thread")]
async fn an_unclosed_phrase_is_a_parse_error() {
    let (engine, _) = engine().await;
    let dsl = r#"kw:"say \""#;
    let mut wrong = Vec::new();
    let routes = [
        ("query_dsl", SearchRequestBuilder::new().query_dsl(dsl)),
        (
            "lexical DSL",
            SearchRequestBuilder::new().lexical_query(LexicalSearchQuery::Dsl(dsl.to_string())),
        ),
    ];
    for (route, builder) in routes {
        if let Ok(hits) = engine
            .search(builder.limit(100).build())
            .await
            .map(|response| response.results)
        {
            wrong.push(format!("'{dsl}' through {route}: found {:?}", found(&hits)));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1758: a quoted vector value reads its escapes as a phrase does,
/// and is embedded without the backslashes; a phrase with `\"` beside a
/// vector clause, or with a vector-like text in it, is one phrase.
#[tokio::test(flavor = "multi_thread")]
async fn a_quoted_vector_value_reads_its_escapes() {
    let (engine, embedder) = engine().await;
    let mut wrong = Vec::new();
    for (dsl, expected_hits, expected_texts) in [
        (r#"vec:"say \"hi\"""#, None, vec!["say \"hi\""]),
        (r#"vec:"a\:b""#, None, vec!["a:b"]),
        (
            r#"kw:"say \"hi\"" +vec:cats"#,
            Some(vec!["say \"hi\""]),
            vec!["cats"],
        ),
        (r#"kw:"x \" vec:y""#, Some(vec!["x \" vec:y"]), vec![]),
    ] {
        embedder.texts.lock().unwrap().clear();
        let result = engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl(dsl)
                    .limit(100)
                    .build(),
            )
            .await
            .map(|response| response.results);
        let texts = embedder.texts.lock().unwrap().clone();
        match result {
            Ok(hits) => {
                if texts != expected_texts {
                    wrong.push(format!(
                        "'{dsl}': embedded {texts:?}, expected {expected_texts:?}"
                    ));
                }
                match expected_hits {
                    Some(expected) if found(&hits) != expected => wrong.push(format!(
                        "'{dsl}': found {:?}, expected {expected:?}",
                        found(&hits)
                    )),
                    _ => {}
                }
            }
            Err(e) => wrong.push(format!("'{dsl}': {e}")),
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
