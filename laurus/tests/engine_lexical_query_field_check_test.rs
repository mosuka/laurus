//! Issue #1635: a lexical query (`SearchQuery::Lexical`, `Obj` or `Dsl`,
//! and the lexical part of a hybrid search) and a `filter_query` are checked
//! against the schema's lexical fields and `_id`, as the unified DSL checks
//! its fields. A field the schema does not have, or a vector field, used to
//! give no hits as if nothing matched.

use std::sync::Arc;

use laurus::lexical::query::advanced_query::MultiFieldQuery;
use laurus::lexical::query::span::{SpanNearQuery, SpanQuery, SpanQueryWrapper, SpanTermQuery};
use laurus::lexical::{Query, TermQuery, TextOption};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::{FlatOption, Vector};
use laurus::{
    DataValue, Document, Engine, FieldOption, LaurusError, LexicalSearchQuery, QueryVector, Schema,
    SearchRequest, SearchRequestBuilder, VectorSearchQuery,
};

/// `title` and `body` (Text, both default fields) and `vec` (Flat), with
/// one document committed.
async fn engine() -> Engine {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("body", FieldOption::Text(TextOption::default()))
        .add_field("vec", FieldOption::Flat(FlatOption::default().dimension(3)))
        .add_default_field("title")
        .add_default_field("body")
        .build();
    let engine = Engine::new(storage, schema).await.unwrap();
    let doc = Document::builder()
        .add_field("title", DataValue::Text("hello".into()))
        .add_field("body", DataValue::Text("world".into()))
        .add_field("vec", DataValue::Vector(vec![1.0, 0.0, 0.0]))
        .build();
    engine.put_document("a", doc).await.unwrap();
    engine.commit().await.unwrap();
    engine
}

fn obj(query: impl Query + 'static) -> LexicalSearchQuery {
    LexicalSearchQuery::Obj(Box::new(query))
}

fn vectors() -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0, 0.0, 0.0]),
        weight: 1.0,
        fields: Some(vec!["vec".to_string()]),
    }])
}

/// What a request is expected to give.
enum Expect {
    /// An `InvalidArgument` whose message contains this text.
    Rejected(&'static str),
    /// This many hits.
    Hits(usize),
}

const UNKNOWN: &str = "unknown field(s) [\"nosuch\"]";
const NOT_LEXICAL: &str = "field(s) [\"vec\"] that are not a lexical field";

fn requests() -> Vec<(&'static str, SearchRequest, Expect)> {
    let child_on_unknown: Vec<Box<dyn SpanQuery>> = vec![
        Box::new(SpanTermQuery::new("title", "hello")),
        Box::new(SpanTermQuery::new("nosuch", "x")),
    ];
    vec![
        (
            "Lexical(Obj) on an unknown field",
            SearchRequestBuilder::new()
                .lexical_query(obj(TermQuery::new("nosuch", "hello")))
                .build(),
            Expect::Rejected(UNKNOWN),
        ),
        (
            "Lexical(Dsl) on an unknown field",
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Dsl("nosuch:hello".into()))
                .build(),
            Expect::Rejected(UNKNOWN),
        ),
        (
            "a hybrid search's lexical query on an unknown field",
            SearchRequestBuilder::new()
                .lexical_query(obj(TermQuery::new("nosuch", "hello")))
                .vector_query(vectors())
                .build(),
            Expect::Rejected(UNKNOWN),
        ),
        (
            "filter_query on an unknown field",
            SearchRequestBuilder::new()
                .query_dsl("title:hello")
                .filter_query(Box::new(TermQuery::new("nosuch", "x")))
                .build(),
            Expect::Rejected(UNKNOWN),
        ),
        (
            "Lexical(Obj) on a vector field",
            SearchRequestBuilder::new()
                .lexical_query(obj(TermQuery::new("vec", "hello")))
                .build(),
            Expect::Rejected(NOT_LEXICAL),
        ),
        (
            "filter_query on a vector field",
            SearchRequestBuilder::new()
                .query_dsl("title:hello")
                .filter_query(Box::new(TermQuery::new("vec", "x")))
                .build(),
            Expect::Rejected(NOT_LEXICAL),
        ),
        (
            "a MultiFieldQuery naming an unknown field",
            SearchRequestBuilder::new()
                .lexical_query(obj(MultiFieldQuery::new("hello".to_string())
                    .add_field("title".to_string(), 1.0)
                    .add_field("nosuch".to_string(), 1.0)))
                .build(),
            Expect::Rejected(UNKNOWN),
        ),
        (
            "a SpanNearQuery whose child reads an unknown field",
            SearchRequestBuilder::new()
                .lexical_query(obj(SpanQueryWrapper::new(Box::new(SpanNearQuery::new(
                    "title",
                    child_on_unknown,
                    1,
                    true,
                )))))
                .build(),
            Expect::Rejected(UNKNOWN),
        ),
        (
            "Lexical(Obj) on _id",
            SearchRequestBuilder::new()
                .lexical_query(obj(TermQuery::new("_id", "a")))
                .build(),
            Expect::Hits(1),
        ),
        (
            "Lexical(Obj) on a lexical field",
            SearchRequestBuilder::new()
                .lexical_query(obj(TermQuery::new("title", "hello")))
                .build(),
            Expect::Hits(1),
        ),
        (
            "Lexical(Dsl) on a lexical field",
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Dsl("title:hello".into()))
                .build(),
            Expect::Hits(1),
        ),
        (
            "filter_query on a lexical field",
            SearchRequestBuilder::new()
                .query_dsl("title:hello")
                .filter_query(Box::new(TermQuery::new("body", "world")))
                .build(),
            Expect::Hits(1),
        ),
    ]
}

#[tokio::test(flavor = "multi_thread")]
async fn lexical_queries_and_filters_are_checked_against_the_lexical_fields() {
    let engine = engine().await;
    let mut wrong = Vec::new();
    for (case, request, expect) in requests() {
        let result = engine.search(request).await.map(|hits| hits.len());
        let ok = match (&expect, &result) {
            (Expect::Rejected(text), Err(LaurusError::InvalidArgument(message))) => {
                message.contains(text)
            }
            (Expect::Hits(n), Ok(hits)) => hits == n,
            _ => false,
        };
        if !ok {
            wrong.push(format!("{case}: {result:?}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// After a default field is deleted, a bare term no longer expands to it:
/// the unified DSL used to reject it as an unknown field, and
/// `LexicalSearchQuery::Dsl` searched the deleted field.
#[tokio::test(flavor = "multi_thread")]
async fn a_bare_term_searches_the_remaining_default_fields() {
    let engine = engine().await;
    engine.delete_field("body").await.unwrap();

    let mut wrong = Vec::new();
    for (case, request, hits) in [
        (
            "unified DSL, a term in title",
            SearchRequestBuilder::new().query_dsl("hello").build(),
            1,
        ),
        (
            "unified DSL, a term only the deleted body had",
            SearchRequestBuilder::new().query_dsl("world").build(),
            0,
        ),
        (
            "Lexical(Dsl), a term in title",
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Dsl("hello".into()))
                .build(),
            1,
        ),
        (
            "Lexical(Dsl), a term only the deleted body had",
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Dsl("world".into()))
                .build(),
            0,
        ),
    ] {
        let result = engine.search(request).await.map(|found| found.len());
        if !matches!(result, Ok(n) if n == hits) {
            wrong.push(format!("{case}: {result:?}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
