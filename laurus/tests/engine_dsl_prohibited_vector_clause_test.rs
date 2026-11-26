//! Issue #1751: the unified DSL rejects a vector clause that a `-` or `NOT`
//! applies to, before it or before a group around it, with
//! `InvalidArgument` that names the clause, before anything is embedded. A
//! vector search ranks the documents nearest to the query and has nothing
//! to exclude. The prohibition used to move to the next lexical clause, or
//! to be lost. And a vector-like text inside a quoted lexical phrase is the
//! phrase's text, not a vector clause.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use laurus::storage::memory::MemoryStorage;
use laurus::vector::{FlatOption, Vector};
use laurus::{DataValue, Document, EmbedInput, EmbedInputType, Embedder, Engine, FieldOption};
use laurus::{LaurusError, Result, Schema, SearchRequestBuilder, TextOption};

/// Embeds any text as `[1, 0]`, and counts its calls.
#[derive(Debug)]
struct CountingEmbedder {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Embedder for CountingEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match input {
            EmbedInput::Text(_) => Ok(Vector::new(vec![1.0, 0.0])),
            _ => Err(LaurusError::invalid_argument(
                "CountingEmbedder only embeds text",
            )),
        }
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "counting"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The documents: id, `title`, and `vec`. `cherry` has the vector of every
/// query text; the others are far from it.
const DOCS: [(&str, &str, [f32; 2]); 4] = [
    ("apple", "apple", [0.0, 1.0]),
    ("banana", "banana", [0.0, 1.0]),
    ("cherry", "cherry", [1.0, 0.0]),
    ("phrase", "apple vec cats", [0.0, 1.0]),
];

/// An engine over a text field `title` and a 2-dimensional Flat field
/// `vec`, and the count of its embedder's calls.
async fn engine() -> (Engine, Arc<AtomicUsize>) {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field("vec", FieldOption::Flat(FlatOption::default().dimension(2)))
        .build();
    let calls = Arc::new(AtomicUsize::new(0));
    let embedder = CountingEmbedder {
        calls: calls.clone(),
    };
    let engine = Engine::builder(Arc::new(MemoryStorage::new(Default::default())), schema)
        .embedder(Arc::new(embedder))
        .build()
        .await
        .unwrap();
    for (id, title, vector) in DOCS {
        engine
            .put_document(
                id,
                Document::builder()
                    .add_text("title", title)
                    .add_field("vec", DataValue::Vector(vector.to_vec()))
                    .build(),
            )
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    (engine, calls)
}

/// The sorted ids that `dsl` finds through `query_dsl`, or the error.
async fn found(engine: &Engine, dsl: &str) -> Result<Vec<String>> {
    let hits = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl(dsl)
                .limit(100)
                .build(),
        )
        .await?;
    let mut ids: Vec<String> = hits.into_iter().map(|hit| hit.id).collect();
    ids.sort();
    Ok(ids)
}

/// Issue #1751: a vector clause with a `-` or `NOT` before it, or inside a
/// group with one before it, is rejected with `InvalidArgument` naming the
/// clause, and nothing is embedded.
#[tokio::test(flavor = "multi_thread")]
async fn a_prohibited_vector_clause_is_rejected_before_it_is_embedded() {
    let (engine, calls) = engine().await;
    let mut wrong = Vec::new();
    for (dsl, clause) in [
        ("-vec:cats", "vec:cats"),
        ("- vec:cats", "vec:cats"),
        ("NOT vec:cats", "vec:cats"),
        ("-vec:cats title:apple", "vec:cats"),
        ("title:apple -vec:cats", "vec:cats"),
        ("title:apple NOT vec:cats", "vec:cats"),
        ("title:apple AND NOT vec:cats", "vec:cats"),
        ("-(vec:cats)", "vec:cats"),
        ("NOT (vec:cats)", "vec:cats"),
        ("NOT(vec:cats) title:apple", "vec:cats"),
        ("NOT (title:banana vec:cats)", "vec:cats"),
        ("-(title:banana vec:cats)", "vec:cats"),
        ("(title:banana -vec:cats)", "vec:cats"),
        ("-(-vec:cats)", "vec:cats"),
        ("-vec:\"cute cats\"", "vec:\"cute cats\""),
        ("-vec:cats^0.5", "vec:cats^0.5"),
        ("-(+vec:cats)", "vec:cats"),
    ] {
        let before = calls.load(Ordering::SeqCst);
        let result = found(&engine, dsl).await;
        let embedded = calls.load(Ordering::SeqCst) - before;
        let rejected = matches!(
            &result,
            Err(LaurusError::InvalidArgument(m))
                if m.contains(&format!("vector clause '{clause}' cannot be excluded"))
        );
        if !rejected || embedded != 0 {
            wrong.push(format!("'{dsl}': {result:?}, {embedded} embedded"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1751: the vector clauses that no `-` or `NOT` applies to search as
/// before: alone, required (`+`, intersection), with `NOT` or `-` inside
/// its quoted text, before a prohibited lexical clause, and in a required
/// group.
#[tokio::test(flavor = "multi_thread")]
async fn a_vector_clause_without_a_prohibition_searches_as_before() {
    let (engine, _) = engine().await;
    let every = vec!["apple", "banana", "cherry", "phrase"];
    let mut wrong = Vec::new();
    for (dsl, expected) in [
        ("title:apple vec:cats", every.clone()),
        ("title:apple +vec:cats", vec!["apple", "phrase"]),
        ("vec:\"not cats\" title:apple", every.clone()),
        ("vec:\"-cats\" title:apple", every.clone()),
        ("vec:\"NOT cats\" title:apple", every.clone()),
        ("vec:cats NOT title:apple", every.clone()),
        ("vec:cats -title:apple", every.clone()),
        ("+(vec:cats) title:apple", every.clone()),
    ] {
        match found(&engine, dsl).await {
            Ok(ids) if ids == expected => {}
            other => wrong.push(format!("'{dsl}': {other:?}, expected {expected:?}")),
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1751: a vector-like text inside a quoted lexical phrase is the
/// phrase's text. `title:"apple vec:cats"` used to be a hybrid search of the
/// phrase `apple` and the vector clause `vec:cats`.
#[tokio::test(flavor = "multi_thread")]
async fn a_vector_clause_inside_a_quoted_phrase_is_text() {
    let (engine, calls) = engine().await;
    let before = calls.load(Ordering::SeqCst);
    let result = found(&engine, "title:\"apple vec:cats\"").await;
    let embedded = calls.load(Ordering::SeqCst) - before;
    assert!(
        matches!(&result, Ok(ids) if ids == &["phrase"]) && embedded == 0,
        "{result:?}, {embedded} embedded"
    );
}
