//! Issue #1247: quoted queries on fields indexed with `term_vectors: false`.
//!
//! Such a field stores no term positions. A quoted value that analyzes to
//! one token is a term match and must still find its documents; a phrase
//! of two or more terms, or a span query, needs positions and must be
//! rejected with an error instead of silently matching nothing.

use laurus::lexical::span::{SpanQueryWrapper, SpanTermQuery};
use laurus::lexical::{BooleanQuery, PhraseQuery, Query};
use laurus::storage::memory::MemoryStorageConfig;
use laurus::{
    Document, Engine, LaurusError, LexicalSearchQuery, Result, Schema, SearchRequestBuilder,
    StorageConfig, StorageFactory,
};

/// `path` and `tags` are exact-match fields, `links` a whitespace-split URL
/// list, `body` and `title` full text. Only `title` stores positions.
const SCHEMA_TOML: &str = r#"
default_fields = ["title", "body"]

[analyzers.url_list]
tokenizer = { type = "whitespace" }

[fields.path.Text]
indexed = true
stored = true
term_vectors = false
analyzer = "keyword"

[fields.tags.Text]
indexed = true
stored = true
term_vectors = false
multi_valued = true
analyzer = "keyword"

[fields.links.Text]
indexed = true
stored = true
term_vectors = false
analyzer = "url_list"

[fields.body.Text]
indexed = true
stored = true
term_vectors = false

[fields.title.Text]
indexed = true
stored = true
term_vectors = true
"#;

async fn engine() -> Result<Engine> {
    let schema = Schema::from_toml(SCHEMA_TOML)?;
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let engine = Engine::new(storage, schema).await?;

    engine
        .put_document(
            "a",
            Document::builder()
                .add_text("path", "file:///m/a.md")
                .add_text_array(
                    "tags",
                    vec!["lang/rust".to_string(), "topic/search".to_string()],
                )
                .add_text("links", "https://example.com/a https://example.com/b?x=1")
                .add_text("body", "rust search engine")
                .add_text("title", "rust search engine")
                .build(),
        )
        .await?;
    engine
        .put_document(
            "b",
            Document::builder()
                .add_text("path", "file:///m/b.md")
                .add_text_array("tags", vec!["lang/go".to_string()])
                .add_text("links", "https://other.example/z")
                .add_text("body", "search for rust")
                .add_text("title", "search for rust")
                .build(),
        )
        .await?;
    engine.commit().await?;
    Ok(engine)
}

async fn search_dsl(engine: &Engine, dsl: &str) -> Result<Vec<String>> {
    let request = SearchRequestBuilder::new().query_dsl(dsl).limit(10).build();
    ids(engine, request).await
}

async fn search_obj(engine: &Engine, query: Box<dyn Query>) -> Result<Vec<String>> {
    let request = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(query))
        .limit(10)
        .build();
    ids(engine, request).await
}

async fn ids(engine: &Engine, request: laurus::SearchRequest) -> Result<Vec<String>> {
    let mut ids: Vec<String> = engine
        .search(request)
        .await?
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    ids.sort();
    Ok(ids)
}

fn phrase(field: &str, terms: &[&str]) -> Box<dyn Query> {
    let terms = terms.iter().map(|t| t.to_string()).collect();
    Box::new(PhraseQuery::new(field, terms))
}

/// The error must be a query error (not an internal one) that names the
/// offending field, so a caller can tell what to fix.
fn assert_rejected(result: Result<Vec<String>>, field: &str) {
    match result {
        Err(LaurusError::Query(msg)) => {
            assert!(msg.contains(field), "message must name {field:?}: {msg}");
            assert!(
                msg.contains("term_vectors"),
                "message must point at term_vectors: {msg}"
            );
        }
        other => panic!("expected a query error naming {field:?}, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn quoted_single_token_matches_without_positions() -> Result<()> {
    let engine = engine().await?;

    assert_eq!(
        search_dsl(&engine, r#"path:"file:///m/a.md""#).await?,
        ["a"]
    );
    assert_eq!(
        search_dsl(&engine, r#"links:"https://example.com/a""#).await?,
        ["a"]
    );
    assert_eq!(search_dsl(&engine, r#"tags:"lang/rust""#).await?, ["a"]);
    assert_eq!(search_dsl(&engine, r#"body:"engine""#).await?, ["a"]);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn one_term_phrase_query_object_matches_without_positions() -> Result<()> {
    let engine = engine().await?;

    assert_eq!(
        search_obj(&engine, phrase("path", &["file:///m/a.md"])).await?,
        ["a"]
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn phrase_on_a_field_with_positions_still_matches() -> Result<()> {
    let engine = engine().await?;

    assert_eq!(search_dsl(&engine, r#"title:"rust search""#).await?, ["a"]);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_term_phrase_without_positions_is_rejected() -> Result<()> {
    let engine = engine().await?;

    assert_rejected(search_dsl(&engine, r#"body:"rust search""#).await, "body");
    assert_rejected(
        search_obj(&engine, phrase("body", &["rust", "search"])).await,
        "body",
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn span_query_without_positions_is_rejected() -> Result<()> {
    let engine = engine().await?;

    let span = SpanQueryWrapper::new(Box::new(SpanTermQuery::new("body", "rust")));
    assert_rejected(search_obj(&engine, Box::new(span)).await, "body");

    Ok(())
}

/// A prohibited phrase would silently exclude nothing.
#[tokio::test(flavor = "multi_thread")]
async fn prohibited_phrase_without_positions_is_rejected() -> Result<()> {
    let engine = engine().await?;

    assert_rejected(
        search_dsl(&engine, r#"title:rust -body:"rust search""#).await,
        "body",
    );

    let mut query = BooleanQuery::new();
    query.add_should(phrase("title", &["rust", "search"]));
    query.add_must_not(phrase("body", &["rust", "search"]));
    assert_rejected(search_obj(&engine, Box::new(query)).await, "body");

    Ok(())
}

/// A filter phrase would silently filter every document out.
#[tokio::test(flavor = "multi_thread")]
async fn filter_phrase_without_positions_is_rejected() -> Result<()> {
    let engine = engine().await?;

    let request = SearchRequestBuilder::new()
        .query_dsl("title:rust")
        .filter_query(phrase("body", &["rust", "search"]))
        .limit(10)
        .build();
    assert_rejected(ids(&engine, request).await, "body");

    Ok(())
}

/// An unfielded phrase fans out over `default_fields`; one of them lacks
/// positions, so the whole query is rejected, as in Lucene.
#[tokio::test(flavor = "multi_thread")]
async fn unfielded_phrase_over_a_default_field_without_positions_is_rejected() -> Result<()> {
    let engine = engine().await?;

    assert_rejected(search_dsl(&engine, r#""rust search""#).await, "body");

    Ok(())
}
