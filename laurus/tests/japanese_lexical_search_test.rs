//! End-to-end tests for Japanese lexical search.
//!
//! These pin two behaviors that together make unquoted Japanese queries
//! actually work:
//!
//! 1. `Engine::unified_query_parser()` (not a hand-built
//!    `StandardAnalyzer`-only parser) is used to analyze the query string,
//!    so a field's Lindera analyzer is applied to the query the same way
//!    it was applied at index time.
//! 2. A bare (unquoted) term that the analyzer splits into several
//!    morphemes is OR'd (`BooleanQuery`/`Should`), not turned into a
//!    slop-0 `PhraseQuery` requiring exact adjacency.
//!
//! Uses the `embedded://ipadic` dev-dependency dictionary (see
//! `laurus/Cargo.toml`'s `[dev-dependencies]` `lindera` entry with
//! `embed-ipadic`).

use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{Document, Engine, Result, Schema};

/// Same `AnalyzerSpec` shape the CLI reads from `schema.toml`
/// (`laurus/src/engine/schema/analyzer.rs`), expressed as JSON since this
/// crate does not depend on the `toml` crate directly.
const JAPANESE_SCHEMA_JSON: &str = r#"
{
    "default_fields": ["title", "body"],
    "fields": {
        "title": { "Text": {
            "indexed": true, "stored": true, "term_vectors": true,
            "analyzer": { "language": "japanese", "mode": "normal", "dict": "embedded://ipadic" }
        }},
        "body": { "Text": {
            "indexed": true, "stored": true, "term_vectors": true,
            "analyzer": { "language": "japanese", "mode": "normal", "dict": "embedded://ipadic" }
        }}
    }
}
"#;

async fn japanese_engine() -> Result<Engine> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema: Schema = serde_json::from_str(JAPANESE_SCHEMA_JSON).expect("valid schema JSON");
    Engine::new(storage, schema).await
}

/// The direct regression for "unquoted Japanese queries return almost
/// nothing": a bare query whose morphemes only partially overlap a
/// document must still match it, because the tokens are OR'd rather than
/// required to appear as an exact phrase.
#[tokio::test(flavor = "multi_thread")]
async fn bare_japanese_query_matches_partially_overlapping_documents() -> Result<()> {
    let engine = japanese_engine().await?;

    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "吾輩は猫である")
                .add_field("body", "吾輩は猫である。名前はまだ無い。")
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("title", "猫が吾輩を見た")
                .add_field("body", "猫が吾輩を見た日のことである。")
                .build(),
        )
        .await?;
    engine.commit().await?;

    let parser = engine.unified_query_parser()?;
    let request = parser.parse("吾輩は猫").await?;
    let results = engine.search(request).await?.results;

    let ids: Vec<&str> = results.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        ids.len(),
        2,
        "both docs share morphemes with the query, expected 2 hits, got {ids:?}"
    );

    Ok(())
}

/// A quoted phrase must still require adjacency — the OR change only
/// applies to unquoted terms.
#[tokio::test(flavor = "multi_thread")]
async fn quoted_japanese_phrase_still_requires_adjacency() -> Result<()> {
    let engine = japanese_engine().await?;

    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "吾輩は猫である")
                .add_field("body", "吾輩は猫である。名前はまだ無い。")
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("title", "猫が吾輩を見た")
                .add_field("body", "猫が吾輩を見た日のことである。")
                .build(),
        )
        .await?;
    engine.commit().await?;

    let parser = engine.unified_query_parser()?;
    let request = parser.parse("title:\"吾輩は猫である\"").await?;
    let results = engine.search(request).await?.results;

    let ids: Vec<&str> = results.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["doc1"],
        "the quoted phrase must only match the doc with the exact morpheme sequence"
    );

    Ok(())
}

/// Library-side pin for the CLI bug: analyzing the query with a plain
/// `StandardAnalyzer` (as the CLI used to do) would return zero hits for
/// this query, because it can't segment "形態素解析" the way Lindera does
/// at index time. Going through `unified_query_parser()` uses the same
/// per-field analyzer for both indexing and querying.
#[tokio::test(flavor = "multi_thread")]
async fn dsl_query_uses_the_per_field_analyzer_not_the_standard_analyzer() -> Result<()> {
    let engine = japanese_engine().await?;

    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "形態素解析の話")
                .add_field("body", "これは形態素解析についての文章です。")
                .build(),
        )
        .await?;
    engine.commit().await?;

    let parser = engine.unified_query_parser()?;
    let request = parser.parse("形態素解析").await?;
    let results = engine.search(request).await?.results;

    assert!(
        !results.is_empty(),
        "expected at least one hit for a per-field-analyzed Japanese query"
    );

    Ok(())
}
