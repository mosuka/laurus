//! Regression tests for #1098: a term removed by its field's analyzer
//! matches nothing in that field instead of aborting the whole query.

use std::sync::Arc;

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::per_field::PerFieldAnalyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::analysis::token::TokenStream;
use laurus::lexical::query::LexicalQueryParser;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{Document, Engine, LaurusError, Result, Schema, SearchRequestBuilder};

async fn test_engine(body_analyzer: &str) -> Result<Engine> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema: Schema = serde_json::from_value(serde_json::json!({
        "default_fields": ["title", "body"],
        "fields": {
            "title": { "Text": { "analyzer": "standard" } },
            "body": { "Text": { "analyzer": body_analyzer } }
        }
    }))
    .unwrap();
    let engine = Engine::new(storage, schema).await?;
    for (id, title, body) in [
        ("title-book", "book", "other"),
        ("body-book", "other", "book"),
        ("body-all", "other", "all"),
    ] {
        engine
            .put_document(
                id,
                Document::builder()
                    .add_text("title", title)
                    .add_text("body", body)
                    .build(),
            )
            .await?;
    }
    engine.commit().await?;
    Ok(engine)
}

async fn search(engine: &Engine, dsl: &str) -> Result<Vec<(String, f32)>> {
    let results = engine
        .search(SearchRequestBuilder::new().query_dsl(dsl).limit(10).build())
        .await?
        .results;
    let mut hits: Vec<_> = results.into_iter().map(|hit| (hit.id, hit.score)).collect();
    hits.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(hits)
}

#[tokio::test]
async fn test_stop_words_match_nothing_without_failing() -> Result<()> {
    let engine = test_engine("standard").await?;
    for dsl in [
        "all",
        "the",
        "a",
        "title:all",
        "body:all",
        "all^4",
        "\"all\"",
    ] {
        assert!(search(&engine, dsl).await?.is_empty(), "{dsl}");
    }
    Ok(())
}

#[tokio::test]
async fn test_stop_words_preserve_boolean_semantics_and_scores() -> Result<()> {
    let engine = test_engine("standard").await?;
    let book = search(&engine, "book").await?;
    let book_ids: Vec<_> = book.iter().map(|hit| hit.0.as_str()).collect();
    assert_eq!(book_ids, vec!["body-book", "title-book"]);
    for dsl in [
        "all book",
        "book all",
        "all OR book",
        "book -all",
        "-all book",
    ] {
        let hits = search(&engine, dsl).await?;
        assert_eq!(
            hits.iter().map(|hit| hit.0.as_str()).collect::<Vec<_>>(),
            book_ids,
            "{dsl}"
        );
        // Compare scores with the existing empty-phrase path: a bare term
        // and a nested BooleanQuery have different field-length handling.
        assert_eq!(
            hits,
            search(&engine, &dsl.replace("all", "\"all\"")).await?,
            "{dsl}"
        );
    }
    for dsl in ["all AND book", "book AND all", "+all book", "all -book"] {
        assert!(search(&engine, dsl).await?.is_empty(), "{dsl}");
    }
    assert_eq!(search(&engine, "-all").await?.len(), 3);
    assert_eq!(
        search(&engine, "book^2 all^4").await?,
        search(&engine, "book^2 \"all\"").await?
    );
    Ok(())
}

#[tokio::test]
async fn test_stop_word_is_empty_only_in_fields_that_remove_it() -> Result<()> {
    let engine = test_engine("keyword").await?;
    let expected = search(&engine, "body:all").await?;
    assert_eq!(expected.len(), 1);
    assert_eq!(expected[0].0, "body-all");
    let hits = search(&engine, "all").await?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, expected[0].0);
    assert!(search(&engine, "title:all").await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_stop_words_keep_unknown_field_validation() -> Result<()> {
    let engine = test_engine("standard").await?;
    for dsl in ["missing:all", "missing:all OR title:book"] {
        let error = search(&engine, dsl).await.unwrap_err().to_string();
        assert!(error.contains("unknown field"), "{error}");
        assert!(error.contains("missing"), "{error}");
    }
    Ok(())
}

#[test]
fn test_stop_word_keeps_field_and_boost() {
    let parser = LexicalQueryParser::with_standard_analyzer()
        .unwrap()
        .with_default_field("body");
    let query = parser.parse("title:all^3").unwrap();
    assert_eq!(query.field(), Some("title"));
    assert_eq!(query.boost(), 3.0);
}

#[test]
fn test_stop_word_does_not_hide_another_fields_analysis_error() {
    #[derive(Debug)]
    struct FailingAnalyzer;

    impl Analyzer for FailingAnalyzer {
        fn analyze(&self, _text: &str) -> Result<TokenStream> {
            Err(LaurusError::parse("test analyzer failure".to_string()))
        }

        fn name(&self) -> &str {
            "failing"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    let analyzer = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
    analyzer.add_analyzer("body", Arc::new(FailingAnalyzer));
    let parser = LexicalQueryParser::new(Arc::new(analyzer))
        .with_default_fields(vec!["title".to_string(), "body".to_string()]);
    let error = parser.parse("all").unwrap_err().to_string();
    assert!(error.contains("test analyzer failure"), "{error}");
}
