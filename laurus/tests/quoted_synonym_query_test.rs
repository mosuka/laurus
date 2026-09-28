//! Issue #1252: quoted queries through an analyzer that stacks synonyms.
//!
//! `SynonymGraphFilter` puts a synonym at the same position as the word it
//! expands (`position_increment = 0`). A quoted value must match any of the
//! stacked alternatives, and a phrase must follow the stacked positions,
//! whether the synonyms run at index time and query time (the `Engine`
//! path, which uses one analyzer for both) or at query time only.

use std::sync::Arc;

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::pipeline::PipelineAnalyzer;
use laurus::analysis::synonym::dictionary::SynonymDictionary;
use laurus::analysis::token_filter::lowercase::LowercaseFilter;
use laurus::analysis::token_filter::synonym_graph::SynonymGraphFilter;
use laurus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use laurus::lexical::query::parser::LexicalQueryParser;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::{
    Document, Engine, LaurusError, LexicalSearchQuery, Result, Schema, SearchRequestBuilder,
    StorageConfig, StorageFactory,
};

/// `syn` stacks synonyms at index time, `plain` does not. `tags` is a
/// multi-valued `syn` field, and `nopos` a `syn` field without positions.
const SCHEMA_TOML: &str = r#"
default_fields = ["syn"]

[fields.syn.Text]
indexed = true
stored = true
analyzer = "syn"

[fields.plain.Text]
indexed = true
stored = true
analyzer = "plain"

[fields.tags.Text]
indexed = true
stored = true
multi_valued = true
analyzer = "syn"

[fields.nopos.Text]
indexed = true
stored = true
term_vectors = false
analyzer = "syn"
"#;

fn plain_analyzer() -> PipelineAnalyzer {
    PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new()))
        .add_filter(Arc::new(LowercaseFilter::new()))
}

fn syn_analyzer() -> Arc<dyn Analyzer> {
    let mut dict = SynonymDictionary::new(None).unwrap();
    dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);
    dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);
    Arc::new(plain_analyzer().add_filter(Arc::new(SynonymGraphFilter::new(dict, true))))
}

const DOCS: [(&str, &str); 4] = [
    ("big_dog", "a big dog barks"),
    ("large_dog", "a large dog barks"),
    ("ml", "ml is fun"),
    ("machine_learning", "machine learning is fun"),
];

async fn engine() -> Result<Engine> {
    let schema = Schema::from_toml(SCHEMA_TOML)?;
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let engine = Engine::builder(storage, schema)
        .register_runtime_analyzer("syn", syn_analyzer())
        .register_runtime_analyzer("plain", Arc::new(plain_analyzer()))
        .build()
        .await?;
    for (id, text) in DOCS {
        engine
            .put_document(
                id,
                Document::builder()
                    .add_text("syn", text)
                    .add_text("plain", text)
                    .add_text("nopos", text)
                    .build(),
            )
            .await?;
    }
    engine
        .put_document(
            "tags",
            Document::builder()
                .add_text_array("tags", vec!["big".to_string(), "dog".to_string()])
                .build(),
        )
        .await?;
    engine.commit().await?;
    Ok(engine)
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

async fn search_dsl(engine: &Engine, dsl: &str) -> Result<Vec<String>> {
    let request = SearchRequestBuilder::new().query_dsl(dsl).limit(10).build();
    ids(engine, request).await
}

/// Parse `dsl` with the synonym analyzer and run it against `engine`, so
/// synonyms apply at query time only when the field is `plain`.
async fn search_query_time(engine: &Engine, dsl: &str) -> Result<Vec<String>> {
    let query = LexicalQueryParser::new(syn_analyzer()).parse(dsl)?;
    let request = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(query))
        .limit(10)
        .build();
    ids(engine, request).await
}

fn sorted(ids: &[&str]) -> Vec<String> {
    let mut ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    ids.sort();
    ids
}

const DOGS: &[&str] = &["big_dog", "large_dog"];
const MLS: &[&str] = &["ml", "machine_learning"];

#[tokio::test(flavor = "multi_thread")]
async fn index_and_query_time_synonyms() -> Result<()> {
    let engine = engine().await?;
    let cases: &[(&str, &[&str])] = &[
        ("syn:big", DOGS),
        // One position with two stacked tokens: either one matches.
        ("syn:\"big\"", DOGS),
        ("syn:\"large\"", DOGS),
        // A phrase through a stacked position.
        ("syn:\"big dog\"", DOGS),
        ("syn:\"a big dog\"", DOGS),
        ("syn:\"a large dog barks\"", DOGS),
        ("syn:\"dog barks\"", DOGS),
        // Multi-word synonyms: one path per alternative.
        ("syn:\"ml\"", MLS),
        ("syn:\"machine learning\"", MLS),
        ("syn:\"ml is\"", MLS),
        ("syn:\"machine learning is fun\"", MLS),
        // With the synonym, "ml is fun" also reads "machine learning is
        // fun", so a phrase inside that span matches it.
        ("syn:\"learning is\"", MLS),
        ("syn:\"big fun\"", &[]),
    ];
    for (dsl, expected) in cases {
        assert_eq!(search_dsl(&engine, dsl).await?, sorted(expected), "{dsl}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_time_only_synonyms() -> Result<()> {
    let engine = engine().await?;
    let cases: &[(&str, &[&str])] = &[
        ("plain:\"big\"", DOGS),
        ("plain:\"big dog\"", DOGS),
        ("plain:\"a large dog\"", DOGS),
        ("plain:\"ml\"", MLS),
        ("plain:\"machine learning\"", MLS),
        ("plain:\"ml is\"", MLS),
        ("plain:\"learning is\"", &["machine_learning"]),
    ];
    for (dsl, expected) in cases {
        assert_eq!(
            search_query_time(&engine, dsl).await?,
            sorted(expected),
            "{dsl}"
        );
    }
    Ok(())
}

/// A stacked token must not push the next element further out: the phrase
/// may cross into the next element only once the slop reaches the gap
/// (100), as for any multi-valued field.
#[tokio::test(flavor = "multi_thread")]
async fn multi_valued_field_keeps_its_gap_after_a_stacked_token() -> Result<()> {
    let engine = engine().await?;
    assert_eq!(search_dsl(&engine, "tags:\"big dog\"").await?, sorted(&[]));
    assert_eq!(
        search_dsl(&engine, "tags:\"big dog\"~99").await?,
        sorted(&[])
    );
    assert_eq!(
        search_dsl(&engine, "tags:\"big dog\"~100").await?,
        sorted(&["tags"])
    );
    Ok(())
}

/// One stacked position is a term match and needs no positions; a phrase
/// still does, even when only one of its paths is a phrase.
#[tokio::test(flavor = "multi_thread")]
async fn positionless_field_accepts_one_position_and_rejects_phrases() -> Result<()> {
    let engine = engine().await?;
    assert_eq!(search_dsl(&engine, "nopos:\"big\"").await?, sorted(DOGS));
    for dsl in ["nopos:\"big dog\"", "nopos:\"ml\""] {
        match search_dsl(&engine, dsl).await {
            Err(LaurusError::Query(message)) => {
                assert!(message.contains("nopos"), "{dsl}: {message}")
            }
            other => panic!("{dsl}: expected a query error, got {other:?}"),
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn phrase_through_a_synonym_is_highlighted() -> Result<()> {
    let engine = engine().await?;
    let request = SearchRequestBuilder::new()
        .query_dsl("syn:\"a big dog\"")
        .highlight(vec!["syn".to_string()])
        .limit(10)
        .build();
    let results = engine.search(request).await?;
    let hit = results
        .iter()
        .find(|hit| hit.id == "large_dog")
        .expect("large_dog must match");
    assert_eq!(
        hit.highlights["syn"],
        vec!["<mark>a large dog</mark> barks"]
    );
    Ok(())
}
