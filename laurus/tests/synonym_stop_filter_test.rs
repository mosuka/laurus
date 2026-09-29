//! Issue #1259: a `StopFilter` after `SynonymGraphFilter`.
//!
//! Removing a stop word from a synonym graph must leave each member as a
//! path without the stop word. A quoted value then matches the documents
//! holding any member, and never one holding only part of a path.

use std::sync::Arc;

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::pipeline::PipelineAnalyzer;
use laurus::analysis::synonym::dictionary::SynonymDictionary;
use laurus::analysis::token_filter::lowercase::LowercaseFilter;
use laurus::analysis::token_filter::stop::StopFilter;
use laurus::analysis::token_filter::synonym_graph::SynonymGraphFilter;
use laurus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::{
    Document, Engine, Result, Schema, SearchRequestBuilder, StorageConfig, StorageFactory,
};

const SCHEMA_TOML: &str = r#"
default_fields = ["body"]

[fields.body.Text]
indexed = true
stored = true
analyzer = "syn_stop"
"#;

fn analyzer() -> Arc<dyn Analyzer> {
    let mut dict = SynonymDictionary::new(None).unwrap();
    let groups: [&[&str]; 2] = [
        &["statue of liberty", "lady liberty"],
        &["usa", "united states of america", "united states"],
    ];
    for group in groups {
        dict.add_synonym_group(group.iter().map(|s| s.to_string()).collect());
    }
    Arc::new(
        PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new()))
            .add_filter(Arc::new(LowercaseFilter::new()))
            .add_filter(Arc::new(SynonymGraphFilter::new(dict, true)))
            .add_filter(Arc::new(StopFilter::from_words(vec!["of", "the"]))),
    )
}

const DOCS: [(&str, &str); 7] = [
    ("statue", "the statue of liberty"),
    ("lady", "lady liberty"),
    ("liberty_bell", "the liberty bell"),
    ("lady_bug", "a lady bug"),
    ("usa_rocks", "usa rocks"),
    ("america_rocks", "united states of america rocks"),
    ("usa_is_big", "usa is big"),
];

async fn engine() -> Result<Engine> {
    let schema = Schema::from_toml(SCHEMA_TOML)?;
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let engine = Engine::builder(storage, schema)
        .register_runtime_analyzer("syn_stop", analyzer())
        .build()
        .await?;
    for (id, text) in DOCS {
        engine
            .put_document(id, Document::builder().add_text("body", text).build())
            .await?;
    }
    engine.commit().await?;
    Ok(engine)
}

async fn search(engine: &Engine, dsl: &str) -> Result<Vec<String>> {
    let request = SearchRequestBuilder::new().query_dsl(dsl).limit(10).build();
    let mut ids: Vec<String> = engine
        .search(request)
        .await?
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    ids.sort();
    Ok(ids)
}

/// Each member of {statue of liberty, lady liberty}, quoted, matches the
/// documents holding either member, and not one holding only a word of a
/// member.
#[tokio::test]
async fn a_member_with_a_stop_word_matches_every_member() -> Result<()> {
    let engine = engine().await?;
    for dsl in ["\"statue of liberty\"", "\"lady liberty\""] {
        assert_eq!(search(&engine, dsl).await?, ["lady", "statue"], "{dsl}");
    }
    Ok(())
}

/// "usa rocks" follows each member of the group up to "rocks", so it does
/// not match "usa is big", where no member is followed by "rocks".
#[tokio::test]
async fn a_quoted_value_does_not_match_a_member_alone() -> Result<()> {
    let engine = engine().await?;
    for dsl in ["\"usa rocks\"", "\"united states rocks\""] {
        assert_eq!(
            search(&engine, dsl).await?,
            ["america_rocks", "usa_rocks"],
            "{dsl}"
        );
    }
    Ok(())
}
