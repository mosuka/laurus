//! Issue #1310: an `[analyzers.*]` entry named after a built-in analyzer.
//!
//! A new schema rejects such an entry (`Schema::validate_for_create`), but a
//! schema persisted before that check may still hold one. Its index must keep
//! opening, and a field naming the entry keeps getting the built-in.

use laurus::storage::memory::MemoryStorageConfig;
use laurus::{
    Document, Engine, Result, Schema, SearchRequestBuilder, StorageConfig, StorageFactory,
};

// The entry tokenizes on whitespace without lowercasing, so a lowercase query
// matches "Hello" only when the built-in `standard` analyzer is used.
const PERSISTED_SCHEMA_TOML: &str = r#"
default_fields = ["body"]

[analyzers.standard]
tokenizer = { type = "whitespace" }

[fields.body.Text]
indexed = true
stored = true
analyzer = "standard"
"#;

async fn engine_with_hello_world(schema: Schema) -> Result<Engine> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let engine = Engine::builder(storage, schema).build().await?;
    engine
        .put_document(
            "doc1",
            Document::builder().add_text("body", "Hello World").build(),
        )
        .await?;
    engine.commit().await?;
    Ok(engine)
}

async fn hits(engine: &Engine, dsl: &str) -> Result<usize> {
    let request = SearchRequestBuilder::new().query_dsl(dsl).limit(10).build();
    Ok(engine.search(request).await?.len())
}

#[tokio::test]
async fn engine_builds_over_persisted_reserved_analyzer_name_and_uses_builtin() -> Result<()> {
    let schema = Schema::from_toml(PERSISTED_SCHEMA_TOML)?;
    assert!(schema.validate_for_create().is_err());

    let engine = engine_with_hello_world(schema).await?;
    assert_eq!(hits(&engine, "body:hello").await?, 1);
    Ok(())
}

// Guards the premise of the test above: the same definition under a
// non-reserved name is used, and does not match a lowercase query.
#[tokio::test]
async fn same_definition_under_another_name_is_used() -> Result<()> {
    let schema = Schema::from_toml(&PERSISTED_SCHEMA_TOML.replace("standard", "ws_only"))?;
    schema.validate_for_create()?;

    let engine = engine_with_hello_world(schema).await?;
    assert_eq!(hits(&engine, "body:hello").await?, 0);
    assert_eq!(hits(&engine, "body:Hello").await?, 1);
    Ok(())
}
