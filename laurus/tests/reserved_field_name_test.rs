//! Issue #1329: a `_`-prefixed field other than `_id` in a schema.
//!
//! A new schema rejects such a field (`Schema::validate_for_create`), but a
//! schema persisted before that check may still hold one. Its index must
//! keep opening, and ingesting a document that sets the field must still be
//! rejected, exactly as it is today.

use laurus::storage::memory::MemoryStorageConfig;
use laurus::{Document, Engine, Result, Schema, StorageConfig, StorageFactory};

const PERSISTED_SCHEMA_TOML: &str = r#"
default_fields = ["body"]

[fields.body.Text]
indexed = true
stored = true

[fields._secret.Text]
indexed = true
stored = true
"#;

#[tokio::test]
async fn engine_opens_over_persisted_reserved_field_name_and_still_rejects_ingestion() -> Result<()>
{
    let schema = Schema::from_toml(PERSISTED_SCHEMA_TOML)?;
    assert!(schema.validate_for_create().is_err());

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let engine = Engine::builder(storage, schema).build().await?;

    let err = engine
        .put_document(
            "doc1",
            Document::builder()
                .add_text("body", "hello")
                .add_text("_secret", "leaked")
                .build(),
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("Field name '_secret' is reserved"),
        "unexpected error: {err}"
    );

    // The engine is otherwise unaffected: a document that doesn't touch the
    // reserved field still indexes normally.
    engine
        .put_document(
            "doc2",
            Document::builder().add_text("body", "hello").build(),
        )
        .await?;
    engine.commit().await?;

    Ok(())
}
