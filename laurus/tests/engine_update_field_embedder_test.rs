//! Issue #1354: `Engine::update_field` must keep a vector field's registered
//! embedder in line with its new option.
//!
//! Removing the `embedder` name from a field used to leave the old embedder
//! registered in the `PerFieldEmbedder`, so text sent to the field was still
//! embedded by the removed model. A field that never named an embedder must
//! keep an embedder the user registered directly through
//! `EngineBuilder::embedder`.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;

use laurus::storage::memory::MemoryStorageConfig;
use laurus::vector::{HnswOption, Vector};
use laurus::{Document, EmbedInput, EmbedInputType, Embedder, EmbedderDefinition, Engine};
use laurus::{FieldOption, LaurusError, PerFieldEmbedder, PrecomputedEmbedder, Result, Schema};
use laurus::{Storage, StorageConfig, StorageFactory, UpdateFieldOptions};

/// A 4-dimensional HNSW field option, naming `embedder` when given.
fn hnsw(embedder: Option<&str>) -> FieldOption {
    let mut option = HnswOption::default().dimension(4).m(16);
    option.embedder = embedder.map(str::to_string);
    FieldOption::Hnsw(option)
}

/// An embedder that turns text into a 4-dimensional vector.
#[derive(Debug)]
struct TextOnlyEmbedder;

#[async_trait]
impl Embedder for TextOnlyEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(t) => Ok(Vector::new(vec![t.len() as f32, 0.0, 0.0, 0.0])),
            _ => Err(LaurusError::invalid_argument(
                "TextOnlyEmbedder does not support this input",
            )),
        }
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "text_only"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn memory_storage() -> Arc<dyn Storage> {
    StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default())).unwrap()
}

/// A `PerFieldEmbedder` falling back to `PrecomputedEmbedder`, with
/// `TextOnlyEmbedder` registered directly for `"vec"`.
fn per_field_with_text_vec() -> Arc<PerFieldEmbedder> {
    let per_field = PerFieldEmbedder::new(Arc::new(PrecomputedEmbedder::new()));
    per_field.add_embedder("vec", Arc::new(TextOnlyEmbedder));
    Arc::new(per_field)
}

/// Fields with an embedder registered in the engine's `PerFieldEmbedder`.
fn configured_fields(engine: &Engine) -> Vec<String> {
    let embedder = engine.embedder();
    embedder
        .as_any()
        .downcast_ref::<PerFieldEmbedder>()
        .expect("the engine's embedder must be a PerFieldEmbedder")
        .configured_fields()
}

async fn put_text(engine: &Engine, id: &str) -> Result<()> {
    engine
        .put_document(id, Document::builder().add_text("vec", "hello").build())
        .await
}

fn reindex() -> UpdateFieldOptions {
    UpdateFieldOptions {
        reindex: true,
        ..Default::default()
    }
}

/// Acceptance criterion 1, with the embedder built from the schema: removing
/// the field's embedder name unregisters it.
#[tokio::test(flavor = "multi_thread")]
async fn update_field_unregisters_a_removed_schema_embedder() {
    let schema = Schema::builder()
        .add_embedder("pre", EmbedderDefinition::Precomputed)
        .add_field("vec", hnsw(Some("pre")))
        .build();
    let engine = Engine::new(memory_storage(), schema).await.unwrap();
    assert!(
        configured_fields(&engine).contains(&"vec".to_string()),
        "the schema embedder must be registered for vec"
    );

    engine
        .update_field("vec", hnsw(None), reindex())
        .await
        .unwrap();

    assert!(
        !configured_fields(&engine).contains(&"vec".to_string()),
        "removing the embedder name must unregister it: {:?}",
        configured_fields(&engine)
    );
}

/// Acceptance criterion 1, observed through ingestion: once the name is
/// removed, text sent to the field is rejected by the default
/// `PrecomputedEmbedder` instead of being embedded by the removed one.
///
/// The field also had an embedder registered directly. It is unregistered
/// too: a field that names a schema embedder has its registration managed by
/// the schema, just as an update that keeps a name overwrites a direct
/// registration with the schema-built embedder.
#[tokio::test(flavor = "multi_thread")]
async fn update_field_removing_the_embedder_name_rejects_text() {
    let schema = Schema::builder()
        .add_embedder("pre", EmbedderDefinition::Precomputed)
        .add_field("vec", hnsw(Some("pre")))
        .build();
    let engine = Engine::builder(memory_storage(), schema)
        .embedder(per_field_with_text_vec())
        .build()
        .await
        .unwrap();
    put_text(&engine, "d1")
        .await
        .expect("text must be accepted before the update");

    engine
        .update_field("vec", hnsw(None), reindex())
        .await
        .unwrap();

    let err = put_text(&engine, "d2")
        .await
        .expect_err("text must be rejected once the embedder name is removed");
    assert!(
        matches!(err, LaurusError::InvalidArgument(_)),
        "unexpected error: {err:?}"
    );
    assert!(!configured_fields(&engine).contains(&"vec".to_string()));
}

/// Acceptance criterion 2: a change to a field that names no embedder,
/// before or after, keeps an embedder the user registered directly.
#[tokio::test(flavor = "multi_thread")]
async fn update_field_keeps_a_directly_registered_embedder() {
    let schema = Schema::builder().add_field("vec", hnsw(None)).build();
    let engine = Engine::builder(memory_storage(), schema)
        .embedder(per_field_with_text_vec())
        .build()
        .await
        .unwrap();

    let FieldOption::Hnsw(option) = hnsw(None) else {
        unreachable!()
    };
    engine
        .update_field("vec", FieldOption::Hnsw(option.m(32)), reindex())
        .await
        .unwrap();

    assert!(configured_fields(&engine).contains(&"vec".to_string()));
    put_text(&engine, "d1")
        .await
        .expect("text must still be embedded by the directly registered embedder");
}
