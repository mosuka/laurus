//! Issue #1326: a document rejected during ingestion (an unsupported input
//! type for the field's embedder, a vector of the wrong dimension, image
//! bytes sent to a text-only embedder) used to leave a WAL record that
//! survived the rejection. Every later open replayed it, hit the same
//! error, and `Engine::builder(...).build()` failed — the index could never
//! be opened again. A rejected `put` also deleted the previous version of
//! the document before the rejection was discovered.
//!
//! Each test below reproduces one rejection, then reopens the engine on the
//! same storage to confirm it still opens and every earlier document
//! survived. The rejected operation is always the last mutation, with no
//! intervening `commit()`, so a poisoned record (if the fix regressed)
//! would still be in the WAL when the engine is reopened.

use std::any::Any;

use async_trait::async_trait;

use laurus::storage::memory::MemoryStorageConfig;
use laurus::vector::{FlatOption, HnswOption, Vector};
use laurus::{DataValue, Document, EmbedInput, EmbedInputType, Embedder};
use laurus::{EmbedderDefinition, Engine, FieldOption, LaurusError, Result, Schema};
use laurus::{Storage, StorageConfig, StorageFactory};
use std::sync::Arc;

/// An HNSW vector field naming `embedder`, matching the house style used by
/// `laurus/src/engine.rs`'s own `#[cfg(test)]` helper of the same name.
fn hnsw_naming(embedder: &str, dimension: usize) -> FieldOption {
    let mut option = HnswOption::default().dimension(dimension);
    option.embedder = Some(embedder.to_string());
    FieldOption::Hnsw(option)
}

/// An embedder that accepts only text, never images — the mirror image of
/// `PrecomputedEmbedder` (which accepts neither).
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

async fn memory_engine(schema: Schema) -> (Engine, Arc<dyn Storage>) {
    let storage: Arc<dyn Storage> =
        StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default())).unwrap();
    let engine = Engine::new(storage.clone(), schema).await.unwrap();
    (engine, storage)
}

/// Reopen `storage` with `schema` and assert it succeeds, returning the new
/// engine. This is the exact failure mode of #1326: before the fix, a
/// rejection left a WAL record that made this call fail forever.
async fn reopen(storage: Arc<dyn Storage>, schema: Schema) -> Engine {
    Engine::new(storage, schema)
        .await
        .expect("index must still be openable after a rejected ingestion call")
}

/// The issue's own reproduction: text sent to a field whose embedder is
/// `PrecomputedEmbedder`, which supports neither text nor images.
#[tokio::test(flavor = "multi_thread")]
async fn reopens_after_text_rejected_by_precomputed_embedder() {
    let schema = Schema::builder()
        .add_embedder("pre", EmbedderDefinition::Precomputed)
        .add_field("vec", hnsw_naming("pre", 4))
        .build();
    let (engine, storage) = memory_engine(schema.clone()).await;

    let err = engine
        .put_document(
            "d1",
            Document::builder().add_text("vec", "hello world").build(),
        )
        .await
        .expect_err("PrecomputedEmbedder must reject text input");
    assert!(
        matches!(&err, LaurusError::InvalidArgument(m) if m.contains("does not support text input")),
        "unexpected error: {err:?}"
    );
    drop(engine);

    let reopened = reopen(storage, schema).await;
    assert_eq!(
        reopened.stats().unwrap().document_count,
        0,
        "the rejected document must not exist"
    );
}

/// A pre-computed vector whose dimension differs from the field's.
#[tokio::test(flavor = "multi_thread")]
async fn reopens_after_dimension_mismatch() {
    let schema = Schema::builder()
        .add_field("vec", FieldOption::Flat(FlatOption::default().dimension(3)))
        .build();
    let (engine, storage) = memory_engine(schema.clone()).await;

    let err = engine
        .put_document(
            "d1",
            Document::builder()
                .add_vector("vec", vec![1.0, 0.0])
                .build(),
        )
        .await
        .expect_err("a 2-dimensional vector must be rejected by a 3-dimensional field");
    assert!(
        matches!(&err, LaurusError::InvalidArgument(m) if m.contains("dimension")),
        "unexpected error: {err:?}"
    );
    drop(engine);

    let reopened = reopen(storage, schema).await;
    assert_eq!(reopened.stats().unwrap().document_count, 0);
}

/// Image bytes sent to a field whose embedder supports only text.
#[tokio::test(flavor = "multi_thread")]
async fn reopens_after_image_rejected_by_text_only_embedder() {
    let schema = Schema::builder()
        // The name just needs to be declared (Issue #1309); the explicit
        // embedder below is what actually resolves it for every field.
        .add_embedder("text_only", EmbedderDefinition::Precomputed)
        .add_field("vec", hnsw_naming("text_only", 4))
        .build();

    let storage: Arc<dyn Storage> =
        StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default())).unwrap();
    let engine = Engine::builder(storage.clone(), schema.clone())
        .embedder(Arc::new(TextOnlyEmbedder))
        .build()
        .await
        .unwrap();

    let err = engine
        .put_document(
            "d1",
            Document::builder()
                .add_field(
                    "vec",
                    DataValue::Bytes(vec![0xFF, 0xD8, 0xFF], Some("image/jpeg".to_string())),
                )
                .build(),
        )
        .await
        .expect_err("a text-only embedder must reject image bytes");
    assert!(
        matches!(&err, LaurusError::InvalidArgument(m) if m.contains("does not support image input")),
        "unexpected error: {err:?}"
    );
    drop(engine);

    let reopened = Engine::builder(storage, schema)
        .embedder(Arc::new(TextOnlyEmbedder))
        .build()
        .await
        .expect("index must still be openable after a rejected ingestion call");
    assert_eq!(reopened.stats().unwrap().document_count, 0);
}

/// A rejected `put` must leave the previous version of the document in
/// place (Issue #1326's second acceptance criterion) — both immediately and
/// after a reopen.
#[tokio::test(flavor = "multi_thread")]
async fn rejected_put_keeps_the_previous_version() {
    let schema = Schema::builder()
        .add_field(
            "label",
            FieldOption::Text(laurus::lexical::TextOption::default()),
        )
        .add_field("vec", FieldOption::Flat(FlatOption::default().dimension(3)))
        .build();
    let (engine, storage) = memory_engine(schema.clone()).await;

    engine
        .put_document(
            "d1",
            Document::builder()
                .add_text("label", "v1")
                .add_vector("vec", vec![1.0, 0.0, 0.0])
                .build(),
        )
        .await
        .unwrap();
    engine.commit().await.unwrap();

    let err = engine
        .put_document(
            "d1",
            Document::builder()
                .add_text("label", "v2")
                .add_vector("vec", vec![1.0, 0.0])
                .build(),
        )
        .await
        .expect_err("the wrong-dimension v2 must be rejected");
    assert!(matches!(&err, LaurusError::InvalidArgument(_)));

    let docs = engine.get_documents("d1").await.unwrap();
    assert_eq!(docs.len(), 1, "v1 must still be the live document");
    assert_eq!(
        docs[0].fields.get("label"),
        Some(&DataValue::Text("v1".to_string()))
    );
    drop(engine);

    let reopened = reopen(storage, schema).await;
    let docs = reopened.get_documents("d1").await.unwrap();
    assert_eq!(docs.len(), 1, "v1 must survive the reopen");
    assert_eq!(
        docs[0].fields.get("label"),
        Some(&DataValue::Text("v1".to_string()))
    );
}
