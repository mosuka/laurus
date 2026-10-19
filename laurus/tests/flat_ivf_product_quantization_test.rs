//! Issue #1457: Flat and IVF fields reject a product quantizer.
//!
//! Their writers used to take a PQ quantizer down the Scalar8Bit path
//! without an error. Defining such a field now fails, and an index whose
//! persisted schema already holds one fails to open until the field's
//! quantizer is set to `Scalar8Bit`, which is what its segments hold.

use std::sync::Arc;

use tempfile::TempDir;

use laurus::storage::Storage;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::Vector;
use laurus::{
    Document, Engine, FieldOption, FlatOption, IvfOption, LaurusError, QuantizationMethod,
    QueryVector, Schema, SearchRequestBuilder, UpdateFieldOptions, VectorSearchQuery,
};

const DIM: usize = 8;

fn pq() -> QuantizationMethod {
    QuantizationMethod::ProductQuantization { subvector_count: 4 }
}

fn flat(quantizer: QuantizationMethod) -> FieldOption {
    FieldOption::Flat(FlatOption::new(DIM).quantizer(quantizer))
}

fn ivf(quantizer: QuantizationMethod) -> FieldOption {
    FieldOption::Ivf(
        IvfOption::new(DIM)
            .n_clusters(2)
            .n_probe(2)
            .quantizer(quantizer),
    )
}

/// A schema with `option` as field `vec`, built without
/// `Schema::validate_for_create` (which `SchemaBuilder::build` runs).
fn schema_with(option: FieldOption) -> Schema {
    let mut schema = Schema::new();
    schema.fields.insert("vec".to_string(), option);
    schema
}

fn open_storage(dir: &TempDir) -> Arc<dyn Storage> {
    let config = StorageConfig::File(FileStorageConfig::new(dir.path()));
    StorageFactory::open(config).unwrap()
}

fn vector(seed: usize) -> Vec<f32> {
    (0..DIM)
        .map(|i| ((seed * 7 + i) % 11) as f32 + 1.0)
        .collect()
}

async fn search_ids(engine: &Engine) -> Vec<String> {
    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vector(1)),
            weight: 1.0,
            fields: Some(vec!["vec".into()]),
        }]))
        .limit(10)
        .build();
    engine
        .search(request)
        .await
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

fn assert_rejects_field(err: &LaurusError, kind: &str) {
    let expected = format!("field 'vec': {kind} does not support quantizer ProductQuantization");
    assert!(
        matches!(err, LaurusError::InvalidArgument(m) if m.contains(&expected)),
        "unexpected error: {err:?}"
    );
}

/// Index creation, `add_field` and `update_field` all reject the field
/// before the schema or the data change.
#[tokio::test(flavor = "multi_thread")]
async fn defining_a_flat_or_ivf_field_with_pq_fails() {
    for (kind, make) in [
        ("Flat", flat as fn(QuantizationMethod) -> FieldOption),
        ("IVF", ivf),
    ] {
        assert_rejects_field(
            &schema_with(make(pq())).validate_for_create().unwrap_err(),
            kind,
        );

        let dir = TempDir::new().unwrap();
        let config = StorageConfig::File(FileStorageConfig::new(dir.path()));
        let storage = StorageFactory::create(config).unwrap();
        let engine = Engine::new(storage, schema_with(make(QuantizationMethod::Scalar8Bit)))
            .await
            .unwrap();
        for i in 0..4 {
            let doc = Document::builder().add_field("vec", vector(i)).build();
            engine.put_document(&format!("d{i}"), doc).await.unwrap();
        }
        engine.commit().await.unwrap();
        let before = search_ids(&engine).await;
        assert_eq!(before.len(), 4, "{kind}");

        let err = engine.add_field("vec2", make(pq())).await.unwrap_err();
        let expected = format!("field 'vec2': {kind} does not support quantizer");
        assert!(
            matches!(&err, LaurusError::InvalidArgument(m) if m.contains(&expected)),
            "{kind}: unexpected error: {err:?}"
        );
        assert!(!engine.schema().fields.contains_key("vec2"), "{kind}");

        let opts = UpdateFieldOptions {
            reindex: true,
            dry_run: false,
        };
        let err = engine
            .update_field("vec", make(pq()), opts)
            .await
            .unwrap_err();
        assert_rejects_field(&err, kind);
        let quantizer = match &engine.schema().fields["vec"] {
            FieldOption::Flat(opt) => opt.quantizer,
            FieldOption::Ivf(opt) => opt.quantizer,
            other => panic!("unexpected option: {other:?}"),
        };
        assert_eq!(quantizer, QuantizationMethod::Scalar8Bit, "{kind}");
        assert_eq!(search_ids(&engine).await, before, "{kind}");
    }
}

/// An index written before the check holds the field exactly as
/// `Scalar8Bit`. Committing with `Scalar8Bit` reproduces it; opening it
/// with PQ in the schema fails, and setting the quantizer back to
/// `Scalar8Bit` opens it with the same hits.
#[tokio::test(flavor = "multi_thread")]
async fn existing_index_with_pq_fails_to_open_and_reopens_as_scalar8bit() {
    for (kind, make) in [
        ("Flat", flat as fn(QuantizationMethod) -> FieldOption),
        ("IVF", ivf),
    ] {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::File(FileStorageConfig::new(dir.path()));
        let storage = StorageFactory::create(config).unwrap();
        let engine = Engine::new(storage, schema_with(make(QuantizationMethod::Scalar8Bit)))
            .await
            .unwrap();
        for i in 0..4 {
            let doc = Document::builder().add_field("vec", vector(i)).build();
            engine.put_document(&format!("d{i}"), doc).await.unwrap();
        }
        engine.commit().await.unwrap();
        let before = search_ids(&engine).await;
        assert_eq!(before.len(), 4, "{kind}");
        drop(engine);

        // The schema.toml such an index holds still parses.
        let persisted = Schema::from_toml(&schema_with(make(pq())).to_toml().unwrap()).unwrap();
        let err = Engine::new(open_storage(&dir), persisted)
            .await
            .err()
            .unwrap_or_else(|| panic!("{kind}: opening with PQ must fail"));
        assert_rejects_field(&err, kind);
        assert!(
            err.to_string()
                .contains("set its quantizer to Scalar8Bit in schema.toml"),
            "{kind}: {err}"
        );

        let reopened = Engine::new(
            open_storage(&dir),
            schema_with(make(QuantizationMethod::Scalar8Bit)),
        )
        .await
        .unwrap();
        assert_eq!(search_ids(&reopened).await, before, "{kind}");
    }
}
