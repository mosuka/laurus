//! Issue #1479: HNSW, IVF and Flat fields reject parameters their index
//! cannot work with.
//!
//! The gRPC server and the HTTP gateway used to persist an omitted HNSW
//! `m` / `ef_construction` or IVF `n_clusters` / `n_probe` as 0. With `m`
//! 0 every document write failed; with `n_probe` 0 every search returned
//! no hits. Defining such a field now fails, while an index whose
//! persisted schema already holds one still opens, so the field can be
//! repaired with `update_field`.

use std::sync::Arc;

use tempfile::TempDir;

use laurus::storage::Storage;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::Vector;
use laurus::{
    Document, Engine, FieldChangeKind, FieldOption, FlatOption, HnswOption, IvfOption, LaurusError,
    QueryVector, Schema, SearchRequestBuilder, UpdateFieldOptions, VectorSearchQuery,
};

const DIM: usize = 8;

const REINDEX: UpdateFieldOptions = UpdateFieldOptions {
    reindex: true,
    dry_run: false,
};

fn ivf() -> IvfOption {
    IvfOption::new(DIM).n_clusters(2).n_probe(2)
}

/// A schema with `option` as field `vec`, built without
/// `Schema::validate_for_create` (which `SchemaBuilder::build` runs).
fn schema_with(option: FieldOption) -> Schema {
    let mut schema = Schema::new();
    schema.fields.insert("vec".to_string(), option);
    schema
}

fn create_storage(dir: &TempDir) -> Arc<dyn Storage> {
    let config = StorageConfig::File(FileStorageConfig::new(dir.path()));
    StorageFactory::create(config).unwrap()
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

async fn put_four(engine: &Engine) {
    for i in 0..4 {
        let doc = Document::builder().add_field("vec", vector(i)).build();
        engine.put_document(&format!("d{i}"), doc).await.unwrap();
    }
    engine.commit().await.unwrap();
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

fn assert_rejects(err: &LaurusError, field: &str, reason: &str) {
    let expected = format!("field '{field}': {reason}");
    assert!(
        matches!(err, LaurusError::InvalidArgument(m) if m == &expected),
        "expected InvalidArgument({expected:?}), got {err:?}"
    );
}

/// Index creation, `add_field` and `update_field` all reject each value
/// before the schema or the data change.
#[tokio::test(flavor = "multi_thread")]
async fn defining_a_field_with_unusable_parameters_fails() {
    let cases: Vec<(FieldOption, FieldOption, &str)> = vec![
        (
            FieldOption::Hnsw(HnswOption::new(DIM).m(1)),
            FieldOption::Hnsw(HnswOption::new(DIM)),
            "HNSW m must be at least 2, got 1",
        ),
        (
            FieldOption::Hnsw(HnswOption::new(DIM).m(0)),
            FieldOption::Hnsw(HnswOption::new(DIM)),
            "HNSW m must be at least 2, got 0",
        ),
        (
            FieldOption::Hnsw(HnswOption::new(DIM).ef_construction(0)),
            FieldOption::Hnsw(HnswOption::new(DIM)),
            "HNSW ef_construction must be at least 1",
        ),
        (
            FieldOption::Hnsw(HnswOption::new(0)),
            FieldOption::Hnsw(HnswOption::new(DIM)),
            "HNSW dimension must be greater than 0",
        ),
        (
            FieldOption::Ivf(ivf().n_clusters(0)),
            FieldOption::Ivf(ivf()),
            "IVF n_clusters must be at least 1",
        ),
        (
            FieldOption::Ivf(ivf().n_probe(0)),
            FieldOption::Ivf(ivf()),
            "IVF n_probe must be at least 1",
        ),
        (
            FieldOption::Ivf(ivf().dimension(0)),
            FieldOption::Ivf(ivf()),
            "IVF dimension must be greater than 0",
        ),
        (
            FieldOption::Flat(FlatOption::new(0)),
            FieldOption::Flat(FlatOption::new(DIM)),
            "Flat dimension must be greater than 0",
        ),
    ];
    for (rejected, valid, reason) in cases {
        assert_rejects(
            &schema_with(rejected.clone())
                .validate_for_create()
                .unwrap_err(),
            "vec",
            reason,
        );

        let dir = TempDir::new().unwrap();
        let engine = Engine::new(create_storage(&dir), schema_with(valid))
            .await
            .unwrap();
        put_four(&engine).await;
        let before = search_ids(&engine).await;
        assert_eq!(before.len(), 4, "{reason}");
        let schema_before = format!("{:?}", engine.schema().fields["vec"]);

        let err = engine
            .add_field("vec2", rejected.clone())
            .await
            .unwrap_err();
        assert_rejects(&err, "vec2", reason);
        assert!(!engine.schema().fields.contains_key("vec2"), "{reason}");

        let err = engine
            .update_field("vec", rejected, REINDEX)
            .await
            .unwrap_err();
        assert_rejects(&err, "vec", reason);
        assert_eq!(
            format!("{:?}", engine.schema().fields["vec"]),
            schema_before,
            "{reason}"
        );
        assert_eq!(search_ids(&engine).await, before, "{reason}");
    }
}

/// An index created with HNSW `m` 0 (what gRPC and HTTP persisted for an
/// omitted `m`) opens, and `update_field` rebuilds the field with a valid
/// `m`, after which it takes documents and reopens.
#[tokio::test(flavor = "multi_thread")]
async fn index_persisted_with_hnsw_m_0_opens_and_update_field_repairs_it() {
    let dir = TempDir::new().unwrap();
    let broken = FieldOption::Hnsw(HnswOption::new(DIM).m(0));
    let engine = Engine::new(create_storage(&dir), schema_with(broken.clone()))
        .await
        .unwrap();
    engine.commit().await.unwrap();
    drop(engine);

    // The schema.toml such an index holds still parses and opens.
    let persisted = Schema::from_toml(&schema_with(broken).to_toml().unwrap()).unwrap();
    let engine = Engine::new(open_storage(&dir), persisted).await.unwrap();

    let outcome = engine
        .update_field("vec", FieldOption::Hnsw(HnswOption::new(DIM)), REINDEX)
        .await
        .unwrap();
    assert_eq!(outcome.classification, FieldChangeKind::Reindex);
    put_four(&engine).await;
    assert_eq!(search_ids(&engine).await.len(), 4);
    let repaired = engine.schema();
    drop(engine);

    let reopened = Engine::new(open_storage(&dir), repaired).await.unwrap();
    assert_eq!(search_ids(&reopened).await.len(), 4);
}

/// An index created with IVF `n_probe` 0 holds documents no search finds.
/// It opens, and a metadata-only `update_field` of `n_probe` makes them
/// searchable once the index is reopened.
#[tokio::test(flavor = "multi_thread")]
async fn index_persisted_with_ivf_n_probe_0_opens_and_update_field_repairs_it() {
    let dir = TempDir::new().unwrap();
    let broken = FieldOption::Ivf(ivf().n_probe(0));
    let engine = Engine::new(create_storage(&dir), schema_with(broken.clone()))
        .await
        .unwrap();
    put_four(&engine).await;
    assert_eq!(search_ids(&engine).await, Vec::<String>::new());
    drop(engine);

    let persisted = Schema::from_toml(&schema_with(broken).to_toml().unwrap()).unwrap();
    let engine = Engine::new(open_storage(&dir), persisted).await.unwrap();
    let outcome = engine
        .update_field("vec", FieldOption::Ivf(ivf()), REINDEX)
        .await
        .unwrap();
    assert_eq!(outcome.classification, FieldChangeKind::MetadataOnly);
    let repaired = engine.schema();
    drop(engine);

    let reopened = Engine::new(open_storage(&dir), repaired).await.unwrap();
    assert_eq!(search_ids(&reopened).await.len(), 4);
}
