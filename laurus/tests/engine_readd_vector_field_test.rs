//! Issue #1535: a vector field removed with `delete_field` and added again
//! under the same name starts empty.
//!
//! Before the fix, `add_field` reopened the files the removed field left.
//! A delete made since the last commit before the removal, or while the
//! field was removed, never reached them, so the deleted document came back;
//! and the files were read under the new option, so a re-add under another
//! dimension or index kind broke the field's searches. A field recreated
//! empty also starts after a commit, so recovery never replays a put made
//! before it into it.

use tempfile::TempDir;

use laurus::DistanceMetric;
use laurus::VectorSearchQuery;
use laurus::lexical::TextOption;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::{FlatOption, HnswOption, IvfOption, Vector};
use laurus::{DataValue, Document, Engine, QuantizationMethod, UpdateFieldOptions};
use laurus::{FieldChangeKind, FieldOption, QueryVector, Schema, SearchRequestBuilder};

const DIM: usize = 24;
const OTHER_DIM: usize = 16;
const FIELD: &str = "embedding";
const DOCS: u64 = 20;
const DELETED: u64 = 7;
/// A document put after the field is added again.
const NEW: u64 = 100;
/// A document put, and not committed, before the field is recreated.
const UNCOMMITTED: u64 = 50;

/// Deterministic values; Float16 stores them rounded, so a document is
/// still the nearest neighbour of its own values.
fn values_of(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn flat(dimension: usize) -> FieldOption {
    FieldOption::Flat(FlatOption {
        dimension,
        distance: DistanceMetric::Euclidean,
        quantizer: QuantizationMethod::Float16,
        ..FlatOption::default()
    })
}

fn hnsw(dimension: usize) -> FieldOption {
    FieldOption::Hnsw(HnswOption {
        dimension,
        distance: DistanceMetric::Euclidean,
        quantizer: QuantizationMethod::Float16,
        ..HnswOption::default()
    })
}

/// Probing every list makes IVF exact.
fn ivf(dimension: usize) -> FieldOption {
    FieldOption::Ivf(IvfOption {
        dimension,
        distance: DistanceMetric::Euclidean,
        n_clusters: 2,
        n_probe: 2,
        quantizer: QuantizationMethod::Float16,
        ..IvfOption::default()
    })
}

fn kinds() -> Vec<(&'static str, FieldOption)> {
    vec![("flat", flat(DIM)), ("ivf", ivf(DIM)), ("hnsw", hnsw(DIM))]
}

fn open_storage(dir: &TempDir) -> std::sync::Arc<dyn Storage> {
    StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path()))).unwrap()
}

/// Put `id` with a title, and with `values` in the vector field if given.
async fn put(engine: &Engine, id: u64, values: Option<Vec<f32>>) {
    let mut doc = Document::builder().add_field("title", DataValue::Text(format!("doc {id}")));
    if let Some(values) = values {
        doc = doc.add_field(FIELD, DataValue::Vector(values));
    }
    engine
        .put_document(&id.to_string(), doc.build())
        .await
        .unwrap();
}

/// The ids of the top `limit` hits for `query`.
async fn search(engine: &Engine, query: Vec<f32>, limit: usize) -> Result<Vec<String>, String> {
    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(query),
            weight: 1.0,
            fields: Some(vec![FIELD.to_string()]),
        }]))
        .limit(limit)
        .build();
    engine
        .search(request)
        .await
        .map(|hits| hits.into_iter().map(|hit| hit.id).collect())
        .map_err(|e| e.to_string())
}

/// Record in `failures` when the field holds anything: a search by
/// `DELETED`'s own values must return nothing.
async fn check_empty(engine: &Engine, dim: usize, what: &str, failures: &mut Vec<String>) {
    match search(engine, values_of(DELETED, dim), DOCS as usize).await {
        Ok(hits) if hits.is_empty() => {}
        Ok(hits) => failures.push(format!("{what}: the field holds {hits:?}")),
        Err(e) => failures.push(format!("{what}: search failed: {e}")),
    }
}

/// Record in `failures` when the field holds anything but `NEW`: a search
/// by `DELETED`'s own values must return `NEW` alone, and `NEW`'s values
/// must find it.
async fn check_only_new(engine: &Engine, dim: usize, what: &str, failures: &mut Vec<String>) {
    let new = NEW.to_string();
    match search(engine, values_of(DELETED, dim), DOCS as usize).await {
        Ok(hits) if hits == [new.clone()] => {}
        Ok(hits) => failures.push(format!("{what}: the field holds {hits:?}, not only {NEW}")),
        Err(e) => failures.push(format!("{what}: search failed: {e}")),
    }
    match search(engine, values_of(NEW, dim), 1).await {
        Ok(hits) if hits == [new] => {}
        Ok(hits) => failures.push(format!("{what}: {NEW} is not found: {hits:?}")),
        Err(e) => failures.push(format!("{what}: search failed: {e}")),
    }
}

/// An engine over a new directory with a title and the vector field under
/// `option`, holding `DOCS` committed documents.
async fn committed_engine(option: FieldOption) -> (TempDir, Engine) {
    let dir = TempDir::new().unwrap();
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field(FIELD, option)
        .build();
    let engine = Engine::new(open_storage(&dir), schema).await.unwrap();
    for id in 0..DOCS {
        put(&engine, id, Some(values_of(id, DIM))).await;
    }
    engine.commit().await.unwrap();
    (dir, engine)
}

/// Reopen the engine over `dir` with its current schema.
async fn reopen(dir: &TempDir, engine: Engine) -> Result<Engine, String> {
    let schema = engine.schema();
    drop(engine);
    Engine::new(open_storage(dir), schema)
        .await
        .map_err(|e| e.to_string())
}

/// Issue #1535's cases, and an update while the field is removed: after a
/// re-add under the same option, `DELETED` is not found, nor any document
/// from before the removal, after the re-add, a commit and a reopen. A
/// document put after the re-add is found once committed.
#[tokio::test(flavor = "multi_thread")]
async fn a_re_added_field_starts_empty() {
    let mut failures = Vec::new();
    for (kind, option) in kinds() {
        for case in [
            "delete, then remove",
            "remove, then delete",
            "remove, then update",
        ] {
            let what = format!("{kind}, {case}");
            let (dir, engine) = committed_engine(option.clone()).await;
            match case {
                "delete, then remove" => {
                    engine.delete_documents(&DELETED.to_string()).await.unwrap();
                    engine.delete_field(FIELD).await.unwrap();
                }
                "remove, then delete" => {
                    engine.delete_field(FIELD).await.unwrap();
                    engine.delete_documents(&DELETED.to_string()).await.unwrap();
                    engine.commit().await.unwrap();
                }
                _ => {
                    engine.delete_field(FIELD).await.unwrap();
                    put(&engine, DELETED, None).await;
                    engine.commit().await.unwrap();
                }
            }
            engine.add_field(FIELD, option.clone()).await.unwrap();
            check_empty(
                &engine,
                DIM,
                &format!("{what}: after the re-add"),
                &mut failures,
            )
            .await;

            put(&engine, NEW, Some(values_of(NEW, DIM))).await;
            engine.commit().await.unwrap();
            check_only_new(
                &engine,
                DIM,
                &format!("{what}: after a commit"),
                &mut failures,
            )
            .await;

            match reopen(&dir, engine).await {
                Ok(engine) => {
                    check_only_new(
                        &engine,
                        DIM,
                        &format!("{what}: after a reopen"),
                        &mut failures,
                    )
                    .await
                }
                Err(e) => failures.push(format!("{what}: the reopen failed: {e}")),
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A re-add under another dimension or index kind does not read the old
/// files: the field is empty, and then holds only what was put after it,
/// also after a reopen.
#[tokio::test(flavor = "multi_thread")]
async fn a_re_add_under_another_option_starts_empty() {
    let mut failures = Vec::new();
    let cases = [
        ("another dimension", flat(OTHER_DIM), OTHER_DIM),
        ("another kind", hnsw(DIM), DIM),
    ];
    for (what, option, dim) in cases {
        let (dir, engine) = committed_engine(flat(DIM)).await;
        engine.delete_field(FIELD).await.unwrap();
        engine.add_field(FIELD, option).await.unwrap();
        check_empty(
            &engine,
            dim,
            &format!("{what}: after the re-add"),
            &mut failures,
        )
        .await;

        put(&engine, NEW, Some(values_of(NEW, dim))).await;
        engine.commit().await.unwrap();
        check_only_new(
            &engine,
            dim,
            &format!("{what}: after a commit"),
            &mut failures,
        )
        .await;
        match reopen(&dir, engine).await {
            Ok(engine) => {
                check_only_new(
                    &engine,
                    dim,
                    &format!("{what}: after a reopen"),
                    &mut failures,
                )
                .await
            }
            Err(e) => failures.push(format!("{what}: the reopen failed: {e}")),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A put made before the field is recreated, and not committed, is not
/// replayed into the recreated field when the engine is dropped without a
/// commit and reopened. Under another dimension that replay failed, and the
/// reopen with it.
#[tokio::test(flavor = "multi_thread")]
async fn a_recreated_field_starts_empty_across_a_reopen_without_a_commit() {
    let reindex = UpdateFieldOptions {
        reindex: true,
        ..Default::default()
    };
    let mut failures = Vec::new();
    for (how, dim) in [
        ("re-add, same dimension", DIM),
        ("re-add, another dimension", OTHER_DIM),
        ("destructive update, another dimension", OTHER_DIM),
    ] {
        let (dir, engine) = committed_engine(flat(DIM)).await;
        put(&engine, UNCOMMITTED, Some(values_of(UNCOMMITTED, DIM))).await;
        if how.starts_with("re-add") {
            engine.delete_field(FIELD).await.unwrap();
            engine.add_field(FIELD, flat(dim)).await.unwrap();
        } else {
            let outcome = engine
                .update_field(FIELD, flat(dim), reindex)
                .await
                .unwrap();
            assert_eq!(outcome.classification, FieldChangeKind::Destructive);
        }
        put(&engine, NEW, Some(values_of(NEW, dim))).await;

        match reopen(&dir, engine).await {
            Ok(engine) => {
                check_only_new(
                    &engine,
                    dim,
                    &format!("{how}: after a reopen"),
                    &mut failures,
                )
                .await
            }
            Err(e) => failures.push(format!("{how}: the reopen failed: {e}")),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
