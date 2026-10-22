//! Issue #1527: a vector field's `update_field` Reindex keeps the deletes
//! and upserts made since the last commit.
//!
//! The Reindex reopens the field's index over its files and merges every
//! segment into one. Before the fix, the reopened index read only the
//! deletion bitmap on storage, so a delete made since the last commit was
//! not in it: the merge kept the deleted document, and the next commit
//! persisted the bitmap without it and truncated the delete's WAL record.

use tempfile::TempDir;

use laurus::DistanceMetric;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::{FlatOption, HnswOption, IvfOption, Vector};
use laurus::{DataValue, Document, Engine, QuantizationMethod, UpdateFieldOptions};
use laurus::{FieldOption, QueryVector, Schema, SearchRequestBuilder, VectorSearchQuery};

const DIM: usize = 24;
const FIELD: &str = "embedding";
const DOCS: u64 = 20;
const DELETED: u64 = 7;
const UPDATED: u64 = 3;

/// Deterministic values; Float16 stores them rounded, so a document is
/// still the nearest neighbour of its own values.
fn values(seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..DIM)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The new values `UPDATED` is put with after the commit.
fn updated_values() -> Vec<f32> {
    values(10_000 + UPDATED)
}

/// `(kind, option before, option after)`: each change needs a Reindex.
fn cases() -> Vec<(&'static str, FieldOption, FieldOption)> {
    let flat = |quantizer| {
        FieldOption::Flat(FlatOption {
            dimension: DIM,
            distance: DistanceMetric::Euclidean,
            quantizer,
            ..FlatOption::default()
        })
    };
    // Probing every list makes IVF exact.
    let ivf = |n_clusters| {
        FieldOption::Ivf(IvfOption {
            dimension: DIM,
            distance: DistanceMetric::Euclidean,
            n_clusters,
            n_probe: n_clusters,
            quantizer: QuantizationMethod::Float16,
            ..IvfOption::default()
        })
    };
    let hnsw = |m| {
        FieldOption::Hnsw(HnswOption {
            dimension: DIM,
            distance: DistanceMetric::Euclidean,
            m,
            quantizer: QuantizationMethod::Float16,
            ..HnswOption::default()
        })
    };
    vec![
        (
            "flat",
            flat(QuantizationMethod::Scalar8Bit),
            flat(QuantizationMethod::Float16),
        ),
        ("ivf", ivf(2), ivf(3)),
        ("hnsw", hnsw(16), hnsw(8)),
    ]
}

fn open_storage(dir: &TempDir) -> std::sync::Arc<dyn Storage> {
    StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path()))).unwrap()
}

async fn put(engine: &Engine, id: u64, values: Vec<f32>) {
    let doc = Document::builder()
        .add_field(FIELD, DataValue::Vector(values))
        .build();
    engine.put_document(&id.to_string(), doc).await.unwrap();
}

/// The ids and scores of the top `limit` hits for `query`.
async fn search(engine: &Engine, query: Vec<f32>, limit: usize) -> Vec<(String, f32)> {
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
        .unwrap()
        .into_iter()
        .map(|r| (r.id, r.score))
        .collect()
}

fn reindex() -> UpdateFieldOptions {
    UpdateFieldOptions {
        reindex: true,
        ..Default::default()
    }
}

/// Record in `failures` when `DELETED` is found, even by its own values,
/// or when another document no longer is.
async fn check_deleted(engine: &Engine, what: &str, failures: &mut Vec<String>) {
    let hits = search(engine, values(DELETED), DOCS as usize).await;
    if hits.iter().any(|(id, _)| id == &DELETED.to_string()) {
        failures.push(format!("{what}: deleted doc {DELETED} came back"));
    }
    let hits = search(engine, values(1), 1).await;
    if hits.first().map(|h| h.0.as_str()) != Some("1") {
        failures.push(format!("{what}: doc 1 is not found: {hits:?}"));
    }
}

/// Record in `failures` when `UPDATED` has more than one hit, or still
/// matches its old values.
async fn check_updated(engine: &Engine, what: &str, failures: &mut Vec<String>) {
    // Euclidean scores are `exp(-distance)`: the old version scores about
    // 1, the new one is far from the old values.
    let hits = search(engine, values(UPDATED), DOCS as usize).await;
    let updated: Vec<f32> = hits
        .iter()
        .filter(|(id, _)| id == &UPDATED.to_string())
        .map(|(_, score)| *score)
        .collect();
    if updated.len() != 1 || updated[0] >= 0.5 {
        failures.push(format!(
            "{what}: doc {UPDATED} scores {updated:?} for its old values"
        ));
    }
    let hits = search(engine, updated_values(), 1).await;
    if hits.first().map(|h| h.0.as_str()) != Some("3") {
        failures.push(format!(
            "{what}: the new values do not find doc {UPDATED}: {hits:?}"
        ));
    }
}

/// Open an engine over a new directory with `option`, put every document
/// and commit.
async fn committed_engine(option: FieldOption) -> (TempDir, Engine) {
    let dir = TempDir::new().unwrap();
    let schema = Schema::builder().add_field(FIELD, option).build();
    let engine = Engine::new(open_storage(&dir), schema).await.unwrap();
    for id in 0..DOCS {
        put(&engine, id, values(id)).await;
    }
    engine.commit().await.unwrap();
    (dir, engine)
}

/// Issue #1527's steps: a delete since the last commit survives a Reindex,
/// the next commit, and a reopen.
#[tokio::test(flavor = "multi_thread")]
async fn a_delete_since_the_last_commit_survives_a_reindex() {
    let mut failures = Vec::new();
    for (kind, before, after) in cases() {
        let (dir, engine) = committed_engine(before).await;
        engine.delete_documents(&DELETED.to_string()).await.unwrap();
        engine.update_field(FIELD, after, reindex()).await.unwrap();
        check_deleted(
            &engine,
            &format!("{kind}: after the reindex"),
            &mut failures,
        )
        .await;

        engine.commit().await.unwrap();
        check_deleted(
            &engine,
            &format!("{kind}: after the next commit"),
            &mut failures,
        )
        .await;

        let schema = engine.schema();
        drop(engine);
        let engine = Engine::new(open_storage(&dir), schema).await.unwrap();
        check_deleted(&engine, &format!("{kind}: after a reopen"), &mut failures).await;
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// An upsert since the last commit marks the old version deleted: after a
/// Reindex, the old values find the document only through its new values.
#[tokio::test(flavor = "multi_thread")]
async fn an_upsert_since_the_last_commit_survives_a_reindex() {
    let mut failures = Vec::new();
    for (kind, before, after) in cases() {
        let (dir, engine) = committed_engine(before).await;
        put(&engine, UPDATED, updated_values()).await;
        engine.update_field(FIELD, after, reindex()).await.unwrap();
        check_updated(
            &engine,
            &format!("{kind}: after the reindex"),
            &mut failures,
        )
        .await;

        engine.commit().await.unwrap();
        check_updated(
            &engine,
            &format!("{kind}: after the next commit"),
            &mut failures,
        )
        .await;

        let schema = engine.schema();
        drop(engine);
        let engine = Engine::new(open_storage(&dir), schema).await.unwrap();
        check_updated(&engine, &format!("{kind}: after a reopen"), &mut failures).await;
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
