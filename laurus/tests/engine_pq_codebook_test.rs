//! End-to-end engine tests for `Engine::train_pq_codebook` (Issue #631 PR-2).
//!
//! Complements `pq_shared_codebook_test.rs` (which drives the HNSW index
//! layer directly) by exercising the schema-level wiring: a
//! `pq_codebook_path` on the schema's `HnswOption` must reach the HNSW
//! writer through `Engine` → `VectorStore` → `from_hnsw_option` →
//! `resolve_pq_codebook`, and `Engine::train_pq_codebook` must produce a
//! codebook that a subsequently opened engine encodes with.
//!
//! Issue #1735: an open engine that rebuilds, adds or recreates the field
//! loads the codebook too, and a change of its search-time parameters keeps
//! the one in use. Each used to leave the field with no codebook.
//!
//! Issue #1738: the path is relative to the root of the vector storage, and a
//! file of that name in the field's own directory, such as the codebook of a
//! field nested under it, is not the field's.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use laurus::storage::file::FileStorageConfig;
use laurus::storage::manifest::load_checksummed_json;
use laurus::storage::prefixed::PrefixedStorage;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::index::hnsw::reader::HnswIndexReader;
use laurus::vector::index::pq_codebook::{read_pq_codebook, train_and_write_pq_codebook};
use laurus::vector::index::storage::VectorStorage;
use laurus::vector::{FlatOption, HnswOption, IvfOption};
use laurus::{
    DataValue, Document, Engine, FieldChangeKind, FieldOption, QueryVector, Schema,
    SearchRequestBuilder, TextOption, UpdateFieldOptions, VectorSearchQuery,
};
use tempfile::TempDir;

const DIM: usize = 32;
const M: usize = 4;

/// Deterministic pseudo-random vectors (same LCG as
/// `pq_shared_codebook_test.rs`); distinct seeds give distinct corpora.
fn make_vectors(count: usize, seed: u64) -> Vec<Vector> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            let data: Vec<f32> = (0..DIM)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            Vector::new(data)
        })
        .collect()
}

/// Schema with one HNSW + PQ field pointing at a shared codebook file.
fn pq_schema(pq_codebook_path: Option<&str>) -> Schema {
    Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Hnsw(HnswOption {
                dimension: DIM,
                distance: DistanceMetric::Euclidean,
                m: 8,
                ef_construction: 32,
                quantizer: QuantizationMethod::ProductQuantization { subvector_count: M },
                pq_codebook_path: pq_codebook_path.map(str::to_string),
                ..HnswOption::default()
            }),
        )
        .build()
}

async fn file_engine(dir: &TempDir, schema: Schema) -> laurus::Result<Engine> {
    let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path())))?;
    Engine::new(storage, schema).await
}

/// `train_pq_codebook` must reject fields it cannot train for, naming the
/// problem: unknown field, non-HNSW field, HNSW field without PQ.
#[tokio::test(flavor = "multi_thread")]
async fn train_pq_codebook_validates_the_field() -> laurus::Result<()> {
    let dir = TempDir::new().unwrap();
    let schema = Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Hnsw(HnswOption {
                dimension: DIM,
                distance: DistanceMetric::Euclidean,
                // Scalar8Bit (the default): no PQ, so training must be refused.
                ..HnswOption::default()
            }),
        )
        .add_field(
            "title",
            FieldOption::Text(laurus::TextOption {
                indexed: true,
                stored: true,
                ..Default::default()
            }),
        )
        .build();
    let engine = file_engine(&dir, schema).await?;
    let vectors = make_vectors(10, 1);

    let err = engine
        .train_pq_codebook("missing", &vectors, None)
        .unwrap_err();
    assert!(
        err.to_string().contains("not defined in the schema"),
        "unknown field must be named: {err}"
    );

    let err = engine
        .train_pq_codebook("title", &vectors, None)
        .unwrap_err();
    assert!(
        err.to_string().contains("not an HNSW vector field"),
        "non-vector field must be refused: {err}"
    );

    let err = engine
        .train_pq_codebook("embedding", &vectors, None)
        .unwrap_err();
    assert!(
        err.to_string().contains("ProductQuantization"),
        "a non-PQ quantizer must be refused: {err}"
    );

    Ok(())
}

/// The full schema-level loop: with `pq_codebook_path` configured,
/// (1) a commit before training hard-errors with the training command,
/// (2) `train_pq_codebook` writes the codebook into the vector namespace,
/// (3) a reopened engine commits successfully and the sealed segment's
/// embedded codebook is byte-identical to the trained file — proving the
/// writer encoded with the shared codebook instead of retraining (the
/// 20-doc commit is far below `PQ_MIN_TRAIN_VECTORS`, so a silent
/// fresh-training fallback would have produced a Scalar8Bit segment).
#[tokio::test(flavor = "multi_thread")]
async fn train_then_reopen_commits_with_the_shared_codebook() -> laurus::Result<()> {
    let dir = TempDir::new().unwrap();
    let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path())))?;

    // (1) Before training: the configured-but-missing codebook must fail the
    // commit loudly (no silent per-segment retrain), pointing at the fix.
    {
        let engine = Engine::new(storage.clone(), pq_schema(Some("embedding.pqcb"))).await?;
        let doc = Document::builder()
            .add_field(
                "embedding",
                DataValue::Vector(make_vectors(1, 7)[0].data.to_vec()),
            )
            .build();
        engine.put_document("doc-early", doc).await?;
        let err = engine.commit().await.unwrap_err();
        assert!(
            err.to_string().contains("train"),
            "an untrained configured codebook must point at the training step: {err}"
        );
    }

    // (2) Train on a corpus DISTINCT from what gets indexed below, so an
    // accidental inline retrain could not reproduce the same centroids.
    let fresh_dir = TempDir::new().unwrap();
    let fresh_storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(
        fresh_dir.path(),
    )))?;
    let trainer = Engine::new(fresh_storage.clone(), pq_schema(Some("embedding.pqcb"))).await?;
    let info = trainer.train_pq_codebook("embedding", &make_vectors(400, 0x0F0F), None)?;
    assert_eq!(info.path, "embedding.pqcb");
    assert_eq!(info.subvector_count, M);
    assert_eq!(info.centroids, 256);
    assert_eq!(info.sub_dimension, DIM / M);
    assert_eq!(info.dimension, DIM);
    assert_eq!(info.training_vectors, 400);
    assert!(
        fresh_storage.file_exists("vector/embedding.pqcb"),
        "the codebook must land inside the engine's vector storage namespace"
    );
    drop(trainer);

    // (3) A reopened engine picks the codebook up at open and commits a
    // tiny (< PQ_MIN_TRAIN_VECTORS) batch on PQ.
    let engine = Engine::new(fresh_storage.clone(), pq_schema(Some("embedding.pqcb"))).await?;
    for (i, v) in make_vectors(20, 0xA5A5).iter().enumerate() {
        let doc = Document::builder()
            .add_field("embedding", DataValue::Vector(v.data.to_vec()))
            .build();
        engine.put_document(&format!("doc{i}"), doc).await?;
    }
    engine.commit().await?;

    // The sealed segment's embedded codebook must equal the trained file.
    // The codebook itself lives at the ROOT of the vector namespace (Issue
    // #948: `pq_codebook_path` is resolved against the root storage,
    // before per-field `PrefixedStorage` wrapping -- see
    // `multi_field`'s module docs), while the field's own segment lives
    // one level deeper, under its own `MultiFieldVectorIndex` sub-namespace.
    let vector_storage: Arc<dyn Storage> =
        Arc::new(PrefixedStorage::new("vector", fresh_storage.clone()));
    let trained = read_pq_codebook(vector_storage.as_ref(), "embedding.pqcb")?;
    let field_storage: Arc<dyn Storage> =
        Arc::new(PrefixedStorage::new("embedding", vector_storage));
    let reader = HnswIndexReader::load(field_storage, "segment_000000", DistanceMetric::Euclidean)?;
    let pool = match reader.vectors() {
        VectorStorage::OwnedPq(pool) => pool.clone(),
        other => panic!("the committed segment must stay on PQ, got {other:?}"),
    };
    assert_eq!(pool.params, trained.params);
    assert_eq!(
        pool.codebook, trained.codebook,
        "the segment must embed the shared codebook byte-for-byte (no retrain)"
    );

    Ok(())
}

/// `output: Some(..)` must override both the configured path and the
/// default `{field}.pqcb` naming, so a v2 codebook can be trained next to
/// the live one.
#[tokio::test(flavor = "multi_thread")]
async fn train_pq_codebook_honors_the_output_override() -> laurus::Result<()> {
    let dir = TempDir::new().unwrap();
    let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path())))?;
    let engine = Engine::new(storage.clone(), pq_schema(None)).await?;

    // No configured path, no override: default name.
    let info = engine.train_pq_codebook("embedding", &make_vectors(300, 3), None)?;
    assert_eq!(info.path, "embedding.pqcb");
    assert!(storage.file_exists("vector/embedding.pqcb"));

    // Override wins.
    let info = engine.train_pq_codebook(
        "embedding",
        &make_vectors(300, 4),
        Some("embedding.v2.pqcb"),
    )?;
    assert_eq!(info.path, "embedding.v2.pqcb");
    assert!(storage.file_exists("vector/embedding.v2.pqcb"));

    Ok(())
}

// --- Issue #1735: changes of an open engine ---

const REINDEX: UpdateFieldOptions = UpdateFieldOptions {
    reindex: true,
    dry_run: false,
};

/// An HNSW option with product quantization and the shared codebook
/// `embedding.pqcb`, changed by `change`.
fn hnsw(change: impl Fn(&mut HnswOption)) -> FieldOption {
    let mut option = HnswOption {
        dimension: DIM,
        distance: DistanceMetric::Euclidean,
        quantizer: QuantizationMethod::ProductQuantization { subvector_count: M },
        pq_codebook_path: Some("embedding.pqcb".to_string()),
        ..HnswOption::default()
    };
    change(&mut option);
    FieldOption::Hnsw(option)
}

fn flat() -> FieldOption {
    FieldOption::Flat(FlatOption {
        dimension: DIM,
        distance: DistanceMetric::Euclidean,
        ..FlatOption::default()
    })
}

fn ivf() -> FieldOption {
    FieldOption::Ivf(IvfOption {
        dimension: DIM,
        distance: DistanceMetric::Euclidean,
        n_clusters: 2,
        n_probe: 2,
        ..IvfOption::default()
    })
}

/// A schema with `option`, if any, as field `embedding`, and a text field.
fn schema_with(option: Option<FieldOption>) -> Schema {
    let mut schema = Schema::new();
    schema.fields.insert(
        "title".to_string(),
        FieldOption::Text(TextOption::default()),
    );
    if let Some(option) = option {
        schema.fields.insert("embedding".to_string(), option);
    }
    schema
}

fn file_storage(dir: &TempDir) -> Arc<dyn Storage> {
    StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path()))).unwrap()
}

async fn open(storage: &Arc<dyn Storage>, option: Option<FieldOption>) -> Engine {
    Engine::new(storage.clone(), schema_with(option))
        .await
        .unwrap()
}

/// The root of the vector storage, to which `pq_codebook_path` is relative.
fn vector_root(storage: &Arc<dyn Storage>) -> Arc<dyn Storage> {
    Arc::new(PrefixedStorage::new("vector", storage.clone()))
}

/// Train codebook `name` on `seed`'s vectors, and return its centroids.
fn train(storage: &Arc<dyn Storage>, name: &str, seed: u64) -> Vec<f32> {
    train_shape(storage, name, seed, DIM, M)
}

fn train_shape(
    storage: &Arc<dyn Storage>,
    name: &str,
    seed: u64,
    dimension: usize,
    subvectors: usize,
) -> Vec<f32> {
    let sample: Vec<Vector> = make_vectors(400, seed)
        .into_iter()
        .map(|v| Vector::new(v.data[..dimension].to_vec()))
        .collect();
    let root = vector_root(storage);
    train_and_write_pq_codebook(
        root.as_ref(),
        name,
        dimension,
        subvectors,
        256,
        false,
        &sample,
    )
    .unwrap();
    read_pq_codebook(root.as_ref(), name).unwrap().codebook
}

/// Damage codebook `embedding.pqcb`, through the storage, which gives up its
/// mapping of the file: Windows refuses to write to a mapped one.
fn damage(storage: &Arc<dyn Storage>) {
    const NAME: &str = "vector/embedding.pqcb";
    const STAGED: &str = "vector/embedding.pqcb.damaged";
    let mut bytes = Vec::new();
    storage
        .open_input(NAME)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    let mut output = storage.create_output(STAGED).unwrap();
    output.write_all(&bytes).unwrap();
    output.close().unwrap();
    storage.rename_file(STAGED, NAME).unwrap();
}

/// Put documents `ids`, each with a vector of its own.
async fn put(engine: &Engine, ids: std::ops::Range<u64>) {
    for id in ids {
        let doc = Document::builder()
            .add_vector("embedding", make_vectors(1, 1000 + id)[0].data.to_vec())
            .build();
        engine.put_document(&format!("d{id}"), doc).await.unwrap();
    }
}

/// The ids a search for everything finds.
async fn found(engine: &Engine) -> BTreeSet<String> {
    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: make_vectors(1, 1)[0].clone(),
            weight: 1.0,
            fields: Some(vec!["embedding".to_string()]),
        }]))
        .limit(200)
        .build();
    engine
        .search(request)
        .await
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect()
}

fn ids(numbers: impl IntoIterator<Item = u64>) -> BTreeSet<String> {
    numbers.into_iter().map(|id| format!("d{id}")).collect()
}

/// The centroids each of the field's HNSW segments embeds, `None` for one
/// that is not product-quantized, in the order of the field's manifest,
/// oldest first. A file of a replaced segment can stay on Windows until the
/// next open, so the files are not listed.
fn segment_codebooks(storage: &Arc<dyn Storage>) -> Vec<Option<Vec<f32>>> {
    segment_codebooks_of(storage, "embedding")
}

fn segment_codebooks_of(storage: &Arc<dyn Storage>, field: &str) -> Vec<Option<Vec<f32>>> {
    let field: Arc<dyn Storage> = Arc::new(PrefixedStorage::new(field, vector_root(storage)));
    let Some((manifest, _)) =
        load_checksummed_json::<serde_json::Value>(field.as_ref(), "segments.json", None).unwrap()
    else {
        return Vec::new();
    };
    manifest["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|segment| {
            let name = segment["segment_id"].as_str().unwrap();
            let reader =
                HnswIndexReader::load(field.clone(), name, DistanceMetric::Euclidean).unwrap();
            match reader.vectors() {
                VectorStorage::OwnedPq(pool) => Some(pool.codebook.clone()),
                _ => None,
            }
        })
        .collect()
}

/// Why the field does not hold exactly `expected` in segments that embed
/// `codebook`, if it does not.
async fn wrong(
    engine: &Engine,
    storage: &Arc<dyn Storage>,
    expected: &BTreeSet<String>,
    codebook: &[f32],
) -> Option<String> {
    let found = found(engine).await;
    if found != *expected {
        return Some(format!(
            "{} of {} documents are found",
            found.intersection(expected).count(),
            expected.len()
        ));
    }
    let embedded = segment_codebooks(storage);
    if embedded.is_empty() || embedded.iter().any(|c| c.as_deref() != Some(codebook)) {
        return Some(format!(
            "segments with the codebook: {:?}",
            embedded
                .iter()
                .map(|c| c.as_deref() == Some(codebook))
                .collect::<Vec<_>>()
        ));
    }
    None
}

fn report(failures: &mut Vec<String>, what: &str, failure: Option<String>) {
    if let Some(failure) = failure {
        failures.push(format!("{what}: {failure}"));
    }
}

/// Every file of the field's directory, with its bytes.
fn field_files(dir: &TempDir) -> BTreeMap<String, Vec<u8>> {
    fn collect(dir: &Path, files: &mut BTreeMap<String, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.display().to_string();
            files.insert(name, std::fs::read(&path).unwrap());
        }
    }
    let mut files = BTreeMap::new();
    collect(&dir.path().join("vector/embedding"), &mut files);
    files
}

/// A rebuild of the field on an open engine writes its segments with the
/// shared codebook, as do the commits after it, and the field opens again:
/// within HNSW, from Flat and from IVF, with a path added, with another path,
/// and with a codebook that was trained again under the same name, which the
/// rebuild takes.
#[tokio::test(flavor = "multi_thread")]
async fn a_rebuild_on_an_open_engine_takes_the_shared_codebook() {
    let mut failures = Vec::new();
    let cases = [
        ("m", hnsw(|_| {}), hnsw(|o| o.m = 24)),
        ("from flat", flat(), hnsw(|_| {})),
        ("from ivf", ivf(), hnsw(|_| {})),
        (
            "a path added",
            hnsw(|o| o.pq_codebook_path = None),
            hnsw(|_| {}),
        ),
        (
            "another path",
            hnsw(|_| {}),
            hnsw(|o| o.pq_codebook_path = Some("embedding.v2.pqcb".to_string())),
        ),
        ("trained again", hnsw(|_| {}), hnsw(|o| o.m = 24)),
    ];
    for (what, before, after) in cases {
        let dir = TempDir::new().unwrap();
        let storage = file_storage(&dir);
        let first = train(&storage, "embedding.pqcb", 1);
        let engine = open(&storage, Some(before)).await;
        put(&engine, 0..20).await;
        engine.commit().await.unwrap();
        put(&engine, 20..25).await;
        let codebook = match what {
            "another path" => train(&storage, "embedding.v2.pqcb", 2),
            "trained again" => train(&storage, "embedding.pqcb", 2),
            _ => first,
        };
        match engine
            .update_field("embedding", after.clone(), REINDEX)
            .await
        {
            Ok(outcome) if outcome.classification == FieldChangeKind::Reindex => {}
            other => {
                failures.push(format!("{what}: {:?}", other.map(|o| o.classification)));
                continue;
            }
        }
        let expected = ids(0..25);
        report(
            &mut failures,
            what,
            wrong(&engine, &storage, &expected, &codebook).await,
        );
        put(&engine, 100..110).await;
        engine.commit().await.unwrap();
        let what = format!("{what}, after a commit");
        let expected = ids((0..25).chain(100..110));
        report(
            &mut failures,
            &what,
            wrong(&engine, &storage, &expected, &codebook).await,
        );
        drop(engine);
        let engine = open(&storage, Some(after)).await;
        put(&engine, 200..201).await;
        engine.commit().await.unwrap();
        let what = format!("{what} and a reopen");
        let expected = ids((0..25).chain(100..110).chain(200..201));
        report(
            &mut failures,
            &what,
            wrong(&engine, &storage, &expected, &codebook).await,
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A field that is added to an open engine, or added again after it was
/// deleted, or recreated by a destructive change, commits with the shared
/// codebook without a reopen.
#[tokio::test(flavor = "multi_thread")]
async fn a_field_added_to_an_open_engine_commits_with_the_shared_codebook() {
    let mut failures = Vec::new();
    for what in ["added", "added again", "recreated"] {
        let dir = TempDir::new().unwrap();
        let storage = file_storage(&dir);
        let codebook = train(&storage, "embedding.pqcb", 1);
        let after = match what {
            "recreated" => hnsw(|o| o.distance = DistanceMetric::Cosine),
            _ => hnsw(|_| {}),
        };
        let engine = match what {
            "added" => {
                let engine = open(&storage, None).await;
                engine.add_field("embedding", after.clone()).await.unwrap();
                engine
            }
            _ => {
                let engine = open(&storage, Some(hnsw(|_| {}))).await;
                put(&engine, 0..5).await;
                engine.commit().await.unwrap();
                if what == "added again" {
                    engine.delete_field("embedding").await.unwrap();
                    engine.add_field("embedding", after.clone()).await.unwrap();
                } else {
                    let outcome = engine
                        .update_field("embedding", after.clone(), REINDEX)
                        .await
                        .unwrap();
                    assert_eq!(outcome.classification, FieldChangeKind::Destructive);
                }
                engine
            }
        };
        for (from, to) in [(10, 30), (30, 31)] {
            put(&engine, from..to).await;
            if let Err(e) = engine.commit().await {
                failures.push(format!("{what}: the commit failed: {e}"));
            }
        }
        let expected = ids(10..31);
        report(
            &mut failures,
            what,
            wrong(&engine, &storage, &expected, &codebook).await,
        );
        drop(engine);
        let engine = open(&storage, Some(after)).await;
        let what = format!("{what}, reopened");
        report(
            &mut failures,
            &what,
            wrong(&engine, &storage, &expected, &codebook).await,
        );
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A change of the field's search-time parameters keeps the codebook the
/// engine was opened with: the commits after it and the merge they trigger
/// use it, whatever happened to the file since. The file takes effect when
/// the engine is opened again.
#[tokio::test(flavor = "multi_thread")]
async fn a_metadata_only_change_keeps_the_codebook_in_use() {
    let mut failures = Vec::new();
    let changes = [
        (
            "default_ef_search",
            hnsw(|o| o.default_ef_search = Some(99)),
        ),
        ("base_weight", hnsw(|o| o.base_weight = 2.0)),
    ];
    for (change, after) in changes {
        for file in ["as it was", "trained again", "damaged", "removed"] {
            let what = format!("{change}, the file {file}");
            let dir = TempDir::new().unwrap();
            let storage = file_storage(&dir);
            let codebook = train(&storage, "embedding.pqcb", 1);
            let engine = open(&storage, Some(hnsw(|_| {}))).await;
            put(&engine, 0..1).await;
            engine.commit().await.unwrap();
            let mut again = codebook.clone();
            match file {
                "trained again" => again = train(&storage, "embedding.pqcb", 2),
                "damaged" => damage(&storage),
                "removed" => storage.delete_file("vector/embedding.pqcb").unwrap(),
                _ => {}
            }
            match engine
                .update_field("embedding", after.clone(), UpdateFieldOptions::default())
                .await
            {
                Ok(outcome) if outcome.classification == FieldChangeKind::MetadataOnly => {}
                other => failures.push(format!("{what}: {:?}", other.map(|o| o.classification))),
            }
            // Ten segments of one document are one merge.
            for id in 1..10 {
                put(&engine, id..id + 1).await;
                if let Err(e) = engine.commit().await {
                    failures.push(format!("{what}: the commit failed: {e}"));
                    break;
                }
            }
            let expected = ids(0..10);
            report(
                &mut failures,
                &what,
                wrong(&engine, &storage, &expected, &codebook).await,
            );
            let segments = segment_codebooks(&storage).len();
            if segments != 1 {
                failures.push(format!("{what}: {segments} segments, expected one merge"));
            }
            drop(engine);

            if file == "trained again" {
                let engine = open(&storage, Some(after.clone())).await;
                put(&engine, 20..21).await;
                engine.commit().await.unwrap();
                let last = segment_codebooks(&storage).pop().flatten();
                if last.as_deref() != Some(again.as_slice()) {
                    failures.push(format!("{what}: a reopen did not take the new file"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A codebook that cannot be read fails a change that needs it before
/// anything changes: the documents, the schema and the field's files stay,
/// also when the change would discard the field's data, and the engine goes
/// on with the codebook it has. Once the file is trained again, the change
/// succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn a_corrupt_codebook_fails_a_change_before_anything_changes() {
    let mut failures = Vec::new();
    let cases = [
        ("reindex", hnsw(|_| {}), hnsw(|o| o.m = 24)),
        ("from flat", flat(), hnsw(|_| {})),
        (
            "destructive",
            hnsw(|_| {}),
            hnsw(|o| o.distance = DistanceMetric::Cosine),
        ),
    ];
    for (what, before, after) in cases {
        let dir = TempDir::new().unwrap();
        let storage = file_storage(&dir);
        let codebook = train(&storage, "embedding.pqcb", 1);
        let engine = open(&storage, Some(before.clone())).await;
        put(&engine, 0..20).await;
        engine.commit().await.unwrap();
        damage(&storage);
        let files = field_files(&dir);
        match engine
            .update_field("embedding", after.clone(), REINDEX)
            .await
        {
            Err(e) if e.to_string().contains("corrupted") => {}
            other => failures.push(format!("{what}: {:?}", other.map(|o| o.classification))),
        }
        if format!("{:?}", engine.schema().fields["embedding"]) != format!("{before:?}") {
            failures.push(format!("{what}: the schema changed"));
        }
        if engine.schema().pending_reindex.contains("embedding") {
            failures.push(format!("{what}: pending_reindex names the field"));
        }
        if field_files(&dir) != files {
            failures.push(format!("{what}: the field's files changed"));
        }
        if found(&engine).await != ids(0..20) {
            failures.push(format!("{what}: documents are missing"));
        }
        put(&engine, 20..21).await;
        if let Err(e) = engine.commit().await {
            failures.push(format!("{what}: a later commit failed: {e}"));
        }
        if found(&engine).await != ids(0..21) {
            failures.push(format!("{what}: a later document is missing"));
        }

        let again = train(&storage, "embedding.pqcb", 1);
        assert_eq!(again, codebook);
        if let Err(e) = engine.update_field("embedding", after, REINDEX).await {
            failures.push(format!("{what}: tried again, it failed: {e}"));
        }
    }

    let dir = TempDir::new().unwrap();
    let storage = file_storage(&dir);
    let codebook = train(&storage, "embedding.pqcb", 1);
    damage(&storage);
    let engine = open(&storage, None).await;
    match engine.add_field("embedding", hnsw(|_| {})).await {
        Err(e) if e.to_string().contains("corrupted") => {}
        other => failures.push(format!("add_field: {:?}", other.map(|_| ()))),
    }
    if engine.schema().fields.contains_key("embedding") {
        failures.push("add_field: the schema has the field".to_string());
    }
    train(&storage, "embedding.pqcb", 1);
    engine.add_field("embedding", hnsw(|_| {})).await.unwrap();
    put(&engine, 0..5).await;
    engine.commit().await.unwrap();
    report(
        &mut failures,
        "add_field, tried again",
        wrong(&engine, &storage, &ids(0..5), &codebook).await,
    );
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A codebook that is missing, or of another shape, is treated as when an
/// engine is opened: it fails where a segment is written. A field is added
/// and its commit fails, and a rebuild fails before it publishes, leaving
/// the field as it was.
#[tokio::test(flavor = "multi_thread")]
async fn a_missing_or_mismatched_codebook_fails_where_a_segment_is_written() {
    let mut failures = Vec::new();
    for (what, shape, message) in [
        ("missing", None, "no codebook has been trained"),
        ("another m", Some((DIM, 8)), "subvector_count 8"),
        ("another dimension", Some((16, M)), "dimension 16"),
    ] {
        let dir = TempDir::new().unwrap();
        let storage = file_storage(&dir);
        if let Some((dimension, subvectors)) = shape {
            train_shape(&storage, "embedding.pqcb", 1, dimension, subvectors);
        }
        let engine = open(&storage, Some(flat())).await;
        put(&engine, 0..20).await;
        engine.commit().await.unwrap();
        match engine
            .update_field("embedding", hnsw(|_| {}), REINDEX)
            .await
        {
            Err(e) if e.to_string().contains(message) => {}
            other => failures.push(format!("{what}: {:?}", other.map(|o| o.classification))),
        }
        if format!("{:?}", engine.schema().fields["embedding"]) != format!("{:?}", flat()) {
            failures.push(format!("{what}: the schema changed"));
        }
        put(&engine, 20..21).await;
        engine.commit().await.unwrap();
        if found(&engine).await != ids(0..21) {
            failures.push(format!("{what}: documents are missing"));
        }
        drop(engine);

        let engine = open(&storage, None).await;
        engine.add_field("embedding", hnsw(|_| {})).await.unwrap();
        put(&engine, 30..31).await;
        match engine.commit().await {
            Err(e) if e.to_string().contains(message) => {}
            other => failures.push(format!("{what}: the commit returned {other:?}")),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The path of field `embedding`'s codebook, and a field whose codebook, under
/// the default name, has the same path in the directory of `embedding`.
const NESTED: [(&str, &str); 2] = [
    ("b.pqcb", "embedding/b"),
    ("embedding.pqcb", "embedding/embedding"),
];

/// Every way the index of a field is opened.
const WAYS: [&str; 6] = [
    "opened",
    "added",
    "added again",
    "rebuilt",
    "rebuilt from flat",
    "recreated",
];

/// What lies in the directory of `embedding`, under the path of its codebook.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Nested {
    /// The codebook of a field of the schema.
    OfAField,
    /// A codebook that no field names.
    OfNoField,
    /// A file that is not a codebook.
    Unreadable,
    /// A codebook with another dimension and number of subvectors.
    OfAnotherShape(usize, usize),
}

/// Put the file there, and return its centroids.
fn put_nested(storage: &Arc<dyn Storage>, field: &str, nested: Nested) -> Option<Vec<f32>> {
    let name = format!("{field}.pqcb");
    match nested {
        Nested::OfAField | Nested::OfNoField => Some(train(storage, &name, 2)),
        Nested::OfAnotherShape(dimension, subvectors) => {
            Some(train_shape(storage, &name, 2, dimension, subvectors))
        }
        Nested::Unreadable => {
            let mut output = vector_root(storage).create_output(&name).unwrap();
            output.write_all(b"not a codebook").unwrap();
            output.close().unwrap();
            None
        }
    }
}

/// Open an engine whose schema has `option`, if any, as field `embedding`, and
/// `nested`, if any, as an HNSW field with the default codebook name.
async fn open_nested(
    storage: &Arc<dyn Storage>,
    option: Option<FieldOption>,
    nested: Option<&str>,
) -> laurus::Result<Engine> {
    let mut schema = schema_with(option);
    if let Some(field) = nested {
        schema.fields.insert(
            field.to_string(),
            hnsw(|o| o.pq_codebook_path = Some(format!("{field}.pqcb"))),
        );
    }
    Engine::new(storage.clone(), schema).await
}

/// Open the index of `embedding` in one of [`WAYS`], with its own codebook at
/// the root and `nested` in its directory, and return why it does not use its
/// own, if it does not.
///
/// Unless the way is an open, the engine is opened before the file is there,
/// so that only the operation can come across it.
async fn check_nested(path: &str, field: &str, nested: Nested, way: &str) -> Result<(), String> {
    let dir = TempDir::new().unwrap();
    let storage = file_storage(&dir);
    let own = train(&storage, path, 1);
    let in_schema = (nested == Nested::OfAField).then_some(field);
    let option = hnsw(|o| o.pq_codebook_path = Some(path.to_string()));
    let after = hnsw(|o| {
        o.pq_codebook_path = Some(path.to_string());
        match way {
            "rebuilt" => o.m = 24,
            "recreated" => o.distance = DistanceMetric::Cosine,
            _ => {}
        }
    });
    let before = match way {
        "added" => None,
        "rebuilt from flat" => Some(flat()),
        _ => Some(option),
    };
    let mut other = None;
    if way == "opened" {
        other = put_nested(&storage, field, nested);
    }
    let engine = open_nested(&storage, before, in_schema)
        .await
        .map_err(|e| format!("the open failed: {e}"))?;
    let mut expected = BTreeSet::new();
    if way != "opened" {
        if way != "added" {
            put(&engine, 0..5).await;
            engine.commit().await.unwrap();
        }
        other = put_nested(&storage, field, nested);
        let changed = match way {
            "added again" => {
                engine.delete_field("embedding").await.unwrap();
                engine.add_field("embedding", after.clone()).await.err()
            }
            "added" => engine.add_field("embedding", after.clone()).await.err(),
            _ => engine
                .update_field("embedding", after.clone(), REINDEX)
                .await
                .err(),
        };
        if let Some(e) = changed {
            return Err(format!("the change failed: {e}"));
        }
        if way.starts_with("rebuilt") {
            expected = ids(0..5);
            if let Some(why) = wrong(&engine, &storage, &expected, &own).await {
                return Err(format!("the change: {why}"));
            }
        }
    }
    for (from, to) in [(10, 30), (30, 31)] {
        put(&engine, from..to).await;
        engine
            .commit()
            .await
            .map_err(|e| format!("the commit failed: {e}"))?;
    }
    expected.extend(ids(10..31));
    if let Some(why) = wrong(&engine, &storage, &expected, &own).await {
        return Err(format!("the commits: {why}"));
    }
    drop(engine);

    let engine = open_nested(&storage, Some(after), in_schema)
        .await
        .map_err(|e| format!("the reopen failed: {e}"))?;
    let mut doc = Document::builder().add_vector("embedding", make_vectors(1, 7)[0].data.to_vec());
    if let Some(field) = in_schema {
        doc = doc.add_vector(field, make_vectors(1, 8)[0].data.to_vec());
    }
    engine.put_document("d40", doc.build()).await.unwrap();
    engine
        .commit()
        .await
        .map_err(|e| format!("the commit after a reopen failed: {e}"))?;
    expected.extend(ids(40..41));
    if let Some(why) = wrong(&engine, &storage, &expected, &own).await {
        return Err(format!("reopened: {why}"));
    }
    if let Some(field) = in_schema
        && segment_codebooks_of(&storage, field) != [other]
    {
        return Err("the nested field does not use its own codebook".to_string());
    }
    Ok(())
}

/// A field takes its codebook from the root of the vector storage, however its
/// index is opened. A codebook of the same path in its directory, where a field
/// nested under it keeps its own, is not taken, and the nested field uses it.
#[tokio::test(flavor = "multi_thread")]
async fn a_codebook_in_the_directory_of_a_field_is_not_taken() {
    let mut failures = Vec::new();
    for (path, field) in NESTED {
        for nested in [Nested::OfAField, Nested::OfNoField] {
            for way in WAYS {
                let result = check_nested(path, field, nested, way).await;
                report(
                    &mut failures,
                    &format!("{path}, {nested:?}, {way}"),
                    result.err(),
                );
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A file there that cannot be read, or is of another geometry, does not fail
/// the operation either, and so does not fail a destructive change after the
/// data is gone.
#[tokio::test(flavor = "multi_thread")]
async fn a_file_in_the_directory_of_a_field_that_does_not_fit_does_not_matter() {
    let mut failures = Vec::new();
    let (path, field) = NESTED[0];
    for nested in [
        Nested::Unreadable,
        Nested::OfAnotherShape(DIM, 8),
        Nested::OfAnotherShape(16, M),
    ] {
        for way in WAYS {
            let result = check_nested(path, field, nested, way).await;
            report(&mut failures, &format!("{nested:?}, {way}"), result.err());
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A codebook in the directory of a field does not stand in for a missing one:
/// the field is opened and added, and its commit fails with the command to
/// train one, and a rebuild fails before it publishes, leaving the field as it
/// was. Once the field's own is trained, the commit succeeds with it.
#[tokio::test(flavor = "multi_thread")]
async fn a_codebook_in_the_directory_of_a_field_does_not_stand_in_for_a_missing_one() {
    let mut failures = Vec::new();
    let missing = |e: &laurus::LaurusError| {
        let message = e.to_string();
        message.contains("no codebook has been trained")
            && message.contains("laurus train pq-codebook")
    };
    for (path, field) in NESTED {
        let option = hnsw(|o| o.pq_codebook_path = Some(path.to_string()));
        for way in ["opened", "added"] {
            let what = format!("{path}, {way}");
            let dir = TempDir::new().unwrap();
            let storage = file_storage(&dir);
            put_nested(&storage, field, Nested::OfNoField);
            let before = (way == "opened").then(|| option.clone());
            let engine = open(&storage, before).await;
            if way == "added" {
                engine.add_field("embedding", option.clone()).await.unwrap();
            }
            put(&engine, 0..20).await;
            match engine.commit().await {
                Err(e) if missing(&e) => {}
                other => failures.push(format!("{what}: the commit returned {other:?}")),
            }
            drop(engine);

            let own = train(&storage, path, 1);
            let engine = open(&storage, Some(option.clone())).await;
            if let Err(e) = engine.commit().await {
                failures.push(format!("{what}: trained, the commit failed: {e}"));
            }
            report(
                &mut failures,
                &format!("{what}, trained"),
                wrong(&engine, &storage, &ids(0..20), &own).await,
            );
        }

        let what = format!("{path}, rebuilt from flat");
        let dir = TempDir::new().unwrap();
        let storage = file_storage(&dir);
        let engine = open(&storage, Some(flat())).await;
        put(&engine, 0..20).await;
        engine.commit().await.unwrap();
        put_nested(&storage, field, Nested::OfNoField);
        match engine.update_field("embedding", option, REINDEX).await {
            Err(e) if missing(&e) => {}
            other => failures.push(format!("{what}: {:?}", other.map(|o| o.classification))),
        }
        if format!("{:?}", engine.schema().fields["embedding"]) != format!("{:?}", flat()) {
            failures.push(format!("{what}: the schema changed"));
        }
        put(&engine, 20..21).await;
        engine.commit().await.unwrap();
        if found(&engine).await != ids(0..21) {
            failures.push(format!("{what}: documents are missing"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
