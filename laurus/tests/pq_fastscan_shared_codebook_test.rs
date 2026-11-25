//! End-to-end engine tests for the FastScan shared PQ codebook (Issue
//! #920 sub-item 2).
//!
//! FastScan mirror of `engine_pq_codebook_test.rs`: with a
//! `ProductQuantizationFastScan` quantizer and a `pq_codebook_path` on
//! the schema, `Engine::train_pq_codebook` must train a k=16 codebook,
//! a commit before training must hard-error (before this fix the path
//! was silently ignored and every segment retrained k-means), and a
//! reopened engine must encode against the shared codebook.

#![cfg(feature = "pq-fastscan")]

use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::HnswOption;
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::index::pq_codebook::read_pq_codebook;
use laurus::{DataValue, Document, Engine, FieldOption, Schema, UpdateFieldOptions};
use tempfile::TempDir;

const DIM: usize = 32;
const M: usize = 4;

/// Deterministic pseudo-random vectors (same LCG as
/// `engine_pq_codebook_test.rs`); distinct seeds give distinct corpora.
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

/// Schema with one HNSW + FastScan field pointing at a shared codebook.
fn fastscan_schema(pq_codebook_path: Option<&str>) -> Schema {
    Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Hnsw(HnswOption {
                dimension: DIM,
                distance: DistanceMetric::Euclidean,
                m: 8,
                ef_construction: 32,
                quantizer: QuantizationMethod::ProductQuantizationFastScan { subvector_count: M },
                pq_codebook_path: pq_codebook_path.map(str::to_string),
                ..HnswOption::default()
            }),
        )
        .build()
}

/// The full schema-level loop for FastScan: (1) a commit before training
/// hard-errors naming the training step, (2) `train_pq_codebook` trains
/// a k=16 codebook into the vector namespace, (3) a reopened engine
/// commits a tiny batch successfully — far below the FastScan min-train
/// threshold, so a silent fallback (the pre-#920 behavior of ignoring
/// the path, or the demotion guard firing) would produce a different
/// outcome.
#[tokio::test(flavor = "multi_thread")]
async fn fastscan_train_then_reopen_commits_with_the_shared_codebook() -> laurus::Result<()> {
    let dir = TempDir::new().unwrap();
    let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path())))?;

    // (1) Before training: the configured-but-missing codebook must fail
    // the commit loudly (pre-#920 this was silently ignored for FastScan).
    {
        let engine = Engine::new(storage.clone(), fastscan_schema(Some("embedding.pqcb"))).await?;
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

    // (2) Train — the FastScan quantizer variant must yield k=16.
    let fresh_dir = TempDir::new().unwrap();
    let fresh_storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(
        fresh_dir.path(),
    )))?;
    let trainer = Engine::new(
        fresh_storage.clone(),
        fastscan_schema(Some("embedding.pqcb")),
    )
    .await?;
    let info = trainer.train_pq_codebook("embedding", &make_vectors(400, 0x0F0F), None)?;
    assert_eq!(info.path, "embedding.pqcb");
    assert_eq!(info.subvector_count, M);
    assert_eq!(
        info.centroids, 16,
        "a FastScan field must train a k=16 codebook, not the standard 256"
    );
    assert_eq!(info.sub_dimension, DIM / M);
    assert!(fresh_storage.file_exists("vector/embedding.pqcb"));
    drop(trainer);

    // The persisted file itself carries k=16.
    let vector_ns =
        laurus::storage::prefixed::PrefixedStorage::new("vector", fresh_storage.clone());
    let loaded = read_pq_codebook(&vector_ns, "embedding.pqcb")?;
    assert_eq!(loaded.params.k, 16);

    // (3) A reopened engine picks the codebook up at open and commits a
    // tiny batch — 5 docs, far below PQ_FASTSCAN_MIN_TRAIN_VECTORS (16),
    // so this only succeeds because the shared codebook exempts the
    // segment from both the demotion guard and per-segment training.
    let engine = Engine::new(
        fresh_storage.clone(),
        fastscan_schema(Some("embedding.pqcb")),
    )
    .await?;
    for (i, v) in make_vectors(5, 0xBEEF).into_iter().enumerate() {
        let doc = Document::builder()
            .add_field("embedding", DataValue::Vector(v.data.to_vec()))
            .build();
        engine.put_document(&format!("d{i}"), doc).await?;
    }
    engine.commit().await?;

    Ok(())
}

/// A k=256 (standard PQ) codebook configured on a FastScan field must
/// fail the commit loudly with the k mismatch named — not encode
/// garbage or silently retrain.
#[tokio::test(flavor = "multi_thread")]
async fn k256_codebook_on_fastscan_field_fails_the_commit() -> laurus::Result<()> {
    let dir = TempDir::new().unwrap();
    let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path())))?;

    // Train a k=256 codebook by declaring the field as standard PQ first.
    let pq_schema = Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Hnsw(HnswOption {
                dimension: DIM,
                distance: DistanceMetric::Euclidean,
                m: 8,
                ef_construction: 32,
                quantizer: QuantizationMethod::ProductQuantization { subvector_count: M },
                pq_codebook_path: Some("embedding.pqcb".to_string()),
                ..HnswOption::default()
            }),
        )
        .build();
    let trainer = Engine::new(storage.clone(), pq_schema).await?;
    let info = trainer.train_pq_codebook("embedding", &make_vectors(400, 0x0F0F), None)?;
    assert_eq!(info.centroids, 256);
    drop(trainer);

    // Reopen with the SAME codebook file but a FastScan quantizer.
    let engine = Engine::new(storage.clone(), fastscan_schema(Some("embedding.pqcb"))).await?;
    for (i, v) in make_vectors(20, 0xBEEF).into_iter().enumerate() {
        let doc = Document::builder()
            .add_field("embedding", DataValue::Vector(v.data.to_vec()))
            .build();
        engine.put_document(&format!("d{i}"), doc).await?;
    }
    let err = engine.commit().await.unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("k = 256") && message.contains("k = 16"),
        "the k mismatch must be named: {message}"
    );

    Ok(())
}

/// Issue #1735: an open engine keeps the shared codebook of a FastScan field
/// through a change of its search-time parameters, and loads it when the
/// field is rebuilt or added. Five documents are far below the count from
/// which a segment trains a codebook of its own.
#[tokio::test(flavor = "multi_thread")]
async fn an_open_engine_keeps_the_shared_codebook() -> laurus::Result<()> {
    let dir = TempDir::new().unwrap();
    let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path())))?;
    let schema = fastscan_schema(Some("embedding.pqcb"));
    let trainer = Engine::new(storage.clone(), schema.clone()).await?;
    trainer.train_pq_codebook("embedding", &make_vectors(400, 0x0F0F), None)?;
    drop(trainer);

    let engine = Engine::new(storage.clone(), schema.clone()).await?;
    let FieldOption::Hnsw(option) = schema.fields["embedding"].clone() else {
        panic!("an HNSW field");
    };
    let searched = FieldOption::Hnsw(HnswOption {
        default_ef_search: Some(99),
        ..option.clone()
    });
    let rebuilt = FieldOption::Hnsw(HnswOption {
        default_ef_search: Some(99),
        m: 12,
        ..option.clone()
    });
    let reindex = UpdateFieldOptions {
        reindex: true,
        dry_run: false,
    };
    for (round, change) in [None, Some(searched), Some(rebuilt)]
        .into_iter()
        .enumerate()
    {
        if let Some(change) = change {
            engine.update_field("embedding", change, reindex).await?;
        }
        for (i, v) in make_vectors(5, 0xBEEF + round as u64)
            .into_iter()
            .enumerate()
        {
            let doc = Document::builder()
                .add_field("embedding", DataValue::Vector(v.data.to_vec()))
                .build();
            engine.put_document(&format!("d{round}-{i}"), doc).await?;
        }
        engine.commit().await?;
    }
    engine.delete_field("embedding").await?;
    engine
        .add_field("embedding", FieldOption::Hnsw(option))
        .await?;
    let doc = Document::builder()
        .add_field(
            "embedding",
            DataValue::Vector(make_vectors(1, 7)[0].data.to_vec()),
        )
        .build();
    engine.put_document("added", doc).await?;
    engine.commit().await?;
    assert_eq!(engine.stats()?.vector_fields["embedding"].vector_count, 1);
    Ok(())
}
