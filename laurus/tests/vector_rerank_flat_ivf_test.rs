//! End-to-end engine wiring tests for Stage-2 rerank on Flat and IVF
//! (Issue #650 PR-2 / #932), mirroring `vector_rerank_engine_test.rs`.
//!
//! Pins the full loop per index type: `rerank_storage: Some(F32)` on the
//! schema field makes commit emit the `.f32` sidecar next to each sealed
//! segment, the reader loads it, and a search with `rerank_factor`
//! rescores against the exact f32 vectors (measurably different scores
//! from the int8 path on non-grid data). A field without the sidecar
//! keeps the silent Stage-1 fallback — including that `rerank_factor` no
//! longer returns `NotImplemented` (the pre-#932 behavior).

use tempfile::TempDir;

use laurus::DistanceMetric;
use laurus::Engine;
use laurus::SearchRequestBuilder;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::vector::{FlatOption, IvfOption, Vector};
use laurus::{DataValue, Document};
use laurus::{FieldOption, QueryVector, Schema, VectorSearchQuery};

/// Non-grid components (not exactly representable on the int8 affine
/// grid) so the exact-f32 rerank score measurably differs from the int8
/// score; a query between the docs keeps similarities off any clamp.
const DOCS: [(&str, [f32; 4]); 4] = [
    ("doc1", [0.92, 0.31, 0.17, 0.05]),
    ("doc2", [0.13, 0.83, 0.41, 0.27]),
    ("doc3", [0.05, 0.19, 0.77, 0.61]),
    ("doc4", [0.33, 0.47, 0.29, 0.71]),
];
const QUERY: [f32; 4] = [0.87, 0.36, 0.21, 0.09];

fn vector_request(rerank_factor: Option<usize>) -> laurus::SearchRequest {
    vector_request_for_fields(Some(vec!["embedding".to_string()]), rerank_factor)
}

/// Like [`vector_request`], but lets the caller pick `fields` -- `None`
/// builds a field-less query (Issue #1388).
fn vector_request_for_fields(
    fields: Option<Vec<String>>,
    rerank_factor: Option<usize>,
) -> laurus::SearchRequest {
    let mut builder = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(QUERY.to_vec()),
            weight: 1.0,
            fields,
        }]))
        .limit(1);
    if let Some(factor) = rerank_factor {
        builder = builder.vector_rerank_factor(factor);
    }
    builder.build()
}

async fn ingest(engine: &Engine) -> laurus::Result<()> {
    for (id, vec) in &DOCS {
        let doc = Document::builder()
            .add_field("embedding", DataValue::Vector(vec.to_vec()))
            .build();
        engine.put_document(id, doc).await?;
    }
    engine.commit().await
}

/// Shared body: build the engine over `field_option`, ingest + commit,
/// assert the sidecar exists at `sidecar_name`, and assert rerank both
/// works (doc1 wins) and *changes* the score vs the int8 path.
async fn assert_rerank_end_to_end(
    field_option: FieldOption,
    sidecar_name: &str,
) -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;
    let schema = Schema::builder()
        .add_field("embedding", field_option)
        .build();
    let engine = Engine::new(storage.clone(), schema).await?;
    ingest(&engine).await?;

    assert!(
        storage.file_exists(sidecar_name),
        "commit with rerank_storage: Some(F32) must emit {sidecar_name}"
    );

    let with_rerank = engine.search(vector_request(Some(4))).await?;
    assert_eq!(with_rerank.len(), 1, "expected exactly 1 hit");
    assert_eq!(
        with_rerank[0].id, "doc1",
        "doc1 is the closest match under exact f32 distances"
    );

    let without_rerank = engine.search(vector_request(None)).await?;
    assert_eq!(without_rerank[0].id, "doc1");
    assert_ne!(
        with_rerank[0].score.to_bits(),
        without_rerank[0].score.to_bits(),
        "rerank_factor must change the score via the f32 sidecar pool; \
         identical scores mean the silent Stage-1 fallback was taken"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn flat_rerank_succeeds_on_stage2_field() -> laurus::Result<()> {
    assert_rerank_end_to_end(
        FieldOption::Flat(FlatOption {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            rerank_storage: Some(RerankStorageKind::F32),
            ..FlatOption::default()
        }),
        // Segment-per-commit is the Flat default since #907. Nested one
        // level under the "embedding" field's own sub-namespace (Issue
        // #948: `MultiFieldVectorIndex`).
        "vector/embedding/segment_000000.flat.f32",
    )
    .await
}

#[tokio::test(flavor = "multi_thread")]
async fn ivf_rerank_succeeds_on_stage2_field() -> laurus::Result<()> {
    assert_rerank_end_to_end(
        FieldOption::Ivf(IvfOption {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            n_clusters: 2,
            n_probe: 2,
            rerank_storage: Some(RerankStorageKind::F32),
            ..IvfOption::default()
        }),
        // Nested under the "embedding" field's own sub-namespace (Issue
        // #948: `MultiFieldVectorIndex`).
        "vector/embedding/segment_000000.ivf.f32",
    )
    .await
}

/// Issue #1388: a field-less query (no `fields` selector) over a Flat or
/// IVF field with `rerank_storage` must apply the Stage-2 rerank exactly
/// like a field-routed query on the same field — same top hit, same exact
/// f32 score — instead of silently keeping the int8 Stage-1 score. Mirrors
/// `vector_multi_field_test.rs`'s `fieldless_query_takes_hnsw_graph_path`
/// (the equivalent HNSW gap, #1371 / #1381).
async fn assert_fieldless_rerank_matches_routed(field_option: FieldOption) -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;
    let schema = Schema::builder()
        .add_field("embedding", field_option)
        .build();
    let engine = Engine::new(storage, schema).await?;
    ingest(&engine).await?;

    let routed = engine
        .search(vector_request_for_fields(
            Some(vec!["embedding".to_string()]),
            Some(4),
        ))
        .await?;
    let fieldless = engine
        .search(vector_request_for_fields(None, Some(4)))
        .await?;

    assert_eq!(
        routed[0].id, "doc1",
        "field-routed query must find doc1 first"
    );
    assert_eq!(
        fieldless[0].id, routed[0].id,
        "a field-less query must pick the same top hit as the field-routed one"
    );
    assert_eq!(
        fieldless[0].score.to_bits(),
        routed[0].score.to_bits(),
        "a field-less query must apply the same exact-f32 rerank score as the field-routed one"
    );

    let fieldless_without_rerank = engine.search(vector_request_for_fields(None, None)).await?;
    assert_ne!(
        fieldless[0].score.to_bits(),
        fieldless_without_rerank[0].score.to_bits(),
        "rerank_factor must change the field-less score via the f32 sidecar; \
         identical scores mean the silent Stage-1 fallback was taken"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn flat_rerank_succeeds_on_fieldless_stage2_field() -> laurus::Result<()> {
    assert_fieldless_rerank_matches_routed(FieldOption::Flat(FlatOption {
        dimension: 4,
        distance: DistanceMetric::Cosine,
        rerank_storage: Some(RerankStorageKind::F32),
        ..FlatOption::default()
    }))
    .await
}

#[tokio::test(flavor = "multi_thread")]
async fn ivf_rerank_succeeds_on_fieldless_stage2_field() -> laurus::Result<()> {
    assert_fieldless_rerank_matches_routed(FieldOption::Ivf(IvfOption {
        dimension: 4,
        distance: DistanceMetric::Cosine,
        n_clusters: 2,
        n_probe: 2,
        rerank_storage: Some(RerankStorageKind::F32),
        ..IvfOption::default()
    }))
    .await
}

/// A Stage-1 field (no `rerank_storage`) with `rerank_factor` set must
/// silently fall back — succeeding with int8-identical scores instead of
/// the pre-#932 `NotImplemented` error.
#[tokio::test(flavor = "multi_thread")]
async fn flat_and_ivf_rerank_factor_falls_back_silently_on_stage1_field() -> laurus::Result<()> {
    for field_option in [
        FieldOption::Flat(FlatOption {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            ..FlatOption::default()
        }),
        FieldOption::Ivf(IvfOption {
            dimension: 4,
            distance: DistanceMetric::Cosine,
            n_clusters: 2,
            n_probe: 2,
            ..IvfOption::default()
        }),
    ] {
        let temp_dir = TempDir::new().unwrap();
        let storage =
            StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;
        let schema = Schema::builder()
            .add_field("embedding", field_option)
            .build();
        let engine = Engine::new(storage, schema).await?;
        ingest(&engine).await?;

        let with_factor = engine.search(vector_request(Some(4))).await?;
        let without = engine.search(vector_request(None)).await?;
        assert_eq!(with_factor[0].id, "doc1");
        assert_eq!(
            with_factor[0].score.to_bits(),
            without[0].score.to_bits(),
            "a sidecar-less field must silently keep Stage-1 scores"
        );
    }
    Ok(())
}

/// Multi-commit (multi-segment) Flat index under rerank: the per-segment
/// exact-f32 scores carry the `score_basis` stamp, so the fan-out keeps
/// them (#927) and the cross-segment order matches the exact computation.
#[tokio::test(flavor = "multi_thread")]
async fn segmented_flat_rerank_orders_across_segments() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;
    let schema = Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Flat(FlatOption {
                dimension: 4,
                distance: DistanceMetric::Cosine,
                rerank_storage: Some(RerankStorageKind::F32),
                ..FlatOption::default()
            }),
        )
        .build();
    let engine = Engine::new(storage.clone(), schema).await?;

    // Two commits = two sealed segments, best match in the second.
    for (id, vec) in &DOCS[..2] {
        let doc = Document::builder()
            .add_field("embedding", DataValue::Vector(vec.to_vec()))
            .build();
        engine.put_document(id, doc).await?;
    }
    engine.commit().await?;
    for (id, vec) in &DOCS[2..] {
        let doc = Document::builder()
            .add_field("embedding", DataValue::Vector(vec.to_vec()))
            .build();
        engine.put_document(id, doc).await?;
    }
    engine.commit().await?;
    assert!(storage.file_exists("vector/embedding/segment_000001.flat.f32"));

    let hits = engine.search(vector_request(Some(4))).await?;
    assert_eq!(hits[0].id, "doc1", "cross-segment rerank must find doc1");
    Ok(())
}
