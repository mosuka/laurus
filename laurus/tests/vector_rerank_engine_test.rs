//! End-to-end engine wiring tests for Issue #481 Stage 2 / Issue #790.
//!
//! Verifies that `rerank_storage: Some(_)` on a schema field actually
//! reaches the HNSW writer through the `Engine` → `VectorStore` config
//! conversion (Issue #790): the commit must emit the LRS1 sidecar
//! (`vector_index.hnsw.f32`), and `SearchRequestBuilder::vector_rerank_factor`
//! must flow to the Stage-2 rerank path — not the silent Stage-1
//! fallback, which by construction returns bit-identical scores with
//! and without `rerank_factor`. Stage-1 fields (no `rerank_storage`)
//! must keep their behavior: no sidecar, silent int8 ranking.

use tempfile::TempDir;

use laurus::DistanceMetric;
use laurus::Engine;
use laurus::SearchRequestBuilder;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::prefixed::PrefixedStorage;
use laurus::storage::{LoadingMode, Storage, StorageConfig, StorageFactory};
use laurus::vector::HnswOption;
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric as VectorDistanceMetric;
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::vector::index::hnsw::reader::HnswIndexReader;
use laurus::{DataValue, Document};
use laurus::{FieldOption, QueryVector, Schema, VectorSearchQuery};
use std::sync::Arc;

/// Sidecar name as seen by the engine's outer storage: the vector
/// store works behind `PrefixedStorage("vector", ..)`, each field then
/// gets its own further `PrefixedStorage("embedding", ..)` sub-namespace
/// (Issue #948: `MultiFieldVectorIndex`), and the first sealed segment of
/// the (default, #882) segmented layout is `segment_000000.hnsw`.
const SIDECAR_NAME: &str = "vector/embedding/segment_000000.hnsw.f32";

/// Build a search request for `query`, optionally asking for Stage-2
/// rerank with the given factor.
fn vector_request(query: &[f32], rerank_factor: Option<usize>) -> laurus::SearchRequest {
    let mut builder = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(query.to_vec()),
            weight: 1.0,
            fields: Some(vec!["embedding".to_string()]),
        }]))
        .limit(1);
    if let Some(factor) = rerank_factor {
        builder = builder.vector_rerank_factor(factor);
    }
    builder.build()
}

#[tokio::test(flavor = "multi_thread")]
async fn engine_search_with_rerank_factor_succeeds_on_stage2_field() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    // mmap on explicitly, not through the platform default that
    // `LAURUS_NO_MMAP=1` turns off, so the segment reopens under `Lazy`
    // below: the reader must load the sidecar in that mode too (#1482).
    let mut config = FileStorageConfig::new(temp_dir.path());
    config.use_mmap = true;
    let storage = StorageFactory::create(StorageConfig::File(config))?;

    let hnsw_opt = HnswOption {
        dimension: 4,
        distance: DistanceMetric::Cosine,
        m: 4,
        ef_construction: 16,
        rerank_storage: Some(RerankStorageKind::F32),
        ..HnswOption::default()
    };

    let schema = Schema::builder()
        .add_field("embedding", FieldOption::Hnsw(hnsw_opt))
        .build();

    let engine = Engine::new(storage.clone(), schema).await?;

    // Non-grid components (not exactly representable on the segment's
    // int8 affine grid) so the exact-f32 rerank score measurably
    // differs from the int8 score, and a query *between* the docs so
    // similarities stay away from the 1.0 clamp.
    let vectors: [(&str, [f32; 4]); 4] = [
        ("doc1", [0.92, 0.31, 0.17, 0.05]),
        ("doc2", [0.13, 0.83, 0.41, 0.27]),
        ("doc3", [0.05, 0.19, 0.77, 0.61]),
        ("doc4", [0.33, 0.47, 0.29, 0.71]),
    ];
    for (id, vec) in &vectors {
        let doc = Document::builder()
            .add_field("embedding", DataValue::Vector(vec.to_vec()))
            .build();
        engine.put_document(id, doc).await?;
    }
    engine.commit().await?;

    // Issue #790 acceptance: the commit must have emitted the LRS1
    // sidecar through the Engine → VectorStore config conversion.
    assert!(
        storage.file_exists(SIDECAR_NAME),
        "commit with rerank_storage: Some(F32) must emit {SIDECAR_NAME} \
         (Issue #790: option was dropped in extract_index_type_config)"
    );

    // Stronger, deterministic guard: reopen the committed segment the
    // way the vector store does (behind `PrefixedStorage("vector", ..)`,
    // first sealed segment `segment_000000`, #882) and assert the sidecar
    // actually loads into the rerank pool. `PrefixedStorage` reports the
    // mmap-backed `FileStorage`'s `Lazy` mode (#554), so this also pins
    // that the sidecar loads under `Lazy` — it proves "the f32 pool
    // exists and is populated", independent of the score comparison
    // below.
    let vector_storage: Arc<dyn Storage> =
        Arc::new(PrefixedStorage::new("vector", storage.clone()));
    let field_storage: Arc<dyn Storage> =
        Arc::new(PrefixedStorage::new("embedding", vector_storage));
    assert_eq!(
        field_storage.loading_mode(),
        LoadingMode::Lazy,
        "the reopened segment must see the mmap-backed storage's Lazy mode (#554)"
    );
    let reader = HnswIndexReader::load(
        field_storage,
        "segment_000000",
        VectorDistanceMetric::Cosine,
    )?;
    let pool = reader
        .rerank_storage()
        .expect("the committed Stage-2 segment must load its rerank pool");
    let positions = pool
        .field_position_index("embedding")
        .expect("the rerank pool must index the 'embedding' field (Issue #790)");
    assert_eq!(
        positions.len(),
        vectors.len(),
        "the rerank pool must hold one f32 vector per committed document"
    );

    let query = [0.87, 0.36, 0.21, 0.09];

    // rerank_factor = 4 rescans ALL 4 docs against the exact f32 pool
    // (rerank_count = limit × factor = 4), so the doc1-wins assertion
    // below cannot depend on which candidates survive the int8 Stage-1
    // cut under an unlucky (parallel-build) graph topology — the flake
    // #841 hit with factor 3, where doc1 once fell outside the rescored
    // top-3. The Stage-2 wiring this test guards (sidecar emitted, pool
    // loaded, score changed by rerank) is unaffected by the factor.
    let with_rerank = engine.search(vector_request(&query, Some(4))).await?;
    assert_eq!(with_rerank.len(), 1, "expected exactly 1 hit");
    assert_eq!(
        with_rerank[0].id, "doc1",
        "doc1 should be the closest match to the rerank-augmented query"
    );

    let without_rerank = engine.search(vector_request(&query, None)).await?;
    assert_eq!(without_rerank.len(), 1);
    assert_eq!(without_rerank[0].id, "doc1");

    // The silent Stage-1 fallback returns bit-identical scores with
    // and without rerank_factor (the searcher's `_` match arm reuses
    // the same int8 candidates). A real Stage-2 rerank rescores the
    // top candidates against the original f32 vectors, whose non-grid
    // components differ from their int8 dequantization — so a score
    // difference proves the f32 pool was used.
    assert_ne!(
        with_rerank[0].score, without_rerank[0].score,
        "rerank_factor must change the score via the f32 sidecar pool; \
         identical scores mean the silent Stage-1 fallback was taken \
         (Issue #790)"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn engine_search_with_rerank_factor_silently_falls_back_on_stage1_field() -> laurus::Result<()>
{
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    // No rerank_storage on the schema -> Stage 1 segment.
    let hnsw_opt = HnswOption {
        dimension: 4,
        distance: DistanceMetric::Cosine,
        m: 4,
        ef_construction: 16,
        rerank_storage: None,
        ..HnswOption::default()
    };

    let schema = Schema::builder()
        .add_field("embedding", FieldOption::Hnsw(hnsw_opt))
        .build();

    let engine = Engine::new(storage.clone(), schema).await?;
    let doc1 = Document::builder()
        .add_field("embedding", DataValue::Vector(vec![1.0, 0.0, 0.0, 0.0]))
        .build();
    let doc2 = Document::builder()
        .add_field("embedding", DataValue::Vector(vec![0.0, 1.0, 0.0, 0.0]))
        .build();
    engine.put_document("doc1", doc1).await?;
    engine.put_document("doc2", doc2).await?;
    engine.commit().await?;

    // Pins the no-behavior-change guarantee of Issue #790: a Stage-1
    // schema must not start emitting a sidecar.
    assert!(
        !storage.file_exists(SIDECAR_NAME),
        "a field without rerank_storage must not emit {SIDECAR_NAME}"
    );

    // Stage 1 segment + rerank_factor must succeed (no NotImplemented)
    // because the searcher silently degrades to int8 ranking.
    let results = engine
        .search(vector_request(&[0.95, 0.05, 0.0, 0.0], Some(5)))
        .await?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "doc1");
    Ok(())
}
