//! Issue #794: a non-Cosine HNSW field created through `VectorStore`/
//! `Engine` must NOT L2-normalize its stored vectors.
//!
//! The store write path used to leave `normalize_vectors` at the
//! always-on default, so a Euclidean field's vectors were normalized at
//! write time — which changes Euclidean distances (a magnitude-invariant
//! transform applied to a magnitude-sensitive metric). This test picks a
//! vector layout where normalization flips the nearest neighbour, so it
//! fails if (and only if) the vectors are wrongly normalized.

use tempfile::TempDir;

use laurus::Engine;
use laurus::SearchRequestBuilder;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::HnswOption;
use laurus::vector::Vector;
use laurus::{DataValue, DistanceMetric, Document};
use laurus::{FieldOption, QueryVector, Schema, VectorSearchQuery};

async fn top_hit_for_euclidean_field() -> laurus::Result<String> {
    let temp_dir = TempDir::new().unwrap();
    let storage =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(temp_dir.path())))?;

    // Euclidean HNSW field — normalize_vectors must be derived from the
    // metric (false for Euclidean) by the store config conversion.
    let schema = Schema::builder()
        .add_field(
            "embedding",
            FieldOption::Hnsw(HnswOption {
                dimension: 4,
                distance: DistanceMetric::Euclidean,
                m: 8,
                ef_construction: 64,
                ..HnswOption::default()
            }),
        )
        .build();

    let engine = Engine::new(storage.clone(), schema).await?;

    // Layout where L2-normalization flips the nearest neighbour:
    //   query          = [10, 0, 0, 0]
    //   "near"         = [11, 1, 0, 0]  -> Euclidean dist sqrt(2) ~= 1.41
    //   "far_same_dir" = [20, 0, 0, 0]  -> Euclidean dist 10, but identical
    //                                       direction to the query
    // True Euclidean nearest neighbour is "near". If the vectors are
    // (wrongly) normalized, the query and "far_same_dir" both collapse to
    // the unit vector [1,0,0,0], making "far_same_dir" the nearest — the
    // bug this test guards against.
    let docs = [
        ("near", vec![11.0, 1.0, 0.0, 0.0]),
        ("far_same_dir", vec![20.0, 0.0, 0.0, 0.0]),
    ];
    for (id, vec) in &docs {
        let doc = Document::builder()
            .add_field("embedding", DataValue::Vector(vec.clone()))
            .build();
        engine.put_document(id, doc.clone()).await?;
    }
    engine.commit().await?;

    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![10.0, 0.0, 0.0, 0.0]),
            weight: 1.0,
            fields: Some(vec!["embedding".to_string()]),
        }]))
        .limit(1)
        .build();

    let results = engine.search(request).await?.results;
    assert_eq!(results.len(), 1, "expected exactly 1 hit");
    Ok(results[0].id.clone())
}

#[tokio::test(flavor = "multi_thread")]
async fn euclidean_store_field_is_not_normalized() -> laurus::Result<()> {
    let top = top_hit_for_euclidean_field().await?;
    assert_eq!(
        top, "near",
        "the Euclidean-nearest doc must win; got '{top}'. A 'far_same_dir' \
         result means the store path L2-normalized the vectors, corrupting \
         Euclidean distances (Issue #794)"
    );
    Ok(())
}
