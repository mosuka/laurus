use tempfile::TempDir;

use laurus::Document;
use laurus::Engine;
use laurus::lexical::TermQuery;
use laurus::lexical::TextOption;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::HnswOption;
use laurus::vector::Vector;
use laurus::{FieldOption, Schema};
use laurus::{
    FusionAlgorithm, LexicalSearchQuery, QueryVector, SearchRequestBuilder, VectorSearchQuery,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_advanced_fusion_normalization() -> laurus::Result<()> {
    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Engine
    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("embedding", FieldOption::Hnsw(HnswOption::default()))
        .build();

    let engine = Engine::new(storage, config).await?;

    // 3. Index Documents
    // Doc 1: Good lexical, Bad vector
    let mut vec1 = vec![0.0; 128];
    vec1[0] = 1.0;
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "apple")
                .add_field("embedding", vec1)
                .build(),
        )
        .await?;

    // Doc 2: Bad lexical, Good vector
    let mut vec2 = vec![0.0; 128];
    vec2[1] = 1.0;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("title", "banana")
                .add_field("embedding", vec2)
                .build(),
        )
        .await?;

    engine.commit().await?;

    let mut query_vec = vec![0.0; 128];
    query_vec[1] = 1.0;
    let request = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "title", "apple",
        ))))
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(query_vec),
            weight: 1.0,
            fields: Some(vec!["embedding".into()]),
        }]))
        .fusion_algorithm(FusionAlgorithm::WeightedSum {
            lexical_weight: 0.5,
            vector_weight: 0.5,
        })
        .build();

    let results = engine.search(request).await?;
    assert_eq!(results.len(), 2);

    // Without normalization, vector scores (cosine similarity with small values)
    // might be much smaller than lexical scores (BM25 for rare term).
    // With Min-Max normalization, both get [0, 1] range.
    // Doc 1: Lexical=1.0, Vector=0.0 -> WeightedSum = 0.5
    // Doc 2: Lexical=0.0, Vector=1.0 -> WeightedSum = 0.5
    // Actually, with only 2 docs, one is min and one is max.
    // They should have equal scores if weights are equal.
    assert_eq!(results[0].score, results[1].score);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_field_boosts() -> laurus::Result<()> {
    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Engine
    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("body", FieldOption::Text(TextOption::default()))
        .build();

    let engine = Engine::new(storage, config).await?;

    // 3. Index Documents
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "rust")
                .add_field("body", "programming")
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("title", "java")
                .add_field("body", "rust")
                .build(),
        )
        .await?;
    engine.commit().await?;

    // 4. Search for "rust" in both fields with different boosts
    // Case A: Boost title
    let req_a = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(
            laurus::lexical::BooleanQueryBuilder::new()
                .should(Box::new(TermQuery::new("title", "rust")))
                .should(Box::new(TermQuery::new("body", "rust")))
                .build(),
        )))
        .add_field_boost("title", 10.0)
        .add_field_boost("body", 1.0)
        .build();

    let res_a = engine.search(req_a).await?;
    // res_a[0].id is the external ID (String)
    let docs_a = engine.get_documents(&res_a[0].id).await?;
    let doc_a = &docs_a[0];
    assert_eq!(
        doc_a.fields.get("_id").and_then(|v| v.as_text()),
        Some("doc1"),
        "Doc 1 should win when title is boosted"
    );

    // Case B: Boost body
    let req_b = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(
            laurus::lexical::BooleanQueryBuilder::new()
                .should(Box::new(TermQuery::new("title", "rust")))
                .should(Box::new(TermQuery::new("body", "rust")))
                .build(),
        )))
        .add_field_boost("title", 1.0)
        .add_field_boost("body", 10.0)
        .build();

    let res_b = engine.search(req_b).await?;
    let docs_b = engine.get_documents(&res_b[0].id).await?;
    let doc_b = &docs_b[0];
    assert_eq!(
        doc_b.fields.get("_id").and_then(|v| v.as_text()),
        Some("doc2"),
        "Doc 2 should win when body is boosted"
    );

    Ok(())
}

/// #1084: with `FusionAlgorithm::WeightedSum` and `lexical_weight: 0.0`, the
/// fused score reduces to the vector side alone, so a higher `base_weight`
/// on the field carrying doc2's vector must make it outrank doc1 even
/// though both documents share an identical lexical hit.
#[tokio::test(flavor = "multi_thread")]
async fn test_base_weight_affects_weighted_sum_fusion() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field(
            "vec_a",
            FieldOption::Hnsw(HnswOption {
                base_weight: 1.0,
                ..HnswOption::default()
            }),
        )
        .add_field(
            "vec_b",
            FieldOption::Hnsw(HnswOption {
                base_weight: 5.0,
                ..HnswOption::default()
            }),
        )
        .build();

    let engine = Engine::new(storage, config).await?;

    let mut shared_vec = vec![0.0; 128];
    shared_vec[0] = 1.0;

    // Same lexical hit, same vector, one in each field.
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "apple")
                .add_field("vec_a", shared_vec.clone())
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("title", "apple")
                .add_field("vec_b", shared_vec.clone())
                .build(),
        )
        .await?;
    engine.commit().await?;

    let request = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "title", "apple",
        ))))
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(shared_vec),
            weight: 1.0,
            fields: Some(vec!["vec_a".into(), "vec_b".into()]),
        }]))
        .fusion_algorithm(FusionAlgorithm::WeightedSum {
            lexical_weight: 0.0,
            vector_weight: 1.0,
        })
        .build();

    let results = engine.search(request).await?;
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].id,
        "doc2",
        "vec_b's higher base_weight must rank doc2 first when the fused \
         score is entirely vector-driven, got {:?}",
        results.iter().map(|h| &h.id).collect::<Vec<_>>()
    );

    Ok(())
}

/// #1084: `base_weight` cannot affect `FusionAlgorithm::RRF` directly — it
/// is rank-only and ignores the underlying score entirely, so a uniform
/// per-field scalar multiplied into two documents that are ALREADY tied on
/// every other axis (same lexical hit, symmetric vector ranks) cancels out
/// exactly (rank 1 + rank 2 sums to the same RRF total regardless of which
/// document holds which rank). This is a deliberate property of RRF, not a
/// bug: it is why the docs correct the "hybrid search fusion weight"
/// claim. What `base_weight` DOES still do end-to-end through the Engine
/// is affect a vector-only query's own ranking (verified below) and,
/// through it, a `WeightedSum`-fused hybrid search (verified above).
#[tokio::test(flavor = "multi_thread")]
async fn test_base_weight_affects_vector_only_query_through_the_engine() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    let config = Schema::builder()
        .add_field(
            "vec_a",
            FieldOption::Hnsw(HnswOption {
                base_weight: 1.0,
                ..HnswOption::default()
            }),
        )
        .add_field(
            "vec_b",
            FieldOption::Hnsw(HnswOption {
                base_weight: 5.0,
                ..HnswOption::default()
            }),
        )
        .build();

    let engine = Engine::new(storage, config).await?;

    let mut vec_close = vec![0.0; 128];
    vec_close[0] = 1.0;
    let mut vec_far = vec![0.0; 128];
    vec_far[0] = 0.9;
    vec_far[1] = 0.1;

    // doc1's vector is the closer match in vec_a (weight 1.0); doc2's is
    // the farther match, but lives in vec_b (weight 5.0). Without
    // base_weight, doc1 would rank first on the vector side; with it,
    // doc2's weighted score overtakes doc1's.
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("vec_a", vec_close.clone())
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder().add_field("vec_b", vec_far).build(),
        )
        .await?;
    engine.commit().await?;

    // A vector-only request (no `lexical_query`) is a single-mode search
    // that bypasses `Engine::fuse_results` entirely — the returned score
    // IS the base_weight-adjusted similarity.
    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec_close),
            weight: 1.0,
            fields: Some(vec!["vec_a".into(), "vec_b".into()]),
        }]))
        .build();

    let results = engine.search(request).await?;
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].id,
        "doc2",
        "vec_b's base_weight must overtake doc1's closer-but-unweighted \
         match through the full Engine pipeline: {:?}",
        results.iter().map(|h| &h.id).collect::<Vec<_>>()
    );

    Ok(())
}
