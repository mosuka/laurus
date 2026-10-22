use std::sync::Arc;

use tempfile::TempDir;

use laurus::Document;
use laurus::Engine;
use laurus::lexical::TermQuery;
use laurus::lexical::TextOption;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::memory::MemoryStorage;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::Vector;
use laurus::vector::{FlatOption, HnswOption};
use laurus::{FieldOption, GroupBy, LaurusError, Schema};
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

/// Two documents that both match `title:rust` and both have a vector: doc1
/// ranks first on the lexical side (the shorter title) and doc2 on the
/// vector side, so min-max normalization maps each side to exactly `1.0`
/// and `0.0` (Issue #1540).
async fn two_sided_engine() -> Engine {
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("vec", FieldOption::Flat(FlatOption::default().dimension(2)))
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    for (id, title, vector) in [
        ("doc1", "rust", vec![0.6, 0.8]),
        ("doc2", "rust go programming language", vec![1.0, 0.0]),
    ] {
        engine
            .put_document(
                id,
                Document::builder()
                    .add_field("title", title)
                    .add_field("vec", vector)
                    .build(),
            )
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

fn hybrid(fusion: FusionAlgorithm) -> SearchRequestBuilder {
    SearchRequestBuilder::new()
        .query_dsl("title:rust")
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![1.0, 0.0]),
            weight: 1.0,
            fields: Some(vec!["vec".into()]),
        }]))
        .fusion_algorithm(fusion)
}

/// Issue #1540: an RRF `k` that is not a finite number of at least `0`, a
/// `WeightedSum` weight outside `0.0..=1.0` and two weights of `0` are
/// rejected with a message naming the value, also where the fusion has no
/// effect (a lexical-only search) and on a grouped request with `limit 0`,
/// which returns before searching.
#[tokio::test(flavor = "multi_thread")]
async fn out_of_range_fusion_parameters_are_rejected() {
    let engine = two_sided_engine().await;
    let weighted = |lexical_weight, vector_weight| FusionAlgorithm::WeightedSum {
        lexical_weight,
        vector_weight,
    };
    let mut cases: Vec<(FusionAlgorithm, String)> = [-1.0, -0.5, f64::NAN, f64::INFINITY]
        .into_iter()
        .map(|k| {
            (
                FusionAlgorithm::RRF { k },
                format!("fusion: rrf: k must be finite and at least 0, got {k}"),
            )
        })
        .collect();
    for (lexical_weight, vector_weight, name, bad) in [
        (2.0, 0.0, "lexical_weight", 2.0),
        (-0.1, 1.0, "lexical_weight", -0.1),
        (f32::NAN, 1.0, "lexical_weight", f32::NAN),
        (1.0, f32::INFINITY, "vector_weight", f32::INFINITY),
    ] {
        cases.push((
            weighted(lexical_weight, vector_weight),
            format!("fusion: weighted_sum: {name} must be between 0 and 1, got {bad}"),
        ));
    }
    cases.push((
        weighted(0.0, 0.0),
        "fusion: weighted_sum: lexical_weight or vector_weight must be greater than 0".to_string(),
    ));

    // Every request that is not rejected as expected, so a failure lists all
    // of them rather than the first.
    let mut wrong = Vec::new();
    for (fusion, expected) in cases {
        for (shape, request) in [
            ("hybrid", hybrid(fusion).build()),
            (
                "lexical-only",
                SearchRequestBuilder::new()
                    .query_dsl("title:rust")
                    .fusion_algorithm(fusion)
                    .build(),
            ),
            (
                "grouped limit 0",
                hybrid(fusion)
                    .group_by(GroupBy::new("_id"))
                    .limit(0)
                    .build(),
            ),
        ] {
            let result = engine.search(request).await;
            if !matches!(&result, Err(LaurusError::InvalidArgument(m)) if m.contains(&expected)) {
                wrong.push(format!("{shape} {fusion:?}: {:?}", result.map(|r| r.len())));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1540: the boundaries are accepted. `k = 0` gives finite scores
/// (`1 / rank`), weights of `1` and `0` either way round are a one-sided
/// ranking, and in-range weights are applied as given: not clamped and not
/// divided by their sum.
#[tokio::test(flavor = "multi_thread")]
async fn boundary_fusion_parameters_are_accepted() {
    let engine = two_sided_engine().await;
    let scores = |results: Vec<laurus::SearchResult>| -> Vec<(String, f32)> {
        results.into_iter().map(|r| (r.id, r.score)).collect()
    };

    // doc1 is 1st lexically and 2nd by vector, doc2 the other way round.
    let rrf = scores(
        engine
            .search(hybrid(FusionAlgorithm::RRF { k: 0.0 }).build())
            .await
            .unwrap(),
    );
    assert_eq!(
        rrf,
        [("doc1".to_string(), 1.5), ("doc2".to_string(), 1.5)],
        "{rrf:?}"
    );

    for (lexical_weight, vector_weight, expected) in [
        (1.0, 0.0, [("doc1", 1.0), ("doc2", 0.0)]),
        (0.0, 1.0, [("doc2", 1.0), ("doc1", 0.0)]),
    ] {
        let one_sided = scores(
            engine
                .search(
                    hybrid(FusionAlgorithm::WeightedSum {
                        lexical_weight,
                        vector_weight,
                    })
                    .build(),
                )
                .await
                .unwrap(),
        );
        let expected: Vec<(String, f32)> = expected
            .iter()
            .map(|&(id, score)| (id.to_string(), score))
            .collect();
        assert_eq!(one_sided, expected, "({lexical_weight}, {vector_weight})");
    }

    // Normalized: doc1 is (lexical 1, vector 0) and doc2 is (0, 1).
    let weighted = scores(
        engine
            .search(
                hybrid(FusionAlgorithm::WeightedSum {
                    lexical_weight: 0.2,
                    vector_weight: 0.4,
                })
                .build(),
            )
            .await
            .unwrap(),
    );
    assert_eq!(
        weighted,
        [("doc2".to_string(), 0.4), ("doc1".to_string(), 0.2)],
        "{weighted:?}"
    );
}
