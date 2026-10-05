//! End-to-end tests of the late-interaction rescore stage (Issue #1345).

use std::collections::HashMap;
use std::sync::Arc;

use laurus::lexical::TermQuery;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::{
    DistanceMetric, Document, Engine, FieldOption, FlatOption, FusionAlgorithm, HybridMode,
    LaurusError, LexicalSearchQuery, MultiVectorOption, QueryVector, RescoreOptions, Result,
    Schema, SearchQuery, SearchRequest, SearchRequestBuilder, SearchResult, SortField, SortOrder,
    TextOption, VectorSearchQuery,
};

const TOKENS: &str = "tokens";

/// Query token vectors used by most tests.
const QUERY: [[f32; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];

/// `(id, title, vec, tokens)` of one test document.
type Entry = (&'static str, &'static str, [f32; 2], Vec<[f32; 2]>);

/// Against [`QUERY`] the late-interaction scores are c 1.1, b 1.0, d 0.9,
/// a 0.1, e 0.05 — an order that matches neither the BM25 nor the `vec`
/// ranking.
fn corpus() -> Vec<Entry> {
    vec![
        ("a", "rust", [1.0, 0.0], vec![[0.1, 0.0]]),
        ("b", "rust rust", [0.9, 0.1], vec![[0.5, 0.5], [0.0, 0.2]]),
        ("c", "rust language", [0.5, 0.5], vec![[0.9, 0.2]]),
        (
            "d",
            "rust rust rust",
            [0.2, 0.8],
            vec![[0.3, 0.3], [0.6, 0.0]],
        ),
        ("e", "learning rust today", [0.0, 1.0], vec![[0.02, 0.03]]),
    ]
}

fn schema(distance: DistanceMetric) -> Schema {
    Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_flat_field("vec", FlatOption::new(2))
        .add_field(
            TOKENS,
            FieldOption::MultiVector(MultiVectorOption::new(2).distance(distance)),
        )
        .build()
}

fn doc(title: &str, vec: [f32; 2], tokens: &[[f32; 2]]) -> Document {
    Document::builder()
        .add_text("title", title)
        .add_vector("vec", vec.to_vec())
        .add_vector_array(TOKENS, tokens.iter().map(|t| t.to_vec()).collect())
        .build()
}

async fn engine(distance: DistanceMetric) -> Result<Engine> {
    let engine = Engine::new(
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
        schema(distance),
    )
    .await?;
    for (id, title, vec, tokens) in corpus() {
        engine.put_document(id, doc(title, vec, &tokens)).await?;
    }
    engine.commit().await?;
    Ok(engine)
}

fn query_vectors(query: &[[f32; 2]]) -> Vec<Vector> {
    query.iter().map(|q| Vector::new(q.to_vec())).collect()
}

fn rescore(window_size: usize) -> RescoreOptions {
    RescoreOptions::late_interaction(TOKENS, query_vectors(&QUERY)).window_size(window_size)
}

fn vector_query() -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0, 0.0]),
        weight: 1.0,
        fields: Some(vec!["vec".to_string()]),
    }])
}

/// `Σ_i max_j q_i · d_j`, computed independently of the engine.
fn reference(query: &[[f32; 2]], tokens: &[[f32; 2]]) -> f32 {
    query
        .iter()
        .map(|q| {
            tokens
                .iter()
                .map(|d| q[0] * d[0] + q[1] * d[1])
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

fn tokens_by_id() -> HashMap<&'static str, Vec<[f32; 2]>> {
    corpus()
        .into_iter()
        .map(|(id, _, _, tokens)| (id, tokens))
        .collect()
}

/// The order the rescore stage must produce from a first-stage ranking:
/// the window by reference score, then the rest unchanged.
fn expected(baseline: &[SearchResult], window: usize) -> Vec<String> {
    let tokens = tokens_by_id();
    let window = window.min(baseline.len());
    let mut head: Vec<(&str, f32)> = baseline[..window]
        .iter()
        .map(|r| (r.id.as_str(), reference(&QUERY, &tokens[r.id.as_str()])))
        .collect();
    head.sort_by(|a, b| b.1.total_cmp(&a.1));
    head.iter()
        .map(|(id, _)| id.to_string())
        .chain(baseline[window..].iter().map(|r| r.id.clone()))
        .collect()
}

fn ids(results: &[SearchResult]) -> Vec<String> {
    results.iter().map(|r| r.id.clone()).collect()
}

fn lexical(rescore: Option<RescoreOptions>, offset: usize, limit: usize) -> SearchRequest {
    let mut builder = SearchRequestBuilder::new()
        .query_dsl("title:rust")
        .offset(offset)
        .limit(limit);
    if let Some(rescore) = rescore {
        builder = builder.rescore(rescore);
    }
    builder.build()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lexical_first_stage_is_reordered_by_late_interaction() -> Result<()> {
    let engine = engine(DistanceMetric::DotProduct).await?;
    let baseline = engine.search(lexical(None, 0, 10)).await?;
    assert_eq!(baseline.len(), 5);

    let results = engine.search(lexical(Some(rescore(100)), 0, 10)).await?;
    assert_eq!(ids(&results), ["c", "b", "d", "a", "e"]);
    assert_ne!(ids(&baseline), ids(&results));
    let tokens = tokens_by_id();
    for result in &results {
        let expected = reference(&QUERY, &tokens[result.id.as_str()]);
        assert!((result.score - expected).abs() < 1e-5, "{result:?}");
    }

    // A one-hit page still comes from the whole window: the first stage
    // fetches window_size candidates, not just offset + limit.
    assert_ne!(baseline[0].id, "c");
    let top = engine.search(lexical(Some(rescore(100)), 0, 1)).await?;
    assert_eq!(ids(&top), ["c"]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vector_first_stage_is_reordered() -> Result<()> {
    let engine = engine(DistanceMetric::DotProduct).await?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(vector_query())
                .rescore(rescore(100))
                .limit(10)
                .build(),
        )
        .await?;
    assert_eq!(ids(&results), ["c", "b", "d", "a", "e"]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_hybrid_first_stage_is_reordered_for_every_fusion() -> Result<()> {
    let engine = engine(DistanceMetric::DotProduct).await?;
    for fusion in [
        FusionAlgorithm::RRF { k: 60.0 },
        FusionAlgorithm::WeightedSum {
            lexical_weight: 0.5,
            vector_weight: 0.5,
        },
    ] {
        let results = engine
            .search(
                SearchRequestBuilder::new()
                    .lexical_query(LexicalSearchQuery::Dsl("title:rust".to_string()))
                    .vector_query(vector_query())
                    .fusion_algorithm(fusion)
                    .rescore(rescore(100))
                    .limit(10)
                    .build(),
            )
            .await?;
        assert_eq!(ids(&results), ["c", "b", "d", "a", "e"], "{fusion:?}");
    }

    let intersection = SearchRequest {
        query: SearchQuery::Hybrid {
            lexical: LexicalSearchQuery::Dsl("title:language".to_string()),
            vector: vector_query(),
            mode: HybridMode::Intersection,
        },
        rescore: Some(rescore(100)),
        ..Default::default()
    };
    assert_eq!(ids(&engine.search(intersection).await?), ["c"]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_only_the_window_is_reordered() -> Result<()> {
    let engine = engine(DistanceMetric::DotProduct).await?;
    let baseline = engine.search(lexical(None, 0, 10)).await?;
    for window in [1, 2, 3, 5] {
        let results = engine.search(lexical(Some(rescore(window)), 0, 10)).await?;
        assert_eq!(
            ids(&results),
            expected(&baseline, window),
            "window {window}"
        );
        // Beyond the window, first-stage scores are kept.
        for (result, first) in results[window..].iter().zip(&baseline[window..]) {
            assert_eq!(result.score, first.score);
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pages_across_the_window_concatenate_to_one_ranking() -> Result<()> {
    let engine = engine(DistanceMetric::DotProduct).await?;
    let baseline = engine.search(lexical(None, 0, 10)).await?;
    let whole = ids(&engine.search(lexical(Some(rescore(3)), 0, 10)).await?);
    assert_eq!(whole, expected(&baseline, 3));

    let mut paged = Vec::new();
    for offset in [0, 2, 4] {
        paged.extend(ids(&engine
            .search(lexical(Some(rescore(3)), offset, 2))
            .await?));
    }
    assert_eq!(paged, whole);

    // A page entirely beyond a smaller window still sees the reordered
    // ranking: the first stage fetches max(window, offset + limit).
    let page = ids(&engine.search(lexical(Some(rescore(1)), 3, 2)).await?);
    assert_eq!(page, expected(&baseline, 1)[3..5]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_candidates_without_token_vectors_follow_the_rescored_ones() -> Result<()> {
    let engine = engine(DistanceMetric::DotProduct).await?;
    engine
        .put_document(
            "f",
            Document::builder()
                .add_text("title", "rust rust rust rust")
                .build(),
        )
        .await?;
    engine.commit().await?;

    let baseline = engine.search(lexical(None, 0, 10)).await?;
    let results = engine.search(lexical(Some(rescore(100)), 0, 10)).await?;
    assert_eq!(ids(&results), ["c", "b", "d", "a", "e", "f"]);
    let first_stage = baseline.iter().find(|r| r.id == "f").unwrap().score;
    assert_eq!(results[5].score, first_stage);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rescore_reads_the_newest_vectors_and_respects_the_filter() -> Result<()> {
    let engine = engine(DistanceMetric::DotProduct).await?;
    engine
        .put_document("a", doc("rust", [1.0, 0.0], &[[2.0, 2.0]]))
        .await?;
    engine.commit().await?;
    let results = engine.search(lexical(Some(rescore(100)), 0, 10)).await?;
    assert_eq!(results[0].id, "a");
    assert!((results[0].score - 4.0).abs() < 1e-5);

    let filtered = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl("title:rust")
                .filter_query(Box::new(TermQuery::new("title", "language")))
                .rescore(rescore(100))
                .build(),
        )
        .await?;
    assert_eq!(ids(&filtered), ["c"]);
    Ok(())
}

/// Equal late-interaction scores fall back to the internal doc id, not to
/// the first-stage order, so a rescored page is deterministic.
#[tokio::test(flavor = "multi_thread")]
async fn test_ties_are_ordered_by_doc_id() -> Result<()> {
    let engine = Engine::new(
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
        schema(DistanceMetric::DotProduct),
    )
    .await?;
    // "y" is indexed first, so it gets the smaller doc id.
    engine
        .put_document("y", doc("rust", [1.0, 0.0], &[[0.5, 0.5]]))
        .await?;
    engine
        .put_document("x", doc("rust rust", [1.0, 0.0], &[[0.5, 0.5]]))
        .await?;
    engine.commit().await?;

    let baseline = engine.search(lexical(None, 0, 10)).await?;
    assert_eq!(ids(&baseline), ["x", "y"]);
    let results = engine.search(lexical(Some(rescore(100)), 0, 10)).await?;
    assert_eq!(ids(&results), ["y", "x"]);
    assert_eq!(results[0].score, results[1].score);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_batch_rescores_each_request() -> Result<()> {
    let engine = engine(DistanceMetric::DotProduct).await?;
    let batch = engine
        .search_batch(vec![
            lexical(Some(rescore(100)), 0, 10),
            lexical(Some(rescore(2)), 0, 10),
        ])
        .await?;
    assert_eq!(
        ids(&batch[0]),
        ids(&engine.search(lexical(Some(rescore(100)), 0, 10)).await?)
    );
    assert_eq!(
        ids(&batch[1]),
        ids(&engine.search(lexical(Some(rescore(2)), 0, 10)).await?)
    );
    Ok(())
}

/// Cosine fields normalize the query, so scaling it changes nothing.
#[tokio::test(flavor = "multi_thread")]
async fn test_cosine_rescore_ignores_query_scale() -> Result<()> {
    let engine = engine(DistanceMetric::Cosine).await?;
    let scaled: Vec<[f32; 2]> = QUERY.iter().map(|q| [q[0] * 3.0, q[1] * 3.0]).collect();
    let plain = engine.search(lexical(Some(rescore(100)), 0, 10)).await?;
    let request = lexical(
        Some(RescoreOptions::late_interaction(
            TOKENS,
            query_vectors(&scaled),
        )),
        0,
        10,
    );
    let scaled = engine.search(request).await?;
    assert_eq!(ids(&plain), ids(&scaled));
    for (a, b) in plain.iter().zip(&scaled) {
        assert!((a.score - b.score).abs() < 1e-5, "{a:?} vs {b:?}");
    }
    // Cosine scores are bounded by the number of query vectors.
    assert!(plain.iter().all(|r| r.score <= QUERY.len() as f32 + 1e-5));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_rescore_options_are_rejected() -> Result<()> {
    let engine = engine(DistanceMetric::DotProduct).await?;
    let with = |options: RescoreOptions| lexical(Some(options), 0, 10);
    let cases: Vec<(SearchRequest, &str)> = vec![
        (with(rescore(0)), "window_size must be between"),
        (
            with(rescore(RescoreOptions::MAX_WINDOW_SIZE + 1)),
            "window_size must be between",
        ),
        (
            with(RescoreOptions::late_interaction(TOKENS, Vec::new())),
            "between 1 and 1024 query vectors",
        ),
        (
            with(RescoreOptions::late_interaction(
                TOKENS,
                vec![Vector::new(vec![1.0, 0.0, 0.0])],
            )),
            "has dimension 3",
        ),
        (
            with(RescoreOptions::late_interaction(
                TOKENS,
                vec![Vector::new(vec![f32::NAN, 0.0])],
            )),
            "non-finite",
        ),
        (
            with(RescoreOptions::late_interaction(
                "vec",
                query_vectors(&QUERY),
            )),
            "needs a MultiVector field",
        ),
        (
            with(RescoreOptions::late_interaction(
                "missing",
                query_vectors(&QUERY),
            )),
            "unknown field 'missing'",
        ),
        (
            SearchRequestBuilder::new()
                .query_dsl("title:rust")
                .sort_by(SortField::Field {
                    name: "title".to_string(),
                    order: SortOrder::Asc,
                })
                .rescore(rescore(100))
                .build(),
            "cannot be combined with sort_by",
        ),
    ];
    for (request, expected) in cases {
        let err = engine.search(request).await.unwrap_err();
        assert!(matches!(err, LaurusError::InvalidArgument(_)), "{err}");
        assert!(err.to_string().contains(expected), "{err}");
    }
    Ok(())
}
