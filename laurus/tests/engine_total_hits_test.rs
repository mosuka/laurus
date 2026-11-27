//! Issue #1472: the total an `Engine` search reports, by the kind of
//! search.
//!
//! Twenty documents have `rust` in their title and ten `python`; the rust
//! ones have vectors near `[1, 0]` in document order, the python ones near
//! `[0, 1]`. Each belongs to one of five parents.

use std::sync::Arc;

use laurus::lexical::TermQuery;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::{
    Document, Engine, FlatOption, GroupBy, HybridMode, LaurusError, LexicalSearchQuery,
    QueryVector, Result, Schema, SearchQuery, SearchRequest, SearchRequestBuilder, SearchResponse,
    TextOption, TotalHits, TotalHitsRelation, TrackTotalHits, VectorSearchQuery,
};

const RUST: u64 = 20;
const PYTHON: u64 = 10;

async fn engine() -> Result<Engine> {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("parent", TextOption::default())
        .add_flat_field("vec", FlatOption::new(2))
        .build();
    let engine = Engine::new(
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
        schema,
    )
    .await?;
    for i in 0..RUST + PYTHON {
        let (title, vec) = if i < RUST {
            ("rust", [1.0, i as f32 * 0.01])
        } else {
            ("python", [i as f32 * 0.01, 1.0])
        };
        engine
            .add_document(
                &format!("d{i:02}"),
                Document::builder()
                    .add_text("title", title)
                    .add_text("parent", format!("p{}", i % 5))
                    .add_vector("vec", vec.to_vec())
                    .build(),
            )
            .await?;
    }
    engine.commit().await?;
    Ok(engine)
}

fn vector(target: [f32; 2]) -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(target.to_vec()),
        weight: 1.0,
        fields: Some(vec!["vec".to_string()]),
    }])
}

fn lexical(term: &str) -> LexicalSearchQuery {
    LexicalSearchQuery::from(format!("title:{term}").as_str())
}

async fn total(engine: &Engine, request: SearchRequest) -> Result<Option<TotalHits>> {
    Ok(engine.search(request).await?.total_hits)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_lexical_total_is_exact_below_the_threshold_and_at_least_it_from_it() -> Result<()> {
    let engine = engine().await?;
    let search = |track| {
        SearchRequestBuilder::new()
            .lexical_query(lexical("rust"))
            .limit(5)
            .track_total_hits(track)
            .build()
    };
    assert_eq!(
        total(&engine, search(TrackTotalHits::Disabled)).await?,
        None
    );
    assert_eq!(
        total(&engine, search(TrackTotalHits::UpTo(100))).await?,
        Some(TotalHits::exact(RUST))
    );
    assert_eq!(
        total(&engine, search(TrackTotalHits::UpTo(10))).await?,
        Some(TotalHits::at_least(10))
    );
    assert_eq!(
        total(&engine, search(TrackTotalHits::Exact)).await?,
        Some(TotalHits::exact(RUST))
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_threshold_of_zero_is_rejected() -> Result<()> {
    let engine = engine().await?;
    let result = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(lexical("rust"))
                .track_total_hits(TrackTotalHits::UpTo(0))
                .build(),
        )
        .await;
    assert!(
        matches!(&result, Err(LaurusError::InvalidArgument(m)) if m.contains("track_total_hits")),
        "{result:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_vector_total_is_at_least_the_hits_it_ranked() -> Result<()> {
    let engine = engine().await?;
    // `offset + limit` hits are ranked, five of the thirty with vectors.
    let response = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(vector([1.0, 0.0]))
                .offset(2)
                .limit(3)
                .track_total_hits(TrackTotalHits::Exact)
                .build(),
        )
        .await?;
    assert_eq!(response.results.len(), 3);
    assert_eq!(response.total_hits, Some(TotalHits::at_least(5)));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_union_total_is_at_least_the_lexical_count_or_the_fused_candidates() -> Result<()> {
    let engine = engine().await?;
    let union = |term: &str, target: [f32; 2]| {
        SearchRequestBuilder::new()
            .lexical_query(lexical(term))
            .vector_query(vector(target))
            .limit(3)
            .track_total_hits(TrackTotalHits::UpTo(1_000))
            .build()
    };
    // Each list fetches 2 × 3 = 6. The lexical count, 20, is the larger.
    assert_eq!(
        total(&engine, union("rust", [1.0, 0.0])).await?,
        Some(TotalHits::at_least(RUST))
    );
    // Six python documents lexically and six rust ones by vector: twelve
    // fused candidates, more than the ten python matches.
    assert_eq!(
        total(&engine, union("python", [1.0, 0.0])).await?,
        Some(TotalHits::at_least(12))
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_intersection_total_is_at_least_the_fused_candidates() -> Result<()> {
    let engine = engine().await?;
    // Both lists' first six are d00 to d05, so six candidates survive the
    // intersection, fewer than the twenty lexical matches.
    let request = SearchRequest {
        query: SearchQuery::Hybrid {
            lexical: lexical("rust"),
            vector: vector([1.0, 0.0]),
            mode: HybridMode::Intersection,
        },
        limit: 3,
        track_total_hits: TrackTotalHits::UpTo(1_000),
        ..Default::default()
    };
    let response = engine.search(request).await?;
    assert_eq!(response.results.len(), 3);
    assert_eq!(response.total_hits, Some(TotalHits::at_least(6)));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_grouped_total_counts_documents_not_groups() -> Result<()> {
    let engine = engine().await?;
    let grouped = |limit| {
        SearchRequestBuilder::new()
            .lexical_query(lexical("rust"))
            .group_by(GroupBy::new("parent"))
            .limit(limit)
            .track_total_hits(TrackTotalHits::UpTo(1_000))
            .build()
    };
    let response = engine.search(grouped(10)).await?;
    assert_eq!(response.results.len(), 5, "five parents");
    assert_eq!(response.total_hits, Some(TotalHits::exact(RUST)));
    // With a limit of 0 the search still counts.
    let response = engine.search(grouped(0)).await?;
    assert!(response.results.is_empty());
    assert_eq!(response.total_hits, Some(TotalHits::exact(RUST)));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_filter_that_matches_nothing_counts_zero_exactly() -> Result<()> {
    let engine = engine().await?;
    let response = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(vector([1.0, 0.0]))
                .filter_query(Box::new(TermQuery::new("title", "nothing")))
                .track_total_hits(TrackTotalHits::UpTo(10))
                .build(),
        )
        .await?;
    assert!(response.results.is_empty());
    assert_eq!(response.total_hits, Some(TotalHits::exact(0)));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn search_batch_returns_a_response_per_request() -> Result<()> {
    let engine = engine().await?;
    let request = |term: &str| {
        SearchRequestBuilder::new()
            .lexical_query(lexical(term))
            .track_total_hits(TrackTotalHits::Exact)
            .build()
    };
    let responses: Vec<SearchResponse> = engine
        .search_batch(vec![request("rust"), request("python")])
        .await?;
    let totals: Vec<(u64, TotalHitsRelation)> = responses
        .iter()
        .map(|r| {
            let t = r.total_hits.unwrap();
            (t.value, t.relation)
        })
        .collect();
    assert_eq!(
        totals,
        [
            (RUST, TotalHitsRelation::Exact),
            (PYTHON, TotalHitsRelation::Exact)
        ]
    );
    Ok(())
}
