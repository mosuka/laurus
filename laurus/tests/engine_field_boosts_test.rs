//! Issue #1374: a `field_boosts` key that names no lexical field is rejected
//! instead of being silently ignored, like an unknown field in the DSL
//! (#1253).
//!
//! Issue #1540: a boost that is not a finite number of at least `0`, and a
//! lexical `min_score` that is not finite, are rejected instead of hiding
//! hits or giving `NaN` or infinite scores.

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::MemoryStorage;
use laurus::vector::{FlatOption, QueryVector, Vector, VectorSearchQuery};
use laurus::{DataValue, Document, Engine, FieldOption, LaurusError, Schema};
use laurus::{SearchRequestBuilder, TextOption};

async fn engine() -> Engine {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field("vec", FieldOption::Flat(FlatOption::default().dimension(2)))
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    engine
        .put_document(
            "d1",
            Document::builder()
                .add_text("title", "rust")
                .add_field("vec", DataValue::Vector(vec![1.0, 0.0]))
                .build(),
        )
        .await
        .unwrap();
    engine.commit().await.unwrap();
    engine
}

fn vectors() -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0, 0.0]),
        weight: 1.0,
        fields: Some(vec!["vec".to_string()]),
    }])
}

#[tokio::test(flavor = "multi_thread")]
async fn unknown_and_vector_field_boosts_are_rejected() {
    let engine = engine().await;
    for (key, expected) in [("titel", "unknown field"), ("vec", "not a lexical field")] {
        let err = engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl("title:rust")
                    .add_field_boost(key, 2.0)
                    .build(),
            )
            .await
            .expect_err(key);
        assert!(
            matches!(&err, LaurusError::InvalidArgument(m) if m.contains(key) && m.contains(expected)),
            "{key}: {err:?}"
        );
    }
}

/// A lexical field is accepted even when the query does not reference it,
/// or when the request has no lexical part at all; the boost is unused.
#[tokio::test(flavor = "multi_thread")]
async fn lexical_field_boosts_are_accepted() {
    let engine = engine().await;
    for builder in [
        SearchRequestBuilder::new().query_dsl("title:rust"),
        SearchRequestBuilder::new().query_dsl("_id:d1"),
        SearchRequestBuilder::new().vector_query(vectors()),
    ] {
        let results = engine
            .search(
                builder
                    .add_field_boost("title", 2.0)
                    .add_field_boost("_id", 1.5)
                    .build(),
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }
}

/// Issue #1540: a negative boost (`-0.0` included) or a non-finite one, and
/// a non-finite lexical `min_score`, are rejected with a message naming the
/// value, also on a vector-only request where they have no effect.
#[tokio::test(flavor = "multi_thread")]
async fn out_of_range_boosts_and_min_scores_are_rejected() {
    let engine = engine().await;
    let requests = || {
        [
            (
                "lexical",
                SearchRequestBuilder::new().query_dsl("title:rust"),
            ),
            (
                "vector-only",
                SearchRequestBuilder::new().vector_query(vectors()),
            ),
        ]
    };
    // Every request that is not rejected as expected, so a failure lists all
    // of them rather than the first.
    let mut wrong = Vec::new();
    for boost in [-1.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let expected =
            format!("boost for field \"title\" must be finite and at least 0, got {boost}");
        for (shape, builder) in requests() {
            let result = engine
                .search(builder.add_field_boost("title", boost).build())
                .await;
            if !matches!(&result, Err(LaurusError::InvalidArgument(m)) if m.contains(&expected)) {
                wrong.push(format!(
                    "{shape} boost {boost}: {:?}",
                    result.map(|r| r.len())
                ));
            }
        }
    }
    for min_score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let expected = format!("lexical min_score must be finite, got {min_score}");
        for (shape, builder) in requests() {
            let result = engine
                .search(builder.lexical_min_score(min_score).build())
                .await;
            if !matches!(&result, Err(LaurusError::InvalidArgument(m)) if m.contains(&expected)) {
                wrong.push(format!(
                    "{shape} min_score {min_score}: {:?}",
                    result.map(|r| r.len())
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1540: a boost of `0` keeps each hit with a score of `+0.0`, on an
/// index of two segments whose per-segment collectors drop scores below a
/// fixed `0.0`; a negative lexical `min_score` returns the same hits as `0`.
#[tokio::test(flavor = "multi_thread")]
async fn zero_boost_and_negative_min_score_keep_the_hits() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .build();
    let engine = Engine::new(storage.clone(), schema).await.unwrap();
    for id in ["d1", "d2"] {
        engine
            .put_document(id, Document::builder().add_text("title", "rust").build())
            .await
            .unwrap();
        engine.commit().await.unwrap();
    }
    let files = storage.list_files().unwrap();
    let segments = files
        .iter()
        .filter(|f| f.starts_with("lexical/segment_") && f.ends_with(".cfs"))
        .count();
    assert_eq!(
        segments, 2,
        "the test needs two lexical segments: {files:?}"
    );

    let hits = |results: Vec<laurus::SearchResult>| {
        let mut hits: Vec<(String, f32)> = results.into_iter().map(|r| (r.id, r.score)).collect();
        hits.sort_by(|a, b| a.0.cmp(&b.0));
        hits
    };

    let zero = hits(
        engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl("title:rust")
                    .add_field_boost("title", 0.0)
                    .build(),
            )
            .await
            .unwrap(),
    );
    assert_eq!(
        zero,
        [("d1".to_string(), 0.0), ("d2".to_string(), 0.0)],
        "{zero:?}"
    );
    assert!(zero.iter().all(|(_, s)| s.is_sign_positive()), "{zero:?}");

    let search_with_min_score = |min_score: f32| {
        engine.search(
            SearchRequestBuilder::new()
                .query_dsl("title:rust")
                .lexical_min_score(min_score)
                .build(),
        )
    };
    let at_zero = hits(search_with_min_score(0.0).await.unwrap());
    let negative = hits(search_with_min_score(-1.0).await.unwrap());
    assert_eq!(at_zero.len(), 2, "{at_zero:?}");
    assert_eq!(negative, at_zero);
}
