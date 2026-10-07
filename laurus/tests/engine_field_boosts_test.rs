//! Issue #1374: a `field_boosts` key that names no lexical field is rejected
//! instead of being silently ignored, like an unknown field in the DSL
//! (#1253).

use std::sync::Arc;

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
