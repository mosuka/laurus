//! Issue #1367: extreme vector search sizes through `Engine::search` must
//! return the ordinary results instead of panicking on an overflow or asking
//! for an allocation sized by the request.

use std::sync::Arc;

use laurus::storage::memory::MemoryStorage;
use laurus::vector::{HnswOption, QueryVector, Vector, VectorSearchQuery};
use laurus::{DataValue, Document, Engine, FieldOption, Schema, SearchRequestBuilder};

/// An HNSW field `vec` with 20 documents on a circle, one of them deleted so
/// the graph search also runs its deletion-aware branch.
async fn engine() -> Engine {
    let schema = Schema::builder()
        .add_field("vec", FieldOption::Hnsw(HnswOption::default().dimension(2)))
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    for i in 0..20 {
        let angle = i as f32 * 0.15;
        engine
            .put_document(
                &format!("d{i:02}"),
                Document::builder()
                    .add_field("vec", DataValue::Vector(vec![angle.cos(), angle.sin()]))
                    .build(),
            )
            .await
            .unwrap();
    }
    engine.commit().await.unwrap();
    engine.delete_documents("d01").await.unwrap();
    engine.commit().await.unwrap();
    engine
}

fn query() -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0, 0.0]),
        weight: 1.0,
        fields: Some(vec!["vec".to_string()]),
    }])
}

async fn ids(engine: &Engine, builder: SearchRequestBuilder) -> Vec<String> {
    engine
        .search(builder.vector_query(query()).limit(5).build())
        .await
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn extreme_vector_search_sizes_return_the_ordinary_results() {
    let engine = engine().await;
    let expected = ids(&engine, SearchRequestBuilder::new()).await;
    assert_eq!(expected, ["d00", "d02", "d03", "d04", "d05"]);

    let cases = [
        (
            "ef_search = usize::MAX",
            SearchRequestBuilder::new().vector_ef_search(usize::MAX),
        ),
        (
            "overfetch = f32::MAX",
            SearchRequestBuilder::new().vector_overfetch(f32::MAX),
        ),
        (
            "rerank_factor = usize::MAX",
            SearchRequestBuilder::new().vector_rerank_factor(usize::MAX),
        ),
    ];
    for (case, builder) in cases {
        assert_eq!(ids(&engine, builder).await, expected, "{case}");
    }
}
