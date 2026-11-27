//! Issue #1691: a top-k search with a span query returns the highest-scoring
//! documents when some have more than 100 spans.
//!
//! A span query scores a document as its number of spans times its boost,
//! but its scorer gave `boost * 100` as the bound of its scores. The search
//! loop stops once that bound is at or below the lowest competitive score,
//! which it checks after each batch of 8 documents, so a top-1 search over
//! eight documents with 150 spans returned one of them and never reached a
//! later document with 200.

use std::sync::Arc;

use laurus::lexical::query::span::{SpanQueryWrapper, SpanTermQuery};
use laurus::storage::memory::MemoryStorage;
use laurus::{Document, Engine, LexicalSearchQuery, Schema, SearchRequestBuilder, TextOption};

/// `world`, `times` times.
fn worlds(times: usize) -> String {
    vec!["world"; times].join(" ")
}

#[tokio::test(flavor = "multi_thread")]
async fn a_top_k_span_search_reaches_a_document_with_more_than_100_spans() {
    let schema = Schema::builder()
        .add_text_field("content", TextOption::default())
        .build();
    let engine = Engine::new(Arc::new(MemoryStorage::new(Default::default())), schema)
        .await
        .unwrap();
    // A full batch of documents scoring 150, then one scoring 200, in one
    // commit, so they share a segment and the loop meets them in this order.
    for i in 0..8 {
        engine
            .put_document(
                &format!("a{i}"),
                Document::builder().add_text("content", worlds(150)).build(),
            )
            .await
            .unwrap();
    }
    engine
        .put_document(
            "b",
            Document::builder().add_text("content", worlds(200)).build(),
        )
        .await
        .unwrap();
    engine.commit().await.unwrap();

    let query = SpanQueryWrapper::new(Box::new(SpanTermQuery::new("content", "world")));
    let hits = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(query)))
                .limit(1)
                .build(),
        )
        .await
        .unwrap()
        .results;
    let top: Vec<(String, f32)> = hits.iter().map(|h| (h.id.clone(), h.score)).collect();
    assert_eq!(top, [("b".to_string(), 200.0)]);
}
