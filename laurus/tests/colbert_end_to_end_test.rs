//! End to end with a real ColBERT model (Issue #1349): a schema declares a
//! `candle_colbert` embedder, documents are indexed as text, and a text
//! query rescores them. Downloads the model, so it is ignored by default:
//!
//! ```text
//! cargo test -p laurus --features embeddings-candle --test colbert_end_to_end_test -- --ignored
//! ```

#![cfg(feature = "embeddings-candle")]

use std::sync::Arc;

use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{
    Document, EmbedderDefinition, Engine, MultiVectorOption, RescoreOptions, Result, Schema,
    SearchRequestBuilder, TextOption,
};

/// answerdotai/answerai-colbert-small-v1 at the commit the parity
/// fixtures were generated from.
const MODEL: &str = "answerdotai/answerai-colbert-small-v1";
const REVISION: &str = "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e";

const DOCUMENTS: [(&str, &str); 4] = [
    (
        "rust",
        "Rust's ownership and borrowing rules guarantee memory safety without a garbage collector.",
    ),
    (
        "coffee",
        "Brewing coffee with a French press takes about four minutes of steeping.",
    ),
    (
        "paris",
        "Paris is the capital of France and is famous for the Eiffel Tower.",
    ),
    (
        "colbert",
        "ColBERT scores a passage by matching every query token against its best passage token.",
    ),
];

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads answerdotai/answerai-colbert-small-v1"]
async fn test_text_ingest_and_text_rescore_with_a_real_model() -> Result<()> {
    let schema = Schema::builder()
        .add_embedder(
            "colbert",
            EmbedderDefinition::CandleColbert {
                model: MODEL.to_string(),
                revision: Some(REVISION.to_string()),
                query_maxlen: None,
                doc_maxlen: None,
            },
        )
        .add_text_field("body", TextOption::default())
        .add_multi_vector_field("body_colbert", MultiVectorOption::new(96).embedder("colbert"))
        .build();
    let engine = Engine::builder(
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
        schema,
    )
    .build()
    .await?;

    for (id, text) in DOCUMENTS {
        // Every body mentions "note", so the first stage returns them all
        // and the rescore alone decides the order.
        let body = format!("{text} (note)");
        let document = Document::builder()
            .add_text("body", body.clone())
            .add_text("body_colbert", body)
            .build();
        engine.put_document(id, document).await?;
    }
    engine.commit().await?;

    for (query, expected) in [
        ("how does rust keep memory safe?", "rust"),
        ("what is the capital of france", "paris"),
        ("how long should coffee steep", "coffee"),
        ("late interaction retrieval with token matching", "colbert"),
    ] {
        let results = engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl("body:note")
                    .rescore(RescoreOptions::late_interaction_text("body_colbert", query))
                    .limit(4)
                    .build(),
            )
            .await?;
        assert_eq!(results.len(), 4, "{query}");
        assert_eq!(results[0].id, expected, "{query}: {results:?}");
        // Cosine MaxSim over 32 query tokens.
        assert!(results[0].score > 0.0 && results[0].score <= 32.0 + 1e-3);
    }
    Ok(())
}
