//! Vector Search Example — semantic similarity search with embeddings.
//!
//! Demonstrates vector search capabilities:
//! - Basic vector search (semantic similarity)
//! - Filtered vector search (with lexical filters)
//! - Vector search via DSL syntax (`field:"query"` / `field:query`)
//!
//! Uses a mock embedder for demonstration. For real embedders, see
//! `search_with_candle` or `search_with_openai`.
//!
//! Run with: `cargo run --example vector_search`

mod common;

use std::sync::Arc;

use laurus::lexical::core::field::{IntegerOption, NumericType};
use laurus::lexical::{NumericRangeQuery, TermQuery, TextOption};
use laurus::vector::{FlatOption, VectorQueryParser};
use laurus::{
    DataValue, Document, Engine, PerFieldEmbedder, QueryPayload, Result, Schema,
    SearchRequestBuilder, VectorSearchQuery,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Laurus Vector Search Example ===\n");

    // ─── Setup ───────────────────────────────────────────────────────────
    let storage = common::memory_storage()?;

    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("text", TextOption::default())
        .add_text_field("category", TextOption::default())
        .add_integer_field("page", IntegerOption::default())
        .add_flat_field("text_vec", FlatOption::default().dimension(4))
        .add_default_field("text")
        .build();

    let analyzer = common::per_field_analyzer(&["title", "category"]);

    let embedder: Arc<dyn laurus::Embedder> = Arc::new(common::MockEmbedder);
    let per_field_embedder = PerFieldEmbedder::new(embedder.clone());
    per_field_embedder.add_embedder("text_vec", embedder.clone());
    let per_field_embedder = Arc::new(per_field_embedder);

    let engine = Engine::builder(storage, schema)
        .analyzer(analyzer)
        .embedder(per_field_embedder.clone())
        .build()
        .await?;

    // ─── Index chunked documents ─────────────────────────────────────────
    println!("--- Indexing chunked documents ---\n");

    let books = json!([
        {
            "id": "book_a",
            "title": "The Rust Programming Language",
            "chunks": [
                { "text": "Chapter 1: Getting Started", "page": 1, "category": "basics" },
                { "text": "Cargo is the Rust build system and package manager. Use cargo new to create a crate.", "page": 2, "category": "basics" },
                { "text": "Every value in Rust has an owner. Ownership rules prevent data races at compile time.", "page": 3, "category": "memory" },
                { "text": "References and borrowing let you use values without taking ownership of them.", "page": 4, "category": "memory" },
                { "text": "Generic types and trait bounds enable polymorphism without runtime overhead.", "page": 5, "category": "type-system" },
                { "text": "Async functions and tokio provide concurrent programming with lightweight tasks and threads.", "page": 6, "category": "concurrency" }
            ]
        },
        {
            "id": "book_b",
            "title": "Programming in Rust",
            "chunks": [
                { "text": "Rust's type system catches many bugs at compile time. Trait objects enable dynamic dispatch.", "page": 1, "category": "type-system" },
                { "text": "The borrow checker ensures memory safety without garbage collection. Lifetime annotations help.", "page": 2, "category": "memory" },
                { "text": "Rust async/await provides zero-cost concurrency for building scalable concurrent network services.", "page": 3, "category": "concurrency" }
            ]
        }
    ]);

    let mut total_chunks = 0;
    for book in books.as_array().unwrap() {
        let id = book["id"].as_str().unwrap();
        let title = book["title"].as_str().unwrap();
        for chunk in book["chunks"].as_array().unwrap() {
            let text = chunk["text"].as_str().unwrap();
            let page = chunk["page"].as_i64().unwrap();
            let category = chunk["category"].as_str().unwrap();
            let doc = Document::builder()
                .add_text("title", title)
                .add_text("text", text)
                .add_text("category", category)
                .add_integer("page", page)
                .add_text("text_vec", text)
                .build();
            engine.add_document(id, doc).await?;
            total_chunks += 1;
        }
    }
    engine.commit().await?;
    println!(
        "Indexed {} books as {} chunks total.\n",
        books.as_array().unwrap().len(),
        total_chunks
    );

    // =====================================================================
    // [A] Basic Vector Search
    // =====================================================================
    println!("{}", "=".repeat(60));
    println!("[A] Basic Vector Search: 'memory safety'");
    println!("{}", "=".repeat(60));
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(VectorSearchQuery::Payloads(vec![QueryPayload::new(
                    "text_vec",
                    DataValue::Text("memory safety".into()),
                )]))
                .limit(3)
                .build(),
        )
        .await?;
    common::print_search_results(&results);

    // =====================================================================
    // [B] Filtered Vector Search — category filter
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("[B] Filtered Vector Search: 'memory safety' + category='concurrency'");
    println!("{}", "=".repeat(60));
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(VectorSearchQuery::Payloads(vec![QueryPayload::new(
                    "text_vec",
                    DataValue::Text("memory safety".into()),
                )]))
                .filter_query(Box::new(TermQuery::new("category", "concurrency")))
                .limit(3)
                .build(),
        )
        .await?;
    common::print_search_results(&results);

    // =====================================================================
    // [C] Filtered Vector Search — numeric range filter
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("[C] Filtered Vector Search: 'type system' + page <= 3");
    println!("{}", "=".repeat(60));
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(VectorSearchQuery::Payloads(vec![QueryPayload::new(
                    "text_vec",
                    DataValue::Text("type system".into()),
                )]))
                .filter_query(Box::new(NumericRangeQuery::new(
                    "page",
                    NumericType::Integer,
                    Some(1.0),
                    Some(3.0),
                    true,
                    true,
                )))
                .limit(3)
                .build(),
        )
        .await?;
    common::print_search_results(&results);

    // =====================================================================
    // [D] Vector Search via DSL
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("[D] Vector DSL: text_vec:\"memory safety\"");
    println!("{}", "=".repeat(60));

    let vector_parser = VectorQueryParser::new(per_field_embedder.clone());

    let parsed = vector_parser.parse("text_vec:\"memory safety\"").await?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(parsed.query)
                .limit(3)
                .build(),
        )
        .await?;
    common::print_search_results(&results);

    println!("\nVector search example completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_search_example() {
        let result = main();
        assert!(result.is_ok(), "vector_search failed: {:?}", result.err());
    }
}
