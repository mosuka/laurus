//! Quickstart — Your first full-text search with Laurus
//!
//! This minimal example shows how to:
//! 1. Create an in-memory search engine
//! 2. Define a schema with text fields
//! 3. Index documents
//! 4. Search with a simple term query
//!
//! Run with: `cargo run --example quickstart`

mod common;

use laurus::lexical::{TermQuery, TextOption};
use laurus::{Document, Engine, LexicalSearchQuery, Schema, SearchRequestBuilder};

#[tokio::main]
async fn main() -> laurus::Result<()> {
    println!("=== Laurus Quickstart ===\n");

    // 1. Create storage and schema
    let storage = common::memory_storage()?;
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .build();

    // 2. Create engine (no embedder needed for lexical-only search)
    let engine = Engine::new(storage, schema).await?;

    // 3. Index documents
    engine
        .add_document(
            "doc1",
            Document::builder()
                .add_text("title", "Introduction to Rust")
                .add_text(
                    "body",
                    "Rust is a systems programming language focused on safety and performance.",
                )
                .build(),
        )
        .await?;

    engine
        .add_document(
            "doc2",
            Document::builder()
                .add_text("title", "Python for Data Science")
                .add_text("body", "Python is a versatile language widely used in data science and machine learning.")
                .build(),
        )
        .await?;

    engine
        .add_document(
            "doc3",
            Document::builder()
                .add_text("title", "Web Development with JavaScript")
                .add_text("body", "JavaScript powers the modern web, from frontend frameworks to backend services.")
                .build(),
        )
        .await?;

    engine.commit().await?;
    println!("Indexed 3 documents.\n");

    // 4. Search for "rust"
    println!("[Search] term 'rust' in body:");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "body", "rust",
                ))))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // 5. Search for "language"
    println!("\n[Search] term 'language' in body:");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "body", "language",
                ))))
                .limit(5)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    Ok(())
}
