//! Search with OpenAI Embedder — real vector search using OpenAI's API.
//!
//! This example mirrors `search.rs` but replaces MockEmbedder with
//! `OpenAIEmbedder` (text-embedding-3-small, 1536 dim).
//!
//! Prerequisites:
//! - Set `OPENAI_API_KEY` environment variable
//! - Ensure you have API credits
//!
//! Run with:
//! ```bash
//! export OPENAI_API_KEY=your-api-key-here
//! cargo run --example search_with_openai --features embeddings-openai
//! ```

#[cfg(feature = "embeddings-openai")]
mod common;

#[cfg(feature = "embeddings-openai")]
use std::sync::Arc;

#[cfg(feature = "embeddings-openai")]
use laurus::lexical::core::field::{IntegerOption, NumericType};
#[cfg(feature = "embeddings-openai")]
use laurus::lexical::{LexicalQueryParser, NumericRangeQuery, TermQuery, TextOption};
#[cfg(feature = "embeddings-openai")]
use laurus::vector::{FlatOption, VectorQueryParser};
#[cfg(feature = "embeddings-openai")]
use laurus::{
    DataValue, Document, Engine, FusionAlgorithm, LexicalSearchQuery, OpenAIEmbedder,
    PerFieldEmbedder, QueryPayload, Result, Schema, SearchRequestBuilder, UnifiedQueryParser,
    VectorSearchQuery,
};
#[cfg(feature = "embeddings-openai")]
use serde_json::json;

#[cfg(feature = "embeddings-openai")]
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Laurus Search with OpenAI Embedder ===\n");

    // ─── Setup ───────────────────────────────────────────────────────────
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: OPENAI_API_KEY environment variable not set");
        eprintln!("Please set it with: export OPENAI_API_KEY=your-api-key-here");
        std::process::exit(1);
    });

    println!("Creating OpenAI embedder (text-embedding-3-small, 1536 dim) ...");
    let openai_embedder =
        OpenAIEmbedder::new(api_key, "text-embedding-3-small".to_string()).await?;
    println!("Embedder ready!\n");

    let storage = common::memory_storage()?;

    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("text", TextOption::default())
        .add_text_field("category", TextOption::default())
        .add_integer_field("page", IntegerOption::default())
        .add_flat_field("text_vec", FlatOption::default().dimension(1536))
        .add_default_field("text")
        .build();

    let analyzer = common::per_field_analyzer(&["title", "category"]);

    let embedder: Arc<dyn laurus::Embedder> = Arc::new(openai_embedder);
    let per_field_embedder = PerFieldEmbedder::new(embedder.clone());
    per_field_embedder.add_embedder("text_vec", embedder.clone());
    let per_field_embedder = Arc::new(per_field_embedder);

    let std_analyzer = Arc::new(laurus::analysis::analyzer::standard::StandardAnalyzer::default());

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
    // PART 1: Builder API
    // =====================================================================
    println!("{}", "=".repeat(60));
    println!("PART 1: Builder API (programmatic query construction)");
    println!("{}", "=".repeat(60));

    // Case A: Vector Search
    println!("\n[A] Vector Search: 'memory safety'");
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
        .await?
        .results;
    common::print_search_results(&results);

    // Case B: Filtered Vector Search (category filter)
    println!("\n[B] Filtered Vector Search: 'memory safety' + category='concurrency'");
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
        .await?
        .results;
    common::print_search_results(&results);

    // Case C: Filtered Vector Search (numeric range)
    println!("\n[C] Filtered Vector Search: 'type system' + page <= 3");
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
        .await?
        .results;
    common::print_search_results(&results);

    // Case D: Lexical Search
    println!("\n[D] Lexical Search: 'ownership'");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "text",
                    "ownership",
                ))))
                .limit(3)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // Case E: Hybrid Search (RRF Fusion)
    println!("\n[E] Hybrid Search (RRF): vector='concurrent' + lexical='async'");
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(VectorSearchQuery::Payloads(vec![QueryPayload::new(
                    "text_vec",
                    DataValue::Text("concurrent".into()),
                )]))
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "text", "async",
                ))))
                .fusion_algorithm(FusionAlgorithm::RRF { k: 60.0 })
                .limit(3)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // =====================================================================
    // PART 2: DSL (text-based query syntax)
    // =====================================================================
    println!("\n{}", "=".repeat(60));
    println!("PART 2: DSL (text-based query syntax)");
    println!("{}", "=".repeat(60));

    let vector_parser = VectorQueryParser::new(per_field_embedder.clone());
    let unified_parser = UnifiedQueryParser::new(
        LexicalQueryParser::new(std_analyzer).with_default_field("text"),
        VectorQueryParser::new(per_field_embedder.clone()),
        ["text_vec".to_string()].into_iter().collect(),
    );

    // Case F: Vector Search via DSL
    println!("\n[F] Vector DSL: text_vec:\"memory safety\"");
    let parsed = vector_parser.parse("text_vec:\"memory safety\"").await?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(parsed.query)
                .limit(3)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // Case G: Hybrid Search via DSL
    println!("\n[G] Hybrid DSL: text:async text_vec:\"concurrent\"");
    let mut request = unified_parser
        .parse("text:async text_vec:\"concurrent\"")
        .await?;
    request.limit = 3;
    let results = engine.search(request).await?.results;
    common::print_search_results(&results);

    // Case H: Lexical-only via DSL
    println!("\n[H] Lexical DSL: text:ownership");
    let mut request = unified_parser.parse("text:ownership").await?;
    request.limit = 3;
    let results = engine.search(request).await?.results;
    common::print_search_results(&results);

    // Case I: Vector-only via DSL
    println!("\n[I] Vector DSL: text_vec:\"type system\"");
    let mut request = unified_parser.parse("text_vec:\"type system\"").await?;
    request.limit = 3;
    let results = engine.search(request).await?.results;
    common::print_search_results(&results);

    println!("\nSearch with OpenAI example completed successfully!");

    Ok(())
}

#[cfg(not(feature = "embeddings-openai"))]
fn main() {
    eprintln!("This example requires the 'embeddings-openai' feature.");
    eprintln!("Run with: cargo run --example search_with_openai --features embeddings-openai");
    std::process::exit(1);
}
