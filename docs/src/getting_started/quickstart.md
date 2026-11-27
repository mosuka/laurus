# Quick Start

This tutorial walks you through building a complete search engine in 5 steps. By the end, you will be able to index documents and search them by keyword.

## Step 1 — Create Storage

Storage determines where Laurus persists index data. For development and testing, use `MemoryStorage`:

```rust
use std::sync::Arc;
use laurus::storage::memory::MemoryStorage;
use laurus::Storage;

let storage: Arc<dyn Storage> = Arc::new(
    MemoryStorage::new(Default::default())
);
```

> **Tip:** For production, consider `FileStorage` (with optional `use_mmap` for memory-mapped I/O). See [Storage](../concepts/storage.md) for details.

## Step 2 — Define a Schema

A `Schema` declares the fields in your documents and how each field should be indexed:

```rust
use laurus::Schema;
use laurus::lexical::TextOption;

let schema = Schema::builder()
    .add_text_field("title", TextOption::default())
    .add_text_field("body", TextOption::default())
    .add_default_field("body")  // used when no field is specified in a query
    .build();
```

Each field has a type. Common types include:

| Method | Field Type | Example Values |
| :--- | :--- | :--- |
| `add_text_field` | Text (full-text searchable) | `"Hello world"` |
| `add_integer_field` | 64-bit integer | `42` |
| `add_float_field` | 64-bit float | `3.14` |
| `add_boolean_field` | Boolean | `true` / `false` |
| `add_datetime_field` | UTC datetime | `2024-01-15T10:30:00Z` |
| `add_hnsw_field` | Vector (HNSW index) | `[0.1, 0.2, ...]` |
| `add_flat_field` | Vector (Flat index) | `[0.1, 0.2, ...]` |

> See [Schema & Fields](../concepts/schema_and_fields.md) for the full list.

## Step 3 — Build an Engine

The `Engine` ties storage, schema, and runtime components together:

```rust
use laurus::Engine;

let engine = Engine::builder(storage, schema)
    .build()
    .await?;
```

When you only use text fields, the default `StandardAnalyzer` is used automatically. To customize analysis or add vector embeddings, see [Architecture](../architecture.md).

## Step 4 — Index Documents

Create documents with the `DocumentBuilder` and add them to the engine:

```rust
use laurus::Document;

// Each document needs a unique external ID (string)
let doc = Document::builder()
    .add_text("title", "Introduction to Rust")
    .add_text("body", "Rust is a systems programming language focused on safety and performance.")
    .build();
engine.add_document("doc-1", doc).await?;

let doc = Document::builder()
    .add_text("title", "Python for Data Science")
    .add_text("body", "Python is widely used in machine learning and data analysis.")
    .build();
engine.add_document("doc-2", doc).await?;

let doc = Document::builder()
    .add_text("title", "Web Development with JavaScript")
    .add_text("body", "JavaScript powers interactive web applications and server-side code with Node.js.")
    .build();
engine.add_document("doc-3", doc).await?;

// Commit to make documents searchable
engine.commit().await?;
```

> **Important:** Documents are not searchable until `commit()` is called.

## Step 5 — Search

Use `SearchRequestBuilder` with a query to search the index:

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::LexicalSearchQuery;

// Search for "rust" in the "body" field
let request = SearchRequestBuilder::new()
    .lexical_query(
        LexicalSearchQuery::Obj(
            Box::new(TermQuery::new("body", "rust"))
        )
    )
    .limit(10)
    .build();

let results = engine.search(request).await?.results;

for result in &results {
    println!("ID: {}, Score: {:.4}", result.id, result.score);
    if let Some(doc) = &result.document {
        if let Some(title) = doc.get("title") {
            println!("  Title: {:?}", title);
        }
    }
}
```

## Complete Example

Here is the full program that you can copy, paste, and run:

```rust
use std::sync::Arc;
use laurus::{
    Document, Engine,
    Result, Schema, SearchRequestBuilder,
};
use laurus::lexical::{TextOption, TermQuery};
use laurus::lexical::search::searcher::LexicalSearchQuery;
use laurus::storage::memory::MemoryStorage;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Storage
    let storage = Arc::new(MemoryStorage::new(Default::default()));

    // 2. Schema
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .add_default_field("body")
        .build();

    // 3. Engine
    let engine = Engine::builder(storage, schema).build().await?;

    // 4. Index documents
    for (id, title, body) in [
        ("doc-1", "Introduction to Rust", "Rust is a systems programming language focused on safety."),
        ("doc-2", "Python for Data Science", "Python is widely used in machine learning."),
        ("doc-3", "Web Development", "JavaScript powers interactive web applications."),
    ] {
        let doc = Document::builder()
            .add_text("title", title)
            .add_text("body", body)
            .build();
        engine.add_document(id, doc).await?;
    }
    engine.commit().await?;

    // 5. Search
    let request = SearchRequestBuilder::new()
        .lexical_query(
            LexicalSearchQuery::Obj(
                Box::new(TermQuery::new("body", "rust"))
            )
        )
        .limit(10)
        .build();

    let results = engine.search(request).await?.results;
    for r in &results {
        println!("{}: score={:.4}", r.id, r.score);
    }

    Ok(())
}
```

## Next Steps

- Learn how the Engine works internally: [Architecture](../architecture.md)
- Understand Schema and field types: [Schema & Fields](../concepts/schema_and_fields.md)
- Add vector search: [Vector Search](../concepts/search/vector_search.md)
- Combine lexical + vector: [Hybrid Search](../concepts/search/hybrid_search.md)
