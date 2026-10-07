# laurus

[![Crates.io](https://img.shields.io/crates/v/laurus.svg)](https://crates.io/crates/laurus)
[![Documentation](https://docs.rs/laurus/badge.svg)](https://docs.rs/laurus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Core search engine library for the [Laurus](https://github.com/mosuka/laurus) project. Provides lexical search (keyword matching via inverted index), vector search (semantic similarity via embeddings), and hybrid search (combining both) through a unified API.

## Features

- **Lexical Search** -- Full-text search powered by an inverted index with BM25 scoring
- **Vector Search** -- Approximate nearest neighbor (ANN) search using Flat, HNSW, or IVF indexes
- **Hybrid Search** -- Combine lexical and vector results with fusion algorithms (RRF, WeightedSum)
- **Text Analysis** -- Pluggable analyzer pipeline: tokenizers, filters, stemmers, synonyms (including CJK support via [Lindera](https://github.com/lindera/lindera))
- **Embeddings** -- Built-in support for Candle (local BERT/CLIP), OpenAI API, or custom embedders
- **Pluggable Storage** -- In-memory, file-based, or memory-mapped backends
- **Faceting & Highlighting** -- Faceted navigation and search result highlighting
- **Spelling Correction** -- Suggest corrections for misspelled query terms
- **Write-Ahead Log** -- Durability via WAL with automatic recovery on restart

## Installation

```toml
# Lexical search only (no embedding)
[dependencies]
laurus = "0.13"

# With local BERT embeddings
[dependencies]
laurus = { version = "0.13", features = ["embeddings-candle"] }

# All embedding backends
[dependencies]
laurus = { version = "0.13", features = ["embeddings-all"] }
```

## Feature Flags

| Feature | Description |
| :--- | :--- |
| `embeddings-candle` | Local BERT embeddings via [Candle](https://github.com/huggingface/candle) |
| `embeddings-openai` | Cloud-based embeddings via the OpenAI API |
| `embeddings-multimodal` | CLIP-based multimodal (text + image) embeddings |
| `embeddings-all` | Enable all embedding backends |

## Quick Start

```rust
use laurus::lexical::{TermQuery, TextOption};
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{Document, Engine, LexicalSearchRequest, Schema, SearchRequestBuilder};

#[tokio::main]
async fn main() -> laurus::Result<()> {
    // 1. Create storage
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;

    // 2. Define schema
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .build();

    // 3. Create engine
    let engine = Engine::new(storage, schema).await?;

    // 4. Index documents
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
    engine.commit().await?;

    // 5. Search
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_search_request(LexicalSearchRequest::new(Box::new(TermQuery::new(
                    "body", "rust",
                ))))
                .limit(5)
                .build(),
        )
        .await?;

    for hit in &results {
        println!("[{}] score={:.4}", hit.id, hit.score);
    }

    Ok(())
}
```

## Key Types

| Type | Module | Description |
| :--- | :--- | :--- |
| `Engine` | `engine` | Unified search engine coordinating lexical and vector search |
| `Schema` | `engine` | Field definitions and routing configuration |
| `Document` | `data` | Collection of named field values |
| `SearchRequestBuilder` | `engine` | Builder for unified search requests (lexical, vector, or hybrid) |
| `FusionAlgorithm` | `engine` | Result merging strategy (RRF or WeightedSum) |
| `LaurusError` | `error` | Comprehensive error type with variants for each subsystem |

## Examples

Usage examples are in the [`examples/`](examples/) directory:

| Example | Description | Feature Flag |
| :--- | :--- | :--- |
| [quickstart](examples/quickstart.rs) | Basic full-text search | -- |
| [lexical_search](examples/lexical_search.rs) | All query types (Term, Phrase, Boolean, Fuzzy, Wildcard, Range, Geo, Span) | -- |
| [vector_search](examples/vector_search.rs) | Semantic similarity search with embeddings | -- |
| [hybrid_search](examples/hybrid_search.rs) | Combining lexical and vector search with fusion | -- |
| [synonym_graph_filter](examples/synonym_graph_filter.rs) | Synonym expansion in analysis pipeline | -- |
| [search_with_candle](examples/search_with_candle.rs) | Local BERT embeddings via Candle | `embeddings-candle` |
| [search_with_openai](examples/search_with_openai.rs) | Cloud-based embeddings via OpenAI | `embeddings-openai` |
| [multimodal_search](examples/multimodal_search.rs) | Text-to-image and image-to-image search | `embeddings-multimodal` |

## Documentation

- [Library Guide](https://mosuka.github.io/laurus/laurus.html)
- [API Reference (docs.rs)](https://docs.rs/laurus)
- [Architecture](https://mosuka.github.io/laurus/architecture.html)
- [Schema & Fields](https://mosuka.github.io/laurus/concepts/schema_and_fields.html)
- [Text Analysis](https://mosuka.github.io/laurus/concepts/analysis.html)
- [Search](https://mosuka.github.io/laurus/concepts/search.html)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
