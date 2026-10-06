# Laurus : Lexically Augmented Unified Retrieval Using Semantics

[![Crates.io](https://img.shields.io/crates/v/laurus.svg)](https://crates.io/crates/laurus)
[![Documentation](https://docs.rs/laurus/badge.svg)](https://docs.rs/laurus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Laurus is a search platform written in Rust — built for Lexically Augmented Unified Retrieval Using Semantics.
Built on a core library covering lexical search, vector search, and hybrid search, it provides multiple ready-to-use interfaces:

- **Core Library** — Modular search engine embeddable into any application
- **CLI & REPL** — Command-line tool for interactive search experiences
- **gRPC Server & HTTP Gateway** — Seamless integration with microservices and existing systems
- **MCP Server** — Direct integration with AI assistants such as Claude
- **Python Bindings** — Native Python package for use in data science and AI workflows
- **Node.js Bindings** — Native Node.js addon for server-side JavaScript applications
- **WebAssembly** — Browser and edge runtime support via wasm-bindgen
- **Ruby Bindings** — Native Ruby gem for Rails and Ruby applications
- **PHP Bindings** — Native PHP extension for web applications and CLI tools

Whether embedded as a library, deployed as a standalone server, called from Python / Node.js / Ruby / PHP, run in the browser via WASM, or woven into AI workflows, Laurus is a composable search foundation.

## Live Demo

Try Laurus directly in your browser — every sample runs entirely client-side via WebAssembly:

**[https://mosuka.github.io/laurus/demo/](https://mosuka.github.io/laurus/demo/)**

| Sample | What it shows |
| --- | --- |
| [basic](https://mosuka.github.io/laurus/demo/basic/) | Japanese full-text, vector, and hybrid search with the unified query DSL |
| [geo](https://mosuka.github.io/laurus/demo/geo/) | Tokyo points-of-interest on a Leaflet map with bounding-box + text + vector queries |
| [geo3d](https://mosuka.github.io/laurus/demo/geo3d/) | Live aircraft on a CesiumJS 3D globe using `geo3d_bbox` / `geo3d_nearest` (true ECEF 3D, including altitude) |

## Documentation

Comprehensive documentation is available online:

- **English**: [https://mosuka.github.io/laurus/](https://mosuka.github.io/laurus/)
- **Japanese (日本語)**: [https://mosuka.github.io/laurus/ja/](https://mosuka.github.io/laurus/ja/)

### Contents

- **Getting Started**
  - [Installation](https://mosuka.github.io/laurus/getting_started/installation.html)
  - [Quick Start](https://mosuka.github.io/laurus/getting_started/quickstart.html)
  - [Examples](https://mosuka.github.io/laurus/getting_started/examples.html)
- **Core Concepts**
  - [Schema & Fields](https://mosuka.github.io/laurus/concepts/schema_and_fields.html)
  - [Text Analysis](https://mosuka.github.io/laurus/concepts/analysis.html)
  - [Embeddings](https://mosuka.github.io/laurus/concepts/embedding.html)
  - [Storage](https://mosuka.github.io/laurus/concepts/storage.html)
  - [Indexing](https://mosuka.github.io/laurus/concepts/indexing.html) (Lexical / Vector)
  - [Search](https://mosuka.github.io/laurus/concepts/search.html) (Lexical / Vector / Hybrid)
  - [Query DSL](https://mosuka.github.io/laurus/concepts/query_dsl.html)
- **Crate Guides**
  - [laurus (Library)](https://mosuka.github.io/laurus/laurus.html) — Engine, Scoring, Faceting, Highlighting, Spelling Correction, Persistence & WAL
  - [laurus-cli](https://mosuka.github.io/laurus/laurus-cli.html) — Command-line interface, REPL, Schema Format
  - [laurus-server](https://mosuka.github.io/laurus/laurus-server.html) — gRPC server, HTTP Gateway, Configuration
  - [laurus-mcp](https://mosuka.github.io/laurus/laurus-mcp.html) — MCP server for AI assistants (Claude, etc.)
  - [laurus-python](https://mosuka.github.io/laurus/laurus-python.html) — Python bindings (PyPI package)
  - [laurus-nodejs](https://mosuka.github.io/laurus/laurus-nodejs.html) — Node.js bindings (npm package)
  - [laurus-wasm](https://mosuka.github.io/laurus/laurus-wasm.html) — WebAssembly bindings (npm package)
  - [laurus-ruby](https://mosuka.github.io/laurus/laurus-ruby.html) — Ruby bindings (RubyGems package)
  - [laurus-php](https://mosuka.github.io/laurus/laurus-php.html) — PHP bindings (PHP extension)
- **Development**
  - [Build & Test](https://mosuka.github.io/laurus/development/build_and_test.html)
  - [Feature Flags](https://mosuka.github.io/laurus/development/feature_flags.html)
  - [Project Structure](https://mosuka.github.io/laurus/development/project_structure.html)
- [**API Reference (docs.rs)**](https://docs.rs/laurus)

## Features

- **Pure Rust Implementation**: Memory-safe and fast performance with zero-cost abstractions.
- **Hybrid Search**: Seamlessly combine BM25 lexical search with HNSW vector search using configurable fusion strategies.
- **Multimodal Capabilities**: Native support for text-to-image and image-to-image search via CLIP embeddings.
- **Rich Query DSL**: Term, phrase, boolean, fuzzy, wildcard, range, 2D / 3D geographic (sphere, bounding box, k-NN), and span queries.
- **Flexible Analysis**: Configurable pipelines for tokenization, normalization, and stemming (including CJK support via [Lindera](https://github.com/lindera/lindera)).
- **Pluggable Storage**: Interfaces for in-memory, file-system, and memory-mapped storage backends.
- **Dynamic Schema**: Optional schema-on-write — undeclared fields are inferred from the value, with `Strict` / `Dynamic` / `Ignore` policies for fine-grained control.
- **Multi-valued Numeric Fields**: Integer and Float fields can hold multiple values per document; range queries match if any value satisfies the predicate (Lucene-style "any match" semantics).
- **Scoring & Ranking**: BM25 scoring with customizable fusion strategies for hybrid results.
- **Faceting & Highlighting**: Built-in support for faceted navigation and search result highlighting.
- **Spelling Correction**: Suggest corrections for misspelled query terms.

## Workspace Structure

Laurus is organized as a Cargo workspace with 9 crates:

| Crate | Description |
| --- | --- |
| [`laurus`](laurus/) | Core search library — schema, analysis, indexing, search, and storage |
| [`laurus-cli`](laurus-cli/) | Command-line interface with REPL for interactive search |
| [`laurus-server`](laurus-server/) | gRPC server with HTTP gateway for deploying Laurus as a service |
| [`laurus-mcp`](laurus-mcp/) | MCP server for AI assistants (Claude, etc.) via stdio transport |
| [`laurus-python`](laurus-python/) | Python bindings (PyPI package) built with PyO3 and Maturin |
| [`laurus-nodejs`](laurus-nodejs/) | Node.js bindings (npm package) built with NAPI-RS |
| [`laurus-wasm`](laurus-wasm/) | WebAssembly bindings (npm package) built with wasm-bindgen |
| [`laurus-ruby`](laurus-ruby/) | Ruby bindings (RubyGems package) built with magnus and rb-sys |
| [`laurus-php`](laurus-php/) | PHP bindings (PHP extension) built with ext-php-rs |

## Feature Flags

The `laurus` crate provides optional feature flags for embedding support:

| Feature | Description |
| --- | --- |
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
    engine
        .add_document(
            "doc2",
            Document::builder()
                .add_text("title", "Python for Data Science")
                .add_text(
                    "body",
                    "Python is a versatile language widely used in data science and machine learning.",
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
        println!("score={:.4}", hit.score);
    }

    Ok(())
}
```

## Examples

You can find usage examples in the [`laurus/examples/`](laurus/examples/) directory:

| Example | Description | Feature Flag |
| --- | --- | --- |
| [quickstart](laurus/examples/quickstart.rs) | Basic full-text search | — |
| [lexical_search](laurus/examples/lexical_search.rs) | All query types (Term, Phrase, Boolean, Fuzzy, Wildcard, Range, Geo, Span) | — |
| [vector_search](laurus/examples/vector_search.rs) | Semantic similarity search with embeddings | — |
| [hybrid_search](laurus/examples/hybrid_search.rs) | Combining lexical and vector search with fusion | — |
| [geo3d_search](laurus/examples/geo3d_search.rs) | 3D ECEF geographic search (sphere, bounding box, k-NN) | — |
| [synonym_graph_filter](laurus/examples/synonym_graph_filter.rs) | Synonym expansion in analysis pipeline | — |
| [search_with_candle](laurus/examples/search_with_candle.rs) | Local BERT embeddings via Candle | `embeddings-candle` |
| [search_with_openai](laurus/examples/search_with_openai.rs) | Cloud-based embeddings via OpenAI | `embeddings-openai` |
| [multimodal_search](laurus/examples/multimodal_search.rs) | Text-to-image and image-to-image search | `embeddings-multimodal` |

## Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
