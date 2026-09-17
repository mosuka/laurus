# Engine

The `Engine` is the central type in Laurus. It coordinates the lexical index, vector index, and document log behind a single async API.

## Engine Struct

```rust
pub struct Engine {
    schema: Schema,
    lexical: LexicalStore,
    vector: VectorStore,
    log: Arc<DocumentLog>,
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `schema` | `Schema` | Field definitions and routing rules |
| `lexical` | `LexicalStore` | Inverted index for keyword search |
| `vector` | `VectorStore` | Vector index for similarity search |
| `log` | `Arc<DocumentLog>` | Write-ahead log for crash recovery and document storage |

## EngineBuilder

Use `EngineBuilder` to configure and create an Engine:

```rust
use std::sync::Arc;
use laurus::{Engine, Schema};
use laurus::lexical::TextOption;
use laurus::storage::memory::MemoryStorage;

let storage = Arc::new(MemoryStorage::new(Default::default()));
let schema = Schema::builder()
    .add_text_field("title", TextOption::default())
    .add_text_field("body", TextOption::default())
    .add_default_field("body")
    .build();

let engine = Engine::builder(storage, schema)
    .analyzer(my_analyzer)            // optional: custom text analyzer
    .embedder(my_embedder)            // optional: vector embedder
    .embedding_cache_capacity(1024)   // optional: cache query embeddings
    .build()
    .await?;
```

### Builder Methods

| Method | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| `analyzer()` | `Arc<dyn Analyzer>` | `StandardAnalyzer` | Text analysis pipeline for lexical fields |
| `embedder()` | `Arc<dyn Embedder>` | None | Embedding model for vector fields |
| `embedding_cache_capacity()` | `usize` | None (disabled) | Enable an LRU cache of up to N query embeddings |
| `wal_sync_policy()` | `WalSyncPolicy` | `PerRecord` | WAL fsync durability (per-record vs group commit) |
| `commit_policy()` | `CommitPolicy` | `Manual` | Auto-commit cadence — e.g. `EveryDocs(n)` commits every `n` documents |
| `build()` | -- | -- | Create the Engine (async) |

### Query embedding cache

`embedding_cache_capacity(n)` enables an LRU cache for **query-time**
embeddings (document-ingestion embedding is unaffected). When a vector or
hybrid search embeds a query payload, the result is cached keyed by
`(field, embedder name, payload hash)` and reused on subsequent identical
queries — both the DSL path (e.g. `content:"cute kitten"`) and the
pre-embedded `Payloads` path share the same cache.

This avoids repeated model inference for local embedders, or network round
trips for remote ones, on repeated-query workloads (autocomplete, dashboard
refreshes, A/B evaluation). It is disabled by default; pick a capacity that
bounds memory to your working set of distinct queries.

Within a single query, all of its payloads are embedded in one
`Embedder::embed_batch` call (cache misses only, grouped per field for
`PerFieldEmbedder`), so a batch-capable embedder pays one round trip for a
multi-vector query instead of one per payload (Issue #671).

### Build Lifecycle

When `build()` is called, the following steps occur:

```mermaid
sequenceDiagram
    participant User
    participant EngineBuilder
    participant Engine

    User->>EngineBuilder: .build().await
    EngineBuilder->>EngineBuilder: split_schema()
    Note over EngineBuilder: Separate fields into<br/>LexicalIndexConfig<br/>+ VectorIndexConfig
    EngineBuilder->>Engine: Create PrefixedStorage (lexical/, vector/, documents/)
    EngineBuilder->>Engine: Create LexicalStore
    EngineBuilder->>Engine: Create VectorStore
    EngineBuilder->>Engine: Create DocumentLog
    EngineBuilder->>Engine: Recover from WAL
    EngineBuilder-->>User: Engine ready
```

1. **Split schema** -- Lexical fields (Text, Integer, Float, etc.) go to `LexicalIndexConfig`, vector fields (HNSW, Flat, IVF) go to `VectorIndexConfig`
2. **Create prefixed storage** -- Each component gets an isolated namespace (`lexical/`, `vector/`, `documents/`)
3. **Initialize stores** -- `LexicalStore` and `VectorStore` are created with their respective configs
4. **Recover from WAL** -- Any uncommitted operations from a previous session are replayed

## Schema Splitting

The Schema contains both lexical and vector fields. At build time, `split_schema()` separates them:

```mermaid
graph LR
    S["Schema<br/>title: Text<br/>body: Text<br/>page: Integer<br/>content_vec: HNSW"]

    S --> LC["LexicalIndexConfig<br/>title: TextOption<br/>body: TextOption<br/>page: IntegerOption<br/>_id: KeywordAnalyzer"]

    S --> VC["VectorIndexConfig<br/>content_vec: HnswOption<br/>(dim=384, m=16, ef=200)"]
```

The reserved `_id` field is always added to the lexical config with `KeywordAnalyzer` for exact match lookups.

## Per-Field Dispatch

### PerFieldAnalyzer

When a `PerFieldAnalyzer` is provided, text analysis is dispatched to field-specific analyzers:

```mermaid
graph LR
    PFA["PerFieldAnalyzer"]
    PFA -->|"title"| KA["KeywordAnalyzer"]
    PFA -->|"body"| SA["StandardAnalyzer"]
    PFA -->|"description"| JA["JapaneseAnalyzer"]
    PFA -->|"_id"| KA2["KeywordAnalyzer<br/>(always)"]
    PFA -->|other fields| DEF["Default Analyzer"]
```

### PerFieldEmbedder

Similarly, `PerFieldEmbedder` routes embedding to field-specific embedders:

```mermaid
graph LR
    PFE["PerFieldEmbedder"]
    PFE -->|"text_vec"| BERT["CandleBertEmbedder<br/>(384 dim)"]
    PFE -->|"image_vec"| CLIP["CandleClipEmbedder<br/>(512 dim)"]
    PFE -->|other fields| DEF["Default Embedder"]
```

## Engine Methods

### Document Operations

| Method | Description |
| :--- | :--- |
| `put_document(id, doc)` | Upsert -- replaces any existing document with the same ID |
| `add_document(id, doc)` | Append -- adds as a new chunk (multiple chunks can share an ID) |
| `put_documents(docs)` | Batched upsert -- applies `(id, doc)` pairs in order with one WAL fsync per batch |
| `add_documents(docs)` | Batched append -- like `put_documents` but never deletes existing chunks |
| `get_documents(id)` | Retrieve all documents/chunks by external ID |
| `delete_documents(id)` | Delete all documents/chunks by external ID |
| `commit()` | Flush pending changes to storage (makes documents searchable) |
| `recover()` | Replay WAL to restore uncommitted state after crash |
| `add_field(name, field_option)` | Dynamically add a new field to the schema at runtime |
| `delete_field(name)` | Remove a field from the schema at runtime |
| `schema()` | Return the current `Schema` |

### Search

| Method | Description |
| :--- | :--- |
| `search(request)` | Execute a unified search (lexical, vector, or hybrid) |

The `search()` method accepts a `SearchRequest` which can contain a lexical query, a vector query, or both. When both are present, results are merged using the specified `FusionAlgorithm`.

```rust
use laurus::{SearchRequestBuilder, FusionAlgorithm};
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::LexicalSearchQuery;

// Lexical-only search
let request = SearchRequestBuilder::new()
    .lexical_query(
        LexicalSearchQuery::Obj(Box::new(TermQuery::new("body", "rust")))
    )
    .limit(10)
    .build();

// Hybrid search with RRF fusion
let request = SearchRequestBuilder::new()
    .lexical_query(lexical_query)
    .vector_query(vector_query)
    .fusion_algorithm(FusionAlgorithm::RRF { k: 60.0 })
    .limit(10)
    .build();

let results = engine.search(request).await?;
```

### Warmup

`engine.warmup()` (Issue #677) primes the vector searcher so the **first**
vector / hybrid query does not pay one-time setup costs. Call it once after
building the engine, before serving traffic:

```rust,ignore
let engine = builder.build()?;
engine.warmup()?;   // optional: move first-query latency to startup
```

It eagerly builds and caches the vector searcher — loading the reader (file →
memory for `InMemory`, the offset table for `Mmap`) — and, for HNSW indexes in
`Mmap` mode, pre-faults the on-disk vector data into the OS page cache. The HNSW
graph is always loaded eagerly, so only the vector data needs warming.
`warmup()` is safe to call multiple times and is a no-op for lexical-only
workloads.

## SearchRequest

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `query` | `SearchQuery` | `Dsl("")` | Search query specification (Dsl, Lexical, Vector, or Hybrid) |
| `limit` | `usize` | 10 | Maximum results to return |
| `offset` | `usize` | 0 | Pagination offset |
| `fusion_algorithm` | `Option<FusionAlgorithm>` | RRF (k=60) | How to merge lexical + vector results |
| `filter_query` | `Option<Box<dyn Query>>` | None | Filter applied to both search types |
| `lexical_options` | `LexicalSearchOptions` | Default | Parameters controlling lexical search behavior |
| `vector_options` | `VectorSearchOptions` | Default | Parameters controlling vector search behavior |

### LexicalSearchOptions

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `field_boosts` | `HashMap<String, f32>` | Empty | Per-field score multipliers |
| `min_score` | `f32` | 0.0 | Minimum score threshold |
| `timeout_ms` | `Option<u64>` | None | Search timeout in milliseconds |
| `parallel` | `bool` | false | Run segment search in parallel |
| `sort_by` | `SortField` | `Score` | Sort order for lexical results |
| `highlight` | `Option<HighlightOptions>` | None | Fields and configuration for search-result highlighting; `None` leaves every hit's `SearchResult::highlights` empty. Set via `SearchRequestBuilder::highlight` / `.highlight_config`, see [Highlighting](./highlighting.md). Ignored by vector-only requests. |

## FusionAlgorithm

| Variant | Description |
| :--- | :--- |
| `RRF { k: f64 }` | Reciprocal Rank Fusion -- rank-based combining. Score = sum(1 / (k + rank)). Handles incomparable score magnitudes. |
| `WeightedSum { lexical_weight, vector_weight }` | Weighted combination with min-max score normalization. Weights clamped to [0.0, 1.0]. |

See also: [Architecture](../architecture.md) for the high-level data flow diagrams.
