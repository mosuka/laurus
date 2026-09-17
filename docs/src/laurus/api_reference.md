# API Reference

This page provides a quick reference of the most important types and methods in Laurus. For full details, generate the Rustdoc:

```bash
cargo doc --open
```

## Engine

The central coordinator for all indexing and search operations.

| Method | Description |
| :--- | :--- |
| `Engine::builder(storage, schema)` | Create an `EngineBuilder` |
| `engine.put_document(id, doc).await?` | Upsert a document (replace if ID exists) |
| `engine.add_document(id, doc).await?` | Add a document as a chunk (multiple chunks can share an ID) |
| `engine.put_documents(docs).await?` | Batched upsert of `(id, doc)` pairs — one WAL fsync per batch (see [Batch Ingestion](persistence.md#batch-ingestion)) |
| `engine.add_documents(docs).await?` | Batched chunk append of `(id, doc)` pairs — same durability and error semantics as `put_documents` |
| `engine.delete_documents(id).await?` | Delete all documents/chunks by external ID |
| `engine.get_documents(id).await?` | Get all documents/chunks by external ID |
| `engine.search(request).await?` | Execute a search request |
| `engine.commit().await?` | Flush all pending changes to storage |
| `engine.flush_wal()?` | Force the WAL durable without a full commit (see [WAL Durability Policy](persistence.md#wal-durability-policy)) |
| `engine.add_field(name, field_option).await?` | Dynamically add a new field to the schema at runtime |
| `engine.delete_field(name).await?` | Remove a field from the schema at runtime |
| `engine.schema()` | Return the current `Schema` |
| `engine.stats()?` | Get index statistics |

> **`put_document` vs `add_document`:** `put_document` performs an upsert — if a document with the same external ID already exists, it is deleted and replaced. `add_document` always appends, allowing multiple document chunks to share the same external ID. See [Schema & Fields — Indexing Documents](../concepts/schema_and_fields.md#indexing-documents) for details.
>
> **Batch forms:** `put_documents` / `add_documents` apply their `(id, doc)` pairs sequentially in input order (duplicate IDs within one `put_documents` batch dedup exactly like sequential puts — the last occurrence wins) and fail fast with `LaurusError::BatchIngest { failed_index, failed_id, applied, .. }` on the first document that cannot be applied. There is no batch rollback: applied documents stay in the WAL and NRT buffers, so retrying the batch (or its suffix) is idempotent. See [Batch Ingestion](persistence.md#batch-ingestion) for the durability details.

### EngineBuilder

| Method | Description |
| :--- | :--- |
| `EngineBuilder::new(storage, schema)` | Create a builder with storage and schema |
| `.analyzer(Arc<dyn Analyzer>)` | Set the text analyzer (default: `StandardAnalyzer`) |
| `.embedder(Arc<dyn Embedder>)` | Set the vector embedder (optional) |
| `.wal_sync_policy(policy)` | Set the WAL durability policy (default: `WalSyncPolicy::PerRecord`; see [WAL Durability Policy](persistence.md#wal-durability-policy)) |
| `.build().await?` | Build the `Engine` |

## Schema

Defines document structure.

| Method | Description |
| :--- | :--- |
| `Schema::builder()` | Create a `SchemaBuilder` |

### SchemaBuilder

| Method | Description |
| :--- | :--- |
| `.add_text_field(name, TextOption)` | Add a full-text field |
| `.add_integer_field(name, IntegerOption)` | Add an integer field (set `IntegerOption::multi_valued = true` for arrays) |
| `.add_float_field(name, FloatOption)` | Add a float field (set `FloatOption::multi_valued = true` for arrays) |
| `.add_boolean_field(name, BooleanOption)` | Add a boolean field |
| `.add_datetime_field(name, DateTimeOption)` | Add a datetime field |
| `.add_geo_field(name, GeoOption)` | Add a 2D geographic (lat/lon) field |
| `.add_geo3d_field(name, Geo3dOption)` | Add a 3D ECEF Cartesian point field (x, y, z in metres) |
| `.add_bytes_field(name, BytesOption)` | Add a binary field |
| `.add_hnsw_field(name, HnswOption)` | Add an HNSW vector field |
| `.add_flat_field(name, FlatOption)` | Add a Flat vector field |
| `.add_ivf_field(name, IvfOption)` | Add an IVF vector field |
| `.add_default_field(name)` | Set a default search field |
| `.add_analyzer(name, AnalyzerDefinition)` | Register a custom analyzer pipeline |
| `.add_embedder(name, EmbedderDefinition)` | Register an embedder definition |
| `.dynamic_field_policy(DynamicFieldPolicy)` | Set the policy for undeclared fields (`Strict` / `Dynamic` / `Ignore`) |
| `.build()` | Build the `Schema` |

## Document

A collection of named field values.

| Method | Description |
| :--- | :--- |
| `Document::builder()` | Create a `DocumentBuilder` |
| `doc.get(name)` | Get a field value by name |
| `doc.has_field(name)` | Check if a field exists |
| `doc.field_names()` | Get all field names |

### DocumentBuilder

| Method | Description |
| :--- | :--- |
| `.add_field(name, DataValue)` | Add an arbitrary `DataValue` |
| `.add_text(name, value)` | Add a text field |
| `.add_integer(name, value)` | Add a single integer field |
| `.add_float(name, value)` | Add a single float field |
| `.add_boolean(name, value)` | Add a boolean field |
| `.add_datetime(name, value)` | Add a datetime field |
| `.add_vector(name, vec)` | Add a pre-computed vector |
| `.add_geo(name, lat, lon)` | Add a 2D geographic point |
| `.add_geo_ecef(name, x, y, z)` | Add a 3D ECEF Cartesian point (metres) |
| `.add_int64_array(name, values)` | Add a multi-valued integer field |
| `.add_float64_array(name, values)` | Add a multi-valued float field |
| `.add_bytes(name, data)` | Add binary data |
| `.build()` | Build the `Document` |

## Search

### SearchRequestBuilder

| Method | Description |
| :--- | :--- |
| `SearchRequestBuilder::new()` | Create a new builder |
| `.query_dsl(dsl)` | Set a unified DSL string (parsed at search time) |
| `.lexical_query(query)` | Set the lexical search query (`LexicalSearchQuery`) |
| `.vector_query(query)` | Set the vector search query (`VectorSearchQuery`) |
| `.filter_query(query)` | Set a pre-filter query |
| `.fusion_algorithm(algo)` | Set the fusion algorithm (default: RRF) |
| `.limit(n)` | Maximum results (default: 10) |
| `.offset(n)` | Skip N results (default: 0) |
| `.add_field_boost(field, boost)` | Add a field-level boost for lexical search |
| `.lexical_min_score(f32)` | Set minimum score threshold for lexical search |
| `.lexical_timeout_ms(u64)` | Set lexical search timeout in milliseconds |
| `.lexical_parallel(bool)` | Enable parallel lexical search |
| `.sort_by(SortField)` | Set sort order for lexical search results |
| `.highlight(fields)` | Request highlights on these stored text fields ([Highlighting](./highlighting.md)) |
| `.highlight_config(HighlightConfig)` | Set the tag/fragment/`require_field_match` settings used for highlighting |
| `.vector_score_mode(VectorScoreMode)` | Set score combination mode for vector search |
| `.vector_min_score(f32)` | Set minimum score threshold for vector search |
| `.build()` | Build the `SearchRequest` |

### LexicalSearchQuery

| Variant | Description |
| :--- | :--- |
| `LexicalSearchQuery::Dsl(String)` | Query specified as a DSL string (parsed at search time) |
| `LexicalSearchQuery::Obj(Box<dyn Query>)` | Query specified as a pre-built Query object |

### VectorSearchQuery

| Variant | Description |
| :--- | :--- |
| `VectorSearchQuery::Payloads(Vec<QueryPayload>)` | Raw payloads (text, bytes, etc.) to be embedded at search time |
| `VectorSearchQuery::Vectors(Vec<QueryVector>)` | Pre-embedded query vectors ready for nearest-neighbor search |

### SearchResult

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `String` | External document ID |
| `score` | `f32` | Relevance score |
| `document` | `Option<Document>` | Document content (if loaded) |
| `highlights` | `HashMap<String, Vec<String>>` | Highlighted fragments per field requested via `.highlight()`; empty when not requested or nothing matched ([Highlighting](./highlighting.md)) |

### FusionAlgorithm

| Variant | Description |
| :--- | :--- |
| `RRF { k: f64 }` | Reciprocal Rank Fusion (default k=60.0) |
| `WeightedSum { lexical_weight, vector_weight }` | Linear combination of scores |

## Query Types (Lexical)

| Query | Description | Example |
| :--- | :--- | :--- |
| `TermQuery::new(field, term)` | Exact term match | `TermQuery::new("body", "rust")` |
| `PhraseQuery::new(field, terms)` | Exact phrase | `PhraseQuery::new("body", vec!["machine".into(), "learning".into()])` |
| `BooleanQueryBuilder::new()` | Boolean combination | `.must(q1).should(q2).must_not(q3).build()` |
| `FuzzyQuery::new(field, term)` | Fuzzy match (default max_edits=2) | `FuzzyQuery::new("body", "programing").max_edits(1)` |
| `WildcardQuery::new(field, pattern)` | Wildcard | `WildcardQuery::new("file", "*.pdf")` |
| `NumericRangeQuery::new(...)` | Numeric range | See [Lexical Search](../concepts/search.md) |
| `GeoDistanceQuery::within_radius(...)` | 2D geo radius | See [Lexical Search](../concepts/search.md) |
| `GeoBoundingBoxQuery::within_bounding_box(...)` | 2D geo bounding box | See [Lexical Search](../concepts/search.md) |
| `Geo3dDistanceQuery::within_sphere(...)` | 3D ECEF sphere | See [3D Geographic Search](../concepts/geo3d.md) |
| `Geo3dBoundingBoxQuery::within_box(...)` | 3D ECEF axis-aligned box | See [3D Geographic Search](../concepts/geo3d.md) |
| `Geo3dNearestQuery::k_nearest(...)` | 3D ECEF k-NN | See [3D Geographic Search](../concepts/geo3d.md) |
| `SpanNearQuery::new(...)` | Proximity | See [Lexical Search](../concepts/search.md) |
| `PrefixQuery::new(field, prefix)` | Prefix match | `PrefixQuery::new("body", "pro")` |
| `RegexpQuery::new(field, pattern)?` | Regex match | `RegexpQuery::new("body", "^pro.*ing$")?` |

## Query Parsers

| Parser | Description |
| :--- | :--- |
| `LexicalQueryParser::new(analyzer)` | Parse lexical DSL queries |
| `VectorQueryParser::new(embedder)` | Parse vector DSL queries |
| `UnifiedQueryParser::new(lexical, vector)` | Parse hybrid DSL queries that mix lexical and vector clauses |

## Analyzers

| Type | Description |
| :--- | :--- |
| `StandardAnalyzer` | RegexTokenizer + lowercase + stop words |
| `SimpleAnalyzer` | Tokenization only (no filtering) |
| `EnglishAnalyzer` | RegexTokenizer + lowercase + English stop words |
| `JapaneseAnalyzer` | Japanese morphological analysis |
| `KeywordAnalyzer` | No tokenization (exact match) |
| `PipelineAnalyzer` | Custom tokenizer + filter chain |
| `PerFieldAnalyzer` | Per-field analyzer dispatch |

## Embedders

| Type | Feature Flag | Description |
| :--- | :--- | :--- |
| `CandleBertEmbedder` | `embeddings-candle` | Local BERT model |
| `OpenAIEmbedder` | `embeddings-openai` | OpenAI API |
| `CandleClipEmbedder` | `embeddings-multimodal` | Local CLIP model |
| `PrecomputedEmbedder` | *(default)* | Pre-computed vectors |
| `PerFieldEmbedder` | *(default)* | Per-field embedder dispatch |

## Storage

| Type | Description |
| :--- | :--- |
| `MemoryStorage` | In-memory (non-durable) |
| `FileStorage` | File-system based (supports `use_mmap` for memory-mapped I/O) |
| `StorageFactory::create(config)` | Create from config |

## DataValue

| Variant | Rust Type |
| :--- | :--- |
| `DataValue::Null` | — |
| `DataValue::Bool(bool)` | `bool` |
| `DataValue::Int64(i64)` | `i64` |
| `DataValue::Float64(f64)` | `f64` |
| `DataValue::Text(String)` | `String` |
| `DataValue::Bytes(Vec<u8>, Option<String>)` | `(data, mime_type)` |
| `DataValue::Vector(Vec<f32>)` | Pre-computed vector |
| `DataValue::DateTime(DateTime<Utc>)` | `chrono::DateTime<Utc>` |
| `DataValue::Geo(GeoPoint)` | `(latitude, longitude)` (WGS84) |
| `DataValue::GeoEcef(GeoEcefPoint)` | `(x, y, z)` ECEF Cartesian (metres) |
| `DataValue::Int64Array(Vec<i64>)` | Multi-valued integers (requires `multi_valued` field option) |
| `DataValue::Float64Array(Vec<f64>)` | Multi-valued floats (requires `multi_valued` field option) |
