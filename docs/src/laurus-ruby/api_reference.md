# API Reference

## Index

The primary entry point. Wraps the Laurus search engine.

```ruby
Laurus::Index.new(path: nil, schema: nil, wal_sync_policy: nil, commit_policy: nil)
```

### Constructor

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path:` | `String \| nil` | `nil` | Directory path for persistent storage. `nil` creates an in-memory index. When given, the directory follows the `<path>/schema.toml` + `<path>/store/` layout `laurus-cli create index`/`--index-dir` uses, so an index built here can also be opened by the CLI (and vice versa). See below. |
| `schema:` | `Schema \| nil` | `nil` | Schema definition. Only meaningful when *creating* a new file-backed index (or an in-memory one); must be omitted when reopening an existing file-backed index -- the persisted schema is loaded instead. An empty schema is used when omitted for a new index. |
| `wal_sync_policy:` | `WalSyncPolicy \| nil` | `nil` | Write-ahead log (WAL) durability policy. `nil` keeps the default per-record fsync. See [WAL sync policy & durability](#wal-sync-policy--durability). |
| `commit_policy:` | `CommitPolicy \| nil` | `nil` | Auto-commit policy. `nil` keeps the default manual mode (the caller drives every `commit`). See [Commit policy & auto-commit](#commit-policy--auto-commit). |

**Creating vs. reopening a file-backed index** (`path:` given): if `<path>/schema.toml` does not yet exist, this call **creates** a new index and persists `schema:` (or an empty schema, if omitted) to it. If `<path>/schema.toml` already exists, this call **reopens** the index -- `schema:` must be omitted, or an `ArgumentError` is raised (it would be ambiguous which schema should win). An `ArgumentError` is also raised if `path:` contains an index in the layout used before this convention was introduced (segment files directly under `path:`, no `schema.toml`).

### Methods

| Method | Description |
| :--- | :--- |
| `put_document(id, doc)` | Upsert a document. Replaces all existing versions with the same ID. |
| `add_document(id, doc)` | Append a document chunk without removing existing versions. |
| `put_documents(docs)` | Batched upsert. `docs` is an `Array` of `[id, hash]` pairs, applied in order with one WAL fsync per batch (duplicate ids dedup, last wins). Fails fast at the first bad entry; the applied prefix is not rolled back. |
| `add_documents(docs)` | Batched chunk append. Like `put_documents` but repeated ids accumulate as separate versions. |
| `get_documents(id) -> Array<Hash>` | Return all stored versions for the given ID. |
| `delete_documents(id)` | Delete all versions for the given ID. |
| `commit` | Flush buffered writes and make all pending changes searchable. |
| `flush_wal` | Force a durable WAL barrier on demand. Synchronously fsyncs any unsynced WAL records and returns `nil`. Useful when running under a group-commit policy (see below). |
| `search(query, limit: 10, offset: 0) -> Array<SearchResult>` | Execute a search query. |
| `search_batch(queries, limit: 10, offset: 0) -> Array<Array<SearchResult>>` | Execute multiple independent searches in one call. Each query is dispatched in parallel on the underlying tokio runtime. `results[i]` corresponds to `queries[i]`. Empty input returns `[]`. |
| `stats -> Hash` | Return index statistics (`"document_count"`, `"vector_fields"`). |

### `search` query argument

The `query` parameter accepts any of the following:

- A **DSL string** (e.g. `"title:hello"`, `"embedding:\"memory safety\""`)
- A **lexical query object** (`TermQuery`, `PhraseQuery`, `BooleanQuery`, ...)
- A **vector query object** (`VectorQuery`, `VectorTextQuery`)
- A **`SearchRequest`** for full control

The same value kinds are accepted as the elements of `search_batch`'s `queries` Array — DSL strings, query objects, and `SearchRequest` instances may be mixed within a single batch.

### WAL sync policy & durability

The write-ahead log (WAL) protects committed data against crashes. By
default the WAL is fully durable: every record is `fsync`ed before the write
returns. You can trade some durability for higher write throughput by opting
into **group commit**, which batches `fsync` calls.

#### WalSyncPolicy

`Laurus::WalSyncPolicy` is an immutable value object describing how the WAL
is flushed. Pass it to `Index.new(wal_sync_policy:)`.

```ruby
# Default: durable per write (each record is fsynced individually).
Laurus::WalSyncPolicy.per_record

# Group commit: batch fsyncs to amortise their cost.
Laurus::WalSyncPolicy.group(
  max_records: nil,      # flush after this many records (default 1024)
  max_bytes: nil,        # flush after this many bytes (default 1 MiB)
  max_interval_ms: nil,  # also flush periodically after this many ms
)
```

| Constructor | Description |
| :--- | :--- |
| `WalSyncPolicy.per_record` | Default. Every record is `fsync`ed before the write returns — fully durable per write. |
| `WalSyncPolicy.group(max_records:, max_bytes:, max_interval_ms:)` | Batch `fsync`s. With no arguments uses the defaults (`max_records: 1024`, `max_bytes: 1 MiB`, no timer). The WAL is flushed when **either** `max_records` **or** `max_bytes` accumulate, and on every `commit`. Pass `max_interval_ms:` to also flush on a periodic timer. |

Group commit is analogous to SQLite's `synchronous = NORMAL`: a crash can lose
at most the last unsynced batch of records, but the index never corrupts.
Records are always made durable at `commit`, so a successful `commit` is a
durability barrier regardless of policy.

#### Forcing a flush

Call `flush_wal` to force a durable barrier between commits — for example
before signalling that a batch has been safely persisted. It synchronously
`fsync`s any unsynced records and returns `nil`. Under the default
per-record policy it is effectively a no-op.

```ruby
# Opt into group commit, then force durability on demand.
policy = Laurus::WalSyncPolicy.group(max_records: 4096, max_bytes: 4 * 1024 * 1024)
index = Laurus::Index.new(path: "./myindex", wal_sync_policy: policy)

index.put_document("doc1", { "title" => "Hello" })
index.flush_wal  # records persisted even though the group batch is not full
```

### Commit policy & auto-commit

By default the caller drives every commit: buffered writes only become
searchable once you call `commit` explicitly. You can hand that responsibility
to the engine with an **auto-commit policy**, which commits automatically after
a fixed number of applied documents or on a periodic timer.

#### CommitPolicy

`Laurus::CommitPolicy` is an immutable value object describing when the engine
materialises buffered writes into the stores. Pass it to
`Index.new(commit_policy:)`.

```ruby
# Default: manual — the caller drives every commit.
Laurus::CommitPolicy.manual

# Auto-commit: commit after every N applied documents.
Laurus::CommitPolicy.every_docs(1000)

# Auto-commit: commit at least every N milliseconds (native only).
Laurus::CommitPolicy.interval_ms(5000)
```

| Constructor | Description |
| :--- | :--- |
| `CommitPolicy.manual` | Default. No auto-commit — the caller drives every `commit`. |
| `CommitPolicy.every_docs(n)` | Auto-commit after every `n` applied documents. Counted across both singular and batch ingest, including every `n` documents **within** a single batch. |
| `CommitPolicy.interval_ms(ms)` | Auto-commit at least every `ms` milliseconds via a background timer, so a trailing partial batch is committed even while ingestion is idle. The time-based counterpart of `every_docs`. Default: none. **Native only** — under WebAssembly (`wasm32`) there are no background threads, so the engine treats it as a no-op (the value still constructs, but no timed commit happens). |

`every_docs(0)` is valid and disables auto-commit, making it equivalent to
`manual`.

Commit policy is orthogonal to [WalSyncPolicy](#wal-sync-policy--durability):
`WalSyncPolicy` governs WAL `fsync` durability, whereas `CommitPolicy` governs
when the stores materialise buffered writes. Set them independently.

```ruby
# Auto-commit after every 1000 applied documents.
policy = Laurus::CommitPolicy.every_docs(1000)
index = Laurus::Index.new(path: "./myindex", commit_policy: policy)
```

---

## Schema

Defines the fields and index types for an `Index`.

```ruby
Laurus::Schema.new
```

### Field methods

| Method | Description |
| :--- | :--- |
| `add_text_field(name, stored: true, indexed: true, term_vectors: true, doc_values: true, analyzer: nil)` | Full-text field (inverted index, BM25). `term_vectors:` controls whether term positions are stored, read by phrase and span queries. `doc_values:` controls whether the value is also copied into DocValues, the column-oriented store sort/facet/aggregation read from (Issue #1047); takes effect only when `stored: true`. `analyzer:` is the name of a parameter-less built-in (`"standard"`, `"english"`, `"keyword"`, `"simple"`, `"noop"`) or a custom name registered via `add_analyzer`. The Japanese preset requires a Lindera dictionary path, so register it as a custom analyzer with a `lindera` tokenizer and reference it by name. |
| `add_integer_field(name, stored: true, indexed: true, multi_valued: false, doc_values: true)` | 64-bit integer field. Pass `multi_valued: true` to accept arrays of integers (range queries match if any value satisfies the predicate). See `doc_values:` above. |
| `add_float_field(name, stored: true, indexed: true, multi_valued: false, doc_values: true)` | 64-bit float field. Pass `multi_valued: true` to accept arrays of floats (range queries match if any value satisfies the predicate). See `doc_values:` above. |
| `add_boolean_field(name, stored: true, indexed: true, doc_values: true)` | Boolean field. See `doc_values:` above. |
| `add_bytes_field(name, stored: true)` | Raw bytes field. No `doc_values:` option: a `Bytes` value is never written to DocValues regardless. |
| `add_geo_field(name, stored: true, indexed: true, doc_values: true)` | Geographic coordinate field (lat/lon). See `doc_values:` above. |
| `add_geo3d_field(name, stored: true, indexed: true, doc_values: true)` | 3D ECEF Cartesian point field (x, y, z in metres). See [Geo3d concepts](../concepts/geo3d.md) and `doc_values:` above. |
| `add_datetime_field(name, stored: true, indexed: true, doc_values: true)` | UTC datetime field. See `doc_values:` above. |
| `add_hnsw_field(name, dimension, distance: "cosine", m: 16, ef_construction: 200, quantizer: nil, subvector_count: nil, rerank_storage: nil, embedder: nil, pq_codebook_path: nil, base_weight: 1.0)` | HNSW approximate nearest-neighbor vector field. `base_weight` sets this field's relative scoring priority when searched alongside other vector fields (Issue #1084); see [Vector Search → Weights](../concepts/search/vector_search.md#weights). |
| `add_flat_field(name, dimension, distance: "cosine", embedder: nil, base_weight: 1.0)` | Flat (brute-force) vector field. |
| `add_ivf_field(name, dimension, distance: "cosine", n_clusters: 100, n_probe: 1, embedder: nil, base_weight: 1.0)` | IVF approximate nearest-neighbor vector field. |

**Vector quantization & rerank storage** (HNSW fields):

- `quantizer` — `"scalar_8bit"` (default, 4× compression) or `"product_quantization"` for higher compression. Product quantization requires `subvector_count` (must divide `dimension`).
- `rerank_storage` — set to `"f32"` to write a full-precision `*.hnsw.f32` sidecar enabling exact Stage-2 rerank; omit to keep the int8-only segment.
- `pq_codebook_path` — storage-relative file name of a shared PQ codebook (Issue #631), trained once via the `laurus train pq-codebook` CLI command. Only meaningful with `quantizer: "product_quantization"`; commits then encode against the pre-trained codebook instead of re-training k-means per segment. Omit to keep per-segment training.

### Other methods

| Method | Description |
| :--- | :--- |
| `add_embedder(name, config)` | Register a named embedder definition. `config` is a Hash with a `"type"` key (see below). |
| `set_default_fields(fields)` | Set the default fields used when no field is specified in a query. `fields` is an Array of Strings. |
| `set_dynamic_field_policy(policy)` | Set how undeclared fields are handled. `policy` is `"strict"`, `"dynamic"` (default), or `"ignore"`. See notes below. |
| `dynamic_field_policy -> String` | Return the current policy as a lowercase string. |
| `field_names -> Array<String>` | Return the list of field names defined in this schema. |

#### Dynamic field policy

Controls what happens when a document is ingested with field names that are
not declared in the schema:

- `"strict"` — Reject the document.
- `"dynamic"` (default) — Infer a type for each undeclared field and add it
  to the schema. **Warning**: integer fields silently truncate incoming
  float values (`3.14` → `3`). Use `"strict"` if you need to reject such
  type mismatches.
- `"ignore"` — Silently drop the undeclared fields.

See [Schema & Fields](../concepts/schema_and_fields.md#dynamic-schema) for
the full behaviour matrix.

### Embedder types

| `"type"` | Required keys | Feature flag |
| :--- | :--- | :--- |
| `"precomputed"` | -- | (always available) |
| `"candle_bert"` | `"model"` | `embeddings-candle` |
| `"candle_clip"` | `"model"` | `embeddings-multimodal` |
| `"openai"` | `"model"` | `embeddings-openai` |

### Distance metrics

| Value | Description |
| :--- | :--- |
| `"cosine"` | Cosine similarity (default) |
| `"euclidean"` | Euclidean distance |
| `"dot_product"` | Dot product |
| `"manhattan"` | Manhattan distance |
| `"angular"` | Angular distance |

---

## Query classes

### TermQuery

```ruby
Laurus::TermQuery.new(field, term)
```

Matches documents containing the exact term in the given field.

### PhraseQuery

```ruby
Laurus::PhraseQuery.new(field, terms)
```

Matches documents containing the terms in order. `terms` is an Array of Strings.

### FuzzyQuery

```ruby
Laurus::FuzzyQuery.new(field, term, max_edits: 2)
```

Approximate match allowing up to `max_edits` edit-distance errors.

### WildcardQuery

```ruby
Laurus::WildcardQuery.new(field, pattern)
```

Pattern match. `*` matches any sequence of characters, `?` matches any single character.

### NumericRangeQuery

```ruby
Laurus::NumericRangeQuery.new(field, min: nil, max: nil)
```

Matches numeric values in the range `[min, max]`. Pass `nil` for an open bound. The type (integer or float) is inferred from the Ruby type of `min`/`max`.

### GeoDistanceQuery

```ruby
Laurus::GeoDistanceQuery.within_radius(field, lat, lon, distance_m)
```

Geo-distance (radius) search. Returns documents whose `(lat, lon)` coordinate
is within `distance_m` metres of the given point.

### GeoBoundingBoxQuery

```ruby
Laurus::GeoBoundingBoxQuery.within_bounding_box(
  field, min_lat, min_lon, max_lat, max_lon,
)
```

Geo bounding-box search. Returns documents whose `(lat, lon)` coordinate lies
inside the axis-aligned `[min_lat, max_lat] × [min_lon, max_lon]` rectangle.

### Geo3dDistanceQuery

```ruby
Laurus::Geo3dDistanceQuery.within_sphere(field, x, y, z, distance_m)
```

Sphere search over a 3D ECEF point field. Returns documents whose `(x, y, z)`
coordinate is within `distance_m` metres of the centre. See
[Geo3d concepts](../concepts/geo3d.md) for ECEF theory.

### Geo3dBoundingBoxQuery

```ruby
Laurus::Geo3dBoundingBoxQuery.within_box(
  field,
  min_x, min_y, min_z,
  max_x, max_y, max_z,
)
```

Axis-aligned 3D bounding-box search.

### Geo3dNearestQuery

```ruby
Laurus::Geo3dNearestQuery.k_nearest(
  field, x, y, z, k,
  initial_radius_m: nil,
  max_radius_m: nil,
)
```

k-nearest-neighbour search over a 3D ECEF point field. The optional
`initial_radius_m:` and `max_radius_m:` keyword arguments tune the
iterative-expansion search cone.

### BooleanQuery

```ruby
bq = Laurus::BooleanQuery.new
bq.must(query)
bq.should(query)
bq.must_not(query)
```

Compound boolean query. `must` clauses all have to match; `must_not` clauses must not match. `should` clauses contribute to scoring; at least one of them must match if there are no `must` clauses.

### SpanQuery

```ruby
# Single term
Laurus::SpanQuery.term(field, term)

# Near: terms within slop positions
Laurus::SpanQuery.near(field, terms, slop: 0, ordered: true)

# Near with nested SpanQuery clauses
Laurus::SpanQuery.near_spans(field, clauses, slop: 0, ordered: true)

# Containing: big span contains little span
Laurus::SpanQuery.containing(field, big, little)

# Within: include span within exclude span at max distance
Laurus::SpanQuery.within(field, include_span, exclude_span, distance)
```

Positional / proximity span queries. `near` takes an Array of term Strings, while `near_spans` takes an Array of `SpanQuery` objects for nested expressions.

### VectorQuery

```ruby
Laurus::VectorQuery.new(field, vector)
```

Approximate nearest-neighbor search using a pre-computed embedding vector. `vector` is an Array of Floats.

### VectorTextQuery

```ruby
Laurus::VectorTextQuery.new(field, text)
```

Converts `text` to an embedding at query time and runs vector search. Requires an embedder configured on the index.

---

## SearchRequest

Full-featured search request for advanced control.

```ruby
Laurus::SearchRequest.new(
  query: nil,
  lexical_query: nil,
  vector_query: nil,
  filter_query: nil,
  fusion: nil,
  limit: 10,
  offset: 0,
)
```

| Parameter | Description |
| :--- | :--- |
| `query:` | A DSL string or single query object. Mutually exclusive with `lexical_query:` / `vector_query:`. |
| `lexical_query:` | Lexical component for explicit hybrid search. |
| `vector_query:` | Vector component for explicit hybrid search. |
| `filter_query:` | Lexical filter applied after scoring. |
| `fusion:` | Fusion algorithm (`RRF` or `WeightedSum`). Defaults to `RRF(k: 60)` when both components are set. |
| `limit:` | Maximum number of results (default 10). |
| `offset:` | Pagination offset (default 0). |

---

## SearchResult

Returned by `Index#search`.

```ruby
result.id        # => String   -- External document identifier
result.score     # => Float    -- Relevance score
result.document  # => Hash|nil -- Retrieved field values, or nil if deleted
```

---

## Fusion algorithms

### RRF

```ruby
Laurus::RRF.new(k: 60.0)
```

Reciprocal Rank Fusion. Merges lexical and vector result lists by rank position. `k` is a smoothing constant; higher values reduce the influence of top-ranked results.

### WeightedSum

```ruby
Laurus::WeightedSum.new(lexical_weight: 0.5, vector_weight: 0.5)
```

Normalises both score lists independently, then combines them as `lexical_weight * lexical_score + vector_weight * vector_score`.

---

## Text analysis

### SynonymDictionary

```ruby
dict = Laurus::SynonymDictionary.new
dict.add_synonym_group(["fast", "quick", "rapid"])
```

A dictionary of synonym groups. All terms in a group are treated as synonyms of each other.

### WhitespaceTokenizer

```ruby
tokenizer = Laurus::WhitespaceTokenizer.new
tokens = tokenizer.tokenize("hello world")
```

Splits text on whitespace boundaries and returns an Array of `Token` objects.

### SynonymGraphFilter

```ruby
filter = Laurus::SynonymGraphFilter.new(dictionary, keep_original: true, boost: 1.0)
expanded = filter.apply(tokens)
```

Token filter that expands tokens with their synonyms from a `SynonymDictionary`.

### Token

```ruby
token.text                # => String  -- The token text
token.position            # => Integer -- Position in the token stream
token.start_offset        # => Integer -- Character start offset in the original text
token.end_offset          # => Integer -- Character end offset in the original text
token.boost               # => Float   -- Score boost factor (1.0 = no adjustment)
token.stopped             # => Boolean -- Whether removed by a stop filter
token.position_increment  # => Integer -- Difference from the previous token's position
token.position_length     # => Integer -- Number of positions spanned
```

---

## Field value types

Ruby values are automatically converted to Laurus `DataValue` types:

| Ruby type | Laurus type | Notes |
| :--- | :--- | :--- |
| `nil` | `Null` | |
| `true` / `false` | `Bool` | |
| `Integer` | `Int64` | |
| `Float` | `Float64` | |
| `String` | `Text` | |
| `Array` of numerics | `Vector` | Elements coerced to `f32` |
| `Hash` with `"lat"`, `"lon"` | `Geo` | Two `Float` values |
| `Hash` with `"x"`, `"y"`, `"z"` | `GeoEcef` | Three `Float` values, meters (3D ECEF Cartesian) |
| `Time` / `String` responding to `iso8601` | `DateTime` | Converted via `iso8601` |
