# API Reference

## Index

The primary entry point. Wraps the Laurus search engine.

```php
new \Laurus\Index(?string $path = null, ?Schema $schema = null, ?WalSyncPolicy $wal_sync_policy = null, ?CommitPolicy $commit_policy = null)
```

### Constructor

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `$path` | `string\|null` | `null` | Directory path for persistent storage. `null` creates an in-memory index. When given, the directory follows the `<path>/schema.toml` + `<path>/store/` layout `laurus-cli create index`/`--index-dir` uses, so an index built here can also be opened by the CLI (and vice versa). See below. |
| `$schema` | `Schema\|null` | `null` | Schema definition. Only meaningful when *creating* a new file-backed index (or an in-memory one); must be omitted (`null`) when reopening an existing file-backed index -- the persisted schema is loaded instead. An empty schema is used when omitted for a new index. |
| `$wal_sync_policy` | `WalSyncPolicy\|null` | `null` | Write-ahead log (WAL) durability policy. `null` keeps the default per-record fsync. See [WAL sync policy & durability](#wal-sync-policy--durability). |
| `$commit_policy` | `CommitPolicy\|null` | `null` | Auto-commit policy. `null` keeps the default manual policy (the caller drives every `commit()`). See [Commit policy & auto-commit](#commit-policy--auto-commit). |

**Creating vs. reopening a file-backed index** (`$path` given): if `<path>/schema.toml` does not yet exist, this call **creates** a new index and persists `$schema` (or an empty schema, if omitted) to it. If `<path>/schema.toml` already exists, this call **reopens** the index -- `$schema` must be omitted, or a `ValueError` is thrown (it would be ambiguous which schema should win). A `ValueError` is also thrown if `$path` contains an index in the layout used before this convention was introduced (segment files directly under `$path`, no `schema.toml`).

### Methods

| Method | Description |
| :--- | :--- |
| `putDocument(string $id, array $doc): void` | Upsert a document. Replaces all existing versions with the same ID. |
| `addDocument(string $id, array $doc): void` | Append a document chunk without removing existing versions. |
| `putDocuments(array $docs): void` | Batched upsert. `$docs` is an array of `[$id, $doc]` pairs, applied in order with one WAL fsync per batch (duplicate ids dedup, last wins). Fails fast at the first bad entry; the applied prefix is not rolled back. |
| `addDocuments(array $docs): void` | Batched chunk append. Like `putDocuments` but repeated ids accumulate as separate versions. |
| `getDocuments(string $id): array` | Return all stored versions for the given ID. |
| `deleteDocuments(string $id): void` | Delete all versions for the given ID. |
| `commit(): void` | Flush buffered writes and make all pending changes searchable. |
| `flushWal(): void` | Force a durable WAL barrier on demand. Synchronously fsyncs any unsynced WAL records. Useful when running under a group-commit policy (see below). |
| `search(mixed $query, int $limit = 10, int $offset = 0): array` | Execute a search query. Returns an array of `SearchResult`. |
| `searchBatch(array $queries, int $limit = 10, int $offset = 0): array` | Execute multiple independent searches in one call. Each query is dispatched in parallel on the underlying tokio runtime. `results[i]` corresponds to `queries[i]`. Returns an array of arrays of `SearchResult`. Empty input returns `[]`. |
| `stats(): array` | Return index statistics (`"documentCount"`, `"vectorFields"`). |

### `search` query argument

The `$query` parameter accepts any of the following:

- A **DSL string** (e.g. `"title:hello"`, `"embedding:\"memory safety\""`)
- A **lexical query object** (`TermQuery`, `PhraseQuery`, `BooleanQuery`, ...)
- A **vector query object** (`VectorQuery`, `VectorTextQuery`)
- A **`SearchRequest`** for full control

The same value kinds are accepted as the elements of `searchBatch`'s `$queries` array — DSL strings, query objects, and `SearchRequest` instances may be mixed within a single batch.

### WAL sync policy & durability

The write-ahead log (WAL) protects committed data against crashes. By
default the WAL is fully durable: every record is `fsync`ed before the write
returns. You can trade some durability for higher write throughput by opting
into **group commit**, which batches `fsync` calls.

#### WalSyncPolicy

`Laurus\WalSyncPolicy` is an immutable value object describing how the WAL is
flushed. Pass it to the `Index` constructor's `$wal_sync_policy` argument.

```php
// Default: durable per write (each record is fsynced individually).
\Laurus\WalSyncPolicy::perRecord(): WalSyncPolicy

// Group commit: batch fsyncs to amortise their cost.
\Laurus\WalSyncPolicy::group(
    ?int $max_records = null,      // flush after this many records (default 1024)
    ?int $max_bytes = null,        // flush after this many bytes (default 1 MiB)
    ?int $max_interval_ms = null,  // also flush periodically after this many ms
): WalSyncPolicy
```

| Constructor | Description |
| :--- | :--- |
| `WalSyncPolicy::perRecord()` | Default. Every record is `fsync`ed before the write returns — fully durable per write. |
| `WalSyncPolicy::group($max_records, $max_bytes, $max_interval_ms)` | Batch `fsync`s. With all arguments `null` uses the defaults (`max_records = 1024`, `max_bytes = 1 MiB`, no timer). The WAL is flushed when **either** `$max_records` **or** `$max_bytes` accumulate, and on every `commit()`. Pass `$max_interval_ms` to also flush on a periodic timer. |

Group commit is analogous to SQLite's `synchronous = NORMAL`: a crash can lose
at most the last unsynced batch of records, but the index never corrupts.
Records are always made durable at `commit()`, so a successful `commit()` is a
durability barrier regardless of policy.

#### Forcing a flush

Call `flushWal()` to force a durable barrier between commits — for example
before signalling that a batch has been safely persisted. It synchronously
`fsync`s any unsynced records. Under the default per-record policy it is
effectively a no-op.

```php
// Opt into group commit, then force durability on demand.
$policy = \Laurus\WalSyncPolicy::group(4096, 4 * 1024 * 1024);
$index = new \Laurus\Index("./myindex", null, $policy);

$index->putDocument("doc1", ["title" => "Hello"]);
$index->flushWal(); // records persisted even though the group batch is not full
```

### Commit policy & auto-commit

A commit materialises buffered writes into the lexical and vector stores and
makes pending changes searchable. By default Laurus never commits on your
behalf — the caller drives every `commit()`. You can instead let the engine
**auto-commit** after a fixed number of applied documents, or at least once
per fixed time interval.

#### CommitPolicy

`Laurus\CommitPolicy` is an immutable value object describing when the engine
commits. Pass it to the `Index` constructor's `$commit_policy` argument.

```php
// Default: no auto-commit — the caller drives every commit().
\Laurus\CommitPolicy::manual(): CommitPolicy

// Auto-commit after every N applied documents.
\Laurus\CommitPolicy::everyDocs(
    int $n,   // commit after this many applied documents
): CommitPolicy

// Auto-commit at least every N milliseconds (native only; no-op on wasm).
\Laurus\CommitPolicy::intervalMs(
    int $ms,   // commit at least this often, in milliseconds
): CommitPolicy
```

| Constructor | Description |
| :--- | :--- |
| `CommitPolicy::manual()` | Default. The engine never commits on its own; the caller drives every `commit()`. |
| `CommitPolicy::everyDocs($n)` | Auto-commit after every `$n` applied documents. Counting spans both singular and batch ingest, and triggers every `$n` documents **within** a batch. |
| `CommitPolicy::intervalMs($ms)` | Auto-commit at least every `$ms` milliseconds via a background timer, so a trailing partial batch is committed even while ingestion is idle. The time-based counterpart of `everyDocs`. Default: none. **Native only** — a no-op on wasm (WebAssembly has no background threads); the value still constructs, but no timed commit happens. |

`CommitPolicy::everyDocs(0)` is valid and disables auto-commit — it is
equivalent to `CommitPolicy::manual()`.

The commit policy is **orthogonal** to the WAL sync policy: `WalSyncPolicy`
governs when the WAL is `fsync`ed for durability, whereas `CommitPolicy`
governs when the stores materialise and pending changes become searchable. The
two are configured independently.

```php
// Auto-commit every 1000 applied documents; keep the default WAL policy.
$index = new \Laurus\Index(null, $schema, null, \Laurus\CommitPolicy::everyDocs(1000));

foreach ($docs as $id => $doc) {
    $index->putDocument($id, $doc); // engine commits automatically every 1000 docs
}
```

---

## Schema

Defines the fields and index types for an `Index`.

```php
new \Laurus\Schema()
```

### Field methods

| Method | Description |
| :--- | :--- |
| `addTextField(string $name, bool $stored = true, bool $indexed = true, bool $termVectors = true, bool $docValues = true, ?string $analyzer = null): void` | Full-text field (inverted index, BM25). `$docValues` controls whether the value is also copied into DocValues, the column-oriented store sort/facet/aggregation read from (Issue #1047); takes effect only when `$stored` is also `true`. `$analyzer` is the name of a parameter-less built-in (`"standard"`, `"english"`, `"keyword"`, `"simple"`, `"noop"`) or a custom name registered via `addAnalyzer`. The Japanese preset requires a Lindera dictionary path, so register it as a custom analyzer with a `lindera` tokenizer and reference it by name. |
| `addIntegerField(string $name, bool $stored = true, bool $indexed = true, bool $multiValued = false, bool $docValues = true): void` | 64-bit integer field. Pass `$multiValued = true` to accept arrays of integers (range queries match if any value satisfies the predicate). See `$docValues` above. |
| `addFloatField(string $name, bool $stored = true, bool $indexed = true, bool $multiValued = false, bool $docValues = true): void` | 64-bit float field. Pass `$multiValued = true` to accept arrays of floats (range queries match if any value satisfies the predicate). See `$docValues` above. |
| `addBooleanField(string $name, bool $stored = true, bool $indexed = true, bool $docValues = true): void` | Boolean field. See `$docValues` above. |
| `addBytesField(string $name, bool $stored = true): void` | Raw bytes field. No `$docValues` option: a `Bytes` value is never written to DocValues regardless. |
| `addGeoField(string $name, bool $stored = true, bool $indexed = true, bool $docValues = true): void` | Geographic coordinate field (lat/lon). See `$docValues` above. |
| `addGeo3dField(string $name, bool $stored = true, bool $indexed = true, bool $docValues = true): void` | 3D ECEF Cartesian point field (x, y, z in metres). See [Geo3d concepts](../concepts/geo3d.md) and `$docValues` above. |
| `addDatetimeField(string $name, bool $stored = true, bool $indexed = true, bool $docValues = true): void` | UTC datetime field. See `$docValues` above. |
| `addHnswField(string $name, int $dimension, ?string $distance = "cosine", int $m = 16, int $efConstruction = 200, ?int $defaultEfSearch = null, ?string $embedder = null, ?string $quantizer = null, ?int $subvectorCount = null, ?string $rerankStorage = null, ?string $pqCodebookPath = null, float $baseWeight = 1.0): void` | HNSW approximate nearest-neighbor vector field. `$baseWeight` sets this field's relative scoring priority when searched alongside other vector fields (Issue #1084); see [Vector Search → Weights](../concepts/search/vector_search.md#weights). |
| `addFlatField(string $name, int $dimension, ?string $distance = "cosine", ?string $embedder = null, float $baseWeight = 1.0): void` | Flat (brute-force) vector field. |
| `addIvfField(string $name, int $dimension, ?string $distance = "cosine", int $nClusters = 100, int $nProbe = 1, ?string $embedder = null, float $baseWeight = 1.0): void` | IVF approximate nearest-neighbor vector field. |

**Vector quantization & rerank storage** (HNSW fields):

- `quantizer` — `"scalar_8bit"` (default, 4× compression) or `"product_quantization"` for higher compression. Product quantization requires `subvectorCount` (must divide `dimension`).
- `rerankStorage` — set to `"f32"` to write a full-precision `*.hnsw.f32` sidecar enabling exact Stage-2 rerank; omit to keep the int8-only segment.
- `pqCodebookPath` — storage-relative file name of a shared PQ codebook (Issue #631), trained once via the `laurus train pq-codebook` CLI command. Only meaningful with `$quantizer = "product_quantization"`; commits then encode against the pre-trained codebook instead of re-training k-means per segment. Omit to keep per-segment training.

### Other methods

| Method | Description |
| :--- | :--- |
| `addEmbedder(string $name, array $config): void` | Register a named embedder definition. `$config` is an associative array with a `"type"` key (see below). |
| `setDefaultFields(array $fields): void` | Set the default fields used when no field is specified in a query. `$fields` is an array of strings. |
| `setDynamicFieldPolicy(string $policy): void` | Set how undeclared fields are handled. `$policy` is `"strict"`, `"dynamic"` (default), or `"ignore"`. See notes below. |
| `dynamicFieldPolicy(): string` | Return the current policy as a lowercase string. |
| `fieldNames(): array` | Return the list of field names defined in this schema. |

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

```php
new \Laurus\TermQuery(string $field, string $term)
```

Matches documents containing the exact term in the given field.

### PhraseQuery

```php
new \Laurus\PhraseQuery(string $field, array $terms)
```

Matches documents containing the terms in order. `$terms` is an array of strings.

### FuzzyQuery

```php
new \Laurus\FuzzyQuery(string $field, string $term, int $maxEdits = 2)
```

Approximate match allowing up to `$maxEdits` edit-distance errors.

### WildcardQuery

```php
new \Laurus\WildcardQuery(string $field, string $pattern)
```

Pattern match. `*` matches any sequence of characters, `?` matches any single character.

### NumericRangeQuery

```php
new \Laurus\NumericRangeQuery(string $field, mixed $min, mixed $max, ?string $numericType = "integer")
```

Matches numeric values in the range `[$min, $max]`. Pass `null` for an open bound. Set `$numericType` to `"integer"` or `"float"`.

### GeoDistanceQuery

```php
\Laurus\GeoDistanceQuery::withinRadius(
    string $field, float $lat, float $lon, float $distanceM,
): GeoDistanceQuery
```

Geo-distance (radius) search. Returns documents whose `(lat, lon)` coordinate
is within `$distanceM` metres of the given point.

### GeoBoundingBoxQuery

```php
\Laurus\GeoBoundingBoxQuery::withinBoundingBox(
    string $field,
    float $minLat, float $minLon,
    float $maxLat, float $maxLon,
): GeoBoundingBoxQuery
```

Geo bounding-box search. Returns documents whose `(lat, lon)` coordinate lies
inside the axis-aligned `[$minLat, $maxLat] × [$minLon, $maxLon]` rectangle.

### Geo3dDistanceQuery

```php
\Laurus\Geo3dDistanceQuery::withinSphere(
    string $field,
    float $x, float $y, float $z,
    float $distanceM,
): Geo3dDistanceQuery
```

Sphere search over a 3D ECEF point field. Returns documents whose `(x, y, z)`
coordinate is within `$distanceM` metres of the centre. See
[Geo3d concepts](../concepts/geo3d.md) for ECEF theory.

### Geo3dBoundingBoxQuery

```php
\Laurus\Geo3dBoundingBoxQuery::withinBox(
    string $field,
    float $minX, float $minY, float $minZ,
    float $maxX, float $maxY, float $maxZ,
): Geo3dBoundingBoxQuery
```

Axis-aligned 3D bounding-box search.

### Geo3dNearestQuery

```php
\Laurus\Geo3dNearestQuery::kNearest(
    string $field,
    float $x, float $y, float $z,
    int $k,
    ?float $initialRadiusM = null,
    ?float $maxRadiusM = null,
): Geo3dNearestQuery
```

k-nearest-neighbour search over a 3D ECEF point field. The optional
`$initialRadiusM` and `$maxRadiusM` parameters tune the iterative-expansion
search cone.

### BooleanQuery

```php
$bq = new \Laurus\BooleanQuery();
$bq->must($query);
$bq->should($query);
$bq->mustNot($query);
```

Compound boolean query. `must` clauses all have to match; `mustNot` clauses must not match. `should` clauses contribute to scoring; at least one of them must match if there are no `must` clauses.

### SpanQuery

```php
// Single term
\Laurus\SpanQuery::term(string $field, string $term): SpanQuery

// Near: terms within slop positions
\Laurus\SpanQuery::near(string $field, array $terms, int $slop = 0, bool $ordered = true): SpanQuery

// NearSpans: nested SpanQuery clauses within slop positions
\Laurus\SpanQuery::nearSpans(string $field, array $clauses, int $slop = 0, bool $ordered = true): SpanQuery

// Containing: big span contains little span
\Laurus\SpanQuery::containing(string $field, SpanQuery $big, SpanQuery $little): SpanQuery

// Within: include span within exclude span at max distance
\Laurus\SpanQuery::within(string $field, SpanQuery $include, SpanQuery $exclude, int $distance): SpanQuery
```

Positional / proximity span queries. `near` takes an array of term strings,
while `nearSpans` takes an array of `SpanQuery` objects for nested expressions
(each clause's field is re-rooted to the outer `$field`).

### VectorQuery

```php
new \Laurus\VectorQuery(string $field, array $vector)
```

Approximate nearest-neighbor search using a pre-computed embedding vector. `$vector` is an array of floats.

### VectorTextQuery

```php
new \Laurus\VectorTextQuery(string $field, string $text)
```

Converts `$text` to an embedding at query time and runs vector search. Requires an embedder configured on the index.

---

## SearchRequest

Full-featured search request for advanced control.

```php
new \Laurus\SearchRequest(
    mixed $query = null,
    mixed $lexicalQuery = null,
    mixed $vectorQuery = null,
    mixed $filterQuery = null,
    mixed $fusion = null,
    int $limit = 10,
    int $offset = 0,
)
```

| Parameter | Description |
| :--- | :--- |
| `$query` | A DSL string or single query object. Mutually exclusive with `$lexicalQuery` / `$vectorQuery`. |
| `$lexicalQuery` | Lexical component for explicit hybrid search. |
| `$vectorQuery` | Vector component for explicit hybrid search. |
| `$filterQuery` | Lexical filter applied after scoring. |
| `$fusion` | Fusion algorithm (`RRF` or `WeightedSum`). Defaults to `RRF(k: 60)` when both components are set. |
| `$limit` | Maximum number of results (default 10). |
| `$offset` | Pagination offset (default 0). |

---

## SearchResult

Returned by `Index->search()`.

```php
$result->getId()        // string   -- External document identifier
$result->getScore()     // float    -- Relevance score
$result->getDocument()  // array|null -- Retrieved field values, or null if not stored
```

---

## Fusion algorithms

### RRF

```php
new \Laurus\RRF(float $k = 60.0)
```

Reciprocal Rank Fusion. Merges lexical and vector result lists by rank position. `$k` is a smoothing constant; higher values reduce the influence of top-ranked results.

### WeightedSum

```php
new \Laurus\WeightedSum(float $lexicalWeight = 0.5, float $vectorWeight = 0.5)
```

Normalises both score lists independently, then combines them as `$lexicalWeight * lexical_score + $vectorWeight * vector_score`.

---

## Text analysis

### SynonymDictionary

```php
$dict = new \Laurus\SynonymDictionary();
$dict->addSynonymGroup(["fast", "quick", "rapid"]);
```

A dictionary of synonym groups. All terms in a group are treated as synonyms of each other.

### WhitespaceTokenizer

```php
$tokenizer = new \Laurus\WhitespaceTokenizer();
$tokens = $tokenizer->tokenize("hello world");
```

Splits text on whitespace boundaries and returns an array of `Token` objects.

### SynonymGraphFilter

```php
new \Laurus\SynonymGraphFilter(SynonymDictionary $dictionary, bool $keepOriginal = true, float $boost = 1.0)
```

| Parameter | Description |
| :--- | :--- |
| `$dictionary` | Source synonym groups. |
| `$keepOriginal` | When `true` (default), keep the original token alongside the synonyms. |
| `$boost` | Score boost applied to the inserted synonym tokens (default `1.0`). |

```php
$filter = new \Laurus\SynonymGraphFilter($dictionary, true, 1.0);
$expanded = $filter->apply($tokens);
```

Token filter that expands tokens with their synonyms from a `SynonymDictionary`.

### Token

```php
$token->getText()               // string  -- The token text
$token->getPosition()           // int     -- Position in the token stream
$token->getStartOffset()        // int     -- Character start offset in the original text
$token->getEndOffset()          // int     -- Character end offset in the original text
$token->getBoost()              // float   -- Score boost factor (1.0 = no adjustment)
$token->isStopped()             // bool    -- Whether removed by a stop filter
$token->getPositionIncrement()  // int     -- Difference from the previous token's position
$token->getPositionLength()     // int     -- Number of positions spanned
```

---

## Field value types

PHP values are automatically converted to Laurus `DataValue` types:

| PHP type | Laurus type | Notes |
| :--- | :--- | :--- |
| `null` | `Null` | |
| `true` / `false` | `Bool` | |
| `int` | `Int64` | |
| `float` | `Float64` | |
| `string` | `Text` | |
| `array` of numerics | `Vector` | Elements coerced to `f32` |
| `array` with `"lat"`, `"lon"` | `Geo` | Two `float` values |
| `array` with `"x"`, `"y"`, `"z"` | `GeoEcef` | Three `float` values, meters (3D ECEF Cartesian) |
| `string` (ISO 8601) | `DateTime` | Parsed from ISO 8601 format |
