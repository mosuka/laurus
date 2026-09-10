# API Reference

## Index

The primary entry point. Wraps the Laurus search engine.

```typescript
class Index {
  static create(
    path?: string | null,
    schema?: Schema,
    walSyncPolicy?: WalSyncPolicy,
    commitPolicy?: CommitPolicy,
  ): Promise<Index>;
}
```

### Factory method

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `string \| null` | `null` | Directory for persistent storage. `null` creates an in-memory index. When given, the directory follows the `<path>/schema.toml` + `<path>/store/` layout `laurus-cli create index`/`--index-dir` uses, so an index built here can also be opened by the CLI (and vice versa). See below. |
| `schema` | `Schema` | empty | Schema definition. Only meaningful when *creating* a new file-backed index (or an in-memory one); must be omitted when reopening an existing file-backed index -- the persisted schema is loaded instead. An empty schema is used when omitted for a new index. |
| `walSyncPolicy` | `WalSyncPolicy` | per-record | WAL durability policy. Omit to keep the default per-record `fsync`. See [WAL sync policy / durability](#wal-sync-policy--durability). |
| `commitPolicy` | `CommitPolicy` | manual | Auto-commit policy. Omit to keep the default manual policy (the caller drives every `commit()`). See [Commit policy / auto-commit](#commit-policy--auto-commit). |

**Creating vs. reopening a file-backed index** (`path` given): if `<path>/schema.toml` does not yet exist, this call **creates** a new index and persists `schema` (or an empty schema, if omitted) to it. If `<path>/schema.toml` already exists, this call **reopens** the index -- `schema` must be omitted, or the call throws (it would be ambiguous which schema should win). It also throws if `path` contains an index in the layout used before this convention was introduced (segment files directly under `path`, no `schema.toml`).

### Methods

| Method | Description |
| :--- | :--- |
| `putDocument(id, doc)` | Upsert a document. Replaces all existing versions. |
| `addDocument(id, doc)` | Append a document chunk without removing existing versions. |
| `putDocuments(docs)` | Batched upsert. `docs` is an `Array<[id, doc]>` applied in order with one WAL fsync per batch (duplicate ids dedup, last wins). Fails fast at the first bad entry; the applied prefix is not rolled back. |
| `addDocuments(docs)` | Batched chunk append. Like `putDocuments` but repeated ids accumulate as separate versions. |
| `getDocuments(id)` | Return all stored versions for the given ID. |
| `deleteDocuments(id)` | Delete all versions for the given ID. |
| `commit()` | Flush writes and make pending changes searchable. |
| `flushWal()` | Force a durable WAL barrier. See [WAL sync policy / durability](#wal-sync-policy--durability). |
| `search(query, limit?, offset?)` | Search with a DSL string. |
| `searchTerm(field, term, limit?, offset?)` | Search with an exact term match. |
| `searchVector(field, vector, limit?, offset?)` | Search with a pre-computed vector. |
| `searchVectorText(field, text, limit?, offset?)` | Search with text (auto-embedded). |
| `searchWithRequest(request)` | Search with a `SearchRequest`. |
| `searchBatch(queries, limit?, offset?)` | Execute multiple DSL string queries in parallel. `results[i]` corresponds to `queries[i]`. Returns `Promise<Array<Array<JsSearchResult>>>`. Empty input returns `[]`. |
| `stats()` | Return index statistics (`documentCount`, `vectorFields`). |

All document methods and search methods are async
and return Promises. `stats()` is synchronous.

`stats()` returns an object shaped like:

```typescript
interface IndexStats {
  documentCount: number;
  vectorFields: Record<string, { count: number; dimension: number }>;
}
```

### WAL sync policy / durability

For a persistent index, every write is appended to a write-ahead log (WAL).
By default the WAL is `fsync`-ed on **every** record, so each write is fully
durable as soon as its Promise resolves. `Index.create` accepts an optional
`walSyncPolicy` to trade some durability for higher write throughput, and
`flushWal()` forces a durable barrier on demand.

```typescript
class WalSyncPolicy {
  static perRecord(): WalSyncPolicy;
  static group(
    maxRecords?: number,
    maxBytes?: number,
    maxIntervalMs?: number,
  ): WalSyncPolicy;
}
```

| Constructor | Description |
| :--- | :--- |
| `WalSyncPolicy.perRecord()` | Default. `fsync` after every WAL record; fully durable per write. |
| `WalSyncPolicy.group(...)` | Group commit. Batches `fsync` across writes. |

`group(...)` parameters (omit any argument to keep its default):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `maxRecords` | `1024` | Flush once this many records have accumulated. |
| `maxBytes` | `1048576` (1 MiB) | Flush once this many unsynced bytes have accumulated. |
| `maxIntervalMs` | none | Optional periodic flush timer (milliseconds). Omit to disable the timer. |

With group commit the WAL is flushed when **either** `maxRecords` or
`maxBytes` is reached, and always at `commit()`. A crash can lose up to the
last unsynced batch — the same trade-off as SQLite's `synchronous = NORMAL`.
Call `flushWal()` to force everything written so far to disk without a full
`commit()`.

| Method | Description |
| :--- | :--- |
| `flushWal()` | Force a durable WAL barrier now. Returns `Promise<void>`. |

```javascript
import { Index, WalSyncPolicy } from "laurus-nodejs";

// Opt into group commit with a 1-second periodic flush timer.
const policy = WalSyncPolicy.group(4096, undefined, 1000);
const index = await Index.create("./myindex", schema, policy);

for (let i = 0; i < 10000; i++) {
  await index.putDocument(`doc${i}`, { title: `Document ${i}` });
}

// Force a durable barrier without committing yet.
await index.flushWal();

await index.commit(); // also flushes the WAL
```

Omit `walSyncPolicy` (or pass `WalSyncPolicy.perRecord()`) to keep the
default, fully durable behaviour.

### Commit policy / auto-commit

By default the caller drives every `commit()`, so pending changes become
searchable only when you call it explicitly. `Index.create` accepts an
optional `commitPolicy` to let the engine auto-commit on an ingestion-driven
cadence instead, so writes are materialized without an explicit `commit()`.

```typescript
class CommitPolicy {
  static manual(): CommitPolicy;
  static everyDocs(n: number): CommitPolicy;
  static intervalMs(ms: number): CommitPolicy;
}
```

| Constructor | Description |
| :--- | :--- |
| `CommitPolicy.manual()` | Default. No auto-commit; the caller drives every `commit()`. |
| `CommitPolicy.everyDocs(n)` | Auto-commit after every `n` applied documents. |
| `CommitPolicy.intervalMs(ms)` | Auto-commit at least every `ms` milliseconds via a background timer (default: none). **Native-only**; a no-op on wasm (no background threads). |

With `everyDocs(n)` the engine commits after every `n` applied documents,
counted across both singular and batch ingest — including every `n`
documents **within** a single batch. Passing `everyDocs(0)` is valid and
disables auto-commit, which is equivalent to `CommitPolicy.manual()`.

`intervalMs(ms)` is the time-based counterpart of `everyDocs(n)`: a background
timer auto-commits at least every `ms` milliseconds, so a trailing partial
batch is materialized even while ingestion is idle. It is **native-only** — on
wasm there are no background threads, so the engine treats `intervalMs` as a
no-op (the factory still constructs the value, but no timed commit happens
under WebAssembly).

`commitPolicy` is orthogonal to `walSyncPolicy`: `walSyncPolicy` governs WAL
`fsync` durability, while `commitPolicy` governs **when** the stores
materialize pending changes into a searchable commit. You can combine them
freely.

```javascript
import { Index, CommitPolicy } from "laurus-nodejs";

// Auto-commit after every 1000 applied documents.
const index = await Index.create(
  null,
  schema,
  undefined,
  CommitPolicy.everyDocs(1000),
);

for (let i = 0; i < 10000; i++) {
  await index.putDocument(`doc${i}`, { title: `Document ${i}` });
}
// The engine has already committed ten times; no explicit commit() needed.
```

Omit `commitPolicy` (or pass `CommitPolicy.manual()`) to keep the default,
caller-driven commit behaviour.

---

## Schema

Defines the fields and index types for an `Index`.

```typescript
class Schema {
  constructor();
}
```

### Field methods

| Method | Description |
| :--- | :--- |
| `addTextField(name, stored?, indexed?, termVectors?, analyzer?)` | Full-text field (inverted index, BM25). `analyzer` is the name of a parameter-less built-in (`"standard"`, `"english"`, `"keyword"`, `"simple"`, `"noop"`) or any custom name registered via `addAnalyzer`. For the parameterised Japanese preset (which requires a Lindera dictionary path), register a custom analyzer with a `lindera` tokenizer and reference it by name. |
| `addIntegerField(name, stored?, indexed?, multiValued?)` | 64-bit integer field. Pass `multiValued: true` to accept arrays of integers (range queries match if any value satisfies the predicate). |
| `addFloatField(name, stored?, indexed?, multiValued?)` | 64-bit float field. Pass `multiValued: true` to accept arrays of floats (range queries match if any value satisfies the predicate). |
| `addBooleanField(name, stored?, indexed?)` | Boolean field. |
| `addBytesField(name, stored?)` | Raw bytes field. |
| `addGeoField(name, stored?, indexed?)` | Geographic coordinate field. |
| `addGeo3dField(name, stored?, indexed?)` | 3D ECEF Cartesian point field (x, y, z in metres). See [Geo3d concepts](../concepts/geo3d.md). |
| `addDatetimeField(name, stored?, indexed?)` | UTC datetime field. |
| `addHnswField(name, dimension, distance?, m?, efConstruction?, defaultEfSearch?, embedder?, quantizer?, subvectorCount?, rerankStorage?, pqCodebookPath?, baseWeight?)` | HNSW vector field. `baseWeight` sets this field's relative scoring priority when searched alongside other vector fields (Issue #1084); see [Vector Search → Weights](../concepts/search/vector_search.md#weights). |
| `addFlatField(name, dimension, distance?, embedder?, baseWeight?)` | Flat (brute-force) vector field. |
| `addIvfField(name, dimension, distance?, nClusters?, nProbe?, embedder?, baseWeight?)` | IVF vector field. |
| `addEmbedder(name, config)` | Register a named embedder. |
| `setDefaultFields(fields)` | Set default search fields. |
| `setDynamicFieldPolicy(policy)` | Set how undeclared fields are handled. `policy` is `"strict"`, `"dynamic"` (default), or `"ignore"`. See notes below. |
| `dynamicFieldPolicy()` | Return the current policy as a lowercase string. |
| `fieldNames()` | Return all field names. |
| `toString()` | Return a string representation of the schema (`"Schema(fields=[...])"`). |

**Vector quantization & rerank storage** (HNSW fields):

- `quantizer` — `"scalar_8bit"` (default, 4× compression) or `"product_quantization"` for higher compression. Product quantization requires `subvectorCount` (must divide `dimension`).
- `rerankStorage` — set to `"f32"` to write a full-precision `*.hnsw.f32` sidecar enabling exact Stage-2 rerank; omit to keep the int8-only segment.
- `pqCodebookPath` — storage-relative file name of a shared PQ codebook (Issue #631), trained once via the `laurus train pq-codebook` CLI command. Only meaningful with `quantizer: "product_quantization"`; commits then encode against the pre-trained codebook instead of re-training k-means per segment. Omit to keep per-segment training.

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

```typescript
new TermQuery(field: string, term: string)
```

Matches documents containing the exact term in the given field.

### PhraseQuery

```typescript
new PhraseQuery(field: string, terms: string[])
```

Matches documents containing the terms in order.

### FuzzyQuery

```typescript
new FuzzyQuery(field: string, term: string, maxEdits?: number)
```

Approximate match allowing up to `maxEdits` edit-distance
errors (default 2).

### WildcardQuery

```typescript
new WildcardQuery(field: string, pattern: string)
```

Pattern match. `*` matches any sequence, `?` matches one
character.

### NumericRangeQuery

```typescript
new NumericRangeQuery(
  field: string,
  min?: number | null,
  max?: number | null,
  numericType?: "integer" | "float",
)
```

Matches numeric values in `[min, max]`. Pass `null` (or omit) for an
open bound. `numericType` selects the underlying range type
(`"integer"` (default) or `"float"`); other values throw.

### GeoDistanceQuery

```typescript
GeoDistanceQuery.withinRadius(
  field: string, lat: number, lon: number, distanceM: number,
): GeoDistanceQuery
```

Geographic distance (radius) search.

### GeoBoundingBoxQuery

```typescript
GeoBoundingBoxQuery.withinBoundingBox(
  field: string,
  minLat: number, minLon: number,
  maxLat: number, maxLon: number,
): GeoBoundingBoxQuery
```

Geographic bounding-box search.

### Geo3dDistanceQuery

```typescript
Geo3dDistanceQuery.withinSphere(
  field: string,
  x: number, y: number, z: number,
  distanceM: number,
): Geo3dDistanceQuery
```

Sphere search over a 3D ECEF point field. Returns documents whose `(x, y, z)`
coordinate is within `distanceM` metres of the centre. See
[Geo3d concepts](../concepts/geo3d.md) for ECEF theory.

### Geo3dBoundingBoxQuery

```typescript
Geo3dBoundingBoxQuery.withinBox(
  field: string,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): Geo3dBoundingBoxQuery
```

Axis-aligned 3D bounding-box search.

### Geo3dNearestQuery

```typescript
Geo3dNearestQuery.kNearest(
  field: string,
  x: number, y: number, z: number,
  k: number,
  initialRadiusM?: number,
  maxRadiusM?: number,
): Geo3dNearestQuery
```

k-nearest-neighbour search over a 3D ECEF point field. The optional
`initialRadiusM` and `maxRadiusM` parameters tune the iterative-expansion
search cone.

### BooleanQuery

```typescript
class BooleanQuery {
  constructor();
  // For each query type X in
  //   { Term, Phrase, Fuzzy, Wildcard, NumericRange,
  //     GeoDistance, GeoBoundingBox,
  //     Geo3dDistance, Geo3dBoundingBox, Geo3dNearest,
  //     Boolean, Span }:
  mustX(query: X): void;
  shouldX(query: X): void;
  mustNotX(query: X): void;
}
```

Compound boolean query with MUST / SHOULD / MUST_NOT clauses. Each clause
takes an instance of a specific query class — for example,
`mustTerm(new TermQuery("body", "rust"))` or
`shouldGeo3dNearest(Geo3dNearestQuery.kNearest(...))`.

The Node.js binding exposes 36 per-type methods (12 query types × 3
polarities) instead of a single polymorphic `must(query)` because of a
limitation in `napi-derive`'s validation of `Either<&T, ...>` arguments
for classes with `js_name` overrides.

`must` clauses all have to match; `mustNot` clauses must not match.
`should` clauses contribute to scoring; at least one of them must match if
there are no `must` clauses.

```javascript
const bq = new BooleanQuery();
bq.mustTerm(new TermQuery("body", "programming"));
bq.mustNotTerm(new TermQuery("title", "python"));
bq.shouldFuzzy(new FuzzyQuery("body", "data", 1));
```

### SpanQuery

```typescript
SpanQuery.term(field: string, term: string): SpanQuery
SpanQuery.near(
  field: string, terms: string[],
  slop?: number, ordered?: boolean,
): SpanQuery
SpanQuery.nearSpans(
  field: string, clauses: SpanQuery[],
  slop?: number, ordered?: boolean,
): SpanQuery
SpanQuery.containing(
  field: string, big: SpanQuery, little: SpanQuery,
): SpanQuery
SpanQuery.within(
  field: string,
  include: SpanQuery, exclude: SpanQuery, distance: number,
): SpanQuery
```

Positional/proximity span queries.

### VectorQuery

```typescript
new VectorQuery(field: string, vector: number[])
```

Nearest-neighbor search using a pre-computed embedding vector.

### VectorTextQuery

```typescript
new VectorTextQuery(field: string, text: string)
```

Converts `text` to an embedding at query time. Requires an
embedder configured on the index.

---

## SearchRequest

Full-featured search request for advanced control.

```typescript
interface SearchRequestOptions {
  queryDsl?: string;
  limit?: number;   // default 10
  offset?: number;  // default 0
}

class SearchRequest {
  constructor(options?: SearchRequestOptions);
}
```

Construct with primitive options first; attach polymorphic clauses with
the per-type setters below. As with `BooleanQuery`, the binding exposes
per-type setters because of `napi-derive`'s limitation on `Either<&T, ...>`
arguments.

### DSL and fusion setters

| Method | Description |
| :--- | :--- |
| `setQueryDsl(dsl: string)` | Set a DSL string query. |
| `setRrfFusion(rrf: RRF)` | Use RRF fusion. |
| `setWeightedSumFusion(ws: WeightedSum)` | Use weighted-sum fusion. |

### Vector setters

| Method | Description |
| :--- | :--- |
| `setVectorQuery(query: VectorQuery)` | Set a pre-computed vector query. |
| `setVectorTextQuery(query: VectorTextQuery)` | Set a text-based vector query (auto-embedded by the configured embedder). |

### Lexical setters (per type)

For each query type `X` in `{ Term, Phrase, Fuzzy, Wildcard, NumericRange,
GeoDistance, GeoBoundingBox, Geo3dDistance, Geo3dBoundingBox, Geo3dNearest,
Boolean, Span }`, the request exposes:

| Method | Description |
| :--- | :--- |
| `setLexicalX(query: X)` | Set the lexical component for an explicit hybrid request. |
| `setFilterX(query: X)` | Set the post-scoring filter component. |

That is, 24 per-type setters in total (12 lexical + 12 filter), in addition
to the DSL, vector, and fusion setters above.

```javascript
const req = new SearchRequest({ limit: 5 });
req.setLexicalTerm(new TermQuery("title", "rust"));
req.setVectorQuery(new VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4]));
req.setRrfFusion(new RRF(60.0));
const results = await index.searchWithRequest(req);
```

---

## SearchResult

Returned by search methods as an array.

```typescript
interface SearchResult {
  id: string;        // External document identifier
  score: number;     // Relevance score
  document: object | null; // Retrieved fields, or null if not stored
}
```

---

## Fusion algorithms

### RRF

```typescript
new RRF(k?: number)  // default 60.0
```

Reciprocal Rank Fusion. Merges lexical and vector result lists
by rank position.

### WeightedSum

```typescript
new WeightedSum(
  lexicalWeight?: number,  // default 0.5
  vectorWeight?: number,   // default 0.5
)
```

Normalises both score lists independently, then combines them.

---

## Text analysis

### SynonymDictionary

```typescript
class SynonymDictionary {
  constructor();
  addSynonymGroup(terms: string[]): void;
}
```

### WhitespaceTokenizer

```typescript
class WhitespaceTokenizer {
  constructor();
  tokenize(text: string): Token[];
}
```

### SynonymGraphFilter

```typescript
class SynonymGraphFilter {
  constructor(
    dictionary: SynonymDictionary,
    keepOriginal?: boolean,  // default true
    boost?: number,          // default 1.0
  );
  apply(tokens: Token[]): Token[];
}
```

### Token

```typescript
interface Token {
  text: string;
  position: number;
  startOffset: number;
  endOffset: number;
  boost: number;
  stopped: boolean;
  positionIncrement: number;
  positionLength: number;
}
```

---

## Field value types

JavaScript values are automatically converted to Laurus
`DataValue` types:

| JavaScript type | Laurus type | Notes |
| :--- | :--- | :--- |
| `null` | `Null` | |
| `boolean` | `Bool` | |
| `number` (integer) | `Int64` | |
| `number` (float) | `Float64` | |
| `string` | `Text` | ISO 8601 strings become `DateTime` |
| `number[]` | `Vector` | Coerced to `f32` |
| `{ lat, lon }` | `Geo` | Two `number` values |
| `{ x, y, z }` | `GeoEcef` | Three `number` values, meters (3D ECEF Cartesian) |
