# API Reference

## Module Functions

### `version()`

Return the laurus-wasm build version string (e.g. `"0.12.1"`).
Applications persisting laurus state in OPFS can stamp it with this
value to detect state written by a different build whose on-disk
format may have changed (the demo samples do exactly this).

## Index

The main entry point for creating and querying search indexes.

### Static Methods

#### `Index.create(schema?, walSyncPolicy?, commitPolicy?)`

Create a new in-memory (ephemeral) index.

- **Parameters:**
  - `schema` (Schema, optional) -- Schema definition. An empty schema is
    used when omitted.
  - `walSyncPolicy` (WalSyncPolicy, optional) -- WAL durability policy. Omit
    to keep the default per-record sync. See
    [WAL sync policy / durability](#wal-sync-policy--durability).
  - `commitPolicy` (CommitPolicy, optional) -- Auto-commit policy. Omit to keep
    the default (manual; caller-driven commits). See
    [Commit policy / auto-commit](#commit-policy--auto-commit).
- **Returns:** `Promise<Index>`

#### `Index.open(name, schema?, walSyncPolicy?, commitPolicy?)`

Open or create a persistent index backed by OPFS.

- **Parameters:**
  - `name` (string) -- Index name (OPFS subdirectory).
  - `schema` (Schema, optional) -- Schema definition, plus any embedder
    callbacks / runtime analyzers this session needs. **Required** the
    first time an index is created, or when opening one persisted before
    schema tracking was added (see below); **optional** afterwards, since
    the field-schema part is persisted alongside the index data and
    reloaded automatically. If `schema` is still passed once a schema is
    already persisted, its field definitions are ignored in favor of the
    persisted ones -- only its embedder callbacks / runtime analyzers are
    used, since those can never be persisted and must be re-supplied every
    time a session needs them.
  - `walSyncPolicy` (WalSyncPolicy, optional) -- WAL durability policy. Omit
    to keep the default per-record sync. See
    [WAL sync policy / durability](#wal-sync-policy--durability).
  - `commitPolicy` (CommitPolicy, optional) -- Auto-commit policy. Omit to keep
    the default (manual; caller-driven commits). See
    [Commit policy / auto-commit](#commit-policy--auto-commit).
- **Returns:** `Promise<Index>`
- **Throws:** if this OPFS index already has data persisted from before
  schema tracking was added, and `schema` was not supplied to complete
  the one-time migration (the schema is then persisted for future opens).

### Instance Methods

#### `putDocument(id, document)`

Replace a document (upsert).

- **Parameters:**
  - `id` (string) -- Document identifier.
  - `document` (object) -- Key-value pairs matching schema fields.
- **Returns:** `Promise<void>`

#### `addDocument(id, document)`

Append a document version (multi-version RAG pattern).

- **Parameters / Returns:** Same as `putDocument`.

#### `putDocuments(docs)`

Batched upsert. Applies the pairs in order with one WAL fsync for the whole batch; duplicate ids within one batch dedup (last occurrence wins). Fails fast at the first bad entry, and the applied prefix is not rolled back (retrying is idempotent).

- **Parameters:**
  - `docs` (`Array<[string, object]>`) -- An array of `[id, document]` pairs.
- **Returns:** `Promise<void>`

#### `addDocuments(docs)`

Batched chunk append. Like `putDocuments` but repeated ids accumulate as separate versions.

- **Parameters / Returns:** Same as `putDocuments`.

#### `getDocuments(id)`

Retrieve all versions of a document.

- **Parameters:**
  - `id` (string)
- **Returns:** `Promise<object[]>`

#### `deleteDocuments(id)`

Delete all versions of a document.

- **Parameters:**
  - `id` (string)
- **Returns:** `Promise<void>`

#### `commit()`

Flush writes and make changes searchable. If opened with
`Index.open()`, data is also persisted to OPFS.

- **Returns:** `Promise<void>`

#### `flushWal()`

Force a durable WAL barrier on the in-memory engine WAL. See
[WAL sync policy / durability](#wal-sync-policy--durability) for the wasm
caveats — notably, this does **not** persist to OPFS; call `commit()` for
durable persistence.

- **Returns:** `Promise<void>`

#### `search(query, limit?, offset?)`

Search using a DSL string query.

- **Parameters:**
  - `query` (string) -- Query DSL (e.g. `"title:hello"`).
  - `limit` (number, default 10)
  - `offset` (number, default 0)
- **Returns:** `Promise<SearchResult[]>`

#### `searchTerm(field, term, limit?, offset?)`

Search for an exact term.

- **Parameters:**
  - `field` (string) -- Field name.
  - `term` (string) -- Exact term.
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchVector(field, vector, limit?, offset?)`

Search by vector similarity.

- **Parameters:**
  - `field` (string) -- Vector field name.
  - `vector` (number[]) -- Query embedding.
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchVectorText(field, text, limit?, offset?)`

Search by text (embedded by the registered embedder).

- **Parameters:**
  - `field` (string) -- Vector field name.
  - `text` (string) -- Text to embed.
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchGeo3dDistance(field, x, y, z, distanceM, limit?, offset?)`

Sphere search over a 3D ECEF point field. Returns documents whose `(x, y, z)`
coordinate is within `distanceM` metres of the centre. See
[Geo3d concepts](../concepts/geo3d.md) for ECEF theory.

- **Parameters:**
  - `field` (string) -- Geo3d field name.
  - `x`, `y`, `z` (number) -- Centre ECEF coordinate (metres).
  - `distanceM` (number) -- Maximum distance from the centre (metres).
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchGeo3dBoundingBox(field, minX, minY, minZ, maxX, maxY, maxZ, limit?, offset?)`

Axis-aligned 3D bounding-box search over a 3D ECEF point field.

- **Parameters:**
  - `field` (string) -- Geo3d field name.
  - `minX`, `minY`, `minZ`, `maxX`, `maxY`, `maxZ` (number) -- Box bounds (metres).
  - `limit`, `offset` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `searchGeo3dNearest(field, x, y, z, k, limit?, offset?, initialRadiusM?, maxRadiusM?)`

k-nearest-neighbour search over a 3D ECEF point field. Returns the `k`
documents closest to `(x, y, z)`. The optional `initialRadiusM` and
`maxRadiusM` parameters tune the iterative-expansion search cone.

- **Parameters:**
  - `field` (string) -- Geo3d field name.
  - `x`, `y`, `z` (number) -- Centre ECEF coordinate (metres).
  - `k` (number) -- Number of nearest neighbours to return.
  - `limit`, `offset` (number, optional)
  - `initialRadiusM`, `maxRadiusM` (number, optional)
- **Returns:** `Promise<SearchResult[]>`

#### `stats()`

Return index statistics.

- **Returns:** `{ documentCount: number, vectorFields: { [name]: { count, dimension } } }`

## WAL sync policy / durability

Each write is appended to the engine's in-memory write-ahead log (WAL).
`Index.create` and `Index.open` accept an optional `walSyncPolicy` that
controls how often that WAL is flushed. The default (omit the argument) is
per-record sync.

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
| `WalSyncPolicy.perRecord()` | Default. Flush after every WAL record. |
| `WalSyncPolicy.group(...)` | Group commit. Batch the flush across writes. |

`group(...)` parameters (omit any argument to keep its default):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `maxRecords` | `1024` | Flush once this many records have accumulated. |
| `maxBytes` | `1048576` (1 MiB) | Flush once this many unsynced bytes have accumulated. |
| `maxIntervalMs` | none | Periodic flush timer (milliseconds). **No-op on wasm** (see caveats). |

With group commit the engine WAL is flushed when **either** `maxRecords` or
`maxBytes` is reached, and always at `commit()`. A crash can lose up to the
last unsynced batch — the same trade-off as SQLite's `synchronous = NORMAL`.

### `flushWal()` (durable barrier)

`flushWal()` forces a flush of the in-memory engine WAL on demand.

- **Returns:** `Promise<void>`

### WASM caveats

WebAssembly has no background threads or direct filesystem, so two behaviours
differ from the native bindings:

- **`maxIntervalMs` is a no-op.** The periodic flush timer requires a
  background thread, which is unavailable on wasm. Group commit still flushes
  on the `maxRecords` / `maxBytes` thresholds and at `commit()`.
- **`flushWal()` flushes the in-memory engine WAL only.** OPFS persistence
  still happens at `commit()`. For durable persistence on wasm, call
  `commit()`.

```javascript
import { Index, Schema, WalSyncPolicy } from "./pkg/laurus_wasm.js";

const schema = new Schema();
schema.addTextField("title");

// Opt into group commit. maxIntervalMs is accepted but ignored on wasm.
const policy = WalSyncPolicy.group(4096, undefined, 1000);
const index = await Index.open("my-index", schema, policy);

for (let i = 0; i < 10000; i++) {
  await index.putDocument(`doc${i}`, { title: `Document ${i}` });
}

await index.flushWal(); // flushes the engine WAL (not OPFS)
await index.commit();   // makes changes searchable AND persists to OPFS
```

## Commit policy / auto-commit

A commit materializes buffered writes into the searchable stores.
`Index.create` and `Index.open` accept an optional `commitPolicy` that controls
whether the engine commits on your behalf. The default (omit the argument) is
manual: you drive every `commit()` yourself.

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
| `CommitPolicy.intervalMs(ms)` | Auto-commit at least every `ms` milliseconds via a background timer (default: none). **Native only — no-op on wasm.** |

With `everyDocs(n)` the engine commits once every `n` applied documents. The
counter spans both singular and batch ingest, and it also fires **within** a
batch — a `putDocuments` call larger than `n` triggers one or more commits mid
batch. `everyDocs(0)` is valid and disables auto-commit, which is equivalent to
`CommitPolicy.manual()`.

`intervalMs(ms)` is the time-based counterpart of `everyDocs`: a background
timer commits at least every `ms` milliseconds, so a trailing partial batch is
committed even while ingestion is idle. It is **native only** — see the WASM
note below.

`commitPolicy` is **orthogonal** to `walSyncPolicy`: `walSyncPolicy` governs how
often the WAL is fsynced for durability, while `commitPolicy` governs when the
stores materialize buffered writes into searchable state. They are configured
independently.

### WASM note

Unlike `walSyncPolicy`'s `maxIntervalMs` background timer (a no-op on wasm),
`everyDocs` needs **no** background thread — the document counter is checked
inline during ingestion — so auto-commit works fully under WebAssembly.

`intervalMs`, by contrast, relies on a background timer just like
`walSyncPolicy`'s `maxIntervalMs`, and wasm has no background threads. The
factory still constructs a value so portable policy code keeps compiling, but
the timer never runs under WebAssembly and **`intervalMs` has no effect on
wasm** — no timed commit ever fires. Use `everyDocs` for auto-commit on wasm.

```javascript
import { Index, Schema, CommitPolicy } from "./pkg/laurus_wasm.js";

const schema = new Schema();
schema.addTextField("title");

// Auto-commit after every 1000 applied documents.
const index = await Index.open(
  "my-index",
  schema,
  undefined,
  CommitPolicy.everyDocs(1000),
);

for (let i = 0; i < 10000; i++) {
  await index.putDocument(`doc${i}`, { title: `Document ${i}` });
}
// The engine has auto-committed 10 times; no explicit commit() required.
```

## Schema

Builder for defining index fields and embedders.

### Constructor

#### `new Schema()`

Create an empty schema.

### Methods

#### `addTextField(name, stored?, indexed?, termVectors?, docValues?, analyzer?)`

Add a full-text field. `docValues` controls whether the value is also
copied into DocValues, the column-oriented store sort/facet/aggregation
read from (Issue #1047, default `true`); takes effect only when `stored`
is also `true`. `analyzer` is the name of a parameter-less built-in
(`"standard"`, `"english"`, `"keyword"`, `"simple"`, `"noop"`) or the
name of a runtime analyzer registered via `addAnalyzer()`.

For Japanese morphological analysis, build a `JapaneseAnalyzer` from
raw IPADIC bytes and register it with `addAnalyzer()` first; see
[`JapaneseAnalyzer.fromBytes`](#japaneseanalyzerfrombytesmetadata-dicttrie--mode)
and [`addAnalyzer`](#addanalyzername-analyzer) below.

#### `addIntegerField(name, stored?, indexed?, multiValued?, docValues?)`

Add a 64-bit integer field. Pass `multiValued: true` to accept arrays of
integers; range queries then match if any value satisfies the predicate
(Lucene-style "any match" with constant scoring). See `docValues` above.

#### `addFloatField(name, stored?, indexed?, multiValued?, docValues?)`

Add a 64-bit float field. Pass `multiValued: true` to accept arrays of
floats; range queries then match if any value satisfies the predicate
(Lucene-style "any match" with constant scoring). See `docValues` above.

#### `addBooleanField(name, stored?, indexed?, docValues?)`

Add a boolean field. See `docValues` above.

#### `addDatetimeField(name, stored?, indexed?, docValues?)`

Add a date/time field. See `docValues` above.

#### `addGeoField(name, stored?, indexed?, docValues?)`

Add a geographic coordinate field. See `docValues` above.

#### `addGeo3dField(name, stored?, indexed?, docValues?)`

Add a 3D ECEF Cartesian point field. Values are submitted as a `{ x, y, z }`
object with metres units. See [Geo3d concepts](../concepts/geo3d.md) for
ECEF theory, and `docValues` above.

The WASM binding does not expose `Geo3dDistanceQuery` / `Geo3dBoundingBoxQuery`
/ `Geo3dNearestQuery` as JS classes (wasm-bindgen cannot expose `dyn Query`
trait objects). Instead, use the `Index.searchGeo3dDistance` /
`Index.searchGeo3dBoundingBox` / `Index.searchGeo3dNearest` methods documented
above.

#### `addBytesField(name, stored?)`

Add a binary data field. No `docValues` option: a `Bytes` value is never
written to DocValues regardless.

#### `addHnswField(name, dimension, distance?, m?, efConstruction?, defaultEfSearch?, embedder?, quantizer?, subvectorCount?, rerankStorage?, pqCodebookPath?, baseWeight?)`

Add an HNSW vector index field.

- `distance`: `"cosine"` (default), `"euclidean"`, `"dot_product"`,
  `"manhattan"`, `"angular"`
- `m`: Branching factor (default 16)
- `efConstruction`: Build-time expansion (default 200)
- `defaultEfSearch`: Schema-level default for the query-time `ef_search`
  candidate-list size; omit to use the internal fallback of 50
- `quantizer`: `"scalar_8bit"` (default) or `"product_quantization"`
  (requires `subvectorCount`)
- `subvectorCount`: number of PQ sub-vectors; must divide `dimension`
- `rerankStorage`: omit (default) or `"f32"` to store a full-precision
  rerank sidecar
- `pqCodebookPath`: omit (default) or the storage-relative file name of
  a shared PQ codebook (Issue #631) to reuse across segments instead of
  per-segment training
- `baseWeight`: this field's relative scoring priority when searched
  alongside other vector fields (default `1.0`, Issue #1084); see
  [Vector Search → Weights](../concepts/search/vector_search.md#weights)

#### `addFlatField(name, dimension, distance?, embedder?, baseWeight?)`

Add a brute-force vector index field.

#### `addIvfField(name, dimension, distance?, nClusters?, nProbe?, embedder?, baseWeight?)`

Add an IVF vector index field.

- `nClusters`: Number of partitioning clusters (default 100)
- `nProbe`: Number of clusters to probe at query time (default 1)

**Vector quantization & rerank storage** (HNSW fields):

- `quantizer` — `"scalar_8bit"` (default, 4× compression) or `"product_quantization"` for higher compression. Product quantization requires `subvectorCount` (must divide `dimension`).
- `rerankStorage` — set to `"f32"` to write a full-precision `*.hnsw.f32` sidecar enabling exact Stage-2 rerank; omit to keep the int8-only segment.
- `pqCodebookPath` — storage-relative file name of a shared PQ codebook (Issue #631), trained once via the `laurus train pq-codebook` CLI command. Only meaningful with `quantizer: "product_quantization"`; commits then encode against the pre-trained codebook instead of re-training k-means per segment. Omit to keep per-segment training.

#### `addAnalyzer(name, analyzer)`

Register a pre-built analyzer instance under `name`. Resolved before the
parameter-less built-in names and before `schema.analyzers` definitions
when text fields reference an analyzer by name.

Currently only `JapaneseAnalyzer` instances built via
[`JapaneseAnalyzer.fromBytes`](#japaneseanalyzerfrombytesmetadata-dicttrie--mode)
are accepted here. The runtime registry is the only practical way to use
the Japanese analyzer in browser WASM, where the
`{ "language": "japanese", "dict": ... }` preset cannot resolve a
filesystem path.

```javascript
import { JapaneseAnalyzer, Schema } from "laurus-wasm";
import { downloadDictionary, loadDictionaryFiles } from "laurus-wasm/opfs";

await downloadDictionary("./dict/lindera-ipadic.zip", "ipadic");
const f = await loadDictionaryFiles("ipadic");
const ja = JapaneseAnalyzer.fromBytes(
  f.metadata, f.dictTrie, f.dictValsIdx, f.dictVals,
  f.dictWordsIdx, f.dictWords, f.matrixMtx, f.charDef, f.unk, "normal",
);

const schema = new Schema();
schema.addAnalyzer("ja-ipadic", ja);
schema.addTextField("body", undefined, undefined, undefined, "ja-ipadic");
```

#### `addEmbedder(name, config)`

Register a named embedder. WASM supports two `type` values:

- `"precomputed"` — No embedding is performed; vectors are passed directly via
  `putDocument()` / `searchVector()`.
- `"callback"` — Provide a JavaScript callback `embed: (text) => Promise<number[]>`
  that the engine will invoke during ingestion and `searchVectorText()`. This
  enables in-engine auto-embedding using Transformers.js or any other in-browser
  embedding library.

```javascript
// Precomputed embedder
schema.addEmbedder("precomputed-embedder", { type: "precomputed" });

// Callback embedder (e.g. backed by Transformers.js)
schema.addEmbedder("callback-embedder", {
  type: "callback",
  embed: async (text) => {
    const output = await pipeline(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  },
});
```

#### `setDefaultFields(fields)`

Set the default search fields.

#### `setDynamicFieldPolicy(policy)`

Set how the engine treats fields that appear in ingested documents but are
absent from the schema. `policy` is one of `"strict"`, `"dynamic"`
(default), or `"ignore"` (case-insensitive). Throws on an invalid value.

- `"strict"` — Reject the document.
- `"dynamic"` — Infer a type for each undeclared field and add it to the
  schema. **Warning**: integer fields silently truncate incoming float
  values (`3.14` → `3`).
- `"ignore"` — Silently drop the undeclared fields.

See [Schema & Fields](../concepts/schema_and_fields.md#dynamic-schema) for
the full behaviour matrix.

#### `dynamicFieldPolicy()`

Returns the current policy as a lowercase string.

#### `fieldNames()`

Returns an array of defined field names.

#### `toString()`

Returns a string representation of the schema (`"Schema(fields=[...])"`).

## SearchResult

```typescript
interface SearchResult {
  id: string;
  score: number;
  document: object | null;
}
```

## Analysis

### JapaneseAnalyzer

Japanese morphological analyzer constructed from raw Lindera dictionary
bytes. Browser WASM has no real filesystem, so the standard
`{ "language": "japanese", "dict": "/path/to/ipadic" }` preset cannot
be used. Instead, fetch a Lindera dictionary archive (typically
`lindera-ipadic-X.Y.Z.zip`), store it in OPFS via the
[OPFS helpers](#opfs-helpers), and pass the nine component byte
arrays to `JapaneseAnalyzer.fromBytes`.

#### `JapaneseAnalyzer.fromBytes(metadata, dictTrie, ..., mode?)`

Static factory that builds an analyzer from raw IPADIC bytes.

Arguments (all `Uint8Array` except `mode`):

| Argument | Source file |
| ---- | ---- |
| `metadata` | `metadata.json` |
| `dictTrie` | `dict.trie` (prefix trie) |
| `dictValsIdx` | `dict.valsidx` |
| `dictVals` | `dict.vals` |
| `dictWordsIdx` | `dict.wordsidx` |
| `dictWords` | `dict.words` |
| `matrixMtx` | `matrix.mtx` |
| `charDef` | `char_def.bin` |
| `unk` | `unk.bin` |
| `mode` | `"normal"` (default) or `"decompose"` |

Throws if any component fails to deserialize or the mode string is
invalid.

```javascript
import { JapaneseAnalyzer } from "laurus-wasm";
import { loadDictionaryFiles } from "laurus-wasm/opfs";

const f = await loadDictionaryFiles("ipadic");
const ja = JapaneseAnalyzer.fromBytes(
  f.metadata, f.dictTrie, f.dictValsIdx, f.dictVals,
  f.dictWordsIdx, f.dictWords, f.matrixMtx, f.charDef, f.unk,
  "normal",
);
```

The pipeline is `NFKC normalization → Japanese iteration mark
normalization → Lindera morphological tokenization → lowercase →
Japanese stop word filter` — identical to the `japanese` preset on the
native side.

### OPFS Helpers

The `laurus-wasm/opfs` subpath bundles helpers for downloading,
storing, and loading Lindera dictionaries from the browser's Origin
Private File System. Used together with `JapaneseAnalyzer.fromBytes`.

```javascript
import {
  downloadDictionary,
  getDictionaryVersion,
  loadDictionaryFiles,
  hasDictionary,
  listDictionaries,
  removeDictionary,
} from "laurus-wasm/opfs";
```

| Function | Description |
| ---- | ---- |
| `downloadDictionary(url, name, options?)` | Fetch a `.zip`, decompress with the Web `DecompressionStream` API, and store the nine Lindera files under `laurus/dictionaries/<name>/` in OPFS. `options.onProgress({ phase, loaded?, total? })` reports progress. `options.version` stores a version stamp next to the files (see below). |
| `getDictionaryVersion(name)` | Return the version stamp stored by `downloadDictionary`, or `null` if the dictionary or its stamp does not exist. |
| `loadDictionaryFiles(name)` | Read the nine files back as a `{ metadata, dictTrie, dictValsIdx, dictVals, dictWordsIdx, dictWords, matrixMtx, charDef, unk }` object suitable for `JapaneseAnalyzer.fromBytes`. |
| `hasDictionary(name)` | `true` if the dictionary directory exists in OPFS. |
| `listDictionaries()` | Return an array of stored dictionary names. |
| `removeDictionary(name)` | Delete the dictionary directory. |

The binary dictionary format is tied to the Lindera (and daachorse)
version compiled into the WASM binary, so a dictionary cached in OPFS
becomes unreadable after your app updates its Lindera version
(deserialization fails with an `InvalidAutomatonError`). Pass the
Lindera version your zip was built for as `options.version` when
downloading, then compare `getDictionaryVersion(name)` against the
version your current build expects on startup and re-download on
mismatch. A `null` stamp should be treated as a mismatch.

Browser CORS prevents fetching directly from GitHub Releases, so host
the zip on the same origin as your app (the Laurus demo bundles
`./dict/lindera-ipadic.zip` alongside the WASM at deploy time).

### WhitespaceTokenizer

```javascript
const tokenizer = new WhitespaceTokenizer();
const tokens = tokenizer.tokenize("hello world");
// [{ text: "hello", position: 0, ... }, { text: "world", position: 1, ... }]
```

### SynonymDictionary

```javascript
const dict = new SynonymDictionary();
dict.addSynonymGroup(["ml", "machine learning"]);
```

### SynonymGraphFilter

```javascript
new SynonymGraphFilter(dictionary, keepOriginal = true, boost = 1.0)
```

- `dictionary` (`SynonymDictionary`) — Source synonym groups.
- `keepOriginal` (boolean, default `true`) — Keep the original token alongside
  the inserted synonyms.
- `boost` (number, default `1.0`) — Score boost applied to inserted synonym
  tokens.

```javascript
const filter = new SynonymGraphFilter(dict, true, 0.8);
const expanded = filter.apply(tokens);
```
