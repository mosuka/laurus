# API Reference

## Index

The primary entry point. Wraps the Laurus search engine.

```python
class Index:
    def __init__(
        self,
        path: str | None = None,
        schema: Schema | None = None,
        wal_sync_policy: WalSyncPolicy | None = None,
        commit_policy: CommitPolicy | None = None,
    ) -> None: ...
```

### Constructor

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str \| None` | `None` | Directory path for persistent storage. `None` creates an in-memory index. When given, the directory follows the `<path>/schema.toml` + `<path>/store/` layout `laurus-cli create index`/`--index-dir` uses, so an index built here can also be opened by the CLI (and vice versa). See below. |
| `schema` | `Schema \| None` | `None` | Schema definition. Only meaningful when *creating* a new file-backed index (or an in-memory one); must be omitted (`None`) when reopening an existing file-backed index — the persisted schema is loaded instead. An empty schema is used when omitted for a new index. |
| `wal_sync_policy` | `WalSyncPolicy \| None` | `None` | WAL durability policy. `None` keeps the default per-record `fsync`. See [WAL sync policy / durability](#wal-sync-policy--durability). |
| `commit_policy` | `CommitPolicy \| None` | `None` | Auto-commit policy. `None` keeps the default manual commit (no auto-commit). See [Commit policy / auto-commit](#commit-policy--auto-commit). |

**Creating vs. reopening a file-backed index** (`path` given): if `<path>/schema.toml` does not yet exist, this call **creates** a new index and persists `schema` (or an empty schema, if omitted) to it. If `<path>/schema.toml` already exists, this call **reopens** the index — `schema` must be omitted, or a `ValueError` is raised (it would be ambiguous which schema should win). A `ValueError` is also raised if `path` contains an index in the layout used before this convention was introduced (segment files directly under `path`, no `schema.toml`).

### Methods

| Method | Description |
| :--- | :--- |
| `put_document(id, doc)` | Upsert a document. Replaces all existing versions with the same ID. |
| `add_document(id, doc)` | Append a document chunk without removing existing versions. |
| `put_documents(docs)` | Batched upsert. `docs` is an iterable of `(id, dict)` pairs, applied in order with one WAL fsync per batch (duplicate ids dedup, last wins). Fails fast at the first bad entry; the applied prefix is not rolled back. |
| `add_documents(docs)` | Batched chunk append. Like `put_documents` but repeated ids accumulate as separate versions. |
| `get_documents(id) -> list[dict]` | Return all stored versions for the given ID. |
| `delete_documents(id)` | Delete all versions for the given ID. |
| `commit()` | Flush buffered writes and make all pending changes searchable. |
| `flush_wal()` | Force a durable WAL barrier. See [WAL sync policy / durability](#wal-sync-policy--durability). |
| `search(query, *, limit=10, offset=0) -> list[SearchResult]` | Execute a search query. |
| `search_batch(queries, *, limit=10, offset=0) -> list[list[SearchResult]]` | Execute multiple independent searches in one call. Each query is dispatched in parallel on the underlying tokio runtime. `results[i]` corresponds to `queries[i]`. Empty input returns `[]`. |
| `stats() -> dict` | Return index statistics (`document_count`, `vector_fields`). |

### `search` query argument

The `query` parameter accepts any of the following:

- A **DSL string** (e.g. `"title:hello"`, `"embedding:\"memory safety\""`)
- A **lexical query object** (`TermQuery`, `PhraseQuery`, `BooleanQuery`, …)
- A **vector query object** (`VectorQuery`, `VectorTextQuery`)
- A **`SearchRequest`** for full control

The same value kinds are accepted as the elements of `search_batch`'s `queries` list — DSL strings, query objects, and `SearchRequest` instances may be mixed within a single batch.

### WAL sync policy / durability

For a persistent index, every write is appended to a write-ahead log (WAL).
By default the WAL is `fsync`-ed on **every** record, so each write is fully
durable as soon as the call returns. The constructor accepts an optional
`wal_sync_policy` to trade some durability for higher write throughput, and
`flush_wal()` forces a durable barrier on demand.

```python
class WalSyncPolicy:
    @staticmethod
    def per_record() -> WalSyncPolicy: ...
    @staticmethod
    def group(
        max_records: int | None = None,
        max_bytes: int | None = None,
        max_interval_ms: int | None = None,
    ) -> WalSyncPolicy: ...
```

| Constructor | Description |
| :--- | :--- |
| `WalSyncPolicy.per_record()` | Default. `fsync` after every WAL record; fully durable per write. |
| `WalSyncPolicy.group(...)` | Group commit. Batches `fsync` across writes. |

`group()` parameters (all keyword-friendly, `None` keeps the default):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `max_records` | `1024` | Flush once this many records have accumulated. |
| `max_bytes` | `1048576` (1 MiB) | Flush once this many unsynced bytes have accumulated. |
| `max_interval_ms` | `None` | Optional periodic flush timer (milliseconds). `None` disables the timer. |

With group commit the WAL is flushed when **either** `max_records` or
`max_bytes` is reached, and always at `commit()`. A crash can lose up to the
last unsynced batch — the same trade-off as SQLite's `synchronous = NORMAL`.
Call `flush_wal()` to force everything written so far to disk without a full
`commit()`.

| Method | Description |
| :--- | :--- |
| `flush_wal()` | Force a durable WAL barrier now. Synchronous; returns `None`. |

```python
import laurus

# Opt into group commit with a 1-second periodic flush timer.
policy = laurus.WalSyncPolicy.group(max_records=4096, max_interval_ms=1000)
index = laurus.Index(path="./myindex", wal_sync_policy=policy)

for i in range(10_000):
    index.put_document(f"doc{i}", {"title": f"Document {i}"})

# Force a durable barrier without committing yet.
index.flush_wal()

index.commit()  # also flushes the WAL
```

Omit `wal_sync_policy` (or pass `WalSyncPolicy.per_record()`) to keep the
default, fully durable behaviour.

### Commit policy / auto-commit

A commit materializes buffered writes into the stores and makes all pending
changes searchable. By default an `Index` never auto-commits, so the caller
drives every `commit()` explicitly. The constructor accepts an optional
`commit_policy` to instead let the engine auto-commit after a fixed number of
applied documents, or at least once every fixed time interval.

```python
class CommitPolicy:
    @staticmethod
    def manual() -> CommitPolicy: ...
    @staticmethod
    def every_docs(n: int) -> CommitPolicy: ...
    @staticmethod
    def interval_ms(ms: int) -> CommitPolicy: ...
```

| Constructor | Description |
| :--- | :--- |
| `CommitPolicy.manual()` | Default. No auto-commit; the caller drives every `commit()`. |
| `CommitPolicy.every_docs(n)` | Auto-commit after every `n` applied documents. |
| `CommitPolicy.interval_ms(ms)` | Auto-commit at least every `ms` milliseconds via a background timer (default: none). Native-only; a no-op on wasm. |

`every_docs(n)` counts applied documents across both singular (`put_document`,
`add_document`) and batch (`put_documents`, `add_documents`) ingest, and
auto-commits after every `n` documents — including *within* a single batch.
`every_docs(0)` is valid and disables auto-commit, making it equivalent to
`manual()`.

`interval_ms(ms)` is the time-based counterpart of `every_docs(n)`: a
background timer commits at least every `ms` milliseconds, so a trailing
partial batch is committed even while ingestion is idle. This factory is
**native-only** — on wasm there are no background threads, so the engine treats
it as a **no-op**: the value still constructs, but no timed commit happens under
WebAssembly.

`commit_policy` is orthogonal to `wal_sync_policy`: `wal_sync_policy` governs
WAL `fsync` durability, whereas `commit_policy` governs when the stores
materialize pending changes into searchable state. Setting one does not affect
the other.

```python
import laurus

# Auto-commit after every 100 applied documents.
index = laurus.Index(
    path="./myindex",
    commit_policy=laurus.CommitPolicy.every_docs(100),
)

for i in range(1_000):
    index.put_document(f"doc{i}", {"title": f"Document {i}"})
# The engine has already committed 10 times (once per 100 documents).
```

Omit `commit_policy` (or pass `CommitPolicy.manual()`) to keep the default,
manual-commit behaviour.

---

## Schema

Defines the fields and index types for an `Index`.

```python
class Schema:
    def __init__(self) -> None: ...
```

### Field methods

| Method | Description |
| :--- | :--- |
| `add_text_field(name, *, stored=True, indexed=True, term_vectors=True, analyzer=None)` | Full-text field (inverted index, BM25). `term_vectors` controls whether term positions are stored, read by phrase and span queries. `analyzer` accepts a built-in name (`"standard"`, `"english"`, `"keyword"`, `"simple"`, `"noop"`, or any custom name registered via `add_analyzer`) or a dict configuring a parameterised preset such as `{"language": "japanese", "mode": "normal", "dict": "/var/lib/lindera/ipadic"}`. The bare string `"japanese"` is rejected because the preset requires a Lindera dictionary path. |
| `add_integer_field(name, *, stored=True, indexed=True, multi_valued=False)` | 64-bit integer field. Set `multi_valued=True` to accept arrays of integers (range queries match if any value satisfies the predicate). |
| `add_float_field(name, *, stored=True, indexed=True, multi_valued=False)` | 64-bit float field. Set `multi_valued=True` to accept arrays of floats (range queries match if any value satisfies the predicate). |
| `add_boolean_field(name, *, stored=True, indexed=True)` | Boolean field. |
| `add_bytes_field(name, *, stored=True)` | Raw bytes field. |
| `add_geo_field(name, *, stored=True, indexed=True)` | Geographic coordinate field (lat/lon). |
| `add_geo3d_field(name, *, stored=True, indexed=True)` | 3D ECEF Cartesian point field (x, y, z in metres). See [Geo3d concepts](../concepts/geo3d.md). |
| `add_datetime_field(name, *, stored=True, indexed=True)` | UTC datetime field. |
| `add_hnsw_field(name, dimension, *, distance="cosine", m=16, ef_construction=200, quantizer=None, subvector_count=None, rerank_storage=None, embedder=None, pq_codebook_path=None, base_weight=1.0)` | HNSW approximate nearest-neighbor vector field. `base_weight` sets this field's relative scoring priority when searched alongside other vector fields (Issue #1084); see [Vector Search → Weights](../concepts/search/vector_search.md#weights). |
| `add_flat_field(name, dimension, *, distance="cosine", embedder=None, base_weight=1.0)` | Flat (brute-force) vector field. |
| `add_ivf_field(name, dimension, *, distance="cosine", n_clusters=100, n_probe=1, embedder=None, base_weight=1.0)` | IVF approximate nearest-neighbor vector field. |

**Vector quantization & rerank storage** (HNSW fields):

- `quantizer` — `"scalar_8bit"` (default, 4× compression) or `"product_quantization"` for higher compression. Product quantization requires `subvector_count` (must divide `dimension`).
- `rerank_storage` — set to `"f32"` to write a full-precision `*.hnsw.f32` sidecar enabling exact Stage-2 rerank; omit to keep the int8-only segment.
- `pq_codebook_path` — storage-relative file name of a shared PQ codebook (Issue #631), trained once via the `laurus train pq-codebook` CLI command. Only meaningful with `quantizer="product_quantization"`; commits then encode against the pre-trained codebook instead of re-training k-means per segment. Omit to keep per-segment training.

### Other methods

| Method | Description |
| :--- | :--- |
| `add_embedder(name, config)` | Register a named embedder definition. `config` is a dict with a `"type"` key (see below). |
| `add_analyzer(name, tokenizer, *, char_filters=None, token_filters=None)` | Register a custom analyzer definition. `tokenizer` is required; `char_filters`/`token_filters` are optional lists of dicts. Each dict uses the same `{"type": "..."}` shape as the schema TOML/JSON format (see below). Semantic validity (e.g. a malformed regex) is checked when the schema is used to build an `Index`, not here. |
| `analyzer_names()` | Return the names of custom analyzers registered via `add_analyzer` or loaded from TOML. |
| `Schema.from_toml(toml_str)` *(static)* | Parse a schema from a TOML string, in the same format `laurus-cli create index --schema` accepts. |
| `Schema.from_toml_file(path)` *(static)* | Load a schema from a TOML file (`path` accepts `str` or `os.PathLike`). |
| `to_toml()` | Serialize this schema to a TOML string in the same format `laurus-cli` accepts. |
| `to_toml_file(path)` | Write this schema to a TOML file. |
| `set_default_fields(fields)` | Set default search fields (list of strings). |
| `set_dynamic_field_policy(policy)` | Set how undeclared fields are handled. `policy` is `"strict"`, `"dynamic"` (default), or `"ignore"`. See notes below. |
| `dynamic_field_policy()` | Return the current policy as a lowercase string. |
| `field_names()` | Return all field names. |

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

### Analyzer components

Used by `add_analyzer(name, tokenizer, *, char_filters=None, token_filters=None)`
and by the `[analyzers.<name>]` TOML section. `tokenizer` is a single dict;
`char_filters`/`token_filters` are lists of dicts, applied in list order.

**Tokenizers** (`tokenizer`, exactly one):

| `"type"` | Required keys | Optional keys |
| :--- | :--- | :--- |
| `"whitespace"` | -- | -- |
| `"unicode_word"` | -- | -- |
| `"regex"` | -- | `"pattern"` (default `\w+`), `"gaps"` (default `false`) |
| `"ngram"` | `"min_gram"`, `"max_gram"` | -- |
| `"lindera"` | `"mode"`, `"dict"` | `"user_dict"` |
| `"whole"` | -- | -- |

**Char filters** (`char_filters`, applied to raw text before tokenization):

| `"type"` | Required keys | Optional keys |
| :--- | :--- | :--- |
| `"unicode_normalization"` | `"form"` (`"nfc"`/`"nfd"`/`"nfkc"`/`"nfkd"`) | -- |
| `"pattern_replace"` | `"pattern"`, `"replacement"` | -- |
| `"mapping"` | `"mapping"` (dict of string replacements) | -- |
| `"japanese_iteration_mark"` | -- | `"kanji"` (default `true`), `"kana"` (default `true`) |

**Token filters** (`token_filters`, applied to the token stream after tokenization):

| `"type"` | Required keys | Optional keys |
| :--- | :--- | :--- |
| `"lowercase"` | -- | -- |
| `"stop"` | -- | `"words"` (default: English stop words) |
| `"stem"` | -- | `"stem_type"` (`"porter"`/`"simple"`/`"identity"`) |
| `"boost"` | `"boost"` | -- |
| `"limit"` | `"limit"` | -- |
| `"strip"` | -- | -- |
| `"remove_empty"` | -- | -- |
| `"flatten_graph"` | -- | -- |

```python
schema = laurus.Schema()
schema.add_analyzer(
    "ja_ipadic",
    {"type": "lindera", "mode": "normal", "dict": "/var/lib/lindera/ipadic"},
    char_filters=[
        {"type": "unicode_normalization", "form": "nfkc"},
        {"type": "japanese_iteration_mark"},
    ],
    token_filters=[{"type": "lowercase"}],
)
schema.add_text_field("title", analyzer="ja_ipadic")
```

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

```python
TermQuery(field: str, term: str)
```

Matches documents containing the exact term in the given field.

### PhraseQuery

```python
PhraseQuery(field: str, terms: list[str])
```

Matches documents containing the terms in order.

### FuzzyQuery

```python
FuzzyQuery(field: str, term: str, *, max_edits: int = 2)
```

Approximate match allowing up to `max_edits` edit-distance errors. `max_edits` is keyword-only.

### WildcardQuery

```python
WildcardQuery(field: str, pattern: str)
```

Pattern match. `*` matches any sequence of characters, `?` matches any single character.

### NumericRangeQuery

```python
NumericRangeQuery(field: str, *, min: int | float | None = None, max: int | float | None = None)
```

Matches numeric values in the range `[min, max]`. Pass `None` (or omit) for
an open bound. `min` and `max` are keyword-only. The numeric type (integer or
float) is inferred from the Python type of `min`/`max`.

### GeoDistanceQuery

```python
GeoDistanceQuery.within_radius(
    field: str, lat: float, lon: float, distance_m: float,
)
```

Geo-distance (radius) search. Returns documents whose `(lat, lon)` coordinate
is within `distance_m` metres of the given point.

### GeoBoundingBoxQuery

```python
GeoBoundingBoxQuery.within_bounding_box(
    field: str,
    min_lat: float, min_lon: float,
    max_lat: float, max_lon: float,
)
```

Geo bounding-box search. Returns documents whose `(lat, lon)` coordinate lies
inside the axis-aligned `[min_lat, max_lat] × [min_lon, max_lon]` rectangle.

### Geo3dDistanceQuery

```python
Geo3dDistanceQuery.within_sphere(
    field: str, x: float, y: float, z: float, distance_m: float,
)
```

Sphere search over a 3D ECEF point field. Returns documents whose `(x, y, z)`
coordinate is within `distance_m` metres of the centre. See
[Geo3d concepts](../concepts/geo3d.md) for ECEF theory.

### Geo3dBoundingBoxQuery

```python
Geo3dBoundingBoxQuery.within_box(
    field: str,
    min_x: float, min_y: float, min_z: float,
    max_x: float, max_y: float, max_z: float,
)
```

Axis-aligned 3D bounding-box search. Returns documents whose ECEF point lies
inside `[min_x, max_x] × [min_y, max_y] × [min_z, max_z]`.

### Geo3dNearestQuery

```python
Geo3dNearestQuery.k_nearest(
    field: str,
    x: float, y: float, z: float,
    k: int,
    *,
    initial_radius_m: float | None = None,
    max_radius_m: float | None = None,
)
```

k-nearest-neighbour search over a 3D ECEF point field. Returns the `k`
documents closest to `(x, y, z)`. The optional `initial_radius_m` and
`max_radius_m` tune the iterative-expansion search cone.

### BooleanQuery

```python
bq = BooleanQuery()
bq.must(query)
bq.should(query)
bq.must_not(query)
```

Compound boolean query. Construct with no arguments and add clauses one at a
time via the `must` / `should` / `must_not` methods. Each method accepts any
query object (including a nested `BooleanQuery`).

`must` clauses all have to match; `must_not` clauses must not match.
`should` clauses contribute to scoring; at least one of them must match if
there are no `must` clauses.

### SpanQuery

```python
# Single term
SpanQuery.term(field: str, term: str)

# Near: terms appearing within `slop` positions of each other
SpanQuery.near(field: str, terms: list[str], *, slop: int = 0, ordered: bool = True)

# Near with nested SpanQuery clauses
SpanQuery.near_spans(field: str, clauses: list[SpanQuery], *, slop: int = 0, ordered: bool = True)

# Containing: big span contains little span
SpanQuery.containing(field: str, big: SpanQuery, little: SpanQuery)

# Within: include span within exclude span at max distance
SpanQuery.within(field: str, include: SpanQuery, exclude: SpanQuery, distance: int)
```

Positional / proximity span queries. Construct via the static factory
methods. `near` takes a list of term strings, while `near_spans` takes a
list of `SpanQuery` objects for nested expressions. `slop` and `ordered`
are keyword-only.

### VectorQuery

```python
VectorQuery(field: str, vector: list[float])
```

Approximate nearest-neighbor search using a pre-computed embedding vector.

### VectorTextQuery

```python
VectorTextQuery(field: str, text: str)
```

Converts `text` to an embedding at query time and runs vector search. Requires an embedder configured on the index.

---

## SearchRequest

Full-featured search request for advanced control.

```python
class SearchRequest:
    def __init__(
        self,
        *,
        query=None,
        lexical_query=None,
        vector_query=None,
        filter_query=None,
        fusion=None,
        limit: int = 10,
        offset: int = 0,
    ) -> None: ...
```

| Parameter | Description |
| :--- | :--- |
| `query` | A DSL string or single query object. Mutually exclusive with `lexical_query` / `vector_query`. |
| `lexical_query` | Lexical component for explicit hybrid search. |
| `vector_query` | Vector component for explicit hybrid search. |
| `filter_query` | Lexical filter applied after scoring. |
| `fusion` | Fusion algorithm (`RRF` or `WeightedSum`). Defaults to `RRF(k=60)` when both components are set. |
| `limit` | Maximum number of results (default 10). |
| `offset` | Pagination offset (default 0). |

---

## SearchResult

Returned by `Index.search()`.

```python
class SearchResult:
    id: str          # External document identifier
    score: float     # Relevance score
    document: dict | None  # Retrieved field values, or None if not stored
```

---

## Fusion algorithms

### RRF

```python
RRF(k: float = 60.0)
```

Reciprocal Rank Fusion. Merges lexical and vector result lists by rank position. `k` is a smoothing constant; higher values reduce the influence of top-ranked results.

### WeightedSum

```python
WeightedSum(lexical_weight: float = 0.5, vector_weight: float = 0.5)
```

Normalises both score lists independently, then combines them as `lexical_weight * lexical_score + vector_weight * vector_score`.

---

## Text analysis

### SynonymDictionary

```python
class SynonymDictionary:
    def __init__(self) -> None: ...
    def add_synonym_group(self, synonyms: list[str]) -> None: ...
```

### WhitespaceTokenizer

```python
class WhitespaceTokenizer:
    def __init__(self) -> None: ...
    def tokenize(self, text: str) -> list[Token]: ...
```

### SynonymGraphFilter

```python
class SynonymGraphFilter:
    def __init__(
        self,
        dictionary: SynonymDictionary,
        keep_original: bool = True,
        boost: float = 1.0,
    ) -> None: ...
    def apply(self, tokens: list[Token]) -> list[Token]: ...
```

### Token

```python
class Token:
    text: str
    position: int
    start_offset: int
    end_offset: int
    boost: float
    stopped: bool
    position_increment: int
    position_length: int
```

---

## Field value types

Python values are automatically converted to Laurus `DataValue` types:

| Python type | Laurus type | Notes |
| :--- | :--- | :--- |
| `None` | `Null` | |
| `bool` | `Bool` | Checked before `int` |
| `int` | `Int64` | |
| `float` | `Float64` | |
| `str` | `Text` | |
| `bytes` | `Bytes` | |
| `list[float]` | `Vector` | Elements coerced to `f32` |
| `(lat, lon)` tuple | `Geo` | Two `float` values |
| `(x, y, z)` tuple | `Geo3d` | Three `float` values (ECEF Cartesian, metres) |
| `datetime.datetime` | `DateTime` | Converted via `isoformat()` |
