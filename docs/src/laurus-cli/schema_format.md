# Schema Format Reference

The schema file defines the structure of your index — what fields exist, their types, and how they are indexed. Laurus uses TOML format for schema files.

## Overview

A schema consists of five top-level elements:

```toml
# Policy for fields not declared below. Optional — defaults to "dynamic".
dynamic_field_policy = "dynamic"

# Fields to search by default when a query does not specify a field.
default_fields = ["title", "body"]

# Custom analyzer definitions, referenced by name from Text fields. Optional.
[analyzers.<analyzer_name>]
# ... tokenizer, char_filters, token_filters

# Embedder definitions, referenced by name from vector fields. Optional.
[embedders.<embedder_name>]
# ... type and type-specific options

# Field definitions. Each field has a name and a typed configuration.
[fields.<field_name>.<FieldType>]
# ... type-specific options
```

- **`dynamic_field_policy`** — How the engine treats fields present in an ingested document but **absent** from this schema. Accepted values: `"strict"`, `"dynamic"`, `"ignore"`. Defaults to `"dynamic"`. See [Dynamic Schema](../concepts/schema_and_fields.md#dynamic-schema) for the full semantics and the warning about silent truncation under `"dynamic"`.
- **`default_fields`** — A list of field names used as default search targets by the [Query DSL](../concepts/query_dsl.md). Only lexical fields (Text, Integer, Float, etc.) can be default fields. This key is optional and defaults to an empty list.
- **`analyzers`** — A map of names to custom text-analysis pipelines. A Text field uses one by naming it in its `analyzer` option. Optional. See [Analyzers](#analyzers).
- **`embedders`** — A map of names to embedding models. A vector field uses one by naming it in its `embedder` option. Optional. See [Embedders](#embedders).
- **`fields`** — A map of field names to their typed configuration. Each field must specify exactly one field type.

## Field Naming

- Field names are arbitrary strings (e.g., `title`, `body_vec`, `created_at`).
- **Field names starting with `_` are reserved** for the engine. The only allow-listed name is `_id` (managed automatically). Declaring any other `_`-prefixed field is rejected when the index is created, with `Field name '_score' is reserved: names starting with '_' are reserved for system fields (allowed: '_id')`; `create index` creates nothing. An index created before this check keeps opening, and the field stays unusable — a document that sets it is still rejected at ingestion, exactly as it is today.
- Field names must be unique within a schema.

## Field Types

Fields fall into two categories: **Lexical** (for keyword/full-text search) and **Vector** (for similarity search). A single field cannot be both.

### Lexical Fields

#### Text

Full-text searchable field. Text is processed by the analysis pipeline (tokenization, normalization, stemming, etc.).

```toml
[fields.title.Text]
indexed = true               # Whether to index this field for search
stored = true                # Whether to store the original value for retrieval
multi_valued = false         # Whether to accept arrays of strings (Issue #1175)
position_increment_gap = 100 # Positions skipped between the elements of a multi-valued field
term_vectors = true          # Whether to store term positions (for phrase and span queries)
doc_values = true            # Whether to also copy the value into DocValues (for sorting/faceting)
analyzer = "standard"        # Analyzer for indexing and querying this field
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables searching this field |
| `stored` | `bool` | `true` | Stores the original value so it can be returned in results |
| `multi_valued` | `bool` | `false` | Accept arrays of strings; a term query matches if **any** element contains the term (Lucene-style "any match"); a phrase query never spans two elements unless its slop reaches `position_increment_gap` |
| `position_increment_gap` | `integer` | `100` | Positions skipped between the elements of a multi-valued field (Lucene `positionIncrementGap`); `0` numbers the elements as if concatenated. Ignored unless `multi_valued = true` |
| `term_vectors` | `bool` | `true` | Stores term positions, read by phrase and span queries; highlighting always re-tokenizes the stored text and does not use them |
| `doc_values` | `bool` | `true` | Copies the value into DocValues, the column-oriented store [sorting](../laurus/faceting.md) and faceting/aggregation read from. Takes effect only when `stored` is also `true` — see [Common option: `doc_values`](#common-option-doc_values) below |
| `analyzer` | `string` or table | *(omit)* | Analyzer used both when indexing and when parsing queries against this field. A string names a built-in analyzer (`"standard"`, `"english"`, `"keyword"`, `"simple"`, `"noop"`) or an entry in [`[analyzers.*]`](#analyzers). A table selects a parameterized built-in preset; today only `{ language = "japanese", mode = "normal", dict = "<path>" }` (see [Text Analysis](../concepts/analysis.md#configuring-per-field-analyzers-from-a-schema)). When omitted, `"standard"` is used |

The interactive generator (`laurus create schema`, see [Generating a Schema](#generating-a-schema)) asks whether a Text field is multi-valued and, when it is, for its position increment gap.

#### Integer

64-bit signed integer field. Supports range queries and exact match.

```toml
[fields.year.Integer]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables range and exact-match queries |
| `stored` | `bool` | `true` | Stores the original value |
| `multi_valued` | `bool` | `false` | Accept arrays of integers; range queries match if **any** value satisfies the predicate (Lucene-style "any match" with constant scoring) |
| `doc_values` | `bool` | `true` | See [Common option: `doc_values`](#common-option-doc_values) below |

#### Float

64-bit floating point field. Supports range queries.

```toml
[fields.rating.Float]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables range queries |
| `stored` | `bool` | `true` | Stores the original value |
| `multi_valued` | `bool` | `false` | Accept arrays of floats; range queries match if **any** value satisfies the predicate (Lucene-style "any match" with constant scoring) |
| `doc_values` | `bool` | `true` | See [Common option: `doc_values`](#common-option-doc_values) below |

#### Boolean

Boolean field (`true` / `false`).

```toml
[fields.published.Boolean]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables filtering by boolean value |
| `stored` | `bool` | `true` | Stores the original value |
| `multi_valued` | `bool` | `false` | Accept arrays of booleans; a term query (`flags:true`) matches if **any** element equals the queried value (Lucene-style "any match"); repeated elements raise the term frequency, not the hit count |
| `doc_values` | `bool` | `true` | See [Common option: `doc_values`](#common-option-doc_values) below |

#### DateTime

UTC timestamp field. Supports range queries.

```toml
[fields.created_at.DateTime]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables range queries on date/time |
| `stored` | `bool` | `true` | Stores the original value |
| `multi_valued` | `bool` | `false` | Accept arrays of instants; range queries match if **any** instant satisfies the predicate (Lucene-style "any match") |
| `doc_values` | `bool` | `true` | See [Common option: `doc_values`](#common-option-doc_values) below |

#### Geo

Geographic point field (latitude/longitude). Supports radius and bounding box queries.

```toml
[fields.location.Geo]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables geo queries (radius, bounding box) |
| `stored` | `bool` | `true` | Stores the original value |
| `multi_valued` | `bool` | `false` | Accept arrays of points; distance / bounding-box queries match if **any** point satisfies the predicate (Lucene-style "any match"), scoring the document by its closest point |
| `doc_values` | `bool` | `true` | See [Common option: `doc_values`](#common-option-doc_values) below |

#### Geo3d

3D Earth-Centered Earth-Fixed (ECEF) Cartesian point field (x / y / z in meters). Supports the `geo3d_distance` (sphere), `geo3d_bbox` (3D AABB), and `geo3d_nearest` (k-NN) queries. See [3D Geographic Search (ECEF)](../concepts/geo3d.md) for the coordinate system and the `wgs84_to_ecef` / `ecef_to_wgs84` conversion utilities.

```toml
[fields.position.Geo3d]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables 3D geo queries (`geo3d_distance`, `geo3d_bbox`, `geo3d_nearest`) |
| `stored` | `bool` | `true` | Stores the original `(x, y, z)` value |
| `multi_valued` | `bool` | `false` | Accept arrays of points; `geo3d_distance` / `geo3d_bbox` / `geo3d_nearest` queries match if **any** point satisfies the predicate (Lucene-style "any match"), scoring the document by its closest point |
| `doc_values` | `bool` | `true` | See [Common option: `doc_values`](#common-option-doc_values) below |

#### Bytes

Raw binary data field. Not indexed — stored only.

```toml
[fields.thumbnail.Bytes]
stored = true
multi_valued = false
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `stored` | `bool` | `true` | Stores the binary data |
| `multi_valued` | `bool` | `false` | Accept arrays of byte strings; a `Bytes` field is never indexed, so unlike every other `multi_valued` option this has no "any match" query semantics — it only governs the stored shape and ingestion arity |

`BytesOption` has no `doc_values` setting: a `Bytes` value is never written to
DocValues regardless, since neither sorting nor faceting can do anything with
it.

#### Common option: `doc_values`

Every lexical field option above except `BytesOption` carries a `doc_values`
option, controlling whether the value is also copied into DocValues — the
column-oriented store [sorting](../laurus/faceting.md) and faceting/aggregation
read from. The effective rule: a DocValues column is written only when
`stored` and `doc_values` are both `true`. Setting `doc_values: false` with
`stored: false` is silently ignored (not an error). Turning `doc_values` off
for a field that is never sorted or faceted on shrinks its segment footprint,
since the value is then written once (to the stored document) instead of
twice; the field remains fully searchable and retrievable either way — sorting
and faceting on it simply fall back to the stored document.

### Vector Fields

Vector fields are indexed for approximate nearest neighbor (ANN) search. They require a `dimension` (the length of each vector) and a `distance` metric.

#### Hnsw

Hierarchical Navigable Small World graph index. Best for most use cases — offers a good balance of speed and recall.

```toml
[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
m = 16
ef_construction = 200
base_weight = 1.0
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | `128` | Vector dimensionality (must match your embedding model) |
| `distance` | `string` | `"Cosine"` | Distance metric (see [Distance Metrics](#distance-metrics)) |
| `m` | `integer` | `16` | Max bi-directional connections per node. Higher = better recall, more memory |
| `ef_construction` | `integer` | `200` | Search width during index construction. Higher = better quality, slower build |
| `base_weight` | `float` | `1.0` | Relative priority vs. other vector fields searched together; no effect on lexical-vs-vector fusion balance (see [Vector Search → Weights](../concepts/search/vector_search.md#weights)) |
| `quantizer` | `object` | `"Scalar8Bit"` | Quantization method (see [Quantization](#quantization)). Mandatory; default keeps the int8 format introduced in Issue #481 Stage 1. |
| `rerank_storage` | `string` | *(omit)* | Optional Stage 2 rerank sidecar (see [Rerank Storage](#rerank-storage)). `"F32"` enables a per-field f32 sidecar so search can rescore int8 candidates against the original vectors. Omit to keep Stage 1 int8-only behavior. |
| `pq_codebook_path` | `string` | *(omit)* | Storage-relative file name of a shared PQ codebook (Issue #631); only meaningful with a `ProductQuantization` quantizer. Train it with `laurus train pq-codebook`; commits then encode against it instead of re-training k-means per segment. When set but not yet trained, commits fail loudly (no silent fallback). Omit to train per segment. |
| `embedder` | `string` | *(omit)* | Name of an entry in [`[embedders.*]`](#embedders). Text (or image) values given for this field are then embedded with that model, both when indexing and when searching. Omit to supply precomputed vectors only |

**Tuning guidelines:**

- `m`: 12–48 is typical. Use higher values for higher-dimensional vectors.
- `ef_construction`: 100–500. Higher values produce a better graph but increase build time.
- `dimension`: Must exactly match the output dimension of your embedding model (e.g., 384 for `all-MiniLM-L6-v2`, 768 for `BERT-base`, 1536 for `text-embedding-3-small`).

#### Flat

Brute-force linear scan index. Provides exact results with no approximation. Best for small datasets (< 10,000 vectors).

```toml
[fields.embedding.Flat]
dimension = 384
distance = "Cosine"
base_weight = 1.0
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | `128` | Vector dimensionality |
| `distance` | `string` | `"Cosine"` | Distance metric (see [Distance Metrics](#distance-metrics)) |
| `base_weight` | `float` | `1.0` | Relative priority vs. other vector fields searched together; no effect on lexical-vs-vector fusion balance (see [Vector Search → Weights](../concepts/search/vector_search.md#weights)) |
| `quantizer` | `object` | `"Scalar8Bit"` | Quantization method (see [Quantization](#quantization)). Mandatory; default keeps the int8 format introduced in Issue #481 Stage 1. |
| `rerank_storage` | `string` | *(omit)* | Optional Stage 2 rerank sidecar (see [Rerank Storage](#rerank-storage)); supported by all three vector index types since #932. `"F32"` enables the per-field f32 sidecar so search can rescore int8 candidates against the original vectors. |
| `embedder` | `string` | *(omit)* | Name of an entry in [`[embedders.*]`](#embedders); see [Hnsw](#hnsw) |

#### Ivf

Inverted File Index. Clusters vectors and searches only a subset of clusters. Suitable for very large datasets.

```toml
[fields.embedding.Ivf]
dimension = 384
distance = "Cosine"
n_clusters = 100
n_probe = 1
base_weight = 1.0
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | *(required)* | Vector dimensionality |
| `distance` | `string` | `"Cosine"` | Distance metric (see [Distance Metrics](#distance-metrics)) |
| `n_clusters` | `integer` | `100` | Number of clusters. More clusters = finer partitioning |
| `n_probe` | `integer` | `1` | Number of clusters to search at query time. Higher = better recall, slower |
| `base_weight` | `float` | `1.0` | Relative priority vs. other vector fields searched together; no effect on lexical-vs-vector fusion balance (see [Vector Search → Weights](../concepts/search/vector_search.md#weights)) |
| `quantizer` | `object` | `"Scalar8Bit"` | Quantization method (see [Quantization](#quantization)). Mandatory; default keeps the int8 format introduced in Issue #481 Stage 1. |
| `rerank_storage` | `string` | *(omit)* | Optional Stage 2 rerank sidecar (see [Rerank Storage](#rerank-storage)); supported by all three vector index types since #932. `"F32"` enables the per-field f32 sidecar so search can rescore int8 candidates against the original vectors. |
| `embedder` | `string` | *(omit)* | Name of an entry in [`[embedders.*]`](#embedders); see [Hnsw](#hnsw) |

> **Note:** Unlike Hnsw and Flat, the `dimension` field in Ivf is **required** and has no default value.

**Tuning guidelines:**

- `n_clusters`: A common heuristic is `sqrt(N)` where N is the total number of vectors.
- `n_probe`: Start with 1 and increase until recall is acceptable. Typical range is 1–20.

#### MultiVector

Every token vector of a document (for example ColBERT-style per-token embeddings), kept for late-interaction rescoring. It has no ANN index and is not a vector-search target; see [Multi-Vector Fields](../concepts/schema_and_fields.md#multi-vector-fields).

```toml
[fields.body_colbert.MultiVector]
dimension = 128
distance = "Cosine"
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | `128` | Dimensionality of every token vector |
| `distance` | `string` | `"Cosine"` | Token similarity: `"Cosine"` (vectors are L2-normalized when written) or `"DotProduct"`. Other metrics are rejected |
| `embedder` | `string` | -- | Name of a token-level embedder (`type = "candle_colbert"`) declared under `[embedders]`. When set, documents may give the field text, which is embedded into token vectors (see [Embedders](#embedders)) |

A document's value is an array of equal-length numeric arrays, between 1 and 8,192 of them, or text when the field has an embedder. The field is not stored in the document store.

## Distance Metrics

The `distance` option for vector fields accepts the following values:

| Value | Description | Use When |
| :--- | :--- | :--- |
| `"Cosine"` | Cosine distance (1 - cosine similarity). Default. | Normalized text/image embeddings |
| `"Euclidean"` | L2 (Euclidean) distance | Spatial data, non-normalized vectors |
| `"Manhattan"` | L1 (Manhattan) distance | Sparse feature vectors |
| `"DotProduct"` | Dot product (higher = more similar) | Pre-normalized vectors where magnitude matters |
| `"Angular"` | Angular distance | Similar to cosine, but based on angle |

For most embedding models (BERT, Sentence Transformers, OpenAI, etc.), `"Cosine"` is the correct choice.

## Quantization

Vector fields are stored on disk as **8-bit scalar-quantized integers**
(Issue #481 Stage 1). Quantization is mandatory; the previous "no
quantization" mode no longer exists. The `quantizer` option defaults to
`Scalar8Bit` and can be omitted from TOML.

### Scalar 8-bit (default)

Per-segment global affine quantization to `u8`. Compresses each `f32`
component to a single byte (~4x memory reduction) with negligible
recall loss in practice.

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"
# quantizer = "Scalar8Bit"  # implicit default; can be omitted
```

### Product Quantization (HNSW-only)

Issue #481 Stage 3. Stores each vector as `subvector_count` one-byte
centroid indexes against a codebook of 256 centroids per sub-vector
(~16-64x compression). Supported by the HNSW index; Flat / IVF reject
it at write time. Usually paired with
[Rerank Storage](#rerank-storage) to recover recall.

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"
# Optional (Issue #631): train the codebook once with
# `laurus train pq-codebook` and share it across segments instead of
# re-training k-means on every commit and merge.
pq_codebook_path = "embedding.pqcb"

[fields.embedding.Hnsw.quantizer.ProductQuantization]
subvector_count = 48
```

| Option | Type | Description |
| :--- | :--- | :--- |
| `subvector_count` | `integer` | Number of subvectors. Must evenly divide `dimension`. |

By default the codebook is trained per segment (segments with fewer
than 256 vectors fall back to `Scalar8Bit`). With `pq_codebook_path`
set, segments encode against the shared pre-trained codebook instead:
commits get dramatically faster, and even tiny per-commit segments
stay on PQ — but a commit before the codebook has been trained fails
with an error naming the `laurus train pq-codebook` command to run
(never a silent fallback to per-segment training). See the
[`train` command](commands.md#train) for the training workflow.

> **Breaking change (Issue #481 Stage 1):** schemas that explicitly
> set `quantizer` to a "none" value are no longer valid. Existing
> vector indexes built with a pre-Stage-1 laurus build cannot be
> read; rebuild from source data after upgrading.

## Rerank Storage

Optional Stage 2 sidecar (Issue #481) that keeps the original
full-precision vectors alongside the int8 segment so the searcher
can do a wide candidate fetch over int8 (cheap) and then rescore
the top `top_k * rerank_factor` candidates against the exact f32
values (accurate). Supported by all three vector index types —
HNSW, Flat, and IVF (#932); on Flat/IVF the rescoring applies to
field-routed queries.

The sidecar is configured per field with `rerank_storage`:

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"
rerank_storage = "F32"  # opt-in; omit for Stage 1 int8-only behavior
```

| Value | On-disk overhead | Description |
| :--- | :--- | :--- |
| `"F32"` | +4 bytes/dim per vector | IEEE-754 single-precision sidecar (Lucene 99 / FAISS convention). |

When omitted, no sidecar is written and the field stays on the
Stage 1 int8-only search path. Queries that pass `rerank_factor`
against a field without `rerank_storage` silently fall back to
Stage 1 ranking — the searcher cannot recover f32 information that
was discarded at index time.

> **Scope:** Stage 2 lands HNSW only. Flat / IVF accept the field
> for schema symmetry but currently neither emit nor consume the
> sidecar.

## Analyzers

An `[analyzers.<name>]` table defines a custom text-analysis pipeline. A Text field uses it by naming it in its `analyzer` option. Define one when no built-in analyzer fits — for example, to add stemming, or to analyze Japanese text without the stop filter of the `japanese` preset. See [Text Analysis](../concepts/analysis.md) for how the pipeline works.

```toml
[analyzers.<name>]
char_filters = [{ type = "...", ... }, ...]   # optional
tokenizer = { type = "...", ... }             # required
token_filters = [{ type = "...", ... }, ...]  # optional
```

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tokenizer` | table | *(required)* | Splits the text into tokens. Exactly one |
| `char_filters` | array of tables | `[]` | Applied to the raw text before tokenization, in array order |
| `token_filters` | array of tables | `[]` | Applied to the token stream after tokenization, in array order |

Each component is a table whose `type` key selects the component; its other keys configure it. Inline tables (`{ type = "lowercase" }`) are the usual TOML spelling. The same shape is used in JSON schemas (`{"type": "lowercase"}`) and by every binding's `addAnalyzer` / `add_analyzer`.

### Tokenizers

| `type` | Required keys | Optional keys | Description |
| :--- | :--- | :--- | :--- |
| `"whitespace"` | -- | -- | Splits on whitespace |
| `"unicode_word"` | -- | -- | Splits on Unicode word boundaries |
| `"regex"` | -- | `pattern` (default `\w+`), `gaps` (default `false`) | Emits each match of `pattern` as a token. With `gaps = true`, `pattern` matches the separators between tokens instead |
| `"ngram"` | `min_gram`, `max_gram` | -- | Emits every n-gram from `min_gram` to `max_gram` characters long |
| `"lindera"` | `mode`, `dict` | `user_dict` | Morphological analysis with [Lindera](https://github.com/lindera/lindera). `mode` is `"normal"` or `"decompose"`. `dict` is the path to a Lindera dictionary directory and `user_dict` the path to a user dictionary; laurus does not embed a dictionary, so `dict` must exist on disk |
| `"whole"` | -- | -- | Emits the whole input as a single token |

### Char filters

| `type` | Required keys | Optional keys | Description |
| :--- | :--- | :--- | :--- |
| `"unicode_normalization"` | `form` (`"nfc"` / `"nfd"` / `"nfkc"` / `"nfkd"`) | -- | Applies Unicode normalization |
| `"pattern_replace"` | `pattern`, `replacement` | -- | Replaces each match of the regular expression `pattern` with `replacement` |
| `"mapping"` | `mapping` (a table of string replacements) | -- | Replaces each key of `mapping` with its value |
| `"japanese_iteration_mark"` | -- | `kanji` (default `true`), `kana` (default `true`) | Expands Japanese iteration marks (踊り字) |

### Token filters

| `type` | Required keys | Optional keys | Description |
| :--- | :--- | :--- | :--- |
| `"lowercase"` | -- | -- | Lowercases each token |
| `"stop"` | -- | `words` (default: English stop words) | Removes stop words |
| `"stem"` | -- | `stem_type` (`"porter"` (default) / `"simple"` / `"identity"`) | Reduces each token to its stem |
| `"boost"` | `boost` | -- | Multiplies each token's boost by `boost` |
| `"limit"` | `limit` | -- | Keeps at most `limit` tokens |
| `"strip"` | -- | -- | Trims leading and trailing whitespace from each token |
| `"remove_empty"` | -- | -- | Removes empty tokens |
| `"flatten_graph"` | -- | -- | Flattens a token graph into a linear stream. Indexing already does this; because the analyzer also parses queries, adding it makes quoted multi-word synonyms inexact at query time |

### Referencing an analyzer

A Text field names an analyzer in its `analyzer` option:

```toml
[fields.body.Text]
analyzer = "english_stemmed"
```

The name is resolved in this order:

1. An analyzer registered at runtime through a binding (for example the WASM binding's `addAnalyzer`)
2. A built-in analyzer: `standard`, `keyword`, `english`, `simple`, or `noop`
3. An entry in `[analyzers.*]`

Because built-ins are checked first, the names `standard`, `keyword`, `english`, `simple` and `noop` are reserved: an `[analyzers.*]` entry under one of them could never be used, so it is rejected. `japanese` is not reserved, because its built-in needs a dictionary and is selected with a table, so `[analyzers.japanese]` is used as defined.

Errors surface at three points:

- An unknown `type` or a missing required key is rejected when the schema is parsed; `create index` creates nothing.
- An entry named after a built-in is rejected when the index is created, with `Analyzer name 'standard' is reserved for a built-in analyzer; choose another name`; `create index` creates nothing. Every binding's `addAnalyzer` / `add_analyzer` (WASM: `addAnalyzerDefinition`) raises the same error. An index created before this check keeps opening and the entry stays unused; opening it emits a warning through the `log` crate, which `laurus-server` prints in its log.
- An invalid value (a malformed regular expression, an unknown `form` or `stem_type`, a missing Lindera dictionary) or an `analyzer` name that resolves to nothing is rejected when the index is built, with an error such as `Failed to resolve analyzer for field 'body': ...`; `create index` creates nothing — `schema.toml` and `store/` are rolled back to whatever state (if any) existed before the call.

### Example: English text with stemming

```toml
default_fields = ["title", "body"]

[analyzers.english_stemmed]
char_filters = [{ type = "unicode_normalization", form = "nfkc" }]
tokenizer = { type = "unicode_word" }
token_filters = [
    { type = "lowercase" },
    { type = "stop" },
    { type = "stem", stem_type = "porter" },
]

[fields.title.Text]
analyzer = "english_stemmed"

[fields.body.Text]
analyzer = "english_stemmed"

[fields.tag.Text]
analyzer = "keyword"
```

A document whose `body` is `"Ｄｏｇｓ are RUNNING in the park."` then matches `body:dog` (NFKC normalization, lowercasing, and stemming) and `body:run`, while `body:the` matches nothing (stop words are removed). The `tag` field keeps the built-in `keyword` analyzer, so it matches only its exact value.

### Example: Japanese text with Lindera

This definition, taken from `examples/aozora/schema.toml`, differs from the `{ language = "japanese" }` preset in that it has no stop filter, so particles such as の and は stay in the index:

```toml
[analyzers.ja_ipadic]
tokenizer = { type = "lindera", mode = "normal", dict = "/var/lib/lindera/ipadic" }
char_filters = [
    { type = "unicode_normalization", form = "nfkc" },
    { type = "japanese_iteration_mark", kanji = true, kana = true },
]
token_filters = [{ type = "lowercase" }]

[fields.title.Text]
analyzer = "ja_ipadic"
```

`dict` must point to an unpacked Lindera dictionary (typically IPADIC). If it does not exist, `create index` fails with `Failed to load dictionary: ... Dictionary path does not exist`.

## Embedders

An `[embedders.<name>]` table declares an embedding model. A vector field (Hnsw, Flat, Ivf, or MultiVector) uses it by naming it in its `embedder` option. Text (or, for CLIP, image) values given for that field are then converted to vectors with the model, both when documents are indexed and when a query targets the field. Several fields can share one embedder. See [Embeddings](../concepts/embedding.md) for how each model works and how to choose one.

```toml
[embedders.<name>]
type = "..."   # required
model = "..."  # required for every type except "precomputed"
```

| `type` | Required keys | Feature flag | Description |
| :--- | :--- | :--- | :--- |
| `"precomputed"` | -- | *(always available)* | Performs no embedding; documents supply the vectors directly |
| `"candle_bert"` | `model` | `embeddings-candle` | Local text embedding with a BERT-family model from Hugging Face Hub, such as `"sentence-transformers/all-MiniLM-L6-v2"` |
| `"candle_clip"` | `model` | `embeddings-multimodal` | Local text and image embedding with a CLIP model from Hugging Face Hub, such as `"openai/clip-vit-base-patch32"` |
| `"openai"` | `model` | `embeddings-openai` | Text embedding through the OpenAI API, such as `"text-embedding-3-small"`. The API key is read from the `OPENAI_API_KEY` environment variable when the engine starts and is never stored in the schema |
| `"candle_colbert"` | `model` | `embeddings-candle` | Local token-level embedding with a BERT-based ColBERT checkpoint, such as `"colbert-ir/colbertv2.0"` or `"answerdotai/answerai-colbert-small-v1"`. Only a MultiVector field can use it |

`"candle_colbert"` also takes these optional keys:

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `revision` | `string` | default branch | Branch, tag or commit of the model repository. Pin a commit: documents not yet committed are embedded again from the write-ahead log on recovery, and a changed model would produce different vectors |
| `query_maxlen` | `integer` | from the checkpoint (`32`) | Number of tokens every query is padded or truncated to |
| `doc_maxlen` | `integer` | from the checkpoint (`180` for colbertv2.0, `300` for answerai-colbert-small-v1) | Maximum number of tokens of a document |

Hugging Face models are downloaded on first use. `candle_bert` and `candle_clip` cache them under `$HF_HOME` (default `~/.cache/huggingface`); `candle_colbert` uses the Hugging Face default cache, `$HF_HOME/hub` (default `~/.cache/huggingface/hub`), which Python tools share. The vector field's `dimension` must equal the model's output dimension; for `candle_colbert` this is checked when the engine starts.

An embedder must fit the field that names it: a MultiVector field accepts only `"candle_colbert"` or `"precomputed"`, and an Hnsw, Flat or Ivf field does not accept `"candle_colbert"`. Any other combination is rejected by `create index`, `add field` and `update field`.

> **Note:** The prebuilt release binaries are built with `--features embeddings-all`. A `laurus` binary installed with `cargo install laurus-cli` or built from source enables none of the embedding features unless you pass them (for example `cargo install laurus-cli --features embeddings-candle`), so only `"precomputed"` works. See [Installation](installation.md) and [Feature Flags](../development/feature_flags.md). A schema that names a type whose feature is missing still parses, but `create index` fails:
>
> ```text
> Error: Not implemented: candle_bert embedder requires the 'embeddings-candle' feature to be enabled
> ```

A vector field's `embedder` must name an `[embedders.*]` entry. An undeclared name is rejected by `create index`, `add field`, and `update field`, and an existing index whose `schema.toml` holds one fails to open:

```text
Error: Invalid argument: Unknown embedder 'missing' for field 'vec': not defined in schema.embedders
```

To open such an index, edit its `schema.toml`. Declaring the name with `type = "precomputed"` keeps the field working as it did, with documents supplying its vectors. Deleting the field's `embedder` line does the same.

### Example: one embedder shared by two fields

```toml
[embedders.text_embedder]
type = "candle_bert"
model = "sentence-transformers/all-MiniLM-L6-v2"

[fields.title_vec.Hnsw]
dimension = 384
distance = "Cosine"
embedder = "text_embedder"

[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
embedder = "text_embedder"
```

### Example: ColBERT token vectors for late-interaction rescoring

```toml
[embedders.colbert]
type = "candle_colbert"
model = "answerdotai/answerai-colbert-small-v1"
revision = "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e"

[fields.body.Text]
indexed = true
stored = true

[fields.body_colbert.MultiVector]
dimension = 96
distance = "Cosine"
embedder = "colbert"
```

Documents then give `body_colbert` the same text as `body`, and a
[late-interaction rescore](../concepts/search/vector_search.md#late-interaction-rescore)
can take its query as text.

## Complete Examples

### Full-text search only

A simple blog post index with lexical search:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
indexed = true
stored = true
term_vectors = true

[fields.body.Text]
indexed = true
stored = true
term_vectors = true

[fields.category.Text]
indexed = true
stored = true
term_vectors = false

[fields.published_at.DateTime]
indexed = true
stored = true
```

### Vector search only

A vector-only index for semantic similarity:

```toml
[fields.embedding.Hnsw]
dimension = 768
distance = "Cosine"
m = 16
ef_construction = 200
```

### Hybrid search (lexical + vector)

Combine lexical and vector search for best-of-both-worlds retrieval:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
indexed = true
stored = true
term_vectors = true

[fields.body.Text]
indexed = true
stored = true
term_vectors = true

[fields.category.Text]
indexed = true
stored = true
term_vectors = false

[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
m = 16
ef_construction = 200
```

> **Tip:** A single field cannot be both lexical and vector. Use separate fields (e.g., `body` for text, `body_vec` for embedding) and map them both to the same source content.

### E-commerce product index

A more complex schema with mixed field types:

```toml
default_fields = ["name", "description"]

[fields.name.Text]
indexed = true
stored = true
term_vectors = true

[fields.description.Text]
indexed = true
stored = true
term_vectors = true

[fields.price.Float]
indexed = true
stored = true

[fields.in_stock.Boolean]
indexed = true
stored = true

[fields.created_at.DateTime]
indexed = true
stored = true

[fields.location.Geo]
indexed = true
stored = true

[fields.description_vec.Hnsw]
dimension = 384
distance = "Cosine"
```

### Custom analysis and automatic embedding

A hybrid index whose Text fields use a custom analyzer and whose vector field embeds its text with a local model. It needs a `laurus` binary with the `embeddings-candle` feature (see [Embedders](#embedders)):

```toml
default_fields = ["title", "body"]

[analyzers.english_stemmed]
tokenizer = { type = "unicode_word" }
token_filters = [
    { type = "lowercase" },
    { type = "stop" },
    { type = "stem" },
]

[embedders.text_embedder]
type = "candle_bert"
model = "sentence-transformers/all-MiniLM-L6-v2"

[fields.title.Text]
analyzer = "english_stemmed"

[fields.body.Text]
analyzer = "english_stemmed"

[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
embedder = "text_embedder"
```

## Generating a Schema

You can generate a schema TOML file interactively using the CLI:

```bash
laurus create schema
laurus create schema --output my_schema.toml
```

See [`create schema`](commands.md#create-schema) for details.

## Using a Schema

Once you have a schema file, create an index from it:

```bash
laurus create index --schema schema.toml
```

Or load it programmatically in Rust:

```rust
use laurus::Schema;

let toml_str = std::fs::read_to_string("schema.toml")?;
let schema: Schema = toml::from_str(&toml_str)?;
```
