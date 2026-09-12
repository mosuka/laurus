# Schema Format Reference

The schema file defines the structure of your index — what fields exist, their types, and how they are indexed. Laurus uses TOML format for schema files.

## Overview

A schema consists of three top-level elements:

```toml
# Policy for fields not declared below. Optional — defaults to "dynamic".
dynamic_field_policy = "dynamic"

# Fields to search by default when a query does not specify a field.
default_fields = ["title", "body"]

# Field definitions. Each field has a name and a typed configuration.
[fields.<field_name>.<FieldType>]
# ... type-specific options
```

- **`dynamic_field_policy`** — How the engine treats fields present in an ingested document but **absent** from this schema. Accepted values: `"strict"`, `"dynamic"`, `"ignore"`. Defaults to `"dynamic"`. See [Dynamic Schema](../concepts/schema_and_fields.md#dynamic-schema) for the full semantics and the warning about silent truncation under `"dynamic"`.
- **`default_fields`** — A list of field names used as default search targets by the [Query DSL](../concepts/query_dsl.md). Only lexical fields (Text, Integer, Float, etc.) can be default fields. This key is optional and defaults to an empty list.
- **`fields`** — A map of field names to their typed configuration. Each field must specify exactly one field type.

## Field Naming

- Field names are arbitrary strings (e.g., `title`, `body_vec`, `created_at`).
- **Field names starting with `_` are reserved** for the engine. The only allow-listed name is `_id` (managed automatically). Attempting to declare any other `_`-prefixed field results in an error.
- Field names must be unique within a schema.

## Field Types

Fields fall into two categories: **Lexical** (for keyword/full-text search) and **Vector** (for similarity search). A single field cannot be both.

### Lexical Fields

#### Text

Full-text searchable field. Text is processed by the analysis pipeline (tokenization, normalization, stemming, etc.).

```toml
[fields.title.Text]
indexed = true      # Whether to index this field for search
stored = true       # Whether to store the original value for retrieval
term_vectors = true # Whether to store term positions (for phrase and span queries)
doc_values = true   # Whether to also copy the value into DocValues (for sorting/faceting)
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables searching this field |
| `stored` | `bool` | `true` | Stores the original value so it can be returned in results |
| `term_vectors` | `bool` | `true` | Stores term positions, read by phrase and span queries; highlighting always re-tokenizes the stored text and does not use them |
| `doc_values` | `bool` | `true` | Copies the value into DocValues, the column-oriented store [sorting](../laurus/faceting.md) and faceting/aggregation read from. Takes effect only when `stored` is also `true` — see [Common option: `doc_values`](#common-option-doc_values) below |

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
doc_values = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables filtering by boolean value |
| `stored` | `bool` | `true` | Stores the original value |
| `doc_values` | `bool` | `true` | See [Common option: `doc_values`](#common-option-doc_values) below |

#### DateTime

UTC timestamp field. Supports range queries.

```toml
[fields.created_at.DateTime]
indexed = true
stored = true
doc_values = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables range queries on date/time |
| `stored` | `bool` | `true` | Stores the original value |
| `doc_values` | `bool` | `true` | See [Common option: `doc_values`](#common-option-doc_values) below |

#### Geo

Geographic point field (latitude/longitude). Supports radius and bounding box queries.

```toml
[fields.location.Geo]
indexed = true
stored = true
doc_values = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables geo queries (radius, bounding box) |
| `stored` | `bool` | `true` | Stores the original value |
| `doc_values` | `bool` | `true` | See [Common option: `doc_values`](#common-option-doc_values) below |

#### Geo3d

3D Earth-Centered Earth-Fixed (ECEF) Cartesian point field (x / y / z in meters). Supports the `geo3d_distance` (sphere), `geo3d_bbox` (3D AABB), and `geo3d_nearest` (k-NN) queries. See [3D Geographic Search (ECEF)](../concepts/geo3d.md) for the coordinate system and the `wgs84_to_ecef` / `ecef_to_wgs84` conversion utilities.

```toml
[fields.position.Geo3d]
indexed = true
stored = true
doc_values = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Enables 3D geo queries (`geo3d_distance`, `geo3d_bbox`, `geo3d_nearest`) |
| `stored` | `bool` | `true` | Stores the original `(x, y, z)` value |
| `doc_values` | `bool` | `true` | See [Common option: `doc_values`](#common-option-doc_values) below |

#### Bytes

Raw binary data field. Not indexed — stored only.

```toml
[fields.thumbnail.Bytes]
stored = true
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `stored` | `bool` | `true` | Stores the binary data |

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

> **Note:** Unlike Hnsw and Flat, the `dimension` field in Ivf is **required** and has no default value.

**Tuning guidelines:**

- `n_clusters`: A common heuristic is `sqrt(N)` where N is the total number of vectors.
- `n_probe`: Start with 1 and increase until recall is acceptable. Typical range is 1–20.

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
