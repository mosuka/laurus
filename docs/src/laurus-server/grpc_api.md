# gRPC API Reference

All services are defined under the `laurus.v1` protobuf package.

## Services Overview

| Service | RPCs | Description |
| :--- | :--- | :--- |
| `HealthService` | `Check` | Health checking |
| `IndexService` | `CreateIndex`, `GetIndex`, `GetSchema`, `AddField`, `DeleteField` | Index lifecycle and schema |
| `DocumentService` | `PutDocument`, `AddDocument`, `PutDocuments`, `AddDocuments`, `GetDocuments`, `DeleteDocuments`, `Commit`, `FlushWal` | Document CRUD, bulk ingestion, commit, and WAL flush |
| `SearchService` | `Search`, `SearchStream` | Unary and streaming search |

---

## HealthService

### `Check`

Returns the current serving status of the server.

```protobuf
rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
```

**Response fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `status` | `ServingStatus` | `SERVING_STATUS_SERVING` when the server is ready |

---

## IndexService

### `CreateIndex`

Create a new index with the given schema. Fails with `ALREADY_EXISTS` if an index is already open.

```protobuf
rpc CreateIndex(CreateIndexRequest) returns (CreateIndexResponse);
```

**Request fields:**

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `schema` | `Schema` | Yes | Index schema definition |

**Schema structure:**

```protobuf
message Schema {
  map<string, FieldOption> fields = 1;
  repeated string default_fields = 2;
  map<string, AnalyzerDefinition> analyzers = 3;
  map<string, EmbedderConfig> embedders = 4;
  DynamicFieldPolicy dynamic_field_policy = 5;
}

enum DynamicFieldPolicy {
  DYNAMIC_FIELD_POLICY_UNSPECIFIED = 0;
  DYNAMIC_FIELD_POLICY_STRICT = 1;
  DYNAMIC_FIELD_POLICY_DYNAMIC = 2;
  DYNAMIC_FIELD_POLICY_IGNORE = 3;
}
```

- **`fields`** — Field definitions keyed by field name.
- **`default_fields`** — Field names used as default search targets when a query does not specify a field.
- **`analyzers`** — Custom analyzer pipelines keyed by name. Referenced by `TextOption.analyzer`.
- **`embedders`** — Embedder configurations keyed by name. Referenced by vector field options (`HnswOption.embedder`, etc.).
- **`dynamic_field_policy`** — How the engine treats fields that appear in an ingested document but are **absent** from `fields`. `UNSPECIFIED` is interpreted as `DYNAMIC` for forward compatibility. See [Schema & Fields](../concepts/schema_and_fields.md#dynamic-schema) for the full behaviour matrix and the warning about silent truncation under `DYNAMIC`.

**AnalyzerDefinition:**

```protobuf
message AnalyzerDefinition {
  repeated ComponentConfig char_filters = 1;
  ComponentConfig tokenizer = 2;
  repeated ComponentConfig token_filters = 3;
}
```

**ComponentConfig** (used for char filters, tokenizer, and token filters):

| Field | Type | Description |
| :--- | :--- | :--- |
| `type` | `string` | Component type name (e.g. `"whitespace"`, `"lowercase"`, `"unicode_normalization"`) |
| `params` | `map<string, string>` | Type-specific parameters as string key-value pairs |

**EmbedderConfig:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `type` | `string` | Embedder type name (e.g. `"precomputed"`, `"candle_bert"`, `"openai"`) |
| `params` | `map<string, string>` | Type-specific parameters (e.g. `"model"` → `"sentence-transformers/all-MiniLM-L6-v2"`) |

Each `FieldOption` is a `oneof` with one of the following field types:

| Lexical Fields | Vector Fields |
| :--- | :--- |
| `TextOption` (`indexed`, `stored`, `term_vectors`, `doc_values`, `analyzer`) | `HnswOption` (`dimension`, `distance`, `m`, `ef_construction`, `base_weight`, `quantizer`, `embedder`, `rerank_storage`, `pq_codebook_path`) |
| `IntegerOption` (`indexed`, `stored`, `multi_valued`, `doc_values`) | `FlatOption` (`dimension`, `distance`, `base_weight`, `quantizer`, `embedder`, `rerank_storage`) |
| `FloatOption` (`indexed`, `stored`, `multi_valued`, `doc_values`) | `IvfOption` (`dimension`, `distance`, `n_clusters`, `n_probe`, `base_weight`, `quantizer`, `embedder`, `rerank_storage`) |
| `BooleanOption` (`indexed`, `stored`, `doc_values`) | |
| `DateTimeOption` (`indexed`, `stored`, `doc_values`) | |
| `GeoOption` (`indexed`, `stored`, `doc_values`) | |
| `Geo3dOption` (`indexed`, `stored`, `doc_values`) | |
| `BytesOption` (`stored`) | |

The `embedder` field in vector options specifies the name of an embedder defined in `Schema.embedders`. When set, the server automatically generates vectors from document text fields at index time. Leave empty to supply pre-computed vectors directly.

**Doc values:** `doc_values` (Issue #1047) is `optional bool` on every lexical option above except `BytesOption`, following the same tri-state contract as `term_vectors`: a client that omits it gets the engine's default (`true`), distinguishable from an explicit `false`. It controls whether the field's value is also copied into DocValues, the column-oriented store sorting and faceting/aggregation read from — a DocValues column is written only when `stored` and `doc_values` are both `true`. `BytesOption` carries no such field: a `Bytes` value is never written to DocValues regardless. Turning `doc_values` off for a field that is never sorted or faceted on shrinks its segment footprint; the field remains fully searchable and retrievable either way.

**Distance metrics:** `COSINE`, `EUCLIDEAN`, `MANHATTAN`, `DOT_PRODUCT`, `ANGULAR`

**Quantization methods:** `SCALAR_8BIT` (default), `PRODUCT_QUANTIZATION` (Issue #481 Stage 3; supported by the HNSW index — Flat / IVF reject it at write time).

`NONE` (no quantization) was removed in Issue #481 Stage 1. The proto enum value `0` (`QUANTIZATION_METHOD_NONE`) is kept as a wire-compat reservation; if the server receives it, it falls back to `SCALAR_8BIT` via `Default::default()`.

**Rerank storage:** the optional `rerank_storage` field (enum `RerankStorageKind`: `UNSPECIFIED` = no sidecar, `F32`) enables the Stage-2 rerank sidecar (Issue #481 / #793). When set to `F32` on an HNSW field, commit writes an extra full-precision `.hnsw.f32` sidecar so searches that set `rerank_factor` rescore int8 candidates against the original vectors. Omitting the field (or `UNSPECIFIED`) keeps Stage-1 int8-only ranking. Since #932 the sidecar is emitted and consumed by all three vector index types (HNSW / Flat / IVF); on Flat/IVF the rescoring applies to field-routed queries.

**Shared PQ codebook:** the optional `pq_codebook_path` field on `HnswOption` (Issue #631) names a storage-relative shared PQ codebook file, trained once via the `laurus train pq-codebook` CLI command. Segments are then encoded against the pre-trained codebook instead of re-training k-means on every commit and merge. Only meaningful with a `PRODUCT_QUANTIZATION` quantizer; when set but not yet trained, commits fail with an error naming the training command (no silent fallback to per-segment training). Unset keeps per-segment training.

**Base weight:** `base_weight` on `HnswOption`/`FlatOption`/`IvfOption` is `optional float` (Issue #1084): a client that omits it gets the engine's default (`1.0`), distinguishable from an explicit `0.0`. It sets the field's relative scoring priority when a query targets it alongside other vector fields — see [Vector Search → Weights](../concepts/search/vector_search.md#weights) for what it does and does not affect.

**QuantizationConfig structure:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `method` | `QuantizationMethod` | Quantization method (`QUANTIZATION_METHOD_SCALAR_8BIT` or `QUANTIZATION_METHOD_PRODUCT_QUANTIZATION`). The reserved value `0` (`NONE`) is silently coerced to `SCALAR_8BIT`. |
| `subvector_count` | `uint32` | Number of subvectors (only used when `method` is `PRODUCT_QUANTIZATION`; must evenly divide `dimension`) |

**Example:**

```json
{
  "schema": {
    "fields": {
      "title": {"text": {"indexed": true, "stored": true, "term_vectors": true}},
      "embedding": {"hnsw": {"dimension": 384, "distance": "DISTANCE_METRIC_COSINE", "m": 16, "ef_construction": 200}}
    },
    "default_fields": ["title"]
  }
}
```

### `GetIndex`

Get index statistics.

```protobuf
rpc GetIndex(GetIndexRequest) returns (GetIndexResponse);
```

**Response fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `document_count` | `uint64` | Total number of documents in the index |
| `vector_fields` | `map<string, VectorFieldStats>` | Per-field vector statistics |

Each `VectorFieldStats` contains `vector_count` and `dimension`.

### `AddField`

Add a new field to the running index at runtime.

```protobuf
rpc AddField(AddFieldRequest) returns (AddFieldResponse);
```

**Request fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | `string` | The field name |
| `field_option` | `FieldOption` | The field configuration |

**Response fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `schema` | `Schema` | The updated schema after the field has been added |

**HTTP gateway:** `POST /v1/schema/fields`

### `DeleteField`

Remove a field from the running index schema.

```protobuf
rpc DeleteField(DeleteFieldRequest) returns (DeleteFieldResponse);
```

**Request fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | `string` | The field name to remove |

**Response fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `schema` | `Schema` | The updated schema after removal |

Existing indexed data for the field remains in storage but becomes
inaccessible. Per-field analyzers and embedders are unregistered.

**HTTP gateway:** `DELETE /v1/schema/fields/{name}`

### `GetSchema`

Retrieve the current index schema.

```protobuf
rpc GetSchema(GetSchemaRequest) returns (GetSchemaResponse);
```

**Response fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `schema` | `Schema` | The index schema |

---

## DocumentService

### `PutDocument`

Insert or replace a document by ID. If a document with the same ID already exists, it is replaced.

```protobuf
rpc PutDocument(PutDocumentRequest) returns (PutDocumentResponse);
```

**Request fields:**

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | `string` | Yes | External document ID |
| `document` | `Document` | Yes | Document content |

**Document structure:**

```protobuf
message Document {
  map<string, Value> fields = 1;
}
```

Each `Value` is a `oneof` with these types:

| Type | Proto Field | Description |
| :--- | :--- | :--- |
| Null | `null_value` | Null value |
| Boolean | `bool_value` | Boolean value |
| Integer | `int64_value` | 64-bit signed integer |
| Float | `float64_value` | 64-bit floating point |
| Text | `text_value` | UTF-8 string |
| Bytes | `bytes_value` | Raw bytes |
| Vector | `vector_value` | `VectorValue` (list of floats) |
| DateTime | `datetime_value` | Unix microseconds (UTC) |
| Geo | `geo_value` | `GeoPoint` (latitude, longitude) |
| Int64Array | `int64_array_value` | `Int64ArrayValue` (multi-valued integers; requires `IntegerOption.multi_valued = true`) |
| Float64Array | `float64_array_value` | `Float64ArrayValue` (multi-valued floats; requires `FloatOption.multi_valued = true`) |
| Geo3d | `geo3d_value` | `Geo3dPoint` (x, y, z meters; ECEF Cartesian) |

**Geo3dPoint:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `x` | `double` | X coordinate in meters (ECEF: equatorial plane, +X toward 0° longitude) |
| `y` | `double` | Y coordinate in meters (ECEF: equatorial plane, +Y toward 90°E) |
| `z` | `double` | Z coordinate in meters (ECEF: +Z toward the North Pole) |

See [3D Geographic Search (ECEF)](../concepts/geo3d.md) for the full coordinate system description and the `wgs84_to_ecef` / `ecef_to_wgs84` conversion utilities.

### `AddDocument`

Add a document. Unlike `PutDocument`, this does not replace existing documents with the same ID — multiple documents can share an ID (chunking pattern).

```protobuf
rpc AddDocument(AddDocumentRequest) returns (AddDocumentResponse);
```

Request fields are the same as `PutDocument`.

### `PutDocuments`

Batched upsert. Entries are applied sequentially, in input order, with one WAL fsync for the whole batch — much faster than one `PutDocument` call per document. Duplicate IDs within one batch dedup exactly like the same puts issued one by one (the last occurrence wins).

```protobuf
rpc PutDocuments(PutDocumentsRequest) returns (PutDocumentsResponse);

message DocumentEntry {
  string id = 1;
  Document document = 2;
}

message PutDocumentsRequest {
  repeated DocumentEntry documents = 1;
}

message PutDocumentsResponse {
  uint32 applied = 1; // equals the request size on success
}
```

The call fails fast at the first entry that cannot be applied. Already-applied entries are **not** rolled back — they are durable at the next commit — and the error status message carries the failing position, its ID, and the applied count, so retrying the batch (or its suffix) is idempotent. A batch that fails on a caller mistake (e.g. a schema violation) returns `INVALID_ARGUMENT`; storage failures return `INTERNAL`.

### `AddDocuments`

Batched chunk append. Like `PutDocuments` but never deletes existing documents, so a batch may legitimately repeat an ID to add multiple chunks of the same logical document.

```protobuf
rpc AddDocuments(AddDocumentsRequest) returns (AddDocumentsResponse);
```

Request/response fields mirror `PutDocuments`.

### `GetDocuments`

Retrieve all documents matching the given external ID.

```protobuf
rpc GetDocuments(GetDocumentsRequest) returns (GetDocumentsResponse);
```

**Request fields:**

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | `string` | Yes | External document ID |

**Response fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `documents` | `repeated Document` | Matching documents |

### `DeleteDocuments`

Delete all documents matching the given external ID.

```protobuf
rpc DeleteDocuments(DeleteDocumentsRequest) returns (DeleteDocumentsResponse);
```

### `Commit`

Commit pending changes (additions and deletions) to the index. Changes are not visible to search until committed.

```protobuf
rpc Commit(CommitRequest) returns (CommitResponse);
```

### `FlushWal`

Force buffered WAL records durable without a full commit. Both messages are empty. This is a near no-op under the default per-record sync policy (every write is already fsync'd); under the group-commit policy it flushes the current partial batch on demand, bounding the crash-loss window. Unlike `Commit`, it does not materialize segments, so buffered changes remain invisible to search until a subsequent `Commit`.

```protobuf
rpc FlushWal(FlushWalRequest) returns (FlushWalResponse);

message FlushWalRequest {}

message FlushWalResponse {}
```

The WAL durability policy is configured server-side via the `[index.wal]` config section. See [Configuration → `[index.wal]` Section](configuration.md#indexwal-section) and [Persistence & WAL → WAL Durability Policy](../laurus/persistence.md#wal-durability-policy).

**HTTP gateway:** `POST /v1/flush_wal`

---

## SearchService

### `Search`

Execute a search query and return results as a single response.

```protobuf
rpc Search(SearchRequest) returns (SearchResponse);
```

**Response fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `results` | `repeated SearchResult` | Search results ordered by relevance |
| `total_hits` | `uint64` | Total number of matching documents (before `limit`/`offset`) |

### `SearchStream`

Execute a search query and stream results back one at a time.

```protobuf
rpc SearchStream(SearchRequest) returns (stream SearchResult);
```

### SearchRequest Fields

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `query` | `string` | No | Lexical search query in [Query DSL](../concepts/query_dsl.md) |
| `query_vectors` | `repeated QueryVector` | No | Vector search queries |
| `limit` | `uint32` | No | Maximum number of results (default: engine default) |
| `offset` | `uint32` | No | Number of results to skip |
| `fusion` | `FusionAlgorithm` | No | Fusion algorithm for hybrid search |
| `lexical_params` | `LexicalParams` | No | Lexical search parameters |
| `vector_params` | `VectorParams` | No | Vector search parameters |
| `field_boosts` | `map<string, float>` | No | Per-field score boosting |

At least one of `query` or `query_vectors` must be provided.

### 3D Geographic Queries

3D ECEF geographic queries are expressed in the lexical DSL string passed via `SearchRequest.query`. There is no dedicated message type — the same DSL forms used by the core library work over gRPC. Three forms are available (see [Query DSL → 3D Geographic Queries](../concepts/query_dsl.md#3d-geographic-queries-geo3d_) for full syntax):

- `position:geo3d_distance(x, y, z, distance_m)` — sphere centered at `(x, y, z)` with maximum distance in meters
- `position:geo3d_bbox(min_x, min_y, min_z, max_x, max_y, max_z)` — 3D axis-aligned bounding box
- `position:geo3d_nearest(x, y, z, k)` — k nearest neighbours to `(x, y, z)`

`position` is the field name; substitute the actual `Geo3d`-typed field declared in your schema. All numeric arguments are signed `double` values; `k` is an unsigned integer.

### QueryVector

| Field | Type | Description |
| :--- | :--- | :--- |
| `vector` | `repeated float` | Query vector |
| `weight` | `float` | Weight for this vector (default: 1.0) |
| `fields` | `repeated string` | Target vector fields (empty = all) |

### FusionAlgorithm

A `oneof` with two options:

- **RRF** (Reciprocal Rank Fusion): `k` parameter (default: 60)
- **WeightedSum**: `lexical_weight` and `vector_weight`

### LexicalParams

| Field | Type | Description |
| :--- | :--- | :--- |
| `min_score` | `float` | Minimum score threshold |
| `timeout_ms` | `uint64` | Search timeout in milliseconds |
| `parallel` | `bool` | Enable parallel search |
| `sort_by` | `SortSpec` | Sort by a field instead of score |

### SortSpec

| Field | Type | Description |
| :--- | :--- | :--- |
| `field` | `string` | Field name to sort by. Empty string means sort by relevance score |
| `order` | `SortOrder` | `SORT_ORDER_ASC` (ascending) or `SORT_ORDER_DESC` (descending) |

### VectorParams

| Field | Type | Description |
| :--- | :--- | :--- |
| `fields` | `repeated string` | Target vector fields |
| `score_mode` | `VectorScoreMode` | `WEIGHTED_SUM`, `MAX_SIM`, or `LATE_INTERACTION` |
| `overfetch` | `float` | Overfetch factor (default: 2.0) |
| `min_score` | `float` | Minimum score threshold |
| `rerank_factor` | `optional uint32` | Stage 2 rerank widening factor (Issue #481). When set on a field whose schema enabled `rerank_storage`, the server widens the int8/PQ candidate fetch to `top_k * rerank_factor` and rescores the candidates against the original full-precision vectors before returning the top `top_k`. Honored on all three vector index types since #932 (HNSW, Flat, IVF — on Flat/IVF this applies to field-routed queries); fields without `rerank_storage = "F32"` silently fall back to int8 ranking — there is no f32 information to recover. A value of `0` or omitting the field disables rerank. |
| `ef_search` | `optional uint32` | Per-query override for the HNSW `ef_search` candidate-list size (Issue #644). Also gates the PQ → SQ → f32 three-stage rerank chain (Issue #673): on a PQ field with `rerank_storage` enabled, setting `ef_search` wider than `top_k * rerank_factor` activates an extra int8 stage that rescores the graph's full candidate set — derived from the same `rerank_storage` sidecar, no extra configuration needed — before the exact stage's narrower budget is carved out of it. Ignored on non-HNSW fields. |

### SearchResult

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | External document ID |
| `score` | `float` | Relevance score |
| `document` | `Document` | Document content |

### Example

```json
{
  "query": "body:rust",
  "query_vectors": [
    {"vector": [0.1, 0.2, 0.3], "weight": 1.0}
  ],
  "limit": 10,
  "fusion": {
    "rrf": {"k": 60}
  },
  "field_boosts": {
    "title": 2.0
  }
}
```

---

## Error Handling

gRPC errors are returned as standard `Status` codes:

| Laurus Error | gRPC Status | When |
| :--- | :--- | :--- |
| Schema / Query / Field / JSON | `INVALID_ARGUMENT` | Malformed request or schema |
| No index open | `FAILED_PRECONDITION` | RPC called before `CreateIndex` |
| Index already exists | `ALREADY_EXISTS` | `CreateIndex` called twice |
| Not implemented | `UNIMPLEMENTED` | Feature not yet supported |
| Internal errors | `INTERNAL` | I/O, storage, or unexpected errors |
