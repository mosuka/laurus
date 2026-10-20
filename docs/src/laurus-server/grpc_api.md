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
- **`dynamic_field_policy`** — How the engine treats fields that appear in an ingested document but are **absent** from `fields`. `UNSPECIFIED` is interpreted as `DYNAMIC` for forward compatibility. `CreateIndex` rejects a value the server does not know as `INVALID_ARGUMENT` instead of treating it as `DYNAMIC` (Issue #1473). See [Schema & Fields](../concepts/schema_and_fields.md#dynamic-schema) for the full behaviour matrix and the warning about silent truncation under `DYNAMIC`.

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
| `type` | `string` | Embedder type name (e.g. `"precomputed"`, `"candle_bert"`, `"candle_colbert"`, `"openai"`) |
| `params` | `map<string, string>` | Type-specific parameters (e.g. `"model"` → `"sentence-transformers/all-MiniLM-L6-v2"`). `"candle_colbert"` (Issue #1349) also takes the optional `"revision"`, `"query_maxlen"` and `"doc_maxlen"`; the lengths are decimal strings such as `"32"`, and anything else is rejected |

Each `FieldOption` is a `oneof` with one of the following field types:

| Lexical Fields | Vector Fields |
| :--- | :--- |
| `TextOption` (`indexed`, `stored`, `multi_valued`, `position_increment_gap`, `term_vectors`, `doc_values`, `analyzer`) | `HnswOption` (`dimension`, `distance`, `m`, `ef_construction`, `base_weight`, `quantizer`, `embedder`, `rerank_storage`, `pq_codebook_path`) |
| `IntegerOption` (`indexed`, `stored`, `multi_valued`, `doc_values`) | `FlatOption` (`dimension`, `distance`, `base_weight`, `quantizer`, `embedder`, `rerank_storage`) |
| `FloatOption` (`indexed`, `stored`, `multi_valued`, `doc_values`) | `IvfOption` (`dimension`, `distance`, `n_clusters`, `n_probe`, `base_weight`, `quantizer`, `embedder`, `rerank_storage`) |
| `BooleanOption` (`indexed`, `stored`, `multi_valued`, `doc_values`) | `MultiVectorOption` (`dimension`, `distance`, `embedder`, `storage`) |
| `DateTimeOption` (`indexed`, `stored`, `multi_valued`, `doc_values`) | |
| `GeoOption` (`indexed`, `stored`, `multi_valued`, `doc_values`) | |
| `Geo3dOption` (`indexed`, `stored`, `multi_valued`, `doc_values`) | |
| `BytesOption` (`stored`, `multi_valued`) | |

The `embedder` field in vector options specifies the name of an embedder defined in `Schema.embedders`. When set, the server automatically generates vectors from document text fields at index time. Leave empty to supply pre-computed vectors directly.

**Doc values:** `doc_values` (Issue #1047) is `optional bool` on every lexical option above except `BytesOption`, following the same tri-state contract as `term_vectors`: a client that omits it gets the engine's default (`true`), distinguishable from an explicit `false`. It controls whether the field's value is also copied into DocValues, the column-oriented store sorting and faceting/aggregation read from — a DocValues column is written only when `stored` and `doc_values` are both `true`. `BytesOption` carries no such field: a `Bytes` value is never written to DocValues regardless. Turning `doc_values` off for a field that is never sorted or faceted on shrinks its segment footprint; the field remains fully searchable and retrievable either way.

**Multi-valued geo, datetime, boolean, text and bytes:** `multi_valued` on `GeoOption` / `Geo3dOption` (Issue #1174), on `DateTimeOption` (Issue #1184) and on `BooleanOption` (Issue #1180) is proto field number `4` — `3` is already taken by `doc_values` there, unlike `IntegerOption` / `FloatOption`; on `TextOption` (Issue #1175) it is field number `6`, because Text already used `3`–`5` for `term_vectors`, `analyzer` and `doc_values`; on `BytesOption` (Issue #1176) it is field number `2` — `1` is already taken by `stored`, and `BytesOption` has neither an `indexed` nor a `doc_values` field. When `true`, a geo field accepts `GeoArrayValue` / `Geo3dArrayValue` (see the `Value` table below) and distance / bounding-box queries (plus `geo3d_nearest`) match a document if **any** of its points satisfies the predicate, scoring it by its closest point. A datetime field accepts `DatetimeArrayValue` (`repeated int64` Unix microseconds UTC, the same encoding as `datetime_value`) and range queries match a document if **any** of its instants is in range, with constant scoring; a document is reported once even when several instants match. A boolean field accepts `BoolArrayValue` (`repeated bool`) and a term query matches a document if **any** element equals the queried value; each element is indexed as its own `true` / `false` term posting — there are no BKD points for booleans — so repeated elements raise the term frequency (and hence the BM25 score), not the hit count. A text field accepts `TextArrayValue` (`repeated string`) and a term query matches a document if **any** element contains the term; `position_increment_gap` (field `7`, `optional uint32`) is the number of positions inserted between consecutive elements, which keeps a phrase query from spanning two elements unless its slop reaches the gap. It is `optional` for the same reason `term_vectors` / `doc_values` are: an omitted value means the engine default (`100`), not `0`. A bytes field accepts `BytesArrayValue` (`repeated bytes`); since a `Bytes` value is never lexically indexed, this has no "any match" query semantics at all — it only governs the stored/wire shape and ingestion arity — and, like the scalar `bytes_value`, the per-element MIME type carried by `DataValue::BytesArray` is not represented on the wire.

**Multi-vector fields:** `MultiVectorOption` (`FieldOption` field `12`, Issue #1177) declares a field that holds each document's token vectors (for example ColBERT-style per-token embeddings) for late-interaction rescoring. It has no ANN index and is not a vector-search target: a `query_vectors` entry naming it is rejected. `distance` must be `COSINE` (vectors are L2-normalized when written) or `DOT_PRODUCT`. Values are `VectorArrayValue`s (see the `Value` table below), and the field is not stored, so documents returned by the server never include it. `embedder` (field `3`, Issue #1349) names a token-level embedder (`"candle_colbert"`) from `Schema.embedders`; when set, a document may give the field a `text_value`, which is embedded into token vectors. A MultiVector field accepts only a `"candle_colbert"` or `"precomputed"` embedder, and other vector fields do not accept `"candle_colbert"`.

**Multi-vector storage:** the optional `storage` field (field `4`, enum `MultiVectorStorage`, Issue #1346) picks the on-disk element kind of the token vectors: `UNSPECIFIED` = same as `F32`, `F32` (default, 4 bytes/element, exact), `F16` (2 bytes/element, ~2⁻¹¹ relative error per element), `INT8` (~1 byte/element plus a small per-vector scale overhead — `dimension + 2` bytes/row total — using a per-vector scale `max(abs(vector)) / 127`, not trained per-segment or over the corpus), or `BINARY` (1 bit/element — `ceil(dimension / 8)` bytes/row — against a center the field learns once; requires `distance` `COSINE`). At 300 tokens × 128 dimensions (~150 KB/document in `F32`), `F16` is ~75 KB/document, `INT8` ~38 KB/document and `BINARY` ~4.7 KB/document. Unset keeps `F32`. `CreateIndex`, `AddField` and `UpdateField` reject a value the server does not know as `INVALID_ARGUMENT` instead of storing the field as `F32` (Issue #1473). A server older than this check stores such a field as `F32` without an error, so check the returned schema when talking to an older server.

**Multi-vector token pooling:** the optional `pool_factor` field (field `5`, `uint32`, Issue #1346) makes each document keep about `1 / pool_factor` of its token vectors: the first one, then the means of Ward clusters of the others. Unset means `1`, no pooling; `0` is rejected. Turning pooling on for an existing field is a reindex; changing or turning off a factor in use is destructive. See [Token pooling](../concepts/indexing/vector_indexing.md#token-pooling).

**Distance metrics:** `COSINE` (the zero value, so the default when `distance` is unset), `EUCLIDEAN`, `MANHATTAN`, `DOT_PRODUCT`, `ANGULAR`. `CreateIndex`, `AddField` and `UpdateField` reject a value the server does not know as `INVALID_ARGUMENT` instead of building the field with `COSINE` (Issue #1473); a server older than this check uses `COSINE` without an error.

**Quantization methods:** `SCALAR_8BIT` (default), `PRODUCT_QUANTIZATION` (Issue #481 Stage 3; supported by the HNSW index only — `CreateIndex`, `AddField` and `UpdateField` reject it on a Flat / IVF field as `INVALID_ARGUMENT`, Issue #1457), `FLOAT16` (Issue #623; IEEE 754 half precision, 2 bytes per dimension, nothing to train; supported by HNSW, Flat and IVF — see [Float16](../concepts/indexing/vector_indexing.md#float16-issue-623)), `PRODUCT_QUANTIZATION_FASTSCAN` (Issue #695; experimental, HNSW only, and only on a server built with the `pq-fastscan` feature). `CreateIndex`, `AddField` and `UpdateField` reject a value the server does not know, and `PRODUCT_QUANTIZATION_FASTSCAN` on a server built without `pq-fastscan`, as `INVALID_ARGUMENT` instead of storing the field as `SCALAR_8BIT` (Issue #1459). A server that predates `FLOAT16` predates this check as well and stores such a field as `SCALAR_8BIT` without an error, so check the returned schema when talking to an older server.

`NONE` (no quantization) was removed in Issue #481 Stage 1. The proto enum value `0` (`QUANTIZATION_METHOD_NONE`) is what a client that leaves `method` unset sends, so it is kept: the server stores the field as `SCALAR_8BIT`, the default.

**Rerank storage:** the optional `rerank_storage` field (enum `RerankStorageKind`: `UNSPECIFIED` = no sidecar, `F32`) enables the Stage-2 rerank sidecar (Issue #481 / #793). When set to `F32` on an HNSW field, commit writes an extra full-precision `.hnsw.f32` sidecar so searches that set `rerank_factor` rescore int8 candidates against the original vectors. Omitting the field (or `UNSPECIFIED`) keeps Stage-1 int8-only ranking; `CreateIndex`, `AddField` and `UpdateField` reject a value the server does not know as `INVALID_ARGUMENT` instead of writing no sidecar (Issue #1473). Since #932 the sidecar is emitted and consumed by all three vector index types (HNSW / Flat / IVF), including a field-less query when the segment holds only one field (Issue #1388).

**Shared PQ codebook:** the optional `pq_codebook_path` field on `HnswOption` (Issue #631) names a storage-relative shared PQ codebook file, trained once via the `laurus train pq-codebook` CLI command. Segments are then encoded against the pre-trained codebook instead of re-training k-means on every commit and merge. Only meaningful with a `PRODUCT_QUANTIZATION` quantizer; when set but not yet trained, commits fail with an error naming the training command (no silent fallback to per-segment training). Unset keeps per-segment training.

**Base weight:** `base_weight` on `HnswOption`/`FlatOption`/`IvfOption` is `optional float` (Issue #1084): a client that omits it gets the engine's default (`1.0`), distinguishable from an explicit `0.0`. It sets the field's relative scoring priority when a query targets it alongside other vector fields — see [Vector Search → Weights](../concepts/search/vector_search.md#weights) for what it does and does not affect.

**Graph and cluster parameters:** `m` and `ef_construction` on `HnswOption` and `n_clusters` and `n_probe` on `IvfOption` are `optional uint32` (Issue #1479): a client that omits one gets the engine's default (`m` `16`, `ef_construction` `200`, `n_clusters` `100`, `n_probe` `1`), the same as a TOML schema. `CreateIndex`, `AddField` and `UpdateField` reject an `m` below `2`, an `ef_construction`, `n_clusters` or `n_probe` of `0`, and a `dimension` of `0` on any vector option as `INVALID_ARGUMENT`. `dimension` has no default over gRPC: it is a plain `uint32`, so an omitted one is `0` and rejected. `UpdateField` replaces the whole option, so a key omitted there also gets the default, not the field's current value. A server older than this change reads an omitted value as `0`, with which every document write fails (`m`) or every search returns no hits (`n_probe`), so set these explicitly when talking to an older server. An index such a server created still opens, with a warning in the server log; repair the field with `UpdateField` (a new `m`, `ef_construction` or `n_clusters` needs `reindex`, and a new `n_probe` takes effect once the index is reopened, which for `laurus serve` means restarting it). If the index does not open because a document write failed on it, that write is still in its write-ahead log: set a valid `m` in the index's `schema.toml` and start the server again, and the write is applied.

**Product quantization `subvector_count`:** with `PRODUCT_QUANTIZATION` or `PRODUCT_QUANTIZATION_FASTSCAN`, `CreateIndex`, `AddField` and `UpdateField` reject a `subvector_count` of `0` (what a client that leaves it unset sends), one that does not divide `dimension`, and one where `subvector_count` or `dimension / subvector_count` is above `65535`, as `INVALID_ARGUMENT` (Issue #1491). A server older than this check accepts such a field and fails a later commit as `INTERNAL`: a count that does not divide `dimension` at the first commit that holds a vector of the field, and `0` once a segment of the field reaches 256 vectors (16 for FastScan); every later commit then fails as well. An index such a server created still opens, with a warning in the server log, and every document write to it fails, without being applied, until the field is repaired with `UpdateField` and a valid `subvector_count` (with `reindex`, which rebuilds the field from its segments). Repair it before deleting many of the field's documents: a compaction of the field fails the commit until then. If the index does not open because a commit failed on it, set a valid `subvector_count` in the index's `schema.toml` and start the server again.

**QuantizationConfig structure:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `method` | `QuantizationMethod` | Quantization method (`QUANTIZATION_METHOD_SCALAR_8BIT`, `QUANTIZATION_METHOD_PRODUCT_QUANTIZATION`, `QUANTIZATION_METHOD_PRODUCT_QUANTIZATION_FASTSCAN` = `3` or `QUANTIZATION_METHOD_FLOAT16` = `4`). `0` (`NONE`, an unset `method`) means `SCALAR_8BIT`; any other value is rejected as `INVALID_ARGUMENT`. |
| `subvector_count` | `uint32` | Number of subvectors (only used when `method` is `PRODUCT_QUANTIZATION` or `PRODUCT_QUANTIZATION_FASTSCAN`; must be at least `1` and evenly divide `dimension`, or the field is rejected as `INVALID_ARGUMENT`; ignored for `SCALAR_8BIT` and `FLOAT16`) |

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
| GeoArray | `geo_array_value` | `GeoArrayValue` (`repeated GeoPoint`; multi-valued 2D points; requires `GeoOption.multi_valued = true`) |
| Geo3dArray | `geo3d_array_value` | `Geo3dArrayValue` (`repeated Geo3dPoint`; multi-valued 3D points; requires `Geo3dOption.multi_valued = true`) |
| DateTimeArray | `datetime_array_value` | `DatetimeArrayValue` (`repeated int64` Unix microseconds; multi-valued instants; requires `DateTimeOption.multi_valued = true`) |
| BoolArray | `bool_array_value` | `BoolArrayValue` (`repeated bool`; multi-valued booleans; requires `BooleanOption.multi_valued = true`) |
| TextArray | `text_array_value` | `TextArrayValue` (`repeated string`; multi-valued text; requires `TextOption.multi_valued = true`) |
| BytesArray | `bytes_array_value` | `BytesArrayValue` (`repeated bytes`; multi-valued bytes, MIME not carried on the wire; requires `BytesOption.multi_valued = true`) |
| VectorArray | `vector_array_value` | `VectorArrayValue` (`uint32 dimension`, `repeated float values`; token vectors packed row-major, `values.len() / dimension` vectors; requires a `MultiVectorOption` field) |

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

The response carries no total hit count. Field `2` (`total_hits`) held the
number of returned results rather than the number of matches its description
promised, and was removed (Issue #1455).

### `SearchStream`

Execute a search query and stream results back one at a time.

```protobuf
rpc SearchStream(SearchRequest) returns (stream SearchResult);
```

With `group_by`, each message is one group: its best hit, with the group's
next best hits in `inner_hits`.

### SearchRequest Fields

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `query` | `string` | No | Unified query in [Query DSL](../concepts/query_dsl.md): lexical clauses and vector clauses. An unknown field is rejected with `INVALID_ARGUMENT`. `lexical_params` and `field_boosts` apply to its lexical part |
| `query_vectors` | `repeated QueryVector` | No | Pre-embedded query vectors. Sent with `query`, they are added to its vector part; a `query` with only lexical clauses becomes a hybrid search (Issue #1366) |
| `limit` | `uint32` | No | Maximum number of results (`0` means the engine default, 10). `offset + limit` must not exceed the server's [`max_result_window`](configuration.md) (default 10,000), or the search is rejected with `INVALID_ARGUMENT` (Issue #1367) |
| `offset` | `uint32` | No | Number of results to skip; counts toward `max_result_window` with `limit` |
| `fusion` | `FusionAlgorithm` | No | Fusion algorithm for hybrid search |
| `lexical_params` | `LexicalParams` | No | Lexical search parameters |
| `vector_params` | `VectorParams` | No | Vector search parameters |
| `field_boosts` | `map<string, float>` | No | Per-field score boosting for the lexical part. A key must name a lexical field of the schema (or `_id`); an unknown or vector field is rejected with `INVALID_ARGUMENT` (Issue #1374). A lexical field the query does not use is accepted and has no effect |
| `highlight` | `HighlightParams` | No | Request highlighted fragments per field (Issue #1134) |
| `rescore` | `RescoreParams` | No | Rescore the top first-stage results with late interaction (Issue #1351) |
| `group_by` | `GroupByParams` | No | Group the results by a field's value; `limit` and `offset` then count groups (Issue #1297) |

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
| `weight` | `float` | Weight for this vector (default: 1.0; `0` also means 1.0, since proto3 cannot tell it from unset) |
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

`vector_params` applies to the request's vector part: `query_vectors` and
the vector clauses of `query` (Issues #1342, #1366). A request with neither
`query_vectors` nor `query` is rejected with `INVALID_ARGUMENT`. The server
cannot tell whether `query` has vector clauses without parsing it; on a
request whose only possible vector part is a `query` without vector
clauses, `vector_params` has no effect. Every other value either takes
effect or is rejected with `INVALID_ARGUMENT`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `fields` | `repeated string` | Vector fields (exact names) to search with query vectors that name no field. A query vector's own `fields` take precedence, and every DSL vector clause names its field, so `fields` is rejected when no searched `query_vectors` entry names no field. An unknown name is rejected. Empty means every vector field whose dimension matches. |
| `score_mode` | `VectorScoreMode` | `WEIGHTED_SUM`, `MAX_SIM`, or `LATE_INTERACTION`. Any other value is rejected. |
| `overfetch` | `float` | Each per-field index query fetches `ceil(limit * overfetch)` candidates before scores are combined. `0` (unset) means the default, `2.0`; a value in `(0, 1]` turns overfetch off. Negative or non-finite values are rejected. |
| `min_score` | `float` | Minimum per-field similarity, compared before query and field weights are applied. `0` (unset) means no threshold. A non-finite value is rejected. |
| `rerank_factor` | `optional uint32` | Stage 2 rerank widening factor (Issue #481). When set on a field whose schema enabled `rerank_storage`, the server widens the int8/PQ candidate fetch to `top_k * rerank_factor` and rescores the candidates against the original full-precision vectors before returning the top `top_k`. Honored on all three vector index types since #932 (HNSW, Flat, IVF), including a field-less query when the segment holds only one field (Issue #1388); fields without `rerank_storage = "F32"` silently fall back to int8 ranking — there is no f32 information to recover. A value of `0` or omitting the field disables rerank. |
| `ef_search` | `optional uint32` | Per-query override for the HNSW `ef_search` candidate-list size (Issue #644). Also gates the PQ → SQ → f32 three-stage rerank chain (Issue #673): on a PQ field with `rerank_storage` enabled, setting `ef_search` wider than `top_k * rerank_factor` activates an extra int8 stage that rescores the graph's full candidate set — derived from the same `rerank_storage` sidecar, no extra configuration needed — before the exact stage's narrower budget is carved out of it. `0` is rejected. Ignored on non-HNSW fields. |

### HighlightParams

Requests highlighted fragments for specific fields (Issue #1134). Highlighting
uses `SearchRequest.query` (or the lexical clause of a DSL query), the same
query that drives lexical scoring — `filter_query` never contributes
highlighted terms, and a vector-only request (no lexical query) produces no
highlights. Only `stored: true` text fields can be highlighted; other names
are silently skipped. See [Highlighting](../laurus/highlighting.md) for full
semantics.

| Field | Type | Description |
| :--- | :--- | :--- |
| `fields` | `repeated string` | Stored text fields to highlight. Required — a request with this unset or empty is rejected with `INVALID_ARGUMENT` |
| `max_fragments` | `optional uint32` | Maximum fragments per field (default: 5). `0` is rejected |
| `fragment_size` | `optional uint32` | Target fragment length in characters (default: 150). `0` is rejected |
| `tag` | `optional string` | HTML tag to wrap matches (default: `"mark"`). An empty string is treated as unset |
| `css_class` | `optional string` | CSS class added to the tag. An empty string is treated as unset |
| `require_field_match` | `optional bool` | Only use query terms that target the highlighted field (default: `true`) |
| `max_analyzed_chars` | `optional uint64` | Maximum characters of a field's text to analyze (default: 1,000,000) |
| `return_entire_field_if_no_highlight` | `optional bool` | Return the full field value as one fragment when nothing matched (default: `false`) |

### Highlights

| Field | Type | Description |
| :--- | :--- | :--- |
| `fragments` | `repeated string` | Highlighted fragments for one field, best fragment first |

### RescoreParams

Reorders the top `window_size` first-stage results (lexical, vector or
hybrid) with ColBERT-style late interaction over a `MultiVector` field
(Issue #1351). See
[Late-Interaction Rescore](../concepts/search/vector_search.md#late-interaction-rescore)
for the scoring and ordering rules.

| Field | Type | Description |
| :--- | :--- | :--- |
| `window_size` | `optional uint32` | How many top first-stage results to rescore, 1 to 10,000. Unset means 100 |
| `late_interaction` | `LateInteractionRescore` | The rescorer (a `oneof`; required) |

**LateInteractionRescore:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `field` | `string` | A `MultiVector` field |
| `vectors` | `VectorArrayValue` | The query's token vectors, packed row-major like a document value (1 to 1,024 vectors of the field's dimension). One of `vectors` and `text` is required |
| `text` | `string` | Query text, embedded by the field's token-level embedder (a `candle_colbert` one) |

The engine validates the values: an out-of-range window, a field that is not
a `MultiVector` field, the wrong number or dimension of vectors, text for a
field without a token-level embedder, or a rescore combined with
`lexical_params.sort_by` is rejected with `INVALID_ARGUMENT`, as is a
`RescoreParams` without `late_interaction` or a query.

### GroupByParams

Folds the ranked results (after fusion and rescore) that share a field's
value into groups, ordered by each group's best result (Issue #1297). See
[Result Grouping](../concepts/search/result_grouping.md) for the rules.

| Field | Type | Description |
| :--- | :--- | :--- |
| `field` | `string` | `_id`, or a single-valued, stored `Text`, `Integer`, `Float`, `Boolean` or `DateTime` field with `doc_values`. A result with no value forms a group of its own |
| `group_size` | `optional uint32` | Results kept per group, the returned one included, 1 to 100. Unset means 1 |

The engine validates the values: a field that cannot group, a `group_size`
outside 1 to 100, or an `offset + limit` above 10,000 is rejected with
`INVALID_ARGUMENT`.

### SearchResult

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | External document ID |
| `score` | `float` | Relevance score; for a rescored result, its late-interaction (MaxSim) score |
| `document` | `Document` | Document content |
| `highlights` | `map<string, Highlights>` | Highlighted fragments per field named in `SearchRequest.highlight.fields`. A field with no highlight is absent from this map |
| `group_key` | `Value` | With `group_by`: the value this result's group shares. Unset for an ungrouped search and for a result without a value |
| `inner_hits` | `repeated SearchResult` | With `group_by`: the group's next best results, in ranking order. Each carries the same `group_key` and has no `inner_hits` of its own |

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

### Example: Highlighting

```json
{
  "query": "body:rust",
  "limit": 10,
  "highlight": {"fields": ["body"], "max_fragments": 2, "tag": "em"}
}
```

A matching hit's `SearchResult` then carries:

```json
{
  "id": "doc1",
  "score": 1.2,
  "document": {"fields": {"body": {"text_value": "Rust is great"}}},
  "highlights": {"body": {"fragments": ["<em>Rust</em> is great"]}}
}
```

### Example: Rescore

A hybrid first stage whose top 100 results are rescored by a ColBERT
embedder on `body_colbert`:

```json
{
  "query": "body:lifetimes body_vec:\"how do lifetimes work\"",
  "limit": 10,
  "rescore": {
    "window_size": 100,
    "late_interaction": {"field": "body_colbert", "text": "how do lifetimes work"}
  }
}
```

With pre-computed query token vectors instead of text:

```json
{
  "late_interaction": {
    "field": "body_colbert",
    "vectors": {"dimension": 2, "values": [1.0, 0.0, 0.0, 1.0]}
  }
}
```

### Example: Grouping

Documents added as chunks with `AddDocuments` share their `_id`; grouping by
it returns each document once, with up to one more of its chunks:

```json
{
  "query": "body:rust",
  "limit": 10,
  "group_by": {"field": "_id", "group_size": 2}
}
```

Each `SearchResult` is then a group's best chunk:

```json
{
  "id": "manual",
  "score": 0.012571429,
  "document": {"fields": {"body": {"text_value": "Rust ownership rules for Rust programs"}}},
  "group_key": {"text_value": "manual"},
  "inner_hits": [
    {
      "id": "manual",
      "score": 0.012359549,
      "document": {"fields": {"body": {"text_value": "Borrowing in Rust"}}},
      "group_key": {"text_value": "manual"}
    }
  ]
}
```

---

## Error Handling

gRPC errors are returned as standard `Status` codes:

| Laurus Error | gRPC Status | When |
| :--- | :--- | :--- |
| Schema / Query / Field / Invalid argument / JSON | `INVALID_ARGUMENT` | Malformed request or schema, unknown field in a query, a search whose `offset + limit` exceeds `max_result_window` |
| No index open | `FAILED_PRECONDITION` | RPC called before `CreateIndex` |
| Index already exists | `ALREADY_EXISTS` | `CreateIndex` called twice |
| Not implemented | `UNIMPLEMENTED` | Feature not yet supported |
| Internal errors | `INTERNAL` | I/O, storage, or unexpected errors |
