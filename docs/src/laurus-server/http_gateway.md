# HTTP Gateway

The HTTP Gateway provides a RESTful HTTP/JSON interface to the Laurus search engine. It runs alongside the gRPC server and proxies requests internally:

```text
Client (HTTP/JSON) --> HTTP Gateway (axum) --> gRPC Server (tonic) --> Engine
```

## Enabling the HTTP Gateway

The gateway starts when `http_port` is configured:

```bash
# Via CLI argument
laurus serve --http-port 8080

# Via environment variable
LAURUS_HTTP_PORT=8080 laurus serve

# Via config file
laurus serve --config config.toml
# (set http_port in [server] section)
```

If `http_port` is not set, only the gRPC server starts.

## Endpoints

| Method | Path | gRPC Method | Description |
| :--- | :--- | :--- | :--- |
| GET | `/v1/health` | `HealthService/Check` | Health check |
| POST | `/v1/index` | `IndexService/CreateIndex` | Create a new index |
| GET | `/v1/index` | `IndexService/GetIndex` | Get index statistics |
| GET | `/v1/schema` | `IndexService/GetSchema` | Get the index schema |
| POST | `/v1/schema/fields` | `IndexService/AddField` | Dynamically add a field |
| DELETE | `/v1/schema/fields/{name}` | `IndexService/DeleteField` | Remove a field from the schema |
| PUT | `/v1/documents/{id}` | `DocumentService/PutDocument` | Upsert a document |
| POST | `/v1/documents/{id}` | `DocumentService/AddDocument` | Add a document (chunk) |
| GET | `/v1/documents/{id}` | `DocumentService/GetDocuments` | Get documents by ID |
| DELETE | `/v1/documents/{id}` | `DocumentService/DeleteDocuments` | Delete documents by ID |
| POST | `/v1/documents:bulk` | `DocumentService/PutDocuments` / `AddDocuments` | Bulk-ingest documents (`?mode=put\|add`, default `put`) |
| POST | `/v1/commit` | `DocumentService/Commit` | Commit pending changes |
| POST | `/v1/flush_wal` | `DocumentService/FlushWal` | Force buffered WAL records durable without a full commit |
| POST | `/v1/search` | `SearchService/Search` | Search (unary) |
| POST | `/v1/search/stream` | `SearchService/SearchStream` | Search (Server-Sent Events) |

## API Examples

### Health Check

```bash
curl http://localhost:8080/v1/health
```

### Create an Index

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "schema": {
      "dynamic_field_policy": "dynamic",
      "fields": {
        "title": {"text": {"indexed": true, "stored": true, "term_vectors": true}},
        "body": {"text": {"indexed": true, "stored": true, "term_vectors": true}}
      },
      "default_fields": ["title", "body"]
    }
  }'
```

The `dynamic_field_policy` key is optional. It controls how fields absent
from the schema are handled at ingest time. Accepted values: `"strict"`,
`"dynamic"` (default), `"ignore"`. See
[Schema & Fields](../concepts/schema_and_fields.md#dynamic-schema) for the
full semantics and the warning about silent truncation under `"dynamic"`.

### Get Index Statistics

```bash
curl http://localhost:8080/v1/index
```

### Get Schema

```bash
curl http://localhost:8080/v1/schema
```

The response always includes `multi_valued` for `integer`, `float`, `date_time`, `geo`, and `geo3d` options (e.g. `"location": {"geo": {"indexed": true, "stored": true, "multi_valued": true, "doc_values": true}}` or `"seen_at": {"date_time": {"indexed": true, "stored": true, "multi_valued": true, "doc_values": true}}`); the same key is accepted on `POST /v1/index` and `POST /v1/schema/fields`.

### Add a Field (Dynamic Schema)

Adds a new field to the running index. The request body uses the same `FieldOption` JSON shape as `POST /v1/index`:

```bash
curl -X POST http://localhost:8080/v1/schema/fields \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "category",
    "field_option": {"text": {"indexed": true, "stored": true}}
  }'
```

The response returns the updated schema.

### Delete a Field

Removes a field from the schema. The field name is supplied in the path:

```bash
curl -X DELETE http://localhost:8080/v1/schema/fields/category
```

Existing indexed data for the field remains in storage but becomes inaccessible. Per-field analyzers and embedders are unregistered.

### Upsert a Document (PUT)

Replaces the document if it already exists:

```bash
curl -X PUT http://localhost:8080/v1/documents/doc1 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Hello World",
      "body": "This is a test document."
    }
  }'
```

### Add a Document (POST)

Adds a new chunk without replacing existing documents with the same ID:

```bash
curl -X POST http://localhost:8080/v1/documents/doc1 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Hello World",
      "body": "This is a test document."
    }
  }'
```

### Bulk-Ingest Documents (POST)

Applies many documents in one call — entries are processed sequentially, in
input order, with one WAL fsync for the whole batch. `?mode=put` (the
default) upserts (duplicate ids dedup, last wins); `?mode=add` appends
chunks, so repeated ids accumulate:

```bash
curl -X POST 'http://localhost:8080/v1/documents:bulk?mode=put' \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {"id": "doc1", "fields": {"title": "Hello"}},
      {"id": "doc2", "fields": {"title": "World"}}
    ]
  }'
# => {"applied": 2}
```

The call fails fast at the first entry that cannot be applied;
already-applied entries are not rolled back (they become durable at the next
commit), and the error names the failing position, so retrying the batch or
its suffix is idempotent.

### Get Documents

```bash
curl http://localhost:8080/v1/documents/doc1
```

### Delete Documents

```bash
curl -X DELETE http://localhost:8080/v1/documents/doc1
```

### Commit

```bash
curl -X POST http://localhost:8080/v1/commit
```

### Flush WAL

Forces buffered WAL records durable without a full commit. Returns `{}` on success. This is a near no-op under the default per-record sync policy; under the group-commit policy it flushes the current partial batch on demand. Buffered changes stay invisible to search until a subsequent `POST /v1/commit`.

```bash
curl -X POST http://localhost:8080/v1/flush_wal
```

### Search

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:test", "limit": 10}'
```

#### Search with Field Boosts

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "rust programming",
    "limit": 10,
    "field_boosts": {"title": 2.0}
  }'
```

#### Hybrid Search

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "body:rust",
    "query_vectors": [{"vector": [0.1, 0.2, 0.3], "weight": 1.0}],
    "limit": 10,
    "fusion": {"rrf": {"k": 60}}
  }'
```

#### Search with Highlighting

`highlight` requests highlighted fragments per field (Issue #1134). The
shorthand form is just a field list:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:rust", "limit": 10, "highlight": ["body"]}'
```

The full object form adds `HighlightConfig` knobs — `max_fragments`,
`fragment_size`, `tag`, `css_class`, `require_field_match`,
`max_analyzed_chars`, `return_entire_field_if_no_highlight`:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "body:rust",
    "limit": 10,
    "highlight": {"fields": ["body"], "max_fragments": 2, "tag": "em"}
  }'
```

Each result gains a `"highlights"` object, present only when at least one
field actually highlighted:

```json
{"id": "doc1", "score": 1.2, "fields": {...}, "highlights": {"body": ["<em>Rust</em> is a systems programming language"]}}
```

`highlight` only affects fields that are `stored: true` text fields in the
schema, and highlighting always follows the request's lexical query — a
`filter_query` never contributes highlighted terms and a vector-only
request produces no `highlights` at all. See [Highlighting](../laurus/highlighting.md)
for the full semantics.

### Streaming Search (SSE)

The `/v1/search/stream` endpoint returns results as Server-Sent Events (SSE). Each result is sent as a separate event:

```bash
curl -N -X POST http://localhost:8080/v1/search/stream \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:test", "limit": 10}'
```

The response is a stream of SSE events:

```text
data: {"id":"doc1","score":0.8532,"fields":{...}}

data: {"id":"doc2","score":0.4210,"fields":{...}}
```

## JSON Field Value Inference

When the gateway accepts a document body (`PUT /v1/documents/{id}` or
`POST /v1/documents/{id}`), each value inside `fields` is converted to the
engine's [`DataValue`](../concepts/schema_and_fields.md) type using the same
canonical `json_to_document` converter laurus-cli and laurus-mcp use, so all
three JSON-accepting transports agree on one document shape and the same
inference rules as schema-less ingestion. This keeps the HTTP and gRPC
paths in sync.

| JSON value | Resulting field type | Notes |
| :--- | :--- | :--- |
| `null` | (skipped) | The field is omitted entirely — it is not sent to the engine at all, not even as an explicit null. |
| `true` / `false` | `boolean` | |
| integer (fits in `i64`) | `integer` | |
| float / large integer | `float` | |
| `"text"` | `text` | |
| `[1, 2, 3]` (all integers) | `integer` with `multi_valued: true` | Multi-valued numeric field. |
| `[1.0, 2.5]` (any non-integer number) | `float` with `multi_valued: true` | |
| `[]` (empty array) | (skipped) | Element type cannot be determined, so the field is skipped. |
| `{"latitude": ..., "longitude": ...}` | `geo` | |
| `{"lat": ..., "lon": ...}` / `{"lat": ..., "lng": ...}` | `geo` | Short aliases for latitude / longitude are accepted. |
| `{"x": ..., "y": ..., "z": ...}` | `geo3d` | All three keys required, finite numbers, ECEF meters. Mixing with `lat`/`lon` keys is rejected. |
| `[{"latitude": 35.6, "longitude": 139.7}, ...]` (all geo objects) | `geo` with `multi_valued: true` | Multi-valued geo field; the `lat` / `lon` / `lng` aliases are accepted. Documents are returned with the field rendered as an array of `{"latitude", "longitude"}` objects. |
| `[{"x": ..., "y": ..., "z": ...}, ...]` (all 3D objects) | `geo3d` with `multi_valued: true` | Multi-valued 3D geo field, returned as an array of `{"x", "y", "z"}` objects. Mixing 2D and 3D objects in one array is rejected. |
| `["2024-01-01T00:00:00Z", "2024-06-15T21:00:00+09:00"]` (all RFC 3339 strings) | `date_time` with `multi_valued: true` | Multi-valued datetime field (Issue #1184). Only RFC 3339 strings are accepted here. Documents are returned with the field rendered as an array of RFC 3339 strings normalized to UTC (e.g. `"2024-06-15T12:00:00+00:00"`). |
| `{"data": "<base64>", "mime": "..."}` | `bytes` | `mime` is optional. Disambiguates a bytes payload from a plain string on a multimodal vector field's `Text`-or-`Bytes` embedder input — see [Schema and Fields](../concepts/schema_and_fields.md). |

The gateway returns an HTTP 400 (`Bad Request`) when:

- An array contains mixed types or non-numeric elements
  (e.g. `[1, "x"]`), or mixes 2D and 3D geo objects
  (e.g. `[{"lat": ...}, {"x": ...}]`).
- An array of strings where any element is not an RFC 3339 datetime
  (e.g. `["2024-01-01T00:00:00Z", "tomorrow"]`) — multi-valued text
  fields are not supported (Issue #1175).
- An object does not match any of the supported shapes above (e.g. missing
  latitude / longitude keys for 2D geo, missing any of `x` / `y` / `z` for
  3D geo, or a non-string `data` key).
- A geographic latitude is outside `[-90, 90]` or a longitude is outside
  `[-180, 180]`.
- An object mixes markers from more than one supported shape (e.g. `lat`
  together with `x`, or `data` together with `lat`).
- A 3D ECEF coordinate is non-finite (`NaN` / `Inf`).
- An object mixes 2D (`lat` / `lon`) and 3D (`x` / `y` / `z`) keys.

Vector and bytes fields cannot be inferred from JSON alone and must be
declared in the schema. Numeric arrays sent against a declared vector
field are coerced to a vector of `f32` values automatically, so REST
clients can post embeddings as plain JSON arrays.

### 3D Geographic Queries

3D ECEF queries reuse the lexical DSL string passed via `query`. The gateway forwards it unchanged to the engine, so the same forms work over HTTP as over gRPC:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "position:geo3d_distance(-3955182, 3350553, 3700276, 5000)",
    "limit": 10
  }'
```

See [Query DSL → 3D Geographic Queries](../concepts/query_dsl.md#3d-geographic-queries-geo3d_) for `geo3d_bbox` and `geo3d_nearest` syntax.

## Request/Response Format

All request and response bodies use JSON. The JSON structure mirrors the gRPC protobuf messages. See [gRPC API Reference](grpc_api.md) for the full message definitions.
