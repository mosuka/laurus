# MCP Tools Reference

The laurus MCP server exposes the following tools.

## connect

Connect to a running laurus-server gRPC endpoint. Call this before using other
tools if the server was started without the `--endpoint` flag, or to switch to
a different laurus-server at runtime.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `endpoint` | string | Yes | gRPC endpoint URL (e.g. `http://localhost:50051`) |

### Example

```text
Tool: connect
endpoint: "http://localhost:50051"
```

Result: `Connected to laurus-server at http://localhost:50051.`

---

## create_index

Create a new search index with the provided schema.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `schema_json` | string | Yes | Schema definition as a JSON string |

### Schema JSON format

FieldOption uses serde's externally-tagged representation where the variant name is the key:

```json
{
  "dynamic_field_policy": "Dynamic",
  "fields": {
    "title":     { "Text":    { "indexed": true, "stored": true } },
    "body":      { "Text":    {} },
    "score":     { "Float":   {} },
    "count":     { "Integer": {} },
    "active":    { "Boolean": {} },
    "created":   { "DateTime": {} },
    "embedding": { "Hnsw":    { "dimension": 384 } }
  }
}
```

The optional `dynamic_field_policy` key controls how fields that appear in
ingested documents but are absent from the schema are handled. Accepted
values: `"Strict"`, `"Dynamic"` (default), `"Ignore"`. **Warning**: under
`"Dynamic"`, integer fields silently truncate incoming float values
(`3.14` → `3`); use `"Strict"` to reject such mismatches. See
[Schema & Fields](../concepts/schema_and_fields.md#dynamic-schema) for the
full behaviour matrix.

### Example

```text
Tool: create_index
schema_json: {"fields": {"title": {"Text": {}}, "body": {"Text": {}}}}
```

Result: `Index created successfully at /path/to/index.`

A schema with a `Geo3d` field for 3D ECEF positions:

```json
{
  "fields": {
    "title":    { "Text":  { "indexed": true, "stored": true } },
    "position": { "Geo3d": { "indexed": true, "stored": true } }
  }
}
```

See [3D Geographic Search (ECEF)](../concepts/geo3d.md) for the coordinate system; `Geo3d` is queryable via the `geo3d_distance` / `geo3d_bbox` / `geo3d_nearest` DSL forms (see the **search** tool below).

---

## add_field

Add a new field to the index.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `name` | string | Yes | The field name |
| `field_option_json` | string | Yes | Field configuration as JSON |

### Example

```json
{
  "name": "category",
  "field_option_json": "{\"Text\": {\"indexed\": true, \"stored\": true}}"
}
```

Result: `Field 'category' added successfully.`

---

## delete_field

Remove a field from the index schema. Existing indexed data remains in
storage but becomes inaccessible. Per-field analyzers and embedders are
unregistered.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `name` | string | Yes | The name of the field to remove |

### Example

```json
{
  "name": "category"
}
```

Result: `Field 'category' deleted successfully.`

---

## get_stats

Get statistics for the current search index, including document count and vector field information.

### Parameters

None.

### Result

```json
{
  "document_count": 42,
  "vector_fields": {
    "embedding": {
      "vector_count": 42,
      "dimension": 384
    }
  }
}
```

`vector_fields` is a map keyed by field name; each entry reports the number of indexed vectors and the field's configured dimension.

---

## get_schema

Get the current index schema, including all field definitions and their configurations.

### Parameters

None.

### Result

```json
{
  "fields": {
    "title": { "Text": { "indexed": true, "stored": true } },
    "body": { "Text": {} },
    "embedding": { "Hnsw": { "dimension": 384 } }
  },
  "default_fields": ["title", "body"]
}
```

---

## put_document

Put (upsert) a document into the index. If a document with the same ID already exists, all its chunks are deleted before the new document is indexed. Call `commit` after adding documents.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | string | Yes | External document identifier |
| `fields` | object | Yes | Document fields as a JSON object |

### Example

```text
Tool: put_document
id: "doc-1"
fields: {"title": "Hello World", "body": "This is a test document."}
```

Result: `Document 'doc-1' put (upserted). Call commit to persist changes.`

**Example with a Geo3d value:**

```text
Tool: put_document
id: "drone-1"
fields: {"title": "Drone over Tokyo", "position": {"x": -3955182.0, "y": 3350553.0, "z": 3700276.0}}
```

The MCP server accepts a 3D ECEF point as a JSON object with `x`, `y`, `z` keys (meters) — the same shape the HTTP gateway and laurus-cli accept (see [JSON Field Value Inference](../laurus-server/http_gateway.md#json-field-value-inference)).

---

## add_document

Add a document as a new chunk to the index. Unlike `put_document`, this appends without deleting existing documents with the same ID. Useful for splitting large documents into chunks. Call `commit` after adding documents.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | string | Yes | External document identifier |
| `fields` | object | Yes | Document fields as a JSON object |

### Example

```text
Tool: add_document
id: "doc-1"
fields: {"title": "Hello World - Part 2", "body": "This is a continuation."}
```

Result: `Document 'doc-1' added as chunk. Call commit to persist changes.`

---

## put_documents

Put (upsert) many documents in one call. Entries are applied sequentially, in input order, with one WAL fsync for the whole batch — much faster than calling `put_document` per document. Duplicate IDs within one batch dedup (the last occurrence wins). Call `commit` afterwards.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `documents` | array | Yes | Array of `{"id": "...", "fields": {...}}` entries; each entry's `fields` has the same shape as `put_document`'s |

### Example

```text
Tool: put_documents
documents: [
  {"id": "doc-1", "fields": {"title": "Hello"}},
  {"id": "doc-2", "fields": {"title": "World"}}
]
```

Result: `2 documents put (upserted). Call commit to persist changes.`

A failure aborts at the offending entry without rolling back the already-applied prefix, so retrying the batch (or its suffix) is idempotent.

---

## add_documents

Add many documents as new chunks in one call. Unlike `put_documents`, existing documents are never deleted, so repeating an ID adds multiple chunks of the same logical document. One WAL fsync covers the whole batch. Call `commit` afterwards.

### Parameters

Same as `put_documents`.

### Example

```text
Tool: add_documents
documents: [
  {"id": "doc-1", "fields": {"title": "Part 1"}},
  {"id": "doc-1", "fields": {"title": "Part 2"}}
]
```

Result: `2 documents added as chunks. Call commit to persist changes.`

---

## get_documents

Retrieve all stored documents (including chunks) by external ID.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | string | Yes | External document identifier |

### Result

```json
{
  "id": "doc-1",
  "documents": [
    { "fields": { "title": "Hello World", "body": "This is a test document." } }
  ]
}
```

---

## delete_documents

Delete all documents and chunks sharing the given external ID from the index. Call `commit` after deletion.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | string | Yes | External document identifier |

Result: `Documents 'doc-1' deleted. Call commit to persist changes.`

---

## commit

Commit pending changes to disk. Must be called after `put_document`, `add_document`, or `delete_documents` to make changes searchable and durable.

### Parameters

None.

Result: `Changes committed successfully.`

---

## search

Search documents using the laurus unified query DSL. Supports lexical search, vector search, and hybrid search in a single query string.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `query` | string | Yes | Search query in laurus unified query DSL |
| `limit` | integer | No | Maximum results (default: 10) |
| `offset` | integer | No | Results to skip for pagination (default: 0) |
| `fusion` | string | No | Fusion algorithm as JSON (for hybrid search) |
| `field_boosts` | string | No | Per-field boost factors as JSON |
| `highlight` | string | No | Highlighted fragments per field, as JSON (see [Highlight example](#highlight-example)) |

### Query DSL examples

#### Lexical search

| Query | Description |
| :--- | :--- |
| `hello` | Term search across default fields |
| `title:hello` | Field-scoped term search |
| `title:hello AND body:world` | Boolean AND |
| `"exact phrase"` | Phrase search |
| `roam~2` | Fuzzy search (edit distance 2) |
| `count:[1 TO 10]` | Range search |
| `title:helo~1` | Fuzzy field search |

#### 3D geographic search

| Query | Description |
| :--- | :--- |
| `position:geo3d_distance(x, y, z, distance_m)` | Sphere with center `(x, y, z)` and maximum distance in meters |
| `position:geo3d_bbox(min_x, min_y, min_z, max_x, max_y, max_z)` | 3D axis-aligned bounding box |
| `position:geo3d_nearest(x, y, z, k)` | k nearest neighbors to `(x, y, z)` |

`position` is the field name; substitute the actual `Geo3d`-typed field declared in your schema. See [Query DSL → 3D Geographic Queries](../concepts/query_dsl.md#3d-geographic-queries-geo3d_) for the full DSL syntax.

#### Vector search

| Query | Description |
| :--- | :--- |
| `content:"cute kitten"` | Vector search on a field (field must be a vector field in schema) |
| `content:python` | Vector search with unquoted text |
| `content:"cute kitten"^0.8` | Vector search with weight/boost |
| `a:"cats" b:"dogs"^0.5` | Multiple vector queries |

#### Hybrid search

| Query | Description |
| :--- | :--- |
| `title:hello content:"cute kitten"` | Lexical + vector (OR/union — results from either) |
| `title:hello +content:"cute kitten"` | Lexical + vector (AND/intersection — only results in both) |
| `+title:hello +content:"cute kitten"` | Both required (AND); `+` on lexical field = required clause |
| `title:hello AND body:world content:"cats"^0.8` | Boolean lexical + weighted vector |

### Fusion algorithm examples

```json
{"rrf": {"k": 60.0}}
```

```json
{"weighted_sum": {"lexical_weight": 0.7, "vector_weight": 0.3}}
```

### Field boosts example

```json
{"title": 2.0, "body": 1.0}
```

### Highlight example

`highlight` requests highlighted fragments per field (Issue #1134),
following this tool's lexical query — a vector-only query produces no
highlights, and only `stored: true` text fields can be highlighted. The
shorthand form is just a field list:

```json
["body"]
```

The full object form adds `HighlightConfig` knobs — `max_fragments`,
`fragment_size`, `tag`, `css_class`, `require_field_match`,
`max_analyzed_chars`, `return_entire_field_if_no_highlight`:

```json
{"fields": ["body"], "max_fragments": 2, "tag": "em"}
```

### Result

Each result gains a `"highlights"` object when `highlight` was requested
and at least one field actually highlighted:

```json
{
  "total": 2,
  "results": [
    {
      "id": "doc-1",
      "score": 3.14,
      "fields": { "title": "Hello World", "body": "..." },
      "highlights": { "body": ["<em>Hello</em> World"] }
    },
    {
      "id": "doc-2",
      "score": 1.57,
      "fields": { "title": "Hello Again", "body": "..." }
    }
  ]
}
```

---

## search_batch

Execute multiple independent searches in a single round trip. All queries
run in parallel on the server; the same `limit` and `offset` apply to every
query. Useful for agents issuing several sub-queries per turn.

### Parameters

| Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `queries` | array of string | Yes | Query strings, each in the laurus unified query DSL (same syntax as `search`) |
| `limit` | integer | No | Maximum results per query (default: 10) |
| `offset` | integer | No | Results to skip per query for pagination (default: 0) |
| `highlight` | string | No | Highlighted fragments per field, as JSON — same format as `search`'s `highlight`, applied identically to every query |

### Result

The `batch` array preserves input order: `batch[i]` is the result set for
`queries[i]`.

```json
{
  "batch": [
    {
      "total": 1,
      "results": [
        { "id": "doc-1", "score": 3.14, "fields": { "title": "Hello World" } }
      ]
    },
    {
      "total": 1,
      "results": [
        { "id": "doc-2", "score": 2.71, "fields": { "title": "Vector Search" } }
      ]
    }
  ]
}
```

---

## Typical Workflow

```text
1. connect          → connect to a running laurus-server
2. create_index     → define the schema (if index does not exist)
3. add_field        → dynamically add fields (optional)
   delete_field     → remove fields (optional)
4. put_document     → upsert documents (repeat as needed)
   add_document     → append document chunks (optional)
5. commit           → persist changes to disk
6. search           → query the index
7. get_documents    → retrieve documents by ID
8. delete_documents → remove documents
9. commit           → persist changes
```
