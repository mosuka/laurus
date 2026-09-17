# Command Reference

## Global Options

Every command accepts these options:

| Option | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--index-dir <PATH>` | `LAURUS_INDEX_DIR` | `./laurus_index` | Path to the index data directory |
| `--format <FORMAT>` | — | `table` | Output format: `table` or `json` |

```bash
# Example: use JSON output with a custom data directory
laurus --index-dir /var/data/my_index --format json search "title:rust"
```

---

## `create` — Create a Resource

### `create index`

Create a new index. If `--schema` is given, uses that TOML file; otherwise launches the interactive schema wizard.

```bash
laurus create index [--schema <FILE>] [--train-pq-codebook <JSONL>]
```

**Arguments:**

| Flag | Required | Description |
| :--- | :--- | :--- |
| `--schema <FILE>` | No | Path to a TOML file defining the index schema. When omitted, the command checks if a `schema.toml` already exists in the index directory and uses it; otherwise the interactive wizard is launched. |
| `--train-pq-codebook <JSONL>` | No | Train shared PQ codebooks as part of creation (Issue #920). Every HNSW field configuring `ProductQuantization` (or, with the `pq-fastscan` feature, `ProductQuantizationFastScan`) + `pq_codebook_path` is trained from this JSONL file (the `put docs` / `add docs` shape; each field value a plain numeric array) immediately after the index is created, so the very first commit can already encode against the codebook — removing the create → `train pq-codebook` → ingest ordering the failure policy otherwise requires you to manage manually. Errors before creating anything if the file is missing or no field is eligible. |

**Schema file format:**

The schema file follows the same structure as the `Schema` type in the Laurus library. See [Schema Format Reference](schema_format.md) for full details. Example:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
stored = true
indexed = true

[fields.body.Text]
stored = true
indexed = true

[fields.category.Text]
stored = true
indexed = true
```

**Examples:**

```bash
# From a schema file
laurus --index-dir ./my_index create index --schema schema.toml
# Index created at ./my_index.

# Interactive wizard (no --schema flag)
laurus --index-dir ./my_index create index
# === Laurus Schema Generator ===
# Field name: title
# ...
# Index created at ./my_index.

# Create and train the shared PQ codebook in one step (Issue #920)
laurus --index-dir ./my_index create index --schema schema.toml \
    --train-pq-codebook train.jsonl
# Index created at ./my_index.
# Training PQ codebook for field 'embedding' on 300 vectors...
# Trained codebook 'embedding.pqcb' (m = 4, k = 256, sub_dim = 8, dimension = 32) from 300 vectors.
```

> **Note:** If both `schema.toml` and `store/` already exist, an error is returned. Delete the index directory to recreate. If only `schema.toml` exists (e.g. after an interrupted creation), running `create index` without `--schema` recovers the index by creating the missing storage from the existing schema.

### `create schema`

Interactively generate a schema TOML file through a guided wizard.

```bash
laurus create schema [--output <FILE>]
```

**Arguments:**

| Flag | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `--output <FILE>` | No | `schema.toml` | Output file path for the generated schema |

The wizard guides you through:

1. **Field definition** — Enter a field name, select the type, and configure type-specific options
2. **Repeat** — Add as many fields as needed
3. **Default fields** — Select which lexical fields to use as default search fields
4. **Preview** — Review the generated TOML before saving
5. **Save** — Write the schema file

**Supported field types:**

| Type | Category | Options |
| :--- | :--- | :--- |
| `Text` | Lexical | `indexed`, `stored`, `term_vectors` |
| `Integer` | Lexical | `indexed`, `stored` |
| `Float` | Lexical | `indexed`, `stored` |
| `Boolean` | Lexical | `indexed`, `stored` |
| `DateTime` | Lexical | `indexed`, `stored` |
| `Geo` | Lexical | `indexed`, `stored` |
| `Geo3d` | Lexical | `indexed`, `stored` |
| `Bytes` | Lexical | `stored` |
| `Hnsw` | Vector | `dimension`, `distance`, `m`, `ef_construction` |
| `Flat` | Vector | `dimension`, `distance` |
| `Ivf` | Vector | `dimension`, `distance`, `n_clusters`, `n_probe` |

**Example:**

```bash
# Generate schema.toml interactively
laurus create schema

# Specify output path
laurus create schema --output my_schema.toml

# Then create an index from the generated schema
laurus create index --schema schema.toml
```

---

## `get` — Get a Resource

### `get stats`

Display statistics about the index.

```bash
laurus get stats
```

**Table output example:**

```text
Document count: 42

Vector fields:
╭──────────┬─────────┬───────────╮
│ Field    │ Vectors │ Dimension │
├──────────┼─────────┼───────────┤
│ text_vec │ 42      │ 384       │
╰──────────┴─────────┴───────────╯
```

**JSON output example:**

```bash
laurus --format json get stats
```

```json
{
  "document_count": 42,
  "fields": {
    "text_vec": {
      "vector_count": 42,
      "dimension": 384
    }
  }
}
```

### `get schema`

Display the current index schema as JSON.

```bash
laurus get schema
```

**Example:**

```bash
laurus get schema
# {
#   "fields": { ... },
#   "default_fields": ["title", "body"],
#   ...
# }
```

### `get docs`

Retrieve all documents (including chunks) by external ID.

```bash
laurus get docs --id <ID>
```

**Table output example:**

```text
╭──────┬─────────────────────────────────────────╮
│ ID   │ Fields                                  │
├──────┼─────────────────────────────────────────┤
│ doc1 │ body: This is a test, title: Hello World │
╰──────┴─────────────────────────────────────────╯
```

**JSON output example:**

```bash
laurus --format json get docs --id doc1
```

```json
[
  {
    "id": "doc1",
    "document": {
      "title": "Hello World",
      "body": "This is a test document."
    }
  }
]
```

---

## `add` — Add a Resource

### `add doc`

Add a document to the index. Documents are not searchable until `commit` is called.

```bash
laurus add doc --id <ID> --data <JSON>
```

**Arguments:**

| Flag | Required | Description |
| :--- | :--- | :--- |
| `--id <ID>` | Yes | External document ID (string) |
| `--data <JSON>` | Yes | Document fields as a JSON string |

The JSON is `{"fields": {...}}`: an object mapping each field name to its
plain value. There are no type tags — the value's declared schema type (or,
for an undeclared field, its inferred type) resolves any ambiguity:

```json
{
  "fields": {
    "title": "Introduction to Rust",
    "body": "Rust is a systems programming language.",
    "year": 2024
  }
}
```

**Example:**

```bash
laurus add doc --id doc1 --data '{"fields":{"title":"Hello World","body":"This is a test document."}}'
# Document 'doc1' added. Run 'commit' to persist changes.
```

> **Tip:** Multiple documents can share the same external ID (chunking pattern). Use `add doc` for each chunk.

### `add docs`

Bulk-add document chunks from a JSONL file — one `{"id": "...", "fields": {...}}` entry per line, with the external ID as a sibling top-level key alongside the same `fields` shape as `add doc --data`. Entries are applied through the engine's batch API (one WAL fsync per batch) and, unlike `add doc`, the command **commits automatically**: every `--commit-every` applied documents and once at the end.

```bash
laurus add docs --file <JSONL> [--batch-size 1000] [--commit-every 0]
```

**Arguments:**

| Flag | Required | Description |
| :--- | :--- | :--- |
| `--file <JSONL>` | Yes | Path to the JSONL file to ingest |
| `--batch-size <N>` | No | Documents per engine batch call (default `1000`) |
| `--commit-every <N>` | No | Commit every N applied documents; `0` = only the final commit (default) |

Repeated IDs accumulate as chunks. On a mid-file failure the error names the offending line, the applied prefix is committed, and re-running the remaining lines continues the ingest.

---

## `put` — Put (Upsert) a Resource

### `put doc`

Put (upsert) a document into the index. If a document with the same ID already exists, all its chunks are deleted before the new document is indexed. Documents are not searchable until `commit` is called.

```bash
laurus put doc --id <ID> --data <JSON>
```

**Arguments:**

| Flag | Required | Description |
| :--- | :--- | :--- |
| `--id <ID>` | Yes | External document ID (string) |
| `--data <JSON>` | Yes | Document fields as a JSON string |

**Example:**

```bash
laurus put doc --id doc1 --data '{"fields":{"title":"Updated Title","body":"This replaces the existing document."}}'
# Document 'doc1' put (upserted). Run 'commit' to persist changes.
```

> **Note:** Unlike `add doc`, `put doc` replaces all existing chunks for the given ID. Use `add doc` when you want to append chunks, and `put doc` when you want to replace the entire document.

### `put docs`

Bulk-upsert documents from a JSONL file — one `{"id": "...", "fields": {...}}` entry per line, applied through the engine's batch API (one WAL fsync per batch). Duplicate IDs dedup in order (the last occurrence wins). Like `add docs`, the command **commits automatically**.

```bash
laurus put docs --file <JSONL> [--batch-size 1000] [--commit-every 0]
```

Arguments are the same as `add docs`. On a mid-file failure the error names the offending line and the applied prefix is committed; because puts are idempotent, re-running the whole file (or its remaining suffix) is safe.

**Example:**

```bash
cat > docs.jsonl <<'JSONL'
{"id": "doc1", "fields": {"title": "Hello"}}
{"id": "doc2", "fields": {"title": "World"}}
JSONL
laurus put docs --file docs.jsonl
# 2 documents put (upserted) and committed.
```

---

### `add field`

Dynamically add a new field to an existing index.

```bash
laurus add field --index-dir ./data \
    --name category \
    --field-option '{"Text": {"indexed": true, "stored": true}}'
```

The `--field-option` argument accepts a JSON string using the same
externally-tagged format as the schema file. The schema is automatically
persisted after the field is added.

---

## `delete` — Delete a Resource

### `delete docs`

Delete all documents (including chunks) by external ID.

```bash
laurus delete docs --id <ID>
```

**Example:**

```bash
laurus delete docs --id doc1
# Documents 'doc1' deleted. Run 'commit' to persist changes.
```

### `delete field`

Remove a field from the index schema.

```bash
laurus delete field --name <FIELD_NAME>
```

**Example:**

```bash
laurus delete field --name category
# Field 'category' deleted.
```

Existing indexed data for the field remains in storage but becomes
inaccessible. Per-field analyzers and embedders are unregistered.

---

## `commit`

Commit pending changes (additions and deletions) to the index. Until committed, changes are not visible to search.

```bash
laurus commit
```

**Example:**

```bash
laurus --index-dir ./my_index commit
# Changes committed successfully.
```

---

## `train`

### `train pq-codebook`

Train a **shared PQ codebook** for an HNSW vector field (Issue #631).
The codebook is trained once on a representative sample and then reused
by every subsequent commit and merge, instead of re-training k-means
per segment — commits on PQ fields get dramatically faster, and even
tiny per-commit segments stay on PQ.

```bash
laurus train pq-codebook --field <FIELD> (--input <JSONL> | --from-index) \
    [--sample-size <N>] [--output <NAME>] [--update-schema]
```

| Argument | Description |
| :--- | :--- |
| `--field` | The HNSW vector field to train for. Must be configured with a `ProductQuantization` quantizer (or `ProductQuantizationFastScan` when the `pq-fastscan` feature is enabled — the codebook is then trained with k=16, Issue #920). |
| `--input` | JSONL training file — the same `{"id": "...", "fields": {...}}` shape as `put docs` / `add docs`. The field value must be a plain numeric array, e.g. `"embedding": [0.1, 0.2, ...]` (embedder-generated input is not supported). Exactly one of `--input` and `--from-index` must be given. |
| `--from-index` | Sample the vectors already committed to this index instead of reading a file (Issue #920) — no JSONL export needed. Exactly one of `--input` and `--from-index` must be given. Note: on a field that is already PQ-encoded, the sampled vectors are lossy reconstructions; the intended flow is to train from vectors committed **before** enabling PQ on the field. |
| `--sample-size` | Use only the first N vectors (deterministic: file order for `--input`, ascending doc_id for `--from-index`). Omit to use all of them; thousands of representative vectors are enough. |
| `--output` | Storage-relative codebook file name. Defaults to the field's configured `pq_codebook_path`, else `{field}.pqcb`. Use to train a v2 codebook alongside a live one. |
| `--update-schema` | Rewrite `schema.toml` so the field's `pq_codebook_path` names the trained file. |

Commits use the codebook only when the schema's `pq_codebook_path`
names it (see [Schema Format](schema_format.md#product-quantization-hnsw-only)) —
pass `--update-schema` to set it as part of training. A commit made
while `pq_codebook_path` is set but the codebook has not been trained
yet fails with an error naming this command; there is no silent
fallback to per-segment training. The codebook is picked up when the
index is opened, so train **before** the ingesting `add` / `put` /
`commit` invocation (each CLI invocation opens the index fresh, so any
subsequent command sees it).

**Example:**

```bash
cat > train.jsonl <<'JSONL'
{"id": "t1", "fields": {"embedding": [0.1, 0.2, 0.3, 0.4]}}
{"id": "t2", "fields": {"embedding": [0.5, 0.6, 0.7, 0.8]}}
JSONL
laurus train pq-codebook --field embedding --input train.jsonl --update-schema
# Training PQ codebook for field 'embedding' on 2 vectors...
# Trained codebook 'embedding.pqcb' (m = 2, k = 256, sub_dim = 2, dimension = 4) from 2 vectors.
# Updated schema.toml: embedding.pq_codebook_path = "embedding.pqcb".
```

Or sample directly from the vectors already committed to the index —
no JSONL export needed:

```bash
laurus train pq-codebook --field embedding --from-index --sample-size 5000 --update-schema
```

---

## `search`

Execute a search query using the [Query DSL](../concepts/query_dsl.md).

```bash
laurus search <QUERY> [--limit <N>] [--offset <N>] [--highlight <FIELD>]...
```

The query string is analyzed with each field's own configured analyzer —
a field declared with a Japanese (Lindera) analyzer in `schema.toml`, for
example, is analyzed the same way at query time as it was at index time.
Referencing a field that is not declared in the schema is rejected with an
error naming the field (helpful for catching typos); the reserved `_id`
field is always queryable even though it does not appear in the schema.

**Arguments:**

| Argument / Flag | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `<QUERY>` | Yes | — | Query string in Laurus Query DSL |
| `--limit <N>` | No | `10` | Maximum number of results |
| `--offset <N>` | No | `0` | Number of results to skip |
| `--highlight <FIELD>` | No | (none) | Stored text field to highlight (Issue #1134); repeat for multiple fields. Uses the default `HighlightConfig` — one-shot only, not available in the [REPL](repl.md) |

**Query syntax examples:**

```bash
# Term query
laurus search "body:rust"

# Phrase query
laurus search 'body:"machine learning"'

# Boolean query
laurus search "+body:programming -body:python"

# Fuzzy query (typo tolerance)
laurus search "body:programing~2"

# Wildcard query
laurus search "title:intro*"

# Range query
laurus search "price:[10 TO 50]"

# 3D geographic queries (sphere / bounding box / k-NN)
laurus search "position:geo3d_distance(-3955182, 3350553, 3700276, 5000)"
laurus search "position:geo3d_bbox(-4000000, 3300000, 3650000, -3900000, 3400000, 3750000)"
laurus search "position:geo3d_nearest(-3955182, 3350553, 3700276, 10)"

# Highlighted fragments on the "body" field
laurus search "body:rust" --highlight body

# Highlight multiple fields
laurus search "body:rust" --highlight title --highlight body
```

**Table output example:**

```text
╭──────┬────────┬─────────────────────────────────────────╮
│ ID   │ Score  │ Fields                                  │
├──────┼────────┼─────────────────────────────────────────┤
│ doc1 │ 0.8532 │ body: Rust is a systems..., title: Intr │
│ doc3 │ 0.4210 │ body: JavaScript powers..., title: Web  │
╰──────┴────────┴─────────────────────────────────────────╯
```

A `Highlights` column appears only when `--highlight` was passed:

```text
╭──────┬────────┬─────────────────────────┬──────────────────────────────╮
│ ID   │ Score  │ Fields                  │ Highlights                   │
├──────┼────────┼─────────────────────────┼──────────────────────────────┤
│ doc1 │ 0.8532 │ body: Rust is a syst... │ body: <mark>Rust</mark> is a │
╰──────┴────────┴─────────────────────────┴──────────────────────────────╯
```

**JSON output example:**

```bash
laurus --format json search "body:rust" --limit 5
```

```json
[
  {
    "id": "doc1",
    "score": 0.8532,
    "fields": {
      "title": "Introduction to Rust",
      "body": "Rust is a systems programming language."
    }
  }
]
```

With `--highlight body`, each matching result also gains a `"highlights"`
key (present only when at least one field actually highlighted):

```json
[
  {
    "id": "doc1",
    "score": 0.8532,
    "fields": {
      "title": "Introduction to Rust",
      "body": "Rust is a systems programming language."
    },
    "highlights": {
      "body": ["<mark>Rust</mark> is a systems programming language."]
    }
  }
]
```

See [Highlighting](../laurus/highlighting.md) for the full semantics —
field selection, per-field analyzers, and why a `filter_query` never
highlights (not reachable from this one-shot form of `search`, but
relevant to any client using the same underlying option).

---

## `repl`

Start an interactive REPL session. See [REPL](repl.md) for details.

```bash
laurus repl
```

---

## `serve`

Start the gRPC server (and optionally the HTTP Gateway).

```bash
laurus serve [OPTIONS]
```

For startup options, configuration, and usage examples, see the [laurus-server documentation](../laurus-server.md):

- [Getting Started](../laurus-server/getting_started.md) — startup options and gRPC connection examples
- [Configuration](../laurus-server/configuration.md) — TOML config file, environment variables, and priority rules
- [Hands-on Tutorial](../laurus-server/tutorial.md) — step-by-step walkthrough

---

## `mcp`

Start the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server on stdio. The MCP server lets AI assistants such as Claude Code or Claude Desktop drive a running laurus-server through a standard set of tools (`create_index`, `add_document`, `search`, etc.).

```bash
laurus mcp [--endpoint <URL>]
```

**Arguments:**

| Flag | Environment Variable | Required | Description |
| :--- | :--- | :--- | :--- |
| `--endpoint <URL>` | `LAURUS_ENDPOINT` | No | gRPC endpoint of a running laurus-server (e.g. `http://localhost:50051`). If omitted, the server starts without a connection; clients can call the `connect` MCP tool later to attach. |

**Examples:**

```bash
# Start the MCP server pre-connected to a local laurus-server
laurus mcp --endpoint http://localhost:50051

# Start the MCP server without a connection; clients call `connect` first
laurus mcp
```

For the full list of MCP tools exposed by this server and how to wire it into Claude Code or Claude Desktop, see the [laurus-mcp documentation](../laurus-mcp.md).
