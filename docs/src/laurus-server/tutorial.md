# Hands-on Tutorial

This tutorial walks you through a complete workflow with laurus-server: starting the server, creating an index, adding documents, searching, updating, and deleting. All examples use `curl` via the HTTP Gateway.

## Prerequisites

- laurus CLI installed (see [Installation](../getting_started/installation.md))
- `curl` available on your system

## Step 1: Start the Server

Start laurus-server with the HTTP Gateway enabled:

```bash
laurus --index-dir /tmp/laurus/tutorial serve --port 50051 --http-port 8080
```

You should see log output indicating the gRPC server (port 50051) and the HTTP Gateway (port 8080) have started.

Verify the server is running:

```bash
curl http://localhost:8080/v1/health
```

Expected response:

```json
{"status":"SERVING_STATUS_SERVING"}
```

## Step 2: Create an Index

Create an index with a schema that defines text fields for lexical search and a vector field for vector search. This example demonstrates **custom analyzers**, **embedder definitions**, and per-field configuration:

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "schema": {
      "analyzers": {
        "body_analyzer": {
          "char_filters": [{"type": "unicode_normalization", "form": "nfkc"}],
          "tokenizer": {"type": "regex"},
          "token_filters": [
            {"type": "lowercase"},
            {"type": "stop", "words": ["the", "a", "an", "is", "it"]}
          ]
        }
      },
      "embedders": {
        "my_embedder": {"type": "precomputed"}
      },
      "fields": {
        "title": {"text": {"indexed": true, "stored": true, "term_vectors": true, "analyzer": "standard"}},
        "body": {"text": {"indexed": true, "stored": true, "term_vectors": true, "analyzer": "body_analyzer"}},
        "category": {"text": {"indexed": true, "stored": true, "term_vectors": false, "analyzer": "keyword"}},
        "embedding": {"hnsw": {"dimension": 4, "distance": "DISTANCE_METRIC_COSINE", "m": 16, "ef_construction": 200, "embedder": "my_embedder"}}
      },
      "default_fields": ["title", "body"]
    }
  }'
```

This creates an index with three text fields and one vector field:

- `title` — uses the built-in `standard` analyzer (tokenizes and lowercases).
- `body` — uses the custom `body_analyzer` defined in the `analyzers` section (NFKC normalization + regex tokenizer + lowercase + custom stop words).
- `category` — uses the `keyword` analyzer (treats the entire value as a single token for exact matching).
- `embedding` — HNSW vector index with 4 dimensions, cosine distance, using the `my_embedder` embedder defined in `embedders`. In this tutorial we use `precomputed` (vectors supplied externally). In production, use a dimension matching your embedding model (e.g. 384 or 768).

The `default_fields` setting means that queries without a field prefix will search both `title` and `body`.

### Built-in analyzers

`standard`, `keyword`, `english`, `japanese`, `simple`, `noop`. If omitted, the engine default (`standard`) is used.

### Custom analyzer components

You can compose custom analyzers from the following components:

- **Tokenizers:** `whitespace`, `unicode_word`, `regex`, `ngram`, `lindera`, `whole`
- **Char filters:** `unicode_normalization`, `pattern_replace`, `mapping`, `japanese_iteration_mark`
- **Token filters:** `lowercase`, `stop`, `stem`, `boost`, `limit`, `strip`, `remove_empty`, `flatten_graph`

### Embedders

The `embedders` section defines how vectors are generated. Each vector field can reference an embedder by name via the `embedder` option. Available types:

- `precomputed` — vectors are supplied externally (no automatic embedding).
- `candle_bert` — local BERT model via Candle. Params: `model` (HuggingFace model ID). Requires `embeddings-candle` feature.
- `candle_clip` — local CLIP multimodal model. Params: `model` (HuggingFace model ID). Requires `embeddings-multimodal` feature.
- `openai` — OpenAI API. Params: `model` (e.g. `"text-embedding-3-small"`). Requires `embeddings-openai` feature and `OPENAI_API_KEY` env var.

Example with a BERT embedder (requires the `embeddings-candle` feature):

```json
{
  "embedders": {
    "bert": {"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"}
  },
  "fields": {
    "embedding": {"hnsw": {"dimension": 384, "embedder": "bert"}}
  }
}
```

Verify the index was created:

```bash
curl http://localhost:8080/v1/index
```

Expected response:

```json
{"document_count":0,"vector_fields":{}}
```

## Step 3: Add Documents

Add a few documents to the index. Use `PUT` to upsert documents by ID. Each document includes text fields and an `embedding` vector (in production, these vectors would come from an embedding model):

```bash
curl -X PUT http://localhost:8080/v1/documents/doc001 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Introduction to Rust Programming",
      "body": "Rust is a modern systems programming language that focuses on safety, speed, and concurrency.",
      "category": "programming",
      "embedding": [0.9, 0.1, 0.2, 0.0]
    }
  }'
```

```bash
curl -X PUT http://localhost:8080/v1/documents/doc002 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Web Development with Rust",
      "body": "Building web applications with Rust has become increasingly popular. Frameworks like Actix and Rocket make it easy to create fast and secure web services.",
      "category": "web-development",
      "embedding": [0.7, 0.3, 0.5, 0.1]
    }
  }'
```

```bash
curl -X PUT http://localhost:8080/v1/documents/doc003 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Python for Data Science",
      "body": "Python is the most popular language for data science and machine learning. Libraries like NumPy and Pandas provide powerful tools for data analysis.",
      "category": "data-science",
      "embedding": [0.1, 0.8, 0.1, 0.9]
    }
  }'
```

Vector fields are specified as JSON arrays of numbers. The array length must match the `dimension` configured in the schema (4 in this tutorial).

## Step 4: Commit Changes

Documents are not searchable until committed. Commit the pending changes:

```bash
curl -X POST http://localhost:8080/v1/commit
```

## Step 5: Search Documents

### Basic Search

Search for documents containing "rust":

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "rust", "limit": 10}'
```

This searches the default fields (`title` and `body`). Expected result: `doc001` and `doc002` are returned.

### Field-Specific Search

Search only in the `title` field:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "title:python", "limit": 10}'
```

Expected result: only `doc003` is returned.

### Search by Category

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "category:programming", "limit": 10}'
```

Expected result: only `doc001` is returned.

### Boolean Queries

Combine conditions with `AND`, `OR`, and `NOT`:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "rust AND web", "limit": 10}'
```

Expected result: only `doc002` is returned (contains both "rust" and "web").

### Field Boosting

Boost the `title` field to prioritize title matches:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "rust",
    "limit": 10,
    "field_boosts": {"title": 2.0}
  }'
```

### Vector Search

Search by vector similarity. Provide a query vector in `query_vectors` and specify which field to search:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query_vectors": [
      {
        "vector": [0.85, 0.15, 0.2, 0.05],
        "fields": ["embedding"]
      }
    ],
    "limit": 10
  }'
```

This finds documents whose `embedding` vectors are closest to the query vector. Expected result: `doc001` ranks highest (most similar vector).

### Hybrid Search

Combine lexical search and vector search for best results. The `fusion` parameter controls how scores from both searches are merged:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "rust",
    "query_vectors": [
      {
        "vector": [0.85, 0.15, 0.2, 0.05],
        "fields": ["embedding"]
      }
    ],
    "fusion": {"rrf": {"k": 60.0}},
    "limit": 10
  }'
```

This uses Reciprocal Rank Fusion (RRF) to merge lexical and vector search results. You can also use weighted sum fusion:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "programming",
    "query_vectors": [
      {
        "vector": [0.85, 0.15, 0.2, 0.05],
        "fields": ["embedding"]
      }
    ],
    "fusion": {"weighted_sum": {"lexical_weight": 0.3, "vector_weight": 0.7}},
    "limit": 10
  }'
```

## Step 6: Retrieve a Document

Fetch a specific document by its ID:

```bash
curl http://localhost:8080/v1/documents/doc001
```

Expected response (includes the stored vector field):

```json
{
  "documents": [
    {
      "fields": {
        "title": "Introduction to Rust Programming",
        "body": "Rust is a modern systems programming language that focuses on safety, speed, and concurrency.",
        "category": "programming",
        "embedding": [0.9, 0.1, 0.2, 0.0]
      }
    }
  ]
}
```

## Step 7: Update a Document

Update a document by `PUT`-ing with the same ID. This replaces the entire document:

```bash
curl -X PUT http://localhost:8080/v1/documents/doc001 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Introduction to Rust Programming",
      "body": "Rust is a modern systems programming language that focuses on safety, speed, and concurrency. It provides memory safety without garbage collection.",
      "category": "programming",
      "embedding": [0.9, 0.1, 0.2, 0.0]
    }
  }'
```

Commit and verify:

```bash
curl -X POST http://localhost:8080/v1/commit
curl http://localhost:8080/v1/documents/doc001
```

The updated body text is now stored.

## Step 8: Delete a Document

Delete a document by its ID:

```bash
curl -X DELETE http://localhost:8080/v1/documents/doc003
```

Commit and verify:

```bash
curl -X POST http://localhost:8080/v1/commit
```

Confirm the document was deleted:

```bash
curl http://localhost:8080/v1/documents/doc003
```

Expected response:

```json
{"documents":[]}
```

Search results will no longer include the deleted document:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "python", "limit": 10}'
```

Expected result: no results returned.

## Step 9: Check Index Statistics

View the current index statistics:

```bash
curl http://localhost:8080/v1/index
```

The `document_count` should reflect the remaining documents after the deletion.

## Step 10: Clean Up

Stop the server with `Ctrl+C`. The server performs a graceful shutdown, committing any pending changes before exiting.

To remove the tutorial data:

```bash
rm -rf /tmp/laurus/tutorial
```

## Going Further: Using a Real Embedding Model

The tutorial above uses `precomputed` vectors for simplicity. In production, you typically use an embedding model to automatically convert text into vectors. Here is how to set up a BERT-based embedder.

### Prerequisites

Build laurus with the `embeddings-candle` feature:

```bash
cargo build --release --features embeddings-candle
```

### Schema with BERT Embedder

Create an index:

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "schema": {
      "embedders": {
        "bert": {
          "type": "candle_bert",
          "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
      },
      "fields": {
        "title": {"text": {"indexed": true, "stored": true, "analyzer": "standard"}},
        "body": {"text": {"indexed": true, "stored": true, "analyzer": "standard"}},
        "embedding": {"hnsw": {"dimension": 384, "distance": "DISTANCE_METRIC_COSINE", "m": 16, "ef_construction": 200, "embedder": "bert"}}
      },
      "default_fields": ["title", "body"]
    }
  }'
```

The model is automatically downloaded from HuggingFace Hub on first use. The `dimension` (384) must match the model's output dimension.

Add documents. Pass text to the `embedding` field — the embedder automatically converts it to a vector:

```bash
curl -X PUT http://localhost:8080/v1/documents/doc001 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Introduction to Rust Programming",
      "body": "Rust is a modern systems programming language.",
      "embedding": "Rust is a modern systems programming language."
    }
  }'
```

```bash
curl -X PUT http://localhost:8080/v1/documents/doc002 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Web Development with Rust",
      "body": "Building web applications with Rust using Actix and Rocket.",
      "embedding": "Building web applications with Rust using Actix and Rocket."
    }
  }'
```

Commit:

```bash
curl -X POST http://localhost:8080/v1/commit
```

Search with both lexical and semantic queries in one `query`: `systems programming` is a lexical clause, and `embedding:"..."` is a vector clause whose text the embedder converts to a vector at search time:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "systems programming embedding:\"systems programming language\"",
    "fusion": {"rrf": {"k": 60.0}},
    "limit": 10
  }'
```

With the `precomputed` embedder you must pass raw vectors (in `query_vectors`, whose `vector` must be an array of numbers), but text-capable embedders like `candle_bert` accept text directly for both indexing and searching.

### Using OpenAI Embeddings

For OpenAI's embedding API, set the `OPENAI_API_KEY` environment variable and build with the `embeddings-openai` feature:

```bash
cargo build --release --features embeddings-openai
export OPENAI_API_KEY="sk-..."
```

Create an index:

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "schema": {
      "embedders": {
        "openai": {
          "type": "openai",
          "model": "text-embedding-3-small"
        }
      },
      "fields": {
        "title": {"text": {"indexed": true, "stored": true}},
        "embedding": {"hnsw": {"dimension": 1536, "distance": "DISTANCE_METRIC_COSINE", "embedder": "openai"}}
      },
      "default_fields": ["title"]
    }
  }'
```

The `text-embedding-3-small` model outputs 1536-dimensional vectors.

### Available Embedding Models

| Type | Feature Flag | Example Model | Dimension |
| :--- | :--- | :--- | :--- |
| `candle_bert` | `embeddings-candle` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `candle_clip` | `embeddings-multimodal` | `openai/clip-vit-base-patch32` | 512 |
| `openai` | `embeddings-openai` | `text-embedding-3-small` | 1536 |

## Next Steps

- Learn about [vector search and hybrid search](../concepts/search/hybrid_search.md) for semantic similarity queries
- Explore the [gRPC API Reference](grpc_api.md) for the full API specification
- Configure the server for production using [Configuration](configuration.md)
- Use `grpcurl` or a gRPC client library for programmatic access — see [Getting Started](getting_started.md)
