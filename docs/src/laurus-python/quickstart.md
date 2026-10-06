# Quick Start

## 1. Create an index

```python
import laurus

# In-memory index (ephemeral, useful for prototyping)
index = laurus.Index()

# File-based index (persistent)
# Writes `./myindex/schema.toml` and `./myindex/store/` -- the same layout
# `laurus-cli create index --schema` uses, so this directory can also be
# opened with the CLI (or vice versa).
schema = laurus.Schema()
schema.add_text_field("title")
schema.add_text_field("body")
index = laurus.Index(path="./myindex", schema=schema)

# Reopening it later only needs the path -- passing `schema` again would
# raise, since the schema is already persisted.
index = laurus.Index(path="./myindex")
```

## 2. Index documents

```python
index.put_document("doc1", {
    "title": "Introduction to Rust",
    "body": "Rust is a systems programming language focused on safety and performance.",
})
index.put_document("doc2", {
    "title": "Python for Data Science",
    "body": "Python is widely used for data analysis and machine learning.",
})
index.commit()
```

## 3. Lexical search

```python
# DSL string
results = index.search("title:rust", limit=5)

# Query object
results = index.search(laurus.TermQuery("body", "python"), limit=5)

# Print results
for r in results:
    print(f"[{r.id}] score={r.score:.4f}  {r.document['title']}")
```

## 4. Vector search

Vector search requires a schema with a vector field and pre-computed embeddings.

```python
import laurus
import numpy as np

schema = laurus.Schema()
schema.add_text_field("title")
schema.add_hnsw_field("embedding", dimension=4)

index = laurus.Index(schema=schema)
index.put_document("doc1", {"title": "Rust", "embedding": [0.1, 0.2, 0.3, 0.4]})
index.put_document("doc2", {"title": "Python", "embedding": [0.9, 0.8, 0.7, 0.6]})
index.commit()

query_vec = [0.1, 0.2, 0.3, 0.4]
results = index.search(laurus.VectorQuery("embedding", query_vec), limit=3)
```

## 5. Hybrid search

```python
request = laurus.SearchRequest(
    lexical_query=laurus.TermQuery("title", "rust"),
    vector_query=laurus.VectorQuery("embedding", query_vec),
    fusion=laurus.RRF(k=60.0),
    limit=5,
)
results = index.search(request)
```

## 6. Late-interaction rescore

A late-interaction rescore reorders the top results of any search by ColBERT-style MaxSim against a multi-vector field, which holds each document's token vectors.

```python
import laurus

schema = laurus.Schema()
schema.add_text_field("title")
schema.add_multi_vector_field("tokens", dimension=2, distance="dot_product")

index = laurus.Index(schema=schema)
index.put_document("doc1", {"title": "Rust", "tokens": [[0.1, 0.0]]})
index.put_document("doc2", {"title": "Rust language", "tokens": [[0.9, 0.2], [0.0, 0.3]]})
index.commit()

results = index.search(
    "title:rust",
    rescore=laurus.LateInteractionRescore("tokens", [[1.0, 0.0], [0.0, 1.0]]),
)
```

To pass text instead of token vectors, register a `"candle_colbert"` embedder (`schema.add_embedder("colbert", {"type": "candle_colbert", "model": "colbert-ir/colbertv2.0"})`) and set `embedder="colbert"` on the multi-vector field; documents then give the field text, and the rescore query can be a string. See [API Reference → LateInteractionRescore](api_reference.md#lateinteractionrescore).

## 7. Update and delete

```python
# Update: put_document replaces all existing versions
index.put_document("doc1", {"title": "Updated Title", "body": "New content."})
index.commit()

# Append a new version without removing existing ones (RAG chunking pattern)
index.add_document("doc1", {"title": "Chunk 2", "body": "Additional chunk."})
index.commit()

# Retrieve all versions
docs = index.get_documents("doc1")

# Delete
index.delete_documents("doc1")
index.commit()
```

## 8. Schema management

```python
schema = laurus.Schema()
schema.add_text_field("title")
schema.add_text_field("body")
schema.add_integer_field("year")
schema.add_float_field("score")
schema.add_boolean_field("published")
schema.add_bytes_field("thumbnail")
schema.add_geo_field("location")
schema.add_datetime_field("created_at")
schema.add_hnsw_field("embedding", dimension=384)
schema.add_flat_field("small_vec", dimension=64)
schema.add_ivf_field("ivf_vec", dimension=128, n_clusters=100)
schema.add_multi_vector_field("tokens", dimension=128)
```

## 9. Index statistics

```python
stats = index.stats()
print(stats["document_count"])
print(stats["vector_fields"])
```
