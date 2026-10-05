# laurus-python

[![PyPI](https://img.shields.io/pypi/v/laurus.svg)](https://pypi.org/project/laurus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python bindings for the [Laurus](https://github.com/mosuka/laurus) search engine. Provides lexical search, vector search, and hybrid search from Python via a native Rust extension built with [PyO3](https://github.com/PyO3/pyo3) and [Maturin](https://github.com/PyO3/maturin).

## Features

- **Lexical Search** -- Full-text search powered by an inverted index with BM25 scoring
- **Vector Search** -- Approximate nearest neighbor (ANN) search using Flat, HNSW, or IVF indexes
- **Hybrid Search** -- Combine lexical and vector results with fusion algorithms (RRF, WeightedSum)
- **Late-Interaction Rescore** -- Reorder the top results by ColBERT-style MaxSim over multi-vector (token vector) fields
- **Rich Query DSL** -- Term, Phrase, Fuzzy, Wildcard, NumericRange, Geo, Boolean, Span queries
- **Text Analysis** -- Tokenizers, filters, stemmers, and synonym expansion
- **Flexible Storage** -- In-memory (ephemeral) or file-based (persistent) indexes
- **Pythonic API** -- Clean, intuitive Python classes with full type information

## Installation

```bash
pip install laurus
```

To build from source (requires Rust toolchain):

```bash
pip install maturin
maturin develop
```

## Quick Start

```python
import laurus

# Create an in-memory index
index = laurus.Index()

# Index documents
index.put_document("doc1", {"title": "Introduction to Rust", "body": "Systems programming language."})
index.put_document("doc2", {"title": "Python for Data Science", "body": "Data analysis with Python."})
index.commit()

# Search with a DSL string
results = index.search("title:rust", limit=5)
for r in results:
    print(f"[{r.id}] score={r.score:.4f}  {r.document['title']}")

# Search with a query object
results = index.search(laurus.TermQuery("body", "python"), limit=5)
```

## Index Types

### In-memory (ephemeral)

```python
index = laurus.Index()
```

### File-based (persistent)

```python
schema = laurus.Schema()
schema.add_text_field("title")
schema.add_text_field("body")
schema.add_hnsw_field("embedding", dimension=384)

index = laurus.Index(path="./myindex", schema=schema)
```

This writes `./myindex/schema.toml` and `./myindex/store/` -- the same
layout `laurus-cli create index --schema` uses, so the directory can be
opened by either. Reopening it later only needs the path (`schema` must be
omitted, since it's loaded from the persisted `schema.toml`):

```python
index = laurus.Index(path="./myindex")
```

A file-based index holds an exclusive lock on its directory for as long as
the `Index` object is alive, so a second `laurus.Index(path=...)` on the
same path fails while the first is still open. Call `index.close()` when
you are done with an index -- especially before reopening the same path --
rather than relying on CPython's garbage collector, whose destruction
timing is not guaranteed for every reference pattern (e.g. reference
cycles). `close()` is idempotent; every other method raises after it has
been called.

```python
index.close()
```

## Reloading an index

`reload()` picks up changes committed by another process without paying the
cost of constructing a brand-new `Index` -- in particular, it reuses the
already-loaded embedding model(s) when the schema's embedding configuration
hasn't changed, instead of reloading them from scratch:

```python
changed = index.reload()  # True if the commit generation actually advanced
```

`reload()` works whether the index is currently open **or already
`close()`d** -- it reopens the same directory either way, so you can hold
onto one `Index` object across a full reload cycle instead of constructing a
new one and swapping references. It requires a file-backed index (`path` was
given at construction); calling it on an in-memory index raises `ValueError`.

`index.commit_generation()` returns the same O(1) counter as
`stats()["commit_generation"]`, without `stats()`'s cost of scanning vector
fields. It only reflects commits made through *this* `Index` object,
though -- it's a snapshot held in memory, not re-read from disk on every
call -- so it confirms whether `reload()` (or your own `commit()`) actually
advanced the state, but it cannot by itself tell you whether *another*
process has committed something new; only `reload()` picks that up.

To cheaply check for another process's changes *without* paying `reload()`'s
cost, use the module-level `laurus.peek_commit_generation(path)`. Unlike
`index.commit_generation()`, it isn't tied to any `Index` object -- it reads
`commit_generation.json` straight off disk, with no `Engine` construction at
all, so it works even before you've opened an `Index` for that path in this
process:

```python
before = laurus.peek_commit_generation(path)
# ... later ...
if laurus.peek_commit_generation(path) != before:
    index.reload()
```

Raises `ValueError` if `path` isn't a laurus index directory (no persisted
schema).

## Durability / WAL

A persistent index writes every change to a write-ahead log (WAL). By default
the WAL is `fsync`-ed on every record, so each write is fully durable. Opt into
group commit to batch `fsync` for higher write throughput (a crash can lose up
to the last unsynced batch, like SQLite's `synchronous = NORMAL`):

```python
import laurus

policy = laurus.WalSyncPolicy.group(max_records=4096, max_interval_ms=1000)
index = laurus.Index(path="./myindex", schema=schema, wal_sync_policy=policy)

index.put_document("doc1", {"title": "Hello"})
index.flush_wal()  # force a durable barrier on demand
index.commit()     # also flushes the WAL
```

Omit `wal_sync_policy` (or pass `laurus.WalSyncPolicy.per_record()`) to keep
the default per-record durability.

## Query Types

| Query class | Description |
| :--- | :--- |
| `TermQuery(field, term)` | Exact term match |
| `PhraseQuery(field, [terms])` | Ordered phrase match |
| `FuzzyQuery(field, term, max_edits)` | Approximate term match |
| `WildcardQuery(field, pattern)` | Wildcard pattern match (`*`, `?`) |
| `NumericRangeQuery(field, min, max)` | Numeric range (int or float) |
| `GeoDistanceQuery.within_radius(field, lat, lon, distance_m)` | Geo-distance radius search |
| `GeoBoundingBoxQuery.within_bounding_box(field, min_lat, min_lon, max_lat, max_lon)` | Geo bounding-box search |
| `Geo3dDistanceQuery.within_sphere(field, x, y, z, distance_m)` | 3D ECEF sphere search |
| `Geo3dBoundingBoxQuery.within_box(field, min_x, min_y, min_z, max_x, max_y, max_z)` | 3D ECEF AABB search |
| `Geo3dNearestQuery.k_nearest(field, x, y, z, k)` | 3D ECEF k-nearest neighbours |
| `BooleanQuery(must, should, must_not)` | Compound boolean logic |
| `SpanNearQuery(field, [terms], slop)` | Proximity / ordered span match |
| `VectorQuery(field, vector)` | Pre-computed vector similarity |
| `VectorTextQuery(field, text)` | Text-to-vector similarity (requires embedder) |

## Hybrid Search

```python
request = laurus.SearchRequest(
    lexical_query=laurus.TermQuery("body", "rust"),
    vector_query=laurus.VectorQuery("embedding", query_vec),
    fusion=laurus.RRF(k=60.0),
    limit=10,
)
results = index.search(request)
```

### Fusion algorithms

| Class | Description |
| :--- | :--- |
| `RRF(k=60.0)` | Reciprocal Rank Fusion (rank-based, default for hybrid) |
| `WeightedSum(lexical_weight=0.5, vector_weight=0.5)` | Score-normalised weighted sum |

## Late-Interaction Rescore

A multi-vector field holds each document's token vectors (e.g. ColBERT embeddings). `LateInteractionRescore` reorders the top `window_size` results (default 100) of any search by MaxSim against it:

```python
schema.add_multi_vector_field("tokens", dimension=2, distance="dot_product")
# ...
index.put_document("doc1", {"title": "Rust", "tokens": [[0.9, 0.2], [0.0, 0.3]]})
# ...
rescore = laurus.LateInteractionRescore("tokens", [[1.0, 0.0], [0.0, 1.0]])
results = index.search("title:rust", rescore=rescore)
```

With a `"candle_colbert"` embedder on the field, documents and the rescore query can be text instead.

## Text Analysis

```python
syn_dict = laurus.SynonymDictionary()
syn_dict.add_synonym_group(["ml", "machine learning"])

tokenizer = laurus.WhitespaceTokenizer()
filt = laurus.SynonymGraphFilter(syn_dict, keep_original=True, boost=0.8)

tokens = tokenizer.tokenize("ml tutorial")
tokens = filt.apply(tokens)
for tok in tokens:
    print(tok.text, tok.position, tok.boost)
```

## Examples

Usage examples are in the [`examples/`](examples/) directory:

| Example | Description |
| :--- | :--- |
| [quickstart.py](examples/quickstart.py) | Basic indexing and full-text search |
| [lexical_search.py](examples/lexical_search.py) | All query types (Term, Phrase, Boolean, Fuzzy, Wildcard, Range, Geo, Span) |
| [vector_search.py](examples/vector_search.py) | Semantic similarity search with embeddings |
| [hybrid_search.py](examples/hybrid_search.py) | Combining lexical and vector search with fusion |
| [synonym_graph_filter.py](examples/synonym_graph_filter.py) | Synonym expansion in the analysis pipeline |
| [search_with_openai.py](examples/search_with_openai.py) | Cloud-based embeddings via OpenAI |
| [multimodal_search.py](examples/multimodal_search.py) | Text-to-image and image-to-image search |

## Documentation

- [Python Binding Guide](https://mosuka.github.io/laurus/laurus-python.html)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
