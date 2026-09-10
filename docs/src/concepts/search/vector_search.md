# Vector Search

Vector search finds documents by semantic similarity. Instead of matching keywords, it compares the meaning of the query against document embeddings in vector space.

## Basic Usage

### Builder API

```rust
use laurus::SearchRequestBuilder;
use laurus::vector::search::searcher::VectorSearchQuery;
use laurus::vector::store::request::QueryPayload;
use laurus::data::DataValue;

let request = SearchRequestBuilder::new()
    .vector_query(
        VectorSearchQuery::Payloads(vec![
            QueryPayload {
                field: "embedding".to_string(),
                payload: DataValue::Text("systems programming language".to_string()),
                weight: 1.0,
            },
        ])
    )
    .limit(10)
    .build();

let results = engine.search(request).await?;
```

The `QueryPayload` stores raw data (text, bytes, etc.) that will be embedded at search time using the configured embedder.

### Query DSL

```rust
use laurus::vector::VectorQueryParser;

let parser = VectorQueryParser::new(embedder.clone())
    .with_default_field("embedding");

let request = parser.parse(r#"embedding:"systems programming""#).await?;
```

## VectorSearchQuery

The vector search query is specified as a `VectorSearchQuery` enum:

| Variant | Description |
| :--- | :--- |
| `Payloads(Vec<QueryPayload>)` | Raw payloads (text, bytes, etc.) to be embedded at search time |
| `Vectors(Vec<QueryVector>)` | Pre-embedded query vectors ready for nearest-neighbor search |

### QueryPayload

| Field | Type | Description |
| :--- | :--- | :--- |
| `field` | `String` | Target vector field name |
| `payload` | `DataValue` | The payload to embed (e.g., `DataValue::Text(...)`) |
| `weight` | `f32` | Score weight (default: 1.0) |

### QueryVector

| Field | Type | Description |
| :--- | :--- | :--- |
| `vector` | `Vector` | Pre-computed dense vector embedding |
| `weight` | `f32` | Score weight (default: 1.0) |
| `fields` | `Option<Vec<String>>` | Optional field restriction |

### Examples

```rust
use laurus::vector::search::searcher::VectorSearchQuery;
use laurus::vector::store::request::{QueryPayload, QueryVector};
use laurus::vector::core::vector::Vector;
use laurus::data::DataValue;

// Text query (will be embedded at search time)
let query = VectorSearchQuery::Payloads(vec![
    QueryPayload {
        field: "text_vec".to_string(),
        payload: DataValue::Text("machine learning".to_string()),
        weight: 1.0,
    },
]);

// Pre-computed vector
let query = VectorSearchQuery::Vectors(vec![
    QueryVector {
        vector: Vector::from(vec![0.1, 0.2, 0.3]),
        weight: 1.0,
        fields: Some(vec!["embedding".to_string()]),
    },
]);
```

## Multi-Field Vector Search

You can search across multiple vector fields in a single request:

```rust
use laurus::vector::search::searcher::VectorSearchQuery;
use laurus::vector::store::request::QueryPayload;
use laurus::data::DataValue;

let query = VectorSearchQuery::Payloads(vec![
    QueryPayload {
        field: "text_vec".to_string(),
        payload: DataValue::Text("cute kitten".to_string()),
        weight: 1.0,
    },
    QueryPayload {
        field: "image_vec".to_string(),
        payload: DataValue::Text("fluffy cat".to_string()),
        weight: 1.0,
    },
]);
```

Each clause produces a vector that is searched against its respective field. Results are combined using the configured score mode.

### Score Modes

| Mode | Description |
| :--- | :--- |
| `WeightedSum` (default) | Sum of (similarity * weight) across all clauses |
| `MaxSim` | Maximum similarity score across clauses |
| `LateInteraction` | ColBERT-style late interaction scoring |

### Parallel Multi-Vector Execution

When a request carries multiple query vectors (e.g., ColBERT-style late
interaction, multi-vector MaxSim, ensemble rerankers), Laurus dispatches the
per-query similarity searches in parallel via [rayon][rayon-crate].

Behaviour:

- The parallelisation lives in
  `VectorIndexSearcher::search_batch`'s default implementation. On native
  builds (default `native` feature), once `queries.len()` reaches the
  searcher's `parallel_threshold` (default `4`, overridable per searcher),
  the per-query HNSW / Flat / IVF searches run on rayon's global thread
  pool. Below that threshold the serial loop wins because rayon's dispatch
  overhead (~1-2 µs) would otherwise dominate a single 50-200 µs query.
- On `wasm32` targets the serial path is always used because rayon is
  unavailable.
- Aggregation (the score-mode merge) and the final sort run serially after
  the per-query phase. Score ties are broken by ascending `doc_id` so the
  results are deterministic regardless of rayon's work-stealing schedule.

The external API surface (`VectorStore::search`, gRPC `Search`, REST
`POST /v1/search`, all language bindings) is unchanged; parallel execution
is purely an internal optimisation enabled by upgrading laurus. Speedups
scale with the host's available cores — on a 4-core / 8-thread laptop CPU
the parallel path reaches roughly 2× throughput at `B = 64` query vectors,
limited by physical core count and HyperThreading sharing.

[rayon-crate]: https://docs.rs/rayon

### Parallel Brute-Force Scan

Flat and IVF indexes rank candidates with an exhaustive distance scan rather
than a graph walk. When one query's candidate count reaches an internal
threshold (`2048`), that scan is dispatched across [rayon][rayon-crate]'s
global thread pool; below it the serial loop wins because rayon's per-job
dispatch (~1-2 µs) would dominate a small scan.

This is orthogonal to the per-query parallelism above: a batch parallelises
across queries, and each large query further parallelises its own scan on the
same pool, with work-stealing bounding total parallelism to the pool size (no
OS-thread oversubscription). The distance kernel has no side effects, so
results are collected in arbitrary order and then sorted, keeping output
deterministic. On `wasm32` (no rayon) the scan is always serial.

Speedup scales with the host's physical cores and is largest for big Flat
indexes or a wide IVF `n_probe`; scans below the threshold are unaffected.

### IVF Cluster Selection

Before the distance scan, an IVF query first chooses *which* clusters to scan:
it scores the query against every centroid and keeps the `n_probe` nearest.
Because the probed clusters are then merged and re-ranked by similarity, the
centroids' relative order does not matter, so the nearest `n_probe` are taken
with an `O(K)` partial selection (`select_nth_unstable_by`) over the `K`
centroids rather than a full `O(K log K)` sort. The saving grows with the
cluster count `K`; at `K = 2048` it cut the per-query coarse step by roughly
18% (Issue #668). The centroid scan itself stays serial — each centroid is a
single distance computation, so dispatching it to rayon costs more than it
saves at realistic cluster counts (`K ≈ √N`).

### Weights

Use the `^` boost syntax in DSL or `weight` in `QueryVector` to adjust how much each field contributes:

```text
text_vec:"cute kitten"^1.0 image_vec:"fluffy cat"^0.5
```

This means text similarity counts twice as much as image similarity.

A vector field's schema-level `base_weight` (Issue #1084) works the same
way but is a per-field default rather than a per-query override: it is
multiplied into the query weight above, so `base_weight` and `^`/`weight`
compose. Because both only affect the vector side's own scoring, neither
can rebalance a hybrid search's lexical-vs-vector split —
`FusionAlgorithm::RRF` is rank-only and ignores the underlying score
entirely, and `FusionAlgorithm::WeightedSum` min-max normalizes each side
before applying `lexical_weight`/`vector_weight`, which cancels out a
uniform per-field scalar. What both weighting mechanisms DO affect is the
relative priority *among* vector fields when a query targets two or more
at once.

### Field Routing

In a multi-field schema, each vector field has its own HNSW graph. By
default a query searches every vector field; restricting it to named
fields skips the others' graph work entirely.

Two routing inputs are honored, in priority order:

1. **Per-query** — `QueryVector.fields`. The DSL parser sets this from the
   field a clause names, so `image_vec:"fluffy cat"` only searches
   `image_vec`.
2. **Request-level** — `VectorSearchParams.fields`, a list of selectors:
   - `Exact("image_vec")` — match a field by exact name.
   - `Prefix("image_")` — match every field whose name starts with the
     prefix (resolved against the index's field names).

When neither is set, all fields are searched (the default). A query routed
to a field never returns documents that lack a vector in that field.

You can apply lexical filters to narrow the vector search results:

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::vector::search::searcher::VectorSearchQuery;
use laurus::vector::store::request::QueryPayload;
use laurus::data::DataValue;

// Vector search with a category filter
let request = SearchRequestBuilder::new()
    .vector_query(
        VectorSearchQuery::Payloads(vec![
            QueryPayload {
                field: "embedding".to_string(),
                payload: DataValue::Text("machine learning".to_string()),
                weight: 1.0,
            },
        ])
    )
    .filter_query(Box::new(TermQuery::new("category", "tutorial")))
    .limit(10)
    .build();

let results = engine.search(request).await?;
```

The filter query runs first on the lexical index to identify allowed document IDs, then the vector search is restricted to those IDs.

#### Filter-Aware HNSW Traversal

For HNSW fields the allowed-ID set is pushed into the graph search itself, not
just applied afterwards. During traversal the frontier still expands through
*every* neighbour — including non-matching ones — so the search can cross
non-matching regions of the graph, but only matching documents enter the
result set.

This matters for selective filters. A plain post-filter inspects the fixed
`ef_search` window of nearest neighbours and keeps whichever happen to match;
when matches are rare they can fall entirely outside that window, so the query
returns far fewer hits than exist — sometimes none, even though matching
documents are reachable. Filter-aware traversal keeps searching until it has
collected enough matches or hits an internal visit cap (a multiple of
`ef_search`) that bounds latency for very selective filters.

The unfiltered path is unchanged: with no filter the traversal behaves exactly
as before. Flat and IVF fields honour the allow-set *inline* — a candidate
whose document ID is not in the set is skipped before the distance kernel runs,
so a selective filter avoids the wasted distance computations a post-filter
would incur. Their scan is exhaustive either way, so recall is unchanged; the
store's post-filter still runs afterwards as a redundant safety net.

When the allow-set is smaller than `ef_search`, the HNSW field skips the graph
walk entirely and scores the allowed documents directly. With so few candidates
there is nothing for the graph to *find*; the direct scan touches exactly the
allowed documents — never more than the walk would — and is **exact**, so a very
selective filter returns the true nearest matches rather than an approximation.
Larger allow-sets keep using the filter-aware traversal above.

#### Deletion-Aware HNSW Traversal

Logically deleted documents stay in the HNSW graph until compaction (see
[Deletions & Compaction](../../laurus/deletions.md)), so the walk must skip them
the same way it skips filtered-out documents. The graph traversal applies a
single admission rule: a node enters the result set only if it matches the
filter (when one is present) **and** is not deleted, while the frontier still
expands through deleted nodes to preserve connectivity.

This is what keeps recall correct as deletions accumulate. If deleted nodes were
allowed into the result heap, they would consume the fixed `ef_search` slots and
push out live neighbours — in the worst case an `ef_search` window made entirely
of deleted documents would return nothing. Skipping them during the walk means
the same slots fill with live results instead, so a 10-hit page stays full even
after the nearest documents are deleted. The exact tiny-allow-set scan above and
the Flat/IVF inline paths apply the same deletion check.

The fast path is preserved: when a search has neither a filter nor any
deletions, the traversal runs the original loop unchanged, paying nothing for
the per-neighbour admission bookkeeping.

#### Allow-Set Representation

The allow-set is a typed structure chosen by shape: a Roaring bitmap for dense
filters and a hash set for sparse ones. For a filtered hybrid search the lexical
side already produced the matching document set as a bitmap (see the lexical
[Filter Result Cache](lexical_search.md#filter-result-cache)); that bitmap is
handed to the vector side **as-is**, so the set is materialised once for the
whole query instead of being rebuilt for each side. This is an internal
optimisation — the public filter API is unchanged.

### Filter with Numeric Range

```rust
use laurus::lexical::NumericRangeQuery;
use laurus::lexical::core::field::NumericType;

let request = SearchRequestBuilder::new()
    .vector_query(
        VectorSearchQuery::Payloads(vec![
            QueryPayload {
                field: "embedding".to_string(),
                payload: DataValue::Text("type systems".to_string()),
                weight: 1.0,
            },
        ])
    )
    .filter_query(Box::new(NumericRangeQuery::new(
        "year", NumericType::Integer,
        Some(2020.0), Some(2024.0), true, true
    )))
    .limit(10)
    .build();
```

## Distance Metrics

The distance metric is configured per field in the schema (see [Vector Indexing](../indexing/vector_indexing.md)):

| Metric | Description | Lower = More Similar |
| :--- | :--- | :--- |
| **Cosine** | 1 - cosine similarity | Yes |
| **Euclidean** | L2 distance | Yes |
| **Manhattan** | L1 distance | Yes |
| **DotProduct** | Negative inner product | Yes |
| **Angular** | Angular distance | Yes |

## Code Example: Complete Vector Search

```rust
use std::sync::Arc;
use laurus::{Document, Engine, Schema, SearchRequestBuilder, PerFieldEmbedder};
use laurus::lexical::TextOption;
use laurus::vector::HnswOption;
use laurus::vector::search::searcher::VectorSearchQuery;
use laurus::vector::store::request::QueryPayload;
use laurus::data::DataValue;
use laurus::storage::memory::MemoryStorage;

#[tokio::main]
async fn main() -> laurus::Result<()> {
    let storage = Arc::new(MemoryStorage::new(Default::default()));

    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_hnsw_field("text_vec", HnswOption {
            dimension: 384,
            ..Default::default()
        })
        .build();

    // Set up per-field embedder
    let embedder = Arc::new(my_embedder);
    let pfe = PerFieldEmbedder::new(embedder.clone());
    pfe.add_embedder("text_vec", embedder.clone());

    let engine = Engine::builder(storage, schema)
        .embedder(Arc::new(pfe))
        .build()
        .await?;

    // Index documents (text in vector field is auto-embedded)
    engine.add_document("doc-1", Document::builder()
        .add_text("title", "Rust Programming")
        .add_text("text_vec", "Rust is a systems programming language.")
        .build()
    ).await?;
    engine.commit().await?;

    // Search by semantic similarity
    let results = engine.search(
        SearchRequestBuilder::new()
            .vector_query(
                VectorSearchQuery::Payloads(vec![
                    QueryPayload {
                        field: "text_vec".to_string(),
                        payload: DataValue::Text("systems language".to_string()),
                        weight: 1.0,
                    },
                ])
            )
            .limit(5)
            .build()
    ).await?;

    for r in &results {
        println!("{}: score={:.4}", r.id, r.score);
    }

    Ok(())
}
```

## Next Steps

- Combine with keyword search: [Hybrid Search](hybrid_search.md)
- DSL syntax for vector queries: [Query DSL](../query_dsl.md)
