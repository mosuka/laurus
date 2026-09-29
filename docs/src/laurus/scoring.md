# Scoring & Ranking

Laurus uses BM25 for lexical search, distance-based similarity for vector search, and a configurable fusion algorithm to combine the two for hybrid search. This page describes each scoring path and how to influence it from the public API.

## Lexical Scoring

### BM25 (Default)

BM25 is the lexical scoring function. It balances term frequency with document length normalization:

```text
score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
```

Where:

- **tf** — term frequency in the document.
- **IDF** — inverse document frequency (rarity of the term across all documents).
- **k1** — term-frequency saturation parameter. Laurus uses **1.2**.
- **b** — document-length normalization factor. Laurus uses **0.75**.
- **doc_len / avg_doc_len** — ratio of document length to average document length. `doc_len` counts field **positions**, not tokens: a synonym the analyzer stacks on another token (`position_increment = 0`) shares its position and does not add to the length, matching Lucene's `discountOverlaps`.

The `(k1, b)` parameters are fixed at the implementation defaults today. The values match Lucene / Elasticsearch defaults, so BM25 scores from Laurus are directly comparable to those engines for tuning intuition.

A quoted or bare value whose analyzer stacks several alternatives at one position (synonyms) is scored as one blended term rather than as a sum of independent term scores, so a document is not rewarded for holding every alternative — see [Searching with synonyms](../concepts/analysis.md#searching-with-synonyms).

### Field Boosts

Per-field score multipliers are configured on the search request, not in a separate scoring struct:

```rust
use laurus::SearchRequestBuilder;

let request = SearchRequestBuilder::new()
    .query_dsl("rust programming")
    .add_field_boost("title", 2.0) // title matches score 2x
    .add_field_boost("body", 1.0)  // body matches score 1x (the default)
    .limit(10)
    .build();
```

The boost is multiplied into the BM25 score contribution of matches in that field. A boost of `1.0` is a no-op; boosts apply only to fields named in the query (or in the schema's default-search fields).

Over gRPC and HTTP, the same setting is exposed as `SearchRequest.field_boosts` (`map<string, float>`). See [gRPC API → SearchRequest](../laurus-server/grpc_api.md#searchrequest-fields).

## Vector Scoring

Vector search ranks results by distance-based similarity. The distance metric is configured per field on the vector index (HNSW / Flat / IVF):

| Metric | Description | Best for |
| :--- | :--- | :--- |
| `Cosine` | 1 − cosine similarity (default) | Normalised text embeddings |
| `Euclidean` | L2 distance | Spatial / pre-normalised data |
| `Manhattan` | L1 distance | Sparse feature vectors |
| `DotProduct` | Negated dot product | Pre-normalised vectors where higher = better |
| `Angular` | Angular distance | Directional similarity |

Distances are converted to similarity scores so that "higher is better" holds across both lexical and vector results, which is what the fusion algorithms below assume.

## Hybrid Search Fusion

When a search request contains both lexical and vector clauses, the two result lists need to be merged. Laurus exposes two fusion algorithms via [`FusionAlgorithm`](api_reference.md#fusionalgorithm).

### RRF (Reciprocal Rank Fusion)

RRF avoids score normalisation entirely by combining **ranks** instead of raw scores:

```text
rrf_score(doc) = Σ 1 / (k + rank_i(doc))
```

The sum runs over each result list the document appears in. The `k` parameter (default **60.0**) smooths the distribution — higher `k` flattens the contribution of top-ranked results.

```rust
use laurus::{FusionAlgorithm, SearchRequestBuilder};

let request = SearchRequestBuilder::new()
    .query_dsl("title:rust ~\"systems programming\"")
    .fusion_algorithm(FusionAlgorithm::Rrf { k: 60.0 })
    .build();
```

### WeightedSum

`WeightedSum` first min-max normalises each list of scores independently, then takes a weighted linear combination:

```text
norm(score)  = (score - min) / (max - min)
final(doc)   = lexical_weight * norm(lexical_score(doc))
             + vector_weight  * norm(vector_score(doc))
```

```rust
use laurus::{FusionAlgorithm, SearchRequestBuilder};

let request = SearchRequestBuilder::new()
    .query_dsl("title:rust ~\"systems programming\"")
    .fusion_algorithm(FusionAlgorithm::WeightedSum {
        lexical_weight: 0.6,
        vector_weight: 0.4,
    })
    .build();
```

Both weights are clamped to `[0.0, 1.0]`. Use RRF when you do not have a calibrated reason to pick specific weights — it is parameter-light and robust to scale differences between lists.

## See Also

- [API Reference → `FusionAlgorithm`](api_reference.md#fusionalgorithm) — variant signatures
- [Hybrid Search](../concepts/search/hybrid_search.md) — when to pick which fusion algorithm
- [Vector Search](../concepts/search/vector_search.md) — distance metric trade-offs
