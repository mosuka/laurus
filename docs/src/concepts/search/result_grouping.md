# Result Grouping

When a long document is indexed as several chunks, the chunks of one
document often fill the top of the ranking and push other documents off the
page. Result grouping (Issue #1297) folds the hits that share a field value
into one group and keeps the best few hits of each group, like
Elasticsearch's `collapse` or Qdrant's `group_by`. It works the same way for
lexical, vector and hybrid searches.

## Basic Usage

Chunks added with `Engine::add_document` share their document's `_id`, so
grouping by `_id` returns each document once, represented by its best
chunk:

```rust
use laurus::{GroupBy, SearchRequestBuilder};

engine.add_document("manual", chapter_1).await?;
engine.add_document("manual", chapter_2).await?;
engine.add_document("faq", faq).await?;
engine.commit().await?;

let request = SearchRequestBuilder::new()
    .query_dsl("body:lifetimes")
    .group_by(GroupBy::new("_id").group_size(3))
    .limit(10)
    .build();
for result in engine.search(request).await? {
    // The group's best chunk, then up to two more of the same document.
    println!("{} {}", result.id, result.score);
    for hit in &result.inner_hits {
        println!("  {}", hit.score);
    }
}
```

Besides the Rust API, grouping is available over
[gRPC](../../laurus-server/grpc_api.md#groupbyparams) and the
[HTTP gateway](../../laurus-server/http_gateway.md#search-with-grouping), in
the [MCP `search` and `search_batch` tools](../../laurus-mcp/tools.md#group-by-example)
and in [`laurus search --group-by`](../../laurus-cli/commands.md#search).

## How It Works

```text
query ─→ first stage: lexical / vector / hybrid (as usual)
              ▼
         rescore (when requested)
              ▼
         group the ranked hits by the field's value
              ▼
         offset / limit over groups → documents
```

Grouping runs on the final ranking, after fusion and rescore and before the
page is cut:

1. Groups are ordered by the position of their best hit in the ranking.
2. Within a group, hits keep their ranking order, up to `group_size` hits.
3. The first hit of a group is the returned result; the others are its
   `inner_hits`.

Grouping follows positions in the ranking and never compares scores, so it
also follows a lexical `sort_by` and a rescore, whose rescored and
unrescored hits have scores of different kinds.

## Options

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `field` | `String` | — | The field whose value groups the hits (see below) |
| `group_size` | `usize` | `1` | Hits kept per group, `1..=100`: the returned result plus up to `group_size - 1` inner hits |

## Grouping Fields

A grouping field is `_id`, or a schema field that is:

- of type `Text`, `Integer`, `Float`, `Boolean` or `DateTime`;
- single-valued (`multi_valued: false`);
- `stored`, with `doc_values` enabled (both are on by default).

Values are compared as stored: a `Text` value is compared as the whole
original string, not as analyzed terms. `Float` values that compare equal
(`0.0` and `-0.0`) share a group, and so do NaN values. A document with no
value for the field forms a group of its own, with no `group_key`, so no hit
is dropped and unrelated documents are never merged.

The search fails with `LaurusError::InvalidArgument` before any search runs
when the field is unknown, of another type, multi-valued, not stored or
without `doc_values`, when `group_size` is outside `1..=100`, or when
`offset + limit` exceeds 10,000. A `limit` of 0 returns no results without
searching.

## Results

`limit` and `offset` count groups, not hits. Each returned `SearchResult` is
a group's best hit, with two fields filled:

| Field | Type | Description |
| :--- | :--- | :--- |
| `group_key` | `Option<DataValue>` | The value the group's hits share; `None` for a hit without a value |
| `inner_hits` | `Vec<SearchResult>` | The group's next best hits, in ranking order |

Each inner hit carries the same `group_key` and has no inner hits of its
own. Highlights are filled for inner hits too. An ungrouped search leaves
both fields empty.

## Candidate Pool

The first stage of a grouped search ranks `(offset + limit) × group_size × 2`
candidates, or the rescore window when that is larger. When those
candidates hold fewer than `offset + limit` groups, the engine doubles the
pool and runs the first stage again, until the pool holds enough groups, the
first stage has no more hits, or the pool reaches 10,000 candidates. Query
embedding runs once; a rescore scores each candidate once across rounds.
Doubling keeps the total work within about twice that of the last round.

The number of groups is complete within that bound. The number of inner hits
is not: a group's further hits are taken from the same pool, so a group can
hold fewer than `group_size` hits even when more of its documents match.

`lexical_timeout_ms` limits the lexical time of all rounds together: when
widening runs past it, the search fails as an ungrouped search would.

Group values are read from the field's DocValues column. The first grouped
search on a field loads that column of each segment it reaches into memory,
where it stays for later searches, as for sorting by the field. For
1,000,000 documents with 12-character `_id` values, the `_id` columns of
every segment took about 111 MB (about 111 bytes per document).

## Next Steps

- [Lexical Search](lexical_search.md)
- [Vector Search](vector_search.md)
- [Hybrid Search](hybrid_search.md)
