# Lexical Search

Lexical search finds documents by matching keywords against an inverted index. Laurus provides a rich set of query types that cover exact matching, phrase matching, fuzzy matching, and more.

## Basic Usage

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::LexicalSearchQuery;

let request = SearchRequestBuilder::new()
    .lexical_query(
        LexicalSearchQuery::Obj(
            Box::new(TermQuery::new("body", "rust"))
        )
    )
    .limit(10)
    .build();

let results = engine.search(request).await?;
```

## Query Types

### TermQuery

Matches documents containing an exact term in a specific field.

```rust
use laurus::lexical::TermQuery;

// Find documents where "body" contains the term "rust"
let query = TermQuery::new("body", "rust");
```

> **Note:** Terms are matched after analysis. If the field uses `StandardAnalyzer`, both the indexed text and the query term are lowercased, so `TermQuery::new("body", "rust")` will match "Rust" in the original text.

### PhraseQuery

Matches documents containing an exact sequence of terms.

```rust
use laurus::lexical::query::phrase::PhraseQuery;

// Find documents containing the exact phrase "machine learning"
let query = PhraseQuery::new("body", vec!["machine".to_string(), "learning".to_string()]);

// Or use the convenience method from a phrase string:
let query = PhraseQuery::from_phrase("body", "machine learning");
```

Phrase queries require term positions to be stored (the default for `TextOption`). On a multi-valued text field (`multi_valued = true`) the elements share one position sequence separated by the field's `position_increment_gap` (default 100), so a phrase never spans two elements unless its slop reaches that gap — see [Schema and Fields](../schema_and_fields.md#multi-valued-fields).

### BooleanQuery

Combines multiple queries with boolean logic.

```rust
use laurus::lexical::query::boolean::{BooleanQuery, BooleanQueryBuilder, Occur};

let query = BooleanQueryBuilder::new()
    .must(Box::new(TermQuery::new("body", "rust")))       // AND
    .must(Box::new(TermQuery::new("body", "programming"))) // AND
    .must_not(Box::new(TermQuery::new("body", "python")))  // NOT
    .build();
```

| Occur | Meaning | DSL Equivalent |
| :--- | :--- | :--- |
| `Must` | Document MUST match | `+term` or `AND` |
| `Should` | Document SHOULD match (boosts score) | `term` or `OR` |
| `MustNot` | Document MUST NOT match | `-term` or `NOT` |
| `Filter` | MUST match, but does not affect score | (no DSL equivalent) |

### FuzzyQuery

Matches terms within a specified edit distance (Levenshtein distance).

```rust
use laurus::lexical::query::fuzzy::FuzzyQuery;

// Find documents matching "programing" within edit distance 2
// This will match "programming", "programing", etc.
let query = FuzzyQuery::new("body", "programing");  // default max_edits = 2
```

### WildcardQuery

Matches terms using wildcard patterns.

```rust
use laurus::lexical::query::wildcard::WildcardQuery;

// '?' matches exactly one character, '*' matches zero or more
let query = WildcardQuery::new("filename", "*.pdf")?;
let query = WildcardQuery::new("body", "pro*")?;
let query = WildcardQuery::new("body", "col?r")?;  // matches "color" and "colour"
```

### PrefixQuery

Matches documents containing terms that start with a specific prefix.

```rust
use laurus::lexical::query::prefix::PrefixQuery;

// Find documents where "body" contains terms starting with "pro"
// This matches "programming", "program", "production", etc.
let query = PrefixQuery::new("body", "pro");
```

### RegexpQuery

Matches documents containing terms that match a regular expression pattern.

```rust
use laurus::lexical::query::regexp::RegexpQuery;

// Find documents where "body" contains terms matching the regex
let query = RegexpQuery::new("body", "^pro.*ing$")?;

// Match version-like patterns
let query = RegexpQuery::new("version", r"^v\d+\.\d+")?;
```

> **Note:** `RegexpQuery::new()` returns `Result` because the regex pattern is validated at construction time. Invalid patterns will produce an error.

### NumericRangeQuery

Matches documents with numeric field values within a range.

```rust
use laurus::lexical::NumericRangeQuery;
use laurus::lexical::core::field::NumericType;

// Find documents where "price" is between 10.0 and 100.0 (inclusive)
let query = NumericRangeQuery::new(
    "price",
    NumericType::Float,
    Some(10.0),   // min
    Some(100.0),  // max
    true,         // include min
    true,         // include max
);

// Open-ended range: price >= 50
let query = NumericRangeQuery::new(
    "price",
    NumericType::Float,
    Some(50.0),
    None,     // no upper bound
    true,
    false,
);
```

Numeric range queries are **constant-scored**: every matching document
receives the same score (the query's boost, `1.0` by default), following
Lucene's `PointRangeQuery` semantics. Matching uses the field's BKD tree
when the segment has one; segments without a BKD tree for the field
(for example, segments none of whose documents carry the field, or a
field configured `indexed = false, stored = true`) fall back to scanning
only the stored documents actually present in that segment.

### DateTimeRangeQuery

Matches documents whose `DateTime` field value falls within a range.

```rust
use chrono::{TimeZone, Utc};
use laurus::lexical::DateTimeRangeQuery;

// Find documents where "created_at" is within 2024 (both bounds inclusive)
let query = DateTimeRangeQuery::between(
    "created_at",
    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
    Utc.with_ymd_and_hms(2024, 12, 31, 23, 59, 59).unwrap(),
);

// Open-ended range: created_at > 2024-06-15T12:00:00Z (exclusive)
let query = DateTimeRangeQuery::after(
    "created_at",
    Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap(),
);

// From textual bounds, using the same literal grammar as the query DSL
let query = DateTimeRangeQuery::from_literals(
    "created_at",
    Some("2024-01-01"),            // min (date only = midnight UTC)
    Some("2024-12-31T23:59:59Z"),  // max
    true,                          // include min
    true,                          // include max
)?;
```

`new(field, lower, upper, lower_inclusive, upper_inclusive)` takes
`Option<DateTime<Utc>>` bounds; `on_or_after`, `before` and `on_or_before`
complete the one-sided helpers, and `.with_boost()` sets the constant score.
The query is evaluated against the field's one-dimensional BKD point
(seconds with a microsecond fraction, so sub-second bounds compare exactly)
with the same constant scoring and stored-document fallback as
`NumericRangeQuery`. `from_literals` accepts RFC 3339, naive
`YYYY-MM-DDTHH:MM:SS[.fff]` (UTC) and `YYYY-MM-DD` (midnight UTC) — the
query the DSL builds for `created_at:[2024-01-01 TO 2024-12-31]` (see
[Query DSL](../query_dsl.md#range-query)). A `NumericRangeQuery` over a
`DateTime` field with epoch-second bounds keeps working.

### GeoDistanceQuery / GeoBoundingBoxQuery

Match documents by 2D geographic location (WGS84 latitude / longitude).

```rust
use laurus::lexical::query::geo::{GeoBoundingBoxQuery, GeoDistanceQuery};

// Find documents within 10 km (= 10 000 m) of Tokyo Station (35.6812, 139.7671)
let query = GeoDistanceQuery::within_radius("location", 35.6812, 139.7671, 10_000.0)?; // distance in metres

// Find documents within a bounding box (min_lat, min_lon, max_lat, max_lon)
let query = GeoBoundingBoxQuery::within_bounding_box(
    "location",
    35.0, 139.0,  // min (lat, lon)
    36.0, 140.0,  // max (lat, lon)
)?;
```

Both queries score by distance (closer documents rank higher: linear decay
from the circle's centre, or from the box's centre). Matching uses the
field's BKD tree when the segment has one, reading the coordinates
directly from the tree — so an index-only field
(`indexed = true, stored = false`) works without stored documents.
Segments without a BKD tree for the field (for example, segments none of
whose documents carry the field, or a field configured
`indexed = false, stored = true`) fall back to scanning only the stored
documents actually present in that segment.

### Geo3dDistanceQuery / Geo3dBoundingBoxQuery / Geo3dNearestQuery

Three queries target 3D `Geo3d` fields backed by ECEF Cartesian coordinates
(metres). Use them when altitude matters or when a 2D `Geo` field would
introduce pole singularities. See [3D Geographic Search](../geo3d.md) for
the coordinate system, WGS84 conversion helpers, and worked examples.

```rust
use laurus::GeoEcefPoint;
use laurus::lexical::query::geo3d::{
    Geo3dDistanceQuery, Geo3dBoundingBoxQuery, Geo3dNearestQuery,
};

let centre = GeoEcefPoint::new(-3_955_182.0, 3_350_553.0, 3_700_276.0);

// Sphere: docs within 5 km of `centre`
let q = Geo3dDistanceQuery::new("position", centre, 5_000.0);

// Axis-aligned 3D bounding box (constructor validates min ≤ max per axis)
let min = GeoEcefPoint::new(-4_000_000.0, 3_300_000.0, 3_650_000.0);
let max = GeoEcefPoint::new(-3_900_000.0, 3_400_000.0, 3_750_000.0);
let q = Geo3dBoundingBoxQuery::new("position", min, max)?;

// k-NN: 10 nearest neighbours, with a custom radius schedule
let q = Geo3dNearestQuery::new("position", centre, 10)
    .with_initial_radius(500.0)
    .with_max_radius(1_000_000.0);
```

| Query | Score |
| :--- | :--- |
| `Geo3dDistanceQuery` | `1 - distance / radius`, clamped to `[0, 1]`. |
| `Geo3dBoundingBoxQuery` | Constant `1.0` for every match. |
| `Geo3dNearestQuery` | Normalised so the closest hit is `1.0`, the farthest in the returned set is `0.0`. |

Geo3d queries require the field to be indexed (`indexed = true`, the
default): they run entirely on the field's BKD tree and return no hits
from segments that lack one — there is no stored-document fallback for
3D queries.

### SpanQuery

Matches terms based on their proximity within a document. Use `SpanTermQuery` and `SpanNearQuery` to build proximity queries:

```rust
use laurus::lexical::query::span::{SpanQuery, SpanTermQuery, SpanNearQuery};

// Find documents where "quick" appears near "fox" (within 3 positions)
let query = SpanNearQuery::new(
    "body",
    vec![
        Box::new(SpanTermQuery::new("body", "quick")) as Box<dyn SpanQuery>,
        Box::new(SpanTermQuery::new("body", "fox")) as Box<dyn SpanQuery>,
    ],
    3,    // slop (max distance between terms)
    true, // in_order (terms must appear in order)
);
```

## Scoring

Lexical search results are scored using **BM25**. The score reflects how relevant a document is to the query:

- Higher term frequency in the document increases the score
- Rarer terms across the index increase the score
- Shorter documents are boosted relative to longer ones

### Field Boosts

You can boost specific fields to influence relevance using the `SearchRequestBuilder`:

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::LexicalSearchQuery;

let request = SearchRequestBuilder::new()
    .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new("body", "rust"))))
    .add_field_boost("title", 2.0)  // title matches count double
    .add_field_boost("body", 1.0)
    .build();
```

## Lexical Search Options

Lexical search behavior is controlled via `LexicalSearchOptions` on the `SearchRequest`, or by using builder methods on `SearchRequestBuilder`:

| Option | Default | Description |
| :--- | :--- | :--- |
| `field_boosts` | empty | Per-field score multipliers |
| `min_score` | 0.0 | Minimum score threshold |
| `timeout_ms` | None | Search time budget in milliseconds (see note below) |
| `parallel` | false | Enable parallel search across segments |
| `sort_by` | `Score` | Sort by relevance score, or by a field (`asc` / `desc`) |
| `highlight` | None | Fields (and configuration) to highlight in each hit's `SearchResult::highlights`; see [Highlighting](../../laurus/highlighting.md) |

Field-sorted searches (`sort_by: Field { .. }`) always scan every candidate
document — there is no early termination for field sorts, unlike the
block-max-driven early termination available for score sorts. This
guarantees the returned hits are the true top-K by field value rather than
an early-terminated approximation, and it means `total_hits` reflects the
true number of matches rather than a scan-truncated count.

When `timeout_ms` is set, the time budget is enforced **cooperatively during**
the search: the scan loops (including each segment of a multi-segment fanout)
check the deadline periodically and abort as soon as it is exceeded, returning a
timeout error rather than running the query to completion first. The check is
batched (every few thousand scanned documents), so an unset `timeout_ms` and the
common in-budget case pay no measurable overhead.

### Builder Methods

`SearchRequestBuilder` provides convenience methods for lexical options:

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::{LexicalSearchQuery, SortField, SortOrder};

let request = SearchRequestBuilder::new()
    .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new("body", "rust"))))
    .lexical_min_score(0.5)
    .lexical_timeout_ms(5000)
    .lexical_parallel(true)
    .sort_by(SortField::Field { name: "date".to_string(), order: SortOrder::Desc })
    .add_field_boost("title", 2.0)
    .add_field_boost("body", 1.0)
    .limit(20)
    .build();
```

`.highlight(fields)` and `.highlight_config(config)` request highlighted fragments per hit — see [Highlighting](../../laurus/highlighting.md).

## Using the Query DSL

Instead of building queries programmatically, you can use the text-based Query DSL:

```rust
use laurus::lexical::QueryParser;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use std::sync::Arc;

let analyzer = Arc::new(StandardAnalyzer::default());
let parser = QueryParser::new(analyzer).with_default_field("body");

// Simple term
let query = parser.parse("rust")?;

// Boolean
let query = parser.parse("rust AND programming")?;

// Phrase
let query = parser.parse("\"machine learning\"")?;

// Field-specific
let query = parser.parse("title:rust AND body:programming")?;

// Fuzzy
let query = parser.parse("programing~2")?;

// Range
let query = parser.parse("year:[2020 TO 2024]")?;

// DateTime range (date-only bounds are midnight UTC)
let query = parser.parse("created_at:[2024-01-01 TO 2024-12-31]")?;
```

See [Query DSL](../query_dsl.md) for the complete syntax reference.

## Filter Result Cache

Filter clauses — tenancy, category, status flags, and similar — are frequently
reused across many requests. Re-evaluating them from scratch every time re-walks
the same posting lists. Laurus memoises the **set of document ids** a filter
matches so a repeated filter becomes a single lookup instead of a full posting
walk.

- **Snapshot-scoped, self-invalidating.** The cache lives on the reader, which
  is rebuilt on every `commit()` / `optimize()` / `refresh()`. Each reader is a
  point-in-time snapshot, so the cache needs no manual invalidation: after the
  index changes, the next search starts from a fresh, empty cache and always
  reflects committed data.
- **Score-independent.** A filter selects documents without affecting relevance,
  so the cached value is a plain doc-id set (a Roaring bitmap). It is used for
  the `filter_query` of a [hybrid / filtered search](hybrid_search.md) and feeds
  both the lexical and vector sides.
- **Reused inside boolean queries.** An `Occur::Filter` clause within a
  `BooleanQuery` (e.g. `must(user_query).filter(tenant_filter)`) also draws its
  matched set from the cache instead of re-walking postings — including the
  per-segment fan-out path on multi-segment indexes.
- **Safe by construction.** Only queries with a canonical key are cached. Term,
  phrase, prefix, wildcard, regexp, fuzzy, range, geo, and geo3d queries are
  cacheable, as are boolean queries composed entirely of cacheable clauses with
  at least one positive (Must / Should / Filter) clause. Boolean queries with no
  positive clause, span queries, and multi-field queries are evaluated fresh
  (never cached), so results are always correct.

The cache is enabled by default. Tune or disable it via the index config:

```rust
use laurus::lexical::store::config::LexicalIndexConfig;

let config = LexicalIndexConfig::builder()
    .query_filter_cache_capacity(4096) // entries per snapshot; 0 disables the cache
    .build();
```

## Parsed Query Cache

Searching with a DSL string (`SearchRequest::from_dsl`, or a `LexicalSearchQuery::Dsl`)
parses the string with the pest grammar and re-tokenises its terms with the analyzer on
every call. Autocomplete and popular-query workloads repeat the same strings, so Laurus
memoises `dsl string → parsed query`: a repeated DSL string is parsed once and then reused
(a cheap clone of the parsed query tree).

Like the filter cache, it is **snapshot-scoped**: the cache lives on the searcher, which is
rebuilt on every `commit()` / `optimize()` / `refresh()`. The analyzer and default fields are
fixed for that searcher, so the DSL string alone is the key; a schema/analyzer change yields a
fresh, empty cache. Enabled by default; tune or disable via the index config:

```rust
use laurus::lexical::store::config::LexicalIndexConfig;

let config = LexicalIndexConfig::builder()
    .parsed_query_cache_capacity(2048) // entries per snapshot; 0 disables the cache
    .build();
```

## Posting Cache

Evaluating a term reads its posting list from the segment's `.post` file and decodes it
(varint doc-ids, deletion filtering, skip table). Without caching, every query for the same
term repeats that read + decode — and on cloud/remote storage the read dominates. Each segment
reader keeps a small cache of decoded, deletion-filtered posting lists, so a repeated
`(field, term)` lookup within a snapshot reuses the decoded list. The per-term iterator
shares the cached list directly (a reference-count bump, not a copy), so evaluating a term
never duplicates its posting arrays — this removed the dominant allocation cost on the
multi-segment scoring path.

Because a segment is immutable for a reader snapshot, the cached list is always consistent with
its deletions; a commit builds new segment readers with empty caches. The cache is **byte-budget
bounded** (posting lists vary widely in size) — least-recently-used lists are evicted once the
budget is exceeded, and a single list larger than the whole budget is not cached. It is enabled
by default and shares the `max_cache_memory` budget; control it via the index config:

```rust
use laurus::lexical::store::config::LexicalIndexConfig;
use laurus::lexical::index::config::InvertedIndexConfig;

let mut inverted = InvertedIndexConfig::default();
inverted.enable_posting_cache = false;        // disable entirely
inverted.max_cache_memory = 256 * 1024 * 1024; // or resize the cache budget (bytes)
let config = LexicalIndexConfig::Inverted(inverted);
```

## Next Steps

- Semantic similarity search: [Vector Search](vector_search.md)
- Combine lexical + vector: [Hybrid Search](hybrid_search.md)
- Full DSL syntax reference: [Query DSL](../query_dsl.md)
