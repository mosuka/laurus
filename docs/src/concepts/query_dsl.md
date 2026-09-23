# Query DSL

Laurus provides a unified query DSL (Domain Specific Language) that allows lexical (keyword) and vector (semantic) search in a single query string. The `UnifiedQueryParser` splits the input into lexical and vector portions and delegates to the appropriate sub-parser.

## Overview

```text
title:hello AND content:"cute kitten"^0.8
|--- lexical --|    |--- vector --------|
```

The field type in the schema determines whether a clause is lexical or vector. If the field is a vector field (e.g., HNSW), the clause is treated as a vector query. Everything else is treated as a lexical query.

### Field validation

Every `field:value` clause is validated against the schema at parse time. A
query that references a field **not declared** in the schema is rejected with
an error rather than returning silently-empty results. This catches typos
early (e.g. `titl:hello` instead of `title:hello`).

If you want the engine to accept documents with previously-unknown fields,
set the schema's [`dynamic_field_policy`](./schema_and_fields.md#dynamic-schema)
so that the field gets added during ingestion. Once a field is part of the
schema, queries referencing it succeed.

## Lexical Query Syntax

Lexical queries search the inverted index using exact or approximate keyword matching.

### Term Query

Match a single term against a field (or the default field):

```text
hello
title:hello
```

If the field's analyzer splits a bare term into multiple tokens — for
example, a Japanese sentence under a morphological (Lindera) analyzer, or
a hyphenated word like `rust-lang` under the default `\w+` tokenizer —
the resulting tokens are combined with `OR` (matching Lucene's `match`
query), not required to appear as an exact phrase. Use a quoted
[Phrase Query](#phrase-query) when exact adjacency is required.

### Boolean Operators

Combine clauses with `AND` and `OR` (case-insensitive):

```text
title:hello AND body:world
title:hello OR title:goodbye
```

`AND` is symmetric — it makes the clauses on **both** sides required (`Must`). For example, `title:hello AND body:world` returns only documents that match **both** clauses. The same is true for chains: `a AND b AND c` requires all three. A clause that was already explicitly marked with `+` (required) or `-` (prohibited) keeps that intent — `AND` does not override an explicit prefix.

Space-separated clauses without an explicit operator use implicit boolean (behaves like OR with scoring), so `a b AND c` reads as "optionally match `a`, and require both `b` and `c`".

### Required / Prohibited Clauses

Use `+` (must match) and `-` (must not match):

```text
+title:hello -title:goodbye
```

### Phrase Query

Match an exact phrase using double quotes. Optional proximity (`~N`) allows N words between terms:

```text
"hello world"
"hello world"~2
```

### Fuzzy Query

Approximate matching with edit distance. Append `~` and optionally the maximum edit distance:

```text
roam~
roam~2
```

### Wildcard Query

Use `?` (single character) and `*` (zero or more characters):

```text
te?t
test*
```

### Range Query

Inclusive `[]` or exclusive `{}` ranges, useful for numeric and date fields:

```text
price:[100 TO 500]
date:{2024-01-01 TO 2024-12-31}
price:[* TO 100]
```

#### Date and datetime bounds

A range whose bounds are datetime literals builds a `DateTimeRangeQuery`,
evaluated against the field's BKD tree with constant scoring like a numeric
range. A bound that is not a bare number is tried as, in order:

| Form | Example | Interpretation |
| :--- | :--- | :--- |
| RFC 3339 | `2024-01-01T00:00:00Z`, `2024-01-01T09:00:00.5+09:00` | Fractional seconds allowed; an offset is normalized to UTC |
| Naive datetime | `2024-01-01T09:00:00`, `2024-01-01T09:00:00.5` | `YYYY-MM-DDTHH:MM:SS[.fff]`, interpreted as UTC |
| Date only | `2024-01-01` | `YYYY-MM-DD`, meaning `00:00:00` UTC of that day (the Lucene / Elasticsearch default) |

```text
# Date-only bounds (upper bound is 2024-12-31T00:00:00Z, see below)
created_at:[2024-01-01 TO 2024-12-31]

# RFC 3339 with an offset (09:00+09:00 is 00:00Z)
created_at:[2024-06-15T09:00:00+09:00 TO 2024-06-16T09:00:00+09:00]

# Naive datetime is UTC; * leaves the upper side open
created_at:[2024-06-15T12:34:56 TO *]

# Exclusive on both sides
created_at:{2024-01-01 TO 2025-01-01}
```

- A date-only bound is midnight UTC, so `[2024-01-01 TO 2024-12-31]` includes `2024-12-31T00:00:00Z` and nothing later that day. To cover the whole year, give the upper bound a time (`[2024-01-01 TO 2024-12-31T23:59:59.999999]`) or end an exclusive range on the next day (`{2024-01-01 TO 2025-01-01}` — note that `{}` also excludes the exact instant `2024-01-01T00:00:00Z`).
- `*` leaves a side open.
- Bare numbers are still numeric bounds: on a `DateTime` field they are epoch seconds and build a `NumericRangeQuery`.
- Mixing a numeric bound with a datetime bound (`[1704067200 TO 2024-12-31]`), or a datetime bound with a string that is not a datetime literal (`[2024-01-01 TO tomorrow]`), is a parse error rather than a silent empty result. A range whose bounds are neither numbers nor datetime literals (`title:[A TO Z]`) still parses but matches nothing, as before.

> Document ingestion coerces text into a `DateTime` field only from RFC 3339;
> the naive and date-only forms are accepted for query literals only.

### 2D Geographic Queries (`geo_*`)

Two function-style forms target `Geo` (2D latitude / longitude) fields. All
arguments are signed floats; latitudes / longitudes are in degrees and the
distance is in metres:

```text
location:geo_distance(lat, lon, distance_m)
location:geo_bbox(min_lat, min_lon, max_lat, max_lon)
```

| Form | Behaviour |
| :--- | :--- |
| `geo_distance(lat, lon, distance_m)` | All docs whose stored `(lat, lon)` lies within `distance_m` metres of the given centre. |
| `geo_bbox(min_lat, min_lon, max_lat, max_lon)` | All docs whose stored `(lat, lon)` lies inside the axis-aligned latitude / longitude rectangle. |

Examples:

```text
# Within 10 km (= 10 000 m) of Tokyo (35.6895, 139.6917)
location:geo_distance(35.6895, 139.6917, 10000)

# Inside an axis-aligned lat/lon bounding box
location:geo_bbox(35.0, 139.0, 36.0, 140.0)
```

> The query field must be declared as a `Geo` field in the schema. Latitudes
> must lie in `[-90, 90]` and longitudes in `[-180, 180]`; the parser rejects
> out-of-range values.

### 3D Geographic Queries (`geo3d_*`)

Three function-style forms target `Geo3d` (3D ECEF Cartesian) fields. All
arguments are signed floats in metres, except `k` (an unsigned integer):

```text
position:geo3d_distance(x, y, z, distance_m)
position:geo3d_bbox(min_x, min_y, min_z, max_x, max_y, max_z)
position:geo3d_nearest(x, y, z, k)
```

| Form | Behaviour |
| :--- | :--- |
| `geo3d_distance(x, y, z, distance_m)` | All docs whose stored point lies within `distance_m` metres of `(x, y, z)`. |
| `geo3d_bbox(min_x, min_y, min_z, max_x, max_y, max_z)` | All docs whose stored point lies inside the axis-aligned 3D box. |
| `geo3d_nearest(x, y, z, k)` | The `k` nearest docs to `(x, y, z)` by Euclidean distance. |

Examples:

```text
# Within 5 km of Tokyo Tower (ECEF coordinates)
position:geo3d_distance(-3955182, 3350553, 3700276, 5000)

# Inside an axis-aligned ECEF bounding box
position:geo3d_bbox(-4000000, 3300000, 3650000, -3900000, 3400000, 3750000)

# 10 nearest indexed points
position:geo3d_nearest(-3955182, 3350553, 3700276, 10)
```

> The query field must be declared as a `Geo3d` field in the schema. See
> [3D Geographic Search](geo3d.md) for the underlying coordinate system,
> WGS84 conversion helpers, and detailed semantics.

### Boost

Increase the weight of a clause with `^`:

```text
title:hello^2
"important phrase"^1.5
```

### Grouping

Use parentheses for sub-expressions:

```text
(title:hello OR title:hi) AND body:world
```

### Lexical PEG Grammar

The full lexical grammar ([parser.pest](https://github.com/mosuka/laurus/blob/main/laurus/src/lexical/query/parser.pest)):

```pest
query          = { SOI ~ boolean_query ~ EOI }
boolean_query  = { clause ~ (boolean_op ~ clause | clause)* }
clause         = { required_clause | prohibited_clause | sub_clause }
required_clause   = { "+" ~ sub_clause }
prohibited_clause = { "-" ~ sub_clause }
sub_clause     = { grouped_query | field_query | term_query }
grouped_query  = { "(" ~ boolean_query ~ ")" ~ boost? }
boolean_op     = { ^"AND" | ^"OR" }
field_query    = { field ~ ":" ~ field_value }
field_value    = { geo3d_query | geo_query | range_query | phrase_query
                 | fuzzy_term | wildcard_term | simple_term }
geo3d_query    = { geo3d_distance | geo3d_bbox | geo3d_nearest }
geo3d_distance = { ^"geo3d_distance" ~ "(" ~ signed_float ~ "," ~ signed_float
                 ~ "," ~ signed_float ~ "," ~ signed_float ~ ")" }
geo3d_bbox     = { ^"geo3d_bbox" ~ "(" ~ signed_float ~ "," ~ signed_float
                 ~ "," ~ signed_float ~ "," ~ signed_float ~ ","
                 ~ signed_float ~ "," ~ signed_float ~ ")" }
geo3d_nearest  = { ^"geo3d_nearest" ~ "(" ~ signed_float ~ "," ~ signed_float
                 ~ "," ~ signed_float ~ "," ~ unsigned_int ~ ")" }
geo_query      = { geo_distance | geo_bbox }
geo_distance   = { ^"geo_distance" ~ "(" ~ signed_float ~ "," ~ signed_float
                 ~ "," ~ signed_float ~ ")" }
geo_bbox       = { ^"geo_bbox" ~ "(" ~ signed_float ~ "," ~ signed_float
                 ~ "," ~ signed_float ~ "," ~ signed_float ~ ")" }
phrase_query   = { "\"" ~ phrase_content ~ "\"" ~ proximity? ~ boost? }
proximity      = { "~" ~ number }
fuzzy_term     = { term ~ "~" ~ fuzziness? ~ boost? }
wildcard_term  = { wildcard_pattern ~ boost? }
simple_term    = { term ~ boost? }
boost          = { "^" ~ boost_value }
```

## Vector Query Syntax

Vector queries embed text into vectors at parse time and perform similarity search.

### Basic Syntax

```text
field:"text"
field:text
field:"text"^weight
```

The field name must refer to a vector field defined in the schema. The parser uses the schema to determine whether a clause is a vector query.

| Element | Required | Description | Example |
| :--- | :---: | :--- | :--- |
| `field:` | **Yes** | Target vector field name (must be a vector field in the schema) | `content:` |
| `"text"` or `text` | **Yes** | Text to embed (quoted or unquoted) | `"cute kitten"`, `python` |
| `^weight` | No | Score weight (default: 1.0) | `^0.8` |

### Vector Query Examples

```text
# Single field (quoted text)
content:"cute kitten"

# Unquoted text
content:python

# With boost weight
content:"cute kitten"^0.8

# Multiple clauses
content:"cats" image:"dogs"^0.5

# Nested field name (dot notation)
metadata.embedding:"text"
```

### Multiple Clauses

Multiple vector clauses are space-separated. All clauses are executed and their scores are combined using the `score_mode` (default: `WeightedSum`):

```text
content:"cats" image:"dogs"^0.5
```

This produces:

```text
score = similarity("cats", content) * 1.0
      + similarity("dogs", image)   * 0.5
```

There are no `AND`/`OR` operators in the vector DSL. Vector search is inherently a ranking operation, and the weight (`^`) controls the contribution of each clause.

### Score Modes

| Mode | Description |
| :--- | :--- |
| `WeightedSum` (default) | Sum of (similarity * weight) across all clauses |
| `MaxSim` | Maximum similarity score across clauses |
| `LateInteraction` | Late interaction scoring |

Score mode cannot be set from DSL syntax. Use the Rust API to override:

```rust
let mut request = parser.parse(r#"content:"cats" image:"dogs""#).await?;
request.vector_options.score_mode = VectorScoreMode::MaxSim;
```

### Vector PEG Grammar

The full vector grammar ([parser.pest](https://github.com/mosuka/laurus/blob/main/laurus/src/vector/query/parser.pest)):

```pest
query          = { SOI ~ vector_clause+ ~ EOI }
vector_clause  = { field_prefix ~ (quoted_text | unquoted_text) ~ boost? }
field_prefix   = { field_name ~ ":" }
field_name     = @{ (ASCII_ALPHA | "_") ~ (ASCII_ALPHANUMERIC | "_" | ".")* }
quoted_text    = ${ "\"" ~ inner_text ~ "\"" }
inner_text     = @{ (!("\"") ~ ANY)* }
unquoted_text  = @{ (!(" " | "^" | "\"") ~ ANY)+ }
boost          = { "^" ~ float_value }
float_value    = @{ ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT+)? }
```

## Unified (Hybrid) Query Syntax

The `UnifiedQueryParser` allows mixing lexical and vector clauses freely in a single query string:

```text
title:hello content:"cute kitten"^0.8
```

### How It Works

1. **Split**: The parser checks each field name against the schema. Fields defined as vector fields (e.g., HNSW, Flat, IVF) are routed to the vector parser; all other fields are routed to the lexical parser.
2. **Delegate**: Vector portion goes to `VectorQueryParser`, remainder goes to lexical `QueryParser`.
3. **Fuse**: If both lexical and vector results exist, they are combined using a fusion algorithm.

### Disambiguation

The parser uses the schema's field type information to distinguish vector clauses from lexical clauses. A clause like `content:"cute kitten"` is a vector query if `content` is a vector field, or a phrase query if `content` is a text field. Lexical `~` syntax (e.g., `roam~2` for fuzzy, `"hello world"~10` for proximity) is unaffected.

### Fusion Algorithms

When a query contains both lexical and vector clauses, results are fused:

| Algorithm | Formula | Description |
| :--- | :--- | :--- |
| **RRF** (default) | `score = sum(1 / (k + rank))` | Reciprocal Rank Fusion. Robust to different score distributions. Default k=60. |
| **WeightedSum** | `score = lexical * a + vector * b` | Linear combination with configurable weights. |

> **Note**: The fusion algorithm cannot be specified in the DSL syntax. It is configured when constructing the `UnifiedQueryParser` via `.with_fusion()`. The default is RRF (k=60). See [Custom Fusion](#custom-fusion) for a code example.

### Hybrid AND/OR Semantics (the `+` Prefix)

By default, hybrid queries use **union (OR)** — documents appearing in *either* the lexical results or the vector results are included. You can switch to **intersection (AND)** by prefixing a vector clause with `+`, which requires documents to appear in *both* result sets.

| Syntax | Mode | Behaviour |
| :--- | :---: | :--- |
| `title:Rust content:"system process"` | OR (union) | Documents matching the lexical query **or** the vector query are returned. |
| `title:Rust +content:"system process"` | AND (intersection) | Only documents matching **both** the lexical and vector results are returned. |
| `+title:Rust +content:"system process"` | AND (intersection) | Both clauses required. `+` on the lexical field is handled by the lexical parser as a required clause. |

Rules:

- When **no** vector clause carries the `+` prefix, the fusion produces a **union** (OR) of lexical and vector results.
- When **at least one** vector clause carries the `+` prefix, the fusion switches to **intersection** (AND) — only documents present in both the lexical and vector result sets are returned.
- `+` on a lexical field (e.g., `+title:Rust`) is interpreted by the lexical query parser as a *required clause*, which is the existing Tantivy/Lucene-style behaviour. It does not by itself trigger intersection mode for the hybrid fusion.

### Unified Query Examples

```text
# Lexical only — no fusion
title:hello AND body:world

# Vector only — no fusion
content:"cute kitten"

# Hybrid — fusion applied automatically (OR / union)
title:hello content:"cute kitten"

# Hybrid with AND / intersection — only docs in both result sets
title:hello +content:"cute kitten"

# Hybrid with boolean operators
title:hello AND category:animal content:"cute kitten"^0.8

# Multiple vector clauses + lexical
category:animal content:"cats" image:"dogs"^0.5

# Unquoted vector text
category:animal content:python
```

## Code Examples

### Lexical Search with DSL

```rust
use std::sync::Arc;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::query::QueryParser;

let analyzer = Arc::new(StandardAnalyzer::new()?);
let parser = QueryParser::new(analyzer)
    .with_default_field("title");

let query = parser.parse("title:hello AND body:world")?;
```

### Vector Search with DSL

```rust
use std::sync::Arc;
use laurus::vector::query::VectorQueryParser;

let parser = VectorQueryParser::new(embedder)
    .with_default_field("content");

let request = parser.parse(r#"content:"cute kitten"^0.8"#).await?;
```

### Hybrid Search with Unified DSL

```rust
use laurus::engine::query::UnifiedQueryParser;

let unified = UnifiedQueryParser::new(lexical_parser, vector_parser);

let request = unified.parse(
    r#"title:hello content:"cute kitten"^0.8"#
).await?;
// request.query              -> SearchQuery::Hybrid { lexical, vector }
// request.fusion_algorithm   -> Some(RRF)  — fusion algorithm
```

### Custom Fusion

```rust
use laurus::engine::search::FusionAlgorithm;

let unified = UnifiedQueryParser::new(lexical_parser, vector_parser)
    .with_fusion(FusionAlgorithm::WeightedSum {
        lexical_weight: 0.3,
        vector_weight: 0.7,
    });
```
