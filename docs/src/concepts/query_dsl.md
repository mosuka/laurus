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

A pre-built lexical query (`SearchQuery::Lexical`, as a query object or a
DSL string, and the lexical part of a hybrid search) and a `filter_query` are
checked too ([#1635](https://github.com/mosuka/laurus/issues/1635)). They
may read only the schema's lexical fields and `_id`: a field the schema does
not have, or a vector field, which a lexical query can never match, is
rejected with `InvalidArgument`. A bare term expands to the schema's
`default_fields`, so it no longer reaches a default field that was deleted.

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

A backslash before a special character of the DSL (`+ - ! ( ) { } [ ] ^ " ~ * ? : \ /`) makes it part of the term: `title:a\:b` searches the term `a:b`, and `title:a\\b` the term `a\b`. The backslash is dropped before the field's analyzer reads the term, as in Lucene's query parser, and so it is in a [fuzzy term](#fuzzy-query) (`title:a\:b~1`). Until Issue #1754 it was kept, so on a field whose analyzer keeps a value whole, such as `keyword`, `title:a\:b` searched `a\:b`.

### Boolean Operators

Combine clauses with `AND` and `OR`, written in capitals:

```text
title:hello AND body:world
title:hello OR title:goodbye
```

`AND`, `OR` and [`NOT`](#required--prohibited-clauses) are operators only in capitals and as whole words, as in Lucene's query parser: `and`, `Or`, `ANDROID` and `NOTE` are terms. They used to be operators in any case, and after another clause a word that started with one was split, so `apple android` searched `apple AND roid` and `apple orange` searched `apple OR ange` (Issue #1747).

`AND` is symmetric — it makes the clauses on **both** sides required (`Must`). For example, `title:hello AND body:world` returns only documents that match **both** clauses. The same is true for chains: `a AND b AND c` requires all three. A clause that was already explicitly marked with `+` (required) or `-` (prohibited) keeps that intent — `AND` does not override an explicit prefix.

Space-separated clauses without an explicit operator use implicit boolean (behaves like OR with scoring), so `a b AND c` reads as "optionally match `a`, and require both `b` and `c`".

### Required / Prohibited Clauses

Use `+` (must match) and `-` (must not match):

```text
+title:hello -title:goodbye
```

`NOT`, in capitals, prohibits the clause after it as `-` does (Issue #1747):

```text
title:hello NOT title:goodbye
title:hello AND NOT title:goodbye
NOT (title:hello OR title:goodbye)
```

`a NOT b` and `a OR NOT b` are `a -b`, and `a AND NOT b` is `+a -b`. A `NOT` with no clause after it is a term. To search the word itself, write it in lowercase (`not`), quote it (`"NOT"`), or write it after a field name (`title:NOT`); `StandardAnalyzer` drops `not` as a stop word, so only a field whose analyzer keeps it holds the word.

A query of `-` or `NOT` clauses alone, such as `-title:goodbye`, matches every other document, and each with the score 0, as in Elasticsearch: a clause that excludes carries no score (Issue #1744). It used to score each match as a term that one document has once, a value that grew with the index.

### Phrase Query

Match an exact phrase using double quotes. Optional proximity (`~N`) allows N words between terms:

```text
"hello world"
"hello world"~2
```

On a multi-valued text field the phrase (or its `~N` proximity window) never crosses from one element into the next unless `N` reaches the field's `position_increment_gap` (default 100).

`N` is a number from 0 to 4294967295, the largest 32-bit unsigned integer. A larger one is rejected with `InvalidArgument` (HTTP 400), also on a quoted value that ignores it (below). It used to become 0, an exact phrase, without an error (Issue #1570).

A quoted value that the field's analyzer turns into a single token is an exact term match, the same as a [Term Query](#term-query) (any `~N` is ignored). This is how to search a `keyword` field for a value the unquoted form does not accept, such as a path, a URL, or an ID containing `.`, `/`, or `:`:

```text
path:"file:///notes/a.md"
links:"https://example.com/a"
```

When the field's analyzer stacks several tokens at one position, as `SynonymGraphFilter` does, they are alternatives at that position. With `big` and `large` as synonyms, `"big"` matches documents that contain either word, and `"a big dog"` also matches "a large dog". A single position's alternatives are scored as one blended term (a `SynonymQuery`), not summed, so a document containing both `big` and `large` does not score higher than one containing `big` twice. A multi-word synonym makes one phrase per alternative, and the value matches when any of them does: with `ml` and `machine learning` as synonyms, `"ml"` matches `ml` or the phrase `machine learning`. These phrases are matched together through the graph, not one by one, so there is no limit on how many a value makes, and a document gets one phrase score however many of them it contains. See [Synonym Expansion](analysis.md#synonym-expansion).

A phrase of two or more positions compares term positions, which a field stores only with `term_vectors: true` (the default). On a field with `term_vectors: false` the search is rejected with a query error instead of returning no hits. This includes an unfielded phrase that expands over the default fields when one of them has `term_vectors: false`, and a quoted word whose multi-word synonym makes it a phrase, such as `"ml"` above.

### Fuzzy Query

Approximate matching with edit distance. Append `~` and optionally the maximum edit distance:

```text
roam~
roam~2
```

Without a number, the maximum edit distance is 2. The number can be anything from 0 to 4294967295; a larger one is rejected with `InvalidArgument` (HTTP 400). One of 256 or more used to become 2 without an error (Issue #1570). There is no smaller limit: a distance at least as long as the term matches every term of the field.

The query finds the documents of the 50 closest matching terms, the exact term first, and among equally close ones of the most frequent. A closer term weighs more, and how rare a term is plays no part (see [FuzzyQuery](search/lexical_search.md#fuzzyquery)). An exclusion, `-roam~`, covers every matching term. Until Issue #1740 the query found the documents of every matching term and ranked a rare misspelling above a common word.

### Wildcard Query

Use `?` (single character) and `*` (zero or more characters):

```text
te?t
test*
```

Both match any character, a line break too. A wildcard takes the characters of a term, `-`, `'` and an escaped special character among them, so `a-b*`, `it's*` and `foo*-bar` are one wildcard each, as in Lucene; to exclude `bar`, put a space before the `-`: `foo* -bar`. A backslash before a special character of the DSL (`+ - ! ( ) { } [ ] ^ " ~ * ? : \ /`) makes it literal: `te\*t*` finds the terms that start with `te*t`. Until Issue #1745 a wildcard ended before any character but a letter, a digit or `_`, and left a bare `*`, which matched every term of the default field; and `*` and `?` did not match a line break.

A wildcard matches every document that has a matching term, however many terms match, and each match scores the boost (`1.0` unless `^` sets another). It used to find only the documents of 50 of the matching terms, the least frequent ones (Issue #1740). The more terms match, the more a search costs, and a pattern that starts with `?` or `*` is compared with every term of the field.

### Range Query

Inclusive `[]` or exclusive `{}` ranges, useful for numeric and date fields:

```text
price:[100 TO 500]
date:{2024-01-01 TO 2024-12-31}
price:[* TO 100]
```

#### Numeric bounds

The parser has no schema, so a range takes its type from its bounds. A bound is a number when it is written in digits: an optional sign, then digits with an optional fraction and exponent (`100`, `-1.5`, `+5`, `.5`, `1e3`). A range with a numeric bound builds a `NumericRangeQuery`, evaluated against the field's BKD tree with constant scoring.

```text
# A sign, a fraction and an exponent
price:[-1.5 TO 1e3]

# * leaves a side open; there is no -inf or inf
price:[* TO 100]
```

- `nan`, `inf` and `infinity`, in any case and with or without a sign, are words, not numbers. Write `*` for an open side: `price:[* TO 100]`, not `price:[-inf TO 100]`.
- A numeric bound next to a bound that is neither a number nor `*`, such as `price:[100 TO abc]` or `price:[nan TO 100]`, is a parse error (HTTP 400). Such a bound used to leave its side open (Issue #1591).
- A number too large for a 64-bit float, such as `1e400`, is rejected with `InvalidArgument` (HTTP 400) instead of becoming an infinite bound. A number too small to tell from `0`, such as `1e-400`, is `0`.
- Quoting a bound does not change its type: `price:["100" TO 500]` is a numeric range.

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
- Numbers ([Numeric bounds](#numeric-bounds)) are still numeric bounds: on a `DateTime` field they are epoch seconds and build a `NumericRangeQuery`.
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
| `geo_bbox(min_lat, min_lon, max_lat, max_lon)` | All docs whose stored `(lat, lon)` lies inside the latitude / longitude rectangle, edges included. With `min_lon > max_lon` the box crosses the antimeridian (±180°), as in GeoJSON; an edge on exactly ±180° matches points stored with either sign. |

Examples:

```text
# Within 10 km (= 10 000 m) of Tokyo (35.6895, 139.6917)
location:geo_distance(35.6895, 139.6917, 10000)

# Inside an axis-aligned lat/lon bounding box
location:geo_bbox(35.0, 139.0, 36.0, 140.0)

# Across the antimeridian: 175 degrees east to 175 degrees west
location:geo_bbox(-20.0, 175.0, -15.0, -175.0)
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
rus*^3
rust~1^2
(title:hello title:world)^2
```

A term, a phrase, a fuzzy or wildcard term and a group take a boost; a range or geo clause does not. The boost multiplies the clause's score, and is written as digits with an optional fraction, so it is never negative. `0` keeps the matches with a score of `0`. A number too large for a 32-bit float, such as forty `1`s, is rejected with `InvalidArgument` (HTTP 400) instead of giving `null` scores (Issue #1550). A query built through the Rust API follows the same rule: a boost must be finite and at least `0` (`-0.0` is rejected). A fuzzy or wildcard term used to drop its boost. A very large finite boost can still overflow a score to infinity, which also shows as `null`; there is no upper bound.

A field's analyzer weights the terms too, as Lucene's query parser does: a term or a phrase carries the boost of the tokens its text analyzes to, multiplied into its `^` boost (Issue #1701). With a `boost` token filter of `2`, `title:hello` scores like `title:hello^2` without the filter, and `title:"hello world"` like `^4`, the product of its two tokens' boosts. Tokens stacked at one position (synonyms) share the largest of their boosts. A phrase whose analyzer adds several multi-word synonym paths keeps only its `^` boost. Indexing does not use a token's boost. Before Issue #1701, no token boost reached a score.

A group's boost multiplies the boosts inside it, whatever the number of clauses or the nesting, as in Lucene: `(title:hello^2)^3` and `(title:hello^2 title:world)^3` both boost `hello` by 6, `((title:hello^2)^3)^4` boosts it by 24, and `(title:hello^2)^1` keeps 2. Two boosts whose product is too large for a 32-bit float are rejected with `InvalidArgument`, as one boost that large is. Before Issue #1568, a group of one clause, and a group around another group, replaced the boost it held, so `(title:hello^2)^3` boosted by 3 and `((title:hello^2)^3)^4` by 4.

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
prohibited_clause = { ("-" | not_op) ~ sub_clause }
sub_clause     = { grouped_query | field_query | term_query }
grouped_query  = { "(" ~ boolean_query ~ ")" ~ boost? }
boolean_op     = @{ ("AND" | "OR") ~ !keyword_tail }
not_op         = @{ "NOT" ~ !keyword_tail }
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
wildcard_pattern = @{ escaped_char* ~ wildcard_char
                    ~ (escaped_char | wildcard_char)* }
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
| `"text"` or `text` | **Yes** | Text to embed (quoted or unquoted). In the unified DSL an unquoted value ends at whitespace or a parenthesis, so `(content:cats)` closes its group; quote the text to include them (Issue #1377). If the embedder returns a vector with a `NaN` or infinite component, the search fails with `InvalidArgument` (HTTP 400) as soon as the clause is embedded (Issue #1573). So does a vector of another dimension than the field, once the DSL is parsed, whatever the data (Issue #1575) | `"cute kitten"`, `python` |
| `^weight` | No | Score weight (default: 1.0). A number too large for a 32-bit float is rejected with `InvalidArgument` (HTTP 400) before any clause is embedded; see [Weights](search/vector_search.md#weights) for the range (Issue #1552) | `^0.8` |

Versions before 0.6.0 marked a vector clause with `~` (`content:~"cute kitten"`). The marker is no longer part of the syntax: both `VectorQueryParser` and the unified parser reject a vector clause whose value starts with an unquoted `~`, with an error that names the marker (Issue #1495). Quote text that starts with `~`: `content:"~text"`.

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
| `LateInteraction` | Currently the same as `WeightedSum`; token-level late interaction is a [rescore](search/vector_search.md#late-interaction-rescore) |

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

1. **Split**: The parser checks each field name against the schema. Fields defined as vector fields (e.g., HNSW, Flat, IVF) are routed to the vector parser; all other fields are routed to the lexical parser. The whole field name must match: with a vector field `content`, `subcontent:x` or `my.content:x` stays a lexical clause (Issue #1373).
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
- A vector clause cannot be prohibited: a `-` or `NOT` before it (`-content:"cats"`), or before a group around it (`NOT (title:Rust content:"cats")`), is rejected with `InvalidArgument` (HTTP 400) before anything is embedded. A vector search ranks the documents nearest to the query and has nothing to exclude; exclude documents with a lexical clause or a filter instead. Such a query used to search the vector clause anyway, and the `-` or `NOT` was lost or moved to the next lexical clause (Issue #1751).
- Only what the lexical grammar reads as a clause is a vector clause: `content:cats` inside a quoted lexical phrase, as in `title:"cats content:cats"`, is the phrase's text. It used to be taken out as a vector clause (Issue #1751).

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
