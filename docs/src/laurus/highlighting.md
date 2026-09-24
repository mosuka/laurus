# Highlighting

Highlighting marks matching terms in search results, helping users see why a document matched their query. Laurus generates highlighted text fragments with configurable HTML tags.

## Highlighting search results

The easiest way to get highlights is through the search API itself: ask for them on a [`SearchRequest`](./engine.md), and each hit's [`SearchResult::highlights`](./api_reference.md) comes back filled in.

```rust
use laurus::{HighlightConfig, SearchRequestBuilder};

let request = SearchRequestBuilder::new()
    .lexical_query(query)
    .highlight(vec!["body".to_string()])
    .highlight_config(HighlightConfig::default().tag("em".to_string()))
    .build();

let results = engine.search(request).await?;
for result in &results {
    if let Some(fragments) = result.highlights.get("body") {
        println!("{}", fragments.join(" ... "));
    }
}
```

`highlight(fields)` and `highlight_config(config)` are independent `SearchRequestBuilder` methods and compose in either order; calling `highlight` again replaces the field list, and the same holds for `highlight_config`. Omitting `highlight` (or requesting no fields) leaves every hit's `highlights` map empty — the engine does no highlighting work in that case.

Semantics to keep in mind:

- **Field selection.** Only fields named in `highlight(fields)` are considered, and only when they are `stored: true` text fields. A field that is absent from the document, not stored, or not a text field is silently skipped — it never appears as a key in `highlights`. A multi-valued text field (`multi_valued: true`, Issue #1175) is highlighted element by element: only the elements that match contribute fragments, a fragment never straddles two elements, and the concatenated fragment list honours `max_fragments`.
- **Analyzer.** Each field is tokenized with its own index-time analyzer (per-field analyzers apply automatically), so highlighting reflects what actually matched at index time — not a generic tokenizer.
- **Which query is highlighted.** The request's lexical query drives highlighting, including in hybrid search — a vector-only request produces no highlights. Terms contributed only by the request-level `filter_query` are never highlighted, since they describe eligibility, not relevance to what was searched for.
- **Empty result.** A field with no highlight fragments (no match, or `return_entire_field_if_no_highlight` not set) is omitted from `highlights` entirely rather than mapped to an empty list.
- **Cost.** Highlighting runs after pagination, so its cost scales with `limit × len(fields)`, not with the total number of matches.

For direct control over highlighting outside the search API — for example to highlight text that never went through `Engine::search` — use `Highlighter` as described below.

## HighlightConfig

`HighlightConfig` controls how highlights are generated:

```rust
use laurus::lexical::search::features::highlight::HighlightConfig;

let config = HighlightConfig::default()
    .tag("mark")
    .css_class("highlight")
    .max_fragments(3)
    .fragment_size(200);
```

### Configuration Options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tag` | `String` | `"mark"` | HTML tag used for highlighting |
| `css_class` | `Option<String>` | `None` | Optional CSS class added to the tag |
| `max_fragments` | `usize` | 5 | Maximum number of fragments to return |
| `fragment_size` | `usize` | 150 | Target fragment length in characters |
| `fragment_overlap` | `usize` | 20 | Reserved; not currently read by fragment selection |
| `fragment_separator` | `String` | `" ... "` | Reserved; not currently read — fragments are returned as a list, not pre-joined |
| `return_entire_field_if_no_highlight` | `bool` | false | Return the full field value if no matches found |
| `max_analyzed_chars` | `usize` | 1,000,000 | Maximum characters to analyze for highlights |
| `require_field_match` | `bool` | true | Use only query terms that target the highlighted field (set to `false` to highlight terms from any field in the query) |

### Builder Methods

| Method | Description |
| :--- | :--- |
| `tag(tag)` | Set the HTML tag (e.g., `"em"`, `"strong"`, `"mark"`) |
| `css_class(class)` | Set the CSS class for the tag |
| `max_fragments(count)` | Set maximum fragment count |
| `fragment_size(size)` | Set target fragment size in characters |
| `require_field_match(flag)` | Set whether only terms targeting the highlighted field are used |
| `opening_tag()` | Get the opening HTML tag string (e.g., `<mark class="highlight">`) |
| `closing_tag()` | Get the closing HTML tag string (e.g., `</mark>`) |

## Supported Queries

Highlight terms come from the query tree (`Query::collect_highlight_terms`), not from the query's description string, and are matched against the tokens produced by the highlighter's analyzer (`StandardAnalyzer` by default; use `Highlighter::with_analyzer` to match the field's analyzer, for example for Japanese text).

| Query | What is highlighted |
| :--- | :--- |
| `TermQuery` | Tokens equal to the term |
| `PhraseQuery` | Consecutive tokens forming the phrase, with the same in-order, per-gap `slop` rule as search; each occurrence is one highlight |
| `PrefixQuery`, `WildcardQuery`, `RegexpQuery`, `FuzzyQuery` | Every token the pattern (or edit distance) matches |
| `BooleanQuery` | Terms of `Must`, `Should` and `Filter` clauses; `MustNot` clauses are skipped |
| `AdvancedQuery` | Terms of the core query, filters and post filters; negative filters are skipped |
| `MultiFieldQuery` | Its text, in each configured field |
| Span queries (`SpanQueryWrapper`) | Every span term in the wrapped query |
| Range, numeric, date-time and geo queries | Nothing |

By default only terms that target the highlighted field are used (`require_field_match`). Set it to `false` to highlight terms from every field in the query.

## HighlightFragment

Each highlight result is a `HighlightFragment`:

```rust
pub struct HighlightFragment {
    pub text: String,
}
```

The `text` field contains the fragment with matching terms wrapped in the configured HTML tags.

## Output Example

Given a document with `body = "Rust is a systems programming language focused on safety and performance."` and a search for "rust programming":

```html
<mark>Rust</mark> is a systems <mark>programming</mark> language focused on safety and performance.
```

With `css_class("highlight")`:

```html
<mark class="highlight">Rust</mark> is a systems <mark class="highlight">programming</mark> language focused on safety and performance.
```

## Fragment Selection

When a field is long, Laurus selects the most relevant fragments:

1. The text is split into windows of `fragment_size` characters
2. Each fragment is scored by how many query terms it contains
3. The top `max_fragments` fragments are returned, in original order, as a `Vec<HighlightFragment>` (callers join or render the list themselves)

For a multi-valued text field these steps run per element, so a fragment never straddles two elements; `max_fragments` caps the list concatenated across all elements.

If no fragments contain matches and `return_entire_field_if_no_highlight` is true, the full field value is returned instead.
