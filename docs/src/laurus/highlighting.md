# Highlighting

Highlighting marks matching terms in search results, helping users see why a document matched their query. Laurus generates highlighted text fragments with configurable HTML tags.

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
| `fragment_overlap` | `usize` | 20 | Character overlap between adjacent fragments |
| `fragment_separator` | `String` | `" ... "` | Separator between fragments |
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

1. The text is split into overlapping windows of `fragment_size` characters
2. Each fragment is scored by how many query terms it contains
3. The top `max_fragments` fragments are returned, joined by `fragment_separator`

If no fragments contain matches and `return_entire_field_if_no_highlight` is true, the full field value is returned instead.
