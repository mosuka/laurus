# Faceting

Faceting enables counting and categorizing search results by field values. It is commonly used to build navigation filters in search UIs (e.g., "Electronics (42)", "Books (18)").

## Concepts

### FacetPath

A `FacetPath` represents a hierarchical facet value. For example, a product category "Electronics > Computers > Laptops" is a facet path with three levels.

```rust
use laurus::lexical::search::features::facet::FacetPath;

// Single-level facet
let facet = FacetPath::from_value("category", "Electronics");

// Hierarchical facet from components
let facet = FacetPath::new("category", vec![
    "Electronics".to_string(),
    "Computers".to_string(),
    "Laptops".to_string(),
]);

// From a delimited string
let facet = FacetPath::from_delimited("category", "Electronics/Computers/Laptops", "/");
```

#### FacetPath Methods

| Method | Description |
| :--- | :--- |
| `new(field, path)` | Create a facet path from field name and path components |
| `from_value(field, value)` | Create a single-level facet |
| `from_delimited(field, path_str, delimiter)` | Parse a delimited path string |
| `depth()` | Number of levels in the path |
| `is_parent_of(other)` | Check if this path is a parent of another |
| `parent()` | Get the parent path (one level up) |
| `child(component)` | Create a child path by appending a component |
| `to_string_with_delimiter(delimiter)` | Convert to a delimited string |

### FacetCount

`FacetCount` represents the result of a facet aggregation:

```rust
pub struct FacetCount {
    pub path: FacetPath,
    pub count: u64,
    pub children: Vec<FacetCount>,
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `path` | `FacetPath` | The facet value |
| `count` | `u64` | Number of matching documents |
| `children` | `Vec<FacetCount>` | Child facets for hierarchical drill-down |

## Example: Hierarchical Facets

```text
Category
├── Electronics (42)
│   ├── Computers (18)
│   │   ├── Laptops (12)
│   │   └── Desktops (6)
│   └── Phones (24)
└── Books (35)
    ├── Fiction (20)
    └── Non-Fiction (15)
```

Each node in this tree corresponds to a `FacetCount` with its `children` populated for drill-down navigation.

## Use Cases

- **E-commerce**: Filter by category, brand, price range, rating
- **Document search**: Filter by author, department, date range, document type
- **Content management**: Filter by tags, topics, content status

## Performance

Facet counts are read from each field's **DocValues** column, not from the
stored document. For every collected hit the collector reads only the facet
field's value via the per-field DocValues lookup, so it never decodes or clones
the whole stored-fields blob when every faceted field has a DocValues column
(the default for any `stored: true` field, unless its type is excluded — see
below — or its `doc_values` option is explicitly set to `false`). A field that
lacks DocValues — because it opted out, isn't stored, or is a `Bytes`/`Vector`
value, which DocValues never carries regardless of the setting — transparently
falls back to the stored document, so results are identical either way; only
the read path changes.

Setting `doc_values: false` on a field that is never sorted or faceted on
shrinks its segment footprint, since the value is then written once (to the
stored document) instead of twice.
