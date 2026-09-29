# Text Analysis

Text analysis is the process of converting raw text into searchable tokens. When a document is indexed, the analyzer breaks text fields into individual terms; when a query is executed, the same analyzer processes the query text to ensure consistency.

## The Analysis Pipeline

```mermaid
graph LR
    Input["Raw Text\n'The quick brown FOX jumps!'"]
    CF["UnicodeNormalizationCharFilter"]
    T["Tokenizer\nSplit into words"]
    F1["LowercaseFilter"]
    F2["StopFilter"]
    F3["StemFilter"]
    Output["Terms\n'quick', 'brown', 'fox', 'jump'"]

    Input --> CF --> T --> F1 --> F2 --> F3 --> Output
```

The analysis pipeline consists of:

1. **Char Filters** — normalize raw text at the character level before tokenization
2. **Tokenizer** — splits text into raw tokens (words, characters, n-grams)
3. **Token Filters** — transform, remove, or expand tokens (lowercase, stop words, stemming, synonyms)

## The Analyzer Trait

All analyzers implement the `Analyzer` trait:

```rust
pub trait Analyzer: Send + Sync + Debug {
    fn analyze(&self, text: &str) -> Result<TokenStream>;
    fn name(&self) -> &str;
    fn as_any(&self) -> &dyn Any;
}
```

`TokenStream` is a `Box<dyn Iterator<Item = Token> + Send>` — a lazy iterator over tokens.

A `Token` contains:

| Field | Type | Description |
| :--- | :--- | :--- |
| `text` | `String` | The token text |
| `position` | `usize` | Position in the original text |
| `start_offset` | `usize` | Start byte offset in original text |
| `end_offset` | `usize` | End byte offset in original text |
| `position_increment` | `usize` | Distance from previous token |
| `position_length` | `usize` | Span of the token (>1 for synonyms) |
| `boost` | `f32` | Token-level scoring weight |
| `stopped` | `bool` | Whether marked as a stop word |
| `metadata` | `Option<TokenMetadata>` | Additional token metadata |

## Built-in Analyzers

### StandardAnalyzer

The default analyzer. Suitable for most Western languages.

Pipeline: `RegexTokenizer` (Unicode word boundaries) → `LowercaseFilter` → `StopFilter` (128 common English stop words)

```rust
use laurus::analysis::analyzer::standard::StandardAnalyzer;

let analyzer = StandardAnalyzer::default();
// "The Quick Brown Fox" → ["quick", "brown", "fox"]
// ("The" is removed by stop word filtering)
```

### JapaneseAnalyzer

Uses morphological analysis for Japanese text segmentation.

Pipeline: `UnicodeNormalizationCharFilter` (NFKC) → `JapaneseIterationMarkCharFilter` → `LinderaTokenizer` → `LowercaseFilter` → `StopFilter` (Japanese stop words)

`JapaneseAnalyzer::new` takes the same arguments as `LinderaTokenizer::new`:
the segmentation mode, a path to a Lindera dictionary directory, and an
optional user dictionary path. `laurus` does not enable Lindera's
`embed-*` features by default, so a real filesystem path (typically an
IPADIC build) is required at runtime.

```rust
use laurus::analysis::analyzer::language::japanese::JapaneseAnalyzer;

// Pass the path where you have unpacked the Lindera dictionary.
let analyzer = JapaneseAnalyzer::new(
    "normal",
    "/var/lib/lindera/ipadic",
    None,
)?;
// "東京都に住んでいる" → ["東京", "都", "住ん", "いる"]
```

When the analyzer is referenced from a `Schema`, supply the parameters
through the structured `AnalyzerSpec` form (see [PerFieldAnalyzer](#perfieldanalyzer) below).

### KeywordAnalyzer

Treats the entire input as a single token. No tokenization or normalization.

```rust
use laurus::analysis::analyzer::keyword::KeywordAnalyzer;

let analyzer = KeywordAnalyzer::new();
// "Hello World" → ["Hello World"]
```

Use this for fields that should match exactly (categories, tags, status codes).

### SimpleAnalyzer

Tokenizes text without any filtering. The original case and all tokens are preserved. Useful when you need complete control over the analysis pipeline or want to test a tokenizer in isolation.

Pipeline: User-specified `Tokenizer` only (no char filters, no token filters)

```rust
use laurus::analysis::analyzer::simple::SimpleAnalyzer;
use laurus::analysis::tokenizer::regex::RegexTokenizer;
use std::sync::Arc;

let tokenizer = Arc::new(RegexTokenizer::new()?);
let analyzer = SimpleAnalyzer::new(tokenizer);
// "Hello World" → ["Hello", "World"]
// (no lowercasing, no stop word removal)
```

Use this for testing tokenizers, or when you want to apply token filters manually in a separate step.

### EnglishAnalyzer

An English-specific analyzer. Tokenizes, lowercases, and removes common English stop words.

Pipeline: `RegexTokenizer` (Unicode word boundaries) → `LowercaseFilter` → `StopFilter` (128 common English stop words)

```rust
use laurus::analysis::analyzer::language::english::EnglishAnalyzer;

let analyzer = EnglishAnalyzer::new()?;
// "The Quick Brown Fox" → ["quick", "brown", "fox"]
// ("The" is removed by stop word filtering, remaining tokens are lowercased)
```

### PipelineAnalyzer

Build a custom pipeline by combining any char filters, a tokenizer, and any sequence of token filters:

```rust
use laurus::analysis::analyzer::pipeline::PipelineAnalyzer;
use laurus::analysis::char_filter::unicode_normalize::{
    NormalizationForm, UnicodeNormalizationCharFilter,
};
use laurus::analysis::tokenizer::regex::RegexTokenizer;
use laurus::analysis::token_filter::lowercase::LowercaseFilter;
use laurus::analysis::token_filter::stop::StopFilter;
use laurus::analysis::token_filter::stem::StemFilter;

let analyzer = PipelineAnalyzer::new(Arc::new(RegexTokenizer::new()?))
    .add_char_filter(Arc::new(UnicodeNormalizationCharFilter::new(NormalizationForm::NFKC)))
    .add_filter(Arc::new(LowercaseFilter::new()))
    .add_filter(Arc::new(StopFilter::new()))
    .add_filter(Arc::new(StemFilter::new()));  // Porter stemmer
```

## PerFieldAnalyzer

`PerFieldAnalyzer` lets you assign different analyzers to different fields within the same engine:

```mermaid
graph LR
    PFA["PerFieldAnalyzer"]
    PFA -->|"title"| KW["KeywordAnalyzer"]
    PFA -->|"body"| STD["StandardAnalyzer"]
    PFA -->|"description_ja"| JP["JapaneseAnalyzer"]
    PFA -->|other fields| DEF["Default\n(StandardAnalyzer)"]
```

```rust
use std::sync::Arc;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::analysis::analyzer::keyword::KeywordAnalyzer;
use laurus::analysis::analyzer::per_field::PerFieldAnalyzer;

// Default analyzer for fields not explicitly configured
let per_field = PerFieldAnalyzer::new(
    Arc::new(StandardAnalyzer::default())
);

// Use KeywordAnalyzer for exact-match fields
per_field.add_analyzer("category", Arc::new(KeywordAnalyzer::new()));
per_field.add_analyzer("status", Arc::new(KeywordAnalyzer::new()));

let engine = Engine::builder(storage, schema)
    .analyzer(Arc::new(per_field))
    .build()
    .await?;
```

> **Note:** The `_id` field is always analyzed with `KeywordAnalyzer` regardless of configuration.

### Configuring per-field analyzers from a Schema

Most callers configure analyzers declaratively on the schema rather than
wiring them up by hand. The `analyzer` setting on a text field accepts
two shapes:

```jsonc
// 1. A bare name for a parameter-less built-in or a user-registered analyzer.
{ "analyzer": "standard" }
{ "analyzer": "english" }
{ "analyzer": "my_custom_pipeline" }

// 2. A structured object for a parameterised built-in preset. Today only
//    the Japanese preset uses this form (it requires a Lindera dictionary
//    path).
{
  "analyzer": {
    "language": "japanese",
    "mode": "normal",
    "dict": "/var/lib/lindera/ipadic"
  }
}
```

The bare string `"japanese"` is rejected because the preset cannot be
constructed without a dictionary. Schemas that previously stored
`"analyzer": "japanese"` must migrate to the structured form above.

For full pipelines that do not fit a preset, register the pipeline under
`schema.analyzers` as an `AnalyzerDefinition` and reference it by name.

## Char Filters

Char filters operate on the raw input text **before** it reaches the tokenizer. They perform character-level normalization such as Unicode normalization, character mapping, and pattern-based replacement. This ensures that the tokenizer receives clean, normalized text.

All char filters implement the `CharFilter` trait:

```rust
pub trait CharFilter: Send + Sync {
    fn filter(&self, input: &str) -> (String, Vec<Transformation>);
    fn name(&self) -> &'static str;
}
```

The `Transformation` records describe how character positions shifted, allowing the engine to map token positions back to the original text.

| Char Filter | Description |
| :--- | :--- |
| `UnicodeNormalizationCharFilter` | Unicode normalization (NFC, NFD, NFKC, NFKD) |
| `MappingCharFilter` | Replaces character sequences based on a mapping dictionary |
| `PatternReplaceCharFilter` | Replaces characters matching a regex pattern |
| `JapaneseIterationMarkCharFilter` | Expands Japanese iteration marks (踊り字) to their base characters |

### UnicodeNormalizationCharFilter

Applies Unicode normalization to the input text. NFKC is recommended for search use cases because it normalizes both compatibility characters and composed forms.

```rust
use laurus::analysis::char_filter::unicode_normalize::{
    NormalizationForm, UnicodeNormalizationCharFilter,
};

let filter = UnicodeNormalizationCharFilter::new(NormalizationForm::NFKC);
// "Ｓｏｎｙ" (fullwidth) → "Sony" (halfwidth)
// "㌂" → "アンペア"
```

| Form | Description |
| :--- | :--- |
| NFC | Canonical decomposition followed by canonical composition |
| NFD | Canonical decomposition |
| NFKC | Compatibility decomposition followed by canonical composition |
| NFKD | Compatibility decomposition |

### MappingCharFilter

Replaces character sequences using a dictionary. Matches are found using the Aho-Corasick algorithm (leftmost-longest match).

```rust
use std::collections::HashMap;
use laurus::analysis::char_filter::mapping::MappingCharFilter;

let mut mapping = HashMap::new();
mapping.insert("ph".to_string(), "f".to_string());
mapping.insert("qu".to_string(), "k".to_string());

let filter = MappingCharFilter::new(mapping)?;
// "phone queue" → "fone keue"
```

### PatternReplaceCharFilter

Replaces all occurrences of a regex pattern with a fixed string.

```rust
use laurus::analysis::char_filter::pattern_replace::PatternReplaceCharFilter;

// Remove hyphens
let filter = PatternReplaceCharFilter::new(r"-", "")?;
// "123-456-789" → "123456789"

// Normalize numbers
let filter = PatternReplaceCharFilter::new(r"\d+", "NUM")?;
// "Year 2024" → "Year NUM"
```

### JapaneseIterationMarkCharFilter

Expands Japanese iteration marks (踊り字) to their base characters. Supports kanji (`々`), hiragana (`ゝ`, `ゞ`), and katakana (`ヽ`, `ヾ`) iteration marks.

```rust
use laurus::analysis::char_filter::japanese_iteration_mark::JapaneseIterationMarkCharFilter;

let filter = JapaneseIterationMarkCharFilter::new(
    true,  // normalize kanji iteration marks
    true,  // normalize kana iteration marks
);
// "佐々木" → "佐佐木"
// "いすゞ" → "いすず"
```

### Using Char Filters in a Pipeline

Add char filters to a `PipelineAnalyzer` with `add_char_filter()`. Multiple char filters are applied in the order they are added, all before the tokenizer runs.

```rust
use std::sync::Arc;
use laurus::analysis::analyzer::pipeline::PipelineAnalyzer;
use laurus::analysis::char_filter::unicode_normalize::{
    NormalizationForm, UnicodeNormalizationCharFilter,
};
use laurus::analysis::char_filter::pattern_replace::PatternReplaceCharFilter;
use laurus::analysis::tokenizer::regex::RegexTokenizer;
use laurus::analysis::token_filter::lowercase::LowercaseFilter;

let analyzer = PipelineAnalyzer::new(Arc::new(RegexTokenizer::new()?))
    .add_char_filter(Arc::new(
        UnicodeNormalizationCharFilter::new(NormalizationForm::NFKC),
    ))
    .add_char_filter(Arc::new(
        PatternReplaceCharFilter::new(r"-", "")?,
    ))
    .add_filter(Arc::new(LowercaseFilter::new()));
// "Ｔｏｋｙｏ-2024" → NFKC → "Tokyo-2024" → remove hyphens → "Tokyo2024" → tokenize → lowercase → ["tokyo2024"]
```

## Tokenizers

| Tokenizer | Description |
| :--- | :--- |
| `RegexTokenizer` | Unicode word boundaries; splits on whitespace and punctuation |
| `UnicodeWordTokenizer` | Splits on Unicode word boundaries |
| `WhitespaceTokenizer` | Splits on whitespace only |
| `WholeTokenizer` | Returns the entire input as a single token |
| `LinderaTokenizer` | Japanese morphological analysis (Lindera/MeCab) |
| `NgramTokenizer` | Generates n-gram tokens of configurable size |

## Token Filters

| Filter | Description |
| :--- | :--- |
| `LowercaseFilter` | Converts tokens to lowercase |
| `StopFilter` | Removes common words ("the", "is", "a") |
| `StemFilter` | Reduces words to their root form ("running" → "run") |
| `SynonymGraphFilter` | Expands tokens with synonyms from a dictionary |
| `BoostFilter` | Adjusts token boost values |
| `LimitFilter` | Limits the number of tokens |
| `StripFilter` | Strips leading/trailing whitespace from tokens |
| `FlattenGraphFilter` | Flattens token graphs. Indexing already does this, so leave it out of an analyzer that also parses queries (see [The token graph](#the-token-graph)) |
| `RemoveEmptyFilter` | Removes empty tokens |

### Synonym Expansion

The `SynonymGraphFilter` expands terms using a synonym dictionary:

```rust
use laurus::analysis::synonym::dictionary::SynonymDictionary;
use laurus::analysis::token_filter::synonym_graph::SynonymGraphFilter;

let mut dict = SynonymDictionary::new(None)?;
dict.add_synonym_group(vec!["ml".into(), "machine learning".into()]);
dict.add_synonym_group(vec!["ai".into(), "artificial intelligence".into()]);

// keep_original=true means original token is preserved alongside synonyms
let filter = SynonymGraphFilter::new(dict, true)
    .with_boost(0.8);  // synonyms get 80% weight
```

The `boost` parameter controls how much weight synonyms receive relative to original tokens. A value of `0.8` means synonym matches contribute 80% as much to the score as exact matches.

#### Matching entries of several words

An entry of several words matches consecutive tokens when all of them are alphanumeric (token type `Alphanum` or `Num`), so English words separated by spaces match, or when their offsets touch (each token's `end_offset` is the next one's `start_offset`), as the CJK words a morphological tokenizer splits do. The type comes from `Token::metadata`, so a token without one needs touching offsets. With `東京大学` in the dictionary, the text `東京大学` split into `東京` and `大学` matches it, but `東京 大学` does not: a space separates the two tokens.

#### The token graph

The filter stacks each synonym on the word it expands (`position_increment = 0`), so the output is a graph: a token is an arc from its position to its position + `position_length`. The matched words and every synonym run from the same start node to the same end node. The last token of a shorter alternative spans the remaining positions, and the token after the match starts at the end node. For `ml tutorial` with `ml` and `machine learning` as synonyms:

| Token | Position | `position_increment` | `position_length` |
| :--- | :--- | :--- | :--- |
| `ml` | 0 | 1 | 2 |
| `machine` | 0 | 0 | 1 |
| `learning` | 1 | 1 | 1 |
| `tutorial` | 2 | 1 | 1 |

A member of several words has inner nodes of its own, so no arc holds words of two members. For `ml` with `machine learning` and `statistical machine learning` as synonyms:

| Token | Arc |
| :--- | :--- |
| `ml` | 0 → 4 |
| `machine` | 0 → 1 |
| `statistical` | 0 → 2 |
| `learning` | 1 → 4 |
| `machine` | 2 → 3 |
| `learning` | 3 → 4 |

The index stores no `position_length`, so it lays the graph out on its longest path: the members' words share positions there. Above, `ml`, `machine` and `statistical` are stored at 0, `learning` and `machine` at 1, and `learning` at 2. A stacked token is stored at its word's position. Positions otherwise stay dense: a stop word removed by `StopFilter` leaves no gap. The index does this itself, so an analyzer needs no `FlattenGraphFilter`. Adding one would also flatten the graph that quoted values are matched through, because the engine parses queries with the same analyzer.

#### Searching with synonyms

An engine uses a field's analyzer both to index and to parse queries, so synonyms are expanded on both sides. An unquoted word matches any of its synonyms. A quoted value is matched through the graph (see [Phrase Query](query_dsl.md#phrase-query)): `"big"` with `big` and `large` as synonyms matches either word, and `"a big dog"` also matches "a large dog". Each member of a group is its own phrase, so with the members above `"ml"` matches `ml`, `machine learning` or `statistical machine learning`, and not "statistical learning". Members of the same length are separate phrases too. The phrases of a quoted value are matched together through the graph, so their number, which multiplies with each synonym in the value, sets no limit.

Because the index keeps positions but not `position_length`, index-time expansion of multi-word synonyms behaves as in Lucene:

- A phrase may start or end inside a synonym. `"learning is"` matches "ml is fun", which reads "machine learning is fun" with the synonym.
- When a group has several multi-word members, their words share positions in the index, so a document with one member matches a phrase that mixes two members' words: "ml is fun" matches `"statistical learning"`.

Other points to keep in mind:

- `keep_original = false` replaces the matched word with the other members of its group, so the word itself is neither indexed nor searched. Use `keep_original = true` for search.
- Filters that remove tokens, such as `StopFilter`, `RemoveEmptyFilter` and `LimitFilter`, can follow `SynonymGraphFilter`. A removed word leaves no position inside the graph either: with `statue of liberty` and `lady liberty` as synonyms and `of` as a stop word, `"statue of liberty"` matches both "statue of liberty" and "lady liberty". A `StopFilter` before `SynonymGraphFilter` removes `of` before the synonym filter sees it, so that member never matches.
- A removed word with an alternative at its position, such as a stacked one-word synonym, leaves its position to it: with `the` and `a` as synonyms, removing `the` leaves `a` there, and no path skips the word. `LimitFilter` cuts every path at the same token.
- Documents indexed before Issue #1252 numbered stacked tokens one after another. Put them again after upgrading if their fields use `SynonymGraphFilter`: merging segments copies the stored positions unchanged.
