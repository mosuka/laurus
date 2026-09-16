# ハイライト

ハイライト（Highlighting）は検索結果内のマッチした単語をマークアップし、ドキュメントがクエリにマッチした理由をユーザーに視覚的に提示します。Laurusは設定可能なHTMLタグでハイライトされたテキストフラグメントを生成します。

## HighlightConfig

`HighlightConfig` はハイライトの生成方法を制御します。

```rust
use laurus::lexical::search::features::highlight::HighlightConfig;

let config = HighlightConfig::default()
    .tag("mark")
    .css_class("highlight")
    .max_fragments(3)
    .fragment_size(200);
```

### 設定オプション

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `tag` | `String` | `"mark"` | ハイライトに使用するHTMLタグ |
| `css_class` | `Option<String>` | `None` | タグに追加するオプションのCSSクラス |
| `max_fragments` | `usize` | 5 | 返却するフラグメントの最大数 |
| `fragment_size` | `usize` | 150 | フラグメントの目標文字数 |
| `fragment_overlap` | `usize` | 20 | 隣接するフラグメント間のオーバーラップ文字数 |
| `fragment_separator` | `String` | `" ... "` | フラグメント間の区切り文字 |
| `return_entire_field_if_no_highlight` | `bool` | false | マッチがない場合にフィールド全体の値を返却する |
| `max_analyzed_chars` | `usize` | 1,000,000 | ハイライト解析対象の最大文字数 |
| `require_field_match` | `bool` | true | ハイライト対象フィールドを対象とするクエリ語のみを使う（`false` にするとクエリ内の全フィールドの語でハイライトする） |

### Builderメソッド

| メソッド | 説明 |
| :--- | :--- |
| `tag(tag)` | HTMLタグを設定（例: `"em"`、`"strong"`、`"mark"`） |
| `css_class(class)` | タグのCSSクラスを設定 |
| `max_fragments(count)` | フラグメントの最大数を設定 |
| `fragment_size(size)` | フラグメントの目標文字数を設定 |
| `require_field_match(flag)` | ハイライト対象フィールドを対象とする語のみを使うかを設定 |
| `opening_tag()` | 開始HTMLタグ文字列を取得（例: `<mark class="highlight">`） |
| `closing_tag()` | 終了HTMLタグ文字列を取得（例: `</mark>`） |

## 対応クエリ

ハイライト対象の語はクエリの説明文字列ではなくクエリ木（`Query::collect_highlight_terms`）から取得され、ハイライタのアナライザ（既定は `StandardAnalyzer`。日本語テキストなどフィールドのアナライザに合わせるには `Highlighter::with_analyzer` を使う）が生成したトークンと照合されます。

| クエリ | ハイライトされるもの |
| :--- | :--- |
| `TermQuery` | 語と一致するトークン |
| `PhraseQuery` | フレーズを構成する連続トークン。検索と同じ「順序どおり・語間の隙間が `slop` 以内」の規則で、出現 1 回が 1 つのハイライト |
| `PrefixQuery`、`WildcardQuery`、`RegexpQuery`、`FuzzyQuery` | パターン（または編集距離）に一致するすべてのトークン |
| `BooleanQuery` | `Must`・`Should`・`Filter` 節の語。`MustNot` 節は除外 |
| `AdvancedQuery` | コアクエリ・フィルタ・ポストフィルタの語。ネガティブフィルタは除外 |
| `MultiFieldQuery` | 設定された各フィールドにおけるクエリ文字列 |
| スパンクエリ（`SpanQueryWrapper`） | ラップされたクエリ内のすべてのスパン語 |
| Range・Numeric・DateTime・Geo クエリ | なし |

既定ではハイライト対象フィールドを対象とする語のみが使われます（`require_field_match`）。クエリ内の全フィールドの語でハイライトするには `false` に設定してください。

## HighlightFragment

各ハイライト結果は `HighlightFragment` です。

```rust
pub struct HighlightFragment {
    pub text: String,
}
```

`text` フィールドには、マッチした単語が設定されたHTMLタグで囲まれたフラグメントが含まれます。

## 出力例

`body = "Rust is a systems programming language focused on safety and performance."` というドキュメントに対して "rust programming" で検索した場合:

```html
<mark>Rust</mark> is a systems <mark>programming</mark> language focused on safety and performance.
```

`css_class("highlight")` を指定した場合:

```html
<mark class="highlight">Rust</mark> is a systems <mark class="highlight">programming</mark> language focused on safety and performance.
```

## フラグメント選択

フィールドが長い場合、Laurusは最も関連性の高いフラグメントを選択します。

1. テキストが `fragment_size` 文字のオーバーラップするウィンドウに分割されます
2. 各フラグメントは含まれるクエリ単語の数でスコアリングされます
3. 上位 `max_fragments` 個のフラグメントが `fragment_separator` で結合されて返却されます

マッチを含むフラグメントがなく、`return_entire_field_if_no_highlight` が true の場合、フィールド全体の値が代わりに返却されます。
