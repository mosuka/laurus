# ハイライト

ハイライト（Highlighting）は検索結果内のマッチした単語をマークアップし、ドキュメントがクエリにマッチした理由をユーザーに視覚的に提示します。Laurusは設定可能なHTMLタグでハイライトされたテキストフラグメントを生成します。

## 検索結果のハイライト

ハイライトを取得する最も簡単な方法は、検索 API 自体に要求することです。[`SearchRequest`](./engine.md) でハイライトを指定すると、各ヒットの [`SearchResult::highlights`](./api_reference.md) に結果が入って返ってきます。

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

`highlight(fields)` と `highlight_config(config)` は独立した `SearchRequestBuilder` のメソッドで、どちらを先に呼んでも構いません。`highlight` を再度呼ぶとフィールド一覧は置き換わり、`highlight_config` も同様です。`highlight` を呼ばない（または空のフィールド一覧を渡す）場合、すべてのヒットの `highlights` は空のままになります。この場合エンジンはハイライト処理を一切行いません。

押さえておくべき挙動:

- **フィールド選択**: `highlight(fields)` で指定したフィールドのうち、`stored: true` のテキストフィールドのみが対象になります。ドキュメントに存在しない、`stored` でない、テキスト型でないフィールドは黙ってスキップされ、`highlights` のキーには現れません。
- **アナライザ**: 各フィールドはそのフィールド自身のインデックス時アナライザでトークナイズされます（フィールド別アナライザは自動的に適用されます）。そのため、ハイライトは汎用トークナイザではなく、インデックス時に実際にマッチした内容を反映します。
- **どのクエリがハイライトされるか**: ハイライトはリクエストの lexical クエリで駆動されます。これはハイブリッド検索でも同様で、vector-only のリクエストはハイライトを生成しません。リクエストレベルの `filter_query` だけが持ち込む語はハイライトされません。フィルタは検索対象への適合性を表すものであり、検索意図そのものではないためです。
- **結果が空の場合**: マッチするフラグメントがないフィールド（または `return_entire_field_if_no_highlight` を設定していない場合）は、空リストとしてではなく `highlights` から完全に除外されます。
- **コスト**: ハイライトはページネーション後に実行されるため、コストはマッチ総数ではなく `limit × フィールド数` に比例します。

`Engine::search` を経由しないテキストをハイライトするなど、検索 API の外でハイライトを直接制御したい場合は、以下で説明する `Highlighter` を使ってください。

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
| `fragment_overlap` | `usize` | 20 | 予約済み。現状フラグメント選択では参照されない |
| `fragment_separator` | `String` | `" ... "` | 予約済み。現状参照されない — フラグメントは結合済み文字列ではなくリストとして返却される |
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

1. テキストが `fragment_size` 文字のウィンドウに分割されます
2. 各フラグメントは含まれるクエリ単語の数でスコアリングされます
3. 上位 `max_fragments` 個のフラグメントが、元の出現順のまま `Vec<HighlightFragment>` として返却されます（結合や整形は呼び出し側が行います）

マッチを含むフラグメントがなく、`return_entire_field_if_no_highlight` が true の場合、フィールド全体の値が代わりに返却されます。
