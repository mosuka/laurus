# テキスト解析

テキスト解析（Text Analysis）は、生のテキストを検索可能なトークンに変換するプロセスです。ドキュメントがインデクシングされる際、Analyzer がテキストフィールドを個々のタームに分割します。クエリが実行される際も、同じ Analyzer がクエリテキストを処理し、一貫性を確保します。

## 解析パイプライン

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

解析パイプラインは以下で構成されます。

1. **Char Filter** — トークン化の前に文字レベルで生テキストを正規化する
2. **Tokenizer** — テキストを生トークン（単語、文字、n-gram）に分割する
3. **Token Filter** — トークンの変換、削除、展開を行う（小文字化、ストップワード除去、ステミング、同義語展開）

## Analyzer トレイト

すべての Analyzer は `Analyzer` トレイトを実装します。

```rust
pub trait Analyzer: Send + Sync + Debug {
    fn analyze(&self, text: &str) -> Result<TokenStream>;
    fn name(&self) -> &str;
    fn as_any(&self) -> &dyn Any;
}
```

`TokenStream` は `Box<dyn Iterator<Item = Token> + Send>` であり、トークンの遅延イテレータです。

`Token` には以下のフィールドが含まれます。

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `text` | `String` | トークンテキスト |
| `position` | `usize` | 元テキスト内の位置 |
| `start_offset` | `usize` | 元テキスト内の開始バイトオフセット |
| `end_offset` | `usize` | 元テキスト内の終了バイトオフセット |
| `position_increment` | `usize` | 前のトークンからの距離 |
| `position_length` | `usize` | トークンのスパン（同義語の場合は 1 より大きい） |
| `boost` | `f32` | トークンレベルのスコアリング重み |
| `stopped` | `bool` | ストップワードとしてマークされているかどうか |
| `metadata` | `Option<TokenMetadata>` | 追加のトークンメタデータ |

## 組み込み Analyzer

### StandardAnalyzer

デフォルトの Analyzer です。ほとんどの西洋言語に適しています。

パイプライン: `RegexTokenizer`（Unicode 単語境界） → `LowercaseFilter` → `StopFilter`（128 個の一般的な英語ストップワード）

```rust
use laurus::analysis::analyzer::standard::StandardAnalyzer;

let analyzer = StandardAnalyzer::default();
// "The Quick Brown Fox" → ["quick", "brown", "fox"]
// ("The" is removed by stop word filtering)
```

### JapaneseAnalyzer

日本語テキストの分割に形態素解析を使用します。

パイプライン: `UnicodeNormalizationCharFilter`（NFKC） → `JapaneseIterationMarkCharFilter` → `LinderaTokenizer` → `LowercaseFilter` → `StopFilter`（日本語ストップワード）

`JapaneseAnalyzer::new` は `LinderaTokenizer::new` と同じ引数（segmentation mode、Lindera 辞書ディレクトリのパス、任意のユーザー辞書パス）を受け取ります。`laurus` は Lindera の `embed-*` features をデフォルトで有効化しないため、IPADIC 等の辞書を実ファイルパスとして必ず指定する必要があります。

```rust
use laurus::analysis::analyzer::language::japanese::JapaneseAnalyzer;

// Lindera 辞書を展開済みのパスを指定する。
let analyzer = JapaneseAnalyzer::new(
    "normal",
    "/var/lib/lindera/ipadic",
    None,
)?;
// "東京都に住んでいる" → ["東京", "都", "住ん", "いる"]
```

`Schema` 経由で参照する場合は構造化された `AnalyzerSpec` 形式でパラメータを渡します（後述の [PerFieldAnalyzer](#perfieldanalyzer) を参照）。

### KeywordAnalyzer

入力全体を単一のトークンとして扱います。トークン化や正規化は行いません。

```rust
use laurus::analysis::analyzer::keyword::KeywordAnalyzer;

let analyzer = KeywordAnalyzer::new();
// "Hello World" → ["Hello World"]
```

完全一致が必要なフィールド（カテゴリ、タグ、ステータスコード）に使用してください。

### SimpleAnalyzer

フィルタリングなしでテキストをトークン化します。元の大文字小文字とすべてのトークンが保持されます。解析パイプラインを完全に制御したい場合や、Tokenizer を単独でテストしたい場合に便利です。

パイプライン: ユーザー指定の `Tokenizer` のみ（Char Filter なし、Token Filter なし）

```rust
use laurus::analysis::analyzer::simple::SimpleAnalyzer;
use laurus::analysis::tokenizer::regex::RegexTokenizer;
use std::sync::Arc;

let tokenizer = Arc::new(RegexTokenizer::new()?);
let analyzer = SimpleAnalyzer::new(tokenizer);
// "Hello World" → ["Hello", "World"]
// (no lowercasing, no stop word removal)
```

Tokenizer のテストや、別のステップで手動で Token Filter を適用したい場合に使用してください。

### EnglishAnalyzer

英語に特化した Analyzer です。トークン化、小文字化、一般的な英語ストップワードの除去を行います。

パイプライン: `RegexTokenizer`（Unicode 単語境界） → `LowercaseFilter` → `StopFilter`（128 個の一般的な英語ストップワード）

```rust
use laurus::analysis::analyzer::language::english::EnglishAnalyzer;

let analyzer = EnglishAnalyzer::new()?;
// "The Quick Brown Fox" → ["quick", "brown", "fox"]
// ("The" is removed by stop word filtering, remaining tokens are lowercased)
```

### PipelineAnalyzer

任意の Char Filter、Tokenizer、Token Filter のシーケンスを組み合わせてカスタムパイプラインを構築します。

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

`PerFieldAnalyzer` を使用すると、同一 Engine 内で異なるフィールドに異なる Analyzer を割り当てることができます。

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

> **注意:** `_id` フィールドは設定に関係なく、常に `KeywordAnalyzer` で解析されます。

### Schema からの per-field analyzer 設定

実装で直接 `PerFieldAnalyzer` を組み立てる代わりに、スキーマ宣言で analyzer を割り当てる場合がほとんどです。テキストフィールドの `analyzer` 設定は次の 2 つの形式を受け付けます。

```jsonc
// 1. パラメータ不要の組込 analyzer、または schema.analyzers に登録した名前。
{ "analyzer": "standard" }
{ "analyzer": "english" }
{ "analyzer": "my_custom_pipeline" }

// 2. パラメータ付きの組込プリセット。現状は Japanese プリセットのみで、Lindera 辞書のパスが必須。
{
  "analyzer": {
    "language": "japanese",
    "mode": "normal",
    "dict": "/var/lib/lindera/ipadic"
  }
}
```

文字列単独の `"japanese"` は辞書パスを伴わないためエラーとなります。既存スキーマで `"analyzer": "japanese"` を保存していた場合は、上記の構造化形式に移行してください。

プリセットに収まらないパイプラインを使いたい場合は、`schema.analyzers` に `AnalyzerDefinition` として登録し、フィールドからは名前で参照します。

## Char Filter

Char Filter は Tokenizer に渡される**前の**生入力テキストに対して動作します。Unicode 正規化、文字マッピング、パターンベースの置換などの文字レベルの正規化を行います。これにより、Tokenizer がクリーンで正規化されたテキストを受け取ることが保証されます。

すべての Char Filter は `CharFilter` トレイトを実装します。

```rust
pub trait CharFilter: Send + Sync {
    fn filter(&self, input: &str) -> (String, Vec<Transformation>);
    fn name(&self) -> &'static str;
}
```

`Transformation` レコードは文字位置がどのようにシフトしたかを記述し、Engine がトークン位置を元テキストにマッピングできるようにします。

| Char Filter | 説明 |
| :--- | :--- |
| `UnicodeNormalizationCharFilter` | Unicode 正規化（NFC、NFD、NFKC、NFKD） |
| `MappingCharFilter` | マッピング辞書に基づいて文字シーケンスを置換 |
| `PatternReplaceCharFilter` | 正規表現パターンに一致する文字を置換 |
| `JapaneseIterationMarkCharFilter` | 日本語の踊り字を基本文字に展開 |

### UnicodeNormalizationCharFilter

入力テキストに Unicode 正規化を適用します。検索用途では NFKC が推奨されます。互換文字と合成形式の両方を正規化するためです。

```rust
use laurus::analysis::char_filter::unicode_normalize::{
    NormalizationForm, UnicodeNormalizationCharFilter,
};

let filter = UnicodeNormalizationCharFilter::new(NormalizationForm::NFKC);
// "Ｓｏｎｙ" (fullwidth) → "Sony" (halfwidth)
// "㌂" → "アンペア"
```

| 形式 | 説明 |
| :--- | :--- |
| NFC | 正準分解後に正準合成 |
| NFD | 正準分解 |
| NFKC | 互換分解後に正準合成 |
| NFKD | 互換分解 |

### MappingCharFilter

辞書を使用して文字シーケンスを置換します。Aho-Corasick アルゴリズム（最左最長一致）によりマッチングが行われます。

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

正規表現パターンのすべての出現箇所を固定文字列で置換します。

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

日本語の踊り字を基本文字に展開します。漢字（`々`）、ひらがな（`ゝ`、`ゞ`）、カタカナ（`ヽ`、`ヾ`）の踊り字をサポートします。

```rust
use laurus::analysis::char_filter::japanese_iteration_mark::JapaneseIterationMarkCharFilter;

let filter = JapaneseIterationMarkCharFilter::new(
    true,  // normalize kanji iteration marks
    true,  // normalize kana iteration marks
);
// "佐々木" → "佐佐木"
// "いすゞ" → "いすず"
```

### パイプラインでの Char Filter の使用

`PipelineAnalyzer` に `add_char_filter()` で Char Filter を追加します。複数の Char Filter は追加された順序で適用され、すべて Tokenizer の実行前に処理されます。

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

## Tokenizer

| Tokenizer | 説明 |
| :--- | :--- |
| `RegexTokenizer` | Unicode 単語境界で分割。空白と句読点で区切る |
| `UnicodeWordTokenizer` | Unicode 単語境界で分割 |
| `WhitespaceTokenizer` | 空白のみで分割 |
| `WholeTokenizer` | 入力全体を単一のトークンとして返す |
| `LinderaTokenizer` | 日本語形態素解析（Lindera/MeCab） |
| `NgramTokenizer` | 設定可能なサイズの n-gram トークンを生成 |

## Token Filter

| フィルタ | 説明 |
| :--- | :--- |
| `LowercaseFilter` | トークンを小文字に変換 |
| `StopFilter` | 一般的な単語を除去（"the"、"is"、"a"） |
| `StemFilter` | 単語を語幹に縮約（"running" → "run"） |
| `SynonymGraphFilter` | 同義語辞書でトークンを展開 |
| `BoostFilter` | トークンのブースト値を調整 |
| `LimitFilter` | トークン数を制限 |
| `StripFilter` | トークンの先頭/末尾の空白を除去 |
| `FlattenGraphFilter` | トークングラフをフラット化。インデックス時は自動でフラット化されるため、クエリのパースにも使うアナライザーには入れない（[トークングラフ](#トークングラフ)を参照） |
| `RemoveEmptyFilter` | 空トークンを除去 |

### 同義語展開

`SynonymGraphFilter` は同義語辞書を使用してタームを展開します。

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

`boost` パラメータは、元のトークンに対する同義語の重みを制御します。値 `0.8` は、同義語のマッチが完全一致のスコアの 80% を寄与することを意味します。

#### 複数語の見出し語の一致

複数語の見出し語は、連続するトークンがすべて英数字（トークン種別 `Alphanum` または `Num`）のとき、またはオフセットが連続している（各トークンの `end_offset` が次のトークンの `start_offset` と等しい）ときに一致します。前者により空白で区切られた英単語が一致し、後者により形態素トークナイザが分割した CJK の語が一致します。種別は `Token::metadata` から読むため、種別のないトークンはオフセットが連続している必要があります。辞書に `東京大学` があるとき、`東京` と `大学` に分割されたテキスト `東京大学` は一致しますが、`東京 大学` は 2 つのトークンの間に空白があるため一致しません。

#### トークングラフ

このフィルタは、同義語を展開元の語と同じ位置に積みます（`position_increment = 0`）。そのため出力はグラフになり、トークンは「位置」から「位置 + `position_length`」への弧（arc）になります。一致した語とすべての同義語は、同じ開始ノードから同じ終了ノードまで続きます。短い代替語の最後のトークンが残りの位置をまたぎ、一致区間の次のトークンは終了ノードから始まります。`ml` と `machine learning` を同義語とした `ml tutorial` の出力は次のとおりです。

| トークン | 位置 | `position_increment` | `position_length` |
| :--- | :--- | :--- | :--- |
| `ml` | 0 | 1 | 2 |
| `machine` | 0 | 0 | 1 |
| `learning` | 1 | 1 | 1 |
| `tutorial` | 2 | 1 | 1 |

複数語のメンバーは、それぞれ専用の内側のノードを持ちます。そのため、2 つのメンバーの語が同じ弧に入ることはありません。`ml` と `machine learning`、`statistical machine learning` を同義語とした `ml` の出力は次のとおりです。

| トークン | 弧 |
| :--- | :--- |
| `ml` | 0 → 4 |
| `machine` | 0 → 1 |
| `statistical` | 0 → 2 |
| `learning` | 1 → 4 |
| `machine` | 2 → 3 |
| `learning` | 3 → 4 |

インデックスは `position_length` を保存しないため、グラフを最長の経路に沿って並べ直してから保存します。このとき、メンバーの語は位置を共有します。上の例では、`ml`・`machine`・`statistical` が位置 0、`learning`・`machine` が位置 1、`learning` が位置 2 に保存されます。積まれたトークンは展開元の語と同じ位置に保存されます。それ以外の位置は詰めて振られます。`StopFilter` が取り除いたストップワードの位置は空きません。この並べ直しはインデックスが行うため、アナライザーに `FlattenGraphFilter` は不要です。エンジンはクエリも同じアナライザーでパースするので、`FlattenGraphFilter` を入れると、引用符で囲んだ値をマッチさせるグラフまでフラット化されてしまいます。

#### 同義語を使った検索

エンジンは、フィールドのアナライザーをインデックスとクエリのパースの両方に使います。そのため、同義語は両側で展開されます。引用符なしの語は、その同義語のどれにでもマッチします。引用符で囲んだ値はグラフに沿ってマッチします（[フレーズクエリ](query_dsl.md#フレーズクエリ)を参照）。`big` と `large` を同義語とすると、`"big"` はどちらの語にもマッチし、`"a big dog"` は「a large dog」にもマッチします。グループの各メンバーはそれぞれ 1 つのフレーズになります。上の例のメンバーなら、`"ml"` は `ml`、`machine learning`、`statistical machine learning` にマッチし、「statistical learning」にはマッチしません。同じ長さのメンバーも別々のフレーズになり、引用符で囲んだ値 1 つあたり 64 フレーズの上限にそれぞれ数えられます。

インデックスは位置を保存しますが `position_length` は保存しないため、複数語の同義語をインデックス時に展開すると、Lucene と同じく次のようになります。

- フレーズが同義語の途中から始まったり、途中で終わったりしてもマッチします。`"learning is"` は「ml is fun」にマッチします。同義語を通すと「machine learning is fun」と読めるためです。
- 1 つのグループに複数語のメンバーが複数あると、インデックスではそれらの語が位置を共有します。そのため、1 つのメンバーを含む文書が、2 つのメンバーの語が混ざったフレーズにもマッチします。「ml is fun」は `"statistical learning"` にマッチします。

そのほかの注意点は次のとおりです。

- `keep_original = false` は、一致した語をグループのほかのメンバーで置き換えます。そのため、その語自体はインデックスも検索もされません。検索には `keep_original = true` を使ってください。
- `StopFilter` や `LimitFilter` のようにトークンを取り除くフィルターは、`SynonymGraphFilter` より前に置いてください。後ろに置くと、グラフに必要なトークンを取り除いてしまいます。
- Issue #1252 より前にインデックスした文書は、積まれたトークンに連続した位置を振っています。フィールドが `SynonymGraphFilter` を使う場合は、アップグレード後に文書を投入し直してください。セグメントのマージは保存済みの位置をそのままコピーするため、マージでは直りません。
