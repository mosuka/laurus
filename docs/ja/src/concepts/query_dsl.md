# Query DSL

Laurus は統合 Query DSL（Domain Specific Language）を提供しており、Lexical（キーワード）検索と Vector（意味的）検索を単一のクエリ文字列で記述できます。`UnifiedQueryParser` は入力を Lexical 部分と Vector 部分に分割し、適切なサブパーサーに委譲します。

## 概要

```text
title:hello AND content:"cute kitten"^0.8
|--- lexical --|    |--- vector --------|
```

Vector 句と Lexical 句の区別は、フィールド名に基づいて行われます。スキーマ上でベクトルフィールドとして定義されたフィールド名が指定された場合、その句は Vector クエリとして扱われます。

### フィールド検証

クエリ中の `field:value` 句は、パース時にスキーマと照合されます。スキーマに**宣言されていない**フィールドを参照したクエリは、結果が黙って空になるのではなく、エラーとして返されます。これにより typo（例: `title:hello` のつもりで `titl:hello`）を早期に検出できます。

未知のフィールドを含むドキュメントを受け付けたい場合は、スキーマの [`dynamic_field_policy`](./schema_and_fields.md#動的スキーマ) を設定して、投入時にフィールドを追加する動作を有効にしてください。一度フィールドがスキーマに登録されれば、それを参照するクエリは正常に動作します。

## Lexical クエリ構文

Lexical クエリは、完全一致または近似のキーワードマッチングを使用して転置インデックスを検索します。

### Term クエリ

フィールド（またはデフォルトフィールド）に対して単一のタームをマッチングします。

```text
hello
title:hello
```

フィールドのアナライザーが引用符なしの語を複数のトークンに分割する場合
（例: 形態素解析（Lindera）アナライザーを通した日本語の文、あるいは
デフォルトの `\w+` トークナイザーを通した `rust-lang` のようなハイフン
入り単語）、それらのトークンは正確な語順のフレーズとしてではなく
`OR` で結合されます（Lucene の `match` クエリと同様）。厳密な隣接一致
が必要な場合は、引用符付きの[フレーズクエリ](#フレーズクエリ)を使用してください。

### ブーリアン演算子

`AND` と `OR`（大文字小文字を区別しない）で句を結合します。

```text
title:hello AND body:world
title:hello OR title:goodbye
```

`AND` は対称的に動作します。両側の句を必須（`Must`）にマークするため、`title:hello AND body:world` は **両方** の句にマッチするドキュメントだけを返します。連鎖（`a AND b AND c`）でも 3 つすべてが必須となります。ただし `+`（必須）または `-`（禁止）が明示されている句は元の意図が維持され、`AND` で上書きされることはありません。

明示的な演算子なしでスペース区切りされた句は、暗黙的なブーリアン（スコアリング付きの OR として動作）を使用します。例えば `a b AND c` は「`a` は任意、`b` と `c` の両方が必須」と解釈されます。

### 必須 / 禁止句

`+`（必ずマッチ）と `-`（マッチ禁止）を使用します。

```text
+title:hello -title:goodbye
```

### フレーズクエリ

ダブルクォートを使用して正確なフレーズをマッチングします。オプションの近接度（`~N`）で、ターム間に N 語を許可します。

```text
"hello world"
"hello world"~2
```

### ファジークエリ

編集距離を使用した近似マッチング。`~` に続けてオプションで最大編集距離を指定します。

```text
roam~
roam~2
```

### ワイルドカードクエリ

`?`（1 文字）と `*`（0 文字以上）を使用します。

```text
te?t
test*
```

### 範囲クエリ

包含的な `[]` または排他的な `{}` の範囲指定。数値フィールドや日付フィールドに有用です。

```text
price:[100 TO 500]
date:{2024-01-01 TO 2024-12-31}
price:[* TO 100]
```

#### 日付・日時の境界

境界が日時リテラルである範囲は `DateTimeRangeQuery` を構築し、数値範囲と同様に定数スコアでフィールドの BKD tree に対して評価されます。数値でない境界は、次の順に解釈を試みます。

| 形式 | 例 | 解釈 |
| :--- | :--- | :--- |
| RFC 3339 | `2024-01-01T00:00:00Z`、`2024-01-01T09:00:00.5+09:00` | 小数秒可。オフセット付きは UTC に正規化 |
| オフセットなし日時 | `2024-01-01T09:00:00`、`2024-01-01T09:00:00.5` | `YYYY-MM-DDTHH:MM:SS[.fff]`。UTC として解釈 |
| 日付のみ | `2024-01-01` | `YYYY-MM-DD`。その日の `00:00:00` UTC（Lucene / Elasticsearch のデフォルト） |

```text
# 日付のみの境界（上限は 2024-12-31T00:00:00Z。下記参照）
created_at:[2024-01-01 TO 2024-12-31]

# オフセット付き RFC 3339（09:00+09:00 は 00:00Z）
created_at:[2024-06-15T09:00:00+09:00 TO 2024-06-16T09:00:00+09:00]

# オフセットなし日時は UTC。* で上限を開放
created_at:[2024-06-15T12:34:56 TO *]

# 両端を含まない
created_at:{2024-01-01 TO 2025-01-01}
```

- 日付のみの境界はその日の 0 時（UTC）を意味するため、`[2024-01-01 TO 2024-12-31]` は `2024-12-31T00:00:00Z` を含み、同日のそれ以降は含みません。年全体を対象にするには、上限に時刻を付ける（`[2024-01-01 TO 2024-12-31T23:59:59.999999]`）か、翌日を上限とする排他的範囲（`{2024-01-01 TO 2025-01-01}`。`{}` は下限の瞬間 `2024-01-01T00:00:00Z` も除外する点に注意）を使います。
- `*` はその側を開放します。
- 数値のみの境界は従来どおり数値範囲です。`DateTime` フィールドではエポック秒として扱われ、`NumericRangeQuery` を構築します。
- 数値の境界と日時の境界の混在（`[1704067200 TO 2024-12-31]`）や、日時の境界と日時リテラルでない文字列の混在（`[2024-01-01 TO tomorrow]`）は、黙って空の結果を返すのではなくパースエラーになります。数値でも日時リテラルでもない境界の範囲（`title:[A TO Z]`）は従来どおりパースは通りますが、何にもマッチしません。

> ドキュメント投入時のテキストから `DateTime` フィールドへの型変換は RFC 3339 のみを
> 受け付けます。オフセットなし日時と日付のみの形式はクエリリテラル専用です。

### 2D 地理クエリ（`geo_*`)

2 種類の関数形式を `Geo`（2D 緯度 / 経度）フィールドに対して使えます。
緯度・経度は度単位、距離はメートル単位の符号付き浮動小数点数：

```text
location:geo_distance(lat, lon, distance_m)
location:geo_bbox(min_lat, min_lon, max_lat, max_lon)
```

| 形式 | 動作 |
| :--- | :--- |
| `geo_distance(lat, lon, distance_m)` | 中心 `(lat, lon)` から `distance_m` メートル以内の格納済み座標を返す |
| `geo_bbox(min_lat, min_lon, max_lat, max_lon)` | 軸並行緯度・経度範囲に含まれる格納済み座標を返す |

例：

```text
# 東京から 10 km（= 10 000 m）以内（35.6895, 139.6917）
location:geo_distance(35.6895, 139.6917, 10000)

# 軸並行緯度・経度範囲
location:geo_bbox(35.0, 139.0, 36.0, 140.0)
```

> クエリ対象フィールドはスキーマで `Geo` フィールドとして宣言されている必要が
> あります。緯度は `[-90, 90]`、経度は `[-180, 180]` の範囲外を拒否します。

### 3D 地理クエリ（`geo3d_*`）

3 種類の関数形式を `Geo3d`（3D ECEF 直交座標）フィールドに対して使えます。
`k` 以外の引数はすべてメートル単位の符号付き浮動小数点数、`k` のみ符号なし整数：

```text
position:geo3d_distance(x, y, z, distance_m)
position:geo3d_bbox(min_x, min_y, min_z, max_x, max_y, max_z)
position:geo3d_nearest(x, y, z, k)
```

| 形式 | 動作 |
| :--- | :--- |
| `geo3d_distance(x, y, z, distance_m)` | `(x, y, z)` から `distance_m` メートル以内の格納済みポイントを返す |
| `geo3d_bbox(min_x, min_y, min_z, max_x, max_y, max_z)` | 軸並行 3D ボックスに含まれる格納済みポイントを返す |
| `geo3d_nearest(x, y, z, k)` | `(x, y, z)` に最も近い `k` 件をユークリッド距離順で返す |

例：

```text
# 東京タワーから 5 km 以内（ECEF 座標）
position:geo3d_distance(-3955182, 3350553, 3700276, 5000)

# 軸並行 ECEF バウンディングボックス内
position:geo3d_bbox(-4000000, 3300000, 3650000, -3900000, 3400000, 3750000)

# 最も近い 10 件
position:geo3d_nearest(-3955182, 3350553, 3700276, 10)
```

> クエリ対象フィールドはスキーマで `Geo3d` として宣言されている必要があります。
> 座標系・WGS84 変換ヘルパー・詳細な意味論については
> [3D 地理検索](geo3d.md) を参照してください。

### ブースト

`^` で句のウェイトを増加させます。

```text
title:hello^2
"important phrase"^1.5
```

### グルーピング

括弧でサブ式を囲みます。

```text
(title:hello OR title:hi) AND body:world
```

### Lexical PEG 文法

完全な Lexical 文法（[parser.pest](https://github.com/mosuka/laurus/blob/main/laurus/src/lexical/query/parser.pest)）:

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

## Vector クエリ構文

Vector クエリは、解析時にテキストをベクトルにエンベディングし、類似性検索を実行します。

### 基本構文

```text
field:"text"
field:text
field:"text"^weight
```

| 要素 | 必須 | 説明 | 例 |
| :--- | :---: | :--- | :--- |
| `field:` | **はい** | 対象のベクトルフィールド名（スキーマでベクトルフィールドとして定義されている必要があります） | `content:` |
| `"text"` または `text` | **はい** | エンベディングするテキスト（クォート付きまたはクォートなし） | `"cute kitten"`、`python` |
| `^weight` | いいえ | スコアウェイト（デフォルト: 1.0） | `^0.8` |

### Vector クエリの例

```text
# Single field (quoted text)
content:"cute kitten"

# Single field (unquoted text)
content:python

# With boost weight
content:"cute kitten"^0.8

# Multiple clauses
content:"cats" image:"dogs"^0.5

# Nested field name (dot notation)
metadata.embedding:"text"
```

### 複数句

複数の Vector 句はスペースで区切ります。すべての句が実行され、スコアは `score_mode`（デフォルト: `WeightedSum`）を使用して結合されます。

```text
content:"cats" image:"dogs"^0.5
```

この場合のスコア計算:

```text
score = similarity("cats", content) * 1.0
      + similarity("dogs", image)   * 0.5
```

Vector DSL には `AND`/`OR` 演算子はありません。Vector 検索は本質的にランキング操作であり、ウェイト（`^`）が各句の寄与度を制御します。

### スコアモード

| モード | 説明 |
| :--- | :--- |
| `WeightedSum`（デフォルト） | すべてのクエリ句にわたる（類似度 * ウェイト）の合計 |
| `MaxSim` | クエリ句間の最大類似度スコア |
| `LateInteraction` | Late Interaction スコアリング |

スコアモードは DSL 構文からは設定できません。Rust API を使用してオーバーライドします。

```rust
let mut request = parser.parse(r#"content:"cats" image:"dogs""#).await?;
request.score_mode = VectorScoreMode::MaxSim;
```

### Vector PEG 文法

完全な Vector 文法（[parser.pest](https://github.com/mosuka/laurus/blob/main/laurus/src/vector/query/parser.pest)）:

```pest
query          = { SOI ~ vector_clause+ ~ EOI }
vector_clause  = { field_prefix ~ (quoted_text | unquoted_text) ~ boost? }
field_prefix   = { field_name ~ ":" }
field_name     = @{ (ASCII_ALPHA | "_") ~ (ASCII_ALPHANUMERIC | "_" | ".")* }
quoted_text    = ${ "\"" ~ inner_text ~ "\"" }
unquoted_text  = @{ (!(WHITE_SPACE | "^" | "\"") ~ ANY)+ }
inner_text     = @{ (!("\"") ~ ANY)* }
boost          = { "^" ~ float_value }
float_value    = @{ ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT+)? }
```

## 統合（ハイブリッド）クエリ構文

`UnifiedQueryParser` を使用すると、単一のクエリ文字列内で Lexical 句と Vector 句を自由に混在させることができます。

```text
title:hello content:"cute kitten"^0.8
```

### 仕組み

1. **分割（Split）**: スキーマのフィールド型に基づいて、各句が Lexical か Vector かを判定する。ベクトルフィールドとして定義されたフィールド名を持つ句は Vector 句として抽出される
2. **委譲（Delegate）**: Vector 部分は `VectorQueryParser` に、残りは Lexical の `QueryParser` に渡される
3. **フュージョン（Fuse）**: Lexical と Vector の両方の結果が存在する場合、フュージョンアルゴリズムで結合される

### 曖昧性の解消

Vector 句と Lexical 句の区別は、スキーマのフィールド型に基づいて行われます。フィールド名がスキーマ上でベクトルフィールド（HNSW、Flat、IVF など）として定義されている場合、その句は Vector クエリとして扱われます。Lexical 構文の `~`（例: `roam~2`、`"hello world"~10`）はファジークエリや近接度クエリとして引き続き正しく解析されます。

### フュージョンアルゴリズム

クエリに Lexical 句と Vector 句の両方が含まれる場合、結果はフュージョンされます。

| アルゴリズム | 計算式 | 説明 |
| :--- | :--- | :--- |
| **RRF**（デフォルト） | `score = sum(1 / (k + rank))` | Reciprocal Rank Fusion。異なるスコア分布に対してロバスト。デフォルト k=60。 |
| **WeightedSum** | `score = lexical * a + vector * b` | 設定可能なウェイトによる線形結合。 |

> **注意**: フュージョンアルゴリズムは DSL 構文では指定できません。`UnifiedQueryParser` の構築時に `.with_fusion()` で設定します。デフォルトは RRF（k=60）です。コード例は[カスタムフュージョン](#カスタムフュージョン)を参照してください。

### ハイブリッド AND/OR セマンティクス（`+` プレフィックス）

デフォルトでは、ハイブリッドクエリは **union（OR）** を使用します。Lexical 結果または Vector 結果の**いずれか**に含まれるドキュメントが返されます。Vector 句に `+` プレフィックスを付けると **intersection（AND）** に切り替わり、**両方**の結果セットに存在するドキュメントのみが返されます。

| 構文 | モード | 動作 |
| :--- | :---: | :--- |
| `title:Rust content:"system process"` | OR（union） | Lexical クエリ**または** Vector クエリにマッチするドキュメントが返される |
| `title:Rust +content:"system process"` | AND（intersection） | Lexical と Vector の**両方**にマッチするドキュメントのみが返される |
| `+title:Rust +content:"system process"` | AND（intersection） | 両方の句が必須。Lexical フィールドの `+` は既存の required clause として処理される |

ルール:

- Vector 句に `+` プレフィックスが**ない**場合、フュージョンは Lexical と Vector の結果を **union（OR）** で結合します
- **1 つでも** Vector 句に `+` プレフィックスがある場合、フュージョンは **intersection（AND）** に切り替わり、Lexical と Vector の両方の結果セットに存在するドキュメントのみが返されます
- Lexical フィールドの `+`（例: `+title:Rust`）は、Lexical クエリパーサーによって *required clause*（必須句）として解釈されます。これは既存の Tantivy/Lucene スタイルの動作であり、それ自体ではハイブリッドフュージョンの intersection モードをトリガーしません

### 統合クエリの例

```text
# Lexical only — no fusion
title:hello AND body:world

# Vector only — no fusion
content:"cute kitten"

# Vector only — unquoted text
content:python

# Hybrid — fusion applied automatically (OR / union)
title:hello content:"cute kitten"

# Hybrid with AND / intersection — both result sets required
title:hello +content:"cute kitten"

# Hybrid with boolean operators
title:hello AND category:animal content:"cute kitten"^0.8

# Multiple vector clauses + lexical
category:animal content:"cats" image:"dogs"^0.5
```

## コード例

### DSL による Lexical 検索

```rust
use std::sync::Arc;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::query::QueryParser;

let analyzer = Arc::new(StandardAnalyzer::new()?);
let parser = QueryParser::new(analyzer)
    .with_default_field("title");

let query = parser.parse("title:hello AND body:world")?;
```

### DSL による Vector 検索

```rust
use std::sync::Arc;
use laurus::vector::query::VectorQueryParser;

let parser = VectorQueryParser::new(embedder)
    .with_default_field("content");

let request = parser.parse(r#"content:"cute kitten"^0.8"#).await?;
```

### 統合 DSL によるハイブリッド検索

```rust
use laurus::engine::query::UnifiedQueryParser;

let unified = UnifiedQueryParser::new(lexical_parser, vector_parser);

let request = unified.parse(
    r#"title:hello content:"cute kitten"^0.8"#
).await?;
// request.query  -> SearchQuery::Hybrid { lexical: ..., vector: ... }
// request.fusion_algorithm  -> Some(RRF)  — fusion algorithm
```

### カスタムフュージョン

```rust
use laurus::engine::search::FusionAlgorithm;

let unified = UnifiedQueryParser::new(lexical_parser, vector_parser)
    .with_fusion(FusionAlgorithm::WeightedSum {
        lexical_weight: 0.3,
        vector_weight: 0.7,
    });
```
