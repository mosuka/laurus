# スキーマフォーマットリファレンス

スキーマファイルはインデックスの構造を定義します。どのフィールドが存在するか、その型、およびインデックスの方法を指定します。Laurus はスキーマファイルに TOML 形式を使用します。

## 概要

スキーマは 5 つのトップレベル要素で構成されます:

```toml
# スキーマに宣言されていないフィールドの扱い。省略時は "dynamic"。
dynamic_field_policy = "dynamic"

# クエリでフィールドが指定されていない場合にデフォルトで検索するフィールド。
default_fields = ["title", "body"]

# カスタムアナライザの定義。Text フィールドから名前で参照する。省略可能。
[analyzers.<analyzer_name>]
# ... tokenizer、char_filters、token_filters

# エンベダーの定義。ベクトルフィールドから名前で参照する。省略可能。
[embedders.<embedder_name>]
# ... type と型固有のオプション

# フィールド定義。各フィールドには名前と型付き設定があります。
[fields.<field_name>.<FieldType>]
# ... 型固有のオプション
```

- **`dynamic_field_policy`** — スキーマに**宣言されていない**フィールドがドキュメントに含まれる場合の挙動を制御します。値は `"strict"` / `"dynamic"` / `"ignore"`。デフォルトは `"dynamic"`。詳細および「`dynamic` では情報損失が起きうる」という警告は [動的スキーマ](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。
- **`default_fields`** — [Query DSL](../concepts/query_dsl.md) でデフォルトの検索対象として使用されるフィールド名のリストです。Lexical フィールド（Text、Integer、Float など）のみデフォルトフィールドに指定できます。このキーはオプションで、デフォルトは空のリストです。
- **`analyzers`** — 名前とカスタムのテキスト解析パイプラインのマップです。Text フィールドは `analyzer` オプションに名前を書いて使います。省略可能です。[アナライザ](#アナライザ) を参照してください。
- **`embedders`** — 名前と埋め込みモデルのマップです。ベクトルフィールドは `embedder` オプションに名前を書いて使います。省略可能です。[エンベダー](#エンベダー) を参照してください。
- **`fields`** — フィールド名とその型付き設定のマップです。各フィールドにはフィールド型を1つだけ指定する必要があります。

## フィールド命名規則

- フィールド名は任意の文字列です（例: `title`、`body_vec`、`created_at`）。
- **アンダースコア（`_`）で始まるフィールド名はエンジンの予約領域**です。例外として `_id`（自動管理）のみ許可されます。それ以外の `_` プレフィックス名はインデックス作成時に拒否され、`Field name '_score' is reserved: names starting with '_' are reserved for system fields (allowed: '_id')` というエラーになります。このとき `create index` は何も作りません。この検査より前に作成したインデックスは引き続き開けますが、そのフィールドは使えないままです — 投入時には今までどおり拒否されます。
- フィールド名はスキーマ内で一意である必要があります。

## フィールド型

フィールドは **Lexical**（キーワード/全文検索用）と **Vector**（類似検索用）の2つのカテゴリに分類されます。1つのフィールドが両方を兼ねることはできません。

### Lexical フィールド

#### Text

全文検索可能なフィールドです。テキストは解析パイプライン（トークン化、正規化、ステミングなど）によって処理されます。

```toml
[fields.title.Text]
indexed = true               # このフィールドを検索用にインデックスするかどうか
stored = true                # 取得用に元の値を保存するかどうか
multi_valued = false         # 文字列の配列を受け付けるかどうか（Issue #1175）
position_increment_gap = 100 # 多値フィールドの要素間で読み飛ばす位置数
term_vectors = true          # タームの位置を保存するかどうか（フレーズクエリ・スパンクエリ用）
doc_values = true            # 値を DocValues にもコピーするかどうか（ソート・ファセット用）
analyzer = "standard"        # このフィールドのインデックス時とクエリ解析時に使うアナライザ
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | このフィールドの検索を有効にする |
| `stored` | `bool` | `true` | 結果に返せるよう元の値を保存する |
| `multi_valued` | `bool` | `false` | 文字列の配列を受け付け、term クエリは**いずれかの要素**がタームを含めばマッチ（Lucene 流の "any match"）。フレーズクエリは slop が `position_increment_gap` に達しない限り 2 つの要素をまたがない |
| `position_increment_gap` | `integer` | `100` | 多値フィールドの要素間で読み飛ばす位置数（Lucene の `positionIncrementGap`）。`0` にすると要素を連結したものとして付番する。`multi_valued = true` でなければ無視される |
| `term_vectors` | `bool` | `true` | フレーズクエリ・スパンクエリが読み取るタームの位置を保存する。ハイライトは常に保存済みテキストを再トークナイズするため使用しない |
| `doc_values` | `bool` | `true` | 値を DocValues（[ソート](../laurus/faceting.md)・ファセット・集計が読み取る列指向ストア）にもコピーする。`stored` も `true` の場合のみ有効 —— 詳細は後述の [共通オプション: `doc_values`](#共通オプション-doc_values) を参照 |
| `analyzer` | `string` またはテーブル | *（省略）* | このフィールドのインデックス時とクエリ解析時の両方で使うアナライザ。文字列は組み込みアナライザ（`"standard"`、`"english"`、`"keyword"`、`"simple"`、`"noop"`）か [`[analyzers.*]`](#アナライザ) の名前を指す。テーブルはパラメータ付きの組み込みプリセットを選び、現在は `{ language = "japanese", mode = "normal", dict = "<path>" }` のみ（[テキスト解析](../concepts/analysis.md#schema-からの-per-field-analyzer-設定) を参照）。省略すると `"standard"` を使う |

対話的なスキーマジェネレータ（`laurus create schema`。[スキーマの生成](#スキーマの生成) を参照）は、Text フィールドについて多値にするかどうかを尋ね、多値にする場合は position increment gap も尋ねます。

#### Integer

64ビット符号付き整数フィールド。範囲クエリと完全一致をサポートします。

```toml
[fields.year.Integer]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | 範囲クエリおよび完全一致クエリを有効にする |
| `stored` | `bool` | `true` | 元の値を保存する |
| `multi_valued` | `bool` | `false` | 整数の配列を受け付け、範囲クエリは**いずれかの値**が条件を満たせばマッチ（Lucene 流の "any match"、constant スコア） |
| `doc_values` | `bool` | `true` | 詳細は後述の [共通オプション: `doc_values`](#共通オプション-doc_values) を参照 |

#### Float

64ビット浮動小数点フィールド。範囲クエリをサポートします。

```toml
[fields.rating.Float]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | 範囲クエリを有効にする |
| `stored` | `bool` | `true` | 元の値を保存する |
| `multi_valued` | `bool` | `false` | 浮動小数点の配列を受け付け、範囲クエリは**いずれかの値**が条件を満たせばマッチ（Lucene 流の "any match"、constant スコア） |
| `doc_values` | `bool` | `true` | 詳細は後述の [共通オプション: `doc_values`](#共通オプション-doc_values) を参照 |

#### Boolean

ブーリアンフィールド（`true` / `false`）。

```toml
[fields.published.Boolean]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | ブーリアン値によるフィルタリングを有効にする |
| `stored` | `bool` | `true` | 元の値を保存する |
| `multi_valued` | `bool` | `false` | ブール値の配列を受け付け、term クエリ（`flags:true`）は**いずれかの要素**がクエリの値と等しければマッチ（Lucene 流の "any match"）。要素の重複はヒット数ではなく term frequency を増やす |
| `doc_values` | `bool` | `true` | 詳細は後述の [共通オプション: `doc_values`](#共通オプション-doc_values) を参照 |

#### DateTime

UTC タイムスタンプフィールド。範囲クエリをサポートします。

```toml
[fields.created_at.DateTime]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | 日時の範囲クエリを有効にする |
| `stored` | `bool` | `true` | 元の値を保存する |
| `multi_valued` | `bool` | `false` | 時刻の配列を受け付け、範囲クエリは**いずれかの時刻**が条件を満たせばマッチ（Lucene 流の "any match"） |
| `doc_values` | `bool` | `true` | 詳細は後述の [共通オプション: `doc_values`](#共通オプション-doc_values) を参照 |

#### Geo

地理座標フィールド（緯度/経度）。半径クエリおよびバウンディングボックスクエリをサポートします。

```toml
[fields.location.Geo]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | Geo クエリ（半径、バウンディングボックス）を有効にする |
| `stored` | `bool` | `true` | 元の値を保存する |
| `multi_valued` | `bool` | `false` | ポイントの配列を受け付け、距離 / バウンディングボックスクエリは**いずれかのポイント**が条件を満たせばマッチ（Lucene 流の "any match"）。スコアはドキュメント内で最も近いポイントで決まる |
| `doc_values` | `bool` | `true` | 詳細は後述の [共通オプション: `doc_values`](#共通オプション-doc_values) を参照 |

#### Geo3d

3D Earth-Centered Earth-Fixed (ECEF) 直交座標系の点フィールド（x / y / z はメートル単位）。`geo3d_distance`（球）、`geo3d_bbox`（3D AABB）、`geo3d_nearest`（k-NN）クエリをサポートします。座標系および `wgs84_to_ecef` / `ecef_to_wgs84` の変換ユーティリティについては [3D 地理検索 (ECEF)](../concepts/geo3d.md) を参照してください。

```toml
[fields.position.Geo3d]
indexed = true
stored = true
multi_valued = false
doc_values = true
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `indexed` | `bool` | `true` | 3D 地理クエリ（`geo3d_distance`、`geo3d_bbox`、`geo3d_nearest`）を有効にする |
| `stored` | `bool` | `true` | 元の `(x, y, z)` 値を保存する |
| `multi_valued` | `bool` | `false` | ポイントの配列を受け付け、`geo3d_distance` / `geo3d_bbox` / `geo3d_nearest` クエリは**いずれかのポイント**が条件を満たせばマッチ（Lucene 流の "any match"）。スコアはドキュメント内で最も近いポイントで決まる |
| `doc_values` | `bool` | `true` | 詳細は後述の [共通オプション: `doc_values`](#共通オプション-doc_values) を参照 |

#### Bytes

生バイナリデータフィールド。インデックスされず、保存のみです。

```toml
[fields.thumbnail.Bytes]
stored = true
multi_valued = false
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `stored` | `bool` | `true` | バイナリデータを保存する |
| `multi_valued` | `bool` | `false` | バイト列の配列を受け付ける。`Bytes` フィールドはそもそもインデックスされないため、他の `multi_valued` オプションと異なり "any match" のクエリ意味論は存在せず、保存時の形と取り込み時の許容個数を変えるだけ |

`BytesOption` に `doc_values` 設定はありません。`Bytes` の値はソートにもファセットにも
使えないため、設定にかかわらず DocValues には一切書き込まれないからです。

#### 共通オプション: `doc_values`

上記の lexical フィールドオプションのうち `BytesOption` を除く全てが `doc_values` オプションを
持ち、値を DocValues ―― [ソート](../laurus/faceting.md)・ファセット・集計が読み取る列指向ストア
―― にもコピーするかどうかを制御します。実効ルールは次のとおりです: DocValues 列が書き込まれる
のは `stored` と `doc_values` の両方が `true` の場合のみです。`doc_values: false` と
`stored: false` の組み合わせは（エラーにせず）黙って無視されます。ソートにもファセットにも
使わないフィールドで `doc_values` を無効にすると、値が二重（stored document と DocValues）
ではなく一度（stored document のみ）しか書き込まれなくなるため、セグメントの使用容量が
削減されます。フィールド自体は引き続き完全に検索・取得可能で、ソートやファセットを行う際は
単に stored document へフォールバックします。

### Vector フィールド

Vector フィールドは近似最近傍探索（ANN: Approximate Nearest Neighbor）用にインデックスされます。`dimension`（各ベクトルの長さ）と `distance` メトリクスの指定が必要です。

#### Hnsw

HNSW（Hierarchical Navigable Small World）グラフインデックス。ほとんどのユースケースに最適で、速度と再現率（Recall）のバランスに優れています。

```toml
[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
m = 16
ef_construction = 200
base_weight = 1.0
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | `128` | ベクトルの次元数（Embedding モデルの出力と一致させる必要あり） |
| `distance` | `string` | `"Cosine"` | 距離メトリクス（[距離メトリクス](#距離メトリクス)を参照） |
| `m` | `integer` | `16` | ノードあたりの最大双方向接続数。大きいほど再現率が向上するがメモリ使用量が増加 |
| `ef_construction` | `integer` | `200` | インデックス構築時の探索幅。大きいほど品質が向上するが構築が遅くなる |
| `base_weight` | `float` | `1.0` | 同時に検索する他の vector フィールドに対する相対的な優先度。ハイブリッド検索の lexical-vs-vector 融合のバランスには影響しない（[ウェイト](../concepts/search/vector_search.md#ウェイト)を参照） |
| `quantizer` | `object` | `"Scalar8Bit"` | 量子化方式（[量子化](#量子化)を参照）。必須。デフォルトは Issue #481 Stage 1 で導入された int8 形式を保つ。 |
| `rerank_storage` | `string` | *（省略）* | Stage 2 rerank sidecar（[Rerank Storage](#rerank-storage)）。`"F32"` でフィールド単位の f32 sidecar を有効化し、検索時に int8 候補を元のベクトルで再スコアできるようにする。省略すると Stage 1 int8-only の挙動を維持。 |
| `pq_codebook_path` | `string` | *（省略）* | 共有 PQ codebook のストレージ相対ファイル名（Issue #631）。`ProductQuantization` quantizer との組み合わせでのみ意味を持つ。`laurus train pq-codebook` で学習すると、以後の commit は segment ごとの k-means 再学習の代わりにこの codebook で encode する。設定済みで未学習の場合、commit は明示的にエラーになる（無言のフォールバック無し）。省略すると segment ごとに学習。 |
| `embedder` | `string` | *（省略）* | [`[embedders.*]`](#エンベダー) のエントリ名。指定すると、このフィールドに与えたテキスト（または画像）をそのモデルでベクトルに変換する。インデックス時と検索時の両方で変換する。省略すると計算済みのベクトルだけを受け付ける |

**チューニングガイドライン:**

- `m`: 12〜48 が一般的です。高次元ベクトルには大きい値を使用してください。
- `ef_construction`: 100〜500。大きい値ほどグラフの品質が向上しますが、構築時間が増加します。
- `dimension`: Embedding モデルの出力次元と正確に一致させる必要があります（例: `all-MiniLM-L6-v2` は 384、`BERT-base` は 768、`text-embedding-3-small` は 1536）。

#### Flat

ブルートフォース線形スキャンインデックス。近似を行わず正確な結果を返します。小規模データセット（10,000 ベクトル未満）に最適です。

```toml
[fields.embedding.Flat]
dimension = 384
distance = "Cosine"
base_weight = 1.0
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | `128` | ベクトルの次元数 |
| `distance` | `string` | `"Cosine"` | 距離メトリクス（[距離メトリクス](#距離メトリクス)を参照） |
| `base_weight` | `float` | `1.0` | 同時に検索する他の vector フィールドに対する相対的な優先度。ハイブリッド検索の lexical-vs-vector 融合のバランスには影響しない（[ウェイト](../concepts/search/vector_search.md#ウェイト)を参照） |
| `quantizer` | `object` | `"Scalar8Bit"` | 量子化方式（[量子化](#量子化)を参照）。必須。デフォルトは Issue #481 Stage 1 で導入された int8 形式を保つ。 |
| `rerank_storage` | `string` | *（省略）* | Stage 2 rerank sidecar（[Rerank Storage](#rerank-storage)）。#932 以降、3 つのベクトルインデックスタイプすべてでサポート。`"F32"` でフィールド単位の f32 sidecar を有効化し、検索時に int8 候補を元のベクトルで再スコアできる。 |
| `embedder` | `string` | *（省略）* | [`[embedders.*]`](#エンベダー) のエントリ名。[Hnsw](#hnsw) を参照 |

#### Ivf

IVF（Inverted File Index）。ベクトルをクラスタリングし、クラスタのサブセットのみを検索します。大規模データセットに適しています。

```toml
[fields.embedding.Ivf]
dimension = 384
distance = "Cosine"
n_clusters = 100
n_probe = 1
base_weight = 1.0
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | *（必須）* | ベクトルの次元数 |
| `distance` | `string` | `"Cosine"` | 距離メトリクス（[距離メトリクス](#距離メトリクス)を参照） |
| `n_clusters` | `integer` | `100` | クラスタ数。多いほど細かい分割が可能 |
| `n_probe` | `integer` | `1` | クエリ時に検索するクラスタ数。大きいほど再現率が向上するが遅くなる |
| `base_weight` | `float` | `1.0` | 同時に検索する他の vector フィールドに対する相対的な優先度。ハイブリッド検索の lexical-vs-vector 融合のバランスには影響しない（[ウェイト](../concepts/search/vector_search.md#ウェイト)を参照） |
| `quantizer` | `object` | `"Scalar8Bit"` | 量子化方式（[量子化](#量子化)を参照）。必須。デフォルトは Issue #481 Stage 1 で導入された int8 形式を保つ。 |
| `rerank_storage` | `string` | *（省略）* | Stage 2 rerank sidecar（[Rerank Storage](#rerank-storage)）。#932 以降、3 つのベクトルインデックスタイプすべてでサポート。`"F32"` でフィールド単位の f32 sidecar を有効化し、検索時に int8 候補を元のベクトルで再スコアできる。 |
| `embedder` | `string` | *（省略）* | [`[embedders.*]`](#エンベダー) のエントリ名。[Hnsw](#hnsw) を参照 |

> **注意:** Hnsw および Flat とは異なり、Ivf の `dimension` フィールドは**必須**であり、デフォルト値はありません。

**チューニングガイドライン:**

- `n_clusters`: 一般的な経験則は `sqrt(N)`（N はベクトルの総数）です。
- `n_probe`: 1 から始めて、再現率が許容範囲になるまで増やしてください。一般的な範囲は 1〜20 です。

#### MultiVector

文書のトークンベクトル（ColBERT 型のトークンごとの埋め込みなど）をすべて保持し、late interaction の再採点に使います。ANN 索引は持たず、ベクトル検索の対象にはなりません。[MultiVector フィールド](../concepts/schema_and_fields.md#multivector-フィールド)を参照してください。

```toml
[fields.body_colbert.MultiVector]
dimension = 128
distance = "Cosine"
```

| オプション | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `dimension` | `integer` | `128` | 各トークンベクトルの次元数 |
| `distance` | `string` | `"Cosine"` | トークン間の類似度。`"Cosine"`（書き込み時に L2 正規化する）または `"DotProduct"`。それ以外は拒否されます |

文書の値は、同じ長さの数値配列の配列（1〜8,192 本）です。このフィールドは文書ストアに保存されません。

## 距離メトリクス

Vector フィールドの `distance` オプションは以下の値を受け付けます:

| 値 | 説明 | 使用場面 |
| :--- | :--- | :--- |
| `"Cosine"` | コサイン距離（1 - コサイン類似度）。デフォルト。 | 正規化されたテキスト/画像 Embedding |
| `"Euclidean"` | L2（ユークリッド）距離 | 空間データ、正規化されていないベクトル |
| `"Manhattan"` | L1（マンハッタン）距離 | スパースな特徴ベクトル |
| `"DotProduct"` | 内積（大きいほど類似度が高い） | 大きさが重要な正規化済みベクトル |
| `"Angular"` | 角度距離 | コサインに似ているが角度に基づく |

ほとんどの Embedding モデル（BERT、Sentence Transformers、OpenAI など）では `"Cosine"` が適切な選択です。

## 量子化

Vector フィールドはディスク上で **8 ビットスカラー量子化された整数**
として保存されます（Issue #481 Stage 1）。量子化は必須となり、以前
の「量子化なし」モードは廃止されました。`quantizer` オプションは
`Scalar8Bit` がデフォルトで、TOML から省略可能です。

### Scalar 8-bit（デフォルト）

per-segment global affine による `u8` 量子化。各 `f32` コンポーネント
を 1 バイトに圧縮（約 4 倍のメモリ削減）し、recall 損失は実用上ほぼ
無視できる範囲。

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"
# quantizer = "Scalar8Bit"  # デフォルトのため省略可
```

### Product Quantization（HNSW のみ）

Issue #481 Stage 3。各ベクトルを、sub-vector ごとに 256 centroid を
持つ codebook への 1 バイトの centroid index × `subvector_count` 個
として保存します（約 16-64 倍の圧縮）。HNSW index がサポートし、
Flat / IVF は書き込み時に拒否します。recall 回復のため
[Rerank Storage](#rerank-storage) との併用を推奨します。

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"
# 任意（Issue #631）: `laurus train pq-codebook` で codebook を一度
# だけ学習し、commit / merge ごとの k-means 再学習の代わりに
# segment 間で共有する。
pq_codebook_path = "embedding.pqcb"

[fields.embedding.Hnsw.quantizer.ProductQuantization]
subvector_count = 48
```

| オプション | 型 | 説明 |
| :--- | :--- | :--- |
| `subvector_count` | `integer` | サブベクトルの数。`dimension` を均等に割り切れる必要があります。 |

デフォルトでは codebook は segment ごとに学習されます（256 ベクトル
未満の segment は `Scalar8Bit` にフォールバック）。`pq_codebook_path`
を設定すると segment は共有の学習済み codebook で encode されます:
commit は大幅に高速化し、小さな per-commit segment も PQ を維持
します — ただし codebook の学習前に commit すると、実行すべき
`laurus train pq-codebook` コマンドを示すエラーで失敗します
（per-segment 学習への無言のフォールバックはありません）。学習
ワークフローは [`train` コマンド](commands.md#train) を参照して
ください。

> **破壊的変更（Issue #481 Stage 1）:** `quantizer` を「なし」に
> 設定するスキーマはもはや有効ではありません。Stage 1 より前の
> laurus でビルドした既存 vector index は読み取れないため、アップ
> グレード後にソースデータから再構築してください。

## Rerank Storage

任意の Stage 2 sidecar（Issue #481）。元の完全精度ベクトルを int8
セグメントの隣に保持し、searcher が int8 で広めに候補を取得
（高速）してから上位 `top_k * rerank_factor` 件を完全な f32 値で
再スコア（高精度）できるようにします。#932 以降、HNSW / Flat /
IVF の 3 タイプすべてでサポートされます（Flat / IVF の再スコアは
フィールド指定クエリに適用）。

sidecar はフィールド単位で `rerank_storage` で設定します:

```toml
[fields.embedding.Hnsw]
dimension = 384
distance = "Cosine"
rerank_storage = "F32"  # opt-in。省略すると Stage 1 int8-only の挙動を維持
```

| 値 | ディスク追加コスト | 説明 |
| :--- | :--- | :--- |
| `"F32"` | +4 bytes/dim/vector | IEEE-754 単精度 sidecar（Lucene 99 / FAISS 互換）。 |

省略した場合 sidecar は書かれず、フィールドは Stage 1 int8-only
の検索パスを維持します。`rerank_storage` を持たないフィールドに
対して `rerank_factor` を渡したクエリは silent に Stage 1
ランキングへフォールバックします — Stage 1 セグメントから index
作成時に捨てられた f32 情報を復元することはできません。

> **スコープ:** Stage 2 は HNSW のみで実装しています。Flat / IVF は
> スキーマの対称性のためにフィールドを受け付けますが、現状 sidecar
> の書き出し・読み込みは行いません。

## アナライザ

`[analyzers.<name>]` テーブルは、カスタムのテキスト解析パイプラインを定義します。Text フィールドは `analyzer` オプションにその名前を書いて使います。組み込みのアナライザでは足りないとき、たとえばステミングを加えたいときや、`japanese` プリセットのストップフィルタを使わずに日本語を解析したいときに定義します。パイプラインの仕組みは [テキスト解析](../concepts/analysis.md) を参照してください。

```toml
[analyzers.<name>]
char_filters = [{ type = "...", ... }, ...]   # 省略可能
tokenizer = { type = "...", ... }             # 必須
token_filters = [{ type = "...", ... }, ...]  # 省略可能
```

| キー | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `tokenizer` | テーブル | *（必須）* | テキストをトークンに分割する。必ず 1 つ |
| `char_filters` | テーブルの配列 | `[]` | トークン化の前に生テキストへ、配列の順に適用する |
| `token_filters` | テーブルの配列 | `[]` | トークン化の後にトークン列へ、配列の順に適用する |

各コンポーネントは、`type` キーで種類を選び、残りのキーで設定するテーブルです。TOML ではインラインテーブル（`{ type = "lowercase" }`）で書くのが一般的です。JSON 形式のスキーマ（`{"type": "lowercase"}`）や、各バインディングの `addAnalyzer` / `add_analyzer` も同じ形を使います。

### トークナイザ

| `type` | 必須キー | 省略可能キー | 説明 |
| :--- | :--- | :--- | :--- |
| `"whitespace"` | -- | -- | 空白で分割する |
| `"unicode_word"` | -- | -- | Unicode の単語境界で分割する |
| `"regex"` | -- | `pattern`（デフォルト `\w+`）、`gaps`（デフォルト `false`） | `pattern` に一致した部分をトークンにする。`gaps = true` のときは、`pattern` がトークン間の区切りに一致するものとして扱う |
| `"ngram"` | `min_gram`、`max_gram` | -- | `min_gram` 文字から `max_gram` 文字までのすべての n-gram を出力する |
| `"lindera"` | `mode`、`dict` | `user_dict` | [Lindera](https://github.com/lindera/lindera) による形態素解析。`mode` は `"normal"` か `"decompose"`。`dict` は Lindera 辞書のディレクトリ、`user_dict` はユーザー辞書のパス。laurus は辞書を同梱しないため、`dict` は実在するパスでなければならない |
| `"whole"` | -- | -- | 入力全体を 1 つのトークンにする |

### 文字フィルタ

| `type` | 必須キー | 省略可能キー | 説明 |
| :--- | :--- | :--- | :--- |
| `"unicode_normalization"` | `form`（`"nfc"` / `"nfd"` / `"nfkc"` / `"nfkd"`） | -- | Unicode 正規化を適用する |
| `"pattern_replace"` | `pattern`、`replacement` | -- | 正規表現 `pattern` に一致した部分を `replacement` に置き換える |
| `"mapping"` | `mapping`（置換用の文字列のテーブル） | -- | `mapping` の各キーを対応する値に置き換える |
| `"japanese_iteration_mark"` | -- | `kanji`（デフォルト `true`）、`kana`（デフォルト `true`） | 踊り字を展開する |

### トークンフィルタ

| `type` | 必須キー | 省略可能キー | 説明 |
| :--- | :--- | :--- | :--- |
| `"lowercase"` | -- | -- | 各トークンを小文字にする |
| `"stop"` | -- | `words`（デフォルト: 英語のストップワード） | ストップワードを取り除く |
| `"stem"` | -- | `stem_type`（`"porter"`（デフォルト）/ `"simple"` / `"identity"`） | 各トークンを語幹にする |
| `"boost"` | `boost` | -- | 各トークンのブーストに `boost` を掛ける |
| `"limit"` | `limit` | -- | 先頭の `limit` 個のトークンだけを残す |
| `"strip"` | -- | -- | 各トークンの前後の空白を取り除く |
| `"remove_empty"` | -- | -- | 空のトークンを取り除く |
| `"flatten_graph"` | -- | -- | トークングラフを直線的なトークン列に平坦化する。インデックス時は常に平坦化されるうえ、アナライザはクエリの解析にも使われるため、これを加えるとクエリ時に引用符付きの複数語の同義語が厳密に一致しなくなる |

### アナライザの参照

Text フィールドは `analyzer` オプションにアナライザの名前を書きます:

```toml
[fields.body.Text]
analyzer = "english_stemmed"
```

名前は次の順に解決されます:

1. バインディングから実行時に登録したアナライザ（例: WASM バインディングの `addAnalyzer`）
2. 組み込みのアナライザ: `standard`、`keyword`、`english`、`simple`、`noop`
3. `[analyzers.*]` のエントリ

組み込みが先に調べられるため、`standard`、`keyword`、`english`、`simple`、`noop` は予約された名前です。この名前の `[analyzers.*]` エントリは決して使われないので、エラーになります。`japanese` は予約されていません。組み込みの `japanese` は辞書が必要でテーブルで指定するため、`[analyzers.japanese]` は定義どおりに使われます。

エラーは次の 3 つの時点で起きます:

- 未知の `type` や必須キーの欠落は、スキーマの解析時にエラーになります。このとき `create index` は何も作りません。
- 組み込みと同じ名前のエントリは、インデックスの作成時に `Analyzer name 'standard' is reserved for a built-in analyzer; choose another name` というエラーになります。このとき `create index` は何も作りません。各バインディングの `addAnalyzer` / `add_analyzer`（WASM では `addAnalyzerDefinition`）も同じエラーを返します。この検査より前に作ったインデックスは、これまでどおり開けます。エントリは使われないままで、開くときに `log` クレートで警告を出します（`laurus-server` はこれをログに表示します）。
- 不正な値（誤った正規表現、未知の `form` や `stem_type`、存在しない Lindera 辞書）や、どこにも見つからない `analyzer` の名前は、インデックスの構築時に `Failed to resolve analyzer for field 'body': ...` のようなエラーになります。このとき `create index` は何も作りません — `schema.toml` と `store/` は、呼び出し前の状態（何もなければ「何もない」状態）まで巻き戻されます。

### 例: ステミング付きの英語テキスト

```toml
default_fields = ["title", "body"]

[analyzers.english_stemmed]
char_filters = [{ type = "unicode_normalization", form = "nfkc" }]
tokenizer = { type = "unicode_word" }
token_filters = [
    { type = "lowercase" },
    { type = "stop" },
    { type = "stem", stem_type = "porter" },
]

[fields.title.Text]
analyzer = "english_stemmed"

[fields.body.Text]
analyzer = "english_stemmed"

[fields.tag.Text]
analyzer = "keyword"
```

`body` が `"Ｄｏｇｓ are RUNNING in the park."` の文書は、`body:dog`（NFKC 正規化・小文字化・ステミングによる）と `body:run` に一致し、`body:the` には一致しません（ストップワードが取り除かれるため）。`tag` フィールドは組み込みの `keyword` アナライザのままなので、値そのものにだけ一致します。

### 例: Lindera による日本語テキスト

次の定義は `examples/aozora/schema.toml` から取ったものです。`{ language = "japanese" }` プリセットと違ってストップフィルタを含まないため、「の」「は」などの助詞もインデックスに残ります:

```toml
[analyzers.ja_ipadic]
tokenizer = { type = "lindera", mode = "normal", dict = "/var/lib/lindera/ipadic" }
char_filters = [
    { type = "unicode_normalization", form = "nfkc" },
    { type = "japanese_iteration_mark", kanji = true, kana = true },
]
token_filters = [{ type = "lowercase" }]

[fields.title.Text]
analyzer = "ja_ipadic"
```

`dict` には展開済みの Lindera 辞書（通常は IPADIC）を指定します。存在しないパスを指定すると、`create index` は `Failed to load dictionary: ... Dictionary path does not exist` で失敗します。

## エンベダー

`[embedders.<name>]` テーブルは、埋め込みモデルを宣言します。ベクトルフィールド（Hnsw、Flat、Ivf）は `embedder` オプションにその名前を書いて使います。するとそのフィールドに与えたテキスト（CLIP の場合は画像も）が、文書のインデックス時と、そのフィールドを対象とするクエリの実行時の両方で、モデルによってベクトルに変換されます。1 つのエンベダーを複数のフィールドで共有できます。各モデルの仕組みと選び方は [Embedding](../concepts/embedding.md) を参照してください。

```toml
[embedders.<name>]
type = "..."   # 必須
model = "..."  # "precomputed" 以外のすべての型で必須
```

| `type` | 必須キー | Feature Flag | 説明 |
| :--- | :--- | :--- | :--- |
| `"precomputed"` | -- | *（常に利用可能）* | 埋め込みを行わない。文書がベクトルを直接与える |
| `"candle_bert"` | `model` | `embeddings-candle` | Hugging Face Hub の BERT 系モデル（例: `"sentence-transformers/all-MiniLM-L6-v2"`）によるローカルでのテキスト埋め込み |
| `"candle_clip"` | `model` | `embeddings-multimodal` | Hugging Face Hub の CLIP モデル（例: `"openai/clip-vit-base-patch32"`）によるローカルでのテキストと画像の埋め込み |
| `"openai"` | `model` | `embeddings-openai` | OpenAI API（例: `"text-embedding-3-small"`）によるテキスト埋め込み。API キーはエンジン起動時に環境変数 `OPENAI_API_KEY` から読み、スキーマには保存しない |

Hugging Face のモデルは初回の使用時にダウンロードされ、`$HF_HOME`（デフォルトは `~/.cache/huggingface`）にキャッシュされます。ベクトルフィールドの `dimension` は、モデルの出力次元と一致させる必要があります。

> **注意:** リリースで配布しているビルド済みバイナリは `--features embeddings-all` 付きでビルドされています。一方、`cargo install laurus-cli` やソースからのビルドでは、feature を指定しない限り（例: `cargo install laurus-cli --features embeddings-candle`）埋め込みの feature がどれも有効にならず、使えるのは `"precomputed"` だけです。[インストール](installation.md) と [Feature Flags](../development/feature_flags.md) を参照してください。feature が無効な型を書いたスキーマも解析は通りますが、`create index` が次のエラーで失敗します:
>
> ```text
> Error: Not implemented: candle_bert embedder requires the 'embeddings-candle' feature to be enabled
> ```

ベクトルフィールドの `embedder` には、`[embedders.*]` のエントリの名前を書く必要があります。宣言していない名前は `create index`、`add field`、`update field` で拒否されます。`schema.toml` にそうした名前を含む既存のインデックスは開けません:

```text
Error: Invalid argument: Unknown embedder 'missing' for field 'vec': not defined in schema.embedders
```

このインデックスを開くには、`schema.toml` を編集します。その名前を `type = "precomputed"` で宣言すれば、フィールドはそれまでどおり、文書が与えるベクトルで動きます。フィールドの `embedder` の行を削除しても同じです。

### 例: 1 つのエンベダーを 2 つのフィールドで共有する

```toml
[embedders.text_embedder]
type = "candle_bert"
model = "sentence-transformers/all-MiniLM-L6-v2"

[fields.title_vec.Hnsw]
dimension = 384
distance = "Cosine"
embedder = "text_embedder"

[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
embedder = "text_embedder"
```

## 完全な例

### 全文検索のみ

Lexical 検索のみのシンプルなブログ記事インデックス:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
indexed = true
stored = true
term_vectors = true

[fields.body.Text]
indexed = true
stored = true
term_vectors = true

[fields.category.Text]
indexed = true
stored = true
term_vectors = false

[fields.published_at.DateTime]
indexed = true
stored = true
```

### Vector 検索のみ

セマンティック類似検索用の Vector のみのインデックス:

```toml
[fields.embedding.Hnsw]
dimension = 768
distance = "Cosine"
m = 16
ef_construction = 200
```

### ハイブリッド検索（Lexical + Vector）

Lexical 検索と Vector 検索を組み合わせた両方の長所を活かす検索:

```toml
default_fields = ["title", "body"]

[fields.title.Text]
indexed = true
stored = true
term_vectors = true

[fields.body.Text]
indexed = true
stored = true
term_vectors = true

[fields.category.Text]
indexed = true
stored = true
term_vectors = false

[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
m = 16
ef_construction = 200
```

> **ヒント:** 1つのフィールドが Lexical と Vector の両方を兼ねることはできません。別々のフィールド（例: テキスト用の `body`、Embedding 用の `body_vec`）を使用し、どちらも同じソースコンテンツにマッピングしてください。

### E コマースの商品インデックス

複数のフィールド型を組み合わせたより複雑なスキーマ:

```toml
default_fields = ["name", "description"]

[fields.name.Text]
indexed = true
stored = true
term_vectors = true

[fields.description.Text]
indexed = true
stored = true
term_vectors = true

[fields.price.Float]
indexed = true
stored = true

[fields.in_stock.Boolean]
indexed = true
stored = true

[fields.created_at.DateTime]
indexed = true
stored = true

[fields.location.Geo]
indexed = true
stored = true

[fields.description_vec.Hnsw]
dimension = 384
distance = "Cosine"
```

### カスタム解析と自動埋め込み

Text フィールドにカスタムアナライザを使い、ベクトルフィールドではテキストをローカルのモデルで埋め込むハイブリッドインデックスです。`embeddings-candle` feature 付きの `laurus` バイナリが必要です（[エンベダー](#エンベダー) を参照）:

```toml
default_fields = ["title", "body"]

[analyzers.english_stemmed]
tokenizer = { type = "unicode_word" }
token_filters = [
    { type = "lowercase" },
    { type = "stop" },
    { type = "stem" },
]

[embedders.text_embedder]
type = "candle_bert"
model = "sentence-transformers/all-MiniLM-L6-v2"

[fields.title.Text]
analyzer = "english_stemmed"

[fields.body.Text]
analyzer = "english_stemmed"

[fields.body_vec.Hnsw]
dimension = 384
distance = "Cosine"
embedder = "text_embedder"
```

## スキーマの生成

CLI を使用して対話的にスキーマ TOML ファイルを生成できます:

```bash
laurus create schema
laurus create schema --output my_schema.toml
```

詳細は [`create schema`](commands.md#create-schema) を参照してください。

## スキーマの使用

スキーマファイルが用意できたら、そこからインデックスを作成します:

```bash
laurus create index --schema schema.toml
```

または Rust でプログラム的に読み込みます:

```rust
use laurus::Schema;

let toml_str = std::fs::read_to_string("schema.toml")?;
let schema: Schema = toml::from_str(&toml_str)?;
```
