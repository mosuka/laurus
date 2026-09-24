# ファセット

ファセット（Faceting）は、フィールド値によって検索結果をカウント・分類する機能です。検索UIでナビゲーションフィルタを構築するために一般的に使用されます（例: 「エレクトロニクス (42)」「書籍 (18)」）。

## 概念

### FacetPath

`FacetPath` は階層的なファセット値を表します。例えば、商品カテゴリ「Electronics > Computers > Laptops」は3階層のFacetPathです。

```rust
use laurus::lexical::search::features::facet::FacetPath;

// 単一レベルのファセット
let facet = FacetPath::from_value("category", "Electronics");

// コンポーネントからの階層的ファセット
let facet = FacetPath::new("category", vec![
    "Electronics".to_string(),
    "Computers".to_string(),
    "Laptops".to_string(),
]);

// 区切り文字付き文字列から
let facet = FacetPath::from_delimited("category", "Electronics/Computers/Laptops", "/");
```

#### FacetPathメソッド

| メソッド | 説明 |
| :--- | :--- |
| `new(field, path)` | フィールド名とパスコンポーネントからFacetPathを作成 |
| `from_value(field, value)` | 単一レベルのファセットを作成 |
| `from_delimited(field, path_str, delimiter)` | 区切り文字付きのパス文字列をパース |
| `depth()` | パスの階層数 |
| `is_parent_of(other)` | このパスが他のパスの親であるか確認 |
| `parent()` | 親パスを取得（1階層上） |
| `child(component)` | コンポーネントを追加して子パスを作成 |
| `to_string_with_delimiter(delimiter)` | 区切り文字付き文字列に変換 |

### FacetCount

`FacetCount` はファセット集計の結果を表します。

```rust
pub struct FacetCount {
    pub path: FacetPath,
    pub count: u64,
    pub children: Vec<FacetCount>,
}
```

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `path` | `FacetPath` | ファセット値 |
| `count` | `u64` | マッチするドキュメント数 |
| `children` | `Vec<FacetCount>` | 階層的なドリルダウン用の子ファセット |

## 例: 階層的ファセット

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

このツリーの各ノードは、ドリルダウンナビゲーション用に `children` が設定された `FacetCount` に対応します。

## ユースケース

- **EC（電子商取引）**: カテゴリ、ブランド、価格帯、評価によるフィルタリング
- **ドキュメント検索**: 著者、部門、日付範囲、ドキュメントタイプによるフィルタリング
- **コンテンツ管理**: タグ、トピック、コンテンツステータスによるフィルタリング

## 多値フィールド

多値フィールド（`multi_valued = true`。[多値フィールド](../concepts/schema_and_fields.md)
を参照）は 1 ドキュメントに配列を保持し、コレクターはそれを展開します: **各要素がそれぞれ
独立したファセットパスになります**（Issue #1187）。`/` を含む `TextArray` の要素は、スカラーの
`Text` 値とまったく同じ規則で階層パスに分割されます。そのため `tags = ["rust", "search"]` は
`rust` と `search` をそれぞれ 1 回ずつ数え、`cat = ["a/b", "a/c"]` は `a/b`・`a/c` と、両者が
共有する祖先 `a` を数えます。

カウントは Lucene の `SortedSetDocValuesFacetCounts` に倣って**ドキュメント単位**です:
1 ドキュメント内で 2 回現れる要素（`["rust", "rust"]`）は 1 回だけ数えられ、2 つの要素から
到達する祖先も同様です（上の `a` は 2 ではなく 1）。したがって `FacetCount::count` は常に
「マッチしたドキュメント数」を意味します。空配列は何も寄与しません。

配列の要素は、同じ型のスカラー値とまったく同じ形式で文字列化されます:

| 値 | ファセット値 |
| :--- | :--- |
| `Text` / `TextArray` | 文字列そのまま。`/` は階層の成分に分割される |
| `Int64` / `Int64Array` | 10 進整数（例: `42`） |
| `Float64` / `Float64Array` | 常に小数点付き（例: `2.0`、`2.5`）。浮動小数点が整数と同じラベルになることはない（浮動小数点を *Text フィールドへ* 変換する経路では `2.0` は `2` になる。別のコードパス） |
| `Bool` / `BoolArray` | `true` / `false` |
| `DateTime` / `DateTimeArray` | UTC の RFC 3339、マイクロ秒精度（例: `2024-01-01T00:00:00+00:00`）。マイクロ秒未満の桁は切り捨てられるため、DocValues から読んでも stored document から読んでもラベルは同じ |

`Null`、地理座標（`Geo`・`GeoEcef` とそれらの配列）、`Vector`、`Bytes` はファセットの対象外で、
何も寄与しません。DocValues がヒットしてファセット値が 0 個になった場合でも、stored document
へのフォールバックは行いません。

階層パスは現状、入れ子の `children` ではなくフラットな兄弟（`["a"]`、`["a", "b"]`）として
返されます。Issue #1192 を参照してください。

## パフォーマンス

ファセットカウントは stored document ではなく、各フィールドの **DocValues** 列から読み取られます。
収集された各ヒットについて、コレクターはファセットフィールドの値だけを per-field の DocValues
ルックアップで読むため、ファセット対象の全フィールドが DocValues 列を持つ場合（`stored: true` な
フィールドは既定でこれに該当します。ただし後述のとおり型によって除外される場合や、`doc_values`
オプションが明示的に `false` に設定されている場合を除きます）、stored fields blob 全体を
decode / clone しません。DocValues を持たないフィールド ―― オプトアウトしている、`stored`
ではない、あるいは `Bytes`/`Vector` の値（DocValues には設定にかかわらず一切格納されません）
であるため ―― は透過的に stored document へフォールバックするため、結果はどちらの経路でも
同一で、変わるのは読み取り経路だけです。多値フィールドの配列値は DocValues に丸ごと
（1 ドキュメント 1 エントリ）格納され、ファセット時に要素へ分割されるため、展開によって
DocValues の読み取り回数が増えることはありません。

ソートにもファセットにも使わないフィールドで `doc_values: false` を設定すると、値が二重（stored
document と DocValues）ではなく一度（stored document のみ）しか書き込まれなくなるため、
セグメントの使用容量が削減されます。
