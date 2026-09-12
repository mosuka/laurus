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

## パフォーマンス

ファセットカウントは stored document ではなく、各フィールドの **DocValues** 列から読み取られます。
収集された各ヒットについて、コレクターはファセットフィールドの値だけを per-field の DocValues
ルックアップで読むため、ファセット対象の全フィールドが DocValues 列を持つ場合（`stored: true` な
フィールドは既定でこれに該当します。ただし後述のとおり型によって除外される場合や、`doc_values`
オプションが明示的に `false` に設定されている場合を除きます）、stored fields blob 全体を
decode / clone しません。DocValues を持たないフィールド ―― オプトアウトしている、`stored`
ではない、あるいは `Bytes`/`Vector` の値（DocValues には設定にかかわらず一切格納されません）
であるため ―― は透過的に stored document へフォールバックするため、結果はどちらの経路でも
同一で、変わるのは読み取り経路だけです。

ソートにもファセットにも使わないフィールドで `doc_values: false` を設定すると、値が二重（stored
document と DocValues）ではなく一度（stored document のみ）しか書き込まれなくなるため、
セグメントの使用容量が削減されます。
