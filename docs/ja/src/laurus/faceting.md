# ファセット

ファセット（Faceting）は、フィールド値によって検索結果をカウント・分類する機能です。検索UIでナビゲーションフィルタを構築するために一般的に使用されます（例: 「エレクトロニクス (42)」「書籍 (18)」）。

## 概念

### FacetPath

`FacetPath` は階層的なファセット値を表します。例えば、商品カテゴリ「Electronics > Computers > Laptops」は3階層のFacetPathです。

```rust
use laurus::lexical::search::features::facet::FacetPath;

// 単一レベルのファセット
let facet = FacetPath::from_value("category".into(), "Electronics".into());

// コンポーネントからの階層的ファセット
let facet = FacetPath::new("category".into(), vec![
    "Electronics".to_string(),
    "Computers".to_string(),
    "Laptops".to_string(),
]);

// 区切り文字付き文字列から
let facet = FacetPath::from_delimited("category".into(), "Electronics/Computers/Laptops", "/");
```

`from_delimited` はコレクターと同じく空の成分を捨てます。そのため
`"/Electronics//Computers"` は `["Electronics", "Computers"]` になります。

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
| `path` | `FacetPath` | ファセット値。フィールドのトップレベルからの完全なパス |
| `count` | `u64` | 値がこのパス、またはその下にある、マッチしたドキュメントの数 |
| `children` | `Vec<FacetCount>` | 1 階層下のファセット。階層的なドリルダウン用 |

### FacetConfig

`FacetConfig` は、コレクターが何を数えて何を返すかを指定します。

| フィールド | 既定値 | 説明 |
| :--- | :--- | :--- |
| `max_facets_per_field` | `100` | 1 階層あたりに残す値の最大数。各フィールドのトップレベルと、各ノードの子に、それぞれ別々に適用される。並べ替えの後に適用する |
| `max_depth` | `10` | 集計時にパスを先頭 `max_depth` 個の成分で切り詰める。それより深い階層は数えない。`0` は何も数えず、`usize::MAX` はすべての階層を残す |
| `min_count` | `1` | 値を返すのに必要な最小ドキュメント数。これに満たない値は、その子と一緒に落とされる |
| `sort_by_count` | `true` | 各階層を件数の降順に並べ、同数ならラベル順にする。`false` なら各階層をラベル順に並べる |

集計したドキュメントのどれも持たない値は返らないため、`min_count` の `0` は `1` と同じ動作になります。

## ファセットの集計

一致した各ドキュメントを `FacetCollector` に渡し、最後に `finalize` を呼びます。

```rust
use laurus::lexical::search::features::facet::{FacetCollector, FacetConfig};

let mut collector = FacetCollector::new(FacetConfig::default(), vec!["category".to_string()]);
for doc_id in matching_doc_ids {
    collector.collect_doc(doc_id, reader.as_ref())?;
}
let results = collector.finalize()?;

for facet in results.get_field_facets("category").into_iter().flatten() {
    println!("{} ({})", facet.path.to_string_with_delimiter("/"), facet.count);
}
```

ファセットのフィールドを stored document から読む必要があり、その読み取りに失敗すると、
`collect_doc` はエラーを返します。エラーの後はコレクターの件数が不完全なので、そのコレクターは
破棄してください。

## 階層的ファセット

`/` を含む `Text` 値は階層パスです。`Electronics/Computers/Laptops` は 3 階層です。コレクターは
パスとその各祖先を、ドキュメントごとに 1 回ずつ数えます。`finalize` はフィールドごとに 1 つの
ツリーを返します。トップレベルの値は `get_field_facets(field)` に、それより深い階層は親の
`children` に入ります。

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

ノードの `count` は、値がそのパス、またはその下にあるドキュメントの数です。したがって、子の件数が
親を上回ることはありません。フラットな値と、階層的な値の根は同じノードです。例えば `cat = "a"` の
ドキュメントと `cat = "a/b"` のドキュメントからは、子 `b (1)` を 1 つ持つ `a (2)` が 1 つだけ
できます。

各階層は、それぞれ独立に絞り込み・並べ替え・切り詰めが行われます。

- `min_count` は、値をその部分木全体と一緒に落とします。
- `max_facets_per_field` は、トップレベルと各ノードの子に別々に適用されます。そのため、祖先が
  子孫の枠を使い切ることはありません。
- 件数が同じ値はラベル順に並ぶので、どの値が残るかがハッシュの順序に左右されません。これは
  Lucene・Tantivy と同じです。

空の成分は捨てられます。`"/a/b"` と `"a//b"` はどちらも `a/b` に、`"a/"` は `a` になり、`""` と
`"/"` は何も数えません。Lucene は索引時にこのような成分を拒否します。laurus はファセットを検索時に
通常のテキスト値から作るので、検索を失敗させる代わりに捨てます。

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
