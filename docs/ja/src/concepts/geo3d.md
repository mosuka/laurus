# 3D 地理検索 (ECEF)

Laurus は用途の異なる 2 種類の地理フィールド型を提供する：

| フィールド型 | バックエンド構造 | 座標系 | こんなときに |
| :--- | :--- | :--- | :--- |
| `Geo` | 2D BKD-Tree | WGS84 緯度・経度（度） | データが地表のみで、高度を考慮しなくてよい場合 |
| `Geo3d` | 3D BKD-Tree | ECEF 直交座標 `(x, y, z)`（メートル） | 高度が一級の次元になる用途（ドローン・衛星・屋内測位・複数フロア建物） |

両者は同じ [BKD-Tree](bkd_tree.md) プリミティブを共有しており、`Geo3d`
は単に第 3 の次元と異なるクエリ語彙を加えただけである。

## なぜ ECEF なのか

`(緯度, 経度, 高度)` の組は人間にとっては便利だが、ユークリッド距離が
そのままでは使えない：経度 1 度は赤道で約 111 km だが極では 0 km、
そして「高度」は地球表面に沿って曲がっている。

ECEF (Earth-Centered Earth-Fixed) はこれを直交座標系にフラット化する：

- **原点**は地球の質量中心。
- **+X 軸**は赤道上の経度 0° を貫く。
- **+Y 軸**は赤道上の経度 90° E を貫く。
- **+Z 軸**は地理北極を貫く。
- 3 軸とも単位は**メートル**。

この座標系では **2 点間の直線距離が単なるユークリッドノルム**になる —
球面三角法も極特異点も経度のラップアラウンドも要らない。これこそが、
`Geo3d` クエリを 3D BKD-Tree 上で（特殊な空間コードなしで）使える
理由である。

## Geo3d フィールドの定義

```rust
use laurus::Schema;
use laurus::lexical::core::field::Geo3dOption;

let schema = Schema::builder()
    .add_geo3d_field("position", Geo3dOption::default())
    .build();
```

`Geo3dOption` は他の lexical フィールドオプションと同じく
`indexed` / `stored` を提供する。インデックス済みの値はフィールドの
3D BKD-Tree に格納され、stored な値は `get_documents` で返ってくる。

TOML 形式（`laurus-cli` で使用）：

```toml
[fields.position.Geo3d]
indexed = true
stored = true
```

## 3D ポイントのインデクシング

`DocumentBuilder::add_geo_ecef` を使って、メートル単位の生の直交座標で
ポイントを追加する：

```rust
use laurus::Document;

// (lat=35.6586°, lon=139.7454°, height=250m) のポイント
// 既に ECEF へ変換済み（後述「座標変換」参照）。
let doc = Document::builder()
    .add_geo_ecef("position", -3_955_182.0, 3_350_553.0, 3_700_276.0)
    .build();

engine.put_document("tokyo-tower", doc).await?;
engine.commit().await?;
```

値が既に `GeoEcefPoint` として手元にある場合は、統一 `DataValue` API を
使う：

```rust
use laurus::{DataValue, GeoEcefPoint};

let p = GeoEcefPoint::new(-3_955_182.0, 3_350_553.0, 3_700_276.0);
let doc = Document::builder()
    .add_field("position", DataValue::GeoEcef(p))
    .build();
```

## 座標変換 (WGS84 ↔ ECEF)

入力は通常 `(緯度°, 経度°, 高度 m)` でやって来る。Laurus は
[`laurus::util::ecef`](https://github.com/mosuka/laurus/blob/main/laurus/src/util/ecef.rs)
で標準的な相互変換を提供する：

```rust
use laurus::util::ecef::{wgs84_to_ecef, ecef_to_wgs84};

// 順変換: lat/lon/height → ECEF
let p = wgs84_to_ecef(35.6586, 139.7454, 250.0);

// 逆変換: ECEF → lat/lon/height
let (lat, lon, height) = ecef_to_wgs84(&p);
```

| 関数 | 方向 | アルゴリズム |
| :--- | :--- | :--- |
| `wgs84_to_ecef(lat°, lon°, h_m)` | 地理 → 直交 | 卯酉線（prime vertical）の閉形式公式 |
| `ecef_to_wgs84(&GeoEcefPoint)` | 直交 → 地理 | Bowring 1985 の閉形式を初期値にした Newton-Raphson 3 反復 |

逆変換は地表下から LEO 軌道高度をはるかに超える領域まで、各軸で
**サブ µm** の精度で往復する。極付近では、`cos(lat) → 0` で発散する
標準形 `h = p / cos(lat) - N` の代わりに、子午線形
`h = z / sin(lat) - N(1 - e²)` に切り替える。

内部で使う WGS84 楕円体定数は `pub const` として公開されている
（`WGS84_A`, `WGS84_F`, `WGS84_B`, `WGS84_E2`, `WGS84_E_PRIME_SQ`）ので、
外部コードからも laurus が使っているのと同じ値を参照できる。

## クエリ種別

`Geo3d` フィールドを対象とするクエリは 3 種類ある。いずれも 3D BKD-Tree
を共有する [`IntersectVisitor`](bkd_tree.md#intersectvisitor-プロトコル)
の実装である。

### 球（半径） — `Geo3dDistanceQuery`

中心点から `distance_m` メートル以内に格納済みポイントが入っているドキュメントを
すべて返す。スコアは `1 - distance / radius` を `[0, 1]` にクランプした
値。

```rust
use laurus::lexical::query::geo3d::Geo3dDistanceQuery;
use laurus::GeoEcefPoint;

let centre = GeoEcefPoint::new(-3_955_182.0, 3_350_553.0, 3_700_276.0);
let query = Geo3dDistanceQuery::new("position", centre, 5_000.0); // 5 km 半径
```

ビジタは [`AABB::min_distance_sq_to_point`] によるセルの枝刈り（`Outside`
判定。2 乗距離空間で比較するため `sqrt` を回避できる）だけを行い、`compare`
から `CellRelation::Inside` を返すことはない。`Inside` 経路は座標なしで
doc id を報告するが、このクエリはスコアと距離昇順の並びのために全ポイントの
実際の距離を必要とするためである。そのため球の内側に完全に収まるセルも
`Crosses` として訪問し、ポイントをデコードする（後述の k-NN ビジタと同じ）。
（#1594 以前はこうしたセルを `Inside` と報告し、その配下の全ポイントが実際の
位置に関係なく距離 `0`・スコア `1.0` になっていた。）

### バウンディングボックス — `Geo3dBoundingBoxQuery`

`(min_x, min_y, min_z)` と `(max_x, max_y, max_z)` で定義された軸並行 3D
ボックスにポイントが入っているドキュメントを返す。

```rust
use laurus::lexical::query::geo3d::Geo3dBoundingBoxQuery;
use laurus::GeoEcefPoint;

let min = GeoEcefPoint::new(-4_000_000.0, 3_300_000.0, 3_650_000.0);
let max = GeoEcefPoint::new(-3_900_000.0, 3_400_000.0, 3_750_000.0);
let query = Geo3dBoundingBoxQuery::new("position", min, max)?;
```

コンストラクタが `Result` を返すのは、すべての軸で `min[i] ≤ max[i]` を
満たす必要があるため。境界がおかしい場合は構築時点で拒否され、
「黙って空結果」を防げる。

> **注意点。** クエリボックスは ECEF 直交座標で軸並行になっている。
> `(lat, lon, height)` の 2 つの隅をそれぞれ ECEF に変換して作った
> ボックスは、ユーザーが直感的に思い描く球殻領域とは**違う**形をしている。
> 地表上の領域クエリを書くなら、`Geo3dDistanceQuery`（真の 3D 球）の方が
> 多くの場合ユーザー直感に近い。

### k 近傍 — `Geo3dNearestQuery`

クエリ点に最も近い `k` 件のドキュメントを返す。クエリは小さなプローブ球
（既定 1 km）から始め、以下のいずれかを満たすまで**半径を倍加**していく：

1. 重複なしで `k` 件以上の候補を集めた、
2. 倍加しても何も新しいヒットが見つからなくなった、
3. 半径が `max_radius_m`（既定 `1.0e10` m＝10⁷ km）に達した。

```rust
use laurus::lexical::query::geo3d::Geo3dNearestQuery;
use laurus::GeoEcefPoint;

let centre = GeoEcefPoint::new(-3_955_182.0, 3_350_553.0, 3_700_276.0);
let query = Geo3dNearestQuery::new("position", centre, 10) // 上位 10 件
    .with_initial_radius(500.0)         // 500 m から開始
    .with_max_radius(1_000_000.0);      // 1 000 km で打ち切り
```

結果が `k` 件未満になるのは、単に `max_radius_m` 以内に `k` 件のポイントが
存在しないことを意味する。スコアは「最も近いヒットが `1.0`、返却された
集合内で最も遠いヒットが `0.0`」となるよう正規化される。

k-NN ビジタは `compare` から `CellRelation::Inside` を返さない —
`Outside` か `Crosses` のみ — ので、すべての候補がその座標と一緒に `visit`
へ届けられ、k-NN の順序付けに真の距離を使える。

## Query DSL

`Geo3d` クエリは統一 Query DSL からも使える（[Query DSL → Geo3d 関数](query_dsl.md#3d-地理クエリgeo3d_)
を参照）：

```text
position:geo3d_distance(-3955182, 3350553, 3700276, 5000)
position:geo3d_bbox(-4000000, 3300000, 3650000, -3900000, 3400000, 3750000)
position:geo3d_nearest(-3955182, 3350553, 3700276, 10)
```

| 形式 | 引数 |
| :--- | :--- |
| `geo3d_distance(x, y, z, distance_m)` | 中心 `(x, y, z)` と最大距離（m） |
| `geo3d_bbox(min_x, min_y, min_z, max_x, max_y, max_z)` | AABB の対角の 2 隅 |
| `geo3d_nearest(x, y, z, k)` | 中心 `(x, y, z)` と整数 `k` |

数値引数はすべて符号付き浮動小数。`geo3d_nearest` の `k` のみ符号なし整数。

## ワイヤーフォーマット

### gRPC

Protocol Buffers では、3D 地理を専用の `Value` バリアント、`Geo3dPoint`
メッセージ、`Geo3dOption` スキーマオプションとして公開している：

```proto
message Geo3dPoint {
  double x = 1;
  double y = 2;
  double z = 3;
}

message Value {
  oneof kind {
    // ...
    Geo3dPoint geo3d_value = 12;
  }
}

message Geo3dOption {
  bool indexed = 1;
  bool stored = 2;
}
```

完全な定義は [`common.proto`](https://github.com/mosuka/laurus/blob/main/laurus-server/proto/laurus/v1/common.proto)
と [`index.proto`](https://github.com/mosuka/laurus/blob/main/laurus-server/proto/laurus/v1/index.proto)
を参照。

### HTTP gateway / MCP

JSON ドキュメントでは `Geo3d` 値を `{ "x": …, "y": …, "z": … }` の
オブジェクトとして受け付ける：

```json
{
  "position": { "x": -3955182.0, "y": 3350553.0, "z": 3700276.0 }
}
```

HTTP gateway はこれをエンジンへ転送する前に `DataValue::GeoEcef` に変換する。
MCP サーバーも同じ規約に従う。

## Geo3d を使うべき**でない**ケース

- **純粋な 2D 地図クエリ**（地表上の座標からの半径検索、タイル地図上の
  単純なバウンディングボックスなど） — 引き続き `Geo` を使うこと。
  2D BKD はひとつ次元が軽く、WGS84 入力もユーザー直感に近い。
- **学習済みロケーション埋め込みの近似的な意味類似度** — それはベクトル
  フィールド（`Hnsw`, `Flat`, `Ivf`）の領分であって、`Geo3d` ではない。

## 関連項目

- [BKD-Tree](bkd_tree.md) — `Integer`, `Float`, `DateTime`, 2D `Geo`
  も含めて支える、共通の多次元インデックス。
- [スキーマとフィールド](schema_and_fields.md) — `Geo3d` を含む
  フィールド型の一覧。
- [Query DSL](query_dsl.md) — geo3d DSL 文法の詳細。
- [Lexical 検索](search/lexical_search.md) — Lexical クエリ全般の Rust
  API エントリポイント。
- [`laurus-wasm/examples/geo3d/`](https://github.com/mosuka/laurus/tree/main/laurus-wasm/examples/geo3d) —
  ブラウザ向けデモ。CesiumJS の 3D 地球儀上に人工衛星のライブ位置を
  プロットし、`geo3d_bbox` と `geo3d_nearest` を実演します。

[`AABB::min_distance_sq_to_point`]: https://github.com/mosuka/laurus/blob/main/laurus/src/lexical/index/structures/aabb.rs
[`AABB::max_distance_sq_to_point`]: https://github.com/mosuka/laurus/blob/main/laurus/src/lexical/index/structures/aabb.rs
