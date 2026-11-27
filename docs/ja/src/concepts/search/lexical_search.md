# Lexical 検索

Lexical 検索は、転置インデックスに対してキーワードをマッチングすることでドキュメントを検索します。Laurus は、完全一致、フレーズ一致、あいまい一致など、豊富なクエリタイプを提供します。

## 基本的な使い方

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::LexicalSearchQuery;

let request = SearchRequestBuilder::new()
    .lexical_query(
        LexicalSearchQuery::Obj(
            Box::new(TermQuery::new("body", "rust"))
        )
    )
    .limit(10)
    .build();

let results = engine.search(request).await?.results;
```

## クエリタイプ

### TermQuery

特定のフィールドに完全一致するタームを含むドキュメントをマッチングします。

```rust
use laurus::lexical::TermQuery;

// Find documents where "body" contains the term "rust"
let query = TermQuery::new("body", "rust");
```

> **注意:** タームは解析後にマッチングされます。フィールドが `StandardAnalyzer` を使用している場合、インデキシングされたテキストとクエリタームの両方が小文字化されるため、`TermQuery::new("body", "rust")` は元テキスト内の "Rust" にもマッチします。

### PhraseQuery

正確なタームの並びを含むドキュメントをマッチングします。

```rust
use laurus::lexical::query::phrase::PhraseQuery;

// Find documents containing the exact phrase "machine learning"
let query = PhraseQuery::new("body", vec!["machine".to_string(), "learning".to_string()]);

// Or use the convenience method from a phrase string:
let query = PhraseQuery::from_phrase("body", "machine learning");
```

フレーズクエリは、ターム位置情報が格納されている必要があります（`TextOption` のデフォルト設定）。多値テキストフィールド（`multi_valued = true`）では、各要素がフィールドの `position_increment_gap`（デフォルト 100）で区切られた 1 本の位置列を共有するため、slop がその gap に達しない限りフレーズが 2 つの要素をまたぐことはありません —— 詳細は [スキーマとフィールド](../schema_and_fields.md) を参照してください。

### BooleanQuery

複数のクエリをブーリアン論理で結合します。

```rust
use laurus::lexical::query::boolean::{BooleanQuery, BooleanQueryBuilder, Occur};

let query = BooleanQueryBuilder::new()
    .must(Box::new(TermQuery::new("body", "rust")))       // AND
    .must(Box::new(TermQuery::new("body", "programming"))) // AND
    .must_not(Box::new(TermQuery::new("body", "python")))  // NOT
    .build();
```

| Occur | 意味 | DSL での表現 |
| :--- | :--- | :--- |
| `Must` | ドキュメントが必ずマッチしなければならない | `+term` または `AND` |
| `Should` | ドキュメントがマッチすべき（スコアをブースト） | `term` または `OR` |
| `MustNot` | ドキュメントがマッチしてはならない | `-term` または `NOT term` |
| `Filter` | 必ずマッチする必要があるが、スコアには影響しない | （DSL に相当するものなし） |

`Must` 句も `Should` 句も無い `BooleanQuery`（filter だけ、除外だけ、またはその両方）は、boost にかかわらず、どのマッチにもスコア 0 を付けます。Lucene と Elasticsearch の filter、Elasticsearch の除外だけのクエリと同じです。別のクエリの中では何も足さず、`min_score` が 0 より大きいと何も返しません。Issue #1744 までは、1 つのドキュメントに 1 回だけ現れる語のスコアを付けていました（並列検索の filter だけのクエリは 0 でした）。

### FuzzyQuery

指定された編集距離（レーベンシュタイン距離）内のタームをマッチングします。

```rust
use laurus::lexical::query::fuzzy::FuzzyQuery;

// Find documents matching "programing" within edit distance 2
// This will match "programming", "programing", etc.
let query = FuzzyQuery::new("body", "programing");  // default max_edits = 2
```

スコアの対象になるのは近い順のタームで、Lucene と同じくデフォルトで 50 個です。近さは `1 - 距離 / 短いほうのタームの長さ`（文字数で数え、`0` 以上）で、完全に一致するタームは `1` になり、外れることはありません。同じ近さなら文書頻度の高い順です。この値は、タームのスコアの係数にもなります。どのタームも、その中で最も頻出するタームの文書頻度でスコアを付けるので、どれだけ希少かはスコアに関わりません。

`max_expansions` はタームの数（1 以上）を、`with_rewrite_method` はスコアの付け方を決めます。2 つは 1 つの設定で、後の呼び出しが有効です。

| `RewriteMethod` | スコア | そのクエリの検索で返るヒット |
| :--- | :--- | :--- |
| `TopTermsBlended { max_expansions }`（デフォルト、50） | 近い順のタームの BM25 に係数を掛けたもの。どのタームも、その中で最も頻出するタームと同じ文書頻度とみなす | そのどれかのタームを含むドキュメント |
| `TopTermsScoring { max_expansions }` | 近い順のタームの BM25 に係数を掛けたもの | そのどれかのタームを含むドキュメント |
| `ConstantScore` | ブーストの値 | すべてのマッチ |
| `BooleanQuery` | マッチしたすべてのタームの BM25 に係数を掛けたもの | すべてのマッチ。コストは[後述](#ワイルドカードプレフィックス正規表現クエリのマッチとスコア) |

フィルターと `MustNot` 句はスコアを使わずマッチだけを使うので、どの方法でも、マッチするすべてのタームが対象になります。

Issue #1740 までは、`max_expansions` や `with_rewrite_method` の指定にかかわらず、マッチするすべてのタームを、係数なしでスコアの対象にしていました。

### WildcardQuery

ワイルドカードパターンを使用してタームをマッチングします。

```rust
use laurus::lexical::query::wildcard::WildcardQuery;

// '?' matches exactly one character, '*' matches zero or more
let query = WildcardQuery::new("filename", "*.pdf")?;
let query = WildcardQuery::new("body", "pro*")?;
let query = WildcardQuery::new("body", "col?r")?;  // matches "color", not "colour"
let query = WildcardQuery::new("body", r"why\?")?;  // matches "why?"
```

パターンは Lucene の `WildcardQuery` と同じに読みます。`*` と `?` は、改行を含むどの文字にもマッチします。バックスラッシュは、次の 1 文字をどの文字でもそのままの文字にします（`\*`、`\?`、`\\` は `*`、`?`、`\` に、`\d` は `d` にマッチします）。パターンの末尾のバックスラッシュは、バックスラッシュそのものです。ほかの文字は、`.` や `+` も含めてその文字にマッチするので、正規表現の大きさの上限を超えるほど長いパターン（たとえば `?` が 10 万個）を除き、エラーになるパターンはありません。Issue #1745 までは、バックスラッシュが次の文字と一緒に正規表現に渡り、`te\dt` が `te1t` にマッチし、`a\pb` はエラーになっていました。

### PrefixQuery

特定のプレフィックスで始まるタームを含むドキュメントをマッチングします。

```rust
use laurus::lexical::query::prefix::PrefixQuery;

// Find documents where "body" contains terms starting with "pro"
// This matches "programming", "program", "production", etc.
let query = PrefixQuery::new("body", "pro");
```

### RegexpQuery

正規表現パターンにマッチするタームを含むドキュメントをマッチングします。

```rust
use laurus::lexical::query::regexp::RegexpQuery;

// Find documents where "body" contains terms matching the regex
let query = RegexpQuery::new("body", "^pro.*ing$")?;

// Match version-like patterns
let query = RegexpQuery::new("version", r"^v\d+\.\d+")?;
```

> **注意:** `RegexpQuery::new()` は `Result` を返します。正規表現パターンは構築時にバリデーションされ、無効なパターンの場合はエラーが返されます。

#### ワイルドカード・プレフィックス・正規表現クエリのマッチとスコア

`WildcardQuery`・`PrefixQuery`・`RegexpQuery` は、マッチするタームがいくつあっても、そのどれかを含むすべてのドキュメントにマッチします。Issue #1740 までは、文書頻度の低い 50 個のタームのドキュメントだけが見つかり、`MustNot` 句としてもそれだけを除外していました。

スコアは Lucene と同じく**定数スコア（constant scoring）**で、どのマッチもクエリのブースト（デフォルト `1.0`）になります。タームの頻度は、パターンにどれだけ合うかをほとんど表さないためです。`with_rewrite_method` で別のスコアの付け方を選べます。

| `RewriteMethod` | スコア | そのクエリの検索で返るヒット |
| :--- | :--- | :--- |
| `ConstantScore`（デフォルト） | ブーストの値 | すべてのマッチ |
| `TopTermsScoring { max_expansions }` | マッチしたタームのうち文書頻度の高い `max_expansions` 個の BM25 | そのどれかのタームを含むドキュメント |
| `TopTermsBlended { max_expansions }` | 同じ。ただし、どのタームも、その中で最も頻出するタームと同じ文書頻度とみなす | そのどれかのタームを含むドキュメント |
| `BooleanQuery` | マッチしたすべてのタームの BM25 | すべてのマッチ。コストはタームの数より速く増え、大きなセグメントでは 1 万個で数分かかります |

フィルターと `MustNot` 句はスコアを使わずマッチだけを使うので、どの方法でも、マッチするすべてのタームが対象になります。

検索のコストは、マッチするタームの数と、それを含むドキュメントの数に応じて増えます。`?` か `*` で始まるワイルドカードと、先頭の `^` の後に文字どおりの部分がない正規表現は、そのフィールドのすべてのタームと比べられます。

### NumericRangeQuery

数値フィールドの値が指定された範囲内にあるドキュメントをマッチングします。

```rust
use laurus::lexical::NumericRangeQuery;
use laurus::lexical::core::field::NumericType;

// Find documents where "price" is between 10.0 and 100.0 (inclusive)
let query = NumericRangeQuery::new(
    "price",
    NumericType::Float,
    Some(10.0),   // min
    Some(100.0),  // max
    true,         // include min
    true,         // include max
);

// Open-ended range: price >= 50
let query = NumericRangeQuery::new(
    "price",
    NumericType::Float,
    Some(50.0),
    None,     // no upper bound
    true,
    false,
);
```

数値範囲クエリは**定数スコア（constant scoring）**です。マッチしたすべての
ドキュメントは同じスコア（クエリの boost、デフォルト `1.0`。有限で
`0` 以上）を受け取ります。これは Lucene の `PointRangeQuery` の
セマンティクスに従うものです。マッチングには
セグメントに BKD tree があればそれを使用し、対象フィールドの BKD tree を持たない
セグメント（例: そのフィールドを持つドキュメントが 1 件もないセグメントや、
`indexed = false, stored = true` と設定されたフィールド）では、そのセグメントに
実際に存在する保存済みドキュメントのみを走査するフォールバックを使用します。
走査も BKD tree と同じく値を数として比べるので、負の値と負の境界も一致し、
`-0.0` は `0.0` と等しく、`NaN` の値はどの範囲にも一致しません。

境界も `f64` として比べます。下限の `f64::NEG_INFINITY` と上限の
`f64::INFINITY` は、`None` と同じ文書に一致します。`NaN` の境界はどの値にも
一致しないので、クエリ、フィルタ、ほかのクエリの節のどれとして渡しても、
検索が `InvalidArgument` で失敗します。`NumericType::Integer` では、境界は
`i64` に切り捨てられ、無限大は `i64::MIN` か `i64::MAX` になります。

### DateTimeRangeQuery

`DateTime` フィールドの値が指定された範囲内にあるドキュメントをマッチングします。

```rust
use chrono::{TimeZone, Utc};
use laurus::lexical::DateTimeRangeQuery;

// Find documents where "created_at" is within 2024 (both bounds inclusive)
let query = DateTimeRangeQuery::between(
    "created_at",
    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
    Utc.with_ymd_and_hms(2024, 12, 31, 23, 59, 59).unwrap(),
);

// Open-ended range: created_at > 2024-06-15T12:00:00Z (exclusive)
let query = DateTimeRangeQuery::after(
    "created_at",
    Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap(),
);

// From textual bounds, using the same literal grammar as the query DSL
let query = DateTimeRangeQuery::from_literals(
    "created_at",
    Some("2024-01-01"),            // min (date only = midnight UTC)
    Some("2024-12-31T23:59:59Z"),  // max
    true,                          // include min
    true,                          // include max
)?;
```

`new(field, lower, upper, lower_inclusive, upper_inclusive)` は境界を
`Option<DateTime<Utc>>` で受け取ります。片側だけのヘルパーとして
`on_or_after`・`before`・`on_or_before` もあり、`.with_boost()` で定数スコア
（有限で `0` 以上）を設定します。クエリはフィールドの 1 次元 BKD ポイント
（マイクロ秒の小数部を持つ秒。秒未満の境界も正確に比較されます）に対して
評価され、定数スコアと保存済み
ドキュメントへのフォールバックは `NumericRangeQuery` と同じです。`from_literals` は
RFC 3339、オフセットなしの `YYYY-MM-DDTHH:MM:SS[.fff]`（UTC）、`YYYY-MM-DD`
（その日の 0 時 UTC）を受け付け、DSL の `created_at:[2024-01-01 TO 2024-12-31]` が
構築するクエリと同一です（[Query DSL](../query_dsl.md#範囲クエリ) 参照）。
`DateTime` フィールドに対してエポック秒を境界とする `NumericRangeQuery` も
従来どおり動作します。

### GeoDistanceQuery / GeoBoundingBoxQuery

2D 地理座標（WGS84 緯度・経度）に基づいてドキュメントをマッチングします。

```rust
use laurus::lexical::query::geo::{GeoBoundingBoxQuery, GeoDistanceQuery};

// Find documents within 10 km (= 10 000 m) of Tokyo Station (35.6812, 139.7671)
let query = GeoDistanceQuery::within_radius("location", 35.6812, 139.7671, 10_000.0)?; // distance in metres

// Find documents within a bounding box (min_lat, min_lon, max_lat, max_lon)
let query = GeoBoundingBoxQuery::within_bounding_box(
    "location",
    35.0, 139.0,  // min (lat, lon)
    36.0, 140.0,  // max (lat, lon)
)?;
```

どちらのクエリも距離ベースでスコアリングします（近いドキュメントほど高スコア。
円の中心、またはボックスの中心からの線形減衰）。マッチングにはセグメントに
BKD tree があればそれを使用し、座標は tree 自体から直接読み取ります —
そのためインデックスのみのフィールド（`indexed = true, stored = false`）でも
保存済みドキュメントなしで動作します。対象フィールドの BKD tree を持たない
セグメント（例: そのフィールドを持つドキュメントが 1 件もないセグメントや、
`indexed = false, stored = true` と設定されたフィールド）では、そのセグメントに
実際に存在する保存済みドキュメントのみを走査するフォールバックを使用します。

`GeoDistanceQuery` は球面上で正確に動作します。haversine による判定の前に、
円を球冠（spherical cap）の緯度・経度範囲で囲い込むため、日付変更線
（経度 ±180°）をまたぐ円、極を含む円、高緯度の大きな円でも、半径内の
ドキュメントをすべてマッチします。`GeoBoundingBoxQuery` は、端を含む緯度・経度の
矩形です。`min_lon > max_lon` のときは、ボックスが日付変更線（経度 ±180°）を
またぎます（GeoJSON、RFC 7946 と同じ規約）。`within_bounding_box("location", -20.0,
175.0, -15.0, -175.0)` は、東経 175° から西経 175° までを覆います。端がちょうど
±180° のときは、もう一方の符号で保存された点（同じ経線）にもマッチし、幅や高さが
0 のボックスは、その線や点の上の点にマッチします。`min_lat > max_lat` はエラーです。

### GeoPolygonQuery

`Geo` フィールドの点が多角形（穴を持てる）の中にあるドキュメントにマッチします
（Issue #1596）。

```rust
use laurus::lexical::query::geo::{GeoPolygon, GeoPolygonQuery};

// 正方形の穴のある正方形: 最初が外周の環、続くのが穴で、
// それぞれ (lat, lon) の並び（閉じていてもいなくてもよい）
let query = GeoPolygonQuery::within_polygon(
    "location",
    &[(35.0, 139.0), (35.0, 140.0), (36.0, 140.0), (36.0, 139.0)],
    &[&[(35.4, 139.4), (35.4, 139.6), (35.6, 139.6), (35.6, 139.4)]],
)?;

// GeoPoint の GeoPolygon からも作れる
let polygon = GeoPolygon::new(exterior, holes)?;
let query = GeoPolygonQuery::new("location", polygon).with_boost(2.0);
```

Lucene の多角形のクエリと同じく、どの一致も boost の値（指定しなければ `1.0`）
の点数になり、複数の値を持つフィールドは、どれか 1 点が中にあればマッチします。
辺は緯度・経度の上の直線です。Lucene と同じく、外周の辺と頂点の上の点は
マッチし、穴の辺と頂点の上の点はマッチしません。

辺は 2 つの頂点の間の近いほう、つまり経度の差が 180 度以下になる向きを通ります。
170 度から -170 度への辺は幅 20 度で日付変更線をまたぐので、太平洋の上の多角形を
分ける必要はなく、±180° の上の点はどちらの符号で保存されていてもマッチします。
180 度より広い辺には、間に頂点を置く必要があります。`GeoPolygon::new` は、
異なる点が 3 つ未満の環、範囲外の座標、どちらの向きにも通れる経度ちょうど
180 度の辺、極を囲む環を拒否します。自己交差は、Lucene と同じく調べません。

BKD ツリーを使うときは、多角形の中のセルを点ごとに判定せずにまとめて取り、
辺が通るセルの点だけを 1 つずつ判定します。そのフィールドの BKD ツリーを
持たないセグメントは、ほかの地理クエリと同じく保存済みドキュメントを走査します。

### GeoPolylineQuery

`Geo` フィールドの点が、頂点を順につないだ折れ線から指定の距離以内にある
ドキュメントにマッチします。道路、線路、川、経路の近くを探すときに使います
（Issue #1596）。

```rust
use laurus::lexical::query::geo::{GeoPolyline, GeoPolylineQuery};

// 東京から名古屋を経て大阪へ向かう経路から 2 km 以内。頂点は (lat, lon)
let query = GeoPolylineQuery::within_distance(
    "location",
    &[(35.68, 139.76), (35.17, 136.91), (34.69, 135.50)],
    2_000.0,
)?;

// GeoPoint の GeoPolyline からも作れる
let polyline = GeoPolyline::new(vertices)?;
let query = GeoPolylineQuery::new("location", polyline, 2_000.0).with_boost(2.0);
```

距離は `GeoDistanceQuery` と同じく地表の距離です。各辺は 2 つの頂点の間の
短いほうの大円の弧で、点から弧の最も近い点まで（垂線の足が弧の外なら近い端
まで）を測るので、端と角は丸くなります。多角形の辺は緯度・経度の上の直線
ですが、折れ線の辺は大円に沿うので、`(10, 0)` から `(10, 170)` への弧は
中ほどで北緯約 63.7 度まで上がります。点数は `GeoDistanceQuery` と同じく、
ドキュメントの最も近い点による `1 - d / distance_m` で、頂点が 1 つなら
`GeoDistanceQuery` と同じドキュメントと点数になります。`GeoPolyline::new`
は 1 つ以上の頂点を受け（同じ頂点が続いてもよく、長さ 0 の辺はその点）、
範囲外の座標と、対蹠点（とそこから約 640 m 以内）を結ぶ辺を拒否します。
対蹠点を結ぶ最短の弧は 1 つに決まらないためです。日付変更線をまたぐ辺に
特別な書き方は要りません。

BKD ツリーを使うときは、どの辺の範囲とも離れたセルを刈ります。辺の範囲は、
頂点の間にある弧の最も高い緯度を含めた緯度と、短いほうの経度を、距離だけ
広げたものです。残ったセルの点は、点数に距離が要るので 1 つずつ測ります。
そのフィールドの BKD ツリーを持たないセグメントは、保存済みドキュメントを
走査します。

### Geo3dDistanceQuery / Geo3dBoundingBoxQuery / Geo3dNearestQuery

3 種類のクエリが、ECEF 直交座標（メートル）にバックアップされた 3D の
`Geo3d` フィールドを対象とします。高度が意味を持つ用途や、2D の `Geo`
フィールドでは極特異点が問題になるケースで使ってください。座標系・WGS84
変換ヘルパー・実例については [3D 地理検索](../geo3d.md) を参照してください。

```rust
use laurus::GeoEcefPoint;
use laurus::lexical::query::geo3d::{
    Geo3dDistanceQuery, Geo3dBoundingBoxQuery, Geo3dNearestQuery,
};

let centre = GeoEcefPoint::new(-3_955_182.0, 3_350_553.0, 3_700_276.0);

// 球: `centre` から 5 km 以内のドキュメント
let q = Geo3dDistanceQuery::new("position", centre, 5_000.0);

// 軸並行 3D バウンディングボックス（コンストラクタが軸ごとに min ≤ max を検証）
let min = GeoEcefPoint::new(-4_000_000.0, 3_300_000.0, 3_650_000.0);
let max = GeoEcefPoint::new(-3_900_000.0, 3_400_000.0, 3_750_000.0);
let q = Geo3dBoundingBoxQuery::new("position", min, max)?;

// k-NN: 最も近い 10 件、半径スケジュールをカスタマイズ
let q = Geo3dNearestQuery::new("position", centre, 10)
    .with_initial_radius(500.0)
    .with_max_radius(1_000_000.0);
```

| クエリ | スコア |
| :--- | :--- |
| `Geo3dDistanceQuery` | `1 - distance / radius` を `[0, 1]` にクランプ |
| `Geo3dBoundingBoxQuery` | マッチした全ドキュメントで定数 `1.0` |
| `Geo3dNearestQuery` | 最も近いヒットが `1.0`、返却された集合内で最も遠いヒットが `0.0` となるよう正規化 |

geo3d クエリはフィールドがインデックスされていること（`indexed = true`、
デフォルト）を必要とします。これらは完全にフィールドの BKD tree 上で動作し、
BKD tree を持たないセグメントからはヒットを返しません — 3D クエリには
保存済みドキュメントへのフォールバックはありません。

### Geo3dPolylineQuery

`Geo3d` フィールドの点が、頂点を順につないだ線分から指定の距離以内にある
ドキュメントにマッチします。飛行経路、トンネル、ケーブルの近くを探すときに
使います（Issue #1596）。

```rust
use laurus::lexical::query::geo3d::{Geo3dPolyline, Geo3dPolylineQuery};
use laurus::util::ecef::wgs84_to_ecef;

// 高度 1,000 m の飛行経路から 500 m 以内
let path = Geo3dPolyline::new(vec![
    wgs84_to_ecef(35.0, 139.0, 1_000.0),
    wgs84_to_ecef(35.4, 139.6, 1_000.0),
])?;
let query = Geo3dPolylineQuery::new("position", path, 500.0);
```

距離は ECEF のユークリッド距離で、点数は `Geo3dDistanceQuery` と同じく、
ドキュメントの最も近い点による `1 - distance / radius`（`[0, 1]` に収める）
です。頂点が 1 つなら `Geo3dDistanceQuery` と同じドキュメントと点数に
なります。辺は弦なので、地表の 2 点を結ぶ辺は地表より下を通り、長さ `L` の
辺の中ほどで約 `L² / 8R` 沈みます（100 km で 196 m、1,000 km で約 20 km）。
長い距離を地表に沿わせたい経路には、頂点を増やすか、`Geo` フィールドと
`GeoPolylineQuery` を使います。`Geo3dPolyline::new` は 1 つ以上の頂点を
受け（同じ頂点が続いてもよい）、NaN と無限大の座標を拒否します。ほかの 3D
クエリと同じくフィールドの BKD ツリーだけを読み、どの線分も、距離だけ四方に
広げたセルと交わらなければ、そのセルを刈ります。

### SpanQuery

ドキュメント内のタームの近接度に基づいてマッチングします。`SpanTermQuery` と `SpanNearQuery` を使用して近接クエリを構築します。

```rust
use laurus::lexical::query::span::{SpanQuery, SpanQueryWrapper, SpanTermQuery, SpanNearQuery};

// Find documents where "quick" appears near "fox" (within 3 positions)
let near = SpanNearQuery::new(
    "body",
    vec![
        Box::new(SpanTermQuery::new("body", "quick")) as Box<dyn SpanQuery>,
        Box::new(SpanTermQuery::new("body", "fox")) as Box<dyn SpanQuery>,
    ],
    3,    // slop (max distance between terms)
    true, // in_order (terms must appear in order)
);

// The query to search: a SpanQueryWrapper, which carries the boost
let query = SpanQueryWrapper::new(Box::new(near)).boost(2.0);
```

span クエリは `SpanQueryWrapper` で包んで検索し、ブーストもそこに付けます。このブーストはスコアに掛かります。中の span クエリ（`SpanTermQuery`、`SpanNearQuery`、`SpanContainingQuery`、`SpanWithinQuery`）は、位置を出すだけなので、Lucene と同じく自分のブーストを持ちません。これらには何も変えない `boost` の setter がありましたが、Issue #1569 で削除しました。

## スコアリング

Lexical 検索結果は **BM25** を使用してスコアリングされます。スコアは、ドキュメントがクエリに対してどの程度関連性があるかを反映します。

- ドキュメント内のターム頻度が高いほどスコアが上昇する
- インデックス全体でタームが希少なほどスコアが上昇する
- 短いドキュメントは長いドキュメントに対してブーストされる

ワイルドカード・プレフィックス・正規表現・範囲クエリは、代わりに定数スコアです。どのマッチもクエリのブーストの値になります。

スコアが同じヒットは、内部ドキュメント ID の昇順（ドキュメントが書き込まれた順）に返ります。そのため、インデックスが変わらない間は、検索のページどうしが重なりません（Issue #1740）。

### フィールドブースト

特定のフィールドをブーストして関連性に影響を与えることができます。

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::search::searcher::LexicalSearchQuery;

let request = SearchRequestBuilder::new()
    .lexical_query(LexicalSearchQuery::Obj(Box::new(query)))
    .add_field_boost("title", 2.0)  // title matches count double
    .add_field_boost("body", 1.0)
    .build();
```

## Lexical 検索オプション（SearchRequestBuilder 経由）

Lexical 検索の動作パラメータは `SearchRequestBuilder` のメソッドで設定します。これらは `SearchRequest` の `lexical_options` フィールドに格納されます。

| オプション | デフォルト | 説明 |
| :--- | :--- | :--- |
| `field_boosts` | 空 | フィールドごとのスコア倍率。各倍率は有限で `0` 以上（`-0.0` は拒否）。`0` では一致した文書がスコア `0` で残る（`CombinedFields` のマルチフィールドクエリは今どおり `0` を拒否する） |
| `min_score` | 0.0 | 最小スコア閾値。有限でなければならない。検索は負のブーストを拒否するので、スコアは負にならず、負の閾値は `0.0` と同じ |
| `timeout_ms` | None | 検索の時間予算（ミリ秒、下記の注記を参照） |
| `parallel` | false | セグメント間の並列検索を有効にする |
| `sort_by` | `Score` | 関連性スコアでソート、またはフィールドでソート（`asc` / `desc`） |
| `highlight` | None | 各ヒットの `SearchResult::highlights` に含めるフィールドと設定（[ハイライト](../../laurus/highlighting.md)参照） |

フィールドソート検索（`sort_by: Field { .. }`）は常に全候補ドキュメントを走査します。
スコアソートで利用可能な block-max ベースの早期終了とは異なり、フィールドソートには早期終了が
ありません。これにより返るヒットは早期終了による近似ではなく真の top-K であることが保証され、
`track_total_hits` で総数を求めたときは、セグメントがいくつあっても上限より下の総数が正確になります。
スコアソートの検索も、上限に届くまで打ち切らないことで、上限まで正確に数えます
（[総ヒット数](../../laurus/engine.md#総ヒット数)参照）。

`timeout_ms` を設定すると、時間予算は検索の**実行中に協調的に**適用されます。スキャンループ
（マルチセグメント fanout の各セグメントを含む）が定期的に deadline をチェックし、予算を使い切った時点で
即座に中断してタイムアウトエラーを返します（従来のようにクエリを完走してから判定するのではありません）。
チェックはバッチ化されている（数千ドキュメント走査ごと）ため、`timeout_ms` 未設定時や予算内に収まる
通常ケースでは計測可能なオーバーヘッドはありません。

### ビルダーメソッド

`SearchRequestBuilder` を使用してクエリとオプションを設定します。

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::{LexicalSearchQuery, SortField, SortOrder};

let request = SearchRequestBuilder::new()
    .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new("body", "rust"))))
    .lexical_min_score(0.5)
    .lexical_timeout_ms(5000)
    .lexical_parallel(true)
    .sort_by(SortField::Field { name: "date".to_string(), order: SortOrder::Desc })
    .add_field_boost("title", 2.0)
    .add_field_boost("body", 1.0)
    .limit(20)
    .build();
```

`.highlight(fields)` と `.highlight_config(config)` は各ヒットのハイライト済みフラグメントを要求します。詳細は[ハイライト](../../laurus/highlighting.md)を参照してください。

## Query DSL の使用

プログラマティックにクエリを構築する代わりに、テキストベースの Query DSL を使用できます。

```rust
use laurus::lexical::QueryParser;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use std::sync::Arc;

let analyzer = Arc::new(StandardAnalyzer::default());
let parser = QueryParser::new(analyzer).with_default_field("body");

// Simple term
let query = parser.parse("rust")?;

// Boolean
let query = parser.parse("rust AND programming")?;

// Phrase
let query = parser.parse("\"machine learning\"")?;

// Field-specific
let query = parser.parse("title:rust AND body:programming")?;

// Fuzzy
let query = parser.parse("programing~2")?;

// Range
let query = parser.parse("year:[2020 TO 2024]")?;

// DateTime range (date-only bounds are midnight UTC)
let query = parser.parse("created_at:[2024-01-01 TO 2024-12-31]")?;
```

完全な構文リファレンスは [Query DSL](../query_dsl.md) を参照してください。

## フィルタ結果キャッシュ（Filter Result Cache）

テナント・カテゴリ・ステータスフラグなどの filter 句は、多数のリクエストで繰り返し
再利用されます。毎回ゼロから評価すると同じ posting list を何度も走査することになります。
laurus は filter がマッチする**ドキュメント ID の集合**をメモ化し、繰り返しの filter を
posting 走査ではなく単一のルックアップで済ませます。

- **スナップショット連動・自己無効化（snapshot-scoped, self-invalidating）。**
  キャッシュは reader 上に存在し、reader は `commit()` / `optimize()` / `refresh()` の
  たびに再構築されます。各 reader は point-in-time スナップショットなので、手動の無効化は
  不要です。インデックス変更後の次の検索は空のキャッシュから始まり、常にコミット済みの
  データを反映します。
- **スコア非依存（score-independent）。** filter は relevance に影響せずドキュメントを
  選別するだけなので、キャッシュ値は単なる doc-id 集合（Roaring ビットマップ）です。
  [ハイブリッド検索 / フィルタ検索](hybrid_search.md) の `filter_query` に使われ、
  lexical 側・vector 側の両方に供給されます。
- **BooleanQuery 内でも再利用。** `BooleanQuery` 内の `Occur::Filter` 句
  （例: `must(user_query).filter(tenant_filter)`）も、posting 再走査ではなくキャッシュから
  マッチ集合を取得します。マルチセグメント時の per-segment fanout 経路でも有効で、
  各セグメントはキャッシュの集合から自分が持つドキュメントだけを取り出します。
  Issue #1743 までは各セグメントが集合の全体を使っていたので、Must 句の無い filter の
  `BooleanQuery` は 1 つのドキュメントをセグメントの数だけ返し、`MustNot` 句はその
  セグメントのドキュメントしか除いていませんでした。
- **構造的に安全（safe by construction）。** canonical なキーを持つクエリのみキャッシュ
  されます。Term・Phrase・Prefix・Wildcard・Regexp・Fuzzy・Range・Geo・Geo3d クエリは
  キャッシュ可能で、キャッシュ可能な句のみで構成され、かつ少なくとも 1 つの正句
  （Must / Should / Filter）を持つ BooleanQuery も同様です。正句のない BooleanQuery・
  Span クエリ・マルチフィールドクエリは毎回新規評価され（キャッシュされず）、結果は常に
  正しくなります。

キャッシュはデフォルトで有効です。インデックス設定で調整・無効化できます。無効にすると、
filter 句は検索のたびに各セグメントの posting で評価されます。

```rust
use laurus::lexical::store::config::LexicalIndexConfig;

let config = LexicalIndexConfig::builder()
    .query_filter_cache_capacity(4096) // スナップショットあたりのエントリ数。0 で無効化
    .build();
```

## パース済みクエリキャッシュ（Parsed Query Cache）

DSL 文字列での検索（`SearchRequest::from_dsl` や `LexicalSearchQuery::Dsl`）は、毎回 pest
文法でパースし、語を analyzer で再トークン化します。オートコンプリートや人気クエリでは同じ
文字列が繰り返されるため、laurus は `DSL 文字列 → パース済みクエリ` をメモ化します。繰り返し
の DSL 文字列は一度だけパースされ、以降は再利用されます（パース済みクエリツリーの安価な複製）。

フィルタキャッシュと同様に**スナップショット連動**です。キャッシュは searcher 上に存在し、
`commit()` / `optimize()` / `refresh()` のたびに再構築されます。analyzer と default fields は
その searcher で固定なので DSL 文字列のみがキーになり、スキーマ/analyzer 変更時は空の新しい
キャッシュになります。デフォルトで有効。インデックス設定で調整・無効化できます。

```rust
use laurus::lexical::store::config::LexicalIndexConfig;

let config = LexicalIndexConfig::builder()
    .parsed_query_cache_capacity(2048) // スナップショットあたりのエントリ数。0 で無効化
    .build();
```

## Posting キャッシュ（Posting Cache）

語の評価ではセグメントの `.post` ファイルから posting list を読み、デコードします
（varint doc-id、削除フィルタ、skip table）。キャッシュがないと同じ語のクエリごとに read +
デコードを繰り返し、クラウド/リモートストレージでは read が支配的になります。各セグメント
リーダーはデコード済み・削除フィルタ後の posting list を小さくキャッシュし、スナップショット
内の同一 `(field, term)` 参照を再利用します。

セグメントリーダーの削除は生存中に変わらないので、キャッシュ済みリストは常にその削除と整合
します。commit がセグメントを変えなければリーダーはそのまま使われ、キャッシュも残ります。削除が
変わったセグメントには、空キャッシュの新しいセグメントリーダーが作られます（[commit 後の読み込み](../indexing/lexical_indexing.md#commit-後の読み込み)
を参照）。キャッシュは
**byte-budget で上限制御**され（posting list はサイズ分散が大きい）、予算超過で
least-recently-used リストを退避し、予算全体より大きい単一リストはキャッシュしません。
デフォルトで有効です。セグメントごとのキャッシュがそれぞれ `max_cache_memory` の予算（デフォルト
128 MiB。reader の term-info キャッシュの上限も兼ねる）を持つため、最悪の使用量はセグメント数に比例して
増えます。インデックス設定で制御できます。

```rust
use laurus::lexical::store::config::LexicalIndexConfig;

let config = LexicalIndexConfig::builder()
    .enable_posting_cache(false) // 完全に無効化
    .build();

let config = LexicalIndexConfig::builder()
    .max_cache_memory(256 * 1024 * 1024) // またはセグメントごとの予算（バイト）を変更
    .build();
```

## 次のステップ

- 意味的類似性検索: [Vector 検索](vector_search.md)
- Lexical + Vector の組み合わせ: [ハイブリッド検索](hybrid_search.md)
- chunk に分けた文書を 1 回ずつ返す: [検索結果のグループ化](result_grouping.md)
- DSL の完全な構文リファレンス: [Query DSL](../query_dsl.md)
