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

let results = engine.search(request).await?;
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

フレーズクエリは、ターム位置情報が格納されている必要があります（`TextOption` のデフォルト設定）。

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
| `MustNot` | ドキュメントがマッチしてはならない | `-term` または `NOT` |
| `Filter` | 必ずマッチする必要があるが、スコアには影響しない | （DSL に相当するものなし） |

### FuzzyQuery

指定された編集距離（レーベンシュタイン距離）内のタームをマッチングします。

```rust
use laurus::lexical::query::fuzzy::FuzzyQuery;

// Find documents matching "programing" within edit distance 2
// This will match "programming", "programing", etc.
let query = FuzzyQuery::new("body", "programing");  // default max_edits = 2
```

### WildcardQuery

ワイルドカードパターンを使用してタームをマッチングします。

```rust
use laurus::lexical::query::wildcard::WildcardQuery;

// '?' matches exactly one character, '*' matches zero or more
let query = WildcardQuery::new("filename", "*.pdf")?;
let query = WildcardQuery::new("body", "pro*")?;
let query = WildcardQuery::new("body", "col?r")?;  // matches "color" and "colour"
```

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
ドキュメントは同じスコア（クエリの boost、デフォルト `1.0`）を受け取ります。
これは Lucene の `PointRangeQuery` のセマンティクスに従うものです。マッチングには
セグメントに BKD tree があればそれを使用し、対象フィールドの BKD tree を持たない
セグメント（例: そのフィールドを持つドキュメントが 1 件もないセグメントや、
`indexed = false, stored = true` と設定されたフィールド）では、そのセグメントに
実際に存在する保存済みドキュメントのみを走査するフォールバックを使用します。

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

### SpanQuery

ドキュメント内のタームの近接度に基づいてマッチングします。`SpanTermQuery` と `SpanNearQuery` を使用して近接クエリを構築します。

```rust
use laurus::lexical::query::span::{SpanQuery, SpanTermQuery, SpanNearQuery};

// Find documents where "quick" appears near "fox" (within 3 positions)
let query = SpanNearQuery::new(
    "body",
    vec![
        Box::new(SpanTermQuery::new("body", "quick")) as Box<dyn SpanQuery>,
        Box::new(SpanTermQuery::new("body", "fox")) as Box<dyn SpanQuery>,
    ],
    3,    // slop (max distance between terms)
    true, // in_order (terms must appear in order)
);
```

## スコアリング

Lexical 検索結果は **BM25** を使用してスコアリングされます。スコアは、ドキュメントがクエリに対してどの程度関連性があるかを反映します。

- ドキュメント内のターム頻度が高いほどスコアが上昇する
- インデックス全体でタームが希少なほどスコアが上昇する
- 短いドキュメントは長いドキュメントに対してブーストされる

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
| `field_boosts` | 空 | フィールドごとのスコア倍率 |
| `min_score` | 0.0 | 最小スコア閾値 |
| `timeout_ms` | None | 検索の時間予算（ミリ秒、下記の注記を参照） |
| `parallel` | false | セグメント間の並列検索を有効にする |
| `sort_by` | `Score` | 関連性スコアでソート、またはフィールドでソート（`asc` / `desc`） |
| `highlight` | None | 各ヒットの `SearchResult::highlights` に含めるフィールドと設定（[ハイライト](../../laurus/highlighting.md)参照） |

フィールドソート検索（`sort_by: Field { .. }`）は常に全候補ドキュメントを走査します。
スコアソートで利用可能な block-max ベースの早期終了とは異なり、フィールドソートには早期終了が
ありません。これにより返るヒットは早期終了による近似ではなく真の top-K であることが保証され、
`total_hits` も走査打ち切りによる件数ではなく真のマッチ数を反映します。

`timeout_ms` を設定すると、時間予算は検索の**実行中に協調的に**適用されます。スキャンループ
（マルチセグメント fanout の各セグメントを含む）が定期的に deadline をチェックし、超過した時点で
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
  マッチ集合を取得します。マルチセグメント時の per-segment fanout 経路でも有効です。
- **構造的に安全（safe by construction）。** canonical なキーを持つクエリのみキャッシュ
  されます。Term・Phrase・Prefix・Wildcard・Regexp・Fuzzy・Range・Geo・Geo3d クエリは
  キャッシュ可能で、キャッシュ可能な句のみで構成され、かつ少なくとも 1 つの正句
  （Must / Should / Filter）を持つ BooleanQuery も同様です。正句のない BooleanQuery・
  Span クエリ・マルチフィールドクエリは毎回新規評価され（キャッシュされず）、結果は常に
  正しくなります。

キャッシュはデフォルトで有効です。インデックス設定で調整・無効化できます。

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

セグメントはスナップショット内で immutable なので、キャッシュ済みリストは常にその削除と整合
します。commit すると空キャッシュの新しいセグメントリーダーが構築されます。キャッシュは
**byte-budget で上限制御**され（posting list はサイズ分散が大きい）、予算超過で
least-recently-used リストを退避し、予算全体より大きい単一リストはキャッシュしません。
デフォルトで有効で `max_cache_memory` の予算を共有します。インデックス設定で制御できます。

```rust
use laurus::lexical::store::config::LexicalIndexConfig;
use laurus::lexical::index::config::InvertedIndexConfig;

let mut inverted = InvertedIndexConfig::default();
inverted.enable_posting_cache = false;        // 完全に無効化
inverted.max_cache_memory = 256 * 1024 * 1024; // またはキャッシュ予算（バイト）を変更
let config = LexicalIndexConfig::Inverted(inverted);
```

## 次のステップ

- 意味的類似性検索: [Vector 検索](vector_search.md)
- Lexical + Vector の組み合わせ: [ハイブリッド検索](hybrid_search.md)
- DSL の完全な構文リファレンス: [Query DSL](../query_dsl.md)
