# Vector 検索

Vector 検索は、意味的類似性によってドキュメントを検索します。キーワードのマッチングではなく、ベクトル空間におけるクエリの意味とドキュメントエンベディングを比較します。

## 基本的な使い方

### Builder API

```rust
use laurus::SearchRequestBuilder;
use laurus::vector::VectorSearchRequestBuilder;

let request = SearchRequestBuilder::new()
    .vector_query(
        VectorSearchRequestBuilder::new()
            .add_text("embedding", "systems programming language")
            .limit(10)
            .build()
    )
    .build();

let results = engine.search(request).await?;
```

`add_text()` メソッドはテキストをクエリペイロードとして格納します。検索時に、エンジンが設定されたエンベッダーを使用してテキストをエンベディングし、ベクトルインデックスを検索します。

### Query DSL

```rust
use laurus::vector::VectorQueryParser;

let parser = VectorQueryParser::new(embedder.clone())
    .with_default_field("embedding");

let request = parser.parse(r#"embedding:"systems programming""#).await?;
```

## VectorSearchRequestBuilder

Builder API により、きめ細かな制御が可能です。

```rust
use laurus::vector::VectorSearchRequestBuilder;
use laurus::vector::store::request::QueryVector;

let request = VectorSearchRequestBuilder::new()
    // Text query (will be embedded at search time)
    .add_text("text_vec", "machine learning")

    // Or use a pre-computed vector directly
    .add_vector("embedding", vec![0.1, 0.2, 0.3, /* ... */])

    // Search parameters
    .limit(20)

    .build();
```

### メソッド

| メソッド | 説明 |
| :--- | :--- |
| `add_text(field, text)` | 特定のフィールドに対するテキストクエリを追加（検索時にエンベディング） |
| `add_vector(field, vector)` | 特定のフィールドに対する事前計算済みクエリベクトルを追加 |
| `add_vector_with_weight(field, vector, weight)` | 明示的なウェイトを持つ事前計算済みベクトルを追加 |
| `add_payload(field, payload)` | エンベディング対象の汎用 `DataValue` ペイロードを追加 |
| `add_bytes(field, bytes, mime)` | バイナリペイロードを追加（例: マルチモーダル用の画像バイト） |
| `field(name)` | 検索を特定のフィールドに制限 |
| `fields(names)` | 検索を複数のフィールドに制限 |
| `limit(n)` | 結果の最大件数（デフォルト: 10） |
| `score_mode(VectorScoreMode)` | スコア結合モード（`WeightedSum`、`MaxSim`、`LateInteraction`） |
| `min_score(f32)` | 最小スコア閾値（デフォルト: 0.0） |
| `overfetch(f32)` | オーバーフェッチ係数（デフォルト: 2.0、#675）。各クエリは `ceil(limit × overfetch)` 件の候補を取得し、融合後に `limit` へ切り詰める。`<= 1.0` で無効化 |
| `build()` | `VectorSearchRequest` を構築 |

## マルチフィールド Vector 検索

単一のリクエストで複数のベクトルフィールドを横断して検索できます。

```rust
let request = VectorSearchRequestBuilder::new()
    .add_text("text_vec", "cute kitten")
    .add_text("image_vec", "fluffy cat")
    .build();
```

各クエリ句はベクトルを生成し、対応するフィールドに対して検索されます。結果は設定されたスコアモードで結合されます。

### スコアモード

| モード | 説明 |
| :--- | :--- |
| `WeightedSum`（デフォルト） | すべてのクエリ句にわたる（類似度 * ウェイト）の合計 |
| `MaxSim` | クエリ句間の最大類似度スコア |
| `LateInteraction` | ColBERT スタイルの Late Interaction スコアリング |

### マルチベクトル検索の並列実行

複数のクエリベクトル（ColBERT スタイルの late interaction、マルチベクトル
MaxSim、ensemble reranker など）を含むリクエストに対して、Laurus は
[rayon][rayon-crate] を使ってクエリごとの類似度検索を並列に実行します。

動作:

- 並列化は `VectorIndexSearcher::search_batch` トレイトメソッドの
  デフォルト実装内に置かれています。ネイティブビルド（`native` feature が
  デフォルトで有効）では、`queries.len()` が searcher の
  `parallel_threshold`（デフォルト `4`、 searcher 単位で override 可能）
  に達した時点で、HNSW / Flat / IVF のクエリごとの検索が rayon のグローバル
  スレッドプール上で並列実行されます。これを下回るとシリアルループのほうが
  速くなります（rayon のディスパッチオーバーヘッド ~1-2 µs が、単一クエリの
  50-200 µs を上回りやすいため）。
- `wasm32` ターゲットでは rayon が使用できないため、常にシリアルパスを
  通ります。
- 集約（スコアモードによるマージ）と最終ソートはクエリ並列フェーズの後に
  シリアル実行されます。スコアが同点の場合は `doc_id` の昇順でタイブレーク
  するため、rayon のワークスチール順序に依存しない決定的な結果になります。

外部 API（`VectorStore::search`、gRPC の `Search`、REST の
`POST /v1/search`、各言語バインディング）は一切変更されません。並列実行は
完全に内部最適化で、laurus をアップグレードするだけで有効になります。
speedup はホストの利用可能コア数に応じてスケールアップします（4 物理コア /
8 スレッドの HT 有効ラップトップ CPU では、B = 64 クエリ時にスループットが
およそ 2× 向上します。これは物理コア数と HT 共有による上限に近い値です）。

[rayon-crate]: https://docs.rs/rayon

### ブルートフォーススキャンの並列化

Flat と IVF インデックスは、グラフ探索ではなく総当たりの距離スキャンで候補を
ランク付けします。1 つのクエリの候補数が内部しきい値（`2048`）に達すると、
そのスキャンは [rayon][rayon-crate] のグローバルスレッドプールに分散されます。
これを下回るとシリアルループのほうが速くなります（rayon のジョブごとの
ディスパッチ ~1-2 µs が小さなスキャンを上回るため）。

これは上記のクエリ単位の並列化とは直交します。バッチはクエリ間で並列化し、
さらに各大規模クエリは同じプール上で自身のスキャンを並列化します。ワーク
スチールにより全体の並列度はプールサイズに制限されます（OS スレッドの
oversubscription は発生しません）。距離カーネルは副作用を持たないため、結果は
任意の順序で収集された後にソートされ、出力は決定的に保たれます。`wasm32`
（rayon なし）では常にシリアルです。

speedup はホストの物理コア数に応じてスケールし、大きな Flat インデックスや
広い IVF `n_probe` で最大になります。しきい値未満のスキャンは影響を受けません。

### IVF クラスタ選択

距離スキャンの前に、IVF クエリはまず *どの* クラスタをスキャンするかを選びます。
クエリを全セントロイドと照合し、最も近い `n_probe` 個を残します。プローブした
クラスタはこの後にマージされ類似度で再ランク付けされるため、セントロイドの
相対順序は無関係です。したがって最近傍 `n_probe` 個は、`K` 個全体の完全ソート
（`O(K log K)`）ではなく `O(K)` の部分選択（`select_nth_unstable_by`）で取得
します。削減効果はクラスタ数 `K` が大きいほど増え、`K = 2048` ではクエリ
あたりの粗選択ステップを約 18% 短縮しました（Issue #668）。セントロイド
スキャン自体はシリアルのままです。各セントロイドは単一の距離計算であり、
現実的なクラスタ数（`K ≈ √N`）では rayon へのディスパッチが節約分を上回る
ためです。

### ウェイト

DSL では `^` ブースト構文を使用するか、`QueryVector` の `weight` で各フィールドの寄与度を調整します。

```text
text_vec:"cute kitten"^1.0 image_vec:"fluffy cat"^0.5
```

これは、テキストの類似度が画像の類似度の 2 倍の重みを持つことを意味します。

vector フィールドのスキーマレベルの `base_weight`（Issue #1084）も同様に
機能しますが、クエリ単位の上書きではなくフィールド単位のデフォルト値
である点が異なります。上記のクエリ重みに乗算されるため、`base_weight`
と `^`/`weight` は組み合わさります。どちらも vector 側自体のスコアリング
にしか影響しないため、ハイブリッド検索の lexical-vs-vector のバランス
を変えることはできません。`FusionAlgorithm::RRF` はランクのみを見て元の
スコアを完全に無視しますし、`FusionAlgorithm::WeightedSum` は
`lexical_weight`/`vector_weight` を適用する前に各側を min-max 正規化
するため、フィールド単位の一様なスカラー倍は打ち消されます。両方の
重み付け機構が実際に影響するのは、1 回のクエリで複数の vector フィール
ドを対象にしたときの、フィールド間の相対的な優先度です。

### フィールドルーティング

マルチフィールドスキーマでは、各 vector フィールドが独自の HNSW graph を
持ちます。デフォルトではクエリは全 vector フィールドを検索しますが、指定し
たフィールドのみに限定すると、それ以外のフィールドの graph 探索を完全に
省略できます。

2 つのルーティング入力が、以下の優先順位で尊重されます。

1. **per-query** — `QueryVector.fields`。DSL パーサがクエリ句の指定する
   フィールドからこれを設定するため、`image_vec:"fluffy cat"` は
   `image_vec` のみを検索します。
2. **request-level** — `VectorSearchParams.fields`（セレクタのリスト）:
   - `Exact("image_vec")` — フィールド名の完全一致。
   - `Prefix("image_")` — 指定 prefix で始まる全フィールドに一致
     （インデックスのフィールド名から解決）。

どちらも未指定の場合は全フィールドを検索します（デフォルト）。フィールドに
ルーティングされたクエリは、そのフィールドに vector を持たないドキュメントを
返しません。

## フィルター付き Vector 検索

Lexical フィルターを適用して Vector 検索の結果を絞り込むことができます。

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::vector::VectorSearchRequestBuilder;

// Vector search with a category filter
let request = SearchRequestBuilder::new()
    .vector_query(
        VectorSearchRequestBuilder::new()
            .add_text("embedding", "machine learning")
            .build()
    )
    .filter_query(Box::new(TermQuery::new("category", "tutorial")))
    .limit(10)
    .build();

let results = engine.search(request).await?;
```

フィルタークエリはまず Lexical インデックス上で実行されて許可されるドキュメント ID のセットを特定し、その後 Vector 検索がそれらの ID に制限されます。

### filter-aware HNSW トラバーサル

HNSW フィールドでは、許可 ID セットは検索後に適用するだけでなく、graph 探索
そのものに渡されます。探索中、フロンティアは**すべての**近傍（非マッチの
近傍も含む）を経由して展開されるため、graph 上の非マッチ領域を横断して
マッチに到達できますが、結果セットに入るのはマッチするドキュメントのみです。

これは選択的フィルターで重要になります。単純な post-filter は最近傍の固定
`ef_search` ウィンドウだけを見てマッチしたものを残すため、マッチが希少だと
ウィンドウの外に完全に外れてしまい、到達可能なマッチが存在しても結果が実際
より大幅に少なく（時にはゼロに）なります。filter-aware トラバーサルは十分な
マッチを集めるまで探索を続け、非常に選択的なフィルターでは内部の訪問上限
（`ef_search` の定数倍）で latency を抑えます。

フィルターなしの経路は不変です（フィルターがなければ従来どおりの挙動）。
Flat / IVF フィールドは許可セットを*インライン*で適用します。ドキュメント ID が
セットに含まれない候補は距離カーネルの実行前にスキップされるため、選択的フィルター
では post-filter が払う無駄な距離計算を回避できます。いずれにせよスキャンは網羅的
なので recall は変わりません。store 側の post-filter は冗長な安全網として後段で
引き続き実行されます。

許可セットが `ef_search` より小さい場合、HNSW フィールドは graph 探索を完全に
スキップし、許可されたドキュメントを直接採点します。候補がこれほど少ないと
graph で「探す」ものは無く、直接スキャンは許可ドキュメントだけを（graph 探索が
触れる数を超えずに）採点し、しかも **exact** です。そのため非常に選択的な
フィルターでは、近似ではなく真の最近傍マッチが返ります。許可セットが大きい場合は
上記の filter-aware トラバーサルを引き続き使います。

#### deletion-aware HNSW トラバーサル

論理削除されたドキュメントはコンパクションまで HNSW graph に残るため
（[削除とコンパクション](../../laurus/deletions.md) を参照）、トラバーサルは
フィルター除外と同じ方法で削除ノードをスキップする必要があります。graph 探索は
単一の admission ルールを適用します。ノードが結果集合に入るのは、フィルターに
マッチし（フィルターがある場合）**かつ** 削除されていない場合のみで、frontier は
連結性を保つために削除ノードも通過します。

これにより削除が蓄積しても recall が正しく保たれます。削除ノードを result heap に
入れてしまうと、固定の `ef_search` スロットを占有して生存ネイバーを押し出し、
最悪の場合は `ef_search` ウィンドウが削除ドキュメントだけで埋まって何も返らなく
なります。トラバーサル中にスキップすれば同じスロットが生存結果で埋まるため、
最近傍のドキュメントを削除しても 10 件のページは満杯のままです。上記の小さな
許可セットの exact スキャンや Flat / IVF のインライン経路も同じ削除チェックを
適用します。

高速経路は維持されます。フィルターも削除も無い検索では、トラバーサルは従来の
ループをそのまま実行し、ネイバーごとの admission 用ブックキーピングのコストを
一切払いません。

#### 許可セットの表現（Allow-Set Representation）

許可セットは形状に応じて型が選ばれます。密なフィルターには Roaring ビットマップ、
疎なフィルターにはハッシュセットを使います。フィルター付きハイブリッド検索では、
lexical 側がマッチ集合をビットマップとして既に構築している
（lexical の [フィルタ結果キャッシュ](lexical_search.md#フィルタ結果キャッシュfilter-result-cache)
参照）ため、そのビットマップを vector 側に**そのまま**渡します。これにより集合は
クエリ全体で一度だけ実体化され、両側で再構築されません。これは内部最適化であり、
公開フィルター API は不変です。

### 数値範囲によるフィルター

```rust
use laurus::lexical::NumericRangeQuery;
use laurus::lexical::core::field::NumericType;

let request = SearchRequestBuilder::new()
    .vector_query(
        VectorSearchRequestBuilder::new()
            .add_text("embedding", "type systems")
            .build()
    )
    .filter_query(Box::new(NumericRangeQuery::new(
        "year", NumericType::Integer,
        Some(2020.0), Some(2024.0), true, true
    )))
    .limit(10)
    .build();
```

## 距離メトリクス（Distance Metrics）

距離メトリクスはスキーマでフィールドごとに設定されます（[Vector インデキシング](../indexing/vector_indexing.md) を参照）。

| メトリクス | 説明 | 小さい値 = より類似 |
| :--- | :--- | :--- |
| **Cosine** | 1 - コサイン類似度 | はい |
| **Euclidean** | L2 距離 | はい |
| **Manhattan** | L1 距離 | はい |
| **DotProduct** | 負の内積 | はい |
| **Angular** | 角度距離 | はい |

## コード例: 完全な Vector 検索

```rust
use std::sync::Arc;
use laurus::{Document, Engine, Schema, SearchRequestBuilder, PerFieldEmbedder};
use laurus::lexical::TextOption;
use laurus::vector::HnswOption;
use laurus::vector::VectorSearchRequestBuilder;
use laurus::storage::memory::MemoryStorage;

#[tokio::main]
async fn main() -> laurus::Result<()> {
    let storage = Arc::new(MemoryStorage::new(Default::default()));

    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_hnsw_field("text_vec", HnswOption {
            dimension: 384,
            ..Default::default()
        })
        .build();

    // Set up per-field embedder
    let embedder = Arc::new(my_embedder);
    let pfe = PerFieldEmbedder::new(embedder.clone());
    pfe.add_embedder("text_vec", embedder.clone());

    let engine = Engine::builder(storage, schema)
        .embedder(Arc::new(pfe))
        .build()
        .await?;

    // Index documents (text in vector field is auto-embedded)
    engine.add_document("doc-1", Document::builder()
        .add_text("title", "Rust Programming")
        .add_text("text_vec", "Rust is a systems programming language.")
        .build()
    ).await?;
    engine.commit().await?;

    // Search by semantic similarity
    let results = engine.search(
        SearchRequestBuilder::new()
            .vector_query(
                VectorSearchRequestBuilder::new()
                    .add_text("text_vec", "systems language")
                    .build()
            )
            .limit(5)
            .build()
    ).await?;

    for r in &results {
        println!("{}: score={:.4}", r.id, r.score);
    }

    Ok(())
}
```

## 次のステップ

- キーワード検索と組み合わせる: [ハイブリッド検索](hybrid_search.md)
- Vector クエリの DSL 構文: [Query DSL](../query_dsl.md)
