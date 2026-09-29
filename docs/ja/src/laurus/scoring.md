# スコアリングとランキング

Laurus は Lexical 検索に BM25、Vector 検索に距離ベースの類似度、そしてハイブリッド検索ではこの 2 つを統合する設定可能なフュージョンアルゴリズムを使用します。本ページでは各スコアリング経路と、公開 API から介入する方法を説明します。

## Lexical スコアリング

### BM25（デフォルト）

BM25 は Lexical 検索のスコアリング関数です。単語頻度（term frequency）とドキュメント長正規化のバランスを取ります。

```text
score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
```

各パラメータ:

- **tf** — ドキュメント内の単語頻度。
- **IDF** — 逆文書頻度（全ドキュメントに対する単語の希少度）。
- **k1** — 単語頻度の飽和パラメータ。Laurus は **1.2** を使用。
- **b** — ドキュメント長正規化の係数。Laurus は **0.75** を使用。
- **doc_len / avg_doc_len** — ドキュメント長と平均ドキュメント長の比率。`doc_len` はトークン数ではなく **position** の数です —— アナライザーが別のトークンにスタックする同義語（`position_increment = 0`）はそのトークンの position を共有し、長さに加算されません。Lucene の `discountOverlaps` と同じ挙動です。

`(k1, b)` の値は現在実装デフォルトに固定です。Lucene / Elasticsearch のデフォルトと同じため、BM25 スコアはチューニングの直感に関してそれらのエンジンと直接比較できます。

クオート済み・未クオートを問わず、アナライザーが 1 つの位置に複数の代替語（同義語）を積む値は、独立したタームのスコアを合算するのではなく、1 つのブレンドされたタームとしてスコア付けされます。そのため、すべての代替語を含んでいるからといって文書のスコアが不当に高くなることはありません。詳細は[同義語を使った検索](../concepts/analysis.md#同義語を使った検索)を参照してください。

### フィールドブースト

フィールドごとのスコア乗数は専用のスコアリング構造体ではなく、検索リクエスト上で設定します。

```rust
use laurus::SearchRequestBuilder;

let request = SearchRequestBuilder::new()
    .query_dsl("rust programming")
    .add_field_boost("title", 2.0) // title のマッチはスコア 2 倍
    .add_field_boost("body", 1.0)  // body のマッチはスコア 1 倍（デフォルト）
    .limit(10)
    .build();
```

ブーストはそのフィールドにマッチした BM25 スコア寄与に乗算されます。`1.0` は無効化と同じです。クエリで指定されたフィールド（またはスキーマの既定検索フィールド）にのみ適用されます。

gRPC / HTTP 経由では同じ設定が `SearchRequest.field_boosts`（`map<string, float>`）として公開されます。[gRPC API → SearchRequest](../laurus-server/grpc_api.md#searchrequest-fields) を参照してください。

## Vector スコアリング

Vector 検索は距離ベースの類似度で結果をランク付けします。距離メトリックはベクトルインデックス（HNSW / Flat / IVF）のフィールドごとに設定します。

| メトリック | 説明 | 適した用途 |
| :--- | :--- | :--- |
| `Cosine` | 1 − コサイン類似度（デフォルト） | 正規化済みテキスト埋め込み |
| `Euclidean` | L2 距離 | 空間データ・事前正規化済みデータ |
| `Manhattan` | L1 距離 | 疎な特徴ベクトル |
| `DotProduct` | 符号反転した内積 | 高いほど良い事前正規化済みベクトル |
| `Angular` | 角度距離 | 方向の類似度 |

距離は類似度スコア（「高いほど良い」）に変換され、Lexical 結果と Vector 結果のいずれにおいてもこの規約が保たれます。下記のフュージョンアルゴリズムはこの前提に依存します。

## ハイブリッド検索フュージョン

検索リクエストが Lexical 句と Vector 句の両方を持つ場合、2 つの結果リストをマージする必要があります。Laurus は [`FusionAlgorithm`](api_reference.md#fusionalgorithm) で 2 種類のフュージョンアルゴリズムを公開しています。

### RRF（Reciprocal Rank Fusion）

RRF は生のスコアではなく**ランク**を統合することでスコア正規化を完全に回避します。

```text
rrf_score(doc) = Σ 1 / (k + rank_i(doc))
```

合計はドキュメントが含まれる各結果リストにわたって取ります。`k` パラメータ（デフォルト **60.0**）は分布を平滑化します — 値が大きいほど上位ランクの結果の貢献が薄まります。

```rust
use laurus::{FusionAlgorithm, SearchRequestBuilder};

let request = SearchRequestBuilder::new()
    .query_dsl("title:rust ~\"systems programming\"")
    .fusion_algorithm(FusionAlgorithm::Rrf { k: 60.0 })
    .build();
```

### WeightedSum

`WeightedSum` は各リストのスコアを個別に min-max 正規化したうえで、重み付き線形結合を取ります。

```text
norm(score)  = (score - min) / (max - min)
final(doc)   = lexical_weight * norm(lexical_score(doc))
             + vector_weight  * norm(vector_score(doc))
```

```rust
use laurus::{FusionAlgorithm, SearchRequestBuilder};

let request = SearchRequestBuilder::new()
    .query_dsl("title:rust ~\"systems programming\"")
    .fusion_algorithm(FusionAlgorithm::WeightedSum {
        lexical_weight: 0.6,
        vector_weight: 0.4,
    })
    .build();
```

両方の重みは `[0.0, 1.0]` にクランプされます。特定の重みを設定する理由がない場合は RRF を選んでください — パラメータが少なく、リスト間のスケール差にも頑健です。

## 関連項目

- [API リファレンス → `FusionAlgorithm`](api_reference.md#fusionalgorithm) — バリアントのシグネチャ
- [ハイブリッド検索](../concepts/search/hybrid_search.md) — どのフュージョンを選ぶかの目安
- [Vector 検索](../concepts/search/vector_search.md) — 距離メトリックのトレードオフ
