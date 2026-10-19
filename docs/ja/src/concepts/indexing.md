# インデキシング（Indexing）

このセクションでは、Laurus がデータを内部的にどのように格納・整理するかについて説明します。インデキシングレイヤーを理解することで、適切なフィールドタイプの選択やパフォーマンスチューニングに役立ちます。

## トピック

### [Lexical インデキシング](indexing/lexical_indexing.md)

転置インデックス（Inverted Index）を使用したテキスト、数値、地理フィールドのインデキシング方法について説明します。

- 転置インデックスの構造（Term Dictionary、Posting Lists）
- 数値範囲クエリのための BKD ツリー
- セグメントファイルとそのフォーマット
- BM25 スコアリング

### [Vector インデキシング](indexing/vector_indexing.md)

近似最近傍探索（Approximate Nearest Neighbor Search）のためのベクトルフィールドのインデキシング方法について説明します。

- インデックスタイプ: Flat、HNSW、IVF
- パラメータチューニング（m、ef_construction、n_clusters、n_probe）
- 距離メトリクス（Cosine、Euclidean、DotProduct）
- 量子化（Quantization）: SQ8、PQ、Float16
