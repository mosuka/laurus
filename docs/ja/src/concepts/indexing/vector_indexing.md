# Vector インデキシング

Vector インデキシングは、類似性ベースの検索を支える仕組みです。ドキュメントのベクトルフィールドがインデキシングされると、Laurus はエンベディングベクトルを専用のインデックス構造に格納し、高速な近似最近傍探索（Approximate Nearest Neighbor, ANN）を可能にします。

## Vector インデキシングの仕組み

```mermaid
sequenceDiagram
    participant Doc as Document
    participant Embedder
    participant Normalize as Normalizer
    participant Index as Vector Index

    Doc->>Embedder: "Rust is a systems language"
    Embedder-->>Normalize: [0.12, -0.45, 0.78, ...]
    Normalize->>Normalize: L2 normalize
    Normalize-->>Index: [0.14, -0.52, 0.90, ...]
    Index->>Index: Insert into index structure
```

### ステップごとの流れ

1. **エンベディング（Embed）**: テキスト（または画像）が、設定されたエンベッダーによってベクトルに変換される
2. **正規化（Normalize）**: ベクトルは **Cosine メトリックの場合のみ** L2 正規化される（大きさ不変なメトリック）。Euclidean / DotProduct / Manhattan のフィールドは距離を保つため元のベクトルのまま保存される
3. **インデキシング（Index）**: ベクトルが設定されたインデックス構造（Flat、HNSW、または IVF）に挿入される
4. **コミット（Commit）**: `commit()` の呼び出し時に、インデックスが永続ストレージにフラッシュされる

## インデックスタイプ

Laurus は 3 種類のベクトルインデックスタイプをサポートしており、それぞれ異なるパフォーマンス特性を持ちます。

### 比較

| 特性 | Flat | HNSW | IVF |
| :--- | :--- | :--- | :--- |
| **精度** | 100%（厳密） | 約 95-99%（近似） | 約 90-98%（近似） |
| **検索速度** | O(n) 線形スキャン | O(log n) グラフ走査 | O(n/k) クラスタスキャン |
| **メモリ使用量** | 低 | 高（グラフエッジ） | 中程度（セントロイド） |
| **インデックス構築時間** | 高速 | 中程度 | 低速（クラスタリング） |
| **最適な用途** | 1 万ベクトル未満 | 1 万 - 1,000 万ベクトル | 100 万ベクトル以上 |

### Flat インデックス

最もシンプルなインデックスです。クエリベクトルを格納されたすべてのベクトルと比較します（総当たり）。

```rust
use laurus::vector::FlatOption;
use laurus::vector::core::distance::DistanceMetric;

let opt = FlatOption {
    dimension: 384,
    distance: DistanceMetric::Cosine,
    ..Default::default()
};
```

- **利点**: 100% の再現率（厳密な結果）、シンプル、低メモリ
- **欠点**: 大規模データセットでは低速（線形スキャン）
- **使用場面**: ベクトル数が約 1 万未満の場合、または厳密な結果が必要な場合

### HNSW インデックス

**Hierarchical Navigable Small World** グラフ。デフォルトで最も一般的に使用されるインデックスタイプです。

```mermaid
graph TB
    subgraph "Layer 2 (sparse)"
        A2["A"] --- C2["C"]
    end

    subgraph "Layer 1 (medium)"
        A1["A"] --- B1["B"]
        A1 --- C1["C"]
        B1 --- D1["D"]
        C1 --- D1
    end

    subgraph "Layer 0 (dense - all vectors)"
        A0["A"] --- B0["B"]
        A0 --- C0["C"]
        B0 --- D0["D"]
        B0 --- E0["E"]
        C0 --- D0
        C0 --- F0["F"]
        D0 --- E0
        E0 --- F0
    end

    A2 -.->|"entry point"| A1
    A1 -.-> A0
    C2 -.-> C1
    C1 -.-> C0
    B1 -.-> B0
    D1 -.-> D0
```

HNSW アルゴリズムは、上位の疎なレイヤーから下位の密なレイヤーへと検索し、各レベルで検索空間を絞り込みます。

```rust
use laurus::vector::HnswOption;
use laurus::vector::core::distance::DistanceMetric;

let opt = HnswOption {
    dimension: 384,
    distance: DistanceMetric::Cosine,
    m: 16,                  // max connections per node per layer
    ef_construction: 200,   // search width during index building
    ..Default::default()
};
```

#### HNSW パラメータ

| パラメータ | デフォルト | 説明 | 影響 |
| :--- | :--- | :--- | :--- |
| `m` | 16 | レイヤーごとのノードあたりの最大双方向接続数 | 大きいほど再現率が向上するが、メモリ消費が増加 |
| `ef_construction` | 200 | インデックス構築時の探索幅 | 大きいほど再現率が向上するが、構築が低速に |
| `dimension` | 128 | ベクトルの次元数 | エンベッダーの出力と一致させる必要あり |
| `distance` | Cosine | 距離メトリクス | 下記の距離メトリクスを参照 |

**チューニングのヒント:**

- 再現率を向上させるには `m` を増やす（例: 32 または 64）。ただしメモリ消費が増加する
- インデックス品質を向上させるには `ef_construction` を増やす（例: 400）。ただし構築時間が増加する
- 検索時には、検索リクエストで設定する `ef_search` パラメータが探索幅を制御する

### IVF インデックス

**Inverted File Index**。ベクトルをクラスタに分割し、関連するクラスタのみを検索します。

```mermaid
graph TB
    Q["Query Vector"]
    Q --> C1["Cluster 1\n(centroid)"]
    Q --> C2["Cluster 2\n(centroid)"]

    C1 --> V1["vec_3"]
    C1 --> V2["vec_7"]
    C1 --> V3["vec_12"]

    C2 --> V4["vec_1"]
    C2 --> V5["vec_9"]
    C2 --> V6["vec_15"]

    style C1 fill:#f9f,stroke:#333
    style C2 fill:#f9f,stroke:#333
```

```rust
use laurus::vector::IvfOption;
use laurus::vector::core::distance::DistanceMetric;

let opt = IvfOption {
    dimension: 384,
    distance: DistanceMetric::Cosine,
    n_clusters: 100,   // number of clusters
    n_probe: 10,       // clusters to search at query time
    ..Default::default()
};
```

#### IVF パラメータ

| パラメータ | デフォルト | 説明 | 影響 |
| :--- | :--- | :--- | :--- |
| `n_clusters` | 100 | ボロノイセル（Voronoi Cell）の数 | クラスタ数が多いほど検索は高速になるが、再現率は低下 |
| `n_probe` | 1 | クエリ時に検索するクラスタ数 | 大きいほど再現率が向上するが、検索が低速に |
| `dimension` | （必須） | ベクトルの次元数 | エンベッダーの出力と一致させる必要あり |
| `distance` | Cosine | 距離メトリクス | 下記の距離メトリクスを参照 |

**チューニングのヒント:**

- `n_clusters` はベクトル数 `n` に対して `sqrt(n)` 程度に設定する
- 再現率と速度のバランスを取るため、`n_probe` を `n_clusters` の 5-20% に設定する
- IVF はトレーニングフェーズが必要なため、初回のインデキシングが遅くなる場合がある

## 距離メトリクス（Distance Metrics）

| メトリクス | 説明 | 値の範囲 | 最適な用途 |
| :--- | :--- | :--- | :--- |
| `Cosine` | 1 - コサイン類似度 | [0, 2] | テキストエンベディング（最も一般的） |
| `Euclidean` | L2 距離 | [0, +inf) | 空間データ |
| `Manhattan` | L1 距離 | [0, +inf) | 特徴ベクトル |
| `DotProduct` | 負の内積 | (-inf, +inf) | 事前正規化済みベクトル |
| `Angular` | 角度距離 | [0, pi] | 方向の類似性 |

```rust
use laurus::vector::core::distance::DistanceMetric;

let metric = DistanceMetric::Cosine;      // Default for text
let metric = DistanceMetric::Euclidean;    // For spatial data
let metric = DistanceMetric::Manhattan;    // L1 distance
let metric = DistanceMetric::DotProduct;   // For pre-normalized vectors
let metric = DistanceMetric::Angular;      // Angular distance
```

> **注意:** ベクトルがインデキシング前に自動的に L2 正規化されるのは **`Cosine` メトリックの場合のみ**です。正規化は大きさ不変なので Cosine では安全（かつ int8 量子化の範囲も締まる）ですが、大きさに依存するメトリックでは距離が変わってしまうため、`Euclidean` / `DotProduct` / `Manhattan` のフィールドは正規化せずに保存されます。距離が小さいほど、より類似していることを示します。
>
> **マイグレーション注記:** この挙動が修正される前に作成した非 Cosine フィールドは、ディスク上に L2 正規化済みのベクトルを持ちます（元の大きさは復元できません）。正しい距離を得るにはインデックスを再構築してください。

## 量子化（Quantization）

ベクトルはディスク上で **8 ビットスカラー量子化された整数** として保存
されます（Issue #481 Stage 1）。以前の 32 ビット浮動小数点形式と比較
して **約 4 倍小さく**、recall 損失は実用上ほぼ無視できる範囲（f32
ground truth に対して Recall@10 ≥ 0.95 — recall テスト
`laurus/tests/vector_recall_test.rs` 参照）です。

| 方式 | Enum バリアント | 説明 | メモリ削減率 |
| :--- | :--- | :--- | :--- |
| **スカラー 8 ビット** *(デフォルト)* | `Scalar8Bit` | per-segment global affine による `u8` 量子化 | 約 4 倍 |
| **プロダクト量子化** | `ProductQuantization { subvector_count }` | Issue #481 Stage 3 — `M × 256` centroid の codebook（segment ごとに学習、または `pq_codebook_path` で[一度だけ学習して共有](#共有-pq-codebookissue-631)）、各ベクトルを `M` バイトで保存 | 約 16-64 倍（HNSW のみ） |

```rust
use laurus::vector::HnswOption;
use laurus::vector::core::quantization::QuantizationMethod;

// `quantizer` は `Scalar8Bit` がデフォルト。下記は
// `HnswOption { dimension: 384, ..Default::default() }` と等価。
let opt = HnswOption {
    dimension: 384,
    quantizer: QuantizationMethod::Scalar8Bit,
    ..Default::default()
};
```

> **破壊的変更（Issue #481 Stage 1）:** `quantizer` フィールドはもはや
> `Option<QuantizationMethod>` ではなく必須となり、デフォルトは
> `Scalar8Bit` です。f32 のままディスクに保存する形式は廃止されまし
> た。Stage 1 より前に作成した既存 vector index は意図的に読み取り
> 不可能で、ソースデータからの再構築が必要です。

### Scalar8Bit のしくみ

- 各 segment は flush 時に f32 ベクトル群から **global** な
  `(offset, scale)` ペア 1 組をトレーニング
  （`offset = min`, `scale = (max - min) / 255`）。
- 各 `f32` 要素は `u8 = clamp(round((v - offset) / scale), 0, 255)`
  でエンコード。
- 各ベクトル単位のメタデータ（`sum_q: u32`, `norm_q: f32`）を
  事前計算して int8 ペイロードと併置するため、cosine 検索の hot loop
  は int8 SIMD multiply-accumulate 1 回 + scalar 補正 3 回に縮約され、
  検索時の per-element dequantize は不要。
- segment ファイルは `LVS1` magic + 16 byte header で始まり、reader
  はロード時にフォーマットを判定。header はバージョン（機能のはしご:
  v2 = HNSW グラフブロックの ordinal エンコード、v3 = セグメントごとの
  フィールド名辞書）を持ち、v3 以降は辞書そのものも header に載る —
  各レコードはフィールド名をインラインで繰り返す代わりに 16 ビット ID
  で参照する。

### Two-stage rerank（Issue #481 Stage 2）

Stage 1 ではベクトルを int8 のみで保持します。グラフ検索は完全に int8
距離に対して行われ、高速ですがわずかな量子化誤差が入ります。Stage 2
ではフィールド単位で **任意の** f32 sidecar を追加し、上位候補を元の
完全精度ベクトルで再スコアできるようにします:

1. HNSW int8 グラフ検索が量子化コサイン距離で最大 `ef_search` 件の
   候補を返す。
2. 上位 `top_k * rerank_factor` 件を [LRS1
   sidecar](#lrs1-rerank-sidecar)（`*.hnsw.f32`）から読み込んだ f32
   ベクトルで再スコアする。
3. 新しいランキングを `top_k` に切り詰めて返す。

Issue #932 以降、同じ sidecar 機構は共有 `RerankPipeline`（#650）を
通じて Flat / IVF でも機能します（以下の説明は Stage 2 の元ホストである
HNSW を例にしています）。Stage 2 はフィールド単位で
[`HnswOption.rerank_storage`](../../laurus-cli/schema_format.md#rerank-storage)
で opt-in します:

```rust
use laurus::vector::HnswOption;
use laurus::vector::core::rerank::RerankStorageKind;

let opt = HnswOption {
    rerank_storage: Some(RerankStorageKind::F32),
    ..HnswOption::default()
};
```

クエリ側は `VectorIndexQuery::rerank_factor`（low-level）、
`SearchRequestBuilder::vector_rerank_factor`（engine）、
gRPC / JSON の `VectorParams.rerank_factor` のいずれかで rerank
factor を渡します。

`rerank_storage` が無効なフィールドでは `rerank_factor` を指定しても
silent に Stage 1 int8 ランキングへフォールバックします — Stage 1
セグメントから f32 情報を復元することはできません。

#### LRS1 rerank sidecar

sidecar は `rerank_storage` が有効なときに LVS1 セグメントの隣に
書かれる別ファイルです:

```text
offset  size  field
------  ----  -------------------------------------------
     0     4  magic         ASCII "LRS1"
     4     2  version       u16 LE  (current = 1)
     6     2  storage_kind  u16 LE  (1 = F32; 0 reserved; 2.. future)
     8     8  reserved      zero-padded
    16     4  dim           u32 LE
    20     4  vector_count  u32 LE
    24     -  payload       vector_count * dim * bytes_per_element
   end     8  footer        magic "LRC1" u32 LE + CRC-32 u32 LE
                             （header + payload が対象）
```

ベクトルは LVS1 セグメントと同じ `(doc_id, field_name)` 順で書かれる
ので、(sidecar position) → (LVS1 position) のマッピングは恒等関数に
なります。HNSW reader は storage の loading mode が Eager のとき初期化
時に sidecar を `RerankStoragePool` にロードします。Lazy mode では
memory savings の前提を尊重するため sidecar 読み込みをスキップします
（Lazy mode で開いた Stage 2 セグメントは silent に Stage 1 へ
degrade します）。

sidecar はスキーマの HNSW オプションの `rerank_storage` でフィールド
ごとに有効化され、その設定はすべての書き込み経路で尊重されます。直接
の commit、アクティブな書き込みセグメント、セグメントの merge いずれも
生成したセグメントに対して sidecar を再出力します。マージ後セグメント
の sidecar は **ソースセグメントの原 f32 サイドカー** から再構築されます
（int8 復元値ではなく）。これによりマージはマージごとに量子化誤差を 1 段
ずつ蓄積させず、rerank ベクトルを無損失に保ちます。

新しい sidecar は末尾に header + payload を対象とする 8 バイトの
CRC-32 footer を持ち、sidecar を読むすべての経路（searcher のロードと
writer の再ロード）で検証されます。これにより、ディスク上の静かな破損
が rerank スコアを歪める代わりに拒否されます。header からコンテンツ長
が一意に決まるため、footer は payload の後ろに残っているバイト数で
検出されます — footer 導入前に書かれた sidecar は末尾バイトがゼロ
なのでそのままロードでき、検証はスキップされます。

#### recall と速度の trade-off

`rerank_factor` は per-query rerank コスト（`top_k * rerank_factor`
回の exact distance 計算 — dim 128 で数 µs）と引き換えに Recall@10 の
向上を得る lever です。実際の効果はコーパスとグラフ検索 budget
（`ef_search`）に依存します:

- 実際の clustered な embedding（text-embedding-3、BERT など）は
  低い `ef_search` で `Recall@10 ≥ 0.99` に到達し、rerank は微小な
  latency 増で順位を磨きます。
- 合成 random unit-norm（HNSW 復元の最悪ケース）では int8 グラフが
  真の top-10 候補を十分に visit するために高い `ef_search` が必要で、
  rerank は visit 済み候補を並び替えられても visit していないものは
  取り戻せません。

recall acceptance は rerank kernel と HNSW フルパイプラインを独立に
落とせるよう、CI gate を 2 段に分けています:

- `stage2_brute_force_rerank_recall_at_10_meets_kernel_gate` は
  `Recall@10 ≥ 0.99` を assert します。HNSW グラフを完全にバイパス
  し（brute-force int8 で corpus 全体を採点 → `top_k * rerank_factor`
  に絞る → f32 で再スコア）miss すれば rerank kernel の regression
  と切り分けられます。
- `hnsw_quantized_recall_at_10_with_rerank_meets_stage2_recall_gate`
  は `Recall@10 ≥ 0.98` を assert します。HNSW build の
  non-determinism（f32 HNSW baseline でも同様に出るノイズ）を
  含むため、合成 adversarial 分布での run-to-run 観測幅に合わせて
  しきい値を緩めています。実 embedding の clustered 分布や
  強めの HNSW config（m=32, ef_construction=500）はこのパス
  でも ≥ 0.99 に到達します（下の diagnostic sweep を参照）。

companion の `stage2_recall_sweep_diagnostic`（`LAURUS_STAGE2_SWEEP=1`
で opt-in）は `(ef_search, rerank_factor)` を 3 種の corpus / query
分布 × 2 種の HNSW config で sweep するので、production deployment
は実際の embedding 分布で budget を calibrate できます。

#### 実データ検証（Issue #498）

3 つめの opt-in CI gate として、 Stage 2 を実 ANN benchmark dataset
（TEXMEX の SIFT1M）で検証します。 synthetic data の gate だけが
signal にならないようにするためのものです。

- `hnsw_quantized_recall_at_10_with_rerank_on_sift_meets_stage2_real_data_recall_gate`
  は SIFT1M の 50 000 ベクトルサブサンプル上で
  `(m=16, ef_construction=200, ef_search=200, rerank_factor=5)` の
  Recall@10 ≥ 0.99 を assert します。
- companion bench `bench_hnsw_graph_search_rerank_real_data`
  （`laurus/benches/vector_search_bench.rs`）は同じ fixture で
  end-to-end Stage 2 latency を測定します。 同梱の
  `laurus/examples/sift_rerank_probe.rs` は
  `(ef_search × rerank_factor × HNSW config)` を sweep して
  `(Recall, latency)` を per-cell で報告するので、運用側は自分の
  データに合った operating point を選べます。

両者は `LAURUS_REAL_BENCHMARK=1` と `.cache/sift/sift/` 配下の
SIFT1M `.fvecs` ファイルの存在で gate されます。 デフォルトの CI
runs は変化しません。 ローカルで有効化するには:

```sh
./scripts/fetch-sift.sh --large   # ~478MB
LAURUS_REAL_BENCHMARK=1 cargo test --release \
    --test vector_recall_test \
    hnsw_quantized_recall_at_10_with_rerank_on_sift_meets_stage2_real_data_recall_gate \
    -- --nocapture
LAURUS_REAL_BENCHMARK=1 cargo bench --bench vector_search_bench \
    -- "HNSW Graph Search Rerank Real"
```

Issue #481 の原文は "≥ 3× speedup vs the pre-Stage-1 f32 baseline"
を要求していましたが、 SIFT1M-50k での cross-branch Criterion 計測
（30 サンプル中央値、同じ `(m, ef_construction, ef_search)`）では
pre-Stage-1 f32 HNSW path が 625 µs/query、 Stage 2 int8 + rerank
path が 323 µs/query となり、 speedup は **1.94×** でした。 この
実測結果を受けて Issue #498 では real-data gate を **≥ 1.5×** に
下げています（recall 側は元の 0.99 を維持）。 3× との gap は、
rerank が int8 graph traversal で visit 済みの候補を re-rank する
だけで、 `ef_search` を下げると candidate set 自体が狭くなり rerank
で回収できないことに起因します。 follow-up としては graph search
budget を `ef_search` から独立に広げる（Lucene 99 pattern）案が
あります。

### Product Quantization + rerank（Issue #481 Stage 3）

Stage 3 は HNSW index 向けに opt-in の **Product Quantization** path を
追加します。 各 segment は per-field の codebook（`M` 個の sub-vector
× `K = 256` centroid）を k-means++ + Lloyd 反復で学習し、 各ベクトル
を `M` バイトで保存します（sub-vector あたり 1 centroid index）。
検索ホットループでは int8 SIMD kernel を **asymmetric distance
computation**（ADC）に置き換えます: query 1 回ごとに query
sub-vector と codebook entry の squared distance を `M × K`
look-up table に展開し、候補ごとに `Σ_m lut[m][codes[m]]` で
スコアリング（1 候補あたり `M` lookups + `M − 1` add）。

PQ は per-field `HnswOption.quantizer` で有効化:

```rust
use laurus::vector::HnswOption;
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::core::rerank::RerankStorageKind;

let opt = HnswOption {
    dimension: 128,
    quantizer: QuantizationMethod::ProductQuantization { subvector_count: 32 },
    // SIFT1M の PQ-only Recall@10 は 0.78-0.92 で頭打ちのため、
    // production では LRS1 rerank sidecar との組み合わせを推奨
    // （Stage 2 と同じ仕組み、 candidate 生成側のみ int8 から PQ
    // に置き換わる）。
    rerank_storage: Some(RerankStorageKind::F32),
    ..Default::default()
};
```

**最小学習サイズ**: PQ はサブ量子化器ごとに `K = 256` 個の k-means
セントロイドを学習するため、256 ベクトル未満のセグメントでは意味のある
codebook を学習できません。そのようなセグメントは **Scalar8Bit** で
書き出されます（`LVS1` ヘッダは自己記述型のため、reader は保存された
種別で透過的にディスパッチします）。セグメントが成長するか、より大きな
セグメントへマージされれば、次の書き込みで設定どおり PQ が学習されます。
`subvector_count` が `dimension` を割り切る検証はセグメントサイズに
かかわらず常に適用されます。なお、
[共有 codebook](#共有-pq-codebookissue-631) を設定した場合はこの
フォールバックも最小サイズも適用されません — セグメントごとの学習が
発生しないため、小さな per-commit セグメントもそのまま PQ を維持します。

`subvector_count` は `dimension` を割り切る値を選択。 `dim = 128`
での候補: `M ∈ {8, 16, 32}`（sub_dim 16 / 8 / 4）。 `M` を大きく
すると compression は下がるが recall は上がります。 Issue #481
Stage 3 は 8 bit（K = 256）のみを ship、 on-disk format は
将来の 4 bit（K = 16）variant 用に枠を予約しています。

#### 共有 PQ codebook（Issue #631）

デフォルトでは segment の書き込みごとに codebook をゼロから学習
します — segment-per-commit レイアウト（#634/#889）では、この数秒
オーダーの k-means コストが **commit と merge のたびに**繰り返され
ます。 FAISS / Lucene99 の前例に倣い、 codebook を代表サンプルで
**一度だけ**学習して以後のすべての segment 書き込みで再利用でき、
encode はテーブル参照だけになります（2,048〜8,192 ベクトルで
インライン学習比 約 91-92% 高速の実測値）。

有効化はフィールドに codebook ファイル名を設定して学習するだけです:

```toml
[fields.embedding.Hnsw]
dimension = 128
pq_codebook_path = "embedding.pqcb"

[fields.embedding.Hnsw.quantizer.ProductQuantization]
subvector_count = 32
```

```shell
laurus train pq-codebook --field embedding --input vectors.jsonl --update-schema
```

（`--input` の代わりに `--from-index` でインデックスにコミット済みの
ベクトルを直接サンプリングすることも、`laurus create index
--train-pq-codebook <jsonl>` で学習をインデックス作成に畳み込んで
train-before-first-commit の順序ハザードを完全に解消することもできます
— いずれも Issue #920）、または
プログラムから
[`Engine::train_pq_codebook`](https://docs.rs/laurus/latest/laurus/struct.Engine.html)
（`engine.train_pq_codebook("embedding", &vectors, None)`。from-index
フローには `engine.sample_committed_vectors("embedding", Some(n))` を
組み合わせます）を使います。学習は同期・CPU-bound で、代表的な
ベクトル数千件で十分です（全コーパスは不要）。

セマンティクス:

- **segment format は不変。** 共有 codebook も従来どおり各 segment
  header にインライン埋め込みされるため、共有 codebook の segment と
  インライン学習の segment はマイグレーション無しで共存でき、生成
  される code は同一サンプルでインライン学習した場合と
  **byte-identical** です。
- codebook ファイルは **index open 時に一度だけ**解決されます。
  エンジンを開いたまま再学習しても mid-session で hot-swap されません
  （意図的な仕様 — 1 セッションの commit 間で codebook が変わっては
  ならないため）。新しいファイルは再オープンで反映されます。
- **失敗ポリシー — 無言のフォールバック無し。** `pq_codebook_path` が
  設定済みで未学習の場合、 open は lenient（学習前にスキーマを作成
  できる）ですが、 encode が必要な commit は実行すべき学習コマンドを
  示すエラーで失敗します。ファイルは存在するが破損・ジオメトリ不一致の
  場合は open 時にエラーです。 laurus が per-segment 学習へ無言で
  フォールバックすることはありません — 数秒オーダーの学習コストへの
  見えない回帰は本機能の意義を損なうためです。
- Cosine メトリックのフィールドでは、 writer がインデックス時に行う
  L2 正規化と同じ正規化を学習サンプルにも適用し、 codebook の基底を
  常に一致させます（#794 の罠への対処）。
- **FastScan フィールドにも対応**（Issue #920、`pq-fastscan` feature）:
  `ProductQuantizationFastScan` フィールドは同じコマンド・同じファイル
  フォーマットで k=16 の共有 codebook を学習・エンコードします —
  `.pqcb` header は `k` をそのまま保存するため 2 つの variant は
  保存された値で区別され、k 不一致の codebook（例: k=256 のファイルを
  FastScan フィールドに設定）は両方の値を明示するエラーで commit が
  失敗します。

#### on-disk format

PQ segment は Scalar8Bit と同じ `LVS1` header を使用（`quant_kind =
2`）し、 codebook は per-segment metadata block 内に格納:

```text
[ Fixed header           16 bytes ]
[ PQ params               8 bytes ]    m / k / sub_dim / padding (u16 × 4)
[ Codebook                m × k × sub_dim × 4 bytes ]
[ Per-vector codes        num_vectors × m bytes ]
```

`dim = 128, M = 32, K = 256` の場合 codebook = `32 × 256 × 4 × 4 =
131 072 bytes`（128 KB）+ ベクトル 1 件あたり 32 byte。

#### Recall と speed gate（Issue #481 Stage 3）

- **Kernel レベルテスト** — synthetic 5 000 ベクトル / dim 128 / 100
  query、 `(m=16, ef_construction=200, ef_search=200,
  rerank_factor=10, M=32)`:
  `hnsw_pq_rerank_recall_at_10_meets_stage3_recall_gate` が
  Recall@10 ≥ 0.95 を assert。 実測 0.9660。
- **実データテスト** — SIFT1M-50k subsample（opt-in:
  `LAURUS_REAL_BENCHMARK=1`、同設定）:
  `hnsw_pq_rerank_recall_at_10_on_sift_meets_stage3_real_data_recall_gate`
  が Recall@10 ≥ 0.95 を assert。 実測 0.9965。
- **実データ speed bench** —
  `bench_hnsw_graph_search_pq_rerank_real_data`（opt-in、Criterion）。
  同 SIFT1M-50k 設定での cross-branch 計測: pre-Stage-1 f32 HNSW =
  625.21 µs/query（PR #500 計測値）、 Stage 3 PQ + rerank =
  **299.54 µs/query** = **2.09× speedup**。

Issue #481 の原文は Recall ≥ 0.95 で ≥ 5× speedup を要求していました
が、 本 PR の実測で SIFT1M ではその target が到達不可能と判明。
Stage 2（#500）と Stage 3 は両方とも 1.9-2.1× の band で着地しており、
candidate set を recall が回収可能な幅まで広げると rerank がホット
パスを支配することが原因です。 これを受けて gate は ≥ 1.5× に
下げられています。 follow-up としては Lucene 99 pattern（graph
search の独立 budget）や 4 bit PQ variant でこの gap を埋める方向が
考えられます。

#### PQ → SQ → f32 の3段 rerank（Issue #673）

Stage 3 の rerank は PQ ADC の候補集合を `top_k * rerank_factor` まで
広げ、その先頭部分だけを exact な f32 サイドカーに対して rescore
します — グラフ探索がその予算を超えて見つけた候補は捨てられます。
以前は PQ フィールドで `ef_search` をこの予算より広く設定しても、
予算を超えた分には何の効果もありませんでした。#673 以降は、その
`ef_search` サイズの候補集合**全体**を、狭い exact-stage 予算で
切り出す前に安価な int8 kernel で再ランキングする **int8（SQ）段**
が追加で有効化されます — グラフがすでに計算済みのサープラス分が、
到着時点の PQ ADC 順序よりもずっと良い代理指標（int8）で競争する
機会を得られるようになり、単純に捨てられることがなくなります。

この int8 ビューは exact 段が読む **同じ LRS1 f32 サイドカーから
導出**されます（初回利用時に一度だけ遅延学習し、セグメントの生存期間
中キャッシュされます）— 新たなディスク上サイドカーも、新しい
フィールド設定も、新しいクエリパラメータも不要です。3段チェーンは
以下すべてを満たすときにクエリごと自動的に有効化されます:

- フィールドが PQ 量子化されている（`ProductQuantization` または
  `ProductQuantizationFastScan`）
- rerank サイドカー（`rerank_storage: Some(F32)`）がロード済み
- クエリに `rerank_factor` が指定されている
- `effective_ef_search > top_k * rerank_factor` — つまりグラフ探索が
  実際に exact 段単体が消費する量より多くの候補を計算している場合。
  そうでない場合（`ef_search` をデフォルトのままにしている一般的な
  ケース）、SQ 段は何も絞り込まないため skip され、クエリはこれまでと
  同一の2段（PQ → f32）経路をたどります。

`score_basis` の契約（Issue #927）はどちらの場合でも影響を受けません
— パイプラインは常に exact な f32 段で終わるため、SQ 段が事前に
走ったかどうかに関わらず、マルチセグメント fan-out はそのスコアを
引き続き exact でセグメント間比較可能な basis として扱います。

### MultiVector ストレージ

上記の量子化方式は ANN インデックスタイプ（Flat、HNSW、IVF）に適用され
ます。[`MultiVector` フィールド](../schema_and_fields.md#multivector-フィールド)
は ANN 索引を持たないため `quantizer` は適用されません。代わりに独自の
`storage` オプション（Issue #1346）で、各トークンベクトルのディスク上
での要素形式を選びます。

| 種類 | Rust Enum バリアント | バイト数/要素 | 誤差 | 1 文書あたりのサイズ（300 × 128 トークン） |
| :--- | :--- | :--- | :--- | :--- |
| **F32** *(デフォルト)* | `MultiVectorStorage::F32` | 4 | 厳密 | 約 150 KB |
| **F16** | `MultiVectorStorage::F16` | 2 | 要素あたり約 2⁻¹¹ の相対誤差 | 約 75 KB |
| **Int8** | `MultiVectorStorage::Int8` | 約 1（1 行あたり `dimension + 2` バイト） | ベクトルごとのスケール `max(abs(vector)) / 127` で上界が決まる | 約 38 KB |

```rust
use laurus::{MultiVectorOption, MultiVectorStorage};

let opt = MultiVectorOption::new(128).storage(MultiVectorStorage::Int8);
```

`quantizer` の per-segment グローバルスケールとは異なり、`Int8` のスケー
ルは**ベクトルごと**です: `scale = max(abs(vector)) / 127` を各トークン
ベクトルごとに個別に計算し、corpus や segment にまたがる学習済み状態は
持ちません。これにより segment のマージ時に行を再量子化せずバイト単位
でコピーできます。

既存フィールドの `storage` を変更するのは **Reindex** 扱いの変更です。
`Engine::update_field` を `reindex: true` で呼び出すと、フィールドの
既存セグメントを新しいディスク上の形式で再構築します。`storage` キーを
持たないスキーマ（このオプションが存在する前に書かれたもの）はデフォルト
の `F32` として扱われ、マイグレーションは不要です。

## セグメントファイル

| インデックスタイプ | ファイル拡張子 | 内容 |
| :--- | :--- | :--- |
| HNSW | `.hnsw` | グラフ構造、ベクトル、メタデータ |
| Flat | `.flat` | 生ベクトルとメタデータ |
| IVF | `.ivf` | クラスタセントロイド、割り当て済みベクトル、メタデータ |

3つのインデックスタイプはすべてデフォルトで **segment-per-commit** レイアウトを
使用します（下記参照）。単一ファイルの monolithic レイアウトは、インデックスを
直接利用する場合に `{Flat,Hnsw,Ivf}IndexConfig { segmented: false, .. }` で
引き続き選択できます。

### segment-per-commit レイアウト（3タイプ共通のデフォルト）

[#634](https://github.com/mosuka/laurus/issues/634)（HNSW）および
[#889](https://github.com/mosuka/laurus/issues/889)（Flat・IVF）以降、各コミットは
**新規追加ベクトルのみ**を不変の `segment_NNNNNN.{hnsw,flat,ivf}` ファイルとして
seal し、原子的・チェックサム付きの `segments.json` manifest に登録します —
コミットあたりのコストは O(インデックス) から O(新規ドキュメント) に下がり、
既存セグメントの全面的な作り直しも発生しません。

- **検索**は sealed セグメント群を（新しい世代から順に）fan-out し、
  同一ドキュメントの複製を newest-wins マスキングで重複排除します。
  再インデックスされたドキュメントは常に最新の埋め込みで解決されます。
  マージ待ちの stale 複製が深く積もっている場合も、セグメントごとの
  over-fetch 予算を幾何級数的に拡張し、マスキング後も十分な生存 hit が
  残るまで再取得する適応的 over-fetch により、結果品質は安定します。
- **マージ**: 世代連続・サイズ類似の tiered マージポリシーがインジェストの
  進行に応じてセグメント列を集約し（各ベクトルの生涯書き直し回数は
  O(log N)）、`optimize()` は全セグメントを 1 つに force-merge します。
  IVF のマージはセグメントごとの cluster ID を引き継ごうとしません
  （セグメントをまたいだ意味を持たないため）— 重複排除済みの生存ベクトルを
  平坦化し、union サイズに応じて適応的に再計算した cluster 数で centroid を
  ゼロから再学習します。Flat には f32 rerank サイドカーによるフォールバックが
  ないため、マージのたびに dequantize → scalar quantization パラメータの
  再学習 → requantize が発生し、HNSW/IVF のほぼロスレスな再量子化経路とは
  異なり、マージを重ねるごとにわずかな量子化ドリフトが蓄積します。
- **セグメントごとの学習**: HNSW のグラフと IVF の centroid はいずれも
  セグメントごとに独立して構築されます（IVF はスキーマの `n_clusters` を
  上限としてセグメント自身のサイズに適応した cluster 数を学習します）。
  検索はセグメントごとに fan-out して top-k をマージするため、
  どちらのインデックスタイプもセグメント間の合意は不要です。
- **削除**は論理削除です。インデックスレベルの bitmap（`{name}.delmap`）を
  すべてのセグメントリーダーが参照し、マージ時に物理回収されます。
- **移行**はゼロコピーです。既存の monolithic セグメントファイルは初回
  オープン時にそのまま最初のセグメントとして登録されます — manifest への
  1書き込みのみで、データ移動はありません。
- **耐久性**: セグメントファイルと manifest は temp ファイル + fsync +
  原子的 rename で書かれ、manifest はカバー対象の状態がすべて durable に
  なった後にのみ公開される WAL checkpoint を保持します —
  [永続化](../../laurus/persistence.md)を参照。

### コミット跨ぎの writer 保持（monolithic レイアウトのみ）

writer の保持は **monolithic** レイアウト（`segmented: false`）にのみ
適用されます。segmented な writer は常に空のバッファから始まるため、
保持すべき状態がそもそも存在しません。monolithic レイアウトの HNSW では、
ストアは writer を**コミットを跨いで**キャッシュし続けます（Issue #864）。
コミット直後の writer のメモリ内状態は、たった今書き出したファイルと等価で
あるため、コミット後最初の upsert はストレージから `.hnsw` 全体をリロード
（+ 全 dequantize）する代わりに、その場で writer を拡張します —
コミットの多いインジェストでは、このリロードがサイクルあたり O(インデックス
サイズ) のコストでした。ただし、writer の**裏で**インデックスが書き換えられる
場合 — コミット中の auto-compaction 発火、または明示的な `optimize()` —
はキャッシュを破棄します。どちらも fresh writer 経由でファイルを再構築して
deletion bitmap をクリアするため、stale な保持 writer は回収済みベクトルを
書き戻してしまうからです。また、コミットが途中で失敗した場合も writer と
ディスクの一致が不明になるため破棄します。副次効果として、保留変更のない
保持 writer はインデックス書き出し自体をスキップするため、変更なしの
`commit()` は `.hnsw` への書き込みコストがゼロになります。monolithic の
Flat と IVF は従来のコミット時破棄の挙動を維持しており、同じ不変条件の
監査はまだ済んでいません — 現在この経路に到達するには segmented デフォルトを
明示的にオプトアウトする必要があります。

## コード例

```rust
use std::sync::Arc;
use laurus::{Document, Engine, Schema};
use laurus::lexical::TextOption;
use laurus::vector::HnswOption;
use laurus::vector::core::distance::DistanceMetric;
use laurus::storage::memory::MemoryStorage;

#[tokio::main]
async fn main() -> laurus::Result<()> {
    let storage = Arc::new(MemoryStorage::new(Default::default()));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_hnsw_field("embedding", HnswOption {
            dimension: 384,
            distance: DistanceMetric::Cosine,
            m: 16,
            ef_construction: 200,
            ..Default::default()
        })
        .build();

    // With an embedder, text in vector fields is automatically embedded
    let engine = Engine::builder(storage, schema)
        .embedder(my_embedder)
        .build()
        .await?;

    // Add text to the vector field — it will be embedded automatically
    engine.add_document("doc-1", Document::builder()
        .add_text("title", "Rust Programming")
        .add_text("embedding", "Rust is a systems programming language.")
        .build()
    ).await?;

    engine.commit().await?;

    Ok(())
}
```

## 次のステップ

- ベクトルインデックスを検索する: [Vector 検索](../search/vector_search.md)
- Lexical 検索と組み合わせる: [ハイブリッド検索](../search/hybrid_search.md)
