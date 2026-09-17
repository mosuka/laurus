# Engine

`Engine` はLaurusの中心的な型です。Lexicalインデックス、Vectorインデックス、およびドキュメントログを単一の非同期APIで統合します。

## Engine構造体

```rust
pub struct Engine {
    schema: Schema,
    lexical: LexicalStore,
    vector: VectorStore,
    log: Arc<DocumentLog>,
}
```

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `schema` | `Schema` | フィールド定義とルーティングルール |
| `lexical` | `LexicalStore` | キーワード検索用の転置インデックス（Inverted Index） |
| `vector` | `VectorStore` | 類似度検索用のベクトルインデックス |
| `log` | `Arc<DocumentLog>` | クラッシュリカバリとドキュメント保存のためのWrite-Ahead Log |

## EngineBuilder

`EngineBuilder` を使用してEngineを設定・構築します。

```rust
use std::sync::Arc;
use laurus::{Engine, Schema};
use laurus::lexical::TextOption;
use laurus::storage::memory::MemoryStorage;

let storage = Arc::new(MemoryStorage::new(Default::default()));
let schema = Schema::builder()
    .add_text_field("title", TextOption::default())
    .add_text_field("body", TextOption::default())
    .add_default_field("body")
    .build();

let engine = Engine::builder(storage, schema)
    .analyzer(my_analyzer)            // オプション: カスタムテキストAnalyzer
    .embedder(my_embedder)            // オプション: ベクトルEmbedder
    .embedding_cache_capacity(1024)   // オプション: クエリEmbeddingをキャッシュ
    .build()
    .await?;
```

### Builderメソッド

| メソッド | パラメータ | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `analyzer()` | `Arc<dyn Analyzer>` | `StandardAnalyzer` | Lexicalフィールド用のテキスト解析パイプライン |
| `embedder()` | `Arc<dyn Embedder>` | None | Vectorフィールド用のEmbeddingモデル |
| `embedding_cache_capacity()` | `usize` | None（無効） | 最大 N 件のクエリEmbeddingをLRUキャッシュする |
| `wal_sync_policy()` | `WalSyncPolicy` | `PerRecord` | WAL fsync の耐久性（per-record か group commit か） |
| `commit_policy()` | `CommitPolicy` | `Manual` | 自動コミットのタイミング — 例: `EveryDocs(n)` は `n` 件ごとにコミット |
| `build()` | -- | -- | Engineを構築（非同期） |

### クエリEmbeddingキャッシュ

`embedding_cache_capacity(n)` は **クエリ時** のEmbeddingに対するLRU
キャッシュを有効化します（ドキュメント取り込み時のEmbeddingは対象外）。
Vector / hybrid 検索でクエリペイロードをEmbeddingする際、結果を
`(field, embedder 名, payload ハッシュ)` をキーにキャッシュし、以降の同一
クエリで再利用します。DSL パス（例: `content:"cute kitten"`）と事前
Embedding 済みの `Payloads` パスは同じキャッシュを共有します。

これにより、繰り返しクエリのワークロード（オートコンプリート、ダッシュ
ボード更新、A/B 評価など）で、ローカル Embedder のモデル推論やリモート
Embedder のネットワークラウンドトリップを回避できます。デフォルトでは
無効です。識別可能なクエリの working set に応じてメモリを抑える容量を
指定してください。

1 つのクエリ内では、その全 payload を 1 回の `Embedder::embed_batch`
呼び出しでEmbedding します（キャッシュミスのみ、`PerFieldEmbedder` では
field ごとにグルーピング）。これにより、バッチ対応 Embedder ではマルチ
ベクトルクエリでも payload ごとではなく 1 往復で済みます（Issue #671）。

### Buildライフサイクル

`build()` が呼び出されると、以下の処理が実行されます。

```mermaid
sequenceDiagram
    participant User
    participant EngineBuilder
    participant Engine

    User->>EngineBuilder: .build().await
    EngineBuilder->>EngineBuilder: split_schema()
    Note over EngineBuilder: Separate fields into<br/>LexicalIndexConfig<br/>+ VectorIndexConfig
    EngineBuilder->>Engine: Create PrefixedStorage (lexical/, vector/, documents/)
    EngineBuilder->>Engine: Create LexicalStore
    EngineBuilder->>Engine: Create VectorStore
    EngineBuilder->>Engine: Create DocumentLog
    EngineBuilder->>Engine: Recover from WAL
    EngineBuilder-->>User: Engine ready
```

1. **スキーマの分割** -- Lexicalフィールド（Text、Integer、Floatなど）は `LexicalIndexConfig` に、Vectorフィールド（HNSW、Flat、IVF）は `VectorIndexConfig` に分割されます
2. **プレフィックス付きストレージの作成** -- 各コンポーネントに独立した名前空間が割り当てられます（`lexical/`、`vector/`、`documents/`）
3. **ストアの初期化** -- `LexicalStore` と `VectorStore` がそれぞれの設定で作成されます
4. **WALからのリカバリ** -- 前回のセッションでコミットされなかった操作がリプレイされます

## スキーマの分割

Schemaにはlexicalフィールドとvectorフィールドの両方が含まれています。ビルド時に `split_schema()` がこれらを分離します。

```mermaid
graph LR
    S["Schema<br/>title: Text<br/>body: Text<br/>page: Integer<br/>content_vec: HNSW"]

    S --> LC["LexicalIndexConfig<br/>title: TextOption<br/>body: TextOption<br/>page: IntegerOption<br/>_id: KeywordAnalyzer"]

    S --> VC["VectorIndexConfig<br/>content_vec: HnswOption<br/>(dim=384, m=16, ef=200)"]
```

予約フィールド `_id` は、完全一致検索のために `KeywordAnalyzer` を用いて常にLexical設定に追加されます。

## フィールドごとのディスパッチ

### PerFieldAnalyzer

`PerFieldAnalyzer` が指定された場合、テキスト解析はフィールドごとのAnalyzerにディスパッチされます。

```mermaid
graph LR
    PFA["PerFieldAnalyzer"]
    PFA -->|"title"| KA["KeywordAnalyzer"]
    PFA -->|"body"| SA["StandardAnalyzer"]
    PFA -->|"description"| JA["JapaneseAnalyzer"]
    PFA -->|"_id"| KA2["KeywordAnalyzer<br/>(always)"]
    PFA -->|other fields| DEF["Default Analyzer"]
```

### PerFieldEmbedder

同様に、`PerFieldEmbedder` はフィールドごとのEmbedderにEmbedding処理をルーティングします。

```mermaid
graph LR
    PFE["PerFieldEmbedder"]
    PFE -->|"text_vec"| BERT["CandleBertEmbedder<br/>(384 dim)"]
    PFE -->|"image_vec"| CLIP["CandleClipEmbedder<br/>(512 dim)"]
    PFE -->|other fields| DEF["Default Embedder"]
```

## Engineメソッド

### ドキュメント操作

| メソッド | 説明 |
| :--- | :--- |
| `put_document(id, doc)` | Upsert -- 同じIDのドキュメントが既存の場合は置き換え |
| `add_document(id, doc)` | 追加 -- 新しいチャンクとして追加（複数のチャンクが同一IDを共有可能） |
| `put_documents(docs)` | バッチ Upsert -- `(id, doc)` ペアを順序どおり適用し、WAL fsync はバッチごとに 1 回 |
| `add_documents(docs)` | バッチ追加 -- `put_documents` と同様だが既存チャンクを削除しない |
| `get_documents(id)` | 外部IDによるすべてのドキュメント/チャンクの取得 |
| `delete_documents(id)` | 外部IDによるすべてのドキュメント/チャンクの削除 |
| `commit()` | 保留中の変更をストレージにフラッシュ（ドキュメントが検索可能になる） |
| `recover()` | クラッシュ後にWALをリプレイして未コミット状態を復元 |
| `add_field(name, field_option)` | 稼働中のエンジンにフィールドを動的に追加し、更新後の `Schema` を返す |
| `delete_field(name)` | 稼働中のエンジンからフィールドを動的に削除し、更新後の `Schema` を返す |
| `schema()` | 現在の `Schema` への参照を返す |

### 検索

| メソッド | 説明 |
| :--- | :--- |
| `search(request)` | 統一検索の実行（Lexical、Vector、またはハイブリッド） |

`search()` メソッドは `SearchRequest` を受け取ります。SearchRequestにはLexicalクエリ、Vectorクエリ、またはその両方を含めることができます。両方が指定された場合、結果は指定された `FusionAlgorithm` で統合されます。

```rust
use laurus::{SearchRequestBuilder, FusionAlgorithm};
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::LexicalSearchQuery;

// Lexicalのみの検索
let request = SearchRequestBuilder::new()
    .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new("body", "rust"))))
    .limit(10)
    .build();

// RRFフュージョンによるハイブリッド検索
let request = SearchRequestBuilder::new()
    .lexical_query(lexical_query)
    .vector_query(vector_query)
    .fusion_algorithm(FusionAlgorithm::RRF { k: 60.0 })
    .limit(10)
    .build();

let results = engine.search(request).await?;
```

### ウォームアップ

`engine.warmup()`（Issue #677）は Vector searcher を事前に準備し、**初回**の
Vector / ハイブリッドクエリが one-time のセットアップコストを払わないようにします。
Engine 構築後、トラフィック処理を始める前に一度呼び出します:

```rust,ignore
let engine = builder.build()?;
engine.warmup()?;   // オプション: 初回クエリの latency を起動時に移す
```

Vector searcher を先行構築・キャッシュし（reader をロード — `InMemory` は
ファイル→メモリ、`Mmap` はオフセット表）、HNSW の `Mmap` モードではディスク上の
Vector データを OS page cache に pre-fault します。HNSW グラフは常に eager に
ロードされるため、warming が必要なのは Vector データのみです。`warmup()` は
複数回呼んでも安全で、lexical のみのワークロードでは no-op です。

## SearchRequest

| フィールド | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `query` | `SearchQuery` | `Dsl("")` | 検索クエリ仕様（Dsl / Lexical / Vector / Hybrid） |
| `limit` | `usize` | 10 | 返却する最大結果数 |
| `offset` | `usize` | 0 | ページネーションのオフセット |
| `fusion_algorithm` | `Option<FusionAlgorithm>` | None（ハイブリッド時はRRF k=60） | LexicalとVectorの結果を統合する方法 |
| `filter_query` | `Option<Box<dyn Query>>` | None | 両方の検索タイプに適用されるフィルタ |
| `lexical_options` | `LexicalSearchOptions` | デフォルト | Lexical検索の動作パラメータ（ブースト、タイムアウト等） |
| `vector_options` | `VectorSearchOptions` | デフォルト | Vector検索の動作パラメータ（スコアモード等） |

### LexicalSearchOptions

| フィールド | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `field_boosts` | `HashMap<String, f32>` | 空 | フィールドごとのスコア倍率 |
| `min_score` | `f32` | 0.0 | 最小スコアしきい値 |
| `timeout_ms` | `Option<u64>` | None | 検索タイムアウト（ミリ秒） |
| `parallel` | `bool` | false | セグメント検索を並列実行するか |
| `sort_by` | `SortField` | `Score` | Lexical結果のソート順 |
| `highlight` | `Option<HighlightOptions>` | None | 検索結果ハイライトの対象フィールドと設定。`None` の場合、各ヒットの `SearchResult::highlights` は空のまま。`SearchRequestBuilder::highlight` / `.highlight_config` で設定する（[ハイライト](./highlighting.md)参照）。Vector-onlyのリクエストでは無視される |

## FusionAlgorithm

| バリアント | 説明 |
| :--- | :--- |
| `RRF { k: f64 }` | Reciprocal Rank Fusion -- ランクベースの結合。スコア = sum(1 / (k + rank))。比較不可能なスコアの大きさを処理します。 |
| `WeightedSum { lexical_weight, vector_weight }` | min-maxスコア正規化を用いた加重結合。重みは[0.0, 1.0]にクランプされます。 |

関連項目: [アーキテクチャ](../architecture.md) -- 高レベルのデータフロー図
