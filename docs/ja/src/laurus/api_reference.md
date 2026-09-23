# APIリファレンス

このページでは、Laurusの最も重要な型とメソッドのクイックリファレンスを提供します。完全な詳細については、Rustdocを生成してください。

```bash
cargo doc --open
```

## Engine

すべてのインデキシングと検索操作を統合する中心的なコーディネーターです。

| メソッド | 説明 |
| :--- | :--- |
| `Engine::builder(storage, schema)` | `EngineBuilder` を作成 |
| `engine.put_document(id, doc).await?` | ドキュメントのUpsert（IDが存在する場合は置き換え） |
| `engine.add_document(id, doc).await?` | ドキュメントをチャンクとして追加（複数のチャンクが同一IDを共有可能） |
| `engine.put_documents(docs).await?` | `(id, doc)` ペアのバッチ Upsert — WAL fsync はバッチごとに 1 回（[バッチインジェスト](persistence.md#バッチインジェスト) 参照） |
| `engine.add_documents(docs).await?` | `(id, doc)` ペアのバッチチャンク追加 — 耐久性・エラー意味論は `put_documents` と同一 |
| `engine.delete_documents(id).await?` | 外部IDによるすべてのドキュメント/チャンクの削除 |
| `engine.get_documents(id).await?` | 外部IDによるすべてのドキュメント/チャンクの取得 |
| `engine.search(request).await?` | 検索リクエストの実行 |
| `engine.commit().await?` | 保留中のすべての変更をストレージにフラッシュ |
| `engine.flush_wal()?` | full commit なしで WAL を durable 化（[WAL 耐久性ポリシー](persistence.md#wal-耐久性ポリシー)参照） |
| `engine.add_field(name, field_option).await?` | 稼働中のエンジンにフィールドを動的に追加 |
| `engine.delete_field(name).await?` | 稼働中のエンジンからフィールドを動的に削除 |
| `engine.schema()` | 現在のスキーマへの参照を取得 |
| `engine.stats()?` | インデックス統計の取得 |

> **`put_document` と `add_document` の違い:** `put_document` はUpsertを実行します。同じ外部IDのドキュメントが既に存在する場合、削除して置き換えます。`add_document` は常に追加し、複数のドキュメントチャンクが同じ外部IDを共有できます。詳細は [Schema & Fields -- ドキュメントのインデキシング](../concepts/schema_and_fields.md#indexing-documents) を参照してください。
>
> **バッチ形式:** `put_documents` / `add_documents` は `(id, doc)` ペアを入力順に逐次適用します（1 回の `put_documents` バッチ内で重複した ID は逐次 put と同じくデデュープされ、最後の出現が勝ちます）。適用できないドキュメントに遭遇すると fail-fast で `LaurusError::BatchIngest { failed_index, failed_id, applied, .. }` を返します。バッチロールバックはありません: 適用済みドキュメントは WAL と NRT バッファに残るため、バッチ（またはその suffix）の再試行は冪等です。耐久性の詳細は[バッチインジェスト](persistence.md#バッチインジェスト)を参照してください。

### EngineBuilder

| メソッド | 説明 |
| :--- | :--- |
| `EngineBuilder::new(storage, schema)` | StorageとSchemaでBuilderを作成 |
| `.analyzer(Arc<dyn Analyzer>)` | テキストAnalyzerを設定（デフォルト: `StandardAnalyzer`） |
| `.embedder(Arc<dyn Embedder>)` | ベクトルEmbedderを設定（オプション） |
| `.wal_sync_policy(policy)` | WAL 耐久性ポリシーを設定（デフォルト: `WalSyncPolicy::PerRecord`。[WAL 耐久性ポリシー](persistence.md#wal-耐久性ポリシー)参照） |
| `.build().await?` | `Engine` を構築 |

## Schema

ドキュメント構造を定義します。

| メソッド | 説明 |
| :--- | :--- |
| `Schema::builder()` | `SchemaBuilder` を作成 |

### SchemaBuilder

| メソッド | 説明 |
| :--- | :--- |
| `.add_text_field(name, TextOption)` | 全文検索フィールドを追加 |
| `.add_integer_field(name, IntegerOption)` | 整数フィールドを追加（`IntegerOption::multi_valued = true` で多値配列対応） |
| `.add_float_field(name, FloatOption)` | 浮動小数点フィールドを追加（`FloatOption::multi_valued = true` で多値配列対応） |
| `.add_boolean_field(name, BooleanOption)` | 真偽値フィールドを追加（`BooleanOption::multi_valued = true` で真偽値配列対応） |
| `.add_datetime_field(name, DateTimeOption)` | 日時フィールドを追加（`DateTimeOption::multi_valued = true` で時刻配列対応） |
| `.add_geo_field(name, GeoOption)` | 2D 地理（緯度/経度）フィールドを追加（`GeoOption::multi_valued = true` でポイント配列対応） |
| `.add_geo3d_field(name, Geo3dOption)` | 3D ECEF 直交座標系の点フィールド（x, y, z メートル単位）を追加（`Geo3dOption::multi_valued = true` でポイント配列対応） |
| `.add_bytes_field(name, BytesOption)` | バイナリフィールドを追加 |
| `.add_hnsw_field(name, HnswOption)` | HNSWベクトルフィールドを追加 |
| `.add_flat_field(name, FlatOption)` | Flatベクトルフィールドを追加 |
| `.add_ivf_field(name, IvfOption)` | IVFベクトルフィールドを追加 |
| `.add_default_field(name)` | デフォルト検索フィールドを設定 |
| `.add_analyzer(name, AnalyzerDefinition)` | カスタム Analyzer パイプラインを登録 |
| `.add_embedder(name, EmbedderDefinition)` | Embedder 定義を登録 |
| `.dynamic_field_policy(DynamicFieldPolicy)` | 未宣言フィールドのポリシー（`Strict` / `Dynamic` / `Ignore`）を設定 |
| `.build()` | `Schema` を構築 |

## Document

名前付きフィールド値のコレクションです。

| メソッド | 説明 |
| :--- | :--- |
| `Document::builder()` | `DocumentBuilder` を作成 |
| `doc.get(name)` | 名前でフィールド値を取得 |
| `doc.has_field(name)` | フィールドが存在するか確認 |
| `doc.field_names()` | すべてのフィールド名を取得 |

### DocumentBuilder

| メソッド | 説明 |
| :--- | :--- |
| `.add_field(name, DataValue)` | 任意の `DataValue` を追加 |
| `.add_text(name, value)` | テキストフィールドを追加 |
| `.add_integer(name, value)` | 単一値の整数フィールドを追加 |
| `.add_float(name, value)` | 単一値の浮動小数点フィールドを追加 |
| `.add_boolean(name, value)` | 真偽値フィールドを追加 |
| `.add_datetime(name, value)` | 日時フィールドを追加 |
| `.add_vector(name, vec)` | 事前計算済みベクトルを追加 |
| `.add_geo(name, lat, lon)` | 2D 地理ポイントを追加 |
| `.add_geo_ecef(name, x, y, z)` | 3D ECEF 直交座標系の点（メートル単位）を追加 |
| `.add_int64_array(name, values)` | 多値整数フィールドを追加 |
| `.add_float64_array(name, values)` | 多値浮動小数点フィールドを追加 |
| `.add_geo_array(name, points)` | 多値 2D 地理フィールドを追加（`Vec<GeoPoint>`） |
| `.add_geo_ecef_array(name, points)` | 多値 3D ECEF フィールドを追加（`Vec<GeoEcefPoint>`） |
| `.add_datetime_array(name, values)` | 多値日時フィールドを追加（`Vec<DateTime<Utc>>`） |
| `.add_bool_array(name, values)` | 多値真偽値フィールドを追加（`Vec<bool>`） |
| `.add_bytes(name, data)` | バイナリデータを追加 |
| `.build()` | `Document` を構築 |

## Search

### SearchRequestBuilder

| メソッド | 説明 |
| :--- | :--- |
| `SearchRequestBuilder::new()` | 新しいBuilderを作成 |
| `.query_dsl(dsl)` | 統合DSLクエリ文字列を設定 |
| `.lexical_query(query)` | Lexical検索クエリを設定（`LexicalSearchQuery`） |
| `.vector_query(query)` | Vector検索クエリを設定（`VectorSearchQuery`） |
| `.filter_query(query)` | プレフィルタクエリを設定 |
| `.fusion_algorithm(algo)` | フュージョンアルゴリズムを設定（デフォルト: RRF） |
| `.limit(n)` | 最大結果数（デフォルト: 10） |
| `.offset(n)` | N件スキップ（デフォルト: 0） |
| `.add_field_boost(field, boost)` | Lexical検索のフィールドブーストを追加 |
| `.lexical_min_score(f32)` | Lexical検索の最小スコアしきい値 |
| `.lexical_timeout_ms(u64)` | Lexical検索のタイムアウト（ミリ秒） |
| `.lexical_parallel(bool)` | Lexical検索の並列実行を有効化 |
| `.sort_by(SortField)` | Lexical検索のソート順を設定 |
| `.highlight(fields)` | これらの保存済みテキストフィールドに対するハイライトを要求（[ハイライト](./highlighting.md)） |
| `.highlight_config(HighlightConfig)` | ハイライトに使うタグ・フラグメント・`require_field_match` の設定を指定 |
| `.vector_score_mode(VectorScoreMode)` | Vector検索のスコア結合モードを設定 |
| `.vector_min_score(f32)` | Vector検索の最小スコアしきい値 |
| `.build()` | `SearchRequest` を構築 |

### VectorSearchRequestBuilder

| メソッド | 説明 |
| :--- | :--- |
| `VectorSearchRequestBuilder::new()` | 新しいBuilderを作成 |
| `.add_text(field, text)` | フィールドのテキストクエリを追加 |
| `.add_vector(field, vector)` | 事前計算済みクエリベクトルを追加 |
| `.add_bytes(field, bytes, mime)` | バイナリペイロードを追加（マルチモーダル用） |
| `.limit(n)` | 最大結果数 |
| `.score_mode(VectorScoreMode)` | スコア結合モード（WeightedSum、MaxSim） |
| `.min_score(f32)` | 最小スコアしきい値 |
| `.field(name)` | 検索を特定のフィールドに制限 |
| `.build()` | リクエストを構築 |

### SearchResult

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `id` | `String` | 外部ドキュメントID |
| `score` | `f32` | 関連度スコア |
| `document` | `Option<Document>` | ドキュメント内容（ロードされた場合） |
| `highlights` | `HashMap<String, Vec<String>>` | `.highlight()` で要求したフィールドごとのハイライト済みフラグメント。未要求またはマッチなしの場合は空（[ハイライト](./highlighting.md)） |

### FusionAlgorithm

| バリアント | 説明 |
| :--- | :--- |
| `RRF { k: f64 }` | Reciprocal Rank Fusion（デフォルト k=60.0） |
| `WeightedSum { lexical_weight, vector_weight }` | スコアの線形結合 |

## クエリタイプ（Lexical）

| クエリ | 説明 | 例 |
| :--- | :--- | :--- |
| `TermQuery::new(field, term)` | 完全一致 | `TermQuery::new("body", "rust")` |
| `PhraseQuery::new(field, terms)` | フレーズ一致 | `PhraseQuery::new("body", vec!["machine".into(), "learning".into()])` |
| `BooleanQueryBuilder::new()` | ブール結合 | `.must(q1).should(q2).must_not(q3).build()` |
| `FuzzyQuery::new(field, term)` | あいまい一致（デフォルト max_edits=2） | `FuzzyQuery::new("body", "programing").max_edits(1)` |
| `WildcardQuery::new(field, pattern)` | ワイルドカード | `WildcardQuery::new("file", "*.pdf")` |
| `NumericRangeQuery::new(...)` | 数値範囲 | [Lexical Search](../concepts/search.md) を参照 |
| `DateTimeRangeQuery::between(field, start, end)` | 日時範囲（`new(...)` は `Option<DateTime<Utc>>` の境界、`from_literals(...)?` は DSL リテラルを受け取る） | [Lexical Search](../concepts/search.md) を参照 |
| `GeoDistanceQuery::within_radius(...)` | 2D 地理半径 | [Lexical Search](../concepts/search.md) を参照 |
| `GeoBoundingBoxQuery::within_bounding_box(...)` | 2D 地理バウンディングボックス | [Lexical Search](../concepts/search.md) を参照 |
| `Geo3dDistanceQuery::within_sphere(...)` | 3D ECEF 球 | [3D 地理検索](../concepts/geo3d.md) を参照 |
| `Geo3dBoundingBoxQuery::within_box(...)` | 3D ECEF 軸並行バウンディングボックス | [3D 地理検索](../concepts/geo3d.md) を参照 |
| `Geo3dNearestQuery::k_nearest(...)` | 3D ECEF k-NN | [3D 地理検索](../concepts/geo3d.md) を参照 |
| `SpanNearQuery::new(...)` | 近接 | [Lexical Search](../concepts/search.md) を参照 |
| `PrefixQuery::new(field, prefix)` | 前方一致 | `PrefixQuery::new("body", "pro")` |
| `RegexpQuery::new(field, pattern)?` | 正規表現一致 | `RegexpQuery::new("body", "^pro.*ing$")?` |

## クエリパーサー

| パーサー | 説明 |
| :--- | :--- |
| `LexicalQueryParser::new(analyzer)` | Lexical DSL クエリをパース |
| `VectorQueryParser::new(embedder)` | Vector DSL クエリをパース |
| `UnifiedQueryParser::new(lexical, vector)` | Lexical / Vector 両方を含むハイブリッド DSL クエリをパース |

## Analyzer

| 型 | 説明 |
| :--- | :--- |
| `StandardAnalyzer` | RegexTokenizer + 小文字化 + ストップワード |
| `SimpleAnalyzer` | トークン化のみ（フィルタリングなし） |
| `EnglishAnalyzer` | RegexTokenizer + 小文字化 + 英語ストップワード |
| `JapaneseAnalyzer` | 日本語形態素解析 |
| `KeywordAnalyzer` | トークン化なし（完全一致） |
| `PipelineAnalyzer` | カスタムTokenizer + フィルタチェーン |
| `PerFieldAnalyzer` | フィールドごとのAnalyzerディスパッチ |

## Embedder

| 型 | Feature Flag | 説明 |
| :--- | :--- | :--- |
| `CandleBertEmbedder` | `embeddings-candle` | ローカルBERTモデル |
| `OpenAIEmbedder` | `embeddings-openai` | OpenAI API |
| `CandleClipEmbedder` | `embeddings-multimodal` | ローカルCLIPモデル |
| `PrecomputedEmbedder` | *(デフォルト)* | 事前計算済みベクトル |
| `PerFieldEmbedder` | *(デフォルト)* | フィールドごとのEmbedderディスパッチ |

## Storage

| 型 | 説明 |
| :--- | :--- |
| `MemoryStorage` | インメモリ（非永続） |
| `FileStorage` | ファイルシステムベース（メモリマップドI/O用の `use_mmap` をサポート） |
| `StorageFactory::create(config)` | 設定から作成 |

## DataValue

| バリアント | Rust型 |
| :--- | :--- |
| `DataValue::Null` | -- |
| `DataValue::Bool(bool)` | `bool` |
| `DataValue::Int64(i64)` | `i64` |
| `DataValue::Float64(f64)` | `f64` |
| `DataValue::Text(String)` | `String` |
| `DataValue::Bytes(Vec<u8>, Option<String>)` | `(data, mime_type)` |
| `DataValue::Vector(Vec<f32>)` | 事前計算済みベクトル |
| `DataValue::DateTime(DateTime<Utc>)` | `chrono::DateTime<Utc>` |
| `DataValue::Geo(GeoPoint)` | `(latitude, longitude)`（WGS84） |
| `DataValue::GeoEcef(GeoEcefPoint)` | `(x, y, z)` ECEF 直交座標系（メートル単位） |
| `DataValue::Int64Array(Vec<i64>)` | 多値整数（`multi_valued` フィールドオプションが必要） |
| `DataValue::Float64Array(Vec<f64>)` | 多値浮動小数点数（`multi_valued` フィールドオプションが必要） |
| `DataValue::GeoArray(Vec<GeoPoint>)` | 多値 2D 地理ポイント（`multi_valued` フィールドオプションが必要） |
| `DataValue::GeoEcefArray(Vec<GeoEcefPoint>)` | 多値 3D ECEF ポイント（`multi_valued` フィールドオプションが必要） |
| `DataValue::DateTimeArray(Vec<DateTime<Utc>>)` | 多値の時刻（`multi_valued` フィールドオプションが必要） |
| `DataValue::BoolArray(Vec<bool>)` | 多値の真偽値（`multi_valued` フィールドオプションが必要） |
