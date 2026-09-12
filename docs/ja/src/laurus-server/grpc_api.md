# gRPC API リファレンス

すべてのサービスは `laurus.v1` protobuf パッケージで定義されています。

## サービス一覧

| サービス | RPC | 説明 |
| :--- | :--- | :--- |
| `HealthService` | `Check` | ヘルスチェック |
| `IndexService` | `CreateIndex`, `GetIndex`, `GetSchema`, `AddField`, `DeleteField` | インデックスのライフサイクルとスキーマ |
| `DocumentService` | `PutDocument`, `AddDocument`, `PutDocuments`, `AddDocuments`, `GetDocuments`, `DeleteDocuments`, `Commit`, `FlushWal` | ドキュメント CRUD・バルクインジェスト・コミット・WAL flush |
| `SearchService` | `Search`, `SearchStream` | 単発検索とストリーミング検索 |

---

## HealthService

### `Check`

サーバーの現在のサービング状態を返します。

```protobuf
rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
```

**レスポンスフィールド:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `status` | `ServingStatus` | サーバーの準備が完了している場合は `SERVING_STATUS_SERVING` |

---

## IndexService

### `CreateIndex`

指定されたスキーマで新しいインデックスを作成します。インデックスが既に開いている場合は `ALREADY_EXISTS` エラーを返します。

```protobuf
rpc CreateIndex(CreateIndexRequest) returns (CreateIndexResponse);
```

**リクエストフィールド:**

| フィールド | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `schema` | `Schema` | はい | インデックスのスキーマ定義 |

**Schema 構造:**

```protobuf
message Schema {
  map<string, FieldOption> fields = 1;
  repeated string default_fields = 2;
  map<string, AnalyzerDefinition> analyzers = 3;
  map<string, EmbedderConfig> embedders = 4;
  DynamicFieldPolicy dynamic_field_policy = 5;
}

enum DynamicFieldPolicy {
  DYNAMIC_FIELD_POLICY_UNSPECIFIED = 0;
  DYNAMIC_FIELD_POLICY_STRICT = 1;
  DYNAMIC_FIELD_POLICY_DYNAMIC = 2;
  DYNAMIC_FIELD_POLICY_IGNORE = 3;
}
```

- **`fields`** — フィールド名をキーとしたフィールド定義。
- **`default_fields`** — クエリでフィールドを指定しない場合のデフォルト検索対象フィールド名。
- **`analyzers`** — 名前をキーとしたカスタムアナライザーパイプライン。`TextOption.analyzer` で参照。
- **`embedders`** — 名前をキーとしたエンベッダー設定。ベクトルフィールドオプション（`HnswOption.embedder` など）で参照。
- **`dynamic_field_policy`** — 投入されたドキュメントに含まれるが `fields` に**宣言されていない**フィールドの扱い。`UNSPECIFIED`（値 0）は後方互換のため `DYNAMIC` として解釈されます。挙動マトリクスおよび `DYNAMIC` での情報損失警告は [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

**AnalyzerDefinition:**

```protobuf
message AnalyzerDefinition {
  repeated ComponentConfig char_filters = 1;
  ComponentConfig tokenizer = 2;
  repeated ComponentConfig token_filters = 3;
}
```

**ComponentConfig**（文字フィルター、トークナイザー、トークンフィルターに使用）:

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `type` | `string` | コンポーネントタイプ名（例: `"whitespace"`, `"lowercase"`, `"unicode_normalization"`） |
| `params` | `map<string, string>` | タイプ固有のパラメータ（文字列のキーと値のペア） |

**EmbedderConfig:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `type` | `string` | エンベッダータイプ名（例: `"precomputed"`, `"candle_bert"`, `"openai"`） |
| `params` | `map<string, string>` | タイプ固有のパラメータ（例: `"model"` → `"sentence-transformers/all-MiniLM-L6-v2"`） |

各 `FieldOption` は以下のフィールドタイプのいずれかを持つ `oneof` です。

| Lexical フィールド | Vector フィールド |
| :--- | :--- |
| `TextOption` (`indexed`, `stored`, `term_vectors`, `doc_values`, `analyzer`) | `HnswOption` (`dimension`, `distance`, `m`, `ef_construction`, `base_weight`, `quantizer`, `embedder`, `rerank_storage`, `pq_codebook_path`) |
| `IntegerOption` (`indexed`, `stored`, `multi_valued`, `doc_values`) | `FlatOption` (`dimension`, `distance`, `base_weight`, `quantizer`, `embedder`, `rerank_storage`) |
| `FloatOption` (`indexed`, `stored`, `multi_valued`, `doc_values`) | `IvfOption` (`dimension`, `distance`, `n_clusters`, `n_probe`, `base_weight`, `quantizer`, `embedder`, `rerank_storage`) |
| `BooleanOption` (`indexed`, `stored`, `doc_values`) | |
| `DateTimeOption` (`indexed`, `stored`, `doc_values`) | |
| `GeoOption` (`indexed`, `stored`, `doc_values`) | |
| `Geo3dOption` (`indexed`, `stored`, `doc_values`) | |
| `BytesOption` (`stored`) | |

ベクトルフィールドオプションの `embedder` フィールドには、`Schema.embedders` で定義したエンベッダー名を指定します。設定すると、インデックス時にドキュメントのテキストフィールドからベクトルを自動生成します。事前計算済みのベクトルを直接供給する場合は空のままにします。

**Doc values:** `doc_values`（Issue #1047）は、上記の `BytesOption` を除く全ての lexical オプションで `optional bool` であり、`term_vectors` と同じ tri-state の契約に従います。クライアントが省略するとエンジンのデフォルト（`true`）になり、明示的な `false` とは区別されます。フィールドの値を DocValues（ソート・ファセット・集計が読み取る列指向ストア）にもコピーするかどうかを制御し、DocValues 列が書き込まれるのは `stored` と `doc_values` の両方が `true` の場合のみです。`BytesOption` にはこのフィールドがありません —— `Bytes` の値は設定にかかわらず DocValues に一切書き込まれないためです。ソートにもファセットにも使わないフィールドで `doc_values` を無効にするとセグメントの使用容量が削減されます。フィールド自体は引き続き完全に検索・取得可能です。

**距離メトリクス:** `COSINE`, `EUCLIDEAN`, `MANHATTAN`, `DOT_PRODUCT`, `ANGULAR`

**量子化手法:** `SCALAR_8BIT`（デフォルト）, `PRODUCT_QUANTIZATION`（Issue #481 Stage 3。HNSW インデックスがサポート — Flat / IVF は書き込み時に拒否）

`NONE`（量子化なし）は Issue #481 Stage 1 で廃止されました。proto enum 値 0（`QUANTIZATION_METHOD_NONE`）は wire 互換のため予約されていますが、サーバ側で受信すると `Default::default()`（`SCALAR_8BIT`）にフォールバックします。

**Rerank storage:** オプションの `rerank_storage` フィールド（enum `RerankStorageKind`: `UNSPECIFIED` = サイドカーなし、`F32`）は Stage-2 rerank サイドカー（Issue #481 / #793）を有効化します。HNSW フィールドで `F32` を設定すると、commit 時に完全精度の `.hnsw.f32` サイドカーを追加で書き出し、`rerank_factor` を指定した検索が int8 候補を元のベクトルで再スコアします。フィールドを省略（または `UNSPECIFIED`）すると Stage-1 の int8 のみのランキングになります。#932 以降、サイドカーは 3 つのベクトルインデックスタイプ（HNSW / Flat / IVF）すべてで出力・利用されます（Flat / IVF の再スコアはフィールド指定クエリに適用）。

**共有 PQ codebook:** `HnswOption` のオプションフィールド `pq_codebook_path`（Issue #631）は、`laurus train pq-codebook` CLI コマンドで一度だけ学習するストレージ相対の共有 PQ codebook ファイルを指定します。設定すると segment は commit / merge のたびに k-means を再学習する代わりに、学習済み codebook で encode されます。`PRODUCT_QUANTIZATION` quantizer との組み合わせでのみ意味を持ち、設定済みで未学習の場合、commit は学習コマンドを示すエラーで失敗します（per-segment 学習への無言のフォールバック無し）。未設定なら per-segment 学習のままです。

**Base weight:** `HnswOption`/`FlatOption`/`IvfOption` の `base_weight` は `optional float`（Issue #1084）です。クライアントが省略するとエンジンのデフォルト（`1.0`）が使われ、これは明示的な `0.0` とは区別されます。他の vector フィールドと同時にクエリ対象になったときの、そのフィールドの相対的なスコアリング優先度を設定します。何に効いて何に効かないかは[ウェイト](../concepts/search/vector_search.md#ウェイト)を参照してください。

**QuantizationConfig 構造:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `method` | `QuantizationMethod` | 量子化手法（`QUANTIZATION_METHOD_SCALAR_8BIT` または `QUANTIZATION_METHOD_PRODUCT_QUANTIZATION`）。0（`NONE`）は予約、サーバ側で `SCALAR_8BIT` にフォールバック。 |
| `subvector_count` | `uint32` | サブベクトルの数（`method` が `PRODUCT_QUANTIZATION` の場合のみ使用。`dimension` を均等に割り切れる値を指定）。 |

**例:**

```json
{
  "schema": {
    "fields": {
      "title": {"text": {"indexed": true, "stored": true, "term_vectors": true}},
      "embedding": {"hnsw": {"dimension": 384, "distance": "DISTANCE_METRIC_COSINE", "m": 16, "ef_construction": 200}}
    },
    "default_fields": ["title"]
  }
}
```

### `GetIndex`

インデックスの統計情報を取得します。

```protobuf
rpc GetIndex(GetIndexRequest) returns (GetIndexResponse);
```

**レスポンスフィールド:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `document_count` | `uint64` | インデックス内のドキュメント総数 |
| `vector_fields` | `map<string, VectorFieldStats>` | フィールドごとのベクトル統計情報 |

各 `VectorFieldStats` には `vector_count` と `dimension` が含まれます。

### `GetSchema`

現在のインデックススキーマを取得します。

```protobuf
rpc GetSchema(GetSchemaRequest) returns (GetSchemaResponse);
```

**レスポンスフィールド:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `schema` | `Schema` | インデックスのスキーマ |

### `AddField`

稼働中のインデックスにフィールドを動的に追加します。

```protobuf
rpc AddField(AddFieldRequest) returns (AddFieldResponse);
```

**リクエストフィールド:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `name` | `string` | フィールド名 |
| `field_option` | `FieldOption` | フィールド設定 |

**レスポンスフィールド:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `schema` | `Schema` | フィールド追加後の更新済みスキーマ |

**HTTP ゲートウェイ:** `POST /v1/schema/fields`

### DeleteField

稼働中のインデックスからフィールドを動的に削除します。既にインデックスされたデータは残りますが、削除されたフィールドにはアクセスできなくなります。

```protobuf
rpc DeleteField(DeleteFieldRequest) returns (DeleteFieldResponse);

message DeleteFieldRequest {
  string name = 1;
}

message DeleteFieldResponse {
  Schema schema = 1;
}
```

**リクエストフィールド:**

| フィールド | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `name` | `string` | はい | 削除するフィールド名 |

**レスポンス:** 更新後の `Schema` を返します。

---

## DocumentService

### `PutDocument`

ID を指定してドキュメントを挿入または置換します。同じ ID のドキュメントが既に存在する場合は置換されます。

```protobuf
rpc PutDocument(PutDocumentRequest) returns (PutDocumentResponse);
```

**リクエストフィールド:**

| フィールド | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `id` | `string` | はい | 外部ドキュメント ID |
| `document` | `Document` | はい | ドキュメントの内容 |

**Document 構造:**

```protobuf
message Document {
  map<string, Value> fields = 1;
}
```

各 `Value` は以下の型のいずれかを持つ `oneof` です。

| 型 | Proto フィールド | 説明 |
| :--- | :--- | :--- |
| Null | `null_value` | Null 値 |
| Boolean | `bool_value` | ブール値 |
| Integer | `int64_value` | 64 ビット符号付き整数 |
| Float | `float64_value` | 64 ビット浮動小数点数 |
| Text | `text_value` | UTF-8 文字列 |
| Bytes | `bytes_value` | バイト列 |
| Vector | `vector_value` | `VectorValue`（浮動小数点数のリスト） |
| DateTime | `datetime_value` | Unix マイクロ秒（UTC） |
| Geo | `geo_value` | `GeoPoint`（緯度、経度） |
| Int64Array | `int64_array_value` | `Int64ArrayValue`（多値整数。`IntegerOption.multi_valued = true` を要求） |
| Float64Array | `float64_array_value` | `Float64ArrayValue`（多値浮動小数点数。`FloatOption.multi_valued = true` を要求） |
| Geo3d | `geo3d_value` | `Geo3dPoint`（x, y, z メートル単位、ECEF 直交座標系） |

**Geo3dPoint:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `x` | `double` | X 座標（メートル単位、ECEF: 赤道面、+X 方向は経度 0°） |
| `y` | `double` | Y 座標（メートル単位、ECEF: 赤道面、+Y 方向は東経 90°） |
| `z` | `double` | Z 座標（メートル単位、ECEF: +Z 方向は北極） |

座標系の詳細および `wgs84_to_ecef` / `ecef_to_wgs84` の変換ユーティリティについては [3D 地理検索 (ECEF)](../concepts/geo3d.md) を参照してください。

### `AddDocument`

ドキュメントを追加します。`PutDocument` と異なり、同じ ID の既存ドキュメントを置換しません。複数のドキュメントが同じ ID を共有できます（チャンキングパターン）。

```protobuf
rpc AddDocument(AddDocumentRequest) returns (AddDocumentResponse);
```

リクエストフィールドは `PutDocument` と同じです。

### `PutDocuments`

バッチ Upsert。エントリは入力順に逐次適用され、バッチ全体で WAL fsync は 1 回です — ドキュメントごとに `PutDocument` を呼ぶよりはるかに高速です。1 バッチ内で重複した ID は、同じ put を 1 件ずつ発行した場合とまったく同じようにデデュープされます（最後の出現が勝ち）。

```protobuf
rpc PutDocuments(PutDocumentsRequest) returns (PutDocumentsResponse);

message DocumentEntry {
  string id = 1;
  Document document = 2;
}

message PutDocumentsRequest {
  repeated DocumentEntry documents = 1;
}

message PutDocumentsResponse {
  uint32 applied = 1; // 成功時はリクエストサイズと一致
}
```

適用できない最初のエントリで fail-fast します。適用済みエントリはロールバック**されず**（次のコミットで永続化）、エラーステータスのメッセージに失敗位置・その ID・適用済み件数が含まれるため、バッチ（またはその suffix）の再試行は冪等です。呼び出し側の誤り（スキーマ違反など）で失敗したバッチは `INVALID_ARGUMENT`、ストレージ障害は `INTERNAL` を返します。

### `AddDocuments`

バッチチャンク追加。`PutDocuments` と同様ですが既存ドキュメントを削除しないため、同一論理ドキュメントの複数チャンクを追加する目的で ID をバッチ内で繰り返せます。

```protobuf
rpc AddDocuments(AddDocumentsRequest) returns (AddDocumentsResponse);
```

リクエスト/レスポンスのフィールドは `PutDocuments` と対になります。

### `GetDocuments`

指定された外部 ID に一致するすべてのドキュメントを取得します。

```protobuf
rpc GetDocuments(GetDocumentsRequest) returns (GetDocumentsResponse);
```

**リクエストフィールド:**

| フィールド | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `id` | `string` | はい | 外部ドキュメント ID |

**レスポンスフィールド:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `documents` | `repeated Document` | 一致するドキュメント |

### `DeleteDocuments`

指定された外部 ID に一致するすべてのドキュメントを削除します。

```protobuf
rpc DeleteDocuments(DeleteDocumentsRequest) returns (DeleteDocumentsResponse);
```

### `Commit`

保留中の変更（追加および削除）をインデックスにコミットします。コミットされるまで、変更は検索に反映されません。

```protobuf
rpc Commit(CommitRequest) returns (CommitResponse);
```

### `FlushWal`

バッファされた WAL レコードを full commit なしで durable 化します。両メッセージとも空です。デフォルトの per-record sync ポリシーでは near no-op です（各書き込みは既に fsync 済み）。グループコミットポリシーでは、現在の partial batch をオンデマンドで flush し、クラッシュ時の損失窓を抑えます。`Commit` と異なりセグメントを materialize しないため、バッファされた変更は後続の `Commit` まで検索に反映されません。

```protobuf
rpc FlushWal(FlushWalRequest) returns (FlushWalResponse);

message FlushWalRequest {}

message FlushWalResponse {}
```

WAL の耐久性ポリシーはサーバ側の `[index.wal]` 設定セクションで構成します。[設定 → `[index.wal]` セクション](configuration.md#indexwal-セクション) および [永続化と WAL → WAL 耐久性ポリシー](../laurus/persistence.md#wal-耐久性ポリシー) を参照してください。

**HTTP ゲートウェイ:** `POST /v1/flush_wal`

---

## SearchService

### `Search`

検索クエリを実行し、結果を単一のレスポンスとして返します。

```protobuf
rpc Search(SearchRequest) returns (SearchResponse);
```

**レスポンスフィールド:**

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `results` | `repeated SearchResult` | 関連度順の検索結果 |
| `total_hits` | `uint64` | マッチするドキュメントの総数（`limit`/`offset` 適用前） |

### `SearchStream`

検索クエリを実行し、結果を 1 件ずつストリーミングで返します。

```protobuf
rpc SearchStream(SearchRequest) returns (stream SearchResult);
```

### SearchRequest フィールド

| フィールド | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `query` | `string` | いいえ | [Query DSL](../concepts/query_dsl.md) による Lexical 検索クエリ |
| `query_vectors` | `repeated QueryVector` | いいえ | ベクトル検索クエリ |
| `limit` | `uint32` | いいえ | 最大結果件数（デフォルト: エンジンのデフォルト値） |
| `offset` | `uint32` | いいえ | スキップする結果件数 |
| `fusion` | `FusionAlgorithm` | いいえ | ハイブリッド検索の Fusion アルゴリズム |
| `lexical_params` | `LexicalParams` | いいえ | Lexical 検索パラメータ |
| `vector_params` | `VectorParams` | いいえ | ベクトル検索パラメータ |
| `field_boosts` | `map<string, float>` | いいえ | フィールドごとのスコアブースト |

`query` または `query_vectors` のいずれか 1 つ以上を指定する必要があります。

### 3D 地理クエリ

3D ECEF の地理クエリは `SearchRequest.query` に渡す Lexical DSL 文字列で表現します。専用のメッセージ型はなく、コアライブラリで使用される DSL 形式がそのまま gRPC 経由でも動作します。3 種類の形式があります（構文の詳細は [Query DSL → 3D 地理クエリ](../concepts/query_dsl.md#3d-geographic-queries-geo3d_) を参照）:

- `position:geo3d_distance(x, y, z, distance_m)` — `(x, y, z)` を中心とした最大距離（メートル単位）の球
- `position:geo3d_bbox(min_x, min_y, min_z, max_x, max_y, max_z)` — 3D 軸並行バウンディングボックス
- `position:geo3d_nearest(x, y, z, k)` — `(x, y, z)` に最も近い k 個の近傍点

`position` はフィールド名で、スキーマで宣言した実際の `Geo3d` 型フィールドに置き換えてください。すべての数値引数は符号付きの `double` 値で、`k` は符号なし整数です。

### QueryVector

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `vector` | `repeated float` | クエリベクトル |
| `weight` | `float` | このベクトルの重み（デフォルト: 1.0） |
| `fields` | `repeated string` | 対象のベクトルフィールド（空の場合は全フィールド） |

### FusionAlgorithm

以下の 2 つのオプションを持つ `oneof` です。

- **RRF** (Reciprocal Rank Fusion): `k` パラメータ（デフォルト: 60）
- **WeightedSum**: `lexical_weight` と `vector_weight`

### LexicalParams

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `min_score` | `float` | 最小スコア閾値 |
| `timeout_ms` | `uint64` | 検索タイムアウト（ミリ秒） |
| `parallel` | `bool` | 並列検索を有効化 |
| `sort_by` | `SortSpec` | スコアの代わりにフィールドでソート |

### SortSpec

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `field` | `string` | ソート対象のフィールド名。空文字列はスコアでソートすることを意味する |
| `order` | `SortOrder` | `SORT_ORDER_ASC`（昇順）または `SORT_ORDER_DESC`（降順） |

### VectorParams

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `fields` | `repeated string` | 対象のベクトルフィールド |
| `score_mode` | `VectorScoreMode` | `WEIGHTED_SUM`, `MAX_SIM`, または `LATE_INTERACTION` |
| `overfetch` | `float` | オーバーフェッチ係数（デフォルト: 2.0） |
| `min_score` | `float` | 最小スコア閾値 |
| `rerank_factor` | `optional uint32` | Stage 2 rerank の widening 係数（Issue #481）。`rerank_storage` を有効にしたフィールドに対してこの値を設定すると、サーバは int8/PQ 候補取得を `top_k * rerank_factor` まで広げ、元の完全精度ベクトルで再スコアしてから上位 `top_k` を返します。#932 以降 3 つのベクトルインデックスタイプ（HNSW・Flat・IVF）すべてで反映されます（Flat/IVF はフィールド指定クエリに適用）。`rerank_storage = "F32"` を設定していないフィールドでは silent に int8 ランキングへフォールバックします — f32 情報を復元することはできません。`0` または省略で rerank 無効。 |
| `ef_search` | `optional uint32` | HNSW の `ef_search` 候補リストサイズをクエリ単位で上書き（Issue #644）。PQ → SQ → f32 の3段 rerank チェーン（Issue #673）も、この値がゲートとなります。`rerank_storage` を有効にした PQ フィールドで `ef_search` を `top_k * rerank_factor` より広く設定すると、グラフが計算した候補集合全体を安価な int8 段で再ランキングしてから exact 段の狭い予算を切り出すようになります（`rerank_storage` サイドカーから導出、追加設定不要）。HNSW 以外のフィールドでは無視されます。 |

### SearchResult

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `id` | `string` | 外部ドキュメント ID |
| `score` | `float` | 関連度スコア |
| `document` | `Document` | ドキュメントの内容 |

### 例

```json
{
  "query": "body:rust",
  "query_vectors": [
    {"vector": [0.1, 0.2, 0.3], "weight": 1.0}
  ],
  "limit": 10,
  "fusion": {
    "rrf": {"k": 60}
  },
  "field_boosts": {
    "title": 2.0
  }
}
```

---

## エラーハンドリング

gRPC エラーは標準の `Status` コードとして返されます。

| Laurus エラー | gRPC ステータス | 発生条件 |
| :--- | :--- | :--- |
| Schema / Query / Field / JSON | `INVALID_ARGUMENT` | 不正なリクエストまたはスキーマ |
| インデックス未オープン | `FAILED_PRECONDITION` | `CreateIndex` の前に RPC が呼び出された場合 |
| インデックスが既に存在 | `ALREADY_EXISTS` | `CreateIndex` が 2 回呼び出された場合 |
| 未実装 | `UNIMPLEMENTED` | まだサポートされていない機能 |
| 内部エラー | `INTERNAL` | I/O、ストレージ、または予期しないエラー |
