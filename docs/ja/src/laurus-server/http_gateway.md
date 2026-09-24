# HTTP ゲートウェイ

HTTP ゲートウェイは Laurus 検索エンジンへの RESTful HTTP/JSON インターフェースを提供します。gRPC サーバーと並行して動作し、リクエストを内部的にプロキシします。

```text
Client (HTTP/JSON) --> HTTP Gateway (axum) --> gRPC Server (tonic) --> Engine
```

## HTTP ゲートウェイの有効化

`http_port` を設定するとゲートウェイが起動します。

```bash
# CLI 引数で指定
laurus serve --http-port 8080

# 環境変数で指定
LAURUS_HTTP_PORT=8080 laurus serve

# 設定ファイルで指定
laurus serve --config config.toml
# （[server] セクションで http_port を設定）
```

`http_port` が未設定の場合、gRPC サーバーのみが起動します。

## エンドポイント

| メソッド | パス | gRPC メソッド | 説明 |
| :--- | :--- | :--- | :--- |
| GET | `/v1/health` | `HealthService/Check` | ヘルスチェック |
| POST | `/v1/index` | `IndexService/CreateIndex` | 新しいインデックスを作成 |
| GET | `/v1/index` | `IndexService/GetIndex` | インデックスの統計情報を取得 |
| GET | `/v1/schema` | `IndexService/GetSchema` | インデックスのスキーマを取得 |
| POST | `/v1/schema/fields` | `IndexService/AddField` | フィールドを動的に追加 |
| DELETE | `/v1/schema/fields/{name}` | `IndexService/DeleteField` | スキーマからフィールドを削除 |
| PUT | `/v1/documents/{id}` | `DocumentService/PutDocument` | ドキュメントの Upsert |
| POST | `/v1/documents/{id}` | `DocumentService/AddDocument` | ドキュメントの追加（チャンク） |
| GET | `/v1/documents/{id}` | `DocumentService/GetDocuments` | ID でドキュメントを取得 |
| DELETE | `/v1/documents/{id}` | `DocumentService/DeleteDocuments` | ID でドキュメントを削除 |
| POST | `/v1/documents:bulk` | `DocumentService/PutDocuments` / `AddDocuments` | ドキュメントのバルクインジェスト（`?mode=put\|add`、既定は `put`） |
| POST | `/v1/commit` | `DocumentService/Commit` | 保留中の変更をコミット |
| POST | `/v1/flush_wal` | `DocumentService/FlushWal` | full commit なしでバッファされた WAL レコードを durable 化 |
| POST | `/v1/search` | `SearchService/Search` | 検索（単発） |
| POST | `/v1/search/stream` | `SearchService/SearchStream` | 検索（Server-Sent Events） |

## API の使用例

### ヘルスチェック

```bash
curl http://localhost:8080/v1/health
```

### インデックスの作成

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "schema": {
      "dynamic_field_policy": "dynamic",
      "fields": {
        "title": {"text": {"indexed": true, "stored": true, "term_vectors": true}},
        "body": {"text": {"indexed": true, "stored": true, "term_vectors": true}}
      },
      "default_fields": ["title", "body"]
    }
  }'
```

`dynamic_field_policy` は省略可能なキーで、スキーマに宣言されていないフィールドの扱いを制御します。指定できる値は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"` の 3 種類です。詳細および `"dynamic"` での情報損失に関する警告は [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

### インデックス統計情報の取得

```bash
curl http://localhost:8080/v1/index
```

### スキーマの取得

```bash
curl http://localhost:8080/v1/schema
```

レスポンスの `text` / `integer` / `float` / `boolean` / `date_time` / `geo` / `geo3d` オプションには常に `multi_valued` が含まれます（例: `"location": {"geo": {"indexed": true, "stored": true, "multi_valued": true, "doc_values": true}}`、`"seen_at": {"date_time": {"indexed": true, "stored": true, "multi_valued": true, "doc_values": true}}`、`"flags": {"boolean": {"indexed": true, "stored": true, "multi_valued": true, "doc_values": true}}`、`"notes": {"text": {"indexed": true, "stored": true, "multi_valued": true, "position_increment_gap": 100}}`）。同じキーは `POST /v1/index` および `POST /v1/schema/fields` でも受け付けます。`text` オプションの `position_increment_gap`（Issue #1175）は入力では省略でき（省略時は `0` ではなくエンジンのデフォルト `100` を意味します）、レスポンスには常に含まれます。

### フィールドの動的追加

稼働中のインデックスにフィールドを追加します。リクエストボディは `POST /v1/index` と同じ `FieldOption` JSON 形式を使います:

```bash
curl -X POST http://localhost:8080/v1/schema/fields \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "category",
    "field_option": {"text": {"indexed": true, "stored": true}}
  }'
```

レスポンスでは更新後のスキーマが返されます。

### フィールドの削除

スキーマからフィールドを削除します。フィールド名はパスで指定します:

```bash
curl -X DELETE http://localhost:8080/v1/schema/fields/category
```

既にインデックスされたデータはストレージに残りますが、アクセスできなくなります。フィールド固有のアナライザとエンベッダーは解除されます。

### ドキュメントの Upsert（PUT）

ドキュメントが既に存在する場合は置換します。

```bash
curl -X PUT http://localhost:8080/v1/documents/doc1 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Hello World",
      "body": "This is a test document."
    }
  }'
```

### ドキュメントの追加（POST）

同じ ID の既存ドキュメントを置換せずに新しいチャンクを追加します。

```bash
curl -X POST http://localhost:8080/v1/documents/doc1 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Hello World",
      "body": "This is a test document."
    }
  }'
```

### ドキュメントのバルクインジェスト（POST）

1 回の呼び出しで多数のドキュメントを適用します — エントリは入力順に逐次処理され、バッチ全体で WAL fsync は 1 回です。`?mode=put`（既定）は Upsert（重複 ID はデデュープ、最後が勝ち）、`?mode=add` はチャンク追加（繰り返した ID は蓄積）です。

```bash
curl -X POST 'http://localhost:8080/v1/documents:bulk?mode=put' \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {"id": "doc1", "fields": {"title": "Hello"}},
      {"id": "doc2", "fields": {"title": "World"}}
    ]
  }'
# => {"applied": 2}
```

適用できない最初のエントリで fail-fast します。適用済みエントリはロールバックされず（次のコミットで永続化）、エラーには失敗位置が含まれるため、バッチまたはその suffix の再試行は冪等です。

### ドキュメントの取得

```bash
curl http://localhost:8080/v1/documents/doc1
```

### ドキュメントの削除

```bash
curl -X DELETE http://localhost:8080/v1/documents/doc1
```

### コミット

```bash
curl -X POST http://localhost:8080/v1/commit
```

### WAL のフラッシュ

バッファされた WAL レコードを full commit なしで durable 化します。成功時は `{}` を返します。デフォルトの per-record sync ポリシーでは near no-op です。グループコミットポリシーでは、現在の partial batch をオンデマンドで flush します。バッファされた変更は後続の `POST /v1/commit` まで検索に反映されません。

```bash
curl -X POST http://localhost:8080/v1/flush_wal
```

### 検索

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:test", "limit": 10}'
```

#### フィールドブースト付き検索

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "rust programming",
    "limit": 10,
    "field_boosts": {"title": 2.0}
  }'
```

#### ハイブリッド検索

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "body:rust",
    "query_vectors": [{"vector": [0.1, 0.2, 0.3], "weight": 1.0}],
    "limit": 10,
    "fusion": {"rrf": {"k": 60}}
  }'
```

#### ハイライト付き検索

`highlight` を指定すると、フィールドごとのハイライト済みフラグメントを要求できます（Issue #1134）。省略形はフィールド名の配列だけです。

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:rust", "limit": 10, "highlight": ["body"]}'
```

オブジェクト形式では `HighlightConfig` の各設定（`max_fragments`、`fragment_size`、`tag`、`css_class`、`require_field_match`、`max_analyzed_chars`、`return_entire_field_if_no_highlight`）を追加できます。

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "body:rust",
    "limit": 10,
    "highlight": {"fields": ["body"], "max_fragments": 2, "tag": "em"}
  }'
```

各結果には、少なくとも1つのフィールドが実際にハイライトされた場合にのみ `"highlights"` オブジェクトが追加されます。

```json
{"id": "doc1", "score": 1.2, "fields": {...}, "highlights": {"body": ["<em>Rust</em> is a systems programming language"]}}
```

`highlight` はスキーマ上で `stored: true` のテキストフィールドにのみ作用し、ハイライトは常にリクエストの lexical クエリに従います。`filter_query` はハイライト対象の語を提供せず、Vector-only のリクエストは `highlights` を一切生成しません。詳細な意味論は[ハイライト](../laurus/highlighting.md)を参照してください。

### ストリーミング検索（SSE）

`/v1/search/stream` エンドポイントは Server-Sent Events（SSE）として結果を返します。各結果は個別のイベントとして送信されます。

```bash
curl -N -X POST http://localhost:8080/v1/search/stream \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:test", "limit": 10}'
```

レスポンスは SSE イベントのストリームです。

```text
data: {"id":"doc1","score":0.8532,"fields":{...}}

data: {"id":"doc2","score":0.4210,"fields":{...}}
```

## JSON フィールド値の型推論

ドキュメント投入リクエスト（`PUT /v1/documents/{id}` または
`POST /v1/documents/{id}`）のボディに含まれる `fields` の各値は、
laurus-cli や laurus-mcp と共通の正典コンバータ `json_to_document` により、
スキーマレス取り込みと同じ推論ルールでエンジンの
[`DataValue`](../concepts/schema_and_fields.md) に変換されます。これにより
HTTP・gRPC・CLI・MCP のすべての経路で挙動が一致します。

| JSON 値 | 推論されるフィールド型 | 備考 |
| :--- | :--- | :--- |
| `null` | (スキップ) | フィールドごと省略されます（明示的な null としてもエンジンには送られません）。 |
| `true` / `false` | `boolean` | |
| 整数（`i64` に収まる） | `integer` | |
| 浮動小数点 / 巨大整数 | `float` | |
| `"text"` | `text` | |
| `[1, 2, 3]`（全要素 integer） | `integer`（`multi_valued: true`） | 多値数値フィールド。 |
| `[1.0, 2.5]`（非整数を含む数値配列） | `float`（`multi_valued: true`） | |
| `[]`（空配列） | (スキップ) | 要素型を決定できないためフィールドはスキップされます。 |
| `{"latitude": ..., "longitude": ...}` | `geo` | |
| `{"lat": ..., "lon": ...}` / `{"lat": ..., "lng": ...}` | `geo` | latitude / longitude の短縮別名を受け付けます。 |
| `{"x": ..., "y": ..., "z": ...}` | `geo3d` | 3 キーすべて必須、有限な数値、ECEF メートル単位。`lat`/`lon` キーとの混在は拒否されます。 |
| `[{"latitude": 35.6, "longitude": 139.7}, ...]`（全要素が地理 object） | `geo`（`multi_valued: true`） | 多値地理フィールド。`lat` / `lon` / `lng` の別名も受け付けます。ドキュメント取得時は `{"latitude", "longitude"}` object の配列として返されます。 |
| `[{"x": ..., "y": ..., "z": ...}, ...]`（全要素が 3D object） | `geo3d`（`multi_valued: true`） | 多値 3D 地理フィールド。取得時は `{"x", "y", "z"}` object の配列として返されます。1 つの配列に 2D と 3D の object を混在させると拒否されます。 |
| `["2024-01-01T00:00:00Z", "2024-06-15T21:00:00+09:00"]`（全要素が RFC 3339 文字列） | `date_time`（`multi_valued: true`） | 多値日時フィールド（Issue #1184）。ここで日時として認識されるのは RFC 3339 文字列のみです。ドキュメント取得時は UTC に正規化した RFC 3339 文字列の配列（例: `"2024-06-15T12:00:00+00:00"`）として返されます。 |
| `["a", "b"]`（全要素が文字列だが、全要素が RFC 3339 ではない） | `text`（`multi_valued: true`） | 多値テキストフィールド（Issue #1175）。各要素は独立に解析されます。term クエリはいずれかの要素がタームを含めばマッチし、フレーズクエリは slop がフィールドの `position_increment_gap`（デフォルト 100）に達しない限り 2 つの要素をまたぎません。ドキュメント取得時は文字列の配列として返されます。 |
| `[true, false]`（全要素がブール値） | `boolean`（`multi_valued: true`） | 多値ブールフィールド（Issue #1180）。`flags:true` のような term クエリはいずれかの要素が値と等しければマッチします。ドキュメント取得時はブール値の配列として返されます。 |
| `{"data": "<base64>", "mime": "..."}` | `bytes` | `mime` は省略可能。マルチモーダルなベクトルフィールド（`Text` と `Bytes` の両方を受け付ける embedder）に対して、素の文字列と画像バイト列を区別するために使います — 詳細は [スキーマとフィールド](../concepts/schema_and_fields.md) を参照してください。 |

以下の場合、ゲートウェイは HTTP 400（`Bad Request`）を返します:

- 配列が混在型もしくは非数値要素を含む（例: `[1, "x"]` や `[true, 1]`）、または 2D と 3D の地理 object を混在させている（例: `[{"lat": ...}, {"x": ...}]`）
- オブジェクトが上記のいずれの形にも一致しない（2D は latitude/longitude
  キーが、3D は `x`/`y`/`z` のいずれかが欠けている、または `data` が
  文字列でない、など）
- 緯度が `[-90, 90]` の範囲外、または経度が `[-180, 180]` の範囲外
- 3D ECEF の座標が有限値でない（`NaN` / `Inf`）
- 同一オブジェクトに複数の形（2D の `lat`/`lon`、3D の `x`/`y`/`z`、
  Bytes の `data`）のキーが混在

ベクトルフィールドは JSON だけからは推論できません（次元数・距離関数・
embedder の設定を値だけから復元できないため）。スキーマで明示的に
宣言する必要があります。宣言済みのベクトルフィールドに数値配列が送られた
場合は自動的に `f32` ベクトルへキャストされるので、REST クライアントは
埋め込みベクトルを通常の JSON 配列として送信できます。バイト列フィールドは
上表の `{"data", "mime"}` オブジェクト形式、または宣言済み `Bytes`
フィールドであれば素の base64 文字列でも投入できます。

### 3D 地理クエリ

3D ECEF クエリは `query` に渡す Lexical DSL 文字列をそのまま再利用します。ゲートウェイは文字列を変更せずエンジンへ転送するため、gRPC 経由と同じ DSL 形式が HTTP 経由でも動作します:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "position:geo3d_distance(-3955182, 3350553, 3700276, 5000)",
    "limit": 10
  }'
```

`geo3d_bbox` および `geo3d_nearest` の構文は [Query DSL → 3D 地理クエリ](../concepts/query_dsl.md#3d-geographic-queries-geo3d_) を参照してください。

## リクエスト/レスポンス形式

すべてのリクエストおよびレスポンスボディは JSON を使用します。JSON の構造は gRPC の protobuf メッセージに対応しています。メッセージ定義の詳細は [gRPC API リファレンス](grpc_api.md)を参照してください。
