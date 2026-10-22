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
| PATCH | `/v1/schema/fields/{name}` | `IndexService/UpdateField` | フィールドの型やオプションを変更 |
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

`dynamic_field_policy` は省略可能なキーで、スキーマに宣言されていないフィールドの扱いを制御します。指定できる値は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"` の 3 種類です。`null` はキーの省略と同じで、ほかの値（文字列でない値を含む）は 400 で拒否されます（Issue #1473）。詳細および `"dynamic"` での情報損失に関する警告は [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

`schema` はオブジェクト、`default_fields` は文字列の配列、`analyzers` と `embedders` はオブジェクトでなければなりません。アナライザーの `char_filters` と `token_filters` は配列です。これらのキーの `null` は省略と同じです。`"default_fields": "title"` や `"default_fields": [1, "title"]` などそれ以外の値は 400 で拒否され、インデックスは作られません（Issue #1524）。この修正より古いサーバーは、そうした値を省略として読むか、文字列でない要素を捨てていました。

### インデックス統計情報の取得

```bash
curl http://localhost:8080/v1/index
```

### スキーマの取得

```bash
curl http://localhost:8080/v1/schema
```

レスポンスの `text` / `integer` / `float` / `boolean` / `date_time` / `geo` / `geo3d` / `bytes` オプションには常に `multi_valued` が含まれます（例: `"location": {"geo": {"indexed": true, "stored": true, "multi_valued": true, "doc_values": true}}`、`"seen_at": {"date_time": {"indexed": true, "stored": true, "multi_valued": true, "doc_values": true}}`、`"flags": {"boolean": {"indexed": true, "stored": true, "multi_valued": true, "doc_values": true}}`、`"notes": {"text": {"indexed": true, "stored": true, "multi_valued": true, "position_increment_gap": 100}}`、または `"thumbnail": {"bytes": {"stored": true, "multi_valued": true}}`）。同じキーは `POST /v1/index` および `POST /v1/schema/fields` でも受け付けます。これらのオプションの真偽値のキー（`indexed`・`stored`・`multi_valued`・`doc_values`・`term_vectors`）は `true` か `false` で、`null` はキーの省略と同じです。省略すると、`indexed`・`stored`・`multi_valued` は `false`、`doc_values` と `term_vectors` はエンジンのデフォルトになります。`"true"`・`1` などそれ以外の値は、省略として読まれるのではなく 400 で拒否され、何も書き込まれません（Issue #1511）。`text` オプションの `position_increment_gap`（Issue #1175）は入力では省略でき（省略時は `0` ではなくエンジンのデフォルト `100` を意味します）、レスポンスには常に含まれます。値は `0` から `4294967295` までの整数で、`null` は省略と同じです。`"0"`・`-1`・`4294967296` などそれ以外の値は、省略として読まれるのではなく 400 で拒否されます（Issue #1490）。`bytes` オプションには `indexed` も `doc_values` もありません —— `Bytes` の値は `multi_valued` にかかわらずインデックスされず、DocValues にも書き込まれません（Issue #1176）。

フィールドオプションの型のキー（`"text"` や `"hnsw"` など）はちょうど 1 つで、その値はオブジェクトです。ほかの型のキーと並んだ `null` の型のキーは、省略と同じです。`{"text": 1}`・`{"text": null}`・`{"text": {}, "integer": {}}` は、`POST /v1/index`・`POST /v1/schema/fields`・`PATCH /v1/schema/fields/{name}` で 400 で拒否され、何も書き込まれません（Issue #1524）。この修正より古いサーバーは、`{"text": 1}` をインデックスも保存もしない `text` フィールドとして読み、型のキーが 2 つあれば先のものを使っていました。`text` オプションの `analyzer` は名前か、`{"language": "japanese", "dict": "...", "mode": "normal"}` のようなオブジェクトで、その `mode` と `user_dict` は文字列です。`analyzer`・`mode`・`user_dict` の `null` は省略と同じです。この修正より古いサーバーは、`"analyzer": null` を 400 で拒否していました。

MultiVector フィールド（Issue #1177）は `"body_colbert": {"multi_vector": {"dimension": 128, "distance": "cosine"}}` のように宣言します（`distance` は `"cosine"` または `"dot_product"`、省略時は `"cosine"`）。文書での値は、トークンベクトルごとの同じ長さの数値配列の配列です（`"body_colbert": [[0.1, 0.2, ...], [0.3, 0.4, ...]]`）。このフィールドは late interaction の再採点のためにトークンベクトルを保持するもので、ベクトル検索の対象にはならず、文書とともに返されることもありません。

MultiVector フィールドには、トークン単位のエンベッダーも指定できます（Issue #1349）。エンベッダーはスキーマの `embedders` で宣言し、数値のオプションは JSON の数値で書きます。例: `"embedders": {"colbert": {"type": "candle_colbert", "model": "answerdotai/answerai-colbert-small-v1", "revision": "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e", "doc_maxlen": 300}}` と `"body_colbert": {"multi_vector": {"dimension": 96, "embedder": "colbert"}}`。こうすると、文書はこのフィールドにテキスト（`"body_colbert": "how lifetimes work in rust"`）を与えられ、テキストはトークンベクトルに埋め込まれます。エンベッダーのパラメータのうち数値と真偽値は文字列として渡され、配列とオブジェクトは拒否されます。

MultiVector フィールドには `storage`（Issue #1346）も指定でき、トークンベクトルのディスク上の要素形式を選べます。`"f32"`（デフォルト、誤差なし）、`"f16"`（`"f32"` の半分のサイズ、要素あたり相対誤差 ~2⁻¹¹）、`"int8"`（典型的な次元数で `"f32"` の約 1/4 のサイズ、ベクトルごとのスケールを使用）、`"binary"`（`"f32"` の 1/32 のサイズ、要素あたり 1 bit。`"distance": "cosine"` が必要）のいずれかです。整数の `pool_factor` も指定でき、各文書のトークンベクトルを約 `1 / pool_factor` に減らします（デフォルトの `1` はすべて保存。`0` はエンジンが拒否し、0 以上の整数でない値は 400 になります）。`1` のとき、このキーはレスポンスから省略されます。値は小文字の文字列で、入力時は大文字小文字を区別せずに受け付けます。例: `"body_colbert": {"multi_vector": {"dimension": 128, "storage": "int8"}}`。認識できない値と文字列でない値は 400 で拒否され（Issue #1473）、`null` はキーの省略と同じです。`storage` がデフォルトの `"f32"` の場合、このキーはレスポンスから省略されます。

`hnsw`・`flat`・`ivf` フィールドには、`distance` として `"cosine"`・`"euclidean"`・`"manhattan"`・`"dot_product"`・`"angular"` のいずれか（MultiVector フィールドも同じ文字列を受け付けます）を、`rerank_storage` として `"F32"` を指定できます。`"F32"` を指定すると、`rerank_factor` を指定したクエリが再スコアに使う完全精度のベクトルの写しを保存します（[Two-stage rerank](../concepts/indexing/vector_indexing.md#two-stage-rerankissue-481-stage-2) を参照）。どちらも入力時は大文字小文字を区別せずに受け付けます。`distance` を省略するか `null` にすると `"cosine"` になり、`rerank_storage` を省略するか `null` にすると写しを保存しません。レスポンスの `rerank_storage` は、`"F32"` のときだけ含まれます。ほかの値（文字列でない値を含む）は 400 で拒否され（Issue #1473）、何も書き込まれません。

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{"schema": {"fields": {"embedding": {"hnsw": {"dimension": 384, "distance": "euclidian"}}}}}'
# => {"error":{"code":400,"message":"field \"embedding\": unknown distance \"euclidian\" (expected \"cosine\", \"euclidean\", \"manhattan\", \"dot_product\", or \"angular\")"}}
```

`hnsw`・`flat`・`ivf` フィールドには `quantizer` を指定でき、ベクトルの保存形式を選べます。例: `"embedding": {"hnsw": {"dimension": 384, "quantizer": {"method": "product_quantization", "subvector_count": 48}}}`。`method` は gRPC の `QuantizationMethod` の名前から接頭辞 `QUANTIZATION_METHOD_` を除いて小文字にしたもので、入力時は大文字小文字を区別せずに受け付けます。

| `method` | インデックスの種類 | 備考 |
| :--- | :--- | :--- |
| `"scalar_8bit"` | HNSW, Flat, IVF | `quantizer` を省略したとき、または `null` のときのデフォルト |
| `"float16"` | HNSW, Flat, IVF | IEEE 754 半精度（Issue #623） |
| `"product_quantization"` | HNSW | `subvector_count` は `1` 以上で `dimension` を割り切る値にする |
| `"product_quantization_fastscan"` | HNSW | 実験的。サーバーを `pq-fastscan` feature 付きでビルドする必要がある。`subvector_count` は `1` 以上で `dimension` を割り切る値にする |

`subvector_count` を使うのは、2 つの直積量子化（product quantization）の方式のときだけです。ただし、どの方式でも値は `0` から `4294967295` までの整数で、`null` は省略と同じです。`"48"`・`48.0`・`-1` などの値は `0` として読まれるのではなく、`4294967295` を超える値は折り返されるのではなく、400 で拒否されます（Issue #1490）。ほかの `method`（`"none"` を含む。量子化なしの保存は Issue #481 で廃止）、文字列の `method` を持たない quantizer、`pq-fastscan` なしでビルドしたサーバーへの `"product_quantization_fastscan"` は 400 で拒否され（Issue #1459）、何も書き込まれません。

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{"schema": {"fields": {"embedding": {"hnsw": {"dimension": 384, "quantizer": {"method": "pq", "subvector_count": 48}}}}}}'
# => {"error":{"code":400,"message":"field \"embedding\": unknown quantizer method \"pq\" (expected \"scalar_8bit\", \"product_quantization\", \"product_quantization_fastscan\", or \"float16\")"}}
```

2 つの直積量子化の方式では、`POST /v1/index`・`POST /v1/schema/fields`・`PATCH /v1/schema/fields/{name}` は、`0` の `subvector_count`（`subvector_count` を省略したときの値）と `dimension` を割り切らない値を 400 で拒否し（Issue #1491）、何も書き込みません。

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{"schema": {"fields": {"embedding": {"hnsw": {"dimension": 384, "quantizer": {"method": "product_quantization", "subvector_count": 5}}}}}}'
# => {"error":{"code":400,"message":"Invalid argument: field 'embedding': HNSW product quantization subvector_count 5 must divide dimension 384 (got 384 % 5 = 4)"}}
```

この検査より古いサーバーはそうしたフィールドを受け付け、後の commit が 500 で失敗します。そのサーバーが作ったインデックスも開けますが、サーバーのログに警告が出ます。フィールドの直し方は [gRPC API → 直積量子化の `subvector_count`](grpc_api.md) を参照してください。

レスポンスには常に `quantizer` が含まれます。`subvector_count` を使わない方式では、その値は `0` です（例: `"quantizer": {"method": "scalar_8bit", "subvector_count": 0}`）。

`hnsw` フィールドには `m` と `ef_construction`、`ivf` フィールドには `n_clusters` と `n_probe` を整数で指定できます（Issue #1479）。省略するか `null` にするとエンジンのデフォルト（`m` は `16`、`ef_construction` は `200`、`n_clusters` は `100`、`n_probe` は `1`）が使われ、TOML のスキーマで省略したときと同じになります。レスポンスには使われた値が出ます。`dimension` にはデフォルトがなく、どの vector フィールドでも指定が必要です。`0` から `4294967295` までの整数でない値（`16.0` や `"16"` など）は 400 で拒否されます。`2` 未満の `m`、`0` の `ef_construction`・`n_clusters`・`n_probe`、`0` の `dimension` も同じく 400 で拒否され、何も書き込まれません。

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{"schema": {"fields": {"embedding": {"hnsw": {"dimension": 384, "m": 1}}}}}'
# => {"error":{"code":400,"message":"Invalid argument: field 'embedding': HNSW m must be at least 2, got 1"}}
```

`PATCH /v1/schema/fields/{name}`（`UpdateField`）はオプション全体を置き換えるので、そこで省略したキーも、フィールドの今の値ではなくデフォルトになります。たとえば `"m": 32` の `hnsw` フィールドに `{"field_option": {"hnsw": {"dimension": 2}}, "reindex": true}` を PATCH すると、`m` は `16` で作り直されます。この変更より古いサーバーは、省略した値を `0` として読みます。`m` が `0` だとすべてのドキュメントの書き込みが失敗し、`n_probe` が `0` だとすべての検索が 0 件になるので、古いサーバーを相手にするときはこれらを明示してください。そうしたサーバーが作ったインデックスも開けますが、サーバーのログに警告が出ます。フィールドの直し方は [gRPC API のグラフとクラスタのパラメータ](grpc_api.md)を参照してください。

`hnsw` フィールドには `default_ef_search` も指定できます。検索が `vector_params.ef_search` を指定しないときに、そのフィールドの検索で使う候補リストの大きさです（Issue #644。[スキーマとフィールドの `default_ef_search`](../concepts/schema_and_fields.md#default_ef_search-検索時の-recall-調整パラメータ) を参照）。`m` と同じく `0` から `4294967295` までの整数で、`"64"`・`64.0`・`-1` などそれ以外の値は 400 で拒否され、何も書き込まれません（Issue #1490）。省略するか `null` にすると未設定のままで、レスポンスには設定したときだけ出ます。この修正より古いサーバーは、そうした値を未設定として読み、`4294967295` を超える値を折り返し（`4294967360` は `64` になる）、レスポンスのスキーマに `default_ef_search` を出しませんでした。`PATCH /v1/schema/fields/{name}` はオプション全体を置き換えるので、そうしたレスポンスからコピーしたフィールドで PATCH すると、コピーに設定し直さない限り `default_ef_search` は消えます。

ベクトルフィールドの `base_weight` は数値、`embedder`（と `hnsw` フィールドの `pq_codebook_path`）は文字列です。`null` は省略と同じです。`"base_weight": "2"` などそれ以外の値は 400 で拒否され、何も書き込まれません（Issue #1524）。この修正より古いサーバーはそうした値を省略として読むので、`"base_weight": "2"` はエンジンのデフォルトの `1.0` になっていました。

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

### フィールドの変更

既存のフィールドの型やオプションを変更します。フィールド名はパスで指定し、本文には新しいオプションと 2 つのフラグを書きます:

| キー | 必須 | 説明 |
| :--- | :--- | :--- |
| `field_option` | はい | フィールドの新しい設定。`POST /v1/schema/fields` と同じ `FieldOption` の JSON 形式。オプション全体を置き換えるので、省略したキーはフィールドの今のオプションから引き継がれず、デフォルトになります（[スキーマの取得](#スキーマの取得)を参照）。`dimension` にはデフォルトがないので、改めて指定する必要があります |
| `reindex` | いいえ | `true` にすると `reindex` か `destructive` の変更を適用します（デフォルト `false`） |
| `dry_run` | いいえ | `true` にすると、何も適用せずに変更を分類します（デフォルト `false`） |

`reindex` と `dry_run` は `true` か `false` で、`null` は省略と同じです。文字列の `"true"` などそれ以外の値は 400 で拒否され、変更の分類も適用も行われません（Issue #1511）。この修正より古いサーバーはそうした値を `false` として読むため、`"dry_run": "true"` は変更を適用してしまいます。

レスポンスには、`classification`（`metadata_only`・`reindex`・`destructive` のいずれか。[フィールドの変更](../concepts/schema_and_fields.md#フィールドの変更)を参照）と、変更後のスキーマが入ります。dry run ではスキーマは変わりません。`destructive` の変更では、フィールドがスキーマの `pending_reindex` にも載ります。`embedding` だけを `{"hnsw": {"dimension": 384}}` で作ったインデックスの場合:

```bash
# m を大きくする変更の分類を、適用せずに確かめる
curl -X PATCH http://localhost:8080/v1/schema/fields/embedding \
  -H 'Content-Type: application/json' \
  -d '{"field_option": {"hnsw": {"dimension": 384, "m": 32}}, "dry_run": true}'
# => {"classification":"reindex","schema":{"default_fields":[],"dynamic_field_policy":"dynamic","fields":{"embedding":{"hnsw":{"base_weight":1.0,"dimension":384,"distance":"cosine","ef_construction":200,"m":16,"quantizer":{"method":"scalar_8bit","subvector_count":0}}}}}}

# 真偽値でないフラグは拒否され、何も適用されない
curl -X PATCH http://localhost:8080/v1/schema/fields/embedding \
  -H 'Content-Type: application/json' \
  -d '{"field_option": {"hnsw": {"dimension": 384, "m": 32}}, "reindex": true, "dry_run": "true"}'
# => {"error":{"code":400,"message":"dry_run must be true or false"}}

# reindex の変更は "reindex": true なしでは拒否される
curl -X PATCH http://localhost:8080/v1/schema/fields/embedding \
  -H 'Content-Type: application/json' \
  -d '{"field_option": {"hnsw": {"dimension": 384, "m": 32}}}'
# => {"error":{"code":400,"message":"Invalid argument: updating field 'embedding' requires rebuilding or discarding existing data (classified as Reindex); pass `UpdateFieldOptions { reindex: true, .. }` to proceed"}}

# 適用する
curl -X PATCH http://localhost:8080/v1/schema/fields/embedding \
  -H 'Content-Type: application/json' \
  -d '{"field_option": {"hnsw": {"dimension": 384, "m": 32}}, "reindex": true}'
# => {"classification":"reindex","schema":{"default_fields":[],"dynamic_field_policy":"dynamic","fields":{"embedding":{"hnsw":{"base_weight":1.0,"dimension":384,"distance":"cosine","ef_construction":200,"m":32,"quantizer":{"method":"scalar_8bit","subvector_count":0}}}}}}
```

エラーメッセージが挙げる `UpdateFieldOptions` は、`reindex` が設定するライブラリのオプションです。

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

write-ahead log への追記以降に失敗した書き込み（I/O エラー、ディスク容量不足）は、書き込みを止めます。以後の書き込み・コミット・フィールド変更は、サーバーを再起動するまで `412 Precondition Failed` を返します。再起動するとログが再生され、レコードがログに完全に届いていればその書き込みが丸ごと適用され、届いていなければ適用されません。一部だけが適用されることはありません（[失敗した書き込み](../laurus/persistence.md#失敗した書き込み)を参照）。リクエストがキャンセルされても、クライアントが切断しても、書き込みは止まりません。各ドキュメントは、すべて適用されているか、まったく書かれていないかのどちらかです。キャンセルまでに適用されたドキュメントは、適用されたまま残ります。

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

`limit` と `offset` は、省略または `null` のとき既定値（10 と 0）になります。Issue #1367 以降、0 以上の整数でない値や、32 ビット符号なし整数に収まらない値は `400 Bad Request` で拒否されます（以前は既定値に置き換わるか、折り返していました）。`offset + limit` がサーバの [`max_result_window`](configuration.md)（既定 10,000）を超える検索も `400 Bad Request` で拒否されます。

`lexical_params` には、gRPC の [`LexicalParams`](grpc_api.md#lexicalparams) メッセージのキーを snake_case で指定します。`timeout_ms` は `0` から `18446744073709551615` までの整数で、省略・`null`・`0` はタイムアウトなしを意味します。`"500"`・`500.0`・`-1` などそれ以外の値は、タイムアウトなしで検索するのではなく `400 Bad Request` で拒否されます（Issue #1490）。`parallel` は `true` か `false` で、省略と `null` は `false` を意味します。`"true"`・`1` などそれ以外の値は、`false` として読まれるのではなく `400 Bad Request` で拒否されます（Issue #1511）。`min_score` は数値で、省略と `null` は `0.0`（しきい値なし）を意味します。

本体は JSON のオブジェクト、`query` は文字列でなければなりません。`query` の省略と `null` はクエリなしを意味します。それ以外の値は `400 Bad Request` で拒否されます（Issue #1524）。この修正より古いサーバーは、文字列でない `query` をクエリなしとして読むので、`query_vectors` 付きの `"query": 5` はベクトルだけの検索になっていました。

#### フィールドの値で並べる検索

`lexical_params.sort_by` を指定すると、ヒットをスコアではなくフィールドの値で並べます。

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "body:rust",
    "limit": 10,
    "lexical_params": {"sort_by": {"field": "year", "order": "desc"}}
  }'
```

`field` は文字列でなければならず、空文字列はスコア順を意味します。`order` は `"asc"` か `"desc"` で、大文字小文字を区別せずに受け付けます。省略と `null` は `"asc"` を意味します。gRPC の `"SORT_ORDER_DESC"` などの名前ではありません。オブジェクトでない `sort_by`、文字列の `field` を持たない `sort_by`、ほかの `order` は `400 Bad Request` で拒否されます（Issue #1476、#1524）。この修正より古いサーバーは、`"DESC"` などほかの `order` を昇順として読み、文字列の `field` を持たない `sort_by` ではスコア順に並べていました。`sort_by` が効くのは `query` のあるリクエストだけです。

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

各キーは、スキーマの lexical フィールド（または `_id`）でなければなりません。打ち間違いなどの未知のフィールドやベクトルフィールドは、黙って無視されずに `400 Bad Request` で拒否されます（Issue #1374）。`field_boosts` はオブジェクトで、各ブーストは数値です。`null` のブーストはそのフィールドを書かないのと同じです。`{"title": "2"}` などそれ以外の値は、黙って捨てられるのではなく `400 Bad Request` で拒否されます（Issue #1524）。

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

`query_vectors` は `query` のベクトル部分に追加されるので、lexical 句だけの `query` は、`lexical_params` / `field_boosts` の有無にかかわらずハイブリッド検索になります（Issue #1366）。各要素には空でない数値の配列 `vector` が必要です。`weight`（数値）と `fields`（文字列の配列）は省略でき、`null` なら既定値のままです。それ以外の形は、黙って捨てずに `400 Bad Request` で拒否されます。

`fusion` には `rrf`（`{"k": 60}`）と `weighted_sum`（`{"lexical_weight": 0.5, "vector_weight": 0.5}`）のどちらか一方だけを指定します。どちらもオブジェクトで、数値を省略するか `null` にすると、ここに示した値になります。`fusion` を省略すると、`k` が 60 の RRF になります。どちらもない、または両方ある `fusion` や、`{"rrf": {"k": "1"}}` のように型の違う値は `400 Bad Request` で拒否されます（Issue #1524）。この修正より古いサーバーは代わりに既定値を使い、両方あれば `rrf` を使っていました。

#### ベクトル検索パラメータ付き検索

`vector_params` には、gRPC の [`VectorParams`](grpc_api.md#vectorparams) メッセージのキーを snake_case で指定します。キーは `fields`、`score_mode`（`"weighted_sum"`、`"max_sim"`、`"late_interaction"` のいずれか）、`overfetch`、`min_score`、`rerank_factor`、`ef_search` です。値が `null` のキーや省略したキーは既定値のままになります。型の違うキー、未知の `score_mode`、32 ビット符号なし整数に収まらない `rerank_factor` / `ef_search` は `400 Bad Request` で拒否されます。gRPC API が拒否する値も同様です（Issue #1342）。

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query_vectors": [{"vector": [0.1, 0.2, 0.3]}],
    "vector_params": {"fields": ["title_vec"], "min_score": 0.5, "ef_search": 128},
    "limit": 10
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

`max_fragments` と `fragment_size` は `0` から `4294967295` までの整数、`max_analyzed_chars` は `0` から `18446744073709551615` までの整数で、`null` はキーの省略と同じく既定値になります。`"2"`・`2.0`・`-1` などそれ以外の値は、既定値に置き換わったり折り返したりするのではなく、`fields` が空のときも含めて `400 Bad Request` で拒否されます（Issue #1490）。gRPC と同じく、`0` の `max_fragments`・`fragment_size` も拒否されます。

`require_field_match` と `return_entire_field_if_no_highlight` は `true` か `false` で、`null` はキーの省略と同じく既定値になります。`"true"`・`1` などそれ以外の値は、既定値に置き換わるのではなく、`fields` が空のときも含めて `400 Bad Request` で拒否されます（Issue #1511）。

省略形の要素はすべて文字列で、オブジェクト形式の `fields` は文字列の配列、`tag` と `css_class` は文字列です。`null` はキーの省略と同じです。配列でもオブジェクトでもない `highlight` や、これらのキーの型の違う値は、黙って捨てられるのではなく `400 Bad Request` で拒否されます（Issue #1524）。空のフィールドのリスト（`[]`・`{}`・`{"fields": []}`）はハイライトを要求しないことを意味します。gRPC API は空の `HighlightParams.fields` を拒否します。

各結果には、少なくとも1つのフィールドが実際にハイライトされた場合にのみ `"highlights"` オブジェクトが追加されます。

```json
{"id": "doc1", "score": 1.2, "fields": {...}, "highlights": {"body": ["<em>Rust</em> is a systems programming language"]}}
```

`highlight` はスキーマ上で `stored: true` のテキストフィールドにのみ作用し、ハイライトは常にリクエストの lexical クエリに従います。`filter_query` はハイライト対象の語を提供せず、Vector-only のリクエストは `highlights` を一切生成しません。詳細な意味論は[ハイライト](../laurus/highlighting.md)を参照してください。

#### 再採点付き検索

`rescore` を指定すると、上位 `window_size` 件（既定 100）を `MultiVector` フィールドに対する late interaction で並べ替えます（Issue #1351）。クエリは、フィールドのトークン単位のエンベッダー（`candle_colbert`）が埋め込むテキストか、クエリのトークンベクトル（同じ長さの数値配列の配列）です。

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "body:lifetimes",
    "limit": 10,
    "rescore": {
      "window_size": 100,
      "late_interaction": {"field": "body_colbert", "text": "how do lifetimes work"}
    }
  }'
```

```json
"rescore": {"late_interaction": {"field": "body_colbert", "vectors": [[0.1, 0.2], [0.3, 0.4]]}}
```

再採点した結果の `score` は late interaction（MaxSim）のスコアです。不正な `rescore`（`field` がない、`vectors` と `text` の両方またはどちらもない、ベクトルの長さがそろわない）は無視されず 400 で拒否されます。エンジンが拒否する値も同じです（[gRPC の `RescoreParams`](grpc_api.md#rescoreparams) と [late interaction による再採点](../concepts/search/vector_search.md#late-interaction-による再採点rescore)を参照）。`vectors` と `text` の `null` は省略と同じです（Issue #1524）。

#### グループ化付き検索

`group_by` を指定すると、結果をフィールド値でグループ化し、`limit` と `offset` はグループの数を数えます（Issue #1297）。`POST /v1/documents/:id` で追加した chunk は `_id` を共有するので、`_id` でグループ化すると、各文書が最上位の chunk を代表として 1 回ずつ返り、その文書の chunk が最大 `group_size - 1` 件添えられます（`group_size` の既定は 1、上限は 100）。

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "body:rust", "limit": 10, "group_by": {"field": "_id", "group_size": 2}}'
```

`manual` として 3 つ、`faq` として 1 つの chunk を追加したとき、レスポンスは次のとおりです。

```json
{"results":[{"fields":{"body":"Rust ownership rules for Rust programs"},"group_key":"manual","id":"manual","inner_hits":[{"fields":{"body":"Borrowing in Rust"},"group_key":"manual","id":"manual","score":0.012359549291431904}],"score":0.012571428902447224},{"fields":{"body":"Why Rust has no garbage collector"},"group_key":"faq","id":"faq","score":0.010891089215874672}]}
```

値を持つ結果には `"group_key"` が、グループにほかの結果がある結果には `"inner_hits"`（各要素は同じ形）が付きます。`rescore` と同じく、不正な `group_by` は無視されず 400 で拒否されます。

```json
{"error":{"code":400,"message":"group_by.field must be a string"}}
```

エンジンが拒否する値も同じです（[gRPC の `GroupByParams`](grpc_api.md#groupbyparams) と[検索結果のグループ化](../concepts/search/result_grouping.md)を参照）。

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

`group_by` を指定すると、各イベントは 1 つのグループになります。グループの最上位の結果で、`group_size` が 2 以上なら `inner_hits` が付きます。`"group_by": {"field": "_id"}` のとき:

```text
data: {"fields":{"body":"Rust ownership rules for Rust programs"},"group_key":"manual","id":"manual","score":0.012571428902447224}

data: {"fields":{"body":"Why Rust has no garbage collector"},"group_key":"faq","id":"faq","score":0.010891089215874672}
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

各キーは、上の各節で説明した JSON の型の値を取ります（多くは 1 つの型で、たとえば `highlight` は配列かオブジェクトを取ります）。省略できるキーでは、`null` は省略と同じです。ほかの型の値は、キーを名指しするメッセージ付きの `400 Bad Request` で拒否され、何も書き込まれず、検索も行われません。真偽値に対する文字列の `"true"` や数値の `1`（Issue #1511）、数値に対する文字列の `"2"`、文字列に対する数値の `1`、オブジェクトに対する文字列や配列（Issue #1524）がこれに当たります。これらの修正より古いサーバーは、そうした値を省略として読むか、捨てていました。アナライザーの文字フィルター・トークナイザー・トークンフィルターのパラメータは、まだこの検査の対象ではありません（Issue #1478）。`POST /v1/search/batch` では、メッセージが失敗したクエリの位置から始まり（`queries[1]: query must be a string` など）、どのクエリも実行されません。
