# MCP ツールリファレンス

laurus MCP サーバーは以下のツールを公開しています。

## connect

実行中の laurus-server gRPC エンドポイントに接続します。`--endpoint` フラグなしでサーバーを起動した場合や、実行時に別の laurus-server に切り替える場合に、他のツールを使用する前にこのツールを呼び出してください。

### パラメーター

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `endpoint` | string | はい | gRPC エンドポイント URL（例: `http://localhost:50051`） |

### 例

```text
Tool: connect
endpoint: "http://localhost:50051"
```

結果: `Connected to laurus-server at http://localhost:50051.`

---

## create_index

指定されたスキーマで新しい検索インデックスを作成します。

### パラメーター

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `schema_json` | string | はい | JSON 文字列としてのスキーマ定義 |

### スキーマ JSON フォーマット

FieldOption は serde の externally-tagged 表現を使用します（バリアント名がキーになります）：

```json
{
  "dynamic_field_policy": "Dynamic",
  "fields": {
    "title":     { "Text":    { "indexed": true, "stored": true } },
    "body":      { "Text":    {} },
    "score":     { "Float":   {} },
    "count":     { "Integer": {} },
    "active":    { "Boolean": {} },
    "created":   { "DateTime": {} },
    "embedding": { "Hnsw":    { "dimension": 384 } }
  }
}
```

オプションの `dynamic_field_policy` キーは、スキーマに宣言されていないフィールドが投入ドキュメントに含まれる場合の挙動を制御します。指定可能な値は `"Strict"` / `"Dynamic"`（デフォルト）/ `"Ignore"`。**警告**: `"Dynamic"` では integer フィールドに入ってきた float 値が静かに切り捨てられます（`3.14` → `3`）。厳密さが必要なら `"Strict"` を使用してください。詳細な挙動マトリクスは [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

### 例

```text
Tool: create_index
schema_json: {"fields": {"title": {"Text": {}}, "body": {"Text": {}}}}
```

結果: `Index created successfully at /path/to/index.`

3D ECEF 座標を扱う `Geo3d` フィールドを含むスキーマ:

```json
{
  "fields": {
    "title":    { "Text":  { "indexed": true, "stored": true } },
    "position": { "Geo3d": { "indexed": true, "stored": true } }
  }
}
```

座標系については [3D 地理検索 (ECEF)](../concepts/geo3d.md) を参照してください。`Geo3d` フィールドは `geo3d_distance` / `geo3d_bbox` / `geo3d_nearest` の DSL 形式で検索できます（後述の **search** ツールを参照）。

---

## get_stats

現在の検索インデックスの統計情報（ドキュメント数、ベクトルフィールド情報など）を取得します。

### パラメーター

なし。

### 結果

```json
{
  "document_count": 42,
  "vector_fields": {
    "embedding": {
      "vector_count": 42,
      "dimension": 384
    }
  }
}
```

`vector_fields` はフィールド名をキーとするマップで、各エントリにはインデックス済みベクトル数とフィールドに設定された次元数が含まれます。

---

## get_schema

現在のインデックスのスキーマ（全フィールド定義と設定）を取得します。

### パラメーター

なし。

### 結果

```json
{
  "fields": {
    "title": { "Text": { "indexed": true, "stored": true } },
    "body": { "Text": {} },
    "embedding": { "Hnsw": { "dimension": 384 } }
  },
  "default_fields": ["title", "body"]
}
```

---

## put_document

インデックスにドキュメントを上書き（upsert）します。同じ ID のドキュメントが既に存在する場合、全チャンクが削除されてから新しいドキュメントがインデックスされます。ドキュメント追加後は `commit` を呼び出してください。

### パラメーター

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `id` | string | はい | 外部ドキュメント識別子 |
| `fields` | object | はい | JSON オブジェクトとしてのドキュメントフィールド |

### 例

```text
Tool: put_document
id: "doc-1"
fields: {"title": "Hello World", "body": "これはテストドキュメントです。"}
```

結果: `Document 'doc-1' put (upserted). Call commit to persist changes.`

**Geo3d 値を含む例:**

```text
Tool: put_document
id: "drone-1"
fields: {"title": "東京上空のドローン", "position": {"x": -3955182.0, "y": 3350553.0, "z": 3700276.0}}
```

MCP サーバーは 3D ECEF 点を `x`、`y`、`z` キーを持つ JSON オブジェクト（メートル単位）として受け付けます。これは HTTP ゲートウェイや laurus-cli と同じ形式です（[HTTP Gateway](../laurus-server/http_gateway.md) の「JSON フィールド値の型推論」セクション参照）。

---

## add_document

インデックスにドキュメントを新しいチャンクとして追加します。`put_document` とは異なり、同じ ID の既存ドキュメントを削除せずに追記します。大きなドキュメントをチャンクに分割する際に便利です。ドキュメント追加後は `commit` を呼び出してください。

### パラメーター

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `id` | string | はい | 外部ドキュメント識別子 |
| `fields` | object | はい | JSON オブジェクトとしてのドキュメントフィールド |

### 例

```text
Tool: add_document
id: "doc-1"
fields: {"title": "Hello World - Part 2", "body": "これは続きです。"}
```

結果: `Document 'doc-1' added as chunk. Call commit to persist changes.`

---

## put_documents

1 回の呼び出しで多数のドキュメントを Put（Upsert）します。エントリは入力順に逐次適用され、バッチ全体で WAL fsync は 1 回です — ドキュメントごとに `put_document` を呼ぶよりはるかに高速です。1 バッチ内で重複した ID はデデュープされます（最後の出現が勝ち）。実行後に `commit` を呼び出してください。

### パラメータ

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `documents` | array | はい | `{"id": "...", "fields": {...}}` エントリの配列。各エントリの `fields` は `put_document` と同じ形式 |

### 例

```text
Tool: put_documents
documents: [
  {"id": "doc-1", "fields": {"title": "Hello"}},
  {"id": "doc-2", "fields": {"title": "World"}}
]
```

結果: `2 documents put (upserted). Call commit to persist changes.`

失敗した場合は該当エントリで中断し、適用済みの prefix はロールバックされないため、バッチ（またはその suffix）の再試行は冪等です。

---

## add_documents

1 回の呼び出しで多数のドキュメントを新しいチャンクとして追加します。`put_documents` と異なり既存ドキュメントは削除されないため、ID を繰り返すと同一論理ドキュメントの複数チャンクになります。バッチ全体で WAL fsync は 1 回です。実行後に `commit` を呼び出してください。

### パラメータ

`put_documents` と同じです。

### 例

```text
Tool: add_documents
documents: [
  {"id": "doc-1", "fields": {"title": "Part 1"}},
  {"id": "doc-1", "fields": {"title": "Part 2"}}
]
```

結果: `2 documents added as chunks. Call commit to persist changes.`

---

## get_documents

外部 ID で全ドキュメント（チャンクを含む）を取得します。

### パラメーター

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `id` | string | はい | 外部ドキュメント識別子 |

### 結果

```json
{
  "id": "doc-1",
  "documents": [
    { "fields": { "title": "Hello World", "body": "これはテストドキュメントです。" } }
  ]
}
```

---

## delete_documents

外部 ID で全ドキュメント（チャンクを含む）を削除します。削除後は `commit` を呼び出してください。

### パラメーター

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `id` | string | はい | 外部ドキュメント識別子 |

結果: `Documents 'doc-1' deleted. Call commit to persist changes.`

---

## commit

保留中の変更をディスクにコミットします。変更を検索可能かつ永続的にするため、`put_document`、`add_document`、または `delete_documents` の後に必ず呼び出してください。

### パラメーター

なし。

結果: `Changes committed successfully.`

---

## add_field

インデックスにフィールドを追加します。

### パラメーター

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `name` | string | はい | フィールド名 |
| `field_option_json` | string | はい | JSON 形式のフィールド設定 |

### 例

```json
{
  "name": "category",
  "field_option_json": "{\"Text\": {\"indexed\": true, \"stored\": true}}"
}
```

---

## delete_field

インデックスからフィールドを削除します。既にインデックスされたデータは残りますが、削除されたフィールドにはアクセスできなくなります。

### パラメーター

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `name` | string | はい | 削除するフィールド名 |

### 例

```text
Tool: delete_field
name: "category"
```

結果: `Field 'category' deleted.`

---

## search

laurus 統一クエリ DSL を使用してドキュメントを検索します。Lexical 検索、Vector 検索、ハイブリッド検索を単一のクエリ文字列でサポートします。

### パラメーター

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `query` | string | はい | laurus 統一クエリ DSL による検索クエリ |
| `limit` | integer | いいえ | 最大結果数（デフォルト: 10） |
| `offset` | integer | いいえ | ページネーション用スキップ数（デフォルト: 0） |
| `fusion` | string | いいえ | ハイブリッド検索用の融合アルゴリズム（JSON） |
| `field_boosts` | string | いいえ | フィールド毎のブースト係数（JSON） |
| `highlight` | string | いいえ | フィールドごとのハイライト済みフラグメント（JSON、[ハイライトの例](#ハイライトの例)を参照） |

### クエリ DSL の例

#### Lexical 検索

| クエリ | 説明 |
| :--- | :--- |
| `hello` | デフォルトフィールド全体のターム検索 |
| `title:hello` | フィールド指定のターム検索 |
| `title:hello AND body:world` | ブール AND |
| `"exact phrase"` | フレーズ検索 |
| `roam~2` | ファジー検索（編集距離 2） |
| `count:[1 TO 10]` | 範囲検索 |
| `title:helo~1` | フィールド指定のファジー検索 |

#### 3D 地理検索

| クエリ | 説明 |
| :--- | :--- |
| `position:geo3d_distance(x, y, z, distance_m)` | `(x, y, z)` を中心とした最大距離（メートル単位）の球 |
| `position:geo3d_bbox(min_x, min_y, min_z, max_x, max_y, max_z)` | 3D 軸並行バウンディングボックス |
| `position:geo3d_nearest(x, y, z, k)` | `(x, y, z)` に最も近い k 個の近傍点 |

`position` はフィールド名で、スキーマで宣言した実際の `Geo3d` 型フィールドに置き換えてください。完全な DSL 構文は [Query DSL → 3D 地理クエリ](../concepts/query_dsl.md#3d-geographic-queries-geo3d_) を参照してください。

#### Vector 検索

| クエリ | 説明 |
| :--- | :--- |
| `content:"cute kitten"` | 特定フィールドでの Vector 検索（クォート付き） |
| `content:python` | 特定フィールドでの Vector 検索（クォートなし） |
| `content:"cute kitten"^0.8` | 重み付き Vector 検索 |
| `a:"cats" b:"dogs"^0.5` | 複数の Vector クエリ |

#### ハイブリッド検索

| クエリ | 説明 |
| :--- | :--- |
| `title:hello content:"cute kitten"` | Lexical + Vector（OR/union — いずれかの結果を返す） |
| `title:hello +content:"cute kitten"` | Lexical + Vector（AND/intersection — 両方にマッチした結果のみ） |
| `+title:hello +content:"cute kitten"` | 両方必須（AND）。Lexical フィールドの `+` は required clause |
| `title:hello AND body:world content:"cats"^0.8` | ブール Lexical + 重み付き Vector |

### 融合アルゴリズムの例

```json
{"rrf": {"k": 60.0}}
```

```json
{"weighted_sum": {"lexical_weight": 0.7, "vector_weight": 0.3}}
```

### フィールドブーストの例

```json
{"title": 2.0, "body": 1.0}
```

### ハイライトの例

`highlight` はフィールドごとのハイライト済みフラグメントを要求します（Issue #1134）。ハイライトはこのツールの lexical クエリに従うため、Vector-only のクエリはハイライトを生成しません。また `stored: true` のテキストフィールドのみハイライト可能です。省略形はフィールド名の配列だけです。

```json
["body"]
```

オブジェクト形式では `HighlightConfig` の各設定（`max_fragments`、`fragment_size`、`tag`、`css_class`、`require_field_match`、`max_analyzed_chars`、`return_entire_field_if_no_highlight`）を追加できます。

```json
{"fields": ["body"], "max_fragments": 2, "tag": "em"}
```

### 結果

`highlight` を要求し、少なくとも1つのフィールドが実際にハイライトされた場合、各結果に `"highlights"` オブジェクトが追加されます。

```json
{
  "total": 2,
  "results": [
    {
      "id": "doc-1",
      "score": 3.14,
      "fields": { "title": "Hello World", "body": "..." },
      "highlights": { "body": ["<em>Hello</em> World"] }
    },
    {
      "id": "doc-2",
      "score": 1.57,
      "fields": { "title": "Hello Again", "body": "..." }
    }
  ]
}
```

---

## search_batch

独立した複数の検索を 1 回のラウンドトリップで実行します。すべてのクエリは
サーバー上で並列に実行され、`limit` と `offset` はすべてのクエリに共通で
適用されます。エージェントが 1 ターンで複数のサブクエリを発行する場合に
有用です。

### パラメータ

| 名前 | 型 | 必須 | 説明 |
| :--- | :--- | :--- | :--- |
| `queries` | 文字列の配列 | はい | laurus 統一クエリ DSL のクエリ文字列（`search` と同じ構文） |
| `limit` | 整数 | いいえ | クエリごとの最大結果数（デフォルト: 10） |
| `offset` | 整数 | いいえ | クエリごとのページネーションオフセット（デフォルト: 0） |
| `highlight` | 文字列 | いいえ | フィールドごとのハイライト済みフラグメント（JSON）。`search` の `highlight` と同じ形式で、バッチ内のすべてのクエリに同一に適用される |

### 結果

`batch` 配列は入力順を保持します。`batch[i]` は `queries[i]` の結果セットです。

```json
{
  "batch": [
    {
      "total": 1,
      "results": [
        { "id": "doc-1", "score": 3.14, "fields": { "title": "Hello World" } }
      ]
    },
    {
      "total": 1,
      "results": [
        { "id": "doc-2", "score": 2.71, "fields": { "title": "Vector Search" } }
      ]
    }
  ]
}
```

---

## 典型的なワークフロー

```text
1. connect          → 実行中の laurus-server に接続
2. create_index     → スキーマを定義（インデックスが存在しない場合）
3. add_field        → フィールドを追加（必要に応じて）
   delete_field     → フィールドを削除（必要に応じて）
4. put_document     → ドキュメントを上書き（必要に応じて繰り返し）
   add_document     → ドキュメントチャンクを追記（必要に応じて）
5. commit           → 変更をディスクに永続化
6. search           → インデックスを検索
7. get_documents    → ID でドキュメントを取得
8. delete_documents → ドキュメントを削除
9. commit           → 変更を永続化
```
