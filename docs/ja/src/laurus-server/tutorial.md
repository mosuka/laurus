# ハンズオンチュートリアル

このチュートリアルでは、laurus-server を使った一連のワークフローを体験します。サーバーの起動、インデックスの作成、ドキュメントの登録、検索、更新、削除を順を追って説明します。すべての操作は HTTP Gateway 経由の `curl` コマンドで行います。

## 前提条件

- laurus CLI がインストール済み（[インストール](../getting_started/installation.md) を参照）
- `curl` が利用可能

## Step 1: サーバーの起動

HTTP Gateway を有効にして laurus-server を起動します:

```bash
laurus --index-dir /tmp/laurus/tutorial serve --port 50051 --http-port 8080
```

gRPC サーバー（ポート 50051）と HTTP Gateway（ポート 8080）が起動したことを示すログが表示されます。

サーバーが正常に動作しているか確認します:

```bash
curl http://localhost:8080/v1/health
```

期待されるレスポンス:

```json
{"status":"SERVING_STATUS_SERVING"}
```

## Step 2: インデックスの作成

Lexical 検索用のテキストフィールドと Vector 検索用のベクトルフィールドを含むスキーマでインデックスを作成します。この例では**カスタムアナライザー**と**エンベッダー定義**、フィールドごとの設定を示しています:

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "schema": {
      "analyzers": {
        "body_analyzer": {
          "char_filters": [{"type": "unicode_normalization", "form": "nfkc"}],
          "tokenizer": {"type": "regex"},
          "token_filters": [
            {"type": "lowercase"},
            {"type": "stop", "words": ["the", "a", "an", "is", "it"]}
          ]
        }
      },
      "embedders": {
        "my_embedder": {"type": "precomputed"}
      },
      "fields": {
        "title": {"text": {"indexed": true, "stored": true, "term_vectors": true, "analyzer": "standard"}},
        "body": {"text": {"indexed": true, "stored": true, "term_vectors": true, "analyzer": "body_analyzer"}},
        "category": {"text": {"indexed": true, "stored": true, "term_vectors": false, "analyzer": "keyword"}},
        "embedding": {"hnsw": {"dimension": 4, "distance": "DISTANCE_METRIC_COSINE", "m": 16, "ef_construction": 200, "embedder": "my_embedder"}}
      },
      "default_fields": ["title", "body"]
    }
  }'
```

3 つのテキストフィールドと 1 つのベクトルフィールドを持つインデックスが作成されます:

- `title` — 組み込みの `standard` アナライザー（トークン化＋小文字化）を使用。
- `body` — `analyzers` セクションで定義したカスタム `body_analyzer`（NFKC 正規化＋正規表現トークナイザー＋小文字化＋カスタムストップワード）を使用。
- `category` — `keyword` アナライザー（値全体を単一トークンとして扱い、完全一致用）を使用。
- `embedding` — HNSW ベクトルインデックス、4 次元、コサイン距離。`embedders` で定義した `my_embedder` を使用。このチュートリアルでは `precomputed`（外部で事前計算したベクトル）を使用。本番環境では、使用する埋め込みモデルに合わせた次元数（例: 384 や 768）を指定してください。

`default_fields` を設定することで、フィールド指定なしのクエリは `title` と `body` の両方を検索します。

### 組み込みアナライザー

`standard`, `keyword`, `english`, `japanese`, `simple`, `noop`。省略時はエンジンのデフォルト（`standard`）が使用されます。

### カスタムアナライザーのコンポーネント

以下のコンポーネントを組み合わせてカスタムアナライザーを構成できます:

- **トークナイザー:** `whitespace`, `unicode_word`, `regex`, `ngram`, `lindera`, `whole`
- **文字フィルター:** `unicode_normalization`, `pattern_replace`, `mapping`, `japanese_iteration_mark`
- **トークンフィルター:** `lowercase`, `stop`, `stem`, `boost`, `limit`, `strip`, `remove_empty`, `flatten_graph`

### エンベッダー

`embedders` セクションでベクトルの生成方法を定義します。各ベクトルフィールドは `embedder` オプションでエンベッダーを名前で参照できます。利用可能なタイプ:

- `precomputed` — ベクトルは外部で事前計算して供給（自動埋め込みなし）。
- `candle_bert` — Candle によるローカル BERT モデル。パラメータ: `model`（HuggingFace モデルID）。`embeddings-candle` フィーチャが必要。
- `candle_clip` — ローカル CLIP マルチモーダルモデル。パラメータ: `model`（HuggingFace モデルID）。`embeddings-multimodal` フィーチャが必要。
- `openai` — OpenAI API。パラメータ: `model`（例: `"text-embedding-3-small"`）。`embeddings-openai` フィーチャと `OPENAI_API_KEY` 環境変数が必要。

BERT エンベッダーの例（`embeddings-candle` フィーチャが必要）:

```json
{
  "embedders": {
    "bert": {"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"}
  },
  "fields": {
    "embedding": {"hnsw": {"dimension": 384, "embedder": "bert"}}
  }
}
```

インデックスが作成されたことを確認します:

```bash
curl http://localhost:8080/v1/index
```

期待されるレスポンス:

```json
{"document_count":0,"vector_fields":{}}
```

## Step 3: ドキュメントの登録

ドキュメントをインデックスに追加します。`PUT` を使って ID 指定でドキュメントを登録します。各ドキュメントにはテキストフィールドと `embedding` ベクトルが含まれます（本番環境では、これらのベクトルは埋め込みモデルから生成されます）:

```bash
curl -X PUT http://localhost:8080/v1/documents/doc001 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Introduction to Rust Programming",
      "body": "Rust is a modern systems programming language that focuses on safety, speed, and concurrency.",
      "category": "programming",
      "embedding": [0.9, 0.1, 0.2, 0.0]
    }
  }'
```

```bash
curl -X PUT http://localhost:8080/v1/documents/doc002 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Web Development with Rust",
      "body": "Building web applications with Rust has become increasingly popular. Frameworks like Actix and Rocket make it easy to create fast and secure web services.",
      "category": "web-development",
      "embedding": [0.7, 0.3, 0.5, 0.1]
    }
  }'
```

```bash
curl -X PUT http://localhost:8080/v1/documents/doc003 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Python for Data Science",
      "body": "Python is the most popular language for data science and machine learning. Libraries like NumPy and Pandas provide powerful tools for data analysis.",
      "category": "data-science",
      "embedding": [0.1, 0.8, 0.1, 0.9]
    }
  }'
```

ベクトルフィールドは数値の JSON 配列として指定します。配列の長さはスキーマで設定した `dimension`（このチュートリアルでは 4）と一致する必要があります。

## Step 4: 変更のコミット

ドキュメントはコミットするまで検索対象になりません。変更をコミットします:

```bash
curl -X POST http://localhost:8080/v1/commit
```

## Step 5: ドキュメントの検索

### 基本的な検索

"rust" を含むドキュメントを検索します:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "rust", "limit": 10}'
```

デフォルトフィールド（`title` と `body`）が検索されます。`doc001` と `doc002` が返されます。

### フィールド指定検索

`title` フィールドのみを検索します:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "title:python", "limit": 10}'
```

`doc003` のみが返されます。

### カテゴリ検索

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "category:programming", "limit": 10}'
```

`doc001` のみが返されます。

### ブーリアンクエリ

`AND`、`OR`、`NOT` で条件を組み合わせます:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "rust AND web", "limit": 10}'
```

"rust" と "web" の両方を含む `doc002` のみが返されます。

### フィールドブースト

`title` フィールドのスコアを引き上げて、タイトルの一致を優先します:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "rust",
    "limit": 10,
    "field_boosts": {"title": 2.0}
  }'
```

### ベクトル検索

ベクトルの類似度で検索します。`query_vectors` にクエリベクトルを指定し、検索対象のフィールドを指定します:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query_vectors": [
      {
        "vector": [0.85, 0.15, 0.2, 0.05],
        "fields": ["embedding"]
      }
    ],
    "limit": 10
  }'
```

`embedding` ベクトルがクエリベクトルに最も近いドキュメントが返されます。`doc001` が最上位にランクされます（最も類似したベクトル）。

### ハイブリッド検索

Lexical 検索と Vector 検索を組み合わせて、より良い結果を得ます。`fusion` パラメータで両方のスコアの統合方法を制御します:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "rust",
    "query_vectors": [
      {
        "vector": [0.85, 0.15, 0.2, 0.05],
        "fields": ["embedding"]
      }
    ],
    "fusion": {"rrf": {"k": 60.0}},
    "limit": 10
  }'
```

Reciprocal Rank Fusion（RRF）を使って Lexical 検索と Vector 検索の結果を統合します。重み付き和による統合も可能です:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "programming",
    "query_vectors": [
      {
        "vector": [0.85, 0.15, 0.2, 0.05],
        "fields": ["embedding"]
      }
    ],
    "fusion": {"weighted_sum": {"lexical_weight": 0.3, "vector_weight": 0.7}},
    "limit": 10
  }'
```

## Step 6: ドキュメントの取得

ID を指定して特定のドキュメントを取得します:

```bash
curl http://localhost:8080/v1/documents/doc001
```

期待されるレスポンス（ベクトルフィールドも含まれます）:

```json
{
  "documents": [
    {
      "fields": {
        "title": "Introduction to Rust Programming",
        "body": "Rust is a modern systems programming language that focuses on safety, speed, and concurrency.",
        "category": "programming",
        "embedding": [0.9, 0.1, 0.2, 0.0]
      }
    }
  ]
}
```

## Step 7: ドキュメントの更新

同じ ID で `PUT` を実行するとドキュメント全体が置き換わります:

```bash
curl -X PUT http://localhost:8080/v1/documents/doc001 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Introduction to Rust Programming",
      "body": "Rust is a modern systems programming language that focuses on safety, speed, and concurrency. It provides memory safety without garbage collection.",
      "category": "programming",
      "embedding": [0.9, 0.1, 0.2, 0.0]
    }
  }'
```

コミットして確認します:

```bash
curl -X POST http://localhost:8080/v1/commit
curl http://localhost:8080/v1/documents/doc001
```

更新された `body` の内容が反映されています。

## Step 8: ドキュメントの削除

ID を指定してドキュメントを削除します:

```bash
curl -X DELETE http://localhost:8080/v1/documents/doc003
```

コミットして反映させます:

```bash
curl -X POST http://localhost:8080/v1/commit
```

ドキュメントが削除されたことを確認します:

```bash
curl http://localhost:8080/v1/documents/doc003
```

期待されるレスポンス:

```json
{"documents":[]}
```

検索結果にも表示されなくなります:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "python", "limit": 10}'
```

結果は返されません。

期待されるレスポンス:

```json
{"results":[]}
```

## Step 9: インデックス統計の確認

現在のインデックス統計を確認します:

```bash
curl http://localhost:8080/v1/index
```

`document_count` は削除後の残りのドキュメント数を反映しています。

## Step 10: クリーンアップ

`Ctrl+C` でサーバーを停止します。サーバーはグレースフルシャットダウンを行い、保留中の変更をコミットしてから終了します。

チュートリアルで作成したデータを削除します:

```bash
rm -rf /tmp/laurus/tutorial
```

## さらに進んだ使い方: 実際の埋め込みモデルの利用

上記のチュートリアルでは簡略化のために `precomputed` ベクトルを使用しました。本番環境では、埋め込みモデルを使ってテキストを自動的にベクトルに変換するのが一般的です。ここでは BERT ベースのエンベッダーの設定方法を示します。

### 前提条件

`embeddings-candle` フィーチャを有効にして laurus をビルドします:

```bash
cargo build --release --features embeddings-candle
```

### BERT エンベッダーを使ったスキーマ

インデックスを作成します:

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "schema": {
      "embedders": {
        "bert": {
          "type": "candle_bert",
          "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
      },
      "fields": {
        "title": {"text": {"indexed": true, "stored": true, "analyzer": "standard"}},
        "body": {"text": {"indexed": true, "stored": true, "analyzer": "standard"}},
        "embedding": {"hnsw": {"dimension": 384, "distance": "DISTANCE_METRIC_COSINE", "m": 16, "ef_construction": 200, "embedder": "bert"}}
      },
      "default_fields": ["title", "body"]
    }
  }'
```

モデルは初回使用時に HuggingFace Hub から自動ダウンロードされます。`dimension`（384）はモデルの出力次元数と一致させる必要があります。

ドキュメントを追加します。`embedding` フィールドにはテキストを渡すだけで、エンベッダーが自動的にベクトルに変換します:

```bash
curl -X PUT http://localhost:8080/v1/documents/doc001 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Introduction to Rust Programming",
      "body": "Rust is a modern systems programming language.",
      "embedding": "Rust is a modern systems programming language."
    }
  }'
```

```bash
curl -X PUT http://localhost:8080/v1/documents/doc002 \
  -H 'Content-Type: application/json' \
  -d '{
    "fields": {
      "title": "Web Development with Rust",
      "body": "Building web applications with Rust using Actix and Rocket.",
      "embedding": "Building web applications with Rust using Actix and Rocket."
    }
  }'
```

コミットします:

```bash
curl -X POST http://localhost:8080/v1/commit
```

1 つの `query` で lexical 検索とセマンティック検索を同時に行います。`systems programming` は lexical 句、`embedding:"..."` はベクトル句で、そのテキストは検索時にエンベッダーがベクトルに変換します:

```bash
curl -X POST http://localhost:8080/v1/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "systems programming embedding:\"systems programming language\"",
    "fusion": {"rrf": {"k": 60.0}},
    "limit": 10
  }'
```

`precomputed` エンベッダーではベクトルを直接渡す必要があります（`query_vectors` の `vector` は数値の配列）が、`candle_bert` のようなテキスト対応エンベッダーを使うと、インデックス時も検索時もテキストを直接渡せます。

### OpenAI Embeddings の利用

OpenAI の Embedding API を使う場合は、`OPENAI_API_KEY` 環境変数を設定し、`embeddings-openai` フィーチャでビルドします:

```bash
cargo build --release --features embeddings-openai
export OPENAI_API_KEY="sk-..."
```

インデックスを作成します:

```bash
curl -X POST http://localhost:8080/v1/index \
  -H 'Content-Type: application/json' \
  -d '{
    "schema": {
      "embedders": {
        "openai": {
          "type": "openai",
          "model": "text-embedding-3-small"
        }
      },
      "fields": {
        "title": {"text": {"indexed": true, "stored": true}},
        "embedding": {"hnsw": {"dimension": 1536, "distance": "DISTANCE_METRIC_COSINE", "embedder": "openai"}}
      },
      "default_fields": ["title"]
    }
  }'
```

`text-embedding-3-small` モデルは 1536 次元のベクトルを出力します。

### 利用可能な埋め込みモデル

| タイプ | フィーチャフラグ | モデル例 | 次元数 |
| :--- | :--- | :--- | :--- |
| `candle_bert` | `embeddings-candle` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `candle_clip` | `embeddings-multimodal` | `openai/clip-vit-base-patch32` | 512 |
| `openai` | `embeddings-openai` | `text-embedding-3-small` | 1536 |

## 次のステップ

- [ベクトル検索とハイブリッド検索](../concepts/search/hybrid_search.md)で意味的な類似検索を試す
- [gRPC API リファレンス](grpc_api.md)で API 仕様の詳細を確認する
- [設定](configuration.md)で本番環境向けの設定を行う
- `grpcurl` や gRPC クライアントライブラリを使ったプログラムからのアクセスについては[はじめに](getting_started.md)を参照
