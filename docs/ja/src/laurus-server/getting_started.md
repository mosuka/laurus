# gRPC サーバーをはじめる

## サーバーの起動

gRPC サーバーは `laurus` CLI の `serve` サブコマンドで起動します。

```bash
laurus serve [OPTIONS]
```

### オプション

| オプション | 短縮形 | 環境変数 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- | :--- |
| `--config <PATH>` | `-c` | `LAURUS_CONFIG` | -- | TOML 設定ファイルのパス |
| `--host <HOST>` | `-H` | `LAURUS_HOST` | `0.0.0.0` | リッスンアドレス |
| `--port <PORT>` | `-p` | `LAURUS_PORT` | `50051` | リッスンポート |
| `--http-port <PORT>` | -- | `LAURUS_HTTP_PORT` | -- | HTTP ゲートウェイポート（設定すると HTTP ゲートウェイが有効化） |
| `--max-result-window <N>` | -- | `LAURUS_MAX_RESULT_WINDOW` | `10000` | 1 回の検索の `offset + limit` の上限。超える検索は拒否される |

ログの詳細度は標準の `RUST_LOG` 環境変数で制御します（デフォルト: `info`）。
`RUST_LOG=laurus=debug,tonic=warn` のようなフィルタディレクティブの詳細は [env_logger の構文](https://docs.rs/env_logger/latest/env_logger/#enabling-logging)を参照してください。

グローバルオプション `--index-dir`（環境変数: `LAURUS_INDEX_DIR`）でインデックスデータのディレクトリを指定します。

```bash
# CLI 引数を使用
laurus --index-dir ./my_index serve --port 8080

# 環境変数を使用
export LAURUS_INDEX_DIR=./my_index
export LAURUS_PORT=8080
export RUST_LOG=debug
laurus serve
```

### 起動時の動作

起動時、サーバーは設定されたデータディレクトリにある既存のインデックスを開こうとします。インデックスが存在しない場合、サーバーはインデックスなしで起動します。後から `CreateIndex` RPC でインデックスを作成できます。

## 設定

コマンドラインオプションの代わりに（または併用して）TOML 設定ファイルを使用できます。詳細は[設定](configuration.md)を参照してください。

```bash
laurus serve --config config.toml
```

## HTTP ゲートウェイ

`--http-port` を設定すると、gRPC サーバーと並行して HTTP/JSON ゲートウェイが起動します。エンドポイントの詳細と使用例は [HTTP ゲートウェイ](http_gateway.md)を参照してください。

```bash
laurus serve --http-port 8080
```

## グレースフルシャットダウン

サーバーがシャットダウンシグナル（Ctrl+C / SIGINT）を受信すると、自動的に以下を実行します。

1. 新しい接続の受け付けを停止
2. インデックスへの保留中の変更をコミット
3. 正常に終了

## gRPC での接続

任意の gRPC クライアントでサーバーに接続できます。簡易テストには [grpcurl](https://github.com/fullstorydev/grpcurl) が便利です。

```bash
# ヘルスチェック
grpcurl -plaintext localhost:50051 laurus.v1.HealthService/Check

# インデックスの作成
grpcurl -plaintext -d '{
  "schema": {
    "fields": {
      "title": {"text": {"indexed": true, "stored": true, "term_vectors": true}},
      "body": {"text": {"indexed": true, "stored": true, "term_vectors": true}}
    },
    "default_fields": ["title", "body"]
  }
}' localhost:50051 laurus.v1.IndexService/CreateIndex

# ドキュメントの追加
grpcurl -plaintext -d '{
  "id": "doc1",
  "document": {
    "fields": {
      "title": {"text_value": "Hello World"},
      "body": {"text_value": "This is a test document."}
    }
  }
}' localhost:50051 laurus.v1.DocumentService/AddDocument

# コミット
grpcurl -plaintext localhost:50051 laurus.v1.DocumentService/Commit

# 検索
grpcurl -plaintext -d '{"query": "body:test", "limit": 10}' \
  localhost:50051 laurus.v1.SearchService/Search
```

詳細は [gRPC API リファレンス](grpc_api.md)を参照してください。HTTP Gateway を使ったステップバイステップの操作ガイドは[ハンズオンチュートリアル](tutorial.md)を参照してください。
