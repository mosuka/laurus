# laurus-server

[![Crates.io](https://img.shields.io/crates/v/laurus-server.svg)](https://crates.io/crates/laurus-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Laurus](https://github.com/mosuka/laurus) 検索エンジンの HTTP/JSON ゲートウェイ付き gRPC サーバーです。

## 機能

- **永続エンジン** -- インデックスはリクエスト間で開いたまま維持され、呼び出しごとの WAL リプレイが不要
- **フル gRPC API** -- インデックス管理、ドキュメント CRUD、コミット、検索（単発 + ストリーミング）
- **HTTP ゲートウェイ** -- gRPC と併用可能なオプションの HTTP/JSON ゲートウェイで REST スタイルのアクセスを提供
- **ヘルスチェック** -- ロードバランサーやオーケストレーター向けの標準ヘルスチェックエンドポイント
- **グレースフルシャットダウン** -- Ctrl+C / SIGINT、または Unix では SIGTERM で保留中の変更を自動的にコミット
- **TOML 設定** -- オプションの設定ファイルと CLI・環境変数によるオーバーライド

## クイックスタート

```bash
# デフォルト設定で起動（gRPC ポート 50051）
laurus serve

# HTTP ゲートウェイ付きで起動
laurus serve --http-port 8080

# 設定ファイルを指定して起動
laurus serve --config config.toml
```

## gRPC サービス

| サービス | RPC |
| :--- | :--- |
| `HealthService` | `Check` |
| `IndexService` | `CreateIndex`, `GetIndex`, `GetSchema`, `AddField`, `DeleteField` |
| `DocumentService` | `PutDocument`, `AddDocument`, `GetDocuments`, `DeleteDocuments`, `Commit` |
| `SearchService` | `Search`, `SearchStream` |

## HTTP ゲートウェイエンドポイント

| メソッド | パス | 説明 |
| :--- | :--- | :--- |
| `GET` | `/v1/health` | ヘルスチェック |
| `POST` | `/v1/index` | インデックス作成 |
| `GET` | `/v1/index` | インデックス統計取得 |
| `GET` | `/v1/schema` | スキーマ取得 |
| `POST` | `/v1/schema/fields` | フィールド追加 |
| `DELETE` | `/v1/schema/fields/{name}` | フィールド削除 |
| `PUT` | `/v1/documents/{id}` | ドキュメント上書き（upsert） |
| `POST` | `/v1/documents/{id}` | ドキュメント追加（チャンク） |
| `GET` | `/v1/documents/{id}` | ドキュメント取得 |
| `DELETE` | `/v1/documents/{id}` | ドキュメント削除 |
| `POST` | `/v1/commit` | 変更をコミット |
| `POST` | `/v1/search` | 検索 |
| `POST` | `/v1/search/stream` | ストリーミング検索（SSE） |

## ドキュメント

- [サーバーガイド](https://mosuka.github.io/laurus/ja/laurus-server.html)
- [はじめに](https://mosuka.github.io/laurus/ja/laurus-server/getting_started.html)
- [設定](https://mosuka.github.io/laurus/ja/laurus-server/configuration.html)
- [gRPC API リファレンス](https://mosuka.github.io/laurus/ja/laurus-server/grpc_api.html)
- [HTTP ゲートウェイ](https://mosuka.github.io/laurus/ja/laurus-server/http_gateway.html)
- [チュートリアル](https://mosuka.github.io/laurus/ja/laurus-server/tutorial.html)

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](../LICENSE) ファイルを参照してください。
