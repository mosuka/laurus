# Summary

- [はじめに](README.md)
- [アーキテクチャ概要](architecture.md)

# Getting Started

- [はじめに](getting_started.md)
  - [インストール](getting_started/installation.md)
  - [クイックスタート](getting_started/quickstart.md)
  - [サンプル](getting_started/examples.md)

# コアコンセプト

- [スキーマとフィールド](concepts/schema_and_fields.md)
- [テキスト解析](concepts/analysis.md)
- [エンベディング](concepts/embedding.md)
- [ストレージ](concepts/storage.md)
- [インデクシング](concepts/indexing.md)
  - [Lexical インデクシング](concepts/indexing/lexical_indexing.md)
  - [Vector インデクシング](concepts/indexing/vector_indexing.md)
- [検索](concepts/search.md)
  - [Lexical 検索](concepts/search/lexical_search.md)
  - [Vector 検索](concepts/search/vector_search.md)
  - [ハイブリッド検索](concepts/search/hybrid_search.md)
  - [検索結果のグループ化](concepts/search/result_grouping.md)
- [Query DSL](concepts/query_dsl.md)
- [BKD-Tree](concepts/bkd_tree.md)
- [3D 地理検索 (ECEF)](concepts/geo3d.md)

# laurus

- [ライブラリ概要](laurus.md)
  - [Engine](laurus/engine.md)
  - [スコアリングとランキング](laurus/scoring.md)
  - [ファセット](laurus/faceting.md)
  - [ハイライト](laurus/highlighting.md)
  - [スペル修正](laurus/spelling_correction.md)
  - [ID 管理](laurus/id_management.md)
  - [永続化と WAL](laurus/persistence.md)
  - [削除とコンパクション](laurus/deletions.md)
  - [エラーハンドリング](laurus/error_handling.md)
  - [拡張性](laurus/extensibility.md)
  - [API リファレンス](laurus/api_reference.md)

# laurus-cli

- [CLI 概要](laurus-cli.md)
  - [インストール](laurus-cli/installation.md)
  - [ハンズオンチュートリアル](laurus-cli/tutorial.md)
  - [コマンド](laurus-cli/commands.md)
  - [スキーマフォーマット](laurus-cli/schema_format.md)
  - [REPL](laurus-cli/repl.md)

# laurus-server

- [サーバー概要](laurus-server.md)
  - [はじめに](laurus-server/getting_started.md)
  - [ハンズオンチュートリアル](laurus-server/tutorial.md)
  - [設定](laurus-server/configuration.md)
  - [gRPC API リファレンス](laurus-server/grpc_api.md)
  - [HTTP Gateway](laurus-server/http_gateway.md)

# laurus-mcp

- [MCP サーバー概要](laurus-mcp.md)
  - [はじめに](laurus-mcp/getting_started.md)
  - [ツールリファレンス](laurus-mcp/tools.md)

# laurus-python

- [Python バインディング概要](laurus-python.md)
  - [インストール](laurus-python/installation.md)
  - [クイックスタート](laurus-python/quickstart.md)
  - [API リファレンス](laurus-python/api_reference.md)
  - [開発](laurus-python/development.md)

# laurus-nodejs

- [Node.js バインディング概要](laurus-nodejs.md)
  - [インストール](laurus-nodejs/installation.md)
  - [クイックスタート](laurus-nodejs/quickstart.md)
  - [API リファレンス](laurus-nodejs/api_reference.md)
  - [開発](laurus-nodejs/development.md)

# laurus-wasm

- [WASM バインディング概要](laurus-wasm.md)
  - [インストール](laurus-wasm/installation.md)
  - [クイックスタート](laurus-wasm/quickstart.md)
  - [API リファレンス](laurus-wasm/api_reference.md)
  - [開発](laurus-wasm/development.md)

# laurus-ruby

- [Ruby バインディング概要](laurus-ruby.md)
  - [インストール](laurus-ruby/installation.md)
  - [クイックスタート](laurus-ruby/quickstart.md)
  - [API リファレンス](laurus-ruby/api_reference.md)
  - [開発](laurus-ruby/development.md)

# laurus-php

- [PHP バインディング概要](laurus-php.md)
  - [インストール](laurus-php/installation.md)
  - [クイックスタート](laurus-php/quickstart.md)
  - [API リファレンス](laurus-php/api_reference.md)
  - [開発](laurus-php/development.md)

# 開発ガイド

- [ビルドとテスト](development/build_and_test.md)
- [ベンチマーク](development/benchmarking.md)
- [Feature Flags](development/feature_flags.md)
- [プロジェクト構成](development/project_structure.md)
