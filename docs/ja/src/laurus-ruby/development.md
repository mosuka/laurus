# 開発環境のセットアップ

このページでは `laurus-ruby` バインディングのローカル開発環境の構築、ビルド、テストスイートの実行方法について説明します。

## 前提条件

- **Rust** 1.89 以降（Cargo 含む）
- **Ruby** 3.1 以降（Bundler 含む）
- リポジトリがローカルにクローンされていること

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
```

## ビルド

### 開発ビルド

Rust ネイティブ拡張をデバッグモードでコンパイルします。Rust ソースを変更した場合は再実行してください。

```bash
cd laurus-ruby
bundle install
bundle exec rake compile
```

### リリースビルド

```bash
gem build laurus.gemspec
```

### ビルドの確認

```ruby
ruby -e "
require 'laurus'
index = Laurus::Index.new
puts index.stats
"
# {"document_count"=>0, "vector_fields"=>{}}
```

## テスト

テストは [Minitest](https://github.com/minitest/minitest) を使用しており、`test/` ディレクトリにあります。

```bash
# 全テスト実行
bundle exec rake test
```

特定のテストファイルを実行する場合：

```bash
bundle exec ruby -Ilib -Itest test/test_index.rb
```

## Lint とフォーマット

```bash
# Rust lint（Clippy）
cargo clippy -p laurus-ruby -- -D warnings

# Rust フォーマットチェック
cargo fmt -p laurus-ruby --check

# フォーマット適用
cargo fmt -p laurus-ruby
```

## クリーンアップ

```bash
# ビルド成果物を削除
bundle exec rake clean

# インストールされた gem を削除
rm -rf vendor/bundle
```

## Makefile リファレンス

| ターゲット | 説明 |
| :--- | :--- |
| `make build-laurus-ruby` | Bundler install + `bundle exec rake compile`（リリース gem） |
| `make test-laurus-ruby` | Rust 単体テスト + Ruby minitest |
| `make lint-laurus-ruby` | Clippy（`-D warnings`） |
| `make format-laurus-ruby` | `cargo fmt -p laurus-ruby` |

## プロジェクト構成

```text
laurus-ruby/
├── Cargo.toml          # Rust クレートマニフェスト
├── laurus.gemspec      # Gem 仕様
├── Gemfile             # Bundler 依存関係ファイル
├── Rakefile            # Rake タスク（compile、test、clean）
├── lib/
│   └── laurus.rb       # Ruby エントリポイント（ネイティブ拡張をロード）
├── ext/
│   └── laurus_ruby/    # ネイティブ拡張ビルド設定
│       └── extconf.rb  # rb_sys 拡張設定
├── src/                # Rust ソース（Magnus バインディング）
│   ├── lib.rs          # モジュール登録
│   ├── index.rs        # Index クラス
│   ├── schema.rs       # Schema クラス
│   ├── query.rs        # クエリクラス
│   ├── search.rs       # SearchRequest / SearchResult / Fusion
│   ├── analysis.rs     # Tokenizer / Filter / Token
│   ├── convert.rs      # Ruby ↔ DataValue 変換
│   └── errors.rs       # エラーマッピング
├── test/               # Minitest テスト
│   ├── test_helper.rb
│   └── test_index.rb
└── examples/           # 実行可能な Ruby サンプル
```
