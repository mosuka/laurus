# インストール

## RubyGems からインストール

```bash
gem install laurus
```

または `Gemfile` に追加します：

```ruby
gem "laurus"
```

その後、以下を実行します：

```bash
bundle install
```

## ソースからビルド

ソースからビルドするには Rust ツールチェーン（1.89 以降）と [rb_sys](https://github.com/oxidize-rb/rb-sys) が必要です。

```bash
# リポジトリをクローン
git clone https://github.com/mosuka/laurus.git
cd laurus/laurus-ruby

# 依存関係をインストール
bundle install

# ネイティブ拡張をコンパイル
bundle exec rake compile

# またはローカルに gem をインストール
gem build laurus.gemspec
gem install laurus-*.gem
```

## 動作確認

```ruby
require "laurus"
index = Laurus::Index.new
puts index  # Index()
```

## 動作要件

- Ruby 3.1 以降
- Rust ツールチェーン（gem インストール時に `rb_sys` 経由で自動的に呼び出されます）
- コンパイル済みネイティブ拡張以外のランタイム依存関係なし
