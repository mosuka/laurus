# インストール

`laurus-php` は Rust で書かれた PHP 拡張です。PECL では配布されておらず、Composer はインストール経路として利用しません — リポジトリの `composer.json` は開発時のテスト依存（PHPUnit）のみを宣言しています。Cargo で共有ライブラリをソースからビルドし、PHP の拡張ディレクトリに配置して `php.ini` で有効化してください。

## ソースからビルド

ソースからビルドするには Rust ツールチェーン（1.89 以降）と PHP 8.1 以降（開発ヘッダー付き）が必要です。

```bash
# リポジトリをクローン
git clone https://github.com/mosuka/laurus.git
cd laurus/laurus-php

# ネイティブ拡張をビルド
cargo build --release

# 共有ライブラリを PHP エクステンションディレクトリにコピー
# （正確なパスは OS と PHP バージョンによって異なります）
cp ../target/release/liblaurus_php.so $(php -r "echo ini_get('extension_dir');")
```

次に `php.ini` にエクステンションを追加します：

```ini
extension=laurus_php.so
```

または、コマンドラインでエクステンションをロードすることもできます：

```bash
php -d extension=liblaurus_php.so your_script.php
```

## 動作確認

```php
<?php

use Laurus\Index;

$index = new Index();
echo $index;  // Index()
```

## 動作要件

- PHP 8.1 以降（開発ヘッダー付き: `php-dev` / `php-devel`）
- Rust ツールチェーン 1.89 以降（Cargo 含む）
- コンパイル済みネイティブ拡張以外のランタイム依存関係なし
