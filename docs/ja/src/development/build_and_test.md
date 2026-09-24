# ビルドとテスト

## 前提条件

- **Rust** 1.85 以降（edition 2024）
- **Cargo**（Rust に付属）
- **protobuf コンパイラ**（`protoc`）-- `laurus-server` のビルドに必要
- **[cargo-zigbuild](https://github.com/rust-cross/cargo-zigbuild)** --
  任意。静的リンクの musl バイナリをローカルでクロスビルドする場合のみ必要
  （[静的リンクの musl バイナリをクロスビルドする](#静的リンクの-musl-バイナリをクロスビルドする)を参照）

## ビルド

```bash
# すべてのクレートをビルド
cargo build

# 特定の Feature を指定してビルド
cargo build --features embeddings-candle

# リリースモードでビルド
cargo build --release
```

## 静的リンクの musl バイナリをクロスビルドする

`laurus-cli` のリリースワークフローは、動的リンクの glibc バイナリに加えて、
完全に静的リンクされた `x86_64-unknown-linux-musl` /
`aarch64-unknown-linux-musl` バイナリもビルドします
（[ビルド済みバイナリ](../laurus-cli/installation.md#ビルド済みバイナリ)を参照）。

前提条件:

```bash
rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl
pip install cargo-zigbuild
```

`cargo build` の代わりに `cargo zigbuild` でビルドします:

```bash
cargo zigbuild --release --target x86_64-unknown-linux-musl \
  -p laurus-cli --features embeddings-all
```

（`x86_64` ターゲットについては `make build-laurus-cli-musl` と同等です）

**なぜ `apt install musl-tools` ではなく `cargo-zigbuild` を使うのか？**
Ubuntu の `musl-tools` パッケージは `musl-gcc`（C）を提供しますが
`musl-g++`（C++）は提供しません。laurus の `--features embeddings-all` の
依存関係の大部分は C のみか純 Rust です（`aws-lc-sys`、`onig_sys`）が、
`musl-tools` だけの構成は依存 Feature を 1 つ切り替えるだけで再び C++ を
要求しかねません（`tokenizers` の `esaxx_fast`。laurus では意図的に無効化
しています。[Feature Flags](feature_flags.md) を参照）。`cargo-zigbuild` は
クロス C/C++ ツールチェーンとして [Zig](https://ziglang.org/) を使用し、
各ターゲット向けの musl ヘッダとライブラリを同梱しているため、Docker が
不要です。

ビルド結果が本当に静的であることを確認する:

```bash
file target/x86_64-unknown-linux-musl/release/laurus
# -> ELF 64-bit LSB executable, ..., statically linked
readelf -d target/x86_64-unknown-linux-musl/release/laurus | grep NEEDED
# -> (出力なし)
```

この手順が対応する CI 設定については
[`.github/workflows/release.yml`](../../../.github/workflows/release.yml)
の `build-binary` を参照してください。

## テスト

```bash
# すべてのワークスペーステストを実行（デフォルト Feature）
cargo test

# 名前を指定して特定のテストを実行
cargo test <test_name>

# 特定のクレートのテストを実行
cargo test -p laurus
cargo test -p laurus-cli
cargo test -p laurus-server
cargo test -p laurus-mcp
```

### 言語バインディングのテスト

各言語バインディングは固有のツールチェーン（Python virtualenv、Node.js
npm、Ruby Bundler、PHP Composer、`wasm32-unknown-unknown` ターゲット）を持ちます。
Makefile はこれらをラップし、各ターゲットがツールチェーンを準備したうえで
テストを実行します。

```bash
make test-laurus-python   # cargo test -p laurus-python + Maturin 経由の pytest
make test-laurus-nodejs   # npm run build:debug + npm test
make test-laurus-wasm     # cargo build -p laurus-wasm --target wasm32-unknown-unknown
make test-laurus-ruby     # cargo test -p laurus-ruby + Ruby minitest
make test-laurus-php      # cargo build -p laurus-php --release + PHPUnit
```

`laurus-php` は `laurus-ruby` との `links = "clang"` 競合のため Cargo ワークスペースから
除外されており、上記の Makefile ターゲット経由でスタンドアロンクレートとしてビルド・テストします。
対応する `format-laurus-*` / `lint-laurus-*` / `build-laurus-*` のバリアントを含む全ターゲットは
[`Makefile`](../../../Makefile) を参照してください。

## Lint

```bash
# clippy を警告エラー扱いで実行
cargo clippy -- -D warnings
```

## フォーマット

```bash
# フォーマットチェック
cargo fmt --check

# フォーマットを適用
cargo fmt
```

## ドキュメント

### API ドキュメント

```bash
# Rust API ドキュメントを生成して開く
cargo doc --no-deps --open
```

### mdBook ドキュメント

```bash
# ドキュメントサイトをビルド
mdbook build docs

# ローカルプレビューサーバーを起動 (http://localhost:3000)
mdbook serve docs

# Markdown ファイルを Lint
markdownlint-cli2 "docs/src/**/*.md"

# バインディングの README / docs / examples のスニペットが Node.js / WASM / PHP の
# 現行シグネチャと一致しているか確認（`make lint` と CI でも実行）
python3 scripts/check-binding-snippets.py
```
