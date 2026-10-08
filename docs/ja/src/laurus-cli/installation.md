# インストール

## ビルド済みバイナリ

[GitHub のリリースページ](https://github.com/mosuka/laurus/releases)では、
以下のターゲット向けにビルド済みの `laurus` バイナリを配布しています。
いずれも `--features embeddings-all` でビルドされています。

| ターゲットトリプル | プラットフォーム | libc | リンク方式 | アーカイブ |
| :--- | :--- | :--- | :--- | :--- |
| `x86_64-unknown-linux-gnu` | Linux x86_64 | glibc | 動的 | `.tar.gz` |
| `aarch64-unknown-linux-gnu` | Linux aarch64 | glibc | 動的 | `.tar.gz` |
| `x86_64-unknown-linux-musl` | Linux x86_64 | musl | 静的 | `.tar.gz` |
| `aarch64-unknown-linux-musl` | Linux aarch64 | musl | 静的 | `.tar.gz` |
| `x86_64-apple-darwin` | macOS Intel | -- | 動的 | `.tar.gz` |
| `aarch64-apple-darwin` | macOS Apple Silicon | -- | 動的 | `.tar.gz` |
| `x86_64-pc-windows-msvc` | Windows x86_64 | MSVC | 動的 | `.zip` |
| `aarch64-pc-windows-msvc` | Windows arm64 | MSVC | 動的 | `.zip` |

```bash
VERSION=v0.13.2
TARGET=x86_64-unknown-linux-musl
curl -fsSL -O "https://github.com/mosuka/laurus/releases/download/${VERSION}/laurus-${VERSION}-${TARGET}.tar.gz"
tar -xzf "laurus-${VERSION}-${TARGET}.tar.gz"
./laurus --version
```

### どのビルドを使うべきか

- 一般的な Linux ディストリビューションでは **gnu** ビルドを使ってください。
  musl のアロケータ（`mallocng`）は、laurus の索引処理のようなマルチスレッド・
  allocation-heavy なワークロードでは glibc のアロケータより明確に遅くなります。
- Alpine、distroless、`scratch`、あるいはビルドランナーより古い glibc しか
  持たないホストでは **musl** ビルドを使ってください。musl バイナリは
  完全に静的リンクされており、動的ライブラリへの依存が一切ありません。

### コンテナで musl バイナリを使う

```dockerfile
FROM alpine:3.22
# Embedding Feature が HTTPS 通信を行う場合は常に必要です: reqwest の
# rustls backend は、embeddings-openai だけでなく Hugging Face からの
# モデルダウンロード（embeddings-candle / embeddings-multimodal）でも
# OS の信頼ストアを参照して検証します。ビルド済みバイナリは
# --features embeddings-all でビルドされているため、Lexical 検索のみの
# 用途でもインストールしておいてください。詳細は開発ガイドの
# 「Feature Flags」を参照してください。
RUN apk add --no-cache ca-certificates
COPY laurus /usr/local/bin/laurus
ENTRYPOINT ["laurus"]
```

パッケージマネージャすら無い環境でも動作しますが、Embedding Feature を一切呼び出さない場合に限ります
（HTTPS ダウンロードを検証する CA バンドルが無いため）:

```dockerfile
FROM scratch
COPY laurus /laurus
ENTRYPOINT ["/laurus"]
```

musl / `scratch` での運用における注意点:

- DNS 解決には musl 内蔵のリゾルバが使われ、`/etc/resolv.conf` を読み込み、
  NSS プラグインは無視されます。コンテナランタイムは常に `/etc/resolv.conf`
  を提供するため通常は問題になりませんが、手動で `scratch` イメージを
  構築する場合はこれを省略しないよう注意してください。
- Rust の標準ライブラリは生成するスレッドに独自の既定スタックサイズ
  （2 MiB）を使用するため、musl のより小さい `pthread` の既定値は
  実質的に問題になりません。

## crates.io からインストール

```bash
cargo install laurus-cli
```

これにより `laurus` バイナリが `~/.cargo/bin/` にインストールされます。

## ソースからインストール

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
cargo install --path laurus-cli
```

## 確認

```bash
laurus --version
```
