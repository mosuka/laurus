# Feature Flags

`laurus` クレートはデフォルトで `native` Feature が有効です。`native` はファイルベースのストレージ
（mmap）、rayon によるマルチスレッド並列化、マルチスレッド tokio ランタイムを提供します。これを無効化する
（`default-features = false`）のは `laurus-wasm` が使う `wasm32-unknown-unknown` 向けの構成であり、
mmap を使うファイルストレージなど native 専用のモジュールがこの Feature に依存しているため、
ネイティブターゲットではビルドできません。必要に応じて Embedding サポートを有効にしてください。

## 利用可能な Feature

| Feature | 説明 | 主な依存クレート |
| :--- | :--- | :--- |
| `native`（デフォルト） | ファイルベースのストレージ、rayon 並列化、マルチスレッド tokio | crossbeam-channel, crossbeam-deque, memmap2, num_cpus, rayon, tempfile |
| `embeddings-candle` | Hugging Face Candle によるローカル BERT Embedding と ColBERT のトークンベクトル | candle-core, candle-nn, candle-transformers, hf-hub, tokenizers |
| `embeddings-openai` | OpenAI API Embedding | reqwest |
| `embeddings-multimodal` | CLIP マルチモーダル Embedding（テキスト + 画像） | candle-core, candle-nn, candle-transformers, hf-hub, tokenizers, image |
| `embeddings-all` | すべての Embedding Feature を統合 | 上記すべて |
| `pq-fastscan` | HNSW インデックス向けの実験的な SIMD 高速化 PQ FastScan パス | -- |

## 各 Feature の詳細

### `embeddings-candle`

`CandleBertEmbedder` を有効にし、CPU 上でローカルに BERT モデルを実行できるようにします。あわせて、late interaction の再採点に使う ColBERT のトークンベクトルを作る `CandleColbertEmbedder`（スキーマのエンベダー `candle_colbert`）も有効にします。モデルは初回使用時に Hugging Face Hub からダウンロードされます。

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-candle"] }
```

### `embeddings-openai`

`OpenAIEmbedder` を有効にし、OpenAI Embeddings API を呼び出せるようにします。実行時に `OPENAI_API_KEY` 環境変数が必要です。

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-openai"] }
```

### `embeddings-multimodal`

`CandleClipEmbedder` を有効にし、CLIP ベースのテキストおよび画像 Embedding を使用できるようにします。
`embeddings-candle` とは独立しており、`CandleBertEmbedder` や `CandleColbertEmbedder` は有効になりません。
BERT/ColBERT と CLIP の両方が必要な場合は `embeddings-candle` も併せて有効にしてください。

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-multimodal"] }
```

### `embeddings-all`

すべての Embedding Feature を有効にする便利な Feature です。

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-all"] }
```

## TLS とネットワークの挙動

HTTP 通信を行う 3 つの Embedding Feature は、いずれも単一の TLS スタックを共有します。

| Feature | HTTP クライアント | TLS backend | 信頼するルート証明書のソース |
| :--- | :--- | :--- | :--- |
| `embeddings-candle`, `embeddings-multimodal` | `hf-hub`（`blocking` Feature 経由の `reqwest`） | rustls | OS の信頼ストア（`rustls-platform-verifier` 経由） |
| `embeddings-openai` | `reqwest` | rustls | OS の信頼ストア（`rustls-platform-verifier` 経由） |

いずれも `rustls-platform-verifier` 経由で信頼を解決します。Linux では `rustls-native-certs` が
システムの CA バンドルを読み込み（`SSL_CERT_FILE` / `SSL_CERT_DIR` も尊重します）、macOS では
Keychain、Windows ではシステムの証明書ストアを使用します。これらの Feature のいずれかを使う
コンテナには、信頼ストアが入っている必要があります（Debian/Alpine 系イメージなら
`ca-certificates`、対象 OS に応じた同等のパッケージ）。CA バンドルの無い `scratch` や distroless
イメージでは、Hugging Face Hub からのダウンロードも OpenAI API 呼び出しも TLS ハンドシェイクに
失敗します。動作する Dockerfile の例は `laurus-cli` の
[インストールガイド](../laurus-cli/installation.md)を参照してください。OS の信頼ストアにのみ
導入された独自 CA（例: 社内の TLS インスペクションプロキシ配下）は、3 つの Feature いずれでも
信頼されます。

## Feature Flag がバイナリサイズに与える影響

Embedding Feature を有効にすると、コンパイル時間とバイナリサイズが増加する依存クレートが追加されます。

| 構成 | おおよその影響 |
| :--- | :--- |
| デフォルト（`native`、Embedding なし） | ベースライン |
| `embeddings-candle` | + Candle ML フレームワーク |
| `embeddings-openai` | + reqwest HTTP クライアント |
| `embeddings-multimodal` | + 画像処理 + Candle |
| `embeddings-all` | 上記すべて |

Lexical（キーワード）検索のみが必要な場合は、デフォルトの Feature 構成（`native`、Embedding なし）のまま Laurus を使用することで、最小のバイナリサイズと最速のコンパイル時間を実現できます。
