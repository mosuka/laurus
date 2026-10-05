# Feature Flags

`laurus` クレートはデフォルトでは Feature が無効の状態で提供されます。必要に応じて Embedding サポートを有効にしてください。

## 利用可能な Feature

| Feature | 説明 | 主な依存クレート |
| :--- | :--- | :--- |
| `embeddings-candle` | Hugging Face Candle によるローカル BERT Embedding と ColBERT のトークンベクトル | candle-core, candle-nn, candle-transformers, hf-hub, tokenizers |
| `embeddings-openai` | OpenAI API Embedding | reqwest |
| `embeddings-multimodal` | CLIP マルチモーダル Embedding（テキスト + 画像） | image, embeddings-candle |
| `embeddings-all` | すべての Embedding Feature を統合 | 上記すべて |

## 各 Feature の詳細

### `embeddings-candle`

`CandleBertEmbedder` を有効にし、CPU 上でローカルに BERT モデルを実行できるようにします。あわせて、late interaction の再採点に使う ColBERT のトークンベクトルを作る `CandleColbertEmbedder`（スキーマのエンベダー `candle_colbert`）も有効にします。モデルは初回使用時に Hugging Face Hub からダウンロードされます。

```toml
[dependencies]
laurus = { version = "0.12", features = ["embeddings-candle"] }
```

### `embeddings-openai`

`OpenAIEmbedder` を有効にし、OpenAI Embeddings API を呼び出せるようにします。実行時に `OPENAI_API_KEY` 環境変数が必要です。

```toml
[dependencies]
laurus = { version = "0.12", features = ["embeddings-openai"] }
```

### `embeddings-multimodal`

`CandleClipEmbedder` を有効にし、CLIP ベースのテキストおよび画像 Embedding を使用できるようにします。`embeddings-candle` を暗黙的に有効にします。

```toml
[dependencies]
laurus = { version = "0.12", features = ["embeddings-multimodal"] }
```

### `embeddings-all`

すべての Embedding Feature を有効にする便利な Feature です。

```toml
[dependencies]
laurus = { version = "0.12", features = ["embeddings-all"] }
```

## TLS とネットワークの挙動

Embedding Feature は、信頼するルート証明書のソースが異なる 2 系統の TLS
スタックを使用します。

| Feature | HTTP クライアント | TLS backend | 信頼するルート証明書のソース |
| :--- | :--- | :--- | :--- |
| `embeddings-candle`, `embeddings-multimodal` | `hf-hub`（`ureq`） | rustls | バイナリに埋め込まれた Mozilla ルート証明書（`webpki-roots`） |
| `embeddings-openai` | `reqwest` | rustls | OS の信頼ストア（`rustls-platform-verifier` 経由） |

Hugging Face Hub からのモデルダウンロード（`embeddings-candle` /
`embeddings-multimodal`）は、OS の信頼ストアではなくバイナリに埋め込まれた
証明書を使用します。これは意図的な設計です。`ca-certificates` パッケージが
入っていない `scratch` や distroless コンテナ内でも、完全静的リンクの musl
バイナリがモデルをダウンロードできるようにするためです。トレードオフとして、
このパスでは `SSL_CERT_FILE` / `SSL_CERT_DIR` は尊重されず、OS の信頼ストア
にのみ導入された独自 CA（例: 社内の TLS インスペクションプロキシ配下）は
信頼されません。そのようなプロキシ経由で Hugging Face へのダウンロードを
行う必要がある場合は、キャッシュを事前に用意して `HF_HOME` でそれを指すか、
信頼された内部ミラーを `HF_ENDPOINT` で指定してください。

`embeddings-openai` は OS の信頼ストアを参照するため、これを使用する
コンテナには引き続き `ca-certificates` のインストールが必要です。

## Feature Flag がバイナリサイズに与える影響

Embedding Feature を有効にすると、コンパイル時間とバイナリサイズが増加する依存クレートが追加されます。

| 構成 | おおよその影響 |
| :--- | :--- |
| Feature なし（Lexical のみ） | ベースライン |
| `embeddings-candle` | + Candle ML フレームワーク |
| `embeddings-openai` | + reqwest HTTP クライアント |
| `embeddings-multimodal` | + 画像処理 + Candle |
| `embeddings-all` | 上記すべて |

Lexical（キーワード）検索のみが必要な場合は、Feature を有効にせずに Laurus を使用することで、最小のバイナリサイズと最速のコンパイル時間を実現できます。
