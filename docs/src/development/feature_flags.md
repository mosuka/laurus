# Feature Flags

The `laurus` crate ships with no default features. Enable embedding support as needed.

## Available Flags

| Feature | Description | Key Dependencies |
| :--- | :--- | :--- |
| `embeddings-candle` | Local BERT embeddings and ColBERT token vectors via Hugging Face Candle | candle-core, candle-nn, candle-transformers, hf-hub, tokenizers |
| `embeddings-openai` | OpenAI API embeddings | reqwest |
| `embeddings-multimodal` | CLIP multimodal embeddings (text + image) | image, embeddings-candle |
| `embeddings-all` | All embedding features combined | All of the above |

## What Each Flag Enables

### `embeddings-candle`

Enables `CandleBertEmbedder` for running BERT models locally on the CPU, and `CandleColbertEmbedder` (the `candle_colbert` schema embedder) for ColBERT token vectors used by late-interaction rescoring. Models are downloaded from Hugging Face Hub on first use.

```toml
[dependencies]
laurus = { version = "0.12", features = ["embeddings-candle"] }
```

### `embeddings-openai`

Enables `OpenAIEmbedder` for calling the OpenAI Embeddings API. Requires an `OPENAI_API_KEY` environment variable at runtime.

```toml
[dependencies]
laurus = { version = "0.12", features = ["embeddings-openai"] }
```

### `embeddings-multimodal`

Enables `CandleClipEmbedder` for CLIP-based text and image embeddings. Implies `embeddings-candle`.

```toml
[dependencies]
laurus = { version = "0.12", features = ["embeddings-multimodal"] }
```

### `embeddings-all`

Convenience flag that enables all embedding features.

```toml
[dependencies]
laurus = { version = "0.12", features = ["embeddings-all"] }
```

## TLS and Network Behavior

The embedding features use two independent TLS stacks with different trust
sources:

| Feature | HTTP client | TLS backend | Trust source |
| :--- | :--- | :--- | :--- |
| `embeddings-candle`, `embeddings-multimodal` | `hf-hub` (`ureq`) | rustls | Bundled Mozilla root certificates (`webpki-roots`) |
| `embeddings-openai` | `reqwest` | rustls | OS trust store (via `rustls-platform-verifier`) |

Model downloads from Hugging Face Hub (`embeddings-candle` /
`embeddings-multimodal`) use certificates bundled into the binary rather than
the operating system's trust store. This is deliberate: it lets a fully
static musl binary download models inside a `scratch` or distroless
container with no `ca-certificates` package installed. The tradeoff is that
`SSL_CERT_FILE` / `SSL_CERT_DIR` are not honored on this path, and a custom
CA installed only in the OS trust store (for example behind a corporate
TLS-inspecting proxy) will not be trusted. If you need to route Hugging Face
downloads through such a proxy, pre-populate the cache and point `HF_HOME` at
it, or set `HF_ENDPOINT` to an internally trusted mirror.

`embeddings-openai` reads the OS trust store, so containers using it still
need `ca-certificates` installed.

## Feature Flag Impact on Binary Size

Enabling embedding features adds dependencies that increase compile time and binary size:

| Configuration | Approximate Impact |
| :--- | :--- |
| No features (lexical only) | Baseline |
| `embeddings-candle` | + Candle ML framework |
| `embeddings-openai` | + reqwest HTTP client |
| `embeddings-multimodal` | + image processing + Candle |
| `embeddings-all` | All of the above |

If you only need lexical (keyword) search, you can use Laurus with no features enabled for the smallest binary and fastest compile time.
