# Feature Flags

The `laurus` crate enables the `native` feature by default, which provides file-based
storage (mmap), multi-threaded parallelism (rayon), and a multi-threaded tokio runtime.
Disabling it (`default-features = false`) is the `wasm32-unknown-unknown` configuration
used by `laurus-wasm`; it does not build on a native target, since native-only modules
such as mmap-backed file storage are gated on it. Enable embedding support as needed.

## Available Flags

| Feature | Description | Key Dependencies |
| :--- | :--- | :--- |
| `native` (default) | File-based storage, rayon parallelism, multi-threaded tokio | crossbeam-channel, crossbeam-deque, memmap2, num_cpus, rayon, tempfile |
| `embeddings-candle` | Local BERT embeddings and ColBERT token vectors via Hugging Face Candle | candle-core, candle-nn, candle-transformers, hf-hub, tokenizers |
| `embeddings-openai` | OpenAI API embeddings | reqwest |
| `embeddings-multimodal` | CLIP multimodal embeddings (text + image) | candle-core, candle-nn, candle-transformers, hf-hub, tokenizers, image |
| `embeddings-all` | All embedding features combined | All of the above |
| `pq-fastscan` | Experimental SIMD-accelerated PQ FastScan path for HNSW indexes | -- |

## What Each Flag Enables

### `embeddings-candle`

Enables `CandleBertEmbedder` for running BERT models locally on the CPU, and `CandleColbertEmbedder` (the `candle_colbert` schema embedder) for ColBERT token vectors used by late-interaction rescoring. Models are downloaded from Hugging Face Hub on first use.

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-candle"] }
```

### `embeddings-openai`

Enables `OpenAIEmbedder` for calling the OpenAI Embeddings API. Requires an `OPENAI_API_KEY` environment variable at runtime.

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-openai"] }
```

### `embeddings-multimodal`

Enables `CandleClipEmbedder` for CLIP-based text and image embeddings. This is independent
of `embeddings-candle`: it does not enable `CandleBertEmbedder` or `CandleColbertEmbedder`,
so combine it with `embeddings-candle` if you need both BERT/ColBERT and CLIP.

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-multimodal"] }
```

### `embeddings-all`

Convenience flag that enables all embedding features.

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-all"] }
```

## TLS and Network Behavior

All three HTTP-calling embedding features now share a single TLS stack:

| Feature | HTTP client | TLS backend | Trust source |
| :--- | :--- | :--- | :--- |
| `embeddings-candle`, `embeddings-multimodal` | `hf-hub` (`reqwest`, via its `blocking` feature) | rustls | OS trust store (via `rustls-platform-verifier`) |
| `embeddings-openai` | `reqwest` | rustls | OS trust store (via `rustls-platform-verifier`) |

All of them resolve trust through `rustls-platform-verifier`: on Linux this reads the
system's CA bundle via `rustls-native-certs` (honoring `SSL_CERT_FILE` / `SSL_CERT_DIR`),
on macOS it uses Keychain, and on Windows the system certificate store. A container
running any of these features needs a populated trust store -- `ca-certificates` on
Debian/Alpine-based images, or the equivalent for the target OS. A `scratch` or distroless
image with no CA bundle will fail the TLS handshake for both Hugging Face Hub downloads
and the OpenAI API call; see the `laurus-cli` [installation guide](../laurus-cli/installation.md)
for a working Dockerfile. A custom CA installed only in the OS trust store (for example
behind a corporate TLS-inspecting proxy) is trusted on all three features.

## Feature Flag Impact on Binary Size

Enabling embedding features adds dependencies that increase compile time and binary size:

| Configuration | Approximate Impact |
| :--- | :--- |
| Default (`native`, no embedding) | Baseline |
| `embeddings-candle` | + Candle ML framework |
| `embeddings-openai` | + reqwest HTTP client |
| `embeddings-multimodal` | + image processing + Candle |
| `embeddings-all` | All of the above |

If you only need lexical (keyword) search, the default feature set (`native`, no embedding features) gives the smallest binary and fastest compile time.
