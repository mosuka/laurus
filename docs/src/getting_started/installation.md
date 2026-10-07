# Installation

## Add Laurus to Your Project

Add `laurus` and `tokio` (async runtime) to your `Cargo.toml`:

```toml
[dependencies]
laurus = "0.13"
tokio = { version = "1", features = ["full"] }
```

## Feature Flags

Laurus ships with a minimal default feature set. Enable additional features as needed:

| Feature | Description | Use Case |
| :--- | :--- | :--- |
| `native` *(default)* | Core library (lexical search, storage, analyzers — no embedding) | Keyword search only |
| `embeddings-candle` | Local BERT embeddings via Hugging Face Candle | Vector search without external API |
| `embeddings-openai` | OpenAI API embeddings (text-embedding-3-small, etc.) | Cloud-based vector search |
| `embeddings-multimodal` | CLIP embeddings for text + image via Candle | Multimodal (text-to-image) search |
| `embeddings-all` | All embedding features above | Full embedding support |

### Examples

**Lexical search only** (no embeddings needed):

```toml
[dependencies]
laurus = "0.13"
```

**Vector search with local model** (no API key required):

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-candle"] }
```

**Vector search with OpenAI**:

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-openai"] }
```

**Everything**:

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-all"] }
```

## Verify Installation

Create a minimal program to verify that Laurus compiles:

```rust
use laurus::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Laurus version: {}", laurus::VERSION);
    Ok(())
}
```

```bash
cargo run
```

If you see the version printed, you are ready to proceed to the [Quick Start](quickstart.md).
