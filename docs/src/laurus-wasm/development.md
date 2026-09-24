# Development

## Prerequisites

- [Rust](https://rustup.rs/) (stable, with `wasm32-unknown-unknown` target)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- [Node.js](https://nodejs.org/) (for testing and npm publish)

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## Build

```bash
cd laurus-wasm

# Debug build (faster compilation)
wasm-pack build --target web --dev

# Release build (optimized)
wasm-pack build --target web --release

# For bundler targets (webpack, vite, etc.)
wasm-pack build --target bundler --release
```

## Project Structure

```text
laurus-wasm/
├── Cargo.toml          # Rust dependencies (wasm-bindgen, laurus core)
├── package.json        # npm package metadata
├── src/
│   ├── lib.rs          # Module declarations
│   ├── index.rs        # Index class (CRUD + search)
│   ├── schema.rs       # Schema builder
│   ├── search.rs       # SearchRequest / SearchResult
│   ├── query.rs        # Query type definitions
│   ├── convert.rs      # JsValue ↔ Document conversion
│   ├── analysis.rs     # Tokenizer / Filter wrappers
│   ├── errors.rs       # LaurusError → JsValue conversion
│   └── storage.rs      # OPFS persistence layer
└── js/
    └── opfs_bridge.js  # JS glue for Origin Private File System
```

## Architecture Notes

### Storage Strategy

laurus-wasm uses a two-layer storage approach:

1. **MemoryStorage** (runtime) -- All read/write operations go
   through Laurus's in-memory storage, which satisfies the
   `Storage` trait's `Send + Sync` requirement.

2. **OPFS** (persistence) -- On `commit()`, the entire
   MemoryStorage state is serialized to OPFS files. On
   `Index.open()`, OPFS files are loaded back into MemoryStorage.

This avoids the `Send + Sync` incompatibility of JS handles
while keeping the core engine unchanged.

### Feature Flags

The `laurus` core uses feature flags to support WASM:

```toml
# laurus-wasm depends on laurus without default features
laurus = { workspace = true, default-features = false }
```

This excludes native-only dependencies (tokio/full, rayon,
memmap2, etc.) and uses `#[cfg(target_arch = "wasm32")]`
fallbacks for parallelism.

### Japanese Morphological Analysis

Browser WASM has no filesystem, so the standard `{ "language": "japanese", "dict": "/path/to/ipadic" }` analyzer preset cannot be used. `laurus-wasm` exposes `JapaneseAnalyzer.fromBytes(...)` (defined in [`src/analysis.rs`](https://github.com/mosuka/laurus/blob/main/laurus-wasm/src/analysis.rs)) so that a Lindera IPADIC dictionary archive can be fetched into OPFS at runtime, read back as the nine raw byte arrays Lindera needs, and handed to the analyzer:

```javascript
import { JapaneseAnalyzer, Schema } from "laurus-wasm";
import { downloadDictionary, loadDictionaryFiles } from "laurus-wasm/opfs";

await downloadDictionary("./dict/lindera-ipadic.zip", "ipadic");
const f = await loadDictionaryFiles("ipadic");
const ja = JapaneseAnalyzer.fromBytes(
  f.metadata, f.dictTrie, f.dictValsIdx, f.dictVals,
  f.dictWordsIdx, f.dictWords, f.matrixMtx, f.charDef, f.unk, "normal",
);

const schema = new Schema();
schema.addAnalyzer("ja-ipadic", ja);
schema.addTextField("body", undefined, undefined, undefined, undefined, "ja-ipadic");
```

The OPFS helpers (`downloadDictionary`, `getDictionaryVersion`, `loadDictionaryFiles`, `hasDictionary`, `listDictionaries`, `removeDictionary`) live in [`js/opfs.js`](https://github.com/mosuka/laurus/blob/main/laurus-wasm/js/opfs.js) and are re-exported as the `laurus-wasm/opfs` subpath in `package.json`. See [API Reference → JapaneseAnalyzer](api_reference.md#japaneseanalyzer) for the argument table.

### Callback Embedder

In addition to the `"precomputed"` embedder (vectors supplied directly by the caller), `laurus-wasm` accepts a `"callback"` embedder where the JS side provides an `async embed: (text) => Promise<number[]>` function. The engine invokes this callback during document ingestion and `searchVectorText()` queries, which lets you wire in any in-browser embedding library (Transformers.js, ONNX Runtime Web, etc.) without rebuilding the WASM module:

```javascript
import { pipeline } from "@xenova/transformers";

const embedder = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
);

schema.addEmbedder("minilm", {
  type: "callback",
  embed: async (text) => {
    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  },
});

schema.addHnswField(
  "embedding", 384, "cosine",
  undefined, undefined, undefined, "minilm",
);
```

The wasm-bindgen glue holds the JS callback via a `Closure`, so it stays alive for the lifetime of the index. There is no `Send + Sync` requirement on the callback because it only runs on the main thread.

## Testing

```bash
# Build check
cargo build -p laurus-wasm --target wasm32-unknown-unknown

# Clippy
cargo clippy -p laurus-wasm --target wasm32-unknown-unknown -- -D warnings
```

Browser tests can be run with `wasm-pack test`:

```bash
wasm-pack test --headless --chrome
```
