# laurus-wasm

WebAssembly bindings for the
[Laurus](https://github.com/mosuka/laurus) search library —
unified lexical, vector, and hybrid search in the browser.

## Features

- **Lexical search** — BM25 scoring with Term, Phrase, Fuzzy,
  Wildcard, Geo, Boolean, and Span queries
- **Vector search** — HNSW, Flat, and IVF indexes with multiple distance metrics
- **Hybrid search** — Combine lexical and vector search
  with RRF or Weighted Sum fusion
- **CJK support** — Japanese, Chinese, and Korean tokenization via [Lindera](https://github.com/lindera/lindera)
- **OPFS persistence** — Data survives page reloads using the
  browser's Origin Private File System
- **JS callback embedder** — Supply your own embedding function
  (e.g. Transformers.js) via a JavaScript callback

## Quick Start

```javascript
import init, { Index, Schema } from "./pkg/laurus_wasm.js";

await init();

// Define schema
const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");
schema.setDefaultFields(["title", "body"]);

// Create an OPFS-persistent index (survives page reloads).
// The schema is persisted to the index on first use; reopening it later
// only needs the name (`Index.open("my-index")`) -- see below.
const index = await Index.open("my-index", schema);

// Index documents
await index.putDocument("doc1", {
  title: "Rust Programming",
  body: "Safety and speed.",
});
await index.putDocument("doc2", {
  title: "Python Basics",
  body: "Versatile language.",
});
await index.commit();

// Search with DSL string
const results = await index.search("programming", 5);
for (const r of results) {
  console.log(r.id, r.score, r.document.title);
}
```

## API Overview

### Index

```javascript
// Create index (in-memory or OPFS-persistent)
const index = await Index.create(schema);              // in-memory (ephemeral); schema is optional (default: empty)
const index = await Index.open("my-index", schema);    // OPFS (persistent), first time: persists schema
const reopened = await Index.open("my-index");         // OPFS (persistent), later: schema loaded automatically

// Document CRUD
await index.putDocument("id", { field: "value" });     // upsert
await index.addDocument("id", { field: "chunk" });     // append (RAG)
const docs = await index.getDocuments("id");
await index.deleteDocuments("id");
await index.commit();                                  // flush + persist to OPFS

// Search
const results = await index.search("query DSL", limit, offset);
const results = await index.searchTerm("field", "term", limit);
const results = await index.searchVector("field", [0.1, ...], limit);
const results = await index.searchVectorText("field", "text", limit);

// Stats
const stats = index.stats();
// { documentCount: 42, vectorFields: {
//     embedding: { count: 42, dimension: 384 }
// } }
```

### Durability / WAL

Each change is appended to the engine's in-memory write-ahead log (WAL). By
default the WAL is flushed on every record. Opt into group commit to batch the
flush for higher write throughput (a crash can lose up to the last unsynced
batch, like SQLite's `synchronous = NORMAL`):

```javascript
import { Index, Schema, WalSyncPolicy } from "./pkg/laurus_wasm.js";

// maxRecords, maxBytes, maxIntervalMs (all optional)
const policy = WalSyncPolicy.group(4096, undefined, 1000);
const index = await Index.open("my-index", schema, policy);

await index.putDocument("doc1", { title: "Hello" });
await index.flushWal(); // flushes the engine WAL only
await index.commit();   // makes changes searchable AND persists to OPFS
```

WASM caveats: the `maxIntervalMs` background timer is a **no-op** on wasm (no
background threads), and `flushWal()` flushes the in-memory engine WAL only —
OPFS persistence still happens at `commit()`. Call `commit()` for durable
persistence. Omit `walSyncPolicy` (or pass `WalSyncPolicy.perRecord()`) to keep
the default per-record behaviour.

### Schema

```javascript
const schema = new Schema();
// 6th argument = analyzer: a built-in name here; custom analyzers (e.g. Japanese)
// are registered with addAnalyzer first and referenced by name (see the API reference).
schema.addTextField("title", true, true, false, true, "english");
schema.addIntegerField("year");
schema.addFloatField("price");
schema.addBooleanField("active");
schema.addDatetimeField("created_at");
schema.addGeoField("location");
schema.addBytesField("thumbnail");
schema.addHnswField("embedding", 384, "cosine", 16, 200, undefined, "minilm");
schema.addFlatField("embedding", 384);
schema.addIvfField("embedding", 384, "cosine", 100, 1);
schema.addEmbedder("minilm", {
  type: "callback",
  embed: async (text) => {
    // Your embedding function here (e.g. Transformers.js)
    return [0.1, 0.2, ...];
  },
});
schema.setDefaultFields(["title", "body"]);
```

## Examples

The [examples/](examples/) directory hosts several self-contained
single-page demos. Open
[`examples/index.html`](examples/index.html) for the landing page,
or jump into a specific sample:

- [`examples/basic/`](examples/basic/) — Basic Japanese hybrid
  search (full-text + vector) via the unified query DSL with
  Transformers.js embeddings.
- [`examples/geo/`](examples/geo/) — Tokyo points-of-interest on a
  Leaflet map, combining a viewport-driven
  `location:geo_bbox(...)` constraint with text and embedding
  queries.

Shared assets (theme stylesheet, logger, dictionary loader,
embedder helper) live under `examples/shared/` so each sample stays
focused on its own behaviour.

## Building from Source

```bash
cd laurus-wasm

# Development build
wasm-pack build --target web --dev

# Release build
wasm-pack build --target web --release

# Bundle the OPFS helper (./opfs subpath) into pkg/
./scripts/postbuild.sh

# Serve the demo landing page
python3 -m http.server 8080
# Open http://localhost:8080/examples/ — pick a sample from there.
```

The post-build script copies `js/opfs.js` and `js/opfs.d.ts` into `pkg/`
and patches `pkg/package.json` so consumers can
`import { downloadDictionary } from "laurus-wasm/opfs"`. Run it after
every `wasm-pack build` (it is idempotent).

## License

MIT
