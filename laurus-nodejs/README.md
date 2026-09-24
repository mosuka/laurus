# laurus-nodejs

Node.js/TypeScript bindings for the
[Laurus](https://github.com/mosuka/laurus) search library —
unified lexical, vector, and hybrid search.

## Features

- **Lexical search** — BM25 scoring with Term, Phrase, Fuzzy,
  Wildcard, Geo, Boolean, and Span queries
- **Vector search** — HNSW, Flat, and IVF indexes with multiple distance metrics
- **Hybrid search** — Combine lexical and vector search
  with RRF or Weighted Sum fusion
- **CJK support** — Japanese, Chinese, and Korean tokenization via [Lindera](https://github.com/lindera/lindera)
- **Native performance** — Rust core via [napi-rs](https://napi.rs), no C API overhead
- **TypeScript types** — Auto-generated `.d.ts` type definitions

## Installation

```bash
npm install laurus-nodejs
```

## Quick Start

```javascript
import { Index, Schema } from "laurus-nodejs";

// Define schema
const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");
schema.setDefaultFields(["title", "body"]);

// Create an in-memory index
const index = await Index.create(null, schema);

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
// Create index (in-memory or file-based)
const index = await Index.create();                    // in-memory
const index = await Index.create("./myindex", schema); // persistent (writes ./myindex/schema.toml + ./myindex/store/)
const reopened = await Index.create("./myindex");       // reopen: omit `schema`, it's loaded from disk

// Document CRUD
await index.putDocument("id", { field: "value" });     // upsert
await index.addDocument("id", { field: "chunk" });     // append (RAG)
const docs = await index.getDocuments("id");
await index.deleteDocuments("id");
await index.commit();

// Search
const results = await index.search("query DSL", limit, offset);
const results = await index.searchTerm("field", "term", limit);
const results = await index.searchVector("field", [0.1, ...], limit);
const results = await index.searchVectorText("field", "text", limit);
const results = await index.searchWithRequest(searchRequest);

// Stats
const stats = index.stats();
// { documentCount: 42, vectorFields: {
//     embedding: { count: 42, dimension: 384 }
// } }

// Close (releases the storage lock deterministically; see below)
index.close();
```

A file-based index holds an exclusive lock on its directory for as long as
the `Index` is open, so a second `Index.create()` on the same path fails
while the first is still around. Call `close()` when you are done with an
index — especially before reopening the same path — rather than relying on
the JS garbage collector, whose timing is not deterministic. `close()` is
idempotent; every other method throws after it has been called.

To cheaply check whether another process committed since you last looked —
without opening (or reopening) the index at all — use the exported
`peekCommitGeneration(path)`:

```javascript
import { peekCommitGeneration } from "laurus-nodejs";

const before = peekCommitGeneration(path);
// ... later ...
if (peekCommitGeneration(path) !== before) {
  // something changed on disk; reopen the index to pick it up
}
```

It reads the persisted commit generation directly off disk, with no `Engine`
construction at all — no storage lock, no WAL recovery, no embedder
loading — so it works even before any `Index` for that path has been created
in this process. Throws if `path` isn't a laurus index directory (no
persisted schema).

### Durability / WAL

A persistent index writes every change to a write-ahead log (WAL). By default
the WAL is `fsync`-ed on every record, so each write is fully durable. Opt into
group commit to batch `fsync` for higher write throughput (a crash can lose up
to the last unsynced batch, like SQLite's `synchronous = NORMAL`):

```javascript
import { Index, WalSyncPolicy } from "laurus-nodejs";

// maxRecords, maxBytes, maxIntervalMs (all optional)
const policy = WalSyncPolicy.group(4096, undefined, 1000);
const index = await Index.create("./myindex", schema, policy);

await index.putDocument("doc1", { title: "Hello" });
await index.flushWal(); // force a durable barrier on demand
await index.commit();   // also flushes the WAL
```

Omit `walSyncPolicy` (or pass `WalSyncPolicy.perRecord()`) to keep the default
per-record durability.

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
schema.addHnswField("embedding", 384, "cosine", 16, 200, undefined, "bert");
schema.addFlatField("embedding", 384);
schema.addIvfField("embedding", 384, "cosine", 100, 1);
schema.addEmbedder("bert", {
  type: "candle_bert",
  model: "sentence-transformers/all-MiniLM-L6-v2",
});
schema.setDefaultFields(["title", "body"]);
```

### Search Request (Advanced)

```javascript
import {
  PhraseQuery,
  RRF,
  SearchRequest,
  TermQuery,
  VectorQuery,
  VectorTextQuery,
  WeightedSum,
} from "laurus-nodejs";

const req = new SearchRequest({ limit: 10, offset: 0 });
req.setQueryDsl("title:hello");
req.setLexicalTerm(new TermQuery("body", "programming"));
req.setLexicalPhrase(new PhraseQuery("title", ["machine", "learning"]));
req.setVectorQuery(new VectorQuery("embedding", [0.1, 0.2, ...]));
req.setVectorTextQuery(new VectorTextQuery("embedding", "query text"));
req.setFilterTerm(new TermQuery("category", "tech"));
req.setRrfFusion(new RRF(60.0));
req.setWeightedSumFusion(new WeightedSum(0.3, 0.7));

const results = await index.searchWithRequest(req);
```

### Text Analysis

```javascript
import { WhitespaceTokenizer, SynonymDictionary, SynonymGraphFilter } from "laurus-nodejs";

const tokenizer = new WhitespaceTokenizer();
const tokens = tokenizer.tokenize("hello world");

const synDict = new SynonymDictionary();
synDict.addSynonymGroup(["ml", "machine learning"]);

const filter = new SynonymGraphFilter(synDict, true, 0.8);
const expanded = filter.apply(tokens);
```

## Data Types

| JavaScript | Laurus Field Type |
| --- | --- |
| `string` | Text |
| `number` (integer) | Int64 |
| `number` (float) | Float64 |
| `boolean` | Boolean |
| `null` | Null |
| `number[]` | Vector |
| `{ lat, lon }` | Geo |
| `Date` / ISO8601 string | DateTime |
| `Buffer` | Bytes |

## Examples

See the [examples/](examples/) directory:

- [quickstart.mjs](examples/quickstart.mjs) — Basic index, document, and search
- [lexical-search.mjs](examples/lexical-search.mjs) — All lexical query types
- [vector-search.mjs](examples/vector-search.mjs) — Vector search with HNSW
- [hybrid-search.mjs](examples/hybrid-search.mjs) — Hybrid search
  with RRF and WeightedSum fusion

## Building from Source

```bash
cd laurus-nodejs
npm install
npm run build        # release build
npm run build:debug  # debug build
npm test             # run tests
```

## License

MIT
