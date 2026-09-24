# Quick Start

## Basic Usage (In-memory)

```javascript
import init, { Index, Schema } from 'laurus-wasm';

// Initialize the WASM module
await init();

// Define a schema
const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");
schema.setDefaultFields(["title", "body"]);

// Create an in-memory index
const index = await Index.create(schema);

// Add documents
await index.putDocument("doc1", {
  title: "Introduction to Rust",
  body: "Rust is a systems programming language"
});
await index.putDocument("doc2", {
  title: "WebAssembly Guide",
  body: "WASM enables near-native performance in the browser"
});
await index.commit();

// Search
const results = await index.search("rust programming");
for (const result of results) {
  console.log(`${result.id}: ${result.score}`);
  console.log(result.document);
}
```

## Persistent Storage (OPFS)

```javascript
import init, { Index, Schema } from 'laurus-wasm';

await init();

const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");

// Open a persistent index (data survives page reloads).
// The first time, the schema is persisted alongside the data.
const index = await Index.open("my-index", schema);

// Add documents
await index.putDocument("doc1", {
  title: "Hello",
  body: "World"
});

// commit() persists to OPFS automatically
await index.commit();

// On the next page load, Index.open("my-index") -- with no schema
// argument -- restores both the data and the persisted schema.
const reopened = await Index.open("my-index");
```

## Japanese Morphological Search

Browser WASM cannot use a filesystem path for the Lindera dictionary,
so build the analyzer from raw IPADIC bytes loaded from OPFS.

```javascript
import init, { Index, Schema, JapaneseAnalyzer } from 'laurus-wasm';
import {
  downloadDictionary,
  getDictionaryVersion,
  loadDictionaryFiles,
  hasDictionary,
} from 'laurus-wasm/opfs';

await init();

// 1. Cache the IPADIC archive in OPFS on first visit. The zip must be
//    served from the same origin as the page — GitHub Releases assets
//    are blocked by CORS. (~16 MB compressed, ~58 MB extracted.)
//    The binary format is tied to the Lindera version compiled into
//    the WASM, so stamp the cache with `version` and re-download when
//    it no longer matches the version your build expects.
const LINDERA_VERSION = "5.0.2"; // keep in sync with your Cargo.lock
if (
  !(await hasDictionary("ipadic"))
  || (await getDictionaryVersion("ipadic")) !== LINDERA_VERSION
) {
  await downloadDictionary("./dict/lindera-ipadic.zip", "ipadic", {
    version: LINDERA_VERSION,
    onProgress: ({ phase, loaded, total }) => console.log(phase, loaded, total),
  });
}

// 2. Read the nine component files back and construct the analyzer.
const f = await loadDictionaryFiles("ipadic");
const ja = JapaneseAnalyzer.fromBytes(
  f.metadata, f.dictTrie, f.dictValsIdx, f.dictVals,
  f.dictWordsIdx, f.dictWords, f.matrixMtx, f.charDef, f.unk,
  "normal",
);

// 3. Register the analyzer on the schema and reference it from text fields.
const schema = new Schema();
schema.addAnalyzer("ja-ipadic", ja);
schema.addTextField("title", undefined, undefined, undefined, undefined, "ja-ipadic");
schema.addTextField("body", undefined, undefined, undefined, undefined, "ja-ipadic");
schema.setDefaultFields(["title", "body"]);

const index = await Index.create(schema);

await index.putDocument("doc1", {
  title: "形態素解析",
  body: "Lindera は Rust 製の形態素解析ライブラリです。",
});
await index.commit();

const results = await index.search("形態素");
console.log(results[0].document.title); // "形態素解析"
```

See the [API Reference](api_reference.md#japaneseanalyzer) for the full
`JapaneseAnalyzer.fromBytes` signature and the OPFS helper API.

## Vector Search

```javascript
import init, { Index, Schema } from 'laurus-wasm';

await init();

const schema = new Schema();
schema.addTextField("title");
schema.addHnswField("embedding", 3); // 3-dimensional vectors

const index = await Index.create(schema);

await index.putDocument("doc1", {
  title: "Rust",
  embedding: [1.0, 0.0, 0.0]
});
await index.putDocument("doc2", {
  title: "Python",
  embedding: [0.0, 1.0, 0.0]
});
await index.commit();

// Search by vector similarity
const results = await index.searchVector("embedding", [0.9, 0.1, 0.0]);
console.log(results[0].document.title); // "Rust"
```

## Usage with Bundlers

### Vite

```javascript
// vite.config.js
import wasm from 'vite-plugin-wasm';

export default {
  plugins: [wasm()]
};
```

### Webpack 5

Webpack 5 supports WASM natively with `asyncWebAssembly`:

```javascript
// webpack.config.js
module.exports = {
  experiments: {
    asyncWebAssembly: true
  }
};
```
