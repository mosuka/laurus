# laurus-php

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PHP bindings for the [Laurus](https://github.com/mosuka/laurus) search engine. Provides lexical search, vector search, and hybrid search from PHP via a native Rust extension built with [ext-php-rs](https://github.com/davidcole1340/ext-php-rs).

## Features

- **Lexical Search** -- Full-text search powered by an inverted index with BM25 scoring
- **Vector Search** -- Approximate nearest neighbor (ANN) search using Flat, HNSW, or IVF indexes
- **Hybrid Search** -- Combine lexical and vector results with fusion algorithms (RRF, WeightedSum)
- **Rich Query DSL** -- Term, Phrase, Fuzzy, Wildcard, NumericRange, Geo, Boolean, Span queries
- **Text Analysis** -- Tokenizers, filters, and synonym expansion
- **Flexible Storage** -- In-memory (ephemeral) or file-based (persistent) indexes

## Requirements

- PHP 8.1 or later
- Rust toolchain (stable)

## Installation

Build the extension from source:

```bash
cd laurus-php
cargo build --release
```

Copy the shared library to your PHP extensions directory:

```bash
cp target/release/liblaurus_php.so $(php -r 'echo ini_get("extension_dir");')/laurus_php.so
```

Enable the extension by adding the following line to your `php.ini`:

```ini
extension=laurus_php.so
```

Verify the installation:

```bash
php -m | grep laurus
```

## Quick Start

```php
<?php
// extension=laurus_php.so in php.ini

use Laurus\Index;
use Laurus\TermQuery;

// Create an in-memory index
$index = new Index();

// Index documents
$index->putDocument("doc1", ["title" => "Introduction to Rust", "body" => "Systems programming language."]);
$index->putDocument("doc2", ["title" => "PHP for Web Development", "body" => "Web application development with PHP."]);
$index->commit();

// Search with a DSL string
$results = $index->search("title:rust", 5);
foreach ($results as $r) {
    $doc = $r->getDocument();
    printf("[%s] score=%.4f  %s\n", $r->getId(), $r->getScore(), $doc["title"]);
}

// Search with a query object
$results = $index->search(new TermQuery("body", "php"), 5);
```

## Index Types

### In-memory (ephemeral)

```php
$index = new Index();
```

### File-based (persistent)

```php
use Laurus\Index;
use Laurus\Schema;

$schema = new Schema();
$schema->addTextField("title");
$schema->addTextField("body");
$schema->addHnswField("embedding", 384);

$index = new Index("./myindex", $schema);
```

This writes `./myindex/schema.toml` and `./myindex/store/` -- the same
layout `laurus-cli create index --schema` uses, so the directory can be
opened by either. Reopening it later only needs the path (`$schema` must
be omitted, since it's loaded from the persisted `schema.toml`):

```php
$index = new Index("./myindex");
```

A file-based index holds an exclusive lock on its directory for as long as
the `Index` object is alive, so a second `new Index("./myindex")` on the
same path fails while the first is still open. Call `$index->close()` when
you are done with an index -- especially before reopening the same path --
rather than relying on PHP's refcounting/cycle-collecting garbage
collector, whose destruction timing is not guaranteed for every reference
pattern (e.g. reference cycles). `close()` is idempotent; every other
method throws after it has been called.

```php
$index->close();
```

To cheaply check whether another process committed since you last looked --
without opening (or reopening) the index at all -- use the namespaced
`Laurus\peek_commit_generation($path)`:

```php
$before = Laurus\peek_commit_generation($path);
// ... later ...
if (Laurus\peek_commit_generation($path) !== $before) {
    // something changed on disk; reopen the index to pick it up
}
```

It reads the persisted commit generation directly off disk, with no
`Engine` construction at all -- no storage lock, no WAL recovery, no
embedder loading -- so it works even before any `Index` for that path has
been created in this process. Throws a `ValueError` if `$path` isn't a
laurus index directory (no persisted schema).

## Schema

The `Schema` class defines the structure of your index. Use the following methods to add fields:

| Method | Description |
| :--- | :--- |
| `addTextField(name, stored, indexed, termVectors, docValues, analyzer, multiValued, positionIncrementGap)` | Full-text searchable text field (`analyzer` is a built-in name: `standard`, `english`, `keyword`, `simple`, `noop`) |
| `addIntegerField(name, stored, indexed, multiValued, docValues)` | Integer (i64) field |
| `addFloatField(name, stored, indexed, multiValued, docValues)` | Float (f64) field |
| `addBooleanField(name, stored, indexed, multiValued, docValues)` | Boolean field |
| `addDatetimeField(name, stored, indexed, multiValued, docValues)` | Date/time field |
| `addGeoField(name, stored, indexed, multiValued, docValues)` | Geographic coordinate field (lat/lon) |
| `addGeo3dField(name, stored, indexed, multiValued, docValues)` | 3D ECEF Cartesian point field (x, y, z in metres) |
| `addBytesField(name, stored)` | Binary data field |
| `addHnswField(name, dimension, distance, m, efConstruction, defaultEfSearch, embedder, quantizer, subvectorCount, rerankStorage, pqCodebookPath, baseWeight)` | HNSW vector index field |
| `addFlatField(name, dimension, distance, embedder, baseWeight)` | Flat (brute-force) vector index field |
| `addIvfField(name, dimension, distance, nClusters, nProbe, embedder, baseWeight)` | IVF vector index field |
| `addEmbedder(name, config)` | Register a named embedder |
| `setDefaultFields(fieldNames)` | Set default search fields |

Every parameter after `name` (and `dimension` for vector fields) has a default, so trailing arguments can be omitted; the arguments are positional, so keep the order above when passing a later one. Types and defaults are listed in the [API reference](../docs/src/laurus-php/api_reference.md).

## Query Types

| Query class | Description |
| :--- | :--- |
| `TermQuery(field, term)` | Exact term match |
| `PhraseQuery(field, terms)` | Ordered phrase match |
| `FuzzyQuery(field, term, maxEdits)` | Approximate term match |
| `WildcardQuery(field, pattern)` | Wildcard pattern match (`*`, `?`) |
| `NumericRangeQuery(field, min, max, numericType)` | Numeric range (integer or float) |
| `GeoDistanceQuery::withinRadius(field, lat, lon, distanceKm)` | Geo-distance radius search |
| `GeoBoundingBoxQuery::withinBoundingBox(field, minLat, minLon, maxLat, maxLon)` | Geo bounding box search |
| `BooleanQuery` | Compound boolean logic (must/should/mustNot) |
| `SpanQuery::near(field, terms, slop, ordered)` | Proximity / ordered span match |
| `VectorQuery(field, vector)` | Pre-computed vector similarity |
| `VectorTextQuery(field, text)` | Text-to-vector similarity (requires embedder) |

### Boolean Query

```php
use Laurus\BooleanQuery;
use Laurus\TermQuery;

$bq = new BooleanQuery();
$bq->must(new TermQuery("body", "rust"));
$bq->should(new TermQuery("title", "introduction"));
$bq->mustNot(new TermQuery("body", "deprecated"));

$results = $index->search($bq, 10);
```

### Geo Query

```php
use Laurus\GeoBoundingBoxQuery;
use Laurus\GeoDistanceQuery;

// Radius search
$results = $index->search(GeoDistanceQuery::withinRadius("location", 35.6895, 139.6917, 10.0), 10);

// Bounding box search
$results = $index->search(GeoBoundingBoxQuery::withinBoundingBox("location", 35.0, 139.0, 36.0, 140.0), 10);
```

## Hybrid Search

```php
use Laurus\SearchRequest;
use Laurus\TermQuery;
use Laurus\VectorQuery;
use Laurus\RRF;

$request = new SearchRequest(
    query: null,
    lexicalQuery: new TermQuery("body", "rust"),
    vectorQuery: new VectorQuery("embedding", $queryVec),
    filterQuery: null,
    fusion: new RRF(60.0),
    limit: 10,
    offset: 0,
);
$results = $index->search($request);
```

### Fusion Algorithms

| Class | Description |
| :--- | :--- |
| `RRF(k)` | Reciprocal Rank Fusion (rank-based, default for hybrid) |
| `WeightedSum(lexicalWeight, vectorWeight)` | Score-normalised weighted sum |

## Text Analysis

```php
use Laurus\SynonymDictionary;
use Laurus\WhitespaceTokenizer;
use Laurus\SynonymGraphFilter;

$synDict = new SynonymDictionary();
$synDict->addSynonymGroup(["ml", "machine learning"]);

$tokenizer = new WhitespaceTokenizer();
$filter = new SynonymGraphFilter($synDict, true, 0.8);

$tokens = $tokenizer->tokenize("ml tutorial");
$tokens = $filter->apply($tokens);
foreach ($tokens as $tok) {
    printf("%s position=%d boost=%.2f\n", $tok->getText(), $tok->getPosition(), $tok->getBoost());
}
```

## Document Operations

```php
// Put (replace) a document
$index->putDocument("doc1", ["title" => "Hello", "body" => "World"]);

// Add (append) a document version
$index->addDocument("doc1", ["title" => "Hello v2", "body" => "World v2"]);

// Retrieve all versions of a document
$docs = $index->getDocuments("doc1");

// Delete all versions of a document
$index->deleteDocuments("doc1");

// Commit changes to make them searchable
$index->commit();

// Get index statistics
$stats = $index->stats();
echo "Document count: " . $stats["documentCount"] . "\n";
```

## Durability / WAL

The write-ahead log (WAL) is fully durable by default: every record is
`fsync`ed before the write returns. To trade some durability for higher write
throughput, opt into **group commit** by passing a `WalSyncPolicy` to the
`Index` constructor. Group commit batches `fsync` calls, flushing when either
the record count (default 1024) or byte threshold (default 1 MiB) is reached,
and on every `commit()`. A crash can lose at most the last unsynced batch
(like SQLite's `synchronous = NORMAL`); the index never corrupts. Call
`flushWal()` to force a durable barrier on demand.

```php
use Laurus\Index;
use Laurus\WalSyncPolicy;

// Default: per-record fsync (omit the argument entirely).
$index = new Index();

// Opt into group commit, then force a flush when needed.
$policy = WalSyncPolicy::group(4096, 4 * 1024 * 1024);
$index = new Index("./myindex", null, $policy);

$index->putDocument("doc1", ["title" => "Hello"]);
$index->flushWal(); // persist now, without waiting for the batch to fill
```

## Feature Flags

Optional Cargo feature flags enable additional embedding backends:

| Feature flag | Description |
| :--- | :--- |
| `embeddings-candle` | Local BERT embeddings via [Candle](https://github.com/huggingface/candle) |
| `embeddings-multimodal` | Multimodal (CLIP) embeddings for text and image search |
| `embeddings-openai` | Cloud-based embeddings via the OpenAI API |
| `embeddings-all` | Enable all embedding backends |

Build with a feature flag:

```bash
cargo build --release --features embeddings-candle
```

## Documentation

- [PHP Binding Guide](https://mosuka.github.io/laurus/laurus-php.html)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
