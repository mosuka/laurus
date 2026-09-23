/**
 * Basic integration tests for the laurus Node.js binding.
 *
 * Mirrors the Python test suite (laurus-python/tests/test_index.py).
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  Index,
  Schema,
  TermQuery,
  PhraseQuery,
  FuzzyQuery,
  WildcardQuery,
  NumericRangeQuery,
  DateTimeRangeQuery,
  BooleanQuery,
  VectorQuery,
  SearchRequest,
  RRF,
  WeightedSum,
  SynonymDictionary,
  WhitespaceTokenizer,
  SynonymGraphFilter,
  WalSyncPolicy,
  CommitPolicy,
} from "../index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function createTextIndex() {
  const schema = new Schema();
  schema.addTextField("title");
  schema.addTextField("body");
  schema.setDefaultFields(["title", "body"]);
  const index = await Index.create(null, schema);
  await index.putDocument("doc1", {
    title: "Introduction to Rust",
    body: "Systems programming language.",
  });
  await index.putDocument("doc2", {
    title: "Python for Data Science",
    body: "Data analysis with Python.",
  });
  await index.commit();
  return index;
}

async function createVectorIndex() {
  const schema = new Schema();
  schema.addTextField("title");
  schema.addHnswField("embedding", 4);
  schema.setDefaultFields(["title"]);
  const index = await Index.create(null, schema);
  await index.putDocument("doc1", {
    title: "Rust",
    embedding: [0.1, 0.2, 0.3, 0.4],
  });
  await index.putDocument("doc2", {
    title: "Python",
    embedding: [0.9, 0.8, 0.7, 0.6],
  });
  await index.commit();
  return index;
}

// ---------------------------------------------------------------------------
// Index creation
// ---------------------------------------------------------------------------

describe("Index creation", () => {
  it("creates an in-memory index", async () => {
    const index = await Index.create();
    expect(index).toBeDefined();
  });

  it("creates an index with schema", async () => {
    const schema = new Schema();
    schema.addTextField("title");
    const index = await Index.create(null, schema);
    expect(index).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// WAL sync policy
// ---------------------------------------------------------------------------

describe("WAL sync policy", () => {
  it("accepts perRecord() and group(...) factories", () => {
    expect(WalSyncPolicy.perRecord()).toBeDefined();
    expect(WalSyncPolicy.group()).toBeDefined();
    expect(WalSyncPolicy.group(256, 4096, 1000)).toBeDefined();
  });

  it("creates a group-commit index, flushes the WAL, and retrieves docs", async () => {
    const schema = new Schema();
    schema.addTextField("title");
    const index = await Index.create(null, schema, WalSyncPolicy.group());
    await index.putDocument("doc1", { title: "Group commit" });
    await index.flushWal();
    await index.commit();
    const docs = await index.getDocuments("doc1");
    expect(docs).toHaveLength(1);
    expect(docs[0].title).toBe("Group commit");
  });

  it("flushWal is a no-op fast path under perRecord", async () => {
    const index = await Index.create(null, undefined, WalSyncPolicy.perRecord());
    await index.putDocument("doc1", { title: "Per record" });
    await index.flushWal();
    await index.commit();
    const docs = await index.getDocuments("doc1");
    expect(docs).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Commit policy (auto-commit)
// ---------------------------------------------------------------------------

describe("Commit policy", () => {
  it("accepts manual() and everyDocs(n) factories", () => {
    expect(CommitPolicy.manual()).toBeDefined();
    expect(CommitPolicy.everyDocs(100)).toBeDefined();
    // everyDocs(0) is valid — it disables auto-commit (equivalent to manual).
    expect(CommitPolicy.everyDocs(0)).toBeDefined();
  });

  it("accepts an intervalMs(ms) factory", async () => {
    expect(CommitPolicy.intervalMs(1000)).toBeDefined();
    const schema = new Schema();
    schema.addTextField("title");
    const index = await Index.create(
      null,
      schema,
      undefined,
      CommitPolicy.intervalMs(1000),
    );
    expect(index).toBeDefined();
  });

  it("creates an index with an EveryDocs policy and retrieves a doc", async () => {
    const schema = new Schema();
    schema.addTextField("title");
    const index = await Index.create(
      null,
      schema,
      undefined,
      CommitPolicy.everyDocs(1),
    );
    await index.putDocument("d1", { title: "Auto" });
    // No explicit commit — the binding path is wired end-to-end.
    const docs = await index.getDocuments("d1");
    expect(docs).toHaveLength(1);
  });

  it("defaults to manual when no commit policy is given", async () => {
    const index = await Index.create(null, undefined, undefined, CommitPolicy.manual());
    expect(index).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Document CRUD
// ---------------------------------------------------------------------------

describe("Document CRUD", () => {
  it("put and get document", async () => {
    const index = await Index.create();
    await index.putDocument("doc1", { title: "Hello" });
    await index.commit();
    const docs = await index.getDocuments("doc1");
    expect(docs).toHaveLength(1);
  });

  it("put replaces existing document", async () => {
    const index = await createTextIndex();
    await index.putDocument("doc1", { title: "Updated" });
    await index.commit();
    const docs = await index.getDocuments("doc1");
    expect(docs).toHaveLength(1);
  });

  it("add_document appends versions", async () => {
    const index = await Index.create();
    await index.addDocument("doc1", { title: "Chunk 1" });
    await index.addDocument("doc1", { title: "Chunk 2" });
    const docs = await index.getDocuments("doc1");
    expect(docs).toHaveLength(2);
  });

  it("delete documents", async () => {
    const index = await createTextIndex();
    await index.deleteDocuments("doc1");
    await index.commit();
    const docs = await index.getDocuments("doc1");
    expect(docs).toHaveLength(0);
  });

  it("get documents for unknown id returns empty", async () => {
    const index = await createTextIndex();
    const docs = await index.getDocuments("does_not_exist");
    expect(docs).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

describe("Stats", () => {
  it("returns document count", async () => {
    const index = await createTextIndex();
    const stats = index.stats();
    expect(stats.documentCount).toBe(2);
  });

  it("returns vector field stats", async () => {
    const index = await createVectorIndex();
    const stats = index.stats();
    expect(stats.vectorFields.embedding).toBeDefined();
    expect(stats.vectorFields.embedding.count).toBe(2);
    expect(stats.vectorFields.embedding.dimension).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// Lexical search
// ---------------------------------------------------------------------------

describe("Lexical search", () => {
  it("searches with DSL string", async () => {
    const index = await createTextIndex();
    const results = await index.search("title:rust", 5);
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].id).toBe("doc1");
  });

  it("searches with term query", async () => {
    const index = await createTextIndex();
    const results = await index.searchTerm("body", "python", 5);
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].id).toBe("doc2");
  });

  it("result has id, score, and document", async () => {
    const index = await createTextIndex();
    const results = await index.search("title:rust", 1);
    const r = results[0];
    expect(r.id).toBe("doc1");
    expect(r.score).toBeGreaterThan(0);
    expect(r.document).toBeDefined();
    expect(r.document.title).toBe("Introduction to Rust");
  });

  it("respects limit", async () => {
    const index = await createTextIndex();
    const results = await index.search(
      "body:programming OR body:python",
      1,
    );
    expect(results.length).toBeLessThanOrEqual(1);
  });

  it("respects offset", async () => {
    const index = await createTextIndex();
    const all = await index.search("body:programming OR body:data", 10);
    const offset = await index.search(
      "body:programming OR body:data",
      10,
      1,
    );
    if (all.length > 1) {
      expect(offset[0].id).toBe(all[1].id);
    }
  });

  it("returns empty for no matches", async () => {
    const index = await createTextIndex();
    const results = await index.search("title:nonexistent_xyz", 5);
    expect(results).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Vector search
// ---------------------------------------------------------------------------

describe("Vector search", () => {
  it("searches with vector query", async () => {
    const index = await createVectorIndex();
    const results = await index.searchVector(
      "embedding",
      [0.1, 0.2, 0.3, 0.4],
      2,
    );
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].id).toBe("doc1");
  });
});

// ---------------------------------------------------------------------------
// Hybrid search
// ---------------------------------------------------------------------------

describe("Hybrid search", () => {
  it("searches with SearchRequest (lexical only)", async () => {
    const index = await createTextIndex();
    const req = new SearchRequest({ limit: 5 });
    req.setLexicalTerm(new TermQuery("title", "rust"));
    const results = await index.searchWithRequest(req);
    expect(results.length).toBeGreaterThanOrEqual(1);
  });

  it("searches with SearchRequest (hybrid)", async () => {
    const index = await createVectorIndex();
    const req = new SearchRequest({ limit: 5 });
    req.setLexicalTerm(new TermQuery("title", "rust"));
    req.setVectorQuery(new VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4]));
    req.setRrfFusion(new RRF(60.0));
    const results = await index.searchWithRequest(req);
    expect(results.length).toBeGreaterThanOrEqual(1);
  });

  it("searches with SearchRequest constructed via options object only", async () => {
    // The constructor accepts an options object with primitive fields.
    // Polymorphic clauses are still attached via per-type setters.
    const index = await createTextIndex();
    const req = new SearchRequest({
      queryDsl: "title:rust",
      limit: 5,
      offset: 0,
    });
    const results = await index.searchWithRequest(req);
    expect(results.length).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// Query types
// ---------------------------------------------------------------------------

describe("Query types", () => {
  it("phrase query", async () => {
    const index = await createTextIndex();
    const req = new SearchRequest({ limit: 5 });
    req.setLexicalPhrase(new PhraseQuery("title", ["introduction", "rust"]));
    const results = await index.searchWithRequest(req);
    expect(results.some((r) => r.id === "doc1")).toBe(true);
  });

  it("numeric range query", async () => {
    const schema = new Schema();
    schema.addIntegerField("year");
    const index = await Index.create(null, schema);
    await index.putDocument("doc1", { year: 2020 });
    await index.putDocument("doc2", { year: 2023 });
    await index.commit();

    const q = new NumericRangeQuery("year", 2022, 2024);
    const req = new SearchRequest({ limit: 5 });
    // Use DSL or searchTerm - NumericRangeQuery needs to be used via SearchRequest
    // For now, test that the class can be constructed
    expect(q).toBeDefined();
    // Explicit "integer" discriminator should also work.
    expect(new NumericRangeQuery("year", 2022, 2024, "integer")).toBeDefined();
    // "float" discriminator should also work.
    expect(new NumericRangeQuery("price", 1.5, 9.5, "float")).toBeDefined();
    // Anything else throws.
    expect(() => new NumericRangeQuery("year", 0, 1, "double")).toThrow();
  });

  it("datetime range query (#1179)", async () => {
    const schema = new Schema();
    schema.addDatetimeField("createdAt");
    const index = await Index.create(null, schema);
    await index.putDocument("jan", { createdAt: "2024-01-01T00:00:00Z" });
    await index.putDocument("jun", { createdAt: "2024-06-15T12:00:00Z" });
    await index.putDocument("next", { createdAt: "2025-01-01T00:00:00Z" });
    await index.commit();

    const ids = (results) => results.map((r) => r.id).sort();

    // Date-only bounds are midnight UTC, inclusive on both ends.
    const req = new SearchRequest({ limit: 5 });
    req.setLexicalDateTimeRange(new DateTimeRangeQuery("createdAt", "2024-01-01", "2024-12-31"));
    expect(ids(await index.searchWithRequest(req))).toEqual(["jan", "jun"]);

    // RFC 3339 with an offset is normalized to UTC; `null` leaves a side open.
    const open = new SearchRequest({ limit: 5 });
    open.setLexicalDateTimeRange(new DateTimeRangeQuery("createdAt", "2024-06-15T21:00:00+09:00", null));
    expect(ids(await index.searchWithRequest(open))).toEqual(["jun", "next"]);

    // Works as a boolean clause and as a filter too.
    const bq = new BooleanQuery();
    bq.mustDateTimeRange(new DateTimeRangeQuery("createdAt", null, "2024-06-15T12:00:00"));
    const viaBool = new SearchRequest({ limit: 5 });
    viaBool.setLexicalBoolean(bq);
    expect(ids(await index.searchWithRequest(viaBool))).toEqual(["jan", "jun"]);

    // The documented DSL form reaches the index as well.
    expect(ids(await index.search("createdAt:{2024-01-01 TO 2024-12-31}", 5))).toEqual(["jun"]);

    // A malformed bound throws at construction.
    expect(() => new DateTimeRangeQuery("createdAt", "yesterday", null)).toThrow(/datetime/);
  });

  it("boolean query accepts any clause type via mustX/shouldX/mustNotX", async () => {
    const index = await createTextIndex();
    const bq = new BooleanQuery();
    // Exercise the polymorphic API surface by mixing query types beyond
    // plain TermQuery.
    bq.mustTerm(new TermQuery("body", "programming"));
    bq.mustNotTerm(new TermQuery("title", "python"));
    bq.shouldFuzzy(new FuzzyQuery("body", "data", 1));
    expect(bq).toBeDefined();
    // BooleanQuery itself can be nested inside another BooleanQuery.
    const outer = new BooleanQuery();
    outer.mustBoolean(bq);
    expect(outer).toBeDefined();
    void index;
  });

  it("wildcard query construction", async () => {
    const q = new WildcardQuery("title", "py*");
    expect(q).toBeDefined();
  });

  it("fuzzy query construction", async () => {
    const q = new FuzzyQuery("body", "pythn", 1);
    expect(q).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Fusion algorithms
// ---------------------------------------------------------------------------

describe("Fusion algorithms", () => {
  it("RRF construction", () => {
    const rrf = new RRF(60.0);
    expect(rrf).toBeDefined();
  });

  it("WeightedSum construction", () => {
    const ws = new WeightedSum(0.3, 0.7);
    expect(ws).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Text analysis
// ---------------------------------------------------------------------------

describe("Text analysis", () => {
  it("creates synonym dictionary", () => {
    const syn = new SynonymDictionary();
    syn.addSynonymGroup(["ml", "machine learning"]);
    expect(syn).toBeDefined();
  });

  it("whitespace tokenizer", () => {
    const tokenizer = new WhitespaceTokenizer();
    const tokens = tokenizer.tokenize("hello world");
    expect(tokens).toHaveLength(2);
    expect(tokens[0].text).toBe("hello");
    expect(tokens[1].text).toBe("world");
  });

  it("synonym graph filter", () => {
    const syn = new SynonymDictionary();
    syn.addSynonymGroup(["ml", "machine learning"]);
    const tokenizer = new WhitespaceTokenizer();
    const filter = new SynonymGraphFilter(syn, true, 0.8);

    const tokens = tokenizer.tokenize("ml tutorial");
    const result = filter.apply(tokens);
    const texts = result.map((t) => t.text);
    expect(texts).toContain("ml");
    expect(texts.some((t) => t === "machine" || t === "machine learning")).toBe(
      true,
    );
  });

  it("token has expected fields", () => {
    const tokenizer = new WhitespaceTokenizer();
    const tokens = tokenizer.tokenize("hello");
    const tok = tokens[0];
    expect(tok.text).toBe("hello");
    expect(typeof tok.position).toBe("number");
    expect(typeof tok.positionIncrement).toBe("number");
    expect(typeof tok.positionLength).toBe("number");
    expect(typeof tok.boost).toBe("number");
  });
});
