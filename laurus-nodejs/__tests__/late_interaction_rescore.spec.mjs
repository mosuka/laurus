/**
 * Integration tests for multi-vector fields and the late-interaction
 * rescore (Issue #1351).
 *
 * Uses the same corpus as the Rust test
 * (`laurus/tests/late_interaction_rescore_test.rs`): against the query
 * token vectors [[1, 0], [0, 1]] the MaxSim scores are c 1.1, b 1.0,
 * d 0.9, a 0.1, e 0.05 — an order that matches neither the BM25 nor the
 * `vec` ranking — so the binding must rank exactly like the Rust API.
 */

import { describe, it, expect } from "vitest";
import { Index, RRF, Schema, SearchRequest, TermQuery, VectorQuery } from "../index.js";

const QUERY = [
  [1.0, 0.0],
  [0.0, 1.0],
];
const EXPECTED = ["c", "b", "d", "a", "e"];
const CORPUS = {
  a: ["rust", [1.0, 0.0], [[0.1, 0.0]]],
  b: ["rust rust", [0.9, 0.1], [[0.5, 0.5], [0.0, 0.2]]],
  c: ["rust language", [0.5, 0.5], [[0.9, 0.2]]],
  d: ["rust rust rust", [0.2, 0.8], [[0.3, 0.3], [0.6, 0.0]]],
  e: ["learning rust today", [0.0, 1.0], [[0.02, 0.03]]],
};

async function createIndex() {
  const schema = new Schema();
  schema.addTextField("title");
  schema.addFlatField("vec", 2);
  schema.addMultiVectorField("tokens", 2, "dot_product");
  const index = await Index.create(null, schema);
  for (const [id, [title, vec, tokens]] of Object.entries(CORPUS)) {
    await index.putDocument(id, { title, vec, tokens });
  }
  await index.commit();
  return index;
}

function maxSim(query, tokens) {
  return query.reduce(
    (sum, q) => sum + Math.max(...tokens.map((t) => q.reduce((s, x, i) => s + x * t[i], 0))),
    0,
  );
}

const ids = (results) => results.map((r) => r.id);
const rescore = (extra = {}) => ({ field: "tokens", vectors: QUERY, ...extra });

describe("late-interaction rescore", () => {
  it("reorders a DSL search by MaxSim", async () => {
    const index = await createIndex();
    expect(ids(await index.search("title:rust"))).not.toEqual(EXPECTED);

    const results = await index.search("title:rust", 10, 0, undefined, rescore());

    expect(ids(results)).toEqual(EXPECTED);
    for (const r of results) {
      expect(r.score).toBeCloseTo(maxSim(QUERY, CORPUS[r.id][2]), 5);
    }
  });

  it("reorders a SearchRequest, including a hybrid one", async () => {
    const index = await createIndex();
    const request = new SearchRequest({ queryDsl: "title:rust", rescore: rescore() });
    expect(ids(await index.searchWithRequest(request))).toEqual(EXPECTED);

    const hybrid = new SearchRequest({ rescore: rescore() });
    hybrid.setLexicalTerm(new TermQuery("title", "rust"));
    hybrid.setVectorQuery(new VectorQuery("vec", [1.0, 0.0]));
    hybrid.setRrfFusion(new RRF());
    expect(ids(await index.searchWithRequest(hybrid))).toEqual(EXPECTED);
  });

  it("reorders only the window", async () => {
    const index = await createIndex();
    const baseline = await index.search("title:rust");

    const results = await index.search("title:rust", 10, 0, undefined, rescore({ windowSize: 1 }));

    expect(ids(results)).toEqual(ids(baseline));
    expect(results[0].score).toBeCloseTo(maxSim(QUERY, CORPUS[results[0].id][2]), 5);
    for (let i = 1; i < results.length; i++) {
      expect(results[i].score).toBe(baseline[i].score);
    }
  });

  it("does not store token vectors", async () => {
    // Token vectors live only in the vector store, unlike a single vector.
    const index = await createIndex();
    const [doc] = await index.getDocuments("b");
    expect(doc).not.toHaveProperty("tokens");
    expect(doc.vec).toHaveLength(2);
  });

  it.each([
    [{ field: "tokens" }],
    [{ field: "tokens", vectors: QUERY, text: "rust" }],
  ])("needs exactly one of vectors or text: %j", async (invalid) => {
    const index = await createIndex();
    await expect(index.search("title:rust", 10, 0, undefined, invalid)).rejects.toThrow(
      expect.objectContaining({
        code: "InvalidArg",
        message: expect.stringContaining("rescore needs exactly one of vectors or text"),
      }),
    );
  });

  it.each([
    [{ field: "tokens", text: "rust" }, "has no token-level embedder"],
    [{ field: "vec", vectors: QUERY }, "needs a MultiVector field"],
    [{ field: "tokens", vectors: [] }, "between 1 and 1024 query vectors"],
    [{ field: "tokens", vectors: [[1.0, 0.0, 0.0]] }, "has dimension 3"],
    [rescore({ windowSize: 0 }), "window_size"],
  ])("rejects an invalid rescore: %j", async (invalid, message) => {
    const index = await createIndex();
    await expect(index.search("title:rust", 10, 0, undefined, invalid)).rejects.toThrow(
      expect.objectContaining({ code: "InvalidArg", message: expect.stringContaining(message) }),
    );
  });

  it("rejects ragged token vectors", async () => {
    const index = await createIndex();
    await expect(
      index.putDocument("x", { title: "rust", tokens: [[1.0, 0.0], [0.0]] }),
    ).rejects.toThrow("token vectors must share one dimension");
  });

  it("accepts integer token vectors", async () => {
    const index = await createIndex();
    await index.putDocument("x", { title: "rust", tokens: [[2, 0]] });
    await index.commit();
    const results = await index.search("title:rust", 10, 0, undefined, rescore());
    expect(results[0].id).toBe("x");
    expect(results[0].score).toBeCloseTo(2.0, 5);
  });
});

describe("addMultiVectorField", () => {
  it.each([
    [0, undefined, "dimension must be greater than 0"],
    [2, "euclidean", "distance must be Cosine or DotProduct"],
  ])("rejects dimension %s / distance %s", (dimension, distance, message) => {
    const schema = new Schema();
    expect(() => schema.addMultiVectorField("tokens", dimension, distance)).toThrow(
      expect.objectContaining({ code: "InvalidArg", message: expect.stringContaining(message) }),
    );
    expect(schema.fieldNames()).toEqual([]);
  });

  it.each([["f16"], ["int8"]])("accepts a compressed storage kind: %s", (storage) => {
    const schema = new Schema();
    expect(() =>
      schema.addMultiVectorField("tokens", 2, undefined, undefined, storage),
    ).not.toThrow();
    expect(schema.fieldNames()).toEqual(["tokens"]);
  });

  it("rejects an unknown storage kind", () => {
    const schema = new Schema();
    expect(() => schema.addMultiVectorField("tokens", 2, undefined, undefined, "bf16")).toThrow(
      expect.objectContaining({
        code: "GenericFailure",
        message: expect.stringContaining("Unknown multi-vector storage"),
      }),
    );
    expect(schema.fieldNames()).toEqual([]);
  });
});

describe("addEmbedder", () => {
  it("accepts every core embedder type", () => {
    const schema = new Schema();
    schema.addEmbedder("colbert", {
      type: "candle_colbert",
      model: "colbert-ir/colbertv2.0",
      revision: "main",
      query_maxlen: 16,
      doc_maxlen: 64,
    });
    schema.addMultiVectorField("tokens", 128, undefined, "colbert");

    const toml = Schema.fromToml(schema.toToml()).toToml();
    expect(toml).toContain('type = "candle_colbert"');
    expect(toml).toContain("query_maxlen = 16");
    expect(toml).toContain('embedder = "colbert"');
  });

  it.each([
    [{ type: "candle_bert" }, "missing field `model`"],
    [{ type: "nope" }, "unknown variant"],
    [{ model: "x" }, "missing field `type`"],
    ["candle_bert", "must be an object"],
  ])("rejects %j", (config, message) => {
    expect(() => new Schema().addEmbedder("e", config)).toThrow(message);
  });
});
