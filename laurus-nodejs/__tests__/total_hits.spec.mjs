/**
 * The total hit count of a search (Issue #1472).
 *
 * `search`, `searchBatch` and `new SearchRequest({ trackTotalHits })` take
 * `trackTotalHits`, read as the HTTP gateway reads `"track_total_hits"`: an
 * integer counts exactly up to it, `true` counts every match, and `false`,
 * `null` or `undefined` counts nothing. Every search method resolves to a
 * `{ results, totalHits? }` response, `totalHits` being
 * `{ value, relation: "eq" | "gte" }`.
 */

import { describe, it, expect } from "vitest";
import { Index, Schema, SearchRequest } from "../index.js";

const MESSAGE = "trackTotalHits must be true, false or an integer from 0 to 4294967295, got ";

/** Five documents match `body:rust`, one matches `body:python`. */
async function createIndex() {
  const schema = new Schema();
  schema.addTextField("body");
  schema.addFlatField("vec", 2);
  const index = await Index.create(null, schema);
  for (let i = 0; i < 5; i++) {
    await index.putDocument(`rust${i}`, { body: `rust ${i}`, vec: [1.0, i] });
  }
  await index.putDocument("python", { body: "python", vec: [0.0, 1.0] });
  await index.commit();
  return index;
}

describe("search's trackTotalHits", () => {
  it.each([
    ["above", 1000, { value: 5, relation: "eq" }],
    ["below", 3, { value: 3, relation: "gte" }],
    ["true", true, { value: 5, relation: "eq" }],
  ])("counts the matches when asked (%s)", async (_label, track, expected) => {
    const index = await createIndex();
    const response = await index.search("body:rust", 2, 0, null, null, null, track);
    expect(response.results).toHaveLength(2);
    expect(response.totalHits).toEqual(expected);
  });

  it.each([
    ["false", false],
    ["null", null],
    ["undefined", undefined],
  ])("counts nothing for %s", async (_label, track) => {
    const index = await createIndex();
    const response = await index.search("body:rust", 2, 0, null, null, null, track);
    expect(response.results).toHaveLength(2);
    expect(response).not.toHaveProperty("totalHits");
  });

  it("counts nothing when left out", async () => {
    const index = await createIndex();
    const response = await index.search("body:rust", 2);
    expect(response.results).toHaveLength(2);
    expect(response).not.toHaveProperty("totalHits");
  });

  it("rejects 0 through the engine", async () => {
    const index = await createIndex();
    await expect(index.search("body:rust", 2, 0, null, null, null, 0)).rejects.toThrow(
      "track_total_hits must count at least 1 match",
    );
  });

  it.each([
    ["a string", "1000", '"1000"'],
    ["a fraction", 1.5, "1.5"],
    ["a negative number", -1, "-1"],
    ["a number too large", 4294967296, "4294967296"],
    ["an object", { upTo: 3 }, '{"upTo":3}'],
  ])("rejects %s, naming the parameter", async (_label, track, shown) => {
    const index = await createIndex();
    await expect(index.search("body:rust", 2, 0, null, null, null, track)).rejects.toThrow(
      `Index.search: ${MESSAGE}${shown}`,
    );
  });

  it("rejects NaN rather than counting nothing", async () => {
    const index = await createIndex();
    // napi-rs cannot hold NaN as JSON and throws when it converts the
    // argument, before the call returns a promise.
    const call = async () => index.search("body:rust", 2, 0, null, null, null, Number.NaN);
    await expect(call()).rejects.toThrow("Failed to convert js number to serde_json::Number");
  });
});

describe("SearchRequest's trackTotalHits", () => {
  it("is carried by searchWithRequest", async () => {
    const index = await createIndex();
    const request = new SearchRequest({ queryDsl: "body:rust", limit: 2, trackTotalHits: 3 });
    const response = await index.searchWithRequest(request);
    expect(response.results).toHaveLength(2);
    expect(response.totalHits).toEqual({ value: 3, relation: "gte" });
  });

  it("counts nothing by default", async () => {
    const index = await createIndex();
    const response = await index.searchWithRequest(new SearchRequest({ queryDsl: "body:rust" }));
    expect(response).not.toHaveProperty("totalHits");
  });

  it("is checked by the constructor", () => {
    expect(() => new SearchRequest({ queryDsl: "body:rust", trackTotalHits: "1000" })).toThrow(
      `SearchRequest: ${MESSAGE}"1000"`,
    );
  });
});

describe("searchBatch's trackTotalHits", () => {
  it("returns each query's total", async () => {
    const index = await createIndex();
    const responses = await index.searchBatch(
      ["body:rust", "body:python", "body:missing"],
      2,
      0,
      null,
      null,
      1000,
    );
    expect(responses.map((r) => r.totalHits)).toEqual([
      { value: 5, relation: "eq" },
      { value: 1, relation: "eq" },
      { value: 0, relation: "eq" },
    ]);
    expect(responses.map((r) => r.results.length)).toEqual([2, 1, 0]);
  });

  it("counts nothing by default", async () => {
    const index = await createIndex();
    const responses = await index.searchBatch(["body:rust"], 2);
    expect(responses[0]).not.toHaveProperty("totalHits");
  });

  it("rejects another value, even for an empty batch", async () => {
    const index = await createIndex();
    await expect(index.searchBatch([], 2, 0, null, null, "1000")).rejects.toThrow(
      `Index.searchBatch: ${MESSAGE}"1000"`,
    );
  });
});

describe("the searches for one kind of query", () => {
  // searchVectorText needs an embedder; it converts its response with the
  // function the others share.
  it.each([
    ["searchTerm", (index) => index.searchTerm("body", "rust", 2)],
    ["searchVector", (index) => index.searchVector("vec", [1.0, 0.0], 2)],
  ])("%s resolves to a response without a total", async (_name, search) => {
    const index = await createIndex();
    const response = await search(index);
    expect(response.results).toHaveLength(2);
    expect(response).not.toHaveProperty("totalHits");
  });
});
