/**
 * Integration tests for result grouping (`groupBy`, Issue #1297).
 *
 * Uses the corpus of `laurus-server/tests/grpc_group_by_test.rs`: three
 * documents added as chunks sharing their `_id`, with Flat vectors so the
 * vector ranking against [1, 0] — and with it the groups — is exact.
 */

import { describe, it, expect } from "vitest";
import { Index, Schema, SearchRequest, VectorQuery } from "../index.js";

const CHUNKS = [
  ["a", "rust rust rust guide", [1.0, 0.0]],
  ["a", "rust rust intro", [0.9, 0.1]],
  ["a", "rust appendix notes", [0.3, 0.7]],
  ["b", "rust rust book chapter", [0.8, 0.2]],
  ["b", "rust history of programming", [0.2, 0.8]],
  ["c", "rust", [0.6, 0.4]],
];

// Each group as [id, titles of its hits, best first] with groupSize 2.
const EXPECTED = [
  ["a", ["rust rust rust guide", "rust rust intro"]],
  ["b", ["rust rust book chapter", "rust history of programming"]],
  ["c", ["rust"]],
];

async function createIndex() {
  const schema = new Schema();
  schema.addTextField("title");
  schema.addFlatField("vec", 2);
  const index = await Index.create(null, schema);
  for (const [id, title, vec] of CHUNKS) {
    await index.addDocument(id, { title, vec });
  }
  await index.commit();
  return index;
}

function groups(results) {
  return results.map((r) => {
    expect(r.groupKey).toBe(r.id);
    for (const hit of r.innerHits) {
      expect(hit.groupKey).toBe(r.groupKey);
      expect(hit.innerHits).toEqual([]);
    }
    return [r.id, [r, ...r.innerHits].map((h) => h.document.title)];
  });
}

// Group sizes per id, for searches whose BM25 order the test does not fix.
const innerCounts = (results) =>
  Object.fromEntries(results.map((r) => [r.id, r.innerHits.length]));

function vectorRequest(groupBy) {
  const request = new SearchRequest({ groupBy });
  request.setVectorQuery(new VectorQuery("vec", [1.0, 0.0]));
  return request;
}

describe("result grouping", () => {
  it("groups a SearchRequest's chunks by _id", async () => {
    const index = await createIndex();
    const results = await index.searchWithRequest(vectorRequest({ field: "_id", groupSize: 2 }));
    expect(groups(results)).toEqual(EXPECTED);

    const oneEach = await index.searchWithRequest(vectorRequest({ field: "_id" }));
    expect(oneEach.map((r) => r.id)).toEqual(["a", "b", "c"]);
    expect(oneEach.every((r) => r.innerHits.length === 0)).toBe(true);
  });

  it("groups a DSL search", async () => {
    const index = await createIndex();
    const results = await index.search("title:rust", 10, 0, undefined, undefined, {
      field: "_id",
      groupSize: 2,
    });
    // BM25 decides the order here; check the groups' shape and sizes.
    groups(results);
    expect(innerCounts(results)).toEqual({ a: 1, b: 1, c: 0 });
  });

  it("leaves an ungrouped search without groups", async () => {
    const index = await createIndex();
    const results = await index.searchWithRequest(vectorRequest(undefined));
    expect(results).toHaveLength(CHUNKS.length);
    for (const r of results) {
      expect(r.groupKey).toBeUndefined();
      expect(r.innerHits).toEqual([]);
    }
  });

  it("groups every query of a batch", async () => {
    const index = await createIndex();
    const batch = await index.searchBatch(["title:rust", "title:rust"], 10, 0, undefined, {
      field: "_id",
      groupSize: 2,
    });
    expect(batch.map(innerCounts)).toEqual([
      { a: 1, b: 1, c: 0 },
      { a: 1, b: 1, c: 0 },
    ]);
  });

  it("rejects a group_by the engine cannot use", async () => {
    const index = await createIndex();
    for (const groupBy of [{ field: "missing" }, { field: "vec" }, { field: "_id", groupSize: 0 }]) {
      await expect(index.searchWithRequest(vectorRequest(groupBy))).rejects.toThrow(/group_by/);
    }
  });
});
