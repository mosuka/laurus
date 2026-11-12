/**
 * Integration tests for `Index.searchBatch` (Phase 3c of #648, issue #718).
 *
 * Covers:
 * - Empty input returns an empty list without invoking the engine.
 * - Single-query batch matches the single-query `search()` result.
 * - Multi-query batch preserves input order and returns one result list
 *   per input query.
 * - `limit` and `offset` apply per query in the batch.
 */

import { describe, it, expect } from "vitest";
import { Index, Schema } from "../index.js";

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
  await index.putDocument("doc3", {
    title: "Distributed Systems",
    body: "Engineering at scale.",
  });
  await index.commit();
  return index;
}

describe("searchBatch", () => {
  it("returns an empty array for empty input", async () => {
    const index = await createTextIndex();
    const results = await index.searchBatch([]);
    expect(results).toEqual([]);
  });

  it("single-query batch matches search()", async () => {
    const index = await createTextIndex();
    const serial = await index.search("title:rust", 5);
    const batch = await index.searchBatch(["title:rust"], 5);

    expect(batch.length).toBe(1);
    expect(batch[0].length).toBe(serial.length);
    for (let i = 0; i < serial.length; i++) {
      expect(batch[0][i].id).toBe(serial[i].id);
    }
  });

  it("multi-query batch preserves input order", async () => {
    const index = await createTextIndex();
    const queries = ["title:rust", "body:python", "title:distributed"];
    const expectedTopIds = ["doc1", "doc2", "doc3"];

    const batch = await index.searchBatch(queries, 5);
    expect(batch.length).toBe(queries.length);

    for (let i = 0; i < queries.length; i++) {
      expect(batch[i].length).toBeGreaterThanOrEqual(1);
      expect(batch[i][0].id).toBe(expectedTopIds[i]);
    }
  });

  it("returns an empty inner list for a no-match query", async () => {
    const index = await createTextIndex();
    const queries = ["title:rust", "title:nonexistent_xyz"];
    const batch = await index.searchBatch(queries, 5);

    expect(batch.length).toBe(2);
    expect(batch[0].length).toBeGreaterThanOrEqual(1);
    expect(batch[1]).toEqual([]);
  });

  it("applies limit per query", async () => {
    const index = await createTextIndex();
    const queries = [
      "body:programming OR body:data",
      "body:programming OR body:data",
    ];
    const batch = await index.searchBatch(queries, 1);

    expect(batch.length).toBe(2);
    for (const results of batch) {
      expect(results.length).toBeLessThanOrEqual(1);
    }
  });

  // Issue #1553: the engine wraps the error of a batch with the query's
  // position, which the binding unwraps, so a query the engine rejects
  // rejects the batch as it rejects `search()`.
  it("rejects a query the engine rejects as search does", async () => {
    const index = await createTextIndex();
    const single = await index.search("nosuch:rust").then(
      () => null,
      (e) => e,
    );
    const batch = await index.searchBatch(["title:rust", "nosuch:rust"]).then(
      () => null,
      (e) => e,
    );

    expect(single).not.toBeNull();
    expect(batch).not.toBeNull();
    expect(batch.code).toBe("InvalidArg");
    expect(batch.message).toContain("unknown field");
    expect(batch.message).toBe(single.message);
  });
});
