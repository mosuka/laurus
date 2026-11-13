/**
 * Integration tests for `Index.putDocuments` / `Index.addDocuments` (#866).
 *
 * Covers:
 * - Empty batch is a no-op.
 * - `putDocuments` applies the batch and deduplicates duplicate ids within
 *   one batch (last occurrence wins).
 * - `addDocuments` accumulates repeated ids as separate versions.
 * - A document the engine rejects fails the batch with the code it gets
 *   alone (#1627).
 */

import { describe, it, expect } from "vitest";
import { Index, Schema } from "../index.js";

async function createTextIndex() {
  const schema = new Schema();
  schema.addTextField("title");
  schema.setDefaultFields(["title"]);
  return await Index.create(null, schema);
}

describe("putDocuments / addDocuments", () => {
  it("empty batch is a no-op", async () => {
    const index = await createTextIndex();
    await index.putDocuments([]);
    await index.addDocuments([]);
    await index.commit();
    expect(index.stats().documentCount).toBe(0);
  });

  it("putDocuments applies the batch and dedups duplicate ids", async () => {
    const index = await createTextIndex();
    await index.putDocuments([
      ["doc1", { title: "One" }],
      ["doc2", { title: "Two" }],
      ["doc1", { title: "One v2" }],
    ]);
    await index.commit();

    expect(index.stats().documentCount).toBe(2);
    const docs = await index.getDocuments("doc1");
    expect(docs.length).toBe(1);
    expect(docs[0].title).toBe("One v2");
  });

  it("addDocuments accumulates repeated ids as chunks", async () => {
    const index = await createTextIndex();
    await index.addDocuments([
      ["doc", { title: "chunk 0" }],
      ["doc", { title: "chunk 1" }],
    ]);
    await index.commit();

    const docs = await index.getDocuments("doc");
    expect(docs.length).toBe(2);
  });

  // Issue #1627: a document the engine rejects fails the batch with the
  // code the same document gets alone (it was `GenericFailure`), the
  // message naming its position and id.
  for (const method of ["putDocuments", "addDocuments"]) {
    it(`${method} rejects with the code of the failing document`, async () => {
      const index = await createTextIndex();
      const single = await index.putDocument("b", { _x: "v" }).then(
        () => null,
        (e) => e,
      );
      const batch = await index[method]([
        ["a", { title: "a" }],
        ["b", { _x: "v" }],
      ]).then(
        () => null,
        (e) => e,
      );

      expect(single).not.toBeNull();
      expect(single.code).toBe("InvalidArg");
      expect(batch).not.toBeNull();
      expect(batch.code).toBe(single.code);
      expect(batch.message).toContain("doc 1 (id 'b')");
      expect(batch.message.endsWith(single.message)).toBe(true);
    });
  }
});
