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
 * - A document that cannot be converted, or a malformed pair, rejects with
 *   `InvalidArg` naming its position (#1630).
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

  // Awaits `call` inside `try`, so an error thrown before a promise is
  // returned is caught as well as a rejection.
  async function errorOf(call) {
    try {
      await call();
      return null;
    } catch (e) {
      return e;
    }
  }

  // Issue #1630: a document whose content cannot be converted rejects with
  // `InvalidArg` (it was `GenericFailure`), alone and in a batch, where the
  // message starts with the document's position.
  for (const method of ["putDocuments", "addDocuments"]) {
    it(`${method} names a document that cannot be converted`, async () => {
      const index = await createTextIndex();
      const single = await errorOf(() => index.putDocument("b", { x: { a: 1 } }));
      const batch = await errorOf(() =>
        index[method]([
          ["a", { title: "a" }],
          ["b", { x: { a: 1 } }],
        ]),
      );

      expect(single).not.toBeNull();
      expect(single.code).toBe("InvalidArg");
      expect(batch).not.toBeNull();
      expect(batch.code).toBe("InvalidArg");
      expect(batch.message).toBe(`documents[1]: ${single.message}`);
    });
  }

  // Issue #1630: each pair is checked, and a malformed one is named by its
  // position (napi named none, and took an array of three or more).
  const malformed = [
    ["not a pair", "x", "documents[1]: expected an [id, doc] pair"],
    ["one element", ["b"], "documents[1]: expected a 2-element [id, doc] pair, got 1 elements"],
    ["three elements", ["b", {}, 1], "documents[1]: expected a 2-element [id, doc] pair, got 3 elements"],
    ["an id that is not a string", [1, {}], "documents[1]: id must be a string"],
    ["a document that is not an object", ["b", "x"], "documents[1]: Document must be a JSON object"],
  ];
  for (const [label, entry, message] of malformed) {
    it(`putDocuments rejects ${label}, naming it`, async () => {
      const index = await createTextIndex();
      const error = await errorOf(() => index.putDocuments([["a", { title: "a" }], entry]));

      expect(error).not.toBeNull();
      expect(error.code).toBe("InvalidArg");
      expect(error.message).toBe(message);
    });
  }
});
