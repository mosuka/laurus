/**
 * Integration tests for search-result highlighting (Issue #1134).
 *
 * Covers the `highlight` parameter on `Index.search` / `searchTerm` /
 * `searchBatch`, and the equivalent `highlight` field on
 * `SearchRequestOptions`:
 *
 * - Highlighted fragments are returned for the requested field.
 * - `highlights` stays empty when `highlight` is omitted.
 * - `HighlightOptions` knobs (`tag`, `maxFragments`, ...) take effect.
 * - `searchBatch` applies the same highlight settings to every query.
 * - `SearchRequest({ highlight })` produces the same highlights as the
 *   equivalent `search(..., highlight)` call.
 */

import { describe, it, expect } from "vitest";
import { Index, Schema, SearchRequest } from "../index.js";

async function createTextIndex() {
  const schema = new Schema();
  schema.addTextField("title");
  schema.addTextField("body");
  schema.setDefaultFields(["title", "body"]);
  const index = await Index.create(null, schema);
  await index.putDocument("doc1", {
    title: "Introduction to Rust",
    body: "Rust is a systems programming language.",
  });
  await index.commit();
  return index;
}

describe("highlighting", () => {
  it("returns fragments for the requested field", async () => {
    const index = await createTextIndex();
    const results = await index.search("body:rust", 10, 0, {
      fields: ["body"],
    });

    expect(results.length).toBe(1);
    const fragments = results[0].highlights.body;
    expect(fragments.length).toBe(1);
    expect(fragments[0]).toContain("<mark>Rust</mark>");
  });

  it("leaves highlights empty when not requested", async () => {
    const index = await createTextIndex();
    const results = await index.search("body:rust", 10, 0);

    expect(results.length).toBe(1);
    expect(results[0].highlights).toEqual({});
  });

  it("omits an unrequested field from highlights", async () => {
    const index = await createTextIndex();
    const results = await index.search("body:rust", 10, 0, {
      fields: ["body"],
    });

    expect(results[0].highlights.title).toBeUndefined();
  });

  it("applies a custom tag via HighlightOptions", async () => {
    const index = await createTextIndex();
    const results = await index.searchTerm("body", "rust", 10, 0, {
      fields: ["body"],
      tag: "em",
    });

    const fragment = results[0].highlights.body[0];
    expect(fragment).toContain("<em>Rust</em>");
    expect(fragment).not.toContain("<mark>");
  });

  it("applies the same highlight settings to every query in a batch", async () => {
    const index = await createTextIndex();
    const batch = await index.searchBatch(
      ["body:rust", "body:programming"],
      10,
      0,
      { fields: ["body"] },
    );

    expect(batch.length).toBe(2);
    for (const results of batch) {
      expect(results.length).toBe(1);
      expect(results[0].highlights.body[0]).toContain("<mark>");
    }
  });

  it("SearchRequest highlight matches the equivalent search() call", async () => {
    const index = await createTextIndex();
    const viaKwarg = await index.search("body:rust", 10, 0, {
      fields: ["body"],
    });
    const request = new SearchRequest({
      queryDsl: "body:rust",
      highlight: { fields: ["body"] },
    });
    const viaRequest = await index.searchWithRequest(request);

    expect(viaKwarg[0].highlights).toEqual(viaRequest[0].highlights);
  });
});
