// Tests for the HNSW quantizer / rerankStorage schema options (Issue #797).
//
// These assert the values configured on `addHnswField` actually reach the
// Rust core via deterministic observables, not merely that search succeeds:
//
// * `rerankStorage: "f32"` makes the core write a `*.hnsw.f32` Stage-2
//   sidecar on disk (mirrors the server-side guard in #793/#800); the
//   default writes no sidecar.
// * `quantizer: "product_quantization"` forwards `subvectorCount` to the
//   core, which rejects a count that does not divide the field dimension
//   when the index is built (Issue #1491).
// * `quantizer: "float16"` (Issue #623) separates two vectors that the
//   default int8 quantizer stores as the same code.

import { describe, it, expect } from "vitest";
import { mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Index, Schema } from "../index.js";

// `addHnswField` is positional:
//   (name, dimension, distance, m, efConstruction, defaultEfSearch,
//    embedder, quantizer, subvectorCount, rerankStorage)
function findSidecars(dir) {
  let found = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) found = found.concat(findSidecars(p));
    else if (entry.name.endsWith(".hnsw.f32")) found.push(p);
  }
  return found;
}

describe("HNSW quantizer / rerankStorage options (#797)", () => {
  it("rerankStorage 'f32' writes a .hnsw.f32 sidecar on disk", async () => {
    const dir = mkdtempSync(join(tmpdir(), "laurus-rerank-"));
    const schema = new Schema();
    schema.addHnswField(
      "embedding", 4, undefined, undefined, undefined, undefined, undefined, undefined, undefined, "f32",
    );
    const index = await Index.create(dir, schema);
    await index.putDocument("doc1", { embedding: [0.1, 0.2, 0.3, 0.4] });
    await index.putDocument("doc2", { embedding: [0.9, 0.8, 0.7, 0.6] });
    await index.commit();
    expect(findSidecars(dir).length).toBeGreaterThan(0);
  });

  it("no rerankStorage writes no sidecar", async () => {
    const dir = mkdtempSync(join(tmpdir(), "laurus-norerank-"));
    const schema = new Schema();
    schema.addHnswField("embedding", 4);
    const index = await Index.create(dir, schema);
    await index.putDocument("doc1", { embedding: [0.1, 0.2, 0.3, 0.4] });
    await index.putDocument("doc2", { embedding: [0.9, 0.8, 0.7, 0.6] });
    await index.commit();
    expect(findSidecars(dir).length).toBe(0);
  });

  it("product_quantization builds and searches the near cluster", async () => {
    const schema = new Schema();
    // PQ is an L2 quantizer, so use Euclidean (matching the core's
    // `test_hnsw_pq_search_returns_corpus_neighbour`).
    schema.addHnswField(
      "embedding", 4, "euclidean", undefined, undefined, undefined, undefined, "product_quantization", 2, undefined,
    );
    const index = await Index.create(null, schema);
    // Stable two-cluster corpus mirroring the core's
    // `test_hnsw_pq_search_returns_corpus_neighbour` (issue #730), sized to
    // 128 points per cluster (256 total) so the segment meets the PQ
    // min-train threshold (#880: smaller PQ-configured segments are written
    // as Scalar8Bit and would not exercise PQ training at all).
    const offset = (i) => [
      (i % 8) * 0.04 - 0.14,
      (Math.floor(i / 8) % 8) * 0.04 - 0.14,
      (Math.floor(i / 64) % 8) * 0.04 - 0.14,
      (i % 16) * 0.04 - 0.32,
    ];
    const nearBase = [10.0, 10.0, 20.0, 20.0];
    const farBase = [-100.0, -100.0, -200.0, -200.0];
    for (let i = 0; i < 128; i++) {
      const off = offset(i);
      await index.putDocument(`near${i}`, {
        embedding: nearBase.map((b, j) => b + off[j]),
      });
      await index.putDocument(`far${i}`, {
        embedding: farBase.map((b, j) => b + off[j]),
      });
    }
    await index.commit();

    const results = await index.searchVector("embedding", nearBase, 3);
    expect(results.length).toBe(3);
    expect(results.every((r) => r.id.startsWith("near"))).toBe(true);
  });

  it("rejects product_quantization whose subvectorCount does not divide the dimension", async () => {
    const schema = new Schema();
    schema.addHnswField(
      "embedding", 4, undefined, undefined, undefined, undefined, undefined, "product_quantization", 3, undefined,
    );
    await expect(Index.create(null, schema)).rejects.toThrow(
      /subvector_count 3 must divide dimension 4/,
    );
  });

  it("rejects an unknown quantizer", () => {
    const schema = new Schema();
    expect(() =>
      schema.addHnswField(
        "embedding", 4, undefined, undefined, undefined, undefined, undefined, "bogus",
      ),
    ).toThrow();
  });

  it("rejects product_quantization without subvectorCount", () => {
    const schema = new Schema();
    expect(() =>
      schema.addHnswField(
        "embedding", 4, undefined, undefined, undefined, undefined, undefined, "product_quantization",
      ),
    ).toThrow();
  });

  it("rejects subvectorCount for a scalar quantizer", () => {
    const schema = new Schema();
    expect(() =>
      schema.addHnswField(
        "embedding", 4, undefined, undefined, undefined, undefined, undefined, undefined, 2,
      ),
    ).toThrow();
  });

  it("rejects an unknown rerankStorage", () => {
    const schema = new Schema();
    expect(() =>
      schema.addHnswField(
        "embedding", 4, undefined, undefined, undefined, undefined, undefined, undefined, undefined, "bogus",
      ),
    ).toThrow();
  });

  it("rejects product_quantization for a Flat field", () => {
    const schema = new Schema();
    // `addFlatField` is positional: (name, dimension, distance, embedder,
    // baseWeight, quantizer)
    expect(() =>
      schema.addFlatField("embedding", 4, undefined, undefined, undefined, "product_quantization"),
    ).toThrow(/only supported by HNSW/);
  });
});

// `low` / `high` stretch the segment's int8 range to [-100, 100] (step
// ~0.78), so `a` and `b` share one int8 code and tie under the default
// quantizer; Float16 keeps them apart, so the exact match `b` must rank
// first.
async function expectFloat16RanksExactMatchFirst(schema) {
  const index = await Index.create(null, schema);
  await index.putDocument("low", { embedding: [-100.0, -100.0] });
  await index.putDocument("high", { embedding: [100.0, 100.0] });
  await index.putDocument("a", { embedding: [0.2, 0.0] });
  await index.putDocument("b", { embedding: [0.25, 0.0] });
  await index.commit();

  const results = await index.searchVector("embedding", [0.25, 0.0], 2);
  expect(results.map((r) => r.id)).toEqual(["b", "a"]);
}

describe("Float16 quantizer (#623)", () => {
  it("an HNSW field ranks the nearest document first", async () => {
    const schema = new Schema();
    schema.addHnswField(
      "embedding", 2, "euclidean", undefined, undefined, undefined, undefined, "float16",
    );
    await expectFloat16RanksExactMatchFirst(schema);
  });

  it("a Flat field ranks the nearest document first", async () => {
    const schema = new Schema();
    schema.addFlatField("embedding", 2, "euclidean", undefined, undefined, "float16");
    await expectFloat16RanksExactMatchFirst(schema);
  });
});
