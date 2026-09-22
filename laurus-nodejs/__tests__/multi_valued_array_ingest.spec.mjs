/**
 * Regression tests for sending JS arrays into multi-valued numeric fields
 * (Issue #1178).
 *
 * `json_to_data_value` used to turn every array into `DataValue::Vector`,
 * which the core's `coerce_to_integer` / `coerce_to_float` multi-valued
 * branches reject — so a `multiValued = true` Integer/Float field could be
 * *read* from Node.js but never *written* from it. Arrays now go through the
 * same `infer_from_json` the server gateway uses (all-integer -> Int64Array,
 * otherwise numeric -> Float64Array) and the schema-aware coercion in the
 * core routes them.
 */

import { describe, it, expect } from "vitest";
import { Index, Schema } from "../index.js";

async function indexWithIntegerField(multiValued) {
  const schema = new Schema();
  schema.addTextField("title");
  // (name, stored, indexed, multiValued)
  schema.addIntegerField("tags", true, true, multiValued);
  return Index.create(null, schema);
}

async function indexWithFloatField() {
  const schema = new Schema();
  schema.addTextField("title");
  schema.addFloatField("scores", true, true, true);
  return Index.create(null, schema);
}

describe("multi-valued numeric arrays from JS (#1178)", () => {
  it("round-trips an integer array through a multiValued integer field", async () => {
    const index = await indexWithIntegerField(true);
    // Before #1178 this putDocument rejected: the array arrived as a Vector,
    // which a multi-valued integer field does not accept.
    await index.putDocument("doc1", { title: "t", tags: [1, 2, 3] });
    await index.commit();

    const docs = await index.getDocuments("doc1");
    expect(docs).toHaveLength(1);
    expect(docs[0].tags).toEqual([1, 2, 3]);
  });

  it("round-trips a float array through a multiValued float field", async () => {
    const index = await indexWithFloatField();
    await index.putDocument("doc1", { title: "t", scores: [1.5, 2.0] });
    await index.commit();

    const docs = await index.getDocuments("doc1");
    expect(docs[0].scores).toEqual([1.5, 2.0]);
  });

  it("widens a mixed int/float array to a float array", async () => {
    const index = await indexWithFloatField();
    await index.putDocument("doc1", { title: "t", scores: [1, 2.5] });
    await index.commit();

    const docs = await index.getDocuments("doc1");
    expect(docs[0].scores).toEqual([1, 2.5]);
  });

  it("accepts an empty array on a multiValued integer field", async () => {
    const index = await indexWithIntegerField(true);
    await index.putDocument("doc1", { title: "t", tags: [] });
    await index.commit();

    const docs = await index.getDocuments("doc1");
    expect(docs[0].tags).toEqual([]);
  });

  it("still rejects an array sent to a single-valued integer field", async () => {
    const index = await indexWithIntegerField(false);
    await expect(
      index.putDocument("doc1", { title: "t", tags: [2020, 2021] }),
    ).rejects.toThrow();
  });

  it("still rejects an array with non-numeric elements", async () => {
    const index = await indexWithIntegerField(true);
    await expect(
      index.putDocument("doc1", { title: "t", tags: [1, "x"] }),
    ).rejects.toThrow();
  });
});
