/**
 * Integer parameters (Issue #1612).
 *
 * napi-rs read a `u32` parameter with `napi_get_value_uint32`, which keeps
 * the low 32 bits of the number without an error: `-1` became 4294967295,
 * `1.5` became 1, 4294967296 became 0, and `NaN` and the infinities became
 * 0. Every integer parameter and option field now takes an integer from 0
 * to 4294967295 and throws (or rejects with) an `InvalidArg` error naming
 * the method and the parameter for anything else.
 */

import { describe, it, expect, beforeAll } from "vitest";
import {
  CommitPolicy,
  FuzzyQuery,
  Geo3dNearestQuery,
  Index,
  Schema,
  SearchRequest,
  SpanQuery,
  SynonymDictionary,
  SynonymGraphFilter,
  WalSyncPolicy,
  WhitespaceTokenizer,
} from "../index.js";

const u = undefined;
const BAD = [-1, 1.5, 4294967296, NaN, Infinity];

let index;

beforeAll(async () => {
  const schema = new Schema();
  schema.addTextField("title");
  schema.addFlatField("embedding", 2);
  index = await Index.create(null, schema);
  for (const id of ["a", "b", "c"]) {
    await index.putDocument(id, { title: `rust ${id}`, embedding: [1, 0] });
  }
  await index.commit();
});

function applyTokens(field, value) {
  const dictionary = new SynonymDictionary();
  dictionary.addSynonymGroup(["rust", "rustlang"]);
  const tokens = new WhitespaceTokenizer().tokenize("rust is");
  tokens[1][field] = value;
  return new SynonymGraphFilter(dictionary).apply(tokens);
}

// [context, parameter, call with the value]
const CASES = [
  [
    "Schema.addTextField",
    "positionIncrementGap",
    (v) => new Schema().addTextField("t", undefined, undefined, undefined, undefined, undefined, true, v),
  ],
  ["Schema.addHnswField", "dimension", (v) => new Schema().addHnswField("v", v)],
  ["Schema.addHnswField", "m", (v) => new Schema().addHnswField("v", 8, u, v)],
  ["Schema.addHnswField", "efConstruction", (v) => new Schema().addHnswField("v", 8, u, u, v)],
  ["Schema.addHnswField", "defaultEfSearch", (v) => new Schema().addHnswField("v", 8, u, u, u, v)],
  [
    "Schema.addHnswField",
    "subvectorCount",
    (v) => new Schema().addHnswField("v", 8, u, u, u, u, u, "product_quantization", v),
  ],
  ["Schema.addFlatField", "dimension", (v) => new Schema().addFlatField("v", v)],
  ["Schema.addIvfField", "dimension", (v) => new Schema().addIvfField("v", v)],
  ["Schema.addIvfField", "nClusters", (v) => new Schema().addIvfField("v", 8, u, v)],
  ["Schema.addIvfField", "nProbe", (v) => new Schema().addIvfField("v", 8, u, u, v)],
  ["Schema.addMultiVectorField", "dimension", (v) => new Schema().addMultiVectorField("v", v)],
  ["Schema.addMultiVectorField", "poolFactor", (v) => new Schema().addMultiVectorField("v", 8, u, u, u, v)],
  ["Index.search", "limit", (v) => index.search("title:rust", v)],
  ["Index.search", "offset", (v) => index.search("title:rust", 10, v)],
  ["Index.searchTerm", "limit", (v) => index.searchTerm("title", "rust", v)],
  ["Index.searchTerm", "offset", (v) => index.searchTerm("title", "rust", 10, v)],
  ["Index.searchVector", "limit", (v) => index.searchVector("embedding", [1, 0], v)],
  ["Index.searchVector", "offset", (v) => index.searchVector("embedding", [1, 0], 10, v)],
  ["Index.searchVectorText", "limit", (v) => index.searchVectorText("embedding", "rust", v)],
  ["Index.searchVectorText", "offset", (v) => index.searchVectorText("embedding", "rust", 10, v)],
  ["Index.searchBatch", "limit", (v) => index.searchBatch(["title:rust"], v)],
  ["Index.searchBatch", "offset", (v) => index.searchBatch(["title:rust"], 10, v)],
  ["FuzzyQuery", "maxEdits", (v) => new FuzzyQuery("title", "rust", v)],
  ["Geo3dNearestQuery.kNearest", "k", (v) => Geo3dNearestQuery.kNearest("p", 0, 0, 0, v)],
  ["SpanQuery.near", "slop", (v) => SpanQuery.near("title", ["rust", "is"], v)],
  [
    "SpanQuery.nearSpans",
    "slop",
    (v) => SpanQuery.nearSpans("title", [SpanQuery.term("title", "rust")], v),
  ],
  [
    "SpanQuery.within",
    "distance",
    (v) => SpanQuery.within("title", SpanQuery.term("title", "a"), SpanQuery.term("title", "b"), v),
  ],
  ["CommitPolicy.everyDocs", "n", (v) => CommitPolicy.everyDocs(v)],
  ["CommitPolicy.intervalMs", "ms", (v) => CommitPolicy.intervalMs(v)],
  ["WalSyncPolicy.group", "maxRecords", (v) => WalSyncPolicy.group(v)],
  ["WalSyncPolicy.group", "maxBytes", (v) => WalSyncPolicy.group(u, v)],
  [
    "Index.search",
    "highlight.fragmentSize",
    (v) => index.search("title:rust", 10, 0, { fields: ["title"], fragmentSize: v }),
  ],
  [
    "Index.search",
    "highlight.maxFragments",
    (v) => index.search("title:rust", 10, 0, { fields: ["title"], maxFragments: v }),
  ],
  [
    "Index.search",
    "rescore.windowSize",
    (v) => index.search("title:rust", 10, 0, u, { field: "tokens", vectors: [[1, 0]], windowSize: v }),
  ],
  [
    "Index.search",
    "groupBy.groupSize",
    (v) => index.search("title:rust", 10, 0, u, u, { field: "_id", groupSize: v }),
  ],
  ["SearchRequest", "limit", (v) => new SearchRequest({ limit: v })],
  ["SearchRequest", "offset", (v) => new SearchRequest({ offset: v })],
  ["SynonymGraphFilter.apply", "tokens[1].position", (v) => applyTokens("position", v)],
  ["SynonymGraphFilter.apply", "tokens[1].startOffset", (v) => applyTokens("startOffset", v)],
  ["SynonymGraphFilter.apply", "tokens[1].endOffset", (v) => applyTokens("endOffset", v)],
  [
    "SynonymGraphFilter.apply",
    "tokens[1].positionIncrement",
    (v) => applyTokens("positionIncrement", v),
  ],
  [
    "SynonymGraphFilter.apply",
    "tokens[1].positionLength",
    (v) => applyTokens("positionLength", v),
  ],
];

/**
 * Call `call` with each of `values` and return a line for every value that
 * did not throw, or reject with, an `InvalidArg` error with the expected
 * message. Collecting them first reports every value at once.
 */
async function mismatches(context, param, call, values = BAD) {
  const out = [];
  for (const value of values) {
    const expected = `${context}: ${param} must be an integer from 0 to 4294967295, got ${value}`;
    try {
      await call(value);
      out.push(`${value}: no error`);
    } catch (e) {
      if (e?.code !== "InvalidArg" || e?.message !== expected) {
        out.push(`${value}: ${e?.code} ${e?.message}`);
      }
    }
  }
  return out;
}

describe("integer parameters reject a value that is not an integer from 0 to 4294967295", () => {
  it.each(CASES)("%s: %s", async (context, param, call) => {
    const lines = await mismatches(context, param, call);
    expect(lines, lines.join(" | ")).toEqual([]);
  });

  it("names the method that converts the options", async () => {
    const lines = [
      ...(await mismatches(
        "Index.searchTerm",
        "highlight.maxFragments",
        (v) => index.searchTerm("title", "rust", 10, 0, { fields: ["title"], maxFragments: v }),
        [-1],
      )),
      ...(await mismatches(
        "Index.searchBatch",
        "groupBy.groupSize",
        (v) => index.searchBatch(["title:rust"], 10, 0, u, { field: "_id", groupSize: v }),
        [-1],
      )),
      ...(await mismatches(
        "SearchRequest",
        "highlight.fragmentSize",
        (v) => {
          const request = new SearchRequest({
            queryDsl: "title:rust",
            highlight: { fields: ["title"], fragmentSize: v },
          });
          return index.searchWithRequest(request);
        },
        [-1],
      )),
    ];
    expect(lines, lines.join(" | ")).toEqual([]);
  });
});

describe("integer parameters accept an integer from 0 to 4294967295 as before", () => {
  function dimension(schema) {
    return schema
      .toToml()
      .split("\n")
      .map((l) => l.trim())
      .find((l) => l.startsWith("dimension"));
  }

  it("keeps 0, -0 and 4294967295, and the defaults", () => {
    const max = new Schema();
    max.addFlatField("v", 4294967295);
    expect(dimension(max)).toBe("dimension = 4294967295");

    const negativeZero = new Schema();
    negativeZero.addFlatField("v", -0);
    expect(dimension(negativeZero)).toBe("dimension = 0");

    const defaults = new Schema();
    defaults.addHnswField("v", 8);
    expect(defaults.toToml()).toMatch(/m = 16\n/);
  });

  it("searches with the limit and offset given", async () => {
    expect((await index.search("title:rust", 4294967295)).length).toBe(3);
    expect((await index.search("title:rust", 2, 0)).length).toBe(2);
    expect((await index.search("title:rust", 10, 2)).length).toBe(1);
    expect((await index.search("title:rust", 0)).length).toBe(0);
    expect((await index.search("title:rust")).length).toBe(3);
  });

  it("keeps the positions of the tokens passed to SynonymGraphFilter.apply", () => {
    const out = applyTokens("position", 1);
    expect(out.map((t) => t.position)).toEqual([0, 0, 1]);
  });
});
