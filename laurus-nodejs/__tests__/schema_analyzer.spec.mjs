/**
 * Tests for `Schema#addAnalyzer` and TOML schema loading/saving (Issue #1062),
 * adapted from `laurus-python`'s `test_schema_analyzer.py` (#1058).
 *
 * These deliberately avoid the Lindera tokenizer: this repository ships no
 * Lindera dictionary. `whitespace`/`ngram`/`regex` tokenizers exercise the
 * same code paths (`addAnalyzer`'s object-to-core-enum conversion, and the
 * TOML serialize/deserialize round trip) without that dependency.
 *
 * Each behavioural test proves the analyzer actually reached the query
 * engine (a deterministic search-result difference), not just that
 * `addAnalyzer`/`fromToml` didn't throw.
 */

import { describe, it, expect } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Index, Schema } from "../index.js";

// ---------------------------------------------------------------------------
// addAnalyzer: behavioural — the analyzer actually takes effect
// ---------------------------------------------------------------------------

describe("Schema#addAnalyzer behaviour", () => {
  it("ngram tokenizer enables substring match", async () => {
    const schema = new Schema();
    schema.addAnalyzer("ngram3", { type: "ngram", min_gram: 3, max_gram: 3 });
    schema.addTextField("title", true, true, true, true, "ngram3");
    schema.addTextField("plain"); // default "standard" analyzer: whole-word tokens

    const index = await Index.create(null, schema);
    await index.putDocument("doc1", { title: "hello", plain: "hello" });
    await index.commit();

    // "ell" is a substring of "hello", only reachable via 3-grams (hel/ell/llo).
    expect((await index.search("title:ell", 5)).length).toBe(1);
    expect((await index.search("plain:ell", 5)).length).toBe(0);
  });

  it("token filters apply lowercase", async () => {
    const schema = new Schema();
    schema.addAnalyzer("ws", { type: "whitespace" });
    schema.addAnalyzer("wsLower", { type: "whitespace" }, undefined, [{ type: "lowercase" }]);
    schema.addTextField("raw", true, true, true, true, "ws");
    schema.addTextField("lower", true, true, true, true, "wsLower");

    const index = await Index.create(null, schema);
    await index.putDocument("doc1", { raw: "HELLO World", lower: "HELLO World" });
    await index.commit();

    expect((await index.search("raw:hello", 5)).length).toBe(0);
    expect((await index.search("lower:hello", 5)).length).toBe(1);
  });

  it("char filters apply pattern replace", async () => {
    const schema = new Schema();
    schema.addAnalyzer("dash", { type: "whitespace" });
    schema.addAnalyzer(
      "dashSplit",
      { type: "whitespace" },
      [{ type: "pattern_replace", pattern: "-", replacement: " " }],
    );
    schema.addTextField("raw", true, true, true, true, "dash");
    schema.addTextField("split", true, true, true, true, "dashSplit");

    const index = await Index.create(null, schema);
    await index.putDocument("doc1", { raw: "state-of-the-art", split: "state-of-the-art" });
    await index.commit();

    expect((await index.search("raw:art", 5)).length).toBe(0);
    expect((await index.search("split:art", 5)).length).toBe(1);
  });

  it("defaults to no filters", () => {
    const schema = new Schema();
    schema.addAnalyzer("ws", { type: "whitespace" });
    expect(schema.analyzerNames()).toEqual(["ws"]);
  });
});

// ---------------------------------------------------------------------------
// addAnalyzer: error surface
// ---------------------------------------------------------------------------

describe("Schema#addAnalyzer errors", () => {
  it("rejects unknown tokenizer type", () => {
    const schema = new Schema();
    expect(() => schema.addAnalyzer("bad", { type: "kuromoji" })).toThrow(/tokenizer/);
  });

  it("rejects a tokenizer missing a required field", () => {
    const schema = new Schema();
    expect(() =>
      schema.addAnalyzer("bad", { type: "ngram", min_gram: 2 }), // missing max_gram
    ).toThrow(/tokenizer/);
  });

  it("rejects an unknown char filter type with an indexed message", () => {
    const schema = new Schema();
    expect(() =>
      schema.addAnalyzer("bad", { type: "whitespace" }, [{ type: "unknown_filter" }]),
    ).toThrow(/charFilters\[0\]/);
  });

  it("rejects an invalid token filter with an indexed message", () => {
    const schema = new Schema();
    expect(() =>
      schema.addAnalyzer("bad", { type: "whitespace" }, undefined, [
        { type: "limit", limit: -1 },
      ]),
    ).toThrow(/tokenFilters\[0\]/);
  });

  it("accepts a boolean gaps value on the regex tokenizer", () => {
    const schema = new Schema();
    schema.addAnalyzer("r", { type: "regex", pattern: "\\w+", gaps: true });
    expect(schema.analyzerNames()).toEqual(["r"]);
  });
});

// ---------------------------------------------------------------------------
// fromToml / fromTomlFile
// ---------------------------------------------------------------------------

const SCHEMA_TOML = `
default_fields = ["title"]

[analyzers.ngram3]
tokenizer = { type = "ngram", min_gram = 3, max_gram = 3 }

[fields.title.Text]
indexed = true
stored = true
term_vectors = false
analyzer = "ngram3"
`;

describe("Schema.fromToml / fromTomlFile", () => {
  it("loads a schema from TOML and searches with it", async () => {
    const schema = Schema.fromToml(SCHEMA_TOML);
    expect(schema.fieldNames()).toEqual(["title"]);
    expect(schema.analyzerNames()).toEqual(["ngram3"]);

    const index = await Index.create(null, schema);
    await index.putDocument("doc1", { title: "hello" });
    await index.commit();
    expect((await index.search("title:ell", 5)).length).toBe(1);
  });

  it("loads a schema from a TOML file", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "laurus-schema-"));
    try {
      const file = path.join(dir, "schema.toml");
      fs.writeFileSync(file, SCHEMA_TOML);

      const schema = Schema.fromTomlFile(file);
      expect(schema.fieldNames()).toEqual(["title"]);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("throws when the TOML file does not exist", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "laurus-schema-"));
    try {
      expect(() => Schema.fromTomlFile(path.join(dir, "does_not_exist.toml"))).toThrow();
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("throws with a helpful message on malformed TOML", () => {
    expect(() => Schema.fromToml("not = [valid")).toThrow(/TOML/);
  });
});

// ---------------------------------------------------------------------------
// toToml / toTomlFile: round trip
// ---------------------------------------------------------------------------

describe("Schema#toToml / toTomlFile round trip", () => {
  it("round-trips through a TOML string", async () => {
    const schema = new Schema();
    schema.addAnalyzer(
      "ngram3",
      { type: "ngram", min_gram: 3, max_gram: 3 },
      [{ type: "unicode_normalization", form: "nfkc" }],
      [{ type: "lowercase" }],
    );
    schema.addTextField("title", true, true, true, true, "ngram3");
    schema.setDefaultFields(["title"]);

    const tomlStr = schema.toToml();
    const restored = Schema.fromToml(tomlStr);

    expect(restored.fieldNames()).toEqual(schema.fieldNames());
    expect(restored.analyzerNames()).toEqual(schema.analyzerNames());

    const index = await Index.create(null, restored);
    await index.putDocument("doc1", { title: "hello" });
    await index.commit();
    expect((await index.search("title:ell", 5)).length).toBe(1);
  });

  it("round-trips through a TOML file", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "laurus-schema-"));
    try {
      const schema = new Schema();
      schema.addAnalyzer("ws", { type: "whitespace" });
      schema.addTextField("title", true, true, true, true, "ws");

      const file = path.join(dir, "schema.toml");
      schema.toTomlFile(file);
      const restored = Schema.fromTomlFile(file);

      expect(restored.fieldNames()).toEqual(schema.fieldNames());
      expect(restored.analyzerNames()).toEqual(schema.analyzerNames());
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
