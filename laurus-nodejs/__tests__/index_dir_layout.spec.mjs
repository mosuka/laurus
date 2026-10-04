// Tests for the on-disk directory layout of a file-backed `Index` (Issue #1059).
//
// Before this change, `Index.create(path)` wrote segment files directly
// under `path`, incompatible with `laurus-cli`'s `<path>/schema.toml` +
// `<path>/store/` convention. These tests verify the new shared layout:
// schema auto-persistence, auto-loading on reopen, the reopen-with-schema
// conflict error, and legacy-layout detection.

import { describe, it, expect } from "vitest";
import { mkdtempSync, existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Index, Schema } from "../index.js";

function freshDir() {
  return mkdtempSync(join(tmpdir(), "laurus-index-dir-"));
}

describe("Index(path) directory layout (#1059)", () => {
  it("creating a file-backed index writes schema.toml and store/", async () => {
    const dir = freshDir();
    const schema = new Schema();
    schema.addTextField("title");

    await Index.create(dir, schema);

    expect(existsSync(join(dir, "schema.toml"))).toBe(true);
    expect(existsSync(join(dir, "store"))).toBe(true);
    // No stray top-level segment directories from the old flat layout.
    expect(existsSync(join(dir, "lexical"))).toBe(false);
  });

  it("reopening without a schema loads the persisted schema and data", async () => {
    const dir = freshDir();
    const schema = new Schema();
    schema.addTextField("title");
    schema.setDefaultFields(["title"]);

    const index = await Index.create(dir, schema);
    await index.putDocument("doc1", { title: "hello world" });
    await index.commit();
    index.close();

    const reopened = await Index.create(dir);
    const results = await reopened.search("title:hello", 5);
    expect(results.length).toBe(1);
  });

  it("reopening with an explicit schema throws", async () => {
    const dir = freshDir();
    const schema = new Schema();
    schema.addTextField("title");
    await Index.create(dir, schema);

    await expect(Index.create(dir, schema)).rejects.toThrow(/schema\.toml/);
  });

  it("reopening with no schema at all succeeds on the empty default", async () => {
    const dir = freshDir();
    const index = await Index.create(dir);
    index.close();
    await expect(Index.create(dir)).resolves.toBeDefined();
  });

  it("rejects a pre-Issue-1059 legacy flat layout", async () => {
    const dir = freshDir();
    // Simulate a directory written by a laurus-nodejs version predating
    // Issue #1059: segment files directly under the path, no schema.toml.
    writeFileSync(join(dir, "engine.wal"), "");

    await expect(Index.create(dir)).rejects.toThrow(/pre-Issue-1059/);
  });

  it("does not treat a merely-empty directory as a legacy layout", async () => {
    const dir = freshDir();
    await Index.create(dir);
    expect(existsSync(join(dir, "schema.toml"))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Build-time failure cleanup (Issue #1308)
//
// A malformed regex parses fine as schema TOML/JS (it's just a string) but
// fails only once the analyzer is resolved during the engine build, after
// schema.toml and store/ already exist. `Index.create` must roll both back
// before throwing, so a retry at the same path isn't blocked by the failed
// attempt's leftovers.
// ---------------------------------------------------------------------------

describe("Index.create build-time failure cleanup (#1308)", () => {
  function brokenSchema() {
    const schema = new Schema();
    schema.addAnalyzer("bad", { type: "regex", pattern: "(" });
    schema.addTextField("body", true, true, true, true, "bad");
    return schema;
  }

  it("a build failure in a fresh directory leaves nothing behind, and a fixed retry succeeds", async () => {
    const dir = freshDir();

    await expect(Index.create(dir, brokenSchema())).rejects.toThrow();
    expect(existsSync(join(dir, "schema.toml"))).toBe(false);
    expect(existsSync(join(dir, "store"))).toBe(false);

    const fixedSchema = new Schema();
    fixedSchema.addTextField("body");
    await expect(Index.create(dir, fixedSchema)).resolves.toBeDefined();
  });

  it("a build failure in a pre-existing directory only removes schema.toml/store, keeping unrelated files", async () => {
    const dir = freshDir();
    writeFileSync(join(dir, "README.txt"), "keep me");

    await expect(Index.create(dir, brokenSchema())).rejects.toThrow();

    expect(existsSync(join(dir, "README.txt"))).toBe(true);
    expect(existsSync(join(dir, "schema.toml"))).toBe(false);
    expect(existsSync(join(dir, "store"))).toBe(false);

    const fixedSchema = new Schema();
    fixedSchema.addTextField("body");
    await expect(Index.create(dir, fixedSchema)).resolves.toBeDefined();
  });
});
