/**
 * Quickstart example for laurus-nodejs.
 *
 * Demonstrates basic index creation, document indexing, and search.
 */

import { Index, Schema } from "../index.js";

// 1. Define a schema
const schema = new Schema();
schema.addTextField("name");
schema.addTextField("description");
schema.setDefaultFields(["name", "description"]);

// 2. Create an in-memory index
const index = await Index.create(null, schema);

// 3. Index documents
await index.putDocument("express", {
  name: "Express",
  description: "Fast, unopinionated, minimalist web framework for Node.js.",
});
await index.putDocument("fastify", {
  name: "Fastify",
  description: "Fast and low overhead web framework for Node.js.",
});
await index.commit();

// 4. Search with DSL string
console.log("=== DSL search: 'framework' ===");
const dslResults = (await index.search("framework", 5)).results;
for (const r of dslResults) {
  console.log(`  ${r.id}  score=${r.score.toFixed(4)}  name="${r.document.name}"`);
}

// 5. Search with term query
console.log("\n=== Term search: description='minimalist' ===");
const termResults = (await index.searchTerm("description", "minimalist", 5)).results;
for (const r of termResults) {
  console.log(`  ${r.id}  score=${r.score.toFixed(4)}  name="${r.document.name}"`);
}

// 6. Stats
const stats = index.stats();
console.log(`\nIndex stats: ${stats.documentCount} documents`);
