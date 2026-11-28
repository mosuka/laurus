/**
 * Vector search example for laurus-nodejs.
 *
 * Demonstrates vector search with pre-computed embeddings.
 */

import { Index, Schema } from "../index.js";

// Create schema with HNSW vector field
const schema = new Schema();
schema.addTextField("name");
schema.addHnswField("embedding", 4, "cosine");
schema.setDefaultFields(["name"]);

const index = await Index.create(null, schema);

// Index npm packages with embeddings
await index.putDocument("express", {
  name: "Express — web framework",
  embedding: [0.9, 0.1, 0.2, 0.3],
});
await index.putDocument("pg", {
  name: "pg — PostgreSQL client",
  embedding: [0.1, 0.9, 0.2, 0.3],
});
await index.putDocument("vitest", {
  name: "Vitest — unit test framework",
  embedding: [0.2, 0.1, 0.9, 0.3],
});
await index.commit();

// Search with a query vector close to "express"
console.log("=== Vector search (close to express) ===");
const { results } = await index.searchVector(
  "embedding", [0.85, 0.15, 0.2, 0.3], 3,
);
for (const r of results) {
  console.log(`  ${r.id}  score=${r.score.toFixed(4)}  name="${r.document.name}"`);
}

// Search with a query vector close to "pg"
console.log("\n=== Vector search (close to pg) ===");
const { results: results2 } = await index.searchVector(
  "embedding", [0.15, 0.85, 0.2, 0.3], 3,
);
for (const r of results2) {
  console.log(`  ${r.id}  score=${r.score.toFixed(4)}  name="${r.document.name}"`);
}

console.log("\nDone.");
