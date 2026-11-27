/**
 * Hybrid search example for laurus-nodejs.
 *
 * Demonstrates combining lexical and vector search
 * with RRF and WeightedSum fusion.
 */

import {
  Index,
  RRF,
  Schema,
  SearchRequest,
  TermQuery,
  VectorQuery,
  WeightedSum,
} from "../index.js";

// Create schema with text + vector fields
const schema = new Schema();
schema.addTextField("name");
schema.addTextField("description");
schema.addHnswField("embedding", 4, "cosine");
schema.setDefaultFields(["name", "description"]);

const index = await Index.create(null, schema);

await index.putDocument("express", {
  name: "Express",
  description: "Fast minimalist web framework for Node.js.",
  embedding: [0.9, 0.1, 0.2, 0.3],
});
await index.putDocument("prisma", {
  name: "Prisma",
  description: "Next-generation ORM for Node.js and TypeScript.",
  embedding: [0.1, 0.9, 0.2, 0.3],
});
await index.putDocument("esbuild", {
  name: "esbuild",
  description: "An extremely fast bundler for the web.",
  embedding: [0.2, 0.1, 0.9, 0.3],
});
await index.commit();

// Hybrid search with RRF fusion
console.log("=== Hybrid search (RRF) ===");
const rrfReq = new SearchRequest({ limit: 5 });
rrfReq.setLexicalTerm(new TermQuery("description", "fast"));
rrfReq.setVectorQuery(new VectorQuery("embedding", [0.85, 0.15, 0.2, 0.3]));
rrfReq.setRrfFusion(new RRF(60.0));
for (const r of (await index.searchWithRequest(rrfReq)).results) {
  console.log(`  ${r.id}  score=${r.score.toFixed(4)}  name="${r.document.name}"`);
}

// Hybrid search with WeightedSum fusion
console.log(
  "\n=== Hybrid search (WeightedSum: 0.3 lexical, 0.7 vector) ===",
);
const wsReq = new SearchRequest({ limit: 5 });
wsReq.setLexicalTerm(new TermQuery("description", "fast"));
wsReq.setVectorQuery(new VectorQuery("embedding", [0.85, 0.15, 0.2, 0.3]));
wsReq.setWeightedSumFusion(new WeightedSum(0.3, 0.7));
for (const r of (await index.searchWithRequest(wsReq)).results) {
  console.log(`  ${r.id}  score=${r.score.toFixed(4)}  name="${r.document.name}"`);
}

console.log("\nDone.");
