/**
 * Lexical search example for laurus-nodejs.
 *
 * Demonstrates various query types: Term, Phrase, Fuzzy, Wildcard,
 * NumericRange, Boolean, and SearchRequest.
 */

import { Index, PhraseQuery, Schema, SearchRequest } from "../index.js";

// Create schema and index
const schema = new Schema();
schema.addTextField("name");
schema.addTextField("description");
schema.addIntegerField("stars");
schema.setDefaultFields(["name", "description"]);

const index = await Index.create(null, schema);

await index.putDocument("express", {
  name: "Express",
  description: "Fast unopinionated minimalist web framework for Node.js.",
  stars: 65000,
});
await index.putDocument("fastify", {
  name: "Fastify",
  description: "Fast and low overhead web framework for Node.js.",
  stars: 33000,
});
await index.putDocument("nextjs", {
  name: "Next.js",
  description: "The React framework for production and server-side rendering.",
  stars: 128000,
});
await index.putDocument("vitest", {
  name: "Vitest",
  description: "A blazing fast unit test framework powered by Vite.",
  stars: 14000,
});
await index.commit();

// DSL search
console.log("=== DSL: 'name:express' ===");
for (const r of await index.search("name:express", 5)) {
  console.log(`  ${r.id}  ${r.document.name}`);
}

// Term query
console.log("\n=== Term: description='framework' ===");
for (const r of await index.searchTerm("description", "framework", 5)) {
  console.log(`  ${r.id}  ${r.document.name}`);
}

// Phrase query via SearchRequest
console.log("\n=== Phrase: description='web framework' ===");
const phraseReq = new SearchRequest({ limit: 5 });
phraseReq.setLexicalPhrase(new PhraseQuery("description", ["web", "framework"]));
for (const r of await index.searchWithRequest(phraseReq)) {
  console.log(`  ${r.id}  ${r.document.name}`);
}

// DSL wildcard
console.log("\n=== DSL: 'name:fast*' ===");
for (const r of await index.search("name:fast*", 5)) {
  console.log(`  ${r.id}  ${r.document.name}`);
}

// DSL fuzzy (tilde after the term)
console.log("\n=== DSL: 'name:exprss~' (fuzzy) ===");
for (const r of await index.search("name:exprss~", 5)) {
  console.log(`  ${r.id}  ${r.document.name}`);
}

console.log("\nDone.");
