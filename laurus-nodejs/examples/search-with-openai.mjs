/**
 * Search with OpenAI Embedder example for laurus-nodejs.
 *
 * Demonstrates real vector search using OpenAI's text-embedding API.
 * Embeddings are produced on the Node.js side using the `openai` npm
 * package, then passed to Laurus as pre-computed vectors.
 *
 * Prerequisites:
 *     npm install openai
 *     export OPENAI_API_KEY=your-api-key-here
 *     npm run build:debug
 *     node examples/search-with-openai.mjs
 */

import { Index, RRF, Schema, SearchRequest, TermQuery, VectorQuery, WeightedSum } from "../index.js";

// ---------------------------------------------------------------------------
// OpenAI embedding helper
// ---------------------------------------------------------------------------

let OpenAI;
try {
  ({ default: OpenAI } = await import("openai"));
} catch {
  console.error("ERROR: openai package not installed.");
  console.error("       Install with: npm install openai");
  process.exit(1);
}

const MODEL = "text-embedding-3-small";
const DIM = 1536;

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error("ERROR: OPENAI_API_KEY environment variable not set.");
  console.error("       export OPENAI_API_KEY=your-api-key-here");
  process.exit(1);
}

const client = new OpenAI({ apiKey });

async function embed(text) {
  const response = await client.embeddings.create({
    input: text,
    model: MODEL,
  });
  return response.data[0].embedding;
}

console.log("=== Laurus Search with OpenAI Embedder ===\n");
console.log(`OpenAI embedder ready (model=${MODEL}, dim=${DIM})\n`);

// ---------------------------------------------------------------------------
// Dataset -- Node.js ecosystem chunks
// ---------------------------------------------------------------------------

const CHUNKS = [
  ["express_guide", "Express Web Framework", "Express provides a minimal and flexible routing layer with middleware support for building REST APIs.", 1, "framework"],
  ["express_guide", "Express Web Framework", "Express middleware functions have access to the request and response objects and can modify the pipeline.", 2, "framework"],
  ["express_guide", "Express Web Framework", "Express template engines like EJS and Pug render dynamic HTML views from server-side data.", 3, "framework"],
  ["fastify_guide", "Fastify Web Framework", "Fastify uses a schema-based validation approach with JSON Schema for fast request and response serialization.", 1, "framework"],
  ["fastify_guide", "Fastify Web Framework", "Fastify plugins provide an encapsulated context for extending the framework with decorators and hooks.", 2, "framework"],
  ["nextjs_docs", "Next.js Full-Stack Framework", "Next.js server components render on the server and stream HTML to the client for faster initial loads.", 1, "framework"],
  ["nextjs_docs", "Next.js Full-Stack Framework", "Next.js file-based routing maps filesystem paths to URL routes with dynamic segments and catch-all patterns.", 2, "framework"],
  ["prisma_guide", "Prisma ORM", "Prisma Client generates a type-safe query builder from your schema for PostgreSQL, MySQL, and SQLite.", 1, "database"],
  ["prisma_guide", "Prisma ORM", "Prisma Migrate generates SQL migration files from schema changes and applies them to the database.", 2, "database"],
  ["vitest_book", "Testing with Vitest", "Vitest provides a Jest-compatible API with native ESM support and fast HMR-powered watch mode.", 1, "testing"],
  ["vitest_book", "Testing with Vitest", "Vitest snapshot testing captures component output and detects unintended changes automatically.", 2, "testing"],
  ["esbuild_docs", "esbuild Bundler", "esbuild compiles TypeScript and JSX to JavaScript with tree-shaking at speeds 10-100x faster than alternatives.", 1, "tooling"],
];

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const schema = new Schema();
schema.addTextField("title");
schema.addTextField("text");
schema.addTextField("category");
schema.addIntegerField("page");
schema.addFlatField("text_vec", DIM, "cosine");
schema.setDefaultFields(["text"]);

const index = await Index.create(null, schema);

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

console.log("--- Indexing chunked documents ---\n");
for (const [docId, title, text, page, category] of CHUNKS) {
  const vec = await embed(text);
  await index.addDocument(docId, {
    title,
    text,
    category,
    page,
    text_vec: vec,
  });
  console.log(`  Indexed ${docId} page ${page}: ${JSON.stringify(text.slice(0, 50))}...`);
}
await index.commit();
console.log(`\nIndexed ${CHUNKS.length} chunks.\n`);

// =====================================================================
// [A] Vector Search
// =====================================================================
console.log("=".repeat(60));
console.log("[A] Vector Search: 'database ORM queries'");
console.log("=".repeat(60));
printResults(
  await index.searchVector("text_vec", await embed("database ORM queries"), 3),
);

// =====================================================================
// [B] Filtered Vector Search -- category filter
// =====================================================================
console.log("\n" + "=".repeat(60));
console.log("[B] Filtered Vector Search: 'database ORM queries' + category='testing'");
console.log("=".repeat(60));
const reqB = new SearchRequest({ limit: 3 });
reqB.setVectorQuery(new VectorQuery("text_vec", await embed("database ORM queries")));
reqB.setFilterTerm(new TermQuery("category", "testing"));
printResults(await index.searchWithRequest(reqB));

// =====================================================================
// [C] Lexical Search
// =====================================================================
console.log("\n" + "=".repeat(60));
console.log("[C] Lexical Search: 'middleware'");
console.log("=".repeat(60));
printResults(await index.searchTerm("text", "middleware", 3));

// =====================================================================
// [D] Hybrid Search (RRF)
// =====================================================================
console.log("\n" + "=".repeat(60));
console.log("[D] Hybrid Search (RRF): vector='server-side rendering' + lexical='server'");
console.log("=".repeat(60));
const reqD = new SearchRequest({ limit: 3 });
reqD.setLexicalTerm(new TermQuery("text", "server"));
reqD.setVectorQuery(new VectorQuery("text_vec", await embed("server-side rendering")));
reqD.setRrfFusion(new RRF(60.0));
printResults(await index.searchWithRequest(reqD));

// =====================================================================
// [E] Hybrid Search (WeightedSum)
// =====================================================================
console.log("\n" + "=".repeat(60));
console.log("[E] Hybrid Search (WeightedSum 0.3/0.7): vector='fast bundler' + lexical='typescript'");
console.log("=".repeat(60));
const reqE = new SearchRequest({ limit: 3 });
reqE.setLexicalTerm(new TermQuery("text", "typescript"));
reqE.setVectorQuery(new VectorQuery("text_vec", await embed("fast bundler")));
reqE.setWeightedSumFusion(new WeightedSum(0.3, 0.7));
printResults(await index.searchWithRequest(reqE));

console.log("\nSearch with OpenAI example completed!");

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function printResults(results) {
  if (results.length === 0) {
    console.log("  (no results)");
    return;
  }
  for (const r of results) {
    const doc = r.document || {};
    const text = (doc.text || "").slice(0, 60);
    console.log(`  id=${JSON.stringify(r.id).padEnd(8)}  score=${r.score.toFixed(4)}  text=${JSON.stringify(text)}`);
  }
}
