/**
 * External Embedder example for laurus-nodejs.
 *
 * Demonstrates vector and hybrid search where embeddings are produced outside
 * laurus and passed as pre-computed vectors via VectorQuery.
 *
 * This approach is useful when you want to:
 * - Use any embedding library (@xenova/transformers, OpenAI, etc.)
 * - Control the embedding model independently of the index schema
 * - Reuse embeddings across multiple indexes
 *
 * Run with:
 *     npm install @xenova/transformers   # optional but recommended
 *     npm run build:debug
 *     node examples/external-embedder.mjs
 */

import { Index, RRF, Schema, SearchRequest, TermQuery, VectorQuery, WeightedSum } from "../index.js";

// ---------------------------------------------------------------------------
// Embedding helper
// ---------------------------------------------------------------------------

let embed;
let DIM;

try {
  const { pipeline } = await import("@xenova/transformers");
  const extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2",
  );
  DIM = 384;

  embed = async (text) => {
    const output = await extractor(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  };

  console.log("Using @xenova/transformers (all-MiniLM-L6-v2) for embeddings.\n");
} catch {
  // Fallback: deterministic pseudo-embeddings for demo purposes only.
  // Real similarity is not meaningful with these vectors.
  DIM = 64;

  embed = async (text) => {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      hash = (Math.imul(31, hash) + text.charCodeAt(i)) | 0;
    }
    const rng = mulberry32(hash >>> 0);
    const raw = Array.from({ length: DIM }, () => gaussianRng(rng));
    const norm = Math.sqrt(raw.reduce((s, x) => s + x * x, 0)) || 1.0;
    return raw.map((x) => x / norm);
  };

  console.log(
    "[NOTE] @xenova/transformers not found -- using random fallback vectors.\n" +
      "       Results will NOT reflect semantic similarity.\n" +
      "       Install with: npm install @xenova/transformers\n",
  );
}

function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianRng(rng) {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Dataset — Node.js ecosystem chunks
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
// Main
// ---------------------------------------------------------------------------

console.log("=== Laurus External Embedder Example ===\n");
console.log(`Embedding model dimension: ${DIM}\n`);

// -- Schema ------------------------------------------------------------------
const schema = new Schema();
schema.addTextField("title");
schema.addTextField("text");
schema.addTextField("category", null, null, null, null, "keyword");
schema.addIntegerField("page");
schema.addFlatField("text_vec", DIM, "cosine");
schema.setDefaultFields(["text"]);

const index = await Index.create(null, schema);

// -- Index -------------------------------------------------------------------
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
}
await index.commit();
console.log(`Indexed ${CHUNKS.length} chunks.\n`);

// =====================================================================
// [A] Basic Vector Search
// =====================================================================
console.log("=".repeat(60));
console.log("[A] Vector-only: 'database ORM queries'");
console.log("=".repeat(60));
printResults(
  await index.searchVector("text_vec", await embed("database ORM queries"), 3),
);

// =====================================================================
// [B] Filtered Vector Search -- category filter
// =====================================================================
console.log("\n" + "=".repeat(60));
console.log("[B] Filtered vector: 'database ORM queries' + category='testing'");
console.log("=".repeat(60));
const reqB = new SearchRequest({ limit: 3 });
reqB.setVectorQuery(new VectorQuery("text_vec", await embed("database ORM queries")));
reqB.setFilterTerm(new TermQuery("category", "testing"));
printResults(await index.searchWithRequest(reqB));

// =====================================================================
// [C] Hybrid search -- RRF Fusion
// =====================================================================
console.log("\n" + "=".repeat(60));
console.log("[C] Hybrid (RRF k=60): vector='middleware pipeline' + lexical='middleware'");
console.log("=".repeat(60));
const reqC = new SearchRequest({ limit: 3 });
reqC.setLexicalTerm(new TermQuery("text", "middleware"));
reqC.setVectorQuery(new VectorQuery("text_vec", await embed("middleware pipeline")));
reqC.setRrfFusion(new RRF(60.0));
printResults(await index.searchWithRequest(reqC));

// =====================================================================
// [D] Hybrid search -- WeightedSum Fusion
// =====================================================================
console.log("\n" + "=".repeat(60));
console.log("[D] Hybrid (WeightedSum 0.3/0.7): vector='fast bundler' + lexical='typescript'");
console.log("=".repeat(60));
const reqD = new SearchRequest({ limit: 3 });
reqD.setLexicalTerm(new TermQuery("text", "typescript"));
reqD.setVectorQuery(new VectorQuery("text_vec", await embed("fast bundler")));
reqD.setWeightedSumFusion(new WeightedSum(0.3, 0.7));
printResults(await index.searchWithRequest(reqD));

// =====================================================================
// [E] Hybrid search with filter
// =====================================================================
console.log("\n" + "=".repeat(60));
console.log("[E] Hybrid + filter: vector='test automation' + lexical='snapshot' + category='testing'");
console.log("=".repeat(60));
const reqE = new SearchRequest({ limit: 3 });
reqE.setLexicalTerm(new TermQuery("text", "snapshot"));
reqE.setVectorQuery(new VectorQuery("text_vec", await embed("test automation")));
reqE.setFilterTerm(new TermQuery("category", "testing"));
reqE.setRrfFusion(new RRF(60.0));
printResults(await index.searchWithRequest(reqE));

console.log("\nExternal embedder example completed!");

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
