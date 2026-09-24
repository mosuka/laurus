/**
 * Multimodal Search example for laurus-nodejs.
 *
 * Demonstrates how to store raw bytes (e.g. image data) in a Laurus index
 * and perform multimodal search using pre-computed CLIP embeddings produced
 * on the Node.js side.
 *
 * The approach:
 * 1. Encode images/text with a CLIP model (e.g. via @xenova/transformers).
 * 2. Store the raw bytes in a `bytes` field and the embedding in a flat field.
 * 3. Query with searchVector using a pre-computed embedding.
 *
 * Requirements (optional -- see fallback below):
 *     npm install @xenova/transformers
 *
 * Run with:
 *     npm run build:debug
 *     node examples/multimodal-search.mjs
 */

import { Index, Schema, SearchRequest, TermQuery, VectorQuery } from "../index.js";

// ---------------------------------------------------------------------------
// CLIP embedding helper
// ---------------------------------------------------------------------------

let embedText;
let embedImage;
let DIM;
let HAS_CLIP = false;

try {
  const { pipeline } = await import("@xenova/transformers");

  const textExtractor = await pipeline(
    "feature-extraction",
    "Xenova/clip-vit-base-patch32",
  );
  DIM = 512;
  HAS_CLIP = true;

  embedText = async (text) => {
    const output = await textExtractor(text, {
      pooling: "mean",
      normalize: true,
    });
    return Array.from(output.data);
  };

  embedImage = async (imageBytes) => {
    // For a real app, decode the image and run through the CLIP vision encoder.
    // @xenova/transformers CLIP pipeline expects decoded images.
    // Here we fall back to text-based embedding of a hash for demo purposes.
    const label = new TextDecoder("utf-8", { fatal: false }).decode(
      imageBytes.slice(-32),
    );
    return embedText(`image of ${label}`);
  };
} catch {
  DIM = 32;

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
    return (
      Math.sqrt(-2 * Math.log(u1 || 1e-10)) * Math.cos(2 * Math.PI * u2)
    );
  }

  function randUnit(seed) {
    const rng = mulberry32(seed >>> 0);
    const raw = Array.from({ length: DIM }, () => gaussianRng(rng));
    const norm = Math.sqrt(raw.reduce((s, x) => s + x * x, 0)) || 1.0;
    return raw.map((x) => x / norm);
  }

  function simpleHash(str) {
    let h = 0;
    for (let i = 0; i < str.length; i++) {
      h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
    }
    return h;
  }

  embedText = async (text) => randUnit(simpleHash(text));

  embedImage = async (imageBytes) => {
    let h = 0;
    const len = Math.min(imageBytes.length, 128);
    for (let i = 0; i < len; i++) {
      h = (Math.imul(31, h) + imageBytes[i]) | 0;
    }
    return randUnit(h);
  };

  console.log(
    "[NOTE] @xenova/transformers not found -- using random fallback vectors.\n" +
      "       Semantic similarity will NOT be meaningful.\n" +
      "       Install with: npm install @xenova/transformers\n",
  );
}

// ---------------------------------------------------------------------------
// Fake image bytes for demo (1x1 white pixel PNG)
// ---------------------------------------------------------------------------

const WHITE_PNG = new Uint8Array([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
  0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
  0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
  0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0x0f, 0x00, 0x00,
  0x01, 0x01, 0x00, 0x05, 0x18, 0xd4, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x49,
  0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
]);

function fakeImage(label) {
  const labelBytes = new TextEncoder().encode(label);
  const combined = new Uint8Array(WHITE_PNG.length + labelBytes.length);
  combined.set(WHITE_PNG);
  combined.set(labelBytes, WHITE_PNG.length);
  return combined;
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

const IMAGES = [
  ["img_express", "express_logo.png", "image"],
  ["img_fastify", "fastify_logo.png", "image"],
  ["img_nextjs", "nextjs_logo.png", "image"],
  ["img_prisma", "prisma_logo.png", "image"],
];

const TEXTS = [
  ["txt1", "Express is a minimal web framework for Node.js with robust routing", "text"],
  ["txt2", "Fastify is a high-performance HTTP framework with schema validation", "text"],
  ["txt3", "Next.js enables server-side rendering and static site generation", "text"],
  ["txt4", "Prisma is a next-generation ORM for TypeScript and Node.js", "text"],
];

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log("=== Laurus Multimodal Search Example ===\n");
if (HAS_CLIP) {
  console.log("Using CLIP (Xenova/clip-vit-base-patch32) for embeddings.\n");
} else {
  console.log("Using random fallback vectors (results not semantically meaningful).\n");
}

// -- Schema ------------------------------------------------------------------
const schema = new Schema();
schema.addBytesField("content");
schema.addTextField("filename");
schema.addTextField("type");
schema.addTextField("description");
schema.addFlatField("content_vec", DIM, "cosine");

const index = await Index.create(null, schema);

// -- Index images ------------------------------------------------------------
console.log("--- Indexing images ---");
for (const [docId, filename, mediaType] of IMAGES) {
  const rawBytes = fakeImage(filename);
  const vec = await embedImage(rawBytes);
  await index.addDocument(docId, {
    content: Array.from(rawBytes),
    filename,
    type: mediaType,
    description: "",
    content_vec: vec,
  });
  console.log(`  Indexed image: ${filename}`);
}

// -- Index text descriptions -------------------------------------------------
console.log("\n--- Indexing text descriptions ---");
for (const [docId, text, mediaType] of TEXTS) {
  const vec = await embedText(text);
  await index.addDocument(docId, {
    content: [],
    filename: "",
    type: mediaType,
    description: text,
    content_vec: vec,
  });
  console.log(`  Indexed text: ${JSON.stringify(text.slice(0, 50))}`);
}

await index.commit();
console.log();

// =====================================================================
// [A] Text-to-Image: find images matching a text query
// =====================================================================
console.log("=".repeat(60));
console.log("[A] Text-to-Image: query='a web framework logo'");
console.log("=".repeat(60));
const queryVecA = await embedText("a web framework logo");
printResults(await index.searchVector("content_vec", queryVecA, 3));

// =====================================================================
// [B] Text-to-Text: find text descriptions
// =====================================================================
console.log("\n" + "=".repeat(60));
console.log("[B] Text-to-Text: query='ORM database', filter type='text'");
console.log("=".repeat(60));
const reqB = new SearchRequest({ limit: 3 });
reqB.setVectorQuery(new VectorQuery("content_vec", await embedText("ORM database")));
reqB.setFilterTerm(new TermQuery("type", "text"));
printResults(await index.searchWithRequest(reqB));

// =====================================================================
// [C] Image-to-Anything: find documents similar to a given image
// =====================================================================
console.log("\n" + "=".repeat(60));
console.log("[C] Image-to-Anything: query from 'express_logo.png'");
console.log("=".repeat(60));
const queryImgBytes = fakeImage("express_logo.png");
const queryVecC = await embedImage(queryImgBytes);
printResults(await index.searchVector("content_vec", queryVecC, 3));

console.log("\nMultimodal search example completed!");

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
    const label = doc.filename || doc.description || "";
    const mediaType = doc.type || "?";
    console.log(
      `  id=${JSON.stringify(r.id).padEnd(8)}  score=${r.score.toFixed(4)}  [${mediaType}] ${JSON.stringify(label.slice(0, 55))}`,
    );
  }
}
