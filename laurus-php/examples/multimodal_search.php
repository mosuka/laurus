<?php

// Multimodal Search Example -- searching across text and image bytes fields.
//
// This example demonstrates how to store raw bytes (e.g. image data) in a
// Laurus index and perform multimodal search using pre-computed vectors.
//
// The Rust multimodal_search example uses the built-in CandleClipEmbedder.
// In PHP you can achieve the same result by:
// 1. Encoding images/text with an external CLIP model or embedding API.
// 2. Storing the raw bytes in a bytes field and the embedding vector in a flat field.
// 3. Querying with VectorQuery using a pre-computed embedding.
//
// This example uses random fallback vectors for portability (no external
// dependencies). For real embeddings, replace embed_text() and embed_image()
// with calls to your embedding API.
//
// Usage:
//   cd laurus-php
//   cargo build --release
//   php -d extension=target/release/liblaurus_php.so examples/multimodal_search.php

use Laurus\Index;
use Laurus\Schema;
use Laurus\SearchRequest;
use Laurus\TermQuery;
use Laurus\VectorQuery;

const DIM = 32;

// ---------------------------------------------------------------------------
// Embedding helpers (random fallback vectors for demo purposes)
// ---------------------------------------------------------------------------

/**
 * Generate a deterministic unit vector from a seed.
 *
 * @param int $seed Random seed value.
 * @return array<float> Normalized vector.
 */
function rand_unit(int $seed): array
{
    mt_srand($seed);
    $raw = [];
    for ($i = 0; $i < DIM; $i++) {
        $u1 = mt_rand(1, PHP_INT_MAX) / PHP_INT_MAX;
        $u2 = mt_rand(1, PHP_INT_MAX) / PHP_INT_MAX;
        $raw[] = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
    }
    $norm = sqrt(array_sum(array_map(fn($x) => $x * $x, $raw))) ?: 1.0;
    return array_map(fn($x) => $x / $norm, $raw);
}

/**
 * Generate a pseudo-embedding for text.
 * In production, replace with a real CLIP text encoder.
 */
function embed_text(string $text): array
{
    return rand_unit(crc32($text));
}

/**
 * Generate a pseudo-embedding for image bytes.
 * In production, replace with a real CLIP image encoder.
 */
function embed_image(string $imageBytes): array
{
    return rand_unit(crc32(substr($imageBytes, 0, 128)));
}

echo "[NOTE] Using random fallback vectors (dim=" . DIM . ")." . PHP_EOL;
echo "       Semantic similarity will NOT be meaningful." . PHP_EOL;
echo "       Replace embed_text()/embed_image() with real CLIP encoders in production." . PHP_EOL . PHP_EOL;

// ---------------------------------------------------------------------------
// Synthetic image bytes (1x1 white pixel PNG)
// ---------------------------------------------------------------------------

// Minimal valid PNG file for demo purposes.
const WHITE_PNG = "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    . "\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    . "\x00\x01\x01\x00\x05\x18\xd4n\x00\x00\x00\x00IEND\xaeB\x60\x82";

/**
 * Return placeholder image bytes (real app would read from disk).
 */
function fake_image(string $label): string
{
    return WHITE_PNG . $label;
}

// ---------------------------------------------------------------------------
// Dataset -- PHP ecosystem screenshots and descriptions
// ---------------------------------------------------------------------------

$images = [
    ["img_laravel_dashboard", "laravel_dashboard.png", "image"],
    ["img_symfony_profiler", "symfony_profiler.png", "image"],
    ["img_wordpress_admin", "wordpress_admin.png", "image"],
    ["img_phpunit_output", "phpunit_test_output.png", "image"],
];

$texts = [
    ["txt1", "Laravel Nova admin panel with resource management and metrics dashboard", "text"],
    ["txt2", "Symfony debug toolbar showing query profiling and request details", "text"],
    ["txt3", "WordPress Gutenberg editor composing a new blog post with blocks", "text"],
    ["txt4", "PHPUnit test runner output showing passing and failing test cases", "text"],
];

function print_results(array $results): void
{
    if (empty($results)) {
        echo "  (no results)" . PHP_EOL;
        return;
    }
    foreach ($results as $r) {
        $doc = $r->getDocument();
        $label = ($doc["filename"] ?? "") ?: ($doc["description"] ?? "");
        $mediaType = $doc["type"] ?? "?";
        printf("  id=%-28s  score=%.4f  [%s] '%s'\n", "'{$r->getId()}'", $r->getScore(), $mediaType, substr($label, 0, 55));
    }
}

echo "=== Laurus Multimodal Search Example ===" . PHP_EOL . PHP_EOL;

// -- Schema -----------------------------------------------------------------
$schema = new Schema();
$schema->addBytesField("content");
$schema->addTextField("filename");
$schema->addTextField("type");
$schema->addTextField("description");
$schema->addFlatField("content_vec", DIM, "cosine");

$index = new Index(null, $schema);

// -- Index images -----------------------------------------------------------
echo "--- Indexing images ---" . PHP_EOL;
foreach ($images as [$docId, $filename, $mediaType]) {
    $rawBytes = fake_image($filename);
    $vec = embed_image($rawBytes);
    $index->addDocument($docId, [
        "content" => $rawBytes,
        "filename" => $filename,
        "type" => $mediaType,
        "description" => "",
        "content_vec" => $vec,
    ]);
    echo "  Indexed image: " . $filename . PHP_EOL;
}

// -- Index text descriptions ------------------------------------------------
echo PHP_EOL . "--- Indexing text descriptions ---" . PHP_EOL;
foreach ($texts as [$docId, $text, $mediaType]) {
    $vec = embed_text($text);
    $index->addDocument($docId, [
        "content" => null,
        "filename" => "",
        "type" => $mediaType,
        "description" => $text,
        "content_vec" => $vec,
    ]);
    echo "  Indexed text: '" . substr($text, 0, 50) . "'" . PHP_EOL;
}

$index->commit();
echo PHP_EOL;

// =====================================================================
// [A] Text-to-Image: find images matching a text query
// =====================================================================
echo str_repeat("=", 60) . PHP_EOL;
echo "[A] Text-to-Image: query='Laravel admin panel'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$queryVec = embed_text("Laravel admin panel");
print_results($index->search(new VectorQuery("content_vec", $queryVec), 3)->getResults());

// =====================================================================
// [B] Text-to-Text: find text descriptions
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[B] Text-to-Text: query='test runner output', filter type='text'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    null,
    new VectorQuery("content_vec", embed_text("test runner output")),
    new TermQuery("type", "text"),
    null,
    3,
);
print_results($index->search($request)->getResults());

// =====================================================================
// [C] Image-to-Anything: find documents similar to a given image
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[C] Image-to-Anything: query from 'symfony_profiler.png'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$queryImgBytes = fake_image("symfony_profiler.png");
$queryVec = embed_image($queryImgBytes);
print_results($index->search(new VectorQuery("content_vec", $queryVec), 3)->getResults());

echo PHP_EOL . "Multimodal search example completed!" . PHP_EOL;
