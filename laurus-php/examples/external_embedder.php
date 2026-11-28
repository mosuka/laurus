<?php

// External Embedder Example -- vector and hybrid search with user-managed embeddings.
//
// Demonstrates vector and hybrid search where embeddings are produced outside
// laurus and passed as pre-computed vectors via VectorQuery.
//
// This approach is useful when you want to:
// - Use any embedding library or external API
// - Control the embedding model independently of the index schema
// - Reuse embeddings across multiple indexes
//
// For an alternative that lets laurus handle embeddings automatically, see
// vector_search.php and hybrid_search.php which use the built-in
// CandleBert embedder via $schema->addEmbedder().
//
// Usage:
//   cd laurus-php
//   cargo build --release
//   php -d extension=target/release/liblaurus_php.so examples/external_embedder.php

use Laurus\Index;
use Laurus\RRF;
use Laurus\Schema;
use Laurus\SearchRequest;
use Laurus\TermQuery;
use Laurus\VectorQuery;
use Laurus\WeightedSum;

// ---------------------------------------------------------------------------
// Embedding helper
// ---------------------------------------------------------------------------
// PHP does not have a convenient sentence-transformers library.
// This example uses deterministic pseudo-embeddings as a fallback.
// In production, replace embed() with a call to your embedding API
// (e.g. OpenAI, Cohere) -- see search_with_openai.php for an example.

const DIM = 64;

/**
 * Generate a deterministic pseudo-embedding vector for demo purposes.
 *
 * Real similarity is NOT meaningful with these vectors.
 * Replace with a real embedding API call in production.
 */
function embed(string $text): array
{
    // Use a seeded RNG based on the text hash for reproducibility.
    mt_srand(crc32($text));
    $raw = [];
    for ($i = 0; $i < DIM; $i++) {
        // Box-Muller transform for approximate Gaussian distribution.
        $u1 = mt_rand(1, PHP_INT_MAX) / PHP_INT_MAX;
        $u2 = mt_rand(1, PHP_INT_MAX) / PHP_INT_MAX;
        $raw[] = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
    }
    // L2 normalize.
    $norm = sqrt(array_sum(array_map(fn($x) => $x * $x, $raw))) ?: 1.0;
    return array_map(fn($x) => $x / $norm, $raw);
}

echo "[NOTE] Using random fallback vectors (dim=" . DIM . ")." . PHP_EOL;
echo "       Results will NOT reflect semantic similarity." . PHP_EOL;
echo "       For real embeddings, see search_with_openai.php." . PHP_EOL . PHP_EOL;

// ---------------------------------------------------------------------------
// Dataset -- PHP ecosystem documentation chunks
// ---------------------------------------------------------------------------

$chunks = [
    ["laravel", "Laravel Framework", "Laravel provides an elegant MVC architecture with Eloquent ORM for database operations.", 1, "framework"],
    ["laravel", "Laravel Framework", "Blade is Laravel's templating engine that compiles to plain PHP for maximum performance.", 2, "framework"],
    ["laravel", "Laravel Framework", "Laravel Artisan CLI provides commands for migrations, seeding, and code generation.", 3, "framework"],
    ["symfony", "Symfony Components", "Symfony's dependency injection container manages service instantiation and configuration.", 1, "framework"],
    ["symfony", "Symfony Components", "Symfony's HttpFoundation component provides an object-oriented layer for HTTP requests and responses.", 2, "framework"],
    ["wordpress", "WordPress Internals", "WordPress uses hooks (actions and filters) to allow plugins to modify core behavior without editing source.", 1, "cms"],
    ["wordpress", "WordPress Internals", "The WordPress REST API enables headless CMS architectures with JSON endpoints.", 2, "cms"],
    ["phpunit", "PHPUnit Testing", "PHPUnit data providers allow running the same test with different input datasets automatically.", 1, "testing"],
    ["phpunit", "PHPUnit Testing", "Mock objects in PHPUnit simulate dependencies to isolate the unit under test.", 2, "testing"],
];

function print_results(array $results): void
{
    if (empty($results)) {
        echo "  (no results)" . PHP_EOL;
        return;
    }
    foreach ($results as $r) {
        $doc = $r->getDocument();
        $text = substr($doc["text"] ?? "", 0, 70);
        printf("  id=%-12s  score=%.4f  text='%s'\n", "'{$r->getId()}'", $r->getScore(), $text);
    }
}

echo "=== Laurus External Embedder Example ===" . PHP_EOL . PHP_EOL;

// -- Schema -----------------------------------------------------------------
// No embedder is registered in the schema; vectors are provided by the
// caller at index and query time via embed().
$schema = new Schema();
$schema->addTextField("title");
$schema->addTextField("text");
$schema->addTextField("category");
$schema->addIntegerField("page");
$schema->addFlatField("text_vec", DIM, "cosine");
$schema->setDefaultFields(["text"]);

$index = new Index(null, $schema);

// -- Index ------------------------------------------------------------------
// Embeddings are computed here in PHP and stored as float vectors.
echo "--- Indexing chunked documents ---" . PHP_EOL . PHP_EOL;
foreach ($chunks as [$docId, $title, $text, $page, $category]) {
    $index->addDocument($docId, [
        "title" => $title,
        "text" => $text,
        "category" => $category,
        "page" => $page,
        "text_vec" => embed($text),
    ]);
}
$index->commit();
echo "Indexed " . count($chunks) . " chunks." . PHP_EOL . PHP_EOL;

// =====================================================================
// [A] Basic Vector Search
// =====================================================================
echo str_repeat("=", 60) . PHP_EOL;
echo "[A] Vector-only: 'database ORM queries'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
print_results($index->search(new VectorQuery("text_vec", embed("database ORM queries")), 3)->getResults());

// =====================================================================
// [B] Filtered Vector Search -- category filter
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[B] Filtered vector: 'database ORM queries' + category='testing'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    null,
    new VectorQuery("text_vec", embed("database ORM queries")),
    new TermQuery("category", "testing"),
    null,
    3,
);
print_results($index->search($request)->getResults());

// =====================================================================
// [C] Hybrid search -- RRF Fusion
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[C] Hybrid (RRF k=60): vector='template rendering' + lexical='blade'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    new TermQuery("text", "blade"),
    new VectorQuery("text_vec", embed("template rendering")),
    null,
    new RRF(60.0),
    3,
);
print_results($index->search($request)->getResults());

// =====================================================================
// [D] Hybrid search -- WeightedSum Fusion
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[D] Hybrid (WeightedSum 0.3/0.7): vector='REST API development' + lexical='rest'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    new TermQuery("text", "rest"),
    new VectorQuery("text_vec", embed("REST API development")),
    null,
    new WeightedSum(0.3, 0.7),
    3,
);
print_results($index->search($request)->getResults());

// =====================================================================
// [E] Hybrid search with filter
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[E] Hybrid + filter: vector='dependency management' + lexical='injection' + category='framework'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    new TermQuery("text", "injection"),
    new VectorQuery("text_vec", embed("dependency management")),
    new TermQuery("category", "framework"),
    new RRF(60.0),
    3,
);
print_results($index->search($request)->getResults());

echo PHP_EOL . "External embedder example completed!" . PHP_EOL;
