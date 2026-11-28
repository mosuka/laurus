<?php

// Hybrid Search Example — combining lexical and vector search.
//
// Demonstrates:
// - Lexical-only search (for comparison)
// - Vector-only search (for comparison)
// - Hybrid search with RRF fusion
// - Hybrid search with WeightedSum fusion
// - Hybrid search with a filter query
//
// The embedder is registered in the schema and laurus automatically converts
// text to vectors at index and query time — no external embedding library needed.
//
// Usage:
//   cd laurus-php
//   cargo build --release --features embeddings-candle
//   php -d extension=target/release/liblaurus_php.so examples/hybrid_search.php

use Laurus\Index;
use Laurus\RRF;
use Laurus\Schema;
use Laurus\SearchRequest;
use Laurus\TermQuery;
use Laurus\VectorTextQuery;
use Laurus\WeightedSum;

const EMBEDDER_NAME = "bert";
const EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2";
const DIM = 384;

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

echo "=== Laurus Hybrid Search Example ===" . PHP_EOL . PHP_EOL;
echo "Embedder: " . EMBEDDER_MODEL . " (dim=" . DIM . ")" . PHP_EOL . PHP_EOL;

// ── Schema ─────────────────────────────────────────────────────────────
$schema = new Schema();
$schema->addEmbedder(EMBEDDER_NAME, ["type" => "candle_bert", "model" => EMBEDDER_MODEL]);
$schema->addTextField("title");
$schema->addTextField("text");
$schema->addTextField("category");
$schema->addIntegerField("page");
$schema->addFlatField("text_vec", DIM, "cosine", EMBEDDER_NAME);
$schema->setDefaultFields(["text"]);

$index = new Index(null, $schema);

// ── Index ──────────────────────────────────────────────────────────────
echo "--- Indexing chunked documents ---" . PHP_EOL . PHP_EOL;
foreach ($chunks as [$docId, $title, $text, $page, $category]) {
    $index->addDocument($docId, [
        "title" => $title,
        "text" => $text,
        "category" => $category,
        "page" => $page,
        "text_vec" => $text,
    ]);
}
$index->commit();
echo "Indexed " . count($chunks) . " chunks." . PHP_EOL . PHP_EOL;

// =====================================================================
// [A] Lexical-only search (baseline)
// =====================================================================
echo str_repeat("=", 60) . PHP_EOL;
echo "[A] Lexical-only: term 'eloquent' in text" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
print_results($index->search(new TermQuery("text", "eloquent"), 3)->getResults());

// =====================================================================
// [B] Vector-only search (baseline)
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[B] Vector-only: 'database query builder'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
print_results($index->search(new VectorTextQuery("text_vec", "database query builder"), 3)->getResults());

// =====================================================================
// [C] Hybrid search — RRF Fusion
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[C] Hybrid (RRF k=60): vector='template rendering' + lexical='blade'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    new TermQuery("text", "blade"),
    new VectorTextQuery("text_vec", "template rendering"),
    null,
    new RRF(60.0),
    3,
);
print_results($index->search($request)->getResults());

// =====================================================================
// [D] Hybrid search — WeightedSum Fusion
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[D] Hybrid (WeightedSum 0.3/0.7): vector='API development' + lexical='rest'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    new TermQuery("text", "rest"),
    new VectorTextQuery("text_vec", "API development"),
    null,
    new WeightedSum(0.3, 0.7),
    3,
);
print_results($index->search($request)->getResults());

// =====================================================================
// [E] Hybrid search with filter query
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[E] Hybrid + filter: vector='dependency management' + lexical='injection' + category='framework'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    new TermQuery("text", "injection"),
    new VectorTextQuery("text_vec", "dependency management"),
    new TermQuery("category", "framework"),
    new RRF(60.0),
    3,
);
print_results($index->search($request)->getResults());

// =====================================================================
// [F] DSL string query
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[F] DSL query string: 'text:hooks'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
print_results($index->search("text:hooks", 3)->getResults());

echo PHP_EOL . "Hybrid search example completed!" . PHP_EOL;
