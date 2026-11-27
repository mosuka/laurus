<?php

// Vector Search Example — semantic similarity search with embeddings.
//
// Demonstrates vector search using laurus's built-in CandleBert embedder:
// - Basic vector search (semantic similarity)
// - Filtered vector search (with lexical filters)
//
// The embedder is registered in the schema and laurus automatically converts
// text to vectors at index and query time — no external embedding library needed.
//
// Usage:
//   cd laurus-php
//   cargo build --release --features embeddings-candle
//   php -d extension=target/release/liblaurus_php.so examples/vector_search.php

use Laurus\Index;
use Laurus\NumericRangeQuery;
use Laurus\Schema;
use Laurus\SearchRequest;
use Laurus\TermQuery;
use Laurus\VectorTextQuery;

const EMBEDDER_NAME = "bert";
const EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2";
const DIM = 384; // dimension for all-MiniLM-L6-v2

// PHP ecosystem documentation chunks — simulating a RAG pattern.
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

echo "=== Laurus Vector Search Example ===" . PHP_EOL . PHP_EOL;
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
// [A] Basic Vector Search
// =====================================================================
echo str_repeat("=", 60) . PHP_EOL;
echo "[A] Basic Vector Search: 'database ORM queries'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
print_results($index->search(new VectorTextQuery("text_vec", "database ORM queries"), 3)->getResults());

// =====================================================================
// [B] Filtered Vector Search — category filter
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[B] Filtered Vector Search: 'testing' + category='testing'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    null,
    new VectorTextQuery("text_vec", "testing and mocking"),
    new TermQuery("category", "testing"),
    null,
    3,
);
print_results($index->search($request)->getResults());

// =====================================================================
// [C] Filtered Vector Search — numeric range filter (page = 1)
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[C] Filtered Vector Search: 'HTTP web server' + page=1" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    null,
    new VectorTextQuery("text_vec", "HTTP web server"),
    new NumericRangeQuery("page", 1, 1),
    null,
    3,
);
print_results($index->search($request)->getResults());

echo PHP_EOL . "Vector search example completed!" . PHP_EOL;
