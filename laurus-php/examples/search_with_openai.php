<?php

// Search with OpenAI Embedder -- real vector search using OpenAI's API.
//
// This example produces embeddings on the PHP side using raw curl calls to
// the OpenAI API, then passes the resulting vectors to Laurus as VectorQuery.
//
// Prerequisites:
//   export OPENAI_API_KEY=your-api-key-here
//
// Usage:
//   cd laurus-php
//   cargo build --release
//   php -d extension=target/release/liblaurus_php.so examples/search_with_openai.php

use Laurus\Index;
use Laurus\NumericRangeQuery;
use Laurus\RRF;
use Laurus\Schema;
use Laurus\SearchRequest;
use Laurus\TermQuery;
use Laurus\VectorQuery;

const OPENAI_MODEL = "text-embedding-3-small";
const DIM = 1536;

// ---------------------------------------------------------------------------
// OpenAI embedding helper (raw curl, no external packages required)
// ---------------------------------------------------------------------------

/**
 * Call the OpenAI Embeddings API and return the vector.
 *
 * @param string $apiKey OpenAI API key.
 * @param string $text   Input text to embed.
 * @return array<float>  Embedding vector.
 */
function embed(string $apiKey, string $text): array
{
    $ch = curl_init("https://api.openai.com/v1/embeddings");
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_POST => true,
        CURLOPT_HTTPHEADER => [
            "Content-Type: application/json",
            "Authorization: Bearer " . $apiKey,
        ],
        CURLOPT_POSTFIELDS => json_encode([
            "input" => $text,
            "model" => OPENAI_MODEL,
        ]),
    ]);

    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $curlError = curl_error($ch);
    curl_close($ch);

    if ($response === false) {
        fprintf(STDERR, "ERROR: curl request failed: %s\n", $curlError);
        exit(1);
    }

    $data = json_decode($response, true);
    if ($httpCode !== 200 || !isset($data["data"][0]["embedding"])) {
        $errorMsg = $data["error"]["message"] ?? "Unknown error (HTTP $httpCode)";
        fprintf(STDERR, "ERROR: OpenAI API error: %s\n", $errorMsg);
        exit(1);
    }

    return $data["data"][0]["embedding"];
}

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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

$apiKey = getenv("OPENAI_API_KEY");
if (!$apiKey) {
    fprintf(STDERR, "ERROR: OPENAI_API_KEY environment variable not set.\n");
    fprintf(STDERR, "       export OPENAI_API_KEY=your-api-key-here\n");
    exit(1);
}

echo "=== Laurus Search with OpenAI Embedder ===" . PHP_EOL . PHP_EOL;
echo "OpenAI embedder ready (model=" . OPENAI_MODEL . ", dim=" . DIM . ")" . PHP_EOL . PHP_EOL;

// -- Schema -----------------------------------------------------------------
$schema = new Schema();
$schema->addTextField("title");
$schema->addTextField("text");
$schema->addTextField("category");
$schema->addIntegerField("page");
$schema->addFlatField("text_vec", DIM, "cosine");
$schema->setDefaultFields(["text"]);

$index = new Index(null, $schema);

// -- Index ------------------------------------------------------------------
echo "--- Indexing chunked documents ---" . PHP_EOL . PHP_EOL;
foreach ($chunks as [$docId, $title, $text, $page, $category]) {
    $vec = embed($apiKey, $text);
    $index->addDocument($docId, [
        "title" => $title,
        "text" => $text,
        "category" => $category,
        "page" => $page,
        "text_vec" => $vec,
    ]);
    printf("  Indexed %s page %d: '%s'...\n", $docId, $page, substr($text, 0, 50));
}
$index->commit();
echo PHP_EOL . "Indexed " . count($chunks) . " chunks." . PHP_EOL . PHP_EOL;

// =====================================================================
// [A] Vector Search
// =====================================================================
echo str_repeat("=", 60) . PHP_EOL;
echo "[A] Vector Search: 'database ORM queries'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
print_results($index->search(new VectorQuery("text_vec", embed($apiKey, "database ORM queries")), 3)->getResults());

// =====================================================================
// [B] Filtered Vector Search -- category filter
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[B] Filtered Vector Search: 'database ORM queries' + category='testing'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    null,
    new VectorQuery("text_vec", embed($apiKey, "database ORM queries")),
    new TermQuery("category", "testing"),
    null,
    3,
);
print_results($index->search($request)->getResults());

// =====================================================================
// [C] Filtered Vector Search -- numeric range filter
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[C] Filtered Vector Search: 'HTTP web server' + page=1" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    null,
    new VectorQuery("text_vec", embed($apiKey, "HTTP web server")),
    new NumericRangeQuery("page", 1, 1),
    null,
    3,
);
print_results($index->search($request)->getResults());

// =====================================================================
// [D] Lexical Search
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[D] Lexical Search: 'hooks'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
print_results($index->search(new TermQuery("text", "hooks"), 3)->getResults());

// =====================================================================
// [E] Hybrid Search (RRF)
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "[E] Hybrid Search (RRF): vector='template rendering' + lexical='blade'" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;
$request = new SearchRequest(
    null,
    new TermQuery("text", "blade"),
    new VectorQuery("text_vec", embed($apiKey, "template rendering")),
    null,
    new RRF(60.0),
    3,
);
print_results($index->search($request)->getResults());

echo PHP_EOL . "Search with OpenAI example completed!" . PHP_EOL;
