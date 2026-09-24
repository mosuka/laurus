<?php

// Search App Example — Unified Query DSL search interface.
//
// A self-contained web application that demonstrates lexical, vector, and
// hybrid search through the Unified Query DSL. One input field handles all
// search modes:
//
//   Lexical:  php              title:laravel       "dependency injection"
//   Vector:   body_vec:"template engine rendering"
//   Hybrid:   testing body_vec:"automated quality"
//
// Uses PHP's built-in web server — no framework required.
//
// Usage:
//   cd laurus-php
//   cargo build --release --features embeddings-candle
//   php -d extension=target/release/liblaurus_php.so -S localhost:8080 examples/search_app.php
//
// Then open http://localhost:8080 in your browser.

use Laurus\Index;
use Laurus\Schema;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EMBEDDER_NAME = "bert";
const EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2";
const DIM = 384;

// ---------------------------------------------------------------------------
// Index setup (in-memory, rebuilt on each request for simplicity)
// ---------------------------------------------------------------------------

function create_index(): Index
{
    $schema = new Schema();
    $schema->addEmbedder(EMBEDDER_NAME, ["type" => "candle_bert", "model" => EMBEDDER_MODEL]);
    $schema->addTextField("title");
    $schema->addTextField("body");
    $schema->addTextField("category", true, true, false, true, "keyword");
    $schema->addIntegerField("year");
    $schema->addFlatField("body_vec", DIM, "cosine", EMBEDDER_NAME);
    $schema->setDefaultFields(["title", "body"]);

    $index = new Index(null, $schema);

    $docs = [
        ["laravel", [
            "title" => "Laravel Framework",
            "body" => "Laravel is an elegant PHP web application framework with expressive syntax. It provides Eloquent ORM, Blade templating, and Artisan CLI for rapid development.",
            "category" => "Framework",
            "year" => 2011,
        ]],
        ["symfony", [
            "title" => "Symfony Components",
            "body" => "Symfony is a set of reusable PHP components and a web framework. Its dependency injection container and HttpFoundation are used by many other projects including Laravel and Drupal.",
            "category" => "Framework",
            "year" => 2005,
        ]],
        ["wordpress", [
            "title" => "WordPress CMS",
            "body" => "WordPress powers over 40 percent of the web. Its plugin and theme architecture with hooks (actions and filters) allows extending core functionality. The REST API enables headless CMS patterns.",
            "category" => "CMS",
            "year" => 2003,
        ]],
        ["composer", [
            "title" => "Composer Package Manager",
            "body" => "Composer is the dependency management tool for PHP. It resolves package versions, manages autoloading via PSR-4, and integrates with Packagist, the main PHP package repository.",
            "category" => "Tooling",
            "year" => 2012,
        ]],
        ["phpunit", [
            "title" => "PHPUnit Testing Framework",
            "body" => "PHPUnit is the standard testing framework for PHP. It supports unit tests, integration tests, data providers, mock objects, and code coverage analysis for quality assurance.",
            "category" => "Testing",
            "year" => 2004,
        ]],
        ["phpstan", [
            "title" => "PHPStan Static Analysis",
            "body" => "PHPStan finds bugs in PHP code without running it. It performs static type analysis, detects dead code, and enforces strict typing rules at multiple strictness levels.",
            "category" => "Tooling",
            "year" => 2016,
        ]],
        ["drupal", [
            "title" => "Drupal CMS",
            "body" => "Drupal is an enterprise content management system built on Symfony components. It provides robust content modeling, multilingual support, and granular access control.",
            "category" => "CMS",
            "year" => 2001,
        ]],
        ["slim", [
            "title" => "Slim Micro Framework",
            "body" => "Slim is a lightweight PHP micro framework for building APIs and small web applications. It focuses on routing, middleware, and PSR-7 HTTP message interfaces.",
            "category" => "Framework",
            "year" => 2010,
        ]],
        ["pest", [
            "title" => "Pest Testing Framework",
            "body" => "Pest is a modern PHP testing framework with an elegant syntax inspired by Jest. Built on top of PHPUnit, it adds expressive assertions, higher-order tests, and architecture testing.",
            "category" => "Testing",
            "year" => 2020,
        ]],
        ["rector", [
            "title" => "Rector Automated Refactoring",
            "body" => "Rector is an automated refactoring tool for PHP. It upgrades legacy code, enforces coding standards, and migrates between framework versions using AST transformations.",
            "category" => "Tooling",
            "year" => 2017,
        ]],
    ];

    foreach ($docs as [$id, $doc]) {
        $doc["body_vec"] = $doc["body"];
        $index->putDocument($id, $doc);
    }
    $index->commit();

    return $index;
}

// ---------------------------------------------------------------------------
// Handle request
// ---------------------------------------------------------------------------

$query = trim($_GET["q"] ?? "");
$results = [];
$elapsed = 0;

$index = create_index();
$stats = $index->stats();

if ($query !== "") {
    $start = microtime(true);
    $results = $index->search($query, 10);
    $elapsed = (microtime(true) - $start) * 1000;
}

// Detect query type for display
function detect_query_type(string $query): string
{
    $hasVector = preg_match('/body_vec:/', $query);
    $hasLexical = preg_match('/(?:^|\s)(?!body_vec:)\S/', $query);
    if ($hasVector && $hasLexical) {
        return "hybrid";
    }
    if ($hasVector) {
        return "vector";
    }
    return "lexical";
}

// ---------------------------------------------------------------------------
// Render HTML
// ---------------------------------------------------------------------------

$categoryColors = [
    "Framework" => "#e8f0fe",
    "CMS" => "#fce8e6",
    "Tooling" => "#e6f4ea",
    "Testing" => "#fef7e0",
];

?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laurus PHP Search</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; }
        .container { max-width: 760px; margin: 0 auto; padding: 24px 16px; }
        header { text-align: center; margin-bottom: 24px; }
        header h1 { font-size: 28px; font-weight: 600; margin-bottom: 4px; }
        header p { color: #666; font-size: 14px; }
        form { margin-bottom: 24px; }
        .search-row { display: flex; gap: 8px; margin-bottom: 10px; }
        .search-row input[type="text"] {
            flex: 1; padding: 10px 14px; font-size: 16px; border: 1px solid #ccc; border-radius: 6px;
            outline: none; transition: border-color 0.2s;
        }
        .search-row input[type="text"]:focus { border-color: #4a90d9; }
        .search-row button {
            padding: 10px 20px; font-size: 16px; background: #4a90d9; color: #fff; border: none;
            border-radius: 6px; cursor: pointer; transition: background 0.2s;
        }
        .search-row button:hover { background: #357abd; }
        .dsl-hint { font-size: 12px; color: #888; line-height: 1.7; }
        .dsl-hint code { background: #e8e8e8; padding: 2px 6px; border-radius: 3px; font-size: 12px; }
        .meta { color: #666; font-size: 13px; margin-bottom: 16px; }
        .meta .mode-label {
            display: inline-block; padding: 1px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 600; text-transform: uppercase;
        }
        .mode-lexical { background: #e8f0fe; color: #3b78de; }
        .mode-vector { background: #e6f4ea; color: #2e7d4f; }
        .mode-hybrid { background: #fef7e0; color: #b8860b; }
        .result { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
        .result-title { font-size: 18px; font-weight: 600; color: #1a0dab; margin-bottom: 4px; }
        .result-meta { font-size: 12px; color: #888; margin-bottom: 8px; }
        .result-body { font-size: 14px; color: #555; line-height: 1.5; }
        .result-tags { margin-top: 8px; }
        .tag { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 12px; margin-right: 4px; }
        .no-results { text-align: center; color: #999; padding: 40px 0; font-size: 16px; }
        .hints { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; }
        .hints h3 { font-size: 14px; font-weight: 600; margin-bottom: 8px; }
        .hints code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 13px; }
        .hints ul { list-style: none; padding: 0; }
        .hints li { font-size: 13px; color: #555; padding: 3px 0; }
        .hints .section { margin-top: 14px; }
        footer { text-align: center; color: #999; font-size: 12px; margin-top: 32px; }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Laurus PHP Search</h1>
        <p>Unified Query DSL -- <?= $stats["documentCount"] ?> PHP ecosystem documents indexed</p>
    </header>

    <form method="GET">
        <div class="search-row">
            <input type="text" name="q" value="<?= htmlspecialchars($query) ?>" placeholder='Try: php, body_vec:"template engine", or both!' autofocus>
            <button type="submit">Search</button>
        </div>
        <div class="dsl-hint">
            Lexical: <code>php</code> <code>title:laravel</code> <code>"dependency injection"</code> &nbsp;
            Vector: <code>body_vec:"template engine rendering"</code> &nbsp;
            Hybrid: <code>testing body_vec:"automated quality"</code>
        </div>
    </form>

<?php if ($query !== ""):
    $qtype = detect_query_type($query);
?>
    <div class="meta">
        <span class="mode-label mode-<?= $qtype ?>"><?= $qtype ?></span>
        <?= count($results) ?> result(s) for "<?= htmlspecialchars($query) ?>" (<?= number_format($elapsed, 1) ?> ms)
    </div>

    <?php if (empty($results)): ?>
        <div class="no-results">No results found.</div>
    <?php else: ?>
        <?php foreach ($results as $r):
            $doc = $r->getDocument();
            $cat = $doc["category"] ?? "";
            $bgColor = $categoryColors[$cat] ?? "#f0f0f0";
        ?>
        <div class="result">
            <div class="result-title"><?= htmlspecialchars($doc["title"] ?? $r->getId()) ?></div>
            <div class="result-meta">
                id: <?= htmlspecialchars($r->getId()) ?> | score: <?= number_format($r->getScore(), 4) ?>
                <?php if (isset($doc["year"])): ?> | <?= (int)$doc["year"] ?><?php endif; ?>
            </div>
            <div class="result-body"><?= htmlspecialchars($doc["body"] ?? "") ?></div>
            <?php if ($cat): ?>
            <div class="result-tags"><span class="tag" style="background:<?= $bgColor ?>"><?= htmlspecialchars($cat) ?></span></div>
            <?php endif; ?>
        </div>
        <?php endforeach; ?>
    <?php endif; ?>

<?php else: ?>
    <div class="hints">
        <h3>Unified Query DSL</h3>
        <p style="font-size:13px;color:#555;margin-bottom:10px;">One input handles lexical, vector, and hybrid search.</p>
        <div class="section">
            <h3>Lexical search (keyword matching)</h3>
            <ul>
                <li><code>php</code> -- search default fields (title, body)</li>
                <li><code>title:laravel</code> -- search a specific field</li>
                <li><code>"dependency injection"</code> -- phrase search</li>
                <li><code>+body:php -body:wordpress</code> -- boolean (must / must not)</li>
                <li><code>laravl~2</code> -- fuzzy search (typo tolerance)</li>
                <li><code>body:php*</code> -- wildcard search</li>
                <li><code>year:[2015 TO 2023]</code> -- numeric range</li>
            </ul>
        </div>
        <div class="section">
            <h3>Vector search (semantic similarity)</h3>
            <ul>
                <li><code>body_vec:"template engine rendering"</code> -- find semantically similar documents</li>
                <li><code>body_vec:"automated code quality tools"</code> -- meaning-based, not keyword-based</li>
            </ul>
        </div>
        <div class="section">
            <h3>Hybrid search (lexical + vector)</h3>
            <ul>
                <li><code>testing body_vec:"automated quality"</code> -- combine both for best results</li>
                <li><code>title:laravel body_vec:"web framework features"</code> -- field-specific + semantic</li>
            </ul>
        </div>
    </div>
<?php endif; ?>

    <footer>
        Powered by laurus-php (ext-php-rs) | CandleBert embedder (<?= DIM ?>-dim) | PHP <?= PHP_VERSION ?>
    </footer>
</div>
</body>
</html>
