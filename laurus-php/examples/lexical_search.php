<?php

// Lexical Search Example — all query types.
//
// Demonstrates every lexical query type Laurus supports:
//
// 1. TermQuery         — exact single-term matching
// 2. PhraseQuery       — exact word sequence matching
// 3. FuzzyQuery        — approximate matching (typo tolerance)
// 4. WildcardQuery     — pattern matching with * and ?
// 5. NumericRangeQuery — numeric range filtering (int and float)
// 6. GeoDistanceQuery / GeoBoundingBoxQuery — geographic radius / bounding box
// 7. BooleanQuery      — AND / OR / NOT combinations
// 8. SpanQuery         — positional / proximity search
//
// Usage:
//   cd laurus-php
//   cargo build --release
//   php -d extension=target/release/liblaurus_php.so examples/lexical_search.php

use Laurus\BooleanQuery;
use Laurus\FuzzyQuery;
use Laurus\GeoBoundingBoxQuery;
use Laurus\GeoDistanceQuery;
use Laurus\Index;
use Laurus\NumericRangeQuery;
use Laurus\PhraseQuery;
use Laurus\Schema;
use Laurus\SpanQuery;
use Laurus\TermQuery;
use Laurus\WildcardQuery;

function print_results(array $results): void
{
    if (empty($results)) {
        echo "  (no results)" . PHP_EOL;
        return;
    }
    foreach ($results as $r) {
        $doc = $r->getDocument();
        $title = $doc["title"] ?? "";
        printf("  id=%-10s  score=%.4f  title=%s\n", "'{$r->getId()}'", $r->getScore(), "'{$title}'");
    }
}

echo "=== Laurus Lexical Search Example ===" . PHP_EOL . PHP_EOL;

// ── Setup ──────────────────────────────────────────────────────────────
$schema = new Schema();
$schema->addTextField("title");
$schema->addTextField("body");
$schema->addTextField("category", true, true, false, true, "keyword");
$schema->addTextField("filename", true, true, false, true, "keyword");
$schema->addBooleanField("maintained");
$schema->addFloatField("rating");
$schema->addIntegerField("year");
$schema->addGeoField("location");
$schema->setDefaultFields(["body"]);

$index = new Index(null, $schema);

// ── Index documents ────────────────────────────────────────────────────
$docs = [
    ["laravel", [
        "title" => "Laravel Framework",
        "body" => "Laravel is an elegant PHP web application framework with expressive syntax and rich ecosystem",
        "category" => "framework",
        "filename" => "laravel_guide.pdf",
        "maintained" => true,
        "rating" => 4.8,
        "year" => 2011,
        "location" => ["lat" => 37.7749, "lon" => -122.4194], // San Francisco
    ]],
    ["symfony", [
        "title" => "Symfony Components",
        "body" => "Symfony is a set of reusable PHP components and a web application framework used by Laravel and Drupal",
        "category" => "framework",
        "filename" => "symfony_docs.epub",
        "maintained" => true,
        "rating" => 4.6,
        "year" => 2005,
        "location" => ["lat" => 48.8566, "lon" => 2.3522], // Paris
    ]],
    ["wordpress", [
        "title" => "WordPress Development",
        "body" => "WordPress powers over 40 percent of the web with its flexible plugin and theme architecture",
        "category" => "cms",
        "filename" => "wordpress_dev.pdf",
        "maintained" => true,
        "rating" => 4.2,
        "year" => 2003,
        "location" => ["lat" => 37.7749, "lon" => -122.4194], // San Francisco
    ]],
    ["phpunit", [
        "title" => "PHPUnit Testing Guide",
        "body" => "PHPUnit is the standard testing framework for PHP applications with assertions and mocking support",
        "category" => "testing",
        "filename" => "phpunit_manual.docx",
        "maintained" => true,
        "rating" => 4.5,
        "year" => 2004,
        "location" => ["lat" => 52.5200, "lon" => 13.4050], // Berlin
    ]],
    ["composer", [
        "title" => "Composer Package Manager",
        "body" => "Composer is a dependency management tool for PHP that manages libraries and autoloading",
        "category" => "tooling",
        "filename" => "composer_docs.pdf",
        "maintained" => true,
        "rating" => 4.7,
        "year" => 2012,
        "location" => ["lat" => 52.3676, "lon" => 4.9041], // Amsterdam
    ]],
    ["pear", [
        "title" => "PEAR Package Repository",
        "body" => "PEAR was the original PHP extension and application repository before Composer replaced it",
        "category" => "tooling",
        "filename" => "pear_archive.txt",
        "maintained" => false,
        "rating" => 2.5,
        "year" => 1999,
        "location" => ["lat" => 51.5074, "lon" => -0.1278], // London
    ]],
];

echo "  Indexing " . count($docs) . " documents..." . PHP_EOL;
foreach ($docs as [$docId, $doc]) {
    $index->addDocument($docId, $doc);
}
$index->commit();
echo "  Done." . PHP_EOL . PHP_EOL;

// =====================================================================
// PART 1: TermQuery — exact single-term matching
// =====================================================================
echo str_repeat("=", 60) . PHP_EOL;
echo "PART 1: TermQuery" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;

echo PHP_EOL . "[1a] Search for 'php' in body:" . PHP_EOL;
print_results($index->search(new TermQuery("body", "php"), 5));

echo PHP_EOL . "[1b] Search for 'framework' in category (exact):" . PHP_EOL;
print_results($index->search(new TermQuery("category", "framework"), 5));

echo PHP_EOL . "[1c] Search for maintained=true (boolean field):" . PHP_EOL;
print_results($index->search(new TermQuery("maintained", "true"), 5));

echo PHP_EOL . "[1d] DSL: 'body:php':" . PHP_EOL;
print_results($index->search("body:php", 5));

// =====================================================================
// PART 2: PhraseQuery — exact word sequence
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "PART 2: PhraseQuery" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;

echo PHP_EOL . "[2a] Phrase 'web application framework' in body:" . PHP_EOL;
print_results($index->search(new PhraseQuery("body", ["web", "application", "framework"]), 5));

echo PHP_EOL . "[2b] Phrase 'dependency management' in body:" . PHP_EOL;
print_results($index->search(new PhraseQuery("body", ["dependency", "management"]), 5));

echo PHP_EOL . "[2c] DSL: 'body:\"web application framework\"':" . PHP_EOL;
print_results($index->search('body:"web application framework"', 5));

// =====================================================================
// PART 3: FuzzyQuery — approximate matching (typo tolerance)
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "PART 3: FuzzyQuery" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;

echo PHP_EOL . "[3a] Fuzzy 'laravl' (missing 'e', max_edits=2):" . PHP_EOL;
print_results($index->search(new FuzzyQuery("body", "laravl", 2), 5));

echo PHP_EOL . "[3b] Fuzzy 'sympfony' (extra 'p', max_edits=1):" . PHP_EOL;
print_results($index->search(new FuzzyQuery("body", "sympfony", 1), 5));

echo PHP_EOL . "[3c] DSL: 'laravl~2':" . PHP_EOL;
print_results($index->search("laravl~2", 5));

// =====================================================================
// PART 4: WildcardQuery — pattern matching with * and ?
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "PART 4: WildcardQuery" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;

echo PHP_EOL . "[4a] Wildcard '*.pdf' in filename:" . PHP_EOL;
print_results($index->search(new WildcardQuery("filename", "*.pdf"), 5));

echo PHP_EOL . "[4b] Wildcard 'php*' in body:" . PHP_EOL;
print_results($index->search(new WildcardQuery("body", "php*"), 5));

echo PHP_EOL . "[4c] DSL: 'body:php*':" . PHP_EOL;
print_results($index->search("body:php*", 5));

// =====================================================================
// PART 5: NumericRangeQuery — numeric range filtering
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "PART 5: NumericRangeQuery" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;

echo PHP_EOL . "[5a] Rating 4.5-5.0 (float range):" . PHP_EOL;
print_results($index->search(new NumericRangeQuery("rating", 4.5, 5.0, "float"), 5));

echo PHP_EOL . "[5b] Projects started from 2010 onwards (integer range):" . PHP_EOL;
print_results($index->search(new NumericRangeQuery("year", 2010, null), 5));

echo PHP_EOL . "[5c] DSL: 'rating:[4.5 TO 5.0]':" . PHP_EOL;
print_results($index->search("rating:[4.5 TO 5.0]", 5));

// =====================================================================
// PART 6: GeoDistanceQuery / GeoBoundingBoxQuery — geographic search
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "PART 6: GeoDistanceQuery / GeoBoundingBoxQuery" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;

echo PHP_EOL . "[6a] Within 100 km of Paris (48.86, 2.35):" . PHP_EOL;
print_results($index->search(GeoDistanceQuery::withinRadius("location", 48.8566, 2.3522, 100_000.0), 5));

echo PHP_EOL . "[6b] Bounding box — Europe (47, -1) to (53, 14):" . PHP_EOL;
print_results($index->search(GeoBoundingBoxQuery::withinBoundingBox("location", 47.0, -1.0, 53.0, 14.0), 5));

// =====================================================================
// PART 7: BooleanQuery — AND / OR / NOT combinations
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "PART 7: BooleanQuery" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;

echo PHP_EOL . "[7a] AND: 'php' in body AND category='framework':" . PHP_EOL;
$bq = new BooleanQuery();
$bq->must(new TermQuery("body", "php"));
$bq->must(new TermQuery("category", "framework"));
print_results($index->search($bq, 5));

echo PHP_EOL . "[7b] OR: category='framework' OR category='cms':" . PHP_EOL;
$bq = new BooleanQuery();
$bq->should(new TermQuery("category", "framework"));
$bq->should(new TermQuery("category", "cms"));
print_results($index->search($bq, 5));

echo PHP_EOL . "[7c] NOT: 'php' in body, NOT 'composer':" . PHP_EOL;
$bq = new BooleanQuery();
$bq->must(new TermQuery("body", "php"));
$bq->mustNot(new TermQuery("body", "composer"));
print_results($index->search($bq, 5));

echo PHP_EOL . "[7d] DSL: '+body:php -body:composer':" . PHP_EOL;
print_results($index->search("+body:php -body:composer", 5));

// =====================================================================
// PART 8: SpanQuery — positional / proximity search
// =====================================================================
echo PHP_EOL . str_repeat("=", 60) . PHP_EOL;
echo "PART 8: SpanQuery (no DSL equivalent)" . PHP_EOL;
echo str_repeat("=", 60) . PHP_EOL;

echo PHP_EOL . "[8a] SpanNear: 'plugin' near 'theme' (slop=2, ordered):" . PHP_EOL;
$spanQ = SpanQuery::near("body", ["plugin", "theme"], 2, true);
print_results($index->search($spanQ, 5));

echo PHP_EOL . "[8b] SpanContaining: 'web..framework' containing 'application':" . PHP_EOL;
$big = SpanQuery::near("body", ["web", "framework"], 2, true);
$little = SpanQuery::term("body", "application");
$containing = SpanQuery::containing("body", $big, $little);
print_results($index->search($containing, 5));

echo PHP_EOL . "Lexical search example completed!" . PHP_EOL;
