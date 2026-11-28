<?php

// Quick start example for the laurus PHP binding.
//
// Usage:
//   cd laurus-php
//   cargo build --release
//   php -d extension=target/release/liblaurus_php.so examples/quickstart.php

use Laurus\Index;
use Laurus\Schema;
use Laurus\TermQuery;

// Create a schema with two text fields.
$schema = new Schema();
$schema->addTextField("title");
$schema->addTextField("body");
$schema->setDefaultFields(["title", "body"]);

// Create an in-memory index.
$index = new Index(null, $schema);

// Add documents.
$index->addDocument("doc1", ["title" => "Laravel Framework", "body" => "Elegant PHP web application framework with expressive syntax."]);
$index->addDocument("doc2", ["title" => "Composer Package Manager", "body" => "Dependency management tool for PHP projects."]);
$index->commit();

// Search with a DSL string.
echo "=== DSL search ===" . PHP_EOL;
$results = $index->search("php", 5)->getResults();
foreach ($results as $r) {
    $doc = $r->getDocument();
    printf("  %s (score: %.4f): %s\n", $r->getId(), $r->getScore(), $doc["title"] ?? "");
}

// Search with a TermQuery object.
echo PHP_EOL . "=== TermQuery search ===" . PHP_EOL;
$results = $index->search(new TermQuery("body", "framework"), 5)->getResults();
foreach ($results as $r) {
    $doc = $r->getDocument();
    printf("  %s (score: %.4f): %s\n", $r->getId(), $r->getScore(), $doc["title"] ?? "");
}

// Index statistics.
echo PHP_EOL . "=== Stats ===" . PHP_EOL;
$stats = $index->stats();
printf("  Document count: %d\n", $stats["documentCount"]);
