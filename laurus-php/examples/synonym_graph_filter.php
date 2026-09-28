<?php

// Synonym Graph Filter Example — token expansion with synonyms.
//
// Demonstrates:
// - Creating a synonym dictionary with bidirectional synonym groups
// - Tokenizing text with WhitespaceTokenizer
// - Applying SynonymGraphFilter without boost
// - Applying SynonymGraphFilter with boost < 1.0 (synonyms weighted down)
//
// Usage:
//   cd laurus-php
//   cargo build --release
//   php -d extension=target/release/liblaurus_php.so examples/synonym_graph_filter.php

use Laurus\SynonymDictionary;
use Laurus\SynonymGraphFilter;
use Laurus\WhitespaceTokenizer;

echo "=== SynonymGraphFilter Usage Example ===" . PHP_EOL . PHP_EOL;

// ── Step 1: Create synonym dictionary ─────────────────────────────────
echo "Step 1: Creating synonym dictionary" . PHP_EOL;
$synDict = new SynonymDictionary();
$synDict->addSynonymGroup(["orm", "object relational mapping"]);
$synDict->addSynonymGroup(["di", "dependency injection"]);

echo "  Added synonyms:" . PHP_EOL;
echo "    - 'orm' <-> 'object relational mapping'" . PHP_EOL;
echo "    - 'di' <-> 'dependency injection'" . PHP_EOL . PHP_EOL;

$tokenizer = new WhitespaceTokenizer();

// ── Step 2: Apply filter WITHOUT boost ────────────────────────────────
echo "Step 2: Applying filter WITHOUT boost" . PHP_EOL;
$filt = new SynonymGraphFilter($synDict, true, 1.0);

$inputText = "orm tutorial";
echo "  Input: \"{$inputText}\"" . PHP_EOL . PHP_EOL;

$tokens = $tokenizer->tokenize($inputText);
$resultTokens = $filt->apply($tokens);

echo "  Output tokens:" . PHP_EOL;
foreach ($resultTokens as $i => $tok) {
    printf(
        "    [%d] %-25s  pos=%d  pos_inc=%d  pos_len=%d  boost=%.2f\n",
        $i,
        "'{$tok->getText()}'",
        $tok->getPosition(),
        $tok->getPositionIncrement(),
        $tok->getPositionLength(),
        $tok->getBoost()
    );
}

echo PHP_EOL;
echo "  Explanation:" . PHP_EOL;
echo "    - All tokens have boost=1.0 (default)" . PHP_EOL;
echo "    - Synonyms have equal weight to original tokens" . PHP_EOL . PHP_EOL;

// ── Step 3: Apply filter WITH boost=0.8 ───────────────────────────────
echo "Step 3: Applying filter WITH boost=0.8" . PHP_EOL;
$filtBoosted = new SynonymGraphFilter($synDict, true, 0.8);

echo "  Input: \"{$inputText}\"" . PHP_EOL . PHP_EOL;

$tokens = $tokenizer->tokenize($inputText);
$resultTokens = $filtBoosted->apply($tokens);

echo "  Output tokens:" . PHP_EOL;
foreach ($resultTokens as $i => $tok) {
    printf(
        "    [%d] %-25s  pos=%d  pos_inc=%d  pos_len=%d  boost=%.2f\n",
        $i,
        "'{$tok->getText()}'",
        $tok->getPosition(),
        $tok->getPositionIncrement(),
        $tok->getPositionLength(),
        $tok->getBoost()
    );
}

echo PHP_EOL;
echo "  Explanation:" . PHP_EOL;
echo "    - Original token 'orm' has boost=1.0" . PHP_EOL;
echo "    - Synonym tokens have boost < 1.0" . PHP_EOL;
echo "    - Lower boost means synonyms contribute less to the final score" . PHP_EOL;
echo "    - This helps prioritize exact matches over synonym matches" . PHP_EOL . PHP_EOL;

// ── Step 4: Multi-word synonym expansion ──────────────────────────────
echo "Step 4: Multi-word synonym expansion (di -> dependency injection)" . PHP_EOL;
$filt2 = new SynonymGraphFilter($synDict, true, 0.9);

$inputText2 = "di container";
echo "  Input: \"{$inputText2}\"" . PHP_EOL . PHP_EOL;

$tokens2 = $tokenizer->tokenize($inputText2);
$resultTokens2 = $filt2->apply($tokens2);

echo "  Output tokens (position graph):" . PHP_EOL;
foreach ($resultTokens2 as $i => $tok) {
    printf(
        "    [%d] %-25s  pos=%d  pos_inc=%d  pos_len=%d  boost=%.2f\n",
        $i,
        "'{$tok->getText()}'",
        $tok->getPosition(),
        $tok->getPositionIncrement(),
        $tok->getPositionLength(),
        $tok->getBoost()
    );
}

echo PHP_EOL;
echo "  Explanation:" . PHP_EOL;
echo "    - 'dependency injection' spans positions 0 and 1" . PHP_EOL;
echo "    - position_length 2 on 'di' makes it span both words of the synonym" . PHP_EOL;
echo "    - Downstream phrase indexing uses this graph to support phrase queries" . PHP_EOL . PHP_EOL;

echo "Use cases for boost:" . PHP_EOL;
echo "  - boost=0.8: Synonyms have 80% weight (common default)" . PHP_EOL;
echo "  - boost=0.5: Synonyms have 50% weight (conservative)" . PHP_EOL;
echo "  - boost=1.0: Synonyms equal to originals (no adjustment)" . PHP_EOL . PHP_EOL;

echo "SynonymGraphFilter example completed!" . PHP_EOL;
