<?php

declare(strict_types=1);

use PHPUnit\Framework\TestCase;

/**
 * Basic integration tests for the laurus PHP binding.
 */
class LaurusTest extends TestCase
{
    // ── Helpers ──────────────────────────────────────────────────────────

    /**
     * Return a fresh in-memory index with two indexed documents.
     */
    private function createIndex(): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addTextField("body");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("doc1", ["title" => "Introduction to Rust", "body" => "Systems programming language."]);
        $idx->putDocument("doc2", ["title" => "Python for Data Science", "body" => "Data analysis with Python."]);
        $idx->commit();
        return $idx;
    }

    /**
     * Return an in-memory HNSW index with two indexed documents.
     */
    private function createVectorIndex(): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addHnswField("embedding", 4);
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("doc1", ["title" => "Rust", "embedding" => [0.1, 0.2, 0.3, 0.4]]);
        $idx->putDocument("doc2", ["title" => "Python", "embedding" => [0.9, 0.8, 0.7, 0.6]]);
        $idx->commit();
        return $idx;
    }

    /**
     * Return an in-memory index with a Geo3d-typed `position` field.
     *
     * Coordinates are precomputed ECEF values (in meters) for well-known
     * landmarks, produced offline via `wgs84_to_ecef(lat, lon, height)`.
     */
    private function createGeo3dIndex(): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("name");
        $schema->addGeo3dField("position");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("tokyo_tower", [
            "name" => "Tokyo Tower",
            "position" => ["x" => -3955182.0, "y" => 3350553.0, "z" => 3700276.0],
        ]);
        $idx->putDocument("tokyo_skytree", [
            "name" => "Tokyo Skytree",
            "position" => ["x" => -3961178.0, "y" => 3346187.0, "z" => 3702490.0],
        ]);
        $idx->putDocument("mt_fuji", [
            "name" => "Mt. Fuji summit",
            "position" => ["x" => -3916073.0, "y" => 3437037.0, "z" => 3672751.0],
        ]);
        $idx->putDocument("sydney", [
            "name" => "Sydney Opera House",
            "position" => ["x" => -4646847.0, "y" => 2553022.0, "z" => -3534121.0],
        ]);
        $idx->commit();
        return $idx;
    }

    // ── Index creation ──────────────────────────────────────────────────

    public function testIndexMemory(): void
    {
        $idx = new Laurus\Index();
        $this->assertNotNull($idx);
    }

    public function testIndexWithSchema(): void
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index(null, $schema);
        $this->assertNotNull($idx);
    }

    // ── Document CRUD ───────────────────────────────────────────────────

    public function testPutAndGetDocument(): void
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("doc1", ["title" => "Hello"]);
        $idx->commit();
        $docs = $idx->getDocuments("doc1");
        $this->assertCount(1, $docs);
    }

    public function testPutReplacesExisting(): void
    {
        $idx = $this->createIndex();
        $idx->putDocument("doc1", ["title" => "Updated"]);
        $idx->commit();
        $docs = $idx->getDocuments("doc1");
        $this->assertCount(1, $docs);
    }

    public function testAddDocumentAppends(): void
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index(null, $schema);
        $idx->addDocument("doc1", ["title" => "Chunk 1"]);
        $idx->addDocument("doc1", ["title" => "Chunk 2"]);
        $docs = $idx->getDocuments("doc1");
        $this->assertCount(2, $docs);
    }

    public function testDeleteDocuments(): void
    {
        $idx = $this->createIndex();
        $idx->deleteDocuments("doc1");
        $idx->commit();
        $docs = $idx->getDocuments("doc1");
        $this->assertCount(0, $docs);
    }

    public function testGetUnknownId(): void
    {
        $idx = $this->createIndex();
        $docs = $idx->getDocuments("unknown");
        $this->assertCount(0, $docs);
    }

    // ── Batch ingestion (#866) ───────────────────────────────────────────

    public function testPutDocumentsEmptyBatchIsNoop(): void
    {
        $idx = new Laurus\Index();
        $idx->putDocuments([]);
        $idx->addDocuments([]);
        $idx->commit();
        $this->assertEquals(0, $idx->stats()["documentCount"]);
    }

    public function testPutDocumentsAppliesAndDedupes(): void
    {
        $idx = new Laurus\Index();
        $idx->putDocuments([
            ["doc1", ["title" => "One"]],
            ["doc2", ["title" => "Two"]],
            ["doc1", ["title" => "One v2"]], // duplicate id: last wins
        ]);
        $idx->commit();

        $this->assertEquals(2, $idx->stats()["documentCount"]);
        $docs = $idx->getDocuments("doc1");
        $this->assertCount(1, $docs);
        $this->assertEquals("One v2", $docs[0]["title"]);
    }

    public function testAddDocumentsAccumulatesChunks(): void
    {
        $idx = new Laurus\Index();
        $idx->addDocuments([
            ["doc", ["title" => "chunk 0"]],
            ["doc", ["title" => "chunk 1"]],
        ]);
        $idx->commit();

        $this->assertCount(2, $idx->getDocuments("doc"));
    }

    // ── Statistics ───────────────────────────────────────────────────────

    public function testDocumentCount(): void
    {
        $idx = $this->createIndex();
        $stats = $idx->stats();
        $this->assertEquals(2, $stats["documentCount"]);
    }

    public function testVectorFieldStats(): void
    {
        $idx = $this->createVectorIndex();
        $stats = $idx->stats();
        $this->assertEquals(2, $stats["documentCount"]);
        $this->assertArrayHasKey("embedding", $stats["vectorFields"]);
        $this->assertEquals(4, $stats["vectorFields"]["embedding"]["dimension"]);
    }

    // ── Lexical search ──────────────────────────────────────────────────

    public function testSearchDsl(): void
    {
        $idx = $this->createIndex();
        $results = $idx->search("title:rust");
        $this->assertCount(1, $results);
        $this->assertEquals("doc1", $results[0]->getId());
        $this->assertGreaterThan(0, $results[0]->getScore());
    }

    public function testSearchTermQuery(): void
    {
        $idx = $this->createIndex();
        $q = new Laurus\TermQuery("title", "python");
        $results = $idx->search($q);
        $this->assertCount(1, $results);
        $this->assertEquals("doc2", $results[0]->getId());
    }

    public function testSearchWithLimit(): void
    {
        $idx = $this->createIndex();
        $results = $idx->search("title:rust OR title:python", 1);
        $this->assertCount(1, $results);
    }

    public function testSearchWithOffset(): void
    {
        $idx = $this->createIndex();
        $all = $idx->search("title:rust OR title:python", 10);
        $offset = $idx->search("title:rust OR title:python", 10, 1);
        $this->assertCount(count($all) - 1, $offset);
    }

    public function testSearchNoResults(): void
    {
        $idx = $this->createIndex();
        $results = $idx->search("title:nonexistent");
        $this->assertCount(0, $results);
    }

    // ── Query types ─────────────────────────────────────────────────────

    public function testPhraseQuery(): void
    {
        $idx = $this->createIndex();
        $q = new Laurus\PhraseQuery("body", ["systems", "programming"]);
        $results = $idx->search($q);
        $this->assertCount(1, $results);
    }

    public function testFuzzyQuery(): void
    {
        $idx = $this->createIndex();
        $q = new Laurus\FuzzyQuery("title", "rast");
        $results = $idx->search($q);
        $this->assertCount(1, $results);
    }

    public function testBooleanQuery(): void
    {
        $idx = $this->createIndex();
        $bq = new Laurus\BooleanQuery();
        $bq->must(new Laurus\TermQuery("title", "rust"));
        $bq->mustNot(new Laurus\TermQuery("title", "python"));
        $results = $idx->search($bq);
        $this->assertCount(1, $results);
        $this->assertEquals("doc1", $results[0]->getId());
    }

    public function testWildcardQuery(): void
    {
        $idx = $this->createIndex();
        $q = new Laurus\WildcardQuery("title", "ru*");
        $results = $idx->search($q);
        $this->assertCount(1, $results);
    }

    // ── Multi-valued numeric arrays (Issue #1178) ────────────────────────
    //
    // `zval_to_data_value` used to turn every sequential array into
    // `DataValue::Vector`, which the core's multi-valued integer/float
    // coercion rejects — so a `multi_valued` field could be read from PHP
    // but never written from it. Arrays now arrive as Int64Array (all int)
    // or Float64Array (otherwise numeric).

    private function indexWithIntegerField(bool $multiValued): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        // (name, stored, indexed, multi_valued)
        $schema->addIntegerField("tags", true, true, $multiValued);
        return new Laurus\Index(null, $schema);
    }

    private function indexWithFloatField(): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addFloatField("scores", true, true, true);
        return new Laurus\Index(null, $schema);
    }

    public function testIntArrayRoundTripsThroughMultiValuedIntegerField(): void
    {
        $idx = $this->indexWithIntegerField(true);
        // Before #1178 this putDocument threw: the array arrived as a Vector,
        // which a multi-valued integer field does not accept.
        $idx->putDocument("doc1", ["title" => "t", "tags" => [1, 2, 3]]);
        $idx->commit();
        $docs = $idx->getDocuments("doc1");
        $this->assertCount(1, $docs);
        $this->assertSame([1, 2, 3], $docs[0]["tags"]);
    }

    public function testFloatArrayRoundTripsThroughMultiValuedFloatField(): void
    {
        $idx = $this->indexWithFloatField();
        $idx->putDocument("doc1", ["title" => "t", "scores" => [1.5, 2.0]]);
        $idx->commit();
        $this->assertSame([1.5, 2.0], $idx->getDocuments("doc1")[0]["scores"]);
    }

    public function testMixedIntAndFloatArrayBecomesFloatArray(): void
    {
        $idx = $this->indexWithFloatField();
        $idx->putDocument("doc1", ["title" => "t", "scores" => [1, 2.5]]);
        $idx->commit();
        $this->assertSame([1.0, 2.5], $idx->getDocuments("doc1")[0]["scores"]);
    }

    public function testEmptyArrayIsAcceptedByMultiValuedIntegerField(): void
    {
        $idx = $this->indexWithIntegerField(true);
        $idx->putDocument("doc1", ["title" => "t", "tags" => []]);
        $idx->commit();
        $this->assertSame([], $idx->getDocuments("doc1")[0]["tags"]);
    }

    public function testArrayIntoSingleValuedIntegerFieldIsStillRejected(): void
    {
        $idx = $this->indexWithIntegerField(false);
        $this->expectException(\Throwable::class);
        $idx->putDocument("doc1", ["title" => "t", "tags" => [2020, 2021]]);
    }

    // ── Multi-valued geo arrays (Issue #1174) ────────────────────────────
    //
    // A sequential array of `["lat" => .., "lon" => ..]` arrays is a
    // multi-valued geo field; each element takes the same associative-array
    // path as a single point.

    private function indexWithGeoField(bool $multiValued): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        // (name, stored, indexed, multi_valued)
        $schema->addGeoField("spots", true, true, $multiValued);
        return new Laurus\Index(null, $schema);
    }

    public function testGeoArrayRoundTripsThroughMultiValuedGeoField(): void
    {
        $idx = $this->indexWithGeoField(true);
        $spots = [["lat" => 35.68, "lon" => 139.76], ["lat" => 34.69, "lon" => 135.5]];
        $idx->putDocument("doc1", ["title" => "t", "spots" => $spots]);
        $idx->commit();
        $this->assertSame($spots, $idx->getDocuments("doc1")[0]["spots"]);
    }

    public function testGeo3dArrayRoundTripsThroughMultiValuedGeo3dField(): void
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addGeo3dField("positions", true, true, true);
        $idx = new Laurus\Index(null, $schema);
        $positions = [["x" => 1.0, "y" => 2.0, "z" => 3.0], ["x" => -4.0, "y" => 5.0, "z" => -6.0]];
        $idx->putDocument("doc1", ["title" => "t", "positions" => $positions]);
        $idx->commit();
        $this->assertSame($positions, $idx->getDocuments("doc1")[0]["positions"]);
    }

    public function testSinglePointIsWrappedOnMultiValuedGeoField(): void
    {
        $idx = $this->indexWithGeoField(true);
        $idx->putDocument("doc1", ["title" => "t", "spots" => ["lat" => 35.68, "lon" => 139.76]]);
        $idx->commit();
        $this->assertSame([["lat" => 35.68, "lon" => 139.76]], $idx->getDocuments("doc1")[0]["spots"]);
    }

    public function testEmptyArrayIsAcceptedByMultiValuedGeoField(): void
    {
        $idx = $this->indexWithGeoField(true);
        $idx->putDocument("doc1", ["title" => "t", "spots" => []]);
        $idx->commit();
        $this->assertSame([], $idx->getDocuments("doc1")[0]["spots"]);
    }

    public function testGeoArrayIntoSingleValuedGeoFieldIsRejected(): void
    {
        $idx = $this->indexWithGeoField(false);
        $this->expectException(\Throwable::class);
        $this->expectExceptionMessageMatches('/multi_valued/');
        $idx->putDocument("doc1", ["title" => "t", "spots" => [["lat" => 35.68, "lon" => 139.76]]]);
    }

    // ── Multi-valued datetime arrays (Issue #1184) ───────────────────────

    private function indexWithMultiValuedDatetimeField(bool $multiValued): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        // (name, stored, indexed, multi_valued)
        $schema->addDatetimeField("seen_at", true, true, $multiValued);
        return new Laurus\Index(null, $schema);
    }

    public function testDatetimeArrayRoundTripsThroughMultiValuedDatetimeField(): void
    {
        $idx = $this->indexWithMultiValuedDatetimeField(true);
        $idx->putDocument("doc1", [
            "title" => "t",
            "seen_at" => ["2024-01-01T00:00:00Z", "2024-06-15T21:00:00+09:00"],
        ]);
        $idx->putDocument("doc2", ["title" => "t", "seen_at" => ["2025-03-01T00:00:00Z"]]);
        $idx->commit();
        $this->assertSame(
            ["2024-01-01T00:00:00+00:00", "2024-06-15T12:00:00+00:00"],
            $idx->getDocuments("doc1")[0]["seen_at"]
        );
        // Any instant matches a range query.
        $this->assertSame(["doc1"], $this->idsOf($idx->search("seen_at:[2024-06-01 TO 2024-12-31]")));
    }

    public function testSingleDatetimeIsWrappedOnMultiValuedDatetimeField(): void
    {
        $idx = $this->indexWithMultiValuedDatetimeField(true);
        $idx->putDocument("doc1", ["title" => "t", "seen_at" => "2024-01-01T00:00:00Z"]);
        $idx->commit();
        $this->assertSame(["2024-01-01T00:00:00+00:00"], $idx->getDocuments("doc1")[0]["seen_at"]);
    }

    public function testEmptyArrayIsAcceptedByMultiValuedDatetimeField(): void
    {
        $idx = $this->indexWithMultiValuedDatetimeField(true);
        $idx->putDocument("doc1", ["title" => "t", "seen_at" => []]);
        $idx->commit();
        $this->assertSame([], $idx->getDocuments("doc1")[0]["seen_at"]);
    }

    public function testDatetimeArrayIntoSingleValuedDatetimeFieldIsRejected(): void
    {
        $idx = $this->indexWithMultiValuedDatetimeField(false);
        $this->expectException(\Throwable::class);
        $this->expectExceptionMessageMatches('/multi_valued/');
        $idx->putDocument("doc1", ["title" => "t", "seen_at" => ["2024-01-01T00:00:00Z"]]);
    }

    public function testNonDatetimeStringArrayIsRejected(): void
    {
        $idx = $this->indexWithMultiValuedDatetimeField(true);
        $this->expectException(\Throwable::class);
        $this->expectExceptionMessageMatches('/datetimes/');
        $idx->putDocument("doc1", ["title" => "t", "seen_at" => ["2024-01-01T00:00:00Z", "tomorrow"]]);
    }

    // ── Multi-valued boolean arrays (Issue #1180) ────────────────────────

    private function indexWithMultiValuedBooleanField(bool $multiValued): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        // (name, stored, indexed, multi_valued)
        $schema->addBooleanField("flags", true, true, $multiValued);
        return new Laurus\Index(null, $schema);
    }

    public function testBoolArrayRoundTripsThroughMultiValuedBooleanField(): void
    {
        $idx = $this->indexWithMultiValuedBooleanField(true);
        $idx->putDocument("doc1", ["title" => "t", "flags" => [true, false]]);
        $idx->putDocument("doc2", ["title" => "t", "flags" => [false]]);
        $idx->commit();
        $this->assertSame([true, false], $idx->getDocuments("doc1")[0]["flags"]);
        // Any element matches a term query.
        $this->assertSame(["doc1"], $this->idsOf($idx->search("flags:true")));
        $falses = $this->idsOf($idx->search("flags:false"));
        sort($falses);
        $this->assertSame(["doc1", "doc2"], $falses);
    }

    public function testSingleBoolIsWrappedOnMultiValuedBooleanField(): void
    {
        $idx = $this->indexWithMultiValuedBooleanField(true);
        $idx->putDocument("doc1", ["title" => "t", "flags" => true]);
        $idx->commit();
        $this->assertSame([true], $idx->getDocuments("doc1")[0]["flags"]);
    }

    public function testEmptyArrayIsAcceptedByMultiValuedBooleanField(): void
    {
        $idx = $this->indexWithMultiValuedBooleanField(true);
        $idx->putDocument("doc1", ["title" => "t", "flags" => []]);
        $idx->commit();
        $this->assertSame([], $idx->getDocuments("doc1")[0]["flags"]);
    }

    public function testBoolArrayIntoSingleValuedBooleanFieldIsRejected(): void
    {
        $idx = $this->indexWithMultiValuedBooleanField(false);
        $this->expectException(\Throwable::class);
        $this->expectExceptionMessageMatches('/multi_valued/');
        $idx->putDocument("doc1", ["title" => "t", "flags" => [true]]);
    }

    public function testMixedBoolAndIntArrayIsRejected(): void
    {
        $idx = $this->indexWithMultiValuedBooleanField(true);
        $this->expectException(\Throwable::class);
        $this->expectExceptionMessageMatches('/numeric/');
        $idx->putDocument("doc1", ["title" => "t", "flags" => [true, 1]]);
    }

    public function testMixedGeoDimensionArrayIsRejected(): void
    {
        $idx = $this->indexWithGeoField(true);
        $this->expectException(\Throwable::class);
        $idx->putDocument("doc1", [
            "title" => "t",
            "spots" => [["lat" => 35.68, "lon" => 139.76], ["x" => 1.0, "y" => 2.0, "z" => 3.0]],
        ]);
    }

    // ── DateTimeRangeQuery (Issue #1179) ────────────────────────────────

    private function indexWithDateField(): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addDatetimeField("created_at");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("d1", ["title" => "jan", "created_at" => "2024-01-01T00:00:00Z"]);
        $idx->putDocument("d2", ["title" => "jun", "created_at" => "2024-06-15T12:00:00Z"]);
        $idx->putDocument("d3", ["title" => "next", "created_at" => "2025-01-01T00:00:00Z"]);
        $idx->commit();
        return $idx;
    }

    private function idsOf(array $results): array
    {
        $ids = array_map(fn ($r) => $r->getId(), $results);
        sort($ids);
        return $ids;
    }

    public function testDateTimeRangeQueryWithDateOnlyBounds(): void
    {
        $idx = $this->indexWithDateField();
        $q = new Laurus\DateTimeRangeQuery("created_at", "2024-01-01", "2024-12-31");
        $this->assertSame(["d1", "d2"], $this->idsOf($idx->search($q)));
        $this->assertStringContainsString("DateTimeRangeQuery", (string) $q);
    }

    public function testDateTimeRangeQueryWithRfc3339AndOpenBound(): void
    {
        $idx = $this->indexWithDateField();
        // +09:00 is normalized to UTC: 2024-06-15T21:00:00+09:00 == 12:00Z.
        $q = new Laurus\DateTimeRangeQuery("created_at", "2024-06-15T21:00:00+09:00", null);
        $this->assertSame(["d2", "d3"], $this->idsOf($idx->search($q)));
        $q = new Laurus\DateTimeRangeQuery("created_at", null, "2024-06-15T12:00:00");
        $this->assertSame(["d1", "d2"], $this->idsOf($idx->search($q)));
    }

    public function testDateTimeRangeQueryDslFormReachesTheIndex(): void
    {
        $idx = $this->indexWithDateField();
        $results = $idx->search("created_at:{2024-01-01 TO 2024-12-31}");
        $this->assertSame(["d2"], $this->idsOf($results));
    }

    public function testDateTimeRangeQueryRejectsMalformedBound(): void
    {
        $idx = $this->indexWithDateField();
        $q = new Laurus\DateTimeRangeQuery("created_at", "yesterday", null);
        $this->expectException(\Throwable::class);
        $idx->search($q);
    }

    public function testNumericRangeQuery(): void
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addIntegerField("year");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("d1", ["title" => "old", "year" => 2000]);
        $idx->putDocument("d2", ["title" => "new", "year" => 2024]);
        $idx->commit();
        $q = new Laurus\NumericRangeQuery("year", 2020, 2030);
        $results = $idx->search($q);
        $this->assertCount(1, $results);
        $this->assertEquals("d2", $results[0]->getId());
    }

    // ── SpanQuery ───────────────────────────────────────────────────────

    public function testSpanQueryNear(): void
    {
        // Plain term-only `near` already works via `SpanQuery::near`.
        $schema = new Laurus\Schema();
        $schema->addTextField("body", true, true, true);
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("d1", ["body" => "the quick brown fox"]);
        $idx->putDocument("d2", ["body" => "the slow red fox"]);
        $idx->commit();
        $q = Laurus\SpanQuery::near("body", ["quick", "fox"], 2, true);
        $results = $idx->search($q);
        $this->assertCount(1, $results);
        $this->assertEquals("d1", $results[0]->getId());
    }

    public function testSpanQueryNearSpansWithNestedClauses(): void
    {
        // `nearSpans` accepts pre-built SpanQuery instances, so callers can
        // nest other span constructions (e.g. another `near`, `containing`,
        // `within`) inside a `SpanNear`. The simplest exercise wraps two
        // bare `term` clauses and verifies the result is identical to the
        // plain `near` form.
        $schema = new Laurus\Schema();
        $schema->addTextField("body", true, true, true);
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("d1", ["body" => "the quick brown fox"]);
        $idx->putDocument("d2", ["body" => "the slow red fox"]);
        $idx->commit();
        $clauses = [
            Laurus\SpanQuery::term("body", "quick"),
            Laurus\SpanQuery::term("body", "fox"),
        ];
        $q = Laurus\SpanQuery::nearSpans("body", $clauses, 2, true);
        $results = $idx->search($q);
        $this->assertCount(1, $results);
        $this->assertEquals("d1", $results[0]->getId());
    }

    // ── Geo3d (3D ECEF) ─────────────────────────────────────────────────

    public function testGeo3dFieldRoundTrip(): void
    {
        $idx = $this->createGeo3dIndex();
        $docs = $idx->getDocuments("tokyo_tower");
        $this->assertCount(1, $docs);
        $this->assertEquals("Tokyo Tower", $docs[0]["name"]);
        $this->assertEqualsWithDelta(-3955182.0, $docs[0]["position"]["x"], 1.0);
        $this->assertEqualsWithDelta(3350553.0, $docs[0]["position"]["y"], 1.0);
        $this->assertEqualsWithDelta(3700276.0, $docs[0]["position"]["z"], 1.0);
    }

    public function testGeo3dDistanceQuerySmallRadius(): void
    {
        // 50 km sphere around Tokyo Tower returns Tower + Skytree only.
        $idx = $this->createGeo3dIndex();
        $q = Laurus\Geo3dDistanceQuery::withinSphere(
            "position", -3955182.0, 3350553.0, 3700276.0, 50000.0
        );
        $results = $idx->search($q, 10);
        $ids = array_map(fn($r) => $r->getId(), $results);
        sort($ids);
        $this->assertEquals(["tokyo_skytree", "tokyo_tower"], $ids);
    }

    public function testGeo3dBoundingBoxQuery(): void
    {
        // Central-Tokyo box returns Tower + Skytree only (Mt. Fuji and Sydney
        // are outside the small AABB).
        $idx = $this->createGeo3dIndex();
        $q = Laurus\Geo3dBoundingBoxQuery::withinBox(
            "position",
            -3962000.0, 3340000.0, 3690000.0,
            -3954000.0, 3360000.0, 3710000.0
        );
        $results = $idx->search($q, 10);
        $ids = array_map(fn($r) => $r->getId(), $results);
        sort($ids);
        $this->assertEquals(["tokyo_skytree", "tokyo_tower"], $ids);
    }

    public function testGeo3dNearestQuery(): void
    {
        // k = 3 around Mt. Fuji returns Fuji + Tower + Skytree.
        $idx = $this->createGeo3dIndex();
        $q = Laurus\Geo3dNearestQuery::kNearest(
            "position", -3916073.0, 3437037.0, 3672751.0, 3
        );
        $results = $idx->search($q, 3);
        $this->assertCount(3, $results);
        $this->assertEquals("mt_fuji", $results[0]->getId());
        $ids = array_map(fn($r) => $r->getId(), $results);
        sort($ids);
        $this->assertEquals(["mt_fuji", "tokyo_skytree", "tokyo_tower"], $ids);
    }

    public function testGeo3dNearestQueryWithRadiusOptions(): void
    {
        // Verify the optional initial / max radius parameters are accepted.
        $idx = $this->createGeo3dIndex();
        $q = Laurus\Geo3dNearestQuery::kNearest(
            "position", -3955182.0, 3350553.0, 3700276.0, 2,
            10000.0,        // initial_radius_m
            10000000.0,     // max_radius_m
        );
        $results = $idx->search($q, 2);
        $ids = array_map(fn($r) => $r->getId(), $results);
        sort($ids);
        $this->assertEquals(["tokyo_skytree", "tokyo_tower"], $ids);
    }

    // ── Vector search ───────────────────────────────────────────────────

    public function testVectorQuery(): void
    {
        $idx = $this->createVectorIndex();
        $q = new Laurus\VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4]);
        $results = $idx->search($q);
        $this->assertGreaterThanOrEqual(1, count($results));
        $this->assertEquals("doc1", $results[0]->getId());
    }

    // ── Hybrid search ───────────────────────────────────────────────────

    public function testSearchRequestLexicalOnly(): void
    {
        $idx = $this->createIndex();
        $req = new Laurus\SearchRequest(
            null, // query
            new Laurus\TermQuery("title", "rust"), // lexical_query
            null, // vector_query
            null, // filter_query
            null, // fusion
        );
        $results = $idx->search($req);
        $this->assertCount(1, $results);
    }

    // ── Fusion algorithms ───────────────────────────────────────────────

    public function testRRFRepr(): void
    {
        $rrf = new Laurus\RRF();
        $this->assertEquals("RRF(k=60)", (string)$rrf);
    }

    public function testWeightedSumRepr(): void
    {
        $ws = new Laurus\WeightedSum();
        $this->assertEquals("WeightedSum(lexical_weight=0.5, vector_weight=0.5)", (string)$ws);
    }

    // ── Analysis pipeline ───────────────────────────────────────────────

    public function testWhitespaceTokenizer(): void
    {
        $tok = new Laurus\WhitespaceTokenizer();
        $tokens = $tok->tokenize("hello world foo");
        $this->assertCount(3, $tokens);
        $this->assertEquals("hello", $tokens[0]->getText());
        $this->assertEquals("world", $tokens[1]->getText());
        $this->assertEquals("foo", $tokens[2]->getText());
    }

    public function testSynonymDictionary(): void
    {
        $dict = new Laurus\SynonymDictionary();
        $dict->addSynonymGroup(["quick", "fast", "speedy"]);
        $this->assertNotNull($dict);
    }

    public function testSynonymGraphFilter(): void
    {
        $dict = new Laurus\SynonymDictionary();
        $dict->addSynonymGroup(["happy", "joyful"]);
        $tok = new Laurus\WhitespaceTokenizer();
        $tokens = $tok->tokenize("I am happy");
        $filter = new Laurus\SynonymGraphFilter($dict);
        $expanded = $filter->apply($tokens);
        $texts = array_map(fn($t) => $t->getText(), $expanded);
        $this->assertContains("happy", $texts);
        $this->assertContains("joyful", $texts);
    }

    // ── SearchResult ────────────────────────────────────────────────────

    public function testSearchResultDocument(): void
    {
        $idx = $this->createIndex();
        $results = $idx->search("title:rust");
        $this->assertCount(1, $results);
        $doc = $results[0]->getDocument();
        $this->assertIsArray($doc);
        $this->assertEquals("Introduction to Rust", $doc["title"]);
    }

    // ── __toString ──────────────────────────────────────────────────────

    public function testTermQueryToString(): void
    {
        $q = new Laurus\TermQuery("title", "hello");
        $this->assertEquals("TermQuery(field='title', term='hello')", (string)$q);
    }

    public function testSearchResultToString(): void
    {
        $idx = $this->createIndex();
        $results = $idx->search("title:rust");
        $str = (string)$results[0];
        $this->assertStringContainsString("SearchResult(", $str);
        $this->assertStringContainsString("doc1", $str);
    }

    // ── searchBatch (Phase 3e of #648, issue #720) ───────────────────────

    /**
     * Return a fresh in-memory index with three indexed documents — used
     * specifically by the searchBatch tests to provide three distinct
     * query targets.
     */
    private function createBatchIndex(): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addTextField("body");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("doc1", ["title" => "Introduction to Rust", "body" => "Systems programming language."]);
        $idx->putDocument("doc2", ["title" => "Python for Data Science", "body" => "Data analysis with Python."]);
        $idx->putDocument("doc3", ["title" => "Distributed Systems", "body" => "Engineering at scale."]);
        $idx->commit();
        return $idx;
    }

    public function testSearchBatchEmpty(): void
    {
        $idx = $this->createBatchIndex();
        $results = $idx->searchBatch([]);
        $this->assertSame([], $results);
    }

    public function testSearchBatchSingleQueryMatchesSearch(): void
    {
        $idx = $this->createBatchIndex();
        $serial = $idx->search("title:rust", 5);
        $batch = $idx->searchBatch(["title:rust"], 5);

        $this->assertCount(1, $batch);
        $this->assertCount(count($serial), $batch[0]);
        foreach ($serial as $i => $s) {
            $this->assertEquals($s->getId(), $batch[0][$i]->getId());
        }
    }

    public function testSearchBatchMultiQueryPreservesOrder(): void
    {
        $idx = $this->createBatchIndex();
        $queries = ["title:rust", "body:python", "title:distributed"];
        $expectedTopIds = ["doc1", "doc2", "doc3"];

        $batch = $idx->searchBatch($queries, 5);
        $this->assertCount(count($queries), $batch);

        foreach ($queries as $i => $q) {
            $this->assertGreaterThanOrEqual(1, count($batch[$i]), "expected at least 1 hit for $q");
            $this->assertEquals($expectedTopIds[$i], $batch[$i][0]->getId());
        }
    }

    public function testSearchBatchWithQueryObjects(): void
    {
        $idx = $this->createBatchIndex();
        $queries = [
            new Laurus\TermQuery("title", "rust"),
            new Laurus\TermQuery("body", "python"),
        ];
        $batch = $idx->searchBatch($queries, 5);

        $this->assertCount(2, $batch);
        $this->assertEquals("doc1", $batch[0][0]->getId());
        $this->assertEquals("doc2", $batch[1][0]->getId());
    }

    public function testSearchBatchNoMatchReturnsEmptyInnerArray(): void
    {
        $idx = $this->createBatchIndex();
        $queries = ["title:rust", "title:nonexistent_xyz"];
        $batch = $idx->searchBatch($queries, 5);

        $this->assertCount(2, $batch);
        $this->assertGreaterThanOrEqual(1, count($batch[0]));
        $this->assertSame([], $batch[1]);
    }

    public function testSearchBatchLimitPerQuery(): void
    {
        $idx = $this->createBatchIndex();
        $queries = ["body:programming OR body:data", "body:programming OR body:data"];
        $batch = $idx->searchBatch($queries, 1);

        $this->assertCount(2, $batch);
        foreach ($batch as $results) {
            $this->assertLessThanOrEqual(1, count($results));
        }
    }

    // ── Highlighting (Issue #1134) ─────────────────────────────────────────

    /**
     * Return a fresh in-memory index with one document whose body contains
     * "Rust", for highlight assertions.
     */
    private function createHighlightIndex(): Laurus\Index
    {
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->addTextField("body");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("doc1", [
            "title" => "Introduction to Rust",
            "body" => "Rust is a systems programming language.",
        ]);
        $idx->commit();
        return $idx;
    }

    public function testHighlightListReturnsFragments(): void
    {
        $idx = $this->createHighlightIndex();
        $results = $idx->search("body:rust", 10, 0, ["body"]);

        $this->assertCount(1, $results);
        $fragments = $results[0]->getHighlights()["body"];
        $this->assertCount(1, $fragments);
        $this->assertStringContainsString("<mark>Rust</mark>", $fragments[0]);
    }

    public function testHighlightOmittedLeavesHighlightsEmpty(): void
    {
        $idx = $this->createHighlightIndex();
        $results = $idx->search("body:rust");

        $this->assertCount(1, $results);
        $this->assertSame([], $results[0]->getHighlights());
    }

    public function testHighlightUnrequestedFieldIsAbsent(): void
    {
        $idx = $this->createHighlightIndex();
        $results = $idx->search("body:rust", 10, 0, ["body"]);

        $this->assertArrayNotHasKey("title", $results[0]->getHighlights());
    }

    public function testHighlightAssociativeArrayAppliesConfig(): void
    {
        $idx = $this->createHighlightIndex();
        $results = $idx->search("body:rust", 10, 0, ["fields" => ["body"], "tag" => "em", "max_fragments" => 2]);

        $fragment = $results[0]->getHighlights()["body"][0];
        $this->assertStringContainsString("<em>Rust</em>", $fragment);
        $this->assertStringNotContainsString("<mark>", $fragment);
    }

    public function testHighlightAssociativeArrayRequiresFieldsKey(): void
    {
        $idx = $this->createHighlightIndex();
        $this->expectException(\Exception::class);
        $idx->search("body:rust", 10, 0, ["max_fragments" => 2]);
    }

    public function testSearchBatchAppliesHighlightToEveryQuery(): void
    {
        $idx = $this->createHighlightIndex();
        // Both queries target "body" so the requested "body" highlight
        // applies to both under the default requireFieldMatch = true.
        $batch = $idx->searchBatch(["body:rust", "body:programming"], 10, 0, ["body"]);

        $this->assertCount(2, $batch);
        foreach ($batch as $results) {
            $this->assertCount(1, $results);
            $this->assertStringContainsString("<mark>", $results[0]->getHighlights()["body"][0]);
        }
    }

    public function testSearchRequestHighlightMatchesSearchArgument(): void
    {
        $idx = $this->createHighlightIndex();
        $viaArgument = $idx->search("body:rust", 10, 0, ["body"]);
        // SearchRequest's constructor args before `highlight` have no PHP
        // default (only `limit`/`offset` do), so they must all be passed
        // positionally here, same as the existing SearchRequest tests above.
        $request = new Laurus\SearchRequest("body:rust", null, null, null, null, 10, 0, ["body"]);
        $viaRequest = $idx->search($request);

        $this->assertEquals($viaArgument[0]->getHighlights(), $viaRequest[0]->getHighlights());
    }

    // ── HNSW quantizer / rerank_storage options (#797) ────────────────────
    //
    // These assert the values configured on addHnswField actually reach the
    // Rust core via deterministic observables, not merely that search
    // succeeds. addHnswField is positional:
    //   (name, dimension, distance, m, efConstruction, defaultEfSearch,
    //    embedder, quantizer, subvectorCount, rerankStorage)

    /**
     * Recursively collect files under $dir whose name ends with $suffix.
     */
    private function findFilesBySuffix(string $dir, string $suffix): array
    {
        $found = [];
        $it = new RecursiveIteratorIterator(
            new RecursiveDirectoryIterator($dir, FilesystemIterator::SKIP_DOTS)
        );
        foreach ($it as $file) {
            if (str_ends_with($file->getFilename(), $suffix)) {
                $found[] = $file->getPathname();
            }
        }
        return $found;
    }

    public function testRerankStorageF32WritesSidecar(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_rerank_" . uniqid();
        mkdir($dir);
        $schema = new Laurus\Schema();
        $schema->addHnswField("embedding", 4, null, 16, 200, null, null, null, null, "f32");
        $idx = new Laurus\Index($dir, $schema);
        $idx->putDocument("doc1", ["embedding" => [0.1, 0.2, 0.3, 0.4]]);
        $idx->putDocument("doc2", ["embedding" => [0.9, 0.8, 0.7, 0.6]]);
        $idx->commit();

        $this->assertNotEmpty(
            $this->findFilesBySuffix($dir, ".hnsw.f32"),
            "rerank_storage 'f32' must write a .hnsw.f32 sidecar"
        );
    }

    public function testNoRerankStorageWritesNoSidecar(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_norerank_" . uniqid();
        mkdir($dir);
        $schema = new Laurus\Schema();
        $schema->addHnswField("embedding", 4);
        $idx = new Laurus\Index($dir, $schema);
        $idx->putDocument("doc1", ["embedding" => [0.1, 0.2, 0.3, 0.4]]);
        $idx->putDocument("doc2", ["embedding" => [0.9, 0.8, 0.7, 0.6]]);
        $idx->commit();

        $this->assertEmpty($this->findFilesBySuffix($dir, ".hnsw.f32"));
    }

    public function testProductQuantizationBuildsAndSearches(): void
    {
        $schema = new Laurus\Schema();
        // PQ is an L2 quantizer, so use Euclidean (matching the core's
        // test_hnsw_pq_search_returns_corpus_neighbour).
        $schema->addHnswField("embedding", 4, "euclidean", 16, 200, null, null, "product_quantization", 2, null);
        $idx = new Laurus\Index(null, $schema);
        // Stable two-cluster corpus mirroring the core (issue #730), sized
        // to 128 points per cluster (256 total) so the segment meets the PQ
        // min-train threshold (#880: smaller PQ-configured segments are
        // written as Scalar8Bit and would not exercise PQ training at all).
        $nearBase = [10.0, 10.0, 20.0, 20.0];
        $farBase = [-100.0, -100.0, -200.0, -200.0];
        for ($i = 0; $i < 128; $i++) {
            $off = [
                ($i % 8) * 0.04 - 0.14,
                (intdiv($i, 8) % 8) * 0.04 - 0.14,
                (intdiv($i, 64) % 8) * 0.04 - 0.14,
                ($i % 16) * 0.04 - 0.32,
            ];
            $near = [];
            $far = [];
            foreach ($nearBase as $j => $b) {
                $near[] = $b + $off[$j];
            }
            foreach ($farBase as $j => $b) {
                $far[] = $b + $off[$j];
            }
            $idx->putDocument("near$i", ["embedding" => $near]);
            $idx->putDocument("far$i", ["embedding" => $far]);
        }
        $idx->commit();

        $q = new Laurus\VectorQuery("embedding", $nearBase);
        $results = $idx->search($q, 3);
        $this->assertCount(3, $results);
        foreach ($results as $r) {
            $this->assertStringStartsWith("near", $r->getId());
        }
    }

    public function testPqSubvectorCountMustDivideDimension(): void
    {
        $schema = new Laurus\Schema();
        $schema->addHnswField("embedding", 4, null, 16, 200, null, null, "product_quantization", 3, null);
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("doc1", ["embedding" => [0.1, 0.2, 0.3, 0.4]]);
        $this->expectException(\Throwable::class);
        $idx->commit();
    }

    public function testUnknownQuantizerRejected(): void
    {
        $schema = new Laurus\Schema();
        $this->expectException(\Throwable::class);
        $schema->addHnswField("embedding", 4, null, 16, 200, null, null, "bogus");
    }

    public function testPqRequiresSubvectorCount(): void
    {
        $schema = new Laurus\Schema();
        $this->expectException(\Throwable::class);
        $schema->addHnswField("embedding", 4, null, 16, 200, null, null, "product_quantization");
    }

    public function testSubvectorCountRejectedForScalar(): void
    {
        $schema = new Laurus\Schema();
        $this->expectException(\Throwable::class);
        $schema->addHnswField("embedding", 4, null, 16, 200, null, null, null, 2);
    }

    public function testUnknownRerankStorageRejected(): void
    {
        $schema = new Laurus\Schema();
        $this->expectException(\Throwable::class);
        $schema->addHnswField("embedding", 4, null, 16, 200, null, null, null, null, "bogus");
    }

    // ── WAL group-commit / WalSyncPolicy (#820) ───────────────────────────

    public function testIndexWithGroupCommitWalPolicy(): void
    {
        // An index built with a group-commit policy must still behave
        // identically: documents survive flushWal() + commit() and remain
        // retrievable.
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index(null, $schema, Laurus\WalSyncPolicy::group());

        $idx->putDocument("doc1", ["title" => "Hello"]);
        // flushWal() forces the WAL durable without materializing the index.
        $idx->flushWal();
        // commit() makes the document searchable / retrievable.
        $idx->commit();

        $docs = $idx->getDocuments("doc1");
        $this->assertCount(1, $docs);
        $this->assertEquals("Hello", $docs[0]["title"]);
    }

    public function testFlushWalUnderPerRecordPolicy(): void
    {
        // flushWal() is also valid under the default per-record policy, where
        // it is effectively a no-op since every append is already durable.
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index(null, $schema, Laurus\WalSyncPolicy::perRecord());

        $idx->putDocument("doc1", ["title" => "World"]);
        $idx->flushWal();
        $idx->commit();

        $docs = $idx->getDocuments("doc1");
        $this->assertCount(1, $docs);
    }

    public function testWalSyncPolicyFactoriesAccepted(): void
    {
        // Both factories build a value object that the Index constructor
        // accepts. group() with explicit thresholds + interval must also work.
        $perRecord = Laurus\WalSyncPolicy::perRecord();
        $this->assertNotNull($perRecord);

        $group = Laurus\WalSyncPolicy::group(256, 4096, 1000);
        $this->assertNotNull($group);

        $schema = new Laurus\Schema();
        $schema->addTextField("title");

        $idx1 = new Laurus\Index(null, $schema, $perRecord);
        $this->assertNotNull($idx1);

        $idx2 = new Laurus\Index(null, $schema, $group);
        $this->assertNotNull($idx2);
    }

    public function testWalSyncPolicyDefaultsOmittedConstruct(): void
    {
        // Omitting wal_sync_policy entirely keeps the existing behavior
        // (per-record durability), preserving backward compatibility.
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index(null, $schema);
        $idx->putDocument("doc1", ["title" => "Backward compatible"]);
        $idx->commit();
        $this->assertCount(1, $idx->getDocuments("doc1"));
    }

    // ── Auto-commit / CommitPolicy (#893) ─────────────────────────────────

    public function testCommitPolicyFactoriesAccepted(): void
    {
        // Both factories build a value object that the Index constructor
        // accepts as the 4th argument. everyDocs(0) is valid (== manual).
        $manual = Laurus\CommitPolicy::manual();
        $this->assertNotNull($manual);

        $auto = Laurus\CommitPolicy::everyDocs(100);
        $this->assertNotNull($auto);

        $this->assertNotNull(Laurus\CommitPolicy::everyDocs(0));

        $schema = new Laurus\Schema();
        $schema->addTextField("title");

        $idx1 = new Laurus\Index(null, $schema, null, $manual);
        $this->assertNotNull($idx1);

        $idx2 = new Laurus\Index(null, $schema, null, $auto);
        $this->assertNotNull($idx2);
    }

    public function testCommitPolicyIntervalMsAccepted(): void
    {
        // intervalMs (#892) builds a value object that the Index constructor
        // accepts as the 4th argument.
        $interval = Laurus\CommitPolicy::intervalMs(1000);
        $this->assertNotNull($interval);

        $schema = new Laurus\Schema();
        $schema->addTextField("title");

        $idx = new Laurus\Index(null, $schema, null, Laurus\CommitPolicy::intervalMs(1000));
        $this->assertNotNull($idx);
    }

    public function testEveryDocsAutoCommitEndToEnd(): void
    {
        // An index built with everyDocs(1) is wired end-to-end: a document is
        // retrievable with no explicit commit() from the caller.
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index(null, $schema, null, Laurus\CommitPolicy::everyDocs(1));
        $idx->putDocument("d1", ["title" => "Auto"]);
        $this->assertCount(1, $idx->getDocuments("d1"));
    }

    // ── Index directory layout (Issue #1059) ──────────────────────────────
    //
    // Before this change, `new Laurus\Index($path)` wrote segment files
    // directly under $path, incompatible with laurus-cli's
    // <path>/schema.toml + <path>/store/ convention. These tests verify the
    // new shared layout: schema auto-persistence, auto-loading on reopen,
    // the reopen-with-schema conflict error, and legacy-layout detection.

    public function testFileBackedIndexWritesSchemaTomlAndStore(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_indexdir_" . uniqid();
        mkdir($dir);
        $schema = new Laurus\Schema();
        $schema->addTextField("title");

        new Laurus\Index($dir, $schema);

        $this->assertFileExists($dir . "/schema.toml");
        $this->assertDirectoryExists($dir . "/store");
        // No stray top-level segment directories from the old flat layout.
        $this->assertDirectoryDoesNotExist($dir . "/lexical");
    }

    public function testReopenWithoutSchemaLoadsPersistedSchemaAndData(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_indexdir_" . uniqid();
        mkdir($dir);
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $schema->setDefaultFields(["title"]);

        $idx = new Laurus\Index($dir, $schema);
        $idx->putDocument("doc1", ["title" => "hello world"]);
        $idx->commit();
        $idx->close();

        $reopened = new Laurus\Index($dir);
        $results = $reopened->search("title:hello", 5);
        $this->assertCount(1, $results);
    }

    public function testReopenWithExplicitSchemaThrowsValueError(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_indexdir_" . uniqid();
        mkdir($dir);
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        new Laurus\Index($dir, $schema);

        $this->expectException(\ValueError::class);
        $this->expectExceptionMessageMatches('/schema\.toml/');
        new Laurus\Index($dir, $schema);
    }

    public function testReopenWithNoSchemaAtAllSucceeds(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_indexdir_" . uniqid();
        mkdir($dir);
        // First call with no schema creates an empty-schema index (unchanged
        // default behavior); reopening it (also with no schema) must not
        // throw.
        $first = new Laurus\Index($dir);
        $first->close();
        $idx = new Laurus\Index($dir);
        $this->assertNotNull($idx);
    }

    public function testLegacyFlatLayoutIsRejected(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_indexdir_" . uniqid();
        mkdir($dir);
        // Simulate a directory written by a laurus-php version predating
        // Issue #1059: segment files directly under the path, no
        // schema.toml.
        file_put_contents($dir . "/engine.wal", "");

        $this->expectException(\ValueError::class);
        $this->expectExceptionMessageMatches('/pre-Issue-1059/');
        new Laurus\Index($dir);
    }

    public function testNewEmptyDirectoryIsNotTreatedAsLegacy(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_indexdir_" . uniqid();
        mkdir($dir);
        // A directory that merely exists but has no laurus files at all is a
        // normal fresh-create case, not a legacy-layout error.
        new Laurus\Index($dir);
        $this->assertFileExists($dir . "/schema.toml");
    }

    // ── close() (Issue #1086/#1097) ────────────────────────────────────────
    //
    // Engine::build() (Issue #1086) takes an exclusive lock on the storage
    // directory, released only when the Engine is dropped. Without an
    // explicit close(), that release depends on PHP's refcounting GC
    // dropping the last reference, which is not guaranteed once an Index is
    // captured by a closure, stored in a container, or involved in a
    // reference cycle. close() makes the release deterministic.

    public function testCloseThenReopenSucceeds(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_indexdir_" . uniqid();
        mkdir($dir);
        $idx = new Laurus\Index($dir);
        $idx->close();

        $reopened = new Laurus\Index($dir);
        $this->assertNotNull($reopened);
    }

    public function testMethodAfterCloseThrows(): void
    {
        $idx = new Laurus\Index();
        $idx->close();

        $this->expectException(\Throwable::class);
        $idx->putDocument("doc1", ["title" => "hello"]);
    }

    public function testCloseIsIdempotent(): void
    {
        $idx = new Laurus\Index();
        $idx->close();
        $idx->close();
        $this->assertNotNull($idx);
    }

    // ── peek_commit_generation (Issue #1101) ──────────────────────────────
    //
    // Unlike Index::stats(), this is a namespaced global function, not tied
    // to any Index instance: it reads commit_generation.json directly from
    // disk without building an Engine at all, so it works even when no
    // Index for the given path has ever been constructed in this process --
    // the point being to let a caller cheaply decide whether opening (or
    // reopening) the index is worth doing at all.

    public function testPeekCommitGenerationRejectsANonIndexDirectory(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_peek_" . uniqid();
        mkdir($dir);

        $this->expectException(\ValueError::class);
        $this->expectExceptionMessageMatches('/not a laurus index directory/');
        Laurus\peek_commit_generation($dir);
    }

    public function testPeekCommitGenerationIsZeroBeforeAnyCommit(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_peek_" . uniqid();
        mkdir($dir);
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        new Laurus\Index($dir, $schema);

        $this->assertEquals(0, Laurus\peek_commit_generation($dir));
    }

    public function testPeekCommitGenerationAdvancesAfterACommit(): void
    {
        // Index::stats() doesn't expose a commitGeneration key yet in this
        // binding (that's currently laurus-python-only), so this only
        // checks the raw counter.
        $dir = sys_get_temp_dir() . "/laurus_peek_" . uniqid();
        mkdir($dir);
        $schema = new Laurus\Schema();
        $schema->addTextField("title");
        $idx = new Laurus\Index($dir, $schema);
        $idx->putDocument("doc1", ["title" => "hello world"]);
        $idx->commit();

        $this->assertEquals(1, Laurus\peek_commit_generation($dir));
    }

    public function testPeekCommitGenerationSeesACommitMadeByAnotherHandle(): void
    {
        $dir = sys_get_temp_dir() . "/laurus_peek_" . uniqid();
        mkdir($dir);
        $a = new Laurus\Index($dir);
        $a->putDocument("doc1", []);
        $a->commit();
        $a->close();

        $before = Laurus\peek_commit_generation($dir);

        $b = new Laurus\Index($dir);
        $b->putDocument("doc2", []);
        $b->commit();
        $b->close();

        $this->assertNotEquals($before, Laurus\peek_commit_generation($dir));
    }
}
