# frozen_string_literal: true

require_relative "test_helper"

# Integration tests for Laurus::Index#search_batch
# (Phase 3d of #648, issue #719).
#
# Covers:
# - Empty input returns an empty Array without invoking the engine.
# - Single-query batch matches the single-query #search result.
# - Multi-query batch preserves input order and returns one result Array
#   per input query.
# - DSL strings, Query objects, and SearchRequest are all accepted.
class TestSearchBatch < Minitest::Test
  def create_index
    idx = Laurus::Index.new
    idx.put_document("doc1", { "title" => "Introduction to Rust", "body" => "Systems programming language." })
    idx.put_document("doc2", { "title" => "Python for Data Science", "body" => "Data analysis with Python." })
    idx.put_document("doc3", { "title" => "Distributed Systems", "body" => "Engineering at scale." })
    idx.commit
    idx
  end

  def test_search_batch_empty
    idx = create_index
    results = idx.search_batch([])
    assert_equal [], results
  end

  def test_search_batch_single_query_matches_search
    idx = create_index
    serial = idx.search("title:rust", limit: 5)
    batch = idx.search_batch(["title:rust"], limit: 5)

    assert_equal 1, batch.length
    assert_equal serial.length, batch[0].length
    batch[0].each_with_index do |result, i|
      assert_equal serial[i].id, result.id
    end
  end

  def test_search_batch_multi_query_preserves_order
    idx = create_index
    queries = ["title:rust", "body:python", "title:distributed"]
    expected_top_ids = %w[doc1 doc2 doc3]

    batch = idx.search_batch(queries, limit: 5)
    assert_equal queries.length, batch.length

    batch.each_with_index do |results, i|
      assert results.length >= 1, "expected at least 1 hit for query #{queries[i]}"
      assert_equal expected_top_ids[i], results[0].id
    end
  end

  def test_search_batch_with_query_objects
    idx = create_index
    queries = [
      Laurus::TermQuery.new("title", "rust"),
      Laurus::TermQuery.new("body", "python"),
    ]
    batch = idx.search_batch(queries, limit: 5)

    assert_equal 2, batch.length
    assert_equal "doc1", batch[0][0].id
    assert_equal "doc2", batch[1][0].id
  end

  def test_search_batch_with_search_requests
    idx = create_index
    requests = [
      Laurus::SearchRequest.new(lexical_query: Laurus::TermQuery.new("title", "rust"), limit: 5),
      Laurus::SearchRequest.new(lexical_query: Laurus::TermQuery.new("body", "python"), limit: 5),
    ]
    batch = idx.search_batch(requests)

    assert_equal 2, batch.length
    assert_equal "doc1", batch[0][0].id
    assert_equal "doc2", batch[1][0].id
  end

  def test_search_batch_no_match_returns_empty_inner_array
    idx = create_index
    queries = ["title:rust", "title:nonexistent_xyz"]
    batch = idx.search_batch(queries, limit: 5)

    assert_equal 2, batch.length
    assert batch[0].length >= 1
    assert_equal [], batch[1]
  end

  def test_search_batch_limit_per_query
    idx = create_index
    queries = ["body:programming OR body:data", "body:programming OR body:data"]
    batch = idx.search_batch(queries, limit: 1)

    assert_equal 2, batch.length
    batch.each do |results|
      assert results.length <= 1
    end
  end

  # Issue #1553: the engine wraps the error of a batch with the query's
  # position, which the binding unwraps, so a query the engine rejects
  # raises the ArgumentError that #search raises for it.
  def test_search_batch_engine_error_raises_as_search_does
    idx = create_index
    single = assert_raises(ArgumentError) { idx.search("nosuch:rust") }
    batch = assert_raises(ArgumentError) { idx.search_batch(["title:rust", "nosuch:rust"]) }

    assert_includes batch.message, "unknown field"
    assert_equal single.message, batch.message
  end
end
