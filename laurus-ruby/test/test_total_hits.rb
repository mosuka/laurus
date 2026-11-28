# frozen_string_literal: true

require_relative "test_helper"

# The total hit count of a search (Issue #1472).
#
# `Index#search`, `Index#search_batch` and `SearchRequest.new` take
# `track_total_hits:`, read as the HTTP gateway reads `"track_total_hits"`:
# an Integer counts exactly up to it, `true` counts every match, and `false`
# or `nil` counts nothing. Each search returns a `Laurus::SearchResponse`
# whose `total_hits` holds the count and whether it is exact (`"eq"`) or a
# lower bound (`"gte"`).
class TestTotalHits < Minitest::Test
  MESSAGE = /\Atrack_total_hits must be true, false or an Integer from 0 to 18446744073709551615, got /

  # Five documents match `body:rust`, one matches `body:python`.
  def setup
    schema = Laurus::Schema.new
    schema.add_text_field("body")
    @index = Laurus::Index.new(schema: schema)
    5.times { |i| @index.put_document("rust#{i}", { "body" => "rust #{i}" }) }
    @index.put_document("python", { "body" => "python" })
    @index.commit
  end

  def total(response)
    t = response.total_hits
    t.nil? ? nil : [t.value, t.relation]
  end

  def test_search_counts_the_matches_as_asked
    [[1000, [5, "eq"]], [3, [3, "gte"]], [true, [5, "eq"]], [false, nil], [nil, nil]].each do |track, expected|
      response = @index.search("body:rust", limit: 2, track_total_hits: track)

      assert_kind_of Laurus::SearchResponse, response
      assert_equal 2, response.results.length, "track_total_hits: #{track.inspect}"
      if expected.nil?
        assert_nil total(response), "track_total_hits: #{track.inspect}"
      else
        assert_equal expected, total(response), "track_total_hits: #{track.inspect}"
      end
    end
  end

  def test_search_counts_nothing_by_default
    response = @index.search("body:rust", limit: 2)

    assert_equal 2, response.results.length
    assert_nil response.total_hits
  end

  def test_search_does_not_read_a_truthy_value_as_true
    # Ruby's truthiness would make any of these "true" (count every match).
    ["1000", 1.5, { "up_to" => 3 }, :all].each do |value|
      error = assert_raises(TypeError) { @index.search("body:rust", track_total_hits: value) }
      assert_match MESSAGE, error.message
    end
  end

  def test_search_rejects_an_integer_out_of_range
    [-1, 2**64].each do |value|
      error = assert_raises(ArgumentError) { @index.search("body:rust", track_total_hits: value) }
      assert_match MESSAGE, error.message
    end
  end

  def test_search_rejects_zero_through_the_engine
    error = assert_raises(ArgumentError) { @index.search("body:rust", track_total_hits: 0) }
    assert_match(/track_total_hits must count at least 1 match/, error.message)
  end

  def test_search_request_carries_its_own_track_total_hits
    request = Laurus::SearchRequest.new(query: "body:rust", limit: 2, track_total_hits: 3)
    assert_equal [3, "gte"], total(@index.search(request))

    # A SearchRequest uses its own value, as it does its own limit.
    plain = Laurus::SearchRequest.new(query: "body:rust", limit: 2)
    assert_nil total(@index.search(plain, track_total_hits: true))
  end

  def test_search_request_takes_track_total_hits_with_group_by
    # Both keywords are taken out of the keyword hash before the rest.
    request = Laurus::SearchRequest.new(
      query: "body:rust", limit: 2, group_by: Laurus::GroupBy.new("_id"), track_total_hits: true
    )
    assert_equal [5, "eq"], total(@index.search(request))
  end

  def test_search_request_rejects_another_value
    error = assert_raises(TypeError) { Laurus::SearchRequest.new(query: "body:rust", track_total_hits: "1000") }
    assert_match MESSAGE, error.message
  end

  def test_search_batch_returns_each_query_s_total
    responses = @index.search_batch(["body:rust", "body:python", "body:missing"], limit: 2, track_total_hits: 1000)

    assert_equal [[5, "eq"], [1, "eq"], [0, "eq"]], responses.map { |r| total(r) }
    assert_equal [2, 1, 0], responses.map { |r| r.results.length }
  end

  def test_search_batch_lets_a_search_request_use_its_own_value
    request = Laurus::SearchRequest.new(query: "body:rust", limit: 2, track_total_hits: 3)
    responses = @index.search_batch(["body:rust", request], limit: 2, track_total_hits: true)

    assert_equal [[5, "eq"], [3, "gte"]], responses.map { |r| total(r) }
  end

  def test_search_batch_counts_nothing_by_default
    responses = @index.search_batch(["body:rust"], limit: 2)

    assert_equal [nil], responses.map { |r| total(r) }
  end

  def test_search_batch_rejects_another_value
    error = assert_raises(TypeError) { @index.search_batch(["body:rust"], track_total_hits: "1000") }
    assert_match MESSAGE, error.message
  end

  def test_response_and_total_hits_inspect
    response = @index.search("body:rust", limit: 2, track_total_hits: 3)

    assert_equal "SearchResponse(results=2, total_hits=TotalHits(value=3, relation='gte'))", response.inspect
    assert_equal "TotalHits(value=3, relation='gte')", response.total_hits.inspect
    assert_equal "SearchResponse(results=2, total_hits=nil)", @index.search("body:rust", limit: 2).inspect
  end
end
