# frozen_string_literal: true

require_relative "test_helper"

# Result grouping (`group_by:`, Issue #1297).
#
# Uses the corpus of `laurus-server/tests/grpc_group_by_test.rs`: three
# documents added as chunks sharing their `_id`, with Flat vectors so the
# vector ranking against [1, 0] — and with it the groups — is exact.
class TestGroupBy < Minitest::Test
  CHUNKS = [
    ["a", "rust rust rust guide", [1.0, 0.0]],
    ["a", "rust rust intro", [0.9, 0.1]],
    ["a", "rust appendix notes", [0.3, 0.7]],
    ["b", "rust rust book chapter", [0.8, 0.2]],
    ["b", "rust history of programming", [0.2, 0.8]],
    ["c", "rust", [0.6, 0.4]]
  ].freeze

  # Each group as [id, titles of its hits, best first] with group_size 2.
  EXPECTED = [
    ["a", ["rust rust rust guide", "rust rust intro"]],
    ["b", ["rust rust book chapter", "rust history of programming"]],
    ["c", ["rust"]]
  ].freeze

  def setup
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_flat_field("vec", 2)
    @index = Laurus::Index.new(schema: schema)
    CHUNKS.each do |id, title, vec|
      @index.add_document(id, { "title" => title, "vec" => vec })
    end
    @index.commit
  end

  def vector_query
    Laurus::VectorQuery.new("vec", [1.0, 0.0])
  end

  def groups(results)
    results.map do |r|
      assert_equal r.id, r.group_key
      r.inner_hits.each do |hit|
        assert_equal r.group_key, hit.group_key
        assert_empty hit.inner_hits
      end
      [r.id, [r, *r.inner_hits].map { |h| h.document["title"] }]
    end
  end

  def test_search_groups_chunks_by_id
    results = @index.search(vector_query, group_by: Laurus::GroupBy.new("_id", group_size: 2))
    assert_equal EXPECTED, groups(results)

    one_each = @index.search(vector_query, group_by: Laurus::GroupBy.new("_id"))
    assert_equal %w[a b c], one_each.map(&:id)
    assert(one_each.all? { |r| r.inner_hits.empty? })
  end

  def test_ungrouped_results_have_no_group
    results = @index.search(vector_query)
    assert_equal CHUNKS.size, results.size
    results.each do |r|
      assert_nil r.group_key
      assert_empty r.inner_hits
    end
  end

  def test_search_request_carries_its_own_group_by
    request = Laurus::SearchRequest.new(
      vector_query: vector_query,
      group_by: Laurus::GroupBy.new("_id", group_size: 2)
    )
    assert_equal EXPECTED, groups(@index.search(request))
  end

  def test_search_batch_groups_every_query
    batch = @index.search_batch(
      [vector_query, vector_query],
      group_by: Laurus::GroupBy.new("_id", group_size: 2)
    )
    assert_equal [EXPECTED, EXPECTED], batch.map { |results| groups(results) }
  end

  def test_group_by_attributes
    group_by = Laurus::GroupBy.new("parent", group_size: 3)
    assert_equal ["parent", 3], [group_by.field, group_by.group_size]
    assert_equal 'GroupBy(field="parent", group_size=3)', group_by.inspect
  end

  def test_group_by_must_be_a_group_by
    assert_raises(TypeError) { @index.search(vector_query, group_by: "_id") }
  end

  def test_invalid_group_by_raises_argument_error
    [
      Laurus::GroupBy.new("missing"),
      Laurus::GroupBy.new("vec"),
      Laurus::GroupBy.new("_id", group_size: 0)
    ].each do |group_by|
      error = assert_raises(ArgumentError) { @index.search(vector_query, group_by: group_by) }
      assert_match(/group_by/, error.message)
    end
  end
end
