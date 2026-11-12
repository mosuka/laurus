# frozen_string_literal: true

require_relative "test_helper"

# SearchRequest's `fusion:` (Issue #1555).
#
# A fusion that was neither an RRF nor a WeightedSum was dropped, so the
# search ran with the default RRF (k = 60) instead of failing.
class TestSearchRequestFusion < Minitest::Test
  DOCS = [
    ["a", "rust a", [1.0, 0.0]],
    ["b", "rust b", [0.0, 1.0]],
    ["c", "python c", [0.8, 0.6]]
  ].freeze

  Other = Class.new

  def setup
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_flat_field("vec", 2)
    @index = Laurus::Index.new(schema: schema)
    DOCS.each do |id, title, vec|
      @index.put_document(id, { "title" => title, "vec" => vec })
    end
    @index.commit
  end

  def hybrid_request(fusion)
    Laurus::SearchRequest.new(
      lexical_query: Laurus::TermQuery.new("title", "rust"),
      vector_query: Laurus::VectorQuery.new("vec", [1.0, 0.0]),
      fusion: fusion,
      limit: 10
    )
  end

  def scores(fusion)
    @index.search(hybrid_request(fusion)).to_h { |r| [r.id, r.score.round(6)] }
  end

  def test_rejects_a_fusion_of_another_type
    expected = "fusion must be a Laurus::RRF or a Laurus::WeightedSum"
    mismatches = ["rrf", { k: 1.0 }, 42, Other.new].filter_map do |fusion|
      hybrid_request(fusion)
      "#{fusion.inspect}: no error"
    rescue TypeError => e
      "#{fusion.inspect}: #{e.message}" unless e.message == expected
    end
    assert_empty mismatches
  end

  def test_applies_nil_rrf_and_weighted_sum
    default = scores(nil)
    assert_equal default, scores(Laurus::RRF.new(k: 60.0))
    rrf1 = scores(Laurus::RRF.new(k: 1.0))
    assert_in_delta 1.0, rrf1["a"]
    refute_equal default, rrf1
    weighted = scores(Laurus::WeightedSum.new(lexical_weight: 0.9, vector_weight: 0.1))
    refute_includes [default, rrf1], weighted
  end
end
