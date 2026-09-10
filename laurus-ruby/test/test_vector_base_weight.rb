# frozen_string_literal: true

require_relative "test_helper"

# Tests for the +base_weight+ vector field option (Issue #1084).
#
# Confirms +base_weight+ reaches the Rust core and actually affects search
# scores, using a deterministic observable rather than only that search
# succeeds: two HNSW fields holding the SAME query-matching vector, with
# different +base_weight+, must produce scores whose ratio tracks the
# +base_weight+ ratio.
class TestVectorBaseWeight < Minitest::Test
  SAME_VECTOR = [1.0, 0.0, 0.0, 0.0].freeze

  def test_base_weight_defaults_to_one
    schema = Laurus::Schema.new
    schema.add_hnsw_field("embedding", 4)
    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "embedding" => SAME_VECTOR })
    idx.commit
    score = idx.search(Laurus::VectorQuery.new("embedding", SAME_VECTOR), limit: 1).first.score

    schema2 = Laurus::Schema.new
    schema2.add_hnsw_field("embedding", 4, base_weight: 1.0)
    idx2 = Laurus::Index.new(schema: schema2)
    idx2.put_document("doc1", { "embedding" => SAME_VECTOR })
    idx2.commit
    score2 = idx2.search(Laurus::VectorQuery.new("embedding", SAME_VECTOR), limit: 1).first.score

    assert_in_delta score, score2, 1e-6,
                     "an unset base_weight must score identically to an explicit 1.0"
  end

  def test_base_weight_affects_vector_score
    schema_a = Laurus::Schema.new
    schema_a.add_hnsw_field("vec_a", 4, base_weight: 1.0)
    idx_a = Laurus::Index.new(schema: schema_a)
    idx_a.put_document("doc1", { "vec_a" => SAME_VECTOR })
    idx_a.commit
    score_a = idx_a.search(Laurus::VectorQuery.new("vec_a", SAME_VECTOR), limit: 1).first.score

    schema_b = Laurus::Schema.new
    schema_b.add_hnsw_field("vec_b", 4, base_weight: 3.0)
    idx_b = Laurus::Index.new(schema: schema_b)
    idx_b.put_document("doc1", { "vec_b" => SAME_VECTOR })
    idx_b.commit
    score_b = idx_b.search(Laurus::VectorQuery.new("vec_b", SAME_VECTOR), limit: 1).first.score

    assert score_b > score_a,
           "base_weight=3.0 must score higher than base_weight=1.0: a=#{score_a}, b=#{score_b}"
    ratio = score_b / score_a
    assert_in_delta 3.0, ratio, 0.05,
                     "score ratio must track base_weight ratio (~3.0), got #{ratio}"
  end

  def test_non_positive_base_weight_falls_back_to_one
    schema_zero = Laurus::Schema.new
    schema_zero.add_hnsw_field("embedding", 4, base_weight: 0.0)
    idx_zero = Laurus::Index.new(schema: schema_zero)
    idx_zero.put_document("doc1", { "embedding" => SAME_VECTOR })
    idx_zero.commit
    score_zero = idx_zero.search(Laurus::VectorQuery.new("embedding", SAME_VECTOR), limit: 1).first.score

    schema_one = Laurus::Schema.new
    schema_one.add_hnsw_field("embedding", 4, base_weight: 1.0)
    idx_one = Laurus::Index.new(schema: schema_one)
    idx_one.put_document("doc1", { "embedding" => SAME_VECTOR })
    idx_one.commit
    score_one = idx_one.search(Laurus::VectorQuery.new("embedding", SAME_VECTOR), limit: 1).first.score

    assert_in_delta score_zero, score_one, 1e-6,
                     "base_weight=0.0 must clamp to 1.0 rather than zeroing the score"
  end
end
