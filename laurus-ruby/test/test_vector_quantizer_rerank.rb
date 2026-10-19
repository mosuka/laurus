# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

# Tests for the HNSW quantizer / rerank_storage schema options (Issue #797).
#
# These assert the values configured on +add_hnsw_field+ actually reach the
# Rust core via deterministic observables, not merely that search succeeds:
#
# * +rerank_storage: "f32"+ makes the core write a +*.hnsw.f32+ Stage-2
#   sidecar on disk (mirrors the server-side guard in #793/#800); the
#   default writes no sidecar.
# * +quantizer: "product_quantization"+ forwards +subvector_count+ to the
#   core's PQ training, which rejects a count that does not divide the
#   field dimension.
# * +quantizer: "float16"+ (Issue #623) separates two vectors that the
#   default int8 quantizer stores as the same code.
class TestVectorQuantizerRerank < Minitest::Test
  # Stable two-cluster corpus mirroring the core's
  # +test_hnsw_pq_search_returns_corpus_neighbour+ (issue #730).
  NEAR_OFFSETS = [
    [0.0, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.1, 0.1],
    [-0.1, -0.1, -0.1, -0.1],
    [0.2, -0.2, 0.2, -0.2],
    [-0.2, 0.2, -0.2, 0.2],
    [0.05, 0.05, -0.05, -0.05],
    [-0.05, -0.05, 0.05, 0.05],
    [0.15, -0.1, 0.1, -0.15],
  ].freeze
  NEAR_BASE = [10.0, 10.0, 20.0, 20.0].freeze
  FAR_BASE = [-100.0, -100.0, -200.0, -200.0].freeze

  def test_rerank_storage_f32_writes_sidecar
    Dir.mktmpdir do |dir|
      schema = Laurus::Schema.new
      schema.add_hnsw_field("embedding", 4, rerank_storage: "f32")
      idx = Laurus::Index.new(path: dir, schema: schema)
      idx.put_document("doc1", { "embedding" => [0.1, 0.2, 0.3, 0.4] })
      idx.put_document("doc2", { "embedding" => [0.9, 0.8, 0.7, 0.6] })
      idx.commit
      refute_empty Dir.glob(File.join(dir, "**", "*.hnsw.f32")),
                   "rerank_storage: 'f32' must write a .hnsw.f32 sidecar"
    end
  end

  def test_no_rerank_storage_writes_no_sidecar
    Dir.mktmpdir do |dir|
      schema = Laurus::Schema.new
      schema.add_hnsw_field("embedding", 4)
      idx = Laurus::Index.new(path: dir, schema: schema)
      idx.put_document("doc1", { "embedding" => [0.1, 0.2, 0.3, 0.4] })
      idx.put_document("doc2", { "embedding" => [0.9, 0.8, 0.7, 0.6] })
      idx.commit
      assert_empty Dir.glob(File.join(dir, "**", "*.hnsw.f32"))
    end
  end

  def test_product_quantization_builds_and_searches
    schema = Laurus::Schema.new
    # PQ is an L2 quantizer, so use Euclidean (matching the core's
    # test_hnsw_pq_search_returns_corpus_neighbour).
    schema.add_hnsw_field(
      "embedding", 4,
      distance: "euclidean", quantizer: "product_quantization", subvector_count: 2
    )
    idx = Laurus::Index.new(schema: schema)
    # 128 points per cluster (256 total) so the segment meets the PQ
    # min-train threshold (#880: smaller PQ-configured segments are written
    # as Scalar8Bit and would not exercise PQ training at all).
    128.times do |i|
      off = [
        (i % 8) * 0.04 - 0.14,
        ((i / 8) % 8) * 0.04 - 0.14,
        ((i / 64) % 8) * 0.04 - 0.14,
        (i % 16) * 0.04 - 0.32
      ]
      idx.put_document("near#{i}", { "embedding" => NEAR_BASE.map.with_index { |b, j| b + off[j] } })
      idx.put_document("far#{i}", { "embedding" => FAR_BASE.map.with_index { |b, j| b + off[j] } })
    end
    idx.commit

    results = idx.search(Laurus::VectorQuery.new("embedding", NEAR_BASE), limit: 3)
    assert_equal 3, results.length
    assert(results.all? { |r| r.id.start_with?("near") })
  end

  def test_pq_subvector_count_must_divide_dimension
    schema = Laurus::Schema.new
    schema.add_hnsw_field(
      "embedding", 4, quantizer: "product_quantization", subvector_count: 3
    )
    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "embedding" => [0.1, 0.2, 0.3, 0.4] })
    assert_raises { idx.commit }
  end

  def test_unknown_quantizer_rejected
    schema = Laurus::Schema.new
    assert_raises(ArgumentError) { schema.add_hnsw_field("embedding", 4, quantizer: "bogus") }
  end

  def test_pq_requires_subvector_count
    schema = Laurus::Schema.new
    assert_raises(ArgumentError) do
      schema.add_hnsw_field("embedding", 4, quantizer: "product_quantization")
    end
  end

  def test_subvector_count_rejected_for_scalar
    schema = Laurus::Schema.new
    assert_raises(ArgumentError) { schema.add_hnsw_field("embedding", 4, subvector_count: 2) }
  end

  def test_unknown_rerank_storage_rejected
    schema = Laurus::Schema.new
    assert_raises(ArgumentError) { schema.add_hnsw_field("embedding", 4, rerank_storage: "bogus") }
  end

  def test_pq_rejected_for_flat_field
    schema = Laurus::Schema.new
    error = assert_raises(ArgumentError) do
      schema.add_flat_field("embedding", 4, quantizer: "product_quantization")
    end
    assert_match(/only supported by HNSW/, error.message)
  end

  def test_float16_hnsw_field_ranks_nearest_first
    schema = Laurus::Schema.new
    schema.add_hnsw_field("embedding", 2, distance: "euclidean", quantizer: "float16")
    assert_float16_ranks_exact_match_first(schema)
  end

  def test_float16_flat_field_ranks_nearest_first
    schema = Laurus::Schema.new
    schema.add_flat_field("embedding", 2, distance: "euclidean", quantizer: "float16")
    assert_float16_ranks_exact_match_first(schema)
  end

  private

  # +low+ / +high+ stretch the segment's int8 range to [-100, 100] (step
  # ~0.78), so +a+ and +b+ share one int8 code and tie under the default
  # quantizer; Float16 keeps them apart, so the exact match +b+ must rank
  # first.
  def assert_float16_ranks_exact_match_first(schema)
    idx = Laurus::Index.new(schema: schema)
    idx.put_document("low", { "embedding" => [-100.0, -100.0] })
    idx.put_document("high", { "embedding" => [100.0, 100.0] })
    idx.put_document("a", { "embedding" => [0.2, 0.0] })
    idx.put_document("b", { "embedding" => [0.25, 0.0] })
    idx.commit

    results = idx.search(Laurus::VectorQuery.new("embedding", [0.25, 0.0]), limit: 2)
    assert_equal %w[b a], results.map(&:id)
  end
end
