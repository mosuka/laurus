# frozen_string_literal: true

require_relative "test_helper"

# Multi-vector fields and the late-interaction rescore (Issue #1351).
#
# Uses the same corpus as the Rust test
# (`laurus/tests/late_interaction_rescore_test.rs`): against the query token
# vectors [[1, 0], [0, 1]] the MaxSim scores are c 1.1, b 1.0, d 0.9,
# a 0.1, e 0.05 — an order that matches neither the BM25 nor the `vec`
# ranking — so the binding must rank exactly like the Rust API.
class TestLateInteractionRescore < Minitest::Test
  QUERY = [[1.0, 0.0], [0.0, 1.0]].freeze
  EXPECTED = %w[c b d a e].freeze
  CORPUS = [
    ["a", "rust", [1.0, 0.0], [[0.1, 0.0]]],
    ["b", "rust rust", [0.9, 0.1], [[0.5, 0.5], [0.0, 0.2]]],
    ["c", "rust language", [0.5, 0.5], [[0.9, 0.2]]],
    ["d", "rust rust rust", [0.2, 0.8], [[0.3, 0.3], [0.6, 0.0]]],
    ["e", "learning rust today", [0.0, 1.0], [[0.02, 0.03]]]
  ].freeze
  TOKENS = CORPUS.to_h { |id, _, _, tokens| [id, tokens] }.freeze

  def setup
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_flat_field("vec", 2)
    schema.add_multi_vector_field("tokens", 2, distance: "dot_product")
    @index = Laurus::Index.new(schema: schema)
    CORPUS.each do |id, title, vec, tokens|
      @index.put_document(id, { "title" => title, "vec" => vec, "tokens" => tokens })
    end
    @index.commit
  end

  def max_sim(query, tokens)
    query.sum { |q| tokens.map { |t| q.zip(t).sum { |a, b| a * b } }.max }
  end

  def rescore(**kwargs)
    Laurus::LateInteractionRescore.new("tokens", QUERY, **kwargs)
  end

  def test_dsl_search_is_reordered_by_late_interaction
    refute_equal EXPECTED, @index.search("title:rust").map(&:id)

    results = @index.search("title:rust", rescore: rescore)

    assert_equal EXPECTED, results.map(&:id)
    results.each do |r|
      assert_in_delta max_sim(QUERY, TOKENS[r.id]), r.score, 1e-5
    end
  end

  def test_search_request_is_reordered_by_late_interaction
    request = Laurus::SearchRequest.new(query: "title:rust", rescore: rescore)
    assert_equal EXPECTED, @index.search(request).map(&:id)

    hybrid = Laurus::SearchRequest.new(
      lexical_query: Laurus::TermQuery.new("title", "rust"),
      vector_query: Laurus::VectorQuery.new("vec", [1.0, 0.0]),
      fusion: Laurus::RRF.new,
      rescore: rescore
    )
    assert_equal EXPECTED, @index.search(hybrid).map(&:id)
  end

  def test_only_the_window_is_reordered
    baseline = @index.search("title:rust")
    window = rescore(window_size: 1)
    assert_equal 1, window.window_size

    results = @index.search("title:rust", rescore: window)

    assert_equal baseline.map(&:id), results.map(&:id)
    assert_in_delta max_sim(QUERY, TOKENS[results[0].id]), results[0].score, 1e-5
    results.drop(1).zip(baseline.drop(1)).each do |r, b|
      assert_equal b.score, r.score
    end
  end

  def test_default_window_size_and_inspect
    assert_equal 100, rescore.window_size
    assert_equal 100, rescore(window_size: nil).window_size
    assert_equal 'LateInteractionRescore(field="tokens", window_size=100)', rescore.inspect
  end

  def test_token_vectors_are_not_stored
    # Token vectors live only in the vector store, unlike a single vector.
    doc = @index.get_documents("b")[0]
    refute doc.key?("tokens")
    assert_equal 2, doc["vec"].length
  end

  def test_query_must_be_text_or_token_vectors
    [[1.0, 0.0], [[1.0], "x"], 42, nil].each do |query|
      err = assert_raises(TypeError, query.inspect) do
        Laurus::LateInteractionRescore.new("tokens", query)
      end
      assert_match(/query must be a String or an Array of numeric Arrays|must hold only numbers/, err.message)
    end
  end

  def test_rescore_must_be_a_late_interaction_rescore
    err = assert_raises(TypeError) { @index.search("title:rust", rescore: QUERY) }
    assert_includes err.message, "rescore must be a Laurus::LateInteractionRescore"
  end

  def test_invalid_rescore_is_rejected
    {
      Laurus::LateInteractionRescore.new("tokens", "rust") => "has no token-level embedder",
      Laurus::LateInteractionRescore.new("vec", QUERY) => "needs a MultiVector field",
      Laurus::LateInteractionRescore.new("tokens", []) => "between 1 and 1024 query vectors",
      Laurus::LateInteractionRescore.new("tokens", [[1.0, 0.0, 0.0]]) => "has dimension 3",
      rescore(window_size: 0) => "window_size"
    }.each do |invalid, message|
      err = assert_raises(ArgumentError, message) { @index.search("title:rust", rescore: invalid) }
      assert_includes err.message, message
    end
  end

  def test_invalid_token_vectors_are_rejected
    assert_raises(ArgumentError) do
      @index.put_document("x", { "title" => "rust", "tokens" => [[1.0, 0.0], [0.0]] })
    end
    [[[1.0, true]], [[1.0, "x"]]].each do |tokens|
      assert_raises(TypeError, tokens.inspect) do
        @index.put_document("x", { "title" => "rust", "tokens" => tokens })
      end
    end
  end

  def test_integer_token_vectors_are_accepted
    @index.put_document("x", { "title" => "rust", "tokens" => [[2, 0]] })
    @index.commit
    results = @index.search("title:rust", rescore: rescore)
    assert_equal "x", results[0].id
    assert_in_delta 2.0, results[0].score, 1e-5
  end

  def test_add_multi_vector_field_rejects_invalid_options
    [[[0], {}, "dimension"], [[2], { distance: "euclidean" }, "distance"]].each do |args, kwargs, message|
      schema = Laurus::Schema.new
      err = assert_raises(ArgumentError, message) { schema.add_multi_vector_field("tokens", *args, **kwargs) }
      assert_includes err.message, message
      assert_equal [], schema.field_names
    end
  end

  def test_add_embedder_accepts_every_core_type_with_symbol_keys
    schema = Laurus::Schema.new
    schema.add_embedder(
      "colbert",
      { type: "candle_colbert", model: "colbert-ir/colbertv2.0", revision: "main", query_maxlen: 16, doc_maxlen: 64 }
    )
    schema.add_multi_vector_field("tokens", 128, embedder: "colbert")

    toml = Laurus::Schema.from_toml(schema.to_toml).to_toml
    assert_includes toml, 'type = "candle_colbert"'
    assert_includes toml, "query_maxlen = 16"
    assert_includes toml, 'embedder = "colbert"'
  end

  def test_add_embedder_rejects_invalid_config
    {
      { "type" => "candle_bert" } => "missing field `model`",
      { "type" => "nope" } => "unknown variant",
      {} => "missing field `type`"
    }.each do |config, message|
      err = assert_raises(ArgumentError, message) { Laurus::Schema.new.add_embedder("e", config) }
      assert_includes err.message, message
    end
  end
end
