# frozen_string_literal: true

require_relative "test_helper"

# An explicit `nil` for a keyword argument whose documented default is `nil`
# (Issue #1616).
#
# magnus's `get_kwargs` passes an explicit `nil` as a value, so a keyword
# read as `Option<Value>` or `Option<Integer/String/Float>` raised for it,
# where the API reference gives `nil` as the default. Each case passes `nil`
# to one keyword and compares the result with the same call leaving the
# keyword out; the mismatches are collected and asserted once.
class TestExplicitNil < Minitest::Test
  TOKYO_TOWER = { "x" => -3_955_182.0, "y" => 3_350_553.0, "z" => 3_700_276.0 }.freeze
  MT_FUJI = { "x" => -3_916_073.0, "y" => 3_437_037.0, "z" => 3_672_751.0 }.freeze

  def setup
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_flat_field("vec", 2)
    schema.add_geo3d_field("p")
    @index = Laurus::Index.new(schema: schema)
    @index.put_document("a", { "title" => "rust a", "vec" => [1.0, 0.0], "p" => TOKYO_TOWER })
    @index.put_document("b", { "title" => "rust b", "vec" => [0.0, 1.0], "p" => MT_FUJI })
    @index.commit
  end

  def hits(results)
    results.map { |r| [r.id, r.score.round(6)] }
  end

  def toml(&block)
    schema = Laurus::Schema.new
    block.call(schema)
    schema.to_toml
  end

  def search_request(**kwargs)
    hits(@index.search(Laurus::SearchRequest.new(**kwargs)))
  end

  def lexical
    Laurus::TermQuery.new("title", "rust")
  end

  def vector
    Laurus::VectorQuery.new("vec", [1.0, 0.0])
  end

  def nearest(**kwargs)
    hits(@index.search(Laurus::Geo3dNearestQuery.k_nearest("p", *TOKYO_TOWER.values, 2, **kwargs)))
  end

  # [keyword, call with `nil`, call without the keyword]
  def cases
    [
      ["Index.new path:", -> { Laurus::Index.new(path: nil).stats }, -> { Laurus::Index.new.stats }],
      ["Index.new schema:", -> { Laurus::Index.new(schema: nil).stats }, -> { Laurus::Index.new.stats }],
      ["Index.new wal_sync_policy:", -> { Laurus::Index.new(wal_sync_policy: nil).stats },
       -> { Laurus::Index.new.stats }],
      ["Index.new commit_policy:", -> { Laurus::Index.new(commit_policy: nil).stats },
       -> { Laurus::Index.new.stats }],
      ["WalSyncPolicy.group max_records:", -> { Laurus::WalSyncPolicy.group(max_records: nil).inspect },
       -> { Laurus::WalSyncPolicy.group.inspect }],
      ["WalSyncPolicy.group max_bytes:", -> { Laurus::WalSyncPolicy.group(max_bytes: nil).inspect },
       -> { Laurus::WalSyncPolicy.group.inspect }],
      ["WalSyncPolicy.group max_interval_ms:", -> { Laurus::WalSyncPolicy.group(max_interval_ms: nil).inspect },
       -> { Laurus::WalSyncPolicy.group.inspect }],
      ["Index#search highlight:", -> { hits(@index.search("title:rust", highlight: nil)) },
       -> { hits(@index.search("title:rust")) }],
      ["Index#search rescore:", -> { hits(@index.search("title:rust", rescore: nil)) },
       -> { hits(@index.search("title:rust")) }],
      ["Index#search group_by:", -> { hits(@index.search("title:rust", group_by: nil)) },
       -> { hits(@index.search("title:rust")) }],
      ["Index#search_batch highlight:", -> { @index.search_batch(["title:rust"], highlight: nil).map { |r| hits(r) } },
       -> { @index.search_batch(["title:rust"]).map { |r| hits(r) } }],
      ["Index#search_batch group_by:", -> { @index.search_batch(["title:rust"], group_by: nil).map { |r| hits(r) } },
       -> { @index.search_batch(["title:rust"]).map { |r| hits(r) } }],
      ["add_text_field analyzer:", -> { toml { |s| s.add_text_field("t", analyzer: nil) } },
       -> { toml { |s| s.add_text_field("t") } }],
      ["add_hnsw_field quantizer:", -> { toml { |s| s.add_hnsw_field("v", 4, quantizer: nil) } },
       -> { toml { |s| s.add_hnsw_field("v", 4) } }],
      ["add_hnsw_field subvector_count:", -> { toml { |s| s.add_hnsw_field("v", 4, subvector_count: nil) } },
       -> { toml { |s| s.add_hnsw_field("v", 4) } }],
      ["add_hnsw_field rerank_storage:", -> { toml { |s| s.add_hnsw_field("v", 4, rerank_storage: nil) } },
       -> { toml { |s| s.add_hnsw_field("v", 4) } }],
      ["add_hnsw_field embedder:", -> { toml { |s| s.add_hnsw_field("v", 4, embedder: nil) } },
       -> { toml { |s| s.add_hnsw_field("v", 4) } }],
      ["add_hnsw_field pq_codebook_path:", -> { toml { |s| s.add_hnsw_field("v", 4, pq_codebook_path: nil) } },
       -> { toml { |s| s.add_hnsw_field("v", 4) } }],
      # Not listed in the API reference, but read the same way.
      ["add_hnsw_field default_ef_search:", -> { toml { |s| s.add_hnsw_field("v", 4, default_ef_search: nil) } },
       -> { toml { |s| s.add_hnsw_field("v", 4) } }],
      ["add_flat_field embedder:", -> { toml { |s| s.add_flat_field("v", 4, embedder: nil) } },
       -> { toml { |s| s.add_flat_field("v", 4) } }],
      ["add_flat_field quantizer:", -> { toml { |s| s.add_flat_field("v", 4, quantizer: nil) } },
       -> { toml { |s| s.add_flat_field("v", 4) } }],
      ["add_ivf_field embedder:", -> { toml { |s| s.add_ivf_field("v", 4, embedder: nil) } },
       -> { toml { |s| s.add_ivf_field("v", 4) } }],
      ["add_ivf_field quantizer:", -> { toml { |s| s.add_ivf_field("v", 4, quantizer: nil) } },
       -> { toml { |s| s.add_ivf_field("v", 4) } }],
      ["add_multi_vector_field embedder:", -> { toml { |s| s.add_multi_vector_field("v", 4, embedder: nil) } },
       -> { toml { |s| s.add_multi_vector_field("v", 4) } }],
      ["add_analyzer char_filters:",
       -> { toml { |s| s.add_analyzer("a", { type: "whitespace" }, char_filters: nil) } },
       -> { toml { |s| s.add_analyzer("a", { type: "whitespace" }) } }],
      ["add_analyzer token_filters:",
       -> { toml { |s| s.add_analyzer("a", { type: "whitespace" }, token_filters: nil) } },
       -> { toml { |s| s.add_analyzer("a", { type: "whitespace" }) } }],
      ["NumericRangeQuery min:", -> { Laurus::NumericRangeQuery.new("n", min: nil, max: 5).inspect },
       -> { Laurus::NumericRangeQuery.new("n", max: 5).inspect }],
      ["NumericRangeQuery max:", -> { Laurus::NumericRangeQuery.new("n", min: 1, max: nil).inspect },
       -> { Laurus::NumericRangeQuery.new("n", min: 1).inspect }],
      ["DateTimeRangeQuery min:", -> { Laurus::DateTimeRangeQuery.new("d", min: nil, max: "2024-01-01").inspect },
       -> { Laurus::DateTimeRangeQuery.new("d", max: "2024-01-01").inspect }],
      ["DateTimeRangeQuery max:", -> { Laurus::DateTimeRangeQuery.new("d", min: "2024-01-01", max: nil).inspect },
       -> { Laurus::DateTimeRangeQuery.new("d", min: "2024-01-01").inspect }],
      ["Geo3dNearestQuery initial_radius_m:", -> { nearest(initial_radius_m: nil) }, -> { nearest }],
      ["Geo3dNearestQuery max_radius_m:", -> { nearest(max_radius_m: nil) }, -> { nearest }],
      ["LateInteractionRescore window_size:",
       -> { Laurus::LateInteractionRescore.new("tokens", [[1.0, 0.0]], window_size: nil).inspect },
       -> { Laurus::LateInteractionRescore.new("tokens", [[1.0, 0.0]]).inspect }],
      ["SearchRequest.new query:", -> { search_request(query: nil, lexical_query: lexical) },
       -> { search_request(lexical_query: lexical) }],
      ["SearchRequest.new lexical_query:", -> { search_request(query: "title:rust", lexical_query: nil) },
       -> { search_request(query: "title:rust") }],
      ["SearchRequest.new vector_query:", -> { search_request(query: "title:rust", vector_query: nil) },
       -> { search_request(query: "title:rust") }],
      ["SearchRequest.new filter_query:", -> { search_request(query: "title:rust", filter_query: nil) },
       -> { search_request(query: "title:rust") }],
      ["SearchRequest.new fusion:", -> { search_request(lexical_query: lexical, vector_query: vector, fusion: nil) },
       -> { search_request(lexical_query: lexical, vector_query: vector) }],
      ["SearchRequest.new highlight:", -> { search_request(query: "title:rust", highlight: nil) },
       -> { search_request(query: "title:rust") }],
      ["SearchRequest.new rescore:", -> { search_request(query: "title:rust", rescore: nil) },
       -> { search_request(query: "title:rust") }],
      ["SearchRequest.new group_by:", -> { search_request(query: "title:rust", group_by: nil) },
       -> { search_request(query: "title:rust") }]
    ]
  end

  def test_an_explicit_nil_is_the_same_as_leaving_the_keyword_out
    mismatches = cases.filter_map do |name, with_nil, omitted|
      expected = omitted.call
      actual = with_nil.call
      "#{name} #{actual.inspect} != #{expected.inspect}" unless actual == expected
    rescue StandardError => e
      "#{name} #{e.class}: #{e.message[0, 60]}"
    end
    assert_empty mismatches
  end
end
