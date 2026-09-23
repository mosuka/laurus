# frozen_string_literal: true

require_relative "test_helper"

# Basic integration tests for the laurus Ruby binding.
class TestIndex < Minitest::Test
  # ---------------------------------------------------------------------------
  # Setup helpers
  # ---------------------------------------------------------------------------

  def create_index
    idx = Laurus::Index.new
    idx.put_document("doc1", { "title" => "Introduction to Rust", "body" => "Systems programming language." })
    idx.put_document("doc2", { "title" => "Python for Data Science", "body" => "Data analysis with Python." })
    idx.commit
    idx
  end

  def create_vector_index
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_hnsw_field("embedding", 4)
    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "title" => "Rust", "embedding" => [0.1, 0.2, 0.3, 0.4] })
    idx.put_document("doc2", { "title" => "Python", "embedding" => [0.9, 0.8, 0.7, 0.6] })
    idx.commit
    idx
  end

  # ---------------------------------------------------------------------------
  # Index creation
  # ---------------------------------------------------------------------------

  def test_index_memory
    idx = Laurus::Index.new
    assert_equal "Index()", idx.inspect
  end

  def test_index_with_schema
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    idx = Laurus::Index.new(schema: schema)
    refute_nil idx
  end

  # ---------------------------------------------------------------------------
  # WAL sync policy / group commit
  # ---------------------------------------------------------------------------

  def test_wal_sync_policy_constructors
    # Both constructors build accepted value objects.
    per_record = Laurus::WalSyncPolicy.per_record
    refute_nil per_record

    group = Laurus::WalSyncPolicy.group(max_records: 256, max_bytes: 4096, max_interval_ms: 1000)
    refute_nil group

    # group with no args == defaults; both are accepted by Index.new.
    refute_nil Laurus::Index.new(wal_sync_policy: Laurus::WalSyncPolicy.per_record)
    refute_nil Laurus::Index.new(wal_sync_policy: Laurus::WalSyncPolicy.group)
    refute_nil Laurus::Index.new(wal_sync_policy: group)
  end

  def test_index_with_group_commit_flush_and_commit
    idx = Laurus::Index.new(wal_sync_policy: Laurus::WalSyncPolicy.group)
    idx.put_document("doc1", { "title" => "Group commit" })
    # flush_wal forces WAL durability without publishing to searches.
    idx.flush_wal
    idx.commit
    docs = idx.get_documents("doc1")
    assert_equal 1, docs.length
  end

  def test_commit_policy_constructors
    # Both constructors build accepted value objects.
    refute_nil Laurus::CommitPolicy.manual
    refute_nil Laurus::CommitPolicy.every_docs(100)
    # every_docs(0) is valid — it disables auto-commit (equivalent to manual).
    refute_nil Laurus::CommitPolicy.every_docs(0)
    # interval_ms builds the time-based auto-commit policy.
    refute_nil Laurus::CommitPolicy.interval_ms(1000)

    # Every variant, and the default (omitted kwarg), is accepted by Index.new.
    refute_nil Laurus::Index.new(commit_policy: Laurus::CommitPolicy.manual)
    refute_nil Laurus::Index.new(commit_policy: Laurus::CommitPolicy.every_docs(100))
    refute_nil Laurus::Index.new(commit_policy: Laurus::CommitPolicy.every_docs(0))
    refute_nil Laurus::Index.new(commit_policy: Laurus::CommitPolicy.interval_ms(1000))
  end

  def test_index_with_every_docs_auto_commit
    idx = Laurus::Index.new(commit_policy: Laurus::CommitPolicy.every_docs(1))
    idx.put_document("d1", { "title" => "Auto" })
    # No explicit commit — the binding path is wired end-to-end.
    docs = idx.get_documents("d1")
    assert_equal 1, docs.length
  end

  # ---------------------------------------------------------------------------
  # Document CRUD
  # ---------------------------------------------------------------------------

  def test_put_and_get_document
    idx = Laurus::Index.new
    idx.put_document("doc1", { "title" => "Hello" })
    idx.commit
    docs = idx.get_documents("doc1")
    assert_equal 1, docs.length
  end

  def test_put_replaces_existing
    idx = create_index
    idx.put_document("doc1", { "title" => "Updated" })
    idx.commit
    docs = idx.get_documents("doc1")
    assert_equal 1, docs.length
  end

  def test_add_document_appends
    idx = Laurus::Index.new
    idx.add_document("doc1", { "title" => "Chunk 1" })
    idx.add_document("doc1", { "title" => "Chunk 2" })
    docs = idx.get_documents("doc1")
    assert_equal 2, docs.length
  end

  def test_delete_documents
    idx = create_index
    idx.delete_documents("doc1")
    idx.commit
    docs = idx.get_documents("doc1")
    assert_equal 0, docs.length
  end

  def test_get_documents_unknown_id
    idx = create_index
    docs = idx.get_documents("does_not_exist")
    assert_equal [], docs
  end

  # ---------------------------------------------------------------------------
  # Stats
  # ---------------------------------------------------------------------------

  def test_stats
    idx = create_index
    stats = idx.stats
    assert_equal 2, stats["document_count"]
  end

  def test_stats_vector_fields
    idx = create_vector_index
    stats = idx.stats
    assert stats["vector_fields"].key?("embedding")
    assert_equal 2, stats["vector_fields"]["embedding"]["count"]
    assert_equal 4, stats["vector_fields"]["embedding"]["dimension"]
  end

  # ---------------------------------------------------------------------------
  # Lexical search
  # ---------------------------------------------------------------------------

  def test_search_dsl
    idx = create_index
    results = idx.search("title:rust", limit: 5)
    assert results.length >= 1
    assert_equal "doc1", results[0].id
  end

  def test_search_term_query
    idx = create_index
    results = idx.search(Laurus::TermQuery.new("body", "python"), limit: 5)
    assert results.length >= 1
    assert_equal "doc2", results[0].id
  end

  def test_search_result_has_id_and_score
    idx = create_index
    results = idx.search("title:rust", limit: 1)
    r = results[0]
    assert_equal "doc1", r.id
    assert r.score > 0
  end

  def test_search_limit
    idx = create_index
    results = idx.search("body:programming OR body:python", limit: 1)
    assert results.length <= 1
  end

  def test_search_offset
    idx = create_index
    all_results = idx.search("body:programming OR body:data", limit: 10)
    offset_results = idx.search("body:programming OR body:data", limit: 10, offset: 1)
    if all_results.length > 1
      assert_equal all_results[1].id, offset_results[0].id
    end
  end

  def test_search_no_results
    idx = create_index
    results = idx.search("title:nonexistent_xyz", limit: 5)
    assert_equal [], results
  end

  # ---------------------------------------------------------------------------
  # Vector search
  # ---------------------------------------------------------------------------

  def test_vector_query
    idx = create_vector_index
    results = idx.search(Laurus::VectorQuery.new("embedding", [0.1, 0.2, 0.3, 0.4]), limit: 2)
    assert results.length >= 1
    assert_equal "doc1", results[0].id
  end

  def test_vector_query_repr
    q = Laurus::VectorQuery.new("embedding", [0.1, 0.2, 0.3, 0.4])
    assert_includes q.inspect, "embedding"
  end

  # ---------------------------------------------------------------------------
  # Hybrid search
  # ---------------------------------------------------------------------------

  def test_search_request_lexical_only
    idx = create_index
    req = Laurus::SearchRequest.new(lexical_query: Laurus::TermQuery.new("title", "rust"), limit: 5)
    results = idx.search(req)
    assert results.length >= 1
  end

  def test_search_request_hybrid
    idx = create_vector_index
    req = Laurus::SearchRequest.new(
      lexical_query: Laurus::TermQuery.new("title", "rust"),
      vector_query: Laurus::VectorQuery.new("embedding", [0.1, 0.2, 0.3, 0.4]),
      fusion: Laurus::RRF.new(k: 60.0),
      limit: 5
    )
    results = idx.search(req)
    assert results.length >= 1
  end

  # ---------------------------------------------------------------------------
  # Query types
  # ---------------------------------------------------------------------------

  def test_phrase_query
    idx = create_index
    results = idx.search(Laurus::PhraseQuery.new("title", ["introduction", "rust"]), limit: 5)
    assert results.any? { |r| r.id == "doc1" }
  end

  def test_fuzzy_query
    idx = create_index
    results = idx.search(Laurus::FuzzyQuery.new("title", "pythn", max_edits: 1), limit: 5)
    assert results.any? { |r| r.id == "doc2" }
  end

  def test_numeric_range_query
    schema = Laurus::Schema.new
    schema.add_integer_field("year")
    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "year" => 2020 })
    idx.put_document("doc2", { "year" => 2023 })
    idx.commit
    results = idx.search(Laurus::NumericRangeQuery.new("year", min: 2022, max: 2024), limit: 5)
    assert_equal 1, results.length
    assert_equal "doc2", results[0].id
  end

  def datetime_index
    schema = Laurus::Schema.new
    schema.add_datetime_field("created_at")
    idx = Laurus::Index.new(schema: schema)
    idx.put_document("jan", { "created_at" => "2024-01-01T00:00:00Z" })
    idx.put_document("jun", { "created_at" => "2024-06-15T12:00:00Z" })
    idx.put_document("next", { "created_at" => "2025-01-01T00:00:00Z" })
    idx.commit
    idx
  end

  # Issue #1179: date-only bounds are midnight UTC, inclusive on both ends.
  def test_datetime_range_query_with_date_only_bounds
    idx = datetime_index
    q = Laurus::DateTimeRangeQuery.new("created_at", min: "2024-01-01", max: "2024-12-31")
    assert_equal %w[jan jun], idx.search(q, limit: 5).map(&:id).sort
    assert_includes q.inspect, "DateTimeRangeQuery"
  end

  def test_datetime_range_query_accepts_time_objects_and_open_bounds
    idx = datetime_index
    # A Time with an offset is normalized to UTC (21:00+09:00 == 12:00Z).
    q = Laurus::DateTimeRangeQuery.new("created_at", min: Time.new(2024, 6, 15, 21, 0, 0, "+09:00"))
    assert_equal %w[jun next], idx.search(q, limit: 5).map(&:id).sort
    q = Laurus::DateTimeRangeQuery.new("created_at", max: "2024-06-15T12:00:00")
    assert_equal %w[jan jun], idx.search(q, limit: 5).map(&:id).sort
  end

  def test_datetime_range_query_dsl_form_reaches_the_index
    idx = datetime_index
    assert_equal ["jun"], idx.search("created_at:{2024-01-01 TO 2024-12-31}", limit: 5).map(&:id)
  end

  def test_datetime_range_query_rejects_malformed_bound
    assert_raises(ArgumentError) do
      Laurus::DateTimeRangeQuery.new("created_at", min: "yesterday")
    end
    assert_raises(TypeError) do
      Laurus::DateTimeRangeQuery.new("created_at", min: 42)
    end
  end

  def test_boolean_query
    idx = create_index
    q = Laurus::BooleanQuery.new
    q.must(Laurus::TermQuery.new("body", "programming"))
    q.must_not(Laurus::TermQuery.new("title", "python"))
    results = idx.search(q, limit: 5)
    assert results.all? { |r| r.id != "doc2" }
  end

  def test_wildcard_query
    idx = create_index
    results = idx.search(Laurus::WildcardQuery.new("title", "py*"), limit: 5)
    assert results.any? { |r| r.id == "doc2" }
  end

  # ---------------------------------------------------------------------------
  # Fusion algorithms
  # ---------------------------------------------------------------------------

  def test_rrf_repr
    rrf = Laurus::RRF.new(k: 60.0)
    assert_includes rrf.inspect, "60"
  end

  def test_weighted_sum_repr
    ws = Laurus::WeightedSum.new(lexical_weight: 0.3, vector_weight: 0.7)
    assert_includes ws.inspect, "0.3"
  end

  # ---------------------------------------------------------------------------
  # Text analysis
  # ---------------------------------------------------------------------------

  def test_synonym_dictionary
    syn = Laurus::SynonymDictionary.new
    syn.add_synonym_group(["ml", "machine learning"])
    # Should not raise
  end

  def test_whitespace_tokenizer
    tokenizer = Laurus::WhitespaceTokenizer.new
    tokens = tokenizer.tokenize("hello world")
    assert_equal 2, tokens.length
    assert_equal "hello", tokens[0].text
    assert_equal "world", tokens[1].text
  end

  def test_synonym_graph_filter
    syn = Laurus::SynonymDictionary.new
    syn.add_synonym_group(["ml", "machine learning"])
    tokenizer = Laurus::WhitespaceTokenizer.new
    filt = Laurus::SynonymGraphFilter.new(syn, keep_original: true, boost: 0.8)

    tokens = tokenizer.tokenize("ml tutorial")
    result = filt.apply(tokens)
    texts = result.map(&:text)
    assert_includes texts, "ml"
    assert(texts.include?("machine") || texts.include?("machine learning"))
  end

  def test_token_fields
    tokenizer = Laurus::WhitespaceTokenizer.new
    tokens = tokenizer.tokenize("hello")
    tok = tokens[0]
    assert_equal "hello", tok.text
    assert_kind_of Integer, tok.position
    assert_kind_of Integer, tok.position_increment
    assert_kind_of Integer, tok.position_length
    assert_kind_of Float, tok.boost
  end
end
