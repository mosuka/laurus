# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

# Tests for `Schema#add_analyzer` and TOML schema loading/saving (Issue #1062),
# adapted from `laurus-python`'s `test_schema_analyzer.py` (#1058).
#
# These deliberately avoid the Lindera tokenizer: this repository ships no
# Lindera dictionary. `whitespace`/`ngram`/`regex` tokenizers exercise the
# same code paths (`add_analyzer`'s Hash-to-core-enum conversion, and the
# TOML serialize/deserialize round trip) without that dependency.
#
# Each behavioural test proves the analyzer actually reached the query
# engine (a deterministic search-result difference), not just that
# `add_analyzer`/`from_toml` didn't raise.
class TestSchemaAnalyzer < Minitest::Test
  # ---------------------------------------------------------------------
  # add_analyzer: behavioural — the analyzer actually takes effect
  # ---------------------------------------------------------------------

  def test_ngram_tokenizer_enables_substring_match
    schema = Laurus::Schema.new
    schema.add_analyzer("ngram3", { type: "ngram", min_gram: 3, max_gram: 3 })
    schema.add_text_field("title", analyzer: "ngram3")
    schema.add_text_field("plain") # default "standard" analyzer: whole-word tokens

    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "title" => "hello", "plain" => "hello" })
    idx.commit

    # "ell" is a substring of "hello", only reachable via 3-grams (hel/ell/llo).
    assert_equal 1, idx.search("title:ell", limit: 5).length
    assert_equal 0, idx.search("plain:ell", limit: 5).length
  end

  def test_token_filters_apply_lowercase
    schema = Laurus::Schema.new
    schema.add_analyzer("ws", { type: "whitespace" })
    schema.add_analyzer("ws_lower", { type: "whitespace" }, token_filters: [{ type: "lowercase" }])
    schema.add_text_field("raw", analyzer: "ws")
    schema.add_text_field("lower", analyzer: "ws_lower")

    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "raw" => "HELLO World", "lower" => "HELLO World" })
    idx.commit

    assert_equal 0, idx.search("raw:hello", limit: 5).length
    assert_equal 1, idx.search("lower:hello", limit: 5).length
  end

  def test_char_filters_apply_pattern_replace
    schema = Laurus::Schema.new
    schema.add_analyzer("dash", { type: "whitespace" })
    schema.add_analyzer(
      "dash_split",
      { type: "whitespace" },
      char_filters: [{ type: "pattern_replace", pattern: "-", replacement: " " }],
    )
    schema.add_text_field("raw", analyzer: "dash")
    schema.add_text_field("split", analyzer: "dash_split")

    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "raw" => "state-of-the-art", "split" => "state-of-the-art" })
    idx.commit

    assert_equal 0, idx.search("raw:art", limit: 5).length
    assert_equal 1, idx.search("split:art", limit: 5).length
  end

  def test_add_analyzer_defaults_to_no_filters
    schema = Laurus::Schema.new
    schema.add_analyzer("ws", { type: "whitespace" })
    assert_equal ["ws"], schema.analyzer_names
  end

  # ---------------------------------------------------------------------
  # add_analyzer: error surface
  # ---------------------------------------------------------------------

  def test_unknown_tokenizer_type_rejected
    schema = Laurus::Schema.new
    err = assert_raises(ArgumentError) do
      schema.add_analyzer("bad", { type: "kuromoji" })
    end
    assert_match(/tokenizer/, err.message)
  end

  def test_missing_required_tokenizer_field_rejected
    schema = Laurus::Schema.new
    err = assert_raises(ArgumentError) do
      schema.add_analyzer("bad", { type: "ngram", min_gram: 2 }) # missing max_gram
    end
    assert_match(/tokenizer/, err.message)
  end

  def test_unknown_char_filter_type_rejected
    schema = Laurus::Schema.new
    err = assert_raises(ArgumentError) do
      schema.add_analyzer("bad", { type: "whitespace" }, char_filters: [{ type: "unknown_filter" }])
    end
    assert_match(/char_filters\[0\]/, err.message)
  end

  def test_negative_limit_rejected
    schema = Laurus::Schema.new
    err = assert_raises(ArgumentError) do
      schema.add_analyzer("bad", { type: "whitespace" }, token_filters: [{ type: "limit", limit: -1 }])
    end
    assert_match(/token_filters\[0\]/, err.message)
  end

  def test_bool_not_coerced_to_int
    schema = Laurus::Schema.new
    # gaps is bool in the core; must accept a Ruby true/false.
    schema.add_analyzer("r", { type: "regex", pattern: '\w+', gaps: true })
    assert_equal ["r"], schema.analyzer_names
  end

  def test_tokenizer_must_be_hash_not_string
    schema = Laurus::Schema.new
    err = assert_raises(ArgumentError) do
      schema.add_analyzer("bad", "whitespace")
    end
    assert_match(/tokenizer/, err.message)
  end

  def test_string_keyed_hash_also_accepted
    schema = Laurus::Schema.new
    schema.add_analyzer("ngram3", { "type" => "ngram", "min_gram" => 3, "max_gram" => 3 })
    assert_equal ["ngram3"], schema.analyzer_names
  end

  # ---------------------------------------------------------------------
  # from_toml / from_toml_file
  # ---------------------------------------------------------------------

  SCHEMA_TOML = <<~TOML
    default_fields = ["title"]

    [analyzers.ngram3]
    tokenizer = { type = "ngram", min_gram = 3, max_gram = 3 }

    [fields.title.Text]
    indexed = true
    stored = true
    term_vectors = false
    analyzer = "ngram3"
  TOML

  def test_from_toml_then_search
    schema = Laurus::Schema.from_toml(SCHEMA_TOML)
    assert_equal ["title"], schema.field_names
    assert_equal ["ngram3"], schema.analyzer_names

    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "title" => "hello" })
    idx.commit
    assert_equal 1, idx.search("title:ell", limit: 5).length
  end

  def test_from_toml_file_round_trip
    Dir.mktmpdir do |dir|
      path = File.join(dir, "schema.toml")
      File.write(path, SCHEMA_TOML)

      from_path = Laurus::Schema.from_toml_file(path)
      assert_equal ["title"], from_path.field_names
      assert_equal ["ngram3"], from_path.analyzer_names
    end
  end

  def test_from_toml_file_missing_raises_io_error
    Dir.mktmpdir do |dir|
      assert_raises(IOError) do
        Laurus::Schema.from_toml_file(File.join(dir, "does_not_exist.toml"))
      end
    end
  end

  def test_from_toml_parse_error_raises_argument_error
    err = assert_raises(ArgumentError) do
      Laurus::Schema.from_toml("not = [valid")
    end
    assert_match(/TOML/, err.message)
  end

  # ---------------------------------------------------------------------
  # to_toml / to_toml_file: round trip
  # ---------------------------------------------------------------------

  def test_to_toml_then_from_toml_round_trip
    schema = Laurus::Schema.new
    schema.add_analyzer(
      "ngram3",
      { type: "ngram", min_gram: 3, max_gram: 3 },
      char_filters: [{ type: "unicode_normalization", form: "nfkc" }],
      token_filters: [{ type: "lowercase" }],
    )
    schema.add_text_field("title", analyzer: "ngram3")
    schema.set_default_fields(["title"])

    toml_str = schema.to_toml
    restored = Laurus::Schema.from_toml(toml_str)

    assert_equal schema.field_names, restored.field_names
    assert_equal schema.analyzer_names, restored.analyzer_names

    idx = Laurus::Index.new(schema: restored)
    idx.put_document("doc1", { "title" => "hello" })
    idx.commit
    assert_equal 1, idx.search("title:ell", limit: 5).length
  end

  def test_to_toml_file_then_from_toml_file_round_trip
    Dir.mktmpdir do |dir|
      schema = Laurus::Schema.new
      schema.add_analyzer("ws", { type: "whitespace" })
      schema.add_text_field("title", analyzer: "ws")

      path = File.join(dir, "schema.toml")
      schema.to_toml_file(path)
      restored = Laurus::Schema.from_toml_file(path)

      assert_equal schema.field_names, restored.field_names
      assert_equal schema.analyzer_names, restored.analyzer_names
    end
  end
end
