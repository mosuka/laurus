# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

# Tests for the on-disk directory layout of a file-backed `Index` (Issue #1059).
#
# Before this change, `Laurus::Index.new(path: X)` wrote segment files
# directly under `X`, incompatible with `laurus-cli`'s `<X>/schema.toml` +
# `<X>/store/` convention. These tests verify the new shared layout: schema
# auto-persistence, auto-loading on reopen, the reopen-with-schema conflict
# error, and legacy-layout detection.
class TestIndexDirLayout < Minitest::Test
  def test_creating_a_file_backed_index_writes_schema_toml_and_store
    Dir.mktmpdir do |dir|
      schema = Laurus::Schema.new
      schema.add_text_field("title")

      Laurus::Index.new(path: dir, schema: schema)

      assert File.file?(File.join(dir, "schema.toml"))
      assert File.directory?(File.join(dir, "store"))
      # No stray top-level segment directories from the old flat layout.
      refute File.exist?(File.join(dir, "lexical"))
    end
  end

  def test_reopen_without_schema_loads_persisted_schema_and_data
    Dir.mktmpdir do |dir|
      schema = Laurus::Schema.new
      schema.add_text_field("title")
      schema.set_default_fields(["title"])

      idx = Laurus::Index.new(path: dir, schema: schema)
      idx.put_document("doc1", { "title" => "hello world" })
      idx.commit
      idx.close

      reopened = Laurus::Index.new(path: dir)
      results = reopened.search("title:hello", limit: 5)
      assert_equal 1, results.length
    end
  end

  def test_reopen_with_explicit_schema_raises
    Dir.mktmpdir do |dir|
      schema = Laurus::Schema.new
      schema.add_text_field("title")
      Laurus::Index.new(path: dir, schema: schema)

      err = assert_raises(ArgumentError) do
        Laurus::Index.new(path: dir, schema: schema)
      end
      assert_match(/schema\.toml/, err.message)
    end
  end

  def test_reopen_with_no_schema_at_all_succeeds_on_empty_default
    Dir.mktmpdir do |dir|
      # First call with no schema creates an empty-schema index (unchanged
      # default behavior); reopening it (also with no schema) must not raise.
      idx = Laurus::Index.new(path: dir)
      idx.close
      Laurus::Index.new(path: dir)
    end
  end

  def test_legacy_flat_layout_is_rejected
    Dir.mktmpdir do |dir|
      # Simulate a directory written by a laurus-ruby version predating
      # Issue #1059: segment files directly under the path, no schema.toml.
      File.write(File.join(dir, "engine.wal"), "")

      err = assert_raises(ArgumentError) do
        Laurus::Index.new(path: dir)
      end
      assert_match(/pre-Issue-1059/, err.message)
    end
  end

  def test_new_empty_directory_is_not_treated_as_legacy
    Dir.mktmpdir do |dir|
      # A directory that merely exists but has no laurus files at all is a
      # normal fresh-create case, not a legacy-layout error.
      Laurus::Index.new(path: dir)
      assert File.file?(File.join(dir, "schema.toml"))
    end
  end

  # ---------------------------------------------------------------------
  # Engine build failure must leave nothing behind (Issue #1308)
  # ---------------------------------------------------------------------

  # A schema that persists fine (schema.toml validation never sees the bad
  # regex) but fails once the engine actually resolves the analyzer.
  MALFORMED_REGEX_SCHEMA_TOML = <<~TOML
    [analyzers.bad]
    tokenizer = { type = "regex", pattern = "(" }

    [fields.body.Text]
    analyzer = "bad"
  TOML

  def test_build_failure_leaves_nothing_and_retry_succeeds
    Dir.mktmpdir do |dir|
      index_dir = File.join(dir, "idx")
      bad_schema = Laurus::Schema.from_toml(MALFORMED_REGEX_SCHEMA_TOML)

      # A bad regex is only caught once the engine resolves the analyzer at
      # build time -- it surfaces as `LaurusError::Analysis`, which
      # `laurus_err` maps to `RuntimeError` (not `ArgumentError`).
      err = assert_raises(RuntimeError) do
        Laurus::Index.new(path: index_dir, schema: bad_schema)
      end
      assert_match(/Failed to resolve analyzer for field 'body'/, err.message)
      refute File.exist?(File.join(index_dir, "schema.toml")),
             "a build failure in a directory this call created must leave nothing behind"
      refute File.exist?(index_dir),
             "a build failure in a directory this call created must leave nothing behind"

      # Retry with a fixed schema at the same path must succeed without
      # deleting anything by hand.
      schema = Laurus::Schema.new
      schema.add_text_field("body")
      idx = Laurus::Index.new(path: index_dir, schema: schema)
      idx.put_document("doc1", { "body" => "hello world" })
      idx.commit
      assert_equal 1, idx.search("body:hello", limit: 5).length
    end
  end

  def test_build_failure_in_a_pre_existing_directory_only_removes_schema_and_store
    Dir.mktmpdir do |dir|
      # An unrelated file already in the directory before the failed attempt.
      File.write(File.join(dir, "README.txt"), "keep me")

      bad_schema = Laurus::Schema.from_toml(MALFORMED_REGEX_SCHEMA_TOML)
      err = assert_raises(RuntimeError) do
        Laurus::Index.new(path: dir, schema: bad_schema)
      end
      assert_match(/Failed to resolve analyzer for field 'body'/, err.message)

      assert File.file?(File.join(dir, "README.txt")),
             "unrelated pre-existing files must survive"
      refute File.exist?(File.join(dir, "schema.toml"))
      refute File.exist?(File.join(dir, "store"))

      schema = Laurus::Schema.new
      schema.add_text_field("body")
      idx = Laurus::Index.new(path: dir, schema: schema)
      idx.put_document("doc1", { "body" => "hello world" })
      idx.commit
      assert_equal 1, idx.search("body:hello", limit: 5).length
      assert File.file?(File.join(dir, "README.txt")), "unrelated file must still survive"
    end
  end
end
