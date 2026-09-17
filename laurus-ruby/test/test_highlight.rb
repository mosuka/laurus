# frozen_string_literal: true

require_relative "test_helper"

# Integration tests for search-result highlighting (Issue #1134).
#
# Covers the Ruby-facing `highlight:` keyword on `Index#search` /
# `Index#search_batch`, and the equivalent `highlight:` keyword on
# `SearchRequest.new`:
#
# - The Array shorthand (`["body"]`) and the Hash form
#   (`{fields: [...], ...}`) are both accepted.
# - `SearchResult#highlights` is empty when `highlight:` is not passed.
# - `HighlightConfig` knobs (`tag:`, `max_fragments:`, ...) take effect.
# - `search_batch` applies the same highlight settings to every query.
# - `SearchRequest.new(highlight: ...)` matches the equivalent
#   `search(..., highlight: ...)` call.
class TestHighlight < Minitest::Test
  def create_index
    idx = Laurus::Index.new
    idx.put_document("doc1", { "title" => "Introduction to Rust", "body" => "Rust is a systems programming language." })
    idx.commit
    idx
  end

  def test_highlight_array_shorthand_returns_fragments
    idx = create_index
    results = idx.search("body:rust", highlight: ["body"])

    assert_equal 1, results.length
    fragments = results[0].highlights["body"]
    assert_equal 1, fragments.length
    assert_includes fragments[0], "<mark>Rust</mark>"
  end

  def test_highlight_omitted_leaves_highlights_empty
    idx = create_index
    results = idx.search("body:rust")

    assert_equal 1, results.length
    assert_equal({}, results[0].highlights)
  end

  def test_highlight_unrequested_field_is_absent
    idx = create_index
    results = idx.search("body:rust", highlight: ["body"])

    assert_nil results[0].highlights["title"]
  end

  def test_highlight_hash_form_applies_config
    idx = create_index
    results = idx.search("body:rust", highlight: { fields: ["body"], tag: "em", max_fragments: 2 })

    fragment = results[0].highlights["body"][0]
    assert_includes fragment, "<em>Rust</em>"
    refute_includes fragment, "<mark>"
  end

  def test_highlight_hash_accepts_string_keys
    idx = create_index
    results = idx.search("body:rust", highlight: { "fields" => ["body"], "tag" => "em" })

    assert_includes results[0].highlights["body"][0], "<em>Rust</em>"
  end

  def test_highlight_hash_requires_fields_key
    idx = create_index
    assert_raises(ArgumentError) do
      idx.search("body:rust", highlight: { max_fragments: 2 })
    end
  end

  def test_highlight_invalid_type_raises
    idx = create_index
    assert_raises(ArgumentError) do
      idx.search("body:rust", highlight: 42)
    end
  end

  def test_search_batch_applies_highlight_to_every_query
    idx = create_index
    # Both queries target "body" so the requested "body" highlight applies
    # to both under the default require_field_match: true.
    batch = idx.search_batch(["body:rust", "body:programming"], highlight: ["body"])

    assert_equal 2, batch.length
    batch.each do |results|
      assert_equal 1, results.length
      assert_includes results[0].highlights["body"][0], "<mark>"
    end
  end

  def test_search_request_highlight_matches_search_kwarg
    idx = create_index
    via_kwarg = idx.search("body:rust", highlight: ["body"])
    request = Laurus::SearchRequest.new(query: "body:rust", highlight: ["body"])
    via_request = idx.search(request)

    assert_equal via_kwarg[0].highlights, via_request[0].highlights
  end
end
