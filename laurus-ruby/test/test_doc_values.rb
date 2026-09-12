# frozen_string_literal: true

require_relative "test_helper"

# Integration tests for the per-field `doc_values` schema option (Issue
# #1047).
#
# The Ruby binding does not currently expose field-sorted search or
# faceting, so these tests cannot observe the DocValues column itself
# disappearing on disk. What they cover is the acceptance criterion that
# matters at this layer: `doc_values: false` must not lose stored-field
# retrieval, and the schema builder must accept the flag on every field
# type that carries it without raising.
class TestDocValues < Minitest::Test
  def test_doc_values_false_keeps_stored_field_retrievable
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_text_field("internal_note", doc_values: false)
    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "title" => "Hello", "internal_note" => "opted out of doc values" })
    idx.commit

    docs = idx.get_documents("doc1")
    assert_equal 1, docs.length
    assert_equal "Hello", docs.first["title"]
    assert_equal "opted out of doc values", docs.first["internal_note"]
  end

  # `doc_values` only controls the column-oriented store; it must not
  # affect whether the field is indexed for search.
  def test_doc_values_false_field_is_still_searchable
    schema = Laurus::Schema.new
    schema.add_text_field("title", doc_values: false)
    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "title" => "Rust programming" })
    idx.commit

    results = idx.search(Laurus::TermQuery.new("title", "rust"), limit: 5)
    assert results.any? { |r| r.id == "doc1" }
  end

  # Every field option except Bytes carries `doc_values`; this must not
  # raise for any of them, in either direction.
  def test_add_field_methods_accept_doc_values_on_every_carrying_type
    schema = Laurus::Schema.new
    schema.add_text_field("t", doc_values: false)
    schema.add_integer_field("i", doc_values: false)
    schema.add_float_field("f", doc_values: false)
    schema.add_boolean_field("b", doc_values: false)
    schema.add_datetime_field("d", doc_values: false)
    schema.add_geo_field("g", doc_values: false)
    schema.add_geo3d_field("g3", doc_values: false)
    # And the default (true) must still be accepted explicitly too.
    schema.add_text_field("t2", doc_values: true)

    idx = Laurus::Index.new(schema: schema)
    refute_nil idx
  end
end
