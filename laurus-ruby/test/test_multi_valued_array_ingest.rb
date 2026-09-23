# frozen_string_literal: true

require_relative "test_helper"

# Regression tests for sending Ruby Arrays into multi-valued numeric fields
# (Issue #1178).
#
# `rb_to_data_value` used to turn every Array into `DataValue::Vector`, which
# the core's `coerce_to_integer` / `coerce_to_float` multi-valued branches
# reject — so a `multi_valued: true` Integer/Float field could be *read* from
# Ruby but never *written* from it. Arrays now arrive as `Int64Array` (all
# `Integer`) or `Float64Array` (otherwise numeric) and the schema-aware
# coercion in the core routes them.
class TestMultiValuedArrayIngest < Minitest::Test
  def index_with_integer_field(multi_valued:)
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_integer_field("tags", multi_valued: multi_valued)
    Laurus::Index.new(schema: schema)
  end

  def index_with_float_field
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_float_field("scores", multi_valued: true)
    Laurus::Index.new(schema: schema)
  end

  def test_integer_array_round_trips_through_multi_valued_integer_field
    idx = index_with_integer_field(multi_valued: true)
    # Before #1178 this put_document raised: the Array arrived as a Vector,
    # which a multi-valued integer field does not accept.
    idx.put_document("doc1", { "title" => "t", "tags" => [1, 2, 3] })
    idx.commit

    docs = idx.get_documents("doc1")
    assert_equal 1, docs.length
    assert_equal [1, 2, 3], docs.first["tags"]
  end

  def test_float_array_round_trips_through_multi_valued_float_field
    idx = index_with_float_field
    idx.put_document("doc1", { "title" => "t", "scores" => [1.5, 2.0] })
    idx.commit

    assert_equal [1.5, 2.0], idx.get_documents("doc1").first["scores"]
  end

  # One non-Integer element is enough to make the whole Array a float array;
  # the Integers are widened, not rejected.
  def test_mixed_integer_and_float_array_becomes_float_array
    idx = index_with_float_field
    idx.put_document("doc1", { "title" => "t", "scores" => [1, 2.5] })
    idx.commit

    assert_equal [1.0, 2.5], idx.get_documents("doc1").first["scores"]
  end

  def test_empty_array_is_accepted_by_multi_valued_integer_field
    idx = index_with_integer_field(multi_valued: true)
    idx.put_document("doc1", { "title" => "t", "tags" => [] })
    idx.commit

    assert_equal [], idx.get_documents("doc1").first["tags"]
  end

  # The fix must not loosen the single-valued contract: an Array sent to a
  # field declared without multi_valued is an error, not a truncation.
  def test_array_into_single_valued_integer_field_is_still_rejected
    idx = index_with_integer_field(multi_valued: false)
    assert_raises(StandardError) do
      idx.put_document("doc1", { "title" => "t", "tags" => [2020, 2021] })
    end
  end

  # ---- Multi-valued geo (Issue #1174) ----

  def index_with_geo_field(multi_valued:)
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_geo_field("spots", multi_valued: multi_valued)
    Laurus::Index.new(schema: schema)
  end

  # An Array of { "lat", "lon" } Hashes is a multi-valued geo field and reads
  # back as the same Array of Hashes.
  def test_hash_array_round_trips_through_multi_valued_geo_field
    idx = index_with_geo_field(multi_valued: true)
    spots = [{ "lat" => 35.68, "lon" => 139.76 }, { "lat" => 34.69, "lon" => 135.5 }]
    idx.put_document("doc1", { "title" => "t", "spots" => spots })
    idx.commit

    assert_equal spots, idx.get_documents("doc1").first["spots"]
  end

  def test_hash_array_round_trips_through_multi_valued_geo3d_field
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_geo3d_field("positions", multi_valued: true)
    idx = Laurus::Index.new(schema: schema)
    positions = [{ "x" => 1.0, "y" => 2.0, "z" => 3.0 }, { "x" => -4.0, "y" => 5.0, "z" => -6.0 }]
    idx.put_document("doc1", { "title" => "t", "positions" => positions })
    idx.commit

    assert_equal positions, idx.get_documents("doc1").first["positions"]
  end

  def test_single_hash_is_wrapped_on_multi_valued_geo_field
    idx = index_with_geo_field(multi_valued: true)
    idx.put_document("doc1", { "title" => "t", "spots" => { "lat" => 35.68, "lon" => 139.76 } })
    idx.commit

    assert_equal [{ "lat" => 35.68, "lon" => 139.76 }], idx.get_documents("doc1").first["spots"]
  end

  def test_empty_array_is_accepted_by_multi_valued_geo_field
    idx = index_with_geo_field(multi_valued: true)
    idx.put_document("doc1", { "title" => "t", "spots" => [] })
    idx.commit

    assert_equal [], idx.get_documents("doc1").first["spots"]
  end

  def test_hash_array_into_single_valued_geo_field_is_rejected
    idx = index_with_geo_field(multi_valued: false)
    err = assert_raises(StandardError) do
      idx.put_document("doc1", { "title" => "t", "spots" => [{ "lat" => 35.68, "lon" => 139.76 }] })
    end
    assert_match(/multi_valued/, err.message)
  end

  # ---- Multi-valued datetime (Issue #1184) ----

  def index_with_datetime_field(multi_valued:)
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_datetime_field("seen_at", multi_valued: multi_valued)
    Laurus::Index.new(schema: schema)
  end

  # An Array of RFC 3339 Strings / Time objects is a multi-valued datetime
  # field; it reads back as RFC 3339 Strings (UTC) and any instant matches.
  def test_datetime_array_round_trips_through_multi_valued_datetime_field
    idx = index_with_datetime_field(multi_valued: true)
    idx.put_document(
      "doc1",
      { "title" => "t", "seen_at" => ["2024-01-01T00:00:00Z", Time.new(2024, 6, 15, 21, 0, 0, "+09:00")] }
    )
    idx.put_document("doc2", { "title" => "t", "seen_at" => ["2025-03-01T00:00:00Z"] })
    idx.commit

    assert_equal ["2024-01-01T00:00:00+00:00", "2024-06-15T12:00:00+00:00"],
                 idx.get_documents("doc1").first["seen_at"]
    assert_equal ["doc1"], idx.search("seen_at:[2024-06-01 TO 2024-12-31]", limit: 5).map(&:id)
  end

  def test_single_datetime_is_wrapped_on_multi_valued_datetime_field
    idx = index_with_datetime_field(multi_valued: true)
    idx.put_document("doc1", { "title" => "t", "seen_at" => "2024-01-01T00:00:00Z" })
    idx.commit

    assert_equal ["2024-01-01T00:00:00+00:00"], idx.get_documents("doc1").first["seen_at"]
  end

  def test_empty_array_is_accepted_by_multi_valued_datetime_field
    idx = index_with_datetime_field(multi_valued: true)
    idx.put_document("doc1", { "title" => "t", "seen_at" => [] })
    idx.commit

    assert_equal [], idx.get_documents("doc1").first["seen_at"]
  end

  def test_datetime_array_into_single_valued_datetime_field_is_rejected
    idx = index_with_datetime_field(multi_valued: false)
    err = assert_raises(StandardError) do
      idx.put_document("doc1", { "title" => "t", "seen_at" => ["2024-01-01T00:00:00Z"] })
    end
    assert_match(/multi_valued/, err.message)
  end

  def test_non_datetime_string_array_is_rejected
    idx = index_with_datetime_field(multi_valued: true)
    err = assert_raises(ArgumentError) do
      idx.put_document("doc1", { "title" => "t", "seen_at" => ["2024-01-01T00:00:00Z", "tomorrow"] })
    end
    assert_match(/datetimes/, err.message)
  end

  # ---- Multi-valued boolean (Issue #1180) ----

  def index_with_boolean_field(multi_valued:)
    schema = Laurus::Schema.new
    schema.add_text_field("title")
    schema.add_boolean_field("flags", multi_valued: multi_valued)
    Laurus::Index.new(schema: schema)
  end

  # An Array of true / false is a multi-valued boolean field; it reads back
  # as the same Array and a term query matches if any element carries the
  # value.
  def test_bool_array_round_trips_through_multi_valued_boolean_field
    idx = index_with_boolean_field(multi_valued: true)
    idx.put_document("doc1", { "title" => "t", "flags" => [true, false] })
    idx.put_document("doc2", { "title" => "t", "flags" => [false] })
    idx.commit

    assert_equal [true, false], idx.get_documents("doc1").first["flags"]
    assert_equal ["doc1"], idx.search("flags:true", limit: 5).map(&:id)
    assert_equal %w[doc1 doc2], idx.search("flags:false", limit: 5).map(&:id).sort
  end

  def test_single_bool_is_wrapped_on_multi_valued_boolean_field
    idx = index_with_boolean_field(multi_valued: true)
    idx.put_document("doc1", { "title" => "t", "flags" => true })
    idx.commit

    assert_equal [true], idx.get_documents("doc1").first["flags"]
  end

  def test_empty_array_is_accepted_by_multi_valued_boolean_field
    idx = index_with_boolean_field(multi_valued: true)
    idx.put_document("doc1", { "title" => "t", "flags" => [] })
    idx.commit

    assert_equal [], idx.get_documents("doc1").first["flags"]
  end

  def test_bool_array_into_single_valued_boolean_field_is_rejected
    idx = index_with_boolean_field(multi_valued: false)
    err = assert_raises(StandardError) do
      idx.put_document("doc1", { "title" => "t", "flags" => [true] })
    end
    assert_match(/multi_valued/, err.message)
  end

  # `[true, 1]` is neither all-bool nor all-Integer, so it takes the Float
  # path, where `true` has no Float conversion.
  def test_mixed_bool_and_integer_array_is_rejected
    idx = index_with_boolean_field(multi_valued: true)
    assert_raises(TypeError) do
      idx.put_document("doc1", { "title" => "t", "flags" => [true, 1] })
    end
  end

  def test_mixed_geo_dimension_array_is_rejected
    idx = index_with_geo_field(multi_valued: true)
    assert_raises(StandardError) do
      idx.put_document(
        "doc1",
        { "title" => "t", "spots" => [{ "lat" => 35.68, "lon" => 139.76 }, { "x" => 1.0, "y" => 2.0, "z" => 3.0 }] }
      )
    end
  end
end
