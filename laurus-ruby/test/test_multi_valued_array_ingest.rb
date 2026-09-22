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
