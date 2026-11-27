# frozen_string_literal: true

require "set"
require_relative "test_helper"

# Integration tests for Geo3d (3D ECEF) APIs.
#
# Covers:
# - Schema declaration via `add_geo3d_field`.
# - Document round-trip with `{ "x", "y", "z" }` Hash values.
# - All three 3D query classes: `Geo3dDistanceQuery`, `Geo3dBoundingBoxQuery`,
#   `Geo3dNearestQuery`.
class TestGeo3d < Minitest::Test
  # Precomputed ECEF coordinates (meters) for landmarks.
  TOKYO_TOWER   = { "x" => -3_955_182.0, "y" =>  3_350_553.0, "z" =>  3_700_276.0 }.freeze
  TOKYO_SKYTREE = { "x" => -3_961_178.0, "y" =>  3_346_187.0, "z" =>  3_702_490.0 }.freeze
  MT_FUJI       = { "x" => -3_916_073.0, "y" =>  3_437_037.0, "z" =>  3_672_751.0 }.freeze
  SYDNEY        = { "x" => -4_646_847.0, "y" =>  2_553_022.0, "z" => -3_534_121.0 }.freeze

  def make_index
    schema = Laurus::Schema.new
    schema.add_text_field("name")
    schema.add_geo3d_field("position")
    idx = Laurus::Index.new(schema: schema)
    idx.put_document("tokyo_tower",   { "name" => "Tokyo Tower",        "position" => TOKYO_TOWER })
    idx.put_document("tokyo_skytree", { "name" => "Tokyo Skytree",      "position" => TOKYO_SKYTREE })
    idx.put_document("mt_fuji",       { "name" => "Mt. Fuji summit",    "position" => MT_FUJI })
    idx.put_document("sydney",        { "name" => "Sydney Opera House", "position" => SYDNEY })
    idx.commit
    idx
  end

  # ---------------------------------------------------------------------------
  # Schema and document round-trip
  # ---------------------------------------------------------------------------

  def test_geo3d_field_round_trip
    idx = make_index
    docs = idx.get_documents("tokyo_tower")
    assert_equal 1, docs.length
    assert_equal "Tokyo Tower", docs.first["name"]
    assert_equal TOKYO_TOWER, docs.first["position"]
  end

  # ---------------------------------------------------------------------------
  # Geo3dDistanceQuery
  # ---------------------------------------------------------------------------

  def test_geo3d_distance_query_small_radius
    idx = make_index
    q = Laurus::Geo3dDistanceQuery.within_sphere(
      "position", TOKYO_TOWER["x"], TOKYO_TOWER["y"], TOKYO_TOWER["z"], 50_000.0,
    )
    results = idx.search(q, limit: 10).results
    ids = results.map(&:id).to_set
    assert_equal Set["tokyo_tower", "tokyo_skytree"], ids
  end

  def test_geo3d_distance_query_wide_radius
    idx = make_index
    q = Laurus::Geo3dDistanceQuery.within_sphere(
      "position", TOKYO_TOWER["x"], TOKYO_TOWER["y"], TOKYO_TOWER["z"], 200_000.0,
    )
    results = idx.search(q, limit: 10).results
    ids = results.map(&:id).to_set
    assert_equal Set["tokyo_tower", "tokyo_skytree", "mt_fuji"], ids
  end

  # ---------------------------------------------------------------------------
  # Geo3dBoundingBoxQuery
  # ---------------------------------------------------------------------------

  def test_geo3d_bounding_box_query
    # Central-Tokyo box: must include Tokyo Tower (x ≈ -3.955M) and Tokyo
    # Skytree (x ≈ -3.961M) but exclude Mt. Fuji (x ≈ -3.916M, above the
    # upper bound) and Sydney (x ≈ -4.65M, well below).
    idx = make_index
    q = Laurus::Geo3dBoundingBoxQuery.within_box(
      "position",
      -3_962_000.0,  3_340_000.0,  3_690_000.0,
      -3_954_000.0,  3_360_000.0,  3_710_000.0,
    )
    results = idx.search(q, limit: 10).results
    ids = results.map(&:id).to_set
    assert_equal Set["tokyo_tower", "tokyo_skytree"], ids
  end

  # ---------------------------------------------------------------------------
  # Geo3dNearestQuery
  # ---------------------------------------------------------------------------

  def test_geo3d_nearest_query
    idx = make_index
    q = Laurus::Geo3dNearestQuery.k_nearest(
      "position", MT_FUJI["x"], MT_FUJI["y"], MT_FUJI["z"], 3,
    )
    results = idx.search(q, limit: 3).results
    assert_equal 3, results.length
    ids = results.map(&:id).to_set
    assert_equal Set["mt_fuji", "tokyo_tower", "tokyo_skytree"], ids
    # Mt. Fuji must be the closest hit.
    assert_equal "mt_fuji", results.first.id
  end

  def test_geo3d_nearest_query_with_radius_options
    # Verify the optional initial / max radius kwargs are accepted.
    idx = make_index
    q = Laurus::Geo3dNearestQuery.k_nearest(
      "position", TOKYO_TOWER["x"], TOKYO_TOWER["y"], TOKYO_TOWER["z"], 2,
      initial_radius_m: 10_000.0,
      max_radius_m: 10_000_000.0,
    )
    results = idx.search(q, limit: 2).results
    ids = results.map(&:id).to_set
    assert_equal Set["tokyo_tower", "tokyo_skytree"], ids
  end

  # ---------------------------------------------------------------------------
  # Factory smoke checks
  # ---------------------------------------------------------------------------

  def test_geo3d_query_factories_create_instances
    refute_nil Laurus::Geo3dDistanceQuery.within_sphere(
      "position", TOKYO_TOWER["x"], TOKYO_TOWER["y"], TOKYO_TOWER["z"], 1000.0,
    )
    refute_nil Laurus::Geo3dBoundingBoxQuery.within_box(
      "position", 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    )
    refute_nil Laurus::Geo3dNearestQuery.k_nearest(
      "position", TOKYO_TOWER["x"], TOKYO_TOWER["y"], TOKYO_TOWER["z"], 5,
    )
  end

  def test_geo3d_query_inspect_includes_class_name
    q1 = Laurus::Geo3dDistanceQuery.within_sphere("position", 1.0, 2.0, 3.0, 1000.0)
    assert_includes q1.inspect, "Geo3dDistanceQuery"

    q2 = Laurus::Geo3dBoundingBoxQuery.within_box("position", 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert_includes q2.inspect, "Geo3dBoundingBoxQuery"

    q3 = Laurus::Geo3dNearestQuery.k_nearest("position", 1.0, 2.0, 3.0, 5)
    assert_includes q3.inspect, "Geo3dNearestQuery"
  end
end
