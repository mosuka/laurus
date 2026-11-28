# frozen_string_literal: true

require "set"
require_relative "test_helper"

# Integration tests for the polygon and polyline queries (Issue #1596).
#
# Covers `GeoPolygonQuery`, `GeoPolylineQuery` and `Geo3dPolylineQuery`:
# the hits and their order, the same hits from the DSL, the queries inside
# a `BooleanQuery` and a `SearchRequest` filter, and the shapes rejected
# when the query is built.
#
# The points are placed so that reading a vertex with its latitude and
# longitude swapped gives another valid shape with other hits: "mirror" is
# "inside" with its coordinates swapped.
class TestGeoShapes < Minitest::Test
  def self.latlon(lat, lon)
    { "lat" => lat, "lon" => lon }.freeze
  end

  def self.xyz(x, y, z)
    { "x" => x, "y" => y, "z" => z }.freeze
  end

  POINTS = {
    # Around the polygon: latitudes 10 to 20, longitudes 30 to 40, with a
    # hole over latitudes 14 to 16, longitudes 34 to 36.
    "inside" => latlon(12.0, 32.0),
    "inside_east" => latlon(18.0, 38.0),
    "in_hole" => latlon(15.0, 35.0),
    "outside" => latlon(25.0, 35.0),
    "mirror" => latlon(32.0, 12.0),
    # Around the polyline along the equator from longitude 0 to 10.
    "on_line" => latlon(0.0, 3.0),
    "near_line" => latlon(1.0, 5.0),
    "past_end" => latlon(0.0, 11.5),
    "far" => latlon(5.0, 5.0)
  }.freeze

  POSITIONS = {
    # Around the 3D polyline (0, 0, 0) -> (1000, 0, 0) -> (1000, 1000, 0).
    "p_on" => xyz(500.0, 0.0, 0.0),
    "p_near" => xyz(1000.0, 500.0, 50.0),
    "p_end" => xyz(1000.0, 1060.0, 0.0),
    "p_far" => xyz(500.0, 500.0, 0.0)
  }.freeze

  EXTERIOR = [latlon(10.0, 30.0), latlon(10.0, 40.0), latlon(20.0, 40.0), latlon(20.0, 30.0)].freeze
  HOLE = [latlon(14.0, 34.0), latlon(14.0, 36.0), latlon(16.0, 36.0), latlon(16.0, 34.0)].freeze
  LINE = [latlon(0.0, 0.0), latlon(0.0, 10.0)].freeze
  LINE_3D = [xyz(0.0, 0.0, 0.0), xyz(1000.0, 0.0, 0.0), xyz(1000.0, 1000.0, 0.0)].freeze

  CAFES = (POINTS.keys + POSITIONS.keys - ["inside_east"]).to_set.freeze

  def setup
    schema = Laurus::Schema.new
    schema.add_text_field("kind")
    schema.add_geo_field("location")
    schema.add_geo3d_field("position")
    @index = Laurus::Index.new(schema: schema)
    POINTS.each do |id, point|
      kind = id == "inside_east" ? "bar" : "cafe"
      @index.put_document(id, { "kind" => kind, "location" => point })
    end
    POSITIONS.each do |id, position|
      @index.put_document(id, { "kind" => "cafe", "position" => position })
    end
    @index.commit
  end

  def polygon
    Laurus::GeoPolygonQuery.within_polygon("location", EXTERIOR, holes: [HOLE])
  end

  def polyline
    Laurus::GeoPolylineQuery.within_distance("location", LINE, 200_000.0)
  end

  def polyline3d
    Laurus::Geo3dPolylineQuery.within_distance("position", LINE_3D, 100.0)
  end

  def results(query)
    @index.search(query, limit: 20).results
  end

  def ids(query)
    results(query).map(&:id).to_set
  end

  def ranked(query)
    results(query).map(&:id)
  end

  def queries
    [
      [polygon, Set["inside", "inside_east"]],
      [polyline, Set["on_line", "near_line", "past_end"]],
      [polyline3d, Set["p_on", "p_near", "p_end"]]
    ]
  end

  # ---------------------------------------------------------------------------
  # The hits
  # ---------------------------------------------------------------------------

  def test_a_polygon_with_a_hole
    assert_equal Set["inside", "inside_east"], ids(polygon)
  end

  def test_a_polygon_without_holes_takes_the_hole
    query = Laurus::GeoPolygonQuery.within_polygon("location", EXTERIOR)
    assert_equal Set["inside", "inside_east", "in_hole"], ids(query)
  end

  def test_a_polyline_ranks_the_closest_first
    # 0 m, about 111 km and about 167 km of 200 km.
    hits = results(polyline)
    assert_equal %w[on_line near_line past_end], hits.map(&:id)
    assert_in_delta 1.0, hits.first.score, 1e-6
  end

  def test_a_geo3d_polyline_ranks_the_closest_first
    # 0 m, 50 m and 60 m of 100 m.
    hits = results(polyline3d)
    assert_equal %w[p_on p_near p_end], hits.map(&:id)
    [1.0, 0.5, 0.4].zip(hits.map(&:score)).each do |expected, actual|
      assert_in_delta expected, actual, 1e-6
    end
  end

  def test_the_dsl_gives_the_same_hits
    assert_equal ids(polygon), ids(
      "location:geo_polygon((10, 30, 10, 40, 20, 40, 20, 30), (14, 34, 14, 36, 16, 36, 16, 34))"
    )
    assert_equal ranked(polyline), ranked("location:geo_polyline((0, 0, 0, 10), 200000)")
    assert_equal ranked(polyline3d),
                 ranked("position:geo3d_polyline((0, 0, 0, 1000, 0, 0, 1000, 1000, 0), 100)")
  end

  # ---------------------------------------------------------------------------
  # In combination
  # ---------------------------------------------------------------------------

  def test_the_queries_in_a_boolean_query
    queries.each do |query, hits|
      must = Laurus::BooleanQuery.new
      must.must(query)
      must.must(Laurus::TermQuery.new("kind", "cafe"))
      assert_equal hits & CAFES, ids(must), query.inspect

      must_not = Laurus::BooleanQuery.new
      must_not.must(Laurus::TermQuery.new("kind", "cafe"))
      must_not.must_not(query)
      assert_equal CAFES - hits, ids(must_not), query.inspect
    end
  end

  def test_the_queries_as_a_filter
    queries.each do |query, hits|
      request = Laurus::SearchRequest.new(
        lexical_query: Laurus::TermQuery.new("kind", "cafe"),
        filter_query: query,
        limit: 20
      )
      assert_equal hits & CAFES, @index.search(request).results.map(&:id).to_set, query.inspect
    end
  end

  # ---------------------------------------------------------------------------
  # Rejected when built
  # ---------------------------------------------------------------------------

  def invalid_shapes
    [
      ["an exterior of 2 points", ArgumentError,
       -> { Laurus::GeoPolygonQuery.within_polygon("location", [self.class.latlon(0.0, 0.0), self.class.latlon(0.0, 1.0)]) }],
      ["a hole of 2 points", ArgumentError,
       lambda {
         Laurus::GeoPolygonQuery.within_polygon(
           "location", EXTERIOR, holes: [[self.class.latlon(15.0, 35.0), self.class.latlon(15.0, 36.0)]]
         )
       }],
      ["a latitude out of range", ArgumentError,
       lambda {
         Laurus::GeoPolygonQuery.within_polygon(
           "location", [self.class.latlon(95.0, 0.0), self.class.latlon(0.0, 1.0), self.class.latlon(1.0, 1.0)]
         )
       }],
      ["an edge across exactly 180 degrees", ArgumentError,
       lambda {
         Laurus::GeoPolygonQuery.within_polygon(
           "location", [self.class.latlon(0.0, -90.0), self.class.latlon(0.0, 90.0), self.class.latlon(10.0, 90.0)]
         )
       }],
      ["a polyline edge between antipodes", ArgumentError,
       lambda {
         Laurus::GeoPolylineQuery.within_distance(
           "location", [self.class.latlon(0.0, 0.0), self.class.latlon(0.0, 180.0)], 1000.0
         )
       }],
      ["a polyline without vertices", ArgumentError,
       -> { Laurus::GeoPolylineQuery.within_distance("location", [], 1000.0) }],
      ["a NaN coordinate in 3D", ArgumentError,
       lambda {
         Laurus::Geo3dPolylineQuery.within_distance(
           "position", [self.class.xyz(0.0, 0.0, 0.0), self.class.xyz(Float::NAN, 0.0, 0.0)], 100.0
         )
       }],
      ["a 3D polyline without vertices", ArgumentError,
       -> { Laurus::Geo3dPolylineQuery.within_distance("position", [], 100.0) }],
      ["a vertex without its longitude", ArgumentError,
       -> { Laurus::GeoPolylineQuery.within_distance("location", [{ "lat" => 1.0 }], 1000.0) }],
      ["a vertex without its z", ArgumentError,
       -> { Laurus::Geo3dPolylineQuery.within_distance("position", [{ "x" => 1.0, "y" => 2.0 }], 100.0) }],
      ["a vertex that is not a Hash", TypeError,
       -> { Laurus::GeoPolylineQuery.within_distance("location", [[1.0, 2.0]], 1000.0) }],
      ["a latitude that is not a number", TypeError,
       -> { Laurus::GeoPolylineQuery.within_distance("location", [{ "lat" => "north", "lon" => 1.0 }], 1000.0) }]
    ]
  end

  def test_an_invalid_shape_is_rejected_when_built
    wrong = invalid_shapes.filter_map do |label, error, build|
      build.call
      "#{label}: accepted"
    rescue error
      nil
    rescue StandardError => e
      "#{label}: #{e.class}: #{e.message[0, 80]}"
    end
    assert_empty wrong
  end

  def test_the_inspects
    assert_equal "GeoPolygonQuery.within_polygon(field='location', exterior=4 vertices, holes=1)",
                 polygon.inspect
    assert_equal "GeoPolylineQuery.within_distance(field='location', vertices=2, distance_m=200000)",
                 polyline.inspect
    assert_equal "Geo3dPolylineQuery.within_distance(field='position', vertices=3, distance_m=100)",
                 polyline3d.inspect
    assert_equal polygon.inspect, polygon.to_s
  end
end
