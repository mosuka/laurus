"""Integration tests for the polygon and polyline queries (Issue #1596).

Covers `GeoPolygonQuery`, `GeoPolylineQuery` and `Geo3dPolylineQuery`:
the hits and their order, the same hits from the DSL, the queries inside a
`BooleanQuery` and a `SearchRequest` filter, and the shapes rejected when
the query is built. A vertex is a dict, as in every binding; a tuple is
rejected.

The points are placed so that reading a vertex with its latitude and
longitude swapped gives another valid shape with other hits: "mirror" is
"inside" with its coordinates swapped.
"""

import math

import pytest
import laurus


POINTS = {
    # Around the polygon: latitudes 10 to 20, longitudes 30 to 40, with a
    # hole over latitudes 14 to 16, longitudes 34 to 36.
    "inside": (12.0, 32.0),
    "inside_east": (18.0, 38.0),
    "in_hole": (15.0, 35.0),
    "outside": (25.0, 35.0),
    "mirror": (32.0, 12.0),
    # Around the polyline along the equator from longitude 0 to 10.
    "on_line": (0.0, 3.0),
    "near_line": (1.0, 5.0),
    "past_end": (0.0, 11.5),
    "far": (5.0, 5.0),
}

POSITIONS = {
    # Around the 3D polyline (0, 0, 0) -> (1000, 0, 0) -> (1000, 1000, 0).
    "p_on": (500.0, 0.0, 0.0),
    "p_near": (1000.0, 500.0, 50.0),
    "p_end": (1000.0, 1060.0, 0.0),
    "p_far": (500.0, 500.0, 0.0),
}

EXTERIOR = [(10.0, 30.0), (10.0, 40.0), (20.0, 40.0), (20.0, 30.0)]
HOLE = [(14.0, 34.0), (14.0, 36.0), (16.0, 36.0), (16.0, 34.0)]
LINE = [(0.0, 0.0), (0.0, 10.0)]
LINE_3D = [(0.0, 0.0, 0.0), (1000.0, 0.0, 0.0), (1000.0, 1000.0, 0.0)]


def latlon(vertices):
    return [{"lat": lat, "lon": lon} for lat, lon in vertices]


def xyz(vertices):
    return [{"x": x, "y": y, "z": z} for x, y, z in vertices]


@pytest.fixture
def index():
    schema = laurus.Schema()
    schema.add_text_field("kind")
    schema.add_geo_field("location")
    schema.add_geo3d_field("position")
    idx = laurus.Index(schema=schema)
    for doc_id, (lat, lon) in POINTS.items():
        kind = "bar" if doc_id == "inside_east" else "cafe"
        idx.put_document(doc_id, {"kind": kind, "location": {"lat": lat, "lon": lon}})
    for doc_id, (x, y, z) in POSITIONS.items():
        idx.put_document(doc_id, {"kind": "cafe", "position": {"x": x, "y": y, "z": z}})
    idx.commit()
    return idx


def polygon():
    return laurus.GeoPolygonQuery.within_polygon(
        "location", latlon(EXTERIOR), holes=[latlon(HOLE)]
    )


def polyline():
    return laurus.GeoPolylineQuery.within_distance("location", latlon(LINE), 200_000.0)


def polyline_3d():
    return laurus.Geo3dPolylineQuery.within_distance("position", xyz(LINE_3D), 100.0)


def ids(index, query):
    return {r.id for r in index.search(query, limit=20).results}


def ranked(index, query):
    return [r.id for r in index.search(query, limit=20).results]


CAFES = (set(POINTS) | set(POSITIONS)) - {"inside_east"}


# ---------------------------------------------------------------------------
# The hits
# ---------------------------------------------------------------------------


def test_a_polygon_with_a_hole(index):
    assert ids(index, polygon()) == {"inside", "inside_east"}


def test_a_polygon_without_holes_takes_the_hole(index):
    query = laurus.GeoPolygonQuery.within_polygon("location", latlon(EXTERIOR))
    assert ids(index, query) == {"inside", "inside_east", "in_hole"}


def test_a_polyline_ranks_the_closest_first(index):
    # 0 m, about 111 km and about 167 km of 200 km.
    results = index.search(polyline(), limit=20).results
    assert [r.id for r in results] == ["on_line", "near_line", "past_end"]
    assert results[0].score == pytest.approx(1.0)


def test_a_geo3d_polyline_ranks_the_closest_first(index):
    # 0 m, 50 m and 60 m of 100 m.
    results = index.search(polyline_3d(), limit=20).results
    assert [r.id for r in results] == ["p_on", "p_near", "p_end"]
    assert [r.score for r in results] == pytest.approx([1.0, 0.5, 0.4])


def test_the_dsl_gives_the_same_hits(index):
    assert ids(
        index,
        "location:geo_polygon((10, 30, 10, 40, 20, 40, 20, 30), "
        "(14, 34, 14, 36, 16, 36, 16, 34))",
    ) == ids(index, polygon())
    assert ranked(index, "location:geo_polyline((0, 0, 0, 10), 200000)") == ranked(
        index, polyline()
    )
    assert ranked(
        index, "position:geo3d_polyline((0, 0, 0, 1000, 0, 0, 1000, 1000, 0), 100)"
    ) == ranked(index, polyline_3d())


# ---------------------------------------------------------------------------
# In combination
# ---------------------------------------------------------------------------


def test_the_queries_in_a_boolean_query(index):
    for query, hits in [
        (polygon(), {"inside", "inside_east"}),
        (polyline(), {"on_line", "near_line", "past_end"}),
        (polyline_3d(), {"p_on", "p_near", "p_end"}),
    ]:
        must = laurus.BooleanQuery()
        must.must(query)
        must.must(laurus.TermQuery("kind", "cafe"))
        assert ids(index, must) == hits & CAFES, repr(query)

        must_not = laurus.BooleanQuery()
        must_not.must(laurus.TermQuery("kind", "cafe"))
        must_not.must_not(query)
        assert ids(index, must_not) == CAFES - hits, repr(query)


def test_the_queries_as_a_filter(index):
    for query, hits in [
        (polygon(), {"inside", "inside_east"}),
        (polyline(), {"on_line", "near_line", "past_end"}),
        (polyline_3d(), {"p_on", "p_near", "p_end"}),
    ]:
        request = laurus.SearchRequest(
            lexical_query=laurus.TermQuery("kind", "cafe"),
            filter_query=query,
            limit=20,
        )
        assert {r.id for r in index.search(request).results} == hits & CAFES, repr(query)


# ---------------------------------------------------------------------------
# Rejected when built
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, build, error",
    [
        (
            "an exterior of 2 points",
            lambda: laurus.GeoPolygonQuery.within_polygon(
                "location", latlon([(0.0, 0.0), (0.0, 1.0)])
            ),
            ValueError,
        ),
        (
            "a hole of 2 points",
            lambda: laurus.GeoPolygonQuery.within_polygon(
                "location", latlon(EXTERIOR), holes=[latlon([(15.0, 35.0), (15.0, 36.0)])]
            ),
            ValueError,
        ),
        (
            "a latitude out of range",
            lambda: laurus.GeoPolygonQuery.within_polygon(
                "location", latlon([(95.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
            ),
            ValueError,
        ),
        (
            "an edge across exactly 180 degrees",
            lambda: laurus.GeoPolygonQuery.within_polygon(
                "location", latlon([(0.0, -90.0), (0.0, 90.0), (10.0, 90.0)])
            ),
            ValueError,
        ),
        (
            "a polyline edge between antipodes",
            lambda: laurus.GeoPolylineQuery.within_distance(
                "location", latlon([(0.0, 0.0), (0.0, 180.0)]), 1000.0
            ),
            ValueError,
        ),
        (
            "a polyline without vertices",
            lambda: laurus.GeoPolylineQuery.within_distance("location", [], 1000.0),
            ValueError,
        ),
        (
            "a NaN coordinate in 3D",
            lambda: laurus.Geo3dPolylineQuery.within_distance(
                "position", xyz([(0.0, 0.0, 0.0), (math.nan, 0.0, 0.0)]), 100.0
            ),
            ValueError,
        ),
        (
            "a 3D polyline without vertices",
            lambda: laurus.Geo3dPolylineQuery.within_distance("position", [], 100.0),
            ValueError,
        ),
        (
            "a vertex without its longitude",
            lambda: laurus.GeoPolylineQuery.within_distance(
                "location", [{"lat": 1.0}], 1000.0
            ),
            ValueError,
        ),
        (
            "a vertex without its z",
            lambda: laurus.Geo3dPolylineQuery.within_distance(
                "position", [{"x": 1.0, "y": 2.0}], 100.0
            ),
            ValueError,
        ),
        (
            "a (lat, lon) tuple vertex",
            lambda: laurus.GeoPolygonQuery.within_polygon("location", EXTERIOR),
            ValueError,
        ),
        (
            "a (lat, lon) tuple vertex of a hole",
            lambda: laurus.GeoPolygonQuery.within_polygon(
                "location", latlon(EXTERIOR), holes=[HOLE]
            ),
            ValueError,
        ),
        (
            "a (lat, lon) tuple vertex of a polyline",
            lambda: laurus.GeoPolylineQuery.within_distance("location", LINE, 1000.0),
            ValueError,
        ),
        (
            "an (x, y, z) tuple vertex",
            lambda: laurus.Geo3dPolylineQuery.within_distance("position", LINE_3D, 100.0),
            ValueError,
        ),
        (
            "a vertex that is a string",
            lambda: laurus.GeoPolygonQuery.within_polygon(
                "location", ["10, 30", {"lat": 10.0, "lon": 40.0}, {"lat": 20.0, "lon": 40.0}]
            ),
            ValueError,
        ),
        (
            "a latitude that is not a number",
            lambda: laurus.GeoPolylineQuery.within_distance(
                "location", [{"lat": "north", "lon": 1.0}], 1000.0
            ),
            ValueError,
        ),
    ],
)
def test_an_invalid_shape_is_rejected_when_built(label, build, error):
    with pytest.raises(error):
        build()


def test_the_reprs():
    assert repr(polygon()) == (
        "GeoPolygonQuery.within_polygon(field='location', exterior=4 vertices, holes=1)"
    )
    assert repr(polyline()) == (
        "GeoPolylineQuery.within_distance(field='location', vertices=2, distance_m=200000)"
    )
    assert repr(polyline_3d()) == (
        "Geo3dPolylineQuery.within_distance(field='position', vertices=3, distance_m=100)"
    )
