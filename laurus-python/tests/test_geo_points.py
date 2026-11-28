"""A geo point in a Python document is a dict (Issue #1771).

`{"lat": ..., "lon": ...}` is a `Geo` point and `{"x": ..., "y": ..., "z": ...}`
a `Geo3d` point, alone or in a list for a multi-valued field, as in the other
bindings, and a stored point comes back as the same dict. A tuple is no
longer a point: a tuple of 2 or 3 numbers raises a ValueError that says how
to write it.
"""

import re

import pytest
import laurus

TOKYO = {"lat": 35.68, "lon": 139.76}
OSAKA = {"lat": 34.69, "lon": 135.50}
TOKYO_TOWER = {"x": -3955182.0, "y": 3350553.0, "z": 3700276.0}
SKYTREE = {"x": -3961178.0, "y": 3346187.0, "z": 3702490.0}


def make_index():
    schema = laurus.Schema()
    schema.add_text_field("name")
    schema.add_geo_field("location")
    schema.add_geo_field("spots", multi_valued=True)
    schema.add_geo3d_field("position")
    schema.add_geo3d_field("positions", multi_valued=True)
    return laurus.Index(schema=schema)


ALL_POINTS = {
    "name": "Tokyo",
    "location": TOKYO,
    "spots": [TOKYO, OSAKA],
    "position": TOKYO_TOWER,
    "positions": [TOKYO_TOWER, SKYTREE],
}


def stored(doc):
    return {k: doc[k] for k in ALL_POINTS if k in doc}


# ---------------------------------------------------------------------------
# A dict point
# ---------------------------------------------------------------------------


def test_put_document_reads_dict_points_and_returns_them():
    idx = make_index()
    idx.put_document("a", ALL_POINTS)
    idx.commit()
    assert stored(idx.get_documents("a")[0]) == ALL_POINTS


def test_put_documents_reads_dict_points():
    idx = make_index()
    idx.put_documents([("a", ALL_POINTS), ("b", {"name": "Osaka", "location": OSAKA})])
    idx.commit()
    assert stored(idx.get_documents("a")[0]) == ALL_POINTS
    assert idx.get_documents("b")[0]["location"] == OSAKA


def test_the_hits_carry_the_stored_points_and_the_queries_find_them():
    idx = make_index()
    idx.put_document("a", ALL_POINTS)
    idx.put_document("b", {"name": "Osaka", "location": OSAKA})
    idx.commit()

    hits = idx.search(
        laurus.GeoDistanceQuery.within_radius("location", 35.68, 139.76, 1_000.0), limit=10
    ).results
    assert [h.id for h in hits] == ["a"]
    assert stored(hits[0].document) == ALL_POINTS

    hits = idx.search(
        laurus.GeoDistanceQuery.within_radius("spots", 34.69, 135.50, 1_000.0), limit=10
    ).results
    assert [h.id for h in hits] == ["a"]

    hits = idx.search(
        laurus.Geo3dDistanceQuery.within_sphere(
            "positions", SKYTREE["x"], SKYTREE["y"], SKYTREE["z"], 100.0
        ),
        limit=10,
    ).results
    assert [h.id for h in hits] == ["a"]


def test_a_dict_with_both_sets_of_keys_is_a_geo_point():
    idx = make_index()
    idx.put_document("a", {"location": {**TOKYO, "x": 1.0, "y": 2.0, "z": 3.0}})
    idx.commit()
    assert idx.get_documents("a")[0]["location"] == TOKYO


def test_other_keys_are_not_read():
    idx = make_index()
    idx.put_document(
        "a", {"location": {**TOKYO, "name": "Tokyo"}, "position": {**TOKYO_TOWER, "w": 0.0}}
    )
    idx.commit()
    doc = idx.get_documents("a")[0]
    assert doc["location"] == TOKYO
    assert doc["position"] == TOKYO_TOWER


# ---------------------------------------------------------------------------
# A tuple is not a point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, field, value, written",
    [
        ("a (lat, lon) tuple", "location", (35.68, 139.76),
         ['{"lat": 35.68, "lon": 139.76}', "(35.68, 139.76)"]),
        ("an (x, y, z) tuple", "position", (1.0, 2.0, 3.0),
         ['{"x": 1.0, "y": 2.0, "z": 3.0}', "(1.0, 2.0, 3.0)"]),
        ("a tuple of integers", "location", (35, 139),
         ['{"lat": 35, "lon": 139}', "(35, 139)"]),
        ("a (lat, lon) tuple in a list", "spots", [TOKYO, (34.69, 135.5)],
         ['{"lat": 34.69, "lon": 135.5}', "(34.69, 135.5)"]),
        ("an (x, y, z) tuple in a list", "positions", [(1.0, 2.0, 3.0)],
         ['{"x": 1.0, "y": 2.0, "z": 3.0}', "(1.0, 2.0, 3.0)"]),
    ],
)
def test_a_tuple_point_says_how_to_write_it(label, field, value, written):
    idx = make_index()
    with pytest.raises(ValueError) as e:
        idx.put_document("a", {field: value})
    message = str(e.value)
    for part in [*written, "a geo point is a dict", "Issue #1771"]:
        assert part in message, (label, message)


def test_put_documents_says_how_to_write_a_tuple_point():
    idx = make_index()
    with pytest.raises(ValueError, match=re.escape('{"lat": 35.68, "lon": 139.76}')):
        idx.put_documents([("a", {"location": (35.68, 139.76)})])


@pytest.mark.parametrize(
    "label, value",
    [
        ("a tuple of strings", ("a", "b")),
        ("a tuple of four numbers", (1.0, 2.0, 3.0, 4.0)),
    ],
)
def test_a_tuple_that_is_not_a_point_keeps_its_error(label, value):
    idx = make_index()
    with pytest.raises(ValueError, match="Cannot convert Python value of type tuple"):
        idx.put_document("a", {"name": value})


# ---------------------------------------------------------------------------
# Rejected dicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, field, value, pattern",
    [
        ("a dict without the keys", "location", {"latitude": 35.68, "longitude": 139.76},
         r'"lat" and "lon"'),
        ("a dict with half the keys", "location", {"lat": 35.68}, r'"lat" and "lon"'),
        ("a latitude that is not a number", "location", {"lat": "north", "lon": 139.76},
         r'"lat".*number'),
        ("a z that is not a number", "position", {"x": 1.0, "y": 2.0, "z": None},
         r'"z".*number'),
        ("a latitude out of range", "location", {"lat": 95.0, "lon": 0.0}, "invalid geo point"),
        ("Geo and Geo3d points in one list", "spots", [TOKYO, TOKYO_TOWER], "all"),
        ("a dict without the keys in a list", "spots", [TOKYO, {"a": 1.0}], r'"lat" and "lon"'),
    ],
)
def test_an_invalid_dict_point_is_rejected(label, field, value, pattern):
    idx = make_index()
    with pytest.raises(ValueError, match=pattern):
        idx.put_document("a", {field: value})
