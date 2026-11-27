"""Integration tests for Geo3d (3D ECEF) APIs.

Covers:
- Schema declaration via `add_geo3d_field`.
- Document round-trip with `(x, y, z)` 3-tuple values.
- All three 3D query classes: `Geo3dDistanceQuery`, `Geo3dBoundingBoxQuery`,
  `Geo3dNearestQuery`.

Coordinates are precomputed ECEF values for well-known landmarks. They were
produced by `laurus::util::ecef::wgs84_to_ecef` so the values match what the
core engine emits at runtime.
"""

import pytest
import laurus


# Precomputed ECEF coordinates (meters) for landmarks. Produced offline via
# `wgs84_to_ecef(lat, lon, height)`.
TOKYO_TOWER = (-3955182.0, 3350553.0, 3700276.0)
TOKYO_SKYTREE = (-3961178.0, 3346187.0, 3702490.0)
MT_FUJI = (-3916073.0, 3437037.0, 3672751.0)
SYDNEY = (-4646847.0, 2553022.0, -3534121.0)


@pytest.fixture
def geo3d_index():
    """Return an in-memory index with a Geo3d-typed `position` field."""
    schema = laurus.Schema()
    schema.add_text_field("name")
    schema.add_geo3d_field("position")
    idx = laurus.Index(schema=schema)
    idx.put_document("tokyo_tower", {"name": "Tokyo Tower", "position": TOKYO_TOWER})
    idx.put_document("tokyo_skytree", {"name": "Tokyo Skytree", "position": TOKYO_SKYTREE})
    idx.put_document("mt_fuji", {"name": "Mt. Fuji summit", "position": MT_FUJI})
    idx.put_document("sydney", {"name": "Sydney Opera House", "position": SYDNEY})
    idx.commit()
    return idx


# ---------------------------------------------------------------------------
# Schema and document round-trip
# ---------------------------------------------------------------------------


def test_geo3d_field_round_trip(geo3d_index):
    docs = geo3d_index.get_documents("tokyo_tower")
    assert len(docs) == 1
    assert docs[0]["name"] == "Tokyo Tower"
    assert docs[0]["position"] == TOKYO_TOWER


# ---------------------------------------------------------------------------
# Geo3dDistanceQuery
# ---------------------------------------------------------------------------


def test_geo3d_distance_query_small_radius(geo3d_index):
    """A 50 km sphere around Tokyo Tower should return Tower + Skytree only."""
    cx, cy, cz = TOKYO_TOWER
    query = laurus.Geo3dDistanceQuery.within_sphere("position", cx, cy, cz, 50_000.0)
    results = geo3d_index.search(query, limit=10).results
    ids = {r.id for r in results}
    assert ids == {"tokyo_tower", "tokyo_skytree"}


def test_geo3d_distance_query_wide_radius(geo3d_index):
    """A 200 km sphere additionally pulls in Mt. Fuji."""
    cx, cy, cz = TOKYO_TOWER
    query = laurus.Geo3dDistanceQuery.within_sphere("position", cx, cy, cz, 200_000.0)
    results = geo3d_index.search(query, limit=10).results
    ids = {r.id for r in results}
    assert ids == {"tokyo_tower", "tokyo_skytree", "mt_fuji"}


# ---------------------------------------------------------------------------
# Geo3dBoundingBoxQuery
# ---------------------------------------------------------------------------


def test_geo3d_bounding_box_query(geo3d_index):
    """Central-Tokyo box returns Tower + Skytree only (Mt. Fuji and Sydney
    are outside the small AABB).

    The X bounds are sized to bracket both `TOKYO_TOWER.x ≈ -3.955M` and
    `TOKYO_SKYTREE.x ≈ -3.961M` while still excluding Mt. Fuji
    (`x ≈ -3.916M`, well above the upper bound) and Sydney (`x ≈ -4.65M`,
    well below the lower bound).
    """
    query = laurus.Geo3dBoundingBoxQuery.within_box(
        "position",
        -3_962_000.0, 3_340_000.0, 3_690_000.0,
        -3_954_000.0, 3_360_000.0, 3_710_000.0,
    )
    results = geo3d_index.search(query, limit=10).results
    ids = {r.id for r in results}
    assert ids == {"tokyo_tower", "tokyo_skytree"}


# ---------------------------------------------------------------------------
# Geo3dNearestQuery
# ---------------------------------------------------------------------------


def test_geo3d_nearest_query(geo3d_index):
    """k = 3 around Mt. Fuji returns Fuji, then Tower / Skytree (in distance
    order — exact ordering of the close-pair is implementation defined, so
    only check membership)."""
    cx, cy, cz = MT_FUJI
    query = laurus.Geo3dNearestQuery.k_nearest("position", cx, cy, cz, 3)
    results = geo3d_index.search(query, limit=3).results
    assert len(results) == 3
    ids = {r.id for r in results}
    assert ids == {"mt_fuji", "tokyo_tower", "tokyo_skytree"}
    # Mt. Fuji must be the closest hit.
    assert results[0].id == "mt_fuji"


def test_geo3d_nearest_query_with_radius_options(geo3d_index):
    """Verify the optional initial / max radius parameters are accepted."""
    cx, cy, cz = TOKYO_TOWER
    query = laurus.Geo3dNearestQuery.k_nearest(
        "position", cx, cy, cz, 2,
        initial_radius_m=10_000.0,
        max_radius_m=10_000_000.0,
    )
    results = geo3d_index.search(query, limit=2).results
    ids = {r.id for r in results}
    assert ids == {"tokyo_tower", "tokyo_skytree"}


# ---------------------------------------------------------------------------
# repr()
# ---------------------------------------------------------------------------


def test_geo3d_query_reprs():
    cx, cy, cz = TOKYO_TOWER
    assert "Geo3dDistanceQuery" in repr(
        laurus.Geo3dDistanceQuery.within_sphere("position", cx, cy, cz, 1000.0)
    )
    assert "Geo3dBoundingBoxQuery" in repr(
        laurus.Geo3dBoundingBoxQuery.within_box(
            "position", 0.0, 0.0, 0.0, 1.0, 1.0, 1.0
        )
    )
    assert "Geo3dNearestQuery" in repr(
        laurus.Geo3dNearestQuery.k_nearest("position", cx, cy, cz, 5)
    )
