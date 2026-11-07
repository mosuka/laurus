# 3D Geographic Search (ECEF)

Laurus offers two geographic field types that target different problems:

| Field type | Backing structure | Coordinate system | Use it when … |
| :--- | :--- | :--- | :--- |
| `Geo` | 2D BKD-Tree | WGS84 latitude / longitude (degrees) | All your data lives at the surface and altitude does not matter. |
| `Geo3d` | 3D BKD-Tree | ECEF Cartesian `(x, y, z)` (metres) | Altitude is a first-class dimension — drones, satellites, indoor positioning, multi-floor buildings. |

Both share the same [BKD-Tree](bkd_tree.md) primitive; `Geo3d` simply adds a
third dimension and a different query vocabulary.

## Why ECEF?

`(latitude, longitude, altitude)` triples are convenient for humans but make
Euclidean distance unusable: a degree of longitude is ~111 km at the equator
and 0 km at the poles, and "altitude" is curved with the Earth's surface.

ECEF (Earth-Centered Earth-Fixed) flattens this into a Cartesian frame:

- The **origin** is the Earth's centre of mass.
- The **+X axis** points through the equator at 0° longitude.
- The **+Y axis** points through the equator at 90° E longitude.
- The **+Z axis** points through the geographic North Pole.
- All three axes are in **metres**.

In this frame the **straight-line distance between two points is just the
Euclidean norm** — no spherical trigonometry, no pole singularity, no
longitude-wrap. That property is what makes `Geo3d` queries usable from a
3D BKD-Tree without bespoke spatial code.

## Defining a Geo3d Field

```rust
use laurus::Schema;
use laurus::lexical::core::field::Geo3dOption;

let schema = Schema::builder()
    .add_geo3d_field("position", Geo3dOption::default())
    .build();
```

`Geo3dOption` exposes the same `indexed` / `stored` switches as the other
lexical field options. Indexed values populate the field's 3D BKD-Tree;
stored values are returned by `get_documents`.

The TOML form (used by `laurus-cli`):

```toml
[fields.position.Geo3d]
indexed = true
stored = true
```

## Indexing 3D Points

Use `DocumentBuilder::add_geo_ecef` to add a point in raw Cartesian
metres:

```rust
use laurus::Document;

// A point at lat=35.6586°, lon=139.7454°, height=250 m
// (already converted to ECEF — see "Coordinate Conversion" below).
let doc = Document::builder()
    .add_geo_ecef("position", -3_955_182.0, 3_350_553.0, 3_700_276.0)
    .build();

engine.put_document("tokyo-tower", doc).await?;
engine.commit().await?;
```

If the value already exists as a `GeoEcefPoint`, build it through the
unified `DataValue` API instead:

```rust
use laurus::{DataValue, GeoEcefPoint};

let p = GeoEcefPoint::new(-3_955_182.0, 3_350_553.0, 3_700_276.0);
let doc = Document::builder()
    .add_field("position", DataValue::GeoEcef(p))
    .build();
```

## Coordinate Conversion (WGS84 ↔ ECEF)

Most input arrives as `(latitude°, longitude°, height_m)`. Laurus exposes
the canonical pair of conversions in [`laurus::util::ecef`](https://github.com/mosuka/laurus/blob/main/laurus/src/util/ecef.rs):

```rust
use laurus::util::ecef::{wgs84_to_ecef, ecef_to_wgs84};

// Forward: lat/lon/height → ECEF
let p = wgs84_to_ecef(35.6586, 139.7454, 250.0);

// Reverse: ECEF → lat/lon/height
let (lat, lon, height) = ecef_to_wgs84(&p);
```

| Function | Direction | Algorithm |
| :--- | :--- | :--- |
| `wgs84_to_ecef(lat°, lon°, h_m)` | geographic → Cartesian | Closed-form prime-vertical formula. |
| `ecef_to_wgs84(&GeoEcefPoint)` | Cartesian → geographic | Bowring 1985 closed-form seed + 3 Newton-Raphson iterations. |

The reverse conversion round-trips with **sub-µm** precision on every axis
from sub-surface up to far beyond LEO altitudes. Near the poles the
implementation switches from the standard `h = p / cos(lat) - N` formula
(which diverges at `cos(lat) → 0`) to the meridian form
`h = z / sin(lat) - N(1 - e²)`.

The WGS84 ellipsoid constants used internally are also exposed as `pub
const`s (`WGS84_A`, `WGS84_F`, `WGS84_B`, `WGS84_E2`, `WGS84_E_PRIME_SQ`)
so external code can reference the exact values laurus uses.

## Query Types

Three query types target `Geo3d` fields. All three are
[`IntersectVisitor`](bkd_tree.md#query-the-intersectvisitor-protocol) implementations
on top of the shared 3D BKD-Tree.

### Sphere (radius) — `Geo3dDistanceQuery`

Find every document whose stored point lies within `distance_m` metres of a
centre point. Score is `1 - distance / radius`, clamped to `[0, 1]`.

```rust
use laurus::lexical::query::geo3d::Geo3dDistanceQuery;
use laurus::GeoEcefPoint;

let centre = GeoEcefPoint::new(-3_955_182.0, 3_350_553.0, 3_700_276.0);
let query = Geo3dDistanceQuery::new("position", centre, 5_000.0); // 5 km radius
```

The visitor prunes cells with [`AABB::min_distance_sq_to_point`] (the
`Outside` test, in squared-distance space so it avoids `sqrt`) and never
returns `CellRelation::Inside` from `compare`: the `Inside` path reports doc
ids without their coordinates, and the query needs every point's real
distance for its score and its distance-ascending order. A cell that lies
wholly within the sphere is therefore visited as `Crosses` and its points
decoded, as the k-NN visitor below does. (Before #1594 such cells were
reported `Inside`, and every point under them got distance `0` and score
`1.0` regardless of where it actually was.)

### Bounding Box — `Geo3dBoundingBoxQuery`

Find every document whose stored point lies inside an axis-aligned 3D box
defined by `(min_x, min_y, min_z)` and `(max_x, max_y, max_z)`.

```rust
use laurus::lexical::query::geo3d::Geo3dBoundingBoxQuery;
use laurus::GeoEcefPoint;

let min = GeoEcefPoint::new(-4_000_000.0, 3_300_000.0, 3_650_000.0);
let max = GeoEcefPoint::new(-3_900_000.0, 3_400_000.0, 3_750_000.0);
let query = Geo3dBoundingBoxQuery::new("position", min, max)?;
```

The constructor returns `Result` because every `min[i]` must be `≤ max[i]`;
mismatched bounds are rejected at construction time rather than producing
silently-empty results.

> **Caveat.** The query box is axis-aligned in ECEF Cartesian space. A box
> built from two `(lat, lon, height)` corners by converting each corner
> separately is **not** the same shape as the spherical-shell region a user
> might intuitively expect. For region queries on the Earth's surface,
> `Geo3dDistanceQuery` (a true 3D sphere) usually matches user intuition
> better.

### k-Nearest Neighbours — `Geo3dNearestQuery`

Find the `k` nearest documents to a query point. The query starts from a
small probe sphere (default 1 km) and **doubles the radius** until either:

1. it has collected at least `k` distinct candidates,
2. doubling makes no progress,
3. the radius reaches `max_radius_m` (default `1.0e10` m, i.e. 10⁷ km).

```rust
use laurus::lexical::query::geo3d::Geo3dNearestQuery;
use laurus::GeoEcefPoint;

let centre = GeoEcefPoint::new(-3_955_182.0, 3_350_553.0, 3_700_276.0);
let query = Geo3dNearestQuery::new("position", centre, 10) // 10 nearest
    .with_initial_radius(500.0)         // start at 500 m
    .with_max_radius(1_000_000.0);      // cap at 1 000 km
```

A smaller-than-`k` result simply means the index does not contain `k`
points within `max_radius_m`. Score is normalised so that the closest hit
gets `1.0` and the farthest hit in the returned set gets `0.0`.

The k-NN visitor never returns `CellRelation::Inside` from `compare` —
only `Outside` or `Crosses` — so every candidate is delivered to `visit`
with its exact coordinates, ensuring k-NN ordering uses the true distance.

## Query DSL

`Geo3d` queries are also exposed in the unified Query DSL (see
[Query DSL → Geo3d Functions](query_dsl.md#3d-geographic-queries-geo3d_)):

```text
position:geo3d_distance(-3955182, 3350553, 3700276, 5000)
position:geo3d_bbox(-4000000, 3300000, 3650000, -3900000, 3400000, 3750000)
position:geo3d_nearest(-3955182, 3350553, 3700276, 10)
```

| Form | Arguments |
| :--- | :--- |
| `geo3d_distance(x, y, z, distance_m)` | centre `(x, y, z)` and maximum distance in metres |
| `geo3d_bbox(min_x, min_y, min_z, max_x, max_y, max_z)` | two opposite corners of the AABB |
| `geo3d_nearest(x, y, z, k)` | centre `(x, y, z)` and the integer `k` |

All numeric arguments are signed floats; the `k` of `geo3d_nearest` is an
unsigned integer.

## Wire Formats

### gRPC

Protocol Buffers expose 3D geo as a dedicated `Value` variant alongside a
`Geo3dPoint` message and a `Geo3dOption` schema option:

```proto
message Geo3dPoint {
  double x = 1;
  double y = 2;
  double z = 3;
}

message Value {
  oneof kind {
    // ...
    Geo3dPoint geo3d_value = 12;
  }
}

message Geo3dOption {
  bool indexed = 1;
  bool stored = 2;
}
```

See [`common.proto`](https://github.com/mosuka/laurus/blob/main/laurus-server/proto/laurus/v1/common.proto) and
[`index.proto`](https://github.com/mosuka/laurus/blob/main/laurus-server/proto/laurus/v1/index.proto) for the full definitions.

### HTTP gateway / MCP

JSON documents accept `Geo3d` values as a `{ "x": …, "y": …, "z": … }`
object:

```json
{
  "position": { "x": -3955182.0, "y": 3350553.0, "z": 3700276.0 }
}
```

The HTTP gateway converts this to `DataValue::GeoEcef` before forwarding
to the engine; the MCP server uses the same convention.

## When *not* to Use Geo3d

- **Pure 2D map queries** (radius from a coordinate on the surface, simple
  bounding boxes on a tile map) — keep using `Geo`. The 2D BKD is one
  dimension lighter and the WGS84 input matches user intuition.
- **Approximate semantic similarity** of a learned location embedding —
  that is a vector field (`Hnsw`, `Flat`, `Ivf`), not a `Geo3d` field.

## See Also

- [BKD-Tree](bkd_tree.md) — the underlying multi-dimensional index that
  also backs `Integer`, `Float`, `DateTime`, and 2D `Geo` fields.
- [Schema & Fields](schema_and_fields.md) — `Geo3d` alongside the rest of
  the field type table.
- [Query DSL](query_dsl.md) — full geo3d DSL grammar.
- [Lexical Search](search/lexical_search.md) — programmatic entry points
  for every lexical query type.
- [`laurus-wasm/examples/geo3d/`](https://github.com/mosuka/laurus/tree/main/laurus-wasm/examples/geo3d) —
  browser-side demo: live satellite positions on a CesiumJS 3D globe with
  `geo3d_bbox` and `geo3d_nearest` queries.

[`AABB::min_distance_sq_to_point`]: https://github.com/mosuka/laurus/blob/main/laurus/src/lexical/index/structures/aabb.rs
[`AABB::max_distance_sq_to_point`]: https://github.com/mosuka/laurus/blob/main/laurus/src/lexical/index/structures/aabb.rs
