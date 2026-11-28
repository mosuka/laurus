/**
 * Integration tests for the polygon and polyline queries (Issue #1596).
 *
 * Covers `GeoPolygonQuery`, `GeoPolylineQuery` and `Geo3dPolylineQuery`:
 * the hits and their order, the same hits from the DSL, every
 * `BooleanQuery` and `SearchRequest` method that takes them, and the
 * shapes rejected when the query is built.
 *
 * The points are placed so that reading a vertex with its latitude and
 * longitude swapped gives another valid shape with other hits: "mirror" is
 * "inside" with its coordinates swapped.
 */

import { describe, it, expect } from "vitest";
import {
  Index,
  Schema,
  SearchRequest,
  BooleanQuery,
  TermQuery,
  GeoPolygonQuery,
  GeoPolylineQuery,
  Geo3dPolylineQuery,
} from "../index.js";

const POINTS = {
  // Around the polygon: latitudes 10 to 20, longitudes 30 to 40, with a
  // hole over latitudes 14 to 16, longitudes 34 to 36.
  inside: { lat: 12, lon: 32 },
  inside_east: { lat: 18, lon: 38 },
  in_hole: { lat: 15, lon: 35 },
  outside: { lat: 25, lon: 35 },
  mirror: { lat: 32, lon: 12 },
  // Around the polyline along the equator from longitude 0 to 10.
  on_line: { lat: 0, lon: 3 },
  near_line: { lat: 1, lon: 5 },
  past_end: { lat: 0, lon: 11.5 },
  far: { lat: 5, lon: 5 },
};

const POSITIONS = {
  // Around the 3D polyline (0, 0, 0) -> (1000, 0, 0) -> (1000, 1000, 0).
  p_on: { x: 500, y: 0, z: 0 },
  p_near: { x: 1000, y: 500, z: 50 },
  p_end: { x: 1000, y: 1060, z: 0 },
  p_far: { x: 500, y: 500, z: 0 },
};

const EXTERIOR = [
  { lat: 10, lon: 30 },
  { lat: 10, lon: 40 },
  { lat: 20, lon: 40 },
  { lat: 20, lon: 30 },
];
const HOLE = [
  { lat: 14, lon: 34 },
  { lat: 14, lon: 36 },
  { lat: 16, lon: 36 },
  { lat: 16, lon: 34 },
];
const LINE = [
  { lat: 0, lon: 0 },
  { lat: 0, lon: 10 },
];
const LINE_3D = [
  { x: 0, y: 0, z: 0 },
  { x: 1000, y: 0, z: 0 },
  { x: 1000, y: 1000, z: 0 },
];

const CAFES = new Set(
  [...Object.keys(POINTS), ...Object.keys(POSITIONS)].filter((id) => id !== "inside_east"),
);

const polygon = () => GeoPolygonQuery.withinPolygon("location", EXTERIOR, [HOLE]);
const polyline = () => GeoPolylineQuery.withinDistance("location", LINE, 200_000);
const polyline3d = () => Geo3dPolylineQuery.withinDistance("position", LINE_3D, 100);

const QUERIES = [
  { name: "GeoPolygon", build: polygon, hits: ["inside", "inside_east"] },
  { name: "GeoPolyline", build: polyline, hits: ["on_line", "near_line", "past_end"] },
  { name: "Geo3dPolyline", build: polyline3d, hits: ["p_on", "p_near", "p_end"] },
];

async function createIndex() {
  const schema = new Schema();
  schema.addTextField("kind");
  schema.addGeoField("location");
  schema.addGeo3dField("position");
  const index = await Index.create(null, schema);
  for (const [id, location] of Object.entries(POINTS)) {
    const kind = id === "inside_east" ? "bar" : "cafe";
    await index.putDocument(id, { kind, location });
  }
  for (const [id, position] of Object.entries(POSITIONS)) {
    await index.putDocument(id, { kind: "cafe", position });
  }
  await index.commit();
  return index;
}

function request() {
  return new SearchRequest({ limit: 20 });
}

async function lexical(index, name, query) {
  const req = request();
  req[`setLexical${name}`](query);
  return (await index.searchWithRequest(req)).results;
}

async function idsOf(index, req) {
  const { results } = await index.searchWithRequest(req);
  return new Set(results.map((r) => r.id));
}

const intersect = (a, b) => new Set([...a].filter((x) => b.has(x)));
const minus = (a, b) => new Set([...a].filter((x) => !b.has(x)));

describe("the hits", () => {
  it("a polygon with a hole", async () => {
    const index = await createIndex();
    const results = await lexical(index, "GeoPolygon", polygon());
    expect(new Set(results.map((r) => r.id))).toEqual(new Set(["inside", "inside_east"]));
  });

  it("a polygon without holes takes the hole", async () => {
    const index = await createIndex();
    const results = await lexical(
      index,
      "GeoPolygon",
      GeoPolygonQuery.withinPolygon("location", EXTERIOR),
    );
    expect(new Set(results.map((r) => r.id))).toEqual(
      new Set(["inside", "inside_east", "in_hole"]),
    );
  });

  it("a polyline ranks the closest first", async () => {
    // 0 m, about 111 km and about 167 km of 200 km.
    const index = await createIndex();
    const results = await lexical(index, "GeoPolyline", polyline());
    expect(results.map((r) => r.id)).toEqual(["on_line", "near_line", "past_end"]);
    expect(results[0].score).toBeCloseTo(1.0, 5);
  });

  it("a Geo3d polyline ranks the closest first", async () => {
    // 0 m, 50 m and 60 m of 100 m.
    const index = await createIndex();
    const results = await lexical(index, "Geo3dPolyline", polyline3d());
    expect(results.map((r) => r.id)).toEqual(["p_on", "p_near", "p_end"]);
    expect(results[0].score).toBeCloseTo(1.0, 5);
    expect(results[1].score).toBeCloseTo(0.5, 5);
    expect(results[2].score).toBeCloseTo(0.4, 5);
  });

  it("the DSL gives the same hits", async () => {
    const index = await createIndex();
    const dsl = async (q) => (await index.search(q, 20)).results.map((r) => r.id);
    const objects = async (name, query) =>
      (await lexical(index, name, query)).map((r) => r.id);
    expect(
      new Set(
        await dsl(
          "location:geo_polygon((10, 30, 10, 40, 20, 40, 20, 30), (14, 34, 14, 36, 16, 36, 16, 34))",
        ),
      ),
    ).toEqual(new Set(await objects("GeoPolygon", polygon())));
    expect(await dsl("location:geo_polyline((0, 0, 0, 10), 200000)")).toEqual(
      await objects("GeoPolyline", polyline()),
    );
    expect(
      await dsl("position:geo3d_polyline((0, 0, 0, 1000, 0, 0, 1000, 1000, 0), 100)"),
    ).toEqual(await objects("Geo3dPolyline", polyline3d()));
  });
});

describe.each(QUERIES)("$name in combination", ({ name, build, hits }) => {
  const expected = new Set(hits);

  it(`setFilter${name}`, async () => {
    const index = await createIndex();
    const req = request();
    req.setLexicalTerm(new TermQuery("kind", "cafe"));
    req[`setFilter${name}`](build());
    expect(await idsOf(index, req)).toEqual(intersect(expected, CAFES));
  });

  it(`must${name}`, async () => {
    const index = await createIndex();
    const bq = new BooleanQuery();
    bq[`must${name}`](build());
    bq.mustTerm(new TermQuery("kind", "cafe"));
    const req = request();
    req.setLexicalBoolean(bq);
    expect(await idsOf(index, req)).toEqual(intersect(expected, CAFES));
  });

  it(`should${name}`, async () => {
    const index = await createIndex();
    const bq = new BooleanQuery();
    bq[`should${name}`](build());
    const req = request();
    req.setLexicalBoolean(bq);
    expect(await idsOf(index, req)).toEqual(expected);
  });

  it(`mustNot${name}`, async () => {
    const index = await createIndex();
    const bq = new BooleanQuery();
    bq.mustTerm(new TermQuery("kind", "cafe"));
    bq[`mustNot${name}`](build());
    const req = request();
    req.setLexicalBoolean(bq);
    expect(await idsOf(index, req)).toEqual(minus(CAFES, expected));
  });
});

describe("rejected when built", () => {
  it.each([
    [
      "an exterior of 2 points",
      () =>
        GeoPolygonQuery.withinPolygon("location", [
          { lat: 0, lon: 0 },
          { lat: 0, lon: 1 },
        ]),
    ],
    [
      "a hole of 2 points",
      () =>
        GeoPolygonQuery.withinPolygon("location", EXTERIOR, [
          [
            { lat: 15, lon: 35 },
            { lat: 15, lon: 36 },
          ],
        ]),
    ],
    [
      "a latitude out of range",
      () =>
        GeoPolygonQuery.withinPolygon("location", [
          { lat: 95, lon: 0 },
          { lat: 0, lon: 1 },
          { lat: 1, lon: 1 },
        ]),
    ],
    [
      "an edge across exactly 180 degrees",
      () =>
        GeoPolygonQuery.withinPolygon("location", [
          { lat: 0, lon: -90 },
          { lat: 0, lon: 90 },
          { lat: 10, lon: 90 },
        ]),
    ],
    [
      "a polyline edge between antipodes",
      () =>
        GeoPolylineQuery.withinDistance(
          "location",
          [
            { lat: 0, lon: 0 },
            { lat: 0, lon: 180 },
          ],
          1000,
        ),
    ],
    ["a polyline without vertices", () => GeoPolylineQuery.withinDistance("location", [], 1000)],
    [
      "a NaN coordinate in 3D",
      () =>
        Geo3dPolylineQuery.withinDistance(
          "position",
          [
            { x: 0, y: 0, z: 0 },
            { x: NaN, y: 0, z: 0 },
          ],
          100,
        ),
    ],
    [
      "a 3D polyline without vertices",
      () => Geo3dPolylineQuery.withinDistance("position", [], 100),
    ],
    [
      "a vertex without its longitude",
      () => GeoPolylineQuery.withinDistance("location", [{ lat: 1 }], 1000),
    ],
    [
      "a vertex without its z",
      () => Geo3dPolylineQuery.withinDistance("position", [{ x: 1, y: 2 }], 100),
    ],
    [
      "a latitude that is not a number",
      () => GeoPolylineQuery.withinDistance("location", [{ lat: "north", lon: 1 }], 1000),
    ],
    [
      "a vertex that is not an object",
      () => GeoPolylineQuery.withinDistance("location", ["10, 30"], 1000),
    ],
  ])("%s", (_label, build) => {
    expect(build).toThrow();
  });
});
