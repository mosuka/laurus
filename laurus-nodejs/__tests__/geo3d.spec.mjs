/**
 * Integration tests for Geo3d (3D ECEF) APIs in the Node.js binding.
 *
 * Covers:
 * - Schema declaration via `addGeo3dField`.
 * - Document round-trip with `{ x, y, z }` JSON values.
 * - All three 3D query setters on `SearchRequest`:
 *   `setLexicalGeo3dDistance`, `setLexicalGeo3dBoundingBox`,
 *   `setLexicalGeo3dNearest`.
 *
 * Coordinates are precomputed ECEF values for well-known landmarks. They were
 * produced by `laurus::util::ecef::wgs84_to_ecef` so the values match what
 * the core engine emits at runtime.
 */

import { describe, it, expect } from "vitest";
import {
  Index,
  Schema,
  SearchRequest,
  Geo3dDistanceQuery,
  Geo3dBoundingBoxQuery,
  Geo3dNearestQuery,
} from "../index.js";

// Precomputed ECEF coordinates (meters) for landmarks.
const TOKYO_TOWER = { x: -3955182.0, y: 3350553.0, z: 3700276.0 };
const TOKYO_SKYTREE = { x: -3961178.0, y: 3346187.0, z: 3702490.0 };
const MT_FUJI = { x: -3916073.0, y: 3437037.0, z: 3672751.0 };
const SYDNEY = { x: -4646847.0, y: 2553022.0, z: -3534121.0 };

async function createGeo3dIndex() {
  const schema = new Schema();
  schema.addTextField("name");
  schema.addGeo3dField("position");
  const index = await Index.create(null, schema);
  await index.putDocument("tokyo_tower", { name: "Tokyo Tower", position: TOKYO_TOWER });
  await index.putDocument("tokyo_skytree", { name: "Tokyo Skytree", position: TOKYO_SKYTREE });
  await index.putDocument("mt_fuji", { name: "Mt. Fuji summit", position: MT_FUJI });
  await index.putDocument("sydney", { name: "Sydney Opera House", position: SYDNEY });
  await index.commit();
  return index;
}

describe("Geo3d field round-trip", () => {
  it("returns the stored {x, y, z} object on get", async () => {
    const index = await createGeo3dIndex();
    const docs = await index.getDocuments("tokyo_tower");
    expect(docs).toHaveLength(1);
    expect(docs[0].name).toBe("Tokyo Tower");
    expect(docs[0].position).toEqual(TOKYO_TOWER);
  });
});

describe("Geo3dDistanceQuery", () => {
  it("50 km sphere around Tokyo Tower returns Tower + Skytree", async () => {
    const index = await createGeo3dIndex();
    const req = new SearchRequest();
    req.setLexicalGeo3dDistance(
      Geo3dDistanceQuery.withinSphere(
        "position",
        TOKYO_TOWER.x,
        TOKYO_TOWER.y,
        TOKYO_TOWER.z,
        50_000.0,
      ),
    );
    const { results } = await index.searchWithRequest(req);
    const ids = new Set(results.map((r) => r.id));
    expect(ids).toEqual(new Set(["tokyo_tower", "tokyo_skytree"]));
  });

  it("200 km sphere additionally pulls in Mt. Fuji", async () => {
    const index = await createGeo3dIndex();
    const req = new SearchRequest();
    req.setLexicalGeo3dDistance(
      Geo3dDistanceQuery.withinSphere(
        "position",
        TOKYO_TOWER.x,
        TOKYO_TOWER.y,
        TOKYO_TOWER.z,
        200_000.0,
      ),
    );
    const { results } = await index.searchWithRequest(req);
    const ids = new Set(results.map((r) => r.id));
    expect(ids).toEqual(new Set(["tokyo_tower", "tokyo_skytree", "mt_fuji"]));
  });
});

describe("Geo3dBoundingBoxQuery", () => {
  it("central-Tokyo box returns Tower + Skytree only", async () => {
    // The X bounds bracket both TOKYO_TOWER.x ≈ -3.955M and
    // TOKYO_SKYTREE.x ≈ -3.961M while still excluding Mt. Fuji
    // (x ≈ -3.916M, well above the upper bound) and Sydney
    // (x ≈ -4.65M, well below the lower bound).
    const index = await createGeo3dIndex();
    const req = new SearchRequest();
    req.setLexicalGeo3dBoundingBox(
      Geo3dBoundingBoxQuery.withinBox(
        "position",
        -3_962_000.0,
        3_340_000.0,
        3_690_000.0,
        -3_954_000.0,
        3_360_000.0,
        3_710_000.0,
      ),
    );
    const { results } = await index.searchWithRequest(req);
    const ids = new Set(results.map((r) => r.id));
    expect(ids).toEqual(new Set(["tokyo_tower", "tokyo_skytree"]));
  });
});

describe("Geo3dNearestQuery", () => {
  it("k = 3 around Mt. Fuji returns Fuji + Tower + Skytree", async () => {
    const index = await createGeo3dIndex();
    const req = new SearchRequest();
    req.setLexicalGeo3dNearest(
      Geo3dNearestQuery.kNearest("position", MT_FUJI.x, MT_FUJI.y, MT_FUJI.z, 3),
    );
    const { results } = await index.searchWithRequest(req);
    expect(results).toHaveLength(3);
    const ids = new Set(results.map((r) => r.id));
    expect(ids).toEqual(new Set(["mt_fuji", "tokyo_tower", "tokyo_skytree"]));
    // Mt. Fuji must be the closest hit.
    expect(results[0].id).toBe("mt_fuji");
  });

  it("accepts optional initial / max radius bounds", async () => {
    const index = await createGeo3dIndex();
    const req = new SearchRequest();
    req.setLexicalGeo3dNearest(
      Geo3dNearestQuery.kNearest(
        "position",
        TOKYO_TOWER.x,
        TOKYO_TOWER.y,
        TOKYO_TOWER.z,
        2,
        10_000.0,
        10_000_000.0,
      ),
    );
    const { results } = await index.searchWithRequest(req);
    const ids = new Set(results.map((r) => r.id));
    expect(ids).toEqual(new Set(["tokyo_tower", "tokyo_skytree"]));
  });
});

describe("Geo3d query factory classes", () => {
  it("Geo3dDistanceQuery.withinSphere creates an instance", () => {
    const q = Geo3dDistanceQuery.withinSphere(
      "position",
      TOKYO_TOWER.x,
      TOKYO_TOWER.y,
      TOKYO_TOWER.z,
      1000.0,
    );
    expect(q).toBeDefined();
  });

  it("Geo3dBoundingBoxQuery.withinBox creates an instance", () => {
    const q = Geo3dBoundingBoxQuery.withinBox(
      "position",
      0.0,
      0.0,
      0.0,
      1.0,
      1.0,
      1.0,
    );
    expect(q).toBeDefined();
  });

  it("Geo3dNearestQuery.kNearest creates an instance", () => {
    const q = Geo3dNearestQuery.kNearest(
      "position",
      TOKYO_TOWER.x,
      TOKYO_TOWER.y,
      TOKYO_TOWER.z,
      5,
    );
    expect(q).toBeDefined();
  });
});
