//! 3D Geographic Search (ECEF) — laurus example
//!
//! This example demonstrates Laurus's 3D Earth-Centered Earth-Fixed (ECEF)
//! geographic search, where positions are stored as Cartesian `(x, y, z)`
//! coordinates in meters from the Earth's center. Unlike 2D `Geo` (lat/lon),
//! `Geo3d` treats altitude as a first-class dimension, so it can represent
//! drones, satellites, multi-floor indoor positions, etc.
//!
//! The example shows how to:
//!
//! 1. Define a schema with a `Geo3d` field.
//! 2. Convert WGS84 (lat / lon / height) to ECEF and index documents.
//! 3. Run all three 3D geo query types:
//!    - `Geo3dDistanceQuery`     — sphere centered at `(x, y, z)`.
//!    - `Geo3dBoundingBoxQuery`  — 3D axis-aligned bounding box.
//!    - `Geo3dNearestQuery`      — k-nearest neighbours.
//!
//! Run with: `cargo run --example geo3d_search`
//!
//! See `docs/src/concepts/geo3d.md` for the coordinate system, the
//! `wgs84_to_ecef` / `ecef_to_wgs84` conversion utilities, and the
//! query DSL reference.

mod common;

use laurus::lexical::core::field::Geo3dOption;
use laurus::lexical::{Geo3dBoundingBoxQuery, Geo3dDistanceQuery, Geo3dNearestQuery, TextOption};
use laurus::util::ecef::wgs84_to_ecef;
use laurus::{
    Document, Engine, GeoEcefPoint, LexicalSearchQuery, Result, Schema, SearchRequestBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Laurus 3D Geographic Search (ECEF) ===\n");

    // ─── Schema with a Geo3d field ──────────────────────────────────────
    let storage = common::memory_storage()?;
    let schema = Schema::builder()
        .add_text_field("name", TextOption::default())
        .add_geo3d_field("position", Geo3dOption::default())
        .build();
    let engine = Engine::new(storage, schema).await?;

    // ─── Index six landmarks ────────────────────────────────────────────
    //
    // Heights span ~65 m (sea level) to 408 km (low Earth orbit), which
    // makes altitude a meaningful third dimension here. Coordinates come
    // from public sources and are rounded to four decimal places.
    let landmarks = [
        ("tokyo_tower", "Tokyo Tower", 35.6586, 139.7454, 333.0),
        ("tokyo_skytree", "Tokyo Skytree", 35.7101, 139.8107, 634.0),
        ("mt_fuji", "Mt. Fuji summit", 35.3606, 138.7274, 3_776.0),
        (
            "statue_of_liberty",
            "Statue of Liberty",
            40.6892,
            -74.0445,
            93.0,
        ),
        (
            "sydney_opera",
            "Sydney Opera House",
            -33.8568,
            151.2153,
            65.0,
        ),
        (
            "iss_sample",
            "ISS sample point",
            35.0000,
            139.0000,
            408_000.0,
        ),
    ];

    for (id, name, lat, lon, height) in landmarks {
        let p = wgs84_to_ecef(lat, lon, height);
        let doc = Document::builder()
            .add_text("name", name)
            .add_geo_ecef("position", p.x, p.y, p.z)
            .build();
        engine.add_document(id, doc).await?;
    }
    engine.commit().await?;
    println!("Indexed {} landmarks.\n", landmarks.len());

    // Helpful query centres precomputed once.
    let tokyo_tower = wgs84_to_ecef(35.6586, 139.7454, 333.0);
    let mt_fuji = wgs84_to_ecef(35.3606, 138.7274, 3_776.0);

    // ─── PART 1: Geo3dDistanceQuery (sphere) ────────────────────────────
    println!("--- PART 1: Geo3dDistanceQuery (sphere) ---");

    println!("\n[1a] Centre: Tokyo Tower, radius: 50 km");
    let query = Geo3dDistanceQuery::new("position", tokyo_tower, 50_000.0);
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(query)))
                .limit(10)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    println!("\n[1b] Centre: Tokyo Tower, radius: 200 km");
    let query = Geo3dDistanceQuery::new("position", tokyo_tower, 200_000.0);
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(query)))
                .limit(10)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // ─── PART 2: Geo3dBoundingBoxQuery (3D AABB) ────────────────────────
    //
    // A small ECEF box that loosely covers central Tokyo at near-surface
    // altitudes. The bounds intentionally exclude Mt. Fuji (too far west)
    // and the ISS sample (408 km up), illustrating that the *third axis*
    // matters: a 2D Geo bbox could not differentiate a surface tower from
    // a satellite directly overhead.
    println!("\n--- PART 2: Geo3dBoundingBoxQuery (3D AABB) ---");
    let tokyo_min = GeoEcefPoint::new(-3_962_000.0, 3_340_000.0, 3_690_000.0);
    let tokyo_max = GeoEcefPoint::new(-3_958_000.0, 3_360_000.0, 3_710_000.0);
    println!(
        "\nBox: ({:.0}, {:.0}, {:.0}) – ({:.0}, {:.0}, {:.0}) (m, ECEF)",
        tokyo_min.x, tokyo_min.y, tokyo_min.z, tokyo_max.x, tokyo_max.y, tokyo_max.z
    );
    let query = Geo3dBoundingBoxQuery::new("position", tokyo_min, tokyo_max)?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(query)))
                .limit(10)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    // ─── PART 3: Geo3dNearestQuery (k-NN) ───────────────────────────────
    println!("\n--- PART 3: Geo3dNearestQuery (k-NN) ---");
    println!("\nCentre: Mt. Fuji summit, k = 3");
    let query = Geo3dNearestQuery::new("position", mt_fuji, 3);
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(query)))
                .limit(3)
                .build(),
        )
        .await?
        .results;
    common::print_search_results(&results);

    Ok(())
}
