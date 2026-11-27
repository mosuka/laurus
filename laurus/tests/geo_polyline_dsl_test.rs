//! Issue #1596: a polyline in the DSL, `field:geo_polyline((lat, lon, ...),
//! distance_m)` on a `Geo` field and `field:geo3d_polyline((x, y, z, ...),
//! distance_m)` on a `Geo3d` field: the documents with a point within the
//! distance of the polyline, scoring `1 - d/r` by the closest point.
//!
//! Every case is searched through both DSL routes of `Engine::search`:
//! `query_dsl`, which the unified parser reads, and a lexical DSL query.

use std::sync::Arc;

use laurus::lexical::core::field::{Geo3dOption, GeoOption};
use laurus::lexical::search::searcher::LexicalSearchQuery;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::util::ecef::wgs84_to_ecef;
use laurus::{Document, Engine, FieldOption, Schema, SearchRequestBuilder, TextOption};

/// The documents with a point of `loc`: an id, a latitude and a longitude.
/// The arc from (35, 139) to (35, 140) bulges north to about 35.0011°N
/// midway, so `on_line` is about 120 m south of it and `near` about 990 m
/// north; `past_end` is about 1.8 km east of its end.
const GEO_DOCS: [(&str, f64, f64); 7] = [
    ("on_line", 35.0, 139.5),
    ("near", 35.01, 139.5),
    ("past_end", 35.0, 140.02),
    ("far", 35.5, 139.5),
    ("pacific_east", -15.0, 179.9),
    ("pacific_west", -15.0, -179.9),
    ("zero", -15.0, 0.0),
];

/// The documents with a point of `pos`: an id, and the latitude, longitude
/// and height the point is converted from.
const GEO3D_DOCS: [(&str, f64, f64, f64); 2] = [
    ("drone_near", 35.2, 139.3, 1_200.0),
    ("drone_far", 36.5, 141.0, 1_000.0),
];

/// A flight path at 1,000 m in ECEF, from (35, 139) to (35.4, 139.6).
fn flight_path() -> String {
    let a = wgs84_to_ecef(35.0, 139.0, 1_000.0);
    let b = wgs84_to_ecef(35.4, 139.6, 1_000.0);
    format!(
        "({:?}, {:?}, {:?}, {:?}, {:?}, {:?})",
        a.x, a.y, a.z, b.x, b.y, b.z
    )
}

async fn engine() -> Engine {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("loc", FieldOption::Geo(GeoOption::default()))
        .add_field("pos", FieldOption::Geo3d(Geo3dOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await.unwrap();
    for (id, lat, lon) in GEO_DOCS {
        let doc = Document::builder()
            .add_text("title", id)
            .add_geo("loc", lat, lon)
            .build();
        engine.put_document(id, doc).await.unwrap();
    }
    for (id, lat, lon, height) in GEO3D_DOCS {
        let p = wgs84_to_ecef(lat, lon, height);
        let doc = Document::builder()
            .add_text("title", id)
            .add_geo_ecef("pos", p.x, p.y, p.z)
            .build();
        engine.put_document(id, doc).await.unwrap();
    }
    engine.commit().await.unwrap();
    engine
}

/// The ids and scores `dsl` finds through each DSL route, sorted by id, or
/// the error.
async fn search(
    engine: &Engine,
    dsl: &str,
) -> Vec<(&'static str, Result<Vec<(String, f32)>, String>)> {
    let routes = [
        ("query_dsl", SearchRequestBuilder::new().query_dsl(dsl)),
        (
            "lexical DSL",
            SearchRequestBuilder::new().lexical_query(LexicalSearchQuery::from(dsl)),
        ),
    ];
    let mut out = Vec::new();
    for (route, builder) in routes {
        let result = engine
            .search(builder.limit(100).build())
            .await
            .map(|response| {
                let mut hits: Vec<(String, f32)> = response
                    .results
                    .into_iter()
                    .map(|hit| (hit.id, hit.score))
                    .collect();
                hits.sort_by(|a, b| a.0.cmp(&b.0));
                hits
            })
            .map_err(|e| e.to_string());
        out.push((route, result));
    }
    out
}

/// What is wrong with what `dsl` finds through each route: not the ids
/// `expected`, or a score outside `(0, 1]`.
async fn wrong_ids(engine: &Engine, dsl: &str, expected: &[&str]) -> Vec<String> {
    let mut wrong = Vec::new();
    for (route, result) in search(engine, dsl).await {
        match result {
            Ok(hits) => {
                let ids: Vec<&str> = hits.iter().map(|(id, _)| id.as_str()).collect();
                let scores_ok = hits.iter().all(|(_, s)| *s > 0.0 && *s <= 1.0);
                if ids != expected || !scores_ok {
                    wrong.push(format!(
                        "'{dsl}' through {route}: found {hits:?}, expected {expected:?}"
                    ));
                }
            }
            Err(e) => wrong.push(format!("'{dsl}' through {route}: {e}")),
        }
    }
    wrong
}

/// What is wrong with `dsl` against `same`: through each route, the hits
/// of `dsl` are those of `same`, with the scores times `factor`.
async fn wrong_against(engine: &Engine, dsl: &str, same: &str, factor: f32) -> Vec<String> {
    let mut wrong = Vec::new();
    let expected = search(engine, same).await;
    for ((route, found), (_, expected)) in search(engine, dsl).await.into_iter().zip(expected) {
        match (found, expected) {
            (Ok(found), Ok(expected)) => {
                let same_ids = found.len() == expected.len()
                    && found.iter().zip(&expected).all(|(f, e)| {
                        f.0 == e.0 && (f.1 - e.1 * factor).abs() <= 1e-6 * factor.max(1.0)
                    });
                if expected.is_empty() || !same_ids {
                    wrong.push(format!(
                        "'{dsl}' through {route}: found {found:?}, '{same}' found {expected:?}"
                    ));
                }
            }
            (found, expected) => wrong.push(format!(
                "'{dsl}' through {route}: {found:?}, '{same}': {expected:?}"
            )),
        }
    }
    wrong
}

/// Issue #1596: a polyline finds the points within the distance of its
/// edges and past its ends, the closest scoring highest.
#[tokio::test(flavor = "multi_thread")]
async fn a_polyline_in_the_dsl() {
    let engine = engine().await;
    let dsl = "loc:geo_polyline((35, 139, 35, 140), 2000)";
    let mut wrong = wrong_ids(&engine, dsl, &["near", "on_line", "past_end"]).await;
    for (route, result) in search(&engine, dsl).await {
        if let Ok(hits) = result {
            let score = |id: &str| hits.iter().find(|(hit, _)| hit == id).map(|(_, s)| *s);
            if !(score("on_line") > score("near") && score("near") > score("past_end")) {
                wrong.push(format!("'{dsl}' through {route}: scores {hits:?}"));
            }
        }
    }
    wrong.extend(
        wrong_ids(
            &engine,
            "loc:geo_polyline((35, 139, 35, 140), 500)",
            &["on_line"],
        )
        .await,
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: an edge across the antimeridian reaches both sides of it.
#[tokio::test(flavor = "multi_thread")]
async fn a_polyline_across_the_antimeridian_in_the_dsl() {
    let engine = engine().await;
    let wrong = wrong_ids(
        &engine,
        "loc:geo_polyline((-15, 179.5, -15, -179.5), 1000)",
        &["pacific_east", "pacific_west"],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a single vertex reads as `geo_distance` and
/// `geo3d_distance`, latitude first and the distance last.
#[tokio::test(flavor = "multi_thread")]
async fn a_single_vertex_in_the_dsl_is_a_distance() {
    let engine = engine().await;
    let p = wgs84_to_ecef(35.2, 139.3, 1_000.0);
    let mut wrong = wrong_against(
        &engine,
        "loc:geo_polyline((35, 139.5), 30000)",
        "loc:geo_distance(35, 139.5, 30000)",
        1.0,
    )
    .await;
    wrong.extend(
        wrong_against(
            &engine,
            &format!(
                "pos:geo3d_polyline(({:?}, {:?}, {:?}), 5000)",
                p.x, p.y, p.z
            ),
            &format!("pos:geo3d_distance({:?}, {:?}, {:?}, 5000)", p.x, p.y, p.z),
            1.0,
        )
        .await,
    );
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a `Geo3d` polyline finds the points within the distance of
/// its straight segments.
#[tokio::test(flavor = "multi_thread")]
async fn a_geo3d_polyline_in_the_dsl() {
    let engine = engine().await;
    let dsl = format!("pos:geo3d_polyline({}, 5000)", flight_path());
    let wrong = wrong_ids(&engine, &dsl, &["drone_near"]).await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a deleted document no longer matches, and a replaced one
/// matches by its new point.
#[tokio::test(flavor = "multi_thread")]
async fn a_deleted_or_replaced_document_in_the_dsl() {
    let engine = engine().await;
    engine.delete_documents("near").await.unwrap();
    let moved = Document::builder()
        .add_text("title", "far")
        .add_geo("loc", 35.003, 139.2)
        .build();
    engine.put_document("far", moved).await.unwrap();
    engine.commit().await.unwrap();
    let wrong = wrong_ids(
        &engine,
        "loc:geo_polyline((35, 139, 35, 140), 2000)",
        &["far", "on_line", "past_end"],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a group's `^` multiplies the scores.
#[tokio::test(flavor = "multi_thread")]
async fn a_polyline_scores_its_boost() {
    let engine = engine().await;
    let dsl = "loc:geo_polyline((35, 139, 35, 140), 2000)";
    let wrong = wrong_against(&engine, &format!("({dsl})^2.5"), dsl, 2.5).await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a count of numbers that is not a multiple of 2 (or 3), an
/// edge between antipodal points, a latitude out of range, a polyline
/// without a distance or a field, and one without a vertex fail to parse.
#[tokio::test(flavor = "multi_thread")]
async fn an_invalid_polyline_in_the_dsl_is_rejected() {
    let engine = engine().await;
    let mut wrong = Vec::new();
    for (dsl, reason) in [
        (
            "loc:geo_polyline((35, 139, 35), 1000)",
            "pairs of latitude and longitude",
        ),
        (
            "pos:geo3d_polyline((1, 2, 3, 4), 1000)",
            "x, y and z for each vertex",
        ),
        ("loc:geo_polyline((0, 0, 0, 180), 1000)", "antipodal"),
        ("loc:geo_polyline((35, 139, 95, 139), 1000)", "latitude"),
        (
            "geo_polyline((35, 139, 35, 140), 1000)",
            "explicit field prefix",
        ),
        ("loc:geo_polyline((35, 139, 35, 140))", ""),
        ("loc:geo_polyline((), 1000)", ""),
    ] {
        for (route, result) in search(&engine, dsl).await {
            match result {
                Ok(hits) => wrong.push(format!("'{dsl}' through {route}: found {hits:?}")),
                Err(e) if e.contains(reason) => {}
                Err(e) => wrong.push(format!("'{dsl}' through {route}: {e}, expected {reason:?}")),
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
