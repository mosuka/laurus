//! Issue #1596: a polygon in the DSL, `field:geo_polygon((lat, lon, ...),
//! (lat, lon, ...))`: the first ring is the exterior and the others are
//! holes, each a list of latitudes and longitudes, latitude first as in the
//! other geo functions. A point on an exterior edge matches and one on a
//! hole's edge does not, an edge takes the shorter way across the
//! antimeridian, and every hit scores the boost.
//!
//! Every case is searched through both DSL routes of `Engine::search`:
//! `query_dsl`, which the unified parser reads, and a lexical DSL query.

use std::sync::Arc;

use laurus::lexical::core::field::GeoOption;
use laurus::lexical::search::searcher::LexicalSearchQuery;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{Document, Engine, FieldOption, Schema, SearchRequestBuilder, TextOption};

/// The documents: an id and a point of `loc`.
const DOCS: [(&str, f64, f64); 9] = [
    ("inside", 35.2, 139.2),
    ("in_hole", 35.5, 139.5),
    ("outside", 37.0, 139.5),
    ("on_edge", 35.0, 139.5),
    ("on_vertex", 36.0, 140.0),
    ("on_hole_edge", 35.4, 139.5),
    ("east", -15.0, 175.0),
    ("west", -15.0, -175.0),
    ("zero", -15.0, 0.0),
];

/// A square from (35, 139) to (36, 140), with a square hole from
/// (35.4, 139.4) to (35.6, 139.6); the rings are not closed, which the DSL
/// accepts.
const SQUARE_WITH_A_HOLE: &str = "loc:geo_polygon((35, 139, 35, 140, 36, 140, 36, 139), \
     (35.4, 139.4, 35.4, 139.6, 35.6, 139.6, 35.6, 139.4))";

/// A rectangle over the Pacific, from 170 degrees east to 170 degrees west.
const ACROSS_THE_ANTIMERIDIAN: &str = "loc:geo_polygon((-20, 170, -20, -170, -10, -170, -10, 170))";

async fn engine() -> Engine {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("loc", FieldOption::Geo(GeoOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await.unwrap();
    for (id, lat, lon) in DOCS {
        let doc = Document::builder()
            .add_text("title", id)
            .add_geo("loc", lat, lon)
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
            .map(|hits| {
                let mut hits: Vec<(String, f32)> =
                    hits.into_iter().map(|hit| (hit.id, hit.score)).collect();
                hits.sort_by(|a, b| a.0.cmp(&b.0));
                hits
            })
            .map_err(|e| e.to_string());
        out.push((route, result));
    }
    out
}

/// The cases that do not find exactly `expected`, each scoring `score`,
/// through one of the two routes.
async fn wrong_cases(engine: &Engine, cases: &[(&str, Vec<&str>, f32)]) -> Vec<String> {
    let mut wrong = Vec::new();
    for (dsl, expected, score) in cases {
        for (route, result) in search(engine, dsl).await {
            match result {
                Ok(hits) => {
                    let ids: Vec<&str> = hits.iter().map(|(id, _)| id.as_str()).collect();
                    let scores_ok = hits.iter().all(|(_, s)| (s - score).abs() < 1e-6);
                    if ids != *expected || !scores_ok {
                        wrong.push(format!(
                            "'{dsl}' through {route}: found {hits:?}, expected {expected:?} scoring {score}"
                        ));
                    }
                }
                Err(e) => wrong.push(format!("'{dsl}' through {route}: {e}")),
            }
        }
    }
    wrong
}

/// Issue #1596: a polygon with a hole matches the points inside it and on
/// its exterior's edges and vertices, and neither those in the hole nor on
/// its edges; a closed ring is read as the same ring unclosed.
#[tokio::test(flavor = "multi_thread")]
async fn a_polygon_with_a_hole_in_the_dsl() {
    let engine = engine().await;
    let closed = "loc:geo_polygon((35, 139, 35, 140, 36, 140, 36, 139, 35, 139), \
         (35.4, 139.4, 35.4, 139.6, 35.6, 139.6, 35.6, 139.4, 35.4, 139.4))";
    let without_the_hole = "loc:geo_polygon((35, 139, 35, 140, 36, 140, 36, 139))";
    let wrong = wrong_cases(
        &engine,
        &[
            (
                SQUARE_WITH_A_HOLE,
                vec!["inside", "on_edge", "on_vertex"],
                1.0,
            ),
            (closed, vec!["inside", "on_edge", "on_vertex"], 1.0),
            (
                without_the_hole,
                vec!["in_hole", "inside", "on_edge", "on_hole_edge", "on_vertex"],
                1.0,
            ),
        ],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: an edge from 170 to -170 degrees takes the shorter way,
/// across the antimeridian, so the rectangle is over the Pacific.
#[tokio::test(flavor = "multi_thread")]
async fn a_polygon_across_the_antimeridian_in_the_dsl() {
    let engine = engine().await;
    let wrong = wrong_cases(
        &engine,
        &[(ACROSS_THE_ANTIMERIDIAN, vec!["east", "west"], 1.0)],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: a deleted document no longer matches, and a replaced one
/// matches by its new point, after a later commit.
#[tokio::test(flavor = "multi_thread")]
async fn a_deleted_or_replaced_document_in_the_dsl() {
    let engine = engine().await;
    engine.delete_documents("inside").await.unwrap();
    let moved_in = Document::builder()
        .add_text("title", "outside")
        .add_geo("loc", 35.3, 139.3)
        .build();
    engine.put_document("outside", moved_in).await.unwrap();
    let moved_out = Document::builder()
        .add_text("title", "on_edge")
        .add_geo("loc", 34.0, 139.5)
        .build();
    engine.put_document("on_edge", moved_out).await.unwrap();
    engine.commit().await.unwrap();
    let wrong = wrong_cases(
        &engine,
        &[(SQUARE_WITH_A_HOLE, vec!["on_vertex", "outside"], 1.0)],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: every hit scores the boost, which a group's `^` sets.
#[tokio::test(flavor = "multi_thread")]
async fn a_polygon_scores_its_boost() {
    let engine = engine().await;
    let boosted = format!("({SQUARE_WITH_A_HOLE})^2.5");
    let wrong = wrong_cases(
        &engine,
        &[(
            boosted.as_str(),
            vec!["inside", "on_edge", "on_vertex"],
            2.5,
        )],
    )
    .await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1596: an odd count of numbers, a ring of fewer than 3 points and
/// a polygon with no field fail to parse, each saying why.
#[tokio::test(flavor = "multi_thread")]
async fn an_invalid_polygon_in_the_dsl_is_rejected() {
    let engine = engine().await;
    let mut wrong = Vec::new();
    for (dsl, reason) in [
        (
            "loc:geo_polygon((35, 139, 35, 140, 36))",
            "pairs of latitude and longitude",
        ),
        (
            "loc:geo_polygon((35, 139, 35, 140))",
            "at least 3 distinct points",
        ),
        ("loc:geo_polygon((35, 139, 35, 140, 95, 140))", "latitude"),
        (
            "geo_polygon((35, 139, 35, 140, 36, 140))",
            "explicit field prefix",
        ),
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
