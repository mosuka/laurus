//! Regression tests for GitHub issue #982: boolean conjunctions that
//! contain a geo clause (`+(term) +location:geo_bbox(...)`) dropped hits
//! because `GeoMatcher` iterated in distance order while `skip_to`
//! assumed ascending doc-id order. Mirrors the WASM `Index.search` path
//! (DSL string via `SearchRequestBuilder::query_dsl`) and the geo demo
//! schema shape (Japanese-analyzed default fields + a Geo field).

use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{DataValue, Document, Engine, GeoPoint, Result, Schema, SearchRequestBuilder};

const GEO_DEMO_SCHEMA_JSON: &str = r#"
{
    "default_fields": ["title", "description", "category"],
    "fields": {
        "title": { "Text": {
            "indexed": true, "stored": true, "term_vectors": false,
            "analyzer": { "language": "japanese", "mode": "normal", "dict": "embedded://ipadic" }
        }},
        "description": { "Text": {
            "indexed": true, "stored": true, "term_vectors": false,
            "analyzer": { "language": "japanese", "mode": "normal", "dict": "embedded://ipadic" }
        }},
        "category": { "Text": {
            "indexed": true, "stored": true, "term_vectors": false,
            "analyzer": { "language": "japanese", "mode": "normal", "dict": "embedded://ipadic" }
        }},
        "location": { "Geo": { "indexed": true, "stored": true } }
    }
}
"#;

async fn geo_demo_engine() -> Result<Engine> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema: Schema = serde_json::from_str(GEO_DEMO_SCHEMA_JSON).expect("valid schema JSON");
    let engine = Engine::new(storage, schema).await?;

    engine
        .put_document(
            "ueno",
            Document::builder()
                .add_field("title", "上野恩賜公園")
                .add_field("description", "美術館や動物園が集まる広大な公園。")
                .add_field("category", "公園")
                .add_field("location", DataValue::Geo(GeoPoint::new(35.712, 139.771)))
                .build(),
        )
        .await?;
    engine
        .put_document(
            "odaiba",
            Document::builder()
                .add_field("title", "お台場海浜公園")
                .add_field("description", "東京湾を一望できるベイエリアの公園。")
                .add_field("category", "公園")
                .add_field("location", DataValue::Geo(GeoPoint::new(35.630, 139.773)))
                .build(),
        )
        .await?;
    engine
        .put_document(
            "tower",
            Document::builder()
                .add_field("title", "東京タワー")
                .add_field("description", "芝公園にそびえる赤と白の電波塔。")
                .add_field("category", "展望")
                .add_field("location", DataValue::Geo(GeoPoint::new(35.658, 139.745)))
                .build(),
        )
        .await?;
    engine.commit().await?;
    Ok(engine)
}

async fn dsl_hits(engine: &Engine, dsl: &str) -> Result<Vec<String>> {
    let request = SearchRequestBuilder::new()
        .limit(10)
        .query_dsl(dsl.to_string())
        .build();
    let results = engine.search(request).await?.results;
    Ok(results.iter().map(|h| h.id.clone()).collect())
}

#[tokio::test(flavor = "multi_thread")]
async fn bbox_alone_matches_all_docs() -> Result<()> {
    let engine = geo_demo_engine().await?;
    let ids = dsl_hits(&engine, "location:geo_bbox(35.6, 139.5, 35.8, 140.0)").await?;
    assert_eq!(ids.len(), 3, "bbox covers all three docs, got {ids:?}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn bare_japanese_term_matches() -> Result<()> {
    let engine = geo_demo_engine().await?;
    let ids = dsl_hits(&engine, "公園").await?;
    assert!(
        ids.len() >= 2,
        "公園 appears in all three docs' default fields, got {ids:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn required_group_japanese_term_matches() -> Result<()> {
    let engine = geo_demo_engine().await?;
    let ids = dsl_hits(&engine, "+(公園)").await?;
    assert!(
        ids.len() >= 2,
        "+(公園) should match like the bare term, got {ids:?}"
    );
    Ok(())
}

/// The exact query shape the geo demo builds: required text group AND
/// required geo bounding-box clause.
#[tokio::test(flavor = "multi_thread")]
async fn required_group_and_bbox_conjunction_matches() -> Result<()> {
    let engine = geo_demo_engine().await?;
    let ids = dsl_hits(
        &engine,
        "+(公園) +location:geo_bbox(35.6, 139.5, 35.8, 140.0)",
    )
    .await?;
    assert!(
        ids.len() >= 2,
        "conjunction of matching term and covering bbox must hit, got {ids:?}"
    );
    Ok(())
}
