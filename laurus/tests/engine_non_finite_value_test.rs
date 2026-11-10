//! Issue #1604: a `NaN` or infinite field value is rejected with
//! `InvalidArgument` when the document is put, before it reaches the WAL.
//!
//! `put_document` used to accept one. A `NaN` in an indexed field then
//! failed the commit, and every later commit too, because the BKD writer
//! cannot hold a `NaN` point; an infinite value, or a `NaN` in a stored-only
//! field, committed but was written as JSON `null`, so every search that
//! returned the document failed. The values came from the Rust API, the
//! bindings, and JSON strings such as `"nan"`, `"inf"` and `"1e400"`, which
//! the `Float` coercion parses; an `Integer` field turned them into 0 or
//! `i64::MAX`.

use std::sync::Arc;

use laurus::lexical::{Query, TermQuery};
use laurus::storage::memory::MemoryStorage;
use laurus::{
    DataValue, Document, DynamicFieldPolicy, Engine, FloatOption, Geo3dOption, GeoEcefPoint,
    GeoOption, GeoPoint, IntegerOption, LaurusError, LexicalSearchQuery, Schema,
    SearchRequestBuilder, TextOption,
};

/// A schema with a field of each kind that holds floats.
fn schema(policy: DynamicFieldPolicy) -> Schema {
    Schema::builder()
        .add_text_field("name", TextOption::default())
        .add_float_field("f", FloatOption::default())
        .add_float_field("f_s", FloatOption::default().indexed(false))
        .add_float_field(
            "fa",
            FloatOption {
                indexed: true,
                stored: true,
                multi_valued: true,
                doc_values: true,
            },
        )
        .add_integer_field("i", IntegerOption::default())
        .add_geo_field("g", GeoOption::default())
        .add_geo3d_field("g3", Geo3dOption::default())
        .dynamic_field_policy(policy)
        .build()
}

async fn engine(policy: DynamicFieldPolicy) -> Engine {
    Engine::new(
        Arc::new(MemoryStorage::new(Default::default())),
        schema(policy),
    )
    .await
    .unwrap()
}

/// The ids a search for `name:<term>` finds.
async fn find(engine: &Engine, term: &str) -> Result<Vec<String>, String> {
    let query: Box<dyn Query> = Box::new(TermQuery::new("name", term));
    engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(query))
                .build(),
        )
        .await
        .map(|results| results.into_iter().map(|r| r.id).collect())
        .map_err(|e| e.to_string())
}

/// `(name, field, value)`: every route a non-finite number took.
fn non_finite_values() -> Vec<(&'static str, &'static str, DataValue)> {
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    vec![
        ("NaN in an indexed Float", "f", DataValue::Float64(nan)),
        ("inf in an indexed Float", "f", DataValue::Float64(inf)),
        ("NaN in a stored-only Float", "f_s", DataValue::Float64(nan)),
        ("text \"nan\"", "f", DataValue::Text("nan".into())),
        ("text \"inf\"", "f_s", DataValue::Text("inf".into())),
        ("text \"1e400\"", "f_s", DataValue::Text("1e400".into())),
        (
            "NaN in a multi-valued Float",
            "fa",
            DataValue::Float64Array(vec![1.0, nan]),
        ),
        ("NaN into an Integer", "i", DataValue::Float64(nan)),
        ("inf into an Integer", "i", DataValue::Float64(inf)),
        (
            "NaN latitude",
            "g",
            DataValue::Geo(GeoPoint { lat: nan, lon: 0.0 }),
        ),
        (
            "NaN Geo3d x",
            "g3",
            DataValue::GeoEcef(GeoEcefPoint::new(nan, 0.0, 0.0)),
        ),
        ("NaN in an undeclared field", "u", DataValue::Float64(nan)),
    ]
}

/// Issue #1604: each route is rejected at put with `InvalidArgument`
/// naming the field, nothing reaches the WAL, and the engine keeps
/// committing: a valid document put afterwards is found.
#[tokio::test(flavor = "multi_thread")]
async fn a_non_finite_value_is_rejected_at_put() {
    let mut wrong = Vec::new();
    for (name, field, value) in non_finite_values() {
        let engine = engine(DynamicFieldPolicy::Dynamic).await;
        let bad = Document::builder()
            .add_text("name", "bad")
            .add_field(field, value)
            .build();
        let put = engine.put_document("bad", bad).await;
        if !matches!(
            &put,
            Err(LaurusError::InvalidArgument(m))
                if m.contains(&format!("field '{field}'")) && m.contains("must be finite")
        ) {
            wrong.push(format!("{name}: put {put:?}"));
            continue;
        }

        let good = Document::builder()
            .add_text("name", "good")
            .add_float("f", 1.0)
            .build();
        engine.put_document("good", good).await.unwrap();
        let commit = engine.commit().await;
        let good_found = find(&engine, "good").await;
        let bad_found = find(&engine, "bad").await;
        if commit.is_err() || good_found != Ok(vec!["good".to_string()]) || bad_found != Ok(vec![])
        {
            wrong.push(format!(
                "{name}: commit {commit:?}, good {good_found:?}, bad {bad_found:?}"
            ));
        }
        if engine.schema().fields.contains_key("u") {
            wrong.push(format!(
                "{name}: the undeclared field was added to the schema"
            ));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Issue #1604: under `DynamicFieldPolicy::Ignore` a value that fails
/// coercion drops the field, as `"abc"` in a `Float` field does, so a
/// non-finite value drops its field and the rest of the document is put.
#[tokio::test(flavor = "multi_thread")]
async fn a_non_finite_value_drops_its_field_under_the_ignore_policy() {
    let engine = engine(DynamicFieldPolicy::Ignore).await;
    let doc = Document::builder()
        .add_text("name", "kept")
        .add_float("f", f64::NAN)
        .build();
    engine.put_document("kept", doc).await.unwrap();
    engine.commit().await.unwrap();
    assert_eq!(find(&engine, "kept").await, Ok(vec!["kept".to_string()]));
    let stored = engine.get_documents("kept").await.unwrap();
    assert_eq!(stored.len(), 1);
    assert!(
        !stored[0].fields.contains_key("f"),
        "the NaN field must be dropped: {:?}",
        stored[0].fields
    );
}
