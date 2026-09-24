//! End-to-end tests for the dynamic schema feature.
//!
//! These tests drive the full `Engine` surface (put → commit → search /
//! get_documents) for each [`DynamicFieldPolicy`] variant, and also cover
//! the type-conflict coercion rules exercised during document ingestion.

use laurus::lexical::TextOption;
use laurus::lexical::core::field::{BooleanOption, DateTimeOption, GeoOption, IntegerOption};
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{
    DataValue, Document, DynamicFieldPolicy, Engine, FieldOption, GeoEcefPoint, GeoPoint,
    LaurusError, Result, Schema,
};

async fn engine_with_policy(policy: DynamicFieldPolicy) -> Result<Engine> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder().dynamic_field_policy(policy).build();
    Engine::new(storage, schema).await
}

/// Strict: an undeclared field must cause the ingest to fail.
#[tokio::test(flavor = "multi_thread")]
async fn strict_rejects_undeclared_fields() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Strict).await?;

    let doc = Document::builder().add_field("title", "hello").build();

    let err = engine
        .put_document("doc1", doc)
        .await
        .expect_err("Strict policy must reject undeclared field 'title'");
    let msg = err.to_string();
    assert!(
        msg.contains("title"),
        "error should mention the field: {msg}"
    );
    assert!(
        msg.contains("Strict") || msg.contains("undeclared"),
        "error should explain the policy: {msg}"
    );
    Ok(())
}

/// Dynamic: undeclared text/integer/float/bool are auto-added.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_auto_adds_primitive_fields() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let doc = Document::builder()
        .add_field("title", "hello world")
        .add_field("count", 42i64)
        .add_field("rating", 4.5f64)
        .add_field("published", true)
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    assert!(
        matches!(schema.fields.get("title"), Some(FieldOption::Text(_))),
        "title should be auto-added as Text"
    );
    assert!(
        matches!(schema.fields.get("count"), Some(FieldOption::Integer(_))),
        "count should be auto-added as Integer"
    );
    assert!(
        matches!(schema.fields.get("rating"), Some(FieldOption::Float(_))),
        "rating should be auto-added as Float"
    );
    assert!(
        matches!(
            schema.fields.get("published"),
            Some(FieldOption::Boolean(_))
        ),
        "published should be auto-added as Boolean"
    );

    // Values should be retrievable.
    let docs = engine.get_documents("doc1").await?;
    assert_eq!(docs.len(), 1);
    let d = &docs[0];
    assert_eq!(
        d.get("title").and_then(|v| v.as_text()),
        Some("hello world")
    );
    assert_eq!(d.get("count").and_then(|v| v.as_integer()), Some(42));
    assert_eq!(d.get("rating").and_then(|v| v.as_float()), Some(4.5));
    assert_eq!(d.get("published").and_then(|v| v.as_boolean()), Some(true));
    Ok(())
}

/// Dynamic: a geo value (DataValue::Geo) is auto-added as a Geo field.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_auto_adds_geo_field() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let doc = Document::builder().add_geo("location", 35.1, 139.0).build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    assert!(matches!(
        schema.fields.get("location"),
        Some(FieldOption::Geo(_))
    ));

    let docs = engine.get_documents("doc1").await?;
    let geo = docs[0].get("location").and_then(|v| v.as_geo()).unwrap();
    assert_eq!(geo.lat, 35.1);
    assert_eq!(geo.lon, 139.0);
    Ok(())
}

/// Dynamic: a numeric array on an undeclared field is auto-added as a
/// multi-valued numeric field (not a vector field).
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_auto_adds_int64_array_field() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let doc = Document::builder()
        .add_int64_array("scores", vec![85, 72, 95])
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    match schema.fields.get("scores") {
        Some(FieldOption::Integer(opt)) => assert!(
            opt.multi_valued,
            "scores should be Integer with multi_valued=true"
        ),
        other => panic!("expected Integer field for 'scores', got {other:?}"),
    }

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("scores").and_then(|v| v.as_int64_array()),
        Some(&[85, 72, 95][..])
    );
    Ok(())
}

/// Dynamic (#1174): a geo point array on an undeclared field is auto-added
/// as a multi-valued Geo field, and the array reads back intact.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_auto_adds_geo_array_field() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let points = vec![GeoPoint::new(35.1, 139.0), GeoPoint::new(-33.9, 151.2)];
    let doc = Document::builder()
        .add_geo_array("locations", points.clone())
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    match schema.fields.get("locations") {
        Some(FieldOption::Geo(opt)) => assert!(
            opt.multi_valued,
            "locations should be Geo with multi_valued=true"
        ),
        other => panic!("expected Geo field for 'locations', got {other:?}"),
    }

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("locations").and_then(|v| v.as_geo_array()),
        Some(points.as_slice())
    );
    Ok(())
}

/// Dynamic (#1174): an ECEF point array is auto-added as a multi-valued
/// Geo3d field.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_auto_adds_geo_ecef_array_field() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let points = vec![
        GeoEcefPoint::new(1.0, 2.0, 3.0),
        GeoEcefPoint::new(-4.0, 5.0, -6.0),
    ];
    let doc = Document::builder()
        .add_geo_ecef_array("positions", points.clone())
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    match schema.fields.get("positions") {
        Some(FieldOption::Geo3d(opt)) => assert!(
            opt.multi_valued,
            "positions should be Geo3d with multi_valued=true"
        ),
        other => panic!("expected Geo3d field for 'positions', got {other:?}"),
    }

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("positions").and_then(|v| v.as_geo_ecef_array()),
        Some(points.as_slice())
    );
    Ok(())
}

/// #1174: a declared single-valued Geo field rejects an array instead of
/// silently truncating it, and a declared multi-valued Geo field wraps a
/// single point into a one-element array.
#[tokio::test(flavor = "multi_thread")]
async fn geo_multi_valued_coercion_at_ingest() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("single", FieldOption::Geo(GeoOption::default()))
        .add_field(
            "multi",
            FieldOption::Geo(GeoOption {
                multi_valued: true,
                ..Default::default()
            }),
        )
        .dynamic_field_policy(DynamicFieldPolicy::Strict)
        .build();
    let engine = Engine::new(storage, schema).await?;

    let err = engine
        .put_document(
            "bad",
            Document::builder()
                .add_geo_array("single", vec![GeoPoint::new(35.1, 139.0)])
                .build(),
        )
        .await
        .expect_err("array into a single-valued Geo field must be rejected");
    assert!(
        err.to_string().contains("multi_valued = true"),
        "error should point at the fix: {err}"
    );

    engine
        .put_document(
            "ok",
            Document::builder().add_geo("multi", 35.1, 139.0).build(),
        )
        .await?;
    engine.commit().await?;
    let docs = engine.get_documents("ok").await?;
    assert_eq!(
        docs[0].get("multi").and_then(|v| v.as_geo_array()),
        Some(&[GeoPoint::new(35.1, 139.0)][..]),
        "a single point is auto-wrapped on a multi-valued field"
    );
    Ok(())
}

/// Dynamic (#1184): a datetime array on an undeclared field is auto-added
/// as a multi-valued DateTime field, and the array reads back intact.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_auto_adds_datetime_array_field() -> Result<()> {
    use chrono::TimeZone;
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let instants = vec![
        chrono::Utc.timestamp_opt(1_700_000_000, 500_000).unwrap(),
        chrono::Utc.timestamp_opt(1_700_003_600, 0).unwrap(),
    ];
    let doc = Document::builder()
        .add_datetime_array("times", instants.clone())
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    match schema.fields.get("times") {
        Some(FieldOption::DateTime(opt)) => assert!(
            opt.multi_valued,
            "times should be DateTime with multi_valued=true"
        ),
        other => panic!("expected DateTime field for 'times', got {other:?}"),
    }

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("times").and_then(|v| v.as_datetime_array()),
        Some(instants.as_slice())
    );
    Ok(())
}

/// #1184: a declared single-valued DateTime field rejects an array instead
/// of silently truncating it; a declared multi-valued DateTime field wraps
/// a single instant — typed or RFC 3339 text — into a one-element array.
#[tokio::test(flavor = "multi_thread")]
async fn datetime_multi_valued_coercion_at_ingest() -> Result<()> {
    use chrono::TimeZone;
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("single", FieldOption::DateTime(DateTimeOption::default()))
        .add_field(
            "multi",
            FieldOption::DateTime(DateTimeOption {
                multi_valued: true,
                ..Default::default()
            }),
        )
        .dynamic_field_policy(DynamicFieldPolicy::Strict)
        .build();
    let engine = Engine::new(storage, schema).await?;

    let instant = chrono::Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let err = engine
        .put_document(
            "bad",
            Document::builder()
                .add_datetime_array("single", vec![instant])
                .build(),
        )
        .await
        .expect_err("array into a single-valued DateTime field must be rejected");
    assert!(
        err.to_string().contains("multi_valued = true"),
        "error should point at the fix: {err}"
    );

    engine
        .put_document(
            "typed",
            Document::builder()
                .add_field("multi", DataValue::DateTime(instant))
                .build(),
        )
        .await?;
    engine
        .put_document(
            "text",
            Document::builder()
                .add_field("multi", DataValue::Text("2023-11-14T22:13:20Z".to_string()))
                .build(),
        )
        .await?;
    engine.commit().await?;
    for id in ["typed", "text"] {
        let docs = engine.get_documents(id).await?;
        assert_eq!(
            docs[0].get("multi").and_then(|v| v.as_datetime_array()),
            Some(&[instant][..]),
            "{id}: a single instant is auto-wrapped on a multi-valued field"
        );
    }
    Ok(())
}

/// Dynamic (#1180): a boolean array on an undeclared field is auto-added
/// as a multi-valued Boolean field, and the array reads back intact.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_auto_adds_bool_array_field() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let doc = Document::builder()
        .add_bool_array("flags", vec![true, false])
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    match schema.fields.get("flags") {
        Some(FieldOption::Boolean(opt)) => assert!(
            opt.multi_valued,
            "flags should be Boolean with multi_valued=true"
        ),
        other => panic!("expected Boolean field for 'flags', got {other:?}"),
    }

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("flags").and_then(|v| v.as_bool_array()),
        Some(&[true, false][..])
    );
    Ok(())
}

/// #1180: a declared single-valued Boolean field rejects an array instead
/// of silently truncating it; a declared multi-valued Boolean field wraps a
/// single value (typed, `0`/`1`, or text) into a one-element array and
/// widens an integer array of `0`/`1` element-wise.
#[tokio::test(flavor = "multi_thread")]
async fn boolean_multi_valued_coercion_at_ingest() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("single", FieldOption::Boolean(BooleanOption::default()))
        .add_field(
            "multi",
            FieldOption::Boolean(BooleanOption {
                multi_valued: true,
                ..Default::default()
            }),
        )
        .dynamic_field_policy(DynamicFieldPolicy::Strict)
        .build();
    let engine = Engine::new(storage, schema).await?;

    let err = engine
        .put_document(
            "bad",
            Document::builder()
                .add_bool_array("single", vec![true])
                .build(),
        )
        .await
        .expect_err("array into a single-valued Boolean field must be rejected");
    assert!(
        err.to_string().contains("multi_valued = true"),
        "error should point at the fix: {err}"
    );

    for (id, value) in [
        ("typed", DataValue::Bool(true)),
        ("int", DataValue::Int64(1)),
        ("text", DataValue::Text("true".to_string())),
    ] {
        engine
            .put_document(id, Document::builder().add_field("multi", value).build())
            .await?;
    }
    engine
        .put_document(
            "ints",
            Document::builder()
                .add_int64_array("multi", vec![0, 1])
                .build(),
        )
        .await?;
    engine
        .put_document(
            "empty",
            Document::builder()
                .add_int64_array("multi", Vec::new())
                .build(),
        )
        .await?;
    engine.commit().await?;

    for id in ["typed", "int", "text"] {
        let docs = engine.get_documents(id).await?;
        assert_eq!(
            docs[0].get("multi").and_then(|v| v.as_bool_array()),
            Some(&[true][..]),
            "{id}: a single value is auto-wrapped on a multi-valued field"
        );
    }
    assert_eq!(
        engine.get_documents("ints").await?[0]
            .get("multi")
            .and_then(|v| v.as_bool_array()),
        Some(&[false, true][..]),
        "an integer array of 0/1 is widened element-wise"
    );
    assert_eq!(
        engine.get_documents("empty").await?[0]
            .get("multi")
            .and_then(|v| v.as_bool_array()),
        Some(&[][..]),
        "the empty numeric array every binding sends for [] is an empty flag list"
    );
    Ok(())
}

/// Dynamic: a raw vector without a declared schema is rejected.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_rejects_undeclared_vector() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let doc = Document::builder()
        .add_vector("embedding", vec![0.1, 0.2, 0.3])
        .build();
    let err = engine
        .put_document("doc1", doc)
        .await
        .expect_err("vector fields must be declared explicitly");
    assert!(
        err.to_string().contains("vector"),
        "unexpected error: {err}"
    );
    Ok(())
}

/// Ignore: undeclared fields are silently dropped.
#[tokio::test(flavor = "multi_thread")]
async fn ignore_drops_undeclared_fields() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Ignore).await?;

    // Declare one field so we can verify declared fields still ingest.
    engine
        .add_field("title", FieldOption::Text(TextOption::default()))
        .await?;

    let doc = Document::builder()
        .add_field("title", "hello")
        .add_field("drop_me", "gone")
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    assert!(schema.fields.contains_key("title"));
    assert!(
        !schema.fields.contains_key("drop_me"),
        "Ignore should not add the field to the schema"
    );

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("title").and_then(|v| v.as_text()),
        Some("hello")
    );
    assert!(docs[0].get("drop_me").is_none());
    Ok(())
}

/// Integer fields truncate incoming float values (documented data loss).
#[tokio::test(flavor = "multi_thread")]
async fn integer_field_truncates_float() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("count", FieldOption::Integer(IntegerOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await?;

    let doc = Document::builder().add_field("count", 4.7f64).build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("count").and_then(|v| v.as_integer()),
        Some(4),
        "integer field must truncate incoming float"
    );
    Ok(())
}

/// Integer field parses numeric strings, and rejects non-numeric strings.
#[tokio::test(flavor = "multi_thread")]
async fn integer_field_parses_numeric_string() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("count", FieldOption::Integer(IntegerOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await?;

    let doc = Document::builder()
        .add_field("count", DataValue::Text("42".to_string()))
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(docs[0].get("count").and_then(|v| v.as_integer()), Some(42));

    let doc_bad = Document::builder()
        .add_field("count", DataValue::Text("abc".to_string()))
        .build();
    let err = engine.put_document("doc2", doc_bad).await.unwrap_err();
    assert!(
        err.to_string().contains("parse") || err.to_string().contains("integer"),
        "{err}"
    );
    Ok(())
}

/// Parsing a query that names a field outside the schema must fail.
#[tokio::test(flavor = "multi_thread")]
async fn query_dsl_rejects_unknown_field() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await?;

    let parser = engine.unified_query_parser()?;

    // Declared field parses fine.
    parser.parse("title:hello").await?;

    // Typo'd / undeclared field is rejected with a message that names it.
    let result = parser.parse("titl:hello").await;
    let err = match result {
        Ok(_) => panic!("expected error for unknown field 'titl'"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("titl"), "{err}");

    Ok(())
}

/// `_id` is a reserved field injected by the engine at ingest time — it is
/// never present in `schema.fields`, so `known_fields` must special-case
/// it. Without that, `_id:doc-001` (previously handled by the CLI's own
/// `StandardAnalyzer`-only parser, which never validated field names) is
/// rejected as an "unknown field" once the CLI moves onto this DSL path.
#[tokio::test(flavor = "multi_thread")]
async fn query_dsl_accepts_the_reserved_id_field() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await?;

    let doc = Document::builder().add_field("title", "hello").build();
    engine.put_document("doc-001", doc).await?;
    engine.commit().await?;

    let parser = engine.unified_query_parser()?;
    let request = parser.parse("_id:doc-001").await?;
    let results = engine.search(request).await?;

    assert_eq!(
        results.len(),
        1,
        "_id must be queryable without being rejected as an unknown field"
    );

    Ok(())
}

/// The `_id` carve-out must not open the door to other underscore-prefixed
/// field names; only the exact reserved name is allowed.
#[tokio::test(flavor = "multi_thread")]
async fn query_dsl_still_rejects_other_underscore_fields() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .build();
    let engine = Engine::new(storage, schema).await?;

    let parser = engine.unified_query_parser()?;
    let err = match parser.parse("_secret:x").await {
        Ok(_) => panic!("undeclared underscore field must still be rejected"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("_secret"), "{err}");

    Ok(())
}

/// User-supplied `_`-prefixed field names are rejected under any policy.
#[tokio::test(flavor = "multi_thread")]
async fn reserved_prefix_rejected() -> Result<()> {
    for policy in [
        DynamicFieldPolicy::Strict,
        DynamicFieldPolicy::Dynamic,
        DynamicFieldPolicy::Ignore,
    ] {
        let engine = engine_with_policy(policy).await?;
        let doc = Document::builder().add_field("_secret", "nope").build();
        let err = engine.put_document("doc1", doc).await.unwrap_err();
        assert!(
            matches!(err, LaurusError::Other(_)) || err.to_string().contains("reserved"),
            "policy {:?}: unexpected error {err}",
            policy
        );
    }
    Ok(())
}

/// Dynamic (#1175): a text array on an undeclared field is auto-added as a
/// multi-valued Text field, and the array reads back intact.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_auto_adds_text_array_field() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let values = vec!["hello world".to_string(), "foo bar".to_string()];
    let doc = Document::builder()
        .add_text_array("notes", values.clone())
        .build();
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    match schema.fields.get("notes") {
        Some(FieldOption::Text(opt)) => {
            assert!(
                opt.multi_valued,
                "notes should be Text with multi_valued=true"
            );
            assert_eq!(
                opt.position_increment_gap,
                laurus::lexical::core::field::DEFAULT_POSITION_INCREMENT_GAP
            );
        }
        other => panic!("expected Text field for 'notes', got {other:?}"),
    }

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0].get("notes").and_then(|v| v.as_text_array()),
        Some(values.as_slice())
    );
    Ok(())
}

/// #1175: the JSON path still infers a multi-valued DateTime from an array
/// whose elements are all RFC 3339 (so the round trip through RFC 3339
/// strings keeps working), and a multi-valued Text from any other string
/// array — which used to be an error.
#[tokio::test(flavor = "multi_thread")]
async fn dynamic_string_array_infers_datetime_when_all_rfc3339_else_text() -> Result<()> {
    let engine = engine_with_policy(DynamicFieldPolicy::Dynamic).await?;

    let doc = laurus::json_to_document(&serde_json::json!({
        "fields": {
            "times": ["2024-01-01T00:00:00Z", "2024-06-15T21:00:00+09:00"],
            "links": ["https://example.com/a", "https://example.com/b"],
        }
    }))?;
    engine.put_document("doc1", doc).await?;
    engine.commit().await?;

    let schema = engine.schema();
    assert!(
        matches!(schema.fields.get("times"), Some(FieldOption::DateTime(o)) if o.multi_valued),
        "{:?}",
        schema.fields.get("times")
    );
    assert!(
        matches!(schema.fields.get("links"), Some(FieldOption::Text(o)) if o.multi_valued),
        "{:?}",
        schema.fields.get("links")
    );

    let docs = engine.get_documents("doc1").await?;
    assert_eq!(
        docs[0]
            .get("times")
            .and_then(|v| v.as_datetime_array())
            .map(|a| a.len()),
        Some(2)
    );
    assert_eq!(
        docs[0].get("links").and_then(|v| v.as_text_array()),
        Some(
            &[
                "https://example.com/a".to_string(),
                "https://example.com/b".to_string()
            ][..]
        )
    );
    Ok(())
}

/// #1175: a declared single-valued Text field rejects an array instead of
/// silently joining or truncating it; a declared multi-valued Text field
/// wraps a scalar, stringifies other arrays element-wise, and treats `Null`
/// as an empty list.
#[tokio::test(flavor = "multi_thread")]
async fn text_multi_valued_coercion_at_ingest() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema = Schema::builder()
        .add_field("single", FieldOption::Text(TextOption::default()))
        .add_field(
            "multi",
            FieldOption::Text(TextOption {
                multi_valued: true,
                ..Default::default()
            }),
        )
        .dynamic_field_policy(DynamicFieldPolicy::Strict)
        .build();
    let engine = Engine::new(storage, schema).await?;

    let err = engine
        .put_document(
            "bad",
            Document::builder()
                .add_text_array("single", vec!["a".into()])
                .build(),
        )
        .await
        .expect_err("array into a single-valued Text field must be rejected");
    assert!(
        err.to_string().contains("multi_valued = true"),
        "error should point at the fix: {err}"
    );

    for (id, value) in [
        ("text", DataValue::Text("hello".to_string())),
        ("int", DataValue::Int64(7)),
        ("ints", DataValue::Int64Array(vec![1, 2])),
        ("null", DataValue::Null),
        ("empty", DataValue::Int64Array(Vec::new())),
    ] {
        engine
            .put_document(id, Document::builder().add_field("multi", value).build())
            .await?;
    }
    engine.commit().await?;

    let get = |id: &str| {
        let engine = &engine;
        let id = id.to_string();
        async move {
            let docs = engine.get_documents(&id).await?;
            Ok::<Vec<String>, LaurusError>(
                docs[0]
                    .get("multi")
                    .and_then(|v| v.as_text_array())
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| panic!("{id}: expected a TextArray, got {:?}", docs[0])),
            )
        }
    };
    assert_eq!(get("text").await?, vec!["hello".to_string()]);
    assert_eq!(get("int").await?, vec!["7".to_string()]);
    assert_eq!(get("ints").await?, vec!["1".to_string(), "2".to_string()]);
    assert_eq!(get("null").await?, Vec::<String>::new());
    assert_eq!(get("empty").await?, Vec::<String>::new());
    Ok(())
}
