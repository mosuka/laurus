use laurus::Document;
use laurus::DynamicFieldPolicy;
use laurus::Engine;
use laurus::Result;
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::TextOption;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{FieldOption, LexicalSearchQuery, Schema};

#[tokio::test(flavor = "multi_thread")]
async fn test_schema_lexical_guardrails() -> Result<()> {
    // 1. Setup Storage
    let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Engine with specific schema.
    //    The test predates the dynamic-schema feature and asserts that a
    //    field absent from the schema is neither stored nor indexed, so we
    //    pin the policy to `Ignore` to preserve those semantics. The default
    //    is now `Dynamic`, which would instead auto-add `unknown_field`.
    let config = Schema::builder()
        .dynamic_field_policy(DynamicFieldPolicy::Ignore)
        .add_field(
            "indexed_and_stored",
            FieldOption::Text(TextOption {
                indexed: true,
                stored: true,
                ..Default::default()
            }),
        )
        .add_field(
            "indexed_only",
            FieldOption::Text(TextOption {
                indexed: true,
                stored: false,
                ..Default::default()
            }),
        )
        .add_field(
            "stored_only",
            FieldOption::Text(TextOption {
                indexed: false,
                stored: true,
                ..Default::default()
            }),
        )
        .build();

    let engine = Engine::new(storage, config).await?;

    // 3. Index a document with various fields (including one NOT in schema)
    let doc = Document::builder()
        .add_field("indexed_and_stored", "value1")
        .add_field("indexed_only", "value2")
        .add_field("stored_only", "value3")
        .add_field("unknown_field", "should be ignored")
        .build();

    engine.put_document("test1", doc).await?;
    engine.commit().await?;

    // 4. Verify Searching

    // Case A: Search "indexed_and_stored" -> Should find it
    let req_a = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "indexed_and_stored",
            "value1",
        ))))
        .build();
    let res_a = engine.search(req_a).await?;
    assert_eq!(res_a.len(), 1, "Should find 'indexed_and_stored' field");
    let docs_a = engine.get_documents(&res_a[0].id).await?;
    let doc_a = &docs_a[0];
    assert_eq!(
        doc_a
            .get_field("indexed_and_stored")
            .and_then(|v: &laurus::DataValue| v.as_text()),
        Some("value1")
    );

    // Case B: Search "indexed_only" -> Should find it but value should NOT be in get_documents results
    let req_b = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "indexed_only",
            "value2",
        ))))
        .build();
    let res_b = engine.search(req_b).await?;
    assert_eq!(res_b.len(), 1, "Should find 'indexed_only' field");
    let docs_b = engine.get_documents(&res_b[0].id).await?;
    let doc_b = &docs_b[0];
    assert!(
        doc_b.get_field("indexed_only").is_none(),
        "Field 'indexed_only' should NOT be stored"
    );

    // Case C: Search "stored_only" -> Should NOT find it
    let req_c = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "stored_only",
            "value3",
        ))))
        .build();
    let res_c = engine.search(req_c).await?;
    assert_eq!(
        res_c.len(),
        0,
        "Should NOT find 'stored_only' field via search"
    );
    // But it should be present in retrieval if we get by ID
    let docs_c = engine.get_documents(&res_a[0].id).await?; // use ID from earlier
    let doc_c = &docs_c[0];
    assert_eq!(
        doc_c
            .get_field("stored_only")
            .and_then(|v: &laurus::DataValue| v.as_text()),
        Some("value3"),
        "Field 'stored_only' should be stored"
    );

    // Case D: "unknown_field" is not in the schema, so a lexical query on
    // it is rejected (Issue #1635) instead of returning no hits.
    let req_d = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "unknown_field",
            "should",
        ))))
        .build();
    let err_d = engine
        .search(req_d)
        .await
        .expect_err("a lexical query on a field not in the schema must fail");
    assert!(
        matches!(&err_d, laurus::LaurusError::InvalidArgument(m) if m.contains("unknown_field")),
        "{err_d}"
    );
    let docs_d = engine.get_documents(&res_a[0].id).await?;
    let doc_d = &docs_d[0];
    assert!(
        doc_d.get_field("unknown_field").is_none(),
        "Field 'unknown_field' should NOT be stored because it's not in schema"
    );

    Ok(())
}
