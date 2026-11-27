use tempfile::TempDir;

use laurus::Engine;
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::TextOption;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::FlatOption;
use laurus::vector::Vector;
use laurus::{DataValue, Document};
use laurus::{FieldOption, LexicalSearchQuery, QueryVector, Schema, VectorSearchQuery};

#[tokio::test(flavor = "multi_thread")]
async fn test_unified_engine_indexing() -> laurus::Result<()> {
    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Engines
    // We define a schema with:
    // - "title": Lexical only
    // - "embedding": Vector only
    // - "description": Lexical only

    // Note: FlatOption default uses Cosine, dim=128
    let vector_opt = FlatOption::default();
    let lexical_opt = TextOption::default();

    let config = Schema::builder()
        .add_field("title", FieldOption::Text(lexical_opt.clone()))
        .add_field("embedding", FieldOption::Flat(vector_opt.clone()))
        .add_field("description", FieldOption::Text(lexical_opt))
        .build();

    // 3. Initialize Engine
    let engine = Engine::new(storage.clone(), config).await?;

    // 4. Index Documents
    let doc1 = Document::builder()
        .add_field("title", DataValue::Text("Rust Programming".into()))
        .add_field("description", DataValue::Text("A systems language".into()))
        .add_field("embedding", DataValue::Vector(vec![0.1; 128])) // Valid vector
        .build();

    let doc2 = Document::builder()
        .add_field("title", DataValue::Text("Vector Search".into()))
        .add_field(
            "description",
            DataValue::Text("Searching with vectors".into()),
        )
        .add_field("embedding", DataValue::Vector(vec![0.2; 128]))
        .build();

    engine.put_document("doc1", doc1).await?;
    engine.put_document("doc2", doc2).await?;

    // 5. Commit
    engine.commit().await?;

    // 6. Verify Files Exist in Storage
    // Lexical engine should create files in "lexical/"
    // Vector engine should create files in "vector/"
    let files = storage.list_files()?;
    println!("Created files: {:?}", files);

    let has_lexical = files.iter().any(|f| f.starts_with("lexical/"));
    let has_vector = files.iter().any(|f| f.starts_with("vector/"));

    assert!(has_lexical, "Should have created lexical index files");
    assert!(has_vector, "Should have created vector index files");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unified_engine_dimension_mismatch() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    let vector_opt = FlatOption::default().dimension(3);
    let config = Schema::builder()
        .add_field("embedding", FieldOption::Flat(vector_opt))
        .build();

    let engine = Engine::new(storage, config).await?;

    // Attempt to index a document with wrong vector dimension (2 instead of 3)
    let doc = Document::builder()
        .add_field("embedding", DataValue::Vector(vec![1.0, 0.0]))
        .build();

    let result = engine.put_document("doc1", doc).await;
    assert!(
        result.is_err(),
        "Should return error for dimension mismatch"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unified_engine_concurrent_reads() -> laurus::Result<()> {
    use std::sync::Arc;

    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    let vector_opt = FlatOption::default().dimension(3);
    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("embedding", FieldOption::Flat(vector_opt))
        .build();

    let engine = Arc::new(Engine::new(storage, config).await?);

    // Index documents
    for i in 0..10 {
        engine
            .put_document(
                &format!("doc{i}"),
                Document::builder()
                    .add_field("title", DataValue::Text(format!("Document {i}")))
                    .add_field("embedding", DataValue::Vector(vec![i as f32, 0.0, 0.0]))
                    .build(),
            )
            .await?;
    }
    engine.commit().await?;

    // Spawn concurrent search tasks
    let mut handles = Vec::new();
    for i in 0..5 {
        let engine = Arc::clone(&engine);
        handles.push(tokio::spawn(async move {
            // Lexical search
            let req = SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "title", "document",
                ))))
                .build();
            let results = engine.search(req).await.unwrap().results;
            assert!(
                !results.is_empty(),
                "Task {i}: lexical search should return results"
            );

            // Vector search
            let req = SearchRequestBuilder::new()
                .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
                    vector: Vector::new(vec![i as f32, 0.0, 0.0]),
                    weight: 1.0,
                    fields: Some(vec!["embedding".into()]),
                }]))
                .limit(3)
                .build();
            let results = engine.search(req).await.unwrap().results;
            assert!(
                !results.is_empty(),
                "Task {i}: vector search should return results"
            );
        }));
    }

    // Wait for all tasks to complete without errors
    for handle in handles {
        handle.await.unwrap();
    }

    Ok(())
}
