use tempfile::TempDir;

use laurus::Document;
use laurus::Engine;
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::TextOption;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::HnswOption;
use laurus::vector::Vector;
use laurus::{FieldOption, LexicalSearchQuery, QueryVector, Schema, VectorSearchQuery};

#[tokio::test(flavor = "multi_thread")]
async fn test_engine_unified_deletion() -> laurus::Result<()> {
    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Engine
    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("embedding", FieldOption::Hnsw(HnswOption::default()))
        .build();

    let engine = Engine::new(storage.clone(), config).await?;

    // 3. Index Document with ID "doc1"
    let doc1 = Document::builder()
        .add_field("title", "Hello Laurus")
        .add_field("embedding", vec![0.1; 128])
        .build();

    engine.put_document("doc1", doc1).await?;
    engine.commit().await?;

    // 4. Verify it exists in both
    // Lexical check
    let req_lexical = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
            "title", "hello",
        ))))
        .build();
    let res_lexical = engine.search(req_lexical).await?.results;
    assert_eq!(res_lexical.len(), 1, "Should be found lexically");

    // Vector check
    let req_vector = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![0.1; 128]),
            weight: 1.0,
            fields: Some(vec!["embedding".into()]),
        }]))
        .build();
    let res_vector = engine.search(req_vector).await?.results;
    assert_eq!(res_vector.len(), 1, "Should be found via vector");

    // 5. Delete by ID
    engine.delete_documents("doc1").await?;
    engine.commit().await?;

    // 6. Verify it is GONE from both
    let res_lexical_after = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "title", "hello",
                ))))
                .build(),
        )
        .await?
        .results;
    assert_eq!(res_lexical_after.len(), 0, "Should be deleted lexically");

    let res_vector_after = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
                    vector: Vector::new(vec![0.1; 128]),
                    weight: 1.0,
                    fields: Some(vec!["embedding".into()]),
                }]))
                .build(),
        )
        .await?
        .results;
    assert_eq!(res_vector_after.len(), 0, "Should be deleted via vector");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_engine_upsert() -> laurus::Result<()> {
    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Engine
    use laurus::vector::FlatOption;

    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field(
            "embedding",
            FieldOption::Flat(FlatOption {
                dimension: 2,
                ..Default::default()
            }),
        )
        .build();

    let engine = Engine::new(storage.clone(), config).await?;

    // 3. Index Document with ID "doc1"
    let doc1 = Document::builder()
        .add_field("title", "Initial Version")
        .add_field("embedding", vec![1.0, 0.0])
        .build();

    engine.put_document("doc1", doc1).await?;
    engine.commit().await?;

    // 4. Verify initial version exists
    let res = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "title", "initial",
                ))))
                .build(),
        )
        .await?
        .results;
    assert_eq!(res.len(), 1);

    // 5. Index updated document with SAME ID "doc1"
    let doc1_v2 = Document::builder()
        .add_field("title", "Updated Version")
        .add_field("embedding", vec![0.0, 1.0])
        .build();

    engine.put_document("doc1", doc1_v2).await?;
    engine.commit().await?;

    // 6. Verify update
    // Old version lookup should fail
    let res_old = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "title", "initial",
                ))))
                .build(),
        )
        .await?
        .results;
    assert_eq!(res_old.len(), 0, "Old version should be replaced");

    // New version lookup should succeed
    let res_new = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "title", "updated",
                ))))
                .build(),
        )
        .await?
        .results;
    assert_eq!(res_new.len(), 1, "New version should be found");

    // Vector check for new vector
    let res_vec = engine
        .search(
            SearchRequestBuilder::new()
                .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
                    vector: Vector::new(vec![0.0, 1.0]),
                    weight: 1.0,
                    fields: Some(vec!["embedding".into()]),
                }]))
                .build(),
        )
        .await?
        .results;
    assert_eq!(res_vec.len(), 1, "New vector should be found");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_engine_delete_nonexistent() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    use laurus::vector::FlatOption;
    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field(
            "embedding",
            FieldOption::Flat(FlatOption {
                dimension: 2,
                ..Default::default()
            }),
        )
        .build();

    let engine = Engine::new(storage, config).await?;

    // Deleting a non-existent document should not error
    let result = engine.delete_documents("nonexistent_doc").await;
    assert!(
        result.is_ok(),
        "Deleting non-existent document should succeed silently"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_engine_double_delete() -> laurus::Result<()> {
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    use laurus::vector::FlatOption;
    let config = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field(
            "embedding",
            FieldOption::Flat(FlatOption {
                dimension: 2,
                ..Default::default()
            }),
        )
        .build();

    let engine = Engine::new(storage, config).await?;

    // Index a document
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "Hello")
                .add_field("embedding", vec![1.0, 0.0])
                .build(),
        )
        .await?;
    engine.commit().await?;

    // Delete it twice — second delete should succeed silently
    engine.delete_documents("doc1").await?;
    engine.commit().await?;

    let result = engine.delete_documents("doc1").await;
    assert!(result.is_ok(), "Double delete should succeed silently");
    engine.commit().await?;

    // Verify document is gone
    let res = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "title", "hello",
                ))))
                .build(),
        )
        .await?
        .results;
    assert_eq!(res.len(), 0, "Document should remain deleted");

    Ok(())
}
