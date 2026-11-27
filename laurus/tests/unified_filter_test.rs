use tempfile::TempDir;

use laurus::Document;
use laurus::Engine;
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::TextOption;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::Vector;
use laurus::{FieldOption, QueryVector, Schema, VectorSearchQuery};

#[tokio::test(flavor = "multi_thread")]
async fn test_unified_filtering() -> laurus::Result<()> {
    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Engines
    use laurus::vector::FlatOption;
    let vector_opt = FlatOption::default().dimension(2);
    let lexical_opt = TextOption::default();

    let config = Schema::builder()
        .add_field("name", FieldOption::Text(lexical_opt.clone()))
        .add_field("category", FieldOption::Text(lexical_opt))
        .add_field("embedding", FieldOption::Flat(vector_opt))
        .build();

    let engine = Engine::new(storage.clone(), config).await?;

    // 3. Index Documents
    // Doc 1: Apple (Fruit), Vector [1.0, 0.0]
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("name", "Apple")
                .add_field("category", "fruit")
                .add_field("embedding", vec![1.0, 0.0])
                .build(),
        )
        .await?;

    // Doc 2: Banana (Fruit), Vector [0.9, 0.1]
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("name", "Banana")
                .add_field("category", "fruit")
                .add_field("embedding", vec![0.9, 0.1])
                .build(),
        )
        .await?;

    // Doc 3: Carrot (Vegetable), Vector [1.0, 0.0] -> Identical vector to Apple!
    engine
        .put_document(
            "doc3",
            Document::builder()
                .add_field("name", "Carrot")
                .add_field("category", "vegetable")
                .add_field("embedding", vec![1.0, 0.0])
                .build(),
        )
        .await?;

    engine.commit().await?;

    // 4. Test Filtering: Search for [1.0, 0.0] (Apple/Carrot) but filter for "vegetable"
    let vector_query = VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0, 0.0]),
        weight: 1.0,
        fields: Some(vec!["embedding".into()]),
    }]);

    let filter_query = Box::new(TermQuery::new("category", "vegetable"));

    let req = SearchRequestBuilder::new()
        .vector_query(vector_query.clone())
        .filter_query(filter_query)
        .build();

    let results = engine.search(req).await?.results;

    // Should pass through Engine -> Lexical Filter -> Vector Store allowed_ids
    println!("Filtered Results (Vegetable): {:?}", results);

    assert_eq!(results.len(), 1, "Should filter down to 1 result");
    // Verify it's Carrot (the only vegetable)
    assert_eq!(
        results[0].id, "doc3",
        "Filtered result should be doc3 (Carrot, vegetable)"
    );

    // 5. Test Filtering: Search for "fruit"
    let filter_query_fruit = Box::new(TermQuery::new("category", "fruit"));
    let req_fruit = SearchRequestBuilder::new()
        .vector_query(vector_query)
        .filter_query(filter_query_fruit)
        .build();

    let results_fruit = engine.search(req_fruit).await?.results;
    println!("Filtered Results (Fruit): {:?}", results_fruit);
    assert_eq!(
        results_fruit.len(),
        2,
        "Should filter down to 2 results (Apple, Banana)"
    );
    // Verify both fruits are present
    let fruit_ids: Vec<&str> = results_fruit.iter().map(|r| r.id.as_str()).collect();
    assert!(fruit_ids.contains(&"doc1"), "Should contain doc1 (Apple)");
    assert!(fruit_ids.contains(&"doc2"), "Should contain doc2 (Banana)");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unified_filtering_hnsw() -> laurus::Result<()> {
    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Engines with HNSW
    use laurus::vector::HnswOption;
    let vector_opt = HnswOption::default().dimension(2);
    let lexical_opt = TextOption::default();

    let config = Schema::builder()
        .add_field("name", FieldOption::Text(lexical_opt.clone()))
        .add_field("category", FieldOption::Text(lexical_opt))
        .add_field("embedding", FieldOption::Hnsw(vector_opt))
        .build();

    let engine = Engine::new(storage.clone(), config).await?;

    // 3. Index Documents
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("name", "Apple")
                .add_field("category", "fruit")
                .add_field("embedding", vec![1.0, 0.0])
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("name", "Banana")
                .add_field("category", "fruit")
                .add_field("embedding", vec![0.9, 0.1])
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc3",
            Document::builder()
                .add_field("name", "Carrot")
                .add_field("category", "vegetable")
                .add_field("embedding", vec![1.0, 0.0])
                .build(),
        )
        .await?;
    engine.commit().await?;

    // 4. Test Filtering: Search for [1.0, 0.0] but filter for "vegetable"
    let vector_query = VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0, 0.0]),
        weight: 1.0,
        fields: Some(vec!["embedding".into()]),
    }]);

    let filter_query = Box::new(TermQuery::new("category", "vegetable"));
    let req = SearchRequestBuilder::new()
        .vector_query(vector_query)
        .filter_query(filter_query)
        .build();

    let results = engine.search(req).await?.results;
    println!("Filtered Results HNSW (Vegetable): {:?}", results);
    assert_eq!(results.len(), 1, "Should filter down to 1 result");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unified_filtering_ivf() -> laurus::Result<()> {
    // 1. Setup Storage
    let temp_dir = TempDir::new().unwrap();
    let storage_config = StorageConfig::File(FileStorageConfig::new(temp_dir.path()));
    let storage = StorageFactory::create(storage_config)?;

    // 2. Configure Engines with IVF
    use laurus::vector::IvfOption;
    // Note: IVF usually requires training or pre-defined centroids.
    // However, our mocked/simple implementation might work with small data or just use simple clustering (if implemented).
    // Or we rely on `IvfFieldReader` behavior validation specifically.
    // If standard IVF needs training, this test might fail if training isn't triggered.
    // Assuming implicit training or minimal setup for test.
    let vector_opt = IvfOption::default().dimension(2).n_clusters(1);
    let lexical_opt = TextOption::default();

    let config = Schema::builder()
        .add_field("name", FieldOption::Text(lexical_opt.clone()))
        .add_field("category", FieldOption::Text(lexical_opt))
        .add_field("embedding", FieldOption::Ivf(vector_opt))
        .build();

    let engine = Engine::new(storage.clone(), config).await?;

    // 3. Index Documents
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("name", "Apple")
                .add_field("category", "fruit")
                .add_field("embedding", vec![1.0, 0.0])
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("name", "Banana")
                .add_field("category", "fruit")
                .add_field("embedding", vec![0.9, 0.1])
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc3",
            Document::builder()
                .add_field("name", "Carrot")
                .add_field("category", "vegetable")
                .add_field("embedding", vec![1.0, 0.0])
                .build(),
        )
        .await?;
    engine.commit().await?;

    // 4. Test Filtering: Search for [1.0, 0.0] but filter for "vegetable"
    let vector_query = VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(vec![1.0, 0.0]),
        weight: 1.0,
        fields: Some(vec!["embedding".into()]),
    }]);

    let filter_query = Box::new(TermQuery::new("category", "vegetable"));
    let req = SearchRequestBuilder::new()
        .vector_query(vector_query)
        .filter_query(filter_query)
        .build();

    let results = engine.search(req).await?.results;
    println!("Filtered Results IVF (Vegetable): {:?}", results);
    assert_eq!(results.len(), 1, "Should filter down to 1 result");
    Ok(())
}
