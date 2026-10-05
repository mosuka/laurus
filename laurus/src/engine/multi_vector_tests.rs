//! End-to-end tests of multi-vector fields through the [`Engine`]
//! (Issue #1177).
//!
//! These live inside the engine module so they can read the stored token
//! vectors through the engine's vector store, which has no public accessor.

use std::sync::Arc;

use serde_json::json;

use crate::data::{DataValue, Document};
use crate::engine::schema::{FieldOption, Schema};
use crate::engine::search::SearchRequestBuilder;
use crate::engine::{Engine, UpdateFieldOptions};
use crate::error::{LaurusError, Result};
use crate::lexical::core::field::TextOption;
use crate::lexical::query::term::TermQuery;
use crate::lexical::search::searcher::LexicalSearchRequest;
use crate::storage::Storage;
use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::field::{FlatOption, HnswOption, MultiVectorOption};
use crate::vector::core::vector::Vector;
use crate::vector::search::searcher::{VectorSearchParams, VectorSearchQuery, VectorSearchRequest};
use crate::vector::store::request::{FieldSelector, QueryVector};

const TOKENS: &str = "tokens";

fn storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
}

fn tokens_option() -> FieldOption {
    FieldOption::MultiVector(MultiVectorOption::new(2).distance(DistanceMetric::DotProduct))
}

/// `title` (Text), `vec` (Flat, 2-dim) and `tokens` (MultiVector, 2-dim):
/// the same dimension, so field-less vector search would reach `tokens` if
/// it were not excluded.
fn schema() -> Schema {
    Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_flat_field("vec", FlatOption::new(2))
        .add_field(TOKENS, tokens_option())
        .build()
}

fn doc(title: &str, vec: [f32; 2], tokens: &[[f32; 2]]) -> Document {
    Document::builder()
        .add_text("title", title)
        .add_vector("vec", vec.to_vec())
        .add_vector_array(TOKENS, tokens.iter().map(|t| t.to_vec()).collect())
        .build()
}

/// Internal ids of the live documents with external id `id`.
fn internal_ids(engine: &Engine, id: &str) -> Vec<u64> {
    engine.lexical.find_doc_ids_by_term("_id", id).unwrap()
}

/// The committed token vectors of the document with external id `id`.
fn stored_tokens(engine: &Engine, id: &str) -> Option<Vec<f32>> {
    let ids = internal_ids(engine, id);
    assert!(ids.len() <= 1, "expected at most one live copy of '{id}'");
    let snapshot = engine.vector.multi_vector_snapshot(TOKENS).unwrap();
    ids.first()
        .and_then(|&doc_id| snapshot.vectors(doc_id).unwrap())
        .map(|v| v.into_owned())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_json_ingest_commit_and_read() -> Result<()> {
    let engine = Engine::new(storage(), schema()).await?;
    let document = crate::json_to_document(&json!({
        "fields": {
            "title": "first",
            "vec": [1.0, 0.0],
            "tokens": [[1, 2], [3.5, -4]]
        }
    }))?;
    engine.put_document("a", document).await?;
    engine.commit().await?;

    assert_eq!(stored_tokens(&engine, "a"), Some(vec![1.0, 2.0, 3.5, -4.0]));

    // Kept out of the document store and therefore out of every result.
    let docs = engine.get_documents("a").await?;
    assert_eq!(docs.len(), 1);
    assert!(docs[0].fields.contains_key("title"));
    assert!(!docs[0].fields.contains_key(TOKENS));

    let stats = engine.stats()?;
    let tokens = stats.vector_fields.get(TOKENS).expect("tokens in stats");
    assert_eq!(tokens.vector_count, 1);
    assert_eq!(tokens.dimension, 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_wal_recovery_restores_tokens_and_does_not_replay_twice() -> Result<()> {
    let storage = storage();
    {
        let engine = Engine::new(storage.clone(), schema()).await?;
        engine
            .put_document("a", doc("first", [1.0, 0.0], &[[1.0, 1.0], [2.0, 2.0]]))
            .await?;
        // Dropped without commit: only the WAL holds the document.
    }
    let (vector_seq, lexical_seq) = {
        let engine = Engine::new(storage.clone(), schema()).await?;
        engine.commit().await?;
        assert_eq!(stored_tokens(&engine, "a"), Some(vec![1.0, 1.0, 2.0, 2.0]));
        (engine.vector.last_wal_seq(), engine.lexical.last_wal_seq())
    };
    // The multi-vector field publishes its checkpoint like every other
    // field, so the aggregate (a minimum across fields) keeps up.
    assert!(vector_seq > 0);
    assert_eq!(vector_seq, lexical_seq);

    let engine = Engine::new(storage, schema()).await?;
    assert_eq!(engine.vector.last_wal_seq(), vector_seq);
    assert_eq!(stored_tokens(&engine, "a"), Some(vec![1.0, 1.0, 2.0, 2.0]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_overwrite_and_delete() -> Result<()> {
    let engine = Engine::new(storage(), schema()).await?;
    engine
        .put_document("a", doc("first", [1.0, 0.0], &[[1.0, 1.0]]))
        .await?;
    engine.commit().await?;
    let old_id = internal_ids(&engine, "a")[0];

    engine
        .put_document("a", doc("second", [0.0, 1.0], &[[5.0, 6.0], [7.0, 8.0]]))
        .await?;
    engine.commit().await?;
    assert_eq!(stored_tokens(&engine, "a"), Some(vec![5.0, 6.0, 7.0, 8.0]));
    let snapshot = engine.vector.multi_vector_snapshot(TOKENS)?;
    assert!(snapshot.vectors(old_id)?.is_none());

    // A put without the field leaves the document with no token vectors.
    engine
        .put_document("a", Document::builder().add_text("title", "third").build())
        .await?;
    engine.commit().await?;
    assert_eq!(stored_tokens(&engine, "a"), None);

    engine
        .put_document("b", doc("other", [1.0, 0.0], &[[9.0, 9.0]]))
        .await?;
    engine.commit().await?;
    let b_id = internal_ids(&engine, "b")[0];
    engine.delete_documents("b").await?;
    engine.commit().await?;
    assert!(internal_ids(&engine, "b").is_empty());
    let snapshot = engine.vector.multi_vector_snapshot(TOKENS)?;
    assert!(snapshot.vectors(b_id)?.is_none());
    Ok(())
}

/// With no lexical field the lexical store keeps every field it receives;
/// the token vectors must not reach it, nor the document store.
#[tokio::test(flavor = "multi_thread")]
async fn test_tokens_stay_out_of_lexical_and_document_stores() -> Result<()> {
    let schema = Schema::builder()
        .add_hnsw_field("vec", HnswOption::new(2))
        .add_field(TOKENS, tokens_option())
        .build();
    let engine = Engine::new(storage(), schema).await?;
    let document = Document::builder()
        .add_vector("vec", vec![1.0, 0.0])
        .add_vector_array(TOKENS, vec![vec![1.0, 2.0]])
        .build();
    engine.put_document("a", document).await?;
    engine.commit().await?;

    assert_eq!(stored_tokens(&engine, "a"), Some(vec![1.0, 2.0]));
    let hits = engine
        .lexical
        .search(
            LexicalSearchRequest::new(Box::new(TermQuery::new("_id", "a"))).load_documents(true),
        )?
        .hits;
    assert_eq!(hits.len(), 1);
    let stored = hits[0].document.as_ref().expect("stored lexical document");
    assert!(!stored.fields.contains_key(TOKENS));
    assert!(
        engine
            .get_documents("a")
            .await?
            .iter()
            .all(|d| !d.fields.contains_key(TOKENS))
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_invalid_values_are_rejected_before_anything_is_written() -> Result<()> {
    let engine = Engine::new(storage(), schema()).await?;
    let cases = [
        // Wrong dimension.
        Document::builder()
            .add_vector_array(TOKENS, vec![vec![1.0, 2.0, 3.0]])
            .build(),
        // A single vector for a MultiVector field.
        Document::builder()
            .add_vector(TOKENS, vec![1.0, 2.0])
            .build(),
        // Token vectors for a single-vector field.
        Document::builder()
            .add_vector_array("vec", vec![vec![1.0, 2.0]])
            .build(),
        // Empty.
        Document::builder()
            .add_vector_array(TOKENS, Vec::new())
            .build(),
    ];
    for document in cases {
        let err = engine.put_document("bad", document).await.unwrap_err();
        assert!(matches!(err, LaurusError::InvalidArgument(_)), "{err}");
    }
    engine.commit().await?;
    assert!(internal_ids(&engine, "bad").is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_field_less_vector_search_skips_multi_vector_fields() -> Result<()> {
    let engine = Engine::new(storage(), schema()).await?;
    engine
        .put_document("a", doc("first", [1.0, 0.0], &[[1.0, 0.0]]))
        .await?;
    engine
        .put_document("b", doc("second", [0.0, 1.0], &[[1.0, 0.0]]))
        .await?;
    engine.commit().await?;

    let query = |fields: Option<Vec<String>>| {
        SearchRequestBuilder::new()
            .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
                vector: Vector::new(vec![1.0, 0.0]),
                weight: 1.0,
                fields,
            }]))
            .limit(10)
            .build()
    };

    let results = engine.search(query(None)).await?;
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids.len(), 2, "each document once, from `vec` only: {ids:?}");
    assert_eq!(ids[0], "a");

    let err = engine
        .search(query(Some(vec![TOKENS.to_string()])))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("not a vector-search target"),
        "{err}"
    );

    let err = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl("tokens:first")
                .build(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("multi-vector field"), "{err}");

    // An empty prefix selects every field by name, yet still skips the
    // multi-vector one instead of routing a query to it.
    let by_prefix = engine.vector.search(VectorSearchRequest {
        query: VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![1.0, 0.0]),
            weight: 1.0,
            fields: None,
        }]),
        params: VectorSearchParams {
            limit: 10,
            fields: Some(vec![FieldSelector::Prefix(String::new())]),
            ..Default::default()
        },
    })?;
    assert_eq!(by_prefix.hits.len(), 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_add_update_and_delete_a_multi_vector_field() -> Result<()> {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_hnsw_field("emb", HnswOption::new(2))
        .build();
    let engine = Engine::new(storage(), schema).await?;

    engine.add_field(TOKENS, tokens_option()).await?;
    engine
        .put_document(
            "a",
            Document::builder()
                .add_text("title", "first")
                .add_vector_array(TOKENS, vec![vec![1.0, 2.0]])
                .build(),
        )
        .await?;
    engine.commit().await?;
    assert_eq!(stored_tokens(&engine, "a"), Some(vec![1.0, 2.0]));

    // Hnsw -> MultiVector shares no on-disk layout: destructive.
    let rejected = engine
        .update_field("emb", tokens_option(), UpdateFieldOptions::default())
        .await;
    assert!(rejected.is_err());
    let outcome = engine
        .update_field(
            "emb",
            tokens_option(),
            UpdateFieldOptions {
                reindex: true,
                ..Default::default()
            },
        )
        .await?;
    assert_eq!(
        outcome.classification,
        crate::engine::schema::FieldChangeKind::Destructive
    );
    assert!(matches!(
        outcome.schema.fields.get("emb"),
        Some(FieldOption::MultiVector(_))
    ));

    engine.delete_field(TOKENS).await?;
    assert!(engine.vector.multi_vector_snapshot(TOKENS).is_err());
    let err = engine
        .put_document(
            "b",
            Document::builder()
                .add_field(TOKENS, DataValue::VectorArray(vec![vec![1.0, 2.0]]))
                .build(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("MultiVector"), "{err}");
    Ok(())
}
