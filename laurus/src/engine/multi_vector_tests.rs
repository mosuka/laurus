//! End-to-end tests of multi-vector fields through the [`Engine`]
//! (Issue #1177).
//!
//! These live inside the engine module so they can read the stored token
//! vectors through the engine's vector store, which has no public accessor.

use std::sync::Arc;

use serde_json::json;

use crate::data::{DataValue, Document};
use crate::embedding::embedder::{EmbedInput, EmbedInputType, EmbedRole, Embedder, TokenEmbedder};
use crate::embedding::per_field::PerFieldEmbedder;
use crate::embedding::precomputed::PrecomputedEmbedder;
use crate::engine::schema::{FieldOption, Schema};
use crate::engine::search::{RescoreOptions, SearchRequestBuilder};
use crate::engine::{Engine, UpdateFieldOptions};
use crate::error::{LaurusError, Result};
use crate::lexical::core::field::TextOption;
use crate::lexical::query::term::TermQuery;
use crate::lexical::search::searcher::LexicalSearchRequest;
use crate::storage::Storage;
use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::field::{FlatOption, HnswOption, MultiVectorOption};
use crate::vector::core::late_interaction::{dispatch_calls, max_sim, max_sim_dyn};
use crate::vector::core::multi_vector::{MultiVectorStorage, binary_max_sim_tolerance};
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

/// Token embedder turning each character into the vector
/// `[code point, role]`, where `role` is 1 for documents and 2 for queries.
#[derive(Debug)]
struct CharTokenEmbedder;

#[async_trait::async_trait]
impl TokenEmbedder for CharTokenEmbedder {
    async fn embed_tokens(
        &self,
        inputs: &[EmbedInput<'_>],
        role: EmbedRole,
    ) -> Result<Vec<Vec<Vector>>> {
        let marker = match role {
            EmbedRole::Document => 1.0,
            EmbedRole::Query => 2.0,
        };
        Ok(inputs
            .iter()
            .map(|input| {
                input
                    .as_text()
                    .unwrap_or_default()
                    .chars()
                    .map(|c| Vector::new(vec![c as u32 as f32, marker]))
                    .collect()
            })
            .collect())
    }

    fn token_dimension(&self) -> usize {
        2
    }
}

#[async_trait::async_trait]
impl Embedder for CharTokenEmbedder {
    async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
        Err(LaurusError::invalid_argument("token embedder"))
    }

    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }

    fn name(&self) -> &str {
        "char-tokens"
    }

    fn as_token_embedder(&self) -> Option<&dyn TokenEmbedder> {
        Some(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// An engine whose `tokens` field embeds text with [`CharTokenEmbedder`].
async fn text_engine(storage: Arc<dyn Storage>, schema: Schema) -> Result<Engine> {
    let per_field = PerFieldEmbedder::new(Arc::new(PrecomputedEmbedder::new()));
    per_field.add_embedder(TOKENS, Arc::new(CharTokenEmbedder));
    Engine::builder(storage, schema)
        .embedder(Arc::new(per_field))
        .build()
        .await
}

fn text_doc(title: &str, tokens: &str) -> Document {
    Document::builder()
        .add_text("title", title)
        .add_text(TOKENS, tokens)
        .build()
}

/// Issue #1349: text in a multi-vector field is embedded into one vector
/// per token, as a document, and stays out of the stored document.
#[tokio::test(flavor = "multi_thread")]
async fn test_text_is_embedded_into_token_vectors() -> Result<()> {
    let engine = text_engine(storage(), schema()).await?;
    engine.put_document("a", text_doc("first", "ab")).await?;
    engine.commit().await?;

    assert_eq!(
        stored_tokens(&engine, "a"),
        Some(vec![97.0, 1.0, 98.0, 1.0])
    );
    let docs = engine.get_documents("a").await?;
    assert!(!docs[0].fields.contains_key(TOKENS));
    Ok(())
}

/// Issue #1349: without a token-level embedder, text is rejected before
/// the WAL append, so the previous version of the document stays intact.
#[tokio::test(flavor = "multi_thread")]
async fn test_text_without_a_token_embedder_is_rejected() -> Result<()> {
    let engine = Engine::new(storage(), schema()).await?;
    engine
        .put_document("a", doc("first", [1.0, 0.0], &[[3.0, 4.0]]))
        .await?;
    engine.commit().await?;

    let err = engine
        .put_document("a", text_doc("second", "ab"))
        .await
        .unwrap_err();
    assert!(matches!(err, LaurusError::InvalidArgument(_)), "{err}");
    assert!(err.to_string().contains("token-level embedder"), "{err}");
    engine.commit().await?;
    assert_eq!(stored_tokens(&engine, "a"), Some(vec![3.0, 4.0]));
    assert_eq!(engine.get_documents("a").await?.len(), 1);

    // Unlike a coercion failure, this also rejects the document when the
    // schema ignores what it cannot use.
    let ignoring = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field(TOKENS, tokens_option())
        .dynamic_field_policy(crate::engine::schema::DynamicFieldPolicy::Ignore)
        .build();
    let engine = Engine::new(storage(), ignoring).await?;
    let err = engine
        .put_document("b", text_doc("third", "ab"))
        .await
        .unwrap_err();
    assert!(matches!(err, LaurusError::InvalidArgument(_)), "{err}");
    Ok(())
}

/// Issue #1349: the WAL keeps the text, and recovery embeds it again.
#[tokio::test(flavor = "multi_thread")]
async fn test_wal_text_is_embedded_again_on_recovery() -> Result<()> {
    let storage = storage();
    {
        let engine = text_engine(storage.clone(), schema()).await?;
        engine.put_document("a", text_doc("first", "xyz")).await?;
        // Dropped without commit: only the WAL holds the document.
    }
    let engine = text_engine(storage, schema()).await?;
    engine.commit().await?;
    assert_eq!(
        stored_tokens(&engine, "a"),
        Some(vec![120.0, 1.0, 121.0, 1.0, 122.0, 1.0])
    );
    Ok(())
}

/// Issue #1349: an embedder producing vectors of another dimension is
/// rejected like a document carrying them.
#[tokio::test(flavor = "multi_thread")]
async fn test_text_embedded_to_the_wrong_dimension_is_rejected() -> Result<()> {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field(TOKENS, FieldOption::MultiVector(MultiVectorOption::new(3)))
        .build();
    // Served by the default embedder, which the build does not check.
    let per_field = PerFieldEmbedder::new(Arc::new(CharTokenEmbedder));
    let engine = Engine::builder(storage(), schema)
        .embedder(Arc::new(per_field))
        .build()
        .await?;
    let err = engine
        .put_document("a", text_doc("first", "ab"))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("dimension 2, expected 3"), "{err}");
    Ok(())
}

/// Issue #1346: changing `storage` (the on-disk element kind) requires an
/// explicit reindex, actually re-encodes the existing segment under the
/// new kind (not just the persisted schema), and loses precision only in
/// the direction the new kind can represent -- it never resurrects it.
#[tokio::test(flavor = "multi_thread")]
async fn test_update_field_storage_compression_reindexes_and_is_lossy_one_way() -> Result<()> {
    let engine = Engine::new(storage(), schema()).await?;
    let original = vec![vec![0.6, 0.8], vec![-0.280_267_7, 0.959_940_1]];
    engine
        .put_document(
            "a",
            Document::builder()
                .add_text("title", "first")
                .add_vector("vec", vec![1.0, 0.0])
                .add_vector_array(TOKENS, original.clone())
                .build(),
        )
        .await?;
    engine.commit().await?;
    let before = stored_tokens(&engine, "a").unwrap();
    assert_eq!(before, original.concat());

    // Without `reindex: true`, a storage change is rejected (it requires
    // rebuilding every existing segment).
    let compressed = FieldOption::MultiVector(
        MultiVectorOption::new(2)
            .distance(DistanceMetric::DotProduct)
            .storage(MultiVectorStorage::F16),
    );
    let rejected = engine
        .update_field(TOKENS, compressed.clone(), UpdateFieldOptions::default())
        .await;
    assert!(rejected.is_err());

    let outcome = engine
        .update_field(
            TOKENS,
            compressed,
            UpdateFieldOptions {
                reindex: true,
                ..Default::default()
            },
        )
        .await?;
    assert_eq!(
        outcome.classification,
        crate::engine::schema::FieldChangeKind::Reindex
    );
    assert!(matches!(
        outcome.schema.fields.get(TOKENS),
        Some(FieldOption::MultiVector(o)) if o.storage == MultiVectorStorage::F16
    ));

    // The existing segment was actually re-encoded, not just the schema:
    // values read back are close to, but not bit-identical to, the f32
    // originals.
    let after_f16 = stored_tokens(&engine, "a").unwrap();
    assert_ne!(after_f16, before, "f16 re-encoding must be lossy");
    for (a, b) in before.iter().zip(&after_f16) {
        assert!((a - b).abs() <= 2e-3, "{a} vs {b}");
    }

    // Switching back to F32 cannot resurrect the precision already lost:
    // the values read back match the f16-quantized ones, not the
    // originals.
    let back_to_f32 =
        FieldOption::MultiVector(MultiVectorOption::new(2).distance(DistanceMetric::DotProduct));
    engine
        .update_field(
            TOKENS,
            back_to_f32,
            UpdateFieldOptions {
                reindex: true,
                ..Default::default()
            },
        )
        .await?;
    let after_round_trip = stored_tokens(&engine, "a").unwrap();
    assert_eq!(after_round_trip, after_f16);
    assert_ne!(after_round_trip, before);
    Ok(())
}

/// Issue #1346: a `Binary` field (Cosine, 4 dimensions) next to a text
/// field.
fn binary_option(distance: DistanceMetric) -> FieldOption {
    FieldOption::MultiVector(
        MultiVectorOption::new(4)
            .distance(distance)
            .storage(MultiVectorStorage::Binary),
    )
}

fn binary_schema(storage: MultiVectorStorage) -> Schema {
    Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field(
            TOKENS,
            FieldOption::MultiVector(
                MultiVectorOption::new(4)
                    .distance(DistanceMetric::Cosine)
                    .storage(storage),
            ),
        )
        .build()
}

/// Token `i` of document `n`: deterministic and anisotropic, like a
/// late-interaction model's.
fn binary_token(n: u64, i: u64) -> Vec<f32> {
    let h = |k: u64| {
        let x = ((n * 1_000 + i) * 4 + k).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    vec![
        0.8 + 0.4 * h(0),
        0.6 + 0.4 * h(1),
        -0.5 + 0.4 * h(2),
        0.4 * h(3),
    ]
}

/// Put 64 documents of 32 tokens each: 2048 token vectors, exactly what a
/// `Binary` field needs to learn its parameters in one flush.
async fn put_binary_corpus(engine: &Engine) -> Result<()> {
    for n in 0..64 {
        let tokens = (0..32).map(|i| binary_token(n, i)).collect();
        let doc = Document::builder()
            .add_text("title", format!("doc {n}"))
            .add_vector_array(TOKENS, tokens)
            .build();
        engine.put_document(&format!("d{n}"), doc).await?;
    }
    Ok(())
}

/// The element kind every live document of the corpus is stored as.
fn stored_kinds(engine: &Engine) -> Vec<MultiVectorStorage> {
    let snapshot = engine.vector.multi_vector_snapshot(TOKENS).unwrap();
    let mut kinds: Vec<MultiVectorStorage> = (0..64)
        .flat_map(|n| internal_ids(engine, &format!("d{n}")))
        .map(|doc_id| snapshot.rows(doc_id).unwrap().unwrap().kind())
        .collect();
    kinds.dedup();
    kinds
}

#[tokio::test(flavor = "multi_thread")]
async fn test_wal_replay_of_a_binary_field_learns_its_parameters() -> Result<()> {
    let storage = storage();
    {
        let engine =
            Engine::new(storage.clone(), binary_schema(MultiVectorStorage::Binary)).await?;
        put_binary_corpus(&engine).await?;
        // Dropped without commit: only the WAL holds the documents.
    }
    let engine = Engine::new(storage, binary_schema(MultiVectorStorage::Binary)).await?;
    engine.commit().await?;
    assert_eq!(stored_kinds(&engine), [MultiVectorStorage::Binary]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_update_field_to_binary_reindexes_every_segment() -> Result<()> {
    let engine = Engine::new(storage(), binary_schema(MultiVectorStorage::F32)).await?;
    put_binary_corpus(&engine).await?;
    engine.commit().await?;
    assert_eq!(stored_kinds(&engine), [MultiVectorStorage::F32]);

    let outcome = engine
        .update_field(
            TOKENS,
            binary_option(DistanceMetric::Cosine),
            UpdateFieldOptions {
                reindex: true,
                ..Default::default()
            },
        )
        .await?;
    assert_eq!(
        outcome.classification,
        crate::engine::schema::FieldChangeKind::Reindex
    );
    assert_eq!(stored_kinds(&engine), [MultiVectorStorage::Binary]);
    Ok(())
}

/// A distance change is destructive (the field is purged, then re-created
/// empty). `Binary` with `DotProduct` is rejected, and that must happen
/// before the purge: the field's data and schema stay as they were.
#[tokio::test(flavor = "multi_thread")]
async fn test_rejected_binary_option_never_purges_the_field() -> Result<()> {
    let engine = Engine::new(storage(), binary_schema(MultiVectorStorage::Binary)).await?;
    put_binary_corpus(&engine).await?;
    engine.commit().await?;
    let before = stored_tokens(&engine, "d3").unwrap();
    let schema_before = format!("{:?}", engine.schema().fields.get(TOKENS));

    for dry_run in [true, false] {
        let err = engine
            .update_field(
                TOKENS,
                binary_option(DistanceMetric::DotProduct),
                UpdateFieldOptions {
                    reindex: true,
                    dry_run,
                },
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("requires distance Cosine"),
            "{err}"
        );
    }
    assert_eq!(stored_tokens(&engine, "d3"), Some(before));
    assert_eq!(
        format!("{:?}", engine.schema().fields.get(TOKENS)),
        schema_before
    );

    let err = engine
        .add_field("more", binary_option(DistanceMetric::DotProduct))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("requires distance Cosine"),
        "{err}"
    );
    assert!(!engine.schema().fields.contains_key("more"));
    Ok(())
}

#[test]
fn test_schema_rejects_binary_with_dot_product() {
    let err = Schema::builder()
        .add_field(TOKENS, binary_option(DistanceMetric::DotProduct))
        .try_build()
        .unwrap_err();
    assert!(
        err.to_string().contains("requires distance Cosine"),
        "{err}"
    );
    assert!(
        Schema::builder()
            .add_field(TOKENS, binary_option(DistanceMetric::Cosine))
            .try_build()
            .is_ok()
    );
}

/// Issue #1427: dimension of the rescore tests' `Binary` field. It is not a
/// multiple of 8, so every row carries padding bits. At 4 dimensions a
/// document's 32 tokens hold most of the 16 bit patterns, and no document
/// of [`put_binary_corpus`] scores differently through the two paths.
const RESCORE_DIMENSION: usize = 20;

fn rescore_schema() -> Schema {
    Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field(
            TOKENS,
            FieldOption::MultiVector(
                MultiVectorOption::new(RESCORE_DIMENSION)
                    .distance(DistanceMetric::Cosine)
                    .storage(MultiVectorStorage::Binary),
            ),
        )
        .build()
}

/// Token `i` of document `n`: deterministic and anisotropic, like
/// [`binary_token`], at [`RESCORE_DIMENSION`].
fn rescore_token(n: u64, i: u64) -> Vec<f32> {
    (0..RESCORE_DIMENSION as u64)
        .map(|k| {
            let x = ((n * 1_000 + i) * 64 + k).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let noise = (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let base = [0.6, -0.3, 0.1][(k % 3) as usize];
            base + 0.4 * noise
        })
        .collect()
}

/// Put documents `n` of `ids`, 32 tokens each, titled `doc <n>`.
async fn put_rescore_corpus(engine: &Engine, ids: std::ops::Range<u64>) -> Result<()> {
    for n in ids {
        let tokens = (0..32).map(|i| rescore_token(n, i)).collect();
        let doc = Document::builder()
            .add_text("title", format!("doc {n}"))
            .add_vector_array(TOKENS, tokens)
            .build();
        engine.put_document(&format!("d{n}"), doc).await?;
    }
    Ok(())
}

/// Rescore every document with `title:doc` and check each score against
/// the query the field's snapshot prepares, bit for bit, and against
/// MaxSim over the decoded rows, within the derived tolerance. Returns the
/// storage kinds of the scored documents.
async fn check_binary_rescore(
    engine: &Engine,
    documents: usize,
) -> Result<Vec<MultiVectorStorage>> {
    let query: Vec<Vector> = (0..5)
        .map(|i| Vector::new(rescore_token(1_000, i)))
        .collect();
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl("title:doc")
                .rescore(RescoreOptions::late_interaction(TOKENS, query.clone()).window_size(100))
                .limit(100)
                .build(),
        )
        .await?;
    assert_eq!(results.len(), documents);

    // The rescore normalizes the query of a Cosine field the same way.
    let mut flat = Vec::new();
    for vector in &query {
        flat.extend_from_slice(&vector.normalized().data);
    }
    let snapshot = engine.vector.multi_vector_snapshot(TOKENS).unwrap();
    let prepared = snapshot.max_sim_query(&flat);
    assert_eq!(prepared.prepared(), 1);

    let mut kinds = Vec::new();
    let mut told_apart = 0;
    for result in &results {
        let doc_id = internal_ids(engine, &result.id)[0];
        let rows = snapshot.rows(doc_id)?.unwrap();
        let expected = prepared.max_sim(&rows, &mut Vec::new());
        assert_eq!(result.score.to_bits(), expected.to_bits(), "{result:?}");

        let through_decode = max_sim(&flat, &rows.to_f32(), RESCORE_DIMENSION);
        if let Some(params) = rows.codec().binary_params() {
            let tolerance = binary_max_sim_tolerance(params, &flat, rows.as_bytes());
            assert!(
                (result.score - through_decode).abs() <= tolerance,
                "{result:?}: decoded {through_decode} (tolerance {tolerance})"
            );
            if through_decode.to_bits() != expected.to_bits() {
                told_apart += 1;
            }
        } else {
            assert_eq!(result.score.to_bits(), through_decode.to_bits());
        }
        kinds.push(rows.kind());
    }
    // Without a document that the decode scores differently, the bit-for-bit
    // check above could not tell a rescore that skipped the tables.
    assert!(told_apart > 0, "no document tells the two paths apart");
    kinds.sort_by_key(|kind| kind.tag());
    kinds.dedup();
    Ok(kinds)
}

/// Issue #1427: a `Binary` field's documents are rescored from their bits.
#[tokio::test(flavor = "multi_thread")]
async fn test_binary_rescore_scores_documents_from_their_bits() -> Result<()> {
    let engine = Engine::new(storage(), rescore_schema()).await?;
    // 64 documents of 32 tokens: the 2,048 a field learns its parameters from.
    put_rescore_corpus(&engine, 0..64).await?;
    engine.commit().await?;
    assert_eq!(
        check_binary_rescore(&engine, 64).await?,
        [MultiVectorStorage::Binary]
    );
    Ok(())
}

/// A commit below the training threshold writes an interim `F16` segment;
/// the rescore decodes its documents and scores the `Binary` ones from
/// their bits in the same search.
#[tokio::test(flavor = "multi_thread")]
async fn test_binary_rescore_of_a_snapshot_mixing_f16_and_binary_segments() -> Result<()> {
    let engine = Engine::new(storage(), rescore_schema()).await?;
    put_rescore_corpus(&engine, 100..108).await?;
    engine.commit().await?;
    put_rescore_corpus(&engine, 0..64).await?;
    engine.commit().await?;
    assert_eq!(
        check_binary_rescore(&engine, 72).await?,
        [MultiVectorStorage::F16, MultiVectorStorage::Binary]
    );
    Ok(())
}

/// Issue #1442: dimension of the `F32` rescore test, one with a fixed-width
/// copy of MaxSim.
const FIXED_WIDTH_DIMENSION: usize = 96;

/// Token `i` of document `n` at [`FIXED_WIDTH_DIMENSION`], built like
/// [`rescore_token`].
fn fixed_width_token(n: u64, i: u64) -> Vec<f32> {
    (0..FIXED_WIDTH_DIMENSION as u64)
        .map(|k| {
            let x = ((n * 1_000 + i) * 128 + k).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let noise = (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let base = [0.6, -0.3, 0.1][(k % 3) as usize];
            base + 0.4 * noise
        })
        .collect()
}

/// Issue #1442: a 96-dimension `F32` field is rescored through MaxSim's
/// fixed-width copy, with the scores of the run-time loop over the
/// snapshot's vectors.
#[tokio::test(flavor = "multi_thread")]
async fn test_f32_rescore_at_a_fixed_width_dimension() -> Result<()> {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field(
            TOKENS,
            FieldOption::MultiVector(
                MultiVectorOption::new(FIXED_WIDTH_DIMENSION).distance(DistanceMetric::Cosine),
            ),
        )
        .build();
    let engine = Engine::new(storage(), schema).await?;
    for n in 0..16 {
        let tokens = (0..8).map(|i| fixed_width_token(n, i)).collect();
        let doc = Document::builder()
            .add_text("title", format!("doc {n}"))
            .add_vector_array(TOKENS, tokens)
            .build();
        engine.put_document(&format!("d{n}"), doc).await?;
    }
    engine.commit().await?;

    let query: Vec<Vector> = (0..4)
        .map(|i| Vector::new(fixed_width_token(1_000, i)))
        .collect();
    let before = dispatch_calls::on_any_thread(dispatch_calls::Loop::Fixed);
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl("title:doc")
                .rescore(RescoreOptions::late_interaction(TOKENS, query.clone()).window_size(100))
                .limit(100)
                .build(),
        )
        .await?;
    assert_eq!(results.len(), 16);
    // The rescore scores on rayon's pool; other tests may add to the count,
    // but without the dispatch nothing would.
    assert!(
        dispatch_calls::on_any_thread(dispatch_calls::Loop::Fixed) >= before + 16,
        "the rescore should take MaxSim's fixed-width copy"
    );

    let mut flat = Vec::new();
    for vector in &query {
        flat.extend_from_slice(&vector.normalized().data);
    }
    let snapshot = engine.vector.multi_vector_snapshot(TOKENS).unwrap();
    for result in &results {
        let doc_id = internal_ids(&engine, &result.id)[0];
        let vectors = snapshot.vectors(doc_id)?.unwrap();
        let expected = max_sim_dyn(&flat, &vectors, FIXED_WIDTH_DIMENSION);
        assert_eq!(result.score.to_bits(), expected.to_bits(), "{result:?}");
    }
    Ok(())
}

/// Issue #1444: dimension of the block4 `F32` rescore test, one without a
/// fixed-width copy of MaxSim.
const BLOCK4_DIMENSION: usize = 100;

/// Token `i` of document `n` at [`BLOCK4_DIMENSION`], built like
/// [`rescore_token`].
fn block4_token(n: u64, i: u64) -> Vec<f32> {
    (0..BLOCK4_DIMENSION as u64)
        .map(|k| {
            let x = ((n * 1_000 + i) * 128 + k).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let noise = (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let base = [0.6, -0.3, 0.1][(k % 3) as usize];
            base + 0.4 * noise
        })
        .collect()
}

/// Issue #1444: a 100-dimension `F32` field (no fixed-width copy) is
/// rescored through MaxSim's four-vector-per-pass loop, with the scores of
/// the one-at-a-time run-time loop over the snapshot's vectors.
#[tokio::test(flavor = "multi_thread")]
async fn test_f32_rescore_at_a_block4_dimension() -> Result<()> {
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field(
            TOKENS,
            FieldOption::MultiVector(
                MultiVectorOption::new(BLOCK4_DIMENSION).distance(DistanceMetric::Cosine),
            ),
        )
        .build();
    let engine = Engine::new(storage(), schema).await?;
    for n in 0..16 {
        let tokens = (0..8).map(|i| block4_token(n, i)).collect();
        let doc = Document::builder()
            .add_text("title", format!("doc {n}"))
            .add_vector_array(TOKENS, tokens)
            .build();
        engine.put_document(&format!("d{n}"), doc).await?;
    }
    engine.commit().await?;

    let query: Vec<Vector> = (0..4)
        .map(|i| Vector::new(block4_token(1_000, i)))
        .collect();
    let before = dispatch_calls::on_any_thread(dispatch_calls::Loop::Block4);
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .query_dsl("title:doc")
                .rescore(RescoreOptions::late_interaction(TOKENS, query.clone()).window_size(100))
                .limit(100)
                .build(),
        )
        .await?;
    assert_eq!(results.len(), 16);
    // The rescore scores on rayon's pool; other tests may add to the count,
    // but without the dispatch nothing would.
    assert!(
        dispatch_calls::on_any_thread(dispatch_calls::Loop::Block4) >= before + 16,
        "the rescore should take MaxSim's block4 loop"
    );

    let mut flat = Vec::new();
    for vector in &query {
        flat.extend_from_slice(&vector.normalized().data);
    }
    let snapshot = engine.vector.multi_vector_snapshot(TOKENS).unwrap();
    for result in &results {
        let doc_id = internal_ids(&engine, &result.id)[0];
        let vectors = snapshot.vectors(doc_id)?.unwrap();
        let expected = max_sim_dyn(&flat, &vectors, BLOCK4_DIMENSION);
        assert_eq!(result.score.to_bits(), expected.to_bits(), "{result:?}");
    }
    Ok(())
}

/// Issue #1438: token counts of the decoded rescore test's documents, in
/// document order. Each long document is followed by a shorter one, so a
/// scratch reused without being cleared would score the shorter one over
/// rows left by the longer one.
const DECODED_TOKEN_COUNTS: [u64; 4] = [24, 3, 17, 1];

/// Issue #1438: `F16` and `Int8` documents of different lengths, decoded one
/// after another into the scratch the rescore reuses, score as a decode of
/// each document into a new buffer does, bit for bit.
#[tokio::test(flavor = "multi_thread")]
async fn test_decoded_rescore_of_documents_of_different_lengths_matches_a_fresh_decode()
-> Result<()> {
    for kind in [MultiVectorStorage::F16, MultiVectorStorage::Int8] {
        let schema = Schema::builder()
            .add_text_field("title", TextOption::default())
            .add_field(
                TOKENS,
                FieldOption::MultiVector(
                    MultiVectorOption::new(RESCORE_DIMENSION)
                        .distance(DistanceMetric::Cosine)
                        .storage(kind),
                ),
            )
            .build();
        let engine = Engine::new(storage(), schema).await?;
        for n in 0..64u64 {
            let count = DECODED_TOKEN_COUNTS[n as usize % DECODED_TOKEN_COUNTS.len()];
            let tokens = (0..count).map(|i| rescore_token(n, i)).collect();
            let doc = Document::builder()
                .add_text("title", format!("doc {n}"))
                .add_vector_array(TOKENS, tokens)
                .build();
            engine.put_document(&format!("d{n}"), doc).await?;
        }
        engine.commit().await?;

        let query: Vec<Vector> = (0..5)
            .map(|i| Vector::new(rescore_token(1_000, i)))
            .collect();
        let results = engine
            .search(
                SearchRequestBuilder::new()
                    .query_dsl("title:doc")
                    .rescore(
                        RescoreOptions::late_interaction(TOKENS, query.clone()).window_size(100),
                    )
                    .limit(100)
                    .build(),
            )
            .await?;
        assert_eq!(results.len(), 64, "{kind:?}");

        let mut flat = Vec::new();
        for vector in &query {
            flat.extend_from_slice(&vector.normalized().data);
        }
        let snapshot = engine.vector.multi_vector_snapshot(TOKENS).unwrap();
        for result in &results {
            let doc_id = internal_ids(&engine, &result.id)[0];
            let rows = snapshot.rows(doc_id)?.unwrap();
            assert_eq!(rows.kind(), kind, "{result:?}");
            let expected = max_sim(&flat, &rows.to_f32(), RESCORE_DIMENSION);
            assert_eq!(
                result.score.to_bits(),
                expected.to_bits(),
                "{kind:?} {result:?}"
            );
        }
    }
    Ok(())
}

/// Issue #1346: a Cosine field of 4-dimensional token vectors pooled by
/// `pool_factor`, next to a text field.
fn pooled_option(pool_factor: u32) -> FieldOption {
    FieldOption::MultiVector(MultiVectorOption::new(4).pool_factor(pool_factor))
}

fn pooled_schema(pool_factor: u32) -> Schema {
    Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_field(TOKENS, pooled_option(pool_factor))
        .build()
}

/// Put 8 documents of 9 tokens each (`d0` .. `d7`).
async fn put_pooled_corpus(engine: &Engine) -> Result<()> {
    for n in 0..8 {
        let tokens = (0..9).map(|i| binary_token(n, i)).collect();
        let doc = Document::builder()
            .add_text("title", format!("doc {n}"))
            .add_vector_array(TOKENS, tokens)
            .build();
        engine.put_document(&format!("d{n}"), doc).await?;
    }
    Ok(())
}

fn reindex() -> UpdateFieldOptions {
    UpdateFieldOptions {
        reindex: true,
        ..Default::default()
    }
}

/// Turning pooling on re-pools the stored token vectors: the field ends up
/// exactly as one that pooled them at ingest.
#[tokio::test(flavor = "multi_thread")]
async fn test_update_field_turning_pooling_on_matches_ingest() -> Result<()> {
    let migrated = Engine::new(storage(), pooled_schema(1)).await?;
    put_pooled_corpus(&migrated).await?;
    migrated.commit().await?;
    let outcome = migrated
        .update_field(TOKENS, pooled_option(4), reindex())
        .await?;
    assert_eq!(
        outcome.classification,
        crate::engine::schema::FieldChangeKind::Reindex
    );

    let fresh = Engine::new(storage(), pooled_schema(4)).await?;
    put_pooled_corpus(&fresh).await?;
    fresh.commit().await?;
    for n in 0..8 {
        let id = format!("d{n}");
        let tokens = stored_tokens(&migrated, &id).unwrap();
        assert_eq!(
            tokens.len(),
            crate::vector::core::token_pooling::pooled_count(9, 4) * 4
        );
        assert_eq!(Some(tokens), stored_tokens(&fresh, &id), "{id}");
    }
    Ok(())
}

/// The original token vectors are gone once a field pools, so changing a
/// factor in use discards the field's data (and says so).
#[tokio::test(flavor = "multi_thread")]
async fn test_update_field_changing_a_pool_factor_discards_the_field() -> Result<()> {
    let engine = Engine::new(storage(), pooled_schema(2)).await?;
    put_pooled_corpus(&engine).await?;
    engine.commit().await?;
    let outcome = engine
        .update_field(TOKENS, pooled_option(4), reindex())
        .await?;
    assert_eq!(
        outcome.classification,
        crate::engine::schema::FieldChangeKind::Destructive
    );
    assert!(outcome.schema.pending_reindex.contains(TOKENS));
    assert_eq!(stored_tokens(&engine, "d0"), None);
    Ok(())
}

/// The WAL keeps the raw input, so replaying it pools each document once.
#[tokio::test(flavor = "multi_thread")]
async fn test_wal_replay_pools_documents_once() -> Result<()> {
    let storage = storage();
    {
        let engine = Engine::new(storage.clone(), pooled_schema(2)).await?;
        put_pooled_corpus(&engine).await?;
        // Dropped without commit: only the WAL holds the documents.
    }
    let replayed = Engine::new(storage, pooled_schema(2)).await?;
    replayed.commit().await?;

    let direct = Engine::new(self::storage(), pooled_schema(2)).await?;
    put_pooled_corpus(&direct).await?;
    direct.commit().await?;
    for n in 0..8 {
        let id = format!("d{n}");
        assert_eq!(
            stored_tokens(&replayed, &id),
            stored_tokens(&direct, &id),
            "{id}"
        );
    }
    Ok(())
}

/// Issue #1429: documents put concurrently are pooled before the writer is
/// taken, and stored exactly as the token pooling reference pools them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_puts_store_the_reference_pooling() -> Result<()> {
    let engine = Arc::new(Engine::new(storage(), pooled_schema(2)).await?);
    let puts = (0..8).map(|n| {
        let engine = Arc::clone(&engine);
        tokio::spawn(async move {
            let tokens = (0..9).map(|i| binary_token(n, i)).collect();
            let doc = Document::builder()
                .add_text("title", format!("doc {n}"))
                .add_vector_array(TOKENS, tokens)
                .build();
            engine.put_document(&format!("d{n}"), doc).await
        })
    });
    for put in futures::future::join_all(puts).await {
        put.expect("the put task panicked")?;
    }
    engine.commit().await?;
    for n in 0..8 {
        let rows: Vec<f32> = (0..9)
            .flat_map(|i| Vector::new(binary_token(n, i)).normalized().data.to_vec())
            .collect();
        let expected = crate::vector::core::token_pooling::pool(&rows, 4, 2, true)
            .expect("9 token vectors pool to 5");
        assert_eq!(
            stored_tokens(&engine, &format!("d{n}")),
            Some(expected),
            "d{n}"
        );
    }
    Ok(())
}

/// A pool factor of 0 is rejected up front, in a dry run too, and the
/// field's data is untouched.
#[tokio::test(flavor = "multi_thread")]
async fn test_pool_factor_zero_is_rejected_before_anything_is_purged() -> Result<()> {
    let engine = Engine::new(storage(), pooled_schema(2)).await?;
    put_pooled_corpus(&engine).await?;
    engine.commit().await?;
    let before = stored_tokens(&engine, "d3").unwrap();
    for dry_run in [true, false] {
        let err = engine
            .update_field(
                TOKENS,
                pooled_option(0),
                UpdateFieldOptions {
                    reindex: true,
                    dry_run,
                },
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("pool_factor must be at least 1"),
            "{err}"
        );
    }
    assert_eq!(stored_tokens(&engine, "d3"), Some(before));
    assert!(
        Schema::builder()
            .add_field(TOKENS, pooled_option(0))
            .try_build()
            .is_err()
    );
    Ok(())
}
