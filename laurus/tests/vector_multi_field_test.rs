//! Regression tests for [`MultiFieldVectorIndex`] (Issue #948).
//!
//! Before this type existed, `VectorStore` routed every vector field
//! through a single `VectorIndex`, and HNSW's graph node IDs are `doc_id`s
//! -- so a document with vectors in two fields silently overwrote one
//! field's vector with the other's (non-deterministic depending on
//! processing order). These tests exercise `MultiFieldVectorIndex` directly
//! (below `VectorStore`, which is not yet wired to it -- Issue #948 Phase 4)
//! to prove the routing itself is correct.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::{HnswIndexConfig, VectorIndexTypeConfig};
use laurus::vector::index::multi_field::MultiFieldVectorIndex;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};
use laurus::vector::{DistanceMetric, Vector};
use laurus::{EmbedInput, EmbedInputType, Embedder, LaurusError, Result};

#[derive(Debug)]
struct MockEmbedder;

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
        Err(LaurusError::invalid_argument(
            "embedding not used by this test",
        ))
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "mock"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
}

fn hnsw_config(dimension: usize, distance_metric: DistanceMetric) -> VectorIndexTypeConfig {
    VectorIndexTypeConfig::HNSW(HnswIndexConfig {
        dimension,
        distance_metric,
        m: 16,
        ef_construction: 100,
        normalize_vectors: distance_metric == DistanceMetric::Cosine,
        ..Default::default()
    })
}

fn vec_of(values: &[f32]) -> Vector {
    Vector::new(values.to_vec())
}

fn query(v: Vector, field_name: Option<&str>, top_k: usize) -> VectorIndexQuery {
    VectorIndexQuery {
        query: v,
        params: VectorIndexQueryParams {
            top_k,
            ..Default::default()
        },
        field_name: field_name.map(str::to_string),
        filter: None,
    }
}

/// Core gate (Issue #948): a document with vectors in two fields must keep
/// BOTH vectors -- neither field may silently overwrite the other.
#[test]
fn two_fields_same_doc_id_both_retained() {
    let mut fields = BTreeMap::new();
    fields.insert(
        "title_vec".to_string(),
        hnsw_config(4, DistanceMetric::Cosine),
    );
    fields.insert(
        "body_vec".to_string(),
        hnsw_config(4, DistanceMetric::Cosine),
    );

    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![
            (1, "title_vec".to_string(), vec_of(&[1.0, 0.0, 0.0, 0.0])),
            (1, "body_vec".to_string(), vec_of(&[0.0, 1.0, 0.0, 0.0])),
        ])
        .unwrap();
    writer.commit().unwrap();

    // Both fields must independently report the vector for doc 1 -- if the
    // bug were present, one of these would come back empty.
    let reader = index.reader().unwrap();
    let title_vectors = reader.get_vectors_by_field("title_vec").unwrap();
    let body_vectors = reader.get_vectors_by_field("body_vec").unwrap();
    assert_eq!(
        title_vectors.len(),
        1,
        "title_vec must keep its vector for doc 1"
    );
    assert_eq!(
        body_vectors.len(),
        1,
        "body_vec must keep its vector for doc 1"
    );
    assert_eq!(title_vectors[0].0, 1);
    assert_eq!(body_vectors[0].0, 1);

    // Aggregate stats must count 2 vectors total (one per field), never 1.
    let stats = index.stats().unwrap();
    assert_eq!(
        stats.vector_count, 2,
        "one doc in two fields = 2 vectors, not 1"
    );
}

/// A field-targeted query must only ever return that field's documents.
#[test]
fn field_routing_search_is_isolated() {
    let mut fields = BTreeMap::new();
    fields.insert(
        "title_vec".to_string(),
        hnsw_config(3, DistanceMetric::Cosine),
    );
    fields.insert(
        "body_vec".to_string(),
        hnsw_config(3, DistanceMetric::Cosine),
    );
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![
            (1, "title_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (2, "body_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
        ])
        .unwrap();
    writer.commit().unwrap();

    let searcher = index.searcher().unwrap();
    let results = searcher
        .search(&query(vec_of(&[1.0, 0.0, 0.0]), Some("title_vec"), 10))
        .unwrap();
    let ids: Vec<u64> = results.results.iter().map(|r| r.doc_id).collect();
    assert_eq!(ids, vec![1], "title_vec query must only return doc 1");
}

/// A field-less query fans out only to fields whose dimension matches the
/// query vector; mismatched-dimension fields must be pruned, not error out
/// or crash.
#[test]
fn fieldless_query_prunes_dimension_mismatched_fields() {
    let mut fields = BTreeMap::new();
    fields.insert(
        "small_vec".to_string(),
        hnsw_config(3, DistanceMetric::Cosine),
    );
    fields.insert(
        "big_vec".to_string(),
        hnsw_config(8, DistanceMetric::Cosine),
    );
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![
            (1, "small_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (2, "big_vec".to_string(), vec_of(&[0.0; 8])),
        ])
        .unwrap();
    writer.commit().unwrap();

    let searcher = index.searcher().unwrap();
    // 3-dimensional field-less query: only `small_vec` (dim 3) is a
    // candidate; `big_vec` (dim 8) must be pruned rather than erroring on a
    // dimension mismatch.
    let results = searcher
        .search(&query(vec_of(&[1.0, 0.0, 0.0]), None, 10))
        .unwrap();
    let ids: Vec<u64> = results.results.iter().map(|r| r.doc_id).collect();
    assert_eq!(
        ids,
        vec![1],
        "only the dimension-matching field may contribute hits"
    );
}

/// A field-less query over two same-dimension, same-metric fields merges
/// and sorts results across both by distance.
#[test]
fn fieldless_query_merges_across_homogeneous_fields() {
    let mut fields = BTreeMap::new();
    fields.insert("a_vec".to_string(), hnsw_config(3, DistanceMetric::Cosine));
    fields.insert("b_vec".to_string(), hnsw_config(3, DistanceMetric::Cosine));
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![
            (1, "a_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (2, "b_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
        ])
        .unwrap();
    writer.commit().unwrap();

    let searcher = index.searcher().unwrap();
    let results = searcher
        .search(&query(vec_of(&[1.0, 0.0, 0.0]), None, 10))
        .unwrap();
    let mut ids: Vec<u64> = results.results.iter().map(|r| r.doc_id).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![1, 2],
        "both fields must contribute to a field-less query"
    );
}

/// Issue #1343: a document holding vectors in two same-dimension fields
/// must come back exactly once from a field-less query, not once per
/// matching field. The surviving hit must be the document's best-scoring
/// field (`a_vec`, which equals the query exactly; `b_vec` is orthogonal).
#[test]
fn fieldless_query_returns_each_document_once() {
    let mut fields = BTreeMap::new();
    fields.insert("a_vec".to_string(), hnsw_config(3, DistanceMetric::Cosine));
    fields.insert("b_vec".to_string(), hnsw_config(3, DistanceMetric::Cosine));
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![
            (1, "a_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (1, "b_vec".to_string(), vec_of(&[0.0, 1.0, 0.0])),
        ])
        .unwrap();
    writer.commit().unwrap();

    let searcher = index.searcher().unwrap();
    let results = searcher
        .search(&query(vec_of(&[1.0, 0.0, 0.0]), None, 10))
        .unwrap();
    assert_eq!(
        results.results.len(),
        1,
        "doc 1 must be returned once, not once per matching field: {:?}",
        results.results
    );
    assert_eq!(results.results[0].doc_id, 1);
    assert_eq!(
        results.results[0].field_name, "a_vec",
        "the surviving hit must be the best-scoring field"
    );
}

/// Issue #1343: `top_k` on a field-less query must count documents, not
/// `(document, field)` pairs. Three documents each hold vectors in both
/// fields; `top_k = 2` must still surface 2 distinct documents rather than
/// exhausting the budget on two field-hits of the same document.
#[test]
fn fieldless_query_top_k_counts_documents() {
    let mut fields = BTreeMap::new();
    fields.insert("a_vec".to_string(), hnsw_config(3, DistanceMetric::Cosine));
    fields.insert("b_vec".to_string(), hnsw_config(3, DistanceMetric::Cosine));
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![
            (1, "a_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (1, "b_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (2, "a_vec".to_string(), vec_of(&[0.0, 1.0, 0.0])),
            (2, "b_vec".to_string(), vec_of(&[0.0, 1.0, 0.0])),
            (3, "a_vec".to_string(), vec_of(&[0.0, 0.0, 1.0])),
            (3, "b_vec".to_string(), vec_of(&[0.0, 0.0, 1.0])),
        ])
        .unwrap();
    writer.commit().unwrap();

    let searcher = index.searcher().unwrap();
    let results = searcher
        .search(&query(vec_of(&[1.0, 0.0, 0.0]), None, 2))
        .unwrap();
    let ids: Vec<u64> = results.results.iter().map(|r| r.doc_id).collect();
    assert_eq!(
        ids.len(),
        2,
        "top_k = 2 must surface 2 documents, not 2 field-hits of the same one: {ids:?}"
    );
    let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
    assert_eq!(
        unique.len(),
        ids.len(),
        "documents must not repeat: {ids:?}"
    );
    assert!(
        ids.contains(&1),
        "doc 1 (exact match in both fields) must be among the top 2: {ids:?}"
    );
}

/// `add_vectors` with an unknown field name must reject the whole batch --
/// never silently drop just the unknown field's vectors while applying the
/// known ones.
#[test]
fn unknown_field_rejects_whole_batch() {
    let mut fields = BTreeMap::new();
    fields.insert(
        "title_vec".to_string(),
        hnsw_config(3, DistanceMetric::Cosine),
    );
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    let err = writer
        .add_vectors(vec![
            (1, "title_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (1, "no_such_field".to_string(), vec_of(&[1.0, 0.0, 0.0])),
        ])
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("no_such_field"),
        "error must name the unknown field: {err:?}"
    );

    // The known field's vector must NOT have been applied either -- the
    // rejection covers the whole batch.
    writer.commit().unwrap();
    let reader = index.reader().unwrap();
    assert_eq!(reader.get_vectors_by_field("title_vec").unwrap().len(), 0);
}

/// Removing a field must not disturb the others' data, and the removed
/// field's on-disk data survives (unregister only, mirroring
/// `VectorStore::delete_field`).
#[test]
fn remove_field_does_not_affect_others() {
    let mut fields = BTreeMap::new();
    fields.insert(
        "title_vec".to_string(),
        hnsw_config(3, DistanceMetric::Cosine),
    );
    fields.insert(
        "body_vec".to_string(),
        hnsw_config(3, DistanceMetric::Cosine),
    );
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![
            (1, "title_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (2, "body_vec".to_string(), vec_of(&[0.0, 1.0, 0.0])),
        ])
        .unwrap();
    writer.commit().unwrap();

    index.remove_field("title_vec", false).unwrap();

    assert_eq!(
        index.field_dimensions().keys().collect::<Vec<_>>(),
        vec!["body_vec"],
        "removed field must no longer be routed to"
    );
    let reader = index.reader().unwrap();
    assert_eq!(reader.get_vectors_by_field("body_vec").unwrap().len(), 1);
}

/// Issue #1080: `remove_field(purge: true)` must physically delete the
/// field's on-disk data, unlike the `purge: false` case above -- a
/// same-name re-add must NOT resurrect the old vectors.
#[test]
fn remove_field_with_purge_deletes_on_disk_data() {
    let mut fields = BTreeMap::new();
    fields.insert(
        "title_vec".to_string(),
        hnsw_config(3, DistanceMetric::Cosine),
    );
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![(1, "title_vec".to_string(), vec_of(&[1.0, 0.0, 0.0]))])
        .unwrap();
    writer.commit().unwrap();

    index.remove_field("title_vec", true).unwrap();
    assert!(
        index.storage().list_files().unwrap().is_empty(),
        "purge must physically delete every file under the field's prefix"
    );

    // Re-adding under the same name must start from a clean slate.
    index
        .add_field("title_vec", hnsw_config(3, DistanceMetric::Cosine))
        .unwrap();
    let reader = index.reader().unwrap();
    assert_eq!(
        reader.get_vectors_by_field("title_vec").unwrap().len(),
        0,
        "the old vector must not resurface after a purge + re-add"
    );
}

/// Issue #1080: `rebuild_field` must preserve existing vectors while
/// applying the new config, and must reject a field name that does not
/// exist.
#[test]
fn rebuild_field_preserves_vectors_under_new_config() {
    let mut fields = BTreeMap::new();
    fields.insert(
        "title_vec".to_string(),
        hnsw_config(3, DistanceMetric::Cosine),
    );
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![
            (1, "title_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (2, "title_vec".to_string(), vec_of(&[0.0, 1.0, 0.0])),
        ])
        .unwrap();
    writer.commit().unwrap();

    let mut new_config = hnsw_config(3, DistanceMetric::Cosine);
    if let VectorIndexTypeConfig::HNSW(c) = &mut new_config {
        c.m = 32;
        c.ef_construction = 400;
    }
    index.rebuild_field("title_vec", new_config).unwrap();

    let reader = index.reader().unwrap();
    assert_eq!(
        reader.get_vectors_by_field("title_vec").unwrap().len(),
        2,
        "rebuild must preserve every existing vector, not just re-create an empty field"
    );
    assert_eq!(
        index.field_dimensions()["title_vec"],
        3,
        "rebuild must keep reporting the (unchanged) dimension"
    );
}

/// Storage decorator failing the next `create_output` whose name has the
/// armed prefix -- same technique as
/// `merge_failure_publication_test.rs::FailingStorage`, used here to kill
/// `rebuild_field`'s internal `optimize()` partway through.
#[derive(Debug)]
struct FailingStorage {
    inner: Arc<dyn Storage>,
    fail_create_with_prefix: parking_lot::Mutex<Option<String>>,
}

impl FailingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            fail_create_with_prefix: parking_lot::Mutex::new(None),
        }
    }

    fn fail_next_create_with_prefix(&self, prefix: &str) {
        *self.fail_create_with_prefix.lock() = Some(prefix.to_string());
    }
}

impl Storage for FailingStorage {
    fn create_output(&self, name: &str) -> Result<Box<dyn laurus::storage::StorageOutput>> {
        let armed = {
            let mut guard = self.fail_create_with_prefix.lock();
            if guard.as_ref().is_some_and(|p| name.starts_with(p)) {
                *guard = None;
                true
            } else {
                false
            }
        };
        if armed {
            return Err(LaurusError::storage(format!(
                "injected failure creating {name}"
            )));
        }
        self.inner.create_output(name)
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn laurus::storage::StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn open_input(&self, name: &str) -> Result<Box<dyn laurus::storage::StorageInput>> {
        self.inner.open_input(name)
    }

    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(name)
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.inner.rename_file(old_name, new_name)
    }

    fn list_files(&self) -> Result<Vec<String>> {
        self.inner.list_files()
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.inner.file_size(name)
    }

    fn sync(&self) -> Result<()> {
        self.inner.sync()
    }

    fn metadata(&self, name: &str) -> Result<laurus::storage::FileMetadata> {
        self.inner.metadata(name)
    }

    fn create_temp_output(
        &self,
        prefix: &str,
    ) -> Result<(String, Box<dyn laurus::storage::StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Issue #1080 acceptance criterion: "A failure mid-rebuild leaves the
/// original field index untouched." Kills the write of the merged
/// segment `optimize()` (inside `rebuild_field`) produces, and confirms
/// the field's original vectors and dimension are exactly as they were.
#[test]
fn rebuild_field_failure_leaves_original_field_untouched() {
    let inner: Arc<dyn Storage> = storage();
    let failing = Arc::new(FailingStorage::new(inner));
    let storage: Arc<dyn Storage> = failing.clone();

    let mut fields = BTreeMap::new();
    fields.insert(
        "title_vec".to_string(),
        hnsw_config(3, DistanceMetric::Cosine),
    );
    let index =
        MultiFieldVectorIndex::open_or_create(storage, &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![
            (1, "title_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (2, "title_vec".to_string(), vec_of(&[0.0, 1.0, 0.0])),
        ])
        .unwrap();
    writer.commit().unwrap();

    // `rebuild_field` reopens under the field's own `PrefixedStorage`, so
    // the merged segment's file name is prefixed with the field name.
    failing.fail_next_create_with_prefix("title_vec/segment_");
    let mut new_config = hnsw_config(3, DistanceMetric::Cosine);
    if let VectorIndexTypeConfig::HNSW(c) = &mut new_config {
        c.m = 32;
        c.ef_construction = 400;
    }
    let result = index.rebuild_field("title_vec", new_config);
    assert!(result.is_err(), "the injected failure must surface");

    // The field's original data and routing are completely untouched: same
    // dimension, same vectors, still searchable.
    assert_eq!(index.field_dimensions()["title_vec"], 3);
    let reader = index.reader().unwrap();
    assert_eq!(
        reader.get_vectors_by_field("title_vec").unwrap().len(),
        2,
        "a failed rebuild must not lose or partially apply the original vectors"
    );
}

#[test]
fn rebuild_field_rejects_nonexistent_field() {
    let fields = BTreeMap::new();
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let result = index.rebuild_field("nonexistent", hnsw_config(3, DistanceMetric::Cosine));
    assert!(result.is_err());
}

/// `search_batch_with_threshold` must return results in the SAME order as
/// the input queries, even when field-targeted and field-less queries are
/// interleaved -- `VectorStore::search_impl` zips this return value
/// against a same-order weights vector.
#[test]
fn batch_search_preserves_query_order() {
    let mut fields = BTreeMap::new();
    fields.insert("a_vec".to_string(), hnsw_config(3, DistanceMetric::Cosine));
    fields.insert("b_vec".to_string(), hnsw_config(3, DistanceMetric::Cosine));
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();

    let mut writer = index.writer().unwrap();
    writer
        .add_vectors(vec![
            (1, "a_vec".to_string(), vec_of(&[1.0, 0.0, 0.0])),
            (2, "b_vec".to_string(), vec_of(&[0.0, 1.0, 0.0])),
        ])
        .unwrap();
    writer.commit().unwrap();

    let searcher = index.searcher().unwrap();
    let queries = vec![
        query(vec_of(&[0.0, 1.0, 0.0]), Some("b_vec"), 10), // 0: field-targeted -> doc 2
        query(vec_of(&[1.0, 0.0, 0.0]), None, 10),          // 1: field-less -> doc 1
        query(vec_of(&[1.0, 0.0, 0.0]), Some("a_vec"), 10), // 2: field-targeted -> doc 1
    ];
    let results = searcher.search_batch(&queries).unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].results.first().map(|r| r.doc_id), Some(2));
    assert_eq!(results[1].results.first().map(|r| r.doc_id), Some(1));
    assert_eq!(results[2].results.first().map(|r| r.doc_id), Some(1));
}

/// Dynamic field addition (Issue #948 Phase 5 surface, implemented in
/// Phase 1's `VectorIndex::add_field` override): a newly added field must
/// not regress the aggregate `last_wal_seq()` (a min across fields).
#[test]
fn add_field_seeds_wal_seq_from_current_minimum() {
    let mut fields = BTreeMap::new();
    fields.insert(
        "title_vec".to_string(),
        hnsw_config(3, DistanceMetric::Cosine),
    );
    let index =
        MultiFieldVectorIndex::open_or_create(storage(), &fields, Arc::new(MockEmbedder)).unwrap();
    assert!(index.supports_dynamic_fields());

    index.set_last_wal_seq(42).unwrap();
    // `SegmentedHnswIndex::set_last_wal_seq` only records a PENDING value;
    // it is published to the manifest (and therefore visible via
    // `last_wal_seq()`) by `persist_deletions()`.
    index.persist_deletions().unwrap();
    assert_eq!(index.last_wal_seq(), 42);

    index
        .add_field("body_vec", hnsw_config(3, DistanceMetric::Cosine))
        .unwrap();
    // `add_field` seeds the new field's PENDING wal_seq with the current
    // aggregate; like any `set_last_wal_seq` call it is only published to
    // the new field's manifest by the new field's own `persist_deletions()`
    // -- which `VectorStore::commit()`'s ladder calls on every commit. This
    // stands in for that next commit: once published, the aggregate must
    // recover to 42, not regress to 0.
    index.persist_deletions().unwrap();
    assert_eq!(index.last_wal_seq(), 42);
    assert_eq!(
        index.field_dimensions().keys().collect::<Vec<_>>(),
        vec!["body_vec", "title_vec"]
    );

    // Adding a field that already exists must error, not silently replace it.
    assert!(
        index
            .add_field("body_vec", hnsw_config(3, DistanceMetric::Cosine))
            .is_err()
    );
}
