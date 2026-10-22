//! Issue #1500: a merge that fails after a commit's own data is durable
//! does not fail the commit.
//!
//! Both stores run their automatic segment merge after they have persisted
//! their commit and its WAL checkpoint. A merge that failed used to fail the
//! commit, which stopped the engine's commit ladder before it committed the
//! documents and truncated the WAL; a merge that fails every time then
//! failed every later commit, until its cause was gone. The commit now
//! completes and logs the failure as a warning, and the next commit retries
//! the merge.

#[path = "common/log_capture.rs"]
mod log_capture;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tempfile::TempDir;

use laurus::lexical::TextOption;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{
    FileMetadata, Storage, StorageConfig, StorageFactory, StorageInput, StorageOutput,
};
use laurus::vector::Vector;
use laurus::{
    Document, Engine, FieldOption, FlatOption, HnswOption, LaurusError, QuantizationMethod,
    QueryVector, Result as LaurusResult, Schema, SearchRequestBuilder, VectorSearchQuery,
};

const DIM: usize = 8;

/// How many captured warnings contain `needle`, from any thread: the engine
/// commits on a runtime worker. Each test's failure logs a text of its own.
fn warnings_containing(needle: &str) -> usize {
    log_capture::warnings()
        .iter()
        .filter(|(_, warning)| warning.contains(needle))
        .count()
}

fn vector(seed: usize) -> Vec<f32> {
    (0..DIM)
        .map(|i| ((seed * 7 + i) % 11) as f32 + 1.0)
        .collect()
}

/// An HNSW field with a product quantizer of `subvector_count` sub-vectors.
fn hnsw_pq(subvector_count: usize) -> FieldOption {
    FieldOption::Hnsw(
        HnswOption::new(DIM).quantizer(QuantizationMethod::ProductQuantization { subvector_count }),
    )
}

/// A schema with `option` as field `vec`, built without
/// `Schema::validate_for_create`, which rejects a `subvector_count` of 0.
fn schema_with(option: FieldOption) -> Schema {
    let mut schema = Schema::new();
    schema.fields.insert("vec".to_string(), option);
    schema
}

fn create_storage(dir: &TempDir) -> Arc<dyn Storage> {
    StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path()))).unwrap()
}

fn open_storage(dir: &TempDir) -> Arc<dyn Storage> {
    StorageFactory::open(StorageConfig::File(FileStorageConfig::new(dir.path()))).unwrap()
}

/// Bytes in the WAL, which a successful commit truncates to zero.
fn wal_bytes(dir: &TempDir) -> u64 {
    std::fs::metadata(dir.path().join("engine.wal")).map_or(0, |metadata| metadata.len())
}

/// The ids a search of field `vec` finds, sorted.
async fn search_ids(engine: &Engine) -> Vec<String> {
    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vector(1)),
            weight: 1.0,
            fields: Some(vec!["vec".into()]),
        }]))
        .limit(10)
        .build();
    let mut ids: Vec<String> = engine
        .search(request)
        .await
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    ids.sort();
    ids
}

/// An HNSW field persisted with a product quantizer `subvector_count` of 0
/// (the index opens with a warning, Issue #1491) fails every merge: deleting
/// half of its documents triggers a compaction that fails on every commit.
/// Each commit still commits the documents and truncates the WAL, and logs
/// the failure with the field and its reason.
#[tokio::test(flavor = "multi_thread")]
async fn a_vector_merge_that_always_fails_does_not_fail_the_commit() {
    log_capture::install();
    let warning = "vector field 'vec': Invalid operation: \
                   HNSW product quantization subvector_count must be at least 1";

    let dir = TempDir::new().unwrap();
    let engine = Engine::new(create_storage(&dir), schema_with(hnsw_pq(2)))
        .await
        .unwrap();
    for i in 0..4 {
        let doc = Document::builder().add_field("vec", vector(i)).build();
        engine.put_document(&format!("d{i}"), doc).await.unwrap();
    }
    engine.commit().await.unwrap();
    drop(engine);

    let broken = Schema::from_toml(&schema_with(hnsw_pq(0)).to_toml().unwrap()).unwrap();
    let engine = Engine::new(open_storage(&dir), broken.clone())
        .await
        .unwrap();
    engine.delete_documents("d0").await.unwrap();
    engine.delete_documents("d1").await.unwrap();
    for attempt in 1..=2 {
        let before = warnings_containing(warning);
        engine
            .commit()
            .await
            .unwrap_or_else(|e| panic!("commit {attempt} failed: {e}"));
        assert_eq!(wal_bytes(&dir), 0, "commit {attempt} must truncate the WAL");
        assert_eq!(
            warnings_containing(warning),
            before + 1,
            "commit {attempt} must log its failed merge with the field"
        );
    }
    assert_eq!(search_ids(&engine).await, ["d2", "d3"]);
    drop(engine);

    // Nothing is left to replay: the deletes and the documents are committed.
    let engine = Engine::new(open_storage(&dir), broken).await.unwrap();
    assert_eq!(search_ids(&engine).await, ["d2", "d3"]);
    assert_eq!(engine.get_documents("d2").await.unwrap().len(), 1);
    assert!(engine.get_documents("d0").await.unwrap().is_empty());
    engine.commit().await.unwrap();
}

/// A [`Storage`] over `inner` that fails every `create_output` of a lexical
/// merge's output, so every lexical auto-merge fails.
#[derive(Debug)]
struct FailLexicalMerges {
    inner: Arc<dyn Storage>,
    /// How many merge outputs it refused to create.
    refused: AtomicUsize,
}

impl Storage for FailLexicalMerges {
    fn create_output(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
        if name.starts_with("lexical/merged_") {
            self.refused.fetch_add(1, Ordering::SeqCst);
            return Err(LaurusError::storage(format!(
                "injected failure creating {name}"
            )));
        }
        self.inner.create_output(name)
    }

    fn create_output_append(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn open_input(&self, name: &str) -> LaurusResult<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }

    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }

    fn delete_file(&self, name: &str) -> LaurusResult<()> {
        self.inner.delete_file(name)
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> LaurusResult<()> {
        self.inner.rename_file(old_name, new_name)
    }

    fn list_files(&self) -> LaurusResult<Vec<String>> {
        self.inner.list_files()
    }

    fn file_size(&self, name: &str) -> LaurusResult<u64> {
        self.inner.file_size(name)
    }

    fn sync(&self) -> LaurusResult<()> {
        self.inner.sync()
    }

    fn metadata(&self, name: &str) -> LaurusResult<FileMetadata> {
        self.inner.metadata(name)
    }

    fn create_temp_output(&self, prefix: &str) -> LaurusResult<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }

    fn close(&mut self) -> LaurusResult<()> {
        Ok(())
    }
}

/// A lexical auto-merge that fails every time: one-document commits pass
/// the default `max_segments` of 100 at the 101st, and every merge from
/// then on fails. The lexical store commits first, so a failed merge used
/// to keep the vector store and the documents uncommitted as well. Each
/// commit now truncates the WAL and logs the failure, and the index holds
/// every document and vector exactly once.
#[tokio::test(flavor = "multi_thread")]
async fn a_lexical_merge_that_always_fails_does_not_fail_the_commit() {
    const COMMITS: usize = 102;
    log_capture::install();
    let warning = "lexical commit succeeded, but the automatic merge after it failed";

    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailLexicalMerges {
        inner: inner.clone(),
        refused: AtomicUsize::new(0),
    });
    let schema = Schema::builder()
        .add_text_field("body", TextOption::default())
        .add_flat_field("vec", FlatOption::new(DIM))
        .build();
    let engine = Engine::new(failing.clone(), schema.clone()).await.unwrap();
    let before = warnings_containing(warning);
    for i in 0..COMMITS {
        let doc = Document::builder()
            .add_text("body", format!("document {i}"))
            .add_field("vec", vector(i))
            .build();
        engine.put_document(&format!("d{i}"), doc).await.unwrap();
        engine
            .commit()
            .await
            .unwrap_or_else(|e| panic!("commit {i} failed: {e}"));
        assert_eq!(
            inner.file_size("engine.wal").unwrap_or(0),
            0,
            "commit {i} must truncate the WAL"
        );
    }
    assert_eq!(
        failing.refused.load(Ordering::SeqCst),
        COMMITS - 100,
        "every commit past max_segments must have tried, and failed, to merge"
    );
    assert_eq!(warnings_containing(warning), before + COMMITS - 100);
    drop(engine);

    let reopened = Engine::new(inner, schema).await.unwrap();
    let stats = reopened.stats().unwrap();
    assert_eq!(stats.document_count, COMMITS as u64);
    assert_eq!(stats.vector_fields["vec"].vector_count, COMMITS);
}
