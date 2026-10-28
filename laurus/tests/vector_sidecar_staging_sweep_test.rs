//! Issue #1532: the staging file of a rerank sidecar that a failed segment
//! write leaves is reclaimed when the index opens again.
//!
//! The Flat, HNSW and IVF writers write a segment's rerank sidecar to
//! `segment_N.{flat,hnsw,ivf}.f32.tmp` and rename it after the main segment
//! file. The orphan sweep that runs on open recognized no such name, so a
//! sidecar write that failed or crashed before its rename left the file for
//! good. A merge takes a new segment id on every attempt, so a merge that
//! fails there on every commit left one more each time.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{FileMetadata, Storage, StorageInput, StorageOutput};
use laurus::vector::DistanceMetric;
use laurus::vector::Vector;
use laurus::vector::core::rerank::RerankStorageKind;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
use laurus::vector::index::flat::segmented::SegmentedFlatIndex;
use laurus::vector::index::hnsw::segmented::SegmentedHnswIndex;
use laurus::vector::index::ivf::segmented::SegmentedIvfIndex;
use laurus::{LaurusError, Result as LaurusResult};

const DIM: usize = 4;
const INDEX_NAME: &str = "vector_index";

/// A [`Storage`] over `inner` that, once `fail` is set, fails every rename
/// of a rerank sidecar's staging file, so a segment write fails after it
/// wrote that file.
#[derive(Debug)]
struct FailSidecarRenames {
    inner: Arc<dyn Storage>,
    fail: AtomicBool,
}

impl Storage for FailSidecarRenames {
    fn create_output(&self, name: &str) -> LaurusResult<Box<dyn StorageOutput>> {
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
        if self.fail.load(Ordering::SeqCst) && old_name.ends_with(".f32.tmp") {
            return Err(LaurusError::storage(format!(
                "injected failure renaming {old_name}"
            )));
        }
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

/// The files of `storage` whose names end with `suffix`, sorted.
fn files_ending_with(storage: &dyn Storage, suffix: &str) -> Vec<String> {
    let mut files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|file| file.ends_with(suffix))
        .collect();
    files.sort();
    files
}

/// Commit two one-vector segments of an index that `open` opens with an F32
/// rerank sidecar, then fail the rename of every sidecar's staging file, so
/// the merge `optimize` runs fails after it wrote
/// `segment_N.{kind}.f32.tmp`. Opened again over the same files, the index
/// has reclaimed that file and kept the two committed segments' sidecars.
fn a_failed_sidecar_write_is_reclaimed_on_reopen(
    kind: &str,
    open: impl Fn(Arc<dyn Storage>) -> Box<dyn VectorIndex>,
) {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailSidecarRenames {
        inner: Arc::clone(&inner),
        fail: AtomicBool::new(false),
    });
    let sidecar = format!(".{kind}.f32");
    let staging = format!(".{kind}.f32.tmp");

    let index = open(failing.clone());
    for (doc_id, values) in [(1u64, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0])] {
        let mut writer = index.writer().unwrap();
        writer
            .add_vectors(vec![(
                doc_id,
                "embedding".to_string(),
                Vector::new(values.to_vec()),
            )])
            .unwrap();
        writer.commit().unwrap();
    }
    let committed_sidecars = files_ending_with(inner.as_ref(), &sidecar);
    assert_eq!(committed_sidecars.len(), 2, "{committed_sidecars:?}");

    failing.fail.store(true, Ordering::SeqCst);
    let error = index
        .optimize()
        .expect_err("the merge must fail on its sidecar's rename");
    assert!(
        error.to_string().contains("injected failure renaming"),
        "{error}"
    );
    let left = files_ending_with(inner.as_ref(), &staging);
    assert_eq!(
        left.len(),
        1,
        "the failed merge must leave {staging}: {left:?}"
    );
    drop(index);

    let _reopened = open(Arc::clone(&inner));
    assert_eq!(
        files_ending_with(inner.as_ref(), &staging),
        Vec::<String>::new(),
        "opening the index must reclaim the staging file the failed merge left"
    );
    assert_eq!(
        files_ending_with(inner.as_ref(), &sidecar),
        committed_sidecars,
        "the committed segments' sidecars must stay"
    );
}

#[test]
fn flat_reclaims_a_failed_sidecar_write_on_reopen() {
    a_failed_sidecar_write_is_reclaimed_on_reopen("flat", |storage| {
        let config = FlatIndexConfig {
            dimension: DIM,
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: false,
            rerank_storage: Some(RerankStorageKind::F32),
            ..Default::default()
        };
        Box::new(SegmentedFlatIndex::open_or_create(storage, INDEX_NAME, config).unwrap())
    });
}

#[test]
fn hnsw_reclaims_a_failed_sidecar_write_on_reopen() {
    a_failed_sidecar_write_is_reclaimed_on_reopen("hnsw", |storage| {
        let config = HnswIndexConfig {
            dimension: DIM,
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: false,
            rerank_storage: Some(RerankStorageKind::F32),
            ..Default::default()
        };
        Box::new(SegmentedHnswIndex::open_or_create(storage, INDEX_NAME, config).unwrap())
    });
}

#[test]
fn ivf_reclaims_a_failed_sidecar_write_on_reopen() {
    a_failed_sidecar_write_is_reclaimed_on_reopen("ivf", |storage| {
        let config = IvfIndexConfig {
            dimension: DIM,
            distance_metric: DistanceMetric::Cosine,
            normalize_vectors: false,
            rerank_storage: Some(RerankStorageKind::F32),
            ..Default::default()
        };
        Box::new(SegmentedIvfIndex::open_or_create(storage, INDEX_NAME, config).unwrap())
    });
}
