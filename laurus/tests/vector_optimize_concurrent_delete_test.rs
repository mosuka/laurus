//! A soft delete marked while `optimize` force-merges a segmented vector
//! index survives the merge (Issue #1518).
//!
//! A soft delete takes no writer lock, so it can land while a full merge
//! runs: `VectorStore::optimize` and the auto-compaction inside
//! `VectorStore::commit` hold only the writer lock. Each test pauses the
//! merge, marks a document deleted, and lets the merge finish. It pauses
//! either before the merge reads any segment, or after the merged segment
//! holds every surviving document, just before the merge is published and
//! `optimize` forgets the deletions it reclaimed.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{FileMetadata, Storage, StorageInput, StorageOutput};
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::{
    FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, MultiVectorIndexConfig,
};
use laurus::vector::index::flat::segmented::SegmentedFlatIndex;
use laurus::vector::index::hnsw::segmented::SegmentedHnswIndex;
use laurus::vector::index::ivf::segmented::SegmentedIvfIndex;
use laurus::vector::index::multivector::segmented::MultiVectorIndex;
use laurus::vector::search::searcher::{
    VectorIndexQuery, VectorIndexQueryParams, VectorIndexSearcher,
};
use laurus::vector::{DistanceMetric, Vector};

/// How long either side waits for the other before failing the test.
const WAIT: Duration = Duration::from_secs(30);

/// The segment manifest every segmented vector index publishes merges to.
const MANIFEST: &str = "segments.json";

/// Where [`PausingStorage`] pauses a merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pause {
    /// Before the merge reads anything: the first `open_input` of a segment
    /// file.
    BeforeRead,
    /// After the merged segment holds every surviving document: the rename
    /// onto [`MANIFEST`] that publishes it.
    AfterCopy,
}

/// An armed pause: where it pauses, and the channels it reports and waits on.
#[derive(Debug)]
struct Armed {
    at: Pause,
    reached: Sender<()>,
    release: Receiver<()>,
}

/// Memory storage that pauses the first operation at an armed [`Pause`].
#[derive(Debug)]
struct PausingStorage {
    inner: Arc<dyn Storage>,
    armed: Mutex<Option<Armed>>,
}

impl PausingStorage {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
            armed: Mutex::new(None),
        })
    }

    /// Arm a pause at `at`.
    ///
    /// # Returns
    ///
    /// A receiver that gets a message once the operation is paused, and the
    /// sender that lets it go on.
    fn arm(&self, at: Pause) -> (Receiver<()>, Sender<()>) {
        let (reached, reached_rx) = mpsc::channel();
        let (release_tx, release) = mpsc::channel();
        *self.armed.lock().unwrap() = Some(Armed {
            at,
            reached,
            release,
        });
        (reached_rx, release_tx)
    }

    /// Pause here if a pause is armed at `at`, until the test lets it go on.
    fn pause_at(&self, at: Pause) {
        let armed = {
            let mut armed = self.armed.lock().unwrap();
            match armed.as_ref() {
                Some(pause) if pause.at == at => armed.take(),
                _ => None,
            }
        };
        if let Some(armed) = armed {
            armed.reached.send(()).unwrap();
            armed
                .release
                .recv_timeout(WAIT)
                .expect("the test did not let the paused merge go on");
        }
    }
}

impl Storage for PausingStorage {
    fn open_input(&self, name: &str) -> laurus::Result<Box<dyn StorageInput>> {
        if name.starts_with("segment_") {
            self.pause_at(Pause::BeforeRead);
        }
        self.inner.open_input(name)
    }
    fn create_output(&self, name: &str) -> laurus::Result<Box<dyn StorageOutput>> {
        self.inner.create_output(name)
    }
    fn create_output_append(&self, name: &str) -> laurus::Result<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }
    fn delete_file(&self, name: &str) -> laurus::Result<()> {
        self.inner.delete_file(name)
    }
    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }
    fn list_files(&self) -> laurus::Result<Vec<String>> {
        self.inner.list_files()
    }
    fn file_size(&self, name: &str) -> laurus::Result<u64> {
        self.inner.file_size(name)
    }
    fn rename_file(&self, from: &str, to: &str) -> laurus::Result<()> {
        if to == MANIFEST {
            self.pause_at(Pause::AfterCopy);
        }
        self.inner.rename_file(from, to)
    }
    fn metadata(&self, name: &str) -> laurus::Result<FileMetadata> {
        self.inner.metadata(name)
    }
    fn create_temp_output(&self, prefix: &str) -> laurus::Result<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }
    fn sync(&self) -> laurus::Result<()> {
        self.inner.sync()
    }
    fn close(&mut self) -> laurus::Result<()> {
        Ok(())
    }
}

/// Run `index.optimize()` on another thread, and soft delete `doc_id` while
/// the merge is paused at `pause`.
fn optimize_deleting_during_merge(
    index: &dyn VectorIndex,
    storage: &PausingStorage,
    pause: Pause,
    doc_id: u64,
) {
    let (reached, release) = storage.arm(pause);
    std::thread::scope(|scope| {
        let merge = scope.spawn(|| index.optimize());
        reached
            .recv_timeout(WAIT)
            .expect("optimize did not reach the pause");
        index.soft_delete_document(doc_id).unwrap();
        release.send(()).unwrap();
        merge.join().unwrap().unwrap();
    });
}

const FIELD: &str = "v";
const DIMENSION: usize = 16;

/// A well-mixed pseudo-random vector per id (the IVF suite's generator), so
/// no two documents tie.
fn doc_vec(i: u64) -> Vector {
    let mut state: u64 = 0x9E3779B97F4A7C15u64.wrapping_add(i.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let data: Vec<f32> = (0..DIMENSION)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Vector::new(data)
}

/// Add `ids` through a fresh writer and commit them as one sealed segment.
fn commit_batch(index: &dyn VectorIndex, ids: std::ops::Range<u64>) {
    let mut writer = index.writer().unwrap();
    let vectors: Vec<_> = ids.map(|i| (i, FIELD.to_string(), doc_vec(i))).collect();
    writer.add_vectors(vectors).unwrap();
    writer.commit().unwrap();
}

/// The ids, sorted, that `searcher` returns for a query that asks for every
/// document.
fn searched_ids(searcher: &dyn VectorIndexSearcher, probe: u64, all: usize) -> Vec<u64> {
    let query = VectorIndexQuery {
        query: doc_vec(probe),
        params: VectorIndexQueryParams {
            top_k: all,
            ..Default::default()
        },
        field_name: Some(FIELD.to_string()),
        filter: None,
    };
    let mut ids: Vec<u64> = searcher
        .search(&query)
        .unwrap()
        .results
        .iter()
        .map(|hit| hit.doc_id)
        .collect();
    ids.sort_unstable();
    ids
}

fn open_flat(storage: Arc<dyn Storage>) -> Box<dyn VectorIndex> {
    let config = FlatIndexConfig {
        dimension: DIMENSION,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    Box::new(SegmentedFlatIndex::open_or_create(storage, "vector_index", config).unwrap())
}

fn open_hnsw(storage: Arc<dyn Storage>) -> Box<dyn VectorIndex> {
    let config = HnswIndexConfig {
        dimension: DIMENSION,
        m: 16,
        ef_construction: 100,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    Box::new(SegmentedHnswIndex::open_or_create(storage, "vector_index", config).unwrap())
}

fn open_ivf(storage: Arc<dyn Storage>) -> Box<dyn VectorIndex> {
    let config = IvfIndexConfig {
        dimension: DIMENSION,
        normalize_vectors: false,
        distance_metric: DistanceMetric::Cosine,
        // Exhaustive per-segment search, as in `vector_segmented_ivf_test`.
        n_probe: 10_000,
        ..Default::default()
    };
    Box::new(SegmentedIvfIndex::open_or_create(storage, "vector_index", config).unwrap())
}

/// Two sealed segments of ten documents each; doc 3 is deleted before the
/// merge (and so reclaimed by it), doc 14 while it runs. Only membership is
/// asserted, since HNSW's graph varies between runs.
fn assert_delete_during_optimize_survives(
    open: fn(Arc<dyn Storage>) -> Box<dyn VectorIndex>,
    pause: Pause,
) {
    let storage = PausingStorage::new();
    let index = open(storage.clone());
    commit_batch(index.as_ref(), 0..10);
    commit_batch(index.as_ref(), 10..20);
    index.soft_delete_document(3).unwrap();
    index.persist_deletions().unwrap();

    optimize_deleting_during_merge(index.as_ref(), &storage, pause, 14);

    let reader = index.reader().unwrap();
    assert!(
        !reader.contains_vector(14, FIELD),
        "doc 14 was deleted while the merge ran, and must stay deleted"
    );
    assert!(!reader.contains_vector(3, FIELD));
    assert!(reader.contains_vector(15, FIELD));
    let ids = searched_ids(index.searcher().unwrap().as_ref(), 14, 20);
    assert!(!ids.contains(&14), "search returned doc 14: {ids:?}");
    assert!(ids.contains(&15), "search lost doc 15: {ids:?}");
    let stats = index.stats().unwrap();
    assert_eq!(
        stats.deleted_count, 1,
        "the merge reclaims doc 3; only doc 14's delete stays marked"
    );
    assert_eq!(
        stats.vector_count, 18,
        "every live document is counted, and only those"
    );

    // The next commit persists the deletions; a reopen still hides doc 14.
    index.persist_deletions().unwrap();
    let reopened = open(storage.clone());
    let reader = reopened.reader().unwrap();
    assert!(!reader.contains_vector(14, FIELD));
    assert!(reader.contains_vector(15, FIELD));
    assert_eq!(reopened.stats().unwrap().vector_count, 18);
}

#[test]
fn flat_delete_before_the_merge_reads_survives() {
    assert_delete_during_optimize_survives(open_flat, Pause::BeforeRead);
}

#[test]
fn flat_delete_after_the_merge_copies_survives() {
    assert_delete_during_optimize_survives(open_flat, Pause::AfterCopy);
}

#[test]
fn hnsw_delete_before_the_merge_reads_survives() {
    assert_delete_during_optimize_survives(open_hnsw, Pause::BeforeRead);
}

#[test]
fn hnsw_delete_after_the_merge_copies_survives() {
    assert_delete_during_optimize_survives(open_hnsw, Pause::AfterCopy);
}

#[test]
fn ivf_delete_before_the_merge_reads_survives() {
    assert_delete_during_optimize_survives(open_ivf, Pause::BeforeRead);
}

#[test]
fn ivf_delete_after_the_merge_copies_survives() {
    assert_delete_during_optimize_survives(open_ivf, Pause::AfterCopy);
}

/// A searcher built before `optimize` keeps hiding the documents the merge
/// reclaimed. The store keeps its cached searcher until the end of a commit
/// or an optimize, and the searcher's source-segment readers still hold
/// those documents, filtered through the bitmap they were built with.
#[test]
fn flat_searcher_built_before_optimize_keeps_hiding_reclaimed_docs() {
    let storage = PausingStorage::new();
    let index = open_flat(storage);
    commit_batch(index.as_ref(), 0..10);
    commit_batch(index.as_ref(), 10..20);
    index.soft_delete_document(3).unwrap();
    let searcher = index.searcher().unwrap();

    index.optimize().unwrap();

    let ids = searched_ids(searcher.as_ref(), 3, 20);
    assert!(
        !ids.contains(&3),
        "the old searcher returned doc 3: {ids:?}"
    );
    assert!(ids.contains(&4), "the old searcher lost doc 4: {ids:?}");
}

const TOKENS: &str = "tokens";

fn open_multi_vector(storage: Arc<dyn Storage>) -> MultiVectorIndex {
    let config = MultiVectorIndexConfig {
        dimension: 2,
        distance_metric: DistanceMetric::DotProduct,
        ..Default::default()
    };
    MultiVectorIndex::open_or_create(storage, "index", config).unwrap()
}

/// Commit `docs` as one segment; each document holds one token vector made
/// of its id.
fn commit_tokens(index: &MultiVectorIndex, docs: &[u64]) {
    let mut writer = index.writer().unwrap();
    for &doc_id in docs {
        let value = doc_id as f32;
        writer
            .add_vectors(vec![(
                doc_id,
                TOKENS.to_string(),
                Vector::new(vec![value, value]),
            )])
            .unwrap();
    }
    writer.commit().unwrap();
}

fn vectors_of(index: &MultiVectorIndex, doc_id: u64) -> Option<Vec<f32>> {
    index
        .snapshot()
        .unwrap()
        .vectors(doc_id)
        .unwrap()
        .map(|v| v.into_owned())
}

/// The multi-vector index filters deletions through its own merge; two
/// segments, doc 1 deleted before the merge and doc 3 while it runs.
fn assert_multi_vector_delete_during_optimize_survives(pause: Pause) {
    let storage = PausingStorage::new();
    let index = open_multi_vector(storage.clone());
    commit_tokens(&index, &[1, 2]);
    commit_tokens(&index, &[3, 4]);
    index.soft_delete_document(1).unwrap();
    index.persist_deletions().unwrap();

    optimize_deleting_during_merge(&index, &storage, pause, 3);

    assert_eq!(
        vectors_of(&index, 3),
        None,
        "doc 3 was deleted while the merge ran, and must stay deleted"
    );
    assert_eq!(vectors_of(&index, 1), None);
    assert_eq!(vectors_of(&index, 4), Some(vec![4.0, 4.0]));
    let stats = index.stats().unwrap();
    assert_eq!(
        stats.deleted_count, 1,
        "the merge reclaims doc 1; only doc 3's delete stays marked"
    );
    assert_eq!(
        stats.vector_count, 2,
        "every live document is counted, and only those"
    );

    index.persist_deletions().unwrap();
    let reopened = open_multi_vector(storage.clone());
    assert_eq!(vectors_of(&reopened, 3), None);
    assert_eq!(vectors_of(&reopened, 4), Some(vec![4.0, 4.0]));
    assert_eq!(reopened.stats().unwrap().vector_count, 2);
}

#[test]
fn multi_vector_delete_before_the_merge_reads_survives() {
    assert_multi_vector_delete_during_optimize_survives(Pause::BeforeRead);
}

#[test]
fn multi_vector_delete_after_the_merge_copies_survives() {
    assert_multi_vector_delete_during_optimize_survives(Pause::AfterCopy);
}

/// A snapshot taken before `optimize`, as late-interaction rescoring holds
/// one, keeps hiding the documents the merge reclaimed.
#[test]
fn multi_vector_snapshot_taken_before_optimize_keeps_hiding_reclaimed_docs() {
    let index = open_multi_vector(PausingStorage::new());
    commit_tokens(&index, &[1, 2]);
    commit_tokens(&index, &[3, 4]);
    index.soft_delete_document(1).unwrap();
    let snapshot = index.snapshot().unwrap();

    index.optimize().unwrap();

    assert_eq!(snapshot.vectors(1).unwrap(), None);
    assert_eq!(
        snapshot.vectors(2).unwrap().map(|v| v.into_owned()),
        Some(vec![2.0, 2.0])
    );
}
