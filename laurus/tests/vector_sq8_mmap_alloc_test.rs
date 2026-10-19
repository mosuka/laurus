//! Issue #1463: `Scalar8Bit` search allocates nothing per candidate.
//!
//! - With mmap-backed storage (`Lazy`) the rows are scored in place, so a
//!   search allocates exactly as often as the same search over heap rows
//!   (`Eager`). Before #1463 the `Lazy` path decoded every candidate into
//!   new buffers (three allocations per candidate).
//! - The Flat and IVF scans no longer build an empty `Vector` (an `Arc`)
//!   per scored candidate, so a search over 1,000 vectors makes far fewer
//!   than 1,000 allocations in either mode.
//!
//! The counter is a process-wide `#[global_allocator]`, so this binary
//! holds a single test, and the corpus stays below the parallel-scan
//! threshold so the scans run on the calling thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use laurus::storage::file::FileStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::vector::core::vector::Vector;
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
use laurus::vector::index::flat::FlatIndex;
use laurus::vector::index::hnsw::HnswIndex;
use laurus::vector::index::ivf::IvfIndex;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
use tempfile::TempDir;

/// Counts every allocation, reallocation included.
struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call to the system allocator unchanged.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

const DIM: usize = 64;
/// Below `PARALLEL_SCAN_THRESHOLD` (2,048), so Flat and IVF scan serially.
const COUNT: u64 = 1_000;
const FIELD: &str = "embedding";

fn values(seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..DIM)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn corpus() -> Vec<(u64, String, Vector)> {
    (0..COUNT)
        .map(|i| (i, FIELD.to_string(), Vector::new(values(i))))
        .collect()
}

/// Two storages over one directory: mmap (`Lazy`) and buffered (`Eager`).
fn storages(dir: &TempDir) -> (Arc<dyn Storage>, Arc<dyn Storage>) {
    let open = |mmap: bool| {
        let mut config = FileStorageConfig::new(dir.path());
        config.use_mmap = mmap;
        StorageFactory::create(StorageConfig::File(config)).unwrap()
    };
    (open(true), open(false))
}

/// Allocations made by one search, after a warm-up search.
fn allocations_per_search(searcher: &dyn VectorIndexSearcher, request: &VectorIndexQuery) -> u64 {
    searcher.search(request).unwrap();
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    let results = searcher.search(request).unwrap();
    let after = ALLOCATIONS.load(Ordering::SeqCst);
    assert!(!results.results.is_empty());
    drop(results);
    after - before
}

fn write(index: &dyn VectorIndex) {
    let mut writer = index.writer().unwrap();
    writer.build(corpus()).unwrap();
    writer.finalize().unwrap();
    writer.write().unwrap();
}

#[test]
fn lazy_search_allocates_as_often_as_eager_search() {
    let request = VectorIndexQuery::new(Vector::new(values(1_000_000)))
        .top_k(10)
        .field_name(FIELD.to_string());
    let metric = DistanceMetric::Cosine;
    let sq8 = QuantizationMethod::Scalar8Bit;

    let mut counts = Vec::new();
    {
        let dir = TempDir::new().unwrap();
        let (lazy, eager) = storages(&dir);
        let config = FlatIndexConfig {
            dimension: DIM,
            distance_metric: metric,
            quantization_method: sq8,
            ..Default::default()
        };
        let index = FlatIndex::create(lazy, "flat", config.clone()).unwrap();
        write(&index);
        let eager_index = FlatIndex::open(eager, "flat", config).unwrap();
        counts.push((
            "flat",
            allocations_per_search(index.searcher().unwrap().as_ref(), &request),
            allocations_per_search(eager_index.searcher().unwrap().as_ref(), &request),
        ));
    }
    {
        let dir = TempDir::new().unwrap();
        let (lazy, eager) = storages(&dir);
        let config = IvfIndexConfig {
            dimension: DIM,
            distance_metric: metric,
            n_clusters: 8,
            n_probe: 4,
            quantization_method: sq8,
            ..Default::default()
        };
        let index = IvfIndex::create(lazy, "ivf", config.clone()).unwrap();
        write(&index);
        let eager_index = IvfIndex::open(eager, "ivf", config).unwrap();
        counts.push((
            "ivf",
            allocations_per_search(index.searcher().unwrap().as_ref(), &request),
            allocations_per_search(eager_index.searcher().unwrap().as_ref(), &request),
        ));
    }
    {
        let dir = TempDir::new().unwrap();
        let (lazy, eager) = storages(&dir);
        let config = HnswIndexConfig {
            dimension: DIM,
            distance_metric: metric,
            quantization_method: sq8,
            ..Default::default()
        };
        let index = HnswIndex::create(lazy, "hnsw", config.clone()).unwrap();
        write(&index);
        let eager_index = HnswIndex::open(eager, "hnsw", config).unwrap();
        counts.push((
            "hnsw",
            allocations_per_search(index.searcher().unwrap().as_ref(), &request),
            allocations_per_search(eager_index.searcher().unwrap().as_ref(), &request),
        ));
    }

    for (index, lazy, eager) in &counts {
        eprintln!("{index}: {lazy} allocations per Lazy search, {eager} per Eager search");
    }
    for (index, lazy, eager) in counts {
        assert_eq!(lazy, eager, "{index}: Lazy allocates more than Eager");
        // Flat scores all 1,000 vectors and IVF about half of them; a
        // per-candidate allocation would put both near the candidate count.
        assert!(
            lazy < COUNT / 10,
            "{index}: {lazy} allocations per search grow with the candidates"
        );
    }
}
