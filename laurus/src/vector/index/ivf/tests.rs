use crate::storage::Storage;
use crate::storage::memory::MemoryStorage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::vector::Vector;
use crate::vector::index::config::IvfIndexConfig;
use crate::vector::index::ivf::writer::IvfIndexWriter;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
use std::sync::Arc;

#[test]
fn test_ivf_partition_rebalancing() {
    let storage = Arc::new(MemoryStorage::default());
    let config = IvfIndexConfig {
        dimension: 2,
        distance_metric: DistanceMetric::Euclidean,
        n_clusters: 4,
        n_probe: 2,
        normalize_vectors: false,
        ..IvfIndexConfig::default()
    };
    let writer_config = VectorIndexWriterConfig::default();

    let mut writer =
        IvfIndexWriter::with_storage(config, writer_config, "test_ivf_vectors", storage).unwrap();

    // Create imbalanced clusters
    // Cluster 0: 1 vector (sparse)
    // Cluster 1: 10 vectors (dense)
    // Cluster 2: 4 vectors (normal)
    // Cluster 3: 4 vectors (normal)

    let mut vectors = Vec::new();

    // Cluster 0 area (around [0, 0])
    vectors.push((0, "f".to_string(), Vector::new(vec![0.0, 0.0])));

    // Cluster 1 area (around [100, 100]) - Dense
    for i in 0..10 {
        vectors.push((
            i + 1,
            "f".to_string(),
            Vector::new(vec![100.0 + i as f32 * 0.1, 100.0 + i as f32 * 0.1]),
        ));
    }

    // Cluster 2 area (around [0, 100])
    for i in 0..4 {
        vectors.push((
            i + 11,
            "f".to_string(),
            Vector::new(vec![0.0 + i as f32 * 0.1, 100.0 + i as f32 * 0.1]),
        ));
    }

    // Cluster 3 area (around [100, 0])
    for i in 0..4 {
        vectors.push((
            i + 15,
            "f".to_string(),
            Vector::new(vec![100.0 + i as f32 * 0.1, 0.0 + i as f32 * 0.1]),
        ));
    }

    writer.build(vectors).unwrap();
    writer.finalize().unwrap();

    let initial_stats = writer.get_cluster_stats();
    println!("Initial stats: {:?}", initial_stats);

    // Optimize should trigger merging and splitting
    // avg = 19 / 4 = 4.75
    // sparse_threshold = 4.75 / 4 = 1.18 -> Cluster 0 (1 vector) should be merged
    // dense_threshold = 4.75 * 4 = 19 -> No cluster is dense enough with factor 4?
    // Let's adjust thresholds for the test or use manual calls.

    // Manual merge test
    let merged = writer.merge_sparse_clusters(2).unwrap();
    assert!(merged > 0, "Should have merged at least one sparse cluster");

    let stats_after_merge = writer.get_cluster_stats();
    assert_eq!(stats_after_merge.len(), initial_stats.len() - merged);

    // Manual split test
    // Let's split anything > 5
    let split = writer.split_dense_clusters(5).unwrap();
    assert!(split > 0, "Should have split at least one dense cluster");

    let stats_after_split = writer.get_cluster_stats();
    assert_eq!(stats_after_split.len(), stats_after_merge.len() + split);

    // Verify all vectors are still present
    let total_vectors: usize = stats_after_split.iter().map(|s| s.count).sum();
    assert_eq!(total_vectors, 19);
}

/// Build 12 well-separated singleton clusters so that the number of vectors
/// examined during a search equals the number of probed clusters.
///
/// Returns the configured writer plus the storage backing it.
fn build_singleton_cluster_index(name: &str) -> Arc<MemoryStorage> {
    let storage = Arc::new(MemoryStorage::default());
    let config = IvfIndexConfig {
        dimension: 2,
        distance_metric: DistanceMetric::Euclidean,
        n_clusters: 12,
        // Persisted n_probe; overridden per-test for the direct-searcher path.
        n_probe: 1,
        normalize_vectors: false,
        ..IvfIndexConfig::default()
    };
    let mut writer = IvfIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        name,
        storage.clone(),
    )
    .unwrap();

    // One vector per cluster, spaced 1000 units apart so k-means++ assigns
    // exactly one vector to each of the 12 centroids.
    let vectors: Vec<(u64, String, Vector)> = (0..12)
        .map(|i| {
            (
                i as u64,
                "f".to_string(),
                Vector::new(vec![(i as f32 + 1.0) * 1000.0, 0.0]),
            )
        })
        .collect();

    writer.build(vectors).unwrap();
    writer.finalize().unwrap();
    writer.write().unwrap();

    storage
}

/// The IVF searcher must probe the configured number of clusters, not the
/// hard-coded single nearest cluster, and must not silently cap `n_probe`
/// at 10 (Issue #741).
#[test]
fn test_ivf_searcher_honors_n_probe() {
    use crate::vector::index::ivf::reader::IvfIndexReader;
    use crate::vector::index::ivf::searcher::IvfSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let storage = build_singleton_cluster_index("test_ivf_honors_n_probe");

    let reader: Arc<dyn crate::vector::reader::VectorIndexReader> = Arc::new(
        IvfIndexReader::load(
            storage,
            "test_ivf_honors_n_probe",
            DistanceMetric::Euclidean,
        )
        .unwrap(),
    );

    // Sanity: k-means produced one centroid per input vector.
    let n_centroids = reader
        .as_any()
        .downcast_ref::<IvfIndexReader>()
        .unwrap()
        .centroids()
        .len();
    assert_eq!(n_centroids, 12, "expected one centroid per input vector");

    // Query coincides with the first vector, so the nearest cluster is its
    // singleton cluster. With singleton clusters, `candidates_examined` is a
    // direct read-out of the effective `n_probe`. The `n_probe = 11` case also
    // proves the former `.min(10)` cap is gone.
    let query = Vector::new(vec![1000.0, 0.0]);
    for (n_probe, expected) in [(1usize, 1usize), (5, 5), (11, 11), (12, 12)] {
        let searcher = IvfSearcher::with_n_probe(reader.clone(), n_probe).unwrap();
        let request = VectorIndexQuery::new(query.clone()).top_k(12);
        let results = searcher.search(&request).unwrap();
        assert_eq!(
            results.candidates_examined, expected,
            "n_probe = {n_probe} should probe {expected} singleton clusters"
        );
    }
}

/// The IVF searcher must skip non-matching candidates before the distance
/// kernel when an allow-set filter is supplied (Issue #740), and leave the
/// no-filter path unchanged.
#[test]
fn test_ivf_searcher_honors_filter_inline() {
    use crate::vector::index::ivf::reader::IvfIndexReader;
    use crate::vector::index::ivf::searcher::IvfSearcher;
    use crate::vector::search::filter_set::FilterSet;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let storage = build_singleton_cluster_index("test_ivf_filter_inline");
    let reader: Arc<dyn crate::vector::reader::VectorIndexReader> = Arc::new(
        IvfIndexReader::load(storage, "test_ivf_filter_inline", DistanceMetric::Euclidean).unwrap(),
    );

    let query = Vector::new(vec![1000.0, 0.0]);
    // Probe all 12 singleton clusters so every doc is a scan candidate.
    let searcher = IvfSearcher::with_n_probe(reader, 12).unwrap();

    // No filter: every singleton cluster is scored — unchanged.
    let unfiltered = searcher
        .search(&VectorIndexQuery::new(query.clone()).top_k(12))
        .unwrap();
    assert_eq!(unfiltered.candidates_examined, 12);

    // Inline allow-set: only the allowed doc_ids reach the distance kernel.
    let allow: Arc<FilterSet> = Arc::new(FilterSet::Hash([0u64, 5, 11].into_iter().collect()));
    let filtered = searcher
        .search(
            &VectorIndexQuery::new(query.clone())
                .top_k(12)
                .filter(allow.clone()),
        )
        .unwrap();
    assert_eq!(
        filtered.candidates_examined, 3,
        "only allowed docs should reach the distance kernel"
    );
    for r in &filtered.results {
        assert!(
            allow.contains(r.doc_id),
            "result {} not in allow-set",
            r.doc_id
        );
    }
}

/// `IvfIndex::searcher()` must inherit `IvfIndexConfig::n_probe` so that
/// configured recall is actually applied at search time (Issue #741).
#[test]
fn test_ivf_index_searcher_uses_configured_n_probe() {
    use crate::vector::index::VectorIndex;
    use crate::vector::index::ivf::IvfIndex;
    use crate::vector::search::searcher::VectorIndexQuery;

    let storage = Arc::new(MemoryStorage::default());
    let config = IvfIndexConfig {
        dimension: 2,
        distance_metric: DistanceMetric::Euclidean,
        n_clusters: 12,
        n_probe: 5,
        normalize_vectors: false,
        ..IvfIndexConfig::default()
    };

    let index = IvfIndex::create(storage, "test_ivf_factory_n_probe", config).unwrap();

    let mut writer = index.writer().unwrap();
    let vectors: Vec<(u64, String, Vector)> = (0..12)
        .map(|i| {
            (
                i as u64,
                "f".to_string(),
                Vector::new(vec![(i as f32 + 1.0) * 1000.0, 0.0]),
            )
        })
        .collect();
    writer.build(vectors).unwrap();
    writer.finalize().unwrap();
    writer.write().unwrap();

    // Before the fix the factory dropped the configured n_probe and always
    // probed a single cluster (candidates_examined == 1).
    let searcher = index.searcher().unwrap();
    let query = Vector::new(vec![1000.0, 0.0]);
    let request = VectorIndexQuery::new(query).top_k(12);
    let results = searcher.search(&request).unwrap();
    assert_eq!(
        results.candidates_examined, 5,
        "IvfIndex::searcher() should probe the configured n_probe (5) clusters"
    );
}

/// Read a whole file back out of a [`MemoryStorage`] for byte-identity
/// comparisons.
fn read_file_bytes(storage: &Arc<MemoryStorage>, name: &str) -> Vec<u8> {
    use std::io::Read;
    let mut input = storage.open_input(name).unwrap();
    let mut buf = Vec::new();
    input.read_to_end(&mut buf).unwrap();
    buf
}

/// Issue #629: the CSR-permutation rewrite of `inverted_lists` must not
/// introduce any nondeterminism into `write()`'s on-disk byte layout.
/// Two independently-built writers over the same corpus (multi-field,
/// including a repeated doc_id across fields to exercise the CSR sort's
/// tie-breaking) must serialize to byte-identical `.ivf` files.
#[test]
fn write_output_is_byte_identical_across_independent_builds() {
    fn build_and_write(name: &str) -> (Arc<MemoryStorage>, Vec<u8>) {
        let storage = Arc::new(MemoryStorage::default());
        let config = IvfIndexConfig {
            dimension: 3,
            distance_metric: DistanceMetric::Euclidean,
            n_clusters: 4,
            n_probe: 2,
            normalize_vectors: false,
            ..IvfIndexConfig::default()
        };
        let mut writer = IvfIndexWriter::with_storage(
            config,
            VectorIndexWriterConfig::default(),
            name,
            storage.clone(),
        )
        .unwrap();

        let mut vectors = Vec::new();
        for i in 0..40u64 {
            let base = (i % 7) as f32 * 10.0;
            vectors.push((
                i,
                "embedding".to_string(),
                Vector::new(vec![base, base + 1.0, base + 2.0]),
            ));
            // Same doc_id, different field, landing in the same cluster as
            // its sibling above -- exercises the CSR build's stable sort
            // tie-breaking on equal doc_id.
            vectors.push((
                i,
                "summary".to_string(),
                Vector::new(vec![base, base + 1.0, base + 2.0]),
            ));
        }

        writer.build(vectors).unwrap();
        writer.finalize().unwrap();
        writer.write().unwrap();

        let bytes = read_file_bytes(&storage, &format!("{name}.ivf"));
        (storage, bytes)
    }

    let (_storage_a, bytes_a) = build_and_write("byte_identity_a");
    let (_storage_b, bytes_b) = build_and_write("byte_identity_b");

    assert!(!bytes_a.is_empty());
    assert_eq!(
        bytes_a, bytes_b,
        "identical input must serialize to identical .ivf bytes"
    );
}

/// Issue #629: `optimize()` (merge_sparse_clusters + split_dense_clusters)
/// followed by `write()` and `load()` was previously uncovered by any
/// test -- the permutation-based CSR rewrite makes this path load-bearing
/// (a wrong offset/order here would corrupt or lose records), so this
/// round-trips it and checks every vector survives with its cluster
/// membership intact.
#[test]
fn optimize_write_load_round_trip_preserves_all_vectors() {
    let storage = Arc::new(MemoryStorage::default());
    let config = IvfIndexConfig {
        dimension: 2,
        distance_metric: DistanceMetric::Euclidean,
        n_clusters: 4,
        n_probe: 2,
        normalize_vectors: false,
        ..IvfIndexConfig::default()
    };
    let writer_config = VectorIndexWriterConfig::default();

    let mut writer = IvfIndexWriter::with_storage(
        config,
        writer_config.clone(),
        "optimize_round_trip",
        storage.clone(),
    )
    .unwrap();

    // Same imbalanced-cluster shape as `test_ivf_partition_rebalancing`,
    // so `optimize()` (default thresholds) actually merges and splits.
    let mut vectors = Vec::new();
    vectors.push((0, "f".to_string(), Vector::new(vec![0.0, 0.0])));
    for i in 0..10 {
        vectors.push((
            i + 1,
            "f".to_string(),
            Vector::new(vec![100.0 + i as f32 * 0.1, 100.0 + i as f32 * 0.1]),
        ));
    }
    for i in 0..4 {
        vectors.push((
            i + 11,
            "f".to_string(),
            Vector::new(vec![0.0 + i as f32 * 0.1, 100.0 + i as f32 * 0.1]),
        ));
    }
    for i in 0..4 {
        vectors.push((
            i + 15,
            "f".to_string(),
            Vector::new(vec![100.0 + i as f32 * 0.1, 0.0 + i as f32 * 0.1]),
        ));
    }
    let expected_count = vectors.len();
    let mut expected_ids: Vec<u64> = vectors.iter().map(|(id, _, _)| *id).collect();
    expected_ids.sort_unstable();

    writer.build(vectors).unwrap();
    writer.finalize().unwrap();
    writer.optimize().unwrap();
    writer.write().unwrap();

    let loaded = IvfIndexWriter::with_storage(
        IvfIndexConfig {
            dimension: 2,
            distance_metric: DistanceMetric::Euclidean,
            n_clusters: 4,
            n_probe: 2,
            normalize_vectors: false,
            ..IvfIndexConfig::default()
        },
        writer_config,
        "optimize_round_trip",
        storage,
    )
    .unwrap();

    assert_eq!(loaded.vectors().len(), expected_count);
    let mut loaded_ids: Vec<u64> = loaded.vectors().iter().map(|(id, _, _)| *id).collect();
    loaded_ids.sort_unstable();
    assert_eq!(loaded_ids, expected_ids);

    let stats = loaded.get_cluster_stats();
    assert_eq!(
        stats.iter().map(|s| s.count).sum::<usize>(),
        expected_count,
        "cluster membership must account for every loaded vector"
    );
}
