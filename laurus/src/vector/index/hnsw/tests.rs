use crate::error::Result;
use crate::storage::StorageConfig;
use crate::storage::StorageFactory;
use crate::storage::memory::MemoryStorageConfig;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::rerank::RerankStorageKind;
use crate::vector::core::vector::Vector;
use crate::vector::index::VectorIndex;
use crate::vector::index::config::HnswIndexConfig;
use crate::vector::index::hnsw::HnswIndex;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::writer::HnswIndexWriter;
use crate::vector::index::rerank_sidecar::read_sidecar;
use crate::vector::writer::{VectorIndexWriter, VectorIndexWriterConfig};
use std::sync::Arc;

#[test]
fn test_hnsw_integration() -> Result<()> {
    let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    let storage = StorageFactory::create(storage_config)?;

    // HNSW Config
    let config = HnswIndexConfig {
        dimension: 3,
        m: 16,
        ef_construction: 100,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let index = HnswIndex::create(storage.clone(), "default_index", config.clone())?;
    let mut writer = index.writer()?;

    // Add vectors
    let vectors = vec![
        (1, "test".to_string(), Vector::new(vec![1.0, 0.0, 0.0])), // A
        (2, "test".to_string(), Vector::new(vec![0.0, 1.0, 0.0])), // B
        (3, "test".to_string(), Vector::new(vec![0.0, 0.0, 1.0])), // C
        (4, "test".to_string(), Vector::new(vec![0.707, 0.707, 0.0])), // Between A and B
    ];

    writer.build(vectors.clone())?;
    writer.finalize()?;
    // Note: commit is handled by VectorIndexWriter trait default which calls write("default_index")
    // Since we are using HnswIndexWriter directly via trait object or concrete?
    // index.writer() returns Box<dyn VectorIndexWriter>.
    writer.commit()?;

    // Read back
    let reader = index.reader()?;

    // Check graph loading
    use crate::vector::index::hnsw::reader::HnswIndexReader;
    let hnsw_reader = reader
        .as_any()
        .downcast_ref::<HnswIndexReader>()
        .expect("Should be HnswIndexReader");
    assert!(hnsw_reader.graph.is_some());

    // Search using Graph
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let searcher = HnswSearcher::new(reader.clone())?;

    // Query close to A (1,0,0)
    let query = Vector::new(vec![0.9, 0.1, 0.0]);
    let request = VectorIndexQuery::new(query)
        .top_k(1)
        .field_name("test".to_string());

    let results = searcher.search(&request)?;

    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results[0].doc_id, 1);

    // Stage 2 (Issue #481): rerank_factor against a Stage 1 segment
    // (no sidecar) must silently degrade to Stage 1 ranking — there
    // is no f32 information to recover, so returning an error would
    // be a worse experience than just returning the int8 ranking.
    let rerank_request = VectorIndexQuery::new(Vector::new(vec![0.9, 0.1, 0.0]))
        .top_k(1)
        .field_name("test".to_string())
        .rerank_factor(2);
    let degraded = searcher.search(&rerank_request)?;
    assert_eq!(degraded.results.len(), 1);
    assert_eq!(degraded.results[0].doc_id, 1);

    Ok(())
}

/// Issue #656: speculative two-hop prefetch lookahead must never change
/// search results — it only issues hardware prefetch hints (no
/// dereference, no mutation of `candidates`/`found`/`visited`). This test
/// uses a corpus large enough (300 points) that `ef_search`'s default of
/// 50 forces many heap pops in `search_graph`, giving `candidates.peek()`
/// real, varied data across many iterations, unlike `test_hnsw_integration`'s
/// 4-point corpus. `HnswIndexConfig::default()` (`Scalar8Bit` +
/// `InMemory`) is exactly the configuration that makes `prefetch_enabled`
/// true and exercises the new lookahead code in both the pristine branch
/// (plain query) and the bookkeeping branch (filtered query) below.
#[test]
fn two_hop_prefetch_lookahead_does_not_change_search_results() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = HnswIndexConfig {
        dimension: 3,
        m: 16,
        ef_construction: 100,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    assert_eq!(
        config.quantization_method,
        crate::QuantizationMethod::Scalar8Bit,
        "this test relies on the default (Scalar8Bit + InMemory) exercising \
         prefetch_enabled = true"
    );

    let index = HnswIndex::create(storage.clone(), "default_index", config.clone())?;
    let mut writer = index.writer()?;

    // 300 points spread widely apart, plus one point placed very close to
    // the query -- far enough from every other point that the true nearest
    // neighbor is unambiguous regardless of the (randomized) graph build,
    // matching `test_hnsw_integration`'s own exact-match precedent above.
    let mut vectors = Vec::with_capacity(301);
    for i in 0..300u64 {
        let angle = i as f32;
        vectors.push((
            i,
            "test".to_string(),
            Vector::new(vec![angle.cos() * 100.0, angle.sin() * 100.0, i as f32]),
        ));
    }
    vectors.push((300, "test".to_string(), Vector::new(vec![0.9, 0.1, 0.0])));

    writer.build(vectors)?;
    writer.finalize()?;
    writer.commit()?;

    let reader = index.reader()?;
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};
    let searcher = HnswSearcher::new(reader.clone())?;

    let query = Vector::new(vec![0.9, 0.1, 0.0]);

    // Pristine branch (no filter): the planted near point must be found.
    let request = VectorIndexQuery::new(query.clone())
        .top_k(1)
        .field_name("test".to_string());
    let results = searcher.search(&request)?;
    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results[0].doc_id, 300);

    // Bookkeeping branch (filter present): restrict the allow-set to a
    // sparse handful of ids, including the planted near point, so the
    // frontier must cross rejected candidates to reach it (#645
    // connectivity guarantee) -- this exercises `search_graph`'s other
    // branch with the same lookahead code active.
    use crate::vector::search::filter_set::FilterSet;
    use ahash::AHashSet;
    use std::sync::Arc;
    let allow: AHashSet<u64> = [5u64, 50, 150, 250, 300].into_iter().collect();
    let filtered_request = VectorIndexQuery::new(query)
        .top_k(1)
        .field_name("test".to_string())
        .filter(Arc::new(FilterSet::Hash(allow)));
    let filtered_results = searcher.search(&filtered_request)?;
    assert_eq!(filtered_results.results.len(), 1);
    assert_eq!(filtered_results.results[0].doc_id, 300);

    Ok(())
}

#[test]
fn writer_omits_sidecar_when_rerank_storage_is_none() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = HnswIndexConfig {
        dimension: 3,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        rerank_storage: None,
        ..Default::default()
    };
    let mut writer = HnswIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "stage1_only",
        Arc::clone(&storage),
    )?;
    writer.add_vectors(vec![
        (1, "f".to_string(), Vector::new(vec![1.0, 0.0, 0.0])),
        (2, "f".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
    ])?;
    writer.finalize()?;
    writer.write()?;

    assert!(
        storage.file_exists("stage1_only.hnsw"),
        "main LVS1 segment must exist"
    );
    assert!(
        !storage.file_exists("stage1_only.hnsw.f32"),
        "no sidecar should be written when rerank_storage is None"
    );
    Ok(())
}

#[test]
fn writer_emits_sidecar_with_matching_header_when_rerank_storage_is_f32() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 3;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let mut writer = HnswIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "stage2_f32",
        Arc::clone(&storage),
    )?;

    let originals = vec![
        (1u64, "f".to_string(), Vector::new(vec![0.1, 0.2, 0.3])),
        (2u64, "f".to_string(), Vector::new(vec![-1.0, 0.5, 0.25])),
        (3u64, "f".to_string(), Vector::new(vec![0.7, -0.7, 0.0])),
    ];
    writer.add_vectors(originals.clone())?;
    writer.finalize()?;
    writer.write()?;

    assert!(
        storage.file_exists("stage2_f32.hnsw.f32"),
        "sidecar must be written when rerank_storage is Some(F32)"
    );
    let mut sidecar_in = storage.open_input("stage2_f32.hnsw.f32")?;
    let sidecar_size = sidecar_in.size()?;
    let (header, payload) = read_sidecar(&mut sidecar_in, sidecar_size)?;
    assert_eq!(header.dim as usize, dim);
    assert_eq!(header.vector_count as usize, originals.len());
    assert_eq!(header.storage_kind, RerankStorageKind::F32);
    assert_eq!(payload.len(), originals.len() * dim * 4);

    // Sidecar order matches the LVS1 sort-by-doc_id order, so the
    // first record must be doc_id 1 with values [0.1, 0.2, 0.3].
    let first_x = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let first_y = f32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
    let first_z = f32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]);
    assert!((first_x - 0.1).abs() < f32::EPSILON);
    assert!((first_y - 0.2).abs() < f32::EPSILON);
    assert!((first_z - 0.3).abs() < f32::EPSILON);
    Ok(())
}

#[test]
fn reader_loads_rerank_storage_when_sidecar_present() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 3;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let originals = vec![
        (1u64, "f".to_string(), Vector::new(vec![0.1, 0.2, 0.3])),
        (2u64, "f".to_string(), Vector::new(vec![-1.0, 0.5, 0.25])),
    ];
    let mut writer = HnswIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "stage2_reader",
        Arc::clone(&storage),
    )?;
    writer.add_vectors(originals.clone())?;
    writer.finalize()?;
    writer.write()?;

    let reader = HnswIndexReader::load(
        Arc::clone(&storage),
        "stage2_reader",
        DistanceMetric::Cosine,
    )?;
    let pool = reader
        .rerank_storage()
        .expect("rerank_storage must be Some when sidecar exists in Eager mode");
    assert_eq!(pool.dim, dim);
    assert_eq!(pool.vector_count, originals.len());
    assert_eq!(pool.kind, RerankStorageKind::F32);
    let v0 = pool
        .get_f32_slice(1, "f")
        .expect("doc 1 must be present in rerank pool");
    assert_eq!(v0, &[0.1, 0.2, 0.3]);
    let v1 = pool
        .get_f32_slice(2, "f")
        .expect("doc 2 must be present in rerank pool");
    assert_eq!(v1, &[-1.0, 0.5, 0.25]);
    Ok(())
}

#[test]
fn reader_rerank_storage_is_none_for_stage1_segment() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 3;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: None,
        ..Default::default()
    };
    let mut writer = HnswIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "stage1_reader",
        Arc::clone(&storage),
    )?;
    writer.add_vectors(vec![
        (1u64, "f".to_string(), Vector::new(vec![1.0, 0.0, 0.0])),
        (2u64, "f".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
    ])?;
    writer.finalize()?;
    writer.write()?;

    let reader = HnswIndexReader::load(
        Arc::clone(&storage),
        "stage1_reader",
        DistanceMetric::Cosine,
    )?;
    assert!(
        reader.rerank_storage().is_none(),
        "Stage 1 segment (no sidecar) must yield rerank_storage = None"
    );
    Ok(())
}

#[test]
fn searcher_returns_exact_f32_distance_when_rerank_storage_is_loaded() -> Result<()> {
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 4;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 32,
        distance_metric: DistanceMetric::Euclidean,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let originals = vec![
        (
            1u64,
            "f".to_string(),
            Vector::new(vec![0.123_456, 0.234_567, 0.345_678, 0.456_789]),
        ),
        (
            2u64,
            "f".to_string(),
            Vector::new(vec![0.987_654, 0.876_543, 0.765_432, 0.654_321]),
        ),
        (
            3u64,
            "f".to_string(),
            Vector::new(vec![-0.111_111, -0.222_222, -0.333_333, -0.444_444]),
        ),
    ];

    let index = HnswIndex::create(Arc::clone(&storage), "rerank_search", config)?;
    let mut writer = index.writer()?;
    writer.build(originals.clone())?;
    writer.finalize()?;
    writer.commit()?;

    let reader = index.reader()?;
    let mut searcher = HnswSearcher::new(reader)?;
    searcher.set_ef_search(50);

    // Query equal to doc 1's vector -> exact f32 distance must be 0.
    let request = VectorIndexQuery::new(Vector::new(vec![
        0.123_456, 0.234_567, 0.345_678, 0.456_789,
    ]))
    .top_k(1)
    .field_name("f".to_string())
    .rerank_factor(3);
    let results = searcher.search(&request)?;

    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results[0].doc_id, 1, "doc 1 must be top match");
    // Stage 1 (int8) returns a small but non-zero approximation for
    // the self-distance because of quantization noise. Stage 2 with
    // rerank rescores against the original f32 vectors and must
    // recover the exact zero.
    assert_eq!(
        results.results[0].distance, 0.0,
        "rerank must restore the exact f32 self-distance, got {}",
        results.results[0].distance
    );
    Ok(())
}

#[test]
fn searcher_silently_falls_back_to_stage1_when_rerank_storage_absent() -> Result<()> {
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 3;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        rerank_storage: None,
        ..Default::default()
    };
    let index = HnswIndex::create(Arc::clone(&storage), "stage1_with_rerank_request", config)?;
    let mut writer = index.writer()?;
    writer.build(vec![
        (1u64, "f".to_string(), Vector::new(vec![1.0, 0.0, 0.0])),
        (2u64, "f".to_string(), Vector::new(vec![0.0, 1.0, 0.0])),
    ])?;
    writer.finalize()?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = HnswSearcher::new(reader)?;

    // Stage 1 segment + rerank_factor request must succeed (no
    // NotImplemented) and return the int8 ranking.
    let request = VectorIndexQuery::new(Vector::new(vec![1.0, 0.0, 0.0]))
        .top_k(1)
        .field_name("f".to_string())
        .rerank_factor(5);
    let results = searcher.search(&request)?;
    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results[0].doc_id, 1);
    Ok(())
}

#[test]
fn writer_load_round_trips_byte_exact_via_sidecar() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let dim = 4;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    let originals = vec![
        (
            10u64,
            "f".to_string(),
            Vector::new(vec![0.123_456, -0.987_654, 1.111_222, 0.000_001]),
        ),
        (
            20u64,
            "f".to_string(),
            Vector::new(vec![-1.5, 0.5, 0.25, -0.75]),
        ),
    ];

    {
        let mut writer = HnswIndexWriter::with_storage(
            config.clone(),
            VectorIndexWriterConfig::default(),
            "stage2_round_trip",
            Arc::clone(&storage),
        )?;
        writer.add_vectors(originals.clone())?;
        writer.finalize()?;
        writer.write()?;
    }

    let loaded = HnswIndexWriter::load(
        config,
        VectorIndexWriterConfig::default(),
        Arc::clone(&storage),
        "stage2_round_trip",
    )?;
    let loaded_vectors = loaded.vectors();
    assert_eq!(loaded_vectors.len(), originals.len());

    for (orig, got) in originals.iter().zip(loaded_vectors.iter()) {
        assert_eq!(orig.0, got.0, "doc_id");
        assert_eq!(orig.1, got.1, "field name");
        assert_eq!(
            orig.2.data, got.2.data,
            "f32 payload must round-trip byte-exact via the LRS1 sidecar"
        );
    }
    Ok(())
}

/// A corrupted sidecar must fail `HnswIndexWriter::load` (Issue #788).
///
/// This is the anti-laundering guarantee: if the writer-reload path
/// accepted a corrupted sidecar, the broken f32 values would enter the
/// writer's in-memory state and be re-emitted with a fresh, valid CRC
/// on the next commit — silently converting detectable corruption into
/// undetectable corruption.
#[test]
fn writer_load_rejects_corrupted_sidecar() -> Result<()> {
    use std::io::{Read, Write};

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = HnswIndexConfig {
        dimension: 4,
        m: 4,
        ef_construction: 16,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        rerank_storage: Some(RerankStorageKind::F32),
        ..Default::default()
    };
    {
        let mut writer = HnswIndexWriter::with_storage(
            config.clone(),
            VectorIndexWriterConfig::default(),
            "stage2_corrupt",
            Arc::clone(&storage),
        )?;
        writer.add_vectors(vec![
            (1u64, "f".to_string(), Vector::new(vec![1.0, 0.0, 0.0, 0.0])),
            (2u64, "f".to_string(), Vector::new(vec![0.0, 1.0, 0.0, 0.0])),
        ])?;
        writer.finalize()?;
        writer.write()?;
    }

    // Flip one byte in the middle of the sidecar payload.
    let sidecar_name = "stage2_corrupt.hnsw.f32";
    let mut bytes = Vec::new();
    storage
        .open_input(sidecar_name)?
        .read_to_end(&mut bytes)
        .map_err(crate::error::LaurusError::from)?;
    let payload_mid = crate::vector::index::rerank_sidecar::HEADER_SIZE
        + (bytes.len()
            - crate::vector::index::rerank_sidecar::HEADER_SIZE
            - crate::vector::index::rerank_sidecar::FOOTER_SIZE)
            / 2;
    bytes[payload_mid] ^= 0xff;
    let mut out = storage.create_output(sidecar_name)?;
    out.write_all(&bytes)
        .map_err(crate::error::LaurusError::from)?;
    out.close()?;

    let result = HnswIndexWriter::load(
        config,
        VectorIndexWriterConfig::default(),
        Arc::clone(&storage),
        "stage2_corrupt",
    );
    // Pin the failure to the CRC check: the load path also has
    // dim/count-mismatch (InvalidOperation) and Io exits right after
    // read_sidecar, and those must not satisfy this test.
    match result {
        Err(crate::error::LaurusError::Index(msg)) => {
            assert!(
                msg.contains("checksum mismatch"),
                "expected a checksum mismatch, got: {msg}"
            );
        }
        Err(other) => panic!(
            "a corrupted .hnsw.f32 must fail the writer reload path \
             with a checksum error, got {other:?}"
        ),
        Ok(_) => panic!(
            "a corrupted .hnsw.f32 must be rejected by the writer \
             reload path, got Ok"
        ),
    }
    Ok(())
}

#[test]
fn test_hnsw_pq_search_returns_corpus_neighbour() -> Result<()> {
    use crate::vector::core::quantization::QuantizationMethod;
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    let storage = StorageFactory::create(storage_config)?;

    // Two well-separated clusters in 4-D Euclidean space. M=2 → sub_dim=2.
    let config = HnswIndexConfig {
        dimension: 4,
        m: 8,
        ef_construction: 50,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantization { subvector_count: 2 },
        ..Default::default()
    };

    let index = HnswIndex::create(storage.clone(), "pq_round_trip", config.clone())?;
    let mut writer = index.writer()?;

    // Two widely-separated clusters with 128 points each — 256 total,
    // meeting the PQ min-train threshold (#880: segments with fewer vectors
    // than the 256 k-means centroids are written as Scalar8Bit, so a
    // smaller corpus would silently stop exercising the PQ path this test
    // exists for). The large cluster separation keeps the quantiser stable
    // across platforms (issue #730: platform-dependent f32 reduction order
    // in k-means could otherwise flip a near/far quantisation code).
    //
    // Near cluster (doc_ids 1..=128) sits around (10, 10, 20, 20); far
    // cluster (doc_ids 129..=256) sits around (-100, -100, -200, -200).
    const POINTS_PER_CLUSTER: usize = 128;
    let offset = |i: usize| -> [f32; 4] {
        [
            ((i % 8) as f32) * 0.04 - 0.14,
            ((i / 8 % 8) as f32) * 0.04 - 0.14,
            ((i / 64 % 8) as f32) * 0.04 - 0.14,
            ((i % 16) as f32) * 0.04 - 0.32,
        ]
    };
    let near_base = [10.0_f32, 10.0, 20.0, 20.0];
    let far_base = [-100.0_f32, -100.0, -200.0, -200.0];

    let mut vectors = Vec::with_capacity(2 * POINTS_PER_CLUSTER);
    for i in 0..POINTS_PER_CLUSTER {
        let off = offset(i);
        let v: Vec<f32> = near_base.iter().zip(&off).map(|(b, o)| b + o).collect();
        vectors.push(((i + 1) as u64, "embedding".to_string(), Vector::new(v)));
    }
    for i in 0..POINTS_PER_CLUSTER {
        let off = offset(i);
        let v: Vec<f32> = far_base.iter().zip(&off).map(|(b, o)| b + o).collect();
        vectors.push((
            (i + 1 + POINTS_PER_CLUSTER) as u64,
            "embedding".to_string(),
            Vector::new(v),
        ));
    }

    writer.build(vectors.clone())?;
    writer.finalize()?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = HnswSearcher::new(reader)?;

    // Query at the near cluster centre — every top-3 result must come from
    // the near cluster (doc_ids 1..=128), never the far cluster. The exact
    // ordering within the near cluster is not asserted because PQ is
    // approximate; only cluster membership is guaranteed by the large
    // separation.
    let query = Vector::new(vec![10.0, 10.0, 20.0, 20.0]);
    let request = VectorIndexQuery::new(query)
        .top_k(3)
        .field_name("embedding".to_string());
    let results = searcher.search(&request)?;
    assert_eq!(results.results.len(), 3, "expected top-3 results");
    let ids: std::collections::HashSet<u64> = results.results.iter().map(|r| r.doc_id).collect();
    for id in &ids {
        assert!(
            (1..=POINTS_PER_CLUSTER as u64).contains(id),
            "top-3 must all be near-cluster doc_ids (1..=128); got {ids:?}",
        );
    }
    Ok(())
}

/// Issue #631: a segment below `PQ_MIN_TRAIN_VECTORS` (256) normally
/// degrades to Scalar8Bit (nothing to train k-means on) -- but when a
/// shared codebook is configured there is nothing to train either way,
/// so the degradation rationale no longer applies and the segment
/// should stay Product Quantization, encoding directly against the
/// shared codebook.
#[test]
fn shared_pq_codebook_keeps_small_segment_on_pq() -> Result<()> {
    use crate::vector::core::quantization::QuantizationMethod;
    use crate::vector::index::pq_codebook::train_and_write_pq_codebook;
    use crate::vector::index::storage::VectorStorage;

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;

    // Train the shared codebook on a representative sample well above the
    // min-train threshold, entirely separate from the tiny corpus the
    // segment itself will hold.
    let mut state: u64 = 0xABCD_EF01_2345_6789;
    let training_sample: Vec<Vector> = (0..300)
        .map(|_| {
            let data: Vec<f32> = (0..4)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            Vector::new(data)
        })
        .collect();
    train_and_write_pq_codebook(
        storage.as_ref(),
        "shared.pqcb",
        4,
        2,
        256,
        false,
        &training_sample,
    )?;

    let mut config = HnswIndexConfig {
        dimension: 4,
        m: 8,
        ef_construction: 50,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantization { subvector_count: 2 },
        pq_codebook_path: Some("shared.pqcb".to_string()),
        ..Default::default()
    };
    config.resolve_pq_codebook(storage.as_ref())?;
    assert!(
        config.pq_codebook.is_some(),
        "the shared codebook must resolve from the file just written"
    );

    let index = HnswIndex::create(storage.clone(), "small_segment", config)?;
    let mut writer = index.writer()?;
    // Only 10 vectors -- far below PQ_MIN_TRAIN_VECTORS (256).
    let vectors: Vec<_> = (0..10u64)
        .map(|i| {
            (
                i + 1,
                "embedding".to_string(),
                Vector::new(vec![i as f32, i as f32, i as f32 * 2.0, i as f32 * 2.0]),
            )
        })
        .collect();
    writer.build(vectors)?;
    writer.finalize()?;
    writer.commit()?;

    let reader = index.reader()?;
    let hnsw_reader = reader
        .as_any()
        .downcast_ref::<HnswIndexReader>()
        .expect("HnswIndex::reader() always returns an HnswIndexReader");
    assert!(
        matches!(hnsw_reader.vectors(), VectorStorage::OwnedPq(_)),
        "a 10-vector segment with a shared codebook configured must stay \
         Product Quantization, not degrade to Scalar8Bit"
    );
    Ok(())
}

/// Issue #631: a `pq_codebook_path` naming a file that has not been
/// trained yet must not block opening the index (the schema may be
/// created before `laurus train pq-codebook` runs) -- but `write()`
/// must fail loudly, with the exact command to fix it, rather than
/// silently falling back to per-segment training (which would defeat
/// the point of configuring a shared codebook at all).
#[test]
fn missing_shared_pq_codebook_is_lenient_at_open_but_errors_at_write() -> Result<()> {
    use crate::vector::core::quantization::QuantizationMethod;

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let mut config = HnswIndexConfig {
        dimension: 4,
        m: 8,
        ef_construction: 50,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantization { subvector_count: 2 },
        pq_codebook_path: Some("not-yet-trained.pqcb".to_string()),
        ..Default::default()
    };
    // Lenient at open: no file exists yet, but resolution must not error.
    config.resolve_pq_codebook(storage.as_ref())?;
    assert!(config.pq_codebook.is_none());

    let index = HnswIndex::create(storage.clone(), "no_codebook_yet", config)?;
    let mut writer = index.writer()?;
    let vectors: Vec<_> = (0..300u64)
        .map(|i| {
            (
                i + 1,
                "embedding".to_string(),
                Vector::new(vec![i as f32, i as f32, i as f32 * 2.0, i as f32 * 2.0]),
            )
        })
        .collect();
    writer.build(vectors)?;
    writer.finalize()?;

    // A path is configured but never resolved to an actual codebook --
    // `write()` must fail loudly (naming the configured path and the fix)
    // rather than silently falling back to training a fresh codebook,
    // which would defeat the point of configuring a shared codebook.
    let err = writer.write().expect_err(
        "write() must reject a configured-but-unresolved pq_codebook_path instead of \
         silently training a fresh codebook",
    );
    let message = err.to_string();
    assert!(
        message.contains("not-yet-trained.pqcb"),
        "error must name the configured path, got: {message}"
    );
    Ok(())
}

/// Issue #920 (FastScan mirror of `shared_pq_codebook_keeps_small_segment_on_pq`):
/// a segment below `PQ_FASTSCAN_MIN_TRAIN_VECTORS` (16) normally
/// degrades to Scalar8Bit, but with a shared k=16 codebook configured
/// nothing is trained, so the segment must stay FastScan and encode
/// against the shared codebook.
#[cfg(feature = "pq-fastscan")]
#[test]
fn shared_pq_codebook_keeps_small_segment_on_fastscan() -> Result<()> {
    use crate::vector::core::quantization::QuantizationMethod;
    use crate::vector::index::pq_codebook::train_and_write_pq_codebook;
    use crate::vector::index::storage::VectorStorage;

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;

    let mut state: u64 = 0xABCD_EF01_2345_6789;
    let training_sample: Vec<Vector> = (0..300)
        .map(|_| {
            let data: Vec<f32> = (0..4)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            Vector::new(data)
        })
        .collect();
    train_and_write_pq_codebook(
        storage.as_ref(),
        "shared-fs.pqcb",
        4,
        2,
        16,
        false,
        &training_sample,
    )?;

    let mut config = HnswIndexConfig {
        dimension: 4,
        m: 8,
        ef_construction: 50,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantizationFastScan { subvector_count: 2 },
        pq_codebook_path: Some("shared-fs.pqcb".to_string()),
        ..Default::default()
    };
    config.resolve_pq_codebook(storage.as_ref())?;
    assert!(
        config.pq_codebook.is_some(),
        "the shared k=16 codebook must resolve from the file just written"
    );

    let index = HnswIndex::create(storage.clone(), "small_fs_segment", config)?;
    let mut writer = index.writer()?;
    // Only 10 vectors -- below PQ_FASTSCAN_MIN_TRAIN_VECTORS (16).
    let vectors: Vec<_> = (0..10u64)
        .map(|i| {
            (
                i + 1,
                "embedding".to_string(),
                Vector::new(vec![i as f32, i as f32, i as f32 * 2.0, i as f32 * 2.0]),
            )
        })
        .collect();
    writer.build(vectors)?;
    writer.finalize()?;
    writer.commit()?;

    let reader = index.reader()?;
    let hnsw_reader = reader
        .as_any()
        .downcast_ref::<HnswIndexReader>()
        .expect("HnswIndex::reader() always returns an HnswIndexReader");
    assert!(
        matches!(hnsw_reader.vectors(), VectorStorage::OwnedPqFastScan(_)),
        "a 10-vector segment with a shared k=16 codebook configured must stay \
         FastScan, not degrade to Scalar8Bit"
    );
    Ok(())
}

/// Issue #920 (FastScan mirror of
/// `missing_shared_pq_codebook_is_lenient_at_open_but_errors_at_write`):
/// a configured-but-untrained `pq_codebook_path` on a FastScan field
/// must fail loudly at write — before this fix it was silently ignored
/// and every segment retrained k-means.
#[cfg(feature = "pq-fastscan")]
#[test]
fn missing_shared_pq_codebook_errors_at_write_for_fastscan() -> Result<()> {
    use crate::vector::core::quantization::QuantizationMethod;

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let mut config = HnswIndexConfig {
        dimension: 4,
        m: 8,
        ef_construction: 50,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantizationFastScan { subvector_count: 2 },
        pq_codebook_path: Some("not-yet-trained-fs.pqcb".to_string()),
        ..Default::default()
    };
    config.resolve_pq_codebook(storage.as_ref())?;
    assert!(config.pq_codebook.is_none());

    let index = HnswIndex::create(storage.clone(), "no_fs_codebook_yet", config)?;
    let mut writer = index.writer()?;
    let vectors: Vec<_> = (0..300u64)
        .map(|i| {
            (
                i + 1,
                "embedding".to_string(),
                Vector::new(vec![i as f32, i as f32, i as f32 * 2.0, i as f32 * 2.0]),
            )
        })
        .collect();
    writer.build(vectors)?;
    writer.finalize()?;

    let err = writer.write().expect_err(
        "write() must reject a configured-but-unresolved pq_codebook_path on a \
         FastScan field instead of silently training a fresh codebook",
    );
    let message = err.to_string();
    assert!(
        message.contains("not-yet-trained-fs.pqcb"),
        "error must name the configured path, got: {message}"
    );
    Ok(())
}

/// Issue #920: a k=256 (standard PQ) codebook configured on a FastScan
/// field must be rejected loudly at write with the k mismatch named.
#[cfg(feature = "pq-fastscan")]
#[test]
fn k256_codebook_on_fastscan_field_errors_at_write() -> Result<()> {
    use crate::vector::core::quantization::QuantizationMethod;
    use crate::vector::index::pq_codebook::train_and_write_pq_codebook;

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;

    let mut state: u64 = 0xABCD_EF01_2345_6789;
    let training_sample: Vec<Vector> = (0..300)
        .map(|_| {
            let data: Vec<f32> = (0..4)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            Vector::new(data)
        })
        .collect();
    // Standard-PQ (k=256) codebook...
    train_and_write_pq_codebook(
        storage.as_ref(),
        "wrong-k.pqcb",
        4,
        2,
        256,
        false,
        &training_sample,
    )?;

    // ...configured on a FastScan (k=16) field.
    let mut config = HnswIndexConfig {
        dimension: 4,
        m: 8,
        ef_construction: 50,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantizationFastScan { subvector_count: 2 },
        pq_codebook_path: Some("wrong-k.pqcb".to_string()),
        ..Default::default()
    };
    config.resolve_pq_codebook(storage.as_ref())?;

    let index = HnswIndex::create(storage.clone(), "wrong_k_fs", config)?;
    let mut writer = index.writer()?;
    let vectors: Vec<_> = (0..300u64)
        .map(|i| {
            (
                i + 1,
                "embedding".to_string(),
                Vector::new(vec![i as f32, i as f32, i as f32 * 2.0, i as f32 * 2.0]),
            )
        })
        .collect();
    writer.build(vectors)?;
    writer.finalize()?;

    let err = writer
        .write()
        .expect_err("a k=256 codebook must not encode a FastScan field");
    let message = err.to_string();
    assert!(
        message.contains("k = 256") && message.contains("k = 16"),
        "error must name both the stored and required k, got: {message}"
    );
    Ok(())
}

/// Regression / parity test for Issue #644: HNSW `ef_search` must honour
/// the per-query override and the schema-level default, instead of being
/// permanently capped at the historical `50` constant.
#[test]
fn hnsw_searcher_honours_per_query_and_schema_ef_search() -> Result<()> {
    use crate::vector::index::hnsw::searcher::HnswSearcher;
    use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = HnswIndexConfig {
        dimension: 3,
        m: 16,
        ef_construction: 64,
        // Schema-level default lifts the searcher's fallback well above
        // the legacy hardcoded `50`.
        default_ef_search: Some(300),
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    let index = HnswIndex::create(storage.clone(), "ef_test", config)?;
    let mut writer = index.writer()?;
    let vectors = (0..32u64)
        .map(|i| {
            let mut v = vec![0.0_f32; 3];
            v[(i as usize) % 3] = 1.0 + (i as f32) * 0.01;
            (i, "vec".to_string(), Vector::new(v))
        })
        .collect::<Vec<_>>();
    writer.build(vectors)?;
    writer.finalize()?;
    writer.commit()?;

    // The searcher built via `HnswIndex::searcher()` must pick up the
    // schema-level `default_ef_search` (Issue #644).
    let searcher = index.searcher()?;
    let request = VectorIndexQuery::new(Vector::new(vec![1.0, 0.0, 0.0]))
        .top_k(5)
        .field_name("vec".to_string());
    let results = searcher.search(&request)?;
    assert!(
        !results.results.is_empty(),
        "expected non-empty results from schema-default search path"
    );

    // Construct a direct HnswSearcher and exercise the per-query override.
    let reader = index.reader()?;
    let direct = HnswSearcher::new(reader.clone())?;
    let override_request = VectorIndexQuery::new(Vector::new(vec![1.0, 0.0, 0.0]))
        .top_k(5)
        .field_name("vec".to_string())
        .ef_search(400);
    let with_override = direct.search(&override_request)?;
    assert!(
        !with_override.results.is_empty(),
        "expected non-empty results from per-query override search path"
    );

    Ok(())
}

/// A [`StorageInput`] that counts every byte read from its inner stream, used to
/// measure the I/O an Eager `.hnsw` load performs (Issue #789).
#[derive(Debug)]
struct CountingInput {
    inner: Box<dyn crate::storage::StorageInput>,
    counter: Arc<std::sync::atomic::AtomicU64>,
}

impl std::io::Read for CountingInput {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.counter
            .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }
}

impl std::io::Seek for CountingInput {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl crate::storage::StorageInput for CountingInput {
    fn size(&self) -> Result<u64> {
        self.inner.size()
    }

    fn clone_input(&self) -> Result<Box<dyn crate::storage::StorageInput>> {
        Ok(Box::new(CountingInput {
            inner: self.inner.clone_input()?,
            counter: Arc::clone(&self.counter),
        }))
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    // Force every read through `read` (and thus the counter); the HNSW reader
    // never takes the zero-copy `as_slice` path, so this matches production.
    fn as_slice(&self) -> Option<&[u8]> {
        None
    }
}

/// A [`Storage`] that wraps another and counts the bytes read from files whose
/// name ends in `.hnsw`, so a test can assert how many passes a load makes over
/// the segment (Issue #789). All other operations delegate unchanged.
#[derive(Debug)]
struct CountingStorage {
    inner: Arc<dyn crate::storage::Storage>,
    hnsw_bytes_read: Arc<std::sync::atomic::AtomicU64>,
}

impl crate::storage::Storage for CountingStorage {
    fn open_input(&self, name: &str) -> Result<Box<dyn crate::storage::StorageInput>> {
        let input = self.inner.open_input(name)?;
        if name.ends_with(".hnsw") {
            Ok(Box::new(CountingInput {
                inner: input,
                counter: Arc::clone(&self.hnsw_bytes_read),
            }))
        } else {
            Ok(input)
        }
    }

    fn create_output(&self, name: &str) -> Result<Box<dyn crate::storage::StorageOutput>> {
        self.inner.create_output(name)
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn crate::storage::StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(name)
    }

    fn list_files(&self) -> Result<Vec<String>> {
        self.inner.list_files()
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.inner.file_size(name)
    }

    fn metadata(&self, name: &str) -> Result<crate::storage::FileMetadata> {
        self.inner.metadata(name)
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.inner.rename_file(old_name, new_name)
    }

    fn create_temp_output(
        &self,
        prefix: &str,
    ) -> Result<(String, Box<dyn crate::storage::StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }

    fn sync(&self) -> Result<()> {
        self.inner.sync()
    }

    fn close(&mut self) -> Result<()> {
        // The inner storage is shared behind an `Arc`; nothing to close here.
        Ok(())
    }
}

/// Eager load must read the `.hnsw` segment exactly once (Issue #789).
///
/// The integrity CRC is folded into the single structural pass, so the only
/// `.hnsw` reads are the 8-byte footer probe plus one sequential pass over the
/// content — `file_size` bytes total. Before #789, verification ran as a
/// separate full pass, so a footer-carrying segment was read ~twice
/// (`2 * content_len + 8`). Asserting the exact single-pass byte count is a
/// deterministic regression guard against the double-read returning.
#[test]
fn eager_load_reads_hnsw_segment_exactly_once() -> Result<()> {
    let inner = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = HnswIndexConfig {
        dimension: 4,
        m: 8,
        ef_construction: 32,
        distance_metric: DistanceMetric::Cosine,
        normalize_vectors: false,
        ..Default::default()
    };
    // A handful of vectors makes the segment comfortably larger than the
    // 8-byte footer, so a single pass and a double pass differ unambiguously.
    let vectors: Vec<(u64, String, Vector)> = (0..32)
        .map(|i| {
            let f = i as f32;
            (
                i,
                "f".to_string(),
                Vector::new(vec![f, f + 1.0, f + 2.0, f + 3.0]),
            )
        })
        .collect();
    let mut writer = HnswIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "count_seg",
        Arc::clone(&inner),
    )?;
    writer.add_vectors(vectors)?;
    writer.finalize()?;
    writer.write()?;

    let file_size = inner.file_size("count_seg.hnsw")?;
    assert!(
        file_size > crate::vector::index::hnsw::HNSW_FOOTER_LEN,
        "segment must carry a footer for this measurement"
    );

    let hnsw_bytes_read = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let counting: Arc<dyn crate::storage::Storage> = Arc::new(CountingStorage {
        inner: Arc::clone(&inner),
        hnsw_bytes_read: Arc::clone(&hnsw_bytes_read),
    });
    // Default loading_mode() is Eager, which is the folded path under test.
    assert!(matches!(
        counting.loading_mode(),
        crate::storage::LoadingMode::Eager
    ));

    let _reader = HnswIndexReader::load(counting, "count_seg", DistanceMetric::Cosine)?;

    let read = hnsw_bytes_read.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        read, file_size,
        "Eager load must read the segment exactly once (footer probe + one \
         folded pass = file_size); a double-read would be ~2x content_len"
    );
    Ok(())
}

/// A corrupted pq-fastscan Eager segment must be rejected by the folded CRC
/// (Issue #789).
///
/// The default-quantizer corruption tests only cover Scalar8Bit. The
/// `OwnedPqFastScan` branch of [`HnswIndexReader::load`] also reads purely
/// sequentially (`read_pq_fastscan_record` is `Read`-bound, never seeks), so
/// `is_sequential()` stays true and the CRC is folded into the single Eager
/// pass on this branch too. This test locks in that a byte flip on a
/// non-Scalar8Bit segment is still detected.
#[cfg(feature = "pq-fastscan")]
#[test]
fn eager_load_rejects_corrupted_pq_fastscan_segment() -> Result<()> {
    use crate::vector::core::quantization::QuantizationMethod;
    use std::io::{Read, Write};

    let dim = 8usize;
    let sub = 4usize;
    let n = 64u64;
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = HnswIndexConfig {
        dimension: dim,
        m: 16,
        ef_construction: 100,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::ProductQuantizationFastScan {
            subvector_count: sub,
        },
        ..Default::default()
    };
    // Deterministic, broadly-spread vectors so the K=16 codebook trainer
    // converges to non-degenerate centroids (mirrors pq_fastscan_search_test).
    let vectors: Vec<(u64, String, Vector)> = (0..n)
        .map(|i| {
            let s = i as usize;
            let v: Vec<f32> = (0..dim)
                .map(|d| ((s * 31 + d * 17) % 257) as f32 - 128.0)
                .collect();
            (i, "f".to_string(), Vector::new(v))
        })
        .collect();
    let mut writer = HnswIndexWriter::with_storage(
        config,
        VectorIndexWriterConfig::default(),
        "pqfs_seg",
        Arc::clone(&storage),
    )?;
    writer.add_vectors(vectors)?;
    writer.finalize()?;
    writer.write()?;

    // Sanity: the clean segment loads.
    HnswIndexReader::load(Arc::clone(&storage), "pqfs_seg", DistanceMetric::Euclidean)?;

    // Flip a byte deep in the content (well before the 8-byte footer); the
    // folded CRC must reject the segment on the next load.
    let mut bytes = {
        let mut input = storage.open_input("pqfs_seg.hnsw")?;
        let mut buf = Vec::new();
        input
            .read_to_end(&mut buf)
            .expect("read pq-fastscan segment");
        buf
    };
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    {
        let mut out = storage.create_output("pqfs_seg.hnsw")?;
        out.write_all(&bytes).expect("rewrite corrupted segment");
        out.close()?;
    }

    let result = HnswIndexReader::load(Arc::clone(&storage), "pqfs_seg", DistanceMetric::Euclidean);
    assert!(
        result.is_err(),
        "a corrupted pq-fastscan .hnsw must be rejected on Eager load, got Ok"
    );
    Ok(())
}

/// Issue #841: HNSW **level assignment** must be deterministic — the
/// level RNG is seeded with a fixed constant ([`LEVEL_RNG_SEED`] in the
/// writer), so building the same vector set twice yields the same entry
/// point, max level, and per-node layer counts.
///
/// Neighbor lists are deliberately NOT compared: graph insertion runs in
/// parallel (`ConcurrentHnswGraph` + rayon), so neighbor selection still
/// depends on thread interleaving. That residual nondeterminism is
/// documented on #841; this test pins exactly the invariant the seeded
/// RNG guarantees.
#[test]
fn graph_build_levels_are_deterministic_across_writers() -> Result<()> {
    /// Loaded level shape: `(entry_point, max_level, per-node layer counts)`.
    type LevelShape = (Option<u64>, usize, Vec<(u64, usize)>);

    /// Build one segment from a fixed 64-vector set and return its
    /// loaded [`LevelShape`].
    fn build(name: &str) -> Result<LevelShape> {
        let storage =
            StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
        let config = HnswIndexConfig {
            dimension: 4,
            m: 4,
            ef_construction: 16,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let mut writer = HnswIndexWriter::with_storage(
            config,
            VectorIndexWriterConfig::default(),
            name,
            Arc::clone(&storage),
        )?;
        // Deterministic non-trivial vectors; 64 nodes give the level
        // RNG room to produce multiple layers.
        let vectors: Vec<(u64, String, Vector)> = (0..64u64)
            .map(|i| {
                let t = i as f32;
                (
                    i,
                    "f".to_string(),
                    Vector::new(vec![
                        (t * 0.37).sin(),
                        (t * 0.73).cos(),
                        (t * 0.11).sin(),
                        (t * 0.53).cos(),
                    ]),
                )
            })
            .collect();
        writer.add_vectors(vectors)?;
        writer.finalize()?;
        writer.write()?;

        let reader = HnswIndexReader::load(storage, name, DistanceMetric::Cosine)?;
        let graph = reader.graph.as_ref().expect("segment must carry a graph");
        // `iter_nodes` yields nodes in ordinal (= ascending doc id) order,
        // so the collected shape is deterministic by construction. The
        // entry point is compared as a doc id (stable across builds),
        // not as an ordinal.
        let levels = graph
            .iter_nodes()
            .map(|(id, layers)| (id, layers.len()))
            .collect();
        Ok((
            graph.entry_point().map(|ord| graph.doc_id(ord)),
            graph.max_level(),
            levels,
        ))
    }

    let a = build("determinism_a")?;
    let b = build("determinism_b")?;
    assert_eq!(a.0, b.0, "entry point must be identical across builds");
    assert_eq!(a.1, b.1, "max level must be identical across builds");
    assert_eq!(
        a.2, b.2,
        "every node's layer count must be identical across builds"
    );
    Ok(())
}
