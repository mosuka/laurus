//! Float16 segments through the HNSW / Flat / IVF writers, readers and
//! searchers, on heap (`Eager`) and mapped (`Lazy`) storage (Issue #623).

use std::sync::Arc;

use tempfile::TempDir;

use crate::error::Result;
use crate::storage::file::FileStorageConfig;
use crate::storage::memory::MemoryStorageConfig;
use crate::storage::{LoadingMode, Storage, StorageConfig, StorageFactory};
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::distance_f16::{decode_f16_row, encode_f16_row};
use crate::vector::core::quantization::QuantizationMethod;
use crate::vector::core::vector::Vector;
use crate::vector::index::VectorIndex;
use crate::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
use crate::vector::index::flat::FlatIndex;
use crate::vector::index::format::quant_kind;
use crate::vector::index::hnsw::HnswIndex;
use crate::vector::index::ivf::IvfIndex;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::searcher::{VectorIndexQuery, VectorIndexSearcher};

/// An odd dimension, so every kernel tail is exercised.
const DIM: usize = 37;
const COUNT: u64 = 120;
const FIELD: &str = "embedding";

/// Deterministic values that are not exactly representable in f16.
fn values(seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..DIM)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0 + 1.0e-4
        })
        .collect()
}

/// What a Float16 segment stores for `v`, decoded.
fn rounded(v: &[f32]) -> Vec<f32> {
    let mut row = Vec::new();
    encode_f16_row(v, &mut row);
    let mut out = Vec::new();
    decode_f16_row(&row, &mut out);
    out
}

fn corpus() -> Vec<(u64, String, Vector)> {
    (0..COUNT)
        .map(|i| (i, FIELD.to_string(), Vector::new(values(i))))
        .collect()
}

/// The storages under test: in memory (`Eager`, heap rows) and on disk
/// with mmap (`Lazy`, rows read in place).
fn storages() -> Vec<(Arc<dyn Storage>, Option<TempDir>)> {
    let memory =
        StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default())).unwrap();
    let dir = TempDir::new().unwrap();
    let mut file_config = FileStorageConfig::new(dir.path());
    file_config.use_mmap = true;
    let file = StorageFactory::create(StorageConfig::File(file_config)).unwrap();
    vec![(memory, None), (file, Some(dir))]
}

/// Read the `quant_kind` of a written segment: the LVS1 header starts
/// after the index-specific preamble (`preamble` bytes).
fn segment_kind(storage: &Arc<dyn Storage>, file: &str, preamble: usize) -> u16 {
    use std::io::Read;
    let mut input = storage.open_input(file).unwrap();
    let mut buf = vec![0u8; preamble + 8];
    input.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[preamble..preamble + 4], b"LVS1");
    u16::from_le_bytes([buf[preamble + 6], buf[preamble + 7]])
}

/// The checks every index type shares once its segment is written.
fn check_reader(
    reader: &Arc<dyn VectorIndexReader>,
    pool_of: impl Fn(&Arc<dyn VectorIndexReader>) -> bool,
    storage: &Arc<dyn Storage>,
) {
    let mapped = pool_of(reader);
    assert_eq!(
        mapped,
        storage.loading_mode() == LoadingMode::Lazy,
        "Lazy storage reads rows in place, Eager copies them"
    );
    for i in 0..COUNT {
        let got = reader.get_vector(i, FIELD).unwrap().unwrap();
        assert_eq!(
            got.data.as_slice(),
            rounded(&values(i)).as_slice(),
            "doc {i}"
        );
    }
}

/// Every result's distance equals the f32 distance on the stored
/// (rounded) values, and the ranking matches brute force on them.
fn check_exact_results(searcher: &dyn VectorIndexSearcher, metric: DistanceMetric, query: &[f32]) {
    let request = VectorIndexQuery::new(Vector::new(query.to_vec()))
        .top_k(10)
        .field_name(FIELD.to_string());
    let results = searcher.search(&request).unwrap();
    let prepared = metric.prepare_query(query);
    let mut expected: Vec<(u64, f32)> = (0..COUNT)
        .map(|i| {
            let d = metric
                .distance_with_prepared(&prepared, &rounded(&values(i)))
                .unwrap();
            (i, d)
        })
        .collect();
    expected.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    assert_eq!(results.results.len(), 10);
    for (got, want) in results.results.iter().zip(&expected) {
        assert_eq!(got.doc_id, want.0, "{metric:?}: rank order");
        assert!(
            (got.distance - want.1).abs() <= 1e-5 * want.1.abs().max(1.0),
            "{metric:?} doc {}: distance {} vs {}",
            got.doc_id,
            got.distance,
            want.1
        );
    }
}

#[test]
fn flat_float16_round_trips_and_searches_exactly() -> Result<()> {
    for metric in [DistanceMetric::Euclidean, DistanceMetric::DotProduct] {
        for (storage, _dir) in storages() {
            let config = FlatIndexConfig {
                dimension: DIM,
                distance_metric: metric,
                quantization_method: QuantizationMethod::Float16,
                normalize_vectors: false,
                ..Default::default()
            };
            let index = FlatIndex::create(storage.clone(), "f16_flat", config)?;
            let mut writer = index.writer()?;
            writer.build(corpus())?;
            writer.finalize()?;
            writer.write()?;
            // Flat preamble: count u32 + dim u32.
            assert_eq!(
                segment_kind(&storage, "f16_flat.flat", 8),
                quant_kind::FLOAT16
            );

            let reader = index.reader()?;
            check_reader(
                &reader,
                |r| {
                    r.as_any()
                        .downcast_ref::<crate::vector::index::flat::reader::FlatVectorIndexReader>()
                        .unwrap()
                        .vectors()
                        .f16_pool()
                        .unwrap()
                        .is_mapped()
                },
                &storage,
            );
            let searcher = index.searcher()?;
            check_exact_results(searcher.as_ref(), metric, &values(1_000));
        }
    }
    Ok(())
}

#[test]
fn ivf_float16_round_trips_and_searches_exactly_across_lists() -> Result<()> {
    for metric in [DistanceMetric::Euclidean, DistanceMetric::Manhattan] {
        for (storage, _dir) in storages() {
            // Probing every list makes IVF exact, so its ranking can be
            // compared with brute force; several lists exercise the pool's
            // per-run addressing in mapped mode.
            let config = IvfIndexConfig {
                dimension: DIM,
                distance_metric: metric,
                n_clusters: 6,
                n_probe: 6,
                quantization_method: QuantizationMethod::Float16,
                normalize_vectors: false,
                ..Default::default()
            };
            let index = IvfIndex::create(storage.clone(), "f16_ivf", config)?;
            let mut writer = index.writer()?;
            writer.build(corpus())?;
            writer.finalize()?;
            writer.write()?;

            let reader = index.reader()?;
            check_reader(
                &reader,
                |r| {
                    r.as_any()
                        .downcast_ref::<crate::vector::index::ivf::reader::IvfIndexReader>()
                        .unwrap()
                        .vectors()
                        .f16_pool()
                        .unwrap()
                        .is_mapped()
                },
                &storage,
            );
            let searcher = index.searcher()?;
            check_exact_results(searcher.as_ref(), metric, &values(2_000));
        }
    }
    Ok(())
}

#[test]
fn hnsw_float16_round_trips_and_finds_each_document() -> Result<()> {
    for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine] {
        for (storage, _dir) in storages() {
            let config = HnswIndexConfig {
                dimension: DIM,
                distance_metric: metric,
                quantization_method: QuantizationMethod::Float16,
                normalize_vectors: false,
                ..Default::default()
            };
            let index = HnswIndex::create(storage.clone(), "f16_hnsw", config)?;
            let mut writer = index.writer()?;
            writer.build(corpus())?;
            writer.finalize()?;
            writer.write()?;
            // HNSW preamble: count u64 + dim u32 + m u32 + ef u32.
            assert_eq!(
                segment_kind(&storage, "f16_hnsw.hnsw", 20),
                quant_kind::FLOAT16
            );

            let reader = index.reader()?;
            check_reader(
                &reader,
                |r| {
                    r.as_any()
                        .downcast_ref::<crate::vector::index::hnsw::reader::HnswIndexReader>()
                        .unwrap()
                        .vectors()
                        .f16_pool()
                        .unwrap()
                        .is_mapped()
                },
                &storage,
            );
            // HNSW is approximate: assert only that each stored vector is
            // its own nearest neighbour, at the f32 distance of its
            // rounded values.
            let searcher = index.searcher()?;
            for i in [0, 17, 63, COUNT - 1] {
                let query = rounded(&values(i));
                let request = VectorIndexQuery::new(Vector::new(query.clone()))
                    .top_k(1)
                    .field_name(FIELD.to_string());
                let results = searcher.search(&request)?;
                assert_eq!(results.results[0].doc_id, i, "{metric:?}");
                let want = metric
                    .distance_with_prepared(&metric.prepare_query(&query), &query)
                    .unwrap();
                assert!(
                    (results.results[0].distance - want).abs() <= 1e-5,
                    "{metric:?}: self distance {}",
                    results.results[0].distance
                );
            }
        }
    }
    Ok(())
}

/// A segment holding two fields uses the reader's ordinal-to-position
/// tables, which must resolve against the Float16 pool.
#[test]
fn hnsw_float16_multi_field_segment_reads_the_right_rows() -> Result<()> {
    for (storage, _dir) in storages() {
        let config = HnswIndexConfig {
            dimension: DIM,
            distance_metric: DistanceMetric::Euclidean,
            quantization_method: QuantizationMethod::Float16,
            normalize_vectors: false,
            ..Default::default()
        };
        let index = HnswIndex::create(storage.clone(), "f16_two_fields", config)?;
        let mut writer = index.writer()?;
        // Field "a" holds even ids, field "b" odd ids plus id 0, so the
        // record order and the per-field orders differ.
        let mut vectors: Vec<(u64, String, Vector)> = (0..COUNT)
            .map(|i| {
                let field = if i % 2 == 0 { "a" } else { "b" };
                (i, field.to_string(), Vector::new(values(i)))
            })
            .collect();
        vectors.push((0, "b".to_string(), Vector::new(values(9_999))));
        writer.build(vectors)?;
        writer.finalize()?;
        writer.write()?;

        let searcher = index.searcher()?;
        for (field, id, seed) in [("a", 42u64, 42u64), ("b", 43, 43), ("b", 0, 9_999)] {
            let query = rounded(&values(seed));
            let request = VectorIndexQuery::new(Vector::new(query))
                .top_k(1)
                .field_name(field.to_string());
            let results = searcher.search(&request)?;
            assert_eq!(results.results[0].doc_id, id, "field {field}");
            assert!(results.results[0].distance.abs() <= 1e-5, "field {field}");
        }
    }
    Ok(())
}

/// Every merge of a Cosine field decodes the rows and re-normalizes them in
/// the writer, and a stored row's norm is not exactly 1. The rescaling is
/// far below half an f16 ulp, so ten rewrites must not move any component
/// by more than one ulp (measured: none moved at 37, 384 or 1024
/// dimensions).
#[test]
fn float16_cosine_rows_do_not_drift_across_rewrites() -> Result<()> {
    for dim in [DIM, 384] {
        let max_ulps = cosine_rewrite_drift(dim)?;
        assert!(
            max_ulps <= 1,
            "dim {dim}: a component drifted {max_ulps} ulps"
        );
    }
    Ok(())
}

/// Rewrite a Cosine Float16 segment ten times from its own decoded rows.
///
/// # Returns
///
/// The largest change of any component, in f16 ulps, against the first
/// generation.
fn cosine_rewrite_drift(dim: usize) -> Result<i32> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = FlatIndexConfig {
        dimension: dim,
        distance_metric: DistanceMetric::Cosine,
        quantization_method: QuantizationMethod::Float16,
        normalize_vectors: true,
        ..Default::default()
    };
    let values = |seed: u64| -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
            })
            .collect()
    };
    let read_all = |name: &str| -> Vec<Vec<f32>> {
        let index = FlatIndex::create(storage.clone(), name, config.clone()).unwrap();
        let reader = index.reader().unwrap();
        (0..COUNT)
            .map(|i| reader.get_vector(i, FIELD).unwrap().unwrap().data.to_vec())
            .collect()
    };
    let write = |name: &str, rows: Vec<Vec<f32>>| {
        let index = FlatIndex::create(storage.clone(), name, config.clone()).unwrap();
        let mut writer = index.writer().unwrap();
        writer
            .build(
                rows.into_iter()
                    .enumerate()
                    .map(|(i, v)| (i as u64, FIELD.to_string(), Vector::new(v)))
                    .collect(),
            )
            .unwrap();
        writer.finalize().unwrap();
        writer.write().unwrap();
    };
    write("gen0", (0..COUNT).map(values).collect());
    let first = read_all("gen0");
    // The writer normalized the input (norm well above 1) on the way in,
    // so every rewrite below goes through the same normalization.
    let norm = |v: &[f32]| {
        v.iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt()
    };
    assert!(norm(&values(0)) > 2.0);
    assert!(
        (norm(&first[0]) - 1.0).abs() < 1e-3,
        "stored rows are normalized"
    );
    let mut current = first.clone();
    let mut max_ulps = 0i32;
    for generation in 1..=10 {
        let name = format!("gen{generation}");
        write(&name, current);
        current = read_all(&name);
        for (a, b) in first.iter().zip(&current) {
            for (x, y) in a.iter().zip(b) {
                let hx = half::f16::from_f32(*x).to_bits() as i32;
                let hy = half::f16::from_f32(*y).to_bits() as i32;
                max_ulps = max_ulps.max((hx - hy).abs());
            }
        }
    }
    Ok(max_ulps)
}

/// Re-encoding the decoded rows (what a merge does through the writer)
/// leaves non-Cosine rows bit-identical.
#[test]
fn float16_rows_survive_a_rewrite_bit_for_bit() -> Result<()> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let config = FlatIndexConfig {
        dimension: DIM,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::Float16,
        normalize_vectors: false,
        ..Default::default()
    };
    let first = FlatIndex::create(storage.clone(), "first", config.clone())?;
    let mut writer = first.writer()?;
    writer.build(corpus())?;
    writer.finalize()?;
    writer.write()?;
    let first_reader = first.reader()?;
    let decoded: Vec<(u64, String, Vector)> = (0..COUNT)
        .map(|i| {
            let v = first_reader.get_vector(i, FIELD).unwrap().unwrap();
            (i, FIELD.to_string(), v)
        })
        .collect();

    let second = FlatIndex::create(storage.clone(), "second", config)?;
    let mut writer = second.writer()?;
    writer.build(decoded)?;
    writer.finalize()?;
    writer.write()?;
    let reader = second.reader()?;
    for i in 0..COUNT {
        let got = reader.get_vector(i, FIELD)?.unwrap();
        let want = rounded(&values(i));
        assert_eq!(
            got.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "doc {i}"
        );
    }
    Ok(())
}
