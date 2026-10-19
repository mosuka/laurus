//! `Scalar8Bit` segments read in place from mapped files (Issue #1463).
//!
//! Each test writes one segment to disk and opens the same file twice: with
//! mmap (`Lazy`, rows read in place) and with buffered I/O (`Eager`, rows
//! copied to the heap). Both must run the same int8 kernels, so every
//! result must match bit for bit. The searchers are used directly: the
//! engine's multi-segment fan-out rescores hits in dequantized f32, which
//! would hide a difference in the per-segment scores.

use std::sync::Arc;

use tempfile::TempDir;

use crate::error::Result;
use crate::storage::file::FileStorageConfig;
use crate::storage::{LoadingMode, Storage, StorageConfig, StorageFactory};
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization::QuantizationMethod;
use crate::vector::core::vector::Vector;
use crate::vector::index::VectorIndex;
use crate::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
use crate::vector::index::flat::FlatIndex;
use crate::vector::index::flat::reader::FlatVectorIndexReader;
use crate::vector::index::hnsw::HnswIndex;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::searcher::HnswSearcher;
use crate::vector::index::ivf::IvfIndex;
use crate::vector::index::ivf::reader::IvfIndexReader;
use crate::vector::index::quantized_storage::QuantizedVectorPool;
use crate::vector::reader::VectorIndexReader;
use crate::vector::search::filter_set::FilterSet;
use crate::vector::search::searcher::{
    VectorIndexQuery, VectorIndexQueryResults, VectorIndexSearcher,
};

const COUNT: u64 = 120;
const FIELD: &str = "embedding";

const METRICS: [DistanceMetric; 5] = [
    DistanceMetric::Cosine,
    DistanceMetric::Euclidean,
    DistanceMetric::Manhattan,
    DistanceMetric::DotProduct,
    DistanceMetric::Angular,
];

/// Dimensions below, at and above the 32-byte SIMD block; 100 also leaves
/// a tail for the NEON (16-byte) and AVX2 (32-byte) kernels.
const DIMS: [usize; 3] = [16, 64, 100];

/// Deterministic values in `[-2, 2)`.
fn values(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

fn corpus(dim: usize) -> Vec<(u64, String, Vector)> {
    (0..COUNT)
        .map(|i| (i, FIELD.to_string(), Vector::new(values(i, dim))))
        .collect()
}

/// Queries that are not stored vectors (so they do not sit on the
/// quantization grid), plus one far outside the trained range.
fn queries(dim: usize) -> Vec<Vec<f32>> {
    let mut qs: Vec<Vec<f32>> = (0..4).map(|k| values(10_000 + k, dim)).collect();
    qs.push(values(20_000, dim).iter().map(|x| x * 3.0).collect());
    qs
}

/// Two storages over one directory: mmap (`Lazy`) and buffered (`Eager`).
fn storages(dir: &TempDir) -> (Arc<dyn Storage>, Arc<dyn Storage>) {
    let open = |mmap: bool| {
        let mut config = FileStorageConfig::new(dir.path());
        config.use_mmap = mmap;
        StorageFactory::create(StorageConfig::File(config)).unwrap()
    };
    let (lazy, eager) = (open(true), open(false));
    assert_eq!(lazy.loading_mode(), LoadingMode::Lazy);
    assert_eq!(eager.loading_mode(), LoadingMode::Eager);
    (lazy, eager)
}

/// The `Scalar8Bit` pool of a reader of any of the three index types.
fn pool_of(reader: &Arc<dyn VectorIndexReader>) -> Arc<QuantizedVectorPool> {
    let any = reader.as_any();
    let vectors = if let Some(r) = any.downcast_ref::<HnswIndexReader>() {
        r.vectors()
    } else if let Some(r) = any.downcast_ref::<FlatVectorIndexReader>() {
        r.vectors()
    } else {
        any.downcast_ref::<IvfIndexReader>().unwrap().vectors()
    };
    vectors.quantized_pool().unwrap().clone()
}

/// Assert that the `Lazy` reader reads rows in place and the `Eager` one
/// copies them, and that both return the same decoded vectors.
fn check_readers(lazy: &Arc<dyn VectorIndexReader>, eager: &Arc<dyn VectorIndexReader>) {
    assert!(pool_of(lazy).is_mapped(), "Lazy rows are read in place");
    assert!(!pool_of(eager).is_mapped(), "Eager rows are on the heap");
    for i in 0..COUNT {
        let l = lazy.get_vector(i, FIELD).unwrap().unwrap();
        let e = eager.get_vector(i, FIELD).unwrap().unwrap();
        assert_eq!(l.data.as_slice(), e.data.as_slice(), "doc {i}");
    }
}

fn assert_same_results(
    lazy: &VectorIndexQueryResults,
    eager: &VectorIndexQueryResults,
    what: &str,
) {
    assert!(!eager.results.is_empty(), "{what}: no results");
    assert_eq!(lazy.results.len(), eager.results.len(), "{what}: length");
    for (rank, (l, e)) in lazy.results.iter().zip(&eager.results).enumerate() {
        assert_eq!(l.doc_id, e.doc_id, "{what}: doc at rank {rank}");
        assert_eq!(l.field_name, e.field_name, "{what}: field at rank {rank}");
        assert_eq!(
            l.distance.to_bits(),
            e.distance.to_bits(),
            "{what}: distance at rank {rank}: {} vs {}",
            l.distance,
            e.distance
        );
        assert_eq!(
            l.similarity.to_bits(),
            e.similarity.to_bits(),
            "{what}: similarity at rank {rank}"
        );
        let (lv, ev) = (l.vector.as_ref(), e.vector.as_ref());
        assert_eq!(
            lv.map(|v| v.data.as_slice()),
            ev.map(|v| v.data.as_slice()),
            "{what}: vector at rank {rank}"
        );
    }
}

/// Run every query against both searchers, with and without vectors.
fn compare_searches(
    lazy: &dyn VectorIndexSearcher,
    eager: &dyn VectorIndexSearcher,
    dim: usize,
    top_k: usize,
    what: &str,
) {
    for (q, query) in queries(dim).into_iter().enumerate() {
        for include_vectors in [false, true] {
            let request = VectorIndexQuery::new(Vector::new(query.clone()))
                .top_k(top_k)
                .field_name(FIELD.to_string())
                .include_vectors(include_vectors);
            let l = lazy.search(&request).unwrap();
            let e = eager.search(&request).unwrap();
            assert_same_results(
                &l,
                &e,
                &format!("{what} query {q} vectors {include_vectors}"),
            );
        }
    }
}

#[test]
fn flat_lazy_scores_exactly_like_eager() -> Result<()> {
    for dim in DIMS {
        for metric in METRICS {
            let dir = TempDir::new()?;
            let (lazy, eager) = storages(&dir);
            let config = FlatIndexConfig {
                dimension: dim,
                distance_metric: metric,
                quantization_method: QuantizationMethod::Scalar8Bit,
                ..Default::default()
            };
            let index = FlatIndex::create(lazy, "sq8_flat", config.clone())?;
            let mut writer = index.writer()?;
            writer.build(corpus(dim))?;
            writer.finalize()?;
            writer.write()?;
            let eager_index = FlatIndex::open(eager, "sq8_flat", config)?;

            let (lazy_reader, eager_reader) = (index.reader()?, eager_index.reader()?);
            check_readers(&lazy_reader, &eager_reader);
            // One run of records: only the run table is on the heap.
            assert_eq!(pool_of(&lazy_reader).heap_size(), 4 + 8 + 8);
            compare_searches(
                index.searcher()?.as_ref(),
                eager_index.searcher()?.as_ref(),
                dim,
                COUNT as usize,
                &format!("flat {metric:?} dim {dim}"),
            );
        }
    }
    Ok(())
}

#[test]
fn ivf_lazy_scores_exactly_like_eager_across_lists() -> Result<()> {
    for dim in DIMS {
        for metric in METRICS {
            let dir = TempDir::new()?;
            let (lazy, eager) = storages(&dir);
            // Several lists: the mapped pool addresses rows across runs.
            let config = IvfIndexConfig {
                dimension: dim,
                distance_metric: metric,
                n_clusters: 6,
                n_probe: 3,
                quantization_method: QuantizationMethod::Scalar8Bit,
                ..Default::default()
            };
            let index = IvfIndex::create(lazy, "sq8_ivf", config.clone())?;
            let mut writer = index.writer()?;
            writer.build(corpus(dim))?;
            writer.finalize()?;
            writer.write()?;
            let eager_index = IvfIndex::open(eager, "sq8_ivf", config)?;

            let (lazy_reader, eager_reader) = (index.reader()?, eager_index.reader()?);
            check_readers(&lazy_reader, &eager_reader);
            assert!(
                pool_of(&lazy_reader).prefetch_base_stride().is_none(),
                "several runs have no single stride"
            );
            compare_searches(
                index.searcher()?.as_ref(),
                eager_index.searcher()?.as_ref(),
                dim,
                10,
                &format!("ivf {metric:?} dim {dim}"),
            );
        }
    }
    Ok(())
}

#[test]
fn hnsw_lazy_scores_exactly_like_eager() -> Result<()> {
    for dim in DIMS {
        for metric in METRICS {
            let dir = TempDir::new()?;
            let (lazy, eager) = storages(&dir);
            let config = HnswIndexConfig {
                dimension: dim,
                distance_metric: metric,
                quantization_method: QuantizationMethod::Scalar8Bit,
                ..Default::default()
            };
            let index = HnswIndex::create(lazy, "sq8_hnsw", config.clone())?;
            let mut writer = index.writer()?;
            writer.build(corpus(dim))?;
            writer.finalize()?;
            writer.write()?;
            let eager_index = HnswIndex::open(eager, "sq8_hnsw", config)?;

            let (lazy_reader, eager_reader) = (index.reader()?, eager_index.reader()?);
            check_readers(&lazy_reader, &eager_reader);
            let (lazy_pool, eager_pool) = (pool_of(&lazy_reader), pool_of(&eager_reader));
            assert_eq!(lazy_pool.heap_size(), 4 + 8 + 8);
            assert!(lazy_pool.prefetch_base_stride().is_some());
            assert!(eager_pool.prefetch_base_stride().is_some());
            let what = format!("hnsw {metric:?} dim {dim}");
            let (lazy_searcher, eager_searcher) = (index.searcher()?, eager_index.searcher()?);
            compare_searches(
                lazy_searcher.as_ref(),
                eager_searcher.as_ref(),
                dim,
                10,
                &what,
            );

            // A small allow-set takes the brute-force path (Issue #738),
            // which scores through the same pool.
            let allowed: Vec<u64> = (0..COUNT).step_by(15).collect();
            let filter = Arc::new(FilterSet::from_doc_ids(&allowed));
            for query in queries(dim) {
                let request = VectorIndexQuery::new(Vector::new(query))
                    .top_k(5)
                    .field_name(FIELD.to_string())
                    .filter(filter.clone());
                assert_same_results(
                    &lazy_searcher.search(&request)?,
                    &eager_searcher.search(&request)?,
                    &format!("{what} filtered"),
                );
            }
        }
    }
    Ok(())
}

/// A segment holding two fields resolves ordinals through the reader's
/// ordinal-to-position tables, which a mapped pool must back too.
#[test]
fn hnsw_lazy_multi_field_segment_scores_exactly_like_eager() -> Result<()> {
    let dim = 64;
    let dir = TempDir::new()?;
    let (lazy, eager) = storages(&dir);
    let config = HnswIndexConfig {
        dimension: dim,
        distance_metric: DistanceMetric::Euclidean,
        quantization_method: QuantizationMethod::Scalar8Bit,
        ..Default::default()
    };
    let index = HnswIndex::create(lazy, "sq8_two_fields", config.clone())?;
    let mut writer = index.writer()?;
    // Field "a" holds even ids, field "b" odd ids plus id 0, so the record
    // order and the per-field orders differ.
    let mut vectors: Vec<(u64, String, Vector)> = (0..COUNT)
        .map(|i| {
            let field = if i % 2 == 0 { "a" } else { "b" };
            (i, field.to_string(), Vector::new(values(i, dim)))
        })
        .collect();
    vectors.push((0, "b".to_string(), Vector::new(values(9_999, dim))));
    writer.build(vectors)?;
    writer.finalize()?;
    writer.write()?;
    let eager_index = HnswIndex::open(eager, "sq8_two_fields", config)?;

    let lazy_reader = index.reader()?;
    assert!(pool_of(&lazy_reader).is_mapped());
    let lazy_hnsw = lazy_reader
        .as_any()
        .downcast_ref::<HnswIndexReader>()
        .unwrap();
    assert!(
        lazy_hnsw.field_ord_to_pos("a").is_some(),
        "a multi-field segment uses ordinal tables"
    );
    let (lazy_searcher, eager_searcher) = (index.searcher()?, eager_index.searcher()?);
    for field in ["a", "b"] {
        for query in queries(dim) {
            let request = VectorIndexQuery::new(Vector::new(query))
                .top_k(10)
                .field_name(field.to_string());
            assert_same_results(
                &lazy_searcher.search(&request)?,
                &eager_searcher.search(&request)?,
                &format!("field {field}"),
            );
        }
    }
    Ok(())
}

/// Warmup touches the mapped rows of a `Lazy` HNSW segment (the engine's
/// fan-out searcher does not forward `warmup`, so this calls it directly).
#[test]
fn hnsw_warmup_touches_mapped_rows() -> Result<()> {
    let dim = 64;
    let dir = TempDir::new()?;
    let (lazy, _eager) = storages(&dir);
    let config = HnswIndexConfig {
        dimension: dim,
        quantization_method: QuantizationMethod::Scalar8Bit,
        ..Default::default()
    };
    let index = HnswIndex::create(lazy, "sq8_warm", config)?;
    let mut writer = index.writer()?;
    writer.build(corpus(dim))?;
    writer.finalize()?;
    writer.write()?;

    let reader = index.reader()?;
    let pool = pool_of(&reader);
    assert!(pool.is_mapped());
    assert!(pool.touch_mapped_rows().is_some());
    let mut searcher = HnswSearcher::new(reader)?;
    searcher.warmup()?;
    let request = VectorIndexQuery::new(Vector::new(values(10_000, dim)))
        .top_k(3)
        .field_name(FIELD.to_string());
    assert_eq!(searcher.search(&request)?.results.len(), 3);
    Ok(())
}

/// A file cut inside its last record's meta fails to load in `Lazy` mode,
/// as it does in `Eager` mode, instead of panicking at query time (a mapped
/// input's seek accepts positions past the end of the file). IVF is used
/// because its list lengths leave the header's count check enough slack to
/// accept the cut file; the per-row check must catch it.
#[test]
fn truncated_ivf_segment_fails_to_load_on_mmap() -> Result<()> {
    let dim = 16;
    let dir = TempDir::new()?;
    {
        let (lazy, _) = storages(&dir);
        let config = IvfIndexConfig {
            dimension: dim,
            n_clusters: 6,
            n_probe: 6,
            quantization_method: QuantizationMethod::Scalar8Bit,
            ..Default::default()
        };
        let index = IvfIndex::create(lazy, "sq8_cut", config)?;
        let mut writer = index.writer()?;
        writer.build(corpus(dim))?;
        writer.finalize()?;
        writer.write()?;
    }
    let path = dir.path().join("sq8_cut.ivf");
    let len = std::fs::metadata(&path)?.len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)?
        .set_len(len - 1)?;

    // Fresh storages, so no mapping of the old length is reused.
    let (lazy, eager) = storages(&dir);
    let err = IvfIndexReader::load(lazy, "sq8_cut", DistanceMetric::Cosine)
        .expect_err("a truncated record must fail the Lazy load");
    assert!(err.to_string().contains("runs past the end"), "{err}");
    assert!(IvfIndexReader::load(eager, "sq8_cut", DistanceMetric::Cosine).is_err());
    Ok(())
}
