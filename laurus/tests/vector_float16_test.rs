//! End-to-end tests for `QuantizationMethod::Float16` dense vector fields
//! (Issue #623) through the engine, on mmap-backed `FileStorage` (the
//! default deployment, where the segment rows are read in place).
//!
//! A Float16 field must rank and score exactly as f32 search over the
//! stored (f16-rounded) values would — `Scalar8Bit` scores differ from
//! that by about 1e-3, so the score comparisons below also prove the
//! segments were written as Float16.

use std::io::Read;

use tempfile::TempDir;

use laurus::DistanceMetric;
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::vector::{FlatOption, HnswOption, IvfOption, Vector};
use laurus::{DataValue, Document, Engine, QuantizationMethod, UpdateFieldOptions};
use laurus::{FieldOption, QueryVector, Schema, SearchRequestBuilder, VectorSearchQuery};

const DIM: usize = 24;
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
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 + 1.0e-4
        })
        .collect()
}

/// The value a Float16 segment stores for `v`, as f32.
fn rounded(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn flat() -> FieldOption {
    FieldOption::Flat(FlatOption {
        dimension: DIM,
        distance: DistanceMetric::Euclidean,
        quantizer: QuantizationMethod::Float16,
        ..FlatOption::default()
    })
}

fn ivf() -> FieldOption {
    // Probing every list makes IVF exact.
    FieldOption::Ivf(IvfOption {
        dimension: DIM,
        distance: DistanceMetric::Euclidean,
        n_clusters: 2,
        n_probe: 2,
        quantizer: QuantizationMethod::Float16,
        ..IvfOption::default()
    })
}

fn hnsw() -> FieldOption {
    FieldOption::Hnsw(HnswOption {
        dimension: DIM,
        distance: DistanceMetric::Euclidean,
        quantizer: QuantizationMethod::Float16,
        ..HnswOption::default()
    })
}

async fn open(dir: &TempDir, option: FieldOption) -> (Engine, std::sync::Arc<dyn Storage>) {
    let mut config = FileStorageConfig::new(dir.path());
    config.use_mmap = true;
    let storage = StorageFactory::create(StorageConfig::File(config)).unwrap();
    let schema = Schema::builder().add_field(FIELD, option).build();
    let engine = Engine::new(storage.clone(), schema).await.unwrap();
    (engine, storage)
}

/// Put `ids` with `values(id)` and commit once.
async fn ingest(engine: &Engine, ids: impl IntoIterator<Item = u64>) {
    for id in ids {
        let doc = Document::builder()
            .add_field(FIELD, DataValue::Vector(values(id)))
            .build();
        engine.put_document(&id.to_string(), doc).await.unwrap();
    }
    engine.commit().await.unwrap();
}

async fn search(engine: &Engine, query: Vec<f32>, limit: usize) -> Vec<(String, f32)> {
    let request = SearchRequestBuilder::new()
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(query),
            weight: 1.0,
            fields: Some(vec![FIELD.to_string()]),
        }]))
        .limit(limit)
        .build();
    engine
        .search(request)
        .await
        .unwrap()
        .into_iter()
        .map(|r| (r.id, r.score))
        .collect()
}

/// The top `limit` ids and Euclidean similarities (`exp(-d)`) by brute
/// force over the stored values of `ids`.
fn expected(ids: impl IntoIterator<Item = u64>, query: &[f32], limit: usize) -> Vec<(String, f32)> {
    let mut all: Vec<(u64, f32)> = ids
        .into_iter()
        .map(|id| (id, euclidean(query, &rounded(&values(id)))))
        .collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    all.into_iter()
        .take(limit)
        .map(|(id, d)| (id.to_string(), (-d).exp()))
        .collect()
}

fn assert_same_ranking(got: &[(String, f32)], want: &[(String, f32)], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: result count");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.0, w.0, "{what}: ranking");
        assert!(
            (g.1 - w.1).abs() <= 1e-5,
            "{what}: doc {} score {} vs {}",
            g.0,
            g.1,
            w.1
        );
    }
}

/// Flat and IVF over several segments (one per commit): the fan-out merges
/// per-segment results into the exact f32 ranking of the stored values.
#[tokio::test(flavor = "multi_thread")]
async fn exact_index_types_rank_and_score_on_the_stored_values() {
    for (name, option) in [("flat", flat()), ("ivf", ivf())] {
        let dir = TempDir::new().unwrap();
        let (engine, _) = open(&dir, option).await;
        for batch in 0..3u64 {
            ingest(&engine, batch * 10..batch * 10 + 10).await;
        }
        let query = values(10_000);
        let got = search(&engine, query.clone(), 5).await;
        assert_same_ranking(&got, &expected(0..30, &query, 5), name);
    }
}

/// HNSW is approximate: every stored vector must be its own nearest
/// neighbour at distance 0 (similarity exactly 1).
#[tokio::test(flavor = "multi_thread")]
async fn hnsw_finds_each_document_at_distance_zero() {
    let dir = TempDir::new().unwrap();
    let (engine, _) = open(&dir, hnsw()).await;
    ingest(&engine, 0..20).await;
    ingest(&engine, 20..40).await;
    for id in [0u64, 7, 19, 20, 39] {
        let got = search(&engine, rounded(&values(id)), 1).await;
        assert_eq!(got[0].0, id.to_string());
        assert_eq!(
            got[0].1, 1.0,
            "doc {id}: stored row must equal its rounded values"
        );
    }
}

/// The committed segment file carries `quant_kind = 4` (Float16).
#[tokio::test(flavor = "multi_thread")]
async fn flat_segment_is_written_as_float16() {
    let dir = TempDir::new().unwrap();
    let (engine, storage) = open(&dir, flat()).await;
    ingest(&engine, 0..5).await;
    let mut input = storage
        .open_input("vector/embedding/segment_000000.flat")
        .unwrap();
    // Flat preamble (count u32 + dim u32), then the LVS1 header.
    let mut head = [0u8; 16];
    input.read_exact(&mut head).unwrap();
    assert_eq!(&head[8..12], b"LVS1");
    assert_eq!(u16::from_le_bytes([head[14], head[15]]), 4);
}

/// A deleted document never comes back from the Float16 fast paths.
#[tokio::test(flavor = "multi_thread")]
async fn deleted_documents_are_not_returned() {
    for (name, option) in [("flat", flat()), ("ivf", ivf()), ("hnsw", hnsw())] {
        let dir = TempDir::new().unwrap();
        let (engine, _) = open(&dir, option).await;
        ingest(&engine, 0..20).await;
        engine.delete_documents("7").await.unwrap();
        engine.commit().await.unwrap();
        let got = search(&engine, rounded(&values(7)), 20).await;
        assert!(
            got.iter().all(|(id, _)| id != "7"),
            "{name}: deleted doc 7 came back"
        );
        assert_eq!(got.len(), 19, "{name}");
    }
}

/// Many small commits trigger merges, which decode and re-encode the rows;
/// Float16 re-encoding is exact, so every document still matches its own
/// rounded values at distance 0.
#[tokio::test(flavor = "multi_thread")]
async fn merges_keep_rows_bit_identical() {
    let dir = TempDir::new().unwrap();
    let (engine, storage) = open(&dir, flat()).await;
    for batch in 0..24u64 {
        ingest(&engine, batch * 2..batch * 2 + 2).await;
    }
    let segments = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.starts_with("vector/embedding/") && f.ends_with(".flat"))
        .count();
    assert!(
        segments < 24,
        "expected merges to have run, found {segments} segments"
    );
    for id in 0..48u64 {
        let got = search(&engine, rounded(&values(id)), 1).await;
        assert_eq!(got[0], (id.to_string(), 1.0), "doc {id}");
    }
}

/// Changing a field from `Scalar8Bit` to `Float16` rebuilds its segments
/// as Float16. Without an f32 sidecar the rebuild starts from the
/// dequantized int8 values, so only the kind and the ranking are checked.
#[tokio::test(flavor = "multi_thread")]
async fn update_field_rebuilds_scalar8bit_segments_as_float16() {
    let dir = TempDir::new().unwrap();
    let sq8 = FieldOption::Flat(FlatOption {
        dimension: DIM,
        distance: DistanceMetric::Euclidean,
        ..FlatOption::default()
    });
    let (engine, storage) = open(&dir, sq8).await;
    ingest(&engine, 0..10).await;
    engine
        .update_field(
            FIELD,
            flat(),
            UpdateFieldOptions {
                reindex: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let segments: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.starts_with("vector/embedding/") && f.ends_with(".flat"))
        .collect();
    assert!(!segments.is_empty(), "a rebuilt segment");
    for segment in &segments {
        let mut head = [0u8; 16];
        storage
            .open_input(segment)
            .unwrap()
            .read_exact(&mut head)
            .unwrap();
        assert_eq!(u16::from_le_bytes([head[14], head[15]]), 4, "{segment}");
    }
    let got = search(&engine, values(3), 1).await;
    assert_eq!(got[0].0, "3");
}

/// The schema spelling is `quantizer = "Float16"` and survives a TOML
/// round trip.
#[test]
fn schema_toml_round_trips_float16() {
    let schema = Schema::builder().add_field(FIELD, flat()).build();
    let toml = schema.to_toml().unwrap();
    assert!(toml.contains("quantizer = \"Float16\""), "{toml}");
    let back = Schema::from_toml(&toml).unwrap();
    assert_eq!(back.to_toml().unwrap(), toml);
}
