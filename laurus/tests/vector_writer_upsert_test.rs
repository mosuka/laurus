//! Re-putting a document that is still buffered keeps only its latest
//! version, for the Flat, HNSW and IVF segmented indexes (Issue #1405).
//!
//! The writers skip the buffer scan on a delete when the id is not
//! buffered. These tests drive each segmented index's writer the way the
//! store does (delete, then add) and check that a re-put of a buffered id
//! still replaces the buffered copy, both in the buffer and after the
//! commit.

use std::sync::Arc;

use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::index::VectorIndex;
use laurus::vector::index::config::{FlatIndexConfig, HnswIndexConfig, IvfIndexConfig};
use laurus::vector::index::flat::segmented::SegmentedFlatIndex;
use laurus::vector::index::hnsw::segmented::SegmentedHnswIndex;
use laurus::vector::index::ivf::segmented::SegmentedIvfIndex;
use laurus::vector::search::searcher::{VectorIndexQuery, VectorIndexQueryParams};
use laurus::vector::writer::VectorIndexWriter;
use laurus::vector::{DistanceMetric, Vector};

const DIM: usize = 4;
const DOCS: u64 = 4;
const REPUT_DOC: u64 = 1;
const NEW_VERSION: u64 = 9000;

fn doc_vec(i: u64) -> Vector {
    let t = i as f32 * 0.001;
    Vector::new(vec![t.cos(), t.sin(), (t * 2.0).cos(), (t * 3.0).sin()])
}

fn storage() -> Arc<dyn Storage> {
    Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
}

/// Put `doc_id` with `vector` the way `VectorStore::write_vectors` does.
fn put(writer: &mut dyn VectorIndexWriter, doc_id: u64, vector: Vector) {
    writer.delete_document(doc_id).unwrap();
    writer
        .add_vectors(vec![(doc_id, "v".to_string(), vector)])
        .unwrap();
}

fn reput_keeps_only_the_latest_version(index: &dyn VectorIndex) {
    let mut writer = index.writer().unwrap();
    for doc_id in 0..DOCS {
        put(writer.as_mut(), doc_id, doc_vec(doc_id));
    }
    put(writer.as_mut(), REPUT_DOC, doc_vec(NEW_VERSION));

    let copies: Vec<&Vector> = writer
        .vectors()
        .iter()
        .filter(|(doc_id, _, _)| *doc_id == REPUT_DOC)
        .map(|(_, _, vector)| vector)
        .collect();
    assert_eq!(copies.len(), 1, "the buffer must hold one copy");
    assert_eq!(
        copies[0].data.as_slice(),
        doc_vec(NEW_VERSION).data.as_slice(),
        "the buffered copy must be the latest version"
    );
    assert_eq!(writer.vectors().len() as u64, DOCS);

    writer.commit().unwrap();
    drop(writer);

    assert_eq!(index.stats().unwrap().vector_count, DOCS);
    let results = index
        .searcher()
        .unwrap()
        .search(&VectorIndexQuery {
            query: doc_vec(NEW_VERSION),
            params: VectorIndexQueryParams {
                top_k: DOCS as usize,
                ..Default::default()
            },
            field_name: Some("v".to_string()),
            filter: None,
        })
        .unwrap();
    assert_eq!(results.results[0].doc_id, REPUT_DOC);
    assert_eq!(
        results
            .results
            .iter()
            .filter(|r| r.doc_id == REPUT_DOC)
            .count(),
        1,
        "the committed segment must hold one copy"
    );
}

#[test]
fn flat_reput_of_a_buffered_document_keeps_only_the_latest_version() {
    let index = SegmentedFlatIndex::open_or_create(
        storage(),
        "vector_index",
        FlatIndexConfig {
            dimension: DIM,
            normalize_vectors: false,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        },
    )
    .unwrap();
    reput_keeps_only_the_latest_version(&index);
}

#[test]
fn hnsw_reput_of_a_buffered_document_keeps_only_the_latest_version() {
    let index = SegmentedHnswIndex::open_or_create(
        storage(),
        "vector_index",
        HnswIndexConfig {
            dimension: DIM,
            normalize_vectors: false,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        },
    )
    .unwrap();
    reput_keeps_only_the_latest_version(&index);
}

#[test]
fn ivf_reput_of_a_buffered_document_keeps_only_the_latest_version() {
    let index = SegmentedIvfIndex::open_or_create(
        storage(),
        "vector_index",
        IvfIndexConfig {
            dimension: DIM,
            normalize_vectors: false,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        },
    )
    .unwrap();
    reput_keeps_only_the_latest_version(&index);
}
