//! Integration tests for CRC-32 integrity checking of the production HNSW
//! index files (issue #786).
//!
//! New `.hnsw` segments carry a CRC-32 footer and `metadata.json` is CRC-framed
//! via `StructWriter`; both are verified on load. A flipped byte must be
//! rejected, while legacy files without a checksum must still load.

use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;

use laurus::lexical::LexicalIndexConfig;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::store::config::VectorFieldConfig;
use laurus::vector::store::request::{
    QueryVector, VectorScoreMode, VectorSearchParams, VectorSearchRequest,
};
use laurus::vector::{FieldOption, VectorIndexConfig, VectorSearchQuery};
use laurus::{DataValue, Document};
use laurus::{EmbedInput, EmbedInputType, Embedder};
use laurus::{LaurusError, Result};

const DIM: usize = 16;
const N: u64 = 50;
const STEP: f32 = 0.01;
/// First sealed segment of the (default, #882) segmented layout — the
/// same LVS file format (and CRC footer semantics) as the monolithic file.
/// Lives under the "vec" field's own sub-namespace (Issue #948:
/// `MultiFieldVectorIndex` gives every vector field its own
/// `PrefixedStorage` directory rather than sharing the storage root).
const HNSW_FILE: &str = "vec/segment_000000.hnsw";
/// The segmented layout's commit pivot (#879): CRC-framed and
/// atomically replaced, so corruption must fail the open loudly.
const METADATA_FILE: &str = "vec/segments.json";

#[derive(Debug)]
struct MockEmbedder {
    dimension: usize,
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
        match input {
            EmbedInput::Text(_) => Ok(Vector::new(vec![0.0; self.dimension])),
            _ => Err(LaurusError::invalid_argument("text only")),
        }
    }
    fn supported_input_types(&self) -> Vec<EmbedInputType> {
        vec![EmbedInputType::Text]
    }
    fn name(&self) -> &str {
        "mock"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn doc_vec(i: u64) -> Vec<f32> {
    let theta = i as f32 * STEP;
    let mut v = vec![0.0; DIM];
    v[0] = theta.cos();
    v[1] = theta.sin();
    v
}

fn query_vec() -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[0] = 1.0;
    v
}

fn make_config() -> VectorIndexConfig {
    let mut field_configs = HashMap::new();
    field_configs.insert(
        "vec".to_string(),
        VectorFieldConfig {
            vector: Some(FieldOption::Hnsw(HnswOption {
                dimension: DIM,
                distance: DistanceMetric::Cosine,
                m: 16,
                ef_construction: 100,
                default_ef_search: None,
                base_weight: 1.0,
                quantizer: Default::default(),
                rerank_storage: None,
                embedder: None,
                pq_codebook_path: None,
            })),
            lexical: None,
        },
    );
    VectorIndexConfig {
        fields: field_configs,
        embedder: Arc::new(MockEmbedder { dimension: DIM }),
        default_fields: vec!["vec".to_string()],
        metadata: HashMap::new(),
        deletion_config: laurus::DeletionConfig {
            auto_compaction: false,
            ..Default::default()
        },
        shard_id: 0,
        metadata_config: LexicalIndexConfig::default(),
    }
}

async fn build_committed(storage: Arc<dyn Storage>) {
    let store = laurus::vector::VectorStore::new(storage, make_config()).unwrap();
    for id in 0..N {
        let doc = Document::builder()
            .add_field("vec", DataValue::Vector(doc_vec(id)))
            .build();
        store.upsert_document_by_internal_id(id, doc).await.unwrap();
    }
    store.commit().await.unwrap();
}

fn read_all(storage: &Arc<dyn Storage>, name: &str) -> Vec<u8> {
    let mut input = storage.open_input(name).unwrap();
    let mut buf = Vec::new();
    input.read_to_end(&mut buf).unwrap();
    buf
}

fn write_all(storage: &Arc<dyn Storage>, name: &str, bytes: &[u8]) {
    let mut out = storage.create_output(name).unwrap();
    out.write_all(bytes).unwrap();
    out.close().unwrap();
}

fn request() -> VectorSearchRequest {
    VectorSearchRequest {
        query: VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(query_vec()),
            weight: 1.0,
            fields: Some(vec!["vec".into()]),
        }]),
        params: VectorSearchParams {
            limit: 10,
            score_mode: VectorScoreMode::WeightedSum,
            fields: None,
            allowed_ids: None,
            ..Default::default()
        },
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn corrupted_hnsw_segment_is_rejected() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    build_committed(storage.clone()).await;

    // Flip a byte deep in the segment content (well before the 8-byte footer);
    // the CRC footer still matches the original, so verification must fail.
    let mut bytes = read_all(&storage, HNSW_FILE);
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    write_all(&storage, HNSW_FILE, &bytes);

    let store = laurus::vector::VectorStore::new(storage.clone(), make_config()).unwrap();
    let result = store.search(request());
    assert!(
        result.is_err(),
        "a corrupted .hnsw must be rejected on load, got {result:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn corrupted_hnsw_segment_is_rejected_lazy_mmap() {
    use laurus::storage::file::FileStorageConfig;
    use laurus::storage::{StorageConfig, StorageFactory};

    // FileStorage with mmap selects the Lazy load path, which cannot fold the
    // CRC into its (seeking) structural parse and so verifies the footer up
    // front in a dedicated pass (Issue #789). This guards that path the same
    // way the Eager test guards the folded path.
    let dir = tempfile::tempdir().unwrap();
    let cfg = FileStorageConfig::new(dir.path());
    assert!(
        cfg.use_mmap,
        "this test must exercise the Lazy (mmap) load path"
    );
    let storage = StorageFactory::create(StorageConfig::File(cfg)).unwrap();
    build_committed(storage.clone()).await;

    // Flip a byte deep in the segment content (well before the 8-byte footer).
    let mut bytes = read_all(&storage, HNSW_FILE);
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    write_all(&storage, HNSW_FILE, &bytes);

    // Open a fresh storage instance so no cached mmap masks the on-disk change.
    let storage2 =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path()))).unwrap();
    let store = laurus::vector::VectorStore::new(storage2, make_config()).unwrap();
    let result = store.search(request());
    assert!(
        result.is_err(),
        "a corrupted .hnsw must be rejected on the Lazy load path, got {result:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn corrupted_metadata_is_rejected() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    build_committed(storage.clone()).await;

    // Flip a byte in the CRC-framed manifest content (#879): a corrupted
    // segment registry must fail the open loudly, never silently start
    // empty (which would orphan every live segment).
    let mut bytes = read_all(&storage, METADATA_FILE);
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    write_all(&storage, METADATA_FILE, &bytes);

    let result = laurus::vector::VectorStore::new(storage.clone(), make_config());
    assert!(
        result.is_err(),
        "a corrupted segments.json must be rejected on open"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn legacy_hnsw_without_footer_still_loads() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    build_committed(storage.clone()).await;

    // Drop the 8-byte CRC footer to mimic a pre-#786 segment with no checksum.
    // The remaining bytes are exactly the legacy content, so it must still load.
    let bytes = read_all(&storage, HNSW_FILE);
    let legacy = &bytes[..bytes.len() - 8];
    write_all(&storage, HNSW_FILE, legacy);

    let store = laurus::vector::VectorStore::new(storage.clone(), make_config()).unwrap();
    let results = store.search(request()).unwrap();
    assert_eq!(
        results.hits.len(),
        10,
        "a footer-less (legacy) segment must still load and search"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn legacy_hnsw_without_footer_still_loads_lazy_mmap() {
    use laurus::storage::file::FileStorageConfig;
    use laurus::storage::{StorageConfig, StorageFactory};

    // Mirror of the Eager legacy test for the Lazy (mmap) path: a
    // footer-less segment yields stored_crc = None, so verification is skipped
    // up front and the segment loads unchanged (Issue #789 back-compat).
    let dir = tempfile::tempdir().unwrap();
    let cfg = FileStorageConfig::new(dir.path());
    assert!(
        cfg.use_mmap,
        "this test must exercise the Lazy (mmap) load path"
    );
    let storage = StorageFactory::create(StorageConfig::File(cfg)).unwrap();
    build_committed(storage.clone()).await;

    // Drop the 8-byte CRC footer to mimic a pre-#786 segment with no checksum.
    let bytes = read_all(&storage, HNSW_FILE);
    let legacy = &bytes[..bytes.len() - 8];
    write_all(&storage, HNSW_FILE, legacy);

    // Open a fresh storage instance so no cached mmap masks the on-disk change.
    let storage2 =
        StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path()))).unwrap();
    let store = laurus::vector::VectorStore::new(storage2, make_config()).unwrap();
    let results = store.search(request()).unwrap();
    assert_eq!(
        results.hits.len(),
        10,
        "a footer-less (legacy) segment must still load and search on the Lazy path"
    );
}
