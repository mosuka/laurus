//! Integration tests for issue #994 — `NumericRangeQuery` on
//! multi-segment indices.
//!
//! Guards two regressions:
//!
//! - the BKD-less matcher fallback must stay segment-bounded and correct
//!   when only some segments carry the numeric field (sparse coverage,
//!   the fan-out path that used to scan `0..global_max_doc` per segment);
//! - stored documents must be decoded once per segment, not re-decoded
//!   on every `document()` lookup.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use laurus::DataValue;
use laurus::Document;
use laurus::lexical::core::field::{FieldOption, IntegerOption, TextOption};
use laurus::lexical::query::NumericRangeQuery;
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{Storage, StorageInput, StorageOutput};

fn chars_doc(chars: i64) -> Document {
    Document::builder()
        .add_field("chars", DataValue::Int64(chars))
        .build()
}

fn body_doc(text: &str) -> Document {
    Document::builder().add_text("body", text).build()
}

fn store_config() -> LexicalIndexConfig {
    LexicalIndexConfig::builder()
        .add_field(
            "chars",
            FieldOption::Integer(IntegerOption {
                indexed: true,
                stored: true,
                multi_valued: false,
                doc_values: true,
            }),
        )
        .add_field("body", FieldOption::Text(TextOption::default()))
        .build()
}

fn range_hits(store: &LexicalStore, lower: i64, upper: i64) -> Vec<u64> {
    let query = Box::new(NumericRangeQuery::i64_range(
        "chars",
        Some(lower),
        Some(upper),
    ));
    let mut ids: Vec<u64> = store
        .search(LexicalSearchRequest::new(query))
        .unwrap()
        .hits
        .iter()
        .map(|hit| hit.doc_id)
        .collect();
    ids.sort_unstable();
    ids
}

/// #994: range queries stay correct when only one of several segments
/// carries the numeric field — the field-less segments have no
/// `.{field}.bkd` file and take the stored-document fallback path.
#[test]
fn range_query_on_sparse_field_across_segments() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(storage, store_config()).unwrap();

    // Segment 1: no doc carries `chars` (no BKD file → fallback path).
    store.upsert_document(1, body_doc("alpha")).unwrap();
    store.upsert_document(2, body_doc("bravo")).unwrap();
    store.commit().unwrap();
    // Segment 2: `chars` present (BKD path).
    store.upsert_document(3, chars_doc(100)).unwrap();
    store.upsert_document(4, chars_doc(955_095)).unwrap();
    store.commit().unwrap();
    // Segment 3: no `chars` again.
    store.upsert_document(5, body_doc("charlie")).unwrap();
    store.upsert_document(6, body_doc("delta")).unwrap();
    store.commit().unwrap();

    assert_eq!(range_hits(&store, 500_000, 2_000_000), vec![4]);
    assert_eq!(range_hits(&store, 0, 1000), vec![3]);
    assert_eq!(range_hits(&store, 0, 2_000_000), vec![3, 4]);
}

/// #994: a stored-but-unindexed numeric field has no BKD tree in any
/// segment, so every hit must come through the stored-document fallback
/// — this gate fails if the fallback stops matching (unlike the sparse
/// test above, whose hits all come from a BKD-backed segment).
#[test]
fn range_query_on_stored_only_field_matches_via_fallback() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field(
            "chars",
            FieldOption::Integer(IntegerOption {
                indexed: false,
                stored: true,
                multi_valued: false,
                doc_values: true,
            }),
        )
        .build();
    let store = LexicalStore::new(storage, config).unwrap();

    store.upsert_document(1, chars_doc(100)).unwrap();
    store.upsert_document(2, chars_doc(955_095)).unwrap();
    store.commit().unwrap();
    store.upsert_document(3, chars_doc(600_000)).unwrap();
    store.upsert_document(4, chars_doc(3_000_000)).unwrap();
    store.commit().unwrap();

    assert_eq!(range_hits(&store, 500_000, 2_000_000), vec![2, 3]);
    assert_eq!(range_hits(&store, 0, 1000), vec![1]);
}

/// Storage decorator counting `open_input` calls on `.docs` files, so a
/// test can assert stored documents are decoded once per segment rather
/// than once per `document()` lookup (#994).
#[derive(Debug)]
struct DocsOpenCountingStorage {
    inner: Arc<dyn Storage>,
    docs_opens: Arc<AtomicU64>,
}

impl Storage for DocsOpenCountingStorage {
    fn open_input(&self, name: &str) -> laurus::Result<Box<dyn StorageInput>> {
        if name.ends_with(".docs") {
            self.docs_opens.fetch_add(1, Ordering::Relaxed);
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
        self.inner.rename_file(from, to)
    }
    fn metadata(&self, name: &str) -> laurus::Result<laurus::storage::FileMetadata> {
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

/// #994: repeated searches (and their stored-field hit retrieval) must
/// decode a segment's `.docs` file once, not once per document lookup.
#[test]
fn stored_documents_decode_once_per_segment() {
    let docs_opens = Arc::new(AtomicU64::new(0));
    let storage: Arc<dyn Storage> = Arc::new(DocsOpenCountingStorage {
        inner: Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
        docs_opens: docs_opens.clone(),
    });
    // Loose layout, explicitly: the gate counts raw `.docs` opens, which a
    // compound segment serves from its container (the decode-once property
    // is layout-independent; this pins it where it is observable).
    let mut config = store_config();
    let LexicalIndexConfig::Inverted(inner) = &mut config;
    inner.use_compound = false;
    let store = LexicalStore::new(storage, config).unwrap();

    store.upsert_document(1, chars_doc(10)).unwrap();
    store.upsert_document(2, chars_doc(20)).unwrap();
    store.upsert_document(3, chars_doc(30)).unwrap();
    store.commit().unwrap();

    for _ in 0..2 {
        let query = Box::new(NumericRangeQuery::i64_range("chars", Some(0), Some(1000)));
        let results = store.search(LexicalSearchRequest::new(query)).unwrap();
        assert_eq!(results.hits.len(), 3);
    }

    assert_eq!(
        docs_opens.load(Ordering::Relaxed),
        1,
        ".docs must be decoded once per segment, not per document lookup"
    );
}
