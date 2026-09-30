//! Integration tests for issue #943 — DocValues availability and
//! freshness at the reader level.

use std::sync::Arc;

use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::writer::LexicalIndexWriter;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};

/// An index and a writer registered with it (#1024): a standalone
/// `InvertedIndexWriter` is ephemeral — its segments enter no manifest, so
/// no reader sees them — so durable fixtures go through
/// `InvertedIndex::create` + `writer()` and read back through
/// `index.reader()`.
fn index_and_writer(
    storage: Arc<dyn laurus::storage::Storage>,
) -> (InvertedIndex, Box<dyn LexicalIndexWriter>) {
    let index = InvertedIndex::create(storage, Default::default()).unwrap();
    let writer = index.writer().unwrap();
    (index, writer)
}

/// #943: `has_doc_values` must answer correctly as the very first
/// operation on a fresh reader — it used to report `false` until some
/// other call happened to load the DocValues cache.
#[test]
fn has_doc_values_is_correct_on_a_fresh_reader() {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let (index, mut writer) = index_and_writer(storage);

    let doc = Document::builder()
        .add_field("popularity", DataValue::Int64(42))
        .add_field("body", DataValue::Text("alpha".into()))
        .build();
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let reader = index.reader().unwrap();

    assert!(
        reader.has_doc_values("popularity"),
        "doc values exist on disk — a fresh reader must report them"
    );
    assert!(!reader.has_doc_values("no_such_field"));
}
