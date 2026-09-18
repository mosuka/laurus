//! Integration tests for chunked, LZ4-compressed stored fields (Issue #548).
//!
//! Unit-level round-trip/corruption tests for the `.docs` format itself live
//! in `laurus/src/lexical/index/structures/stored_fields.rs`; these tests
//! exercise the format through the full `LexicalStore` API (commit, merge,
//! both storage layouts) as the end-to-end regression net.

use std::collections::BTreeMap;
use std::sync::Arc;

use laurus::lexical::{
    InvertedIndexConfig, LexicalIndexConfig, LexicalSearchRequest, LexicalStore, TermQuery,
};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::{DataValue, Document};

fn store_with_layout(use_compound: bool) -> (Arc<dyn Storage>, LexicalStore) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let store = LexicalStore::new(
        storage.clone(),
        LexicalIndexConfig::Inverted(InvertedIndexConfig {
            use_compound,
            ..Default::default()
        }),
    )
    .unwrap();
    (storage, store)
}

/// Search `field:term` with `load_documents(true)` and return every hit's
/// full document, keyed by doc id.
fn documents_by_id(
    store: &LexicalStore,
    field: &str,
    term: &str,
    limit: usize,
) -> BTreeMap<u64, Document> {
    let query = Box::new(TermQuery::new(field, term));
    let request = LexicalSearchRequest::new(query)
        .limit(limit)
        .load_documents(true);
    store
        .search(request)
        .unwrap()
        .hits
        .into_iter()
        .filter_map(|h| h.document.map(|d| (h.doc_id, d)))
        .collect()
}

fn total_raw_field_bytes(bodies: &[String]) -> usize {
    bodies.iter().map(|b| b.len()).sum()
}

fn docs_file_bytes(storage: &Arc<dyn Storage>) -> usize {
    storage
        .list_files()
        .unwrap()
        .iter()
        .filter(|f| f.ends_with(".docs"))
        .map(|f| storage.metadata(f).unwrap().size)
        .sum::<u64>() as usize
}

#[test]
fn stored_fields_round_trip_across_chunk_boundaries() {
    for use_compound in [false, true] {
        let (_, store) = store_with_layout(use_compound);

        for i in 0..2000u64 {
            let doc = Document::builder()
                .add_text("title", format!("sample document number {i}"))
                .add_integer("n", i as i64)
                .build();
            store.upsert_document(i, doc).unwrap();
        }
        store.commit().unwrap();

        let all = documents_by_id(&store, "title", "sample", 2000);
        assert_eq!(all.len(), 2000, "every document must be retrievable");

        for i in [0u64, 1, 500, 1000, 1999] {
            let d = &all[&i];
            assert_eq!(
                d.fields.get("title"),
                Some(&DataValue::Text(format!("sample document number {i}")))
            );
            assert_eq!(d.fields.get("n"), Some(&DataValue::Int64(i as i64)));
        }
    }
}

#[test]
fn the_docs_part_shrinks_on_a_repetitive_corpus() {
    let (storage, store) = store_with_layout(false);

    let bodies: Vec<String> = (0..500)
        .map(|_| "the quick brown fox jumps over the lazy dog. ".repeat(20))
        .collect();
    for (i, body) in bodies.iter().enumerate() {
        let doc = Document::builder().add_text("body", body.clone()).build();
        store.upsert_document(i as u64, doc).unwrap();
    }
    store.commit().unwrap();

    let raw_bytes = total_raw_field_bytes(&bodies);
    let on_disk = docs_file_bytes(&storage);
    assert!(
        on_disk < raw_bytes / 2,
        "expected substantial compression on a repetitive corpus: \
         {on_disk} on-disk bytes vs {raw_bytes} raw field bytes"
    );
}

#[test]
fn a_stored_bytes_field_survives_a_commit_and_a_merge() {
    let (_, store) = store_with_layout(false);

    let payload = vec![0xDEu8, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03];
    let doc = Document::builder()
        .add_field(
            "attachment",
            DataValue::Bytes(
                payload.clone(),
                Some("application/octet-stream".to_string()),
            ),
        )
        .add_text("title", "has an attachment")
        .build();
    store.upsert_document(1, doc).unwrap();
    store.commit().unwrap();

    let loaded = documents_by_id(&store, "title", "attachment", 10);
    let d = &loaded[&1];
    assert_eq!(
        d.fields.get("attachment"),
        Some(&DataValue::Bytes(
            payload.clone(),
            Some("application/octet-stream".to_string())
        )),
        "a field after a stored Bytes field must not desync (pre-#548 bug)"
    );
    assert_eq!(
        d.fields.get("title"),
        Some(&DataValue::Text("has an attachment".to_string())),
        "the field stored after Bytes must survive intact"
    );

    store.optimize().unwrap();
    let loaded_after_merge = documents_by_id(&store, "title", "attachment", 10);
    let d = &loaded_after_merge[&1];
    assert_eq!(
        d.fields.get("attachment"),
        Some(&DataValue::Bytes(
            payload,
            Some("application/octet-stream".to_string())
        ))
    );
    assert_eq!(
        d.fields.get("title"),
        Some(&DataValue::Text("has an attachment".to_string()))
    );
}

#[test]
fn a_stored_vector_field_round_trips() {
    let (_, store) = store_with_layout(false);

    let doc = Document::builder()
        .add_text("marker", "vector doc")
        .add_vector("embedding", vec![0.1, 0.2, 0.3, 0.4])
        .build();
    store.upsert_document(1, doc).unwrap();
    store.commit().unwrap();

    let loaded = documents_by_id(&store, "marker", "vector", 10);
    let d = &loaded[&1];
    assert_eq!(
        d.fields.get("embedding"),
        Some(&DataValue::Vector(vec![0.1, 0.2, 0.3, 0.4])),
        "a stored Vector field must round-trip (pre-#548: always errored, tag 9 unhandled)"
    );
}
