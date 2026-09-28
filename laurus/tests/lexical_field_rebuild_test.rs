//! Issue #1081 (Phase 3) acceptance tests: `Engine::update_field`'s lexical
//! `Reindex` path must actually rebuild a field's on-disk postings from its
//! stored original value -- not just accept the call -- and a rebuild that
//! fails partway through must leave the existing segments and manifest
//! completely untouched.
//!
//! Mirrors `vector_field_rebuild_recall_test.rs` (a real behavior-change
//! proof, not just a "the call returns Ok" check) and
//! `merge_failure_publication_test.rs` (a `Storage` decorator that injects a
//! failure into the merge's segment write, then inspects the manifest).

use std::io::Read;
use std::sync::Arc;

use laurus::lexical::core::field::TextOption;
use laurus::lexical::{
    FieldOption as LexicalFieldOption, LexicalIndexConfig, LexicalSearchRequest, LexicalStore,
};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::storage::{Storage, StorageConfig, StorageFactory, StorageInput, StorageOutput};
use laurus::{
    BuiltinAnalyzerSpec, Document, Engine, FieldChangeKind, FieldOption as SchemaFieldOption,
    LaurusError, Result, Schema, UpdateFieldOptions,
};

/// Same `AnalyzerSpec` shape `japanese_lexical_search_test.rs` uses, but
/// `body` starts with NO analyzer override (the index's default --
/// `StandardAnalyzer`) so the test can `update_field` it to Japanese and
/// compare before/after.
const SCHEMA_JSON: &str = r#"
{
    "default_fields": ["body"],
    "fields": {
        "body": { "Text": { "indexed": true, "stored": true, "term_vectors": false } }
    }
}
"#;

async fn engine_with_default_analyzer() -> Result<Engine> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let schema: Schema = serde_json::from_str(SCHEMA_JSON).expect("valid schema JSON");
    Engine::new(storage, schema).await
}

/// Before the switch, `StandardAnalyzer`'s `\w+` tokenizer treats a whole
/// unspaced Japanese clause as a single token (Japanese script is
/// `\p{Alphabetic}`, so nothing splits it except punctuation) -- so a
/// sub-string query for part of that clause does not match. After
/// `update_field` swaps `body`'s analyzer to Japanese (Lindera) with
/// `reindex: true`, the field's postings are rebuilt from the stored
/// original text into morphemes, and the same query -- now parsed under the
/// SAME new analyzer via `unified_query_parser` -- matches both documents,
/// exactly like a field defined with the Japanese analyzer from the start
/// (`japanese_lexical_search_test.rs`).
#[tokio::test(flavor = "multi_thread")]
async fn update_field_switches_to_japanese_analyzer_and_changes_search_behavior() -> Result<()> {
    let engine = engine_with_default_analyzer().await?;

    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_text("body", "吾輩は猫である。名前はまだ無い。")
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_text("body", "猫が吾輩を見た日のことである。")
                .build(),
        )
        .await?;
    engine.commit().await?;

    // Before: the default analyzer does not split the clause into
    // morphemes, so a partial-clause query matches nothing.
    let parser = engine.unified_query_parser()?;
    let request = parser.parse("吾輩は猫").await?;
    let before = engine.search(request).await?;
    assert!(
        before.is_empty(),
        "the default analyzer should not tokenize unspaced Japanese into \
         matchable morphemes, got {before:?}"
    );

    let new_option = SchemaFieldOption::Text(TextOption::default().analyzer(
        BuiltinAnalyzerSpec::Japanese {
            mode: "normal".into(),
            dict: "embedded://ipadic".into(),
            user_dict: None,
        },
    ));
    let outcome = engine
        .update_field(
            "body",
            new_option,
            UpdateFieldOptions {
                reindex: true,
                ..Default::default()
            },
        )
        .await?;
    assert_eq!(outcome.classification, FieldChangeKind::Reindex);

    // After: postings were rebuilt from the stored original text under
    // Lindera, so the bare query's morphemes OR-match both documents.
    let parser = engine.unified_query_parser()?;
    let request = parser.parse("吾輩は猫").await?;
    let results = engine.search(request).await?;
    let ids: Vec<&str> = results.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        ids.len(),
        2,
        "both docs share morphemes with the query after the rebuild, got {ids:?}"
    );

    Ok(())
}

/// #1083: before `term_vectors` was wired to the write path, flipping it
/// `false -> true` had no observable effect at all -- postings were
/// unconditionally written with positions regardless of the setting. Now
/// that positions are actually withheld when `term_vectors: false`, the
/// same `update_field(reindex: true)` path proven above for an analyzer
/// change must also rebuild a field's postings to add positions, turning
/// a previously-non-matching phrase query into a match.
#[tokio::test(flavor = "multi_thread")]
async fn update_field_enables_term_vectors_and_phrase_query_starts_matching() -> Result<()> {
    let schema: Schema = serde_json::from_str(
        r#"{
            "default_fields": ["body"],
            "fields": {
                "body": { "Text": { "indexed": true, "stored": true, "term_vectors": false } }
            }
        }"#,
    )
    .expect("valid schema JSON");
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let engine = Engine::new(storage, schema).await?;

    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_text("body", "the quick brown fox")
                .build(),
        )
        .await?;
    engine.commit().await?;

    // Before: `term_vectors: false` means no positions on disk, so the
    // phrase query is rejected rather than silently matching nothing (#1247).
    let parser = engine.unified_query_parser()?;
    let request = parser.parse("body:\"quick brown\"").await?;
    let before = engine.search(request).await;
    assert!(
        matches!(before, Err(LaurusError::Query(_))),
        "a phrase on a term_vectors: false field must be rejected, got {before:?}"
    );

    let new_option = SchemaFieldOption::Text(TextOption::default().term_vectors(true));
    let outcome = engine
        .update_field(
            "body",
            new_option,
            UpdateFieldOptions {
                reindex: true,
                ..Default::default()
            },
        )
        .await?;
    assert_eq!(outcome.classification, FieldChangeKind::Reindex);

    // After: postings were rebuilt from the stored original text with
    // positions, so the same phrase query now matches.
    let parser = engine.unified_query_parser()?;
    let request = parser.parse("body:\"quick brown\"").await?;
    let after = engine.search(request).await?;
    let ids: Vec<&str> = after.iter().map(|h| h.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["doc1"],
        "the rebuild must add positions so the phrase query matches"
    );

    Ok(())
}

/// Storage decorator failing the next `create_output` whose name has the
/// armed prefix -- aimed at `InvertedIndex::rebuild_field`'s rebuilt
/// segment (named `merged_<generation>`, see `inverted.rs`), so the rebuild
/// dies before it ever calls `segment_manifest::publish_with`.
#[derive(Debug)]
struct FailingStorage {
    inner: Arc<dyn Storage>,
    fail_create_with_prefix: parking_lot::Mutex<Option<String>>,
}

impl FailingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            fail_create_with_prefix: parking_lot::Mutex::new(None),
        }
    }

    fn fail_next_create_with_prefix(&self, prefix: &str) {
        *self.fail_create_with_prefix.lock() = Some(prefix.to_string());
    }
}

impl Storage for FailingStorage {
    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        let armed = {
            let mut guard = self.fail_create_with_prefix.lock();
            if guard.as_ref().is_some_and(|p| name.starts_with(p)) {
                *guard = None;
                true
            } else {
                false
            }
        };
        if armed {
            return Err(LaurusError::storage(format!(
                "injected failure creating {name}"
            )));
        }
        self.inner.create_output(name)
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.inner.open_input(name)
    }

    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(name)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(name)
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.inner.rename_file(old_name, new_name)
    }

    fn list_files(&self) -> Result<Vec<String>> {
        self.inner.list_files()
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.inner.file_size(name)
    }

    fn sync(&self) -> Result<()> {
        self.inner.sync()
    }

    fn metadata(&self, name: &str) -> Result<laurus::storage::FileMetadata> {
        self.inner.metadata(name)
    }

    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn doc(body: &str) -> Document {
    Document::builder().add_text("title", body).build()
}

/// The segment ids the manifest -- the sole publication record -- lists.
/// Mirrors `merge_failure_publication_test.rs`'s helper: the manifest file
/// is a varint-length-prefixed payload, not bare JSON.
fn list_manifest_ids(storage: &Arc<dyn Storage>) -> Vec<String> {
    let mut input = storage.open_input("segments.json").unwrap();
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes).unwrap();
    let payload: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => {
            let mut len: u64 = 0;
            let mut shift = 0;
            let mut cursor = 0usize;
            loop {
                let byte = bytes[cursor];
                cursor += 1;
                len |= u64::from(byte & 0x7F) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            serde_json::from_slice(&bytes[cursor..cursor + len as usize]).unwrap()
        }
    };
    let mut ids: Vec<String> = payload["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["segment_id"].as_str().unwrap().to_string())
        .collect();
    ids.sort();
    ids
}

/// A rebuild that dies partway through (here, writing the rebuilt
/// segment's first file) must not publish anything: the manifest must
/// still list exactly the original segment, and the field's data under
/// the OLD analyzer must still be intact and searchable.
#[test]
fn rebuild_field_failure_leaves_original_segment_and_manifest_untouched() {
    let inner: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let failing = Arc::new(FailingStorage::new(inner.clone()));
    let storage: Arc<dyn Storage> = failing.clone();

    let config = LexicalIndexConfig::builder()
        .add_field("title", LexicalFieldOption::Text(TextOption::default()))
        .build();
    let store = LexicalStore::new(storage.clone(), config).unwrap();
    store.upsert_document(1, doc("Rust Programming")).unwrap();
    store.commit().unwrap();

    let before_ids = list_manifest_ids(&inner);

    // `rebuild_field` reserves a fresh `merged_<generation>` id for every
    // source segment up front (see `InvertedIndex::rebuild_field`) and
    // writes each rebuilt segment before publishing any of them.
    failing.fail_next_create_with_prefix("merged_");
    let new_analyzer: Arc<dyn laurus::Analyzer> =
        Arc::new(laurus::analysis::analyzer::keyword::KeywordAnalyzer::new());
    let result = store.rebuild_field(
        "title",
        LexicalFieldOption::Text(TextOption::default().analyzer("keyword")),
        Some(new_analyzer),
    );
    assert!(
        result.is_err(),
        "the injected failure creating the rebuilt segment must surface"
    );

    // The manifest is byte-for-byte the same set of segment ids as before
    // the failed rebuild -- nothing was published.
    let after_ids = list_manifest_ids(&inner);
    assert_eq!(
        before_ids, after_ids,
        "a failed rebuild must not change the published segment set"
    );

    // The original document is still searchable under the ORIGINAL
    // (default) analyzer -- e.g. a single tokenized word still matches,
    // which it would NOT under `keyword` (whole-field exact match only).
    let hits = store
        .search(LexicalSearchRequest::new(Box::new(
            laurus::lexical::TermQuery::new("title", "programming"),
        )))
        .unwrap();
    assert_eq!(
        hits.hits.len(),
        1,
        "a failed rebuild must leave the field's existing data intact and searchable"
    );
}
