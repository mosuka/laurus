//! Issue #1546: a lexical field deleted with `delete_field` can be added
//! again under the same name, and starts empty.
//!
//! Before the fix, a field from the schema the engine was opened with could
//! not be added again in the same process ("already exists in the lexical
//! index"), and kept being indexed under its old option. A field added at
//! run time could, but got its old postings back. Deleting a lexical field
//! now discards its data from every segment.

use std::sync::Arc;

use tempfile::TempDir;

use laurus::lexical::{NumericRangeQuery, Query, TermQuery, TextOption};
use laurus::storage::file::FileStorageConfig;
use laurus::storage::{Storage, StorageConfig, StorageFactory};
use laurus::{
    DataValue, Document, DynamicFieldPolicy, Engine, FieldOption, IntegerOption,
    LexicalSearchQuery, Schema, SearchRequestBuilder,
};

const FIELD: &str = "tag";

fn open_storage(dir: &TempDir) -> Arc<dyn Storage> {
    StorageFactory::create(StorageConfig::File(FileStorageConfig::new(dir.path()))).unwrap()
}

fn text() -> FieldOption {
    FieldOption::Text(TextOption::default())
}

/// Put `id` with a title and, if given, `value` in [`FIELD`].
async fn put(engine: &Engine, id: &str, value: Option<DataValue>) {
    let mut doc = Document::builder().add_field("title", DataValue::Text(format!("doc {id}")));
    if let Some(value) = value {
        doc = doc.add_field(FIELD, value);
    }
    engine.put_document(id, doc.build()).await.unwrap();
}

/// The ids `query` finds, sorted.
async fn ids(engine: &Engine, query: Box<dyn Query>) -> Vec<String> {
    let request = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(query))
        .limit(100)
        .build();
    let mut ids: Vec<String> = engine
        .search(request)
        .await
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    ids.sort();
    ids
}

async fn tagged(engine: &Engine, term: &str) -> Vec<String> {
    ids(engine, Box::new(TermQuery::new(FIELD, term))).await
}

/// A text field from the schema at open, and one added at run time, are
/// deleted and added again. Neither a committed document nor one not yet
/// committed at the deletion is found by its old term afterwards; a
/// document put after the re-add is, also after a reopen.
#[tokio::test(flavor = "multi_thread")]
async fn a_deleted_text_field_can_be_added_again_and_starts_empty() {
    let mut wrong = Vec::new();
    for case in ["from the schema at open", "added at run time"] {
        let dir = TempDir::new().unwrap();
        let mut schema = Schema::builder().add_field("title", text());
        if case == "from the schema at open" {
            schema = schema.add_field(FIELD, text());
        }
        let engine = Engine::new(open_storage(&dir), schema.build())
            .await
            .unwrap();
        if case == "added at run time" {
            engine.add_field(FIELD, text()).await.unwrap();
        }

        put(&engine, "1", Some(DataValue::Text("alpha".into()))).await;
        engine.commit().await.unwrap();
        // Still buffered in the lexical writer when the field is deleted, so
        // not searchable yet: a search sees committed documents only.
        put(&engine, "2", Some(DataValue::Text("alpha".into()))).await;
        if tagged(&engine, "alpha").await != ["1"] {
            wrong.push(format!("{case}: the setup does not find 1"));
            continue;
        }

        engine.delete_field(FIELD).await.unwrap();
        if let Err(e) = engine.add_field(FIELD, text()).await {
            wrong.push(format!("{case}: the re-add fails: {e}"));
            continue;
        }
        let after_readd = tagged(&engine, "alpha").await;
        if !after_readd.is_empty() {
            wrong.push(format!("{case}: the re-added field finds {after_readd:?}"));
        }
        // The document store keeps the value document 1 was put with, as for
        // a vector field (#1535), though no search finds it by that value.
        let stored = engine.get_documents("1").await.unwrap();
        let stored_value = stored.first().and_then(|doc| doc.get(FIELD)).cloned();
        if stored_value != Some(DataValue::Text("alpha".into())) {
            wrong.push(format!(
                "{case}: the stored value after the re-add: {stored_value:?}"
            ));
        }

        put(&engine, "3", Some(DataValue::Text("alpha".into()))).await;
        engine.commit().await.unwrap();
        let new_put = tagged(&engine, "alpha").await;
        if new_put != ["3"] {
            wrong.push(format!("{case}: after a new put: {new_put:?}"));
        }

        let schema = engine.schema();
        drop(engine);
        let reopened = Engine::new(open_storage(&dir), schema).await.unwrap();
        let after_reopen = tagged(&reopened, "alpha").await;
        if after_reopen != ["3"] {
            wrong.push(format!("{case}: after a reopen: {after_reopen:?}"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// An integer field's BKD points are discarded with it: a range over the
/// re-added field finds only a document put after the re-add.
#[tokio::test(flavor = "multi_thread")]
async fn a_re_added_integer_field_finds_no_old_points() {
    let dir = TempDir::new().unwrap();
    let integer = || FieldOption::Integer(IntegerOption::default());
    let schema = Schema::builder()
        .add_field("title", text())
        .add_field(FIELD, integer())
        .build();
    let engine = Engine::new(open_storage(&dir), schema).await.unwrap();
    let range =
        || -> Box<dyn Query> { Box::new(NumericRangeQuery::i64_range(FIELD, Some(0), Some(100))) };

    put(&engine, "1", Some(DataValue::Int64(10))).await;
    put(&engine, "2", Some(DataValue::Int64(20))).await;
    engine.commit().await.unwrap();
    assert_eq!(ids(&engine, range()).await, ["1", "2"]);

    engine.delete_field(FIELD).await.unwrap();
    engine.add_field(FIELD, integer()).await.unwrap();
    put(&engine, "3", Some(DataValue::Int64(30))).await;
    engine.commit().await.unwrap();

    assert_eq!(ids(&engine, range()).await, ["3"]);
    drop(engine);
}

/// Under `DynamicFieldPolicy::Dynamic`, a put that carries a field deleted
/// from the schema at open adds the field back, with the option inferred
/// from its value, and only that put's document is found by it. Before the
/// fix the re-add failed, the failure was taken for a concurrent add, and
/// the value was indexed under the deleted field's old option while the
/// schema did not have the field.
#[tokio::test(flavor = "multi_thread")]
async fn a_dynamic_put_adds_a_deleted_field_back_empty() {
    let dir = TempDir::new().unwrap();
    let mut schema = Schema::builder()
        .add_field("title", text())
        .add_field(FIELD, text())
        .build();
    schema.dynamic_field_policy = DynamicFieldPolicy::Dynamic;
    let engine = Engine::new(open_storage(&dir), schema).await.unwrap();

    put(&engine, "1", Some(DataValue::Text("alpha".into()))).await;
    engine.commit().await.unwrap();
    engine.delete_field(FIELD).await.unwrap();

    put(&engine, "2", Some(DataValue::Text("alpha".into()))).await;
    engine.commit().await.unwrap();

    assert!(
        engine.schema().fields.contains_key(FIELD),
        "the put must add the field back to the schema"
    );
    assert_eq!(tagged(&engine, "alpha").await, ["2"]);
    drop(engine);
}
