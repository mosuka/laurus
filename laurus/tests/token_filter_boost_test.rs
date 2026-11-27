//! A `boost` token filter weights the field's query terms (Issue #1701).
//!
//! The filter set each token's boost, but nothing read it: neither indexing
//! nor the query parser, so a field with `boost = 10.0`, `nan` or `-2.0`
//! scored exactly like one without the filter. The query parser now
//! multiplies a query term's boost by its tokens' boost, as Lucene's
//! `QueryBuilder` does, and a factor that is not finite or is negative is
//! rejected when an index is created. An index created before that check
//! still opens, with such a filter left out, so it keeps the results it had.

use std::collections::HashMap;
use std::path::Path;

use laurus::index_dir;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{Document, Engine, Schema};

/// The fields hold the same text and differ only in their analyzer, a
/// whitespace tokenizer with no filter or with a `boost` filter of the
/// factor named in the field.
fn schema_toml(boosts: &[(&str, &str)]) -> String {
    let mut toml = String::from(
        "[fields.plain.Text]\nindexed = true\nstored = true\nanalyzer = \"plain\"\n\n\
         [analyzers.plain.tokenizer]\ntype = \"whitespace\"\n\n",
    );
    for (field, boost) in boosts {
        toml.push_str(&format!(
            "[fields.{field}.Text]\nindexed = true\nstored = true\nanalyzer = \"{field}\"\n\n\
             [analyzers.{field}.tokenizer]\ntype = \"whitespace\"\n\n\
             [[analyzers.{field}.token_filters]]\ntype = \"boost\"\nboost = {boost}\n\n"
        ));
    }
    toml
}

/// Put three documents into every field of `fields`, commit, and return
/// each field's scores for `rust`, by document id.
async fn rust_scores(engine: &Engine, fields: &[&str]) -> HashMap<String, HashMap<String, f32>> {
    for (id, text) in [
        ("doc1", "rust is fast"),
        ("doc2", "rust rust and go"),
        ("doc3", "go is simple"),
    ] {
        let mut builder = Document::builder();
        for field in fields {
            builder = builder.add_field(*field, text);
        }
        engine.put_document(id, builder.build()).await.unwrap();
    }
    engine.commit().await.unwrap();

    let parser = engine.unified_query_parser().unwrap();
    let mut scores = HashMap::new();
    for field in fields {
        let request = parser.parse(&format!("{field}:rust")).await.unwrap();
        let results = engine
            .search(request)
            .await
            .map(|response| response.results)
            .unwrap_or_else(|e| panic!("{field}:rust: {e}"));
        let by_id = results
            .iter()
            .map(|hit| (hit.id.clone(), hit.score))
            .collect::<HashMap<_, _>>();
        assert_eq!(by_id.len(), 2, "{field}: {by_id:?}");
        scores.insert(field.to_string(), by_id);
    }
    scores
}

/// BM25 is linear in a query's boost, so a field with `boost = 10.0`
/// scores 10 times the field without the filter. It used to score the same.
#[tokio::test(flavor = "multi_thread")]
async fn a_boost_filter_multiplies_the_fields_scores() {
    let storage =
        StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default())).unwrap();
    let schema = Schema::from_toml(&schema_toml(&[("boosted", "10.0")])).unwrap();
    let engine = Engine::new(storage, schema).await.unwrap();

    let scores = rust_scores(&engine, &["plain", "boosted"]).await;
    for (id, plain) in &scores["plain"] {
        let boosted = scores["boosted"][id];
        assert!(
            (boosted - 10.0 * plain).abs() <= 1e-5 * boosted.abs(),
            "{id}: plain {plain}, boosted {boosted}"
        );
    }
}

/// An index whose `schema.toml` holds a factor creation now rejects (here
/// written by hand) opens, and the filter is left out: the field scores
/// exactly like the one without the filter, as before Issue #1701.
#[tokio::test(flavor = "multi_thread")]
async fn an_existing_index_with_an_out_of_range_factor_opens_without_the_filter() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join(index_dir::SCHEMA_FILE),
        schema_toml(&[("nan", "nan"), ("negative", "-2.0")]),
    )
    .unwrap();

    let (schema, storage, claim) = index_dir::open_or_create(dir.path(), None).unwrap();
    let engine = Engine::new(storage, schema).await.unwrap();
    drop(claim);

    let scores = rust_scores(&engine, &["plain", "nan", "negative"]).await;
    assert_eq!(scores["nan"], scores["plain"]);
    assert_eq!(scores["negative"], scores["plain"]);
}

/// Creating an index whose schema has a factor that is not finite fails,
/// naming the analyzer and the filter, and creates nothing.
#[test]
fn creating_an_index_with_an_out_of_range_factor_fails_and_creates_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let index = dir.path().join("index");
    for boost in ["nan", "inf", "-1.0"] {
        let schema = Schema::from_toml(&schema_toml(&[("boosted", boost)])).unwrap();
        let err = index_dir::open_or_create(&index, Some(schema))
            .err()
            .unwrap_or_else(|| panic!("{boost}: the index was created"));
        let message = err.to_string();
        assert!(
            message.contains(
                "Analyzer 'boosted': token_filters[0]: boost must be finite and at least 0"
            ),
            "{boost}: {message}"
        );
        assert!(nothing_in(&index), "{boost}: the create left files behind");
    }
}

fn nothing_in(index: &Path) -> bool {
    !index.join(index_dir::SCHEMA_FILE).exists() && !index.join(index_dir::STORE_DIR).exists()
}
