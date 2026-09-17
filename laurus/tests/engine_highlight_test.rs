//! End-to-end tests for search-result highlighting through `Engine::search`
//! (Issue #1134): `SearchRequestBuilder::highlight` / `highlight_config` in,
//! `SearchResult::highlights` out.

use laurus::lexical::Query;
use laurus::lexical::TermQuery;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::vector::{FlatOption, Vector};
use laurus::{
    DataValue, Document, Engine, FieldOption, FusionAlgorithm, HighlightConfig, HighlightOptions,
    IntegerOption, LexicalSearchQuery, QueryVector, Result, Schema, SearchRequestBuilder,
    SearchResult, TextOption, VectorSearchQuery,
};

async fn engine_with(schema: Schema) -> Result<Engine> {
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    Engine::new(storage, schema).await
}

/// `title` + `body`, both stored text fields with the default analyzer.
async fn text_engine() -> Result<Engine> {
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("body", FieldOption::Text(TextOption::default()))
        .build();
    let engine = engine_with(schema).await?;
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "Systems languages")
                .add_field("body", "Rust is a systems programming language")
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("title", "Vector search")
                .add_field("body", "Nearest neighbour search over embeddings")
                .build(),
        )
        .await?;
    engine.commit().await?;
    Ok(engine)
}

fn term(field: &str, term: &str) -> LexicalSearchQuery {
    LexicalSearchQuery::Obj(Box::new(TermQuery::new(field, term)) as Box<dyn Query>)
}

fn only<'a>(results: &'a [SearchResult], id: &str) -> &'a SearchResult {
    results
        .iter()
        .find(|r| r.id == id)
        .unwrap_or_else(|| panic!("expected a hit for {id}, got {results:?}"))
}

#[tokio::test(flavor = "multi_thread")]
async fn lexical_search_highlights_requested_stored_text_field() -> Result<()> {
    let engine = text_engine().await?;

    let request = SearchRequestBuilder::new()
        .lexical_query(term("body", "rust"))
        .highlight(vec!["body".to_string()])
        .build();
    let results = engine.search(request).await?;

    assert_eq!(results.len(), 1);
    let hit = only(&results, "doc1");
    let body = &hit.highlights["body"];
    assert_eq!(body.len(), 1);
    assert!(
        body[0].contains("<mark>Rust</mark>"),
        "original casing inside the tag: {body:?}"
    );
    assert!(
        !hit.highlights.contains_key("title"),
        "unrequested fields carry no highlights: {:?}",
        hit.highlights
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn dsl_request_highlights_after_setting_lexical_options() -> Result<()> {
    let engine = text_engine().await?;

    let mut request = engine.unified_query_parser()?.parse("body:rust").await?;
    request.lexical_options.highlight = Some(HighlightOptions::new(vec!["body".to_string()]));
    let results = engine.search(request).await?;

    assert!(only(&results, "doc1").highlights["body"][0].contains("<mark>Rust</mark>"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn highlights_are_empty_when_not_requested() -> Result<()> {
    let engine = text_engine().await?;

    let request = SearchRequestBuilder::new()
        .lexical_query(term("body", "rust"))
        .build();
    let results = engine.search(request).await?;

    assert!(!results.is_empty());
    assert!(results.iter().all(|r| r.highlights.is_empty()));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn hybrid_search_highlights_lexical_matches_only() -> Result<()> {
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field(
            "embedding",
            FieldOption::Flat(FlatOption::default().dimension(3)),
        )
        .build();
    let engine = engine_with(schema).await?;
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", DataValue::Text("Rust Programming".into()))
                .add_field("embedding", DataValue::Vector(vec![1.0, 0.0, 0.0]))
                .build(),
        )
        .await?;
    engine
        .put_document(
            "doc2",
            Document::builder()
                .add_field("title", DataValue::Text("Vector Search".into()))
                .add_field("embedding", DataValue::Vector(vec![0.0, 1.0, 0.0]))
                .build(),
        )
        .await?;
    engine.commit().await?;

    let request = SearchRequestBuilder::new()
        .lexical_query(term("title", "rust"))
        .vector_query(VectorSearchQuery::Vectors(vec![QueryVector {
            vector: Vector::new(vec![0.0, 1.0, 0.0]),
            weight: 1.0,
            fields: None,
        }]))
        .fusion_algorithm(FusionAlgorithm::RRF { k: 60.0 })
        .highlight(vec!["title".to_string()])
        .build();
    let results = engine.search(request).await?;

    assert_eq!(
        results.len(),
        2,
        "fused branch returns both hits: {results:?}"
    );
    assert!(
        only(&results, "doc1").highlights["title"][0].contains("<mark>Rust</mark> Programming")
    );
    assert!(
        only(&results, "doc2").highlights.is_empty(),
        "a vector-only hit has no lexical match to highlight"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn non_stored_field_is_not_highlighted() -> Result<()> {
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field(
            "body",
            FieldOption::Text(TextOption::default().stored(false)),
        )
        .build();
    let engine = engine_with(schema).await?;
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "stored")
                .add_field("body", "rust is indexed but not stored")
                .build(),
        )
        .await?;
    engine.commit().await?;

    let request = SearchRequestBuilder::new()
        .lexical_query(term("body", "rust"))
        .highlight(vec!["body".to_string()])
        .build();
    let results = engine.search(request).await?;

    let hit = only(&results, "doc1");
    assert!(
        hit.document
            .as_ref()
            .is_some_and(|d| d.get("body").is_none()),
        "the non-stored field is not in the document: {:?}",
        hit.document
    );
    assert!(hit.highlights.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn non_text_and_unknown_fields_are_skipped() -> Result<()> {
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("count", FieldOption::Integer(IntegerOption::default()))
        .build();
    let engine = engine_with(schema).await?;
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "rust")
                .add_field("count", 3i64)
                .build(),
        )
        .await?;
    engine.commit().await?;

    let request = SearchRequestBuilder::new()
        .lexical_query(term("title", "rust"))
        .highlight(vec![
            "count".to_string(),
            "nope".to_string(),
            "title".to_string(),
        ])
        .build();
    let results = engine.search(request).await?;

    let keys: Vec<&String> = only(&results, "doc1").highlights.keys().collect();
    assert_eq!(keys, [&"title".to_string()]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn require_field_match_false_highlights_terms_from_other_fields() -> Result<()> {
    let engine = engine_with(
        Schema::builder()
            .add_field("title", FieldOption::Text(TextOption::default()))
            .add_field("body", FieldOption::Text(TextOption::default()))
            .build(),
    )
    .await?;
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "Rust")
                .add_field("body", "learn rust")
                .build(),
        )
        .await?;
    engine.commit().await?;

    let fields = || vec!["title".to_string(), "body".to_string()];

    let strict = SearchRequestBuilder::new()
        .lexical_query(term("title", "rust"))
        .highlight(fields())
        .build();
    let hit_strict = engine.search(strict).await?;
    let strict_keys: Vec<&String> = only(&hit_strict, "doc1").highlights.keys().collect();
    assert_eq!(strict_keys, [&"title".to_string()]);

    let relaxed = SearchRequestBuilder::new()
        .lexical_query(term("title", "rust"))
        .highlight(fields())
        .highlight_config(HighlightConfig::default().require_field_match(false))
        .build();
    let hit_relaxed = engine.search(relaxed).await?;
    let highlights = &only(&hit_relaxed, "doc1").highlights;
    assert!(highlights.contains_key("title") && highlights.contains_key("body"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn highlight_config_tag_and_css_class_are_applied() -> Result<()> {
    let engine = text_engine().await?;

    let request = SearchRequestBuilder::new()
        .lexical_query(term("body", "rust"))
        .highlight(vec!["body".to_string()])
        .highlight_config(
            HighlightConfig::default()
                .tag("em".to_string())
                .css_class("hl".to_string()),
        )
        .build();
    let results = engine.search(request).await?;

    let fragment = &only(&results, "doc1").highlights["body"][0];
    assert!(
        fragment.contains(r#"<em class="hl">Rust</em>"#),
        "{fragment}"
    );
    assert!(!fragment.contains("<mark>"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn filter_query_terms_are_not_highlighted() -> Result<()> {
    let schema = Schema::builder()
        .add_field("title", FieldOption::Text(TextOption::default()))
        .add_field("category", FieldOption::Text(TextOption::default()))
        .build();
    let engine = engine_with(schema).await?;
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("title", "search engine")
                .add_field("category", "rust")
                .build(),
        )
        .await?;
    engine.commit().await?;

    let request = SearchRequestBuilder::new()
        .lexical_query(term("title", "search"))
        .filter_query(Box::new(TermQuery::new("category", "rust")))
        .highlight(vec!["title".to_string(), "category".to_string()])
        .build();
    let results = engine.search(request).await?;

    let highlights = &only(&results, "doc1").highlights;
    assert!(highlights["title"][0].contains("<mark>search</mark>"));
    assert!(
        !highlights.contains_key("category"),
        "request-level filter terms must not highlight: {highlights:?}"
    );
    Ok(())
}

/// The `keyword` analyzer keeps the whole value as one token, so the
/// highlight covers `Rust Lang` in one go — a `StandardAnalyzer` would
/// have produced two tokens. This proves the engine picks the field's
/// own analyzer.
#[tokio::test(flavor = "multi_thread")]
async fn per_field_keyword_analyzer_drives_highlighting() -> Result<()> {
    let schema = Schema::builder()
        .add_field(
            "category",
            FieldOption::Text(TextOption::default().analyzer("keyword")),
        )
        .build();
    let engine = engine_with(schema).await?;
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("category", "Rust Lang")
                .build(),
        )
        .await?;
    engine.commit().await?;

    let mut request = engine
        .unified_query_parser()?
        .parse(r#"category:"Rust Lang""#)
        .await?;
    request.lexical_options.highlight = Some(HighlightOptions::new(vec!["category".to_string()]));
    let results = engine.search(request).await?;

    let fragment = &only(&results, "doc1").highlights["category"][0];
    assert_eq!(fragment, "<mark>Rust Lang</mark>");
    Ok(())
}

/// Schema JSON shape from `japanese_lexical_search_test.rs`: a per-field
/// Lindera analyzer. With the default `\w+` tokenizer the whole sentence is
/// one token and nothing would highlight.
const JAPANESE_SCHEMA_JSON: &str = r#"
{
    "default_fields": ["body"],
    "fields": {
        "body": { "Text": {
            "indexed": true, "stored": true,
            "analyzer": { "language": "japanese", "mode": "normal", "dict": "embedded://ipadic" }
        }}
    }
}
"#;

#[tokio::test(flavor = "multi_thread")]
async fn per_field_japanese_analyzer_drives_highlighting() -> Result<()> {
    let schema: Schema = serde_json::from_str(JAPANESE_SCHEMA_JSON).expect("valid schema JSON");
    let engine = engine_with(schema).await?;
    engine
        .put_document(
            "doc1",
            Document::builder()
                .add_field("body", "吾輩は猫である。名前はまだ無い。")
                .build(),
        )
        .await?;
    engine.commit().await?;

    let mut request = engine.unified_query_parser()?.parse("body:猫").await?;
    request.lexical_options.highlight = Some(HighlightOptions::new(vec!["body".to_string()]));
    let results = engine.search(request).await?;

    let fragment = &only(&results, "doc1").highlights["body"][0];
    assert!(fragment.contains("<mark>猫</mark>"), "{fragment}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn search_batch_propagates_highlights() -> Result<()> {
    let engine = text_engine().await?;

    let with_highlight = SearchRequestBuilder::new()
        .lexical_query(term("body", "rust"))
        .highlight(vec!["body".to_string()])
        .build();
    let without_highlight = SearchRequestBuilder::new()
        .lexical_query(term("body", "rust"))
        .build();
    let batches = engine
        .search_batch(vec![with_highlight, without_highlight])
        .await?;

    assert_eq!(batches.len(), 2);
    assert!(only(&batches[0], "doc1").highlights["body"][0].contains("<mark>Rust</mark>"));
    assert!(only(&batches[1], "doc1").highlights.is_empty());
    Ok(())
}
