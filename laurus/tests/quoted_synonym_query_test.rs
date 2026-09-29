//! Issue #1252: quoted queries through an analyzer that stacks synonyms.
//!
//! `SynonymGraphFilter` puts a synonym at the same position as the word it
//! expands (`position_increment = 0`). A quoted value must match any of the
//! stacked alternatives, and a phrase must follow the stacked positions,
//! whether the synonyms run at index time and query time (the `Engine`
//! path, which uses one analyzer for both) or at query time only.

use std::sync::Arc;

use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::pipeline::PipelineAnalyzer;
use laurus::analysis::synonym::dictionary::SynonymDictionary;
use laurus::analysis::token_filter::lowercase::LowercaseFilter;
use laurus::analysis::token_filter::synonym_graph::SynonymGraphFilter;
use laurus::analysis::tokenizer::whitespace::WhitespaceTokenizer;
use laurus::lexical::query::parser::LexicalQueryParser;
use laurus::storage::memory::MemoryStorageConfig;
use laurus::{
    Document, Engine, LaurusError, LexicalSearchQuery, Result, Schema, SearchRequestBuilder,
    StorageConfig, StorageFactory,
};

/// `syn` stacks synonyms at index time, `plain` does not. `tags` is a
/// multi-valued `syn` field, and `nopos` a `syn` field without positions.
const SCHEMA_TOML: &str = r#"
default_fields = ["syn"]

[fields.syn.Text]
indexed = true
stored = true
analyzer = "syn"

[fields.plain.Text]
indexed = true
stored = true
analyzer = "plain"

[fields.tags.Text]
indexed = true
stored = true
multi_valued = true
analyzer = "syn"

[fields.nopos.Text]
indexed = true
stored = true
term_vectors = false
analyzer = "syn"
"#;

fn plain_analyzer() -> PipelineAnalyzer {
    PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new()))
        .add_filter(Arc::new(LowercaseFilter::new()))
}

fn syn_analyzer() -> Arc<dyn Analyzer> {
    let mut dict = SynonymDictionary::new(None).unwrap();
    dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);
    dict.add_synonym_group(vec!["ml".to_string(), "machine learning".to_string()]);
    Arc::new(plain_analyzer().add_filter(Arc::new(SynonymGraphFilter::new(dict, true))))
}

const DOCS: [(&str, &str); 4] = [
    ("big_dog", "a big dog barks"),
    ("large_dog", "a large dog barks"),
    ("ml", "ml is fun"),
    ("machine_learning", "machine learning is fun"),
];

async fn engine() -> Result<Engine> {
    let schema = Schema::from_toml(SCHEMA_TOML)?;
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let engine = Engine::builder(storage, schema)
        .register_runtime_analyzer("syn", syn_analyzer())
        .register_runtime_analyzer("plain", Arc::new(plain_analyzer()))
        .build()
        .await?;
    for (id, text) in DOCS {
        engine
            .put_document(
                id,
                Document::builder()
                    .add_text("syn", text)
                    .add_text("plain", text)
                    .add_text("nopos", text)
                    .build(),
            )
            .await?;
    }
    engine
        .put_document(
            "tags",
            Document::builder()
                .add_text_array("tags", vec!["big".to_string(), "dog".to_string()])
                .build(),
        )
        .await?;
    engine.commit().await?;
    Ok(engine)
}

async fn ids(engine: &Engine, request: laurus::SearchRequest) -> Result<Vec<String>> {
    let mut ids: Vec<String> = engine
        .search(request)
        .await?
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    ids.sort();
    Ok(ids)
}

async fn search_dsl(engine: &Engine, dsl: &str) -> Result<Vec<String>> {
    let request = SearchRequestBuilder::new().query_dsl(dsl).limit(10).build();
    ids(engine, request).await
}

/// Parse `dsl` with the synonym analyzer and run it against `engine`, so
/// synonyms apply at query time only when the field is `plain`.
async fn search_query_time(engine: &Engine, dsl: &str) -> Result<Vec<String>> {
    let query = LexicalQueryParser::new(syn_analyzer()).parse(dsl)?;
    let request = SearchRequestBuilder::new()
        .lexical_query(LexicalSearchQuery::Obj(query))
        .limit(10)
        .build();
    ids(engine, request).await
}

fn sorted(ids: &[&str]) -> Vec<String> {
    let mut ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    ids.sort();
    ids
}

const DOGS: &[&str] = &["big_dog", "large_dog"];
const MLS: &[&str] = &["ml", "machine_learning"];

#[tokio::test(flavor = "multi_thread")]
async fn index_and_query_time_synonyms() -> Result<()> {
    let engine = engine().await?;
    let cases: &[(&str, &[&str])] = &[
        ("syn:big", DOGS),
        // One position with two stacked tokens: either one matches.
        ("syn:\"big\"", DOGS),
        ("syn:\"large\"", DOGS),
        // A phrase through a stacked position.
        ("syn:\"big dog\"", DOGS),
        ("syn:\"a big dog\"", DOGS),
        ("syn:\"a large dog barks\"", DOGS),
        ("syn:\"dog barks\"", DOGS),
        // Multi-word synonyms: one path per alternative.
        ("syn:\"ml\"", MLS),
        ("syn:\"machine learning\"", MLS),
        ("syn:\"ml is\"", MLS),
        ("syn:\"machine learning is fun\"", MLS),
        // With the synonym, "ml is fun" also reads "machine learning is
        // fun", so a phrase inside that span matches it.
        ("syn:\"learning is\"", MLS),
        ("syn:\"big fun\"", &[]),
    ];
    for (dsl, expected) in cases {
        assert_eq!(search_dsl(&engine, dsl).await?, sorted(expected), "{dsl}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn query_time_only_synonyms() -> Result<()> {
    let engine = engine().await?;
    let cases: &[(&str, &[&str])] = &[
        ("plain:\"big\"", DOGS),
        ("plain:\"big dog\"", DOGS),
        ("plain:\"a large dog\"", DOGS),
        ("plain:\"ml\"", MLS),
        ("plain:\"machine learning\"", MLS),
        ("plain:\"ml is\"", MLS),
        ("plain:\"learning is\"", &["machine_learning"]),
    ];
    for (dsl, expected) in cases {
        assert_eq!(
            search_query_time(&engine, dsl).await?,
            sorted(expected),
            "{dsl}"
        );
    }
    Ok(())
}

/// A stacked token must not push the next element further out: the phrase
/// may cross into the next element only once the slop reaches the gap
/// (100), as for any multi-valued field.
#[tokio::test(flavor = "multi_thread")]
async fn multi_valued_field_keeps_its_gap_after_a_stacked_token() -> Result<()> {
    let engine = engine().await?;
    assert_eq!(search_dsl(&engine, "tags:\"big dog\"").await?, sorted(&[]));
    assert_eq!(
        search_dsl(&engine, "tags:\"big dog\"~99").await?,
        sorted(&[])
    );
    assert_eq!(
        search_dsl(&engine, "tags:\"big dog\"~100").await?,
        sorted(&["tags"])
    );
    Ok(())
}

/// One stacked position is a term match and needs no positions; a phrase
/// still does, even when only one of its paths is a phrase.
#[tokio::test(flavor = "multi_thread")]
async fn positionless_field_accepts_one_position_and_rejects_phrases() -> Result<()> {
    let engine = engine().await?;
    assert_eq!(search_dsl(&engine, "nopos:\"big\"").await?, sorted(DOGS));
    for dsl in ["nopos:\"big dog\"", "nopos:\"ml\"", "nopos:\"ml is\""] {
        match search_dsl(&engine, dsl).await {
            Err(LaurusError::Query(message)) => {
                assert!(message.contains("nopos"), "{dsl}: {message}")
            }
            other => panic!("{dsl}: expected a query error, got {other:?}"),
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn phrase_through_a_synonym_is_highlighted() -> Result<()> {
    let engine = engine().await?;
    let request = SearchRequestBuilder::new()
        .query_dsl("syn:\"a big dog\"")
        .highlight(vec!["syn".to_string()])
        .limit(10)
        .build();
    let results = engine.search(request).await?;
    let hit = results
        .iter()
        .find(|hit| hit.id == "large_dog")
        .expect("large_dog must match");
    assert_eq!(
        hit.highlights["syn"],
        vec!["<mark>a large dog</mark> barks"]
    );
    Ok(())
}

// ---- Groups with several multi-word members (#1262) ----

const MEMBERS_SCHEMA_TOML: &str = r#"
default_fields = ["body"]

[fields.body.Text]
indexed = true
stored = true
analyzer = "members"
"#;

const MEMBER_DOCS: [(&str, &str); 7] = [
    ("ml", "ml is fun"),
    ("machine_learning", "machine learning is fun"),
    (
        "statistical_machine_learning",
        "statistical machine learning is fun",
    ),
    ("statistical_learning", "statistical learning is fun"),
    ("new_york", "new york is big"),
    ("big_apple", "the big apple is big"),
    ("new_apple", "a new apple is big"),
];

const MLS3: &[&str] = &["ml", "machine_learning", "statistical_machine_learning"];
const NYS: &[&str] = &["new_york", "big_apple"];

/// {ml, machine learning, statistical machine learning} has members of
/// different lengths, {new york, big apple} of the same length.
fn members_analyzer() -> Arc<dyn Analyzer> {
    let mut dict = SynonymDictionary::new(None).unwrap();
    dict.add_synonym_group(vec![
        "ml".to_string(),
        "machine learning".to_string(),
        "statistical machine learning".to_string(),
    ]);
    dict.add_synonym_group(vec!["new york".to_string(), "big apple".to_string()]);
    Arc::new(plain_analyzer().add_filter(Arc::new(SynonymGraphFilter::new(dict, true))))
}

async fn members_engine() -> Result<Engine> {
    members_engine_with(&MEMBER_DOCS).await
}

async fn members_engine_with(docs: &[(&str, &str)]) -> Result<Engine> {
    let schema = Schema::from_toml(MEMBERS_SCHEMA_TOML)?;
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
    let engine = Engine::builder(storage, schema)
        .register_runtime_analyzer("members", members_analyzer())
        .build()
        .await?;
    for (id, text) in docs {
        engine
            .put_document(id, Document::builder().add_text("body", *text).build())
            .await?;
    }
    engine.commit().await?;
    Ok(engine)
}

/// The highlights of every hit of `dsl` in `field`, by id.
async fn highlights(engine: &Engine, dsl: &str, field: &str) -> Result<Vec<(String, Vec<String>)>> {
    let request = SearchRequestBuilder::new()
        .query_dsl(dsl)
        .highlight(vec![field.to_string()])
        .limit(20)
        .build();
    let mut hits: Vec<(String, Vec<String>)> = engine
        .search(request)
        .await?
        .into_iter()
        .map(|hit| {
            let marked = hit.highlights.get(field).cloned().unwrap_or_default();
            (hit.id, marked)
        })
        .collect();
    hits.sort();
    Ok(hits)
}

/// A quoted member expands into each member of its group, not into a mix
/// of their words such as "statistical learning" or "new apple".
#[tokio::test(flavor = "multi_thread")]
async fn quoted_member_matches_each_member_and_no_mix_of_them() -> Result<()> {
    let engine = members_engine().await?;
    let cases: &[(&str, &[&str])] = &[
        ("body:\"ml\"", MLS3),
        ("body:\"machine learning\"", MLS3),
        ("body:\"statistical machine learning\"", MLS3),
        ("body:\"ml is fun\"", MLS3),
        ("body:\"new york\"", NYS),
        ("body:\"big apple\"", NYS),
    ];
    for (dsl, expected) in cases {
        assert_eq!(search_dsl(&engine, dsl).await?, sorted(expected), "{dsl}");
    }
    Ok(())
}

/// The index stores no `position_length`, so the members' words still
/// share positions there, as in Lucene: a document with one member
/// matches a phrase mixing two of them.
#[tokio::test(flavor = "multi_thread")]
async fn members_still_share_positions_in_the_index() -> Result<()> {
    let engine = members_engine().await?;
    let mut expected = MLS3.to_vec();
    expected.push("statistical_learning");
    assert_eq!(
        search_dsl(&engine, "body:\"statistical learning\"").await?,
        sorted(&expected)
    );
    Ok(())
}

/// The highlighter numbers the text as the index does, so the phrase it
/// marks is the one the index matched: the kept "learning" before "is",
/// not the synonym's "learning" that spans the whole match.
#[tokio::test(flavor = "multi_thread")]
async fn phrase_is_highlighted_where_the_index_matches_it() -> Result<()> {
    let engine = members_engine().await?;
    let request = SearchRequestBuilder::new()
        .query_dsl("body:\"learning is\"")
        .highlight(vec!["body".to_string()])
        .limit(10)
        .build();
    let results = engine.search(request).await?;
    let hit = results
        .iter()
        .find(|hit| hit.id == "statistical_machine_learning")
        .expect("statistical_machine_learning must match");
    assert_eq!(
        hit.highlights["body"],
        vec!["statistical machine <mark>learning is</mark> fun"]
    );
    Ok(())
}

// ---- Values with several paths (#1271) ----

/// Members followed by other words at gaps of zero to two positions.
const GAP_DOCS: [(&str, &str); 9] = [
    ("ml", "ml is fun"),
    ("ml_gap1", "ml really is fun"),
    ("ml_gap2", "ml really truly is fun"),
    ("machine_learning_gap1", "machine learning really is fun"),
    (
        "statistical_machine_learning",
        "statistical machine learning is fun",
    ),
    ("statistical_learning", "statistical learning is fun"),
    ("new_york_city", "new york city is big"),
    ("big_apple_gap1", "the big apple really is big"),
    ("apple_big", "apple big is new york"),
];

/// The documents each value with several paths matches at slop 0 to 2.
#[tokio::test(flavor = "multi_thread")]
async fn multi_path_values_match_the_same_documents() -> Result<()> {
    let engine = members_engine_with(&GAP_DOCS).await?;
    const ML_IS: [&[&str]; 3] = [
        &["ml", "statistical_machine_learning"],
        &[
            "machine_learning_gap1",
            "ml",
            "ml_gap1",
            "statistical_machine_learning",
        ],
        &[
            "machine_learning_gap1",
            "ml",
            "ml_gap1",
            "ml_gap2",
            "statistical_machine_learning",
        ],
    ];
    const NEW_YORK: [&[&str]; 3] = [
        &[],
        &["big_apple_gap1", "new_york_city"],
        &["big_apple_gap1", "new_york_city"],
    ];
    let cases: &[(&str, [&[&str]; 3])] = &[
        ("ml is", ML_IS),
        ("ml is fun", ML_IS),
        (
            "ml really is",
            [
                &["machine_learning_gap1", "ml_gap1"],
                &["machine_learning_gap1", "ml_gap1", "ml_gap2"],
                &["machine_learning_gap1", "ml_gap1", "ml_gap2"],
            ],
        ),
        ("new york is", NEW_YORK),
        ("big apple is big", NEW_YORK),
    ];
    for (value, by_slop) in cases {
        for (slop, expected) in by_slop.iter().enumerate() {
            let dsl = format!("body:\"{value}\"~{slop}");
            assert_eq!(search_dsl(&engine, &dsl).await?, sorted(expected), "{dsl}");
        }
    }
    Ok(())
}

/// Every path of the value that occurs in a document is marked, from its
/// first word to its last.
#[tokio::test(flavor = "multi_thread")]
async fn every_member_path_is_highlighted() -> Result<()> {
    let engine = members_engine_with(&GAP_DOCS).await?;
    let marked = |hits: &[(&str, &str)]| -> Vec<(String, Vec<String>)> {
        hits.iter()
            .map(|(id, text)| (id.to_string(), vec![text.to_string()]))
            .collect()
    };
    assert_eq!(
        highlights(&engine, "body:\"ml is\"~1", "body").await?,
        marked(&[
            (
                "machine_learning_gap1",
                "<mark>machine learning really is</mark> fun"
            ),
            ("ml", "<mark>ml is</mark> fun"),
            ("ml_gap1", "<mark>ml really is</mark> fun"),
            (
                "statistical_machine_learning",
                "<mark>statistical machine learning is</mark> fun"
            ),
        ])
    );
    assert_eq!(
        highlights(&engine, "body:\"new york is\"~1", "body").await?,
        marked(&[
            ("big_apple_gap1", "the <mark>big apple really is</mark> big"),
            ("new_york_city", "<mark>new york city is</mark> big"),
        ])
    );
    Ok(())
}
