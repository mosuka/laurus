//! End-to-end tests of result grouping (`SearchRequest::group_by`, Issue
//! #1297).
//!
//! Most tests compare a grouped search with an ungrouped one folded by the
//! test itself: on a corpus smaller than the candidate pool both first
//! stages return every match, so the fold of the ungrouped ranking is what
//! the grouped search must return. Vectors are Flat, so rankings are exact.

use std::sync::Arc;

use laurus::lexical::TermQuery;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::Vector;
use laurus::{
    DataValue, Document, Engine, FieldOption, FlatOption, GroupBy, IntegerOption, LaurusError,
    LexicalSearchQuery, MultiVectorOption, QueryVector, RescoreOptions, Result, Schema,
    SearchRequestBuilder, SearchResult, SortField, SortOrder, TextOption, VectorSearchQuery,
};

const TOKENS: &str = "tokens";

fn schema() -> Schema {
    Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("parent", TextOption::default())
        .add_integer_field("year", IntegerOption::default())
        .add_text_field(
            "tags",
            TextOption {
                multi_valued: true,
                ..Default::default()
            },
        )
        .add_integer_field(
            "no_dv",
            IntegerOption {
                doc_values: false,
                ..Default::default()
            },
        )
        .add_flat_field("vec", FlatOption::new(2))
        .add_field(TOKENS, FieldOption::MultiVector(MultiVectorOption::new(2)))
        .build()
}

async fn empty_engine() -> Result<Engine> {
    Engine::new(
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
        schema(),
    )
    .await
}

fn chunk(title: &str, vec: [f32; 2]) -> Document {
    Document::builder()
        .add_text("title", title)
        .add_vector("vec", vec.to_vec())
        .build()
}

/// Three documents split into chunks with `add_document`, so chunks share
/// their document's `_id`, plus one unrelated document. Every chunk has a
/// distinct BM25 score for `title:rust` and a distinct similarity to
/// `[1, 0]`.
async fn chunked_engine() -> Result<Engine> {
    let engine = empty_engine().await?;
    let chunks = [
        ("a", "rust rust rust guide", [1.0, 0.0]),
        ("a", "rust rust intro", [0.9, 0.1]),
        ("a", "rust appendix notes", [0.3, 0.7]),
        ("b", "rust rust book chapter", [0.8, 0.2]),
        ("b", "rust history of programming", [0.2, 0.8]),
        ("c", "rust", [0.6, 0.4]),
        ("d", "python only", [0.0, 1.0]),
    ];
    for (id, title, vec) in chunks {
        engine.add_document(id, chunk(title, vec)).await?;
    }
    engine.commit().await?;
    Ok(engine)
}

fn lexical_query() -> LexicalSearchQuery {
    LexicalSearchQuery::from("title:rust")
}

fn vector_query(target: [f32; 2]) -> VectorSearchQuery {
    VectorSearchQuery::Vectors(vec![QueryVector {
        vector: Vector::new(target.to_vec()),
        weight: 1.0,
        fields: Some(vec!["vec".to_string()]),
    }])
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Lexical,
    Vector,
    Hybrid,
}

fn request(mode: Mode) -> SearchRequestBuilder {
    let builder = SearchRequestBuilder::new();
    match mode {
        Mode::Lexical => builder.lexical_query(lexical_query()),
        Mode::Vector => builder.vector_query(vector_query([1.0, 0.0])),
        Mode::Hybrid => builder
            .lexical_query(lexical_query())
            .vector_query(vector_query([1.0, 0.0])),
    }
}

fn title(result: &SearchResult) -> String {
    match result
        .document
        .as_ref()
        .and_then(|doc| doc.fields.get("title"))
    {
        Some(DataValue::Text(title)) => title.clone(),
        other => panic!("result {} has no title: {other:?}", result.id),
    }
}

/// Each group as the titles of its hits, leader first.
fn grouped_titles(results: &[SearchResult]) -> Vec<Vec<String>> {
    results
        .iter()
        .map(|leader| {
            assert!(
                leader
                    .inner_hits
                    .iter()
                    .all(|hit| hit.inner_hits.is_empty()),
                "inner hits have no inner hits of their own"
            );
            assert!(
                leader
                    .inner_hits
                    .iter()
                    .all(|hit| hit.group_key == leader.group_key),
                "inner hits carry their leader's group_key"
            );
            std::iter::once(leader)
                .chain(&leader.inner_hits)
                .map(title)
                .collect()
        })
        .collect()
}

/// Fold an ungrouped ranking the way grouping must: groups ordered by
/// their best hit, up to `group_size` hits each in rank order, a hit with
/// no key alone in its group, at most `needed` groups.
fn fold(
    ranking: &[SearchResult],
    key: impl Fn(&SearchResult) -> Option<String>,
    group_size: usize,
    needed: usize,
) -> Vec<Vec<String>> {
    let mut groups: Vec<(Option<String>, Vec<String>)> = Vec::new();
    for result in ranking {
        let key = key(result);
        if let Some(group) = groups
            .iter_mut()
            .find(|(group_key, _)| key.is_some() && *group_key == key)
        {
            if group.1.len() < group_size {
                group.1.push(title(result));
            }
        } else if groups.len() < needed {
            groups.push((key, vec![title(result)]));
        }
    }
    groups.into_iter().map(|(_, titles)| titles).collect()
}

fn by_id(result: &SearchResult) -> Option<String> {
    Some(result.id.clone())
}

fn by_field(field: &'static str) -> impl Fn(&SearchResult) -> Option<String> {
    move |result| {
        result
            .document
            .as_ref()
            .and_then(|doc| doc.fields.get(field))
            .map(|value| format!("{value:?}"))
    }
}

async fn ranking(engine: &Engine, builder: SearchRequestBuilder) -> Result<Vec<SearchResult>> {
    engine.search(builder.limit(100).build()).await
}

#[tokio::test]
async fn chunks_collapse_by_id_in_every_search_type() -> Result<()> {
    let engine = chunked_engine().await?;
    for mode in [Mode::Lexical, Mode::Vector, Mode::Hybrid] {
        let ungrouped = ranking(&engine, request(mode)).await?;
        for group_size in [1, 2, 3] {
            let grouped = engine
                .search(
                    request(mode)
                        .group_by(GroupBy::new("_id").group_size(group_size))
                        .limit(10)
                        .build(),
                )
                .await?;
            assert_eq!(
                grouped_titles(&grouped),
                fold(&ungrouped, by_id, group_size, 10),
                "{mode:?}, group_size {group_size}"
            );
            let ids: Vec<&str> = grouped.iter().map(|result| result.id.as_str()).collect();
            let mut distinct = ids.clone();
            distinct.dedup();
            assert_eq!(ids, distinct, "{mode:?}: one result per document");
            for result in &grouped {
                assert_eq!(
                    result.group_key,
                    Some(DataValue::Text(result.id.clone())),
                    "{mode:?}"
                );
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn offset_and_limit_count_groups() -> Result<()> {
    let engine = chunked_engine().await?;
    let ungrouped = ranking(&engine, request(Mode::Vector)).await?;
    let all = fold(&ungrouped, by_id, 2, 10);
    assert_eq!(all.len(), 4, "a, b, c and d");

    let page = engine
        .search(
            request(Mode::Vector)
                .group_by(GroupBy::new("_id").group_size(2))
                .offset(1)
                .limit(2)
                .build(),
        )
        .await?;
    assert_eq!(grouped_titles(&page), all[1..3].to_vec());
    Ok(())
}

/// `(id, title, parent, year, vec)` of a document with one value per
/// grouping field, or none.
type Record = (
    &'static str,
    &'static str,
    Option<&'static str>,
    Option<i64>,
    [f32; 2],
);

async fn record_engine(records: &[Record]) -> Result<Engine> {
    let engine = empty_engine().await?;
    for &(id, title, parent, year, vec) in records {
        let mut builder = Document::builder()
            .add_text("title", title)
            .add_vector("vec", vec.to_vec());
        if let Some(parent) = parent {
            builder = builder.add_text("parent", parent);
        }
        if let Some(year) = year {
            builder = builder.add_integer("year", year);
        }
        engine.put_document(id, builder.build()).await?;
    }
    engine.commit().await?;
    Ok(engine)
}

fn records() -> Vec<Record> {
    vec![
        ("r1", "rust ownership", Some("book"), Some(2021), [1.0, 0.0]),
        ("r2", "rust traits", Some("book"), Some(2023), [0.95, 0.05]),
        ("r3", "rust async", Some("blog"), Some(2021), [0.9, 0.1]),
        ("r4", "rust macros", None, Some(2022), [0.85, 0.15]),
        ("r5", "rust lifetimes", Some("book"), None, [0.8, 0.2]),
        ("r6", "rust unsafe", Some("blog"), Some(2023), [0.75, 0.25]),
        ("r7", "rust testing", None, None, [0.7, 0.3]),
    ]
}

#[tokio::test]
async fn groups_by_a_text_and_an_integer_field() -> Result<()> {
    let engine = record_engine(&records()).await?;
    let ungrouped = ranking(&engine, request(Mode::Vector)).await?;
    for (field, group_size) in [("parent", 1), ("parent", 2), ("year", 1), ("year", 3)] {
        let grouped = engine
            .search(
                request(Mode::Vector)
                    .group_by(GroupBy::new(field).group_size(group_size))
                    .limit(10)
                    .build(),
            )
            .await?;
        assert_eq!(
            grouped_titles(&grouped),
            fold(&ungrouped, by_field(field), group_size, 10),
            "{field}, group_size {group_size}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn a_hit_without_a_value_forms_its_own_group() -> Result<()> {
    let engine = record_engine(&records()).await?;
    let grouped = engine
        .search(
            request(Mode::Vector)
                .group_by(GroupBy::new("parent").group_size(5))
                .limit(10)
                .build(),
        )
        .await?;
    let groups: Vec<(Option<DataValue>, Vec<String>)> = grouped
        .iter()
        .map(|leader| {
            (
                leader.group_key.clone(),
                std::iter::once(leader)
                    .chain(&leader.inner_hits)
                    .map(|hit| hit.id.clone())
                    .collect(),
            )
        })
        .collect();
    let text = |value: &str| Some(DataValue::Text(value.to_string()));
    assert_eq!(
        groups,
        vec![
            (text("book"), vec!["r1".into(), "r2".into(), "r5".into()]),
            (text("blog"), vec!["r3".into(), "r6".into()]),
            (None, vec!["r4".into()]),
            (None, vec!["r7".into()]),
        ]
    );
    Ok(())
}

#[tokio::test]
async fn grouping_follows_a_field_sort() -> Result<()> {
    let engine = record_engine(&records()).await?;
    let sorted = || {
        SearchRequestBuilder::new()
            .lexical_query(lexical_query())
            .sort_by(SortField::Field {
                name: "year".to_string(),
                order: SortOrder::Desc,
            })
    };
    let ungrouped = ranking(&engine, sorted()).await?;
    let grouped = engine
        .search(
            sorted()
                .group_by(GroupBy::new("parent").group_size(2))
                .limit(10)
                .build(),
        )
        .await?;
    assert_eq!(
        grouped_titles(&grouped),
        fold(&ungrouped, by_field("parent"), 2, 10)
    );
    Ok(())
}

/// One document's chunks fill the top of the ranking: the first pool holds
/// one group, so the search widens it until other documents appear.
#[tokio::test]
async fn widens_the_pool_when_one_document_fills_the_top() -> Result<()> {
    let engine = empty_engine().await?;
    for i in 0..40 {
        let tilt = i as f32 * 0.001;
        engine
            .add_document("big", chunk(&format!("big chunk {i}"), [1.0, tilt]))
            .await?;
    }
    for (id, vec) in [("x", [0.6, 0.4]), ("y", [0.5, 0.5]), ("z", [0.4, 0.6])] {
        engine.add_document(id, chunk(id, vec)).await?;
    }
    engine.commit().await?;

    // limit 3, group_size 1: the first pool is 6 candidates, all "big".
    let grouped = engine
        .search(
            request(Mode::Vector)
                .group_by(GroupBy::new("_id"))
                .limit(3)
                .build(),
        )
        .await?;
    let ids: Vec<&str> = grouped.iter().map(|result| result.id.as_str()).collect();
    assert_eq!(ids, ["big", "x", "y"]);
    assert_eq!(title(&grouped[0]), "big chunk 0");
    Ok(())
}

#[tokio::test]
async fn stops_when_the_first_stage_runs_out() -> Result<()> {
    let engine = chunked_engine().await?;
    for mode in [Mode::Lexical, Mode::Vector, Mode::Hybrid] {
        let grouped = engine
            .search(
                request(mode)
                    .group_by(GroupBy::new("_id").group_size(10))
                    .limit(50)
                    .build(),
            )
            .await?;
        let expected = match mode {
            // "d" does not mention rust.
            Mode::Lexical => 3,
            Mode::Vector | Mode::Hybrid => 4,
        };
        assert_eq!(grouped.len(), expected, "{mode:?}");
        let hits: usize = grouped.iter().map(|g| 1 + g.inner_hits.len()).sum();
        assert_eq!(hits, if expected == 3 { 6 } else { 7 }, "{mode:?}");
    }
    Ok(())
}

/// `(id, title, vec, tokens)` of one chunk with token vectors.
type TokenChunk = (&'static str, &'static str, [f32; 2], Vec<[f32; 2]>);

/// The late-interaction rescore reorders chunks before grouping, so a
/// group's leader is its best chunk by the rescored order.
#[tokio::test]
async fn the_rescored_order_chooses_each_leader() -> Result<()> {
    let engine = empty_engine().await?;
    let chunks: [TokenChunk; 3] = [
        // First by the vector, last by late interaction.
        ("a", "a first stage", [1.0, 0.0], vec![[0.1, 0.0]]),
        ("a", "a rescored", [0.7, 0.3], vec![[0.9, 0.9]]),
        ("b", "b", [0.9, 0.1], vec![[0.5, 0.5]]),
    ];
    for (id, title, vec, tokens) in chunks {
        let doc = Document::builder()
            .add_text("title", title)
            .add_vector("vec", vec.to_vec())
            .add_vector_array(TOKENS, tokens.iter().map(|t| t.to_vec()).collect())
            .build();
        engine.add_document(id, doc).await?;
    }
    engine.commit().await?;

    let query_tokens = vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])];
    let search = |rescore: bool| {
        let builder = request(Mode::Vector)
            .group_by(GroupBy::new("_id"))
            .limit(10);
        let builder = if rescore {
            builder.rescore(RescoreOptions::late_interaction(
                TOKENS,
                query_tokens.clone(),
            ))
        } else {
            builder
        };
        engine.search(builder.build())
    };

    let plain = search(false).await?;
    assert_eq!(
        plain.iter().map(title).collect::<Vec<_>>(),
        ["a first stage", "b"]
    );
    let rescored = search(true).await?;
    assert_eq!(
        rescored.iter().map(title).collect::<Vec<_>>(),
        ["a rescored", "b"]
    );
    Ok(())
}

#[tokio::test]
async fn a_replaced_document_groups_by_its_new_value() -> Result<()> {
    let engine = record_engine(&[
        ("p1", "rust one", Some("old"), None, [1.0, 0.0]),
        ("p2", "rust two", Some("old"), None, [0.9, 0.1]),
    ])
    .await?;
    let replacement = Document::builder()
        .add_text("title", "rust one again")
        .add_text("parent", "new")
        .add_vector("vec", vec![1.0, 0.0])
        .build();
    engine.put_document("p1", replacement).await?;
    engine.commit().await?;

    let grouped = engine
        .search(
            request(Mode::Vector)
                .group_by(GroupBy::new("parent").group_size(5))
                .limit(10)
                .build(),
        )
        .await?;
    let groups: Vec<(Option<DataValue>, Vec<String>)> = grouped
        .iter()
        .map(|leader| {
            (
                leader.group_key.clone(),
                std::iter::once(leader)
                    .chain(&leader.inner_hits)
                    .map(title)
                    .collect(),
            )
        })
        .collect();
    assert_eq!(
        groups,
        vec![
            (
                Some(DataValue::Text("new".into())),
                vec!["rust one again".into()]
            ),
            (Some(DataValue::Text("old".into())), vec!["rust two".into()]),
        ]
    );
    Ok(())
}

#[tokio::test]
async fn inner_hits_are_highlighted() -> Result<()> {
    let engine = chunked_engine().await?;
    let grouped = engine
        .search(
            request(Mode::Lexical)
                .group_by(GroupBy::new("_id").group_size(3))
                .highlight(vec!["title".to_string()])
                .limit(10)
                .build(),
        )
        .await?;
    let a = grouped
        .iter()
        .find(|result| result.id == "a")
        .expect("document a matches");
    assert_eq!(a.inner_hits.len(), 2);
    for hit in std::iter::once(a).chain(&a.inner_hits) {
        let fragments = hit.highlights.get("title").expect("title is highlighted");
        assert!(
            fragments.iter().any(|f| f.contains("<mark>rust</mark>")),
            "{fragments:?}"
        );
    }
    Ok(())
}

fn assert_invalid(result: Result<Vec<SearchResult>>, fragment: &str) {
    match result {
        Err(LaurusError::InvalidArgument(message)) => {
            assert!(message.contains(fragment), "{message}");
        }
        other => panic!("expected InvalidArgument containing {fragment:?}, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_requests_that_cannot_group() -> Result<()> {
    let engine = chunked_engine().await?;
    let search = |group_by: GroupBy, offset: usize, limit: usize| {
        engine.search(
            request(Mode::Lexical)
                .group_by(group_by)
                .offset(offset)
                .limit(limit)
                .build(),
        )
    };
    assert_invalid(
        search(GroupBy::new("_id").group_size(0), 0, 10).await,
        "group_size",
    );
    assert_invalid(
        search(
            GroupBy::new("_id").group_size(GroupBy::MAX_GROUP_SIZE + 1),
            0,
            10,
        )
        .await,
        "group_size",
    );
    assert_invalid(
        search(GroupBy::new("_id"), GroupBy::MAX_CANDIDATES, 1).await,
        "offset + limit",
    );
    assert_invalid(
        search(GroupBy::new("missing"), 0, 10).await,
        "unknown field",
    );
    assert_invalid(search(GroupBy::new("vec"), 0, 10).await, "cannot group");
    assert_invalid(search(GroupBy::new(TOKENS), 0, 10).await, "cannot group");
    assert_invalid(search(GroupBy::new("tags"), 0, 10).await, "multi_valued");
    assert_invalid(search(GroupBy::new("no_dv"), 0, 10).await, "doc_values");

    assert!(search(GroupBy::new("_id"), 0, 0).await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn an_ungrouped_search_returns_no_group_fields() -> Result<()> {
    let engine = chunked_engine().await?;
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_query(LexicalSearchQuery::Obj(Box::new(TermQuery::new(
                    "title", "rust",
                ))))
                .limit(10)
                .build(),
        )
        .await?;
    assert_eq!(results.len(), 6, "every rust chunk, ungrouped");
    assert!(
        results
            .iter()
            .all(|result| result.group_key.is_none() && result.inner_hits.is_empty())
    );
    Ok(())
}
