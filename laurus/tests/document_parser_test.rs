//! Integration tests for #1243: a document parsed by `DocumentParser` and
//! added with `add_analyzed_document` must be indexed exactly like the same
//! document passed to `add_document`.
//!
//! The parser used to keep its own copy of the writer's per-type analysis.
//! That copy collapsed a repeated term into one entry, so the stored tf was
//! 1 and every later position was lost. It also numbered positions from the
//! tokenizer instead of densely, counted distinct terms as the field length,
//! and wrote `DateTime` terms as RFC3339 instead of epoch seconds.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};

use laurus::Document;
use laurus::analysis::analyzer::analyzer::Analyzer;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::core::field::{DEFAULT_POSITION_INCREMENT_GAP, DateTimeOption};
use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::{
    DocumentParser, FieldOption, InvertedIndexConfig, LexicalIndexReader, PhraseQuery, Query,
    TextOption,
};
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// How the documents reach the writer.
#[derive(Clone, Copy)]
enum Ingest {
    /// `DocumentParser::parse`, then `add_analyzed_document`.
    Parser,
    /// `add_document`, which analyzes inside the writer.
    AddDocument,
}

fn fields() -> HashMap<String, FieldOption> {
    let mut fields = HashMap::new();
    fields.insert("body".to_string(), FieldOption::Text(TextOption::default()));
    fields.insert(
        "tags".to_string(),
        FieldOption::Text(TextOption {
            multi_valued: true,
            ..Default::default()
        }),
    );
    fields.insert(
        "when".to_string(),
        FieldOption::DateTime(DateTimeOption::default()),
    );
    fields
}

/// Indexes `docs` into a fresh in-memory index with the same schema and
/// analyzer for both ingest paths.
fn build_index(ingest: Ingest, docs: Vec<Document>) -> TestResult<Arc<dyn LexicalIndexReader>> {
    let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let analyzer: Arc<dyn Analyzer> = Arc::new(StandardAnalyzer::new()?);
    let index = InvertedIndex::create(
        storage,
        InvertedIndexConfig {
            analyzer: analyzer.clone(),
            fields: fields(),
            ..Default::default()
        },
    )?;
    let parser = DocumentParser::new(analyzer).with_fields(fields());

    let mut writer = index.writer()?;
    for doc in docs {
        match ingest {
            Ingest::Parser => {
                writer.add_analyzed_document(parser.parse(doc)?)?;
            }
            Ingest::AddDocument => {
                writer.add_document(doc)?;
            }
        }
    }
    writer.commit()?;
    Ok(writer.build_reader()?)
}

/// Every `(doc_id, term_freq, positions)` posting of `term` in `field`.
fn postings(
    reader: &dyn LexicalIndexReader,
    field: &str,
    term: &str,
) -> TestResult<Vec<(u64, u64, Vec<u64>)>> {
    let mut found = Vec::new();
    if let Some(mut it) = reader.postings(field, term)? {
        while it.next()? {
            found.push((it.doc_id(), it.term_freq(), it.positions()?));
        }
    }
    Ok(found)
}

/// The documents a slop-0 phrase query over `terms` matches.
fn phrase_matches(
    reader: &dyn LexicalIndexReader,
    field: &str,
    terms: &[&str],
) -> TestResult<Vec<u64>> {
    let query = PhraseQuery::new(field, terms.iter().map(|t| t.to_string()).collect());
    let mut matcher = query.matcher(reader)?;
    let mut docs = Vec::new();
    if !matcher.is_exhausted() {
        docs.push(matcher.doc_id());
        while matcher.next()? {
            docs.push(matcher.doc_id());
        }
    }
    Ok(docs)
}

fn sample_datetime() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2024, 5, 6, 7, 8, 9).unwrap()
}

#[test]
fn repeated_term_keeps_every_occurrence() -> TestResult {
    let reader = build_index(
        Ingest::Parser,
        vec![
            Document::builder()
                .add_text("body", "cat cat cat dog")
                .build(),
        ],
    )?;

    assert_eq!(
        postings(reader.as_ref(), "body", "cat")?,
        vec![(0, 3, vec![0, 1, 2])],
        "every occurrence of \"cat\" must be stored"
    );
    // Only the third "cat" is followed by "dog".
    assert_eq!(
        phrase_matches(reader.as_ref(), "body", &["cat", "dog"])?,
        vec![0],
        "the phrase must match through a later occurrence"
    );
    Ok(())
}

#[test]
fn repeated_term_in_text_array_keeps_every_occurrence() -> TestResult {
    let reader = build_index(
        Ingest::Parser,
        vec![
            Document::builder()
                .add_text_array("tags", vec!["cat".to_string(), "cat cat dog".to_string()])
                .build(),
        ],
    )?;

    // The second element starts one token plus the gap after the first.
    let base = 1 + u64::from(DEFAULT_POSITION_INCREMENT_GAP);
    assert_eq!(
        postings(reader.as_ref(), "tags", "cat")?,
        vec![(0, 3, vec![0, base, base + 1])],
        "every occurrence of \"cat\" across the elements must be stored"
    );
    // Only the second "cat" of the second element is followed by "dog".
    assert_eq!(
        phrase_matches(reader.as_ref(), "tags", &["cat", "dog"])?,
        vec![0],
        "the phrase must match through a later occurrence"
    );
    Ok(())
}

#[test]
fn parsed_document_indexes_like_add_document() -> TestResult {
    let doc = || {
        Document::builder()
            .add_text("body", "The cat sat. The cat, the dog and the cat ran!")
            .add_text_array(
                "tags",
                vec!["the cat".to_string(), "cat and dog".to_string()],
            )
            .add_datetime("when", sample_datetime())
            .build()
    };
    let parsed = build_index(Ingest::Parser, vec![doc()])?;
    let added = build_index(Ingest::AddDocument, vec![doc()])?;

    assert!(
        !postings(added.as_ref(), "body", "cat")?.is_empty(),
        "test precondition: \"cat\" must be indexed"
    );
    // Stop words ("the", "and") are compared too: absent from both.
    let checks: [(&str, &[&str]); 2] = [
        ("body", &["cat", "sat", "dog", "ran", "the", "and"]),
        ("tags", &["cat", "dog", "the", "and"]),
    ];
    for (field, terms) in checks {
        for term in terms {
            assert_eq!(
                postings(parsed.as_ref(), field, term)?,
                postings(added.as_ref(), field, term)?,
                "{field}:{term} must have the same tf and positions on both paths"
            );
        }

        let parsed_stats = parsed.field_stats(field)?.expect("parsed field stats");
        let added_stats = added.field_stats(field)?.expect("added field stats");
        assert_eq!(
            (parsed_stats.avg_length, parsed_stats.max_length),
            (added_stats.avg_length, added_stats.max_length),
            "{field} must have the same field length on both paths"
        );
    }

    let epoch = sample_datetime().timestamp().to_string();
    let added_when = postings(added.as_ref(), "when", &epoch)?;
    assert_eq!(
        added_when.len(),
        1,
        "test precondition: add_document indexes the epoch-seconds term"
    );
    assert_eq!(postings(parsed.as_ref(), "when", &epoch)?, added_when);
    assert!(
        postings(parsed.as_ref(), "when", &sample_datetime().to_rfc3339())?.is_empty(),
        "the parser must not index an RFC3339 term that add_document does not"
    );
    Ok(())
}
