//! Integration tests for #1083: per-field `TextOption::term_vectors` wiring.
//!
//! `term_vectors` controls whether a `Text` field's postings carry
//! positions. Positions are read only by `PhraseQuery` of two or more terms
//! and span queries (`SpanTermQuery` and friends) — everything else (term
//! matching, one-term phrases, BM25 scoring, stored-field retrieval) is
//! unaffected either way.

use std::sync::Arc;

use laurus::Document;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::span::{SpanQuery, SpanTermQuery};
use laurus::lexical::{
    FieldOption, InvertedIndexConfig, PhraseQuery, Query, TermQuery, TextOption,
};
use laurus::storage::memory::MemoryStorage;

type TestIndex = Arc<dyn laurus::lexical::LexicalIndexReader>;

/// One field with `term_vectors: true` ("vec_field") and one with
/// `term_vectors: false` ("novec_field"), otherwise identically indexed
/// and stored. Doc 0 carries "the quick brown fox" in both fields; doc 1
/// carries "cat cat cat" in both, to probe repeated-term frequency.
fn create_test_index() -> Result<TestIndex, Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(
        laurus::storage::memory::MemoryStorageConfig::default(),
    ));

    let mut fields = std::collections::HashMap::new();
    fields.insert(
        "vec_field".to_string(),
        FieldOption::Text(TextOption {
            term_vectors: true,
            ..Default::default()
        }),
    );
    fields.insert(
        "novec_field".to_string(),
        FieldOption::Text(TextOption {
            term_vectors: false,
            ..Default::default()
        }),
    );

    let index = InvertedIndex::create(
        storage,
        InvertedIndexConfig {
            analyzer: Arc::new(StandardAnalyzer::new()?),
            fields,
            ..Default::default()
        },
    )?;

    let mut writer = index.writer()?;

    writer.add_document(
        Document::builder()
            .add_text("vec_field", "the quick brown fox")
            .add_text("novec_field", "the quick brown fox")
            .build(),
    )?;
    writer.add_document(
        Document::builder()
            .add_text("vec_field", "cat cat cat")
            .add_text("novec_field", "cat cat cat")
            .build(),
    )?;

    writer.commit()?;
    Ok(index.reader()?)
}

#[test]
fn phrase_query_only_matches_the_field_with_term_vectors_enabled()
-> Result<(), Box<dyn std::error::Error>> {
    let reader = create_test_index()?;

    let with_vectors =
        PhraseQuery::new("vec_field", vec!["quick".to_string(), "brown".to_string()]);
    let mut matcher = with_vectors.matcher(reader.as_ref())?;
    assert!(
        !matcher.is_exhausted(),
        "vec_field stores positions, the phrase must match"
    );
    assert_eq!(matcher.doc_id(), 0);
    assert!(!matcher.next()?);

    let without_vectors = PhraseQuery::new(
        "novec_field",
        vec!["quick".to_string(), "brown".to_string()],
    );
    let matcher = without_vectors.matcher(reader.as_ref())?;
    assert!(
        matcher.is_exhausted(),
        "novec_field has no positions, the phrase must not match"
    );

    Ok(())
}

/// #1247: a one-term phrase has no adjacency to check, so it must match
/// from the posting list whether or not the field stores positions, and
/// score the same in both fields.
#[test]
fn one_term_phrase_query_matches_with_or_without_term_vectors()
-> Result<(), Box<dyn std::error::Error>> {
    let reader = create_test_index()?;

    let mut scores = Vec::new();
    for field in ["vec_field", "novec_field"] {
        let query = PhraseQuery::new(field, vec!["cat".to_string()]);
        let mut matcher = query.matcher(reader.as_ref())?;
        assert!(!matcher.is_exhausted(), "doc 1 must match 'cat' in {field}");
        assert_eq!(matcher.doc_id(), 1);
        assert!(!matcher.next()?, "only doc 1 contains 'cat' in {field}");

        scores.push(query.scorer(reader.as_ref())?.score(1, 3.0, None));
    }
    assert!(scores[0] > 0.0, "the match must score, got {scores:?}");
    assert_eq!(
        scores[0], scores[1],
        "the phrase frequency must come from the term frequency in both fields"
    );

    Ok(())
}

#[test]
fn span_term_query_only_returns_spans_for_the_field_with_term_vectors_enabled()
-> Result<(), Box<dyn std::error::Error>> {
    let reader = create_test_index()?;

    let with_vectors = SpanTermQuery::new("vec_field", "quick");
    let spans = with_vectors.get_spans(0, reader.as_ref())?;
    assert_eq!(
        spans.len(),
        1,
        "vec_field stores positions, a span must be found"
    );

    let without_vectors = SpanTermQuery::new("novec_field", "quick");
    let spans = without_vectors.get_spans(0, reader.as_ref())?;
    assert!(
        spans.is_empty(),
        "novec_field has no positions, no span can be produced"
    );

    Ok(())
}

#[test]
fn term_vectors_false_does_not_affect_term_query_or_stored_retrieval()
-> Result<(), Box<dyn std::error::Error>> {
    let reader = create_test_index()?;

    // Plain term matching is unaffected by term_vectors.
    let query = TermQuery::new("novec_field", "brown");
    let mut matcher = query.matcher(reader.as_ref())?;
    assert!(!matcher.is_exhausted());
    assert_eq!(matcher.doc_id(), 0);
    assert!(!matcher.next()?);

    // Stored-field retrieval is unaffected by term_vectors.
    let doc = reader
        .document(0)?
        .expect("doc 0 must be retrievable regardless of term_vectors");
    let stored = doc
        .get_field("novec_field")
        .expect("novec_field must still be stored");
    assert_eq!(stored.as_text(), Some("the quick brown fox"));

    Ok(())
}

/// #1083: before the aggregation fix, `add_analyzed_document_to_index`
/// added one posting per occurrence, and `PostingList::add_posting`
/// summed each occurrence's already-cumulative `AnalyzedTerm::frequency`
/// on top of the rest — turning 3 occurrences of "cat" into a
/// triangular-number frequency of 6 instead of 3. Pinned in both the
/// positions-enabled and positions-disabled fields.
#[test]
fn repeated_term_frequency_matches_actual_occurrence_count()
-> Result<(), Box<dyn std::error::Error>> {
    let reader = create_test_index()?;

    for field in ["vec_field", "novec_field"] {
        let query = TermQuery::new(field, "cat");
        let matcher = query.matcher(reader.as_ref())?;
        assert!(!matcher.is_exhausted(), "doc 1 must match 'cat' in {field}");
        assert_eq!(matcher.doc_id(), 1);
        assert_eq!(
            matcher.term_freq(),
            3,
            "3 occurrences of 'cat' in {field} must report frequency 3, not accumulate triangularly"
        );
    }

    Ok(())
}
