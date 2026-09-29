use std::sync::Arc;

use laurus::Document;
use laurus::analysis::analyzer::standard::StandardAnalyzer;
use laurus::lexical::Query;
use laurus::lexical::span::{SpanQuery, SpanQueryBuilder, SpanQueryWrapper, SpanTermQuery};
use laurus::storage::memory::MemoryStorage;

type TestIndex = (
    Arc<laurus::storage::memory::MemoryStorage>,
    Arc<dyn laurus::lexical::LexicalIndexReader>,
);

fn create_test_index() -> Result<TestIndex, Box<dyn std::error::Error>> {
    let storage = Arc::new(MemoryStorage::new(
        laurus::storage::memory::MemoryStorageConfig::default(),
    ));
    let index = laurus::lexical::index::inverted::InvertedIndex::create(
        storage.clone(),
        laurus::lexical::InvertedIndexConfig {
            analyzer: Arc::new(StandardAnalyzer::new()?),
            ..Default::default()
        },
    )?;
    use laurus::lexical::index::LexicalIndex;
    let mut writer = index.writer()?;

    // Doc 0: "hello world"
    writer.add_document(
        Document::builder()
            .add_text("content", "hello world")
            .build(),
    )?;
    // Doc 1: "world hello"
    writer.add_document(
        Document::builder()
            .add_text("content", "world hello")
            .build(),
    )?;
    // Doc 2: "hello iris world"
    writer.add_document(
        Document::builder()
            .add_text("content", "hello iris world")
            .build(),
    )?;
    // Doc 3: "foo bar baz"
    writer.add_document(
        Document::builder()
            .add_text("content", "foo bar baz")
            .build(),
    )?;

    writer.commit()?;
    let reader = index.reader()?;

    Ok((storage, reader))
}

#[test]
fn test_span_term_query_integration() -> Result<(), Box<dyn std::error::Error>> {
    let (_, reader) = create_test_index()?;

    // Test SpanTermQuery
    let query = SpanTermQuery::new("content", "hello");

    // Doc 0: "hello" is at 0
    let spans0 = query.get_spans(0, reader.as_ref())?;
    assert_eq!(spans0.len(), 1);
    assert_eq!(spans0[0].start, 0);
    assert_eq!(spans0[0].end, 1);

    // Doc 1: "hello" is at 1
    let spans1 = query.get_spans(1, reader.as_ref())?;
    assert_eq!(spans1.len(), 1);
    assert_eq!(spans1[0].start, 1);
    assert_eq!(spans1[0].end, 2);

    Ok(())
}

#[test]
fn test_span_query_wrapper_integration() -> Result<(), Box<dyn std::error::Error>> {
    let (_, reader) = create_test_index()?;

    // Test SpanQueryWrapper with SpanTermQuery
    let span_query = Box::new(SpanTermQuery::new("content", "world"));
    let wrapper = SpanQueryWrapper::new(span_query);

    let mut matcher = wrapper.matcher(reader.as_ref())?;

    // Should match Doc 0 ("world" at 1)
    assert!(!matcher.is_exhausted());
    assert_eq!(matcher.doc_id(), 0);

    // Should match Doc 1 ("world" at 0)
    assert!(matcher.next()?);
    assert_eq!(matcher.doc_id(), 1);

    // Should match Doc 2 ("world" at 2)
    assert!(matcher.next()?);
    assert_eq!(matcher.doc_id(), 2);

    assert!(!matcher.next()?);

    Ok(())
}

#[test]
fn test_span_near_query_integration() -> Result<(), Box<dyn std::error::Error>> {
    let (_, reader) = create_test_index()?;
    let builder = SpanQueryBuilder::new("content");

    // Case 1: "hello world", in_order=true, slop=0
    // Doc 0: "hello"(0) "world"(1) -> dist=0. Match.
    // Doc 1: "world"(0) "hello"(1) -> disorder. No match.
    // Doc 2: "hello"(0) "iris"(1) "world"(2) -> dist=1. No match (slop=0).
    let q1 = builder.near(
        vec![
            Box::new(builder.term("hello")),
            Box::new(builder.term("world")),
        ],
        0,
        true,
    );
    let wrapper1 = SpanQueryWrapper::new(Box::new(q1));
    let mut matcher1 = wrapper1.matcher(reader.as_ref())?;
    assert!(!matcher1.is_exhausted());
    assert_eq!(matcher1.doc_id(), 0);
    assert!(!matcher1.next()?);

    // Case 2: "hello world", in_order=false, slop=0
    // Doc 0: Match.
    // Doc 1: "world"(0) "hello"(1). dist=0?
    // SpanNear logic for unordered:
    // It finds combinations. For Doc 1, "world"(0-1), "hello"(1-2).
    // combine_spans gets start=0, end=2. length=2. term_len=1+1=2. gaps=0.
    // So yes, it should match.
    let q2 = builder.near(
        vec![
            Box::new(builder.term("hello")),
            Box::new(builder.term("world")),
        ],
        0,
        false,
    );
    let wrapper2 = SpanQueryWrapper::new(Box::new(q2));
    let mut matcher2 = wrapper2.matcher(reader.as_ref())?;
    // Should match 0 and 1
    let mut docs = Vec::new();
    while !matcher2.is_exhausted() {
        docs.push(matcher2.doc_id());
        if !matcher2.next()? {
            break;
        }
    }
    docs.sort();
    assert_eq!(docs, vec![0, 1]);

    // Case 3: "hello world", in_order=true, slop=1
    // Doc 0: dist=0 <= 1. Match.
    // Doc 2: "hello"(0) ... "world"(2). dist=1 <= 1. Match.
    let q3 = builder.near(
        vec![
            Box::new(builder.term("hello")),
            Box::new(builder.term("world")),
        ],
        1,
        true,
    );
    let wrapper3 = SpanQueryWrapper::new(Box::new(q3));
    let mut matcher3 = wrapper3.matcher(reader.as_ref())?;
    let mut docs3 = Vec::new();
    while !matcher3.is_exhausted() {
        docs3.push(matcher3.doc_id());
        if !matcher3.next()? {
            break;
        }
    }
    docs3.sort();
    assert_eq!(docs3, vec![0, 2]);

    Ok(())
}

#[test]
fn test_span_containing_query_integration() -> Result<(), Box<dyn std::error::Error>> {
    let (_, reader) = create_test_index()?;
    let builder = SpanQueryBuilder::new("content");

    // Doc 2: "hello iris world"
    // We want to find a span that contains "iris".
    // Let's define a big span "hello ... world" (near, slop=1, in_order=true)
    // And a little span "iris".
    // In Doc 2:
    // "hello"(0) ... "world"(2) -> combined span (0, 3)
    // "iris"(1) -> span (1, 2)
    // (0,3) contains (1,2)?
    // 0 <= 1 && 2 <= 3. Yes.

    let big = builder.near(
        vec![
            Box::new(builder.term("hello")),
            Box::new(builder.term("world")),
        ],
        1,
        true,
    );
    let little = builder.term("iris");

    let q = builder.containing(Box::new(big), Box::new(little));
    let wrapper = SpanQueryWrapper::new(Box::new(q));
    let mut matcher = wrapper.matcher(reader.as_ref())?;

    assert!(!matcher.is_exhausted());
    assert_eq!(matcher.doc_id(), 2);
    assert!(!matcher.next()?);

    Ok(())
}

#[test]
fn test_span_within_query_integration() -> Result<(), Box<dyn std::error::Error>> {
    let (_, reader) = create_test_index()?;
    let builder = SpanQueryBuilder::new("content");

    // Doc 2: "hello iris world"
    // positions: hello=0, iris=1, world=2
    // find "iris" within 1 distance of "hello"
    // include="iris", exclude="hello", distance=1
    // iris(1,2), hello(0,1).
    // distance?
    // overlap? no.
    // hello end=1, iris start=1.
    // distance = 1 - 1 = 0.
    // So distance 0 implies immediately following?
    // Let's check Span::distance_to
    // if self.end <= other.start { other.start - self.end }
    // 1 <= 1 -> 1-1=0.
    // So distance 0 means adjacent.

    let include = builder.term("iris");
    let exclude = builder.term("hello");
    let q = builder.within(Box::new(include), Box::new(exclude), 0); // Adjacent

    let wrapper = SpanQueryWrapper::new(Box::new(q));
    let mut matcher = wrapper.matcher(reader.as_ref())?;

    assert!(!matcher.is_exhausted());
    assert_eq!(matcher.doc_id(), 2);
    assert!(!matcher.next()?);

    // Try distance 0 with "world"
    // iris(1,2), world(2,3).
    // exclude="world".
    // include="iris".
    // iris end=2, world start=2. dist=0.
    // match.
    let include2 = builder.term("iris");
    let exclude2 = builder.term("world");
    let q2 = builder.within(Box::new(include2), Box::new(exclude2), 0);
    let wrapper2 = SpanQueryWrapper::new(Box::new(q2));
    let matcher2 = wrapper2.matcher(reader.as_ref())?;
    assert!(!matcher2.is_exhausted());
    assert_eq!(matcher2.doc_id(), 2);

    Ok(())
}
