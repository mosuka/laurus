//! End-to-end facet tests for multi-valued (array) field values (Issue
//! #1187) on a real, on-disk index rather than a mock reader. They prove
//! that an array survives writer → DocValues → reader → `FacetCollector`
//! as one facet path per element, that the stored-document fallback (a
//! segment whose `.dv` file is missing — the #1047 situation) yields the
//! same counts, and that a scalar datetime gets a single label on both
//! paths even though the stored copy carries more precision than the
//! DocValues copy.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use laurus::Document;
use laurus::lexical::index::LexicalIndex;
use laurus::lexical::index::config::InvertedIndexConfig;
use laurus::lexical::index::inverted::InvertedIndex;
use laurus::lexical::reader::LexicalIndexReader;
use laurus::lexical::search::features::facet::{FacetCollector, FacetConfig, FacetResults};
use laurus::lexical::writer::LexicalIndexWriter;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};

/// Loose (non-compound) config so each segment's DocValues live in a
/// standalone `{segment}.dv` file that a test can delete directly; a high
/// `max_segments` keeps auto-merge from folding segments back together.
fn loose_config() -> InvertedIndexConfig {
    InvertedIndexConfig {
        use_compound: false,
        max_segments: 1000,
        ..Default::default()
    }
}

/// `.dv` files present in `storage`, sorted by name (= commit order).
fn dv_files_sorted(storage: &Arc<dyn Storage>) -> Vec<String> {
    let mut files: Vec<String> = storage
        .list_files()
        .unwrap()
        .into_iter()
        .filter(|f| f.ends_with(".dv"))
        .collect();
    files.sort();
    files
}

fn path(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| (*s).to_string()).collect()
}

fn tags_doc(tags: &[&str]) -> Document {
    Document::builder()
        .add_text_array("tags", tags.iter().map(|s| (*s).to_string()).collect())
        .build()
}

/// Sorted `(path, count)` pairs for `field`, so two collection runs can be
/// compared for exact equivalence.
fn flatten(results: &FacetResults, field: &str) -> Vec<(Vec<String>, u64)> {
    let mut out: Vec<(Vec<String>, u64)> = results
        .get_field_facets(field)
        .into_iter()
        .flatten()
        .map(|c| (c.path.path.clone(), c.count))
        .collect();
    out.sort();
    out
}

fn facet(reader: &dyn LexicalIndexReader, doc_ids: &[u64], field: &str) -> Vec<(Vec<String>, u64)> {
    let mut collector = FacetCollector::new(FacetConfig::default(), vec![field.to_string()]);
    for doc_id in doc_ids {
        collector.collect_doc(*doc_id, reader).unwrap();
    }
    flatten(&collector.finalize().unwrap(), field)
}

/// Segment 0 of every `tags` corpus below: a plain array, a duplicate
/// element, two hierarchical elements sharing an ancestor, and an empty
/// array. Expected once-per-document counts: `rust` 2, `search` 1, `a/b`
/// 1, `a/c` 1, ancestor `a` 1 (not 2), nothing from the empty array.
const SEGMENT_0: &[&[&str]] = &[&["rust", "search"], &["rust", "rust"], &["a/b", "a/c"], &[]];
/// Segment 1 adds one more `rust` and one more `a/b`.
const SEGMENT_1: &[&[&str]] = &[&["rust"], &["a/b"]];

fn expected_two_segment_tag_facets() -> Vec<(Vec<String>, u64)> {
    let mut expected = vec![
        (path(&["a"]), 2),
        (path(&["a", "b"]), 2),
        (path(&["a", "c"]), 1),
        (path(&["rust"]), 3),
        (path(&["search"]), 1),
    ];
    expected.sort();
    expected
}

/// Build the two-segment `tags` index, optionally deleting segment 0's
/// `.dv` file so its documents take the stored-document fallback while
/// `has_doc_values("tags")` stays `true` index-wide.
fn two_segment_tags_index(delete_first_dv: bool) -> (Arc<dyn LexicalIndexReader>, Vec<u64>) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = InvertedIndex::create(storage.clone(), loose_config()).unwrap();
    let mut writer: Box<dyn LexicalIndexWriter> = index.writer().unwrap();

    let mut doc_ids = Vec::new();
    for tags in SEGMENT_0 {
        doc_ids.push(writer.add_document(tags_doc(tags)).unwrap());
    }
    writer.commit().unwrap();
    for tags in SEGMENT_1 {
        doc_ids.push(writer.add_document(tags_doc(tags)).unwrap());
    }
    writer.commit().unwrap();

    if delete_first_dv {
        let dv_files = dv_files_sorted(&storage);
        assert_eq!(
            dv_files.len(),
            2,
            "expected one standalone .dv file per segment, found {dv_files:?}"
        );
        storage.delete_file(&dv_files[0]).unwrap();
    }

    let reader = writer.build_reader().unwrap();
    assert!(
        reader.has_doc_values("tags"),
        "a multi-valued text field is written to DocValues like any other stored value"
    );
    (reader, doc_ids)
}

#[test]
fn text_array_elements_are_counted_once_per_document_via_docvalues() {
    let (reader, doc_ids) = two_segment_tags_index(false);
    assert_eq!(
        facet(reader.as_ref(), &doc_ids, "tags"),
        expected_two_segment_tag_facets()
    );
}

#[test]
fn text_array_facets_match_when_a_segments_dv_file_is_missing() {
    // Segment 0's documents now miss on `get_doc_value` and fall back to
    // the stored document, which holds the same `TextArray`.
    let (reader, doc_ids) = two_segment_tags_index(true);
    let via_fallback = facet(reader.as_ref(), &doc_ids, "tags");

    let (dv_reader, dv_doc_ids) = two_segment_tags_index(false);
    assert_eq!(via_fallback, facet(dv_reader.as_ref(), &dv_doc_ids, "tags"));
    assert_eq!(via_fallback, expected_two_segment_tag_facets());
}

#[test]
fn scalar_datetime_yields_one_label_on_both_paths() {
    // 2023-11-14T22:13:20.123456789Z: DocValues archive it floored to
    // microseconds, the stored document keeps the nanoseconds. The facet
    // label must not depend on which copy the collector happened to read.
    let instant: DateTime<Utc> = DateTime::from_timestamp(1_700_000_000, 123_456_789).unwrap();
    let doc = || Document::builder().add_datetime("ts", instant).build();

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = InvertedIndex::create(storage.clone(), loose_config()).unwrap();
    let mut writer: Box<dyn LexicalIndexWriter> = index.writer().unwrap();
    let first = writer.add_document(doc()).unwrap();
    writer.commit().unwrap();
    let second = writer.add_document(doc()).unwrap();
    writer.commit().unwrap();

    let dv_files = dv_files_sorted(&storage);
    assert_eq!(dv_files.len(), 2, "found {dv_files:?}");
    storage.delete_file(&dv_files[0]).unwrap();

    let reader = writer.build_reader().unwrap();
    assert!(reader.has_doc_values("ts"));
    assert_eq!(
        facet(reader.as_ref(), &[first, second], "ts"),
        vec![(path(&["2023-11-14T22:13:20.123456+00:00"]), 2)]
    );
}

#[test]
fn datetime_array_elements_are_counted_as_rfc3339_labels() {
    let jan: DateTime<Utc> = "2024-01-01T00:00:00Z".parse().unwrap();
    let jun: DateTime<Utc> = "2024-06-15T12:00:00Z".parse().unwrap();

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let index = InvertedIndex::create(storage, loose_config()).unwrap();
    let mut writer: Box<dyn LexicalIndexWriter> = index.writer().unwrap();
    let doc_ids = vec![
        writer
            .add_document(
                Document::builder()
                    .add_datetime_array("seen_at", vec![jan, jun])
                    .build(),
            )
            .unwrap(),
        writer
            .add_document(
                Document::builder()
                    .add_datetime_array("seen_at", vec![jan])
                    .build(),
            )
            .unwrap(),
    ];
    writer.commit().unwrap();

    let reader = writer.build_reader().unwrap();
    assert_eq!(
        facet(reader.as_ref(), &doc_ids, "seen_at"),
        vec![
            (path(&["2024-01-01T00:00:00+00:00"]), 2),
            (path(&["2024-06-15T12:00:00+00:00"]), 1),
        ]
    );
}
