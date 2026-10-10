//! Analyzed document structures for indexing.
//!
//! This module defines the data structures that represent documents after
//! analysis (tokenization and filtering), ready to be written to an index.
//!
//! # Overview
//!
//! The analysis pipeline transforms raw documents into analyzed documents:
//!
//! ```text
//! Document → Analyzer → AnalyzedDocument → Index
//! ```
//!
//! An [`AnalyzedDocument`] contains:
//! - Analyzed terms with positions for each field
//! - Stored field values (for retrieval)
//! - Field length statistics (for ranking)
//!
//! # Examples
//!
//! Creating an analyzed document (typically done by DocumentParser):
//!
//! ```
//! use laurus::lexical::core::analyzed::{AnalyzedDocument, AnalyzedTerm};
//! use laurus::lexical::core::field::FieldValue;
//! use ahash::AHashMap;
//!
//! let mut field_terms = AHashMap::new();
//! field_terms.insert(
//!     "content".to_string(),
//!     vec![
//!         AnalyzedTerm {
//!             term: "rust".to_string(),
//!             position: 0,
//!             frequency: 1,
//!             offset: (0, 4),
//!         },
//!         AnalyzedTerm {
//!             term: "programming".to_string(),
//!             position: 1,
//!             frequency: 1,
//!             offset: (5, 16),
//!         },
//!     ],
//! );
//!
//! let mut stored_fields = AHashMap::new();
//! stored_fields.insert("content".to_string(), FieldValue::Text("rust programming".to_string()));
//!
//! let mut field_lengths = AHashMap::new();
//! field_lengths.insert("content".to_string(), 2);
//!
//! let analyzed_doc = AnalyzedDocument {
//!     field_terms,
//!     stored_fields,
//!     field_lengths,
//!     point_values: AHashMap::new(),
//! };
//!
//! assert_eq!(analyzed_doc.field_lengths["content"], 2);
//! ```

use ahash::AHashMap;

use crate::lexical::core::field::FieldValue;

/// A document with analyzed terms ready for indexing.
///
/// This structure represents a document after analysis (tokenization),
/// ready to be written to the inverted index. The document ID is assigned
/// automatically by the index writer when the document is added.
///
/// # Fields
///
/// - `field_terms` - Map of field names to their analyzed terms
/// - `stored_fields` - Original field values to be stored (for retrieval)
/// - `field_lengths` - Number of *positions* per field (used for BM25 scoring;
///   see [`field_length_from_terms`])
/// - `point_values` - Numeric point values per field (for BKD tree range queries)
///
/// # Usage
///
/// Typically created by [`DocumentParser`](crate::lexical::core::parser::DocumentParser)
/// during the indexing process. Can also be constructed manually for
/// pre-analyzed documents from external systems.
#[derive(Debug, Clone)]
pub struct AnalyzedDocument {
    /// Field name to analyzed terms mapping.
    pub field_terms: AHashMap<String, Vec<AnalyzedTerm>>,
    /// Stored field values with original types preserved.
    pub stored_fields: AHashMap<String, FieldValue>,
    /// Field name to field length mapping. The length is the number of
    /// distinct positions, not of terms: a synonym stacked on another token
    /// shares its position and is not counted again (Issue #1257). Compute
    /// it with [`field_length_from_terms`] rather than `terms.len()`.
    pub field_lengths: AHashMap<String, u32>,
    /// Field name to numeric point values for the BKD tree.
    ///
    /// Each value in the map is a list of points for that field. A point is
    /// itself a `Vec<f64>` whose length is the BKD dimensionality (1 for
    /// integer/float/datetime, 2 for geo, etc.). A field can have multiple
    /// points per document — single-valued numeric fields contribute one
    /// point, multi-valued numeric fields contribute one point per element.
    /// Each `(point, doc_id)` pair is emitted as a distinct BKD entry, and
    /// the BKD reader deduplicates `doc_id`s during range search so a
    /// document is reported at most once per query.
    pub point_values: AHashMap<String, Vec<Vec<f64>>>,
}

/// What an index writer keeps per document of an [`AnalyzedDocument`] once
/// the document's postings are built: its stored fields and field lengths.
///
/// The postings are the only consumer of the analyzed terms, so keeping
/// them in the buffer as well would hold every term of every buffered
/// document twice until the flush (Issue #1168). The point values go to the
/// writer's per-field columns instead of a per-document map (Issue #1165).
/// Making the buffered type lack both fields means no flush-time code can
/// read them from here.
#[derive(Debug, Clone, Default)]
pub(crate) struct BufferedDocument {
    /// Stored field values with original types preserved.
    pub(crate) stored_fields: AHashMap<String, FieldValue>,
    /// Field name to field length (distinct positions), as
    /// [`AnalyzedDocument::field_lengths`].
    pub(crate) field_lengths: AHashMap<String, u32>,
}

/// An analyzed term with position and metadata.
#[derive(Debug, Clone)]
pub struct AnalyzedTerm {
    /// The term text.
    pub term: String,
    /// Position in the field.
    pub position: u32,
    /// Term frequency in the document.
    pub frequency: u32,
    /// Offset in the original text.
    pub offset: (usize, usize),
}

impl AnalyzedDocument {
    /// Create a new empty analyzed document.
    pub fn new() -> Self {
        Self {
            field_terms: AHashMap::new(),
            stored_fields: AHashMap::new(),
            field_lengths: AHashMap::new(),
            point_values: AHashMap::new(),
        }
    }

    /// Get the number of fields in this document.
    pub fn field_count(&self) -> usize {
        self.field_terms.len()
    }

    /// Get the total number of terms across all fields.
    pub fn total_terms(&self) -> usize {
        self.field_terms.values().map(|terms| terms.len()).sum()
    }

    /// Get the length (number of positions, see [`Self::field_lengths`]) for
    /// a specific field.
    pub fn field_length(&self, field: &str) -> Option<u32> {
        self.field_lengths.get(field).copied()
    }
}

impl Default for AnalyzedDocument {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyzedTerm {
    /// Create a new analyzed term.
    pub fn new(term: String, position: u32, frequency: u32, offset: (usize, usize)) -> Self {
        Self {
            term,
            position,
            frequency,
            offset,
        }
    }
}

/// The field length Lucene's `discountOverlaps` would report: the number of
/// distinct positions, not the number of terms. A synonym stacked on
/// another token (`position_increment = 0`) shares its anchor's position
/// and is not counted again (Issue #1257).
///
/// `terms` must be in non-decreasing position order — every
/// [`analyze_field_value`](crate::lexical::index::inverted::writer::analyze_field_value)
/// arm produces terms in that order, including the `TextArray` gap
/// numbering, so a single linear pass suffices. Terms rebuilt from a term
/// dictionary are in alphabetical order instead and must not be passed
/// here; the `debug_assert!` below turns any such call into a test-time
/// panic rather than a silently wrong length norm.
pub(crate) fn field_length_from_terms(terms: &[AnalyzedTerm]) -> u32 {
    debug_assert!(
        terms.windows(2).all(|w| w[0].position <= w[1].position),
        "field_length_from_terms requires non-decreasing positions"
    );
    let mut length = 0u32;
    let mut last_position: Option<u32> = None;
    for t in terms {
        if last_position != Some(t.position) {
            length += 1;
            last_position = Some(t.position);
        }
    }
    length
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzed_document_new() {
        let doc = AnalyzedDocument::new();
        assert_eq!(doc.field_count(), 0);
        assert_eq!(doc.total_terms(), 0);
    }

    #[test]
    fn test_analyzed_document_field_count() {
        let mut doc = AnalyzedDocument::new();
        doc.field_terms.insert("title".to_string(), vec![]);
        doc.field_terms.insert("content".to_string(), vec![]);
        assert_eq!(doc.field_count(), 2);
    }

    #[test]
    fn test_analyzed_document_total_terms() {
        let mut doc = AnalyzedDocument::new();
        doc.field_terms.insert(
            "title".to_string(),
            vec![
                AnalyzedTerm::new("hello".to_string(), 0, 1, (0, 5)),
                AnalyzedTerm::new("world".to_string(), 1, 1, (6, 11)),
            ],
        );
        doc.field_terms.insert(
            "content".to_string(),
            vec![AnalyzedTerm::new("test".to_string(), 0, 1, (0, 4))],
        );
        assert_eq!(doc.total_terms(), 3);
    }

    #[test]
    fn test_analyzed_term_new() {
        let term = AnalyzedTerm::new("search".to_string(), 5, 2, (10, 16));
        assert_eq!(term.term, "search");
        assert_eq!(term.position, 5);
        assert_eq!(term.frequency, 2);
        assert_eq!(term.offset, (10, 16));
    }

    #[test]
    fn field_length_from_terms_counts_positions_not_terms() {
        // "a big dog" with "large" stacked on "big": 4 terms, 3 positions.
        let terms = vec![
            AnalyzedTerm::new("a".to_string(), 0, 1, (0, 1)),
            AnalyzedTerm::new("big".to_string(), 1, 1, (2, 5)),
            AnalyzedTerm::new("large".to_string(), 1, 1, (2, 5)),
            AnalyzedTerm::new("dog".to_string(), 2, 1, (6, 9)),
        ];
        assert_eq!(field_length_from_terms(&terms), 3);
    }

    #[test]
    fn field_length_from_terms_counts_the_gap_between_text_array_elements() {
        // Two one-token elements at a position_increment_gap of 100: the
        // gap positions (1..=100) hold no term and are not counted, but
        // the two elements' own positions (0 and 101) are.
        let terms = vec![
            AnalyzedTerm::new("big".to_string(), 0, 1, (0, 3)),
            AnalyzedTerm::new("dog".to_string(), 101, 1, (0, 3)),
        ];
        assert_eq!(field_length_from_terms(&terms), 2);
    }

    #[test]
    fn field_length_from_terms_of_empty_slice_is_zero() {
        assert_eq!(field_length_from_terms(&[]), 0);
    }
}
