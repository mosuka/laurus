//! Document parser for converting documents into analyzed documents.
//!
//! This module provides [`DocumentParser`] that works similarly to QueryParser,
//! analyzing document fields and producing tokenized, index-ready documents.
//!
//! # Overview
//!
//! The `DocumentParser` bridges the gap between raw documents and the inverted
//! index by:
//!
//! 1. Analyzing text fields with configured analyzers (per-field or default)
//! 2. Converting non-text fields (numbers, dates, geo) to indexable terms
//! 3. Calculating term frequencies and positions
//! 4. Preserving both indexed and stored field values
//!
//! Every field is analyzed by the same function the writer uses for
//! [`InvertedIndexWriter::add_document`](crate::lexical::index::inverted::writer::InvertedIndexWriter::add_document),
//! so each occurrence of a term becomes its own entry, positions are
//! numbered as the index stores them (stacked synonyms share one), and the
//! field length counts positions, not tokens, so a stacked synonym does not
//! add to it (Issue #1243, Issue #1257).
//!
//! # Architecture
//!
//! ```text
//! Document → DocumentParser → AnalyzedDocument → Index
//!              ↓
//!        PerFieldAnalyzer
//!              ↓
//!        Tokenizer + Filters
//! ```
//!
//! # Field Type Handling
//!
//! - **Text fields**: Analyzed with tokenizers and filters
//! - **Integer/Float**: Converted to string representation for indexing
//! - **Boolean**: Converted to "true"/"false" strings
//! - **DateTime**: Converted to its whole-second Unix timestamp, plus a BKD point
//! - **Geo**: Converted to "lat,lon" format
//! - **Binary**: Stored only, not indexed
//! - **Null**: Stored only, not indexed
//!
//! # Schema Awareness (Issue #1114)
//!
//! By default a [`DocumentParser`] is schema-less: every field above is
//! both indexed and stored, regardless of type. Attach a schema via
//! [`DocumentParser::with_fields`] to honor each field's `indexed`/
//! `stored` settings instead -- the same gate
//! [`InvertedIndexWriter::add_document`](crate::lexical::index::inverted::writer::InvertedIndexWriter::add_document)
//! applies, so a document parsed here and fed to
//! [`InvertedIndexWriter::add_analyzed_document`](crate::lexical::index::inverted::writer::InvertedIndexWriter::add_analyzed_document)
//! lands in the index identically to one passed to `add_document`
//! directly.
//!
//! # Examples
//!
//! Basic usage with default analyzer:
//!
//! ```
//! use laurus::lexical::core::document::Document;
//! use laurus::lexical::core::parser::DocumentParser;
//! use laurus::lexical::core::field::{TextOption, IntegerOption};
//! use laurus::analysis::analyzer::standard::StandardAnalyzer;
//! use std::sync::Arc;
//!
//! let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()));
//!
//! let doc = Document::builder()
//!     .add_text("title", "Rust Programming Language")
//!     .add_integer("year", 2024)
//!     .build();
//!
//! let analyzed = parser.parse(doc).unwrap();
//! assert!(analyzed.field_terms.contains_key("title"));
//! assert!(analyzed.field_terms.contains_key("year"));
//! ```
//!
//! With per-field analyzers:
//!
//! ```
//! use laurus::lexical::core::document::Document;
//! use laurus::lexical::core::parser::DocumentParser;
//! use laurus::lexical::core::field::TextOption;
//! use laurus::analysis::analyzer::per_field::PerFieldAnalyzer;
//! use laurus::analysis::analyzer::standard::StandardAnalyzer;
//! use laurus::analysis::analyzer::keyword::KeywordAnalyzer;
//! use std::sync::Arc;
//!
//! // Configure per-field analyzers
//! let per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
//! per_field.add_analyzer("id", Arc::new(KeywordAnalyzer::new()));
//!
//! let parser = DocumentParser::new(Arc::new(per_field));
//!
//! let doc = Document::builder()
//!     .add_text("title", "Getting Started")  // Uses StandardAnalyzer
//!     .add_text("id", "DOC-001")             // Uses KeywordAnalyzer
//!     .build();
//!
//! let analyzed = parser.parse(doc).unwrap();
//! // "id" field is treated as a single keyword token
//! assert_eq!(analyzed.field_terms.get("id").unwrap()[0].term, "DOC-001");
//! ```
//!
//! With schema field options (Issue #1114) -- `indexed: false` keeps a
//! field out of the index while it's still retrievable:
//!
//! ```
//! use laurus::lexical::core::document::Document;
//! use laurus::lexical::core::parser::DocumentParser;
//! use laurus::lexical::{FieldOption, TextOption};
//! use laurus::analysis::analyzer::standard::StandardAnalyzer;
//! use std::collections::HashMap;
//! use std::sync::Arc;
//!
//! let mut fields = HashMap::new();
//! fields.insert("title".to_string(), FieldOption::Text(TextOption::default()));
//! fields.insert(
//!     "internal_note".to_string(),
//!     FieldOption::Text(TextOption::default().indexed(false)),
//! );
//!
//! let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()))
//!     .with_fields(fields);
//!
//! let doc = Document::builder()
//!     .add_text("title", "Rust Programming")
//!     .add_text("internal_note", "not searchable, but still retrievable")
//!     .build();
//!
//! let analyzed = parser.parse(doc).unwrap();
//! assert!(analyzed.field_terms.contains_key("title"));
//! assert!(!analyzed.field_terms.contains_key("internal_note"));
//! assert!(analyzed.stored_fields.contains_key("internal_note"));
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use ahash::AHashMap;

use crate::analysis::analyzer::analyzer::Analyzer;
use crate::error::Result;
use crate::lexical::core::analyzed::{AnalyzedDocument, field_length_from_terms};
use crate::lexical::core::document::Document;
use crate::lexical::core::field::{FieldOption, FieldValue};
use crate::lexical::index::inverted::writer::analyze_field_value;

/// A document parser that converts Documents into AnalyzedDocuments.
///
/// Similar to how QueryParser analyzes query strings, DocumentParser
/// analyzes Document fields using a PerFieldAnalyzer to produce
/// tokenized, indexed-ready AnalyzedDocuments.
///
/// Schema-less by default (every field is indexed and stored); chain
/// [`Self::with_fields`] to gate fields by a schema's `indexed`/`stored`
/// settings instead (Issue #1114).
///
/// # Example
///
/// ```
/// use laurus::lexical::core::document::Document;
/// use laurus::lexical::core::parser::DocumentParser;
/// use laurus::lexical::core::field::TextOption;
/// use laurus::analysis::analyzer::per_field::PerFieldAnalyzer;
/// use laurus::analysis::analyzer::standard::StandardAnalyzer;
/// use laurus::analysis::analyzer::keyword::KeywordAnalyzer;
/// use std::sync::Arc;
///
/// let per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
/// per_field.add_analyzer("id", Arc::new(KeywordAnalyzer::new()));
///
/// let parser = DocumentParser::new(Arc::new(per_field));
///
/// let doc = Document::builder()
///     .add_text("title", "Rust Programming")
///     .add_text("id", "BOOK-001")
///     .build();
///
/// let analyzed = parser.parse(doc).unwrap();
/// ```
pub struct DocumentParser {
    /// Analyzer (typically PerFieldAnalyzerWrapper) for analyzing fields.
    analyzer: Arc<dyn Analyzer>,
    /// Per-field schema options, keyed by field name -- the same map as
    /// `InvertedIndexWriterConfig::fields`, so this parser's output
    /// matches what `InvertedIndexWriter::add_document` would produce
    /// for the same document and schema (Issue #1114).
    ///
    /// Empty (the default from [`Self::new`]) means schema-less: every
    /// field is indexed and stored, which is what this parser did
    /// unconditionally before #1114. Attach a schema via
    /// [`Self::with_fields`] to gate fields by their `indexed`/`stored`
    /// settings instead.
    fields: HashMap<String, FieldOption>,
}

impl std::fmt::Debug for DocumentParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocumentParser")
            .field("analyzer", &self.analyzer.name())
            .field("fields", &self.fields.len())
            .finish()
    }
}

impl DocumentParser {
    /// Create a new document parser with the given analyzer.
    ///
    /// Typically, you would pass a PerFieldAnalyzer here,
    /// similar to how it's used with QueryParser.
    ///
    /// The parser starts in schema-less mode: every field is indexed and
    /// stored regardless of type, exactly as it always has been. Chain
    /// [`Self::with_fields`] to gate fields by a schema's `indexed`/
    /// `stored` settings instead (Issue #1114).
    pub fn new(analyzer: Arc<dyn Analyzer>) -> Self {
        DocumentParser {
            analyzer,
            fields: HashMap::new(),
        }
    }

    /// Attach per-field schema options, switching this parser out of
    /// schema-less mode (Issue #1114).
    ///
    /// Pass the same map the writer holds
    /// (`InvertedIndexWriterConfig::fields`) so that a document parsed
    /// here and fed to `InvertedIndexWriter::add_analyzed_document` lands
    /// in the index identically to one passed to
    /// `InvertedIndexWriter::add_document` directly. A field absent from
    /// a non-empty map is dropped entirely, unless its name starts with
    /// `_` (reserved/internal fields always index and store, matching
    /// `InvertedIndexWriter::analyze_document`'s convention).
    ///
    /// # Arguments
    ///
    /// * `fields` - Field options keyed by field name.
    ///
    /// # Returns
    ///
    /// The parser, for chaining.
    pub fn with_fields(mut self, fields: HashMap<String, FieldOption>) -> Self {
        self.fields = fields;
        self
    }

    /// Resolve `(should_index, should_store)` for `field_name`, or `None`
    /// when the field must be skipped entirely.
    ///
    /// Mirrors `InvertedIndexWriter::analyze_document`'s resolution
    /// exactly (Issue #1114): a field declared in `self.fields` follows
    /// its own option; an internal (`_`-prefixed) field or any field
    /// while `self.fields` is empty (schema-less mode) defaults to
    /// index-and-store; a field absent from an otherwise non-empty
    /// schema is skipped.
    fn field_flags(&self, field_name: &str) -> Option<(bool, bool)> {
        match self.fields.get(field_name) {
            Some(opt) => Some((opt.indexed(), opt.stored())),
            None if field_name.starts_with('_') || self.fields.is_empty() => Some((true, true)),
            None => None,
        }
    }

    /// The position-increment gap for `field_name` (Issue #1175), resolved
    /// through the same helper the writer uses so a multi-valued Text field
    /// parsed here is numbered exactly as `InvertedIndexWriter` would
    /// number it (the equivalence [`Self::with_fields`] documents).
    fn position_increment_gap(&self, field_name: &str) -> u32 {
        crate::lexical::index::inverted::writer::position_increment_gap_for(
            self.fields.get(field_name),
        )
    }

    /// Parse a document into an AnalyzedDocument.
    ///
    /// This converts text fields into tokenized terms with position information,
    /// ready to be written to the inverted index. The document ID will be assigned
    /// automatically by the index writer when the document is added.
    ///
    /// # Arguments
    ///
    /// * `doc` - The document to parse
    pub fn parse(&self, doc: Document) -> Result<AnalyzedDocument> {
        let mut field_terms = AHashMap::new();
        let mut stored_fields = AHashMap::new();
        let mut point_values = AHashMap::new();

        for (field_name, field) in &doc.fields {
            // Issue #1114: resolve the schema's (indexed, stored) gate
            // before any analysis, mirroring
            // `InvertedIndexWriter::analyze_document` exactly.
            let Some((should_index, should_store)) = self.field_flags(field_name) else {
                continue;
            };

            // Bytes, Vector and Null have no term representation.
            let indexable = !matches!(
                field,
                FieldValue::Bytes(_, _) | FieldValue::Vector(_) | FieldValue::Null
            );
            if should_index && indexable {
                // The writer's own analysis (Issue #1243), so a parsed
                // document is indexed exactly like one passed to
                // `add_document`. Unlike the writer, a field that analyzes
                // to no terms is still recorded, with length 0.
                let (terms, points) = analyze_field_value(
                    field_name,
                    field,
                    &self.analyzer,
                    self.position_increment_gap(field_name),
                )?;
                field_terms.insert(field_name.clone(), terms);
                if !points.is_empty() {
                    point_values.insert(field_name.clone(), points);
                }
            }

            if should_store {
                stored_fields.insert(field_name.clone(), field.clone());
            }
        }

        // Calculate field lengths (number of positions per field; a
        // stacked synonym shares its anchor's position and does not add to
        // the length, Issue #1257).
        let mut field_lengths = AHashMap::new();
        for (field_name, terms) in &field_terms {
            field_lengths.insert(field_name.clone(), field_length_from_terms(terms));
        }

        Ok(AnalyzedDocument {
            field_terms,
            stored_fields,
            field_lengths,
            point_values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyzer::keyword::KeywordAnalyzer;
    use crate::analysis::analyzer::per_field::PerFieldAnalyzer;
    use crate::analysis::analyzer::pipeline::PipelineAnalyzer;
    use crate::analysis::analyzer::standard::StandardAnalyzer;
    use crate::analysis::synonym::dictionary::SynonymDictionary;
    use crate::analysis::token_filter::synonym_graph::SynonymGraphFilter;
    use crate::analysis::tokenizer::whitespace::WhitespaceTokenizer;

    #[test]
    fn test_basic_parsing() {
        let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()));

        let doc = Document::builder()
            .add_text("title", "Rust Programming")
            .add_text("body", "Learn Rust")
            .build();

        let analyzed = parser.parse(doc).unwrap();

        assert!(analyzed.field_terms.contains_key("title"));
        assert!(analyzed.field_terms.contains_key("body"));
    }

    #[test]
    fn test_per_field_analyzer() {
        let per_field = PerFieldAnalyzer::new(Arc::new(StandardAnalyzer::new().unwrap()));
        per_field.add_analyzer("id", Arc::new(KeywordAnalyzer::new()));

        let parser = DocumentParser::new(Arc::new(per_field));

        let doc = Document::builder()
            .add_text("title", "Rust Programming")
            .add_text("id", "BOOK-001")
            .build();

        let analyzed = parser.parse(doc).unwrap();

        // title should be tokenized
        assert!(!analyzed.field_terms.get("title").unwrap().is_empty());
        // id should be one token (KeywordAnalyzer)
        assert_eq!(analyzed.field_terms.get("id").unwrap().len(), 1);
        assert_eq!(analyzed.field_terms.get("id").unwrap()[0].term, "BOOK-001"); // KeywordAnalyzer preserves case
    }

    #[test]
    fn test_numeric_fields() {
        let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()));

        let doc = Document::builder()
            .add_text("title", "Test")
            .add_integer("year", 2024)
            .add_float("price", 19.99)
            .add_boolean("active", true)
            .build();

        let analyzed = parser.parse(doc).unwrap();

        assert!(analyzed.field_terms.contains_key("year"));
        assert!(analyzed.field_terms.contains_key("price"));
        assert!(analyzed.field_terms.contains_key("active"));
    }

    // ------------------------------------------------------------------
    // Issue #1114: `with_fields` must gate indexed/stored the same way
    // `InvertedIndexWriter::analyze_document` does.
    // ------------------------------------------------------------------

    use crate::lexical::core::field::{BytesOption, IntegerOption, TextOption};

    #[test]
    fn test_indexed_false_skips_terms_and_lengths() {
        let mut fields = HashMap::new();
        fields.insert(
            "title".to_string(),
            FieldOption::Text(TextOption::default()),
        );
        fields.insert(
            "secret".to_string(),
            FieldOption::Text(TextOption {
                indexed: false,
                ..Default::default()
            }),
        );
        let parser =
            DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap())).with_fields(fields);

        let doc = Document::builder()
            .add_text("title", "Rust Programming")
            .add_text("secret", "classified")
            .build();

        let analyzed = parser.parse(doc).unwrap();

        assert!(analyzed.field_terms.contains_key("title"));
        assert!(
            !analyzed.field_terms.contains_key("secret"),
            "indexed: false must keep the field out of field_terms"
        );
        assert!(
            !analyzed.field_lengths.contains_key("secret"),
            "a field excluded from field_terms must also be excluded from field_lengths"
        );
        // stored: true (the default) must still be honored regardless.
        assert!(analyzed.stored_fields.contains_key("title"));
        assert!(analyzed.stored_fields.contains_key("secret"));
    }

    #[test]
    fn test_stored_false_skips_stored_value() {
        let mut fields = HashMap::new();
        fields.insert(
            "title".to_string(),
            FieldOption::Text(TextOption::default()),
        );
        fields.insert(
            "ephemeral".to_string(),
            FieldOption::Text(TextOption {
                stored: false,
                ..Default::default()
            }),
        );
        let parser =
            DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap())).with_fields(fields);

        let doc = Document::builder()
            .add_text("title", "Rust Programming")
            .add_text("ephemeral", "not kept")
            .build();

        let analyzed = parser.parse(doc).unwrap();

        assert!(
            !analyzed.stored_fields.contains_key("ephemeral"),
            "stored: false must keep the field out of stored_fields"
        );
        // indexed: true (the default) must still be honored regardless.
        assert!(analyzed.field_terms.contains_key("ephemeral"));
    }

    #[test]
    fn test_indexed_false_skips_point_values() {
        let mut fields = HashMap::new();
        fields.insert(
            "title".to_string(),
            FieldOption::Text(TextOption::default()),
        );
        fields.insert(
            "year".to_string(),
            FieldOption::Integer(IntegerOption {
                indexed: false,
                ..Default::default()
            }),
        );
        let parser =
            DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap())).with_fields(fields);

        let doc = Document::builder()
            .add_text("title", "Test")
            .add_integer("year", 2024)
            .build();

        let analyzed = parser.parse(doc).unwrap();

        assert!(
            !analyzed.point_values.contains_key("year"),
            "indexed: false must keep a numeric field's BKD points out of point_values"
        );
        assert!(!analyzed.field_terms.contains_key("year"));
        match analyzed.stored_fields.get("year") {
            Some(FieldValue::Int64(2024)) => {}
            other => panic!("expected the stored value to survive, got {other:?}"),
        }
    }

    #[test]
    fn test_bytes_field_honors_stored() {
        let mut fields = HashMap::new();
        fields.insert(
            "title".to_string(),
            FieldOption::Text(TextOption::default()),
        );
        fields.insert(
            "thumb".to_string(),
            FieldOption::Bytes(BytesOption {
                stored: true,
                ..Default::default()
            }),
        );
        fields.insert(
            "blob".to_string(),
            FieldOption::Bytes(BytesOption {
                stored: false,
                ..Default::default()
            }),
        );
        let parser =
            DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap())).with_fields(fields);

        let doc = Document::builder()
            .add_text("title", "Test")
            .add_bytes("thumb", vec![1, 2, 3])
            .add_bytes("blob", vec![4, 5, 6])
            .build();

        let analyzed = parser.parse(doc).unwrap();

        assert!(analyzed.stored_fields.contains_key("thumb"));
        assert!(
            !analyzed.stored_fields.contains_key("blob"),
            "stored: false must be honored for Bytes fields too"
        );
        // Bytes is never lexically indexed, regardless of `indexed`
        // (which BytesOption doesn't even have).
        assert!(!analyzed.field_terms.contains_key("thumb"));
        assert!(!analyzed.field_terms.contains_key("blob"));
    }

    #[test]
    fn test_schemaless_parser_indexes_and_stores_everything() {
        // No `with_fields` call: schema-less mode, exactly like #1114's
        // acceptance criterion requires.
        let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()));

        let doc = Document::builder()
            .add_text("title", "Test")
            .add_integer("year", 2024)
            .add_bytes("thumb", vec![1, 2, 3])
            .add_boolean("active", true)
            .build();

        let analyzed = parser.parse(doc).unwrap();

        for field in ["title", "year", "thumb", "active"] {
            assert!(
                analyzed.stored_fields.contains_key(field),
                "{field} must be stored in schema-less mode"
            );
        }
        for field in ["title", "year", "active"] {
            assert!(
                analyzed.field_terms.contains_key(field),
                "{field} must be indexed in schema-less mode"
            );
        }
        assert!(
            !analyzed.field_terms.contains_key("thumb"),
            "Bytes is never lexically indexed, schema-less or not"
        );
    }

    #[test]
    fn test_fields_absent_from_schema_are_skipped() {
        let mut fields = HashMap::new();
        fields.insert(
            "title".to_string(),
            FieldOption::Text(TextOption::default()),
        );
        let parser =
            DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap())).with_fields(fields);

        let doc = Document::builder()
            .add_text("title", "Test")
            .add_text("unknown", "not in schema")
            .build();

        let analyzed = parser.parse(doc).unwrap();

        assert!(analyzed.field_terms.contains_key("title"));
        assert!(analyzed.stored_fields.contains_key("title"));
        assert!(
            !analyzed.field_terms.contains_key("unknown")
                && !analyzed.stored_fields.contains_key("unknown")
                && !analyzed.point_values.contains_key("unknown"),
            "a field absent from a non-empty schema must be dropped entirely"
        );
    }

    #[test]
    fn test_internal_fields_bypass_schema() {
        let mut fields = HashMap::new();
        fields.insert(
            "title".to_string(),
            FieldOption::Text(TextOption::default()),
        );
        let parser =
            DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap())).with_fields(fields);

        let doc = Document::builder()
            .add_text("title", "Test")
            .add_text("_id", "doc-1")
            .build();

        let analyzed = parser.parse(doc).unwrap();

        assert!(
            analyzed.field_terms.contains_key("_id"),
            "an internal (_-prefixed) field must index/store by default \
             even though it isn't declared in the schema"
        );
        assert!(analyzed.stored_fields.contains_key("_id"));
    }

    // ------------------------------------------------------------------
    // Issue #1243: the parser must analyze a field exactly like the writer.
    // ------------------------------------------------------------------

    #[test]
    fn test_text_emits_one_term_per_token_with_dense_positions() {
        let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()));

        let doc = Document::builder()
            .add_text("repeated", "cat cat cat")
            .add_text("stopped", "rust the search")
            .build();

        let analyzed = parser.parse(doc).unwrap();
        let terms_of = |field: &str| -> Vec<(String, u32)> {
            analyzed.field_terms[field]
                .iter()
                .map(|t| (t.term.clone(), t.position))
                .collect()
        };

        assert_eq!(
            terms_of("repeated"),
            vec![
                ("cat".to_string(), 0),
                ("cat".to_string(), 1),
                ("cat".to_string(), 2)
            ],
            "every occurrence must be its own term"
        );
        assert_eq!(analyzed.field_lengths["repeated"], 3);

        // "the" is a stop word; the survivors are numbered densely.
        assert_eq!(
            terms_of("stopped"),
            vec![("rust".to_string(), 0), ("search".to_string(), 1)]
        );
        assert_eq!(analyzed.field_lengths["stopped"], 2);
    }

    /// Issue #1257: a synonym stacked at one position must not lengthen the
    /// field. "a big dog" with "large" stacked on "big" is 4 terms but 3
    /// positions.
    #[test]
    fn test_stacked_synonyms_do_not_lengthen_the_field() {
        let mut dict = SynonymDictionary::new(None).unwrap();
        dict.add_synonym_group(vec!["big".to_string(), "large".to_string()]);
        let analyzer = PipelineAnalyzer::new(Arc::new(WhitespaceTokenizer::new()))
            .add_filter(Arc::new(SynonymGraphFilter::new(dict, true)));
        let parser = DocumentParser::new(Arc::new(analyzer));

        let doc = Document::builder().add_text("syn", "a big dog").build();
        let analyzed = parser.parse(doc).unwrap();

        assert_eq!(analyzed.field_terms["syn"].len(), 4, "a/big/large/dog");
        assert_eq!(analyzed.field_lengths["syn"], 3);
    }

    #[test]
    fn test_datetime_term_is_epoch_seconds() {
        let parser = DocumentParser::new(Arc::new(StandardAnalyzer::new().unwrap()));
        let dt = chrono::DateTime::parse_from_rfc3339("2024-05-06T07:08:09Z")
            .unwrap()
            .with_timezone(&chrono::Utc);

        let doc = Document::builder().add_datetime("when", dt).build();
        let analyzed = parser.parse(doc).unwrap();

        let terms = &analyzed.field_terms["when"];
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].term, dt.timestamp().to_string());
        assert_eq!(analyzed.point_values["when"].len(), 1);
    }
}
