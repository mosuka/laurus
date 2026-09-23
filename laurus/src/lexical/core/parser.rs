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
//! - **DateTime**: Converted to RFC3339 format
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
use crate::analysis::analyzer::per_field::PerFieldAnalyzer;
use crate::analysis::token::Token;
use crate::error::Result;
use crate::lexical::core::analyzed::{AnalyzedDocument, AnalyzedTerm};
use crate::lexical::core::document::Document;
use crate::lexical::core::field::{FieldOption, FieldValue};

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

        // Process each field in the document
        for (field_name, field) in &doc.fields {
            // Issue #1114: resolve the schema's (indexed, stored) gate
            // before doing any type-specific analysis below, mirroring
            // `InvertedIndexWriter::analyze_document` exactly.
            let Some((should_index, should_store)) = self.field_flags(field_name) else {
                continue;
            };

            if should_index {
                match field {
                    FieldValue::Text(text) => {
                        // Analyze text field with per-field analyzer
                        let tokens = if let Some(per_field) =
                            self.analyzer.as_any().downcast_ref::<PerFieldAnalyzer>()
                        {
                            per_field.analyze_field(field_name.as_str(), text.as_str())?
                        } else {
                            self.analyzer.analyze(text.as_str())?
                        };

                        let token_vec: Vec<Token> = tokens.collect();
                        let analyzed_terms = self.tokens_to_analyzed_terms(token_vec);

                        field_terms.insert(field_name.clone(), analyzed_terms);
                    }
                    FieldValue::Int64(num) => {
                        // Convert integer to text for indexing
                        let text = num.to_string();

                        let analyzed_term = AnalyzedTerm {
                            term: text.clone(),
                            position: 0,
                            frequency: 1,
                            offset: (0, text.len()),
                        };

                        field_terms.insert(field_name.clone(), vec![analyzed_term]);
                        point_values.insert(field_name.clone(), vec![vec![*num as f64]]);
                    }
                    FieldValue::Float64(num) => {
                        // Convert float to text for indexing
                        let text = num.to_string();

                        let analyzed_term = AnalyzedTerm {
                            term: text.clone(),
                            position: 0,
                            frequency: 1,
                            offset: (0, text.len()),
                        };

                        field_terms.insert(field_name.clone(), vec![analyzed_term]);
                        point_values.insert(field_name.clone(), vec![vec![*num]]);
                    }
                    FieldValue::Bool(b) => {
                        // Convert boolean to text
                        let text = b.to_string();

                        let analyzed_term = AnalyzedTerm {
                            term: text.clone(),
                            position: 0,
                            frequency: 1,
                            offset: (0, text.len()),
                        };

                        field_terms.insert(field_name.clone(), vec![analyzed_term]);
                    }
                    FieldValue::DateTime(dt) => {
                        // Convert datetime to RFC3339 string
                        let text = dt.to_rfc3339();

                        let analyzed_term = AnalyzedTerm {
                            term: text.clone(),
                            position: 0,
                            frequency: 1,
                            offset: (0, text.len()),
                        };

                        field_terms.insert(field_name.clone(), vec![analyzed_term]);
                        // Same BKD encoding as the inverted-index writer (#1179).
                        let ts = crate::lexical::core::datetime::datetime_to_point(dt);
                        point_values.insert(field_name.clone(), vec![vec![ts]]);
                    }
                    FieldValue::Geo(point) => {
                        // Convert geo point to string representation
                        let text = format!("{},{}", point.lat, point.lon);

                        let analyzed_term = AnalyzedTerm {
                            term: text.clone(),
                            position: 0,
                            frequency: 1,
                            offset: (0, text.len()),
                        };

                        field_terms.insert(field_name.clone(), vec![analyzed_term]);
                        // Geo is a single 2D point.
                        point_values.insert(field_name.clone(), vec![vec![point.lat, point.lon]]);
                    }
                    FieldValue::GeoEcef(point) => {
                        // 3D ECEF point: index the (x, y, z) tuple as a single
                        // BKD entry. Mirrors the 2D Geo flow (text term + point
                        // values); `FieldOption::Geo3d` (#298) drives the
                        // schema-side decisions and `BKDWriter` infers the
                        // dimensionality from the point length, so emitting a
                        // 3-element point here is enough to land on a 3D BKD.
                        let text = format!("{},{},{}", point.x, point.y, point.z);

                        let analyzed_term = AnalyzedTerm {
                            term: text.clone(),
                            position: 0,
                            frequency: 1,
                            offset: (0, text.len()),
                        };

                        field_terms.insert(field_name.clone(), vec![analyzed_term]);
                        point_values
                            .insert(field_name.clone(), vec![vec![point.x, point.y, point.z]]);
                    }
                    FieldValue::Int64Array(arr) => {
                        // Multi-valued integer: each element becomes its own
                        // analyzed term and a separate 1D BKD point so range
                        // queries match when any value satisfies the predicate.
                        let mut terms: Vec<AnalyzedTerm> = Vec::with_capacity(arr.len());
                        let mut points: Vec<Vec<f64>> = Vec::with_capacity(arr.len());
                        let mut offset = 0usize;
                        for (idx, num) in arr.iter().enumerate() {
                            let text = num.to_string();
                            let len = text.len();
                            terms.push(AnalyzedTerm {
                                term: text,
                                position: idx as u32,
                                frequency: 1,
                                offset: (offset, offset + len),
                            });
                            offset += len + 1;
                            points.push(vec![*num as f64]);
                        }
                        field_terms.insert(field_name.clone(), terms);
                        point_values.insert(field_name.clone(), points);
                    }
                    FieldValue::Float64Array(arr) => {
                        // Multi-valued float: same shape as Int64Array.
                        let mut terms: Vec<AnalyzedTerm> = Vec::with_capacity(arr.len());
                        let mut points: Vec<Vec<f64>> = Vec::with_capacity(arr.len());
                        let mut offset = 0usize;
                        for (idx, num) in arr.iter().enumerate() {
                            let text = num.to_string();
                            let len = text.len();
                            terms.push(AnalyzedTerm {
                                term: text,
                                position: idx as u32,
                                frequency: 1,
                                offset: (offset, offset + len),
                            });
                            offset += len + 1;
                            points.push(vec![*num]);
                        }
                        field_terms.insert(field_name.clone(), terms);
                        point_values.insert(field_name.clone(), points);
                    }
                    FieldValue::GeoArray(arr) => {
                        // Multi-valued geo (#1174): one 2-D BKD point per
                        // element, same shape as Int64Array.
                        let mut terms: Vec<AnalyzedTerm> = Vec::with_capacity(arr.len());
                        let mut points: Vec<Vec<f64>> = Vec::with_capacity(arr.len());
                        let mut offset = 0usize;
                        for (idx, p) in arr.iter().enumerate() {
                            let text = format!("{},{}", p.lat, p.lon);
                            let len = text.len();
                            terms.push(AnalyzedTerm {
                                term: text,
                                position: idx as u32,
                                frequency: 1,
                                offset: (offset, offset + len),
                            });
                            offset += len + 1;
                            points.push(vec![p.lat, p.lon]);
                        }
                        field_terms.insert(field_name.clone(), terms);
                        point_values.insert(field_name.clone(), points);
                    }
                    FieldValue::GeoEcefArray(arr) => {
                        // Multi-valued ECEF (#1174): one 3-D BKD point per
                        // element.
                        let mut terms: Vec<AnalyzedTerm> = Vec::with_capacity(arr.len());
                        let mut points: Vec<Vec<f64>> = Vec::with_capacity(arr.len());
                        let mut offset = 0usize;
                        for (idx, p) in arr.iter().enumerate() {
                            let text = format!("{},{},{}", p.x, p.y, p.z);
                            let len = text.len();
                            terms.push(AnalyzedTerm {
                                term: text,
                                position: idx as u32,
                                frequency: 1,
                                offset: (offset, offset + len),
                            });
                            offset += len + 1;
                            points.push(vec![p.x, p.y, p.z]);
                        }
                        field_terms.insert(field_name.clone(), terms);
                        point_values.insert(field_name.clone(), points);
                    }
                    FieldValue::DateTimeArray(arr) => {
                        // Multi-valued datetime (#1184): one 1-D BKD point
                        // per element, same shape as Int64Array.
                        let mut terms: Vec<AnalyzedTerm> = Vec::with_capacity(arr.len());
                        let mut points: Vec<Vec<f64>> = Vec::with_capacity(arr.len());
                        let mut offset = 0usize;
                        for (idx, dt) in arr.iter().enumerate() {
                            let text = dt.to_rfc3339();
                            let len = text.len();
                            terms.push(AnalyzedTerm {
                                term: text,
                                position: idx as u32,
                                frequency: 1,
                                offset: (offset, offset + len),
                            });
                            offset += len + 1;
                            points
                                .push(vec![crate::lexical::core::datetime::datetime_to_point(dt)]);
                        }
                        field_terms.insert(field_name.clone(), terms);
                        point_values.insert(field_name.clone(), points);
                    }
                    FieldValue::BoolArray(arr) => {
                        // Multi-valued boolean (#1180): one "true"/"false"
                        // term per element and, like the scalar `Bool` arm,
                        // no BKD point.
                        let mut terms: Vec<AnalyzedTerm> = Vec::with_capacity(arr.len());
                        let mut offset = 0usize;
                        for (idx, b) in arr.iter().enumerate() {
                            let text = b.to_string();
                            let len = text.len();
                            terms.push(AnalyzedTerm {
                                term: text,
                                position: idx as u32,
                                frequency: 1,
                                offset: (offset, offset + len),
                            });
                            offset += len + 1;
                        }
                        field_terms.insert(field_name.clone(), terms);
                    }
                    FieldValue::TextArray(arr) => {
                        // Multi-valued text (#1175): each element analyzed
                        // separately onto one ascending position sequence,
                        // separated by the field's position-increment gap.
                        // Mirrors `analyze_field_value`'s arm; see it for
                        // why the sequence must stay contiguous.
                        let gap = self.position_increment_gap(field_name);
                        let mut terms: Vec<AnalyzedTerm> = Vec::new();
                        let mut base = 0u32;
                        for (idx, text) in arr.iter().enumerate() {
                            if idx > 0 {
                                base = base.saturating_add(gap);
                            }
                            let tokens = if let Some(per_field) =
                                self.analyzer.as_any().downcast_ref::<PerFieldAnalyzer>()
                            {
                                per_field.analyze_field(field_name.as_str(), text.as_str())?
                            } else {
                                self.analyzer.analyze(text.as_str())?
                            };
                            let token_vec: Vec<Token> = tokens.collect();
                            let token_count = token_vec.len() as u32;
                            for mut term in self.tokens_to_analyzed_terms(token_vec) {
                                term.position = term.position.saturating_add(base);
                                terms.push(term);
                            }
                            base = base.saturating_add(token_count);
                        }
                        field_terms.insert(field_name.clone(), terms);
                    }
                    // Not lexically indexable: no term representation
                    // exists for these, regardless of `should_index`.
                    // Spelled out rather than a wildcard `_ =>` so a new
                    // `FieldValue` variant still fails exhaustiveness
                    // checking here.
                    FieldValue::Bytes(_, _) | FieldValue::Vector(_) | FieldValue::Null => {}
                }
            }

            if should_store {
                stored_fields.insert(field_name.clone(), field.clone());
            }
        }

        // Calculate field lengths (number of tokens per field)
        let mut field_lengths = AHashMap::new();
        for (field_name, terms) in &field_terms {
            field_lengths.insert(field_name.clone(), terms.len() as u32);
        }

        Ok(AnalyzedDocument {
            field_terms,
            stored_fields,
            field_lengths,
            point_values,
        })
    }

    /// Convert tokens to analyzed terms with position and frequency information.
    fn tokens_to_analyzed_terms(&self, tokens: Vec<Token>) -> Vec<AnalyzedTerm> {
        // Type alias for clarity: maps term text to list of (position, (start_offset, end_offset))
        type TermPositionMap = AHashMap<String, Vec<(u32, (usize, usize))>>;
        let mut term_positions: TermPositionMap = AHashMap::new();

        // Group positions by term
        for token in tokens {
            term_positions.entry(token.text.clone()).or_default().push((
                token.position as u32,
                (token.start_offset, token.end_offset),
            ));
        }

        // Create analyzed terms
        term_positions
            .into_iter()
            .map(|(term, positions)| {
                let frequency = positions.len() as u32;
                let position = positions[0].0; // Use first position
                let offset = positions[0].1; // Use first offset

                AnalyzedTerm {
                    term,
                    position,
                    frequency,
                    offset,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::analyzer::keyword::KeywordAnalyzer;
    use crate::analysis::analyzer::standard::StandardAnalyzer;

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
            FieldOption::Bytes(BytesOption { stored: true }),
        );
        fields.insert(
            "blob".to_string(),
            FieldOption::Bytes(BytesOption { stored: false }),
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
}
