//! Lexical index reader traits for searching and retrieving documents.

use crate::error::Result;
use crate::lexical::core::document::Document;
use crate::lexical::core::field::FieldValue;
use crate::lexical::index::structures::bkd_tree::BKDTree;
use crate::lexical::index::structures::dictionary::BlockMax;
use std::sync::Arc;

/// Information about a term in the index.
#[derive(Debug, Clone)]
pub struct ReaderTermInfo {
    /// The field name.
    pub field: String,

    /// The term text.
    pub term: String,

    /// Number of documents containing this term.
    pub doc_freq: u64,

    /// Total number of occurrences of this term.
    pub total_freq: u64,

    /// Position of the term in the posting list.
    pub posting_offset: u64,

    /// Size of the posting list in bytes.
    pub posting_size: u64,

    /// Tightened BM25 TF-component upper bound precomputed at index
    /// time using the default BM25 parameters (`k1 = 1.2`, `b = 0.75`)
    /// and the segment's average field length (#403 PR-B2).
    ///
    /// Concretely, this is the maximum of
    /// `(tf · (k1 + 1)) / (tf + k1 · (1 - b + b · (L / avg_L)))` over
    /// every posting `(tf, L)`. `BM25Scorer::block_max_score` uses it
    /// to expose a tight per-term upper bound to the searcher loop's
    /// MaxScore early-break.
    ///
    /// `0.0` is treated as "unset" (legacy v1 segments / aggregated
    /// cross-segment views): scorers fall back to the loose
    /// `k1 + 1` synthetic bound in that case.
    pub max_score_factor: f32,

    /// Per-block (`BLOCK_SIZE = 128`-wide) max-impact metadata used by
    /// Block-Max-WAND (#403 PR-C). Empty if not available (legacy v2
    /// dictionaries or aggregated cross-segment views) — scorers
    /// then fall back to the term-level [`Self::max_score_factor`].
    pub block_max: Vec<BlockMax>,
}

/// Statistics about a field in the index.
#[derive(Debug, Clone)]
pub struct FieldStats {
    /// The field name.
    pub field: String,

    /// Number of unique terms in this field.
    pub unique_terms: u64,

    /// Total number of term occurrences.
    pub total_terms: u64,

    /// Number of documents with this field.
    pub doc_count: u64,

    /// Average field length.
    pub avg_length: f64,

    /// Minimum field length.
    pub min_length: u64,

    /// Maximum field length.
    pub max_length: u64,
}

/// Simplified field statistics for query scoring.
#[derive(Debug, Clone)]
pub struct FieldStatistics {
    /// Average field length.
    pub avg_field_length: f64,

    /// Number of documents with this field.
    pub doc_count: u64,

    /// Total number of terms.
    pub total_terms: u64,
}

/// Trait for lexical index readers.
///
/// Provides read-only access to a committed lexical index. Implementations
/// must be `Send + Sync` so that readers can be shared across threads
/// (e.g., to serve concurrent search requests).
///
/// A reader represents a point-in-time snapshot of the index. To see
/// newly committed data, obtain a fresh reader via
/// [`LexicalStore::refresh()`](crate::lexical::store::LexicalStore::refresh).
pub trait LexicalIndexReader: Send + Sync + std::fmt::Debug {
    /// Get the number of non-deleted documents in the index.
    fn doc_count(&self) -> u64;

    /// Get the maximum document ID in the index.
    ///
    /// This is the upper bound (exclusive) for valid document IDs and is useful
    /// for iterating over all possible document slots.
    fn max_doc(&self) -> u64;

    /// Whether any document the index holds is deleted (Issue #1211).
    ///
    /// The count fast path answers a term count from the term dictionary's
    /// `doc_freq`, which still counts deleted documents, so it may run only
    /// when this is `false`. The default compares the live count with
    /// [`Self::max_doc`]; a reader whose live count can be an upper bound
    /// overrides it with an exact test.
    fn has_effective_deletions(&self) -> bool {
        self.doc_count() != self.max_doc()
    }

    /// Check if a document has been deleted.
    ///
    /// Returns `true` if the document with the given `doc_id` has been marked
    /// as deleted, `false` otherwise.
    fn is_deleted(&self, doc_id: u64) -> bool;

    /// Get a document's stored fields by ID.
    ///
    /// Returns `Ok(Some(document))` if the document exists and has stored fields,
    /// `Ok(None)` if the document ID is not found or has no stored fields.
    fn document(&self, doc_id: u64) -> Result<Option<Document>>;

    /// Fetch a **subset** of a document's stored fields by name (#410).
    ///
    /// For wide schemas where a search request only retrieves a few of
    /// many stored fields, the default `document()` path clones the
    /// entire `HashMap<String, DataValue>` (including bytes / vector
    /// payloads of every other field) before the result-processor
    /// drops the unrequested ones. This method returns only the
    /// requested fields, so callers avoid that wasted clone.
    ///
    /// The default implementation forwards to `document()` and filters
    /// — concrete readers should override it with a path that clones
    /// each requested field individually out of the on-disk / cached
    /// document map.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - Document id to fetch.
    /// * `field_names` - Field names to retrieve.
    ///
    /// # Returns
    ///
    /// `Ok(None)` if the document does not exist; `Ok(Some(map))`
    /// where `map` contains only the requested fields that actually
    /// exist on the document.
    fn document_fields(
        &self,
        doc_id: u64,
        field_names: &[&str],
    ) -> Result<Option<std::collections::HashMap<String, crate::data::DataValue>>> {
        match self.document(doc_id)? {
            Some(doc) => {
                let mut out = std::collections::HashMap::with_capacity(field_names.len());
                for name in field_names {
                    if let Some(value) = doc.fields.get(*name) {
                        out.insert((*name).to_string(), value.clone());
                    }
                }
                Ok(Some(out))
            }
            None => Ok(None),
        }
    }

    /// Get term information for a specific field and term.
    ///
    /// Returns a [`ReaderTermInfo`] containing the document frequency, total
    /// frequency, and posting list location for the given term. Returns `None`
    /// if the term does not exist in the specified field.
    fn term_info(&self, field: &str, term: &str) -> Result<Option<ReaderTermInfo>>;

    /// Get a posting list iterator for a field and term.
    ///
    /// Returns an iterator over the documents that contain the given term
    /// in the specified field, or `None` if the term is not found.
    fn postings(&self, field: &str, term: &str) -> Result<Option<Box<dyn PostingIterator>>>;

    /// Get statistics for a field.
    ///
    /// Returns a [`FieldStats`] containing term counts, document counts, and
    /// field length statistics. Returns `None` if the field does not exist
    /// in the index or no statistics have been recorded for it.
    fn field_stats(&self, field: &str) -> Result<Option<FieldStats>>;

    /// Close the reader and release resources.
    fn close(&mut self) -> Result<()>;

    /// Check if the reader is closed.
    fn is_closed(&self) -> bool;

    /// Get BKD Tree for a numeric field, if available.
    fn get_bkd_tree(&self, field: &str) -> Result<Option<Arc<dyn BKDTree>>> {
        // Default implementation returns None (no BKD Tree support)
        let _ = field;
        Ok(None)
    }

    /// Get document frequency for a specific term in a field.
    fn term_doc_freq(&self, field: &str, term: &str) -> Result<u64> {
        match self.term_info(field, term)? {
            Some(term_info) => Ok(term_info.doc_freq),
            None => Ok(0),
        }
    }

    /// Whether [`Self::term_info`] and [`Self::term_doc_freq`] account for
    /// every document this reader can match (Issue #1196).
    ///
    /// `false` when part of the index has no term dictionary and answers
    /// term queries by scanning stored documents instead: such documents are
    /// reachable through `postings` but invisible to the dictionary, so a
    /// missing entry no longer proves a term matches nothing. Callers that
    /// short-circuit on the dictionary — `TermQuery::is_empty`, the `count`
    /// fast path — must fall through to the matcher when this is `false`.
    /// Readers without such partial-index states keep the default `true`.
    fn term_info_is_authoritative(&self) -> bool {
        true
    }

    /// Get field statistics including average field length.
    ///
    /// If the requested field is not found in the index, this default implementation
    /// returns a fallback `FieldStatistics` with `avg_field_length: 10.0`,
    /// `doc_count: 0`, and `total_terms: 0`.
    fn field_statistics(&self, field: &str) -> Result<FieldStatistics> {
        match self.field_stats(field)? {
            Some(field_stats) => Ok(FieldStatistics {
                avg_field_length: field_stats.avg_length,
                doc_count: field_stats.doc_count,
                total_terms: field_stats.total_terms,
            }),
            None => Ok(FieldStatistics {
                avg_field_length: 10.0, // Default fallback
                doc_count: 0,
                total_terms: 0,
            }),
        }
    }

    /// Get this reader as Any for downcasting.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get a DocValues field value for a document.
    /// Returns None if DocValues are not available for this field or document.
    fn get_doc_value(&self, field: &str, doc_id: u64) -> Result<Option<FieldValue>> {
        // Default implementation returns None (no DocValues support)
        let _ = (field, doc_id);
        Ok(None)
    }

    /// Check if DocValues are available for a field.
    fn has_doc_values(&self, field: &str) -> bool {
        // Default implementation returns false
        let _ = field;
        false
    }

    /// Get all document IDs available in the index.
    fn doc_ids(&self) -> Result<Vec<u64>> {
        // Default implementation returns empty (no ID enumeration support)
        Ok(Vec::new())
    }
}

/// Enumerate the document ids a stored-document scan should probe.
///
/// Prefers [`LexicalIndexReader::doc_ids`], which yields exactly the ids
/// present in the reader. This keeps the scan segment-bounded under the
/// per-segment fanout (where `max_doc()` reports the *global* document
/// count) and correct for sparse id spaces (post-merge segments whose
/// surviving ids exceed `max_doc()`, non-zero shard offsets) — see #994,
/// #996. The returned ids are sorted and deduplicated so matchers built
/// from them observe the ascending doc-id contract.
///
/// Readers without id enumeration support (their `doc_ids()` is empty
/// while `max_doc()` is non-zero) fall back to the legacy dense
/// `0..max_doc()` range.
///
/// # Arguments
///
/// * `reader` - The index reader whose document ids should be probed.
///
/// # Returns
///
/// An iterator over the document ids to probe, in ascending order.
pub(crate) fn scan_doc_ids(
    reader: &dyn LexicalIndexReader,
) -> Result<Box<dyn Iterator<Item = u64>>> {
    let mut ids = reader.doc_ids()?;
    if ids.is_empty() && reader.max_doc() > 0 {
        return Ok(Box::new(0..reader.max_doc()));
    }
    ids.sort_unstable();
    ids.dedup();
    Ok(Box::new(ids.into_iter()))
}

/// Iterator over a posting list for a single term.
///
/// Yields document IDs (and associated term frequencies/positions) in
/// ascending order. Callers must call [`next()`](Self::next) before reading
/// [`doc_id()`](Self::doc_id) for the first time.
///
/// # Iteration Protocol
///
/// 1. Call [`next()`](Self::next) to advance to the first/next posting.
/// 2. If `next()` returns `Ok(true)`, read [`doc_id()`](Self::doc_id),
///    [`term_freq()`](Self::term_freq), and optionally [`positions()`](Self::positions).
/// 3. If `next()` returns `Ok(false)`, the iterator is exhausted.
///
/// Use [`skip_to()`](Self::skip_to) to efficiently jump ahead to a target document ID.
pub trait PostingIterator: Send + std::fmt::Debug {
    /// Get the current document ID.
    ///
    /// The value is only valid after a successful call to [`next()`](Self::next)
    /// or [`skip_to()`](Self::skip_to) that returned `Ok(true)`.
    fn doc_id(&self) -> u64;

    /// Get the term frequency in the current document.
    ///
    /// Returns the number of times the term appears in the document at the
    /// current iterator position.
    fn term_freq(&self) -> u64;

    /// Get the term positions within the current document.
    ///
    /// Returns an ordered list of 0-based token positions where the term
    /// occurs. Only meaningful if position data was stored during indexing.
    fn positions(&self) -> Result<Vec<u64>>;

    /// Advance the iterator to the next posting.
    ///
    /// Returns `Ok(true)` if a next posting exists, or `Ok(false)` if the
    /// iterator is exhausted. After `Ok(false)`, no further calls should
    /// be made.
    fn next(&mut self) -> Result<bool>;

    /// Skip forward to the first posting with `doc_id >= target`.
    ///
    /// Returns `Ok(true)` if such a posting exists, or `Ok(false)` if all
    /// remaining postings have `doc_id < target` (i.e., the iterator is
    /// exhausted). This is more efficient than repeatedly calling
    /// [`next()`](Self::next) when large gaps in document IDs are expected.
    fn skip_to(&mut self, target: u64) -> Result<bool>;

    /// Get the estimated cost (number of postings) of this iterator.
    ///
    /// Used by query planning to choose efficient execution strategies
    /// (e.g., deciding iteration order for conjunctive queries).
    fn cost(&self) -> u64;
}
