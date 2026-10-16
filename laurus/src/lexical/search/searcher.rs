//! Searcher trait for lexical search execution.

use std::collections::HashMap;
use std::sync::Arc;

use roaring::RoaringTreemap;

use crate::error::Result;
use crate::lexical::core::field::FieldValue;
use crate::lexical::query::{LexicalSearchResults, Query};

/// Sort order for search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    /// Ascending order (lowest to highest).
    #[default]
    Asc,
    /// Descending order (highest to lowest).
    Desc,
}

/// Field to sort search results by.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SortField {
    /// Sort by relevance score (default).
    #[default]
    Score,
    /// Sort by a document field value.
    Field {
        /// Field name to sort by.
        name: String,
        /// Sort order.
        order: SortOrder,
    },
}

/// Configuration for search operations.
#[derive(Debug, Clone)]
pub struct LexicalSearchParams {
    /// Maximum number of results to return.
    pub limit: usize,
    /// Minimum score threshold.
    pub min_score: f32,
    /// Whether to load document content.
    pub load_documents: bool,
    /// Timeout for search operations in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Enable parallel search for better performance on multi-core systems.
    pub parallel: bool,
    /// Sort results by field or score.
    pub sort_by: SortField,
}

impl Default for LexicalSearchParams {
    fn default() -> Self {
        LexicalSearchParams {
            limit: 10,
            min_score: 0.0,
            load_documents: true,
            timeout_ms: None,
            parallel: false,
            sort_by: SortField::default(),
        }
    }
}

/// Query representation that can be either a DSL string or a Query object.
#[derive(Debug)]
pub enum LexicalSearchQuery {
    /// Query specified as a DSL string (will be parsed at search time).
    Dsl(String),
    /// Query specified as a Query object.
    Obj(Box<dyn Query>),
}

/// Search request containing query and configuration.
#[derive(Debug)]
pub struct LexicalSearchRequest {
    /// The query to execute.
    pub query: LexicalSearchQuery,
    /// Search configuration.
    pub params: LexicalSearchParams,
    /// Field-level boosts for lexical scoring.
    /// Applied at the Engine level before search execution.
    pub field_boosts: HashMap<String, f32>,
}

impl Clone for LexicalSearchQuery {
    fn clone(&self) -> Self {
        match self {
            LexicalSearchQuery::Dsl(s) => LexicalSearchQuery::Dsl(s.clone()),
            LexicalSearchQuery::Obj(q) => LexicalSearchQuery::Obj(q.clone_box()),
        }
    }
}

impl Clone for LexicalSearchRequest {
    fn clone(&self) -> Self {
        LexicalSearchRequest {
            query: self.query.clone(),
            params: self.params.clone(),
            field_boosts: self.field_boosts.clone(),
        }
    }
}

impl From<String> for LexicalSearchQuery {
    fn from(s: String) -> Self {
        LexicalSearchQuery::Dsl(s)
    }
}

impl From<&str> for LexicalSearchQuery {
    fn from(s: &str) -> Self {
        LexicalSearchQuery::Dsl(s.to_string())
    }
}

impl From<Box<dyn Query>> for LexicalSearchQuery {
    fn from(q: Box<dyn Query>) -> Self {
        LexicalSearchQuery::Obj(q)
    }
}

impl LexicalSearchQuery {
    /// Parse DSL string into Query object using the given analyzer.
    pub fn into_query(
        self,
        analyzer: &Arc<dyn crate::analysis::analyzer::analyzer::Analyzer>,
    ) -> crate::error::Result<Box<dyn Query>> {
        match self {
            LexicalSearchQuery::Dsl(dsl_string) => {
                let parser =
                    crate::lexical::query::parser::LexicalQueryParser::new(analyzer.clone());
                parser.parse(&dsl_string)
            }
            LexicalSearchQuery::Obj(query) => Ok(query),
        }
    }

    /// Extract the Query object.
    ///
    /// Returns an error if this is a DSL string variant.
    pub fn unwrap_query(self) -> crate::error::Result<Box<dyn Query>> {
        match self {
            LexicalSearchQuery::Obj(query) => Ok(query),
            LexicalSearchQuery::Dsl(_) => Err(crate::error::LaurusError::invalid_argument(
                "Expected Query object, found DSL string",
            )),
        }
    }
}

impl LexicalSearchRequest {
    /// Create a new search request from a query object.
    pub fn new(query: Box<dyn Query>) -> Self {
        LexicalSearchRequest {
            query: LexicalSearchQuery::Obj(query),
            params: LexicalSearchParams::default(),
            field_boosts: HashMap::new(),
        }
    }

    /// Create a new search request from a DSL query string.
    pub fn from_dsl(dsl: impl Into<String>) -> Self {
        LexicalSearchRequest {
            query: LexicalSearchQuery::Dsl(dsl.into()),
            params: LexicalSearchParams::default(),
            field_boosts: HashMap::new(),
        }
    }

    /// Set the maximum number of results to return.
    pub fn limit(mut self, limit: usize) -> Self {
        self.params.limit = limit;
        self
    }

    /// Set the minimum score threshold.
    pub fn min_score(mut self, min_score: f32) -> Self {
        self.params.min_score = min_score;
        self
    }

    /// Set whether to load document content.
    pub fn load_documents(mut self, load_documents: bool) -> Self {
        self.params.load_documents = load_documents;
        self
    }

    /// Set the search timeout.
    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.params.timeout_ms = Some(timeout_ms);
        self
    }

    /// Enable parallel search.
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.params.parallel = parallel;
        self
    }

    /// Sort results by a field in ascending order.
    pub fn sort_by_field_asc(mut self, field: &str) -> Self {
        self.params.sort_by = SortField::Field {
            name: field.to_string(),
            order: SortOrder::Asc,
        };
        self
    }

    /// Sort results by a field in descending order.
    pub fn sort_by_field_desc(mut self, field: &str) -> Self {
        self.params.sort_by = SortField::Field {
            name: field.to_string(),
            order: SortOrder::Desc,
        };
        self
    }

    /// Sort results by relevance score (default).
    pub fn sort_by_score(mut self) -> Self {
        self.params.sort_by = SortField::Score;
        self
    }

    /// Add a field-level boost for lexical scoring.
    pub fn with_field_boost(mut self, field: impl Into<String>, boost: f32) -> Self {
        self.field_boosts.insert(field.into(), boost);
        self
    }
}

/// One document's answer from [`LexicalSearcher::doc_values_batch`]
/// (Issue #1297).
#[derive(Debug, Clone, PartialEq)]
pub enum DocValueLookup {
    /// The document's DocValues value.
    Value(FieldValue),
    /// The document's segment has a DocValues column for the field and no
    /// value in it for the document: the document has no value.
    Absent,
    /// No DocValues column answers for the document: its segment predates
    /// the column, the snapshot does not hold the document, or the searcher
    /// keeps no DocValues. Read the stored document instead.
    Unknown,
}

/// Trait for lexical search implementations.
///
/// This trait defines the interface for executing searches against lexical indexes.
pub trait LexicalSearcher: Send + Sync + std::fmt::Debug {
    /// Execute a search with the given request.
    fn search(&self, request: LexicalSearchRequest) -> Result<LexicalSearchResults>;

    /// Count the number of matching documents for a request.
    ///
    /// Returns the number of documents that match the given search request,
    /// applying the min_score threshold if specified in the request parameters.
    fn count(&self, request: LexicalSearchRequest) -> Result<u64>;

    /// The live documents whose `field` holds exactly `term`, looked up
    /// without running a search (Issue #1420).
    ///
    /// Must return the same documents as an unbounded search for a
    /// `TermQuery` on the term, or `None` when the implementation cannot
    /// answer that way; the caller then runs that search. The default
    /// implementation always returns `None`.
    ///
    /// # Arguments
    ///
    /// * `field` - Field the term belongs to.
    /// * `term` - The exact term, as indexed.
    ///
    /// # Returns
    ///
    /// The matching document ids, in no particular order, or `None`.
    fn live_doc_ids_for_term(&self, field: &str, term: &str) -> Result<Option<Vec<u64>>> {
        let _ = (field, term);
        Ok(None)
    }

    /// The DocValues value of `field` for each of `doc_ids`, in the same
    /// order, read from this searcher's snapshot (Issue #1297).
    ///
    /// The default implementation answers
    /// [`Unknown`](DocValueLookup::Unknown) for every id.
    ///
    /// # Arguments
    ///
    /// * `field` - The DocValues field.
    /// * `doc_ids` - The internal document ids to look up.
    ///
    /// # Returns
    ///
    /// One [`DocValueLookup`] per id.
    fn doc_values_batch(&self, field: &str, doc_ids: &[u64]) -> Result<Vec<DocValueLookup>> {
        let _ = field;
        Ok(vec![DocValueLookup::Unknown; doc_ids.len()])
    }

    /// Return the set of document ids matching `query`, ignoring score
    /// thresholds (Issue [#578](https://github.com/mosuka/laurus/issues/578)).
    ///
    /// This powers filter evaluation: a filter selects documents without
    /// affecting relevance, so the result is a score-independent doc-id set.
    /// Implementations backed by a snapshot-scoped cache (notably
    /// [`InvertedIndexSearcher`](crate::lexical::index::inverted::searcher::InvertedIndexSearcher))
    /// memoise the set so repeated filters are served without re-walking
    /// posting lists.
    ///
    /// The default implementation drains an unbounded [`search`](Self::search)
    /// and collects the matching doc ids — correct for any searcher but
    /// uncached. The returned set excludes deleted documents.
    ///
    /// # Arguments
    ///
    /// * `query` - The query whose matching document set is requested.
    ///
    /// # Returns
    ///
    /// An `Arc<RoaringTreemap>` of matching document ids.
    fn matching_doc_ids(&self, query: Box<dyn Query>) -> Result<Arc<RoaringTreemap>> {
        let request = LexicalSearchRequest::new(query)
            .limit(usize::MAX)
            .load_documents(false);
        let results = self.search(request)?;
        let mut bitmap = RoaringTreemap::new();
        for hit in results.hits {
            bitmap.insert(hit.doc_id);
        }
        Ok(Arc::new(bitmap))
    }
}
