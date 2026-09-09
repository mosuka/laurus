use std::collections::HashMap;

use crate::lexical::query::Query;
use crate::lexical::search::searcher::{LexicalSearchQuery, SortField};
// Re-export VectorSearchQuery so engine.rs and query.rs can refer to it
// via `self::search::VectorSearchQuery` without reaching into vector internals.
use crate::vector::VectorScoreMode;
pub use crate::vector::search::searcher::VectorSearchQuery;

// ── Query types (what to search for) ─────────────────────────────────────────

/// Unified search query specification.
///
/// Determines **what** to search for. Search parameters (limits, score
/// thresholds, fusion, etc.) are separate fields on [`SearchRequest`].
///
/// Four variants cover all search modes:
///
/// - [`Dsl`](Self::Dsl) — unified query DSL string, parsed at search time.
/// - [`Lexical`](Self::Lexical) — lexical (BM25) search only.
/// - [`Vector`](Self::Vector) — vector (nearest-neighbor) search only.
/// - [`Hybrid`](Self::Hybrid) — both lexical and vector search with fusion.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum SearchQuery {
    /// Unified query DSL string — parsed at search time by
    /// [`UnifiedQueryParser`](super::query::UnifiedQueryParser).
    ///
    /// Supports lexical, vector, and hybrid queries in a single string:
    ///
    /// - **Lexical**: `title:hello`, `"exact phrase"`, `AND`/`OR`, `term~2`,
    ///   `[a TO z]`, etc.
    /// - **Vector**: `field:"text"`, `field:text^0.8` (with boost).
    /// - **Hybrid**: mix both — `title:hello content:"cute kitten"^0.8`.
    Dsl(String),

    /// Pre-built lexical (BM25) search query.
    Lexical(LexicalSearchQuery),

    /// Pre-built vector (nearest-neighbor) search query.
    Vector(VectorSearchQuery),

    /// Hybrid search combining lexical and vector components.
    ///
    /// Results are merged using the [`fusion_algorithm`](SearchRequest::fusion_algorithm)
    /// specified on the [`SearchRequest`]. The [`mode`](HybridMode) controls
    /// whether results are unioned (OR) or intersected (AND).
    Hybrid {
        /// Lexical search component.
        lexical: LexicalSearchQuery,
        /// Vector search component.
        vector: VectorSearchQuery,
        /// Controls how lexical and vector results are combined.
        /// Defaults to [`HybridMode::Union`].
        mode: HybridMode,
    },
}

/// Controls how lexical and vector results are combined in hybrid search.
///
/// - [`Union`](Self::Union) — documents from **either** lexical or vector
///   results are included (OR semantics). This is the default.
/// - [`Intersection`](Self::Intersection) — only documents appearing in
///   **both** result sets are included (AND semantics). Triggered by
///   the `+` prefix on vector field clauses in the query DSL, e.g.
///   `title:hello +embedding:"cute kitten"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HybridMode {
    /// Documents from either source are included (default).
    #[default]
    Union,
    /// Only documents appearing in BOTH result sets are included.
    Intersection,
}

// ── Option types (how to search) ─────────────────────────────────────────────

/// Parameters controlling lexical search behavior.
///
/// These are separated from the query itself so that the same options can
/// be applied regardless of how the query was specified (DSL string or
/// pre-built query object).
#[derive(Debug, Clone)]
pub struct LexicalSearchOptions {
    /// Per-field boost factors for relevance scoring.
    ///
    /// Example: `{"title": 2.0, "body": 1.0}` gives title matches twice
    /// the weight of body matches.
    pub field_boosts: HashMap<String, f32>,

    /// Minimum score threshold. Results below this score are discarded.
    /// Defaults to `0.0` (no threshold).
    pub min_score: f32,

    /// Timeout for the search operation in milliseconds.
    /// `None` means no timeout.
    pub timeout_ms: Option<u64>,

    /// Enable parallel search across index segments for better performance
    /// on multi-core systems. Defaults to `false`.
    pub parallel: bool,

    /// Sort results by field value or by relevance score.
    /// Defaults to [`SortField::Score`].
    pub sort_by: SortField,
}

impl Default for LexicalSearchOptions {
    fn default() -> Self {
        Self {
            field_boosts: HashMap::new(),
            min_score: 0.0,
            timeout_ms: None,
            parallel: false,
            sort_by: SortField::Score,
        }
    }
}

/// Parameters controlling vector search behavior.
///
/// These are separated from the query itself so that the same options can
/// be applied regardless of how the query was specified (payloads or
/// pre-embedded vectors).
#[derive(Debug, Clone)]
pub struct VectorSearchOptions {
    /// How to combine scores from multiple query vectors.
    /// Defaults to [`VectorScoreMode::WeightedSum`].
    pub score_mode: VectorScoreMode,

    /// Minimum score threshold. Results below this score are discarded.
    /// Defaults to `0.0` (no threshold). Compared against the raw
    /// per-field similarity BEFORE `QueryVector.weight` and the field's
    /// `base_weight` (Issue #1084) are applied — a similarity floor, not a
    /// final-score floor.
    pub min_score: f32,

    /// Optional Stage 2 rerank factor (Issue #481).
    ///
    /// When `Some(factor)`, the underlying vector index searcher widens
    /// the int8 candidate fetch to `top_k * factor` and rescores the
    /// candidates against the original full-precision vectors via the
    /// LRS1 sidecar. Honored only on HNSW fields whose schema enabled
    /// `rerank_storage`; other configurations silently ignore the value.
    /// `None` keeps Stage 1 behavior (int8-only).
    pub rerank_factor: Option<usize>,

    /// Per-query override for the HNSW `ef_search` candidate-list size
    /// (Issue [#644](https://github.com/mosuka/laurus/issues/644)).
    ///
    /// When `None` (the default), the searcher uses the schema-level
    /// [`HnswOption::default_ef_search`](crate::vector::core::field::HnswOption::default_ef_search)
    /// or its internal fallback (`50`). Ignored by non-HNSW index types.
    /// The effective `ef_search` is always lifted to at least `top_k`
    /// (and `top_k * rerank_factor` when both are set) so the candidate
    /// heap is never undersized for the requested `top_k`.
    pub ef_search: Option<usize>,
}

impl Default for VectorSearchOptions {
    fn default() -> Self {
        Self {
            score_mode: VectorScoreMode::WeightedSum,
            min_score: 0.0,
            rerank_factor: None,
            ef_search: None,
        }
    }
}

// ── SearchRequest ────────────────────────────────────────────────────────────

/// Unified search request combining query specification with pagination,
/// options, and fusion settings.
///
/// The query specifies **what** to search for ([`SearchQuery`]), while
/// [`lexical_options`](Self::lexical_options) and
/// [`vector_options`](Self::vector_options) control **how** to search.
///
/// Use [`SearchRequestBuilder`] for a fluent construction API.
pub struct SearchRequest {
    /// The search query specification.
    pub query: SearchQuery,

    /// Maximum number of results to return. Defaults to `10`.
    pub limit: usize,

    /// Number of results to skip before returning (for pagination).
    /// Defaults to `0`.
    pub offset: usize,

    /// Fusion algorithm for combining lexical and vector scores.
    ///
    /// Only used when both lexical and vector search components are
    /// present (i.e., [`SearchQuery::Hybrid`] or a [`SearchQuery::Dsl`]
    /// that contains both clause types). Defaults to
    /// [`FusionAlgorithm::RRF { k: 60.0 }`](FusionAlgorithm::RRF) when
    /// `None`.
    pub fusion_algorithm: Option<FusionAlgorithm>,

    /// Optional filter query (lexical) to restrict the search space.
    ///
    /// When set, the filter is evaluated first and **both** lexical and
    /// vector searches are restricted to documents matching this filter.
    pub filter_query: Option<Box<dyn Query>>,

    /// Parameters controlling lexical search behavior.
    pub lexical_options: LexicalSearchOptions,

    /// Parameters controlling vector search behavior.
    pub vector_options: VectorSearchOptions,
}

/// Algorithm used to combine lexical and vector scores in hybrid search.
///
/// The default fusion algorithm (when none is specified in a
/// [`SearchRequest`]) is [`RRF`](Self::RRF) with `k = 60.0`.
#[derive(Debug, Clone, Copy)]
pub enum FusionAlgorithm {
    /// Reciprocal Rank Fusion (RRF).
    ///
    /// Combines results based on rank position rather than raw scores,
    /// making it effective when score magnitudes are not comparable
    /// (e.g. BM25 vs cosine similarity). The score for each document is
    /// `sum(1 / (k + rank))` across the result lists.
    RRF {
        /// Smoothing constant `k`. Higher values reduce the influence of
        /// top-ranked documents. Typical default is `60.0`.
        k: f64,
    },

    /// Weighted Sum with automatic min-max score normalization.
    ///
    /// Before weighting, the engine independently normalizes lexical and
    /// vector scores to the `[0.0, 1.0]` range using min-max normalization
    /// over their respective result sets.
    WeightedSum {
        /// Weight for the normalized lexical score (clamped to `0.0..=1.0`).
        lexical_weight: f32,
        /// Weight for the normalized vector score (clamped to `0.0..=1.0`).
        vector_weight: f32,
    },
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            query: SearchQuery::Dsl(String::new()),
            limit: 10,
            offset: 0,
            fusion_algorithm: None,
            filter_query: None,
            lexical_options: LexicalSearchOptions::default(),
            vector_options: VectorSearchOptions::default(),
        }
    }
}

// ── SearchRequestBuilder ─────────────────────────────────────────────────────

/// Fluent builder for constructing a [`SearchRequest`].
///
/// Supports three construction patterns:
///
/// 1. **DSL string** (via [`query_dsl`](Self::query_dsl)): Pass a unified
///    query DSL string. The engine parses it at search time.
/// 2. **Single mode** (via [`lexical_query`](Self::lexical_query) or
///    [`vector_query`](Self::vector_query)): Set one search mode.
/// 3. **Hybrid** (via both [`lexical_query`](Self::lexical_query) and
///    [`vector_query`](Self::vector_query)): Set both for hybrid search.
///
/// If [`query_dsl`](Self::query_dsl) is called, the builder produces a
/// [`SearchQuery::Dsl`] variant. Otherwise, it determines the variant from
/// which query methods were called.
pub struct SearchRequestBuilder {
    dsl: Option<String>,
    lexical_query: Option<LexicalSearchQuery>,
    vector_query: Option<VectorSearchQuery>,
    limit: usize,
    offset: usize,
    fusion_algorithm: Option<FusionAlgorithm>,
    filter_query: Option<Box<dyn Query>>,
    lexical_options: LexicalSearchOptions,
    vector_options: VectorSearchOptions,
}

impl Default for SearchRequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchRequestBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            dsl: None,
            lexical_query: None,
            vector_query: None,
            limit: 10,
            offset: 0,
            fusion_algorithm: None,
            filter_query: None,
            lexical_options: LexicalSearchOptions::default(),
            vector_options: VectorSearchOptions::default(),
        }
    }

    // ── Query setters ────────────────────────────────────────────────────

    /// Set a unified query DSL string.
    ///
    /// When set, the built request uses [`SearchQuery::Dsl`] and any
    /// lexical/vector queries set via other methods are ignored.
    pub fn query_dsl(mut self, dsl: impl Into<String>) -> Self {
        self.dsl = Some(dsl.into());
        self
    }

    /// Set the lexical search query.
    ///
    /// If [`vector_query`](Self::vector_query) is also set, the result is
    /// [`SearchQuery::Hybrid`]. Otherwise [`SearchQuery::Lexical`].
    pub fn lexical_query(mut self, query: LexicalSearchQuery) -> Self {
        self.lexical_query = Some(query);
        self
    }

    /// Set the vector search query.
    ///
    /// If [`lexical_query`](Self::lexical_query) is also set, the result is
    /// [`SearchQuery::Hybrid`]. Otherwise [`SearchQuery::Vector`].
    pub fn vector_query(mut self, query: VectorSearchQuery) -> Self {
        self.vector_query = Some(query);
        self
    }

    // ── Pagination & fusion ──────────────────────────────────────────────

    /// Set the maximum number of results to return.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the number of results to skip (for pagination).
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Set the fusion algorithm for hybrid search.
    ///
    /// For [`FusionAlgorithm::WeightedSum`], the weights are clamped to
    /// `0.0..=1.0` to prevent NaN/Inf propagation.
    pub fn fusion_algorithm(mut self, fusion: FusionAlgorithm) -> Self {
        let fusion = match fusion {
            FusionAlgorithm::WeightedSum {
                lexical_weight,
                vector_weight,
            } => FusionAlgorithm::WeightedSum {
                lexical_weight: lexical_weight.clamp(0.0, 1.0),
                vector_weight: vector_weight.clamp(0.0, 1.0),
            },
            other => other,
        };
        self.fusion_algorithm = Some(fusion);
        self
    }

    /// Set a filter query to restrict the search space.
    ///
    /// The filter applies to **both** lexical and vector searches.
    pub fn filter_query(mut self, query: Box<dyn Query>) -> Self {
        self.filter_query = Some(query);
        self
    }

    // ── Lexical options ──────────────────────────────────────────────────

    /// Add a field-level boost for lexical search.
    pub fn add_field_boost(mut self, field: impl Into<String>, boost: f32) -> Self {
        self.lexical_options
            .field_boosts
            .insert(field.into(), boost);
        self
    }

    /// Set the minimum score threshold for lexical search.
    pub fn lexical_min_score(mut self, min_score: f32) -> Self {
        self.lexical_options.min_score = min_score;
        self
    }

    /// Set the timeout for lexical search in milliseconds.
    pub fn lexical_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.lexical_options.timeout_ms = Some(timeout_ms);
        self
    }

    /// Enable or disable parallel lexical search.
    pub fn lexical_parallel(mut self, parallel: bool) -> Self {
        self.lexical_options.parallel = parallel;
        self
    }

    /// Set the sort order for lexical search results.
    pub fn sort_by(mut self, sort_by: SortField) -> Self {
        self.lexical_options.sort_by = sort_by;
        self
    }

    // ── Vector options ───────────────────────────────────────────────────

    /// Set the score combination mode for vector search.
    pub fn vector_score_mode(mut self, score_mode: VectorScoreMode) -> Self {
        self.vector_options.score_mode = score_mode;
        self
    }

    /// Set the minimum score threshold for vector search.
    pub fn vector_min_score(mut self, min_score: f32) -> Self {
        self.vector_options.min_score = min_score;
        self
    }

    /// Set the Stage 2 rerank factor (Issue #481) for vector search.
    ///
    /// Honored by HNSW fields whose schema enabled `rerank_storage`.
    /// Other vector configurations silently ignore the value.
    pub fn vector_rerank_factor(mut self, factor: usize) -> Self {
        self.vector_options.rerank_factor = Some(factor);
        self
    }

    /// Override the HNSW `ef_search` candidate-list size for this
    /// search (Issue [#644](https://github.com/mosuka/laurus/issues/644)).
    ///
    /// When unset, the searcher uses the schema-level
    /// [`HnswOption::default_ef_search`](crate::vector::core::field::HnswOption::default_ef_search)
    /// or its internal fallback (`50`). Per-query overrides always
    /// take precedence. Ignored by non-HNSW index types.
    pub fn vector_ef_search(mut self, ef: usize) -> Self {
        self.vector_options.ef_search = Some(ef);
        self
    }

    // ── Build ────────────────────────────────────────────────────────────

    /// Consume the builder and return the constructed [`SearchRequest`].
    pub fn build(self) -> SearchRequest {
        let query = if let Some(dsl) = self.dsl {
            SearchQuery::Dsl(dsl)
        } else {
            match (self.lexical_query, self.vector_query) {
                (Some(lexical), Some(vector)) => SearchQuery::Hybrid {
                    lexical,
                    vector,
                    mode: HybridMode::default(),
                },
                (Some(lexical), None) => SearchQuery::Lexical(lexical),
                (None, Some(vector)) => SearchQuery::Vector(vector),
                (None, None) => SearchQuery::Dsl(String::new()),
            }
        };

        SearchRequest {
            query,
            limit: self.limit,
            offset: self.offset,
            fusion_algorithm: self.fusion_algorithm,
            filter_query: self.filter_query,
            lexical_options: self.lexical_options,
            vector_options: self.vector_options,
        }
    }
}

// ── SearchResult ─────────────────────────────────────────────────────────────

/// A single result from an [`Engine`](super::Engine) search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// External document ID (the `_id` field value).
    pub id: String,
    /// Relevance score. The meaning depends on the search mode:
    /// - Lexical only: BM25 score.
    /// - Vector only: similarity score (e.g. cosine similarity).
    /// - Hybrid: fused score produced by the [`FusionAlgorithm`].
    pub score: f32,
    /// The stored fields of the document, or `None` if the document could
    /// not be retrieved (e.g. it was deleted between scoring and retrieval).
    pub document: Option<crate::data::Document>,
}
