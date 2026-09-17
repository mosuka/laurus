//! PHP wrappers for search request/result and fusion algorithm types.

use std::collections::HashMap;

use ext_php_rs::convert::FromZval;
use ext_php_rs::prelude::*;
use ext_php_rs::types::{ZendClassObject, ZendHashTable, Zval};
use laurus::{
    Document, FusionAlgorithm, HighlightConfig, HighlightOptions, LexicalSearchQuery,
    SearchRequestBuilder, SearchResult, VectorSearchQuery,
};

use crate::convert::document_to_hashtable;
use crate::query::{
    extract_lexical_query, is_vector_query, zval_to_lexical_search_query,
    zval_to_vector_search_query,
};

// ---------------------------------------------------------------------------
// Highlighting (Issue #1134)
// ---------------------------------------------------------------------------

/// Convert a PHP `?array $highlight` argument into [`HighlightOptions`].
///
/// Accepts either a plain list of field names (`["title", "body"]`), or an
/// associative array with a `"fields"` key plus any of the optional
/// [`HighlightConfig`] knobs: `"max_fragments"`, `"fragment_size"`,
/// `"tag"`, `"css_class"`, `"require_field_match"`, `"max_analyzed_chars"`,
/// `"return_entire_field_if_no_highlight"`. PHP has no separate list/map
/// array type, so the `"fields"` key is checked first — its presence is
/// what selects the associative form.
pub fn parse_highlight_option(
    highlight: Option<&ZendHashTable>,
) -> PhpResult<Option<HighlightOptions>> {
    highlight.map(php_array_to_highlight_options).transpose()
}

fn php_array_to_highlight_options(arr: &ZendHashTable) -> PhpResult<HighlightOptions> {
    if arr.get("fields").is_some() {
        let fields = ht_get_vec_string(arr, "fields")?.ok_or_else(|| {
            PhpException::from("highlight['fields'] must be an array of strings".to_string())
        })?;

        let mut config = HighlightConfig::new();
        if let Some(v) = ht_get_usize(arr, "max_fragments")? {
            config = config.max_fragments(v);
        }
        if let Some(v) = ht_get_usize(arr, "fragment_size")? {
            config = config.fragment_size(v);
        }
        if let Some(v) = ht_get_string_opt(arr, "tag")? {
            config = config.tag(v);
        }
        if let Some(v) = ht_get_string_opt(arr, "css_class")? {
            config = config.css_class(v);
        }
        if let Some(v) = ht_get_bool(arr, "require_field_match")? {
            config = config.require_field_match(v);
        }
        if let Some(v) = ht_get_usize(arr, "max_analyzed_chars")? {
            config.max_analyzed_chars = v;
        }
        if let Some(v) = ht_get_bool(arr, "return_entire_field_if_no_highlight")? {
            config.return_entire_field_if_no_highlight = v;
        }

        return Ok(HighlightOptions::new(fields).with_config(config));
    }

    // No "fields" key: treat the whole array as a plain list of field
    // name strings, e.g. ["title", "body"].
    let fields: Vec<String> = arr
        .values()
        .map(|zv| {
            String::from_zval(zv).ok_or_else(|| {
                PhpException::from(
                    "highlight must be a list of field names or an associative array, \
                     e.g. ['body'] or ['fields' => ['body'], 'tag' => 'em']"
                        .to_string(),
                )
            })
        })
        .collect::<PhpResult<Vec<_>>>()?;
    Ok(HighlightOptions::new(fields))
}

fn ht_get_vec_string(ht: &ZendHashTable, key: &str) -> PhpResult<Option<Vec<String>>> {
    let Some(zv) = ht.get(key) else {
        return Ok(None);
    };
    Vec::from_zval(zv)
        .map(Some)
        .ok_or_else(|| format!("'{key}' must be an array of strings").into())
}

fn ht_get_string_opt(ht: &ZendHashTable, key: &str) -> PhpResult<Option<String>> {
    let Some(zv) = ht.get(key) else {
        return Ok(None);
    };
    String::from_zval(zv)
        .map(Some)
        .ok_or_else(|| format!("'{key}' must be a string").into())
}

fn ht_get_usize(ht: &ZendHashTable, key: &str) -> PhpResult<Option<usize>> {
    let Some(zv) = ht.get(key) else {
        return Ok(None);
    };
    i64::from_zval(zv)
        .map(|n| n as usize)
        .map(Some)
        .ok_or_else(|| format!("'{key}' must be an integer").into())
}

fn ht_get_bool(ht: &ZendHashTable, key: &str) -> PhpResult<Option<bool>> {
    let Some(zv) = ht.get(key) else {
        return Ok(None);
    };
    bool::from_zval(zv)
        .map(Some)
        .ok_or_else(|| format!("'{key}' must be a bool").into())
}

// ---------------------------------------------------------------------------
// Fusion algorithm types
// ---------------------------------------------------------------------------

/// Reciprocal Rank Fusion — rank-based result merging for hybrid search
/// (`Laurus\RRF`).
#[php_class]
#[php(name = "Laurus\\RRF")]
#[derive(Clone)]
pub struct PhpRRF {
    pub k: f64,
}

#[php_impl]
impl PhpRRF {
    /// Create a new RRF fusion algorithm.
    ///
    /// # Arguments
    ///
    /// * `k` - RRF constant (default: 60.0).
    #[php(defaults(k = 60.0))]
    pub fn __construct(k: f64) -> Self {
        Self { k }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!("RRF(k={})", self.k)
    }
}

/// Weighted sum fusion — normalises lexical and vector scores then combines them
/// (`Laurus\WeightedSum`).
#[php_class]
#[php(name = "Laurus\\WeightedSum")]
#[derive(Clone)]
pub struct PhpWeightedSum {
    pub lexical_weight: f32,
    pub vector_weight: f32,
}

#[php_impl]
impl PhpWeightedSum {
    /// Create a new weighted sum fusion algorithm.
    ///
    /// # Arguments
    ///
    /// * `lexical_weight` - Weight for lexical scores (default: 0.5).
    /// * `vector_weight` - Weight for vector scores (default: 0.5).
    #[php(defaults(lexical_weight = 0.5, vector_weight = 0.5))]
    pub fn __construct(lexical_weight: f64, vector_weight: f64) -> Self {
        Self {
            lexical_weight: lexical_weight as f32,
            vector_weight: vector_weight as f32,
        }
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "WeightedSum(lexical_weight={}, vector_weight={})",
            self.lexical_weight, self.vector_weight
        )
    }
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// A single search result returned by `Index->search()` (`Laurus\SearchResult`).
///
/// Properties:
///   - `id` (string): External document identifier.
///   - `score` (float): Relevance score (BM25, similarity, or fused).
///   - `document` (array|null): Retrieved document fields.
///   - `highlights` (array): Highlighted fragments per field requested via
///     `$highlight`, best fragment first. Empty when highlighting was not
///     requested, the field was not a stored text field, or nothing matched.
#[php_class]
#[php(name = "Laurus\\SearchResult")]
pub struct PhpSearchResult {
    id: String,
    score: f32,
    /// Stores the Rust Document to avoid serialization issues.
    document: Option<Document>,
    highlights: HashMap<String, Vec<String>>,
}

#[php_impl]
impl PhpSearchResult {
    /// Return the external document identifier.
    pub fn get_id(&self) -> String {
        self.id.clone()
    }

    /// Return the relevance score.
    pub fn get_score(&self) -> f64 {
        self.score as f64
    }

    /// Return the document fields as an associative array, or null.
    pub fn get_document(&self) -> PhpResult<Zval> {
        match &self.document {
            Some(doc) => {
                let ht = document_to_hashtable(doc)?;
                let mut zv = Zval::new();
                zv.set_hashtable(ht);
                Ok(zv)
            }
            None => {
                let mut zv = Zval::new();
                zv.set_null();
                Ok(zv)
            }
        }
    }

    /// Return the highlighted fragments as an associative array of field
    /// name -> array of fragment strings.
    pub fn get_highlights(&self) -> PhpResult<HashMap<String, Vec<String>>> {
        Ok(self.highlights.clone())
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!("SearchResult(id='{}', score={:.4})", self.id, self.score)
    }
}

/// Convert a [`SearchResult`] from the engine into a [`PhpSearchResult`].
///
/// # Arguments
///
/// * `r` - Engine search result.
///
/// # Returns
///
/// A PHP-wrapped search result.
pub fn to_php_search_result(r: SearchResult) -> PhpSearchResult {
    PhpSearchResult {
        id: r.id,
        score: r.score,
        document: r.document,
        highlights: r.highlights,
    }
}

// ---------------------------------------------------------------------------
// SearchRequest
// ---------------------------------------------------------------------------

/// Encapsulated query representation that is Send-safe (no PHP Zvals).
enum QueryRepr {
    /// A DSL string.
    Dsl(String),
    /// A lexical query.
    Lexical(LexicalSearchQuery),
    /// A vector query.
    Vector(VectorSearchQuery),
}

/// Full-featured search request for advanced control over query, fusion, and
/// filtering (`Laurus\SearchRequest`).
#[php_class]
#[php(name = "Laurus\\SearchRequest")]
pub struct PhpSearchRequest {
    /// A DSL string, or any single lexical/vector query object.
    query: Option<QueryRepr>,
    /// Lexical component for explicit hybrid search.
    lexical_query: Option<LexicalSearchQuery>,
    /// Vector component for explicit hybrid search.
    vector_query: Option<VectorSearchQuery>,
    /// Optional lexical filter query applied after scoring.
    #[allow(dead_code)]
    filter_query: Option<Box<dyn laurus::lexical::Query>>,
    /// Fusion algorithm for hybrid results.
    fusion: Option<FusionAlgorithm>,
    /// Maximum number of results.
    limit: usize,
    /// Pagination offset.
    offset: usize,
    /// Highlight request (Issue #1134).
    highlight: Option<HighlightOptions>,
}

#[php_impl]
impl PhpSearchRequest {
    /// Create a new search request.
    ///
    /// All arguments are optional:
    ///
    /// # Arguments
    ///
    /// * `query` - DSL string or query object (mutually exclusive with lexical/vector_query).
    /// * `lexical_query` - Lexical query for hybrid search.
    /// * `vector_query` - Vector query for hybrid search.
    /// * `filter_query` - Post-scoring filter query.
    /// * `fusion` - `RRF` or `WeightedSum` fusion algorithm.
    /// * `limit` - Maximum results (default: 10).
    /// * `offset` - Pagination offset (default: 0).
    /// * `highlight` - Field list or config array for search-result
    ///   highlighting (Issue #1134).
    #[php(defaults(limit = 10, offset = 0))]
    #[allow(clippy::too_many_arguments)]
    pub fn __construct(
        query: &Zval,
        lexical_query: &Zval,
        vector_query: &Zval,
        filter_query: &Zval,
        fusion: &Zval,
        limit: i64,
        offset: i64,
        highlight: Option<&ZendHashTable>,
    ) -> PhpResult<Self> {
        // Convert fusion
        let fusion_alg = if !fusion.is_null() {
            if let Some(rrf_obj) = <&ZendClassObject<PhpRRF>>::from_zval(fusion) {
                let rrf: &PhpRRF = rrf_obj;
                Some(FusionAlgorithm::RRF { k: rrf.k })
            } else if let Some(ws_obj) = <&ZendClassObject<PhpWeightedSum>>::from_zval(fusion) {
                let ws: &PhpWeightedSum = ws_obj;
                Some(FusionAlgorithm::WeightedSum {
                    lexical_weight: ws.lexical_weight,
                    vector_weight: ws.vector_weight,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Convert filter query
        let filter = if !filter_query.is_null() {
            Some(extract_lexical_query(filter_query)?)
        } else {
            None
        };

        // Convert lexical query
        let lex_q = if !lexical_query.is_null() {
            Some(zval_to_lexical_search_query(lexical_query)?)
        } else {
            None
        };

        // Convert vector query
        let vec_q = if !vector_query.is_null() {
            Some(zval_to_vector_search_query(vector_query)?)
        } else {
            None
        };

        // Convert single query
        let q = if !query.is_null() {
            if let Some(s) = String::from_zval(query) {
                Some(QueryRepr::Dsl(s))
            } else if is_vector_query(query) {
                Some(QueryRepr::Vector(zval_to_vector_search_query(query)?))
            } else {
                Some(QueryRepr::Lexical(zval_to_lexical_search_query(query)?))
            }
        } else {
            None
        };

        let highlight = parse_highlight_option(highlight)?;

        Ok(Self {
            query: q,
            lexical_query: lex_q,
            vector_query: vec_q,
            filter_query: filter,
            fusion: fusion_alg,
            limit: limit as usize,
            offset: offset as usize,
            highlight,
        })
    }

    /// Return a string representation.
    pub fn __to_string(&self) -> String {
        format!(
            "SearchRequest(limit={}, offset={})",
            self.limit, self.offset
        )
    }
}

impl PhpSearchRequest {
    /// Build the Laurus [`laurus::SearchRequest`] from this PHP wrapper.
    pub fn build(&self) -> PhpResult<laurus::SearchRequest> {
        let mut builder = SearchRequestBuilder::new()
            .limit(self.limit)
            .offset(self.offset);

        // Fusion algorithm
        if let Some(ref fusion) = self.fusion {
            builder = builder.fusion_algorithm(*fusion);
        }

        // Highlighting (Issue #1134). Applied before every branch below
        // returns, so it takes effect regardless of which query shape is set.
        if let Some(options) = &self.highlight {
            builder = builder
                .highlight(options.fields.clone())
                .highlight_config(options.config.clone());
        }

        // Explicit hybrid: lexical_query + vector_query both set
        if let (Some(lq), Some(vq)) = (&self.lexical_query, &self.vector_query) {
            builder = builder.lexical_query(lq.clone()).vector_query(vq.clone());
            if self.fusion.is_none() {
                builder = builder.fusion_algorithm(FusionAlgorithm::RRF { k: 60.0 });
            }
            return Ok(builder.build());
        }

        // Only lexical_query set
        if let Some(ref lq) = self.lexical_query {
            builder = builder.lexical_query(lq.clone());
            return Ok(builder.build());
        }

        // Only vector_query set
        if let Some(ref vq) = self.vector_query {
            builder = builder.vector_query(vq.clone());
            return Ok(builder.build());
        }

        // Single `query` field: DSL string, lexical, or vector
        if let Some(ref q) = self.query {
            match q {
                QueryRepr::Dsl(s) => {
                    builder = builder.query_dsl(s.clone());
                }
                QueryRepr::Vector(vq) => {
                    builder = builder.vector_query(vq.clone());
                }
                QueryRepr::Lexical(lq) => {
                    builder = builder.lexical_query(lq.clone());
                }
            }
        }

        Ok(builder.build())
    }
}

// ---------------------------------------------------------------------------
// Helper: build a SearchRequest from `Index->search()` arguments
// ---------------------------------------------------------------------------

/// Build a [`laurus::SearchRequest`] from the arguments passed to
/// `Index->search($query, $limit, $offset, $highlight)`.
///
/// `query` may be:
/// - A `string` (DSL)
/// - A `SearchRequest` (full request)
/// - Any lexical query class
/// - `VectorQuery` or `VectorTextQuery`
///
/// When `query` is a `SearchRequest`, `limit`/`offset`/`highlight` are used
/// as-is from the request without overriding — same precedent as
/// `limit`/`offset` already had here before highlighting existed.
///
/// # Arguments
///
/// * `query` - PHP Zval for the query.
/// * `limit` - Maximum results.
/// * `offset` - Pagination offset.
/// * `highlight` - Already-parsed highlight options, if `$highlight` was given.
///
/// # Returns
///
/// A `laurus::SearchRequest`.
pub fn build_request_from_php(
    query: &Zval,
    limit: usize,
    offset: usize,
    highlight: Option<&HighlightOptions>,
) -> PhpResult<laurus::SearchRequest> {
    // Full SearchRequest object
    if let Some(req_obj) = <&ZendClassObject<PhpSearchRequest>>::from_zval(query) {
        let req: &PhpSearchRequest = req_obj;
        return req.build();
    }

    let mut builder = SearchRequestBuilder::new().limit(limit).offset(offset);

    if let Some(options) = highlight {
        builder = builder
            .highlight(options.fields.clone())
            .highlight_config(options.config.clone());
    }

    // DSL string
    if let Some(s) = String::from_zval(query) {
        builder = builder.query_dsl(s);
        return Ok(builder.build());
    }

    // Vector queries
    if is_vector_query(query) {
        builder = builder.vector_query(zval_to_vector_search_query(query)?);
        return Ok(builder.build());
    }

    // Lexical queries
    builder = builder.lexical_query(LexicalSearchQuery::Obj(extract_lexical_query(query)?));
    Ok(builder.build())
}
