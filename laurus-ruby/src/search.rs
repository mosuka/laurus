//! Ruby wrappers for search request/result and fusion algorithm types.

use std::collections::HashMap;

use crate::convert::document_to_hash;
use crate::query::{
    extract_lexical_query, is_vector_query, rb_to_lexical_search_query, rb_to_vector_search_query,
};
use laurus::{
    Document, FusionAlgorithm, HighlightConfig, HighlightOptions, LexicalSearchQuery,
    SearchRequestBuilder, SearchResult, VectorSearchQuery,
};
use magnus::prelude::*;
use magnus::scan_args::{get_kwargs, scan_args};
use magnus::{Error, RArray, RHash, RModule, Ruby, TryConvert, Value};

// ---------------------------------------------------------------------------
// Highlighting (Issue #1134)
// ---------------------------------------------------------------------------

/// Convert a Ruby `highlight:` value into [`HighlightOptions`].
///
/// Accepts either an `Array` of field names (`["title", "body"]`), or a
/// `Hash` with a `:fields`/`"fields"` key plus any of the optional
/// [`HighlightConfig`] knobs: `:max_fragments`, `:fragment_size`, `:tag`,
/// `:css_class`, `:require_field_match`, `:max_analyzed_chars`,
/// `:return_entire_field_if_no_highlight` (String or Symbol keys, both
/// accepted). The `Array` check comes first — a `Hash`'s default iteration
/// also yields entries in a way that could otherwise be mistaken for a
/// list, so type-sniff before assuming either shape.
pub fn rb_to_highlight_options(ruby: &Ruby, value: Value) -> Result<HighlightOptions, Error> {
    if value.is_kind_of(ruby.class_array()) {
        let arr = RArray::from_value(value)
            .ok_or_else(|| Error::new(ruby.exception_type_error(), "expected an Array"))?;
        let fields: Vec<String> = arr.to_vec()?;
        return Ok(HighlightOptions::new(fields));
    }

    if value.is_kind_of(ruby.class_hash()) {
        let hash = RHash::from_value(value)
            .ok_or_else(|| Error::new(ruby.exception_type_error(), "expected a Hash"))?;

        let fields_val = hash_get_str_or_sym(ruby, hash, "fields")?.ok_or_else(|| {
            Error::new(
                ruby.exception_arg_error(),
                "highlight Hash must have a 'fields' key",
            )
        })?;
        let fields: Vec<String> = RArray::try_convert(fields_val)?.to_vec()?;

        let mut config = HighlightConfig::new();
        if let Some(v) = hash_get_str_or_sym(ruby, hash, "max_fragments")? {
            config = config.max_fragments(usize::try_convert(v)?);
        }
        if let Some(v) = hash_get_str_or_sym(ruby, hash, "fragment_size")? {
            config = config.fragment_size(usize::try_convert(v)?);
        }
        if let Some(v) = hash_get_str_or_sym(ruby, hash, "tag")? {
            config = config.tag(String::try_convert(v)?);
        }
        if let Some(v) = hash_get_str_or_sym(ruby, hash, "css_class")? {
            config = config.css_class(String::try_convert(v)?);
        }
        if let Some(v) = hash_get_str_or_sym(ruby, hash, "require_field_match")? {
            config = config.require_field_match(bool::try_convert(v)?);
        }
        if let Some(v) = hash_get_str_or_sym(ruby, hash, "max_analyzed_chars")? {
            config.max_analyzed_chars = usize::try_convert(v)?;
        }
        if let Some(v) = hash_get_str_or_sym(ruby, hash, "return_entire_field_if_no_highlight")? {
            config.return_entire_field_if_no_highlight = bool::try_convert(v)?;
        }

        return Ok(HighlightOptions::new(fields).with_config(config));
    }

    Err(Error::new(
        ruby.exception_arg_error(),
        "highlight must be an Array of field names or a Hash, \
         e.g. ['body'] or {fields: ['body'], tag: 'em'}",
    ))
}

/// Look up `key` in `hash`, trying the Symbol form first (the natural shape
/// of a Ruby kwargs-style Hash literal) and falling back to the String form.
fn hash_get_str_or_sym(ruby: &Ruby, hash: RHash, key: &str) -> Result<Option<Value>, Error> {
    if let Some(v) = hash.get(ruby.to_symbol(key)) {
        return Ok(Some(v));
    }
    Ok(hash.get(ruby.str_new(key)))
}

// ---------------------------------------------------------------------------
// Fusion algorithm types
// ---------------------------------------------------------------------------

/// Reciprocal Rank Fusion — rank-based result merging for hybrid search
/// (`Laurus::RRF`).
#[magnus::wrap(class = "Laurus::RRF")]
#[derive(Clone)]
pub struct RbRRF {
    pub k: f64,
}

impl RbRRF {
    /// Create a new RRF fusion algorithm.
    ///
    /// # Arguments
    ///
    /// * `args` - Keyword arguments:
    ///   - `k:` (f64, default 60.0): RRF constant.
    fn new(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(), (), (), (), RHash, ()>(args)?;
        let kwargs = get_kwargs::<_, (), (Option<f64>,), ()>(args.keywords, &[], &["k"])?;
        let (k,) = kwargs.optional;
        Ok(Self {
            k: k.unwrap_or(60.0),
        })
    }

    fn inspect(&self) -> String {
        format!("RRF(k={})", self.k)
    }
}

/// Weighted sum fusion — normalises lexical and vector scores then combines them
/// (`Laurus::WeightedSum`).
#[magnus::wrap(class = "Laurus::WeightedSum")]
#[derive(Clone)]
pub struct RbWeightedSum {
    pub lexical_weight: f32,
    pub vector_weight: f32,
}

impl RbWeightedSum {
    /// Create a new weighted sum fusion algorithm.
    ///
    /// # Arguments
    ///
    /// * `args` - Keyword arguments:
    ///   - `lexical_weight:` (f32, default 0.5): Weight for lexical scores.
    ///   - `vector_weight:` (f32, default 0.5): Weight for vector scores.
    fn new(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(), (), (), (), RHash, ()>(args)?;
        let kwargs = get_kwargs::<_, (), (Option<f32>, Option<f32>), ()>(
            args.keywords,
            &[],
            &["lexical_weight", "vector_weight"],
        )?;
        let (lexical_weight, vector_weight) = kwargs.optional;
        Ok(Self {
            lexical_weight: lexical_weight.unwrap_or(0.5),
            vector_weight: vector_weight.unwrap_or(0.5),
        })
    }

    fn inspect(&self) -> String {
        format!(
            "WeightedSum(lexical_weight={}, vector_weight={})",
            self.lexical_weight, self.vector_weight
        )
    }
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// A single search result returned by `Index#search` (`Laurus::SearchResult`).
///
/// Attributes:
///   - `id` (String): External document identifier.
///   - `score` (Float): Relevance score (BM25, similarity, or fused).
///   - `document` (Hash | nil): Retrieved document fields.
///   - `highlights` (Hash): Highlighted fragments per field requested via
///     `highlight:`, best fragment first. Empty when highlighting was not
///     requested, the field was not a stored text field, or nothing matched.
#[magnus::wrap(class = "Laurus::SearchResult")]
pub struct RbSearchResult {
    pub id: String,
    pub score: f32,
    /// Stores the Rust Document to avoid Send issues with Ruby RHash values.
    pub document: Option<Document>,
    pub highlights: HashMap<String, Vec<String>>,
}

impl RbSearchResult {
    /// Return the external document identifier.
    fn id(&self) -> String {
        self.id.clone()
    }

    /// Return the relevance score.
    fn score(&self) -> f32 {
        self.score
    }

    /// Return the document fields as a Hash, or nil if the document was deleted.
    fn document(&self) -> Result<Value, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        match &self.document {
            Some(doc) => Ok(document_to_hash(&ruby, doc)?.as_value()),
            None => Ok(ruby.qnil().as_value()),
        }
    }

    /// Return the highlighted fragments as a Hash of field name -> Array of
    /// fragment strings.
    fn highlights(&self) -> Result<RHash, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let hash = ruby.hash_new();
        for (field, fragments) in &self.highlights {
            hash.aset(ruby.str_new(field), fragments.clone())?;
        }
        Ok(hash)
    }

    fn inspect(&self) -> String {
        format!("SearchResult(id='{}', score={:.4})", self.id, self.score)
    }
}

/// Convert a [`SearchResult`] from the engine into a [`RbSearchResult`].
///
/// # Arguments
///
/// * `r` - Engine search result.
///
/// # Returns
///
/// A Ruby-wrapped search result.
pub fn to_rb_search_result(r: SearchResult) -> RbSearchResult {
    RbSearchResult {
        id: r.id,
        score: r.score,
        document: r.document,
        highlights: r.highlights,
    }
}

// ---------------------------------------------------------------------------
// SearchRequest - stores Rust types only (no Ruby Values)
// ---------------------------------------------------------------------------

/// Encapsulated query representation that is Send-safe (no Ruby Values).
enum QueryRepr {
    /// A DSL string.
    Dsl(String),
    /// A lexical query.
    Lexical(LexicalSearchQuery),
    /// A vector query.
    Vector(VectorSearchQuery),
}

/// Full-featured search request for advanced control over query, fusion, and
/// filtering (`Laurus::SearchRequest`).
#[magnus::wrap(class = "Laurus::SearchRequest")]
pub struct RbSearchRequest {
    /// A DSL string, or any single lexical/vector query object.
    query: Option<QueryRepr>,
    /// Lexical component for explicit hybrid search.
    lexical_query: Option<LexicalSearchQuery>,
    /// Vector component for explicit hybrid search.
    vector_query: Option<VectorSearchQuery>,
    /// Optional lexical filter query applied after scoring.
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

impl RbSearchRequest {
    /// Create a new search request.
    ///
    /// All arguments are keyword-only:
    ///
    /// # Arguments
    ///
    /// * `args` - Keyword arguments:
    ///   - `query:` - DSL string or query object (mutually exclusive with lexical/vector_query).
    ///   - `lexical_query:` - Lexical query for hybrid search.
    ///   - `vector_query:` - Vector query for hybrid search.
    ///   - `filter_query:` - Post-scoring filter query.
    ///   - `fusion:` - `RRF` or `WeightedSum` fusion algorithm.
    ///   - `limit:` (usize, default 10): Maximum results.
    ///   - `offset:` (usize, default 0): Pagination offset.
    ///   - `highlight:` - Field list (Array) or config Hash for
    ///     search-result highlighting (Issue #1134).
    fn new(args: &[Value]) -> Result<Self, Error> {
        let ruby = Ruby::get().expect("called from Ruby thread");
        let args = scan_args::<(), (), (), (), RHash, ()>(args)?;
        let kwargs = get_kwargs::<
            _,
            (),
            (
                Option<Value>,
                Option<Value>,
                Option<Value>,
                Option<Value>,
                Option<Value>,
                Option<usize>,
                Option<usize>,
                Option<Value>,
            ),
            (),
        >(
            args.keywords,
            &[],
            &[
                "query",
                "lexical_query",
                "vector_query",
                "filter_query",
                "fusion",
                "limit",
                "offset",
                "highlight",
            ],
        )?;
        let (
            query_val,
            lexical_query_val,
            vector_query_val,
            filter_query_val,
            fusion_val,
            limit,
            offset,
            highlight_val,
        ) = kwargs.optional;

        let highlight = highlight_val
            .map(|v| rb_to_highlight_options(&ruby, v))
            .transpose()?;

        // Convert fusion
        let fusion = if let Some(f) = fusion_val {
            if let Ok(rrf) = <&RbRRF>::try_convert(f) {
                Some(FusionAlgorithm::RRF { k: rrf.k })
            } else if let Ok(ws) = <&RbWeightedSum>::try_convert(f) {
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
        let filter_query = filter_query_val.map(extract_lexical_query).transpose()?;

        // Convert lexical query
        let lexical_query = lexical_query_val
            .map(rb_to_lexical_search_query)
            .transpose()?;

        // Convert vector query
        let vector_query = vector_query_val
            .map(rb_to_vector_search_query)
            .transpose()?;

        // Convert single query
        let query = if let Some(q) = query_val {
            if let Ok(s) = String::try_convert(q) {
                Some(QueryRepr::Dsl(s))
            } else if is_vector_query(q) {
                Some(QueryRepr::Vector(rb_to_vector_search_query(q)?))
            } else {
                Some(QueryRepr::Lexical(rb_to_lexical_search_query(q)?))
            }
        } else {
            None
        };

        Ok(Self {
            query,
            lexical_query,
            vector_query,
            filter_query,
            fusion,
            limit: limit.unwrap_or(10),
            offset: offset.unwrap_or(0),
            highlight,
        })
    }

    fn inspect(&self) -> String {
        format!(
            "SearchRequest(limit={}, offset={})",
            self.limit, self.offset
        )
    }
}

impl RbSearchRequest {
    /// Build the Laurus [`laurus::SearchRequest`] from this Ruby wrapper.
    pub fn build(&self) -> Result<laurus::SearchRequest, Error> {
        let mut builder = SearchRequestBuilder::new()
            .limit(self.limit)
            .offset(self.offset);

        // Fusion algorithm
        if let Some(ref fusion) = self.fusion {
            builder = builder.fusion_algorithm(*fusion);
        }

        // Filter query - we cannot move out of &self, so we need to handle this differently.
        // Since build() takes &self, we cannot consume filter_query. This is a design issue.
        // For now, we skip filter in the &self case. The actual search path uses build_request_from_rb.

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
// Helper: build a SearchRequest from `Index#search` arguments
// ---------------------------------------------------------------------------

/// Build a [`laurus::SearchRequest`] from the arguments passed to
/// `Index#search(query, limit:, offset:, highlight:)`.
///
/// `query` may be:
/// - A `String` (DSL)
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
/// * `query` - Ruby value for the query.
/// * `limit` - Maximum results.
/// * `offset` - Pagination offset.
/// * `highlight` - Already-parsed highlight options, if `highlight:` was given.
///
/// # Returns
///
/// A `laurus::SearchRequest`.
pub fn build_request_from_rb(
    query: Value,
    limit: usize,
    offset: usize,
    highlight: Option<&HighlightOptions>,
) -> Result<laurus::SearchRequest, Error> {
    // Full SearchRequest object
    if let Ok(req) = <&RbSearchRequest>::try_convert(query) {
        return req.build();
    }

    let mut builder = SearchRequestBuilder::new().limit(limit).offset(offset);

    if let Some(options) = highlight {
        builder = builder
            .highlight(options.fields.clone())
            .highlight_config(options.config.clone());
    }

    // DSL string
    if let Ok(s) = String::try_convert(query) {
        builder = builder.query_dsl(s);
        return Ok(builder.build());
    }

    // Vector queries
    if is_vector_query(query) {
        builder = builder.vector_query(rb_to_vector_search_query(query)?);
        return Ok(builder.build());
    }

    // Lexical queries
    builder = builder.lexical_query(LexicalSearchQuery::Obj(extract_lexical_query(query)?));
    Ok(builder.build())
}

// ---------------------------------------------------------------------------
// Class registration
// ---------------------------------------------------------------------------

/// Register search-related classes under the `Laurus` module.
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `module` - The `Laurus` module.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    // RRF
    let rrf = module.define_class("RRF", ruby.class_object())?;
    rrf.define_singleton_method("new", magnus::function!(RbRRF::new, -1))?;
    rrf.define_method("inspect", magnus::method!(RbRRF::inspect, 0))?;
    rrf.define_method("to_s", magnus::method!(RbRRF::inspect, 0))?;

    // WeightedSum
    let ws = module.define_class("WeightedSum", ruby.class_object())?;
    ws.define_singleton_method("new", magnus::function!(RbWeightedSum::new, -1))?;
    ws.define_method("inspect", magnus::method!(RbWeightedSum::inspect, 0))?;
    ws.define_method("to_s", magnus::method!(RbWeightedSum::inspect, 0))?;

    // SearchResult
    let sr = module.define_class("SearchResult", ruby.class_object())?;
    sr.define_method("id", magnus::method!(RbSearchResult::id, 0))?;
    sr.define_method("score", magnus::method!(RbSearchResult::score, 0))?;
    sr.define_method("document", magnus::method!(RbSearchResult::document, 0))?;
    sr.define_method("highlights", magnus::method!(RbSearchResult::highlights, 0))?;
    sr.define_method("inspect", magnus::method!(RbSearchResult::inspect, 0))?;
    sr.define_method("to_s", magnus::method!(RbSearchResult::inspect, 0))?;

    // SearchRequest
    let sreq = module.define_class("SearchRequest", ruby.class_object())?;
    sreq.define_singleton_method("new", magnus::function!(RbSearchRequest::new, -1))?;
    sreq.define_method("inspect", magnus::method!(RbSearchRequest::inspect, 0))?;
    sreq.define_method("to_s", magnus::method!(RbSearchRequest::inspect, 0))?;

    Ok(())
}
