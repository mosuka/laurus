//! Node.js wrappers for search request/result and fusion algorithm types.

use std::collections::HashMap;

use crate::convert::data_value_to_json;
use crate::query::{
    JsBooleanQuery, JsFuzzyQuery, JsGeo3dBoundingBoxQuery, JsGeo3dDistanceQuery,
    JsGeo3dNearestQuery, JsGeoBoundingBoxQuery, JsGeoDistanceQuery, JsNumericRangeQuery,
    JsPhraseQuery, JsQuery, JsSpanQuery, JsTermQuery, JsVectorQuery, JsVectorQueryInner,
    JsVectorTextQuery, JsWildcardQuery, extract_lexical_query, query_to_lexical_search_query,
    vector_query_to_search_query,
};
use laurus::{
    FusionAlgorithm, HighlightConfig, HighlightOptions, LexicalSearchQuery, SearchRequestBuilder,
    SearchResult,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ---------------------------------------------------------------------------
// Highlighting (Issue #1134)
// ---------------------------------------------------------------------------

/// Options object accepted by `search`/`searchTerm`/`searchBatch`'s
/// `highlight` parameter and by [`JsSearchRequestOptions::highlight`].
///
/// Only `fields` is required; everything else falls back to
/// [`HighlightConfig::default`]. Unlike the polymorphic query/fusion
/// fields on [`JsSearchRequestOptions`], every field here is plain data
/// (no class-instance union), so it needs none of that struct's
/// setter-method workaround.
#[napi(object)]
pub struct JsHighlightOptions {
    /// Stored text fields to highlight.
    pub fields: Vec<String>,
    /// Target fragment length in characters (default 150).
    pub fragment_size: Option<u32>,
    /// Maximum number of fragments per field (default 5).
    pub max_fragments: Option<u32>,
    /// HTML tag to wrap matches (default `"mark"`).
    pub tag: Option<String>,
    /// CSS class added to the tag.
    pub css_class: Option<String>,
    /// Only use query terms that target the highlighted field (default `true`).
    pub require_field_match: Option<bool>,
}

/// Convert [`JsHighlightOptions`] into the core [`HighlightOptions`]. Takes
/// `&JsHighlightOptions` rather than by value since `JsSearchRequest::build`
/// only has `&self`.
pub fn js_highlight_options_to_core(options: &JsHighlightOptions) -> HighlightOptions {
    let mut config = HighlightConfig::new();
    if let Some(fragment_size) = options.fragment_size {
        config = config.fragment_size(fragment_size as usize);
    }
    if let Some(max_fragments) = options.max_fragments {
        config = config.max_fragments(max_fragments as usize);
    }
    if let Some(tag) = &options.tag {
        config = config.tag(tag.clone());
    }
    if let Some(css_class) = &options.css_class {
        config = config.css_class(css_class.clone());
    }
    if let Some(require_field_match) = options.require_field_match {
        config = config.require_field_match(require_field_match);
    }
    HighlightOptions::new(options.fields.clone()).with_config(config)
}

// ---------------------------------------------------------------------------
// Fusion algorithm types
// ---------------------------------------------------------------------------

/// Reciprocal Rank Fusion — rank-based result merging for hybrid search.
///
/// ## Example
///
/// ```javascript
/// const { RRF } = require("laurus-nodejs");
/// const fusion = new RRF(60.0);
/// ```
#[napi(js_name = "RRF")]
pub struct JsRRF {
    pub(crate) k: f64,
}

#[napi]
impl JsRRF {
    /// Create a new RRF fusion algorithm.
    ///
    /// # Arguments
    ///
    /// * `k` - The RRF parameter (default 60.0). Higher values reduce the impact of rank differences.
    #[napi(constructor)]
    pub fn new(k: Option<f64>) -> Self {
        Self {
            k: k.unwrap_or(60.0),
        }
    }
}

/// Weighted sum fusion — normalises lexical and vector scores then combines them.
///
/// ## Example
///
/// ```javascript
/// const { WeightedSum } = require("laurus-nodejs");
/// const fusion = new WeightedSum(0.3, 0.7);
/// ```
#[napi(js_name = "WeightedSum")]
pub struct JsWeightedSum {
    pub(crate) lexical_weight: f64,
    pub(crate) vector_weight: f64,
}

#[napi]
impl JsWeightedSum {
    /// Create a new weighted sum fusion algorithm.
    ///
    /// # Arguments
    ///
    /// * `lexical_weight` - Weight for lexical search scores (default 0.5).
    /// * `vector_weight` - Weight for vector search scores (default 0.5).
    #[napi(constructor)]
    pub fn new(lexical_weight: Option<f64>, vector_weight: Option<f64>) -> Self {
        Self {
            lexical_weight: lexical_weight.unwrap_or(0.5),
            vector_weight: vector_weight.unwrap_or(0.5),
        }
    }
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// A single search result returned by `Index.search()`.
///
/// Properties:
///   - `id` (string): External document identifier.
///   - `score` (number): Relevance score (BM25, similarity, or fused).
///   - `document` (object | null): Retrieved document fields, or `null` if deleted.
///   - `highlights` (object): Highlighted fragments per field requested via
///     `highlight`, best fragment first. Empty when highlighting was not
///     requested, the field was not a stored text field, or nothing matched.
#[napi(object)]
pub struct JsSearchResult {
    /// External document identifier.
    pub id: String,
    /// Relevance score.
    pub score: f64,
    /// Retrieved document fields as a key-value object, or `null`.
    pub document: Option<serde_json::Value>,
    /// Highlighted fragments per field.
    pub highlights: HashMap<String, Vec<String>>,
}

/// Convert a [`SearchResult`] from the engine into a serializable [`JsSearchResult`].
///
/// # Arguments
///
/// * `r` - The engine search result.
///
/// # Returns
///
/// A `JsSearchResult` with document fields converted to JSON.
pub fn to_js_search_result(r: SearchResult) -> JsSearchResult {
    let document = r.document.map(|doc| {
        let mut map = serde_json::Map::new();
        for (field, value) in doc.fields {
            map.insert(field, data_value_to_json(&value));
        }
        serde_json::Value::Object(map)
    });
    JsSearchResult {
        id: r.id,
        score: r.score as f64,
        document,
        highlights: r.highlights,
    }
}

// ---------------------------------------------------------------------------
// SearchRequest
// ---------------------------------------------------------------------------

/// Full-featured search request for advanced control over query, fusion, and
/// filtering.
///
/// ## Example — hybrid search with filter
///
/// ```javascript
/// const { SearchRequest, VectorTextQuery, TermQuery, RRF } = require("laurus-nodejs");
///
/// const request = new SearchRequest({
///     vectorQuery: new VectorTextQuery("text_vec", "type system"),
///     filterQuery: new TermQuery("category", "type-system"),
///     fusion: new RRF(60.0),
///     limit: 3,
/// });
/// const results = await index.search(request);
/// ```
#[napi(js_name = "SearchRequest")]
pub struct JsSearchRequest {
    /// A DSL string query.
    pub(crate) query_dsl: Option<String>,
    /// Lexical query component.
    pub(crate) lexical_query: Option<JsQuery>,
    /// Vector query component.
    pub(crate) vector_query: Option<JsVectorQuery>,
    /// Filter query applied after scoring.
    pub(crate) filter_query: Option<JsQuery>,
    /// Fusion algorithm.
    pub(crate) fusion: Option<FusionChoice>,
    pub(crate) limit: usize,
    pub(crate) offset: usize,
    /// Highlight request (Issue #1134).
    pub(crate) highlight: Option<JsHighlightOptions>,
}

pub enum FusionChoice {
    RRF(f64),
    WeightedSum(f32, f32),
}

/// Options object accepted by [`JsSearchRequest::new`].
///
/// Only the primitive fields (`queryDsl`, `limit`, `offset`) live here.
/// Polymorphic fields (lexical / filter / vector / fusion) are populated
/// after construction via the per-type `setLexicalX` / `setFilterX` /
/// `setVector*` / `setRrfFusion` / `setWeightedSumFusion` methods —
/// napi-rs cannot accept a class-instance union as an `#[napi(object)]`
/// field because the auto-generated `ValidateNapiValue` for `&T` looks
/// up the JS class constructor by the *Rust* struct name instead of the
/// JS name set via `js_name = "..."`, so any polymorphic field on a
/// napi options struct rejects every instance at runtime.
#[napi(object)]
pub struct JsSearchRequestOptions {
    /// Optional query DSL string (e.g. `"title:hello"`).
    pub query_dsl: Option<String>,
    /// Maximum number of results (default 10).
    pub limit: Option<u32>,
    /// Pagination offset (default 0).
    pub offset: Option<u32>,
    /// Request highlighted fragments per field (Issue #1134). Plain data,
    /// so — unlike the polymorphic fields below — it lives here directly
    /// rather than behind a setter.
    pub highlight: Option<JsHighlightOptions>,
}

#[napi]
impl JsSearchRequest {
    /// Create a new search request.
    ///
    /// Accepts an options object containing the primitive fields. Use
    /// the `setLexicalX` / `setFilterX` / `setVector*` /
    /// `setRrfFusion` / `setWeightedSumFusion` setters to attach
    /// query objects after construction.
    ///
    /// ## Example
    ///
    /// ```javascript
    /// const req = new SearchRequest({ queryDsl: "title:rust", limit: 5 });
    ///
    /// // Or build one piece at a time:
    /// const req2 = new SearchRequest({ limit: 5 });
    /// req2.setLexicalTerm(new TermQuery("title", "rust"));
    /// req2.setVectorQuery(new VectorQuery("embedding", [0.1, 0.2]));
    /// req2.setRrfFusion(new RRF(60.0));
    /// ```
    #[napi(constructor)]
    pub fn new(options: Option<JsSearchRequestOptions>) -> Self {
        let options = options.unwrap_or(JsSearchRequestOptions {
            query_dsl: None,
            limit: None,
            offset: None,
            highlight: None,
        });
        Self {
            query_dsl: options.query_dsl,
            lexical_query: None,
            vector_query: None,
            filter_query: None,
            fusion: None,
            limit: options.limit.unwrap_or(10) as usize,
            offset: options.offset.unwrap_or(0) as usize,
            highlight: options.highlight,
        }
    }

    /// Set a DSL string query.
    #[napi]
    pub fn set_query_dsl(&mut self, dsl: String) {
        self.query_dsl = Some(dsl);
    }

    // ── Lexical query setters (one per query type) ──────────────────────
    //
    // Each takes a `&JsXxxQuery` class instance — the same pattern used
    // by `JsBooleanQuery` for the same napi-derive limitation around
    // class-ref unions described above on `JsSearchRequestOptions`.

    /// Set a [`JsTermQuery`] as the lexical clause.
    #[napi]
    pub fn set_lexical_term(&mut self, query: &JsTermQuery) {
        self.lexical_query = Some(JsQuery::TermQuery(query.clone()));
    }

    /// Set a [`JsPhraseQuery`] as the lexical clause.
    #[napi]
    pub fn set_lexical_phrase(&mut self, query: &JsPhraseQuery) {
        self.lexical_query = Some(JsQuery::PhraseQuery(query.clone()));
    }

    /// Set a [`JsFuzzyQuery`] as the lexical clause.
    #[napi]
    pub fn set_lexical_fuzzy(&mut self, query: &JsFuzzyQuery) {
        self.lexical_query = Some(JsQuery::FuzzyQuery(query.clone()));
    }

    /// Set a [`JsWildcardQuery`] as the lexical clause.
    #[napi]
    pub fn set_lexical_wildcard(&mut self, query: &JsWildcardQuery) {
        self.lexical_query = Some(JsQuery::WildcardQuery(query.clone()));
    }

    /// Set a [`JsNumericRangeQuery`] as the lexical clause.
    #[napi]
    pub fn set_lexical_numeric_range(&mut self, query: &JsNumericRangeQuery) {
        self.lexical_query = Some(JsQuery::NumericRangeQuery(query.clone()));
    }

    /// Set a [`JsGeoDistanceQuery`] as the lexical clause.
    #[napi]
    pub fn set_lexical_geo_distance(&mut self, query: &JsGeoDistanceQuery) {
        self.lexical_query = Some(JsQuery::GeoDistanceQuery(query.clone()));
    }

    /// Set a [`JsGeoBoundingBoxQuery`] as the lexical clause.
    #[napi]
    pub fn set_lexical_geo_bounding_box(&mut self, query: &JsGeoBoundingBoxQuery) {
        self.lexical_query = Some(JsQuery::GeoBoundingBoxQuery(query.clone()));
    }

    /// Set a [`JsGeo3dDistanceQuery`] as the lexical clause.
    #[napi(js_name = "setLexicalGeo3dDistance")]
    pub fn set_lexical_geo3d_distance(&mut self, query: &JsGeo3dDistanceQuery) {
        self.lexical_query = Some(JsQuery::Geo3dDistanceQuery(query.clone()));
    }

    /// Set a [`JsGeo3dBoundingBoxQuery`] as the lexical clause.
    #[napi(js_name = "setLexicalGeo3dBoundingBox")]
    pub fn set_lexical_geo3d_bounding_box(&mut self, query: &JsGeo3dBoundingBoxQuery) {
        self.lexical_query = Some(JsQuery::Geo3dBoundingBoxQuery(query.clone()));
    }

    /// Set a [`JsGeo3dNearestQuery`] as the lexical clause.
    #[napi(js_name = "setLexicalGeo3dNearest")]
    pub fn set_lexical_geo3d_nearest(&mut self, query: &JsGeo3dNearestQuery) {
        self.lexical_query = Some(JsQuery::Geo3dNearestQuery(query.clone()));
    }

    /// Set a [`JsBooleanQuery`] as the lexical clause.
    #[napi]
    pub fn set_lexical_boolean(&mut self, query: &JsBooleanQuery) {
        self.lexical_query = Some(JsQuery::BooleanQuery(query.clone()));
    }

    /// Set a [`JsSpanQuery`] as the lexical clause.
    #[napi]
    pub fn set_lexical_span(&mut self, query: &JsSpanQuery) {
        self.lexical_query = Some(JsQuery::SpanQuery(query.clone()));
    }

    // ── Filter query setters (one per query type) ───────────────────────

    /// Set a [`JsTermQuery`] as the filter clause (applied after scoring).
    #[napi]
    pub fn set_filter_term(&mut self, query: &JsTermQuery) {
        self.filter_query = Some(JsQuery::TermQuery(query.clone()));
    }

    /// Set a [`JsPhraseQuery`] as the filter clause.
    #[napi]
    pub fn set_filter_phrase(&mut self, query: &JsPhraseQuery) {
        self.filter_query = Some(JsQuery::PhraseQuery(query.clone()));
    }

    /// Set a [`JsFuzzyQuery`] as the filter clause.
    #[napi]
    pub fn set_filter_fuzzy(&mut self, query: &JsFuzzyQuery) {
        self.filter_query = Some(JsQuery::FuzzyQuery(query.clone()));
    }

    /// Set a [`JsWildcardQuery`] as the filter clause.
    #[napi]
    pub fn set_filter_wildcard(&mut self, query: &JsWildcardQuery) {
        self.filter_query = Some(JsQuery::WildcardQuery(query.clone()));
    }

    /// Set a [`JsNumericRangeQuery`] as the filter clause.
    #[napi]
    pub fn set_filter_numeric_range(&mut self, query: &JsNumericRangeQuery) {
        self.filter_query = Some(JsQuery::NumericRangeQuery(query.clone()));
    }

    /// Set a [`JsGeoDistanceQuery`] as the filter clause.
    #[napi]
    pub fn set_filter_geo_distance(&mut self, query: &JsGeoDistanceQuery) {
        self.filter_query = Some(JsQuery::GeoDistanceQuery(query.clone()));
    }

    /// Set a [`JsGeoBoundingBoxQuery`] as the filter clause.
    #[napi]
    pub fn set_filter_geo_bounding_box(&mut self, query: &JsGeoBoundingBoxQuery) {
        self.filter_query = Some(JsQuery::GeoBoundingBoxQuery(query.clone()));
    }

    /// Set a [`JsGeo3dDistanceQuery`] as the filter clause.
    #[napi(js_name = "setFilterGeo3dDistance")]
    pub fn set_filter_geo3d_distance(&mut self, query: &JsGeo3dDistanceQuery) {
        self.filter_query = Some(JsQuery::Geo3dDistanceQuery(query.clone()));
    }

    /// Set a [`JsGeo3dBoundingBoxQuery`] as the filter clause.
    #[napi(js_name = "setFilterGeo3dBoundingBox")]
    pub fn set_filter_geo3d_bounding_box(&mut self, query: &JsGeo3dBoundingBoxQuery) {
        self.filter_query = Some(JsQuery::Geo3dBoundingBoxQuery(query.clone()));
    }

    /// Set a [`JsGeo3dNearestQuery`] as the filter clause.
    #[napi(js_name = "setFilterGeo3dNearest")]
    pub fn set_filter_geo3d_nearest(&mut self, query: &JsGeo3dNearestQuery) {
        self.filter_query = Some(JsQuery::Geo3dNearestQuery(query.clone()));
    }

    /// Set a [`JsBooleanQuery`] as the filter clause.
    #[napi]
    pub fn set_filter_boolean(&mut self, query: &JsBooleanQuery) {
        self.filter_query = Some(JsQuery::BooleanQuery(query.clone()));
    }

    /// Set a [`JsSpanQuery`] as the filter clause.
    #[napi]
    pub fn set_filter_span(&mut self, query: &JsSpanQuery) {
        self.filter_query = Some(JsQuery::SpanQuery(query.clone()));
    }

    // ── Vector query setters ────────────────────────────────────────────

    /// Set a [`JsVectorQueryInner`] (pre-computed embedding) as the
    /// vector clause.
    #[napi]
    pub fn set_vector_query(&mut self, query: &JsVectorQueryInner) {
        self.vector_query = Some(JsVectorQuery::VectorQuery(query.clone()));
    }

    /// Set a [`JsVectorTextQuery`] (text-to-be-embedded) as the vector
    /// clause.
    #[napi]
    pub fn set_vector_text_query(&mut self, query: &JsVectorTextQuery) {
        self.vector_query = Some(JsVectorQuery::VectorTextQuery(query.clone()));
    }

    // ── Fusion algorithm setters ────────────────────────────────────────

    /// Set RRF fusion using a [`JsRRF`] instance.
    #[napi]
    pub fn set_rrf_fusion(&mut self, rrf: &JsRRF) {
        self.fusion = Some(FusionChoice::RRF(rrf.k));
    }

    /// Set weighted-sum fusion using a [`JsWeightedSum`] instance.
    #[napi]
    pub fn set_weighted_sum_fusion(&mut self, ws: &JsWeightedSum) {
        self.fusion = Some(FusionChoice::WeightedSum(
            ws.lexical_weight as f32,
            ws.vector_weight as f32,
        ));
    }
}

impl JsSearchRequest {
    /// Build the Laurus [`laurus::SearchRequest`] from this wrapper.
    pub fn build(&self) -> Result<laurus::SearchRequest> {
        let mut builder = SearchRequestBuilder::new()
            .limit(self.limit)
            .offset(self.offset);

        // Fusion algorithm
        if let Some(fusion) = &self.fusion {
            match fusion {
                FusionChoice::RRF(k) => {
                    builder = builder.fusion_algorithm(FusionAlgorithm::RRF { k: *k });
                }
                FusionChoice::WeightedSum(lw, vw) => {
                    builder = builder.fusion_algorithm(FusionAlgorithm::WeightedSum {
                        lexical_weight: *lw,
                        vector_weight: *vw,
                    });
                }
            }
        }

        // Filter query
        if let Some(fq) = &self.filter_query {
            builder = builder.filter_query(extract_lexical_query(fq)?);
        }

        // Highlighting (Issue #1134). Applied before every branch below
        // returns, so it takes effect regardless of which query shape is set.
        if let Some(h) = &self.highlight {
            let options = js_highlight_options_to_core(h);
            builder = builder
                .highlight(options.fields)
                .highlight_config(options.config);
        }

        // Explicit hybrid: lexical_query + vector_query both set
        if let (Some(lq), Some(vq)) = (&self.lexical_query, &self.vector_query) {
            builder = builder
                .lexical_query(query_to_lexical_search_query(lq)?)
                .vector_query(vector_query_to_search_query(vq));
            // Apply default RRF if no fusion specified
            if self.fusion.is_none() {
                builder = builder.fusion_algorithm(FusionAlgorithm::RRF { k: 60.0 });
            }
            return Ok(builder.build());
        }

        // Only lexical_query set
        if let Some(lq) = &self.lexical_query {
            builder = builder.lexical_query(query_to_lexical_search_query(lq)?);
            return Ok(builder.build());
        }

        // Only vector_query set
        if let Some(vq) = &self.vector_query {
            builder = builder.vector_query(vector_query_to_search_query(vq));
            return Ok(builder.build());
        }

        // DSL string
        if let Some(dsl) = &self.query_dsl {
            builder = builder.query_dsl(dsl.clone());
            return Ok(builder.build());
        }

        Ok(builder.build())
    }
}

// ---------------------------------------------------------------------------
// Helper: build a SearchRequest from index.search() arguments
// ---------------------------------------------------------------------------

/// Build a [`laurus::SearchRequest`] from a DSL string with limit/offset.
///
/// # Arguments
///
/// * `dsl` - The query DSL string.
/// * `limit` - Maximum results.
/// * `offset` - Pagination offset.
///
/// # Returns
///
/// A `SearchRequest` configured with the DSL query.
pub fn build_dsl_request(dsl: String, limit: usize, offset: usize) -> laurus::SearchRequest {
    SearchRequestBuilder::new()
        .limit(limit)
        .offset(offset)
        .query_dsl(dsl)
        .build()
}

/// Build a [`laurus::SearchRequest`] from a lexical query.
///
/// # Arguments
///
/// * `query` - The lexical query.
/// * `limit` - Maximum results.
/// * `offset` - Pagination offset.
///
/// # Returns
///
/// A `SearchRequest` configured with the lexical query.
pub fn build_lexical_request(
    query: &JsQuery,
    limit: usize,
    offset: usize,
) -> Result<laurus::SearchRequest> {
    Ok(SearchRequestBuilder::new()
        .limit(limit)
        .offset(offset)
        .lexical_query(LexicalSearchQuery::Obj(extract_lexical_query(query)?))
        .build())
}

/// Build a [`laurus::SearchRequest`] from a vector query.
///
/// # Arguments
///
/// * `query` - The vector query.
/// * `limit` - Maximum results.
/// * `offset` - Pagination offset.
///
/// # Returns
///
/// A `SearchRequest` configured with the vector query.
pub fn build_vector_request(
    query: &JsVectorQuery,
    limit: usize,
    offset: usize,
) -> laurus::SearchRequest {
    SearchRequestBuilder::new()
        .limit(limit)
        .offset(offset)
        .vector_query(vector_query_to_search_query(query))
        .build()
}
