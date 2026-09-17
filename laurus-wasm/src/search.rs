//! WASM wrappers for search request/result and fusion algorithm types.

use crate::query::{
    JsQuery, JsVectorQuery, extract_lexical_query, query_to_lexical_search_query,
    vector_query_to_search_query,
};
use laurus::{
    FusionAlgorithm, HighlightConfig, HighlightOptions, LexicalSearchQuery, SearchRequestBuilder,
};
use serde::Deserialize;
use wasm_bindgen::JsValue;

// ---------------------------------------------------------------------------
// Highlighting (Issue #1134)
// ---------------------------------------------------------------------------

/// JS-facing shape of [`HighlightOptions`] plus the [`HighlightConfig`]
/// knobs exposed to bindings: `fields` (required) and a reduced set of tag /
/// fragment / field-match settings. `fragment_overlap`, `fragment_separator`
/// and `max_analyzed_chars` are not exposed — they either have no effect
/// (the first two, currently unused by fragment selection) or are unlikely
/// to be worth tuning from JS (the last one).
///
/// Deserialized via `serde_wasm_bindgen` from a plain JS object, e.g.
/// `{ fields: ["body"], maxFragments: 2, tag: "em" }`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmHighlightOptions {
    fields: Vec<String>,
    fragment_size: Option<u32>,
    max_fragments: Option<u32>,
    tag: Option<String>,
    css_class: Option<String>,
    require_field_match: Option<bool>,
}

/// Parse an optional highlight-options JS object into [`HighlightOptions`].
/// `None` (no object passed) means "don't highlight" and returns `Ok(None)`.
pub fn parse_highlight_options(
    options: Option<js_sys::Object>,
) -> Result<Option<HighlightOptions>, JsValue> {
    let Some(options) = options else {
        return Ok(None);
    };
    let parsed: WasmHighlightOptions = serde_wasm_bindgen::from_value(options.into())
        .map_err(|e| JsValue::from_str(&format!("Invalid highlight options: {e}")))?;

    let mut config = HighlightConfig::new();
    if let Some(fragment_size) = parsed.fragment_size {
        config = config.fragment_size(fragment_size as usize);
    }
    if let Some(max_fragments) = parsed.max_fragments {
        config = config.max_fragments(max_fragments as usize);
    }
    if let Some(tag) = parsed.tag {
        config = config.tag(tag);
    }
    if let Some(css_class) = parsed.css_class {
        config = config.css_class(css_class);
    }
    if let Some(require_field_match) = parsed.require_field_match {
        config = config.require_field_match(require_field_match);
    }

    Ok(Some(
        HighlightOptions::new(parsed.fields).with_config(config),
    ))
}

// ---------------------------------------------------------------------------
// SearchRequest internal types
// ---------------------------------------------------------------------------

pub enum FusionChoice {
    RRF(f64),
    WeightedSum(f32, f32),
}

/// Internal search request state.
pub struct WasmSearchRequestInner {
    pub query_dsl: Option<String>,
    pub lexical_query: Option<JsQuery>,
    pub vector_query: Option<JsVectorQuery>,
    pub filter_query: Option<JsQuery>,
    pub fusion: Option<FusionChoice>,
    pub limit: usize,
    pub offset: usize,
}

impl WasmSearchRequestInner {
    pub fn new(limit: usize, offset: usize) -> Self {
        Self {
            query_dsl: None,
            lexical_query: None,
            vector_query: None,
            filter_query: None,
            fusion: None,
            limit,
            offset,
        }
    }

    /// Build the Laurus [`laurus::SearchRequest`] from this wrapper.
    pub fn build(&self) -> Result<laurus::SearchRequest, JsValue> {
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

        // Explicit hybrid: lexical_query + vector_query both set
        if let (Some(lq), Some(vq)) = (&self.lexical_query, &self.vector_query) {
            builder = builder
                .lexical_query(query_to_lexical_search_query(lq)?)
                .vector_query(vector_query_to_search_query(vq));
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
pub fn build_dsl_request(dsl: String, limit: usize, offset: usize) -> laurus::SearchRequest {
    SearchRequestBuilder::new()
        .limit(limit)
        .offset(offset)
        .query_dsl(dsl)
        .build()
}

/// Build a [`laurus::SearchRequest`] from a lexical query.
pub fn build_lexical_request(
    query: &JsQuery,
    limit: usize,
    offset: usize,
) -> Result<laurus::SearchRequest, JsValue> {
    Ok(SearchRequestBuilder::new()
        .limit(limit)
        .offset(offset)
        .lexical_query(LexicalSearchQuery::Obj(extract_lexical_query(query)?))
        .build())
}

/// Build a [`laurus::SearchRequest`] from a vector query.
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
