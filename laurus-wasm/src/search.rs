//! WASM wrappers for search request/result and fusion algorithm types.

use crate::convert::optional_unsigned_param;
use crate::query::{
    JsQuery, JsVectorQuery, extract_lexical_query, query_to_lexical_search_query,
    vector_query_to_search_query,
};
use laurus::vector::Vector;
use laurus::{
    FusionAlgorithm, GroupBy, HighlightConfig, HighlightOptions, LexicalSearchQuery,
    RescoreOptions, SearchRequestBuilder,
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
/// `{ fields: ["body"], maxFragments: 2, tag: "em" }`. The integers are
/// numbers here, checked by [`optional_unsigned_param`] so that an error
/// names the field (Issue #1612).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmHighlightOptions {
    fields: Vec<String>,
    fragment_size: Option<f64>,
    max_fragments: Option<f64>,
    tag: Option<String>,
    css_class: Option<String>,
    require_field_match: Option<bool>,
}

/// Parse an optional highlight-options JS object into [`HighlightOptions`].
/// `None` (no object passed) means "don't highlight" and returns `Ok(None)`.
///
/// # Arguments
///
/// * `options` - The highlight options object.
/// * `context` - The method the options were passed to, such as
///   `Index.search`, which starts an error message.
///
/// # Errors
///
/// Returns an error string if the object does not have the shape above, or
/// if `fragmentSize` or `maxFragments` is not an integer from 0 to
/// 4294967295.
pub fn parse_highlight_options(
    options: Option<js_sys::Object>,
    context: &str,
) -> Result<Option<HighlightOptions>, JsValue> {
    let Some(options) = options else {
        return Ok(None);
    };
    let parsed: WasmHighlightOptions = serde_wasm_bindgen::from_value(options.into())
        .map_err(|e| JsValue::from_str(&format!("Invalid highlight options: {e}")))?;
    let fragment_size =
        optional_unsigned_param(context, "highlight.fragmentSize", parsed.fragment_size)?;
    let max_fragments =
        optional_unsigned_param(context, "highlight.maxFragments", parsed.max_fragments)?;

    let mut config = HighlightConfig::new();
    if let Some(fragment_size) = fragment_size {
        config = config.fragment_size(fragment_size as usize);
    }
    if let Some(max_fragments) = max_fragments {
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
// Rescore (Issue #1351)
// ---------------------------------------------------------------------------

/// JS-facing shape of a late-interaction rescore, deserialized via
/// `serde_wasm_bindgen` from a plain JS object, e.g.
/// `{ field: "tokens", vectors: [[1, 0], [0, 1]], windowSize: 50 }` or
/// `{ field: "body_colbert", text: "how do lifetimes work" }`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmRescoreOptions {
    field: String,
    vectors: Option<Vec<Vec<f32>>>,
    text: Option<String>,
    window_size: Option<f64>,
}

/// Parse an optional rescore JS object into [`RescoreOptions`]. Only the
/// shape and the integer are checked here; the engine checks the values
/// when searching. `None` (no object passed) means "don't rescore" and
/// returns `Ok(None)`.
///
/// # Arguments
///
/// * `options` - The rescore options object.
/// * `context` - The method the options were passed to, such as
///   `Index.search`, which starts an error message.
///
/// # Errors
///
/// Returns an error string if the object does not have the shape above,
/// if `windowSize` is not an integer from 0 to 4294967295, or unless
/// exactly one of `vectors` and `text` is set.
pub fn parse_rescore_options(
    options: Option<js_sys::Object>,
    context: &str,
) -> Result<Option<RescoreOptions>, JsValue> {
    let Some(options) = options else {
        return Ok(None);
    };
    let parsed: WasmRescoreOptions = serde_wasm_bindgen::from_value(options.into())
        .map_err(|e| JsValue::from_str(&format!("Invalid rescore options: {e}")))?;
    let window_size = optional_unsigned_param(context, "rescore.windowSize", parsed.window_size)?;

    let rescore = match (parsed.vectors, parsed.text) {
        (Some(vectors), None) => RescoreOptions::late_interaction(
            parsed.field,
            vectors.into_iter().map(Vector::new).collect(),
        ),
        (None, Some(text)) => RescoreOptions::late_interaction_text(parsed.field, text),
        _ => {
            return Err(JsValue::from_str(
                "Invalid rescore options: set exactly one of vectors or text",
            ));
        }
    };
    Ok(Some(match window_size {
        Some(window_size) => rescore.window_size(window_size as usize),
        None => rescore,
    }))
}

// ---------------------------------------------------------------------------
// Grouping (Issue #1297)
// ---------------------------------------------------------------------------

/// JS-facing shape of [`GroupBy`], deserialized via `serde_wasm_bindgen`
/// from a plain JS object, e.g. `{ field: "_id", groupSize: 3 }`.
///
/// Each result is then a group's best hit: its `groupKey` is the value the
/// group shares and its `innerHits` hold the group's next best hits, up to
/// `groupSize` hits per group. `limit` and `offset` count groups.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WasmGroupByOptions {
    field: String,
    group_size: Option<f64>,
}

/// Parse an optional grouping JS object into [`GroupBy`]. Only the shape and
/// the integer are checked here; the engine checks the values when
/// searching. `None` (no object passed) means "don't group" and returns
/// `Ok(None)`.
///
/// # Arguments
///
/// * `options` - The grouping options object.
/// * `context` - The method the options were passed to, such as
///   `Index.search`, which starts an error message.
///
/// # Errors
///
/// Returns an error string if the object does not have the shape above, or
/// if `groupSize` is not an integer from 0 to 4294967295.
pub fn parse_group_by_options(
    options: Option<js_sys::Object>,
    context: &str,
) -> Result<Option<GroupBy>, JsValue> {
    let Some(options) = options else {
        return Ok(None);
    };
    let parsed: WasmGroupByOptions = serde_wasm_bindgen::from_value(options.into())
        .map_err(|e| JsValue::from_str(&format!("Invalid groupBy options: {e}")))?;
    let group_size = optional_unsigned_param(context, "groupBy.groupSize", parsed.group_size)?;

    let group_by = GroupBy::new(parsed.field);
    Ok(Some(match group_size {
        Some(group_size) => group_by.group_size(group_size as usize),
        None => group_by,
    }))
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

/// Convert the `limit` and `offset` passed to the search method `context`,
/// which default to 10 and 0.
///
/// # Errors
///
/// Returns an error string if `limit` or `offset` is not an integer from 0
/// to 4294967295 (Issue #1612).
pub fn limit_and_offset(
    context: &str,
    limit: Option<f64>,
    offset: Option<f64>,
) -> Result<(usize, usize), JsValue> {
    let limit = optional_unsigned_param(context, "limit", limit)?.unwrap_or(10);
    let offset = optional_unsigned_param(context, "offset", offset)?.unwrap_or(0);
    Ok((limit as usize, offset as usize))
}

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
