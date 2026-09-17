//! Python wrappers for search request/result and fusion algorithm types.

use std::collections::HashMap;

use crate::convert::document_to_dict;
use crate::query::{
    extract_lexical_query, is_vector_query, py_to_lexical_search_query, py_to_vector_search_query,
};
use laurus::{
    FusionAlgorithm, HighlightConfig, HighlightOptions, LexicalSearchQuery, SearchRequestBuilder,
    SearchResult,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Highlighting (Issue #1134)
// ---------------------------------------------------------------------------

/// Convert a Python `highlight` argument into [`HighlightOptions`].
///
/// Accepts either a list of field names (`["title", "body"]`), or a dict
/// with a required `"fields"` key plus any of the optional
/// [`HighlightConfig`] knobs: `max_fragments`, `fragment_size`, `tag`,
/// `css_class`, `require_field_match`, `max_analyzed_chars`,
/// `return_entire_field_if_no_highlight`. The dict form is checked first
/// (a Python `dict`'s default iteration yields keys, which would otherwise
/// silently mis-extract as a one-element field list).
fn py_to_highlight_options(obj: &Bound<PyAny>) -> PyResult<HighlightOptions> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        let fields: Vec<String> = dict
            .get_item("fields")?
            .ok_or_else(|| PyValueError::new_err("highlight dict requires a 'fields' key"))?
            .extract()?;

        let mut config = HighlightConfig::new();
        if let Some(v) = dict.get_item("max_fragments")? {
            config = config.max_fragments(v.extract()?);
        }
        if let Some(v) = dict.get_item("fragment_size")? {
            config = config.fragment_size(v.extract()?);
        }
        if let Some(v) = dict.get_item("tag")? {
            config = config.tag(v.extract()?);
        }
        if let Some(v) = dict.get_item("css_class")? {
            config = config.css_class(v.extract()?);
        }
        if let Some(v) = dict.get_item("require_field_match")? {
            config = config.require_field_match(v.extract()?);
        }
        if let Some(v) = dict.get_item("max_analyzed_chars")? {
            config.max_analyzed_chars = v.extract()?;
        }
        if let Some(v) = dict.get_item("return_entire_field_if_no_highlight")? {
            config.return_entire_field_if_no_highlight = v.extract()?;
        }

        return Ok(HighlightOptions::new(fields).with_config(config));
    }

    if let Ok(fields) = obj.extract::<Vec<String>>() {
        return Ok(HighlightOptions::new(fields));
    }

    Err(PyValueError::new_err(
        "highlight must be a list of field names or a dict, \
         e.g. ['body'] or {'fields': ['body'], 'tag': 'em'}",
    ))
}

// ---------------------------------------------------------------------------
// Fusion algorithm types
// ---------------------------------------------------------------------------

/// Reciprocal Rank Fusion — rank-based result merging for hybrid search.
///
/// ## Example
///
/// ```python
/// fusion = laurus.RRF(k=60.0)
/// ```
#[pyclass(name = "RRF", from_py_object)]
#[derive(Clone)]
pub struct PyRRF {
    pub k: f64,
}

#[pymethods]
impl PyRRF {
    #[new]
    #[pyo3(signature = (k=60.0))]
    pub fn new(k: f64) -> Self {
        Self { k }
    }
    fn __repr__(&self) -> String {
        format!("RRF(k={})", self.k)
    }
}

/// Weighted sum fusion — normalises lexical and vector scores then combines them.
///
/// ## Example
///
/// ```python
/// fusion = laurus.WeightedSum(lexical_weight=0.3, vector_weight=0.7)
/// ```
#[pyclass(name = "WeightedSum", from_py_object)]
#[derive(Clone)]
pub struct PyWeightedSum {
    pub lexical_weight: f32,
    pub vector_weight: f32,
}

#[pymethods]
impl PyWeightedSum {
    #[new]
    #[pyo3(signature = (lexical_weight=0.5, vector_weight=0.5))]
    pub fn new(lexical_weight: f32, vector_weight: f32) -> Self {
        Self {
            lexical_weight,
            vector_weight,
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "WeightedSum(lexical_weight={}, vector_weight={})",
            self.lexical_weight, self.vector_weight
        )
    }
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// A single search result returned by [`Index.search`].
///
/// Attributes:
///     id (str): External document identifier.
///     score (float): Relevance score (BM25, similarity, or fused).
///     document (dict | None): Retrieved document fields, or `None` if deleted.
///     highlights (dict[str, list[str]]): Highlighted fragments per field
///         requested via `highlight=`, best fragment first. Empty when
///         highlighting was not requested, the field was not a stored text
///         field, or nothing in it matched.
#[pyclass(name = "SearchResult")]
pub struct PySearchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub highlights: HashMap<String, Vec<String>>,
    document: Option<Py<PyAny>>,
}

#[pymethods]
impl PySearchResult {
    #[getter]
    pub fn document(&self, py: Python) -> Py<PyAny> {
        match &self.document {
            Some(d) => d.clone_ref(py),
            None => py.None(),
        }
    }
    fn __repr__(&self) -> String {
        format!("SearchResult(id='{}', score={:.4})", self.id, self.score)
    }
}

/// Convert a [`SearchResult`] from the engine into a [`PySearchResult`].
pub fn to_py_search_result(py: Python, r: SearchResult) -> PyResult<PySearchResult> {
    let document = r
        .document
        .as_ref()
        .map(|doc| document_to_dict(py, doc))
        .transpose()?;
    Ok(PySearchResult {
        id: r.id,
        score: r.score,
        highlights: r.highlights,
        document,
    })
}

// ---------------------------------------------------------------------------
// SearchRequest
// ---------------------------------------------------------------------------

/// Full-featured search request for advanced control over query, fusion, and
/// filtering.
///
/// For simple queries, prefer passing a query object or DSL string directly
/// to `Index.search()`.
///
/// ## Example — hybrid search with filter
///
/// ```python
/// request = laurus.SearchRequest(
///     vector_query=laurus.VectorTextQuery("text_vec", "type system"),
///     filter_query=laurus.TermQuery("category", "type-system"),
///     fusion=laurus.RRF(k=60.0),
///     limit=3,
/// )
/// results = index.search(request)
/// ```
#[pyclass(name = "SearchRequest")]
pub struct PySearchRequest {
    /// A DSL string, or any single lexical/vector query object.
    /// Mutually exclusive with `lexical_query` + `vector_query`.
    pub query: Option<Py<PyAny>>,
    /// Lexical component for explicit hybrid search.
    pub lexical_query: Option<Py<PyAny>>,
    /// Vector component for explicit hybrid search.
    pub vector_query: Option<Py<PyAny>>,
    /// Optional lexical filter query applied after scoring.
    pub filter_query: Option<Py<PyAny>>,
    /// Fusion algorithm for hybrid results (`RRF` or `WeightedSum`).
    pub fusion: Option<Py<PyAny>>,
    pub limit: usize,
    pub offset: usize,
    /// Highlight request: a list of field names, or a dict adding
    /// `HighlightConfig` knobs (Issue #1134). See `Index.search`'s
    /// `highlight` parameter for the accepted shapes.
    pub highlight: Option<Py<PyAny>>,
}

#[pymethods]
impl PySearchRequest {
    #[new]
    #[pyo3(signature = (
        *,
        query=None,
        lexical_query=None,
        vector_query=None,
        filter_query=None,
        fusion=None,
        limit=10,
        offset=0,
        highlight=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query: Option<Py<PyAny>>,
        lexical_query: Option<Py<PyAny>>,
        vector_query: Option<Py<PyAny>>,
        filter_query: Option<Py<PyAny>>,
        fusion: Option<Py<PyAny>>,
        limit: usize,
        offset: usize,
        highlight: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            query,
            lexical_query,
            vector_query,
            filter_query,
            fusion,
            limit,
            offset,
            highlight,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchRequest(limit={}, offset={})",
            self.limit, self.offset
        )
    }
}

impl PySearchRequest {
    /// Build the Laurus [`laurus::SearchRequest`] from this Python wrapper.
    pub fn build(&self, py: Python) -> PyResult<laurus::SearchRequest> {
        let mut builder = SearchRequestBuilder::new()
            .limit(self.limit)
            .offset(self.offset);

        // ── Fusion algorithm ──────────────────────────────────────────────
        if let Some(f) = &self.fusion {
            let fobj: &Bound<'_, PyAny> = f.bind(py);
            if let Ok(rrf) = fobj.extract::<PyRef<PyRRF>>() {
                builder = builder.fusion_algorithm(FusionAlgorithm::RRF { k: rrf.k });
            } else if let Ok(ws) = fobj.extract::<PyRef<PyWeightedSum>>() {
                builder = builder.fusion_algorithm(FusionAlgorithm::WeightedSum {
                    lexical_weight: ws.lexical_weight,
                    vector_weight: ws.vector_weight,
                });
            }
        }

        // ── Filter query ──────────────────────────────────────────────────
        if let Some(fq) = &self.filter_query {
            let fq_obj: &Bound<'_, PyAny> = fq.bind(py);
            builder = builder.filter_query(extract_lexical_query(py, fq_obj)?);
        }

        // ── Highlighting (Issue #1134) ──────────────────────────────────────
        // Applied before every branch below returns, so it takes effect
        // regardless of which query shape (`query`, `lexical_query` +
        // `vector_query`, or either alone) is set.
        if let Some(h) = &self.highlight {
            let h_obj: &Bound<'_, PyAny> = h.bind(py);
            let options = py_to_highlight_options(h_obj)?;
            builder = builder
                .highlight(options.fields)
                .highlight_config(options.config);
        }

        // ── Explicit hybrid: lexical_query + vector_query both set ────────
        if let (Some(lq), Some(vq)) = (&self.lexical_query, &self.vector_query) {
            let lq_obj: &Bound<'_, PyAny> = lq.bind(py);
            let vq_obj: &Bound<'_, PyAny> = vq.bind(py);
            builder = builder
                .lexical_query(py_to_lexical_search_query(py, lq_obj)?)
                .vector_query(py_to_vector_search_query(vq_obj)?);
            // Apply default RRF if no fusion specified
            if self.fusion.is_none() {
                builder = builder.fusion_algorithm(FusionAlgorithm::RRF { k: 60.0 });
            }
            return Ok(builder.build());
        }

        // ── Only lexical_query set ────────────────────────────────────────
        if let Some(lq) = &self.lexical_query {
            let lq_obj: &Bound<'_, PyAny> = lq.bind(py);
            builder = builder.lexical_query(py_to_lexical_search_query(py, lq_obj)?);
            return Ok(builder.build());
        }

        // ── Only vector_query set ─────────────────────────────────────────
        if let Some(vq) = &self.vector_query {
            let vq_obj: &Bound<'_, PyAny> = vq.bind(py);
            builder = builder.vector_query(py_to_vector_search_query(vq_obj)?);
            return Ok(builder.build());
        }

        // ── Single `query` field: DSL string, lexical, or vector ──────────
        if let Some(q) = &self.query {
            let qobj: &Bound<'_, PyAny> = q.bind(py);
            if let Ok(s) = qobj.extract::<String>() {
                builder = builder.query_dsl(s);
            } else if is_vector_query(qobj) {
                builder = builder.vector_query(py_to_vector_search_query(qobj)?);
            } else {
                builder = builder.lexical_query(py_to_lexical_search_query(py, qobj)?);
            }
        }

        Ok(builder.build())
    }
}

// ---------------------------------------------------------------------------
// Helper: build a SearchRequest from `index.search()` arguments
// ---------------------------------------------------------------------------

/// Build a [`laurus::SearchRequest`] from the arguments passed to
/// `Index.search(query, limit, offset, highlight)`.
///
/// `query` may be:
/// - A `str` (DSL)
/// - A `PySearchRequest` (full request)
/// - Any lexical query class (`TermQuery`, `BooleanQuery`, …)
/// - `VectorQuery` or `VectorTextQuery`
/// - A `PyRRF` / `PyWeightedSum` are not valid here
///
/// When `query` is a `PySearchRequest`, `limit`/`offset`/`highlight` are
/// used as-is from the request without overriding — same precedent as
/// `limit`/`offset` already had here before highlighting existed.
pub fn build_request_from_py(
    py: Python,
    query: &Bound<PyAny>,
    limit: usize,
    offset: usize,
    highlight: Option<&Bound<PyAny>>,
) -> PyResult<laurus::SearchRequest> {
    // Full SearchRequest object — use limit/offset as-is without overriding.
    if let Ok(req) = query.extract::<PyRef<PySearchRequest>>() {
        return req.build(py);
    }

    let mut builder = SearchRequestBuilder::new().limit(limit).offset(offset);

    if let Some(h) = highlight {
        let options = py_to_highlight_options(h)?;
        builder = builder
            .highlight(options.fields)
            .highlight_config(options.config);
    }

    // DSL string
    if let Ok(s) = query.extract::<String>() {
        builder = builder.query_dsl(s);
        return Ok(builder.build());
    }

    // Vector queries
    if is_vector_query(query) {
        builder = builder.vector_query(py_to_vector_search_query(query)?);
        return Ok(builder.build());
    }

    // Lexical queries
    builder = builder.lexical_query(LexicalSearchQuery::Obj(extract_lexical_query(py, query)?));
    Ok(builder.build())
}
