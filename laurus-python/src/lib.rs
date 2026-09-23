//! Python bindings for the Laurus unified search library.
//!
//! Build with [maturin](https://github.com/PyO3/maturin):
//!
//! ```bash
//! cd laurus-python
//! maturin develop        # install into current virtualenv
//! maturin build --release  # produce a wheel
//! ```

mod analysis;
mod convert;
mod errors;
mod index;
mod query;
mod schema;
mod search;

use analysis::{PySynonymDictionary, PySynonymGraphFilter, PyToken, PyWhitespaceTokenizer};
use index::{PyCommitPolicy, PyIndex, PyWalSyncPolicy, peek_commit_generation};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use query::{
    PyBooleanQuery, PyDateTimeRangeQuery, PyFuzzyQuery, PyGeo3dBoundingBoxQuery,
    PyGeo3dDistanceQuery, PyGeo3dNearestQuery, PyGeoBoundingBoxQuery, PyGeoDistanceQuery,
    PyNumericRangeQuery, PyPhraseQuery, PySpanQuery, PyTermQuery, PyVectorQuery, PyVectorTextQuery,
    PyWildcardQuery,
};
use schema::PySchema;
use search::{PyRRF, PySearchRequest, PySearchResult, PyWeightedSum};

/// Laurus — unified lexical, vector, and hybrid search for Python.
///
/// ## Quick start
///
/// ```python
/// import laurus
///
/// schema = laurus.Schema()
/// schema.add_text_field("title")
/// schema.add_text_field("body")
/// schema.set_default_fields(["title", "body"])
///
/// index = laurus.Index(schema=schema)
///
/// index.add_document("doc1", {"title": "Rust Programming", "body": "Safety and speed."})
/// index.add_document("doc2", {"title": "Python Basics",    "body": "Versatile language."})
/// index.commit()
///
/// for r in index.search("programming", limit=5):
///     print(r.id, r.score, r.document["title"])
/// ```
#[pymodule]
fn laurus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Core ─────────────────────────────────────────────────────────────
    m.add_class::<PyIndex>()?;
    m.add_class::<PySchema>()?;
    m.add_class::<PyWalSyncPolicy>()?;
    m.add_class::<PyCommitPolicy>()?;
    m.add_function(wrap_pyfunction!(peek_commit_generation, m)?)?;

    // ── Search result & request ───────────────────────────────────────────
    m.add_class::<PySearchResult>()?;
    m.add_class::<PySearchRequest>()?;

    // ── Fusion algorithms ─────────────────────────────────────────────────
    m.add_class::<PyRRF>()?;
    m.add_class::<PyWeightedSum>()?;

    // ── Lexical query types ───────────────────────────────────────────────
    m.add_class::<PyTermQuery>()?;
    m.add_class::<PyPhraseQuery>()?;
    m.add_class::<PyFuzzyQuery>()?;
    m.add_class::<PyWildcardQuery>()?;
    m.add_class::<PyNumericRangeQuery>()?;
    m.add_class::<PyDateTimeRangeQuery>()?;
    m.add_class::<PyGeoDistanceQuery>()?;
    m.add_class::<PyGeoBoundingBoxQuery>()?;
    m.add_class::<PyGeo3dDistanceQuery>()?;
    m.add_class::<PyGeo3dBoundingBoxQuery>()?;
    m.add_class::<PyGeo3dNearestQuery>()?;
    m.add_class::<PyBooleanQuery>()?;
    m.add_class::<PySpanQuery>()?;

    // ── Vector query types ────────────────────────────────────────────────
    m.add_class::<PyVectorQuery>()?;
    m.add_class::<PyVectorTextQuery>()?;

    // ── Analysis pipeline ─────────────────────────────────────────────────
    m.add_class::<PyToken>()?;
    m.add_class::<PySynonymDictionary>()?;
    m.add_class::<PyWhitespaceTokenizer>()?;
    m.add_class::<PySynonymGraphFilter>()?;

    Ok(())
}
