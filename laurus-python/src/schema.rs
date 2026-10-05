//! Python wrapper for the Laurus [`Schema`] type.

use std::path::PathBuf;
use std::str::FromStr;

use laurus::{
    AnalyzerDefinition, AnalyzerSpec, BooleanOption, BuiltinAnalyzerSpec, BytesOption,
    CharFilterConfig, DateTimeOption, DistanceMetric, DynamicFieldPolicy, EmbedderDefinition,
    FieldOption, FlatOption, FloatOption, Geo3dOption, GeoOption, HnswOption, IntegerOption,
    IvfOption, MultiVectorOption, QuantizationMethod, RerankStorageKind, Schema, TextOption,
    TokenFilterConfig, TokenizerConfig,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::py_to_json_value;
use crate::errors::io_err_with_path;

/// Convert a Python analyzer reference into an [`AnalyzerSpec`].
///
/// Accepts either a `str` (resolved as [`AnalyzerSpec::Named`]) or a
/// `dict` describing a parameterized built-in preset. The dict must
/// include a `"language"` key naming the preset (currently only
/// `"japanese"`) plus its required parameters (`"dict"` for Japanese,
/// optionally `"mode"` and `"user_dict"`).
fn analyzer_spec_from_py(py: Python<'_>, obj: Py<PyAny>) -> PyResult<AnalyzerSpec> {
    if let Ok(name) = obj.extract::<String>(py) {
        return Ok(AnalyzerSpec::Named(name));
    }
    let bound = obj.bind(py);
    if let Ok(dict) = bound.cast::<PyDict>() {
        let language = dict
            .get_item("language")?
            .ok_or_else(|| PyValueError::new_err("analyzer dict requires a 'language' key"))?
            .extract::<String>()?;
        match language.as_str() {
            "japanese" => {
                let dict_path = dict
                    .get_item("dict")?
                    .ok_or_else(|| {
                        PyValueError::new_err("japanese analyzer requires a 'dict' path")
                    })?
                    .extract::<String>()?;
                let mode = dict
                    .get_item("mode")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_else(|| "normal".to_string());
                let user_dict = dict
                    .get_item("user_dict")?
                    .map(|v| v.extract::<String>())
                    .transpose()?;
                return Ok(AnalyzerSpec::Builtin(BuiltinAnalyzerSpec::Japanese {
                    mode,
                    dict: dict_path,
                    user_dict,
                }));
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported analyzer language: {other}"
                )));
            }
        }
    }
    Err(PyValueError::new_err(
        "analyzer must be a str or a dict (e.g. {'language': 'japanese', 'dict': '/path'})",
    ))
}

/// Convert a [`laurus::LaurusError`] from [`Schema::from_toml`]/[`Schema::to_toml`]
/// into a `ValueError`, preserving their message text as-is (both always
/// return the `Schema` variant; the fallback exists only for type safety).
fn schema_toml_err(e: laurus::LaurusError) -> PyErr {
    match e {
        laurus::LaurusError::Schema(m) => PyValueError::new_err(m),
        other => crate::errors::laurus_err(other),
    }
}

/// Convert a Python dict into a [`TokenizerConfig`], using the same
/// `{"type": "..."}`-tagged shape as the schema TOML/JSON format.
fn tokenizer_from_py(obj: &Bound<PyAny>) -> PyResult<TokenizerConfig> {
    let value = py_to_json_value(obj)?;
    serde_json::from_value(value)
        .map_err(|e| PyValueError::new_err(format!("invalid tokenizer: {e}")))
}

/// Convert a Python dict into a [`CharFilterConfig`].
fn char_filter_from_py(obj: &Bound<PyAny>, index: usize) -> PyResult<CharFilterConfig> {
    let value = py_to_json_value(obj)?;
    serde_json::from_value(value)
        .map_err(|e| PyValueError::new_err(format!("invalid char_filters[{index}]: {e}")))
}

/// Convert a Python dict into a [`TokenFilterConfig`].
fn token_filter_from_py(obj: &Bound<PyAny>, index: usize) -> PyResult<TokenFilterConfig> {
    let value = py_to_json_value(obj)?;
    serde_json::from_value(value)
        .map_err(|e| PyValueError::new_err(format!("invalid token_filters[{index}]: {e}")))
}

/// Convert an optional Python list of dicts into a `Vec<T>`, defaulting to
/// an empty vector when `None` (mirroring the core's `#[serde(default)]`
/// on `AnalyzerDefinition::char_filters`/`token_filters`).
fn filter_list_from_py<T>(
    obj: Option<&Bound<PyAny>>,
    label: &str,
    convert: impl Fn(&Bound<PyAny>, usize) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    let Some(obj) = obj else {
        return Ok(Vec::new());
    };
    let seq = obj
        .try_iter()
        .map_err(|_| PyValueError::new_err(format!("{label} must be a list of dicts")))?;
    seq.enumerate()
        .map(|(i, item)| convert(&item?, i))
        .collect()
}

/// Parse a distance metric string into [`DistanceMetric`].
fn parse_distance(s: &str) -> PyResult<DistanceMetric> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "dot_product" | "dot" => Ok(DistanceMetric::DotProduct),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "angular" => Ok(DistanceMetric::Angular),
        other => Err(PyValueError::new_err(format!(
            "Unknown distance metric: '{}'. Valid: cosine, euclidean, dot_product, manhattan, angular",
            other
        ))),
    }
}

/// Parse a quantizer name plus optional `subvector_count` into a
/// [`QuantizationMethod`].
///
/// Accepts `"scalar_8bit"` / `"scalar"` (the default when `name` is
/// `None`) and `"product_quantization"` / `"pq"`. Product quantization
/// requires a `subvector_count` (which must divide the field dimension —
/// the divisibility itself is validated by the core at index-build time);
/// supplying `subvector_count` for any other quantizer is rejected so an
/// incoherent configuration cannot silently reach the core.
fn parse_quantizer(
    name: Option<&str>,
    subvector_count: Option<usize>,
) -> PyResult<QuantizationMethod> {
    match name.map(|s| s.to_lowercase()).as_deref() {
        None | Some("scalar_8bit") | Some("scalar") => {
            if subvector_count.is_some() {
                return Err(PyValueError::new_err(
                    "subvector_count is only valid with quantizer='product_quantization'",
                ));
            }
            Ok(QuantizationMethod::Scalar8Bit)
        }
        Some("product_quantization") | Some("pq") => {
            let subvector_count = subvector_count.ok_or_else(|| {
                PyValueError::new_err(
                    "quantizer='product_quantization' requires subvector_count \
                     (must divide the field dimension)",
                )
            })?;
            Ok(QuantizationMethod::ProductQuantization { subvector_count })
        }
        Some(other) => Err(PyValueError::new_err(format!(
            "Unknown quantizer: '{other}'. Valid: scalar_8bit, product_quantization"
        ))),
    }
}

/// Parse a rerank-storage name into an optional [`RerankStorageKind`].
///
/// `None` (the default) keeps the Stage-1 int8-only segment; `"f32"`
/// enables the Stage-2 full-precision rerank sidecar (`*.hnsw.f32`).
fn parse_rerank_storage(name: Option<&str>) -> PyResult<Option<RerankStorageKind>> {
    match name.map(|s| s.to_lowercase()).as_deref() {
        None => Ok(None),
        Some("f32") => Ok(Some(RerankStorageKind::F32)),
        Some(other) => Err(PyValueError::new_err(format!(
            "Unknown rerank_storage: '{other}'. Valid: f32"
        ))),
    }
}

/// Python-facing schema builder.
///
/// ## Example
///
/// ```python
/// schema = laurus.Schema()
/// schema.add_text_field("title")
/// schema.add_hnsw_field("embedding", dimension=384, distance="cosine")
/// schema.add_integer_field("year")
/// schema.set_default_fields(["title"])
/// ```
#[pyclass(name = "Schema")]
pub struct PySchema {
    pub inner: Schema,
}

#[pymethods]
impl PySchema {
    /// Create a new empty schema.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Schema::new(),
        }
    }

    /// Add a full-text searchable text field.
    ///
    /// Args:
    ///     name: Field name.
    ///     stored: Whether the original value is retrievable (default True).
    ///     indexed: Whether the field is searchable (default True).
    ///     term_vectors: Whether term positions are stored, required by
    ///         phrase and span queries over this field (default True).
    ///     doc_values: Whether the value is also copied into DocValues,
    ///         the column-oriented store sort/facet/aggregation read
    ///         from (default True). Takes effect only when ``stored`` is
    ///         also True.
    ///     analyzer: Either a string analyzer name (``"standard"``,
    ///         ``"english"``, ``"keyword"``, ``"simple"``, ``"noop"``, or
    ///         a custom name registered via ``add_analyzer``), or a dict
    ///         configuring a parameterized built-in preset such as
    ///         ``{"language": "japanese", "dict": "/var/lib/lindera/ipadic"}``.
    ///     multi_valued: When True, the field accepts a ``list[str]``; a
    ///         term query matches if any element contains the term, and a
    ///         phrase query never spans two elements (Lucene-style). Default
    ///         False.
    ///     position_increment_gap: Positions skipped between the elements
    ///         of a multi-valued field, so a phrase needs a slop of at least
    ///         this value to cross an element boundary (default 100; ``0``
    ///         numbers the elements as if concatenated). Ignored unless
    ///         ``multi_valued`` is True.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields, or if ``analyzer`` is
    ///         neither a str nor a dict, or is a dict that names an
    ///         unsupported ``language`` or lacks a required key.
    #[pyo3(signature = (name, *, stored=true, indexed=true, term_vectors=true, doc_values=true, multi_valued=false, position_increment_gap=laurus::lexical::core::field::DEFAULT_POSITION_INCREMENT_GAP, analyzer=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_text_field(
        &mut self,
        py: Python<'_>,
        name: &str,
        stored: bool,
        indexed: bool,
        term_vectors: bool,
        doc_values: bool,
        multi_valued: bool,
        position_increment_gap: u32,
        analyzer: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let analyzer = analyzer
            .map(|obj| analyzer_spec_from_py(py, obj))
            .transpose()?;
        self.insert_field(
            name,
            FieldOption::Text(TextOption {
                indexed,
                stored,
                multi_valued,
                position_increment_gap,
                term_vectors,
                doc_values,
                analyzer,
            }),
        )
    }

    /// Add an integer (i64) field.
    ///
    /// Args:
    ///     name: Field name.
    ///     stored: Whether the value is retrievable (default True).
    ///     indexed: Whether the field is searchable for range queries
    ///         (default True).
    ///     multi_valued: When True, the field accepts arrays of integers
    ///         and range queries match if any value satisfies the
    ///         predicate (Lucene-style "any match"). Default False.
    ///     doc_values: Whether the value is also copied into DocValues
    ///         (default True). Takes effect only when ``stored`` is also
    ///         True.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields.
    #[pyo3(signature = (name, *, stored=true, indexed=true, multi_valued=false, doc_values=true))]
    pub fn add_integer_field(
        &mut self,
        name: &str,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PyResult<()> {
        self.insert_field(
            name,
            FieldOption::Integer(IntegerOption {
                indexed,
                stored,
                multi_valued,
                doc_values,
            }),
        )
    }

    /// Add a float (f64) field.
    ///
    /// Args:
    ///     name: Field name.
    ///     stored: Whether the value is retrievable (default True).
    ///     indexed: Whether the field is searchable for range queries
    ///         (default True).
    ///     multi_valued: When True, the field accepts arrays of floats
    ///         and range queries match if any value satisfies the
    ///         predicate (Lucene-style "any match"). Default False.
    ///     doc_values: Whether the value is also copied into DocValues
    ///         (default True). Takes effect only when ``stored`` is also
    ///         True.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields.
    #[pyo3(signature = (name, *, stored=true, indexed=true, multi_valued=false, doc_values=true))]
    pub fn add_float_field(
        &mut self,
        name: &str,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PyResult<()> {
        self.insert_field(
            name,
            FieldOption::Float(FloatOption {
                indexed,
                stored,
                multi_valued,
                doc_values,
            }),
        )
    }

    /// Add a boolean field.
    ///
    /// Args:
    ///     multi_valued: When True, the field accepts a ``list[bool]`` and a
    ///         term query / DSL ``flags:true`` matches if any element is
    ///         ``True`` (Lucene-style "any match"). Default False.
    ///     doc_values: Whether the value is also copied into DocValues
    ///         (default True). Takes effect only when ``stored`` is also
    ///         True.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields.
    #[pyo3(signature = (name, *, stored=true, indexed=true, multi_valued=false, doc_values=true))]
    pub fn add_boolean_field(
        &mut self,
        name: &str,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PyResult<()> {
        self.insert_field(
            name,
            FieldOption::Boolean(BooleanOption {
                indexed,
                stored,
                multi_valued,
                doc_values,
            }),
        )
    }

    /// Add a date/time field.
    ///
    /// Args:
    ///     multi_valued: When True, the field accepts a list of datetimes
    ///         (``datetime`` objects or RFC 3339 strings) and range
    ///         queries match if any instant satisfies the predicate
    ///         (Lucene-style "any match"). Default False.
    ///     doc_values: Whether the value is also copied into DocValues
    ///         (default True). Takes effect only when ``stored`` is also
    ///         True.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields.
    #[pyo3(signature = (name, *, stored=true, indexed=true, multi_valued=false, doc_values=true))]
    pub fn add_datetime_field(
        &mut self,
        name: &str,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PyResult<()> {
        self.insert_field(
            name,
            FieldOption::DateTime(DateTimeOption {
                indexed,
                stored,
                multi_valued,
                doc_values,
            }),
        )
    }

    /// Add a geographic coordinate field (latitude, longitude).
    ///
    /// Args:
    ///     multi_valued: When True, the field accepts a list of
    ///         ``(lat, lon)`` tuples and distance / bounding-box queries
    ///         match if any point satisfies the predicate (Lucene-style
    ///         "any match"), scoring the document by its closest point.
    ///         Default False.
    ///     doc_values: Whether the value is also copied into DocValues
    ///         (default True). Takes effect only when ``stored`` is also
    ///         True.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields.
    #[pyo3(signature = (name, *, stored=true, indexed=true, multi_valued=false, doc_values=true))]
    pub fn add_geo_field(
        &mut self,
        name: &str,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PyResult<()> {
        self.insert_field(
            name,
            FieldOption::Geo(GeoOption {
                indexed,
                stored,
                multi_valued,
                doc_values,
            }),
        )
    }

    /// Add a 3D ECEF Cartesian point field (x, y, z in meters).
    ///
    /// Values are submitted as a 3-tuple `(x, y, z)` of floats and are
    /// queryable via `Geo3dDistanceQuery`, `Geo3dBoundingBoxQuery`, and
    /// `Geo3dNearestQuery`. See the conceptual docs at
    /// `docs/src/concepts/geo3d.md` for the coordinate system.
    ///
    /// Args:
    ///     multi_valued: When True, the field accepts a list of
    ///         ``(x, y, z)`` tuples and the geo3d queries match if any
    ///         point satisfies the predicate (Lucene-style "any match"),
    ///         scoring the document by its closest point. Default False.
    ///     doc_values: Whether the value is also copied into DocValues
    ///         (default True). Takes effect only when ``stored`` is also
    ///         True.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields.
    #[pyo3(signature = (name, *, stored=true, indexed=true, multi_valued=false, doc_values=true))]
    pub fn add_geo3d_field(
        &mut self,
        name: &str,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PyResult<()> {
        self.insert_field(
            name,
            FieldOption::Geo3d(Geo3dOption {
                indexed,
                stored,
                multi_valued,
                doc_values,
            }),
        )
    }

    /// Add a binary data field.
    ///
    /// Args:
    ///     name: Field name.
    ///     stored: Whether the value is retrievable (default True).
    ///     multi_valued: When True, the field accepts a list of binary
    ///         values (each with its own optional MIME type). Default False.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields.
    #[pyo3(signature = (name, *, stored=true, multi_valued=false))]
    pub fn add_bytes_field(
        &mut self,
        name: &str,
        stored: bool,
        multi_valued: bool,
    ) -> PyResult<()> {
        self.insert_field(
            name,
            FieldOption::Bytes(BytesOption {
                stored,
                multi_valued,
            }),
        )
    }

    /// Add an HNSW approximate nearest-neighbor vector index field.
    ///
    /// Args:
    ///     name: Field name.
    ///     dimension: Vector dimensionality.
    ///     distance: Distance metric — "cosine" (default), "euclidean", "dot_product".
    ///     m: HNSW branching factor (default 16).
    ///     ef_construction: Build-time expansion factor (default 200).
    ///     default_ef_search: Schema-level default for the search-time
    ///         `ef_search` candidate-list size (Issue #644). When unset,
    ///         the searcher uses an internal fallback of 50. Per-query
    ///         overrides via the search request still take precedence.
    ///     quantizer: Vector quantizer — "scalar_8bit" (default) or
    ///         "product_quantization". Product quantization requires
    ///         `subvector_count`.
    ///     subvector_count: Number of PQ sub-vectors. Required when
    ///         `quantizer="product_quantization"` and must divide
    ///         `dimension`; rejected for other quantizers.
    ///     rerank_storage: Stage-2 rerank sidecar — None (default) keeps
    ///         the int8-only segment, "f32" stores full-precision vectors
    ///         in a `*.hnsw.f32` sidecar for exact rerank distances.
    ///     embedder: Optional embedder name registered via `add_embedder`.
    ///         When set, text payloads are automatically embedded by the Rust engine.
    ///     pq_codebook_path: Storage-relative file name of a shared PQ
    ///         codebook (Issue #631), trained once via the
    ///         `laurus train pq-codebook` CLI command. Only meaningful with
    ///         `quantizer="product_quantization"`; commits then encode
    ///         against the pre-trained codebook instead of re-training
    ///         k-means per segment. None (default) keeps per-segment training.
    ///     base_weight: This field's relative scoring priority when
    ///         searched alongside other vector fields (Issue #1084).
    ///         Defaults to 1.0. Only matters when a query targets two or
    ///         more specific vector fields at once; has no effect on the
    ///         lexical-vs-vector balance of a hybrid search.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields, if ``distance``,
    ///         ``quantizer`` or ``rerank_storage`` is not a recognized name,
    ///         or if ``subvector_count`` is missing with
    ///         ``quantizer="product_quantization"`` or given with any other
    ///         quantizer.
    #[pyo3(signature = (name, dimension, *, distance="cosine", m=16, ef_construction=200, default_ef_search=None, quantizer=None, subvector_count=None, rerank_storage=None, embedder=None, pq_codebook_path=None, base_weight=1.0))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_hnsw_field(
        &mut self,
        name: &str,
        dimension: usize,
        distance: &str,
        m: usize,
        ef_construction: usize,
        default_ef_search: Option<usize>,
        quantizer: Option<String>,
        subvector_count: Option<usize>,
        rerank_storage: Option<String>,
        embedder: Option<String>,
        pq_codebook_path: Option<String>,
        base_weight: f32,
    ) -> PyResult<()> {
        let opt = HnswOption {
            dimension,
            distance: parse_distance(distance)?,
            m,
            ef_construction,
            default_ef_search,
            quantizer: parse_quantizer(quantizer.as_deref(), subvector_count)?,
            rerank_storage: parse_rerank_storage(rerank_storage.as_deref())?,
            embedder,
            pq_codebook_path,
            base_weight,
        };
        self.insert_field(name, FieldOption::Hnsw(opt))
    }

    /// Add a flat (brute-force) vector index field.
    ///
    /// Args:
    ///     name: Field name.
    ///     dimension: Vector dimensionality.
    ///     distance: Distance metric — "cosine" (default), "euclidean", "dot_product".
    ///     embedder: Optional embedder name registered via `add_embedder`.
    ///         When set, text payloads are automatically embedded by the Rust engine.
    ///     base_weight: This field's relative scoring priority when
    ///         searched alongside other vector fields (Issue #1084).
    ///         Defaults to 1.0. See `add_hnsw_field` for the full contract.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields, or if ``distance`` is
    ///         not a recognized metric.
    #[pyo3(signature = (name, dimension, *, distance="cosine", embedder=None, base_weight=1.0))]
    pub fn add_flat_field(
        &mut self,
        name: &str,
        dimension: usize,
        distance: &str,
        embedder: Option<String>,
        base_weight: f32,
    ) -> PyResult<()> {
        let opt = FlatOption {
            dimension,
            distance: parse_distance(distance)?,
            embedder,
            base_weight,
            ..Default::default()
        };
        self.insert_field(name, FieldOption::Flat(opt))
    }

    /// Add a multi-vector field holding each document's token vectors for
    /// late-interaction rescoring (e.g. ColBERT embeddings).
    ///
    /// The field has no ANN index and cannot be queried directly; a search
    /// rescores its top results against it with `LateInteractionRescore`.
    /// A document gives it a list of equal-length float lists, or text when
    /// ``embedder`` names a token-level embedder (`"candle_colbert"`).
    ///
    /// Args:
    ///     name: Field name.
    ///     dimension: Dimensionality of every token vector.
    ///     distance: ``"cosine"`` (default; vectors are L2-normalized when
    ///         written) or ``"dot_product"``.
    ///     embedder: Optional token-level embedder registered via
    ///         `add_embedder`.
    ///
    /// Raises:
    ///     ValueError: if ``name`` is reserved, ``dimension`` is zero, or
    ///         ``distance`` is neither cosine nor dot product.
    #[pyo3(signature = (name, dimension, *, distance="cosine", embedder=None))]
    pub fn add_multi_vector_field(
        &mut self,
        name: &str,
        dimension: usize,
        distance: &str,
        embedder: Option<String>,
    ) -> PyResult<()> {
        let mut opt = MultiVectorOption::new(dimension).distance(parse_distance(distance)?);
        opt.embedder = embedder;
        opt.validate(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.insert_field(name, FieldOption::MultiVector(opt))
    }

    /// Add an IVF (Inverted File Index) approximate nearest-neighbor vector field.
    ///
    /// Args:
    ///     name: Field name.
    ///     dimension: Vector dimensionality.
    ///     distance: Distance metric — "cosine" (default), "euclidean", "dot_product".
    ///     n_clusters: Number of Voronoi clusters (default 100).
    ///     n_probe: Number of clusters to probe at search time (default 1).
    ///     embedder: Optional embedder name registered via `add_embedder`.
    ///         When set, text payloads are automatically embedded by the Rust engine.
    ///     base_weight: This field's relative scoring priority when
    ///         searched alongside other vector fields (Issue #1084).
    ///         Defaults to 1.0. See `add_hnsw_field` for the full contract.
    ///
    /// Raises:
    ///     ValueError: if ``name`` starts with ``_`` (other than ``_id``),
    ///         which is reserved for system fields, or if ``distance`` is
    ///         not a recognized metric.
    #[pyo3(signature = (name, dimension, *, distance="cosine", n_clusters=100, n_probe=1, embedder=None, base_weight=1.0))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_ivf_field(
        &mut self,
        name: &str,
        dimension: usize,
        distance: &str,
        n_clusters: usize,
        n_probe: usize,
        embedder: Option<String>,
        base_weight: f32,
    ) -> PyResult<()> {
        let opt = IvfOption {
            dimension,
            distance: parse_distance(distance)?,
            n_clusters,
            n_probe,
            embedder,
            base_weight,
            ..Default::default()
        };
        self.insert_field(name, FieldOption::Ivf(opt))
    }

    /// Register a named embedder definition in the schema.
    ///
    /// The embedder can then be referenced by name from vector field options
    /// (e.g. `add_hnsw_field(..., embedder="my-bert")`).
    ///
    /// The `config` dict must have a `"type"` key selecting the backend:
    ///
    /// | type               | required keys | optional keys                              | feature flag            |
    /// |--------------------|---------------|--------------------------------------------|-------------------------|
    /// | `"precomputed"`    | —             | —                                          | (always available)      |
    /// | `"candle_bert"`    | `"model"`     | —                                          | `embeddings-candle`     |
    /// | `"candle_clip"`    | `"model"`     | —                                          | `embeddings-multimodal` |
    /// | `"openai"`         | `"model"`     | —                                          | `embeddings-openai`     |
    /// | `"candle_colbert"` | `"model"`     | `"revision"`, `"query_maxlen"`, `"doc_maxlen"` | `embeddings-candle`     |
    ///
    /// `"candle_colbert"` produces token vectors and only serves a
    /// multi-vector field (`add_multi_vector_field`).
    ///
    /// Args:
    ///     name: Unique embedder name referenced from vector fields.
    ///     config: Dict describing the embedder, e.g.
    ///         `{"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"}`.
    ///
    /// Raises:
    ///     ValueError: if ``config`` is not a dict, has no or an unknown
    ///         ``"type"``, or lacks a required key.
    ///
    /// Example:
    ///     ```python
    ///     schema.add_embedder("bert", {"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"})
    ///     schema.add_hnsw_field("embedding", dimension=384, embedder="bert")
    ///     ```
    pub fn add_embedder(&mut self, name: &str, config: &Bound<PyAny>) -> PyResult<()> {
        if !config.is_instance_of::<PyDict>() {
            return Err(PyValueError::new_err(
                "embedder config must be a dict, e.g. {\"type\": \"candle_bert\", \"model\": \"...\"}",
            ));
        }
        // The core definition's serde decides the accepted types and keys,
        // so every embedder type the engine knows is available here.
        let definition: EmbedderDefinition = serde_json::from_value(py_to_json_value(config)?)
            .map_err(|e| PyValueError::new_err(format!("invalid embedder config: {e}")))?;
        self.inner.embedders.insert(name.to_string(), definition);
        Ok(())
    }

    /// Register a custom analyzer definition composed of a tokenizer and
    /// optional char/token filter chains.
    ///
    /// Each of `tokenizer`, `char_filters`, and `token_filters` uses the
    /// same `{"type": "..."}`-tagged dict shape as the schema TOML/JSON
    /// format (see [`Schema.from_toml`]), so a definition can be copied
    /// verbatim between a `schema.toml` file and Python code.
    ///
    /// | tokenizer type   | required keys                | optional keys  |
    /// |-------------------|-------------------------------|-----------------|
    /// | `"whitespace"`    | —                              | —               |
    /// | `"unicode_word"`  | —                              | —               |
    /// | `"regex"`         | —                              | `pattern`, `gaps` |
    /// | `"ngram"`         | `min_gram`, `max_gram`         | —               |
    /// | `"lindera"`       | `mode`, `dict`                 | `user_dict`     |
    /// | `"whole"`         | —                              | —               |
    ///
    /// | char filter type              | required keys           |
    /// |----------------------------------|-------------------------|
    /// | `"unicode_normalization"`        | `form`                  |
    /// | `"pattern_replace"`              | `pattern`, `replacement` |
    /// | `"mapping"`                      | `mapping`               |
    /// | `"japanese_iteration_mark"`      | — (optional `kanji`, `kana`) |
    ///
    /// | token filter type   | required keys | optional keys |
    /// |-----------------------|----------------|-----------------|
    /// | `"lowercase"`          | —              | —               |
    /// | `"stop"`               | —              | `words`         |
    /// | `"stem"`               | —              | `stem_type`     |
    /// | `"boost"`              | `boost`        | —               |
    /// | `"limit"`              | `limit`        | —               |
    /// | `"strip"`              | —              | —               |
    /// | `"remove_empty"`       | —              | —               |
    /// | `"flatten_graph"`      | —              | —               |
    ///
    /// Args:
    ///     name: Unique analyzer name, referenced from
    ///         ``add_text_field(analyzer=...)``.
    ///     tokenizer: Dict describing the tokenizer, e.g.
    ///         ``{"type": "ngram", "min_gram": 2, "max_gram": 3}``.
    ///     char_filters: Optional list of dicts applied to raw text before
    ///         tokenization (default: none).
    ///     token_filters: Optional list of dicts applied to the token
    ///         stream after tokenization (default: none).
    ///
    /// Raises:
    ///     ValueError: if ``name`` is reserved for a built-in analyzer
    ///         (``standard``, ``keyword``, ``english``, ``simple``,
    ///         ``noop``), or if any component has an unknown ``type`` or is
    ///         missing a required key.
    ///
    /// Note:
    ///     Semantic validity (e.g. a malformed regex pattern, or
    ///     ``min_gram > max_gram``) is not checked here; it is validated
    ///     when the analyzer is compiled, which happens when a
    ///     ``laurus.Index`` is built from this schema.
    ///
    /// Example:
    ///     ```python
    ///     schema.add_analyzer(
    ///         "ja_ipadic",
    ///         {"type": "lindera", "mode": "normal", "dict": "/var/lib/lindera/ipadic"},
    ///         char_filters=[
    ///             {"type": "unicode_normalization", "form": "nfkc"},
    ///             {"type": "japanese_iteration_mark"},
    ///         ],
    ///         token_filters=[{"type": "lowercase"}],
    ///     )
    ///     schema.add_text_field("title", analyzer="ja_ipadic")
    ///     ```
    #[pyo3(signature = (name, tokenizer, *, char_filters=None, token_filters=None))]
    pub fn add_analyzer(
        &mut self,
        name: &str,
        tokenizer: &Bound<PyAny>,
        char_filters: Option<&Bound<PyAny>>,
        token_filters: Option<&Bound<PyAny>>,
    ) -> PyResult<()> {
        laurus::analysis::analyzer::registry::validate_analyzer_name(name)
            .map_err(crate::errors::laurus_err)?;
        let definition = AnalyzerDefinition {
            char_filters: filter_list_from_py(char_filters, "char_filters", char_filter_from_py)?,
            tokenizer: tokenizer_from_py(tokenizer)?,
            token_filters: filter_list_from_py(
                token_filters,
                "token_filters",
                token_filter_from_py,
            )?,
        };
        self.inner.analyzers.insert(name.to_string(), definition);
        Ok(())
    }

    /// Return the names of custom analyzers registered in this schema, via
    /// [`Schema.add_analyzer`] or loaded from TOML.
    pub fn analyzer_names(&self) -> Vec<String> {
        self.inner.analyzers.keys().cloned().collect()
    }

    /// Parse a schema from a TOML string, using the same format accepted
    /// by ``laurus-cli create index --schema``.
    ///
    /// Raises:
    ///     ValueError: if the TOML is malformed or does not match the
    ///         schema shape.
    #[staticmethod]
    pub fn from_toml(toml_str: &str) -> PyResult<Self> {
        let inner = Schema::from_toml(toml_str).map_err(schema_toml_err)?;
        Ok(Self { inner })
    }

    /// Load a schema from a TOML file, e.g. one written by
    /// ``laurus-cli create index --schema`` or by
    /// [`Schema.to_toml_file`].
    ///
    /// Args:
    ///     path: Filesystem path (``str`` or ``os.PathLike``).
    ///
    /// Raises:
    ///     FileNotFoundError: if the file does not exist.
    ///     ValueError: if the file's contents are not valid schema TOML.
    #[staticmethod]
    pub fn from_toml_file(py: Python, path: PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| -> PyResult<Schema> {
            let content = std::fs::read_to_string(&path).map_err(|e| io_err_with_path(&path, e))?;
            Schema::from_toml(&content).map_err(schema_toml_err)
        })?;
        Ok(Self { inner })
    }

    /// Serialize this schema to a TOML string, in the same format
    /// ``laurus-cli create index --schema`` accepts — so an index created
    /// from Python can also be opened with ``laurus-cli``.
    ///
    /// Note:
    ///     Table order in the output is stable and sorted by key across
    ///     calls (Issue #1060). Comparing parsed structures rather than
    ///     raw text is still recommended when round-tripping, since it
    ///     doesn't depend on this ordering guarantee.
    pub fn to_toml(&self) -> PyResult<String> {
        self.inner.to_toml().map_err(schema_toml_err)
    }

    /// Write this schema to a TOML file (see [`Schema.to_toml`]).
    pub fn to_toml_file(&self, py: Python, path: PathBuf) -> PyResult<()> {
        let content = self.to_toml()?;
        py.detach(|| std::fs::write(&path, content).map_err(|e| io_err_with_path(&path, e)))
    }

    /// Set the default fields used when no field is specified in a query.
    pub fn set_default_fields(&mut self, fields: Vec<String>) {
        self.inner.default_fields = fields;
    }

    /// Set the policy for fields that are not declared in this schema.
    ///
    /// Args:
    ///     policy: One of ``"strict"``, ``"dynamic"`` (default), or
    ///         ``"ignore"``. Case-insensitive.
    ///
    /// Behaviour:
    ///     * ``"strict"``: reject documents containing undeclared fields.
    ///     * ``"dynamic"``: infer a type for each undeclared field and add
    ///       it to the schema during ingestion. **Warning**: integer fields
    ///       silently truncate incoming float values (e.g. ``3.14`` → ``3``).
    ///     * ``"ignore"``: silently drop undeclared fields.
    ///
    /// Raises:
    ///     ValueError: if ``policy`` is not one of the accepted names.
    pub fn set_dynamic_field_policy(&mut self, policy: &str) -> PyResult<()> {
        let parsed = DynamicFieldPolicy::from_str(policy)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.dynamic_field_policy = parsed;
        Ok(())
    }

    /// Return the currently configured dynamic field policy as a lowercase
    /// string (``"strict"`` / ``"dynamic"`` / ``"ignore"``).
    pub fn dynamic_field_policy(&self) -> &'static str {
        match self.inner.dynamic_field_policy {
            DynamicFieldPolicy::Strict => "strict",
            DynamicFieldPolicy::Dynamic => "dynamic",
            DynamicFieldPolicy::Ignore => "ignore",
        }
    }

    /// Return the list of field names defined in this schema.
    pub fn field_names(&self) -> Vec<String> {
        self.inner.fields.keys().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Schema(fields={:?})",
            self.inner.fields.keys().collect::<Vec<_>>()
        )
    }
}

impl PySchema {
    /// Insert a field after rejecting a name reserved for system fields.
    fn insert_field(&mut self, name: &str, option: FieldOption) -> PyResult<()> {
        laurus::validate_field_name(name).map_err(crate::errors::laurus_err)?;
        self.inner.fields.insert(name.to_string(), option);
        Ok(())
    }
}
