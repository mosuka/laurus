//! Python wrapper for the Laurus [`Schema`] type.

use std::path::PathBuf;
use std::str::FromStr;

use laurus::{
    AnalyzerDefinition, AnalyzerSpec, BooleanOption, BuiltinAnalyzerSpec, BytesOption,
    CharFilterConfig, DateTimeOption, DistanceMetric, DynamicFieldPolicy, EmbedderDefinition,
    FieldOption, FlatOption, FloatOption, Geo3dOption, GeoOption, HnswOption, IntegerOption,
    IvfOption, QuantizationMethod, RerankStorageKind, Schema, TextOption, TokenFilterConfig,
    TokenizerConfig,
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
    ///     analyzer: Either a string analyzer name (``"standard"``,
    ///         ``"english"``, ``"keyword"``, ``"simple"``, ``"noop"``, or
    ///         a custom name registered via ``add_analyzer``), or a dict
    ///         configuring a parameterized built-in preset such as
    ///         ``{"language": "japanese", "dict": "/var/lib/lindera/ipadic"}``.
    #[pyo3(signature = (name, *, stored=true, indexed=true, term_vectors=true, analyzer=None))]
    pub fn add_text_field(
        &mut self,
        py: Python<'_>,
        name: &str,
        stored: bool,
        indexed: bool,
        term_vectors: bool,
        analyzer: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let analyzer = analyzer
            .map(|obj| analyzer_spec_from_py(py, obj))
            .transpose()?;
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Text(TextOption {
                indexed,
                stored,
                term_vectors,
                doc_values: true,
                analyzer,
            }),
        );
        Ok(())
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
    #[pyo3(signature = (name, *, stored=true, indexed=true, multi_valued=false))]
    pub fn add_integer_field(
        &mut self,
        name: &str,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
    ) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Integer(IntegerOption {
                indexed,
                stored,
                multi_valued,
                doc_values: true,
            }),
        );
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
    #[pyo3(signature = (name, *, stored=true, indexed=true, multi_valued=false))]
    pub fn add_float_field(&mut self, name: &str, stored: bool, indexed: bool, multi_valued: bool) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Float(FloatOption {
                indexed,
                stored,
                multi_valued,
                doc_values: true,
            }),
        );
    }

    /// Add a boolean field.
    #[pyo3(signature = (name, *, stored=true, indexed=true))]
    pub fn add_boolean_field(&mut self, name: &str, stored: bool, indexed: bool) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Boolean(BooleanOption {
                indexed,
                stored,
                doc_values: true,
            }),
        );
    }

    /// Add a date/time field.
    #[pyo3(signature = (name, *, stored=true, indexed=true))]
    pub fn add_datetime_field(&mut self, name: &str, stored: bool, indexed: bool) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::DateTime(DateTimeOption {
                indexed,
                stored,
                doc_values: true,
            }),
        );
    }

    /// Add a geographic coordinate field (latitude, longitude).
    #[pyo3(signature = (name, *, stored=true, indexed=true))]
    pub fn add_geo_field(&mut self, name: &str, stored: bool, indexed: bool) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Geo(GeoOption {
                indexed,
                stored,
                doc_values: true,
            }),
        );
    }

    /// Add a 3D ECEF Cartesian point field (x, y, z in meters).
    ///
    /// Values are submitted as a 3-tuple `(x, y, z)` of floats and are
    /// queryable via `Geo3dDistanceQuery`, `Geo3dBoundingBoxQuery`, and
    /// `Geo3dNearestQuery`. See the conceptual docs at
    /// `docs/src/concepts/geo3d.md` for the coordinate system.
    #[pyo3(signature = (name, *, stored=true, indexed=true))]
    pub fn add_geo3d_field(&mut self, name: &str, stored: bool, indexed: bool) {
        self.inner.fields.insert(
            name.to_string(),
            FieldOption::Geo3d(Geo3dOption {
                indexed,
                stored,
                doc_values: true,
            }),
        );
    }

    /// Add a binary data field.
    #[pyo3(signature = (name, *, stored=true))]
    pub fn add_bytes_field(&mut self, name: &str, stored: bool) {
        self.inner
            .fields
            .insert(name.to_string(), FieldOption::Bytes(BytesOption { stored }));
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
        self.inner
            .fields
            .insert(name.to_string(), FieldOption::Hnsw(opt));
        Ok(())
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
        self.inner
            .fields
            .insert(name.to_string(), FieldOption::Flat(opt));
        Ok(())
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
        self.inner
            .fields
            .insert(name.to_string(), FieldOption::Ivf(opt));
        Ok(())
    }

    /// Register a named embedder definition in the schema.
    ///
    /// The embedder can then be referenced by name from vector field options
    /// (e.g. `add_hnsw_field(..., embedder="my-bert")`).
    ///
    /// The `config` dict must have a `"type"` key selecting the backend:
    ///
    /// | type            | required keys | feature flag          |
    /// |-----------------|---------------|-----------------------|
    /// | `"precomputed"` | —             | (always available)    |
    /// | `"candle_bert"` | `"model"`     | `embeddings-candle`   |
    /// | `"candle_clip"` | `"model"`     | `embeddings-multimodal` |
    /// | `"openai"`      | `"model"`     | `embeddings-openai`   |
    ///
    /// Args:
    ///     name: Unique embedder name referenced from vector fields.
    ///     config: Dict describing the embedder, e.g.
    ///         `{"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"}`.
    ///
    /// Example:
    ///     ```python
    ///     schema.add_embedder("bert", {"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"})
    ///     schema.add_hnsw_field("embedding", dimension=384, embedder="bert")
    ///     ```
    pub fn add_embedder(&mut self, name: &str, config: &Bound<PyAny>) -> PyResult<()> {
        let dict = config.extract::<Bound<PyDict>>().map_err(|_| {
            PyValueError::new_err("embedder config must be a dict, e.g. {\"type\": \"candle_bert\", \"model\": \"...\"}")
        })?;
        let dict = &dict;

        let embedder_type: String = dict
            .get_item("type")?
            .ok_or_else(|| PyValueError::new_err("embedder config must have a 'type' key"))?
            .extract()?;

        let definition = match embedder_type.as_str() {
            "precomputed" => EmbedderDefinition::Precomputed,
            "candle_bert" => {
                let model: String = dict
                    .get_item("model")?
                    .ok_or_else(|| {
                        PyValueError::new_err("candle_bert embedder requires a 'model' key")
                    })?
                    .extract()?;
                EmbedderDefinition::CandleBert { model }
            }
            "candle_clip" => {
                let model: String = dict
                    .get_item("model")?
                    .ok_or_else(|| {
                        PyValueError::new_err("candle_clip embedder requires a 'model' key")
                    })?
                    .extract()?;
                EmbedderDefinition::CandleClip { model }
            }
            "openai" => {
                let model: String = dict
                    .get_item("model")?
                    .ok_or_else(|| PyValueError::new_err("openai embedder requires a 'model' key"))?
                    .extract()?;
                EmbedderDefinition::Openai { model }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown embedder type: '{}'. Valid types: precomputed, candle_bert, candle_clip, openai",
                    other
                )));
            }
        };

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
    ///     ValueError: if any component has an unknown ``type`` or is
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
