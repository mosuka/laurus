//! PHP wrapper for the Laurus [`Schema`] type.

use std::cell::RefCell;
use std::str::FromStr;

use ext_php_rs::convert::FromZval;
use ext_php_rs::prelude::*;
use ext_php_rs::types::ZendHashTable;
use laurus::{
    AnalyzerDefinition, BooleanOption, BytesOption, CharFilterConfig, DateTimeOption,
    DistanceMetric, DynamicFieldPolicy, EmbedderDefinition, FieldOption, FloatOption, Geo3dOption,
    GeoOption, HnswOption, IntegerOption, IvfOption, QuantizationMethod, RerankStorageKind, Schema,
    TextOption, TokenFilterConfig, TokenizerConfig,
};

use crate::convert::hashtable_to_json_value;
use crate::errors::{io_err_with_path, laurus_err};

/// Parse a distance metric string into [`DistanceMetric`].
///
/// # Arguments
///
/// * `s` - Distance metric name (e.g. "cosine", "euclidean", "dot_product").
///
/// # Returns
///
/// The corresponding `DistanceMetric`.
fn parse_distance(s: &str) -> PhpResult<DistanceMetric> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(DistanceMetric::Cosine),
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "dot_product" | "dot" => Ok(DistanceMetric::DotProduct),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "angular" => Ok(DistanceMetric::Angular),
        other => Err(format!(
            "Unknown distance metric: '{}'. Valid: cosine, euclidean, dot_product, manhattan, angular",
            other
        )
        .into()),
    }
}

/// Parse a quantizer name plus optional `subvector_count` into a
/// [`QuantizationMethod`].
///
/// Accepts `"scalar_8bit"` / `"scalar"` (the default when `name` is
/// `None`) and `"product_quantization"` / `"pq"`. Product quantization
/// requires a `subvector_count` (which must divide the field dimension —
/// validated by the core at index-build time); supplying it for any other
/// quantizer is rejected so an incoherent configuration cannot silently
/// reach the core.
fn parse_quantizer(
    name: Option<&str>,
    subvector_count: Option<usize>,
) -> PhpResult<QuantizationMethod> {
    match name.map(|s| s.to_lowercase()).as_deref() {
        None | Some("scalar_8bit") | Some("scalar") => {
            if subvector_count.is_some() {
                return Err(
                    "subvector_count is only valid with quantizer 'product_quantization'".into(),
                );
            }
            Ok(QuantizationMethod::Scalar8Bit)
        }
        Some("product_quantization") | Some("pq") => match subvector_count {
            Some(subvector_count) => {
                Ok(QuantizationMethod::ProductQuantization { subvector_count })
            }
            None => Err("quantizer 'product_quantization' requires subvector_count \
                         (must divide the field dimension)"
                .into()),
        },
        Some(other) => Err(format!(
            "Unknown quantizer: '{other}'. Valid: scalar_8bit, product_quantization"
        )
        .into()),
    }
}

/// Parse a rerank-storage name into an optional [`RerankStorageKind`].
///
/// `None` (the default) keeps the Stage-1 int8-only segment; `"f32"`
/// enables the Stage-2 full-precision rerank sidecar (`*.hnsw.f32`).
fn parse_rerank_storage(name: Option<&str>) -> PhpResult<Option<RerankStorageKind>> {
    match name.map(|s| s.to_lowercase()).as_deref() {
        None => Ok(None),
        Some("f32") => Ok(Some(RerankStorageKind::F32)),
        Some(other) => Err(format!("Unknown rerank_storage: '{other}'. Valid: f32").into()),
    }
}

/// Helper to extract a string from a [`ZendHashTable`] by key.
fn ht_get_string(ht: &ZendHashTable, key: &str) -> PhpResult<String> {
    let zv = ht.get(key).ok_or(format!("missing key '{key}'"))?;
    String::from_zval(zv).ok_or_else(|| format!("'{key}' must be a string").into())
}

/// Convert a PHP array into a [`TokenizerConfig`], using the same
/// `{"type": "..."}`-tagged shape as the schema TOML/JSON format.
fn tokenizer_from_ht(ht: &ZendHashTable) -> PhpResult<TokenizerConfig> {
    let value = hashtable_to_json_value(ht)?;
    serde_json::from_value(value).map_err(|e| format!("invalid tokenizer: {e}").into())
}

/// Convert a PHP array into a [`CharFilterConfig`].
fn char_filter_from_ht(ht: &ZendHashTable, index: usize) -> PhpResult<CharFilterConfig> {
    let value = hashtable_to_json_value(ht)?;
    serde_json::from_value(value).map_err(|e| format!("invalid charFilters[{index}]: {e}").into())
}

/// Convert a PHP array into a [`TokenFilterConfig`].
fn token_filter_from_ht(ht: &ZendHashTable, index: usize) -> PhpResult<TokenFilterConfig> {
    let value = hashtable_to_json_value(ht)?;
    serde_json::from_value(value).map_err(|e| format!("invalid tokenFilters[{index}]: {e}").into())
}

/// Convert an optional PHP array of arrays into a `Vec<T>`, defaulting to an
/// empty vector when `None` (mirroring the core's `#[serde(default)]` on
/// `AnalyzerDefinition::char_filters`/`token_filters`).
fn filter_list_from_ht<T>(
    list: Option<&ZendHashTable>,
    label: &str,
    convert: impl Fn(&ZendHashTable, usize) -> PhpResult<T>,
) -> PhpResult<Vec<T>> {
    let Some(list) = list else {
        return Ok(Vec::new());
    };
    list.values()
        .enumerate()
        .map(|(i, v)| {
            let item = v
                .array()
                .ok_or_else(|| format!("{label}[{i}] must be an array"))?;
            convert(item, i)
        })
        .collect()
}

/// PHP-facing schema builder (`Laurus\Schema`).
///
/// Uses `RefCell` for interior mutability since ext-php-rs methods receive `&self`.
#[php_class]
#[php(name = "Laurus\\Schema")]
pub struct PhpSchema {
    pub inner: RefCell<Schema>,
}

#[php_impl]
impl PhpSchema {
    /// Create a new empty schema.
    pub fn __construct() -> Self {
        Self {
            inner: RefCell::new(Schema::new()),
        }
    }

    /// Add a full-text searchable text field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the original value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    /// * `term_vectors` - Whether term positions are stored, required by
    ///   phrase and span queries over this field (default: true).
    /// * `doc_values` - Whether the value is also copied into DocValues,
    ///   the column-oriented store sort/facet/aggregation read from
    ///   (default: true). Takes effect only when `stored` is also true.
    /// * `analyzer` - Optional analyzer name. For parameter-less built-in
    ///   analyzers (`"standard"`, `"english"`, `"keyword"`, `"simple"`,
    ///   `"noop"`) pass the name directly. For parameterized presets such
    ///   as the Japanese analyzer (which needs a Lindera dictionary path),
    ///   register a custom analyzer via `addAnalyzer` and reference it by
    ///   name.
    /// * `multi_valued` - When true, the field accepts a sequential array of
    ///   strings; a term query matches if any element contains the term,
    ///   and a phrase query never spans two elements (Lucene-style).
    ///   Default: false. Appended after `analyzer` so existing positional
    ///   callers keep working.
    /// * `position_increment_gap` - Positions skipped between the elements
    ///   of a multi-valued field, so a phrase needs a slop of at least this
    ///   value to cross an element boundary (default 100; `0` numbers the
    ///   elements as if concatenated). Ignored unless `multi_valued` is
    ///   true.
    ///
    /// # Errors
    ///
    /// Throws `\Exception` if `position_increment_gap` is outside
    /// `0..=4294967295`, or `\ValueError` if `name` starts with `_` (other
    /// than `_id`), which is reserved for system fields.
    #[php(defaults(
        stored = true,
        indexed = true,
        term_vectors = true,
        doc_values = true,
        multi_valued = false,
        position_increment_gap = 100
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_text_field(
        &self,
        name: String,
        stored: bool,
        indexed: bool,
        term_vectors: bool,
        doc_values: bool,
        analyzer: Option<String>,
        multi_valued: bool,
        position_increment_gap: i64,
    ) -> PhpResult<()> {
        let position_increment_gap = u32::try_from(position_increment_gap)
            .map_err(|_| "position_increment_gap must be between 0 and 4294967295")?;
        self.insert_field(
            name,
            FieldOption::Text(TextOption {
                indexed,
                stored,
                multi_valued,
                position_increment_gap,
                term_vectors,
                doc_values,
                analyzer: analyzer.map(laurus::AnalyzerSpec::Named),
            }),
        )
    }

    /// Add an integer (i64) field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    /// * `multi_valued` - When true, the field accepts arrays of integers
    ///   and range queries match if any value satisfies the predicate
    ///   (Lucene-style "any match"). Default: false.
    /// * `doc_values` - Whether the value is also copied into DocValues
    ///   (default: true). Takes effect only when `stored` is also true.
    ///
    /// # Errors
    ///
    /// Throws `\ValueError` if `name` starts with `_` (other than `_id`),
    /// which is reserved for system fields.
    #[php(defaults(stored = true, indexed = true, multi_valued = false, doc_values = true))]
    pub fn add_integer_field(
        &self,
        name: String,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PhpResult<()> {
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
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    /// * `multi_valued` - When true, the field accepts arrays of floats
    ///   and range queries match if any value satisfies the predicate
    ///   (Lucene-style "any match"). Default: false.
    /// * `doc_values` - Whether the value is also copied into DocValues
    ///   (default: true). Takes effect only when `stored` is also true.
    ///
    /// # Errors
    ///
    /// Throws `\ValueError` if `name` starts with `_` (other than `_id`),
    /// which is reserved for system fields.
    #[php(defaults(stored = true, indexed = true, multi_valued = false, doc_values = true))]
    pub fn add_float_field(
        &self,
        name: String,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PhpResult<()> {
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
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    /// * `multi_valued` - When true, the field accepts an array of bools and
    ///   a term query / DSL `flags:true` matches if any element is `true`
    ///   (Lucene-style "any match"). Default: false.
    /// * `doc_values` - Whether the value is also copied into DocValues
    ///   (default: true). Takes effect only when `stored` is also true.
    ///
    /// # Errors
    ///
    /// Throws `\ValueError` if `name` starts with `_` (other than `_id`),
    /// which is reserved for system fields.
    #[php(defaults(stored = true, indexed = true, multi_valued = false, doc_values = true))]
    pub fn add_boolean_field(
        &self,
        name: String,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PhpResult<()> {
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
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    /// * `multi_valued` - When true, the field accepts an array of RFC 3339
    ///   strings and range queries match if any instant satisfies the
    ///   predicate (Lucene-style "any match"). Default: false.
    /// * `doc_values` - Whether the value is also copied into DocValues
    ///   (default: true). Takes effect only when `stored` is also true.
    ///
    /// # Errors
    ///
    /// Throws `\ValueError` if `name` starts with `_` (other than `_id`),
    /// which is reserved for system fields.
    #[php(defaults(stored = true, indexed = true, multi_valued = false, doc_values = true))]
    pub fn add_datetime_field(
        &self,
        name: String,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PhpResult<()> {
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
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    /// * `multi_valued` - When true, the field accepts an array of
    ///   `["lat" => .., "lon" => ..]` arrays and distance / bounding-box
    ///   queries match if any point satisfies the predicate (Lucene-style
    ///   "any match"), scoring the document by its closest point. Default:
    ///   false.
    /// * `doc_values` - Whether the value is also copied into DocValues
    ///   (default: true). Takes effect only when `stored` is also true.
    ///
    /// # Errors
    ///
    /// Throws `\ValueError` if `name` starts with `_` (other than `_id`),
    /// which is reserved for system fields.
    #[php(defaults(stored = true, indexed = true, multi_valued = false, doc_values = true))]
    pub fn add_geo_field(
        &self,
        name: String,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PhpResult<()> {
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
    /// Values are submitted as an associative array `["x" => ..., "y" => ...,
    /// "z" => ...]` and are queryable via `Geo3dDistanceQuery`,
    /// `Geo3dBoundingBoxQuery`, and `Geo3dNearestQuery`. See the conceptual
    /// docs at `docs/src/concepts/geo3d.md`.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    /// * `multi_valued` - When true, the field accepts an array of
    ///   `["x" => .., "y" => .., "z" => ..]` arrays and the geo3d queries
    ///   match if any point satisfies the predicate (Lucene-style "any
    ///   match"), scoring the document by its closest point. Default: false.
    /// * `doc_values` - Whether the value is also copied into DocValues
    ///   (default: true). Takes effect only when `stored` is also true.
    ///
    /// # Errors
    ///
    /// Throws `\ValueError` if `name` starts with `_` (other than `_id`),
    /// which is reserved for system fields.
    #[php(defaults(stored = true, indexed = true, multi_valued = false, doc_values = true))]
    pub fn add_geo3d_field(
        &self,
        name: String,
        stored: bool,
        indexed: bool,
        multi_valued: bool,
        doc_values: bool,
    ) -> PhpResult<()> {
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
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `multi_valued` - When true, the field accepts an array of binary
    ///   values (each with its own optional MIME type). Default: false.
    ///
    /// # Errors
    ///
    /// Throws `\ValueError` if `name` starts with `_` (other than `_id`),
    /// which is reserved for system fields.
    #[php(defaults(stored = true, multi_valued = false))]
    pub fn add_bytes_field(&self, name: String, stored: bool, multi_valued: bool) -> PhpResult<()> {
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
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `dimension` - Vector dimensionality.
    /// * `distance` - Distance metric (default: "cosine").
    /// * `m` - HNSW branching factor (default: 16).
    /// * `ef_construction` - Build-time expansion factor (default: 200).
    /// * `default_ef_search` - Schema-level default for the search-time
    ///   `ef_search` candidate-list size (Issue #644). When `null`, the
    ///   searcher uses an internal fallback of 50. Per-query overrides
    ///   via the search request still take precedence.
    /// * `embedder` - Embedder name registered via `addEmbedder` (default: "" for none).
    /// * `quantizer` - Vector quantizer — "scalar_8bit" (default) or
    ///   "product_quantization" (requires `subvector_count`).
    /// * `subvector_count` - Number of PQ sub-vectors. Required when
    ///   `quantizer` is "product_quantization" and must divide `dimension`;
    ///   rejected for other quantizers.
    /// * `rerank_storage` - Stage-2 rerank sidecar — omitted (default) keeps
    ///   the int8-only segment, "f32" stores full-precision vectors in a
    ///   `*.hnsw.f32` sidecar for exact rerank distances.
    /// * `pq_codebook_path` - Storage-relative file name of a shared PQ
    ///   codebook (Issue #631), trained once via the
    ///   `laurus train pq-codebook` CLI command. Only meaningful when
    ///   `quantizer` is "product_quantization"; commits then encode against
    ///   the pre-trained codebook instead of re-training k-means per
    ///   segment. Omitted (default) keeps per-segment training.
    /// * `base_weight` - This field's relative scoring priority when
    ///   searched alongside other vector fields (Issue #1084). Defaults to
    ///   1.0. Only matters when a query targets two or more specific
    ///   vector fields at once; has no effect on the lexical-vs-vector
    ///   balance of a hybrid search.
    ///
    /// # Errors
    ///
    /// Throws `\Exception` if `distance`, `quantizer` or `rerank_storage`
    /// is not a known name, or if `subvector_count` is missing for, or
    /// supplied without, `"product_quantization"`. Throws `\ValueError` if
    /// `name` starts with `_` (other than `_id`), which is reserved for
    /// system fields.
    #[php(defaults(m = 16, ef_construction = 200, base_weight = 1.0))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_hnsw_field(
        &self,
        name: String,
        dimension: i64,
        distance: Option<String>,
        m: i64,
        ef_construction: i64,
        default_ef_search: Option<i64>,
        embedder: Option<String>,
        quantizer: Option<String>,
        subvector_count: Option<i64>,
        rerank_storage: Option<String>,
        pq_codebook_path: Option<String>,
        base_weight: f64,
    ) -> PhpResult<()> {
        let dist_str = distance.unwrap_or_else(|| "cosine".to_string());
        let opt = HnswOption {
            dimension: dimension as usize,
            distance: parse_distance(&dist_str)?,
            m: m as usize,
            ef_construction: ef_construction as usize,
            default_ef_search: default_ef_search.map(|v| v as usize),
            quantizer: parse_quantizer(quantizer.as_deref(), subvector_count.map(|v| v as usize))?,
            rerank_storage: parse_rerank_storage(rerank_storage.as_deref())?,
            embedder,
            pq_codebook_path,
            base_weight: base_weight as f32,
        };
        self.insert_field(name, FieldOption::Hnsw(opt))
    }

    /// Add a flat (brute-force) vector index field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `dimension` - Vector dimensionality.
    /// * `distance` - Distance metric (default: "cosine").
    /// * `embedder` - Embedder name registered via `addEmbedder` (default: "" for none).
    /// * `base_weight` - This field's relative scoring priority when
    ///   searched alongside other vector fields (Issue #1084). Defaults to
    ///   1.0. See `add_hnsw_field` for the full contract.
    ///
    /// # Errors
    ///
    /// Throws `\Exception` if `distance` is not a known metric, or
    /// `\ValueError` if `name` starts with `_` (other than `_id`), which is
    /// reserved for system fields.
    #[php(defaults(base_weight = 1.0))]
    pub fn add_flat_field(
        &self,
        name: String,
        dimension: i64,
        distance: Option<String>,
        embedder: Option<String>,
        base_weight: f64,
    ) -> PhpResult<()> {
        let dist_str = distance.unwrap_or_else(|| "cosine".to_string());
        let opt = laurus::FlatOption {
            dimension: dimension as usize,
            distance: parse_distance(&dist_str)?,
            embedder,
            base_weight: base_weight as f32,
            ..Default::default()
        };
        self.insert_field(name, FieldOption::Flat(opt))
    }

    /// Add an IVF (Inverted File Index) approximate nearest-neighbor vector field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `dimension` - Vector dimensionality.
    /// * `distance` - Distance metric (default: "cosine").
    /// * `n_clusters` - Number of Voronoi clusters (default: 100).
    /// * `n_probe` - Number of clusters to probe at search time (default: 1).
    /// * `embedder` - Embedder name registered via `addEmbedder` (default: "" for none).
    /// * `base_weight` - This field's relative scoring priority when
    ///   searched alongside other vector fields (Issue #1084). Defaults to
    ///   1.0. See `add_hnsw_field` for the full contract.
    ///
    /// # Errors
    ///
    /// Throws `\Exception` if `distance` is not a known metric, or
    /// `\ValueError` if `name` starts with `_` (other than `_id`), which is
    /// reserved for system fields.
    #[php(defaults(n_clusters = 100, n_probe = 1, base_weight = 1.0))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_ivf_field(
        &self,
        name: String,
        dimension: i64,
        distance: Option<String>,
        n_clusters: i64,
        n_probe: i64,
        embedder: Option<String>,
        base_weight: f64,
    ) -> PhpResult<()> {
        let dist_str = distance.unwrap_or_else(|| "cosine".to_string());
        let opt = IvfOption {
            dimension: dimension as usize,
            distance: parse_distance(&dist_str)?,
            n_clusters: n_clusters as usize,
            n_probe: n_probe as usize,
            embedder,
            base_weight: base_weight as f32,
            ..Default::default()
        };
        self.insert_field(name, FieldOption::Ivf(opt))
    }

    /// Register a named embedder definition in the schema.
    ///
    /// The `config` array must have a `"type"` key selecting the backend:
    ///
    /// | type              | required keys | feature flag            |
    /// |-------------------|---------------|-------------------------|
    /// | `"precomputed"`   | —             | (always available)      |
    /// | `"candle_bert"`   | `"model"`     | `embeddings-candle`     |
    /// | `"candle_clip"`   | `"model"`     | `embeddings-multimodal` |
    /// | `"openai"`        | `"model"`     | `embeddings-openai`     |
    ///
    /// # Arguments
    ///
    /// * `name` - Unique embedder name referenced from vector fields.
    /// * `config` - Associative array describing the embedder.
    pub fn add_embedder(&self, name: String, config: &ZendHashTable) -> PhpResult<()> {
        let embedder_type = ht_get_string(config, "type")?;

        let definition = match embedder_type.as_str() {
            "precomputed" => EmbedderDefinition::Precomputed,
            "candle_bert" => {
                let model = ht_get_string(config, "model")?;
                EmbedderDefinition::CandleBert { model }
            }
            "candle_clip" => {
                let model = ht_get_string(config, "model")?;
                EmbedderDefinition::CandleClip { model }
            }
            "openai" => {
                let model = ht_get_string(config, "model")?;
                EmbedderDefinition::Openai { model }
            }
            other => {
                return Err(format!(
                    "Unknown embedder type: '{}'. Valid types: precomputed, candle_bert, candle_clip, openai",
                    other
                )
                .into());
            }
        };

        self.inner.borrow_mut().embedders.insert(name, definition);
        Ok(())
    }

    /// Register a named custom analyzer definition in the schema.
    ///
    /// `tokenizer` is required; `charFilters`/`tokenFilters` are optional
    /// lists of components, applied in list order. Each is an associative
    /// array with a `"type"` key, using the same shape as the schema
    /// TOML/JSON format shared with `laurus-cli` and the other language
    /// bindings (see the "Analyzer components" tables in the API
    /// reference). Semantic validity of the registered name (e.g.
    /// referencing an analyzer from `addTextField` that was never
    /// registered) is checked when the schema is used to build an
    /// `Index`, not here.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique analyzer name, referenced from `addTextField`'s
    ///   `analyzer` parameter.
    /// * `tokenizer` - Associative array describing the tokenizer.
    /// * `char_filters` - Optional sequential array of char-filter
    ///   definitions, applied to raw text before tokenization.
    /// * `token_filters` - Optional sequential array of token-filter
    ///   definitions, applied to the token stream after tokenization.
    ///
    /// # Errors
    ///
    /// Throws `\ValueError` if `name` is reserved for a built-in analyzer
    /// (`standard`, `keyword`, `english`, `simple`, `noop`), or if a
    /// component does not match a known shape.
    pub fn add_analyzer(
        &self,
        name: String,
        tokenizer: &ZendHashTable,
        char_filters: Option<&ZendHashTable>,
        token_filters: Option<&ZendHashTable>,
    ) -> PhpResult<()> {
        laurus::analysis::analyzer::registry::validate_analyzer_name(&name).map_err(laurus_err)?;
        let definition = AnalyzerDefinition {
            tokenizer: tokenizer_from_ht(tokenizer)?,
            char_filters: filter_list_from_ht(char_filters, "charFilters", char_filter_from_ht)?,
            token_filters: filter_list_from_ht(
                token_filters,
                "tokenFilters",
                token_filter_from_ht,
            )?,
        };
        self.inner.borrow_mut().analyzers.insert(name, definition);
        Ok(())
    }

    /// Return the names of custom analyzers registered via `addAnalyzer`
    /// or loaded from TOML.
    pub fn analyzer_names(&self) -> Vec<String> {
        self.inner.borrow().analyzers.keys().cloned().collect()
    }

    /// Parse a schema from a TOML string, in the same format
    /// `laurus-cli create index --schema` accepts.
    ///
    /// # Arguments
    ///
    /// * `toml_str` - Schema definition in TOML.
    ///
    /// # Errors
    ///
    /// Throws a PHP `ValueError` if `toml_str` is not a valid schema.
    pub fn from_toml(toml_str: String) -> PhpResult<Self> {
        let inner = Schema::from_toml(&toml_str).map_err(laurus_err)?;
        Ok(Self {
            inner: RefCell::new(inner),
        })
    }

    /// Load a schema from a TOML file (see `fromToml`).
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the TOML file.
    ///
    /// # Errors
    ///
    /// Throws a PHP `Exception` if the file cannot be read, or a
    /// `ValueError` if its content is not a valid schema.
    pub fn from_toml_file(path: String) -> PhpResult<Self> {
        let content = std::fs::read_to_string(&path).map_err(|e| io_err_with_path(&path, e))?;
        Self::from_toml(content)
    }

    /// Serialize this schema to a TOML string in the same format
    /// `laurus-cli` accepts.
    ///
    /// Tables are emitted in sorted key order, not insertion order, so
    /// compare round-tripped schemas by their parsed content rather than
    /// by raw text.
    ///
    /// # Errors
    ///
    /// Throws a PHP `ValueError` if the schema cannot be serialized.
    pub fn to_toml(&self) -> PhpResult<String> {
        self.inner.borrow().to_toml().map_err(laurus_err)
    }

    /// Write this schema to a TOML file (see `toToml`).
    ///
    /// # Arguments
    ///
    /// * `path` - Destination path; an existing file is overwritten.
    ///
    /// # Errors
    ///
    /// Throws a PHP `Exception` if the file cannot be written, or a
    /// `ValueError` if the schema cannot be serialized.
    pub fn to_toml_file(&self, path: String) -> PhpResult<()> {
        let content = self.to_toml()?;
        std::fs::write(&path, content).map_err(|e| io_err_with_path(&path, e))
    }

    /// Set the default fields used when no field is specified in a query.
    ///
    /// # Arguments
    ///
    /// * `fields` - Array of field name strings.
    pub fn set_default_fields(&self, fields: Vec<String>) {
        self.inner.borrow_mut().default_fields = fields;
    }

    /// Set the policy for fields that are not declared in this schema.
    ///
    /// Accepted values (case-insensitive): `"strict"`, `"dynamic"`,
    /// `"ignore"`. Behaviour:
    ///
    /// - `"strict"`: reject documents containing undeclared fields.
    /// - `"dynamic"` (default): infer a type for each undeclared field and
    ///   add it to the schema during ingestion. **Warning**: integer fields
    ///   silently truncate incoming float values (e.g. `3.14` → `3`).
    /// - `"ignore"`: silently drop undeclared fields.
    ///
    /// # Arguments
    ///
    /// * `policy` - One of `"strict"`, `"dynamic"`, `"ignore"`.
    ///
    /// # Errors
    ///
    /// Throws a PHP `Exception` if `policy` is not one of the accepted names.
    pub fn set_dynamic_field_policy(&self, policy: String) -> PhpResult<()> {
        let parsed = DynamicFieldPolicy::from_str(&policy)
            .map_err(|e| PhpException::from_message(e.to_string()))?;
        self.inner.borrow_mut().dynamic_field_policy = parsed;
        Ok(())
    }

    /// Return the currently configured dynamic field policy as a lowercase
    /// string (`"strict"` / `"dynamic"` / `"ignore"`).
    pub fn dynamic_field_policy(&self) -> String {
        match self.inner.borrow().dynamic_field_policy {
            DynamicFieldPolicy::Strict => "strict".to_string(),
            DynamicFieldPolicy::Dynamic => "dynamic".to_string(),
            DynamicFieldPolicy::Ignore => "ignore".to_string(),
        }
    }

    /// Return the list of field names defined in this schema.
    pub fn field_names(&self) -> Vec<String> {
        self.inner.borrow().fields.keys().cloned().collect()
    }

    /// Return a string representation of this schema.
    pub fn __to_string(&self) -> String {
        format!(
            "Schema(fields={:?})",
            self.inner.borrow().fields.keys().collect::<Vec<_>>()
        )
    }
}

impl PhpSchema {
    /// Insert a field after rejecting a name reserved for system fields.
    fn insert_field(&self, name: String, option: FieldOption) -> PhpResult<()> {
        laurus::validate_field_name(&name).map_err(laurus_err)?;
        self.inner.borrow_mut().fields.insert(name, option);
        Ok(())
    }
}
