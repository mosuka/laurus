//! PHP wrapper for the Laurus [`Schema`] type.

use std::cell::RefCell;
use std::str::FromStr;

use ext_php_rs::convert::FromZval;
use ext_php_rs::prelude::*;
use ext_php_rs::types::ZendHashTable;
use laurus::{
    BooleanOption, BytesOption, DateTimeOption, DistanceMetric, DynamicFieldPolicy,
    EmbedderDefinition, FieldOption, FloatOption, Geo3dOption, GeoOption, HnswOption,
    IntegerOption, IvfOption, QuantizationMethod, RerankStorageKind, Schema, TextOption,
};

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
    /// * `analyzer` - Optional analyzer name. For parameter-less built-in
    ///   analyzers (`"standard"`, `"english"`, `"keyword"`, `"simple"`,
    ///   `"noop"`) pass the name directly. Parameterized presets such as
    ///   the Japanese analyzer (which needs a Lindera dictionary path)
    ///   should be registered via `addAnalyzer` and referenced by name.
    #[php(defaults(stored = true, indexed = true, term_vectors = true))]
    pub fn add_text_field(
        &self,
        name: String,
        stored: bool,
        indexed: bool,
        term_vectors: bool,
        analyzer: Option<String>,
    ) {
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Text(TextOption {
                indexed,
                stored,
                term_vectors,
                analyzer: analyzer.map(laurus::AnalyzerSpec::Named),
            }),
        );
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
    #[php(defaults(stored = true, indexed = true, multi_valued = false))]
    pub fn add_integer_field(&self, name: String, stored: bool, indexed: bool, multi_valued: bool) {
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Integer(IntegerOption {
                indexed,
                stored,
                multi_valued,
            }),
        );
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
    #[php(defaults(stored = true, indexed = true, multi_valued = false))]
    pub fn add_float_field(&self, name: String, stored: bool, indexed: bool, multi_valued: bool) {
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Float(FloatOption {
                indexed,
                stored,
                multi_valued,
            }),
        );
    }

    /// Add a boolean field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    #[php(defaults(stored = true, indexed = true))]
    pub fn add_boolean_field(&self, name: String, stored: bool, indexed: bool) {
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::Boolean(BooleanOption { indexed, stored }),
        );
    }

    /// Add a date/time field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    #[php(defaults(stored = true, indexed = true))]
    pub fn add_datetime_field(&self, name: String, stored: bool, indexed: bool) {
        self.inner.borrow_mut().fields.insert(
            name,
            FieldOption::DateTime(DateTimeOption { indexed, stored }),
        );
    }

    /// Add a geographic coordinate field (latitude, longitude).
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    /// * `indexed` - Whether the field is searchable (default: true).
    #[php(defaults(stored = true, indexed = true))]
    pub fn add_geo_field(&self, name: String, stored: bool, indexed: bool) {
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Geo(GeoOption { indexed, stored }));
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
    #[php(defaults(stored = true, indexed = true))]
    pub fn add_geo3d_field(&self, name: String, stored: bool, indexed: bool) {
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Geo3d(Geo3dOption { indexed, stored }));
    }

    /// Add a binary data field.
    ///
    /// # Arguments
    ///
    /// * `name` - Field name.
    /// * `stored` - Whether the value is retrievable (default: true).
    #[php(defaults(stored = true))]
    pub fn add_bytes_field(&self, name: String, stored: bool) {
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Bytes(BytesOption { stored }));
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
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Hnsw(opt));
        Ok(())
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
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Flat(opt));
        Ok(())
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
        self.inner
            .borrow_mut()
            .fields
            .insert(name, FieldOption::Ivf(opt));
        Ok(())
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
            .map_err(|e| PhpException::default(e.to_string()))?;
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
