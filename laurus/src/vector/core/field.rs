//! Vector field configuration options.
//!
//! This module defines options for configuring vector fields, including
//! index types and parameters for different algorithms (Flat, HNSW, IVF),
//! and the multi-vector field used for late-interaction rescoring.

use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::multi_vector::MultiVectorStorage;
use crate::vector::core::quantization;
use crate::vector::core::rerank::RerankStorageKind;

fn default_dimension() -> usize {
    128
}

/// [`HnswOption::m`] when a schema omits it.
///
/// Public so a front end that builds options from its own wire format (the
/// gRPC server, Issue #1479) gives an omitted value the same default a TOML
/// schema gets.
pub const DEFAULT_HNSW_M: usize = 16;

/// [`HnswOption::ef_construction`] when a schema omits it. See
/// [`DEFAULT_HNSW_M`].
pub const DEFAULT_HNSW_EF_CONSTRUCTION: usize = 200;

/// [`IvfOption::n_clusters`] when a schema omits it. See [`DEFAULT_HNSW_M`].
pub const DEFAULT_IVF_N_CLUSTERS: usize = 100;

/// [`IvfOption::n_probe`] when a schema omits it. See [`DEFAULT_HNSW_M`].
pub const DEFAULT_IVF_N_PROBE: usize = 1;

fn default_getting_m() -> usize {
    DEFAULT_HNSW_M
}

fn default_getting_ef_construction() -> usize {
    DEFAULT_HNSW_EF_CONSTRUCTION
}

fn default_getting_n_clusters() -> usize {
    DEFAULT_IVF_N_CLUSTERS
}

fn default_getting_n_probe() -> usize {
    DEFAULT_IVF_N_PROBE
}

/// Options for vector fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "options", rename_all = "snake_case")]
pub enum FieldOption {
    /// Flat index options.
    Flat(FlatOption),
    /// HNSW index options.
    Hnsw(HnswOption),
    /// IVF index options.
    Ivf(IvfOption),
    /// Multi-vector (late-interaction) options.
    MultiVector(MultiVectorOption),
}

impl Default for FieldOption {
    fn default() -> Self {
        FieldOption::Hnsw(HnswOption::default())
    }
}

impl FieldOption {
    /// Get the dimension of the vector field.
    pub fn dimension(&self) -> usize {
        match self {
            FieldOption::Flat(opt) => opt.dimension,
            FieldOption::Hnsw(opt) => opt.dimension,
            FieldOption::Ivf(opt) => opt.dimension,
            FieldOption::MultiVector(opt) => opt.dimension,
        }
    }

    /// Get the distance metric.
    pub fn distance(&self) -> DistanceMetric {
        match self {
            FieldOption::Flat(opt) => opt.distance,
            FieldOption::Hnsw(opt) => opt.distance,
            FieldOption::Ivf(opt) => opt.distance,
            FieldOption::MultiVector(opt) => opt.distance,
        }
    }

    /// Get this field's relative scoring priority (Issue #1084).
    ///
    /// Read by `VectorStore::search_impl` and multiplied into a query's
    /// per-field weight, so it only matters when a query is routed to two
    /// or more specific vector fields at once — it has no effect on a
    /// single-field query, and it does not affect the lexical-vs-vector
    /// balance in a hybrid search's fusion (`FusionAlgorithm::RRF` is
    /// rank-only and `WeightedSum` min-max normalizes each side before
    /// weighting, so a uniform per-field scalar is normalized away on
    /// either side). Should be a positive, finite value — non-positive or
    /// non-finite values are clamped to `1.0` (with a warning) wherever
    /// they are read.
    ///
    /// A [`FieldOption::MultiVector`] field is never a vector-search
    /// target, so it always reports `1.0`.
    pub fn base_weight(&self) -> f32 {
        match self {
            FieldOption::Flat(opt) => opt.base_weight,
            FieldOption::Hnsw(opt) => opt.base_weight,
            FieldOption::Ivf(opt) => opt.base_weight,
            FieldOption::MultiVector(_) => default_weight(),
        }
    }

    /// Get the index kind.
    pub fn index_kind(&self) -> VectorIndexKind {
        match self {
            FieldOption::Flat(_) => VectorIndexKind::Flat,
            FieldOption::Hnsw(_) => VectorIndexKind::Hnsw,
            FieldOption::Ivf(_) => VectorIndexKind::Ivf,
            FieldOption::MultiVector(_) => VectorIndexKind::MultiVector,
        }
    }

    /// Whether this field stores per-document token vectors for
    /// late-interaction rescoring instead of one searchable vector.
    pub fn is_multi_vector(&self) -> bool {
        matches!(self, FieldOption::MultiVector(_))
    }

    /// Check that the options describe a usable field, with the check of
    /// the field's own kind ([`FlatOption::validate`],
    /// [`HnswOption::validate`], [`IvfOption::validate`] or
    /// [`MultiVectorOption::validate`]).
    ///
    /// # Arguments
    ///
    /// * `field_name` - The field name, used in the error message.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::invalid_argument`] when the kind's check
    /// rejects the options.
    pub fn validate(&self, field_name: &str) -> Result<()> {
        match self {
            FieldOption::Flat(opt) => opt.validate(field_name),
            FieldOption::Hnsw(opt) => opt.validate(field_name),
            FieldOption::Ivf(opt) => opt.validate(field_name),
            FieldOption::MultiVector(opt) => opt.validate(field_name),
        }
    }

    /// Why the field's index cannot work with these parameters, if it
    /// cannot: everything [`Self::validate`] checks except whether the
    /// index kind implements the quantizer ([`quantizer_support_error`]),
    /// which opening an index still refuses (Issue #1457). Product
    /// quantization's `subvector_count` and supported distance metrics are
    /// parameter errors (Issues #1491 and #1458), reported here.
    ///
    /// Opening an index uses this to warn about a field persisted before
    /// the checks existed (Issue #1479) instead of refusing to open it, so
    /// the field can still be repaired with `Engine::update_field`.
    pub(crate) fn params_error(&self) -> Option<String> {
        match self {
            FieldOption::Flat(opt) => opt.params_error(),
            FieldOption::Hnsw(opt) => opt.params_error(),
            FieldOption::Ivf(opt) => opt.params_error(),
            FieldOption::MultiVector(opt) => {
                multi_vector_params_error(opt.dimension, opt.distance, opt.storage, opt.pool_factor)
            }
        }
    }
}

/// Turn the reason a field's options are unusable into the error the
/// `validate` methods return.
fn reject_field(field_name: &str, reason: Option<String>) -> Result<()> {
    match reason {
        Some(reason) => Err(LaurusError::invalid_argument(format!(
            "field '{field_name}': {reason}"
        ))),
        None => Ok(()),
    }
}

/// Options for Flat vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatOption {
    /// Number of dimensions for each vector. Defaults to `128`.
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    /// Distance metric used for similarity computation. Defaults to [`DistanceMetric::Cosine`].
    #[serde(default = "default_distance_metric")]
    pub distance: DistanceMetric,
    /// This field's relative scoring priority when searched alongside
    /// other vector fields (Issue #1084). See
    /// [`FieldOption::base_weight`] for the full contract. Defaults to
    /// `1.0`.
    #[serde(default = "default_weight")]
    pub base_weight: f32,
    /// Quantization method used for the on-disk vector format.
    /// Defaults to [`quantization::QuantizationMethod::Scalar8Bit`]
    /// (Issue #481 Stage 1: int8 SQ is mandatory; the previous
    /// `Option::None` "no quantization" path no longer exists).
    ///
    /// Flat supports only [`Scalar8Bit`](quantization::QuantizationMethod::Scalar8Bit)
    /// and [`Float16`](quantization::QuantizationMethod::Float16);
    /// `ProductQuantization` (and, with the `pq-fastscan` feature,
    /// `ProductQuantizationFastScan`) are HNSW-only and are rejected by
    /// [`Self::validate`] (Issue #1457).
    #[serde(default)]
    pub quantizer: quantization::QuantizationMethod,
    /// Two-stage rerank storage backend (Issue #481 Stage 2).
    ///
    /// When `Some(RerankStorageKind::F32)`, an `*.{ext}.f32` sidecar
    /// is written alongside the int8 segment so the searcher can
    /// re-score the top `top_k * rerank_factor` candidates with the
    /// original f32 vectors. Costs ~4x extra disk and memory per
    /// vector but recovers Stage 1 recall close to the f32 baseline.
    ///
    /// `None` (the default) leaves the field on the Stage 1 int8-only
    /// path; queries that set `rerank_factor` against such a field
    /// silently fall back to Stage 1 ranking (the original
    /// information was discarded at index time).
    #[serde(default)]
    pub rerank_storage: Option<RerankStorageKind>,
    /// Embedder name for this vector field.
    /// When set, the engine automatically embeds input using the named embedder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder: Option<String>,
}

impl Default for FlatOption {
    fn default() -> Self {
        Self {
            dimension: 128,
            distance: default_distance_metric(),
            base_weight: default_weight(),
            quantizer: quantization::QuantizationMethod::Scalar8Bit,
            rerank_storage: None,
            embedder: None,
        }
    }
}

/// Options for HNSW vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswOption {
    /// Number of dimensions for each vector. Defaults to `128`.
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    /// Distance metric used for similarity computation. Defaults to [`DistanceMetric::Cosine`].
    #[serde(default = "default_distance_metric")]
    pub distance: DistanceMetric,
    /// Maximum number of bi-directional links per node in the HNSW graph.
    /// Higher values improve recall but increase memory usage. Defaults to `16`.
    #[serde(default = "default_getting_m")]
    pub m: usize,
    /// Size of the dynamic candidate list during index construction.
    /// Higher values produce a higher-quality graph at the cost of slower
    /// build times. Defaults to `200`.
    #[serde(default = "default_getting_ef_construction")]
    pub ef_construction: usize,
    /// Default size of the dynamic candidate list during search (`ef_search`).
    ///
    /// Controls the recall / latency trade-off at query time. Higher values
    /// explore more graph neighbours, improving recall at the cost of latency.
    ///
    /// When `None` (the default), the engine searches the field with
    /// `ef_construction.max(50) * 2`, `400` with the default
    /// `ef_construction` of `200`, so a smaller value narrows the search.
    /// The engine opens every HNSW field in the segmented layout, which
    /// searches each segment's graph with this wider list to keep
    /// multi-segment recall stable (#644, #880); only a monolithic index
    /// ([`HnswIndexConfig::segmented`](crate::vector::index::config::HnswIndexConfig::segmented)
    /// `= false`) falls back to `50`. Per-query
    /// [`VectorIndexQueryParams::ef_search`](crate::vector::search::searcher::VectorIndexQueryParams::ef_search)
    /// always takes precedence over this schema-level default.
    ///
    /// Regardless of which source is used, the effective `ef_search` is also
    /// lifted to `max(ef_search, top_k * rerank_factor.unwrap_or(1), top_k)`
    /// so the candidate heap is never undersized for the requested `top_k`
    /// (or for the candidate-widening implied by Stage-2 rerank).
    ///
    /// Issue [#644](https://github.com/mosuka/laurus/issues/644).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_ef_search: Option<usize>,
    /// This field's relative scoring priority when searched alongside
    /// other vector fields (Issue #1084). See
    /// [`FieldOption::base_weight`] for the full contract. Defaults to
    /// `1.0`.
    #[serde(default = "default_weight")]
    pub base_weight: f32,
    /// Quantization method used for the on-disk vector format.
    /// Defaults to [`quantization::QuantizationMethod::Scalar8Bit`]
    /// (Issue #481 Stage 1: int8 SQ is mandatory; the previous
    /// `Option::None` "no quantization" path no longer exists).
    #[serde(default)]
    pub quantizer: quantization::QuantizationMethod,
    /// Two-stage rerank storage backend (Issue #481 Stage 2).
    ///
    /// When `Some(RerankStorageKind::F32)`, an `*.{ext}.f32` sidecar
    /// is written alongside the int8 segment so the searcher can
    /// re-score the top `top_k * rerank_factor` candidates with the
    /// original f32 vectors. Costs ~4x extra disk and memory per
    /// vector but recovers Stage 1 recall close to the f32 baseline.
    ///
    /// `None` (the default) leaves the field on the Stage 1 int8-only
    /// path; queries that set `rerank_factor` against such a field
    /// silently fall back to Stage 1 ranking (the original
    /// information was discarded at index time).
    #[serde(default)]
    pub rerank_storage: Option<RerankStorageKind>,
    /// Embedder name for this vector field.
    /// When set, the engine automatically embeds input using the named embedder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder: Option<String>,
    /// Storage-relative file name of a shared PQ codebook (Issue #631).
    ///
    /// Only meaningful when [`Self::quantizer`] is
    /// [`quantization::QuantizationMethod::ProductQuantization`] (k=256)
    /// or, with the `pq-fastscan` feature, `ProductQuantizationFastScan`
    /// (k=16 — Issue #920; the same `.pqcb` file format carries either
    /// variant, distinguished by the stored `k`). When set, segment
    /// writes encode against the named pre-trained codebook (trained
    /// once via `Engine::train_pq_codebook` / the
    /// `laurus train pq-codebook` CLI command) instead of re-running
    /// k-means from scratch on every commit and merge. The segment
    /// format is unchanged: the shared codebook is still embedded
    /// inline in each segment header, so old and new segments coexist.
    ///
    /// When the named file does not exist yet, opening the index stays
    /// lenient but a commit that needs to encode hard-errors with the
    /// training command to run — there is no silent fallback to
    /// per-segment training.
    ///
    /// `None` (the default) keeps per-segment inline training.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pq_codebook_path: Option<String>,
}

impl Default for HnswOption {
    fn default() -> Self {
        Self {
            dimension: 128,
            distance: default_distance_metric(),
            m: default_getting_m(),
            ef_construction: default_getting_ef_construction(),
            default_ef_search: None,
            base_weight: default_weight(),
            quantizer: quantization::QuantizationMethod::Scalar8Bit,
            rerank_storage: None,
            embedder: None,
            pq_codebook_path: None,
        }
    }
}

/// Options for IVF vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfOption {
    /// Number of dimensions for each vector.
    pub dimension: usize,
    /// Distance metric used for similarity computation. Defaults to [`DistanceMetric::Cosine`].
    #[serde(default = "default_distance_metric")]
    pub distance: DistanceMetric,
    /// Number of Voronoi clusters used to partition the vector space.
    /// More clusters speed up search but increase build time. Defaults to `100`.
    #[serde(default = "default_getting_n_clusters")]
    pub n_clusters: usize,
    /// Number of clusters to probe during search.
    /// Higher values improve recall at the cost of query latency. Defaults to `1`.
    #[serde(default = "default_getting_n_probe")]
    pub n_probe: usize,
    /// This field's relative scoring priority when searched alongside
    /// other vector fields (Issue #1084). See
    /// [`FieldOption::base_weight`] for the full contract. Defaults to
    /// `1.0`.
    #[serde(default = "default_weight")]
    pub base_weight: f32,
    /// Quantization method used for the on-disk vector format.
    /// Defaults to [`quantization::QuantizationMethod::Scalar8Bit`]
    /// (Issue #481 Stage 1: int8 SQ is mandatory; the previous
    /// `Option::None` "no quantization" path no longer exists).
    ///
    /// IVF supports only [`Scalar8Bit`](quantization::QuantizationMethod::Scalar8Bit)
    /// and [`Float16`](quantization::QuantizationMethod::Float16);
    /// `ProductQuantization` (and, with the `pq-fastscan` feature,
    /// `ProductQuantizationFastScan`) are HNSW-only and are rejected by
    /// [`Self::validate`] (Issue #1457).
    #[serde(default)]
    pub quantizer: quantization::QuantizationMethod,
    /// Two-stage rerank storage backend (Issue #481 Stage 2).
    ///
    /// When `Some(RerankStorageKind::F32)`, an `*.{ext}.f32` sidecar
    /// is written alongside the int8 segment so the searcher can
    /// re-score the top `top_k * rerank_factor` candidates with the
    /// original f32 vectors. Costs ~4x extra disk and memory per
    /// vector but recovers Stage 1 recall close to the f32 baseline.
    ///
    /// `None` (the default) leaves the field on the Stage 1 int8-only
    /// path; queries that set `rerank_factor` against such a field
    /// silently fall back to Stage 1 ranking (the original
    /// information was discarded at index time).
    #[serde(default)]
    pub rerank_storage: Option<RerankStorageKind>,
    /// Embedder name for this vector field.
    /// When set, the engine automatically embeds input using the named embedder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder: Option<String>,
}

impl Default for IvfOption {
    fn default() -> Self {
        Self {
            dimension: 128,
            distance: default_distance_metric(),
            n_clusters: default_getting_n_clusters(),
            n_probe: default_getting_n_probe(),
            base_weight: default_weight(),
            quantizer: quantization::QuantizationMethod::Scalar8Bit,
            rerank_storage: None,
            embedder: None,
        }
    }
}

/// Options for a multi-vector field (Issue #1177).
///
/// The field stores every token vector of a document (e.g. the per-token
/// embeddings of a ColBERT-style model) with no ANN index. It is not a
/// vector-search target; it is read by late-interaction rescoring, which
/// scores a document as `Σ_i max_j sim(q_i, d_j)` over the query's and the
/// document's token vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorOption {
    /// Number of dimensions of every token vector. Defaults to `128`.
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    /// Token similarity. Only [`DistanceMetric::Cosine`] (vectors are
    /// L2-normalized, so the similarity is their dot product) and
    /// [`DistanceMetric::DotProduct`] are supported, and
    /// [`MultiVectorStorage::Binary`] requires `Cosine`. Defaults to
    /// [`DistanceMetric::Cosine`].
    #[serde(default = "default_distance_metric")]
    pub distance: DistanceMetric,
    /// On-disk element kind of every token vector (Issue #1346). Defaults
    /// to [`MultiVectorStorage::F32`], the only kind ever written before
    /// this option existed, so a schema with no `storage` key keeps
    /// reading old segments unchanged.
    #[serde(default)]
    pub storage: MultiVectorStorage,
    /// Token-pooling factor (Issue #1346): at index time, every document's
    /// token vectors after the first are clustered into about `1 /
    /// pool_factor` as many and replaced by the cluster means (see
    /// [`crate::vector::core::token_pooling`]). `1`, the default, stores
    /// every token vector; `0` is rejected. Pooling cannot be undone:
    /// turning it on for an existing field re-pools the stored vectors, but
    /// changing or turning off a factor already in use discards the field's
    /// data.
    #[serde(
        default = "default_pool_factor",
        skip_serializing_if = "is_default_pool_factor"
    )]
    pub pool_factor: u32,
    /// Name of the schema embedder that turns text into this field's token
    /// vectors; it must be a token-level embedder such as `candle_colbert`
    /// (Issue #1349). `None` means documents supply the token vectors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder: Option<String>,
}

impl Default for MultiVectorOption {
    fn default() -> Self {
        Self {
            dimension: default_dimension(),
            distance: default_distance_metric(),
            storage: MultiVectorStorage::default(),
            pool_factor: default_pool_factor(),
            embedder: None,
        }
    }
}

/// Serde default of [`MultiVectorOption::pool_factor`]: `1`, no pooling. A
/// plain `#[serde(default)]` would give `0`, which validation rejects.
pub(crate) const fn default_pool_factor() -> u32 {
    1
}

fn is_default_pool_factor(factor: &u32) -> bool {
    *factor == default_pool_factor()
}

impl MultiVectorOption {
    /// Most token vectors one document may hold (the same limit as
    /// Elasticsearch's `rank_vectors`).
    pub const MAX_VECTORS_PER_DOCUMENT: usize = 8192;

    /// Create options for token vectors of `dimension` dimensions.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    /// Set the dimension of every token vector.
    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Set the token similarity (`Cosine` or `DotProduct`).
    pub fn distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    /// Set the on-disk element kind of every token vector.
    pub fn storage(mut self, storage: MultiVectorStorage) -> Self {
        self.storage = storage;
        self
    }

    /// Set the token-pooling factor (`1` disables pooling).
    pub fn pool_factor(mut self, factor: u32) -> Self {
        self.pool_factor = factor;
        self
    }

    /// Embed text values with the schema embedder `name`, which must be a
    /// token-level embedder.
    pub fn embedder(mut self, name: impl Into<String>) -> Self {
        self.embedder = Some(name.into());
        self
    }

    /// Check that the options describe a usable field.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The field name, used in the error message.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::invalid_argument`] when the dimension is zero,
    /// the distance is neither `Cosine` nor `DotProduct`, or the storage is
    /// `Binary` and the distance is not `Cosine` (see
    /// [`multi_vector_params_error`]).
    pub fn validate(&self, field_name: &str) -> Result<()> {
        reject_field(
            field_name,
            multi_vector_params_error(
                self.dimension,
                self.distance,
                self.storage,
                self.pool_factor,
            ),
        )
    }

    /// Check one document's token vectors against this field.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The field name, used in the error message.
    /// * `vectors` - The document's token vectors.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::invalid_argument`] when there are no vectors,
    /// more than [`Self::MAX_VECTORS_PER_DOCUMENT`], a vector whose length
    /// is not [`Self::dimension`], or a non-finite value.
    pub fn validate_vectors(&self, field_name: &str, vectors: &[Vec<f32>]) -> Result<()> {
        let invalid = |reason: String| {
            Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': {reason}"
            )))
        };
        if vectors.is_empty() {
            return invalid("a multi-vector value needs at least one vector".to_string());
        }
        if vectors.len() > Self::MAX_VECTORS_PER_DOCUMENT {
            return invalid(format!(
                "a multi-vector value holds at most {} vectors, got {}",
                Self::MAX_VECTORS_PER_DOCUMENT,
                vectors.len()
            ));
        }
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != self.dimension {
                return invalid(format!(
                    "vector {i} has dimension {}, expected {}",
                    vector.len(),
                    self.dimension
                ));
            }
            if vector.iter().any(|v| !v.is_finite()) {
                return invalid(format!("vector {i} contains a non-finite value"));
            }
        }
        Ok(())
    }
}

/// Why `dimension` / `distance` / `storage` / `pool_factor` cannot
/// describe a multi-vector field, if they cannot.
///
/// Late interaction sums each query vector's best similarity, which is
/// meaningless for a distance where smaller is better, so only `Cosine` and
/// `DotProduct` are accepted. `Binary` storage keeps only each token
/// vector's direction relative to the field's center, so it requires
/// `Cosine`, where every token is unit length anyway. A pool factor of `0`
/// has no meaning (`1` disables pooling).
pub(crate) fn multi_vector_params_error(
    dimension: usize,
    distance: DistanceMetric,
    storage: MultiVectorStorage,
    pool_factor: u32,
) -> Option<String> {
    if dimension == 0 {
        return Some("MultiVector dimension must be greater than 0".to_string());
    }
    if pool_factor == 0 {
        return Some("MultiVector pool_factor must be at least 1 (1 disables pooling)".to_string());
    }
    match distance {
        DistanceMetric::Cosine => None,
        DistanceMetric::DotProduct if storage == MultiVectorStorage::Binary => Some(
            "MultiVector storage Binary requires distance Cosine: it keeps each token \
             vector's direction, not its magnitude"
                .to_string(),
        ),
        DistanceMetric::DotProduct => None,
        other => Some(format!(
            "MultiVector distance must be Cosine or DotProduct, got {}",
            other.name()
        )),
    }
}

/// Why an index of kind `index_kind` cannot store vectors with
/// `quantizer`, if it cannot.
///
/// Product quantization is implemented only by HNSW. Flat and IVF store
/// `Scalar8Bit` and `Float16`; their writers used to take a PQ quantizer
/// down the `Scalar8Bit` path without an error (Issue #1457).
pub(crate) fn quantizer_support_error(
    index_kind: VectorIndexKind,
    quantizer: quantization::QuantizationMethod,
) -> Option<String> {
    use quantization::QuantizationMethod;

    let kind = match index_kind {
        VectorIndexKind::Flat => "Flat",
        VectorIndexKind::Ivf => "IVF",
        VectorIndexKind::Hnsw | VectorIndexKind::MultiVector => return None,
    };
    let supported = match quantizer {
        QuantizationMethod::Scalar8Bit | QuantizationMethod::Float16 => true,
        QuantizationMethod::ProductQuantization { .. } => false,
        #[cfg(feature = "pq-fastscan")]
        QuantizationMethod::ProductQuantizationFastScan { .. } => false,
    };
    (!supported).then(|| {
        format!(
            "{kind} does not support quantizer {quantizer:?}: product quantization is \
             HNSW-only; use Scalar8Bit or Float16"
        )
    })
}

/// The type of vector index to use.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndexKind {
    /// Flat (brute-force) index - exact but slower for large datasets.
    Flat,
    /// HNSW (Hierarchical Navigable Small World) - approximate but fast.
    Hnsw,
    /// IVF (Inverted File Index) - approximate with clustering.
    Ivf,
    /// Per-document token vectors with no ANN index, read by
    /// late-interaction rescoring.
    MultiVector,
}

// From implementations for VectorOption
impl From<FlatOption> for FieldOption {
    fn from(opt: FlatOption) -> Self {
        FieldOption::Flat(opt)
    }
}

impl From<HnswOption> for FieldOption {
    fn from(opt: HnswOption) -> Self {
        FieldOption::Hnsw(opt)
    }
}

impl From<IvfOption> for FieldOption {
    fn from(opt: IvfOption) -> Self {
        FieldOption::Ivf(opt)
    }
}

impl From<MultiVectorOption> for FieldOption {
    fn from(opt: MultiVectorOption) -> Self {
        FieldOption::MultiVector(opt)
    }
}

// Builder pattern for FlatOption
impl FlatOption {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    pub fn distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    pub fn base_weight(mut self, weight: f32) -> Self {
        self.base_weight = weight;
        self
    }

    pub fn quantizer(mut self, quantizer: quantization::QuantizationMethod) -> Self {
        self.quantizer = quantizer;
        self
    }

    /// Enable two-stage rerank (Issue #481 Stage 2) for this field.
    ///
    /// Storing rerank data writes a sidecar file at index time so the
    /// searcher can re-score the top `top_k * rerank_factor`
    /// candidates against the original full-precision vectors. Pass
    /// `None` (or omit) to stay on the Stage 1 int8-only path.
    pub fn rerank_storage(mut self, kind: RerankStorageKind) -> Self {
        self.rerank_storage = Some(kind);
        self
    }

    /// Check that the options describe a usable Flat field.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The field name, used in the error message.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::invalid_argument`] when [`Self::dimension`] is
    /// `0` (Issue #1479), which every non-empty vector would then fail, or
    /// when [`Self::quantizer`] is a product quantizer, which Flat does not
    /// implement (see [`quantizer_support_error`]).
    pub fn validate(&self, field_name: &str) -> Result<()> {
        reject_field(
            field_name,
            self.params_error()
                .or_else(|| quantizer_support_error(VectorIndexKind::Flat, self.quantizer)),
        )
    }

    /// The part of [`Self::validate`] that does not look at the quantizer.
    fn params_error(&self) -> Option<String> {
        (self.dimension == 0).then(|| "Flat dimension must be greater than 0".to_string())
    }
}

// Builder pattern for HnswOption
impl HnswOption {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    pub fn distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set the schema-level default for `ef_search` at query time.
    ///
    /// Per-query [`crate::vector::search::searcher::VectorIndexQueryParams::ef_search`]
    /// still takes precedence. When this builder method is not called, the
    /// engine searches with `ef_construction.max(50) * 2` (`400` with the
    /// default `ef_construction`). See [`HnswOption::default_ef_search`] for
    /// the full precedence rules.
    pub fn default_ef_search(mut self, ef: usize) -> Self {
        self.default_ef_search = Some(ef);
        self
    }

    pub fn base_weight(mut self, weight: f32) -> Self {
        self.base_weight = weight;
        self
    }

    pub fn quantizer(mut self, quantizer: quantization::QuantizationMethod) -> Self {
        self.quantizer = quantizer;
        self
    }

    /// Enable two-stage rerank (Issue #481 Stage 2) for this field.
    ///
    /// Storing rerank data writes a sidecar file at index time so the
    /// searcher can re-score the top `top_k * rerank_factor`
    /// candidates against the original full-precision vectors. Pass
    /// `None` (or omit) to stay on the Stage 1 int8-only path.
    pub fn rerank_storage(mut self, kind: RerankStorageKind) -> Self {
        self.rerank_storage = Some(kind);
        self
    }

    /// Check that the options describe a usable HNSW field (Issues #1458,
    /// #1479 and #1491).
    ///
    /// HNSW implements every quantizer. Product Quantization (including
    /// FastScan) additionally supports only the `Euclidean` and `Cosine`
    /// distance metrics; its `subvector_count` must also fit
    /// [`Self::dimension`].
    ///
    /// # Arguments
    ///
    /// * `field_name` - The field name, used in the error message.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::invalid_argument`] when [`Self::dimension`] is
    /// `0`, which every non-empty vector would then fail, when [`Self::m`]
    /// is below `2`, which the HNSW writer refuses to build a graph with,
    /// when [`Self::ef_construction`] is `0`, which would link each
    /// inserted node to a single neighbor per layer, or when
    /// [`Self::quantizer`] is a product quantizer whose `subvector_count`
    /// is `0`, does not divide the dimension, or leaves it or the
    /// sub-vector dimension above `u16::MAX`, which the writer cannot
    /// encode with, or whose distance is neither `Euclidean` nor `Cosine`.
    pub fn validate(&self, field_name: &str) -> Result<()> {
        reject_field(field_name, self.params_error())
    }

    /// The checks of [`Self::validate`].
    fn params_error(&self) -> Option<String> {
        if self.dimension == 0 {
            return Some("HNSW dimension must be greater than 0".to_string());
        }
        if self.m < 2 {
            return Some(format!("HNSW m must be at least 2, got {}", self.m));
        }
        if self.ef_construction == 0 {
            return Some("HNSW ef_construction must be at least 1".to_string());
        }
        self.quantizer
            .dimension_error(self.dimension)
            .or_else(|| self.quantizer.distance_error(self.distance))
            .map(|reason| format!("HNSW {reason}"))
    }
}

// Builder pattern for IvfOption
impl IvfOption {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    pub fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    pub fn distance(mut self, distance: DistanceMetric) -> Self {
        self.distance = distance;
        self
    }

    pub fn n_clusters(mut self, n: usize) -> Self {
        self.n_clusters = n;
        self
    }

    pub fn n_probe(mut self, n: usize) -> Self {
        self.n_probe = n;
        self
    }

    pub fn base_weight(mut self, weight: f32) -> Self {
        self.base_weight = weight;
        self
    }

    pub fn quantizer(mut self, quantizer: quantization::QuantizationMethod) -> Self {
        self.quantizer = quantizer;
        self
    }

    /// Enable two-stage rerank (Issue #481 Stage 2) for this field.
    ///
    /// Storing rerank data writes a sidecar file at index time so the
    /// searcher can re-score the top `top_k * rerank_factor`
    /// candidates against the original full-precision vectors. Pass
    /// `None` (or omit) to stay on the Stage 1 int8-only path.
    pub fn rerank_storage(mut self, kind: RerankStorageKind) -> Self {
        self.rerank_storage = Some(kind);
        self
    }

    /// Check that the options describe a usable IVF field.
    ///
    /// # Arguments
    ///
    /// * `field_name` - The field name, used in the error message.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::invalid_argument`] when [`Self::dimension`] is
    /// `0`, which every non-empty vector would then fail, when
    /// [`Self::n_clusters`] is `0`, when [`Self::n_probe`] is `0`, which
    /// would make every search probe no cluster (Issue #1479), or when
    /// [`Self::quantizer`] is a product quantizer, which IVF does not
    /// implement (see [`quantizer_support_error`]).
    pub fn validate(&self, field_name: &str) -> Result<()> {
        reject_field(
            field_name,
            self.params_error()
                .or_else(|| quantizer_support_error(VectorIndexKind::Ivf, self.quantizer)),
        )
    }

    /// The part of [`Self::validate`] that does not look at the quantizer.
    fn params_error(&self) -> Option<String> {
        if self.dimension == 0 {
            return Some("IVF dimension must be greater than 0".to_string());
        }
        if self.n_clusters == 0 {
            return Some("IVF n_clusters must be at least 1".to_string());
        }
        if self.n_probe == 0 {
            return Some("IVF n_probe must be at least 1".to_string());
        }
        None
    }
}

// Helpers

fn default_distance_metric() -> DistanceMetric {
    DistanceMetric::Cosine
}

fn default_weight() -> f32 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #631: `pq_codebook_path` must survive a serde round-trip, and
    /// a schema written before the field existed must still deserialize
    /// (backward compatibility via `#[serde(default)]`).
    #[test]
    fn hnsw_option_pq_codebook_path_round_trips_and_defaults() {
        let opt = HnswOption {
            pq_codebook_path: Some("embedding.pqcb".to_string()),
            ..HnswOption::default()
        };
        let json = serde_json::to_string(&opt).unwrap();
        let back: HnswOption = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pq_codebook_path, Some("embedding.pqcb".to_string()));

        // A pre-#631 schema (no such key) still parses, defaulting to None.
        let legacy: HnswOption = serde_json::from_str(r#"{"dimension": 8}"#).unwrap();
        assert_eq!(legacy.pq_codebook_path, None);

        // `None` must not serialize a key at all, so freshly written
        // schemas stay readable by pre-#631 binaries.
        let default_json = serde_json::to_string(&HnswOption::default()).unwrap();
        assert!(!default_json.contains("pq_codebook_path"));
    }

    #[test]
    fn multi_vector_option_round_trips_and_defaults() {
        let opt =
            FieldOption::from(MultiVectorOption::new(96).distance(DistanceMetric::DotProduct));
        let json = serde_json::to_string(&opt).unwrap();
        let back: FieldOption = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dimension(), 96);
        assert_eq!(back.distance(), DistanceMetric::DotProduct);
        assert_eq!(back.index_kind(), VectorIndexKind::MultiVector);
        assert!(back.is_multi_vector());
        assert_eq!(back.base_weight(), 1.0);

        let defaulted: MultiVectorOption = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted.dimension, 128);
        assert_eq!(defaulted.distance, DistanceMetric::Cosine);
        assert_eq!(defaulted.embedder, None);
        assert!(!json.contains("embedder"), "an unset embedder is omitted");
        // Issue #1346: a schema persisted before `storage` existed has no
        // such key and must keep reading as F32 -- the only kind LMV1 v1
        // ever wrote.
        assert_eq!(defaulted.storage, MultiVectorStorage::F32);

        let named = MultiVectorOption::new(128).embedder("colbert");
        let back: MultiVectorOption =
            serde_json::from_str(&serde_json::to_string(&named).unwrap()).unwrap();
        assert_eq!(back.embedder.as_deref(), Some("colbert"));

        let compressed = MultiVectorOption::new(128).storage(MultiVectorStorage::Int8);
        let back: MultiVectorOption =
            serde_json::from_str(&serde_json::to_string(&compressed).unwrap()).unwrap();
        assert_eq!(back.storage, MultiVectorStorage::Int8);

        // Issue #1346: a schema without `pool_factor` reads as 1 (a plain
        // `#[serde(default)]` would give 0, which validation rejects), and
        // 1 is not written back.
        assert_eq!(defaulted.pool_factor, 1);
        assert!(!json.contains("pool_factor"), "{json}");
        let pooled = MultiVectorOption::new(128).pool_factor(3);
        let json = serde_json::to_string(&pooled).unwrap();
        assert!(json.contains("\"pool_factor\":3"), "{json}");
        let back: MultiVectorOption = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pool_factor, 3);
    }

    #[test]
    fn multi_vector_option_validate_rejects_unusable_settings() {
        assert!(MultiVectorOption::new(128).validate("t").is_ok());
        assert!(
            MultiVectorOption::new(8)
                .distance(DistanceMetric::DotProduct)
                .validate("t")
                .is_ok()
        );
        let err = MultiVectorOption::new(0).validate("t").unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid argument: field 't': MultiVector dimension must be greater than 0"
        );
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Angular,
        ] {
            let err = MultiVectorOption::new(8)
                .distance(metric)
                .validate("t")
                .unwrap_err();
            assert!(err.to_string().contains("Cosine or DotProduct"), "{err}");
        }
        let err = MultiVectorOption::new(8)
            .pool_factor(0)
            .validate("t")
            .unwrap_err();
        assert!(
            err.to_string().contains("pool_factor must be at least 1"),
            "{err}"
        );
        for factor in [1, 2, 100] {
            assert!(
                MultiVectorOption::new(8)
                    .pool_factor(factor)
                    .validate("t")
                    .is_ok()
            );
        }
    }

    #[test]
    fn multi_vector_option_validate_vectors() {
        let opt = MultiVectorOption::new(2);
        assert!(
            opt.validate_vectors("t", &[vec![1.0, 0.0], vec![0.5, 0.5]])
                .is_ok()
        );

        let cases: Vec<(Vec<Vec<f32>>, &str)> = vec![
            (Vec::new(), "at least one vector"),
            (vec![vec![1.0, 0.0], vec![1.0]], "vector 1 has dimension 1"),
            (vec![Vec::new()], "vector 0 has dimension 0"),
            (vec![vec![f32::NAN, 0.0]], "non-finite"),
            (vec![vec![0.0, f32::INFINITY]], "non-finite"),
            (
                vec![vec![0.0, 0.0]; MultiVectorOption::MAX_VECTORS_PER_DOCUMENT + 1],
                "at most 8192 vectors",
            ),
        ];
        for (vectors, expected) in cases {
            let err = opt.validate_vectors("t", &vectors).unwrap_err();
            let msg = err.to_string();
            assert!(msg.starts_with("Invalid argument: field 't': "), "{msg}");
            assert!(msg.contains(expected), "{msg}");
        }
    }

    /// Issue #1457: Flat and IVF accept only the quantizers they implement.
    #[test]
    fn flat_and_ivf_validate_accept_scalar8bit_and_float16() {
        use quantization::QuantizationMethod;

        for quantizer in [QuantizationMethod::Scalar8Bit, QuantizationMethod::Float16] {
            assert!(
                FlatOption::new(8)
                    .quantizer(quantizer)
                    .validate("vec")
                    .is_ok()
            );
            assert!(
                IvfOption::new(8)
                    .quantizer(quantizer)
                    .validate("vec")
                    .is_ok()
            );
        }
    }

    /// Issue #1457: a product quantizer on a Flat or IVF field is rejected
    /// with an error that names the field and the quantizer.
    #[test]
    fn flat_and_ivf_validate_reject_product_quantization() {
        use quantization::QuantizationMethod;

        let pq = QuantizationMethod::ProductQuantization { subvector_count: 4 };
        let flat = FlatOption::new(8)
            .quantizer(pq)
            .validate("vec")
            .unwrap_err();
        assert_eq!(
            flat.to_string(),
            "Invalid argument: field 'vec': Flat does not support quantizer \
             ProductQuantization { subvector_count: 4 }: product quantization is HNSW-only; \
             use Scalar8Bit or Float16"
        );
        let ivf = IvfOption::new(8).quantizer(pq).validate("vec").unwrap_err();
        assert!(
            ivf.to_string().starts_with(
                "Invalid argument: field 'vec': IVF does not support quantizer \
                 ProductQuantization { subvector_count: 4 }"
            ),
            "{ivf}"
        );
    }

    /// Issue #1457: the FastScan variant is rejected the same way.
    #[cfg(feature = "pq-fastscan")]
    #[test]
    fn flat_and_ivf_validate_reject_product_quantization_fastscan() {
        use quantization::QuantizationMethod;

        let fastscan = QuantizationMethod::ProductQuantizationFastScan { subvector_count: 4 };
        for err in [
            FlatOption::new(8).quantizer(fastscan).validate("vec"),
            IvfOption::new(8).quantizer(fastscan).validate("vec"),
        ] {
            let msg = err.unwrap_err().to_string();
            assert!(
                msg.contains("field 'vec'") && msg.contains("ProductQuantizationFastScan"),
                "{msg}"
            );
        }
    }

    /// Issue #1479: each graph and cluster parameter is rejected at the
    /// first value the index cannot work with, and accepted at the next.
    #[test]
    fn validate_rejects_unusable_parameters_at_their_boundaries() {
        let cases: Vec<(FieldOption, FieldOption, &str)> = vec![
            (
                HnswOption::new(8).m(1).into(),
                HnswOption::new(8).m(2).into(),
                "HNSW m must be at least 2, got 1",
            ),
            (
                HnswOption::new(8).ef_construction(0).into(),
                HnswOption::new(8).ef_construction(1).into(),
                "HNSW ef_construction must be at least 1",
            ),
            (
                HnswOption::new(0).into(),
                HnswOption::new(1).into(),
                "HNSW dimension must be greater than 0",
            ),
            (
                IvfOption::new(8).n_clusters(0).into(),
                IvfOption::new(8).n_clusters(1).into(),
                "IVF n_clusters must be at least 1",
            ),
            (
                IvfOption::new(8).n_probe(0).into(),
                IvfOption::new(8).n_probe(1).into(),
                "IVF n_probe must be at least 1",
            ),
            (
                IvfOption::new(0).into(),
                IvfOption::new(1).into(),
                "IVF dimension must be greater than 0",
            ),
            (
                FlatOption::new(0).into(),
                FlatOption::new(1).into(),
                "Flat dimension must be greater than 0",
            ),
        ];
        for (rejected, accepted, reason) in cases {
            let err = rejected.validate("vec").unwrap_err();
            assert_eq!(
                err.to_string(),
                format!("Invalid argument: field 'vec': {reason}")
            );
            assert_eq!(rejected.params_error().as_deref(), Some(reason));
            assert!(accepted.validate("vec").is_ok(), "{accepted:?}");
            assert_eq!(accepted.params_error(), None, "{accepted:?}");
        }
    }

    /// The defaults a schema gets for an omitted key pass the checks.
    #[test]
    fn default_options_validate() {
        for option in [
            FieldOption::Hnsw(HnswOption::default()),
            FieldOption::Ivf(IvfOption::default()),
            FieldOption::Flat(FlatOption::default()),
            FieldOption::MultiVector(MultiVectorOption::default()),
        ] {
            assert!(option.validate("vec").is_ok(), "{option:?}");
        }
        let hnsw: HnswOption = serde_json::from_str(r#"{"dimension": 8}"#).unwrap();
        assert_eq!(hnsw.m, DEFAULT_HNSW_M);
        assert_eq!(hnsw.ef_construction, DEFAULT_HNSW_EF_CONSTRUCTION);
        let ivf: IvfOption = serde_json::from_str(r#"{"dimension": 8}"#).unwrap();
        assert_eq!(ivf.n_clusters, DEFAULT_IVF_N_CLUSTERS);
        assert_eq!(ivf.n_probe, DEFAULT_IVF_N_PROBE);
    }

    /// Issue #1491: an HNSW product quantizer's `subvector_count` is
    /// rejected at the first value the writer cannot encode with, and
    /// accepted at the next, for both product quantizers. `params_error`
    /// reports it, so opening an index that holds one only warns.
    #[test]
    fn validate_rejects_a_subvector_count_that_does_not_fit_the_dimension() {
        let product_quantizers: &[fn(usize) -> quantization::QuantizationMethod] = &[
            |subvector_count| quantization::QuantizationMethod::ProductQuantization {
                subvector_count,
            },
            #[cfg(feature = "pq-fastscan")]
            |subvector_count| quantization::QuantizationMethod::ProductQuantizationFastScan {
                subvector_count,
            },
        ];
        // (dimension, rejected count, accepted count, reason)
        let cases = [
            (8, 0, 1, "subvector_count must be at least 1"),
            (
                8,
                3,
                4,
                "subvector_count 3 must divide dimension 8 (got 8 % 3 = 2)",
            ),
            (
                8,
                9,
                8,
                "subvector_count 9 must divide dimension 8 (got 8 % 9 = 8)",
            ),
            (
                131_072,
                2,
                4,
                "sub-vector dimension 65536 (dimension 131072 / subvector_count 2) exceeds 65535",
            ),
            (65_536, 65_536, 2, "subvector_count 65536 exceeds 65535"),
        ];
        for quantizer in product_quantizers {
            for (dimension, rejected, accepted, reason) in cases {
                let reason = format!("HNSW product quantization {reason}");
                let rejected =
                    FieldOption::Hnsw(HnswOption::new(dimension).quantizer(quantizer(rejected)));
                let err = rejected.validate("vec").unwrap_err();
                assert_eq!(
                    err.to_string(),
                    format!("Invalid argument: field 'vec': {reason}")
                );
                assert_eq!(rejected.params_error(), Some(reason));
                let accepted =
                    FieldOption::Hnsw(HnswOption::new(dimension).quantizer(quantizer(accepted)));
                assert!(accepted.validate("vec").is_ok(), "{accepted:?}");
                assert_eq!(accepted.params_error(), None, "{accepted:?}");
            }
        }
    }

    /// `params_error` is the part of `validate` that opening an index only
    /// warns about, so it must not report whether the index kind implements
    /// the quantizer, which opening still rejects (Issue #1457).
    #[test]
    fn params_error_ignores_quantizer_support() {
        let pq = quantization::QuantizationMethod::ProductQuantization { subvector_count: 4 };
        for option in [
            FieldOption::Flat(FlatOption::new(8).quantizer(pq)),
            FieldOption::Ivf(IvfOption::new(8).quantizer(pq)),
        ] {
            assert!(option.validate("vec").is_err(), "{option:?}");
            assert_eq!(option.params_error(), None, "{option:?}");
        }
        let multi_vector = FieldOption::MultiVector(MultiVectorOption::new(0));
        assert_eq!(
            multi_vector.params_error().as_deref(),
            Some("MultiVector dimension must be greater than 0")
        );
    }

    /// HNSW implements every quantizer, and a MultiVector field has none.
    #[test]
    fn quantizer_support_error_ignores_hnsw_and_multi_vector() {
        let pq = quantization::QuantizationMethod::ProductQuantization { subvector_count: 4 };
        assert!(quantizer_support_error(VectorIndexKind::Hnsw, pq).is_none());
        assert!(quantizer_support_error(VectorIndexKind::MultiVector, pq).is_none());
    }
}
