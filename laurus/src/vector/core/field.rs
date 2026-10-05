//! Vector field configuration options.
//!
//! This module defines options for configuring vector fields, including
//! index types and parameters for different algorithms (Flat, HNSW, IVF),
//! and the multi-vector field used for late-interaction rescoring.

use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization;
use crate::vector::core::rerank::RerankStorageKind;

fn default_dimension() -> usize {
    128
}

fn default_getting_m() -> usize {
    16
}

fn default_getting_ef_construction() -> usize {
    200
}

fn default_getting_n_clusters() -> usize {
    100
}

fn default_getting_n_probe() -> usize {
    1
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
    /// When `None` (the default), the searcher uses an internal fallback of
    /// `50` so existing schemas behave unchanged. Per-query
    /// [`VectorIndexQueryParams::ef_search`] always takes precedence over this
    /// schema-level default.
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
    /// [`DistanceMetric::DotProduct`] are supported. Defaults to
    /// [`DistanceMetric::Cosine`].
    #[serde(default = "default_distance_metric")]
    pub distance: DistanceMetric,
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
            embedder: None,
        }
    }
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
    /// Returns [`LaurusError::invalid_argument`] when the dimension is zero
    /// or the distance is neither `Cosine` nor `DotProduct` (see
    /// [`multi_vector_params_error`]).
    pub fn validate(&self, field_name: &str) -> Result<()> {
        match multi_vector_params_error(self.dimension, self.distance) {
            Some(reason) => Err(LaurusError::invalid_argument(format!(
                "field '{field_name}': {reason}"
            ))),
            None => Ok(()),
        }
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

/// Why `dimension` / `distance` cannot describe a multi-vector field, if
/// they cannot.
///
/// Late interaction sums each query vector's best similarity, which is
/// meaningless for a distance where smaller is better, so only `Cosine` and
/// `DotProduct` are accepted.
pub(crate) fn multi_vector_params_error(
    dimension: usize,
    distance: DistanceMetric,
) -> Option<String> {
    if dimension == 0 {
        return Some("MultiVector dimension must be greater than 0".to_string());
    }
    match distance {
        DistanceMetric::Cosine | DistanceMetric::DotProduct => None,
        other => Some(format!(
            "MultiVector distance must be Cosine or DotProduct, got {}",
            other.name()
        )),
    }
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
    /// searcher falls back to its internal default (`50`). See
    /// [`HnswOption::default_ef_search`] for the full precedence rules.
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

        let named = MultiVectorOption::new(128).embedder("colbert");
        let back: MultiVectorOption =
            serde_json::from_str(&serde_json::to_string(&named).unwrap()).unwrap();
        assert_eq!(back.embedder.as_deref(), Some("colbert"));
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
}
