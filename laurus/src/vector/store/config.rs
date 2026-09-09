//! VectorStore configuration types.
//!
//! This module provides engine configuration, field configuration, and embedder settings.
//!
//! # Configuration with Embedder
//!
//! The recommended way to configure a VectorStore is to provide an `Embedder` directly
//! in the configuration, similar to how `Analyzer` is used in `LexicalStore`.
//!
//! ```no_run
//! # #[cfg(feature = "embeddings-candle")]
//! # {
//! use laurus::embedding::per_field::PerFieldEmbedder;
//! use laurus::embedding::candle_bert_embedder::CandleBertEmbedder;
//! use laurus::embedding::embedder::Embedder;
//! use laurus::vector::store::config::VectorIndexConfig;
//! use laurus::vector::core::field::FlatOption;
//! use std::sync::Arc;
//!
//! # fn example() -> laurus::Result<()> {
//! let text_embedder: Arc<dyn Embedder> = Arc::new(
//!     CandleBertEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
//! );
//!
//! let embedder = Arc::new(PerFieldEmbedder::new(text_embedder));
//!
//! let config = VectorIndexConfig::builder()
//!     .embedder(embedder)
//!     .add_field("title", FlatOption::new(384))?
//!     .build()?;
//! # Ok(())
//! # }
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::embedding::embedder::Embedder;
use crate::embedding::precomputed::PrecomputedEmbedder;
use crate::error::{LaurusError, Result};
use crate::lexical::store::config::LexicalIndexConfig;
use crate::maintenance::deletion::DeletionConfig;
use crate::vector::core::field::FieldOption;

/// Configuration for a single vector collection.
///
/// This configuration should be created using the builder pattern with an `Embedder`.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "embeddings-candle")]
/// # {
/// use laurus::embedding::per_field::PerFieldEmbedder;
/// use laurus::embedding::candle_bert_embedder::CandleBertEmbedder;
/// use laurus::embedding::embedder::Embedder;
/// use laurus::vector::store::config::{VectorIndexConfig, VectorFieldConfig};
/// use laurus::vector::core::field::{VectorIndexKind, FlatOption};
/// use laurus::vector::core::distance::DistanceMetric;
/// use std::sync::Arc;
///
/// # fn example() -> laurus::Result<()> {
/// let text_embedder: Arc<dyn Embedder> = Arc::new(
///     CandleBertEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?
/// );
///
/// let embedder = Arc::new(PerFieldEmbedder::new(text_embedder));
///
/// let config = VectorIndexConfig::builder()
///     .embedder(embedder)
///     .add_field("title", FlatOption::new(384))?
///     .build()?;
/// # Ok(())
/// # }
/// # }
/// ```
#[derive(Clone)]
pub struct VectorIndexConfig {
    /// Field configurations.
    pub fields: HashMap<String, VectorFieldConfig>,

    /// Default fields for search.
    pub default_fields: Vec<String>,

    /// Metadata for the collection.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Embedder for text and image fields.
    pub embedder: Arc<dyn Embedder>,

    /// Deletion maintenance configuration.
    pub deletion_config: DeletionConfig,

    /// Shard ID for the collection.
    pub shard_id: u16,

    /// Metadata index configuration (LexicalStore).
    pub metadata_config: LexicalIndexConfig,
}

impl std::fmt::Debug for VectorIndexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndexConfig")
            .field("fields", &self.fields)
            .field("default_fields", &self.default_fields)
            .field("metadata", &self.metadata)
            .field("embedder", &format_args!("{:?}", self.embedder))
            .field("deletion_config", &self.deletion_config)
            .field("shard_id", &self.shard_id)
            .field("metadata_config", &self.metadata_config)
            .finish()
    }
}

impl VectorIndexConfig {
    /// Create a new builder for VectorIndexConfig.
    pub fn builder() -> VectorIndexConfigBuilder {
        VectorIndexConfigBuilder::new()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        for field in &self.default_fields {
            if !self.fields.contains_key(field) {
                return Err(LaurusError::invalid_config(format!(
                    "default field '{field}' is not defined"
                )));
            }
        }
        Ok(())
    }

    /// Get the embedder for this configuration.
    pub fn get_embedder(&self) -> &Arc<dyn Embedder> {
        &self.embedder
    }

    /// Build a per-field [`VectorIndexTypeConfig`] map from this
    /// collection's schema (Issue [#948](https://github.com/mosuka/laurus/issues/948)).
    ///
    /// Replaces the old `VectorStore::extract_index_type_config`, which
    /// collapsed every vector field down to whichever one happened to be
    /// first out of `fields` (a `HashMap`, so non-deterministic) --
    /// silently discarding every other field's dimension, distance metric,
    /// and HNSW parameters. This converts EVERY field carrying a `vector`
    /// option, keyed by field name in a [`BTreeMap`] for deterministic
    /// ordering (feeds directly into
    /// [`MultiFieldVectorIndex::open_or_create`](crate::vector::index::multi_field::MultiFieldVectorIndex::open_or_create)).
    ///
    /// Fields with `vector: None` (lexical-only fields) are skipped. An
    /// empty result (no field carries a `vector` option) is valid: it
    /// means this collection has no vector fields at all.
    pub fn field_index_configs(
        &self,
    ) -> std::collections::BTreeMap<String, crate::vector::index::config::VectorIndexTypeConfig>
    {
        let mut out = std::collections::BTreeMap::new();
        for (name, field_config) in &self.fields {
            let Some(vector_opt) = &field_config.vector else {
                continue;
            };
            out.insert(
                name.clone(),
                build_field_index_config(vector_opt, self.embedder.clone(), &self.deletion_config),
            );
        }
        out
    }
}

/// Convert one field's schema-level [`FieldOption`] into a full
/// [`VectorIndexTypeConfig`], given the collection-wide embedder and
/// deletion policy (Issue [#948](https://github.com/mosuka/laurus/issues/948)).
///
/// Shared by [`VectorIndexConfig::field_index_configs`] (batch conversion
/// at `VectorStore` construction) and
/// [`crate::vector::store::VectorStore::add_field`] (single-field
/// conversion at dynamic schema growth), so the two paths convert a
/// [`FieldOption`] identically and can never drift apart.
pub fn build_field_index_config(
    vector_opt: &FieldOption,
    embedder: Arc<dyn Embedder>,
    deletion_config: &DeletionConfig,
) -> crate::vector::index::config::VectorIndexTypeConfig {
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::index::config::{
        FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, VectorIndexTypeConfig,
    };

    match vector_opt {
        FieldOption::Flat(opt) => VectorIndexTypeConfig::Flat(FlatIndexConfig {
            dimension: opt.dimension,
            distance_metric: opt.distance,
            rerank_storage: opt.rerank_storage,
            quantization_method: opt.quantizer,
            normalize_vectors: opt.distance == DistanceMetric::Cosine,
            auto_compaction: deletion_config.auto_compaction,
            compaction_threshold: deletion_config.compaction_threshold,
            embedder,
            ..Default::default()
        }),
        FieldOption::Hnsw(opt) => VectorIndexTypeConfig::HNSW(HnswIndexConfig {
            auto_compaction: deletion_config.auto_compaction,
            compaction_threshold: deletion_config.compaction_threshold,
            embedder,
            ..HnswIndexConfig::from_hnsw_option(opt)
        }),
        FieldOption::Ivf(opt) => VectorIndexTypeConfig::IVF(IvfIndexConfig {
            dimension: opt.dimension,
            distance_metric: opt.distance,
            n_clusters: opt.n_clusters,
            n_probe: opt.n_probe,
            rerank_storage: opt.rerank_storage,
            quantization_method: opt.quantizer,
            normalize_vectors: opt.distance == DistanceMetric::Cosine,
            auto_compaction: deletion_config.auto_compaction,
            compaction_threshold: deletion_config.compaction_threshold,
            embedder,
            ..Default::default()
        }),
    }
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self::builder()
            .build()
            .expect("Default config should be valid")
    }
}

/// Builder for VectorIndexConfig.
pub struct VectorIndexConfigBuilder {
    fields: HashMap<String, VectorFieldConfig>,
    default_fields: Vec<String>,
    metadata: HashMap<String, serde_json::Value>,
    embedder: Option<Arc<dyn Embedder>>,
    deletion_config: Option<DeletionConfig>,
    shard_id: Option<u16>,
    metadata_config: Option<LexicalIndexConfig>,
}

impl VectorIndexConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            default_fields: Vec::new(),
            metadata: HashMap::new(),
            embedder: None,
            deletion_config: None,
            shard_id: None,
            metadata_config: None,
        }
    }

    /// Set the embedder for all fields.
    ///
    /// Use `PerFieldEmbedder` for field-specific embedders.
    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Add a field configuration.
    pub fn field(mut self, name: impl Into<String>, config: VectorFieldConfig) -> Self {
        let name = name.into();
        if !self.default_fields.contains(&name) {
            self.default_fields.push(name.clone());
        }
        self.fields.insert(name, config);
        self
    }

    /// Add a vector field with explicit options.
    ///
    /// The option can be a `VectorOption` or any type that converts into it
    /// (e.g. `FlatOption`, `HnswOption`).
    ///
    /// # Example
    /// ```no_run
    /// # use laurus::vector::store::config::VectorIndexConfig;
    /// # use laurus::vector::core::field::FlatOption;
    /// # fn example() {
    /// let _ = VectorIndexConfig::builder()
    ///     .add_field("title", FlatOption::default().dimension(384));
    /// # }
    /// ```
    pub fn add_field(
        mut self,
        name: impl Into<String>,
        option: impl Into<FieldOption>,
    ) -> Result<Self> {
        let name = name.into();
        let config = VectorFieldConfig {
            vector: Some(option.into()),
            lexical: None,
        };

        if !self.default_fields.contains(&name) {
            self.default_fields.push(name.clone());
        }
        self.fields.insert(name, config);
        Ok(self)
    }

    /// Add an image field.
    ///
    /// This is an alias for `add_field` but intended for image vectors.
    pub fn image_field(
        self,
        name: impl Into<String>,
        option: impl Into<FieldOption>,
    ) -> Result<Self> {
        self.add_field(name, option)
    }

    /// Add a default field for search.
    pub fn default_field(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        if !self.default_fields.contains(&name) {
            self.default_fields.push(name);
        }
        self
    }

    /// Set the default fields for search.
    pub fn default_fields(mut self, fields: Vec<String>) -> Self {
        self.default_fields = fields;
        self
    }

    /// Add metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set deletion configuration.
    pub fn deletion_config(mut self, config: DeletionConfig) -> Self {
        self.deletion_config = Some(config);
        self
    }

    /// Set shard ID.
    pub fn shard_id(mut self, shard_id: u16) -> Self {
        self.shard_id = Some(shard_id);
        self
    }

    /// Set metadata index configuration.
    pub fn metadata_config(mut self, config: LexicalIndexConfig) -> Self {
        self.metadata_config = Some(config);
        self
    }

    /// Build the configuration.
    ///
    /// If no embedder is set, defaults to `PrecomputedEmbedder` for pre-computed vectors.
    pub fn build(self) -> Result<VectorIndexConfig> {
        let embedder = self
            .embedder
            .unwrap_or_else(|| Arc::new(PrecomputedEmbedder::new()));

        let config = VectorIndexConfig {
            fields: self.fields,
            default_fields: self.default_fields,
            metadata: self.metadata,
            embedder,
            deletion_config: self.deletion_config.unwrap_or_default(),
            shard_id: self.shard_id.unwrap_or(0),
            metadata_config: self.metadata_config.unwrap_or_default(),
        };
        config.validate()?;
        Ok(config)
    }
}

impl Default for VectorIndexConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Implement Serialize manually to skip the embedder field
impl Serialize for VectorIndexConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("VectorIndexConfig", 5)?;
        state.serialize_field("fields", &self.fields)?;
        state.serialize_field("default_fields", &self.default_fields)?;
        state.serialize_field("metadata", &self.metadata)?;
        state.serialize_field("deletion_config", &self.deletion_config)?;
        state.serialize_field("shard_id", &self.shard_id)?;
        state.serialize_field("metadata_config", &self.metadata_config)?;
        state.end()
    }
}

// Implement Deserialize manually to handle the embedder field
impl<'de> Deserialize<'de> for VectorIndexConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct VectorIndexConfigHelper {
            fields: HashMap<String, VectorFieldConfig>,
            default_fields: Vec<String>,
            #[serde(default)]
            metadata: HashMap<String, serde_json::Value>,
            #[serde(default)]
            deletion_config: DeletionConfig,
            #[serde(default)]
            shard_id: u16,
            #[serde(default)]
            metadata_config: LexicalIndexConfig,
        }

        let helper = VectorIndexConfigHelper::deserialize(deserializer)?;
        Ok(VectorIndexConfig {
            fields: helper.fields,
            default_fields: helper.default_fields,
            metadata: helper.metadata,
            deletion_config: helper.deletion_config,
            shard_id: helper.shard_id,
            metadata_config: helper.metadata_config,
            // Default to PrecomputedEmbedder; can be replaced programmatically
            embedder: Arc::new(PrecomputedEmbedder::new()),
        })
    }
}

/// Configuration for a single vector field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldConfig {
    /// Configuration options for the vector field (index type, dimension, distance metric, etc.).
    ///
    /// When `None`, the field has no vector index.
    #[serde(default)]
    pub vector: Option<FieldOption>,
    /// Configuration options for the lexical field.
    pub lexical: Option<crate::lexical::core::field::FieldOption>,
}

impl Default for VectorFieldConfig {
    fn default() -> Self {
        Self {
            vector: Some(FieldOption::default()),
            lexical: Some(crate::lexical::core::field::FieldOption::default()),
        }
    }
}

// Moved to crate::vector::core::field
// use crate::vector::core::field::{VectorOption, FlatOption, HnswOption, IvfOption, VectorIndexKind};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::distance::DistanceMetric;
    use crate::vector::core::field::{FieldOption, FlatOption, HnswOption, IvfOption};
    use crate::vector::core::quantization::QuantizationMethod;
    use crate::vector::core::rerank::RerankStorageKind;
    use crate::vector::index::config::VectorIndexTypeConfig;

    /// Core gate (Issue #948): `field_index_configs` must convert EVERY
    /// field, not just the first one out of the `HashMap` -- the exact bug
    /// the old (now-removed) `VectorStore::extract_index_type_config`
    /// single-config collapse caused.
    #[test]
    fn field_index_configs_converts_every_field_not_just_the_first() {
        let config = VectorIndexConfig::builder()
            .field(
                "title_vec",
                VectorFieldConfig {
                    vector: Some(FieldOption::Hnsw(HnswOption {
                        dimension: 8,
                        distance: DistanceMetric::Cosine,
                        ..Default::default()
                    })),
                    lexical: None,
                },
            )
            .field(
                "body_vec",
                VectorFieldConfig {
                    vector: Some(FieldOption::Hnsw(HnswOption {
                        dimension: 32,
                        distance: DistanceMetric::Euclidean,
                        ..Default::default()
                    })),
                    lexical: None,
                },
            )
            .build()
            .unwrap();

        let field_configs = config.field_index_configs();
        assert_eq!(
            field_configs.len(),
            2,
            "both fields must be present, not just one"
        );

        match &field_configs["title_vec"] {
            VectorIndexTypeConfig::HNSW(c) => {
                assert_eq!(c.dimension, 8);
                assert_eq!(c.distance_metric, DistanceMetric::Cosine);
            }
            other => panic!("expected HNSW config for title_vec, got {other:?}"),
        }
        match &field_configs["body_vec"] {
            VectorIndexTypeConfig::HNSW(c) => {
                assert_eq!(c.dimension, 32);
                assert_eq!(c.distance_metric, DistanceMetric::Euclidean);
            }
            other => panic!("expected HNSW config for body_vec, got {other:?}"),
        }
    }

    /// Lexical-only fields (`vector: None`) are skipped, and a collection
    /// with zero vector fields yields an empty map (valid: no vector index
    /// is needed).
    #[test]
    fn field_index_configs_skips_lexical_only_fields() {
        let config = VectorIndexConfig::builder()
            .field(
                "lexical_only",
                VectorFieldConfig {
                    vector: None,
                    lexical: Some(crate::lexical::core::field::FieldOption::default()),
                },
            )
            .build()
            .unwrap();

        assert!(config.field_index_configs().is_empty());
    }

    /// Issue #790-style regression, now per-field: rerank storage and
    /// quantizer must propagate through `field_index_configs`, not just
    /// the pre-existing dimension/distance/m fields.
    #[test]
    fn field_index_configs_propagates_rerank_and_quantizer() {
        let config = VectorIndexConfig::builder()
            .field(
                "vec",
                VectorFieldConfig {
                    vector: Some(FieldOption::Hnsw(HnswOption {
                        dimension: 8,
                        distance: DistanceMetric::Euclidean,
                        m: 5,
                        ef_construction: 33,
                        default_ef_search: Some(77),
                        quantizer: QuantizationMethod::ProductQuantization { subvector_count: 4 },
                        rerank_storage: Some(RerankStorageKind::F32),
                        ..Default::default()
                    })),
                    lexical: None,
                },
            )
            .build()
            .unwrap();

        match &config.field_index_configs()["vec"] {
            VectorIndexTypeConfig::HNSW(c) => {
                assert_eq!(c.rerank_storage, Some(RerankStorageKind::F32));
                assert_eq!(
                    c.quantization_method,
                    QuantizationMethod::ProductQuantization { subvector_count: 4 }
                );
                assert_eq!(c.dimension, 8);
                assert_eq!(c.distance_metric, DistanceMetric::Euclidean);
                assert_eq!(c.m, 5);
                assert_eq!(c.ef_construction, 33);
                assert_eq!(c.default_ef_search, Some(77));
            }
            other => panic!("expected HNSW config, got {other:?}"),
        }
    }

    /// Issue #1080: `build_field_index_config`'s Flat and IVF branches used
    /// to drop `opt.quantizer` on the floor (falling back to
    /// `..Default::default()`'s `Scalar8Bit`), while the HNSW branch
    /// propagated it correctly. A `Flat`/`Ivf` field configured with a
    /// non-default quantizer must convert with that same quantizer, or
    /// `Engine::update_field` rebuilding a quantizer change would silently
    /// no-op for these two index kinds.
    #[test]
    fn field_index_configs_propagates_quantizer_for_flat_and_ivf() {
        let flat_config = VectorIndexConfig::builder()
            .field(
                "vec",
                VectorFieldConfig {
                    vector: Some(FieldOption::Flat(FlatOption {
                        quantizer: QuantizationMethod::ProductQuantization { subvector_count: 4 },
                        rerank_storage: Some(RerankStorageKind::F32),
                        ..Default::default()
                    })),
                    lexical: None,
                },
            )
            .build()
            .unwrap();
        match &flat_config.field_index_configs()["vec"] {
            VectorIndexTypeConfig::Flat(c) => {
                assert_eq!(
                    c.quantization_method,
                    QuantizationMethod::ProductQuantization { subvector_count: 4 }
                );
                assert_eq!(c.rerank_storage, Some(RerankStorageKind::F32));
            }
            other => panic!("expected Flat config, got {other:?}"),
        }

        let ivf_config = VectorIndexConfig::builder()
            .field(
                "vec",
                VectorFieldConfig {
                    vector: Some(FieldOption::Ivf(IvfOption {
                        quantizer: QuantizationMethod::ProductQuantization { subvector_count: 4 },
                        rerank_storage: Some(RerankStorageKind::F32),
                        ..Default::default()
                    })),
                    lexical: None,
                },
            )
            .build()
            .unwrap();
        match &ivf_config.field_index_configs()["vec"] {
            VectorIndexTypeConfig::IVF(c) => {
                assert_eq!(
                    c.quantization_method,
                    QuantizationMethod::ProductQuantization { subvector_count: 4 }
                );
                assert_eq!(c.rerank_storage, Some(RerankStorageKind::F32));
            }
            other => panic!("expected IVF config, got {other:?}"),
        }
    }

    /// `normalize_vectors` stays metric-conditional (Issue #794) across all
    /// three index kinds when derived per-field.
    #[test]
    fn field_index_configs_normalize_is_metric_conditional() {
        fn normalize_of(opt: FieldOption) -> bool {
            let config = VectorIndexConfig::builder()
                .field(
                    "vec",
                    VectorFieldConfig {
                        vector: Some(opt),
                        lexical: None,
                    },
                )
                .build()
                .unwrap();
            match &config.field_index_configs()["vec"] {
                VectorIndexTypeConfig::HNSW(c) => c.normalize_vectors,
                VectorIndexTypeConfig::Flat(c) => c.normalize_vectors,
                VectorIndexTypeConfig::IVF(c) => c.normalize_vectors,
            }
        }

        let hnsw = |distance| {
            FieldOption::Hnsw(HnswOption {
                distance,
                ..Default::default()
            })
        };
        let flat = |distance| {
            FieldOption::Flat(FlatOption {
                distance,
                ..Default::default()
            })
        };
        let ivf = |distance| {
            FieldOption::Ivf(IvfOption {
                distance,
                ..Default::default()
            })
        };

        for opt_fn in [hnsw, flat, ivf] {
            assert!(normalize_of(opt_fn(DistanceMetric::Cosine)));
            assert!(!normalize_of(opt_fn(DistanceMetric::Euclidean)));
        }
    }
}
