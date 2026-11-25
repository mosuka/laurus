//! Factory for creating vector index instances.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::embedding::embedder::Embedder;
use crate::error::Result;
use crate::storage::Storage;
use crate::vector::index::VectorIndex;
use crate::vector::index::config::VectorIndexTypeConfig;
use crate::vector::index::flat::FlatIndex;
use crate::vector::index::hnsw::HnswIndex;
use crate::vector::index::ivf::IvfIndex;
use crate::vector::index::multi_field::MultiFieldVectorIndex;

/// Factory for creating vector index instances.
///
/// This factory follows the Factory design pattern to create appropriate
/// index implementations based on the provided configuration.
///
/// # Design Benefits
///
/// - **Decoupling**: Client code doesn't need to know about concrete index types
/// - **Extensibility**: New index types can be added by extending the enum
/// - **Type safety**: Pattern matching ensures all cases are handled
///
/// # Example with StorageFactory
///
/// ```
/// use laurus::vector::index::factory::VectorIndexFactory;
/// use laurus::vector::index::config::VectorIndexTypeConfig;
/// use laurus::storage::{StorageFactory, StorageConfig};
/// use laurus::storage::memory::MemoryStorageConfig;
///
/// # fn main() -> laurus::Result<()> {
/// // Create storage using factory
/// let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
///
/// // Create index using factory
/// let config = VectorIndexTypeConfig::default();
/// let index = VectorIndexFactory::create(storage, "test_index", config)?;
/// # Ok(())
/// # }
/// ```
pub struct VectorIndexFactory;

impl VectorIndexFactory {
    /// Create a new vector index with the given storage and configuration.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend (created using `StorageFactory`)
    /// * `config` - Index configuration enum containing type-specific settings
    ///
    /// # Returns
    ///
    /// A boxed trait object implementing `VectorIndex` trait.
    /// The concrete type is determined by the config variant.
    ///
    /// # PQ codebook resolution
    ///
    /// The shared PQ codebook that an HNSW config names
    /// ([`HnswIndexConfig::pq_codebook_path`](crate::vector::index::config::HnswIndexConfig::pq_codebook_path))
    /// is loaded from `storage`, as [`Self::open`] and
    /// [`Self::open_or_create`] do too. For an index with a storage of its
    /// own, that is the root the path is relative to.
    ///
    /// # Errors
    ///
    /// Returns an error when the index cannot be created, or when the file of
    /// the codebook exists and cannot be read.
    ///
    /// # Example
    ///
    /// ```
    /// use laurus::vector::index::factory::VectorIndexFactory;
    /// use laurus::vector::index::config::{VectorIndexTypeConfig, FlatIndexConfig};
    /// use laurus::storage::{StorageFactory, StorageConfig};
    /// use laurus::storage::file::FileStorageConfig;
    /// use tempfile::TempDir;
    ///
    /// # fn main() -> laurus::Result<()> {
    /// // Create file storage in a private temporary directory. A fixed
    /// // path would race the other doctests: index control files
    /// // (metadata.json, segments.json) are per-directory, and this
    /// // doctest once failed on another doctest's lexical manifest.
    /// let dir = TempDir::new().unwrap();
    /// let storage_config = StorageConfig::File(FileStorageConfig::new(dir.path()));
    /// let storage = StorageFactory::create(storage_config)?;
    ///
    /// // Create flat index
    /// let index_config = VectorIndexTypeConfig::Flat(FlatIndexConfig::default());
    /// let index = VectorIndexFactory::create(storage, "test_index", index_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create(
        storage: Arc<dyn Storage>,
        name: &str,
        mut config: VectorIndexTypeConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        config.resolve_pq_codebook(storage.as_ref())?;
        Self::dispatch(storage, name, config, false)
    }

    /// Open or create the index of `config`'s kind.
    ///
    /// The config is taken as it is: a shared PQ codebook must be in it
    /// already (Issue #1738).
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend
    /// * `name` - Index name
    /// * `config` - Index configuration
    /// * `open_only` - Whether a single-file index is opened (`true`) or
    ///   created (`false`). The segmented layout takes the same path for both.
    fn dispatch(
        storage: Arc<dyn Storage>,
        name: &str,
        config: VectorIndexTypeConfig,
        open_only: bool,
    ) -> Result<Box<dyn VectorIndex>> {
        match config {
            VectorIndexTypeConfig::Flat(flat_config) => {
                // Both arms come through here (Issue #889 PR-4): the
                // segmented path's `open_or_create` opens existing manifests
                // and migrates legacy monolithic indexes; the flag-off path
                // is reverse-guarded against segmented directories.
                Self::dispatch_flat(storage, name, flat_config, open_only)
            }
            VectorIndexTypeConfig::HNSW(hnsw_config) => {
                // As for Flat (#882).
                Self::dispatch_hnsw(storage, name, hnsw_config, open_only)
            }
            VectorIndexTypeConfig::IVF(ivf_config) => {
                // As for Flat (Issue #889 PR-6).
                Self::dispatch_ivf(storage, name, ivf_config, open_only)
            }
            VectorIndexTypeConfig::MultiVector(mv_config) => {
                Self::dispatch_multi_vector(storage, name, mv_config)
            }
        }
    }

    /// Open or create a multi-vector index (Issue #1177).
    ///
    /// Multi-vector indexes only exist in the segmented layout and never
    /// write a `metadata.json`, so creating and opening take the same path.
    fn dispatch_multi_vector(
        storage: Arc<dyn Storage>,
        name: &str,
        config: crate::vector::index::config::MultiVectorIndexConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        let index = crate::vector::index::multivector::MultiVectorIndex::open_or_create(
            storage, name, config,
        )?;
        Ok(Box::new(index))
    }

    /// Dispatch a Flat config to the segmented or monolithic implementation
    /// (Issue #889 PR-4, mirroring `dispatch_hnsw`).
    ///
    /// The `segmented` flag (defaulting `false` until #889 PR-7) routes to
    /// [`SegmentedFlatIndex::open_or_create`], which handles create, open,
    /// AND the zero-copy migration of a legacy monolithic index — so BOTH
    /// factory arms must come through here: a legacy index always has a
    /// `metadata.json` and therefore always takes the *open* arm, which is
    /// exactly where the migration must fire.
    ///
    /// With the flag off, opening a directory that contains a segment
    /// manifest is rejected: the monolithic index would silently serve only
    /// the migrated segment-0 data and hide every later segment.
    fn dispatch_flat(
        storage: Arc<dyn Storage>,
        name: &str,
        flat_config: crate::vector::index::config::FlatIndexConfig,
        open_only: bool,
    ) -> Result<Box<dyn VectorIndex>> {
        use crate::vector::index::flat::segmented::SegmentedFlatIndex;

        if flat_config.segmented {
            let index = SegmentedFlatIndex::open_or_create(storage, name, flat_config)?;
            return Ok(Box::new(index));
        }
        // Reverse guard: a segmented directory must not be opened
        // monolithically — the single-file view would silently hide every
        // segment sealed after the migration.
        if storage.file_exists("segments.json") {
            return Err(crate::error::LaurusError::invalid_config(format!(
                "index '{name}' uses the segmented layout (segments.json exists) but \
                 `segmented` is disabled; enable it to open this index"
            )));
        }
        let index = if open_only {
            FlatIndex::open(storage, name, flat_config)?
        } else {
            FlatIndex::create(storage, name, flat_config)?
        };
        Ok(Box::new(index))
    }

    /// Dispatch an HNSW config to the segmented or monolithic implementation
    /// (#634 / #882).
    ///
    /// The `segmented` flag (the default since #882) routes to
    /// [`SegmentedHnswIndex::open_or_create`], which handles create, open,
    /// AND the zero-copy migration of a legacy monolithic index — so BOTH
    /// factory arms must come through here: a legacy index always has a
    /// `metadata.json` and therefore always takes the *open* arm, which is
    /// exactly where the migration must fire (#882 review).
    ///
    /// With the flag off, opening a directory that contains a segment
    /// manifest is rejected: the monolithic index would silently serve only
    /// the migrated segment-0 data and hide every later segment.
    fn dispatch_hnsw(
        storage: Arc<dyn Storage>,
        name: &str,
        hnsw_config: crate::vector::index::config::HnswIndexConfig,
        open_only: bool,
    ) -> Result<Box<dyn VectorIndex>> {
        use crate::vector::index::hnsw::segmented::SegmentedHnswIndex;

        if hnsw_config.segmented {
            let index = SegmentedHnswIndex::open_or_create(storage, name, hnsw_config)?;
            return Ok(Box::new(index));
        }
        // Reverse guard (#882 review): a segmented directory must not be
        // opened monolithically — the single-file view would silently hide
        // every segment sealed after the migration.
        if storage.file_exists("segments.json") {
            return Err(crate::error::LaurusError::invalid_config(format!(
                "index '{name}' uses the segmented layout (segments.json exists) but \
                 `segmented` is disabled; enable it to open this index"
            )));
        }
        let index = if open_only {
            HnswIndex::open(storage, name, hnsw_config)?
        } else {
            HnswIndex::create(storage, name, hnsw_config)?
        };
        Ok(Box::new(index))
    }

    /// Dispatch an IVF config to the segmented or monolithic implementation
    /// (Issue #889 PR-6, mirroring `dispatch_flat`/`dispatch_hnsw`).
    ///
    /// The `segmented` flag (defaulting `false` until #889 PR-7) routes to
    /// [`SegmentedIvfIndex::open_or_create`], which handles create, open,
    /// AND the zero-copy migration of a legacy monolithic index — so BOTH
    /// factory arms must come through here: a legacy index always has a
    /// `metadata.json` and therefore always takes the *open* arm, which is
    /// exactly where the migration must fire.
    ///
    /// With the flag off, opening a directory that contains a segment
    /// manifest is rejected: the monolithic index would silently serve only
    /// the migrated segment-0 data and hide every later segment.
    fn dispatch_ivf(
        storage: Arc<dyn Storage>,
        name: &str,
        ivf_config: crate::vector::index::config::IvfIndexConfig,
        open_only: bool,
    ) -> Result<Box<dyn VectorIndex>> {
        use crate::vector::index::ivf::segmented::SegmentedIvfIndex;

        if ivf_config.segmented {
            let index = SegmentedIvfIndex::open_or_create(storage, name, ivf_config)?;
            return Ok(Box::new(index));
        }
        // Reverse guard: a segmented directory must not be opened
        // monolithically — the single-file view would silently hide every
        // segment sealed after the migration.
        if storage.file_exists("segments.json") {
            return Err(crate::error::LaurusError::invalid_config(format!(
                "index '{name}' uses the segmented layout (segments.json exists) but \
                 `segmented` is disabled; enable it to open this index"
            )));
        }
        let index = if open_only {
            IvfIndex::open(storage, name, ivf_config)?
        } else {
            IvfIndex::create(storage, name, ivf_config)?
        };
        Ok(Box::new(index))
    }

    /// Open an existing index or create a new one if it doesn't exist.
    ///
    /// This is the recommended method for general use, as it handles both
    /// creation and opening transparently.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend
    /// * `name` - Index name
    /// * `config` - Index configuration
    ///
    /// # Returns
    ///
    /// A boxed index implementation.
    ///
    /// # Errors
    ///
    /// As [`Self::create`], which says where a shared PQ codebook is loaded
    /// from, and [`Self::open`].
    pub fn open_or_create(
        storage: Arc<dyn Storage>,
        name: &str,
        mut config: VectorIndexTypeConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        config.resolve_pq_codebook(storage.as_ref())?;
        Self::open_or_create_resolved(storage, name, config)
    }

    /// [`Self::open_or_create`] with a config whose shared PQ codebook, if it
    /// names one, has been loaded, for the index of a field, whose `storage`
    /// is a directory of the vector storage (Issue #1738).
    ///
    /// `pq_codebook_path` is relative to the root of the vector storage. A
    /// file of that name in the field's directory, such as the codebook of a
    /// field `a/b` in the directory of a field `a`, is not the field's, also
    /// when its own is missing.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend
    /// * `name` - Index name
    /// * `config` - Index configuration, with its codebook
    ///
    /// # Returns
    ///
    /// A boxed index implementation.
    pub(crate) fn open_or_create_resolved(
        storage: Arc<dyn Storage>,
        name: &str,
        config: VectorIndexTypeConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        let open_only = storage.file_exists("metadata.json");
        Self::dispatch(storage, name, config, open_only)
    }

    /// Open an existing vector index with the given storage and configuration.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend containing the existing index
    /// * `config` - Index configuration (must match the stored index type)
    ///
    /// # Returns
    ///
    /// A boxed index implementation based on the configured index type.
    ///
    /// # Errors
    ///
    /// Returns an error when the index cannot be opened, or when the file of
    /// the shared PQ codebook the config names exists and cannot be read (see
    /// [`Self::create`]).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use laurus::vector::index::factory::VectorIndexFactory;
    /// use laurus::vector::index::config::{VectorIndexTypeConfig, FlatIndexConfig};
    /// use laurus::storage::file::{FileStorage, FileStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> laurus::Result<()> {
    /// let storage = Arc::new(FileStorage::new("./index", FileStorageConfig::new("./index"))?);
    /// let config = VectorIndexTypeConfig::Flat(FlatIndexConfig::default());
    /// let index = VectorIndexFactory::open(storage, "test_index", config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(
        storage: Arc<dyn Storage>,
        name: &str,
        mut config: VectorIndexTypeConfig,
    ) -> Result<Box<dyn VectorIndex>> {
        config.resolve_pq_codebook(storage.as_ref())?;
        Self::dispatch(storage, name, config, true)
    }

    /// Open or create a [`MultiFieldVectorIndex`], one independent
    /// sub-index per entry in `field_configs` (Issue
    /// [#948](https://github.com/mosuka/laurus/issues/948)).
    ///
    /// Callers typically derive `field_configs` from
    /// [`VectorIndexConfig::field_index_configs`](crate::vector::store::config::VectorIndexConfig::field_index_configs),
    /// the per-field replacement for the old
    /// `VectorStore::extract_index_type_config`.
    ///
    /// # PQ codebook resolution
    ///
    /// Any HNSW field config carrying a
    /// [`HnswIndexConfig::pq_codebook_path`](crate::vector::index::config::HnswIndexConfig::pq_codebook_path)
    /// has its shared codebook resolved here, against the ROOT `storage` --
    /// *before* [`MultiFieldVectorIndex`] wraps each field in its own
    /// `PrefixedStorage` sub-namespace. Nothing looks for the codebook file
    /// in the field-scoped storage, where a file of that name would belong to
    /// something else (see the `multi_field` module docs' PQ codebook caveat,
    /// Issue #1738).
    pub fn open_or_create_multi_field(
        storage: Arc<dyn Storage>,
        field_configs: &BTreeMap<String, VectorIndexTypeConfig>,
        embedder: Arc<dyn Embedder>,
    ) -> Result<MultiFieldVectorIndex> {
        let mut resolved = BTreeMap::new();
        for (name, config) in field_configs {
            let mut config = config.clone();
            config.resolve_pq_codebook(storage.as_ref())?;
            resolved.insert(name.clone(), config);
        }
        MultiFieldVectorIndex::open_or_create(storage, &resolved, embedder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;
    use crate::vector::index::config::{
        FlatIndexConfig, HnswIndexConfig, IvfIndexConfig, VectorIndexTypeConfig,
    };

    #[test]
    fn test_vector_index_creation() {
        let config = VectorIndexTypeConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = VectorIndexFactory::create(storage, "test_index", config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_vector_index_open() {
        let config = VectorIndexTypeConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Create index
        let index =
            VectorIndexFactory::create(storage.clone(), "test_index", config.clone()).unwrap();
        index.close().unwrap();

        // Open index
        let index = VectorIndexFactory::open(storage, "test_index", config).unwrap();

        assert!(!index.is_closed());
    }

    /// `open_or_create` opens the single-file index that is there, instead of
    /// creating it again, so another dimension is refused.
    #[test]
    fn open_or_create_opens_an_existing_single_file_index() {
        let configs = |dimension| {
            [
                VectorIndexTypeConfig::Flat(FlatIndexConfig {
                    dimension,
                    segmented: false,
                    ..Default::default()
                }),
                VectorIndexTypeConfig::HNSW(HnswIndexConfig {
                    dimension,
                    segmented: false,
                    ..Default::default()
                }),
                VectorIndexTypeConfig::IVF(IvfIndexConfig {
                    dimension,
                    segmented: false,
                    ..Default::default()
                }),
            ]
        };
        let mut failures = Vec::new();
        let cases = ["Flat", "HNSW", "IVF"].into_iter().zip(configs(4));
        for ((what, first), second) in cases.zip(configs(8)) {
            let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
            let index =
                VectorIndexFactory::open_or_create(storage.clone(), "test_index", first).unwrap();
            index.close().unwrap();
            match VectorIndexFactory::open_or_create(storage, "test_index", second) {
                Err(e)
                    if e.to_string()
                        .contains("Dimension mismatch: stored 4, config 8") => {}
                other => failures.push(format!("{what}: {:?}", other.map(|_| ()))),
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[test]
    fn test_vector_index_stats() {
        // Pinned to the monolithic layout (Issue #907 audit): `stats().last_modified`
        // tracks a real timestamp only for the monolithic `IndexMetadata`; the
        // segmented layout (the default since #907) hardcodes it to 0, since a
        // multi-segment manifest has no single natural "last modified" moment.
        let config = VectorIndexTypeConfig::Flat(FlatIndexConfig {
            segmented: false,
            ..Default::default()
        });
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = VectorIndexFactory::create(storage, "test_index", config).unwrap();
        let stats = index.stats().unwrap();

        assert_eq!(stats.vector_count, 0);
        assert_eq!(stats.dimension, 128); // Default dimension
        assert_eq!(stats.deleted_count, 0);
        assert!(stats.last_modified > 0);
    }

    #[test]
    fn test_vector_index_close() {
        let config = VectorIndexTypeConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = VectorIndexFactory::create(storage, "test_index", config).unwrap();

        assert!(!index.is_closed());

        index.close().unwrap();

        assert!(index.is_closed());

        // Operations should fail after close
        let result = index.stats();
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_index_config() {
        let config = VectorIndexTypeConfig::default();

        // Test that default is Flat and check its configuration
        match config {
            VectorIndexTypeConfig::Flat(flat) => {
                assert_eq!(flat.dimension, 128);
            }
            _ => panic!("Expected Flat config as default"),
        }
    }

    #[test]
    fn test_factory_create_flat() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexTypeConfig::Flat(FlatIndexConfig::default());

        let index = VectorIndexFactory::create(storage, "flat_index", config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_factory_create_hnsw() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexTypeConfig::HNSW(HnswIndexConfig::default());

        let index = VectorIndexFactory::create(storage, "hnsw_index", config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_factory_create_ivf() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = VectorIndexTypeConfig::IVF(IvfIndexConfig::default());

        let index = VectorIndexFactory::create(storage, "ivf_index", config).unwrap();

        assert!(!index.is_closed());
    }

    /// Issue #948: `open_or_create_multi_field` must build one independent
    /// sub-index per field, each honoring its own dimension/distance metric
    /// -- not collapse to a single shared config the way the old
    /// `VectorStore::extract_index_type_config` did.
    #[test]
    fn open_or_create_multi_field_honors_each_fields_own_config() {
        use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
        use crate::vector::index::config::HnswIndexConfig;

        #[derive(Debug)]
        struct NoopEmbedder;
        #[async_trait::async_trait]
        impl Embedder for NoopEmbedder {
            async fn embed(
                &self,
                _input: &EmbedInput<'_>,
            ) -> Result<crate::vector::core::vector::Vector> {
                unreachable!("not used by this test")
            }
            fn supported_input_types(&self) -> Vec<EmbedInputType> {
                vec![EmbedInputType::Text]
            }
            fn name(&self) -> &str {
                "noop"
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let mut fields = BTreeMap::new();
        fields.insert(
            "small_vec".to_string(),
            VectorIndexTypeConfig::HNSW(HnswIndexConfig {
                dimension: 3,
                ..Default::default()
            }),
        );
        fields.insert(
            "big_vec".to_string(),
            VectorIndexTypeConfig::HNSW(HnswIndexConfig {
                dimension: 16,
                ..Default::default()
            }),
        );

        let index = VectorIndexFactory::open_or_create_multi_field(
            storage,
            &fields,
            Arc::new(NoopEmbedder),
        )
        .unwrap();

        let dims = index.field_dimensions();
        assert_eq!(dims.get("small_vec"), Some(&3));
        assert_eq!(dims.get("big_vec"), Some(&16));
    }

    /// Issue #948: a shared PQ codebook trained against the ROOT storage
    /// must be found by `open_or_create_multi_field`'s pre-resolution --
    /// resolving against a field-scoped `PrefixedStorage` instead (where
    /// the file does not exist) would leave the codebook unresolved and
    /// fail the eventual commit.
    #[test]
    fn open_or_create_multi_field_resolves_pq_codebook_against_root_storage() {
        use crate::embedding::embedder::{EmbedInput, EmbedInputType, Embedder};
        use crate::vector::core::quantization::QuantizationMethod;
        use crate::vector::core::vector::Vector;
        use crate::vector::index::config::HnswIndexConfig;
        use crate::vector::index::pq_codebook::train_and_write_pq_codebook;

        #[derive(Debug)]
        struct NoopEmbedder;
        #[async_trait::async_trait]
        impl Embedder for NoopEmbedder {
            async fn embed(&self, _input: &EmbedInput<'_>) -> Result<Vector> {
                unreachable!("not used by this test")
            }
            fn supported_input_types(&self) -> Vec<EmbedInputType> {
                vec![EmbedInputType::Text]
            }
            fn name(&self) -> &str {
                "noop"
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        fn sample_vectors(count: usize, dim: usize) -> Vec<Vector> {
            let mut state: u64 = 42;
            (0..count)
                .map(|_| {
                    let data: Vec<f32> = (0..dim)
                        .map(|_| {
                            state = state
                                .wrapping_mul(6_364_136_223_846_793_005)
                                .wrapping_add(1_442_695_040_888_963_407);
                            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                        })
                        .collect();
                    Vector::new(data)
                })
                .collect()
        }

        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        // Train the codebook against the ROOT storage BEFORE the field is
        // created -- mirrors `laurus-cli`'s `train pq-codebook` command.
        train_and_write_pq_codebook(
            storage.as_ref(),
            "embedding.pqcb",
            8,
            2,
            256,
            false,
            &sample_vectors(300, 8),
        )
        .unwrap();

        let mut fields = BTreeMap::new();
        fields.insert(
            "vec".to_string(),
            VectorIndexTypeConfig::HNSW(HnswIndexConfig {
                dimension: 8,
                quantization_method: QuantizationMethod::ProductQuantization { subvector_count: 2 },
                pq_codebook_path: Some("embedding.pqcb".to_string()),
                ..Default::default()
            }),
        );

        let index = VectorIndexFactory::open_or_create_multi_field(
            storage,
            &fields,
            Arc::new(NoopEmbedder),
        )
        .unwrap();

        let mut writer = index.writer().unwrap();
        writer
            .add_vectors(vec![(1, "vec".to_string(), sample_vectors(1, 8).remove(0))])
            .unwrap();
        // If the codebook had been resolved against the wrong (field-scoped)
        // storage namespace, this commit would fail with "no codebook has
        // been trained there yet" (`hnsw/writer.rs`'s `write()` hard-error).
        writer.commit().unwrap();
    }
}
