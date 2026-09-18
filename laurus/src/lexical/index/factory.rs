//! Factory for creating lexical index instances.

use std::sync::Arc;

use crate::error::Result;
use crate::lexical::index::LexicalIndex;
use crate::lexical::index::inverted::InvertedIndex;
use crate::lexical::store::config::LexicalIndexConfig;
use crate::storage::Storage;

/// Factory for creating lexical index instances.
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
/// use laurus::lexical::store::config::LexicalIndexConfig;
/// use laurus::lexical::index::factory::LexicalIndexFactory;
/// use laurus::storage::{StorageFactory, StorageConfig};
/// use laurus::storage::memory::MemoryStorageConfig;
///
/// # fn main() -> laurus::Result<()> {
/// // Create storage using factory
/// let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;
///
/// // Create index using factory
/// let config = LexicalIndexConfig::default();
/// let index = LexicalIndexFactory::create(storage, config)?;
/// # Ok(())
/// # }
/// ```
pub struct LexicalIndexFactory;

impl LexicalIndexFactory {
    /// Create a new lexical index with the given storage and configuration.
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend (created using `StorageFactory`)
    /// * `config` - Index configuration enum containing type-specific settings
    ///
    /// # Returns
    ///
    /// A boxed trait object implementing `LexicalIndex` trait.
    /// The concrete type is determined by the config variant.
    ///
    /// # Example
    ///
    /// ```
    /// use laurus::lexical::store::config::LexicalIndexConfig;
    /// use laurus::lexical::index::config::InvertedIndexConfig;
    /// use laurus::lexical::index::factory::LexicalIndexFactory;
    /// use laurus::storage::{StorageFactory, StorageConfig};
    /// use laurus::storage::file::FileStorageConfig;
    /// use tempfile::TempDir;
    ///
    /// # fn main() -> laurus::Result<()> {
    /// // Create file storage in a private temporary directory. A fixed
    /// // path would race the other doctests: index control files
    /// // (metadata.json, segments.json) are per-directory.
    /// let dir = TempDir::new().unwrap();
    /// let storage_config = StorageConfig::File(FileStorageConfig::new(dir.path()));
    /// let storage = StorageFactory::create(storage_config)?;
    ///
    /// // Create inverted index
    /// let index_config = LexicalIndexConfig::Inverted(InvertedIndexConfig::default());
    /// let index = LexicalIndexFactory::create(storage, index_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create(
        storage: Arc<dyn Storage>,
        config: LexicalIndexConfig,
    ) -> Result<Box<dyn LexicalIndex>> {
        match config {
            LexicalIndexConfig::Inverted(inverted_config) => {
                let index = InvertedIndex::create(storage, inverted_config)?;
                Ok(Box::new(index))
            } // Future implementations will be added here
        }
    }

    /// Open an existing lexical index with the given storage and configuration.
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
    /// # Example
    ///
    /// ```no_run
    /// use laurus::lexical::store::config::LexicalIndexConfig;
    /// use laurus::lexical::index::config::InvertedIndexConfig;
    /// use laurus::lexical::index::factory::LexicalIndexFactory;
    /// use laurus::storage::file::{FileStorage, FileStorageConfig};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> laurus::Result<()> {
    /// let storage = Arc::new(FileStorage::new("./index", FileStorageConfig::new("./index"))?);
    /// let config = LexicalIndexConfig::Inverted(InvertedIndexConfig::default());
    /// let index = LexicalIndexFactory::open(storage, config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(
        storage: Arc<dyn Storage>,
        config: LexicalIndexConfig,
    ) -> Result<Box<dyn LexicalIndex>> {
        match config {
            LexicalIndexConfig::Inverted(inverted_config) => {
                let index = InvertedIndex::open(storage, inverted_config)?;
                Ok(Box::new(index))
            } // Future implementations will be added here
        }
    }

    /// Open an existing lexical index or create a new one.
    ///
    /// Dispatches to the appropriate index implementation based on the configuration variant.
    /// Currently, only [`LexicalIndexConfig::Inverted`] is supported, which creates or opens
    /// an [`InvertedIndex`].
    ///
    /// # Arguments
    ///
    /// * `storage` - The storage backend used for persisting and loading index data.
    /// * `config` - Configuration that determines the index type and its parameters.
    ///
    /// # Returns
    ///
    /// A boxed [`LexicalIndex`] trait object on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying index implementation fails to open or create
    /// the index (e.g., due to storage I/O errors or corrupt metadata).
    pub fn open_or_create(
        storage: Arc<dyn Storage>,
        config: LexicalIndexConfig,
    ) -> Result<Box<dyn LexicalIndex>> {
        match config {
            LexicalIndexConfig::Inverted(inverted_config) => {
                let index = InvertedIndex::open_or_create(storage, inverted_config)?;
                Ok(Box::new(index))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::lexical::store::config::LexicalIndexConfig;
    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;

    #[test]
    fn test_lexical_index_creation() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = LexicalIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_lexical_index_open() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Create index
        let index = LexicalIndexFactory::create(storage.clone(), config.clone()).unwrap();
        index.close().unwrap();

        // Open index
        let index = LexicalIndexFactory::open(storage, config).unwrap();

        assert!(!index.is_closed());
    }

    #[test]
    fn test_lexical_index_stats() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = LexicalIndexFactory::create(storage, config).unwrap();
        let stats = index.stats().unwrap();

        assert_eq!(stats.doc_count, 0);
        assert_eq!(stats.term_count, 0);
        assert_eq!(stats.segment_count, 0);
        assert_eq!(stats.deleted_count, 0);
        assert!(stats.last_modified > 0);
    }

    #[test]
    fn test_lexical_index_close() {
        let config = LexicalIndexConfig::default();
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let index = LexicalIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());

        index.close().unwrap();

        assert!(index.is_closed());

        // Operations should fail after close
        let result = index.stats();
        assert!(result.is_err());
    }

    #[test]
    fn test_lexical_index_config() {
        let config = LexicalIndexConfig::default();

        // Test that default is Inverted and check its configuration
        match config {
            LexicalIndexConfig::Inverted(inverted) => {
                assert_eq!(inverted.max_docs_per_segment, 1000000);
                assert_eq!(inverted.write_buffer_size, 1024 * 1024);
                assert!(inverted.store_term_vectors);
                assert_eq!(inverted.merge_factor, 10);
                assert_eq!(inverted.max_segments, 100);
            }
        }
    }

    #[test]
    fn test_factory_create() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let config = LexicalIndexConfig::default();

        let index = LexicalIndexFactory::create(storage, config).unwrap();

        assert!(!index.is_closed());
    }
}
