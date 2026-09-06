//! Index lifecycle helpers for the CLI.
//!
//! Provides convenience functions for creating a new index from a schema TOML
//! file and for opening an existing index from a index directory. These are
//! used by the various CLI subcommands to obtain an [`Engine`] instance.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use laurus::storage::file::FileStorageConfig;
use laurus::{CommitPolicy, Engine, LaurusError, Schema, StorageConfig, StorageFactory};

/// File name used to persist the schema inside the index directory.
const SCHEMA_FILE: &str = "schema.toml";

/// Subdirectory name used for the storage backend within the index directory.
const STORE_DIR: &str = "store";

/// Build a [`laurus::SchemaPersistHook`] that writes `schema.toml` inside
/// `index_dir` (Issue #1078).
///
/// Attached to every [`Engine`] this module builds, so
/// [`Engine::add_field`]/[`Engine::delete_field`] persist the schema
/// themselves instead of relying on each call site to do it.
fn schema_persist_hook(index_dir: &Path) -> laurus::SchemaPersistHook {
    let index_dir = index_dir.to_path_buf();
    Arc::new(move |schema| save_schema(&index_dir, schema).map_err(LaurusError::from))
}

/// Create a new index in the given index directory from a schema TOML file.
///
/// Reads the schema from `schema_path`, creates the index directory (if it
/// does not already exist), persists the schema as `schema.toml` inside
/// `index_dir`, and initialises the underlying storage and engine.
///
/// # Arguments
///
/// * `index_dir` - Path to the index directory where the index will be stored.
/// * `schema_path` - Path to the source schema TOML file that defines fields
///   and their options.
///
/// # Returns
///
/// Returns `Ok(())` on success.
///
/// # Errors
///
/// Returns an error if:
/// - A complete index already exists in `index_dir` (both `schema.toml` and
///   `store/` are present).
/// - The schema file cannot be read or parsed.
/// - The index directory cannot be created.
/// - The engine or storage initialisation fails.
pub async fn create_index(index_dir: &Path, schema_path: &Path) -> Result<()> {
    // Read and parse the schema file.
    let schema_content =
        std::fs::read_to_string(schema_path).context("Failed to read schema file")?;
    let schema: Schema = toml::from_str(&schema_content).context("Failed to parse schema TOML")?;

    init_index(index_dir, schema).await
}

/// Create a new index in the given index directory from an in-memory schema.
///
/// Persists the schema as `schema.toml` inside `index_dir` and initialises
/// the underlying storage and engine. This is used when the schema was built
/// interactively rather than loaded from an existing TOML file.
///
/// # Arguments
///
/// * `index_dir` - Path to the index directory where the index will be stored.
/// * `schema` - The schema to use for the new index.
///
/// # Returns
///
/// Returns `Ok(())` on success.
///
/// # Errors
///
/// Returns an error if:
/// - A complete index already exists in `index_dir` (both `schema.toml` and
///   `store/` are present).
/// - The index directory cannot be created.
/// - The engine or storage initialisation fails.
pub async fn create_index_from_schema(index_dir: &Path, schema: Schema) -> Result<()> {
    init_index(index_dir, schema).await
}

/// Shared implementation for index creation.
///
/// Behaviour depends on the current state of the index directory:
///
/// | `schema.toml` | `store/` | Action |
/// |:---:|:---:|:---|
/// | absent | absent | Write schema, create storage |
/// | absent | present | Write schema, create storage (stale store overwritten) |
/// | present | absent | **Use existing schema**, create storage (recovery) |
/// | present | present | Error — index already exists |
///
/// When `schema.toml` already exists but `store/` does not, the function
/// ignores the `schema` argument and reads the existing file instead so that
/// a plain `create index` (without `--schema`) recovers correctly.
///
/// # Arguments
///
/// * `index_dir` - Path to the index directory.
/// * `schema` - The schema to persist and use for initialisation. Ignored
///   when an existing `schema.toml` is found without a `store/` directory.
///
/// # Errors
///
/// Returns an error if the index already fully exists, the directory cannot
/// be created, or engine/storage initialisation fails.
async fn init_index(index_dir: &Path, schema: Schema) -> Result<()> {
    let schema_path = index_dir.join(SCHEMA_FILE);
    let store_path = index_dir.join(STORE_DIR);
    let schema_exists = schema_path.exists();
    let store_exists = store_path.exists();

    if schema_exists && store_exists {
        bail!(
            "Index already exists at {}. Delete the directory first to recreate.",
            index_dir.display()
        );
    }

    // If schema.toml exists but store/ is missing, recover using the existing
    // schema rather than the one passed in (which may come from the wizard).
    let schema = if schema_exists && !store_exists {
        let content =
            std::fs::read_to_string(&schema_path).context("Failed to read existing schema file")?;
        toml::from_str(&content).context("Failed to parse existing schema TOML")?
    } else {
        // Create the index directory and write the schema.
        std::fs::create_dir_all(index_dir).context("Failed to create index directory")?;
        let schema_toml =
            toml::to_string_pretty(&schema).context("Failed to serialize schema to TOML")?;
        std::fs::write(&schema_path, &schema_toml).context("Failed to write schema file")?;
        schema
    };

    // Create the storage and engine to initialize the index structure.
    let storage_config = StorageConfig::File(FileStorageConfig::new(&store_path));
    let storage = StorageFactory::create(storage_config)?;
    let _engine = Engine::builder(storage, schema)
        .persist_schema_with(schema_persist_hook(index_dir))
        .build()
        .await?;

    Ok(())
}

/// Open an existing index from the given index directory.
///
/// Reads the persisted `schema.toml` and opens the file-based storage
/// backend. If `schema.toml` exists but the `store/` directory is missing
/// (partial state from an interrupted creation), the storage is created
/// automatically to recover.
///
/// # Arguments
///
/// * `index_dir` - Path to the index directory that contains an existing index
///   (must have at least a `schema.toml` file).
///
/// # Returns
///
/// Returns the opened [`Engine`] on success.
///
/// # Errors
///
/// Returns an error if:
/// - No `schema.toml` file is found in `index_dir`.
/// - The schema file cannot be read or parsed.
/// - The storage backend cannot be opened (or created) or the engine cannot
///   be initialised.
pub async fn open_index(index_dir: &Path) -> Result<Engine> {
    open_index_with_commit_policy(index_dir, CommitPolicy::Manual).await
}

/// Open an existing index with an explicit auto-commit policy (Issue #890).
///
/// Identical to [`open_index`] but builds the engine with the given
/// [`CommitPolicy`], so a bulk ingest can hand commit cadence to the engine
/// (e.g. `EveryDocs(n)`) instead of committing client-side.
///
/// # Arguments
///
/// * `index_dir` - Path to the index directory that contains an existing index.
/// * `commit_policy` - The engine auto-commit policy.
///
/// # Returns
///
/// Returns the opened [`Engine`] on success.
///
/// # Errors
///
/// Same as [`open_index`].
pub async fn open_index_with_commit_policy(
    index_dir: &Path,
    commit_policy: CommitPolicy,
) -> Result<Engine> {
    let schema_path = index_dir.join(SCHEMA_FILE);
    if !schema_path.exists() {
        bail!(
            "No index found at {}. Run 'create index' first.",
            index_dir.display()
        );
    }

    // Read the schema.
    let schema_toml =
        std::fs::read_to_string(&schema_path).context("Failed to read schema file")?;
    let schema: Schema = toml::from_str(&schema_toml).context("Failed to parse schema TOML")?;

    // Open or create storage depending on whether the store directory exists.
    let store_path = index_dir.join(STORE_DIR);
    let storage_config = StorageConfig::File(FileStorageConfig::new(&store_path));
    let storage = if store_path.exists() {
        StorageFactory::open(storage_config)?
    } else {
        StorageFactory::create(storage_config)?
    };
    let engine = Engine::builder(storage, schema)
        .commit_policy(commit_policy)
        .persist_schema_with(schema_persist_hook(index_dir))
        .build()
        .await?;

    Ok(engine)
}

/// Read the schema from the index directory.
///
/// # Arguments
///
/// * `index_dir` - The index directory containing `schema.toml`.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn read_schema(index_dir: &Path) -> Result<Schema> {
    let schema_path = index_dir.join(SCHEMA_FILE);
    let schema_toml =
        std::fs::read_to_string(&schema_path).context("Failed to read schema file")?;
    let schema: Schema = toml::from_str(&schema_toml).context("Failed to parse schema TOML")?;
    Ok(schema)
}

/// Persist the schema to the index directory.
///
/// # Arguments
///
/// * `index_dir` - The index directory in which to write `schema.toml`.
/// * `schema` - The schema to persist.
///
/// # Errors
///
/// Returns an error if serialization or file write fails.
pub fn save_schema(index_dir: &Path, schema: &Schema) -> Result<()> {
    let schema_toml =
        toml::to_string_pretty(schema).context("Failed to serialize schema to TOML")?;
    let schema_dest = index_dir.join(SCHEMA_FILE);
    std::fs::write(&schema_dest, &schema_toml).context("Failed to write schema file")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #1078: `add_field`/`delete_field` must persist `schema.toml`
    /// themselves via the hook this module attaches, so a dynamic schema
    /// change survives closing and reopening the index (simulating a
    /// process restart) without any call site needing to call
    /// `save_schema` explicitly.
    #[tokio::test]
    async fn dynamic_field_add_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        create_index_from_schema(dir.path(), Schema::new())
            .await
            .unwrap();

        {
            let engine = open_index(dir.path()).await.unwrap();
            engine
                .add_field(
                    "title",
                    laurus::FieldOption::Text(laurus::TextOption::default()),
                )
                .await
                .unwrap();
            // No explicit `save_schema` call here — the point of this test.
        }

        // Reopen as a fresh process would, and confirm the change survived.
        let reopened_schema = read_schema(dir.path()).unwrap();
        assert!(
            reopened_schema.fields.contains_key("title"),
            "add_field should have persisted schema.toml via the engine's hook"
        );

        let engine = open_index(dir.path()).await.unwrap();
        assert!(engine.schema().fields.contains_key("title"));
    }

    #[tokio::test]
    async fn dynamic_field_delete_survives_reopen() {
        let schema = Schema::builder()
            .add_field(
                "title",
                laurus::FieldOption::Text(laurus::TextOption::default()),
            )
            .build();
        let dir = tempfile::tempdir().unwrap();
        create_index_from_schema(dir.path(), schema).await.unwrap();

        {
            let engine = open_index(dir.path()).await.unwrap();
            engine.delete_field("title").await.unwrap();
        }

        let reopened_schema = read_schema(dir.path()).unwrap();
        assert!(
            !reopened_schema.fields.contains_key("title"),
            "delete_field should have persisted schema.toml via the engine's hook"
        );
    }
}
