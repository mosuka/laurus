//! Index lifecycle helpers for creating, opening, and inspecting indices on disk.
//!
//! Each index is stored under a *data directory* that contains:
//!
//! * `schema.toml` – the serialized [`Schema`] definition.
//! * `store/`      – the underlying storage directory managed by [`Engine`].

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, bail};
use laurus::index_dir::CreateRollback;
use laurus::storage::file::FileStorageConfig;
use laurus::{
    CommitPolicy, Engine, LaurusError, Schema, StorageConfig, StorageFactory, WalSyncPolicy,
};

/// Filename used to persist the index schema inside the data directory.
const SCHEMA_FILE: &str = "schema.toml";

/// Subdirectory name for the underlying storage inside the data directory.
const STORE_DIR: &str = "store";

/// Build a [`laurus::SchemaPersistHook`] that writes `schema.toml` inside
/// `data_dir` (Issue #1078).
///
/// Attached to every [`Engine`] this module builds, so
/// [`Engine::add_field`]/[`Engine::delete_field`] persist the schema
/// themselves instead of relying on the gRPC handlers to do it.
fn schema_persist_hook(data_dir: &Path) -> laurus::SchemaPersistHook {
    let data_dir = data_dir.to_path_buf();
    Arc::new(move |schema| save_schema(&data_dir, schema).map_err(LaurusError::from))
}

/// Create a new index at the given data directory with the provided schema.
///
/// The function persists the schema as `schema.toml`, initialises file-based
/// storage under the `store/` subdirectory, and returns a ready-to-use [`Engine`].
///
/// # Arguments
///
/// * `data_dir`  - Root directory where the index files will be stored.
/// * `schema`    - The schema definition describing the index fields.
/// * `wal_policy` - WAL durability policy threaded into the engine builder.
/// * `commit_policy` - Auto-commit policy threaded into the engine builder.
///
/// # Returns
///
/// A newly constructed [`Engine`] backed by the created storage.
///
/// # Errors
///
/// Returns an error if an index already exists at `data_dir`, if directory
/// creation fails, or if the engine cannot be initialised. A failure in
/// storage/engine initialisation leaves `data_dir` in the state it was in
/// before this call -- nothing is left behind to block a retry (Issue
/// #1308).
pub async fn create_index(
    data_dir: &Path,
    schema: &Schema,
    wal_policy: WalSyncPolicy,
    commit_policy: CommitPolicy,
) -> anyhow::Result<Engine> {
    let schema_path = data_dir.join(SCHEMA_FILE);
    if schema_path.exists() {
        bail!(
            "Index already exists at {}. Delete the directory first.",
            data_dir.display()
        );
    }

    let rollback = CreateRollback::snapshot(data_dir);

    // Ensure the data directory exists.
    std::fs::create_dir_all(data_dir)
        .with_context(|| format!("Failed to create data directory: {}", data_dir.display()))?;

    // Serialize the schema to TOML and persist it.
    let schema_toml =
        toml::to_string_pretty(schema).context("Failed to serialize schema to TOML")?;
    std::fs::write(&schema_path, &schema_toml).context("Failed to write schema file")?;

    // Initialize storage and create the engine. A failure here must not
    // leave schema.toml/store/ behind to block a retry (Issue #1308), so
    // undo exactly what this call added above.
    let store_path = data_dir.join(STORE_DIR);
    let storage_config = StorageConfig::File(FileStorageConfig::new(&store_path));
    let build_result: anyhow::Result<Engine> = async {
        let storage = StorageFactory::create(storage_config)?;
        let engine = Engine::builder(storage, schema.clone())
            .wal_sync_policy(wal_policy)
            .commit_policy(commit_policy)
            .persist_schema_with(schema_persist_hook(data_dir))
            .build()
            .await?;
        Ok(engine)
    }
    .await;

    match build_result {
        Ok(engine) => Ok(engine),
        Err(err) => {
            rollback.rollback();
            Err(err)
        }
    }
}

/// Open an existing index from the given data directory.
///
/// Reads the persisted `schema.toml`, opens the file-based storage, and
/// constructs an [`Engine`] ready for querying and indexing.
///
/// # Arguments
///
/// * `data_dir`  - Root directory of an existing index.
/// * `wal_policy` - WAL durability policy threaded into the engine builder.
/// * `commit_policy` - Auto-commit policy threaded into the engine builder.
///
/// # Returns
///
/// An [`Engine`] loaded from the existing storage.
///
/// # Errors
///
/// Returns an error if no index exists at `data_dir` (i.e. `schema.toml` is
/// missing), if the schema file cannot be read or parsed, or if the engine
/// fails to initialise.
pub async fn open_index(
    data_dir: &Path,
    wal_policy: WalSyncPolicy,
    commit_policy: CommitPolicy,
) -> anyhow::Result<Engine> {
    let schema_path = data_dir.join(SCHEMA_FILE);
    if !schema_path.exists() {
        bail!(
            "No index found at {}. Create one first via the CreateIndex RPC.",
            data_dir.display()
        );
    }

    let schema_toml =
        std::fs::read_to_string(&schema_path).context("Failed to read schema file")?;
    let schema: Schema = toml::from_str(&schema_toml).context("Failed to parse schema TOML")?;

    let store_path = data_dir.join(STORE_DIR);
    let storage_config = StorageConfig::File(FileStorageConfig::new(&store_path));
    let storage = StorageFactory::open(storage_config)?;
    let engine = Engine::builder(storage, schema)
        .wal_sync_policy(wal_policy)
        .commit_policy(commit_policy)
        .persist_schema_with(schema_persist_hook(data_dir))
        .build()
        .await?;

    Ok(engine)
}

/// Persist the current schema back to the data directory.
///
/// Serializes the given schema as TOML and writes it to `schema.toml`
/// inside `data_dir`, overwriting the existing file.
///
/// # Arguments
///
/// * `data_dir` - Path to the data directory containing the index.
/// * `schema` - The schema to persist.
///
/// # Errors
///
/// Returns an error if serialization or file write fails.
pub fn save_schema(data_dir: &Path, schema: &Schema) -> anyhow::Result<()> {
    let schema_toml =
        toml::to_string_pretty(schema).context("Failed to serialize schema to TOML")?;
    let schema_dest = data_dir.join(SCHEMA_FILE);
    std::fs::write(&schema_dest, &schema_toml).context("Failed to write schema file")?;
    Ok(())
}

/// Read the schema from the data directory without opening the full engine.
///
/// This is a lightweight operation that only deserializes `schema.toml`
/// and does not touch the storage layer.
///
/// # Arguments
///
/// * `data_dir` - Root directory containing the `schema.toml` file.
///
/// # Returns
///
/// The deserialized [`Schema`].
///
/// # Errors
///
/// Returns an error if the schema file cannot be read or parsed.
pub fn read_schema(data_dir: &Path) -> anyhow::Result<Schema> {
    let schema_path = data_dir.join(SCHEMA_FILE);
    let schema_toml =
        std::fs::read_to_string(&schema_path).context("Failed to read schema file")?;
    let schema: Schema = toml::from_str(&schema_toml).context("Failed to parse schema TOML")?;
    Ok(schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #1308: a malformed regex parses fine as TOML (it's just a
    /// string) but fails when the analyzer is resolved during the engine
    /// build -- after `schema.toml` and `store/` already exist.
    const MALFORMED_REGEX_SCHEMA_TOML: &str = r#"
        [analyzers.bad]
        tokenizer = { type = "regex", pattern = "(" }

        [fields.body.Text]
        analyzer = "bad"
    "#;

    #[tokio::test]
    async fn create_index_build_failure_leaves_nothing_and_retry_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("idx");
        let bad_schema = Schema::from_toml(MALFORMED_REGEX_SCHEMA_TOML).unwrap();

        let err = create_index(
            &data_dir,
            &bad_schema,
            WalSyncPolicy::default(),
            CommitPolicy::default(),
        )
        .await
        .map(|_engine| ())
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("Failed to resolve analyzer for field 'body'"),
            "got: {err:#}"
        );
        assert!(
            !data_dir.exists(),
            "a build failure in a directory this call created must leave nothing behind"
        );

        create_index(
            &data_dir,
            &Schema::new(),
            WalSyncPolicy::default(),
            CommitPolicy::default(),
        )
        .await
        .expect("retry with a fixed schema must succeed without deleting anything by hand");
    }

    #[tokio::test]
    async fn create_index_build_failure_in_a_pre_existing_directory_only_removes_schema_and_store()
    {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("idx");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::write(data_dir.join("README.txt"), b"keep me").unwrap();

        let bad_schema = Schema::from_toml(MALFORMED_REGEX_SCHEMA_TOML).unwrap();
        create_index(
            &data_dir,
            &bad_schema,
            WalSyncPolicy::default(),
            CommitPolicy::default(),
        )
        .await
        .map(|_engine| ())
        .unwrap_err();

        assert!(
            data_dir.join("README.txt").is_file(),
            "a build failure must not remove files that predate the call"
        );
        assert!(!data_dir.join(SCHEMA_FILE).exists());
        assert!(!data_dir.join(STORE_DIR).exists());
    }
}
