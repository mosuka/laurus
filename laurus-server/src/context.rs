//! Index lifecycle helpers for creating, opening, and inspecting indices on disk.
//!
//! Each index is stored under a *data directory* that contains:
//!
//! * `schema.toml` – the serialized [`Schema`] definition.
//! * `store/`      – the underlying storage directory managed by [`Engine`].

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use laurus::index_dir::{IndexDirClaim, IndexDirError};
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
/// Both are written by [`IndexDirClaim::create_schema_and_store`]: the
/// directories the claim created, then `schema.toml`, then `store/`, each
/// synced before the next, so once this returns they survive a crash, and a
/// crash before that leaves at most `schema.toml`, which start-up opens
/// (Issue #1587).
///
/// The call holds the directory's [`IndexDirClaim`] from its check until the
/// engine is built or what it wrote is rolled back (Issue #1665), so a
/// create or open of the same directory by another process cannot change
/// it meanwhile, and a failed create never removes an index another
/// process created or opened.
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
/// Returns [`IndexDirError::AlreadyExists`] if an index already exists at
/// `data_dir` or another call holds its claim, and an error if the
/// directory, `schema.toml` or `store/` cannot be created or synced, or if
/// the engine cannot be initialised. Any of these failures leaves
/// `data_dir` in the state it was in before this call -- nothing is left
/// behind to block a retry (Issue #1308, #1587).
///
/// A `store/` left without `schema.toml` is checked here only once the
/// claim is held ([`IndexDirError::LeftoverStore`]): the caller,
/// `IndexService::create_index`, refuses it first with
/// [`laurus::index_dir::check_no_leftover_store`] (Issue #1560). Both errors
/// are returned as [`IndexDirError`], so that the caller can report them
/// with their own status codes.
pub async fn create_index(
    data_dir: &Path,
    schema: &Schema,
    wal_policy: WalSyncPolicy,
    commit_policy: CommitPolicy,
) -> anyhow::Result<Engine> {
    let already_exists = || IndexDirError::AlreadyExists {
        path: data_dir.to_path_buf(),
    };
    if data_dir.join(SCHEMA_FILE).exists() {
        return Err(already_exists().into());
    }

    let claim = IndexDirClaim::acquire_for_create(data_dir)?;

    // A failure from here on must not leave schema.toml/store/ behind to
    // block a retry (Issue #1308), so undo exactly what this call added.
    let result: anyhow::Result<Engine> = async {
        if claim.schema_existed() {
            return Err(already_exists().into());
        }
        laurus::index_dir::check_no_leftover_store(data_dir)?;

        // schema.toml, then store/, durably and in that order (Issue #1587).
        let storage = claim
            .create_schema_and_store(schema)
            .with_context(|| format!("Failed to create data directory: {}", data_dir.display()))?;
        let engine = Engine::builder(storage, schema.clone())
            .wal_sync_policy(wal_policy)
            .commit_policy(commit_policy)
            .persist_schema_with(schema_persist_hook(data_dir))
            .build()
            .await?;
        Ok(engine)
    }
    .await;

    match result {
        Ok(engine) => Ok(engine),
        Err(err) => {
            claim.rollback();
            Err(err)
        }
    }
}

/// Returns `true` when `data_dir` holds an index, i.e. `schema.toml` exists.
///
/// Lets a caller distinguish "no index here" from "an index exists but
/// failed to open" without inspecting the error [`open_index`] returns for
/// the latter (Issue #1480).
pub fn index_exists(data_dir: &Path) -> bool {
    data_dir.join(SCHEMA_FILE).exists()
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
/// The call holds the directory's [`IndexDirClaim`] until the engine is
/// built, so that no create can change or roll back the directory
/// meanwhile (Issue #1665).
///
/// # Errors
///
/// Returns an error if no index exists at `data_dir` (i.e. `schema.toml` is
/// missing), if another call holds the directory's claim
/// ([`IndexDirError::InUse`]), if the schema file cannot be read or parsed,
/// or if the engine fails to initialise.
pub async fn open_index(
    data_dir: &Path,
    wal_policy: WalSyncPolicy,
    commit_policy: CommitPolicy,
) -> anyhow::Result<Engine> {
    let no_index = || {
        anyhow::anyhow!(
            "No index found at {}. Create one first via the CreateIndex RPC.",
            data_dir.display()
        )
    };
    let schema_path = data_dir.join(SCHEMA_FILE);
    if !schema_path.exists() {
        return Err(no_index());
    }
    let claim = match IndexDirClaim::acquire_for_open(data_dir) {
        Ok(claim) => claim,
        Err(IndexDirError::NotAnIndexDirectory { .. }) => return Err(no_index()),
        Err(err) => return Err(err.into()),
    };

    // Made durable before the engine writes under it, since it may have been
    // edited by hand while the server was down (Issue #1668).
    let schema = claim.read_schema().context("Failed to read schema file")?;

    let store_path = data_dir.join(STORE_DIR);
    let storage_config = StorageConfig::File(FileStorageConfig::new(&store_path));
    let storage = StorageFactory::open(storage_config)?;
    let engine = Engine::builder(storage, schema)
        .wal_sync_policy(wal_policy)
        .commit_policy(commit_policy)
        .persist_schema_with(schema_persist_hook(data_dir))
        .build()
        .await?;

    drop(claim);
    Ok(engine)
}

/// Persist the current schema back to the data directory.
///
/// Serializes the given schema as TOML and replaces `schema.toml` inside
/// `data_dir` atomically and durably with
/// [`laurus::index_dir::write_schema`] (Issue #1587), so a crash during a
/// schema change leaves either the old schema or the new one, never a
/// partial file the server cannot start with.
///
/// # Arguments
///
/// * `data_dir` - Path to the data directory containing the index.
/// * `schema` - The schema to persist.
///
/// # Errors
///
/// Returns an error if serialization, the write or a sync fails.
pub fn save_schema(data_dir: &Path, schema: &Schema) -> anyhow::Result<()> {
    laurus::index_dir::write_schema(data_dir, schema).context("Failed to write schema file")
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
    use laurus::storage::durable::{SyncEvent, SyncTrace};

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

    /// Issue #1504: `policy = "interval"` without `interval_ms` disables
    /// auto-commit, as the config docs say, on the path the server takes:
    /// the config file, `to_policy`, then `create_index`. The engine used to
    /// start a zero-interval timer that committed back to back.
    #[tokio::test(flavor = "multi_thread")]
    async fn interval_policy_without_interval_ms_never_auto_commits() {
        use laurus::{Document, TextOption};

        let config: crate::config::Config = toml::from_str(
            r#"
            [index.commit]
            policy = "interval"
        "#,
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("idx");
        // Declare the field, so the put does not add it dynamically (which
        // commits the stores).
        let schema = Schema::builder()
            .add_text_field("title", TextOption::default())
            .build();
        let engine = create_index(
            &data_dir,
            &schema,
            config.index.wal.to_policy(),
            config.index.commit.to_policy(),
        )
        .await
        .unwrap();

        engine
            .put_document("d1", Document::builder().add_text("title", "zero").build())
            .await
            .unwrap();
        // A zero-interval timer would commit within microseconds.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let wal = data_dir.join(STORE_DIR).join("engine.wal");
        assert_eq!(
            engine.stats().unwrap().commit_generation,
            0,
            "an unset interval_ms must not auto-commit"
        );
        assert!(
            std::fs::metadata(&wal).unwrap().len() > 0,
            "the put stays in the WAL until a commit"
        );

        engine.commit().await.unwrap();
        assert_eq!(engine.stats().unwrap().commit_generation, 1);
        assert_eq!(
            std::fs::metadata(&wal).unwrap().len(),
            0,
            "an explicit commit still truncates the WAL"
        );
    }

    /// The recorded events outside `store/`, which the storage and the
    /// engine sync on their own.
    fn outside_store(events: Vec<SyncEvent>, data_dir: &Path) -> Vec<SyncEvent> {
        let store = data_dir.join(STORE_DIR);
        events
            .into_iter()
            .filter(|event| !event.path().starts_with(&store))
            .collect()
    }

    /// Issue #1668: opening an index at start-up makes its `schema.toml`,
    /// which may have been edited by hand while the server was down, durable,
    /// and then the data directory, before the engine is built over it.
    #[tokio::test]
    async fn open_index_syncs_schema_toml_then_the_directory() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("idx");
        let engine = create_index(
            &data_dir,
            &Schema::new(),
            WalSyncPolicy::default(),
            CommitPolicy::default(),
        )
        .await
        .unwrap();
        drop(engine);

        let trace = SyncTrace::start();
        let engine = open_index(&data_dir, WalSyncPolicy::default(), CommitPolicy::default())
            .await
            .unwrap();
        drop(engine);

        assert_eq!(
            outside_store(trace.events(), &data_dir),
            vec![
                SyncEvent::File(data_dir.join(SCHEMA_FILE)),
                SyncEvent::Dir(data_dir.clone()),
            ]
        );
    }

    /// Issue #1587: `CreateIndex` syncs each directory it creates into its
    /// parent, then makes `schema.toml` durable, and only then creates
    /// `store/` and syncs the data directory again.
    #[tokio::test]
    async fn create_index_syncs_new_parents_then_schema_then_store() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("new").join("idx");
        let schema_path = data_dir.join(SCHEMA_FILE);
        let tmp_path = data_dir.join(laurus::index_dir::SCHEMA_TMP_FILE);

        let trace = SyncTrace::start();
        create_index(
            &data_dir,
            &Schema::new(),
            WalSyncPolicy::default(),
            CommitPolicy::default(),
        )
        .await
        .unwrap();

        assert_eq!(
            outside_store(trace.events(), &data_dir),
            vec![
                SyncEvent::Dir(root.path().to_path_buf()),
                SyncEvent::Dir(root.path().join("new")),
                SyncEvent::File(tmp_path.clone()),
                SyncEvent::Rename {
                    from: tmp_path,
                    to: schema_path,
                },
                SyncEvent::Dir(data_dir.clone()),
                SyncEvent::Dir(data_dir.clone()),
            ]
        );
    }

    /// Issue #1666: when the engine build fails in a directory `CreateIndex`
    /// made, the rollback removes `store/`, then `schema.toml`, then the
    /// directory, and syncs the directory that held each before the next
    /// removal, so what it removed stays removed after a crash.
    #[tokio::test]
    async fn create_index_build_failure_removes_what_it_wrote_durably() {
        let root = tempfile::tempdir().unwrap();
        let data_dir = root.path().join("idx");
        let bad_schema = Schema::from_toml(MALFORMED_REGEX_SCHEMA_TOML).unwrap();

        let trace = SyncTrace::start();
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
        assert!(!data_dir.exists());
        let rollback_events: Vec<SyncEvent> = trace
            .events()
            .into_iter()
            .skip_while(|event| !matches!(event, SyncEvent::Remove(_)))
            .collect();
        assert_eq!(
            rollback_events,
            vec![
                SyncEvent::Remove(data_dir.join(STORE_DIR)),
                SyncEvent::Dir(data_dir.clone()),
                SyncEvent::Remove(data_dir.join(SCHEMA_FILE)),
                SyncEvent::Dir(data_dir.clone()),
                SyncEvent::Remove(data_dir.clone()),
                SyncEvent::Dir(root.path().to_path_buf()),
            ]
        );
    }

    /// Issue #1587: a schema change replaces `schema.toml` through a synced
    /// temporary file and syncs the data directory, instead of rewriting the
    /// file in place.
    #[tokio::test]
    async fn schema_change_replaces_schema_toml_durably() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("idx");
        let engine = create_index(
            &data_dir,
            &Schema::new(),
            WalSyncPolicy::default(),
            CommitPolicy::default(),
        )
        .await
        .unwrap();
        let tmp_path = data_dir.join(laurus::index_dir::SCHEMA_TMP_FILE);

        let trace = SyncTrace::start();
        engine
            .add_field(
                "title",
                laurus::FieldOption::Text(laurus::TextOption::default()),
            )
            .await
            .unwrap();

        assert_eq!(
            outside_store(trace.events(), &data_dir),
            vec![
                SyncEvent::File(tmp_path.clone()),
                SyncEvent::Rename {
                    from: tmp_path,
                    to: data_dir.join(SCHEMA_FILE),
                },
                SyncEvent::Dir(data_dir.clone()),
            ]
        );
        assert!(read_schema(&data_dir).unwrap().fields.contains_key("title"));
    }
}
