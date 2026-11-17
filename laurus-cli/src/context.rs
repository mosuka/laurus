//! Index lifecycle helpers for the CLI.
//!
//! Provides convenience functions for creating a new index from a schema TOML
//! file and for opening an existing index from a index directory. These are
//! used by the various CLI subcommands to obtain an [`Engine`] instance.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use laurus::index_dir::CreateRollback;
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
/// - `index_dir` has a `store/` but no `schema.toml` (Issue #1560).
/// - The schema file cannot be read or parsed.
/// - The index directory cannot be created.
/// - The engine or storage initialisation fails.
pub async fn create_index(index_dir: &Path, schema_path: &Path) -> Result<()> {
    // Read and parse the schema file.
    let schema_content =
        std::fs::read_to_string(schema_path).context("Failed to read schema file")?;
    let schema = Schema::from_toml(&schema_content).context("Failed to parse schema TOML")?;

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
/// - `index_dir` has a `store/` but no `schema.toml` (Issue #1560).
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
/// | absent | absent | Write schema, create storage (durably, in that order) |
/// | absent | present | Error — leftover store (see below) |
/// | present | absent | **Use existing schema**, create storage (recovery) |
/// | present | present | Error — index already exists |
///
/// A `store/` without `schema.toml` is refused rather than opened as the
/// new index's storage, which would serve its documents (Issue #1560). It
/// may hold the documents of an earlier index, so it is not removed either:
/// restoring `schema.toml` opens that index again, and removing `store/`
/// lets the create go ahead.
///
/// When `schema.toml` already exists but `store/` does not, the function
/// ignores the `schema` argument and reads the existing file instead so that
/// a plain `create index` (without `--schema`) recovers correctly.
///
/// A new index's `schema.toml` and `store/` are written by
/// [`laurus::index_dir::create_schema_and_store`]: the directories it
/// creates, then `schema.toml`, then `store/`, each synced before the next,
/// so once this returns they survive a crash, and a crash before that leaves
/// at most `schema.toml`, never `store/` alone (Issue #1587).
///
/// If engine/storage initialisation then fails (analyzer resolution,
/// embedder construction, ...), whatever this call wrote above is rolled
/// back: `index_dir` ends up exactly as it was before the call (Issue
/// #1308). Because of this, the recovery row above only ever matters when a
/// process crashes between writing `schema.toml` and finishing storage/engine
/// initialisation -- a build failure no longer leaves that state behind.
///
/// # Arguments
///
/// * `index_dir` - Path to the index directory.
/// * `schema` - The schema to persist and use for initialisation. Ignored
///   when an existing `schema.toml` is found without a `store/` directory.
///
/// # Errors
///
/// Returns an error if the index already fully exists, the directory,
/// `schema.toml` or `store/` cannot be created or synced, or engine/storage
/// initialisation fails. Any of these failures leaves `index_dir` in the
/// state it was in before this call -- nothing is left behind to block a
/// retry (Issue #1308, #1587).
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
    laurus::index_dir::check_no_leftover_store(index_dir)?;

    let rollback = CreateRollback::snapshot(index_dir);

    // If schema.toml exists but store/ is missing, recover using the existing
    // schema rather than the one passed in (which may come from the wizard).
    let (schema, created_storage) = if schema_exists && !store_exists {
        let content =
            std::fs::read_to_string(&schema_path).context("Failed to read existing schema file")?;
        let schema = Schema::from_toml(&content).context("Failed to parse existing schema TOML")?;
        (schema, None)
    } else {
        // Before anything is written, so a rejected schema leaves no
        // schema.toml behind to block the retry.
        schema.validate_for_create()?;
        // Create the index directory, schema.toml and store/, durably and in
        // that order (Issue #1587). A failure there is rolled back by the
        // call itself.
        let storage = laurus::index_dir::create_schema_and_store(index_dir, &schema)
            .context("Failed to create index directory")?;
        (schema, Some(storage))
    };

    // Create the storage (on recovery) and engine to initialize the index
    // structure. A failure here must not leave schema.toml/store/ behind to
    // block a retry (Issue #1308), so undo exactly what this call added
    // above.
    let storage_config = StorageConfig::File(FileStorageConfig::new(&store_path));
    let build_result: Result<()> = async {
        let storage = match created_storage {
            Some(storage) => storage,
            None => StorageFactory::create(storage_config)?,
        };
        Engine::builder(storage, schema)
            .persist_schema_with(schema_persist_hook(index_dir))
            .build()
            .await?;
        Ok(())
    }
    .await;

    if let Err(err) = build_result {
        rollback.rollback();
        return Err(err);
    }

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
    let schema = Schema::from_toml(&schema_toml).context("Failed to parse schema TOML")?;

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
    let schema = Schema::from_toml(&schema_toml).context("Failed to parse schema TOML")?;
    Ok(schema)
}

/// Persist the schema to the index directory.
///
/// Replaces `schema.toml` atomically and durably with
/// [`laurus::index_dir::write_schema`] (Issue #1587), so a crash during a
/// schema change leaves either the old schema or the new one, never a
/// partial file the index cannot be opened with.
///
/// # Arguments
///
/// * `index_dir` - The index directory in which to write `schema.toml`.
/// * `schema` - The schema to persist.
///
/// # Errors
///
/// Returns an error if serialization, the write or a sync fails.
pub fn save_schema(index_dir: &Path, schema: &Schema) -> Result<()> {
    laurus::index_dir::write_schema(index_dir, schema).context("Failed to write schema file")
}

#[cfg(test)]
mod tests {
    use super::*;
    use laurus::storage::durable::{SyncEvent, SyncTrace};

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

    /// Issue #1082: like `dynamic_field_add_survives_reopen`, but for
    /// `update_field` -- it must persist `schema.toml` itself via the same
    /// hook, so a field's changed option survives closing and reopening
    /// the index.
    #[tokio::test]
    async fn dynamic_field_update_survives_reopen() {
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
            engine
                .update_field(
                    "title",
                    laurus::FieldOption::Text(laurus::TextOption::default().analyzer("keyword")),
                    laurus::UpdateFieldOptions {
                        reindex: true,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            // No explicit `save_schema` call here — the point of this test.
        }

        // Reopen as a fresh process would, and confirm the change survived.
        let reopened_schema = read_schema(dir.path()).unwrap();
        match reopened_schema.fields.get("title") {
            Some(laurus::FieldOption::Text(opt)) => assert_eq!(
                opt.analyzer,
                Some(laurus::AnalyzerSpec::Named("keyword".into())),
                "update_field should have persisted schema.toml via the engine's hook"
            ),
            other => panic!("expected FieldOption::Text, got {other:?}"),
        }

        let engine = open_index(dir.path()).await.unwrap();
        match engine.schema().fields.get("title") {
            Some(laurus::FieldOption::Text(opt)) => {
                assert_eq!(
                    opt.analyzer,
                    Some(laurus::AnalyzerSpec::Named("keyword".into()))
                )
            }
            other => panic!("expected FieldOption::Text, got {other:?}"),
        }
    }

    /// Issue #1310: a schema with an `[analyzers.*]` entry named after a
    /// built-in analyzer.
    const RESERVED_ANALYZER_SCHEMA_TOML: &str = r#"
        [analyzers.standard]
        tokenizer = { type = "whitespace" }

        [fields.body.Text]
        analyzer = "standard"
    "#;

    fn assert_reserved_analyzer_error(err: &anyhow::Error) {
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Analyzer name 'standard' is reserved for a built-in analyzer"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn create_index_rejects_reserved_analyzer_name_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let schema_path = dir.path().join("schema_in.toml");
        std::fs::write(&schema_path, RESERVED_ANALYZER_SCHEMA_TOML).unwrap();
        let index_dir = dir.path().join("idx");

        let err = create_index(&index_dir, &schema_path).await.unwrap_err();
        assert_reserved_analyzer_error(&err);
        assert!(!index_dir.exists(), "a rejected create must write nothing");
    }

    #[tokio::test]
    async fn create_index_from_schema_rejects_reserved_analyzer_name() {
        let dir = tempfile::tempdir().unwrap();
        let index_dir = dir.path().join("idx");
        let schema = Schema::from_toml(RESERVED_ANALYZER_SCHEMA_TOML).unwrap();

        let err = create_index_from_schema(&index_dir, schema)
            .await
            .unwrap_err();
        assert_reserved_analyzer_error(&err);
        assert!(!index_dir.exists(), "a rejected create must write nothing");
    }

    /// The recovery branch reuses a persisted schema.toml, which predates
    /// the check and must not be rejected.
    #[tokio::test]
    async fn create_index_recovery_keeps_persisted_reserved_analyzer_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(SCHEMA_FILE), RESERVED_ANALYZER_SCHEMA_TOML).unwrap();

        create_index_from_schema(dir.path(), Schema::new())
            .await
            .unwrap();
        assert!(
            read_schema(dir.path())
                .unwrap()
                .analyzers
                .contains_key("standard")
        );
    }

    #[tokio::test]
    async fn open_index_with_persisted_reserved_analyzer_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(SCHEMA_FILE), RESERVED_ANALYZER_SCHEMA_TOML).unwrap();

        let engine = open_index(dir.path()).await.unwrap();
        assert!(engine.schema().analyzers.contains_key("standard"));
    }

    // --- Issue #1308: a build-time failure must leave nothing behind ---

    /// A malformed regex pattern parses fine as TOML (it's just a string)
    /// but fails when the analyzer is resolved during the engine build --
    /// after `schema.toml` and `store/` already exist, unlike the
    /// reserved-name checks above which run before anything is written.
    const MALFORMED_REGEX_SCHEMA_TOML: &str = r#"
        [analyzers.bad]
        tokenizer = { type = "regex", pattern = "(" }

        [fields.body.Text]
        analyzer = "bad"
    "#;

    const VALID_SCHEMA_TOML: &str = r#"
        [fields.body.Text]
    "#;

    fn assert_build_failure(err: &anyhow::Error) {
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Failed to resolve analyzer for field 'body'"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn create_index_build_failure_in_a_fresh_directory_leaves_nothing_and_retry_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let index_dir = dir.path().join("idx");
        let bad_schema_path = dir.path().join("bad.toml");
        std::fs::write(&bad_schema_path, MALFORMED_REGEX_SCHEMA_TOML).unwrap();

        let err = create_index(&index_dir, &bad_schema_path)
            .await
            .unwrap_err();
        assert_build_failure(&err);
        assert!(
            !index_dir.exists(),
            "a build failure in a directory this call created must leave nothing behind"
        );

        let good_schema_path = dir.path().join("good.toml");
        std::fs::write(&good_schema_path, VALID_SCHEMA_TOML).unwrap();
        create_index(&index_dir, &good_schema_path)
            .await
            .expect("retry with a fixed schema must succeed without deleting anything by hand");
    }

    #[tokio::test]
    async fn create_index_build_failure_in_a_pre_existing_directory_only_removes_schema_and_store()
    {
        let dir = tempfile::tempdir().unwrap();
        let index_dir = dir.path().join("idx");
        std::fs::create_dir_all(&index_dir).unwrap();
        std::fs::write(index_dir.join("README.txt"), b"keep me").unwrap();

        let schema = Schema::from_toml(MALFORMED_REGEX_SCHEMA_TOML).unwrap();
        let err = create_index_from_schema(&index_dir, schema)
            .await
            .unwrap_err();
        assert_build_failure(&err);
        assert!(
            index_dir.join("README.txt").is_file(),
            "a build failure must not remove files that predate the call"
        );
        assert!(!index_dir.join(SCHEMA_FILE).exists());
        assert!(!index_dir.join(STORE_DIR).exists());

        let good_schema = Schema::from_toml(VALID_SCHEMA_TOML).unwrap();
        create_index_from_schema(&index_dir, good_schema)
            .await
            .expect("retry with a fixed schema must succeed without deleting anything by hand");
    }

    #[tokio::test]
    async fn create_index_build_failure_on_recovery_path_keeps_the_persisted_schema_toml() {
        let dir = tempfile::tempdir().unwrap();
        let index_dir = dir.path().join("idx");
        std::fs::create_dir_all(&index_dir).unwrap();
        // schema.toml predates the call (no store/ yet): the recovery path.
        std::fs::write(index_dir.join(SCHEMA_FILE), MALFORMED_REGEX_SCHEMA_TOML).unwrap();

        let err = create_index_from_schema(&index_dir, Schema::new())
            .await
            .unwrap_err();
        assert_build_failure(&err);
        assert!(
            index_dir.join(SCHEMA_FILE).is_file(),
            "a schema.toml that predates the call must never be removed"
        );
        assert!(!index_dir.join(STORE_DIR).exists());

        // Fix schema.toml by hand (as the recovery rule expects) and retry.
        std::fs::write(index_dir.join(SCHEMA_FILE), VALID_SCHEMA_TOML).unwrap();
        create_index_from_schema(&index_dir, Schema::new())
            .await
            .expect("retry after fixing the persisted schema must succeed");
    }

    /// Issue #1329: a schema with a `_`-prefixed field other than `_id`.
    const RESERVED_FIELD_SCHEMA_TOML: &str = r#"
        [fields._secret.Text]
    "#;

    fn assert_reserved_field_error(err: &anyhow::Error) {
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Field name '_secret' is reserved"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn create_index_rejects_reserved_field_name_and_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let schema_path = dir.path().join("schema_in.toml");
        std::fs::write(&schema_path, RESERVED_FIELD_SCHEMA_TOML).unwrap();
        let index_dir = dir.path().join("idx");

        let err = create_index(&index_dir, &schema_path).await.unwrap_err();
        assert_reserved_field_error(&err);
        assert!(!index_dir.exists(), "a rejected create must write nothing");
    }

    #[tokio::test]
    async fn create_index_from_schema_rejects_reserved_field_name() {
        let dir = tempfile::tempdir().unwrap();
        let index_dir = dir.path().join("idx");
        let schema = Schema::from_toml(RESERVED_FIELD_SCHEMA_TOML).unwrap();

        let err = create_index_from_schema(&index_dir, schema)
            .await
            .unwrap_err();
        assert_reserved_field_error(&err);
        assert!(!index_dir.exists(), "a rejected create must write nothing");
    }

    /// The recovery branch reuses a persisted schema.toml, which predates
    /// the check and must not be rejected.
    #[tokio::test]
    async fn create_index_recovery_keeps_persisted_reserved_field_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(SCHEMA_FILE), RESERVED_FIELD_SCHEMA_TOML).unwrap();

        create_index_from_schema(dir.path(), Schema::new())
            .await
            .unwrap();
        assert!(
            read_schema(dir.path())
                .unwrap()
                .fields
                .contains_key("_secret")
        );
    }

    #[tokio::test]
    async fn open_index_with_persisted_reserved_field_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(SCHEMA_FILE), RESERVED_FIELD_SCHEMA_TOML).unwrap();

        let engine = open_index(dir.path()).await.unwrap();
        assert!(engine.schema().fields.contains_key("_secret"));
    }

    // --- Issue #1560: a store/ left without schema.toml ---

    /// A schema with a `title` text field and, when `with_vector`, a Flat
    /// vector field `vec` of dimension 4.
    fn leftover_store_schema(with_vector: bool) -> Schema {
        let builder = Schema::builder().add_field(
            "title",
            laurus::FieldOption::Text(laurus::TextOption::default()),
        );
        if with_vector {
            builder.add_field(
                "vec",
                laurus::FieldOption::Flat(laurus::vector::FlatOption {
                    dimension: 4,
                    distance: laurus::DistanceMetric::Cosine,
                    ..laurus::vector::FlatOption::default()
                }),
            )
        } else {
            builder
        }
        .build()
    }

    /// Create an index at `index_dir`, put and commit the document `old`,
    /// then remove `schema.toml` by hand. Returns the removed file.
    async fn leave_a_store_without_schema(index_dir: &Path, with_vector: bool) -> String {
        create_index_from_schema(index_dir, leftover_store_schema(with_vector))
            .await
            .unwrap();
        {
            let engine = open_index(index_dir).await.unwrap();
            let mut document = laurus::Document::builder().add_text("title", "hello");
            if with_vector {
                document = document.add_vector("vec", vec![1.0, 0.0, 0.0, 0.0]);
            }
            engine.put_document("old", document.build()).await.unwrap();
            engine.commit().await.unwrap();
        }
        let schema_toml = std::fs::read_to_string(index_dir.join(SCHEMA_FILE)).unwrap();
        std::fs::remove_file(index_dir.join(SCHEMA_FILE)).unwrap();
        schema_toml
    }

    /// `create index` over a leftover store/ is refused and writes nothing;
    /// the store keeps its document (restoring `schema.toml` opens it), and
    /// removing the store, as the error says, lets the create go ahead with
    /// an empty index.
    async fn assert_create_over_leftover_store_is_refused(with_vector: bool) {
        let case = if with_vector { "vector" } else { "text only" };
        let dir = tempfile::tempdir().unwrap();
        let index_dir = dir.path().join("idx");
        let schema_toml = leave_a_store_without_schema(&index_dir, with_vector).await;

        let result = create_index_from_schema(&index_dir, leftover_store_schema(with_vector)).await;
        let msg = match result {
            Ok(()) => panic!("{case}: create index over a leftover store/ must be refused"),
            Err(err) => format!("{err:#}"),
        };
        assert!(
            msg.contains("has no schema.toml but has a store/"),
            "{case}: got: {msg}"
        );
        assert!(
            !index_dir.join(SCHEMA_FILE).exists(),
            "{case}: a refused create must write nothing"
        );

        std::fs::write(index_dir.join(SCHEMA_FILE), schema_toml).unwrap();
        {
            let engine = open_index(&index_dir).await.unwrap();
            assert_eq!(
                engine.get_documents("old").await.unwrap().len(),
                1,
                "{case}: the leftover store/ must keep its document"
            );
        }

        std::fs::remove_file(index_dir.join(SCHEMA_FILE)).unwrap();
        std::fs::remove_dir_all(index_dir.join(STORE_DIR)).unwrap();
        create_index_from_schema(&index_dir, leftover_store_schema(with_vector))
            .await
            .unwrap();
        let engine = open_index(&index_dir).await.unwrap();
        assert!(
            engine.get_documents("old").await.unwrap().is_empty(),
            "{case}: a create after removing store/ must start empty"
        );
    }

    #[tokio::test]
    async fn create_index_over_a_leftover_text_only_store_is_refused() {
        assert_create_over_leftover_store_is_refused(false).await;
    }

    #[tokio::test]
    async fn create_index_over_a_leftover_store_with_a_vector_field_is_refused() {
        assert_create_over_leftover_store_is_refused(true).await;
    }

    /// The recorded events outside `store/`, which the storage and the
    /// engine sync on their own.
    fn outside_store(events: Vec<SyncEvent>, index_dir: &Path) -> Vec<SyncEvent> {
        let store = index_dir.join(STORE_DIR);
        events
            .into_iter()
            .filter(|event| !event.path().starts_with(&store))
            .collect()
    }

    /// Issue #1587: `create index` syncs each directory it creates into its
    /// parent, then makes `schema.toml` durable, and only then creates
    /// `store/` and syncs the index directory again.
    #[tokio::test]
    async fn create_index_syncs_new_parents_then_schema_then_store() {
        let root = tempfile::tempdir().unwrap();
        let index_dir = root.path().join("new").join("idx");
        let schema_path = index_dir.join(SCHEMA_FILE);
        let tmp_path = index_dir.join(laurus::index_dir::SCHEMA_TMP_FILE);

        let trace = SyncTrace::start();
        create_index_from_schema(&index_dir, Schema::new())
            .await
            .unwrap();

        assert_eq!(
            outside_store(trace.events(), &index_dir),
            vec![
                SyncEvent::Dir(root.path().to_path_buf()),
                SyncEvent::Dir(root.path().join("new")),
                SyncEvent::File(tmp_path.clone()),
                SyncEvent::Rename {
                    from: tmp_path,
                    to: schema_path,
                },
                SyncEvent::Dir(index_dir.clone()),
                SyncEvent::Dir(index_dir.clone()),
            ]
        );
    }

    /// Issue #1666: when the engine build fails in a directory the create
    /// made, the rollback removes `store/`, then `schema.toml`, then the
    /// directory, and syncs the directory that held each before the next
    /// removal, so what it removed stays removed after a crash.
    #[tokio::test]
    async fn create_index_build_failure_removes_what_it_wrote_durably() {
        let root = tempfile::tempdir().unwrap();
        let index_dir = root.path().join("idx");
        let schema = Schema::from_toml(MALFORMED_REGEX_SCHEMA_TOML).unwrap();

        let trace = SyncTrace::start();
        let err = create_index_from_schema(&index_dir, schema)
            .await
            .unwrap_err();

        assert_build_failure(&err);
        assert!(!index_dir.exists());
        let rollback_events: Vec<SyncEvent> = trace
            .events()
            .into_iter()
            .skip_while(|event| !matches!(event, SyncEvent::Remove(_)))
            .collect();
        assert_eq!(
            rollback_events,
            vec![
                SyncEvent::Remove(index_dir.join(STORE_DIR)),
                SyncEvent::Dir(index_dir.clone()),
                SyncEvent::Remove(index_dir.join(SCHEMA_FILE)),
                SyncEvent::Dir(index_dir.clone()),
                SyncEvent::Remove(index_dir.clone()),
                SyncEvent::Dir(root.path().to_path_buf()),
            ]
        );
    }

    /// Issue #1587: a schema change replaces `schema.toml` through a synced
    /// temporary file and syncs the index directory, instead of rewriting
    /// the file in place.
    #[tokio::test]
    async fn schema_change_replaces_schema_toml_durably() {
        let dir = tempfile::tempdir().unwrap();
        create_index_from_schema(dir.path(), Schema::new())
            .await
            .unwrap();
        let engine = open_index(dir.path()).await.unwrap();
        let tmp_path = dir.path().join(laurus::index_dir::SCHEMA_TMP_FILE);

        let trace = SyncTrace::start();
        engine
            .add_field(
                "title",
                laurus::FieldOption::Text(laurus::TextOption::default()),
            )
            .await
            .unwrap();

        assert_eq!(
            outside_store(trace.events(), dir.path()),
            vec![
                SyncEvent::File(tmp_path.clone()),
                SyncEvent::Rename {
                    from: tmp_path,
                    to: dir.path().join(SCHEMA_FILE),
                },
                SyncEvent::Dir(dir.path().to_path_buf()),
            ]
        );
        assert!(
            read_schema(dir.path())
                .unwrap()
                .fields
                .contains_key("title")
        );
    }
}
