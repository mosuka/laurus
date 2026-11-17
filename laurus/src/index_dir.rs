//! Directory-layout convention shared by the language bindings' `Index`
//! constructors (Python, Node.js, Ruby, PHP), giving them the same
//! `<index_dir>/schema.toml` + `<index_dir>/store/` layout `laurus-cli`
//! and `laurus-server` already use.
//!
//! Those two entry points implement this convention independently
//! (`laurus-cli/src/context.rs`, `laurus-server/src/context.rs`), each
//! with separate `create`/`open` commands. The bindings, by contrast,
//! expose a single constructor that conflates create-or-open into one
//! call, so [`open_or_create`] collapses the CLI's create/open pair (and
//! its schema-recovery rule) into one function with two cases:
//!
//! - `schema.toml` exists: this is an **open**. The caller must not also
//!   pass a schema ([`IndexDirError::SchemaConflict`] if they do); the
//!   persisted schema is loaded and the `store/` directory is opened.
//! - `schema.toml` is absent: this is a **create**. If old segment files
//!   are found directly under `index_dir` (this project's pre-Issue-1059
//!   flat layout), that's refused as [`IndexDirError::LegacyFlatLayout`]
//!   rather than silently starting a fresh, empty index alongside
//!   orphaned data. A `store/` left without `schema.toml` is refused as
//!   [`IndexDirError::LeftoverStore`] rather than opened as the new
//!   index's storage, which would serve its documents (Issue #1560).
//!   Otherwise, the given schema (or [`Schema::default`] when omitted) is
//!   checked with [`Schema::validate_for_create`], and
//!   [`create_schema_and_store`] writes it to `schema.toml` and creates a
//!   new `store/` directory, durably and in that order (Issue #1587).
//!
//! [`open_or_create`] itself is not used by `laurus-cli`/`laurus-server`
//! (see Issue #1061) — it exists solely for the four bindings, which had no
//! shared convention before Issue #1059. [`check_no_leftover_store`],
//! [`create_schema_and_store`] and [`CreateRollback`], however, are shared
//! by all three entry points, since the write-then-build-engine shape (and
//! the checks, ordering and cleanup it needs) is the same everywhere, and
//! `laurus-cli` and `laurus-server` persist every schema change with
//! [`write_schema`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::storage::durable;
use crate::storage::file::FileStorageConfig;
use crate::storage::{Storage, StorageConfig, StorageFactory};
use crate::{LaurusError, Schema};

/// Name of the schema file within an index directory.
pub const SCHEMA_FILE: &str = "schema.toml";

/// Name of the temporary file [`write_schema`] writes and then renames over
/// [`SCHEMA_FILE`] (Issue #1587). One that a crash left behind is truncated
/// and reused by the next write.
pub const SCHEMA_TMP_FILE: &str = "schema.toml.tmp";

/// Name of the storage subdirectory within an index directory.
pub const STORE_DIR: &str = "store";

/// A file that only ever exists directly under an index directory created
/// by this project's pre-Issue-1059 flat layout (segments were written
/// straight into the given path, with no `store/` wrapper). Used to detect
/// that layout and fail loudly instead of silently starting a fresh, empty
/// index next to it.
const LEGACY_LAYOUT_MARKER: &str = "engine.wal";

/// Error from [`open_or_create`].
#[derive(Debug, thiserror::Error)]
pub enum IndexDirError {
    /// `schema.toml` already exists at `path`, but the caller also passed
    /// an explicit schema. Reopening an existing index only needs the
    /// directory path; pass no schema (or `None`) to use the persisted
    /// one.
    #[error(
        "{path} already exists; pass no schema to reopen this index with its persisted schema \
         (a schema argument is only accepted when creating a new index)"
    )]
    SchemaConflict {
        /// Path to the existing `schema.toml`.
        path: PathBuf,
    },

    /// `path` contains segment files from this project's pre-Issue-1059
    /// flat layout (no `schema.toml`, but segment files are present
    /// directly under the directory).
    #[error(
        "{path} contains an index in the pre-Issue-1059 flat layout (no schema.toml, but \
         segment files are present directly under this directory); move its contents into \
         {path}/store/ and write a schema.toml file, or choose a new, empty directory"
    )]
    LegacyFlatLayout {
        /// The index directory containing the legacy layout.
        path: PathBuf,
    },

    /// `path` has a `store/` but no `schema.toml` (Issue #1560).
    ///
    /// Creating an index there would open the leftover `store/` as the new
    /// index's storage and serve its documents. They may be the documents
    /// of an earlier index whose `schema.toml` was removed by hand or by a
    /// rollback that failed to remove `store/`, so the store is neither
    /// reused nor removed: restoring `schema.toml` opens that index again,
    /// and removing `store/` lets the create go ahead.
    #[error(
        "{path} has no {SCHEMA_FILE} but has a {STORE_DIR}/, which may hold the documents of an \
         earlier index; restore {SCHEMA_FILE} to open that index, or remove the {STORE_DIR}/ to \
         create a new one"
    )]
    LeftoverStore {
        /// The index directory holding the leftover `store/`.
        path: PathBuf,
    },

    /// A filesystem operation on `path` failed.
    #[error("{path}: {source}")]
    Io {
        /// The path the failing operation was on.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// `path` has no `schema.toml`, so it isn't a laurus index directory at
    /// all (Issue #1101's [`peek_commit_generation`], unlike
    /// [`open_or_create`], never creates one on the caller's behalf).
    #[error(
        "{path} is not a laurus index directory (no {SCHEMA_FILE} found); \
         check the path, or create the index first"
    )]
    NotAnIndexDirectory {
        /// The path that was checked.
        path: PathBuf,
    },

    /// Schema (de)serialization or storage creation/opening failed.
    #[error(transparent)]
    Core(#[from] LaurusError),
}

/// Snapshot of an index directory's on-disk state, taken before a create
/// attempt writes `schema.toml`/`store/` and builds an [`crate::Engine`],
/// used to undo exactly what that attempt added if the build fails
/// afterwards (Issue #1308).
///
/// Every entry point that writes the `schema.toml` + `store/` layout before
/// building an engine (`laurus-cli`, `laurus-server`, and each binding's
/// `Index` constructor, all built on this module's convention) takes a
/// snapshot right after checking that a complete index doesn't already
/// exist and that no `store/` is left without `schema.toml`
/// ([`check_no_leftover_store`]), and calls [`Self::rollback`] if the
/// build fails -- otherwise a retry is blocked by state the failed attempt
/// left behind. [`create_schema_and_store`] takes its own snapshot too, and
/// rolls back when writing the layout itself fails (Issue #1587).
#[must_use = "take the snapshot before the create attempt and call rollback() if it fails"]
pub struct CreateRollback {
    index_dir: PathBuf,
    dir_existed: bool,
    schema_existed: bool,
    store_existed: bool,
}

impl CreateRollback {
    /// Record `index_dir`'s state. Call this before writing `schema.toml`
    /// or creating `store/`.
    pub fn snapshot(index_dir: &Path) -> Self {
        Self {
            index_dir: index_dir.to_path_buf(),
            dir_existed: index_dir.exists(),
            schema_existed: index_dir.join(SCHEMA_FILE).exists(),
            store_existed: index_dir.join(STORE_DIR).exists(),
        }
    }

    /// Undo whatever the create attempt added, and never remove anything
    /// that predates the snapshot. The removals run in the reverse order of
    /// the create:
    ///
    /// 1. `store/`, if it didn't exist.
    /// 2. `schema.toml`, if it didn't exist.
    /// 3. `index_dir` itself, with anything else in it, if it didn't exist.
    ///
    /// A reopen (`schema.toml` and `store/` both existed) touches nothing,
    /// and the CLI's recovery path (only `schema.toml` existed) keeps
    /// `schema.toml`.
    ///
    /// Each removal is made durable, by syncing the directory that held the
    /// removed entry, before the next one runs (Issue #1666). What the
    /// rollback removed therefore stays removed after a crash, and since
    /// `store/` is gone for good before `schema.toml` is removed, a crash
    /// during the rollback never leaves a `store/` without `schema.toml`,
    /// which a create refuses ([`IndexDirError::LeftoverStore`]). Missing
    /// parents that the create made above `index_dir` are kept, empty.
    ///
    /// A `store/` that predates the snapshot without `schema.toml` is
    /// refused before any snapshot is taken ([`check_no_leftover_store`]),
    /// so keeping it only matters for a caller that skips that check: the
    /// store may hold the documents of an earlier index (Issue #1560).
    ///
    /// Best-effort: a failed removal or sync is logged through the `log`
    /// crate and the rollback goes on, since the caller is already on its
    /// way to returning the original build error.
    pub fn rollback(self) {
        if !self.store_existed {
            Self::remove_best_effort(&self.index_dir.join(STORE_DIR), true);
        }
        if !self.schema_existed {
            Self::remove_best_effort(&self.index_dir.join(SCHEMA_FILE), false);
        }
        if !self.dir_existed {
            Self::remove_best_effort(&self.index_dir, true);
        }
    }

    fn remove_best_effort(path: &Path, is_dir: bool) {
        let result = if is_dir {
            durable::remove_dir_all(path)
        } else {
            durable::remove_file(path)
        };
        if let Err(err) = result
            && err.kind() != std::io::ErrorKind::NotFound
        {
            log::warn!(
                "failed to remove {} durably after a failed create: {err}",
                path.display()
            );
        }
    }
}

/// Refuse to create an index in `index_dir` when it has a `store/` but no
/// `schema.toml` (Issue #1560).
///
/// Creating the storage does not empty an existing `store/`, so a create
/// there would serve the leftover store's documents as the new index's.
/// Every entry point that creates the `schema.toml` + `store/` layout
/// (`laurus-cli`, `laurus-server`, and [`open_or_create`] for the bindings)
/// calls this before it writes anything or takes a [`CreateRollback`]
/// snapshot.
///
/// # Errors
///
/// Returns [`IndexDirError::LeftoverStore`] when `schema.toml` is absent and
/// `store` exists, whether it is a directory or not.
pub fn check_no_leftover_store(index_dir: &Path) -> Result<(), IndexDirError> {
    if !index_dir.join(SCHEMA_FILE).exists() && index_dir.join(STORE_DIR).exists() {
        return Err(IndexDirError::LeftoverStore {
            path: index_dir.to_path_buf(),
        });
    }
    Ok(())
}

/// Write `schema` to `<index_dir>/schema.toml`, replacing the file
/// atomically and durably (Issue #1587).
///
/// The TOML goes to [`SCHEMA_TMP_FILE`] first, which is synced and renamed
/// over `schema.toml`, and then `index_dir` is synced, so a crash leaves
/// either the old schema or the new one, never a partial file that no entry
/// point can parse. On Unix, the new file gets the old one's permissions. A
/// symlinked `schema.toml` is replaced by a regular file. `index_dir` must
/// exist.
///
/// Every create path writes the first `schema.toml` with this (through
/// [`create_schema_and_store`]), and `laurus-cli` and `laurus-server`
/// persist each schema change with it.
///
/// # Errors
///
/// Returns [`IndexDirError::Core`] if the schema cannot be serialized, and
/// [`IndexDirError::Io`] if writing, syncing or renaming the file, or
/// syncing `index_dir`, fails. After a failed sync of `index_dir`, the new
/// `schema.toml` is already in place.
pub fn write_schema(index_dir: &Path, schema: &Schema) -> Result<(), IndexDirError> {
    let toml = schema.to_toml()?;
    let schema_path = index_dir.join(SCHEMA_FILE);
    durable::replace_file(
        &schema_path,
        &index_dir.join(SCHEMA_TMP_FILE),
        toml.as_bytes(),
    )
    .map_err(|source| IndexDirError::Io {
        path: schema_path,
        source,
    })
}

/// Create a new index's `schema.toml` and `store/` in `index_dir`, and
/// return the storage over `store/` (Issue #1587).
///
/// The steps run in this order, each made durable before the next:
///
/// 1. `index_dir` and any missing parent are created, each synced into the
///    directory that holds it.
/// 2. `schema.toml` is written with [`write_schema`].
/// 3. `store/` is created, and `index_dir` synced, by
///    [`FileStorage::new`](crate::storage::file::FileStorage::new).
///
/// Once this returns, `schema.toml` and `store/` survive a crash. A crash
/// before that leaves nothing, empty directories, [`SCHEMA_TMP_FILE`] or
/// `schema.toml` alone, and every entry point recovers from each of them;
/// it never leaves a `store/` without `schema.toml`
/// ([`IndexDirError::LeftoverStore`]) or a partial `schema.toml`.
///
/// The caller checks first that neither `schema.toml` nor `store/` exists
/// (see [`check_no_leftover_store`]) and validates the schema
/// ([`Schema::validate_for_create`]). Every entry point that creates an
/// index (`laurus-cli`, `laurus-server`, and [`open_or_create`] for the
/// bindings) calls this.
///
/// # Errors
///
/// Returns [`IndexDirError::Io`] if a directory cannot be created or
/// synced, or `schema.toml` cannot be written, and [`IndexDirError::Core`]
/// if the schema cannot be serialized or the storage cannot be created. On
/// an error, what this call created is removed again, as by
/// [`CreateRollback::rollback`], so it does not block a retry.
pub fn create_schema_and_store(
    index_dir: &Path,
    schema: &Schema,
) -> Result<Arc<dyn Storage>, IndexDirError> {
    let rollback = CreateRollback::snapshot(index_dir);
    let created = (|| {
        durable::create_dir_all(index_dir).map_err(|source| IndexDirError::Io {
            path: index_dir.to_path_buf(),
            source,
        })?;
        write_schema(index_dir, schema)?;
        let config = StorageConfig::File(FileStorageConfig::new(index_dir.join(STORE_DIR)));
        Ok(StorageFactory::create(config)?)
    })();
    if created.is_err() {
        rollback.rollback();
    }
    created
}

/// Resolve `index_dir` into a `(Schema, Storage)` pair, creating a new
/// index or opening an existing one as appropriate. See the module docs
/// for the exact rules.
pub fn open_or_create(
    index_dir: &Path,
    schema: Option<Schema>,
) -> Result<(Schema, Arc<dyn Storage>), IndexDirError> {
    let schema_path = index_dir.join(SCHEMA_FILE);
    let store_path = index_dir.join(STORE_DIR);

    if !schema_path.exists() {
        if index_dir.join(LEGACY_LAYOUT_MARKER).exists() {
            return Err(IndexDirError::LegacyFlatLayout {
                path: index_dir.to_path_buf(),
            });
        }
        check_no_leftover_store(index_dir)?;
        let schema = schema.unwrap_or_default();
        // Before anything is written, so a rejected schema leaves no
        // schema.toml behind to block the retry.
        schema.validate_for_create()?;
        let storage = create_schema_and_store(index_dir, &schema)?;
        return Ok((schema, storage));
    }

    if schema.is_some() {
        return Err(IndexDirError::SchemaConflict { path: schema_path });
    }
    let content = std::fs::read_to_string(&schema_path).map_err(|e| IndexDirError::Io {
        path: schema_path.clone(),
        source: e,
    })?;
    let resolved_schema = Schema::from_toml(&content)?;

    // A `store/` missing beside `schema.toml` (a create interrupted between
    // the two) is created here, durably, by `FileStorage::new`.
    let config = StorageConfig::File(FileStorageConfig::new(&store_path));
    let storage = if store_path.exists() {
        StorageFactory::open(config)
    } else {
        StorageFactory::create(config)
    }?;

    Ok((resolved_schema, storage))
}

/// Read the persisted commit generation (Issue #1088) for `index_dir`
/// directly from disk, without building an `Engine` -- no storage lock, no
/// WAL recovery, no embedder loading (Issue #1101). Lets a caller cheaply
/// decide whether reopening or reloading the index is worth doing at all.
///
/// # Returns
///
/// * `0` if the index exists but no commit has happened yet (matches the
///   fallback `Engine::builder`'s own generation tracker uses when
///   `commit_generation.json` doesn't exist).
///
/// # Errors
///
/// Returns [`IndexDirError::NotAnIndexDirectory`] if `index_dir` has no
/// `schema.toml` -- unlike [`open_or_create`], this function never creates
/// one. Returns [`IndexDirError::Core`] if the persisted file exists but is
/// corrupt (checksum mismatch or malformed JSON).
pub fn peek_commit_generation(index_dir: &Path) -> Result<u64, IndexDirError> {
    let schema_path = index_dir.join(SCHEMA_FILE);
    if !schema_path.is_file() {
        return Err(IndexDirError::NotAnIndexDirectory {
            path: index_dir.to_path_buf(),
        });
    }

    let store_path = index_dir.join(STORE_DIR);
    if !store_path.is_dir() {
        // schema.toml was written but no Engine has been built over this
        // directory yet, so there's nothing to read -- and constructing a
        // `FileStorage` here would create an empty `store/` as a side
        // effect of merely peeking.
        return Ok(0);
    }

    let storage = StorageFactory::open(StorageConfig::File(FileStorageConfig::new(&store_path)))?;
    let generation =
        crate::storage::manifest::load_checksummed_json::<crate::engine::CommitGenerationFile>(
            storage.as_ref(),
            crate::engine::COMMIT_GENERATION_FILE,
            None,
        )?
        .map(|(value, _format)| value.generation)
        .unwrap_or_default();
    Ok(generation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::durable::{SyncEvent, SyncTrace};

    #[test]
    fn test_create_writes_schema_and_store() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut schema = Schema::new();
        schema.default_fields = vec!["title".to_string()];

        let (resolved, _storage) = open_or_create(dir.path(), Some(schema)).unwrap();
        assert_eq!(resolved.default_fields, vec!["title".to_string()]);
        assert!(dir.path().join(SCHEMA_FILE).is_file());
        assert!(dir.path().join(STORE_DIR).is_dir());
    }

    #[test]
    fn test_create_with_no_schema_uses_default() {
        let dir = tempfile::TempDir::new().unwrap();
        let (resolved, _storage) = open_or_create(dir.path(), None).unwrap();
        assert!(resolved.fields.is_empty());
        assert!(dir.path().join(SCHEMA_FILE).is_file());
    }

    #[test]
    fn test_reopen_without_schema_loads_persisted_schema() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut schema = Schema::new();
        schema.default_fields = vec!["title".to_string()];
        open_or_create(dir.path(), Some(schema)).unwrap();

        let (reopened, _storage) = open_or_create(dir.path(), None).unwrap();
        assert_eq!(reopened.default_fields, vec!["title".to_string()]);
    }

    #[test]
    fn test_reopen_with_schema_is_rejected() {
        let dir = tempfile::TempDir::new().unwrap();
        open_or_create(dir.path(), Some(Schema::new())).unwrap();

        let err = open_or_create(dir.path(), Some(Schema::new())).unwrap_err();
        assert!(matches!(err, IndexDirError::SchemaConflict { .. }));
    }

    const RESERVED_ANALYZER_SCHEMA_TOML: &str = r#"
        [analyzers.standard]
        tokenizer = { type = "whitespace" }

        [fields.body.Text]
        analyzer = "standard"
    "#;

    #[test]
    fn test_create_rejects_reserved_analyzer_name_before_writing() {
        let dir = tempfile::TempDir::new().unwrap();
        let index_dir = dir.path().join("idx");
        let schema = Schema::from_toml(RESERVED_ANALYZER_SCHEMA_TOML).unwrap();

        let err = open_or_create(&index_dir, Some(schema)).unwrap_err();
        assert!(
            err.to_string()
                .contains("Analyzer name 'standard' is reserved for a built-in analyzer"),
            "got: {err}"
        );
        assert!(!index_dir.exists(), "a rejected create must write nothing");
    }

    #[test]
    fn test_reopen_with_persisted_reserved_analyzer_name() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(SCHEMA_FILE), RESERVED_ANALYZER_SCHEMA_TOML).unwrap();

        let (reopened, _storage) = open_or_create(dir.path(), None).unwrap();
        assert!(reopened.analyzers.contains_key("standard"));
    }

    const RESERVED_FIELD_SCHEMA_TOML: &str = r#"
        [fields._secret.Text]
    "#;

    #[test]
    fn test_create_rejects_reserved_field_name_before_writing() {
        let dir = tempfile::TempDir::new().unwrap();
        let index_dir = dir.path().join("idx");
        let schema = Schema::from_toml(RESERVED_FIELD_SCHEMA_TOML).unwrap();

        let err = open_or_create(&index_dir, Some(schema)).unwrap_err();
        assert!(
            err.to_string().contains("Field name '_secret' is reserved"),
            "got: {err}"
        );
        assert!(!index_dir.exists(), "a rejected create must write nothing");
    }

    #[test]
    fn test_reopen_with_persisted_reserved_field_name() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(SCHEMA_FILE), RESERVED_FIELD_SCHEMA_TOML).unwrap();

        let (reopened, _storage) = open_or_create(dir.path(), None).unwrap();
        assert!(reopened.fields.contains_key("_secret"));
    }

    // --- CreateRollback (Issue #1308) ---
    //
    // Each test also checks, through `SyncTrace`, that the rollback removes
    // `store/`, then `schema.toml`, then the directory, and syncs the
    // directory that held each before the next removal (Issue #1666).

    #[test]
    fn create_rollback_removes_the_whole_directory_when_it_did_not_exist_before() {
        let parent = tempfile::TempDir::new().unwrap();
        let index_dir = parent.path().join("idx");

        let rollback = CreateRollback::snapshot(&index_dir);
        open_or_create(&index_dir, Some(Schema::new())).unwrap();
        assert!(index_dir.join(SCHEMA_FILE).is_file());
        assert!(index_dir.join(STORE_DIR).is_dir());

        let trace = SyncTrace::start();
        rollback.rollback();
        assert!(!index_dir.exists());
        assert_eq!(
            trace.events(),
            vec![
                SyncEvent::Remove(index_dir.join(STORE_DIR)),
                SyncEvent::Dir(index_dir.clone()),
                SyncEvent::Remove(index_dir.join(SCHEMA_FILE)),
                SyncEvent::Dir(index_dir.clone()),
                SyncEvent::Remove(index_dir.clone()),
                SyncEvent::Dir(parent.path().to_path_buf()),
            ]
        );
    }

    #[test]
    fn create_rollback_removes_only_schema_and_store_when_the_directory_predates_it() {
        let dir = tempfile::TempDir::new().unwrap();
        // An unrelated file already in the directory before the attempt.
        std::fs::write(dir.path().join("README.txt"), b"keep me").unwrap();

        let rollback = CreateRollback::snapshot(dir.path());
        open_or_create(dir.path(), Some(Schema::new())).unwrap();

        let trace = SyncTrace::start();
        rollback.rollback();
        assert_eq!(
            trace.events(),
            vec![
                SyncEvent::Remove(dir.path().join(STORE_DIR)),
                SyncEvent::Dir(dir.path().to_path_buf()),
                SyncEvent::Remove(dir.path().join(SCHEMA_FILE)),
                SyncEvent::Dir(dir.path().to_path_buf()),
            ]
        );
        assert!(
            dir.path().is_dir(),
            "the pre-existing directory must survive"
        );
        assert!(
            dir.path().join("README.txt").is_file(),
            "unrelated pre-existing files must survive"
        );
        assert!(!dir.path().join(SCHEMA_FILE).exists());
        assert!(!dir.path().join(STORE_DIR).exists());
    }

    #[test]
    fn create_rollback_never_removes_a_store_that_predates_it() {
        let dir = tempfile::TempDir::new().unwrap();
        // A store/ left without schema.toml. Every entry point refuses to
        // create over it before taking a snapshot (Issue #1560), but a
        // caller that skips that check must still not lose it to a
        // rollback: it may hold the documents of an earlier index.
        let stale = dir.path().join(STORE_DIR).join("stale");
        std::fs::create_dir_all(&stale).unwrap();

        let rollback = CreateRollback::snapshot(dir.path());
        // What such a caller's create attempt would write first.
        std::fs::write(
            dir.path().join(SCHEMA_FILE),
            Schema::new().to_toml().unwrap(),
        )
        .unwrap();

        let trace = SyncTrace::start();
        rollback.rollback();
        assert_eq!(
            trace.events(),
            vec![
                SyncEvent::Remove(dir.path().join(SCHEMA_FILE)),
                SyncEvent::Dir(dir.path().to_path_buf()),
            ]
        );
        assert!(
            stale.is_dir(),
            "a store/ that predates the snapshot must survive the rollback"
        );
        assert!(
            !dir.path().join(SCHEMA_FILE).exists(),
            "the schema.toml the attempt wrote must be removed"
        );
    }

    #[test]
    fn create_rollback_on_reopen_never_touches_a_pre_existing_schema() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(SCHEMA_FILE),
            RESERVED_ANALYZER_SCHEMA_TOML.trim(),
        )
        .unwrap();

        let rollback = CreateRollback::snapshot(dir.path());
        open_or_create(dir.path(), None).unwrap();
        assert!(dir.path().join(STORE_DIR).is_dir());

        let trace = SyncTrace::start();
        rollback.rollback();
        assert_eq!(
            trace.events(),
            vec![
                SyncEvent::Remove(dir.path().join(STORE_DIR)),
                SyncEvent::Dir(dir.path().to_path_buf()),
            ]
        );
        assert!(
            dir.path().join(SCHEMA_FILE).is_file(),
            "a schema.toml that predates the attempt must never be removed"
        );
        assert!(
            !dir.path().join(STORE_DIR).exists(),
            "a store/ this attempt created must still be removed"
        );
    }

    #[test]
    fn create_rollback_on_reopen_never_removes_a_pre_existing_store() {
        let dir = tempfile::TempDir::new().unwrap();
        open_or_create(dir.path(), Some(Schema::new())).unwrap();

        // Simulate a later call that only reopens (schema.toml and store/
        // both already exist): a build failure here must not touch either.
        let rollback = CreateRollback::snapshot(dir.path());
        let trace = SyncTrace::start();
        rollback.rollback();

        assert_eq!(trace.events(), Vec::new());
        assert!(dir.path().join(SCHEMA_FILE).is_file());
        assert!(dir.path().join(STORE_DIR).is_dir());
    }

    #[test]
    fn test_legacy_flat_layout_is_rejected() {
        let dir = tempfile::TempDir::new().unwrap();
        // Simulate the pre-Issue-1059 flat layout: segment files directly
        // under the index dir, no schema.toml.
        std::fs::write(dir.path().join("engine.wal"), b"").unwrap();

        let err = open_or_create(dir.path(), None).unwrap_err();
        assert!(matches!(err, IndexDirError::LegacyFlatLayout { .. }));
    }

    // --- A store/ left without schema.toml (Issue #1560) ---

    /// A schema with a `title` text field and, when `dimension` is given, a
    /// Flat vector field `vec` of that dimension.
    fn leftover_store_schema(dimension: Option<usize>) -> Schema {
        let builder = Schema::builder().add_text_field("title", crate::TextOption::default());
        match dimension {
            Some(dimension) => builder.add_field(
                "vec",
                crate::FieldOption::Flat(crate::vector::FlatOption {
                    dimension,
                    distance: crate::DistanceMetric::Cosine,
                    ..crate::vector::FlatOption::default()
                }),
            ),
            None => builder,
        }
        .build()
    }

    /// Create an index with `schema` at `dir`, put and commit the document
    /// `old`, then remove `schema.toml` by hand, which leaves the state a
    /// removed `schema.toml` or a rollback that failed to remove `store/`
    /// leaves. Returns the removed `schema.toml`.
    async fn leave_a_store_without_schema(dir: &Path, schema: Schema) -> String {
        let has_vector = schema.fields.contains_key("vec");
        let (schema, storage) = open_or_create(dir, Some(schema)).unwrap();
        let engine = crate::Engine::builder(storage, schema)
            .build()
            .await
            .unwrap();
        let mut document = crate::Document::builder().add_text("title", "hello");
        if has_vector {
            document = document.add_vector("vec", vec![1.0, 0.0, 0.0, 0.0]);
        }
        engine.put_document("old", document.build()).await.unwrap();
        engine.commit().await.unwrap();
        drop(engine);

        let schema_toml = std::fs::read_to_string(dir.join(SCHEMA_FILE)).unwrap();
        std::fs::remove_file(dir.join(SCHEMA_FILE)).unwrap();
        schema_toml
    }

    /// A create with `requested` over a store written with `written` is
    /// refused, writes nothing, and keeps the store's document: restoring
    /// `schema.toml` opens it again.
    async fn assert_create_over_leftover_store_is_refused(
        case: &str,
        written: Option<usize>,
        requested: Option<usize>,
    ) {
        let dir = tempfile::TempDir::new().unwrap();
        let schema_toml =
            leave_a_store_without_schema(dir.path(), leftover_store_schema(written)).await;

        let err = match open_or_create(dir.path(), Some(leftover_store_schema(requested))) {
            Ok(_) => panic!("{case}: a create over a leftover store/ must be refused"),
            Err(err) => err,
        };
        assert!(
            matches!(err, IndexDirError::LeftoverStore { .. }),
            "{case}: got {err}"
        );
        assert!(
            err.to_string()
                .contains("has no schema.toml but has a store/"),
            "{case}: got {err}"
        );
        assert!(
            !dir.path().join(SCHEMA_FILE).exists(),
            "{case}: a refused create must write nothing"
        );

        std::fs::write(dir.path().join(SCHEMA_FILE), schema_toml).unwrap();
        let (schema, storage) = open_or_create(dir.path(), None).unwrap();
        let engine = crate::Engine::builder(storage, schema)
            .build()
            .await
            .unwrap();
        assert_eq!(
            engine.get_documents("old").await.unwrap().len(),
            1,
            "{case}: the leftover store/ must keep its document"
        );
    }

    #[tokio::test]
    async fn create_over_a_leftover_text_only_store_is_refused() {
        assert_create_over_leftover_store_is_refused("text only", None, None).await;
    }

    #[tokio::test]
    async fn create_over_a_leftover_store_with_the_same_vector_field_is_refused() {
        assert_create_over_leftover_store_is_refused("same vector field", Some(4), Some(4)).await;
    }

    /// Before Issue #1560, this create failed on the vector field check of
    /// Issue #1547, and its rollback removed the store that predated it.
    #[tokio::test]
    async fn create_over_a_leftover_store_with_another_vector_dimension_is_refused() {
        assert_create_over_leftover_store_is_refused("another dimension", Some(4), Some(8)).await;
    }

    #[test]
    fn create_over_a_file_named_store_is_refused() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(STORE_DIR), b"").unwrap();

        let err = match open_or_create(dir.path(), None) {
            Ok(_) => panic!("a create over a file named store must be refused"),
            Err(err) => err,
        };
        assert!(
            matches!(err, IndexDirError::LeftoverStore { .. }),
            "got {err}"
        );
        assert!(!dir.path().join(SCHEMA_FILE).exists());
    }

    #[test]
    fn peek_commit_generation_rejects_a_directory_with_no_schema_toml() {
        let dir = tempfile::TempDir::new().unwrap();
        let err = peek_commit_generation(dir.path()).unwrap_err();
        assert!(matches!(err, IndexDirError::NotAnIndexDirectory { .. }));
    }

    #[test]
    fn peek_commit_generation_is_zero_before_any_engine_is_built() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(SCHEMA_FILE),
            Schema::new().to_toml().unwrap(),
        )
        .unwrap();

        assert_eq!(peek_commit_generation(dir.path()).unwrap(), 0);
    }

    #[test]
    fn peek_commit_generation_is_zero_before_any_commit() {
        let dir = tempfile::TempDir::new().unwrap();
        open_or_create(dir.path(), Some(Schema::new())).unwrap();

        // `store/` now exists (created by `open_or_create`), but no Engine
        // has ever committed, so `commit_generation.json` doesn't exist yet.
        assert_eq!(peek_commit_generation(dir.path()).unwrap(), 0);
    }

    #[tokio::test]
    async fn peek_commit_generation_matches_a_real_engine_after_a_commit() {
        let dir = tempfile::TempDir::new().unwrap();
        let (schema, storage) = open_or_create(dir.path(), Some(Schema::new())).unwrap();
        let engine = crate::Engine::builder(storage, schema)
            .build()
            .await
            .unwrap();
        engine
            .put_document(
                "doc1",
                crate::Document::builder()
                    .add_text("title", "hello")
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();
        let expected = engine.commit_generation();
        drop(engine);

        assert_eq!(peek_commit_generation(dir.path()).unwrap(), expected);
        assert_eq!(expected, 1);
    }

    fn titled_schema() -> Schema {
        let mut schema = Schema::new();
        schema.default_fields = vec!["title".to_string()];
        schema
    }

    /// Issue #1587: `schema.toml` is replaced through a synced temporary
    /// file, and the directory is synced after the rename.
    #[test]
    fn write_schema_syncs_the_temp_file_renames_it_then_syncs_the_directory() {
        let dir = tempfile::TempDir::new().unwrap();
        let schema_path = dir.path().join(SCHEMA_FILE);
        let tmp_path = dir.path().join(SCHEMA_TMP_FILE);
        std::fs::write(&schema_path, "old").unwrap();
        let schema = titled_schema();

        let trace = SyncTrace::start();
        write_schema(dir.path(), &schema).unwrap();

        assert_eq!(
            trace.events(),
            vec![
                SyncEvent::File(tmp_path.clone()),
                SyncEvent::Rename {
                    from: tmp_path.clone(),
                    to: schema_path.clone(),
                },
                SyncEvent::Dir(dir.path().to_path_buf()),
            ]
        );
        assert_eq!(
            std::fs::read_to_string(&schema_path).unwrap(),
            schema.to_toml().unwrap()
        );
        assert!(!tmp_path.exists());
    }

    /// Issue #1587: a temporary file left by a crash, longer than the new
    /// schema, is truncated before it is reused.
    #[test]
    fn write_schema_replaces_a_leftover_temp_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let schema = titled_schema();
        let toml = schema.to_toml().unwrap();
        let junk = "#".repeat(toml.len() + 100);
        std::fs::write(dir.path().join(SCHEMA_TMP_FILE), junk).unwrap();

        write_schema(dir.path(), &schema).unwrap();

        assert_eq!(
            std::fs::read_to_string(dir.path().join(SCHEMA_FILE)).unwrap(),
            toml
        );
    }

    /// Issue #1587: a write that fails removes its temporary file and leaves
    /// what was at `schema.toml` (here a directory, so the rename fails on
    /// every platform).
    #[test]
    fn write_schema_that_fails_removes_its_temp_file() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(SCHEMA_FILE)).unwrap();

        let err = write_schema(dir.path(), &titled_schema()).unwrap_err();

        assert!(matches!(err, IndexDirError::Io { .. }), "got: {err}");
        assert!(!dir.path().join(SCHEMA_TMP_FILE).exists());
        assert!(dir.path().join(SCHEMA_FILE).is_dir());
    }

    /// Issue #1587: the replaced `schema.toml` keeps the permission bits the
    /// old one had, not those a new file gets from the umask.
    #[cfg(unix)]
    #[test]
    fn write_schema_keeps_the_file_mode() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::TempDir::new().unwrap();
        let probe = dir.path().join("probe");
        std::fs::write(&probe, "").unwrap();
        let new_file_mode = std::fs::metadata(&probe).unwrap().permissions().mode() & 0o777;
        let kept_mode = if new_file_mode == 0o600 { 0o640 } else { 0o600 };
        assert_ne!(kept_mode, new_file_mode, "the test needs a distinct mode");

        let schema_path = dir.path().join(SCHEMA_FILE);
        std::fs::write(&schema_path, "old").unwrap();
        std::fs::set_permissions(&schema_path, std::fs::Permissions::from_mode(kept_mode)).unwrap();

        write_schema(dir.path(), &titled_schema()).unwrap();

        let mode = std::fs::metadata(&schema_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, kept_mode);
    }

    /// Issue #1587: a create through `open_or_create` (the bindings' path)
    /// syncs each directory it creates into its parent, then makes
    /// `schema.toml` durable, and only then creates `store/` and syncs the
    /// index directory again, so a crash never leaves `store/` alone.
    #[test]
    fn open_or_create_syncs_new_parents_then_schema_then_store() {
        let root = tempfile::TempDir::new().unwrap();
        let index_dir = root.path().join("new").join("idx");
        let schema_path = index_dir.join(SCHEMA_FILE);
        let tmp_path = index_dir.join(SCHEMA_TMP_FILE);

        let trace = SyncTrace::start();
        open_or_create(&index_dir, Some(titled_schema())).unwrap();

        assert_eq!(
            trace.events(),
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
        assert!(index_dir.join(STORE_DIR).is_dir());
    }

    /// Issue #1587: a create whose sync of the index directory fails (after
    /// `schema.toml` was renamed into place) removes what it created, so
    /// the retry is not blocked.
    #[test]
    fn create_whose_directory_sync_fails_leaves_nothing() {
        let root = tempfile::TempDir::new().unwrap();
        let index_dir = root.path().join("idx");

        let trace = SyncTrace::start();
        trace.fail_dir_sync(&index_dir);
        let err = open_or_create(&index_dir, Some(titled_schema())).unwrap_err();
        drop(trace);

        assert!(matches!(err, IndexDirError::Io { .. }), "got: {err}");
        assert!(!index_dir.exists(), "the failed create must be rolled back");

        open_or_create(&index_dir, Some(titled_schema())).unwrap();
        assert!(index_dir.join(SCHEMA_FILE).is_file());
    }
}
