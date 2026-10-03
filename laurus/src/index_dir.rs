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
//!   orphaned data. Otherwise, the given schema (or [`Schema::default`]
//!   when omitted) is checked with [`Schema::validate_for_create`],
//!   written to `schema.toml`, and a fresh `store/` directory is created.
//!
//! `laurus-cli`/`laurus-server` are not migrated to use this module (see
//! Issue #1061) — it exists solely for the four bindings, which had no
//! shared convention before Issue #1059.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::storage::file::FileStorageConfig;
use crate::storage::{Storage, StorageConfig, StorageFactory};
use crate::{LaurusError, Schema};

/// Name of the schema file within an index directory.
pub const SCHEMA_FILE: &str = "schema.toml";

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

/// Resolve `index_dir` into a `(Schema, Storage)` pair, creating a new
/// index or opening an existing one as appropriate. See the module docs
/// for the exact rules.
pub fn open_or_create(
    index_dir: &Path,
    schema: Option<Schema>,
) -> Result<(Schema, Arc<dyn Storage>), IndexDirError> {
    let schema_path = index_dir.join(SCHEMA_FILE);
    let store_path = index_dir.join(STORE_DIR);

    let resolved_schema = if schema_path.exists() {
        if schema.is_some() {
            return Err(IndexDirError::SchemaConflict { path: schema_path });
        }
        let content = std::fs::read_to_string(&schema_path).map_err(|e| IndexDirError::Io {
            path: schema_path.clone(),
            source: e,
        })?;
        Schema::from_toml(&content)?
    } else {
        if index_dir.join(LEGACY_LAYOUT_MARKER).exists() {
            return Err(IndexDirError::LegacyFlatLayout {
                path: index_dir.to_path_buf(),
            });
        }
        let schema = schema.unwrap_or_default();
        // Before anything is written, so a rejected schema leaves no
        // schema.toml behind to block the retry.
        schema.validate_for_create()?;
        std::fs::create_dir_all(index_dir).map_err(|e| IndexDirError::Io {
            path: index_dir.to_path_buf(),
            source: e,
        })?;
        let toml = schema.to_toml()?;
        std::fs::write(&schema_path, toml).map_err(|e| IndexDirError::Io {
            path: schema_path.clone(),
            source: e,
        })?;
        schema
    };

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

    #[test]
    fn test_legacy_flat_layout_is_rejected() {
        let dir = tempfile::TempDir::new().unwrap();
        // Simulate the pre-Issue-1059 flat layout: segment files directly
        // under the index dir, no schema.toml.
        std::fs::write(dir.path().join("engine.wal"), b"").unwrap();

        let err = open_or_create(dir.path(), None).unwrap_err();
        assert!(matches!(err, IndexDirError::LegacyFlatLayout { .. }));
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
}
