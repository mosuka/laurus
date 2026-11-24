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
//!   persisted schema is loaded, after it is made durable since it may have
//!   been edited by hand ([`IndexDirClaim::read_schema`], Issue #1668), and
//!   the `store/` directory is opened.
//! - `schema.toml` is absent: this is a **create**. If old segment files
//!   are found directly under `index_dir` (this project's pre-Issue-1059
//!   flat layout), that's refused as [`IndexDirError::LegacyFlatLayout`]
//!   rather than silently starting a fresh, empty index alongside
//!   orphaned data. A `store/` left without `schema.toml` is refused as
//!   [`IndexDirError::LeftoverStore`] rather than opened as the new
//!   index's storage, which would serve its documents (Issue #1560).
//!   Otherwise, the given schema (or [`Schema::default`] when omitted) is
//!   checked with [`Schema::validate_for_create`], and
//!   [`IndexDirClaim::create_schema_and_store`] writes it to `schema.toml`
//!   and creates a new `store/` directory, durably and in that order (Issue
//!   #1587).
//!
//! [`open_or_create`] itself is not used by `laurus-cli`/`laurus-server`
//! (see Issue #1061) — it exists solely for the four bindings, which had no
//! shared convention before Issue #1059. [`check_no_leftover_store`] and
//! [`IndexDirClaim`], however, are shared by all three entry points, since
//! the write-then-build-engine shape (and the checks, ordering, locking and
//! cleanup it needs) is the same everywhere, and `laurus-cli` and
//! `laurus-server` persist every schema change with [`write_schema`].
//!
//! # Concurrent creates and opens (Issue #1665)
//!
//! Every create and every open of an index directory holds an
//! [`IndexDirClaim`], an OS lock on `<index_dir>/index.lock`
//! ([`LOCK_FILE`]), from before it checks the directory until its engine
//! holds the engine lock, or until its rollback has finished. Only the
//! holder changes the layout, and its rollback removes only what it created
//! after taking the lock, so a create that fails never removes an index
//! that another call created or opened. A call that cannot take the lock
//! fails at once: a create with [`IndexDirError::AlreadyExists`], an open
//! with [`IndexDirError::InUse`].

use std::fs::{File, TryLockError};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::storage::durable;
use crate::storage::file::FileStorageConfig;
use crate::storage::platform;
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

/// Name of the lock file within an index directory, whose OS lock an
/// [`IndexDirClaim`] holds (Issue #1665).
///
/// The file is empty and kept once an index exists. The OS releases its
/// lock when the holding process exits, crash or not, so unlike
/// `store/engine.lock` it is never stale and must not be removed by hand.
pub const LOCK_FILE: &str = "index.lock";

/// How many times a claim opens the lock file again after finding that the
/// file it locked no longer has the name (see [`IndexDirClaim`]).
const CLAIM_ATTEMPTS: usize = 3;

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

    /// A create could not take `path`'s lock because another call, in this
    /// process or another, holds it while creating or opening an index there
    /// (Issue #1665). An index exists there, or is being written.
    #[error(
        "an index already exists at {path}, or another call (in this process or another) is \
         creating or opening one there"
    )]
    AlreadyExists {
        /// The index directory.
        path: PathBuf,
    },

    /// An open could not take `path`'s lock because another call, in this
    /// process or another, holds it while creating or opening the index
    /// (Issue #1665).
    #[error(
        "the index at {path} is being created or opened by another call (in this process or \
         another); try again once it has finished"
    )]
    InUse {
        /// The index directory.
        path: PathBuf,
    },

    /// A filesystem operation on `path` failed.
    ///
    /// `Display` includes the I/O error, so it is not also the error's
    /// [`source`](std::error::Error::source): an error printed with its
    /// chain (`{:#}` or `{:?}` of an `anyhow::Error`, or the CLI's
    /// `Caused by:`) would name the OS error twice (Issue #1724).
    #[error("{path}: {error}")]
    Io {
        /// The path the failing operation was on.
        path: PathBuf,
        /// The underlying I/O error. Not named `source`, which thiserror
        /// would take as the error's source.
        error: std::io::Error,
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

/// Exclusive hold on an index directory while a create or an open checks
/// it, writes `schema.toml`/`store/` and builds its [`crate::Engine`]
/// (Issue #1665).
///
/// A claim holds an OS lock ([`File::try_lock`]: `flock` on Unix,
/// `LockFileEx` on Windows) on `<index_dir>/index.lock` ([`LOCK_FILE`]).
/// Every entry point (`laurus-cli`, `laurus-server`, and the bindings
/// through [`open_or_create`]) takes one before it checks the directory,
/// and keeps it until its engine holds the engine lock -- then it drops the
/// claim -- or until [`Self::rollback`] has undone a failed attempt. So:
///
/// - Only the holder changes the layout, and it records which of
///   `schema.toml` and `store/` existed only once it holds the lock. A
///   rollback therefore removes only what this call created, never an index
///   that another call created or opened meanwhile.
/// - A holder that finds `schema.toml` without `store/` knows that a crash
///   left it there, since a create in progress would hold the lock, and can
///   recover it.
///
/// The lock is not waited for: a call that cannot take it fails at once,
/// with [`IndexDirError::AlreadyExists`] for a create and
/// [`IndexDirError::InUse`] for an open. The lock belongs to the file
/// handle, so two claims in one process exclude each other as two
/// processes do, and the OS releases it when the process exits.
///
/// Not covered: a process running a laurus that predates the claim, a
/// network file system whose `flock` does not lock across clients, and an
/// `index.lock` removed by hand while a claim holds it.
#[must_use = "hold the claim until the engine is built, and call rollback() if the attempt fails"]
#[derive(Debug)]
pub struct IndexDirClaim {
    index_dir: PathBuf,
    /// The locked file. `None` once the rollback has taken it.
    lock: Option<File>,
    created_dir: bool,
    created_lock_file: bool,
    schema_existed: bool,
    store_existed: bool,
}

/// Which kind of call takes an [`IndexDirClaim`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaimPurpose {
    Create,
    Open,
}

impl ClaimPurpose {
    /// The error for a call that found the lock held.
    fn contended(self, index_dir: &Path) -> IndexDirError {
        let path = index_dir.to_path_buf();
        match self {
            Self::Create => IndexDirError::AlreadyExists { path },
            Self::Open => IndexDirError::InUse { path },
        }
    }
}

impl IndexDirClaim {
    /// Claim `index_dir` to create an index there: create the directory
    /// and any missing parent durably if needed, take the lock, and then
    /// record whether `schema.toml` and `store/` exist.
    ///
    /// The caller decides from [`Self::schema_existed`] and
    /// [`Self::store_existed`] whether an index is already there.
    ///
    /// # Errors
    ///
    /// Returns [`IndexDirError::AlreadyExists`] if another call holds the
    /// lock, and [`IndexDirError::Io`] if the directory or the lock file
    /// cannot be created or locked.
    pub fn acquire_for_create(index_dir: &Path) -> Result<Self, IndexDirError> {
        Self::acquire(index_dir, ClaimPurpose::Create)
    }

    /// Claim the index at `index_dir` to open it. Creates neither the
    /// directory nor `schema.toml`; it only creates the lock file, if the
    /// index has none yet.
    ///
    /// # Errors
    ///
    /// Returns [`IndexDirError::NotAnIndexDirectory`] if `index_dir` has no
    /// `schema.toml`, before or after the lock is taken (a failed create
    /// may have removed it meanwhile), [`IndexDirError::InUse`] if another
    /// call holds the lock, and [`IndexDirError::Io`] if the lock file
    /// cannot be created or locked.
    pub fn acquire_for_open(index_dir: &Path) -> Result<Self, IndexDirError> {
        Self::acquire(index_dir, ClaimPurpose::Open)
    }

    fn acquire(index_dir: &Path, purpose: ClaimPurpose) -> Result<Self, IndexDirError> {
        let lock_path = index_dir.join(LOCK_FILE);
        let lock_err = |error| IndexDirError::Io {
            path: lock_path.clone(),
            error,
        };
        for _ in 0..CLAIM_ATTEMPTS {
            let created_dir = match purpose {
                ClaimPurpose::Create => {
                    durable::create_dir_all(index_dir).map_err(|error| IndexDirError::Io {
                        path: index_dir.to_path_buf(),
                        error,
                    })?
                }
                ClaimPurpose::Open => {
                    if !index_dir.join(SCHEMA_FILE).exists() {
                        return Err(not_an_index_directory(index_dir));
                    }
                    false
                }
            };
            let (lock, created_lock_file) = match platform::open_lock_file(&lock_path) {
                Ok(opened) => opened,
                // A failed create's rollback removed the directory meanwhile.
                Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
                Err(err) => return Err(lock_err(err)),
            };
            #[cfg(test)]
            test_hooks::run_before_lock();
            match lock.try_lock() {
                Ok(()) => {}
                Err(TryLockError::WouldBlock) => return Err(purpose.contended(index_dir)),
                Err(TryLockError::Error(err)) => return Err(lock_err(err)),
            }
            // The file was unlinked between the open and the lock, so another
            // claim may hold a new file under the name (Unix only; see
            // `platform::is_lock_file_current`). Dropping `lock` releases it.
            if !platform::is_lock_file_current(&lock, &lock_path).map_err(lock_err)? {
                continue;
            }

            let claim = Self {
                index_dir: index_dir.to_path_buf(),
                lock: Some(lock),
                created_dir,
                created_lock_file,
                schema_existed: index_dir.join(SCHEMA_FILE).exists(),
                store_existed: index_dir.join(STORE_DIR).exists(),
            };
            if purpose == ClaimPurpose::Open && !claim.schema_existed {
                // A failed create's rollback removed schema.toml meanwhile.
                claim.rollback();
                return Err(not_an_index_directory(index_dir));
            }
            return Ok(claim);
        }
        Err(purpose.contended(index_dir))
    }

    /// The claimed index directory.
    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    /// Whether `schema.toml` existed when the lock was taken.
    pub fn schema_existed(&self) -> bool {
        self.schema_existed
    }

    /// Whether `store/` existed when the lock was taken.
    pub fn store_existed(&self) -> bool {
        self.store_existed
    }

    /// Read the claimed directory's `schema.toml`, after making it durable
    /// (Issue #1668).
    ///
    /// The file is synced through the handle it is read from, and then the
    /// index directory is synced, so that a `schema.toml` edited by hand while
    /// the index was closed -- also one an editor saved by renaming a new
    /// file over it -- survives a crash once the engine has written to
    /// `store/` under it. Every entry point that opens an index reads its
    /// schema with this before it builds the engine.
    ///
    /// On Windows, a sync needs write access to the file; one that cannot be
    /// opened for writing (a read-only file, for example) is read without
    /// the sync, and a warning is logged. The directory sync persists the
    /// directory's entries on NTFS (Issue #1667).
    ///
    /// # Errors
    ///
    /// Returns [`IndexDirError::Io`] if `schema.toml` cannot be read or
    /// synced, or the directory cannot be synced, and
    /// [`IndexDirError::Core`] with a [`LaurusError::Schema`] naming the
    /// file if it is not a valid schema, such as one with a key the schema
    /// does not declare (Issue #1700).
    pub fn read_schema(&self) -> Result<Schema, IndexDirError> {
        let schema_path = self.index_dir.join(SCHEMA_FILE);
        let content = durable::read_synced(&schema_path).map_err(|error| IndexDirError::Io {
            path: schema_path.clone(),
            error,
        })?;
        durable::sync_dir(&self.index_dir).map_err(|error| IndexDirError::Io {
            path: self.index_dir.clone(),
            error,
        })?;
        parse_schema_file(&schema_path, &content)
    }

    /// Create a new index's `schema.toml` and `store/` in the claimed
    /// directory, and return the storage over `store/` (Issue #1587).
    ///
    /// The directory and any missing parent were created, durably, when
    /// the claim was taken. Then, each made durable before the next:
    ///
    /// 1. `schema.toml` is written with [`write_schema`].
    /// 2. `store/` is created, and the directory synced, by
    ///    [`FileStorage::new`](crate::storage::file::FileStorage::new).
    ///
    /// Once this returns, `schema.toml` and `store/` survive a crash. A
    /// crash before that leaves nothing, empty directories,
    /// [`SCHEMA_TMP_FILE`] or `schema.toml` alone, and every entry point
    /// recovers from each of them; it never leaves a `store/` without
    /// `schema.toml` ([`IndexDirError::LeftoverStore`]) or a partial
    /// `schema.toml`.
    ///
    /// The caller checks first that neither `schema.toml` nor `store/`
    /// exists ([`Self::schema_existed`], [`check_no_leftover_store`]) and
    /// validates the schema ([`Schema::validate_for_create`]).
    ///
    /// # Errors
    ///
    /// Returns [`IndexDirError::Io`] if `schema.toml` cannot be written or a
    /// directory cannot be synced, and [`IndexDirError::Core`] if the schema
    /// cannot be serialized or the storage cannot be created. The caller
    /// then calls [`Self::rollback`], which removes what this wrote.
    pub fn create_schema_and_store(
        &self,
        schema: &Schema,
    ) -> Result<Arc<dyn Storage>, IndexDirError> {
        write_schema(&self.index_dir, schema)?;
        let config = StorageConfig::File(FileStorageConfig::new(self.index_dir.join(STORE_DIR)));
        Ok(StorageFactory::create(config)?)
    }

    /// Undo whatever this call added after it took the lock, and release
    /// the lock. Nothing that existed then is removed. The removals run in
    /// the reverse order of the create:
    ///
    /// 1. `store/`, if it didn't exist.
    /// 2. `schema.toml`, if it didn't exist.
    /// 3. `index.lock` and then the directory itself, if this call created
    ///    them and found neither `schema.toml` nor `store/`. The directory is
    ///    removed only if it is empty, so nothing another call put in it
    ///    meanwhile is removed (Issue #1665).
    ///
    /// A reopen (`schema.toml` and `store/` both existed) touches nothing,
    /// and the recovery of a crashed create (only `schema.toml` existed)
    /// keeps `schema.toml`.
    ///
    /// Each removal is made durable, by syncing the directory that held the
    /// removed entry, before the next one runs (Issue #1666), except for
    /// `index.lock`, which only lets the next claim reuse it if a crash
    /// brings it back. What the rollback removed therefore stays removed
    /// after a crash, and since `store/` is gone for good before
    /// `schema.toml` is removed, a crash during the rollback never leaves a
    /// `store/` without `schema.toml`, which a create refuses
    /// ([`IndexDirError::LeftoverStore`]). Missing parents that the create
    /// made above the index directory are kept, empty.
    ///
    /// Best-effort: a failed removal or sync is logged through the `log`
    /// crate and the rollback goes on, since the caller is already on its
    /// way to returning the original error.
    pub fn rollback(mut self) {
        if !self.store_existed {
            remove_best_effort(&self.index_dir.join(STORE_DIR), true);
        }
        if !self.schema_existed {
            remove_best_effort(&self.index_dir.join(SCHEMA_FILE), false);
        }
        let lock = self.lock.take();
        if self.created_lock_file
            && !self.schema_existed
            && !self.store_existed
            && let Some(lock) = lock
        {
            let lock_path = self.index_dir.join(LOCK_FILE);
            // On Windows, the file stays if another call opened it after this
            // call released the lock; that call then goes on with the directory.
            if let Err(err) = platform::remove_lock_file(lock, &lock_path) {
                log::warn!("kept {} after a failed create: {err}", lock_path.display());
            } else if self.created_dir {
                match durable::remove_dir(&self.index_dir) {
                    Ok(()) => {}
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                    Err(err) if err.kind() == io::ErrorKind::DirectoryNotEmpty => {
                        log::debug!(
                            "kept {} after a failed create: another call wrote into it",
                            self.index_dir.display()
                        );
                    }
                    Err(err) => log::warn!(
                        "failed to remove {} durably after a failed create: {err}",
                        self.index_dir.display()
                    ),
                }
            }
        }
    }
}

impl Drop for IndexDirClaim {
    fn drop(&mut self) {
        if let Some(lock) = self.lock.take() {
            // Explicitly, before the file is closed: Windows may release the
            // lock of a closed handle only some time later.
            let _ = lock.unlock();
        }
    }
}

/// Remove `path` durably on behalf of a rollback, logging a failure other
/// than its absence.
fn remove_best_effort(path: &Path, is_dir: bool) {
    let result = if is_dir {
        durable::remove_dir_all(path)
    } else {
        durable::remove_file(path)
    };
    if let Err(err) = result
        && err.kind() != io::ErrorKind::NotFound
    {
        log::warn!(
            "failed to remove {} durably after a failed create: {err}",
            path.display()
        );
    }
}

fn not_an_index_directory(index_dir: &Path) -> IndexDirError {
    IndexDirError::NotAnIndexDirectory {
        path: index_dir.to_path_buf(),
    }
}

/// A hook a test can run in the middle of [`IndexDirClaim::acquire`], after
/// the lock file is opened and before it is locked, to run another call
/// there in a fixed order (Issue #1665).
#[cfg(test)]
mod test_hooks {
    use std::cell::RefCell;

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static BEFORE_LOCK: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    /// Run `hook` once, in the next claim on this thread, after it opens the
    /// lock file and before it locks it.
    pub(super) fn set_before_lock(hook: impl FnOnce() + 'static) {
        BEFORE_LOCK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    pub(super) fn run_before_lock() {
        // Taken out before it runs, so a claim inside the hook runs none.
        let hook = BEFORE_LOCK.with(|slot| slot.borrow_mut().take());
        if let Some(hook) = hook {
            hook();
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
/// calls this before it writes anything or takes an [`IndexDirClaim`], and
/// checks the state again once it holds the claim.
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
/// [`IndexDirClaim::create_schema_and_store`]), and `laurus-cli` and
/// `laurus-server` persist each schema change with it. Two calls must not
/// write the same directory's `schema.toml` at once, since they share
/// [`SCHEMA_TMP_FILE`]: a create holds the directory's [`IndexDirClaim`],
/// and a schema change holds the engine lock (Issue #1665).
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
    .map_err(|error| IndexDirError::Io {
        path: schema_path,
        error,
    })
}

/// Read the `schema.toml` of the index at `index_dir`, to show or change
/// the schema without opening the index.
///
/// Unlike [`IndexDirClaim::read_schema`], which every open path uses, this
/// takes no claim and does not sync the file. Both parse it the same way: a
/// file that is not a valid schema, such as one with a key the schema does
/// not declare (Issue #1700), fails with an error that names the file.
///
/// # Errors
///
/// Returns [`IndexDirError::Io`] if `schema.toml` cannot be read, and
/// [`IndexDirError::Core`] with a [`LaurusError::Schema`] naming the file
/// if it is not a valid schema.
pub fn read_schema(index_dir: &Path) -> Result<Schema, IndexDirError> {
    let schema_path = index_dir.join(SCHEMA_FILE);
    let content = std::fs::read_to_string(&schema_path).map_err(|error| IndexDirError::Io {
        path: schema_path.clone(),
        error,
    })?;
    parse_schema_file(&schema_path, &content)
}

/// Parse `content`, read from the `schema.toml` at `schema_path`, adding the
/// path to a [`LaurusError::Schema`] so that the error says which file is
/// not a valid schema.
fn parse_schema_file(schema_path: &Path, content: &str) -> Result<Schema, IndexDirError> {
    Schema::from_toml(content).map_err(|err| match err {
        LaurusError::Schema(message) => IndexDirError::Core(LaurusError::Schema(format!(
            "{}: {message}",
            schema_path.display()
        ))),
        other => IndexDirError::Core(other),
    })
}

/// Resolve `index_dir` into a `(Schema, Storage)` pair, creating a new
/// index or opening an existing one as appropriate, and return it with the
/// directory's [`IndexDirClaim`]. See the module docs for the exact rules.
///
/// The caller holds the claim while it builds the engine over the storage,
/// calls [`IndexDirClaim::rollback`] if the build fails, and drops the
/// claim once the engine holds the engine lock (Issue #1665). Every error
/// this returns has already undone what the call wrote.
///
/// # Errors
///
/// Besides the errors the module docs describe, returns
/// [`IndexDirError::AlreadyExists`] when a create, or
/// [`IndexDirError::InUse`] when an open, finds another call holding the
/// claim.
pub fn open_or_create(
    index_dir: &Path,
    schema: Option<Schema>,
) -> Result<(Schema, Arc<dyn Storage>, IndexDirClaim), IndexDirError> {
    if index_dir.join(SCHEMA_FILE).exists() {
        match IndexDirClaim::acquire_for_open(index_dir) {
            Ok(claim) => return open_claimed(claim, schema),
            // A failed create's rollback removed it meanwhile: create one.
            Err(IndexDirError::NotAnIndexDirectory { .. }) => {}
            Err(err) => return Err(err),
        }
    }

    // Before anything is written or locked, so a refused create leaves
    // nothing behind to block the retry.
    check_create_preconditions(index_dir)?;
    let new_schema = schema.clone().unwrap_or_default();
    new_schema.validate_for_create()?;

    let claim = IndexDirClaim::acquire_for_create(index_dir)?;
    if claim.schema_existed() {
        // Another create finished between the check and the lock.
        return open_claimed(claim, schema);
    }
    match check_create_preconditions(index_dir)
        .and_then(|()| claim.create_schema_and_store(&new_schema))
    {
        Ok(storage) => Ok((new_schema, storage, claim)),
        Err(err) => {
            claim.rollback();
            Err(err)
        }
    }
}

/// Refuse a create over the pre-Issue-1059 flat layout or a `store/` left
/// without `schema.toml`.
fn check_create_preconditions(index_dir: &Path) -> Result<(), IndexDirError> {
    if index_dir.join(LEGACY_LAYOUT_MARKER).exists() {
        return Err(IndexDirError::LegacyFlatLayout {
            path: index_dir.to_path_buf(),
        });
    }
    check_no_leftover_store(index_dir)
}

/// The open branch of [`open_or_create`], for a claim that found
/// `schema.toml`.
fn open_claimed(
    claim: IndexDirClaim,
    schema: Option<Schema>,
) -> Result<(Schema, Arc<dyn Storage>, IndexDirClaim), IndexDirError> {
    let schema_path = claim.index_dir().join(SCHEMA_FILE);
    if schema.is_some() {
        return Err(IndexDirError::SchemaConflict { path: schema_path });
    }
    let opened = (|| {
        // Made durable first, since the engine writes to store/ under it
        // (Issue #1668).
        let resolved_schema = claim.read_schema()?;

        // A `store/` missing beside `schema.toml` is what a crash during a
        // create leaves, since a create in progress would hold the claim. It
        // is created here, durably, by `FileStorage::new`.
        let store_path = claim.index_dir().join(STORE_DIR);
        let config = StorageConfig::File(FileStorageConfig::new(&store_path));
        let storage = if claim.store_existed() {
            StorageFactory::open(config)
        } else {
            StorageFactory::create(config)
        }?;
        Ok((resolved_schema, storage))
    })();
    match opened {
        Ok((resolved_schema, storage)) => Ok((resolved_schema, storage, claim)),
        Err(err) => {
            claim.rollback();
            Err(err)
        }
    }
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

        let (resolved, _storage, _claim) = open_or_create(dir.path(), Some(schema)).unwrap();
        assert_eq!(resolved.default_fields, vec!["title".to_string()]);
        assert!(dir.path().join(SCHEMA_FILE).is_file());
        assert!(dir.path().join(STORE_DIR).is_dir());
    }

    #[test]
    fn test_create_with_no_schema_uses_default() {
        let dir = tempfile::TempDir::new().unwrap();
        let (resolved, _storage, _claim) = open_or_create(dir.path(), None).unwrap();
        assert!(resolved.fields.is_empty());
        assert!(dir.path().join(SCHEMA_FILE).is_file());
    }

    #[test]
    fn test_reopen_without_schema_loads_persisted_schema() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut schema = Schema::new();
        schema.default_fields = vec!["title".to_string()];
        let _ = open_or_create(dir.path(), Some(schema)).unwrap();

        let (reopened, _storage, _claim) = open_or_create(dir.path(), None).unwrap();
        assert_eq!(reopened.default_fields, vec!["title".to_string()]);
    }

    #[test]
    fn test_reopen_with_schema_is_rejected() {
        let dir = tempfile::TempDir::new().unwrap();
        let _ = open_or_create(dir.path(), Some(Schema::new())).unwrap();

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

        let (reopened, _storage, _claim) = open_or_create(dir.path(), None).unwrap();
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

        let (reopened, _storage, _claim) = open_or_create(dir.path(), None).unwrap();
        assert!(reopened.fields.contains_key("_secret"));
    }

    /// Issue #1700: a persisted `schema.toml` with a key the schema does
    /// not declare (a hand edit, or a newer laurus) is not opened, unlike a
    /// reserved name above, since opening it would run without what the key
    /// was meant to set. Opening and the display read both fail with a
    /// schema error that names the file and the key.
    #[test]
    fn reopen_with_an_unknown_key_in_schema_toml_fails_naming_the_file_and_the_key() {
        let dir = tempfile::TempDir::new().unwrap();
        let schema_path = dir.path().join(SCHEMA_FILE);
        std::fs::write(&schema_path, "[fields.body.Text]\nindexd = false\n").unwrap();
        let prefix = format!("{}: invalid schema TOML:", schema_path.display());

        let mut failures = Vec::new();
        let open = open_or_create(dir.path(), None).map(|(schema, _, _)| schema);
        for (label, result) in [("open", open), ("display read", read_schema(dir.path()))] {
            match result {
                Err(IndexDirError::Core(LaurusError::Schema(message)))
                    if message.starts_with(&prefix)
                        && message.contains("unknown field `indexd`") => {}
                Err(other) => failures.push(format!("{label}: {other}")),
                Ok(_) => failures.push(format!("{label}: opened")),
            }
        }
        assert!(
            failures.is_empty(),
            "each must fail naming {} and `indexd`: {failures:?}",
            schema_path.display()
        );
    }

    // --- IndexDirClaim::rollback (Issue #1308, #1665) ---
    //
    // Each test also checks, through `SyncTrace`, that the rollback removes
    // `store/`, then `schema.toml`, then the directory, and syncs the
    // directory that held each before the next removal (Issue #1666).

    #[test]
    fn create_rollback_removes_the_whole_directory_when_it_did_not_exist_before() {
        let parent = tempfile::TempDir::new().unwrap();
        let index_dir = parent.path().join("idx");

        let (_, storage, claim) = open_or_create(&index_dir, Some(Schema::new())).unwrap();
        drop(storage);
        assert!(index_dir.join(SCHEMA_FILE).is_file());
        assert!(index_dir.join(STORE_DIR).is_dir());
        assert!(index_dir.join(LOCK_FILE).is_file());

        let trace = SyncTrace::start();
        claim.rollback();
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

        let (_, storage, claim) = open_or_create(dir.path(), Some(Schema::new())).unwrap();
        drop(storage);

        let trace = SyncTrace::start();
        claim.rollback();
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
        assert!(
            !dir.path().join(LOCK_FILE).exists(),
            "the lock file the create added must be removed"
        );
    }

    #[test]
    fn create_rollback_never_removes_a_store_that_predates_it() {
        let dir = tempfile::TempDir::new().unwrap();
        // A store/ left without schema.toml. Every entry point refuses to
        // create over it (Issue #1560), but a caller that skips that check
        // must still not lose it to a rollback: it may hold the documents of
        // an earlier index.
        let stale = dir.path().join(STORE_DIR).join("stale");
        std::fs::create_dir_all(&stale).unwrap();

        let claim = IndexDirClaim::acquire_for_create(dir.path()).unwrap();
        // What such a caller's create attempt would write first.
        std::fs::write(
            dir.path().join(SCHEMA_FILE),
            Schema::new().to_toml().unwrap(),
        )
        .unwrap();

        let trace = SyncTrace::start();
        claim.rollback();
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

        let (_, storage, claim) = open_or_create(dir.path(), None).unwrap();
        drop(storage);
        assert!(dir.path().join(STORE_DIR).is_dir());

        let trace = SyncTrace::start();
        claim.rollback();
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
        let _ = open_or_create(dir.path(), Some(Schema::new())).unwrap();

        // Simulate a later call that only reopens (schema.toml and store/
        // both already exist): a build failure here must not touch either.
        let claim = IndexDirClaim::acquire_for_open(dir.path()).unwrap();
        let trace = SyncTrace::start();
        claim.rollback();

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
        let (schema, storage, _claim) = open_or_create(dir, Some(schema)).unwrap();
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
        let (schema, storage, _claim) = open_or_create(dir.path(), None).unwrap();
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
        let _ = open_or_create(dir.path(), Some(Schema::new())).unwrap();

        // `store/` now exists (created by `open_or_create`), but no Engine
        // has ever committed, so `commit_generation.json` doesn't exist yet.
        assert_eq!(peek_commit_generation(dir.path()).unwrap(), 0);
    }

    #[tokio::test]
    async fn peek_commit_generation_matches_a_real_engine_after_a_commit() {
        let dir = tempfile::TempDir::new().unwrap();
        let (schema, storage, _claim) = open_or_create(dir.path(), Some(Schema::new())).unwrap();
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

    /// Issue #1724: an `IndexDirError::Io` names its path and its OS error
    /// once, alone and printed with its chain, since `Display` includes the
    /// OS error and the error has no source. It used to return the
    /// `io::Error` as its source as well, so `{:#}` and the CLI's
    /// `Caused by:` named the OS error twice.
    #[test]
    fn an_io_error_names_its_os_error_once() {
        let dir = tempfile::TempDir::new().unwrap();
        let err = read_schema(dir.path()).unwrap_err();
        assert!(matches!(err, IndexDirError::Io { .. }), "got: {err}");
        assert!(std::error::Error::source(&err).is_none());

        let alone = err.to_string();
        let path = dir.path().join(SCHEMA_FILE).display().to_string();
        assert!(alone.starts_with(&format!("{path}: ")), "{alone}");
        let chained = anyhow::Error::new(err).context("Failed to read schema file");
        for printed in [alone, format!("{chained:#}"), format!("{chained:?}")] {
            assert_eq!(printed.matches("os error").count(), 1, "{printed}");
        }
    }

    /// Issue #1726: `IndexDirError::Core` prints its `LaurusError` and
    /// returns that error's source, which is none, so it too names what the
    /// `LaurusError` holds once.
    #[test]
    fn a_core_error_names_what_it_holds_once() {
        let core = LaurusError::from(std::io::Error::other("the disk is full"));
        let err = IndexDirError::from(core);
        assert!(std::error::Error::source(&err).is_none());

        let alone = err.to_string();
        assert_eq!(alone, "I/O error: the disk is full");
        let chained = anyhow::Error::new(err).context("Failed to open the index");
        for printed in [format!("{chained:#}"), format!("{chained:?}")] {
            assert_eq!(printed.matches("the disk is full").count(), 1, "{printed}");
        }
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
        let _ = open_or_create(&index_dir, Some(titled_schema())).unwrap();

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

    /// Issue #1668: opening an existing index (the bindings' path) makes its
    /// `schema.toml`, which may have been edited by hand, durable, and then
    /// the directory that holds it, before the engine is built over it.
    #[test]
    fn open_or_create_syncs_schema_toml_then_the_directory_on_open() {
        let dir = tempfile::TempDir::new().unwrap();
        let _ = open_or_create(dir.path(), Some(titled_schema())).unwrap();

        let trace = SyncTrace::start();
        let (schema, _storage, _claim) = open_or_create(dir.path(), None).unwrap();

        assert_eq!(schema.default_fields, titled_schema().default_fields);
        assert_eq!(
            trace.events(),
            vec![
                SyncEvent::File(dir.path().join(SCHEMA_FILE)),
                SyncEvent::Dir(dir.path().to_path_buf()),
            ]
        );
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

        let _ = open_or_create(&index_dir, Some(titled_schema())).unwrap();
        assert!(index_dir.join(SCHEMA_FILE).is_file());
    }

    // --- Concurrent creates and opens (Issue #1665) ---
    //
    // Each test runs two callers in a fixed order on one thread. The lock
    // belongs to the file handle, so two claims in one process exclude each
    // other as two processes do. The surviving index is checked first, so
    // that a regression fails on the lost data itself; the loser's error is
    // checked after that.

    fn schema_with(field: &str) -> Schema {
        Schema::builder()
            .add_text_field(field, crate::TextOption::default())
            .build()
    }

    /// What a binding does with `open_or_create`'s result: build the
    /// engine, then drop the claim, or roll back if the build fails.
    async fn build_like_a_binding(
        resolved: Result<(Schema, Arc<dyn Storage>, IndexDirClaim), IndexDirError>,
    ) -> Result<crate::Engine, String> {
        let (schema, storage, claim) = resolved.map_err(|err| err.to_string())?;
        match crate::Engine::builder(storage, schema).build().await {
            Ok(engine) => {
                drop(claim);
                Ok(engine)
            }
            Err(err) => {
                claim.rollback();
                Err(err.to_string())
            }
        }
    }

    /// Put and commit the document `id` through `engine`.
    async fn commit_document(engine: &crate::Engine, id: &str, field: &str) {
        let document = crate::Document::builder().add_text(field, "hello").build();
        engine.put_document(id, document).await.unwrap();
        engine.commit().await.unwrap();
    }

    /// Put and commit the document `doc` through `engine`, close it, reopen
    /// the index at `index_dir`, and check that its `schema.toml` still has
    /// `field` and that `doc` and the documents `earlier` (committed before
    /// the other call ran) are there.
    async fn assert_index_survives(
        index_dir: &Path,
        engine: crate::Engine,
        field: &str,
        earlier: &[&str],
    ) {
        commit_document(&engine, "doc", field).await;
        drop(engine);

        let (schema, storage, _claim) = open_or_create(index_dir, None).unwrap();
        assert!(
            schema.fields.contains_key(field),
            "schema.toml must still be the surviving index's, got fields {:?}",
            schema.fields.keys().collect::<Vec<_>>()
        );
        let engine = crate::Engine::builder(storage, schema)
            .build()
            .await
            .unwrap();
        for id in earlier.iter().copied().chain(["doc"]) {
            assert_eq!(
                engine.get_documents(id).await.unwrap().len(),
                1,
                "the surviving index must keep its committed document {id:?}"
            );
        }
    }

    /// R1: a create that runs while another create holds the claim fails
    /// and removes nothing; before Issue #1665, its rollback removed the
    /// first index after its build failed on the engine lock.
    #[tokio::test]
    async fn a_create_during_another_create_fails_and_removes_nothing() {
        let parent = tempfile::TempDir::new().unwrap();
        let index_dir = parent.path().join("idx");

        // A has checked the directory and holds the claim.
        let a = IndexDirClaim::acquire_for_create(&index_dir).unwrap();
        // B, a binding's create, runs now and builds after A.
        let b = open_or_create(&index_dir, Some(schema_with("b_field")));
        let storage = a.create_schema_and_store(&schema_with("a_field")).unwrap();
        let engine = crate::Engine::builder(storage, schema_with("a_field"))
            .build()
            .await
            .unwrap();
        drop(a);
        commit_document(&engine, "before", "a_field").await;
        let b = build_like_a_binding(b).await;

        assert_index_survives(&index_dir, engine, "a_field", &["before"]).await;
        let err = b.err().expect("the second create must fail");
        assert!(err.contains("already exists"), "got: {err}");
    }

    /// R2: an open that runs between a create's `schema.toml` and `store/`
    /// fails and removes nothing; before Issue #1665, it created `store/`,
    /// lost the engine lock, and its rollback removed the `store/` the
    /// create's engine was using.
    #[tokio::test]
    async fn an_open_between_a_creates_schema_and_store_fails_and_removes_nothing() {
        let parent = tempfile::TempDir::new().unwrap();
        let index_dir = parent.path().join("idx");
        let schema = schema_with("a_field");

        let a = IndexDirClaim::acquire_for_create(&index_dir).unwrap();
        write_schema(&index_dir, &schema).unwrap();
        // B, a binding's open, runs now and builds after A.
        let b = open_or_create(&index_dir, None);
        let storage = StorageFactory::create(StorageConfig::File(FileStorageConfig::new(
            index_dir.join(STORE_DIR),
        )))
        .unwrap();
        let engine = crate::Engine::builder(storage, schema)
            .build()
            .await
            .unwrap();
        drop(a);
        commit_document(&engine, "before", "a_field").await;
        let b = build_like_a_binding(b).await;

        assert_index_survives(&index_dir, engine, "a_field", &["before"]).await;
        let err = b.err().expect("the open must fail");
        assert!(err.contains("being created or opened"), "got: {err}");
    }

    /// R3: an open that runs after a create wrote the index but before it
    /// built its engine fails; before Issue #1665, it built first, and the
    /// create's rollback removed everything under the open engine.
    #[tokio::test]
    async fn an_open_before_a_creates_build_fails_and_the_create_keeps_its_index() {
        let parent = tempfile::TempDir::new().unwrap();
        let index_dir = parent.path().join("idx");
        let schema = schema_with("a_field");

        let a = IndexDirClaim::acquire_for_create(&index_dir).unwrap();
        let storage = a.create_schema_and_store(&schema).unwrap();
        // B, a binding's open, runs and builds now, before A builds.
        let b = build_like_a_binding(open_or_create(&index_dir, None)).await;
        let a = match crate::Engine::builder(storage, schema).build().await {
            Ok(engine) => {
                drop(a);
                Ok(engine)
            }
            Err(err) => {
                a.rollback();
                Err(err)
            }
        };

        let (survivor, b_err) = match (a, b) {
            (Ok(engine), Err(b_err)) => (engine, Some(b_err)),
            (Err(_), Ok(engine)) => (engine, None),
            (a, b) => panic!(
                "exactly one engine must be built, got create: {:?}, open: {:?}",
                a.err(),
                b.err()
            ),
        };
        assert_index_survives(&index_dir, survivor, "a_field", &[]).await;
        let err = b_err.expect("the open must fail and the create must keep its engine");
        assert!(err.contains("being created or opened"), "got: {err}");
    }

    /// A claim records which of `schema.toml` and `store/` exist only once
    /// it holds the lock: B opened the lock file, then another create ran to
    /// the end, and B must find its index rather than create over it.
    #[test]
    fn a_claim_looks_at_the_directory_only_once_it_holds_the_lock() {
        let parent = tempfile::TempDir::new().unwrap();
        let index_dir = parent.path().join("idx");
        let runtime = std::rc::Rc::new(tokio::runtime::Runtime::new().unwrap());
        let a_engine = std::rc::Rc::new(std::cell::RefCell::new(None));

        let (dir, rt, slot) = (index_dir.clone(), runtime.clone(), a_engine.clone());
        test_hooks::set_before_lock(move || {
            let resolved = open_or_create(&dir, Some(schema_with("a_field")));
            let engine = rt.block_on(build_like_a_binding(resolved)).unwrap();
            *slot.borrow_mut() = Some(engine);
        });
        let b = runtime.block_on(build_like_a_binding(open_or_create(&index_dir, None)));

        let engine = a_engine.borrow_mut().take().expect("A's create ran");
        runtime.block_on(assert_index_survives(&index_dir, engine, "a_field", &[]));
        let err = b.err().expect("B must not build over A's engine");
        assert!(err.contains("already locked"), "got: {err}");
    }

    /// A claim that locks a lock file whose name was removed meanwhile, and
    /// now names another claim's file, must not count as holding the
    /// directory (Unix), and on Windows the file cannot be removed while
    /// the claim has it open; either way it ends as `AlreadyExists`.
    #[test]
    fn a_claim_on_a_lock_file_removed_meanwhile_does_not_hold_the_directory() {
        let parent = tempfile::TempDir::new().unwrap();
        let index_dir = parent.path().join("idx");

        // Y made the directory and rolls back while X is between opening and
        // locking the lock file; then Z claims the directory again.
        let y = IndexDirClaim::acquire_for_create(&index_dir).unwrap();
        let z = std::rc::Rc::new(std::cell::RefCell::new(None));
        let (dir, z_slot) = (index_dir.clone(), z.clone());
        test_hooks::set_before_lock(move || {
            y.rollback();
            *z_slot.borrow_mut() = Some(IndexDirClaim::acquire_for_create(&dir).unwrap());
        });
        let x = IndexDirClaim::acquire_for_create(&index_dir);

        assert!(z.borrow().is_some());
        assert!(
            matches!(x, Err(IndexDirError::AlreadyExists { .. })),
            "X must not hold the directory while Z does, got: {x:?}"
        );
    }

    /// A rollback removes the directory it made only if it is empty, so it
    /// never removes what another call put in it meanwhile.
    #[test]
    fn a_rollback_keeps_what_another_call_put_in_the_directory_it_made() {
        let parent = tempfile::TempDir::new().unwrap();
        let index_dir = parent.path().join("idx");

        let claim = IndexDirClaim::acquire_for_create(&index_dir).unwrap();
        std::fs::write(index_dir.join("other.txt"), b"not the create's").unwrap();
        let storage = claim.create_schema_and_store(&titled_schema()).unwrap();
        drop(storage);
        claim.rollback();

        assert!(index_dir.join("other.txt").is_file());
        assert!(!index_dir.join(SCHEMA_FILE).exists());
        assert!(!index_dir.join(STORE_DIR).exists());
    }

    /// An open of a directory without `schema.toml` creates nothing there.
    #[test]
    fn a_claim_to_open_a_non_index_directory_creates_nothing() {
        let dir = tempfile::TempDir::new().unwrap();

        let err = IndexDirClaim::acquire_for_open(dir.path()).unwrap_err();

        assert!(
            matches!(err, IndexDirError::NotAnIndexDirectory { .. }),
            "got: {err}"
        );
        assert!(!dir.path().join(LOCK_FILE).exists());
    }

    /// A claim held by one call makes a create fail with `AlreadyExists`
    /// and an open with `InUse`, and dropping it lets both through.
    #[test]
    fn a_held_claim_turns_away_creates_and_opens_until_it_is_dropped() {
        let dir = tempfile::TempDir::new().unwrap();
        let (_, storage, _) = open_or_create(dir.path(), Some(titled_schema())).unwrap();
        drop(storage);
        let empty = tempfile::TempDir::new().unwrap();

        let held = IndexDirClaim::acquire_for_open(dir.path()).unwrap();
        let held_empty = IndexDirClaim::acquire_for_create(empty.path()).unwrap();
        let open = open_or_create(dir.path(), None).map(|_| ());
        let create = open_or_create(empty.path(), None).map(|_| ());
        assert!(
            matches!(open, Err(IndexDirError::InUse { .. })),
            "got: {open:?}"
        );
        assert!(
            matches!(create, Err(IndexDirError::AlreadyExists { .. })),
            "got: {create:?}"
        );

        drop((held, held_empty));
        let _ = open_or_create(dir.path(), None).unwrap();
        let _ = open_or_create(empty.path(), None).unwrap();
    }
}
