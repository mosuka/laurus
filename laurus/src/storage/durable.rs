//! Durable file-system operations on plain paths (Issue #1587).
//!
//! A [`Storage`](crate::storage::Storage) makes the files inside its root
//! durable with [`Storage::sync`](crate::storage::Storage::sync). This
//! module covers what lies outside a root: creating the root directory
//! itself (and any missing parent), and replacing a file such as an index
//! directory's `schema.toml` atomically.
//!
//! A name created, renamed or removed in a directory is durable only once
//! that directory is synced, so every operation here syncs the directory
//! that holds what it created. On Windows, directories cannot be synced
//! through `std` (see [`super::platform`]), and only the file contents and
//! the atomicity of a rename hold.
//!
//! # Test hooks
//!
//! [`SyncTrace`] records the syncs and renames that the functions here
//! request on the current thread, so tests (including those of
//! `laurus-cli` and `laurus-server`) can check what was made durable and in
//! which order, since a crash cannot be simulated in a test. Those items are
//! `#[doc(hidden)]` and not part of the supported API.

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use super::platform;

/// A sync or rename requested through this module, as recorded by
/// [`SyncTrace`].
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncEvent {
    /// A file's contents were synced.
    File(PathBuf),
    /// A directory's entries were synced.
    Dir(PathBuf),
    /// A file was renamed.
    Rename {
        /// The old name.
        from: PathBuf,
        /// The new name.
        to: PathBuf,
    },
}

impl SyncEvent {
    /// The path the event is about: the synced file or directory, or a
    /// rename's new name.
    pub fn path(&self) -> &Path {
        match self {
            SyncEvent::File(path) | SyncEvent::Dir(path) => path,
            SyncEvent::Rename { to, .. } => to,
        }
    }
}

#[derive(Default)]
struct TraceState {
    events: Vec<SyncEvent>,
    failing_dirs: Vec<PathBuf>,
}

thread_local! {
    static TRACE: RefCell<Option<TraceState>> = const { RefCell::new(None) };
}

/// Records the syncs and renames this module requests on the current
/// thread while it is alive, for tests.
///
/// Only operations on the thread that started the trace are recorded, so a
/// test sees its own operations and none of the tests running beside it.
/// Code under test that moves the work to another thread (`spawn_blocking`,
/// rayon) is not seen. Every event is recorded before the operation runs and
/// on every platform, including the directory syncs that do nothing on
/// Windows.
#[doc(hidden)]
#[derive(Debug)]
pub struct SyncTrace {
    // Thread-bound: the recorder lives in a thread-local.
    _not_send: PhantomData<*const ()>,
}

impl SyncTrace {
    /// Start recording on the current thread.
    ///
    /// # Panics
    ///
    /// Panics if a trace is already recording on this thread.
    pub fn start() -> Self {
        TRACE.with(|trace| {
            let mut trace = trace.borrow_mut();
            assert!(
                trace.is_none(),
                "a SyncTrace is already recording on this thread"
            );
            *trace = Some(TraceState::default());
        });
        Self {
            _not_send: PhantomData,
        }
    }

    /// The events recorded so far, in order.
    pub fn events(&self) -> Vec<SyncEvent> {
        TRACE.with(|trace| {
            trace
                .borrow()
                .as_ref()
                .map(|state| state.events.clone())
                .unwrap_or_default()
        })
    }

    /// Make every later sync of the directory `dir` fail (after it is
    /// recorded), to test what a failed sync leaves behind.
    pub fn fail_dir_sync(&self, dir: impl Into<PathBuf>) {
        let dir = dir.into();
        TRACE.with(|trace| {
            if let Some(state) = trace.borrow_mut().as_mut() {
                state.failing_dirs.push(dir);
            }
        });
    }
}

impl Drop for SyncTrace {
    fn drop(&mut self) {
        let _ = TRACE.try_with(|trace| *trace.borrow_mut() = None);
    }
}

/// Record `event` if a [`SyncTrace`] is recording on this thread, and
/// return whether its directory sync was told to fail.
fn record(event: SyncEvent) -> bool {
    TRACE
        .try_with(|trace| {
            let mut trace = trace.borrow_mut();
            let Some(state) = trace.as_mut() else {
                return false;
            };
            let fails = matches!(&event, SyncEvent::Dir(dir) if state.failing_dirs.contains(dir));
            state.events.push(event);
            fails
        })
        .unwrap_or(false)
}

/// The directory that holds `path`: its parent, or `.` for a bare relative
/// name such as `idx`, whose parent is the empty path.
pub(crate) fn parent_dir(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

/// Sync the entries of the directory `dir` (a no-op on Windows).
pub(crate) fn sync_dir(dir: &Path) -> io::Result<()> {
    if record(SyncEvent::Dir(dir.to_path_buf())) {
        return Err(io::Error::other(format!(
            "injected failure syncing {}",
            dir.display()
        )));
    }
    platform::sync_dir(dir)
}

/// Sync the contents of `file`, which was opened at `path`.
pub(crate) fn sync_file(file: &File, path: &Path) -> io::Result<()> {
    record(SyncEvent::File(path.to_path_buf()));
    file.sync_all()
}

/// Rename `from` to `to`, replacing `to` if it exists.
pub(crate) fn rename(from: &Path, to: &Path) -> io::Result<()> {
    record(SyncEvent::Rename {
        from: from.to_path_buf(),
        to: to.to_path_buf(),
    });
    std::fs::rename(from, to)
}

/// Create `dir` and any missing parent, and make each directory it creates
/// durable by syncing the directory that holds it.
///
/// The directories are created from the top down, and each one's parent is
/// synced right after it is created, so a parent's entry is durable before
/// its child's. Nothing is synced when `dir` already exists. A directory
/// that another process creates meanwhile is left to that process.
///
/// # Errors
///
/// Returns the error from inspecting, creating or syncing a directory,
/// including when a path component exists but is not a directory.
pub(crate) fn create_dir_all(dir: &Path) -> io::Result<()> {
    // The missing directories, deepest first. The empty path that ends a
    // relative path's ancestors is the current directory, which exists.
    let mut missing = Vec::new();
    for ancestor in dir.ancestors() {
        if ancestor.as_os_str().is_empty() {
            break;
        }
        match std::fs::metadata(ancestor) {
            Ok(_) => break,
            Err(err) if err.kind() == io::ErrorKind::NotFound => missing.push(ancestor),
            Err(err) => return Err(err),
        }
    }

    for created in missing.into_iter().rev() {
        match std::fs::create_dir(created) {
            Ok(()) => sync_dir(parent_dir(created))?,
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists && created.is_dir() => {}
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

/// Replace the file at `path` with `contents` atomically and durably, by
/// way of the temporary file `tmp` in the same directory.
///
/// Writes `tmp` (truncating one an earlier crash left), gives it the
/// permissions of the file at `path` on Unix, syncs it, renames it over
/// `path` and syncs the directory. A crash leaves either the old file or
/// the new one, never a partial one. On an error before the rename, `tmp`
/// is removed; after the rename, the new file is in place even though the
/// error is returned.
///
/// # Errors
///
/// Returns the error from writing, syncing or renaming `tmp`, or from
/// syncing the directory.
pub(crate) fn replace_file(path: &Path, tmp: &Path, contents: &[u8]) -> io::Result<()> {
    let written = write_synced(path, tmp, contents).and_then(|()| rename(tmp, path));
    if let Err(err) = written {
        if let Err(remove_err) = std::fs::remove_file(tmp)
            && remove_err.kind() != io::ErrorKind::NotFound
        {
            log::warn!(
                "failed to remove {} after a failed write: {remove_err}",
                tmp.display()
            );
        }
        return Err(err);
    }
    sync_dir(parent_dir(path))
}

/// Write `contents` to `tmp` with `path`'s permissions and sync it.
fn write_synced(path: &Path, tmp: &Path, contents: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(tmp)?;
    file.write_all(contents)?;
    platform::copy_permissions(path, &file)?;
    sync_file(&file, tmp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_of_a_bare_relative_name_is_the_current_dir() {
        assert_eq!(parent_dir(Path::new("idx")), Path::new("."));
        assert_eq!(parent_dir(Path::new("a/idx")), Path::new("a"));
    }

    #[test]
    fn create_dir_all_syncs_the_parent_of_each_directory_it_creates() {
        let root = tempfile::TempDir::new().unwrap();
        let dir = root.path().join("a").join("b");

        let trace = SyncTrace::start();
        create_dir_all(&dir).unwrap();

        assert!(dir.is_dir());
        assert_eq!(
            trace.events(),
            vec![
                SyncEvent::Dir(root.path().to_path_buf()),
                SyncEvent::Dir(root.path().join("a")),
            ]
        );
    }

    #[test]
    fn create_dir_all_over_an_existing_directory_syncs_nothing() {
        let root = tempfile::TempDir::new().unwrap();

        let trace = SyncTrace::start();
        create_dir_all(root.path()).unwrap();

        assert_eq!(trace.events(), Vec::new());
    }

    #[test]
    fn create_dir_all_under_a_file_fails() {
        let root = tempfile::TempDir::new().unwrap();
        let file = root.path().join("file");
        std::fs::write(&file, b"").unwrap();

        assert!(create_dir_all(&file.join("dir")).is_err());
    }

    #[test]
    fn a_failing_dir_sync_is_recorded_and_returns_an_error() {
        let root = tempfile::TempDir::new().unwrap();

        let trace = SyncTrace::start();
        trace.fail_dir_sync(root.path());
        let err = create_dir_all(&root.path().join("a")).unwrap_err();

        assert!(err.to_string().contains("injected failure"), "got: {err}");
        assert_eq!(
            trace.events(),
            vec![SyncEvent::Dir(root.path().to_path_buf())]
        );
    }

    #[test]
    fn nothing_is_recorded_without_a_trace() {
        let root = tempfile::TempDir::new().unwrap();
        create_dir_all(&root.path().join("a")).unwrap();

        let trace = SyncTrace::start();
        assert_eq!(trace.events(), Vec::new());
    }
}
