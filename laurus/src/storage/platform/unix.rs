//! Unix-family storage platform knobs (Issue #504, #508).
//!
//! On Unix the kernel keeps mmap-backed inodes alive across
//! `unlink` / `truncate` / overwrite, so the file storage backend
//! can default `use_mmap` to `true` without colliding with the
//! segment-overwrite path. Operators opt out via the
//! `LAURUS_NO_MMAP=1` environment variable for debug sessions or
//! hosts where mmap misbehaves.
//!
//! Directories can be opened as files and fsynced, which makes the names
//! created, renamed or removed in them durable.
//!
//! An index directory's lock file (Issue #1665) can be unlinked while a
//! process has it open, so a process that opened it just before the
//! unlink can lock a file that no longer has the name; a claim checks the
//! name again after it takes the lock ([`is_lock_file_current`]).

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

/// Returns the platform default for [`crate::storage::file::FileStorageConfig::use_mmap`].
///
/// Reads `LAURUS_NO_MMAP` at construction time and returns `false`
/// when it is set to `"1"`. Otherwise returns `true`.
///
/// Reading the env var here (instead of at every call site) keeps
/// the toggle close to the policy decision and lets tests / fixtures
/// pick either path without code changes.
pub fn default_use_mmap() -> bool {
    !matches!(std::env::var("LAURUS_NO_MMAP").as_deref(), Ok("1"))
}

/// Flush `dir`'s entries (the names created, renamed or removed in it) to
/// disk, by opening the directory and calling `fsync(2)` on it (Issue #1587).
///
/// # Errors
///
/// Returns the error from opening or syncing the directory.
pub fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

/// Give `to` the permission bits of the file at `from`, when there is one
/// (Issue #1587).
///
/// A file replaced by renaming a new file over it would otherwise get the
/// new file's mode, which comes from the umask, instead of the mode the
/// operator gave the old one.
///
/// # Errors
///
/// Returns the error from reading `from`'s metadata (other than its
/// absence) or from setting `to`'s permissions.
pub fn copy_permissions(from: &Path, to: &File) -> io::Result<()> {
    match std::fs::metadata(from) {
        Ok(metadata) => to.set_permissions(metadata.permissions()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

/// Open an index directory's lock file at `path` for reading and writing,
/// creating it if it is missing (Issue #1665).
///
/// Returns the file and whether this call created it. The file stays empty;
/// only its lock matters.
///
/// # Errors
///
/// Returns the error from opening or creating the file, including
/// [`io::ErrorKind::NotFound`] when the directory that should hold it is
/// gone.
pub fn open_lock_file(path: &Path) -> io::Result<(File, bool)> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    match options.clone().create_new(true).open(path) {
        Ok(file) => Ok((file, true)),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => Ok((options.open(path)?, false)),
        Err(err) => Err(err),
    }
}

/// Whether `path` still names the file `file` was opened from (Issue
/// #1665).
///
/// A claim that opened the lock file just before another claim unlinked it
/// and released its lock would otherwise hold a lock on a file no one else
/// can open, while a third claim locks a new file under the same name.
///
/// # Errors
///
/// Returns the error from reading either file's metadata, other than the
/// absence of `path`, which is `Ok(false)`.
pub fn is_lock_file_current(file: &File, path: &Path) -> io::Result<bool> {
    let held = file.metadata()?;
    match std::fs::metadata(path) {
        Ok(named) => Ok(held.dev() == named.dev() && held.ino() == named.ino()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err),
    }
}

/// Remove the lock file at `path`, whose lock `file` holds, and release
/// the lock (Issue #1665).
///
/// The name is removed while the lock is still held, so every claim that
/// takes the lock afterwards finds the name gone or naming another file
/// ([`is_lock_file_current`]). The removal is not synced: a lock file that
/// a crash brings back only lets the next claim reuse it.
///
/// # Errors
///
/// Returns the error from removing the file. The lock is released either
/// way.
pub fn remove_lock_file(file: File, path: &Path) -> io::Result<()> {
    let removed = std::fs::remove_file(path);
    drop(file);
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_dir_syncs_a_directory() {
        let dir = tempfile::TempDir::new().unwrap();
        sync_dir(dir.path()).unwrap();
    }

    /// The directory is really opened: a body that only returns `Ok(())`,
    /// as on Windows, would not fail here.
    #[test]
    fn sync_dir_fails_on_a_missing_directory() {
        let dir = tempfile::TempDir::new().unwrap();
        let err = sync_dir(&dir.path().join("missing")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    /// Issue #1665: the first open creates the lock file, a later one
    /// reuses it.
    #[test]
    fn open_lock_file_reports_whether_it_created_the_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("index.lock");

        let (_first, created_first) = open_lock_file(&path).unwrap();
        let (_second, created_second) = open_lock_file(&path).unwrap();

        assert!(created_first);
        assert!(!created_second);
    }

    /// Issue #1665: a lock file whose name was removed, or now names
    /// another file, is not current.
    #[test]
    fn a_lock_file_is_current_only_while_its_name_points_at_it() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("index.lock");
        let (held, _) = open_lock_file(&path).unwrap();
        assert!(is_lock_file_current(&held, &path).unwrap());

        std::fs::remove_file(&path).unwrap();
        assert!(!is_lock_file_current(&held, &path).unwrap());

        let (_new, created) = open_lock_file(&path).unwrap();
        assert!(created);
        assert!(!is_lock_file_current(&held, &path).unwrap());
    }

    /// Issue #1665: removing the lock file also releases its lock.
    #[test]
    fn remove_lock_file_removes_the_name_and_releases_the_lock() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("index.lock");
        let (held, _) = open_lock_file(&path).unwrap();
        held.try_lock().unwrap();
        let (other, _) = open_lock_file(&path).unwrap();
        assert!(other.try_lock().is_err(), "the lock is held");

        remove_lock_file(held, &path).unwrap();

        assert!(!path.exists());
        other.try_lock().unwrap();
    }
}
