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

use std::fs::File;
use std::io;
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
}
