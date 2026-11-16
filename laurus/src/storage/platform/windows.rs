//! Windows storage platform knobs (Issue #504, #508).
//!
//! Windows holds an exclusive lock on memory-mapped files
//! (`ERROR_USER_MAPPED_FILE`, os error 1224) which prevents the
//! writer from truncating / deleting a segment file while a reader
//! still holds an mmap. Issue #508 (Approach A) resolves this at the
//! storage layer by evicting the cached `Arc<Mmap>` in
//! `FileStorage::create_output` / `delete_file` before the file
//! operation — laurus readers consume their `StorageInput` within a
//! single function scope, so eviction is sufficient to release the
//! lock without coordination at the engine layer.
//!
//! With that contract in place, this platform module is now
//! symmetric with [`super::unix`]: mmap is the default on Windows
//! too, with `LAURUS_NO_MMAP=1` available as an opt-out for hosts
//! where mmap misbehaves.
//!
//! Directories cannot be opened as files through `std`, so
//! [`sync_dir`] is a no-op here.

use std::fs::File;
use std::io;
use std::path::Path;

/// Returns the platform default for [`crate::storage::file::FileStorageConfig::use_mmap`].
///
/// Reads `LAURUS_NO_MMAP` at construction time and returns `false`
/// only when it is set to `"1"`. Otherwise returns `true` — mmap is
/// the default on Windows as of Issue #508.
///
/// Reading the env var here (instead of at every call site) keeps
/// the toggle close to the policy decision and lets tests / fixtures
/// pick either path without code changes.
pub fn default_use_mmap() -> bool {
    !matches!(std::env::var("LAURUS_NO_MMAP").as_deref(), Ok("1"))
}

/// Does nothing: `std` cannot open a directory as a file on Windows, so
/// there is no handle to flush (Issue #1587).
///
/// The file storage's own directory sync skips Windows the same way. A
/// file's contents are still flushed by `File::sync_all`, and a rename is
/// atomic, but the directory entries are not forced to disk.
///
/// # Errors
///
/// Never returns an error.
pub fn sync_dir(_dir: &Path) -> io::Result<()> {
    Ok(())
}

/// Does nothing: a new file inherits its access control list from the
/// directory it is created in, so a file renamed over another one already
/// has the permissions the old one would have inherited (Issue #1587).
///
/// Copying the read-only attribute would instead make the new file one
/// that a failed rename cannot delete and the next write cannot reuse.
///
/// # Errors
///
/// Never returns an error.
pub fn copy_permissions(_from: &Path, _to: &File) -> io::Result<()> {
    Ok(())
}
