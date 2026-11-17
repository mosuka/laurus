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
//!
//! An index directory's lock file (Issue #1665) is opened without
//! `FILE_SHARE_DELETE`, so it cannot be removed or renamed while any call
//! has it open, and the name a claim locked always names the locked file.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::windows::fs::OpenOptionsExt;
use std::path::Path;

/// `FILE_SHARE_READ | FILE_SHARE_WRITE`, without `FILE_SHARE_DELETE`.
const LOCK_FILE_SHARE_MODE: u32 = 0x1 | 0x2;

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

/// Open the existing file at `path` to read it and then sync it, for
/// reading and writing without truncating it (Issue #1668):
/// `FlushFileBuffers`, which `File::sync_all` calls, needs write access.
///
/// # Errors
///
/// Returns the error from opening the file, including the one for a file
/// that cannot be opened for writing (a read-only file, for example).
pub fn open_for_sync(path: &Path) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).open(path)
}

/// Open an index directory's lock file at `path` for reading and writing,
/// creating it if it is missing, without `FILE_SHARE_DELETE` (Issue #1665).
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
    options
        .read(true)
        .write(true)
        .share_mode(LOCK_FILE_SHARE_MODE);
    match options.clone().create_new(true).open(path) {
        Ok(file) => Ok((file, true)),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => Ok((options.open(path)?, false)),
        Err(err) => Err(err),
    }
}

/// Always `true`: a lock file opened by [`open_lock_file`] cannot be
/// removed or renamed while it is open, so its name still names it (Issue
/// #1665).
///
/// # Errors
///
/// Never returns an error.
pub fn is_lock_file_current(_file: &File, _path: &Path) -> io::Result<bool> {
    Ok(true)
}

/// Release the lock `file` holds on the lock file at `path`, close it, and
/// then remove the file (Issue #1665).
///
/// The file cannot be removed while it is open, so it is closed first. If
/// another call opened it in between, the removal fails and the file stays,
/// which only lets that call go on.
///
/// # Errors
///
/// Returns the error from removing the file. The lock is released either
/// way.
pub fn remove_lock_file(file: File, path: &Path) -> io::Result<()> {
    let _ = file.unlock();
    drop(file);
    std::fs::remove_file(path)
}
