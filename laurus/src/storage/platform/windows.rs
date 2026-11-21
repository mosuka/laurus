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
//! `std` opens a directory as a file only with
//! `FILE_FLAG_BACKUP_SEMANTICS`, which [`sync_dir`] passes to flush the
//! directory's entries (Issue #1667).
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

/// `FILE_FLAG_BACKUP_SEMANTICS`, without which a directory cannot be opened.
const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

/// `FILE_ADD_FILE` (`FILE_WRITE_DATA`): the right to create a file in a
/// directory.
const FILE_ADD_FILE: u32 = 0x2;

/// `FILE_ADD_SUBDIRECTORY` (`FILE_APPEND_DATA`): the right to create a
/// subdirectory in a directory.
const FILE_ADD_SUBDIRECTORY: u32 = 0x4;

/// `ERROR_INVALID_FUNCTION`, which an SMB share returns for the flush of a
/// directory.
const ERROR_INVALID_FUNCTION: i32 = 1;

/// `ERROR_NOT_SUPPORTED`.
const ERROR_NOT_SUPPORTED: i32 = 50;

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

/// Flush `dir`'s entries (the names created, renamed or removed in it) to
/// disk, by opening the directory and calling `FlushFileBuffers` on it
/// (Issue #1667).
///
/// The flush needs a handle with the right to add a file or a subdirectory
/// to `dir`. The directory is opened with the first, or with the second if
/// the first is denied: a volume's root directory grants a user without
/// administrator rights only the second.
///
/// Windows documents that the flush persists the directory's entries on
/// NTFS only (MS-FSA, section 2.1.5.7); other file systems, such as ReFS
/// and FAT, complete it without doing so. A file system that cannot flush
/// a directory at all, such as an SMB share, fails it with
/// `ERROR_INVALID_FUNCTION` or `ERROR_NOT_SUPPORTED`; the sync then does
/// nothing, logs that at debug level, and leaves the entries to the file
/// system, as on any file system before Issue #1667.
///
/// # Errors
///
/// Returns the error from opening the directory, including
/// [`io::ErrorKind::NotFound`] for a missing one and
/// [`io::ErrorKind::PermissionDenied`] when neither right is granted, or
/// from flushing it, other than the two above.
pub fn sync_dir(dir: &Path) -> io::Result<()> {
    let file = open_dir_to_flush(dir)?;
    match file.sync_all() {
        Err(err) if cannot_flush_a_directory(&err) => {
            log::debug!(
                "{} cannot be flushed on its file system ({err}); its entries are left to the file system",
                dir.display()
            );
            Ok(())
        }
        flushed => flushed,
    }
}

/// Open the directory `dir` with the access a flush needs: the right to add
/// a file, or the right to add a subdirectory if the first is denied.
fn open_dir_to_flush(dir: &Path) -> io::Result<File> {
    let open = |access| {
        OpenOptions::new()
            .access_mode(access)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(dir)
    };
    match open(FILE_ADD_FILE) {
        Err(err) if err.kind() == io::ErrorKind::PermissionDenied => open(FILE_ADD_SUBDIRECTORY),
        opened => opened,
    }
}

/// Whether `err`, from flushing a directory, means that its file system
/// cannot flush a directory.
fn cannot_flush_a_directory(err: &io::Error) -> bool {
    matches!(
        err.raw_os_error(),
        Some(ERROR_INVALID_FUNCTION | ERROR_NOT_SUPPORTED)
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_dir_syncs_a_directory() {
        let dir = tempfile::TempDir::new().unwrap();
        sync_dir(dir.path()).unwrap();
    }

    /// Issue #1667: the directory is really opened; the body that only
    /// returned `Ok(())` does not fail here.
    #[test]
    fn sync_dir_fails_on_a_missing_directory() {
        let dir = tempfile::TempDir::new().unwrap();
        let err = sync_dir(&dir.path().join("missing")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    /// Issue #1667: a directory that denies the right to add a file is
    /// opened with the right to add a subdirectory, which is all a volume's
    /// root grants a user without administrator rights.
    #[test]
    fn sync_dir_opens_with_the_right_to_add_a_subdirectory_when_adding_a_file_is_denied() {
        let parent = tempfile::TempDir::new().unwrap();
        let dir = parent.path().join("no_add_file");
        std::fs::create_dir(&dir).unwrap();
        // `*S-1-1-0` is Everyone, and `WD` is `FILE_ADD_FILE`.
        let output = std::process::Command::new("icacls")
            .arg(&dir)
            .args(["/deny", "*S-1-1-0:(WD)"])
            .output()
            .unwrap();
        assert!(output.status.success(), "icacls failed: {output:?}");

        let add_file = OpenOptions::new()
            .access_mode(FILE_ADD_FILE)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(&dir);
        if add_file.is_ok() {
            // A process whose backup privilege is enabled bypasses the ACL,
            // so the first open succeeds and the fallback is not reached.
            eprintln!("the ACL is not enforced for this process; skipping");
            return;
        }
        assert_eq!(
            add_file.unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );

        sync_dir(&dir).unwrap();
    }

    /// Issue #1667: only the errors of a file system that cannot flush a
    /// directory make the sync do nothing.
    #[test]
    fn only_the_errors_of_a_file_system_without_directory_flushes_are_ignored() {
        for code in [ERROR_INVALID_FUNCTION, ERROR_NOT_SUPPORTED] {
            assert!(
                cannot_flush_a_directory(&io::Error::from_raw_os_error(code)),
                "os error {code}"
            );
        }
        // ERROR_FILE_NOT_FOUND, ERROR_ACCESS_DENIED and ERROR_SHARING_VIOLATION.
        for code in [2, 5, 32] {
            assert!(
                !cannot_flush_a_directory(&io::Error::from_raw_os_error(code)),
                "os error {code}"
            );
        }
        assert!(!cannot_flush_a_directory(&io::Error::other(
            "not an os error"
        )));
    }
}
