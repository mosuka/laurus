//! In-memory storage implementation for testing and caching.
//!
//! This module provides a complete [`Storage`] implementation backed entirely
//! by in-process memory (`HashMap<String, Arc<MemFile>>`). It is designed for:
//!
//! - **Unit and integration testing** -- fast, deterministic, no filesystem
//!   side effects.
//! - **Temporary indexes** -- building an ephemeral index that does not need
//!   to survive process restarts.
//! - **Benchmarking** -- removing disk I/O from the critical path so that
//!   algorithmic performance can be measured in isolation.
//!
//! ## Thread Safety
//!
//! The file map is protected by a [`parking_lot::RwLock`] so that concurrent
//! readers do not block each other -- important because reads vastly outnumber
//! writes during search. [`MemoryStorage`] can be shared across threads via
//! `Arc<MemoryStorage>`. Individual I/O handles ([`MemoryInput`] /
//! [`MemoryOutput`]) are **not** `Sync` but are safe to send between threads.
//!
//! ## When to Use `MemoryStorage` vs `FileStorage`
//!
//! | Criterion | `MemoryStorage` | `FileStorage` |
//! |---|---|---|
//! | Persistence | None (lost on drop) | Durable on disk |
//! | Performance | Very fast (no syscalls) | Disk-bound |
//! | Capacity | Limited by process memory | Limited by disk |
//! | Use case | Tests, benchmarks, transient data | Production indexes |

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

use parking_lot::RwLock;

use crate::error::{LaurusError, Result};
use crate::storage::{
    LockManager, Storage, StorageError, StorageInput, StorageLock, StorageOutput,
};

/// Configuration specific to memory-based storage.
///
/// Allows callers to tune the initial capacity of the internal file map,
/// reducing re-allocations when the approximate number of files is known
/// ahead of time.
#[derive(Debug, Clone)]
pub struct MemoryStorageConfig {
    /// Initial capacity hint for the file map.
    pub initial_capacity: usize,
}

impl Default for MemoryStorageConfig {
    fn default() -> Self {
        MemoryStorageConfig {
            initial_capacity: 16,
        }
    }
}

/// The mutable interior of a [`MemFile`].
///
/// `data` is the full backing buffer for one logical file; it grows as an open
/// writer appends and is **never** truncated. `committed` is the number of
/// leading bytes that have been flushed (via
/// [`StorageOutput::flush_and_sync`](crate::storage::StorageOutput::flush_and_sync)
/// or `close`) and are therefore visible to readers. Bytes in
/// `data[committed..]` are a written-but-not-yet-flushed tail and stay invisible
/// to [`MemoryStorage::open_input`] until the next flush advances `committed`.
#[derive(Debug)]
struct MemFileInner {
    /// Full backing buffer; may include a not-yet-flushed tail beyond `committed`.
    data: Vec<u8>,
    /// Number of leading bytes flushed and visible to readers.
    committed: usize,
}

/// A single in-memory file shared between the storage map and any open writer.
///
/// The writer and the [`MemoryStorage`] file map hold the **same** `Arc<MemFile>`,
/// so a flush only has to advance the `committed` length instead of cloning the
/// whole buffer — making the WAL append-then-flush pattern amortized O(1) per
/// flush rather than O(buffer) (Issue #812). `data` and `committed` live under one
/// [`parking_lot::RwLock`] so a reader holding the read guard always observes a
/// consistent `(committed, bytes)` pair.
#[derive(Debug)]
struct MemFile {
    /// Backing buffer and committed length under a single lock.
    inner: RwLock<MemFileInner>,
}

impl MemFile {
    /// Create a new `MemFile` from `data` with the first `committed` bytes
    /// marked as flushed and visible to readers.
    ///
    /// # Arguments
    ///
    /// * `data` - The backing buffer.
    /// * `committed` - Number of leading bytes already flushed/visible.
    fn new(data: Vec<u8>, committed: usize) -> Self {
        MemFile {
            inner: RwLock::new(MemFileInner { data, committed }),
        }
    }
}

/// An in-memory storage implementation.
///
/// All files are held in a shared `HashMap<String, Arc<MemFile>>` guarded by a
/// [`parking_lot::RwLock`]. Each `MemFile` keeps a growing backing buffer plus
/// a `committed` length marking the reader-visible prefix; an open writer shares
/// the same `Arc<MemFile>` as the map, so flushing only advances `committed`
/// instead of cloning the buffer (Issue #812). The outer `RwLock` allows multiple
/// concurrent readers without blocking, which is important because reads vastly
/// outnumber writes during search operations.
///
/// This is useful for testing and for creating temporary indexes in memory.
#[derive(Debug)]
pub struct MemoryStorage {
    /// The files stored in memory, each a shared growing buffer with a committed
    /// (reader-visible) length.
    files: Arc<RwLock<HashMap<String, Arc<MemFile>>>>,
    /// Lock manager for coordinating access.
    lock_manager: Arc<MemoryLockManager>,
    /// Storage configuration.
    #[allow(dead_code)]
    config: MemoryStorageConfig,
    /// Whether the storage is closed.
    closed: bool,
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new(MemoryStorageConfig::default())
    }
}

impl MemoryStorage {
    /// Create a new memory storage with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration controlling initial capacity and other
    ///   tuning parameters.
    ///
    /// # Returns
    ///
    /// A new, empty `MemoryStorage`.
    pub fn new(config: MemoryStorageConfig) -> Self {
        let initial_capacity = config.initial_capacity;
        MemoryStorage {
            files: Arc::new(RwLock::new(HashMap::with_capacity(initial_capacity))),
            lock_manager: Arc::new(MemoryLockManager::new()),
            config,
            closed: false,
        }
    }

    /// Check if the storage is closed, returning an error if so.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::StorageClosed`] when the storage has been
    /// closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            Err(StorageError::StorageClosed.into())
        } else {
            Ok(())
        }
    }

    /// Get the number of files currently stored.
    ///
    /// # Returns
    ///
    /// The file count.
    #[inline]
    pub fn file_count(&self) -> usize {
        self.files.read().len()
    }

    /// Get the total size of all files in bytes.
    ///
    /// # Returns
    ///
    /// The sum of all file sizes.
    pub fn total_size(&self) -> u64 {
        let files = self.files.read();
        files
            .values()
            .map(|file| file.inner.read().committed as u64)
            .sum()
    }

    /// Remove all files from storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage is closed.
    pub fn clear(&self) -> Result<()> {
        self.check_closed()?;
        let mut files = self.files.write();
        files.clear();
        Ok(())
    }
}

impl Storage for MemoryStorage {
    #[inline]
    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.check_closed()?;

        // Use read lock: concurrent readers do not block each other. Clone the
        // shared `Arc<MemFile>` out and drop the map lock before touching the
        // file's `inner` lock, so the map lock and the file lock are never held
        // at the same time (avoids a lock-ordering cycle with `flush_and_sync`).
        let file = {
            let files = self.files.read();
            files
                .get(name)
                .cloned()
                .ok_or_else(|| StorageError::FileNotFound(name.to_string()))?
        };

        // Snapshot only the committed (flushed) prefix so a not-yet-flushed tail
        // stays invisible to readers.
        let snapshot = {
            let inner = file.inner.read();
            inner.data[..inner.committed].to_vec()
        };

        Ok(Box::new(MemoryInput::new(snapshot)))
    }

    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.check_closed()?;

        Ok(Box::new(MemoryOutput::new(
            name.to_string(),
            Arc::clone(&self.files),
        )))
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.check_closed()?;

        // For memory storage, append is same as create (data persists in memory)
        Ok(Box::new(MemoryOutput::new_append(
            name.to_string(),
            Arc::clone(&self.files),
        )))
    }

    fn file_exists(&self, name: &str) -> bool {
        if self.closed {
            return false;
        }

        let files = self.files.read();
        files.contains_key(name)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.check_closed()?;

        let mut files = self.files.write();
        files.remove(name);
        Ok(())
    }

    fn list_files(&self) -> Result<Vec<String>> {
        self.check_closed()?;

        let files = self.files.read();
        let mut file_names: Vec<String> = files.keys().cloned().collect();
        file_names.sort();
        Ok(file_names)
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.check_closed()?;

        let files = self.files.read();
        let file = files
            .get(name)
            .ok_or_else(|| StorageError::FileNotFound(name.to_string()))?;

        Ok(file.inner.read().committed as u64)
    }

    /// Returns metadata for the named in-memory file.
    ///
    /// **Note:** Because in-memory storage does not track real modification or
    /// creation timestamps, both `modified` and `created` are set to the
    /// current wall-clock time at the moment this method is called. They do
    /// **not** reflect when the file was actually written or first created.
    fn metadata(&self, name: &str) -> Result<crate::storage::FileMetadata> {
        self.check_closed()?;

        let files = self.files.read();
        if let Some(file) = files.get(name) {
            let now = crate::util::time::now_secs();

            Ok(crate::storage::FileMetadata {
                size: file.inner.read().committed as u64,
                modified: now,
                created: now,
                readonly: false,
            })
        } else {
            Err(LaurusError::storage(format!("File not found: {name}")))
        }
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.check_closed()?;

        let mut files = self.files.write();
        let data = files
            .remove(old_name)
            .ok_or_else(|| StorageError::FileNotFound(old_name.to_string()))?;

        files.insert(new_name.to_string(), data);
        Ok(())
    }

    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        self.check_closed()?;

        let mut counter = 0;
        let mut temp_name;

        loop {
            temp_name = format!("{prefix}_{counter}.tmp");
            if !self.file_exists(&temp_name) {
                break;
            }
            counter += 1;

            if counter > 10000 {
                return Err(
                    StorageError::IoError("Could not create temporary file".to_string()).into(),
                );
            }
        }

        let output = self.create_output(&temp_name)?;
        Ok((temp_name, output))
    }

    fn sync(&self) -> Result<()> {
        self.check_closed()?;
        // For memory storage, sync is a no-op
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.closed = true;
        self.lock_manager.release_all()?;
        Ok(())
    }

    fn lock_manager(&self) -> Option<Arc<dyn LockManager>> {
        Some(self.lock_manager.clone())
    }
}

/// A read handle backed by an in-memory byte buffer.
///
/// `MemoryInput` wraps a [`Cursor<Vec<u8>>`] and implements [`StorageInput`]
/// so that callers can seek and read just as they would with a file-backed
/// input.
#[derive(Debug)]
pub struct MemoryInput {
    /// Cursor providing Read + Seek over the byte buffer.
    cursor: Cursor<Vec<u8>>,
    /// Cached size of the data (avoids re-computing from the cursor).
    size: u64,
}

impl MemoryInput {
    /// Build a read handle that owns `data` as its private snapshot.
    ///
    /// The buffer is decoupled from the shared [`MemFile`], so subsequent writes
    /// or map mutations cannot invalidate it (including the zero-copy
    /// [`as_slice`](StorageInput::as_slice) path).
    ///
    /// # Arguments
    ///
    /// * `data` - The snapshot bytes this input reads over.
    fn new(data: Vec<u8>) -> Self {
        let size = data.len() as u64;
        let cursor = Cursor::new(data);
        MemoryInput { cursor, size }
    }
}

impl Read for MemoryInput {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl Seek for MemoryInput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl StorageInput for MemoryInput {
    fn size(&self) -> Result<u64> {
        Ok(self.size)
    }

    fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
        Ok(Box::new(MemoryInput::new(self.cursor.get_ref().clone())))
    }

    fn close(&mut self) -> Result<()> {
        // Nothing to close for memory input
        Ok(())
    }

    /// Borrow the in-memory buffer from the current read position to
    /// the end. Issue #504 zero-copy path: callers (the lexical
    /// posting decoder) can take a slice into the buffer instead of
    /// re-allocating + `copy_from_slice` through `Read`.
    fn as_slice(&self) -> Option<&[u8]> {
        let pos = self.cursor.position() as usize;
        let data = self.cursor.get_ref();
        Some(&data[pos.min(data.len())..])
    }
}

/// A write handle backed by an in-memory byte buffer.
///
/// `MemoryOutput` writes directly into a shared `MemFile` buffer. Each
/// [`flush_and_sync`](StorageOutput::flush_and_sync) (or `close`) advances the
/// file's `committed` length and publishes the shared `Arc<MemFile>` into the
/// file map, so the flushed prefix becomes visible to subsequent
/// [`MemoryStorage::open_input`] calls. Because writer and map share one buffer,
/// a flush is amortized O(1) instead of cloning the whole buffer (Issue #812).
#[derive(Debug)]
pub struct MemoryOutput {
    /// The logical file name.
    name: String,
    /// Shared backing file this handle writes into. Held privately until the
    /// first flush/close publishes it into the map (preserving truncate
    /// semantics: a fresh `create_output` does not replace the old map entry
    /// until the new content is flushed).
    memfile: Arc<MemFile>,
    /// Shared reference to the storage file map for publishing on flush/close.
    files: Arc<RwLock<HashMap<String, Arc<MemFile>>>>,
    /// Current write position within the buffer.
    position: u64,
    /// Whether this output handle has been closed.
    closed: bool,
}

impl MemoryOutput {
    /// Create a truncating write handle.
    ///
    /// The handle starts with an empty private [`MemFile`]; the existing map
    /// entry (if any) stays readable until the first flush/close publishes this
    /// new content.
    ///
    /// # Arguments
    ///
    /// * `name` - The logical file name.
    /// * `files` - The shared storage file map to publish into on flush/close.
    fn new(name: String, files: Arc<RwLock<HashMap<String, Arc<MemFile>>>>) -> Self {
        MemoryOutput {
            name,
            memfile: Arc::new(MemFile::new(Vec::new(), 0)),
            files,
            position: 0,
            closed: false,
        }
    }

    /// Create an appending write handle.
    ///
    /// The committed prefix of the existing file is copied once into a new
    /// private [`MemFile`] and the write position is set to its end, so appended
    /// records extend the existing content. This one-time O(existing) copy keeps
    /// the new handle independent of the live map entry until it is published.
    ///
    /// # Arguments
    ///
    /// * `name` - The logical file name.
    /// * `files` - The shared storage file map to publish into on flush/close.
    fn new_append(name: String, files: Arc<RwLock<HashMap<String, Arc<MemFile>>>>) -> Self {
        // Preload only the committed (flushed) prefix of the existing entry.
        let existing_data = {
            let files_guard = files.read();
            files_guard
                .get(&name)
                .map(|file| {
                    let inner = file.inner.read();
                    inner.data[..inner.committed].to_vec()
                })
                .unwrap_or_default()
        };

        let position = existing_data.len() as u64;
        let committed = existing_data.len();

        MemoryOutput {
            name,
            memfile: Arc::new(MemFile::new(existing_data, committed)),
            files,
            position,
            closed: false,
        }
    }
}

impl Write for MemoryOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.closed {
            return Err(std::io::Error::other("Output is closed"));
        }

        let end_pos = (self.position as usize)
            .checked_add(buf.len())
            .ok_or_else(|| std::io::Error::other("File too large"))?;

        // Write into the shared backing buffer. Bytes land in `data` but stay
        // beyond `committed` (invisible to readers) until `flush_and_sync`.
        let mut inner = self.memfile.inner.write();
        if end_pos > inner.data.len() {
            // Resize buffer if needed, filling gaps with zeros.
            inner.data.resize(end_pos, 0);
        }
        inner.data[self.position as usize..end_pos].copy_from_slice(buf);
        drop(inner);

        self.position += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // For memory output, flushing is a no-op
        Ok(())
    }
}

impl Seek for MemoryOutput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        if self.closed {
            return Err(std::io::Error::other("Output is closed"));
        }

        let new_pos = match pos {
            SeekFrom::Start(offset) => offset,
            SeekFrom::End(offset) => {
                let len = self.memfile.inner.read().data.len() as u64;
                if offset < 0 {
                    let abs_offset = (-offset) as u64;
                    if abs_offset > len {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Invalid seek position",
                        ));
                    }
                    len - abs_offset
                } else {
                    len + offset as u64
                }
            }
            SeekFrom::Current(offset) => {
                if offset < 0 {
                    let abs_offset = (-offset) as u64;
                    if abs_offset > self.position {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Invalid seek position",
                        ));
                    }
                    self.position - abs_offset
                } else {
                    self.position + offset as u64
                }
            }
        };

        self.position = new_pos;
        Ok(new_pos)
    }
}

impl StorageOutput for MemoryOutput {
    fn flush_and_sync(&mut self) -> Result<()> {
        self.publish();
        Ok(())
    }

    fn position(&self) -> Result<u64> {
        Ok(self.position)
    }

    fn close(&mut self) -> Result<()> {
        if !self.closed {
            self.publish();
            self.closed = true;
        }
        Ok(())
    }
}

impl MemoryOutput {
    /// Make every byte written so far visible to readers and publish the shared
    /// [`MemFile`] into the storage map.
    ///
    /// Advances `committed` to the full written extent, then inserts an
    /// `Arc::clone` of the backing file under this handle's name. This mimics
    /// filesystem behavior where flushed data is visible to readers even while
    /// the writer is still open. The insert is O(1) (an `Arc` refcount bump, not
    /// a buffer copy), so repeated flushes on a growing WAL stay amortized O(1)
    /// (Issue #812).
    ///
    /// Inserting on **every** flush/close (rather than only the first)
    /// faithfully preserves last-writer-wins and resurrection semantics: a file
    /// deleted while this writer is open reappears on the next flush.
    ///
    /// The `inner` lock is released before the map lock is taken, so the two are
    /// never held simultaneously — this avoids a lock-ordering cycle with
    /// [`MemoryStorage::open_input`], which takes the map lock first.
    fn publish(&mut self) {
        {
            let mut inner = self.memfile.inner.write();
            inner.committed = inner.data.len();
        }
        self.files
            .write()
            .insert(self.name.clone(), Arc::clone(&self.memfile));
    }
}

impl Drop for MemoryOutput {
    fn drop(&mut self) {
        // Ensure the file is stored when the output is dropped
        let _ = self.close();
    }
}

/// A memory-based lock manager for coordinating concurrent access.
///
/// Locks are tracked in a `HashMap` keyed by lock name. This provides the
/// same semantics as file-based locks but without touching the filesystem.
#[derive(Debug)]
pub struct MemoryLockManager {
    /// Map of active locks, keyed by lock name.
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<MemoryLock>>>>>,
}

impl MemoryLockManager {
    fn new() -> Self {
        MemoryLockManager {
            locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl LockManager for MemoryLockManager {
    fn acquire_lock(&self, name: &str) -> Result<Box<dyn StorageLock>> {
        let mut locks = self.locks.lock().unwrap();

        if locks.contains_key(name) {
            return Err(StorageError::LockFailed(name.to_string()).into());
        }

        let lock = Arc::new(Mutex::new(MemoryLock::new(name.to_string())));
        locks.insert(name.to_string(), lock.clone());

        Ok(Box::new(MemoryLockWrapper {
            lock,
            name: name.to_string(),
            manager_locks: self.locks.clone(),
        }))
    }

    fn try_acquire_lock(&self, name: &str) -> Result<Option<Box<dyn StorageLock>>> {
        match self.acquire_lock(name) {
            Ok(lock) => Ok(Some(lock)),
            Err(e) => {
                if let LaurusError::Storage(ref msg) = e
                    && msg.contains("Failed to acquire lock")
                {
                    return Ok(None);
                }
                Err(e)
            }
        }
    }

    fn lock_exists(&self, name: &str) -> bool {
        let locks = self.locks.lock().unwrap();
        locks.contains_key(name)
    }

    fn release_all(&self) -> Result<()> {
        let mut locks = self.locks.lock().unwrap();
        locks.clear();
        Ok(())
    }
}

/// A memory-based lock implementation.
#[derive(Debug)]
struct MemoryLock {
    #[allow(dead_code)]
    name: String,
    released: bool,
}

impl MemoryLock {
    fn new(name: String) -> Self {
        MemoryLock {
            name,
            released: false,
        }
    }
}

/// A wrapper around [`MemoryLock`] that implements [`StorageLock`].
#[derive(Debug)]
struct MemoryLockWrapper {
    lock: Arc<Mutex<MemoryLock>>,
    /// This lock's name, and a handle to the owning [`MemoryLockManager`]'s
    /// `locks` map, so `release()` can remove its own entry (Issue #1086).
    /// Without this, a name could never be re-acquired after a proper
    /// `release()` -- `acquire_lock` only checks map membership, and
    /// nothing but `release_all()`'s full clear used to remove an entry.
    name: String,
    manager_locks: Arc<Mutex<HashMap<String, Arc<Mutex<MemoryLock>>>>>,
}

impl StorageLock for MemoryLockWrapper {
    /// Returns a fixed identifier `"memory_lock"` rather than the actual lock name.
    ///
    /// Because the real name is stored behind a `Mutex`, returning a borrowed
    /// reference to it is not possible with the current `&str` return type.
    fn name(&self) -> &str {
        "memory_lock"
    }

    fn release(&mut self) -> Result<()> {
        {
            let mut lock = self.lock.lock().unwrap();
            lock.released = true;
        }
        self.manager_locks.lock().unwrap().remove(&self.name);
        Ok(())
    }

    fn is_valid(&self) -> bool {
        let lock = self.lock.lock().unwrap();
        !lock.released
    }
}

impl Drop for MemoryLockWrapper {
    /// Releases the lock automatically once every handle holding it is
    /// dropped (Issue #1086), mirroring `FileLockWrapper`'s Drop impl and
    /// the existing `_wal_flush_timer`/`_commit_timer` "held only for its
    /// Drop side effect" pattern in `Engine`.
    fn drop(&mut self) {
        let _ = self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_memory_storage_creation() {
        let storage = MemoryStorage::default();
        assert_eq!(storage.file_count(), 0);
        assert_eq!(storage.total_size(), 0);
    }

    /// Issue #1086: `Storage::lock_manager()` must actually be wired to
    /// `MemoryStorage`'s lock manager, not just return `None` (the trait
    /// default every other implementor keeps).
    #[test]
    fn lock_manager_returns_the_memory_lock_manager() {
        let storage = MemoryStorage::default();
        assert!(
            storage.lock_manager().is_some(),
            "MemoryStorage must override the Storage::lock_manager() default"
        );
    }

    /// Two `Engine`s sharing the SAME `Arc<dyn Storage>` (the realistic
    /// in-process analogue of two processes opening the same directory)
    /// must not both succeed in acquiring the same lock name.
    #[test]
    fn try_acquire_lock_rejects_a_second_holder_over_the_same_manager() {
        let manager = MemoryLockManager::new();

        let first_lock = manager.try_acquire_lock("engine").unwrap();
        assert!(first_lock.is_some(), "the first holder must succeed");

        let second_lock = manager.try_acquire_lock("engine").unwrap();
        assert!(
            second_lock.is_none(),
            "a second holder over the same manager must be rejected, not silently succeed"
        );
    }

    /// Issue #1086 bug fix: before this fix, `acquire_lock` only checked
    /// map membership and `release()` never removed the manager's entry
    /// (only `release_all()`'s full clear did), so a name could never be
    /// re-acquired after a proper release -- even though nothing was
    /// actually holding it any more. This is the direct regression test:
    /// dropping the lock (the normal path when an `Engine` goes out of
    /// scope) must let a fresh acquisition of the SAME name succeed.
    #[test]
    fn dropping_the_lock_lets_a_fresh_acquisition_succeed() {
        let manager = MemoryLockManager::new();

        let first_lock = manager.try_acquire_lock("engine").unwrap();
        assert!(first_lock.is_some());
        drop(first_lock);

        let second_lock = manager.try_acquire_lock("engine").unwrap();
        assert!(
            second_lock.is_some(),
            "dropping the first lock must release it so a fresh acquisition succeeds"
        );
    }

    /// Releasing a lock (whether via an explicit `release()` call or via
    /// `Drop`) must clear the manager's own bookkeeping -- otherwise
    /// `lock_exists()` would keep reporting `true` for a name nothing
    /// holds any more.
    #[test]
    fn release_clears_lock_exists() {
        let manager = MemoryLockManager::new();

        let mut lock = manager.acquire_lock("engine").unwrap();
        assert!(manager.lock_exists("engine"));

        lock.release().unwrap();
        assert!(
            !manager.lock_exists("engine"),
            "release() must clear the manager's bookkeeping for this name"
        );
    }

    #[test]
    fn test_create_and_read_file() {
        let storage = MemoryStorage::default();

        // Create a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"Hello, Memory!").unwrap();
        output.close().unwrap();

        // Read the file
        let mut input = storage.open_input("test.txt").unwrap();
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer).unwrap();

        assert_eq!(buffer, b"Hello, Memory!");
        assert_eq!(input.size().unwrap(), 14);
        assert_eq!(storage.file_count(), 1);
        assert_eq!(storage.total_size(), 14);
    }

    #[test]
    fn as_slice_returns_remaining_bytes() {
        // Issue #504: MemoryInput exposes a zero-copy slice into the
        // in-memory buffer for the lexical posting decoder.
        let storage = MemoryStorage::default();
        let mut output = storage.create_output("data.bin").unwrap();
        output.write_all(b"abcdefghij").unwrap();
        output.close().unwrap();

        let mut input = storage.open_input("data.bin").unwrap();
        // Initial slice covers the whole file.
        assert_eq!(input.as_slice(), Some(&b"abcdefghij"[..]));

        // After advancing the read cursor by 3 bytes the slice tail
        // matches.
        let mut head = [0u8; 3];
        input.read_exact(&mut head).unwrap();
        assert_eq!(&head, b"abc");
        assert_eq!(input.as_slice(), Some(&b"defghij"[..]));

        // Seek to end → empty slice.
        input.seek(SeekFrom::End(0)).unwrap();
        assert_eq!(input.as_slice(), Some(&[][..]));
    }

    #[test]
    fn test_file_operations() {
        let storage = MemoryStorage::default();

        // File doesn't exist initially
        assert!(!storage.file_exists("nonexistent.txt"));

        // Create a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"Test content").unwrap();
        output.close().unwrap();

        // File exists now
        assert!(storage.file_exists("test.txt"));

        // Check file size
        assert_eq!(storage.file_size("test.txt").unwrap(), 12);

        // List files
        let files = storage.list_files().unwrap();
        assert_eq!(files, vec!["test.txt"]);

        // Rename file
        storage.rename_file("test.txt", "renamed.txt").unwrap();
        assert!(!storage.file_exists("test.txt"));
        assert!(storage.file_exists("renamed.txt"));

        // Delete file
        storage.delete_file("renamed.txt").unwrap();
        assert!(!storage.file_exists("renamed.txt"));
        assert_eq!(storage.file_count(), 0);
    }

    #[test]
    fn test_multiple_files() {
        let storage = MemoryStorage::default();

        // Create multiple files
        for i in 0..5 {
            let mut output = storage.create_output(&format!("file_{i}.txt")).unwrap();
            output.write_all(format!("Content {i}").as_bytes()).unwrap();
            output.close().unwrap();
        }

        assert_eq!(storage.file_count(), 5);

        let files = storage.list_files().unwrap();
        assert_eq!(files.len(), 5);

        // Check that files are sorted
        for (i, file) in files.iter().enumerate().take(5) {
            assert_eq!(file, &format!("file_{i}.txt"));
        }
    }

    #[test]
    fn test_temp_file_creation() {
        let storage = MemoryStorage::default();

        let (temp_name, mut output) = storage.create_temp_output("test").unwrap();

        assert!(temp_name.starts_with("test_"));
        assert!(temp_name.ends_with(".tmp"));

        output.write_all(b"Temporary content").unwrap();
        output.close().unwrap();

        assert!(storage.file_exists(&temp_name));
        assert_eq!(storage.file_size(&temp_name).unwrap(), 17);
    }

    #[test]
    fn test_input_clone() {
        let storage = MemoryStorage::default();

        // Create a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"Hello, Clone!").unwrap();
        output.close().unwrap();

        // Open input and clone it
        let mut input1 = storage.open_input("test.txt").unwrap();
        let mut input2 = input1.clone_input().unwrap();

        // Read from both inputs
        let mut buffer1 = Vec::new();
        let mut buffer2 = Vec::new();

        input1.read_to_end(&mut buffer1).unwrap();
        input2.read_to_end(&mut buffer2).unwrap();

        assert_eq!(buffer1, b"Hello, Clone!");
        assert_eq!(buffer2, b"Hello, Clone!");
        assert_eq!(buffer1, buffer2);
    }

    #[test]
    fn test_seek_operations() {
        let storage = MemoryStorage::default();

        // Create a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"0123456789").unwrap();
        output.close().unwrap();

        // Test seeking in input
        let mut input = storage.open_input("test.txt").unwrap();

        // Seek to position 5
        input.seek(SeekFrom::Start(5)).unwrap();
        let mut buffer = [0u8; 3];
        input.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"567");

        // Seek from end
        input.seek(SeekFrom::End(-2)).unwrap();
        let mut buffer = [0u8; 2];
        input.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"89");
    }

    #[test]
    fn test_file_not_found() {
        let storage = MemoryStorage::default();

        let result = storage.open_input("nonexistent.txt");
        assert!(result.is_err());

        let result = storage.file_size("nonexistent.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_storage_close() {
        let mut storage = MemoryStorage::default();

        storage.close().unwrap();
        assert!(storage.closed);

        // Operations should fail after close
        let result = storage.create_output("test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_storage() {
        let storage = MemoryStorage::default();

        // Create some files
        for i in 0..3 {
            let mut output = storage.create_output(&format!("file_{i}.txt")).unwrap();
            output.write_all(b"content").unwrap();
            output.close().unwrap();
        }

        assert_eq!(storage.file_count(), 3);

        // Clear storage
        storage.clear().unwrap();

        assert_eq!(storage.file_count(), 0);
        assert_eq!(storage.total_size(), 0);
    }

    /// Reads a file's full bytes via `open_input`.
    fn read_file(storage: &MemoryStorage, name: &str) -> Vec<u8> {
        let mut input = storage.open_input(name).unwrap();
        let mut buf = Vec::new();
        input.read_to_end(&mut buf).unwrap();
        buf
    }

    #[test]
    fn flush_then_open_input_sees_only_committed() {
        // Issue #812: a written-but-not-yet-flushed tail must stay invisible to
        // readers until the next flush advances the committed length.
        let storage = MemoryStorage::default();
        let mut output = storage.create_output("wal").unwrap();

        output.write_all(b"AAAA").unwrap();
        output.flush_and_sync().unwrap();
        assert_eq!(read_file(&storage, "wal"), b"AAAA");

        // Write more but do NOT flush: readers still see only the committed prefix.
        output.write_all(b"BBBB").unwrap();
        assert_eq!(read_file(&storage, "wal"), b"AAAA");
        assert_eq!(storage.file_size("wal").unwrap(), 4);

        // Flush makes the tail visible.
        output.flush_and_sync().unwrap();
        assert_eq!(read_file(&storage, "wal"), b"AAAABBBB");
        assert_eq!(storage.file_size("wal").unwrap(), 8);
    }

    #[test]
    fn repeated_flush_is_amortized_and_correct() {
        // The WAL append-then-flush pattern: one long-lived appender, a flush per
        // record. Each flush must publish exactly the cumulative committed bytes.
        let storage = MemoryStorage::default();
        let mut output = storage.create_output_append("wal").unwrap();

        let mut expected = Vec::new();
        for i in 0u32..64 {
            let record = i.to_le_bytes();
            output.write_all(&record).unwrap();
            output.flush_and_sync().unwrap();
            expected.extend_from_slice(&record);

            assert_eq!(storage.file_size("wal").unwrap(), expected.len() as u64);
        }
        assert_eq!(read_file(&storage, "wal"), expected);
    }

    #[test]
    fn truncate_keeps_old_content_until_republish() {
        // `create_output` (truncate) must not replace the visible map entry until
        // the new writer flushes/closes.
        let storage = MemoryStorage::default();
        let mut old = storage.create_output("f").unwrap();
        old.write_all(b"oldcontent").unwrap();
        old.close().unwrap();
        assert_eq!(read_file(&storage, "f"), b"oldcontent");

        // A fresh truncate handle that has not written/flushed yet leaves the old
        // content visible.
        let mut fresh = storage.create_output("f").unwrap();
        assert_eq!(read_file(&storage, "f"), b"oldcontent");

        // Publishing the new content replaces it.
        fresh.write_all(b"new").unwrap();
        fresh.close().unwrap();
        assert_eq!(read_file(&storage, "f"), b"new");
        assert_eq!(storage.file_size("f").unwrap(), 3);
    }

    #[test]
    fn create_output_close_without_write_publishes_empty_file() {
        // Closing a fresh handle with no writes must materialize a zero-length
        // file (the v3 WAL re-stamp path in store/log.rs relies on this).
        let storage = MemoryStorage::default();
        let mut output = storage.create_output("empty.log").unwrap();
        output.close().unwrap();

        assert!(storage.file_exists("empty.log"));
        assert_eq!(storage.file_size("empty.log").unwrap(), 0);
        assert_eq!(read_file(&storage, "empty.log"), b"");
    }

    #[test]
    fn delete_then_writer_flush_resurrects_file() {
        // Deleting a file out from under an open writer, then flushing, must
        // resurrect it with the writer's full content (last-writer-wins).
        let storage = MemoryStorage::default();
        let mut output = storage.create_output_append("wal").unwrap();

        output.write_all(b"AAAA").unwrap();
        output.flush_and_sync().unwrap();
        assert!(storage.file_exists("wal"));

        storage.delete_file("wal").unwrap();
        assert!(!storage.file_exists("wal"));

        output.write_all(b"BBBB").unwrap();
        output.flush_and_sync().unwrap();
        assert!(storage.file_exists("wal"));
        assert_eq!(read_file(&storage, "wal"), b"AAAABBBB");
    }

    #[test]
    fn seek_back_overwrite_header_then_commit() {
        // The placeholder-header pattern (e.g. BKD tree writer): reserve a header,
        // write the payload, seek back to fill the header in place, then commit.
        let storage = MemoryStorage::default();
        let mut output = storage.create_output("bkd").unwrap();

        // Reserve a 4-byte header placeholder, then append the payload.
        output.write_all(&[0u8; 4]).unwrap();
        output.write_all(b"PAYLOAD").unwrap();

        // Seek back and overwrite the header in place.
        output.seek(SeekFrom::Start(0)).unwrap();
        output.write_all(b"HEAD").unwrap();

        // Seek to end and commit.
        output.seek(SeekFrom::End(0)).unwrap();
        output.close().unwrap();

        assert_eq!(read_file(&storage, "bkd"), b"HEADPAYLOAD");
        assert_eq!(storage.file_size("bkd").unwrap(), 11);
    }

    #[test]
    fn size_surfaces_report_committed_not_written_tail() {
        // file_size / total_size / metadata must report the committed (flushed)
        // length, never the larger in-progress buffer length — the #791/#806
        // allocation guard depends on `size()` being the true visible length.
        let storage = MemoryStorage::default();
        let mut output = storage.create_output("f").unwrap();

        output.write_all(b"AAAA").unwrap();
        output.flush_and_sync().unwrap();

        // Write an unflushed tail.
        output.write_all(b"BBBBBB").unwrap();

        assert_eq!(storage.file_size("f").unwrap(), 4);
        assert_eq!(storage.total_size(), 4);
        assert_eq!(storage.metadata("f").unwrap().size, 4);

        // The open_input snapshot's size also reflects only the committed prefix.
        assert_eq!(storage.open_input("f").unwrap().size().unwrap(), 4);
    }
}
