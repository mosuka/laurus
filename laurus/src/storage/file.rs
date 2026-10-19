//! File-based storage implementation.
//!
//! This module provides disk-based persistent storage with support for both
//! traditional file I/O and memory-mapped files (mmap).
//!
//! # Features
//!
//! - **Traditional I/O**: Buffered reads/writes with configurable buffer size
//! - **Memory-mapped I/O**: High-performance reads using mmap with caching
//! - **File locking**: Concurrent access control
//! - **Flexible configuration**: Buffer size, sync writes, temp directory, etc.
//!
//! # Memory-Mapped Mode
//!
//! When `FileStorageConfig.use_mmap` is enabled:
//! - Files are mapped into memory for reading
//! - Mapped files are cached for reuse
//! - File modifications are detected and cache is invalidated
//! - Supports prefaulting and huge pages for performance
//!
//! # Example
//!
//! ```
//! use laurus::storage::file::{FileStorage, FileStorageConfig};
//! use laurus::storage::Storage;
//! use std::io::Write;
//! use tempfile::TempDir;
//!
//! # fn main() -> laurus::Result<()> {
//! // Create storage with mmap enabled
//! let temp_dir = TempDir::new().unwrap();
//! let mut config = FileStorageConfig::new(temp_dir.path());
//! config.use_mmap = true;
//! let storage = FileStorage::new(temp_dir.path(), config)?;
//!
//! // Write a file
//! let mut output = storage.create_output("test.dat")?;
//! output.write_all(b"Hello, world!")?;
//! output.close()?;
//!
//! // Read using mmap
//! let mut input = storage.open_input("test.dat")?;
//! let mut buffer = Vec::new();
//! input.read_to_end(&mut buffer)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::SystemTime;

use memmap2::{Mmap, MmapOptions};

use crate::error::{LaurusError, Result};
use crate::storage::{
    LockManager, Storage, StorageError, StorageInput, StorageLock, StorageOutput,
};

/// Configuration specific to file-based storage.
///
/// This configuration includes the storage path and various options for
/// file I/O, memory-mapping, and locking behavior.
///
/// # Memory-Mapped Files (mmap)
///
/// When `use_mmap` is enabled, FileStorage uses memory-mapped I/O for reading files,
/// which can significantly improve performance for large files by:
/// - Avoiding system call overhead
/// - Leveraging the OS page cache
/// - Enabling zero-copy reads
///
/// Additional mmap options:
/// - `mmap_cache_size`: Number of mmap files to keep cached
/// - `mmap_enable_prefault`: Pre-populate page tables for faster initial access
/// - `mmap_enable_hugepages`: Use huge pages if available (Linux)
///
/// # Example
///
/// ```
/// use laurus::storage::file::FileStorageConfig;
///
/// // Basic file storage
/// let config = FileStorageConfig::new("/data/index");
///
/// // High-performance configuration with mmap
/// let mut config = FileStorageConfig::new("/data/index");
/// config.use_mmap = true;
/// config.mmap_enable_prefault = true;
/// config.buffer_size = 131072; // 128KB for non-mmap operations
/// ```
#[derive(Debug, Clone)]
pub struct FileStorageConfig {
    /// Path to the storage directory.
    pub path: std::path::PathBuf,

    /// Whether to use memory-mapped files for reading.
    /// When true, files are read using mmap instead of traditional
    /// I/O. **Default `true` on every platform** as of Issue #504
    /// (Linux/macOS) and Issue #508 (Windows); set the
    /// `LAURUS_NO_MMAP=1` environment variable when constructing a
    /// `FileStorageConfig` via [`Self::new`] to opt out.
    pub use_mmap: bool,

    /// Buffer size for traditional I/O operations (bytes).
    /// Default: 65536 (64KB). Used when `use_mmap` is false.
    pub buffer_size: usize,

    /// Whether to sync writes immediately to disk.
    /// When true, calls fsync after each write for durability.
    pub sync_writes: bool,

    /// Whether to use file locking for concurrency control.
    pub use_locking: bool,

    /// Temporary directory for temp files.
    /// If None, uses the storage directory.
    pub temp_dir: Option<String>,

    /// Maximum number of memory-mapped files to cache.
    /// Only used when `use_mmap` is true. Default: 100.
    pub mmap_cache_size: usize,

    /// Enable prefaulting for memory-mapped files.
    /// Pre-populates page tables for faster initial access.
    /// Only used when `use_mmap` is true.
    pub mmap_enable_prefault: bool,

    /// Enable huge pages for memory-mapped files if available.
    /// Can improve TLB performance for large files (Linux only).
    /// Only used when `use_mmap` is true.
    ///
    /// **Note**: This field is currently a placeholder and is not yet used
    /// by the `get_mmap()` implementation. Setting it has no effect at this time.
    pub mmap_enable_hugepages: bool,
}

impl FileStorageConfig {
    /// Create a new FileStorageConfig with the given path and default settings.
    ///
    /// # Default Settings
    ///
    /// - `use_mmap`: **`true` on every platform** (Linux / macOS /
    ///   other Unix per Issue #504; Windows per Issue #508 once the
    ///   cache eviction in [`FileStorage::create_output`] /
    ///   [`FileStorage::delete_file`] landed). mmap-backed reads are the
    ///   default so the lexical posting decoder can take the
    ///   zero-copy path through `StorageInput::as_slice`. Set the
    ///   `LAURUS_NO_MMAP=1` environment variable to opt out (debug /
    ///   fallback for hosts where mmap misbehaves).
    /// - `buffer_size`: 65536 (64KB)
    /// - `sync_writes`: false
    /// - `use_locking`: true
    /// - `mmap_cache_size`: 100
    /// - `mmap_enable_prefault`: false
    /// - `mmap_enable_hugepages`: false
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Self {
        // mmap is on unless `LAURUS_NO_MMAP=1` on every platform
        // (Issue #504, #508); the per-OS policy lives in
        // `super::platform` so this call site stays platform-agnostic.
        FileStorageConfig {
            path: path.as_ref().to_path_buf(),
            use_mmap: super::platform::default_use_mmap(),
            buffer_size: 65536,
            sync_writes: false,
            use_locking: true,
            temp_dir: None,
            mmap_cache_size: 100,
            mmap_enable_prefault: false,
            mmap_enable_hugepages: false,
        }
    }
}

/// Metadata information for cached files.
#[derive(Debug, Clone)]
struct MmapFileMetadata {
    size: u64,
    modified: u64,
}

/// A file-based storage implementation.
///
/// FileStorage provides persistent disk-based storage with two read modes:
///
/// 1. **Traditional I/O** (default): Uses buffered file reads with `BufReader`
/// 2. **Memory-mapped I/O**: Uses mmap for zero-copy reads when `config.use_mmap` is true
///
/// The mmap mode includes caching and automatic invalidation on file changes,
/// making it suitable for read-heavy workloads with large files.
#[derive(Debug)]
pub struct FileStorage {
    /// The root directory for storage.
    directory: PathBuf,
    /// Storage configuration.
    config: FileStorageConfig,
    /// Lock manager for coordinating access.
    lock_manager: Arc<FileLockManager>,
    /// Whether the storage is closed.
    closed: bool,
    /// Cache of memory-mapped files (only used when use_mmap is true).
    mmap_cache: Arc<RwLock<HashMap<String, Arc<Mmap>>>>,
    /// Cache of file metadata for mmap files.
    mmap_metadata_cache: Arc<RwLock<HashMap<String, MmapFileMetadata>>>,
}

impl FileStorage {
    /// Create a new file storage in the given directory.
    pub fn new<P: AsRef<Path>>(directory: P, config: FileStorageConfig) -> Result<Self> {
        let directory = directory.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !directory.exists() {
            std::fs::create_dir_all(&directory)
                .map_err(|e| LaurusError::storage(format!("Failed to create directory: {e}")))?;
        }

        // Verify it's a directory
        if !directory.is_dir() {
            return Err(LaurusError::storage(format!(
                "Path is not a directory: {}",
                directory.display()
            )));
        }

        let lock_manager = Arc::new(FileLockManager::new(directory.clone()));

        Ok(FileStorage {
            directory,
            config,
            lock_manager,
            closed: false,
            mmap_cache: Arc::new(RwLock::new(HashMap::new())),
            mmap_metadata_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Get the full path for a file name.
    fn file_path(&self, name: &str) -> PathBuf {
        self.directory.join(name)
    }

    /// Check if the storage is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed {
            Err(StorageError::StorageClosed.into())
        } else {
            Ok(())
        }
    }

    /// Recursively fsync a directory and its subdirectories.
    ///
    /// On Unix, opening a directory and calling `sync_all()` ensures that
    /// directory entries (file creation, rename, delete) are flushed to disk.
    ///
    /// On Windows, directories cannot be opened as regular files, so directory
    /// fsync is not applicable. Instead, the file-level `sync_all()` in
    /// `FileOutput::close()` ensures data durability, and the explicit handle
    /// release ensures file visibility for subsequent reads.
    fn sync_directory_recursive(dir: &Path) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        // Fsync subdirectories first (depth-first).
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    Self::sync_directory_recursive(&entry.path())?;
                }
            }
        }

        // Fsync the directory itself (Unix only).
        // Windows does not support opening directories as files for fsync.
        #[cfg(unix)]
        {
            let dir_file = File::open(dir).map_err(|e| {
                LaurusError::storage(format!(
                    "Failed to open directory {:?} for sync: {}",
                    dir, e
                ))
            })?;
            dir_file.sync_all().map_err(|e| {
                LaurusError::storage(format!("Failed to sync directory {:?}: {}", dir, e))
            })?;
        }

        Ok(())
    }

    /// Drop any cached memory map (and its metadata entry) for `name`
    /// so a subsequent `truncate(true)` or `remove_file` on the
    /// underlying file can proceed.
    ///
    /// # Why this exists
    ///
    /// On Windows the OS holds an exclusive lock on memory-mapped
    /// files (`ERROR_USER_MAPPED_FILE`, os error 1224). Without this
    /// eviction, [`Self::create_output`] and [`Self::delete_file`]
    /// would fail whenever this `FileStorage` still owns a cached
    /// `Arc<Mmap>` for the target file (Issue #508).
    ///
    /// On Unix the call is a no-op for correctness — the kernel keeps
    /// the inode alive across truncate / unlink — but evicting the
    /// stale entry is still useful so subsequent reads do not hand
    /// out a mapping that no longer matches the file content.
    ///
    /// # Lifetime contract
    ///
    /// Any `Arc<Mmap>` clones previously handed out via
    /// [`Self::open_input`] stay alive through the `Arc` refcount;
    /// this method only drops *the cache's* clone. Callers must
    /// ensure no other code path holds a `StorageInput` (or
    /// `as_slice()` borrow derived from it) for the same `name` when
    /// they invoke a mutation, otherwise the outstanding `Arc<Mmap>`
    /// will keep the Windows file lock alive. Today every laurus
    /// reader consumes its `StorageInput` within a single function
    /// scope, so this contract holds by construction.
    fn evict_mmap(&self, name: &str) {
        self.mmap_cache.write().unwrap().remove(name);
        self.mmap_metadata_cache.write().unwrap().remove(name);
    }

    /// Get or create a memory map for a file.
    fn get_mmap(&self, name: &str) -> Result<Arc<Mmap>> {
        let file_path = self.file_path(name);

        // Check cache first
        {
            let cache = self.mmap_cache.read().unwrap();
            if let Some(mmap) = cache.get(name) {
                // Verify the file hasn't changed
                if self.is_mmap_file_unchanged(name, &file_path)? {
                    return Ok(Arc::clone(mmap));
                }
            }
        }

        // Create new memory map
        let file = File::open(&file_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::FileNotFound(name.to_string())
            } else {
                StorageError::IoError(format!("Failed to open file {name}: {e}"))
            }
        })?;

        let mut mmap_opts = MmapOptions::new();
        if self.config.mmap_enable_prefault {
            mmap_opts.populate();
        }

        let mmap = unsafe {
            mmap_opts
                .map(&file)
                .map_err(|e| LaurusError::storage(format!("Failed to mmap file {name}: {e}")))?
        };

        let mmap_arc = Arc::new(mmap);

        // Update cache
        {
            let mut cache = self.mmap_cache.write().unwrap();
            cache.insert(name.to_string(), Arc::clone(&mmap_arc));
        }

        // Update metadata cache
        self.update_mmap_metadata_cache(name, &file_path)?;

        Ok(mmap_arc)
    }

    /// Check if a memory-mapped file has been modified since last cached.
    fn is_mmap_file_unchanged(&self, name: &str, path: &Path) -> Result<bool> {
        let metadata_cache = self.mmap_metadata_cache.read().unwrap();

        if let Some(cached_meta) = metadata_cache.get(name) {
            let current_meta = std::fs::metadata(path)
                .map_err(|e| LaurusError::storage(format!("Failed to get metadata: {e}")))?;

            let current_size = current_meta.len();
            let current_modified = current_meta
                .modified()
                .unwrap_or(SystemTime::UNIX_EPOCH)
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            return Ok(cached_meta.size == current_size && cached_meta.modified == current_modified);
        }

        Ok(false)
    }

    /// Update metadata cache for a memory-mapped file.
    fn update_mmap_metadata_cache(&self, name: &str, path: &Path) -> Result<()> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| LaurusError::storage(format!("Failed to get metadata: {e}")))?;

        let size = metadata.len();
        let modified = metadata
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut cache = self.mmap_metadata_cache.write().unwrap();
        cache.insert(name.to_string(), MmapFileMetadata { size, modified });

        Ok(())
    }
}

impl Storage for FileStorage {
    fn loading_mode(&self) -> crate::storage::LoadingMode {
        if self.config.use_mmap {
            crate::storage::LoadingMode::Lazy
        } else {
            crate::storage::LoadingMode::Eager
        }
    }

    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.check_closed()?;

        if self.config.use_mmap {
            // Use memory-mapped file
            let mmap = self.get_mmap(name)?;
            Ok(Box::new(MmapInput::new(mmap)))
        } else {
            // Use traditional file I/O
            let path = self.file_path(name);
            let file = File::open(&path).map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    StorageError::FileNotFound(name.to_string())
                } else {
                    StorageError::IoError(e.to_string())
                }
            })?;

            Ok(Box::new(FileInput::new(file, self.config.buffer_size)?))
        }
    }

    /// Create a fresh output for `name`, truncating any existing file.
    ///
    /// # Concurrency
    ///
    /// On Windows the OS holds an exclusive lock on memory-mapped
    /// files. This method evicts the storage's own mmap cache entry
    /// for `name` before the truncate so the file lock can be
    /// released (Issue #508). Any `StorageInput` previously returned
    /// by [`Self::open_input`] for the same name must already be
    /// dropped before another code path invokes this method —
    /// otherwise the outstanding `Arc<Mmap>` clone will keep the
    /// Windows lock alive and the OS returns `ERROR_USER_MAPPED_FILE`.
    /// See [`Self::evict_mmap`] for the full lifetime contract.
    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.check_closed()?;

        // Release the Windows file lock before opening with truncate.
        // No-op on Unix; correctness-critical on Windows (Issue #508).
        self.evict_mmap(name);

        let path = self.file_path(name);

        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                LaurusError::storage(format!("Failed to create directory {:?}: {}", parent, e))
            })?;
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| StorageError::IoError(e.to_string()))?;

        Ok(Box::new(FileOutput::new(
            file,
            self.config.buffer_size,
            self.config.sync_writes,
        )?))
    }

    /// Open `name` for append, creating it if necessary.
    ///
    /// # Concurrency
    ///
    /// Append-mode targets (WAL, log files) are not memory-mapped in
    /// the current codebase, so the cache eviction below is purely
    /// defensive. If a future change starts mmap'ing one of these
    /// files, the Windows lifecycle contract documented on
    /// [`Self::evict_mmap`] already applies (Issue #508).
    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.check_closed()?;

        // Defensive: keep symmetry with `create_output` / `delete_file`
        // so a future caller cannot reintroduce the Windows lock bug
        // by mmap'ing an append-mode target.
        self.evict_mmap(name);

        let path = self.file_path(name);

        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent).map_err(|e| {
                LaurusError::storage(format!("Failed to create directory {:?}: {}", parent, e))
            })?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| StorageError::IoError(e.to_string()))?;

        Ok(Box::new(FileOutput::new(
            file,
            self.config.buffer_size,
            self.config.sync_writes,
        )?))
    }

    fn file_exists(&self, name: &str) -> bool {
        if self.closed {
            return false;
        }

        self.file_path(name).exists()
    }

    /// Delete `name` from the storage directory.
    ///
    /// # Concurrency
    ///
    /// On Windows, `remove_file` returns `ERROR_USER_MAPPED_FILE` if
    /// any process still holds an mmap for the file. This method
    /// evicts the storage's own cached `Arc<Mmap>` first to release
    /// the lock (Issue #508). Outstanding `StorageInput` clones held
    /// elsewhere keep the lock alive until they drop; the same
    /// lifetime contract documented on [`Self::evict_mmap`] applies.
    fn delete_file(&self, name: &str) -> Result<()> {
        self.check_closed()?;

        // Release the Windows file lock before unlinking. No-op on
        // Unix; correctness-critical on Windows (Issue #508).
        self.evict_mmap(name);

        let path = self.file_path(name);
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| StorageError::IoError(format!("Failed to delete file: {e}")))?;
        }

        Ok(())
    }

    fn list_files(&self) -> Result<Vec<String>> {
        self.check_closed()?;

        let mut files = Vec::new();
        let mut queue = vec![self.directory.clone()];

        while let Some(dir) = queue.pop() {
            // Skip if error reading directory
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };

            for entry in entries {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let path = entry.path();

                if path.is_dir() {
                    queue.push(path);
                } else if path.is_file()
                    && let Ok(rel) = path.strip_prefix(&self.directory)
                    && let Some(name) = rel.to_str()
                {
                    // Normalize path separators to forward slashes for
                    // cross-platform consistency. PrefixedStorage and other
                    // consumers expect '/' separators regardless of OS.
                    files.push(name.replace('\\', "/"));
                }
            }
        }

        files.sort();
        Ok(files)
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.check_closed()?;

        let path = self.file_path(name);
        let metadata = path.metadata().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::FileNotFound(name.to_string())
            } else {
                StorageError::IoError(e.to_string())
            }
        })?;

        Ok(metadata.len())
    }

    fn metadata(&self, name: &str) -> Result<crate::storage::FileMetadata> {
        self.check_closed()?;

        let path = self.file_path(name);
        let metadata = path.metadata().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                StorageError::FileNotFound(name.to_string())
            } else {
                StorageError::IoError(e.to_string())
            }
        })?;

        let modified = metadata
            .modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let created = metadata
            .created()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(crate::storage::FileMetadata {
            size: metadata.len(),
            modified,
            created,
            readonly: metadata.permissions().readonly(),
        })
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.check_closed()?;

        // Evict both names' mappings (#1031). The destination's cached bytes
        // are pre-rename content that `is_mmap_file_unchanged`'s size +
        // whole-second-mtime probe cannot tell apart from the new file, so
        // without this every later `open_input` serves stale data. The
        // source's mapping would point at a path that no longer exists. On
        // Windows this also releases the file locks before the rename,
        // mirroring `create_output` and `delete_file` (Issue #508).
        self.evict_mmap(old_name);
        self.evict_mmap(new_name);

        let old_path = self.file_path(old_name);
        let new_path = self.file_path(new_name);

        std::fs::rename(&old_path, &new_path)
            .map_err(|e| StorageError::IoError(format!("Failed to rename file: {e}")))?;

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

        // Recursively fsync directories to ensure that all file metadata
        // (creation, rename, size changes) is visible to subsequent readers.
        // This is essential on Windows where directory listings may be cached.
        Self::sync_directory_recursive(&self.directory)?;

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

/// A file input implementation.
#[derive(Debug)]
pub struct FileInput {
    reader: BufReader<File>,
    size: u64,
}

impl FileInput {
    fn new(file: File, buffer_size: usize) -> Result<Self> {
        let metadata = file
            .metadata()
            .map_err(|e| LaurusError::storage(format!("Failed to get file metadata: {e}")))?;

        let size = metadata.len();
        let reader = BufReader::with_capacity(buffer_size, file);

        Ok(FileInput { reader, size })
    }
}

impl Read for FileInput {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.reader.read(buf)
    }
}

impl Seek for FileInput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.reader.seek(pos)
    }
}

impl StorageInput for FileInput {
    fn size(&self) -> Result<u64> {
        Ok(self.size)
    }

    fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
        // For file inputs, we can't easily clone the underlying file
        // This would require reopening the file, which we'll implement later
        Err(LaurusError::storage("Clone not supported for file inputs"))
    }

    fn close(&mut self) -> Result<()> {
        // BufReader doesn't have an explicit close method
        // The file will be closed when the BufReader is dropped
        Ok(())
    }
}

/// A memory-mapped file input implementation.
#[derive(Debug)]
pub struct MmapInput {
    mmap: Arc<Mmap>,
    #[allow(dead_code)]
    cursor: Cursor<Vec<u8>>,
    position: u64,
}

impl MmapInput {
    fn new(mmap: Arc<Mmap>) -> Self {
        MmapInput {
            mmap,
            cursor: Cursor::new(Vec::new()),
            position: 0,
        }
    }
}

impl Read for MmapInput {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let available = (self.mmap.len() as u64).saturating_sub(self.position) as usize;
        let to_read = buf.len().min(available);

        if to_read == 0 {
            return Ok(0);
        }

        let start = self.position as usize;
        let end = start + to_read;
        buf[..to_read].copy_from_slice(&self.mmap[start..end]);
        self.position += to_read as u64;

        Ok(to_read)
    }
}

impl Seek for MmapInput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(offset) => self.mmap.len() as i64 + offset,
            SeekFrom::Current(offset) => self.position as i64 + offset,
        };

        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid seek to a negative position",
            ));
        }

        self.position = new_pos as u64;
        Ok(self.position)
    }
}

impl StorageInput for MmapInput {
    fn size(&self) -> Result<u64> {
        Ok(self.mmap.len() as u64)
    }

    fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
        Ok(Box::new(MmapInput {
            mmap: Arc::clone(&self.mmap),
            cursor: Cursor::new(Vec::new()),
            position: 0,
        }))
    }

    fn close(&mut self) -> Result<()> {
        // Memory map will be automatically unmapped when dropped
        Ok(())
    }

    /// Borrow the mmap region from the current read position to the end
    /// of the file. The slice is valid for the lifetime of `&self`; the
    /// underlying `Arc<Mmap>` keeps the page mapping alive as long as
    /// this `MmapInput` exists. Callers can advance through `seek`
    /// independently of any slices they continue to hold.
    fn as_slice(&self) -> Option<&[u8]> {
        let start = (self.position as usize).min(self.mmap.len());
        Some(&self.mmap[start..])
    }
}

/// A file output implementation.
///
/// Uses `Option<BufWriter<File>>` so that `close()` can explicitly release
/// the file handle via `take()` + `into_inner()`. This is critical on Windows
/// where file handles must be fully released before other processes (or the
/// same process) can read or delete the file.
#[derive(Debug)]
pub struct FileOutput {
    writer: Option<BufWriter<File>>,
    sync_writes: bool,
    position: u64,
    /// Whether bytes have been written since the last successful
    /// `sync_all()` (via [`Self::flush_and_sync`] or [`Self::close`]).
    ///
    /// Starts `true` so a handle that is closed without ever calling
    /// `flush_and_sync` still syncs on close, matching the historical
    /// behavior. Set by [`StorageOutput::flush_and_sync`]'s write path;
    /// cleared once that sync succeeds — lets [`StorageOutput::close`] skip a
    /// redundant `sync_all()` when `flush_and_sync` already made everything
    /// durable and nothing was written afterward (Issue #877).
    dirty: bool,
    /// Test-only: number of real `sync_all()` (fsync) calls issued by this
    /// handle. Lets tests assert the exact fsync count instead of inferring it
    /// from timing or `strace`.
    #[cfg(test)]
    sync_count: usize,
}

impl FileOutput {
    fn new(file: File, buffer_size: usize, sync_writes: bool) -> Result<Self> {
        let writer = BufWriter::with_capacity(buffer_size, file);

        Ok(FileOutput {
            writer: Some(writer),
            sync_writes,
            position: 0,
            dirty: true,
            #[cfg(test)]
            sync_count: 0,
        })
    }
}

impl Write for FileOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| std::io::Error::other("FileOutput already closed"))?;
        let bytes_written = writer.write(buf)?;
        self.position += bytes_written as u64;
        // Any successful write, buffered or not, means the file is no longer
        // fully represented by the last `sync_all()` (Issue #877).
        self.dirty = true;

        if self.sync_writes {
            // Re-borrow since position assignment ended the previous borrow.
            self.writer.as_mut().unwrap().flush()?;
        }

        Ok(bytes_written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer
            .as_mut()
            .ok_or_else(|| std::io::Error::other("FileOutput already closed"))?
            .flush()
    }
}

impl Seek for FileOutput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = self
            .writer
            .as_mut()
            .ok_or_else(|| std::io::Error::other("FileOutput already closed"))?
            .seek(pos)?;
        self.position = new_pos;
        Ok(new_pos)
    }
}

impl StorageOutput for FileOutput {
    fn flush_and_sync(&mut self) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| LaurusError::storage("FileOutput already closed".to_string()))?;

        writer
            .flush()
            .map_err(|e| LaurusError::storage(format!("Failed to flush: {e}")))?;

        writer
            .get_ref()
            .sync_all()
            .map_err(|e| LaurusError::storage(format!("Failed to sync: {e}")))?;
        self.dirty = false;
        #[cfg(test)]
        {
            self.sync_count += 1;
        }

        Ok(())
    }

    fn position(&self) -> Result<u64> {
        Ok(self.position)
    }

    fn close(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.take() {
            // Flush buffered data and get the underlying File handle. Needed
            // regardless of `dirty`: if nothing was written since the last
            // `flush_and_sync`, the buffer is already empty and this is a
            // no-op, but the handle must still be extracted to sync/drop it.
            let file = writer
                .into_inner()
                .map_err(|e| LaurusError::storage(format!("Failed to flush on close: {e}")))?;

            // Sync all data and metadata to disk — unless `flush_and_sync`
            // already made everything durable and nothing was written since
            // (Issue #877: avoids a second redundant `fsync` per file for
            // every caller that flushes before closing, e.g. `StructWriter`).
            if self.dirty {
                file.sync_all()
                    .map_err(|e| LaurusError::storage(format!("Failed to sync on close: {e}")))?;
                #[cfg(test)]
                {
                    self.sync_count += 1;
                }
            }

            // `file` is dropped here, explicitly releasing the OS file handle.
        }
        Ok(())
    }
}

/// A file-based lock manager.
#[derive(Debug)]
pub struct FileLockManager {
    directory: PathBuf,
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<FileLock>>>>>,
}

impl FileLockManager {
    fn new(directory: PathBuf) -> Self {
        FileLockManager {
            directory,
            locks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn lock_path(&self, name: &str) -> PathBuf {
        self.directory.join(format!("{name}.lock"))
    }
}

impl LockManager for FileLockManager {
    fn acquire_lock(&self, name: &str) -> Result<Box<dyn StorageLock>> {
        let lock_path = self.lock_path(name);

        // Try to create the lock file
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    StorageError::LockFailed(name.to_string())
                } else {
                    StorageError::IoError(e.to_string())
                }
            })?;

        let lock = Arc::new(Mutex::new(FileLock::new(name.to_string(), lock_path, file)));

        // Store the lock
        {
            let mut locks = self.locks.lock().unwrap();
            locks.insert(name.to_string(), lock.clone());
        }

        Ok(Box::new(FileLockWrapper {
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

        for (_, lock) in locks.drain() {
            let mut file_lock = lock.lock().unwrap();
            file_lock.release()?;
        }

        Ok(())
    }
}

/// A file-based lock implementation.
#[derive(Debug)]
struct FileLock {
    #[allow(dead_code)]
    name: String,
    path: PathBuf,
    _file: File,
    released: bool,
}

impl FileLock {
    fn new(name: String, path: PathBuf, file: File) -> Self {
        FileLock {
            name,
            path,
            _file: file,
            released: false,
        }
    }

    fn release(&mut self) -> Result<()> {
        if !self.released {
            std::fs::remove_file(&self.path)
                .map_err(|e| LaurusError::storage(format!("Failed to release lock: {e}")))?;
            self.released = true;
        }
        Ok(())
    }
}

/// A wrapper around FileLock that implements StorageLock.
#[derive(Debug)]
struct FileLockWrapper {
    lock: Arc<Mutex<FileLock>>,
    /// This lock's name, and a handle to the owning [`FileLockManager`]'s
    /// `locks` map, so `release()` can remove its own entry (Issue #1086)
    /// instead of leaving [`LockManager::lock_exists`] reporting `true`
    /// for a lock nothing holds any more.
    name: String,
    manager_locks: Arc<Mutex<HashMap<String, Arc<Mutex<FileLock>>>>>,
}

impl StorageLock for FileLockWrapper {
    /// Returns a fixed identifier `"file_lock"` rather than the actual lock name.
    ///
    /// Because the real name is stored behind a `Mutex`, returning a borrowed
    /// reference to it is not possible with the current `&str` return type.
    fn name(&self) -> &str {
        "file_lock"
    }

    fn release(&mut self) -> Result<()> {
        let result = {
            let mut lock = self.lock.lock().unwrap();
            lock.release()
        };
        // Drop the manager's bookkeeping entry regardless of whether the
        // filesystem removal succeeded -- once the caller has asked to
        // release, `lock_exists` must stop claiming this name is held.
        self.manager_locks.lock().unwrap().remove(&self.name);
        result
    }

    fn is_valid(&self) -> bool {
        let lock = self.lock.lock().unwrap();
        !lock.released
    }
}

impl Drop for FileLockWrapper {
    /// Releases the lock automatically once every handle holding it is
    /// dropped (Issue #1086), e.g. when the `Engine` field storing it goes
    /// out of scope. Mirrors `MemoryOutput`'s "best-effort cleanup on
    /// drop" pattern elsewhere in this module -- a failure here (e.g. the
    /// lock file was already removed out of band) has nowhere to
    /// propagate to.
    fn drop(&mut self) {
        let _ = self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_storage() -> (TempDir, FileStorage) {
        let temp_dir = TempDir::new().unwrap();
        let config = FileStorageConfig::new(temp_dir.path());
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();
        (temp_dir, storage)
    }

    #[test]
    fn test_file_storage_creation() {
        let (_temp_dir, storage) = create_test_storage();
        assert!(!storage.closed);
    }

    /// Issue #1086: `Storage::lock_manager()` must actually be wired to
    /// `FileStorage`'s lock manager, not just return `None` (the trait
    /// default every other implementor keeps).
    #[test]
    fn lock_manager_returns_the_file_lock_manager() {
        let (_temp_dir, storage) = create_test_storage();
        assert!(
            storage.lock_manager().is_some(),
            "FileStorage must override the Storage::lock_manager() default"
        );
    }

    /// Two independent lock managers over the SAME directory (simulating
    /// two `FileStorage` instances -- i.e. two processes, or two `Engine`s
    /// in this one -- opening the same index) must not both succeed in
    /// acquiring the same lock name.
    #[test]
    fn try_acquire_lock_rejects_a_second_holder_over_the_same_directory() {
        let temp_dir = TempDir::new().unwrap();
        let first = FileLockManager::new(temp_dir.path().to_path_buf());
        let second = FileLockManager::new(temp_dir.path().to_path_buf());

        let first_lock = first.try_acquire_lock("engine").unwrap();
        assert!(first_lock.is_some(), "the first holder must succeed");

        let second_lock = second.try_acquire_lock("engine").unwrap();
        assert!(
            second_lock.is_none(),
            "a second holder over the same directory must be rejected, not silently succeed"
        );
    }

    /// Once the first holder's lock is dropped, a fresh acquisition attempt
    /// (as a fresh `FileStorage`/process reopening the same directory
    /// would make) must succeed -- the lock releases itself automatically
    /// (Issue #1086's `Drop for FileLockWrapper`), it isn't stuck until an
    /// explicit `release()` call nobody makes.
    #[test]
    fn dropping_the_lock_lets_a_fresh_acquisition_succeed() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileLockManager::new(temp_dir.path().to_path_buf());

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
    /// `Drop`) must clear the manager's own bookkeeping, not just remove
    /// the on-disk file -- otherwise `lock_exists()` would keep reporting
    /// `true` for a name nothing holds any more.
    #[test]
    fn release_clears_lock_exists() {
        let temp_dir = TempDir::new().unwrap();
        let manager = FileLockManager::new(temp_dir.path().to_path_buf());

        let mut lock = manager.acquire_lock("engine").unwrap();
        assert!(manager.lock_exists("engine"));

        lock.release().unwrap();
        assert!(
            !manager.lock_exists("engine"),
            "release() must clear the manager's bookkeeping for this name"
        );
    }

    /// Construct a `FileOutput` directly (bypassing the `Storage` trait) so
    /// tests can read its test-only `sync_count`.
    fn open_file_output(dir: &TempDir, name: &str) -> FileOutput {
        let file = File::create(dir.path().join(name)).unwrap();
        FileOutput::new(file, 8192, false).unwrap()
    }

    /// #877: `flush_and_sync()` followed by `close()` with no intervening
    /// write must fsync exactly once, not twice — `close()` must recognize
    /// `flush_and_sync` already made everything durable.
    #[test]
    fn flush_and_sync_then_close_syncs_once() {
        let temp_dir = TempDir::new().unwrap();
        let mut output = open_file_output(&temp_dir, "a.bin");

        output.write_all(b"hello").unwrap();
        output.flush_and_sync().unwrap();
        assert_eq!(output.sync_count, 1, "flush_and_sync must fsync once");

        output.close().unwrap();
        assert_eq!(
            output.sync_count, 1,
            "close() must skip a redundant fsync when nothing was written \
             since the last flush_and_sync (#877)"
        );
    }

    /// #877: a handle that is closed WITHOUT ever calling `flush_and_sync`
    /// must still fsync on close — the optimization only applies when
    /// `flush_and_sync` already covered everything.
    #[test]
    fn close_without_flush_and_sync_still_syncs() {
        let temp_dir = TempDir::new().unwrap();
        let mut output = open_file_output(&temp_dir, "b.bin");

        output.write_all(b"hello").unwrap();
        output.close().unwrap();

        assert_eq!(
            output.sync_count, 1,
            "close() must still fsync when nothing has synced this data yet"
        );
    }

    /// #877: a write AFTER `flush_and_sync()` re-dirties the handle, so the
    /// following `close()` must fsync again — the skip is only safe when
    /// nothing changed since the last sync.
    #[test]
    fn write_after_flush_and_sync_forces_close_to_sync_again() {
        let temp_dir = TempDir::new().unwrap();
        let mut output = open_file_output(&temp_dir, "c.bin");

        output.write_all(b"first").unwrap();
        output.flush_and_sync().unwrap();
        assert_eq!(output.sync_count, 1);

        output.write_all(b"second").unwrap();
        output.close().unwrap();
        assert_eq!(
            output.sync_count, 2,
            "a write after flush_and_sync must still be synced on close"
        );
    }

    /// #877: an empty handle (never written to) closed with no prior
    /// `flush_and_sync` still fsyncs once, matching historical behavior for
    /// callers that create-then-immediately-close (e.g. recreating an empty
    /// file).
    #[test]
    fn close_with_no_writes_still_syncs_once() {
        let temp_dir = TempDir::new().unwrap();
        let mut output = open_file_output(&temp_dir, "d.bin");
        output.close().unwrap();
        assert_eq!(output.sync_count, 1);
    }

    /// #877: durability is unaffected when `close()` skips its fsync — every
    /// byte written before `close()` is readable back after reopening, on the
    /// path where `flush_and_sync` ran first and `close()` relies on it
    /// (the exact case the skip applies to).
    #[test]
    fn data_survives_close_after_prior_flush_and_sync() {
        let temp_dir = TempDir::new().unwrap();
        let storage =
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap();

        let mut out = storage.create_output("synced_then_closed.bin").unwrap();
        out.write_all(b"synced then closed").unwrap();
        out.flush_and_sync().unwrap();
        out.close().unwrap();

        let mut input = storage.open_input("synced_then_closed.bin").unwrap();
        let mut buf = Vec::new();
        input.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"synced then closed");
    }

    /// #877: durability holds on the other path too — `close()` alone, with
    /// no prior `flush_and_sync`, must still fsync (per
    /// `close_without_flush_and_sync_still_syncs`) and the data must survive.
    #[test]
    fn data_survives_close_without_prior_flush_and_sync() {
        let temp_dir = TempDir::new().unwrap();
        let storage =
            FileStorage::new(temp_dir.path(), FileStorageConfig::new(temp_dir.path())).unwrap();

        let mut out = storage.create_output("closed_only.bin").unwrap();
        out.write_all(b"closed without a prior sync").unwrap();
        out.close().unwrap();

        let mut input = storage.open_input("closed_only.bin").unwrap();
        let mut buf = Vec::new();
        input.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"closed without a prior sync");
    }

    #[test]
    fn mmap_input_as_slice_returns_remaining_bytes() {
        // Issue #504: MmapInput exposes a zero-copy slice into the
        // mapped region from the current read position. Without mmap
        // (default config), the BufReader-backed FileInput returns
        // None via the trait default.
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = true;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();

        let mut output = storage.create_output("data.bin").unwrap();
        output.write_all(b"abcdefghij").unwrap();
        output.close().unwrap();

        let mut input = storage.open_input("data.bin").unwrap();
        assert_eq!(input.as_slice(), Some(&b"abcdefghij"[..]));

        let mut head = [0u8; 3];
        input.read_exact(&mut head).unwrap();
        assert_eq!(&head, b"abc");
        assert_eq!(input.as_slice(), Some(&b"defghij"[..]));

        input.seek(SeekFrom::End(0)).unwrap();
        assert_eq!(input.as_slice(), Some(&[][..]));
    }

    #[test]
    fn buffered_file_input_falls_back_to_none() {
        // When use_mmap is disabled, FileInput is buffered I/O and
        // cannot expose a zero-copy slice. Callers must use the
        // Read+Seek fallback. Issue #504 flipped the default to mmap,
        // so this test now constructs a config that opts out.
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = false;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();
        let mut output = storage.create_output("data.bin").unwrap();
        output.write_all(b"abc").unwrap();
        output.close().unwrap();
        let input = storage.open_input("data.bin").unwrap();
        assert_eq!(input.as_slice(), None);
    }

    #[test]
    fn create_output_releases_mmap_lock() {
        // Issue #508: with `use_mmap=true`, rewriting a file via
        // `create_output` would hit `ERROR_USER_MAPPED_FILE` on
        // Windows if the storage still held a cached `Arc<Mmap>`.
        // `evict_mmap` (called from `create_output`) drops the cache
        // entry before the truncate so the OS releases the lock.
        //
        // On Unix this assertion is a tautology — the kernel allows
        // truncate while mapped. On Windows it is the actual
        // regression gate; without the eviction the second
        // `create_output` would fail with os error 1224.
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = true;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();

        let mut output = storage.create_output("segment.bin").unwrap();
        output.write_all(b"old-content").unwrap();
        output.close().unwrap();

        {
            // Take a read mapping so the cache holds an Arc<Mmap>.
            // Scope ends here, but the cached clone in FileStorage
            // outlives this drop until we call `create_output` again.
            let mut input = storage.open_input("segment.bin").unwrap();
            assert_eq!(input.as_slice(), Some(&b"old-content"[..]));
            let mut buf = Vec::new();
            input.read_to_end(&mut buf).unwrap();
            assert_eq!(buf, b"old-content");
        }

        // Re-write — must succeed on every OS now that
        // `create_output` evicts the cache first.
        let mut output = storage.create_output("segment.bin").unwrap();
        output.write_all(b"new-content").unwrap();
        output.close().unwrap();

        let mut input = storage.open_input("segment.bin").unwrap();
        let mut buf = Vec::new();
        input.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"new-content");
    }

    #[test]
    fn delete_file_releases_mmap_lock() {
        // Sibling of `create_output_releases_mmap_lock` — same
        // mechanism, exercised through the `delete_file` path that
        // segment-manager merge / compaction calls (Issue #508).
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = true;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();

        let mut output = storage.create_output("doomed.bin").unwrap();
        output.write_all(b"will-be-removed").unwrap();
        output.close().unwrap();

        {
            let _input = storage.open_input("doomed.bin").unwrap();
            // _input drops here; FileStorage still holds the cached
            // Arc<Mmap> at this point.
        }

        // Must succeed on Windows after the eviction in
        // `delete_file`; without it, os error 1224 fires here.
        storage.delete_file("doomed.bin").unwrap();
        assert!(!storage.file_exists("doomed.bin"));

        // Re-create to confirm the cache slot is genuinely gone, not
        // just shadowed.
        let mut output = storage.create_output("doomed.bin").unwrap();
        output.write_all(b"fresh").unwrap();
        output.close().unwrap();
        let mut input = storage.open_input("doomed.bin").unwrap();
        let mut buf = Vec::new();
        input.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"fresh");
    }

    #[test]
    fn config_new_defaults_use_mmap_true() {
        // Issue #504 (Unix) and Issue #508 (Windows): mmap is the
        // default on every platform now. The opt-out is the
        // `LAURUS_NO_MMAP=1` env var, applied symmetrically by both
        // `platform/unix.rs` and `platform/windows.rs`.
        let temp_dir = TempDir::new().unwrap();
        let prior = std::env::var("LAURUS_NO_MMAP").ok();
        // SAFETY: this test temporarily mutates a process-global env
        // var. It restores the prior state before returning so other
        // tests in the binary are unaffected. Rust 2024 marks
        // `remove_var`/`set_var` unsafe because they are not
        // synchronised with concurrent threads; cargo test parallel
        // sandboxing means another concurrent test could observe the
        // intermediate state, but in practice the storage tests do
        // not branch on LAURUS_NO_MMAP outside of this test, so the
        // race is harmless.
        unsafe {
            std::env::remove_var("LAURUS_NO_MMAP");
        }
        let cfg = FileStorageConfig::new(temp_dir.path());
        assert!(cfg.use_mmap, "default config must enable mmap");
        unsafe {
            std::env::set_var("LAURUS_NO_MMAP", "1");
        }
        let cfg = FileStorageConfig::new(temp_dir.path());
        assert!(
            !cfg.use_mmap,
            "LAURUS_NO_MMAP=1 must opt out of the mmap default"
        );
        unsafe {
            match prior {
                Some(v) => std::env::set_var("LAURUS_NO_MMAP", v),
                None => std::env::remove_var("LAURUS_NO_MMAP"),
            }
        }
    }

    #[test]
    fn test_create_and_read_file() {
        let (_temp_dir, storage) = create_test_storage();

        // Create a file
        let mut output = storage.create_output("test.txt").unwrap();
        output.write_all(b"Hello, World!").unwrap();
        output.close().unwrap();

        // Read the file
        let mut input = storage.open_input("test.txt").unwrap();
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer).unwrap();

        assert_eq!(buffer, b"Hello, World!");
        assert_eq!(input.size().unwrap(), 13);
    }

    #[test]
    fn test_file_operations() {
        let (_temp_dir, storage) = create_test_storage();

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
    }

    #[test]
    fn test_temp_file_creation() {
        let (_temp_dir, storage) = create_test_storage();

        let (temp_name, mut output) = storage.create_temp_output("test").unwrap();

        assert!(temp_name.starts_with("test_"));
        assert!(temp_name.ends_with(".tmp"));

        output.write_all(b"Temporary content").unwrap();
        output.close().unwrap();

        assert!(storage.file_exists(&temp_name));
        assert_eq!(storage.file_size(&temp_name).unwrap(), 17);
    }

    #[test]
    fn test_file_not_found() {
        let (_temp_dir, storage) = create_test_storage();

        let result = storage.open_input("nonexistent.txt");
        assert!(result.is_err());

        let result = storage.file_size("nonexistent.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_storage_close() {
        let (_temp_dir, mut storage) = create_test_storage();

        storage.close().unwrap();
        assert!(storage.closed);

        // Operations should fail after close
        let result = storage.create_output("test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_storage() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = true;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();

        // Create a file
        let mut output = storage.create_output("test_mmap.txt").unwrap();
        output.write_all(b"Hello, Memory-Mapped World!").unwrap();
        output.close().unwrap();

        // Read the file using mmap
        let mut input = storage.open_input("test_mmap.txt").unwrap();
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer).unwrap();

        assert_eq!(buffer, b"Hello, Memory-Mapped World!");
        assert_eq!(input.size().unwrap(), 27);
    }

    #[test]
    fn test_mmap_cache() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = true;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();

        // Create a file
        let mut output = storage.create_output("cached.txt").unwrap();
        output.write_all(b"Cached content").unwrap();
        output.close().unwrap();

        // Read the file twice to test cache
        let mut input1 = storage.open_input("cached.txt").unwrap();
        let mut buffer1 = Vec::new();
        input1.read_to_end(&mut buffer1).unwrap();

        let mut input2 = storage.open_input("cached.txt").unwrap();
        let mut buffer2 = Vec::new();
        input2.read_to_end(&mut buffer2).unwrap();

        assert_eq!(buffer1, buffer2);
        assert_eq!(buffer1, b"Cached content");

        // Check that cache was used
        let cache = storage.mmap_cache.read().unwrap();
        assert!(cache.contains_key("cached.txt"));
    }

    #[test]
    fn test_mmap_clone_input() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = true;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();

        // Create a file
        let mut output = storage.create_output("clone_test.txt").unwrap();
        output.write_all(b"Clone me!").unwrap();
        output.close().unwrap();

        // Open and clone the input
        let mut input1 = storage.open_input("clone_test.txt").unwrap();
        let input2 = input1.clone_input().unwrap();

        // Read from both
        let mut buffer1 = Vec::new();
        input1.read_to_end(&mut buffer1).unwrap();

        let mut buffer2 = Vec::new();
        let mut input2_mut = input2;
        input2_mut.read_to_end(&mut buffer2).unwrap();

        assert_eq!(buffer1, buffer2);
        assert_eq!(buffer1, b"Clone me!");
    }
    /// #1031: `rename_file` must evict the mmap cache for the destination.
    ///
    /// #1028 switched the `metadata.json` writers to the atomic tmp+rename
    /// path, which removed the eviction `create_output`'s truncate used to
    /// perform. With the old mapping cached and the rewrite keeping the same
    /// byte length and whole-second mtime, `is_mmap_file_unchanged` judges
    /// the cache fresh and every later `open_input` serves pre-rename bytes.
    ///
    /// The mtime is pinned explicitly so the test cannot pass by luck when
    /// the two writes straddle a second boundary.
    #[test]
    fn test_rename_file_evicts_destination_mmap() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = FileStorageConfig::new(temp_dir.path());
        config.use_mmap = true;
        let storage = FileStorage::new(temp_dir.path(), config).unwrap();

        // Write the original content and cache its mapping.
        let mut output = storage.create_output("manifest.json").unwrap();
        output.write_all(b"OLD-CONTENT").unwrap();
        output.close().unwrap();
        let mut buf = Vec::new();
        storage
            .open_input("manifest.json")
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(buf, b"OLD-CONTENT");

        let path = temp_dir.path().join("manifest.json");
        let cached_mtime = std::fs::metadata(&path).unwrap().modified().unwrap();

        // Rewrite through the atomic tmp+rename path with the SAME length.
        let mut output = storage.create_output("manifest.json.tmp").unwrap();
        output.write_all(b"NEW-CONTENT").unwrap();
        output.close().unwrap();
        storage
            .rename_file("manifest.json.tmp", "manifest.json")
            .unwrap();

        // Pin the mtime to the cached mapping's, exactly reproducing a
        // same-length same-second rewrite.
        let file = File::options().write(true).open(&path).unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(cached_mtime))
            .unwrap();
        drop(file);

        let mut buf = Vec::new();
        storage
            .open_input("manifest.json")
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(
            buf, b"NEW-CONTENT",
            "open_input after rename_file must see the renamed content, not the stale mapping"
        );
    }
}
