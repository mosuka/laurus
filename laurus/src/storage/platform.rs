//! Storage-layer platform abstractions (Issue #504, #508).
//!
//! Centralises every `cfg(target_os)` / `cfg(unix)` switch used by
//! the file storage backend so the rest of `storage/` can stay
//! platform-agnostic. New per-OS knobs go here, not inline in the
//! call site.
//!
//! # Layout
//!
//! - [`unix`] — Unix-family impl (Linux, macOS, BSDs).
//! - [`windows`] — Windows impl.
//! - The dispatcher re-exports each function from the matching
//!   submodule via a `#[cfg]` gate, so callers see a single uniform
//!   API regardless of host OS.

#[cfg(unix)]
pub mod unix;
#[cfg(windows)]
pub mod windows;

#[cfg(unix)]
pub use unix::{
    copy_permissions, default_use_mmap, is_lock_file_current, open_for_sync, open_lock_file,
    remove_lock_file, sync_dir,
};
#[cfg(windows)]
pub use windows::{
    copy_permissions, default_use_mmap, is_lock_file_current, open_for_sync, open_lock_file,
    remove_lock_file, sync_dir,
};
