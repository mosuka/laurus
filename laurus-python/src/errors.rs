//! Error conversion between Laurus errors and Python exceptions.

use std::io;
use std::path::Path;

use laurus::LaurusError;
use pyo3::PyErr;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};

/// Error returned by any `Index` method called after
/// [`crate::index::PyIndex::close`] (Issue #1097).
pub fn closed_err() -> PyErr {
    PyRuntimeError::new_err("Index is closed")
}

/// Error returned by [`crate::index::PyIndex::reload`] when called on an
/// in-memory index (Issue #1089): there is no directory to reopen against.
pub fn reload_requires_path_err() -> PyErr {
    PyValueError::new_err(
        "reload() requires a file-backed index (this Index was created without a path)",
    )
}

/// Convert a [`LaurusError`] into a Python exception.
pub fn laurus_err(err: LaurusError) -> PyErr {
    match err {
        LaurusError::Io(e) => PyIOError::new_err(e.to_string()),
        LaurusError::Schema(m) => PyValueError::new_err(format!("Schema error: {m}")),
        LaurusError::Query(m) => PyValueError::new_err(format!("Query error: {m}")),
        LaurusError::Field(m) => PyValueError::new_err(format!("Field error: {m}")),
        LaurusError::InvalidArgument(m) => PyValueError::new_err(format!("Invalid argument: {m}")),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Wrap a filesystem I/O error with the path that caused it, then convert
/// it via PyO3's built-in `io::Error` -> Python exception mapping (which
/// picks `FileNotFoundError`/`PermissionError`/etc. based on `e.kind()`).
///
/// Intentionally does NOT go through [`laurus_err`]: `LaurusError::Io`
/// always maps to the generic `OSError`, which would prevent callers from
/// writing `except FileNotFoundError:` for a missing file.
pub fn io_err_with_path(path: &Path, e: io::Error) -> PyErr {
    io::Error::new(e.kind(), format!("{}: {e}", path.display())).into()
}

/// Convert a [`laurus::index_dir::IndexDirError`] into a Python exception.
///
/// `SchemaConflict`/`LegacyFlatLayout` are both caller-fixable misuse, so
/// they become `ValueError` (matching how [`laurus_err`] treats
/// `LaurusError::Schema`/`InvalidArgument`). `Io` goes through [`io_err_with_path`] rather
/// than `laurus_err` for the same reason `laurus_err` isn't used for plain
/// I/O elsewhere in this crate: it preserves `FileNotFoundError` etc.
/// instead of flattening to a generic `OSError`.
pub fn index_dir_err(err: laurus::index_dir::IndexDirError) -> PyErr {
    use laurus::index_dir::IndexDirError;
    match err {
        IndexDirError::SchemaConflict { .. }
        | IndexDirError::LegacyFlatLayout { .. }
        | IndexDirError::NotAnIndexDirectory { .. } => PyValueError::new_err(err.to_string()),
        IndexDirError::Io { path, source } => io_err_with_path(&path, source),
        IndexDirError::Core(e) => laurus_err(e),
    }
}
