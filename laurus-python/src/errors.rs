//! Error conversion between Laurus errors and Python exceptions.

use std::io;
use std::path::Path;

use laurus::LaurusError;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::{PyErr, Python};

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
///
/// The exception is chosen by [`exception_for`] and carries [`message`].
pub fn laurus_err(err: LaurusError) -> PyErr {
    exception_for(&err)(message(&err))
}

/// Choose the Python exception a [`LaurusError`] raises.
///
/// - `OSError` for `LaurusError::Io`.
/// - `ValueError` for a caller's mistake: `LaurusError::Schema`, `Query`,
///   `Field` and `InvalidArgument`.
/// - `RuntimeError` for anything else.
///
/// A failed batch (`LaurusError::BatchIngest` and `BatchSearch`) raises the
/// exception of its source, so a batch that failed on a caller's mistake
/// raises `ValueError` as the same call on its own would (Issue #1627).
fn exception_for(err: &LaurusError) -> fn(String) -> PyErr {
    match err {
        LaurusError::Io(_) => PyIOError::new_err::<String>,
        LaurusError::Schema(_)
        | LaurusError::Query(_)
        | LaurusError::Field(_)
        | LaurusError::InvalidArgument(_) => PyValueError::new_err::<String>,
        LaurusError::BatchIngest { source, .. } | LaurusError::BatchSearch { source, .. } => {
            exception_for(source)
        }
        _ => PyRuntimeError::new_err::<String>,
    }
}

/// The message of the Python exception a [`LaurusError`] raises.
///
/// It is the error's `Display`, except that `LaurusError::Io` gives the
/// `io::Error` alone and `LaurusError::BatchSearch` gives its source's
/// message with the failing query's position in front, as
/// `queries[{failed_index}]: ...` (Issue #1627). `LaurusError::BatchIngest`
/// keeps its `Display`, which names the failing document's position and id
/// and the number of documents applied before it.
fn message(err: &LaurusError) -> String {
    match err {
        LaurusError::Io(e) => e.to_string(),
        LaurusError::BatchSearch {
            failed_index,
            source,
        } => format!("queries[{failed_index}]: {}", message(source)),
        other => other.to_string(),
    }
}

/// Put `context` (such as `queries[1]`) in front of `err`'s message, keeping
/// its exception type, and chain `err` as the new exception's `__cause__`.
///
/// `search_batch` uses it to name the query a conversion error came from
/// (Issue #1627).
pub fn with_context(py: Python<'_>, err: PyErr, context: &str) -> PyErr {
    let wrapped = PyErr::from_type(err.get_type(py), format!("{context}: {}", err.value(py)));
    wrapped.set_cause(py, Some(err));
    wrapped
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
