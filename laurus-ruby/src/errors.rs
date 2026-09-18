//! Error conversion between Laurus errors and Ruby exceptions.

use laurus::LaurusError;
use magnus::{Error, Ruby};

/// Error returned by any `Index` method called after
/// [`crate::index::RbIndex::close`] (Issue #1097).
pub fn closed_err() -> Error {
    let ruby = Ruby::get().expect("called from Ruby thread");
    Error::new(ruby.exception_runtime_error(), "Index is closed")
}

/// Convert a [`LaurusError`] into a Ruby exception.
///
/// # Mapping
///
/// | Laurus variant            | Ruby exception    |
/// |---------------------------|-------------------|
/// | `LaurusError::Io`         | `IOError`         |
/// | `LaurusError::Schema`     | `ArgumentError`   |
/// | `LaurusError::Query`      | `ArgumentError`   |
/// | `LaurusError::Field`      | `ArgumentError`   |
/// | other                     | `RuntimeError`    |
pub fn laurus_err(err: LaurusError) -> Error {
    let ruby = Ruby::get().expect("called from Ruby thread");
    match err {
        LaurusError::Io(e) => Error::new(ruby.exception_io_error(), e.to_string()),
        LaurusError::Schema(m) => {
            Error::new(ruby.exception_arg_error(), format!("Schema error: {m}"))
        }
        LaurusError::Query(m) => {
            Error::new(ruby.exception_arg_error(), format!("Query error: {m}"))
        }
        LaurusError::Field(m) => {
            Error::new(ruby.exception_arg_error(), format!("Field error: {m}"))
        }
        other => Error::new(ruby.exception_runtime_error(), other.to_string()),
    }
}

/// Wrap a filesystem I/O error with the path that caused it, then map it to
/// an `IOError`.
///
/// Used by `Schema::from_toml_file`/`to_toml_file`, whose I/O happens in the
/// binding layer (not through [`LaurusError::Io`], which doesn't carry a
/// path) — mirrors `laurus-python`'s `io_err_with_path`.
pub fn io_err_with_path(path: &str, e: std::io::Error) -> Error {
    let ruby = Ruby::get().expect("called from Ruby thread");
    Error::new(ruby.exception_io_error(), format!("{path}: {e}"))
}

/// Convert a [`laurus::index_dir::IndexDirError`] into a Ruby exception.
///
/// # Mapping
///
/// | `IndexDirError` variant | Ruby exception  |
/// |--------------------------|-----------------|
/// | `SchemaConflict`         | `ArgumentError` |
/// | `LegacyFlatLayout`       | `ArgumentError` |
/// | `NotAnIndexDirectory`    | `ArgumentError` |
/// | `Io`                     | `IOError`       |
/// | `Core`                   | see [`laurus_err`] |
///
/// `SchemaConflict`/`LegacyFlatLayout`/`NotAnIndexDirectory` are all
/// caller-fixable misuse, so they map to `ArgumentError` — the same class
/// [`laurus_err`] uses for
/// `LaurusError::Schema`/`Query`/`Field`. `Io` maps to `IOError` like
/// `LaurusError::Io` above; unlike the Python binding, there is no
/// path-preserving exception hierarchy to lose here, and
/// `IndexDirError::Io`'s `Display` already includes the offending path, so
/// `err.to_string()` carries it through.
pub fn index_dir_err(err: laurus::index_dir::IndexDirError) -> Error {
    use laurus::index_dir::IndexDirError;
    let ruby = Ruby::get().expect("called from Ruby thread");
    match err {
        IndexDirError::SchemaConflict { .. }
        | IndexDirError::LegacyFlatLayout { .. }
        | IndexDirError::NotAnIndexDirectory { .. } => {
            Error::new(ruby.exception_arg_error(), err.to_string())
        }
        IndexDirError::Io { .. } => Error::new(ruby.exception_io_error(), err.to_string()),
        IndexDirError::Core(e) => laurus_err(e),
    }
}
