//! Error conversion between Laurus errors and PHP exceptions.

use ext_php_rs::exception::PhpException;
use ext_php_rs::zend::ce;
use laurus::LaurusError;

/// Error returned by any `Index` method called after
/// [`crate::index::PhpIndex::close`] (Issue #1097).
pub fn closed_err() -> PhpException {
    PhpException::new("Index is closed".to_string(), 0, ce::exception())
}

/// Convert a [`LaurusError`] into a PHP exception.
///
/// # Mapping
///
/// | Laurus variant                 | PHP exception    |
/// |--------------------------------|------------------|
/// | `LaurusError::Io`              | `Exception`      |
/// | `LaurusError::Schema`          | `ValueError`     |
/// | `LaurusError::Query`           | `ValueError`     |
/// | `LaurusError::Field`           | `ValueError`     |
/// | `LaurusError::InvalidArgument` | `ValueError`     |
/// | other                          | `Exception`      |
pub fn laurus_err(err: LaurusError) -> PhpException {
    match err {
        LaurusError::Io(e) => PhpException::new(e.to_string(), 0, ce::exception()),
        LaurusError::Schema(m) => {
            PhpException::new(format!("Schema error: {m}"), 0, ce::value_error())
        }
        LaurusError::Query(m) => {
            PhpException::new(format!("Query error: {m}"), 0, ce::value_error())
        }
        LaurusError::Field(m) => {
            PhpException::new(format!("Field error: {m}"), 0, ce::value_error())
        }
        LaurusError::InvalidArgument(m) => {
            PhpException::new(format!("Invalid argument: {m}"), 0, ce::value_error())
        }
        other => PhpException::new(other.to_string(), 0, ce::exception()),
    }
}

/// Convert a [`laurus::index_dir::IndexDirError`] into a PHP exception.
///
/// `SchemaConflict`/`LegacyFlatLayout`/`NotAnIndexDirectory` are all
/// caller-fixable misuse, so they map to `ValueError` (matching how
/// [`laurus_err`] treats `LaurusError::Schema`/`Query`/`Field`/`InvalidArgument`). `Io` maps to
/// `Exception`, same as `LaurusError::Io` in [`laurus_err`]; there is no
/// PHP-specific path-preserving I/O exception in this crate, so
/// `err.to_string()` (which already formats as `"{path}: {source}"`) carries
/// the path in the message instead. `Core` delegates to [`laurus_err`].
pub fn index_dir_err(err: laurus::index_dir::IndexDirError) -> PhpException {
    use laurus::index_dir::IndexDirError;
    match err {
        IndexDirError::SchemaConflict { .. }
        | IndexDirError::LegacyFlatLayout { .. }
        | IndexDirError::NotAnIndexDirectory { .. } => {
            PhpException::new(err.to_string(), 0, ce::value_error())
        }
        IndexDirError::Io { .. } => PhpException::new(err.to_string(), 0, ce::exception()),
        IndexDirError::Core(e) => laurus_err(e),
    }
}

/// Convert a file I/O error into a PHP `Exception` whose message is
/// `"{path}: {err}"`.
///
/// Maps to `Exception`, same as `LaurusError::Io` in [`laurus_err`]; a bare
/// [`std::io::Error`] does not name the file, so the path is prefixed here.
pub fn io_err_with_path(path: &str, err: std::io::Error) -> PhpException {
    PhpException::new(format!("{path}: {err}"), 0, ce::exception())
}
