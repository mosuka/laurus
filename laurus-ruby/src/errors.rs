//! Error conversion between Laurus errors and Ruby exceptions.

use laurus::LaurusError;
use magnus::error::ErrorType;
use magnus::{Error, ExceptionClass, Ruby};

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
/// | Laurus variant                 | Ruby exception             |
/// |--------------------------------|----------------------------|
/// | `LaurusError::Io`              | `IOError`                  |
/// | `LaurusError::Schema`          | `ArgumentError`            |
/// | `LaurusError::Query`           | `ArgumentError`            |
/// | `LaurusError::Field`           | `ArgumentError`            |
/// | `LaurusError::InvalidArgument` | `ArgumentError`            |
/// | `LaurusError::BatchIngest`     | the exception of its source |
/// | `LaurusError::BatchSearch`     | the exception of its source |
/// | other                          | `RuntimeError`             |
///
/// A failed batch raises the exception of its source, so a batch that
/// failed on a caller's mistake raises `ArgumentError` as the same call on
/// its own would (Issue #1627). The message is [`message`].
pub fn laurus_err(err: LaurusError) -> Error {
    let ruby = Ruby::get().expect("called from Ruby thread");
    Error::new(class_for(&ruby, &err), message(&err))
}

/// Choose the Ruby exception class a [`LaurusError`] raises; see the table
/// of [`laurus_err`].
fn class_for(ruby: &Ruby, err: &LaurusError) -> ExceptionClass {
    match err {
        LaurusError::Io(_) => ruby.exception_io_error(),
        LaurusError::Schema(_)
        | LaurusError::Query(_)
        | LaurusError::Field(_)
        | LaurusError::InvalidArgument(_) => ruby.exception_arg_error(),
        LaurusError::BatchIngest { source, .. } | LaurusError::BatchSearch { source, .. } => {
            class_for(ruby, source)
        }
        _ => ruby.exception_runtime_error(),
    }
}

/// The message of the Ruby exception a [`LaurusError`] raises.
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

/// Put `context` (such as `queries[1]`) in front of `err`'s message,
/// keeping its exception class.
///
/// `search_batch` uses it to name the query a conversion error came from
/// (Issue #1627). An exception raised by Ruby code keeps its class too; an
/// interrupt (`break`, `throw`) is returned as it is.
pub fn with_context(err: Error, context: &str) -> Error {
    match err.error_type() {
        ErrorType::Error(class, message) => Error::new(*class, format!("{context}: {message}")),
        ErrorType::Exception(exception) => Error::new(
            exception.exception_class(),
            format!("{context}: {exception}"),
        ),
        ErrorType::Jump(_) => err,
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
/// `LaurusError::Schema`/`Query`/`Field`/`InvalidArgument`. `Io` maps to `IOError` like
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
