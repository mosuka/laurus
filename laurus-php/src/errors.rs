//! Error conversion between Laurus errors and PHP exceptions.

use ext_php_rs::exception::PhpException;
use ext_php_rs::zend::{ClassEntry, ExecutorGlobals, ce};
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
/// | Laurus variant                 | PHP exception               |
/// |--------------------------------|-----------------------------|
/// | `LaurusError::Io`              | `Exception`                 |
/// | `LaurusError::Schema`          | `ValueError`                |
/// | `LaurusError::Query`           | `ValueError`                |
/// | `LaurusError::Field`           | `ValueError`                |
/// | `LaurusError::InvalidArgument` | `ValueError`                |
/// | `LaurusError::BatchIngest`     | the exception of its source |
/// | `LaurusError::BatchSearch`     | the exception of its source |
/// | other                          | `Exception`                 |
///
/// A failed batch throws the exception of its source, so a batch that
/// failed on a caller's mistake throws `ValueError` as the same call on its
/// own would (Issue #1627). The message is [`message`].
pub fn laurus_err(err: LaurusError) -> PhpException {
    PhpException::new(message(&err), 0, class_for(&err))
}

/// Choose the PHP exception class a [`LaurusError`] throws; see the table
/// of [`laurus_err`].
fn class_for(err: &LaurusError) -> &'static ClassEntry {
    match err {
        LaurusError::Schema(_)
        | LaurusError::Query(_)
        | LaurusError::Field(_)
        | LaurusError::InvalidArgument(_) => ce::value_error(),
        LaurusError::BatchIngest { source, .. } | LaurusError::BatchSearch { source, .. } => {
            class_for(source)
        }
        _ => ce::exception(),
    }
}

/// The message of the PHP exception a [`LaurusError`] throws.
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
/// keeping its class and code.
///
/// `searchBatch` uses it to name the query a conversion error came from
/// (Issue #1627). A [`PhpException`] keeps its class and message private,
/// so this does what `catch` and a rethrow would do in PHP: it throws `err`,
/// takes the thrown object back, reads `getMessage()`, `getCode()` and its
/// class, and makes a new exception of that class. If another exception is
/// pending already, `err` is returned as it is.
pub fn with_context(err: PhpException, context: &str) -> PhpException {
    if ExecutorGlobals::has_exception() {
        return err;
    }
    err.throw();
    let Some(thrown) = ExecutorGlobals::take_exception() else {
        return PhpException::new(
            format!("{context}: the error could not be read back"),
            0,
            ce::exception(),
        );
    };
    let message = thrown
        .try_call_method("getMessage", vec![])
        .ok()
        .and_then(|zv| zv.string())
        .unwrap_or_default();
    let code = thrown
        .try_call_method("getCode", vec![])
        .ok()
        .and_then(|zv| zv.long())
        .and_then(|code| i32::try_from(code).ok())
        .unwrap_or(0);
    PhpException::new(
        format!("{context}: {message}"),
        code,
        thrown.get_class_entry(),
    )
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
