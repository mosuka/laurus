//! Error conversion between Laurus errors and napi-rs errors.

use laurus::LaurusError;
use napi::Status;

/// Error returned by any `Index` method called after
/// [`crate::index::JsIndex::close`] (Issue #1097).
pub fn closed_err() -> napi::Error {
    napi::Error::new(Status::InvalidArg, "Index is closed")
}

/// Convert a [`LaurusError`] into a napi [`napi::Error`].
///
/// The error's code is chosen by [`status_for`] and its message is
/// [`message`].
pub fn laurus_err(err: LaurusError) -> napi::Error {
    napi::Error::new(status_for(&err), message(&err))
}

/// Choose the code of the napi error a [`LaurusError`] becomes.
///
/// - `InvalidArg` for a caller's mistake: `LaurusError::Schema`, `Query`,
///   `Field` and `InvalidArgument`.
/// - `GenericFailure` for anything else, `LaurusError::Io` included.
///
/// A failed batch (`LaurusError::BatchIngest` and `BatchSearch`) takes the
/// code of its source, so a batch that failed on a caller's mistake is
/// `InvalidArg` as the same call on its own would be (Issue #1627).
fn status_for(err: &LaurusError) -> Status {
    match err {
        LaurusError::Schema(_)
        | LaurusError::Query(_)
        | LaurusError::Field(_)
        | LaurusError::InvalidArgument(_) => Status::InvalidArg,
        LaurusError::BatchIngest { source, .. } | LaurusError::BatchSearch { source, .. } => {
            status_for(source)
        }
        _ => Status::GenericFailure,
    }
}

/// The message of the napi error a [`LaurusError`] becomes.
///
/// It is the error's `Display`, except that `LaurusError::Io` is
/// `IO error: ...` and `LaurusError::BatchSearch` gives its source's
/// message with the failing query's position in front, as
/// `queries[{failed_index}]: ...` (Issue #1627). `LaurusError::BatchIngest`
/// keeps its `Display`, which names the failing document's position and id
/// and the number of documents applied before it.
fn message(err: &LaurusError) -> String {
    match err {
        LaurusError::Io(e) => format!("IO error: {e}"),
        LaurusError::BatchSearch {
            failed_index,
            source,
        } => format!("queries[{failed_index}]: {}", message(source)),
        other => other.to_string(),
    }
}

/// Convert a [`laurus::index_dir::IndexDirError`] into a napi [`napi::Error`].
///
/// `SchemaConflict`/`LegacyFlatLayout`/`NotAnIndexDirectory` are
/// caller-fixable misuse, so they map to `InvalidArg` (matching how
/// [`laurus_err`] treats `LaurusError::Schema`/`Query`/`Field`/`InvalidArgument`). `Io` matches
/// the `LaurusError::Io` treatment above (`GenericFailure`, `"IO error: "`
/// prefix) for consistency.
pub fn index_dir_err(err: laurus::index_dir::IndexDirError) -> napi::Error {
    use laurus::index_dir::IndexDirError;
    match err {
        IndexDirError::SchemaConflict { .. }
        | IndexDirError::LegacyFlatLayout { .. }
        | IndexDirError::NotAnIndexDirectory { .. } => {
            napi::Error::new(Status::InvalidArg, err.to_string())
        }
        IndexDirError::Io { .. } => {
            napi::Error::new(Status::GenericFailure, format!("IO error: {err}"))
        }
        IndexDirError::Core(e) => laurus_err(e),
    }
}
