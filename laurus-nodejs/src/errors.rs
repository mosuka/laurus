//! Error conversion between Laurus errors and napi-rs errors.

use laurus::LaurusError;
use napi::Status;

/// Error returned by any `Index` method called after
/// [`crate::index::JsIndex::close`] (Issue #1097).
pub fn closed_err() -> napi::Error {
    napi::Error::new(Status::InvalidArg, "Index is closed")
}

/// Convert a [`LaurusError`] into a napi [`napi::Error`].
pub fn laurus_err(err: LaurusError) -> napi::Error {
    match err {
        LaurusError::Io(e) => napi::Error::new(Status::GenericFailure, format!("IO error: {e}")),
        LaurusError::Schema(m) => {
            napi::Error::new(Status::InvalidArg, format!("Schema error: {m}"))
        }
        LaurusError::Query(m) => napi::Error::new(Status::InvalidArg, format!("Query error: {m}")),
        LaurusError::Field(m) => napi::Error::new(Status::InvalidArg, format!("Field error: {m}")),
        LaurusError::InvalidArgument(m) => {
            napi::Error::new(Status::InvalidArg, format!("Invalid argument: {m}"))
        }
        other => napi::Error::new(Status::GenericFailure, other.to_string()),
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
