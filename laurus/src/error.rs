//! Error types for the Laurus library.
//!
//! This module provides comprehensive error handling for all Laurus operations.
//! All errors are represented by the [`LaurusError`] enum, which provides
//! detailed information about what went wrong.
//!
//! # Examples
//!
//! ```
//! use laurus::{LaurusError, Result};
//!
//! fn example_operation() -> Result<()> {
//!     // Return an error
//!     Err(LaurusError::invalid_argument("Invalid input"))
//! }
//!
//! match example_operation() {
//!     Ok(_) => println!("Success"),
//!     Err(e) => eprintln!("Error: {}", e),
//! }
//! ```

use std::io;

use thiserror::Error;

/// The main error type for Laurus operations.
///
/// This enum represents all possible errors that can occur in the Laurus library.
/// It uses the `thiserror` crate for automatic `Error` trait implementation and
/// provides convenient constructor methods for creating specific error types.
#[derive(Error, Debug)]
pub enum LaurusError {
    /// I/O errors (file operations, network, etc.)
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Index-related errors
    #[error("Index error: {0}")]
    Index(String),

    /// Schema-related errors
    #[error("Schema error: {0}")]
    Schema(String),

    /// Analysis-related errors (tokenization, filtering, etc.)
    #[error("Analysis error: {0}")]
    Analysis(String),

    /// Query-related errors (parsing, invalid queries, etc.)
    #[error("Query error: {0}")]
    Query(String),

    /// Storage-related errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// Field-related errors
    #[error("Field error: {0}")]
    Field(String),

    /// The caller passed an invalid argument (unknown field, wrong value
    /// type, out-of-range value, malformed input, ...). Bindings and
    /// laurus-server report it as a caller error, not an internal failure.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Benchmark-related errors
    #[error("Benchmark error: {0}")]
    BenchmarkFailed(String),

    /// Thread join errors
    #[error("Thread join error: {0}")]
    ThreadJoinError(String),

    /// Operation cancelled
    #[error("Operation cancelled: {0}")]
    OperationCancelled(String),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Resource exhausted
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Incompatible on-disk format (wrong magic bytes, unsupported
    /// version, or unsupported quantization kind). Most commonly this
    /// is raised when a vector segment from before Issue #481 Stage 1
    /// (pre-quantization, no header) is opened against the new reader.
    #[error("Incompatible format: {0}")]
    IncompatibleFormat(String),

    /// A batch ingestion call failed partway through (fail-fast semantics).
    ///
    /// Batch ingestion applies documents sequentially and stops at the first
    /// failure. The `applied` documents that preceded the failure remain in
    /// the WAL and NRT buffers — they are searchable, become durable at the
    /// next commit, and are replayed on crash recovery. There is no batch
    /// rollback; retrying the batch (or its suffix starting at
    /// `failed_index`) is idempotent under put semantics. With fail-fast
    /// sequential processing `applied == failed_index` always holds; both are
    /// carried so callers get the count without knowing that invariant.
    #[error(
        "batch ingest failed at doc {failed_index} (id '{failed_id}') after {applied} documents were applied: {source}"
    )]
    BatchIngest {
        /// Zero-based position of the failing document in the input batch.
        failed_index: usize,
        /// External id of the failing document.
        failed_id: String,
        /// Number of documents successfully applied before the failure.
        applied: usize,
        /// The underlying error that failed the document at `failed_index`.
        #[source]
        source: Box<LaurusError>,
    },

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Generic error for other cases
    #[error("Error: {0}")]
    Other(String),

    /// Generic anyhow error
    #[error("Anyhow error: {0}")]
    Anyhow(#[from] anyhow::Error),
}

/// Result type alias for operations that may fail with LaurusError.
pub type Result<T> = std::result::Result<T, LaurusError>;

impl LaurusError {
    /// Creates an [`Index`](Self::Index) variant with the given message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the index error.
    pub fn index<S: Into<String>>(msg: S) -> Self {
        LaurusError::Index(msg.into())
    }

    /// Creates a [`Schema`](Self::Schema) variant with the given message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the schema error.
    pub fn schema<S: Into<String>>(msg: S) -> Self {
        LaurusError::Schema(msg.into())
    }

    /// Creates an [`Analysis`](Self::Analysis) variant with the given message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the analysis error.
    pub fn analysis<S: Into<String>>(msg: S) -> Self {
        LaurusError::Analysis(msg.into())
    }

    /// Creates a [`Query`](Self::Query) variant with the given message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the query error.
    pub fn query<S: Into<String>>(msg: S) -> Self {
        LaurusError::Query(msg.into())
    }

    /// Creates a [`Query`](Self::Query) variant for parse errors.
    ///
    /// Parse errors are treated as query errors because they typically
    /// originate from malformed user query strings.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the parse error.
    pub fn parse<S: Into<String>>(msg: S) -> Self {
        LaurusError::Query(msg.into()) // Parse errors are treated as query errors
    }

    /// Creates a [`Storage`](Self::Storage) variant with the given message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the storage error.
    pub fn storage<S: Into<String>>(msg: S) -> Self {
        LaurusError::Storage(msg.into())
    }

    /// Creates a [`Field`](Self::Field) variant with the given message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the field error.
    pub fn field<S: Into<String>>(msg: S) -> Self {
        LaurusError::Field(msg.into())
    }

    /// Creates an [`Other`](Self::Other) variant with the given message.
    ///
    /// Use this for errors that do not fit into any specific category.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive error message.
    pub fn other<S: Into<String>>(msg: S) -> Self {
        LaurusError::Other(msg.into())
    }

    /// Creates an [`Other`](Self::Other) variant with a `"Timeout: "` prefixed message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the timeout condition.
    pub fn timeout<S: Into<String>>(msg: S) -> Self {
        LaurusError::Other(format!("Timeout: {}", msg.into()))
    }

    /// Creates an [`Other`](Self::Other) variant with an `"Invalid configuration: "` prefixed message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the configuration error.
    pub fn invalid_config<S: Into<String>>(msg: S) -> Self {
        LaurusError::Other(format!("Invalid configuration: {}", msg.into()))
    }

    /// Creates an [`InvalidArgument`](Self::InvalidArgument) variant with the given message.
    ///
    /// Use this only for mistakes the caller can fix by changing its input.
    /// Failures of the server's environment or of internal invariants
    /// belong in [`invalid_config`](Self::invalid_config) or
    /// [`internal`](Self::internal) instead.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the invalid argument.
    pub fn invalid_argument<S: Into<String>>(msg: S) -> Self {
        LaurusError::InvalidArgument(msg.into())
    }

    /// Creates an [`Other`](Self::Other) variant with an `"Internal error: "` prefixed message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the internal error.
    pub fn internal<S: Into<String>>(msg: S) -> Self {
        LaurusError::Other(format!("Internal error: {}", msg.into()))
    }

    /// Creates an [`Other`](Self::Other) variant with a `"Not found: "` prefixed message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about the missing resource.
    pub fn not_found<S: Into<String>>(msg: S) -> Self {
        LaurusError::Other(format!("Not found: {}", msg.into()))
    }

    /// Creates an [`OperationCancelled`](Self::OperationCancelled) variant with the given message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about why the operation was cancelled.
    pub fn cancelled<S: Into<String>>(msg: S) -> Self {
        LaurusError::OperationCancelled(msg.into())
    }

    /// Creates a [`NotImplemented`](Self::NotImplemented) variant with the given message.
    ///
    /// # Parameters
    ///
    /// - `msg` - A descriptive message about what is not yet implemented.
    pub fn not_implemented<S: Into<String>>(msg: S) -> Self {
        LaurusError::NotImplemented(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_construction() {
        let error = LaurusError::index("Test index error");
        assert_eq!(error.to_string(), "Index error: Test index error");

        let error = LaurusError::schema("Test schema error");
        assert_eq!(error.to_string(), "Schema error: Test schema error");

        let error = LaurusError::analysis("Test analysis error");
        assert_eq!(error.to_string(), "Analysis error: Test analysis error");
    }

    #[test]
    fn invalid_argument_builds_its_own_variant() {
        let error = LaurusError::invalid_argument("dimension must be > 0");
        assert!(
            matches!(&error, LaurusError::InvalidArgument(m) if m == "dimension must be > 0"),
            "unexpected variant: {error:?}"
        );
        assert_eq!(error.to_string(), "Invalid argument: dimension must be > 0");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = io::Error::new(io::ErrorKind::NotFound, "File not found");
        let iris_error = LaurusError::from(io_error);

        match iris_error {
            LaurusError::Io(_) => {} // Expected
            _ => panic!("Expected IO error variant"),
        }
    }
}
