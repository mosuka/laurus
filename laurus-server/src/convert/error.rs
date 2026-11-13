//! Error-to-[`tonic::Status`] conversion utilities.
//!
//! Maps [`LaurusError`] variants to appropriate gRPC status codes (e.g.
//! `INVALID_ARGUMENT`, `FAILED_PRECONDITION`, `UNIMPLEMENTED`, `INTERNAL`)
//! and provides a catch-all converter for [`anyhow::Error`].

use laurus::LaurusError;
use tonic::Status;

/// Convert a LaurusError into a tonic Status.
pub fn to_status(err: LaurusError) -> Status {
    Status::new(code_for(&err), err.to_string())
}

/// Convert the error of [`laurus::Engine::search_batch`] into a tonic
/// Status that names the failing query as `queries[i]`, the name
/// `SearchBatch` gives a malformed query too (Issue #1553). The code is
/// that of the query's own error. Any other error converts as in
/// [`to_status`].
pub fn batch_search_to_status(err: LaurusError) -> Status {
    match err {
        LaurusError::BatchSearch {
            failed_index,
            source,
        } => Status::new(
            code_for(&source),
            format!("queries[{failed_index}]: {source}"),
        ),
        other => to_status(other),
    }
}

/// Classify a [`LaurusError`] into a gRPC status code.
///
/// [`LaurusError::BatchIngest`] and [`LaurusError::BatchSearch`] are
/// classified by their **source** — a batch that failed on a caller mistake
/// (e.g. a schema violation at position k) surfaces as `INVALID_ARGUMENT`,
/// while a storage failure stays `INTERNAL`. The status *message* still
/// carries the batch context (`failed_index` and, for ingestion, `failed_id`
/// and `applied`) from the error's `Display`.
fn code_for(err: &LaurusError) -> tonic::Code {
    match err {
        LaurusError::Schema(_)
        | LaurusError::Query(_)
        | LaurusError::Field(_)
        | LaurusError::InvalidArgument(_)
        | LaurusError::SerializationError(_)
        | LaurusError::Json(_) => tonic::Code::InvalidArgument,
        LaurusError::NotImplemented(_) => tonic::Code::Unimplemented,
        // Retrying cannot succeed until the index is reopened (Issue #1492).
        LaurusError::WritesStopped(_) => tonic::Code::FailedPrecondition,
        LaurusError::BatchIngest { source, .. } | LaurusError::BatchSearch { source, .. } => {
            code_for(source)
        }
        _ => tonic::Code::Internal,
    }
}

/// Convert an anyhow::Error into a tonic Status.
pub fn anyhow_to_status(err: anyhow::Error) -> Status {
    Status::internal(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_ingest_is_classified_by_its_source() {
        let caller_fault = LaurusError::BatchIngest {
            failed_index: 3,
            failed_id: "bad".into(),
            applied: 3,
            source: Box::new(LaurusError::schema("undeclared field")),
        };
        let status = to_status(caller_fault);
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("doc 3")
                && status.message().contains("'bad'")
                && status.message().contains("3 documents were applied"),
            "the status message must keep the batch context: {}",
            status.message()
        );

        let server_fault = LaurusError::BatchIngest {
            failed_index: 0,
            failed_id: "x".into(),
            applied: 0,
            source: Box::new(LaurusError::storage("disk on fire")),
        };
        assert_eq!(to_status(server_fault).code(), tonic::Code::Internal);
    }

    /// Issue #1253: `invalid_argument` reports a caller mistake, directly
    /// and as the source of a failed batch.
    #[test]
    fn invalid_argument_is_a_caller_error() {
        let status = to_status(LaurusError::invalid_argument("unknown field 'nope'"));
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert_eq!(status.message(), "Invalid argument: unknown field 'nope'");

        let batch = LaurusError::BatchIngest {
            failed_index: 0,
            failed_id: "x".into(),
            applied: 0,
            source: Box::new(LaurusError::invalid_argument("bad value")),
        };
        assert_eq!(to_status(batch).code(), tonic::Code::InvalidArgument);
    }

    /// Issue #1492: once the engine stopped accepting writes, a write is
    /// refused until the index is reopened, directly and inside a batch, so
    /// it is `FAILED_PRECONDITION` rather than a retryable `INTERNAL`.
    #[test]
    fn writes_stopped_is_a_failed_precondition() {
        let stopped = || LaurusError::writes_stopped("a write failed after reaching the WAL");
        let status = to_status(stopped());
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
        assert_eq!(
            status.message(),
            "Writes stopped: a write failed after reaching the WAL"
        );

        let batch = LaurusError::BatchIngest {
            failed_index: 2,
            failed_id: "x".into(),
            applied: 2,
            source: Box::new(stopped()),
        };
        assert_eq!(to_status(batch).code(), tonic::Code::FailedPrecondition);
    }

    /// Issue #1553: a failed search batch names its query as `queries[i]`
    /// and is classified by the query's own error, by both converters.
    #[test]
    fn batch_search_names_the_query_and_is_classified_by_its_source() {
        let batch = |source: LaurusError| LaurusError::BatchSearch {
            failed_index: 3,
            source: Box::new(source),
        };

        let status = batch_search_to_status(batch(LaurusError::invalid_argument("bad value")));
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert_eq!(status.message(), "queries[3]: Invalid argument: bad value");

        let status = batch_search_to_status(batch(LaurusError::storage("disk on fire")));
        assert_eq!(status.code(), tonic::Code::Internal);
        assert_eq!(status.message(), "queries[3]: Storage error: disk on fire");

        let status = to_status(batch(LaurusError::invalid_argument("bad value")));
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert_eq!(
            status.message(),
            "search request 3 failed: Invalid argument: bad value"
        );

        // Any other error converts as `to_status` does.
        let status = batch_search_to_status(LaurusError::invalid_argument("bad value"));
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert_eq!(status.message(), "Invalid argument: bad value");
    }
}
