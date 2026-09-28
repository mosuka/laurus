//! Error-to-[`tonic::Status`] conversion utilities.
//!
//! Maps [`LaurusError`] variants to appropriate gRPC status codes (e.g.
//! `INVALID_ARGUMENT`, `UNIMPLEMENTED`, `INTERNAL`) and provides a catch-all
//! converter for [`anyhow::Error`].

use laurus::LaurusError;
use tonic::Status;

/// Convert a LaurusError into a tonic Status.
pub fn to_status(err: LaurusError) -> Status {
    Status::new(code_for(&err), err.to_string())
}

/// Classify a [`LaurusError`] into a gRPC status code.
///
/// [`LaurusError::BatchIngest`] is classified by its **source** — a batch
/// that failed on a caller mistake (e.g. a schema violation at position k)
/// surfaces as `INVALID_ARGUMENT`, while a storage failure stays `INTERNAL`.
/// The status *message* still carries the full batch context
/// (`failed_index`, `failed_id`, `applied`) from the error's `Display`.
fn code_for(err: &LaurusError) -> tonic::Code {
    match err {
        LaurusError::Schema(_)
        | LaurusError::Query(_)
        | LaurusError::Field(_)
        | LaurusError::InvalidArgument(_)
        | LaurusError::SerializationError(_)
        | LaurusError::Json(_) => tonic::Code::InvalidArgument,
        LaurusError::NotImplemented(_) => tonic::Code::Unimplemented,
        LaurusError::BatchIngest { source, .. } => code_for(source),
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
}
