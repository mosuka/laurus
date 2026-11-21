//! Conversion helpers between laurus domain types and protobuf (gRPC) types.
//!
//! Each sub-module handles a specific domain area:
//!
//! * [`document`] – [`laurus::Document`] <-> `proto::Document`.
//! * [`error`]    – [`laurus::LaurusError`] / [`anyhow::Error`] -> [`tonic::Status`].
//! * `json`       – readers for the optional keys of a JSON object (crate-private).
//! * `proto_struct` – JSON <-> `google.protobuf.Struct`, the form of an analyzer
//!   component (crate-private).
//! * [`schema`]   – [`laurus::Schema`] <-> `proto::Schema`.
//! * [`search`]   – [`laurus::SearchRequest`] / [`laurus::SearchResult`] <-> proto types.

pub mod document;
pub mod error;
pub(crate) mod json;
pub(crate) mod proto_struct;
pub mod schema;
pub mod search;
