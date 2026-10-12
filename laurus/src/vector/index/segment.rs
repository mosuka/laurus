//! Shared segment-per-commit infrastructure for vector indexes (Issue #889).
//!
//! Extracted from the HNSW segment-per-commit implementation (Issue #634):
//! the segment manifest/manager, merge policies, and the per-segment reader
//! cache are index-type-agnostic and are shared here across HNSW, Flat, and
//! IVF instead of being duplicated per index type.
//!
//! [`segmented_index`] goes further for HNSW, Flat and IVF (Issue #1409):
//! `SegmentedIndex`/`SegmentedWriter` implement the index and writer once,
//! generic over a `SegmentedBackend` that supplies each type's concrete
//! reader, writer, merge engine, per-segment searcher construction, and a
//! [`manager::SegmentFileLayout`] describing its on-disk file suffixes.
//! MultiVector has a different reader/writer/on-disk format and builds
//! directly on `segmented_core`/`reader_cache` instead.

pub mod fanout;
pub mod manager;
pub mod merge;
pub mod merge_policy;
pub mod reader_cache;
pub(crate) mod segmented_core;
pub mod segmented_index;
