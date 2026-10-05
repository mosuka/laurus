//! Shared segment-per-commit infrastructure for vector indexes (Issue #889).
//!
//! Extracted from the HNSW segment-per-commit implementation (Issue #634):
//! the segment manifest/manager, merge policies, and the per-segment reader
//! cache are index-type-agnostic and are shared here across HNSW, Flat, and
//! IVF instead of being duplicated per index type. Each index type keeps its
//! own merge engine (the actual segment-reader/segment-writer I/O is
//! type-specific) and supplies a [`manager::SegmentFileLayout`] describing
//! its on-disk file suffixes.

pub mod fanout;
pub mod manager;
pub mod merge;
pub mod merge_policy;
pub mod reader_cache;
pub(crate) mod segmented_core;
