//! Multi-vector index: per-document token vectors for late-interaction
//! rescoring (Issue #1177).
//!
//! A multi-vector field stores every token vector of a document (e.g. the
//! per-token embeddings of a ColBERT-style model) with no ANN structure.
//! It is never a vector-search target; late-interaction rescoring reads a
//! candidate document's vectors by doc id and scores them with MaxSim.
//!
//! The index is segment-per-commit like Flat / HNSW / IVF: each commit seals
//! the buffered documents into one immutable LMV1 file (see [`format`]),
//! registered in the shared segment manifest. Deletions are logical
//! (a document-level bitmap held by `SegmentedCore`), a newer segment's copy
//! of a document shadows older ones, and merges collapse both.

pub(crate) mod format;
pub mod reader;
pub mod segmented;

pub use reader::MultiVectorSnapshot;
pub use segmented::MultiVectorIndex;

use crate::vector::index::segment::manager::SegmentFileLayout;

/// On-disk file-suffix layout of multi-vector segments: one `.mv` file per
/// segment, staged through `.mv.tmp`.
pub const LAYOUT: SegmentFileLayout = SegmentFileLayout {
    primary: ".mv",
    sidecars: &[],
    tmp: ".mv.tmp",
};
