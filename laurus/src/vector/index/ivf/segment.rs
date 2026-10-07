//! IVF-specific segment file layout and merge engine.
//!
//! The generic segment manifest/manager, merge policies, reader cache, and
//! fan-out search layer shared across index types live in
//! [`crate::vector::index::segment`] (Issue #889); this module keeps only
//! what is IVF-specific: the merge engine (its I/O is IVF-typed, and it
//! re-clusters the merged union rather than just concatenating) and this
//! index type's on-disk file layout descriptor.

use crate::vector::index::segment::manager::SegmentFileLayout;

pub mod merge_engine;

/// On-disk file-suffix layout for IVF segments: a primary `.ivf` file plus
/// its `.ivf.f32` rerank sidecar (Issue #932), staged through a `.ivf.tmp`
/// temp file.
pub const LAYOUT: SegmentFileLayout = SegmentFileLayout {
    primary: ".ivf",
    sidecars: &[".ivf.f32"],
    tmp: ".ivf.tmp",
};
