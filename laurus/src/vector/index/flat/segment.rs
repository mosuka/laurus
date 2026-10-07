//! Flat-specific segment file layout and merge engine.
//!
//! The generic segment manifest/manager, merge policies, reader cache, and
//! fan-out search layer shared across index types live in
//! [`crate::vector::index::segment`] (Issue #889); this module keeps only
//! what is Flat-specific: the merge engine (its I/O is Flat-typed) and this
//! index type's on-disk file layout descriptor.

use crate::vector::index::segment::manager::SegmentFileLayout;

pub mod merge_engine;

/// On-disk file-suffix layout for Flat segments: a primary `.flat` file plus
/// its `.flat.f32` rerank sidecar (Issue #932), staged through a `.flat.tmp`
/// temp file.
pub const LAYOUT: SegmentFileLayout = SegmentFileLayout {
    primary: ".flat",
    sidecars: &[".flat.f32"],
    tmp: ".flat.tmp",
};
