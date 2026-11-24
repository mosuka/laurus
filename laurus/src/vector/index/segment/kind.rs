//! The index kind of a segment (Issue #1579).
//!
//! A segment knows the kind it was written as
//! ([`ManagedSegmentInfo::kind`](super::manager::ManagedSegmentInfo::kind)),
//! as it knows its quantizer. The schema only decides the kind of the
//! segments written from now on, so a field can hold segments of several of
//! the kinds in [`SegmentKind`]: after `update_field` changed the kind and
//! crashed before it saved the schema, or after `schema.toml` was edited by
//! hand. A segment of another kind than the field's is searched exactly, by
//! scanning its vectors, until a merge rewrites it as the field's kind.
//!
//! MultiVector is not among them: it holds token vectors per document, which
//! no other kind can read or be written from.

use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::maintenance::deletion::DeletionBitmap;
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;
use crate::vector::index::flat::reader::FlatVectorIndexReader;
use crate::vector::index::flat::segmented::FlatSegmentedBackend;
use crate::vector::index::hnsw::reader::HnswIndexReader;
use crate::vector::index::hnsw::segmented::HnswSegmentedBackend;
use crate::vector::index::ivf::reader::IvfIndexReader;
use crate::vector::index::ivf::segmented::IvfSegmentedBackend;
use crate::vector::index::rerank_storage::RerankStoragePool;
use crate::vector::reader::VectorIndexReader;

use super::manager::SegmentFileLayout;
use super::segmented_index::SegmentedBackend;

/// The name of the kind whose segments use `layout`: the suffix of their
/// primary file without its dot (`flat`, `hnsw`, `ivf` or `mv`).
pub(crate) fn kind_of_layout(layout: &SegmentFileLayout) -> &'static str {
    layout.primary.trim_start_matches('.')
}

/// The name a schema gives the kind named `kind`, or `kind` itself for one
/// no built-in kind has.
pub(crate) fn schema_name(kind: &str) -> &str {
    if kind == kind_of_layout(&crate::vector::index::multivector::LAYOUT) {
        return "MultiVector";
    }
    match SegmentKind::from_name(kind) {
        Some(known) => known.schema_name(),
        None => kind,
    }
}

/// Whether an index of kind `own` reads segments of kind `kind`.
pub(crate) fn reads(own: &str, kind: &str) -> bool {
    own == kind || (SegmentKind::from_name(own).is_some() && SegmentKind::from_name(kind).is_some())
}

/// The layout of every kind whose segments an index with `layout` can hold,
/// `layout` first.
pub(crate) fn layouts_read_with(layout: &SegmentFileLayout) -> Vec<SegmentFileLayout> {
    let own = kind_of_layout(layout);
    let others = SegmentKind::ALL
        .into_iter()
        .map(SegmentKind::layout)
        .filter(|other| kind_of_layout(other) != own && reads(own, kind_of_layout(other)));
    std::iter::once(*layout).chain(others).collect()
}

/// A sealed segment's reader, with what a merge needs beside it.
pub(crate) struct LoadedSegment {
    /// The segment's reader.
    pub(crate) reader: Box<dyn VectorIndexReader>,

    /// The segment's f32 rerank sidecar, when it has one. The reader has
    /// loaded it already, in either loading mode, so it costs no extra I/O.
    pub(crate) rerank: Option<Arc<RerankStoragePool>>,
}

/// The kinds whose segments one field can hold side by side: each can be
/// read without knowing which of them the field is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmentKind {
    /// Flat.
    Flat,
    /// HNSW.
    Hnsw,
    /// IVF.
    Ivf,
}

impl SegmentKind {
    /// Every kind.
    pub(crate) const ALL: [Self; 3] = [Self::Flat, Self::Hnsw, Self::Ivf];

    /// The file layout of the kind's segments.
    pub(crate) fn layout(self) -> SegmentFileLayout {
        match self {
            Self::Flat => crate::vector::index::flat::segment::LAYOUT,
            Self::Hnsw => crate::vector::index::hnsw::segment::LAYOUT,
            Self::Ivf => crate::vector::index::ivf::segment::LAYOUT,
        }
    }

    /// The name of the kind in a manifest (see [`kind_of_layout`]).
    pub(crate) fn name(self) -> &'static str {
        kind_of_layout(&self.layout())
    }

    /// The name a schema gives the kind.
    fn schema_name(self) -> &'static str {
        match self {
            Self::Flat => "Flat",
            Self::Hnsw => "Hnsw",
            Self::Ivf => "Ivf",
        }
    }

    /// The kind named `name` in a manifest, if it is one of these.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    /// The kind named `name` in a manifest, which segment `segment_id` was
    /// written as.
    ///
    /// # Errors
    ///
    /// Returns an error when it is not one of these.
    pub(crate) fn of_segment(segment_id: &str, name: &str) -> Result<Self> {
        Self::from_name(name).ok_or_else(|| {
            LaurusError::index(format!(
                "vector segment '{segment_id}' was written as kind '{name}', which cannot be read"
            ))
        })
    }

    /// Load the reader of segment `segment_id`, written as this kind, with
    /// `deletions` attached when it is `Some`.
    ///
    /// # Errors
    ///
    /// Returns an error when the segment's files cannot be read.
    pub(crate) fn load(
        self,
        storage: &Arc<dyn Storage>,
        segment_id: &str,
        distance: DistanceMetric,
        deletions: Option<Arc<DeletionBitmap>>,
    ) -> Result<LoadedSegment> {
        macro_rules! load {
            ($reader:ty) => {{
                let mut reader = <$reader>::load(storage.clone(), segment_id, distance)?;
                if let Some(deletions) = deletions {
                    reader.set_deletion_bitmap(deletions);
                }
                let rerank = reader.rerank_storage().cloned();
                Ok(LoadedSegment {
                    reader: Box::new(reader),
                    rerank,
                })
            }};
        }
        match self {
            Self::Flat => load!(FlatVectorIndexReader),
            Self::Hnsw => load!(HnswIndexReader),
            Self::Ivf => load!(IvfIndexReader),
        }
    }

    /// Read the vector count and the dimension from the header of `file`, a
    /// primary file of this kind.
    ///
    /// # Errors
    ///
    /// Returns an error when the header cannot be read.
    pub(crate) fn read_header(
        self,
        storage: &Arc<dyn Storage>,
        file: &str,
    ) -> Result<(u64, usize)> {
        match self {
            Self::Flat => FlatSegmentedBackend::read_segment_header(storage, file),
            Self::Hnsw => HnswSegmentedBackend::read_segment_header(storage, file),
            Self::Ivf => IvfSegmentedBackend::read_segment_header(storage, file),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(layouts: &[SegmentFileLayout]) -> Vec<&'static str> {
        layouts.iter().map(kind_of_layout).collect()
    }

    /// Flat, HNSW and IVF read each other's segments, and MultiVector and a
    /// kind no built-in index has read only their own.
    #[test]
    fn an_index_reads_the_kinds_of_its_family() {
        for own in SegmentKind::ALL {
            let mut read = kinds(&layouts_read_with(&own.layout()));
            assert_eq!(read[0], own.name());
            read.sort_unstable();
            assert_eq!(read, ["flat", "hnsw", "ivf"]);
            for other in SegmentKind::ALL {
                assert!(reads(own.name(), other.name()), "{own:?}, {other:?}");
            }
            assert!(!reads(own.name(), "mv"), "{own:?}");
            assert!(!reads("mv", own.name()), "{own:?}");
            assert!(!reads(own.name(), "later"), "{own:?}");
        }
        let multi_vector = crate::vector::index::multivector::LAYOUT;
        assert_eq!(kinds(&layouts_read_with(&multi_vector)), ["mv"]);
        assert!(reads("mv", "mv"));
    }

    /// A segment of a kind no built-in index has is an error that names
    /// both.
    #[test]
    fn a_segment_of_an_unknown_kind_is_an_error() {
        for kind in SegmentKind::ALL {
            assert_eq!(
                SegmentKind::of_segment("segment_000003", kind.name()).unwrap(),
                kind
            );
        }
        for unknown in ["mv", "later"] {
            let error = SegmentKind::of_segment("segment_000003", unknown)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("segment_000003") && error.contains(&format!("'{unknown}'")),
                "{error}"
            );
        }
    }
}
