//! What a vector field's segments were written with (Issue #1547).
//!
//! A field's segments can only be read under the dimension and distance they
//! were written with, and under an index kind that reads theirs: Flat, HNSW
//! and IVF read each other's segments (Issue #1579), and MultiVector only
//! its own. Under another dimension every search panics or fails, and under
//! another distance it scores against a graph and vectors built for the old
//! one. A schema can still change between two opens: the CLI, the server
//! and the bindings read the schema from `schema.toml`, which a user can
//! edit, and `Engine::new` opens the field under whatever schema it gets.
//! [`FieldRecord`] keeps the kind, the dimension and the distance in the
//! field's `segments.json`, beside the kind of each segment, and
//! [`check_field_record`] fails the open when the config disagrees.
//!
//! Every other option may change between opens and is left out on purpose:
//! the quantizer and its parameters, HNSW `m` and `ef_construction`, IVF
//! `n_clusters` and `n_probe`, the rerank storage, the PQ codebook path,
//! the multi-vector storage kind and pool factor, and the embedder. Each
//! segment records the quantizer it was written with and readers dispatch
//! on it, and `update_field` rebuilds a field under such a change by
//! opening the new config over the existing segments.

use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};
use crate::storage::Storage;
use crate::vector::core::distance::DistanceMetric;

use super::kind::{SegmentKind, kind_of_layout, reads, schema_name};
use super::manager::{SegmentFileLayout, SegmentManager};

/// The options a vector field's segments were written with that they cannot
/// be read under once changed, as recorded in `segments.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FieldRecord {
    /// The index kind the field was last opened as, as the primary file
    /// suffix of its segments without the dot: `flat`, `hnsw`, `ivf` or
    /// `mv`. Each segment records its own.
    pub(crate) kind: String,
    /// The vector dimension.
    pub(crate) dimension: usize,
    /// The distance metric.
    pub(crate) distance: DistanceMetric,
}

impl FieldRecord {
    /// The record of a field whose segments use `layout`, with `dimension`
    /// and `distance`.
    pub(crate) fn new(
        layout: &SegmentFileLayout,
        dimension: usize,
        distance: DistanceMetric,
    ) -> Self {
        Self {
            kind: kind_of_layout(layout).to_string(),
            dimension,
            distance,
        }
    }
}

/// Join `parts` as "a", "a and b" or "a, b and c".
fn join(parts: &[String]) -> String {
    match parts {
        [] => String::new(),
        [only] => only.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// The schema names of the kinds `manager`'s segments were written as, as
/// "kind A" or "kinds A and B".
fn written_kinds(manager: &SegmentManager) -> String {
    let mut names: Vec<String> = manager
        .list_segments()
        .iter()
        .map(|segment| schema_name(manager.kind_of(segment)).to_string())
        .collect();
    names.sort();
    names.dedup();
    let noun = if names.len() == 1 { "kind" } else { "kinds" };
    format!("{noun} {}", join(&names))
}

/// Why a field with record `recorded` cannot be opened as `configured`, if
/// it cannot: a segment of a kind that `configured`'s does not read, another
/// dimension, or another distance.
fn recorded_mismatch(
    manager: &SegmentManager,
    recorded: &FieldRecord,
    configured: &FieldRecord,
) -> Option<String> {
    let mut changed = Vec::new();
    let segments = manager.list_segments();
    if segments
        .iter()
        .any(|segment| !reads(&configured.kind, manager.kind_of(segment)))
    {
        changed.push(format!("kind {}", schema_name(&configured.kind)));
    }
    if recorded.dimension != configured.dimension {
        changed.push(format!("dimension {}", configured.dimension));
    }
    if recorded.distance != configured.distance {
        changed.push(format!("distance {:?}", configured.distance));
    }
    if changed.is_empty() {
        return None;
    }
    Some(format!(
        "its files were written with {}, dimension {} and distance {:?}, so it \
         cannot be opened with {}",
        written_kinds(manager),
        recorded.dimension,
        recorded.distance,
        join(&changed)
    ))
}

/// Why the segments of a field that has no record cannot be opened as
/// `configured`, if their files prove it (an index written before Issue
/// #1547, or a manifest an older build saved again).
///
/// Only evidence fails the open: a segment of a kind that `configured`'s
/// does not read, or a segment header that reads and gives another
/// dimension. A segment with no file, or whose header does not read, is left
/// to fail where it is read, as before. The distance is recorded nowhere
/// else, so it is accepted.
fn unrecorded_mismatch(
    manager: &SegmentManager,
    storage: &dyn Storage,
    configured: &FieldRecord,
    read_dimension: impl Fn(&str, &str) -> Result<usize>,
) -> Option<String> {
    for segment in manager.list_segments() {
        let kind = manager.kind_of(&segment);
        let exists = |kind: &str| storage.file_exists(&format!("{}.{kind}", segment.segment_id));
        // The manager infers a kind that its index reads, so the file of
        // one that it does not read shows only here.
        let multi_vector = kind_of_layout(&crate::vector::index::multivector::LAYOUT);
        let written_as = if exists(kind) {
            Some(kind)
        } else {
            SegmentKind::ALL
                .into_iter()
                .map(SegmentKind::name)
                .chain([multi_vector])
                .find(|other| exists(other))
        };
        let Some(written_as) = written_as else {
            continue;
        };
        if !reads(&configured.kind, written_as) {
            return Some(format!(
                "its files were written with kind {}, so it cannot be opened with kind {}",
                schema_name(written_as),
                schema_name(&configured.kind)
            ));
        }
        let primary = format!("{}.{written_as}", segment.segment_id);
        match read_dimension(written_as, &primary) {
            Ok(dimension) if dimension != configured.dimension => {
                return Some(format!(
                    "its files were written with dimension {dimension}, so it cannot be \
                     opened with dimension {}",
                    configured.dimension
                ));
            }
            Ok(_) => {}
            Err(e) => log::warn!(
                "cannot read the dimension of vector segment '{primary}' to check it \
                 against the configured {}: {e}",
                configured.dimension
            ),
        }
    }
    None
}

/// Check that a field's segments can be read as `configured`, and record
/// `configured` for the next manifest save (Issue #1547).
///
/// A field with no segments takes any config: nothing written depends on
/// its options, and every commit saves a manifest for every field, so the
/// options of a field that holds no vector yet can still be fixed. A field
/// with a record must match it in dimension and distance, and each of its
/// segments must be of a kind that `configured`'s reads. A field with
/// segments but no record is checked against what its files prove; see
/// [`unrecorded_mismatch`]. `read_dimension` reads the dimension from the
/// header of a primary segment file, given the kind of the segment and the
/// file.
///
/// # Errors
///
/// Returns [`LaurusError::InvalidArgument`] when the field's segments were
/// written with another dimension or distance, or with a kind that
/// `configured`'s does not read.
pub(crate) fn check_field_record(
    manager: &SegmentManager,
    storage: &dyn Storage,
    configured: FieldRecord,
    read_dimension: impl Fn(&str, &str) -> Result<usize>,
) -> Result<()> {
    if !manager.list_segments().is_empty() {
        let reason = match manager.field_record() {
            Some(recorded) => recorded_mismatch(manager, &recorded, &configured),
            None => unrecorded_mismatch(manager, storage, &configured, read_dimension),
        };
        if let Some(reason) = reason {
            return Err(LaurusError::invalid_argument(reason));
        }
    }
    manager.set_field_record(configured);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Arc;

    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::index::segment::manager::{ManagedSegmentInfo, SegmentManagerConfig};

    const HNSW: SegmentFileLayout = crate::vector::index::hnsw::segment::LAYOUT;
    const FLAT: SegmentFileLayout = crate::vector::index::flat::segment::LAYOUT;
    const IVF: SegmentFileLayout = crate::vector::index::ivf::segment::LAYOUT;
    const MULTI_VECTOR: SegmentFileLayout = crate::vector::index::multivector::LAYOUT;

    fn hnsw(dimension: usize, distance: DistanceMetric) -> FieldRecord {
        FieldRecord::new(&HNSW, dimension, distance)
    }

    /// An HNSW manager over `storage` with `recorded` as its record and, if
    /// `segment` is set, segment `segment_000000` registered. Write the
    /// segment's files after this: creating the manager sweeps the HNSW
    /// files of segments not yet registered.
    fn new_manager(
        storage: &Arc<dyn Storage>,
        recorded: Option<FieldRecord>,
        segment: bool,
    ) -> SegmentManager {
        let manager =
            SegmentManager::new(SegmentManagerConfig::default(), storage.clone(), HNSW).unwrap();
        if let Some(recorded) = recorded {
            manager.set_field_record(recorded);
        }
        if segment {
            manager
                .add_segment(ManagedSegmentInfo::new("segment_000000".into(), 2, 0, 0))
                .unwrap();
        }
        manager
    }

    fn memory_storage() -> Arc<dyn Storage> {
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
    }

    fn write_file(storage: &Arc<dyn Storage>, name: &str) {
        let mut output = storage.create_output(name).unwrap();
        output.write_all(b"segment").unwrap();
        output.close().unwrap();
    }

    fn rejection(result: Result<()>) -> String {
        match result {
            Err(LaurusError::InvalidArgument(reason)) => reason,
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// A field with no segment opens under any config, which it records.
    #[test]
    fn a_field_with_no_segments_takes_any_config() {
        let storage = memory_storage();
        let manager = new_manager(&storage, Some(hnsw(8, DistanceMetric::Cosine)), false);
        let configured = FieldRecord::new(&FLAT, 4, DistanceMetric::Euclidean);

        check_field_record(&manager, storage.as_ref(), configured.clone(), |_, _| {
            unreachable!("no segment header is read")
        })
        .unwrap();
        assert_eq!(manager.field_record(), Some(configured));
    }

    /// A recorded field opens only under the dimension and distance it
    /// records, and a kind that reads its segments: Flat, HNSW and IVF read
    /// each other's (Issue #1579). The message names what was recorded and
    /// what changed.
    #[test]
    fn a_recorded_field_opens_only_under_its_record() {
        let storage = memory_storage();
        let recorded = hnsw(8, DistanceMetric::Cosine);
        let cases = [
            (
                FieldRecord::new(&MULTI_VECTOR, 8, DistanceMetric::Cosine),
                "kind MultiVector",
            ),
            (hnsw(4, DistanceMetric::Cosine), "dimension 4"),
            (
                FieldRecord::new(&FLAT, 4, DistanceMetric::Cosine),
                "dimension 4",
            ),
            (hnsw(8, DistanceMetric::Euclidean), "distance Euclidean"),
            (
                FieldRecord::new(&FLAT, 4, DistanceMetric::DotProduct),
                "dimension 4 and distance DotProduct",
            ),
            (
                FieldRecord::new(&MULTI_VECTOR, 4, DistanceMetric::DotProduct),
                "kind MultiVector, dimension 4 and distance DotProduct",
            ),
        ];
        for (configured, changed) in cases {
            let manager = new_manager(&storage, Some(recorded.clone()), true);
            let reason = rejection(check_field_record(
                &manager,
                storage.as_ref(),
                configured,
                |_, _| unreachable!("a recorded field reads no header"),
            ));
            assert_eq!(
                reason,
                format!(
                    "its files were written with kind Hnsw, dimension 8 and distance Cosine, \
                     so it cannot be opened with {changed}"
                )
            );
            assert_eq!(manager.field_record(), Some(recorded.clone()));
        }

        for layout in [HNSW, FLAT, IVF] {
            let configured = FieldRecord::new(&layout, 8, DistanceMetric::Cosine);
            let manager = new_manager(&storage, Some(recorded.clone()), true);
            check_field_record(&manager, storage.as_ref(), configured.clone(), |_, _| {
                unreachable!("a recorded field reads no header")
            })
            .unwrap();
            assert_eq!(manager.field_record(), Some(configured));
        }
    }

    /// The message names every kind the segments were written as, whatever
    /// the field was last opened as.
    #[test]
    fn the_reason_names_every_kind_of_the_segments() {
        let storage = memory_storage();
        let manager = new_manager(&storage, Some(hnsw(8, DistanceMetric::Cosine)), true);
        let mut flat = ManagedSegmentInfo::new("segment_000001".into(), 2, 0, 0);
        flat.kind = Some("flat".to_string());
        manager.add_segment(flat).unwrap();
        let reason = rejection(check_field_record(
            &manager,
            storage.as_ref(),
            FieldRecord::new(&MULTI_VECTOR, 8, DistanceMetric::Cosine),
            |_, _| unreachable!("a recorded field reads no header"),
        ));
        assert_eq!(
            reason,
            "its files were written with kinds Flat and Hnsw, dimension 8 and distance Cosine, \
             so it cannot be opened with kind MultiVector"
        );
    }

    /// A field with segments but no record fails only on what its files
    /// prove, and otherwise records the config.
    #[test]
    fn an_unrecorded_field_fails_only_on_evidence() {
        let configured = hnsw(8, DistanceMetric::Cosine);

        // The primary file of a kind that HNSW does not read, under the
        // segment's id.
        let storage = memory_storage();
        let manager = new_manager(&storage, None, true);
        write_file(&storage, "segment_000000.mv");
        let reason = rejection(check_field_record(
            &manager,
            storage.as_ref(),
            configured.clone(),
            |_, _| unreachable!("the kind is rejected first"),
        ));
        assert_eq!(
            reason,
            "its files were written with kind MultiVector, so it cannot be opened with kind Hnsw"
        );
        assert_eq!(manager.field_record(), None);

        // The other way around.
        let storage = memory_storage();
        let manager = SegmentManager::new(
            SegmentManagerConfig::default(),
            storage.clone(),
            MULTI_VECTOR,
        )
        .unwrap();
        manager
            .add_segment(ManagedSegmentInfo::new("segment_000000".into(), 2, 0, 0))
            .unwrap();
        write_file(&storage, "segment_000000.ivf");
        let reason = rejection(check_field_record(
            &manager,
            storage.as_ref(),
            FieldRecord::new(&MULTI_VECTOR, 8, DistanceMetric::Cosine),
            |_, _| unreachable!("the kind is rejected first"),
        ));
        assert_eq!(
            reason,
            "its files were written with kind Ivf, so it cannot be opened with kind MultiVector"
        );

        // A header that reads and gives another dimension, read as the kind
        // the segment was written as.
        for kind in ["hnsw", "flat", "ivf"] {
            let storage = memory_storage();
            let manager = new_manager(&storage, None, false);
            let mut segment = ManagedSegmentInfo::new("segment_000000".into(), 2, 0, 0);
            segment.kind = Some(kind.to_string());
            manager.add_segment(segment).unwrap();
            let file = format!("segment_000000.{kind}");
            write_file(&storage, &file);
            let reason = rejection(check_field_record(
                &manager,
                storage.as_ref(),
                configured.clone(),
                |read_as, read| {
                    assert_eq!((read_as, read), (kind, file.as_str()));
                    Ok(4)
                },
            ));
            assert_eq!(
                reason,
                "its files were written with dimension 4, so it cannot be opened with dimension 8"
            );
        }

        // No evidence: no file of any kind, a header that does not read, or
        // a matching header (whatever the distance, which no file records).
        type ReadDimension = fn(&str, &str) -> Result<usize>;
        let unreadable: ReadDimension = |_, _| Err(LaurusError::index("truncated header"));
        let matching: ReadDimension = |_, _| Ok(8);
        for (file, read_dimension) in [
            (None, matching),
            (Some("segment_000000.hnsw"), unreadable),
            (Some("segment_000000.hnsw"), matching),
        ] {
            let storage = memory_storage();
            let manager = new_manager(&storage, None, true);
            if let Some(file) = file {
                write_file(&storage, file);
            }
            let configured = hnsw(8, DistanceMetric::DotProduct);
            check_field_record(
                &manager,
                storage.as_ref(),
                configured.clone(),
                read_dimension,
            )
            .unwrap();
            assert_eq!(manager.field_record(), Some(configured));
        }
    }
}
