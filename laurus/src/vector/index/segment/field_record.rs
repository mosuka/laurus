//! What a vector field's segments were written with (Issue #1547).
//!
//! A field's segments can only be read under the index kind, dimension and
//! distance they were written with. Under another dimension every search
//! panics or fails, under another kind it finds no segment file, and under
//! another distance it scores against a graph and vectors built for the old
//! one. A schema can still change between two opens: the CLI, the server
//! and the bindings read the schema from `schema.toml`, which a user can
//! edit, and `Engine::new` opens the field under whatever schema it gets.
//! [`FieldRecord`] keeps the three in the field's `segments.json`, and
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

use super::manager::{SegmentFileLayout, SegmentManager};

/// The primary segment file suffix of each index kind, with the name a
/// schema gives the kind.
const KINDS: [(&str, &str); 4] = [
    (crate::vector::index::flat::segment::LAYOUT.primary, "Flat"),
    (crate::vector::index::hnsw::segment::LAYOUT.primary, "Hnsw"),
    (crate::vector::index::ivf::segment::LAYOUT.primary, "Ivf"),
    (
        crate::vector::index::multivector::LAYOUT.primary,
        "MultiVector",
    ),
];

/// The options a vector field's segments were written with that they cannot
/// be read under once changed, as recorded in `segments.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FieldRecord {
    /// The index kind, as the primary file suffix of its segments without
    /// the dot: `flat`, `hnsw`, `ivf` or `mv`.
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
            kind: layout.primary.trim_start_matches('.').to_string(),
            dimension,
            distance,
        }
    }

    /// The primary file suffix of the kind, with its dot.
    fn primary_suffix(&self) -> String {
        format!(".{}", self.kind)
    }
}

/// The schema name of the kind whose primary file suffix, without its dot,
/// is `kind`, or `kind` itself for a suffix no built-in kind uses.
fn kind_name(kind: &str) -> &str {
    KINDS
        .iter()
        .find(|(suffix, _)| suffix.trim_start_matches('.') == kind)
        .map_or(kind, |(_, name)| name)
}

/// Join `parts` as "a", "a and b" or "a, b and c".
fn join(parts: &[String]) -> String {
    match parts {
        [] => String::new(),
        [only] => only.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// Why a field whose segments were written as `recorded` cannot be opened
/// as `configured`, if it cannot.
fn mismatch(recorded: &FieldRecord, configured: &FieldRecord) -> Option<String> {
    let mut changed = Vec::new();
    if recorded.kind != configured.kind {
        changed.push(format!("kind {}", kind_name(&configured.kind)));
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
        "its files were written with kind {}, dimension {} and distance {:?}, so it \
         cannot be opened with {}",
        kind_name(&recorded.kind),
        recorded.dimension,
        recorded.distance,
        join(&changed)
    ))
}

/// Why the segments of a field that has no record cannot be opened as
/// `configured`, if their files prove it (an index written before Issue
/// #1547, or a manifest an older build saved again).
///
/// Only evidence fails the open: a segment whose primary file is missing
/// while another kind's primary file exists under its id, or a segment
/// header that reads and gives another dimension. A segment with neither
/// file, or whose header does not read, is left to fail where it is read,
/// as before. The distance is recorded nowhere else, so it is accepted.
fn unrecorded_mismatch(
    manager: &SegmentManager,
    storage: &dyn Storage,
    configured: &FieldRecord,
    read_dimension: impl Fn(&str) -> Result<usize>,
) -> Option<String> {
    let primary_suffix = configured.primary_suffix();
    for segment in manager.list_segments() {
        let primary = format!("{}{primary_suffix}", segment.segment_id);
        if !storage.file_exists(&primary) {
            let written_as = KINDS.iter().find(|(suffix, _)| {
                *suffix != primary_suffix
                    && storage.file_exists(&format!("{}{suffix}", segment.segment_id))
            });
            if let Some((_, name)) = written_as {
                return Some(format!(
                    "its files were written with kind {name}, so it cannot be opened with \
                     kind {}",
                    kind_name(&configured.kind)
                ));
            }
            continue;
        }
        match read_dimension(&primary) {
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
/// with a record must match it in kind, dimension and distance. A field
/// with segments but no record is checked against what its files prove;
/// see [`unrecorded_mismatch`]. `read_dimension` reads the dimension from
/// the header of a primary segment file.
///
/// # Errors
///
/// Returns [`LaurusError::InvalidArgument`] when the field's segments were
/// written with another kind, dimension or distance.
pub(crate) fn check_field_record(
    manager: &SegmentManager,
    storage: &dyn Storage,
    configured: FieldRecord,
    read_dimension: impl Fn(&str) -> Result<usize>,
) -> Result<()> {
    if !manager.list_segments().is_empty() {
        let reason = match manager.field_record() {
            Some(recorded) => mismatch(&recorded, &configured),
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

        check_field_record(&manager, storage.as_ref(), configured.clone(), |_| {
            unreachable!("no segment header is read")
        })
        .unwrap();
        assert_eq!(manager.field_record(), Some(configured));
    }

    /// A recorded field opens only under the kind, dimension and distance it
    /// records, and the message names what was recorded and what changed.
    #[test]
    fn a_recorded_field_opens_only_under_its_record() {
        let storage = memory_storage();
        let recorded = hnsw(8, DistanceMetric::Cosine);
        let cases = [
            (
                FieldRecord::new(&FLAT, 8, DistanceMetric::Cosine),
                "kind Flat",
            ),
            (hnsw(4, DistanceMetric::Cosine), "dimension 4"),
            (hnsw(8, DistanceMetric::Euclidean), "distance Euclidean"),
            (
                FieldRecord::new(&FLAT, 4, DistanceMetric::DotProduct),
                "kind Flat, dimension 4 and distance DotProduct",
            ),
        ];
        for (configured, changed) in cases {
            let manager = new_manager(&storage, Some(recorded.clone()), true);
            let reason = rejection(check_field_record(
                &manager,
                storage.as_ref(),
                configured,
                |_| unreachable!("a recorded field reads no header"),
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

        let manager = new_manager(&storage, Some(recorded.clone()), true);
        check_field_record(&manager, storage.as_ref(), recorded.clone(), |_| {
            unreachable!("a recorded field reads no header")
        })
        .unwrap();
    }

    /// A field with segments but no record fails only on what its files
    /// prove, and otherwise records the config.
    #[test]
    fn an_unrecorded_field_fails_only_on_evidence() {
        let configured = hnsw(8, DistanceMetric::Cosine);

        // Another kind's primary file under the segment's id.
        let storage = memory_storage();
        let manager = new_manager(&storage, None, true);
        write_file(&storage, "segment_000000.flat");
        let reason = rejection(check_field_record(
            &manager,
            storage.as_ref(),
            configured.clone(),
            |_| unreachable!("the primary file is missing"),
        ));
        assert_eq!(
            reason,
            "its files were written with kind Flat, so it cannot be opened with kind Hnsw"
        );
        assert_eq!(manager.field_record(), None);

        // A header that reads and gives another dimension.
        let storage = memory_storage();
        let manager = new_manager(&storage, None, true);
        write_file(&storage, "segment_000000.hnsw");
        let reason = rejection(check_field_record(
            &manager,
            storage.as_ref(),
            configured.clone(),
            |file| {
                assert_eq!(file, "segment_000000.hnsw");
                Ok(4)
            },
        ));
        assert_eq!(
            reason,
            "its files were written with dimension 4, so it cannot be opened with dimension 8"
        );

        // No evidence: no file of any kind, a header that does not read, or
        // a matching header (whatever the distance, which no file records).
        type ReadDimension = fn(&str) -> Result<usize>;
        let unreadable: ReadDimension = |_| Err(LaurusError::index("truncated header"));
        let matching: ReadDimension = |_| Ok(8);
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
