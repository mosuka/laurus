//! A writer's buffered BKD points, held as per-field flat columns
//! (Issue #1165).
//!
//! Each field keeps its points as one row-major `Vec<f64>` plus a parallel
//! `Vec<u32>` of the buffered entry each point came from, appended in
//! arrival order. That costs `8 * dims + 4` bytes per point, against a
//! per-document `AHashMap<String, Vec<Vec<f64>>>` whose fixed costs came to
//! about 214 bytes per point for a single-valued field. It also lets the
//! flush hand the columns to
//! [`BKDWriter`](crate::lexical::index::structures::bkd_tree::BKDWriter) as
//! they are, instead of first copying every point into a second buffer in
//! doc-id order.
//!
//! A point is tagged with its entry's sequence number rather than its doc
//! id, so removing or replacing a buffered document never touches the
//! columns: its entry leaves the writer's buffer, and the flush skips every
//! point whose entry is gone. Rewriting the columns on each removal instead
//! cost O(points) per re-upsert. The flush also resolves the sequence
//! numbers to doc ids, which a column never stores.

use std::collections::BTreeMap;

use crate::error::{LaurusError, Result};

/// One field's buffered points of a single dimensionality, in arrival
/// order.
#[derive(Debug)]
pub(crate) struct PointColumn {
    dims: usize,
    /// `dims` coordinates per point, row-major.
    values: Vec<f64>,
    /// The sequence number of the buffered entry each point came from.
    seqs: Vec<u32>,
}

impl PointColumn {
    fn new(dims: usize) -> Self {
        PointColumn {
            dims,
            values: Vec::new(),
            seqs: Vec::new(),
        }
    }

    /// Coordinates per point.
    pub(crate) fn dims(&self) -> usize {
        self.dims
    }

    /// Every point's coordinates, `dims` per point, in arrival order.
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }

    /// Every point's entry sequence number, parallel to [`Self::values`].
    pub(crate) fn seqs(&self) -> &[u32] {
        &self.seqs
    }

    fn heap_bytes(&self) -> usize {
        self.values.capacity() * std::mem::size_of::<f64>()
            + self.seqs.capacity() * std::mem::size_of::<u32>()
    }
}

/// Per-field point columns, by field name.
#[derive(Debug, Default)]
pub(crate) struct PointColumns {
    /// One column per dimensionality a field's points have had. A
    /// well-formed field has exactly one; see [`Self::flush_columns`].
    fields: BTreeMap<String, Vec<PointColumn>>,
}

impl PointColumns {
    /// Approximate per-field overhead beyond the columns' own buffers: the
    /// map entry and the column list.
    const FIELD_OVERHEAD: usize = 64;

    /// Append the points for `field` of the buffered entry numbered `seq`,
    /// in order. An empty list adds nothing, so a field whose documents
    /// never carry a point has no column and writes no `.bkd`.
    pub(crate) fn append(&mut self, field: String, seq: u32, points: Vec<Vec<f64>>) {
        if points.is_empty() {
            return;
        }
        let columns = self.fields.entry(field).or_default();
        for point in points {
            let index = match columns.iter().position(|c| c.dims == point.len()) {
                Some(index) => index,
                None => {
                    columns.push(PointColumn::new(point.len()));
                    columns.len() - 1
                }
            };
            let column = &mut columns[index];
            column.values.extend_from_slice(&point);
            column.seqs.push(seq);
        }
    }

    /// Bytes the columns hold, counted by capacity: what the buffers
    /// actually occupy, including points of entries removed since the
    /// columns were last dropped.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.fields
            .iter()
            .map(|(name, columns)| {
                Self::FIELD_OVERHEAD
                    + name.capacity()
                    + columns.capacity() * std::mem::size_of::<PointColumn>()
                    + columns.iter().map(PointColumn::heap_bytes).sum::<usize>()
            })
            .sum()
    }

    /// The column to write for each field, by field name: the one holding
    /// points of live entries (those for which `live(seq)` holds). A field
    /// with no live point is left out, so it writes no `.bkd`.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::index`] when a field has live points of more
    /// than one dimensionality: a BKD tree has a single one, and there is
    /// no right way to choose.
    pub(crate) fn flush_columns(
        &self,
        live: impl Fn(u32) -> bool,
    ) -> Result<Vec<(&str, &PointColumn)>> {
        let mut out = Vec::with_capacity(self.fields.len());
        for (name, columns) in &self.fields {
            let mut with_live = columns
                .iter()
                .filter(|c| c.seqs.iter().any(|&seq| live(seq)));
            let Some(column) = with_live.next() else {
                continue;
            };
            if let Some(other) = with_live.next() {
                return Err(LaurusError::index(format!(
                    "Field '{name}' has buffered points of {} and {} dimensions; \
                     a BKD field needs a single dimensionality",
                    column.dims, other.dims
                )));
            }
            out.push((name.as_str(), column));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Snapshot = Vec<(String, usize, Vec<f64>, Vec<u32>)>;

    fn snapshot(columns: &PointColumns, live: impl Fn(u32) -> bool) -> Snapshot {
        columns
            .flush_columns(live)
            .unwrap()
            .into_iter()
            .map(|(name, c)| {
                (
                    name.to_string(),
                    c.dims(),
                    c.values().to_vec(),
                    c.seqs().to_vec(),
                )
            })
            .collect()
    }

    fn all(_: u32) -> bool {
        true
    }

    #[test]
    fn append_keeps_arrival_order_across_entries_and_within_one() {
        let mut columns = PointColumns::default();
        columns.append("n".into(), 0, vec![vec![3.0], vec![1.0]]);
        columns.append("n".into(), 1, vec![vec![5.0]]);
        columns.append("loc".into(), 0, vec![vec![1.0, 2.0]]);

        assert_eq!(
            snapshot(&columns, all),
            vec![
                ("loc".to_string(), 2, vec![1.0, 2.0], vec![0]),
                ("n".to_string(), 1, vec![3.0, 1.0, 5.0], vec![0, 0, 1]),
            ]
        );
    }

    #[test]
    fn an_empty_point_list_creates_no_column() {
        let mut columns = PointColumns::default();
        columns.append("tags".into(), 0, Vec::new());

        assert!(snapshot(&columns, all).is_empty());
        assert_eq!(columns.heap_bytes(), 0);
    }

    #[test]
    fn a_field_without_live_points_is_left_out() {
        let mut columns = PointColumns::default();
        columns.append("gone".into(), 0, vec![vec![1.0]]);
        columns.append("kept".into(), 1, vec![vec![2.0]]);

        assert_eq!(
            snapshot(&columns, |seq| seq != 0),
            vec![("kept".to_string(), 1, vec![2.0], vec![1])]
        );
    }

    #[test]
    fn flush_columns_rejects_a_field_with_two_live_dimensionalities() {
        let mut columns = PointColumns::default();
        columns.append("x".into(), 0, vec![vec![1.0]]);
        columns.append("x".into(), 1, vec![vec![1.0, 2.0]]);

        let err = columns.flush_columns(all).unwrap_err().to_string();
        assert!(err.contains("'x'"), "{err}");
        assert!(err.contains("1 and 2 dimensions"), "{err}");

        // Once the other dimensionality's only entry is gone, the field
        // writes.
        assert_eq!(
            snapshot(&columns, |seq| seq != 1),
            vec![("x".to_string(), 1, vec![1.0], vec![0])]
        );
    }

    #[test]
    fn heap_bytes_covers_every_buffered_point() {
        let mut columns = PointColumns::default();
        for seq in 0..100 {
            columns.append("loc".into(), seq, vec![vec![seq as f64, -(seq as f64)]]);
        }
        let floor = 100 * (2 * std::mem::size_of::<f64>() + std::mem::size_of::<u32>());
        let bytes = columns.heap_bytes();
        assert!(bytes >= floor, "{bytes} < {floor}");
    }
}
