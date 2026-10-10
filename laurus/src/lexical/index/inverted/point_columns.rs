//! A writer's buffered BKD points, held as per-field flat columns
//! (Issue #1165).
//!
//! Each field keeps its points as one row-major `Vec<f64>` plus a parallel
//! `Vec<u64>` of doc ids, appended in arrival order. That costs
//! `8 * dims + 8` bytes per point, against a per-document
//! `AHashMap<String, Vec<Vec<f64>>>` whose fixed costs came to about 214
//! bytes per point for a single-valued field. It also lets the flush hand
//! the columns to [`BKDWriter`](crate::lexical::index::structures::bkd_tree::BKDWriter)
//! as they are, instead of first copying every point into a second buffer
//! in doc-id order.
//!
//! The columns hold every buffered entry's points, including duplicate
//! entries for one doc id, and drop a removed document's points only when
//! [`PointColumns::retain_docs`] is called, which the writer does where it
//! purges the same document's postings.

use std::collections::BTreeMap;

use crate::error::{LaurusError, Result};

/// One field's buffered points of a single dimensionality, in arrival
/// order.
#[derive(Debug)]
pub(crate) struct PointColumn {
    dims: usize,
    /// `dims` coordinates per point, row-major.
    values: Vec<f64>,
    /// The owning document of each point.
    doc_ids: Vec<u64>,
}

impl PointColumn {
    fn new(dims: usize) -> Self {
        PointColumn {
            dims,
            values: Vec::new(),
            doc_ids: Vec::new(),
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

    /// Every point's doc id, parallel to [`Self::values`].
    pub(crate) fn doc_ids(&self) -> &[u64] {
        &self.doc_ids
    }

    fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    fn heap_bytes(&self) -> usize {
        self.values.capacity() * std::mem::size_of::<f64>()
            + self.doc_ids.capacity() * std::mem::size_of::<u64>()
    }

    /// Keep the points whose doc id satisfies `keep`, in their order.
    /// Capacity is kept: the memory is not returned until the column is
    /// dropped, which is what the writer's estimate assumes.
    fn retain_docs(&mut self, keep: &mut impl FnMut(u64) -> bool) {
        let dims = self.dims;
        let mut kept = 0;
        for read in 0..self.doc_ids.len() {
            let doc_id = self.doc_ids[read];
            if !keep(doc_id) {
                continue;
            }
            if kept != read {
                self.doc_ids[kept] = doc_id;
                self.values
                    .copy_within(read * dims..(read + 1) * dims, kept * dims);
            }
            kept += 1;
        }
        self.doc_ids.truncate(kept);
        self.values.truncate(kept * dims);
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

    /// Append `doc_id`'s points for `field`, in order. An empty list adds
    /// nothing, so a field whose documents never carry a point has no
    /// column and writes no `.bkd`.
    pub(crate) fn append(&mut self, field: String, doc_id: u64, points: Vec<Vec<f64>>) {
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
            column.doc_ids.push(doc_id);
        }
    }

    /// Keep only the points whose doc id satisfies `keep`, preserving their
    /// order, and drop columns and fields left empty.
    pub(crate) fn retain_docs(&mut self, mut keep: impl FnMut(u64) -> bool) {
        self.fields.retain(|_, columns| {
            for column in columns.iter_mut() {
                column.retain_docs(&mut keep);
            }
            columns.retain(|c| !c.is_empty());
            !columns.is_empty()
        });
    }

    /// Bytes the columns hold, counted by capacity: what the buffers
    /// actually occupy, including points removed by [`Self::retain_docs`]
    /// since the last time the columns were dropped.
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

    /// The column to write for each field, by field name.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::index`] when a field has points of more than
    /// one dimensionality: a BKD tree has a single one, and there is no
    /// right way to choose.
    pub(crate) fn flush_columns(&self) -> Result<Vec<(&str, &PointColumn)>> {
        let mut out = Vec::with_capacity(self.fields.len());
        for (name, columns) in &self.fields {
            let mut live = columns.iter().filter(|c| !c.is_empty());
            let Some(column) = live.next() else {
                continue;
            };
            if let Some(other) = live.next() {
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

    fn snapshot(columns: &PointColumns) -> Vec<(String, usize, Vec<f64>, Vec<u64>)> {
        columns
            .flush_columns()
            .unwrap()
            .into_iter()
            .map(|(name, c)| {
                (
                    name.to_string(),
                    c.dims(),
                    c.values().to_vec(),
                    c.doc_ids().to_vec(),
                )
            })
            .collect()
    }

    #[test]
    fn append_keeps_arrival_order_across_documents_and_within_one() {
        let mut columns = PointColumns::default();
        columns.append("n".into(), 9, vec![vec![3.0], vec![1.0]]);
        columns.append("n".into(), 2, vec![vec![5.0]]);
        columns.append("loc".into(), 9, vec![vec![1.0, 2.0]]);

        assert_eq!(
            snapshot(&columns),
            vec![
                ("loc".to_string(), 2, vec![1.0, 2.0], vec![9]),
                ("n".to_string(), 1, vec![3.0, 1.0, 5.0], vec![9, 9, 2]),
            ]
        );
    }

    #[test]
    fn an_empty_point_list_creates_no_column() {
        let mut columns = PointColumns::default();
        columns.append("tags".into(), 1, Vec::new());

        assert!(snapshot(&columns).is_empty());
        assert_eq!(columns.heap_bytes(), 0);
    }

    #[test]
    fn retain_docs_compacts_multi_dimensional_points_in_order() {
        let mut columns = PointColumns::default();
        columns.append("p".into(), 1, vec![vec![1.0, 10.0]]);
        columns.append("p".into(), 2, vec![vec![2.0, 20.0], vec![2.5, 25.0]]);
        columns.append("p".into(), 3, vec![vec![3.0, 30.0]]);
        columns.append("p".into(), 2, vec![vec![4.0, 40.0]]);

        columns.retain_docs(|id| id != 2);

        assert_eq!(
            snapshot(&columns),
            vec![("p".to_string(), 2, vec![1.0, 10.0, 3.0, 30.0], vec![1, 3])]
        );
    }

    #[test]
    fn retain_docs_drops_a_field_left_without_points() {
        let mut columns = PointColumns::default();
        columns.append("gone".into(), 7, vec![vec![1.0]]);
        columns.append("kept".into(), 8, vec![vec![2.0]]);

        columns.retain_docs(|id| id != 7);

        assert_eq!(
            snapshot(&columns),
            vec![("kept".to_string(), 1, vec![2.0], vec![8])]
        );
    }

    #[test]
    fn flush_columns_rejects_a_field_with_two_live_dimensionalities() {
        let mut columns = PointColumns::default();
        columns.append("x".into(), 1, vec![vec![1.0]]);
        columns.append("x".into(), 2, vec![vec![1.0, 2.0]]);

        let err = columns.flush_columns().unwrap_err().to_string();
        assert!(err.contains("'x'"), "{err}");
        assert!(err.contains("1 and 2 dimensions"), "{err}");

        // Once the other dimensionality's points are gone, the field writes.
        columns.retain_docs(|id| id != 2);
        assert_eq!(
            snapshot(&columns),
            vec![("x".to_string(), 1, vec![1.0], vec![1])]
        );
    }

    #[test]
    fn heap_bytes_covers_every_buffered_point_and_survives_retain() {
        let mut columns = PointColumns::default();
        for id in 0..100 {
            columns.append("loc".into(), id, vec![vec![id as f64, -(id as f64)]]);
        }
        let floor = 100 * (2 * std::mem::size_of::<f64>() + std::mem::size_of::<u64>());
        let before = columns.heap_bytes();
        assert!(before >= floor, "{before} < {floor}");

        // Removed points still occupy the buffers until they are dropped.
        columns.retain_docs(|id| id < 10);
        assert_eq!(columns.heap_bytes(), before);
    }
}
