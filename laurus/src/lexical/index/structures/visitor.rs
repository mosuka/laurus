//! Visitor abstraction for walking a BKD tree with subtree pruning.
//!
//! A BKD reader can be queried in two flavours:
//!
//! - The legacy [`BKDTree::range_search`] returns a flat `Vec<u64>` of doc
//!   ids whose points fall inside an axis-aligned range. It is implemented
//!   on top of `intersect` via [`RangeQueryVisitor`].
//! - The new [`BKDTree::intersect`] takes any `&mut dyn IntersectVisitor`
//!   and lets the caller decide what to do with each cell (Inside / Outside
//!   / Crosses) and each candidate point. This is the primitive 3D distance,
//!   k-NN, and bounding-box queries are built on (#300 onwards).
//!
//! [`BKDTree::range_search`]: super::bkd_tree::BKDTree::range_search
//! [`BKDTree::intersect`]: super::bkd_tree::BKDTree::intersect

use super::aabb::AABB;

/// Relationship between a BKD subtree's bounding box and the query region.
///
/// The reader uses this to prune entire subtrees: an `Inside` cell can be
/// collected without per-point checks, an `Outside` cell can be skipped
/// entirely, and a `Crosses` cell forces recursion (or per-point filtering
/// at a leaf).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellRelation {
    /// The cell's AABB is entirely inside the query region — every point
    /// in the subtree is a hit.
    Inside,
    /// The cell's AABB is entirely outside the query region — the subtree
    /// can be skipped without inspection.
    Outside,
    /// The cell's AABB intersects the query boundary — recurse into the
    /// subtree (or filter the leaf per-point).
    Crosses,
}

/// Callback object passed to [`BKDTree::intersect`] to drive the traversal.
///
/// Implementors decide what counts as a hit and how to record it. The reader
/// invokes the methods in the order:
///
/// 1. `compare(cell)` for each subtree's AABB.
/// 2. If the relation is `Inside`, every doc beneath the subtree is reported
///    via `visit_inside(doc_id)` — no point coordinates passed, since the
///    visitor already knows the cell is fully inside the query.
/// 3. If the relation is `Crosses` and the subtree is a leaf, every doc in
///    the leaf is reported via `visit(doc_id, point)` and the visitor must
///    perform any final per-point filtering against the query.
/// 4. `Outside` cells are skipped entirely.
///
/// [`BKDTree::intersect`]: super::bkd_tree::BKDTree::intersect
pub trait IntersectVisitor {
    /// Classify a cell's AABB against the query region.
    ///
    /// Implementations should be conservative: returning `Crosses` for a
    /// truly Outside cell only loses pruning efficiency, but returning
    /// `Inside` / `Outside` incorrectly produces wrong results.
    fn compare(&self, cell: &AABB) -> CellRelation;

    /// Record a hit known to be inside the query region.
    ///
    /// Called for every doc id in a subtree whose AABB compared as `Inside`.
    /// The point coordinates are not provided because the visitor does not
    /// need them — the cell is known to be fully inside the query.
    fn visit_inside(&mut self, doc_id: u64);

    /// Filter and (optionally) record a candidate from a `Crosses` leaf.
    ///
    /// Called for every doc id in a leaf whose AABB compared as `Crosses`.
    /// The visitor must check whether `point` actually lies inside the query
    /// region before recording the hit.
    fn visit(&mut self, doc_id: u64, point: &[f64]);
}

/// Half-open / closed range visitor used to back the legacy
/// `BKDTree::range_search` API on top of the new `intersect` primitive.
///
/// The visitor accepts the same `mins` / `maxs` / `include_min` /
/// `include_max` shape that `range_search` callers were already passing,
/// converts unbounded `None` slots into `±INFINITY`, and handles the
/// inclusive-vs-exclusive boundary check itself.
pub struct RangeQueryVisitor {
    mins: Vec<f64>,
    maxs: Vec<f64>,
    include_min: bool,
    include_max: bool,
    hits: Vec<u64>,
}

impl RangeQueryVisitor {
    /// Build a visitor from the `range_search` parameter shape.
    ///
    /// `mins.len()` and `maxs.len()` must equal `num_dims`; otherwise the
    /// visitor would silently misbehave when the BKD reader supplies an
    /// AABB of a different dimensionality.
    pub fn new(
        mins: &[Option<f64>],
        maxs: &[Option<f64>],
        include_min: bool,
        include_max: bool,
    ) -> Self {
        let mins_f: Vec<f64> = mins
            .iter()
            .map(|m| m.unwrap_or(f64::NEG_INFINITY))
            .collect();
        let maxs_f: Vec<f64> = maxs.iter().map(|m| m.unwrap_or(f64::INFINITY)).collect();
        RangeQueryVisitor {
            mins: mins_f,
            maxs: maxs_f,
            include_min,
            include_max,
            hits: Vec::new(),
        }
    }

    /// Consume the visitor, returning the collected (and not yet sorted)
    /// doc ids. Callers that expect deduplicated, sorted output should
    /// post-process the returned vector.
    pub fn into_hits(self) -> Vec<u64> {
        self.hits
    }

    fn point_matches(&self, point: &[f64]) -> bool {
        point.iter().enumerate().all(|(d, &v)| {
            value_in_range(
                v,
                self.mins[d],
                self.maxs[d],
                self.include_min,
                self.include_max,
            )
        })
    }
}

/// Whether `value` lies within `[min, max]` on one axis, with
/// `include_min` and `include_max` deciding whether a value equal to a
/// bound matches.
///
/// This is the per-point check of [`RangeQueryVisitor`], and the
/// stored-document scan of
/// [`NumericRangeQuery`](crate::lexical::query::NumericRangeQuery), which
/// runs when a segment has no BKD tree for the field, uses it too, so the
/// two give the same documents (Issue #1600). Values are compared as `f64`:
/// a `NaN` value compares false with every bound and so is in no range,
/// and `-0.0` equals `0.0`. An open side is `f64::NEG_INFINITY` or
/// `f64::INFINITY`, as [`RangeQueryVisitor::new`] makes it.
///
/// # Arguments
///
/// * `value` - The value to test.
/// * `min` - The lower bound.
/// * `max` - The upper bound.
/// * `include_min` - Whether a value equal to `min` matches.
/// * `include_max` - Whether a value equal to `max` matches.
///
/// # Returns
///
/// `true` when `value` is within the range.
#[inline]
pub(crate) fn value_in_range(
    value: f64,
    min: f64,
    max: f64,
    include_min: bool,
    include_max: bool,
) -> bool {
    let lower_ok = if include_min {
        value >= min
    } else {
        value > min
    };
    let upper_ok = if include_max {
        value <= max
    } else {
        value < max
    };
    lower_ok && upper_ok
}

impl IntersectVisitor for RangeQueryVisitor {
    fn compare(&self, cell: &AABB) -> CellRelation {
        debug_assert_eq!(cell.num_dims(), self.mins.len());
        let cell_min = cell.min();
        let cell_max = cell.max();

        // Disjoint check: if any axis separates the cell from the query,
        // the cell is fully Outside. Treat exclusive boundaries as
        // disjoint when they touch (cell.max == query.min with `>` query).
        for d in 0..cell_min.len() {
            if cell_max[d] < self.mins[d] {
                return CellRelation::Outside;
            }
            if !self.include_min && cell_max[d] <= self.mins[d] {
                return CellRelation::Outside;
            }
            if cell_min[d] > self.maxs[d] {
                return CellRelation::Outside;
            }
            if !self.include_max && cell_min[d] >= self.maxs[d] {
                return CellRelation::Outside;
            }
        }

        // Containment check: every axis of the cell must be inside the
        // query (respecting inclusivity flags).
        for d in 0..cell_min.len() {
            let lower_ok = if self.include_min {
                cell_min[d] >= self.mins[d]
            } else {
                cell_min[d] > self.mins[d]
            };
            let upper_ok = if self.include_max {
                cell_max[d] <= self.maxs[d]
            } else {
                cell_max[d] < self.maxs[d]
            };
            if !(lower_ok && upper_ok) {
                return CellRelation::Crosses;
            }
        }
        CellRelation::Inside
    }

    fn visit_inside(&mut self, doc_id: u64) {
        self.hits.push(doc_id);
    }

    fn visit(&mut self, doc_id: u64, point: &[f64]) {
        if self.point_matches(point) {
            self.hits.push(doc_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(min: Vec<f64>, max: Vec<f64>) -> AABB {
        AABB::new(min, max).unwrap()
    }

    #[test]
    fn range_query_visitor_inside_outside_crosses() {
        // 1D query [10, 20] inclusive on both ends.
        let v = RangeQueryVisitor::new(&[Some(10.0)], &[Some(20.0)], true, true);
        assert_eq!(
            v.compare(&cell(vec![12.0], vec![18.0])),
            CellRelation::Inside
        );
        assert_eq!(
            v.compare(&cell(vec![25.0], vec![30.0])),
            CellRelation::Outside
        );
        assert_eq!(
            v.compare(&cell(vec![5.0], vec![8.0])),
            CellRelation::Outside
        );
        assert_eq!(
            v.compare(&cell(vec![15.0], vec![25.0])),
            CellRelation::Crosses
        );
        // Touching the inclusive boundary is Inside (single-point cell).
        assert_eq!(
            v.compare(&cell(vec![10.0], vec![10.0])),
            CellRelation::Inside
        );
    }

    #[test]
    fn range_query_visitor_exclusive_boundary_outside() {
        // Exclusive lower bound: query is x > 10.
        let v = RangeQueryVisitor::new(&[Some(10.0)], &[None], false, true);
        // A cell touching exactly at the exclusive lower bound is fully
        // Outside.
        assert_eq!(
            v.compare(&cell(vec![5.0], vec![10.0])),
            CellRelation::Outside
        );
        // A cell strictly above is Inside.
        assert_eq!(
            v.compare(&cell(vec![11.0], vec![20.0])),
            CellRelation::Inside
        );
        // A cell straddling the boundary is Crosses (visitor must filter
        // the boundary point per visit()).
        assert_eq!(
            v.compare(&cell(vec![10.0], vec![20.0])),
            CellRelation::Crosses
        );
    }

    #[test]
    fn range_query_visitor_unbounded_dimensions() {
        // 2D query: only x bounded, y unbounded.
        let v = RangeQueryVisitor::new(&[Some(0.0), None], &[Some(100.0), None], true, true);
        assert_eq!(
            v.compare(&cell(vec![10.0, -1e9], vec![90.0, 1e9])),
            CellRelation::Inside
        );
        assert_eq!(
            v.compare(&cell(vec![-50.0, 0.0], vec![-10.0, 10.0])),
            CellRelation::Outside
        );
    }

    #[test]
    fn range_query_visitor_visit_filters_points_on_crosses() {
        let mut v = RangeQueryVisitor::new(&[Some(10.0)], &[Some(20.0)], true, true);
        v.visit(1, &[5.0]); // outside
        v.visit(2, &[15.0]); // inside
        v.visit(3, &[20.0]); // boundary, inclusive
        v.visit(4, &[21.0]); // outside
        let hits = v.into_hits();
        assert_eq!(hits, vec![2, 3]);
    }

    #[test]
    fn range_query_visitor_visit_inside_skips_filter() {
        let mut v = RangeQueryVisitor::new(&[Some(10.0)], &[Some(20.0)], true, true);
        // visit_inside trusts the caller about the cell being Inside.
        v.visit_inside(7);
        v.visit_inside(8);
        let hits = v.into_hits();
        assert_eq!(hits, vec![7, 8]);
    }
}
