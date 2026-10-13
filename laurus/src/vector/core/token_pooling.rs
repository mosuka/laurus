//! Index-time token pooling for `MultiVector` fields (#1346).
//!
//! A late-interaction document keeps one vector per token. Pooling merges
//! similar token vectors into their mean, so a document keeps about
//! `1 / factor` of its rows, and both its storage and its MaxSim cost
//! shrink by about that factor.
//!
//! The method is the hierarchical pooling of Clavié et al. (arXiv
//! 2409.14683) as Sentence Transformers' `HierarchicalTokenPooling` and
//! PyLate (>= 1.5) implement it:
//!
//! 1. The first row (the `[CLS]` token of a ColBERT model) is kept as is.
//! 2. The remaining `r` rows are clustered into `k = max(r / factor, 1)`
//!    clusters by Ward agglomerative clustering on the cosine distance
//!    `1 - cos`. A document with `k >= r` is left unchanged.
//! 3. Each cluster is replaced by the mean of its rows, in order of the
//!    cluster's first row. In a `Cosine` field the means are
//!    re-normalized; on BEIR this ranked better than the raw means that
//!    Sentence Transformers and PyLate keep.
//!
//! The clustering reproduces SciPy's `linkage(condensed, method="ward")`:
//! the same nearest-neighbour chain, the same Lance-Williams update in the
//! same arithmetic order, and the clusters obtained by applying the
//! `r - k` lowest merges. The result is deterministic, so replaying a
//! document from the WAL pools it to the same rows.
//!
//! Cost per document: `O(r²)` time and memory (an `f64` condensed distance
//! matrix of `r (r - 1) / 2` entries).

use crate::vector::core::vector::normalize_slice;

/// Rows kept verbatim at the start of every document: the first token
/// (`[CLS]` for a ColBERT model), as Sentence Transformers and PyLate do.
pub(crate) const PROTECTED_ROWS: usize = 1;

/// Clusters the `rest` rows after the protected ones are pooled into.
fn cluster_count(rest: usize, factor: u32) -> usize {
    (rest / factor.max(1) as usize).max(1)
}

/// The number of rows a document of `rows` token vectors keeps after
/// pooling by `factor`: a pure function of the two, so a merge can write a
/// segment's document table before pooling the rows.
pub(crate) fn pooled_count(rows: usize, factor: u32) -> usize {
    let protected = rows.min(PROTECTED_ROWS);
    let rest = rows - protected;
    let clusters = cluster_count(rest, factor);
    if factor <= 1 || clusters >= rest {
        rows
    } else {
        protected + clusters
    }
}

/// Pool one document's token vectors (`rows`, row-major).
///
/// # Arguments
///
/// * `rows` - The document's token vectors, a whole number of rows.
/// * `dimension` - Length of every row.
/// * `factor` - Pooling factor; `<= 1` disables pooling.
/// * `renormalize` - L2-normalize every cluster mean (for a `Cosine`
///   field, whose rows are unit vectors).
///
/// # Returns
///
/// `None` when pooling leaves the document unchanged (see
/// [`pooled_count`]); otherwise the protected rows followed by one mean
/// per cluster, [`pooled_count`] rows in all.
pub(crate) fn pool(
    rows: &[f32],
    dimension: usize,
    factor: u32,
    renormalize: bool,
) -> Option<Vec<f32>> {
    debug_assert!(dimension > 0 && rows.len().is_multiple_of(dimension));
    let n = rows.len() / dimension;
    let protected = n.min(PROTECTED_ROWS);
    let rest = &rows[protected * dimension..];
    let clusters = cluster_count(n - protected, factor);
    if pooled_count(n, factor) == n {
        return None;
    }

    let labels = ward_labels(rest, dimension, clusters);
    let mut sums = vec![0f64; clusters * dimension];
    let mut counts = vec![0usize; clusters];
    for (row, &label) in rest.chunks_exact(dimension).zip(&labels) {
        for (sum, value) in sums[label * dimension..(label + 1) * dimension]
            .iter_mut()
            .zip(row)
        {
            *sum += f64::from(*value);
        }
        counts[label] += 1;
    }

    let mut pooled = Vec::with_capacity((protected + clusters) * dimension);
    pooled.extend_from_slice(&rows[..protected * dimension]);
    for (sum, &count) in sums.chunks_exact(dimension).zip(&counts) {
        let start = pooled.len();
        pooled.extend(sum.iter().map(|s| (s / count as f64) as f32));
        if renormalize {
            normalize_slice(&mut pooled[start..]);
        }
    }
    Some(pooled)
}

/// Index of the pair `(i, j)`, `i < j`, in a condensed distance matrix of
/// `n` points (SciPy's `condensed_index`).
fn condensed_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j && j < n);
    n * i - i * (i + 1) / 2 + (j - i - 1)
}

/// The condensed matrix of cosine distances `1 - cos` between the rows,
/// computed in `f64` in element order. A row of norm 0 has cosine 0 with
/// every other row; a non-finite distance becomes 2 (the maximum), so no
/// value can stall the clustering.
fn cosine_distances(rows: &[f32], dimension: usize) -> Vec<f64> {
    let n = rows.len() / dimension;
    let row = |i: usize| &rows[i * dimension..(i + 1) * dimension];
    let norms: Vec<f64> = (0..n)
        .map(|i| {
            row(i)
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let mut distances = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f64 = row(i)
                .iter()
                .zip(row(j))
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let norm = norms[i] * norms[j];
            let cos = if norm > 0.0 { dot / norm } else { 0.0 };
            let distance = 1.0 - cos;
            distances.push(if distance.is_finite() {
                distance.clamp(0.0, 2.0)
            } else {
                2.0
            });
        }
    }
    distances
}

/// SciPy's Lance-Williams update for `method="ward"` (`_ward` in
/// `_hierarchy_distance_update.pxi`), in the same arithmetic order. The
/// clamp only fences off a negative radicand from rounding.
fn ward_update(d_xi: f64, d_yi: f64, d_xy: f64, nx: usize, ny: usize, ni: usize) -> f64 {
    let t = 1.0 / (nx + ny + ni) as f64;
    ((ni + nx) as f64 * t * d_xi * d_xi + (ni + ny) as f64 * t * d_yi * d_yi
        - ni as f64 * t * d_xy * d_xy)
        .max(0.0)
        .sqrt()
}

/// Ward clustering of `rows` into `clusters` clusters.
///
/// Builds the full hierarchy with SciPy's nearest-neighbour chain
/// (`nn_chain` in `_hierarchy.pyx`), then applies the `n - clusters`
/// lowest merges (a stable sort by height, as SciPy sorts its linkage)
/// with a union-find over the rows. A merge's children always come before
/// it in that order, so this is the partition SciPy's linkage gives when
/// cut into `clusters` clusters.
///
/// # Returns
///
/// One label per row in `0..clusters`, numbered by first appearance.
fn ward_labels(rows: &[f32], dimension: usize, clusters: usize) -> Vec<usize> {
    let n = rows.len() / dimension;
    debug_assert!(clusters >= 1 && clusters <= n);
    let mut distances = cosine_distances(rows, dimension);
    let index = |a: usize, b: usize| {
        if a < b {
            condensed_index(n, a, b)
        } else {
            condensed_index(n, b, a)
        }
    };

    let mut size = vec![1usize; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut merges: Vec<(f64, usize, usize)> = Vec::with_capacity(n.saturating_sub(1));
    for _ in 0..n.saturating_sub(1) {
        if chain.is_empty() {
            chain.push(size.iter().position(|&s| s > 0).expect("an active cluster"));
        }
        // Walk the chain of nearest neighbours until two clusters are each
        // other's nearest. The previous element is preferred on ties, so
        // the walk cannot cycle.
        let (mut x, mut y, height) = loop {
            let x = chain[chain.len() - 1];
            let (mut y, mut nearest) = if chain.len() > 1 {
                let y = chain[chain.len() - 2];
                (y, distances[index(x, y)])
            } else {
                (usize::MAX, f64::INFINITY)
            };
            for i in 0..n {
                if size[i] == 0 || i == x {
                    continue;
                }
                let distance = distances[index(x, i)];
                if distance < nearest {
                    nearest = distance;
                    y = i;
                }
            }
            if chain.len() > 1 && y == chain[chain.len() - 2] {
                break (x, y, nearest);
            }
            chain.push(y);
        };
        chain.truncate(chain.len() - 2);
        // fastcluster's convention, kept by SciPy: the merged cluster
        // lives on at the larger index.
        if x > y {
            std::mem::swap(&mut x, &mut y);
        }
        let (nx, ny) = (size[x], size[y]);
        merges.push((height, x, y));
        size[x] = 0;
        size[y] = nx + ny;
        for i in 0..n {
            let ni = size[i];
            if ni == 0 || i == y {
                continue;
            }
            distances[index(i, y)] = ward_update(
                distances[index(i, x)],
                distances[index(i, y)],
                height,
                nx,
                ny,
                ni,
            );
        }
    }

    let mut order: Vec<usize> = (0..merges.len()).collect();
    order.sort_by(|a, b| {
        merges[*a]
            .0
            .partial_cmp(&merges[*b].0)
            .expect("merge heights are finite")
    });
    let mut parent: Vec<usize> = (0..n).collect();
    for &m in order.iter().take(n - clusters) {
        let (_, x, y) = merges[m];
        let (root_x, root_y) = (find(&mut parent, x), find(&mut parent, y));
        parent[root_x] = root_y;
    }
    let mut label_of_root = vec![usize::MAX; n];
    let mut next = 0;
    (0..n)
        .map(|i| {
            let root = find(&mut parent, i);
            if label_of_root[root] == usize::MAX {
                label_of_root[root] = next;
                next += 1;
            }
            label_of_root[root]
        })
        .collect()
}

/// Union-find root with path halving.
fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use serde::Deserialize;

    use super::*;

    /// Deterministic pseudo-random rows in `[-0.5, 0.5)`.
    fn rows(count: usize, dimension: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..count * dimension)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// Unit vectors on the circle at `degrees`.
    fn on_circle(degrees: &[f64]) -> Vec<f32> {
        degrees
            .iter()
            .flat_map(|d| {
                let r = d.to_radians();
                [r.cos() as f32, r.sin() as f32]
            })
            .collect()
    }

    fn norm(row: &[f32]) -> f32 {
        row.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    #[test]
    fn test_pooled_count_follows_the_cluster_count() {
        // (rows, factor, rows kept): the first row, then max(rest / f, 1)
        // clusters, unless that would not reduce the document.
        for (n, factor, kept) in [
            (0, 2, 0),
            (1, 4, 1),
            (2, 4, 2),
            (3, 2, 2),
            (5, 2, 3),
            (9, 4, 3),
            (10, 3, 4),
            (10, 1, 10),
            (3, 100, 2),
            (300, 2, 150),
        ] {
            assert_eq!(pooled_count(n, factor), kept, "{n} rows, factor {factor}");
        }
    }

    #[test]
    fn test_pool_returns_exactly_pooled_count_rows() {
        let dimension = 4;
        for n in 1..=64 {
            for factor in 1..=6 {
                let input = rows(n, dimension, (n * 10 + factor as usize) as u64);
                let kept = pool(&input, dimension, factor, true)
                    .map_or(n, |pooled| pooled.len() / dimension);
                assert_eq!(kept, pooled_count(n, factor), "{n} rows, factor {factor}");
            }
        }
    }

    #[test]
    fn test_first_row_is_kept_verbatim() {
        // Row 1 is the first row's nearest neighbour, so pooling the first
        // row with the others would merge it.
        let mut input = vec![1.0, 0.0, 0.0, 0.99, 0.141, 0.0];
        input.extend(rows(6, 3, 7));
        let pooled = pool(&input, 3, 2, true).expect("8 rows pool to 4");
        assert_eq!(pooled.len(), pooled_count(8, 2) * 3);
        assert_eq!(&pooled[..3], &input[..3]);
    }

    #[test]
    fn test_means_are_renormalized_only_when_asked() {
        // Two orthogonal unit rows after the first form the single cluster.
        let input = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let raw = pool(&input, 3, 2, false).unwrap();
        assert_eq!(&raw[3..], &[0.5, 0.5, 0.0]);
        let unit = pool(&input, 3, 2, true).unwrap();
        assert!((norm(&unit[3..]) - 1.0).abs() < 1e-6, "{unit:?}");
    }

    #[test]
    fn test_clusters_are_cut_in_height_order() {
        // SciPy cuts this into {0°}, {84.3°}, {200°..221.3°}. Applying the
        // merges in the order the chain finds them instead gives
        // {0°, 84.3°}, {200°, 205°}, {215°, 221.3°}.
        let input = on_circle(&[0.0, 84.3, 200.0, 205.0, 215.0, 221.3]);
        assert_eq!(ward_labels(&input, 2, 3), [0, 1, 2, 2, 2, 2]);
    }

    #[test]
    fn test_degenerate_rows_still_give_the_exact_cluster_count() {
        let dimension = 3;
        let mut input = vec![0.0; 3 * dimension];
        input.extend([f32::NAN, 1.0, 0.0, f32::INFINITY, 0.0, 1.0, 1.0, 1.0, 1.0]);
        input.extend(rows(3, dimension, 11));
        for factor in 2..=4 {
            let pooled = pool(&input, dimension, factor, true).unwrap();
            assert_eq!(pooled.len() / dimension, pooled_count(9, factor));
        }
        // A row whose cosine with every other row is NaN (infinite norm
        // against non-zero rows), first among the clustered rows: the
        // chain starts at it and must still find a neighbour.
        let mut input = vec![1.0, 0.0, 0.0, f32::INFINITY, 0.0, 0.0];
        input.extend(rows(5, dimension, 13).iter().map(|v| v + 1.0));
        let pooled = pool(&input, dimension, 2, true).unwrap();
        assert_eq!(pooled.len() / dimension, pooled_count(7, 2));
        // All rows equal: every distance ties, the cut still yields k.
        let same = [0.6f32, 0.8, 0.0].repeat(9);
        let labels = ward_labels(&same, 3, 4);
        assert_eq!(labels.iter().max(), Some(&3));
    }

    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        dimension: usize,
        clusters: usize,
        rows: String,
        labels: Vec<usize>,
    }

    /// `scripts/token_pooling_reference.py` writes, for each row set, the
    /// partition SciPy's `linkage(condensed, method="ward")` gives when cut
    /// into `clusters` clusters.
    #[test]
    fn test_ward_partitions_match_scipy() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../tests/fixtures/token_pooling/ward_scipy.json"
        ))
        .unwrap();
        assert!(!fixture.cases.is_empty());
        for (i, case) in fixture.cases.iter().enumerate() {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&case.rows)
                .unwrap();
            let values: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(
                ward_labels(&values, case.dimension, case.clusters),
                case.labels,
                "case {i}"
            );
        }
    }
}
