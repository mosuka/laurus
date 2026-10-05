//! Late-interaction (MaxSim) scoring (Issue #1177).
//!
//! ColBERT-style late interaction scores a document by matching every query
//! token vector against its best document token vector and summing those
//! best similarities:
//!
//! ```text
//! score(q, d) = Σ_i max_j (q_i · d_j)
//! ```
//!
//! The similarity is a plain dot product. For cosine fields both sides are
//! L2-normalized beforehand, so the dot product is the cosine similarity.
//! The sum is taken on the raw similarities: a non-affine transform applied
//! per pair before summing would change the ranking.

use crate::vector::core::distance::dot_product;

/// MaxSim of one document.
///
/// # Arguments
///
/// * `query` - Query token vectors, row-major (`query.len()` is a multiple
///   of `dimension`).
/// * `document` - Document token vectors, row-major, at least one vector.
/// * `dimension` - Length of every vector.
///
/// # Returns
///
/// `Σ_i max_j q_i · d_j` over the query rows `q_i` and document rows `d_j`.
pub(crate) fn max_sim(query: &[f32], document: &[f32], dimension: usize) -> f32 {
    debug_assert!(dimension > 0 && query.len().is_multiple_of(dimension));
    debug_assert!(!document.is_empty() && document.len().is_multiple_of(dimension));
    query
        .chunks_exact(dimension)
        .map(|q| {
            document
                .chunks_exact(dimension)
                .map(|d| dot_product(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward scalar MaxSim, the reference the SIMD path must match.
    fn reference(query: &[Vec<f32>], document: &[Vec<f32>]) -> f32 {
        query
            .iter()
            .map(|q| {
                document
                    .iter()
                    .map(|d| q.iter().zip(d).map(|(a, b)| a * b).sum::<f32>())
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()
    }

    /// Deterministic pseudo-random values in `[-1, 1)`.
    fn values(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn rows(flat: &[f32], dimension: usize) -> Vec<Vec<f32>> {
        flat.chunks_exact(dimension).map(<[f32]>::to_vec).collect()
    }

    #[test]
    fn test_matches_scalar_reference_across_dimensions() {
        for dimension in [1, 7, 8, 9, 128] {
            for (query_rows, doc_rows) in [(1, 1), (3, 5), (32, 17)] {
                let query = values(dimension as u64, query_rows * dimension);
                let document = values(1000 + dimension as u64, doc_rows * dimension);
                let expected = reference(&rows(&query, dimension), &rows(&document, dimension));
                let actual = max_sim(&query, &document, dimension);
                assert!(
                    (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                    "dimension {dimension}, {query_rows}x{doc_rows}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn test_takes_the_best_document_vector_per_query_vector() {
        // Query rows pick different document rows; the sum is 2 + 3.
        let query = [1.0, 0.0, 0.0, 1.0];
        let document = [2.0, 0.0, 0.0, 3.0, -5.0, -5.0];
        assert_eq!(max_sim(&query, &document, 2), 5.0);
    }

    #[test]
    fn test_negative_similarities_are_kept() {
        // Every pair is negative; the best (least negative) one counts.
        let query = [1.0, 1.0];
        let document = [-1.0, -2.0, -0.5, -0.5];
        assert_eq!(max_sim(&query, &document, 2), -1.0);
    }

    #[test]
    fn test_single_document_vector() {
        let query = [1.0, 2.0, 3.0, 4.0];
        let document = [0.5, 0.5];
        assert_eq!(max_sim(&query, &document, 2), 1.5 + 3.5);
    }
}
