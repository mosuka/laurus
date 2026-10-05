//! Distance metrics for vector similarity calculation.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};

/// Distance metrics for vector similarity calculation.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    Default,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]

pub enum DistanceMetric {
    /// Cosine distance (1 - cosine similarity)
    #[default]
    Cosine,
    /// Euclidean (L2) distance
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
    /// Dot product distance.
    ///
    /// Computed as `-(a . b)` (negated dot product) so that smaller values
    /// indicate more similar vectors, consistent with the other distance
    /// metrics. Raw dot product similarity is higher for more similar vectors,
    /// so the negation converts it into a distance. This means the returned
    /// distance values are typically **negative** for vectors with positive
    /// dot product similarity.
    DotProduct,
    /// Angular distance
    Angular,
}

/// Cached query-side state used by [`DistanceMetric::distance_with_prepared`]
/// to skip the per-candidate `||query||²` accumulation (#414).
///
/// Constructed via [`DistanceMetric::prepare_query`] once per search;
/// borrows the query slice for the lifetime of the prepared value.
/// Only `Cosine` and `Angular` actually consume the cached norm — the
/// other metrics receive a placeholder `norm_sq = 0.0` and forward
/// calls to the regular [`DistanceMetric::distance`] path.
#[derive(Debug, Clone, Copy)]
pub struct PreparedQuery<'a> {
    /// Query vector data (borrowed for the lifetime of the prepared
    /// value).
    pub data: &'a [f32],
    /// `||query||²`, precomputed at preparation time. `0.0` when the
    /// metric (`Euclidean` / `Manhattan` / `DotProduct`) does not use
    /// the query norm.
    pub norm_sq: f32,
}

impl DistanceMetric {
    /// Calculate the distance between two vectors using this metric.
    ///
    /// Lower values indicate more similar vectors for all metrics. For
    /// [`DotProduct`](Self::DotProduct), the result is `-(a . b)`, which is
    /// typically negative when vectors have positive similarity.
    ///
    /// # Arguments
    ///
    /// * `a` - The first vector (as a float slice).
    /// * `b` - The second vector (as a float slice). Must have the same length
    ///   as `a`.
    ///
    /// # Returns
    ///
    /// The distance between the two vectors according to this metric.
    ///
    /// # Errors
    ///
    /// Returns an error if the two vectors have different dimensions.
    pub fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(LaurusError::InvalidOperation(
                "Vector dimensions must match for distance calculation".to_string(),
            ));
        }

        let result = match self {
            DistanceMetric::Cosine => {
                let (dot_product, norm_a_sq, norm_b_sq) = self.simd_dot_and_norms(a, b);
                let norm_a = norm_a_sq.sqrt();
                let norm_b = norm_b_sq.sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0 // Maximum distance for zero vectors
                } else {
                    let cosine = (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0);
                    1.0 - cosine
                }
            }
            DistanceMetric::Euclidean => self.simd_euclidean_sq(a, b).sqrt(),
            DistanceMetric::Manhattan => self.simd_manhattan(a, b),
            DistanceMetric::DotProduct => -dot_product(a, b),
            DistanceMetric::Angular => {
                let (dot_product, norm_a_sq, norm_b_sq) = self.simd_dot_and_norms(a, b);
                let norm_a = norm_a_sq.sqrt();
                let norm_b = norm_b_sq.sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    std::f32::consts::PI
                } else {
                    let cosine = (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0);
                    cosine.acos()
                }
            }
        };

        Ok(result)
    }

    /// Calculate dot product and squared norms in a single pass using SIMD.
    fn simd_dot_and_norms(&self, a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        use wide::f32x8;

        let mut dot_sum = f32x8::ZERO;
        let mut norm_a_sum = f32x8::ZERO;
        let mut norm_b_sum = f32x8::ZERO;

        let (chunks_a, rem_a) = a.as_chunks::<8>();
        let (chunks_b, rem_b) = b.as_chunks::<8>();

        for (ca, cb) in chunks_a.iter().zip(chunks_b) {
            let va = f32x8::from(*ca);
            let vb = f32x8::from(*cb);
            dot_sum += va * vb;
            norm_a_sum += va * va;
            norm_b_sum += vb * vb;
        }

        let mut dot_product: f32 = dot_sum.reduce_add();
        let mut norm_a_sq: f32 = norm_a_sum.reduce_add();
        let mut norm_b_sq: f32 = norm_b_sum.reduce_add();

        // Tail
        for (x, y) in rem_a.iter().zip(rem_b.iter()) {
            dot_product += x * y;
            norm_a_sq += x * x;
            norm_b_sq += y * y;
        }

        (dot_product, norm_a_sq, norm_b_sq)
    }

    /// Calculate dot product and the squared norm of `b` only — skipping
    /// the `||a||²` accumulation that [`Self::simd_dot_and_norms`] does
    /// alongside (#414). The query-side norm is constant within one
    /// search and is precomputed at [`Self::prepare_query`] time, so
    /// every candidate scan can pay 2 multiply-add chains per
    /// 8-element chunk instead of 3.
    fn simd_dot_and_norm_b(a: &[f32], b: &[f32]) -> (f32, f32) {
        use wide::f32x8;

        let mut dot_sum = f32x8::ZERO;
        let mut norm_b_sum = f32x8::ZERO;

        let (chunks_a, rem_a) = a.as_chunks::<8>();
        let (chunks_b, rem_b) = b.as_chunks::<8>();

        for (ca, cb) in chunks_a.iter().zip(chunks_b) {
            let va = f32x8::from(*ca);
            let vb = f32x8::from(*cb);
            dot_sum += va * vb;
            norm_b_sum += vb * vb;
        }

        let mut dot_product: f32 = dot_sum.reduce_add();
        let mut norm_b_sq: f32 = norm_b_sum.reduce_add();

        // Tail
        for (x, y) in rem_a.iter().zip(rem_b.iter()) {
            dot_product += x * y;
            norm_b_sq += y * y;
        }

        (dot_product, norm_b_sq)
    }

    /// Calculate squared Euclidean distance using SIMD.
    fn simd_euclidean_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        use wide::f32x8;

        let mut sum = f32x8::ZERO;
        let (chunks_a, rem_a) = a.as_chunks::<8>();
        let (chunks_b, rem_b) = b.as_chunks::<8>();

        for (ca, cb) in chunks_a.iter().zip(chunks_b) {
            let diff = f32x8::from(*ca) - f32x8::from(*cb);
            sum += diff * diff;
        }

        let mut dist_sq: f32 = sum.reduce_add();
        for (x, y) in rem_a.iter().zip(rem_b.iter()) {
            dist_sq += (x - y).powi(2);
        }
        dist_sq
    }

    /// Calculate Manhattan distance using SIMD.
    fn simd_manhattan(&self, a: &[f32], b: &[f32]) -> f32 {
        use wide::f32x8;

        let mut sum = f32x8::ZERO;
        let (chunks_a, rem_a) = a.as_chunks::<8>();
        let (chunks_b, rem_b) = b.as_chunks::<8>();

        for (ca, cb) in chunks_a.iter().zip(chunks_b) {
            let va = f32x8::from(*ca);
            let vb = f32x8::from(*cb);
            sum += (va - vb).abs();
        }

        let mut dist: f32 = sum.reduce_add();
        for (x, y) in rem_a.iter().zip(rem_b.iter()) {
            dist += (x - y).abs();
        }
        dist
    }

    /// Build a query-side `PreparedQuery` that caches the squared norm
    /// of `query` so subsequent [`Self::distance_with_prepared`] calls
    /// can skip the redundant per-candidate `||query||²` accumulation
    /// (#414).
    ///
    /// Only `Cosine` and `Angular` consume the cached norm; the other
    /// metrics carry it as `0.0` and recompute everything from
    /// `prepared.data` on each call. The cost saved over a top-K query
    /// scales with `candidates × dimension`.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector. The returned `PreparedQuery`
    ///   borrows it for the lifetime of the prepared value.
    pub fn prepare_query<'a>(&self, query: &'a [f32]) -> PreparedQuery<'a> {
        let norm_sq = match self {
            DistanceMetric::Cosine | DistanceMetric::Angular => {
                use wide::f32x8;
                let mut sum = f32x8::ZERO;
                let (chunks, rem) = query.as_chunks::<8>();
                for c in chunks {
                    let v = f32x8::from(*c);
                    sum += v * v;
                }
                let mut s: f32 = sum.reduce_add();
                for x in rem {
                    s += x * x;
                }
                s
            }
            _ => 0.0,
        };
        PreparedQuery {
            data: query,
            norm_sq,
        }
    }

    /// Distance from a `PreparedQuery` to a candidate vector. Uses the
    /// cached `||query||²` (when meaningful for the metric) and
    /// only recomputes the `b`-side and dot-product accumulators on
    /// each call.
    ///
    /// For `Cosine` and `Angular` this saves one `||a||²` accumulation
    /// per candidate; for `Euclidean`, `Manhattan`, and `DotProduct`
    /// the result is identical to [`Self::distance`] — there is no
    /// usable per-query cache for those metrics, so the prepared API
    /// just forwards to the existing implementation.
    pub fn distance_with_prepared(&self, prepared: &PreparedQuery<'_>, b: &[f32]) -> Result<f32> {
        if prepared.data.len() != b.len() {
            return Err(LaurusError::InvalidOperation(
                "Vector dimensions must match for distance calculation".to_string(),
            ));
        }

        let result = match self {
            DistanceMetric::Cosine => {
                let (dot_product, norm_b_sq) = Self::simd_dot_and_norm_b(prepared.data, b);
                let norm_a = prepared.norm_sq.sqrt();
                let norm_b = norm_b_sq.sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0
                } else {
                    let cosine = (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0);
                    1.0 - cosine
                }
            }
            DistanceMetric::Angular => {
                let (dot_product, norm_b_sq) = Self::simd_dot_and_norm_b(prepared.data, b);
                let norm_a = prepared.norm_sq.sqrt();
                let norm_b = norm_b_sq.sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    std::f32::consts::PI
                } else {
                    let cosine = (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0);
                    cosine.acos()
                }
            }
            DistanceMetric::Euclidean | DistanceMetric::Manhattan | DistanceMetric::DotProduct => {
                self.distance(prepared.data, b)?
            }
        };

        Ok(result)
    }

    /// Calculate similarity (0-1, higher is more similar) between two vectors.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let distance = self.distance(a, b)?;

        let similarity = match self {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::Euclidean => (-distance).exp(),
            DistanceMetric::Manhattan => (-distance).exp(),
            DistanceMetric::DotProduct => -distance,
            DistanceMetric::Angular => 1.0 - (distance / std::f32::consts::PI),
        };

        Ok(similarity.clamp(0.0, 1.0))
    }

    /// Convert a pre-computed distance value to a similarity score without
    /// re-reading the original vectors.
    ///
    /// This is the pure-arithmetic inverse of the per-metric transform applied
    /// in [`distance()`](Self::distance), so it is **much** cheaper than calling
    /// [`similarity()`](Self::similarity) (which reloads both vectors and
    /// recomputes dot products / norms).
    ///
    /// # Arguments
    ///
    /// * `distance` - A distance value previously returned by
    ///   [`distance()`](Self::distance) for the same metric.
    ///
    /// # Returns
    ///
    /// A similarity score in [0, 1] (higher is more similar).
    pub fn distance_to_similarity(&self, distance: f32) -> f32 {
        let similarity = match self {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::Euclidean => (-distance).exp(),
            DistanceMetric::Manhattan => (-distance).exp(),
            DistanceMetric::DotProduct => -distance,
            DistanceMetric::Angular => 1.0 - (distance / std::f32::consts::PI),
        };
        similarity.clamp(0.0, 1.0)
    }

    /// Get the name of this distance metric.
    pub fn name(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Euclidean => "euclidean",
            DistanceMetric::Manhattan => "manhattan",
            DistanceMetric::DotProduct => "dot_product",
            DistanceMetric::Angular => "angular",
        }
    }

    /// Parse a distance metric from a string.
    pub fn parse_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(DistanceMetric::Cosine),
            "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
            "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
            "dot_product" | "dot" => Ok(DistanceMetric::DotProduct),
            "angular" => Ok(DistanceMetric::Angular),
            _ => Err(LaurusError::InvalidOperation(format!(
                "Unknown distance metric: {s}"
            ))),
        }
    }

    /// Calculate distance between a query vector and multiple vectors in parallel.
    pub fn batch_distance_parallel(&self, query: &[f32], vectors: &[&[f32]]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        if vectors.len() < 100 {
            return vectors
                .iter()
                .map(|v| self.distance(query, v))
                .collect::<Result<Vec<_>>>();
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            vectors
                .par_iter()
                .map(|v| self.distance(query, v))
                .collect::<Result<Vec<_>>>()
        }
        #[cfg(target_arch = "wasm32")]
        {
            vectors
                .iter()
                .map(|v| self.distance(query, v))
                .collect::<Result<Vec<_>>>()
        }
    }

    /// Calculate similarities between a query vector and multiple vectors in parallel.
    pub fn batch_similarity_parallel(&self, query: &[f32], vectors: &[&[f32]]) -> Result<Vec<f32>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        if vectors.len() < 100 {
            return vectors
                .iter()
                .map(|v| self.similarity(query, v))
                .collect::<Result<Vec<_>>>();
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            vectors
                .par_iter()
                .map(|v| self.similarity(query, v))
                .collect::<Result<Vec<_>>>()
        }
        #[cfg(target_arch = "wasm32")]
        {
            vectors
                .iter()
                .map(|v| self.similarity(query, v))
                .collect::<Result<Vec<_>>>()
        }
    }
}

/// Dot product of two equal-length slices, 8 lanes at a time.
///
/// Extra elements of the longer slice are ignored; callers pass slices of
/// the same length.
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    use wide::f32x8;

    let mut sum = f32x8::ZERO;
    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_b, rem_b) = b.as_chunks::<8>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        sum += f32x8::from(*ca) * f32x8::from(*cb);
    }

    let mut dot_product: f32 = sum.reduce_add();
    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        dot_product += x * y;
    }
    dot_product
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `distance_with_prepared` must agree with `distance` for every
    /// metric (#414). Cosine / Angular use the cached query norm;
    /// Euclidean / Manhattan / DotProduct fall back to the unprepared
    /// path internally — the result must still match.
    #[test]
    fn distance_with_prepared_matches_distance() {
        let a: Vec<f32> = (0..768).map(|i| (i as f32) * 0.01 + 1.0).collect();
        let b: Vec<f32> = (0..768).map(|i| (i as f32) * 0.02 - 0.5).collect();

        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::DotProduct,
            DistanceMetric::Angular,
        ] {
            let direct = metric.distance(&a, &b).unwrap();
            let prepared = metric.prepare_query(&a);
            let via_prep = metric.distance_with_prepared(&prepared, &b).unwrap();
            assert!(
                (direct - via_prep).abs() < 1e-5,
                "{metric:?}: direct={direct}, prepared={via_prep}"
            );
        }
    }

    #[test]
    fn prepared_query_norm_only_set_for_cosine_and_angular() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let expected_norm_sq: f32 = 1.0 + 4.0 + 9.0 + 16.0;

        for metric in [DistanceMetric::Cosine, DistanceMetric::Angular] {
            let p = metric.prepare_query(&v);
            assert!((p.norm_sq - expected_norm_sq).abs() < 1e-6);
        }
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::DotProduct,
        ] {
            let p = metric.prepare_query(&v);
            assert_eq!(
                p.norm_sq, 0.0,
                "{metric:?}: norm_sq must be left at 0.0 (placeholder)"
            );
        }
    }
}
