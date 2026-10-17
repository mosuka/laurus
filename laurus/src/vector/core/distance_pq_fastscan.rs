//! FastScan-style PQ ADC query state and scalar reference kernel
//! (Issue [#651](https://github.com/mosuka/laurus/issues/651) /
//! part A [#692](https://github.com/mosuka/laurus/issues/692)).
//!
//! Parallel to
//! [`crate::vector::core::distance_quantized::PqQuery`] (the existing
//! K=256 8-bit PQ query); the differences are:
//!
//! - The per-query lookup table is sized for K=16 (16 entries per
//!   sub-quantiser) so it fits in a single 128-bit SIMD register, which
//!   the AVX2 / NEON kernels (parts B #693, C #694) consume via
//!   `pshufb` / `vqtbl1q_u8`.
//! - The LUT is **quantised to `u8`** per sub-quantiser with a
//!   min-max affine map so the SIMD path can accumulate distances in
//!   `u8` lanes; the [`PqFastScanQuery::lut_f32`] field preserves the
//!   unquantised LUT so the scalar reference here can be cross-checked
//!   against the true f32 distance.
//!
//! This module only ships scalar kernels. The SIMD kernels and the
//! production wiring live in later parts of #651.

use crate::error::{LaurusError, Result};
use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization::PqParams;
use crate::vector::index::pq_fastscan_storage::{BLOCK_SIZE, BYTES_PER_SUB_PER_BLOCK};

/// K constant for FastScan (4-bit codes).
const K: usize = 16;

/// Per-query state for the FastScan PQ ADC kernel.
///
/// Built once per search via [`PqFastScanQuery::prepare`] and reused
/// for every candidate. The scalar reference kernels offer two paths:
///
/// - [`distance_pq_fastscan_f32_scalar`] uses [`Self::lut_f32`]
///   directly — this is the mathematically-exact ADC distance and
///   serves as the ground truth for tests.
/// - [`distance_pq_fastscan_u8_scalar`] uses [`Self::lut4`] plus the
///   per-sub-quantiser [`Self::lut_scale`] / [`Self::lut_bias`] to
///   recover an f32 distance, matching what the SIMD kernels in
///   parts B / C will compute.
#[derive(Debug, Clone)]
pub struct PqFastScanQuery {
    /// PQ parameters this LUT was built against. `params.k` must be
    /// `16`; the constructor returns an error otherwise.
    pub params: PqParams,
    /// Unquantised LUT, row-major `[m][k]` of length `m * K`. Kept
    /// alongside the quantised LUT so tests can isolate the kernel
    /// logic from the quantisation drift.
    pub lut_f32: Vec<f32>,
    /// Quantised LUT, row-major `[m][k]` of length `m * K`. Each
    /// entry is in `[0, 255]` and decodes to
    /// `lut_f32 ≈ u8 * lut_scale[m] + lut_bias[m]`.
    pub lut4: Vec<u8>,
    /// Per-sub-quantiser scale of the affine quantisation. Length is
    /// `M`.
    pub lut_scale: Vec<f32>,
    /// Per-sub-quantiser bias of the affine quantisation. Length is
    /// `M`.
    pub lut_bias: Vec<f32>,
    /// Globally-quantised LUT for the SIMD path (#693 AVX2 / #694 NEON
    /// kernels). Row-major `[m][k]` of length `m * K`. All entries
    /// share the single [`Self::lut_scale_global`] and decode to
    /// `lut_f32 ≈ u8 * lut_scale_global + lut_bias_per_sub[m]`, where
    /// the per-sub bias is folded into [`Self::lut_bias_sum`] for the
    /// distance reconstruction.
    pub lut4_global: Vec<u8>,
    /// Single global scale shared across all sub-quantisers. Built by
    /// [`quantise_lut_global`] so the SIMD u8 saturating accumulator
    /// pattern is correct (FAISS pq4_fast_scan convention).
    pub lut_scale_global: f32,
    /// Sum of per-sub-quantiser biases (`Σ_m bias[m]`). The SIMD
    /// kernel adds this once per vector after the u16 reduction:
    /// `distance = u16_sum * lut_scale_global + lut_bias_sum`.
    pub lut_bias_sum: f32,
    /// `||query||² = Σ_i query[i]²`, cached for the metrics that need
    /// it (currently unused by the scalar kernels but kept for API
    /// parity with [`crate::vector::core::distance_quantized::PqQuery`]).
    pub query_norm_sq: f32,
}

impl PqFastScanQuery {
    /// Build the per-query LUT (`f32` and `u8` variants) from a raw
    /// f32 query and the segment's K=16 codebook.
    ///
    /// # Arguments
    ///
    /// * `query` - The raw f32 query vector, length must match
    ///   `params.original_dim()`.
    /// * `params` - PQ parameters with `k == 16`.
    /// * `codebook` - Row-major `[m][k][sub_dim]` codebook from
    ///   [`crate::vector::core::quantization::pq_train_codebook`].
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::InvalidOperation`] if `params.k != 16`,
    /// if `query.len() != params.original_dim()`, or if
    /// `codebook.len() != params.codebook_len()`.
    pub fn prepare(query: &[f32], params: PqParams, codebook: &[f32]) -> Result<Self> {
        if params.k as usize != K {
            return Err(LaurusError::InvalidOperation(format!(
                "PqFastScanQuery requires PqParams::k == {K} (got {})",
                params.k
            )));
        }
        if query.len() != params.original_dim() {
            return Err(LaurusError::InvalidOperation(format!(
                "PqFastScanQuery: query length {} does not match params.original_dim() {}",
                query.len(),
                params.original_dim()
            )));
        }
        if codebook.len() != params.codebook_len() {
            return Err(LaurusError::InvalidOperation(format!(
                "PqFastScanQuery: codebook length {} does not match params.codebook_len() {}",
                codebook.len(),
                params.codebook_len()
            )));
        }

        let m = params.m as usize;
        let sub_dim = params.sub_dim as usize;
        let mut lut_f32 = vec![0.0_f32; m * K];

        for sub in 0..m {
            let q_sub = &query[sub * sub_dim..(sub + 1) * sub_dim];
            let cb_base = sub * K * sub_dim;
            for ki in 0..K {
                let c = &codebook[cb_base + ki * sub_dim..cb_base + (ki + 1) * sub_dim];
                let mut acc = 0.0_f32;
                for d in 0..sub_dim {
                    let diff = q_sub[d] - c[d];
                    acc += diff * diff;
                }
                lut_f32[sub * K + ki] = acc;
            }
        }

        let (lut4, lut_scale, lut_bias) = quantise_lut_per_sub(&lut_f32, m);
        let (lut4_global, lut_scale_global, lut_bias_sum) = quantise_lut_global(&lut_f32, m);
        let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();

        Ok(Self {
            params,
            lut_f32,
            lut4,
            lut_scale,
            lut_bias,
            lut4_global,
            lut_scale_global,
            lut_bias_sum,
            query_norm_sq,
        })
    }
}

/// Quantise a per-sub-quantiser-grouped f32 LUT to `u8`.
///
/// For each sub-quantiser `m`, finds the min / max of
/// `lut_f32[m * K .. (m + 1) * K]`, builds an affine map
/// `u8 = round((f32 - min) / (max - min) * 255)`, and records the
/// inverse `(scale, bias)` so the kernel can recover an f32
/// approximation.
///
/// Returns `(lut4, lut_scale, lut_bias)`:
///
/// - `lut4.len() == m * K`
/// - `lut_scale.len() == m`, `lut_bias.len() == m`
/// - `f32_approx = u8_value * lut_scale[m] + lut_bias[m]`
///
/// If a sub-quantiser's LUT has zero range (min == max), the affine
/// map collapses to a zero-scale dequantiser; every code decodes to
/// the constant `min` value.
pub fn quantise_lut_per_sub(lut_f32: &[f32], m: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(lut_f32.len(), m * K);
    let mut lut4 = vec![0u8; m * K];
    let mut lut_scale = vec![0.0_f32; m];
    let mut lut_bias = vec![0.0_f32; m];

    for sub in 0..m {
        let chunk = &lut_f32[sub * K..(sub + 1) * K];
        let min = chunk.iter().copied().fold(f32::INFINITY, f32::min);
        let max = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let span = max - min;
        if span > 0.0 {
            let scale = span / 255.0;
            lut_scale[sub] = scale;
            lut_bias[sub] = min;
            for (i, &v) in chunk.iter().enumerate() {
                let q = ((v - min) / span * 255.0).round().clamp(0.0, 255.0) as u8;
                lut4[sub * K + i] = q;
            }
        } else {
            // Degenerate LUT: every entry collapses to `min`. The u8
            // codes are arbitrary; we choose 0 so the dequantiser
            // recovers `bias = min` exactly.
            lut_scale[sub] = 0.0;
            lut_bias[sub] = min;
        }
    }

    (lut4, lut_scale, lut_bias)
}

/// Quantise an f32 LUT to `u8` using a single **global scale** and
/// per-sub-quantiser biases, the form the FastScan SIMD kernels in
/// parts B (#693 AVX2) and C (#694 NEON) consume.
///
/// Unlike [`quantise_lut_per_sub`], all sub-quantisers share one
/// `scale_global`, so the SIMD path can accumulate distances in a u8
/// saturating accumulator across sub-quantisers and dequantise once
/// per vector at the end via
/// `distance = u8_sum * scale_global + bias_sum`.
///
/// Algorithm:
/// 1. For each sub `m`, `bias[m] = min(lut_f32[m * K .. (m + 1) * K])`.
/// 2. `max_normalised = max(lut_f32[m, k] - bias[m])` across all `m, k`.
/// 3. `scale_global = max_normalised / 255.0` (or `0.0` if everything
///    collapses to the per-sub minima).
/// 4. `lut4[m * K + k] = round((lut_f32[m, k] - bias[m]) / scale_global)`,
///    clamped to `[0, 255]`. Degenerate (constant) sub-quantisers get
///    all-zero codes so the dequantiser recovers `bias[m]` exactly.
/// 5. `bias_sum = Σ_m bias[m]`, folded into the per-vector constant.
///
/// Returns `(lut4_global, scale_global, bias_sum)`:
///
/// - `lut4_global.len() == m * K`, each entry in `[0, 255]`.
/// - `scale_global` is a single non-negative `f32`.
/// - `bias_sum` is `Σ_m bias[m]` (any sign).
pub fn quantise_lut_global(lut_f32: &[f32], m: usize) -> (Vec<u8>, f32, f32) {
    debug_assert_eq!(lut_f32.len(), m * K);
    let mut bias = vec![0.0_f32; m];
    let mut max_normalised: f32 = 0.0;
    for sub in 0..m {
        let chunk = &lut_f32[sub * K..(sub + 1) * K];
        let min = chunk.iter().copied().fold(f32::INFINITY, f32::min);
        bias[sub] = min;
        for &v in chunk {
            let normalised = v - min;
            if normalised > max_normalised {
                max_normalised = normalised;
            }
        }
    }
    let scale_global = if max_normalised > 0.0 {
        max_normalised / 255.0
    } else {
        0.0
    };
    let mut lut4 = vec![0u8; m * K];
    if scale_global > 0.0 {
        for sub in 0..m {
            for k in 0..K {
                let v = lut_f32[sub * K + k] - bias[sub];
                lut4[sub * K + k] = (v / scale_global).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    let bias_sum: f32 = bias.iter().sum();
    (lut4, scale_global, bias_sum)
}

/// Decode the M 4-bit codes for `vec_idx_in_block` out of a single
/// block of packed FastScan storage.
///
/// `packed_block` must point at the start of one block, i.e. its
/// length is `m * BYTES_PER_SUB_PER_BLOCK`. `vec_idx_in_block` is in
/// `[0, BLOCK_SIZE)`.
fn decode_codes_in_block(packed_block: &[u8], m: usize, vec_idx_in_block: usize) -> Vec<u8> {
    debug_assert_eq!(packed_block.len(), m * BYTES_PER_SUB_PER_BLOCK);
    debug_assert!(vec_idx_in_block < BLOCK_SIZE);
    let mut codes = vec![0u8; m];
    let (j, shift) = if vec_idx_in_block < 16 {
        (vec_idx_in_block, 0)
    } else {
        (vec_idx_in_block - 16, 4)
    };
    for sub in 0..m {
        let byte = packed_block[sub * BYTES_PER_SUB_PER_BLOCK + j];
        codes[sub] = (byte >> shift) & 0x0F;
    }
    codes
}

/// Scalar FastScan ADC distance using the **unquantised** f32 LUT.
///
/// Returns the mathematically-exact ADC distance (i.e. the value the
/// per-block sum `Σ_m lut_f32[m * K + code_m]` produces), with the
/// metric-specific post-processing applied. Used as the ground truth
/// for testing the `u8` kernel and (later) the SIMD kernels.
///
/// `packed_block` must be one block's worth of storage and
/// `vec_idx_in_block` is in `[0, BLOCK_SIZE)`. Out-of-range candidates
/// (positions past `n_vectors` inside the trailing block) are the
/// caller's responsibility — this kernel decodes whatever nibbles are
/// in the buffer.
pub fn distance_pq_fastscan_f32_scalar(
    metric: DistanceMetric,
    query: &PqFastScanQuery,
    packed_block: &[u8],
    vec_idx_in_block: usize,
) -> f32 {
    let m = query.params.m as usize;
    let codes = decode_codes_in_block(packed_block, m, vec_idx_in_block);
    let mut l2_sq = 0.0_f32;
    for (sub, &code) in codes.iter().enumerate() {
        l2_sq += query.lut_f32[sub * K + code as usize];
    }
    apply_metric(metric, l2_sq)
}

/// Scalar FastScan ADC distance using the **quantised** u8 LUT.
///
/// Mirrors what the SIMD kernels in parts B (#693) and C (#694) will
/// compute: accumulate `u8` LUT entries (using a `u32` accumulator
/// here to side-step the saturation bookkeeping the SIMD path needs)
/// and convert back to f32 via the per-sub-quantiser scale + bias.
///
/// Equivalent to `Σ_m (lut4[m * K + code_m] * lut_scale[m] + lut_bias[m])`,
/// which matches the kernel-level dequantisation step the SIMD path
/// will perform once it has emitted the final accumulator value.
pub fn distance_pq_fastscan_u8_scalar(
    metric: DistanceMetric,
    query: &PqFastScanQuery,
    packed_block: &[u8],
    vec_idx_in_block: usize,
) -> f32 {
    let m = query.params.m as usize;
    let codes = decode_codes_in_block(packed_block, m, vec_idx_in_block);
    let mut l2_sq = 0.0_f32;
    for (sub, &code) in codes.iter().enumerate() {
        let q = query.lut4[sub * K + code as usize] as f32;
        l2_sq += q * query.lut_scale[sub] + query.lut_bias[sub];
    }
    apply_metric(metric, l2_sq)
}

/// Scalar FastScan ADC distance using the **global**-scale u8 LUT.
///
/// Mirrors the FAISS-style SIMD pipeline that parts B (#693 AVX2) and
/// C (#694 NEON) implement: accumulate `Σ_m lut4_global[m * K + code_m]`
/// in u16, then dequantise once via
/// `distance = u16_sum as f32 * lut_scale_global + lut_bias_sum`.
///
/// The arithmetic here is bit-identical to the SIMD kernels (no
/// intermediate floats inside the accumulator) so this function serves
/// as the **correctness anchor** for the SIMD kernels' property tests.
pub fn distance_pq_fastscan_u8_global_scalar(
    metric: DistanceMetric,
    query: &PqFastScanQuery,
    packed_block: &[u8],
    vec_idx_in_block: usize,
) -> f32 {
    let m = query.params.m as usize;
    let codes = decode_codes_in_block(packed_block, m, vec_idx_in_block);
    let mut u16_sum: u32 = 0;
    for (sub, &code) in codes.iter().enumerate() {
        u16_sum += query.lut4_global[sub * K + code as usize] as u32;
    }
    let l2_sq = u16_sum as f32 * query.lut_scale_global + query.lut_bias_sum;
    apply_metric(metric, l2_sq)
}

/// Apply the metric-specific post-processing to the raw L2² sum.
///
/// Same convention as
/// [`crate::vector::core::distance_quantized::distance_pq_adc`]:
/// Euclidean returns `sqrt(L2²)`, Cosine halves and clamps to `[0, 2]`.
/// Other metrics return `f32::INFINITY` as a defensive fallback for
/// callers that bypass HNSW field validation.
#[inline]
pub(crate) fn apply_metric(metric: DistanceMetric, l2_sq: f32) -> f32 {
    match metric {
        DistanceMetric::Euclidean => l2_sq.max(0.0).sqrt(),
        DistanceMetric::Cosine => (l2_sq * 0.5).clamp(0.0, 2.0),
        _ => f32::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::quantization::{PqParams, pq_train_codebook};
    use crate::vector::core::vector::Vector;
    use crate::vector::index::pq_fastscan_storage::{PqFastScanPool, pack_codes_into_blocks};

    /// Train a small K=16 codebook from `n` random-ish vectors of
    /// dimension `m * sub_dim`. Deterministic for reproducible tests.
    fn train_small_codebook(
        m: usize,
        sub_dim: usize,
        n: usize,
    ) -> (PqParams, Vec<f32>, Vec<Vector>) {
        let dim = m * sub_dim;
        let params = PqParams::new(m as u16, K as u16, sub_dim as u16).unwrap();
        let mut vectors = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = Vec::with_capacity(dim);
            for d in 0..dim {
                // Cheap pseudo-random pattern with broad spread so the
                // K-means inside `pq_train_codebook` produces a real
                // (non-degenerate) codebook.
                let x = ((i * 31 + d * 17) % 257) as f32 - 128.0;
                let y = ((i.wrapping_mul(d + 1)) % 91) as f32 * 0.1;
                v.push(x + y);
            }
            vectors.push(Vector::new(v));
        }
        let codebook = pq_train_codebook(dim, params, &vectors).unwrap();
        (params, codebook, vectors)
    }

    /// Round-trip a single vector through encode → decode (codebook
    /// lookup) to obtain the reconstruction the kernel implicitly uses.
    fn decode_via_codebook(codes: &[u8], params: PqParams, codebook: &[f32]) -> Vec<f32> {
        let sub_dim = params.sub_dim as usize;
        let mut out = vec![0.0_f32; params.original_dim()];
        for (sub, &code) in codes.iter().enumerate() {
            let cb_base = sub * K * sub_dim;
            let centroid = &codebook
                [cb_base + (code as usize) * sub_dim..cb_base + (code as usize + 1) * sub_dim];
            out[sub * sub_dim..(sub + 1) * sub_dim].copy_from_slice(centroid);
        }
        out
    }

    fn encode_vec(v: &[f32], params: PqParams, codebook: &[f32]) -> Vec<u8> {
        crate::vector::core::quantization::pq_encode(v, params, codebook)
    }

    #[test]
    fn quantise_lut_round_trip_within_tolerance() {
        // Build a tiny synthetic LUT with a known range per sub.
        let m = 4;
        let mut lut_f32 = vec![0.0_f32; m * K];
        for sub in 0..m {
            for i in 0..K {
                lut_f32[sub * K + i] = (sub as f32) * 10.0 + (i as f32) * (1.0 + sub as f32);
            }
        }
        let (lut4, scale, bias) = quantise_lut_per_sub(&lut_f32, m);
        // Dequantise and compare against the original LUT. The max
        // per-sub error is `span / 255 / 2` for round-to-nearest.
        for sub in 0..m {
            let chunk = &lut_f32[sub * K..(sub + 1) * K];
            let span = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - chunk.iter().copied().fold(f32::INFINITY, f32::min);
            let tolerance = (span / 255.0 / 2.0) + 1e-5;
            for i in 0..K {
                let reconstructed = lut4[sub * K + i] as f32 * scale[sub] + bias[sub];
                let original = lut_f32[sub * K + i];
                assert!(
                    (reconstructed - original).abs() <= tolerance,
                    "sub={sub} i={i}: |{reconstructed} - {original}| > {tolerance}"
                );
            }
        }
    }

    #[test]
    fn quantise_lut_handles_constant_sub() {
        // A sub-quantiser whose LUT is entirely the same value
        // (degenerate range). The dequantiser should still return
        // that constant.
        let m = 2;
        let mut lut_f32 = vec![0.0_f32; m * K];
        // sub 0: all 7.5
        for slot in lut_f32.iter_mut().take(K) {
            *slot = 7.5;
        }
        // sub 1: linearly increasing
        for (i, slot) in lut_f32.iter_mut().enumerate().skip(K).take(K) {
            *slot = (i - K) as f32;
        }
        let (lut4, scale, bias) = quantise_lut_per_sub(&lut_f32, m);
        assert_eq!(scale[0], 0.0);
        assert_eq!(bias[0], 7.5);
        for &code in lut4.iter().take(K) {
            let r = code as f32 * scale[0] + bias[0];
            assert_eq!(r, 7.5);
        }
        // sub 1 is non-degenerate and round-trips within tolerance.
        let span1 = 15.0_f32; // 15 - 0
        let tol = span1 / 255.0 / 2.0 + 1e-5;
        for i in 0..K {
            let r = lut4[K + i] as f32 * scale[1] + bias[1];
            assert!((r - lut_f32[K + i]).abs() <= tol);
        }
    }

    #[test]
    fn f32_scalar_matches_direct_decoded_distance() {
        // The f32-LUT kernel must equal the squared distance between
        // the query and the reconstruction(codes through the codebook).
        let (params, codebook, vectors) = train_small_codebook(4, 3, 64);
        let query: Vec<f32> = vectors[0].data.to_vec();
        let pq_query = PqFastScanQuery::prepare(&query, params, &codebook).unwrap();

        // Encode and pack vectors[1..17] into a single block.
        let n_test = 16;
        let codes: Vec<Vec<u8>> = (1..=n_test)
            .map(|i| encode_vec(&vectors[i].data, params, &codebook))
            .collect();
        let m = params.m as usize;
        let packed = pack_codes_into_blocks(&codes, m);

        for (i, vec_codes) in codes.iter().enumerate() {
            let decoded = decode_via_codebook(vec_codes, params, &codebook);
            let mut direct_l2_sq = 0.0_f32;
            for d in 0..query.len() {
                let diff = query[d] - decoded[d];
                direct_l2_sq += diff * diff;
            }
            let direct_dist = direct_l2_sq.max(0.0).sqrt();
            let kernel_dist =
                distance_pq_fastscan_f32_scalar(DistanceMetric::Euclidean, &pq_query, &packed, i);
            assert!(
                (kernel_dist - direct_dist).abs() < 1e-3,
                "vec {i}: kernel={kernel_dist} direct={direct_dist}"
            );
        }
    }

    #[test]
    fn u8_scalar_matches_f32_scalar_within_quantisation_tolerance() {
        // The u8-LUT kernel must equal the f32-LUT kernel up to the
        // documented per-sub-quantiser quantisation drift.
        let (params, codebook, vectors) = train_small_codebook(8, 4, 64);
        let query: Vec<f32> = vectors[0].data.to_vec();
        let pq_query = PqFastScanQuery::prepare(&query, params, &codebook).unwrap();
        let m = params.m as usize;

        let codes: Vec<Vec<u8>> = (1..=BLOCK_SIZE)
            .map(|i| encode_vec(&vectors[i].data, params, &codebook))
            .collect();
        let packed = pack_codes_into_blocks(&codes, m);

        // Per-sub-quantiser quantisation drift accumulates over M
        // sub-quantisers — bound the total tolerance by the sum of
        // the per-sub max errors so the test remains a tight check
        // rather than a "looks roughly right" smoke screen.
        let mut total_l2_sq_tolerance = 0.0_f32;
        for sub in 0..m {
            let chunk = &pq_query.lut_f32[sub * K..(sub + 1) * K];
            let span = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                - chunk.iter().copied().fold(f32::INFINITY, f32::min);
            total_l2_sq_tolerance += span / 255.0 / 2.0;
        }

        for i in 0..codes.len() {
            let f32_dist =
                distance_pq_fastscan_f32_scalar(DistanceMetric::Euclidean, &pq_query, &packed, i);
            let u8_dist =
                distance_pq_fastscan_u8_scalar(DistanceMetric::Euclidean, &pq_query, &packed, i);
            // Both distances are sqrt(L2²); compare L2² to bound the
            // tolerance correctly.
            let f32_l2_sq = f32_dist * f32_dist;
            let u8_l2_sq = u8_dist * u8_dist;
            assert!(
                (u8_l2_sq - f32_l2_sq).abs() <= total_l2_sq_tolerance + 1e-4,
                "vec {i}: u8_l2_sq={u8_l2_sq} f32_l2_sq={f32_l2_sq} tol={total_l2_sq_tolerance}"
            );
        }
    }

    #[test]
    fn prepare_rejects_wrong_k() {
        let params = PqParams::new(4, 256, 2).unwrap();
        let codebook = vec![0.0f32; params.codebook_len()];
        let query = vec![0.0f32; params.original_dim()];
        let err = PqFastScanQuery::prepare(&query, params, &codebook).unwrap_err();
        assert!(err.to_string().contains("k == 16"));
    }

    #[test]
    fn prepare_rejects_dim_mismatch() {
        let params = PqParams::new(4, 16, 2).unwrap();
        let codebook = vec![0.0f32; params.codebook_len()];
        let query = vec![0.0f32; params.original_dim() + 1];
        let err = PqFastScanQuery::prepare(&query, params, &codebook).unwrap_err();
        assert!(
            err.to_string()
                .contains("does not match params.original_dim()")
        );
    }

    #[test]
    fn apply_metric_unsupported_returns_infinity() {
        // Manhattan / Angular / DotProduct return INFINITY through
        // either scalar kernel, mirroring `distance_pq_adc`.
        let (params, codebook, vectors) = train_small_codebook(4, 2, 16);
        let query: Vec<f32> = vectors[0].data.to_vec();
        let pq_query = PqFastScanQuery::prepare(&query, params, &codebook).unwrap();
        let _m = params.m as usize;
        let codes: Vec<Vec<u8>> = (0..BLOCK_SIZE)
            .map(|i| encode_vec(&vectors[i % vectors.len()].data, params, &codebook))
            .collect();
        let packed = pack_codes_into_blocks(&codes, params.m as usize);
        for metric in [
            DistanceMetric::Manhattan,
            DistanceMetric::Angular,
            DistanceMetric::DotProduct,
        ] {
            let d = distance_pq_fastscan_f32_scalar(metric, &pq_query, &packed, 0);
            assert!(
                d.is_infinite(),
                "f32 scalar metric={metric:?} should be inf"
            );
            let d = distance_pq_fastscan_u8_scalar(metric, &pq_query, &packed, 0);
            assert!(d.is_infinite(), "u8 scalar metric={metric:?} should be inf");
        }
    }

    #[test]
    fn end_to_end_pool_and_query_round_trip() {
        // Use the full PqFastScanPool path (which calls
        // pack_codes_into_blocks internally) and assert the f32
        // scalar kernel still produces the right distance for each
        // stored vector.
        let (params, codebook, vectors) = train_small_codebook(4, 3, 50);
        let query: Vec<f32> = vectors[0].data.to_vec();
        let pq_query = PqFastScanQuery::prepare(&query, params, &codebook).unwrap();

        let records: Vec<(u64, String, Vec<u8>)> = vectors
            .iter()
            .enumerate()
            .skip(1) // skip the query itself
            .map(|(i, v)| {
                (
                    i as u64,
                    "f".to_string(),
                    encode_vec(&v.data, params, &codebook),
                )
            })
            .collect();
        let pool = PqFastScanPool::build(params, codebook.clone(), records).unwrap();

        for vec_idx in 0..pool.n_vectors {
            let block_idx = vec_idx / BLOCK_SIZE;
            let in_block = vec_idx % BLOCK_SIZE;
            let block_base = block_idx * pool.block_stride();
            let block = &pool.packed[block_base..block_base + pool.block_stride()];

            let codes = pool.codes_at(vec_idx);
            let decoded = decode_via_codebook(&codes, params, &codebook);
            let mut l2 = 0.0_f32;
            for d in 0..query.len() {
                let diff = query[d] - decoded[d];
                l2 += diff * diff;
            }
            let direct = l2.max(0.0).sqrt();
            let kernel = distance_pq_fastscan_f32_scalar(
                DistanceMetric::Euclidean,
                &pq_query,
                block,
                in_block,
            );
            assert!(
                (kernel - direct).abs() < 1e-3,
                "vec_idx={vec_idx}: kernel={kernel} direct={direct}"
            );
        }
    }

    #[test]
    fn quantise_lut_global_round_trip_within_tolerance() {
        // Build a synthetic LUT with a known per-sub range. The
        // expected max element-wise error after dequantise is
        // `scale_global / 2` (round-to-nearest).
        let m = 4;
        let mut lut_f32 = vec![0.0_f32; m * K];
        for sub in 0..m {
            for i in 0..K {
                lut_f32[sub * K + i] = (sub as f32) * 20.0 + (i as f32) * (1.0 + sub as f32);
            }
        }
        let (lut4_global, scale_global, bias_sum) = quantise_lut_global(&lut_f32, m);

        // Reconstruct per-sub bias to dequantise: we know
        // `bias[sub] = min(lut_f32[sub * K..(sub + 1) * K])`.
        let mut bias = vec![0.0_f32; m];
        for sub in 0..m {
            bias[sub] = lut_f32[sub * K..(sub + 1) * K]
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
        }
        assert!((bias.iter().sum::<f32>() - bias_sum).abs() < 1e-4);

        let tolerance = scale_global / 2.0 + 1e-5;
        for sub in 0..m {
            for i in 0..K {
                let reconstructed = lut4_global[sub * K + i] as f32 * scale_global + bias[sub];
                let original = lut_f32[sub * K + i];
                assert!(
                    (reconstructed - original).abs() <= tolerance,
                    "sub={sub} i={i}: |{reconstructed} - {original}| > {tolerance}"
                );
            }
        }
    }

    #[test]
    fn quantise_lut_global_handles_degenerate_lut() {
        // All-zero LUT collapses to scale_global = 0 and bias_sum = 0.
        let m = 2;
        let lut_f32 = vec![0.0_f32; m * K];
        let (lut4, scale, bias_sum) = quantise_lut_global(&lut_f32, m);
        assert!(lut4.iter().all(|&c| c == 0));
        assert_eq!(scale, 0.0);
        assert_eq!(bias_sum, 0.0);
    }

    #[test]
    fn u8_global_scalar_matches_f32_scalar_within_tolerance() {
        // Codebook + query setup identical to the per-sub property
        // test above so we can reuse the expectation framework.
        let m = 4;
        let sub_dim = 2;
        let dim = m * sub_dim;
        let (params, codebook, vectors) = train_small_codebook(m, sub_dim, 256);
        let query: Vec<f32> = vectors[0].data.to_vec();
        assert_eq!(query.len(), dim);
        let q = PqFastScanQuery::prepare(&query, params, &codebook).unwrap();

        // Pack a couple of vectors into one block and walk every
        // in-block index.
        let codes: Vec<Vec<u8>> = (0..32)
            .map(|i| encode_vec(&vectors[i % vectors.len()].data, params, &codebook))
            .collect();
        let block = pack_codes_into_blocks(&codes, m);

        let max_error = (m as f32) * q.lut_scale_global / 2.0 + 1e-4;
        for vec_idx in 0..32 {
            let f32_val =
                distance_pq_fastscan_f32_scalar(DistanceMetric::Euclidean, &q, &block, vec_idx);
            let u8_val = distance_pq_fastscan_u8_global_scalar(
                DistanceMetric::Euclidean,
                &q,
                &block,
                vec_idx,
            );
            // The two are computed in different precision regimes
            // (sqrt of the L2² sum) so we compare the squared values.
            let diff_sq = (f32_val.powi(2) - u8_val.powi(2)).abs();
            assert!(
                diff_sq <= max_error,
                "vec {vec_idx}: |{f32_val}² - {u8_val}²| = {diff_sq} > {max_error}"
            );
        }
    }
}
