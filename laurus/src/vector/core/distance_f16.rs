//! Float16 storage codec and distance kernels for dense vector fields
//! (Issue [#623](https://github.com/mosuka/laurus/issues/623)).
//!
//! A [`QuantizationMethod::Float16`](crate::vector::core::quantization::QuantizationMethod::Float16)
//! segment stores each vector as `dim` IEEE 754 binary16 values,
//! little-endian (`2 × dim` bytes). The query stays f32; each kernel widens
//! the stored row to f32 eight elements at a time and accumulates in f32,
//! so no row is ever decoded into a buffer on the search path.
//!
//! Rows are read in place from wherever the segment lives (a heap buffer,
//! or the mapped file in `Lazy` mode), so they are not aligned: every
//! kernel uses unaligned byte loads.
//!
//! # Dispatch
//!
//! | Target | Kernel | Widening |
//! | --- | --- | --- |
//! | aarch64 | [`crate::vector::core::f16_neon`] | `fcvtl` (base NEON, no runtime check) |
//! | x86_64 with F16C + AVX2 + FMA | [`crate::vector::core::f16_avx2`] | `vcvtph2ps`, detected at run time |
//! | everything else (incl. wasm32) | [`accumulate_portable`] | branch-free integer conversion |
//!
//! The backends sum in different orders, so their results agree to within
//! f32 rounding, not bit for bit.

use half::f16;
use wide::f32x8;

use crate::vector::core::distance::{DistanceMetric, PreparedQuery};

/// Bytes per stored element.
pub const F16_BYTES: usize = 2;

/// Accumulate `Σ q·x` and `Σ x²` (Cosine, Angular).
pub(crate) const OP_DOT_NORM: u8 = 0;
/// Accumulate `Σ q·x` (DotProduct).
pub(crate) const OP_DOT: u8 = 1;
/// Accumulate `Σ (q - x)²` (Euclidean).
pub(crate) const OP_SQ_DIFF: u8 = 2;
/// Accumulate `Σ |q - x|` (Manhattan).
pub(crate) const OP_ABS_DIFF: u8 = 3;

/// Convert to f16, saturating at `±f16::MAX` (65504) instead of
/// overflowing to infinity; NaN passes through unchanged.
///
/// `half`'s plain `f16::from_f32` follows IEEE 754 round-to-nearest and
/// produces `inf` past `f16::MAX`, which would poison every distance that
/// touches the value.
pub(crate) fn f16_saturating(v: f32) -> f16 {
    if v.is_nan() {
        return f16::from_f32(v);
    }
    f16::from_f32(v.clamp(f16::MIN.to_f32(), f16::MAX.to_f32()))
}

/// Append the stored form of `values` (`2 × values.len()` bytes) to `out`.
pub(crate) fn encode_f16_row(values: &[f32], out: &mut Vec<u8>) {
    out.reserve(values.len() * F16_BYTES);
    for &v in values {
        out.extend_from_slice(&f16_saturating(v).to_le_bytes());
    }
}

/// Decode a stored row into f32 values, appending them to `out`.
///
/// # Arguments
///
/// * `row` - The stored row, `2 × dim` bytes.
/// * `out` - Receives `row.len() / 2` values.
pub(crate) fn decode_f16_row(row: &[u8], out: &mut Vec<f32>) {
    out.extend(
        row.as_chunks::<F16_BYTES>()
            .0
            .iter()
            .map(|b| f16_bits_to_f32(u16::from_le_bytes(*b))),
    );
}

/// Exact binary16 → f32 conversion without branches or hardware support.
///
/// The algorithm of Maratyszcza's FP16 library: normal values are rebased
/// by adding to the exponent and scaling by 2⁻¹¹², subnormals are rebuilt
/// with a magic-number subtraction, and the two are selected by comparing
/// the shifted exponent. Infinities and NaNs come out of the normal path
/// unchanged in class. Kept free of branches so the portable kernel's
/// eight-lane loop vectorizes.
#[inline(always)]
pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let w = (h as u32) << 16;
    let sign = w & 0x8000_0000;
    let two_w = w.wrapping_add(w);
    // 2^-112: re-biases the f16 exponent (bias 15) to f32 (bias 127).
    let exp_scale = f32::from_bits(0x0780_0000);
    let normalized = f32::from_bits((two_w >> 4).wrapping_add(0xE0 << 23)) * exp_scale;
    let denormalized = f32::from_bits((two_w >> 17) | (126 << 23)) - 0.5;
    let bits = if two_w < (1 << 27) {
        denormalized.to_bits()
    } else {
        normalized.to_bits()
    };
    f32::from_bits(sign | bits)
}

/// Distance between a prepared f32 query and a stored Float16 row.
///
/// Mirrors [`DistanceMetric::distance_with_prepared`] on the decoded row,
/// including its zero-norm results (`1.0` for Cosine, `π` for Angular) and
/// the cosine clamp.
///
/// # Arguments
///
/// * `metric` - The field's distance metric.
/// * `query` - The query prepared with [`DistanceMetric::prepare_query`].
/// * `row` - The stored row, `2 × query.data.len()` bytes.
///
/// # Returns
///
/// The distance (lower is closer).
#[inline]
pub fn distance_f16(metric: DistanceMetric, query: &PreparedQuery<'_>, row: &[u8]) -> f32 {
    debug_assert_eq!(row.len(), query.data.len() * F16_BYTES);
    match metric {
        DistanceMetric::Cosine | DistanceMetric::Angular => {
            let (dot, norm_b_sq) = accumulate::<OP_DOT_NORM>(query.data, row);
            let norm_a = query.norm_sq.sqrt();
            let norm_b = norm_b_sq.sqrt();
            let zero = norm_a == 0.0 || norm_b == 0.0;
            let cosine = || (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
            match (metric, zero) {
                (DistanceMetric::Cosine, true) => 1.0,
                (DistanceMetric::Cosine, false) => 1.0 - cosine(),
                (_, true) => std::f32::consts::PI,
                (_, false) => cosine().acos(),
            }
        }
        DistanceMetric::DotProduct => -accumulate::<OP_DOT>(query.data, row).0,
        DistanceMetric::Euclidean => accumulate::<OP_SQ_DIFF>(query.data, row).0.sqrt(),
        DistanceMetric::Manhattan => accumulate::<OP_ABS_DIFF>(query.data, row).0,
    }
}

/// Run the fastest available kernel for `OP`.
///
/// # Returns
///
/// `(primary, Σ x²)`; the second value is only meaningful for
/// [`OP_DOT_NORM`] and is `0.0` otherwise.
#[inline]
pub(crate) fn accumulate<const OP: u8>(q: &[f32], row: &[u8]) -> (f32, f32) {
    debug_assert_eq!(row.len(), q.len() * F16_BYTES);
    // SAFETY: NEON is part of the aarch64 baseline, and the kernel reads
    // at most `q.len()` floats and `row.len()` bytes.
    #[cfg(target_arch = "aarch64")]
    let result = unsafe { crate::vector::core::f16_neon::accumulate_neon::<OP>(q, row) };
    #[cfg(target_arch = "x86_64")]
    let result = if crate::vector::core::f16_avx2::is_f16c_avx2_supported() {
        // SAFETY: F16C, AVX2 and FMA were detected at run time, and the
        // kernel reads at most `q.len()` floats and `row.len()` bytes.
        unsafe { crate::vector::core::f16_avx2::accumulate_avx2::<OP>(q, row) }
    } else {
        accumulate_portable::<OP>(q, row)
    };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let result = accumulate_portable::<OP>(q, row);
    result
}

/// Portable kernel over [`wide::f32x8`]: the fallback for targets without
/// a SIMD conversion (and wasm32), the reference the SIMD kernels are
/// tested against, and the shared tail handler for them.
#[doc(hidden)]
pub fn accumulate_portable<const OP: u8>(q: &[f32], row: &[u8]) -> (f32, f32) {
    debug_assert_eq!(row.len(), q.len() * F16_BYTES);
    let (q_chunks, q_rest) = q.as_chunks::<8>();
    let (r_chunks, r_rest) = row.as_chunks::<16>();
    let mut acc = f32x8::ZERO;
    let mut norm = f32x8::ZERO;
    for (qc, rc) in q_chunks.iter().zip(r_chunks) {
        let x = f32x8::from(std::array::from_fn::<f32, 8, _>(|j| {
            f16_bits_to_f32(u16::from_le_bytes([rc[2 * j], rc[2 * j + 1]]))
        }));
        let qv = f32x8::from(*qc);
        match OP {
            OP_DOT_NORM => {
                acc = qv.mul_add(x, acc);
                norm = x.mul_add(x, norm);
            }
            OP_DOT => acc = qv.mul_add(x, acc),
            OP_SQ_DIFF => {
                let d = qv - x;
                acc = d.mul_add(d, acc);
            }
            _ => acc += (qv - x).abs(),
        }
    }
    let mut primary = acc.reduce_add();
    let mut norm_sq = norm.reduce_add();
    let tail = accumulate_scalar::<OP>(q_rest, &r_rest[..q_rest.len() * F16_BYTES]);
    primary += tail.0;
    norm_sq += tail.1;
    (primary, norm_sq)
}

/// Scalar loop for kernel tails (fewer than one SIMD step of elements).
#[inline]
pub(crate) fn accumulate_scalar<const OP: u8>(q: &[f32], row: &[u8]) -> (f32, f32) {
    let mut primary = 0.0f32;
    let mut norm_sq = 0.0f32;
    for (&qv, b) in q.iter().zip(row.as_chunks::<F16_BYTES>().0) {
        let x = f16_bits_to_f32(u16::from_le_bytes(*b));
        match OP {
            OP_DOT_NORM => {
                primary += qv * x;
                norm_sq += x * x;
            }
            OP_DOT => primary += qv * x,
            OP_SQ_DIFF => primary += (qv - x) * (qv - x),
            _ => primary += (qv - x).abs(),
        }
    }
    (primary, norm_sq)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_rows(seed: u64, rows: usize, dim: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut flat: Vec<f32> = (0..rows * dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        for row in flat.chunks_exact_mut(dim) {
            crate::vector::core::vector::normalize_slice(row);
        }
        flat
    }

    fn encode(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_f16_row(values, &mut out);
        out
    }

    fn decoded(row: &[u8]) -> Vec<f32> {
        let mut out = Vec::new();
        decode_f16_row(row, &mut out);
        out
    }

    #[test]
    fn f16_bits_to_f32_matches_half_for_every_bit_pattern() {
        for bits in 0..=u16::MAX {
            let expected = f16::from_bits(bits).to_f32();
            let got = f16_bits_to_f32(bits);
            if expected.is_nan() {
                assert!(got.is_nan(), "bits {bits:#06x}: expected NaN, got {got}");
                assert_eq!(
                    got.is_sign_negative(),
                    expected.is_sign_negative(),
                    "bits {bits:#06x}: NaN sign"
                );
            } else {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "bits {bits:#06x}: expected {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn encode_saturates_and_keeps_nan() {
        let row = encode(&[1.0e6, -1.0e6, f32::INFINITY, f32::NAN, 0.5]);
        let back = decoded(&row);
        assert_eq!(back[0], 65504.0);
        assert_eq!(back[1], -65504.0);
        assert_eq!(back[2], 65504.0);
        assert!(back[3].is_nan());
        assert_eq!(back[4], 0.5);
    }

    #[test]
    fn encode_rounds_to_nearest_even() {
        // 1 + 2^-11 lies exactly between 1 and the next f16 (1 + 2^-10):
        // ties go to the even mantissa, i.e. 1.0.
        let tie = 1.0 + 2f32.powi(-11);
        assert_eq!(decoded(&encode(&[tie]))[0], 1.0);
        // Just above the tie rounds up.
        let above = 1.0 + 2f32.powi(-11) + 2f32.powi(-20);
        assert_eq!(decoded(&encode(&[above]))[0], 1.0 + 2f32.powi(-10));
    }

    #[test]
    fn decode_is_exact_for_values_that_round_trip() {
        // Includes the smallest normal (2^-14) and subnormal (2^-24) f16.
        let values = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            65504.0,
            2f32.powi(-14),
            2f32.powi(-24),
        ];
        assert_eq!(decoded(&encode(&values)), values);
    }

    const METRICS: [DistanceMetric; 5] = [
        DistanceMetric::Cosine,
        DistanceMetric::Angular,
        DistanceMetric::DotProduct,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
    ];

    /// `distance_f16` against the f32 path on the decoded row, for every
    /// metric and dimensions that exercise every tail length.
    #[test]
    fn distance_matches_f32_on_decoded_rows() {
        for dim in (1..=40).chain([127, 128, 383, 384, 768, 1031]) {
            let rows = unit_rows(dim as u64, 4, dim);
            let query = unit_rows(dim as u64 + 99, 1, dim);
            for metric in METRICS {
                let prepared = metric.prepare_query(&query);
                for row in rows.chunks_exact(dim) {
                    let stored = encode(row);
                    let expected = metric
                        .distance_with_prepared(&prepared, &decoded(&stored))
                        .unwrap();
                    let got = distance_f16(metric, &prepared, &stored);
                    assert!(
                        (got - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                        "{metric:?} dim {dim}: expected {expected}, got {got}"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_vectors_follow_the_f32_conventions() {
        let zero = encode(&[0.0; 16]);
        let query = [1.0f32; 16];
        let cos = DistanceMetric::Cosine;
        assert_eq!(distance_f16(cos, &cos.prepare_query(&query), &zero), 1.0);
        let ang = DistanceMetric::Angular;
        assert_eq!(
            distance_f16(ang, &ang.prepare_query(&query), &zero),
            std::f32::consts::PI
        );
        let zero_q = [0.0f32; 16];
        let row = encode(&[1.0; 16]);
        assert_eq!(distance_f16(cos, &cos.prepare_query(&zero_q), &row), 1.0);
    }

    /// Every backend compiled for this target agrees with the portable
    /// kernel, and the portable kernel agrees with an f64 reference.
    #[test]
    fn backends_agree_with_portable_and_reference() {
        for dim in [1usize, 7, 8, 9, 15, 16, 17, 100, 384, 1031] {
            let rows = unit_rows(dim as u64 * 7, 3, dim);
            let query = unit_rows(dim as u64 * 7 + 1, 1, dim);
            for row in rows.chunks_exact(dim) {
                let stored = encode(row);
                let x: Vec<f64> = decoded(&stored).iter().map(|&v| v as f64).collect();
                let q: Vec<f64> = query.iter().map(|&v| v as f64).collect();
                let dot: f64 = q.iter().zip(&x).map(|(a, b)| a * b).sum();
                let norm: f64 = x.iter().map(|b| b * b).sum();
                let sq: f64 = q.iter().zip(&x).map(|(a, b)| (a - b) * (a - b)).sum();
                let abs: f64 = q.iter().zip(&x).map(|(a, b)| (a - b).abs()).sum();
                let check = |name: &str, got: (f32, f32), want: (f64, f64)| {
                    let tol = |w: f64| 1e-5 * w.abs().max(1.0);
                    assert!(
                        (got.0 as f64 - want.0).abs() <= tol(want.0)
                            && (got.1 as f64 - want.1).abs() <= tol(want.1),
                        "{name} dim {dim}: got {got:?}, want {want:?}"
                    );
                };
                check(
                    "portable dot_norm",
                    accumulate_portable::<OP_DOT_NORM>(&query, &stored),
                    (dot, norm),
                );
                check(
                    "portable dot",
                    accumulate_portable::<OP_DOT>(&query, &stored),
                    (dot, 0.0),
                );
                check(
                    "portable sq",
                    accumulate_portable::<OP_SQ_DIFF>(&query, &stored),
                    (sq, 0.0),
                );
                check(
                    "portable abs",
                    accumulate_portable::<OP_ABS_DIFF>(&query, &stored),
                    (abs, 0.0),
                );
                check(
                    "dispatch dot_norm",
                    accumulate::<OP_DOT_NORM>(&query, &stored),
                    (dot, norm),
                );
                check(
                    "dispatch dot",
                    accumulate::<OP_DOT>(&query, &stored),
                    (dot, 0.0),
                );
                check(
                    "dispatch sq",
                    accumulate::<OP_SQ_DIFF>(&query, &stored),
                    (sq, 0.0),
                );
                check(
                    "dispatch abs",
                    accumulate::<OP_ABS_DIFF>(&query, &stored),
                    (abs, 0.0),
                );
            }
        }
    }

    /// Rows are read in place, so they may start at any byte offset.
    #[test]
    fn kernels_accept_unaligned_rows() {
        let dim = 33;
        let row = unit_rows(5, 1, dim);
        let query = unit_rows(6, 1, dim);
        let stored = encode(&row);
        let mut shifted = vec![0u8];
        shifted.extend_from_slice(&stored);
        let metric = DistanceMetric::Cosine;
        let prepared = metric.prepare_query(&query);
        assert_eq!(
            distance_f16(metric, &prepared, &stored).to_bits(),
            distance_f16(metric, &prepared, &shifted[1..]).to_bits()
        );
    }
}
