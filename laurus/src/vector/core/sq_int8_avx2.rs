//! AVX2 SIMD kernels for the int8 scalar-quantization (SQ) distance
//! primitives (Issue [#652](https://github.com/mosuka/laurus/issues/652)).
//!
//! These modernise the three per-pair integer accumulators used by
//! [`crate::vector::core::distance_quantized`]:
//!
//! - `dot_u8_to_i32_avx2` — `Σ a[i] * b[i]`
//! - `sq_diff_u8_to_i32_avx2` — `Σ (a[i] - b[i])²`
//! - `abs_diff_u8_to_i32_avx2` — `Σ |a[i] - b[i]|`
//!
//! Each kernel processes **32 bytes per iteration** in a 256-bit
//! register, versus the portable [`wide::i32x8`] fallback's 8 bytes.
//!
//! # Why not `PMADDUBSW`
//!
//! `_mm256_maddubs_epi16` computes `u8 × i8`, treating its second
//! operand as **signed** i8. Scalar quantization produces full-range
//! `u8` (`0..=255`), so candidate bytes ≥ 128 would be misread as
//! negative. Instead the dot / sq_diff kernels widen both operands
//! `u8 → i16` (`_mm256_unpacklo_epi8` / `_mm256_unpackhi_epi8` with a
//! zero register) and use `_mm256_madd_epi16` (`i16 × i16 → i32`,
//! adjacent-pair summed). The abs_diff kernel uses `_mm256_sad_epu8`,
//! which sums `|a - b|` over unsigned bytes directly in one
//! instruction.
//!
//! # Bit-exactness
//!
//! All arithmetic is integer, so the SIMD result is bit-identical to
//! the scalar reference regardless of summation order. The tail
//! (`len % 32` trailing bytes) is delegated to the scalar kernels in
//! [`crate::vector::core::distance_quantized`], keeping a single source
//! of truth for the remainder.
//!
//! # Overflow
//!
//! The per-pair bounds match the scalar path: `u8 * u8 ≤ 65025`,
//! `(a - b)² ≤ 65025`, `|a - b| ≤ 255`. For all realistic dimensions
//! (`dim ≤ 4096`) the i32 / u64 accumulators stay well below their
//! maxima, exactly as documented on the scalar kernels.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::vector::core::distance_quantized::{
    abs_diff_u8_to_i32_scalar, dot_u8_to_i32_scalar, sq_diff_u8_to_i32_scalar,
};

/// Returns `true` when the current x86_64 CPU supports AVX2.
///
/// Always returns `false` on non-x86_64 targets. The result of
/// `is_x86_feature_detected!` is cached after the first call, so
/// subsequent invocations cost roughly one branch.
#[inline]
pub fn is_avx2_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX2 `Σ a[i] * b[i]` over u8 inputs, accumulating in i32.
///
/// Widens both operands `u8 → i16` and multiply-accumulates with
/// `_mm256_madd_epi16`, 32 elements per iteration. The `len % 32` tail
/// is handled by [`dot_u8_to_i32_scalar`].
///
/// # Safety
///
/// Caller must guarantee:
/// - The CPU supports AVX2. Use [`is_avx2_supported`] before calling.
/// - `a.len() == b.len()` (the callers assert this).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_u8_to_i32_avx2(a: &[u8], b: &[u8]) -> i32 {
    // SAFETY: caller contract documented above; the intrinsics are
    // sound when called from a `#[target_feature(enable = "avx2")]`
    // function on an AVX2-capable CPU, and every load reads exactly
    // `simd_len <= a.len()` bytes.
    unsafe {
        let len = a.len();
        let simd_len = len & !31; // round down to a multiple of 32
        let zero = _mm256_setzero_si256();
        let mut acc = _mm256_setzero_si256();
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut i = 0;
        while i < simd_len {
            let va = _mm256_loadu_si256(pa.add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(pb.add(i) as *const __m256i);

            // Widen u8 -> i16 (per 128-bit lane; lane ordering is
            // irrelevant to the final horizontal sum).
            let a_lo = _mm256_unpacklo_epi8(va, zero);
            let a_hi = _mm256_unpackhi_epi8(va, zero);
            let b_lo = _mm256_unpacklo_epi8(vb, zero);
            let b_hi = _mm256_unpackhi_epi8(vb, zero);

            // i16 * i16 -> i32, adjacent pairs summed.
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_lo, b_lo));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_hi, b_hi));

            i += 32;
        }

        let mut total = hsum_epi32(acc);
        if simd_len < len {
            total += dot_u8_to_i32_scalar(&a[simd_len..], &b[simd_len..]);
        }
        total
    }
}

/// AVX2 `Σ (a[i] - b[i])²` over u8 inputs, accumulating in i32.
///
/// Widens both operands `u8 → i16`, subtracts in i16 (range
/// `-255..=255`), then squares and pair-sums with
/// `_mm256_madd_epi16`. The `len % 32` tail is handled by
/// [`sq_diff_u8_to_i32_scalar`].
///
/// # Safety
///
/// Same contract as [`dot_u8_to_i32_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sq_diff_u8_to_i32_avx2(a: &[u8], b: &[u8]) -> i32 {
    // SAFETY: see the module-level and `dot_u8_to_i32_avx2` contracts;
    // AVX2 is required and every load stays within `simd_len`.
    unsafe {
        let len = a.len();
        let simd_len = len & !31;
        let zero = _mm256_setzero_si256();
        let mut acc = _mm256_setzero_si256();
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut i = 0;
        while i < simd_len {
            let va = _mm256_loadu_si256(pa.add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(pb.add(i) as *const __m256i);

            let a_lo = _mm256_unpacklo_epi8(va, zero);
            let a_hi = _mm256_unpackhi_epi8(va, zero);
            let b_lo = _mm256_unpacklo_epi8(vb, zero);
            let b_hi = _mm256_unpackhi_epi8(vb, zero);

            let d_lo = _mm256_sub_epi16(a_lo, b_lo);
            let d_hi = _mm256_sub_epi16(a_hi, b_hi);

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_lo, d_lo));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_hi, d_hi));

            i += 32;
        }

        let mut total = hsum_epi32(acc);
        if simd_len < len {
            total += sq_diff_u8_to_i32_scalar(&a[simd_len..], &b[simd_len..]);
        }
        total
    }
}

/// AVX2 `Σ |a[i] - b[i]|` over u8 inputs, accumulating in i32.
///
/// Uses `_mm256_sad_epu8`, which computes the sum of absolute
/// differences of unsigned bytes over each 8-byte group in one
/// instruction, accumulated in four u64 lanes. The `len % 32` tail is
/// handled by [`abs_diff_u8_to_i32_scalar`].
///
/// # Safety
///
/// Same contract as [`dot_u8_to_i32_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn abs_diff_u8_to_i32_avx2(a: &[u8], b: &[u8]) -> i32 {
    // SAFETY: see the module-level and `dot_u8_to_i32_avx2` contracts;
    // AVX2 is required and every load stays within `simd_len`.
    unsafe {
        let len = a.len();
        let simd_len = len & !31;
        let mut acc = _mm256_setzero_si256();
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut i = 0;
        while i < simd_len {
            let va = _mm256_loadu_si256(pa.add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(pb.add(i) as *const __m256i);
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
            i += 32;
        }

        // Horizontal sum of the four u64 lanes. The total for realistic
        // dims (`dim * 255`) fits comfortably in i32.
        let mut lanes = [0i64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
        let mut total = (lanes[0] + lanes[1] + lanes[2] + lanes[3]) as i32;
        if simd_len < len {
            total += abs_diff_u8_to_i32_scalar(&a[simd_len..], &b[simd_len..]);
        }
        total
    }
}

/// Horizontal sum of the eight i32 lanes of a 256-bit register.
///
/// # Safety
///
/// Requires AVX2; only called from `#[target_feature(enable = "avx2")]`
/// kernels in this module.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    // SAFETY: AVX2 guaranteed by the `#[target_feature]` attribute and
    // the calling kernels' own contracts.
    unsafe {
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v);
        lanes.iter().sum()
    }
}
