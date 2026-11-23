//! NEON SIMD kernels for the int8 scalar-quantization (SQ) distance
//! primitives (Issue [#652](https://github.com/mosuka/laurus/issues/652)).
//!
//! Mirrors [`crate::vector::core::sq_int8_avx2`] using aarch64 NEON
//! intrinsics, processing **16 bytes per iteration** in a 128-bit
//! register:
//!
//! - `dot_u8_to_i32_neon` — widen `u8 → u16`, multiply-accumulate
//!   with `vmlal_u16` into a u32x4 accumulator.
//! - `sq_diff_u8_to_i32_neon` — `vabdq_u8` gives `|a - b|` as u8;
//!   squaring the absolute difference equals `(a - b)²`, accumulated
//!   with `vmlal_u16`.
//! - `abs_diff_u8_to_i32_neon` — `vabdq_u8` then `vpadalq_u16`
//!   pairwise-accumulate into u32x4 (a u16 accumulator would overflow
//!   for `dim > 257`).
//!
//! NEON is mandatory on aarch64 (ARMv8 baseline), so the kernels are
//! gated solely on `#[cfg(target_arch = "aarch64")]` and need no
//! runtime feature check the way the AVX2 path does.
//!
//! # Bit-exactness / overflow
//!
//! As with the AVX2 kernels, the arithmetic is integer and the result
//! is bit-identical to the scalar reference. The per-pair bounds
//! (`≤ 65025` for dot / sq_diff, `≤ 255` for abs_diff) keep the u32
//! accumulator well within range for realistic dimensions, and the
//! `len % 16` tail is delegated to the scalar kernels in
//! [`crate::vector::core::distance_quantized`].

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use crate::vector::core::distance_quantized::{
    abs_diff_u8_to_i32_scalar, dot_u8_to_i32_scalar, sq_diff_u8_to_i32_scalar,
};

/// Returns `true` when the current target supports NEON.
///
/// On `aarch64` this is unconditionally `true` because NEON is
/// mandatory since ARMv8. On any other architecture it returns `false`.
#[inline]
pub fn is_neon_supported() -> bool {
    cfg!(target_arch = "aarch64")
}

/// NEON `Σ a[i] * b[i]` over u8 inputs, accumulating in i32.
///
/// The `len % 16` tail is handled by [`dot_u8_to_i32_scalar`].
///
/// # Safety
///
/// Caller must guarantee:
/// - Target architecture is `aarch64` (also enforced by `#[cfg]`).
/// - `a.len() == b.len()` (the callers assert this).
#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_u8_to_i32_neon(a: &[u8], b: &[u8]) -> i32 {
    // SAFETY: caller contract documented above; NEON intrinsics are
    // sound on aarch64 (baseline ISA) and every load reads exactly
    // `simd_len <= a.len()` bytes.
    unsafe {
        let len = a.len();
        let simd_len = len & !15; // round down to a multiple of 16
        let mut acc = vdupq_n_u32(0);
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut i = 0;
        while i < simd_len {
            let va = vld1q_u8(pa.add(i));
            let vb = vld1q_u8(pb.add(i));

            let a_lo = vmovl_u8(vget_low_u8(va)); // u16x8
            let a_hi = vmovl_u8(vget_high_u8(va));
            let b_lo = vmovl_u8(vget_low_u8(vb));
            let b_hi = vmovl_u8(vget_high_u8(vb));

            acc = vmlal_u16(acc, vget_low_u16(a_lo), vget_low_u16(b_lo));
            acc = vmlal_u16(acc, vget_high_u16(a_lo), vget_high_u16(b_lo));
            acc = vmlal_u16(acc, vget_low_u16(a_hi), vget_low_u16(b_hi));
            acc = vmlal_u16(acc, vget_high_u16(a_hi), vget_high_u16(b_hi));

            i += 16;
        }

        let mut total = vaddvq_u32(acc) as i32;
        if simd_len < len {
            total += dot_u8_to_i32_scalar(&a[simd_len..], &b[simd_len..]);
        }
        total
    }
}

/// NEON `Σ (a[i] - b[i])²` over u8 inputs, accumulating in i32.
///
/// `vabdq_u8` yields `|a - b|` as u8; squaring it equals `(a - b)²`.
/// The `len % 16` tail is handled by [`sq_diff_u8_to_i32_scalar`].
///
/// # Safety
///
/// Same contract as [`dot_u8_to_i32_neon`].
#[cfg(target_arch = "aarch64")]
pub unsafe fn sq_diff_u8_to_i32_neon(a: &[u8], b: &[u8]) -> i32 {
    // SAFETY: see the module-level and `dot_u8_to_i32_neon` contracts;
    // NEON is baseline on aarch64 and every load stays within `simd_len`.
    unsafe {
        let len = a.len();
        let simd_len = len & !15;
        let mut acc = vdupq_n_u32(0);
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut i = 0;
        while i < simd_len {
            let va = vld1q_u8(pa.add(i));
            let vb = vld1q_u8(pb.add(i));

            let ad = vabdq_u8(va, vb); // |a - b| as u8
            let ad_lo = vmovl_u8(vget_low_u8(ad)); // u16x8
            let ad_hi = vmovl_u8(vget_high_u8(ad));

            acc = vmlal_u16(acc, vget_low_u16(ad_lo), vget_low_u16(ad_lo));
            acc = vmlal_u16(acc, vget_high_u16(ad_lo), vget_high_u16(ad_lo));
            acc = vmlal_u16(acc, vget_low_u16(ad_hi), vget_low_u16(ad_hi));
            acc = vmlal_u16(acc, vget_high_u16(ad_hi), vget_high_u16(ad_hi));

            i += 16;
        }

        let mut total = vaddvq_u32(acc) as i32;
        if simd_len < len {
            total += sq_diff_u8_to_i32_scalar(&a[simd_len..], &b[simd_len..]);
        }
        total
    }
}

/// NEON `Σ |a[i] - b[i]|` over u8 inputs, accumulating in i32.
///
/// `vabdq_u8` yields `|a - b|` as u8; `vpadalq_u16` pairwise-adds the
/// widened u16 lanes into a u32x4 accumulator (a u16 accumulator would
/// overflow past `dim = 257`). The `len % 16` tail is handled by
/// [`abs_diff_u8_to_i32_scalar`].
///
/// # Safety
///
/// Same contract as [`dot_u8_to_i32_neon`].
#[cfg(target_arch = "aarch64")]
pub unsafe fn abs_diff_u8_to_i32_neon(a: &[u8], b: &[u8]) -> i32 {
    // SAFETY: see the module-level and `dot_u8_to_i32_neon` contracts;
    // NEON is baseline on aarch64 and every load stays within `simd_len`.
    unsafe {
        let len = a.len();
        let simd_len = len & !15;
        let mut acc = vdupq_n_u32(0);
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut i = 0;
        while i < simd_len {
            let va = vld1q_u8(pa.add(i));
            let vb = vld1q_u8(pb.add(i));

            let ad = vabdq_u8(va, vb); // |a - b| as u8
            let ad_lo = vmovl_u8(vget_low_u8(ad)); // u16x8
            let ad_hi = vmovl_u8(vget_high_u8(ad));

            acc = vpadalq_u16(acc, ad_lo);
            acc = vpadalq_u16(acc, ad_hi);

            i += 16;
        }

        let mut total = vaddvq_u32(acc) as i32;
        if simd_len < len {
            total += abs_diff_u8_to_i32_scalar(&a[simd_len..], &b[simd_len..]);
        }
        total
    }
}
