//! NEON kernel for Float16 rows (Issue
//! [#623](https://github.com/mosuka/laurus/issues/623)).
//!
//! Widens eight stored binary16 values per step with `fcvtl`
//! (`vcvt_f32_f16`, part of the aarch64 baseline) and accumulates in f32
//! with fused multiply-adds, two accumulators per sum. The `len % 8` tail
//! goes to [`crate::vector::core::distance_f16::accumulate_scalar`].
//!
//! Phase 0 of #623 measured this kernel at 0.84–0.99× the instructions of
//! the f32 kernel over the same dimension, and 45–49% fewer than decoding
//! the row first.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use crate::vector::core::distance_f16::{
    F16_BYTES, OP_ABS_DIFF, OP_DOT, OP_DOT_NORM, OP_SQ_DIFF, accumulate_scalar,
};

/// NEON accumulation for one of the `OP_*` kernels in
/// [`crate::vector::core::distance_f16`].
///
/// # Returns
///
/// `(primary, Σ x²)`; the second value is `0.0` unless `OP` is
/// `OP_DOT_NORM`.
///
/// # Safety
///
/// Caller must guarantee `row.len() == 2 × q.len()`. NEON itself is part
/// of the aarch64 baseline.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn accumulate_neon<const OP: u8>(q: &[f32], row: &[u8]) -> (f32, f32) {
    debug_assert_eq!(row.len(), q.len() * F16_BYTES);
    // SAFETY: every load reads elements `i..i + 8` with `i + 8 <= simd_len
    // <= q.len()`, i.e. 32 bytes of `q` and 16 bytes of `row`; `vld1q_u8`
    // has no alignment requirement.
    unsafe {
        let len = q.len();
        let simd_len = len & !7;
        let pq = q.as_ptr();
        let pr = row.as_ptr();
        let mut a0 = vdupq_n_f32(0.0);
        let mut a1 = vdupq_n_f32(0.0);
        let mut n0 = vdupq_n_f32(0.0);
        let mut n1 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i < simd_len {
            let h = vreinterpretq_u16_u8(vld1q_u8(pr.add(i * F16_BYTES)));
            let lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h)));
            let hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h)));
            let q0 = vld1q_f32(pq.add(i));
            let q1 = vld1q_f32(pq.add(i + 4));
            match OP {
                OP_DOT_NORM => {
                    a0 = vfmaq_f32(a0, q0, lo);
                    a1 = vfmaq_f32(a1, q1, hi);
                    n0 = vfmaq_f32(n0, lo, lo);
                    n1 = vfmaq_f32(n1, hi, hi);
                }
                OP_DOT => {
                    a0 = vfmaq_f32(a0, q0, lo);
                    a1 = vfmaq_f32(a1, q1, hi);
                }
                OP_SQ_DIFF => {
                    let d0 = vsubq_f32(q0, lo);
                    let d1 = vsubq_f32(q1, hi);
                    a0 = vfmaq_f32(a0, d0, d0);
                    a1 = vfmaq_f32(a1, d1, d1);
                }
                _ => {
                    debug_assert_eq!(OP, OP_ABS_DIFF);
                    a0 = vaddq_f32(a0, vabdq_f32(q0, lo));
                    a1 = vaddq_f32(a1, vabdq_f32(q1, hi));
                }
            }
            i += 8;
        }
        let tail = accumulate_scalar::<OP>(&q[simd_len..], &row[simd_len * F16_BYTES..]);
        (
            vaddvq_f32(vaddq_f32(a0, a1)) + tail.0,
            vaddvq_f32(vaddq_f32(n0, n1)) + tail.1,
        )
    }
}
