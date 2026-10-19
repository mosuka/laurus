//! F16C + AVX2 + FMA kernel for Float16 rows (Issue
//! [#623](https://github.com/mosuka/laurus/issues/623)).
//!
//! Widens eight stored binary16 values per step with `vcvtph2ps`
//! (`_mm256_cvtph_ps`) and accumulates in f32 with fused multiply-adds.
//! The CPU features are detected at run time, as for the int8 kernels in
//! [`crate::vector::core::sq_int8_avx2`]; without them the dispatcher in
//! [`crate::vector::core::distance_f16`] uses the portable kernel. The
//! `len % 8` tail goes to
//! [`crate::vector::core::distance_f16::accumulate_scalar`].

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::vector::core::distance_f16::{
    F16_BYTES, OP_ABS_DIFF, OP_DOT, OP_DOT_NORM, OP_SQ_DIFF, accumulate_scalar,
};

/// Returns `true` when the CPU supports F16C, AVX2 and FMA.
///
/// Always `false` off x86_64. `is_x86_feature_detected!` caches its
/// result, so repeated calls cost a few loads and branches.
#[inline]
pub fn is_f16c_avx2_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// F16C + AVX2 accumulation for one of the `OP_*` kernels in
/// [`crate::vector::core::distance_f16`].
///
/// # Returns
///
/// `(primary, Σ x²)`; the second value is `0.0` unless `OP` is
/// `OP_DOT_NORM`.
///
/// # Safety
///
/// Caller must guarantee that the CPU supports F16C, AVX2 and FMA (see
/// [`is_f16c_avx2_supported`]) and that `row.len() == 2 × q.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn accumulate_avx2<const OP: u8>(q: &[f32], row: &[u8]) -> (f32, f32) {
    debug_assert_eq!(row.len(), q.len() * F16_BYTES);
    // SAFETY: the caller guarantees the CPU features; every load reads
    // elements `i..i + 8` with `i + 8 <= simd_len <= q.len()`, i.e. 32
    // bytes of `q` and 16 bytes of `row`, through unaligned loads.
    unsafe {
        let len = q.len();
        let simd_len = len & !7;
        let pq = q.as_ptr();
        let pr = row.as_ptr();
        let sign = _mm256_set1_ps(-0.0);
        let mut acc = _mm256_setzero_ps();
        let mut norm = _mm256_setzero_ps();
        let mut i = 0;
        while i < simd_len {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(pr.add(i * F16_BYTES) as *const __m128i));
            let qv = _mm256_loadu_ps(pq.add(i));
            match OP {
                OP_DOT_NORM => {
                    acc = _mm256_fmadd_ps(qv, x, acc);
                    norm = _mm256_fmadd_ps(x, x, norm);
                }
                OP_DOT => acc = _mm256_fmadd_ps(qv, x, acc),
                OP_SQ_DIFF => {
                    let d = _mm256_sub_ps(qv, x);
                    acc = _mm256_fmadd_ps(d, d, acc);
                }
                _ => {
                    debug_assert_eq!(OP, OP_ABS_DIFF);
                    acc = _mm256_add_ps(acc, _mm256_andnot_ps(sign, _mm256_sub_ps(qv, x)));
                }
            }
            i += 8;
        }
        let tail = accumulate_scalar::<OP>(&q[simd_len..], &row[simd_len * F16_BYTES..]);
        (hsum_ps(acc) + tail.0, hsum_ps(norm) + tail.1)
    }
}

/// Horizontal sum of the eight lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hsum_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<0b01>(s, s));
    _mm_cvtss_f32(s)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::vector::core::distance_f16::{accumulate_portable, encode_f16_row, f16_bits_to_f32};

    #[test]
    fn avx2_matches_portable() {
        if !is_f16c_avx2_supported() {
            return;
        }
        for dim in [1usize, 7, 8, 9, 15, 16, 17, 100, 384, 1031] {
            let q: Vec<f32> = (0..dim)
                .map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0)
                .collect();
            let x: Vec<f32> = (0..dim)
                .map(|i| ((i * 53) % 97) as f32 / 48.0 - 1.0)
                .collect();
            let mut row = Vec::new();
            encode_f16_row(&x, &mut row);
            for (got, want) in [
                (
                    unsafe { accumulate_avx2::<OP_DOT_NORM>(&q, &row) },
                    accumulate_portable::<OP_DOT_NORM>(&q, &row),
                ),
                (
                    unsafe { accumulate_avx2::<OP_DOT>(&q, &row) },
                    accumulate_portable::<OP_DOT>(&q, &row),
                ),
                (
                    unsafe { accumulate_avx2::<OP_SQ_DIFF>(&q, &row) },
                    accumulate_portable::<OP_SQ_DIFF>(&q, &row),
                ),
                (
                    unsafe { accumulate_avx2::<OP_ABS_DIFF>(&q, &row) },
                    accumulate_portable::<OP_ABS_DIFF>(&q, &row),
                ),
            ] {
                let tol = |w: f32| 1e-5 * w.abs().max(1.0);
                assert!(
                    (got.0 - want.0).abs() <= tol(want.0) && (got.1 - want.1).abs() <= tol(want.1),
                    "dim {dim}: avx2 {got:?}, portable {want:?}"
                );
            }
        }
    }

    /// `vcvtph2ps` and the portable conversion agree on every bit pattern.
    #[test]
    fn cvtph_matches_portable_conversion() {
        if !is_f16c_avx2_supported() {
            return;
        }
        for start in (0..=u16::MAX as u32).step_by(8) {
            let bits: [u16; 8] = std::array::from_fn(|j| (start + j as u32) as u16);
            let got: [f32; 8] = unsafe {
                let mut out = [0f32; 8];
                _mm256_storeu_ps(
                    out.as_mut_ptr(),
                    convert(_mm_loadu_si128(bits.as_ptr() as *const __m128i)),
                );
                out
            };
            for (j, &b) in bits.iter().enumerate() {
                let want = f16_bits_to_f32(b);
                if want.is_nan() {
                    assert!(got[j].is_nan(), "bits {b:#06x}");
                } else {
                    assert_eq!(got[j].to_bits(), want.to_bits(), "bits {b:#06x}");
                }
            }
        }
    }

    #[target_feature(enable = "f16c,avx2")]
    fn convert(v: __m128i) -> __m256 {
        _mm256_cvtph_ps(v)
    }
}
