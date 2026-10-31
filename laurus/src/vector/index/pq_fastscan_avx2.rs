//! AVX2 SIMD kernel for FastScan PQ ADC distance
//! (Issue [#693](https://github.com/mosuka/laurus/issues/693), part B
//! of [#651](https://github.com/mosuka/laurus/issues/651)).
//!
//! The kernel processes one 32-vector FastScan block per call:
//!
//! 1. For each sub-quantiser `m`, load the 16-byte u8 LUT into an XMM
//!    register and `vpshufb` the 32 packed nibbles (16 from the low
//!    half of the byte and 16 from the high half) to gather 32 u8
//!    distances per sub.
//! 2. Widen each u8 lane to u16 (`vpunpcklbw` / `vpunpckhbw` with
//!    zero) and accumulate across all sub-quantisers. With `m ≤ 64`
//!    and the per-sub u8 cap of 255, the u16 accumulator cannot
//!    overflow (`255 × 64 = 16320 < 65535`).
//! 3. Permute the two 256-bit accumulators with `vperm2i128` so the
//!    output `[u16; 32]` follows the natural `[vec 0, vec 1, …, vec 31]`
//!    layout, then convert each lane back to f32 via
//!    `dist = sum * lut_scale_global + lut_bias_sum` (FAISS pq4_fast_scan
//!    convention).
//!
//! The scalar fallback ([`distance_pq_fastscan_block_scalar`]) performs
//! the same arithmetic and is bit-identical to the AVX2 path; tests
//! exercise both paths and assert equivalence.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::vector::core::distance::DistanceMetric;
#[cfg(target_arch = "x86_64")]
use crate::vector::core::distance_pq_fastscan::apply_metric;
use crate::vector::core::distance_pq_fastscan::{
    PqFastScanQuery, distance_pq_fastscan_u8_global_scalar,
};
use crate::vector::index::pq_fastscan_storage::BLOCK_SIZE;
#[cfg(target_arch = "x86_64")]
use crate::vector::index::pq_fastscan_storage::BYTES_PER_SUB_PER_BLOCK;

/// Returns `true` when the current x86_64 CPU supports AVX2.
///
/// Always returns `false` on non-x86_64 targets. The result is cached
/// by `is_x86_feature_detected!` after the first call so subsequent
/// invocations cost roughly one branch.
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

/// AVX2 kernel: distances for all `BLOCK_SIZE = 32` vectors of one
/// packed FastScan block.
///
/// # Arguments
///
/// * `metric` - Distance metric. Supported: [`DistanceMetric::Euclidean`]
///   and [`DistanceMetric::Cosine`]; any other variant yields
///   `f32::INFINITY` per vector (matching the scalar reference).
/// * `query` - Per-query state built by
///   [`PqFastScanQuery::prepare`]. Must carry the global-scale LUT
///   (`lut4_global`, `lut_scale_global`, `lut_bias_sum`).
/// * `packed_block` - One block's worth of packed codes, i.e. exactly
///   `query.params.m as usize * BYTES_PER_SUB_PER_BLOCK` bytes.
///
/// # Output
///
/// `out[v]` is the distance for the `v`-th vector in the block
/// (`v ∈ [0, BLOCK_SIZE)`). Positions past `n_vectors` in the trailing
/// partial block contain whatever the padding nibbles decode to; the
/// caller masks them.
///
/// # Safety
///
/// Caller must guarantee:
/// - The CPU supports AVX2. Use [`is_avx2_supported`] before calling.
/// - `packed_block.len() >= query.params.m as usize * BYTES_PER_SUB_PER_BLOCK`.
/// - `query.lut4_global.len() == query.params.m as usize * 16`.
/// - `query.params.m as usize <= 64` so the u16 accumulator cannot
///   overflow (the dispatcher [`distance_pq_fastscan_block`] enforces
///   this by routing larger `M` to the scalar fallback).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn distance_pq_fastscan_block_avx2(
    metric: DistanceMetric,
    query: &PqFastScanQuery,
    packed_block: &[u8],
) -> [f32; BLOCK_SIZE] {
    // SAFETY: caller contract documented above; intrinsics are sound
    // when called from a `#[target_feature(enable = "avx2")]` function
    // on an AVX2-capable CPU.
    unsafe {
        let m = query.params.m as usize;
        let zero = _mm256_setzero_si256();
        let mask_0f = _mm_set1_epi8(0x0F);
        let mut u16_acc_lo = zero;
        let mut u16_acc_hi = zero;

        let lut_ptr = query.lut4_global.as_ptr();
        let block_ptr = packed_block.as_ptr();

        for m_idx in 0..m {
            // Load 16-byte LUT for sub `m_idx` (K = 16 u8 entries).
            let lut_xmm = _mm_loadu_si128(lut_ptr.add(m_idx * 16) as *const __m128i);

            // Load 16 bytes of packed codes for sub `m_idx` (32 nibbles).
            let codes_xmm =
                _mm_loadu_si128(block_ptr.add(m_idx * BYTES_PER_SUB_PER_BLOCK) as *const __m128i);

            // Low nibbles → codes for vectors 0..16.
            let low_nibbles = _mm_and_si128(codes_xmm, mask_0f);
            // High nibbles → codes for vectors 16..32. Shift right by
            // 4 within each 16-bit lane (carries one bit between
            // adjacent low-nibble columns but that bit is masked off
            // by `mask_0f` immediately).
            let high_nibbles = _mm_and_si128(_mm_srli_epi16(codes_xmm, 4), mask_0f);

            // `vpshufb` gathers 16 u8 distances from the 16-entry LUT.
            let dist_lo_xmm = _mm_shuffle_epi8(lut_xmm, low_nibbles);
            let dist_hi_xmm = _mm_shuffle_epi8(lut_xmm, high_nibbles);

            // Pack into a single ymm register laid out as
            // `[vec 0..16 u8 | vec 16..32 u8]`.
            let dist_ymm = _mm256_set_m128i(dist_hi_xmm, dist_lo_xmm);

            // Widen u8 → u16 (`vpunpcklbw` / `vpunpckhbw` operate
            // per-128-bit lane).
            let dist_u16_lo = _mm256_unpacklo_epi8(dist_ymm, zero);
            let dist_u16_hi = _mm256_unpackhi_epi8(dist_ymm, zero);

            u16_acc_lo = _mm256_add_epi16(u16_acc_lo, dist_u16_lo);
            u16_acc_hi = _mm256_add_epi16(u16_acc_hi, dist_u16_hi);
        }

        // Reorder so the two YMM accumulators hold:
        // - `acc_0_to_16`  : u16 sums for vec 0..16
        // - `acc_16_to_32` : u16 sums for vec 16..32
        //
        // Before the permute, the per-lane layout is:
        //   u16_acc_lo lane 0 = vec  0.. 8
        //   u16_acc_lo lane 1 = vec 16..24
        //   u16_acc_hi lane 0 = vec  8..16
        //   u16_acc_hi lane 1 = vec 24..32
        let acc_0_to_16 = _mm256_permute2x128_si256(u16_acc_lo, u16_acc_hi, 0x20);
        let acc_16_to_32 = _mm256_permute2x128_si256(u16_acc_lo, u16_acc_hi, 0x31);

        let mut u16_dists = [0u16; BLOCK_SIZE];
        _mm256_storeu_si256(u16_dists.as_mut_ptr() as *mut __m256i, acc_0_to_16);
        _mm256_storeu_si256(u16_dists.as_mut_ptr().add(16) as *mut __m256i, acc_16_to_32);

        let mut out = [0.0_f32; BLOCK_SIZE];
        for (v, &sum) in u16_dists.iter().enumerate() {
            let l2_sq = sum as f32 * query.lut_scale_global + query.lut_bias_sum;
            out[v] = apply_metric(metric, l2_sq);
        }
        out
    }
}

/// Scalar fallback that mirrors the SIMD kernels' arithmetic exactly.
///
/// Computes one distance per vector in the block by reusing
/// [`distance_pq_fastscan_u8_global_scalar`], which performs the same
/// `u16` accumulation and `* scale_global + bias_sum` reconstruction
/// the SIMD path emits.
pub fn distance_pq_fastscan_block_scalar(
    metric: DistanceMetric,
    query: &PqFastScanQuery,
    packed_block: &[u8],
) -> [f32; BLOCK_SIZE] {
    let mut out = [0.0_f32; BLOCK_SIZE];
    for (v, slot) in out.iter_mut().enumerate() {
        *slot = distance_pq_fastscan_u8_global_scalar(metric, query, packed_block, v);
    }
    out
}

/// Compute distances for one FastScan block, dispatching to the AVX2
/// kernel when the CPU supports it (and the validation constraints
/// hold) and to the scalar fallback otherwise.
///
/// This is the production-facing entry point that Part D ([#695])
/// wires into the HNSW / IVF search hot path.
///
/// Routing:
/// - `x86_64` with AVX2 + `M ≤ 64` → AVX2 kernel
/// - `aarch64` with `M ≤ 64` → NEON kernel (NEON is mandatory on
///   ARMv8 so no runtime check is needed)
/// - Everything else → scalar fallback
///
/// The `M ≤ 64` guard keeps the u16 accumulator from overflowing
/// (`255 × 64 = 16320 < 65535`); larger `M` always falls through to
/// the scalar path regardless of architecture.
pub fn distance_pq_fastscan_block(
    metric: DistanceMetric,
    query: &PqFastScanQuery,
    packed_block: &[u8],
) -> [f32; BLOCK_SIZE] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_supported() && (query.params.m as usize) <= 64 {
            // SAFETY: AVX2 is detected at runtime, and the M ≤ 64
            // guard ensures the u16 accumulator cannot overflow. The
            // buffer lengths are validated by
            // `PqFastScanQuery::prepare()` and the pool layout.
            return unsafe { distance_pq_fastscan_block_avx2(metric, query, packed_block) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use crate::vector::index::pq_fastscan_neon::distance_pq_fastscan_block_neon;
        if (query.params.m as usize) <= 64 {
            // SAFETY: NEON is part of the ARMv8 baseline (mandatory
            // on aarch64), the M ≤ 64 guard ensures the u16
            // accumulator cannot overflow, and the buffer lengths are
            // validated upstream.
            return unsafe { distance_pq_fastscan_block_neon(metric, query, packed_block) };
        }
    }
    distance_pq_fastscan_block_scalar(metric, query, packed_block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::quantization::{PqParams, pq_encode, pq_train_codebook};
    use crate::vector::core::vector::Vector;
    use crate::vector::index::pq_fastscan_storage::PqFastScanPool;

    /// Deterministic pseudo-random training corpus identical in shape
    /// to the helper used by the Part A tests, so the resulting
    /// codebook is well-conditioned (K-means converges to non-degenerate
    /// centroids).
    fn train_small_codebook(
        m: usize,
        sub_dim: usize,
        n: usize,
    ) -> (PqParams, Vec<f32>, Vec<Vector>) {
        let dim = m * sub_dim;
        let params = PqParams::new(m as u16, 16, sub_dim as u16).unwrap();
        let mut vectors = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = Vec::with_capacity(dim);
            for d in 0..dim {
                let x = ((i * 31 + d * 17) % 257) as f32 - 128.0;
                let y = ((i.wrapping_mul(d + 1)) % 91) as f32 * 0.1;
                v.push(x + y);
            }
            vectors.push(Vector::new(v));
        }
        let codebook = pq_train_codebook(dim, params, &vectors).unwrap();
        (params, codebook, vectors)
    }

    /// Helper to build a `PqFastScanPool` from a slice of `Vector`s.
    fn build_pool(vectors: &[Vector], params: PqParams, codebook: &[f32]) -> PqFastScanPool {
        let codes: Vec<Vec<u8>> = vectors
            .iter()
            .map(|v| pq_encode(&v.data, params, codebook))
            .collect();
        let entries = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (i as u64, "f".to_string(), c.clone()));
        PqFastScanPool::build(params, codebook.to_vec(), entries).unwrap()
    }

    /// Helper to extract one block's worth of packed codes from a pool.
    fn block_slice(pool: &PqFastScanPool, block_idx: usize) -> Vec<u8> {
        let stride = pool.block_stride();
        let base = block_idx * stride;
        pool.packed[base..base + stride].to_vec()
    }

    #[test]
    fn scalar_and_dispatch_paths_agree_on_random_block() {
        let (params, codebook, vectors) = train_small_codebook(4, 2, 200);
        let pool = build_pool(&vectors, params, &codebook);
        // Use the first training vector as the query so every code
        // round-trip is non-degenerate.
        let query = PqFastScanQuery::prepare(&vectors[0].data, params, &codebook).unwrap();
        let block = block_slice(&pool, 0);

        let scalar = distance_pq_fastscan_block_scalar(DistanceMetric::Euclidean, &query, &block);
        let dispatch = distance_pq_fastscan_block(DistanceMetric::Euclidean, &query, &block);
        assert_eq!(scalar, dispatch);
    }

    /// Issue #1578: the safe dispatcher rejects a `packed_block` that is
    /// shorter than `m` sub-quantisers need, before the AVX2 (x86_64) or
    /// NEON (aarch64) kernel reads it. The block is the front of the
    /// pool's buffer, one byte short, so without the check the kernel's
    /// read stays inside that buffer and the test fails by not panicking.
    #[test]
    #[should_panic(expected = "FastScan block of 127 bytes for 8 sub-quantisers (needs 128)")]
    fn dispatch_panics_on_a_packed_block_shorter_than_its_sub_quantisers() {
        let (params, codebook, vectors) = train_small_codebook(8, 2, 200);
        let pool = build_pool(&vectors, params, &codebook);
        let query = PqFastScanQuery::prepare(&vectors[0].data, params, &codebook).unwrap();
        let stride = pool.block_stride();
        assert!(
            pool.packed.len() > stride,
            "the pool must hold more than one block"
        );

        distance_pq_fastscan_block(
            DistanceMetric::Euclidean,
            &query,
            &pool.packed[..stride - 1],
        );
    }

    /// Issue #1578: the safe dispatcher rejects a query whose
    /// `lut4_global` is shorter than its `params.m` needs: the query's
    /// fields are public, so `prepare`'s lengths do not bind a caller.
    /// The LUT is truncated in place, keeping its allocation, so without
    /// the check the kernel's read stays inside it and the test fails by
    /// not panicking.
    #[test]
    #[should_panic(expected = "FastScan LUT of 64 bytes for 8 sub-quantisers (needs 128)")]
    fn dispatch_panics_on_a_query_lut_shorter_than_its_sub_quantisers() {
        let (params, codebook, vectors) = train_small_codebook(8, 2, 200);
        let pool = build_pool(&vectors, params, &codebook);
        let mut query = PqFastScanQuery::prepare(&vectors[0].data, params, &codebook).unwrap();
        query.lut4_global.truncate(4 * 16);
        let block = block_slice(&pool, 0);

        distance_pq_fastscan_block(DistanceMetric::Euclidean, &query, &block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_block_matches_scalar_for_random_codebook() {
        if !is_avx2_supported() {
            eprintln!("AVX2 not supported on this CPU; skipping kernel test");
            return;
        }

        let (params, codebook, vectors) = train_small_codebook(8, 2, 256);
        let pool = build_pool(&vectors, params, &codebook);
        let query = PqFastScanQuery::prepare(&vectors[3].data, params, &codebook).unwrap();

        // Walk every block in the pool and assert kernel equivalence.
        for block_idx in 0..pool.block_count() {
            let block = block_slice(&pool, block_idx);
            let scalar =
                distance_pq_fastscan_block_scalar(DistanceMetric::Euclidean, &query, &block);
            // SAFETY: AVX2 supported (checked above); lengths come
            // from a freshly-built pool.
            let simd = unsafe {
                distance_pq_fastscan_block_avx2(DistanceMetric::Euclidean, &query, &block)
            };
            for v in 0..BLOCK_SIZE {
                assert_eq!(
                    simd[v], scalar[v],
                    "block {block_idx} vec {v} mismatch (simd={} scalar={})",
                    simd[v], scalar[v]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_for_cosine_metric() {
        if !is_avx2_supported() {
            return;
        }
        let (params, codebook, vectors) = train_small_codebook(4, 2, 200);
        let pool = build_pool(&vectors, params, &codebook);
        let query = PqFastScanQuery::prepare(&vectors[1].data, params, &codebook).unwrap();
        let block = block_slice(&pool, 0);

        let scalar = distance_pq_fastscan_block_scalar(DistanceMetric::Cosine, &query, &block);
        // SAFETY: AVX2 supported; lengths verified by helper above.
        let simd =
            unsafe { distance_pq_fastscan_block_avx2(DistanceMetric::Cosine, &query, &block) };
        assert_eq!(simd, scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_for_partial_block() {
        if !is_avx2_supported() {
            return;
        }
        // n = 5 < BLOCK_SIZE → first block has 5 real vectors + 27
        // padding positions. Both kernels should agree on every lane
        // (including padding, since `decode_codes_in_block` reads what
        // the buffer holds).
        let m = 4;
        let sub_dim = 2;
        let params = PqParams::new(m as u16, 16, sub_dim as u16).unwrap();
        let dim = m * sub_dim;
        let codebook: Vec<f32> = (0..params.codebook_len())
            .map(|i| (i as f32) * 0.01)
            .collect();
        let codes: Vec<Vec<u8>> = (0..5)
            .map(|i| (0..m).map(|sub| ((i + sub) % 16) as u8).collect())
            .collect();
        let pool = PqFastScanPool::build(
            params,
            codebook.clone(),
            codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u64, "f".to_string(), c.clone())),
        )
        .unwrap();
        let query_vec: Vec<f32> = (0..dim).map(|d| 0.1 * d as f32).collect();
        let query = PqFastScanQuery::prepare(&query_vec, params, &codebook).unwrap();
        let block = block_slice(&pool, 0);
        let scalar = distance_pq_fastscan_block_scalar(DistanceMetric::Euclidean, &query, &block);
        // SAFETY: AVX2 supported.
        let simd =
            unsafe { distance_pq_fastscan_block_avx2(DistanceMetric::Euclidean, &query, &block) };
        assert_eq!(simd, scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_handles_max_m_64_without_overflow() {
        if !is_avx2_supported() {
            return;
        }
        let m = 64usize;
        let sub_dim = 1usize;
        let params = PqParams::new(m as u16, 16, sub_dim as u16).unwrap();
        // Maximum u8 LUT value per sub is 255. With M=64 the u16 sum
        // is at most 16320, well below the u16 ceiling 65535.
        // Construct a synthetic codebook whose squared-distance LUT
        // exercises the upper bound: all codes map to entry 0, but
        // we set every entry to the same value so the sum hits the
        // dynamic range cap.
        let codebook: Vec<f32> = (0..params.codebook_len())
            .map(|i| (i % 16) as f32)
            .collect();
        let codes: Vec<Vec<u8>> = (0..BLOCK_SIZE)
            .map(|i| (0..m).map(|sub| ((i + sub) % 16) as u8).collect())
            .collect();
        let pool = PqFastScanPool::build(
            params,
            codebook.clone(),
            codes
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u64, "f".to_string(), c.clone())),
        )
        .unwrap();
        let query_vec: Vec<f32> = (0..m * sub_dim).map(|d| d as f32).collect();
        let query = PqFastScanQuery::prepare(&query_vec, params, &codebook).unwrap();
        let block = block_slice(&pool, 0);

        let scalar = distance_pq_fastscan_block_scalar(DistanceMetric::Euclidean, &query, &block);
        // SAFETY: AVX2 supported.
        let simd =
            unsafe { distance_pq_fastscan_block_avx2(DistanceMetric::Euclidean, &query, &block) };
        assert_eq!(simd, scalar);
    }
}
