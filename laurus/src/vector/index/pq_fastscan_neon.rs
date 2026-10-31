//! NEON SIMD kernel for FastScan PQ ADC distance
//! (Issue [#694](https://github.com/mosuka/laurus/issues/694), part C
//! of [#651](https://github.com/mosuka/laurus/issues/651)).
//!
//! Mirrors the AVX2 kernel in
//! [`crate::vector::index::pq_fastscan_avx2`] using NEON intrinsics:
//!
//! 1. For each sub-quantiser `m`, load the 16-byte u8 LUT and the
//!    16-byte packed nibble block into `uint8x16_t` registers.
//! 2. Split low / high nibbles, gather 32 u8 distances via
//!    `vqtbl1q_u8` (NEON's equivalent of `vpshufb`).
//! 3. Widen each u8 lane to u16 (`vmovl_u8` for the lower half of a
//!    16-byte register, `vmovl_high_u8` for the upper half) and
//!    accumulate across all sub-quantisers in four `uint16x8_t`
//!    registers covering vectors 0..8, 8..16, 16..24, 24..32.
//! 4. Store the four u16 accumulators sequentially into a `[u16; 32]`
//!    buffer (NEON registers are already in natural order — no
//!    `vperm2i128`-style fix-up needed unlike AVX2) and convert to f32
//!    via `dist = sum * lut_scale_global + lut_bias_sum`.
//!
//! NEON is mandatory on aarch64 (ARMv8 baseline), so the kernel is
//! gated solely on `#[cfg(target_arch = "aarch64")]` and does not need
//! a runtime feature check the way the AVX2 path does.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::vector::core::distance::DistanceMetric;
#[cfg(target_arch = "aarch64")]
use crate::vector::core::distance_pq_fastscan::apply_metric;
use crate::vector::core::distance_pq_fastscan::{
    PqFastScanQuery, distance_pq_fastscan_u8_global_scalar,
};
use crate::vector::index::pq_fastscan_storage::BLOCK_SIZE;
#[cfg(target_arch = "aarch64")]
use crate::vector::index::pq_fastscan_storage::BYTES_PER_SUB_PER_BLOCK;

/// Returns `true` when the current target supports NEON.
///
/// On `aarch64` this is unconditionally `true` because NEON is
/// mandatory since ARMv8. On any other architecture this returns
/// `false`.
#[inline]
pub fn is_neon_supported() -> bool {
    cfg!(target_arch = "aarch64")
}

/// NEON kernel: distances for all `BLOCK_SIZE = 32` vectors of one
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
/// - Target architecture is `aarch64`. Use [`is_neon_supported`]
///   before calling (the function is also `#[cfg]`-gated so it is
///   only compiled on aarch64).
/// - `packed_block.len() >= query.params.m as usize * BYTES_PER_SUB_PER_BLOCK`.
/// - `query.lut4_global.len() >= query.params.m as usize * 16`.
/// - `query.params.m as usize <= 64` so the u16 accumulator cannot
///   overflow (`255 × 64 = 16320 < 65535`).
///
/// The safe dispatcher
/// [`crate::vector::index::pq_fastscan_avx2::distance_pq_fastscan_block`]
/// checks the two lengths (Issue #1578) and routes larger `M` to the
/// scalar fallback.
#[cfg(target_arch = "aarch64")]
pub unsafe fn distance_pq_fastscan_block_neon(
    metric: DistanceMetric,
    query: &PqFastScanQuery,
    packed_block: &[u8],
) -> [f32; BLOCK_SIZE] {
    // SAFETY: caller contract documented above; NEON intrinsics are
    // sound on aarch64 where NEON is part of the baseline ISA.
    unsafe {
        let m = query.params.m as usize;
        let mut acc_0_7 = vdupq_n_u16(0);
        let mut acc_8_15 = vdupq_n_u16(0);
        let mut acc_16_23 = vdupq_n_u16(0);
        let mut acc_24_31 = vdupq_n_u16(0);

        let lut_ptr = query.lut4_global.as_ptr();
        let block_ptr = packed_block.as_ptr();

        for m_idx in 0..m {
            // Load 16-byte LUT for sub `m_idx` (K = 16 u8 entries).
            let lut = vld1q_u8(lut_ptr.add(m_idx * 16));
            // Load 16 bytes of packed codes for sub `m_idx` (32 nibbles).
            let codes = vld1q_u8(block_ptr.add(m_idx * BYTES_PER_SUB_PER_BLOCK));

            // Low nibbles → codes for vectors 0..16.
            let low_nib = vandq_u8(codes, vdupq_n_u8(0x0F));
            // High nibbles → codes for vectors 16..32. `vshrq_n_u8`
            // is byte-wise (no carry between adjacent bytes), so no
            // AND mask is needed afterwards.
            let high_nib = vshrq_n_u8(codes, 4);

            // `vqtbl1q_u8` gathers 16 u8 distances from the 16-entry
            // LUT (NEON equivalent of `vpshufb`).
            let dist_lo = vqtbl1q_u8(lut, low_nib);
            let dist_hi = vqtbl1q_u8(lut, high_nib);

            // Widen u8 → u16 (lower / upper 8 bytes of each 16-byte
            // register) and accumulate.
            acc_0_7 = vaddq_u16(acc_0_7, vmovl_u8(vget_low_u8(dist_lo)));
            acc_8_15 = vaddq_u16(acc_8_15, vmovl_high_u8(dist_lo));
            acc_16_23 = vaddq_u16(acc_16_23, vmovl_u8(vget_low_u8(dist_hi)));
            acc_24_31 = vaddq_u16(acc_24_31, vmovl_high_u8(dist_hi));
        }

        // NEON registers are already in natural [vec 0..32] order;
        // no permute step required (unlike AVX2's `vperm2i128`).
        let mut u16_dists = [0u16; BLOCK_SIZE];
        vst1q_u16(u16_dists.as_mut_ptr(), acc_0_7);
        vst1q_u16(u16_dists.as_mut_ptr().add(8), acc_8_15);
        vst1q_u16(u16_dists.as_mut_ptr().add(16), acc_16_23);
        vst1q_u16(u16_dists.as_mut_ptr().add(24), acc_24_31);

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
/// Identical to
/// [`crate::vector::index::pq_fastscan_avx2::distance_pq_fastscan_block_scalar`]
/// — exposed here too so callers that route through `pq_fastscan_neon`
/// (e.g. tests in this module) do not need to import the AVX2 module
/// on non-x86_64 builds.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::quantization::{PqParams, pq_encode, pq_train_codebook};
    use crate::vector::core::vector::Vector;
    use crate::vector::index::pq_fastscan_storage::PqFastScanPool;

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

    fn block_slice(pool: &PqFastScanPool, block_idx: usize) -> Vec<u8> {
        let stride = pool.block_stride();
        let base = block_idx * stride;
        pool.packed[base..base + stride].to_vec()
    }

    #[test]
    fn scalar_path_returns_dispatch_compatible_distances() {
        // Smoke test: the scalar wrapper here matches the per-vector
        // reference (`distance_pq_fastscan_u8_global_scalar`). This
        // does not call NEON intrinsics — kept arch-agnostic so the
        // test runs on every host even when the SIMD kernel does not.
        let (params, codebook, vectors) = train_small_codebook(4, 2, 200);
        let pool = build_pool(&vectors, params, &codebook);
        let query = PqFastScanQuery::prepare(&vectors[0].data, params, &codebook).unwrap();
        let block = block_slice(&pool, 0);
        let scalar = distance_pq_fastscan_block_scalar(DistanceMetric::Euclidean, &query, &block);
        for (v, &got) in scalar.iter().enumerate() {
            let one_off =
                distance_pq_fastscan_u8_global_scalar(DistanceMetric::Euclidean, &query, &block, v);
            assert_eq!(got, one_off, "vec {v} mismatch");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_block_matches_scalar_for_random_codebook() {
        if !is_neon_supported() {
            eprintln!("NEON not supported on this target; skipping kernel test");
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
            // SAFETY: NEON is mandatory on aarch64 and the buffers
            // come from a freshly-built pool.
            let simd = unsafe {
                distance_pq_fastscan_block_neon(DistanceMetric::Euclidean, &query, &block)
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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_for_cosine_metric() {
        if !is_neon_supported() {
            return;
        }
        let (params, codebook, vectors) = train_small_codebook(4, 2, 200);
        let pool = build_pool(&vectors, params, &codebook);
        let query = PqFastScanQuery::prepare(&vectors[1].data, params, &codebook).unwrap();
        let block = block_slice(&pool, 0);

        let scalar = distance_pq_fastscan_block_scalar(DistanceMetric::Cosine, &query, &block);
        // SAFETY: NEON mandatory on aarch64.
        let simd =
            unsafe { distance_pq_fastscan_block_neon(DistanceMetric::Cosine, &query, &block) };
        assert_eq!(simd, scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_for_partial_block() {
        if !is_neon_supported() {
            return;
        }
        // n = 5 < BLOCK_SIZE → first block has 5 real vectors + 27
        // padding positions. Both kernels should agree on every lane.
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
        // SAFETY: NEON mandatory on aarch64.
        let simd =
            unsafe { distance_pq_fastscan_block_neon(DistanceMetric::Euclidean, &query, &block) };
        assert_eq!(simd, scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_handles_max_m_64_without_overflow() {
        if !is_neon_supported() {
            return;
        }
        let m = 64usize;
        let sub_dim = 1usize;
        let params = PqParams::new(m as u16, 16, sub_dim as u16).unwrap();
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
        // SAFETY: NEON mandatory on aarch64.
        let simd =
            unsafe { distance_pq_fastscan_block_neon(DistanceMetric::Euclidean, &query, &block) };
        assert_eq!(simd, scalar);
    }
}
