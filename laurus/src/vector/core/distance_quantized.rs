//! int8 SIMD distance kernel for per-segment global affine SQ.
//!
//! Companion to [`crate::vector::core::distance`] (the f32 path) and
//! [`crate::vector::core::quantization`] (the SQ encoder). After
//! Issue #481 Stage 1 the HNSW / Flat / IVF search hot path uses
//! the functions in this module instead of the f32 kernels.
//!
//! # Algorithm
//!
//! With per-segment global affine SQ, every f32 element is encoded
//! as `q = clamp(round((v - offset) / scale), 0, 255)`, sharing one
//! `(offset, scale)` pair across the whole segment. The dot product
//! reconstructs from the quantized form via the affine identity:
//!
//! ```text
//! a · b ≈ N·offset² + scale·offset·(Σaq + Σbq) + scale²·Σ(aq · bq)
//! ```
//!
//! Of those four terms only `Σ(aq · bq)` is per-pair: it is computed
//! in i32 SIMD using [`dot_u8_to_i32`]. The other three collapse into
//! constants once the candidate's `sum_q` is loaded — that constant
//! is what makes the int8 kernel fast vs the f32 kernel.
//!
//! For Euclidean and Manhattan, the `offset` cancels in `(a - b)`, so
//! distance computation reduces to a pure int8 difference accumulator
//! plus a single `scale²` (Euclidean) or `scale` (Manhattan) post-multiply.
//!
//! # SIMD strategy
//!
//! All accumulators use [`wide::i32x8`] (8-wide i32 lanes, portable
//! across x86_64 / aarch64 / wasm32 via the `wide` crate). Per-element
//! `u8 → i32` zero-extension happens at load time. `i32` is wide
//! enough to hold any per-pair product (`u8 * u8 ≤ 65025`) and the
//! per-chunk accumulation across realistic dimensions
//! (`dim ≤ 4096` keeps the running sum well below `i32::MAX`).

use wide::i32x8;

use crate::vector::core::distance::DistanceMetric;
use crate::vector::core::quantization::{PqParams, QuantizedVectorMeta, ScalarQuantParams};

/// SIMD block size (bytes) the int8 kernels consume per iteration.
///
/// The AVX2 / NEON kernels process a whole number of these blocks; a
/// payload padded to a multiple of `SIMD_BLOCK` has no scalar tail.
pub const SIMD_BLOCK: usize = 32;

/// Round `dim` up to the next multiple of [`SIMD_BLOCK`].
///
/// Used to size the zero-padded int8 payloads so the distance kernels
/// run over whole 32-byte blocks with no scalar remainder. This is the
/// canonical definition shared by the in-memory
/// [`crate::vector::index::quantized_storage::QuantizedVectorPool`] and
/// the per-query [`QuantizedQuery`] padding, so both agree on the stride.
#[inline]
pub const fn padded_dim(dim: usize) -> usize {
    // SIMD_BLOCK is a power of two, so mask off the low bits after adding
    // (SIMD_BLOCK - 1) to round up.
    (dim + SIMD_BLOCK - 1) & !(SIMD_BLOCK - 1)
}

/// Per-query state cached once per HNSW search invocation.
///
/// Built by [`Self::prepare`] from the f32 query and the segment-level
/// quantization params. Subsequent [`distance_quantized`] calls read
/// only from this struct (no further f32 access required for
/// Euclidean / Manhattan / DotProduct; Cosine / Angular still need
/// `norm_query` precomputed here).
#[derive(Debug, Clone)]
pub struct QuantizedQuery {
    /// Query elements quantized with the segment's `(offset, scale)`,
    /// zero-padded to [`Self::pad_dim`]. The first [`Self::dim`] bytes
    /// are the real quantized values; the rest are zero so the padded
    /// lanes contribute 0 to every kernel (matching the candidate's
    /// zero padding in the pool).
    pub q_data: Vec<u8>,
    /// `Σ q_data[i]`, used in the affine dot-product reconstruction.
    pub sum_q: u32,
    /// `||query||_f32 = sqrt(Σ query[i]²)` computed from the original
    /// f32 query before quantization. Used as the cosine denominator
    /// (paired with `cand_meta.norm_q`, which is the f32 norm of the
    /// dequantized candidate).
    pub norm_query: f32,
    /// Cached `params.scale` so the distance kernel does not need to
    /// dereference [`ScalarQuantParams`] for each candidate.
    pub scale: f32,
    /// Cached `params.offset`.
    pub offset: f32,
    /// True vector dimension (number of meaningful query elements). Used
    /// for the `N·offset²` term of the affine dot-product reconstruction,
    /// which must count real dimensions only, not the zero padding.
    pub dim: usize,
    /// Padded length of `q_data`, equal to [`padded_dim`]`(dim)`. This is
    /// also the length of every candidate int8 slice the kernels receive,
    /// so both operands are the same multiple-of-32 length.
    pub pad_dim: usize,
}

impl QuantizedQuery {
    /// Quantize the f32 query under the segment's params and precompute
    /// the constants used by [`distance_quantized`].
    ///
    /// # Arguments
    ///
    /// * `query` - f32 query vector. Saturated outside the segment's
    ///   trained range (see [`ScalarQuantParams::quantize_value`]).
    /// * `params` - Segment-level quantization params recovered from
    ///   the segment header at search time.
    pub fn prepare(query: &[f32], params: &ScalarQuantParams) -> Self {
        let dim = query.len();
        let mut q_data = params.quantize_slice(query);
        // `sum_q` and `norm_query` are computed over the real `dim`
        // elements only, before padding. (Zero padding would not change
        // `sum_q`, but it *would* corrupt `norm_query` because
        // `dequantize_value(0)` is `offset`, not 0.)
        let sum_q: u32 = q_data.iter().map(|&x| x as u32).sum();
        // Compute the norm of the *dequantized* query so it lives in
        // the same basis as `cand_meta.norm_q` (which is the norm of
        // the dequantized candidate). Using `||query_f32||` here
        // would mix bases and bias the cosine ranking — verified via
        // the AC-2 recall test (Issue #481 Stage 1).
        let norm_query = q_data
            .iter()
            .map(|&q| {
                let dq = params.dequantize_value(q);
                dq * dq
            })
            .sum::<f32>()
            .sqrt();
        // Zero-pad the quantized query to a multiple of `SIMD_BLOCK` so
        // the distance kernels run over whole 32-byte blocks with no
        // scalar tail, matching the candidate padding in
        // `QuantizedVectorPool`.
        let pad_dim = padded_dim(dim);
        q_data.resize(pad_dim, 0);
        Self {
            q_data,
            sum_q,
            norm_query,
            scale: params.scale,
            offset: params.offset,
            dim,
            pad_dim,
        }
    }
}

/// Compute distance from a [`QuantizedQuery`] to one quantized candidate.
///
/// `cand` and `cand_meta` together describe one candidate vector as
/// stored in the segment: `cand` is the int8 payload, `cand_meta`
/// carries the per-vector `sum_q` (cosine cross-term) and `norm_q`
/// (cosine denominator).
///
/// `cand` is either the zero-padded row of a heap pool (`pad_dim` bytes,
/// so the kernels run whole 32-byte blocks) or the unpadded row of a
/// mapped pool (`dim` bytes, Issue #1463). The query is scored over the
/// same number of bytes; its padding is zero, so both lengths give the
/// same integer sums and the same distance.
///
/// # Returns
///
/// Distance in the metric's natural scale, matching the convention of
/// [`DistanceMetric::distance`] (smaller = more similar; DotProduct
/// returns the negated dot product so higher similarity → smaller
/// distance).
///
/// # Panics
///
/// If `cand` is longer than `query.pad_dim`; in debug builds, also if
/// `cand.len()` is neither `query.dim` nor `query.pad_dim`.
pub fn distance_quantized(
    metric: DistanceMetric,
    query: &QuantizedQuery,
    cand: &[u8],
    cand_meta: QuantizedVectorMeta,
) -> f32 {
    debug_assert!(
        cand.len() == query.pad_dim || cand.len() == query.dim,
        "candidate row of {} bytes for a {}-dimensional query",
        cand.len(),
        query.dim
    );
    // Slicing the query to the row's length is what makes the kernel calls
    // below sound: both operands then have `cand.len()` bytes.
    let q = &query.q_data[..cand.len()];
    match metric {
        DistanceMetric::Cosine => {
            // SAFETY: `q.len() == cand.len()` (sliced above).
            let dot_q = unsafe { dot_same_len(q, cand) };
            let approx_dot = approx_dot_product(query, dot_q, cand_meta.sum_q);
            let denom = cand_meta.norm_q * query.norm_query;
            if denom == 0.0 {
                1.0
            } else {
                let cosine = (approx_dot / denom).clamp(-1.0, 1.0);
                1.0 - cosine
            }
        }
        DistanceMetric::Angular => {
            // SAFETY: `q.len() == cand.len()` (sliced above).
            let dot_q = unsafe { dot_same_len(q, cand) };
            let approx_dot = approx_dot_product(query, dot_q, cand_meta.sum_q);
            let denom = cand_meta.norm_q * query.norm_query;
            if denom == 0.0 {
                std::f32::consts::PI
            } else {
                let cosine = (approx_dot / denom).clamp(-1.0, 1.0);
                cosine.acos()
            }
        }
        DistanceMetric::Euclidean => {
            // (a - b)² with affine offset cancellation: the offset term
            // disappears, leaving scale² as the per-pair multiplier.
            // SAFETY: `q.len() == cand.len()` (sliced above).
            let sq_diff = unsafe { sq_diff_same_len(q, cand) };
            (query.scale * query.scale * sq_diff as f32).sqrt()
        }
        DistanceMetric::Manhattan => {
            // |a - b| with the same offset cancellation; the per-pair
            // multiplier is just scale (no square).
            // SAFETY: `q.len() == cand.len()` (sliced above).
            let abs_diff = unsafe { abs_diff_same_len(q, cand) };
            query.scale * abs_diff as f32
        }
        DistanceMetric::DotProduct => {
            // SAFETY: `q.len() == cand.len()` (sliced above).
            let dot_q = unsafe { dot_same_len(q, cand) };
            -approx_dot_product(query, dot_q, cand_meta.sum_q)
        }
    }
}

/// Reconstruct an approximate f32 dot product from quantized data.
///
/// Implements `a·b ≈ N·offset² + scale·offset·(sum_q_a + sum_q_b)
/// + scale²·Σ(aq · bq)`. Only the last term is per-pair (`dot_q`, the
/// int8 dot product); the other three collapse into a constant once
/// `cand_sum_q` is loaded.
#[inline]
fn approx_dot_product(query: &QuantizedQuery, dot_q: i32, cand_sum_q: u32) -> f32 {
    let n = query.dim as f32;
    let off = query.offset;
    let scale = query.scale;
    n * off * off
        + scale * off * (query.sum_q as f32 + cand_sum_q as f32)
        + scale * scale * dot_q as f32
}

/// `Σ a[i] * b[i]` over u8 inputs, accumulating in i32.
///
/// Dispatches at runtime to an AVX2 kernel on x86_64 (when the CPU
/// reports AVX2 support), a NEON kernel on aarch64, and the portable
/// [`dot_u8_to_i32_scalar`] fallback otherwise. All three produce
/// bit-identical results because the arithmetic is integer.
///
/// # Overflow
///
/// `u8 * u8 ≤ 65025` fits in i32. Accumulating up to `i32::MAX / 65025
/// ≈ 33,000` per-pair products fits without overflow, comfortably
/// covering all realistic vector dimensions.
///
/// # Panics
///
/// If `a.len() != b.len()`. The SIMD kernels read both inputs up to
/// `a.len()`, so the check also runs in release builds.
#[inline]
pub fn dot_u8_to_i32(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len(), "int8 kernel inputs differ in length");
    // SAFETY: the lengths were checked above.
    unsafe { dot_same_len(a, b) }
}

/// [`dot_u8_to_i32`] without the length check, for callers that sliced
/// both inputs to the same length themselves ([`distance_quantized`]).
///
/// # Safety
///
/// `a.len() == b.len()`: the SIMD kernels read both inputs up to
/// `a.len()`.
#[inline]
unsafe fn dot_same_len(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if crate::vector::core::sq_int8_avx2::is_avx2_supported() {
        // SAFETY: AVX2 is detected at runtime and `a.len() == b.len()`.
        return unsafe { crate::vector::core::sq_int8_avx2::dot_u8_to_i32_avx2(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::vector::core::sq_int8_neon::is_neon_supported() {
        // SAFETY: NEON is part of the ARMv8 baseline and `a.len() == b.len()`.
        return unsafe { crate::vector::core::sq_int8_neon::dot_u8_to_i32_neon(a, b) };
    }
    dot_u8_to_i32_scalar(a, b)
}

/// Portable `wide`-based reference for [`dot_u8_to_i32`].
///
/// Uses [`wide::i32x8`] for an 8-wide accumulator. Per-element
/// `u8 → i32` zero-extension is folded into the SIMD load. The tail
/// (≤ 7 elements) falls back to scalar. Retained as the fallback for
/// non-AVX2 / non-NEON targets, as the correctness anchor the SIMD
/// kernels are asserted against, and as the shared `len % 32` /
/// `len % 16` tail handler for those kernels.
///
/// Exposed (hidden from docs) so the `vector_search_bench` A/B
/// micro-benchmark can time it against the runtime-dispatched
/// [`dot_u8_to_i32`]; not part of the stable public API.
#[doc(hidden)]
pub fn dot_u8_to_i32_scalar(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = i32x8::ZERO;
    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_b, rem_b) = b.as_chunks::<8>();
    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        let va = i32x8::from([
            ca[0] as i32,
            ca[1] as i32,
            ca[2] as i32,
            ca[3] as i32,
            ca[4] as i32,
            ca[5] as i32,
            ca[6] as i32,
            ca[7] as i32,
        ]);
        let vb = i32x8::from([
            cb[0] as i32,
            cb[1] as i32,
            cb[2] as i32,
            cb[3] as i32,
            cb[4] as i32,
            cb[5] as i32,
            cb[6] as i32,
            cb[7] as i32,
        ]);
        acc += va * vb;
    }
    let mut total: i32 = acc.reduce_add();
    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        total += (*x as i32) * (*y as i32);
    }
    total
}

/// `Σ (a[i] - b[i])²` over u8 inputs, accumulating in i32.
///
/// Dispatches at runtime to AVX2 / NEON / [`sq_diff_u8_to_i32_scalar`]
/// exactly like [`dot_u8_to_i32`]. Per-pair squared diff is at most
/// `255² = 65025`, identical to the dot-product kernel's per-pair
/// bound, so the same overflow analysis applies.
///
/// # Panics
///
/// If `a.len() != b.len()`, like [`dot_u8_to_i32`].
#[inline]
pub fn sq_diff_u8_to_i32(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len(), "int8 kernel inputs differ in length");
    // SAFETY: the lengths were checked above.
    unsafe { sq_diff_same_len(a, b) }
}

/// [`sq_diff_u8_to_i32`] without the length check (see [`dot_same_len`]).
///
/// # Safety
///
/// `a.len() == b.len()`.
#[inline]
unsafe fn sq_diff_same_len(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if crate::vector::core::sq_int8_avx2::is_avx2_supported() {
        // SAFETY: AVX2 is detected at runtime and `a.len() == b.len()`.
        return unsafe { crate::vector::core::sq_int8_avx2::sq_diff_u8_to_i32_avx2(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::vector::core::sq_int8_neon::is_neon_supported() {
        // SAFETY: NEON is part of the ARMv8 baseline and `a.len() == b.len()`.
        return unsafe { crate::vector::core::sq_int8_neon::sq_diff_u8_to_i32_neon(a, b) };
    }
    sq_diff_u8_to_i32_scalar(a, b)
}

/// Portable `wide`-based reference for [`sq_diff_u8_to_i32`].
///
/// The subtraction widens to i32 first so the result remains correct
/// even when `a[i] < b[i]` (would underflow in u8). See
/// [`dot_u8_to_i32_scalar`] for the fallback / tail-handler role.
#[doc(hidden)]
pub fn sq_diff_u8_to_i32_scalar(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = i32x8::ZERO;
    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_b, rem_b) = b.as_chunks::<8>();
    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        let va = i32x8::from([
            ca[0] as i32,
            ca[1] as i32,
            ca[2] as i32,
            ca[3] as i32,
            ca[4] as i32,
            ca[5] as i32,
            ca[6] as i32,
            ca[7] as i32,
        ]);
        let vb = i32x8::from([
            cb[0] as i32,
            cb[1] as i32,
            cb[2] as i32,
            cb[3] as i32,
            cb[4] as i32,
            cb[5] as i32,
            cb[6] as i32,
            cb[7] as i32,
        ]);
        let diff = va - vb;
        acc += diff * diff;
    }
    let mut total: i32 = acc.reduce_add();
    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        let d = (*x as i32) - (*y as i32);
        total += d * d;
    }
    total
}

/// `Σ |a[i] - b[i]|` over u8 inputs, accumulating in i32.
///
/// Dispatches at runtime to AVX2 / NEON / [`abs_diff_u8_to_i32_scalar`]
/// exactly like [`dot_u8_to_i32`]. Per-pair magnitude is at most
/// `255`, so accumulation safety is the same as the other kernels.
///
/// # Panics
///
/// If `a.len() != b.len()`, like [`dot_u8_to_i32`].
#[inline]
pub fn abs_diff_u8_to_i32(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len(), "int8 kernel inputs differ in length");
    // SAFETY: the lengths were checked above.
    unsafe { abs_diff_same_len(a, b) }
}

/// [`abs_diff_u8_to_i32`] without the length check (see [`dot_same_len`]).
///
/// # Safety
///
/// `a.len() == b.len()`.
#[inline]
unsafe fn abs_diff_same_len(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if crate::vector::core::sq_int8_avx2::is_avx2_supported() {
        // SAFETY: AVX2 is detected at runtime and `a.len() == b.len()`.
        return unsafe { crate::vector::core::sq_int8_avx2::abs_diff_u8_to_i32_avx2(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::vector::core::sq_int8_neon::is_neon_supported() {
        // SAFETY: NEON is part of the ARMv8 baseline and `a.len() == b.len()`.
        return unsafe { crate::vector::core::sq_int8_neon::abs_diff_u8_to_i32_neon(a, b) };
    }
    abs_diff_u8_to_i32_scalar(a, b)
}

/// Portable `wide`-based reference for [`abs_diff_u8_to_i32`].
///
/// Uses widened `i32` subtraction + [`i32x8::abs`] for the absolute
/// value. See [`dot_u8_to_i32_scalar`] for the fallback / tail-handler
/// role.
#[doc(hidden)]
pub fn abs_diff_u8_to_i32_scalar(a: &[u8], b: &[u8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = i32x8::ZERO;
    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_b, rem_b) = b.as_chunks::<8>();
    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        let va = i32x8::from([
            ca[0] as i32,
            ca[1] as i32,
            ca[2] as i32,
            ca[3] as i32,
            ca[4] as i32,
            ca[5] as i32,
            ca[6] as i32,
            ca[7] as i32,
        ]);
        let vb = i32x8::from([
            cb[0] as i32,
            cb[1] as i32,
            cb[2] as i32,
            cb[3] as i32,
            cb[4] as i32,
            cb[5] as i32,
            cb[6] as i32,
            cb[7] as i32,
        ]);
        let diff = va - vb;
        acc += diff.abs();
    }
    let mut total: i32 = acc.reduce_add();
    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        total += ((*x as i32) - (*y as i32)).abs();
    }
    total
}

// ============================================================================
// Stage 3 — Product Quantization (PQ) ADC kernel
// ============================================================================

/// Per-query state for the Stage 3 PQ ADC distance.
///
/// Built once per HNSW search invocation from the f32 query and the
/// segment's PQ codebook. Stores an `M × K` look-up table where each
/// entry `lut[m * K + k]` is the squared L2 distance from sub-vector
/// `m` of the query to centroid `k` of sub-vector `m`'s codebook.
///
/// The hot per-candidate distance call collapses to `M` table lookups
/// and `M - 1` adds — the asymmetric distance computation (ADC) trick
/// that gives PQ its per-call latency advantage.
#[derive(Debug, Clone)]
pub struct PqQuery {
    /// Per-query L2² look-up table, row-major `[m][k]` of length
    /// `params.codebook_len() / sub_dim = m * k`.
    pub lut: Vec<f32>,
    /// `||query||² = Σ_i query[i]²`. Cached so the Cosine path can
    /// compute the cosine numerator without re-scanning `query`.
    pub query_norm_sq: f32,
    /// PQ parameters this LUT was built against (so the kernel can
    /// recover `m` and `k` without an extra arg).
    pub params: PqParams,
}

impl PqQuery {
    /// Build the per-query LUT from a raw f32 query and the segment's
    /// codebook.
    ///
    /// `codebook` must have length `params.codebook_len()` and be laid
    /// out row-major `[m][k][sub_dim]` as produced by
    /// [`crate::vector::core::quantization::pq_train_codebook`].
    pub fn prepare(query: &[f32], params: PqParams, codebook: &[f32]) -> Self {
        debug_assert_eq!(query.len(), params.original_dim());
        debug_assert_eq!(codebook.len(), params.codebook_len());

        let m = params.m as usize;
        let k = params.k as usize;
        let sub_dim = params.sub_dim as usize;
        let mut lut = vec![0.0_f32; m * k];

        for sub in 0..m {
            let q_sub = &query[sub * sub_dim..(sub + 1) * sub_dim];
            let cb_base = sub * k * sub_dim;
            for ki in 0..k {
                let c = &codebook[cb_base + ki * sub_dim..cb_base + (ki + 1) * sub_dim];
                let mut acc = 0.0_f32;
                for d in 0..sub_dim {
                    let diff = q_sub[d] - c[d];
                    acc += diff * diff;
                }
                lut[sub * k + ki] = acc;
            }
        }

        let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();

        Self {
            lut,
            query_norm_sq,
            params,
        }
    }
}

/// Compute distance from a [`PqQuery`] to one PQ-encoded candidate.
///
/// `codes` must have length `query.params.m`.
///
/// Returns the same metric scale as
/// [`crate::vector::core::distance::DistanceMetric::distance`]
/// (smaller = more similar). Only Cosine and Euclidean are supported
/// for PQ; HNSW field validation rejects other metrics on field definition
/// and warns when reopening a persisted schema. The infinity fallback
/// protects callers that bypass those checks.
pub fn distance_pq_adc(metric: DistanceMetric, query: &PqQuery, codes: &[u8]) -> f32 {
    debug_assert_eq!(codes.len(), query.params.m as usize);
    let m = query.params.m as usize;
    let k = query.params.k as usize;

    // `Σ_m lut[m][codes[m]]` — squared L2 distance between the query
    // and the decoded candidate. The hot loop is bounded by `M`
    // (typically 8-32) so SIMD is unnecessary at this layer; the win
    // vs the f32 path is the `M` lookups vs `dim` multiplications.
    let mut l2_sq = 0.0_f32;
    for (sub, &code) in codes.iter().enumerate().take(m) {
        l2_sq += query.lut[sub * k + code as usize];
    }

    match metric {
        DistanceMetric::Euclidean => l2_sq.max(0.0).sqrt(),
        DistanceMetric::Cosine => {
            // For unit-norm query and corpus (the laurus convention for
            // Cosine via [`crate::vector::core::distance::DistanceMetric::prepare_query`]),
            // `||q - v||² = 2 - 2 q·v`, so `cos_dist = 1 - q·v ≈ l2_sq /
            // 2`. The clamp guards against numerical drift outside
            // `[0, 2]`.
            (l2_sq * 0.5).clamp(0.0, 2.0)
        }
        // Keep a defensive sentinel for callers that bypass field
        // validation. A field persisted before Issue #1458 opens with a
        // warning and still reaches this arm.
        _ => f32::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::vector::Vector;

    /// Reference scalar implementations used to validate the SIMD
    /// kernels regardless of the underlying instruction set.
    fn scalar_dot(a: &[u8], b: &[u8]) -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i32) * (y as i32))
            .sum()
    }
    fn scalar_sq_diff(a: &[u8], b: &[u8]) -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = (x as i32) - (y as i32);
                d * d
            })
            .sum()
    }
    fn scalar_abs_diff(a: &[u8], b: &[u8]) -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| ((x as i32) - (y as i32)).abs())
            .sum()
    }

    fn pseudo_random_u8(seed: u32, len: usize) -> Vec<u8> {
        // Tiny LCG, deterministic across runs. Good enough for the
        // SIMD vs scalar consistency check.
        let mut state = seed.wrapping_mul(0x9E37_79B9).wrapping_add(0xDEAD_BEEF);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    fn pseudo_random_f32(seed: u32, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        let bytes = pseudo_random_u8(seed, len);
        let range = hi - lo;
        bytes
            .into_iter()
            .map(|b| lo + (b as f32 / 255.0) * range)
            .collect()
    }

    #[test]
    fn dot_simd_matches_scalar_for_various_lengths() {
        for &dim in &[
            1, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 96, 100, 128, 384, 768,
        ] {
            let a = pseudo_random_u8(1, dim);
            let b = pseudo_random_u8(2, dim);
            assert_eq!(
                dot_u8_to_i32(&a, &b),
                scalar_dot(&a, &b),
                "dim = {dim}: SIMD dot disagrees with scalar"
            );
        }
    }

    #[test]
    fn sq_diff_simd_matches_scalar_for_various_lengths() {
        for &dim in &[
            1, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 96, 100, 128, 384, 768,
        ] {
            let a = pseudo_random_u8(3, dim);
            let b = pseudo_random_u8(4, dim);
            assert_eq!(
                sq_diff_u8_to_i32(&a, &b),
                scalar_sq_diff(&a, &b),
                "dim = {dim}: SIMD sq_diff disagrees with scalar"
            );
        }
    }

    #[test]
    fn abs_diff_simd_matches_scalar_for_various_lengths() {
        for &dim in &[
            1, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 96, 100, 128, 384, 768,
        ] {
            let a = pseudo_random_u8(5, dim);
            let b = pseudo_random_u8(6, dim);
            assert_eq!(
                abs_diff_u8_to_i32(&a, &b),
                scalar_abs_diff(&a, &b),
                "dim = {dim}: SIMD abs_diff disagrees with scalar"
            );
        }
    }

    /// The portable `wide` fallbacks must match the plain scalar
    /// reference for every length, independent of which kernel the
    /// dispatcher selects at runtime. This keeps the non-AVX2 / non-NEON
    /// path covered on an AVX2 host.
    #[test]
    fn scalar_fallbacks_match_reference_for_various_lengths() {
        for &dim in &[
            1, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 96, 100, 128, 384, 768,
        ] {
            let a = pseudo_random_u8(7, dim);
            let b = pseudo_random_u8(8, dim);
            assert_eq!(
                dot_u8_to_i32_scalar(&a, &b),
                scalar_dot(&a, &b),
                "dot dim={dim}"
            );
            assert_eq!(
                sq_diff_u8_to_i32_scalar(&a, &b),
                scalar_sq_diff(&a, &b),
                "sq_diff dim={dim}"
            );
            assert_eq!(
                abs_diff_u8_to_i32_scalar(&a, &b),
                scalar_abs_diff(&a, &b),
                "abs_diff dim={dim}"
            );
        }
    }

    /// Directly exercise the AVX2 kernels (not just the dispatcher) so
    /// the SIMD path is verified even if the dispatcher's runtime probe
    /// ever changes. Skipped when the host lacks AVX2. Lengths span the
    /// 32-byte block boundary (`31/32/33/96/100`) to stress the scalar
    /// tail delegation.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernels_match_scalar_when_supported() {
        use crate::vector::core::sq_int8_avx2::{
            abs_diff_u8_to_i32_avx2, dot_u8_to_i32_avx2, is_avx2_supported, sq_diff_u8_to_i32_avx2,
        };
        if !is_avx2_supported() {
            return;
        }
        for &dim in &[
            1, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 96, 100, 128, 384, 768, 4096,
        ] {
            let a = pseudo_random_u8(9, dim);
            let b = pseudo_random_u8(10, dim);
            // SAFETY: guarded by `is_avx2_supported()` above.
            unsafe {
                assert_eq!(
                    dot_u8_to_i32_avx2(&a, &b),
                    scalar_dot(&a, &b),
                    "dot dim={dim}"
                );
                assert_eq!(
                    sq_diff_u8_to_i32_avx2(&a, &b),
                    scalar_sq_diff(&a, &b),
                    "sq_diff dim={dim}"
                );
                assert_eq!(
                    abs_diff_u8_to_i32_avx2(&a, &b),
                    scalar_abs_diff(&a, &b),
                    "abs_diff dim={dim}"
                );
            }
        }
        // All-255 extreme at a realistic dim: exercises the i32 / u64
        // accumulator bounds documented on the kernels.
        let a = vec![255u8; 4096];
        let b = vec![0u8; 4096];
        // SAFETY: guarded by `is_avx2_supported()` above.
        unsafe {
            assert_eq!(dot_u8_to_i32_avx2(&a, &a), 255 * 255 * 4096);
            assert_eq!(sq_diff_u8_to_i32_avx2(&a, &b), 255 * 255 * 4096);
            assert_eq!(abs_diff_u8_to_i32_avx2(&a, &b), 255 * 4096);
        }
    }

    #[test]
    fn dot_extremes_do_not_overflow_at_realistic_dim() {
        // Worst case: every element 255. dim = 4096 keeps Σ 255*255
        // ≈ 266M, well below i32::MAX (~2.1B).
        let dim = 4096;
        let a = vec![255u8; dim];
        let b = vec![255u8; dim];
        let expected = 255i32 * 255 * dim as i32;
        assert_eq!(dot_u8_to_i32(&a, &b), expected);
    }

    /// End-to-end correctness: quantized cosine distance must agree
    /// with the f32 cosine distance to within a small per-pair error
    /// bound. The bound scales with `params.scale`, so we use vectors
    /// in a fixed `[-1, 1]` range and assert `|delta| < 0.02` on the
    /// returned 1 - cos(θ) scalar.
    #[test]
    fn cosine_quantized_matches_f32_within_tolerance() {
        let dim = 128;
        let a_f32 = pseudo_random_f32(11, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(12, dim, -1.0, 1.0);
        let training = vec![Vector::new(a_f32.clone()), Vector::new(b_f32.clone())];
        let params = ScalarQuantParams::train(&training).unwrap();

        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);

        let prepared = QuantizedQuery::prepare(&b_f32, &params);
        let approx_cosine_dist =
            distance_quantized(DistanceMetric::Cosine, &prepared, &q_a, meta_a);

        let exact_cosine_dist = DistanceMetric::Cosine.distance(&a_f32, &b_f32).unwrap();
        let delta = (approx_cosine_dist - exact_cosine_dist).abs();
        assert!(
            delta < 0.02,
            "cosine: quantized = {approx_cosine_dist}, f32 = {exact_cosine_dist}, |Δ| = {delta}"
        );
    }

    /// Padding invariant end-to-end: for a dim that is *not* a multiple
    /// of 32, the query is zero-padded by `prepare` and the candidate is
    /// zero-padded exactly as `QuantizedVectorPool` does. The result must
    /// still match the f32 cosine distance, proving the padded lanes
    /// contribute 0 and `sum_q` / `norm_query` / `N` are unaffected.
    #[test]
    fn cosine_matches_f32_for_non_block_dim_with_padding() {
        let dim = 100; // padded_dim(100) == 128
        let a_f32 = pseudo_random_f32(31, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(32, dim, -1.0, 1.0);
        let params =
            ScalarQuantParams::train(&[Vector::new(a_f32.clone()), Vector::new(b_f32.clone())])
                .unwrap();

        // `meta_a` is derived from the UNPADDED candidate (real dim), as
        // the writer does; then the candidate is zero-padded to pad_dim,
        // as the in-memory pool does.
        let mut q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);
        q_a.resize(padded_dim(dim), 0);

        let prepared = QuantizedQuery::prepare(&b_f32, &params);
        assert_eq!(prepared.pad_dim, 128);
        assert_eq!(q_a.len(), prepared.pad_dim);

        let approx = distance_quantized(DistanceMetric::Cosine, &prepared, &q_a, meta_a);
        let exact = DistanceMetric::Cosine.distance(&a_f32, &b_f32).unwrap();
        assert!(
            (approx - exact).abs() < 0.02,
            "padded cosine: quantized = {approx}, f32 = {exact}"
        );
    }

    #[test]
    fn euclidean_quantized_matches_f32_within_tolerance() {
        let dim = 128;
        let a_f32 = pseudo_random_f32(21, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(22, dim, -1.0, 1.0);
        let params =
            ScalarQuantParams::train(&[Vector::new(a_f32.clone()), Vector::new(b_f32.clone())])
                .unwrap();

        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);
        let prepared = QuantizedQuery::prepare(&b_f32, &params);

        let approx_dist = distance_quantized(DistanceMetric::Euclidean, &prepared, &q_a, meta_a);
        let exact_dist = DistanceMetric::Euclidean.distance(&a_f32, &b_f32).unwrap();
        let rel_err = (approx_dist - exact_dist).abs() / exact_dist.max(1e-6);
        assert!(
            rel_err < 0.05,
            "euclidean: quantized = {approx_dist}, f32 = {exact_dist}, rel_err = {rel_err}"
        );
    }

    #[test]
    fn manhattan_quantized_matches_f32_within_tolerance() {
        let dim = 128;
        let a_f32 = pseudo_random_f32(31, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(32, dim, -1.0, 1.0);
        let params =
            ScalarQuantParams::train(&[Vector::new(a_f32.clone()), Vector::new(b_f32.clone())])
                .unwrap();

        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);
        let prepared = QuantizedQuery::prepare(&b_f32, &params);

        let approx_dist = distance_quantized(DistanceMetric::Manhattan, &prepared, &q_a, meta_a);
        let exact_dist = DistanceMetric::Manhattan.distance(&a_f32, &b_f32).unwrap();
        let rel_err = (approx_dist - exact_dist).abs() / exact_dist.max(1e-6);
        assert!(
            rel_err < 0.02,
            "manhattan: quantized = {approx_dist}, f32 = {exact_dist}, rel_err = {rel_err}"
        );
    }

    #[test]
    fn dot_product_quantized_matches_f32_within_tolerance() {
        let dim = 128;
        let a_f32 = pseudo_random_f32(41, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(42, dim, -1.0, 1.0);
        let params =
            ScalarQuantParams::train(&[Vector::new(a_f32.clone()), Vector::new(b_f32.clone())])
                .unwrap();

        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);
        let prepared = QuantizedQuery::prepare(&b_f32, &params);

        let approx_dist = distance_quantized(DistanceMetric::DotProduct, &prepared, &q_a, meta_a);
        let exact_dist = DistanceMetric::DotProduct.distance(&a_f32, &b_f32).unwrap();
        let abs_err = (approx_dist - exact_dist).abs();
        assert!(
            abs_err < 0.5,
            "dot_product: quantized = {approx_dist}, f32 = {exact_dist}, |Δ| = {abs_err}"
        );
    }

    #[test]
    fn angular_quantized_matches_f32_within_tolerance() {
        let dim = 128;
        let a_f32 = pseudo_random_f32(51, dim, -1.0, 1.0);
        let b_f32 = pseudo_random_f32(52, dim, -1.0, 1.0);
        let params =
            ScalarQuantParams::train(&[Vector::new(a_f32.clone()), Vector::new(b_f32.clone())])
                .unwrap();

        let q_a = params.quantize_slice(&a_f32);
        let meta_a = QuantizedVectorMeta::from_quantized(&q_a, &params);
        let prepared = QuantizedQuery::prepare(&b_f32, &params);

        let approx_dist = distance_quantized(DistanceMetric::Angular, &prepared, &q_a, meta_a);
        let exact_dist = DistanceMetric::Angular.distance(&a_f32, &b_f32).unwrap();
        let abs_err = (approx_dist - exact_dist).abs();
        assert!(
            abs_err < 0.05,
            "angular: quantized = {approx_dist}, f32 = {exact_dist}, |Δ| = {abs_err}"
        );
    }

    #[test]
    fn cosine_handles_zero_norm_query() {
        // The cosine path must short-circuit to the max distance
        // (1.0) when the prepared query has zero norm in the
        // dequantized basis. Use offset = 0 / scale = 1 so the f32
        // zero query also dequantizes to exactly zero -- this is the
        // only configuration in which the `denom == 0.0` guard is
        // exercised in production (a query with truly empty signal).
        let dim = 16;
        let params = ScalarQuantParams {
            offset: 0.0,
            scale: 1.0,
        };
        let mut a: Vec<u8> = (0..dim as u8).collect();
        let meta_a = QuantizedVectorMeta::from_quantized(&a, &params);
        // Pad the candidate to pad_dim as the in-memory pool does.
        a.resize(padded_dim(dim), 0);
        let zero_query = vec![0.0_f32; dim];
        let prepared = QuantizedQuery::prepare(&zero_query, &params);

        // The zero query rounds to all-zero ints in this basis, so
        // norm_query is exactly 0.0 and the early-return path fires.
        assert_eq!(prepared.norm_query, 0.0);

        let dist = distance_quantized(DistanceMetric::Cosine, &prepared, &a, meta_a);
        assert_eq!(
            dist, 1.0,
            "zero-norm query should yield max cosine distance, got {dist}"
        );
    }

    #[test]
    fn prepared_query_caches_segment_params() {
        let dim = 8;
        let query = vec![0.5_f32; dim];
        let params = ScalarQuantParams {
            offset: -1.0,
            scale: 2.0 / 255.0,
        };
        let prepared = QuantizedQuery::prepare(&query, &params);
        assert_eq!(prepared.dim, dim);
        assert_eq!(prepared.offset, params.offset);
        assert_eq!(prepared.scale, params.scale);
        // q_data is zero-padded to pad_dim (dim 8 -> 32).
        assert_eq!(prepared.pad_dim, padded_dim(dim));
        assert_eq!(prepared.q_data.len(), prepared.pad_dim);
        // sum_q counts the real dim only; the zero padding adds nothing,
        // so summing the whole padded buffer yields the same value.
        let expected_sum: u32 = prepared.q_data.iter().map(|&x| x as u32).sum();
        assert_eq!(prepared.sum_q, expected_sum);
    }

    /// Issue #1463: a mapped pool hands the kernel the stored row as is
    /// (`dim` bytes), a heap pool the zero-padded row (`pad_dim` bytes).
    /// Both must give bit-identical distances for every metric, including
    /// dimensions where the SIMD kernels run a tail.
    #[test]
    fn unpadded_rows_score_exactly_like_padded_rows() {
        let metrics = [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::DotProduct,
            DistanceMetric::Angular,
        ];
        for &dim in &[1usize, 31, 32, 33, 100, 384] {
            let a = pseudo_random_f32(61 + dim as u32, dim, -1.0, 1.0);
            let b = pseudo_random_f32(62 + dim as u32, dim, -0.7, 1.3);
            let params =
                ScalarQuantParams::train(&[Vector::new(a.clone()), Vector::new(b.clone())])
                    .unwrap();
            let row = params.quantize_slice(&a);
            let meta = QuantizedVectorMeta::from_quantized(&row, &params);
            let mut padded = row.clone();
            padded.resize(padded_dim(dim), 0);
            let prepared = QuantizedQuery::prepare(&b, &params);
            for metric in metrics {
                let unpadded_d = distance_quantized(metric, &prepared, &row, meta);
                let padded_d = distance_quantized(metric, &prepared, &padded, meta);
                assert_eq!(
                    unpadded_d.to_bits(),
                    padded_d.to_bits(),
                    "dim {dim} {metric:?}: {unpadded_d} vs {padded_d}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "candidate row of 31 bytes")]
    fn distance_quantized_rejects_a_row_of_another_length() {
        let params = ScalarQuantParams {
            offset: 0.0,
            scale: 1.0,
        };
        let prepared = QuantizedQuery::prepare(&[0.5; 30], &params);
        let meta = QuantizedVectorMeta {
            sum_q: 0,
            norm_q: 0.0,
        };
        distance_quantized(DistanceMetric::Euclidean, &prepared, &[0; 31], meta);
    }

    #[test]
    #[should_panic(expected = "differ in length")]
    fn int8_kernels_reject_inputs_of_different_lengths() {
        dot_u8_to_i32(&[1; 64], &[1; 33]);
    }

    // ------------------------------------------------------------------
    // Stage 3 PQ ADC kernel tests
    // ------------------------------------------------------------------

    use crate::vector::core::quantization::{pq_decode, pq_encode, pq_train_codebook};

    /// Build a small reproducible PQ codebook (dim 4, M=2, sub_dim=2)
    /// from two well-separated clusters so the encoder picks deterministic
    /// codes and the ADC kernel has unambiguous LUT entries to look up.
    fn small_pq_setup() -> (PqParams, Vec<f32>, Vec<Vector>) {
        let dim = 4;
        let m = 2;
        let params = PqParams::from_dim_and_m(dim, m).unwrap();
        let training: Vec<Vector> = vec![
            Vector::new(vec![5.0, 5.0, 10.0, 10.0]),
            Vector::new(vec![-5.0, -5.0, -10.0, -10.0]),
            Vector::new(vec![5.1, 5.1, 10.1, 10.1]),
            Vector::new(vec![-4.9, -4.9, -9.9, -9.9]),
        ];
        let codebook = pq_train_codebook(dim, params, &training).unwrap();
        (params, codebook, training)
    }

    #[test]
    fn pq_lut_has_expected_dimensions() {
        let (params, codebook, _) = small_pq_setup();
        let query = vec![5.0_f32, 5.0, 10.0, 10.0];
        let pq_query = PqQuery::prepare(&query, params, &codebook);
        let m = params.m as usize;
        let k = params.k as usize;
        assert_eq!(pq_query.lut.len(), m * k);
        // The query is a corpus point, so for at least one centroid in
        // each sub-vector the LUT entry should be very small.
        for sub in 0..m {
            let row = &pq_query.lut[sub * k..(sub + 1) * k];
            let min = row.iter().cloned().fold(f32::INFINITY, f32::min);
            assert!(
                min < 1.0,
                "sub-vector {sub}: nearest centroid should have L2² near 0, got {min}"
            );
        }
    }

    #[test]
    fn pq_adc_euclidean_matches_decoded_l2() {
        let (params, codebook, training) = small_pq_setup();
        // Encode the training set so we know each candidate's decoded form.
        let encoded: Vec<Vec<u8>> = training
            .iter()
            .map(|v| pq_encode(&v.data, params, &codebook))
            .collect();
        // Pick an arbitrary query (not in the training set).
        let query = vec![5.0_f32, 5.0, 9.5, 9.5];
        let pq_query = PqQuery::prepare(&query, params, &codebook);
        for codes in &encoded {
            let approx = distance_pq_adc(DistanceMetric::Euclidean, &pq_query, codes);
            // Reference: decode and compute exact L2 against the
            // decoded vector. ADC must match this within FP noise.
            let decoded = pq_decode(codes, params, &codebook);
            let mut ref_l2_sq = 0.0_f32;
            for (q, d) in query.iter().zip(decoded.iter()) {
                let diff = q - d;
                ref_l2_sq += diff * diff;
            }
            let ref_l2 = ref_l2_sq.sqrt();
            assert!(
                (approx - ref_l2).abs() < 1e-3,
                "ADC Euclidean = {approx}, reference = {ref_l2}, codes = {codes:?}"
            );
        }
    }

    #[test]
    fn pq_adc_cosine_matches_l2_over_two_for_unit_norm() {
        // L2-normalise both query and corpus so cos_dist = L2² / 2
        // becomes exact (ignoring the PQ approximation).
        let dim = 4;
        let m = 2;
        let params = PqParams::from_dim_and_m(dim, m).unwrap();

        fn unit_norm(v: &mut [f32]) {
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 0.0 {
                for x in v.iter_mut() {
                    *x /= n;
                }
            }
        }

        let mut training: Vec<Vector> = vec![
            Vector::new({
                let mut v = vec![1.0_f32, 0.5, 0.2, 0.1];
                unit_norm(&mut v);
                v
            }),
            Vector::new({
                let mut v = vec![-1.0_f32, -0.5, -0.2, -0.1];
                unit_norm(&mut v);
                v
            }),
            Vector::new({
                let mut v = vec![0.9_f32, 0.45, 0.2, 0.1];
                unit_norm(&mut v);
                v
            }),
            Vector::new({
                let mut v = vec![-0.9_f32, -0.45, -0.2, -0.1];
                unit_norm(&mut v);
                v
            }),
        ];
        let codebook = pq_train_codebook(dim, params, &training).unwrap();

        // Query in unit-norm basis matching the training distribution.
        let mut query = vec![0.95_f32, 0.48, 0.2, 0.1];
        unit_norm(&mut query);

        let pq_query = PqQuery::prepare(&query, params, &codebook);
        for v in training.iter_mut() {
            let codes = pq_encode(&v.data, params, &codebook);
            let cos = distance_pq_adc(DistanceMetric::Cosine, &pq_query, &codes);
            let euc = distance_pq_adc(DistanceMetric::Euclidean, &pq_query, &codes);
            // cos = euc² / 2 by construction; allow FP slack.
            let expected = (euc * euc) * 0.5;
            assert!(
                (cos - expected).abs() < 1e-3,
                "cosine {cos} vs euc²/2 {expected} (euc = {euc})"
            );
            assert!((0.0..=2.0).contains(&cos), "cosine {cos} outside [0, 2]");
        }
    }

    #[test]
    fn pq_adc_unsupported_metric_returns_infinity() {
        let (params, codebook, training) = small_pq_setup();
        let codes = pq_encode(&training[0].data, params, &codebook);
        let query = vec![1.0_f32, 1.0, 1.0, 1.0];
        let pq_query = PqQuery::prepare(&query, params, &codebook);
        for metric in [
            DistanceMetric::Manhattan,
            DistanceMetric::DotProduct,
            DistanceMetric::Angular,
        ] {
            let d = distance_pq_adc(metric, &pq_query, &codes);
            assert!(
                d.is_infinite(),
                "metric {metric:?} should yield +inf, got {d}"
            );
        }
    }
}
