//! Late-interaction (MaxSim) scoring (Issue #1177).
//!
//! ColBERT-style late interaction scores a document by matching every query
//! token vector against its best document token vector and summing those
//! best similarities:
//!
//! ```text
//! score(q, d) = Σ_i max_j (q_i · d_j)
//! ```
//!
//! The similarity is a plain dot product. For cosine fields both sides are
//! L2-normalized beforehand, so the dot product is the cosine similarity.
//! The sum is taken on the raw similarities: a non-affine transform applied
//! per pair before summing would change the ranking.

use std::sync::Arc;

#[cfg(test)]
use crate::vector::core::distance::dot_product;
use crate::vector::core::distance::dot_product_inline;
use crate::vector::core::multi_vector::{BinaryParams, BinaryQuery, MultiVectorRows};

/// MaxSim of one document.
///
/// Vectors of 64, 96 and 128 dimensions, the widths of common ColBERT-style
/// models, are scored by a copy of the loop with the dimension fixed at
/// compile time (Issue #1442). Every other dimension scores four document
/// vectors per pass over a query vector ([`max_sim_block4`], Issue #1444),
/// which cuts 31 to 36% of the instructions per pair a plain one-at-a-time
/// run-time loop takes. Every path sums a pair as [`dot_product`](crate::vector::core::distance::dot_product) does, so
/// the score does not depend on the loop that computed it.
///
/// # Arguments
///
/// * `query` - Query token vectors, row-major (`query.len()` is a multiple
///   of `dimension`).
/// * `document` - Document token vectors, row-major, at least one vector.
/// * `dimension` - Length of every vector.
///
/// # Returns
///
/// `Σ_i max_j q_i · d_j` over the query rows `q_i` and document rows `d_j`.
pub(crate) fn max_sim(query: &[f32], document: &[f32], dimension: usize) -> f32 {
    debug_assert!(dimension > 0 && query.len().is_multiple_of(dimension));
    debug_assert!(!document.is_empty() && document.len().is_multiple_of(dimension));
    match dimension {
        64 => max_sim_fixed::<64>(query, document),
        96 => max_sim_fixed::<96>(query, document),
        128 => max_sim_fixed::<128>(query, document),
        _ => max_sim_block4(query, document, dimension),
    }
}

/// [`max_sim`] with the dimension given at run time, one document vector
/// per pass; any dimension. Superseded by [`max_sim_block4`] in production
/// (Issue #1444); kept under `#[cfg(test)]` as the bit-identical reference
/// the faster loops are checked against.
#[cfg(test)]
pub(crate) fn max_sim_dyn(query: &[f32], document: &[f32], dimension: usize) -> f32 {
    query
        .chunks_exact(dimension)
        .map(|q| {
            document
                .chunks_exact(dimension)
                .map(|d| dot_product(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// [`max_sim`] with the dimension `D` fixed at compile time, so that the
/// loop over a pair's 8-lane chunks unrolls.
fn max_sim_fixed<const D: usize>(query: &[f32], document: &[f32]) -> f32 {
    #[cfg(test)]
    dispatch_calls::record(dispatch_calls::Loop::Fixed);
    let query = query.as_chunks::<D>().0;
    let document = document.as_chunks::<D>().0;
    query
        .iter()
        .map(|q| {
            document
                .iter()
                .map(|d| dot_product_inline(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// [`max_sim`] with the dimension given at run time, scoring four document
/// vectors per pass over a query vector (Issue #1444). At 32, 100, 256,
/// 384, 768 and 1,024 dimensions this retires 31 to 36% fewer instructions
/// per pair than [`max_sim_dyn`], amortizing the loop setup and the query
/// row's loads over four document rows instead of one. Each pair sums in
/// the same order as [`dot_product`](crate::vector::core::distance::dot_product), so scores stay bit-identical.
fn max_sim_block4(query: &[f32], document: &[f32], dimension: usize) -> f32 {
    #[cfg(test)]
    dispatch_calls::record(dispatch_calls::Loop::Block4);
    let block = dimension * 4;
    let full = (document.len() / block) * block;
    let (quads, rem) = document.split_at(full);
    query
        .chunks_exact(dimension)
        .map(|q| {
            let mut best = f32::NEG_INFINITY;
            for quad in quads.chunks_exact(block) {
                let (d0, rest) = quad.split_at(dimension);
                let (d1, rest) = rest.split_at(dimension);
                let (d2, d3) = rest.split_at(dimension);
                let (s0, s1, s2, s3) = dot4(q, d0, d1, d2, d3);
                best = best.max(s0).max(s1).max(s2).max(s3);
            }
            for d in rem.chunks_exact(dimension) {
                best = best.max(dot_product_inline(q, d));
            }
            best
        })
        .sum()
}

/// Four dot products of `a` against `b0..b3`, one pass over `a`'s 8-lane
/// chunks. Sums in the same order as [`dot_product`](crate::vector::core::distance::dot_product).
#[inline(always)]
fn dot4(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> (f32, f32, f32, f32) {
    use wide::f32x8;

    let mut sum0 = f32x8::ZERO;
    let mut sum1 = f32x8::ZERO;
    let mut sum2 = f32x8::ZERO;
    let mut sum3 = f32x8::ZERO;
    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_0, rem_0) = b0.as_chunks::<8>();
    let (chunks_1, rem_1) = b1.as_chunks::<8>();
    let (chunks_2, rem_2) = b2.as_chunks::<8>();
    let (chunks_3, rem_3) = b3.as_chunks::<8>();
    for i in 0..chunks_a.len() {
        let va = f32x8::from(chunks_a[i]);
        sum0 += va * f32x8::from(chunks_0[i]);
        sum1 += va * f32x8::from(chunks_1[i]);
        sum2 += va * f32x8::from(chunks_2[i]);
        sum3 += va * f32x8::from(chunks_3[i]);
    }
    let mut r0 = sum0.reduce_add();
    let mut r1 = sum1.reduce_add();
    let mut r2 = sum2.reduce_add();
    let mut r3 = sum3.reduce_add();
    for j in 0..rem_a.len() {
        r0 += rem_a[j] * rem_0[j];
        r1 += rem_a[j] * rem_1[j];
        r2 += rem_a[j] * rem_2[j];
        r3 += rem_a[j] * rem_3[j];
    }
    (r0, r1, r2, r3)
}

/// Calls into [`max_sim_fixed`] and [`max_sim_block4`], so that tests can
/// see which one `max_sim` dispatched to — the scores, the same through
/// either loop, cannot show it.
#[cfg(test)]
pub(crate) mod dispatch_calls {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Which run-time loop `max_sim` dispatched to.
    #[derive(Debug, Clone, Copy)]
    pub(crate) enum Loop {
        /// A dimension with a fixed-width copy (64, 96 or 128).
        Fixed,
        /// Any other dimension, [`super::max_sim_block4`].
        Block4,
    }

    static FIXED_ANY_THREAD: AtomicUsize = AtomicUsize::new(0);
    static BLOCK4_ANY_THREAD: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        static FIXED_THIS_THREAD: Cell<usize> = const { Cell::new(0) };
        static BLOCK4_THIS_THREAD: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn record(kind: Loop) {
        match kind {
            Loop::Fixed => {
                FIXED_ANY_THREAD.fetch_add(1, Ordering::Relaxed);
                FIXED_THIS_THREAD.with(|calls| calls.set(calls.get() + 1));
            }
            Loop::Block4 => {
                BLOCK4_ANY_THREAD.fetch_add(1, Ordering::Relaxed);
                BLOCK4_THIS_THREAD.with(|calls| calls.set(calls.get() + 1));
            }
        }
    }

    /// Calls of `kind` made on the current thread.
    pub(crate) fn on_this_thread(kind: Loop) -> usize {
        match kind {
            Loop::Fixed => FIXED_THIS_THREAD.with(Cell::get),
            Loop::Block4 => BLOCK4_THIS_THREAD.with(Cell::get),
        }
    }

    /// Calls of `kind` made on any thread, for scoring that runs on
    /// rayon's pool. Tests running at the same time add to it as well.
    pub(crate) fn on_any_thread(kind: Loop) -> usize {
        match kind {
            Loop::Fixed => FIXED_ANY_THREAD.load(Ordering::Relaxed),
            Loop::Block4 => BLOCK4_ANY_THREAD.load(Ordering::Relaxed),
        }
    }
}

/// Query token vectors ready to score documents stored under any
/// [`MultiVectorStorage`] kind (#1346, #1427).
///
/// - A borrowed, aligned `F32` document is scored zero-copy.
/// - A `Binary` document whose parameters were prepared with
///   [`Self::prepare_binary`] is scored from its bits through
///   [`BinaryQuery`]'s lookup tables, without decoding it.
/// - Any other document (`F16`, `Int8`, an unprepared `Binary`) is decoded
///   into the caller's scratch first: decoding is `O(document_rows)`
///   against MaxSim's `O(query_rows × document_rows)` dot products.
///
/// [`MultiVectorStorage`]: crate::vector::core::multi_vector::MultiVectorStorage
#[derive(Debug)]
pub(crate) struct MaxSimQuery<'q> {
    /// Row-major query vectors.
    rows: &'q [f32],
    dimension: usize,
    /// Prepared tables, one per distinct parameter set.
    binary: Vec<(Arc<BinaryParams>, BinaryQuery)>,
}

impl<'q> MaxSimQuery<'q> {
    /// Most bytes of `Binary` lookup tables one query is prepared with.
    /// The tables take `ceil(dimension / 8)` KiB per query vector: 512 KiB
    /// for 32 vectors of 128 dimensions, so the cap admits up to 256 such
    /// vectors. A query needing more is decoded instead: it may hold up to
    /// 1,024 vectors, and a dimension has no upper bound.
    pub(crate) const MAX_BINARY_TABLE_BYTES: usize = 4 << 20;

    /// Wrap `rows` (row-major, whole rows of `dimension` elements). No
    /// `Binary` parameter set is prepared yet.
    pub(crate) fn new(rows: &'q [f32], dimension: usize) -> Self {
        debug_assert!(dimension > 0 && rows.len().is_multiple_of(dimension));
        Self {
            rows,
            dimension,
            binary: Vec::new(),
        }
    }

    /// Prepare to score rows encoded with `params` from their bits.
    ///
    /// Does nothing when an equal parameter set is already prepared, or
    /// when the tables would exceed [`Self::MAX_BINARY_TABLE_BYTES`]: such
    /// rows are decoded instead.
    pub(crate) fn prepare_binary(&mut self, params: &Arc<BinaryParams>) {
        debug_assert_eq!(params.dimension(), self.dimension);
        let query_rows = self.rows.len() / self.dimension;
        if BinaryQuery::table_bytes(self.dimension, query_rows) > Self::MAX_BINARY_TABLE_BYTES
            || self.binary.iter().any(|(prepared, _)| prepared == params)
        {
            return;
        }
        self.binary
            .push((params.clone(), BinaryQuery::new(params, self.rows)));
    }

    /// How many `Binary` parameter sets are prepared.
    #[cfg(test)]
    pub(crate) fn prepared(&self) -> usize {
        self.binary.len()
    }

    /// MaxSim of one document: `Σ_i max_j q_i · d_j`.
    ///
    /// # Arguments
    ///
    /// * `document` - The document's rows, at least one, of this query's
    ///   dimension.
    /// * `scratch` - Reused across calls: decoded rows, or a prepared
    ///   `Binary` document's per-row factors.
    pub(crate) fn max_sim(&self, document: &MultiVectorRows<'_>, scratch: &mut Vec<f32>) -> f32 {
        debug_assert_eq!(document.dimension(), self.dimension);
        if let Some(floats) = document.as_f32() {
            return max_sim(self.rows, floats, self.dimension);
        }
        if let Some(params) = document.codec().binary_params()
            && let Some((_, query)) = self
                .binary
                .iter()
                .find(|(prepared, _)| std::ptr::eq(&**prepared, params) || **prepared == *params)
        {
            return query.max_sim(document.as_bytes(), scratch);
        }
        document.decode_into(scratch);
        max_sim(self.rows, scratch, self.dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward scalar MaxSim, the reference the SIMD path must match.
    fn reference(query: &[Vec<f32>], document: &[Vec<f32>]) -> f32 {
        query
            .iter()
            .map(|q| {
                document
                    .iter()
                    .map(|d| q.iter().zip(d).map(|(a, b)| a * b).sum::<f32>())
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()
    }

    /// Deterministic pseudo-random values in `[-1, 1)`.
    fn values(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn rows(flat: &[f32], dimension: usize) -> Vec<Vec<f32>> {
        flat.chunks_exact(dimension).map(<[f32]>::to_vec).collect()
    }

    #[test]
    fn test_matches_scalar_reference_across_dimensions() {
        for dimension in [1, 7, 8, 9, 64, 96, 128] {
            for (query_rows, doc_rows) in [(1, 1), (3, 5), (32, 17)] {
                let query = values(dimension as u64, query_rows * dimension);
                let document = values(1000 + dimension as u64, doc_rows * dimension);
                let expected = reference(&rows(&query, dimension), &rows(&document, dimension));
                let actual = max_sim(&query, &document, dimension);
                assert!(
                    (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                    "dimension {dimension}, {query_rows}x{doc_rows}: {actual} vs {expected}"
                );
            }
        }
    }

    /// Issue #1442: the dimensions with a fixed copy go through it and
    /// score bit for bit as the run-time loop.
    #[test]
    fn test_fixed_dimensions_match_the_run_time_loop_bit_for_bit() {
        for dimension in [64, 96, 128] {
            for (query_rows, doc_rows) in [(1, 1), (3, 5), (32, 17)] {
                let query = values(dimension as u64, query_rows * dimension);
                let document = values(2000 + dimension as u64, doc_rows * dimension);
                let before = dispatch_calls::on_this_thread(dispatch_calls::Loop::Fixed);
                let actual = max_sim(&query, &document, dimension);
                assert_eq!(
                    dispatch_calls::on_this_thread(dispatch_calls::Loop::Fixed),
                    before + 1,
                    "dimension {dimension} should take its fixed copy"
                );
                assert_eq!(
                    actual.to_bits(),
                    max_sim_dyn(&query, &document, dimension).to_bits(),
                    "dimension {dimension}, {query_rows}x{doc_rows}"
                );
            }
        }
    }

    /// Issue #1444: every dimension without a fixed copy takes the
    /// four-vector-per-pass loop and scores bit for bit as the
    /// one-at-a-time run-time loop.
    #[test]
    fn test_other_dimensions_take_the_block4_loop() {
        for dimension in [1, 7, 8, 63, 65, 95, 97, 127, 129, 768] {
            for (query_rows, doc_rows) in [(1, 1), (3, 5), (32, 17)] {
                let query = values(dimension as u64, query_rows * dimension);
                let document = values(3000 + dimension as u64, doc_rows * dimension);
                let before = dispatch_calls::on_this_thread(dispatch_calls::Loop::Block4);
                let actual = max_sim(&query, &document, dimension);
                assert_eq!(
                    dispatch_calls::on_this_thread(dispatch_calls::Loop::Block4),
                    before + 1,
                    "dimension {dimension} should take the block4 loop"
                );
                assert_eq!(
                    actual.to_bits(),
                    max_sim_dyn(&query, &document, dimension).to_bits(),
                    "dimension {dimension}, {query_rows}x{doc_rows}"
                );
            }
        }
    }

    /// Issue #1444: `max_sim_block4` against `max_sim_dyn`, directly and at
    /// document-row counts that are and are not a multiple of 4, so both
    /// `dot4`'s tail (dimensions not a multiple of 8) and
    /// `max_sim_block4`'s remainder block (rows not a multiple of 4) are
    /// exercised.
    #[test]
    fn test_block4_matches_the_run_time_loop_bit_for_bit() {
        for dimension in [1, 7, 20, 32, 100, 256] {
            for (query_rows, doc_rows) in [(1, 1), (3, 4), (3, 5), (2, 8), (32, 17)] {
                let query = values(4000 + dimension as u64, query_rows * dimension);
                let document = values(5000 + dimension as u64, doc_rows * dimension);
                let actual = max_sim_block4(&query, &document, dimension);
                let expected = max_sim_dyn(&query, &document, dimension);
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "dimension {dimension}, {query_rows}x{doc_rows}"
                );
            }
        }
    }

    #[test]
    fn test_takes_the_best_document_vector_per_query_vector() {
        // Query rows pick different document rows; the sum is 2 + 3.
        let query = [1.0, 0.0, 0.0, 1.0];
        let document = [2.0, 0.0, 0.0, 3.0, -5.0, -5.0];
        assert_eq!(max_sim(&query, &document, 2), 5.0);
    }

    #[test]
    fn test_negative_similarities_are_kept() {
        // Every pair is negative; the best (least negative) one counts.
        let query = [1.0, 1.0];
        let document = [-1.0, -2.0, -0.5, -0.5];
        assert_eq!(max_sim(&query, &document, 2), -1.0);
    }

    #[test]
    fn test_single_document_vector() {
        let query = [1.0, 2.0, 3.0, 4.0];
        let document = [0.5, 0.5];
        assert_eq!(max_sim(&query, &document, 2), 1.5 + 3.5);
    }

    use std::borrow::Cow;

    use crate::vector::core::multi_vector::{
        MultiVectorStorage, RowCodec, binary_max_sim_tolerance, test_binary_params,
    };

    fn encode(codec: RowCodec<'_>, dimension: usize, rows: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in rows.chunks_exact(dimension) {
            codec.encode_row(row, &mut out);
        }
        out
    }

    /// The document MaxSim is checked against, and the tolerance. For the
    /// kinds with a per-element error bound it is the input itself: f16's
    /// 11-bit significand bounds its relative error to ~2⁻¹¹; int8's
    /// per-vector scale (dividing the `[-1, 1]`-ish ColBERT range into 254
    /// steps) accumulates more, with ample safety margin at these
    /// dimensions. The lossy `Binary` is held to its own row-by-row decode,
    /// exactly when its rows are decoded.
    fn reference_document(
        codec: RowCodec<'_>,
        dimension: usize,
        document: &[f32],
        query_rows: usize,
    ) -> (Vec<f32>, f32) {
        match codec.kind() {
            MultiVectorStorage::F32 => (document.to_vec(), 0.0),
            MultiVectorStorage::F16 => (document.to_vec(), 2e-3 * query_rows as f32),
            MultiVectorStorage::Int8 => (document.to_vec(), 5e-2 * query_rows as f32),
            MultiVectorStorage::Binary => {
                let mut decoded = Vec::new();
                let encoded = encode(codec, dimension, document);
                for row in encoded.chunks_exact(codec.row_bytes(dimension)) {
                    codec.decode_row(row, dimension, &mut decoded);
                }
                (decoded, 0.0)
            }
        }
    }

    #[test]
    fn test_max_sim_query_matches_plain_max_sim_for_every_storage_kind() {
        let dimension = 128;
        let params = Arc::new(test_binary_params(dimension));
        for (query_rows, doc_rows) in [(1, 1), (3, 5), (32, 17)] {
            let query = values(1, query_rows * dimension);
            let document = values(2, doc_rows * dimension);
            let unprepared = MaxSimQuery::new(&query, dimension);
            let mut prepared = MaxSimQuery::new(&query, dimension);
            prepared.prepare_binary(&params);
            for codec in [
                RowCodec::F32,
                RowCodec::F16,
                RowCodec::Int8,
                RowCodec::Binary(&params),
            ] {
                let (target, tol) = reference_document(codec, dimension, &document, query_rows);
                let expected = max_sim(&query, &target, dimension);
                let encoded = encode(codec, dimension, &document);
                let view = MultiVectorRows::new(codec, dimension, Cow::Borrowed(&encoded));
                let actual = unprepared.max_sim(&view, &mut Vec::new());
                assert!(
                    (actual - expected).abs() <= tol,
                    "{:?} {query_rows}x{doc_rows}, decoded: {actual} vs {expected} (tol {tol})",
                    codec.kind()
                );
                // Prepared, a `Binary` document is scored from its bits.
                let tol = match codec.binary_params() {
                    Some(params) => binary_max_sim_tolerance(params, &query, &encoded),
                    None => tol,
                };
                let actual = prepared.max_sim(&view, &mut Vec::new());
                assert!(
                    (actual - expected).abs() <= tol,
                    "{:?} {query_rows}x{doc_rows}, prepared: {actual} vs {expected} (tol {tol})",
                    codec.kind()
                );
            }
        }
    }

    /// An encoded `Binary` document whose score through the prepared tables
    /// differs in its bits from the decode path's, so that a test can tell
    /// which path scored it.
    fn distinguishing_document(params: &BinaryParams, query: &[f32]) -> Vec<u8> {
        let dimension = params.dimension();
        let codec = RowCodec::Binary(params);
        let tables = BinaryQuery::new(params, query);
        (2..64)
            .map(|seed| encode(codec, dimension, &values(seed, 17 * dimension)))
            .find(|bytes| {
                let mut decoded = Vec::new();
                MultiVectorRows::new(codec, dimension, Cow::Borrowed(bytes))
                    .decode_into(&mut decoded);
                tables.max_sim(bytes, &mut Vec::new()).to_bits()
                    != max_sim(query, &decoded, dimension).to_bits()
            })
            .expect("some document scores differently through the decode")
    }

    #[test]
    fn test_prepared_binary_documents_are_scored_from_their_bits() {
        let dimension = 96;
        let params = Arc::new(test_binary_params(dimension));
        let query = values(1, 32 * dimension);
        let bytes = distinguishing_document(&params, &query);
        let from_bits = BinaryQuery::new(&params, &query).max_sim(&bytes, &mut Vec::new());
        let view =
            MultiVectorRows::new(RowCodec::Binary(&params), dimension, Cow::Borrowed(&bytes));
        let mut decoded = Vec::new();
        view.decode_into(&mut decoded);
        let through_decode = max_sim(&query, &decoded, dimension);

        let mut prepared = MaxSimQuery::new(&query, dimension);
        assert_eq!(
            prepared.max_sim(&view, &mut Vec::new()).to_bits(),
            through_decode.to_bits()
        );
        prepared.prepare_binary(&params);
        assert_eq!(prepared.prepared(), 1);
        assert_eq!(
            prepared.max_sim(&view, &mut Vec::new()).to_bits(),
            from_bits.to_bits()
        );

        // Another segment's copy of the same parameters is matched by value
        // and not prepared again.
        let copy = Arc::new(test_binary_params(dimension));
        assert!(!Arc::ptr_eq(&params, &copy));
        prepared.prepare_binary(&copy);
        assert_eq!(prepared.prepared(), 1);
        let copy_view =
            MultiVectorRows::new(RowCodec::Binary(&copy), dimension, Cow::Borrowed(&bytes));
        assert_eq!(
            prepared.max_sim(&copy_view, &mut Vec::new()).to_bits(),
            from_bits.to_bits()
        );

        // Different parameters get tables of their own.
        let other = Arc::new(BinaryParams::train(dimension, &values(5, 8 * dimension)).unwrap());
        assert_ne!(*other, *params);
        prepared.prepare_binary(&other);
        assert_eq!(prepared.prepared(), 2);
    }

    #[test]
    fn test_queries_over_the_table_cap_are_decoded() {
        let dimension = 1024;
        let params = Arc::new(test_binary_params(dimension));
        let at_cap = MaxSimQuery::MAX_BINARY_TABLE_BYTES / BinaryQuery::table_bytes(dimension, 1);
        assert_eq!(at_cap, 32);
        let bytes = encode(
            RowCodec::Binary(&params),
            dimension,
            &values(2, 3 * dimension),
        );
        let view =
            MultiVectorRows::new(RowCodec::Binary(&params), dimension, Cow::Borrowed(&bytes));
        let mut decoded = Vec::new();
        view.decode_into(&mut decoded);
        for (query_rows, expected) in [(at_cap, 1), (at_cap + 1, 0)] {
            let query = values(1, query_rows * dimension);
            let mut prepared = MaxSimQuery::new(&query, dimension);
            prepared.prepare_binary(&params);
            assert_eq!(prepared.prepared(), expected, "{query_rows} query rows");
            if expected == 0 {
                assert_eq!(
                    prepared.max_sim(&view, &mut Vec::new()).to_bits(),
                    max_sim(&query, &decoded, dimension).to_bits()
                );
            }
        }
    }
}
