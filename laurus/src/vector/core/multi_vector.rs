//! On-disk element kinds for `MultiVector` token-vector storage (#1346).
//!
//! A `MultiVector` field (#1177) stores every token vector of a document
//! with no ANN index; it is only read back for exact late-interaction
//! (MaxSim) rescoring. [`MultiVectorStorage`] picks how each token vector
//! is encoded on disk:
//!
//! - [`MultiVectorStorage::F32`]: 4 bytes/element, exact. Default; the
//!   only kind ever written before this field existed, so old segments
//!   keep opening unchanged.
//! - [`MultiVectorStorage::F16`]: 2 bytes/element (2× smaller), IEEE-754
//!   binary16. ~2⁻¹¹ relative error per element.
//! - [`MultiVectorStorage::Int8`]: 1 byte/element plus a 2-byte f16 scale
//!   *per vector* (not per segment): `scale = max(abs(v)) / 127`,
//!   `code_i = round(v_i / scale)`. A self-contained row needs no
//!   corpus- or segment-trained state, which is what lets a segment
//!   merge copy rows byte-for-byte instead of re-quantizing (and
//!   compounding loss) on every merge.
//! - [`MultiVectorStorage::Binary`]: 1 bit/element. Every row is the sign
//!   pattern of the token's offset from a field-level center `c`; see
//!   [`BinaryParams`] for the encoding and why it is centered and
//!   projected.
//!
//! [`RowCodec`] pairs a kind with the parameters it needs (only `Binary`
//! has any), so rows of a parameterized kind cannot be encoded or decoded
//! without them.

use std::borrow::Cow;
use std::sync::Arc;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::error::{LaurusError, Result};

/// On-disk element kind of a `MultiVector` field's token vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MultiVectorStorage {
    /// IEEE-754 binary32, 4 bytes/element. Exact. Default.
    #[default]
    F32,
    /// IEEE-754 binary16, 2 bytes/element.
    F16,
    /// Signed 8-bit codes with a trailing per-vector f16 scale,
    /// `1 + 2/dimension` bytes/element.
    Int8,
    /// One bit per element: the sign of each element's offset from a
    /// center learned once per field (`ceil(dimension / 8)` bytes per
    /// row). Requires `DistanceMetric::Cosine`.
    Binary,
}

impl MultiVectorStorage {
    /// Numeric tag written to the LMV1 header's element-kind byte.
    pub const fn tag(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Int8 => 2,
            Self::Binary => 3,
        }
    }

    /// Inverse of [`Self::tag`]. `None` for any tag this build cannot
    /// read.
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Int8),
            3 => Some(Self::Binary),
            _ => None,
        }
    }

    /// Bytes one row of `dimension` elements occupies on disk.
    pub const fn row_bytes(self, dimension: usize) -> usize {
        match self {
            Self::F32 => dimension * 4,
            Self::F16 => dimension * 2,
            Self::Int8 => dimension + 2,
            Self::Binary => dimension.div_ceil(8),
        }
    }

    /// Bytes of the per-segment parameter block this kind stores after
    /// the LMV1 header: [`BinaryParams::block_len`] for `Binary`, `0`
    /// otherwise.
    pub(crate) const fn params_len(self, dimension: usize) -> usize {
        match self {
            Self::Binary => BinaryParams::block_len(dimension),
            Self::F32 | Self::F16 | Self::Int8 => 0,
        }
    }
}

/// Field-level parameters of [`MultiVectorStorage::Binary`].
///
/// A token `v` (L2-normalized: `Binary` requires a `Cosine` field) is
/// stored as the bits `b_i = (v_i - c_i) > 0` and decoded as
///
/// ```text
/// x_i = w_i * (b_i ? +1 : -1)
/// out = c + x - ((x·c) / (c·c)) c
/// ```
///
/// - `c` is the mean training token and `w_i` the mean `|v_i - c_i|`, both
///   stored as f16. Token vectors of a late-interaction model are highly
///   anisotropic (they share one dominant direction), so the plain sign
///   of `v` is nearly the same for every token; centering on `c` keeps
///   the bits informative.
/// - The decoded offset `x` is projected off `c`. The true offset `v - c`
///   is nearly orthogonal to `c` for unit `v`, but its sign pattern is
///   not, and every query token (itself close to `c`) would read that
///   spurious component as a query-independent bias that swamps MaxSim.
/// - There is deliberately no per-row scale: on BEIR any per-row or
///   per-document term (a scale, the offset's true component along `c`,
///   a per-document center) lost far more than it recovered (#1346).
///
/// The parameters are learned once per field and stored in every
/// `Binary` segment (see the LMV1 format), so each segment decodes on its
/// own. Their identity is the f16 bit pattern: two parameter sets are
/// equal exactly when their stored bits are, which is when a merge may
/// copy rows byte-for-byte.
#[derive(Debug, Clone)]
pub(crate) struct BinaryParams {
    /// `c` then `w`, as f16 bit patterns: what is stored on disk.
    bits: Box<[u16]>,
    center: Box<[f32]>,
    spread: Box<[f32]>,
    /// `w_i * c_i`, so decoding gets `x·c` as a signed sum.
    spread_center: Box<[f32]>,
    /// `c·c`.
    center_norm_sq: f32,
}

impl PartialEq for BinaryParams {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl Eq for BinaryParams {}

impl BinaryParams {
    /// Token vectors a field needs before its parameters are learned.
    /// Measured on BEIR (#1346): parameters from ~1.2K tokens already
    /// rank as well as ones from the whole corpus; 236 tokens fall short.
    pub(crate) const MIN_TRAINING_ROWS: usize = 2048;

    /// At most this many token vectors are used to learn the parameters,
    /// whatever the size of the flush or merge that learns them.
    pub(crate) const MAX_TRAINING_ROWS: usize = 16_384;

    /// Bytes of the parameter block for `dimension`: `dimension` f16
    /// values of `c`, then `dimension` of `w`, zero-padded to a multiple
    /// of 8 so the following data stays 8-byte aligned.
    pub(crate) const fn block_len(dimension: usize) -> usize {
        (4 * dimension).next_multiple_of(8)
    }

    /// Learn the parameters from `rows` (row-major, whole rows).
    ///
    /// `c` is the mean in f64, rounded to f16; `w` is then measured
    /// against the **rounded** `c` and rounded to f16, the same `c` every
    /// row is later encoded against.
    ///
    /// # Errors
    ///
    /// Returns an error when `rows` is empty or not whole rows, or when a
    /// learned value is not finite in f16.
    pub(crate) fn train(dimension: usize, rows: &[f32]) -> Result<Self> {
        if dimension == 0 || rows.is_empty() || !rows.len().is_multiple_of(dimension) {
            return Err(LaurusError::internal(
                "binary parameters need at least one whole row to learn from",
            ));
        }
        let count = (rows.len() / dimension) as f64;
        let mut sum = vec![0f64; dimension];
        for row in rows.chunks_exact(dimension) {
            for (s, v) in sum.iter_mut().zip(row) {
                *s += f64::from(*v);
            }
        }
        let center: Vec<f16> = sum.iter().map(|s| f16::from_f64(s / count)).collect();
        let mut deviation = vec![0f64; dimension];
        for row in rows.chunks_exact(dimension) {
            for ((d, v), c) in deviation.iter_mut().zip(row).zip(&center) {
                *d += f64::from((v - c.to_f32()).abs());
            }
        }
        let spread = deviation.iter().map(|d| f16::from_f64(d / count));
        let bits = center
            .iter()
            .copied()
            .chain(spread)
            .map(f16::to_bits)
            .collect();
        Self::from_f16_bits(dimension, bits)
    }

    /// Build from stored f16 bits: `c`, then `w`.
    ///
    /// # Errors
    ///
    /// Returns an error when `bits` is not `2 * dimension` values, a value
    /// is not finite, or a spread is negative.
    pub(crate) fn from_f16_bits(dimension: usize, bits: Box<[u16]>) -> Result<Self> {
        if dimension == 0 || bits.len() != 2 * dimension {
            return Err(LaurusError::index(
                "binary parameters do not match the dimension",
            ));
        }
        let widened: Vec<f32> = bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
        let (center, spread) = widened.split_at(dimension);
        if widened.iter().any(|v| !v.is_finite()) || spread.iter().any(|w| *w < 0.0) {
            return Err(LaurusError::index(
                "binary parameters hold a non-finite value or a negative spread",
            ));
        }
        let spread_center = spread.iter().zip(center).map(|(w, c)| w * c).collect();
        let center_norm_sq = center.iter().map(|c| c * c).sum();
        Ok(Self {
            bits,
            center: center.into(),
            spread: spread.into(),
            spread_center,
            center_norm_sq,
        })
    }

    /// Dimension these parameters were learned for.
    pub(crate) fn dimension(&self) -> usize {
        self.center.len()
    }

    /// Append the parameter block ([`Self::block_len`] bytes) to `out`.
    pub(crate) fn write_block(&self, out: &mut Vec<u8>) {
        let start = out.len();
        for b in &self.bits {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out.resize(start + Self::block_len(self.dimension()), 0);
    }

    /// Parse a parameter block written by [`Self::write_block`].
    ///
    /// # Errors
    ///
    /// Returns an error when `bytes` is shorter than the block or the
    /// values are rejected by [`Self::from_f16_bits`].
    pub(crate) fn read_block(bytes: &[u8], dimension: usize) -> Result<Self> {
        let values = 2 * dimension;
        let Some(stored) = bytes.get(..values * 2) else {
            return Err(LaurusError::index("binary parameter block is truncated"));
        };
        let bits = stored
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        Self::from_f16_bits(dimension, bits)
    }

    /// Append one row's bits (`ceil(dimension / 8)` bytes, LSB first,
    /// padding bits zero) to `out`.
    fn encode_row(&self, values: &[f32], out: &mut Vec<u8>) {
        let start = out.len();
        out.resize(start + values.len().div_ceil(8), 0);
        for (i, (v, c)) in values.iter().zip(&*self.center).enumerate() {
            if v - c > 0.0 {
                out[start + i / 8] |= 1 << (i % 8);
            }
        }
    }

    /// Decode one row's bits and push its `dimension` values onto `out`.
    fn decode_row(&self, bytes: &[u8], out: &mut Vec<f32>) {
        let start = out.len();
        let mut along = 0f32;
        out.extend(
            self.spread
                .iter()
                .zip(&*self.spread_center)
                .enumerate()
                .map(|(i, (w, wc))| {
                    if bytes[i / 8] >> (i % 8) & 1 == 1 {
                        along += wc;
                        *w
                    } else {
                        along -= wc;
                        -w
                    }
                }),
        );
        let off = if self.center_norm_sq > 0.0 {
            along / self.center_norm_sq
        } else {
            0.0
        };
        for (x, c) in out[start..].iter_mut().zip(&*self.center) {
            *x += c * (1.0 - off);
        }
    }
}

/// How a set of rows is encoded: the element kind plus, for `Binary`, the
/// parameters its rows were encoded with.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RowCodec<'p> {
    F32,
    F16,
    Int8,
    Binary(&'p BinaryParams),
}

impl<'p> RowCodec<'p> {
    pub(crate) fn kind(self) -> MultiVectorStorage {
        match self {
            Self::F32 => MultiVectorStorage::F32,
            Self::F16 => MultiVectorStorage::F16,
            Self::Int8 => MultiVectorStorage::Int8,
            Self::Binary(_) => MultiVectorStorage::Binary,
        }
    }

    /// The `Binary` parameters, `None` for any other kind.
    pub(crate) fn binary_params(self) -> Option<&'p BinaryParams> {
        match self {
            Self::Binary(params) => Some(params),
            Self::F32 | Self::F16 | Self::Int8 => None,
        }
    }

    pub(crate) fn row_bytes(self, dimension: usize) -> usize {
        self.kind().row_bytes(dimension)
    }

    /// Whether rows encoded with `self` read the same under `other`, i.e.
    /// whether a merge may copy them byte-for-byte: the same kind and,
    /// for `Binary`, equal parameters. Re-encoding a decoded `Binary` row
    /// can flip bits even with the same parameters, so this must hold for
    /// a copy, not just for a cheaper path.
    pub(crate) fn same_encoding(self, other: RowCodec<'_>) -> bool {
        match (self, other) {
            (Self::Binary(a), RowCodec::Binary(b)) => std::ptr::eq(a, b) || a == b,
            (a, b) => a.kind() == b.kind(),
        }
    }

    /// Append one row's worth of encoded bytes to `out`.
    ///
    /// `values.len()` must equal the dimension (and, for `Binary`, the
    /// parameters' dimension); the caller (the LMV1 writer) already
    /// enforces whole-row writes.
    pub(crate) fn encode_row(self, values: &[f32], out: &mut Vec<u8>) {
        match self {
            Self::F32 => {
                for v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Self::F16 => {
                for v in values {
                    out.extend_from_slice(&f16_saturating(*v).to_le_bytes());
                }
            }
            Self::Int8 => encode_int8_row(values, out),
            Self::Binary(params) => params.encode_row(values, out),
        }
    }

    /// Decode one row (`self.row_bytes(dimension)` bytes) and push its
    /// `dimension` values onto `out`.
    pub(crate) fn decode_row(self, bytes: &[u8], dimension: usize, out: &mut Vec<f32>) {
        match self {
            Self::F32 => {
                out.extend(
                    bytes[..dimension * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c)),
                );
            }
            Self::F16 => {
                out.extend(
                    bytes[..dimension * 2]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| f16::from_bits(u16::from_le_bytes(*c)).to_f32()),
                );
            }
            Self::Int8 => decode_int8_row(bytes, dimension, out),
            Self::Binary(params) => params.decode_row(bytes, out),
        }
    }
}

/// [`RowCodec`] owning its parameters: what a segment reader or a segment
/// being written holds.
#[derive(Debug, Clone)]
pub(crate) enum OwnedCodec {
    F32,
    F16,
    Int8,
    Binary(Arc<BinaryParams>),
}

impl OwnedCodec {
    /// The codec of a kind that needs no parameters; `None` for `Binary`.
    pub(crate) fn plain(kind: MultiVectorStorage) -> Option<Self> {
        match kind {
            MultiVectorStorage::F32 => Some(Self::F32),
            MultiVectorStorage::F16 => Some(Self::F16),
            MultiVectorStorage::Int8 => Some(Self::Int8),
            MultiVectorStorage::Binary => None,
        }
    }

    pub(crate) fn kind(&self) -> MultiVectorStorage {
        self.as_codec().kind()
    }

    pub(crate) fn as_codec(&self) -> RowCodec<'_> {
        match self {
            Self::F32 => RowCodec::F32,
            Self::F16 => RowCodec::F16,
            Self::Int8 => RowCodec::Int8,
            Self::Binary(params) => RowCodec::Binary(params),
        }
    }
}

/// Convert to f16, saturating instead of overflowing to infinity. `half`'s
/// plain `f16::from_f32` follows IEEE-754 round-to-nearest and produces
/// `inf` past `f16::MAX` (65504), which would poison a `DotProduct`
/// field's MaxSim with a non-finite score; NaN passes through unchanged.
fn f16_saturating(v: f32) -> f16 {
    if v.is_nan() {
        return f16::from_f32(v);
    }
    f16::from_f32(v.clamp(f16::MIN.to_f32(), f16::MAX.to_f32()))
}

/// Encode one int8 row: `dimension` signed codes, then a trailing f16
/// scale. Quantizes using the **f16-rounded** scale (not the f32 scale
/// before rounding), so decoding never produces a code outside `[-127,
/// 127]` from the f16 round-trip itself.
fn encode_int8_row(values: &[f32], out: &mut Vec<u8>) {
    let max_abs = values.iter().fold(0f32, |acc, v| acc.max(v.abs()));
    let scale = if max_abs > 0.0 && max_abs.is_finite() {
        f16::from_f32(max_abs / 127.0)
    } else {
        f16::from_f32(0.0)
    };
    let scale_f32 = scale.to_f32();
    for &v in values {
        let code = if scale_f32 > 0.0 {
            (v / scale_f32).round().clamp(-127.0, 127.0) as i8
        } else {
            0i8
        };
        out.push(code as u8);
    }
    out.extend_from_slice(&scale.to_le_bytes());
}

fn decode_int8_row(bytes: &[u8], dimension: usize, out: &mut Vec<f32>) {
    let scale =
        f16::from_bits(u16::from_le_bytes([bytes[dimension], bytes[dimension + 1]])).to_f32();
    out.extend(bytes[..dimension].iter().map(|&b| (b as i8) as f32 * scale));
}

/// Reinterpret little-endian `f32` bytes in place, if they are aligned.
fn try_as_f32_slice(bytes: &[u8]) -> Option<&[f32]> {
    if cfg!(target_endian = "big") {
        return None;
    }
    // SAFETY: every bit pattern is a valid `f32`, and `align_to` only puts
    // bytes into the middle slice when they are correctly aligned for it.
    let (head, floats, tail) = unsafe { bytes.align_to::<f32>() };
    (head.is_empty() && tail.is_empty()).then_some(floats)
}

/// One document's token vectors, still encoded in their on-disk element
/// kind.
///
/// Returned by the multi-vector reader; decoded on demand by
/// [`crate::vector::core::late_interaction::max_sim_rows`]. `bytes` holds
/// exactly `row_count() * kind().row_bytes(dimension)` bytes.
pub struct MultiVectorRows<'a> {
    codec: RowCodec<'a>,
    dimension: usize,
    bytes: Cow<'a, [u8]>,
}

impl<'a> MultiVectorRows<'a> {
    /// Wrap already-validated row bytes. `bytes.len()` must be a multiple
    /// of `codec.row_bytes(dimension)`.
    pub(crate) fn new(codec: RowCodec<'a>, dimension: usize, bytes: Cow<'a, [u8]>) -> Self {
        debug_assert!(dimension > 0);
        debug_assert_eq!(bytes.len() % codec.row_bytes(dimension), 0);
        debug_assert!(
            codec
                .binary_params()
                .is_none_or(|p| p.dimension() == dimension)
        );
        Self {
            codec,
            dimension,
            bytes,
        }
    }

    pub fn kind(&self) -> MultiVectorStorage {
        self.codec.kind()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn row_count(&self) -> usize {
        self.bytes.len() / self.codec.row_bytes(self.dimension)
    }

    /// How these rows are encoded, parameters included.
    pub(crate) fn codec(&self) -> RowCodec<'a> {
        self.codec
    }

    /// The raw encoded bytes, for a byte-copy merge.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Zero-copy `&[f32]` view, only possible for a borrowed `F32` row
    /// set on a little-endian, 4-byte-aligned payload.
    ///
    /// Tied to `'a` (this row set's underlying data), not to `&self`, so
    /// it outlives a temporary `MultiVectorRows` — e.g. inside
    /// `reader.rows(id)?.map(|r| r.as_f32())`.
    pub fn as_f32(&self) -> Option<&'a [f32]> {
        if !matches!(self.codec, RowCodec::F32) {
            return None;
        }
        match &self.bytes {
            // Deref coercion copies the inner `&'a [u8]` out of the
            // `&self`-tied `&&'a [u8]` match binding here, so the result
            // keeps the longer lifetime `'a` instead of being tied to
            // this method call.
            Cow::Borrowed(bytes) => try_as_f32_slice(bytes),
            Cow::Owned(_) => None,
        }
    }

    /// Decode every row into `out` (cleared first), row-major.
    pub fn decode_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.reserve(self.row_count() * self.dimension);
        let row_bytes = self.codec.row_bytes(self.dimension);
        for row in self.bytes.chunks_exact(row_bytes) {
            self.codec.decode_row(row, self.dimension, out);
        }
    }

    /// `as_f32`, or a freshly decoded owned vector when that is not
    /// possible.
    pub fn to_f32(&self) -> Cow<'a, [f32]> {
        match self.as_f32() {
            Some(floats) => Cow::Borrowed(floats),
            None => {
                let mut out = Vec::new();
                self.decode_into(&mut out);
                Cow::Owned(out)
            }
        }
    }
}

/// Fixed, f16-exact `Binary` parameters for tests: `c_i = ±(i % 4 + 1) / 16`
/// (alternating sign) and `w_i = (i % 3 + 1) / 8`.
#[cfg(test)]
pub(crate) fn test_binary_params(dimension: usize) -> BinaryParams {
    let center = (0..dimension).map(|i| {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        f16::from_f32(sign * (i % 4 + 1) as f32 / 16.0)
    });
    let spread = (0..dimension).map(|i| f16::from_f32((i % 3 + 1) as f32 / 8.0));
    BinaryParams::from_f16_bits(dimension, center.chain(spread).map(f16::to_bits).collect())
        .expect("test parameters are finite")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(kind: MultiVectorStorage, values: &[f32]) -> Vec<f32> {
        let owned = OwnedCodec::plain(kind).expect("a kind without parameters");
        let codec = owned.as_codec();
        let mut bytes = Vec::new();
        codec.encode_row(values, &mut bytes);
        assert_eq!(bytes.len(), kind.row_bytes(values.len()));
        let mut out = Vec::new();
        codec.decode_row(&bytes, values.len(), &mut out);
        out
    }

    fn encode(codec: RowCodec<'_>, values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        codec.encode_row(values, &mut bytes);
        bytes
    }

    fn decode(codec: RowCodec<'_>, bytes: &[u8], dimension: usize) -> Vec<f32> {
        let mut out = Vec::new();
        codec.decode_row(bytes, dimension, &mut out);
        out
    }

    /// The documented decode, written out independently in f64:
    /// `x_i = ±w_i`, `out = c + x - ((x·c) / (c·c)) c`.
    fn reference_decode(params: &BinaryParams, bits: &[bool]) -> Vec<f32> {
        let c: Vec<f64> = params.center.iter().map(|&v| f64::from(v)).collect();
        let x: Vec<f64> = params
            .spread
            .iter()
            .zip(bits)
            .map(|(&w, &b)| if b { f64::from(w) } else { -f64::from(w) })
            .collect();
        let cc: f64 = c.iter().map(|v| v * v).sum();
        let off = if cc > 0.0 {
            x.iter().zip(&c).map(|(a, b)| a * b).sum::<f64>() / cc
        } else {
            0.0
        };
        x.iter()
            .zip(&c)
            .map(|(x, c)| (c + x - off * c) as f32)
            .collect()
    }

    /// Deterministic pseudo-random unit vectors.
    fn unit_vectors(count: usize, dimension: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        let mut out = Vec::with_capacity(count * dimension);
        for _ in 0..count {
            let row: Vec<f32> = (0..dimension)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    // A shared offset makes the set anisotropic, like real
                    // late-interaction token vectors.
                    (state >> 8) as f32 / (1u32 << 24) as f32 - 0.2
                })
                .collect();
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            out.extend(row.iter().map(|v| v / norm));
        }
        out
    }

    #[test]
    fn test_tag_round_trips_for_every_supported_kind() {
        for kind in [
            MultiVectorStorage::F32,
            MultiVectorStorage::F16,
            MultiVectorStorage::Int8,
            MultiVectorStorage::Binary,
        ] {
            assert_eq!(MultiVectorStorage::from_tag(kind.tag()), Some(kind));
        }
    }

    #[test]
    fn test_f32_tag_is_zero() {
        // LMV1 v1 always wrote element kind 0; the default must keep
        // reading those segments.
        assert_eq!(MultiVectorStorage::F32.tag(), 0);
        assert_eq!(MultiVectorStorage::default(), MultiVectorStorage::F32);
    }

    #[test]
    fn test_out_of_range_tags_are_rejected() {
        assert_eq!(MultiVectorStorage::from_tag(4), None);
        assert_eq!(MultiVectorStorage::from_tag(255), None);
    }

    #[test]
    fn test_row_bytes_matches_the_documented_layout() {
        assert_eq!(MultiVectorStorage::F32.row_bytes(128), 512);
        assert_eq!(MultiVectorStorage::F16.row_bytes(128), 256);
        assert_eq!(MultiVectorStorage::Int8.row_bytes(128), 130);
        assert_eq!(MultiVectorStorage::Binary.row_bytes(128), 16);
        assert_eq!(MultiVectorStorage::Binary.row_bytes(96), 12);
        assert_eq!(MultiVectorStorage::Binary.row_bytes(10), 2);
    }

    #[test]
    fn test_only_binary_has_a_parameter_block() {
        assert_eq!(MultiVectorStorage::Binary.params_len(96), 384);
        assert_eq!(MultiVectorStorage::Binary.params_len(3), 16);
        for kind in [
            MultiVectorStorage::F32,
            MultiVectorStorage::F16,
            MultiVectorStorage::Int8,
        ] {
            assert_eq!(kind.params_len(96), 0);
            assert_eq!(OwnedCodec::plain(kind).map(|c| c.kind()), Some(kind));
        }
        assert!(OwnedCodec::plain(MultiVectorStorage::Binary).is_none());
    }

    #[test]
    fn test_f32_round_trip_is_exact() {
        let values = [1.0f32, -2.5, 0.0, 1e10, -1e-10];
        assert_eq!(roundtrip(MultiVectorStorage::F32, &values), values);
    }

    #[test]
    fn test_f16_round_trip_is_close() {
        let values = [0.1f32, -0.3, 0.0, 1.0, -1.0, 0.004];
        let decoded = roundtrip(MultiVectorStorage::F16, &values);
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() <= 2e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn test_f16_exhaustive_widening_is_finite_or_matches_ieee754() {
        // Every possible f16 bit pattern must decode without panicking,
        // and round-trip through f32 and back to the same bits (f16 is
        // representable exactly within f32).
        for bits in 0u16..=u16::MAX {
            let h = f16::from_bits(bits);
            let f = h.to_f32();
            let back = f16::from_f32(f);
            if h.is_nan() {
                assert!(back.is_nan());
            } else {
                assert_eq!(h.to_bits(), back.to_bits(), "bits {bits:#06x}");
            }
        }
    }

    #[test]
    fn test_f16_overflow_saturates_instead_of_producing_infinity() {
        let bytes = encode(RowCodec::F16, &[1e30, -1e30]);
        let out = decode(RowCodec::F16, &bytes, 2);
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
        assert!(out[0] > 0.0 && out[1] < 0.0);
    }

    #[test]
    fn test_int8_round_trip_error_is_within_the_derived_bound() {
        // L2-normalized-ish values typical of ColBERT token vectors.
        let values: Vec<f32> = (0..128).map(|i| ((i as f32 * 0.073).sin()) * 0.3).collect();
        let decoded = roundtrip(MultiVectorStorage::Int8, &values);
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() <= 5e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn test_int8_zero_vector_round_trips_to_exact_zero() {
        let values = [0.0f32; 16];
        let decoded = roundtrip(MultiVectorStorage::Int8, &values);
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_int8_non_finite_values_round_trip_to_zero_without_panicking() {
        let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5];
        let decoded = roundtrip(MultiVectorStorage::Int8, &values);
        assert!(decoded.iter().all(|v| *v == 0.0), "{decoded:?}");
    }

    #[test]
    fn test_int8_codes_never_exceed_127_in_magnitude() {
        // The scale is derived from the f16-rounded value, not the raw
        // f32 max, which is exactly what prevents a code of 128.
        let bytes = encode(RowCodec::Int8, &[1.0f32; 64]);
        for &b in &bytes[..64] {
            assert!((b as i8).unsigned_abs() <= 127);
        }
    }

    #[test]
    fn test_binary_bits_are_lsb_first_with_zero_padding() {
        // c = [1/16, -2/16, 3/16, -4/16, 1/16, -2/16, 3/16, -4/16, 1/16, -2/16]
        let params = test_binary_params(10);
        let values = [1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.1875, -0.3, 0.07, -0.124];
        // v - c > 0 at dimensions 0, 3, 5, 8, 9; dimension 6 is a tie
        // (v == c), which is not > 0.
        assert_eq!(
            encode(RowCodec::Binary(&params), &values),
            [0b0010_1001, 0b0000_0011]
        );
    }

    #[test]
    fn test_binary_decode_projects_the_offset_off_the_center() {
        let params = test_binary_params(10);
        let codec = RowCodec::Binary(&params);
        for pattern in [
            0u16,
            0b11_1111_1111,
            0b10_1010_1010,
            0b01_1001_0110,
            0b11_0000_0001,
        ] {
            let bytes = pattern.to_le_bytes();
            let bits: Vec<bool> = (0..10).map(|i| pattern >> i & 1 == 1).collect();
            let decoded = decode(codec, &bytes, 10);
            let expected = reference_decode(&params, &bits);
            for (a, e) in decoded.iter().zip(&expected) {
                assert!((a - e).abs() <= 1e-6, "{decoded:?} vs {expected:?}");
            }
            // The decoded offset from c has no component along c.
            let along: f32 = decoded
                .iter()
                .zip(&*params.center)
                .map(|(v, c)| (v - c) * c)
                .sum();
            assert!(along.abs() <= 1e-6, "offset along c: {along}");
        }
    }

    #[test]
    fn test_binary_decode_with_a_zero_center_is_the_plain_offset() {
        let bits: Box<[u16]> = [0.0f32, 0.0, 0.0, 0.5, 0.25, 0.125]
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect();
        let params = BinaryParams::from_f16_bits(3, bits).unwrap();
        let decoded = decode(RowCodec::Binary(&params), &[0b101], 3);
        assert_eq!(decoded, [0.5, -0.25, 0.125]);
    }

    #[test]
    fn test_binary_train_learns_the_rounded_mean_and_spread() {
        let params = BinaryParams::train(2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]).unwrap();
        assert_eq!(&*params.center, &[0.5, 0.5]);
        // |v - c| per dimension: 0.5, 0.5, 0 -> mean 1/3, rounded to f16.
        let third = f16::from_f64(1.0 / 3.0).to_f32();
        assert_eq!(&*params.spread, &[third, third]);
        assert!(BinaryParams::train(2, &[]).is_err());
        assert!(BinaryParams::train(2, &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_binary_encoding_survives_writing_and_reading_the_parameters() {
        let dimension = 96;
        let sample = unit_vectors(64, dimension, 7);
        let trained = BinaryParams::train(dimension, &sample).unwrap();
        let mut block = Vec::new();
        trained.write_block(&mut block);
        assert_eq!(block.len(), BinaryParams::block_len(dimension));
        let read = BinaryParams::read_block(&block, dimension).unwrap();
        assert_eq!(read, trained);
        for row in unit_vectors(16, dimension, 11).chunks_exact(dimension) {
            assert_eq!(
                encode(RowCodec::Binary(&trained), row),
                encode(RowCodec::Binary(&read), row)
            );
        }
    }

    #[test]
    fn test_binary_parameter_block_rejects_bad_values() {
        let mut block = Vec::new();
        test_binary_params(3).write_block(&mut block);
        assert!(BinaryParams::read_block(&block[..11], 3).is_err());

        let nan = f16::NAN.to_bits();
        let mut bits: Vec<u16> = vec![0; 6];
        bits[1] = nan;
        assert!(BinaryParams::from_f16_bits(3, bits.into()).is_err());
        let mut bits: Vec<u16> = vec![0; 6];
        bits[4] = f16::from_f32(-0.5).to_bits();
        assert!(BinaryParams::from_f16_bits(3, bits.into()).is_err());
        assert!(BinaryParams::from_f16_bits(3, vec![0; 5].into()).is_err());
    }

    #[test]
    fn test_same_encoding_compares_binary_parameters_by_value() {
        let a = Arc::new(test_binary_params(4));
        let b = Arc::new(test_binary_params(4));
        assert!(!Arc::ptr_eq(&a, &b));
        assert!(RowCodec::Binary(&a).same_encoding(RowCodec::Binary(&b)));

        let other = BinaryParams::train(4, &[0.5, 0.5, 0.5, 0.5]).unwrap();
        assert!(!RowCodec::Binary(&a).same_encoding(RowCodec::Binary(&other)));
        assert!(!RowCodec::Binary(&a).same_encoding(RowCodec::F16));
        assert!(!RowCodec::F16.same_encoding(RowCodec::Binary(&a)));
        assert!(RowCodec::F16.same_encoding(RowCodec::F16));
        assert!(!RowCodec::F16.same_encoding(RowCodec::Int8));
    }

    #[test]
    fn test_binary_decoded_norm_stays_within_the_unit_ball() {
        // For parameters learned from unit vectors, ‖c‖² + ‖w‖² <= 1, so
        // every decoded row keeps a cosine score per query vector <= 1
        // (up to f16 rounding).
        let dimension = 96;
        let params = BinaryParams::train(dimension, &unit_vectors(256, dimension, 3)).unwrap();
        let codec = RowCodec::Binary(&params);
        for row in unit_vectors(64, dimension, 5).chunks_exact(dimension) {
            let decoded = decode(codec, &encode(codec, row), dimension);
            let norm_sq: f32 = decoded.iter().map(|v| v * v).sum();
            assert!(norm_sq <= 1.0 + 1e-3, "{norm_sq}");
        }
    }

    #[test]
    fn test_rows_decode_into_matches_per_row_decode() {
        let dim = 4;
        let row_a = [1.0f32, 2.0, 3.0, 4.0];
        let row_b = [-1.0f32, 0.5, 0.0, -0.25];
        let params = test_binary_params(dim);
        for codec in [
            RowCodec::F32,
            RowCodec::F16,
            RowCodec::Int8,
            RowCodec::Binary(&params),
        ] {
            let mut bytes = encode(codec, &row_a);
            bytes.extend(encode(codec, &row_b));
            let rows = MultiVectorRows::new(codec, dim, Cow::Owned(bytes));
            assert_eq!(rows.row_count(), 2);
            assert_eq!(rows.kind(), codec.kind());
            let mut decoded = Vec::new();
            rows.decode_into(&mut decoded);
            let mut expected = decode(codec, &encode(codec, &row_a), dim);
            expected.extend(decode(codec, &encode(codec, &row_b), dim));
            assert_eq!(decoded, expected, "{:?}", codec.kind());
        }
    }

    #[test]
    fn test_as_f32_is_zero_copy_only_for_borrowed_f32() {
        let dim = 4;
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = encode(RowCodec::F32, &values);

        let borrowed = MultiVectorRows::new(RowCodec::F32, dim, Cow::Borrowed(&bytes));
        assert!(borrowed.as_f32().is_some());
        assert_eq!(borrowed.as_f32().unwrap(), &values);

        let owned = MultiVectorRows::new(RowCodec::F32, dim, Cow::Owned(bytes.clone()));
        assert!(owned.as_f32().is_none());
        assert_eq!(&*owned.to_f32(), &values);

        let f16_bytes = encode(RowCodec::F16, &values);
        let f16_rows = MultiVectorRows::new(RowCodec::F16, dim, Cow::Borrowed(&f16_bytes));
        assert!(f16_rows.as_f32().is_none());
    }
}
