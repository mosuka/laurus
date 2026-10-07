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
//!
//! Element kind `3` is reserved for a future 1-bit kind and is rejected by
//! [`MultiVectorStorage::from_tag`].

use std::borrow::Cow;

use half::f16;
use serde::{Deserialize, Serialize};

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
}

impl MultiVectorStorage {
    /// Numeric tag written to the LMV1 header's element-kind byte. `3` is
    /// reserved for a future 1-bit kind.
    pub const fn tag(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Int8 => 2,
        }
    }

    /// Inverse of [`Self::tag`]. `None` for any tag this build cannot
    /// read, including the reserved 1-bit tag `3`.
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Int8),
            _ => None,
        }
    }

    /// Bytes one row of `dimension` elements occupies on disk.
    pub const fn row_bytes(self, dimension: usize) -> usize {
        match self {
            Self::F32 => dimension * 4,
            Self::F16 => dimension * 2,
            Self::Int8 => dimension + 2,
        }
    }

    /// Append one row's worth of encoded bytes to `out`.
    ///
    /// `values.len()` must equal `dimension`; the caller (the LMV1 writer)
    /// already enforces whole-row writes.
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
/// exactly `row_count() * kind.row_bytes(dimension)` bytes.
pub struct MultiVectorRows<'a> {
    kind: MultiVectorStorage,
    dimension: usize,
    bytes: Cow<'a, [u8]>,
}

impl<'a> MultiVectorRows<'a> {
    /// Wrap already-validated row bytes. `bytes.len()` must be a multiple
    /// of `kind.row_bytes(dimension)`.
    pub(crate) fn new(kind: MultiVectorStorage, dimension: usize, bytes: Cow<'a, [u8]>) -> Self {
        debug_assert!(dimension > 0);
        debug_assert_eq!(bytes.len() % kind.row_bytes(dimension), 0);
        Self {
            kind,
            dimension,
            bytes,
        }
    }

    pub fn kind(&self) -> MultiVectorStorage {
        self.kind
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn row_count(&self) -> usize {
        self.bytes.len() / self.kind.row_bytes(self.dimension)
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
        if self.kind != MultiVectorStorage::F32 {
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
        let row_bytes = self.kind.row_bytes(self.dimension);
        for row in self.bytes.chunks_exact(row_bytes) {
            self.kind.decode_row(row, self.dimension, out);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(kind: MultiVectorStorage, values: &[f32]) -> Vec<f32> {
        let mut bytes = Vec::new();
        kind.encode_row(values, &mut bytes);
        assert_eq!(bytes.len(), kind.row_bytes(values.len()));
        let mut out = Vec::new();
        kind.decode_row(&bytes, values.len(), &mut out);
        out
    }

    #[test]
    fn test_tag_round_trips_for_every_supported_kind() {
        for kind in [
            MultiVectorStorage::F32,
            MultiVectorStorage::F16,
            MultiVectorStorage::Int8,
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
    fn test_reserved_and_out_of_range_tags_are_rejected() {
        assert_eq!(MultiVectorStorage::from_tag(3), None);
        assert_eq!(MultiVectorStorage::from_tag(255), None);
    }

    #[test]
    fn test_row_bytes_matches_the_documented_layout() {
        assert_eq!(MultiVectorStorage::F32.row_bytes(128), 512);
        assert_eq!(MultiVectorStorage::F16.row_bytes(128), 256);
        assert_eq!(MultiVectorStorage::Int8.row_bytes(128), 130);
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
        let mut bytes = Vec::new();
        MultiVectorStorage::F16.encode_row(&[1e30, -1e30], &mut bytes);
        let mut out = Vec::new();
        MultiVectorStorage::F16.decode_row(&bytes, 2, &mut out);
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
        let values = [1.0f32; 64];
        let mut bytes = Vec::new();
        MultiVectorStorage::Int8.encode_row(&values, &mut bytes);
        for &b in &bytes[..64] {
            assert!((b as i8).unsigned_abs() <= 127);
        }
    }

    #[test]
    fn test_rows_decode_into_matches_per_row_decode() {
        let dim = 4;
        let row_a = [1.0f32, 2.0, 3.0, 4.0];
        let row_b = [-1.0f32, 0.5, 0.0, -0.25];
        for kind in [
            MultiVectorStorage::F32,
            MultiVectorStorage::F16,
            MultiVectorStorage::Int8,
        ] {
            let mut bytes = Vec::new();
            kind.encode_row(&row_a, &mut bytes);
            kind.encode_row(&row_b, &mut bytes);
            let rows = MultiVectorRows::new(kind, dim, Cow::Owned(bytes));
            assert_eq!(rows.row_count(), 2);
            let mut decoded = Vec::new();
            rows.decode_into(&mut decoded);
            let mut expected = Vec::new();
            kind.decode_row(
                &{
                    let mut b = Vec::new();
                    kind.encode_row(&row_a, &mut b);
                    b
                },
                dim,
                &mut expected,
            );
            kind.decode_row(
                &{
                    let mut b = Vec::new();
                    kind.encode_row(&row_b, &mut b);
                    b
                },
                dim,
                &mut expected,
            );
            assert_eq!(decoded, expected, "{kind:?}");
        }
    }

    #[test]
    fn test_as_f32_is_zero_copy_only_for_borrowed_f32() {
        let dim = 4;
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let mut bytes = Vec::new();
        MultiVectorStorage::F32.encode_row(&values, &mut bytes);

        let borrowed = MultiVectorRows::new(MultiVectorStorage::F32, dim, Cow::Borrowed(&bytes));
        assert!(borrowed.as_f32().is_some());
        assert_eq!(borrowed.as_f32().unwrap(), &values);

        let owned = MultiVectorRows::new(MultiVectorStorage::F32, dim, Cow::Owned(bytes.clone()));
        assert!(owned.as_f32().is_none());
        assert_eq!(&*owned.to_f32(), &values);

        let mut f16_bytes = Vec::new();
        MultiVectorStorage::F16.encode_row(&values, &mut f16_bytes);
        let f16_rows =
            MultiVectorRows::new(MultiVectorStorage::F16, dim, Cow::Borrowed(&f16_bytes));
        assert!(f16_rows.as_f32().is_none());
    }
}
