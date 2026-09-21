//! Shared writer / reader helpers for the HNSW quantized vector
//! payload (Issue #481 Stage 1, Step 5).
//!
//! The HNSW segment layout puts `(doc_id, field_name)` and the
//! quantized vector record together for each entry, which prevents
//! reusing [`crate::vector::index::quantized_segment::QuantizedSegmentVectors`]
//! directly (that type assumes a homogeneous AoS payload).
//!
//! This module centralises the per-vector record encoding so
//! `HnswIndexWriter::write` (writer) and `HnswIndexWriter::load` /
//! `HnswIndexReader::load` (readers) cannot drift apart.
//!
//! # On-disk layout for the quantized region
//!
//! ```text
//! [ VectorSegmentHeader  24 bytes ]   <- written / read separately
//! repeat num_vectors times:
//!   [ doc_id           u64  LE   8 bytes ]
//!   [ field ref: v3+ = field_id u16 (per-segment dictionary,     ]
//!   [   Issue #633); v1/v2 = name_len u32 + UTF-8 name           ]
//!   [ int8 data            dim bytes ]
//!   [ sum_q             u32  LE   4 bytes ]
//!   [ norm_q            f32  LE   4 bytes ]
//! ```

use std::io::{Read, Write};

use crate::error::{LaurusError, Result};
use crate::vector::core::quantization::{QuantizedVectorMeta, ScalarQuantParams};

/// Bytes consumed by the int8 + meta portion of one vector record
/// (everything after the field_name string ends).
#[inline]
pub(super) const fn quantized_record_payload_size(dim: usize) -> usize {
    dim + QuantizedVectorMeta::SERIALIZED_SIZE
}

/// Train segment-level [`ScalarQuantParams`] from an iterator over each
/// vector's raw f32 slice, substituting neutral `(0.0, 1.0)` defaults
/// for an empty segment instead of propagating
/// [`ScalarQuantParams::train_from_slices`]'s "empty input" error --
/// every caller still needs *some* params to serialize an empty
/// segment's header (Issue #1151: centralizes the
/// `if is_empty { defaults } else { train(...) }` pattern each writer
/// previously duplicated around the old `quantize_segment`).
///
/// Does NOT substitute defaults for a non-empty iterator whose vectors
/// are all zero-dimensional (`train_from_slices`'s other error
/// condition) -- that case is left erroring, matching prior behavior.
pub(super) fn train_quant_params_or_neutral<'a>(
    slices: impl ExactSizeIterator<Item = &'a [f32]>,
) -> Result<ScalarQuantParams> {
    if slices.len() == 0 {
        return Ok(ScalarQuantParams {
            offset: 0.0,
            scale: 1.0,
        });
    }
    ScalarQuantParams::train_from_slices(slices)
}

/// Quantize one vector's raw f32 data into `out` and return its
/// per-vector meta, without allocating a new `Vec<u8>` per call --
/// `out` is cleared and reused, so a caller can hoist one scratch
/// buffer across an entire segment's emission loop instead of
/// collecting a `Vec<QuantizedRecord>` up front (Issue #1151).
///
/// Mirrors `VectorQuantizer::quantize`'s Scalar8Bit arm exactly,
/// including its dimension check -- [`ScalarQuantParams::quantize`]
/// itself does not check, so reproducing it here is the one guard this
/// split must not drop.
///
/// # Errors
///
/// [`LaurusError::InvalidOperation`] if `data.len() != dim`.
pub(super) fn quantize_into(
    params: &ScalarQuantParams,
    dim: usize,
    data: &[f32],
    out: &mut Vec<u8>,
) -> Result<QuantizedVectorMeta> {
    if data.len() != dim {
        return Err(LaurusError::InvalidOperation(format!(
            "Vector dimension mismatch: expected {dim}, got {}",
            data.len()
        )));
    }
    out.clear();
    out.extend(data.iter().map(|&v| params.quantize_value(v)));
    Ok(QuantizedVectorMeta::from_quantized(out, params))
}

/// Write the int8 + meta tail of one vector record.
///
/// The caller is responsible for writing the preceding `doc_id` /
/// `field_name_len` / `field_name` fields.
pub(super) fn write_quantized_record<W: Write>(
    output: &mut W,
    int8_data: &[u8],
    meta: QuantizedVectorMeta,
) -> Result<()> {
    output.write_all(int8_data)?;
    output.write_all(&meta.sum_q.to_le_bytes())?;
    output.write_all(&meta.norm_q.to_le_bytes())?;
    Ok(())
}

/// Read the int8 + meta tail of one vector record and dequantize it
/// back to a `Vec<f32>` of length `dim`.
///
/// Used by load paths that keep the in-memory representation as f32
/// (Step 5 of #481 Stage 1; Step 6 will switch the in-memory form to
/// int8 directly).
pub(super) fn read_dequantized_vector<R: Read>(
    input: &mut R,
    dim: usize,
    params: &ScalarQuantParams,
) -> Result<Vec<f32>> {
    let mut int8_buf = vec![0u8; dim];
    input.read_exact(&mut int8_buf)?;
    // Skip the 8-byte meta (sum_q + norm_q): the in-memory form is
    // f32 in this step, so we don't need them here. They will become
    // load-time state in Step 6.
    let mut meta_buf = [0u8; QuantizedVectorMeta::SERIALIZED_SIZE];
    input.read_exact(&mut meta_buf)?;
    Ok(int8_buf
        .iter()
        .map(|&b| params.dequantize_value(b))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::core::quantization::{QuantizationMethod, VectorQuantizer};
    use crate::vector::core::vector::Vector;
    use std::io::Cursor;

    /// Issue #1151: `train_quant_params_or_neutral` + `quantize_into`
    /// (the streaming replacement) must produce results identical to
    /// `VectorQuantizer` (the non-streaming API that is NOT being
    /// removed) for the same input -- this is the invariant the split
    /// must preserve forever.
    #[test]
    fn streaming_primitives_match_vector_quantizer() {
        let dim = 3;
        let vectors = vec![
            Vector::new(vec![-1.0, 0.0, 1.0]),
            Vector::new(vec![-0.5, 0.5, 0.25]),
            Vector::new(vec![0.1, -0.4, 0.9]),
        ];

        let params =
            train_quant_params_or_neutral(vectors.iter().map(|v| v.data.as_slice())).unwrap();

        let mut quantizer = VectorQuantizer::new(QuantizationMethod::Scalar8Bit, dim);
        quantizer.train(&vectors).unwrap();
        let expected_params = *quantizer.params().unwrap();
        assert_eq!(params, expected_params);

        let mut scratch = Vec::new();
        for (i, v) in vectors.iter().enumerate() {
            let meta = quantize_into(&params, dim, &v.data, &mut scratch).unwrap();
            let (expected_q, expected_meta) = quantizer.quantize(v).unwrap();
            assert_eq!(scratch, expected_q, "vector {i}");
            assert_eq!(meta.sum_q, expected_meta.sum_q, "vector {i}");
            assert_eq!(
                meta.norm_q.to_bits(),
                expected_meta.norm_q.to_bits(),
                "vector {i}: norm_q must be bit-identical"
            );
        }
    }

    /// Issue #1151: an empty segment must not error, matching the
    /// `if is_empty { defaults } else { train(...) }` convention every
    /// writer previously duplicated around `quantize_segment`.
    #[test]
    fn train_quant_params_or_neutral_returns_defaults_for_empty_input() {
        let empty: Vec<&[f32]> = Vec::new();
        let params = train_quant_params_or_neutral(empty.into_iter()).unwrap();
        assert_eq!(
            params,
            ScalarQuantParams {
                offset: 0.0,
                scale: 1.0
            }
        );
    }

    #[test]
    fn write_then_read_dequantized_roundtrips_within_scale() {
        let dim = 8;
        let vectors = vec![
            Vector::new(vec![-1.0, -0.7, -0.3, 0.0, 0.2, 0.5, 0.8, 1.0]),
            Vector::new(vec![0.1, 0.2, 0.3, 0.4, -0.4, -0.3, -0.2, -0.1]),
        ];
        let params =
            train_quant_params_or_neutral(vectors.iter().map(|v| v.data.as_slice())).unwrap();

        let mut buf = Vec::new();
        let mut scratch = Vec::new();
        for v in &vectors {
            let meta = quantize_into(&params, dim, &v.data, &mut scratch).unwrap();
            write_quantized_record(&mut buf, &scratch, meta).unwrap();
        }
        assert_eq!(
            buf.len(),
            vectors.len() * quantized_record_payload_size(dim)
        );

        let mut cursor = Cursor::new(&buf);
        for (i, original) in vectors.iter().enumerate() {
            let recovered = read_dequantized_vector(&mut cursor, dim, &params).unwrap();
            for (j, (orig, rec)) in original.data.iter().zip(recovered.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() <= params.scale + 1e-6,
                    "vector {i} dim {j}: orig = {orig}, rec = {rec}, scale = {}",
                    params.scale
                );
            }
        }
    }

    #[test]
    fn payload_size_is_dim_plus_eight() {
        assert_eq!(quantized_record_payload_size(0), 8);
        assert_eq!(quantized_record_payload_size(128), 136);
    }
}
