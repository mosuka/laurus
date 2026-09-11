//! In-memory representation and on-disk I/O for one quantized
//! vector segment (Issue #481 Stage 1).
//!
//! A segment groups N vectors that share one [`ScalarQuantParams`]
//! pair. Per-vector data is stored AoS (Array-of-Structs) for cache
//! locality on the search hot path: each record packs the int8
//! payload immediately followed by its [`QuantizedVectorMeta`]
//! (`sum_q` + `norm_q`).
//!
//! # On-disk layout
//!
//! ```text
//! [ VectorSegmentHeader        24 bytes (16 fixed + 8 SQ params) ]
//! [ dim          :u32 LE        4 bytes ]
//! [ vector_count :u32 LE        4 bytes ]
//! [ AoS records  vector_count * (dim + 8) bytes ]
//! ```
//!
//! `dim` and `vector_count` are encoded in the segment so a reader
//! can reconstruct the layout without external metadata. They live
//! after [`VectorSegmentHeader`] (rather than inside it) to keep the
//! header focused on type identification, allowing future quant kinds
//! (PQ in Stage 3) to reuse the same dim / count prefix.
//!
//! # In-memory layout
//!
//! [`QuantizedSegmentVectors::data`] is one contiguous `Vec<u8>` of
//! length `vector_count * (dim + 8)`. Vector `i`'s record starts at
//! byte offset `i * record_size(dim)`.

use std::io::{Read, Write};

use crate::error::Result;
use crate::vector::core::quantization::{
    QuantizationMethod, QuantizedVectorMeta, ScalarQuantParams, VectorQuantizer,
};
use crate::vector::core::vector::Vector;
use crate::vector::index::format::{QuantHeader, VectorSegmentHeader};

/// A complete quantized vector segment held in memory.
///
/// Built from f32 vectors via [`Self::from_f32_vectors`] at flush time
/// or read back from disk via [`Self::read_from`] at search time.
#[derive(Debug, Clone)]
pub struct QuantizedSegmentVectors {
    /// Segment-level quantization params. Shared by every vector.
    pub params: ScalarQuantParams,
    /// Vector dimension (number of `u8` elements per record).
    pub dim: usize,
    /// Number of vectors in this segment.
    pub vector_count: usize,
    /// Tightly-packed AoS payload, length
    /// `vector_count * (dim + QuantizedVectorMeta::SERIALIZED_SIZE)`.
    pub data: Vec<u8>,
}

impl QuantizedSegmentVectors {
    /// Bytes per vector record on disk and in memory:
    /// `dim` int8 elements + 8 bytes meta (`sum_q` + `norm_q`).
    #[inline]
    pub const fn record_size(dim: usize) -> usize {
        dim + QuantizedVectorMeta::SERIALIZED_SIZE
    }

    /// Train per-segment params on `vectors` and quantize each.
    ///
    /// All input vectors must have dimension `dim`; otherwise an
    /// [`crate::error::LaurusError::InvalidOperation`] is returned.
    ///
    /// # Errors
    ///
    /// Forwards any error from
    /// [`VectorQuantizer::train`] / [`VectorQuantizer::quantize`].
    pub fn from_f32_vectors(vectors: &[Vector], dim: usize) -> Result<Self> {
        let mut quantizer = VectorQuantizer::new(QuantizationMethod::Scalar8Bit, dim);
        quantizer.train(vectors)?;
        let params = *quantizer
            .params()
            .expect("quantizer trained successfully implies params are set");

        let vector_count = vectors.len();
        let record = Self::record_size(dim);
        let mut data = Vec::with_capacity(vector_count * record);
        for v in vectors {
            let (q, meta) = quantizer.quantize(v)?;
            debug_assert_eq!(q.len(), dim);
            data.extend_from_slice(&q);
            data.extend_from_slice(&meta.sum_q.to_le_bytes());
            data.extend_from_slice(&meta.norm_q.to_le_bytes());
        }

        Ok(Self {
            params,
            dim,
            vector_count,
            data,
        })
    }

    /// Borrow the int8 payload slice for vector `i`.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if `i >= vector_count`. Release mode
    /// returns potentially-incorrect bytes — the caller (HNSW
    /// searcher) keeps the index within bounds via the graph
    /// structure.
    #[inline]
    pub fn vector_data(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.vector_count, "index out of bounds: {i}");
        let start = i * Self::record_size(self.dim);
        &self.data[start..start + self.dim]
    }

    /// Read back the per-vector metadata for vector `i`.
    ///
    /// Decodes the 4-byte `sum_q` (LE u32) and 4-byte `norm_q`
    /// (LE f32) that follow the int8 payload.
    #[inline]
    pub fn vector_meta(&self, i: usize) -> QuantizedVectorMeta {
        debug_assert!(i < self.vector_count, "index out of bounds: {i}");
        let meta_start = i * Self::record_size(self.dim) + self.dim;
        let sum_q_bytes: [u8; 4] = self.data[meta_start..meta_start + 4]
            .try_into()
            .expect("4 bytes available");
        let norm_q_bytes: [u8; 4] = self.data[meta_start + 4..meta_start + 8]
            .try_into()
            .expect("4 bytes available");
        QuantizedVectorMeta {
            sum_q: u32::from_le_bytes(sum_q_bytes),
            norm_q: f32::from_le_bytes(norm_q_bytes),
        }
    }

    /// Total serialized size including header and dim/count prefix.
    pub fn serialized_size(&self) -> usize {
        let header = VectorSegmentHeader::scalar_8bit(self.params);
        header.serialized_size() + 4 + 4 + self.data.len()
    }

    /// Write the segment to `writer`: header, dim, count, then AoS data.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        let header = VectorSegmentHeader::scalar_8bit(self.params);
        header.write_to(writer)?;
        writer.write_all(&(self.dim as u32).to_le_bytes())?;
        writer.write_all(&(self.vector_count as u32).to_le_bytes())?;
        writer.write_all(&self.data)?;
        Ok(())
    }

    /// Read a segment from `reader`. Self-describing: `dim` and
    /// `vector_count` are recovered from the on-disk prefix.
    ///
    /// # Arguments
    ///
    /// * `reader` - Stream positioned at the start of the segment.
    /// * `available` - Bytes the source can still supply (the file or
    ///   buffer length), bounding the header's PQ codebook allocation
    ///   and this segment's record-data allocation against ground truth
    ///   (Issue #921 / #806).
    ///
    /// # Errors
    ///
    /// * Forwards any error from [`VectorSegmentHeader::read_from`]
    ///   (incompatible magic / version / quant_kind, oversized codebook).
    /// * [`crate::error::LaurusError::Index`] if the declared record
    ///   data cannot fit in `available` bytes (corruption).
    /// * Returns [`crate::error::LaurusError::Io`] for any underlying
    ///   read failure (EOF, etc.).
    pub fn read_from<R: Read>(reader: &mut R, available: u64) -> Result<Self> {
        let header = VectorSegmentHeader::read_from(reader, available)?;
        // Stage 3 (PQ) segments are handled by a separate code path
        // (see `crate::vector::index::pq_segment`); this Stage 1
        // helper only reads Scalar8Bit segments.
        let params = match header.quant {
            QuantHeader::Scalar8Bit(p) => p,
            QuantHeader::ProductQuantization { .. } => {
                return Err(crate::error::LaurusError::NotImplemented(
                    "QuantizedSegmentVectors (Stage 1) cannot read PQ segments; \
                     the PQ-aware reader lives in vector::index::pq_segment"
                        .to_string(),
                ));
            }
            #[cfg(feature = "pq-fastscan")]
            QuantHeader::ProductQuantizationFastScan { .. } => {
                return Err(crate::error::LaurusError::NotImplemented(
                    "QuantizedSegmentVectors (Stage 1) cannot read PQ FastScan \
                     segments; the FastScan reader lives in vector::index::hnsw::reader"
                        .to_string(),
                ));
            }
        };

        let mut dim_bytes = [0u8; 4];
        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut dim_bytes)?;
        reader.read_exact(&mut count_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        let vector_count = u32::from_le_bytes(count_bytes) as usize;

        // Issue #921: `dim`/`vector_count` are unverified u32s; bound the
        // data buffer against the source's real length before allocating
        // (the header + prefix already consumed can only make the true
        // remainder smaller, so `available` is a safe upper bound).
        let total = vector_count.saturating_mul(Self::record_size(dim));
        crate::util::alloc_bounds::checked_len(total, available, "quantized segment data size")?;
        let mut data = vec![0u8; total];
        reader.read_exact(&mut data)?;

        Ok(Self {
            params,
            dim,
            vector_count,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_vectors(n: usize, dim: usize) -> Vec<Vector> {
        let mut state: u32 = 0xDEAD_BEEF;
        (0..n)
            .map(|_| {
                let data: Vec<f32> = (0..dim)
                    .map(|_| {
                        state = state.wrapping_mul(1103515245).wrapping_add(12345);
                        // Project into [-1.0, 1.0]
                        let bits = (state >> 16) as u16;
                        (bits as f32 / u16::MAX as f32) * 2.0 - 1.0
                    })
                    .collect();
                Vector::new(data)
            })
            .collect()
    }

    #[test]
    fn from_f32_quantizes_each_vector_with_segment_params() {
        let dim = 16;
        let vectors = sample_vectors(8, dim);
        let seg = QuantizedSegmentVectors::from_f32_vectors(&vectors, dim).unwrap();
        assert_eq!(seg.dim, dim);
        assert_eq!(seg.vector_count, 8);
        assert_eq!(
            seg.data.len(),
            8 * QuantizedSegmentVectors::record_size(dim)
        );

        // Each vector's first byte should round-trip via params.
        for (i, v) in vectors.iter().enumerate() {
            let q = seg.vector_data(i);
            let expected_first = seg.params.quantize_value(v.data[0]);
            assert_eq!(q[0], expected_first, "vector {i} first element mismatch");
        }
    }

    #[test]
    fn vector_meta_matches_recomputed_meta() {
        let dim = 32;
        let vectors = sample_vectors(4, dim);
        let seg = QuantizedSegmentVectors::from_f32_vectors(&vectors, dim).unwrap();
        for i in 0..seg.vector_count {
            let bytes = seg.vector_data(i);
            let expected = QuantizedVectorMeta::from_quantized(bytes, &seg.params);
            let actual = seg.vector_meta(i);
            assert_eq!(actual.sum_q, expected.sum_q, "sum_q at vector {i}");
            assert!(
                (actual.norm_q - expected.norm_q).abs() < 1e-5,
                "norm_q at vector {i}: expected {}, got {}",
                expected.norm_q,
                actual.norm_q
            );
        }
    }

    #[test]
    fn write_then_read_roundtrips() {
        let dim = 16;
        let vectors = sample_vectors(5, dim);
        let original = QuantizedSegmentVectors::from_f32_vectors(&vectors, dim).unwrap();

        let mut buf: Vec<u8> = Vec::new();
        original.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), original.serialized_size());

        let mut cursor = Cursor::new(&buf);
        let recovered = QuantizedSegmentVectors::read_from(&mut cursor, buf.len() as u64).unwrap();
        assert_eq!(recovered.dim, original.dim);
        assert_eq!(recovered.vector_count, original.vector_count);
        assert_eq!(recovered.params, original.params);
        assert_eq!(recovered.data, original.data);
    }

    #[test]
    fn read_rejects_pre_stage1_segment() {
        // Build a buffer that LOOKS like raw f32 vectors (no LVS1 header).
        let raw: Vec<u8> = (0..16).flat_map(|i: u8| [i, 0, 0, 0]).collect();
        let mut cursor = Cursor::new(&raw);
        let err = QuantizedSegmentVectors::read_from(&mut cursor, raw.len() as u64).unwrap_err();
        match err {
            crate::error::LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("LVS1"), "msg: {msg}");
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn read_rejects_truncated_data_section() {
        let dim = 8;
        let vectors = sample_vectors(3, dim);
        let original = QuantizedSegmentVectors::from_f32_vectors(&vectors, dim).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        original.write_to(&mut buf).unwrap();

        // Truncate the last vector record.
        let truncated_len = buf.len() - QuantizedSegmentVectors::record_size(dim) / 2;
        buf.truncate(truncated_len);
        let err = QuantizedSegmentVectors::read_from(&mut Cursor::new(&buf), buf.len() as u64)
            .unwrap_err();
        assert!(matches!(err, crate::error::LaurusError::Io(_)));
    }

    #[test]
    fn record_size_is_dim_plus_eight() {
        assert_eq!(QuantizedSegmentVectors::record_size(0), 8);
        assert_eq!(QuantizedSegmentVectors::record_size(128), 136);
        assert_eq!(QuantizedSegmentVectors::record_size(768), 776);
    }

    #[test]
    fn empty_segment_writes_only_header_plus_prefix() {
        let dim = 4;
        let single = vec![Vector::new(vec![0.1, 0.2, 0.3, 0.4])];
        let trained = QuantizedSegmentVectors::from_f32_vectors(&single, dim).unwrap();
        // Synthesize an empty segment using the same params.
        let empty = QuantizedSegmentVectors {
            params: trained.params,
            dim,
            vector_count: 0,
            data: Vec::new(),
        };
        let mut buf: Vec<u8> = Vec::new();
        empty.write_to(&mut buf).unwrap();
        // 24 (header) + 4 (dim) + 4 (count) = 32 bytes total.
        assert_eq!(buf.len(), 32);

        let mut cursor = Cursor::new(&buf);
        let recovered = QuantizedSegmentVectors::read_from(&mut cursor, buf.len() as u64).unwrap();
        assert_eq!(recovered.vector_count, 0);
        assert_eq!(recovered.dim, dim);
        assert_eq!(recovered.data, Vec::<u8>::new());
    }

    #[test]
    fn roundtrip_with_distance_quantized_agrees() {
        use crate::vector::core::distance::DistanceMetric;
        use crate::vector::core::distance_quantized::{QuantizedQuery, distance_quantized};

        let dim = 32;
        let vectors = sample_vectors(6, dim);
        let seg = QuantizedSegmentVectors::from_f32_vectors(&vectors, dim).unwrap();

        let mut buf: Vec<u8> = Vec::new();
        seg.write_to(&mut buf).unwrap();
        let mut cursor = Cursor::new(&buf);
        let recovered = QuantizedSegmentVectors::read_from(&mut cursor, buf.len() as u64).unwrap();

        // Pick query = vectors[0] (so cosine distance to vectors[0]
        // should be ~ 0, modulo quantization noise).
        let query_f32 = &vectors[0].data;
        let prepared = QuantizedQuery::prepare(query_f32, &recovered.params);

        let dist_self = distance_quantized(
            DistanceMetric::Cosine,
            &prepared,
            recovered.vector_data(0),
            recovered.vector_meta(0),
        );
        assert!(
            dist_self < 0.05,
            "expected near-zero cosine self-distance after roundtrip, got {dist_self}"
        );
    }
}
