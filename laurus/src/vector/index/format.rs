//! On-disk format header shared by all quantized vector segments
//! (HNSW / Flat / IVF) introduced in Issue #481 Stage 1, extended in
//! Stage 3 with the Product Quantization variant.
//!
//! # Layout
//!
//! Every vector segment file written by laurus v0.10+ starts with a
//! fixed 16-byte header followed by a variable-size quantization
//! metadata block. The reader inspects the header first to dispatch
//! on quantization kind:
//!
//! ```text
//! offset  size  field
//! ------  ----  -------------------------------------------
//!      0     4  magic         ASCII "LVS1"
//!      4     2  version       u16 LE  (1..=3; see the feature ladder
//!                                       on CURRENT_VERSION)
//!      6     2  quant_kind    u16 LE  (1 = Scalar8Bit, 2 = PQ;
//!                                       0 reserved for "no quant")
//!      8     8  reserved      zero-padded
//!     16     -  quantization metadata (variant-specific, see below)
//!      :     -  field dictionary (version >= 3 only, Issue #633):
//!                 field_dict_count u16 LE, then per entry
//!                 name_len u16 LE + UTF-8 name.
//!                 The u16 domain bounds every allocation here.
//!      :     -  vector data
//! ```
//!
//! Per-record prefix inside the vector data: `[doc_id u64][name_len
//! u32][name UTF-8]` below version 3, `[doc_id u64][field_id u16]`
//! (index into the dictionary) from version 3 on.
//!
//! For `quant_kind = 1` (Scalar8Bit, Stage 1):
//!
//! ```text
//!     16     8  scalar_8bit_params
//!                              offset: f32 LE
//!                              scale:  f32 LE
//!     24     -  vector data (per-vector: dim bytes int8 +
//!                            QuantizedVectorMeta::SERIALIZED_SIZE)
//! ```
//!
//! For `quant_kind = 2` (Product Quantization, Stage 3):
//!
//! ```text
//!     16     2  m             u16 LE  (number of sub-vectors)
//!     18     2  k             u16 LE  (centroids per sub-vector; = 256)
//!     20     2  sub_dim       u16 LE  (dim / m)
//!     22     2  padding       u16 LE  (zero, alignment)
//!     24     -  codebook      m * k * sub_dim * 4 bytes (f32 LE,
//!                              row-major: codebook[m][k][sub_dim])
//!      :     -  vector data (per-vector: m bytes of u8 codes)
//! ```
//!
//! Endianness is **little-endian** for all multi-byte integers and
//! IEEE-754 floats.
//!
//! # Forward compatibility
//!
//! `quant_kind = 0` is reserved for a future re-introduction of an
//! unquantized variant. The reader currently rejects it with
//! [`LaurusError::IncompatibleFormat`].
//!
//! # Backward compatibility
//!
//! Pre-Stage-1 (`f32`-only) segments do **not** have this header.
//! Reading one yields an [`LaurusError::IncompatibleFormat`] with a
//! message instructing the user to rebuild the vector index.

use std::io::{Read, Write};

use crate::error::{LaurusError, Result};
use crate::vector::core::quantization::{PqParams, ScalarQuantParams};

/// 4-byte ASCII magic at offset 0 of every quantized vector segment.
pub const VECTOR_SEGMENT_MAGIC: [u8; 4] = *b"LVS1";

/// Bytes [`VectorSegmentHeader::read_from`] has consumed by the time a PQ
/// branch sizes its codebook: the 16-byte fixed header (magic + version +
/// quant_kind + reserved) plus the 8-byte `m`/`k`/`sub_dim`/padding block.
/// Subtracted from the caller-supplied `available` budget so the codebook
/// allocation bound (Issue #921) compares against the bytes truly left.
const PQ_PREFIX_SIZE: u64 = 24;

/// Lowest header format version this build can read (and the version
/// legacy Flat/IVF segments were written with).
///
/// Header versions form a monotone feature ladder:
///
/// * `1` — base format (inline per-record field names).
/// * `>= 2` — the HNSW graph block stores segment-local u32 ordinals
///   ([`VERSION_ORDINAL_GRAPH`], Issue #686; HNSW-only).
/// * `>= 3` — records reference field names through the per-segment
///   dictionary in this header ([`VERSION_FIELD_DICT`], Issue #633;
///   all index types).
pub const CURRENT_VERSION: u16 = 1;

/// Header version for HNSW segments whose graph block stores
/// segment-local u32 ordinals instead of u64 doc ids (Issue #686).
///
/// The ordinal of a vector is its rank in the ascending, deduplicated
/// record doc-id sequence — derivable for both versions because records
/// are always written sorted by doc id.
pub const VERSION_ORDINAL_GRAPH: u16 = 2;

/// Header version for segments whose per-vector records reference field
/// names through the per-segment dictionary carried by this header
/// (Issue #633, Lucene `fnm` precedent).
///
/// At this version the record prefix is `[doc_id u64][field_id u16]`
/// (see [`record_prefix_size`]) instead of the inline
/// `[doc_id u64][name_len u32][name UTF-8]`, and the dictionary block
/// follows the quantization metadata (see the module docs). Implies the
/// [`VERSION_ORDINAL_GRAPH`] graph encoding for HNSW segments.
pub const VERSION_FIELD_DICT: u16 = 3;

/// Highest header version this build can read. The reader accepts the
/// inclusive range `CURRENT_VERSION..=MAX_SUPPORTED_VERSION`.
pub const MAX_SUPPORTED_VERSION: u16 = VERSION_FIELD_DICT;

/// `quant_kind` numeric values stored in the header.
///
/// Kept as a module-level constant set rather than a Rust enum so the
/// disk format can carry "unknown but reserved" values forward without
/// requiring a Rust-side enum extension. Reader code matches on these
/// constants explicitly.
pub mod quant_kind {
    /// Reserved for future "no quantization" variant. Reader currently
    /// rejects this value (no Rust-side `None` exists in [`super::super::super::core::quantization::QuantizationMethod`]).
    pub const NONE: u16 = 0;
    /// Per-segment global affine 8-bit scalar quantization. Stage 1.
    pub const SCALAR_8BIT: u16 = 1;
    /// Product quantization (Issue #481 Stage 3). The metadata block
    /// carries `m`/`k`/`sub_dim` plus the inline codebook.
    pub const PRODUCT_QUANTIZATION: u16 = 2;
    /// FastScan Product Quantization (K=16 4-bit codes + SIMD LUT
    /// distance). Issue [#695](https://github.com/mosuka/laurus/issues/695)
    /// / part D of [#651](https://github.com/mosuka/laurus/issues/651).
    /// Same on-disk layout as [`PRODUCT_QUANTIZATION`] (m / k / sub_dim
    /// header followed by the codebook) but `k == 16` and the per-vector
    /// records are 4-bit packed instead of 8-bit AoS. Only available
    /// with the `pq-fastscan` cargo feature.
    #[cfg(feature = "pq-fastscan")]
    pub const PRODUCT_QUANTIZATION_FASTSCAN: u16 = 3;
}

/// Size in bytes of the fixed portion of the segment header (the part
/// before the quant-kind-specific metadata).
pub const FIXED_HEADER_SIZE: usize = 16;

/// Size in bytes of the [`QuantHeader::Scalar8Bit`] metadata block
/// (offset: f32 + scale: f32).
pub const SCALAR_8BIT_METADATA_SIZE: usize = 8;

/// Size in bytes of the fixed portion of the
/// [`QuantHeader::ProductQuantization`] metadata block (the `m`, `k`,
/// `sub_dim` + padding fields, before the codebook payload). The
/// codebook itself adds `m * k * sub_dim * 4` bytes.
pub const PQ_FIXED_METADATA_SIZE: usize = 8;

/// Centroids per sub-vector in the Stage 3 PQ format. Encoded in the
/// segment header so future 4-bit variants can be added without
/// breaking the layout, but the writer always emits this value today
/// (Issue #481 Stage 3 ships only 8-bit codes).
pub const PQ_CENTROIDS_PER_SUBVECTOR: u16 = 256;

/// Per-segment header read from / written to a vector segment file.
///
/// Combines the fixed 16-byte header (magic / version / quant_kind /
/// reserved) with the quantization-kind-specific metadata block and,
/// from [`VERSION_FIELD_DICT`] on, the per-segment field-name
/// dictionary. The reader and writer treat all of it as a single unit
/// so callers don't have to track parsing state.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSegmentHeader {
    /// Header format version (see the ladder on [`CURRENT_VERSION`]).
    pub version: u16,
    /// Quantization kind discriminator plus its metadata payload.
    pub quant: QuantHeader,
    /// Per-segment field-name dictionary (Issue #633).
    ///
    /// Serialized only when `version >= VERSION_FIELD_DICT`; always
    /// empty on v1/v2 headers. Records reference entries by index
    /// (`field_id u16`), assigned in first-appearance order over the
    /// segment's emitted record sequence.
    pub field_dict: Vec<String>,
}

/// Quantization-kind-specific portion of [`VectorSegmentHeader`].
///
/// New variants must update `quant_kind::*` constants and the
/// `read_from` / `write_to` dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantHeader {
    /// `quant_kind = 1` — per-segment global affine SQ (Stage 1).
    Scalar8Bit(ScalarQuantParams),
    /// `quant_kind = 2` — Product Quantization (Stage 3).
    ///
    /// Carries the PQ parameters (`m`, `k`, `sub_dim`) plus the
    /// per-segment codebook (`m * k * sub_dim` floats, row-major:
    /// `codebook[m * k * sub_dim + k * sub_dim + d]`).
    ProductQuantization {
        /// Stage 3 quantizer parameters (M / K / sub_dim).
        params: PqParams,
        /// Per-segment codebook in row-major layout. The length is
        /// always `m * k * sub_dim` floats.
        codebook: Vec<f32>,
    },
    /// `quant_kind = 3` — FastScan Product Quantization (#695 / #651).
    ///
    /// Wire-identical to [`Self::ProductQuantization`] on disk (same
    /// m / k / sub_dim header + codebook); the distinguishing factor is
    /// `k == 16` (4-bit codes) and the per-vector record format used
    /// by the reader to build a
    /// [`crate::vector::index::pq_fastscan_storage::PqFastScanPool`]
    /// instead of a [`crate::vector::index::pq_storage::PqVectorPool`].
    /// Only constructible when the `pq-fastscan` cargo feature is on.
    #[cfg(feature = "pq-fastscan")]
    ProductQuantizationFastScan {
        /// FastScan quantizer parameters (M / K=16 / sub_dim).
        params: PqParams,
        /// Per-segment codebook in row-major layout. The length is
        /// always `m * 16 * sub_dim` floats.
        codebook: Vec<f32>,
    },
}

impl QuantHeader {
    /// Numeric `quant_kind` discriminator written to the header.
    pub fn kind_code(&self) -> u16 {
        match self {
            Self::Scalar8Bit(_) => quant_kind::SCALAR_8BIT,
            Self::ProductQuantization { .. } => quant_kind::PRODUCT_QUANTIZATION,
            #[cfg(feature = "pq-fastscan")]
            Self::ProductQuantizationFastScan { .. } => quant_kind::PRODUCT_QUANTIZATION_FASTSCAN,
        }
    }

    /// Size in bytes of this variant's metadata block following the
    /// fixed 16-byte header.
    pub fn metadata_size(&self) -> usize {
        match self {
            Self::Scalar8Bit(_) => SCALAR_8BIT_METADATA_SIZE,
            Self::ProductQuantization { params, .. } => {
                PQ_FIXED_METADATA_SIZE + params.codebook_byte_size()
            }
            #[cfg(feature = "pq-fastscan")]
            Self::ProductQuantizationFastScan { params, .. } => {
                PQ_FIXED_METADATA_SIZE + params.codebook_byte_size()
            }
        }
    }
}

impl VectorSegmentHeader {
    /// Build a Stage-1 header with the given quantization params.
    pub fn scalar_8bit(params: ScalarQuantParams) -> Self {
        Self {
            version: CURRENT_VERSION,
            quant: QuantHeader::Scalar8Bit(params),
            field_dict: Vec::new(),
        }
    }

    /// Build a Stage-3 (PQ) header with the given params and codebook.
    ///
    /// The codebook must contain exactly `params.codebook_len()` floats
    /// in row-major layout (`m × k × sub_dim`).
    pub fn product_quantization(params: PqParams, codebook: Vec<f32>) -> Self {
        debug_assert_eq!(codebook.len(), params.codebook_len());
        Self {
            version: CURRENT_VERSION,
            quant: QuantHeader::ProductQuantization { params, codebook },
            field_dict: Vec::new(),
        }
    }

    /// Build a FastScan PQ header (Issue #695 / part D).
    ///
    /// `params.k` must be 16. The codebook layout is the same as
    /// [`Self::product_quantization`] (row-major `m × k × sub_dim`
    /// floats) — only the per-vector record format on disk differs
    /// (4-bit packed instead of 8-bit AoS).
    #[cfg(feature = "pq-fastscan")]
    pub fn product_quantization_fastscan(params: PqParams, codebook: Vec<f32>) -> Self {
        debug_assert_eq!(params.k, 16, "FastScan requires k == 16");
        debug_assert_eq!(codebook.len(), params.codebook_len());
        Self {
            version: CURRENT_VERSION,
            quant: QuantHeader::ProductQuantizationFastScan { params, codebook },
            field_dict: Vec::new(),
        }
    }

    /// Return `self` with the header `version` replaced.
    ///
    /// Used by the HNSW writer to stamp [`VERSION_ORDINAL_GRAPH`] on
    /// segments whose graph block stores u32 ordinals (Issue #686);
    /// Flat/IVF writers keep the [`CURRENT_VERSION`] default.
    ///
    /// # Arguments
    ///
    /// * `version` - The header version to stamp.
    ///
    /// # Returns
    ///
    /// The header value with `version` set.
    pub fn with_version(mut self, version: u16) -> Self {
        self.version = version;
        self
    }

    /// Return `self` with the field-name dictionary replaced
    /// (Issue #633).
    ///
    /// Only meaningful together with
    /// [`with_version`](Self::with_version)`(VERSION_FIELD_DICT)` (or
    /// higher) — the dictionary is serialized exclusively at those
    /// versions.
    ///
    /// # Arguments
    ///
    /// * `field_dict` - Field names in first-appearance order over the
    ///   segment's emitted record sequence.
    ///
    /// # Returns
    ///
    /// The header value with `field_dict` set.
    pub fn with_field_dict(mut self, field_dict: Vec<String>) -> Self {
        self.field_dict = field_dict;
        self
    }

    /// Serialized size of the field-dictionary block, 0 below
    /// [`VERSION_FIELD_DICT`].
    fn dict_size(&self) -> usize {
        if self.version >= VERSION_FIELD_DICT {
            2 + self
                .field_dict
                .iter()
                .map(|name| 2 + name.len())
                .sum::<usize>()
        } else {
            0
        }
    }

    /// Total serialized size: fixed header + quant-kind metadata +
    /// (from [`VERSION_FIELD_DICT`]) the field-name dictionary.
    pub fn serialized_size(&self) -> usize {
        FIXED_HEADER_SIZE + self.quant.metadata_size() + self.dict_size()
    }

    /// Write the header (fixed portion + quant metadata + versioned
    /// field dictionary) to `writer`.
    ///
    /// All multi-byte fields are little-endian. The reserved 8 bytes
    /// at offset 8 are zero-filled.
    ///
    /// # Errors
    ///
    /// * [`LaurusError::InvalidOperation`] if the dictionary exceeds
    ///   `u16::MAX` entries or an entry exceeds `u16::MAX` bytes.
    /// * [`LaurusError::Io`] for any underlying write failure.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&VECTOR_SEGMENT_MAGIC)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.quant.kind_code().to_le_bytes())?;
        writer.write_all(&[0u8; 8])?; // reserved
        match &self.quant {
            QuantHeader::Scalar8Bit(params) => {
                writer.write_all(&params.offset.to_le_bytes())?;
                writer.write_all(&params.scale.to_le_bytes())?;
            }
            QuantHeader::ProductQuantization { params, codebook } => {
                writer.write_all(&params.m.to_le_bytes())?;
                writer.write_all(&params.k.to_le_bytes())?;
                writer.write_all(&params.sub_dim.to_le_bytes())?;
                writer.write_all(&0u16.to_le_bytes())?; // padding
                debug_assert_eq!(codebook.len(), params.codebook_len());
                for &f in codebook {
                    writer.write_all(&f.to_le_bytes())?;
                }
            }
            #[cfg(feature = "pq-fastscan")]
            QuantHeader::ProductQuantizationFastScan { params, codebook } => {
                writer.write_all(&params.m.to_le_bytes())?;
                writer.write_all(&params.k.to_le_bytes())?;
                writer.write_all(&params.sub_dim.to_le_bytes())?;
                writer.write_all(&0u16.to_le_bytes())?; // padding
                debug_assert_eq!(codebook.len(), params.codebook_len());
                for &f in codebook {
                    writer.write_all(&f.to_le_bytes())?;
                }
            }
        }
        if self.version >= VERSION_FIELD_DICT {
            if self.field_dict.len() > u16::MAX as usize {
                return Err(LaurusError::InvalidOperation(format!(
                    "vector segment field dictionary has {} entries; the v3 \
                     format supports at most {}",
                    self.field_dict.len(),
                    u16::MAX
                )));
            }
            writer.write_all(&(self.field_dict.len() as u16).to_le_bytes())?;
            for name in &self.field_dict {
                if name.len() > u16::MAX as usize {
                    return Err(LaurusError::InvalidOperation(format!(
                        "vector field name is {} bytes; the v3 format supports \
                         at most {} bytes per name",
                        name.len(),
                        u16::MAX
                    )));
                }
                writer.write_all(&(name.len() as u16).to_le_bytes())?;
                writer.write_all(name.as_bytes())?;
            }
        } else {
            debug_assert!(
                self.field_dict.is_empty(),
                "field_dict is only serialized at version >= VERSION_FIELD_DICT"
            );
        }
        Ok(())
    }

    /// Read and validate the header (fixed portion + quant metadata).
    ///
    /// # Arguments
    ///
    /// * `reader` - Stream positioned at the start of the `LVS1` header.
    /// * `available` - Bytes the file can still supply from the current
    ///   position (typically `file_size - stream_position`), used to
    ///   bound the PQ codebook allocation against ground truth before
    ///   `Vec::with_capacity` (Issue #921, generalizing #806): the
    ///   header's `m`/`sub_dim` are unverified `u16`s, and a flipped
    ///   byte could otherwise request a ~4 TB allocation that aborts
    ///   the process via `handle_alloc_error`.
    ///
    /// # Errors
    ///
    /// * [`LaurusError::IncompatibleFormat`] if the magic does not
    ///   match `LVS1`. Most commonly this means the segment was written
    ///   by a pre-quantization (f32-only) build of laurus and must be
    ///   rebuilt — Issue #481 Stage 1 is a deliberate format break.
    /// * [`LaurusError::IncompatibleFormat`] if the version is outside
    ///   the supported `CURRENT_VERSION..=MAX_SUPPORTED_VERSION` range
    ///   or if `quant_kind` is the reserved value 0.
    /// * [`LaurusError::Index`] if a PQ / PQ FastScan codebook declares
    ///   more entries than `available` bytes can physically hold — the
    ///   segment is corrupted (Issue #921).
    /// * [`LaurusError::Io`] for any underlying read failure.
    pub fn read_from<R: Read>(reader: &mut R, available: u64) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != VECTOR_SEGMENT_MAGIC {
            return Err(LaurusError::IncompatibleFormat(format!(
                "expected vector segment magic {:?} (\"LVS1\"), found {:?}. \
                 Pre-quantization (f32) segments must be rebuilt — \
                 Issue #481 Stage 1 introduced int8 scalar quantization \
                 with a new on-disk format.",
                VECTOR_SEGMENT_MAGIC, magic
            )));
        }

        let mut version_bytes = [0u8; 2];
        reader.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);
        if !(CURRENT_VERSION..=MAX_SUPPORTED_VERSION).contains(&version) {
            return Err(LaurusError::IncompatibleFormat(format!(
                "unsupported vector segment header version {version} \
                 (this build supports {CURRENT_VERSION}..={MAX_SUPPORTED_VERSION})"
            )));
        }

        let mut kind_bytes = [0u8; 2];
        reader.read_exact(&mut kind_bytes)?;
        let kind = u16::from_le_bytes(kind_bytes);

        let mut reserved = [0u8; 8];
        reader.read_exact(&mut reserved)?;

        let quant = match kind {
            quant_kind::SCALAR_8BIT => {
                let mut offset_bytes = [0u8; 4];
                let mut scale_bytes = [0u8; 4];
                reader.read_exact(&mut offset_bytes)?;
                reader.read_exact(&mut scale_bytes)?;
                QuantHeader::Scalar8Bit(ScalarQuantParams {
                    offset: f32::from_le_bytes(offset_bytes),
                    scale: f32::from_le_bytes(scale_bytes),
                })
            }
            quant_kind::PRODUCT_QUANTIZATION => {
                let mut buf2 = [0u8; 2];
                reader.read_exact(&mut buf2)?;
                let m = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?;
                let k = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?;
                let sub_dim = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?; // padding
                let params = PqParams::new(m, k, sub_dim).map_err(|e| {
                    LaurusError::IncompatibleFormat(format!("invalid PQ params: {e}"))
                })?;
                let codebook_len = params.codebook_len();
                // Issue #921: `m`/`sub_dim` are unverified u16s straight
                // off disk; bound the codebook allocation against the
                // bytes the file actually has left (the fixed prefix +
                // PQ params block consumed so far is PQ_PREFIX_SIZE)
                // before reserving anything.
                crate::util::alloc_bounds::checked_capacity(
                    codebook_len,
                    4,
                    available.saturating_sub(PQ_PREFIX_SIZE),
                    "vector segment PQ codebook length",
                )?;
                let mut codebook = Vec::with_capacity(codebook_len);
                let mut fbuf = [0u8; 4];
                for _ in 0..codebook_len {
                    reader.read_exact(&mut fbuf)?;
                    codebook.push(f32::from_le_bytes(fbuf));
                }
                QuantHeader::ProductQuantization { params, codebook }
            }
            #[cfg(feature = "pq-fastscan")]
            quant_kind::PRODUCT_QUANTIZATION_FASTSCAN => {
                let mut buf2 = [0u8; 2];
                reader.read_exact(&mut buf2)?;
                let m = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?;
                let k = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?;
                let sub_dim = u16::from_le_bytes(buf2);
                reader.read_exact(&mut buf2)?; // padding
                let params = PqParams::new(m, k, sub_dim).map_err(|e| {
                    LaurusError::IncompatibleFormat(format!("invalid PQ FastScan params: {e}"))
                })?;
                if params.k != 16 {
                    return Err(LaurusError::IncompatibleFormat(format!(
                        "PQ FastScan segment must declare k == 16, got k = {}",
                        params.k
                    )));
                }
                let codebook_len = params.codebook_len();
                // Issue #921: same allocation bound as the PQ branch above.
                crate::util::alloc_bounds::checked_capacity(
                    codebook_len,
                    4,
                    available.saturating_sub(PQ_PREFIX_SIZE),
                    "vector segment PQ FastScan codebook length",
                )?;
                let mut codebook = Vec::with_capacity(codebook_len);
                let mut fbuf = [0u8; 4];
                for _ in 0..codebook_len {
                    reader.read_exact(&mut fbuf)?;
                    codebook.push(f32::from_le_bytes(fbuf));
                }
                QuantHeader::ProductQuantizationFastScan { params, codebook }
            }
            quant_kind::NONE => {
                return Err(LaurusError::IncompatibleFormat(
                    "vector segment header reports quant_kind = 0 (no quantization), \
                     but the Rust-side QuantizationMethod has no None variant; \
                     this build cannot read unquantized segments"
                        .to_string(),
                ));
            }
            other => {
                return Err(LaurusError::IncompatibleFormat(format!(
                    "unknown vector segment quant_kind = {other}"
                )));
            }
        };

        // Field-name dictionary (Issue #633), present from v3 on. The
        // u16 length domain itself bounds every allocation here
        // (≤ 65535 entries of ≤ 64 KiB names), preserving the #806
        // no-huge-allocation property without file-size plumbing;
        // `read_exact` fails at EOF before unbounded accumulation.
        let field_dict = if version >= VERSION_FIELD_DICT {
            let mut count_bytes = [0u8; 2];
            reader.read_exact(&mut count_bytes)?;
            let count = u16::from_le_bytes(count_bytes) as usize;
            let mut dict = Vec::with_capacity(count);
            for _ in 0..count {
                let mut len_bytes = [0u8; 2];
                reader.read_exact(&mut len_bytes)?;
                let len = u16::from_le_bytes(len_bytes) as usize;
                let mut name_bytes = vec![0u8; len];
                reader.read_exact(&mut name_bytes)?;
                let name = String::from_utf8(name_bytes).map_err(|e| {
                    LaurusError::IncompatibleFormat(format!(
                        "invalid UTF-8 in vector segment field dictionary: {e}"
                    ))
                })?;
                dict.push(name);
            }
            dict
        } else {
            Vec::new()
        };

        Ok(Self {
            version,
            quant,
            field_dict,
        })
    }

    /// Read one record's field name according to this header's version
    /// (Issue #633).
    ///
    /// * `version >= VERSION_FIELD_DICT`: reads a `field_id u16` and
    ///   resolves it against [`field_dict`](Self::field_dict).
    /// * older versions: reads the inline `name_len u32 + UTF-8 name`.
    ///
    /// # Arguments
    ///
    /// * `reader` - Stream positioned right after the record's doc id.
    /// * `records_remaining` - Bytes left for the record section, used
    ///   to bound the inline-name allocation (Issue #806; ignored on
    ///   the v3 path, whose u16 id needs no bounding).
    /// * `what` - Label for corruption error messages (e.g.
    ///   `"hnsw field_name_len"`).
    ///
    /// # Returns
    ///
    /// The record's field name, or an error when the id is out of
    /// dictionary range / the inline name is oversized or not UTF-8.
    pub(crate) fn read_record_field<R: Read>(
        &self,
        reader: &mut R,
        records_remaining: u64,
        what: &str,
    ) -> Result<String> {
        if self.version >= VERSION_FIELD_DICT {
            let mut id_bytes = [0u8; 2];
            reader.read_exact(&mut id_bytes)?;
            let field_id = u16::from_le_bytes(id_bytes);
            self.field_dict
                .get(field_id as usize)
                .cloned()
                .ok_or_else(|| {
                    LaurusError::index(format!(
                        "vector segment corrupt: record field_id {field_id} out \
                         of dictionary range ({} entries)",
                        self.field_dict.len()
                    ))
                })
        } else {
            use crate::util::alloc_bounds::checked_len;

            let mut len_bytes = [0u8; 4];
            reader.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            checked_len(len, records_remaining, what)?;
            let mut name_bytes = vec![0u8; len];
            reader.read_exact(&mut name_bytes)?;
            String::from_utf8(name_bytes).map_err(|e| {
                LaurusError::InvalidOperation(format!("Invalid UTF-8 in field name: {}", e))
            })
        }
    }
}

/// Per-segment field-name interner for reader load paths (Issue #633,
/// PR-B).
///
/// Readers parse records into `(doc_id, u16)` pairs plus one shared
/// dictionary instead of one heap `String` per record:
///
/// * v3 segments: the dictionary comes fixed from the header; record
///   ids are validated against it.
/// * v1/v2 segments: the dictionary is synthesized on the fly in
///   first-appearance order, so legacy files get the same interned
///   in-memory representation.
pub(crate) struct FieldInterner {
    names: Vec<std::sync::Arc<str>>,
    /// Whether the dictionary is fixed by a v3 header (no growth).
    fixed: bool,
}

impl FieldInterner {
    /// Build an interner for the given parsed header.
    ///
    /// # Arguments
    ///
    /// * `header` - The segment's parsed LVS header.
    ///
    /// # Returns
    ///
    /// A fixed interner seeded from the v3 dictionary, or an empty
    /// growable one for v1/v2 segments.
    pub(crate) fn from_header(header: &VectorSegmentHeader) -> Self {
        if header.version >= VERSION_FIELD_DICT {
            Self {
                names: header
                    .field_dict
                    .iter()
                    .map(|s| std::sync::Arc::from(s.as_str()))
                    .collect(),
                fixed: true,
            }
        } else {
            Self {
                names: Vec::new(),
                fixed: false,
            }
        }
    }

    /// Read one record's field reference and return its interned id.
    ///
    /// v3: reads the `field_id u16` and validates it against the fixed
    /// dictionary. v1/v2: reads the inline name and interns it
    /// (first-appearance order).
    ///
    /// # Arguments
    ///
    /// * `header` - The segment's parsed header (drives the version).
    /// * `reader` - Stream positioned right after the record's doc id.
    /// * `records_remaining` - Bound for the inline-name allocation.
    /// * `what` - Label for corruption error messages.
    ///
    /// # Returns
    ///
    /// The record's field id within [`Self::dict`].
    pub(crate) fn read_record_field_id<R: Read>(
        &mut self,
        header: &VectorSegmentHeader,
        reader: &mut R,
        records_remaining: u64,
        what: &str,
    ) -> Result<u16> {
        if self.fixed {
            let mut id_bytes = [0u8; 2];
            reader.read_exact(&mut id_bytes)?;
            let field_id = u16::from_le_bytes(id_bytes);
            if field_id as usize >= self.names.len() {
                return Err(LaurusError::index(format!(
                    "vector segment corrupt: record field_id {field_id} out of \
                     dictionary range ({} entries)",
                    self.names.len()
                )));
            }
            Ok(field_id)
        } else {
            let name = header.read_record_field(reader, records_remaining, what)?;
            if let Some(pos) = self.names.iter().position(|n| **n == *name) {
                return Ok(pos as u16);
            }
            if self.names.len() >= u16::MAX as usize {
                return Err(LaurusError::index(format!(
                    "vector segment has more than {} distinct field names",
                    u16::MAX
                )));
            }
            self.names.push(std::sync::Arc::from(name.as_str()));
            Ok((self.names.len() - 1) as u16)
        }
    }

    /// Borrow the interned name for `fid` (valid ids only — callers
    /// pass ids this interner just produced).
    pub(crate) fn name(&self, fid: u16) -> &std::sync::Arc<str> {
        &self.names[fid as usize]
    }

    /// Finalize into the per-segment shared dictionary.
    pub(crate) fn into_dict(self) -> std::sync::Arc<[std::sync::Arc<str>]> {
        std::sync::Arc::from(self.names)
    }
}

/// Resolve a field name to its id in a per-segment dictionary by linear
/// scan (Issue #633 PR-B).
///
/// Dictionaries hold 1–3 entries in practice, so a scan beats hashing
/// and allocates nothing — this is the once-per-call replacement for
/// the former per-call `field_name.to_string()` key materialization.
///
/// # Arguments
///
/// * `dict` - The segment's field dictionary.
/// * `field_name` - The name to resolve.
///
/// # Returns
///
/// The field id, or `None` when the segment has no such field.
#[inline]
pub(crate) fn resolve_field_id(dict: &[std::sync::Arc<str>], field_name: &str) -> Option<u16> {
    dict.iter()
        .position(|n| **n == *field_name)
        .map(|pos| pos as u16)
}

/// Size in bytes of the per-record prefix (`doc_id` + field reference)
/// for the given header version (Issue #633).
///
/// * `version >= VERSION_FIELD_DICT`: `doc_id u64 + field_id u16` = 10.
/// * older versions: `doc_id u64 + name_len u32` = 12 (plus the
///   variable inline name bytes, which callers account separately).
///
/// # Arguments
///
/// * `version` - The segment header version.
///
/// # Returns
///
/// The fixed prefix size in bytes.
pub(crate) fn record_prefix_size(version: u16) -> u64 {
    if version >= VERSION_FIELD_DICT {
        10
    } else {
        12
    }
}

/// Build a per-segment field-name dictionary in first-appearance order
/// (Issue #633).
///
/// # Arguments
///
/// * `names` - The record field names in the exact order the writer
///   will emit the records.
///
/// # Returns
///
/// `(dictionary, name → field_id map)` for the writer's emit loop, or
/// an error if more than `u16::MAX` distinct names are present.
pub(crate) fn build_field_dict<'a>(
    names: impl Iterator<Item = &'a str>,
) -> Result<(Vec<String>, std::collections::HashMap<String, u16>)> {
    let mut dict: Vec<String> = Vec::new();
    let mut ids: std::collections::HashMap<String, u16> = std::collections::HashMap::new();
    for name in names {
        if !ids.contains_key(name) {
            if dict.len() >= u16::MAX as usize {
                return Err(LaurusError::InvalidOperation(format!(
                    "vector segment has more than {} distinct field names; \
                     the v3 format supports at most that many",
                    u16::MAX
                )));
            }
            ids.insert(name.to_string(), dict.len() as u16);
            dict.push(name.to_string());
        }
    }
    Ok((dict, ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_params() -> ScalarQuantParams {
        ScalarQuantParams {
            offset: -1.5,
            scale: 3.0 / 255.0,
        }
    }

    #[test]
    fn roundtrip_scalar_8bit_header() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params());
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), header.serialized_size());
        assert_eq!(buf.len(), FIXED_HEADER_SIZE + SCALAR_8BIT_METADATA_SIZE);

        let mut cursor = Cursor::new(&buf);
        let parsed = VectorSegmentHeader::read_from(&mut cursor, buf.len() as u64).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn roundtrip_v2_ordinal_graph_header() {
        let header =
            VectorSegmentHeader::scalar_8bit(sample_params()).with_version(VERSION_ORDINAL_GRAPH);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(
            u16::from_le_bytes([buf[4], buf[5]]),
            VERSION_ORDINAL_GRAPH,
            "with_version must stamp the version bytes at offset 4"
        );

        let mut cursor = Cursor::new(&buf);
        let parsed = VectorSegmentHeader::read_from(&mut cursor, buf.len() as u64).unwrap();
        assert_eq!(parsed.version, VERSION_ORDINAL_GRAPH);
        assert_eq!(parsed, header);
    }

    #[test]
    fn rejects_header_version_above_max_supported() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params())
            .with_version(MAX_SUPPORTED_VERSION + 1);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = VectorSegmentHeader::read_from(&mut cursor, buf.len() as u64).unwrap_err();
        assert!(
            err.to_string().contains("unsupported vector segment"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn roundtrip_v3_header_with_field_dict() {
        for dict in [
            Vec::new(),
            vec!["embedding".to_string()],
            vec![
                "a".to_string(),
                "b".to_string(),
                "long_field_name".to_string(),
            ],
        ] {
            let header = VectorSegmentHeader::scalar_8bit(sample_params())
                .with_version(VERSION_FIELD_DICT)
                .with_field_dict(dict.clone());
            let mut buf: Vec<u8> = Vec::new();
            header.write_to(&mut buf).unwrap();
            assert_eq!(
                buf.len(),
                header.serialized_size(),
                "serialized_size must match written bytes for dict {dict:?}"
            );

            let mut cursor = Cursor::new(&buf);
            let parsed = VectorSegmentHeader::read_from(&mut cursor, buf.len() as u64).unwrap();
            assert_eq!(parsed.field_dict, dict);
            assert_eq!(parsed, header);
        }
    }

    #[test]
    fn v1_header_serialization_is_unchanged_by_the_dict_field() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params());
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), 24, "v1 SQ header must stay 24 bytes");
        assert_eq!(header.serialized_size(), 24);
        let parsed =
            VectorSegmentHeader::read_from(&mut Cursor::new(&buf), buf.len() as u64).unwrap();
        assert!(parsed.field_dict.is_empty());
    }

    #[test]
    fn truncated_field_dict_returns_io_error() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params())
            .with_version(VERSION_FIELD_DICT)
            .with_field_dict(vec!["embedding".to_string()]);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        buf.truncate(buf.len() - 3); // cut into the dict entry

        let err =
            VectorSegmentHeader::read_from(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        assert!(matches!(err, LaurusError::Io(_)), "got: {err:?}");
    }

    #[test]
    fn write_rejects_field_name_longer_than_u16_max() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params())
            .with_version(VERSION_FIELD_DICT)
            .with_field_dict(vec!["x".repeat(u16::MAX as usize + 1)]);
        let mut buf: Vec<u8> = Vec::new();
        let err = header.write_to(&mut buf).unwrap_err();
        assert!(err.to_string().contains("at most"), "got: {err}");
    }

    #[test]
    fn read_record_field_resolves_v3_ids_and_rejects_out_of_range() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params())
            .with_version(VERSION_FIELD_DICT)
            .with_field_dict(vec!["a".to_string(), "b".to_string()]);

        let mut ok = Cursor::new(1u16.to_le_bytes().to_vec());
        assert_eq!(header.read_record_field(&mut ok, 0, "test").unwrap(), "b");

        let mut bad = Cursor::new(7u16.to_le_bytes().to_vec());
        let err = header.read_record_field(&mut bad, 0, "test").unwrap_err();
        assert!(
            err.to_string().contains("out of dictionary range"),
            "got: {err}"
        );
    }

    #[test]
    fn read_record_field_parses_inline_names_below_v3() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params());
        let mut bytes = (5u32).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"field");
        let mut cursor = Cursor::new(bytes);
        assert_eq!(
            header.read_record_field(&mut cursor, 64, "test").unwrap(),
            "field"
        );
    }

    #[test]
    fn interner_synthesizes_dict_for_legacy_versions() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params());
        let mut interner = FieldInterner::from_header(&header);

        let mut rec = |name: &str| {
            let mut bytes = (name.len() as u32).to_le_bytes().to_vec();
            bytes.extend_from_slice(name.as_bytes());
            let mut cursor = Cursor::new(bytes);
            interner
                .read_record_field_id(&header, &mut cursor, 64, "test")
                .unwrap()
        };
        assert_eq!(rec("b"), 0);
        assert_eq!(rec("a"), 1);
        assert_eq!(rec("b"), 0, "repeat name must intern to the same id");

        let dict = interner.into_dict();
        assert_eq!(dict.len(), 2);
        assert_eq!(&*dict[0], "b");
        assert_eq!(&*dict[1], "a");
    }

    #[test]
    fn interner_uses_fixed_dict_for_v3_and_rejects_out_of_range() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params())
            .with_version(VERSION_FIELD_DICT)
            .with_field_dict(vec!["f".to_string()]);
        let mut interner = FieldInterner::from_header(&header);

        let mut ok = Cursor::new(0u16.to_le_bytes().to_vec());
        assert_eq!(
            interner
                .read_record_field_id(&header, &mut ok, 0, "test")
                .unwrap(),
            0
        );
        let mut bad = Cursor::new(3u16.to_le_bytes().to_vec());
        let err = interner
            .read_record_field_id(&header, &mut bad, 0, "test")
            .unwrap_err();
        assert!(err.to_string().contains("out of"), "got: {err}");
    }

    #[test]
    fn resolve_field_id_scans_the_dict() {
        let dict: Vec<std::sync::Arc<str>> =
            vec![std::sync::Arc::from("a"), std::sync::Arc::from("b")];
        assert_eq!(resolve_field_id(&dict, "b"), Some(1));
        assert_eq!(resolve_field_id(&dict, "missing"), None);
    }

    #[test]
    fn record_prefix_size_matches_the_version_ladder() {
        assert_eq!(record_prefix_size(CURRENT_VERSION), 12);
        assert_eq!(record_prefix_size(VERSION_ORDINAL_GRAPH), 12);
        assert_eq!(record_prefix_size(VERSION_FIELD_DICT), 10);
    }

    #[test]
    fn build_field_dict_assigns_first_appearance_order() {
        let names = ["b", "a", "b", "c", "a"];
        let (dict, ids) = build_field_dict(names.into_iter()).unwrap();
        assert_eq!(
            dict,
            vec!["b".to_string(), "a".to_string(), "c".to_string()]
        );
        assert_eq!(ids["b"], 0);
        assert_eq!(ids["a"], 1);
        assert_eq!(ids["c"], 2);
    }

    #[test]
    fn header_starts_with_magic_then_version_then_kind() {
        let header = VectorSegmentHeader::scalar_8bit(sample_params());
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(&buf[0..4], b"LVS1");
        assert_eq!(u16::from_le_bytes([buf[4], buf[5]]), CURRENT_VERSION);
        assert_eq!(
            u16::from_le_bytes([buf[6], buf[7]]),
            quant_kind::SCALAR_8BIT
        );
        // reserved bytes are zero
        assert_eq!(&buf[8..16], &[0u8; 8]);
    }

    #[test]
    fn missing_magic_returns_incompatible_format() {
        // Simulate a pre-Stage-1 f32 segment: starts with raw float bytes.
        let f32_bytes = [
            0xCD, 0xCC, 0x4C, 0x3F, // 0.8 in LE f32
            0x00, 0x00, 0x80, 0x3F, // 1.0
            0x00, 0x00, 0x00, 0x40, // 2.0
            0x00, 0x00, 0x40, 0x40, // 3.0
        ];
        let mut cursor = Cursor::new(&f32_bytes[..]);
        let err = VectorSegmentHeader::read_from(&mut cursor, f32_bytes.len() as u64).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("LVS1"), "message should mention LVS1");
                assert!(
                    msg.contains("rebuilt"),
                    "message should instruct to rebuild"
                );
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_version_returns_incompatible_format() {
        // Same magic but bumped version.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LVS1");
        buf.extend_from_slice(&99u16.to_le_bytes()); // unknown version
        buf.extend_from_slice(&quant_kind::SCALAR_8BIT.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&0.0_f32.to_le_bytes());
        buf.extend_from_slice(&1.0_f32.to_le_bytes());

        let err =
            VectorSegmentHeader::read_from(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("99"), "message should mention the version");
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn quant_kind_zero_is_rejected_for_now() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LVS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&quant_kind::NONE.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);

        let err =
            VectorSegmentHeader::read_from(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        assert!(matches!(err, LaurusError::IncompatibleFormat(_)));
    }

    /// Issue #921: a corrupted PQ header whose `m`/`sub_dim` declare a
    /// codebook far larger than the file must be rejected as corruption
    /// *before* the codebook `Vec` is reserved — previously this
    /// requested a ~4 TB allocation and aborted the process via
    /// `handle_alloc_error` instead of returning a clean error.
    #[test]
    fn oversized_pq_codebook_declaration_is_rejected_cleanly() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LVS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&quant_kind::PRODUCT_QUANTIZATION.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        // m = 65535, k = 256, sub_dim = 65535 → ~1.1e12 codebook f32s
        // (~4 TB) declared by a 24-byte file.
        buf.extend_from_slice(&u16::MAX.to_le_bytes());
        buf.extend_from_slice(&256u16.to_le_bytes());
        buf.extend_from_slice(&u16::MAX.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());

        let err =
            VectorSegmentHeader::read_from(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        match err {
            LaurusError::Index(msg) => {
                assert!(msg.contains("corrupted"), "corruption must be named: {msg}");
            }
            other => panic!("expected LaurusError::Index, got {other:?}"),
        }
    }

    /// Issue #921: the FastScan PQ branch shares the same codebook
    /// allocation bound as the 8-bit PQ branch above.
    #[cfg(feature = "pq-fastscan")]
    #[test]
    fn oversized_pq_fastscan_codebook_declaration_is_rejected_cleanly() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LVS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&quant_kind::PRODUCT_QUANTIZATION_FASTSCAN.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        // k must be 16 for FastScan; m/sub_dim maxed out still declare
        // a ~256 GB codebook this 24-byte file cannot hold.
        buf.extend_from_slice(&u16::MAX.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(&u16::MAX.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());

        let err =
            VectorSegmentHeader::read_from(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        match err {
            LaurusError::Index(msg) => {
                assert!(msg.contains("corrupted"), "corruption must be named: {msg}");
            }
            other => panic!("expected LaurusError::Index, got {other:?}"),
        }
    }

    fn sample_pq_params() -> PqParams {
        PqParams::new(4, 256, 2).expect("valid PQ params")
    }

    fn sample_pq_codebook(params: &PqParams) -> Vec<f32> {
        // Deterministic codebook so the roundtrip test compares
        // every byte: codebook[m_idx][k_idx][d] = m_idx * 100 +
        // k_idx + d as f32.
        let mut cb = Vec::with_capacity(params.codebook_len());
        for m in 0..params.m as usize {
            for k in 0..params.k as usize {
                for d in 0..params.sub_dim as usize {
                    cb.push((m * 100 + k) as f32 + d as f32 * 0.01);
                }
            }
        }
        cb
    }

    #[test]
    fn roundtrip_product_quantization_header() {
        let params = sample_pq_params();
        let codebook = sample_pq_codebook(&params);
        let header = VectorSegmentHeader::product_quantization(params, codebook);

        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), header.serialized_size());
        // Fixed header (16) + PQ fixed metadata (8) + codebook
        // (m * k * sub_dim * 4) = 16 + 8 + 4 * 256 * 2 * 4 = 24 +
        // 8192.
        assert_eq!(
            buf.len(),
            FIXED_HEADER_SIZE + PQ_FIXED_METADATA_SIZE + 4 * 256 * 2 * 4
        );

        let parsed =
            VectorSegmentHeader::read_from(&mut Cursor::new(&buf), buf.len() as u64).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn pq_header_serialised_starts_with_magic_and_kind_two() {
        let params = sample_pq_params();
        let codebook = sample_pq_codebook(&params);
        let header = VectorSegmentHeader::product_quantization(params, codebook);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(&buf[0..4], b"LVS1");
        assert_eq!(u16::from_le_bytes([buf[4], buf[5]]), CURRENT_VERSION);
        assert_eq!(
            u16::from_le_bytes([buf[6], buf[7]]),
            quant_kind::PRODUCT_QUANTIZATION
        );
        // PQ params start at offset 16.
        assert_eq!(u16::from_le_bytes([buf[16], buf[17]]), 4); // m
        assert_eq!(u16::from_le_bytes([buf[18], buf[19]]), 256); // k
        assert_eq!(u16::from_le_bytes([buf[20], buf[21]]), 2); // sub_dim
        // padding zero
        assert_eq!(u16::from_le_bytes([buf[22], buf[23]]), 0);
    }

    #[test]
    fn pq_header_metadata_size_matches_codebook() {
        let params = sample_pq_params();
        let header = QuantHeader::ProductQuantization {
            params,
            codebook: vec![0.0; params.codebook_len()],
        };
        assert_eq!(
            header.metadata_size(),
            PQ_FIXED_METADATA_SIZE + params.codebook_byte_size()
        );
    }

    #[test]
    fn pq_header_rejects_invalid_params() {
        // Construct a buffer with m = 0 (invalid).
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LVS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&quant_kind::PRODUCT_QUANTIZATION.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&0u16.to_le_bytes()); // m = 0
        buf.extend_from_slice(&256u16.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        let err =
            VectorSegmentHeader::read_from(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        assert!(matches!(err, LaurusError::IncompatibleFormat(_)));
    }

    #[test]
    fn unknown_quant_kind_is_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LVS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&999u16.to_le_bytes()); // unknown
        buf.extend_from_slice(&[0u8; 8]);

        let err =
            VectorSegmentHeader::read_from(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("999"));
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn truncated_header_returns_io_error() {
        // Magic only, no version → reader hits EOF on the version read.
        let buf = b"LVS1".to_vec();
        let err =
            VectorSegmentHeader::read_from(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        assert!(matches!(err, LaurusError::Io(_)));
    }

    #[test]
    fn serialized_size_constants_are_consistent() {
        let h = VectorSegmentHeader::scalar_8bit(sample_params());
        assert_eq!(h.serialized_size(), 24);
        assert_eq!(FIXED_HEADER_SIZE + SCALAR_8BIT_METADATA_SIZE, 24);
    }

    #[test]
    fn fixed_header_size_is_sixteen() {
        assert_eq!(FIXED_HEADER_SIZE, 16);
    }

    #[test]
    fn quant_header_kind_code_matches_constants() {
        let h = QuantHeader::Scalar8Bit(sample_params());
        assert_eq!(h.kind_code(), quant_kind::SCALAR_8BIT);
        assert_eq!(h.metadata_size(), SCALAR_8BIT_METADATA_SIZE);
    }
}
