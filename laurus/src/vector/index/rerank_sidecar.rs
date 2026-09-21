//! On-disk format for the rerank sidecar (Issue #481 Stage 2).
//!
//! Stage 1 stored vectors as int8 only (LVS1 segment, see
//! [`super::format`]). Stage 2 adds an *optional* per-field sidecar
//! that carries the original full-precision vectors so the searcher
//! can do a wide candidate fetch over int8 (cheap) and then rescore
//! the top `top_k * rerank_factor` candidates against the original
//! values (accurate).
//!
//! The sidecar lives in a separate file (`*.<index_ext>.f32` for
//! [`crate::vector::core::rerank::RerankStorageKind::F32`]) and is
//! produced/consumed in lockstep with the LVS1 segment: vectors are
//! written in the same `(doc_id, field_name)` order as the matching
//! LVS1 segment so a (sidecar position) → (LVS1 position) mapping is
//! the identity. This keeps the sidecar self-contained — the LVS1
//! segment never references it.
//!
//! # On-disk layout
//!
//! ```text
//! offset  size  field
//! ------  ----  -------------------------------------------
//!      0     4  magic         ASCII "LRS1"
//!      4     2  version       u16 LE  (current = 1)
//!      6     2  storage_kind  u16 LE  (1 = F32; 0 reserved for "no
//!                                       sidecar"; 2.. reserved for
//!                                       future bf16 / fp16)
//!      8     8  reserved      zero-padded
//!     16     4  dim           u32 LE
//!     20     4  vector_count  u32 LE
//!     24     -  payload       vector_count × dim × bytes_per_element
//!                              of the configured RerankStorageKind
//!    end     8  footer        magic u32 LE ("LRC1") + CRC-32 u32 LE
//!                              over header + payload (Issue #788)
//! ```
//!
//! All multi-byte integers and IEEE-754 floats are little-endian.
//!
//! # Checksum footer (Issue #788)
//!
//! New sidecars end with an 8-byte footer
//! `[RERANK_SIDECAR_FOOTER_MAGIC u32 LE][crc-32 u32 LE]` whose CRC-32
//! covers every preceding byte (header + payload), mirroring the
//! `.hnsw` footer added by Issue #786. Because the header fully
//! determines the content length, [`read_sidecar`] *detects* the footer
//! by the number of bytes remaining after the payload — footer
//! detection itself needs no file-size heuristics: zero remaining bytes
//! is a legacy footer-less sidecar (verification is skipped), exactly
//! eight is a footer (verified), anything else is corruption.
//!
//! Separately, the on-disk file size *is* used up front to bound the
//! payload allocation before the header is trusted (Issue #791); see
//! [`read_sidecar`]. That bound only caps how many bytes may be
//! allocated — it does not participate in the footer classification
//! above.
//!
//! # Backward compatibility
//!
//! Pre-Stage-2 segments do not have a sidecar file at all. Readers
//! must treat "sidecar file missing" as Stage 1 and fall back to the
//! int8 LVS1 path. This module never sees that case — it only knows
//! how to read a sidecar that exists.
//!
//! Sidecars written before Issue #788 lack the checksum footer and
//! still load (the footer is optional on read). Conversely, readers
//! that predate the footer stop at the payload end and never see the
//! trailing bytes, so new files remain readable by old code; the
//! header version stays at 1.
//!
//! One consequence of the optional footer: a footer-carrying sidecar
//! whose trailing 8 bytes are lost is byte-identical to a legacy file
//! and loads with verification skipped (the payload itself is still
//! bit-exact in that case — only the integrity guarantee is
//! downgraded). Partial footer loss (1–7 trailing bytes) is detected.
//! The same limitation applies to the `.hnsw` footer from Issue #786;
//! atomic temp+rename writes (Issue #784) make such tail loss
//! improbable in practice.

use std::io::{Read, Write};

use crate::error::{LaurusError, Result};
use crate::vector::core::rerank::RerankStorageKind;

/// 4-byte ASCII magic at offset 0 of every rerank sidecar file.
pub const RERANK_SIDECAR_MAGIC: [u8; 4] = *b"LRS1";

/// Current sidecar header version. Reader rejects anything else.
pub const CURRENT_VERSION: u16 = 1;

/// Size in bytes of the fixed magic+version+kind+reserved portion of
/// the sidecar header (the part shared with [`super::format`]).
pub const FIXED_HEADER_SIZE: usize = 16;

/// Size in bytes of the dim + vector_count fields that follow the
/// fixed header.
pub const DIM_AND_COUNT_SIZE: usize = 8;

/// Total serialized size of a [`RerankSidecarHeader`]: fixed header
/// plus the dim/count pair.
pub const HEADER_SIZE: usize = FIXED_HEADER_SIZE + DIM_AND_COUNT_SIZE;

/// Magic marker for the CRC-32 footer appended to a rerank sidecar
/// (Issue #788). Spelled "LRC1" (Laurus Rerank Checksum v1); distinct
/// from both the sidecar header magic `LRS1` and the `.hnsw` footer
/// magic `LVC1` so a footer can never be mistaken for another format.
pub const RERANK_SIDECAR_FOOTER_MAGIC: u32 = 0x4C52_4331; // "LRC1"

/// Byte length of the sidecar CRC footer: `magic u32` + `crc-32 u32`.
pub const FOOTER_SIZE: usize = 8;

/// Reserved `storage_kind` numeric values stored in the sidecar header.
pub mod storage_kind {
    /// Reserved for "no rerank storage". Sidecars with this value
    /// must not be written; callers expressing "no sidecar" should
    /// simply omit the file. Reader rejects this value.
    pub const NONE: u16 = 0;
    /// Full IEEE-754 single-precision floats. Stage 2.
    pub const F32: u16 = 1;
}

/// Header parsed from / written to a rerank sidecar file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankSidecarHeader {
    /// Header format version. Always [`CURRENT_VERSION`] when written
    /// by this build.
    pub version: u16,
    /// On-disk encoding of each stored element.
    pub storage_kind: RerankStorageKind,
    /// Vector dimension (must match the matching LVS1 segment).
    pub dim: u32,
    /// Number of vectors stored in this sidecar (must match the
    /// matching LVS1 segment's vector count).
    pub vector_count: u32,
}

impl RerankSidecarHeader {
    /// Build a header for a Stage 2 sidecar with the given parameters.
    pub fn new(storage_kind: RerankStorageKind, dim: u32, vector_count: u32) -> Self {
        Self {
            version: CURRENT_VERSION,
            storage_kind,
            dim,
            vector_count,
        }
    }

    /// Bytes occupied by the payload that follows the header.
    ///
    /// Computed with checked arithmetic (Issue #791): the
    /// `vector_count * dim * bytes_per_element` product is evaluated in
    /// `u64` with `checked_mul` so a corrupt or hostile header (whose
    /// `dim` / `vector_count` an attacker controls) can never wrap in a
    /// release build, and the result is range-checked into `usize` so it
    /// stays valid on 32-bit targets (e.g. wasm32). Both failures surface
    /// as a clean error instead of an undersized buffer or a panic.
    ///
    /// # Returns
    ///
    /// The payload length in bytes.
    ///
    /// # Errors
    ///
    /// [`LaurusError::Index`] if the product overflows `u64`, or if the
    /// resulting byte count does not fit in `usize` on this platform.
    pub fn payload_size(&self) -> Result<usize> {
        let bytes = (self.vector_count as u64)
            .checked_mul(self.dim as u64)
            .and_then(|v| v.checked_mul(self.storage_kind.bytes_per_element() as u64))
            .ok_or_else(|| {
                LaurusError::index(format!(
                    "rerank sidecar payload size overflow: vector_count={} * dim={} * \
                     bytes_per_element={} exceeds u64",
                    self.vector_count,
                    self.dim,
                    self.storage_kind.bytes_per_element()
                ))
            })?;
        usize::try_from(bytes).map_err(|_| {
            LaurusError::index(format!(
                "rerank sidecar payload size {bytes} bytes does not fit in usize on this platform"
            ))
        })
    }

    /// Write the header to `writer`.
    ///
    /// All multi-byte fields are little-endian. The reserved 8 bytes
    /// at offset 8 are zero-filled.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&RERANK_SIDECAR_MAGIC)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.storage_kind.tag().to_le_bytes())?;
        writer.write_all(&[0u8; 8])?;
        writer.write_all(&self.dim.to_le_bytes())?;
        writer.write_all(&self.vector_count.to_le_bytes())?;
        Ok(())
    }

    /// Read and validate the header.
    ///
    /// # Errors
    ///
    /// * [`LaurusError::IncompatibleFormat`] if the magic does not
    ///   match `LRS1`, the version is not [`CURRENT_VERSION`], or
    ///   `storage_kind` is unknown / reserved (0).
    /// * [`LaurusError::Io`] for any underlying read failure.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != RERANK_SIDECAR_MAGIC {
            return Err(LaurusError::IncompatibleFormat(format!(
                "expected rerank sidecar magic {:?} (\"LRS1\"), found {:?}",
                RERANK_SIDECAR_MAGIC, magic
            )));
        }

        let mut version_bytes = [0u8; 2];
        reader.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);
        if version != CURRENT_VERSION {
            return Err(LaurusError::IncompatibleFormat(format!(
                "unsupported rerank sidecar header version {version} \
                 (this build supports {CURRENT_VERSION})"
            )));
        }

        let mut kind_bytes = [0u8; 2];
        reader.read_exact(&mut kind_bytes)?;
        let kind_code = u16::from_le_bytes(kind_bytes);
        let storage_kind = match kind_code {
            storage_kind::NONE => {
                return Err(LaurusError::IncompatibleFormat(
                    "rerank sidecar storage_kind = 0 is reserved (no storage); \
                     callers expressing 'no sidecar' must omit the file entirely"
                        .to_string(),
                ));
            }
            storage_kind::F32 => RerankStorageKind::F32,
            other => {
                return Err(LaurusError::IncompatibleFormat(format!(
                    "unknown rerank sidecar storage_kind = {other}"
                )));
            }
        };

        let mut reserved = [0u8; 8];
        reader.read_exact(&mut reserved)?;

        let mut dim_bytes = [0u8; 4];
        reader.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes);

        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        let vector_count = u32::from_le_bytes(count_bytes);

        Ok(Self {
            version,
            storage_kind,
            dim,
            vector_count,
        })
    }
}

/// Fill `buf` from `reader`, stopping early on EOF.
///
/// Unlike [`Read::read_exact`], a clean EOF before the buffer is full
/// is not an error — it is the signal that distinguishes a legacy
/// footer-less sidecar (zero trailing bytes) from a footer or
/// corruption. Transient [`std::io::ErrorKind::Interrupted`] reads are
/// retried.
///
/// # Arguments
///
/// * `reader` - The source to read from.
/// * `buf` - The buffer to fill.
///
/// # Returns
///
/// The number of bytes actually read (`0..=buf.len()`).
///
/// # Errors
///
/// Any non-`Interrupted` I/O error from `reader`.
fn read_up_to<R: Read>(reader: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// Write a complete sidecar (header + raw f32 payload + CRC-32 footer)
/// to `writer`.
///
/// `vectors` is a flat slice of length `vector_count * dim` containing
/// `vector_count` vectors stored back-to-back. The function infers
/// `vector_count` from the slice length and `dim`. The trailing footer
/// (`[RERANK_SIDECAR_FOOTER_MAGIC u32 LE][crc-32 u32 LE]`, Issue #788)
/// carries a CRC-32 over the header and payload so [`read_sidecar`]
/// can detect silent on-disk corruption.
///
/// # Errors
///
/// * [`LaurusError::InvalidOperation`] if `dim == 0` or `vectors.len()`
///   is not a multiple of `dim`.
/// * Any I/O error from `writer`.
pub fn write_sidecar<W: Write>(
    writer: &mut W,
    storage_kind: RerankStorageKind,
    dim: u32,
    vectors: &[f32],
) -> Result<()> {
    if dim == 0 {
        return Err(LaurusError::InvalidOperation(
            "rerank sidecar dim must be > 0".to_string(),
        ));
    }
    let dim_usize = dim as usize;
    if !vectors.len().is_multiple_of(dim_usize) {
        return Err(LaurusError::InvalidOperation(format!(
            "rerank sidecar payload length {} is not divisible by dim {dim}",
            vectors.len()
        )));
    }
    // `dim > 0` and `len % dim == 0` are both established above, so
    // `chunks_exact` never has a leftover partial chunk to silently
    // drop -- this is byte-identical to iterating `vectors` directly.
    write_sidecar_streaming(writer, storage_kind, dim, vectors.chunks_exact(dim_usize))
}

/// Write the `.f32` rerank sidecar directly from an iterator over
/// per-vector f32 slices, without requiring the caller to first
/// flatten every vector into one contiguous `Vec<f32>` (Issue #1151).
///
/// `vector_count` is derived from `vectors.len()` (an
/// [`ExactSizeIterator`]) rather than accepted as a separate argument:
/// a caller-supplied count that disagreed with the actual number of
/// vectors written would silently produce a header/payload mismatch
/// that only surfaces later, as a read failure. The actual number of
/// items iterated is also verified against that count before the
/// footer is written, so a violation of the `ExactSizeIterator`
/// contract itself is caught here rather than producing a corrupt
/// file.
///
/// # Errors
///
/// * [`LaurusError::InvalidOperation`] if `dim == 0`.
/// * [`LaurusError::InvalidOperation`] if any yielded slice's length
///   does not equal `dim`.
/// * [`LaurusError::InvalidOperation`] if the iterator yields a
///   different number of items than its own `len()` reported.
pub(super) fn write_sidecar_streaming<'a, W: Write>(
    writer: &mut W,
    storage_kind: RerankStorageKind,
    dim: u32,
    vectors: impl ExactSizeIterator<Item = &'a [f32]>,
) -> Result<()> {
    if dim == 0 {
        return Err(LaurusError::InvalidOperation(
            "rerank sidecar dim must be > 0".to_string(),
        ));
    }
    let dim_usize = dim as usize;
    let vector_count: u32 = vectors.len().try_into().map_err(|_| {
        LaurusError::InvalidOperation(format!(
            "rerank sidecar vector count {} exceeds u32::MAX",
            vectors.len()
        ))
    })?;
    let header = RerankSidecarHeader::new(storage_kind, dim, vector_count);
    // One scratch buffer reused for every vector instead of flattening
    // the whole segment into a single `Vec<f32>` up front, and written
    // in one `write_all` per vector rather than one per f32 element.
    let mut scratch = vec![0u8; dim_usize * 4];
    let mut written: u32 = 0;
    let content_crc = {
        let mut crc_writer = crate::storage::checksum::CrcWriter::new(&mut *writer);
        header.write_to(&mut crc_writer)?;
        match storage_kind {
            RerankStorageKind::F32 => {
                for v in vectors {
                    if v.len() != dim_usize {
                        return Err(LaurusError::InvalidOperation(format!(
                            "rerank sidecar vector length {} does not match dim {dim}",
                            v.len()
                        )));
                    }
                    for (chunk, &f) in scratch.as_chunks_mut::<4>().0.iter_mut().zip(v) {
                        *chunk = f.to_le_bytes();
                    }
                    crc_writer.write_all(&scratch)?;
                    written += 1;
                }
                if written != vector_count {
                    return Err(LaurusError::InvalidOperation(format!(
                        "rerank sidecar iterator yielded {written} vectors but its \
                         ExactSizeIterator::len() reported {vector_count}"
                    )));
                }
            }
        }
        crc_writer.checksum()
    };
    // The footer itself is excluded from the CRC, mirroring the .hnsw
    // footer layout (Issue #786).
    writer.write_all(&RERANK_SIDECAR_FOOTER_MAGIC.to_le_bytes())?;
    writer.write_all(&content_crc.to_le_bytes())?;
    Ok(())
}

/// Read a complete sidecar (header + raw payload) from `reader`,
/// verifying the CRC-32 footer when present (Issue #788).
///
/// Returns `(header, payload_bytes)` where the byte length of
/// `payload_bytes` equals [`RerankSidecarHeader::payload_size`]. The
/// caller decides how to reinterpret the bytes (e.g. transmute to
/// `&[f32]` for [`RerankStorageKind::F32`]).
///
/// Since the header fully determines the content length, the bytes
/// remaining after the payload identify the file shape unambiguously:
///
/// * zero bytes — legacy footer-less sidecar; verification is skipped
///   (back-compat).
/// * exactly [`FOOTER_SIZE`] bytes starting with
///   [`RERANK_SIDECAR_FOOTER_MAGIC`] — the CRC-32 accumulated over the
///   header and payload is compared against the stored value.
/// * anything else — the file is corrupted.
///
/// # Allocation safety (Issue #791)
///
/// The payload buffer is sized from the just-parsed header, which has
/// **not** been integrity-checked yet (the CRC footer is necessarily
/// verified only after the payload is read). A corrupt or hostile
/// header could therefore declare a multi-GiB payload and abort the
/// process via `handle_alloc_error` (OOM) before the footer ever
/// reports the corruption. To prevent this, `file_size` (the real,
/// on-disk byte length of the sidecar — ground truth the caller obtains
/// from [`crate::storage::StorageInput::size`]) bounds the allocation:
/// because the header fully determines the content length, the file
/// must be at least `HEADER_SIZE + payload_size` bytes (the optional
/// footer only ever follows the payload). A header that declares more
/// payload than the file can hold is rejected as corruption *before*
/// any buffer is allocated.
///
/// # Arguments
///
/// * `reader` - The source to read the sidecar bytes from.
/// * `file_size` - The total byte length of the sidecar file, used to
///   bound the payload allocation before the header is trusted.
///
/// # Errors
///
/// * Any error from [`RerankSidecarHeader::read_from`].
/// * [`LaurusError::Index`] if the header declares a payload that does
///   not fit within `file_size`, or if [`RerankSidecarHeader::payload_size`]
///   overflows (both treated as corruption).
/// * [`LaurusError::Io`] if the payload bytes can't be fully read.
/// * [`LaurusError::Index`] if the checksum does not match the
///   content, the footer magic is wrong, the footer is truncated, or
///   data trails a valid footer (all corruption).
pub fn read_sidecar<R: Read>(
    reader: &mut R,
    file_size: u64,
) -> Result<(RerankSidecarHeader, Vec<u8>)> {
    let mut crc_reader = crate::storage::checksum::CrcReader::new(&mut *reader);
    let header = RerankSidecarHeader::read_from(&mut crc_reader)?;

    // Bound the payload allocation by ground truth (the real file size)
    // before trusting the not-yet-verified header (Issue #791). The
    // header fully determines the content length, so the file must hold
    // at least `HEADER_SIZE + payload_size` bytes (the optional #788
    // footer only follows the payload). Rejecting an oversized claim
    // here keeps a corrupt `dim` / `vector_count` from requesting a huge
    // `vec![0u8; payload_size]` that would abort the process before the
    // CRC footer could report the corruption cleanly.
    let payload_size = header.payload_size()?;
    let min_content = (HEADER_SIZE as u64)
        .checked_add(payload_size as u64)
        .ok_or_else(|| {
            LaurusError::index(
                "rerank sidecar declared content length overflows u64: \
                 .hnsw.f32 file is corrupted",
            )
        })?;
    if min_content > file_size {
        return Err(LaurusError::index(format!(
            "rerank sidecar header declares a {payload_size}-byte payload but the file is only \
             {file_size} bytes: .hnsw.f32 file is corrupted"
        )));
    }

    let mut payload = vec![0u8; payload_size];
    crc_reader.read_exact(&mut payload)?;
    let computed = crc_reader.checksum();
    // The footer must not enter the running CRC, so read it from the
    // inner reader (the exact use case of `CrcReader::get_mut`).
    let inner = crc_reader.get_mut();
    let mut footer = [0u8; FOOTER_SIZE];
    let trailing = read_up_to(inner, &mut footer)?;
    match trailing {
        // Legacy sidecar written before Issue #788: no footer, no
        // verification (back-compat).
        0 => Ok((header, payload)),
        FOOTER_SIZE => {
            let magic = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
            if magic != RERANK_SIDECAR_FOOTER_MAGIC {
                return Err(LaurusError::index(
                    "rerank sidecar has unexpected trailing bytes: .hnsw.f32 file is corrupted",
                ));
            }
            let stored = u32::from_le_bytes([footer[4], footer[5], footer[6], footer[7]]);
            if stored != computed {
                return Err(LaurusError::index(
                    "rerank sidecar checksum mismatch: .hnsw.f32 file is corrupted",
                ));
            }
            let mut extra = [0u8; 1];
            if read_up_to(inner, &mut extra)? != 0 {
                return Err(LaurusError::index(
                    "rerank sidecar has trailing bytes after the checksum footer: \
                     .hnsw.f32 file is corrupted",
                ));
            }
            Ok((header, payload))
        }
        _ => Err(LaurusError::index(
            "rerank sidecar has a truncated checksum footer: .hnsw.f32 file is corrupted",
        )),
    }
}

/// Load a segment's optional `.f32` rerank sidecar into a
/// [`crate::vector::index::rerank_storage::RerankStoragePool`]
/// (Issue #481; shared across HNSW/Flat/IVF readers by #650 PR-2 / #932).
///
/// Lenient-if-absent: returns `Ok(None)` when no sidecar file exists.
/// A present sidecar whose `dim`/`vector_count` disagree with the segment
/// fails loudly.
///
/// The sidecar loads regardless of the storage's loading mode. An
/// Eager-only gate used to sit here "to honor Lazy's memory-savings
/// promise", but it never actually fired behind the engine: the engine
/// wraps every store in `PrefixedStorage`, which misreported the trait
/// default `Eager` until #554 made it delegate — so the shipped, tested
/// behavior has always been "rerank works on mmap", and honoring the
/// corrected mode would have silently degraded Stage 2 to Stage 1 in the
/// default production configuration. Buffering the sidecar is the
/// documented cost of opting a field into `rerank_storage` (#481).
///
/// The pool's positions pair with `vector_ids` (the segment's record
/// order, which the writer also used for the sidecar payload), giving an
/// identity (sidecar position) -> (record position) mapping.
///
/// # Arguments
///
/// * `storage` - The segment's storage backend.
/// * `file_name` - The main segment file name (`"{id}.hnsw"` / `.flat` /
///   `.ivf`); the sidecar is `"{file_name}.f32"`.
/// * `dimension` - The segment's vector dimension.
/// * `vector_ids` - Interned `(doc_id, field_id)` records in segment
///   order.
/// * `field_dict` - The segment's field-name dictionary.
///
/// # Errors
///
/// Forwards [`read_sidecar`] errors and fails on `dim` / `vector_count`
/// mismatches.
pub(crate) fn load_rerank_sidecar(
    storage: &dyn crate::storage::Storage,
    file_name: &str,
    dimension: usize,
    vector_ids: &[(u64, u16)],
    field_dict: &[std::sync::Arc<str>],
) -> Result<Option<std::sync::Arc<crate::vector::index::rerank_storage::RerankStoragePool>>> {
    let sidecar_name = format!("{file_name}.f32");
    if !storage.file_exists(&sidecar_name) {
        return Ok(None);
    }
    let mut sidecar_in = storage.open_input(&sidecar_name)?;
    let sidecar_size = sidecar_in.size()?;
    let (header, payload) = read_sidecar(&mut sidecar_in, sidecar_size)?;
    if header.dim as usize != dimension {
        return Err(LaurusError::InvalidOperation(format!(
            "rerank sidecar dim mismatch: segment uses {dimension}, sidecar uses {}",
            header.dim
        )));
    }
    if header.vector_count as usize != vector_ids.len() {
        return Err(LaurusError::InvalidOperation(format!(
            "rerank sidecar vector_count mismatch: segment has {} vectors, sidecar has {}",
            vector_ids.len(),
            header.vector_count
        )));
    }
    // Transient rehydration for the pool's String-shaped assignment input
    // (eager-only path; nothing retained).
    let assignment: Vec<(u64, String)> = vector_ids
        .iter()
        .map(|&(id, fid)| (id, field_dict[fid as usize].to_string()))
        .collect();
    let pool = crate::vector::index::rerank_storage::RerankStoragePool::from_sidecar_payload(
        header.storage_kind,
        dimension,
        header.vector_count as usize,
        payload,
        &assignment,
    )?;
    Ok(Some(std::sync::Arc::new(pool)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_header() -> RerankSidecarHeader {
        RerankSidecarHeader::new(RerankStorageKind::F32, 8, 4)
    }

    #[test]
    fn roundtrip_header() {
        let header = sample_header();
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);
        let parsed = RerankSidecarHeader::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn header_layout_starts_with_magic_then_version_then_kind() {
        let header = sample_header();
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(&buf[0..4], b"LRS1");
        assert_eq!(u16::from_le_bytes([buf[4], buf[5]]), CURRENT_VERSION);
        assert_eq!(u16::from_le_bytes([buf[6], buf[7]]), storage_kind::F32);
        assert_eq!(&buf[8..16], &[0u8; 8]);
        assert_eq!(u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]), 8);
        assert_eq!(u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]), 4);
    }

    #[test]
    fn payload_size_is_count_times_dim_times_element_bytes() {
        let header = RerankSidecarHeader::new(RerankStorageKind::F32, 16, 100);
        assert_eq!(header.payload_size().unwrap(), 100 * 16 * 4);
    }

    #[test]
    fn payload_size_rejects_overflowing_dimensions() {
        // A hostile header whose `dim` * `vector_count` * 4 overflows
        // u64 must surface a clean error instead of wrapping to a tiny
        // (or zero) allocation in release builds (Issue #791).
        let header = RerankSidecarHeader::new(RerankStorageKind::F32, u32::MAX, u32::MAX);
        let err = header.payload_size().unwrap_err();
        assert_index_error(err, "payload size overflow");
    }

    #[test]
    fn missing_magic_returns_incompatible_format() {
        let buf = [0u8; HEADER_SIZE];
        let err = RerankSidecarHeader::read_from(&mut Cursor::new(&buf[..])).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("LRS1"), "message should mention LRS1");
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_version_returns_incompatible_format() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LRS1");
        buf.extend_from_slice(&99u16.to_le_bytes());
        buf.extend_from_slice(&storage_kind::F32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());

        let err = RerankSidecarHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("99"), "message should mention the version");
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn storage_kind_zero_is_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LRS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&storage_kind::NONE.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());

        let err = RerankSidecarHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, LaurusError::IncompatibleFormat(_)));
    }

    #[test]
    fn unknown_storage_kind_is_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"LRS1");
        buf.extend_from_slice(&CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&999u16.to_le_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());

        let err = RerankSidecarHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        match err {
            LaurusError::IncompatibleFormat(msg) => {
                assert!(msg.contains("999"));
            }
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }

    #[test]
    fn truncated_header_returns_io_error() {
        let buf = b"LRS1".to_vec();
        let err = RerankSidecarHeader::read_from(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, LaurusError::Io(_)));
    }

    #[test]
    fn write_then_read_sidecar_roundtrips_f32_payload() {
        let dim = 4u32;
        let vectors: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -1.0, -2.0, -3.0, -4.0,
        ];
        let mut buf: Vec<u8> = Vec::new();
        write_sidecar(&mut buf, RerankStorageKind::F32, dim, &vectors).unwrap();

        let (header, payload) = read_sidecar(&mut Cursor::new(&buf), buf.len() as u64).unwrap();
        assert_eq!(header.dim, dim);
        assert_eq!(header.vector_count, 3);
        assert_eq!(header.storage_kind, RerankStorageKind::F32);
        assert_eq!(payload.len(), header.payload_size().unwrap());

        for (i, expected) in vectors.iter().enumerate() {
            let lo = i * 4;
            let actual = f32::from_le_bytes([
                payload[lo],
                payload[lo + 1],
                payload[lo + 2],
                payload[lo + 3],
            ]);
            assert_eq!(actual, *expected, "element {i}");
        }
    }

    #[test]
    fn write_sidecar_rejects_zero_dim() {
        let mut buf: Vec<u8> = Vec::new();
        let err = write_sidecar(&mut buf, RerankStorageKind::F32, 0, &[]).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
    }

    #[test]
    fn write_sidecar_rejects_misaligned_payload() {
        let mut buf: Vec<u8> = Vec::new();
        let err = write_sidecar(&mut buf, RerankStorageKind::F32, 4, &[1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
    }

    /// Issue #1151: `write_sidecar_streaming` must produce byte-identical
    /// output to `write_sidecar` for the same logical input, across the
    /// empty / single-vector / multi-vector cases.
    #[test]
    fn write_sidecar_streaming_matches_write_sidecar_byte_for_byte() {
        let dim = 4u32;
        for vectors in [
            Vec::<f32>::new(),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -1.0, -2.0, -3.0, -4.0,
            ],
        ] {
            let mut buf_a: Vec<u8> = Vec::new();
            write_sidecar(&mut buf_a, RerankStorageKind::F32, dim, &vectors).unwrap();

            let mut buf_b: Vec<u8> = Vec::new();
            write_sidecar_streaming(
                &mut buf_b,
                RerankStorageKind::F32,
                dim,
                vectors.chunks_exact(dim as usize),
            )
            .unwrap();

            assert_eq!(buf_a, buf_b, "vectors = {vectors:?}");
        }
    }

    /// Issue #1151: an `ExactSizeIterator` that lies about its own
    /// length (yields fewer items than `len()` reports) must not
    /// silently produce a header/payload mismatch -- the write must
    /// fail loudly instead.
    #[test]
    fn write_sidecar_streaming_rejects_iterator_that_lies_about_its_length() {
        struct LyingIter<'a> {
            remaining: std::slice::Iter<'a, [f32; 4]>,
            claimed_len: usize,
        }
        impl<'a> Iterator for LyingIter<'a> {
            type Item = &'a [f32];
            fn next(&mut self) -> Option<Self::Item> {
                self.remaining.next().map(|v| v.as_slice())
            }
        }
        impl<'a> ExactSizeIterator for LyingIter<'a> {
            fn len(&self) -> usize {
                self.claimed_len
            }
        }

        let data: [[f32; 4]; 1] = [[1.0, 2.0, 3.0, 4.0]];
        let lying = LyingIter {
            remaining: data.iter(),
            claimed_len: 2, // actually yields only 1 item
        };

        let mut buf: Vec<u8> = Vec::new();
        let err = write_sidecar_streaming(&mut buf, RerankStorageKind::F32, 4, lying).unwrap_err();
        assert!(matches!(err, LaurusError::InvalidOperation(_)));
    }

    #[test]
    fn header_constants_are_consistent() {
        assert_eq!(FIXED_HEADER_SIZE, 16);
        assert_eq!(DIM_AND_COUNT_SIZE, 8);
        assert_eq!(HEADER_SIZE, 24);
    }

    /// Build a complete footer-carrying sidecar byte buffer for the
    /// corruption tests below.
    fn sample_sidecar_bytes() -> Vec<u8> {
        let vectors: Vec<f32> = vec![0.5, -1.5, 2.5, -3.5, 4.5, 5.5, -6.5, 7.5];
        let mut buf: Vec<u8> = Vec::new();
        write_sidecar(&mut buf, RerankStorageKind::F32, 4, &vectors).unwrap();
        buf
    }

    fn assert_index_error(err: LaurusError, expected_fragment: &str) {
        match err {
            LaurusError::Index(msg) => {
                assert!(
                    msg.contains(expected_fragment),
                    "message {msg:?} should contain {expected_fragment:?}"
                );
            }
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn write_sidecar_appends_crc_footer() {
        let buf = sample_sidecar_bytes();
        let payload_len = 8 * 4; // 8 f32 elements
        assert_eq!(buf.len(), HEADER_SIZE + payload_len + FOOTER_SIZE);

        let content_end = buf.len() - FOOTER_SIZE;
        let magic = u32::from_le_bytes([
            buf[content_end],
            buf[content_end + 1],
            buf[content_end + 2],
            buf[content_end + 3],
        ]);
        assert_eq!(magic, RERANK_SIDECAR_FOOTER_MAGIC);

        let stored = u32::from_le_bytes([
            buf[content_end + 4],
            buf[content_end + 5],
            buf[content_end + 6],
            buf[content_end + 7],
        ]);
        assert_eq!(stored, crc32fast::hash(&buf[..content_end]));
    }

    #[test]
    fn read_sidecar_accepts_legacy_footerless_payload() {
        // Hand-assemble the pre-#788 layout: header + raw payload, no
        // footer. It must load with verification skipped.
        let vectors: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let header = RerankSidecarHeader::new(RerankStorageKind::F32, 4, 1);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        for v in &vectors {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let (parsed, payload) = read_sidecar(&mut Cursor::new(&buf), buf.len() as u64).unwrap();
        assert_eq!(parsed, header);
        assert_eq!(payload.len(), parsed.payload_size().unwrap());
    }

    #[test]
    fn corrupted_payload_with_footer_is_rejected() {
        let mut buf = sample_sidecar_bytes();
        let payload_mid = HEADER_SIZE + (buf.len() - HEADER_SIZE - FOOTER_SIZE) / 2;
        buf[payload_mid] ^= 0xff;

        let err = read_sidecar(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        assert_index_error(err, "checksum mismatch");
    }

    #[test]
    fn corrupted_header_with_footer_is_rejected() {
        // The reserved region (offset 8..16) parses fine whatever its
        // contents, so a flip there is only caught if the CRC covers
        // the header — which this test locks in.
        let mut buf = sample_sidecar_bytes();
        buf[10] ^= 0xff;

        let err = read_sidecar(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        assert_index_error(err, "checksum mismatch");
    }

    #[test]
    fn footer_with_wrong_magic_is_rejected() {
        let mut buf = sample_sidecar_bytes();
        let magic_pos = buf.len() - FOOTER_SIZE;
        buf[magic_pos] ^= 0xff;

        let err = read_sidecar(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        assert_index_error(err, "unexpected trailing bytes");
    }

    #[test]
    fn truncated_footer_is_rejected() {
        let buf = sample_sidecar_bytes();
        for keep in 1..FOOTER_SIZE {
            let truncated = &buf[..buf.len() - FOOTER_SIZE + keep];
            let err =
                read_sidecar(&mut Cursor::new(truncated), truncated.len() as u64).unwrap_err();
            assert_index_error(err, "truncated checksum footer");
        }
    }

    #[test]
    fn trailing_bytes_after_footer_are_rejected() {
        let mut buf = sample_sidecar_bytes();
        buf.push(0u8);

        let err = read_sidecar(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        assert_index_error(err, "trailing bytes after the checksum footer");
    }

    #[test]
    fn oversized_header_is_rejected_before_allocating() {
        // Reproduce the Issue #791 hazard: a valid-looking header whose
        // `vector_count` declares far more payload than the file can
        // possibly hold. Here the buffer is a bare 24-byte header but the
        // header claims `dim=16 * vector_count=2^28` f32 elements
        // (~17 GiB). `read_sidecar` must reject this against the real
        // file size *before* allocating `vec![0u8; payload_size]`, so the
        // process is never asked for gigabytes (which would abort via
        // `handle_alloc_error`).
        let dim = 16u32;
        let huge_count = 1u32 << 28; // 268_435_456 vectors
        let header = RerankSidecarHeader::new(RerankStorageKind::F32, dim, huge_count);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);
        // Sanity-check the declared payload really is huge and in-range
        // for `payload_size` (so the rejection is the file-size bound,
        // not the overflow guard).
        assert_eq!(
            header.payload_size().unwrap(),
            huge_count as usize * dim as usize * 4
        );

        let err = read_sidecar(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        assert_index_error(err, "declares a");
    }

    #[test]
    fn header_declaring_one_byte_too_many_is_rejected() {
        // The bound is exact: a footer-less sidecar whose payload is one
        // byte short of the header's claim must be rejected, never read
        // with a truncating allocation.
        let header = RerankSidecarHeader::new(RerankStorageKind::F32, 4, 1);
        let mut buf: Vec<u8> = Vec::new();
        header.write_to(&mut buf).unwrap();
        // payload should be 4 * 4 = 16 bytes; write only 15.
        buf.extend_from_slice(&[0u8; 15]);

        let err = read_sidecar(&mut Cursor::new(&buf), buf.len() as u64).unwrap_err();
        assert_index_error(err, "declares a");
    }
}
