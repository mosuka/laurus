//! LMV1: the on-disk format of one multi-vector segment (Issue #1177).
//!
//! All integers are little-endian.
//!
//! | Offset | Size | Content |
//! | --- | --- | --- |
//! | 0 | 64 | Header: magic `LMV1`, version `u16`, element kind `u8`, reserved `u8`, dimension `u32`, reserved `u32`, document count `u64`, vector count `u64`, zero padding |
//! | 64 | element kind's parameter block | `Binary` only: `dimension` f16 values of the center, then `dimension` of the spread, zero-padded to a multiple of 8 bytes; empty for every other kind |
//! | | 24 × documents | Document table, ascending by doc id: doc id `u64`, first vector `u64`, vector count `u32`, reserved `u32` |
//! | | row size × vectors | Payload: encoded rows, one document's vectors after another |
//! | end − 16 | 16 | Footer: CRC-32 of header + parameters + table `u32`, CRC-32 of payload `u32`, magic `LMVF`, reserved `u32` |
//!
//! The payload starts at an 8-byte-aligned offset, so a memory-mapped `F32`
//! segment is read as `&[f32]` without copying. Opening a segment checks
//! the header, the exact file size and the header + parameters + table
//! checksum; the payload checksum, which covers nearly the whole file, is
//! checked only by [`SegmentReader::verify_payload`] (merges and
//! validation), so opening a multi-gigabyte segment does not read it.
//!
//! See [`crate::vector::core::multi_vector`] for the element kinds' row
//! layouts and codecs. Every segment carries what it takes to decode its
//! own rows, so segments written under different kinds (or different
//! `Binary` parameters) coexist in one field.

use std::borrow::Cow;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{LaurusError, Result};
use crate::storage::{Storage, StorageInput, StorageOutput};
use crate::vector::core::multi_vector::{
    BinaryParams, MultiVectorRows, MultiVectorStorage, OwnedCodec, RowCodec,
};

const MAGIC: [u8; 4] = *b"LMV1";
const FOOTER_MAGIC: [u8; 4] = *b"LMVF";
const VERSION: u16 = 1;

const HEADER_LEN: usize = 64;
const DOC_ENTRY_LEN: usize = 24;
const FOOTER_LEN: usize = 16;

/// One document of a segment: its id and how many vectors it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DocEntry {
    /// Internal document id.
    pub doc_id: u64,
    /// Number of token vectors.
    pub vector_count: u32,
}

/// Counts and element kind declared by a segment header.
#[derive(Debug, Clone, Copy)]
struct Header {
    kind: MultiVectorStorage,
    dimension: usize,
    doc_count: u64,
    vector_count: u64,
}

impl Header {
    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut bytes = [0u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
        bytes[6] = self.kind.tag();
        bytes[8..12].copy_from_slice(&(self.dimension as u32).to_le_bytes());
        bytes[16..24].copy_from_slice(&self.doc_count.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.vector_count.to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes[0..4] != MAGIC {
            return Err(LaurusError::IncompatibleFormat(
                "not a multi-vector segment (bad magic)".to_string(),
            ));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(LaurusError::IncompatibleFormat(format!(
                "multi-vector segment version {version} is not supported (expected {VERSION})"
            )));
        }
        let kind = MultiVectorStorage::from_tag(bytes[6]).ok_or_else(|| {
            LaurusError::IncompatibleFormat(format!(
                "multi-vector element kind {} is not supported by this version",
                bytes[6]
            ))
        })?;
        let dimension = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize;
        if dimension == 0 {
            return Err(LaurusError::index(
                "multi-vector segment declares dimension 0",
            ));
        }
        Ok(Self {
            kind,
            dimension,
            doc_count: u64::from_le_bytes(bytes[16..24].try_into().expect("8 bytes")),
            vector_count: u64::from_le_bytes(bytes[24..32].try_into().expect("8 bytes")),
        })
    }

    /// Byte length of the element kind's parameter block.
    fn params_len(&self) -> usize {
        self.kind.params_len(self.dimension)
    }

    /// Byte offset of the payload.
    fn payload_offset(&self) -> Option<u64> {
        self.doc_count
            .checked_mul(DOC_ENTRY_LEN as u64)?
            .checked_add(HEADER_LEN as u64)?
            .checked_add(self.params_len() as u64)
    }

    /// Byte length of the payload.
    fn payload_len(&self) -> Option<u64> {
        self.vector_count
            .checked_mul(self.kind.row_bytes(self.dimension) as u64)
    }

    /// Exact byte length of a segment with these counts.
    fn file_len(&self) -> Option<u64> {
        self.payload_offset()?
            .checked_add(self.payload_len()?)?
            .checked_add(FOOTER_LEN as u64)
    }
}

/// Receives one document's vectors while a segment is written.
pub(crate) struct PayloadSink<'a> {
    output: &'a mut Box<dyn StorageOutput>,
    hasher: &'a mut crc32fast::Hasher,
    buffer: &'a mut Vec<u8>,
    codec: RowCodec<'a>,
    dimension: usize,
    /// Rows still expected for the current document.
    remaining: usize,
}

impl PayloadSink<'_> {
    /// Encode and append `values` (whole vectors, row-major) to the
    /// current document.
    ///
    /// # Errors
    ///
    /// Returns an error when `values` is not a whole number of rows, when
    /// it holds more rows than the document's table entry declared, or
    /// when writing fails.
    pub(crate) fn write(&mut self, values: &[f32]) -> Result<()> {
        if !values.len().is_multiple_of(self.dimension) {
            return Err(LaurusError::internal(
                "multi-vector payload is not a whole number of rows",
            ));
        }
        let rows = values.len() / self.dimension;
        if rows > self.remaining {
            return Err(LaurusError::internal(
                "multi-vector payload is longer than its table entry declares",
            ));
        }
        self.buffer.clear();
        for row in values.chunks_exact(self.dimension) {
            self.codec.encode_row(row, self.buffer);
        }
        self.hasher.update(self.buffer);
        self.output.write_all(self.buffer)?;
        self.remaining -= rows;
        Ok(())
    }

    /// Append rows read from another segment (a merge): copied verbatim
    /// when they are encoded exactly as this segment encodes (same kind
    /// and, for `Binary`, equal parameters), transcoded through f32
    /// otherwise.
    ///
    /// This is the only place that decides whether encoded bytes may be
    /// copied. Re-encoding a decoded `Binary` row can flip bits even under
    /// the same parameters, so a copy across different encodings would
    /// misread the rows, and a needless transcode would lose precision on
    /// every merge.
    ///
    /// # Errors
    ///
    /// Returns an error when `rows` has another dimension, holds more rows
    /// than the document's table entry declared, or when writing fails.
    pub(crate) fn append(&mut self, rows: &MultiVectorRows<'_>) -> Result<()> {
        if rows.dimension() != self.dimension {
            return Err(LaurusError::internal(
                "multi-vector rows have another dimension than the segment",
            ));
        }
        if rows.codec().same_encoding(self.codec) {
            self.write_encoded(rows.as_bytes())
        } else {
            self.write(&rows.to_f32())
        }
    }

    /// Append already-encoded row bytes verbatim: `bytes.len()` must be a
    /// whole number of `self.codec.row_bytes(self.dimension)`-sized rows.
    fn write_encoded(&mut self, bytes: &[u8]) -> Result<()> {
        let row_bytes = self.codec.row_bytes(self.dimension);
        if !bytes.len().is_multiple_of(row_bytes) {
            return Err(LaurusError::internal(
                "multi-vector payload is not a whole number of rows",
            ));
        }
        let rows = bytes.len() / row_bytes;
        if rows > self.remaining {
            return Err(LaurusError::internal(
                "multi-vector payload is longer than its table entry declares",
            ));
        }
        self.hasher.update(bytes);
        self.output.write_all(bytes)?;
        self.remaining -= rows;
        Ok(())
    }
}

/// Write a segment atomically: to `{file_name}.tmp`, fsynced, then renamed.
///
/// `fill` is called once per entry of `docs`, in order, and must write
/// exactly that document's `vector_count × dimension` values.
///
/// # Arguments
///
/// * `storage` - Storage to write into.
/// * `file_name` - Final file name (e.g. `segment_000003.mv`).
/// * `codec` - How every vector is encoded: the element kind and, for
///   `Binary`, the parameters stored in the segment.
/// * `dimension` - Dimension of every vector.
/// * `docs` - The documents, strictly ascending by doc id, each with at
///   least one vector.
/// * `fill` - Writes the payload of the `i`-th document.
///
/// # Errors
///
/// Returns an error when `docs` is not strictly ascending, declares an
/// empty document, `fill` writes the wrong number of values, or I/O fails.
/// The temporary file is removed on failure.
pub(crate) fn write_segment(
    storage: &dyn Storage,
    file_name: &str,
    codec: RowCodec<'_>,
    dimension: usize,
    docs: &[DocEntry],
    fill: impl FnMut(usize, &mut PayloadSink<'_>) -> Result<()>,
) -> Result<()> {
    let tmp = format!("{file_name}.tmp");
    match write_segment_to(storage, &tmp, codec, dimension, docs, fill) {
        Ok(()) => storage.rename_file(&tmp, file_name),
        Err(e) => {
            if storage.file_exists(&tmp) {
                let _ = storage.delete_file(&tmp);
            }
            Err(e)
        }
    }
}

fn write_segment_to(
    storage: &dyn Storage,
    file_name: &str,
    codec: RowCodec<'_>,
    dimension: usize,
    docs: &[DocEntry],
    mut fill: impl FnMut(usize, &mut PayloadSink<'_>) -> Result<()>,
) -> Result<()> {
    if dimension == 0 {
        return Err(LaurusError::internal(
            "multi-vector segment dimension must be greater than 0",
        ));
    }
    if codec
        .binary_params()
        .is_some_and(|params| params.dimension() != dimension)
    {
        return Err(LaurusError::internal(
            "binary parameters do not match the segment dimension",
        ));
    }
    if docs.windows(2).any(|w| w[0].doc_id >= w[1].doc_id) {
        return Err(LaurusError::internal(
            "multi-vector segment documents must be strictly ascending by doc id",
        ));
    }
    if docs.iter().any(|d| d.vector_count == 0) {
        return Err(LaurusError::internal(
            "multi-vector segment documents must hold at least one vector",
        ));
    }

    let header = Header {
        kind: codec.kind(),
        dimension,
        doc_count: docs.len() as u64,
        vector_count: docs.iter().map(|d| u64::from(d.vector_count)).sum(),
    };
    let mut meta =
        Vec::with_capacity(HEADER_LEN + header.params_len() + docs.len() * DOC_ENTRY_LEN);
    meta.extend_from_slice(&header.encode());
    if let Some(params) = codec.binary_params() {
        params.write_block(&mut meta);
    }
    debug_assert_eq!(meta.len(), HEADER_LEN + header.params_len());
    let mut first_vector = 0u64;
    for doc in docs {
        meta.extend_from_slice(&doc.doc_id.to_le_bytes());
        meta.extend_from_slice(&first_vector.to_le_bytes());
        meta.extend_from_slice(&doc.vector_count.to_le_bytes());
        meta.extend_from_slice(&0u32.to_le_bytes());
        first_vector += u64::from(doc.vector_count);
    }
    let meta_crc = crc32fast::hash(&meta);

    let mut output = storage.create_output(file_name)?;
    output.write_all(&meta)?;

    let mut payload_hasher = crc32fast::Hasher::new();
    let mut buffer = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        let mut sink = PayloadSink {
            output: &mut output,
            hasher: &mut payload_hasher,
            buffer: &mut buffer,
            codec,
            dimension,
            remaining: doc.vector_count as usize,
        };
        fill(i, &mut sink)?;
        if sink.remaining != 0 {
            return Err(LaurusError::internal(format!(
                "multi-vector payload for doc {} is shorter than its table entry declares",
                doc.doc_id
            )));
        }
    }

    let mut footer = [0u8; FOOTER_LEN];
    footer[0..4].copy_from_slice(&meta_crc.to_le_bytes());
    footer[4..8].copy_from_slice(&payload_hasher.finalize().to_le_bytes());
    footer[8..12].copy_from_slice(&FOOTER_MAGIC);
    output.write_all(&footer)?;
    output.flush_and_sync()?;
    output.close()
}

/// Read access to one sealed multi-vector segment.
#[derive(Debug)]
pub(crate) struct SegmentReader {
    /// The segment's element kind and, for `Binary`, its own parameters.
    codec: OwnedCodec,
    dimension: usize,
    vector_count: u64,
    doc_ids: Vec<u64>,
    /// `(first vector, vector count)` per entry of `doc_ids`.
    spans: Vec<(u64, u32)>,
    payload_offset: u64,
    payload_crc: u32,
    /// Zero-copy view of the whole file. This input is never read or
    /// seeked, so `as_slice` keeps returning the file from offset 0.
    mapped: Option<Box<dyn StorageInput>>,
    /// Seekable handle used when the storage offers no zero-copy view.
    stream: Option<Mutex<Box<dyn StorageInput>>>,
}

impl SegmentReader {
    /// Open the segment `file_name`.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::IncompatibleFormat`] for a file that is not an
    /// LMV1 segment or uses an unsupported version / element kind, and an
    /// index error when the size, the document table or the header + table
    /// checksum is inconsistent.
    pub(crate) fn open(storage: &dyn Storage, file_name: &str) -> Result<Self> {
        let mut input = storage.open_input(file_name)?;
        let file_len = input.size()?;
        if file_len < (HEADER_LEN + FOOTER_LEN) as u64 {
            return Err(corrupt(file_name, "file is shorter than header + footer"));
        }

        if let Some(bytes) = input.as_slice() {
            let header = Header::decode(&bytes[..HEADER_LEN])?;
            check_len(file_name, &header, file_len)?;
            let table_end = header.payload_offset().expect("checked by check_len") as usize;
            let footer = &bytes[bytes.len() - FOOTER_LEN..];
            let table = parse_meta(file_name, &header, &bytes[..table_end], footer)?;
            return Ok(Self {
                codec: table.codec,
                dimension: header.dimension,
                vector_count: header.vector_count,
                doc_ids: table.doc_ids,
                spans: table.spans,
                payload_offset: table_end as u64,
                payload_crc: table.payload_crc,
                mapped: Some(input),
                stream: None,
            });
        }

        let mut header_bytes = [0u8; HEADER_LEN];
        input.read_exact(&mut header_bytes)?;
        let header = Header::decode(&header_bytes)?;
        check_len(file_name, &header, file_len)?;
        let table_end = header.payload_offset().expect("checked by check_len") as usize;
        let mut meta = vec![0u8; table_end];
        meta[..HEADER_LEN].copy_from_slice(&header_bytes);
        input.read_exact(&mut meta[HEADER_LEN..])?;
        let mut footer = [0u8; FOOTER_LEN];
        input.seek(SeekFrom::Start(file_len - FOOTER_LEN as u64))?;
        input.read_exact(&mut footer)?;
        let table = parse_meta(file_name, &header, &meta, &footer)?;
        Ok(Self {
            codec: table.codec,
            dimension: header.dimension,
            vector_count: header.vector_count,
            doc_ids: table.doc_ids,
            spans: table.spans,
            payload_offset: table_end as u64,
            payload_crc: table.payload_crc,
            mapped: None,
            stream: Some(Mutex::new(input)),
        })
    }

    /// Dimension of every vector.
    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    /// This segment's `Binary` parameters, `None` for any other kind.
    pub(crate) fn binary_params(&self) -> Option<&Arc<BinaryParams>> {
        match &self.codec {
            OwnedCodec::Binary(params) => Some(params),
            OwnedCodec::F32 | OwnedCodec::F16 | OwnedCodec::Int8 => None,
        }
    }

    /// Document ids, ascending.
    pub(crate) fn doc_ids(&self) -> &[u64] {
        &self.doc_ids
    }

    /// The documents of this segment, ascending by doc id.
    pub(crate) fn entries(&self) -> impl Iterator<Item = DocEntry> + '_ {
        self.doc_ids
            .iter()
            .zip(&self.spans)
            .map(|(&doc_id, &(_, vector_count))| DocEntry {
                doc_id,
                vector_count,
            })
    }

    /// The still-encoded rows of `doc_id`, or `None` when this segment
    /// does not hold it.
    ///
    /// The returned bytes are borrowed without copying when the storage
    /// offers a zero-copy view; read into a new buffer otherwise. Used
    /// both for scoring (via [`MultiVectorRows::to_f32`]) and for a
    /// same-kind merge byte copy (via the raw bytes it wraps).
    ///
    /// # Errors
    ///
    /// Returns an error when reading from storage fails.
    pub(crate) fn rows(&self, doc_id: u64) -> Result<Option<MultiVectorRows<'_>>> {
        let Ok(index) = self.doc_ids.binary_search(&doc_id) else {
            return Ok(None);
        };
        let (first, count) = self.spans[index];
        let row = self.codec.kind().row_bytes(self.dimension) as u64;
        let start = self.payload_offset + first * row;
        let len = (u64::from(count) * row) as usize;

        if let Some(input) = &self.mapped {
            let bytes = input
                .as_slice()
                .ok_or_else(|| LaurusError::internal("zero-copy view disappeared"))?;
            let bytes = &bytes[start as usize..start as usize + len];
            return Ok(Some(MultiVectorRows::new(
                self.codec.as_codec(),
                self.dimension,
                Cow::Borrowed(bytes),
            )));
        }

        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| LaurusError::internal("multi-vector segment has no input"))?;
        let mut input = stream.lock();
        input.seek(SeekFrom::Start(start))?;
        let mut bytes = vec![0u8; len];
        input.read_exact(&mut bytes)?;
        Ok(Some(MultiVectorRows::new(
            self.codec.as_codec(),
            self.dimension,
            Cow::Owned(bytes),
        )))
    }

    /// Check the payload checksum, which opening does not verify.
    ///
    /// # Errors
    ///
    /// Returns an index error when the payload does not match its checksum,
    /// or an I/O error when reading fails.
    pub(crate) fn verify_payload(&self) -> Result<()> {
        let payload_len =
            (self.vector_count * self.codec.kind().row_bytes(self.dimension) as u64) as usize;
        let start = self.payload_offset as usize;
        let actual = if let Some(input) = &self.mapped {
            let bytes = input
                .as_slice()
                .ok_or_else(|| LaurusError::internal("zero-copy view disappeared"))?;
            crc32fast::hash(&bytes[start..start + payload_len])
        } else {
            let stream = self
                .stream
                .as_ref()
                .ok_or_else(|| LaurusError::internal("multi-vector segment has no input"))?;
            let mut input = stream.lock();
            input.seek(SeekFrom::Start(self.payload_offset))?;
            let mut hasher = crc32fast::Hasher::new();
            let mut remaining = payload_len;
            let mut chunk = vec![0u8; 1 << 16];
            while remaining > 0 {
                let n = remaining.min(chunk.len());
                input.read_exact(&mut chunk[..n])?;
                hasher.update(&chunk[..n]);
                remaining -= n;
            }
            hasher.finalize()
        };
        if actual != self.payload_crc {
            return Err(LaurusError::index(
                "multi-vector segment payload checksum mismatch",
            ));
        }
        Ok(())
    }
}

fn corrupt(file_name: &str, reason: &str) -> LaurusError {
    LaurusError::index(format!(
        "multi-vector segment '{file_name}' is corrupted: {reason}"
    ))
}

/// The file must be exactly as long as its header declares, which also
/// bounds every allocation sized from the header.
fn check_len(file_name: &str, header: &Header, file_len: u64) -> Result<()> {
    match header.file_len() {
        Some(expected) if expected == file_len => Ok(()),
        Some(expected) => Err(corrupt(
            file_name,
            &format!("expected {expected} bytes, found {file_len}"),
        )),
        None => Err(corrupt(file_name, "declared counts overflow")),
    }
}

/// The decoded parameters and document table of a segment, with its
/// payload checksum.
struct DocTable {
    codec: OwnedCodec,
    doc_ids: Vec<u64>,
    /// `(first vector, vector count)` per entry of `doc_ids`.
    spans: Vec<(u64, u32)>,
    payload_crc: u32,
}

/// Verify the footer and the header + parameters + table checksum, then
/// decode the parameters and the table.
fn parse_meta(file_name: &str, header: &Header, meta: &[u8], footer: &[u8]) -> Result<DocTable> {
    if footer[8..12] != FOOTER_MAGIC {
        return Err(corrupt(file_name, "bad footer magic"));
    }
    let meta_crc = u32::from_le_bytes(footer[0..4].try_into().expect("4 bytes"));
    let payload_crc = u32::from_le_bytes(footer[4..8].try_into().expect("4 bytes"));
    if crc32fast::hash(meta) != meta_crc {
        return Err(corrupt(
            file_name,
            "header / document table checksum mismatch",
        ));
    }

    let table_start = HEADER_LEN + header.params_len();
    let codec = match OwnedCodec::plain(header.kind) {
        Some(codec) => codec,
        None => OwnedCodec::Binary(Arc::new(
            BinaryParams::read_block(&meta[HEADER_LEN..table_start], header.dimension)
                .map_err(|e| corrupt(file_name, &e.to_string()))?,
        )),
    };

    let doc_count = header.doc_count as usize;
    let mut doc_ids = Vec::with_capacity(doc_count);
    let mut spans = Vec::with_capacity(doc_count);
    let mut next_vector = 0u64;
    for entry in meta[table_start..].as_chunks::<DOC_ENTRY_LEN>().0 {
        let doc_id = u64::from_le_bytes(entry[0..8].try_into().expect("8 bytes"));
        let first = u64::from_le_bytes(entry[8..16].try_into().expect("8 bytes"));
        let count = u32::from_le_bytes(entry[16..20].try_into().expect("4 bytes"));
        if doc_ids.last().is_some_and(|&last| last >= doc_id) {
            return Err(corrupt(file_name, "document table is not ascending"));
        }
        if first != next_vector || count == 0 {
            return Err(corrupt(
                file_name,
                "document table spans are not contiguous",
            ));
        }
        next_vector += u64::from(count);
        doc_ids.push(doc_id);
        spans.push((first, count));
    }
    if next_vector != header.vector_count {
        return Err(corrupt(
            file_name,
            "document table does not cover the declared vector count",
        ));
    }
    Ok(DocTable {
        codec,
        doc_ids,
        spans,
        payload_crc,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::storage::file::{FileStorage, FileStorageConfig};
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};
    use crate::vector::core::multi_vector::test_binary_params;

    fn memory() -> Arc<dyn Storage> {
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
    }

    fn file(dir: &tempfile::TempDir, use_mmap: bool) -> Arc<dyn Storage> {
        let mut config = FileStorageConfig::new(dir.path());
        config.use_mmap = use_mmap;
        Arc::new(FileStorage::new(dir.path(), config).unwrap())
    }

    /// Two documents, dimension 3: doc 4 holds two vectors, doc 9 one.
    fn sample() -> Vec<(u64, Vec<Vec<f32>>)> {
        vec![
            (4, vec![vec![1.0, 2.0, 3.0], vec![-1.0, 0.5, 0.25]]),
            (9, vec![vec![7.0, 8.0, 9.0]]),
        ]
    }

    fn write(
        storage: &dyn Storage,
        name: &str,
        codec: RowCodec<'_>,
        dim: usize,
        docs: &[(u64, Vec<Vec<f32>>)],
    ) {
        let entries: Vec<DocEntry> = docs
            .iter()
            .map(|(doc_id, vectors)| DocEntry {
                doc_id: *doc_id,
                vector_count: vectors.len() as u32,
            })
            .collect();
        write_segment(storage, name, codec, dim, &entries, |i, sink| {
            for v in &docs[i].1 {
                sink.write(v)?;
            }
            Ok(())
        })
        .unwrap();
    }

    fn read_all(storage: &dyn Storage, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        storage
            .open_input(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    fn overwrite(storage: &dyn Storage, name: &str, bytes: &[u8]) {
        let mut output = storage.create_output(name).unwrap();
        output.write_all(bytes).unwrap();
        output.close().unwrap();
    }

    /// Every codec, with `params` for `Binary`.
    fn every_codec(params: &BinaryParams) -> [RowCodec<'_>; 4] {
        [
            RowCodec::F32,
            RowCodec::F16,
            RowCodec::Int8,
            RowCodec::Binary(params),
        ]
    }

    /// What reading `rows` back must return, and the per-element
    /// tolerance. `Binary` is lossy by design, so it is held exactly to
    /// its own decode of the encoded rows instead of to the input.
    fn expected(codec: RowCodec<'_>, rows: &[Vec<f32>]) -> (Vec<f32>, f32) {
        let flat = rows.concat();
        match codec.kind() {
            MultiVectorStorage::F32 => (flat, 0.0),
            MultiVectorStorage::F16 => (flat, 2e-3),
            MultiVectorStorage::Int8 => (flat, 5e-2),
            MultiVectorStorage::Binary => {
                let dim = rows[0].len();
                let mut bytes = Vec::new();
                for row in rows {
                    codec.encode_row(row, &mut bytes);
                }
                let mut out = Vec::new();
                for row in bytes.chunks_exact(codec.row_bytes(dim)) {
                    codec.decode_row(row, dim, &mut out);
                }
                (out, 0.0)
            }
        }
    }

    /// Test-only convenience wrapping [`SegmentReader::rows`] +
    /// [`MultiVectorRows::to_f32`], mirroring the removed
    /// `SegmentReader::vectors` that only tests still needed.
    fn vectors(reader: &SegmentReader, doc_id: u64) -> Option<Cow<'_, [f32]>> {
        reader.rows(doc_id).unwrap().map(|rows| rows.to_f32())
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= tol,
                "{actual:?} vs {expected:?} (tol {tol})"
            );
        }
    }

    fn assert_round_trip(storage: &dyn Storage, codec: RowCodec<'_>) {
        write(storage, "seg.mv", codec, 3, &sample());
        let reader = SegmentReader::open(storage, "seg.mv").unwrap();
        assert_eq!(reader.rows(4).unwrap().unwrap().kind(), codec.kind());
        assert!(
            reader
                .rows(4)
                .unwrap()
                .unwrap()
                .codec()
                .same_encoding(codec)
        );
        assert_eq!(
            reader.binary_params().map(|p| p.as_ref()),
            codec.binary_params()
        );
        assert_eq!(reader.dimension(), 3);
        assert_eq!(reader.doc_ids(), &[4, 9]);
        assert_eq!(
            reader.entries().collect::<Vec<_>>(),
            vec![
                DocEntry {
                    doc_id: 4,
                    vector_count: 2
                },
                DocEntry {
                    doc_id: 9,
                    vector_count: 1
                },
            ]
        );
        for (doc_id, rows) in sample() {
            let (values, tol) = expected(codec, &rows);
            assert_close(&vectors(&reader, doc_id).unwrap(), &values, tol);
        }
        assert!(vectors(&reader, 5).is_none());
        reader.verify_payload().unwrap();
        assert!(!storage.file_exists("seg.mv.tmp"));
    }

    #[test]
    fn test_round_trip_in_memory_for_every_storage_kind() {
        let params = test_binary_params(3);
        for codec in every_codec(&params) {
            assert_round_trip(memory().as_ref(), codec);
        }
    }

    #[test]
    fn test_round_trip_on_file_with_mmap_is_zero_copy_only_for_f32() {
        let params = test_binary_params(3);
        for codec in every_codec(&params) {
            let dir = tempfile::TempDir::new().unwrap();
            let storage = file(&dir, true);
            assert_round_trip(storage.as_ref(), codec);
            // A memory map is page-aligned and the payload offset is a
            // multiple of 8, so an `F32` row set is borrowed, not copied.
            // Other kinds' rows can never be reinterpreted as `&[f32]`.
            let reader = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap();
            let rows = reader.rows(4).unwrap().unwrap();
            assert_eq!(
                rows.as_f32().is_some(),
                codec.kind() == MultiVectorStorage::F32
            );
        }
    }

    #[test]
    fn test_round_trip_on_file_without_mmap() {
        // The stream path reads the header, parameters and table itself.
        let params = test_binary_params(3);
        for codec in [RowCodec::F32, RowCodec::Binary(&params)] {
            let dir = tempfile::TempDir::new().unwrap();
            assert_round_trip(file(&dir, false).as_ref(), codec);
        }
    }

    /// `64 + pad8(4 * dim) + 24 * docs + ceil(dim / 8) * rows + 16`.
    #[test]
    fn test_binary_segment_has_the_documented_length() {
        let storage = memory();
        let params = test_binary_params(3);
        write(
            storage.as_ref(),
            "seg.mv",
            RowCodec::Binary(&params),
            3,
            &sample(),
        );
        let len = read_all(storage.as_ref(), "seg.mv").len();
        assert_eq!(len, HEADER_LEN + 16 + 2 * DOC_ENTRY_LEN + 3 + FOOTER_LEN);
    }

    #[test]
    fn test_binary_parameter_tamper_is_caught_by_the_table_checksum() {
        let storage = memory();
        let params = test_binary_params(3);
        write(
            storage.as_ref(),
            "seg.mv",
            RowCodec::Binary(&params),
            3,
            &sample(),
        );
        let mut bytes = read_all(storage.as_ref(), "seg.mv");
        bytes[HEADER_LEN] ^= 0x01; // the first center value
        overwrite(storage.as_ref(), "seg.mv", &bytes);
        let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
    }

    #[test]
    fn test_non_finite_binary_parameters_are_rejected_even_with_a_valid_checksum() {
        let storage = memory();
        let params = test_binary_params(3);
        write(
            storage.as_ref(),
            "seg.mv",
            RowCodec::Binary(&params),
            3,
            &sample(),
        );
        let mut bytes = read_all(storage.as_ref(), "seg.mv");
        bytes[HEADER_LEN..HEADER_LEN + 2].copy_from_slice(&half::f16::NAN.to_le_bytes());
        let meta_end = HEADER_LEN + 16 + 2 * DOC_ENTRY_LEN;
        let meta_crc = crc32fast::hash(&bytes[..meta_end]);
        let footer = bytes.len() - FOOTER_LEN;
        bytes[footer..footer + 4].copy_from_slice(&meta_crc.to_le_bytes());
        overwrite(storage.as_ref(), "seg.mv", &bytes);
        let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
        assert!(err.to_string().contains("non-finite"), "{err}");
    }

    #[test]
    fn test_empty_binary_segment_keeps_its_parameters() {
        let storage = memory();
        let params = test_binary_params(4);
        write(
            storage.as_ref(),
            "empty.mv",
            RowCodec::Binary(&params),
            4,
            &[],
        );
        let reader = SegmentReader::open(storage.as_ref(), "empty.mv").unwrap();
        assert!(reader.doc_ids().is_empty());
        assert_eq!(reader.binary_params().map(|p| p.as_ref()), Some(&params));
        reader.verify_payload().unwrap();
    }

    /// Writes `rows` of `source` into a one-document segment encoded with
    /// `target` through [`PayloadSink::append`], and returns that segment's
    /// raw payload bytes and decoded values.
    fn append_into(
        source: &SegmentReader,
        doc_id: u64,
        target: RowCodec<'_>,
    ) -> (Vec<u8>, Vec<f32>) {
        let storage = memory();
        let rows = source.rows(doc_id).unwrap().unwrap();
        let entry = DocEntry {
            doc_id,
            vector_count: rows.row_count() as u32,
        };
        write_segment(
            storage.as_ref(),
            "out.mv",
            target,
            3,
            &[entry],
            |_, sink| sink.append(&rows),
        )
        .unwrap();
        let reader = SegmentReader::open(storage.as_ref(), "out.mv").unwrap();
        let out = reader.rows(doc_id).unwrap().unwrap();
        (out.as_bytes().to_vec(), out.to_f32().into_owned())
    }

    #[test]
    fn test_append_copies_only_rows_encoded_with_equal_parameters() {
        // c = [0.5; 3], w = [0.1, 0.1, 0.05]: the bits 0b111 decode to a
        // row whose re-encoding clears bit 2, so only a verbatim copy keeps
        // them.
        let params = || {
            let bits = [0.5f32, 0.5, 0.5, 0.1, 0.1, 0.05].map(|v| half::f16::from_f32(v).to_bits());
            BinaryParams::from_f16_bits(3, bits.into()).unwrap()
        };
        let p1 = params();
        let codec = RowCodec::Binary(&p1);
        let mut bits = Vec::new();
        codec.encode_row(&[0.6, 0.6, 0.6], &mut bits);
        assert_eq!(bits, [0b111]);
        let mut decoded = Vec::new();
        codec.decode_row(&bits, 3, &mut decoded);
        let mut again = Vec::new();
        codec.encode_row(&decoded, &mut again);
        assert_ne!(again, bits, "the data must exercise the copy rule");

        let storage = memory();
        let docs = [(4, vec![vec![0.6, 0.6, 0.6], vec![1.0, 2.0, 3.0]])];
        write(storage.as_ref(), "p1.mv", codec, 3, &docs);
        let source = SegmentReader::open(storage.as_ref(), "p1.mv").unwrap();
        let source_rows = source.rows(4).unwrap().unwrap();

        // Equal parameters in another allocation: a verbatim copy.
        let p1_again = params();
        let (bytes, _) = append_into(&source, 4, RowCodec::Binary(&p1_again));
        assert_eq!(bytes, source_rows.as_bytes());

        // Different parameters: decoded under P1, re-encoded under P2.
        let p2 = test_binary_params(3);
        let (_, values) = append_into(&source, 4, RowCodec::Binary(&p2));
        let decoded_p1 = source_rows.to_f32();
        let rows: Vec<Vec<f32>> = decoded_p1
            .as_chunks::<3>()
            .0
            .iter()
            .map(|r| r.to_vec())
            .collect();
        let (want, _) = expected(RowCodec::Binary(&p2), &rows);
        assert_eq!(values, want);

        // Another kind: transcoded through f32.
        let (_, values) = append_into(&source, 4, RowCodec::F32);
        assert_eq!(values, &*decoded_p1);
    }

    #[test]
    fn test_unaligned_f32_bytes_are_decoded() {
        let values = [1.5f32, -2.0, 0.125];
        let mut bytes = vec![0u8];
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let unaligned = Cow::Borrowed(&bytes[1..]);
        let rows = MultiVectorRows::new(RowCodec::F32, 3, unaligned);
        // At most one of the two offsets can be 4-byte aligned, but
        // `to_f32` decodes correctly either way.
        assert_eq!(&*rows.to_f32(), &values);
    }

    #[test]
    fn test_empty_segment_round_trips() {
        let storage = memory();
        write(storage.as_ref(), "empty.mv", RowCodec::F32, 4, &[]);
        let reader = SegmentReader::open(storage.as_ref(), "empty.mv").unwrap();
        assert!(reader.doc_ids().is_empty());
        assert!(vectors(&reader, 1).is_none());
        reader.verify_payload().unwrap();
    }

    #[test]
    fn test_table_corruption_is_detected_on_open() {
        let storage = memory();
        write(storage.as_ref(), "seg.mv", RowCodec::F32, 3, &sample());
        let mut bytes = read_all(storage.as_ref(), "seg.mv");
        bytes[HEADER_LEN] ^= 0x01; // first doc id
        overwrite(storage.as_ref(), "seg.mv", &bytes);
        let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
    }

    /// Byte 6 holds the element kind; at dimension 2 an `F16` row set
    /// (`2 * dim` = 4 bytes/row) and an `Int8` one (`dim + 2` = 4
    /// bytes/row) are the same file length, so only this checksum tells
    /// them apart.
    #[test]
    fn test_element_kind_tamper_is_caught_by_the_table_checksum_even_at_matching_file_length() {
        let storage = memory();
        write(
            storage.as_ref(),
            "seg.mv",
            RowCodec::Int8,
            2,
            &[(1, vec![vec![1.0, 2.0]])],
        );
        let mut bytes = read_all(storage.as_ref(), "seg.mv");
        assert_eq!(bytes[6], MultiVectorStorage::Int8.tag());
        bytes[6] = MultiVectorStorage::F16.tag();
        overwrite(storage.as_ref(), "seg.mv", &bytes);
        let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
    }

    #[test]
    fn test_payload_corruption_is_detected_by_verify() {
        let storage = memory();
        write(storage.as_ref(), "seg.mv", RowCodec::F32, 3, &sample());
        let mut bytes = read_all(storage.as_ref(), "seg.mv");
        let payload_start = HEADER_LEN + 2 * DOC_ENTRY_LEN;
        bytes[payload_start] ^= 0x01;
        overwrite(storage.as_ref(), "seg.mv", &bytes);
        let reader = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap();
        let err = reader.verify_payload().unwrap_err();
        assert!(err.to_string().contains("payload checksum"), "{err}");
    }

    #[test]
    fn test_truncated_file_is_rejected() {
        let storage = memory();
        write(storage.as_ref(), "seg.mv", RowCodec::F32, 3, &sample());
        let bytes = read_all(storage.as_ref(), "seg.mv");
        overwrite(storage.as_ref(), "seg.mv", &bytes[..bytes.len() - 4]);
        let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
        assert!(err.to_string().contains("expected"), "{err}");
    }

    #[test]
    fn test_out_of_range_element_kinds_are_incompatible() {
        let storage = memory();
        write(storage.as_ref(), "seg.mv", RowCodec::F32, 3, &sample());
        let original = read_all(storage.as_ref(), "seg.mv");

        for bad_kind in [4u8, 255] {
            let mut bytes = original.clone();
            bytes[6] = bad_kind;
            overwrite(storage.as_ref(), "seg.mv", &bytes);
            let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
            assert!(matches!(err, LaurusError::IncompatibleFormat(_)), "{err}");
        }
    }

    #[test]
    fn test_bad_magic_is_incompatible() {
        let storage = memory();
        write(storage.as_ref(), "seg.mv", RowCodec::F32, 3, &sample());
        let mut bytes = read_all(storage.as_ref(), "seg.mv");
        bytes[0] = b'X';
        overwrite(storage.as_ref(), "seg.mv", &bytes);
        let err = SegmentReader::open(storage.as_ref(), "seg.mv").unwrap_err();
        assert!(matches!(err, LaurusError::IncompatibleFormat(_)), "{err}");
    }

    #[test]
    fn test_writer_rejects_bad_input_and_cleans_up() {
        let storage = memory();
        let unsorted = [
            DocEntry {
                doc_id: 2,
                vector_count: 1,
            },
            DocEntry {
                doc_id: 1,
                vector_count: 1,
            },
        ];
        assert!(
            write_segment(
                storage.as_ref(),
                "a.mv",
                RowCodec::F32,
                2,
                &unsorted,
                |_, _| Ok(())
            )
            .is_err()
        );

        let one = [DocEntry {
            doc_id: 1,
            vector_count: 2,
        }];
        let short = write_segment(
            storage.as_ref(),
            "b.mv",
            RowCodec::F32,
            2,
            &one,
            |_, sink| sink.write(&[1.0, 2.0]),
        );
        assert!(short.unwrap_err().to_string().contains("shorter"));
        let long = write_segment(
            storage.as_ref(),
            "c.mv",
            RowCodec::F32,
            2,
            &one,
            |_, sink| sink.write(&[1.0; 6]),
        );
        assert!(long.unwrap_err().to_string().contains("longer"));

        for name in ["a.mv", "b.mv", "c.mv"] {
            assert!(!storage.file_exists(name));
            assert!(!storage.file_exists(&format!("{name}.tmp")));
        }
    }

    #[test]
    fn test_writer_rejects_a_partial_row() {
        let storage = memory();
        let one = [DocEntry {
            doc_id: 1,
            vector_count: 2,
        }];
        // Dimension 2, but 3 values: not a whole number of rows.
        let err = write_segment(
            storage.as_ref(),
            "partial.mv",
            RowCodec::F32,
            2,
            &one,
            |_, sink| sink.write(&[1.0, 2.0, 3.0]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("whole number of rows"), "{err}");
    }
}
