//! Chunked, LZ4-compressed stored fields (Issue #548).
//!
//! Stores the original document field values (retrieved at search time,
//! separate from the inverted index used to search) chunked and
//! LZ4-compressed, following the precedent set by Lucene's
//! `Lucene90StoredFieldsFormat` (chunk size 16 KiB by default) and Tantivy's
//! `StoreWriter`.
//!
//! ## The `.docs` segment part
//!
//! ```text
//! magic "SDOC"(4B) | version_major(1B) | version_minor(1B)
//! total_doc_count(varint)
//! repeat chunks until total_doc_count documents have been written:
//!   chunk_doc_count(varint) | codec(1B: 0=raw, 1=lz4) | uncompressed_size(varint)
//!   | payload_size(varint) | crc32(payload)(4B) | payload(payload_size bytes)
//!   -- payload decompresses (codec=1) or is (codec=0) chunk_doc_count documents:
//!   { doc_id(u64) | field_count(varint)
//!     | { name(string) | type_tag(1B) | tag-specific payload } * field_count } * chunk_doc_count
//! -- trailer: u32 CRC-32 checksum (StructWriter::close) --
//! ```
//!
//! Design notes:
//!
//! - **No chunk index / random-access seeking.** The reader ([`StoredFieldsReader::load`])
//!   decodes every chunk sequentially into an in-memory `BTreeMap<u64, Document>`, exactly
//!   matching the pre-#548 format's "decode the whole segment once" contract (locked in by
//!   the `stored_documents_decode_once_per_segment` test, Issue #994). Nothing in this codebase
//!   ever seeks to an individual chunk, so the Lucene-style `doc_id -> chunk_offset` directory
//!   this issue's suggested format described has no consumer and is intentionally omitted. A
//!   directory can be added later as a pure tail-append without touching this layout.
//! - **Documents and field names are sorted before encoding.** `AnalyzedDocument::stored_fields`
//!   is an `AHashMap` (hash-seed-dependent iteration order), so sorting makes the on-disk bytes
//!   deterministic and improves LZ4's ability to match repeated byte sequences (e.g. identical
//!   field-name sequences) across adjacent documents in the same chunk.
//! - **Per-chunk raw fallback (`codec = 0`).** If LZ4 does not shrink a chunk (e.g. it is already
//!   compressed binary data), the chunk is stored uncompressed instead — this is what makes a
//!   separate "disable compression" configuration knob unnecessary.
//! - **Bounds-checking a compressed format needs different math than an uncompressed one.**
//!   `payload_size` (compressed) is bounded against the bytes actually left in the file, but
//!   `uncompressed_size` must NOT be bounded that way — a well-compressed chunk's declared
//!   uncompressed size is, by definition, larger than its on-disk footprint. Instead it is
//!   bounded against LZ4's documented maximum expansion ratio (255x the compressed size).
//!   `chunk_doc_count` is bounded against `uncompressed_size` (not the file), since that is the
//!   space it actually has to fit in once decompressed.
//! - **Format break, no legacy reader.** A missing/mismatched magic is rejected outright
//!   (mirroring `bkd_tree`'s "pre-release, format changes do not support older revisions"
//!   policy) rather than keeping a second, permanently-maintained decoder for the old
//!   uncompressed layout — which, unlike `.norms`'s `.lens`/`.fstats` migration, could not even
//!   be made fully correct: the pre-#548 format has two independent decode bugs of its own (a
//!   stored `Bytes` field desyncs every later field in the segment; a stored `Vector` field
//!   always errored) that this rewrite fixes. The separate `.json`-mirror legacy fallback
//!   (pre-#756 segments, in `SegmentReader::load_stored_documents`) is unrelated and unaffected.

use std::collections::BTreeMap;

use ahash::AHashMap;

use crate::data::{DataValue, Document, GeoEcefPoint, GeoPoint};
use crate::error::{LaurusError, Result};
use crate::lexical::core::analyzed::AnalyzedDocument;
use crate::storage::structured::{StructReader, StructWriter};
use crate::storage::{StorageInput, StorageOutput};
use crate::util::alloc_bounds::{checked_capacity, checked_len};
use crate::util::varint::{read_varint, write_varint};

const MAGIC: &[u8; 4] = b"SDOC";
const VERSION_MAJOR: u8 = 1;
const VERSION_MINOR: u8 = 0;

const CODEC_RAW: u8 = 0;
const CODEC_LZ4: u8 = 1;

/// Target uncompressed size (bytes) before a chunk is flushed — matches
/// Lucene's `Lucene90StoredFieldsFormat` default.
const CHUNK_TARGET_BYTES: usize = 16 * 1024;

/// Maximum documents per chunk regardless of byte size (matches Lucene's
/// `BEST_SPEED` cap), so a corpus of many tiny documents doesn't produce
/// unboundedly large chunks.
const MAX_DOCS_PER_CHUNK: usize = 128;

/// LZ4's documented maximum expansion ratio: a length-extension byte can
/// encode a multiplier of at most 255, so `uncompressed_size` can never
/// legitimately exceed `payload_size * 255`. Used to bound a header-declared
/// `uncompressed_size` against `payload_size` instead of against remaining
/// file bytes (see module docs).
const LZ4_MAX_EXPANSION_RATIO: u64 = 255;

/// Minimum on-disk bytes one document can occupy once decompressed: an
/// 8-byte doc id plus a 1-byte varint field count (zero fields).
const MIN_DOC_RECORD_SIZE: u64 = 8 + 1;

// ---- Type tags (unchanged from the pre-#548 format, for on-disk stability
// of anything that inspects raw tag values) -------------------------------

const TAG_TEXT: u8 = 0;
const TAG_INT64: u8 = 1;
const TAG_FLOAT64: u8 = 2;
const TAG_BOOL: u8 = 3;
const TAG_BYTES: u8 = 4;
const TAG_DATETIME: u8 = 5;
const TAG_GEO: u8 = 6;
const TAG_NULL: u8 = 7;
const TAG_VECTOR: u8 = 9;
const TAG_INT64_ARRAY: u8 = 10;
const TAG_FLOAT64_ARRAY: u8 = 11;
const TAG_GEO_ECEF: u8 = 12;
// Multi-valued geo (#1174). Tag 8 is skipped for historical reasons and is
// deliberately not reused. Like tags 10–12 these were added without a
// format-version bump: a pre-#1174 reader rejects a segment containing them
// with `Unknown field type tag` (loud, recorded on Issue #1040).
const TAG_GEO_ARRAY: u8 = 13;
const TAG_GEO_ECEF_ARRAY: u8 = 14;
// Multi-valued datetime (#1184): varint length + i64 LE Unix micro-seconds
// per element — the persisted precision of a `DataValue::DateTime` (rkyv
// `MicroSeconds`) and the proto representation. Unlike the scalar tag 5,
// which stores RFC 3339 text, sub-microsecond detail is truncated. Same
// no-version-bump policy as tags 10–14 (recorded on Issue #1040).
const TAG_DATETIME_ARRAY: u8 = 15;
// Multi-valued boolean (#1180): varint length + one byte per element (`0` /
// `1`, the same representation as the scalar tag 3 — not bit-packed). Same
// no-version-bump policy as tags 10–15 (recorded on Issue #1040).
const TAG_BOOL_ARRAY: u8 = 16;
// Multi-valued text (#1175): varint element count + per element the same
// varint-length-prefixed UTF-8 body the scalar tag 0 uses. Same
// no-version-bump policy as tags 10–16 (recorded on Issue #1040).
const TAG_TEXT_ARRAY: u8 = 17;

// ---------------------------------------------------------------------------
// Encoding (document -> plain bytes, before compression)
// ---------------------------------------------------------------------------

/// Encode one document's fields into `buf`, sorted by field name for
/// deterministic output and better LZ4 matching across adjacent documents.
fn encode_document(buf: &mut Vec<u8>, doc_id: u64, fields: &AHashMap<String, DataValue>) {
    buf.extend_from_slice(&doc_id.to_le_bytes());
    write_varint(buf, fields.len() as u64);

    let mut names: Vec<&String> = fields.keys().collect();
    names.sort_unstable();

    for name in names {
        write_varint(buf, name.len() as u64);
        buf.extend_from_slice(name.as_bytes());

        match &fields[name] {
            DataValue::Text(text) => {
                buf.push(TAG_TEXT);
                write_varint(buf, text.len() as u64);
                buf.extend_from_slice(text.as_bytes());
            }
            DataValue::Int64(num) => {
                buf.push(TAG_INT64);
                buf.extend_from_slice(&(*num as u64).to_le_bytes());
            }
            DataValue::Float64(num) => {
                buf.push(TAG_FLOAT64);
                buf.extend_from_slice(&num.to_le_bytes());
            }
            DataValue::Bool(b) => {
                buf.push(TAG_BOOL);
                buf.push(u8::from(*b));
            }
            DataValue::DateTime(dt) => {
                buf.push(TAG_DATETIME);
                let s = dt.to_rfc3339();
                write_varint(buf, s.len() as u64);
                buf.extend_from_slice(s.as_bytes());
            }
            DataValue::Geo(p) => {
                buf.push(TAG_GEO);
                buf.extend_from_slice(&p.lat.to_le_bytes());
                buf.extend_from_slice(&p.lon.to_le_bytes());
            }
            DataValue::GeoEcef(p) => {
                buf.push(TAG_GEO_ECEF);
                buf.extend_from_slice(&p.x.to_le_bytes());
                buf.extend_from_slice(&p.y.to_le_bytes());
                buf.extend_from_slice(&p.z.to_le_bytes());
            }
            DataValue::Bytes(bytes, mime) => {
                buf.push(TAG_BYTES);
                let mime_str = mime.as_deref().unwrap_or("");
                write_varint(buf, mime_str.len() as u64);
                buf.extend_from_slice(mime_str.as_bytes());
                // Single length prefix (Issue #548 fixes the pre-existing
                // bug where the writer emitted this length twice).
                write_varint(buf, bytes.len() as u64);
                buf.extend_from_slice(bytes);
            }
            DataValue::Null => {
                buf.push(TAG_NULL);
            }
            DataValue::Vector(v) => {
                buf.push(TAG_VECTOR);
                write_varint(buf, v.len() as u64);
                for &f in v {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
            DataValue::Int64Array(arr) => {
                buf.push(TAG_INT64_ARRAY);
                write_varint(buf, arr.len() as u64);
                for &v in arr {
                    buf.extend_from_slice(&(v as u64).to_le_bytes());
                }
            }
            DataValue::Float64Array(arr) => {
                buf.push(TAG_FLOAT64_ARRAY);
                write_varint(buf, arr.len() as u64);
                for &v in arr {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            DataValue::GeoArray(arr) => {
                buf.push(TAG_GEO_ARRAY);
                write_varint(buf, arr.len() as u64);
                for p in arr {
                    buf.extend_from_slice(&p.lat.to_le_bytes());
                    buf.extend_from_slice(&p.lon.to_le_bytes());
                }
            }
            DataValue::GeoEcefArray(arr) => {
                buf.push(TAG_GEO_ECEF_ARRAY);
                write_varint(buf, arr.len() as u64);
                for p in arr {
                    buf.extend_from_slice(&p.x.to_le_bytes());
                    buf.extend_from_slice(&p.y.to_le_bytes());
                    buf.extend_from_slice(&p.z.to_le_bytes());
                }
            }
            DataValue::DateTimeArray(arr) => {
                buf.push(TAG_DATETIME_ARRAY);
                write_varint(buf, arr.len() as u64);
                for dt in arr {
                    buf.extend_from_slice(&dt.timestamp_micros().to_le_bytes());
                }
            }
            DataValue::BoolArray(arr) => {
                buf.push(TAG_BOOL_ARRAY);
                write_varint(buf, arr.len() as u64);
                for &b in arr {
                    buf.push(u8::from(b));
                }
            }
            DataValue::TextArray(arr) => {
                buf.push(TAG_TEXT_ARRAY);
                write_varint(buf, arr.len() as u64);
                for text in arr {
                    write_varint(buf, text.len() as u64);
                    buf.extend_from_slice(text.as_bytes());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decoding (plain bytes, after decompression -> document)
// ---------------------------------------------------------------------------

fn read_u8(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u8> {
    let b = *bytes
        .get(*cursor)
        .ok_or_else(|| LaurusError::index(format!("{what}: truncated")))?;
    *cursor += 1;
    Ok(b)
}

fn read_fixed<const N: usize>(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<[u8; N]> {
    let end = *cursor + N;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| LaurusError::index(format!("{what}: truncated")))?;
    *cursor = end;
    Ok(slice
        .try_into()
        .expect("slice length matches N by construction"))
}

fn read_u64_le(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u64> {
    Ok(u64::from_le_bytes(read_fixed::<8>(bytes, cursor, what)?))
}

fn read_f32_le(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<f32> {
    Ok(f32::from_le_bytes(read_fixed::<4>(bytes, cursor, what)?))
}

fn read_f64_le(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<f64> {
    Ok(f64::from_le_bytes(read_fixed::<8>(bytes, cursor, what)?))
}

fn read_len_prefixed_str(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<String> {
    let len = read_varint(bytes, cursor, what)? as usize;
    let len = checked_len(len, (bytes.len() - *cursor) as u64, what)?;
    let slice = bytes
        .get(*cursor..*cursor + len)
        .ok_or_else(|| LaurusError::index(format!("{what}: truncated")))?;
    *cursor += len;
    String::from_utf8(slice.to_vec())
        .map_err(|e| LaurusError::index(format!("{what}: invalid utf-8: {e}")))
}

fn read_len_prefixed_bytes(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<Vec<u8>> {
    let len = read_varint(bytes, cursor, what)? as usize;
    let len = checked_len(len, (bytes.len() - *cursor) as u64, what)?;
    let slice = bytes
        .get(*cursor..*cursor + len)
        .ok_or_else(|| LaurusError::index(format!("{what}: truncated")))?;
    *cursor += len;
    Ok(slice.to_vec())
}

/// Decode one document from `bytes` starting at `*cursor`, advancing
/// `*cursor` past the bytes consumed.
fn decode_document(bytes: &[u8], cursor: &mut usize) -> Result<(u64, Document)> {
    let doc_id = read_u64_le(bytes, cursor, "stored-fields document id")?;
    let field_count = read_varint(bytes, cursor, "stored-fields field count")? as usize;
    let field_count = checked_capacity(
        field_count,
        2,
        (bytes.len() - *cursor) as u64,
        "stored-fields field count",
    )?;

    let mut doc = Document::new();
    for _ in 0..field_count {
        let name = read_len_prefixed_str(bytes, cursor, "stored-fields field name")?;
        let tag = read_u8(bytes, cursor, "stored-fields type tag")?;
        let value = match tag {
            TAG_TEXT => DataValue::Text(read_len_prefixed_str(bytes, cursor, "stored Text field")?),
            TAG_INT64 => DataValue::Int64(read_u64_le(bytes, cursor, "stored Int64 field")? as i64),
            TAG_FLOAT64 => DataValue::Float64(read_f64_le(bytes, cursor, "stored Float64 field")?),
            TAG_BOOL => DataValue::Bool(read_u8(bytes, cursor, "stored Bool field")? != 0),
            TAG_BYTES => {
                let mime = read_len_prefixed_str(bytes, cursor, "stored Bytes field mime")?;
                let data = read_len_prefixed_bytes(bytes, cursor, "stored Bytes field data")?;
                DataValue::Bytes(data, if mime.is_empty() { None } else { Some(mime) })
            }
            TAG_DATETIME => {
                let s = read_len_prefixed_str(bytes, cursor, "stored DateTime field")?;
                let dt = chrono::DateTime::parse_from_rfc3339(&s)
                    .map_err(|e| LaurusError::index(format!("Failed to parse DateTime: {e}")))?
                    .with_timezone(&chrono::Utc);
                DataValue::DateTime(dt)
            }
            TAG_GEO => {
                let lat = read_f64_le(bytes, cursor, "stored Geo field lat")?;
                let lon = read_f64_le(bytes, cursor, "stored Geo field lon")?;
                DataValue::Geo(GeoPoint::new(lat, lon))
            }
            TAG_GEO_ECEF => {
                let x = read_f64_le(bytes, cursor, "stored GeoEcef field x")?;
                let y = read_f64_le(bytes, cursor, "stored GeoEcef field y")?;
                let z = read_f64_le(bytes, cursor, "stored GeoEcef field z")?;
                DataValue::GeoEcef(GeoEcefPoint::new(x, y, z))
            }
            TAG_NULL => DataValue::Null,
            TAG_VECTOR => {
                let len = read_varint(bytes, cursor, "stored Vector field length")? as usize;
                let len = checked_capacity(
                    len,
                    4,
                    (bytes.len() - *cursor) as u64,
                    "stored Vector field length",
                )?;
                let mut v = Vec::with_capacity(len);
                for _ in 0..len {
                    v.push(read_f32_le(bytes, cursor, "stored Vector field element")?);
                }
                DataValue::Vector(v)
            }
            TAG_INT64_ARRAY => {
                let len = read_varint(bytes, cursor, "stored Int64Array field length")? as usize;
                let len = checked_capacity(
                    len,
                    8,
                    (bytes.len() - *cursor) as u64,
                    "stored Int64Array field length",
                )?;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_u64_le(bytes, cursor, "stored Int64Array field element")? as i64);
                }
                DataValue::Int64Array(arr)
            }
            TAG_FLOAT64_ARRAY => {
                let len = read_varint(bytes, cursor, "stored Float64Array field length")? as usize;
                let len = checked_capacity(
                    len,
                    8,
                    (bytes.len() - *cursor) as u64,
                    "stored Float64Array field length",
                )?;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_f64_le(
                        bytes,
                        cursor,
                        "stored Float64Array field element",
                    )?);
                }
                DataValue::Float64Array(arr)
            }
            TAG_GEO_ARRAY => {
                let len = read_varint(bytes, cursor, "stored GeoArray field length")? as usize;
                let len = checked_capacity(
                    len,
                    16,
                    (bytes.len() - *cursor) as u64,
                    "stored GeoArray field length",
                )?;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    let lat = read_f64_le(bytes, cursor, "stored GeoArray field lat")?;
                    let lon = read_f64_le(bytes, cursor, "stored GeoArray field lon")?;
                    arr.push(GeoPoint::new(lat, lon));
                }
                DataValue::GeoArray(arr)
            }
            TAG_GEO_ECEF_ARRAY => {
                let len = read_varint(bytes, cursor, "stored GeoEcefArray field length")? as usize;
                let len = checked_capacity(
                    len,
                    24,
                    (bytes.len() - *cursor) as u64,
                    "stored GeoEcefArray field length",
                )?;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    let x = read_f64_le(bytes, cursor, "stored GeoEcefArray field x")?;
                    let y = read_f64_le(bytes, cursor, "stored GeoEcefArray field y")?;
                    let z = read_f64_le(bytes, cursor, "stored GeoEcefArray field z")?;
                    arr.push(GeoEcefPoint::new(x, y, z));
                }
                DataValue::GeoEcefArray(arr)
            }
            TAG_DATETIME_ARRAY => {
                let len = read_varint(bytes, cursor, "stored DateTimeArray field length")? as usize;
                let len = checked_capacity(
                    len,
                    8,
                    (bytes.len() - *cursor) as u64,
                    "stored DateTimeArray field length",
                )?;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    let micros =
                        read_u64_le(bytes, cursor, "stored DateTimeArray field element")? as i64;
                    // Every value the encoder writes is representable; a
                    // failure here is corruption and must not decode as the
                    // epoch silently.
                    let dt = chrono::DateTime::from_timestamp_micros(micros).ok_or_else(|| {
                        LaurusError::index(format!(
                            "stored DateTimeArray element out of range: {micros} µs"
                        ))
                    })?;
                    arr.push(dt);
                }
                DataValue::DateTimeArray(arr)
            }
            TAG_BOOL_ARRAY => {
                let len = read_varint(bytes, cursor, "stored BoolArray field length")? as usize;
                let len = checked_capacity(
                    len,
                    1,
                    (bytes.len() - *cursor) as u64,
                    "stored BoolArray field length",
                )?;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_u8(bytes, cursor, "stored BoolArray field element")? != 0);
                }
                DataValue::BoolArray(arr)
            }
            TAG_TEXT_ARRAY => {
                let len = read_varint(bytes, cursor, "stored TextArray field length")? as usize;
                // Elements are variable-width, so the only honest bound is
                // the 1-byte minimum a zero-length element occupies.
                let len = checked_capacity(
                    len,
                    1,
                    (bytes.len() - *cursor) as u64,
                    "stored TextArray field length",
                )?;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_len_prefixed_str(
                        bytes,
                        cursor,
                        "stored TextArray field element",
                    )?);
                }
                DataValue::TextArray(arr)
            }
            other => {
                return Err(LaurusError::index(format!(
                    "Unknown field type tag: {other}"
                )));
            }
        };
        doc.fields.insert(name, value);
    }
    Ok((doc_id, doc))
}

// ---------------------------------------------------------------------------
// StoredFieldsWriter
// ---------------------------------------------------------------------------

pub(crate) struct StoredFieldsWriter;

impl StoredFieldsWriter {
    /// Write `docs` to `writer` in the chunked, LZ4-compressed `.docs`
    /// format described in the module docs.
    pub(crate) fn write_to<W: StorageOutput>(
        writer: &mut StructWriter<W>,
        docs: &[(u64, AnalyzedDocument)],
    ) -> Result<()> {
        writer.write_raw(MAGIC)?;
        writer.write_raw(&[VERSION_MAJOR, VERSION_MINOR])?;
        writer.write_varint(docs.len() as u64)?;

        // Ascending doc-id order (see module docs): deterministic output,
        // better LZ4 matches, and keeps a future chunk directory retrofit
        // simple. Does NOT reorder `docs` itself — only this local index —
        // since the caller's `buffered_docs` is shared with norms/DocValues/
        // posting construction.
        let mut order: Vec<usize> = (0..docs.len()).collect();
        order.sort_unstable_by_key(|&i| docs[i].0);

        let mut chunk_buf = Vec::new();
        let mut chunk_doc_count = 0usize;
        let mut compress_buf = Vec::new();

        for (seen, &i) in order.iter().enumerate() {
            let (doc_id, doc) = (docs[i].0, &docs[i].1);
            encode_document(&mut chunk_buf, doc_id, &doc.stored_fields);
            chunk_doc_count += 1;

            let is_last = seen + 1 == order.len();
            if is_last
                || chunk_buf.len() >= CHUNK_TARGET_BYTES
                || chunk_doc_count >= MAX_DOCS_PER_CHUNK
            {
                Self::flush_chunk(writer, &chunk_buf, chunk_doc_count, &mut compress_buf)?;
                chunk_buf.clear();
                chunk_doc_count = 0;
            }
        }

        Ok(())
    }

    fn flush_chunk<W: StorageOutput>(
        writer: &mut StructWriter<W>,
        uncompressed: &[u8],
        doc_count: usize,
        compress_buf: &mut Vec<u8>,
    ) -> Result<()> {
        let max_compressed = lz4_flex::block::get_maximum_output_size(uncompressed.len());
        compress_buf.resize(max_compressed, 0);
        let compressed_len = lz4_flex::compress_into(uncompressed, compress_buf).map_err(|e| {
            LaurusError::index(format!("stored-fields LZ4 compression failed: {e}"))
        })?;

        // Raw fallback: never let compression make the chunk bigger (e.g.
        // already-compressed Bytes payloads).
        let (codec, payload): (u8, &[u8]) = if compressed_len < uncompressed.len() {
            (CODEC_LZ4, &compress_buf[..compressed_len])
        } else {
            (CODEC_RAW, uncompressed)
        };

        writer.write_varint(doc_count as u64)?;
        writer.write_u8(codec)?;
        writer.write_varint(uncompressed.len() as u64)?;
        writer.write_varint(payload.len() as u64)?;
        writer.write_u32(crc32fast::hash(payload))?;
        writer.write_raw(payload)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// StoredFieldsReader
// ---------------------------------------------------------------------------

pub(crate) struct StoredFieldsReader;

impl StoredFieldsReader {
    /// Read the chunked `.docs` format from `reader`, decoding every chunk
    /// into an in-memory map (see module docs for why this stays eager
    /// rather than lazy/random-access).
    ///
    /// # Errors
    ///
    /// Returns an error if the magic/version don't match (format break, no
    /// legacy support — see module docs), or if any chunk fails its CRC,
    /// bounds, or decompression checks.
    pub(crate) fn load<R: StorageInput>(
        reader: &mut StructReader<R>,
    ) -> Result<BTreeMap<u64, Document>> {
        let magic: [u8; 4] = reader
            .read_raw(4)?
            .try_into()
            .map_err(|_| LaurusError::index("stored-fields header: truncated magic"))?;
        if &magic != MAGIC {
            return Err(LaurusError::index(
                "Unsupported stored-fields format (expected SDOC). Pre-release format changes \
                 do not support older revisions; rebuild the index.",
            ));
        }
        let version_major = reader.read_u8()?;
        let _version_minor = reader.read_u8()?;
        if version_major != VERSION_MAJOR {
            return Err(LaurusError::index(format!(
                "Unsupported stored-fields version: {version_major} (expected {VERSION_MAJOR}). \
                 Pre-release format changes do not support older revisions; rebuild the index."
            )));
        }

        let total_doc_count = reader.read_varint()?;
        let mut documents = BTreeMap::new();
        let mut decoded_count = 0u64;
        let mut scratch = Vec::new();

        while decoded_count < total_doc_count {
            let file_size = reader.size();
            let position = reader.position();
            let available = file_size.saturating_sub(position);

            let chunk_doc_count = reader.read_varint()? as usize;
            let codec = reader.read_u8()?;
            let uncompressed_size = reader.read_varint()? as usize;
            let payload_size = reader.read_varint()? as usize;
            let stored_crc = reader.read_u32()?;

            // `payload_size` is bounded against the file: compressed bytes
            // must physically exist. `uncompressed_size` must NOT be bounded
            // against the file — a compressed chunk's declared uncompressed
            // size is larger than its on-disk footprint by design. Bound it
            // against LZ4's documented maximum expansion ratio instead.
            let payload_size = checked_len(payload_size, available, "stored-fields chunk payload")?;
            let max_uncompressed = (payload_size as u64).saturating_mul(LZ4_MAX_EXPANSION_RATIO);
            if uncompressed_size as u64 > max_uncompressed.max(payload_size as u64) {
                return Err(LaurusError::index(format!(
                    "stored-fields chunk: declares {uncompressed_size} uncompressed bytes, \
                     impossible for a {payload_size}-byte payload — segment is corrupted"
                )));
            }
            let chunk_doc_count = checked_capacity(
                chunk_doc_count,
                MIN_DOC_RECORD_SIZE,
                uncompressed_size as u64,
                "stored-fields chunk doc count",
            )?;

            let payload = reader.read_raw(payload_size)?;
            let actual_crc = crc32fast::hash(&payload);
            if actual_crc != stored_crc {
                return Err(LaurusError::index(
                    "stored-fields chunk: CRC-32 mismatch — segment is corrupted",
                ));
            }

            scratch.clear();
            let chunk_bytes: &[u8] = match codec {
                CODEC_RAW => &payload,
                CODEC_LZ4 => {
                    scratch.resize(uncompressed_size, 0);
                    let written =
                        lz4_flex::decompress_into(&payload, &mut scratch).map_err(|e| {
                            LaurusError::index(format!(
                                "stored-fields chunk: LZ4 decompression failed: {e}"
                            ))
                        })?;
                    if written != uncompressed_size {
                        return Err(LaurusError::index(format!(
                            "stored-fields chunk: expected {uncompressed_size} decompressed \
                             bytes, got {written} — segment is corrupted"
                        )));
                    }
                    &scratch
                }
                other => {
                    return Err(LaurusError::index(format!(
                        "stored-fields chunk: unknown codec {other}"
                    )));
                }
            };

            let mut cursor = 0usize;
            for _ in 0..chunk_doc_count {
                let (doc_id, doc) = decode_document(chunk_bytes, &mut cursor)?;
                documents.insert(doc_id, doc);
            }
            decoded_count += chunk_doc_count as u64;
        }

        if decoded_count != total_doc_count {
            return Err(LaurusError::index(format!(
                "stored-fields: header declares {total_doc_count} documents but chunks totalled \
                 {decoded_count} — segment is corrupted"
            )));
        }

        Ok(documents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn doc(fields: &[(&str, DataValue)]) -> AnalyzedDocument {
        let mut d = AnalyzedDocument::new();
        for (name, value) in fields {
            d.stored_fields.insert((*name).to_string(), value.clone());
        }
        d
    }

    fn round_trip(
        storage: &MemoryStorage,
        segment_id: &str,
        docs: &[(u64, AnalyzedDocument)],
    ) -> BTreeMap<u64, Document> {
        let output = storage
            .create_output(&format!("{segment_id}.docs"))
            .unwrap();
        let mut writer = StructWriter::new(output);
        StoredFieldsWriter::write_to(&mut writer, docs).unwrap();
        writer.close().unwrap();
        load_docs_file(storage, segment_id).unwrap()
    }

    fn load_docs_file(
        storage: &MemoryStorage,
        segment_id: &str,
    ) -> Result<BTreeMap<u64, Document>> {
        let input = storage.open_input(&format!("{segment_id}.docs")).unwrap();
        let mut reader = StructReader::new(input)?;
        StoredFieldsReader::load(&mut reader)
    }

    /// Test-only: count how many chunk records a written `.docs` blob
    /// contains, without fully decoding it — proves chunking actually
    /// happened (an implementation detail the round-trip output alone
    /// can't observe).
    fn count_chunks(storage: &MemoryStorage, segment_id: &str) -> usize {
        let input = storage.open_input(&format!("{segment_id}.docs")).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        reader.read_raw(4).unwrap();
        reader.read_u8().unwrap();
        reader.read_u8().unwrap();
        let total_doc_count = reader.read_varint().unwrap();
        let mut decoded = 0u64;
        let mut chunks = 0usize;
        while decoded < total_doc_count {
            let doc_count = reader.read_varint().unwrap();
            let _codec = reader.read_u8().unwrap();
            let _uncompressed_size = reader.read_varint().unwrap();
            let payload_size = reader.read_varint().unwrap();
            let _crc = reader.read_u32().unwrap();
            reader.read_raw(payload_size as usize).unwrap();
            decoded += doc_count;
            chunks += 1;
        }
        chunks
    }

    fn first_chunk_codec(storage: &MemoryStorage, segment_id: &str) -> u8 {
        let input = storage.open_input(&format!("{segment_id}.docs")).unwrap();
        let mut reader = StructReader::new(input).unwrap();
        reader.read_raw(4).unwrap();
        reader.read_u8().unwrap();
        reader.read_u8().unwrap();
        reader.read_varint().unwrap(); // total_doc_count
        reader.read_varint().unwrap(); // chunk_doc_count
        reader.read_u8().unwrap() // codec
    }

    #[allow(clippy::too_many_arguments)]
    fn write_single_chunk_file(
        storage: &MemoryStorage,
        segment_id: &str,
        total_doc_count: u64,
        chunk_doc_count: u64,
        codec: u8,
        uncompressed_size: u64,
        payload: &[u8],
        crc: u32,
    ) {
        let output = storage
            .create_output(&format!("{segment_id}.docs"))
            .unwrap();
        let mut writer = StructWriter::new(output);
        writer.write_raw(MAGIC).unwrap();
        writer.write_raw(&[VERSION_MAJOR, VERSION_MINOR]).unwrap();
        writer.write_varint(total_doc_count).unwrap();
        writer.write_varint(chunk_doc_count).unwrap();
        writer.write_u8(codec).unwrap();
        writer.write_varint(uncompressed_size).unwrap();
        writer.write_varint(payload.len() as u64).unwrap();
        writer.write_u32(crc).unwrap();
        writer.write_raw(payload).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn round_trips_every_data_value_variant() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        // A field after `Bytes` is essential: the pre-#548 double-length
        // bug desynced every field that came after a stored Bytes field.
        let docs = vec![(
            1u64,
            doc(&[
                ("a_text", DataValue::Text("hello".to_string())),
                ("b_int", DataValue::Int64(-42)),
                ("c_float", DataValue::Float64(3.5)),
                ("d_bool", DataValue::Bool(true)),
                (
                    "e_bytes",
                    DataValue::Bytes(vec![1, 2, 3, 4], Some("application/x-test".to_string())),
                ),
                ("f_bytes_no_mime", DataValue::Bytes(vec![9, 9], None)),
                (
                    "g_datetime",
                    DataValue::DateTime(
                        chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
                    ),
                ),
                ("h_geo", DataValue::Geo(GeoPoint::new(35.6, 139.7))),
                (
                    "i_geo_ecef",
                    DataValue::GeoEcef(GeoEcefPoint::new(1.0, 2.0, 3.0)),
                ),
                ("j_null", DataValue::Null),
                ("k_vector", DataValue::Vector(vec![0.1, 0.2, 0.3])),
                ("l_int_array", DataValue::Int64Array(vec![1, 2, 3])),
                ("m_float_array", DataValue::Float64Array(vec![1.5, 2.5])),
                (
                    "n_geo_array",
                    DataValue::GeoArray(vec![
                        GeoPoint::new(35.6, 139.7),
                        GeoPoint::new(-33.9, 151.2),
                    ]),
                ),
                (
                    "o_geo_ecef_array",
                    DataValue::GeoEcefArray(vec![
                        GeoEcefPoint::new(1.0, 2.0, 3.0),
                        GeoEcefPoint::new(-4.0, 5.0, -6.0),
                    ]),
                ),
                // Empty point lists take the length-0 path (#1174).
                ("p_geo_array_empty", DataValue::GeoArray(Vec::new())),
                (
                    "q_geo_ecef_array_empty",
                    DataValue::GeoEcefArray(Vec::new()),
                ),
                // Multi-valued datetimes (#1184): micro-second precision,
                // pre-1970 instants, and the empty list.
                (
                    "r_datetime_array",
                    DataValue::DateTimeArray(vec![
                        chrono::DateTime::from_timestamp_micros(1_700_000_000_500_000).unwrap(),
                        chrono::DateTime::from_timestamp_micros(-86_400_000_001).unwrap(),
                    ]),
                ),
                (
                    "s_datetime_array_empty",
                    DataValue::DateTimeArray(Vec::new()),
                ),
                // Multi-valued booleans (#1180): one byte per element, and
                // the empty list.
                (
                    "t_bool_array",
                    DataValue::BoolArray(vec![true, false, true]),
                ),
                ("u_bool_array_empty", DataValue::BoolArray(Vec::new())),
                // Multi-valued text (#1175): a long element, an empty one
                // and multi-byte UTF-8, plus the empty list.
                (
                    "v_text_array",
                    DataValue::TextArray(vec![
                        "hello world".to_string(),
                        String::new(),
                        "日本語のテキスト".to_string(),
                    ]),
                ),
                ("w_text_array_empty", DataValue::TextArray(Vec::new())),
            ]),
        )];

        let documents = round_trip(&storage, "seg", &docs);
        let d = &documents[&1];
        assert_eq!(
            d.fields.get("a_text"),
            Some(&DataValue::Text("hello".to_string()))
        );
        assert_eq!(d.fields.get("b_int"), Some(&DataValue::Int64(-42)));
        assert_eq!(d.fields.get("c_float"), Some(&DataValue::Float64(3.5)));
        assert_eq!(d.fields.get("d_bool"), Some(&DataValue::Bool(true)));
        assert_eq!(
            d.fields.get("e_bytes"),
            Some(&DataValue::Bytes(
                vec![1, 2, 3, 4],
                Some("application/x-test".to_string())
            ))
        );
        assert_eq!(
            d.fields.get("f_bytes_no_mime"),
            Some(&DataValue::Bytes(vec![9, 9], None))
        );
        assert!(matches!(
            d.fields.get("g_datetime"),
            Some(DataValue::DateTime(_))
        ));
        assert_eq!(
            d.fields.get("h_geo"),
            Some(&DataValue::Geo(GeoPoint::new(35.6, 139.7)))
        );
        assert_eq!(
            d.fields.get("i_geo_ecef"),
            Some(&DataValue::GeoEcef(GeoEcefPoint::new(1.0, 2.0, 3.0)))
        );
        assert_eq!(d.fields.get("j_null"), Some(&DataValue::Null));
        assert_eq!(
            d.fields.get("k_vector"),
            Some(&DataValue::Vector(vec![0.1, 0.2, 0.3]))
        );
        assert_eq!(
            d.fields.get("l_int_array"),
            Some(&DataValue::Int64Array(vec![1, 2, 3]))
        );
        assert_eq!(
            d.fields.get("m_float_array"),
            Some(&DataValue::Float64Array(vec![1.5, 2.5]))
        );
        assert_eq!(
            d.fields.get("n_geo_array"),
            Some(&DataValue::GeoArray(vec![
                GeoPoint::new(35.6, 139.7),
                GeoPoint::new(-33.9, 151.2),
            ]))
        );
        assert_eq!(
            d.fields.get("o_geo_ecef_array"),
            Some(&DataValue::GeoEcefArray(vec![
                GeoEcefPoint::new(1.0, 2.0, 3.0),
                GeoEcefPoint::new(-4.0, 5.0, -6.0),
            ]))
        );
        assert_eq!(
            d.fields.get("p_geo_array_empty"),
            Some(&DataValue::GeoArray(Vec::new()))
        );
        assert_eq!(
            d.fields.get("q_geo_ecef_array_empty"),
            Some(&DataValue::GeoEcefArray(Vec::new()))
        );
        assert_eq!(
            d.fields.get("r_datetime_array"),
            Some(&DataValue::DateTimeArray(vec![
                chrono::DateTime::from_timestamp_micros(1_700_000_000_500_000).unwrap(),
                chrono::DateTime::from_timestamp_micros(-86_400_000_001).unwrap(),
            ]))
        );
        assert_eq!(
            d.fields.get("s_datetime_array_empty"),
            Some(&DataValue::DateTimeArray(Vec::new()))
        );
        assert_eq!(
            d.fields.get("t_bool_array"),
            Some(&DataValue::BoolArray(vec![true, false, true]))
        );
        assert_eq!(
            d.fields.get("u_bool_array_empty"),
            Some(&DataValue::BoolArray(Vec::new()))
        );
        assert_eq!(
            d.fields.get("v_text_array"),
            Some(&DataValue::TextArray(vec![
                "hello world".to_string(),
                String::new(),
                "日本語のテキスト".to_string(),
            ]))
        );
        assert_eq!(
            d.fields.get("w_text_array_empty"),
            Some(&DataValue::TextArray(Vec::new()))
        );
    }

    /// #1175: a length header that overshoots the bytes left must be
    /// rejected by `checked_capacity` up front, before any allocation.
    #[test]
    fn rejects_a_truncated_text_array() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes()); // doc id
        write_varint(&mut buf, 1); // field count
        write_varint(&mut buf, 1); // name length
        buf.extend_from_slice(b"t");
        buf.push(TAG_TEXT_ARRAY);
        write_varint(&mut buf, 64); // declares 64 elements ...
        buf.extend_from_slice(&[1, b'a']); // ... but only one 1-byte element

        let mut cursor = 0;
        let err = decode_document(&buf, &mut cursor).unwrap_err();
        assert!(err.to_string().contains("header declares"), "{err}");
    }

    /// #1175: unlike a bool array, a text element can be structurally
    /// impossible — invalid UTF-8 must be a loud decode error.
    #[test]
    fn rejects_invalid_utf8_in_a_text_array_element() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes()); // doc id
        write_varint(&mut buf, 1); // field count
        write_varint(&mut buf, 1); // name length
        buf.extend_from_slice(b"t");
        buf.push(TAG_TEXT_ARRAY);
        write_varint(&mut buf, 1); // one element ...
        write_varint(&mut buf, 2); // ... of two bytes
        buf.extend_from_slice(&[0xff, 0xfe]); // not valid UTF-8

        let mut cursor = 0;
        let err = decode_document(&buf, &mut cursor).unwrap_err();
        assert!(err.to_string().contains("invalid utf-8"), "{err}");
    }

    /// #1180: tag 16 has no "impossible element" (every byte decodes as a
    /// bool), so the loud-failure pin is a length header that overshoots the
    /// bytes left — `checked_capacity` must reject it up front, before any
    /// allocation, rather than the element loop merely running out of input.
    #[test]
    fn rejects_a_truncated_bool_array() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes()); // doc id
        write_varint(&mut buf, 1); // field count
        write_varint(&mut buf, 1); // name length
        buf.extend_from_slice(b"t");
        buf.push(TAG_BOOL_ARRAY);
        write_varint(&mut buf, 8); // declares 8 elements ...
        buf.extend_from_slice(&[1, 0]); // ... but only 2 bytes follow

        let mut cursor = 0;
        let err = decode_document(&buf, &mut cursor).unwrap_err();
        assert!(err.to_string().contains("header declares"), "{err}");
    }

    /// #1184: tag 15 stores micro-seconds, so sub-microsecond detail is
    /// truncated (the scalar tag 5 keeps nanoseconds via RFC 3339). Pinned so
    /// a change of encoding is a deliberate decision.
    #[test]
    fn datetime_array_truncates_sub_microsecond_precision() {
        use chrono::TimeZone;
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let precise = chrono::Utc
            .timestamp_opt(1_700_000_000, 123_456_789)
            .unwrap();
        let docs = vec![(1u64, doc(&[("t", DataValue::DateTimeArray(vec![precise]))]))];
        let documents = round_trip(&storage, "seg", &docs);
        assert_eq!(
            documents[&1].fields.get("t"),
            Some(&DataValue::DateTimeArray(vec![
                chrono::Utc
                    .timestamp_opt(1_700_000_000, 123_456_000)
                    .unwrap(),
            ]))
        );
    }

    /// #1184: a micro-second value outside chrono's range can only come from
    /// corruption; it must be a loud decode error, never a silent epoch.
    #[test]
    fn rejects_a_corrupted_datetime_array_element() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes()); // doc id
        write_varint(&mut buf, 1); // field count
        write_varint(&mut buf, 1); // name length
        buf.extend_from_slice(b"t");
        buf.push(TAG_DATETIME_ARRAY);
        write_varint(&mut buf, 1); // element count
        buf.extend_from_slice(&i64::MAX.to_le_bytes());

        let mut cursor = 0;
        let err = decode_document(&buf, &mut cursor).unwrap_err();
        assert!(
            err.to_string()
                .contains("DateTimeArray element out of range"),
            "{err}"
        );
    }

    #[test]
    fn splits_into_multiple_chunks_at_the_target_size() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        // MAX_DOCS_PER_CHUNK=128, so 300 tiny docs alone force >= 3 chunks.
        let docs: Vec<(u64, AnalyzedDocument)> = (0..300)
            .map(|i| (i as u64, doc(&[("id", DataValue::Int64(i))])))
            .collect();

        let documents = round_trip(&storage, "seg", &docs);
        assert_eq!(documents.len(), 300);
        for i in 0..300u64 {
            assert_eq!(
                documents[&i].fields.get("id"),
                Some(&DataValue::Int64(i as i64))
            );
        }
        assert!(
            count_chunks(&storage, "seg") >= 3,
            "300 docs at MAX_DOCS_PER_CHUNK=128 must span at least 3 chunks"
        );
    }

    #[test]
    fn a_document_larger_than_the_chunk_target_forms_its_own_chunk() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let docs = vec![
            (
                1u64,
                // A fixed, generous size rather than deriving from
                // `CHUNK_TARGET_BYTES` — keeps this test's own data
                // construction safe (no giant allocation) even under a
                // mutation-check that inflates the constant.
                doc(&[("big", DataValue::Text("x".repeat(100_000)))]),
            ),
            (2u64, doc(&[("small", DataValue::Text("y".to_string()))])),
        ];

        let documents = round_trip(&storage, "seg", &docs);
        assert_eq!(documents.len(), 2);
        assert_eq!(
            count_chunks(&storage, "seg"),
            2,
            "an oversized document must be flushed as its own chunk, separate from the next"
        );
    }

    #[test]
    fn an_empty_document_set_round_trips() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let documents = round_trip(&storage, "seg", &[]);
        assert!(documents.is_empty());
    }

    #[test]
    fn a_document_with_no_stored_fields_round_trips() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let docs = vec![(1u64, AnalyzedDocument::new())];
        let documents = round_trip(&storage, "seg", &docs);
        assert!(documents[&1].fields.is_empty());
    }

    #[test]
    fn rejects_a_foreign_magic() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        write_single_chunk_file(&storage, "seg", 0, 0, CODEC_RAW, 0, &[], 0);
        // Overwrite with a bad magic by writing a fresh file directly.
        let output = storage.create_output("seg.docs").unwrap();
        let mut writer = StructWriter::new(output);
        writer.write_raw(b"XXXX").unwrap();
        writer.write_raw(&[VERSION_MAJOR, VERSION_MINOR]).unwrap();
        writer.write_varint(0).unwrap();
        writer.close().unwrap();

        let err = load_docs_file(&storage, "seg").unwrap_err();
        match err {
            LaurusError::Index(msg) => assert!(msg.contains("SDOC"), "{msg}"),
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_a_future_major_version() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let output = storage.create_output("seg.docs").unwrap();
        let mut writer = StructWriter::new(output);
        writer.write_raw(MAGIC).unwrap();
        writer.write_raw(&[VERSION_MAJOR + 1, 0]).unwrap();
        writer.write_varint(0).unwrap();
        writer.close().unwrap();

        let err = load_docs_file(&storage, "seg").unwrap_err();
        match err {
            LaurusError::Index(msg) => {
                assert!(msg.contains("version"), "{msg}");
                assert!(msg.contains("rebuild"), "{msg}");
            }
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_a_chunk_whose_crc_does_not_match() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut fields = AHashMap::new();
        fields.insert("title".to_string(), DataValue::Text("hello".to_string()));
        let mut payload = Vec::new();
        encode_document(&mut payload, 1, &fields);
        let bad_crc = crc32fast::hash(&payload) ^ 1;

        write_single_chunk_file(
            &storage,
            "seg",
            1,
            1,
            CODEC_RAW,
            payload.len() as u64,
            &payload,
            bad_crc,
        );

        let err = load_docs_file(&storage, "seg").unwrap_err();
        match err {
            LaurusError::Index(msg) => assert!(msg.contains("CRC"), "{msg}"),
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_a_truncated_file() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let output = storage.create_output("seg.docs").unwrap();
        let mut writer = StructWriter::new(output);
        writer.write_raw(MAGIC).unwrap();
        writer.write_raw(&[VERSION_MAJOR, VERSION_MINOR]).unwrap();
        writer.write_varint(1).unwrap(); // total_doc_count
        writer.write_varint(1).unwrap(); // chunk_doc_count
        writer.write_u8(CODEC_RAW).unwrap();
        writer.write_varint(50).unwrap(); // uncompressed_size
        writer.write_varint(50).unwrap(); // payload_size (claims 50 bytes)
        writer.write_u32(0).unwrap();
        writer.write_raw(&[0u8; 10]).unwrap(); // but only 10 bytes follow
        writer.close().unwrap();

        let err = load_docs_file(&storage, "seg").unwrap_err();
        match err {
            LaurusError::Index(msg) => assert!(msg.contains("corrupted"), "{msg}"),
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_a_chunk_payload_size_that_overruns_the_file() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let output = storage.create_output("seg.docs").unwrap();
        let mut writer = StructWriter::new(output);
        writer.write_raw(MAGIC).unwrap();
        writer.write_raw(&[VERSION_MAJOR, VERSION_MINOR]).unwrap();
        writer.write_varint(1).unwrap();
        writer.write_varint(1).unwrap();
        writer.write_u8(CODEC_RAW).unwrap();
        writer.write_varint(u64::MAX / 2).unwrap();
        // A header-declared payload size far beyond anything this tiny file
        // could hold — must be rejected cleanly (no OOM allocation attempt).
        writer.write_varint(u64::MAX / 2).unwrap();
        writer.write_u32(0).unwrap();
        writer.close().unwrap();

        let err = load_docs_file(&storage, "seg").unwrap_err();
        match err {
            LaurusError::Index(msg) => assert!(msg.contains("corrupted"), "{msg}"),
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    #[test]
    fn accepts_a_chunk_whose_uncompressed_size_exceeds_the_bytes_left_in_the_file() {
        // Regression test for a bounds-checking bug caught in design review:
        // bounding `uncompressed_size` against remaining FILE bytes (rather
        // than against `payload_size * max expansion ratio`) would wrongly
        // reject every well-compressed chunk, since a legitimately
        // compressed chunk's declared uncompressed size is, by definition,
        // larger than its on-disk footprint.
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut fields = AHashMap::new();
        fields.insert("body".to_string(), DataValue::Text("a".repeat(10_000)));
        let mut uncompressed = Vec::new();
        encode_document(&mut uncompressed, 1, &fields);

        let max_compressed = lz4_flex::block::get_maximum_output_size(uncompressed.len());
        let mut compressed = vec![0u8; max_compressed];
        let compressed_len = lz4_flex::compress_into(&uncompressed, &mut compressed).unwrap();
        compressed.truncate(compressed_len);
        assert!(
            compressed.len() < uncompressed.len(),
            "test setup: payload must actually compress for this test to be meaningful"
        );

        write_single_chunk_file(
            &storage,
            "seg",
            1,
            1,
            CODEC_LZ4,
            uncompressed.len() as u64,
            &compressed,
            crc32fast::hash(&compressed),
        );

        let documents = load_docs_file(&storage, "seg").unwrap();
        assert_eq!(documents.len(), 1);
        match documents[&1].fields.get("body") {
            Some(DataValue::Text(t)) => assert_eq!(t.len(), 10_000),
            other => panic!("expected Text field, got {other:?}"),
        }
    }

    #[test]
    fn falls_back_to_raw_when_compression_does_not_help() {
        // Tests `flush_chunk`'s raw-fallback decision directly rather than
        // going through `encode_document` + a real document: an earlier
        // version of this test used a random `Bytes` field payload inside a
        // real document and flaked, because the document encoding's framing
        // bytes (notably a small doc_id's mostly-zero `u64` LE bytes) gave
        // LZ4 an easy match even though the field payload itself was random
        // — incompressibility of the *whole chunk* is what matters, and
        // that's simpler to guarantee directly.
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut state = 0x9E3779B97F4A7C15u64;
        let uncompressed: Vec<u8> = (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 0xFF) as u8
            })
            .collect();

        let output = storage.create_output("seg.docs").unwrap();
        let mut writer = StructWriter::new(output);
        writer.write_raw(MAGIC).unwrap();
        writer.write_raw(&[VERSION_MAJOR, VERSION_MINOR]).unwrap();
        writer.write_varint(0).unwrap(); // total_doc_count is irrelevant here
        let mut compress_buf = Vec::new();
        StoredFieldsWriter::flush_chunk(&mut writer, &uncompressed, 0, &mut compress_buf).unwrap();
        writer.close().unwrap();

        assert_eq!(first_chunk_codec(&storage, "seg"), CODEC_RAW);
    }

    #[test]
    fn non_contiguous_doc_ids_round_trip() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let docs = vec![
            (10u64, doc(&[("n", DataValue::Text("ten".to_string()))])),
            (20u64, doc(&[("n", DataValue::Text("twenty".to_string()))])),
            (30u64, doc(&[("n", DataValue::Text("thirty".to_string()))])),
        ];

        let documents = round_trip(&storage, "seg", &docs);
        assert_eq!(documents.len(), 3);
        assert_eq!(
            documents[&10].fields.get("n"),
            Some(&DataValue::Text("ten".to_string()))
        );
        assert_eq!(
            documents[&20].fields.get("n"),
            Some(&DataValue::Text("twenty".to_string()))
        );
        assert_eq!(
            documents[&30].fields.get("n"),
            Some(&DataValue::Text("thirty".to_string()))
        );
    }
}
