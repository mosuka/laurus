//! Structured file I/O for binary data serialization.
//!
//! This module provides efficient binary serialization for search index data structures,
//! similar to Whoosh's structfile.py but optimized for Rust and modern hardware.
//!
//! Two layers of abstraction are provided:
//!
//! - **Structured I/O** ([`StructWriter`] / [`StructReader`]) -- typed field-level
//!   reading and writing of primitives, variable-length integers, strings, byte
//!   arrays, and compound structures, with a CRC-32 footer for integrity
//!   verification.
//! - **Block I/O** ([`BlockWriter`] / [`BlockReader`]) -- higher-level block-based
//!   batching built on top of structured I/O, designed for posting lists and
//!   other data that benefits from fixed-size block buffering.
//!
//! # Footer
//!
//! [`StructWriter::close`] ends every file with an 8-byte footer (Issue
//! #1214):
//!
//! ```text
//! payload | crc32(payload): u32 LE | FOOTER_MAGIC ("LCRC"): u32 LE
//! ```
//!
//! Files written before Issue #1214 end in a 4-byte trailer instead, holding
//! the CRC-32 of the file's *last write* only — it covers nothing else, so a
//! reader accepts it as [`ChecksumStatus::Legacy`] rather than as verified.
//! The magic is what tells the two apart: a reader that parsed the payload
//! knows exactly how many bytes follow it, and a reader that did not (a part
//! read by random access, see [`verify_file_checksum`]) checks for the magic
//! at the end of the file.

use std::collections::HashMap;
use std::io::{Read, SeekFrom};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{LaurusError, Result};
use crate::storage::{StorageInput, StorageOutput};
use crate::util::alloc_bounds::{checked_capacity_u64, checked_len, checked_len_u64};
use crate::util::varint::{decode_u64, encode_u64};

/// Marker that ends a [`StructWriter`] footer: `LCRC` in little-endian byte
/// order (Issue #1214).
///
/// It follows the CRC rather than preceding it, so a legacy file is misread
/// as footed only when its trailer CRC happens to equal the magic (a 1 in
/// 2^32 chance), never because of what its payload contains. Its bytes also
/// cannot start a legacy `deletions.log` record (a varint length followed by
/// `{`), which keeps a log that mixes both forms unambiguous.
pub const FOOTER_MAGIC: u32 = u32::from_le_bytes(*b"LCRC");

/// Length of a [`StructWriter`] footer: the CRC-32, then [`FOOTER_MAGIC`].
pub const FOOTER_LEN: u64 = 8;

/// Length of the trailer files carried before Issue #1214: a CRC-32 of the
/// file's last write.
const LEGACY_TRAILER_LEN: u64 = 4;

/// Chunk size for hashing a file that cannot lend a slice.
const VERIFY_CHUNK: usize = 64 * 1024;

/// What the end of a [`StructWriter`] file says about the bytes before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumStatus {
    /// The footer's CRC-32 matches every byte before it.
    Verified,
    /// A 4-byte trailer from before Issue #1214. It holds the CRC-32 of the
    /// file's last write only, so on its own it verifies nothing; the stored
    /// value is returned for callers that know what that last write was.
    Legacy(u32),
    /// The footer does not match the bytes: the file is corrupted.
    Mismatch,
}

impl ChecksumStatus {
    /// Whether this status vouches for the file, for a caller that knows
    /// the file's last write was `last_write`: a footer must have verified,
    /// and a legacy trailer must be the CRC-32 of `last_write` — which is
    /// all a legacy trailer ever covered.
    ///
    /// # Arguments
    ///
    /// * `last_write` - The bytes of the file's last write.
    pub fn vouches_for(self, last_write: &[u8]) -> bool {
        match self {
            ChecksumStatus::Verified => true,
            ChecksumStatus::Legacy(stored) => stored == crc32fast::hash(last_write),
            ChecksumStatus::Mismatch => false,
        }
    }
}

/// Structured binary writer with typed fields and CRC-32 checksumming.
///
/// `StructWriter` wraps a [`StorageOutput`] and provides typed write methods
/// for primitive values, variable-length integers, strings, byte arrays, and
/// compound structures such as delta-compressed integer lists and string-to-u64
/// maps. A running CRC-32 of every byte written is kept and emitted in the
/// footer when the writer is closed (see the [module docs](self)), enabling
/// integrity verification on read.
///
/// The output is written front to back: there is no general `seek`, so the
/// footer always covers the bytes in the order they sit in the file. A format
/// whose header depends on what follows it reserves the header with
/// [`reserve_header`](Self::reserve_header) and writes it last with
/// [`fill_header`](Self::fill_header).
///
/// All multi-byte numeric values are encoded in **little-endian** byte order.
pub struct StructWriter<W: StorageOutput> {
    /// The underlying storage output handle.
    writer: W,
    /// Running CRC-32 of every byte written.
    hasher: crc32fast::Hasher,
    /// Current byte position in the output stream.
    position: u64,
    /// Length of a header reserved by `reserve_header` and not yet filled.
    reserved_header: Option<u64>,
}

impl<W: StorageOutput> StructWriter<W> {
    /// Create a new structured file writer wrapping the given output.
    ///
    /// # Arguments
    ///
    /// * `writer` - The underlying [`StorageOutput`] to write to.
    ///
    /// # Returns
    ///
    /// A new `StructWriter` positioned at byte 0 with an empty checksum.
    pub fn new(writer: W) -> Self {
        StructWriter {
            writer,
            hasher: crc32fast::Hasher::new(),
            position: 0,
            reserved_header: None,
        }
    }

    /// Write a single `u8` value.
    ///
    /// # Arguments
    ///
    /// * `value` - The byte value to write.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_u8(&mut self, value: u8) -> Result<()> {
        self.writer.write_u8(value)?;
        self.update_checksum(&[value]);
        self.position += 1;
        Ok(())
    }

    /// Write a `u16` value in little-endian byte order.
    ///
    /// # Arguments
    ///
    /// * `value` - The 16-bit unsigned integer to write.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_u16(&mut self, value: u16) -> Result<()> {
        self.writer.write_u16::<LittleEndian>(value)?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 2;
        Ok(())
    }

    /// Write a `u32` value in little-endian byte order.
    ///
    /// # Arguments
    ///
    /// * `value` - The 32-bit unsigned integer to write.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_u32(&mut self, value: u32) -> Result<()> {
        self.writer.write_u32::<LittleEndian>(value)?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 4;
        Ok(())
    }

    /// Write a `u64` value in little-endian byte order.
    ///
    /// # Arguments
    ///
    /// * `value` - The 64-bit unsigned integer to write.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_u64(&mut self, value: u64) -> Result<()> {
        self.writer.write_u64::<LittleEndian>(value)?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 8;
        Ok(())
    }

    /// Write a variable-length encoded unsigned integer.
    ///
    /// Smaller values use fewer bytes, making this efficient for values that
    /// are typically small (e.g. string lengths, deltas).
    ///
    /// # Arguments
    ///
    /// * `value` - The unsigned integer to encode and write.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_varint(&mut self, value: u64) -> Result<()> {
        let encoded = encode_u64(value);
        self.writer.write_all(&encoded)?;
        self.update_checksum(&encoded);
        self.position += encoded.len() as u64;
        Ok(())
    }

    /// Write an `f32` value in little-endian byte order.
    ///
    /// # Arguments
    ///
    /// * `value` - The 32-bit floating-point number to write.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_f32(&mut self, value: f32) -> Result<()> {
        self.writer.write_f32::<LittleEndian>(value)?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 4;
        Ok(())
    }

    /// Write an `f64` value in little-endian byte order.
    ///
    /// # Arguments
    ///
    /// * `value` - The 64-bit floating-point number to write.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_f64(&mut self, value: f64) -> Result<()> {
        self.writer.write_f64::<LittleEndian>(value)?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 8;
        Ok(())
    }

    /// Write a UTF-8 string with a varint length prefix.
    ///
    /// The string is encoded as a varint byte-length followed by the raw UTF-8
    /// bytes, matching the format read by [`StructReader::read_string`].
    ///
    /// # Arguments
    ///
    /// * `value` - The string slice to write.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_string(&mut self, value: &str) -> Result<()> {
        let bytes = value.as_bytes();
        self.write_varint(bytes.len() as u64)?;
        self.writer.write_all(bytes)?;
        self.update_checksum(bytes);
        self.position += bytes.len() as u64;
        Ok(())
    }

    /// Write a byte slice with a varint length prefix.
    ///
    /// # Arguments
    ///
    /// * `value` - The byte slice to write.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_bytes(&mut self, value: &[u8]) -> Result<()> {
        self.write_varint(value.len() as u64)?;
        self.writer.write_all(value)?;
        self.update_checksum(value);
        self.position += value.len() as u64;
        Ok(())
    }

    /// Write raw bytes directly without any length prefix.
    ///
    /// The caller is responsible for knowing the exact byte count on the
    /// reading side.
    ///
    /// # Arguments
    ///
    /// * `value` - The byte slice to write verbatim.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_raw(&mut self, value: &[u8]) -> Result<()> {
        self.writer.write_all(value)?;
        self.update_checksum(value);
        self.position += value.len() as u64;
        Ok(())
    }

    /// Write a `u32` array using delta encoding for compression.
    ///
    /// The values are stored as a varint count followed by varint-encoded
    /// deltas between consecutive elements, which is particularly efficient
    /// for monotonically increasing sequences such as sorted document ID
    /// posting lists.
    ///
    /// # Arguments
    ///
    /// * `values` - The slice of `u32` values to write (should ideally be
    ///   sorted for best compression).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_delta_compressed_u32s(&mut self, values: &[u32]) -> Result<()> {
        if values.is_empty() {
            return self.write_varint(0);
        }

        self.write_varint(values.len() as u64)?;

        let mut previous = 0u32;
        for &value in values {
            let delta = value.wrapping_sub(previous);
            self.write_varint(delta as u64)?;
            previous = value;
        }

        Ok(())
    }

    /// Write a `HashMap<String, u64>` as a varint-counted sequence of
    /// key-value pairs.
    ///
    /// # Arguments
    ///
    /// * `map` - The map to serialize.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn write_string_u64_map(&mut self, map: &HashMap<String, u64>) -> Result<()> {
        self.write_varint(map.len() as u64)?;

        for (key, value) in map {
            self.write_string(key)?;
            self.write_u64(*value)?;
        }

        Ok(())
    }

    /// Get the current byte position in the output stream.
    ///
    /// # Returns
    ///
    /// The number of bytes written so far.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get the CRC-32 of the bytes written so far.
    ///
    /// While a reserved header is unfilled, this covers only the bytes
    /// written after it.
    ///
    /// # Returns
    ///
    /// The running checksum value.
    pub fn checksum(&self) -> u32 {
        self.hasher.clone().finalize()
    }

    /// Fold `data` into the running checksum.
    fn update_checksum(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    /// Reserve the first `len` bytes of the output for a header that is
    /// written last, with [`fill_header`](Self::fill_header).
    ///
    /// For formats whose header records where the rest of the file ended up.
    /// The reserved bytes are written as zeros and left out of the checksum;
    /// `fill_header` hashes the real header and combines it with the rest.
    ///
    /// # Arguments
    ///
    /// * `len` - The exact length the header will have.
    ///
    /// # Errors
    ///
    /// Returns an error if anything has been written yet (the header must be
    /// the start of the file), if a header is already reserved, or if the
    /// write fails.
    pub fn reserve_header(&mut self, len: u64) -> Result<()> {
        if self.position != 0 || self.reserved_header.is_some() {
            return Err(LaurusError::internal(
                "StructWriter: a header can only be reserved at the start of the output",
            ));
        }
        std::io::copy(&mut std::io::repeat(0).take(len), &mut self.writer)?;
        self.position = len;
        self.reserved_header = Some(len);
        Ok(())
    }

    /// Write the header reserved by [`reserve_header`](Self::reserve_header)
    /// through `write`, then return to the end of the output.
    ///
    /// The checksum becomes that of the header followed by everything written
    /// after it, i.e. of the bytes as they sit in the file.
    ///
    /// # Arguments
    ///
    /// * `write` - Writes the header through the writer it is given; it must
    ///   write exactly the reserved length.
    ///
    /// # Errors
    ///
    /// Returns an error if no header is reserved, if `write` fails or writes
    /// a different length than was reserved, or if seeking fails.
    pub fn fill_header<F>(&mut self, write: F) -> Result<()>
    where
        F: FnOnce(&mut Self) -> Result<()>,
    {
        let Some(len) = self.reserved_header.take() else {
            return Err(LaurusError::internal(
                "StructWriter: fill_header called without a reserved header",
            ));
        };
        let end = self.position;
        let body = std::mem::take(&mut self.hasher);

        self.writer.seek(SeekFrom::Start(0))?;
        self.position = 0;
        write(self)?;
        if self.position != len {
            return Err(LaurusError::internal(format!(
                "StructWriter: wrote a {}-byte header into {len} reserved bytes",
                self.position
            )));
        }

        self.writer.seek(SeekFrom::Start(end))?;
        self.position = end;
        self.hasher.combine(&body);
        Ok(())
    }

    /// Write the footer (the CRC-32 of every byte written, then
    /// [`FOOTER_MAGIC`]), flush, and close the writer.
    ///
    /// # Errors
    ///
    /// Returns an error if a reserved header was never filled, or if writing,
    /// flushing, or closing the underlying output fails.
    pub fn close(mut self) -> Result<()> {
        if let Some(len) = self.reserved_header {
            return Err(LaurusError::internal(format!(
                "StructWriter: closed with a {len}-byte header reserved but never filled"
            )));
        }
        let checksum = self.hasher.clone().finalize();
        self.writer.write_u32::<LittleEndian>(checksum)?;
        self.writer.write_u32::<LittleEndian>(FOOTER_MAGIC)?;
        self.writer.flush_and_sync()?;
        self.writer.close()?;
        Ok(())
    }
}

/// Structured binary reader with typed fields and CRC-32 verification.
///
/// `StructReader` is the read counterpart of [`StructWriter`]. It wraps a
/// [`StorageInput`] and provides typed read methods that mirror the writer's
/// format. A reader that parses a file front to back keeps a running CRC-32
/// of what it read, so the footer written by [`StructWriter::close`] can be
/// verified via [`verify_checksum`](Self::verify_checksum) with no second
/// pass. The first [`seek`](Self::seek) ends that: a random-access reader
/// hashes nothing, and verifies a whole file with [`verify_file_checksum`]
/// instead.
///
/// Every read that allocates from a length or count taken from the stream
/// first bounds it by the bytes left in the input (Issue #1218), so a
/// corrupt prefix is reported as corruption instead of driving an
/// allocation large enough to abort the process.
///
/// All multi-byte numeric values are expected in **little-endian** byte order.
pub struct StructReader<R: StorageInput> {
    /// The underlying storage input handle.
    reader: R,
    /// Running CRC-32 of the bytes read so far, while `sequential`.
    hasher: crc32fast::Hasher,
    /// Whether every read so far has been one contiguous run from offset 0.
    /// Cleared by the first seek that moves the cursor; from then on reads
    /// are not hashed and the checksum cannot be verified.
    sequential: bool,
    /// Current byte position in the input stream.
    position: u64,
    /// Total size of the underlying file in bytes.
    file_size: u64,
    /// Length of the file's trailer (a footer or a legacy trailer), learned
    /// on first use by `is_eof`.
    trailer_len: Option<u64>,
}

impl<R: StorageInput> StructReader<R> {
    /// Create a new structured file reader wrapping the given input.
    ///
    /// # Arguments
    ///
    /// * `reader` - The underlying [`StorageInput`] to read from.
    ///
    /// # Returns
    ///
    /// A new `StructReader` positioned at byte 0.
    ///
    /// # Errors
    ///
    /// Returns an error if determining the input size fails.
    pub fn new(reader: R) -> Result<Self> {
        let file_size = reader.size()?;
        Ok(StructReader {
            reader,
            hasher: crc32fast::Hasher::new(),
            sequential: true,
            position: 0,
            file_size,
            trailer_len: None,
        })
    }

    /// Bytes left in the input — the bound for a length or count read from
    /// the stream (Issue #1218). The trailer is not subtracted: not every
    /// input ends in one, and none of the bytes a prefix describes can lie
    /// past the end of the input anyway.
    pub(crate) fn remaining(&self) -> u64 {
        self.file_size.saturating_sub(self.position)
    }

    /// Seek to a position in the input stream.
    ///
    /// A seek that moves the cursor ends checksum tracking: later reads are
    /// not hashed, and [`verify_checksum`](Self::verify_checksum) refuses to
    /// run.
    ///
    /// # Arguments
    ///
    /// * `pos` - The seek target (start, end, or current-relative).
    ///
    /// # Returns
    ///
    /// The new absolute byte position after seeking.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying seek fails.
    pub fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        let new_pos = self.reader.seek(pos)?;
        if new_pos != self.position {
            self.sequential = false;
        }
        self.position = new_pos;
        Ok(new_pos)
    }

    /// Get the current stream position from the underlying reader.
    ///
    /// # Returns
    ///
    /// The absolute byte position reported by the underlying reader.
    ///
    /// # Errors
    ///
    /// Returns an error if querying the position fails.
    pub fn stream_position(&mut self) -> Result<u64> {
        self.reader.stream_position().map_err(LaurusError::from)
    }

    /// Read a single `u8` value.
    ///
    /// # Returns
    ///
    /// The byte value read from the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn read_u8(&mut self) -> Result<u8> {
        let value = self.reader.read_u8()?;
        self.update_checksum(&[value]);
        self.position += 1;
        Ok(value)
    }

    /// Read a `u16` value in little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 16-bit unsigned integer read from the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn read_u16(&mut self) -> Result<u16> {
        let value = self.reader.read_u16::<LittleEndian>()?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 2;
        Ok(value)
    }

    /// Read a `u32` value in little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 32-bit unsigned integer read from the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn read_u32(&mut self) -> Result<u32> {
        let value = self.reader.read_u32::<LittleEndian>()?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 4;
        Ok(value)
    }

    /// Read a `u64` value in little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 64-bit unsigned integer read from the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn read_u64(&mut self) -> Result<u64> {
        let value = self.reader.read_u64::<LittleEndian>()?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 8;
        Ok(value)
    }

    /// Read a variable-length encoded unsigned integer.
    ///
    /// # Returns
    ///
    /// The decoded `u64` value.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation or decoding fails.
    pub fn read_varint(&mut self) -> Result<u64> {
        // A u64 varint is at most 10 bytes (7 data bits per byte, 64 / 7 = 9.14).
        // Using a stack-allocated buffer avoids the per-call heap allocation that
        // `Vec::new()` + `push()` would incur — a hot path uncovered by perf
        // profiling (see #520).
        let mut buf = [0u8; 10];
        let mut len = 0;
        loop {
            let byte = self.reader.read_u8()?;
            buf[len] = byte;
            len += 1;
            if byte & 0x80 == 0 {
                break;
            }
            if len == buf.len() {
                // Malformed varint: continuation bit still set after the 10th byte.
                // `decode_u64` would also catch this via its `shift >= 64` check,
                // but we'd panic on the next `buf[len] = byte` before reaching it.
                return Err(LaurusError::other("VarInt overflow"));
            }
        }

        let (value, _) = decode_u64(&buf[..len])?;
        self.update_checksum(&buf[..len]);
        self.position += len as u64;
        Ok(value)
    }

    /// Read an `f32` value in little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 32-bit floating-point number read from the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn read_f32(&mut self) -> Result<f32> {
        let value = self.reader.read_f32::<LittleEndian>()?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 4;
        Ok(value)
    }

    /// Read an `f64` value in little-endian byte order.
    ///
    /// # Returns
    ///
    /// The 64-bit floating-point number read from the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails.
    pub fn read_f64(&mut self) -> Result<f64> {
        let value = self.reader.read_f64::<LittleEndian>()?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 8;
        Ok(value)
    }

    /// Read a UTF-8 string with a varint length prefix.
    ///
    /// # Returns
    ///
    /// The decoded string.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O fails, the bytes are not valid
    /// UTF-8, or the length prefix exceeds the bytes left in the input (the
    /// file is corrupted).
    pub fn read_string(&mut self) -> Result<String> {
        let length = self.read_varint()?;
        let length = checked_len_u64(length, self.remaining(), "StructReader::read_string length")?;
        let mut bytes = vec![0u8; length];
        self.reader.read_exact(&mut bytes)?;
        self.update_checksum(&bytes);
        self.position += length as u64;

        String::from_utf8(bytes).map_err(|e| LaurusError::storage(format!("Invalid UTF-8: {e}")))
    }

    /// Read a byte array with a varint length prefix.
    ///
    /// # Returns
    ///
    /// The raw bytes read from the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails, or the length
    /// prefix exceeds the bytes left in the input (the file is corrupted).
    pub fn read_bytes(&mut self) -> Result<Vec<u8>> {
        let length = self.read_varint()?;
        let length = checked_len_u64(length, self.remaining(), "StructReader::read_bytes length")?;
        let mut bytes = vec![0u8; length];
        self.reader.read_exact(&mut bytes)?;
        self.update_checksum(&bytes);
        self.position += length as u64;
        Ok(bytes)
    }

    /// Read an exact number of raw bytes without a length prefix.
    ///
    /// # Arguments
    ///
    /// * `length` - The number of bytes to read.
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` containing exactly `length` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails, or `length`
    /// exceeds the bytes left in the input — checked before the buffer is
    /// allocated, since a caller's `length` often comes from the file itself
    /// (Issue #1218).
    pub fn read_raw(&mut self, length: usize) -> Result<Vec<u8>> {
        let length = checked_len(length, self.remaining(), "StructReader::read_raw length")?;
        let mut bytes = vec![0u8; length];
        self.reader.read_exact(&mut bytes)?;
        self.update_checksum(&bytes);
        self.position += length as u64;
        Ok(bytes)
    }

    /// Run `f` against the next `length` raw bytes, taking a zero-copy
    /// slice from the underlying storage when possible (mmap-backed
    /// `FileStorage`, in-memory storage). Falls through to a heap-
    /// allocated buffer + [`Read`] when the input cannot lend a slice
    /// (buffered file I/O).
    ///
    /// This is the Issue #504 hot path for the lexical posting
    /// decoder: every PFOR block is a contiguous run of bytes that
    /// `bitpacking::BitPacker4x::decompress*` consumes via `&[u8]`,
    /// so calling it on a borrowed mmap region skips the per-block
    /// allocation + `copy_from_slice` that `read_raw` would otherwise
    /// perform.
    ///
    /// The running checksum and stream position are updated
    /// identically to [`Self::read_raw`]; callers see the same
    /// state-machine progression regardless of which path was taken.
    ///
    /// # Errors
    ///
    /// * Underlying I/O failure (mmap seek or `read_exact` short
    ///   read).
    /// * `length` larger than the bytes left in the input: the slice
    ///   path is taken only when the input can lend that many bytes, and
    ///   the fallback is bounded by [`Self::read_raw`] (Issue #1218).
    pub fn read_raw_with<F, T>(&mut self, length: usize, f: F) -> Result<T>
    where
        F: FnOnce(&[u8]) -> T,
    {
        // Zero-copy path: only when the underlying storage can lend
        // a slice large enough.
        if let Some(slice) = self.reader.as_slice()
            && slice.len() >= length
        {
            let chunk = &slice[..length];
            // Hash and run the callback while the immutable borrow on
            // `self.reader` is still live — through the field, since a
            // `&mut self` helper would conflict with that borrow.
            if self.sequential {
                self.hasher.update(chunk);
            }
            let result = f(chunk);
            // NLL: `chunk` is no longer used after the callback,
            // so the immutable borrow ends here and we can mutate
            // `self`.
            self.position += length as u64;
            self.reader
                .seek(std::io::SeekFrom::Current(length as i64))?;
            return Ok(result);
        }
        // Fallback: heap-allocated buffer + Read.
        let bytes = self.read_raw(length)?;
        Ok(f(&bytes))
    }

    /// Read a delta-compressed `u32` array.
    ///
    /// This reverses the encoding performed by
    /// [`StructWriter::write_delta_compressed_u32s`].
    ///
    /// # Returns
    ///
    /// A vector of reconstructed `u32` values.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails, or the
    /// element count exceeds what the bytes left in the input can hold (the
    /// file is corrupted).
    pub fn read_delta_compressed_u32s(&mut self) -> Result<Vec<u32>> {
        let length = self.read_varint()?;
        // Each element is a varint of at least one byte.
        let length = checked_capacity_u64(
            length,
            1,
            self.remaining(),
            "StructReader::read_delta_compressed_u32s count",
        )?;
        if length == 0 {
            return Ok(Vec::new());
        }

        let mut values = Vec::with_capacity(length);
        let mut previous = 0u32;

        for _ in 0..length {
            let delta = self.read_varint()? as u32;
            let value = previous.wrapping_add(delta);
            values.push(value);
            previous = value;
        }

        Ok(values)
    }

    /// Read a `HashMap<String, u64>` previously written by
    /// [`StructWriter::write_string_u64_map`].
    ///
    /// # Returns
    ///
    /// The deserialized map.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying I/O operation fails, or the entry
    /// count exceeds what the bytes left in the input can hold (the file is
    /// corrupted).
    pub fn read_string_u64_map(&mut self) -> Result<HashMap<String, u64>> {
        let length = self.read_varint()?;
        // Each entry is at least an empty key's 1-byte length and a u64.
        let length = checked_capacity_u64(
            length,
            1 + 8,
            self.remaining(),
            "StructReader::read_string_u64_map count",
        )?;
        let mut map = HashMap::with_capacity(length);

        for _ in 0..length {
            let key = self.read_string()?;
            let value = self.read_u64()?;
            map.insert(key, value);
        }

        Ok(map)
    }

    /// Get the current byte position in the input stream.
    ///
    /// # Returns
    ///
    /// The number of bytes consumed so far.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get the total file size.
    ///
    /// # Returns
    ///
    /// The size of the underlying file in bytes.
    pub fn size(&self) -> u64 {
        self.file_size
    }

    /// Check whether the reader has reached the end of the payload.
    ///
    /// The file ends in a footer (8 bytes) or a legacy trailer (4 bytes),
    /// so this returns `true` once the position is within that region. The
    /// first call peeks at the file's last 4 bytes to learn which one it is.
    ///
    /// # Returns
    ///
    /// `true` if no more data blocks remain to be read.
    ///
    /// # Errors
    ///
    /// Returns an error if peeking at the end of the file fails.
    pub fn is_eof(&mut self) -> Result<bool> {
        let trailer_len = self.trailer_len()?;
        Ok(self.position >= self.file_size.saturating_sub(trailer_len))
    }

    /// The next 4 bytes as a little-endian `u32`, without consuming them.
    ///
    /// # Returns
    ///
    /// `None` when fewer than 4 bytes remain.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or restoring the cursor fails.
    pub(crate) fn peek_u32(&mut self) -> Result<Option<u32>> {
        if self.remaining() < 4 {
            return Ok(None);
        }
        let value = self.reader.read_u32::<LittleEndian>()?;
        self.reader.seek(SeekFrom::Start(self.position))?;
        Ok(Some(value))
    }

    /// The length of the file's trailer, detected once from its last 4
    /// bytes. The peek restores the cursor directly, so it does not count
    /// as a seek that ends checksum tracking.
    fn trailer_len(&mut self) -> Result<u64> {
        if let Some(len) = self.trailer_len {
            return Ok(len);
        }
        let mut len = LEGACY_TRAILER_LEN;
        if self.file_size >= FOOTER_LEN {
            self.reader
                .seek(SeekFrom::Start(self.file_size - LEGACY_TRAILER_LEN))?;
            let last = self.reader.read_u32::<LittleEndian>()?;
            self.reader.seek(SeekFrom::Start(self.position))?;
            if last == FOOTER_MAGIC {
                len = FOOTER_LEN;
            }
        }
        self.trailer_len = Some(len);
        Ok(len)
    }

    /// Get the CRC-32 of the bytes read so far.
    ///
    /// Meaningful only while the reader has not seeked; see
    /// [`seek`](Self::seek).
    ///
    /// # Returns
    ///
    /// The running checksum value.
    pub fn checksum(&self) -> u32 {
        self.hasher.clone().finalize()
    }

    /// Fold `data` into the running checksum, unless a seek ended tracking.
    fn update_checksum(&mut self, data: &[u8]) {
        if self.sequential {
            self.hasher.update(data);
        }
    }

    /// Read the file's trailer and check it against the bytes parsed so
    /// far — call this once the whole payload has been read.
    ///
    /// Having parsed the payload, the reader knows exactly how many bytes
    /// follow it, which is what tells the two trailer forms apart:
    ///
    /// * 8 bytes: a footer. [`Verified`](ChecksumStatus::Verified) when it
    ///   ends in [`FOOTER_MAGIC`] and its CRC matches, otherwise
    ///   [`Mismatch`](ChecksumStatus::Mismatch).
    /// * 4 bytes: a legacy trailer, returned as
    ///   [`Legacy`](ChecksumStatus::Legacy) — unless it *is* the magic, which
    ///   means a footed file was read 4 bytes too far.
    /// * Anything else: the parse did not end where the payload does, so
    ///   [`Mismatch`](ChecksumStatus::Mismatch).
    ///
    /// # Errors
    ///
    /// Returns an error if the reader has seeked (its checksum no longer
    /// covers the payload — a caller bug), or if reading the trailer fails.
    pub fn verify_checksum(&mut self) -> Result<ChecksumStatus> {
        if !self.sequential {
            return Err(LaurusError::internal(
                "StructReader: a checksum cannot be verified after a seek",
            ));
        }
        let computed = self.hasher.clone().finalize();
        let status = match self.remaining() {
            FOOTER_LEN => {
                let stored = self.reader.read_u32::<LittleEndian>()?;
                let magic = self.reader.read_u32::<LittleEndian>()?;
                self.position += FOOTER_LEN;
                if magic == FOOTER_MAGIC && stored == computed {
                    ChecksumStatus::Verified
                } else {
                    ChecksumStatus::Mismatch
                }
            }
            LEGACY_TRAILER_LEN => {
                let stored = self.reader.read_u32::<LittleEndian>()?;
                self.position += LEGACY_TRAILER_LEN;
                if stored == FOOTER_MAGIC {
                    ChecksumStatus::Mismatch
                } else {
                    ChecksumStatus::Legacy(stored)
                }
            }
            _ => ChecksumStatus::Mismatch,
        };
        Ok(status)
    }

    /// [`verify_checksum`](Self::verify_checksum), with a mismatch reported
    /// as corruption of `what`. A legacy trailer passes: it never covered
    /// the payload, so there is nothing to check it against.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError::Index`] on a mismatch, and whatever
    /// `verify_checksum` returns.
    pub(crate) fn expect_checksum(&mut self, what: &str) -> Result<()> {
        match self.verify_checksum()? {
            ChecksumStatus::Verified | ChecksumStatus::Legacy(_) => Ok(()),
            ChecksumStatus::Mismatch => Err(LaurusError::index(format!(
                "{what}: checksum mismatch — the file is corrupted"
            ))),
        }
    }

    /// Close the reader and release the underlying input handle.
    ///
    /// # Errors
    ///
    /// Returns an error if closing the underlying input fails.
    pub fn close(mut self) -> Result<()> {
        self.reader.close()
    }
}

/// Verify a whole [`StructWriter`] file against its footer in one pass,
/// without parsing it — for parts that are read by random access, whose
/// reads never form the contiguous run [`StructReader::verify_checksum`]
/// needs (Issue #1214).
///
/// Not knowing where the payload ends, this goes by the file's last 4
/// bytes: [`FOOTER_MAGIC`] means a footer, whose CRC is compared against
/// every byte before it; anything else is a legacy trailer, returned as
/// [`ChecksumStatus::Legacy`]. A legacy file whose trailer happens to equal
/// the magic (a 1 in 2^32 chance) therefore reads as a mismatch.
///
/// # Arguments
///
/// * `input` - The file, from any cursor position.
///
/// # Returns
///
/// The file's [`ChecksumStatus`]. A file too short for even a legacy
/// trailer is a [`Mismatch`](ChecksumStatus::Mismatch).
///
/// # Errors
///
/// Returns an error if reading the file fails.
pub fn verify_file_checksum<R: StorageInput>(mut input: R) -> Result<ChecksumStatus> {
    let size = input.size()?;
    if size < LEGACY_TRAILER_LEN {
        return Ok(ChecksumStatus::Mismatch);
    }
    input.seek(SeekFrom::Start(size - LEGACY_TRAILER_LEN))?;
    let last = input.read_u32::<LittleEndian>()?;
    if size < FOOTER_LEN || last != FOOTER_MAGIC {
        return Ok(ChecksumStatus::Legacy(last));
    }

    let payload_len = size - FOOTER_LEN;
    input.seek(SeekFrom::Start(payload_len))?;
    let stored = input.read_u32::<LittleEndian>()?;

    input.seek(SeekFrom::Start(0))?;
    let mut hasher = crc32fast::Hasher::new();
    match input.as_slice() {
        // No longer than the slice, so `payload_len` fits in `usize`.
        Some(slice) if slice.len() as u64 >= payload_len => {
            hasher.update(&slice[..payload_len as usize]);
        }
        _ => {
            let mut buf = vec![0u8; VERIFY_CHUNK];
            let mut left = payload_len;
            while left > 0 {
                let n = left.min(VERIFY_CHUNK as u64) as usize;
                input.read_exact(&mut buf[..n])?;
                hasher.update(&buf[..n]);
                left -= n as u64;
            }
        }
    }

    Ok(if hasher.finalize() == stored {
        ChecksumStatus::Verified
    } else {
        ChecksumStatus::Mismatch
    })
}

/// Whether a file ends in a [`StructWriter`] footer, read from its last 4
/// bytes alone.
///
/// # Errors
///
/// Returns an error if reading the file fails.
pub fn ends_in_footer<R: StorageInput>(mut input: R) -> Result<bool> {
    let size = input.size()?;
    if size < FOOTER_LEN {
        return Ok(false);
    }
    input.seek(SeekFrom::Start(size - LEGACY_TRAILER_LEN))?;
    Ok(input.read_u32::<LittleEndian>()? == FOOTER_MAGIC)
}

/// Recompute the footer of a [`StructWriter`] file's bytes after a test
/// patched them, so the file reads as intact rather than corrupted.
#[cfg(test)]
pub(crate) fn restamp_footer(bytes: &mut [u8]) {
    let payload_len = bytes.len() - FOOTER_LEN as usize;
    assert_eq!(
        bytes[bytes.len() - 4..],
        FOOTER_MAGIC.to_le_bytes(),
        "restamp_footer expects a footed file"
    );
    let checksum = crc32fast::hash(&bytes[..payload_len]);
    bytes[payload_len..payload_len + 4].copy_from_slice(&checksum.to_le_bytes());
}

/// Block-based writer for efficient batched I/O.
///
/// `BlockWriter` buffers data into fixed-size blocks on top of a
/// [`StructWriter`]. When the current block fills up (or is explicitly
/// flushed), it is written to the underlying stream with a header
/// containing the block size and sequence number. This is well-suited
/// for posting lists and other data that benefits from block-level
/// compression or batched disk I/O.
pub struct BlockWriter<W: StorageOutput> {
    /// The underlying structured writer.
    writer: StructWriter<W>,
    /// Maximum size of a single block in bytes.
    block_size: usize,
    /// Buffer accumulating data for the current block.
    current_block: Vec<u8>,
    /// Number of blocks flushed so far.
    blocks_written: u64,
}

impl<W: StorageOutput> BlockWriter<W> {
    /// Create a new block writer with the specified block size.
    ///
    /// # Arguments
    ///
    /// * `writer` - The underlying [`StorageOutput`] to write to.
    /// * `block_size` - The maximum number of bytes per block.
    ///
    /// # Returns
    ///
    /// A new `BlockWriter` with an empty block buffer.
    pub fn new(writer: W, block_size: usize) -> Self {
        BlockWriter {
            writer: StructWriter::new(writer),
            block_size,
            current_block: Vec::with_capacity(block_size),
            blocks_written: 0,
        }
    }

    /// Write data into the current block buffer.
    ///
    /// If appending `data` would exceed the block size, the current block
    /// is flushed first. Data larger than the block size is written directly
    /// to the underlying stream without buffering.
    ///
    /// # Arguments
    ///
    /// * `data` - The byte slice to write.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or writing fails.
    pub fn write_to_block(&mut self, data: &[u8]) -> Result<()> {
        if self.current_block.len() + data.len() > self.block_size {
            self.flush_block()?;
        }

        if data.len() > self.block_size {
            // Data is larger than block size, write directly
            self.writer.write_raw(data)?;
        } else {
            self.current_block.extend_from_slice(data);
        }

        Ok(())
    }

    /// Flush the current block buffer to the underlying storage.
    ///
    /// A block header (size + sequence number) is written before the data.
    /// This is a no-op if the buffer is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if writing the block header or data fails.
    pub fn flush_block(&mut self) -> Result<()> {
        if !self.current_block.is_empty() {
            // Write block header: size + block number
            self.writer.write_u32(self.current_block.len() as u32)?;
            self.writer.write_u64(self.blocks_written)?;

            // Write block data
            self.writer.write_raw(&self.current_block)?;

            self.current_block.clear();
            self.blocks_written += 1;
        }
        Ok(())
    }

    /// Get the number of blocks written so far.
    ///
    /// # Returns
    ///
    /// The count of flushed blocks.
    pub fn blocks_written(&self) -> u64 {
        self.blocks_written
    }

    /// Flush any remaining buffered data and close the writer.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or closing fails.
    pub fn close(mut self) -> Result<()> {
        self.flush_block()?;
        self.writer.close()
    }
}

/// Block-based reader for efficient batched I/O.
///
/// `BlockReader` reads data written by [`BlockWriter`], loading one block
/// at a time into an internal cache. Callers can then read sub-slices from
/// the cached block without additional I/O. Block sequence numbers are
/// verified on read to detect corruption or out-of-order access.
pub struct BlockReader<R: StorageInput> {
    /// The underlying structured reader.
    reader: StructReader<R>,
    /// Cache holding the most recently read block data.
    block_cache: Vec<u8>,
    /// Size (in bytes) of the currently cached block.
    current_block_size: usize,
    /// Current read position within the cached block.
    current_block_pos: usize,
    /// Number of blocks read so far.
    blocks_read: u64,
}

impl<R: StorageInput> BlockReader<R> {
    /// Create a new block reader wrapping the given input.
    ///
    /// # Arguments
    ///
    /// * `reader` - The underlying [`StorageInput`] to read from.
    ///
    /// # Returns
    ///
    /// A new `BlockReader` with an empty block cache.
    ///
    /// # Errors
    ///
    /// Returns an error if initializing the underlying reader fails.
    pub fn new(reader: R) -> Result<Self> {
        Ok(BlockReader {
            reader: StructReader::new(reader)?,
            block_cache: Vec::new(),
            current_block_size: 0,
            current_block_pos: 0,
            blocks_read: 0,
        })
    }

    /// Read the next block from the stream into the internal cache.
    ///
    /// Returns `None` when the end of file is reached.
    ///
    /// # Returns
    ///
    /// `Some(bytes)` containing the block data, or `None` at EOF.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails, the block sequence number
    /// does not match the expected value, or the block size exceeds the
    /// bytes left in the input (see [`StructReader::read_raw`]).
    pub fn read_block(&mut self) -> Result<Option<&[u8]>> {
        if self.reader.is_eof()? {
            return Ok(None);
        }

        // Read block header
        let block_size = self.reader.read_u32()? as usize;
        let block_number = self.reader.read_u64()?;

        // Verify block number
        if block_number != self.blocks_read {
            return Err(LaurusError::storage(format!(
                "Block number mismatch: expected {}, got {}",
                self.blocks_read, block_number
            )));
        }

        // Read block data
        self.block_cache = self.reader.read_raw(block_size)?;
        self.current_block_size = block_size;
        self.current_block_pos = 0;
        self.blocks_read += 1;

        Ok(Some(&self.block_cache))
    }

    /// Read a sub-slice of the given length from the currently cached block.
    ///
    /// Returns `None` if there are not enough bytes remaining in the block.
    ///
    /// # Arguments
    ///
    /// * `length` - The number of bytes to read from the current block.
    ///
    /// # Returns
    ///
    /// `Some(bytes)` on success, or `None` if the block has insufficient
    /// remaining data.
    ///
    /// # Errors
    ///
    /// This method does not perform I/O and currently always returns `Ok`.
    pub fn read_from_block(&mut self, length: usize) -> Result<Option<&[u8]>> {
        if self.current_block_pos + length > self.current_block_size {
            return Ok(None);
        }

        let start = self.current_block_pos;
        let end = start + length;
        self.current_block_pos = end;

        Ok(Some(&self.block_cache[start..end]))
    }

    /// Get the number of blocks read so far.
    ///
    /// # Returns
    ///
    /// The count of blocks loaded into the cache.
    pub fn blocks_read(&self) -> u64 {
        self.blocks_read
    }

    /// Close the reader and release the underlying input handle.
    ///
    /// # Errors
    ///
    /// Returns an error if closing the underlying input fails.
    pub fn close(self) -> Result<()> {
        self.reader.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;

    use crate::storage::memory::MemoryStorage;
    use crate::storage::memory::MemoryStorageConfig;
    use std::sync::Arc;

    #[test]
    fn test_struct_writer_reader() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Write structured data
        {
            let output = storage.create_output("test.struct").unwrap();
            let mut writer = StructWriter::new(output);

            writer.write_u8(42).unwrap();
            writer.write_u16(1234).unwrap();
            writer.write_u32(5678).unwrap();
            writer.write_u64(9876543210).unwrap();
            writer.write_varint(12345).unwrap();
            writer.write_f32(std::f32::consts::PI).unwrap();
            writer.write_f64(std::f64::consts::E).unwrap();
            writer.write_string("Hello, World!").unwrap();
            writer.write_bytes(b"binary data").unwrap();

            let values = vec![1, 5, 10, 15, 25];
            writer.write_delta_compressed_u32s(&values).unwrap();

            writer.close().unwrap();
        }

        // Read structured data
        {
            let input = storage.open_input("test.struct").unwrap();
            let mut reader = StructReader::new(input).unwrap();

            assert_eq!(reader.read_u8().unwrap(), 42);
            assert_eq!(reader.read_u16().unwrap(), 1234);
            assert_eq!(reader.read_u32().unwrap(), 5678);
            assert_eq!(reader.read_u64().unwrap(), 9876543210);
            assert_eq!(reader.read_varint().unwrap(), 12345);
            assert!((reader.read_f32().unwrap() - std::f32::consts::PI).abs() < 0.0001);
            assert!((reader.read_f64().unwrap() - std::f64::consts::E).abs() < 0.000000001);
            assert_eq!(reader.read_string().unwrap(), "Hello, World!");
            assert_eq!(reader.read_bytes().unwrap(), b"binary data");

            let decoded_values = reader.read_delta_compressed_u32s().unwrap();
            assert_eq!(decoded_values, vec![1, 5, 10, 15, 25]);

            // Verify checksum
            assert_eq!(reader.verify_checksum().unwrap(), ChecksumStatus::Verified);
        }
    }

    #[test]
    fn test_block_writer_reader() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Write blocks
        {
            let output = storage.create_output("test.blocks").unwrap();
            let mut writer = BlockWriter::new(output, 1024);

            writer.write_to_block(b"First block data").unwrap();
            writer.write_to_block(b"More data in first block").unwrap();
            writer.flush_block().unwrap();

            writer.write_to_block(b"Second block data").unwrap();
            writer.close().unwrap();
        }

        // Read blocks
        {
            let input = storage.open_input("test.blocks").unwrap();
            let mut reader = BlockReader::new(input).unwrap();

            // Read first block
            let block1 = reader.read_block().unwrap().unwrap();
            assert!(block1.starts_with(b"First block data"));

            // Read second block
            let block2 = reader.read_block().unwrap().unwrap();
            assert!(block2.starts_with(b"Second block data"));

            // No more blocks
            assert!(reader.read_block().unwrap().is_none());

            reader.close().unwrap();
        }
    }

    #[test]
    fn test_string_u64_map() {
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        let mut original_map = HashMap::new();
        original_map.insert("term1".to_string(), 100);
        original_map.insert("term2".to_string(), 200);
        original_map.insert("term3".to_string(), 300);

        // Write map
        {
            let output = storage.create_output("test.map").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_string_u64_map(&original_map).unwrap();
            writer.close().unwrap();
        }

        // Read map
        {
            let input = storage.open_input("test.map").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            let read_map = reader.read_string_u64_map().unwrap();

            assert_eq!(read_map.len(), original_map.len());
            for (key, value) in &original_map {
                assert_eq!(read_map.get(key), Some(value));
            }

            reader.close().unwrap();
        }
    }

    #[test]
    fn test_delta_compression() {
        let values = vec![1000, 1005, 1010, 1020, 1050, 1100];
        let storage = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));

        // Write compressed values
        {
            let output = storage.create_output("test.delta").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_delta_compressed_u32s(&values).unwrap();
            writer.close().unwrap();
        }

        // Read and verify
        {
            let input = storage.open_input("test.delta").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            let decoded = reader.read_delta_compressed_u32s().unwrap();
            assert_eq!(decoded, values);
            reader.close().unwrap();
        }
    }

    /// Writes a lone varint `prefix` — nothing follows it but the trailer —
    /// and opens a reader over it.
    fn reader_over_a_lone_prefix(
        storage: &MemoryStorage,
        name: &str,
        prefix: u64,
    ) -> StructReader<Box<dyn StorageInput>> {
        let output = storage.create_output(name).unwrap();
        let mut writer = StructWriter::new(output);
        writer.write_varint(prefix).unwrap();
        writer.close().unwrap();
        StructReader::new(storage.open_input(name).unwrap()).unwrap()
    }

    fn assert_corrupted<T: std::fmt::Debug>(result: Result<T>) {
        match result.unwrap_err() {
            LaurusError::Index(msg) => assert!(msg.contains("corrupted"), "{msg}"),
            other => panic!("expected Index error, got {other:?}"),
        }
    }

    // A length or count prefix larger than the bytes left in the input is
    // rejected as corruption before anything is allocated for it (Issue
    // #1218). Each of these reads used to size an allocation straight from
    // the prefix; with `u64::MAX`, that panics with a capacity overflow (a
    // smaller impossible value would abort the process instead).

    #[test]
    fn read_string_rejects_a_length_the_input_cannot_back() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        assert_corrupted(reader_over_a_lone_prefix(&storage, "s", u64::MAX).read_string());
    }

    #[test]
    fn read_bytes_rejects_a_length_the_input_cannot_back() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        assert_corrupted(reader_over_a_lone_prefix(&storage, "b", u64::MAX).read_bytes());
    }

    #[test]
    fn read_raw_rejects_a_length_the_input_cannot_back() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        assert_corrupted(reader_over_a_lone_prefix(&storage, "r", 0).read_raw(usize::MAX));
    }

    #[test]
    fn read_delta_compressed_u32s_rejects_a_count_the_input_cannot_back() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        assert_corrupted(
            reader_over_a_lone_prefix(&storage, "d", u64::MAX).read_delta_compressed_u32s(),
        );
    }

    #[test]
    fn read_string_u64_map_rejects_a_count_the_input_cannot_back() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        assert_corrupted(reader_over_a_lone_prefix(&storage, "m", u64::MAX).read_string_u64_map());
    }

    /// The bound is the bytes left in the input, trailer or not: a length
    /// that ends exactly at the end of a trailer-less input is read.
    #[test]
    fn a_length_prefix_that_ends_exactly_at_the_end_of_the_input_is_read() {
        use std::io::Write;

        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut output = storage.create_output("exact").unwrap();
        output.write_all(&[3, b'a', b'b', b'c']).unwrap();
        output.close().unwrap();

        let mut reader = StructReader::new(storage.open_input("exact").unwrap()).unwrap();
        assert_eq!(reader.read_string().unwrap(), "abc");
    }

    // The footer (Issue #1214).

    /// An input that cannot lend a slice, forcing the buffered fallbacks.
    #[derive(Debug)]
    struct NoSlice(Box<dyn StorageInput>);

    impl Read for NoSlice {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.0.read(buf)
        }
    }

    impl std::io::Seek for NoSlice {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.0.seek(pos)
        }
    }

    impl StorageInput for NoSlice {
        fn size(&self) -> Result<u64> {
            self.0.size()
        }
        fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
            self.0.clone_input()
        }
        fn close(&mut self) -> Result<()> {
            self.0.close()
        }
    }

    fn put(storage: &MemoryStorage, name: &str, bytes: &[u8]) {
        use std::io::Write;

        let mut output = storage.create_output(name).unwrap();
        output.write_all(bytes).unwrap();
        output.close().unwrap();
    }

    fn get(storage: &MemoryStorage, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        storage
            .open_input(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    /// The last write of [`write_fixed`], which a legacy trailer covered.
    const LAST_WRITE: &[u8] = b"tail!";

    /// Writes fixed-width fields only, so any flipped byte still parses and
    /// has to be caught by the checksum.
    fn write_fixed(storage: &MemoryStorage, name: &str) -> Vec<u8> {
        let mut writer = StructWriter::new(storage.create_output(name).unwrap());
        writer.write_u8(7).unwrap();
        writer.write_u16(0xBEEF).unwrap();
        writer.write_u32(123_456).unwrap();
        writer.write_u64(u64::MAX - 1).unwrap();
        writer.write_f64(std::f64::consts::PI).unwrap();
        writer.write_raw(LAST_WRITE).unwrap();
        writer.close().unwrap();
        get(storage, name)
    }

    fn read_fixed<R: StorageInput>(reader: &mut StructReader<R>) -> Result<ChecksumStatus> {
        reader.read_u8()?;
        reader.read_u16()?;
        reader.read_u32()?;
        reader.read_u64()?;
        reader.read_f64()?;
        reader.read_raw(LAST_WRITE.len())?;
        reader.verify_checksum()
    }

    fn status_of(storage: &MemoryStorage, name: &str, bytes: &[u8]) -> ChecksumStatus {
        put(storage, name, bytes);
        read_fixed(&mut StructReader::new(storage.open_input(name).unwrap()).unwrap()).unwrap()
    }

    /// The pre-#1214 form of a file: its payload, then the CRC of its last
    /// write in place of the footer.
    fn to_legacy(bytes: &[u8], last_write: &[u8]) -> Vec<u8> {
        let mut legacy = bytes[..bytes.len() - FOOTER_LEN as usize].to_vec();
        legacy.extend_from_slice(&crc32fast::hash(last_write).to_le_bytes());
        legacy
    }

    #[test]
    fn the_footer_is_the_crc_of_every_byte_then_the_magic() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let bytes = write_fixed(&storage, "f");
        let payload_len = bytes.len() - FOOTER_LEN as usize;

        assert_eq!(
            bytes[payload_len..payload_len + 4],
            crc32fast::hash(&bytes[..payload_len]).to_le_bytes()
        );
        assert_eq!(bytes[payload_len + 4..], FOOTER_MAGIC.to_le_bytes());
        assert_eq!(status_of(&storage, "g", &bytes), ChecksumStatus::Verified);
    }

    /// Before Issue #1214 the trailer was the CRC of the last write, so a
    /// flipped byte anywhere before it went unnoticed.
    #[test]
    fn a_flipped_byte_anywhere_is_a_mismatch() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let bytes = write_fixed(&storage, "f");

        for at in 0..bytes.len() {
            let mut flipped = bytes.clone();
            flipped[at] ^= 0x01;
            assert_eq!(
                status_of(&storage, "g", &flipped),
                ChecksumStatus::Mismatch,
                "byte {at} of {}",
                bytes.len()
            );
        }
    }

    #[test]
    fn a_legacy_trailer_reads_as_legacy_and_vouches_only_for_the_last_write() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let legacy = to_legacy(&write_fixed(&storage, "f"), LAST_WRITE);

        let status = status_of(&storage, "g", &legacy);
        assert_eq!(status, ChecksumStatus::Legacy(crc32fast::hash(LAST_WRITE)));
        assert!(status.vouches_for(LAST_WRITE));
        assert!(!status.vouches_for(b"other"));
        assert!(ChecksumStatus::Verified.vouches_for(b"anything"));
        assert!(!ChecksumStatus::Mismatch.vouches_for(LAST_WRITE));
    }

    /// Four bytes left that spell the magic are a footed file read four
    /// bytes too far, not a legacy trailer.
    #[test]
    fn a_legacy_trailer_equal_to_the_magic_is_a_mismatch() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let bytes = write_fixed(&storage, "f");
        let mut truncated = bytes[..bytes.len() - FOOTER_LEN as usize].to_vec();
        truncated.extend_from_slice(&FOOTER_MAGIC.to_le_bytes());

        assert_eq!(
            status_of(&storage, "g", &truncated),
            ChecksumStatus::Mismatch
        );
    }

    #[test]
    fn a_parse_that_stops_short_of_the_footer_is_a_mismatch() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        write_fixed(&storage, "f");

        let mut reader = StructReader::new(storage.open_input("f").unwrap()).unwrap();
        reader.read_u8().unwrap();
        assert_eq!(reader.verify_checksum().unwrap(), ChecksumStatus::Mismatch);
    }

    #[test]
    fn a_checksum_cannot_be_verified_after_a_seek() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        write_fixed(&storage, "f");

        let mut reader = StructReader::new(storage.open_input("f").unwrap()).unwrap();
        reader.read_u8().unwrap();
        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.read_u8().unwrap();
        assert!(read_fixed_rest(&mut reader).is_err());

        // A seek that does not move the cursor keeps tracking intact.
        let mut reader = StructReader::new(storage.open_input("f").unwrap()).unwrap();
        reader.read_u8().unwrap();
        reader.seek(SeekFrom::Current(0)).unwrap();
        assert_eq!(
            read_fixed_rest(&mut reader).unwrap(),
            ChecksumStatus::Verified
        );
    }

    /// [`read_fixed`] after its first field.
    fn read_fixed_rest<R: StorageInput>(reader: &mut StructReader<R>) -> Result<ChecksumStatus> {
        reader.read_u16()?;
        reader.read_u32()?;
        reader.read_u64()?;
        reader.read_f64()?;
        reader.read_raw(LAST_WRITE.len())?;
        reader.verify_checksum()
    }

    /// The zero-copy path and the buffered one hash the same bytes.
    #[test]
    fn read_raw_with_hashes_the_same_on_both_paths() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let data: Vec<u8> = (0..=255u8).collect();
        let mut writer = StructWriter::new(storage.create_output("r").unwrap());
        writer.write_raw(&data).unwrap();
        writer.close().unwrap();

        let mut sliced = StructReader::new(storage.open_input("r").unwrap()).unwrap();
        let mut buffered = StructReader::new(NoSlice(storage.open_input("r").unwrap())).unwrap();
        for read in [
            sliced.read_raw_with(data.len(), |b| b.to_vec()).unwrap(),
            buffered.read_raw_with(data.len(), |b| b.to_vec()).unwrap(),
        ] {
            assert_eq!(read, data);
        }
        assert_eq!(sliced.checksum(), crc32fast::hash(&data));
        assert_eq!(buffered.checksum(), crc32fast::hash(&data));
        assert_eq!(sliced.verify_checksum().unwrap(), ChecksumStatus::Verified);
        assert_eq!(
            buffered.verify_checksum().unwrap(),
            ChecksumStatus::Verified
        );
    }

    #[test]
    fn is_eof_stops_before_either_trailer_form() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut writer = StructWriter::new(storage.create_output("f").unwrap());
        for v in 0..3u32 {
            writer.write_u32(v).unwrap();
        }
        writer.close().unwrap();
        let footed = get(&storage, "f");
        put(&storage, "l", &to_legacy(&footed, &2u32.to_le_bytes()));

        for name in ["f", "l"] {
            let mut reader = StructReader::new(storage.open_input(name).unwrap()).unwrap();
            let mut values = Vec::new();
            while !reader.is_eof().unwrap() {
                values.push(reader.read_u32().unwrap());
            }
            assert_eq!(values, [0, 1, 2], "{name}");
            // Peeking at the trailer does not end checksum tracking.
            assert_ne!(
                reader.verify_checksum().unwrap(),
                ChecksumStatus::Mismatch,
                "{name}"
            );
        }
    }

    /// A header reserved up front and filled last is hashed where it sits:
    /// at the start of the file, ahead of the body written before it.
    #[test]
    fn a_filled_header_is_covered_in_file_order() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut writer = StructWriter::new(storage.create_output("h").unwrap());
        writer.reserve_header(12).unwrap();
        writer.write_raw(b"body bytes").unwrap();
        let body_end = writer.position();
        writer
            .fill_header(|w| {
                w.write_u32(0xABCD)?;
                w.write_u64(body_end)
            })
            .unwrap();
        assert_eq!(writer.position(), body_end, "back at the end of the body");
        writer.write_u8(b'!').unwrap();
        writer.close().unwrap();

        let bytes = get(&storage, "h");
        let payload_len = bytes.len() - FOOTER_LEN as usize;
        assert_eq!(bytes[..4], 0xABCDu32.to_le_bytes());
        assert_eq!(bytes[4..12], body_end.to_le_bytes());
        assert_eq!(&bytes[12..payload_len], b"body bytes!");
        assert_eq!(
            bytes[payload_len..payload_len + 4],
            crc32fast::hash(&bytes[..payload_len]).to_le_bytes()
        );
        assert_eq!(
            verify_file_checksum(storage.open_input("h").unwrap()).unwrap(),
            ChecksumStatus::Verified
        );
    }

    #[test]
    fn a_header_is_reserved_once_at_the_start_and_filled_to_its_length() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());

        let mut writer = StructWriter::new(storage.create_output("a").unwrap());
        writer.write_u8(1).unwrap();
        assert!(writer.reserve_header(4).is_err(), "not at the start");

        let mut writer = StructWriter::new(storage.create_output("b").unwrap());
        assert!(writer.fill_header(|_| Ok(())).is_err(), "nothing reserved");

        let mut writer = StructWriter::new(storage.create_output("c").unwrap());
        writer.reserve_header(4).unwrap();
        assert!(writer.reserve_header(4).is_err(), "already reserved");
        assert!(
            writer.fill_header(|w| w.write_u16(1)).is_err(),
            "shorter than reserved"
        );

        let mut writer = StructWriter::new(storage.create_output("d").unwrap());
        writer.reserve_header(4).unwrap();
        assert!(writer.close().is_err(), "closed unfilled");
    }

    #[test]
    fn verify_file_checksum_tells_footed_legacy_and_corrupt_files_apart() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let bytes = write_fixed(&storage, "f");
        let verify = |name: &str| verify_file_checksum(storage.open_input(name).unwrap()).unwrap();

        assert_eq!(verify("f"), ChecksumStatus::Verified);
        let mut flipped = bytes.clone();
        flipped[3] ^= 0x80;
        put(&storage, "x", &flipped);
        assert_eq!(verify("x"), ChecksumStatus::Mismatch);
        put(&storage, "l", &to_legacy(&bytes, LAST_WRITE));
        assert_eq!(
            verify("l"),
            ChecksumStatus::Legacy(crc32fast::hash(LAST_WRITE))
        );
        put(&storage, "e", &[1, 2]);
        assert_eq!(verify("e"), ChecksumStatus::Mismatch);
    }

    /// A file larger than one hashing chunk, read without a slice.
    #[test]
    fn verify_file_checksum_streams_an_input_without_a_slice() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let data: Vec<u8> = (0..VERIFY_CHUNK * 2 + 17).map(|i| i as u8).collect();
        let mut writer = StructWriter::new(storage.create_output("big").unwrap());
        writer.write_raw(&data).unwrap();
        writer.close().unwrap();

        let verify = || verify_file_checksum(NoSlice(storage.open_input("big").unwrap())).unwrap();
        assert_eq!(verify(), ChecksumStatus::Verified);

        let mut flipped = get(&storage, "big");
        flipped[VERIFY_CHUNK + 5] ^= 0x01;
        put(&storage, "big", &flipped);
        assert_eq!(verify(), ChecksumStatus::Mismatch);
    }

    #[test]
    fn ends_in_footer_reads_the_last_four_bytes() {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let bytes = write_fixed(&storage, "f");
        put(&storage, "l", &to_legacy(&bytes, LAST_WRITE));

        assert!(ends_in_footer(storage.open_input("f").unwrap()).unwrap());
        assert!(!ends_in_footer(storage.open_input("l").unwrap()).unwrap());
    }
}
