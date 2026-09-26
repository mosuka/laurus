//! Structured file I/O for binary data serialization.
//!
//! This module provides efficient binary serialization for search index data structures,
//! similar to Whoosh's structfile.py but optimized for Rust and modern hardware.
//!
//! Two layers of abstraction are provided:
//!
//! - **Structured I/O** ([`StructWriter`] / [`StructReader`]) -- typed field-level
//!   reading and writing of primitives, variable-length integers, strings, byte
//!   arrays, and compound structures, with a CRC-32 checksum trailer for
//!   integrity verification.
//! - **Block I/O** ([`BlockWriter`] / [`BlockReader`]) -- higher-level block-based
//!   batching built on top of structured I/O, designed for posting lists and
//!   other data that benefits from fixed-size block buffering.

use std::collections::HashMap;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{LaurusError, Result};
use crate::storage::{StorageInput, StorageOutput};
use crate::util::alloc_bounds::{checked_capacity_u64, checked_len, checked_len_u64};
use crate::util::varint::{decode_u64, encode_u64};

/// Structured binary writer with typed fields and CRC-32 checksumming.
///
/// `StructWriter` wraps a [`StorageOutput`] and provides typed write methods
/// for primitive values, variable-length integers, strings, byte arrays, and
/// compound structures such as delta-compressed integer lists and string-to-u64
/// maps. A running CRC-32 checksum is maintained and written as a trailer when
/// the writer is closed, enabling integrity verification on read.
///
/// All multi-byte numeric values are encoded in **little-endian** byte order.
pub struct StructWriter<W: StorageOutput> {
    /// The underlying storage output handle.
    writer: W,
    /// Running CRC-32 checksum of written data.
    checksum: u32,
    /// Current byte position in the output stream.
    position: u64,
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
    /// A new `StructWriter` positioned at byte 0 with a zeroed checksum.
    pub fn new(writer: W) -> Self {
        StructWriter {
            writer,
            checksum: 0,
            position: 0,
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

    /// Get the current CRC-32 checksum of written data.
    ///
    /// # Returns
    ///
    /// The running checksum value.
    pub fn checksum(&self) -> u32 {
        self.checksum
    }

    /// Replace the CRC-32 checksum with the hash of the given data.
    ///
    /// Note: this does **not** accumulate a running checksum. Each call
    /// overwrites the previous value with `crc32fast::hash(data)`, so the
    /// stored checksum only reflects the last chunk passed to this method.
    fn update_checksum(&mut self, data: &[u8]) {
        self.checksum = crc32fast::hash(data);
    }

    /// Write the trailing CRC-32 checksum, flush, and close the writer.
    ///
    /// The checksum written is the value computed by the most recent
    /// `update_checksum` call, which only covers the
    /// last chunk of data passed to that method (not all data written).
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or closing the underlying output fails.
    pub fn close(mut self) -> Result<()> {
        // Write final checksum
        self.writer.write_u32::<LittleEndian>(self.checksum)?;
        self.writer.flush_and_sync()?;
        self.writer.close()?;
        Ok(())
    }

    /// Seek to a position in the output stream.
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
    pub fn seek(&mut self, pos: std::io::SeekFrom) -> Result<u64> {
        let new_pos = self.writer.seek(pos)?;
        self.position = new_pos;
        Ok(new_pos)
    }

    /// Get the current stream position from the underlying writer.
    ///
    /// This is useful when mixing raw writes with structured writes to
    /// ensure the tracked position stays in sync.
    ///
    /// # Returns
    ///
    /// The absolute byte position reported by the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an error if querying the position fails.
    pub fn stream_position(&mut self) -> Result<u64> {
        self.writer.stream_position().map_err(LaurusError::from)
    }
}

/// Structured binary reader with typed fields and CRC-32 verification.
///
/// `StructReader` is the read counterpart of [`StructWriter`]. It wraps a
/// [`StorageInput`] and provides typed read methods that mirror the writer's
/// format. A running CRC-32 checksum is maintained so that the trailing
/// checksum written by [`StructWriter::close`] can be verified via
/// [`verify_checksum`](Self::verify_checksum).
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
    /// Running CRC-32 checksum of data read so far.
    checksum: u32,
    /// Current byte position in the input stream.
    position: u64,
    /// Total size of the underlying file in bytes.
    file_size: u64,
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
            checksum: 0,
            position: 0,
            file_size,
        })
    }

    /// Bytes left in the input — the bound for a length or count read from
    /// the stream (Issue #1218). The trailer is not subtracted: not every
    /// input ends in one, and none of the bytes a prefix describes can lie
    /// past the end of the input anyway.
    fn remaining(&self) -> u64 {
        self.file_size.saturating_sub(self.position)
    }

    /// Seek to a position in the input stream.
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
    pub fn seek(&mut self, pos: std::io::SeekFrom) -> Result<u64> {
        let new_pos = self.reader.seek(pos)?;
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
    /// The CRC checksum and stream position are updated identically
    /// to [`Self::read_raw`]; callers see the same state-machine
    /// progression regardless of which path was taken.
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
            // Compute the CRC and run the callback while the
            // immutable borrow on `self.reader` is still live.
            let new_checksum = crc32fast::hash(chunk);
            let result = f(chunk);
            // NLL: `chunk` is no longer used after the callback,
            // so the immutable borrow ends here and we can mutate
            // `self`.
            self.checksum = new_checksum;
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

    /// Check whether the reader has reached the end of file.
    ///
    /// The last 4 bytes of the file are reserved for the CRC-32 checksum
    /// trailer, so this returns `true` once the position is within that
    /// trailing region.
    ///
    /// # Returns
    ///
    /// `true` if no more data blocks remain to be read.
    pub fn is_eof(&self) -> bool {
        self.position >= self.file_size.saturating_sub(4) // Account for checksum
    }

    /// Get the current CRC-32 checksum of data read so far.
    ///
    /// # Returns
    ///
    /// The running checksum value.
    pub fn checksum(&self) -> u32 {
        self.checksum
    }

    /// Replace the CRC-32 checksum with the hash of the given data.
    ///
    /// Note: this does **not** accumulate a running checksum. Each call
    /// overwrites the previous value with `crc32fast::hash(data)`, so the
    /// stored checksum only reflects the last chunk passed to this method.
    fn update_checksum(&mut self, data: &[u8]) {
        self.checksum = crc32fast::hash(data);
    }

    /// Verify file integrity by comparing the running checksum against the
    /// CRC-32 trailer stored at the end of the file.
    ///
    /// # Returns
    ///
    /// `true` if the stored checksum matches the computed one.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is too short to contain a checksum or
    /// if reading the trailer fails.
    pub fn verify_checksum(&mut self) -> Result<bool> {
        if self.position + 4 > self.file_size {
            return Err(LaurusError::storage("File too short for checksum"));
        }

        // Read the stored checksum from the end of file
        let stored_checksum = self.reader.read_u32::<LittleEndian>()?;
        Ok(stored_checksum == self.checksum)
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
        if self.reader.is_eof() {
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
            assert!(reader.verify_checksum().unwrap());
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
}
