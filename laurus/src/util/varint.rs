//! Variable-length integer encoding utilities.
//!
//! This module provides efficient variable-length integer encoding and decoding,
//! similar to what's used in protocol buffers and other binary formats.

use crate::error::{LaurusError, Result};

/// Encode a u64 value using variable-length encoding.
///
/// Each output byte stores 7 data bits in the lower bits and a
/// continuation flag in bit 7. The encoding uses 1 to 10 bytes depending
/// on the magnitude of `value`.
///
/// # Arguments
///
/// * `value` - The unsigned 64-bit integer to encode.
///
/// # Returns
///
/// A `Vec<u8>` containing the variable-length encoded bytes.
pub fn encode_u64(value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut val = value;

    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;

        if val != 0 {
            byte |= 0x80; // Set continuation bit
        }

        bytes.push(byte);

        if val == 0 {
            break;
        }
    }

    bytes
}

/// Decode a u64 value from variable-length encoding.
///
/// Reads bytes from `bytes` until a byte without the continuation bit
/// (0x80) is found, reconstructing the original u64 value.
///
/// # Arguments
///
/// * `bytes` - A byte slice containing the variable-length encoded integer.
///   Only the leading bytes that form the encoded value are consumed;
///   trailing bytes are ignored.
///
/// # Returns
///
/// A tuple `(value, bytes_consumed)` where `value` is the decoded integer
/// and `bytes_consumed` is the number of bytes read from the input slice.
///
/// # Errors
///
/// Returns an error if the encoding is incomplete (all bytes have the
/// continuation bit set) or if the value would overflow a u64.
pub fn decode_u64(bytes: &[u8]) -> Result<(u64, usize)> {
    let mut result = 0u64;
    let mut shift = 0;
    let mut bytes_read = 0;

    for &byte in bytes {
        bytes_read += 1;

        if shift >= 64 {
            return Err(LaurusError::other("VarInt overflow"));
        }

        result |= ((byte & 0x7F) as u64) << shift;

        if (byte & 0x80) == 0 {
            return Ok((result, bytes_read));
        }

        shift += 7;
    }

    Err(LaurusError::other("Incomplete VarInt"))
}

/// Append a variable-length encoding of `value` directly onto `buf`.
///
/// Unlike [`encode_u64`] (which allocates and returns a fresh `Vec<u8>` per
/// call), this writes in place — useful in hot loops that incrementally
/// build up one larger buffer, e.g. per-document/per-field encoding inside
/// a stored-fields chunk (Issue #548) before the whole chunk is compressed.
pub(crate) fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

/// Read a variable-length integer from `bytes` starting at `*cursor`,
/// advancing `*cursor` past the bytes consumed.
///
/// Unlike [`decode_u64`] (which always starts at the beginning of the
/// slice), this reads from an arbitrary offset into a larger buffer —
/// useful when parsing a sequence of varints out of one already-decoded
/// chunk (Issue #548) without slicing/copying between each one.
///
/// # Arguments
///
/// * `container` - Names the enclosing structure in error messages
///   (truncation / overflow), e.g. `"stored-fields chunk"`.
pub(crate) fn read_varint(bytes: &[u8], cursor: &mut usize, container: &str) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = *bytes
            .get(*cursor)
            .ok_or_else(|| LaurusError::storage(format!("{container}: truncated varint")))?;
        *cursor += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err(LaurusError::storage(format!(
                "{container}: varint overflow"
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_u64() {
        let test_values = [0, 1, 127, 128, 255, 256, 16383, 16384, u64::MAX];

        for &value in &test_values {
            let encoded = encode_u64(value);
            let (decoded, bytes_read) = decode_u64(&encoded).unwrap();

            assert_eq!(value, decoded);
            assert_eq!(encoded.len(), bytes_read);
        }
    }

    #[test]
    fn test_encoding_efficiency() {
        // Large values should use more bytes
        assert!(encode_u64(u64::MAX).len() <= 10);
    }

    #[test]
    fn test_incomplete_varint() {
        // Test with incomplete data (continuation bit set but no more bytes)
        let incomplete = vec![0x80]; // Continuation bit set but no more data
        assert!(decode_u64(&incomplete).is_err());
    }

    #[test]
    fn test_overflow() {
        // Test with data that would overflow u64 (more than 10 bytes usually, or just massive bytes)
        let overflow_data = vec![0xFF; 20]; // Too many bytes for u64
        let result = decode_u64(&overflow_data);
        assert!(result.is_err());
    }
}
