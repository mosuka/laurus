//! Front-coding (shared-prefix) encoder and decoder for block term bytes.
//!
//! Each block (≤ 128 sorted terms) is encoded as:
//!
//! - the first term, length-prefixed, in full
//! - each subsequent term as `(shared_prefix_len, suffix_len, suffix_bytes)`
//!   where `shared_prefix_len` is the number of leading bytes shared with
//!   the **previous** term in the block
//!
//! Lengths are stored as unsigned LEB128 (uleb128). At typical scales
//! (term length ≤ 64, shared_prefix_len ≤ 64) every length fits in a
//! single byte, so the per-term overhead beyond the suffix bytes is
//! 2 bytes for non-leading terms and 1 byte for the leading term.
//!
//! Decoding requires walking the block sequentially; random in-block
//! access is `O(N · avg_term_len)` but with `N ≤ 128` the constant is
//! small.
//!
//! This module is private to the dictionary module — its types are not
//! part of the public API.

// Skeleton wiring: these `pub(super)` items are consumed by
// `block_reader` (Phase 5) and `builder` (Phase 6). Until then,
// `cargo check` flags them as dead code. Remove this allow once the
// integration is complete.
#![allow(dead_code)]

use crate::error::{LaurusError, Result};
use crate::util::alloc_bounds::checked_len_u64;

/// Append the unsigned LEB128 (uleb128) encoding of `value` to `buf`.
fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push(((value as u8) & 0x7F) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

/// Read the unsigned LEB128 (uleb128) encoding starting at `*cursor`,
/// advancing `*cursor` past the consumed bytes.
///
/// Shared by every varint in a dictionary block — the front-coded lengths
/// here and the block header in [`super::block_reader`].
///
/// # Errors
///
/// [`LaurusError::Index`] when `bytes` ends mid-varint, when the varint
/// runs past the tenth byte, where the shift would pass 63 bits (Issue
/// #1224), or when the tenth byte sets bits above bit 63, which the shift
/// would drop (Issue #1235).
pub(super) fn read_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let Some(&byte) = bytes.get(*cursor) else {
            return Err(LaurusError::index(
                "term dictionary: truncated varint — segment is corrupted",
            ));
        };
        *cursor += 1;
        // The tenth byte carries bit 63 alone; the writer's minimal
        // encoding never sets more (Issue #1235).
        if shift == 63 && byte & 0x7F > 1 {
            return Err(LaurusError::index(
                "term dictionary: varint overflows u64 — segment is corrupted",
            ));
        }
        result |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 64 {
            return Err(LaurusError::index(
                "term dictionary: varint longer than 10 bytes — segment is corrupted",
            ));
        }
    }
}

/// Compute the number of leading bytes shared by `a` and `b`.
fn shared_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Encode a block of sorted term byte slices using front-coding.
///
/// # Arguments
///
/// * `terms` - sorted term byte slices (must be ≤ 128 entries to match
///   `BLOCK_TERM_COUNT`, though this function does not enforce it).
///   The first element is stored in full; the rest as
///   `(shared_prefix_len, suffix_len, suffix_bytes)` against the
///   previous term.
///
/// # Returns
///
/// A `Vec<u8>` containing the encoded block term bytes. Layout:
///
/// ```text
/// [first_term_len: varint]
/// [first_term_bytes]
/// for i in 1..terms.len():
///   [shared_prefix_len: varint]
///   [suffix_len: varint]
///   [suffix_bytes]
/// ```
///
/// Returns an empty `Vec` for an empty `terms` slice.
pub(super) fn encode_block_terms(terms: &[&[u8]]) -> Vec<u8> {
    if terms.is_empty() {
        return Vec::new();
    }

    // Heuristic capacity: average term length + 2 bytes per term for
    // length headers. Under-estimate is fine — Vec re-allocates on push.
    let cap = terms.iter().map(|t| t.len() + 2).sum();
    let mut out = Vec::with_capacity(cap);

    // First term: full bytes.
    write_varint(&mut out, terms[0].len() as u64);
    out.extend_from_slice(terms[0]);

    // Subsequent terms: (shared_prefix_len, suffix_len, suffix_bytes).
    for window in terms.windows(2) {
        let prev = window[0];
        let curr = window[1];
        let shared = shared_prefix_len(prev, curr);
        let suffix = &curr[shared..];
        write_varint(&mut out, shared as u64);
        write_varint(&mut out, suffix.len() as u64);
        out.extend_from_slice(suffix);
    }
    out
}

/// Streaming decoder over a front-coded block.
///
/// Maintains a reusable buffer of the current term bytes; each call to
/// [`Self::next`] advances to the next term and returns a reference to
/// the buffer. The buffer is invalidated on the next call.
pub(super) struct FrontCodingDecoder<'a> {
    /// Encoded block bytes (output of [`encode_block_terms`]).
    bytes: &'a [u8],
    /// Current read position within `bytes`.
    cursor: usize,
    /// Number of terms remaining to be yielded.
    remaining: u32,
    /// Reusable buffer holding the current term's bytes.
    current: Vec<u8>,
    /// `true` until the first call to [`Self::next`].
    is_first: bool,
}

impl<'a> FrontCodingDecoder<'a> {
    /// Create a new decoder over `bytes` for `term_count` terms.
    ///
    /// `term_count` must match the number of terms originally passed
    /// to [`encode_block_terms`]; otherwise `next` will read past the
    /// encoded data or stop short.
    pub(super) fn new(bytes: &'a [u8], term_count: u32) -> Self {
        FrontCodingDecoder {
            bytes,
            cursor: 0,
            remaining: term_count,
            current: Vec::with_capacity(64),
            is_first: true,
        }
    }

    /// Advance to the next term and return a reference to the current
    /// term's bytes. Returns `Ok(None)` once the block is exhausted.
    ///
    /// The returned slice is valid until the next call to `next`.
    ///
    /// # Errors
    ///
    /// [`LaurusError::Index`] when the encoded bytes are corrupt: a
    /// truncated or overlong varint, a length running past the end of the
    /// block, or a shared prefix longer than the previous term (Issue
    /// #1224).
    #[allow(clippy::should_implement_trait)]
    pub(super) fn next(&mut self) -> Result<Option<&[u8]>> {
        if self.remaining == 0 {
            return Ok(None);
        }

        if self.is_first {
            // First term: full bytes.
            let len = read_varint(self.bytes, &mut self.cursor)?;
            let term = self.take(len, "front-coded first term")?;
            self.current.clear();
            self.current.extend_from_slice(term);
            self.is_first = false;
        } else {
            // Subsequent term: (shared_prefix_len, suffix_len, suffix_bytes).
            let shared = read_varint(self.bytes, &mut self.cursor)?;
            // The writer shares at most the whole previous term. Compared
            // as `u64`, since narrowing first would truncate it on a 32-bit
            // target; a larger value would otherwise make `truncate` a
            // silent no-op and yield a wrong term.
            if shared > self.current.len() as u64 {
                return Err(LaurusError::index(format!(
                    "term dictionary: front-coded term shares {shared} bytes with a previous \
                     term of {} bytes — segment is corrupted",
                    self.current.len()
                )));
            }
            let suffix_len = read_varint(self.bytes, &mut self.cursor)?;
            let suffix = self.take(suffix_len, "front-coded term suffix")?;
            self.current.truncate(shared as usize);
            self.current.extend_from_slice(suffix);
        }

        self.remaining -= 1;
        Ok(Some(&self.current))
    }

    /// Take the next `len` bytes of the block, advancing the cursor.
    ///
    /// `len` comes straight from the block, so it is compared against the
    /// bytes left before it is narrowed or added to the cursor.
    fn take(&mut self, len: u64, what: &str) -> Result<&'a [u8]> {
        let available = self.bytes.len() - self.cursor;
        let len = checked_len_u64(len, available as u64, what)?;
        let bytes = &self.bytes[self.cursor..self.cursor + len];
        self.cursor += len;
        Ok(bytes)
    }

    /// Number of terms still to be yielded by [`Self::next`].
    pub(super) fn remaining(&self) -> u32 {
        self.remaining
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: encode a slice of `&str` and decode it back, returning
    /// the recovered terms as `Vec<Vec<u8>>`.
    fn round_trip(terms: &[&str]) -> Vec<Vec<u8>> {
        let term_bytes: Vec<&[u8]> = terms.iter().map(|t| t.as_bytes()).collect();
        let encoded = encode_block_terms(&term_bytes);
        let mut decoder = FrontCodingDecoder::new(&encoded, terms.len() as u32);

        let mut out = Vec::with_capacity(terms.len());
        while let Some(term) = decoder.next().unwrap() {
            out.push(term.to_vec());
        }
        out
    }

    #[test]
    fn varint_round_trip_small() {
        for v in [0u64, 1, 0x7F, 0x80, 0xFF, 0x3FFF, 0x4000, u64::MAX] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let mut cursor = 0;
            assert_eq!(read_varint(&buf, &mut cursor).unwrap(), v);
            assert_eq!(cursor, buf.len());
        }
    }

    #[test]
    fn shared_prefix_len_basic() {
        assert_eq!(shared_prefix_len(b"", b""), 0);
        assert_eq!(shared_prefix_len(b"abc", b""), 0);
        assert_eq!(shared_prefix_len(b"", b"abc"), 0);
        assert_eq!(shared_prefix_len(b"abc", b"abd"), 2);
        assert_eq!(shared_prefix_len(b"abc", b"abc"), 3);
        assert_eq!(shared_prefix_len(b"abc", b"abcd"), 3);
        assert_eq!(shared_prefix_len(b"abc", b"xyz"), 0);
    }

    #[test]
    fn empty_terms_returns_empty_bytes() {
        let encoded = encode_block_terms(&[]);
        assert!(encoded.is_empty());

        let mut decoder = FrontCodingDecoder::new(&encoded, 0);
        assert_eq!(decoder.remaining(), 0);
        assert!(decoder.next().unwrap().is_none());
    }

    #[test]
    fn single_term_round_trip() {
        let recovered = round_trip(&["hello"]);
        assert_eq!(recovered, vec![b"hello".to_vec()]);
    }

    #[test]
    fn multi_term_no_shared_prefix() {
        let terms = ["aaaa", "bbbb", "cccc"];
        let recovered = round_trip(&terms);
        assert_eq!(
            recovered,
            terms
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn multi_term_full_shared_prefix() {
        let terms = ["foobar1", "foobar2", "foobar3"];
        let recovered = round_trip(&terms);
        assert_eq!(
            recovered,
            terms
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn multi_term_one_char_diff_sequence() {
        let terms = ["a", "ab", "abc", "abcd"];
        let recovered = round_trip(&terms);
        assert_eq!(
            recovered,
            terms
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn multi_term_utf8_multibyte() {
        // "日本" (U+65E5 U+672C) shares the leading byte 0xE6 0x97 with
        // "日本語"; encoding/decoding must preserve the raw byte stream
        // regardless of UTF-8 codepoint boundaries.
        let terms = ["日", "日本", "日本語"];
        let recovered = round_trip(&terms);
        assert_eq!(
            recovered,
            terms
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn long_term_followed_by_shorter_term() {
        // Truncate-then-extend path: previous term is a longer extension
        // of the next term's prefix.
        let terms = ["abcdef", "abcd"];
        let recovered = round_trip(&terms);
        assert_eq!(
            recovered,
            terms
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn round_trip_128_random_sorted_terms() {
        // Deterministic LCG to generate a sorted, unique 128-term corpus.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next_u32 = || -> u32 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 32) as u32
        };

        let mut set = std::collections::BTreeSet::new();
        while set.len() < 128 {
            let len = 5 + (next_u32() % 6) as usize;
            let mut s = String::with_capacity(len);
            for _ in 0..len {
                s.push((b'a' + (next_u32() as u8 % 26)) as char);
            }
            set.insert(s);
        }
        let terms: Vec<String> = set.into_iter().collect();
        let term_strs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();

        let recovered = round_trip(&term_strs);
        let recovered_strings: Vec<String> = recovered
            .into_iter()
            .map(|b| String::from_utf8(b).unwrap())
            .collect();
        assert_eq!(recovered_strings, terms);
    }

    #[test]
    fn next_returns_none_after_exhaustion() {
        let term_bytes: Vec<&[u8]> = vec![b"alpha", b"beta"];
        let encoded = encode_block_terms(&term_bytes);
        let mut decoder = FrontCodingDecoder::new(&encoded, 2);
        assert_eq!(decoder.next().unwrap().unwrap(), b"alpha");
        assert_eq!(decoder.next().unwrap().unwrap(), b"beta");
        assert!(decoder.next().unwrap().is_none());
        // Calling next again must remain None and not panic.
        assert!(decoder.next().unwrap().is_none());
        assert_eq!(decoder.remaining(), 0);
    }

    #[test]
    fn empty_string_term() {
        // Edge case: a block containing an empty term must encode/decode
        // correctly. Sorted dictionaries can hit this if "" is a valid
        // term (rare but legal).
        let term_bytes: Vec<&[u8]> = vec![b"", b"abc"];
        let encoded = encode_block_terms(&term_bytes);
        let mut decoder = FrontCodingDecoder::new(&encoded, 2);
        assert_eq!(decoder.next().unwrap().unwrap(), b"");
        assert_eq!(decoder.next().unwrap().unwrap(), b"abc");
        assert!(decoder.next().unwrap().is_none());
    }

    // Issue #1224: corrupt front-coded bytes are reported, not decoded
    // past the end of the block.

    fn assert_corrupted<T: std::fmt::Debug>(result: Result<T>, label: &str) {
        match result {
            Ok(value) => panic!("expected a corruption error mentioning {label:?}, got {value:?}"),
            Err(LaurusError::Index(msg)) => {
                assert!(msg.contains("corrupted") && msg.contains(label), "{msg}");
            }
            Err(other) => panic!("expected Index error, got {other:?}"),
        }
    }

    /// Decodes the first term of `bytes`, declared as a one-term block.
    fn first_term(bytes: &[u8]) -> Result<Option<Vec<u8>>> {
        let mut decoder = FrontCodingDecoder::new(bytes, 1);
        decoder.next().map(|term| term.map(<[u8]>::to_vec))
    }

    /// A varint whose continuation bit runs off the end of the block is
    /// rejected (Issue #1224).
    #[test]
    fn a_truncated_varint_is_rejected() {
        assert_corrupted(first_term(&[0x80]), "truncated varint");
    }

    /// A varint longer than ten bytes would shift past 63 bits (Issue
    /// #1224).
    #[test]
    fn a_varint_longer_than_ten_bytes_is_rejected() {
        let mut bytes = vec![0x80; 11];
        bytes.push(0x00);
        assert_corrupted(first_term(&bytes), "longer than 10 bytes");
    }

    /// A tenth byte with data bits above bit 63 would have them shifted out
    /// (Issue #1235); `u64::MAX`, whose tenth byte is `0x01`, still decodes.
    #[test]
    fn a_tenth_varint_byte_above_bit_63_is_rejected() {
        let mut bytes = vec![0x80; 9];
        bytes.push(0x02);
        assert_corrupted(first_term(&bytes), "overflows u64");

        let mut bytes = Vec::new();
        write_varint(&mut bytes, u64::MAX);
        assert_eq!(bytes.len(), 10);
        assert_eq!(read_varint(&bytes, &mut 0).unwrap(), u64::MAX);
    }

    /// A first-term length beyond the block is rejected before
    /// `cursor + len` can overflow (Issue #1224).
    #[test]
    fn a_first_term_longer_than_the_block_is_rejected() {
        let mut bytes = Vec::new();
        write_varint(&mut bytes, u64::MAX);
        bytes.extend_from_slice(b"abc");
        assert_corrupted(first_term(&bytes), "front-coded first term");
    }

    /// A suffix length beyond the block is rejected (Issue #1224).
    #[test]
    fn a_suffix_longer_than_the_block_is_rejected() {
        // "ab", then (shared 2, suffix_len 10, "c"): 9 suffix bytes missing.
        let bytes = [2, b'a', b'b', 2, 10, b'c'];
        let mut decoder = FrontCodingDecoder::new(&bytes, 2);
        assert_eq!(decoder.next().unwrap().unwrap(), b"ab");
        assert_corrupted(decoder.next(), "front-coded term suffix");
    }

    /// A shared prefix longer than the previous term is rejected, where
    /// `truncate` would otherwise ignore it and yield a wrong term (Issue
    /// #1224).
    #[test]
    fn a_shared_prefix_longer_than_the_previous_term_is_rejected() {
        // "ab", then (shared 3, suffix_len 1, "c").
        let bytes = [2, b'a', b'b', 3, 1, b'c'];
        let mut decoder = FrontCodingDecoder::new(&bytes, 2);
        assert_eq!(decoder.next().unwrap().unwrap(), b"ab");
        assert_corrupted(decoder.next(), "shares 3 bytes");
    }
}
