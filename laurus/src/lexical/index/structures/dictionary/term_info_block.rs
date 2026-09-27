//! Bit-packed fixed-size [`crate::lexical::index::structures::dictionary::TermInfo`] block.
//!
//! Each block holds up to [`BLOCK_TERM_COUNT`] [`FixedTermInfo`] entries
//! (the fixed-size portion of `TermInfo`, excluding the variable-length
//! `block_max` Vec). Encoding strategy:
//!
//! - For each of the four `u64` fields (`posting_offset`,
//!   `posting_length`, `doc_frequency`, `total_frequency`), find the
//!   block-wide minimum and store it as a header value. Each entry's
//!   field is then encoded as the **unsigned delta** from the block
//!   minimum, using the smallest bit width that fits the largest
//!   delta in the block.
//! - `max_score_factor` is non-monotonic, so it is **not** delta
//!   encoded. Either all entries share the same value (then we store
//!   it once as `ref_max_score_factor`) or each entry gets a full
//!   `f32` slot.
//!
//! This trades a small amount of CPU at decode time (≤ 4 bit-pack
//! reads per lookup) for a significant on-disk size reduction at
//! production scale (10M-100M+ terms / segment).
//!
//! Variable-length `block_max` data is stored separately in
//! [`super::block_max_data::BlockMaxData`].

// Skeleton wiring: these `pub(super)` items are consumed by
// `block_reader` (Phase 5) and `builder` (Phase 6). Until then,
// `cargo check` flags them as dead code. Remove this allow once the
// integration is complete.
#![allow(dead_code)]

use crate::error::{LaurusError, Result};

/// Maximum number of terms in a single dictionary block. Matches the
/// posting list block size (#403 PR-C) so that block boundaries align
/// across the dictionary and posting layers.
pub(super) const BLOCK_TERM_COUNT: usize = 128;

/// Fixed-size portion of `TermInfo` (excluding the variable-length
/// `block_max` Vec). Stored bit-packed in [`FixedTermInfoBlock`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FixedTermInfo {
    /// Offset to the posting list in the posting file.
    pub(super) posting_offset: u64,
    /// Length of the posting list in bytes.
    pub(super) posting_length: u64,
    /// Document frequency.
    pub(super) doc_frequency: u64,
    /// Total frequency across all documents.
    pub(super) total_frequency: u64,
    /// Term-level tightened BM25 TF-component upper bound.
    /// `0.0` means unset (#403 PR-B2).
    pub(super) max_score_factor: f32,
}

/// A bit-packed block of up to [`BLOCK_TERM_COUNT`] [`FixedTermInfo`]
/// entries.
///
/// On disk layout (within the BlockSection of `.dict`):
///
/// ```text
/// [block_min_posting_offset:    u64]
/// [block_min_posting_length:    u64]
/// [block_min_doc_frequency:     u64]
/// [block_min_total_frequency:   u64]
/// [ref_max_score_factor:        f32]   (used when max_score_factor_size == 0)
/// [posting_offset_nbits:        u8]
/// [posting_length_nbits:        u8]
/// [doc_frequency_nbits:         u8]
/// [total_frequency_nbits:       u8]
/// [max_score_factor_size:       u8]    (0 or 32)
/// [payload_len:                 u32]
/// [payload:                     u8 × payload_len]
/// ```
///
/// `payload` holds two consecutive sections:
///
/// 1. Bit-packed deltas for the 4 `u64` fields. Each entry occupies
///    `posting_offset_nbits + posting_length_nbits + doc_frequency_nbits
///     + total_frequency_nbits` bits, in that field order, LSB-first
///    within each byte. Total bytes:
///    `ceil(bits_per_entry × term_count / 8)`.
/// 2. Optional `f32` array of `max_score_factor` values, only if
///    `max_score_factor_size == 32`. 4 bytes per entry, little-endian.
pub(super) struct FixedTermInfoBlock {
    /// Number of entries in this block (`1..=BLOCK_TERM_COUNT`).
    pub(super) term_count: u16,
    pub(super) block_min_posting_offset: u64,
    pub(super) block_min_posting_length: u64,
    pub(super) block_min_doc_frequency: u64,
    pub(super) block_min_total_frequency: u64,
    /// Reference `max_score_factor`. Used as the value for every entry
    /// when `max_score_factor_size == 0`.
    pub(super) ref_max_score_factor: f32,
    pub(super) posting_offset_nbits: u8,
    pub(super) posting_length_nbits: u8,
    pub(super) doc_frequency_nbits: u8,
    pub(super) total_frequency_nbits: u8,
    /// `0` = all entries' `max_score_factor` equal `ref_max_score_factor`.
    /// `32` = each entry has its own `f32` in `payload` after the
    /// bit-packed section.
    pub(super) max_score_factor_size: u8,
    pub(super) payload: Vec<u8>,
}

impl FixedTermInfoBlock {
    /// Encode `entries` (must be `1..=BLOCK_TERM_COUNT` long) into a
    /// bit-packed block.
    ///
    /// Computes the per-field block-wide minimum, then bit-packs each
    /// entry's `value - block_min` delta with the smallest width that
    /// fits the largest delta. `max_score_factor` is encoded as either
    /// "all equal" (zero extra bytes) or a flat `f32` array.
    ///
    /// # Panics
    ///
    /// Panics if `entries.is_empty()` or `entries.len() > BLOCK_TERM_COUNT`.
    pub(super) fn encode(entries: &[FixedTermInfo]) -> Self {
        assert!(
            !entries.is_empty() && entries.len() <= BLOCK_TERM_COUNT,
            "FixedTermInfoBlock::encode: entries.len() must be in 1..={}",
            BLOCK_TERM_COUNT
        );
        let term_count = entries.len();

        // Compute block-wide minimum and maximum for each u64 field.
        let mut min_off = u64::MAX;
        let mut max_off = 0u64;
        let mut min_len = u64::MAX;
        let mut max_len = 0u64;
        let mut min_df = u64::MAX;
        let mut max_df = 0u64;
        let mut min_tf = u64::MAX;
        let mut max_tf = 0u64;
        for e in entries {
            min_off = min_off.min(e.posting_offset);
            max_off = max_off.max(e.posting_offset);
            min_len = min_len.min(e.posting_length);
            max_len = max_len.max(e.posting_length);
            min_df = min_df.min(e.doc_frequency);
            max_df = max_df.max(e.doc_frequency);
            min_tf = min_tf.min(e.total_frequency);
            max_tf = max_tf.max(e.total_frequency);
        }

        let posting_offset_nbits = bits_required(max_off - min_off);
        let posting_length_nbits = bits_required(max_len - min_len);
        let doc_frequency_nbits = bits_required(max_df - min_df);
        let total_frequency_nbits = bits_required(max_tf - min_tf);

        // max_score_factor strategy: all-equal or per-entry f32.
        let ref_max_score_factor = entries[0].max_score_factor;
        let ref_bits = ref_max_score_factor.to_bits();
        let all_same = entries
            .iter()
            .all(|e| e.max_score_factor.to_bits() == ref_bits);
        let max_score_factor_size: u8 = if all_same { 0 } else { 32 };

        let bits_per_entry = posting_offset_nbits as usize
            + posting_length_nbits as usize
            + doc_frequency_nbits as usize
            + total_frequency_nbits as usize;
        let bytes_4fields = packed_len(bits_per_entry, term_count);
        let bytes_f32 = factor_len(max_score_factor_size, term_count);

        let mut payload = vec![0u8; bytes_4fields + bytes_f32];

        // Pack 4-field deltas, LSB-first within each byte.
        for (i, entry) in entries.iter().enumerate() {
            let mut bit_offset = bits_per_entry * i;

            write_bits(
                &mut payload[..bytes_4fields],
                bit_offset,
                entry.posting_offset - min_off,
                posting_offset_nbits,
            );
            bit_offset += posting_offset_nbits as usize;

            write_bits(
                &mut payload[..bytes_4fields],
                bit_offset,
                entry.posting_length - min_len,
                posting_length_nbits,
            );
            bit_offset += posting_length_nbits as usize;

            write_bits(
                &mut payload[..bytes_4fields],
                bit_offset,
                entry.doc_frequency - min_df,
                doc_frequency_nbits,
            );
            bit_offset += doc_frequency_nbits as usize;

            write_bits(
                &mut payload[..bytes_4fields],
                bit_offset,
                entry.total_frequency - min_tf,
                total_frequency_nbits,
            );
        }

        // Pack the optional f32 array.
        if max_score_factor_size == 32 {
            for (i, entry) in entries.iter().enumerate() {
                let off = bytes_4fields + 4 * i;
                payload[off..off + 4].copy_from_slice(&entry.max_score_factor.to_le_bytes());
            }
        }

        FixedTermInfoBlock {
            term_count: term_count as u16,
            block_min_posting_offset: min_off,
            block_min_posting_length: min_len,
            block_min_doc_frequency: min_df,
            block_min_total_frequency: min_tf,
            ref_max_score_factor,
            posting_offset_nbits,
            posting_length_nbits,
            doc_frequency_nbits,
            total_frequency_nbits,
            max_score_factor_size,
            payload,
        }
    }

    /// Check the header against the payload before any entry is decoded.
    ///
    /// The writer derives every bit width from a `u64` delta, writes a
    /// `max_score_factor_size` of 0 or 32, and sizes the payload from
    /// both. A block that breaks any of these would make
    /// [`Self::decode_at`] shift past 63 bits or read beyond the payload,
    /// so it is rejected here, once per block (Issue #1224).
    ///
    /// # Errors
    ///
    /// [`LaurusError::Index`] when a bit width exceeds 64, the factor size
    /// is neither 0 nor 32, or the payload length differs from what the
    /// header describes.
    pub(super) fn validate(&self) -> Result<()> {
        for (field, nbits) in [
            ("posting_offset", self.posting_offset_nbits),
            ("posting_length", self.posting_length_nbits),
            ("doc_frequency", self.doc_frequency_nbits),
            ("total_frequency", self.total_frequency_nbits),
        ] {
            if nbits > 64 {
                return Err(LaurusError::index(format!(
                    "BlockReader: FixedTermInfoBlock packs {field} in {nbits} bits, more than 64 \
                     — segment is corrupted"
                )));
            }
        }
        if !matches!(self.max_score_factor_size, 0 | 32) {
            return Err(LaurusError::index(format!(
                "BlockReader: FixedTermInfoBlock max_score_factor size {} is neither 0 nor 32 \
                 — segment is corrupted",
                self.max_score_factor_size
            )));
        }
        let term_count = self.term_count as usize;
        let expected = packed_len(self.bits_per_entry(), term_count)
            + factor_len(self.max_score_factor_size, term_count);
        if self.payload.len() != expected {
            return Err(LaurusError::index(format!(
                "BlockReader: FixedTermInfoBlock payload is {} bytes but its header describes \
                 {expected} — segment is corrupted",
                self.payload.len()
            )));
        }
        Ok(())
    }

    /// Decode the entry at `inner_offset` (`0..term_count`).
    ///
    /// The block must have passed [`Self::validate`], which keeps every
    /// read within the payload.
    ///
    /// # Errors
    ///
    /// [`LaurusError::Index`] when a field's block minimum plus its delta
    /// overflows `u64` (Issue #1224). The header cannot rule this out: a
    /// valid block may pack a 64-bit delta over a non-zero minimum.
    ///
    /// # Panics
    ///
    /// Panics if `inner_offset >= self.term_count`.
    pub(super) fn decode_at(&self, inner_offset: usize) -> Result<FixedTermInfo> {
        assert!(
            inner_offset < self.term_count as usize,
            "decode_at: inner_offset {} >= term_count {}",
            inner_offset,
            self.term_count
        );

        let bits_per_entry = self.bits_per_entry();
        let bytes_4fields = packed_len(bits_per_entry, self.term_count as usize);

        let mut bit_offset = bits_per_entry * inner_offset;

        let posting_offset_delta = read_bits(
            &self.payload[..bytes_4fields],
            bit_offset,
            self.posting_offset_nbits,
        );
        bit_offset += self.posting_offset_nbits as usize;

        let posting_length_delta = read_bits(
            &self.payload[..bytes_4fields],
            bit_offset,
            self.posting_length_nbits,
        );
        bit_offset += self.posting_length_nbits as usize;

        let doc_frequency_delta = read_bits(
            &self.payload[..bytes_4fields],
            bit_offset,
            self.doc_frequency_nbits,
        );
        bit_offset += self.doc_frequency_nbits as usize;

        let total_frequency_delta = read_bits(
            &self.payload[..bytes_4fields],
            bit_offset,
            self.total_frequency_nbits,
        );

        let max_score_factor = if self.max_score_factor_size == 0 {
            self.ref_max_score_factor
        } else {
            let off = bytes_4fields + 4 * inner_offset;
            let bytes: [u8; 4] = self.payload[off..off + 4]
                .try_into()
                .expect("f32 slice must be exactly 4 bytes");
            f32::from_le_bytes(bytes)
        };

        let add = |field: &str, min: u64, delta: u64| {
            min.checked_add(delta).ok_or_else(|| {
                LaurusError::index(format!(
                    "term dictionary: {field} {min} + {delta} overflows u64 — segment is corrupted"
                ))
            })
        };
        Ok(FixedTermInfo {
            posting_offset: add(
                "posting_offset",
                self.block_min_posting_offset,
                posting_offset_delta,
            )?,
            posting_length: add(
                "posting_length",
                self.block_min_posting_length,
                posting_length_delta,
            )?,
            doc_frequency: add(
                "doc_frequency",
                self.block_min_doc_frequency,
                doc_frequency_delta,
            )?,
            total_frequency: add(
                "total_frequency",
                self.block_min_total_frequency,
                total_frequency_delta,
            )?,
            max_score_factor,
        })
    }

    /// Bits each entry occupies in the bit-packed section.
    fn bits_per_entry(&self) -> usize {
        self.posting_offset_nbits as usize
            + self.posting_length_nbits as usize
            + self.doc_frequency_nbits as usize
            + self.total_frequency_nbits as usize
    }
}

/// Byte length of the bit-packed delta section: `term_count` entries of
/// `bits_per_entry` bits each, rounded up to whole bytes. Shared by the
/// encoder and [`FixedTermInfoBlock::validate`] so the two cannot drift.
fn packed_len(bits_per_entry: usize, term_count: usize) -> usize {
    (bits_per_entry * term_count).div_ceil(8)
}

/// Byte length of the optional `f32` `max_score_factor` section that
/// follows the bit-packed deltas.
fn factor_len(max_score_factor_size: u8, term_count: usize) -> usize {
    if max_score_factor_size == 0 {
        0
    } else {
        4 * term_count
    }
}

/// Number of bits required to represent `value` (0 → 0, 1 → 1, 2-3 → 2,
/// 4-7 → 3, …).
fn bits_required(value: u64) -> u8 {
    if value == 0 {
        0
    } else {
        (64 - value.leading_zeros()) as u8
    }
}

/// Write `value` (low `nbits` bits) into `bits` starting at
/// `bit_offset`, LSB-first within each byte. The destination region
/// must be zero-initialised.
///
/// `nbits == 0` is a no-op.
fn write_bits(bits: &mut [u8], bit_offset: usize, value: u64, nbits: u8) {
    if nbits == 0 {
        return;
    }
    debug_assert!(
        nbits <= 64 && (nbits == 64 || value < (1u64 << nbits)),
        "value {value} does not fit in {nbits} bits"
    );

    let mut remaining = nbits;
    let mut value = value;
    let mut byte_idx = bit_offset / 8;
    let mut bit_in_byte = (bit_offset % 8) as u8;

    while remaining > 0 {
        let take = remaining.min(8 - bit_in_byte);
        // u16 intermediate avoids `1u8 << 8` overflow when take == 8.
        let mask: u8 = ((1u16 << take) - 1) as u8;
        let chunk = (value as u8) & mask;
        bits[byte_idx] |= chunk << bit_in_byte;
        // Avoid `value >> 64` (undefined in Rust): when take == 64 the
        // remaining bits are zero anyway.
        if take == 64 {
            value = 0;
        } else {
            value >>= take;
        }
        remaining -= take;
        bit_in_byte += take;
        if bit_in_byte == 8 {
            bit_in_byte = 0;
            byte_idx += 1;
        }
    }
}

/// Read `nbits` bits from `bits` starting at `bit_offset`, LSB-first
/// within each byte. Returns the value as `u64`.
///
/// `nbits == 0` returns `0`.
fn read_bits(bits: &[u8], bit_offset: usize, nbits: u8) -> u64 {
    if nbits == 0 {
        return 0;
    }
    debug_assert!(nbits <= 64);

    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut remaining = nbits;
    let mut byte_idx = bit_offset / 8;
    let mut bit_in_byte = (bit_offset % 8) as u8;

    while remaining > 0 {
        let take = remaining.min(8 - bit_in_byte);
        // u16 intermediate avoids `1u8 << 8` overflow when take == 8.
        let mask: u8 = ((1u16 << take) - 1) as u8;
        let chunk = (bits[byte_idx] >> bit_in_byte) & mask;
        result |= u64::from(chunk) << shift;
        shift += u32::from(take);
        remaining -= take;
        bit_in_byte += take;
        if bit_in_byte == 8 {
            bit_in_byte = 0;
            byte_idx += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fti(po: u64, pl: u64, df: u64, tf: u64, msf: f32) -> FixedTermInfo {
        FixedTermInfo {
            posting_offset: po,
            posting_length: pl,
            doc_frequency: df,
            total_frequency: tf,
            max_score_factor: msf,
        }
    }

    #[test]
    fn bits_required_basic() {
        assert_eq!(bits_required(0), 0);
        assert_eq!(bits_required(1), 1);
        assert_eq!(bits_required(2), 2);
        assert_eq!(bits_required(3), 2);
        assert_eq!(bits_required(4), 3);
        assert_eq!(bits_required(7), 3);
        assert_eq!(bits_required(8), 4);
        assert_eq!(bits_required(255), 8);
        assert_eq!(bits_required(256), 9);
        assert_eq!(bits_required(u64::MAX), 64);
    }

    #[test]
    fn write_read_bits_round_trip() {
        // Pack 4 values of varying widths into a buffer and read back.
        let mut buf = vec![0u8; 16];
        write_bits(&mut buf, 0, 0xA5, 8); // 8 bits
        write_bits(&mut buf, 8, 0b1011, 4); // 4 bits
        write_bits(&mut buf, 12, 0xDEAD, 16); // 16 bits
        write_bits(&mut buf, 28, 0, 0); // 0 bits no-op
        write_bits(&mut buf, 28, 1, 1); // 1 bit

        assert_eq!(read_bits(&buf, 0, 8), 0xA5);
        assert_eq!(read_bits(&buf, 8, 4), 0b1011);
        assert_eq!(read_bits(&buf, 12, 16), 0xDEAD);
        assert_eq!(read_bits(&buf, 28, 1), 1);
        assert_eq!(read_bits(&buf, 0, 0), 0);
    }

    #[test]
    fn write_read_bits_wide_values() {
        let mut buf = vec![0u8; 32];
        let v = 0xDEAD_BEEF_CAFE_BABE_u64;
        write_bits(&mut buf, 7, v, 64);
        assert_eq!(read_bits(&buf, 7, 64), v);
    }

    #[test]
    fn encode_decode_single_entry() {
        let entries = [fti(100, 50, 5, 20, 1.5)];
        let block = FixedTermInfoBlock::encode(&entries);

        // With one entry, all deltas are 0, so all nbits should be 0.
        assert_eq!(block.posting_offset_nbits, 0);
        assert_eq!(block.posting_length_nbits, 0);
        assert_eq!(block.doc_frequency_nbits, 0);
        assert_eq!(block.total_frequency_nbits, 0);
        assert_eq!(block.max_score_factor_size, 0);

        let decoded = block.decode_at(0).unwrap();
        assert_eq!(decoded, entries[0]);
    }

    #[test]
    fn encode_decode_all_same_values() {
        let entries: Vec<FixedTermInfo> = (0..64).map(|_| fti(100, 50, 5, 20, 1.5)).collect();
        let block = FixedTermInfoBlock::encode(&entries);
        assert_eq!(block.posting_offset_nbits, 0);
        assert_eq!(block.payload.len(), 0); // no payload needed
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(block.decode_at(i).unwrap(), *e);
        }
    }

    #[test]
    fn encode_decode_small_deltas() {
        // posting_offset increases by 1 each entry; everything else
        // stays constant.
        let entries: Vec<FixedTermInfo> = (0..8)
            .map(|i| fti(1000 + i as u64, 50, 5, 20, 1.5))
            .collect();
        let block = FixedTermInfoBlock::encode(&entries);
        assert_eq!(block.block_min_posting_offset, 1000);
        // 8 entries with deltas 0..7, max delta = 7, requires 3 bits.
        assert_eq!(block.posting_offset_nbits, 3);
        assert_eq!(block.posting_length_nbits, 0);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(block.decode_at(i).unwrap(), *e);
        }
    }

    #[test]
    fn encode_decode_per_entry_max_score_factor() {
        let entries: Vec<FixedTermInfo> = (0..16)
            .map(|i| fti(1000 + i as u64, 50, 5, 20, 1.0 + i as f32 * 0.1))
            .collect();
        let block = FixedTermInfoBlock::encode(&entries);
        assert_eq!(block.max_score_factor_size, 32);
        // 16 entries × 4 bytes f32 + bit-packed 4-field section
        assert!(block.payload.len() >= 16 * 4);
        for (i, _) in entries.iter().enumerate() {
            let decoded = block.decode_at(i).unwrap();
            assert_eq!(decoded.posting_offset, 1000 + i as u64);
            assert!((decoded.max_score_factor - (1.0 + i as f32 * 0.1)).abs() < 1e-6);
        }
    }

    #[test]
    fn encode_decode_128_entries_random() {
        let mut state: u64 = 0xCAFEBABE_DEADBEEF;
        let mut next_u64 = || -> u64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };

        let entries: Vec<FixedTermInfo> = (0..BLOCK_TERM_COUNT)
            .map(|i| {
                fti(
                    1000 + i as u64 * 16,
                    100 + (next_u64() % 1000),
                    1 + (next_u64() % 100),
                    1 + (next_u64() % 1000),
                    f32::from_bits(next_u64() as u32),
                )
            })
            .collect();
        let block = FixedTermInfoBlock::encode(&entries);
        for (i, e) in entries.iter().enumerate() {
            let d = block.decode_at(i).unwrap();
            assert_eq!(d.posting_offset, e.posting_offset);
            assert_eq!(d.posting_length, e.posting_length);
            assert_eq!(d.doc_frequency, e.doc_frequency);
            assert_eq!(d.total_frequency, e.total_frequency);
            assert_eq!(
                d.max_score_factor.to_bits(),
                e.max_score_factor.to_bits(),
                "max_score_factor mismatch at i={i}"
            );
        }
    }

    #[test]
    fn encode_decode_large_max_delta_in_one_field() {
        // posting_length spans 0..u32::MAX in 4 entries → requires 32
        // bits. Other fields stay narrow.
        let entries = [
            fti(0, 0, 1, 1, 0.0),
            fti(100, u32::MAX as u64, 2, 2, 0.0),
            fti(200, 1, 3, 3, 0.0),
            fti(300, 100_000, 4, 4, 0.0),
        ];
        let block = FixedTermInfoBlock::encode(&entries);
        assert_eq!(block.posting_length_nbits, 32);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(block.decode_at(i).unwrap(), *e);
        }
    }

    #[test]
    #[should_panic(expected = "entries.len() must be in 1..=128")]
    fn encode_panics_on_empty_entries() {
        FixedTermInfoBlock::encode(&[]);
    }

    #[test]
    #[should_panic(expected = "entries.len() must be in 1..=128")]
    fn encode_panics_on_oversize_entries() {
        let entries: Vec<FixedTermInfo> = (0..BLOCK_TERM_COUNT + 1)
            .map(|_| fti(0, 0, 0, 0, 0.0))
            .collect();
        FixedTermInfoBlock::encode(&entries);
    }

    #[test]
    #[should_panic(expected = "decode_at: inner_offset")]
    fn decode_panics_on_out_of_range() {
        let entries = [fti(0, 0, 0, 0, 0.0); 4];
        let block = FixedTermInfoBlock::encode(&entries);
        let _ = block.decode_at(4);
    }

    // Issue #1224: a header that disagrees with its payload is rejected
    // before `decode_at` can shift past 63 bits or read beyond the payload.

    fn assert_corrupted<T: std::fmt::Debug>(result: Result<T>, label: &str) {
        match result {
            Ok(value) => panic!("expected a corruption error mentioning {label:?}, got {value:?}"),
            Err(LaurusError::Index(msg)) => {
                assert!(msg.contains("corrupted") && msg.contains(label), "{msg}");
            }
            Err(other) => panic!("expected Index error, got {other:?}"),
        }
    }

    /// A four-entry block whose deltas need a few bits per field and whose
    /// factors differ, so every section of the payload is present.
    fn sample_block() -> FixedTermInfoBlock {
        let entries = [
            fti(100, 10, 1, 1, 1.0),
            fti(200, 20, 2, 4, 2.0),
            fti(300, 30, 3, 9, 3.0),
            fti(400, 40, 4, 16, 4.0),
        ];
        FixedTermInfoBlock::encode(&entries)
    }

    /// Every block the encoder writes passes validation.
    #[test]
    fn encoded_blocks_validate() {
        sample_block().validate().unwrap();
        FixedTermInfoBlock::encode(&[fti(0, 0, 0, 0, 0.0)])
            .validate()
            .unwrap();
        FixedTermInfoBlock::encode(&[fti(1, 0, 0, 0, 0.0), fti(u64::MAX, 0, 0, 0, 0.0)])
            .validate()
            .unwrap();
    }

    /// A bit width above 64 is rejected (Issue #1224).
    #[test]
    fn a_bit_width_above_64_is_rejected() {
        let mut block = sample_block();
        block.doc_frequency_nbits = 65;
        assert_corrupted(block.validate(), "more than 64");
    }

    /// A factor size other than 0 or 32 is rejected; `decode_at` would
    /// otherwise read an `f32` past the payload (Issue #1224).
    #[test]
    fn a_factor_size_other_than_0_or_32_is_rejected() {
        let mut block = sample_block();
        block.max_score_factor_size = 7;
        assert_corrupted(block.validate(), "neither 0 nor 32");
    }

    /// A payload shorter or longer than the header describes is rejected
    /// (Issue #1224).
    #[test]
    fn a_payload_of_the_wrong_length_is_rejected() {
        let mut short = sample_block();
        short.payload.pop();
        assert_corrupted(short.validate(), "payload is");

        let mut long = sample_block();
        long.payload.push(0);
        assert_corrupted(long.validate(), "payload is");
    }

    /// A block minimum plus its delta that overflows `u64` is rejected
    /// (Issue #1224). Valid data can pack a 64-bit delta over a non-zero
    /// minimum, so only the addition itself can catch this.
    #[test]
    fn a_minimum_plus_delta_overflowing_u64_is_rejected() {
        let mut block = FixedTermInfoBlock::encode(&[fti(0, 0, 0, 0, 0.0), fti(0, 0, 1, 0, 0.0)]);
        block.validate().unwrap();
        block.block_min_doc_frequency = u64::MAX;
        block.decode_at(0).unwrap();
        assert_corrupted(block.decode_at(1), "doc_frequency");
    }
}
