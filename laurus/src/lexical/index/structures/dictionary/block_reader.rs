//! In-memory reader and iterator over the BlockSection of a `.dict` file.
//!
//! A [`BlockReader`] points to the bytes of a single block within the
//! BlockSection. It can perform an in-block lookup (linear front-coded
//! scan) or yield each `(term_bytes, TermInfo)` pair sequentially.
//!
//! [`BlockSectionIter`] walks the entire BlockSection, lazily decoding
//! one block at a time so that `iter()` does not need to materialise
//! the whole dictionary into memory.
//!
//! # Block layout (within `BlockSection`)
//!
//! ```text
//! [term_count:                varint (u16 bounded)]
//! [front_coded_terms_len:     varint]
//! [front_coded_terms_bytes:   u8 × front_coded_terms_len]
//! [FixedTermInfoBlock serialized]
//!   block_min_posting_offset:    u64 LE  (8 bytes)
//!   block_min_posting_length:    u64 LE  (8 bytes)
//!   block_min_doc_frequency:     u64 LE  (8 bytes)
//!   block_min_total_frequency:   u64 LE  (8 bytes)
//!   ref_max_score_factor:        f32 LE  (4 bytes)
//!   posting_offset_nbits:        u8
//!   posting_length_nbits:        u8
//!   doc_frequency_nbits:         u8
//!   total_frequency_nbits:       u8
//!   max_score_factor_size:       u8
//!   payload_len:                 u32 LE  (4 bytes)
//!   payload:                     u8 × payload_len
//! [BlockMaxData serialized]
//!   bm_offsets:                  u32 LE × (term_count + 1)
//!   bm_data_len:                 u32 LE  (4 bytes)
//!   bm_data:                     u8 × bm_data_len
//! ```

// Skeleton wiring: these `pub(super)` items are consumed by
// `builder` (Phase 6). Until then, `cargo check` flags some of them
// as dead code. Remove this allow once the integration is complete.
#![allow(dead_code)]

use crate::error::{LaurusError, Result};
use crate::lexical::index::structures::dictionary::TermInfo;

use super::block_max_data::{BLOCK_MAX_BYTES, BlockMaxData};
use super::front_coding::FrontCodingDecoder;
use super::term_info_block::FixedTermInfoBlock;

/// Reader for a single block parsed out of the BlockSection.
pub(super) struct BlockReader<'a> {
    /// Number of terms in this block (`1..=BLOCK_TERM_COUNT`).
    pub(super) block_term_count: u16,
    /// Front-coded term bytes section of the block (sub-slice of the
    /// BlockSection).
    pub(super) term_bytes: &'a [u8],
    /// Bit-packed fixed-size `TermInfo` portion (payload bytes copied
    /// from the BlockSection — see module-level note about
    /// zero-copy).
    pub(super) term_info_block: FixedTermInfoBlock,
    /// Variable-length `block_max` storage (offsets + bytes copied
    /// from the BlockSection).
    pub(super) block_max_data: BlockMaxData,
}

impl<'a> BlockReader<'a> {
    /// Parse a block starting at `cursor` within the BlockSection
    /// `bytes`. Returns the parsed reader and the cursor advanced past
    /// the block.
    pub(super) fn parse(bytes: &'a [u8], cursor: usize) -> Result<(Self, usize)> {
        let mut c = cursor;

        let term_count_u64 = read_varint(bytes, &mut c)?;
        let term_count = u16::try_from(term_count_u64).map_err(|_| {
            LaurusError::index(format!(
                "BlockReader: term_count {term_count_u64} exceeds u16::MAX"
            ))
        })?;
        if term_count == 0 {
            return Err(LaurusError::index("BlockReader: empty block"));
        }

        let fc_len = read_varint(bytes, &mut c)?;
        // Compared before it is narrowed, which would truncate it on a
        // 32-bit target, and without `c + fc_len`, which could overflow
        // (Issue #1220).
        if fc_len > bytes.len().saturating_sub(c) as u64 {
            return Err(LaurusError::index(
                "BlockReader: front-coded section overruns BlockSection",
            ));
        }
        let fc_len = fc_len as usize;
        let term_bytes = &bytes[c..c + fc_len];
        c += fc_len;

        // FixedTermInfoBlock fixed header (36 bytes block_min + ref f32,
        // 5 bytes nbits + size, 4 bytes payload_len = 45 bytes).
        const FTIB_FIXED_HEADER_BYTES: usize = 8 + 8 + 8 + 8 + 4 + 5 + 4;
        if c + FTIB_FIXED_HEADER_BYTES > bytes.len() {
            return Err(LaurusError::index(
                "BlockReader: FixedTermInfoBlock header overruns BlockSection",
            ));
        }

        let block_min_posting_offset = read_u64_le(bytes, &mut c);
        let block_min_posting_length = read_u64_le(bytes, &mut c);
        let block_min_doc_frequency = read_u64_le(bytes, &mut c);
        let block_min_total_frequency = read_u64_le(bytes, &mut c);
        let ref_max_score_factor = read_f32_le(bytes, &mut c);
        let posting_offset_nbits = read_u8(bytes, &mut c);
        let posting_length_nbits = read_u8(bytes, &mut c);
        let doc_frequency_nbits = read_u8(bytes, &mut c);
        let total_frequency_nbits = read_u8(bytes, &mut c);
        let max_score_factor_size = read_u8(bytes, &mut c);
        let payload_len = read_u32_le(bytes, &mut c) as usize;

        if c + payload_len > bytes.len() {
            return Err(LaurusError::index(
                "BlockReader: FixedTermInfoBlock payload overruns BlockSection",
            ));
        }
        let payload = bytes[c..c + payload_len].to_vec();
        c += payload_len;

        let term_info_block = FixedTermInfoBlock {
            term_count,
            block_min_posting_offset,
            block_min_posting_length,
            block_min_doc_frequency,
            block_min_total_frequency,
            ref_max_score_factor,
            posting_offset_nbits,
            posting_length_nbits,
            doc_frequency_nbits,
            total_frequency_nbits,
            max_score_factor_size,
            payload,
        };

        // BlockMaxData: (term_count + 1) u32 offsets, then bm_data_len + bytes.
        let offsets_count = term_count as usize + 1;
        let offsets_bytes = offsets_count * 4;
        if c + offsets_bytes + 4 > bytes.len() {
            return Err(LaurusError::index(
                "BlockReader: BlockMaxData header overruns BlockSection",
            ));
        }
        let mut offsets = Vec::with_capacity(offsets_count);
        for _ in 0..offsets_count {
            offsets.push(read_u32_le(bytes, &mut c));
        }
        let bm_data_len = read_u32_le(bytes, &mut c) as usize;
        if c + bm_data_len > bytes.len() {
            return Err(LaurusError::index(
                "BlockReader: BlockMaxData data overruns BlockSection",
            ));
        }
        // The writer's offsets start at 0, never decrease, span whole
        // entries and end at the data's length (Issue #1220).
        // `BlockMaxData::get` sizes a buffer and indexes the data by them, so
        // a corrupt offset would otherwise drive a huge allocation or read
        // out of bounds.
        let spans_are_whole_entries = offsets.windows(2).all(|pair| {
            pair[0] <= pair[1] && ((pair[1] - pair[0]) as usize).is_multiple_of(BLOCK_MAX_BYTES)
        });
        if offsets.first() != Some(&0)
            || !spans_are_whole_entries
            || offsets.last().map(|&last| last as usize) != Some(bm_data_len)
        {
            return Err(LaurusError::index(
                "BlockReader: BlockMaxData offsets are not ascending spans of whole entries \
                 ending at the data length — segment is corrupted",
            ));
        }
        let bm_data = bytes[c..c + bm_data_len].to_vec();
        c += bm_data_len;

        let block_max_data = BlockMaxData {
            offsets,
            data: bm_data,
        };

        Ok((
            BlockReader {
                block_term_count: term_count,
                term_bytes,
                term_info_block,
                block_max_data,
            },
            c,
        ))
    }

    /// Look up `target` within this block. Returns `Some(TermInfo)` on
    /// hit, `None` if not present.
    ///
    /// Performs a linear scan over front-coded term bytes, comparing
    /// each decoded term to `target`. Stops early if a decoded term
    /// is greater than `target` (terms are sorted within a block).
    pub(super) fn lookup(&self, target: &[u8]) -> Option<TermInfo> {
        let mut decoder = FrontCodingDecoder::new(self.term_bytes, self.block_term_count as u32);
        let mut inner: usize = 0;
        while let Some(term) = decoder.next() {
            match term.cmp(target) {
                std::cmp::Ordering::Equal => {
                    return Some(self.materialise_term_info(inner));
                }
                std::cmp::Ordering::Greater => return None,
                std::cmp::Ordering::Less => {
                    inner += 1;
                }
            }
        }
        None
    }

    /// Iterate `(term_bytes_owned, TermInfo)` pairs in sorted order.
    pub(super) fn iter(&self) -> BlockReaderIter<'_, 'a> {
        BlockReaderIter {
            decoder: FrontCodingDecoder::new(self.term_bytes, self.block_term_count as u32),
            term_info_block: &self.term_info_block,
            block_max_data: &self.block_max_data,
            inner: 0,
        }
    }

    /// Build a [`TermInfo`] for the entry at `inner_offset` by
    /// combining the bit-packed fixed fields with the variable-length
    /// `block_max` array.
    fn materialise_term_info(&self, inner_offset: usize) -> TermInfo {
        let fti = self.term_info_block.decode_at(inner_offset);
        let block_max = self.block_max_data.get(inner_offset);
        TermInfo {
            posting_offset: fti.posting_offset,
            posting_length: fti.posting_length,
            doc_frequency: fti.doc_frequency,
            total_frequency: fti.total_frequency,
            max_score_factor: fti.max_score_factor,
            block_max,
        }
    }
}

/// Iterator yielded by [`BlockReader::iter`]. Yields fresh `Vec<u8>`
/// term bytes (copy of the decoder's reusable buffer) so the caller
/// can move the term independently of the iterator.
pub(super) struct BlockReaderIter<'r, 'a: 'r> {
    decoder: FrontCodingDecoder<'a>,
    term_info_block: &'r FixedTermInfoBlock,
    block_max_data: &'r BlockMaxData,
    inner: usize,
}

impl<'r, 'a: 'r> Iterator for BlockReaderIter<'r, 'a> {
    type Item = (Vec<u8>, TermInfo);

    fn next(&mut self) -> Option<Self::Item> {
        let term = self.decoder.next()?;
        let term_owned = term.to_vec();
        let fti = self.term_info_block.decode_at(self.inner);
        let block_max = self.block_max_data.get(self.inner);
        let info = TermInfo {
            posting_offset: fti.posting_offset,
            posting_length: fti.posting_length,
            doc_frequency: fti.doc_frequency,
            total_frequency: fti.total_frequency,
            max_score_factor: fti.max_score_factor,
            block_max,
        };
        self.inner += 1;
        Some((term_owned, info))
    }
}

/// Streaming iterator over the entire BlockSection, yielding
/// `(term_string, TermInfo)` pairs in sorted order.
///
/// Internally walks block-by-block using [`BlockReader`], decoding
/// front-coded term bytes lazily so memory usage stays at one block
/// + one decode buffer regardless of dictionary size.
pub(super) struct BlockSectionIter<'a> {
    /// BlockSection bytes (full).
    bytes: &'a [u8],
    /// Cursor into `bytes` for the next block to parse.
    cursor: usize,
    /// Number of blocks remaining to be parsed (excluding the current
    /// in-flight one).
    blocks_remaining: u32,
    /// Currently-active block reader (or `None` if exhausted / not yet
    /// loaded).
    current: Option<CurrentBlock<'a>>,
}

/// Owned per-block state held by [`BlockSectionIter`].
///
/// Holds a stateful [`FrontCodingDecoder`] alongside the parsed block
/// metadata so consecutive `next()` calls advance through the block in
/// `O(1)` per term instead of re-decoding the front-coded prefix from
/// the start of the block each time. Both `decoder` and the
/// `term_info_block` / `block_max_data` borrows live for `'a`, the
/// lifetime of the underlying BlockSection bytes — there is no
/// self-reference between fields.
struct CurrentBlock<'a> {
    block_term_count: u16,
    decoder: FrontCodingDecoder<'a>,
    term_info_block: FixedTermInfoBlock,
    block_max_data: BlockMaxData,
    inner: usize,
}

impl<'a> BlockSectionIter<'a> {
    /// Create a new iterator over `bytes` covering exactly `block_count`
    /// blocks.
    pub(super) fn new(bytes: &'a [u8], block_count: u32) -> Self {
        BlockSectionIter {
            bytes,
            cursor: 0,
            blocks_remaining: block_count,
            current: None,
        }
    }

    /// Advance into the next block, populating `self.current`. Returns
    /// `Ok(false)` when no further blocks remain.
    fn advance_block(&mut self) -> Result<bool> {
        if self.blocks_remaining == 0 {
            self.current = None;
            return Ok(false);
        }
        let (reader, next_cursor) = BlockReader::parse(self.bytes, self.cursor)?;
        self.cursor = next_cursor;
        self.blocks_remaining -= 1;

        let decoder = FrontCodingDecoder::new(reader.term_bytes, reader.block_term_count as u32);
        self.current = Some(CurrentBlock {
            block_term_count: reader.block_term_count,
            decoder,
            term_info_block: reader.term_info_block,
            block_max_data: reader.block_max_data,
            inner: 0,
        });
        Ok(true)
    }
}

impl<'a> Iterator for BlockSectionIter<'a> {
    type Item = (String, TermInfo);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current.is_none() {
                match self.advance_block() {
                    Ok(true) => {}
                    Ok(false) => return None,
                    Err(_) => return None, // corruption: end iteration silently
                }
            }
            // SAFETY of unwrap: just set in advance_block.
            let cb = self.current.as_mut().unwrap();
            if cb.inner >= cb.block_term_count as usize {
                self.current = None;
                continue;
            }

            // Single decoder advance per call. Buffer is reused across
            // calls, so per-step cost is the front-coding decode
            // (≈ 5–10 ns) rather than a `O(N²)` full-block re-walk.
            let term_bytes = match cb.decoder.next() {
                Some(b) => b,
                None => {
                    // Decoder exhausted unexpectedly — treat as
                    // corruption and end iteration silently.
                    self.current = None;
                    return None;
                }
            };
            let term_string = String::from_utf8_lossy(term_bytes).into_owned();
            let fti = cb.term_info_block.decode_at(cb.inner);
            let block_max = cb.block_max_data.get(cb.inner);
            let info = TermInfo {
                posting_offset: fti.posting_offset,
                posting_length: fti.posting_length,
                doc_frequency: fti.doc_frequency,
                total_frequency: fti.total_frequency,
                max_score_factor: fti.max_score_factor,
                block_max,
            };
            cb.inner += 1;
            return Some((term_string, info));
        }
    }
}

// ---------- byte-level helpers ----------

/// Read an unsigned LEB128 varint at `*cursor`, advancing the cursor.
fn read_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *cursor >= bytes.len() {
            return Err(LaurusError::index("BlockReader: truncated varint"));
        }
        let byte = bytes[*cursor];
        *cursor += 1;
        result |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 64 {
            return Err(LaurusError::index("BlockReader: varint overflow"));
        }
    }
}

fn read_u64_le(bytes: &[u8], cursor: &mut usize) -> u64 {
    let v = u64::from_le_bytes(
        bytes[*cursor..*cursor + 8]
            .try_into()
            .expect("8-byte slice"),
    );
    *cursor += 8;
    v
}

fn read_u32_le(bytes: &[u8], cursor: &mut usize) -> u32 {
    let v = u32::from_le_bytes(
        bytes[*cursor..*cursor + 4]
            .try_into()
            .expect("4-byte slice"),
    );
    *cursor += 4;
    v
}

fn read_f32_le(bytes: &[u8], cursor: &mut usize) -> f32 {
    let v = f32::from_le_bytes(
        bytes[*cursor..*cursor + 4]
            .try_into()
            .expect("4-byte slice"),
    );
    *cursor += 4;
    v
}

fn read_u8(bytes: &[u8], cursor: &mut usize) -> u8 {
    let v = bytes[*cursor];
    *cursor += 1;
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::index::structures::dictionary::BlockMax;

    use super::super::front_coding::encode_block_terms;
    use super::super::term_info_block::FixedTermInfo;

    fn fti(po: u64, pl: u64, df: u64, tf: u64, msf: f32) -> FixedTermInfo {
        FixedTermInfo {
            posting_offset: po,
            posting_length: pl,
            doc_frequency: df,
            total_frequency: tf,
            max_score_factor: msf,
        }
    }

    fn sample_block_max(doc: u64, factor: f32) -> BlockMax {
        BlockMax {
            last_doc_id: doc,
            max_factor: factor,
        }
    }

    /// Encode one block's bytes (BlockHeader + TermBytes +
    /// FixedTermInfoBlock + BlockMaxData) into `out`. Used by tests
    /// to build a synthetic BlockSection without depending on the
    /// (Phase 6) builder.
    fn encode_block_into(
        out: &mut Vec<u8>,
        terms: &[&[u8]],
        infos: &[(FixedTermInfo, Vec<BlockMax>)],
    ) {
        assert_eq!(terms.len(), infos.len());
        let term_count = terms.len();
        // Term count varint.
        write_varint_into(out, term_count as u64);
        // Front-coded term bytes.
        let fc = encode_block_terms(terms);
        write_varint_into(out, fc.len() as u64);
        out.extend_from_slice(&fc);
        // FixedTermInfoBlock.
        let fixed: Vec<FixedTermInfo> = infos.iter().map(|(f, _)| *f).collect();
        let ftib = FixedTermInfoBlock::encode(&fixed);
        out.extend_from_slice(&ftib.block_min_posting_offset.to_le_bytes());
        out.extend_from_slice(&ftib.block_min_posting_length.to_le_bytes());
        out.extend_from_slice(&ftib.block_min_doc_frequency.to_le_bytes());
        out.extend_from_slice(&ftib.block_min_total_frequency.to_le_bytes());
        out.extend_from_slice(&ftib.ref_max_score_factor.to_le_bytes());
        out.push(ftib.posting_offset_nbits);
        out.push(ftib.posting_length_nbits);
        out.push(ftib.doc_frequency_nbits);
        out.push(ftib.total_frequency_nbits);
        out.push(ftib.max_score_factor_size);
        out.extend_from_slice(&(ftib.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&ftib.payload);
        // BlockMaxData.
        let per_term: Vec<Vec<BlockMax>> = infos.iter().map(|(_, bm)| bm.clone()).collect();
        let bmd = BlockMaxData::encode(&per_term);
        for off in &bmd.offsets {
            out.extend_from_slice(&off.to_le_bytes());
        }
        out.extend_from_slice(&(bmd.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&bmd.data);
    }

    fn write_varint_into(buf: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            buf.push(((value as u8) & 0x7F) | 0x80);
            value >>= 7;
        }
        buf.push(value as u8);
    }

    /// Build an in-memory BlockSection containing two blocks for the
    /// integration tests below. Returns the bytes plus the test
    /// fixtures so tests can verify lookups.
    fn build_two_block_section() -> (Vec<u8>, Vec<(Vec<u8>, TermInfo)>) {
        // Block 0: "alpha", "beta", "gamma" (sorted).
        // Block 1: "delta", "epsilon", "zeta" — note "delta" < "gamma"
        //          would violate the global sort order, so we use
        //          "kappa", "lambda", "mu" instead.
        let block0_terms: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma"];
        let block0_infos = vec![
            (
                fti(0, 100, 5, 50, 1.5),
                vec![sample_block_max(10, 1.5), sample_block_max(20, 2.5)],
            ),
            (fti(100, 80, 3, 40, 1.5), Vec::new()),
            (fti(180, 200, 8, 100, 2.0), vec![sample_block_max(30, 3.5)]),
        ];

        let block1_terms: Vec<&[u8]> = vec![b"kappa", b"lambda", b"mu"];
        let block1_infos = vec![
            (fti(380, 50, 1, 1, 0.5), Vec::new()),
            (fti(430, 70, 2, 5, 0.5), Vec::new()),
            (fti(500, 90, 4, 8, 0.5), vec![sample_block_max(99, 9.9)]),
        ];

        let mut section = Vec::new();
        encode_block_into(&mut section, &block0_terms, &block0_infos);
        encode_block_into(&mut section, &block1_terms, &block1_infos);

        let mut expected: Vec<(Vec<u8>, TermInfo)> = Vec::new();
        for (term, (f, bm)) in block0_terms.iter().zip(block0_infos.iter()) {
            expected.push((
                term.to_vec(),
                TermInfo {
                    posting_offset: f.posting_offset,
                    posting_length: f.posting_length,
                    doc_frequency: f.doc_frequency,
                    total_frequency: f.total_frequency,
                    max_score_factor: f.max_score_factor,
                    block_max: bm.clone(),
                },
            ));
        }
        for (term, (f, bm)) in block1_terms.iter().zip(block1_infos.iter()) {
            expected.push((
                term.to_vec(),
                TermInfo {
                    posting_offset: f.posting_offset,
                    posting_length: f.posting_length,
                    doc_frequency: f.doc_frequency,
                    total_frequency: f.total_frequency,
                    max_score_factor: f.max_score_factor,
                    block_max: bm.clone(),
                },
            ));
        }
        (section, expected)
    }

    /// A block's BlockMaxData offsets must be ascending spans of whole
    /// entries from 0 to the data's length: `BlockMaxData::get` sizes a
    /// buffer and indexes the data by them (Issue #1220).
    #[test]
    fn parse_rejects_block_max_offsets_that_do_not_describe_the_data() {
        let terms: Vec<&[u8]> = vec![b"apple", b"banana", b"cherry"];
        let infos = vec![
            (fti(0, 50, 1, 10, 1.0), Vec::new()),
            (fti(50, 60, 2, 20, 1.0), vec![sample_block_max(5, 0.5)]),
            (fti(110, 70, 3, 30, 1.0), Vec::new()),
        ];
        let mut section = Vec::new();
        encode_block_into(&mut section, &terms, &infos);
        assert!(BlockReader::parse(&section, 0).is_ok());

        // Offsets [0, 0, 12, 12], then the data's length (12) and its bytes.
        let offsets_at = section.len() - 12 - 4 - 4 * (terms.len() + 1);
        let with_offset = |index: usize, value: u32| {
            let mut patched = section.clone();
            let at = offsets_at + 4 * index;
            patched[at..at + 4].copy_from_slice(&value.to_le_bytes());
            patched
        };
        for (index, value, why) in [
            (0, 12, "does not start at 0"),
            (2, 24, "decreases"),
            (1, 6, "splits an entry"),
            (3, 0, "does not end at the data length"),
        ] {
            let patched = with_offset(index, value);
            match BlockReader::parse(&patched, 0) {
                Err(LaurusError::Index(msg)) => assert!(msg.contains("corrupted"), "{why}: {msg}"),
                Err(other) => panic!("{why}: expected Index error, got {other:?}"),
                Ok(_) => panic!("offsets that {why} must be rejected"),
            }
        }
    }

    #[test]
    fn parse_single_block_round_trip() {
        let terms: Vec<&[u8]> = vec![b"apple", b"banana", b"cherry"];
        let infos = vec![
            (fti(0, 50, 1, 10, 1.0), Vec::new()),
            (fti(50, 60, 2, 20, 1.0), vec![sample_block_max(5, 0.5)]),
            (fti(110, 70, 3, 30, 1.0), Vec::new()),
        ];
        let mut section = Vec::new();
        encode_block_into(&mut section, &terms, &infos);

        let (reader, next_cursor) = BlockReader::parse(&section, 0).unwrap();
        assert_eq!(reader.block_term_count, 3);
        assert_eq!(next_cursor, section.len());

        // Lookup hits.
        let info = reader.lookup(b"banana").unwrap();
        assert_eq!(info.posting_offset, 50);
        assert_eq!(info.block_max.len(), 1);
        assert_eq!(info.block_max[0].last_doc_id, 5);

        // Lookup miss (not in block, alphabetically before).
        assert!(reader.lookup(b"aardvark").is_none());
        // Lookup miss (alphabetically between).
        assert!(reader.lookup(b"blueberry").is_none());
        // Lookup miss (alphabetically after).
        assert!(reader.lookup(b"date").is_none());
    }

    #[test]
    fn block_reader_iter_yields_in_order() {
        let terms: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma"];
        let infos = vec![
            (fti(0, 100, 5, 50, 1.5), Vec::new()),
            (fti(100, 80, 3, 40, 1.5), Vec::new()),
            (fti(180, 200, 8, 100, 2.0), Vec::new()),
        ];
        let mut section = Vec::new();
        encode_block_into(&mut section, &terms, &infos);

        let (reader, _) = BlockReader::parse(&section, 0).unwrap();
        let collected: Vec<(Vec<u8>, TermInfo)> = reader.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].0, b"alpha");
        assert_eq!(collected[1].0, b"beta");
        assert_eq!(collected[2].0, b"gamma");
    }

    #[test]
    fn block_section_iter_walks_two_blocks() {
        let (section, expected) = build_two_block_section();
        let iter = BlockSectionIter::new(&section, 2);
        let collected: Vec<(String, TermInfo)> = iter.collect();
        assert_eq!(collected.len(), expected.len());
        for ((got_term, got_info), (exp_term, exp_info)) in collected.iter().zip(expected.iter()) {
            assert_eq!(got_term.as_bytes(), exp_term.as_slice());
            assert_eq!(got_info.posting_offset, exp_info.posting_offset);
            assert_eq!(got_info.posting_length, exp_info.posting_length);
            assert_eq!(got_info.doc_frequency, exp_info.doc_frequency);
            assert_eq!(got_info.total_frequency, exp_info.total_frequency);
            assert_eq!(
                got_info.max_score_factor.to_bits(),
                exp_info.max_score_factor.to_bits()
            );
            assert_eq!(got_info.block_max.len(), exp_info.block_max.len());
        }
    }

    #[test]
    fn parse_rejects_truncated_section() {
        let terms: Vec<&[u8]> = vec![b"alpha", b"beta"];
        let infos = vec![
            (fti(0, 50, 1, 10, 1.0), Vec::new()),
            (fti(50, 60, 2, 20, 1.0), Vec::new()),
        ];
        let mut section = Vec::new();
        encode_block_into(&mut section, &terms, &infos);
        let truncated = &section[..section.len() - 5];
        assert!(BlockReader::parse(truncated, 0).is_err());
    }

    #[test]
    fn empty_section_iter() {
        let iter = BlockSectionIter::new(&[], 0);
        let collected: Vec<(String, TermInfo)> = iter.collect();
        assert!(collected.is_empty());
    }

    #[test]
    fn varied_term_lengths_in_block() {
        // A block with a short and a 100-byte term in sorted order,
        // exercising the front-coding decoder buffer growth path.
        // "long_aaaa..." sorts before "short" (l < s).
        let mut long_term = b"long_".to_vec();
        long_term.extend(std::iter::repeat_n(b'a', 100));
        let terms: Vec<&[u8]> = vec![long_term.as_slice(), b"short"];
        let infos = vec![
            (fti(0, 50, 1, 1, 0.0), Vec::new()),
            (fti(50, 60, 2, 2, 0.0), Vec::new()),
        ];
        let mut section = Vec::new();
        encode_block_into(&mut section, &terms, &infos);
        let (reader, _) = BlockReader::parse(&section, 0).unwrap();
        assert!(reader.lookup(&long_term).is_some());
        assert!(reader.lookup(b"short").is_some());
        assert!(reader.lookup(b"medium").is_none());
    }

    #[test]
    fn lookup_miss_after_last_block_term() {
        // Target alphabetically greater than every term in the block —
        // must return None without panicking.
        let terms: Vec<&[u8]> = vec![b"alpha", b"beta"];
        let infos = vec![
            (fti(0, 50, 1, 1, 0.0), Vec::new()),
            (fti(50, 60, 2, 2, 0.0), Vec::new()),
        ];
        let mut section = Vec::new();
        encode_block_into(&mut section, &terms, &infos);
        let (reader, _) = BlockReader::parse(&section, 0).unwrap();
        assert!(reader.lookup(b"zulu").is_none());
    }
}
