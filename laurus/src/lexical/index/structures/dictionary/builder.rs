//! Block-encoder helpers used by [`super::TermDictionaryBuilder::build`].
//!
//! The build pipeline (driven from `dictionary.rs`):
//!
//! 1. Convert the builder's `BTreeMap<String, TermInfo>` to a sorted
//!    `Vec<(String, TermInfo)>`.
//! 2. Split into 128-term blocks (the last block may be partial).
//! 3. For each block, call [`encode_block_into`] to append the block's
//!    bytes to a growing `BlockSection` buffer.
//! 4. Build an [`fst::Map`] keyed by each block's last term, valued by
//!    the block's start offset within `BlockSection`.

#![allow(dead_code)]

use crate::lexical::index::structures::dictionary::BlockMax;

use super::block_max_data::BlockMaxData;
use super::front_coding::encode_block_terms;
use super::term_info_block::{BLOCK_TERM_COUNT, FixedTermInfo, FixedTermInfoBlock};

/// Append the encoded bytes of a single block (BlockHeader + TermBytes
/// + FixedTermInfoBlock + BlockMaxData) to `out`.
///
/// The block layout matches [`super::block_reader`]'s parser. See the
/// module-level doc on `block_reader.rs` for the exact byte layout.
///
/// # Arguments
///
/// * `out` - target byte buffer; the encoded block is appended at the
///   current end.
/// * `terms` - sorted term byte slices for this block (`1..=128`
///   entries).
/// * `fixed_infos` - parallel-indexed fixed-size `TermInfo` portion
///   for each term.
/// * `block_max_per_term` - parallel-indexed variable-length
///   `BlockMax` array for each term (each may be empty).
///
/// # Panics
///
/// Panics if the three input slices have different lengths or if
/// `terms.len()` is outside `1..=128`.
pub(super) fn encode_block_into(
    out: &mut Vec<u8>,
    terms: &[&[u8]],
    fixed_infos: &[FixedTermInfo],
    block_max_per_term: &[Vec<BlockMax>],
) {
    assert_eq!(
        terms.len(),
        fixed_infos.len(),
        "terms and fixed_infos must have equal length"
    );
    assert_eq!(
        terms.len(),
        block_max_per_term.len(),
        "terms and block_max_per_term must have equal length"
    );
    assert!(
        !terms.is_empty() && terms.len() <= BLOCK_TERM_COUNT,
        "terms.len() must be in 1..={BLOCK_TERM_COUNT}"
    );

    let term_count = terms.len();

    // BlockHeader: term_count varint.
    write_varint_into(out, term_count as u64);

    // TermBytes: front-coded.
    let fc = encode_block_terms(terms);
    write_varint_into(out, fc.len() as u64);
    out.extend_from_slice(&fc);

    // FixedTermInfoBlock.
    let ftib = FixedTermInfoBlock::encode(fixed_infos);
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
    let bmd = BlockMaxData::encode(block_max_per_term);
    for off in &bmd.offsets {
        out.extend_from_slice(&off.to_le_bytes());
    }
    out.extend_from_slice(&(bmd.data.len() as u32).to_le_bytes());
    out.extend_from_slice(&bmd.data);
}

/// Append the unsigned LEB128 (uleb128) encoding of `value` to `buf`.
///
/// Mirrors the reader-side decoder in
/// [`super::front_coding::read_varint`]. Kept private to the builder
/// module so the writer/reader pair stays in lockstep.
fn write_varint_into(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push(((value as u8) & 0x7F) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}
