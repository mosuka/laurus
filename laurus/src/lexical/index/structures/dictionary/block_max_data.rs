//! Per-block storage of variable-length [`crate::lexical::index::structures::dictionary::BlockMax`] arrays.
//!
//! Each term in a dictionary block has its own `Vec<BlockMax>` (used by
//! Block-Max-WAND, #403 PR-C). Lengths vary widely (proportional to
//! `doc_frequency`), so this layer stores the data as:
//!
//! - a parallel `offsets: Vec<u32>` of length `term_count + 1`, where
//!   `data[offsets[i]..offsets[i + 1]]` is the `BlockMax` byte stream
//!   for the i-th term in the block
//! - a flat `data: Vec<u8>` of all `BlockMax` entries concatenated
//!   (12 bytes each: `u64 last_doc_id` little-endian followed by
//!   `f32 max_factor` little-endian)
//!
//! Random access to ordinal `i` is `O(1)`: read `offsets[i..=i+1]`,
//! slice `data`, decode 12 bytes per `BlockMax`.

// Skeleton wiring: these `pub(super)` items are consumed by
// `block_reader` (Phase 5) and `builder` (Phase 6). Until then,
// `cargo check` flags them as dead code. Remove this allow once the
// integration is complete.
#![allow(dead_code)]

use crate::lexical::index::structures::dictionary::BlockMax;

/// Bytes per [`BlockMax`] entry on the wire (`u64 last_doc_id` +
/// `f32 max_factor`).
pub(super) const BLOCK_MAX_BYTES: usize = 12;

/// Variable-length per-term `BlockMax` data for a single dictionary
/// block.
///
/// On disk layout (within the BlockSection of `.dict`):
///
/// ```text
/// [bm_offsets_len: u32]       (= term_count + 1)
/// [bm_offsets:     u32 × bm_offsets_len]
/// [bm_data_len:    u32]
/// [bm_data:        u8 × bm_data_len]
/// ```
pub(super) struct BlockMaxData {
    /// Byte offsets into `data`, length `term_count + 1`. The i-th
    /// term's `BlockMax` byte slice is
    /// `data[offsets[i] as usize..offsets[i + 1] as usize]`.
    pub(super) offsets: Vec<u32>,
    /// Flat byte stream of all `BlockMax` entries concatenated.
    pub(super) data: Vec<u8>,
}

impl BlockMaxData {
    /// Encode `per_term[i]` as the i-th term's `BlockMax` array. The
    /// resulting `offsets` has length `per_term.len() + 1`.
    ///
    /// # Panics
    ///
    /// Panics if the total encoded size would exceed `u32::MAX` (4 GB
    /// per block, far beyond any realistic block).
    pub(super) fn encode(per_term: &[Vec<BlockMax>]) -> Self {
        let mut offsets = Vec::with_capacity(per_term.len() + 1);
        offsets.push(0u32);

        let total_block_max_count: usize = per_term.iter().map(|blocks| blocks.len()).sum();
        let mut data = Vec::with_capacity(total_block_max_count * BLOCK_MAX_BYTES);

        for blocks in per_term {
            for bm in blocks {
                data.extend_from_slice(&bm.last_doc_id.to_le_bytes());
                data.extend_from_slice(&bm.max_factor.to_le_bytes());
            }
            assert!(
                data.len() <= u32::MAX as usize,
                "BlockMaxData: encoded size exceeds u32::MAX"
            );
            offsets.push(data.len() as u32);
        }

        BlockMaxData { offsets, data }
    }

    /// Decode the `BlockMax` array for the i-th term in the block.
    /// Returns an empty `Vec` if the term has no block_max.
    ///
    /// # Panics
    ///
    /// Panics if `inner_offset >= self.term_count()`.
    pub(super) fn get(&self, inner_offset: usize) -> Vec<BlockMax> {
        assert!(
            inner_offset < self.term_count(),
            "BlockMaxData::get: inner_offset {} >= term_count {}",
            inner_offset,
            self.term_count()
        );

        let start = self.offsets[inner_offset] as usize;
        let end = self.offsets[inner_offset + 1] as usize;
        debug_assert!(
            (end - start).is_multiple_of(BLOCK_MAX_BYTES),
            "BlockMaxData: byte range not a multiple of {BLOCK_MAX_BYTES}"
        );

        let count = (end - start) / BLOCK_MAX_BYTES;
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let off = start + i * BLOCK_MAX_BYTES;
            let last_doc_id = u64::from_le_bytes(
                self.data[off..off + 8]
                    .try_into()
                    .expect("8-byte slice for u64"),
            );
            let max_factor = f32::from_le_bytes(
                self.data[off + 8..off + 12]
                    .try_into()
                    .expect("4-byte slice for f32"),
            );
            result.push(BlockMax {
                last_doc_id,
                max_factor,
            });
        }
        result
    }

    /// Number of terms covered by this block_max data (= `offsets.len() - 1`).
    pub(super) fn term_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Total encoded data size in bytes.
    pub(super) fn data_len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bm(doc: u64, factor: f32) -> BlockMax {
        BlockMax {
            last_doc_id: doc,
            max_factor: factor,
        }
    }

    #[test]
    fn empty_block() {
        let data = BlockMaxData::encode(&[]);
        assert_eq!(data.term_count(), 0);
        assert_eq!(data.data_len(), 0);
        assert_eq!(data.offsets, vec![0u32]);
    }

    #[test]
    fn all_terms_empty() {
        let per_term: Vec<Vec<BlockMax>> = vec![Vec::new(); 5];
        let data = BlockMaxData::encode(&per_term);
        assert_eq!(data.term_count(), 5);
        assert_eq!(data.data_len(), 0);
        // offsets all zero
        for &off in &data.offsets {
            assert_eq!(off, 0);
        }
        for i in 0..5 {
            assert!(data.get(i).is_empty());
        }
    }

    #[test]
    fn single_term_with_blocks() {
        let per_term = vec![vec![bm(10, 1.5), bm(20, 2.5), bm(30, 3.5)]];
        let data = BlockMaxData::encode(&per_term);
        assert_eq!(data.term_count(), 1);
        assert_eq!(data.data_len(), 3 * BLOCK_MAX_BYTES);
        assert_eq!(data.offsets, vec![0, 36]);

        let got = data.get(0);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].last_doc_id, 10);
        assert!((got[0].max_factor - 1.5).abs() < 1e-6);
        assert_eq!(got[2].last_doc_id, 30);
    }

    #[test]
    fn mixed_lengths() {
        let per_term = vec![
            Vec::new(),
            vec![bm(100, 0.5)],
            (0..10).map(|i| bm(i, i as f32 * 0.1)).collect(),
            vec![bm(1, 9.9), bm(2, 8.8)],
        ];
        let data = BlockMaxData::encode(&per_term);
        assert_eq!(data.term_count(), 4);

        // term 0: empty
        assert!(data.get(0).is_empty());

        // term 1: 1 entry
        let t1 = data.get(1);
        assert_eq!(t1.len(), 1);
        assert_eq!(t1[0].last_doc_id, 100);
        assert!((t1[0].max_factor - 0.5).abs() < 1e-6);

        // term 2: 10 entries
        let t2 = data.get(2);
        assert_eq!(t2.len(), 10);
        for (i, b) in t2.iter().enumerate() {
            assert_eq!(b.last_doc_id, i as u64);
            assert!((b.max_factor - i as f32 * 0.1).abs() < 1e-6);
        }

        // term 3: 2 entries
        let t3 = data.get(3);
        assert_eq!(t3.len(), 2);
        assert_eq!(t3[0].last_doc_id, 1);
        assert_eq!(t3[1].last_doc_id, 2);
    }

    #[test]
    fn round_trip_128_terms_random() {
        let mut state: u64 = 0xBADC0FFE_E0DDF00D;
        let mut next_u64 = || -> u64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };

        let per_term: Vec<Vec<BlockMax>> = (0..128)
            .map(|i| {
                let count = (i as u64 % 6) as usize; // 0..5 entries per term
                (0..count)
                    .map(|j| bm(next_u64() % 1_000_000, j as f32 * 1.5))
                    .collect()
            })
            .collect();
        let data = BlockMaxData::encode(&per_term);
        for (i, expected) in per_term.iter().enumerate() {
            let got = data.get(i);
            assert_eq!(got.len(), expected.len(), "length mismatch at i={i}");
            for (g, e) in got.iter().zip(expected.iter()) {
                assert_eq!(g.last_doc_id, e.last_doc_id);
                assert_eq!(g.max_factor.to_bits(), e.max_factor.to_bits());
            }
        }
    }

    #[test]
    #[should_panic(expected = "inner_offset 5 >= term_count 3")]
    fn get_panics_on_out_of_range() {
        let data = BlockMaxData::encode(&[Vec::new(), Vec::new(), Vec::new()]);
        let _ = data.get(5);
    }
}
