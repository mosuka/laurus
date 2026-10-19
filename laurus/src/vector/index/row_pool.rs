//! Building blocks shared by the row pools of the HNSW / Flat / IVF readers:
//! [`crate::vector::index::f16_storage::F16VectorPool`] (Issue
//! [#623](https://github.com/mosuka/laurus/issues/623)) and
//! [`crate::vector::index::quantized_storage::QuantizedVectorPool`] (Issue
//! [#1463](https://github.com/mosuka/laurus/issues/1463)).
//!
//! - [`MappedRows`] reads rows in place from a segment's mapped input, so
//!   a `Lazy` segment keeps no row bytes on the heap and the page cache
//!   decides what stays resident. The pool keeps the input until it is
//!   dropped, which keeps the file mapped; on Windows a segment deleted by
//!   a merge stays on disk until the reader is dropped (the next open
//!   removes leftover files).
//! - [`FieldPositions`] collects the per-field `doc_id -> position` maps.
//!
//! # Runs
//!
//! The rows of one segment form runs of equally spaced records: HNSW and
//! Flat write a single run, IVF one run per non-empty inverted list (each
//! list is preceded by its 4-byte length). Records of v1/v2 segments carry
//! the field name inline, so their spacing also depends on the name length.
//! [`MappedRowsBuilder`] therefore takes each run's stride from the run's
//! own rows: the second row of a run fixes the stride, and a row that does
//! not continue it starts a new run. Every address the table computes is
//! an offset the reader actually parsed.

use std::collections::HashMap;
use std::io::{Seek, SeekFrom};
use std::sync::Arc;

use crate::error::{LaurusError, Result};
use crate::storage::StorageInput;

/// Page size assumed by [`MappedRows::touch`].
const PAGE: usize = 4096;

/// Rows read in place from a segment's mapped input.
///
/// Built by [`MappedRowsBuilder`]; positions are the order in which the
/// rows were pushed.
pub(crate) struct MappedRows {
    /// A clone of the segment's input, positioned at 0 and never moved, so
    /// `as_slice()` spans the whole file.
    input: Box<dyn StorageInput>,
    /// First position of each run.
    run_starts: Vec<u32>,
    /// Byte offset of the first row of each run.
    run_offsets: Vec<u64>,
    /// Bytes from one row to the next within each run (`0` for a run of
    /// one row).
    run_strides: Vec<u64>,
}

impl std::fmt::Debug for MappedRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedRows")
            .field("runs", &self.run_starts.len())
            .finish()
    }
}

impl MappedRows {
    /// The whole mapped file.
    #[inline]
    fn bytes(&self) -> &[u8] {
        self.input
            .as_slice()
            .expect("mapped rows are only built over inputs that expose a slice")
    }

    /// Byte offset of the row at position `pos`.
    #[inline]
    fn offset(&self, pos: u32) -> usize {
        let run = if self.run_starts.len() == 1 {
            0
        } else {
            self.run_starts.partition_point(|&s| s <= pos) - 1
        };
        (self.run_offsets[run] + (pos - self.run_starts[run]) as u64 * self.run_strides[run])
            as usize
    }

    /// The `len` bytes of the row at position `pos`.
    ///
    /// # Panics
    ///
    /// If `pos` was never pushed, or `len` exceeds the row length the
    /// builder checked against the file size.
    #[inline]
    pub(crate) fn row(&self, pos: u32, len: usize) -> &[u8] {
        let start = self.offset(pos);
        &self.bytes()[start..start + len]
    }

    /// A view for one search: the file slice is resolved once, and a
    /// single-run table is reduced to its base offset and stride, so a
    /// row lookup is arithmetic plus one bounds-checked slice.
    pub(crate) fn view(&self) -> MappedRowsView<'_> {
        let single_run = (self.run_starts.len() == 1)
            .then(|| (self.run_offsets[0] as usize, self.run_strides[0] as usize));
        MappedRowsView {
            bytes: self.bytes(),
            single_run,
            rows: self,
        }
    }

    /// Number of runs.
    #[cfg(test)]
    pub(crate) fn run_count(&self) -> usize {
        self.run_starts.len()
    }

    /// Bytes of the run tables (the only heap this type holds).
    pub(crate) fn heap_size(&self) -> usize {
        self.run_starts.len() * 4 + self.run_offsets.len() * 8 + self.run_strides.len() * 8
    }

    /// Read one byte from every page of the first `count` rows of `len`
    /// bytes, so their pages are faulted in before the first query
    /// (searcher warmup, Issue #677).
    ///
    /// # Returns
    ///
    /// A checksum of the touched bytes, for the caller to keep alive with
    /// `std::hint::black_box`.
    pub(crate) fn touch(&self, count: u32, len: usize) -> u64 {
        let mut acc = 0u64;
        for pos in 0..count {
            let row = self.row(pos, len);
            for b in row.iter().step_by(PAGE) {
                acc = acc.wrapping_add(*b as u64);
            }
            if let Some(last) = row.last() {
                acc = acc.wrapping_add(*last as u64);
            }
        }
        acc
    }
}

/// [`MappedRows`] borrowed for one search (see [`MappedRows::view`]).
#[derive(Clone, Copy)]
pub(crate) struct MappedRowsView<'a> {
    /// The whole mapped file.
    bytes: &'a [u8],
    /// `(offset of row 0, stride)` when all rows form one run.
    single_run: Option<(usize, usize)>,
    rows: &'a MappedRows,
}

impl<'a> MappedRowsView<'a> {
    /// The `len` bytes of the row at position `pos`.
    ///
    /// # Panics
    ///
    /// Like [`MappedRows::row`].
    #[inline]
    pub(crate) fn row(&self, pos: u32, len: usize) -> &'a [u8] {
        let start = match self.single_run {
            Some((base, stride)) => base + pos as usize * stride,
            None => self.rows.offset(pos),
        };
        &self.bytes[start..start + len]
    }

    /// Base address and stride for direct-address software prefetch, when
    /// all rows form one run. The address is only used as a prefetch hint.
    pub(crate) fn base_stride(&self) -> Option<(usize, usize)> {
        self.single_run
            .map(|(base, stride)| (self.bytes.as_ptr() as usize + base, stride))
    }
}

/// Builds [`MappedRows`] while a reader parses a segment's records.
pub(crate) struct MappedRowsBuilder {
    input: Box<dyn StorageInput>,
    file_len: u64,
    /// Bytes each row must have left in the file.
    row_len: u64,
    run_starts: Vec<u32>,
    run_offsets: Vec<u64>,
    run_strides: Vec<u64>,
    /// Offset of the last row pushed.
    last: Option<u64>,
}

impl MappedRowsBuilder {
    /// A builder over `input`, or `None` when `input` cannot expose the
    /// whole file as one slice (the caller then copies the rows instead).
    ///
    /// # Arguments
    ///
    /// * `input` - A fresh clone of the segment's input (position 0); it is
    ///   never moved, so its slice covers the whole file.
    /// * `row_len` - Bytes of one row, checked against the file size on
    ///   every push.
    ///
    /// # Errors
    ///
    /// Forwards a failure to read the input's size.
    pub(crate) fn new(input: Box<dyn StorageInput>, row_len: usize) -> Result<Option<Self>> {
        let file_len = input.size()?;
        let whole_file = input
            .as_slice()
            .is_some_and(|bytes| bytes.len() as u64 == file_len);
        if !whole_file {
            return Ok(None);
        }
        Ok(Some(Self {
            input,
            file_len,
            row_len: row_len as u64,
            run_starts: Vec::new(),
            run_offsets: Vec::new(),
            run_strides: Vec::new(),
            last: None,
        }))
    }

    /// Add the row that starts at the current position of `reader` as
    /// position `pos`, leaving `reader` positioned after the row.
    ///
    /// # Arguments
    ///
    /// * `pos` - The row's position: one more than the previous push's
    ///   (`0` for the first).
    /// * `reader` - The parse stream, positioned at the row.
    ///
    /// # Errors
    ///
    /// [`LaurusError::Index`] if the row runs past the end of the file;
    /// I/O errors from `reader`.
    pub(crate) fn push<R: Seek>(&mut self, pos: u32, reader: &mut R) -> Result<()> {
        let offset = reader.stream_position().map_err(LaurusError::Io)?;
        if offset
            .checked_add(self.row_len)
            .is_none_or(|end| end > self.file_len)
        {
            return Err(LaurusError::index(format!(
                "vector segment corrupt: row at byte {offset} runs past the end of the \
                 file ({} bytes)",
                self.file_len
            )));
        }
        if !self.continues_run(pos, offset) {
            self.run_starts.push(pos);
            self.run_offsets.push(offset);
            self.run_strides.push(0);
        }
        self.last = Some(offset);
        reader
            .seek(SeekFrom::Current(self.row_len as i64))
            .map_err(LaurusError::Io)?;
        Ok(())
    }

    /// Whether the row at `offset` continues the current run, fixing the
    /// run's stride when it is the run's second row.
    fn continues_run(&mut self, pos: u32, offset: u64) -> bool {
        let (Some(&start), Some(last), Some(stride)) = (
            self.run_starts.last(),
            self.last,
            self.run_strides.last_mut(),
        ) else {
            return false;
        };
        // Rows never overlap, so a valid gap is at least one row.
        let Some(gap) = offset.checked_sub(last).filter(|&g| g >= self.row_len) else {
            return false;
        };
        if pos - start == 1 {
            *stride = gap;
            true
        } else {
            gap == *stride
        }
    }

    /// Finish the table, or `None` when no row was pushed.
    pub(crate) fn build(self) -> Option<MappedRows> {
        if self.run_starts.is_empty() {
            return None;
        }
        Some(MappedRows {
            input: self.input,
            run_starts: self.run_starts,
            run_offsets: self.run_offsets,
            run_strides: self.run_strides,
        })
    }
}

/// Collects the per-field `doc_id -> position` maps of a row pool.
#[derive(Default)]
pub(crate) struct FieldPositions {
    /// Positions per field id, in field-dictionary order.
    by_field: Vec<HashMap<u64, u32>>,
}

impl FieldPositions {
    /// Record that `(doc_id, field_id)` lives at `pos`; a later record for
    /// the same key replaces it.
    pub(crate) fn insert(&mut self, field_id: u16, doc_id: u64, pos: u32) {
        let fid = field_id as usize;
        if self.by_field.len() <= fid {
            self.by_field.resize_with(fid + 1, HashMap::new);
        }
        self.by_field[fid].insert(doc_id, pos);
    }

    /// The maps keyed by field name.
    ///
    /// # Arguments
    ///
    /// * `field_dict` - The segment's field dictionary, indexed by the
    ///   field ids passed to [`Self::insert`].
    pub(crate) fn into_index(
        self,
        field_dict: &[Arc<str>],
    ) -> HashMap<String, Arc<HashMap<u64, u32>>> {
        self.by_field
            .into_iter()
            .enumerate()
            .filter(|(_, map)| !map.is_empty())
            .map(|(fid, map)| (field_dict[fid].to_string(), Arc::new(map)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use crate::storage::Storage;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn mapped_input(bytes: &[u8]) -> Box<dyn StorageInput> {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());
        let mut out = storage.create_output("seg").unwrap();
        std::io::Write::write_all(&mut out, bytes).unwrap();
        out.close().unwrap();
        storage.open_input("seg").unwrap()
    }

    /// Push rows of `row_len` bytes at `offsets` and build the table.
    fn build(bytes: &[u8], row_len: usize, offsets: &[u64]) -> MappedRows {
        let mut builder = MappedRowsBuilder::new(mapped_input(bytes), row_len)
            .unwrap()
            .unwrap();
        let mut cursor = Cursor::new(bytes);
        for (pos, &offset) in offsets.iter().enumerate() {
            cursor.set_position(offset);
            builder.push(pos as u32, &mut cursor).unwrap();
            assert_eq!(cursor.position(), offset + row_len as u64);
        }
        builder.build().unwrap()
    }

    fn numbered(len: usize) -> Vec<u8> {
        (0..len).map(|i| i as u8).collect()
    }

    #[test]
    fn equally_spaced_rows_form_one_run() {
        let bytes = numbered(64);
        // A 3-byte prefix before each 4-byte row: stride 7.
        let offsets: Vec<u64> = (0..8).map(|i| 3 + i * 7).collect();
        let rows = build(&bytes, 4, &offsets);
        assert_eq!(rows.run_count(), 1);
        let view = rows.view();
        for (pos, &offset) in offsets.iter().enumerate() {
            let o = offset as usize;
            assert_eq!(rows.row(pos as u32, 4), &bytes[o..o + 4]);
            assert_eq!(view.row(pos as u32, 4), &bytes[o..o + 4]);
        }
        let (base, stride) = rows.view().base_stride().unwrap();
        assert_eq!(stride, 7);
        assert_eq!(base, bytes_base(&rows) + 3);
    }

    fn bytes_base(rows: &MappedRows) -> usize {
        rows.bytes().as_ptr() as usize
    }

    #[test]
    fn a_gap_starts_a_new_run_and_rows_stay_exact() {
        let bytes = numbered(128);
        // IVF-like: two lists of three rows (stride 6), a 4-byte length
        // between them, then a list of one row and a list of two.
        let offsets = [0u64, 6, 12, 22, 28, 34, 44, 54, 60];
        let rows = build(&bytes, 5, &offsets);
        assert!(rows.run_count() > 1);
        let view = rows.view();
        assert!(view.base_stride().is_none());
        for (pos, &offset) in offsets.iter().enumerate() {
            let o = offset as usize;
            assert_eq!(rows.row(pos as u32, 5), &bytes[o..o + 5], "pos {pos}");
            assert_eq!(view.row(pos as u32, 5), &bytes[o..o + 5], "pos {pos}");
        }
    }

    #[test]
    fn varying_gaps_split_runs_without_losing_rows() {
        let bytes = numbered(200);
        // Legacy multi-field records whose names differ in length: the
        // gap alternates between 9 and 12.
        let offsets: Vec<u64> = (0..12u64).map(|i| (i / 2) * 21 + (i % 2) * 9).collect();
        let rows = build(&bytes, 6, &offsets);
        for (pos, &offset) in offsets.iter().enumerate() {
            let o = offset as usize;
            assert_eq!(rows.row(pos as u32, 6), &bytes[o..o + 6], "pos {pos}");
        }
        assert_eq!(rows.heap_size(), rows.run_count() * 20);
    }

    #[test]
    fn single_row_has_one_run_with_zero_stride() {
        let bytes = numbered(16);
        let rows = build(&bytes, 4, &[5]);
        assert_eq!(rows.run_count(), 1);
        assert_eq!(rows.row(0, 4), &bytes[5..9]);
        assert_eq!(rows.view().base_stride().unwrap().1, 0);
    }

    #[test]
    fn rejects_a_row_past_the_end() {
        let bytes = numbered(20);
        let mut builder = MappedRowsBuilder::new(mapped_input(&bytes), 8)
            .unwrap()
            .unwrap();
        let mut cursor = Cursor::new(&bytes[..]);
        cursor.set_position(13);
        let err = builder.push(0, &mut cursor).unwrap_err();
        assert!(err.to_string().contains("runs past the end"), "{err}");
        // A position past EOF, which a mapped input's seek accepts.
        cursor.set_position(40);
        assert!(builder.push(0, &mut cursor).is_err());
    }

    #[test]
    fn no_rows_builds_nothing() {
        let builder = MappedRowsBuilder::new(mapped_input(&numbered(8)), 4)
            .unwrap()
            .unwrap();
        assert!(builder.build().is_none());
    }

    #[test]
    fn touch_reads_the_first_and_last_byte_of_small_rows() {
        let bytes = numbered(32);
        let rows = build(&bytes, 4, &[0, 8, 16]);
        // Rows 0..4, 8..12, 16..20: first + last bytes.
        let expected: u64 = [0u64, 3, 8, 11, 16, 19].iter().sum();
        assert_eq!(rows.touch(3, 4), expected);
    }

    /// An input that refuses to expose its bytes.
    #[derive(Debug)]
    struct NoSlice(Cursor<Vec<u8>>);

    impl std::io::Read for NoSlice {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.0.read(buf)
        }
    }

    impl Seek for NoSlice {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.0.seek(pos)
        }
    }

    impl StorageInput for NoSlice {
        fn size(&self) -> Result<u64> {
            Ok(self.0.get_ref().len() as u64)
        }

        fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
            Ok(Box::new(NoSlice(Cursor::new(self.0.get_ref().clone()))))
        }

        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn an_input_without_a_slice_is_not_mapped() {
        let input = Box::new(NoSlice(Cursor::new(numbered(16))));
        assert!(MappedRowsBuilder::new(input, 4).unwrap().is_none());
    }

    #[test]
    fn field_positions_key_maps_by_name_and_skip_empty_fields() {
        let mut positions = FieldPositions::default();
        positions.insert(0, 10, 0);
        positions.insert(2, 10, 1);
        positions.insert(0, 20, 2);
        positions.insert(0, 10, 3); // replaces the first record
        let dict: Vec<Arc<str>> = vec![Arc::from("a"), Arc::from("b"), Arc::from("c")];
        let index = positions.into_index(&dict);
        assert_eq!(index.len(), 2);
        assert_eq!(index["a"].get(&10), Some(&3));
        assert_eq!(index["a"].get(&20), Some(&2));
        assert_eq!(index["c"].get(&10), Some(&1));
        assert!(!index.contains_key("b"));
    }
}
