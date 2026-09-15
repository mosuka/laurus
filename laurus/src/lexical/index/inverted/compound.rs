//! Compound segment file (#554) — Lucene-CFS-style container fusing the
//! per-segment data files into one `segment_<N>.cfs`.
//!
//! # Format
//!
//! ```text
//! [parts, concatenated — each part keeps its own internal framing]
//! [table: varint part_count, repeat { varint suffix_len, suffix bytes,
//!         u64 offset, u64 len }]
//! [trailer, fixed 20 bytes: u64 table_offset, u32 table_crc32,
//!         u32 version, u32 magic "CFND"]
//! ```
//!
//! All integers little-endian. Suffixes are opaque strings — everything
//! after `"{segment_id}."` (e.g. `"post"`, `"score.bkd"`). The `.delmap`
//! is deliberately NOT part of the container: it is the only segment file
//! rewritten after sealing (deferred deletion flushes, #875).
//!
//! # Roles
//!
//! - [`CompoundSegmentWriter`] writes parts sequentially through
//!   [`PartOutput`] wrappers over one shared output — one `create`, one
//!   fsync, one `close` per segment instead of one per part.
//! - [`CompoundSegmentStorage`] is a per-segment [`Storage`] facade
//!   (shaped like `PrefixedStorage`): part names resolve to windowed
//!   views over the container, everything else passes through to the
//!   inner storage — crucially including `file_exists` misses, because
//!   the `.delmap` probe treats a missing file as "no deletions" and a
//!   table-only answer would silently resurrect deleted documents.

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;
use std::sync::Mutex;

use crate::storage::{FileMetadata, LoadingMode, Storage, StorageInput, StorageOutput};
use crate::{LaurusError, Result};

/// File suffix of the container: `{segment_id}.cfs`.
pub(crate) const COMPOUND_SUFFIX: &str = "cfs";

/// Trailer magic, "CFND".
const COMPOUND_MAGIC: u32 = 0x4346_4E44;

/// Container format version written by this build.
const COMPOUND_VERSION: u32 = 1;

/// Fixed trailer size: table_offset (8) + table_crc (4) + version (4) +
/// magic (4).
const TRAILER_LEN: u64 = 20;

/// The default for the `use_compound` layout knobs (#554).
///
/// `true` — new flushes write compound `.cfs` containers — unless
/// `LAURUS_NO_COMPOUND=1` is set, a one-release escape hatch mirroring
/// `LAURUS_NO_MMAP`. Readers detect the layout per segment either way,
/// so flipping the knob never affects existing segments.
pub(crate) fn default_use_compound() -> bool {
    !matches!(std::env::var("LAURUS_NO_COMPOUND").as_deref(), Ok("1"))
}

/// The container file name for a segment.
pub(crate) fn container_name(segment_id: &str) -> String {
    format!("{segment_id}.{COMPOUND_SUFFIX}")
}

/// One table entry: a part's suffix and its window in the container.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PartEntry {
    /// Everything after `"{segment_id}."` in the loose file name.
    pub(crate) suffix: String,
    /// Byte offset of the part's first byte in the container.
    pub(crate) offset: u64,
    /// Part length in bytes.
    pub(crate) len: u64,
}

// ---------------------------------------------------------------------------
// Write side
// ---------------------------------------------------------------------------

/// Shared state behind every [`PartOutput`] of one container write.
#[derive(Debug)]
struct ContainerWriteState {
    /// The single real output. `None` only after [`CompoundSegmentWriter::finish`].
    out: Option<Box<dyn StorageOutput>>,
    /// Base offset of the part currently being written.
    part_base: u64,
    /// High-water mark of the current part, as an ABSOLUTE container
    /// position. Part lengths derive from this logical mark, never from
    /// storage sizes: `FileOutput` buffers 64 KiB, and `MemoryStorage`
    /// reports size 0 until the final publish.
    part_max: u64,
}

impl ContainerWriteState {
    fn out(&mut self) -> Result<&mut Box<dyn StorageOutput>> {
        self.out
            .as_mut()
            .ok_or_else(|| LaurusError::storage("compound container already finished"))
    }
}

/// Writes one compound container, handing out a [`PartOutput`] per part.
#[derive(Debug)]
pub(crate) struct CompoundSegmentWriter {
    state: Arc<Mutex<ContainerWriteState>>,
    entries: Vec<PartEntry>,
    container: String,
}

impl CompoundSegmentWriter {
    /// Open the container output.
    pub(crate) fn create(storage: &dyn Storage, segment_id: &str) -> Result<Self> {
        let container = container_name(segment_id);
        let out = storage.create_output(&container)?;
        Ok(Self {
            state: Arc::new(Mutex::new(ContainerWriteState {
                out: Some(out),
                part_base: 0,
                part_max: 0,
            })),
            entries: Vec::new(),
            container,
        })
    }

    /// Begin the next part and return the output to write it through.
    ///
    /// Parts are strictly sequential: the previous part's extent seeds the
    /// next base, and the shared handle is repositioned there (a part like
    /// the BKD tree ends on a backfill seek, not necessarily at its end).
    pub(crate) fn begin_part(&mut self, suffix: &str) -> Result<PartOutput> {
        let mut state = self.state.lock().unwrap();
        let base = state.part_max;
        let out = state.out()?;
        out.seek(SeekFrom::Start(base))?;
        state.part_base = base;
        state.part_max = base;
        drop(state);
        self.entries.push(PartEntry {
            suffix: suffix.to_string(),
            offset: base,
            len: 0, // filled by end_part
        });
        Ok(PartOutput {
            state: Arc::clone(&self.state),
        })
    }

    /// The container's file name.
    pub(crate) fn container_name_owned(&self) -> String {
        self.container.clone()
    }

    /// Seal the part begun by the last [`Self::begin_part`], recording its
    /// length from the logical high-water mark.
    pub(crate) fn end_part(&mut self) -> Result<()> {
        let state = self.state.lock().unwrap();
        let entry = self
            .entries
            .last_mut()
            .ok_or_else(|| LaurusError::storage("end_part without begin_part"))?;
        entry.len = state.part_max - entry.offset;
        Ok(())
    }

    /// Write the table + trailer, fsync once, close once.
    pub(crate) fn finish(self) -> Result<Vec<PartEntry>> {
        let mut state = self.state.lock().unwrap();
        let table_offset = state.part_max;
        let mut table = Vec::new();
        write_varint(&mut table, self.entries.len() as u64);
        for entry in &self.entries {
            write_varint(&mut table, entry.suffix.len() as u64);
            table.extend_from_slice(entry.suffix.as_bytes());
            table.extend_from_slice(&entry.offset.to_le_bytes());
            table.extend_from_slice(&entry.len.to_le_bytes());
        }
        let table_crc = crc32fast::hash(&table);

        let out = state.out()?;
        out.seek(SeekFrom::Start(table_offset))?;
        out.write_all(&table)?;
        out.write_all(&table_offset.to_le_bytes())?;
        out.write_all(&table_crc.to_le_bytes())?;
        out.write_all(&COMPOUND_VERSION.to_le_bytes())?;
        out.write_all(&COMPOUND_MAGIC.to_le_bytes())?;
        out.flush_and_sync()?;
        let mut out = state.out.take().expect("checked by out() above");
        out.close()?;
        drop(state);
        Ok(self.entries)
    }
}

/// A [`StorageOutput`] view over the shared container output, offset by
/// the current part's base.
///
/// Seeks are translated in BOTH directions — arguments AND return values.
/// The return-value half is load-bearing: `StructWriter::seek` adopts the
/// returned position as its internal counter, and `BKDWriter` persists
/// stream positions into the part (header backfill via `Start(0)`,
/// `index_start_offset` from `stream_position()`), which its reader then
/// interprets as part-relative offsets.
///
/// `close()` and `flush_and_sync()` are deliberate no-ops: part writers
/// end in `StructWriter::close`, which would otherwise fsync per part and
/// poison the shared handle for the parts after it. The single real fsync
/// + close happens in [`CompoundSegmentWriter::finish`].
#[derive(Debug)]
pub(crate) struct PartOutput {
    state: Arc<Mutex<ContainerWriteState>>,
}

impl Write for PartOutput {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut state = self.state.lock().unwrap();
        let base = state.part_base;
        let out = state
            .out
            .as_mut()
            .ok_or_else(|| std::io::Error::other("compound container already finished"))?;
        let written = out.write(buf)?;
        let pos = out
            .position()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        debug_assert!(pos >= base);
        state.part_max = state.part_max.max(pos);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // Plain flush is forwarded (it does not fsync); harmless and it
        // keeps `Write` adapters that flush explicitly working.
        let mut state = self.state.lock().unwrap();
        match state.out.as_mut() {
            Some(out) => out.flush(),
            None => Ok(()),
        }
    }
}

impl Seek for PartOutput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let mut state = self.state.lock().unwrap();
        let base = state.part_base;
        let part_end = state.part_max;
        let out = state
            .out
            .as_mut()
            .ok_or_else(|| std::io::Error::other("compound container already finished"))?;
        let translated = match pos {
            SeekFrom::Start(offset) => SeekFrom::Start(base + offset),
            // Relative to the part's extent so far, NOT the container end:
            // during a sequential write both coincide, which is exactly why
            // this must be translated deliberately (tests cannot tell the
            // difference) — a future non-tail part would corrupt silently.
            SeekFrom::End(delta) => SeekFrom::Start((part_end as i64 + delta) as u64),
            SeekFrom::Current(delta) => SeekFrom::Current(delta),
        };
        let absolute = out.seek(translated)?;
        debug_assert!(absolute >= base);
        Ok(absolute - base)
    }
}

impl StorageOutput for PartOutput {
    fn flush_and_sync(&mut self) -> Result<()> {
        // No-op: the container is synced exactly once, in `finish`.
        Ok(())
    }

    fn position(&self) -> Result<u64> {
        let state = self.state.lock().unwrap();
        let base = state.part_base;
        let out = state
            .out
            .as_ref()
            .ok_or_else(|| LaurusError::storage("compound container already finished"))?;
        Ok(out.position()? - base)
    }

    fn close(&mut self) -> Result<()> {
        // No-op: the shared handle stays open for the remaining parts.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Read side
// ---------------------------------------------------------------------------

/// How part reads obtain container bytes.
#[derive(Debug, Clone)]
enum ContainerHandle {
    /// Open the container per part read. Cheap where `open_input` is —
    /// mmap-backed FileStorage (an `Arc<Mmap>` clone from the cache) and
    /// buffered non-mmap files (a fresh descriptor).
    PerOpen,
    /// One buffered copy shared by every window. Used for eager in-memory
    /// backends, whose `open_input` deep-copies the whole file per call —
    /// the postings path re-opens per query, so per-part opens would copy
    /// the container per search.
    Buffered(Arc<Vec<u8>>),
}

/// Per-segment [`Storage`] facade resolving part names to windowed views
/// over the container; every miss passes through to the inner storage.
#[derive(Debug)]
pub(crate) struct CompoundSegmentStorage {
    inner: Arc<dyn Storage>,
    segment_prefix: String,
    container: String,
    parts: Vec<PartEntry>,
    handle: ContainerHandle,
}

impl CompoundSegmentStorage {
    /// Wrap `inner` for `segment_id` if a container exists.
    ///
    /// # Returns
    ///
    /// `Ok(Some(facade))` when `{segment_id}.cfs` exists and parses;
    /// `Ok(None)` when it does not (a loose, pre-#554 segment).
    ///
    /// # Errors
    ///
    /// Returns an error when the container exists but its trailer or table
    /// is invalid — a referenced-but-torn container must surface, not
    /// degrade to "no parts".
    pub(crate) fn try_open(inner: Arc<dyn Storage>, segment_id: &str) -> Result<Option<Arc<Self>>> {
        let container = container_name(segment_id);
        if !inner.file_exists(&container) {
            return Ok(None);
        }
        let mut input = inner.open_input(&container)?;
        let size = input.size()?;
        if size < TRAILER_LEN {
            return Err(LaurusError::storage(format!(
                "{container}: too short for a compound trailer ({size} bytes)"
            )));
        }
        input.seek(SeekFrom::Start(size - TRAILER_LEN))?;
        let mut trailer = [0u8; TRAILER_LEN as usize];
        input.read_exact(&mut trailer)?;
        let table_offset = u64::from_le_bytes(trailer[0..8].try_into().expect("fixed len"));
        let table_crc = u32::from_le_bytes(trailer[8..12].try_into().expect("fixed len"));
        let version = u32::from_le_bytes(trailer[12..16].try_into().expect("fixed len"));
        let magic = u32::from_le_bytes(trailer[16..20].try_into().expect("fixed len"));
        if magic != COMPOUND_MAGIC {
            return Err(LaurusError::storage(format!(
                "{container}: bad compound magic {magic:#010x}"
            )));
        }
        if version != COMPOUND_VERSION {
            return Err(LaurusError::storage(format!(
                "{container}: unsupported compound version {version}"
            )));
        }
        if table_offset > size - TRAILER_LEN {
            return Err(LaurusError::storage(format!(
                "{container}: table offset {table_offset} out of bounds"
            )));
        }
        let table_len = (size - TRAILER_LEN - table_offset) as usize;
        input.seek(SeekFrom::Start(table_offset))?;
        let mut table = vec![0u8; table_len];
        input.read_exact(&mut table)?;
        if crc32fast::hash(&table) != table_crc {
            return Err(LaurusError::storage(format!(
                "{container}: compound table checksum mismatch — the file is corrupted"
            )));
        }
        let parts = parse_table(&table, table_offset, &container)?;

        // Handle strategy (see `ContainerHandle`). `loading_mode` is what
        // separates paging-friendly backends from eager ones; the slice
        // probe separates in-memory (cheap to buffer once, expensive to
        // re-open) from buffered files (the reverse).
        let handle = if inner.loading_mode() == LoadingMode::Eager && input.as_slice().is_some() {
            input.seek(SeekFrom::Start(0))?;
            let mut bytes = Vec::with_capacity(size as usize);
            input.read_to_end(&mut bytes)?;
            ContainerHandle::Buffered(Arc::new(bytes))
        } else {
            ContainerHandle::PerOpen
        };

        Ok(Some(Arc::new(Self {
            inner,
            segment_prefix: format!("{segment_id}."),
            container,
            parts,
            handle,
        })))
    }

    /// The table entry for a full loose-style file name, if it is a part.
    fn entry_for(&self, name: &str) -> Option<&PartEntry> {
        let suffix = name.strip_prefix(&self.segment_prefix)?;
        self.parts.iter().find(|p| p.suffix == suffix)
    }

    /// The BKD field names recorded in the container (#554 BLOCKER-1: the
    /// merge engine used to enumerate `*.bkd` files by listing raw
    /// storage, which finds nothing once the parts live in a container —
    /// silently dropping every numeric/geo point at the first merge).
    pub(crate) fn bkd_field_names(&self) -> Vec<String> {
        self.parts
            .iter()
            .filter_map(|p| p.suffix.strip_suffix(".bkd"))
            .map(str::to_string)
            .collect()
    }

    fn open_window(&self, entry: &PartEntry) -> Result<Box<dyn StorageInput>> {
        match &self.handle {
            ContainerHandle::Buffered(bytes) => Ok(Box::new(PartInput {
                backing: WindowBacking::Buffered(Arc::clone(bytes)),
                base: entry.offset,
                len: entry.len,
                pos: 0,
                handle_pos: entry.offset,
            })),
            ContainerHandle::PerOpen => {
                let mut input = self.inner.open_input(&self.container)?;
                // Establish the alignment invariant `as_slice` relies on:
                // the handle sits at the window's cursor from the start.
                input.seek(SeekFrom::Start(entry.offset))?;
                Ok(Box::new(PartInput {
                    backing: WindowBacking::Handle(input),
                    base: entry.offset,
                    len: entry.len,
                    pos: 0,
                    handle_pos: entry.offset,
                }))
            }
        }
    }
}

impl Storage for CompoundSegmentStorage {
    fn loading_mode(&self) -> LoadingMode {
        self.inner.loading_mode()
    }

    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        match self.entry_for(name) {
            Some(entry) => self.open_window(&entry.clone()),
            None => self.inner.open_input(name),
        }
    }

    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output(name)
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output_append(name)
    }

    fn file_exists(&self, name: &str) -> bool {
        self.entry_for(name).is_some() || self.inner.file_exists(name)
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(name)
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.inner.rename_file(old_name, new_name)
    }

    fn list_files(&self) -> Result<Vec<String>> {
        // Passthrough: the facade serves named part reads; enumeration of
        // parts goes through the typed API (`bkd_field_names`), not
        // through directory listings.
        self.inner.list_files()
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        match self.entry_for(name) {
            Some(entry) => Ok(entry.len),
            None => self.inner.file_size(name),
        }
    }

    fn metadata(&self, name: &str) -> Result<FileMetadata> {
        match self.entry_for(name) {
            Some(entry) => {
                let container_meta = self.inner.metadata(&self.container)?;
                Ok(FileMetadata {
                    size: entry.len,
                    ..container_meta
                })
            }
            None => self.inner.metadata(name),
        }
    }

    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        self.inner.create_temp_output(prefix)
    }

    fn sync(&self) -> Result<()> {
        self.inner.sync()
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// The bytes behind a window.
enum WindowBacking {
    Handle(Box<dyn StorageInput>),
    Buffered(Arc<Vec<u8>>),
}

impl std::fmt::Debug for WindowBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Handle(_) => f.write_str("WindowBacking::Handle"),
            Self::Buffered(b) => write!(f, "WindowBacking::Buffered({} bytes)", b.len()),
        }
    }
}

/// A windowed [`StorageInput`] over `[base, base + len)` of the container.
///
/// Everything is clamped to the window: `size()` reports the window
/// length (`StructReader`'s framing depends on it), `Read` cannot run
/// past the window end even when the underlying bytes continue (a corrupt
/// part boundary must fail, not silently consume the next part), and
/// `as_slice` is clamped on BOTH ends — the #504 zero-copy posting
/// decode only checks that the slice is long *enough*, so an unclamped
/// tail would hand it the next part's bytes.
///
/// A `Handle` window keeps the container handle parked at `base + pos`
/// (#1046). `as_slice` takes `&self` and so cannot seek, and the inner
/// slice starts at the inner cursor — without the invariant the window
/// could only decline, which routed every posting-block decode through
/// the heap-allocating fallback once the compound layout became the
/// default. The seek is issued only when the handle has drifted: a
/// buffered file input discards its whole read buffer on every `Seek`,
/// so reseeking before each sequential read would thrash it.
#[derive(Debug)]
struct PartInput {
    backing: WindowBacking,
    base: u64,
    len: u64,
    /// Cursor within the window.
    pos: u64,
    /// Absolute container position the `Handle` currently sits at.
    /// Unused for `Buffered` backing, which needs no cursor.
    handle_pos: u64,
}

impl PartInput {
    /// Park the container handle at `base + pos` if it has drifted.
    fn align_handle(&mut self) -> std::io::Result<()> {
        let target = self.base + self.pos;
        if let WindowBacking::Handle(input) = &mut self.backing
            && self.handle_pos != target
        {
            input.seek(SeekFrom::Start(target))?;
            self.handle_pos = target;
        }
        Ok(())
    }
}

impl Read for PartInput {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let remaining = self.len.saturating_sub(self.pos);
        if remaining == 0 {
            return Ok(0);
        }
        let want = buf.len().min(remaining as usize);
        self.align_handle()?;
        let read = match &mut self.backing {
            WindowBacking::Buffered(bytes) => {
                let start = (self.base + self.pos) as usize;
                let end = start + want;
                buf[..want].copy_from_slice(&bytes[start..end]);
                want
            }
            WindowBacking::Handle(input) => input.read(&mut buf[..want])?,
        };
        self.pos += read as u64;
        self.handle_pos += read as u64;
        Ok(read)
    }
}

impl Seek for PartInput {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let target = match pos {
            SeekFrom::Start(offset) => offset as i64,
            SeekFrom::End(delta) => self.len as i64 + delta,
            SeekFrom::Current(delta) => self.pos as i64 + delta,
        };
        if target < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before window start",
            ));
        }
        self.pos = target as u64;
        // Keep the handle parked at the new cursor: `as_slice` cannot seek
        // and reads from wherever the handle sits (#1046).
        self.align_handle()?;
        Ok(self.pos)
    }
}

impl StorageInput for PartInput {
    fn size(&self) -> Result<u64> {
        Ok(self.len)
    }

    fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
        let backing = match &self.backing {
            WindowBacking::Buffered(bytes) => WindowBacking::Buffered(Arc::clone(bytes)),
            WindowBacking::Handle(input) => {
                // A cloned handle restarts at the container base, so park
                // it at this part's base to hold the alignment invariant.
                let mut cloned = input.clone_input()?;
                cloned.seek(SeekFrom::Start(self.base))?;
                WindowBacking::Handle(cloned)
            }
        };
        Ok(Box::new(PartInput {
            backing,
            base: self.base,
            len: self.len,
            pos: 0,
            handle_pos: self.base,
        }))
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn as_slice(&self) -> Option<&[u8]> {
        let start = (self.base + self.pos) as usize;
        let end = (self.base + self.len) as usize;
        match &self.backing {
            WindowBacking::Buffered(bytes) => bytes.get(start..end),
            WindowBacking::Handle(input) => {
                // The handle is parked at `base + pos` (see `align_handle`),
                // so the inner slice already starts at this window's cursor;
                // clamp its tail to the window end. Without the clamp the
                // #504 decode — which only checks the slice is long *enough*
                // — would run into the next part's bytes.
                if self.handle_pos != self.base + self.pos {
                    return None;
                }
                let remaining = self.len.saturating_sub(self.pos) as usize;
                input.as_slice().and_then(|slice| slice.get(..remaining))
            }
        }
    }
}

/// Parse the table bytes.
fn parse_table(table: &[u8], table_offset: u64, container: &str) -> Result<Vec<PartEntry>> {
    let mut cursor = 0usize;
    let count = read_varint(table, &mut cursor, container)?;
    let mut parts = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let suffix_len = read_varint(table, &mut cursor, container)? as usize;
        let suffix_end = cursor
            .checked_add(suffix_len)
            .filter(|&end| end <= table.len())
            .ok_or_else(|| LaurusError::storage(format!("{container}: truncated table")))?;
        let suffix = std::str::from_utf8(&table[cursor..suffix_end])
            .map_err(|_| LaurusError::storage(format!("{container}: non-UTF-8 part suffix")))?
            .to_string();
        cursor = suffix_end;
        if cursor + 16 > table.len() {
            return Err(LaurusError::storage(format!(
                "{container}: truncated table entry"
            )));
        }
        let offset = u64::from_le_bytes(table[cursor..cursor + 8].try_into().expect("checked"));
        let len = u64::from_le_bytes(table[cursor + 8..cursor + 16].try_into().expect("checked"));
        cursor += 16;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| LaurusError::storage(format!("{container}: part range overflow")))?;
        if end > table_offset {
            return Err(LaurusError::storage(format!(
                "{container}: part {suffix} overlaps the table"
            )));
        }
        parts.push(PartEntry {
            suffix,
            offset,
            len,
        });
    }
    Ok(parts)
}

fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
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

fn read_varint(bytes: &[u8], cursor: &mut usize, container: &str) -> Result<u64> {
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
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn memory() -> Arc<dyn Storage> {
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
    }

    /// Sequential writes cannot distinguish translation from passthrough,
    /// so the wrapper's seek semantics are pinned directly, at a non-zero
    /// base.
    #[test]
    fn part_output_translates_seeks_and_positions() {
        let storage = memory();
        let mut writer = CompoundSegmentWriter::create(storage.as_ref(), "segment_000000").unwrap();

        // Part 1 pushes part 2's base past zero.
        let mut p1 = writer.begin_part("first").unwrap();
        p1.write_all(b"0123456789").unwrap();
        writer.end_part().unwrap();

        let mut p2 = writer.begin_part("second").unwrap();
        assert_eq!(p2.position().unwrap(), 0, "part positions start at 0");

        // The BKD pattern: reserve a header, write payload, backfill.
        p2.write_all(&[0u8; 4]).unwrap(); // header placeholder
        p2.write_all(b"PAYLOAD").unwrap();
        let end = p2.seek(SeekFrom::End(0)).unwrap();
        assert_eq!(
            end, 11,
            "End(0) must be the part extent, not the container's"
        );
        let back = p2.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(back, 0, "seek return values must be part-relative");
        p2.write_all(b"HDR!").unwrap();
        let sp = p2.stream_position().unwrap();
        assert_eq!(sp, 4, "stream_position must be part-relative");
        p2.seek(SeekFrom::End(0)).unwrap();
        writer.end_part().unwrap();

        let entries = writer.finish().unwrap();
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[0].len, 10);
        assert_eq!(entries[1].offset, 10);
        assert_eq!(entries[1].len, 11);

        // Read back through the facade: the backfilled header is at the
        // part's byte 0, exactly as a part-relative reader expects.
        let facade = CompoundSegmentStorage::try_open(storage, "segment_000000")
            .unwrap()
            .expect("container must be detected");
        let mut input = facade.open_input("segment_000000.second").unwrap();
        let mut bytes = Vec::new();
        input.read_to_end(&mut bytes).unwrap();
        assert_eq!(&bytes, b"HDR!PAYLOAD");
    }

    /// Reads, seeks, `size()` and `as_slice` are all clamped to the window.
    #[test]
    fn part_input_clamps_to_the_window() {
        let storage = memory();
        let mut writer = CompoundSegmentWriter::create(storage.as_ref(), "segment_000001").unwrap();
        let mut p = writer.begin_part("a").unwrap();
        p.write_all(b"AAAA").unwrap();
        writer.end_part().unwrap();
        let mut p = writer.begin_part("b").unwrap();
        p.write_all(b"BBBB").unwrap();
        writer.end_part().unwrap();
        writer.finish().unwrap();

        let facade = CompoundSegmentStorage::try_open(storage, "segment_000001")
            .unwrap()
            .unwrap();
        let mut input = facade.open_input("segment_000001.a").unwrap();
        assert_eq!(input.size().unwrap(), 4);

        // Read cannot cross into part b even with a bigger buffer.
        let mut buf = [0u8; 16];
        let n = input.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"AAAA", "read must stop at the window end");
        assert_eq!(input.read(&mut buf).unwrap(), 0, "EOF at the window end");

        // as_slice starts at the cursor and clamps at the window end.
        // Asserted unconditionally: an `if let Some(..)` here passes
        // vacuously the moment the fast path stops being served (#1046).
        input.seek(SeekFrom::Start(1)).unwrap();
        assert_eq!(
            input.as_slice(),
            Some(&b"AAA"[..]),
            "as_slice must clamp both ends"
        );

        // Seek relative to the window end.
        let pos = input.seek(SeekFrom::End(-2)).unwrap();
        assert_eq!(pos, 2);
        let mut two = [0u8; 2];
        input.read_exact(&mut two).unwrap();
        assert_eq!(&two, b"AA");
    }

    /// A storage that reports `Lazy` so `try_open` picks
    /// [`ContainerHandle::PerOpen`], reproducing the mmap-backed
    /// `FileStorage` configuration while staying deterministic. Its inner
    /// inputs still lend slices, exactly as `MmapInput` does.
    #[derive(Debug)]
    struct LazyStorage {
        inner: Arc<dyn Storage>,
    }

    impl Storage for LazyStorage {
        fn loading_mode(&self) -> LoadingMode {
            LoadingMode::Lazy
        }
        fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
            self.inner.create_output(name)
        }
        fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
            self.inner.create_output_append(name)
        }
        fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
            self.inner.open_input(name)
        }
        fn file_exists(&self, name: &str) -> bool {
            self.inner.file_exists(name)
        }
        fn delete_file(&self, name: &str) -> Result<()> {
            self.inner.delete_file(name)
        }
        fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
            self.inner.rename_file(old_name, new_name)
        }
        fn list_files(&self) -> Result<Vec<String>> {
            self.inner.list_files()
        }
        fn file_size(&self, name: &str) -> Result<u64> {
            self.inner.file_size(name)
        }
        fn sync(&self) -> Result<()> {
            self.inner.sync()
        }
        fn metadata(&self, name: &str) -> Result<FileMetadata> {
            self.inner.metadata(name)
        }
        fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
            self.inner.create_temp_output(prefix)
        }
        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    /// #1046 — a paging-backed container must still lend the #504
    /// zero-copy slice.
    ///
    /// `ContainerHandle::PerOpen` (what every mmap-backed `FileStorage`
    /// gets) used to decline `as_slice` outright, so making the compound
    /// layout the default silently routed every posting-block decode
    /// through the heap-allocating fallback in
    /// [`crate::storage::structured::StructReader::read_raw_with`].
    #[test]
    fn a_paging_backed_window_still_lends_a_zero_copy_slice() {
        let backing = memory();
        let storage: Arc<dyn Storage> = Arc::new(LazyStorage {
            inner: Arc::clone(&backing),
        });
        let mut writer = CompoundSegmentWriter::create(storage.as_ref(), "segment_000009").unwrap();
        let mut p = writer.begin_part("a").unwrap();
        p.write_all(b"AAAA").unwrap();
        writer.end_part().unwrap();
        let mut p = writer.begin_part("b").unwrap();
        p.write_all(b"BBBB").unwrap();
        writer.end_part().unwrap();
        writer.finish().unwrap();

        let facade = CompoundSegmentStorage::try_open(storage, "segment_000009")
            .unwrap()
            .expect("container must be detected");
        assert!(
            matches!(facade.handle, ContainerHandle::PerOpen),
            "precondition: a Lazy inner must select the per-open handle"
        );

        let mut input = facade.open_input("segment_000009.a").unwrap();
        assert_eq!(
            input.as_slice(),
            Some(&b"AAAA"[..]),
            "a fresh window must lend the whole part, not decline"
        );

        // The slice tracks the window cursor and stops at the window end —
        // an unclamped tail would feed the decoder part b's bytes.
        input.seek(SeekFrom::Start(1)).unwrap();
        assert_eq!(
            input.as_slice(),
            Some(&b"AAA"[..]),
            "the slice must start at the cursor and clamp at the window end"
        );

        // A window reached through `clone_input` starts over at its own
        // part base, not at the container base. Probed on part `b`, whose
        // base is non-zero — on part `a` (base 0) the two coincide and the
        // assertion would pass without exercising anything.
        let b = facade.open_input("segment_000009.b").unwrap();
        assert_eq!(
            b.as_slice(),
            Some(&b"BBBB"[..]),
            "a window at a non-zero base must lend its own part"
        );
        let cloned = b.clone_input().unwrap();
        assert_eq!(
            cloned.as_slice(),
            Some(&b"BBBB"[..]),
            "a cloned window must lend its own part from position 0"
        );

        // Reads still work after the slice hand-out.
        input.seek(SeekFrom::Start(0)).unwrap();
        let mut buf = Vec::new();
        input.read_to_end(&mut buf).unwrap();
        assert_eq!(&buf, b"AAAA");
    }

    /// The facade is table-then-passthrough: part names resolve virtually,
    /// everything else (the `.delmap` probe above all) reaches the inner
    /// storage.
    #[test]
    fn facade_is_table_then_passthrough() {
        let storage = memory();
        let mut writer = CompoundSegmentWriter::create(storage.as_ref(), "segment_000002").unwrap();
        let mut p = writer.begin_part("post").unwrap();
        p.write_all(b"P").unwrap();
        writer.end_part().unwrap();
        writer.finish().unwrap();

        // A loose sibling file next to the container (the `.delmap` shape).
        {
            let mut out = storage.create_output("segment_000002.delmap").unwrap();
            out.write_all(b"D").unwrap();
            out.close().unwrap();
        }

        let facade = CompoundSegmentStorage::try_open(storage, "segment_000002")
            .unwrap()
            .unwrap();
        assert!(facade.file_exists("segment_000002.post"), "table hit");
        assert!(
            facade.file_exists("segment_000002.delmap"),
            "passthrough hit — a table-only answer would resurrect deletions"
        );
        assert!(!facade.file_exists("segment_000002.norms"), "true miss");
        assert_eq!(facade.file_size("segment_000002.post").unwrap(), 1);
        let mut input = facade.open_input("segment_000002.delmap").unwrap();
        let mut bytes = Vec::new();
        input.read_to_end(&mut bytes).unwrap();
        assert_eq!(&bytes, b"D");
    }

    /// A torn container must refuse loudly, never degrade to "no parts".
    #[test]
    fn torn_container_is_refused() {
        let storage = memory();
        let mut writer = CompoundSegmentWriter::create(storage.as_ref(), "segment_000003").unwrap();
        let mut p = writer.begin_part("post").unwrap();
        p.write_all(b"PPPP").unwrap();
        writer.end_part().unwrap();
        writer.finish().unwrap();

        // Truncate the trailer.
        let mut bytes = {
            let mut input = storage.open_input("segment_000003.cfs").unwrap();
            let mut b = Vec::new();
            input.read_to_end(&mut b).unwrap();
            b
        };
        bytes.truncate(bytes.len() - 6);
        let mut out = storage.create_output("segment_000003.cfs").unwrap();
        out.write_all(&bytes).unwrap();
        out.close().unwrap();

        let err = CompoundSegmentStorage::try_open(storage, "segment_000003");
        assert!(err.is_err(), "a torn container must surface as an error");
    }

    /// Absent container → None (a loose, pre-#554 segment).
    #[test]
    fn loose_segment_is_not_wrapped() {
        let storage = memory();
        assert!(
            CompoundSegmentStorage::try_open(storage, "segment_000004")
                .unwrap()
                .is_none()
        );
    }

    /// BKD field names come from the table.
    #[test]
    fn bkd_field_names_from_table() {
        let storage = memory();
        let mut writer = CompoundSegmentWriter::create(storage.as_ref(), "segment_000005").unwrap();
        for suffix in ["post", "rank.bkd", "geo.field.bkd"] {
            let mut p = writer.begin_part(suffix).unwrap();
            p.write_all(b"x").unwrap();
            writer.end_part().unwrap();
        }
        writer.finish().unwrap();

        let facade = CompoundSegmentStorage::try_open(storage, "segment_000005")
            .unwrap()
            .unwrap();
        let mut fields = facade.bkd_field_names();
        fields.sort();
        assert_eq!(fields, vec!["geo.field".to_string(), "rank".to_string()]);
    }
}
