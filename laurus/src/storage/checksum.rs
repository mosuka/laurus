//! CRC-32 streaming adapters for integrity-checked file formats (Issue #786).
//!
//! [`CrcWriter`] and [`CrcReader`] are thin pass-through wrappers that maintain
//! a running CRC-32 ([`crc32fast`]) over the bytes written or read, plus a byte
//! counter. They let a format that writes/reads its payload directly (e.g. the
//! HNSW `.hnsw` segment) accumulate a checksum *during* its existing I/O — so a
//! CRC footer can be appended on write and verified on read with no extra pass
//! over the data.
//!
//! Unlike [`crate::storage::structured::StructWriter`], these add no framing:
//! the bytes on disk are exactly the bytes written, and the checksum is the
//! caller's to place (typically as a trailing footer).

use std::io::{Read, Result as IoResult, Seek, SeekFrom, Write};

use crate::error::Result;
use crate::storage::StorageInput;

/// A [`Write`] wrapper that accumulates a CRC-32 over everything written.
///
/// All writes pass through to the inner writer unchanged; call
/// [`CrcWriter::checksum`] after writing the payload to obtain the CRC to store
/// (e.g. in a footer), and [`CrcWriter::bytes_written`] for the payload length.
pub struct CrcWriter<W: Write> {
    inner: W,
    hasher: crc32fast::Hasher,
    bytes_written: u64,
}

impl<W: Write> CrcWriter<W> {
    /// Wrap `inner`, starting from a zero checksum and byte count.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
            bytes_written: 0,
        }
    }

    /// The CRC-32 of all bytes written so far.
    pub fn checksum(&self) -> u32 {
        self.hasher.clone().finalize()
    }

    /// The number of bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Consume the wrapper and return the inner writer.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for CrcWriter<W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.bytes_written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> IoResult<()> {
        self.inner.flush()
    }
}

/// A [`Read`] wrapper that accumulates a CRC-32 over everything read.
///
/// All reads pass through from the inner reader unchanged; call
/// [`CrcReader::checksum`] after reading the payload to compare against a
/// stored CRC, and [`CrcReader::bytes_read`] to learn how many payload bytes
/// were consumed (e.g. to locate a trailing footer via the file size).
pub struct CrcReader<R: Read> {
    inner: R,
    hasher: crc32fast::Hasher,
    bytes_read: u64,
}

impl<R: Read> CrcReader<R> {
    /// Wrap `inner`, starting from a zero checksum and byte count.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
            bytes_read: 0,
        }
    }

    /// The CRC-32 of all bytes read so far.
    pub fn checksum(&self) -> u32 {
        self.hasher.clone().finalize()
    }

    /// The number of bytes read so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Consume the wrapper and return the inner reader.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Borrow the inner reader (e.g. to read a footer that must not be
    /// included in the running checksum).
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }
}

impl<R: Read> Read for CrcReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        self.bytes_read += n as u64;
        Ok(n)
    }
}

/// A [`StorageInput`] wrapper that accumulates a CRC-32 over bytes read
/// **sequentially** from offset 0, so a reader that parses its payload in a
/// single forward pass (e.g. the Eager `.hnsw` structural parse) can verify a
/// trailing CRC footer during that existing pass instead of a second full read
/// (Issue #789).
///
/// Tracking is only meaningful while reads stay sequential: any real seek
/// (the Lazy parse, which skips the vector rows) clears
/// [`Self::is_sequential`], after which
/// [`Self::checksum`] must not be trusted. A no-op `Seek(Current(0))` (used by
/// [`std::io::Seek::stream_position`]) is served from the byte counter and does
/// **not** break tracking. Pass `track = false` to skip hashing entirely on
/// paths that will not use the running CRC, so the wrapper degrades to a thin
/// position-tracking pass-through.
pub struct ChecksumTrackingInput {
    inner: Box<dyn StorageInput>,
    hasher: crc32fast::Hasher,
    /// Logical position; equals bytes consumed while reads stay sequential.
    pos: u64,
    /// Cleared once a real seek desynchronizes the running CRC.
    sequential: bool,
    /// When `false`, reads pass through without updating the hasher.
    track: bool,
}

impl ChecksumTrackingInput {
    /// Wrap `inner`, starting from a zero checksum at offset 0. When `track`
    /// is `false` the running CRC is not maintained (use for paths that will
    /// not call [`Self::checksum`]).
    pub fn new(inner: Box<dyn StorageInput>, track: bool) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
            pos: 0,
            sequential: true,
            track,
        }
    }

    /// The CRC-32 of all bytes read so far.
    pub fn checksum(&self) -> u32 {
        self.hasher.clone().finalize()
    }

    /// The number of bytes consumed so far.
    pub fn bytes_read(&self) -> u64 {
        self.pos
    }

    /// Whether every read so far has been sequential (no real seek), i.e.
    /// whether [`Self::checksum`] reflects a contiguous prefix from offset 0.
    pub fn is_sequential(&self) -> bool {
        self.sequential
    }

    /// Read and hash any remaining bytes up to `len` (a no-op once `pos >=
    /// len`), so the running CRC covers exactly `len` content bytes after a
    /// structural parse that may have stopped a few bytes short of the footer.
    pub fn absorb_to(&mut self, len: u64) -> IoResult<()> {
        let mut remaining = len.saturating_sub(self.pos);
        let mut buf = [0u8; 64 * 1024];
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            self.read_exact(&mut buf[..want])?;
            remaining -= want as u64;
        }
        Ok(())
    }
}

impl std::fmt::Debug for ChecksumTrackingInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChecksumTrackingInput")
            .field("inner", &self.inner)
            .field("pos", &self.pos)
            .field("sequential", &self.sequential)
            .field("track", &self.track)
            .finish()
    }
}

impl Read for ChecksumTrackingInput {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        let n = self.inner.read(buf)?;
        if self.track {
            self.hasher.update(&buf[..n]);
        }
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for ChecksumTrackingInput {
    fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> {
        // Serve `stream_position()` (a no-op `Current(0)` seek) from the byte
        // counter without touching the inner stream, so it does not count as a
        // real seek that would break sequential CRC tracking.
        if let SeekFrom::Current(0) = pos {
            return Ok(self.pos);
        }
        // Any real seek desynchronizes the running CRC; mark it untrustworthy
        // and resync the position from the inner stream.
        self.sequential = false;
        let new_pos = self.inner.seek(pos)?;
        self.pos = new_pos;
        Ok(new_pos)
    }
}

impl StorageInput for ChecksumTrackingInput {
    fn size(&self) -> Result<u64> {
        self.inner.size()
    }

    fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
        // Return an unwrapped clone of the inner input: clones are used by the
        // Lazy read path (the mapped row pools), which does not participate in
        // the folded CRC and must not inherit this wrapper's running-checksum
        // state.
        self.inner.clone_input()
    }

    fn close(&mut self) -> Result<()> {
        self.inner.close()
    }

    // `as_slice` deliberately keeps the default `None`: forcing reads through
    // `read` guarantees the CRC sees every content byte. The HNSW reader never
    // uses the zero-copy slice path, so this costs nothing there.
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn writer_checksum_matches_standalone_hash() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let mut w = CrcWriter::new(Vec::new());
        w.write_all(data).unwrap();
        assert_eq!(w.bytes_written(), data.len() as u64);

        let mut h = crc32fast::Hasher::new();
        h.update(data);
        assert_eq!(w.checksum(), h.finalize());
    }

    #[test]
    fn reader_checksum_matches_writer() {
        let data = b"laurus integrity check payload";
        let mut w = CrcWriter::new(Vec::new());
        w.write_all(data).unwrap();
        let expected = w.checksum();
        let buf = w.into_inner();

        let mut r = CrcReader::new(Cursor::new(buf));
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, data);
        assert_eq!(r.bytes_read(), data.len() as u64);
        assert_eq!(r.checksum(), expected);
    }

    #[test]
    fn reader_detects_corruption() {
        let data = b"a payload that will be corrupted";
        let mut w = CrcWriter::new(Vec::new());
        w.write_all(data).unwrap();
        let good = w.checksum();
        let mut buf = w.into_inner();
        buf[5] ^= 0xff; // flip a bit

        let mut r = CrcReader::new(Cursor::new(buf));
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_ne!(r.checksum(), good, "a corrupted byte must change the CRC");
    }

    /// Minimal in-memory [`StorageInput`] over a byte buffer, used to exercise
    /// [`ChecksumTrackingInput`] without pulling in a full storage backend.
    #[derive(Debug, Clone)]
    struct TestInput {
        cursor: Cursor<Vec<u8>>,
    }

    impl TestInput {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                cursor: Cursor::new(bytes),
            }
        }
    }

    impl Read for TestInput {
        fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
            self.cursor.read(buf)
        }
    }

    impl Seek for TestInput {
        fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> {
            self.cursor.seek(pos)
        }
    }

    impl StorageInput for TestInput {
        fn size(&self) -> Result<u64> {
            Ok(self.cursor.get_ref().len() as u64)
        }

        fn clone_input(&self) -> Result<Box<dyn StorageInput>> {
            Ok(Box::new(self.clone()))
        }

        fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    /// Hash `bytes` with the same CRC-32 used by the footer writer.
    fn crc32(bytes: &[u8]) -> u32 {
        let mut h = crc32fast::Hasher::new();
        h.update(bytes);
        h.finalize()
    }

    #[test]
    fn tracking_input_sequential_checksum_matches_writer() {
        let data = b"folded crc over a single forward pass".to_vec();
        let expected = crc32(&data);

        let mut input = ChecksumTrackingInput::new(Box::new(TestInput::new(data.clone())), true);
        let mut out = Vec::new();
        input.read_to_end(&mut out).unwrap();

        assert_eq!(out, data);
        assert!(
            input.is_sequential(),
            "a pure forward read stays sequential"
        );
        assert_eq!(input.bytes_read(), data.len() as u64);
        assert_eq!(input.checksum(), expected);
    }

    #[test]
    fn tracking_input_stream_position_keeps_sequential() {
        let data = b"position probe must not break tracking".to_vec();
        let expected = crc32(&data);

        let mut input = ChecksumTrackingInput::new(Box::new(TestInput::new(data.clone())), true);
        let mut head = [0u8; 5];
        input.read_exact(&mut head).unwrap();

        // `stream_position()` issues a no-op `Current(0)` seek; it must be
        // served from the byte counter without clearing sequential tracking.
        assert_eq!(input.stream_position().unwrap(), 5);
        assert!(input.is_sequential());

        let mut rest = Vec::new();
        input.read_to_end(&mut rest).unwrap();
        assert!(input.is_sequential());
        assert_eq!(input.checksum(), expected);
    }

    #[test]
    fn tracking_input_real_seek_clears_sequential() {
        let data = b"a real seek desynchronizes the running crc".to_vec();
        let mut input = ChecksumTrackingInput::new(Box::new(TestInput::new(data)), true);

        let mut head = [0u8; 4];
        input.read_exact(&mut head).unwrap();
        assert!(input.is_sequential());

        input.seek(SeekFrom::Start(10)).unwrap();
        assert!(
            !input.is_sequential(),
            "a real seek must clear sequential tracking"
        );
        assert_eq!(input.bytes_read(), 10, "position resyncs from the seek");
    }

    #[test]
    fn tracking_input_absorb_to_covers_remainder() {
        let data = b"structural parse stops a few bytes short of the footer".to_vec();
        let expected = crc32(&data);

        let mut input = ChecksumTrackingInput::new(Box::new(TestInput::new(data.clone())), true);
        // Read only a prefix, mimicking a parse that stopped early.
        let mut prefix = [0u8; 8];
        input.read_exact(&mut prefix).unwrap();

        input.absorb_to(data.len() as u64).unwrap();
        assert_eq!(input.bytes_read(), data.len() as u64);
        assert_eq!(input.checksum(), expected);

        // A second call is a no-op once the position already covers `len`.
        input.absorb_to(data.len() as u64).unwrap();
        assert_eq!(input.checksum(), expected);
    }

    #[test]
    fn tracking_input_track_false_skips_hashing() {
        let data = b"pass-through with no running checksum".to_vec();

        let mut input = ChecksumTrackingInput::new(Box::new(TestInput::new(data.clone())), false);
        let mut out = Vec::new();
        input.read_to_end(&mut out).unwrap();

        assert_eq!(out, data);
        assert_eq!(
            input.bytes_read(),
            data.len() as u64,
            "position still tracks"
        );
        assert_eq!(
            input.checksum(),
            0,
            "track = false leaves the hasher at the empty-input CRC"
        );
    }

    #[test]
    fn tracking_input_clone_input_unwraps_inner() {
        // `clone_input` must return a plain inner stream, not another
        // `ChecksumTrackingInput`: the Lazy path uses clones and must
        // not inherit this wrapper's running-checksum state (Issue #789).
        let data = b"clone returns a plain unwrapped inner stream".to_vec();
        let mut input = ChecksumTrackingInput::new(Box::new(TestInput::new(data.clone())), true);

        // Advance the wrapper so it carries non-trivial pos + accumulated CRC.
        let mut head = [0u8; 6];
        input.read_exact(&mut head).unwrap();
        let wrapper_crc = input.checksum();
        assert_ne!(wrapper_crc, 0, "wrapper accumulated a CRC over the prefix");

        // The clone reads the full payload from offset 0 independently, and
        // touching it does not perturb the wrapper's running CRC.
        let mut clone = input.clone_input().unwrap();
        clone.seek(SeekFrom::Start(0)).unwrap();
        let mut out = Vec::new();
        clone.read_to_end(&mut out).unwrap();
        assert_eq!(
            out, data,
            "clone reads the full inner payload from the start"
        );
        assert_eq!(
            input.checksum(),
            wrapper_crc,
            "operations on the clone must not touch the wrapper's CRC"
        );
    }
}
