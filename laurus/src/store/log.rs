//! Unified document log combining WAL, doc_id generation, and document storage.
//!
//! [`DocumentLog`] provides a single component that:
//!
//! - Generates monotonically increasing document IDs
//! - Writes all operations to a durable append-only log (WAL)
//! - Stores documents in segmented files for retrieval
//! - Supports recovery by replaying the log
//!
//! ## Architecture
//!
//! ```text
//! DocumentLog
//! ├── WAL (append-only log file)
//! │   └── All fields stored for recovery
//! └── Document Store (segmented files)
//!     └── Only stored fields kept for retrieval
//! ```
//!
//! ## File format
//!
//! The WAL log file stores a sequence of length-prefixed records. There are
//! three framings:
//!
//! - **v3** (current, #822): a 5-byte file header (`b"LWAL"` magic + version
//!   byte `3`), then `[u32 len][u32 crc32][rkyv payload]` per entry. Same CRC
//!   framing as v2, but each payload is a compact rkyv binary record instead of
//!   JSON — typically 3-5x smaller (vectors store as raw `f32`, not decimal
//!   strings) and faster to parse on recovery.
//! - **v2** (#815): a 5-byte header (`b"LWAL"` magic + version byte `2`), then
//!   `[u32 len][u32 crc32][json payload]` per entry. Still read for back-compat;
//!   never written by current code.
//! - **legacy** (pre-#542): no header, `[u32 len][json payload]` per entry, no
//!   checksum. Still read for back-compat.
//!
//! The CRC-32 (v2/v3) is computed over `len || payload`, so both a corrupted
//! length and a corrupted body are detected. A file keeps its framing for its
//! whole life; an older file that survives an upgrade is rewritten as v3 only on
//! the next commit/truncate, and the framings are never mixed within one file.
//!
//! Each entry is followed by `flush_and_sync()` for durability.
//!
//! ## Recovery
//!
//! [`DocumentLog::read_all`] replays the file and stops at the first record it
//! cannot read in full — a short length prefix, a short body, or a body that
//! fails to deserialize. Such a torn trailing record (e.g. from a crash
//! mid-append) is dropped along with everything after it, and the valid prefix
//! is recovered so the engine can still open. Recovery never skips a bad record
//! to continue, which keeps the recovered sequence gap-free and consistent.

use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::{Mutex, RwLock};
use rkyv::api::high::HighSerializer;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
use rkyv::with::Inline;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

use crate::data::Document;
use crate::error::Result;
use crate::storage::Storage;
use crate::store::document::UnifiedDocumentStore;

/// Sequence number for log entries.
pub type SeqNumber = u64;

/// A single operation in the document log.
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum LogEntry {
    /// Insert or update a document.
    Upsert {
        doc_id: u64,
        external_id: String,
        document: Document,
    },
    /// Delete a document.
    Delete {
        doc_id: u64,
        /// External ID of the deleted document.
        /// Uses `#[serde(default)]` for backward compatibility with old WAL
        /// entries that lack this field.
        #[serde(default)]
        external_id: String,
    },
}

/// A log record combining a sequence number with an entry.
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct LogRecord {
    pub seq: SeqNumber,
    pub entry: LogEntry,
}

/// A borrowed mirror of [`LogEntry`], field for field and variant for
/// variant, used only to *write* a record without cloning its `Document`
/// (Issue #1416). `document` is archived with [`Inline`], which gives the
/// exact same archived bytes as the owned `Document` field in [`LogEntry`]
/// (`Inline`'s `Archived` type is `Document`'s own, see `rkyv::with::Inline`),
/// so the v3 WAL format is unchanged. References archived with `Inline`
/// cannot be deserialized, so this type derives `Serialize` (for the
/// v2/legacy JSON path) and rkyv's `Serialize`, but never `Deserialize`.
#[derive(Debug, Serialize, Archive, RkyvSerialize)]
enum LogEntryRef<'a> {
    Upsert {
        doc_id: u64,
        external_id: String,
        #[rkyv(with = Inline)]
        document: &'a Document,
    },
    Delete {
        doc_id: u64,
        external_id: String,
    },
}

/// A borrowed mirror of [`LogRecord`]; see [`LogEntryRef`].
#[derive(Debug, Serialize, Archive, RkyvSerialize)]
struct LogRecordRef<'a> {
    seq: SeqNumber,
    entry: LogEntryRef<'a>,
}

/// Either [`LogRecord`] (owned, used by `append_delete` and the
/// truncate/rewrite path) or [`LogRecordRef`] (borrowed, used by
/// [`DocumentLog::append`]) can be encoded through the same WAL-writing path
/// (Issue #1416). The bound mirrors what `rkyv::to_bytes` itself requires
/// (`rkyv::Serialize<S>` already has `Archive` as a supertrait).
trait WalEncodable:
    serde::Serialize
    + for<'x> RkyvSerialize<HighSerializer<AlignedVec, ArenaHandle<'x>, rkyv::rancor::Error>>
{
}

impl<T> WalEncodable for T where
    T: serde::Serialize
        + for<'x> RkyvSerialize<HighSerializer<AlignedVec, ArenaHandle<'x>, rkyv::rancor::Error>>
{
}

/// Durability policy governing when WAL appends are fsync'd.
///
/// This trades the durability of an individual `append` against ingest
/// throughput; [`commit`](crate::Engine::commit) is a hard barrier under both
/// policies (it always forces the WAL durable before materializing any store).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WalSyncPolicy {
    /// Fsync after every record (the default). An `append` returns only once its
    /// record is durable, so a successful add/put can never be lost to a crash.
    PerRecord,
    /// Defer the fsync and amortize it over a batch: flush when **either**
    /// `max_records` records **or** `max_bytes` bytes have accumulated since the
    /// last sync (whichever comes first), and unconditionally at
    /// [`commit`](crate::Engine::commit). An `append` may return before its
    /// record is durable, so a crash can lose up to the last unsynced batch —
    /// comparable to SQLite's `synchronous = NORMAL`. A torn trailing record is
    /// never resurrected: it is dropped on recovery via the CRC + prefix-stop
    /// path, keeping the recovered log gap-free.
    Group {
        /// Flush once this many records have accumulated since the last sync.
        max_records: usize,
        /// Flush once this many appended bytes have accumulated since the last
        /// sync.
        max_bytes: usize,
        /// Optional periodic flush interval. When `Some`, the engine runs a
        /// background timer that forces the WAL durable at least this often, so
        /// a trailing partial batch under a low ingest rate is not left unsynced
        /// indefinitely (the record/byte thresholds may never be reached).
        /// `None` disables the timer, leaving durability to the thresholds,
        /// [`commit`](crate::Engine::commit), and
        /// [`flush_wal`](crate::Engine::flush_wal).
        ///
        /// Honored on native targets only; on `wasm32` (no background threads)
        /// the interval is ignored.
        max_interval: Option<Duration>,
    },
}

impl Default for WalSyncPolicy {
    /// The default is [`WalSyncPolicy::PerRecord`], preserving per-record
    /// durability for callers that do not opt into group commit.
    fn default() -> Self {
        Self::PerRecord
    }
}

/// Default batch size (record count) for [`WalSyncPolicy::Group`].
pub const DEFAULT_GROUP_MAX_RECORDS: usize = 1024;

/// Default batch size (bytes) for [`WalSyncPolicy::Group`]: 1 MiB.
pub const DEFAULT_GROUP_MAX_BYTES: usize = 1024 * 1024;

impl WalSyncPolicy {
    /// A [`WalSyncPolicy::Group`] policy using the default batch thresholds
    /// ([`DEFAULT_GROUP_MAX_RECORDS`] records / [`DEFAULT_GROUP_MAX_BYTES`]
    /// bytes, whichever is reached first) and **no** periodic flush timer.
    pub fn group_with_defaults() -> Self {
        Self::Group {
            max_records: DEFAULT_GROUP_MAX_RECORDS,
            max_bytes: DEFAULT_GROUP_MAX_BYTES,
            max_interval: None,
        }
    }

    /// A [`WalSyncPolicy::Group`] policy using the default batch thresholds plus
    /// a periodic flush `interval`, so a trailing partial batch is forced
    /// durable at least every `interval` even under a low ingest rate.
    ///
    /// The timer is honored on native targets only; on `wasm32` the interval is
    /// ignored (see [`WalSyncPolicy::Group::max_interval`]).
    ///
    /// # Arguments
    ///
    /// * `interval` - Maximum time a partial batch may remain unsynced.
    pub fn group_with_interval(interval: Duration) -> Self {
        Self::Group {
            max_records: DEFAULT_GROUP_MAX_RECORDS,
            max_bytes: DEFAULT_GROUP_MAX_BYTES,
            max_interval: Some(interval),
        }
    }

    /// The periodic flush interval for this policy, if any.
    ///
    /// Returns the `max_interval` of a [`WalSyncPolicy::Group`], or `None` for
    /// [`WalSyncPolicy::PerRecord`] or a `Group` without a timer.
    pub fn flush_interval(&self) -> Option<Duration> {
        match self {
            Self::Group { max_interval, .. } => *max_interval,
            Self::PerRecord => None,
        }
    }
}

/// Magic bytes at the start of a CRC-framed (v2/v3) WAL file.
const WAL_MAGIC: &[u8; 4] = b"LWAL";

/// Current WAL file-format version stamped into the header of a freshly created
/// or truncated file. v3 keeps the v2 CRC framing but stores each payload as a
/// compact rkyv binary record instead of JSON (#822).
const WAL_VERSION: u8 = 3;

/// v2 file-format version: CRC framing with a JSON payload (#815). Still read
/// for back-compat; never written by current code.
const WAL_VERSION_V2: u8 = 2;

/// Length of the file header: 4-byte [`WAL_MAGIC`] + 1-byte version.
const WAL_HEADER_LEN: u64 = 5;

/// On-disk framing of the WAL file currently being appended to / read.
///
/// A WAL file is written in a single, consistent format for its whole life: a
/// fresh or truncated file is always [`WalFormat::V3`]; an older file that
/// survives an upgrade keeps its existing format ([`WalFormat::V2`] or
/// [`WalFormat::Legacy`]) until the next commit/truncate recreates it as v3.
/// The formats are never mixed within a file, so the reader detects the format
/// once from the file header.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WalFormat {
    /// Pre-#542 frames: `[u32 len][json payload]`, no per-record checksum.
    Legacy,
    /// v2 frames (#815): `[u32 len][u32 crc32][json payload]`, CRC over
    /// `len || payload`.
    V2,
    /// v3 frames (#822): `[u32 len][u32 crc32][rkyv payload]`, same CRC framing
    /// as v2 but a compact binary payload.
    V3,
}

impl WalFormat {
    /// Whether this framing carries a per-record CRC-32 (v2 and v3 do; the
    /// pre-#542 legacy framing does not).
    fn has_crc(self) -> bool {
        matches!(self, WalFormat::V2 | WalFormat::V3)
    }
}

/// The open WAL append handle together with the framing of the file it points
/// at, so [`DocumentLog::write_record`] knows whether to emit a CRC.
#[derive(Debug)]
struct WalWriterState {
    out: Box<dyn crate::storage::StorageOutput>,
    format: WalFormat,
    /// Whether bytes have been appended since the last successful
    /// `flush_and_sync()`. Set by [`DocumentLog::append_record_bytes`] and
    /// cleared by [`DocumentLog::flush_writer`]; it lets [`DocumentLog::flush_wal`]
    /// skip a redundant fsync at commit time when nothing is pending (true under
    /// the default per-record contract, where every append already synced).
    dirty: bool,
    /// Number of records appended since the last successful `flush_and_sync()`.
    /// Drives the [`WalSyncPolicy::Group`] record-count threshold; reset to 0
    /// on flush.
    unsynced_records: usize,
    /// Number of bytes appended since the last successful `flush_and_sync()`
    /// (framed length, including the CRC for v2). Drives the
    /// [`WalSyncPolicy::Group`] byte threshold; reset to 0 on flush.
    unsynced_bytes: usize,
    /// Test-only: number of `flush_and_sync()` calls performed on this writer.
    /// Lets tests assert fsync amortization exactly (e.g. the batch-ingest
    /// path syncing once per batch instead of once per record).
    #[cfg(test)]
    sync_count: usize,
}

/// Unified document log providing WAL, doc_id generation, and document storage.
///
/// This component replaces the separate `WalManager` and `UnifiedDocumentStore`,
/// combining:
/// - **WAL**: durable append-only log for crash recovery
/// - **doc_id generation**: monotonically increasing document IDs
/// - **Document storage**: segmented files for stored-field retrieval
///
/// # Thread safety
///
/// WAL writes are serialized through an internal [`Mutex`].
/// Document store access uses [`parking_lot::RwLock`] for concurrent reads.
/// The `next_doc_id` and `next_seq` counters use [`AtomicU64`] for lock-free reads.
#[derive(Debug)]
pub struct DocumentLog {
    wal_storage: Arc<dyn Storage>,
    wal_path: String,
    next_doc_id: AtomicU64,
    wal_writer: Mutex<Option<WalWriterState>>,
    next_seq: AtomicU64,
    doc_store: RwLock<UnifiedDocumentStore>,
    /// Durability policy controlling when appends are fsync'd. Defaults to
    /// [`WalSyncPolicy::PerRecord`]; [`WalSyncPolicy::Group`] defers the fsync to
    /// amortize it over a batch.
    sync_policy: WalSyncPolicy,
    /// Number of active [`WalSyncDeferral`] scopes (see [`Self::defer_sync`]).
    /// While non-zero, the per-append fsync of [`WalSyncPolicy::PerRecord`] is
    /// suppressed so a batch-ingest caller can amortize one fsync over the
    /// whole batch via [`Self::flush_wal`]. [`WalSyncPolicy::Group`] thresholds
    /// are unaffected (its loss-window contract is preserved mid-batch).
    sync_deferral_depth: AtomicUsize,
}

impl DocumentLog {
    /// Create a new document log with WAL and document storage.
    ///
    /// The internal `next_doc_id` counter starts at **1** and is **not**
    /// automatically recovered from any existing WAL file or document store.
    /// Callers must invoke [`read_all`](Self::read_all) after construction to
    /// replay the WAL and synchronize both the `next_doc_id` and `next_seq`
    /// counters with the persisted state. Failing to do so may cause
    /// duplicate document IDs.
    ///
    /// Equivalent to [`with_sync_policy`](Self::with_sync_policy) with the
    /// default [`WalSyncPolicy::PerRecord`], where every append is fsync'd before
    /// returning.
    ///
    /// # Arguments
    ///
    /// * `wal_storage` - Storage backend for the WAL file.
    /// * `wal_path` - Path (relative to the storage root) of the WAL file.
    /// * `doc_store_storage` - Storage backend for the document store.
    ///
    /// # Errors
    ///
    /// Returns an error if the document store cannot be opened.
    pub fn new(
        wal_storage: Arc<dyn Storage>,
        wal_path: &str,
        doc_store_storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        Self::with_sync_policy(
            wal_storage,
            wal_path,
            doc_store_storage,
            WalSyncPolicy::PerRecord,
        )
    }

    /// Create a new document log with WAL and document storage under an explicit
    /// [`WalSyncPolicy`].
    ///
    /// The internal `next_doc_id` counter starts at **1** and is **not**
    /// automatically recovered from any existing WAL file or document store.
    /// Callers must invoke [`read_all`](Self::read_all) after construction to
    /// replay the WAL and synchronize both the `next_doc_id` and `next_seq`
    /// counters with the persisted state. Failing to do so may cause
    /// duplicate document IDs.
    ///
    /// # Arguments
    ///
    /// * `wal_storage` - Storage backend for the WAL file.
    /// * `wal_path` - Path (relative to the storage root) of the WAL file.
    /// * `doc_store_storage` - Storage backend for the document store.
    /// * `sync_policy` - When to fsync WAL appends. [`WalSyncPolicy::PerRecord`]
    ///   syncs each append; [`WalSyncPolicy::Group`] defers and batches the
    ///   fsync.
    ///
    /// # Errors
    ///
    /// Returns an error if the document store cannot be opened.
    pub fn with_sync_policy(
        wal_storage: Arc<dyn Storage>,
        wal_path: &str,
        doc_store_storage: Arc<dyn Storage>,
        sync_policy: WalSyncPolicy,
    ) -> Result<Self> {
        let doc_store = UnifiedDocumentStore::open(doc_store_storage)?;
        Ok(Self {
            wal_storage,
            wal_path: wal_path.to_string(),
            next_doc_id: AtomicU64::new(1),
            wal_writer: Mutex::new(None),
            next_seq: AtomicU64::new(1),
            doc_store: RwLock::new(doc_store),
            sync_policy,
            sync_deferral_depth: AtomicUsize::new(0),
        })
    }

    /// Detect the framing of an existing (non-empty) WAL file from its header.
    ///
    /// Reads the 5-byte header and classifies the file: the [`WAL_MAGIC`] magic
    /// followed by version `3` is [`WalFormat::V3`], version `2` is
    /// [`WalFormat::V2`], and anything without the magic is the pre-#542
    /// [`WalFormat::Legacy`] framing. An unknown but magic-prefixed version is
    /// treated as v2 (CRC-framed JSON), the most conservative readable framing.
    ///
    /// # Arguments
    ///
    /// * `existing_size` - Size in bytes of the file to classify.
    fn detect_existing_format(&self, existing_size: u64) -> Result<WalFormat> {
        if existing_size < WAL_HEADER_LEN {
            return Ok(WalFormat::Legacy);
        }
        let mut header = [0u8; WAL_HEADER_LEN as usize];
        let mut input = self.wal_storage.open_input(&self.wal_path)?;
        if input.read_exact(&mut header).is_err() || &header[0..4] != WAL_MAGIC {
            return Ok(WalFormat::Legacy);
        }
        Ok(match header[4] {
            WAL_VERSION => WalFormat::V3,
            WAL_VERSION_V2 => WalFormat::V2,
            // An unknown but magic-prefixed version falls back to the v2 (CRC +
            // JSON) reader — the most conservative readable CRC framing.
            _ => WalFormat::V2,
        })
    }

    /// Open or create the WAL file for appending, detecting its framing.
    ///
    /// Operates on an already-held writer guard so a caller can ensure the
    /// writer and append within a single critical section. A fresh or truncated
    /// (empty) file is initialized as [`WalFormat::V3`] by writing the 5-byte
    /// header; an existing file keeps whatever framing it already has (v3/v2 by
    /// header version, else legacy) so the formats are never mixed within one
    /// file.
    fn ensure_writer(&self, writer_guard: &mut Option<WalWriterState>) -> Result<()> {
        if writer_guard.is_some() {
            return Ok(());
        }

        // Inspect the existing file (if any) before opening the append handle.
        let existing_size = if self.wal_storage.file_exists(&self.wal_path) {
            self.wal_storage.open_input(&self.wal_path)?.size()?
        } else {
            0
        };

        let mut out = self.wal_storage.create_output_append(&self.wal_path)?;
        let format = if existing_size == 0 {
            // Fresh/truncated file: stamp the current (v3) header and use CRC
            // framing with a binary payload.
            out.write_all(WAL_MAGIC)?;
            out.write_all(&[WAL_VERSION])?;
            WalFormat::V3
        } else {
            // Existing file: keep appending in its own framing until the next
            // commit/truncate recreates the file as v3.
            self.detect_existing_format(existing_size)?
        };

        *writer_guard = Some(WalWriterState {
            out,
            format,
            dirty: false,
            unsynced_records: 0,
            unsynced_bytes: 0,
            #[cfg(test)]
            sync_count: 0,
        });
        Ok(())
    }

    // ── WAL operations ──────────────────────────────────────────────

    /// Append an upsert entry to the log.
    ///
    /// Atomically assigns a new doc_id and sequence number, then writes the
    /// entry to the log file. Whether the write is fsync'd before returning
    /// depends on the configured [`WalSyncPolicy`]: always under
    /// [`WalSyncPolicy::PerRecord`], or only when the batch threshold is reached
    /// under [`WalSyncPolicy::Group`].
    ///
    /// Returns `(doc_id, seq_number)`.
    ///
    /// Takes `doc` by reference and serializes it borrowed (Issue #1416): a
    /// caller that still needs `doc` after this call (the engine's put path
    /// does, for the lexical store and stored-field filtering) no longer has
    /// to clone it just to hand the WAL a copy to own and drop.
    pub fn append(&self, external_id: &str, doc: &Document) -> Result<(u64, SeqNumber)> {
        // Single critical section: ensure the writer, allocate ids, and write
        // under one lock so the on-disk record order matches id/seq allocation.
        let mut writer_guard = self.wal_writer.lock();
        self.ensure_writer(&mut writer_guard)?;

        let doc_id = self.next_doc_id.fetch_add(1, Ordering::SeqCst);
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);

        let record = LogRecordRef {
            seq,
            entry: LogEntryRef::Upsert {
                doc_id,
                external_id: external_id.to_string(),
                document: doc,
            },
        };

        self.write_record(&mut writer_guard, &record)?;

        Ok((doc_id, seq))
    }

    /// Append a delete entry to the log.
    ///
    /// As with [`append`](Self::append), the fsync timing follows the configured
    /// [`WalSyncPolicy`].
    ///
    /// Returns the assigned sequence number.
    pub fn append_delete(&self, doc_id: u64, external_id: &str) -> Result<SeqNumber> {
        let mut writer_guard = self.wal_writer.lock();
        self.ensure_writer(&mut writer_guard)?;

        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);

        let record = LogRecordRef {
            seq,
            entry: LogEntryRef::Delete {
                doc_id,
                external_id: external_id.to_string(),
            },
        };

        self.write_record(&mut writer_guard, &record)?;

        Ok(seq)
    }

    /// Encode a record's payload for the given framing.
    ///
    /// v3 emits a compact rkyv binary record; v2 and legacy emit a JSON
    /// document. The bytes returned here are the *payload* only — the length
    /// prefix and (for v2/v3) the CRC are added by [`Self::append_record_bytes`].
    ///
    /// # Arguments
    ///
    /// * `record` - The log record to encode.
    /// * `format` - The framing of the file being appended to.
    ///
    /// # Errors
    ///
    /// Returns a [`LaurusError::SerializationError`](crate::error::LaurusError::SerializationError)
    /// if rkyv (v3) or `serde_json` (v2/legacy) encoding fails.
    fn encode_payload<T: WalEncodable>(record: &T, format: WalFormat) -> Result<Vec<u8>> {
        match format {
            WalFormat::V3 => rkyv::to_bytes::<rkyv::rancor::Error>(record)
                .map(|bytes| bytes.to_vec())
                .map_err(|e| {
                    crate::error::LaurusError::SerializationError(format!(
                        "WAL rkyv encode failed: {e}"
                    ))
                }),
            WalFormat::V2 | WalFormat::Legacy => Ok(serde_json::to_vec(record)?),
        }
    }

    /// Decode a record's payload according to the file's framing.
    ///
    /// The inverse of [`Self::encode_payload`]: v3 reads a compact rkyv binary
    /// record (validated by `rkyv`/`bytecheck`), v2 and legacy read a JSON
    /// document.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The payload bytes (length prefix and CRC already stripped).
    /// * `format` - The framing of the file being read.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes do not decode under `format`; the caller
    /// ([`Self::read_all`]) treats this as a torn trailing record and stops at
    /// the last valid record.
    fn decode_payload(buffer: &[u8], format: WalFormat) -> Result<LogRecord> {
        match format {
            WalFormat::V3 => {
                rkyv::from_bytes::<LogRecord, rkyv::rancor::Error>(buffer).map_err(|e| {
                    crate::error::LaurusError::SerializationError(format!(
                        "WAL rkyv decode failed: {e}"
                    ))
                })
            }
            WalFormat::V2 | WalFormat::Legacy => Ok(serde_json::from_slice(buffer)?),
        }
    }

    /// Encode and append a record's framed bytes to the WAL buffer **without**
    /// fsyncing.
    ///
    /// v2/v3 writers emit `[u32 len][u32 crc32][payload]` (CRC over `len ||
    /// payload`); legacy writers emit `[u32 len][payload]`. The payload encoding
    /// (rkyv binary for v3, JSON for v2/legacy) follows the writer's
    /// [`WalFormat`]. Durability is the caller's responsibility via
    /// [`Self::flush_writer`] — this split lets the group-commit path amortize
    /// one fsync over many appended records (#542). A no-op if no writer is open.
    fn append_record_bytes<T: WalEncodable>(
        state: &mut Option<WalWriterState>,
        record: &T,
    ) -> Result<()> {
        let Some(state) = state.as_mut() else {
            return Ok(());
        };

        let bytes = Self::encode_payload(record, state.format)?;
        let len: u32 = bytes.len().try_into().map_err(|_| {
            crate::error::LaurusError::InvalidOperation(format!(
                "WAL record size {} exceeds u32::MAX",
                bytes.len()
            ))
        })?;
        let len_bytes = len.to_le_bytes();

        // Frame size: length prefix (+ CRC for v2/v3) + payload.
        let mut frame_len = len_bytes.len() + bytes.len();
        state.out.write_all(&len_bytes)?;
        if state.format.has_crc() {
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(&len_bytes);
            hasher.update(&bytes);
            let crc = hasher.finalize().to_le_bytes();
            state.out.write_all(&crc)?;
            frame_len += crc.len();
        }
        state.out.write_all(&bytes)?;
        // Bytes are now buffered/written but not yet fsynced. Track how much
        // is pending so a Group policy can flush once its batch threshold is
        // reached.
        state.dirty = true;
        state.unsynced_records += 1;
        state.unsynced_bytes = state.unsynced_bytes.saturating_add(frame_len);

        Ok(())
    }

    /// Flush and fsync the open WAL writer, making every appended-but-unsynced
    /// record durable, then mark it clean.
    ///
    /// Honors the [`WalWriterState::dirty`] guard: if nothing has been appended
    /// since the last sync (the steady state under the per-record contract,
    /// where every append already synced), the fsync is skipped so a commit-time
    /// [`Self::flush_wal`] is a true no-op. A no-op if no writer is open.
    fn flush_writer(state: &mut Option<WalWriterState>) -> Result<()> {
        if let Some(state) = state.as_mut()
            && state.dirty
        {
            state.out.flush_and_sync()?;
            state.dirty = false;
            state.unsynced_records = 0;
            state.unsynced_bytes = 0;
            #[cfg(test)]
            {
                state.sync_count += 1;
            }
        }
        Ok(())
    }

    /// Append a record and fsync according to the effective sync policy.
    ///
    /// Under [`WalSyncPolicy::PerRecord`] the record is always fsync'd before
    /// returning (per-record durability — the default contract), unless a
    /// [`WalSyncDeferral`] scope is active (see [`Self::defer_sync`]), in which
    /// case the fsync is left to the scope owner's batch-end
    /// [`Self::flush_wal`]. Under [`WalSyncPolicy::Group`] the fsync is
    /// deferred and only happens once the batch reaches `max_records` records
    /// or `max_bytes` bytes since the last sync, amortizing one fsync over the
    /// batch — deferral scopes do not suppress these threshold flushes, so the
    /// Group policy's bounded loss window holds even mid-batch;
    /// [`Self::flush_wal`] (called at commit) forces any trailing partial
    /// batch durable.
    fn write_record<T: WalEncodable>(
        &self,
        state: &mut Option<WalWriterState>,
        record: &T,
    ) -> Result<()> {
        Self::append_record_bytes(state, record)?;
        let flush_now = match self.sync_policy {
            WalSyncPolicy::PerRecord => self.sync_deferral_depth.load(Ordering::Acquire) == 0,
            group @ WalSyncPolicy::Group { .. } => Self::batch_ready(state, group),
        };
        if flush_now {
            Self::flush_writer(state)?;
        }
        Ok(())
    }

    /// Whether the pending (unsynced) batch should be flushed now under `policy`.
    ///
    /// Always `true` for [`WalSyncPolicy::PerRecord`]; for
    /// [`WalSyncPolicy::Group`] it is `true` once either batch threshold is met.
    /// `false` when no writer is open.
    fn batch_ready(state: &Option<WalWriterState>, policy: WalSyncPolicy) -> bool {
        match policy {
            WalSyncPolicy::PerRecord => true,
            WalSyncPolicy::Group {
                max_records,
                max_bytes,
                // The timer is driven by the engine, not the per-append path.
                max_interval: _,
            } => state.as_ref().is_some_and(|s| {
                s.unsynced_records >= max_records.max(1) || s.unsynced_bytes >= max_bytes.max(1)
            }),
        }
    }

    /// Force every appended-but-unsynced WAL record durable.
    ///
    /// Under the default per-record contract this is a near-no-op: each append
    /// already self-syncs, so the [`WalWriterState::dirty`] guard skips the
    /// fsync. It is the load-bearing commit-time barrier once a future
    /// group-commit path (#542 Phase 4) defers per-append fsync — [`Engine::commit`]
    /// calls it before any store materializes its state, so the WAL is never
    /// less durable than the committed lexical/vector indexes. A no-op if no
    /// writer is currently open (e.g. right after a [`Self::truncate`]).
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or fsyncing the open WAL writer fails.
    pub fn flush_wal(&self) -> Result<()> {
        let mut writer_guard = self.wal_writer.lock();
        Self::flush_writer(&mut writer_guard)
    }

    /// Re-assert the per-record durability contract after a singular append.
    ///
    /// Under [`WalSyncPolicy::PerRecord`] this forces any appended-but-unsynced
    /// WAL bytes durable. It is a no-op when the append already self-synced
    /// (the common case — the dirty guard skips the fsync), but it is the
    /// load-bearing fsync when a **concurrent batch** holds a
    /// [`Self::defer_sync`] scope: the scope suppresses the per-record fsync
    /// globally, so a singular write acknowledged during a batch would
    /// otherwise silently lose its durability guarantee. Singular write entry
    /// points call this before returning. Under [`WalSyncPolicy::Group`] it is
    /// a no-op by design — that policy's bounded loss window applies to
    /// singular writes too.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or fsyncing the open WAL writer fails.
    pub fn ensure_per_record_durability(&self) -> Result<()> {
        match self.sync_policy {
            WalSyncPolicy::PerRecord => self.flush_wal(),
            WalSyncPolicy::Group { .. } => Ok(()),
        }
    }

    /// Open a WAL sync-deferral scope for a batch of appends.
    ///
    /// While the returned [`WalSyncDeferral`] guard is alive, appends under
    /// [`WalSyncPolicy::PerRecord`] skip their per-record fsync; the scope
    /// owner is responsible for calling [`Self::flush_wal`] once at batch end
    /// (on both success and error paths) so the whole batch becomes durable
    /// with a single fsync. Under [`WalSyncPolicy::Group`] the scope changes
    /// nothing: the group thresholds keep firing, preserving that policy's
    /// bounded loss window.
    ///
    /// Scopes may nest (e.g. concurrent batches on different tasks); the
    /// per-record fsync resumes once every guard has been dropped. Dropping
    /// the guard does **not** flush — records appended inside an abandoned
    /// scope stay buffered until the next append, [`Self::flush_wal`], or
    /// commit makes them durable.
    ///
    /// # Returns
    ///
    /// An RAII guard that re-enables per-record fsync when dropped.
    pub fn defer_sync(&self) -> WalSyncDeferral<'_> {
        self.sync_deferral_depth.fetch_add(1, Ordering::AcqRel);
        WalSyncDeferral { log: self }
    }

    /// Test-only: whether the open writer has appended-but-unsynced bytes.
    ///
    /// Lets tests observe that a deferred (group-commit) batch has been flushed
    /// — e.g. by the background flush timer — without exposing the writer
    /// internals outside the crate.
    #[cfg(test)]
    pub(crate) fn wal_is_dirty(&self) -> bool {
        self.wal_writer
            .lock()
            .as_ref()
            .is_some_and(|state| state.dirty)
    }

    /// Test-only: number of fsyncs performed by the currently open WAL writer.
    ///
    /// The counter belongs to the open [`WalWriterState`] and therefore resets
    /// when the writer is recreated (e.g. after [`Self::truncate`]). Used to
    /// assert fsync amortization exactly — e.g. that a deferred batch of N
    /// appends syncs once, not N times.
    #[cfg(test)]
    pub(crate) fn wal_sync_count(&self) -> usize {
        self.wal_writer
            .lock()
            .as_ref()
            .map_or(0, |state| state.sync_count)
    }

    /// Read all records from the WAL.
    ///
    /// Also updates internal counters (`next_seq`, `next_doc_id`) to be
    /// greater than the maximum values found in the log, and syncs
    /// `next_doc_id` with the committed document store segments.
    pub fn read_all(&self) -> Result<Vec<LogRecord>> {
        if !self.wal_storage.file_exists(&self.wal_path) {
            // Even with an empty WAL, sync next_doc_id with doc_store.
            let store_next = self.doc_store.read().next_doc_id();
            self.set_next_doc_id(store_next);
            return Ok(Vec::new());
        }

        let size = self.wal_storage.open_input(&self.wal_path)?.size()?;

        // Detect the framing from the file header: v3/v2 files start with the
        // magic + version byte, a legacy file goes straight into `[u32 len]`
        // records. The payload encoding (rkyv for v3, JSON for v2/legacy) and
        // CRC presence (v2/v3 only) follow from the format.
        let format = self.detect_existing_format(size)?;
        let has_header = format != WalFormat::Legacy;

        let mut reader = self.wal_storage.open_input(&self.wal_path)?;
        let mut records = Vec::new();
        let mut max_seq: u64 = 0;
        let mut max_doc_id: u64 = 0;

        // Skip the file header (magic + version) so the loop starts at the first
        // record; legacy files have no header.
        let mut position = if has_header {
            let mut header = [0u8; WAL_HEADER_LEN as usize];
            reader.read_exact(&mut header)?;
            WAL_HEADER_LEN
        } else {
            0
        };

        while position < size {
            if position + 4 > size {
                break;
            }
            let mut len_bytes = [0u8; 4];
            reader.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as u64;
            position += 4;

            // v2/v3 frames carry a CRC-32 (over `len || payload`) after the
            // length; legacy frames do not.
            let crc_expected = if format.has_crc() {
                if position + 4 > size {
                    break;
                }
                let mut crc_bytes = [0u8; 4];
                reader.read_exact(&mut crc_bytes)?;
                position += 4;
                Some(u32::from_le_bytes(crc_bytes))
            } else {
                None
            };

            if position + len > size {
                break;
            }

            let mut buffer = vec![0u8; len as usize];
            reader.read_exact(&mut buffer)?;
            position += len;

            // Verify the CRC before trusting the body: a mismatch means a torn
            // or bit-rotted record and is treated exactly like a short read —
            // stop at the valid prefix (Issue #542, Phase 1).
            if let Some(expected) = crc_expected {
                let mut hasher = crc32fast::Hasher::new();
                hasher.update(&len_bytes);
                hasher.update(&buffer);
                if hasher.finalize() != expected {
                    ::log::warn!(
                        "WAL recovery: CRC mismatch at byte offset {} ({len} body bytes); \
                         recovered {} valid record(s) before it",
                        position - len,
                        records.len()
                    );
                    break;
                }
            }

            // A trailing record with an intact length prefix but a corrupt or
            // partially written body indicates a torn write (e.g. a crash
            // mid-append). Stop at the last valid record and recover the durable
            // prefix instead of aborting recovery — this mirrors the short-read
            // breaks above and lets the engine open. Recovery stops at the FIRST
            // bad record (never skip-and-continue), so a later intact-looking
            // record can never resurrect an op whose predecessors were dropped.
            // (Issue #542, Phase 0.)
            let record: LogRecord = match Self::decode_payload(&buffer, format) {
                Ok(record) => record,
                Err(e) => {
                    ::log::warn!(
                        "WAL recovery: dropping corrupt trailing record at byte offset {} \
                         ({len} body bytes): {e}; recovered {} valid record(s) before it",
                        position - len,
                        records.len()
                    );
                    break;
                }
            };
            if record.seq > max_seq {
                max_seq = record.seq;
            }
            if let LogEntry::Upsert { doc_id, .. } = &record.entry
                && *doc_id > max_doc_id
            {
                max_doc_id = *doc_id;
            }
            records.push(record);
        }

        // Update counters to continue from the highest values found.
        let current_next_seq = self.next_seq.load(Ordering::SeqCst);
        if max_seq >= current_next_seq {
            self.next_seq.store(max_seq + 1, Ordering::SeqCst);
        }
        let current_next_doc = self.next_doc_id.load(Ordering::SeqCst);
        if max_doc_id >= current_next_doc {
            self.next_doc_id.store(max_doc_id + 1, Ordering::SeqCst);
        }

        // Sync next_doc_id with committed doc_store segments.
        let store_next = self.doc_store.read().next_doc_id();
        self.set_next_doc_id(store_next);

        Ok(records)
    }

    /// Truncate (clear) the WAL.
    ///
    /// Called after a successful commit to discard processed entries.
    pub fn truncate(&self) -> Result<()> {
        // Retain nothing: the caller asserts every record is covered by a
        // durable checkpoint. `SeqNumber::MAX` leaves no record with a greater
        // seq, so this is the historical whole-file truncate.
        self.truncate_retaining_after(SeqNumber::MAX)
    }

    /// Truncate the WAL, **retaining every record whose seq is greater than
    /// `retain_after_seq`** (Issue #876).
    ///
    /// [`Self::truncate`] wipes the whole file, which is only safe when every
    /// record it discards is already covered by the stores' persisted
    /// checkpoints. A mutation that lands *while* a commit is running is not:
    /// the commit ladder is not serialized against ingestion, so a record can
    /// be appended after the stores materialized their state but before the
    /// truncate. Wiping it would lose an acknowledged write on the next crash —
    /// its data lives only in the fresh in-memory writer. Passing the stores'
    /// durable checkpoint keeps exactly those uncovered records, so recovery
    /// replays them.
    ///
    /// # Arguments
    ///
    /// * `retain_after_seq` - The highest sequence number the caller has made
    ///   durable. Records with `seq > retain_after_seq` are preserved; the rest
    ///   are discarded.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing, reading back, recreating, or re-appending
    /// to the WAL file fails.
    pub fn truncate_retaining_after(&self, retain_after_seq: SeqNumber) -> Result<()> {
        // Hold the writer lock across the whole operation so a concurrent
        // append cannot interleave with the read-back and rewrite; it blocks
        // and lands cleanly on the file this call produces.
        let mut writer_guard = self.wal_writer.lock();
        // Flush + close the open handle before reading/discarding it. The flush
        // is guarded by `dirty`, so it's skipped when every record is already
        // self-synced (the per-record default) but still runs once fsync is
        // deferred — dropping the handle with unsynced bytes would lose them
        // (#542 Phase 2/3).
        if let Some(mut state) = writer_guard.take() {
            if state.dirty {
                state.out.flush_and_sync()?;
            }
            state.out.close()?;
        }

        // Fast path: nothing was appended past the caller's checkpoint, so the
        // whole file is covered and an in-place wipe is safe — every record it
        // discards is already durably represented in the stores' committed
        // segments, so there is nothing at risk in the moment the file is
        // empty. This is the no-concurrency common case and is byte-identical
        // to the historical `truncate()`.
        if self.last_seq() <= retain_after_seq {
            let mut writer = self.wal_storage.create_output(&self.wal_path)?;
            writer.flush_and_sync()?;
            writer.close()?;
            self.wal_storage.sync()?;
            return Ok(());
        }

        // Slow path: a mutation raced this commit. The writer is closed, so the
        // file is complete. `read_all` only ever advances the id/seq counters
        // (never regresses them), so replaying it here is safe.
        let retained: Vec<LogRecord> = self
            .read_all()?
            .into_iter()
            .filter(|record| record.seq > retain_after_seq)
            .collect();

        // Build the retained tail in a TEMP file and atomically rename it over
        // the WAL, rather than truncating the live file in place and rewriting
        // it afterwards. `retained` holds exactly the records that are NOT yet
        // durably represented anywhere else (that is why this path exists at
        // all) — an in-place wipe would make the WAL durably empty for the
        // window before the rewrite completes, so a crash in that window would
        // permanently lose them, reintroducing the very bug (#876) this method
        // fixes. The rename is a single atomic filesystem operation: the WAL is
        // never observably in a state that is missing a retained record.
        let (tmp_name, tmp_out) = self.wal_storage.create_temp_output(&self.wal_path)?;
        let mut tmp_state = Some(WalWriterState {
            out: tmp_out,
            format: WalFormat::V3,
            dirty: false,
            unsynced_records: 0,
            unsynced_bytes: 0,
            #[cfg(test)]
            sync_count: 0,
        });
        if let Some(state) = tmp_state.as_mut() {
            state.out.write_all(WAL_MAGIC)?;
            state.out.write_all(&[WAL_VERSION])?;
        }
        // Defer the per-record fsync `write_record` would otherwise do under
        // `WalSyncPolicy::PerRecord` (one syscall per retained record) and
        // sync once after the whole tail is written instead.
        {
            let _deferral = self.defer_sync();
            for record in &retained {
                self.write_record(&mut tmp_state, record)?;
            }
        }
        if let Some(state) = tmp_state.as_mut() {
            state.out.flush_and_sync()?;
            state.out.close()?;
        }

        self.wal_storage.rename_file(&tmp_name, &self.wal_path)?;
        // Sync storage to ensure the renamed WAL file is visible. Critical on
        // Windows where directory listings and file metadata may be cached.
        self.wal_storage.sync()?;

        Ok(())
    }

    /// Get the last used sequence number.
    pub fn last_seq(&self) -> SeqNumber {
        self.next_seq.load(Ordering::SeqCst).saturating_sub(1)
    }

    /// Get the current next_doc_id value.
    pub fn next_doc_id(&self) -> u64 {
        self.next_doc_id.load(Ordering::SeqCst)
    }

    /// Set the next_doc_id if the given value is higher than the current one.
    pub fn set_next_doc_id(&self, id: u64) {
        let current = self.next_doc_id.load(Ordering::SeqCst);
        if id > current {
            self.next_doc_id.store(id, Ordering::SeqCst);
        }
    }

    // ── Document store operations ───────────────────────────────────

    /// Store a document with a specific doc_id.
    ///
    /// This stores the document in the segmented document store for later
    /// retrieval. The document should already have non-stored fields
    /// filtered out.
    pub fn store_document(&self, doc_id: u64, doc: Document) {
        self.doc_store.write().put_document_with_id(doc_id, doc);
    }

    /// Get a document by its internal doc_id.
    pub fn get_document(&self, doc_id: u64) -> Result<Option<Document>> {
        self.doc_store.read().get_document(doc_id)
    }

    /// Retrieve multiple documents by their internal IDs in a single batch.
    ///
    /// More efficient than individual [`get_document()`](Self::get_document) calls
    /// because each segment file is opened and scanned only once.
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Slice of internal document IDs to retrieve.
    ///
    /// # Returns
    ///
    /// A map of doc_id to [`Document`] for all found documents.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`] on storage I/O or deserialization failure.
    pub fn get_documents_batch(
        &self,
        doc_ids: &[u64],
    ) -> Result<std::collections::HashMap<u64, Document>> {
        self.doc_store.read().get_documents_batch(doc_ids)
    }

    /// The value of `field` in the stored documents of `doc_ids`, without
    /// populating the document cache (see
    /// [`UnifiedDocumentStore::get_field_values_batch`](crate::store::document::UnifiedDocumentStore::get_field_values_batch)).
    ///
    /// # Arguments
    ///
    /// * `doc_ids` - Internal document IDs to look up.
    /// * `field` - The stored field to read.
    ///
    /// # Returns
    ///
    /// A map of doc_id to the field's value, for every document found that
    /// has the field.
    ///
    /// # Errors
    ///
    /// Returns [`LaurusError`](crate::error::LaurusError) on storage I/O or
    /// deserialization failure.
    pub fn get_field_values_batch(
        &self,
        doc_ids: &[u64],
        field: &str,
    ) -> Result<std::collections::HashMap<u64, crate::data::DataValue>> {
        self.doc_store.read().get_field_values_batch(doc_ids, field)
    }

    /// Find internal doc_id by external ID.
    pub fn find_by_external_id(&self, external_id: &str) -> Result<Option<u64>> {
        self.doc_store.read().find_by_external_id(external_id)
    }

    /// Find all internal doc_ids by external ID.
    pub fn find_all_by_external_id(&self, external_id: &str) -> Result<Vec<u64>> {
        self.doc_store.read().find_all_by_external_id(external_id)
    }

    /// Commit the document store (flush pending docs to segments).
    pub fn commit_documents(&self) -> Result<()> {
        self.doc_store.write().commit()
    }
}

/// RAII guard for a WAL sync-deferral scope (see [`DocumentLog::defer_sync`]).
///
/// While alive, appends under [`WalSyncPolicy::PerRecord`] skip their
/// per-record fsync so a batch caller can amortize one fsync over the whole
/// batch. Dropping the guard only re-enables per-record fsync; it does not
/// flush — the scope owner must call [`DocumentLog::flush_wal`] at batch end.
#[derive(Debug)]
pub struct WalSyncDeferral<'a> {
    log: &'a DocumentLog,
}

impl Drop for WalSyncDeferral<'_> {
    fn drop(&mut self) {
        self.log.sync_deferral_depth.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataValue, Document};
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    fn make_storage() -> Arc<dyn Storage> {
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()))
    }

    /// `LogRecordRef` (the borrowed type `append` builds, Issue #1416) must
    /// encode to exactly the same bytes as the owned `LogRecord` holding the
    /// same values, under every WAL framing — otherwise the WAL format would
    /// silently change. Fails if `LogEntryRef`'s variant order, field order,
    /// or `external_id`'s representation (e.g. switched to `&str` +
    /// `InlineAsBox`, which archives as `ArchivedBox<str>` instead of
    /// `ArchivedString`) drifts from `LogEntry`'s.
    #[test]
    fn borrowed_record_encodes_identically_to_owned() {
        let doc = Document::builder()
            .add_field(
                "tokens",
                DataValue::VectorArray(vec![vec![0.1, 0.2, 0.3], vec![1.0, 2.0, 3.0]]),
            )
            .add_field("title", DataValue::Text("hello".to_string()))
            .build();

        let owned = LogRecord {
            seq: 7,
            entry: LogEntry::Upsert {
                doc_id: 1,
                external_id: "ext_1".to_string(),
                document: doc.clone(),
            },
        };
        let borrowed = LogRecordRef {
            seq: 7,
            entry: LogEntryRef::Upsert {
                doc_id: 1,
                external_id: "ext_1".to_string(),
                document: &doc,
            },
        };

        for format in [WalFormat::V3, WalFormat::V2, WalFormat::Legacy] {
            let owned_bytes = DocumentLog::encode_payload(&owned, format).unwrap();
            let borrowed_bytes = DocumentLog::encode_payload(&borrowed, format).unwrap();
            assert_eq!(
                owned_bytes, borrowed_bytes,
                "archived bytes must match under {format:?}"
            );
        }

        let v3_bytes = DocumentLog::encode_payload(&borrowed, WalFormat::V3).unwrap();
        let decoded = DocumentLog::decode_payload(&v3_bytes, WalFormat::V3).unwrap();
        assert_eq!(decoded.seq, 7);
        match decoded.entry {
            LogEntry::Upsert {
                doc_id,
                external_id,
                document,
            } => {
                assert_eq!(doc_id, 1);
                assert_eq!(external_id, "ext_1");
                assert_eq!(document.fields.len(), doc.fields.len());
            }
            _ => panic!("wrong variant"),
        }
    }

    fn make_log() -> DocumentLog {
        let wal_storage = make_storage();
        let doc_storage = make_storage();
        DocumentLog::new(wal_storage, "test.log", doc_storage).unwrap()
    }

    /// A log under [`WalSyncPolicy::Group`] with the given batch thresholds and
    /// no flush timer.
    fn make_group_log(max_records: usize, max_bytes: usize) -> DocumentLog {
        DocumentLog::with_sync_policy(
            make_storage(),
            "test.log",
            make_storage(),
            WalSyncPolicy::Group {
                max_records,
                max_bytes,
                max_interval: None,
            },
        )
        .unwrap()
    }

    /// A small upsert document for tests that only care about append counting.
    fn small_doc() -> Document {
        Document::builder()
            .add_field("body", DataValue::Text("x".to_string()))
            .build()
    }

    #[test]
    fn test_append_and_read() {
        let log = make_log();

        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello".to_string()))
            .build();

        // Append upsert.
        let (doc_id, seq1) = log.append("ext_1", &doc).unwrap();
        assert_eq!(doc_id, 1);
        assert_eq!(seq1, 1);

        // Append delete.
        let seq2 = log.append_delete(doc_id, "ext_1").unwrap();
        assert_eq!(seq2, 2);

        // Read all.
        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 2);

        assert_eq!(records[0].seq, 1);
        match &records[0].entry {
            LogEntry::Upsert {
                doc_id,
                external_id,
                ..
            } => {
                assert_eq!(*doc_id, 1);
                assert_eq!(external_id, "ext_1");
            }
            _ => panic!("Expected Upsert"),
        }

        assert_eq!(records[1].seq, 2);
        match &records[1].entry {
            LogEntry::Delete {
                doc_id,
                external_id,
            } => {
                assert_eq!(*doc_id, 1);
                assert_eq!(external_id, "ext_1");
            }
            _ => panic!("Expected Delete"),
        }
    }

    #[test]
    fn test_truncate() {
        let log = make_log();

        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello".to_string()))
            .build();

        log.append("ext_1", &doc).unwrap();
        log.truncate().unwrap();

        let records = log.read_all().unwrap();
        assert!(records.is_empty());

        // Sequence and doc_id should continue monotonically.
        let doc2 = Document::builder()
            .add_field("body", DataValue::Text("world".to_string()))
            .build();
        let (doc_id, seq) = log.append("ext_2", &doc2).unwrap();
        assert_eq!(doc_id, 2);
        assert_eq!(seq, 2);
    }

    /// After a truncate the WAL is recreated as a fresh v2 file: the discarded
    /// handle is flushed and closed, the file is reset to empty, and the next
    /// append re-stamps the magic+version header so records round-trip gap-free.
    /// Guards the Phase 2 truncate seam, which now flushes and closes the open
    /// writer before dropping it rather than discarding the handle outright
    /// (Issue #542, Phase 2).
    #[test]
    fn truncate_recreates_fresh_v3_wal() {
        use std::io::Read as _;

        let wal_storage = make_storage();
        let doc_storage = make_storage();
        let log = DocumentLog::new(wal_storage.clone(), "test.log", doc_storage).unwrap();

        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello".to_string()))
            .build();
        log.append("ext_1", &doc).unwrap();
        log.append("ext_2", &doc).unwrap();

        log.truncate().unwrap();

        // Truncate resets the file to empty (the v3 header is re-stamped lazily
        // on the next append), so recovery finds nothing.
        assert_eq!(
            wal_storage.open_input("test.log").unwrap().size().unwrap(),
            0,
            "truncate leaves an empty WAL file"
        );
        assert!(log.read_all().unwrap().is_empty());

        // The next append re-initializes the file in v3 framing.
        log.append("ext_3", &doc).unwrap();
        let mut header = [0u8; WAL_HEADER_LEN as usize];
        wal_storage
            .open_input("test.log")
            .unwrap()
            .read_exact(&mut header)
            .unwrap();
        assert_eq!(&header[0..4], WAL_MAGIC);
        assert_eq!(header[4], WAL_VERSION);

        // The post-truncate record round-trips and the sequence stays gap-free.
        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, 3);
    }

    /// `flush_wal` is a no-op when no writer is open — before the first append
    /// or right after a truncate — returning Ok without opening anything
    /// (Issue #542, Phase 3).
    #[test]
    fn flush_wal_is_noop_without_writer() {
        let log = make_log();
        log.flush_wal().unwrap();
        assert!(
            log.wal_writer.lock().is_none(),
            "flush_wal must not open a writer"
        );
    }

    /// Under the per-record default every append self-syncs, leaving the writer
    /// clean, so a commit-time `flush_wal` skips the fsync (a true no-op) while
    /// the records stay intact (Issue #542, Phase 3).
    #[test]
    fn append_leaves_writer_clean_so_flush_wal_is_noop() {
        let log = make_log();
        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello".to_string()))
            .build();
        log.append("ext_1", &doc).unwrap();

        assert!(
            !log.wal_writer.lock().as_ref().unwrap().dirty,
            "a per-record append leaves the writer synced/clean"
        );
        log.flush_wal().unwrap();
        assert!(
            !log.wal_writer.lock().as_ref().unwrap().dirty,
            "flush_wal on a clean writer stays clean"
        );

        assert_eq!(log.read_all().unwrap().len(), 1);
    }

    /// When bytes are appended without an immediate sync — the deferred path a
    /// future group-commit mode uses — the writer is dirty and `flush_wal` makes
    /// the record durable, clears the flag, and the record round-trips (Issue
    /// #542, Phase 3).
    #[test]
    fn flush_wal_syncs_a_dirty_writer() {
        let log = make_log();
        let record = LogRecord {
            seq: 1,
            entry: LogEntry::Upsert {
                doc_id: 1,
                external_id: "ext_1".to_string(),
                document: Document::builder()
                    .add_field("body", DataValue::Text("deferred".to_string()))
                    .build(),
            },
        };

        // Simulate a deferred (group-commit) append: open the writer and write
        // the framed bytes without syncing.
        {
            let mut guard = log.wal_writer.lock();
            log.ensure_writer(&mut guard).unwrap();
            DocumentLog::append_record_bytes(&mut guard, &record).unwrap();
            assert!(
                guard.as_ref().unwrap().dirty,
                "appended-but-unsynced bytes mark the writer dirty"
            );
        }

        // flush_wal makes the deferred record durable and clears the flag.
        log.flush_wal().unwrap();
        assert!(
            !log.wal_writer.lock().as_ref().unwrap().dirty,
            "flush_wal clears the dirty flag after syncing"
        );

        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, 1);
    }

    /// #551: while a `defer_sync` scope is alive, per-record appends skip
    /// their fsync; the batch-end `flush_wal` makes everything durable with a
    /// single sync, and the records round-trip.
    #[test]
    fn defer_sync_suppresses_per_record_fsync_until_flush() {
        let log = make_log();

        let deferral = log.defer_sync();
        for i in 0..5 {
            log.append(&format!("ext_{i}"), &small_doc()).unwrap();
        }
        assert!(
            log.wal_is_dirty(),
            "deferred appends must stay unsynced until the batch-end flush"
        );
        assert_eq!(
            log.wal_sync_count(),
            0,
            "no per-record fsync may run inside the deferral scope"
        );

        drop(deferral);
        log.flush_wal().unwrap();
        assert!(!log.wal_is_dirty());
        assert_eq!(
            log.wal_sync_count(),
            1,
            "the whole batch must amortize to exactly one fsync"
        );
        assert_eq!(log.read_all().unwrap().len(), 5);
    }

    /// #551: dropping the deferral guard restores the per-record contract —
    /// the next append self-syncs (including any bytes left over from an
    /// abandoned scope).
    #[test]
    fn defer_sync_drop_restores_per_record_fsync() {
        let log = make_log();

        {
            let _deferral = log.defer_sync();
            log.append("ext_0", &small_doc()).unwrap();
            assert!(log.wal_is_dirty());
            // Abandoned scope: no flush_wal before drop.
        }

        log.append("ext_1", &small_doc()).unwrap();
        assert!(
            !log.wal_is_dirty(),
            "the first append after the scope must sync itself and the leftover bytes"
        );
        assert_eq!(log.read_all().unwrap().len(), 2);
    }

    /// #551: a deferral scope does not suppress the `Group` policy's batch
    /// thresholds — its bounded loss window holds even mid-batch.
    #[test]
    fn defer_sync_keeps_group_thresholds_firing() {
        let log = make_group_log(2, usize::MAX);

        let _deferral = log.defer_sync();
        log.append("ext_0", &small_doc()).unwrap();
        assert!(
            log.wal_is_dirty(),
            "below the record threshold the group batch stays unsynced"
        );
        log.append("ext_1", &small_doc()).unwrap();
        assert!(
            !log.wal_is_dirty(),
            "the group record threshold must fire despite the deferral scope"
        );
        assert_eq!(log.wal_sync_count(), 1);
    }

    /// The default policy is per-record, and `group_with_defaults` uses the
    /// documented batch thresholds with no flush timer (Issue #542, Phase 4).
    #[test]
    fn sync_policy_defaults() {
        assert_eq!(WalSyncPolicy::default(), WalSyncPolicy::PerRecord);
        assert_eq!(
            WalSyncPolicy::group_with_defaults(),
            WalSyncPolicy::Group {
                max_records: DEFAULT_GROUP_MAX_RECORDS,
                max_bytes: DEFAULT_GROUP_MAX_BYTES,
                max_interval: None,
            }
        );
        assert_eq!(WalSyncPolicy::group_with_defaults().flush_interval(), None);
        assert_eq!(
            WalSyncPolicy::group_with_interval(Duration::from_millis(50)).flush_interval(),
            Some(Duration::from_millis(50))
        );
    }

    /// Under `Group`, appends below the record threshold buffer their bytes
    /// without fsyncing; the append that reaches the threshold flushes the whole
    /// batch and resets the counters, and every record round-trips (Issue #542,
    /// Phase 4).
    #[test]
    fn group_policy_defers_then_flushes_at_record_threshold() {
        // Flush every 3 records; effectively no byte limit.
        let log = make_group_log(3, usize::MAX);

        log.append("e1", &small_doc()).unwrap();
        log.append("e2", &small_doc()).unwrap();
        {
            let guard = log.wal_writer.lock();
            let state = guard.as_ref().unwrap();
            assert!(state.dirty, "appends under the threshold stay unsynced");
            assert_eq!(state.unsynced_records, 2);
            assert!(state.unsynced_bytes > 0);
        }

        log.append("e3", &small_doc()).unwrap();
        {
            let guard = log.wal_writer.lock();
            let state = guard.as_ref().unwrap();
            assert!(
                !state.dirty,
                "reaching the record threshold flushes the batch"
            );
            assert_eq!(state.unsynced_records, 0);
            assert_eq!(state.unsynced_bytes, 0);
        }

        assert_eq!(log.read_all().unwrap().len(), 3);
    }

    /// Under `Group`, a single record that exceeds the byte threshold flushes
    /// immediately even when the record-count limit is far from reached (Issue
    /// #542, Phase 4).
    #[test]
    fn group_policy_flushes_at_byte_threshold() {
        // 1-byte budget so any record trips the byte threshold; no count limit.
        let log = make_group_log(usize::MAX, 1);

        log.append("e1", &small_doc()).unwrap();
        let guard = log.wal_writer.lock();
        let state = guard.as_ref().unwrap();
        assert!(
            !state.dirty,
            "a record past the byte threshold flushes immediately"
        );
        assert_eq!(state.unsynced_bytes, 0);
    }

    /// Under `Group`, a trailing partial batch (below both thresholds) is left
    /// unsynced until `flush_wal` — the commit-time barrier — makes it durable
    /// and round-trips it (Issue #542, Phase 4).
    #[test]
    fn group_partial_batch_is_made_durable_by_flush_wal() {
        let log = make_group_log(1000, usize::MAX);

        log.append("e1", &small_doc()).unwrap();
        log.append("e2", &small_doc()).unwrap();
        assert!(
            log.wal_writer.lock().as_ref().unwrap().dirty,
            "a partial batch stays unsynced"
        );

        log.flush_wal().unwrap();
        {
            let guard = log.wal_writer.lock();
            let state = guard.as_ref().unwrap();
            assert!(!state.dirty, "flush_wal forces the partial batch durable");
            assert_eq!(state.unsynced_records, 0);
            assert_eq!(state.unsynced_bytes, 0);
        }

        assert_eq!(log.read_all().unwrap().len(), 2);
    }

    #[test]
    fn test_doc_id_recovery() {
        let wal_storage = make_storage();
        let doc_storage = make_storage();

        // Write some entries.
        {
            let log =
                DocumentLog::new(wal_storage.clone(), "test.log", doc_storage.clone()).unwrap();
            let doc = Document::builder()
                .add_field("body", DataValue::Text("hello".to_string()))
                .build();
            log.append("ext_1", &doc).unwrap();
            log.append("ext_2", &doc).unwrap();
        }

        // Reopen and verify counters are restored.
        {
            let log =
                DocumentLog::new(wal_storage.clone(), "test.log", doc_storage.clone()).unwrap();
            let records = log.read_all().unwrap();
            assert_eq!(records.len(), 2);
            assert_eq!(log.next_doc_id(), 3); // max doc_id was 2

            let doc = Document::builder()
                .add_field("body", DataValue::Text("world".to_string()))
                .build();
            let (doc_id, seq) = log.append("ext_3", &doc).unwrap();
            assert_eq!(doc_id, 3);
            assert_eq!(seq, 3);
        }
    }

    /// Write one framed record manually, returning its `[u32 len][body]` bytes.
    fn frame(body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + body.len());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        out
    }

    /// A trailing record with an intact length prefix but a corrupt body (e.g. a
    /// crash mid-append) must NOT abort recovery: `read_all` recovers the valid
    /// prefix, drops the torn record, and resyncs counters from the prefix only
    /// (Issue #542, Phase 0).
    #[test]
    fn read_all_recovers_prefix_when_trailing_record_body_is_corrupt() {
        use std::io::Write as _;

        let wal_storage = make_storage();
        let doc_storage = make_storage();

        // One valid record followed by a frame whose length prefix is intact but
        // whose body is not valid JSON.
        let valid = LogRecord {
            seq: 1,
            entry: LogEntry::Upsert {
                doc_id: 1,
                external_id: "ext_1".to_string(),
                document: Document::builder()
                    .add_field("body", DataValue::Text("hello".to_string()))
                    .build(),
            },
        };
        {
            let mut out = wal_storage.create_output("test.log").unwrap();
            out.write_all(&frame(&serde_json::to_vec(&valid).unwrap()))
                .unwrap();
            out.write_all(&frame(b"this is not valid json")).unwrap();
            out.flush_and_sync().unwrap();
            out.close().unwrap();
        }

        let log = DocumentLog::new(wal_storage, "test.log", doc_storage).unwrap();

        // Recovery must succeed (not error) and yield only the valid prefix.
        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 1, "only the valid prefix is recovered");
        assert_eq!(records[0].seq, 1);

        // Counters are resynced from the recovered prefix only: the next append
        // continues monotonically without a gap.
        assert_eq!(log.last_seq(), 1);
        assert_eq!(log.next_doc_id(), 2);
        let doc = Document::builder()
            .add_field("body", DataValue::Text("next".to_string()))
            .build();
        let (doc_id, seq) = log.append("ext_2", &doc).unwrap();
        assert_eq!(doc_id, 2);
        assert_eq!(seq, 2);
    }

    /// An empty/garbage-only WAL (no valid leading record) recovers nothing
    /// rather than erroring.
    #[test]
    fn read_all_recovers_nothing_when_first_record_is_corrupt() {
        use std::io::Write as _;

        let wal_storage = make_storage();
        let doc_storage = make_storage();
        {
            let mut out = wal_storage.create_output("test.log").unwrap();
            out.write_all(&frame(b"garbage")).unwrap();
            out.flush_and_sync().unwrap();
            out.close().unwrap();
        }

        let log = DocumentLog::new(wal_storage, "test.log", doc_storage).unwrap();
        let records = log.read_all().unwrap();
        assert!(records.is_empty(), "no valid prefix to recover");
    }

    /// A fresh WAL is written in v3 framing: the file starts with the magic +
    /// version (3) header and records round-trip through `read_all` with CRC
    /// verification (Issue #542, Phase 1; #822 binary payload).
    #[test]
    fn wal_v3_fresh_file_has_header_and_round_trips() {
        use std::io::Read as _;

        let wal_storage = make_storage();
        let doc_storage = make_storage();
        let log = DocumentLog::new(wal_storage.clone(), "test.log", doc_storage).unwrap();

        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello".to_string()))
            .build();
        log.append("ext_1", &doc).unwrap();
        log.append_delete(1, "ext_1").unwrap();

        // The file begins with the v2 header.
        let mut header = [0u8; WAL_HEADER_LEN as usize];
        wal_storage
            .open_input("test.log")
            .unwrap()
            .read_exact(&mut header)
            .unwrap();
        assert_eq!(&header[0..4], WAL_MAGIC);
        assert_eq!(header[4], WAL_VERSION);

        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[1].seq, 2);
    }

    /// A v3 record whose payload is corrupted fails CRC verification and is
    /// dropped as a torn tail, recovering the valid prefix (Issue #542, Phase 1;
    /// #822 binary payload).
    #[test]
    fn wal_v3_crc_mismatch_recovers_prefix() {
        use std::io::{Read as _, Write as _};

        let wal_storage = make_storage();
        let doc_storage = make_storage();
        {
            let log =
                DocumentLog::new(wal_storage.clone(), "test.log", doc_storage.clone()).unwrap();
            let doc = Document::builder()
                .add_field("body", DataValue::Text("first".to_string()))
                .build();
            log.append("ext_1", &doc).unwrap();
            log.append("ext_2", &doc).unwrap();
        }

        // Flip the last byte (inside the 2nd record's payload) so its CRC fails.
        let mut bytes = Vec::new();
        wal_storage
            .open_input("test.log")
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        {
            let mut out = wal_storage.create_output("test.log").unwrap();
            out.write_all(&bytes).unwrap();
            out.flush_and_sync().unwrap();
            out.close().unwrap();
        }

        let log = DocumentLog::new(wal_storage, "test.log", doc_storage).unwrap();
        let records = log.read_all().unwrap();
        assert_eq!(
            records.len(),
            1,
            "CRC mismatch drops the corrupt 2nd record"
        );
        assert_eq!(records[0].seq, 1);
    }

    /// A legacy (pre-#542, header-less) WAL still recovers, and subsequent
    /// appends continue in legacy framing rather than mixing formats within the
    /// same file (Issue #542, Phase 1).
    #[test]
    fn legacy_wal_recovers_and_appends_stay_legacy() {
        use std::io::{Read as _, Write as _};

        let wal_storage = make_storage();
        let doc_storage = make_storage();

        // Write one legacy-framed record (no header, no CRC) directly.
        let rec = LogRecord {
            seq: 1,
            entry: LogEntry::Upsert {
                doc_id: 1,
                external_id: "ext_1".to_string(),
                document: Document::builder()
                    .add_field("body", DataValue::Text("legacy".to_string()))
                    .build(),
            },
        };
        {
            let mut out = wal_storage.create_output("test.log").unwrap();
            out.write_all(&frame(&serde_json::to_vec(&rec).unwrap()))
                .unwrap();
            out.flush_and_sync().unwrap();
            out.close().unwrap();
        }

        let log = DocumentLog::new(wal_storage.clone(), "test.log", doc_storage).unwrap();
        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 1, "legacy record recovers");

        // A new append continues the file in legacy framing.
        let doc = Document::builder()
            .add_field("body", DataValue::Text("more".to_string()))
            .build();
        log.append("ext_2", &doc).unwrap();

        // The file still has no v2 header (it was not switched mid-file).
        let mut magic = [0u8; 4];
        wal_storage
            .open_input("test.log")
            .unwrap()
            .read_exact(&mut magic)
            .unwrap();
        assert_ne!(
            &magic, WAL_MAGIC,
            "appends must not switch a legacy file to v2 mid-file"
        );

        // Both records read back through the legacy path.
        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 2);
    }

    /// Build a CRC-framed v2 frame (`[u32 len][u32 crc32][payload]`) for a JSON
    /// payload — the framing produced by pre-#822 writers.
    fn frame_v2(body: &[u8]) -> Vec<u8> {
        let len_bytes = (body.len() as u32).to_le_bytes();
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&len_bytes);
        hasher.update(body);
        let crc = hasher.finalize().to_le_bytes();
        let mut out = Vec::with_capacity(4 + 4 + body.len());
        out.extend_from_slice(&len_bytes);
        out.extend_from_slice(&crc);
        out.extend_from_slice(body);
        out
    }

    /// Write a complete v2 (CRC-framed JSON) WAL file with the given records.
    fn write_v2_file(storage: &Arc<dyn Storage>, path: &str, records: &[LogRecord]) {
        use std::io::Write as _;
        let mut out = storage.create_output(path).unwrap();
        out.write_all(WAL_MAGIC).unwrap();
        out.write_all(&[WAL_VERSION_V2]).unwrap();
        for rec in records {
            out.write_all(&frame_v2(&serde_json::to_vec(rec).unwrap()))
                .unwrap();
        }
        out.flush_and_sync().unwrap();
        out.close().unwrap();
    }

    fn sample_record(seq: u64, doc_id: u64) -> LogRecord {
        LogRecord {
            seq,
            entry: LogEntry::Upsert {
                doc_id,
                external_id: format!("ext_{doc_id}"),
                document: Document::builder()
                    .add_field("title", DataValue::Text("hello world".to_string()))
                    .add_field("score", DataValue::Float64(1.5))
                    .add_field("embedding", DataValue::Vector(vec![0.25; 64]))
                    .build(),
            },
        }
    }

    /// A pre-#822 v2 WAL (CRC-framed JSON payloads) still recovers fully through
    /// the back-compat reader (#822 acceptance: back-compat for JSON WALs).
    #[test]
    fn wal_v2_json_payload_recovers() {
        let wal_storage = make_storage();
        let doc_storage = make_storage();
        let records = [sample_record(1, 1), sample_record(2, 2)];
        write_v2_file(&wal_storage, "test.log", &records);

        let log = DocumentLog::new(wal_storage, "test.log", doc_storage).unwrap();
        let recovered = log.read_all().unwrap();
        assert_eq!(recovered.len(), 2, "both v2 JSON records recover");
        assert_eq!(recovered[0].seq, 1);
        assert_eq!(recovered[1].seq, 2);
        // The vector field round-trips through the JSON reader unchanged.
        if let LogEntry::Upsert { document, .. } = &recovered[0].entry {
            assert_eq!(
                document.fields.get("embedding"),
                Some(&DataValue::Vector(vec![0.25; 64]))
            );
        } else {
            panic!("expected an Upsert");
        }
    }

    /// v3 (rkyv binary) records round-trip exactly through `read_all`, including
    /// a vector field, across both Upsert and Delete entries (#822).
    #[test]
    fn wal_v3_binary_round_trips_all_value_types() {
        let wal_storage = make_storage();
        let doc_storage = make_storage();
        let log = DocumentLog::new(wal_storage, "test.log", doc_storage).unwrap();

        let doc = Document::builder()
            .add_field("title", DataValue::Text("hello".to_string()))
            .add_field("count", DataValue::Int64(-7))
            .add_field("score", DataValue::Float64(2.5))
            .add_field("flag", DataValue::Bool(true))
            .add_field("embedding", DataValue::Vector(vec![0.1, 0.2, 0.3, 0.4]))
            .add_field("tags", DataValue::Int64Array(vec![1, 2, 3]))
            .build();
        log.append("ext_1", &doc).unwrap();
        log.append_delete(1, "ext_1").unwrap();

        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 2);
        match &records[0].entry {
            LogEntry::Upsert {
                doc_id,
                external_id,
                document,
            } => {
                assert_eq!(*doc_id, 1);
                assert_eq!(external_id, "ext_1");
                assert_eq!(document.fields, doc.fields, "all value types round-trip");
            }
            _ => panic!("expected Upsert first"),
        }
        match &records[1].entry {
            LogEntry::Delete {
                doc_id,
                external_id,
            } => {
                assert_eq!(*doc_id, 1);
                assert_eq!(external_id, "ext_1");
            }
            _ => panic!("expected Delete second"),
        }
    }

    /// Upgrade path: a v2 file recovers, then a `truncate` recreates the file in
    /// v3 framing and subsequent appends round-trip — old and new formats are
    /// never mixed within one file (#822).
    #[test]
    fn wal_upgrade_v2_to_v3_after_truncate() {
        use std::io::Read as _;

        let wal_storage = make_storage();
        let doc_storage = make_storage();
        write_v2_file(&wal_storage, "test.log", &[sample_record(1, 1)]);

        let log = DocumentLog::new(wal_storage.clone(), "test.log", doc_storage).unwrap();
        assert_eq!(log.read_all().unwrap().len(), 1, "v2 record recovers");

        // Truncate recreates the file; the next append stamps a v3 header.
        log.truncate().unwrap();
        let doc = Document::builder()
            .add_field("embedding", DataValue::Vector(vec![0.5; 32]))
            .build();
        log.append("ext_2", &doc).unwrap();

        let mut header = [0u8; WAL_HEADER_LEN as usize];
        wal_storage
            .open_input("test.log")
            .unwrap()
            .read_exact(&mut header)
            .unwrap();
        assert_eq!(&header[0..4], WAL_MAGIC);
        assert_eq!(header[4], WAL_VERSION, "recreated file is v3");

        let records = log.read_all().unwrap();
        assert_eq!(records.len(), 1, "post-upgrade v3 record recovers");
        assert_eq!(records[0].seq, 2);
    }

    /// Measurement (#822 acceptance): for a vector-heavy record the v3 rkyv
    /// payload is materially smaller than the v2 JSON payload, because each
    /// `f32` is 4 raw bytes instead of a decimal string. Prints the ratio for
    /// the implementation report (`cargo test -- --nocapture`).
    #[test]
    fn wal_v3_payload_smaller_than_v2_for_vectors() {
        // A 384-dim embedding — the dominant payload in a vector workload.
        let record = LogRecord {
            seq: 1,
            entry: LogEntry::Upsert {
                doc_id: 1,
                external_id: "doc-1".to_string(),
                document: Document::builder()
                    .add_field(
                        "title",
                        DataValue::Text("a representative title".to_string()),
                    )
                    .add_field("embedding", DataValue::Vector(vec![0.123_456_7; 384]))
                    .build(),
            },
        };

        let json = DocumentLog::encode_payload(&record, WalFormat::V2).unwrap();
        let rkyv = DocumentLog::encode_payload(&record, WalFormat::V3).unwrap();

        println!(
            "WAL payload size (384-dim vector record): v2 JSON = {} B, v3 rkyv = {} B, ratio = {:.2}x",
            json.len(),
            rkyv.len(),
            json.len() as f64 / rkyv.len() as f64,
        );
        assert!(
            rkyv.len() < json.len(),
            "v3 binary payload ({} B) must be smaller than v2 JSON ({} B)",
            rkyv.len(),
            json.len()
        );
        // The raw vector alone is 384 * 4 = 1536 B; the binary record should be
        // close to that, far below JSON's decimal-string encoding.
        assert!(
            rkyv.len() < json.len() / 2,
            "expected a large reduction for vector-heavy records: v2 {} B vs v3 {} B",
            json.len(),
            rkyv.len()
        );
    }

    /// Measurement (#822 acceptance): replay time of a v3 (rkyv) WAL versus a
    /// v2 (JSON) WAL of the same vector-heavy records. Print-only — wall-clock is
    /// environment-dependent, so this asserts nothing and just reports the parse
    /// speedup for the implementation report (`cargo test -- --nocapture`).
    #[test]
    fn wal_v3_replay_time_vs_v2() {
        use std::time::Instant;

        const N: u64 = 2000;
        let records: Vec<LogRecord> = (1..=N).map(|i| sample_record(i, i)).collect();

        // v2 (JSON) file.
        let v2_storage = make_storage();
        let v2_doc = make_storage();
        write_v2_file(&v2_storage, "test.log", &records);
        let v2_log = DocumentLog::new(v2_storage, "test.log", v2_doc).unwrap();

        // v3 (rkyv) file, produced by the current writer.
        let v3_storage = make_storage();
        let v3_doc = make_storage();
        let v3_log = DocumentLog::new(v3_storage, "test.log", v3_doc).unwrap();
        for rec in &records {
            if let LogEntry::Upsert {
                external_id,
                document,
                ..
            } = &rec.entry
            {
                v3_log.append(external_id, document).unwrap();
            }
        }

        let t0 = Instant::now();
        let v2_recovered = v2_log.read_all().unwrap();
        let v2_elapsed = t0.elapsed();

        let t1 = Instant::now();
        let v3_recovered = v3_log.read_all().unwrap();
        let v3_elapsed = t1.elapsed();

        assert_eq!(v2_recovered.len() as u64, N);
        assert_eq!(v3_recovered.len() as u64, N);
        println!(
            "WAL replay ({N} vector records): v2 JSON = {:?}, v3 rkyv = {:?}, speedup = {:.2}x",
            v2_elapsed,
            v3_elapsed,
            v2_elapsed.as_secs_f64() / v3_elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
        );
    }

    #[test]
    fn test_set_next_doc_id() {
        let log = make_log();

        // Sync with a higher doc_id from document store.
        log.set_next_doc_id(100);
        assert_eq!(log.next_doc_id(), 100);

        // Setting a lower value should be ignored.
        log.set_next_doc_id(50);
        assert_eq!(log.next_doc_id(), 100);

        // Append should use the higher value.
        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello".to_string()))
            .build();
        let (doc_id, _) = log.append("ext_1", &doc).unwrap();
        assert_eq!(doc_id, 100);
    }

    #[test]
    fn test_store_and_get_document() {
        let log = make_log();

        let doc = Document::builder()
            .add_field("body", DataValue::Text("hello world".to_string()))
            .build();

        // Store document.
        log.store_document(1, doc.clone());

        // Retrieve from pending.
        let retrieved = log.get_document(1).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().fields.get("body"),
            doc.fields.get("body")
        );

        // After commit, retrieve from segment.
        log.commit_documents().unwrap();
        let retrieved = log.get_document(1).unwrap();
        assert!(retrieved.is_some());
    }

    /// Append `external_id` and return the seq the WAL assigned it.
    fn append_named(log: &DocumentLog, external_id: &str) -> SeqNumber {
        log.append(external_id, &small_doc()).unwrap().1
    }

    /// The `external_id`s still present in the WAL, in seq order.
    fn surviving_ids(log: &DocumentLog) -> Vec<String> {
        log.read_all()
            .unwrap()
            .into_iter()
            .map(|r| match r.entry {
                LogEntry::Upsert { external_id, .. } => external_id,
                LogEntry::Delete { external_id, .. } => external_id,
            })
            .collect()
    }

    /// #876: `truncate_retaining_after` keeps only records with `seq` strictly
    /// greater than the retain point — this is the primitive the commit ladder
    /// relies on to preserve a mutation that raced the commit instead of losing
    /// it to a whole-file truncate.
    #[test]
    fn truncate_retaining_after_keeps_only_later_records() {
        let log = make_log();
        append_named(&log, "a"); // seq 1
        let retain_from = append_named(&log, "b"); // seq 2 — the commit's snapshot
        append_named(&log, "c"); // seq 3 — raced the commit
        append_named(&log, "d"); // seq 4 — also raced

        log.truncate_retaining_after(retain_from).unwrap();

        assert_eq!(
            surviving_ids(&log),
            vec!["c".to_string(), "d".to_string()],
            "only records appended after the retain point must survive"
        );
    }

    /// #876: when nothing was appended past the retain point, the result must
    /// be indistinguishable from the historical whole-file truncate — the
    /// no-concurrency common case must not pay for the read-back/rewrite path.
    #[test]
    fn truncate_retaining_after_is_a_full_truncate_when_nothing_raced() {
        let log = make_log();
        append_named(&log, "a");
        let retain_from = append_named(&log, "b");

        log.truncate_retaining_after(retain_from).unwrap();

        assert!(
            surviving_ids(&log).is_empty(),
            "no record raced the retain point, so none should survive"
        );
        // The file itself is empty (the fast path recreates it, same as `truncate`).
        assert_eq!(
            log.wal_storage.file_size(&log.wal_path).unwrap(),
            0,
            "the fast path must leave a zero-byte file, matching `truncate()`"
        );
    }

    /// #876: `truncate()` (used by every caller that has nothing to retain,
    /// e.g. tests exercising the historical behavior) is unaffected — it is
    /// `truncate_retaining_after(SeqNumber::MAX)`, so it always takes the fast
    /// path and wipes everything.
    #[test]
    fn truncate_still_wipes_everything() {
        let log = make_log();
        append_named(&log, "a");
        append_named(&log, "b");

        log.truncate().unwrap();

        assert!(surviving_ids(&log).is_empty());
    }

    /// #876: a delete record appended after the retain point must survive the
    /// truncate exactly like an upsert — the retain filter is entry-agnostic.
    #[test]
    fn truncate_retaining_after_preserves_a_racing_delete() {
        let log = make_log();
        let retain_from = append_named(&log, "a");
        log.append_delete(1, "b").unwrap();

        log.truncate_retaining_after(retain_from).unwrap();

        assert_eq!(surviving_ids(&log), vec!["b".to_string()]);
    }

    /// #876: after a partial truncate, the WAL is still a well-formed file that
    /// further appends and a fresh `read_all` can build on — the rewritten tail
    /// must stamp a valid header, not just raw frames.
    #[test]
    fn truncate_retaining_after_leaves_a_well_formed_wal_for_further_appends() {
        let log = make_log();
        let retain_from = append_named(&log, "a");
        append_named(&log, "b");

        log.truncate_retaining_after(retain_from).unwrap();
        append_named(&log, "c");

        assert_eq!(
            surviving_ids(&log),
            vec!["b".to_string(), "c".to_string()],
            "a further append after the partial truncate must be readable \
             alongside the retained tail"
        );
    }

    /// #1010: the batch fetch must agree with N individual fetches,
    /// including the newest-wins resolution when the same doc id exists
    /// in several committed segments (`get_document` scans segments in
    /// reverse and takes the first hit; the batch path scans forward and
    /// overwrites — both must land on the newest copy).
    #[test]
    fn get_documents_batch_matches_individual_gets() {
        let log = make_log();

        let body = |t: &str| {
            Document::builder()
                .add_field("body", DataValue::Text(t.to_string()))
                .build()
        };

        // Segment 1: docs 1 and 2.
        log.store_document(1, body("one-old"));
        log.store_document(2, body("two"));
        log.commit_documents().unwrap();
        // Segment 2: doc 1 superseded, doc 3 added.
        log.store_document(1, body("one-new"));
        log.store_document(3, body("three"));
        log.commit_documents().unwrap();

        let ids = [1u64, 2, 3, 999];
        let batch = log.get_documents_batch(&ids).unwrap();

        for id in ids {
            let single = log.get_document(id).unwrap();
            assert_eq!(
                batch.get(&id).map(|d| d.fields.get("body")),
                single.as_ref().map(|d| d.fields.get("body")),
                "batch and single fetch must agree for doc {id}"
            );
        }
        assert_eq!(
            batch.get(&1).unwrap().fields.get("body"),
            Some(&DataValue::Text("one-new".to_string())),
            "the newest copy must win across segments"
        );
        assert!(!batch.contains_key(&999), "missing ids must be absent");
    }

    /// #1010: the batch path now populates and reads the document cache.
    /// A doc superseded after being cached must still resolve to its new
    /// content — i.e. the cache stays coherent.
    #[test]
    fn get_documents_batch_stays_coherent_after_update() {
        let log = make_log();

        let body = |t: &str| {
            Document::builder()
                .add_field("body", DataValue::Text(t.to_string()))
                .build()
        };

        log.store_document(1, body("before"));
        log.commit_documents().unwrap();

        // Warm the cache through the batch path.
        let warm = log.get_documents_batch(&[1]).unwrap();
        assert_eq!(
            warm.get(&1).unwrap().fields.get("body"),
            Some(&DataValue::Text("before".to_string()))
        );

        // Supersede it and re-fetch: the stale cached copy must not win.
        log.store_document(1, body("after"));
        log.commit_documents().unwrap();

        let refetched = log.get_documents_batch(&[1]).unwrap();
        assert_eq!(
            refetched.get(&1).unwrap().fields.get("body"),
            Some(&DataValue::Text("after".to_string())),
            "the batch path must not serve a stale cached document"
        );
        // And the single-document path must agree.
        assert_eq!(
            log.get_document(1).unwrap().unwrap().fields.get("body"),
            Some(&DataValue::Text("after".to_string()))
        );
    }
}
