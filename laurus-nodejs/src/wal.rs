//! Node.js wrapper for the WAL (write-ahead log) durability policy.
//!
//! Exposes [`laurus::WalSyncPolicy`] to JavaScript as the `WalSyncPolicy`
//! value object, mirroring the same factory-method surface used by the other
//! language bindings.

use std::time::Duration;

use laurus::{DEFAULT_GROUP_MAX_BYTES, DEFAULT_GROUP_MAX_RECORDS, WalSyncPolicy};
use napi_derive::napi;

// ---------------------------------------------------------------------------
// WalSyncPolicy
// ---------------------------------------------------------------------------

/// WAL durability policy — controls when write-ahead-log appends are fsync'd.
///
/// This trades the durability of an individual write against ingest
/// throughput. `commit()` is always a hard durability barrier under either
/// policy (it forces the WAL durable before materializing the index), and
/// `Index.flushWal()` forces a flush on demand.
///
/// Construct one of the two variants with the static factory methods and pass
/// it to `Index.create(path, schema, policy)`.
///
/// ## Example
///
/// ```javascript
/// const { Index, Schema, WalSyncPolicy } = require("laurus-nodejs");
///
/// const schema = new Schema();
/// schema.addTextField("title");
///
/// // Group commit with default thresholds (higher ingest throughput, but a
/// // crash can lose the last unsynced batch — comparable to SQLite's
/// // synchronous = NORMAL).
/// const index = await Index.create("./idx", schema, WalSyncPolicy.group());
///
/// // Group commit with explicit thresholds and a 1s periodic flush timer.
/// const policy = WalSyncPolicy.group(256, 4096, 1000);
///
/// // Per-record durability (the default if no policy is supplied): every
/// // append is fsync'd before it returns.
/// const strict = WalSyncPolicy.perRecord();
/// ```
#[napi(js_name = "WalSyncPolicy")]
#[derive(Clone, Copy)]
pub struct JsWalSyncPolicy {
    /// The wrapped Laurus durability policy.
    pub(crate) inner: WalSyncPolicy,
}

#[napi]
impl JsWalSyncPolicy {
    /// Create a per-record durability policy.
    ///
    /// Every WAL append is fsync'd before it returns, so a successful
    /// `putDocument` / `addDocument` / `deleteDocuments` can never be lost to a
    /// crash. This is the default policy when no policy is passed to
    /// `Index.create`.
    ///
    /// # Returns
    ///
    /// A `WalSyncPolicy` representing `PerRecord`.
    #[napi(factory)]
    pub fn per_record() -> Self {
        Self {
            inner: WalSyncPolicy::PerRecord,
        }
    }

    /// Create a group-commit durability policy.
    ///
    /// The fsync is deferred and amortized over a batch: the WAL is flushed
    /// when **either** `maxRecords` records **or** `maxBytes` bytes have
    /// accumulated since the last sync (whichever comes first), unconditionally
    /// at `commit()`, and — when `maxIntervalMs` is supplied — at least once per
    /// interval via a background timer. An append may return before its record
    /// is durable, so a crash can lose up to the last unsynced batch.
    ///
    /// All arguments are optional; omitted thresholds fall back to the Laurus
    /// defaults. `WalSyncPolicy.group()` therefore yields group commit with the
    /// default thresholds and no timer, and passing only `maxIntervalMs` yields
    /// the defaults plus a periodic flush.
    ///
    /// # Arguments
    ///
    /// * `max_records` - Flush once this many records have accumulated since the
    ///   last sync. Defaults to `laurus::DEFAULT_GROUP_MAX_RECORDS` (1024).
    /// * `max_bytes` - Flush once this many appended bytes have accumulated
    ///   since the last sync. Defaults to `laurus::DEFAULT_GROUP_MAX_BYTES`
    ///   (1 MiB).
    /// * `max_interval_ms` - Optional periodic flush interval in milliseconds.
    ///   When omitted, no background timer runs and durability is left to the
    ///   thresholds, `commit()`, and `flushWal()`. `0` runs no timer but
    ///   flushes on every append; the value is truncated to whole
    ///   milliseconds, so anything below `1` (and a negative value or `NaN`)
    ///   counts as `0`. A non-zero interval is honored on native targets only.
    ///
    /// # Returns
    ///
    /// A `WalSyncPolicy` representing `Group`.
    #[napi(factory)]
    pub fn group(
        max_records: Option<u32>,
        max_bytes: Option<u32>,
        max_interval_ms: Option<f64>,
    ) -> Self {
        Self {
            inner: WalSyncPolicy::Group {
                max_records: max_records
                    .map(|v| v as usize)
                    .unwrap_or(DEFAULT_GROUP_MAX_RECORDS),
                max_bytes: max_bytes
                    .map(|v| v as usize)
                    .unwrap_or(DEFAULT_GROUP_MAX_BYTES),
                max_interval: max_interval_ms.map(|ms| Duration::from_millis(ms as u64)),
            },
        }
    }
}
