//! Ruby wrapper for the Laurus [`laurus::WalSyncPolicy`] value object.
//!
//! Exposes `Laurus::WalSyncPolicy` with the two construction class methods
//! `per_record` and `group`, mirroring the durability policy plumbed through
//! [`laurus::EngineBuilder::wal_sync_policy`]. The resulting value object is
//! passed to `Laurus::Index.new(wal_sync_policy: ...)`.

use std::time::Duration;

use laurus::{DEFAULT_GROUP_MAX_BYTES, DEFAULT_GROUP_MAX_RECORDS, WalSyncPolicy};
use magnus::prelude::*;
use magnus::scan_args::{get_kwargs, scan_args};
use magnus::{Error, RHash, RModule, Ruby, Value};

// ---------------------------------------------------------------------------
// WalSyncPolicy
// ---------------------------------------------------------------------------

/// Ruby value object wrapping a [`laurus::WalSyncPolicy`].
///
/// A WAL (Write-Ahead Log) sync policy controls when buffered WAL appends are
/// fsync'd to durable storage:
///
/// * `Laurus::WalSyncPolicy.per_record` — the default. Every `put_document` /
///   `add_document` / `delete_documents` append is fsync'd before the call
///   returns. Highest durability (no data loss on crash), lowest throughput.
/// * `Laurus::WalSyncPolicy.group(...)` — group commit. Appends are buffered
///   and fsync'd once a record-count, byte, or time threshold is reached,
///   trading a small crash-loss window for much higher write throughput.
///
/// # Examples
///
/// ```ruby
/// require "laurus"
///
/// # Default per-record durability.
/// policy = Laurus::WalSyncPolicy.per_record
///
/// # Group commit with the built-in defaults (1024 records / 1 MiB).
/// policy = Laurus::WalSyncPolicy.group
///
/// # Group commit with custom thresholds and a 1 s flush timer.
/// policy = Laurus::WalSyncPolicy.group(
///   max_records: 256,
///   max_bytes: 4096,
///   max_interval_ms: 1000
/// )
///
/// index = Laurus::Index.new(wal_sync_policy: policy)
/// ```
#[magnus::wrap(class = "Laurus::WalSyncPolicy")]
pub struct RbWalSyncPolicy {
    /// The wrapped core policy, forwarded verbatim to the engine builder.
    pub inner: WalSyncPolicy,
}

impl RbWalSyncPolicy {
    /// Build the per-record durability policy
    /// ([`laurus::WalSyncPolicy::PerRecord`]).
    ///
    /// Every WAL append is fsync'd before the originating write returns, so no
    /// committed-before-crash data can be lost. This is the engine default and
    /// matches the behavior when `wal_sync_policy:` is omitted from
    /// `Laurus::Index.new`.
    ///
    /// # Returns
    ///
    /// A `Laurus::WalSyncPolicy` wrapping
    /// [`laurus::WalSyncPolicy::PerRecord`].
    fn per_record() -> Self {
        Self {
            inner: WalSyncPolicy::PerRecord,
        }
    }

    /// Build the group-commit durability policy
    /// ([`laurus::WalSyncPolicy::Group`]).
    ///
    /// Group commit buffers WAL appends and defers the fsync until a threshold
    /// is reached, batching the durability cost across many writes for higher
    /// throughput at the price of a bounded crash-loss window.
    ///
    /// # Arguments
    ///
    /// * `args` - Keyword arguments (all optional):
    ///   - `max_records:` (Integer): Flush after this many buffered records.
    ///     Defaults to [`laurus::DEFAULT_GROUP_MAX_RECORDS`] (1024).
    ///   - `max_bytes:` (Integer): Flush after this many buffered bytes.
    ///     Defaults to [`laurus::DEFAULT_GROUP_MAX_BYTES`] (1 MiB).
    ///   - `max_interval_ms:` (Integer): Maximum time, in milliseconds, that an
    ///     append may stay un-fsync'd before a background timer flushes it.
    ///     When omitted, no time-based flush timer is installed and the policy
    ///     relies solely on the record-count and byte thresholds. `0` installs
    ///     no timer either, but flushes on every append.
    ///
    /// Calling `group` with no arguments yields the default group-commit
    /// thresholds with no flush timer.
    ///
    /// # Returns
    ///
    /// A `Laurus::WalSyncPolicy` wrapping [`laurus::WalSyncPolicy::Group`], or
    /// an `ArgumentError` if an unexpected keyword is supplied.
    fn group(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(), (), (), (), RHash, ()>(args)?;
        let kwargs = get_kwargs::<_, (), (Option<usize>, Option<usize>, Option<u64>), ()>(
            args.keywords,
            &[],
            &["max_records", "max_bytes", "max_interval_ms"],
        )?;
        let (max_records, max_bytes, max_interval_ms) = kwargs.optional;
        Ok(Self {
            inner: WalSyncPolicy::Group {
                max_records: max_records.unwrap_or(DEFAULT_GROUP_MAX_RECORDS),
                max_bytes: max_bytes.unwrap_or(DEFAULT_GROUP_MAX_BYTES),
                max_interval: max_interval_ms.map(Duration::from_millis),
            },
        })
    }

    /// Human-readable representation for `inspect` / `to_s`.
    fn inspect(&self) -> String {
        match self.inner {
            WalSyncPolicy::PerRecord => "WalSyncPolicy(PerRecord)".to_string(),
            WalSyncPolicy::Group {
                max_records,
                max_bytes,
                max_interval,
            } => format!(
                "WalSyncPolicy(Group max_records={max_records} max_bytes={max_bytes} max_interval={max_interval:?})"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Class registration
// ---------------------------------------------------------------------------

/// Register the `Laurus::WalSyncPolicy` class and its methods.
///
/// # Arguments
///
/// * `ruby` - Ruby interpreter handle.
/// * `module` - The `Laurus` module.
pub fn define(ruby: &Ruby, module: &RModule) -> Result<(), Error> {
    let class = module.define_class("WalSyncPolicy", ruby.class_object())?;
    class.define_singleton_method(
        "per_record",
        magnus::function!(RbWalSyncPolicy::per_record, 0),
    )?;
    class.define_singleton_method("group", magnus::function!(RbWalSyncPolicy::group, -1))?;
    class.define_method("inspect", magnus::method!(RbWalSyncPolicy::inspect, 0))?;
    class.define_method("to_s", magnus::method!(RbWalSyncPolicy::inspect, 0))?;
    Ok(())
}
