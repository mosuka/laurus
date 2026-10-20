//! WASM wrapper for the Laurus [`WalSyncPolicy`] value object (Issue #820).
//!
//! Exposes the write-ahead-log durability policy to JavaScript so that callers
//! can opt into group-commit batching when constructing an [`crate::index::WasmIndex`].

use std::time::Duration;

use laurus::{DEFAULT_GROUP_MAX_BYTES, DEFAULT_GROUP_MAX_RECORDS, WalSyncPolicy};
use wasm_bindgen::prelude::*;

/// Write-ahead-log (WAL) durability policy for an [`crate::index::WasmIndex`].
///
/// Controls when WAL appends are made durable (fsync'd):
///
/// - **Per-record** (the default) — every `putDocument` / `addDocument` /
///   `deleteDocuments` fsyncs the WAL before returning, so a successful write
///   can never be lost to a crash.
/// - **Group commit** — fsyncs are deferred and batched until a record-count or
///   byte threshold is reached, yielding much higher ingest throughput at the
///   cost of losing up to the last unsynced batch on a crash. Use
///   `commit()` (a hard durability barrier) or `flushWal()` to bound the
///   crash-loss window on demand.
///
/// ```javascript
/// import { Index, Schema, WalSyncPolicy } from "laurus-wasm";
///
/// const schema = new Schema();
/// schema.addTextField("title");
///
/// // Group commit with default thresholds (1024 records / 1 MiB).
/// const policy = WalSyncPolicy.group();
///
/// // Or customize the thresholds.
/// const tuned = WalSyncPolicy.group(2048, 2 * 1024 * 1024, 50);
///
/// const index = await Index.create(schema, policy);
/// ```
///
/// **wasm note:** the `maxIntervalMs` background flush timer is a no-op under
/// WebAssembly (the core gates it on native targets only). The record-count and
/// byte thresholds still apply; call `commit()` or `flushWal()` to flush
/// explicitly.
#[wasm_bindgen(js_name = "WalSyncPolicy")]
#[derive(Clone, Copy)]
pub struct WasmWalSyncPolicy {
    /// The wrapped core durability policy passed to `EngineBuilder`.
    pub(crate) inner: WalSyncPolicy,
}

#[wasm_bindgen(js_class = "WalSyncPolicy")]
impl WasmWalSyncPolicy {
    /// The per-record durability policy: every WAL append is fsync'd before the
    /// write returns. This is the default and the safest policy.
    ///
    /// # Returns
    ///
    /// A `WalSyncPolicy` wrapping [`WalSyncPolicy::PerRecord`].
    #[wasm_bindgen(js_name = "perRecord")]
    pub fn per_record() -> WasmWalSyncPolicy {
        WasmWalSyncPolicy {
            inner: WalSyncPolicy::PerRecord,
        }
    }

    /// The group-commit durability policy: WAL fsyncs are deferred and batched
    /// until a threshold is reached.
    ///
    /// Each threshold falls back to the core default when omitted, so
    /// `WalSyncPolicy.group()` is equivalent to the documented defaults
    /// (1024 records / 1 MiB / no timer).
    ///
    /// # Arguments
    ///
    /// * `max_records` - Flush after this many buffered records. Defaults to
    ///   [`DEFAULT_GROUP_MAX_RECORDS`] (1024) when omitted.
    /// * `max_bytes` - Flush after this many buffered bytes. Defaults to
    ///   [`DEFAULT_GROUP_MAX_BYTES`] (1 MiB) when omitted.
    /// * `max_interval_ms` - Optional time-based flush interval in
    ///   milliseconds. A non-zero value is a **no-op under WebAssembly** (the
    ///   background timer is native-only in the core); the thresholds above
    ///   still apply. `0` needs no timer, so it applies here too and flushes
    ///   on every append. The value is truncated to whole milliseconds, so
    ///   anything below `1` (and a negative value or `NaN`) counts as `0`.
    ///
    /// # Returns
    ///
    /// A `WalSyncPolicy` wrapping [`WalSyncPolicy::Group`].
    #[wasm_bindgen(js_name = "group")]
    pub fn group(
        max_records: Option<u32>,
        max_bytes: Option<u32>,
        max_interval_ms: Option<f64>,
    ) -> WasmWalSyncPolicy {
        WasmWalSyncPolicy {
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

// The laurus core crate only compiles when its default `native` feature is on,
// which laurus-wasm disables (`default-features = false`). As a result the
// dependency cannot be built for the native host target, so these tests — like
// the rest of the crate — are exercised on `wasm32` via `wasm_bindgen_test`
// (run with `wasm-pack test --node`). The assertions are pure value-object
// checks with no JS or browser interaction.
#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod tests {
    use super::WasmWalSyncPolicy;
    use laurus::{DEFAULT_GROUP_MAX_BYTES, DEFAULT_GROUP_MAX_RECORDS, WalSyncPolicy};
    use std::time::Duration;
    use wasm_bindgen_test::wasm_bindgen_test;

    /// `perRecord()` must wrap [`WalSyncPolicy::PerRecord`].
    #[wasm_bindgen_test]
    fn per_record_wraps_per_record() {
        let policy = WasmWalSyncPolicy::per_record();
        assert!(matches!(policy.inner, WalSyncPolicy::PerRecord));
    }

    /// `group()` with no arguments must equal the core defaults.
    #[wasm_bindgen_test]
    fn group_no_args_uses_defaults() {
        let policy = WasmWalSyncPolicy::group(None, None, None);
        match policy.inner {
            WalSyncPolicy::Group {
                max_records,
                max_bytes,
                max_interval,
            } => {
                assert_eq!(max_records, DEFAULT_GROUP_MAX_RECORDS);
                assert_eq!(max_bytes, DEFAULT_GROUP_MAX_BYTES);
                assert_eq!(max_interval, None);
            }
            WalSyncPolicy::PerRecord => panic!("expected Group, got PerRecord"),
        }
    }

    /// Explicit arguments must be threaded through, including the interval as a
    /// [`Duration`] in milliseconds.
    #[wasm_bindgen_test]
    fn group_explicit_args_are_honored() {
        let policy = WasmWalSyncPolicy::group(Some(2048), Some(2 * 1024 * 1024), Some(50.0));
        match policy.inner {
            WalSyncPolicy::Group {
                max_records,
                max_bytes,
                max_interval,
            } => {
                assert_eq!(max_records, 2048);
                assert_eq!(max_bytes, 2 * 1024 * 1024);
                assert_eq!(max_interval, Some(Duration::from_millis(50)));
            }
            WalSyncPolicy::PerRecord => panic!("expected Group, got PerRecord"),
        }
    }

    /// A partially specified `group()` call must mix explicit values with
    /// defaults for the omitted thresholds.
    #[wasm_bindgen_test]
    fn group_partial_args_mix_with_defaults() {
        let policy = WasmWalSyncPolicy::group(Some(64), None, None);
        match policy.inner {
            WalSyncPolicy::Group {
                max_records,
                max_bytes,
                max_interval,
            } => {
                assert_eq!(max_records, 64);
                assert_eq!(max_bytes, DEFAULT_GROUP_MAX_BYTES);
                assert_eq!(max_interval, None);
            }
            WalSyncPolicy::PerRecord => panic!("expected Group, got PerRecord"),
        }
    }
}
