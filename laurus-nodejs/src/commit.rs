//! Node.js wrapper for the auto-commit policy.
//!
//! Exposes [`laurus::CommitPolicy`] to JavaScript as the `CommitPolicy` value
//! object, mirroring the same factory-method surface used by the other
//! language bindings and by [`crate::wal::JsWalSyncPolicy`].

use laurus::CommitPolicy;
use napi::Result;
use napi_derive::napi;

use crate::convert::unsigned_param;

// ---------------------------------------------------------------------------
// CommitPolicy
// ---------------------------------------------------------------------------

/// Auto-commit policy — controls when the engine automatically runs the commit
/// ladder during ingestion.
///
/// By default the engine commits only when `commit()` is called explicitly; a
/// non-`manual` policy makes it commit automatically at an ingestion-driven
/// cadence (one full ladder per auto-commit). This is orthogonal to
/// `WalSyncPolicy`.
///
/// Construct one of the variants with the static factory methods and pass it to
/// `Index.create(path, schema, walSyncPolicy, commitPolicy)`.
///
/// ## Example
///
/// ```javascript
/// const { Index, Schema, CommitPolicy } = require("laurus-nodejs");
///
/// const schema = new Schema();
/// schema.addTextField("title");
///
/// // Auto-commit after every 1000 applied documents.
/// const index = await Index.create("./idx", schema, undefined, CommitPolicy.everyDocs(1000));
///
/// // Manual (the default if no policy is supplied): the caller drives commit().
/// const manual = CommitPolicy.manual();
/// ```
#[napi(js_name = "CommitPolicy")]
#[derive(Clone, Copy)]
pub struct JsCommitPolicy {
    /// The wrapped Laurus auto-commit policy.
    pub(crate) inner: CommitPolicy,
}

#[napi]
impl JsCommitPolicy {
    /// Create a manual (no auto-commit) policy.
    ///
    /// The engine commits only when `Index.commit()` is called explicitly. This
    /// is the default policy when no policy is passed to `Index.create`.
    ///
    /// # Returns
    ///
    /// A `CommitPolicy` representing `Manual`.
    #[napi(factory)]
    pub fn manual() -> Self {
        Self {
            inner: CommitPolicy::Manual,
        }
    }

    /// Create an auto-commit-every-`n`-documents policy.
    ///
    /// The engine runs the commit ladder after every `n` applied documents,
    /// across the singular and batch ingest APIs (and every `n` documents
    /// within a single batch). `everyDocs(0)` disables auto-commit, which is
    /// equivalent to `CommitPolicy.manual()`.
    ///
    /// # Arguments
    ///
    /// * `n` - Commit after this many applied documents. `0` disables
    ///   auto-commit.
    ///
    /// # Returns
    ///
    /// A `CommitPolicy` representing `EveryDocs(n)`.
    ///
    /// # Errors
    ///
    /// Throws an `InvalidArg` error if `n` is not an integer from 0 to
    /// 4294967295 (Issue #1612).
    #[napi(factory)]
    pub fn every_docs(n: f64) -> Result<Self> {
        let n = unsigned_param("CommitPolicy.everyDocs", "n", n)?;
        Ok(Self {
            inner: CommitPolicy::EveryDocs(n as usize),
        })
    }

    /// Create an auto-commit-at-least-every-`ms`-milliseconds policy.
    ///
    /// A background timer runs the commit ladder at least once per `ms`
    /// milliseconds while ingestion is active — the time-based counterpart of
    /// `everyDocs`.
    ///
    /// **Native-only:** on `wasm32` targets the engine never starts the timer,
    /// so this policy is a documented no-op there (the value is still
    /// constructed).
    ///
    /// # Arguments
    ///
    /// * `ms` - Commit at least this often, in milliseconds. `0` disables
    ///   auto-commit.
    ///
    /// # Returns
    ///
    /// A `CommitPolicy` representing `Interval(Duration::from_millis(ms))`.
    ///
    /// # Errors
    ///
    /// Throws an `InvalidArg` error if `ms` is not an integer from 0 to
    /// 4294967295 (Issue #1612).
    #[napi(factory)]
    pub fn interval_ms(ms: f64) -> Result<Self> {
        let ms = unsigned_param("CommitPolicy.intervalMs", "ms", ms)?;
        Ok(Self {
            inner: CommitPolicy::Interval(std::time::Duration::from_millis(u64::from(ms))),
        })
    }
}
