//! WASM wrapper for the Laurus [`CommitPolicy`] value object (Issue #893).
//!
//! Exposes the auto-commit policy to JavaScript so that callers can opt into
//! automatic commits when constructing an [`crate::index::WasmIndex`].

use laurus::CommitPolicy;
use wasm_bindgen::prelude::*;

use crate::convert::unsigned_param;

/// Auto-commit policy for an [`crate::index::WasmIndex`].
///
/// Controls when the engine automatically runs the commit ladder during
/// ingestion:
///
/// - **Manual** (the default) — the caller drives every `commit()`.
/// - **Every-docs** — the engine commits after every `n` applied documents,
///   across the singular and batch ingest APIs (and every `n` documents within
///   a single batch).
///
/// This is orthogonal to `WalSyncPolicy`.
///
/// ```javascript
/// import { Index, Schema, CommitPolicy } from "laurus-wasm";
///
/// const schema = new Schema();
/// schema.addTextField("title");
///
/// // Auto-commit after every 1000 applied documents.
/// const index = await Index.create(schema, undefined, CommitPolicy.everyDocs(1000));
///
/// // Manual (the default if no policy is supplied).
/// const manual = CommitPolicy.manual();
/// ```
///
/// **wasm note:** unlike `WalSyncPolicy`'s `maxIntervalMs` timer, `everyDocs`
/// needs no background thread — it is checked inline during ingestion — so it
/// works fully under WebAssembly.
#[wasm_bindgen(js_name = "CommitPolicy")]
#[derive(Clone, Copy)]
pub struct WasmCommitPolicy {
    /// The wrapped core auto-commit policy passed to `EngineBuilder`.
    pub(crate) inner: CommitPolicy,
}

#[wasm_bindgen(js_class = "CommitPolicy")]
impl WasmCommitPolicy {
    /// The manual policy: the engine commits only when `commit()` is called
    /// explicitly. This is the default when no policy is supplied.
    ///
    /// # Returns
    ///
    /// A `CommitPolicy` wrapping [`CommitPolicy::Manual`].
    #[wasm_bindgen(js_name = "manual")]
    pub fn manual() -> WasmCommitPolicy {
        WasmCommitPolicy {
            inner: CommitPolicy::Manual,
        }
    }

    /// The auto-commit-every-`n`-documents policy.
    ///
    /// `everyDocs(0)` disables auto-commit, which is equivalent to
    /// [`WasmCommitPolicy::manual`].
    ///
    /// # Arguments
    ///
    /// * `n` - Commit after this many applied documents. `0` disables
    ///   auto-commit.
    ///
    /// # Returns
    ///
    /// A `CommitPolicy` wrapping [`CommitPolicy::EveryDocs`].
    ///
    /// # Errors
    ///
    /// Throws if `n` is not an integer from 0 to 4294967295 (Issue #1612).
    #[wasm_bindgen(js_name = "everyDocs")]
    pub fn every_docs(n: f64) -> Result<WasmCommitPolicy, JsValue> {
        let n = unsigned_param("CommitPolicy.everyDocs", "n", n)?;
        Ok(WasmCommitPolicy {
            inner: CommitPolicy::EveryDocs(n as usize),
        })
    }

    /// The auto-commit-at-least-every-interval policy (Issue #892).
    ///
    /// The engine runs the commit ladder at least once every `ms`
    /// milliseconds via a background timer — the time-based counterpart of
    /// [`WasmCommitPolicy::every_docs`].
    ///
    /// **wasm note:** unlike under native builds, WebAssembly has no background
    /// thread, so the engine never starts the timer and `intervalMs` is a
    /// documented no-op at runtime under wasm. The factory still constructs the
    /// value so the same policy code is portable across targets.
    ///
    /// # Arguments
    ///
    /// * `ms` - Commit at least every this many milliseconds.
    ///
    /// # Returns
    ///
    /// A `CommitPolicy` wrapping [`CommitPolicy::Interval`].
    ///
    /// # Errors
    ///
    /// Throws if `ms` is not an integer from 0 to 4294967295 (Issue #1612).
    #[wasm_bindgen(js_name = "intervalMs")]
    pub fn interval_ms(ms: f64) -> Result<WasmCommitPolicy, JsValue> {
        let ms = unsigned_param("CommitPolicy.intervalMs", "ms", ms)?;
        Ok(WasmCommitPolicy {
            inner: CommitPolicy::Interval(std::time::Duration::from_millis(u64::from(ms))),
        })
    }
}
