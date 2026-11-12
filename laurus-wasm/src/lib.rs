//! WebAssembly bindings for the Laurus unified search library.
//!
//! Build with [wasm-pack](https://rustwasm.github.io/wasm-pack/):
//!
//! ```bash
//! cd laurus-wasm
//! wasm-pack build --target web
//! wasm-pack build --target bundler
//! ```

#![deny(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::enum_variant_names)]

mod analysis;
mod commit;
mod convert;
mod embedder;
mod errors;
mod index;
mod query;
mod schema;
mod search;
mod storage;
mod wal;

use wasm_bindgen::prelude::wasm_bindgen;

/// Return the laurus-wasm crate version (e.g. `"0.12.1"`).
///
/// Applications that persist laurus state in OPFS can stamp it with this
/// value and detect on the next visit that the state was written by a
/// different build whose on-disk format may have changed (see GitHub
/// issue #981 — a format cutover otherwise surfaces lazily as
/// search-time "Rebuild required" errors while `Index.open` succeeds).
///
/// # Returns
///
/// The compile-time `CARGO_PKG_VERSION` of the binding.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
