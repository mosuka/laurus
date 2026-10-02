//! OPFS persistence layer for laurus-wasm.
//!
//! This module provides functions to load and save index data between
//! a [`MemoryStorage`] instance and the browser's Origin Private File System (OPFS).
//!
//! The design uses `MemoryStorage` as the runtime backend (which satisfies the
//! `Storage` trait's `Send + Sync` requirement) and OPFS as a persistence layer.
//! Data is loaded from OPFS into memory on `open`, and persisted back on `commit`.

use std::io::{Read, Write};
use std::sync::Arc;

use laurus::Schema;
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use wasm_bindgen::prelude::*;

use crate::errors::laurus_err;

/// Name of the persisted-schema file inside an OPFS index directory.
///
/// This file is managed directly through the OPFS bridge functions, NOT
/// through the engine's `Storage` trait / `MemoryStorage` -- see
/// [`OpfsPersistence::resolve_schema`], [`OpfsPersistence::load`], and
/// [`OpfsPersistence::save`] for why: `save()` deletes any OPFS file that
/// is not tracked in `MemoryStorage`, so a `schema.json` written outside
/// that tracking would be deleted on the very next `commit()`.
const SCHEMA_FILE: &str = "schema.json";

// ---------------------------------------------------------------------------
// JS FFI: OPFS bridge functions
// ---------------------------------------------------------------------------

#[wasm_bindgen(module = "/js/opfs_bridge.js")]
extern "C" {
    /// Initialize an OPFS directory for an index.
    ///
    /// Returns a `FileSystemDirectoryHandle` as an opaque `JsValue`.
    #[wasm_bindgen(catch)]
    async fn opfs_init(name: &str) -> Result<JsValue, JsValue>;

    /// List all file names in an OPFS directory.
    #[wasm_bindgen(catch)]
    async fn opfs_list_files(dir: &JsValue) -> Result<JsValue, JsValue>;

    /// Check if a file exists in the OPFS directory.
    #[wasm_bindgen(catch)]
    async fn opfs_file_exists(dir: &JsValue, name: &str) -> Result<JsValue, JsValue>;

    /// Read a file's contents as a `Uint8Array`.
    #[wasm_bindgen(catch)]
    async fn opfs_read_file(dir: &JsValue, name: &str) -> Result<JsValue, JsValue>;

    /// Write data to a file in the OPFS directory.
    #[wasm_bindgen(catch)]
    async fn opfs_write_file(dir: &JsValue, name: &str, data: &[u8]) -> Result<JsValue, JsValue>;

    /// Delete a file from the OPFS directory.
    #[wasm_bindgen(catch)]
    async fn opfs_delete_file(dir: &JsValue, name: &str) -> Result<JsValue, JsValue>;

    /// Delete an entire index directory.
    #[wasm_bindgen(catch)]
    async fn opfs_delete_index(name: &str) -> Result<JsValue, JsValue>;
}

// ---------------------------------------------------------------------------
// OpfsPersistence
// ---------------------------------------------------------------------------

/// OPFS persistence handle for a single index.
///
/// Holds a reference to an OPFS directory (`FileSystemDirectoryHandle`)
/// and provides methods to load/save data to/from a [`MemoryStorage`].
pub struct OpfsPersistence {
    /// The OPFS directory handle (opaque JS object).
    dir: JsValue,
    /// The index name (used for error messages and deletion).
    name: String,
}

impl OpfsPersistence {
    /// Open or create an OPFS directory for the given index name.
    ///
    /// # Arguments
    ///
    /// * `name` - Index name used as the OPFS subdirectory name.
    ///
    /// # Returns
    ///
    /// A new `OpfsPersistence` handle.
    pub async fn open(name: &str) -> Result<Self, JsValue> {
        let dir = opfs_init(name).await?;
        Ok(Self {
            dir,
            name: name.to_string(),
        })
    }

    /// Load all files from OPFS into a new [`MemoryStorage`].
    ///
    /// Creates a new `MemoryStorage`, reads every file from the OPFS directory,
    /// and writes it into the in-memory store. Skips [`SCHEMA_FILE`], which is
    /// managed separately by [`resolve_schema`](Self::resolve_schema) and is
    /// not part of the engine's storage namespace.
    ///
    /// # Returns
    ///
    /// An `Arc<MemoryStorage>` populated with all OPFS files, ready to be
    /// passed to `Engine::new()`.
    pub async fn load(&self) -> Result<Arc<MemoryStorage>, JsValue> {
        let storage = MemoryStorage::new(MemoryStorageConfig::default());

        let file_names_js = opfs_list_files(&self.dir).await?;
        let file_names: Vec<String> = serde_wasm_bindgen::from_value(file_names_js)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse file list: {e}")))?;

        for file_name in &file_names {
            if file_name == SCHEMA_FILE {
                continue;
            }
            let data_js = opfs_read_file(&self.dir, file_name).await?;
            let data: Vec<u8> = js_sys::Uint8Array::new(&data_js).to_vec();

            let mut output = storage.create_output(file_name).map_err(|e| {
                JsValue::from_str(&format!("Failed to create output '{file_name}': {e}"))
            })?;
            output
                .write_all(&data)
                .map_err(|e| JsValue::from_str(&format!("Failed to write '{file_name}': {e}")))?;
            output
                .close()
                .map_err(|e| JsValue::from_str(&format!("Failed to close '{file_name}': {e}")))?;
        }

        Ok(Arc::new(storage))
    }

    /// Save all files from a [`MemoryStorage`] to OPFS.
    ///
    /// Lists all files in the `MemoryStorage`, reads each one, and writes
    /// it to the OPFS directory. Files in OPFS that no longer exist in
    /// memory are deleted -- **except** [`SCHEMA_FILE`], which this method
    /// never touches (it is written directly by
    /// [`resolve_schema`](Self::resolve_schema), not through
    /// `MemoryStorage`, so it would otherwise look "stale" here and get
    /// deleted on every commit).
    ///
    /// # Arguments
    ///
    /// * `storage` - The in-memory storage to persist.
    pub async fn save(&self, storage: &dyn Storage) -> Result<(), JsValue> {
        let memory_files: Vec<String> = storage
            .list_files()
            .map_err(|e| JsValue::from_str(&format!("Failed to list memory files: {e}")))?;

        // Delete OPFS files that are no longer in memory
        let opfs_files_js = opfs_list_files(&self.dir).await?;
        let opfs_files: Vec<String> = serde_wasm_bindgen::from_value(opfs_files_js)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse OPFS file list: {e}")))?;

        for opfs_file in &opfs_files {
            if opfs_file != SCHEMA_FILE && !memory_files.contains(opfs_file) {
                opfs_delete_file(&self.dir, opfs_file).await?;
            }
        }

        // Write all memory files to OPFS
        for file_name in &memory_files {
            let mut input = storage
                .open_input(file_name)
                .map_err(|e| JsValue::from_str(&format!("Failed to open '{file_name}': {e}")))?;
            let mut data = Vec::new();
            input
                .read_to_end(&mut data)
                .map_err(|e| JsValue::from_str(&format!("Failed to read '{file_name}': {e}")))?;
            opfs_write_file(&self.dir, file_name, &data).await?;
        }

        Ok(())
    }

    /// Delete the entire index from OPFS.
    pub async fn delete(&self) -> Result<(), JsValue> {
        opfs_delete_index(&self.name).await?;
        Ok(())
    }

    /// Resolve which [`Schema`] to use for this OPFS index, persisting it
    /// to [`SCHEMA_FILE`] as needed.
    ///
    /// Three cases:
    ///
    /// - `SCHEMA_FILE` already exists: this is a normal reopen. The
    ///   persisted schema is loaded and returned; a `schema` passed by the
    ///   caller is intentionally **ignored** for this purpose (not an
    ///   error) -- unlike the native bindings' `Schema`, `WasmSchema` also
    ///   carries per-call JS embedder callbacks and runtime analyzers that
    ///   can never be persisted, so callers must keep passing a
    ///   `WasmSchema` on every reopen just to supply those, even once the
    ///   field-schema part is fully persisted. Rejecting that would make
    ///   reopening with custom embedders/analyzers impossible.
    /// - `SCHEMA_FILE` is absent and the directory has no other files
    ///   either: this is a fresh index. `schema` (or [`Schema::default`]
    ///   if omitted) is checked with [`Schema::validate_for_create`],
    ///   written to `SCHEMA_FILE` and returned.
    /// - `SCHEMA_FILE` is absent but other files already exist (an index
    ///   persisted before this method existed): `schema` must be `Some`
    ///   (an `Err` asking the caller to supply it once is returned
    ///   otherwise, since guessing would silently mismatch the real,
    ///   already-persisted data). The given schema is trusted as-is and
    ///   backfilled into `SCHEMA_FILE` so future reopens don't need it
    ///   again.
    ///
    /// # Arguments
    ///
    /// * `schema` - The schema the caller passed to `Index.open`, if any.
    pub async fn resolve_schema(&self, schema: Option<Schema>) -> Result<Schema, JsValue> {
        let schema_file_exists = opfs_file_exists(&self.dir, SCHEMA_FILE)
            .await?
            .as_bool()
            .unwrap_or(false);
        if schema_file_exists {
            let data_js = opfs_read_file(&self.dir, SCHEMA_FILE).await?;
            let bytes = js_sys::Uint8Array::new(&data_js).to_vec();
            return serde_json::from_slice(&bytes).map_err(|e| {
                JsValue::from_str(&format!(
                    "Failed to parse persisted schema for OPFS index '{}': {e}",
                    self.name
                ))
            });
        }

        let file_names_js = opfs_list_files(&self.dir).await?;
        let file_names: Vec<String> = serde_wasm_bindgen::from_value(file_names_js)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse file list: {e}")))?;

        let schema = schema_for_unpersisted_index(&self.name, schema, file_names.is_empty())?;

        let json = serde_json::to_vec(&schema).map_err(|e| {
            JsValue::from_str(&format!(
                "Failed to serialize schema for OPFS index '{}': {e}",
                self.name
            ))
        })?;
        opfs_write_file(&self.dir, SCHEMA_FILE, &json).await?;
        Ok(schema)
    }
}

/// Choose the schema to persist for an OPFS index that has no
/// `SCHEMA_FILE` yet (the last two cases of
/// [`OpfsPersistence::resolve_schema`]).
///
/// # Arguments
///
/// * `index_name` - The OPFS index name, for error messages.
/// * `schema` - The schema the caller passed to `Index.open`, if any.
/// * `dir_is_empty` - Whether the index directory holds no files, i.e.
///   whether this creates a fresh index rather than backfilling the schema
///   of data persisted before schema tracking was added.
///
/// # Errors
///
/// Returns an error if a fresh index's `schema` fails
/// [`Schema::validate_for_create`], or if the directory holds data but no
/// `schema` was given.
fn schema_for_unpersisted_index(
    index_name: &str,
    schema: Option<Schema>,
    dir_is_empty: bool,
) -> Result<Schema, JsValue> {
    match schema {
        Some(schema) if dir_is_empty => {
            schema.validate_for_create().map_err(laurus_err)?;
            Ok(schema)
        }
        // A backfill describes data that already exists, so it is trusted
        // as-is, like a persisted schema on reopen.
        Some(schema) => Ok(schema),
        None if dir_is_empty => Ok(Schema::default()),
        None => Err(JsValue::from_str(&format!(
            "OPFS index '{index_name}' contains data persisted before schema tracking was \
             added, but no schema was provided; pass the original schema once to migrate this \
             index (it will be persisted for future opens)"
        ))),
    }
}

// See the note on `wal.rs`'s tests: this crate's tests run on `wasm32` only.
#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod tests {
    use super::schema_for_unpersisted_index;
    use laurus::Schema;
    use wasm_bindgen_test::wasm_bindgen_test;

    fn schema_with_reserved_analyzer_name() -> Schema {
        Schema::from_toml(
            r#"
            [analyzers.standard]
            tokenizer = { type = "whitespace" }

            [fields.body.Text]
            analyzer = "standard"
            "#,
        )
        .unwrap()
    }

    #[wasm_bindgen_test]
    fn fresh_index_rejects_reserved_analyzer_name() {
        let err =
            schema_for_unpersisted_index("idx", Some(schema_with_reserved_analyzer_name()), true)
                .unwrap_err();
        let msg = err.as_string().unwrap();
        assert!(
            msg.contains("Analyzer name 'standard' is reserved for a built-in analyzer"),
            "got: {msg}"
        );
    }

    #[wasm_bindgen_test]
    fn backfill_keeps_reserved_analyzer_name() {
        let schema =
            schema_for_unpersisted_index("idx", Some(schema_with_reserved_analyzer_name()), false)
                .unwrap();
        assert!(schema.analyzers.contains_key("standard"));
    }

    #[wasm_bindgen_test]
    fn fresh_index_without_schema_uses_default() {
        let schema = schema_for_unpersisted_index("idx", None, true).unwrap();
        assert!(schema.fields.is_empty() && schema.analyzers.is_empty());
    }

    #[wasm_bindgen_test]
    fn backfill_without_schema_is_rejected() {
        let err = schema_for_unpersisted_index("idx", None, false).unwrap_err();
        assert!(
            err.as_string()
                .unwrap()
                .contains("persisted before schema tracking")
        );
    }
}
