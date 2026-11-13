//! Implementations for the `update` subcommand.
//!
//! Handles changing resources in an existing index:
//!
//! - [`run_field`] - Change an existing field's type/options.

use std::path::Path;

use anyhow::{Context, Result};
use laurus::{FieldOption, UpdateFieldOptions};

use crate::context;

/// Execute the `update field` command.
///
/// Opens the index at `index_dir`, parses the field option JSON, and calls
/// [`Engine::update_field`], which persists the updated schema itself (the
/// engine returned by [`context::open_index`] is configured with a
/// schema-persist hook — see Issue #1078). A `Reindex`- or
/// `Destructive`-classified change (e.g. a text field's `analyzer`, or a
/// vector field's `dimension`) is rejected unless `reindex` is `true` --
/// see [`UpdateFieldOptions::reindex`].
///
/// # Arguments
///
/// * `name` - The name of the field to update. Must already exist.
/// * `field_option_json` - A JSON string describing the field's new
///   configuration (e.g. `{"Text": {"indexed": true, "stored": true,
///   "analyzer": "english"}}`).
/// * `reindex` - Opt-in gate for a `Reindex`/`Destructive` change.
/// * `dry_run` - When `true`, classifies the change and reports it without
///   applying anything.
/// * `index_dir` - Path to the index directory holding the index.
///
/// # Errors
///
/// Returns an error if:
/// - The index cannot be opened.
/// - The JSON string cannot be parsed into a [`FieldOption`].
/// - The engine rejects the change (e.g. unknown field, a `Reindex`/
///   `Destructive` change without `--reindex`).
/// - The updated schema cannot be persisted to disk.
pub async fn run_field(
    name: &str,
    field_option_json: &str,
    reindex: bool,
    dry_run: bool,
    index_dir: &Path,
) -> Result<()> {
    let engine = context::open_index(index_dir).await?;

    let field_option: FieldOption =
        serde_json::from_str(field_option_json).context("Failed to parse field option JSON")?;

    let outcome = engine
        .update_field(name, field_option, UpdateFieldOptions { reindex, dry_run })
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if dry_run {
        println!(
            "Dry run: change to field '{name}' classified as {:?}. Nothing was applied.",
            outcome.classification
        );
    } else {
        println!(
            "Field '{name}' updated successfully (classified as {:?}).",
            outcome.classification
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use laurus::Schema;

    /// Issue #1512: a reindex change without `--reindex` is rejected with a
    /// message that names the opt-in as `reindex`, not by the library's
    /// `UpdateFieldOptions`, which a command-line user cannot pass.
    #[tokio::test]
    async fn run_field_rejection_names_reindex() {
        let dir = tempfile::tempdir().unwrap();
        let schema = Schema::from_toml("[fields.embedding.Hnsw]\ndimension = 4\n").unwrap();
        context::create_index_from_schema(dir.path(), schema)
            .await
            .unwrap();

        let error = run_field(
            "embedding",
            r#"{"Hnsw": {"dimension": 4, "m": 32}}"#,
            false,
            false,
            dir.path(),
        )
        .await
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Invalid argument: updating field 'embedding' requires rebuilding or discarding \
             existing data (classified as Reindex); enable `reindex` to proceed"
        );
    }
}
