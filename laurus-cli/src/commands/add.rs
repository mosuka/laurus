//! Implementations for the `add` subcommand.
//!
//! Handles adding resources to an existing index:
//!
//! - [`run_field`] - Dynamically add a new field to the schema.
//! - [`run_doc`] - Add a document to the index.

use std::path::Path;

use anyhow::{Context, Result};
use laurus::FieldOption;

use crate::context;
use crate::json_doc::parse_document_json;

/// Execute the `add field` command.
///
/// Opens the index at `index_dir`, parses the field option JSON, and calls
/// [`Engine::add_field`](laurus::Engine::add_field), which persists the
/// updated schema itself (the
/// engine returned by [`context::open_index`] is configured with a
/// schema-persist hook — see Issue #1078).
///
/// # Arguments
///
/// * `name` - The name of the new field to add.
/// * `field_option_json` - A JSON string describing the field configuration
///   (e.g. `{"Text": {"indexed": true, "stored": true}}`).
/// * `index_dir` - Path to the index directory holding the index.
///
/// # Errors
///
/// Returns an error if:
/// - The index cannot be opened.
/// - The JSON string cannot be parsed into a [`FieldOption`].
/// - The engine rejects the field (e.g. duplicate name, unknown analyzer).
/// - The updated schema cannot be persisted to disk.
pub async fn run_field(name: &str, field_option_json: &str, index_dir: &Path) -> Result<()> {
    let engine = context::open_index(index_dir).await?;

    let field_option: FieldOption =
        serde_json::from_str(field_option_json).context("Failed to parse field option JSON")?;

    engine
        .add_field(name, field_option)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Field '{name}' added successfully.");
    Ok(())
}

/// Execute the `add doc` command.
///
/// Opens the index at `index_dir`, parses the document JSON, and adds it
/// to the index. Changes are not committed automatically.
///
/// # Arguments
///
/// * `id` - External document ID.
/// * `data_json` - A JSON string of the shape `{"fields": {...}}`.
/// * `index_dir` - Path to the index directory holding the index.
///
/// # Errors
///
/// Returns an error if:
/// - The index cannot be opened.
/// - The JSON string cannot be parsed into a [`laurus::Document`].
/// - The engine rejects the document.
pub async fn run_doc(id: &str, data_json: &str, index_dir: &Path) -> Result<()> {
    let engine = context::open_index(index_dir).await?;
    let doc = parse_document_json(data_json).context("Failed to parse document JSON")?;
    engine.add_document(id, doc).await?;
    println!("Document '{id}' added. Run 'commit' to persist changes.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use laurus::Schema;

    /// Issue #1726: an `add field` that cannot write `schema.toml` says why.
    /// What it printed used to end at `Failed to write schema file`.
    #[tokio::test]
    async fn run_field_says_why_it_cannot_write_schema_toml() {
        let dir = tempfile::tempdir().unwrap();
        let schema = Schema::from_toml("[fields.body.Text]\n").unwrap();
        context::create_index_from_schema(dir.path(), schema)
            .await
            .unwrap();
        // The temporary file that replaces schema.toml cannot be created.
        std::fs::create_dir(dir.path().join(laurus::index_dir::SCHEMA_TMP_FILE)).unwrap();

        let error = run_field("n", r#"{"Integer": {}}"#, dir.path())
            .await
            .unwrap_err();

        let printed = format!("{error:?}");
        let schema_path = dir.path().join("schema.toml").display().to_string();
        assert!(
            printed.starts_with(&format!(
                "Anyhow error: Failed to write schema file: {schema_path}: "
            )),
            "{printed}"
        );
        assert_eq!(printed.matches("os error").count(), 1, "{printed}");
    }

    /// Issue #1700: an unknown key in `--field-option` is an error naming
    /// it, instead of a field whose `indexed` silently keeps its default,
    /// and no field is added.
    #[tokio::test]
    async fn run_field_rejects_an_unknown_key_in_the_field_option() {
        let dir = tempfile::tempdir().unwrap();
        let schema = Schema::from_toml("[fields.body.Text]\n").unwrap();
        context::create_index_from_schema(dir.path(), schema)
            .await
            .unwrap();

        let error = run_field("title", r#"{"Text": {"indexd": false}}"#, dir.path())
            .await
            .unwrap_err();

        assert!(
            format!("{error:#}").contains("unknown field `indexd`"),
            "{error:#}"
        );
        assert!(
            !context::read_schema(dir.path())
                .unwrap()
                .fields
                .contains_key("title"),
            "no field may be added"
        );
    }
}
