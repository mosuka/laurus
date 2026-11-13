//! Implementations for the `delete` subcommand.
//!
//! Handles deleting resources from an existing index:
//!
//! - [`run_field`] - Remove a field from the schema.
//! - [`run_docs`] - Delete all documents (including chunks) by ID.

use std::path::Path;

use anyhow::Result;

use crate::context;

/// Execute the `delete field` command.
///
/// Opens the index at `index_dir` and calls [`Engine::delete_field`], which
/// persists the updated schema itself (the engine returned by
/// [`context::open_index`] is configured with a schema-persist hook — see
/// Issue #1078). The field is no longer indexed or searchable. A lexical
/// field's data is discarded from every segment (Issue #1546), and a
/// vector field's when a field with the same name is added again (Issue
/// #1535), so a field added again starts empty.
///
/// # Arguments
///
/// * `name` - The name of the field to delete.
/// * `index_dir` - Path to the index directory holding the index.
///
/// # Errors
///
/// Returns an error if:
/// - The index cannot be opened.
/// - The engine rejects the deletion (e.g. field does not exist).
/// - The updated schema cannot be persisted to disk.
pub async fn run_field(name: &str, index_dir: &Path) -> Result<()> {
    let engine = context::open_index(index_dir).await?;

    engine
        .delete_field(name)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Field '{name}' deleted successfully.");
    Ok(())
}

/// Execute the `delete docs` command.
///
/// Opens the index at `index_dir` and deletes all documents (including
/// chunks) with the given external ID. Changes are not committed
/// automatically.
///
/// # Arguments
///
/// * `id` - External document ID.
/// * `index_dir` - Path to the index directory holding the index.
///
/// # Errors
///
/// Returns an error if:
/// - The index cannot be opened.
/// - The engine rejects the deletion.
pub async fn run_docs(id: &str, index_dir: &Path) -> Result<()> {
    let engine = context::open_index(index_dir).await?;
    engine.delete_documents(id).await?;
    println!("Documents '{id}' deleted. Run 'commit' to persist changes.");
    Ok(())
}
