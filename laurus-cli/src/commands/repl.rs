//! Interactive Read-Eval-Print Loop (REPL) for the laurus search engine.
//!
//! Provides a terminal-based interactive session where users can create
//! indexes, search the index, add/get/delete fields and documents, commit
//! changes, and view index statistics without restarting the CLI. Command
//! history is supported via `rustyline`.
//!
//! Commands follow the same `<operation> <resource>` ordering as the CLI
//! (e.g. `add doc`, `delete field`, `get stats`).
//!
//! If no index exists in the configured `--index-dir`, the REPL starts
//! without a loaded index and prompts the user to run `create index` or
//! `create schema` first.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use laurus::{Engine, FieldOption, Schema, SearchRequestBuilder};
use rustyline::DefaultEditor;

use crate::commands::create;
use crate::context;
use crate::json_doc::parse_document_json;
use crate::output::{self, OutputFormat};

/// Error message shown when a command requires an open index but none is loaded.
const NO_INDEX_MSG: &str = "No index loaded. Use 'create index <schema_path>' first.";

/// Run the interactive REPL session.
///
/// If an index already exists at `index_dir`, it is opened automatically.
/// Otherwise the REPL starts without a loaded index and the user can create
/// one interactively via `create index` or `create schema`.
///
/// # Arguments
///
/// * `index_dir` - Path to the index directory holding the index.
/// * `format` - The desired output format (table or JSON) for results.
///
/// # Returns
///
/// Returns `Ok(())` when the user exits the REPL via `quit`, `exit`, Ctrl-C,
/// or Ctrl-D.
///
/// # Errors
///
/// Returns an error if the readline editor fails to initialise.
pub async fn run(index_dir: &Path, format: OutputFormat) -> Result<()> {
    // Try to open an existing index; if it fails, start without one.
    let (mut engine, mut schema) = match try_open_index(index_dir).await {
        Some((e, s)) => (Some(e), Some(s)),
        None => (None, None),
    };

    let mut rl = DefaultEditor::new()?;

    if engine.is_some() {
        println!("Laurus REPL (type 'help' for commands, 'quit' to exit)");
    } else {
        println!("Laurus REPL — no index found at {}.", index_dir.display());
        println!("Use 'create index <schema_path>' to create one, or 'help' for commands.");
    }

    loop {
        let line = match rl.readline("laurus> ") {
            Ok(line) => line,
            Err(
                rustyline::error::ReadlineError::Interrupted | rustyline::error::ReadlineError::Eof,
            ) => {
                break;
            }
            Err(err) => {
                eprintln!("Error: {err}");
                break;
            }
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(line);

        let parts: Vec<&str> = line.splitn(3, ' ').collect();
        let result = match parts[0] {
            "help" => {
                print_help();
                Ok(())
            }
            "quit" | "exit" => break,
            "create" => {
                if parts.len() < 2 {
                    eprintln!("Usage: create <index|schema> ...");
                    continue;
                }
                handle_create(
                    index_dir,
                    parts[1],
                    parts.get(2).copied(),
                    &mut engine,
                    &mut schema,
                )
                .await
            }
            "search" => {
                if parts.len() < 2 {
                    eprintln!("Usage: search <query>");
                    continue;
                }
                let Some(ref eng) = engine else {
                    eprintln!("{NO_INDEX_MSG}");
                    continue;
                };
                let query_str = parts[1..].join(" ");
                handle_search(eng, &query_str, format).await
            }
            "add" => {
                if parts.len() < 2 {
                    eprintln!("Usage: add <field|doc> ...");
                    continue;
                }
                let Some(ref eng) = engine else {
                    eprintln!("{NO_INDEX_MSG}");
                    continue;
                };
                handle_add(eng, parts[1], parts.get(2).copied()).await
            }
            "put" => {
                if parts.len() < 2 {
                    eprintln!("Usage: put <doc> ...");
                    continue;
                }
                let Some(ref eng) = engine else {
                    eprintln!("{NO_INDEX_MSG}");
                    continue;
                };
                handle_put(eng, parts[1], parts.get(2).copied()).await
            }
            "get" => {
                if parts.len() < 2 {
                    eprintln!("Usage: get <stats|schema|docs> ...");
                    continue;
                }
                let Some(ref eng) = engine else {
                    eprintln!("{NO_INDEX_MSG}");
                    continue;
                };
                handle_get(eng, index_dir, parts[1], parts.get(2).copied(), format).await
            }
            "delete" => {
                if parts.len() < 2 {
                    eprintln!("Usage: delete <field|docs> ...");
                    continue;
                }
                let Some(ref eng) = engine else {
                    eprintln!("{NO_INDEX_MSG}");
                    continue;
                };
                handle_delete(eng, parts[1], parts.get(2).copied()).await
            }
            "commit" => {
                let Some(ref eng) = engine else {
                    eprintln!("{NO_INDEX_MSG}");
                    continue;
                };
                eng.commit().await?;
                println!("Changes committed.");
                Ok(())
            }
            _ => {
                eprintln!(
                    "Unknown command: '{}'. Type 'help' for available commands.",
                    parts[0]
                );
                Ok(())
            }
        };

        if let Err(e) = result {
            eprintln!("Error: {e:#}");
        }
    }

    println!("Goodbye.");
    Ok(())
}

/// Try to open an existing index. Returns `None` if the index does not exist.
async fn try_open_index(index_dir: &Path) -> Option<(Engine, Schema)> {
    let schema_path = index_dir.join("schema.toml");
    if !schema_path.exists() {
        return None;
    }

    match context::open_index(index_dir).await {
        Ok(engine) => match context::read_schema(index_dir) {
            Ok(schema) => Some((engine, schema)),
            Err(e) => {
                eprintln!("Warning: failed to read schema: {e:#}");
                None
            }
        },
        Err(e) => {
            eprintln!("Warning: failed to open index: {e:#}");
            None
        }
    }
}

fn print_help() {
    println!(
        "\
Available commands:
  create index [schema_path]   Create a new index (interactive wizard if no path given)
  create schema <output_path>  Interactive schema generation wizard
  search <query>               Search the index (lexical / vector / hybrid DSL)
  add field <name> <json>      Add a field to the schema
  add doc <id> <json>          Add a document (append, allows multiple chunks per ID)
  put doc <id> <json>          Put (upsert) a document (replaces existing with same ID)
  get stats                    Show index statistics
  get schema                   Show current schema
  get docs <id>                Get all documents (including chunks) by ID
  delete field <name>          Remove a field from the schema
  delete docs <id>             Delete all documents (including chunks) by ID
  commit                       Commit pending changes
  help                         Show this help
  quit                         Exit the REPL"
    );
}

/// Handle `create index ...` and `create schema ...` commands.
///
/// # Arguments
///
/// * `index_dir` - Path to the index directory.
/// * `resource` - The resource type (`index` or `schema`).
/// * `rest` - Remaining arguments (file path).
/// * `engine` - Mutable reference to the optional engine slot.
/// * `schema` - Mutable reference to the optional schema slot.
async fn handle_create(
    index_dir: &Path,
    resource: &str,
    rest: Option<&str>,
    engine: &mut Option<Engine>,
    schema: &mut Option<Schema>,
) -> Result<()> {
    match resource {
        "index" => {
            match rest {
                Some(schema_path_str) => {
                    let schema_path = PathBuf::from(schema_path_str);
                    context::create_index(index_dir, &schema_path).await?;
                }
                None => {
                    // If schema.toml already exists, use it directly instead
                    // of launching the wizard (recovery for missing store/).
                    if index_dir.join("schema.toml").exists() {
                        let existing = context::read_schema(index_dir)?;
                        context::create_index_from_schema(index_dir, existing).await?;
                    } else {
                        let new_schema = create::build_schema_interactive()?;
                        context::create_index_from_schema(index_dir, new_schema).await?;
                    }
                }
            }

            // Open the newly created index.
            let eng = context::open_index(index_dir).await?;
            let sch = context::read_schema(index_dir)?;
            *engine = Some(eng);
            *schema = Some(sch);

            println!("Index created at {}.", index_dir.display());
            Ok(())
        }
        "schema" => {
            let output_str = rest.context("Usage: create schema <output_path>")?;
            let output = PathBuf::from(output_str);
            create::run_schema(&output)?;
            Ok(())
        }
        _ => {
            eprintln!("Unknown resource: '{resource}'. Use index or schema.");
            Ok(())
        }
    }
}

/// Result limit used by the REPL's `search` command.
///
/// The REPL takes no flags, so this is fixed. Use the one-shot
/// `laurus search --limit N` for a different page size.
const REPL_SEARCH_LIMIT: usize = 10;

/// Run a search and print the results.
///
/// The query string is handed to the engine as DSL so it is parsed with
/// the index's own `PerFieldAnalyzer` (a Japanese field declared in
/// `schema.toml` is analyzed with its Lindera analyzer, not with the
/// English `StandardAnalyzer`). As a side effect the REPL now accepts the
/// full unified DSL, including vector and hybrid clauses, which the
/// previous lexical-only path rejected.
async fn handle_search(engine: &Engine, query_str: &str, format: OutputFormat) -> Result<()> {
    let request = SearchRequestBuilder::new()
        .query_dsl(query_str)
        .limit(REPL_SEARCH_LIMIT)
        .build();

    let results = engine
        .search(request)
        .await
        .context("Failed to execute search")?;
    output::print_search_results(&results, format);
    Ok(())
}

/// Handle `add field ...` and `add doc ...` commands.
async fn handle_add(engine: &Engine, resource: &str, rest: Option<&str>) -> Result<()> {
    match resource {
        "field" => {
            let rest = rest.context("Usage: add field <name> <json>")?;
            let (name, json_str) = rest
                .split_once(' ')
                .context("Usage: add field <name> <json>")?;
            let field_option: FieldOption =
                serde_json::from_str(json_str).context("Failed to parse field option JSON")?;
            engine
                .add_field(name, field_option)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("Field '{name}' added.");
            Ok(())
        }
        "doc" => {
            let rest = rest.context("Usage: add doc <id> <json>")?;
            let (id, json_str) = rest.split_once(' ').context("Usage: add doc <id> <json>")?;
            let doc = parse_document_json(json_str).context("Failed to parse document JSON")?;
            engine.add_document(id, doc).await?;
            println!("Document '{id}' added.");
            Ok(())
        }
        _ => {
            eprintln!("Unknown resource: '{resource}'. Use field or doc.");
            Ok(())
        }
    }
}

/// Handle `put doc ...` command.
///
/// Upserts a document into the index. If a document with the same ID
/// already exists, all its chunks are deleted before the new document
/// is indexed.
///
/// # Arguments
///
/// * `engine` - Reference to the search engine instance.
/// * `resource` - The resource type (currently only `doc`).
/// * `rest` - Remaining arguments (`<id> <json>`).
async fn handle_put(engine: &Engine, resource: &str, rest: Option<&str>) -> Result<()> {
    match resource {
        "doc" => {
            let rest = rest.context("Usage: put doc <id> <json>")?;
            let (id, json_str) = rest.split_once(' ').context("Usage: put doc <id> <json>")?;
            let doc = parse_document_json(json_str).context("Failed to parse document JSON")?;
            engine.put_document(id, doc).await?;
            println!("Document '{id}' put (upserted).");
            Ok(())
        }
        _ => {
            eprintln!("Unknown resource: '{resource}'. Use doc.");
            Ok(())
        }
    }
}

/// Handle `get stats`, `get schema`, and `get docs ...` commands.
async fn handle_get(
    engine: &Engine,
    index_dir: &Path,
    resource: &str,
    rest: Option<&str>,
    format: OutputFormat,
) -> Result<()> {
    match resource {
        "stats" => {
            let stats = engine.stats()?;
            output::print_stats(&stats, format);
            Ok(())
        }
        "schema" => {
            let schema = context::read_schema(index_dir)?;
            let json = serde_json::to_string_pretty(&schema)
                .context("Failed to serialize schema to JSON")?;
            println!("{json}");
            Ok(())
        }
        "docs" => {
            let id = rest.context("Usage: get docs <id>")?;
            let documents = engine.get_documents(id).await?;
            output::print_documents(id, &documents, format);
            Ok(())
        }
        _ => {
            eprintln!("Unknown resource: '{resource}'. Use stats, schema, or docs.");
            Ok(())
        }
    }
}

/// Handle `delete field ...` and `delete docs ...` commands.
async fn handle_delete(engine: &Engine, resource: &str, rest: Option<&str>) -> Result<()> {
    match resource {
        "field" => {
            let name = rest.context("Usage: delete field <name>")?;
            engine
                .delete_field(name)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("Field '{name}' deleted.");
            Ok(())
        }
        "docs" => {
            let id = rest.context("Usage: delete docs <id>")?;
            engine.delete_documents(id).await?;
            println!("Documents '{id}' deleted.");
            Ok(())
        }
        _ => {
            eprintln!("Unknown resource: '{resource}'. Use field or docs.");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use laurus::Document;

    use super::*;

    /// Same dictionary-free bigram fixture as `commands::search::tests` —
    /// a stand-in for a morphological analyzer so a bare query term
    /// analyzes into several tokens without requiring a Lindera
    /// dictionary.
    const BIGRAM_SCHEMA: &str = r#"
default_fields = ["body"]

[fields.body.Text]
indexed = true
stored = true
analyzer = "ja_bigram"

[analyzers.ja_bigram]
tokenizer = { type = "ngram", min_gram = 2, max_gram = 2 }
"#;

    /// `handle_search` used to build a `StandardAnalyzer`-only
    /// `LexicalQueryParser` regardless of the schema, so a query against a
    /// custom-analyzer field either matched nothing or (for CJK text)
    /// errored out. It must now succeed and print without panicking.
    #[tokio::test]
    async fn repl_search_uses_the_schema_analyzer() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("schema.toml"), BIGRAM_SCHEMA).unwrap();

        let engine = context::open_index(dir.path()).await.unwrap();
        engine
            .put_document(
                "doc1",
                Document::builder()
                    .add_field("body", "吾輩は猫である")
                    .build(),
            )
            .await
            .unwrap();
        engine.commit().await.unwrap();

        // 2+ characters: the bigram analyzer needs at least min_gram (2).
        let result = handle_search(&engine, "吾輩は", OutputFormat::Json).await;
        assert!(result.is_ok(), "{result:?}");
    }
}
