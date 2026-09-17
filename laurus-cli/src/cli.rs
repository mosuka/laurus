//! CLI argument definitions for the laurus command-line tool.
//!
//! This module defines the top-level [`Cli`] struct and all subcommand
//! structures parsed by [`clap`]. Each subcommand maps to a specific
//! operation such as creating an index, querying documents, or starting
//! a gRPC server.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::output::OutputFormat;

/// Laurus - Unified search engine CLI
#[derive(Parser)]
#[command(name = "laurus", version, about)]
pub struct Cli {
    /// Path to the index directory.
    #[arg(long, env = "LAURUS_INDEX_DIR", default_value = "./laurus_index")]
    pub index_dir: PathBuf,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
    pub format: OutputFormat,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Create a resource.
    Create(CreateCommand),
    /// Get a resource.
    Get(GetCommand),
    /// Add a resource.
    Add(AddCommand),
    /// Put (upsert) a resource.
    Put(PutCommand),
    /// Delete a resource.
    Delete(DeleteCommand),
    /// Update a resource.
    Update(UpdateCommand),
    /// Commit pending changes.
    Commit,
    /// Train an auxiliary index structure (e.g. a shared PQ codebook).
    Train(TrainCommand),
    /// Execute a search query.
    Search(SearchCommand),
    /// Start an interactive REPL session.
    Repl,
    /// Start the gRPC server.
    Serve(ServeCommand),
    /// Start the MCP (Model Context Protocol) server on stdio.
    Mcp(McpCommand),
}

// --- Create ---

/// CLI arguments for the `create` subcommand.
///
/// Holds the target resource to create (e.g. an index or a schema file).
#[derive(Parser)]
pub struct CreateCommand {
    #[command(subcommand)]
    pub resource: CreateResource,
}

#[derive(Subcommand)]
pub enum CreateResource {
    /// Create a new index. If --schema is given, uses that TOML file;
    /// otherwise launches the interactive schema wizard. With
    /// --train-pq-codebook, shared PQ codebooks are trained as part of
    /// creation (Issue #920), removing the train-before-first-commit
    /// ordering hazard for fields that configure pq_codebook_path.
    Index {
        /// Path to an existing schema TOML file. When omitted, the
        /// interactive schema wizard is launched instead.
        #[arg(long)]
        schema: Option<PathBuf>,
        /// Path to a JSONL training file (the `put docs` / `add docs` shape;
        /// each field value a plain numeric array, e.g. `"embedding": [0.1,
        /// 0.2, ...]`). When given, every HNSW field that configures
        /// ProductQuantization + pq_codebook_path gets its shared codebook
        /// trained from this file immediately after creation, so the very
        /// first commit can already encode against it.
        #[arg(long)]
        train_pq_codebook: Option<PathBuf>,
    },
    /// Interactively generate a schema TOML file.
    Schema {
        /// Output file path for the generated schema TOML.
        #[arg(long, default_value = "schema.toml")]
        output: PathBuf,
    },
}

// --- Get ---

/// CLI arguments for the `get` subcommand.
///
/// Holds the target resource to retrieve (e.g. index stats or a document).
#[derive(Parser)]
pub struct GetCommand {
    #[command(subcommand)]
    pub resource: GetResource,
}

#[derive(Subcommand)]
pub enum GetResource {
    /// Show index statistics.
    Stats,
    /// Show the current schema.
    Schema,
    /// Get all documents (including chunks) by ID.
    Docs {
        /// External document ID.
        #[arg(long)]
        id: String,
    },
}

// --- Add ---

/// CLI arguments for the `add` subcommand.
///
/// Holds the target resource to add (e.g. a document).
#[derive(Parser)]
pub struct AddCommand {
    #[command(subcommand)]
    pub resource: AddResource,
}

#[derive(Subcommand)]
pub enum AddResource {
    /// Add a document to the index.
    Doc {
        /// External document ID.
        #[arg(long)]
        id: String,
        /// Document data as a JSON string, e.g. `{"fields": {"title": "Hello"}}`.
        #[arg(long)]
        data: String,
    },
    /// Bulk-add document chunks from a JSONL file (one entry per line).
    ///
    /// Each line is `{"id": "...", "fields": {...}}` — the same document
    /// JSON shape as `add doc --data`, with the external ID as a sibling
    /// top-level key. Unlike `put docs`, repeated ids accumulate as chunks.
    /// Commits automatically (per `--commit-every` and once at the end).
    Docs {
        /// Path to the JSONL file to ingest.
        #[arg(long)]
        file: std::path::PathBuf,
        /// Documents per engine batch call.
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        /// Commit every N applied documents (0 = only the final commit).
        #[arg(long, default_value_t = 0)]
        commit_every: usize,
    },
    /// Dynamically add a new field to an existing index.
    Field {
        /// The name of the new field.
        #[arg(long)]
        name: String,
        /// Field configuration as a JSON string.
        ///
        /// Uses the same externally-tagged format as the schema TOML.
        /// Examples:
        ///   '{"Text": {"indexed": true, "stored": true}}'
        ///   '{"Hnsw": {"dimension": 384}}'
        ///   '{"Integer": {}}'
        #[arg(long)]
        field_option: String,
    },
}

// --- Put ---

/// CLI arguments for the `put` subcommand.
///
/// Holds the target resource to put (upsert).
#[derive(Parser)]
pub struct PutCommand {
    #[command(subcommand)]
    pub resource: PutResource,
}

#[derive(Subcommand)]
pub enum PutResource {
    /// Put (upsert) a document into the index.
    ///
    /// If a document with the same ID already exists, all its chunks are
    /// deleted before the new document is indexed.
    Doc {
        /// External document ID.
        #[arg(long)]
        id: String,
        /// Document data as a JSON string, e.g. `{"fields": {"title": "Hello"}}`.
        #[arg(long)]
        data: String,
    },
    /// Bulk-upsert documents from a JSONL file (one entry per line).
    ///
    /// Each line is `{"id": "...", "fields": {...}}` — the same document
    /// JSON shape as `put doc --data`, with the external ID as a sibling
    /// top-level key. Entries are applied in order (duplicate ids dedup,
    /// last wins) with one WAL fsync per batch. Commits automatically (per
    /// `--commit-every` and once at the end).
    Docs {
        /// Path to the JSONL file to ingest.
        #[arg(long)]
        file: std::path::PathBuf,
        /// Documents per engine batch call.
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        /// Commit every N applied documents (0 = only the final commit).
        #[arg(long, default_value_t = 0)]
        commit_every: usize,
    },
}

// --- Delete ---

/// CLI arguments for the `delete` subcommand.
///
/// Holds the target resource to delete (e.g. a document by ID).
#[derive(Parser)]
pub struct DeleteCommand {
    #[command(subcommand)]
    pub resource: DeleteResource,
}

#[derive(Subcommand)]
pub enum DeleteResource {
    /// Delete all documents (including chunks) by ID.
    Docs {
        /// External document ID.
        #[arg(long)]
        id: String,
    },
    /// Remove a field from the index schema.
    Field {
        /// The name of the field to delete.
        #[arg(long)]
        name: String,
    },
}

// --- Update ---

/// CLI arguments for the `update` subcommand.
///
/// Holds the target resource to update (e.g. an existing field).
#[derive(Parser)]
pub struct UpdateCommand {
    #[command(subcommand)]
    pub resource: UpdateResource,
}

#[derive(Subcommand)]
pub enum UpdateResource {
    /// Change an existing field's type/options.
    ///
    /// Metadata-only changes (e.g. an HNSW field's `default_ef_search`)
    /// apply immediately. A change that requires rebuilding or discarding
    /// existing on-disk data (e.g. a text field's `analyzer`, or a vector
    /// field's `dimension`) is rejected unless `--reindex` is given.
    Field {
        /// The name of the field to update. Must already exist.
        #[arg(long)]
        name: String,
        /// The field's new configuration as a JSON string.
        ///
        /// Uses the same externally-tagged format as the schema TOML.
        /// Examples:
        ///   '{"Text": {"indexed": true, "stored": true, "analyzer": "english"}}'
        ///   '{"Hnsw": {"dimension": 384, "m": 32, "ef_construction": 400}}'
        #[arg(long)]
        field_option: String,
        /// Opt in to a change that requires rebuilding or discarding
        /// existing on-disk data. Ignored for a metadata-only change.
        #[arg(long, default_value_t = false)]
        reindex: bool,
        /// Classify the change and print the result without applying
        /// anything.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
}

// --- Train ---

/// CLI arguments for the `train` subcommand.
///
/// Holds the target structure to train (currently only a shared PQ
/// codebook, Issue #631).
#[derive(Parser)]
pub struct TrainCommand {
    #[command(subcommand)]
    pub resource: TrainResource,
}

#[derive(Subcommand)]
pub enum TrainResource {
    /// Train a shared PQ codebook for an HNSW vector field (Issue #631).
    ///
    /// Reads training vectors either from a JSONL file (--input, the same
    /// `{"id": "...", "fields": {...}}` shape as `put docs` / `add docs`;
    /// the field value must be a plain numeric array, e.g. `"embedding":
    /// [0.1, 0.2, ...]`) or from the vectors already committed to this
    /// index (--from-index, Issue #920), and persists the codebook into
    /// the index's vector storage.
    /// Subsequent commits encode segments against it instead of re-training
    /// k-means per segment — provided the schema's `pq_codebook_path` names
    /// the trained file (pass --update-schema to set it automatically).
    PqCodebook {
        /// The HNSW vector field to train for. Must be configured with
        /// ProductQuantization.
        #[arg(long)]
        field: String,
        /// Path to the JSONL training file. Exactly one of --input and
        /// --from-index must be given.
        #[arg(long)]
        input: Option<std::path::PathBuf>,
        /// Sample the vectors already committed to this index instead of
        /// reading a JSONL file (Issue #920). Exactly one of --input and
        /// --from-index must be given.
        #[arg(long, default_value_t = false)]
        from_index: bool,
        /// Use only the first N vectors (deterministic: file order for
        /// --input, ascending doc_id for --from-index; omit to use all).
        #[arg(long)]
        sample_size: Option<usize>,
        /// Storage-relative codebook file name (defaults to the field's
        /// configured pq_codebook_path, else "{field}.pqcb").
        #[arg(long)]
        output: Option<String>,
        /// Rewrite schema.toml so the field's pq_codebook_path names the
        /// trained file.
        #[arg(long, default_value_t = false)]
        update_schema: bool,
    },
}

// --- Mcp ---

/// CLI arguments for the `mcp` subcommand.
///
/// Configures the MCP stdio server and its connection to a running
/// laurus-server instance.
#[derive(Parser)]
pub struct McpCommand {
    /// gRPC endpoint of a running laurus-server to connect to at startup.
    ///
    /// If omitted, the server starts without a connection.  Use the `connect`
    /// MCP tool to connect later.
    #[arg(long, env = "LAURUS_ENDPOINT")]
    pub endpoint: Option<String>,
}

// --- Serve ---

/// CLI arguments for the `serve` subcommand.
///
/// Configures the gRPC server (and optional HTTP gateway) including
/// listen address, ports, and an optional TOML configuration file.
/// Values can be supplied via CLI flags or environment variables.
/// Use the `RUST_LOG` environment variable to control log verbosity.
#[derive(Parser)]
pub struct ServeCommand {
    /// Path to the configuration file (TOML).
    #[arg(short = 'c', long = "config", env = "LAURUS_CONFIG")]
    pub config: Option<PathBuf>,

    /// Listen address.
    #[arg(short = 'H', long = "host", env = "LAURUS_HOST")]
    pub host: Option<String>,

    /// Listen port.
    #[arg(short = 'p', long = "port", env = "LAURUS_PORT")]
    pub port: Option<u16>,

    /// HTTP Gateway port. If set, starts an HTTP gateway alongside the gRPC server.
    #[arg(long = "http-port", env = "LAURUS_HTTP_PORT")]
    pub http_port: Option<u16>,
}

// --- Search ---

/// CLI arguments for the `search` subcommand.
///
/// Accepts a query string written in the Laurus query DSL along with
/// pagination parameters (`limit` and `offset`).
#[derive(Parser)]
pub struct SearchCommand {
    /// Search query string (Laurus query DSL).
    pub query: String,

    /// Maximum number of results.
    #[arg(long, default_value_t = 10)]
    pub limit: usize,

    /// Number of results to skip.
    #[arg(long, default_value_t = 0)]
    pub offset: usize,

    /// Stored text field to highlight (Issue #1134). Repeat to request
    /// highlights on multiple fields, e.g. `--highlight title --highlight
    /// body`. Highlighting follows this query's lexical clause — a
    /// vector-only query produces no highlights. Uses the default
    /// `HighlightConfig` (tag `mark`, up to 5 fragments); there is no flag
    /// for the other `HighlightConfig` knobs.
    #[arg(long = "highlight", value_name = "FIELD")]
    pub highlight: Vec<String>,
}
