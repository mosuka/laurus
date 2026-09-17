//! MCP server implementation for the laurus search engine.
//!
//! [`LaurusMcpServer`] is a [`rmcp::ServerHandler`] that proxies MCP tool calls
//! to a running laurus-server instance via gRPC.  Use [`run`] to start the
//! server on stdio.

use std::sync::Arc;

use anyhow::Context as _;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::schemars;
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::RwLock;
use tonic::transport::Channel;
use tracing::info;

use laurus_server::proto::laurus::v1::{
    AddDocumentRequest, AddDocumentsRequest, AddFieldRequest, CommitRequest, CreateIndexRequest,
    DeleteDocumentsRequest, DeleteFieldRequest, DocumentEntry, FieldChangeKind,
    GetDocumentsRequest, GetIndexRequest, GetSchemaRequest, PutDocumentRequest,
    PutDocumentsRequest, SearchBatchRequest, SearchRequest, UpdateFieldRequest,
    document_service_client::DocumentServiceClient, index_service_client::IndexServiceClient,
    search_service_client::SearchServiceClient,
};

use crate::convert;

// ── Parameter structs ─────────────────────────────────────────────────────────

/// Parameters for the `connect` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ConnectParams {
    /// gRPC endpoint of the laurus-server to connect to.
    ///
    /// Must include the scheme and port, for example `http://localhost:50051`.
    endpoint: String,
}

/// Parameters for the `create_index` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct CreateIndexParams {
    /// Schema definition as a JSON string.
    ///
    /// The schema must conform to the laurus schema format.  See the laurus
    /// documentation for the full field type reference
    /// (Text, Integer, Float, Boolean, DateTime, Geo, Hnsw, Flat, Ivf, …).
    ///
    /// FieldOption uses serde's externally-tagged representation where the
    /// variant name is the key.  The optional `dynamic_field_policy` key
    /// (values: `"Strict"`, `"Dynamic"`, `"Ignore"`) controls how fields
    /// that appear in ingested documents but are absent from the schema
    /// are handled.  It defaults to `"Dynamic"`, which silently truncates
    /// incoming float values for integer fields (e.g. `3.14` → `3`) —
    /// use `"Strict"` if you need to reject such type mismatches.
    ///
    /// Example:
    ///
    /// ```json
    /// {
    ///   "dynamic_field_policy": "Dynamic",
    ///   "fields": {
    ///     "title": { "Text": { "indexed": true, "stored": true } },
    ///     "body":  { "Text": {} },
    ///     "score": { "Float": {} },
    ///     "vec":   { "Hnsw": { "dimension": 384 } }
    ///   }
    /// }
    /// ```
    schema_json: String,
}

/// Parameters for the `put_document` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct PutDocumentParams {
    /// External document identifier (used for retrieval and deduplication).
    id: String,

    /// Document fields as a JSON object, e.g. `{"title": "Hello", "year": 2024}`.
    ///
    /// Field names and value types must match the index schema.
    fields: Value,
}

/// Parameters for the `add_document` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct AddDocumentParams {
    /// External document identifier (used for retrieval and deduplication).
    id: String,

    /// Document fields as a JSON object, e.g. `{"title": "Hello", "year": 2024}`.
    ///
    /// Field names and value types must match the index schema.
    fields: Value,
}

/// Parameters for the `put_documents` / `add_documents` tools.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct BulkDocumentsParams {
    /// Array of `{"id": "...", "fields": {...}}` entries, applied
    /// sequentially in input order with one WAL fsync for the whole batch.
    ///
    /// Each entry's `fields` is a JSON object of fields matching the index
    /// schema (the same shape as the `put_document` tool's `fields`).
    documents: Vec<Value>,
}

/// Parameters for the `get_documents` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GetDocumentsParams {
    /// External document identifier to look up.
    id: String,
}

/// Parameters for the `delete_documents` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct DeleteDocumentsParams {
    /// External document identifier to delete.
    id: String,
}

/// Parameters for the `search` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchParams {
    /// Search query string in the laurus unified query DSL.
    ///
    /// Supports three search modes in a single query string:
    ///
    /// **Lexical search** — standard text search syntax:
    /// - Term queries: `title:hello`
    /// - Boolean operators: `AND`, `OR`, `NOT`
    /// - Phrase queries: `"exact phrase"`
    /// - Fuzzy queries: `roam~2`
    /// - Range queries: `date:[2024-01-01 TO 2024-12-31]`
    ///
    /// **Vector search** — semantic similarity on vector fields:
    /// - `content:"cute kitten"` — vector search on a specific field
    /// - `content:python` — unquoted text vector search
    /// - `content:"cute kitten"^0.8` — with weight/boost
    ///
    /// **Hybrid search** — combine both in one query:
    /// - `title:hello content:"cute kitten"` — OR: union of lexical + vector results
    /// - `title:hello +content:"cute kitten"` — AND: only documents matching both
    query: String,

    /// Maximum number of results to return. Defaults to `10`.
    limit: Option<u32>,

    /// Number of results to skip for pagination. Defaults to `0`.
    offset: Option<u32>,

    /// Fusion algorithm for hybrid search as a JSON string. Only used when
    /// the query contains both lexical and vector clauses.
    ///
    /// Examples:
    /// - `{"rrf": {"k": 60.0}}` — Reciprocal Rank Fusion (default)
    /// - `{"weighted_sum": {"lexical_weight": 0.7, "vector_weight": 0.3}}`
    fusion: Option<String>,

    /// Per-field boost factors as a JSON string. Boosts the relevance score
    /// of matches in specific fields for lexical search.
    ///
    /// Example: `{"title": 2.0, "body": 1.0}`
    field_boosts: Option<String>,

    /// Request highlighted fragments per field, as a JSON string (Issue
    /// #1134). Highlighting follows this tool's lexical query; a
    /// vector-only query produces no highlights. Only `stored: true` text
    /// fields can be highlighted.
    ///
    /// Accepts either a bare field-name array or an object adding
    /// `HighlightConfig` knobs:
    /// - `["title", "body"]`
    /// - `{"fields": ["body"], "max_fragments": 2, "tag": "em"}`
    highlight: Option<String>,
}

/// Parameters for the `search_batch` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchBatchParams {
    /// Array of query strings, each in the laurus unified query DSL (same
    /// syntax as the `search` tool's `query`). All queries are executed in
    /// parallel on the server in a single round trip.
    queries: Vec<String>,

    /// Maximum number of results to return per query. Defaults to `10`.
    limit: Option<u32>,

    /// Number of results to skip per query for pagination. Defaults to `0`.
    offset: Option<u32>,

    /// Request highlighted fragments per field, as a JSON string — same
    /// format as the `search` tool's `highlight` parameter. Applied
    /// identically to every query in the batch (Issue #1134).
    highlight: Option<String>,
}

/// Parameters for the `add_field` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct AddFieldParams {
    /// The name of the new field to add.
    name: String,

    /// Field configuration as a JSON string.
    ///
    /// Uses the same externally-tagged serde representation as the schema.
    /// The variant name is the key.  Example:
    ///
    /// ```json
    /// {"Text": {"indexed": true, "stored": true}}
    /// {"Hnsw": {"dimension": 384, "distance": "Cosine"}}
    /// {"Integer": {}}
    /// ```
    field_option_json: String,
}

/// Parameters for the `delete_field` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct DeleteFieldParams {
    /// The name of the field to remove from the index schema.
    name: String,
}

/// Parameters for the `update_field` tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct UpdateFieldParams {
    /// The name of the field to update. Must already exist.
    name: String,

    /// The field's new configuration as a JSON string.
    ///
    /// Uses the same externally-tagged serde representation as the schema.
    /// Example:
    ///
    /// ```json
    /// {"Text": {"indexed": true, "stored": true, "analyzer": "english"}}
    /// {"Hnsw": {"dimension": 384, "m": 32, "ef_construction": 400}}
    /// ```
    field_option_json: String,

    /// Opt in to a change that requires rebuilding or discarding existing
    /// on-disk data (e.g. a text field's `analyzer`, or a vector field's
    /// `dimension`). Ignored for a metadata-only change. Defaults to
    /// `false`, which rejects such a change.
    #[serde(default)]
    reindex: bool,

    /// When `true`, classifies the change and reports it without applying
    /// anything. Defaults to `false`.
    #[serde(default)]
    dry_run: bool,
}

// ── Server struct ─────────────────────────────────────────────────────────────

/// MCP server that proxies tool calls to a laurus-server gRPC instance.
///
/// The gRPC channel is stored in [`Arc<RwLock<Option<Channel>>>`].  When
/// `None`, no connection has been established yet; use the `connect` tool to
/// connect to a running laurus-server.
#[derive(Clone)]
pub struct LaurusMcpServer {
    channel: Arc<RwLock<Option<Channel>>>,
    // Consumed by the `#[tool_handler]` macro expansion; not visible to dead-code analysis.
    #[allow(dead_code)]
    tool_router: ToolRouter<LaurusMcpServer>,
}

// ── Tool implementations ───────────────────────────────────────────────────────

#[tool_router]
impl LaurusMcpServer {
    fn new(channel: Option<Channel>) -> Self {
        Self {
            channel: Arc::new(RwLock::new(channel)),
            tool_router: Self::tool_router(),
        }
    }

    /// Return a tool-level error result (not a protocol error).
    fn tool_error(msg: impl Into<String>) -> CallToolResult {
        CallToolResult::error(vec![ContentBlock::text(msg.into())])
    }

    // ── Connection tool ───────────────────────────────────────────────────────

    /// Connect to a running laurus-server gRPC endpoint.
    ///
    /// Call this tool before using any other tools when the MCP server was
    /// started without an `--endpoint` argument.  You can also call it to
    /// switch to a different laurus-server at any time.
    #[tool(
        description = "Connect to a laurus-server gRPC endpoint (e.g. http://localhost:50051). Call this before using other tools if the server was started without --endpoint."
    )]
    async fn connect(
        &self,
        Parameters(params): Parameters<ConnectParams>,
    ) -> Result<CallToolResult, McpError> {
        match Channel::from_shared(params.endpoint.clone())
            .map_err(|e| format!("{e}"))
            .map(|b| b.connect_lazy())
        {
            Ok(ch) => {
                *self.channel.write().await = Some(ch);
                info!("Connected to laurus-server at {}", params.endpoint);
                Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                    "Connected to laurus-server at {}.",
                    params.endpoint
                ))]))
            }
            Err(e) => Ok(Self::tool_error(format!("Failed to connect: {e}"))),
        }
    }

    // ── Index tools ───────────────────────────────────────────────────────────

    /// Create a new search index with the provided schema.
    ///
    /// The schema describes the fields of the documents that will be indexed.
    #[tool(
        description = "Create a new search index with the provided schema. The schema_json must be a JSON string defining index fields (Text, Integer, Float, Boolean, DateTime, Hnsw, Flat, Ivf, etc.). An optional top-level \"dynamic_field_policy\" key controls how fields not listed in the schema are treated at ingest time: \"Strict\" rejects them, \"Dynamic\" (default) infers a type and adds the field (note: Integer fields silently truncate incoming float values), \"Ignore\" drops them silently. Call this before add_document or search if the index does not exist yet."
    )]
    async fn create_index(
        &self,
        Parameters(params): Parameters<CreateIndexParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        let laurus_schema: laurus::Schema = match serde_json::from_str(&params.schema_json) {
            Ok(s) => s,
            Err(e) => {
                return Ok(Self::tool_error(format!(
                    "Failed to parse schema JSON: {e}"
                )));
            }
        };

        let proto_schema = laurus_server::convert::schema::to_proto(&laurus_schema);
        let request = CreateIndexRequest {
            schema: Some(proto_schema),
        };

        match IndexServiceClient::new(channel).create_index(request).await {
            Ok(_) => Ok(CallToolResult::success(vec![ContentBlock::text(
                "Index created successfully.",
            )])),
            Err(e) => Ok(Self::tool_error(format!("Failed to create index: {e}"))),
        }
    }

    /// Get statistics for the open index.
    #[tool(
        description = "Get statistics for the current search index, including document count and vector field information."
    )]
    async fn get_stats(&self) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        match IndexServiceClient::new(channel)
            .get_index(GetIndexRequest {})
            .await
        {
            Ok(resp) => {
                let r = resp.into_inner();
                let output = json!({
                    "document_count": r.document_count,
                    "vector_fields": r.vector_fields.keys().collect::<Vec<_>>(),
                });
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    output.to_string(),
                )]))
            }
            Err(e) => Ok(Self::tool_error(format!("Failed to get index stats: {e}"))),
        }
    }

    /// Get the current index schema.
    #[tool(
        description = "Get the current index schema, including all field definitions and their configurations. Returns the schema as a JSON object."
    )]
    async fn get_schema(&self) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        match IndexServiceClient::new(channel)
            .get_schema(GetSchemaRequest {})
            .await
        {
            Ok(resp) => {
                let r = resp.into_inner();
                match r.schema {
                    Some(proto_schema) => {
                        match laurus_server::convert::schema::from_proto(&proto_schema) {
                            Ok(schema) => {
                                let json = serde_json::to_value(&schema).unwrap_or_default();
                                Ok(CallToolResult::success(vec![ContentBlock::text(
                                    json.to_string(),
                                )]))
                            }
                            Err(e) => {
                                Ok(Self::tool_error(format!("Failed to convert schema: {e}")))
                            }
                        }
                    }
                    None => Ok(Self::tool_error("No schema returned by server.")),
                }
            }
            Err(e) => Ok(Self::tool_error(format!("Failed to get schema: {e}"))),
        }
    }

    /// Dynamically add a new field to the current index.
    #[tool(
        description = "Dynamically add a new field to an existing index. The field_option_json must be a JSON string describing the field type and options (e.g. '{\"Text\": {\"indexed\": true, \"stored\": true}}', '{\"Hnsw\": {\"dimension\": 384}}', '{\"Integer\": {}}'). Returns the updated schema."
    )]
    async fn add_field(
        &self,
        Parameters(params): Parameters<AddFieldParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        let field_option: laurus::FieldOption =
            match serde_json::from_str(&params.field_option_json) {
                Ok(fo) => fo,
                Err(e) => {
                    return Ok(Self::tool_error(format!(
                        "Failed to parse field_option_json: {e}"
                    )));
                }
            };

        let proto_field_option =
            laurus_server::convert::schema::field_option_to_proto(&field_option);
        let request = AddFieldRequest {
            name: params.name.clone(),
            field_option: Some(proto_field_option),
        };

        match IndexServiceClient::new(channel).add_field(request).await {
            Ok(resp) => {
                let r = resp.into_inner();
                let output = if let Some(schema) = r.schema {
                    let field_names: Vec<&String> = schema.fields.keys().collect();
                    json!({
                        "message": format!("Field '{}' added successfully.", params.name),
                        "fields": field_names,
                    })
                } else {
                    json!({
                        "message": format!("Field '{}' added successfully.", params.name),
                    })
                };
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    output.to_string(),
                )]))
            }
            Err(e) => Ok(Self::tool_error(format!("Failed to add field: {e}"))),
        }
    }

    /// Remove a field from the index schema.
    #[tool(
        description = "Remove a field from the index schema. The field will no longer be available for indexing or searching, but existing data in the index is not deleted. Returns the updated schema."
    )]
    async fn delete_field(
        &self,
        Parameters(params): Parameters<DeleteFieldParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        let request = DeleteFieldRequest {
            name: params.name.clone(),
        };

        match IndexServiceClient::new(channel).delete_field(request).await {
            Ok(resp) => {
                let r = resp.into_inner();
                let output = if let Some(schema) = r.schema {
                    let field_names: Vec<&String> = schema.fields.keys().collect();
                    json!({
                        "message": format!("Field '{}' deleted successfully.", params.name),
                        "fields": field_names,
                    })
                } else {
                    json!({
                        "message": format!("Field '{}' deleted successfully.", params.name),
                    })
                };
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    output.to_string(),
                )]))
            }
            Err(e) => Ok(Self::tool_error(format!("Failed to delete field: {e}"))),
        }
    }

    /// Change an existing field's type/options.
    #[tool(
        description = "Change an existing field's type/options (e.g. a text field's analyzer, or a vector field's dimension/m/ef_construction). field_option_json is a JSON string in the same format as add_field's. Metadata-only changes (e.g. default_ef_search) apply immediately; a change that requires rebuilding or discarding existing on-disk data is rejected unless reindex is true. Set dry_run to true to see how a change would be classified (metadata_only / reindex / destructive) without applying it. Returns the classification and the updated schema."
    )]
    async fn update_field(
        &self,
        Parameters(params): Parameters<UpdateFieldParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        let field_option: laurus::FieldOption =
            match serde_json::from_str(&params.field_option_json) {
                Ok(fo) => fo,
                Err(e) => {
                    return Ok(Self::tool_error(format!(
                        "Failed to parse field_option_json: {e}"
                    )));
                }
            };

        let proto_field_option =
            laurus_server::convert::schema::field_option_to_proto(&field_option);
        let request = UpdateFieldRequest {
            name: params.name.clone(),
            field_option: Some(proto_field_option),
            reindex: params.reindex,
            dry_run: params.dry_run,
        };

        match IndexServiceClient::new(channel).update_field(request).await {
            Ok(resp) => {
                let r = resp.into_inner();
                let classification = match FieldChangeKind::try_from(r.classification) {
                    Ok(FieldChangeKind::MetadataOnly) => Some("metadata_only"),
                    Ok(FieldChangeKind::Reindex) => Some("reindex"),
                    Ok(FieldChangeKind::Destructive) => Some("destructive"),
                    Ok(FieldChangeKind::Unspecified) | Err(_) => None,
                };
                let output = if let Some(schema) = r.schema {
                    let field_names: Vec<&String> = schema.fields.keys().collect();
                    json!({
                        "message": if params.dry_run {
                            format!("Dry run: change to field '{}' classified.", params.name)
                        } else {
                            format!("Field '{}' updated successfully.", params.name)
                        },
                        "classification": classification,
                        "fields": field_names,
                    })
                } else {
                    json!({
                        "message": format!("Field '{}' updated successfully.", params.name),
                        "classification": classification,
                    })
                };
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    output.to_string(),
                )]))
            }
            Err(e) => Ok(Self::tool_error(format!("Failed to update field: {e}"))),
        }
    }

    // ── Document tools ────────────────────────────────────────────────────────

    /// Put (upsert) a document into the index.
    ///
    /// If a document with the same ID already exists, all its chunks are
    /// deleted before the new document is indexed.
    #[tool(
        description = "Put (upsert) a document into the index. If a document with the same id already exists, it is replaced. Call commit after adding documents to persist changes."
    )]
    async fn put_document(
        &self,
        Parameters(params): Parameters<PutDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        let doc = match convert::json_to_document(&json!({"fields": params.fields})) {
            Ok(d) => d,
            Err(e) => {
                return Ok(Self::tool_error(format!("Invalid document: {e}")));
            }
        };

        match DocumentServiceClient::new(channel)
            .put_document(PutDocumentRequest {
                id: params.id.clone(),
                document: Some(doc),
            })
            .await
        {
            Ok(_) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "Document '{}' put (upserted). Call commit to persist changes.",
                params.id
            ))])),
            Err(e) => Ok(Self::tool_error(format!("Failed to put document: {e}"))),
        }
    }

    /// Add a document as a new chunk (append, never deletes existing).
    ///
    /// Multiple chunks can share the same ID, which is useful for indexing
    /// parts of a large document (e.g. paragraphs or pages) separately.
    #[tool(
        description = "Add a document as a new chunk to the index. Unlike put_document, this appends without deleting existing documents with the same id. Useful for splitting large documents into chunks. Call commit after adding documents to persist changes."
    )]
    async fn add_document(
        &self,
        Parameters(params): Parameters<AddDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        let doc = match convert::json_to_document(&json!({"fields": params.fields})) {
            Ok(d) => d,
            Err(e) => {
                return Ok(Self::tool_error(format!("Invalid document: {e}")));
            }
        };

        match DocumentServiceClient::new(channel)
            .add_document(AddDocumentRequest {
                id: params.id.clone(),
                document: Some(doc),
            })
            .await
        {
            Ok(_) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "Document '{}' added as chunk. Call commit to persist changes.",
                params.id
            ))])),
            Err(e) => Ok(Self::tool_error(format!("Failed to add document: {e}"))),
        }
    }

    /// Convert the bulk tools' JSON entries — each a `{"id": "...", "fields":
    /// {...}}` object — into proto `DocumentEntry` values, naming the
    /// offending position on the first invalid entry.
    fn bulk_entries(documents: Vec<Value>) -> Result<Vec<DocumentEntry>, String> {
        documents
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let id = entry
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("documents[{index}]: missing string \"id\""))?
                    .to_string();
                let document = convert::json_to_document(entry)
                    .map_err(|e| format!("documents[{index}]: {e}"))?;
                Ok(DocumentEntry {
                    id,
                    document: Some(document),
                })
            })
            .collect()
    }

    /// Batched upsert of documents in one round trip.
    ///
    /// Entries are applied sequentially, in input order, with one WAL fsync
    /// for the whole batch; a failure aborts at the offending entry without
    /// rolling back the already-applied prefix (retrying is idempotent).
    #[tool(
        description = "Put (upsert) MANY documents in one call. Pass documents as an array of {\"id\": \"...\", \"fields\": {...}} entries; they are applied in order (duplicate ids dedup, last wins) with one WAL fsync for the whole batch, which is much faster than calling put_document per document. Call commit afterwards to persist changes."
    )]
    async fn put_documents(
        &self,
        Parameters(params): Parameters<BulkDocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        let documents = match Self::bulk_entries(params.documents) {
            Ok(entries) => entries,
            Err(e) => return Ok(Self::tool_error(format!("Invalid batch: {e}"))),
        };

        match DocumentServiceClient::new(channel)
            .put_documents(PutDocumentsRequest { documents })
            .await
        {
            Ok(resp) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "{} documents put (upserted). Call commit to persist changes.",
                resp.into_inner().applied
            ))])),
            Err(e) => Ok(Self::tool_error(format!("Failed to put documents: {e}"))),
        }
    }

    /// Batched chunk append of documents in one round trip.
    ///
    /// Like `put_documents` but never deletes existing documents, so a batch
    /// may repeat an id to add multiple chunks of one logical document.
    #[tool(
        description = "Add MANY documents as new chunks in one call. Pass documents as an array of {\"id\": \"...\", \"fields\": {...}} entries; unlike put_documents, existing documents are never deleted, so repeating an id adds multiple chunks. One WAL fsync covers the whole batch. Call commit afterwards to persist changes."
    )]
    async fn add_documents(
        &self,
        Parameters(params): Parameters<BulkDocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        let documents = match Self::bulk_entries(params.documents) {
            Ok(entries) => entries,
            Err(e) => return Ok(Self::tool_error(format!("Invalid batch: {e}"))),
        };

        match DocumentServiceClient::new(channel)
            .add_documents(AddDocumentsRequest { documents })
            .await
        {
            Ok(resp) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "{} documents added as chunks. Call commit to persist changes.",
                resp.into_inner().applied
            ))])),
            Err(e) => Ok(Self::tool_error(format!("Failed to add documents: {e}"))),
        }
    }

    /// Get all stored documents for a given ID.
    #[tool(
        description = "Retrieve all stored documents (including chunks) by external ID. Returns a JSON array of documents matching the ID."
    )]
    async fn get_documents(
        &self,
        Parameters(params): Parameters<GetDocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        match DocumentServiceClient::new(channel)
            .get_documents(GetDocumentsRequest {
                id: params.id.clone(),
            })
            .await
        {
            Ok(resp) => {
                let json_docs: Vec<Value> = resp
                    .into_inner()
                    .documents
                    .iter()
                    .map(convert::document_to_json)
                    .collect();
                let output = json!({
                    "id": params.id,
                    "documents": json_docs,
                });
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    output.to_string(),
                )]))
            }
            Err(e) => Ok(Self::tool_error(format!("Failed to get documents: {e}"))),
        }
    }

    /// Delete all documents (including chunks) with the given external ID.
    #[tool(
        description = "Delete all documents and chunks sharing the given external ID from the index. Call commit to persist changes."
    )]
    async fn delete_documents(
        &self,
        Parameters(params): Parameters<DeleteDocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        match DocumentServiceClient::new(channel)
            .delete_documents(DeleteDocumentsRequest {
                id: params.id.clone(),
            })
            .await
        {
            Ok(_) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "Documents '{}' deleted. Call commit to persist changes.",
                params.id
            ))])),
            Err(e) => Ok(Self::tool_error(format!("Failed to delete documents: {e}"))),
        }
    }

    /// Commit pending changes to disk.
    #[tool(
        description = "Commit pending changes to disk. Must be called after put_document, add_document, or delete_documents to make changes searchable and durable."
    )]
    async fn commit(&self) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        match DocumentServiceClient::new(channel)
            .commit(CommitRequest {})
            .await
        {
            Ok(_) => Ok(CallToolResult::success(vec![ContentBlock::text(
                "Changes committed successfully.",
            )])),
            Err(e) => Ok(Self::tool_error(format!("Failed to commit: {e}"))),
        }
    }

    // ── Search tools ──────────────────────────────────────────────────────────

    /// Search documents using the laurus unified query DSL.
    #[tool(
        description = "Search documents using the laurus unified query DSL. Supports three modes: (1) Lexical search: term queries (title:hello), boolean operators (AND, OR, NOT), phrase queries (\"exact phrase\"), fuzzy queries (roam~2), range queries (field:[from TO to]). (2) Vector search: ~\"text\" syntax for semantic similarity (content:~\"cute kitten\", ~\"text\"^0.8). (3) Hybrid search: mix both in one query (title:hello content:~\"cute kitten\"). Returns JSON with total count and array of results (id, score, fields, highlights when requested via the highlight parameter)."
    )]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        // Parse optional fusion algorithm
        let fusion = if let Some(ref fusion_json) = params.fusion {
            match convert::json_to_fusion_algorithm(fusion_json) {
                Ok(f) => Some(f),
                Err(e) => return Ok(Self::tool_error(format!("Invalid fusion JSON: {e}"))),
            }
        } else {
            None
        };

        // Parse optional field boosts
        let field_boosts = if let Some(ref boosts_json) = params.field_boosts {
            match convert::json_to_field_boosts(boosts_json) {
                Ok(b) => b,
                Err(e) => return Ok(Self::tool_error(format!("Invalid field_boosts JSON: {e}"))),
            }
        } else {
            std::collections::HashMap::new()
        };

        // Parse optional highlight request (Issue #1134)
        let highlight = if let Some(ref highlight_json) = params.highlight {
            match convert::json_to_highlight_params(highlight_json) {
                Ok(h) => Some(h),
                Err(e) => return Ok(Self::tool_error(format!("Invalid highlight JSON: {e}"))),
            }
        } else {
            None
        };

        let request = SearchRequest {
            query: params.query,
            limit: params.limit.unwrap_or(10),
            offset: params.offset.unwrap_or(0),
            fusion,
            field_boosts,
            highlight,
            ..Default::default()
        };

        match SearchServiceClient::new(channel).search(request).await {
            Ok(resp) => {
                let r = resp.into_inner();
                let json_results: Vec<Value> = r
                    .results
                    .iter()
                    .map(convert::search_result_to_json)
                    .collect();

                let output = json!({
                    "total": r.total_hits,
                    "results": json_results,
                });
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    output.to_string(),
                )]))
            }
            Err(e) => Ok(Self::tool_error(format!("Search failed: {e}"))),
        }
    }

    #[tool(
        description = "Execute multiple independent searches in a single round trip. Takes an array of query strings (each in the laurus unified query DSL, same syntax as the search tool) and runs them in parallel on the server. The same limit, offset and highlight settings apply to every query. Returns JSON with a `batch` array; batch[i] holds the total count and results (id, score, fields, highlights when requested) for queries[i], in input order. Useful for agents issuing several sub-queries per turn."
    )]
    async fn search_batch(
        &self,
        Parameters(params): Parameters<SearchBatchParams>,
    ) -> Result<CallToolResult, McpError> {
        let channel = match self.channel.read().await.clone() {
            Some(ch) => ch,
            None => {
                return Ok(Self::tool_error(
                    "Not connected. Call the connect tool first.",
                ));
            }
        };

        if params.queries.is_empty() {
            let output = json!({ "batch": [] });
            return Ok(CallToolResult::success(vec![ContentBlock::text(
                output.to_string(),
            )]));
        }

        // Parse optional highlight request (Issue #1134), applied identically
        // to every query in the batch.
        let highlight = if let Some(ref highlight_json) = params.highlight {
            match convert::json_to_highlight_params(highlight_json) {
                Ok(h) => Some(h),
                Err(e) => return Ok(Self::tool_error(format!("Invalid highlight JSON: {e}"))),
            }
        } else {
            None
        };

        let limit = params.limit.unwrap_or(10);
        let offset = params.offset.unwrap_or(0);
        let queries: Vec<SearchRequest> = params
            .queries
            .into_iter()
            .map(|query| SearchRequest {
                query,
                limit,
                offset,
                highlight: highlight.clone(),
                ..Default::default()
            })
            .collect();

        let request = SearchBatchRequest { queries };

        match SearchServiceClient::new(channel)
            .search_batch(request)
            .await
        {
            Ok(resp) => {
                let r = resp.into_inner();
                let batch: Vec<Value> = r
                    .results
                    .iter()
                    .map(|per_query| {
                        let json_results: Vec<Value> = per_query
                            .results
                            .iter()
                            .map(convert::search_result_to_json)
                            .collect();
                        json!({
                            "total": per_query.total_hits,
                            "results": json_results,
                        })
                    })
                    .collect();

                let output = json!({ "batch": batch });
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    output.to_string(),
                )]))
            }
            Err(e) => Ok(Self::tool_error(format!("Batch search failed: {e}"))),
        }
    }
}

// ── ServerHandler impl ────────────────────────────────────────────────────────

#[tool_handler]
impl ServerHandler for LaurusMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
            .with_instructions(
                "Laurus search engine MCP server (gRPC client). \
             Tools: connect, create_index, get_stats, get_schema, add_field, delete_field, \
             update_field, put_document, add_document, put_documents, add_documents, \
             get_documents, delete_documents, commit, search. \
             Start by calling connect(endpoint) to connect to a running laurus-server, \
             then use the other tools to manage and search the index."
                    .to_string(),
            )
    }
}

// ── Public entry point ─────────────────────────────────────────────────────────

/// Start the MCP server on stdio.
///
/// If `endpoint` is provided, connects to the laurus-server immediately.
/// Otherwise the server starts without a connection; use the `connect` tool
/// to connect to a running laurus-server before using other tools.
///
/// This function runs until stdin is closed or an unrecoverable error occurs.
///
/// # Arguments
///
/// * `endpoint` - Optional gRPC endpoint URL (e.g. `http://localhost:50051`).
///
/// # Errors
///
/// Returns an error if the server transport fails to start or encounters a
/// fatal runtime error.
pub async fn run(endpoint: Option<&str>) -> anyhow::Result<()> {
    let channel = if let Some(ep) = endpoint {
        info!("Connecting to laurus-server at {ep}");
        let ch = Channel::from_shared(ep.to_string())
            .context("Invalid endpoint URI")?
            .connect_lazy();
        Some(ch)
    } else {
        info!("No endpoint specified. Use the connect tool to connect to a laurus-server.");
        None
    };

    let server = LaurusMcpServer::new(channel);
    let transport = (tokio::io::stdin(), tokio::io::stdout());
    let service = server
        .serve(transport)
        .await
        .context("Failed to start MCP server")?;

    service.waiting().await.context("MCP server error")?;

    Ok(())
}
