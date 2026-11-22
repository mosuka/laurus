//! Document CRUD endpoints.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::GatewayState;
use super::convert;
use super::error::{BadRequest, GatewayError};
use crate::proto::laurus::v1;

/// `PUT /v1/documents/:id` — Inserts or replaces a document.
///
/// Body shape: `{"fields": {...}}` — the same document JSON shape used by
/// laurus-cli and laurus-mcp.
pub async fn put_document(
    State(mut state): State<GatewayState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let document =
        convert::json_to_proto_document_body(&body).map_err(|e| BadRequest(e).into_response())?;

    state
        .document_client
        .put_document(v1::PutDocumentRequest {
            id,
            document: Some(document),
        })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}

/// `POST /v1/documents/:id` — Adds a document as a chunk.
///
/// Body shape: `{"fields": {...}}` — the same document JSON shape used by
/// laurus-cli and laurus-mcp.
pub async fn add_document(
    State(mut state): State<GatewayState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let document =
        convert::json_to_proto_document_body(&body).map_err(|e| BadRequest(e).into_response())?;

    state
        .document_client
        .add_document(v1::AddDocumentRequest {
            id,
            document: Some(document),
        })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}

/// `POST /v1/documents:bulk?mode=put|add` — Batched document ingestion.
///
/// Body shape: `{"documents": [{"id": "...", "fields": {...}}, ...]}` — the
/// same `{"id", "fields"}` entry shape as laurus-cli's bulk JSONL. Entries
/// are applied sequentially, in input order, with one WAL fsync for the
/// whole batch (see the core `Engine::put_documents` semantics): `mode=put`
/// (the default) upserts — duplicate ids within one batch dedup, last
/// occurrence wins — while `mode=add` appends chunks, so repeated ids
/// accumulate. Fails fast at the first entry that cannot be applied;
/// already-applied entries are not rolled back, and the error carries the
/// failing position, so retrying the batch (or its suffix) is idempotent.
/// Responds `{"applied": N}`.
pub async fn bulk_documents(
    State(mut state): State<GatewayState>,
    Query(params): Query<std::collections::HashMap<String, String>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let mode = params.get("mode").map(String::as_str).unwrap_or("put");
    if mode != "put" && mode != "add" {
        return Err(
            BadRequest(format!("invalid mode '{mode}' (expected 'put' or 'add')")).into_response(),
        );
    }

    let documents = convert::json_to_proto_document_entries(&body)
        .map_err(|e| BadRequest(e).into_response())?;

    let applied = if mode == "add" {
        state
            .document_client
            .add_documents(v1::AddDocumentsRequest { documents })
            .await
            .map_err(|s| GatewayError(s).into_response())?
            .into_inner()
            .applied
    } else {
        state
            .document_client
            .put_documents(v1::PutDocumentsRequest { documents })
            .await
            .map_err(|s| GatewayError(s).into_response())?
            .into_inner()
            .applied
    };

    Ok(Json(json!({ "applied": applied })))
}

/// `GET /v1/documents/:id` — Retrieves documents with the specified ID.
pub async fn get_documents(
    State(mut state): State<GatewayState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, Response> {
    let response = state
        .document_client
        .get_documents(v1::GetDocumentsRequest { id })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    let inner = response.into_inner();
    let documents: Vec<Value> = inner
        .documents
        .iter()
        .map(convert::proto_document_to_json)
        .collect();

    Ok(Json(json!({ "documents": documents })))
}

/// `DELETE /v1/documents/:id` — Deletes documents with the specified ID.
pub async fn delete_documents(
    State(mut state): State<GatewayState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, Response> {
    state
        .document_client
        .delete_documents(v1::DeleteDocumentsRequest { id })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}

/// `POST /v1/commit` — Persists pending changes.
pub async fn commit(State(mut state): State<GatewayState>) -> Result<Json<Value>, Response> {
    state
        .document_client
        .commit(v1::CommitRequest {})
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}

/// `POST /v1/flush_wal` — Forces buffered WAL records durable without a full
/// commit (group-commit on-demand barrier; near no-op under per-record sync).
pub async fn flush_wal(State(mut state): State<GatewayState>) -> Result<Json<Value>, Response> {
    state
        .document_client
        .flush_wal(v1::FlushWalRequest {})
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}
