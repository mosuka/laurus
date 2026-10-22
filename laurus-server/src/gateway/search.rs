//! Search endpoints (unary + SSE streaming + batch).

use std::convert::Infallible;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio_stream::StreamExt;

use super::GatewayState;
use super::convert;
use super::error::{BadRequest, GatewayError};
use crate::proto::laurus::v1::SearchBatchRequest;

/// `POST /v1/search` — Executes a search and returns all results at once.
pub async fn search(
    State(mut state): State<GatewayState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let request =
        convert::json_to_proto_search_request(&body).map_err(|e| BadRequest(e).into_response())?;

    let response = state
        .search_client
        .search(request)
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(convert::proto_search_response_to_json(
        &response.into_inner(),
    )))
}

/// `POST /v1/search/stream` — Executes a search and returns results incrementally via SSE.
pub async fn search_stream(
    State(mut state): State<GatewayState>,
    Json(body): Json<Value>,
) -> Response {
    let request = match convert::json_to_proto_search_request(&body) {
        Ok(r) => r,
        Err(e) => return BadRequest(e).into_response(),
    };

    let response = match state.search_client.search_stream(request).await {
        Ok(r) => r,
        Err(s) => return GatewayError(s).into_response(),
    };

    let grpc_stream = response.into_inner();

    let sse_stream = grpc_stream.map(|result| -> Result<Event, Infallible> {
        match result {
            Ok(search_result) => {
                let json = convert::proto_search_result_to_json(&search_result);
                let data = serde_json::to_string(&json).unwrap_or_default();
                Ok(Event::default().data(data))
            }
            Err(status) => {
                let error_json = json!({
                    "error": {
                        "code": status.code() as i32,
                        "message": status.message(),
                    }
                });
                let data = serde_json::to_string(&error_json).unwrap_or_default();
                Ok(Event::default().event("error").data(data))
            }
        }
    });

    Sse::new(sse_stream).into_response()
}

/// `POST /v1/search/batch` — Executes multiple independent searches in one
/// round trip and returns all responses in input order.
///
/// Request body: `{ "queries": [SearchRequest, SearchRequest, ...] }`.
/// Each entry follows the same schema as `POST /v1/search`'s body. Empty
/// input returns `{ "results": [] }` without invoking the gRPC service. An
/// entry that cannot be converted fails the whole batch with a 400 whose
/// message starts with its position (`queries[1]: ...`), as for
/// `POST /v1/documents:bulk` (Issue #1524).
///
/// Issue [#716](https://github.com/mosuka/laurus/issues/716) Phase 3a of
/// [#648](https://github.com/mosuka/laurus/issues/648).
pub async fn search_batch(
    State(mut state): State<GatewayState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let queries_value = body
        .get("queries")
        .ok_or_else(|| BadRequest("missing `queries` field".to_string()).into_response())?;
    let queries_array = queries_value
        .as_array()
        .ok_or_else(|| BadRequest("`queries` must be an array".to_string()).into_response())?;

    let queries = queries_array
        .iter()
        .enumerate()
        .map(|(index, query)| {
            convert::json_to_proto_search_request(query)
                .map_err(|e| format!("queries[{index}]: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| BadRequest(e).into_response())?;

    let response = state
        .search_client
        .search_batch(SearchBatchRequest { queries })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    let results: Vec<Value> = response
        .into_inner()
        .results
        .iter()
        .map(convert::proto_search_response_to_json)
        .collect();

    Ok(Json(json!({ "results": results })))
}
