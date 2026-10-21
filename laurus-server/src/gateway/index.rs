//! Index management endpoints.

use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::GatewayState;
use super::convert;
use super::error::{BadRequest, GatewayError};
use crate::proto::laurus::v1;

/// `POST /v1/index` — Creates a new index.
pub async fn create(
    State(mut state): State<GatewayState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let schema_json = body
        .get("schema")
        .ok_or_else(|| BadRequest("missing \"schema\" key".to_string()).into_response())?;

    let schema =
        convert::json_to_proto_schema(schema_json).map_err(|e| BadRequest(e).into_response())?;

    state
        .index_client
        .create_index(v1::CreateIndexRequest {
            schema: Some(schema),
        })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}

/// `GET /v1/index` — Returns index statistics.
pub async fn get_index(State(mut state): State<GatewayState>) -> Result<Json<Value>, Response> {
    let response = state
        .index_client
        .get_index(v1::GetIndexRequest {})
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    let inner = response.into_inner();
    let vector_fields: serde_json::Map<String, Value> = inner
        .vector_fields
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                json!({
                    "vector_count": v.vector_count,
                    "dimension": v.dimension,
                }),
            )
        })
        .collect();

    Ok(Json(json!({
        "document_count": inner.document_count,
        "vector_fields": vector_fields,
    })))
}

/// `GET /v1/schema` — Returns the current schema.
pub async fn get_schema(State(mut state): State<GatewayState>) -> Result<Json<Value>, Response> {
    let response = state
        .index_client
        .get_schema(v1::GetSchemaRequest {})
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    let inner = response.into_inner();
    let schema_json = inner
        .schema
        .as_ref()
        .map(convert::proto_schema_to_json)
        .unwrap_or(Value::Null);

    Ok(Json(json!({ "schema": schema_json })))
}

/// `POST /v1/schema/fields` — Adds a new field to the index schema.
pub async fn add_field(
    State(mut state): State<GatewayState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let name = body
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| BadRequest("missing or invalid \"name\" key".to_string()).into_response())?
        .to_string();

    let field_option_json = body
        .get("field_option")
        .ok_or_else(|| BadRequest("missing \"field_option\" key".to_string()).into_response())?;

    let field_option = convert::json_to_proto_field_option(field_option_json)
        .map_err(|e| BadRequest(e).into_response())?;

    let response = state
        .index_client
        .add_field(v1::AddFieldRequest {
            name,
            field_option: Some(field_option),
        })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    let inner = response.into_inner();
    let schema_json = inner
        .schema
        .as_ref()
        .map(convert::proto_schema_to_json)
        .unwrap_or(Value::Null);

    Ok(Json(json!({ "schema": schema_json })))
}

/// `DELETE /v1/schema/fields/:name` — Removes a field from the index schema.
pub async fn delete_field(
    State(mut state): State<GatewayState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<Value>, Response> {
    let response = state
        .index_client
        .delete_field(v1::DeleteFieldRequest { name })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    let inner = response.into_inner();
    let schema_json = inner
        .schema
        .as_ref()
        .map(convert::proto_schema_to_json)
        .unwrap_or(Value::Null);

    Ok(Json(json!({ "schema": schema_json })))
}

/// `PATCH /v1/schema/fields/:name` — Changes an existing field's
/// type/options, optionally rebuilding or discarding its on-disk data.
pub async fn update_field(
    State(mut state): State<GatewayState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let field_option_json = body
        .get("field_option")
        .ok_or_else(|| BadRequest("missing \"field_option\" key".to_string()).into_response())?;

    let field_option = convert::json_to_proto_field_option(field_option_json)
        .map_err(|e| BadRequest(e).into_response())?;

    // Both default to `false`, matching `UpdateFieldOptions::default()`.
    // Anything but `true`, `false` or `null` is a 400 rather than `false`,
    // so `"dry_run": "true"` cannot apply the change (Issue #1511).
    let flag = |key: &str| {
        convert::json_optional_bool(&body, key)
            .map(Option::unwrap_or_default)
            .map_err(|e| BadRequest(e).into_response())
    };
    let reindex = flag("reindex")?;
    let dry_run = flag("dry_run")?;

    let response = state
        .index_client
        .update_field(v1::UpdateFieldRequest {
            name,
            field_option: Some(field_option),
            reindex,
            dry_run,
        })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    let inner = response.into_inner();
    let schema_json = inner
        .schema
        .as_ref()
        .map(convert::proto_schema_to_json)
        .unwrap_or(Value::Null);
    let classification = convert::proto_field_change_kind_to_json(inner.classification);

    Ok(Json(json!({
        "classification": classification,
        "schema": schema_json,
    })))
}
