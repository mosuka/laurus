//! The HTTP gateway accepts only `true`, `false` and `null` for a boolean
//! key (Issue #1511). Anything else is a 400 from the gateway itself, and
//! nothing is written or searched. Such a value used to be read as omitted,
//! so `"dry_run": "true"` applied a field change.
//!
//! The gateway forwards each call to the gRPC services and maps the status
//! they return, so these requests go through both, over a gRPC connection
//! to the index and search services on an ephemeral port.

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use tokio::sync::RwLock;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Endpoint, Server};
use tower::ServiceExt;

use laurus::{CommitPolicy, WalSyncPolicy};
use laurus_server::config::DEFAULT_MAX_RESULT_WINDOW;
use laurus_server::gateway::{self, GatewayState};
use laurus_server::proto::laurus::v1::index_service_server::IndexServiceServer;
use laurus_server::proto::laurus::v1::search_service_server::SearchServiceServer;
use laurus_server::service::index::IndexService;
use laurus_server::service::search::SearchService;

/// Serve an `IndexService` and a `SearchService` over `data_dir`, with no
/// index yet, on an ephemeral port, and return the gateway router in front
/// of them.
async fn gateway(data_dir: PathBuf) -> Router {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine = Arc::new(RwLock::new(None));
    let index = IndexService {
        engine: engine.clone(),
        data_dir,
        wal_policy: WalSyncPolicy::PerRecord,
        commit_policy: CommitPolicy::Manual,
    };
    let search = SearchService {
        engine,
        max_result_window: DEFAULT_MAX_RESULT_WINDOW,
    };
    tokio::spawn(
        Server::builder()
            .add_service(IndexServiceServer::new(index))
            .add_service(SearchServiceServer::new(search))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_lazy();
    gateway::create_router(GatewayState::new(channel))
}

/// Send `body` as JSON to `uri` and return the status and the body: its
/// JSON, or its text when it is not JSON (a search stream).
async fn send(router: &Router, method: &str, uri: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

/// Create an index with `fields` through the gateway.
async fn create_index(router: &Router, fields: Value) {
    let schema = json!({"schema": {"fields": fields}});
    let (status, body) = send(router, "POST", "/v1/index", schema).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// The schema `GET /v1/schema` returns.
async fn schema(router: &Router) -> Value {
    let (status, body) = send(router, "GET", "/v1/schema", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

/// Values of a boolean key that are not `true`, `false` or `null`.
fn non_booleans() -> [Value; 4] {
    [json!("true"), json!("false"), json!(1), json!(0)]
}

/// The boolean keys of a `text` field option.
const TEXT_KEYS: [&str; 5] = [
    "indexed",
    "stored",
    "multi_valued",
    "term_vectors",
    "doc_values",
];

/// The case in the Issue: a `PATCH /v1/schema/fields/:name` with a
/// non-boolean `dry_run` or `reindex` is a 400, and the field keeps its
/// `m`. `"dry_run": "true"` used to be read as `false`, which rebuilt the
/// field with the new `m`.
#[tokio::test]
async fn update_field_answers_400_for_a_non_boolean_reindex_or_dry_run() {
    // Every request that was not rejected as expected, so a failure lists
    // all of them rather than the first.
    let mut wrong = Vec::new();
    for key in ["dry_run", "reindex"] {
        for bad in non_booleans() {
            let dir = tempfile::tempdir().unwrap();
            let router = gateway(dir.path().join("data")).await;
            create_index(&router, json!({"vec": {"hnsw": {"dimension": 8, "m": 32}}})).await;
            let before = schema(&router).await;

            let mut update = json!({
                "field_option": {"hnsw": {"dimension": 8, "m": 16}},
                "reindex": true,
            });
            update[key] = bad.clone();
            let (status, body) = send(&router, "PATCH", "/v1/schema/fields/vec", update).await;
            let after = schema(&router).await;
            let expected = json!(format!("{key} must be true or false"));
            if status != StatusCode::BAD_REQUEST || body["error"]["message"] != expected {
                wrong.push(format!("{key} = {bad}: {status} {body}"));
            }
            if after != before {
                wrong.push(format!("{key} = {bad}: the schema changed to {after}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `POST /v1/index` with a non-boolean field option key is a 400 that
/// names the field and the key, and creates nothing.
#[tokio::test]
async fn create_index_answers_400_for_a_non_boolean_field_option_key() {
    let mut wrong = Vec::new();
    for key in TEXT_KEYS {
        for bad in non_booleans() {
            let dir = tempfile::tempdir().unwrap();
            let data_dir = dir.path().join("data");
            let router = gateway(data_dir.clone()).await;

            let schema = json!({"schema": {"fields": {"title": {"text": {key: bad}}}}});
            let (status, body) = send(&router, "POST", "/v1/index", schema).await;
            let expected = json!(format!("field \"title\": {key} must be true or false"));
            if status != StatusCode::BAD_REQUEST || body["error"]["message"] != expected {
                wrong.push(format!("{key} = {bad}: {status} {body}"));
            }
            if data_dir.exists() {
                wrong.push(format!("{key} = {bad}: an index was created"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `POST /v1/schema/fields` with a non-boolean field option key is a 400,
/// and the schema is unchanged.
#[tokio::test]
async fn add_field_answers_400_for_a_non_boolean_field_option_key() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    create_index(&router, json!({"title": {"text": {"indexed": true}}})).await;
    let before = schema(&router).await;

    let mut wrong = Vec::new();
    for key in TEXT_KEYS {
        for bad in non_booleans() {
            let request = json!({"name": "body", "field_option": {"text": {key: bad}}});
            let (status, body) = send(&router, "POST", "/v1/schema/fields", request).await;
            let expected = json!(format!("{key} must be true or false"));
            if status != StatusCode::BAD_REQUEST || body["error"]["message"] != expected {
                wrong.push(format!("{key} = {bad}: {status} {body}"));
            }
            let after = schema(&router).await;
            if after != before {
                wrong.push(format!("{key} = {bad}: the schema changed to {after}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `PATCH /v1/schema/fields/:name` with a non-boolean field option key is
/// a 400, and the field is unchanged.
#[tokio::test]
async fn update_field_answers_400_for_a_non_boolean_field_option_key() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    let title = json!({"text": {"indexed": true, "stored": true}});
    create_index(&router, json!({ "title": title })).await;
    let before = schema(&router).await;

    let mut wrong = Vec::new();
    for key in TEXT_KEYS {
        for bad in non_booleans() {
            let mut option = title.clone();
            option["text"][key] = bad.clone();
            let update = json!({"field_option": option, "reindex": true});
            let (status, body) = send(&router, "PATCH", "/v1/schema/fields/title", update).await;
            let expected = json!(format!("{key} must be true or false"));
            if status != StatusCode::BAD_REQUEST || body["error"]["message"] != expected {
                wrong.push(format!("{key} = {bad}: {status} {body}"));
            }
            let after = schema(&router).await;
            if after != before {
                wrong.push(format!("{key} = {bad}: the schema changed to {after}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A search with a non-boolean `lexical_params.parallel`,
/// `highlight.require_field_match` or
/// `highlight.return_entire_field_if_no_highlight` is a 400 naming the key
/// on `POST /v1/search`, `/v1/search/stream` and `/v1/search/batch`,
/// instead of a search that reads the value as omitted.
#[tokio::test]
async fn search_answers_400_for_a_non_boolean_key() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    create_index(
        &router,
        json!({"body": {"text": {"indexed": true, "stored": true}}}),
    )
    .await;

    let cases = [
        ("lexical_params", "parallel", json!({})),
        (
            "highlight",
            "require_field_match",
            json!({"fields": ["body"]}),
        ),
        (
            "highlight",
            "return_entire_field_if_no_highlight",
            json!({"fields": ["body"]}),
        ),
    ];
    let mut wrong = Vec::new();
    for (object, key, params) in cases {
        for bad in non_booleans() {
            let mut params = params.clone();
            params[key] = bad.clone();
            let search = json!({"query": "body:rust", object: params});
            let expected = json!(format!("{object}: {key} must be true or false"));
            for (uri, request) in [
                ("/v1/search", search.clone()),
                ("/v1/search/stream", search.clone()),
                ("/v1/search/batch", json!({"queries": [search.clone()]})),
            ] {
                let (status, body) = send(&router, "POST", uri, request).await;
                if status != StatusCode::BAD_REQUEST || body["error"]["message"] != expected {
                    wrong.push(format!("{uri} {object}.{key} = {bad}: {status} {body}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `true`, `false` and `null` are still accepted: `null` is the same as
/// omitting `dry_run`, and an explicit `true` classifies the change without
/// applying it.
#[tokio::test]
async fn update_field_accepts_booleans_and_null() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    create_index(&router, json!({"vec": {"hnsw": {"dimension": 8, "m": 32}}})).await;
    let m = |schema: &Value| schema["schema"]["fields"]["vec"]["hnsw"]["m"].clone();

    let option = json!({"hnsw": {"dimension": 8, "m": 16}});
    let update = json!({"field_option": option, "reindex": true, "dry_run": true});
    let (status, body) = send(&router, "PATCH", "/v1/schema/fields/vec", update).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["classification"], json!("reindex"), "{body}");
    assert_eq!(m(&schema(&router).await), json!(32));

    let update = json!({"field_option": option, "reindex": true, "dry_run": null});
    let (status, body) = send(&router, "PATCH", "/v1/schema/fields/vec", update).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(m(&schema(&router).await), json!(16));
}
