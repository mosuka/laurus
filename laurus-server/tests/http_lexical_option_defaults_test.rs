//! Lexical fields created or changed over HTTP with `indexed` and `stored`
//! left out or `null` (Issue #1709).
//!
//! The gateway used to read such a key as `false`, so `{"text": {}}`
//! created a field that was neither indexed nor stored. It now leaves the
//! key unset, and the server reads it as the core option's default
//! (`true`), as an omitted key in `schema.toml` does.
//!
//! The gateway forwards each call to the gRPC services, so these requests
//! go through both, over a gRPC connection on an ephemeral port.

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
use laurus_server::proto::laurus::v1::document_service_server::DocumentServiceServer;
use laurus_server::proto::laurus::v1::index_service_server::IndexServiceServer;
use laurus_server::proto::laurus::v1::search_service_server::SearchServiceServer;
use laurus_server::service::document::DocumentService;
use laurus_server::service::index::IndexService;
use laurus_server::service::search::SearchService;

/// The eight lexical field option types.
const KINDS: [&str; 8] = [
    "text",
    "integer",
    "float",
    "boolean",
    "date_time",
    "geo",
    "geo3d",
    "bytes",
];

/// The flag keys of a field option type: `bytes` has no `indexed`.
fn keys(kind: &str) -> &'static [&'static str] {
    if kind == "bytes" {
        &["stored"]
    } else {
        &["indexed", "stored"]
    }
}

/// Serve the index, document and search services over `data_dir`, with no
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
    let documents = DocumentService {
        engine: engine.clone(),
    };
    let search = SearchService {
        engine,
        max_result_window: DEFAULT_MAX_RESULT_WINDOW,
    };
    tokio::spawn(
        Server::builder()
            .add_service(IndexServiceServer::new(index))
            .add_service(DocumentServiceServer::new(documents))
            .add_service(SearchServiceServer::new(search))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_lazy();
    gateway::create_router(GatewayState::new(channel))
}

/// Send `body` as JSON to `uri` and return the status and the JSON body.
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

/// Send `body` and assert a 200.
async fn ok(router: &Router, method: &str, uri: &str, body: Value) -> Value {
    let (status, body) = send(router, method, uri, body).await;
    assert_eq!(status, StatusCode::OK, "{method} {uri}: {body}");
    body
}

/// The fields of the schema `GET /v1/schema` returns.
async fn schema_fields(router: &Router) -> Value {
    ok(router, "GET", "/v1/schema", json!({})).await["schema"]["fields"].clone()
}

/// Create an index with one field per lexical type, named after it, whose
/// flag keys are all `value` (left out when `None`), and return the
/// "kind.key: value" of each flag `GET /v1/schema` reports otherwise than
/// `want`.
async fn flags_not(value: Option<Value>, want: bool) -> Vec<String> {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    let fields: serde_json::Map<String, Value> = KINDS
        .iter()
        .map(|kind| {
            let option: serde_json::Map<String, Value> = match &value {
                Some(value) => keys(kind)
                    .iter()
                    .map(|key| (key.to_string(), value.clone()))
                    .collect(),
                None => serde_json::Map::new(),
            };
            (kind.to_string(), json!({ *kind: option }))
        })
        .collect();
    ok(
        &router,
        "POST",
        "/v1/index",
        json!({"schema": {"fields": fields}}),
    )
    .await;
    let schema = schema_fields(&router).await;
    let mut wrong = Vec::new();
    for kind in KINDS {
        for key in keys(kind) {
            let got = &schema[kind][kind][*key];
            if *got != json!(want) {
                wrong.push(format!("{kind}.{key}: {got}"));
            }
        }
    }
    wrong
}

/// Each lexical type created with `indexed` and `stored` left out is
/// indexed and stored.
#[tokio::test]
async fn create_index_without_indexed_or_stored_gives_true() {
    let wrong = flags_not(None, true).await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `null` is the same as leaving the key out.
#[tokio::test]
async fn create_index_with_null_indexed_and_stored_gives_true() {
    let wrong = flags_not(Some(Value::Null), true).await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// An explicit `false` is kept: the default applies only to a key left out.
#[tokio::test]
async fn create_index_keeps_an_explicit_false() {
    let wrong = flags_not(Some(json!(false)), false).await;
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A text field created as `{"text": {}}` finds the document it holds and
/// returns its value.
#[tokio::test]
async fn text_field_without_indexed_or_stored_is_searchable_and_returned() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    ok(
        &router,
        "POST",
        "/v1/index",
        json!({"schema": {"fields": {"body": {"text": {}}}}}),
    )
    .await;
    ok(
        &router,
        "PUT",
        "/v1/documents/doc1",
        json!({"fields": {"body": "rust"}}),
    )
    .await;
    ok(&router, "POST", "/v1/commit", json!({})).await;

    let search = ok(
        &router,
        "POST",
        "/v1/search",
        json!({"query": "body:rust", "limit": 10}),
    )
    .await;
    let ids: Vec<&Value> = search["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| &hit["id"])
        .collect();
    assert_eq!(ids, [&json!("doc1")], "{search}");

    let documents = ok(&router, "GET", "/v1/documents/doc1", json!({})).await;
    assert_eq!(
        documents["documents"][0]["fields"]["body"],
        json!("rust"),
        "{documents}"
    );
}

/// `POST /v1/schema/fields` and `PATCH /v1/schema/fields/{name}`, which
/// replaces the whole option, give `true` for `indexed` and `stored` left
/// out.
#[tokio::test]
async fn added_and_updated_fields_without_indexed_or_stored_give_true() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    ok(
        &router,
        "POST",
        "/v1/index",
        json!({"schema": {"fields": {"title": {"text": {"indexed": true, "stored": true}}}}}),
    )
    .await;
    ok(
        &router,
        "POST",
        "/v1/schema/fields",
        json!({"name": "body", "field_option": {"text": {}}}),
    )
    .await;
    ok(
        &router,
        "PATCH",
        "/v1/schema/fields/title",
        json!({"field_option": {"text": {}}, "reindex": true}),
    )
    .await;

    let schema = schema_fields(&router).await;
    let mut wrong = Vec::new();
    for name in ["body", "title"] {
        for key in ["indexed", "stored"] {
            let got = &schema[name]["text"][key];
            if *got != json!(true) {
                wrong.push(format!("{name}.{key}: {got}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
