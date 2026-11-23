//! `GET /v1/schema` with no index answers 412, as the other routes that need
//! an index do (Issue #1724).
//!
//! It used to read `schema.toml` whatever the server had open, and answer
//! 500 with the OS error named twice when there was none:
//! `Failed to read schema file: <dir>/schema.toml: No such file or directory
//! (os error 2): No such file or directory (os error 2)`.
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
use laurus_server::gateway::{self, GatewayState};
use laurus_server::proto::laurus::v1::index_service_server::IndexServiceServer;
use laurus_server::service::index::IndexService;

/// Serve an `IndexService` over `data_dir`, with no index yet, on an
/// ephemeral port, and return the gateway router in front of it.
async fn gateway(data_dir: PathBuf) -> Router {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let index = IndexService {
        engine: Arc::new(RwLock::new(None)),
        data_dir,
        wal_policy: WalSyncPolicy::PerRecord,
        commit_policy: CommitPolicy::Manual,
    };
    tokio::spawn(
        Server::builder()
            .add_service(IndexServiceServer::new(index))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_lazy();
    gateway::create_router(GatewayState::new(channel))
}

/// Send `body` (or nothing, for `Value::Null`) to `uri` and return the
/// status and the JSON body.
async fn send(router: &Router, method: &str, uri: &str, body: Value) -> (StatusCode, Value) {
    let body = if body.is_null() {
        Body::empty()
    } else {
        Body::from(body.to_string())
    };
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(body)
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

#[tokio::test]
async fn get_schema_with_no_index_is_412_and_with_one_the_schema() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;

    let (status, body) = send(&router, "GET", "/v1/schema", Value::Null).await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED, "{body}");
    assert_eq!(
        body,
        json!({"error": {"code": 412, "message": "No index is open"}})
    );
    // The answer of the other routes that need an index.
    let (status, body) = send(&router, "GET", "/v1/index", Value::Null).await;
    assert_eq!(status, StatusCode::PRECONDITION_FAILED, "{body}");

    let schema = json!({"schema": {"fields": {"body": {"text": {}}}}});
    let (status, body) = send(&router, "POST", "/v1/index", schema).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = send(&router, "GET", "/v1/schema", Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body["schema"]["fields"]["body"]["text"].is_object(),
        "{body}"
    );
}
