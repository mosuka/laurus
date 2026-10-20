//! The HTTP gateway answers 400 when the engine rejects a vector field's
//! options on CreateIndex, AddField and UpdateField (Issues #1479, #1491),
//! or when the gateway cannot parse one of its integer keys (Issue #1490),
//! and its schema responses keep `default_ef_search` (Issue #1490).
//!
//! The gateway forwards each call to the gRPC service and maps the status
//! it returns, so these requests go through both, over a gRPC connection
//! to a service on an ephemeral port.

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
    let service = IndexService {
        engine: Arc::new(RwLock::new(None)),
        data_dir,
        wal_policy: WalSyncPolicy::PerRecord,
        commit_policy: CommitPolicy::Manual,
    };
    tokio::spawn(
        Server::builder()
            .add_service(IndexServiceServer::new(service))
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
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// The JSON of an HNSW field with a product quantizer of
/// `subvector_count` sub-vectors.
fn hnsw_pq(subvector_count: u32) -> Value {
    json!({"hnsw": {
        "dimension": 8,
        "quantizer": {"method": "product_quantization", "subvector_count": subvector_count},
    }})
}

/// Field options the engine rejects, each with the reason it gives.
fn unusable_options() -> [(Value, &'static str); 3] {
    [
        (
            hnsw_pq(0),
            "HNSW product quantization subvector_count must be at least 1",
        ),
        (
            hnsw_pq(3),
            "HNSW product quantization subvector_count 3 must divide dimension 8 (got 8 % 3 = 2)",
        ),
        (
            json!({"hnsw": {"dimension": 8, "m": 1}}),
            "HNSW m must be at least 2, got 1",
        ),
    ]
}

fn assert_bad_request(status: StatusCode, body: &Value, field: &str, reason: &str) {
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reason}: {body}");
    assert_eq!(
        body["error"]["message"],
        format!("Invalid argument: field '{field}': {reason}")
    );
}

#[tokio::test]
async fn create_index_answers_400_and_writes_nothing() {
    for (option, reason) in unusable_options() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let router = gateway(data_dir.clone()).await;

        let schema = json!({"schema": {"fields": {"vec": option}}});
        let (status, body) = send(&router, "POST", "/v1/index", schema).await;
        assert_bad_request(status, &body, "vec", reason);
        assert!(!data_dir.exists(), "{reason}");
    }
}

#[tokio::test]
async fn add_and_update_field_answer_400_and_leave_the_schema() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    let schema = json!({"schema": {"fields": {"vec": hnsw_pq(2)}}});
    let (status, body) = send(&router, "POST", "/v1/index", schema).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, before) = send(&router, "GET", "/v1/schema", json!({})).await;

    for (option, reason) in unusable_options() {
        let add = json!({"name": "vec2", "field_option": option});
        let (status, body) = send(&router, "POST", "/v1/schema/fields", add).await;
        assert_bad_request(status, &body, "vec2", reason);

        let update = json!({"field_option": option, "reindex": true});
        let (status, body) = send(&router, "PATCH", "/v1/schema/fields/vec", update).await;
        assert_bad_request(status, &body, "vec", reason);

        let (_, after) = send(&router, "GET", "/v1/schema", json!({})).await;
        assert_eq!(after, before, "{reason}");
    }
}

/// Issue #1490: a `default_ef_search` that is not an integer that fits
/// `u32` is a 400 from the gateway itself, instead of a field without one,
/// and nothing is written.
#[tokio::test]
async fn create_index_answers_400_for_a_malformed_default_ef_search() {
    for bad in [json!("64"), json!(64.0), json!(-1), json!(4_294_967_360u64)] {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let router = gateway(data_dir.clone()).await;

        let option = json!({"hnsw": {"dimension": 8, "default_ef_search": bad}});
        let schema = json!({"schema": {"fields": {"vec": option}}});
        let (status, body) = send(&router, "POST", "/v1/index", schema).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {body}");
        assert_eq!(
            body["error"]["message"],
            "field \"vec\": default_ef_search must be an integer between 0 and 4294967295",
            "{bad}"
        );
        assert!(!data_dir.exists(), "{bad}");
    }
}

/// Issue #1490: `GET /v1/schema` shows an HNSW field's `default_ef_search`,
/// so a `PATCH /v1/schema/fields/:name` with the field copied from it, to
/// change another key, keeps the value instead of clearing it.
#[tokio::test]
async fn schema_responses_keep_default_ef_search() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    let option = json!({"hnsw": {"dimension": 8, "default_ef_search": 64}});
    let schema = json!({"schema": {"fields": {"vec": option}}});
    let (status, body) = send(&router, "POST", "/v1/index", schema).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, body) = send(&router, "GET", "/v1/schema", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let mut option = body["schema"]["fields"]["vec"].clone();
    assert_eq!(option["hnsw"]["default_ef_search"], json!(64), "{body}");

    option["hnsw"]["m"] = json!(24);
    let update = json!({"field_option": option, "reindex": true});
    let (status, body) = send(&router, "PATCH", "/v1/schema/fields/vec", update).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (_, body) = send(&router, "GET", "/v1/schema", json!({})).await;
    let hnsw = &body["schema"]["fields"]["vec"]["hnsw"];
    assert_eq!(hnsw["m"], json!(24), "{body}");
    assert_eq!(hnsw["default_ef_search"], json!(64), "{body}");

    // UpdateField replaces the whole option, so a copy without the key, as
    // a server before this fix rendered it, clears the value.
    option["hnsw"]
        .as_object_mut()
        .unwrap()
        .remove("default_ef_search");
    let update = json!({"field_option": option});
    let (status, body) = send(&router, "PATCH", "/v1/schema/fields/vec", update).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, body) = send(&router, "GET", "/v1/schema", json!({})).await;
    let hnsw = &body["schema"]["fields"]["vec"]["hnsw"];
    assert!(hnsw.get("default_ef_search").is_none(), "{body}");
}
