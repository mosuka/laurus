//! The total hit count over the HTTP gateway (Issue #1472).
//!
//! A search body's `"track_total_hits"` takes an integer, `true` or `false`,
//! as in Elasticsearch, and the response gets `"total_hits": {"value",
//! "relation": "eq" | "gte"}` only when one was asked for. A value of
//! another type is a 400 naming the key, `0` is the engine's 400, and the
//! streamed search, which has no message for a total, rejects a request for
//! one.
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

/// The number of documents [`gateway_with_matches`] stores, each matching
/// `body:rust`.
const MATCHES: u64 = 5;

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

/// Send `body` as JSON to `uri` and return the status and the body: its
/// JSON, or its text when it is not JSON.
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

/// Send `body`, assert a 400 whose message contains `part`, and return the
/// message.
async fn rejected(router: &Router, uri: &str, body: Value, part: &str) -> String {
    let (status, response) = send(router, "POST", uri, body.clone()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{uri} {body}: {response}");
    let message = response["error"]["message"]
        .as_str()
        .unwrap_or("")
        .to_string();
    assert!(
        message.contains(part),
        "{uri} {body}: missing {part:?}: {message}"
    );
    message
}

/// A gateway over an index of [`MATCHES`] committed documents whose `body`
/// is "rust".
async fn gateway_with_matches(dir: &tempfile::TempDir) -> Router {
    let router = gateway(dir.path().join("idx")).await;
    ok(
        &router,
        "POST",
        "/v1/index",
        json!({"schema": {"fields": {"body": {"text": {"stored": true}}}}}),
    )
    .await;
    for i in 0..MATCHES {
        ok(
            &router,
            "PUT",
            &format!("/v1/documents/doc{i}"),
            json!({"fields": {"body": "rust"}}),
        )
        .await;
    }
    ok(&router, "POST", "/v1/commit", json!({})).await;
    router
}

/// A search for a page of 2 of the matches, with `track_total_hits` set to
/// `value` unless it is `None`.
fn search_body(value: Option<Value>) -> Value {
    let mut body = json!({"query": "body:rust", "limit": 2});
    if let Some(value) = value {
        body["track_total_hits"] = value;
    }
    body
}

/// Issue #1472: `"track_total_hits"` counts the matches beyond the page:
/// exactly below the threshold and with `true`, as a lower bound of the
/// threshold from it on. `false`, `null` and an absent key return the body
/// without `"total_hits"`, as before.
#[tokio::test]
async fn a_search_returns_the_total_it_asks_for() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway_with_matches(&dir).await;

    for (value, expected) in [
        (
            Some(json!(1000)),
            Some(json!({"value": MATCHES, "relation": "eq"})),
        ),
        (Some(json!(3)), Some(json!({"value": 3, "relation": "gte"}))),
        (
            Some(json!(true)),
            Some(json!({"value": MATCHES, "relation": "eq"})),
        ),
        (Some(json!(false)), None),
        (Some(Value::Null), None),
        (None, None),
    ] {
        let label = format!("{value:?}");
        let body = ok(&router, "POST", "/v1/search", search_body(value)).await;
        assert_eq!(
            body["results"].as_array().map(Vec::len),
            Some(2),
            "{label}: {body}"
        );
        assert_eq!(body.get("total_hits"), expected.as_ref(), "{label}: {body}");
        let keys: Vec<&str> = body
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let expected_keys: &[&str] = if expected.is_some() {
            &["results", "total_hits"]
        } else {
            &["results"]
        };
        assert_eq!(keys, expected_keys, "{label}: {body}");
    }
}

/// Issue #1472: a `"track_total_hits"` of another type is a 400 naming the
/// key, instead of a search without a total; `0` is the engine's 400.
#[tokio::test]
async fn a_search_rejects_a_track_total_hits_it_cannot_read() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway_with_matches(&dir).await;

    for value in [json!(1.5), json!(-1), json!("1000"), json!({"up_to": 3})] {
        rejected(
            &router,
            "/v1/search",
            search_body(Some(value)),
            "track_total_hits must be true, false or an integer between 0 and",
        )
        .await;
    }
    rejected(
        &router,
        "/v1/search",
        search_body(Some(json!(0))),
        "track_total_hits must count at least 1 match",
    )
    .await;
}

/// Issue #1472: the streamed search has no event for a total, so a request
/// for one is a 400 before any event; a request for none is streamed.
#[tokio::test]
async fn a_streamed_search_rejects_a_request_for_a_total() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway_with_matches(&dir).await;

    for value in [json!(10), json!(true)] {
        rejected(
            &router,
            "/v1/search/stream",
            search_body(Some(value)),
            "a streamed search returns no total hit count",
        )
        .await;
    }
    for value in [Some(json!(false)), None] {
        let (status, body) = send(&router, "POST", "/v1/search/stream", search_body(value)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
}

/// Issue #1472: each query of a batch takes its own `"track_total_hits"`,
/// and each entry of the response carries its own total.
#[tokio::test]
async fn each_query_of_a_batch_returns_its_own_total() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway_with_matches(&dir).await;

    let body = ok(
        &router,
        "POST",
        "/v1/search/batch",
        json!({"queries": [
            search_body(Some(json!(1000))),
            search_body(None),
            search_body(Some(json!(3))),
        ]}),
    )
    .await;
    let totals: Vec<Option<&Value>> = body["results"]
        .as_array()
        .expect("a results array")
        .iter()
        .map(|entry| entry.get("total_hits"))
        .collect();
    assert_eq!(
        totals,
        [
            Some(&json!({"value": MATCHES, "relation": "eq"})),
            None,
            Some(&json!({"value": 3, "relation": "gte"})),
        ],
        "{body}"
    );

    rejected(
        &router,
        "/v1/search/batch",
        json!({"queries": [search_body(None), search_body(Some(json!("1000")))]}),
        "queries[1]: track_total_hits must be",
    )
    .await;
}
