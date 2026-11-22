//! The HTTP gateway passes embedders through as sent, and `GET /v1/schema`
//! returns them in the shape and with the types a request uses (Issue
//! #1702).
//!
//! The gateway used to turn every parameter into a string for the gRPC
//! services, which read each type's keys by hand: `"query_maxlen": "32"` was
//! stored as the number 32, `"revision": 123` and `"model": true` as the
//! strings `"123"` and `"true"`, and a key the type does not take, such as a
//! misspelled `"doc_max_len"` or `"model"` on a `precomputed` embedder, was
//! dropped, all with a 200. The server now reads an embedder with the core
//! type's serde, by the rules `schema.toml` is read by.
//!
//! The gateway forwards each call to the gRPC services, so these requests
//! go through both, over a gRPC connection on an ephemeral port. Only
//! embedders that a field uses are built from a model, so no model is
//! downloaded.

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

/// A schema with one text field and `embedders`, which no field uses.
fn schema(embedders: Value) -> Value {
    json!({
        "schema": {
            "fields": {"body": {"text": {"indexed": true, "stored": true}}},
            "embedders": embedders
        }
    })
}

/// One embedder of every type, `candle_colbert` with all of its parameters
/// and with none, so that the response is the request.
fn embedders_sent() -> Value {
    json!({
        "pre": {"type": "precomputed"},
        "bert": {"type": "candle_bert", "model": "sentence-transformers/all-MiniLM-L6-v2"},
        "clip": {"type": "candle_clip", "model": "openai/clip-vit-base-patch32"},
        "openai": {"type": "openai", "model": "text-embedding-3-small"},
        "colbert": {
            "type": "candle_colbert",
            "model": "colbert-ir/colbertv2.0",
            "revision": "c1e84128",
            "query_maxlen": 32,
            "doc_maxlen": 180
        },
        "colbert_small": {"type": "candle_colbert", "model": "answerdotai/answerai-colbert-small-v1"}
    })
}

#[tokio::test]
async fn embedders_come_back_as_sent_and_post_back_to_the_same_schema() {
    let first = tempfile::tempdir().unwrap();
    let first_dir = first.path().join("data");
    let router = gateway(first_dir.clone()).await;
    let (status, body) = send(&router, "POST", "/v1/index", schema(embedders_sent())).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, response) = send(&router, "GET", "/v1/schema", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["schema"]["embedders"], embedders_sent());
    let query_maxlen = &response["schema"]["embedders"]["colbert"]["query_maxlen"];
    assert!(query_maxlen.is_u64(), "{query_maxlen}");

    // The response posted to another server gives the same schema.toml.
    let second = tempfile::tempdir().unwrap();
    let second_dir = second.path().join("data");
    let router = gateway(second_dir.clone()).await;
    let (status, body) = send(&router, "POST", "/v1/index", response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let first_toml = std::fs::read_to_string(first_dir.join("schema.toml")).unwrap();
    let second_toml = std::fs::read_to_string(second_dir.join("schema.toml")).unwrap();
    assert_eq!(first_toml, second_toml);
}

/// A whole number written as a float fills a length, and a key set to
/// `null` is the same as omitting it, also one the type does not take: it
/// carries no value to lose.
#[tokio::test]
async fn whole_numbers_fill_lengths_and_null_is_the_same_as_omitting_a_key() {
    let dir = tempfile::tempdir().unwrap();
    let router = gateway(dir.path().join("data")).await;
    let embedders = json!({
        "colbert": {
            "type": "candle_colbert",
            "model": "m",
            "revision": null,
            "query_maxlen": 32.0
        },
        "pre": {"type": "precomputed", "model": null}
    });
    let (status, body) = send(&router, "POST", "/v1/index", schema(embedders)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, response) = send(&router, "GET", "/v1/schema", json!({})).await;
    assert_eq!(
        response["schema"]["embedders"],
        json!({
            "colbert": {"type": "candle_colbert", "model": "m", "query_maxlen": 32},
            "pre": {"type": "precomputed"}
        })
    );
}

/// An embedder the core type cannot read is a 400 naming the embedder and
/// the problem, and creates nothing. The first five are the embedders of
/// the reproduction in Issue #1702, which used to be stored with another
/// type or without the key.
#[tokio::test]
async fn an_embedder_of_the_wrong_shape_is_a_400_and_creates_nothing() {
    // Nested deeper than prost's recursion limit allows, which the
    // gateway's depth limit turns into a 400 rather than a 500.
    let mut deep = json!("x");
    for _ in 0..40 {
        deep = json!({"k": deep});
    }
    let cases = [
        (
            json!({"type": "candle_colbert", "model": "m", "query_maxlen": "32"}),
            "Embedder 'e': invalid type: string \"32\", expected usize",
        ),
        (
            json!({"type": "candle_colbert", "model": "m", "revision": 123}),
            "Embedder 'e': invalid type: integer `123`, expected a string",
        ),
        (
            json!({"type": "candle_bert", "model": true}),
            "Embedder 'e': invalid type: boolean `true`, expected a string",
        ),
        (
            json!({"type": "candle_colbert", "model": "m", "doc_max_len": 300}),
            "Embedder 'e': unknown field `doc_max_len`, expected one of `model`, `revision`, \
             `query_maxlen`, `doc_maxlen`",
        ),
        (
            json!({"type": "precomputed", "model": "x"}),
            "Embedder 'e': unknown field `model`, there are no fields",
        ),
        (
            json!({"type": "candle_colbert", "model": "m", "query_maxlen": 32.5}),
            "Embedder 'e': invalid type: floating point `32.5`, expected usize",
        ),
        (
            json!({"type": "candle_bert"}),
            "Embedder 'e': missing field `model`",
        ),
        (json!({"model": "m"}), "Embedder 'e': missing field `type`"),
        (
            json!({"type": "candle_bert", "model": ["m"]}),
            "Embedder 'e': invalid type: sequence, expected a string",
        ),
        (
            json!("precomputed"),
            "embedder \"e\": embedder definition must be an object",
        ),
        (
            json!({"type": "precomputed", "x": deep}),
            "embedder \"e\": embedder definition must not nest more than 8 levels deep",
        ),
    ];
    // Every request that was not rejected as expected, so a failure lists
    // all of them rather than the first.
    let mut wrong = Vec::new();
    for (e, expected) in cases {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let router = gateway(data_dir.clone()).await;
        let (status, body) = send(&router, "POST", "/v1/index", schema(json!({"e": e}))).await;
        if status != StatusCode::BAD_REQUEST || body["error"]["message"] != json!(expected) {
            wrong.push(format!("{e}: {status} {body}"));
        }
        if data_dir.exists() {
            wrong.push(format!("{e}: an index was created"));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
